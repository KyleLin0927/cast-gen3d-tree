#!/usr/bin/env python3
"""
Transformer Prior Training Script
---------------------------------

You already have offline latent datasets:

    latent_root/
        train/*.npy
        val/*.npy
        test/*.npy

Each npy file is shaped:
    [512, latent_dim]

This script trains a GPT-style Transformer to model p(z).

Training task:
    Input:  z[0], z[1], ..., z[510]
    Target: z[1], z[2], ..., z[511]

At inference:
    - Start from z[0] = zero vector
    - Autoregressively sample 512 latent tokens
    - Use your VAE.decoder to generate voxel

Author: ChatGPT
"""

import os
import math
import argparse
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime
import csv
import math
import importlib.util
from pathlib import Path
import sys

# Progress UI
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------------------------------------
# Utility
# ------------------------------------------------

def seed_all(s):
    """
    Seed all relevant RNGs for reproducibility.
    
    Parameters
    ----------
    s : int
        Random seed to set. Default in this script: 42 (via --seed).
    """
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# ------------------------------------------------
# Dataset
# ------------------------------------------------

class LatentDataset(Dataset):
    """
    Loads latent tokens from .npy files.
    Expected shape: [512, latent_dim]
    
    Parameters
    ----------
    files : List[str]
        List of file paths to .npy latent files. Each file should have
        shape [512, latent_dim] (e.g., latent_dim=256 by default in this script).
    """
    def __init__(self, files, preload: bool = True, console=None):
        self.files = files
        self.preload = bool(preload)
        self._cache = None
        if self.preload:
            self._cache = []
            if console:
                # Avoid storing console (not picklable); use it only during init-time messages
                console.print(f"[cyan]Preloading {len(files)} latent files into RAM...[/cyan]")
            for i, fp in enumerate(files):
                arr = np.load(fp)  # expect (512, D)
                if arr.ndim != 2 or arr.shape[0] != 512:
                    raise ValueError(f"Expected shape [512, D], got {arr.shape} from {fp}")
                t = torch.from_numpy(arr).float()
                self._cache.append(t)
                if console and (i + 1) % 500 == 0:
                    console.print(f"  Loaded {i + 1}/{len(files)} files...")
            if console:
                console.print(f"[green]✓[/green] Preloaded {len(files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        z = np.load(self.files[idx])       # shape (512, latent_dim)
        z = torch.from_numpy(z).float()    # convert to tensor
        return z                           # return full sequence

    def release(self):
        """Release in-memory cache to free RAM."""
        self._cache = None

# ------------------------------------------------
# Transformer Model (GPT-like)
# ------------------------------------------------

class TransformerPrior(nn.Module):
    def __init__(
        self,
        token_dim=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_seq=512
    ):
        """
        GPT-style Transformer prior over latent tokens, predicting Gaussian parameters.
        
        Parameters
        ----------
        token_dim : int, default=256
            Dimensionality of each latent token vector (projected output size).
        hidden_dim : int, default=512
            Model hidden size (Transformer d_model).
        num_layers : int, default=8
            Number of Transformer encoder layers.
        num_heads : int, default=8
            Number of attention heads per layer.
        dropout : float, default=0.1
            Dropout probability used in Transformer layers.
        max_seq : int, default=512
            Maximum supported sequence length (number of tokens).
        """
        super().__init__()

        self.token_dim = token_dim
        self.max_seq = max_seq

        # Learnable BOS (Beginning of Sequence) token for consistent train/inference
        self.bos = nn.Parameter(torch.randn(1, 1, token_dim))

        # Project latent token dim -> model hidden dim
        self.input_proj = nn.Linear(token_dim, hidden_dim)

        # Positional Embedding
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq, hidden_dim))

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection to Gaussian params (mu, logvar) per token dim
        self.output_proj = nn.Linear(hidden_dim, token_dim * 2)

    def forward(self, x):
        """
        Forward pass through the prior to predict next-token Gaussian parameters.
        Uses causal mask to ensure autoregressive property (no future information).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, S, token_dim], where S <= max_seq.
        
        Returns
        -------
        (mu, logvar) : Tuple[torch.Tensor, torch.Tensor]
            mu/logvar tensors with shape [B, S, token_dim], representing
            parameters of predicted next-token Gaussian per time step.
        """
        B, S, D = x.shape
        assert S <= self.max_seq, "Input sequence longer than max_seq"

        # Input projection + positional embedding
        h = self.input_proj(x) + self.pos_emb[:, :S, :]

        # =============== ADD CAUSAL MASK HERE ===============
        # mask shape must be [S, S] even when batch_first=True
        # mask[i,j] = -inf 代表 token_i 不可以看到 token_j
        mask = torch.triu(
            torch.full((S, S), float('-inf'), device=x.device),
            diagonal=1
        )
        # =====================================================

        # Transformer encoder (now behaves like GPT decoder)
        h = self.transformer(h, mask=mask)  # <--- apply causal mask

        out = self.output_proj(h)
        mu, logvar = out[..., :D], out[..., D:]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar   # next-token Gaussian params

# ------------------------------------------------
# Training step
# ------------------------------------------------

def train_one_epoch(model, loader, opt, device):
    """
    Autoregressive Transformer prior training:
      - model predicts (mu, logvar)
      - sample z_pred = mu + std * eps
      - compute MSE on raw latent (no normalization, no KL to avoid collapse)
      - Pure autoregressive learning like GPT/LLaMA
      - Uses learnable BOS token for consistent train/inference
    """
    model.train()
    total_loss = 0.0
    count = 0

    mse = nn.MSELoss()

    for z in loader:
        z = z.to(device)    # [B,512,latent_dim]
        B = z.size(0)

        # Prepend BOS token: [BOS, z[0], z[1], ..., z[510]]
        bos_expanded = model.bos.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([bos_expanded, z[:, :-1, :]], dim=1)  # [B,512,D]
        y_target = z[:, 1:, :]  # target: [z[1], z[2], ..., z[511]]  # [B,511,D]

        mu, logvar = model(x)     # [B,512,D] each
        # Use predictions from position 1 onwards (skip BOS position, which predicts z[0])
        mu_pred = mu[:, 1:, :]    # [B,511,D]
        logvar_pred = logvar[:, 1:, :]  # [B,511,D]
        
        std_pred = torch.exp(0.5 * logvar_pred)
        eps = torch.randn_like(std_pred)
        z_pred = mu_pred + std_pred * eps  # [B,511,D]

        # Pure MSE loss on raw latent (no normalization, no KL)
        loss = mse(z_pred, y_target)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * z.size(0)
        count += z.size(0)

    return total_loss / count

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    """
    Deterministic eval using mu (no sampling).
    Uses raw latent without normalization, pure MSE loss.
    Uses learnable BOS token for consistent train/inference.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    mse = nn.MSELoss()

    for z in loader:
        z = z.to(device)  # [B,512,latent_dim]
        B = z.size(0)

        # Prepend BOS token: [BOS, z[0], z[1], ..., z[510]]
        bos_expanded = model.bos.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([bos_expanded, z[:, :-1, :]], dim=1)  # [B,512,D]
        y_target = z[:, 1:, :]  # target: [z[1], z[2], ..., z[511]]  # [B,511,D]

        mu, logvar = model(x)  # [B,512,D] each
        # Use predictions from position 1 onwards (skip BOS position)
        mu_pred = mu[:, 1:, :]  # [B,511,D]

        # Pure MSE loss on raw latent (no normalization, no KL)
        loss = mse(mu_pred, y_target)

        total_loss += loss.item() * z.size(0)
        count += z.size(0)

    return total_loss / count

# ------------------------------------------------
# Sampling
# ------------------------------------------------

@torch.no_grad()
def sample_latent(model, token_dim=256, seq_len=512, device="cpu", mode="mean"):
    """
    Sample one latent sequence:
        Start with BOS token (learnable embedding)
        Predict next tokens autoregressively.
    
    Parameters
    ----------
    model : nn.Module
        Trained Transformer prior model.
    token_dim : int, default=256
        Dimensionality of latent tokens to sample.
    seq_len : int, default=512
        Number of tokens to sample (sequence length).
    device : str, default="cpu"
        Device to run sampling on.
    mode : str, default="mean"
        "mean" uses mu as next token; "sample" samples from N(mu, std).
    
    Returns
    -------
    numpy.ndarray
        Array of shape [seq_len, token_dim] containing the sampled latent sequence.
    """
    model.eval()

    # Start with BOS token, then sample seq_len tokens
    bos = model.bos.to(device)  # [1,1,D]
    z = torch.zeros(1, seq_len, token_dim, device=device)
    
    # Current sequence: [BOS]
    current_seq = bos  # [1,1,D]

    for i in range(seq_len):
        mu, logvar = model(current_seq)  # [1, S, D] where S is current sequence length
        mu_t = mu[:, -1, :]  # [1, D] - prediction for next token (z[i])
        
        if mode == "sample":
            std_t = torch.exp(0.5 * logvar[:, -1, :])
            eps = torch.randn_like(std_t)
            next_token = mu_t + std_t * eps
        else:
            next_token = mu_t
        
        z[:, i, :] = next_token  # Store sampled token
        
        # Append to sequence for next prediction: [BOS, z[0], z[1], ..., z[i]]
        current_seq = torch.cat([current_seq, next_token.unsqueeze(1)], dim=1)  # [1, i+2, D]

    return z[0].cpu().numpy()

def compute_decoder_occupancy(labels, air_class=0):
    """
    Compute decoder occupancy metrics from decoded voxel labels.
    
    Parameters
    ----------
    labels : numpy.ndarray
        Array of shape [32,32,32] containing class labels after argmax.
    air_class : int, default=0
        Class ID representing "air" voxels (typically 0).
    
    Returns
    -------
    non_air_ratio : float
        Ratio of non-air voxels to total voxels.
    bbox_volume_ratio : float
        Ratio of bounding box volume (containing all non-air voxels) to total volume.
    """
    mask = (labels != air_class)
    total_voxels = mask.size
    non_air = mask.sum()
    non_air_ratio = float(non_air) / float(total_voxels + 1e-8)

    if non_air == 0:
        return non_air_ratio, 0.0

    # bounding box
    idx = np.where(mask)
    zmin, zmax = idx[0].min(), idx[0].max()
    ymin, ymax = idx[1].min(), idx[1].max()
    xmin, xmax = idx[2].min(), idx[2].max()
    bbox_vol = (zmax - zmin + 1) * (ymax - ymin + 1) * (xmax - xmin + 1)
    bbox_volume_ratio = float(bbox_vol) / float(total_voxels + 1e-8)

    return non_air_ratio, bbox_volume_ratio

# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        "Transformer Prior Training",
        description="Train a GPT-style Transformer to model latent token sequences p(z).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing latent npy files under train/ val/ test/."
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent tokens in the dataset and model outputs. 若未提供，將自動從 train/ 的第一個 .npy 檔推斷。"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="訓練總回合數（必填）。"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the AdamW optimizer."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader 的背景工作執行緒數。"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="強制使用 CPU。若未指定，預設使用 CUDA（若可用），否則 MPS（若可用），再者 CPU。"
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="停用自動混合精度（AMP）。僅在 CUDA 可用時預設啟用。"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="將整個資料集預載入記憶體以提升 I/O 效率（預設關閉）。"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Transformer model hidden size (d_model)."
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=8,
        help="Number of Transformer encoder layers."
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads per Transformer layer."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability used in Transformer layers."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="輸出目錄（必填），用於儲存最佳權重與範例採樣結果。"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="本次實驗名稱（必填），所有輸出將寫入 out_dir/exp_name/ 之下。"
    )
    parser.add_argument(
        "--analytics_every",
        type=int,
        default=20,
        help="每隔 N 個 epoch 執行分析（variance、self-similarity、distribution drift）。"
    )
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default=None,
        help="（選用）已訓練 VAE 的 checkpoint 路徑；若提供，會在分析回合解碼 latent 並輸出投影圖。"
    )
    parser.add_argument(
        "--vae_model_def",
        type=str,
        default=None,
        help="（選用）VAE 模型定義檔案（含 UNet3DVAE）；預設使用 train_3D_UNetVAE_8latent.py。"
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.1,
        help="KL 正則項權重（對齊 VAE-style prior：KL(N(mu,std)||N(z_mean,z_std))）。"
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="mean",
        choices=["mean", "sample"],
        help="採樣時使用 mu（mean）或從 N(mu,std) 取樣（sample）。"
    )

    args = parser.parse_args()

    seed_all(args.seed)

    # 裝置選擇：CUDA > MPS > CPU（可用時），可用 --cpu 強制 CPU
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console = Console()
    console.print(f"[bold]Device:[/bold] {device}")

    # AMP（僅 CUDA 時預設啟用，可用 --no_amp 停用）
    use_amp = (device.type == "cuda") and (not args.no_amp)
    if device.type == "cuda":
        console.print(f"[bold]AMP:[/bold] {'ENABLED' if use_amp else 'DISABLED'}")
    else:
        console.print("[bold]AMP:[/bold] N/A")

    train_files = sorted(glob(os.path.join(args.data_root, "train", "*.npy")))
    val_files = sorted(glob(os.path.join(args.data_root, "val", "*.npy")))
    test_files = sorted(glob(os.path.join(args.data_root, "test", "*.npy")))

    print(f"Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # ------------------------------------------------
    # 自動推斷 latent_dim（若使用者未提供）
    # ------------------------------------------------
    if args.latent_dim is None:
        if len(train_files) == 0:
            raise ValueError("未提供 --latent_dim，且 train/ 資料夾為空，無法推斷 latent_dim。")
        # 從 train/ 的第一個檔案推斷維度；要求形狀為 [512, D]
        probe = np.load(train_files[0], mmap_mode="r")
        if probe.ndim != 2:
            raise ValueError(f"檔案形狀不正確（期望 2 維）：{train_files[0]} 取得 {probe.shape}")
        if probe.shape[0] != 512:
            raise ValueError(f"序列長度不為 512：{train_files[0]} 取得 {probe.shape}")
        args.latent_dim = int(probe.shape[1])
        print(f"[Info] 自動推斷 latent_dim = {args.latent_dim} 來自 {os.path.basename(train_files[0])}")
        # 基本一致性檢查：各 split 的第一個檔案（若存在）需有相同維度
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if len(files) == 0:
                continue
            probe2 = np.load(files[0], mmap_mode="r")
            if probe2.ndim != 2 or probe2.shape[0] != 512 or probe2.shape[1] != args.latent_dim:
                raise ValueError(
                    f"{split_name}/ 形狀不一致或不為 [512, {args.latent_dim}]：" +
                    f"{files[0]} 取得 {probe2.shape}"
                )

    # 預先載入全部訓練/驗證/測試資料到記憶體以提高效能
    train_ds = LatentDataset(train_files, preload=args.preload, console=console)
    val_ds = LatentDataset(val_files, preload=args.preload, console=console)
    test_ds = LatentDataset(test_files, preload=args.preload, console=console)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin_memory
    )

    # Build model
    model = TransformerPrior(
        token_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_seq=512,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    # 建立 out_dir/exp_name 實驗資料夾
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    best_val = 1e10
    best_path = os.path.join(exp_dir, "best_prior.pt")
    analytics_dir = os.path.join(exp_dir, "analytics")
    os.makedirs(analytics_dir, exist_ok=True)

    # 可選：載入 VAE decoder 以產生投影圖
    def _load_vae_decoder(ckpt_path: str, model_def_path: str | None, latent_dim: int, device):
        if ckpt_path is None:
            return None
        # 解析模型定義檔
        if model_def_path is None:
            # 預設與此腳本同層的 8latent 版本
            default_path = Path(__file__).parent / "train_3D_UNetVAE_8latent.py"
            model_def_path = str(default_path)
        spec = importlib.util.spec_from_file_location("vae_def_mod", model_def_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"無法載入 VAE 模型定義：{model_def_path}")
        vae_mod = importlib.util.module_from_spec(spec)
        sys.modules["vae_def_mod"] = vae_mod
        spec.loader.exec_module(vae_mod)
        if not hasattr(vae_mod, "UNet3DVAE"):
            raise RuntimeError("VAE 定義檔未提供 UNet3DVAE 類別")
        UNet3DVAE = getattr(vae_mod, "UNet3DVAE")
        vae = UNet3DVAE(in_ch=3, out_ch=3, base=64, latent_dim=latent_dim, skip_levels=0).to(device)
        # PyTorch 2.6 defaults weights_only=True; try that first, then fallback (unsafe) to False if needed
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        vae.load_state_dict(state, strict=False)
        vae.eval()
        return vae.decoder

    vae_decoder = None
    if args.vae_ckpt:
        try:
            vae_decoder = _load_vae_decoder(args.vae_ckpt, args.vae_model_def, args.latent_dim, device)
            console.print(f"[green]✓[/green] Loaded VAE decoder from {args.vae_ckpt}")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] 無法載入 VAE decoder：{e}")
            vae_decoder = None

    # 記錄
    training_history = []  # list of dicts
    start_time_dt = datetime.now()
    start_time = time.time()
    # Compute global train latent mean/std for normalization
    with torch.no_grad():
        total = 0
        sum_vec = torch.zeros(args.latent_dim, dtype=torch.float64)
        sumsq_vec = torch.zeros(args.latent_dim, dtype=torch.float64)
        for z in train_loader:
            z_t = z if isinstance(z, torch.Tensor) else torch.from_numpy(z)
            z_flat = z_t.reshape(-1, args.latent_dim).to(dtype=torch.float64)  # [N,D]
            sum_vec += z_flat.sum(dim=0)
            sumsq_vec += (z_flat * z_flat).sum(dim=0)
            total += z_flat.shape[0]
        total = max(int(total), 1)
        mean = sum_vec / total
        var = sumsq_vec / total - mean * mean
        var = torch.clamp(var, min=1e-8)
        train_mean_t = mean.to(dtype=torch.float32)
        train_std_t = torch.sqrt(var).to(dtype=torch.float32)
    # Note: train_one_epoch no longer uses latent_mean/std or kl_beta (pure MSE loss)
    # These are still computed for analytics/diagnostics only

    # Prepare metadata paths
    history_csv = os.path.join(exp_dir, f"training_history_{args.exp_name}.csv")
    metadata_csv = os.path.join(exp_dir, f"experiment_metadata_{args.exp_name}.csv")

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Create initial metadata (all fields that can be determined before training)
    initial_metadata = {
        "exp_name": args.exp_name,
        "data_root": args.data_root,
        "out_dir": args.out_dir,
        "exp_dir": exp_dir,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "workers": args.workers,
        "device": str(device),
        "amp_enabled": "TRUE" if use_amp else "FALSE",
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "loss_function": "MSE(z_pred, y) - pure autoregressive loss (no normalization, no KL)",
        "loss_formula": "z_pred = mu + std*eps; MSE computed on raw latent; std=exp(0.5*logvar); No KL loss to avoid collapse to global mean",
        "kl_beta": "N/A (KL loss removed)",
        "training_started": start_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "best_checkpoint": best_path,
        "sample_latent_file": os.path.join(exp_dir, "sample_latent.npy"),
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_test_files": len(test_files),
        "train_dataset_size": len(train_ds),
        "val_dataset_size": len(val_ds),
        "test_dataset_size": len(test_ds),
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "model_non_trainable_params": non_trainable_params,
        "preload": "TRUE" if args.preload else "FALSE",
        "sample_mode": args.sample_mode,
        "analytics_every": args.analytics_every,
        "vae_ckpt": args.vae_ckpt if args.vae_ckpt else "None",
    }

    # Save initial metadata (key-value format)
    with open(metadata_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in initial_metadata.items():
            writer.writerow([k, v])
    console.print(
        f"[green]✓[/green] Created initial metadata file: [cyan]{metadata_csv}[/cyan]"
    )

    # Create training_history.csv with header
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "epoch_time_secs",
                "cumulative_time_secs",
                "kl_train_gen",
                "mmd_rbf",
                "is_best",
            ],
        )
        writer.writeheader()
    console.print(
        f"[green]✓[/green] Created training history file: [cyan]{history_csv}[/cyan]"
    )

    # Progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[cyan]Training", total=args.epochs)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # Train with optional AMP + inner progress
            model.train()
            running = 0.0
            train_task = progress.add_task(
                f"[green]Epoch {epoch}/{args.epochs} - Train", total=len(train_loader)
            )
            mse = nn.MSELoss()
            for z in train_loader:
                z = z.to(device, non_blocking=True)
                B = z.size(0)
                
                # Prepend BOS token: [BOS, z[0], z[1], ..., z[510]]
                bos_expanded = model.bos.expand(B, -1, -1)  # [B,1,D]
                x = torch.cat([bos_expanded, z[:, :-1, :]], dim=1)  # [B,512,D]
                y_target = z[:, 1:, :]  # target: [z[1], z[2], ..., z[511]]  # [B,511,D]

                opt.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        mu, logvar = model(x)  # [B,512,D] each
                        # Use predictions from position 1 onwards (skip BOS position)
                        mu_pred = mu[:, 1:, :]  # [B,511,D]
                        logvar_pred = logvar[:, 1:, :]  # [B,511,D]
                        std_pred = torch.exp(0.5 * logvar_pred)
                        eps = torch.randn_like(std_pred)
                        z_pred = mu_pred + std_pred * eps
                        # Pure MSE loss on raw latent (no normalization, no KL)
                        loss = mse(z_pred, y_target)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    mu, logvar = model(x)  # [B,512,D] each
                    # Use predictions from position 1 onwards (skip BOS position)
                    mu_pred = mu[:, 1:, :]  # [B,511,D]
                    logvar_pred = logvar[:, 1:, :]  # [B,511,D]
                    std_pred = torch.exp(0.5 * logvar_pred)
                    eps = torch.randn_like(std_pred)
                    z_pred = mu_pred + std_pred * eps
                    # Pure MSE loss on raw latent (no normalization, no KL)
                    loss = mse(z_pred, y_target)
                    loss.backward()
                    opt.step()

                running += loss.item() * z.size(0)
                progress.update(train_task, advance=1)
            train_loss = running / len(train_loader.dataset)
            progress.remove_task(train_task)

            # Val
            val_task = progress.add_task(
                f"[yellow]Epoch {epoch}/{args.epochs} - Val", total=len(val_loader)
            )
            val_loss = eval_one_epoch(model, val_loader, device)
            progress.update(val_task, completed=len(val_loader))
            progress.remove_task(val_task)

            dt = time.time() - t0

            # Save best
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                    },
                    best_path,
                )

            # Defer console print and history write until after analytics (so KL/MMD 可顯示在 val 旁邊)
            current_kl = None
            current_mmd = None

            progress.update(overall, advance=1)

            # -------------------------
            # Analytics every N epochs
            # -------------------------
            if (epoch % args.analytics_every == 0) or (epoch == args.epochs):
                ep_dir = os.path.join(analytics_dir, f"epoch_{epoch:04d}")
                os.makedirs(ep_dir, exist_ok=True)

                # (1) Per-token output variance over validation predictions
                with torch.no_grad():
                    sum_pos = torch.zeros(511, dtype=torch.float64)
                    sumsq_pos = torch.zeros(511, dtype=torch.float64)
                    count_pos = torch.zeros(511, dtype=torch.float64)
                    mse_local = nn.MSELoss()
                    for z in val_loader:
                        z = z.to(device, non_blocking=True)
                        x = z[:, :-1, :]
                        mu, _ = model(x)  # [B,511,D]
                        pred_cpu = mu.detach().to("cpu").to(torch.float64)
                        # aggregate over batch and dim for each position
                        sum_pos += pred_cpu.sum(dim=(0, 2))
                        sumsq_pos += (pred_cpu.pow(2)).sum(dim=(0, 2))
                        count_pos += torch.tensor(
                            [pred_cpu.shape[0] * pred_cpu.shape[2]] * pred_cpu.shape[1],
                            dtype=torch.float64,
                        )
                    mean_pos = sum_pos / count_pos.clamp_min(1.0)
                    var_pos = (sumsq_pos / count_pos.clamp_min(1.0)) - mean_pos.pow(2)
                    np.save(os.path.join(ep_dir, "val_pred_token_variance.npy"), var_pos.numpy())
                    # Plot variance curve
                    plt.figure(figsize=(10, 3))
                    plt.plot(var_pos.numpy())
                    plt.title("Per-token Output Variance (validation predictions)")
                    plt.xlabel("Token index (1..511)")
                    plt.ylabel("Variance")
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_dir, "val_pred_token_variance.png"), dpi=140)
                    plt.close()

                # (2) Self-Similarity Heatmap from one sampled latent
                z_sample = sample_latent(model, token_dim=args.latent_dim, seq_len=512, device=device)
                z_norm = z_sample / (np.linalg.norm(z_sample, axis=1, keepdims=True) + 1e-8)
                sim = np.clip(z_norm @ z_norm.T, -1.0, 1.0)  # cosine similarity
                np.save(os.path.join(ep_dir, "self_similarity.npy"), sim)
                plt.figure(figsize=(6, 5))
                plt.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
                plt.colorbar()
                plt.title("Self-Similarity (Cosine) of Sampled Latent")
                plt.tight_layout()
                plt.savefig(os.path.join(ep_dir, "self_similarity.png"), dpi=140)
                plt.close()

                # (2b) 若提供 VAE decoder：將 latent 轉回體素，輸出 3 視角投影 + 佔空比
                if vae_decoder is not None:
                    with torch.no_grad():
                        # z_sample: [512, D] -> [1, D, 8, 8, 8]
                        D = z_sample.shape[1]
                        z_reshaped = torch.from_numpy(z_sample).to(device=device, dtype=torch.float32)
                        z_reshaped = z_reshaped.view(1, 8, 8, 8, D).permute(0, 4, 1, 2, 3).contiguous()
                        logits = vae_decoder(z_reshaped, skips=None)[0].detach().cpu().numpy()  # [C,32,32,32]
                        labels = np.argmax(logits, axis=0).astype(np.uint8)  # [32,32,32]

                        # ====== 新增：佔空比計算 ======
                        non_air_ratio, bbox_ratio = compute_decoder_occupancy(labels, air_class=0)

                        with open(os.path.join(ep_dir, "decoder_occupancy.txt"), "w") as f_occ:
                            f_occ.write("Decoder occupancy diagnostics (generated latent)\n")
                            f_occ.write(f"  non_air_ratio        = {non_air_ratio:.6f}\n")
                            f_occ.write(f"  bbox_volume_ratio    = {bbox_ratio:.6f}\n")
                            f_occ.write("\n建議參考：\n")
                            f_occ.write("  • 若 non_air_ratio 非常接近 0 → 幾乎全空氣，代表 Transformer latent 掉出 manifold。\n")
                            f_occ.write("  • 可以跟 VAE 在真實資料上的平均 non_air_ratio 比較（約 ~0.02 左右）。\n")
                            f_occ.write("  • bbox_volume_ratio 很小 → 只剩一小撮點，疑似 collapsed blob。\n")

                        # ====== 原本的 3 視角投影 ======
                        max_z = labels.max(axis=0)
                        max_y = labels.max(axis=1)
                        max_x = labels.max(axis=2)
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                        axes[0].imshow(max_z); axes[0].set_title("MaxProj Z (Y,X)")
                        axes[1].imshow(max_y); axes[1].set_title("MaxProj Y (Z,X)")
                        axes[2].imshow(max_x); axes[2].set_title("MaxProj X (Z,Y)")
                        for ax in axes: ax.axis("off")
                        fig.tight_layout()
                        plt.savefig(os.path.join(ep_dir, "vae_projection.png"), dpi=140)
                        plt.close(fig)

                # (3) Distribution Drift: KL (diag-Gaussian), MMD (RBF), mean/std compare
                def diagonal_gaussian_kl(mu_p, std_p, mu_q, std_q, eps=1e-8):
                    var_p = np.maximum(std_p**2, eps)
                    var_q = np.maximum(std_q**2, eps)
                    term = np.log(var_q / var_p + eps) + (var_p + (mu_p - mu_q) ** 2) / (var_q + eps) - 1.0
                    return 0.5 * float(np.sum(term))

                def rbf_mmd(x, y, sigma=None):
                    # x:[N,D], y:[M,D]
                    def pdist(a, b):
                        aa = np.sum(a * a, axis=1, keepdims=True)
                        bb = np.sum(b * b, axis=1, keepdims=True).T
                        ab = a @ b.T
                        return np.maximum(aa + bb - 2 * ab, 0.0)
                    if sigma is None:
                        # median heuristic on combined
                        combined = np.concatenate([x, y], axis=0)
                        d2 = pdist(combined, combined)
                        med = np.median(d2[np.triu_indices_from(d2, k=1)])
                        sigma = np.sqrt(med / 2.0 + 1e-8)
                        sigma = max(float(sigma), 1e-6)
                    gamma = 1.0 / (2.0 * sigma * sigma)
                    k_xx = np.exp(-gamma * pdist(x, x))
                    k_yy = np.exp(-gamma * pdist(y, y))
                    k_xy = np.exp(-gamma * pdist(x, y))
                    n = x.shape[0]
                    m = y.shape[0]
                    # Unbiased MMD^2
                    np.fill_diagonal(k_xx, 0.0)
                    np.fill_diagonal(k_yy, 0.0)
                    mmd2 = k_xx.sum() / (n * (n - 1) + 1e-8) + k_yy.sum() / (m * (m - 1) + 1e-8) - 2.0 * k_xy.mean()
                    return float(max(mmd2, 0.0))

                # Collect train tokens statistics (streaming)
                def stream_train_stats(dataset, max_tokens=None):
                    count = 0
                    mean = None
                    M2 = None
                    collected = []
                    for z in dataset:
                        z_np = z.numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
                        # z_np shape [512, D]
                        if max_tokens is not None and count >= max_tokens:
                            break
                        take = z_np
                        if max_tokens is not None and count + take.shape[0] > max_tokens:
                            need = max_tokens - count
                            take = take[:need]
                        # per-dimension Welford
                        for row in take:
                            x = row  # [D]
                            if mean is None:
                                mean = np.zeros_like(x, dtype=np.float64)
                                M2 = np.zeros_like(x, dtype=np.float64)
                            count += 1
                            delta = x - mean
                            mean += delta / count
                            delta2 = x - mean
                            M2 += delta * delta2
                    if count < 2:
                        var = np.zeros_like(mean) if mean is not None else None
                    else:
                        var = M2 / (count - 1)
                    std = np.sqrt(np.maximum(var, 1e-8))
                    return mean.astype(np.float64), std.astype(np.float64)

                # Prepare stats and samples
                # Use up to 8192 tokens for train stats to balance speed/memory
                train_mean, train_std = stream_train_stats(train_ds, max_tokens=8192)
                # Generated sample tokens: sample K sequences
                K = 4
                gen_list = []
                for _ in range(K):
                    s = sample_latent(model, token_dim=args.latent_dim, seq_len=512, device=device, mode=args.sample_mode)  # [512,D]
                    gen_list.append(s)
                gen_tokens = np.concatenate(gen_list, axis=0)  # [K*512, D]
                gen_mean = gen_tokens.mean(axis=0)
                gen_std = gen_tokens.std(axis=0) + 1e-8

                kl_pq = diagonal_gaussian_kl(train_mean, train_std, gen_mean, gen_std)
                # MMD on downsampled sets
                n_mmd_train = min(4096, train_mean.shape[0] * 0 + gen_tokens.shape[0])  # fallback
                # Sample train tokens again up to 4096
                train_samples = []
                total_needed = 4096
                for z in train_ds:
                    arr = z.numpy()
                    train_samples.append(arr)
                    if sum(x.shape[0] for x in train_samples) >= total_needed:
                        break
                if len(train_samples) > 0:
                    train_tokens_sample = np.concatenate(train_samples, axis=0)[:total_needed]
                else:
                    train_tokens_sample = gen_tokens[:total_needed]  # fallback to same shape
                gen_tokens_sample = gen_tokens[:2048]
                mmd_val = rbf_mmd(train_tokens_sample, gen_tokens_sample)
                mean_l2 = float(np.linalg.norm(train_mean - gen_mean))
                std_l2 = float(np.linalg.norm(train_std - gen_std))

                # Save metrics
                drift_csv = os.path.join(ep_dir, "distribution_drift_metrics.csv")
                with open(drift_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "value"])
                    w.writerow(["kl_diag_gaussian_train||gen", f"{kl_pq:.6f}"])
                    w.writerow(["mmd_rbf", f"{mmd_val:.6f}"])
                    w.writerow(["mean_l2", f"{mean_l2:.6f}"])
                    w.writerow(["std_l2", f"{std_l2:.6f}"])
                console.print(f"[dim]Analytics saved to {ep_dir}[/dim]")
                current_kl = kl_pq
                current_mmd = mmd_val

                # ====== 新增：Real vs Generated latent PCA ======
                try:
                    # 取相同數量的 token 來做 PCA，避免不平衡
                    n_real = train_tokens_sample.shape[0]
                    n_gen = gen_tokens.shape[0]
                    n = min(n_real, n_gen, 4096)  # 最多 4096 個點
                    real_for_pca = train_tokens_sample[:n]
                    gen_for_pca = gen_tokens[:n]

                    X = np.concatenate([real_for_pca, gen_for_pca], axis=0)  # [2n, D]
                    labels_pca = np.array([0] * n + [1] * n)  # 0=real, 1=gen

                    pca = PCA(n_components=2)
                    X2 = pca.fit_transform(X)  # [2n, 2]

                    # 繪圖
                    plt.figure(figsize=(6, 6))
                    plt.scatter(
                        X2[labels_pca == 0, 0],
                        X2[labels_pca == 0, 1],
                        s=4,
                        alpha=0.5,
                        label="real"
                    )
                    plt.scatter(
                        X2[labels_pca == 1, 0],
                        X2[labels_pca == 1, 1],
                        s=4,
                        alpha=0.5,
                        label="generated"
                    )
                    plt.legend()
                    plt.title("Real vs Generated Latent (PCA)")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_dir, "latent_pca_real_vs_gen.png"), dpi=140)
                    plt.close()

                    # 也順便存一些簡單數值指標
                    mean_real = real_for_pca.mean(axis=0)
                    mean_gen = gen_for_pca.mean(axis=0)
                    std_real = real_for_pca.std(axis=0)
                    std_gen = gen_for_pca.std(axis=0)

                    mean_l2_pca = float(np.linalg.norm(mean_real - mean_gen))
                    std_l2_pca = float(np.linalg.norm(std_real - std_gen))

                    with open(os.path.join(ep_dir, "latent_pca_stats.csv"), "w", newline="") as f_pca:
                        w_pca = csv.writer(f_pca)
                        w_pca.writerow(["metric", "value"])
                        w_pca.writerow(["mean_l2_real_gen", f"{mean_l2_pca:.6f}"])
                        w_pca.writerow(["std_l2_real_gen", f"{std_l2_pca:.6f}"])
                        w_pca.writerow(["n_points_per_class", n])

                except Exception as e:
                    console.print(f"[yellow]PCA diagnostics failed:[/yellow] {e}")

            # Console line with KL beside val when available
            best_marker = " | ★ Best!" if is_best else ""
            kl_str = f" | KL={current_kl:.6f}" if current_kl is not None else ""
            mmd_str = f" | MMD={current_mmd:.6f}" if current_mmd is not None else ""
            console.print(
                f"Epoch {epoch:03d}: train={train_loss:.6f} | val={val_loss:.6f}{kl_str}{mmd_str} | {dt:.1f}s{best_marker}"
            )

            # History
            cum_secs = time.time() - start_time
            history_row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "kl_train_gen": f"{current_kl:.6f}" if current_kl is not None else "",
                "mmd_rbf": f"{current_mmd:.6f}" if current_mmd is not None else "",
                "epoch_time_secs": f"{dt:.2f}",
                "cumulative_time_secs": f"{cum_secs:.2f}",
                "is_best": "TRUE" if is_best else "FALSE",
            }
            training_history.append(history_row)

            # Immediately append to training_history.csv
            with open(history_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=history_row.keys())
                writer.writerow(history_row)

    # Final Test
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_loss = eval_one_epoch(model, test_loader, device)
    print(f"Final Test Loss = {test_loss:.6f}")

    # Training history CSV is already being written incrementally during training
    console.print(
        f"[green]✓[/green] Training history saved incrementally to [cyan]{history_csv}[/cyan]"
    )

    # Append final metadata fields to metadata file (append to end to maintain field order)
    end_time_dt = datetime.now()
    total_secs = time.time() - start_time
    final_metadata = {
        "best_val_loss": f"{best_val:.6f}",
        "final_test_loss": f"{test_loss:.6f}",
        "training_finished": end_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_time_secs": f"{total_secs:.2f}",
    }

    # Append to key-value format metadata
    with open(metadata_csv, "a", newline="") as f:
        writer = csv.writer(f)
        for k, v in final_metadata.items():
            writer.writerow([k, v])
    console.print(
        f"[green]✓[/green] Updated experiment metadata with final fields: [cyan]{metadata_csv}[/cyan]"
    )

    # Sample a latent
    sample = sample_latent(model, token_dim=args.latent_dim, seq_len=512, device=device, mode=args.sample_mode)
    np.save(os.path.join(exp_dir, "sample_latent.npy"), sample)
    print(f"Saved sample latent to {os.path.join(exp_dir, 'sample_latent.npy')}.")

    # 釋放資料集快取，歸還記憶體
    if hasattr(train_ds, "release"):
        train_ds.release()
    if hasattr(val_ds, "release"):
        val_ds.release()
    if hasattr(test_ds, "release"):
        test_ds.release()

if __name__ == "__main__":
    main()
