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
from rich.panel import Panel

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.stats import entropy

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

def latent_coverage_ratio(real_latent, gen_latent, percentile=20):
    """
    Compute coverage ratio: how many real latents are covered by generated latents.
    
    Parameters
    ----------
    real_latent : np.ndarray
        Array of shape [N, D] containing real latent tokens.
    gen_latent : np.ndarray
        Array of shape [M, D] containing generated latent tokens.
    percentile : int, default=20
        Percentile to use for radius determination from real distribution.
    
    Returns
    -------
    coverage : float
        Ratio of real latents covered by generated latents.
    radius : float
        Radius used for coverage calculation.
    """
    # (1) Determine radius r from REAL distribution
    real_dist = pairwise_distances(real_latent)
    # Only take off-diagonal (exclude self-distances) - use upper triangle indices
    triu_indices = np.triu_indices_from(real_dist, k=1)
    real_dist_off_diag = real_dist[triu_indices]
    r = np.percentile(real_dist_off_diag, percentile)

    # (2) Compute coverage
    dist_rg = pairwise_distances(real_latent, gen_latent)
    min_dist = dist_rg.min(axis=1)
    covered = (min_dist < r).mean()

    return covered, r

def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two probability distributions."""
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def pairwise_distance_js(real_latent, gen_latent, bins=50):
    """
    Compute JS divergence between pairwise distance distributions of real and generated latents.
    
    Parameters
    ----------
    real_latent : np.ndarray
        Array of shape [N, D] containing real latent tokens.
    gen_latent : np.ndarray
        Array of shape [M, D] containing generated latent tokens.
    bins : int, default=50
        Number of bins for histogram.
    
    Returns
    -------
    js : float
        Jensen-Shannon divergence between pairwise distance distributions.
    """
    # 1. Pairwise distances
    d_real = pdist(real_latent)
    d_gen = pdist(gen_latent)

    # 2. Histogram (normalize)
    hist_real, edges = np.histogram(d_real, bins=bins, density=True)
    hist_gen, _ = np.histogram(d_gen, bins=bins, density=True)

    # 3. JS divergence
    js = js_divergence(hist_real, hist_gen)

    return js

def novelty_score(real_latent, gen_latent):
    """
    Compute novelty score: how far generated latents are from nearest real latents.
    
    Parameters
    ----------
    real_latent : np.ndarray
        Array of shape [N, D] containing real latent tokens.
    gen_latent : np.ndarray
        Array of shape [M, D] containing generated latent tokens.
    
    Returns
    -------
    dict
        Dictionary containing novelty_mean, novelty_std, novelty_min, novelty_max.
    """
    # Distance from each generated latent to nearest real latent
    dist_rg = pairwise_distances(gen_latent, real_latent)
    min_dist = dist_rg.min(axis=1)

    return {
        "novelty_mean": float(min_dist.mean()),
        "novelty_std": float(min_dist.std()),
        "novelty_min": float(min_dist.min()),
        "novelty_max": float(min_dist.max()),
    }

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
    # 自動推斷 latent_dim 和 latent grid 大小（若使用者未提供）
    # ------------------------------------------------
    seq_len = None
    latent_grid_size = None
    if args.latent_dim is None:
        if len(train_files) == 0:
            raise ValueError("未提供 --latent_dim，且 train/ 資料夾為空，無法推斷 latent_dim。")
        # 從 train/ 的第一個檔案推斷維度
        probe = np.load(train_files[0], mmap_mode="r")
        if probe.ndim != 2:
            raise ValueError(f"檔案形狀不正確（期望 2 維）：{train_files[0]} 取得 {probe.shape}")
        seq_len = int(probe.shape[0])
        args.latent_dim = int(probe.shape[1])
        print(f"[Info] 自動推斷 latent_dim = {args.latent_dim} 來自 {os.path.basename(train_files[0])}")
        print(f"[Info] 序列長度 = {seq_len}")
        
        # 推斷 latent grid 大小（假設是立方體：grid_size^3 = seq_len）
        grid_size_candidate = round(seq_len ** (1.0 / 3.0))
        if grid_size_candidate ** 3 == seq_len:
            latent_grid_size = grid_size_candidate
            print(f"[Info] 自動推斷 latent_grid_size = {latent_grid_size} (因為 {latent_grid_size}^3 = {seq_len})")
        else:
            # 如果不是完美的立方數，嘗試其他常見配置
            print(f"[Warning] 序列長度 {seq_len} 不是完美立方數（{grid_size_candidate}^3 = {grid_size_candidate**3} ≠ {seq_len}）")
            print(f"[Warning] 無法自動推斷 latent grid 大小，將設為未知")
            latent_grid_size = None
        
        # 基本一致性檢查：各 split 的第一個檔案（若存在）需有相同維度
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if len(files) == 0:
                continue
            probe2 = np.load(files[0], mmap_mode="r")
            if probe2.ndim != 2 or probe2.shape[0] != seq_len or probe2.shape[1] != args.latent_dim:
                raise ValueError(
                    f"{split_name}/ 形狀不一致或不為 [{seq_len}, {args.latent_dim}]：" +
                    f"{files[0]} 取得 {probe2.shape}"
                )
    else:
        # 如果 latent_dim 已提供，仍需要檢查序列長度來推斷 grid 大小
        if len(train_files) > 0:
            probe = np.load(train_files[0], mmap_mode="r")
            if probe.ndim == 2:
                seq_len = int(probe.shape[0])
                grid_size_candidate = round(seq_len ** (1.0 / 3.0))
                if grid_size_candidate ** 3 == seq_len:
                    latent_grid_size = grid_size_candidate
                    print(f"[Info] 自動推斷 latent_grid_size = {latent_grid_size} (因為 {latent_grid_size}^3 = {seq_len})")
                else:
                    latent_grid_size = None
                    print(f"[Warning] 序列長度 {seq_len} 不是完美立方數，無法自動推斷 latent grid 大小")
            else:
                seq_len = 512  # 預設值
                latent_grid_size = 8  # 預設值
        else:
            seq_len = 512  # 預設值
            latent_grid_size = 8  # 預設值

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

    # Build model (use inferred sequence length or default to 512)
    max_seq_len = seq_len if seq_len is not None else 512
    model = TransformerPrior(
        token_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_seq=max_seq_len,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    # 建立 out_dir/exp_name 實驗資料夾
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    
    # 檢查實驗目錄是否存在且不為空，避免覆蓋現有資料
    if os.path.exists(exp_dir) and os.listdir(exp_dir):
        console.print(
            Panel.fit(
                "[bold red]ERROR: Experiment Directory Not Empty[/bold red]\n\n"
                f"[yellow]{exp_dir}[/yellow]\n"
                "Use another --exp_name, clean directory, or delete the existing directory.",
                border_style="red",
            )
        )
        raise SystemExit(1)
    
    os.makedirs(exp_dir, exist_ok=True)
    best_val = 1e10
    best_path = os.path.join(exp_dir, "best_prior.pt")
    analytics_dir = os.path.join(exp_dir, "analytics")
    os.makedirs(analytics_dir, exist_ok=True)
    analytics_summary_csv = os.path.join(exp_dir, f"analytics_summary_{args.exp_name}.csv")
    
    # Create samples directory (only if VAE decoder is available)
    samples_dir = None
    if args.vae_ckpt:
        samples_dir = os.path.join(exp_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

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
    metadata_flat_csv = os.path.join(exp_dir, f"experiment_metadata_flat_{args.exp_name}.csv")

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
        "latent_sequence_length": seq_len if seq_len is not None else "unknown",
        "latent_grid_size": f"{latent_grid_size}x{latent_grid_size}x{latent_grid_size}" if latent_grid_size is not None else "unknown",
        "latent_grid_size_d": latent_grid_size if latent_grid_size is not None else "unknown",
        "latent_total_tokens": seq_len if seq_len is not None else "unknown",
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

    # Save initial metadata (flat format - single row with all fields as columns)
    with open(metadata_flat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=initial_metadata.keys())
        writer.writeheader()
        writer.writerow(initial_metadata)
    console.print(
        f"[green]✓[/green] Created initial metadata (flat) file: [cyan]{metadata_flat_csv}[/cyan]"
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
                z_sample = sample_latent(model, token_dim=args.latent_dim, seq_len=max_seq_len, device=device)
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

                # (2a) Sample Variance & Multi-Sample PCA
                K = 5
                multi_samples = []
                multi_samples_flat = []
                for _ in range(K):
                    s = sample_latent(model, token_dim=args.latent_dim, seq_len=max_seq_len, device=device, mode=args.sample_mode)
                    multi_samples.append(s)  # keep original shape [seq_len, D]
                    multi_samples_flat.append(s.reshape(-1))  # flatten to 1D for variance calculation
                
                multi_samples_stacked = np.stack(multi_samples_flat)  # shape [K, seq_len*D]
                
                # [S1] Sample Variance: pairwise L2 distances
                dists = []
                for i in range(K):
                    for j in range(i + 1, K):
                        d = np.linalg.norm(multi_samples_stacked[i] - multi_samples_stacked[j])
                        dists.append(d)
                sample_variance = np.mean(dists) if len(dists) > 0 else 0.0
                
                # Save sample variance metric
                with open(os.path.join(ep_dir, "sample_variance.txt"), "w") as f:
                    f.write("Sample Variance Diagnostics\n")
                    f.write(f"  sample_variance (mean pairwise L2) = {sample_variance:.6f}\n")
                    f.write(f"  n_samples = {K}\n")
                    f.write("\n建議參考：\n")
                    f.write("  • 若 sample_variance < 1e-3 → collapse（所有樣本幾乎相同）\n")
                    f.write("  • 若 sample_variance 明顯大於 0 → 有多樣性\n")
                
                # [S2] Multi-Sample PCA: visualize all K samples in PCA space
                Z_flat = np.concatenate(multi_samples, axis=0)  # [K*seq_len, D]
                pca_multi = PCA(n_components=2)
                Z_pca = pca_multi.fit_transform(Z_flat)  # [K*seq_len, 2]
                
                # Plot: color by which sample (K different colors)
                plt.figure(figsize=(8, 6))
                colors = plt.cm.tab10(np.linspace(0, 1, K))
                for k in range(K):
                    start_idx = k * max_seq_len
                    end_idx = (k + 1) * max_seq_len
                    plt.scatter(
                        Z_pca[start_idx:end_idx, 0],
                        Z_pca[start_idx:end_idx, 1],
                        s=2,
                        alpha=0.5,
                        label=f"Sample {k+1}",
                        color=colors[k],
                    )
                plt.legend()
                plt.title(f"Multi-Sample PCA (K={K} samples)")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.tight_layout()
                plt.savefig(os.path.join(ep_dir, "multi_sample_pca.png"), dpi=140)
                plt.close()
                
                # Save sample variance to CSV for easy tracking
                with open(os.path.join(ep_dir, "sample_variance.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "value"])
                    w.writerow(["sample_variance", f"{sample_variance:.6f}"])
                    w.writerow(["n_samples", K])

                # (2b) 若提供 VAE decoder：將 latent 轉回體素，輸出 3 視角投影 + 佔空比
                if vae_decoder is not None:
                    with torch.no_grad():
                        # z_sample: [seq_len, D] -> [1, D, grid_size, grid_size, grid_size]
                        D = z_sample.shape[1]
                        grid_size_to_use = latent_grid_size if latent_grid_size is not None else 8
                        z_reshaped = torch.from_numpy(z_sample).to(device=device, dtype=torch.float32)
                        z_reshaped = z_reshaped.view(1, grid_size_to_use, grid_size_to_use, grid_size_to_use, D).permute(0, 4, 1, 2, 3).contiguous()
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

                # (2c) 若提供 VAE decoder：生成 3 個樣本並保存到 samples 目錄
                if vae_decoder is not None and samples_dir is not None:
                    n_samples = 3
                    grid_size_to_use = latent_grid_size if latent_grid_size is not None else 8
                    for sample_idx in range(n_samples):
                        with torch.no_grad():
                            # Sample a new latent sequence
                            z_sample_new = sample_latent(
                                model, 
                                token_dim=args.latent_dim, 
                                seq_len=max_seq_len, 
                                device=device, 
                                mode=args.sample_mode
                            )  # [seq_len, D]
                            
                            # Decode latent to voxel
                            D = z_sample_new.shape[1]
                            z_reshaped = torch.from_numpy(z_sample_new).to(device=device, dtype=torch.float32)
                            z_reshaped = z_reshaped.view(
                                1, grid_size_to_use, grid_size_to_use, grid_size_to_use, D
                            ).permute(0, 4, 1, 2, 3).contiguous()
                            logits = vae_decoder(z_reshaped, skips=None)[0].detach().cpu().numpy()  # [C,32,32,32]
                            labels = np.argmax(logits, axis=0).astype(np.uint8)  # [32,32,32]
                            
                            # Save voxel as .npz file (sample_idx starts from 0, but filename uses 1-based indexing)
                            sample_npz_name = f"sample_e{epoch}_{sample_idx + 1}_{args.exp_name}.npz"
                            sample_npz_path = os.path.join(samples_dir, sample_npz_name)
                            np.savez_compressed(sample_npz_path, labels)
                            
                            # Create 3-view projection
                            max_z = labels.max(axis=0)
                            max_y = labels.max(axis=1)
                            max_x = labels.max(axis=2)
                            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                            axes[0].imshow(max_z); axes[0].set_title("MaxProj Z (Y,X)")
                            axes[1].imshow(max_y); axes[1].set_title("MaxProj Y (Z,X)")
                            axes[2].imshow(max_x); axes[2].set_title("MaxProj X (Z,Y)")
                            for ax in axes: ax.axis("off")
                            fig.tight_layout()
                            
                            # Save PNG file (sample_idx starts from 0, but filename uses 1-based indexing)
                            sample_png_name = f"sample_e{epoch}_{sample_idx + 1}_{args.exp_name}.png"
                            sample_png_path = os.path.join(samples_dir, sample_png_name)
                            plt.savefig(sample_png_path, dpi=140)
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
                    s = sample_latent(model, token_dim=args.latent_dim, seq_len=max_seq_len, device=device, mode=args.sample_mode)  # [seq_len,D]
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

                    # ====== 新增：Latent Coverage Ratio, Pairwise Distance JS Divergence, Novelty Score ======
                    # Use the same samples as PCA for consistency
                    # Ensure we have enough samples (at least 10 for meaningful statistics)
                    min_samples = 10
                    if len(real_for_pca) < min_samples or len(gen_for_pca) < min_samples:
                        console.print(f"[yellow]Warning:[/yellow] Insufficient samples for advanced metrics (real: {len(real_for_pca)}, gen: {len(gen_for_pca)}), skipping")
                    else:
                        # (1) Latent Coverage Ratio
                        coverage, coverage_radius = latent_coverage_ratio(real_for_pca, gen_for_pca, percentile=20)
                        
                        # (2) Pairwise Distance JS Divergence
                        js_div = pairwise_distance_js(real_for_pca, gen_for_pca, bins=50)
                        
                        # (3) Novelty Score
                        novelty_dict = novelty_score(real_for_pca, gen_for_pca)
                        
                        # Save to CSV
                        advanced_metrics_csv = os.path.join(ep_dir, "advanced_latent_metrics.csv")
                        with open(advanced_metrics_csv, "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["metric", "value"])
                            w.writerow(["coverage_ratio", f"{coverage:.6f}"])
                            w.writerow(["coverage_radius", f"{coverage_radius:.6f}"])
                            w.writerow(["pairwise_distance_js_divergence", f"{js_div:.6f}"])
                            w.writerow(["novelty_mean", f"{novelty_dict['novelty_mean']:.6f}"])
                            w.writerow(["novelty_std", f"{novelty_dict['novelty_std']:.6f}"])
                            w.writerow(["novelty_min", f"{novelty_dict['novelty_min']:.6f}"])
                            w.writerow(["novelty_max", f"{novelty_dict['novelty_max']:.6f}"])

                except Exception as e:
                    console.print(f"[yellow]PCA diagnostics failed:[/yellow] {e}")

                # ====== 追加到匯總 CSV ======
                try:
                    import re
                    row = {"epoch": epoch}
                    
                    # 解析 decoder_occupancy.txt (如果存在)
                    occ_file = os.path.join(ep_dir, "decoder_occupancy.txt")
                    if os.path.exists(occ_file):
                        try:
                            with open(occ_file, "r") as f:
                                content = f.read()
                                non_air_match = re.search(r"non_air_ratio\s*=\s*([\d.]+)", content)
                                bbox_match = re.search(r"bbox_volume_ratio\s*=\s*([\d.]+)", content)
                                if non_air_match:
                                    row["non_air_ratio"] = float(non_air_match.group(1))
                                if bbox_match:
                                    row["bbox_volume_ratio"] = float(bbox_match.group(1))
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse decoder_occupancy.txt: {e}")
                    
                    # 讀取 distribution_drift_metrics.csv
                    drift_file = os.path.join(ep_dir, "distribution_drift_metrics.csv")
                    if os.path.exists(drift_file):
                        try:
                            with open(drift_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "kl_diag_gaussian_train||gen":
                                        row["kl_diag_gaussian_train_gen"] = float(value) if value else ""
                                    elif metric == "mmd_rbf":
                                        row["mmd_rbf"] = float(value) if value else ""
                                    elif metric == "mean_l2":
                                        row["mean_l2"] = float(value) if value else ""
                                    elif metric == "std_l2":
                                        row["std_l2"] = float(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse distribution_drift_metrics.csv: {e}")
                    
                    # 讀取 latent_pca_stats.csv
                    pca_file = os.path.join(ep_dir, "latent_pca_stats.csv")
                    if os.path.exists(pca_file):
                        try:
                            with open(pca_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "mean_l2_real_gen":
                                        row["pca_mean_l2_real_gen"] = float(value) if value else ""
                                    elif metric == "std_l2_real_gen":
                                        row["pca_std_l2_real_gen"] = float(value) if value else ""
                                    elif metric == "n_points_per_class":
                                        row["pca_n_points_per_class"] = int(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse latent_pca_stats.csv: {e}")
                    
                    # 讀取 advanced_latent_metrics.csv
                    advanced_metrics_file = os.path.join(ep_dir, "advanced_latent_metrics.csv")
                    if os.path.exists(advanced_metrics_file):
                        try:
                            with open(advanced_metrics_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "coverage_ratio":
                                        row["coverage_ratio"] = float(value) if value else ""
                                    elif metric == "coverage_radius":
                                        row["coverage_radius"] = float(value) if value else ""
                                    elif metric == "pairwise_distance_js_divergence":
                                        row["pairwise_distance_js_divergence"] = float(value) if value else ""
                                    elif metric == "novelty_mean":
                                        row["novelty_mean"] = float(value) if value else ""
                                    elif metric == "novelty_std":
                                        row["novelty_std"] = float(value) if value else ""
                                    elif metric == "novelty_min":
                                        row["novelty_min"] = float(value) if value else ""
                                    elif metric == "novelty_max":
                                        row["novelty_max"] = float(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse advanced_latent_metrics.csv: {e}")
                    
                    # 讀取 sample_variance.csv
                    sample_variance_file = os.path.join(ep_dir, "sample_variance.csv")
                    if os.path.exists(sample_variance_file):
                        try:
                            with open(sample_variance_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "sample_variance":
                                        row["sample_variance"] = float(value) if value else ""
                                    elif metric == "n_samples":
                                        row["sample_variance_n_samples"] = int(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse sample_variance.csv: {e}")
                    
                    # 處理匯總 CSV：讀取現有數據，合併新欄位，追加新行
                    file_exists = os.path.exists(analytics_summary_csv)
                    existing_rows = []
                    existing_fields = []
                    
                    if file_exists:
                        # 讀取現有數據和欄位
                        with open(analytics_summary_csv, "r", newline="") as f:
                            reader = csv.DictReader(f)
                            existing_fields = list(reader.fieldnames) if reader.fieldnames else []
                            existing_rows = list(reader)
                    
                    # 確定所有欄位（epoch 在前，其他按字母順序）
                    all_fields = set(existing_fields) if existing_fields else set()
                    all_fields.update(row.keys())
                    all_fields_sorted = ["epoch"] + sorted([f for f in all_fields if f != "epoch"])
                    
                    # 重寫整個文件（包含新行）
                    with open(analytics_summary_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=all_fields_sorted)
                        writer.writeheader()
                        # 寫入現有行（補充缺失欄位為空字符串）
                        for er in existing_rows:
                            complete_row = {field: er.get(field, "") for field in all_fields_sorted}
                            writer.writerow(complete_row)
                        # 追加新行（補充缺失欄位為空字符串）
                        complete_row = {field: row.get(field, "") for field in all_fields_sorted}
                        writer.writerow(complete_row)
                    
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Failed to append to analytics summary CSV: {e}")

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

    # For flat format, read existing metadata, merge with final fields, and rewrite
    existing_flat_metadata = {}
    if os.path.exists(metadata_flat_csv):
        with open(metadata_flat_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                row = next(reader, None)
                if row:
                    existing_flat_metadata = row

    # Merge with final metadata
    all_flat_metadata = {**existing_flat_metadata, **final_metadata}
    
    # Rewrite flat format with all fields
    with open(metadata_flat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_flat_metadata.keys())
        writer.writeheader()
        writer.writerow(all_flat_metadata)
    console.print(
        f"[green]✓[/green] Updated experiment metadata (flat) with final fields: [cyan]{metadata_flat_csv}[/cyan]"
    )

    # Sample a latent
    sample = sample_latent(model, token_dim=args.latent_dim, seq_len=max_seq_len, device=device, mode=args.sample_mode)
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
