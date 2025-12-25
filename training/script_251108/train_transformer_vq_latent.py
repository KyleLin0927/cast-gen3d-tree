#!/usr/bin/env python3

import os
import math
import argparse
import time
from glob import glob
import zipfile
import tempfile
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime
import csv
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
from scipy import ndimage

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
    Loads latent token indices from .npy files.
    Expected shape: [512] (each element is a codebook index)
    
    Parameters
    ----------
    files : List[str]
        List of file paths to .npy latent files. Each file should have
        shape [512] containing integer codebook indices.
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
                arr = np.load(fp)  # expect (512,) or (512, 1)
                # Handle both 1D and 2D arrays
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.squeeze(1)
                if arr.ndim != 1 or arr.shape[0] != 512:
                    raise ValueError(f"Expected shape [512], got {arr.shape} from {fp}")
                t = torch.from_numpy(arr.astype(np.int64))  # int64 for embedding indices
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
        arr = np.load(self.files[idx])  # shape (512,)
        # Handle both 1D and 2D arrays
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        z = torch.from_numpy(arr.astype(np.int64))  # int64 for embedding indices
        return z  # return full sequence of token indices

    def release(self):
        """Release in-memory cache to free RAM."""
        self._cache = None

# ------------------------------------------------
# Transformer Model (GPT-like)
# ------------------------------------------------

class TransformerPrior(nn.Module):
    def __init__(
        self,
        num_tokens,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_seq=512
    ):
        """
        GPT-style Transformer prior over discrete codebook token indices.
        
        Uses a learned BOS (Beginning of Sequence) token at index `num_tokens` to avoid
        forcing all generations to start with token #0, which would cause:
        - Biased transition entropy for token #0
        - Reduced generation diversity
        - Pseudo collapse bias in training and analytics
        
        Parameters
        ----------
        num_tokens : int
            Codebook size (number of distinct token indices: [0, num_tokens-1]).
            BOS token is at index `num_tokens`.
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

        self.num_tokens = num_tokens
        self.max_seq = max_seq

        # BOS token ID: use codebook_size as special BOS token
        # This allows the model to learn a proper starting token instead of forcing token #0
        self.bos_token_id = num_tokens  # BOS token is at index num_tokens

        # Token embedding: maps codebook indices + BOS token to hidden_dim vectors
        # num_tokens + 1: regular tokens [0, num_tokens-1] + BOS token [num_tokens]
        self.embedding = nn.Embedding(num_tokens + 1, hidden_dim)

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

        # Output head: predict logits over codebook indices (not including BOS)
        # BOS is only used as input, never as output
        self.output_head = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x):
        """
        Forward pass through the prior to predict next-token logits.
        Uses causal mask to ensure autoregressive property (no future information).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [B, S] containing token indices (int64), where S <= max_seq.
            Token indices can be in range [0, num_tokens-1] for regular tokens, or
            `num_tokens` for BOS token (used during generation).
        
        Returns
        -------
        logits : torch.Tensor
            Logits tensor with shape [B, S, num_tokens], representing
            predicted next-token probabilities over codebook indices [0, num_tokens-1].
            Note: BOS token is never predicted as output, only used as input.
        """
        B, S = x.shape
        assert S <= self.max_seq, "Input sequence longer than max_seq"

        # Embedding lookup + positional embedding
        h = self.embedding(x) + self.pos_emb[:, :S, :]  # [B, S, hidden_dim]

        # =============== CAUSAL MASK ===============
        # mask shape must be [S, S] even when batch_first=True
        # mask[i,j] = -inf 代表 token_i 不可以看到 token_j
        mask = torch.triu(
            torch.full((S, S), float('-inf'), device=x.device),
            diagonal=1
        )
        # ============================================

        # Transformer encoder (now behaves like GPT decoder)
        # Explicitly set src_key_padding_mask=None to avoid future padding token issues
        # (Currently seq_len is fixed, but this is good practice)
        h = self.transformer(h, mask=mask, src_key_padding_mask=None)

        # Output logits over codebook indices
        logits = self.output_head(h)  # [B, S, num_tokens]
        return logits

# ------------------------------------------------
# Training step
# ------------------------------------------------

def train_one_epoch(model, loader, opt, device, use_amp=False, scaler=None):
    """
    Autoregressive Transformer prior training:
      - model predicts logits over codebook indices
      - compute CrossEntropy loss on next-token prediction
      - Uses BOS token at the beginning of sequences during training
      - Pure autoregressive learning like GPT/LLaMA
    """
    model.train()
    total_loss = 0.0
    count = 0

    ce_loss = nn.CrossEntropyLoss()

    # 只在 CUDA 裝置上使用 AMP
    use_amp = use_amp and (device.type == "cuda") and (scaler is not None)

    for z in loader:
        z = z.to(device)  # [B, 512] - token indices
        B = z.size(0)

        # Insert BOS token at the beginning: [BOS, z[0], z[1], ..., z[511]]
        # This ensures the model learns what BOS represents during training
        bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
        x = torch.cat([bos, z], dim=1)  # [B, 513]: [BOS, z[0], z[1], ..., z[511]]
        
        # Target: [z[0], z[1], ..., z[511]] (does not include BOS)
        y_target = z  # [B, 512]

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(x)  # [B, 513, num_tokens]
                # Remove the last position prediction (we don't predict after the last token)
                logits = logits[:, :-1, :]  # [B, 512, num_tokens]
                loss = ce_loss(
                    logits.reshape(-1, model.num_tokens),
                    y_target.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)  # [B, 513, num_tokens]
            # Remove the last position prediction (we don't predict after the last token)
            logits = logits[:, :-1, :]  # [B, 512, num_tokens]
            loss = ce_loss(
                logits.reshape(-1, model.num_tokens),
                y_target.reshape(-1)
            )
            loss.backward()
            opt.step()

        total_loss += loss.item() * z.size(0)
        count += z.size(0)

    return total_loss / count

@torch.no_grad()
def eval_one_epoch(model, loader, device, use_amp=False):
    """
    Deterministic eval using argmax (no sampling).
    Uses CrossEntropy loss on next-token prediction.
    Uses BOS token at the beginning of sequences (consistent with training).
    """
    model.eval()
    total_loss = 0.0
    count = 0

    ce_loss = nn.CrossEntropyLoss()

    # 只在 CUDA 裝置上使用 AMP
    use_amp = use_amp and (device.type == "cuda")

    for z in loader:
        z = z.to(device)  # [B, 512] - token indices
        B = z.size(0)

        # Insert BOS token at the beginning: [BOS, z[0], z[1], ..., z[511]]
        # This matches the training setup
        bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
        x = torch.cat([bos, z], dim=1)  # [B, 513]: [BOS, z[0], z[1], ..., z[511]]
        
        # Target: [z[0], z[1], ..., z[511]] (does not include BOS)
        y_target = z  # [B, 512]

        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(x)  # [B, 513, num_tokens]
                # Remove the last position prediction
                logits = logits[:, :-1, :]  # [B, 512, num_tokens]
                loss = ce_loss(
                    logits.reshape(-1, model.num_tokens),
                    y_target.reshape(-1)
                )
        else:
            logits = model(x)  # [B, 513, num_tokens]
            # Remove the last position prediction
            logits = logits[:, :-1, :]  # [B, 512, num_tokens]
            loss = ce_loss(
                logits.reshape(-1, model.num_tokens),
                y_target.reshape(-1)
            )

        total_loss += loss.item() * z.size(0)
        count += z.size(0)

    return total_loss / count

# ------------------------------------------------
# Sampling
# ------------------------------------------------

@torch.no_grad()
def sample_latent(model, seq_len=512, device="cpu", mode="argmax", temperature=1.0):
    """
    Sample one latent sequence of discrete token indices:
        Start from first token, predict next tokens autoregressively.
    
    Parameters
    ----------
    model : nn.Module
        Trained Transformer prior model.
    seq_len : int, default=512
        Number of tokens to sample (sequence length).
    device : str, default="cpu"
        Device to run sampling on.
    mode : str, default="argmax"
        "argmax" uses argmax as next token; "sample" samples from softmax distribution.
    temperature : float, default=1.0
        Temperature for sampling (only used when mode="sample").
        Higher temperature = more diverse, lower = more deterministic.
    
    Returns
    -------
    numpy.ndarray
        Array of shape [seq_len] containing the sampled token indices (int64).
    """
    model.eval()

    # Start with learned BOS token instead of forcing token #0
    # This prevents pseudo collapse bias and allows proper grammar learning
    z = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    
    # Current sequence: start with BOS token (learned starting token)
    # BOS token ID = num_tokens (one beyond the regular codebook indices)
    current_seq = torch.full((1, 1), model.bos_token_id, dtype=torch.long, device=device)  # [1, 1]

    for i in range(seq_len):
        logits = model(current_seq)  # [1, S, num_tokens] where S is current sequence length
        logits_t = logits[:, -1, :]  # [1, num_tokens] - prediction for next token
        
        if mode == "sample":
            # Sample from softmax distribution with temperature
            probs = torch.softmax(logits_t / temperature, dim=-1)  # [1, num_tokens]
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        else:
            # Use argmax (deterministic)
            next_token = logits_t.argmax(dim=-1, keepdim=True)  # [1, 1]
        
        z[:, i] = next_token.squeeze(1)  # Store sampled token index
        
        # Append to sequence for next prediction: [z[0], z[1], ..., z[i]]
        current_seq = torch.cat([current_seq, next_token], dim=1)  # [1, i+2]

    return z[0].cpu().numpy().astype(np.int64)

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

def compute_largest_component_ratio(labels, air_class=0):
    """
    Compute the ratio of the largest connected component to all non-air voxels.
    
    Parameters
    ----------
    labels : numpy.ndarray
        Array of shape [32,32,32] containing class labels after argmax.
    air_class : int, default=0
        Class ID representing "air" voxels (typically 0).
    
    Returns
    -------
    largest_component_ratio : float
        Ratio of largest connected component size to total non-air voxels.
        Returns 0.0 if no non-air voxels exist.
    """
    mask = (labels != air_class).astype(np.int32)
    non_air_count = mask.sum()
    
    if non_air_count == 0:
        return 0.0
    
    # Find connected components (using 6-connectivity for 3D)
    labeled_mask, num_features = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    
    if num_features == 0:
        return 0.0
    
    # Count voxels in each component
    component_sizes = []
    for i in range(1, num_features + 1):
        component_size = (labeled_mask == i).sum()
        component_sizes.append(component_size)
    
    # Get largest component size
    largest_component_size = max(component_sizes) if component_sizes else 0
    largest_component_ratio = float(largest_component_size) / float(non_air_count + 1e-8)
    
    return largest_component_ratio


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
        default=None,
        help="Root directory containing latent npy files under train/ val/ test/. 與 --data_zip 二選一。"
    )
    parser.add_argument(
        "--data_zip",
        type=str,
        default=None,
        help="（選用）壓縮檔路徑（.zip），會自動解壓到臨時目錄使用。與 --data_root 二選一。"
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=None,
        help="Codebook size (number of distinct token indices). 如果提供 --resume，可從 checkpoint 讀取；否則必填。"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="（選用）從 checkpoint 恢復訓練。會自動讀取 codebook_size 和其他模型參數。"
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
        "--vqvae_ckpt",
        type=str,
        default=None,
        help="（選用）已訓練 VQ-VAE 的 checkpoint 路徑（.pt 檔案）；若提供，會在分析回合解碼 latent 並輸出投影圖與 occupancy ratio。"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="（選用）與 --vqvae_ckpt 相同，已訓練 VQ-VAE 的 checkpoint 路徑。"
    )
    parser.add_argument(
        "--vqvae_model_def",
        type=str,
        default=None,
        help="（選用）VQ-VAE 模型定義檔案；預設使用 train_VQVAE.py。"
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
        default="argmax",
        choices=["argmax", "sample"],
        help="採樣時使用 argmax（deterministic）或從 softmax 分佈取樣（sample）。"
    )
    parser.add_argument(
        "--sample_temperature",
        type=float,
        default=1.0,
        help="採樣溫度（僅在 sample_mode=sample 時使用）。較高溫度 = 更多樣性。"
    )
    parser.add_argument(
        "--analytics_n_samples",
        type=int,
        default=1000,
        help="分析指標使用的訓練和生成樣本數量（預設 1000）。用於 MMD/Coverage、decode quality 分布比較等。較大值更準確但更慢。"
    )
    parser.add_argument(
        "--n_sample_images",
        type=int,
        default=3,
        help="每次 analytics 回合保存的 sample 圖片數量（預設 3）。每個 sample 會生成 .npy（token indices）、.npz（decoded voxels）和 .png（3-view projection）檔案。"
    )

    args = parser.parse_args()

    # 初始化 console（用於後續訊息）
    console = Console()
    
    # 初始化臨時目錄變數（用於 --data_zip）
    temp_data_dir = None

    # 如果提供了 --resume，從 checkpoint 讀取 codebook_size 和其他參數
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint 檔案不存在：{args.resume}")
        try:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            ckpt_args = ckpt.get("args", {})
            if isinstance(ckpt_args, dict):
                # 從 checkpoint 讀取 codebook_size
                if "codebook_size" in ckpt_args:
                    if args.codebook_size is not None and args.codebook_size != ckpt_args["codebook_size"]:
                        console.print(
                            f"[yellow]Warning:[/yellow] --codebook_size={args.codebook_size} 與 checkpoint 中的 "
                            f"codebook_size={ckpt_args['codebook_size']} 不一致，將使用 checkpoint 的值。"
                        )
                    args.codebook_size = ckpt_args["codebook_size"]
                    console.print(f"[green]✓[/green] 從 checkpoint 讀取 codebook_size = {args.codebook_size}")
                else:
                    raise ValueError(f"Checkpoint 中找不到 codebook_size 參數：{args.resume}")
            else:
                raise ValueError(f"Checkpoint 格式不正確：args 應為 dict，但得到 {type(ckpt_args)}")
        except Exception as e:
            raise RuntimeError(f"無法從 checkpoint 讀取參數：{e}")
    
    # 如果還沒有 codebook_size，嘗試從 VQ-VAE checkpoint 讀取
    if args.codebook_size is None:
        vqvae_ckpt_path = args.vqvae_ckpt or args.ckpt
        if vqvae_ckpt_path:
            if not os.path.exists(vqvae_ckpt_path):
                console.print(f"[yellow]Warning:[/yellow] VQ-VAE checkpoint 不存在：{vqvae_ckpt_path}")
            else:
                try:
                    vqvae_ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=False)
                    vqvae_args = vqvae_ckpt.get("args", {})
                    if isinstance(vqvae_args, dict) and "codebook_size" in vqvae_args:
                        args.codebook_size = vqvae_args["codebook_size"]
                        console.print(f"[green]✓[/green] 從 VQ-VAE checkpoint 讀取 codebook_size = {args.codebook_size}")
                    else:
                        console.print(f"[yellow]Warning:[/yellow] VQ-VAE checkpoint 中找不到 codebook_size 參數")
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] 無法從 VQ-VAE checkpoint 讀取 codebook_size：{e}")
    
    # 驗證 codebook_size 是否已設定
    if args.codebook_size is None:
        raise ValueError(
            "必須提供 --codebook_size、--resume 或 --vqvae_ckpt/--ckpt（包含 codebook_size）。\n"
            "  • 使用 --codebook_size <size> 指定 codebook 大小\n"
            "  • 或使用 --resume <checkpoint_path> 從 Transformer prior checkpoint 自動讀取\n"
            "  • 或使用 --vqvae_ckpt/--ckpt <checkpoint_path> 從 VQ-VAE checkpoint 自動讀取"
        )

    seed_all(args.seed)

    # 處理 data_root 和 data_zip：二選一（與 train_VQVAE.py 保持一致）
    temp_dir_holder = []
    if args.data_zip and args.data_root:
        raise ValueError("不能同時提供 --data_root 和 --data_zip，請選擇其中一個。")
    elif args.data_zip:
        # 解壓 zip 文件到臨時目錄（與 train_VQVAE.py 相同的方式）
        if not os.path.exists(args.data_zip):
            raise FileNotFoundError(f"壓縮檔不存在：{args.data_zip}")
        if not zipfile.is_zipfile(args.data_zip):
            raise ValueError(f"不是有效的 zip 文件：{args.data_zip}")
        
        temp_dir = tempfile.TemporaryDirectory(prefix="train_transformer_zip_")
        extract_dir = temp_dir.name
        temp_dir_holder.append(temp_dir)
        
        console.print(f"[cyan]正在解壓縮檔：{args.data_zip}[/cyan]")
        console.print(f"[dim]臨時解壓目錄：{extract_dir}[/dim]")
        
        with zipfile.ZipFile(args.data_zip, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            console.print(f"[green]✓[/green] 已解壓 {len(zip_ref.namelist())} 個項目")
        
        # 驗證 train/val/test 結構
        train_dir = os.path.join(extract_dir, "train")
        val_dir = os.path.join(extract_dir, "val")
        test_dir = os.path.join(extract_dir, "test")
        
        if not os.path.exists(train_dir):
            # 嘗試尋找 train/val/test 目錄（可能在不同層級）
            found_dirs = []
            for root, dirs, files in os.walk(extract_dir):
                if os.path.basename(root) in ["train", "val", "test"]:
                    found_dirs.append(root)
            if not found_dirs:
                raise ValueError("Zip 文件必須包含 train/val/test 子目錄")
            if len(found_dirs) >= 3:
                common_parent = os.path.commonpath(found_dirs)
                extract_dir = common_parent
                train_dir = os.path.join(extract_dir, "train")
                val_dir = os.path.join(extract_dir, "val")
                test_dir = os.path.join(extract_dir, "test")
        
        if not os.path.exists(train_dir):
            raise ValueError(f"train/ 未找到（檢查了 {train_dir}）")
        if not os.path.exists(val_dir):
            raise ValueError(f"val/ 未找到（檢查了 {val_dir}）")
        if not os.path.exists(test_dir):
            raise ValueError(f"test/ 未找到（檢查了 {test_dir}）")
        
        console.print(f"[green]✓[/green] 已驗證 train/val/test 結構")
        args.data_root = extract_dir
    elif args.data_root:
        # 使用提供的 data_root
        if not os.path.isdir(args.data_root):
            raise ValueError(f"--data_root 必須是目錄：{args.data_root}")
    else:
        raise ValueError("必須提供 --data_root 或 --data_zip。")

    # 裝置選擇：CUDA > MPS > CPU（可用時），可用 --cpu 強制 CPU
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"[bold]Device:[/bold] {device}")

    # AMP（僅 CUDA 時預設啟用，可用 --no_amp 停用）
    use_amp = (device.type == "cuda") and (not args.no_amp)
    if device.type == "cuda":
        console.print(f"[bold]AMP:[/bold] {'ENABLED' if use_amp else 'DISABLED'}")
    else:
        console.print("[bold]AMP:[/bold] N/A")

    # 從 data_root 讀取文件列表（已解壓或原本就是目錄）
    train_files = sorted(glob(os.path.join(args.data_root, "train", "*.npy")))
    val_files = sorted(glob(os.path.join(args.data_root, "val", "*.npy")))
    test_files = sorted(glob(os.path.join(args.data_root, "test", "*.npy")))

    print(f"Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # ------------------------------------------------
    # 自動推斷序列長度和 latent grid 大小
    # ------------------------------------------------
    seq_len = None
    latent_grid_size = None
    if len(train_files) == 0:
        raise ValueError("train/ 資料夾為空，無法推斷序列長度。")
    
    # 從 train/ 的第一個檔案推斷序列長度
    probe = np.load(train_files[0], mmap_mode="r")
    # Handle both 1D and 2D arrays
    if probe.ndim == 2 and probe.shape[1] == 1:
        probe = probe.squeeze(1)
    if probe.ndim != 1:
        raise ValueError(f"檔案形狀不正確（期望 1 維）：{train_files[0]} 取得 {probe.shape}")
    
    seq_len = int(probe.shape[0])
    print(f"[Info] 自動推斷序列長度 = {seq_len} 來自 {os.path.basename(train_files[0])}")
    
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
    
    # 基本一致性檢查：各 split 的第一個檔案（若存在）需有相同長度
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if len(files) == 0:
            continue
        probe2 = np.load(files[0], mmap_mode="r")
        if probe2.ndim == 2 and probe2.shape[1] == 1:
            probe2 = probe2.squeeze(1)
        if probe2.ndim != 1 or probe2.shape[0] != seq_len:
            raise ValueError(
                f"{split_name}/ 形狀不一致或不為 [{seq_len}]：" +
                f"{files[0]} 取得 {probe2.shape}"
            )
    
    # 檢查 token indices 是否在有效範圍內 [0, codebook_size)
    max_idx = probe.max()
    min_idx = probe.min()
    if min_idx < 0 or max_idx >= args.codebook_size:
        raise ValueError(
            f"Token indices 超出範圍：min={min_idx}, max={max_idx}, "
            f"codebook_size={args.codebook_size}。"
            f"所有 indices 應在 [0, {args.codebook_size}) 範圍內。"
        )
    print(f"[Info] Token indices 範圍：[{min_idx}, {max_idx}], codebook_size={args.codebook_size}")

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
    # Note: max_seq needs to be seq_len + 1 because we prepend BOS token during training
    # Input: [BOS, z[0], z[1], ..., z[511]] has length 513 when seq_len=512
    # For sampling, we still use seq_len (not max_seq_len) because we start with BOS and sample seq_len tokens
    max_seq_len = (seq_len + 1) if seq_len is not None else 513
    actual_seq_len = seq_len if seq_len is not None else 512  # For sampling, use original seq_len
    model = TransformerPrior(
        num_tokens=args.codebook_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_seq=max_seq_len,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    # Use correct API: device parameter should be a string like "cuda" or "cpu"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp) if use_amp else None

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
    
    # Resolve vqvae_ckpt (support both --vqvae_ckpt and --ckpt)
    vqvae_ckpt_path = args.vqvae_ckpt or args.ckpt
    
    # Create samples directory (only if VQ-VAE is available)
    samples_dir = None
    if vqvae_ckpt_path:
        samples_dir = os.path.join(exp_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

    # 可選：載入完整的 VQ-VAE 模型（包含 decoder）以產生投影圖與 occupancy ratio
    def _load_vqvae_model(ckpt_path: str, model_def_path: str | None, codebook_size: int, latent_dim: int | None, device):
        """
        載入完整的 VQ-VAE 模型（包含 encoder, quantizer, decoder）。
        
        Returns:
            vqvae_model: 完整的 VQVAE3D 模型，或 None 如果載入失敗
        """
        if ckpt_path is None:
            return None
        
        # 解析模型定義檔
        if model_def_path is None:
            # 預設與此腳本同層的 VQVAE 版本
            default_path = Path(__file__).parent / "train_VQVAE.py"
            model_def_path = str(default_path)
        
        spec = importlib.util.spec_from_file_location("vqvae_def_mod", model_def_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"無法載入 VQ-VAE 模型定義：{model_def_path}")
        vqvae_mod = importlib.util.module_from_spec(spec)
        sys.modules["vqvae_def_mod"] = vqvae_mod
        spec.loader.exec_module(vqvae_mod)
        
        # Try to find VQ-VAE class
        vqvae_class = None
        for name in ["VQVAE3D", "VQVAE", "VQUNetVAE"]:
            if hasattr(vqvae_mod, name):
                vqvae_class = getattr(vqvae_mod, name)
                break
        if vqvae_class is None:
            raise RuntimeError(f"VQ-VAE 定義檔未提供 VQ-VAE 類別（尋找 VQVAE3D, VQVAE, VQUNetVAE）")
        
        # Load checkpoint
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Get model state dict
        state = ckpt["model"] if "model" in ckpt else ckpt
        
        # Try to infer model parameters from checkpoint args or use defaults
        ckpt_args = ckpt.get("args", {})
        # Default values (matching train_VQVAE.py defaults)
        base = ckpt_args.get("base", 64)
        latent_dim_actual = latent_dim or ckpt_args.get("latent_dim", 64)
        codebook_size_actual = ckpt_args.get("codebook_size", codebook_size)
        
        # Create model
        try:
            vqvae_model = vqvae_class(
                in_ch=3,
                out_ch=3,
                base=base,
                latent_dim=latent_dim_actual,
                codebook_size=codebook_size_actual,
                commitment_cost=0.25,  # Fixed in train_VQVAE.py
            ).to(device)
            
            # Load state dict
            vqvae_model.load_state_dict(state, strict=False)
            vqvae_model.eval()
            
            console.print(f"[green]✓[/green] Loaded VQ-VAE model from {ckpt_path}")
            console.print(f"[dim]  Model params: base={base}, latent_dim={latent_dim_actual}, codebook_size={codebook_size_actual}[/dim]")
            
            return vqvae_model
        except Exception as e:
            raise RuntimeError(f"載入 VQ-VAE 模型時出錯：{e}")

    def decode_indices_to_voxels(vqvae_model, indices_tensor, device):
        """
        安全地將 token indices 解碼為 voxel logits，兼容不同的 VQ-VAE 實現。
        
        Args:
            vqvae_model: VQ-VAE 模型
            indices_tensor: token indices, shape [B, D, H, W] 或 [D, H, W]
            device: device to run on
        
        Returns:
            logits: voxel logits, shape [B, 3, 32, 32, 32]
        """
        # Ensure batch dimension
        if indices_tensor.dim() == 3:
            indices_tensor = indices_tensor.unsqueeze(0)  # [1, D, H, W]
        
        B, D, H, W = indices_tensor.shape
        
        # Try method 1: use decode_from_indices if available (standard VQVAE3D API)
        if hasattr(vqvae_model, 'decode_from_indices'):
            try:
                logits = vqvae_model.decode_from_indices(indices_tensor)
                return logits
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] decode_from_indices failed: {e}, trying fallback method")
        
        # Fallback method 2: manual embedding lookup + decoder
        # This handles VQ-VAE implementations that don't have decode_from_indices
        try:
            # Get quantizer (could be vq, quantizer, or vector_quantizer)
            quantizer = None
            for attr_name in ['vq', 'quantizer', 'vector_quantizer']:
                if hasattr(vqvae_model, attr_name):
                    quantizer = getattr(vqvae_model, attr_name)
                    break
            
            if quantizer is None:
                raise RuntimeError("Cannot find quantizer in VQ-VAE model")
            
            # Get embedding layer
            if not hasattr(quantizer, 'embedding'):
                raise RuntimeError("Quantizer does not have embedding attribute")
            
            # Embedding lookup: indices -> quantized latent
            flat_idx = indices_tensor.view(-1)  # [B*D*H*W]
            z_q_flat = quantizer.embedding(flat_idx)  # [B*D*H*W, C]
            
            # Reshape to [B, C, D, H, W] format expected by decoder
            C = z_q_flat.shape[1]
            z_q = z_q_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
            
            # Decode using decoder
            decoder = None
            for attr_name in ['decoder', 'dec']:
                if hasattr(vqvae_model, attr_name):
                    decoder = getattr(vqvae_model, attr_name)
                    break
            
            if decoder is None:
                raise RuntimeError("Cannot find decoder in VQ-VAE model")
            
            logits = decoder(z_q)  # [B, 3, 32, 32, 32]
            return logits
            
        except Exception as e:
            raise RuntimeError(f"Failed to decode indices using fallback method: {e}")

    vqvae_model = None
    if vqvae_ckpt_path:
        try:
            # Try to infer latent_dim from checkpoint or use a reasonable default
            # We'll try to load and let it fail if incompatible
            vqvae_model = _load_vqvae_model(
                vqvae_ckpt_path, 
                args.vqvae_model_def, 
                args.codebook_size,
                latent_dim=None,  # Will be inferred from checkpoint
                device=device
            )
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] 無法載入 VQ-VAE 模型：{e}")
            console.print(f"[yellow]  Decode quality metrics (occupancy ratio, projection views) 將被跳過[/yellow]")
            vqvae_model = None

    # 記錄
    training_history = []  # list of dicts
    start_time_dt = datetime.now()
    start_time = time.time()
    # Note: For discrete token indices, we don't compute mean/std statistics
    # Analytics will use token frequency distributions instead

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
        "codebook_size": args.codebook_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "loss_function": "CrossEntropyLoss - discrete token prediction",
        "loss_formula": "logits = model(x); loss = CrossEntropy(logits.reshape(-1, num_tokens), y_target.reshape(-1))",
        "sample_temperature": args.sample_temperature,
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
        "analytics_n_samples": args.analytics_n_samples,
        "n_sample_images": args.n_sample_images,
        "vqvae_ckpt": vqvae_ckpt_path if vqvae_ckpt_path else "None",
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
                "transition_kl",
                "token_coverage",
                "sample_diversity",
                "transition_entropy_diff",
                "mmd_hamming",
                "coverage_hamming",
                "mmd_cosine",
                "coverage_cosine",
                "epoch_time_secs",
                "cumulative_time_secs",
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
            ce_loss = nn.CrossEntropyLoss()
            for z in train_loader:
                z = z.to(device, non_blocking=True)  # [B, 512] - token indices
                B = z.size(0)
                
                # Insert BOS token at the beginning: [BOS, z[0], z[1], ..., z[511]]
                # This ensures the model learns what BOS represents during training
                bos = torch.full((B, 1), model.bos_token_id, dtype=torch.long, device=device)
                x = torch.cat([bos, z], dim=1)  # [B, 513]: [BOS, z[0], z[1], ..., z[511]]
                
                # Target: [z[0], z[1], ..., z[511]] (does not include BOS)
                y_target = z  # [B, 512]

                opt.zero_grad(set_to_none=True)
                # 只在 CUDA 裝置上使用 AMP
                use_amp_here = use_amp and (device.type == "cuda") and (scaler is not None)
                if use_amp_here:
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        logits = model(x)  # [B, 513, num_tokens]
                        # Remove the last position prediction (we don't predict after the last token)
                        logits = logits[:, :-1, :]  # [B, 512, num_tokens]
                        loss = ce_loss(
                            logits.reshape(-1, model.num_tokens),
                            y_target.reshape(-1)
                        )
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(x)  # [B, 513, num_tokens]
                    # Remove the last position prediction (we don't predict after the last token)
                    logits = logits[:, :-1, :]  # [B, 512, num_tokens]
                    loss = ce_loss(
                        logits.reshape(-1, model.num_tokens),
                        y_target.reshape(-1)
                    )
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
            val_loss = eval_one_epoch(model, val_loader, device, use_amp=use_amp)
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

            # Defer console print and history write until after analytics (so transition drift 可顯示在 val 旁邊)
            current_kl = None
            current_token_coverage = None
            current_sample_diversity = None
            current_transition_entropy_diff = None
            current_mmd_hamming = None
            current_coverage_hamming = None
            current_mmd_cosine = None
            current_coverage_cosine = None

            progress.update(overall, advance=1)

            # -------------------------
            # Analytics every N epochs
            # -------------------------
            if (epoch % args.analytics_every == 0) or (epoch == args.epochs):
                ep_dir = os.path.join(analytics_dir, f"epoch_{epoch:04d}")
                os.makedirs(ep_dir, exist_ok=True)

                # ========== 統一收集訓練序列和生成樣本（避免重複） ==========
                # Helper function: convert tensor/array to numpy array
                def to_numpy(z):
                    """Convert tensor or array to numpy array."""
                    return z.numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
                
                # Collect train sequences once (used by multiple analytics)
                # Use max(100, auto_corr_eval_size, analytics_n_samples) to cover all needs
                auto_corr_eval_size = min(500, len(train_ds))
                analytics_train_size = min(args.analytics_n_samples, len(train_ds))
                max_collect_size = max(100, auto_corr_eval_size, analytics_train_size)
                train_sequences_collected = []
                if max_collect_size > 100:  # Only show progress bar if collecting many samples
                    collect_task = progress.add_task(
                        f"[cyan]Collecting {max_collect_size} train sequences", total=max_collect_size
                    )
                for z in train_ds:
                    train_sequences_collected.append(to_numpy(z))
                    if max_collect_size > 100:
                        progress.update(collect_task, advance=1)
                    if len(train_sequences_collected) >= max_collect_size:
                        break
                if max_collect_size > 100:
                    progress.remove_task(collect_task)
                
                # Generate samples once (used by multiple analytics)
                # Generate enough samples for all analytics (max of all needs)
                # MMD/Coverage and decode quality need analytics_n_samples samples
                num_gen_samples = max(5, 5, 4, args.analytics_n_samples)  # entropy(5), diversity(5), drift(4), mmd_coverage/decode_quality(N)
                gen_samples_all = []
                if num_gen_samples > 10:  # Only show progress bar if generating many samples
                    gen_task = progress.add_task(
                        f"[cyan]Generating {num_gen_samples} latent samples", total=num_gen_samples
                    )
                for i in range(num_gen_samples):
                    s = sample_latent(
                        model, 
                        seq_len=actual_seq_len, 
                        device=device, 
                        mode=args.sample_mode,
                        temperature=args.sample_temperature
                    )
                    gen_samples_all.append(s)
                    if num_gen_samples > 10:
                        progress.update(gen_task, advance=1)
                if num_gen_samples > 10:
                    progress.remove_task(gen_task)
                # ============================================================

                # (1) Token Transition Entropy: 評估 Prior 是否在學語法，不是純 token 頻率
                # 目的：Prior 是否在學語法（例如：樹葉後面不能接樹根 token）
                # entropy 高：轉換無章法
                # entropy 低：collapsed
                # entropy 適中：模型正在學規則
                
                # Shared helper function: compute transition count matrix (vectorized)
                def compute_transition_counts(sequences, codebook_size):
                    """
                    計算 transition count matrix（向量化實現，重用邏輯避免重複代碼）。
                    
                    Args:
                        sequences: list of sequences, each is array-like of token indices
                        codebook_size: size of codebook
                    
                    Returns:
                        counts: transition count matrix [codebook_size, codebook_size]
                    """
                    all_from = []
                    all_to = []
                    
                    for seq in sequences:
                        seq = np.asarray(seq, dtype=np.int64)
                        if len(seq) < 2:
                            continue
                        # Extract (from_token, to_token) pairs
                        from_tokens = seq[:-1]
                        to_tokens = seq[1:]
                        # Filter valid indices
                        valid_mask = (from_tokens >= 0) & (from_tokens < codebook_size) & \
                                     (to_tokens >= 0) & (to_tokens < codebook_size)
                        all_from.append(from_tokens[valid_mask])
                        all_to.append(to_tokens[valid_mask])
                    
                    if len(all_from) == 0:
                        return np.zeros((codebook_size, codebook_size), dtype=np.float64)
                    
                    # Concatenate all pairs
                    all_from = np.concatenate(all_from)
                    all_to = np.concatenate(all_to)
                    
                    # Vectorized counting using np.add.at (much faster than for-loop)
                    counts = np.zeros((codebook_size, codebook_size), dtype=np.float64)
                    np.add.at(counts, (all_from, all_to), 1)
                    
                    return counts
                
                def compute_transition_entropy(sequences, codebook_size):
                    """
                    計算 P(token_t | token_t-1) 的 entropy。
                    使用向量化操作提升效率（避免 Python for-loop）。
                    重用 compute_transition_counts 避免重複代碼。
                    
                    Args:
                        sequences: list of sequences, each is array-like of token indices
                        codebook_size: size of codebook
                    
                    Returns:
                        mean_entropy: 平均 transition entropy
                        entropy: array of entropy for each token [codebook_size]
                    """
                    # Reuse shared transition counts computation
                    counts = compute_transition_counts(sequences, codebook_size)
                    
                    # Compute conditional probabilities P(b|a) = counts[a,b] / sum_b counts[a,b]
                    row_sums = counts.sum(axis=1, keepdims=True)
                    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
                    
                    # Compute entropy for each token: H(P(·|token)) = -sum_b P(b|token) * log(P(b|token))
                    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
                    
                    # Mean entropy (only for tokens that appear)
                    mean_entropy = entropy.mean()
                    
                    return mean_entropy, entropy
                
                # Compute transition entropy for training and generated data
                # Reuse collected sequences
                train_mean_entropy, train_token_entropy = compute_transition_entropy(
                    train_sequences_collected[:100], args.codebook_size
                )
                gen_mean_entropy, gen_token_entropy = compute_transition_entropy(
                    gen_samples_all[:5], args.codebook_size
                )
                
                # Save grammar strength metrics
                with open(os.path.join(ep_dir, "grammar_strength_metrics.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "value"])
                    w.writerow(["train_avg_transition_entropy", f"{train_mean_entropy:.6f}"])
                    w.writerow(["gen_avg_transition_entropy", f"{gen_mean_entropy:.6f}"])
                    w.writerow(["entropy_difference", f"{gen_mean_entropy - train_mean_entropy:.6f}"])
                    w.writerow(["n_train_sequences", len(train_sequences_collected[:100])])
                    w.writerow(["n_gen_sequences", 5])
                
                # Visualize per-token entropy (if codebook_size is not too large)
                if args.codebook_size <= 1024:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                    
                    # Plot train token entropy
                    axes[0].bar(range(args.codebook_size), train_token_entropy, alpha=0.7, color='blue', label='Train')
                    axes[0].axhline(y=train_mean_entropy, color='blue', linestyle='--', linewidth=2, label=f'Train Mean: {train_mean_entropy:.4f}')
                    axes[0].set_xlabel("Token Index")
                    axes[0].set_ylabel("Transition Entropy")
                    axes[0].set_title("Train: Per-Token Transition Entropy")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot generated token entropy
                    axes[1].bar(range(args.codebook_size), gen_token_entropy, alpha=0.7, color='red', label='Generated')
                    axes[1].axhline(y=gen_mean_entropy, color='red', linestyle='--', linewidth=2, label=f'Gen Mean: {gen_mean_entropy:.4f}')
                    axes[1].set_xlabel("Token Index")
                    axes[1].set_ylabel("Transition Entropy")
                    axes[1].set_title("Generated: Per-Token Transition Entropy")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_dir, "token_transition_entropy.png"), dpi=140)
                    plt.close(fig)
                else:
                    # For large codebook, just plot comparison of mean entropy
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    ax.bar(['Train', 'Generated'], [train_mean_entropy, gen_mean_entropy], color=['blue', 'red'], alpha=0.7)
                    ax.set_ylabel("Average Transition Entropy")
                    ax.set_title("Transition Entropy Comparison")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_dir, "transition_entropy_comparison.png"), dpi=140)
                    plt.close(fig)

                # (1b) Auto-correlation of Token Grid: 評估 Transformer prior 是否學到空間連續性
                # 目的：鄰近 latent token 是否相關（目前處理方式是 flat = 完全忽略 8 個鄰居）
                def grid_autocorrelation(z, grid_size):
                    """
                    計算 token grid 的空間自相關性。
                    
                    Args:
                        z: token sequence, shape [grid_size^3]
                        grid_size: spatial grid size (D)
                    
                    Returns:
                        mean_corr: 三個方向的平均 correlation
                        corrs: list of correlations for each direction [(1,0,0), (0,1,0), (0,0,1)]
                    """
                    if len(z) != grid_size ** 3:
                        return None, None
                    
                    # Reshape to [D, D, D]
                    z_grid = z.reshape(grid_size, grid_size, grid_size)
                    
                    shifts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                    corrs = []
                    
                    for dz, dy, dx in shifts:
                        # Shift by (dz, dy, dx) and compute correlation
                        z1 = z_grid[dz:, dy:, dx:].flatten()
                        z2 = z_grid[:-dz or None, :-dy or None, :-dx or None].flatten()
                        
                        # Compute correlation coefficient
                        if len(z1) > 0 and len(z2) > 0 and len(z1) == len(z2):
                            corr = np.corrcoef(z1, z2)[0, 1]
                            if not np.isnan(corr):
                                corrs.append(corr)
                    
                    mean_corr = float(np.mean(corrs)) if len(corrs) > 0 else 0.0
                    return mean_corr, corrs
                
                # Compute auto-correlation for training and generated data
                # Reuse collected sequences
                train_autocorr_mean = None
                train_autocorr_dirs = None
                train_autocorr_samples = []
                gen_autocorr_mean = None
                gen_autocorr_dirs = None
                gen_autocorr_samples = []
                
                if latent_grid_size is not None:
                    # Use collected sequences (up to auto_corr_eval_size for train)
                    for seq in train_sequences_collected[:auto_corr_eval_size]:
                        mean_corr, dir_corrs = grid_autocorrelation(seq, latent_grid_size)
                        if mean_corr is not None:
                            train_autocorr_samples.append(mean_corr)
                            if train_autocorr_dirs is None:
                                train_autocorr_dirs = dir_corrs
                    
                    if len(train_autocorr_samples) > 0:
                        train_autocorr_mean = np.mean(train_autocorr_samples)
                    
                    # Use generated samples
                    for seq in gen_samples_all[:5]:
                        mean_corr, dir_corrs = grid_autocorrelation(seq, latent_grid_size)
                        if mean_corr is not None:
                            gen_autocorr_samples.append(mean_corr)
                            if gen_autocorr_dirs is None:
                                gen_autocorr_dirs = dir_corrs
                    
                    if len(gen_autocorr_samples) > 0:
                        gen_autocorr_mean = np.mean(gen_autocorr_samples)
                
                # Save auto-correlation metrics
                with open(os.path.join(ep_dir, "grid_autocorrelation_metrics.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "value"])
                    if latent_grid_size is not None:
                        w.writerow(["grid_size", latent_grid_size])
                        w.writerow(["n_train_sequences_used", len(train_autocorr_samples)])
                        w.writerow(["auto_corr_eval_size", auto_corr_eval_size])
                        if train_autocorr_mean is not None:
                            w.writerow(["train_mean_autocorrelation", f"{train_autocorr_mean:.6f}"])
                            if train_autocorr_dirs is not None:
                                w.writerow(["train_dir_1_0_0", f"{train_autocorr_dirs[0]:.6f}"])
                                w.writerow(["train_dir_0_1_0", f"{train_autocorr_dirs[1]:.6f}"])
                                w.writerow(["train_dir_0_0_1", f"{train_autocorr_dirs[2]:.6f}"])
                        if gen_autocorr_mean is not None:
                            w.writerow(["gen_mean_autocorrelation", f"{gen_autocorr_mean:.6f}"])
                            if gen_autocorr_dirs is not None:
                                w.writerow(["gen_dir_1_0_0", f"{gen_autocorr_dirs[0]:.6f}"])
                                w.writerow(["gen_dir_0_1_0", f"{gen_autocorr_dirs[1]:.6f}"])
                                w.writerow(["gen_dir_0_0_1", f"{gen_autocorr_dirs[2]:.6f}"])
                        if train_autocorr_mean is not None and gen_autocorr_mean is not None:
                            w.writerow(["autocorr_difference", f"{gen_autocorr_mean - train_autocorr_mean:.6f}"])
                    else:
                        w.writerow(["grid_size", "unknown"])
                        w.writerow(["error", "Sequence length is not a perfect cube"])
                
                # Visualize auto-correlation comparison
                if latent_grid_size is not None and train_autocorr_mean is not None and gen_autocorr_mean is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Plot mean auto-correlation comparison
                    axes[0].bar(['Train', 'Generated'], [train_autocorr_mean, gen_autocorr_mean], 
                               color=['blue', 'red'], alpha=0.7)
                    axes[0].set_ylabel("Mean Auto-correlation")
                    axes[0].set_title("Token Grid Auto-correlation Comparison")
                    axes[0].set_ylim([-1, 1])
                    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot directional auto-correlation
                    if train_autocorr_dirs is not None and gen_autocorr_dirs is not None:
                        directions = ['(1,0,0)', '(0,1,0)', '(0,0,1)']
                        x = np.arange(len(directions))
                        width = 0.35
                        axes[1].bar(x - width/2, train_autocorr_dirs, width, label='Train', color='blue', alpha=0.7)
                        axes[1].bar(x + width/2, gen_autocorr_dirs, width, label='Generated', color='red', alpha=0.7)
                        axes[1].set_ylabel("Correlation")
                        axes[1].set_title("Directional Auto-correlation")
                        axes[1].set_xticks(x)
                        axes[1].set_xticklabels(directions)
                        axes[1].set_ylim([-1, 1])
                        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_dir, "grid_autocorrelation.png"), dpi=140)
                    plt.close(fig)

                # (2) Sample Diversity
                # Reuse generated samples
                K = 5
                multi_samples = gen_samples_all[:K]
                
                # [S1] Sample Diversity: Hamming distance (fraction of different tokens)
                hamming_dists = []
                for i in range(K):
                    for j in range(i + 1, K):
                        # Hamming distance: fraction of positions where tokens differ
                        hamming = (multi_samples[i] != multi_samples[j]).mean()
                        hamming_dists.append(hamming)
                sample_diversity = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
                
                # [S2] Unique tokens across all samples
                all_tokens = np.concatenate(multi_samples)
                unique_tokens = len(np.unique(all_tokens))
                token_coverage = unique_tokens / args.codebook_size
                
                # [S3] Token Sequence Visualization: show token indices over sequence
                fig, axes = plt.subplots(K, 1, figsize=(12, 2*K))
                if K == 1:
                    axes = [axes]
                for k in range(K):
                    axes[k].plot(multi_samples[k][:200], marker='o', markersize=1, linewidth=0.5)  # Show first 200 tokens
                    axes[k].set_title(f"Sample {k+1} - Token Sequence (first 200 tokens)")
                    axes[k].set_xlabel("Position")
                    axes[k].set_ylabel("Token Index")
                    axes[k].set_ylim(0, args.codebook_size)
                plt.tight_layout()
                plt.savefig(os.path.join(ep_dir, "token_sequences.png"), dpi=140)
                plt.close()
                
                # Save sample diversity to CSV for easy tracking
                with open(os.path.join(ep_dir, "sample_diversity.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "value"])
                    w.writerow(["sample_diversity", f"{sample_diversity:.6f}"])
                    w.writerow(["unique_tokens", unique_tokens])
                    w.writerow(["token_coverage", f"{token_coverage:.6f}"])
                    w.writerow(["n_samples", K])

                # (4) Decode Quality: Distribution Comparison (Train vs Generated)
                # Decode multiple samples and compare distributions
                if vqvae_model is not None:
                    grid_size_to_use = latent_grid_size if latent_grid_size is not None else 8
                    
                    # Decode train samples (use same number as analytics_n_samples)
                    n_decode_samples = min(args.analytics_n_samples, len(train_sequences_collected), len(gen_samples_all))
                    train_decode_samples = train_sequences_collected[:n_decode_samples]
                    gen_decode_samples = gen_samples_all[:n_decode_samples]
                    
                    train_non_air_ratios = []
                    train_bbox_ratios = []
                    train_largest_component_ratios = []
                    gen_non_air_ratios = []
                    gen_bbox_ratios = []
                    gen_largest_component_ratios = []
                    
                    decode_task = progress.add_task(
                        f"[cyan]Decoding {n_decode_samples * 2} samples for distribution comparison", 
                        total=n_decode_samples * 2
                    )
                    
                    with torch.no_grad():
                        # Decode train samples
                        for train_seq in train_decode_samples:
                            if len(train_seq) != grid_size_to_use ** 3:
                                progress.update(decode_task, advance=1)
                                continue
                            
                            indices_grid = train_seq.reshape(grid_size_to_use, grid_size_to_use, grid_size_to_use)
                            indices_tensor = torch.from_numpy(indices_grid).long().to(device)
                            
                            logits = decode_indices_to_voxels(vqvae_model, indices_tensor, device)
                            logits_np = logits[0].detach().cpu().numpy()
                            labels = np.argmax(logits_np, axis=0).astype(np.uint8)
                            
                            non_air_ratio, bbox_ratio = compute_decoder_occupancy(labels, air_class=0)
                            largest_component_ratio = compute_largest_component_ratio(labels, air_class=0)
                            
                            train_non_air_ratios.append(non_air_ratio)
                            train_bbox_ratios.append(bbox_ratio)
                            train_largest_component_ratios.append(largest_component_ratio)
                            progress.update(decode_task, advance=1)
                        
                        # Decode generated samples
                        for gen_seq in gen_decode_samples:
                            if len(gen_seq) != grid_size_to_use ** 3:
                                progress.update(decode_task, advance=1)
                                continue
                            
                            indices_grid = gen_seq.reshape(grid_size_to_use, grid_size_to_use, grid_size_to_use)
                            indices_tensor = torch.from_numpy(indices_grid).long().to(device)
                            
                            logits = decode_indices_to_voxels(vqvae_model, indices_tensor, device)
                            logits_np = logits[0].detach().cpu().numpy()
                            labels = np.argmax(logits_np, axis=0).astype(np.uint8)
                            
                            non_air_ratio, bbox_ratio = compute_decoder_occupancy(labels, air_class=0)
                            largest_component_ratio = compute_largest_component_ratio(labels, air_class=0)
                            
                            gen_non_air_ratios.append(non_air_ratio)
                            gen_bbox_ratios.append(bbox_ratio)
                            gen_largest_component_ratios.append(largest_component_ratio)
                            progress.update(decode_task, advance=1)
                    
                    progress.remove_task(decode_task)
                    
                    # Convert to numpy arrays for statistics
                    train_non_air_ratios = np.array(train_non_air_ratios)
                    train_bbox_ratios = np.array(train_bbox_ratios)
                    train_largest_component_ratios = np.array(train_largest_component_ratios)
                    gen_non_air_ratios = np.array(gen_non_air_ratios)
                    gen_bbox_ratios = np.array(gen_bbox_ratios)
                    gen_largest_component_ratios = np.array(gen_largest_component_ratios)
                    
                    # Save CSV
                    with open(os.path.join(ep_dir, "decoder_occupancy_distribution.csv"), "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["metric", "train_mean", "train_std", "gen_mean", "gen_std", "n_samples"])
                        w.writerow(["non_air_ratio", f"{train_non_air_ratios.mean():.6f}", f"{train_non_air_ratios.std():.6f}",
                                   f"{gen_non_air_ratios.mean():.6f}", f"{gen_non_air_ratios.std():.6f}", n_decode_samples])
                        w.writerow(["bbox_volume_ratio", f"{train_bbox_ratios.mean():.6f}", f"{train_bbox_ratios.std():.6f}",
                                   f"{gen_bbox_ratios.mean():.6f}", f"{gen_bbox_ratios.std():.6f}", n_decode_samples])
                        w.writerow(["largest_component_ratio", f"{train_largest_component_ratios.mean():.6f}", f"{train_largest_component_ratios.std():.6f}",
                                   f"{gen_largest_component_ratios.mean():.6f}", f"{gen_largest_component_ratios.std():.6f}", n_decode_samples])
                    
                    # Visualize non_air_ratio distribution (histogram: train vs gen)
                    if len(train_non_air_ratios) > 0 and len(gen_non_air_ratios) > 0:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Histogram of non_air_ratio
                        axes[0].hist(train_non_air_ratios, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
                        axes[0].hist(gen_non_air_ratios, bins=50, alpha=0.7, label='Generated', color='red', edgecolor='black')
                        axes[0].axvline(x=train_non_air_ratios.mean(), color='blue', linestyle='--', linewidth=2, label=f'Train Mean: {train_non_air_ratios.mean():.4f}')
                        axes[0].axvline(x=gen_non_air_ratios.mean(), color='red', linestyle='--', linewidth=2, label=f'Gen Mean: {gen_non_air_ratios.mean():.4f}')
                        axes[0].set_xlabel("non_air_ratio")
                        axes[0].set_ylabel("Frequency")
                        axes[0].set_title("non_air_ratio Distribution (Train vs Generated)")
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        # Box plot comparison
                        axes[1].boxplot([train_non_air_ratios, gen_non_air_ratios], labels=['Train', 'Generated'])
                        axes[1].set_ylabel("non_air_ratio")
                        axes[1].set_title("non_air_ratio Box Plot Comparison")
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(ep_dir, "non_air_ratio_distribution.png"), dpi=140)
                        plt.close(fig)

                    console.print(f"[dim]Decode quality distribution: train non_air={train_non_air_ratios.mean():.4f}±{train_non_air_ratios.std():.4f}, gen={gen_non_air_ratios.mean():.4f}±{gen_non_air_ratios.std():.4f}[/dim]")

                # (5) Token Usage: Perplexity-like Metrics
                # Compute perplexity-like metrics to measure grammar token variety
                # Reuse collected sequences
                def compute_perplexity_like_metrics(sequences, codebook_size):
                    """Compute perplexity-like metrics for token usage."""
                    # Collect all tokens
                    all_tokens = np.concatenate(sequences) if len(sequences) > 0 else np.array([], dtype=np.int64)
                    if len(all_tokens) == 0:
                        return {"perplexity": 0.0, "effective_vocab_size": 0.0, "token_entropy": 0.0}
                    
                    # Token frequency distribution
                    token_counts = np.bincount(all_tokens, minlength=codebook_size).astype(np.float32)
                    token_probs = token_counts / token_counts.sum() if token_counts.sum() > 0 else token_counts
                    token_probs = np.maximum(token_probs, 1e-8)  # avoid log(0)
                    
                    # Token entropy
                    token_entropy = -np.sum(token_probs * np.log(token_probs))
                    
                    # Perplexity = exp(entropy)
                    perplexity = np.exp(token_entropy)
                    
                    # Effective vocabulary size (number of tokens with prob > threshold)
                    threshold = 1.0 / len(all_tokens)  # tokens that appear at least once
                    effective_vocab_size = (token_probs > threshold).sum()
                    
                    return {
                        "perplexity": float(perplexity),
                        "effective_vocab_size": float(effective_vocab_size),
                        "token_entropy": float(token_entropy),
                    }
                
                gen_ppl_metrics = compute_perplexity_like_metrics(multi_samples, args.codebook_size)
                train_ppl_metrics = compute_perplexity_like_metrics(train_sequences_collected[:100], args.codebook_size)
                
                # Save perplexity metrics
                with open(os.path.join(ep_dir, "token_usage_metrics.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "train", "generated"])
                    w.writerow(["perplexity", f"{train_ppl_metrics['perplexity']:.6f}", f"{gen_ppl_metrics['perplexity']:.6f}"])
                    w.writerow(["effective_vocab_size", f"{train_ppl_metrics['effective_vocab_size']:.0f}", f"{gen_ppl_metrics['effective_vocab_size']:.0f}"])
                    w.writerow(["token_entropy", f"{train_ppl_metrics['token_entropy']:.6f}", f"{gen_ppl_metrics['token_entropy']:.6f}"])
                
                # (6) 若提供 VQ-VAE 模型：保存 token indices 樣本並解碼為 voxels
                if vqvae_model is not None and samples_dir is not None:
                    n_samples = args.n_sample_images
                    grid_size_to_use = latent_grid_size if latent_grid_size is not None else 8
                    for sample_idx in range(n_samples):
                        with torch.no_grad():
                            # Sample a new latent sequence (token indices)
                            z_sample_new = sample_latent(
                                model, 
                                seq_len=actual_seq_len, 
                                device=device, 
                                mode=args.sample_mode,
                                temperature=args.sample_temperature
                            )  # [seq_len] - token indices
                            
                            # Save token indices as .npy file
                            sample_npy_name = f"sample_e{epoch}_{sample_idx + 1}_{args.exp_name}.npy"
                            sample_npy_path = os.path.join(samples_dir, sample_npy_name)
                            np.save(sample_npy_path, z_sample_new)
                            
                            # Decode to voxels if sequence length matches grid
                            if len(z_sample_new) == grid_size_to_use ** 3:
                                # Reshape to [8,8,8]
                                indices_grid = z_sample_new.reshape(grid_size_to_use, grid_size_to_use, grid_size_to_use)
                                indices_tensor = torch.from_numpy(indices_grid).long().to(device)  # [8, 8, 8]
                                
                                # Decode using VQ-VAE (compatible with different VQ-VAE implementations)
                                logits = decode_indices_to_voxels(vqvae_model, indices_tensor, device)  # [1, 3, 32, 32, 32]
                                logits_np = logits[0].detach().cpu().numpy()  # [3, 32, 32, 32]
                                labels = np.argmax(logits_np, axis=0).astype(np.uint8)  # [32, 32, 32]
                                
                                # Save decoded voxel as .npz
                                sample_npz_name = f"sample_e{epoch}_{sample_idx + 1}_{args.exp_name}.npz"
                                sample_npz_path = os.path.join(samples_dir, sample_npz_name)
                                np.savez_compressed(sample_npz_path, labels)
                                
                                # Create 3-view projection
                                max_z = labels.max(axis=0)
                                max_y = labels.max(axis=1)
                                max_x = labels.max(axis=2)
                                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                                axes[0].imshow(max_z, cmap="viridis")
                                axes[0].set_title("MaxProj Z (Y,X)")
                                axes[1].imshow(max_y, cmap="viridis")
                                axes[1].set_title("MaxProj Y (Z,X)")
                                axes[2].imshow(max_x, cmap="viridis")
                                axes[2].set_title("MaxProj X (Z,Y)")
                                for ax in axes:
                                    ax.axis("off")
                                fig.tight_layout()
                                
                                # Save PNG file
                                sample_png_name = f"sample_e{epoch}_{sample_idx + 1}_{args.exp_name}.png"
                                sample_png_path = os.path.join(samples_dir, sample_png_name)
                                plt.savefig(sample_png_path, dpi=140)
                                plt.close(fig)

                                console.print(f"[dim]Saved sample {sample_idx + 1}: token indices + decoded voxel[/dim]")
                            else:
                                console.print(f"[dim]Saved token indices to {sample_npy_name} (sequence length mismatch, skipping decode)[/dim]")

                # (7) Transition-based Distribution Drift: Compare train vs generated transition patterns
                # This is the correct metric for categorical tokens (not KL/MMD which are for continuous distributions)
                
                def compute_transition_drift(train_transitions, gen_transitions, eps=1e-8):
                    """Compute transition-based drift metrics."""
                    # Normalize transition matrices to probabilities
                    train_total = train_transitions.sum()
                    gen_total = gen_transitions.sum()
                    if train_total == 0 or gen_total == 0:
                        return {"transition_kl": 0.0, "transition_l1": 0.0, "transition_js": 0.0}
                    
                    train_probs = train_transitions.astype(np.float32) / train_total
                    gen_probs = gen_transitions.astype(np.float32) / gen_total
                    
                    # KL divergence on transition probabilities
                    train_probs = np.maximum(train_probs, eps)
                    gen_probs = np.maximum(gen_probs, eps)
                    transition_kl = float(np.sum(train_probs * np.log(train_probs / gen_probs)))
                    
                    # L1 distance
                    transition_l1 = float(np.sum(np.abs(train_probs - gen_probs)))
                    
                    # JS divergence
                    m = 0.5 * (train_probs + gen_probs)
                    m = np.maximum(m, eps)
                    js = 0.5 * np.sum(train_probs * np.log(train_probs / m)) + 0.5 * np.sum(gen_probs * np.log(gen_probs / m))
                    transition_js = float(js)
                    
                    return {
                        "transition_kl": transition_kl,
                        "transition_l1": transition_l1,
                        "transition_js": transition_js,
                    }
                
                # Compute transition matrices
                # Reuse collected sequences and shared compute_transition_counts function
                train_transition = compute_transition_counts(train_sequences_collected[:100], args.codebook_size)
                gen_transition = compute_transition_counts(gen_samples_all[:4], args.codebook_size)
                
                # Compute transition-based drift
                drift_metrics = compute_transition_drift(train_transition, gen_transition)
                
                # Save transition drift metrics
                drift_csv = os.path.join(ep_dir, "transition_drift_metrics.csv")
                with open(drift_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "value"])
                    w.writerow(["transition_kl", f"{drift_metrics['transition_kl']:.6f}"])
                    w.writerow(["transition_l1", f"{drift_metrics['transition_l1']:.6f}"])
                    w.writerow(["transition_js", f"{drift_metrics['transition_js']:.6f}"])
                console.print(f"[dim]Analytics saved to {ep_dir}[/dim]")
                current_kl = drift_metrics['transition_kl']
                
                # (8) MMD and Coverage: Distance-based metrics between train and generated latents
                # Collect larger sets for MMD/Coverage computation
                n_train_samples = min(args.analytics_n_samples, len(train_sequences_collected))
                n_gen_samples = min(args.analytics_n_samples, len(gen_samples_all))
                
                train_samples_mmd = train_sequences_collected[:n_train_samples]
                gen_samples_mmd = gen_samples_all[:n_gen_samples]
                
                def compute_hamming_distance(seq1, seq2):
                    """Compute normalized Hamming distance between two token sequences."""
                    seq1 = np.asarray(seq1, dtype=np.int64)
                    seq2 = np.asarray(seq2, dtype=np.int64)
                    if len(seq1) != len(seq2):
                        # Pad or truncate to same length
                        min_len = min(len(seq1), len(seq2))
                        seq1 = seq1[:min_len]
                        seq2 = seq2[:min_len]
                    # Normalized Hamming distance: fraction of positions that differ
                    return float(np.mean(seq1 != seq2))
                
                def compute_cosine_distance_onehot(seq1, seq2, codebook_size):
                    """Compute cosine distance using one-hot encoding."""
                    seq1 = np.asarray(seq1, dtype=np.int64)
                    seq2 = np.asarray(seq2, dtype=np.int64)
                    
                    # Create one-hot encodings
                    onehot1 = np.zeros((len(seq1), codebook_size), dtype=np.float32)
                    onehot2 = np.zeros((len(seq2), codebook_size), dtype=np.float32)
                    
                    # Set valid indices
                    valid1 = (seq1 >= 0) & (seq1 < codebook_size)
                    valid2 = (seq2 >= 0) & (seq2 < codebook_size)
                    onehot1[np.arange(len(seq1))[valid1], seq1[valid1]] = 1.0
                    onehot2[np.arange(len(seq2))[valid2], seq2[valid2]] = 1.0
                    
                    # Flatten and compute cosine distance
                    vec1 = onehot1.flatten()
                    vec2 = onehot2.flatten()
                    
                    # Normalize to unit vectors
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 == 0 or norm2 == 0:
                        return 1.0  # Maximum distance if one is zero vector
                    
                    vec1 = vec1 / norm1
                    vec2 = vec2 / norm2
                    
                    # Cosine distance = 1 - cosine similarity
                    cosine_sim = np.dot(vec1, vec2)
                    return float(1.0 - cosine_sim)
                
                def compute_mmd_coverage(train_samples, gen_samples, codebook_size, distance_type="hamming", epsilon=0.6, progress_obj=None, task_desc=None):
                    """
                    Compute MMD (mean minimum distance) and Coverage metrics.
                    
                    Args:
                        train_samples: list of training token sequences
                        gen_samples: list of generated token sequences
                        codebook_size: size of codebook
                        distance_type: "hamming" or "cosine_onehot"
                        epsilon: threshold for coverage (distance < epsilon means "covered")
                        progress_obj: optional Progress object for progress bar
                        task_desc: optional task description for progress bar
                    
                    Returns:
                        dict with mmd_mean, mmd_std, coverage_ratio, coverage_count
                    """
                    n_train = len(train_samples)
                    n_gen = len(gen_samples)
                    
                    if n_train == 0 or n_gen == 0:
                        return {
                            "mmd_mean": 0.0,
                            "mmd_std": 0.0,
                            "coverage_ratio": 0.0,
                            "coverage_count": 0,
                        }
                    
                    # Compute distance function
                    if distance_type == "hamming":
                        dist_fn = compute_hamming_distance
                    elif distance_type == "cosine_onehot":
                        dist_fn = lambda s1, s2: compute_cosine_distance_onehot(s1, s2, codebook_size)
                    else:
                        raise ValueError(f"Unknown distance_type: {distance_type}")
                    
                    # Add progress bar if requested and sample size is large
                    mmd_task = None
                    if progress_obj is not None and task_desc is not None and n_train > 50:
                        mmd_task = progress_obj.add_task(task_desc, total=n_train)
                    
                    # For each train sample, find minimum distance to any gen sample
                    min_distances = []
                    covered_count = 0
                    
                    for idx, train_seq in enumerate(train_samples):
                        min_dist = float('inf')
                        for gen_seq in gen_samples:
                            dist = dist_fn(train_seq, gen_seq)
                            min_dist = min(min_dist, dist)
                        
                        min_distances.append(min_dist)
                        if min_dist < epsilon:
                            covered_count += 1
                        
                        # Update progress
                        if mmd_task is not None:
                            progress_obj.update(mmd_task, advance=1)
                    
                    if mmd_task is not None:
                        progress_obj.remove_task(mmd_task)
                    
                    mmd_mean = float(np.mean(min_distances))
                    mmd_std = float(np.std(min_distances))
                    coverage_ratio = float(covered_count) / float(n_train) if n_train > 0 else 0.0
                    
                    return {
                        "mmd_mean": mmd_mean,
                        "mmd_std": mmd_std,
                        "coverage_ratio": coverage_ratio,
                        "coverage_count": covered_count,
                        "n_train": n_train,
                        "n_gen": n_gen,
                    }
                
                # Compute MMD and Coverage with Hamming distance
                mmd_coverage_hamming = compute_mmd_coverage(
                    train_samples_mmd, 
                    gen_samples_mmd, 
                    args.codebook_size, 
                    distance_type="hamming",
                    epsilon=0.6,
                    progress_obj=progress,
                    task_desc=f"[yellow]Computing MMD/Coverage (Hamming, {n_train_samples} train × {n_gen_samples} gen)"
                )
                
                # Compute MMD and Coverage with Cosine distance (one-hot)
                mmd_coverage_cosine = compute_mmd_coverage(
                    train_samples_mmd, 
                    gen_samples_mmd, 
                    args.codebook_size, 
                    distance_type="cosine_onehot",
                    epsilon=0.6,
                    progress_obj=progress,
                    task_desc=f"[yellow]Computing MMD/Coverage (Cosine, {n_train_samples} train × {n_gen_samples} gen)"
                )
                
                # Save MMD and Coverage metrics
                mmd_coverage_csv = os.path.join(ep_dir, "mmd_coverage_metrics.csv")
                with open(mmd_coverage_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "hamming", "cosine_onehot"])
                    w.writerow(["mmd_mean", f"{mmd_coverage_hamming['mmd_mean']:.6f}", f"{mmd_coverage_cosine['mmd_mean']:.6f}"])
                    w.writerow(["mmd_std", f"{mmd_coverage_hamming['mmd_std']:.6f}", f"{mmd_coverage_cosine['mmd_std']:.6f}"])
                    w.writerow(["coverage_ratio", f"{mmd_coverage_hamming['coverage_ratio']:.6f}", f"{mmd_coverage_cosine['coverage_ratio']:.6f}"])
                    w.writerow(["coverage_count", mmd_coverage_hamming['coverage_count'], mmd_coverage_cosine['coverage_count']])
                    w.writerow(["n_train", mmd_coverage_hamming['n_train'], mmd_coverage_cosine['n_train']])
                    w.writerow(["n_gen", mmd_coverage_hamming['n_gen'], mmd_coverage_cosine['n_gen']])
                    w.writerow(["epsilon", "0.6", "0.6"])
                
                # Visualize MMD distribution
                # Compute sample of pairwise distances for visualization (limit to 100 samples for efficiency)
                viz_train_size = min(100, n_train_samples)
                viz_gen_size = min(100, n_gen_samples)
                hamming_dists_all = []
                for train_seq in train_samples_mmd[:viz_train_size]:
                    min_dists = []
                    for gen_seq in gen_samples_mmd[:viz_gen_size]:
                        dist = compute_hamming_distance(train_seq, gen_seq)
                        min_dists.append(dist)
                    hamming_dists_all.extend(min_dists)
                
                if len(hamming_dists_all) > 0:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram of distances
                    axes[0].hist(hamming_dists_all, bins=50, alpha=0.7, color='blue', edgecolor='black')
                    axes[0].axvline(x=mmd_coverage_hamming['mmd_mean'], color='red', linestyle='--', linewidth=2, label=f"MMD Mean: {mmd_coverage_hamming['mmd_mean']:.4f}")
                    axes[0].axvline(x=0.6, color='green', linestyle='--', linewidth=2, label=f"Coverage Threshold: 0.6")
                    axes[0].set_xlabel("Hamming Distance")
                    axes[0].set_ylabel("Frequency")
                    axes[0].set_title(f"Distance Distribution (Hamming, {viz_train_size}x{viz_gen_size} samples)")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Coverage comparison
                    axes[1].bar(['Hamming', 'Cosine'], 
                               [mmd_coverage_hamming['coverage_ratio'], mmd_coverage_cosine['coverage_ratio']],
                               color=['blue', 'orange'], alpha=0.7)
                    axes[1].set_ylabel("Coverage Ratio")
                    axes[1].set_title("Coverage Comparison")
                    axes[1].set_ylim([0, 1])
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(ep_dir, "mmd_coverage_visualization.png"), dpi=140)
                    plt.close(fig)
                
                # Store for summary CSV and console output
                current_mmd_hamming = mmd_coverage_hamming['mmd_mean']
                current_coverage_hamming = mmd_coverage_hamming['coverage_ratio']
                current_mmd_cosine = mmd_coverage_cosine['mmd_mean']
                current_coverage_cosine = mmd_coverage_cosine['coverage_ratio']

                # Create transition matrix comparison plots (already done in token grammar section)
                # Transition matrices are visualized in token_transition_histogram.png

                # ====== 追加到匯總 CSV ======
                try:
                    import re
                    row = {"epoch": epoch}
                    
                    # 讀取 grammar_strength_metrics.csv
                    grammar_strength_file = os.path.join(ep_dir, "grammar_strength_metrics.csv")
                    if os.path.exists(grammar_strength_file):
                        try:
                            with open(grammar_strength_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "train_avg_transition_entropy":
                                        row["train_avg_transition_entropy"] = float(value) if value else ""
                                    elif metric == "gen_avg_transition_entropy":
                                        row["gen_avg_transition_entropy"] = float(value) if value else ""
                                    elif metric == "entropy_difference":
                                        row["entropy_difference"] = float(value) if value else ""
                                        # Store for history CSV and console output
                                        current_transition_entropy_diff = float(value) if value else None
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse grammar_strength_metrics.csv: {e}")
                    
                    # 讀取 decoder_occupancy_distribution.csv
                    occ_file = os.path.join(ep_dir, "decoder_occupancy_distribution.csv")
                    if os.path.exists(occ_file):
                        try:
                            with open(occ_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    if metric == "non_air_ratio":
                                        row["non_air_ratio_train_mean"] = float(r.get("train_mean", "")) if r.get("train_mean") else ""
                                        row["non_air_ratio_train_std"] = float(r.get("train_std", "")) if r.get("train_std") else ""
                                        row["non_air_ratio_gen_mean"] = float(r.get("gen_mean", "")) if r.get("gen_mean") else ""
                                        row["non_air_ratio_gen_std"] = float(r.get("gen_std", "")) if r.get("gen_std") else ""
                                    elif metric == "bbox_volume_ratio":
                                        row["bbox_volume_ratio_train_mean"] = float(r.get("train_mean", "")) if r.get("train_mean") else ""
                                        row["bbox_volume_ratio_train_std"] = float(r.get("train_std", "")) if r.get("train_std") else ""
                                        row["bbox_volume_ratio_gen_mean"] = float(r.get("gen_mean", "")) if r.get("gen_mean") else ""
                                        row["bbox_volume_ratio_gen_std"] = float(r.get("gen_std", "")) if r.get("gen_std") else ""
                                    elif metric == "largest_component_ratio":
                                        row["largest_component_ratio_train_mean"] = float(r.get("train_mean", "")) if r.get("train_mean") else ""
                                        row["largest_component_ratio_train_std"] = float(r.get("train_std", "")) if r.get("train_std") else ""
                                        row["largest_component_ratio_gen_mean"] = float(r.get("gen_mean", "")) if r.get("gen_mean") else ""
                                        row["largest_component_ratio_gen_std"] = float(r.get("gen_std", "")) if r.get("gen_std") else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse decoder_occupancy_distribution.csv: {e}")
                    
                    # 讀取 grid_autocorrelation_metrics.csv
                    autocorr_file = os.path.join(ep_dir, "grid_autocorrelation_metrics.csv")
                    if os.path.exists(autocorr_file):
                        try:
                            with open(autocorr_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "train_mean_autocorrelation":
                                        row["train_mean_autocorrelation"] = float(value) if value else ""
                                    elif metric == "gen_mean_autocorrelation":
                                        row["gen_mean_autocorrelation"] = float(value) if value else ""
                                    elif metric == "autocorr_difference":
                                        row["autocorr_difference"] = float(value) if value else ""
                                    elif metric == "train_dir_1_0_0":
                                        row["train_dir_1_0_0"] = float(value) if value else ""
                                    elif metric == "train_dir_0_1_0":
                                        row["train_dir_0_1_0"] = float(value) if value else ""
                                    elif metric == "train_dir_0_0_1":
                                        row["train_dir_0_0_1"] = float(value) if value else ""
                                    elif metric == "gen_dir_1_0_0":
                                        row["gen_dir_1_0_0"] = float(value) if value else ""
                                    elif metric == "gen_dir_0_1_0":
                                        row["gen_dir_0_1_0"] = float(value) if value else ""
                                    elif metric == "gen_dir_0_0_1":
                                        row["gen_dir_0_0_1"] = float(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse grid_autocorrelation_metrics.csv: {e}")
                    
                    # 讀取 transition_drift_metrics.csv
                    drift_file = os.path.join(ep_dir, "transition_drift_metrics.csv")
                    if os.path.exists(drift_file):
                        try:
                            with open(drift_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "transition_kl":
                                        row["transition_kl"] = float(value) if value else ""
                                    elif metric == "transition_l1":
                                        row["transition_l1"] = float(value) if value else ""
                                    elif metric == "transition_js":
                                        row["transition_js"] = float(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse distribution_drift_metrics.csv: {e}")
                    
                    # 讀取 sample_diversity.csv
                    sample_diversity_file = os.path.join(ep_dir, "sample_diversity.csv")
                    if os.path.exists(sample_diversity_file):
                        try:
                            with open(sample_diversity_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "sample_diversity":
                                        row["sample_diversity"] = float(value) if value else ""
                                        # Store for history CSV
                                        current_sample_diversity = float(value) if value else None
                                    elif metric == "unique_tokens":
                                        row["unique_tokens"] = int(value) if value else ""
                                    elif metric == "token_coverage":
                                        row["token_coverage"] = float(value) if value else ""
                                        # Store for history CSV and console output
                                        current_token_coverage = float(value) if value else None
                                    elif metric == "n_samples":
                                        row["sample_diversity_n_samples"] = int(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse sample_diversity.csv: {e}")
                    
                    # 讀取 token_grammar_metrics.csv
                    grammar_file = os.path.join(ep_dir, "token_grammar_metrics.csv")
                    if os.path.exists(grammar_file):
                        try:
                            with open(grammar_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    value = r.get("value", "").strip()
                                    if metric == "avg_train_entropy":
                                        row["avg_train_entropy"] = float(value) if value else ""
                                    elif metric == "avg_gen_entropy":
                                        row["avg_gen_entropy"] = float(value) if value else ""
                                    elif metric == "entropy_diff":
                                        row["entropy_diff"] = float(value) if value else ""
                                    elif metric == "mean_train_pos_entropy":
                                        row["mean_train_pos_entropy"] = float(value) if value else ""
                                    elif metric == "mean_gen_pos_entropy":
                                        row["mean_gen_pos_entropy"] = float(value) if value else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse token_grammar_metrics.csv: {e}")
                    
                    # 讀取 token_usage_metrics.csv
                    usage_file = os.path.join(ep_dir, "token_usage_metrics.csv")
                    if os.path.exists(usage_file):
                        try:
                            with open(usage_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    train_val = r.get("train", "").strip()
                                    gen_val = r.get("generated", "").strip()
                                    if metric == "perplexity":
                                        row["train_perplexity"] = float(train_val) if train_val else ""
                                        row["gen_perplexity"] = float(gen_val) if gen_val else ""
                                    elif metric == "effective_vocab_size":
                                        row["train_effective_vocab_size"] = float(train_val) if train_val else ""
                                        row["gen_effective_vocab_size"] = float(gen_val) if gen_val else ""
                                    elif metric == "token_entropy":
                                        row["train_token_entropy"] = float(train_val) if train_val else ""
                                        row["gen_token_entropy"] = float(gen_val) if gen_val else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse token_usage_metrics.csv: {e}")
                    
                    # 讀取 mmd_coverage_metrics.csv
                    mmd_coverage_file = os.path.join(ep_dir, "mmd_coverage_metrics.csv")
                    if os.path.exists(mmd_coverage_file):
                        try:
                            with open(mmd_coverage_file, "r", newline="") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    metric = r.get("metric", "").strip()
                                    hamming_val = r.get("hamming", "").strip()
                                    cosine_val = r.get("cosine_onehot", "").strip()
                                    if metric == "mmd_mean":
                                        row["mmd_mean_hamming"] = float(hamming_val) if hamming_val else ""
                                        row["mmd_mean_cosine"] = float(cosine_val) if cosine_val else ""
                                    elif metric == "mmd_std":
                                        row["mmd_std_hamming"] = float(hamming_val) if hamming_val else ""
                                        row["mmd_std_cosine"] = float(cosine_val) if cosine_val else ""
                                    elif metric == "coverage_ratio":
                                        row["coverage_ratio_hamming"] = float(hamming_val) if hamming_val else ""
                                        row["coverage_ratio_cosine"] = float(cosine_val) if cosine_val else ""
                                    elif metric == "coverage_count":
                                        row["coverage_count_hamming"] = int(hamming_val) if hamming_val else ""
                                        row["coverage_count_cosine"] = int(cosine_val) if cosine_val else ""
                                    elif metric == "n_train":
                                        row["mmd_n_train"] = int(hamming_val) if hamming_val else ""
                                    elif metric == "n_gen":
                                        row["mmd_n_gen"] = int(hamming_val) if hamming_val else ""
                        except Exception as e:
                            console.print(f"[yellow]Warning:[/yellow] Failed to parse mmd_coverage_metrics.csv: {e}")
                    
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
                    
                    # 確定所有欄位（使用指定的順序）
                    all_fields = set(existing_fields) if existing_fields else set()
                    all_fields.update(row.keys())
                    
                    # 定義字段順序（用戶指定）
                    field_order = [
                        "epoch",
                        # Decode quality metrics
                        "non_air_ratio_train_mean",
                        "non_air_ratio_train_std",
                        "bbox_volume_ratio_train_mean",
                        "bbox_volume_ratio_train_std",
                        "largest_component_ratio_train_mean",
                        "largest_component_ratio_train_std",
                        # MMD/Coverage metrics
                        "coverage_ratio_hamming",
                        "coverage_count_hamming",
                        "mmd_mean_hamming",
                        "mmd_std_hamming",
                        # Sample diversity metrics
                        "sample_diversity",
                        "unique_tokens",
                        "token_coverage",
                        # Auto-correlation metrics
                        "autocorr_difference",
                        "gen_mean_autocorrelation",
                        "train_mean_autocorrelation",
                        # Transition metrics
                        "transition_entropy_diff",
                        "transition_kl",
                        "transition_js",
                        "transition_l1",
                        "gen_avg_transition_entropy",
                        "train_avg_transition_entropy",
                        # Token usage metrics
                        "gen_token_entropy",
                        "train_token_entropy",
                        "gen_perplexity",
                        "train_perplexity",
                        "gen_effective_vocab_size",
                        "train_effective_vocab_size",
                    ]
                    
                    # 按照指定順序排列，未列出的字段放在最後（按字母順序）
                    ordered_fields = []
                    remaining_fields = set(all_fields)
                    
                    for field in field_order:
                        if field in remaining_fields:
                            ordered_fields.append(field)
                            remaining_fields.remove(field)
                    
                    # 添加剩餘字段（按字母順序）
                    ordered_fields.extend(sorted(remaining_fields))
                    all_fields_sorted = ordered_fields
                    
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

            # Console line with transition drift beside val when available
            best_marker = " | ★ Best!" if is_best else ""
            drift_str = f" | TransKL={current_kl:.6f}" if current_kl is not None else ""
            token_coverage_str = f" | TokenCov={current_token_coverage:.4f}" if current_token_coverage is not None else ""
            mmd_str = f" | MMD_H={current_mmd_hamming:.4f}" if current_mmd_hamming is not None else ""
            coverage_str = f" | Cov_H={current_coverage_hamming:.3f}" if current_coverage_hamming is not None else ""
            console.print(
                f"Epoch {epoch:03d}: train={train_loss:.6f} | val={val_loss:.6f}{drift_str}{token_coverage_str}{mmd_str}{coverage_str} | {dt:.1f}s{best_marker}"
            )

            # History
            cum_secs = time.time() - start_time
            history_row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "transition_kl": f"{current_kl:.6f}" if current_kl is not None else "",
                "token_coverage": f"{current_token_coverage:.6f}" if current_token_coverage is not None else "",
                "sample_diversity": f"{current_sample_diversity:.6f}" if current_sample_diversity is not None else "",
                "transition_entropy_diff": f"{current_transition_entropy_diff:.6f}" if current_transition_entropy_diff is not None else "",
                "mmd_hamming": f"{current_mmd_hamming:.6f}" if current_mmd_hamming is not None else "",
                "coverage_hamming": f"{current_coverage_hamming:.6f}" if current_coverage_hamming is not None else "",
                "mmd_cosine": f"{current_mmd_cosine:.6f}" if current_mmd_cosine is not None else "",
                "coverage_cosine": f"{current_coverage_cosine:.6f}" if current_coverage_cosine is not None else "",
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
    test_loss = eval_one_epoch(model, test_loader, device, use_amp=use_amp)
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

    # Sample a latent (use actual_seq_len, not max_seq_len, because we sample seq_len tokens after BOS)
    sample = sample_latent(model, seq_len=actual_seq_len, device=device, mode=args.sample_mode, temperature=args.sample_temperature)
    np.save(os.path.join(exp_dir, "sample_latent.npy"), sample)
    print(f"Saved sample latent (token indices) to {os.path.join(exp_dir, 'sample_latent.npy')}.")

    # 釋放資料集快取，歸還記憶體
    if hasattr(train_ds, "release"):
        train_ds.release()
    if hasattr(val_ds, "release"):
        val_ds.release()
    if hasattr(test_ds, "release"):
        test_ds.release()
    
    # 清理臨時解壓目錄（如果使用了 --data_zip）
    # temp_dir_holder 中的 TemporaryDirectory 會在程序結束時自動清理
    # 但我們可以提前清理以釋放空間
    if temp_dir_holder:
        for temp_dir in temp_dir_holder:
            try:
                temp_dir.cleanup()
                console.print(f"[green]✓[/green] 臨時目錄已清理")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] 清理臨時目錄時出錯：{e}")

if __name__ == "__main__":
    main()
