#!/usr/bin/env python3
"""
最簡單的 3D Voxel Diffusion 訓練腳本 (第一階段)

直接在 16x16x16 voxel 空間訓練 DDPM，使用 3D U-Net 作為 denoiser。
輸入: one-hot encoded voxels [B, 3, 16, 16, 16] (0=air, 1=oak_log, 2=oak_leaves)
輸出: 訓練好的 diffusion model，可用於生成新的 voxel trees

這是實驗計劃的第一階段，先建立 baseline diffusion model。

監控功能 (W&B):
- 第一層：生命跡象 (Health Check)
  * Gradient Norm (梯度範數) - 防止訓練發散
  * Learning Rate - 確認學習率調度
- 第二層：模型行為 (Model Behavior)
  * Predicted Mean/Std (預測值的平均與標準差) - 確認模型輸出合理
  * Param Norm (權重範數) - 監控模型參數大小
- 第三層：專案特有指標 (Domain Specific)
  * Occupancy Rate (佔用率) - 生成 voxel 的非空氣比例
  * Component Count (連通數量) - 生成物體的連通組件數量

使用方式:
  python unet_diffusion_16_voxel_wand.py --data_root <path> --use_wandb

需要安裝:
  pip install wandb
  wandb login  # 首次使用需要登入
"""

import argparse
import os
import math
import random
import shlex
import sys
import time
import csv
import zipfile
import tempfile
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[WARNING] wandb not installed. Monitoring features will be disabled.")

try:
    from scipy.ndimage import label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not installed. Component count will be disabled.")

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich import print  # 用 rich 的 print 覆蓋原生的 print（避免破壞 Progress 顯示）

# ----------------------
# Utility functions
# ----------------------


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fmt_secs(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def get_invocation_command() -> str:
    """Reconstruct the current command in a shell-safe format."""
    if not sys.argv:
        return ""

    executable_name = Path(sys.executable).name if sys.executable else "python"
    if executable_name.startswith("python"):
        executable_name = "python"

    return shlex.join([executable_name, *sys.argv])


def parse_class_weights(arg: str):
    """Parse class weights from comma-separated string.
    
    Args:
        arg: String like "1.0,10.0,10.0" or "none"
    
    Returns:
        torch.Tensor with shape [3] or None if "none"
    """
    if arg is None or arg.strip().lower() == "none":
        return None
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise ValueError("--class_weights must have exactly 3 numbers or 'none'")
    vals = [float(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError("All class weights must be > 0")
    return torch.tensor(vals, dtype=torch.float32)


def extract_zip_to_temp(zip_path: str, console=None) -> Tuple[str, tempfile.TemporaryDirectory]:
    """Extract zip file to a temporary directory and verify train/val/test."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    temp_dir = tempfile.TemporaryDirectory(prefix="train_diffusion_zip_")
    extract_dir = temp_dir.name

    if console:
        console.print(f"[cyan]Extracting zip file: {zip_path}[/cyan]")
        console.print(f"[dim]Temporary extraction directory: {extract_dir}[/dim]")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
        if console:
            console.print(f"[green]✓[/green] Extracted {len(zip_ref.namelist())} items from zip")

    train_dir = os.path.join(extract_dir, "train")
    val_dir = os.path.join(extract_dir, "val")
    test_dir = os.path.join(extract_dir, "test")

    if not os.path.exists(train_dir):
        candidates = []
        for root, dirs, _ in os.walk(extract_dir):
            if "__MACOSX" in root:
                continue
            if {"train", "val", "test"}.issubset(set(dirs)):
                candidates.append(root)
        
        print("FOUND_DIRS:")
        for d in candidates:
            print(" ", d)
        
        if not candidates:
            raise ValueError("Could not find directory containing train/val/test")
        
        extract_dir = candidates[0]
        train_dir = os.path.join(extract_dir, "train")
        val_dir = os.path.join(extract_dir, "val")
        test_dir = os.path.join(extract_dir, "test")

    if not os.path.exists(train_dir):
        raise ValueError(f"train/ not found (checked {train_dir})")
    if not os.path.exists(val_dir):
        raise ValueError(f"val/ not found (checked {val_dir})")
    if not os.path.exists(test_dir):
        raise ValueError(f"test/ not found (checked {test_dir})")

    if console:
        console.print(f"[green]✓[/green] Verified train/val/test structure in extracted directory")

    return extract_dir, temp_dir


# ----------------------
# Dataset
# ----------------------


class VoxelDataset(Dataset):
    """Loads 16x16x16 int8 npz files as class labels; returns one-hot float.
    
    Supports augmentation modes: 'enumerate' or 'random'
    Compatible with train_VQVAE.py data format.
    """

    def __init__(
        self,
        files,
        aug_mode: str = "random",
        aug_flip_x: bool = False,
        aug_flip_y: bool = False,
        aug_flip_z: bool = False,
        aug_rot_x: bool = False,
        aug_rot_y: bool = False,
        aug_rot_z: bool = False,
        aug_perturb: bool = False,
        perturb_prob: float = 0.01,
        preload: bool = False,
        console=None,
    ):
        from itertools import product

        self.files = files
        self.aug_mode = aug_mode
        self.rot_enabled = {"x": aug_rot_x, "y": aug_rot_y, "z": aug_rot_z}
        self.flip_enabled = {"x": aug_flip_x, "y": aug_flip_y, "z": aug_flip_z}
        self.aug_perturb = aug_perturb
        self.perturb_prob = perturb_prob
        self.preload = preload
        self.data_cache = None

        if aug_mode not in ("enumerate", "random"):
            raise ValueError("aug_mode must be 'enumerate' or 'random'")

        if self.preload:
            self.data_cache = []
            if console:
                console.print(f"[cyan]Preloading {len(files)} files into RAM...[/cyan]")
            for i, path in enumerate(files):
                with np.load(path, allow_pickle=False) as data:
                    key = "arr_0" if "arr_0" in data else list(data.files)[0]
                    arr = data[key]
                assert arr.shape == (16, 16, 16), f"Expected (16,16,16), got {arr.shape} from {path}"
                # 優化：使用 int8 而不是 int64，節省記憶體
                arr_int8 = arr.astype(np.int8)
                tensor = torch.from_numpy(arr_int8)
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()
                self.data_cache.append(tensor)
                if console and (i + 1) % 1000 == 0:
                    console.print(f"  Loaded {i + 1}/{len(files)} files...")
            if console:
                # 更新記憶體計算：int8 而不是 int64
                mem_mb = len(files) * 16 * 16 * 16 * 1 / (1024**2)
                console.print(f"[green]✓[/green] Preloaded {len(files)} files (~{mem_mb:.1f} MB)")

        if aug_mode == "enumerate":
            kx = list(range(4)) if self.rot_enabled["x"] else [0]
            ky = list(range(4)) if self.rot_enabled["y"] else [0]
            kz = list(range(4)) if self.rot_enabled["z"] else [0]
            fx = [0, 1] if self.flip_enabled["x"] else [0]
            fy = [0, 1] if self.flip_enabled["y"] else [0]
            fz = [0, 1] if self.flip_enabled["z"] else [0]
            self.combos = list(product(kx, ky, kz, fx, fy, fz))
        else:
            self.combos = None

    def __len__(self):
        if self.aug_mode == "enumerate":
            return len(self.files) * len(self.combos)
        return len(self.files)

    @staticmethod
    def _one_hot(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
        """Convert labels [Z,Y,X] to one-hot [C,Z,Y,X]."""
        return F.one_hot(labels.long(), num_classes=num_classes).permute(3, 0, 1, 2).float()

    @staticmethod
    def _apply_rot_flip(
        labels: torch.Tensor, kx: int, ky: int, kz: int, fx: int, fy: int, fz: int
    ) -> torch.Tensor:
        x = labels
        if kx % 4:
            x = torch.rot90(x, k=int(kx) % 4, dims=(0, 1))
        if ky % 4:
            x = torch.rot90(x, k=int(ky) % 4, dims=(0, 2))
        if kz % 4:
            x = torch.rot90(x, k=int(kz) % 4, dims=(1, 2))
        if fz:
            x = torch.flip(x, dims=[0])
        if fy:
            x = torch.flip(x, dims=[1])
        if fx:
            x = torch.flip(x, dims=[2])
        return x

    def _random_choice(self):
        kx = torch.randint(0, 4, (1,)).item() if self.rot_enabled["x"] else 0
        ky = torch.randint(0, 4, (1,)).item() if self.rot_enabled["y"] else 0
        kz = torch.randint(0, 4, (1,)).item() if self.rot_enabled["z"] else 0
        fx = int(torch.rand(()) < 0.5) if self.flip_enabled["x"] else 0
        fy = int(torch.rand(()) < 0.5) if self.flip_enabled["y"] else 0
        fz = int(torch.rand(()) < 0.5) if self.flip_enabled["z"] else 0
        return kx, ky, kz, fx, fy, fz

    def _perturb(self, labels: torch.Tensor) -> torch.Tensor:
        if self.perturb_prob <= 0:
            return labels
        p = self.perturb_prob
        priors = torch.tensor([0.7, 0.15, 0.15], dtype=torch.float32, device=labels.device)
        mask = torch.rand_like(labels.float()) < p
        new_vals = torch.multinomial(priors, num_samples=labels.numel(), replacement=True).view_as(labels)
        return torch.where(mask, new_vals.to(labels.dtype), labels)

    def __getitem__(self, idx):
        if self.aug_mode == "enumerate":
            file_idx = idx // len(self.combos)
            combo_idx = idx % len(self.combos)
            kx, ky, kz, fx, fy, fz = self.combos[combo_idx]
        else:
            file_idx = idx
            kx, ky, kz, fx, fy, fz = self._random_choice()

        if self.preload:
            # 優化：只在需要時 clone，並且轉換為 int64（one_hot 需要）
            labels = self.data_cache[file_idx].clone().to(torch.int64)
        else:
            path = self.files[file_idx]
            with np.load(path, allow_pickle=False) as data:
                key = "arr_0" if "arr_0" in data else list(data.files)[0]
                arr = data[key]
            assert arr.shape == (16, 16, 16), f"Expected (16,16,16), got {arr.shape} from {path}"
            labels = torch.from_numpy(arr.astype(np.int64))

        labels = self._apply_rot_flip(labels, kx, ky, kz, fx, fy, fz)
        if self.aug_perturb and self.perturb_prob > 0:
            labels = self._perturb(labels)

        onehot = self._one_hot(labels, 3)  # [3,16,16,16]
        return onehot, labels


# ----------------------
# Diffusion Model: 3D U-Net Denoiser
# ----------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: [B] int or float in [0, T)
        return: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ResBlock3D(nn.Module):
    """3D Residual block with GroupNorm and SiLU."""

    def __init__(self, ch: int, time_emb_dim: int = None):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1, padding_mode='replicate'),
        )

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, ch)

    def forward(self, x, t_emb=None):
        h = self.net(x)
        if t_emb is not None and self.time_emb_dim is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None, None]
        return x + h


class UNet3DDiffusion(nn.Module):
    """
    3D U-Net for voxel diffusion denoising.
    
    Architecture: 16 -> 8 -> 4 -> 8 -> 16 (with skip connections)
    Input:  x_t: [B, 3, 16, 16, 16], t: [B]
    Output: eps_hat: [B, 3, 16, 16, 16] (predicted noise)
    """

    def __init__(self, in_ch: int = 3, base: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder (downsampling)
        self.enc1_conv = nn.Conv3d(in_ch, base, 3, padding=1, padding_mode='replicate')
        self.enc1_res = ResBlock3D(base, time_dim)
        
        self.enc2_conv = nn.Conv3d(base, base * 2, 4, stride=2, padding=1, padding_mode='replicate')  # 16 -> 8
        self.enc2_res = ResBlock3D(base * 2, time_dim)
        
        self.enc3_conv = nn.Conv3d(base * 2, base * 4, 4, stride=2, padding=1, padding_mode='replicate')  # 8 -> 4
        self.enc3_res = ResBlock3D(base * 4, time_dim)

        # Middle
        self.mid1 = ResBlock3D(base * 4, time_dim)
        self.mid2 = ResBlock3D(base * 4, time_dim)

        # Decoder (upsampling with skip connections)
        self.dec3_res = ResBlock3D(base * 4, time_dim)
        self.dec3_up = nn.ConvTranspose3d(base * 4, base * 2, 4, stride=2, padding=1)  # 4 -> 8
        
        self.dec2_res = ResBlock3D(base * 2 * 2, time_dim)  # *2 for skip connection
        self.dec2_up = nn.ConvTranspose3d(base * 2 * 2, base, 4, stride=2, padding=1)  # 8 -> 16
        
        self.dec1_res = ResBlock3D(base * 2, time_dim)  # *2 for skip connection
        self.dec1_conv = nn.Conv3d(base * 2, base, 3, padding=1, padding_mode='replicate')  # Reduce channels after skip connection

        # Output
        self.out_conv = nn.Conv3d(base, in_ch, 3, padding=1, padding_mode='replicate')

    def forward(self, x, t):
        """
        x: [B, 3, 16, 16, 16]
        t: [B] timestep indices
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_dim]

        # Encoder with skip connections
        e1 = self.enc1_conv(x)  # [B, base, 16, 16, 16]
        e1 = self.enc1_res(e1, t_emb)

        e2 = self.enc2_conv(e1)  # [B, base*2, 8, 8, 8]
        e2 = self.enc2_res(e2, t_emb)

        e3 = self.enc3_conv(e2)  # [B, base*4, 4, 4, 4]
        e3 = self.enc3_res(e3, t_emb)

        # Middle
        h = self.mid1(e3, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder with skip connections
        h = self.dec3_res(h, t_emb)
        h = self.dec3_up(h)  # [B, base*2, 8, 8, 8]
        h = torch.cat([h, e2], dim=1)  # Skip connection

        h = self.dec2_res(h, t_emb)
        h = self.dec2_up(h)  # [B, base, 16, 16, 16]
        h = torch.cat([h, e1], dim=1)  # Skip connection

        h = self.dec1_res(h, t_emb)
        h = self.dec1_conv(h)  # [B, base, 16, 16, 16] - reduce channels after skip connection

        # Output
        out = self.out_conv(h)  # [B, 3, 16, 16, 16]
        return out


# ----------------------
# Diffusion Process
# ----------------------


class BetaSchedule:
    """Beta schedule for DDPM forward process."""

    def __init__(self, T: int, schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.schedule = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._build()

    def _build(self):
        if self.schedule == "linear":
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.T)
        elif self.schedule == "cosine":
            # Cosine schedule (simplified)
            s = 0.008
            t = torch.linspace(0, self.T, self.T + 1)
            alphas_cumprod = torch.cos(((t / self.T) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clamp(self.beta, 1e-4, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def onehot_to_centered(onehot: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot encoding from [0,1] to [-1,1] range for better DDPM training stability.
    
    Transformation:
        [1.0, 0.0, 0.0] (air)   -> [1.0, -1.0, -1.0]
        [0.0, 1.0, 0.0] (log)    -> [-1.0, 1.0, -1.0]
        [0.0, 0.0, 1.0] (leaves) -> [-1.0, -1.0, 1.0]
    
    Args:
        onehot: [B, C, H, W, D] one-hot encoding in [0,1]
    
    Returns:
        centered: [B, C, H, W, D] in [-1,1] range
    """
    # Formula: 2 * onehot - 1
    # This maps [0,1] -> [-1,1] while preserving one-hot structure
    return 2.0 * onehot - 1.0


def centered_to_onehot(centered: torch.Tensor) -> torch.Tensor:
    """
    Convert from [-1,1] range back to [0,1] one-hot-like values.
    
    Args:
        centered: [B, C, H, W, D] in [-1,1] range
    
    Returns:
        onehot: [B, C, H, W, D] in [0,1] range
    """
    # Inverse: (centered + 1) / 2
    return (centered + 1.0) / 2.0


def q_sample(x_0, t, alpha_bar, device):
    """
    Forward diffusion process: q(x_t | x_0)
    
    Note: x_0 should already be in [-1,1] range (use onehot_to_centered first).
    
    Args:
        x_0: [B, C, H, W, D] clean voxels in [-1,1] range
        t: [B] timestep indices
        alpha_bar: [T] cumulative product of alphas
        device: torch device
    
    Returns:
        x_t: [B, C, H, W, D] noisy voxels
        eps: [B, C, H, W, D] noise that was added
    """
    B = x_0.shape[0]
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t])[:, None, None, None, None]  # [B, 1, 1, 1, 1]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar[t])[:, None, None, None, None]

    # Sample noise
    eps = torch.randn_like(x_0)

    # Add noise
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * eps

    return x_t, eps


@torch.no_grad()
def sample_voxels(
    model,
    betas,
    shape,
    device,
    n_steps=None,
    use_amp=False,
    track_every=None,
    track_callback=None,
    verbose: bool = True,
):
    """
    Reverse diffusion process: sample voxels from noise.
    
    Args:
        model: UNet3DDiffusion model
        betas: BetaSchedule instance
        shape: (B, C, H, W, D) where C=3, H=W=D=16
        device: torch device
        n_steps: number of sampling steps (default: T, can use fewer for speed)
        use_amp: whether to use mixed precision
        track_every: if not None, track metrics every N steps (calls track_callback)
        track_callback: callback function(sample_idx, step_idx, t_int, x_current, x0_hat) called every track_every steps
        verbose: if False, skip step-wise debug prints (useful for bulk generation)
    
    Returns:
        x_0: [B, C, H, W, D] sampled voxels in [-1,1] range
    """
    T = betas.T
    if n_steps is None:
        n_steps = T

    B, C, H, W, D = shape
    # Start from pure noise
    x = torch.randn(shape, device=device)  # x_T ~ N(0,I)

    # Create timestep schedule for sampling (can use fewer steps for speed)
    if n_steps < T:
        # Use evenly spaced timesteps
        timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, device=device)

    # Reverse diffusion: T -> T-1 -> ... -> 0
    for i, t_int_tensor in enumerate(timesteps):
        t_int = t_int_tensor.item() if isinstance(t_int_tensor, torch.Tensor) else int(t_int_tensor)
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        
        # Predict noise (with autocast if enabled)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            eps_pred = model(x, t)

        # Get schedule values
        beta_t = betas.beta[t_int]
        alpha_t = betas.alpha[t_int]
        alpha_bar_t = betas.alpha_bar[t_int]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Predict x_0 from x_t and predicted noise
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / torch.sqrt(alpha_bar_t)

        # === 檢查數值是否失控 ===
        # 我們只在第 0 號樣本 (i==0) 且每隔 200 步印一次，避免洗版
        if verbose and i % 200 == 0:
            # 計算 sqrt(alpha_bar_t) - 這是導致數值爆炸的關鍵
            sab = torch.sqrt(alpha_bar_t).item()
            
            # U-Net 原始輸出 (eps_pred) 的統計
            eps_min = eps_pred.min().item()
            eps_max = eps_pred.max().item()
            eps_mean = eps_pred.mean().item()
            eps_std = eps_pred.std().item()
            
            # pred_x0 的統計
            pred_min = pred_x0.min().item()
            pred_max = pred_x0.max().item()
            pred_mean = pred_x0.mean().item()
            
            print(f"[Debug] Step {t_int}:")
            print(f"  sqrt(alpha_bar_t)={sab:.6e}")
            print(f"  eps_pred (U-Net輸出): range=[{eps_min:.2f}, {eps_max:.2f}], mean={eps_mean:.2f}, std={eps_std:.2f}")
            print(f"  pred_x0 (計算後): range=[{pred_min:.2f}, {pred_max:.2f}], mean={pred_mean:.2f}")
            
            # 判斷是否異常
            if pred_max > 2.0 or pred_min < -2.0:
                print(f"  [警報] pred_x0 數值飄移偵測！range=[{pred_min:.2f}, {pred_max:.2f}], mean={pred_mean:.2f}")
            if abs(eps_mean) > 10.0 or eps_std > 50.0:
                print(f"  [警報] eps_pred 數值異常！mean={eps_mean:.2f}, std={eps_std:.2f}")
        # =======================================

        # === [修正] 強制鉗制 (先 Log 再 Clamp) ===
        # 記錄 clamp 前的數值（用於對比）
        pred_x0_before_clamp_min = pred_x0.min().item()
        pred_x0_before_clamp_max = pred_x0.max().item()
        pred_x0 = pred_x0.clamp(-1., 1.)
        pred_x0_after_clamp_min = pred_x0.min().item()
        pred_x0_after_clamp_max = pred_x0.max().item()
        
        # 如果數值被 clamp 了，記錄日誌
        if verbose and i % 200 == 0:
            was_clamped = (pred_x0_before_clamp_min < -1.0) or (pred_x0_before_clamp_max > 1.0)
            if was_clamped:
                print(f"  [Clamp修正] pred_x0 已鉗制至: range=[{pred_x0_after_clamp_min:.2f}, {pred_x0_after_clamp_max:.2f}]")
        # ===================================================

        # Compute posterior mean
        if t_int > 0:
            alpha_bar_prev = betas.alpha_bar[t_int - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        posterior_mean = coef1 * pred_x0 + coef2 * x

        # Sample next step
        if t_int > 0:
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x)
            x = posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            x = posterior_mean
        
        # Track metrics if enabled
        if track_every is not None and track_callback is not None and i % track_every == 0:
            # Call callback for each sample in batch
            # Pass both x_current (x_t) and x0_hat (pred_x0) for tracking
            for sample_idx in range(B):
                track_callback(sample_idx, i, t_int, x[sample_idx], pred_x0[sample_idx])

    return x  # x_0 in [-1,1] range


def compute_occupancy_rate(labels: np.ndarray, air_class: int = 0) -> float:
    """
    Compute occupancy rate (non-air voxels / total voxels).
    
    Args:
        labels: [Z, Y, X] numpy array with class labels
        air_class: Class ID for air (default: 0)
    
    Returns:
        occupancy_rate: float in [0, 1]
    """
    return float((labels != air_class).sum()) / labels.size


def compute_component_count(labels: np.ndarray, air_class: int = 0) -> int:
    """
    Compute number of connected components in non-air voxels.
    
    Args:
        labels: [Z, Y, X] numpy array with class labels
        air_class: Class ID for air (default: 0)
    
    Returns:
        num_components: int, number of connected components
    """
    if not HAS_SCIPY:
        return -1  # Indicate scipy not available
    
    # Create binary mask for non-air voxels
    mask = (labels != air_class).astype(np.int32)
    
    if mask.sum() == 0:
        return 0
    
    # Use 3D connectivity (6-connectivity: faces only)
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled_array, num_components = label(mask, structure=structure)
    
    return int(num_components)


def compute_occupancy_rates(labels: np.ndarray) -> dict:
    """
    Compute occupancy rates for different material types.
    
    Args:
        labels: [Z, Y, X] numpy array with class labels (0=air, 1=log, 2=leaf)
    
    Returns:
        dict with keys: 'non_air', 'log', 'leaf'
    """
    total = labels.size
    non_air = ((labels == 1) | (labels == 2)).sum()
    log = (labels == 1).sum()
    leaf = (labels == 2).sum()
    
    return {
        'non_air': float(non_air) / total,
        'log': float(log) / total,
        'leaf': float(leaf) / total,
    }


def compute_component_counts_26neighbor(labels: np.ndarray) -> dict:
    """
    Compute connected component counts using 26-neighbor connectivity.
    
    Args:
        labels: [Z, Y, X] numpy array with class labels (0=air, 1=log, 2=leaf)
    
    Returns:
        dict with keys: 'non_air', 'log', 'leaf', each containing component count
    """
    if not HAS_SCIPY:
        return {'non_air': -1, 'log': -1, 'leaf': -1}
    
    # 26-neighbor connectivity structure (3x3x3 with all ones)
    structure = np.ones((3, 3, 3), dtype=np.int32)
    
    results = {}
    
    # Non-air (log + leaf)
    mask_non_air = ((labels == 1) | (labels == 2)).astype(np.int32)
    if mask_non_air.sum() == 0:
        results['non_air'] = 0
    else:
        _, num_components = label(mask_non_air, structure=structure)
        results['non_air'] = int(num_components)
    
    # Log only
    mask_log = (labels == 1).astype(np.int32)
    if mask_log.sum() == 0:
        results['log'] = 0
    else:
        _, num_components = label(mask_log, structure=structure)
        results['log'] = int(num_components)
    
    # Leaf only
    mask_leaf = (labels == 2).astype(np.int32)
    if mask_leaf.sum() == 0:
        results['leaf'] = 0
    else:
        _, num_components = label(mask_leaf, structure=structure)
        results['leaf'] = int(num_components)
    
    return results


def compute_trunk_breakage(labels: np.ndarray, debug: bool = False) -> dict:
    """
    Compute trunk breakage: check if wood voxels form a connected path from base (ground) to highest wood (top).
    
    Note: Ground is at Y=0 (the bottom layer in Y dimension).
    The function finds wood voxels at Y=0 as the base.
    
    Args:
        labels: [Z, Y, X] numpy array with class labels (0=air, 1=log, 2=leaf)
        debug: if True, print debug information
    
    Returns:
        dict with keys:
            'is_main_trunk_broken': bool, True if main trunk is broken (no connected path from ground to top)
            'is_broken': bool, True if there are any disconnected wood components (not connected to ground)
            'break_count': int, number of disconnected wood components (excluding base-connected one)
            'base_connected_size': int, size of base-connected component
            'total_wood_size': int, total number of wood voxels
    """
    if not HAS_SCIPY:
        return {'is_main_trunk_broken': False, 'is_broken': False, 'break_count': -1, 'base_connected_size': -1, 'total_wood_size': -1}
    
    # Create binary mask for wood (log)
    mask_log = (labels == 1).astype(np.int32)
    
    if mask_log.sum() == 0:
        return {'is_main_trunk_broken': False, 'is_broken': False, 'break_count': 0, 'base_connected_size': 0, 'total_wood_size': 0}
    
    # 26-neighbor connectivity
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled_array, num_components = label(mask_log, structure=structure)
    
    # Find base seeds: ONLY check Y=0 (ground layer, the bottom layer in Y dimension)
    # labels shape is [Z, Y, X], so ground is at Y=0
    y_ground = 0  # Ground is at Y=0
    base_slice = labeled_array[:, y_ground, :]  # [Z, X] slice at Y=0 (ground)
    wood_at_ground = base_slice[base_slice > 0]
    
    if len(wood_at_ground) == 0:
        # No wood at ground layer - base_connected_size is 0
        if debug:
            print(f"[TrunkDebug] No wood found at ground layer (Y={y_ground}), base_connected_size=0")
        
        # Still need to check if trunk is broken (no path from ground to top)
        # Find the highest wood voxel for is_main_trunk_broken calculation
        wood_coords = np.argwhere(mask_log > 0)
        if len(wood_coords) == 0:
            return {'is_main_trunk_broken': False, 'break_count': 0, 'base_connected_size': 0, 'total_wood_size': 0}
        
        max_y = wood_coords[:, 1].max()  # Largest Y = highest point (top)
        # Since there's no wood at ground, trunk is definitely broken
        # All wood components are disconnected from ground
        return {
            'is_main_trunk_broken': True,
            'is_broken': True,  # All components are disconnected
            'break_count': num_components,
            'base_connected_size': 0,  # No wood at ground, so 0
            'total_wood_size': mask_log.sum(),
        }
    
    # Found wood at ground layer - get all component labels at ground
    base_labels = np.unique(wood_at_ground).tolist()
    if debug:
        print(f"[TrunkDebug] Found wood at ground layer (Y={y_ground}), wood voxels: {len(wood_at_ground)}, component labels: {base_labels}")
    
    # Find the highest wood voxel (largest y coordinate, y=15 is top)
    wood_coords = np.argwhere(mask_log > 0)
    if len(wood_coords) == 0:
        return {'is_main_trunk_broken': False, 'is_broken': False, 'break_count': 0, 'base_connected_size': 0, 'total_wood_size': 0}
    
    max_y = wood_coords[:, 1].max()  # Largest Y = highest point (top)
    highest_wood_coords = wood_coords[wood_coords[:, 1] == max_y]
    highest_labels = np.unique([labeled_array[tuple(coord)] for coord in highest_wood_coords])
    highest_labels = highest_labels[highest_labels > 0]
    
    # Check if any base-connected component reaches the highest wood
    base_connected_labels = set(base_labels)
    reaches_top = len(base_connected_labels & set(highest_labels)) > 0
    
    if debug:
        print(f"[TrunkDebug] Total components: {num_components}")
        print(f"[TrunkDebug] Base-connected labels: {base_connected_labels}")
        print(f"[TrunkDebug] Highest wood at Y={max_y}, labels: {set(highest_labels)}")
        print(f"[TrunkDebug] Reaches top: {reaches_top}")
    
    # Count disconnected components (excluding base-connected ones)
    all_labels = set(range(1, num_components + 1))
    disconnected_labels = all_labels - base_connected_labels
    break_count = len(disconnected_labels)
    is_broken = break_count > 0  # True if there are any disconnected wood components
    
    # Calculate base-connected component size
    base_connected_size = 0
    for label_id in base_connected_labels:
        size = (labeled_array == label_id).sum()
        base_connected_size += size
        if debug:
            print(f"[TrunkDebug] Component {label_id} size: {size}")
    
    if debug:
        print(f"[TrunkDebug] Result: is_main_trunk_broken={not reaches_top}, is_broken={is_broken}, break_count={break_count}, base_connected_size={base_connected_size}, total_wood={mask_log.sum()}")
    
    return {
        'is_main_trunk_broken': not reaches_top,
        'is_broken': is_broken,
        'break_count': break_count,
        'base_connected_size': base_connected_size,
        'total_wood_size': mask_log.sum(),
    }


def compute_largest_log_component_ratio(labels: np.ndarray) -> float:
    """
    Compute ratio: largest_log_component_size / total_log_size.
    
    Args:
        labels: [Z, Y, X] numpy array with class labels (0=air, 1=log, 2=leaf)
    
    Returns:
        ratio: float in [0, 1], or -1 if no wood voxels
    """
    if not HAS_SCIPY:
        return -1.0
    
    # Create binary mask for wood (log)
    mask_log = (labels == 1).astype(np.int32)
    
    if mask_log.sum() == 0:
        return 0.0
    
    # 26-neighbor connectivity
    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled_array, num_components = label(mask_log, structure=structure)
    
    if num_components == 0:
        return 0.0
    
    # Find largest component
    largest_size = 0
    for label_id in range(1, num_components + 1):
        size = (labeled_array == label_id).sum()
        largest_size = max(largest_size, size)
    
    total_size = mask_log.sum()
    return float(largest_size) / total_size if total_size > 0 else 0.0


@torch.no_grad()
def save_volume_and_projections(vol_logits, out_npz, out_png, title_suffix=""):
    """
    Save voxel volume as npz and generate 3-view projection PNG and slices PNG.
    
    Args:
        vol_logits: [3,16,16,16] tensor in [0,1] range (after centered_to_onehot conversion)
        out_npz: output npz file path
        out_png: output png file path for maximum intensity projections
        title_suffix: optional suffix to add to the figure title
    """
    import matplotlib.pyplot as plt

    # Get discrete labels using argmax (softmax is monotonic, so argmax order is unchanged)
    # Note: probs is still computed below for heatmap visualization
    labels = vol_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    # Compute probabilities for heatmap visualization (needed for structure_prob)
    probs = F.softmax(vol_logits, dim=0)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, labels)

    # Generate 3-view projections (參考 train_VQVAE.py)
    max_z = labels.max(axis=0)  # Max projection along Z axis -> XY view
    max_y = labels.max(axis=1)  # Max projection along Y axis -> XZ view
    max_x = labels.max(axis=2)  # Max projection along X axis -> YZ view

    # Save maximum intensity projections (original format)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(max_z)
    axes[0].set_title("Z" + title_suffix)
    axes[1].imshow(max_y)
    axes[1].set_title("Y" + title_suffix)
    axes[2].imshow(max_x)
    axes[2].set_title("X" + title_suffix)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # Generate slices at Z=4, 8, 12 (for debugging "box effect")
    slice_z_4 = labels[4, :, :]   # XY slice at Z=4
    slice_z_8 = labels[8, :, :]  # XY slice at Z=8 (middle)
    slice_z_12 = labels[12, :, :]  # XY slice at Z=12

    # Save slices to separate file
    out_slices_png = str(out_png).replace(".png", "_slices.png")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(slice_z_4)
    axes[0].set_title(f"Slice Z=4 (Y,X)" + title_suffix)
    axes[1].imshow(slice_z_8)
    axes[1].set_title(f"Slice Z=8 (Y,X)" + title_suffix)
    axes[2].imshow(slice_z_12)
    axes[2].set_title(f"Slice Z=12 (Y,X)" + title_suffix)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_slices_png, dpi=140)
    plt.close(fig)

    # Generate heatmap showing "non-air" probability (wood + leaves)
    # This helps visualize weak signals that argmax would miss
    # vol_logits shape: [3, 16, 16, 16]
    structure_prob = probs[1] + probs[2]  # Wood probability + leaves probability, shape [16, 16, 16]
    
    # Convert to numpy for plotting
    struct_map = structure_prob.cpu().numpy()
    
    # Max projection along each axis
    max_z = struct_map.max(axis=0)  # XY view
    max_y = struct_map.max(axis=1)   # XZ view
    max_x = struct_map.max(axis=2)   # YZ view
    
    # Create heatmap figure
    out_heatmap_png = str(out_png).replace(".png", "_heatmap.png")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Use 'magma' colormap to make low probabilities (like 0.1) visible
    im0 = axes[0].imshow(max_z, cmap='magma', vmin=0, vmax=1)
    axes[0].set_title("Prob Z" + title_suffix)
    axes[0].axis("off")
    
    im1 = axes[1].imshow(max_y, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title("Prob Y" + title_suffix)
    axes[1].axis("off")
    
    im2 = axes[2].imshow(max_x, cmap='magma', vmin=0, vmax=1)
    axes[2].set_title("Prob X" + title_suffix)
    axes[2].axis("off")
    
    # Add colorbar to show probability scale
    fig.colorbar(im2, ax=axes.ravel().tolist(), label="Non-air probability")
    
    fig.tight_layout()
    fig.savefig(out_heatmap_png, dpi=140)
    plt.close(fig)


@torch.no_grad()
def save_labels_and_projections(labels: np.ndarray, out_png: str, title_suffix: str = "", exp_name: str = ""):
    """
    Save 3-view projections from discrete labels (no softmax).
    Priority: wood (1) > leaf (2) > air (0)
    
    Args:
        labels: [16,16,16] uint8 array, values in {0,1,2}
        out_png: output png file path
        title_suffix: optional suffix to add to the figure title
        exp_name: experiment name to display as main title
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    def priority_max(arr, axis):
        """Max projection with priority: wood (1) > leaf (2) > air (0)"""
        # Create priority mask: wood=3, leaf=2, air=1
        priority = np.where(arr == 1, 3, np.where(arr == 2, 2, 1))
        # Find max priority along axis
        max_priority = priority.max(axis=axis)
        # Map back to original labels: 3->1 (wood), 2->2 (leaf), 1->0 (air)
        result = np.where(max_priority == 3, 1, np.where(max_priority == 2, 2, 0))
        return result

    max_z = priority_max(labels, axis=0)  # XY view
    max_y = priority_max(labels, axis=1)  # XZ view
    max_x = priority_max(labels, axis=2)  # YZ view

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(max_z)
    axes[0].set_title("Z" + title_suffix)
    axes[1].imshow(max_y)
    axes[1].set_title("Y" + title_suffix)
    axes[2].imshow(max_x)
    axes[2].set_title("X" + title_suffix)
    for ax in axes:
        ax.axis("off")
    
    # Add main title (experiment name) if provided
    if exp_name:
        fig.suptitle(exp_name, fontsize=12, fontweight='bold', y=1.02)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ----------------------
# Training loop
# ----------------------


class DummyProgress:
    """Dummy progress object that does nothing (for --no_progress mode)."""
    
    def __init__(self, console):
        self.console = console
        self._task_counter = 0
    
    def add_task(self, *args, **kwargs):
        """Return a dummy task ID."""
        task_id = self._task_counter
        self._task_counter += 1
        return task_id
    
    def update(self, *args, **kwargs):
        """Do nothing."""
        pass
    
    def remove_task(self, *args, **kwargs):
        """Do nothing."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def train_diffusion(args):
    global_t0 = time.time()
    train_start_time = datetime.now()
    seed_everything(args.seed)
    
    # 根據 --no_progress 參數決定 Console 設定
    if args.no_progress:
        # 不使用進度條模式：保留顏色輸出，但不強制互動模式（適合 screen/tmux/nohup）
        # force_terminal=True 確保顏色輸出，但不設置 force_interactive 避免進度條相關的游標操作
        console = Console(force_terminal=True)
    else:
        # 使用進度條模式：強制開啟終端機與互動模式（讓 Progress/游標覆寫更穩定）
        console = Console(force_terminal=True, force_interactive=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Using device: {device}[/cyan]")

    # Load data
    console.print(f"\n[bold]Loading data...[/bold]")
    train_files = sorted(glob(os.path.join(args.data_root, "train", "*.npz")))
    val_files = sorted(glob(os.path.join(args.data_root, "val", "*.npz")))

    if not train_files:
        raise ValueError(f"No training files found in {args.data_root}/train/")
    if not val_files:
        raise ValueError(f"No validation files found in {args.data_root}/val/")

    console.print(f"Found {len(train_files)} training files, {len(val_files)} validation files")

    train_dataset = VoxelDataset(
        train_files,
        aug_mode=args.aug_mode,
        aug_flip_x=args.aug_flip_x,
        aug_flip_y=args.aug_flip_y,
        aug_flip_z=args.aug_flip_z,
        aug_rot_x=args.aug_rot_x,
        aug_rot_y=args.aug_rot_y,
        aug_rot_z=args.aug_rot_z,
        aug_perturb=args.aug_perturb,
        perturb_prob=args.perturb_prob,
        preload=args.preload,
        console=console,
    )
    val_dataset = VoxelDataset(
        val_files,
        aug_mode="random",  # Validation uses random augmentation
        aug_perturb=False,
        preload=args.preload,
        console=console,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model
    console.print(f"\n[bold]Initializing model...[/bold]")
    model = UNet3DDiffusion(in_ch=3, base=args.base_channels, time_dim=args.time_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize W&B
    if HAS_WANDB and args.use_wandb:
        wandb.init(
            project=args.wandb_project if hasattr(args, 'wandb_project') else "3d-voxel-diffusion",
            name=args.exp_name,
            config=vars(args),
            resume="allow" if args.resume else None,
        )
        # Note: wandb.watch() is disabled to use epoch-level logging only
        # wandb.watch(model, log="all", log_freq=100)
        # All wandb.log() calls use step=epoch+1 to ensure x-axis shows epochs
        
        # Define custom metrics for sampling tracking (use sampling_step as x-axis)
        # This allows sampling tracking to use its own step counter without conflicting with epoch-level logging
        track_every = getattr(args, 'sample_track_every', None)
        if track_every is not None:
            wandb.define_metric("tracking/*", step_metric="sampling_step")
            console.print(f"[dim]  W&B sampling tracking: using 'sampling_step' as x-axis[/dim]")
        
        console.print(f"[green]✓[/green] W&B initialized: project={wandb.run.project}, name={wandb.run.name}")
        console.print(f"[dim]  W&B logging mode: epoch-level (step=epoch+1)[/dim]")
    elif args.use_wandb:
        console.print("[yellow]⚠[/yellow] W&B requested but not installed. Monitoring disabled.")

    # Beta schedule
    betas = BetaSchedule(T=args.T, schedule=args.beta_schedule).to(device)

    # Mixed precision
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    # Class weights for channel weighting
    class_weights_base = parse_class_weights(args.class_weights)
    if class_weights_base is None:
        console.print("[bold]Class weights:[/bold] NONE (uniform)")
        loss_weights = None
    else:
        console.print(f"[bold]Class weights:[/bold] {class_weights_base.tolist()}")
        console.print(
            f"[dim]  air={class_weights_base[0]:.4f}, "
            f"log={class_weights_base[1]:.4f}, "
            f"leaf={class_weights_base[2]:.4f}[/dim]"
        )
        # Format: [1, 3, 1, 1, 1] for broadcasting with [B, 3, 16, 16, 16]
        loss_weights = class_weights_base.to(device=device).view(1, 3, 1, 1, 1)

    # Output directory
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    
    # Check if experiment directory already exists and is not empty (unless resuming)
    if not args.resume:
        if os.path.exists(exp_dir) and os.listdir(exp_dir):
            console.print(
                Panel.fit(
                    "[bold red]ERROR: Experiment Directory Not Empty[/bold red]\n\n"
                    f"[yellow]{exp_dir}[/yellow]\n"
                    "Use another --exp_name, clean directory, or use --resume.",
                    border_style="red",
                )
            )
            raise SystemExit(1)
    
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    samples_dir = os.path.join(exp_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Copy script backup to experiment directory
    if "__file__" in globals():
        script_path = Path(__file__).resolve()
        if script_path.exists():
            script_backup_path = os.path.join(exp_dir, script_path.name)
            shutil.copy2(script_path, script_backup_path)
            console.print(f"[green]✓[/green] Copied script backup to: [cyan]{script_backup_path}[/cyan]")

    # Training state
    start_epoch = 0
    best_val_loss = math.inf
    training_history = []
    cumulative_time_offset = 0.0

    # Resume from checkpoint
    if args.resume:
        ckpt_path = os.path.join(ckpt_dir, args.resume)
        if os.path.exists(ckpt_path):
            console.print(f"[cyan]Resuming from {ckpt_path}[/cyan]")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", math.inf)
            training_history = ckpt.get("history", [])
            cumulative_time_offset = ckpt.get("cumulative_time_secs", 0.0)
    
    # Note: Using epoch-level logging only, no step counter needed
    # global_wandb_step is kept for compatibility but not used for wandb.log step parameter

    # Prepare metadata paths
    csv1_path = os.path.join(exp_dir, f"training_history_{args.exp_name}.csv")
    csv2_path = os.path.join(exp_dir, f"experiment_metadata_{args.exp_name}.csv")
    csv3_path = os.path.join(exp_dir, f"experiment_metadata_flat_{args.exp_name}.csv")

    # Helper function for boolean to string
    def bool_to_str(v):
        return "TRUE" if v else "FALSE"

    # Prepare loss function description
    if loss_weights is not None:
        loss_desc = f"MSE with channel weights [{class_weights_base[0]:.2f}, {class_weights_base[1]:.2f}, {class_weights_base[2]:.2f}]"
    else:
        loss_desc = "MSE (uniform weighting)"
    
    current_script = Path(__file__).name if "__file__" in globals() else "interactive_session"
    invocation_command = get_invocation_command()

    # Create initial metadata (all fields that can be determined before training)
    # Note: start_epoch is set correctly after resume check above
    initial_metadata = {
        "exp_name": args.exp_name,
        "resumed_from": args.resume if args.resume else "None",
        "start_epoch": start_epoch,
        "end_epoch": args.epochs,
        "training_start_time": train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_cwd": os.getcwd(),
        "execution_command": invocation_command,
        "best_model_path": os.path.join(ckpt_dir, "best.pt"),
        "last_checkpoint_path": os.path.join(ckpt_dir, "last.pt"),
        "samples_directory": samples_dir,
        "data_root": args.data_root,
        "out_dir": args.out_dir,
        "exp_dir": exp_dir,
        "script_name": current_script,
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_test_files": 0,  # Diffusion script doesn't use test set
        "n_total_files": len(train_files) + len(val_files),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": 0,  # Diffusion script doesn't use test set
        "class_weights": args.class_weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.num_workers,  # Use "workers" to match train_VQVAE.py naming
        "seed": args.seed,
        "force_cpu": bool_to_str(False),  # Diffusion script doesn't have --cpu flag, but keep for consistency
        "device": str(device),
        "amp_enabled": bool_to_str(use_amp),
        "no_amp": bool_to_str(args.no_amp),
        "preload": bool_to_str(args.preload),
        "base_channels": args.base_channels,
        "time_dim": args.time_dim,
        "model_total_params": num_params,
        "model_trainable_params": num_params,  # All params are trainable
        "model_non_trainable_params": 0,
        "aug_mode": args.aug_mode,
        "aug_rot_x": bool_to_str(args.aug_rot_x),
        "aug_rot_y": bool_to_str(args.aug_rot_y),
        "aug_rot_z": bool_to_str(args.aug_rot_z),
        "aug_flip_x": bool_to_str(args.aug_flip_x),
        "aug_flip_y": bool_to_str(args.aug_flip_y),
        "aug_flip_z": bool_to_str(args.aug_flip_z),
        "aug_perturb": bool_to_str(args.aug_perturb),
        "perturb_prob": args.perturb_prob,
        "sample_every": args.sample_every,
        "n_samples": args.n_samples,
        "sample_steps": args.sample_steps if args.sample_steps is not None else args.T,
        "save_every": args.save_every,
        "val_loss_explosion_factor": args.val_loss_explosion_factor,
        "loss_function": loss_desc,
        "loss_reconstruction": loss_desc,  # For diffusion, reconstruction loss is the MSE loss
        "T": args.T,
        "beta_schedule": args.beta_schedule,
        "gradient_accumulation_steps": 1,  # Diffusion script doesn't support gradient accumulation, but keep for consistency
        "effective_batch_size": args.batch_size,  # Same as batch_size since no gradient accumulation
        "clear_cache_every": 0,  # Diffusion script doesn't support this, but keep for consistency
        "notes": args.notes if args.notes else "None",  # Experiment notes/comments
    }

    # Save initial metadata (key-value format)
    with open(csv2_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in initial_metadata.items():
            writer.writerow([k, v])
    console.print(
        f"[green]✓[/green] Created initial metadata file: [cyan]{csv2_path}[/cyan]"
    )

    # Save initial metadata (flat format)
    with open(csv3_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=initial_metadata.keys())
        writer.writeheader()
        writer.writerow(initial_metadata)
    console.print(
        f"[green]✓[/green] Created initial metadata (flat) file: [cyan]{csv3_path}[/cyan]"
    )

    # Define fieldnames for training_history.csv
    history_fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "is_best",
        "grad_norm_max",
        "grad_norm_mean",
        "grad_norm_before_clip_max",
        "grad_norm_before_clip_mean",
        "amp_scale_min",
        "amp_overflow_steps",
        "eps_pred_absmax_max",
        "update_ratio_max",
        "loss_tbin_0",
        "loss_tbin_1",
        "loss_tbin_2",
        "loss_tbin_3",
        "loss_tbin_4",
        "loss_tbin_5",
        "loss_tbin_6",
        "loss_tbin_7",
        "loss_tbin_8",
        "loss_tbin_9",
        "epoch_time_secs",
        "cumulative_time_secs",
        "occupancy_mean",
        "occupancy_std",
        "component_count_mean",
        "component_count_std",
        "margin_mean",
        "margin_std",
        "margin_p50",
        "margin_p95",
        "air_confidence_mean",
        "air_confidence_std",
        "frac_air_mean",
        "frac_air_std",
        "frac_log_mean",
        "frac_log_std",
        "frac_leaf_mean",
        "frac_leaf_std",
        "p_air_mean",
        "p_air_std",
        "p_log_mean",
        "p_log_std",
        "p_leaf_mean",
        "p_leaf_std",
        "conf_air_on_air_mean",
        "conf_air_on_air_std",
        "conf_log_on_log_mean",
        "conf_log_on_log_std",
        "conf_leaf_on_leaf_mean",
        "conf_leaf_on_leaf_std",
    ]
    
    # Create training_history.csv with header
    # If resuming, restore previous history from checkpoint if file doesn't exist
    if args.resume and os.path.exists(csv1_path):
        # File exists, will append new epochs
        console.print(
            f"[cyan]Training history file exists, will append new epochs: [cyan]{csv1_path}[/cyan]"
        )
    else:
        # Create new file with header
        with open(csv1_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fieldnames)
            writer.writeheader()
            # If resuming and file doesn't exist, restore previous history from checkpoint
            if args.resume and training_history:
                writer.writerows(training_history)
                console.print(
                    f"[cyan]Restored {len(training_history)} previous epochs from checkpoint[/cyan]"
                )
        console.print(
            f"[green]✓[/green] Created training history file: [cyan]{csv1_path}[/cyan]"
        )

    # Training loop
    console.print(f"\n[bold]Starting training...[/bold]")
    console.print(f"Total epochs: {args.epochs}, T={args.T}, batch_size={args.batch_size}")

    remaining_epochs = args.epochs - start_epoch
    
    # 根據 --no_progress 參數選擇進度顯示方式
    if args.no_progress:
        # 不使用 Rich Progress（適合 screen/tmux/nohup）
        progress_context = DummyProgress(console)
    else:
        # 使用 Rich Progress（預設模式）
        progress_context = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=1,  # 限制每秒最多更新 1 次
        )

    # Early stop (val loss explosion); updated inside training loop if triggered
    stopped_early_val_explosion = False
    training_stop_detail = ""

    with progress_context as progress:
        overall_task = progress.add_task(
            "[cyan]Training", total=remaining_epochs + 1
        )

        for epoch in range(start_epoch, args.epochs):
            epoch_t0 = time.time()
            total_steps = len(train_loader) + len(val_loader)
            epoch_task = progress.add_task(
                f"[green]Epoch {epoch+1}/{args.epochs} - Training",
                total=total_steps,
            )

            # Training
            model.train()
            train_losses = []
            
            # Debug mode: track if we should log all steps for this epoch
            debug_mode = False
            debug_trigger_reason = ""
            debug_csv_path = os.path.join(exp_dir, f"debug_epoch_{epoch+1:04d}.csv")
            debug_rows = []  # Collect debug rows for this epoch
            
            # Accumulate metrics for epoch-level logging
            grad_norms = []  # After clipping
            grad_norms_before_clip = []  # Before clipping
            pred_means = []
            pred_stds = []
            param_norms = []
            # New stability metrics
            amp_scales = []  # AMP scaler scale values
            amp_overflow_count = 0  # Count of overflow steps
            eps_pred_absmax_values = []  # Maximum absolute values of eps_pred
            update_ratios = []  # Parameter update ratios
            # Skip step tracking (for epoch-level summary)
            skipped_steps_count = 0
            skipped_reasons = []  # Track reasons for skipping
            # Loss grouped by timestep bins (GPU tensors to avoid per-step CPU sync)
            loss_tbin_sum = torch.zeros(10, device=device)
            loss_tbin_cnt = torch.zeros(10, device=device, dtype=torch.long)

            for batch_idx, (onehot, labels) in enumerate(train_loader):
                onehot = onehot.to(device)  # [B, 3, 16, 16, 16] in [0,1]

                # Convert one-hot from [0,1] to [-1,1] for better DDPM stability
                x_0 = onehot_to_centered(onehot)  # [B, 3, 16, 16, 16] in [-1,1]

                # Sample timesteps
                t = torch.randint(0, args.T, (onehot.shape[0],), device=device)

                # Forward diffusion: add noise
                x_t, eps = q_sample(x_0, t, betas.alpha_bar, device)

                # Predict noise
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    eps_pred = model(x_t, t)
                    
                    # Compute weighted loss
                    if loss_weights is not None:
                        # 1. Compute element-wise MSE (keep shape [B, C, D, H, W])
                        loss_elementwise = F.mse_loss(eps_pred, eps, reduction='none')
                        # 2. Apply channel weights
                        loss_weighted = loss_elementwise * loss_weights
                        # 3. Take mean as final loss
                        loss = loss_weighted.mean()
                    else:
                        # Standard MSE loss (uniform weighting)
                        loss = F.mse_loss(eps_pred, eps)
                    
                    # Compute loss by timestep bins (for stability analysis)
                    # Group timesteps into 10 bins: [0, T/10), [T/10, 2T/10), ..., [9T/10, T)
                    if loss_weights is not None:
                        loss_val_per_sample = loss_elementwise.mean(dim=(1, 2, 3, 4))
                    else:
                        loss_val_per_sample = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=(1, 2, 3, 4))
                
                # === CRITICAL: Move .item() calls outside autocast to avoid GPU synchronization ===
                # These operations trigger GPU sync and should be done outside autocast context
                # or only computed when needed (debug_mode or every N steps)
                
                # Compute eps_pred_absmax (stability metric) - only when needed
                # Detach first to avoid gradient tracking, then compute outside autocast
                # Always compute if in debug_mode, otherwise sample occasionally for epoch-level stats
                if debug_mode or batch_idx % 50 == 0:  # Sample every 50 steps for epoch-level stats
                    eps_pred_absmax = eps_pred.detach().abs().max().item()
                    # Append to list for epoch-level stats
                    eps_pred_absmax_values.append(eps_pred_absmax)
                else:
                    # For epoch-level max calculation, we can skip individual steps
                    # The max will still be accurate since we sample regularly
                    eps_pred_absmax = None  # Not computed this step
                    # Don't append to avoid affecting max calculation
                
                # Process loss_by_tbin on GPU using scatter_add_ (avoid per-sample CPU sync)
                # Group timesteps into 10 bins: [0, T/10), [T/10, 2T/10), ..., [9T/10, T)
                with torch.no_grad():
                    # Match original logic: int(t_val * 10 / T), clamped to [0, 9]
                    bin_idx = torch.clamp((t.float() * 10 / args.T).long(), 0, 9)  # [B] on GPU, values in [0, 9]
                    loss_val_per_sample_detached = loss_val_per_sample.detach()  # [B] on GPU
                    loss_tbin_sum.scatter_add_(0, bin_idx, loss_val_per_sample_detached)
                    ones = torch.ones_like(loss_val_per_sample_detached, dtype=loss_tbin_cnt.dtype)
                    loss_tbin_cnt.scatter_add_(0, bin_idx, ones)

                # Backward
                if scaler is not None:
                    # === AMP Mode: Correct order is critical ===
                    # 1. Scale loss and backward (gradients are scaled)
                    scaler.scale(loss).backward()
                    # 2. Unscale gradients (convert back to true gradient values)
                    #    MUST do this before computing grad_norm, otherwise norm will be
                    #    inflated by the scale factor (e.g., 8192x or 65536x)
                    scaler.unscale_(optimizer)
                else:
                    # Non-AMP mode: direct backward
                    loss.backward()
                
                # === CRITICAL: Compute gradient norm AFTER unscale (if AMP) and BEFORE clipping ===
                # This is the "true" gradient that will be used for update
                # In AMP mode, gradients are now unscaled, so norm reflects actual gradient magnitude
                # Use clip_grad_norm_ with max_norm=inf to get unclipped norm (more consistent with foreach optimization)
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
                grad_norm_before_clip_val = grad_norm_before_clip.item()
                
                # Check for nan/inf in loss and grad_norm (BEFORE clipping)
                loss_val = loss.item()
                has_nan_inf = (not math.isfinite(loss_val)) or (not math.isfinite(grad_norm_before_clip_val))
                
                # Check for grad_norm threshold (safety skip)
                # Threshold: grad_norm > 90 triggers safety skip to prevent unstable updates
                GRAD_NORM_SKIP_THRESHOLD = 90.0
                grad_norm_too_large = grad_norm_before_clip_val > GRAD_NORM_SKIP_THRESHOLD
                
                # Check for other anomaly conditions to trigger debug mode
                # Threshold: grad_norm > 50 is already suspicious, > 100 is definitely abnormal
                grad_norm_anomaly = (not math.isfinite(grad_norm_before_clip_val)) or (grad_norm_before_clip_val > 50.0)
                
                # Trigger debug mode if any anomaly detected
                if not debug_mode and (has_nan_inf or grad_norm_anomaly):
                    debug_mode = True
                    if has_nan_inf:
                        debug_trigger_reason = "loss_or_grad_nan_inf"
                    elif grad_norm_anomaly:
                        debug_trigger_reason = f"grad_norm_anomaly_{grad_norm_before_clip_val:.2f}"
                    console.print(f"[yellow]⚠[/yellow] [bold]Debug mode activated for epoch {epoch+1}[/bold] - Reason: {debug_trigger_reason}")
                
                # === CRITICAL: Skip step BEFORE clipping and BEFORE optimizer.step() ===
                # Skip conditions:
                # 1. has_nan_inf: Loss or grad_norm contains nan/inf
                # 2. grad_norm_too_large: Grad norm exceeds safety threshold (> 90)
                should_skip = has_nan_inf or grad_norm_too_large
                if should_skip:
                    # === Skip this step: completely prevent any parameter updates ===
                    # 
                    # What we do:
                    # 1. optimizer.zero_grad() - Clear gradients (prevents accumulation)
                    # 2. scaler.update() - Reset scaler state (CRITICAL: must call after unscale_())
                    # 3. Do NOT call optimizer.step() - Prevents parameter updates
                    # 4. continue - Skip all subsequent update logic
                    #
                    # What we ensure:
                    # - No gradient accumulation: zero_grad() clears all gradients
                    # - No optimizer state update: step() is not called, so AdamW momentum buffers remain unchanged
                    # - No parameter updates: step() is not called, so model parameters remain unchanged
                    # - Scaler state reset: scaler.update() resets the internal state after unscale_()
                    #
                    # Note: Optimizer internal state (e.g., AdamW momentum buffers) is NOT updated when
                    # we skip step(). This is correct behavior - we want to preserve historical state
                    # from previous valid steps, and only update it when we have valid gradients.
                    optimizer.zero_grad()
                    
                    # === CRITICAL: MUST call scaler.update() when skipping step after unscale_() ===
                    # After calling scaler.unscale_(), we MUST call scaler.update() to reset the scaler's
                    # internal state, even if we skip the optimizer.step(). Otherwise, the next iteration
                    # will fail with "unscale_() has already been called on this optimizer since the last update()".
                    # 
                    # Note: scaler.update() without scaler.step() will:
                    # 1. Reset the scaler's internal state (marking that unscale_() was handled)
                    # 2. NOT adjust the scale factor (scale only changes if step() detected overflow)
                    # 3. Allow the next iteration to proceed normally
                    if scaler is not None:
                        scaler.update()
                    
                    # Determine skip reason for logging
                    if has_nan_inf:
                        skip_reason = "nan/inf"
                    elif grad_norm_too_large:
                        skip_reason = f"grad_norm_too_large_{grad_norm_before_clip_val:.2f}"
                    else:
                        skip_reason = "unknown"
                    
                    # Log the problematic step
                    t_min = t.min().item()
                    t_max = t.max().item()
                    t_mean = t.float().mean().item()
                    console.print(
                        f"[red]⚠[/red] [bold]Skipping step: {skip_reason}[/bold] "
                        f"epoch={epoch+1}, batch={batch_idx}, "
                        f"loss={loss_val:.6e}, grad_norm_before_clip={grad_norm_before_clip_val:.6e}, "
                        f"timestep_range=[{t_min}, {t_max}], timestep_mean={t_mean:.1f}"
                    )
                    
                    # Track skipped steps for epoch-level logging
                    skipped_steps_count += 1
                    skipped_reasons.append(skip_reason)
                    
                    # Skip appending to train_losses and grad_norms
                    continue
                
                # === Now safe to clip gradients (only if we didn't skip) ===
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm if hasattr(args, 'max_grad_norm') else float('inf'))
                grad_norm_val = grad_norm.item()
                
                # === CRITICAL: Only clone parameters when needed (debug_mode) ===
                # Cloning all parameters for update_ratio calculation is very expensive for large models
                # and can affect performance and reproducibility. Only compute when necessary.
                # Note: Removed wandb_log_freq check since we use epoch-level logging only
                should_compute_update_ratio = debug_mode
                params_before = {}
                if should_compute_update_ratio:
                    # Save parameters before update (for update_ratio calculation)
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            params_before[name] = param.data.clone()
                
                # Normal step: update optimizer
                if scaler is not None:
                    # === AMP Mode: Correct order is critical ===
                    # 1. Get scale before step (to detect overflow)
                    old_scale = scaler.get_scale()
                    # 2. Step optimizer (if scaler detected inf/nan during unscale_, this will skip)
                    scaler.step(optimizer)
                    # 3. Update scaler state (reduces scale if inf/nan was detected)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    
                    # Record AMP scaler metrics
                    amp_scales.append(new_scale)
                    # Check if overflow occurred (scale decreased)
                    amp_overflow_this_step = (new_scale < old_scale)
                    if amp_overflow_this_step:
                        amp_overflow_count += 1
                        # Trigger debug mode if AMP overflow detected
                        if not debug_mode:
                            debug_mode = True
                            debug_trigger_reason = f"amp_overflow_scale_{old_scale:.2e}_to_{new_scale:.2e}"
                            console.print(f"[yellow]⚠[/yellow] [bold]Debug mode activated for epoch {epoch+1}[/bold] - Reason: {debug_trigger_reason}")
                else:
                    optimizer.step()
                    amp_overflow_this_step = False
                    old_scale = None
                    new_scale = None
                
                # Compute update_ratio: ||Δθ|| / ||θ|| (only when needed)
                update_ratio = None
                if should_compute_update_ratio:
                    update_norm = 0.0
                    param_norm = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in params_before:
                            param_update = param.data - params_before[name]
                            update_norm += param_update.norm(2).item() ** 2
                            param_norm += param.data.norm(2).item() ** 2
                    if param_norm > 0:
                        update_ratio = (update_norm ** 0.5) / (param_norm ** 0.5)
                        update_ratios.append(update_ratio)
                        # Trigger debug mode if update_ratio is too high
                        # Threshold: 0.05 (5%) is already high, 0.1 (10%) is definitely abnormal
                        if not debug_mode and update_ratio > 0.05:
                            debug_mode = True
                            debug_trigger_reason = f"update_ratio_high_{update_ratio:.6f}"
                            console.print(f"[yellow]⚠[/yellow] [bold]Debug mode activated for epoch {epoch+1}[/bold] - Reason: {debug_trigger_reason}")

                train_losses.append(loss_val)
                
                # Always record grad_norm for history.csv (both before and after clipping)
                grad_norms.append(grad_norm_val)  # After clipping
                grad_norms_before_clip.append(grad_norm_before_clip_val)  # Before clipping
                
                # Compute additional metrics for debug logging (only when needed)
                # Detach to avoid gradient tracking and compute outside autocast
                if debug_mode:
                    eps_pred_detached = eps_pred.detach()
                    eps_pred_mean = eps_pred_detached.mean().item()
                    eps_pred_std = eps_pred_detached.std().item()
                else:
                    eps_pred_mean = None
                    eps_pred_std = None
                
                # Compute clip coefficient (ratio of clipped vs original grad norm)
                clip_coef = grad_norm_val / grad_norm_before_clip_val if grad_norm_before_clip_val > 0 else 1.0
                was_clipped = clip_coef < 1.0
                
                # Compute param_norm (weight norm)
                param_norm_total = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        param_norm_total += p.data.norm(2).item() ** 2
                param_norm_total = param_norm_total ** 0.5
                
                # Get timestep statistics
                t_cpu = t.cpu().numpy()
                t_mean = float(t_cpu.mean())
                t_min = int(t_cpu.min())
                t_max = int(t_cpu.max())
                t_bin = min(int(t_mean * 10 / args.T), 9)
                
                # Get learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Get AMP scale
                amp_scale_current = new_scale if scaler is not None else None
                
                # Record debug information if in debug mode
                if debug_mode:
                    # Try to get data file information (approximate, since batch may contain multiple files)
                    data_id = f"batch_{batch_idx}"
                    # If possible, try to get file index (this is approximate due to augmentation)
                    try:
                        # For random augmentation, we can't easily track exact file, so use batch_idx
                        # For enumerate mode, we could compute file_idx, but it's complex
                        data_id = f"batch_{batch_idx}"
                    except:
                        pass
                    
                    # Calculate step identifier for debug logging (epoch-based)
                    current_global_step = epoch * len(train_loader) + batch_idx
                    
                    # Ensure eps_pred_absmax is computed in debug_mode
                    if eps_pred_absmax is None:
                        # Fallback: compute it now if not already computed
                        eps_pred_absmax = eps_pred.detach().abs().max().item()
                    
                    # Ensure update_ratio is computed in debug_mode (if not already computed)
                    # Note: This is expensive, but necessary for debug logging
                    if update_ratio is None:
                        # Need to compute update_ratio now (parameters already updated, so we can't get exact value)
                        # We'll compute a simplified version using current parameter norms
                        # This is less accurate but avoids the need to clone parameters again
                        update_ratio = 0.0  # Set to 0.0 as fallback (indicates not computed before update)
                    
                    debug_row = {
                        "global_step": current_global_step,
                        "epoch": epoch + 1,
                        "step_in_epoch": batch_idx,
                        "lr": current_lr,
                        "data_id": data_id,
                        "t_mean": t_mean,
                        "t_min": t_min,
                        "t_max": t_max,
                        "t_bin": t_bin,
                        "loss": loss_val,
                        "eps_pred_absmax": eps_pred_absmax,
                        "eps_pred_mean": eps_pred_mean if eps_pred_mean is not None else 0.0,
                        "eps_pred_std": eps_pred_std if eps_pred_std is not None else 0.0,
                        "grad_norm_before_clip": grad_norm_before_clip_val,
                        "grad_norm_after_clip": grad_norm_val,
                        "clip_coef": clip_coef,
                        "was_clipped": "TRUE" if was_clipped else "FALSE",
                        "update_ratio": update_ratio if update_ratio is not None else 0.0,
                        "param_norm": param_norm_total,
                        "amp_scale": amp_scale_current if amp_scale_current is not None else "",
                        "amp_overflow": "TRUE" if amp_overflow_this_step else "FALSE",
                    }
                    debug_rows.append(debug_row)
                
                # Accumulate metrics for epoch-level logging (sample occasionally to avoid overhead)
                # Sample every 50 steps for epoch-level statistics
                if batch_idx % 50 == 0:
                    # Predicted noise statistics (detach to avoid gradient tracking)
                    eps_pred_detached = eps_pred.detach()
                    eps_pred_mean = eps_pred_detached.mean().item()
                    eps_pred_std = eps_pred_detached.std().item()
                    
                    # Parameter norm
                    param_norm = 0.0
                    for p in model.parameters():
                        if p.requires_grad:
                            param_norm += p.data.norm(2).item() ** 2
                    param_norm = param_norm ** 0.5
                    
                    # Accumulate for epoch-level logging
                    pred_means.append(eps_pred_mean)
                    pred_stds.append(eps_pred_std)
                    param_norms.append(param_norm)
                
                progress.update(epoch_task, advance=1)

            train_loss = np.mean(train_losses)
            
            # Save debug CSV if debug mode was activated
            if debug_mode and debug_rows:
                debug_fieldnames = [
                    "global_step", "epoch", "step_in_epoch", "lr", "data_id",
                    "t_mean", "t_min", "t_max", "t_bin",
                    "loss", "eps_pred_absmax", "eps_pred_mean", "eps_pred_std",
                    "grad_norm_before_clip", "grad_norm_after_clip", "clip_coef", "was_clipped",
                    "update_ratio", "param_norm",
                    "amp_scale", "amp_overflow"
                ]
                try:
                    with open(debug_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=debug_fieldnames)
                        writer.writeheader()
                        writer.writerows(debug_rows)
                    console.print(f"[yellow]📊[/yellow] Debug CSV saved: [cyan]{debug_csv_path}[/cyan] ({len(debug_rows)} steps, triggered by: {debug_trigger_reason})")
                except Exception as e:
                    console.print(f"[red]⚠[/red] Failed to save debug CSV: {e}")

            # Validation
            progress.update(
                epoch_task,
                description=f"[yellow]Epoch {epoch+1}/{args.epochs} - Validation",
            )

            model.eval()
            val_losses = []
            val_pred_means = []
            val_pred_stds = []

            with torch.no_grad():
                for onehot, labels in val_loader:
                    onehot = onehot.to(device)  # [B, 3, 16, 16, 16] in [0,1]
                    
                    # Convert one-hot from [0,1] to [-1,1] for better DDPM stability
                    x_0 = onehot_to_centered(onehot)  # [B, 3, 16, 16, 16] in [-1,1]
                    
                    t = torch.randint(0, args.T, (onehot.shape[0],), device=device)
                    x_t, eps = q_sample(x_0, t, betas.alpha_bar, device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        eps_pred = model(x_t, t)
                        
                        # Compute weighted loss (same as training)
                        if loss_weights is not None:
                            loss_elementwise = F.mse_loss(eps_pred, eps, reduction='none')
                            loss_weighted = loss_elementwise * loss_weights
                            loss = loss_weighted.mean()
                        else:
                            loss = F.mse_loss(eps_pred, eps)

                    val_losses.append(loss.item())
                    val_pred_means.append(eps_pred.mean().item())
                    val_pred_stds.append(eps_pred.std().item())
                    progress.update(epoch_task, advance=1)

            val_loss = np.mean(val_losses)
            val_pred_mean = np.mean(val_pred_means)
            val_pred_std = np.mean(val_pred_stds)
            progress.remove_task(epoch_task)

            # Stop if val loss explodes vs best val seen so far (before this epoch's update)
            stop_for_val_explosion = False
            val_explosion_detail = ""
            fac = args.val_loss_explosion_factor
            if fac > 0:
                if not math.isfinite(val_loss):
                    stop_for_val_explosion = True
                    val_explosion_detail = f"val_loss is not finite ({val_loss})"
                elif math.isfinite(best_val_loss) and best_val_loss > 0 and val_loss > fac * best_val_loss:
                    stop_for_val_explosion = True
                    thr = fac * best_val_loss
                    val_explosion_detail = (
                        f"val_loss {val_loss:.6g} > {fac}×best_val_loss "
                        f"({best_val_loss:.6g}, threshold {thr:.6g})"
                    )

            # Logging
            epoch_secs = time.time() - epoch_t0
            cum_secs = cumulative_time_offset + (time.time() - global_t0)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Initialize sample metrics (will be filled if samples are generated)
            sample_occupancy_mean = ""
            sample_occupancy_std = ""
            sample_component_count_mean = ""
            sample_component_count_std = ""
            sample_margin_mean = ""
            sample_margin_std = ""
            sample_margin_p50 = ""
            sample_margin_p95 = ""
            sample_air_confidence_mean = ""
            sample_air_confidence_std = ""
            sample_frac_air_mean = ""
            sample_frac_air_std = ""
            sample_frac_log_mean = ""
            sample_frac_log_std = ""
            sample_frac_leaf_mean = ""
            sample_frac_leaf_std = ""
            sample_p_air_mean = ""
            sample_p_air_std = ""
            sample_p_log_mean = ""
            sample_p_log_std = ""
            sample_p_leaf_mean = ""
            sample_p_leaf_std = ""
            sample_conf_air_on_air_mean = ""
            sample_conf_air_on_air_std = ""
            sample_conf_log_on_log_mean = ""
            sample_conf_log_on_log_std = ""
            sample_conf_leaf_on_leaf_mean = ""
            sample_conf_leaf_on_leaf_std = ""

            # Generate samples for visualization (with error handling)
            # Do this BEFORE creating history_row so we can include metrics
            if (epoch + 1) % args.sample_every == 0:
                try:
                    total_samples = args.n_samples
                    sample_task = progress.add_task(
                        "[blue]Generating samples", total=total_samples
                    )
                    model.eval()
                    sample_occupancies = []
                    sample_component_counts = []
                    sample_margins = []  # Per-sample margin means
                    sample_margin_all_voxels = []  # All voxel margins across all samples (for percentiles)
                    sample_air_confidences = []  # Per-sample air confidence means
                    sample_frac_airs = []  # Per-sample air fraction
                    sample_frac_logs = []  # Per-sample log fraction
                    sample_frac_leaves = []  # Per-sample leaf fraction
                    sample_p_airs = []  # Per-sample global mean probability of air
                    sample_p_logs = []  # Per-sample global mean probability of log
                    sample_p_leaves = []  # Per-sample global mean probability of leaf
                    sample_conf_air_on_air = []  # Per-sample confidence of air where air is predicted
                    sample_conf_log_on_log = []  # Per-sample confidence of log where log is predicted
                    sample_conf_leaf_on_leaf = []  # Per-sample confidence of leaf where leaf is predicted
                    
                    # Sampling tracking data (if enabled)
                    sampling_tracking_data = {}  # {sample_idx: [list of tracking rows]}
                    for i in range(args.n_samples):
                        sampling_tracking_data[i] = []
                    
                    # Debug flag: enable debug for first tracking call only
                    debug_trunk = True  # Will be set to False after first call
                    
                    def track_sampling_callback(sample_idx, step_idx, t_int, x_current, x0_hat):
                        """Callback function to track sampling metrics at each tracking step.
                        
                        Computes metrics based on both x_t (current noisy state) and x_0_hat (predicted clean state).
                        """
                        nonlocal debug_trunk
                        
                        # === Metrics based on x_t (current noisy state) ===
                        # Convert from [-1,1] to [0,1] for argmax
                        x_onehot_like = centered_to_onehot(x_current.unsqueeze(0))[0]  # [3,16,16,16]
                        
                        # Get labels using argmax (no softmax)
                        labels_xt = x_onehot_like.argmax(dim=0).cpu().numpy().astype(np.uint8)  # [16,16,16]
                        
                        # Compute all metrics based on x_t
                        occ_rates_xt = compute_occupancy_rates(labels_xt)
                        comp_counts_xt = compute_component_counts_26neighbor(labels_xt)
                        trunk_info_xt = compute_trunk_breakage(labels_xt, debug=debug_trunk)
                        largest_log_ratio_xt = compute_largest_log_component_ratio(labels_xt)
                        
                        # === Metrics based on x_0_hat (predicted clean state) ===
                        # Convert from [-1,1] to [0,1] for argmax
                        x0hat_onehot_like = centered_to_onehot(x0_hat.unsqueeze(0))[0]  # [3,16,16,16]
                        
                        # Get labels using argmax (no softmax)
                        labels_x0hat = x0hat_onehot_like.argmax(dim=0).cpu().numpy().astype(np.uint8)  # [16,16,16]
                        
                        # Compute all metrics based on x_0_hat
                        occ_rates_x0hat = compute_occupancy_rates(labels_x0hat)
                        comp_counts_x0hat = compute_component_counts_26neighbor(labels_x0hat)
                        trunk_info_x0hat = compute_trunk_breakage(labels_x0hat, debug=False)  # Disable detailed debug for x0_hat
                        largest_log_ratio_x0hat = compute_largest_log_component_ratio(labels_x0hat)
                        
                        # Print summary debug info on first call only
                        if debug_trunk:
                            print(f"[TrunkDebug] First tracking call - sample={sample_idx}, step={step_idx}, t={t_int}")
                            print(f"[TrunkDebug] x_t: is_main_trunk_broken={trunk_info_xt['is_main_trunk_broken']}, is_broken={trunk_info_xt['is_broken']}, base_connected_size={trunk_info_xt['base_connected_size']}, total_wood={trunk_info_xt['total_wood_size']}")
                            print(f"[TrunkDebug] x_0_hat: is_main_trunk_broken={trunk_info_x0hat['is_main_trunk_broken']}, is_broken={trunk_info_x0hat['is_broken']}, base_connected_size={trunk_info_x0hat['base_connected_size']}, total_wood={trunk_info_x0hat['total_wood_size']}")
                            debug_trunk = False  # Disable after first call
                        
                        # Store tracking data (both x_t and x_0_hat metrics)
                        # Order: each x0hat metric immediately follows its corresponding xt metric for easy comparison
                        tracking_row = {
                            'sample_idx': sample_idx,
                            'step_idx': step_idx,
                            't_int': t_int,
                            # Occupancy metrics: xt followed by x0hat
                            'occupancy_non_air': occ_rates_xt['non_air'],
                            'occupancy_non_air_x0hat': occ_rates_x0hat['non_air'],
                            'occupancy_log': occ_rates_xt['log'],
                            'occupancy_log_x0hat': occ_rates_x0hat['log'],
                            'occupancy_leaf': occ_rates_xt['leaf'],
                            'occupancy_leaf_x0hat': occ_rates_x0hat['leaf'],
                            # Component count metrics: xt followed by x0hat
                            'components_non_air': comp_counts_xt['non_air'],
                            'components_non_air_x0hat': comp_counts_x0hat['non_air'],
                            'components_log': comp_counts_xt['log'],
                            'components_log_x0hat': comp_counts_x0hat['log'],
                            'components_leaf': comp_counts_xt['leaf'],
                            'components_leaf_x0hat': comp_counts_x0hat['leaf'],
                            # Trunk metrics: xt followed by x0hat
                            'trunk_is_main_trunk_broken': trunk_info_xt['is_main_trunk_broken'],
                            'trunk_is_main_trunk_broken_x0hat': trunk_info_x0hat['is_main_trunk_broken'],
                            'trunk_is_broken': trunk_info_xt['is_broken'],
                            'trunk_is_broken_x0hat': trunk_info_x0hat['is_broken'],
                            'trunk_base_connected_size': trunk_info_xt['base_connected_size'],
                            'trunk_base_connected_size_x0hat': trunk_info_x0hat['base_connected_size'],
                            'total_log_size': trunk_info_xt['total_wood_size'],
                            'total_log_size_x0hat': trunk_info_x0hat['total_wood_size'],
                            # Largest log component ratio: xt followed by x0hat
                            'largest_log_component_ratio': largest_log_ratio_xt,
                            'largest_log_component_ratio_x0hat': largest_log_ratio_x0hat,
                        }
                        sampling_tracking_data[sample_idx].append(tracking_row)
                    
                    with torch.no_grad():
                        # Sample voxels using reverse diffusion
                        sample_steps = args.sample_steps if args.sample_steps is not None else args.T
                        progress.console.print(f"[dim]  Starting reverse diffusion (n_steps={sample_steps})...[/dim]")
                        
                        # Check if tracking is enabled
                        track_every = getattr(args, 'sample_track_every', None)
                        track_callback = track_sampling_callback if track_every is not None else None
                        
                        x_0_samples = sample_voxels(
                            model,
                            betas,
                            shape=(args.n_samples, 3, 16, 16, 16),
                            device=device,
                            n_steps=sample_steps,
                            use_amp=use_amp,
                            track_every=track_every,
                            track_callback=track_callback,
                        )  # [n_samples, 3, 16, 16, 16] in [-1,1]

                        # Convert from [-1,1] to [0,1] for softmax
                        x_0_onehot_like = centered_to_onehot(x_0_samples)  # [n_samples, 3, 16, 16, 16] in [0,1]

                        # Save each sample and compute metrics
                        sample_images = []
                        for i in range(args.n_samples):
                            # Base path for this sample (defined outside try block for W&B access)
                            base = os.path.join(samples_dir, f"sample_e{epoch+1:04d}_{i:02d}")
                            
                            try:
                                # Get both centered and onehot-like versions
                                vol_center = x_0_samples[i].cpu()          # [-1,1], [3,16,16,16]
                                vol_linear = x_0_onehot_like[i].cpu()      # [0,1],  [3,16,16,16]
                                
                                # Generate labels using argmax (softmax is monotonic, so argmax order is unchanged)
                                # Note: softmax is computed separately below for probability-based metrics
                                labels_softmax = vol_linear.argmax(dim=0).numpy().astype(np.uint8)
                                
                                # Save npz using softmax labels (for compatibility)
                                npz_path = base + ".npz"
                                Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(npz_path, labels_softmax)
                                
                                # Save projection image (softmax then argmax)
                                save_labels_and_projections(
                                    labels_softmax, 
                                    base + "_softmax_then_argmax.png", 
                                    title_suffix=f" softmax_argmax E{epoch+1}#{i}",
                                    exp_name=args.exp_name
                                )
                                
                                progress.console.print(f"[dim]  Saved sample {i+1}/{args.n_samples}: {os.path.basename(base)}_softmax_then_argmax.png[/dim]")
                                
                                # Compute occupancy and component count (using softmax labels for consistency)
                                occupancy = compute_occupancy_rate(labels_softmax, air_class=0)
                                component_count = compute_component_count(labels_softmax, air_class=0)
                                
                                sample_occupancies.append(occupancy)
                                if component_count >= 0:  # Only add if scipy is available
                                    sample_component_counts.append(component_count)
                                
                                # Compute margin (confidence gap) and air confidence
                                # vol_linear is [3, 16, 16, 16] in [0,1] range
                                p = F.softmax(vol_linear, dim=0)  # [3, 16, 16, 16] probabilities
                                
                                # Get top-2 probabilities for each voxel
                                top2 = torch.topk(p, k=2, dim=0).values  # [2, 16, 16, 16]
                                margin_per_voxel = top2[0] - top2[1]  # [16, 16, 16] - confidence gap
                                
                                # Per-sample mean margin
                                margin_mean = margin_per_voxel.mean().item()
                                sample_margins.append(margin_mean)
                                
                                # Collect all voxel margins for percentile calculation
                                sample_margin_all_voxels.append(margin_per_voxel.flatten().cpu().numpy())
                                
                                # Air confidence (probability of air class)
                                air_conf_mean = p[0].mean().item()  # Average air probability
                                sample_air_confidences.append(air_conf_mean)
                                
                                # Compute class distribution (histogram)
                                # labels_softmax shape: [16, 16, 16], values in {0, 1, 2}
                                counts = np.bincount(labels_softmax.flatten(), minlength=3)  # [count_air, count_log, count_leaf]
                                fracs = counts / counts.sum()  # [frac_air, frac_log, frac_leaf]
                                
                                sample_frac_airs.append(fracs[0])
                                sample_frac_logs.append(fracs[1])
                                sample_frac_leaves.append(fracs[2])
                                
                                # Compute global mean probabilities for each class
                                # p shape: [3, 16, 16, 16]
                                p_air_mean = p[0].mean().item()
                                p_log_mean = p[1].mean().item()
                                p_leaf_mean = p[2].mean().item()
                                
                                sample_p_airs.append(p_air_mean)
                                sample_p_logs.append(p_log_mean)
                                sample_p_leaves.append(p_leaf_mean)
                                
                                # Compute confidence on predicted positions
                                # labels_softmax shape: [16, 16, 16], values in {0, 1, 2}
                                pred = labels_softmax  # Use softmax labels as prediction
                                
                                # Convert p to numpy for indexing
                                p_np = p.cpu().numpy()  # [3, 16, 16, 16]
                                
                                # Confidence of air where air is predicted
                                mask_air = (pred == 0)
                                if mask_air.sum() > 0:
                                    conf_air_on_air = p_np[0][mask_air].mean()
                                else:
                                    conf_air_on_air = 0.0  # No air predicted
                                
                                # Confidence of log where log is predicted
                                mask_log = (pred == 1)
                                if mask_log.sum() > 0:
                                    conf_log_on_log = p_np[1][mask_log].mean()
                                else:
                                    conf_log_on_log = 0.0  # No log predicted
                                
                                # Confidence of leaf where leaf is predicted
                                mask_leaf = (pred == 2)
                                if mask_leaf.sum() > 0:
                                    conf_leaf_on_leaf = p_np[2][mask_leaf].mean()
                                else:
                                    conf_leaf_on_leaf = 0.0  # No leaf predicted
                                
                                sample_conf_air_on_air.append(conf_air_on_air)
                                sample_conf_log_on_log.append(conf_log_on_log)
                                sample_conf_leaf_on_leaf.append(conf_leaf_on_leaf)
                            except Exception as e:
                                progress.console.print(f"[yellow]⚠[/yellow] Failed to save sample {i+1}: {e}")
                            
                            # Collect images for W&B (read after saving, outside the main try block)
                            if HAS_WANDB and args.use_wandb:
                                try:
                                    # Use softmax version as the main image for W&B
                                    png_path_softmax = base + "_softmax_then_argmax.png"
                                    # Wait a bit to ensure file is fully written
                                    time.sleep(0.1)
                                    if os.path.exists(png_path_softmax):
                                        # Use file path directly - more reliable than reading into memory
                                        sample_images.append(wandb.Image(png_path_softmax, caption=f"Epoch {epoch+1} Sample {i} (softmax)"))
                                    else:
                                        progress.console.print(f"[yellow]⚠[/yellow] Image file not found: {png_path_softmax}")
                                except Exception as e:
                                    progress.console.print(f"[yellow]⚠[/yellow] Failed to add image to W&B: {e}")
                                    import traceback
                                    progress.console.print(f"[dim]{traceback.format_exc()}[/dim]")
                            
                            progress.update(sample_task, advance=1)
                        
                        # Compute mean and std for history.csv
                        if sample_occupancies:
                            sample_occupancy_mean = f"{np.mean(sample_occupancies):.6f}"
                            sample_occupancy_std = f"{np.std(sample_occupancies):.6f}"
                        if sample_component_counts:
                            sample_component_count_mean = f"{np.mean(sample_component_counts):.6f}"
                            sample_component_count_std = f"{np.std(sample_component_counts):.6f}"
                        if sample_margins:
                            sample_margin_mean = f"{np.mean(sample_margins):.6f}"
                            sample_margin_std = f"{np.std(sample_margins):.6f}"
                            # Compute percentiles across all voxels
                            if sample_margin_all_voxels:
                                all_margins = np.concatenate(sample_margin_all_voxels)
                                sample_margin_p50 = f"{np.percentile(all_margins, 50):.6f}"
                                sample_margin_p95 = f"{np.percentile(all_margins, 95):.6f}"
                        if sample_air_confidences:
                            sample_air_confidence_mean = f"{np.mean(sample_air_confidences):.6f}"
                            sample_air_confidence_std = f"{np.std(sample_air_confidences):.6f}"
                        if sample_frac_airs:
                            sample_frac_air_mean = f"{np.mean(sample_frac_airs):.6f}"
                            sample_frac_air_std = f"{np.std(sample_frac_airs):.6f}"
                        if sample_frac_logs:
                            sample_frac_log_mean = f"{np.mean(sample_frac_logs):.6f}"
                            sample_frac_log_std = f"{np.std(sample_frac_logs):.6f}"
                        if sample_frac_leaves:
                            sample_frac_leaf_mean = f"{np.mean(sample_frac_leaves):.6f}"
                            sample_frac_leaf_std = f"{np.std(sample_frac_leaves):.6f}"
                        if sample_p_airs:
                            sample_p_air_mean = f"{np.mean(sample_p_airs):.6f}"
                            sample_p_air_std = f"{np.std(sample_p_airs):.6f}"
                        if sample_p_logs:
                            sample_p_log_mean = f"{np.mean(sample_p_logs):.6f}"
                            sample_p_log_std = f"{np.std(sample_p_logs):.6f}"
                        if sample_p_leaves:
                            sample_p_leaf_mean = f"{np.mean(sample_p_leaves):.6f}"
                            sample_p_leaf_std = f"{np.std(sample_p_leaves):.6f}"
                        if sample_conf_air_on_air:
                            sample_conf_air_on_air_mean = f"{np.mean(sample_conf_air_on_air):.6f}"
                            sample_conf_air_on_air_std = f"{np.std(sample_conf_air_on_air):.6f}"
                        if sample_conf_log_on_log:
                            sample_conf_log_on_log_mean = f"{np.mean(sample_conf_log_on_log):.6f}"
                            sample_conf_log_on_log_std = f"{np.std(sample_conf_log_on_log):.6f}"
                        if sample_conf_leaf_on_leaf:
                            sample_conf_leaf_on_leaf_mean = f"{np.mean(sample_conf_leaf_on_leaf):.6f}"
                            sample_conf_leaf_on_leaf_std = f"{np.std(sample_conf_leaf_on_leaf):.6f}"
                        
                        # Log sample metrics to W&B (epoch-level)
                        if HAS_WANDB and args.use_wandb:
                            log_dict = {
                                "epoch": epoch + 1,  # Ensure epoch is included for x-axis grouping
                                "samples/occupancy_mean": np.mean(sample_occupancies),
                                "samples/occupancy_std": np.std(sample_occupancies),
                            }
                            if sample_component_counts:
                                log_dict["samples/component_count_mean"] = np.mean(sample_component_counts)
                                log_dict["samples/component_count_std"] = np.std(sample_component_counts)
                            if sample_margins:
                                log_dict["samples/margin_mean"] = np.mean(sample_margins)
                                log_dict["samples/margin_std"] = np.std(sample_margins)
                                if sample_margin_all_voxels:
                                    all_margins = np.concatenate(sample_margin_all_voxels)
                                    log_dict["samples/margin_p50"] = np.percentile(all_margins, 50)
                                    log_dict["samples/margin_p95"] = np.percentile(all_margins, 95)
                            if sample_air_confidences:
                                log_dict["samples/air_confidence_mean"] = np.mean(sample_air_confidences)
                                log_dict["samples/air_confidence_std"] = np.std(sample_air_confidences)
                            if sample_frac_airs:
                                log_dict["samples/frac_air_mean"] = np.mean(sample_frac_airs)
                                log_dict["samples/frac_air_std"] = np.std(sample_frac_airs)
                            if sample_frac_logs:
                                log_dict["samples/frac_log_mean"] = np.mean(sample_frac_logs)
                                log_dict["samples/frac_log_std"] = np.std(sample_frac_logs)
                            if sample_frac_leaves:
                                log_dict["samples/frac_leaf_mean"] = np.mean(sample_frac_leaves)
                                log_dict["samples/frac_leaf_std"] = np.std(sample_frac_leaves)
                            if sample_p_airs:
                                log_dict["samples/p_air_mean"] = np.mean(sample_p_airs)
                                log_dict["samples/p_air_std"] = np.std(sample_p_airs)
                            if sample_p_logs:
                                log_dict["samples/p_log_mean"] = np.mean(sample_p_logs)
                                log_dict["samples/p_log_std"] = np.std(sample_p_logs)
                            if sample_p_leaves:
                                log_dict["samples/p_leaf_mean"] = np.mean(sample_p_leaves)
                                log_dict["samples/p_leaf_std"] = np.std(sample_p_leaves)
                            if sample_conf_air_on_air:
                                log_dict["samples/conf_air_on_air_mean"] = np.mean(sample_conf_air_on_air)
                                log_dict["samples/conf_air_on_air_std"] = np.std(sample_conf_air_on_air)
                            if sample_conf_log_on_log:
                                log_dict["samples/conf_log_on_log_mean"] = np.mean(sample_conf_log_on_log)
                                log_dict["samples/conf_log_on_log_std"] = np.std(sample_conf_log_on_log)
                            if sample_conf_leaf_on_leaf:
                                log_dict["samples/conf_leaf_on_leaf_mean"] = np.mean(sample_conf_leaf_on_leaf)
                                log_dict["samples/conf_leaf_on_leaf_std"] = np.std(sample_conf_leaf_on_leaf)
                            
                            # Log sample images
                            if sample_images:
                                log_dict["samples/projections"] = sample_images
                                progress.console.print(f"[dim]  Uploading {len(sample_images)} images to W&B...[/dim]")
                            else:
                                progress.console.print(f"[yellow]⚠[/yellow] No images collected for W&B (sample_images is empty)")
                            
                            # Log with epoch as step for epoch-level visualization
                            # Use commit=False to avoid updating global step, will be committed with epoch-level logging
                            wandb.log(log_dict, step=epoch + 1, commit=False)
                            
                        # Save sampling tracking data to CSV and W&B (if enabled)
                        track_every = getattr(args, 'sample_track_every', None)
                        if track_every is not None and sampling_tracking_data:
                            # Save tracking data to CSV files (one per sample)
                            for sample_idx in range(args.n_samples):
                                if sampling_tracking_data[sample_idx]:
                                    tracking_csv_path = os.path.join(
                                        samples_dir,
                                        f"tracking_e{epoch+1:04d}_s{sample_idx}.csv"
                                    )
                                    try:
                                        tracking_fieldnames = [
                                            'sample_idx', 'step_idx', 't_int',
                                            # Occupancy metrics: xt followed by x0hat
                                            'occupancy_non_air', 'occupancy_non_air_x0hat',
                                            'occupancy_log', 'occupancy_log_x0hat',
                                            'occupancy_leaf', 'occupancy_leaf_x0hat',
                                            # Component count metrics: xt followed by x0hat
                                            'components_non_air', 'components_non_air_x0hat',
                                            'components_log', 'components_log_x0hat',
                                            'components_leaf', 'components_leaf_x0hat',
                                            # Trunk metrics: xt followed by x0hat
                                            'trunk_is_main_trunk_broken', 'trunk_is_main_trunk_broken_x0hat',
                                            'trunk_is_broken', 'trunk_is_broken_x0hat',
                                            'trunk_base_connected_size', 'trunk_base_connected_size_x0hat',
                                            'total_log_size', 'total_log_size_x0hat',
                                            # Largest log component ratio: xt followed by x0hat
                                            'largest_log_component_ratio', 'largest_log_component_ratio_x0hat',
                                        ]
                                        with open(tracking_csv_path, "w", newline="") as f:
                                            writer = csv.DictWriter(f, fieldnames=tracking_fieldnames)
                                            writer.writeheader()
                                            writer.writerows(sampling_tracking_data[sample_idx])
                                        progress.console.print(f"[dim]  Saved tracking data: {os.path.basename(tracking_csv_path)} ({len(sampling_tracking_data[sample_idx])} steps)[/dim]")
                                    except Exception as e:
                                        progress.console.print(f"[yellow]⚠[/yellow] Failed to save tracking CSV for sample {sample_idx}: {e}")
                            
                            # Log tracking data to W&B (aggregate across all samples)
                            if HAS_WANDB and args.use_wandb:
                                # Collect all tracking steps across all samples
                                all_tracking_steps = []
                                for sample_idx in range(args.n_samples):
                                    all_tracking_steps.extend(sampling_tracking_data[sample_idx])
                                
                                if all_tracking_steps:
                                    # Group by step_idx to compute statistics across samples
                                    step_groups = {}
                                    for row in all_tracking_steps:
                                        step_idx = row['step_idx']
                                        if step_idx not in step_groups:
                                            step_groups[step_idx] = []
                                        step_groups[step_idx].append(row)
                                    
                                    # Log metrics for each tracked step
                                    # Use sampling_step as x-axis (defined via wandb.define_metric)
                                    # This avoids conflicts with epoch-level logging which uses step=epoch+1
                                    for step_idx in sorted(step_groups.keys()):
                                        step_rows = step_groups[step_idx]
                                        t_int = step_rows[0]['t_int']  # All rows in same step have same t_int
                                        
                                        # Compute mean/std across samples for this step
                                        log_dict_tracking = {
                                            "epoch": epoch + 1,
                                            "sampling_step": step_idx,  # This will be used as x-axis for tracking/* metrics
                                            "sampling_t": t_int,
                                        }
                                        
                                        # Occupancy rates (based on x_t)
                                        log_dict_tracking["tracking/occupancy_non_air_mean"] = np.mean([r['occupancy_non_air'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_non_air_std"] = np.std([r['occupancy_non_air'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_log_mean"] = np.mean([r['occupancy_log'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_log_std"] = np.std([r['occupancy_log'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_leaf_mean"] = np.mean([r['occupancy_leaf'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_leaf_std"] = np.std([r['occupancy_leaf'] for r in step_rows])
                                        
                                        # Component counts (based on x_t)
                                        log_dict_tracking["tracking/components_non_air_mean"] = np.mean([r['components_non_air'] for r in step_rows])
                                        log_dict_tracking["tracking/components_log_mean"] = np.mean([r['components_log'] for r in step_rows])
                                        log_dict_tracking["tracking/components_leaf_mean"] = np.mean([r['components_leaf'] for r in step_rows])
                                        
                                        # Trunk breakage (based on x_t)
                                        log_dict_tracking["tracking/trunk_is_main_trunk_broken_frac"] = np.mean([1.0 if r['trunk_is_main_trunk_broken'] else 0.0 for r in step_rows])
                                        log_dict_tracking["tracking/trunk_is_broken_frac"] = np.mean([1.0 if r['trunk_is_broken'] else 0.0 for r in step_rows])
                                        log_dict_tracking["tracking/trunk_base_connected_size_mean"] = np.mean([r['trunk_base_connected_size'] for r in step_rows])
                                        log_dict_tracking["tracking/total_log_size_mean"] = np.mean([r['total_log_size'] for r in step_rows])
                                        
                                        # Largest log component ratio (based on x_t)
                                        log_dict_tracking["tracking/largest_log_component_ratio_mean"] = np.mean([r['largest_log_component_ratio'] for r in step_rows if r['largest_log_component_ratio'] >= 0])
                                        
                                        # Occupancy rates (based on x_0_hat)
                                        log_dict_tracking["tracking/occupancy_non_air_x0hat_mean"] = np.mean([r['occupancy_non_air_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_non_air_x0hat_std"] = np.std([r['occupancy_non_air_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_log_x0hat_mean"] = np.mean([r['occupancy_log_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_log_x0hat_std"] = np.std([r['occupancy_log_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_leaf_x0hat_mean"] = np.mean([r['occupancy_leaf_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/occupancy_leaf_x0hat_std"] = np.std([r['occupancy_leaf_x0hat'] for r in step_rows])
                                        
                                        # Component counts (based on x_0_hat)
                                        log_dict_tracking["tracking/components_non_air_x0hat_mean"] = np.mean([r['components_non_air_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/components_log_x0hat_mean"] = np.mean([r['components_log_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/components_leaf_x0hat_mean"] = np.mean([r['components_leaf_x0hat'] for r in step_rows])
                                        
                                        # Trunk breakage (based on x_0_hat)
                                        log_dict_tracking["tracking/trunk_is_main_trunk_broken_x0hat_frac"] = np.mean([1.0 if r['trunk_is_main_trunk_broken_x0hat'] else 0.0 for r in step_rows])
                                        log_dict_tracking["tracking/trunk_is_broken_x0hat_frac"] = np.mean([1.0 if r['trunk_is_broken_x0hat'] else 0.0 for r in step_rows])
                                        log_dict_tracking["tracking/trunk_base_connected_size_x0hat_mean"] = np.mean([r['trunk_base_connected_size_x0hat'] for r in step_rows])
                                        log_dict_tracking["tracking/total_log_size_x0hat_mean"] = np.mean([r['total_log_size_x0hat'] for r in step_rows])
                                        
                                        # Largest log component ratio (based on x_0_hat)
                                        log_dict_tracking["tracking/largest_log_component_ratio_x0hat_mean"] = np.mean([r['largest_log_component_ratio_x0hat'] for r in step_rows if r['largest_log_component_ratio_x0hat'] >= 0])
                                        
                                        # Log to W&B without specifying step (will use sampling_step as x-axis via define_metric)
                                        # Use commit=False to avoid updating global step, which would conflict with epoch-level logging
                                        # This allows sampling tracking to use its own x-axis (sampling_step) without affecting epoch-level step
                                        wandb.log(log_dict_tracking, commit=False)
                            
                            progress.console.print(f"[dim]  Sampling tracking: {sum(len(v) for v in sampling_tracking_data.values())} total tracking points recorded[/dim]")
                            
                        progress.console.print(f"[dim]  Sample occupancy: {np.mean(sample_occupancies):.4f} ± {np.std(sample_occupancies):.4f}[/dim]")
                        if sample_component_counts:
                            progress.console.print(f"[dim]  Sample components: {np.mean(sample_component_counts):.2f} ± {np.std(sample_component_counts):.2f}[/dim]")
                        if sample_margins:
                            all_margins = np.concatenate(sample_margin_all_voxels) if sample_margin_all_voxels else np.array(sample_margins)
                            progress.console.print(f"[dim]  Sample margin: {np.mean(sample_margins):.4f} ± {np.std(sample_margins):.4f} (p50={np.percentile(all_margins, 50):.4f}, p95={np.percentile(all_margins, 95):.4f})[/dim]")
                        if sample_air_confidences:
                            progress.console.print(f"[dim]  Air confidence: {np.mean(sample_air_confidences):.4f} ± {np.std(sample_air_confidences):.4f}[/dim]")
                    
                    progress.remove_task(sample_task)
                    progress.console.print(f"[green]✓[/green] Generated {args.n_samples} samples for epoch {epoch+1}")
                except Exception as e:
                    progress.console.print(f"[red]⚠[/red] Failed to generate samples: {e}")
                    import traceback
                    progress.console.print(f"[dim]{traceback.format_exc()}[/dim]")

            # Log epoch-level metrics to W&B
            if HAS_WANDB and args.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "val/pred_mean": val_pred_mean,
                    "val/pred_std": val_pred_std,
                }
                
                # Add accumulated training metrics if available
                if grad_norms:
                    log_dict["train/grad_norm"] = np.mean(grad_norms)
                    log_dict["train/grad_norm_max"] = max(grad_norms)
                if pred_means:
                    log_dict["train/pred_mean"] = np.mean(pred_means)
                if pred_stds:
                    log_dict["train/pred_std"] = np.mean(pred_stds)
                if param_norms:
                    log_dict["train/param_norm"] = np.mean(param_norms)
                
                # Add skipped steps summary
                if skipped_steps_count > 0:
                    log_dict["train/skipped_steps_count"] = skipped_steps_count
                    # Count most common skip reason
                    if skipped_reasons:
                        from collections import Counter
                        reason_counts = Counter(skipped_reasons)
                        most_common_reason, count = reason_counts.most_common(1)[0]
                        log_dict["train/skipped_most_common_reason"] = most_common_reason
                        log_dict["train/skipped_most_common_count"] = count
                
                # Add learning rate
                log_dict["train/lr"] = optimizer.param_groups[0]['lr']

                if stop_for_val_explosion:
                    log_dict["train/val_explosion_stop"] = 1
                    log_dict["train/stopped_early"] = 1
                    log_dict["train/stop_reason"] = "val_loss_explosion"
                    log_dict["train/val_explosion_detail"] = val_explosion_detail[:1024]
                
                # Use epoch number as step for epoch-level logging
                # All metrics are logged at epoch level, x-axis will show epochs (1, 2, 3, ...)
                # Use commit=True to commit all pending data (including sample metrics and sampling tracking with commit=False)
                wandb.log(log_dict, step=epoch + 1, commit=True)

            # Compute grad_norm_max and grad_norm_mean (after clipping)
            grad_norm_max = ""
            grad_norm_mean = ""
            if grad_norms:
                grad_norm_max = f"{max(grad_norms):.6f}"
                grad_norm_mean = f"{np.mean(grad_norms):.6f}"
            
            # Compute grad_norm_before_clip_max and grad_norm_before_clip_mean
            grad_norm_before_clip_max = ""
            grad_norm_before_clip_mean = ""
            if grad_norms_before_clip:
                grad_norm_before_clip_max = f"{max(grad_norms_before_clip):.6f}"
                grad_norm_before_clip_mean = f"{np.mean(grad_norms_before_clip):.6f}"
            
            # Compute AMP scaler metrics
            amp_scale_min = ""
            amp_overflow_steps = ""
            if amp_scales:
                amp_scale_min = f"{min(amp_scales):.6f}"
            if scaler is not None:
                amp_overflow_steps = str(amp_overflow_count)
            else:
                amp_overflow_steps = "0"  # No AMP, no overflow
            
            # Compute eps_pred_absmax_max
            eps_pred_absmax_max = ""
            if eps_pred_absmax_values:
                eps_pred_absmax_max = f"{max(eps_pred_absmax_values):.6f}"
            
            # Compute update_ratio_max
            update_ratio_max = ""
            if update_ratios:
                update_ratio_max = f"{max(update_ratios):.6f}"
            
            # Compute loss by timestep bins (only sync once at end of epoch)
            loss_tbin_cnt_cpu = loss_tbin_cnt.detach().cpu()  # Sync once
            loss_tbin_mean = (loss_tbin_sum / torch.clamp(loss_tbin_cnt.float(), min=1.0)).detach().cpu().tolist()
            loss_tbin_values = [""] * 10
            for bin_idx in range(10):
                if loss_tbin_cnt_cpu[bin_idx].item() > 0:
                    loss_tbin_values[bin_idx] = f"{loss_tbin_mean[bin_idx]:.6f}"
            
            history_row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "grad_norm_max": grad_norm_max,
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_before_clip_max": grad_norm_before_clip_max,
                "grad_norm_before_clip_mean": grad_norm_before_clip_mean,
                "amp_scale_min": amp_scale_min,
                "amp_overflow_steps": amp_overflow_steps,
                "eps_pred_absmax_max": eps_pred_absmax_max,
                "update_ratio_max": update_ratio_max,
                "loss_tbin_0": loss_tbin_values[0],
                "loss_tbin_1": loss_tbin_values[1],
                "loss_tbin_2": loss_tbin_values[2],
                "loss_tbin_3": loss_tbin_values[3],
                "loss_tbin_4": loss_tbin_values[4],
                "loss_tbin_5": loss_tbin_values[5],
                "loss_tbin_6": loss_tbin_values[6],
                "loss_tbin_7": loss_tbin_values[7],
                "loss_tbin_8": loss_tbin_values[8],
                "loss_tbin_9": loss_tbin_values[9],
                "epoch_time_secs": epoch_secs,
                "cumulative_time_secs": cum_secs,
                "is_best": "TRUE" if is_best else "FALSE",
                "occupancy_mean": sample_occupancy_mean,
                "occupancy_std": sample_occupancy_std,
                "component_count_mean": sample_component_count_mean,
                "component_count_std": sample_component_count_std,
                "margin_mean": sample_margin_mean,
                "margin_std": sample_margin_std,
                "margin_p50": sample_margin_p50,
                "margin_p95": sample_margin_p95,
                "air_confidence_mean": sample_air_confidence_mean,
                "air_confidence_std": sample_air_confidence_std,
                "frac_air_mean": sample_frac_air_mean,
                "frac_air_std": sample_frac_air_std,
                "frac_log_mean": sample_frac_log_mean,
                "frac_log_std": sample_frac_log_std,
                "frac_leaf_mean": sample_frac_leaf_mean,
                "frac_leaf_std": sample_frac_leaf_std,
                "p_air_mean": sample_p_air_mean,
                "p_air_std": sample_p_air_std,
                "p_log_mean": sample_p_log_mean,
                "p_log_std": sample_p_log_std,
                "p_leaf_mean": sample_p_leaf_mean,
                "p_leaf_std": sample_p_leaf_std,
                "conf_air_on_air_mean": sample_conf_air_on_air_mean,
                "conf_air_on_air_std": sample_conf_air_on_air_std,
                "conf_log_on_log_mean": sample_conf_log_on_log_mean,
                "conf_log_on_log_std": sample_conf_log_on_log_std,
                "conf_leaf_on_leaf_mean": sample_conf_leaf_on_leaf_mean,
                "conf_leaf_on_leaf_std": sample_conf_leaf_on_leaf_std,
            }
            training_history.append(history_row)

            # Immediately append to training_history.csv (with error handling)
            try:
                with open(csv1_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=history_fieldnames)
                    writer.writerow(history_row)
            except Exception as e:
                progress.console.print(f"[red]⚠[/red] Failed to write to CSV: {e}")

            # Save checkpoint (only best model to save space)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": training_history,
                "cumulative_time_secs": cum_secs,
                "args": vars(args),
            }

            # Save best model (when validation loss improves)
            if is_best:
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            # Save last checkpoint every epoch (for resume / latest state)
            torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
            # Save periodic checkpoint every save_every epochs
            if (epoch + 1) % args.save_every == 0:
                periodic_path = os.path.join(ckpt_dir, f"ckpt_e{epoch+1:04d}.pt")
                torch.save(ckpt, periodic_path)

            # Display epoch results (similar to train_VQVAE.py format)
            best_marker = " | ★ Best!" if is_best else ""
            ckpt_marker = " | 💾 Saved" if is_best else ""
            if (epoch + 1) % args.save_every == 0:
                ckpt_marker += f" | ckpt_e{epoch+1:04d}"
            progress.console.print(
                f"Epoch {epoch+1:03d}: train {train_loss:.6f} | "
                f"val {val_loss:.6f} | {fmt_secs(epoch_secs)}"
                f"{best_marker}{ckpt_marker}"
            )

            progress.update(overall_task, advance=1)

            if stop_for_val_explosion:
                progress.console.print(
                    Panel.fit(
                        "[bold red]Training terminated: validation loss explosion[/bold red]\n\n"
                        f"{val_explosion_detail}",
                        border_style="red",
                    )
                )
                stopped_early_val_explosion = True
                training_stop_detail = val_explosion_detail
                break

    total_secs = cumulative_time_offset + (time.time() - global_t0)
    train_end_time = datetime.now()

    # Append final metadata fields (append to end to maintain field order)
    final_metadata = {
        "training_end_time": train_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_checkpoint_path": os.path.join(ckpt_dir, "last.pt") if os.path.exists(os.path.join(ckpt_dir, "last.pt")) else "not_available",
        "best_val_loss": best_val_loss,
        "final_test_loss": None,  # Diffusion script doesn't have test set
        "total_training_time_secs": total_secs,
        "total_training_time_formatted": fmt_secs(total_secs),
        "stopped_early_val_explosion": "TRUE" if stopped_early_val_explosion else "FALSE",
        "training_stop_detail": training_stop_detail if training_stop_detail else "None",
    }

    # Append to key-value format metadata
    with open(csv2_path, "a", newline="") as f:
        writer = csv.writer(f)
        for k, v in final_metadata.items():
            writer.writerow([k, v])
    console.print(
        f"[green]✓[/green] Updated experiment metadata with final fields: [cyan]{csv2_path}[/cyan]"
    )

    # For flat format, read existing, merge, and rewrite
    existing_flat_metadata = {}
    if os.path.exists(csv3_path):
        with open(csv3_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                row = next(reader, None)
                if row:
                    existing_flat_metadata = row

    # Merge with final metadata
    all_flat_metadata = {**existing_flat_metadata, **final_metadata}

    # Rewrite flat format with all fields
    with open(csv3_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_flat_metadata.keys())
        writer.writeheader()
        writer.writerow(all_flat_metadata)
    console.print(
        f"[green]✓[/green] Updated experiment metadata (flat) with final fields: [cyan]{csv3_path}[/cyan]"
    )

    if stopped_early_val_explosion:
        console.print(f"\n[bold yellow]Training stopped early (val loss explosion).[/bold yellow]")
        console.print(f"[dim]{training_stop_detail}[/dim]")
    else:
        console.print(f"\n[bold green]Training completed![/bold green]")
    console.print(f"Best validation loss: {best_val_loss:.6f}")
    console.print(f"Total training time: {fmt_secs(total_secs)}")
    console.print(f"Checkpoints saved to: {ckpt_dir}")
    console.print(f"Samples saved to: {samples_dir}")
    console.print(f"Training history: {csv1_path}")
    console.print(f"Experiment metadata: {csv2_path}")
    
    # Finish W&B run
    if HAS_WANDB and args.use_wandb:
        wandb.finish()
        console.print(f"[green]✓[/green] W&B run completed")


def main():
    parser = argparse.ArgumentParser(description="Train 3D Voxel Diffusion Model")
    
    # Data
    parser.add_argument("--data_root", type=str, default=None, help="Path to data directory (with train/val subdirs)")
    parser.add_argument("--data_zip", type=str, default=None, help="Path to zip file containing train/val/test subdirs")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory for experiments")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (default: timestamp)")

    # Model
    parser.add_argument("--base_channels", type=int, default=64, help="Base number of channels")
    parser.add_argument("--time_dim", type=int, default=128, help="Time embedding dimension")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--preload", action="store_true", help="Preload all data into RAM")

    # Diffusion
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Beta schedule")
    parser.add_argument("--sample_every", type=int, default=10, help="Generate samples every N epochs")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate for visualization")
    parser.add_argument("--sample_steps", type=int, default=None, help="Number of sampling steps (default: T, can use fewer for speed)")
    parser.add_argument("--sample_track_every", type=int, default=None, help="Track sampling metrics every N steps during generation (saves to CSV and W&B)")

    # Augmentation
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="random",
        choices=["enumerate", "random"],
        help="Augmentation mode: 'random' (on-the-fly, default) or 'enumerate' (cartesian product, 512x data if all augs enabled)",
    )
    parser.add_argument("--aug_rot_x", action="store_true", help="Enable rotation around X axis")
    parser.add_argument("--aug_rot_y", action="store_true", help="Enable rotation around Y axis")
    parser.add_argument("--aug_rot_z", action="store_true", help="Enable rotation around Z axis")
    parser.add_argument("--aug_flip_x", action="store_true", help="Enable flip along X axis")
    parser.add_argument("--aug_flip_y", action="store_true", help="Enable flip along Y axis")
    parser.add_argument("--aug_flip_z", action="store_true", help="Enable flip along Z axis")
    parser.add_argument("--aug_perturb", action="store_true", help="Enable label perturbation")
    parser.add_argument("--perturb_prob", type=float, default=0.01, help="Probability of perturbing each voxel")

    # Loss weighting
    parser.add_argument(
        "--class_weights",
        type=str,
        default="none",
        help="Channel weights for loss: comma-separated values like '1.0,10.0,10.0' (air,log,leaf) or 'none' for uniform",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (filename in checkpoints/)")
    parser.add_argument("--save_every", type=int, default=50, help="Save periodic checkpoint every N epochs (e.g. ckpt_e0050.pt)")
    parser.add_argument(
        "--val_loss_explosion_factor",
        type=float,
        default=10.0,
        help="Stop training if val_loss > factor * best_val_loss (historical best before this epoch). "
        "Non-finite val_loss also stops. Use 0 to disable.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--notes", type=str, default="", help="Notes or comments for this experiment (will be saved to metadata)")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable rich progress bar (recommended for screen/tmux/nohup)"
    )
    
    # W&B monitoring
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases monitoring")
    parser.add_argument("--wandb_project", type=str, default="3d-voxel-diffusion", help="W&B project name")
    parser.add_argument("--wandb_log_freq", type=int, default=50, help="[DEPRECATED] Not used - all logging is epoch-level now")

    args = parser.parse_args()

    # Validate data arguments
    if not args.data_root and not args.data_zip:
        parser.error("Either --data_root or --data_zip is required.")
    if args.data_root and args.data_zip:
        parser.error("Use only one of --data_root or --data_zip.")

    # Handle zip file extraction
    temp_dir_holder = []
    if args.data_zip:
        console = Console()
        try:
            extract_dir, temp_dir = extract_zip_to_temp(args.data_zip, console=console)
            args.data_root = extract_dir
            temp_dir_holder.append(temp_dir)
            console.print(f"[cyan]Using extracted data from zip: {extract_dir}[/cyan]")
        except Exception as e:
            console.print(f"[red]Error extracting zip file: {e}[/red]")
            console.print(f"[red]Zip path: {args.data_zip}[/red]")
            raise

    # Set default experiment name
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_diffusion(args)
    
    # Note: temp_dir will be cleaned up automatically when temp_dir_holder goes out of scope


if __name__ == "__main__":
    main()
