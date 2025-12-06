#!/usr/bin/env python3
"""
Minecraft 3D VQ-VAE (32x32x32 voxels, 3 classes: 0=air,1=oak_log,2=oak_leaves)

This version uses an 8x8x8 discrete latent map with vector quantization:

  - Encoder produces continuous latent: z_e shape [B, latent_dim, 8,8,8]
  - Vector Quantizer maps z_e to discrete codebook indices
  - Two downsampling steps: 32 -> 16 -> 8
  - Decoder upsamples: 8 -> 16 -> 32
  - No skip connections (pure bottleneck architecture)

Loss: CrossEntropy(reconstruction) + VQ_loss(codebook + commitment)
"""

import argparse
import os
import math
import random
import time
import csv
import zipfile
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

# ----------------------
# Utility: seeding / timer
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


def extract_zip_to_temp(zip_path: str, console=None) -> tuple[str, tempfile.TemporaryDirectory]:
    """Extract zip file to a temporary directory and verify train/val/test."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    temp_dir = tempfile.TemporaryDirectory(prefix="train_vae_zip_")
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
        found_dirs = []
        for root, dirs, files in os.walk(extract_dir):
            if os.path.basename(root) in ["train", "val", "test"]:
                found_dirs.append(root)
        if not found_dirs:
            raise ValueError("Zip file must contain train/val/test subdirectories.")
        if len(found_dirs) >= 3:
            common_parent = os.path.commonpath(found_dirs)
            extract_dir = common_parent
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
# Dataset & Augmentations
# ----------------------


class VoxelDataset(Dataset):
    """Loads 32x32x32 int8 npz as class labels; returns (onehot[3,Z,Y,X], labels[Z,Y,X])."""

    def __init__(
        self,
        files,
        aug_mode: str = "enumerate",
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
                assert arr.shape == (32, 32, 32), f"Expected (32,32,32), got {arr.shape} from {path}"
                # 优化：使用 int8 而不是 int64，节省 8 倍内存
                # voxel 值只有 0,1,2，int8 足够（范围 -128 到 127）
                arr_int8 = arr.astype(np.int8)
                # 使用 torch.from_numpy 直接创建 tensor，避免额外复制
                # 注意：需要确保 tensor 不会被修改，所以后续需要 clone
                tensor = torch.from_numpy(arr_int8)
                # 使用 pin_memory 可以加速 GPU 传输（如果使用 CUDA）
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()
                self.data_cache.append(tensor)
                if console and (i + 1) % 1000 == 0:
                    console.print(f"  Loaded {i + 1}/{len(files)} files...")
            if console:
                # 更新内存计算：int8 而不是 int64
                mem_mb = len(files) * 32 * 32 * 32 * 1 / (1024**2)
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
            # 优化：只在需要时 clone，并且转换为 int64（one_hot 需要）
            # 注意：augmentation 操作（rot90, flip）会创建新 tensor，所以这里 clone 是必要的
            labels = self.data_cache[file_idx].clone().to(torch.int64)
        else:
            path = self.files[file_idx]
            with np.load(path, allow_pickle=False) as data:
                key = "arr_0" if "arr_0" in data else list(data.files)[0]
                arr = data[key]
            assert arr.shape == (32, 32, 32), f"Expected (32,32,32), got {arr.shape} from {path}"
            labels = torch.from_numpy(arr.astype(np.int64))

        labels = self._apply_rot_flip(labels, kx, ky, kz, fx, fy, fz)
        if self.aug_perturb and self.perturb_prob > 0:
            labels = self._perturb(labels)

        onehot = self._one_hot(labels, 3)
        return onehot, labels


# ----------------------
# Model: 3D VQ-VAE (8x8x8 discrete latent)
# ----------------------


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE codebook (non-EMA, straight-through).

    z_e: encoder output, shape [B, C, D, H, W]
    回傳:
      z_q: quantized latent, same shape as z_e
      loss: codebook + commitment loss
      perplexity: scalar
      indices: [B, D, H, W] (int64)
    """

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: [B, C, D, H, W]
        B, C, D, H, W = z.shape
        assert C == self.embedding_dim, f"z channels ({C}) must equal embedding_dim ({self.embedding_dim})"

        # [B, C, D, H, W] -> [B, D, H, W, C] -> [B*D*H*W, C]
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        flat_z = z_perm.view(-1, C)  # [N, C]

        # Compute distances to embedding vectors
        # ||z||^2 - 2 z·e + ||e||^2
        emb_weight = self.embedding.weight  # [K, C]
        emb_norm_sq = emb_weight.pow(2).sum(dim=1)  # [K]
        z_norm_sq = flat_z.pow(2).sum(dim=1, keepdim=True)  # [N, 1]

        distances = (
            z_norm_sq
            - 2 * flat_z @ emb_weight.t()
            + emb_norm_sq.unsqueeze(0)  # [1, K]
        )  # [N, K]

        # 對每一個位置找最近的 embedding index
        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        z_q = self.embedding(encoding_indices).view(B, D, H, W, C)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]

        # VQ Loss: ||sg[z_e] - e||^2 + beta * ||z_e - sg[e]||^2
        z_q_detached = z_q.detach()
        z_detached = z.detach()

        codebook_loss = F.mse_loss(z_q_detached, z)
        commitment_loss = F.mse_loss(z_q, z_detached)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through
        z_q = z + (z_q - z).detach()

        # perplexity
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings_one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape indices to spatial map [B, D, H, W]
        indices = encoding_indices.view(B, D, H, W)

        return z_q, loss, perplexity, indices


class Encoder3DVQVAE(nn.Module):
    """
    32 -> 16 -> 8, 然後 1x1 conv -> latent_dim channels
    沒有 skip connection.
    """

    def __init__(self, in_ch=3, base=64, latent_dim=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 4, stride=2, padding=1),  # 32 -> 16
            ResBlock3D(base * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base * 2, base * 4, 4, stride=2, padding=1),  # 16 -> 8
            ResBlock3D(base * 4),
        )
        self.to_latent = nn.Conv3d(base * 4, latent_dim, 1)

    def forward(self, x):
        h = self.enc1(x)   # [B, base, 32,32,32]
        h = self.enc2(h)   # [B, base*2, 16,16,16]
        h = self.enc3(h)   # [B, base*4,  8, 8, 8]
        z_e = self.to_latent(h)  # [B, latent_dim, 8,8,8]
        return z_e


class Decoder3DVQVAE(nn.Module):
    """
    8 -> 16 -> 32, 無 skip connections.
    輸出 logits: [B, out_ch, 32, 32, 32]
    """

    def __init__(self, out_ch=3, base=64, latent_dim=64):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(latent_dim, base * 4, 4, stride=2, padding=1)  # 8 -> 16
        self.rb1 = ResBlock3D(base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 4, stride=2, padding=1)    # 16 -> 32
        self.rb2 = ResBlock3D(base * 2)
        self.out_block = nn.Sequential(
            nn.Conv3d(base * 2, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, z_q):
        h = self.up1(z_q)
        h = self.rb1(h)
        h = self.up2(h)
        h = self.rb2(h)
        h = self.out_block(h)
        logits = self.out(h)
        return logits  # [B, out_ch, 32,32,32]


class VQVAE3D(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base=64,
        latent_dim=64,
        codebook_size=512,
        commitment_cost=0.25,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder3DVQVAE(in_ch=in_ch, base=base, latent_dim=latent_dim)
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=latent_dim,
                                  commitment_cost=commitment_cost)
        self.decoder = Decoder3DVQVAE(out_ch=out_ch, base=base, latent_dim=latent_dim)

    def forward(self, x):
        """
        x: [B, 3, 32,32,32]
        回傳:
          logits: [B,3,32,32,32]
          vq_loss: scalar
          perplexity: scalar
          indices: [B,8,8,8]
        """
        z_e = self.encoder(x)                           # [B, C, 8,8,8]
        z_q, vq_loss, perplexity, indices = self.vq(z_e)
        logits = self.decoder(z_q)
        return logits, vq_loss, perplexity, indices

    @torch.no_grad()
    def encode_to_indices(self, x):
        z_e = self.encoder(x)
        _, _, _, indices = self.vq(z_e)
        return indices  # [B,8,8,8]

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor):
        """
        indices: [B,8,8,8] (long)
        回傳 logits: [B,3,32,32,32]
        """
        B, D, H, W = indices.shape
        flat_idx = indices.view(-1)
        z_q = self.vq.embedding(flat_idx)  # [B*D*H*W, C]
        C = self.vq.embedding_dim
        z_q = z_q.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]
        logits = self.decoder(z_q)
        return logits

    @torch.no_grad()
    def sample_random(self, batch_size: int, device, spatial_size=(8, 8, 8)):
        D, H, W = spatial_size
        indices = torch.randint(
            low=0,
            high=self.vq.num_embeddings,
            size=(batch_size, D, H, W),
            device=device,
        )
        logits = self.decode_from_indices(indices)
        return logits, indices


# ----------------------
# Losses
# ----------------------


def parse_class_weights(arg: str):
    if arg is None or arg.strip().lower() == "none":
        return None
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise ValueError("--class_weights must have exactly 3 numbers or 'none'")
    vals = [float(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError("All class weights must be > 0")
    return torch.tensor(vals, dtype=torch.float32)


# ----------------------
# Save projections
# ----------------------


@torch.no_grad()
def save_volume_and_projections(vol_logits, out_npz, out_png):
    """
    vol_logits: [3,32,32,32] logits.
    Saves argmax labels npz + 3-view max projection PNG.
    """
    import matplotlib.pyplot as plt

    probs = F.softmax(vol_logits, dim=0)
    labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, labels)

    max_z = labels.max(axis=0)
    max_y = labels.max(axis=1)
    max_x = labels.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(max_z)
    axes[0].set_title("MaxProj Z (Y,X)")
    axes[1].imshow(max_y)
    axes[1].set_title("MaxProj Y (Z,X)")
    axes[2].imshow(max_x)
    axes[2].set_title("MaxProj X (Z,Y)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


@torch.no_grad()
def compute_codebook_usage(indices: torch.Tensor, codebook_size: int) -> dict:
    """
    计算 codebook usage 统计信息。
    
    Args:
        indices: [B, D, H, W] 或 [N] 的 tensor，包含 codebook indices
        codebook_size: codebook 的大小
    
    Returns:
        dict with:
            - active_codes: 被使用的 code 数量
            - usage_rate: 使用率 (active_codes / codebook_size)
            - code_counts: 每个 code 的使用次数（可选，用于 histogram）
    """
    if indices.dim() > 1:
        flat_indices = indices.view(-1).cpu().numpy()
    else:
        flat_indices = indices.cpu().numpy()
    
    unique_codes = np.unique(flat_indices)
    active_codes = len(unique_codes)
    usage_rate = active_codes / codebook_size
    
    # 计算每个 code 的使用次数
    code_counts = np.bincount(flat_indices, minlength=codebook_size)
    
    return {
        "active_codes": active_codes,
        "usage_rate": usage_rate,
        "code_counts": code_counts,
    }


@torch.no_grad()
def compute_latent_entropy(indices: torch.Tensor, codebook_size: int) -> float:
    """
    计算 latent grid 的熵（每个位置上的 latent index 熵）。
    
    Args:
        indices: [B, D, H, W] tensor，包含 codebook indices
        codebook_size: codebook 的大小
    
    Returns:
        average entropy across all spatial positions
    """
    B, D, H, W = indices.shape
    indices_np = indices.cpu().numpy()
    
    # 对每个空间位置 (d, h, w)，计算该位置在不同 batch 上的分布熵
    entropies = []
    for d in range(D):
        for h in range(H):
            for w in range(W):
                pos_indices = indices_np[:, d, h, w]  # [B]
                # 计算该位置的分布
                counts = np.bincount(pos_indices, minlength=codebook_size)
                probs = counts / len(pos_indices)
                probs = probs[probs > 0]  # 只考虑非零概率
                if len(probs) > 0:
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(entropy)
    
    return np.mean(entropies) if entropies else 0.0


@torch.no_grad()
def compute_per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> dict:
    """
    计算每个类别的准确率。
    
    Args:
        logits: [B, C, D, H, W] 预测 logits
        labels: [B, D, H, W] 真实标签
        num_classes: 类别数量
    
    Returns:
        dict with per-class accuracy and overall accuracy
    """
    preds = logits.argmax(dim=1)  # [B, D, H, W]
    
    # Flatten
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()
    
    # 总体准确率
    overall_acc = (preds_flat == labels_flat).mean()
    
    # 每个类别的准确率
    per_class_acc = {}
    per_class_counts = {}
    for c in range(num_classes):
        mask = labels_flat == c
        if mask.sum() > 0:
            class_acc = (preds_flat[mask] == labels_flat[mask]).mean()
            per_class_acc[f"class_{c}_acc"] = class_acc
            per_class_counts[f"class_{c}_count"] = mask.sum()
        else:
            per_class_acc[f"class_{c}_acc"] = 0.0
            per_class_counts[f"class_{c}_count"] = 0
    
    return {
        "overall_acc": overall_acc,
        **per_class_acc,
        **per_class_counts,
    }


@torch.no_grad()
def export_latent_indices_for_split(model, loader, device, out_dir: str, split_name: str, console=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idx = 0
    for batch_i, (onehot, labels) in enumerate(loader):
        onehot = onehot.to(device)
        indices = model.encode_to_indices(onehot)  # [B,8,8,8]
        B = indices.shape[0]
        for b in range(B):
            arr = indices[b].cpu().numpy().astype(np.int16)
            out_path = os.path.join(out_dir, f"{split_name}_{idx:06d}.npz")
            np.savez_compressed(out_path, indices=arr)
            idx += 1
        if console and (batch_i + 1) % 10 == 0:
            console.print(
                f"[dim]{split_name}: exported {idx} latent grids so far...[/dim]"
            )
    if console:
        console.print(f"[green]✓[/green] Exported {idx} {split_name} latent grids to {out_dir}")


# ----------------------
# Training / Evaluation
# ----------------------


def train(args, resume_checkpoint=None):
    global_t0 = time.time()
    train_start_time = datetime.now()

    # Device
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    console = Console()

    # Exp dir
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    best_checkpoint_path = os.path.join(exp_dir, f"best_{args.exp_name}.pt")
    last_checkpoint_path = os.path.join(exp_dir, f"last_{args.exp_name}.pt")

    if not resume_checkpoint:
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

    # Header
    header_text = (
        f"[bold cyan]Minecraft 3D VQ-VAE Training (8x8x8 discrete latent)[/bold cyan]\n"
        f"Experiment: [magenta]{args.exp_name}[/magenta]\n"
        f"Device: [yellow]{device}[/yellow]\n"
        f"Output: [cyan]{exp_dir}[/cyan]\n"
    )
    if resume_checkpoint:
        header_text += (
            f"Mode: [yellow]RESUME from epoch {resume_checkpoint['epoch']}[/yellow]\n"
        )
    header_text += f"Started: [green]{train_start_time.strftime('%Y-%m-%d %H:%M:%S')}[/green]"
    console.print(Panel.fit(header_text, border_style="cyan"))

    # Files
    train_files = sorted(glob(os.path.join(args.data_root, "train", "*.npz")))
    val_files = sorted(glob(os.path.join(args.data_root, "val", "*.npz")))
    test_files = sorted(glob(os.path.join(args.data_root, "test", "*.npz")))

    assert len(train_files) > 0, "No train .npz found"
    assert len(val_files) > 0, "No val .npz found"
    assert len(test_files) > 0, "No test .npz found"

    n_train, n_val, n_test = len(train_files), len(val_files), len(test_files)
    console.print(
        f"\n[bold]Dataset:[/bold] {n_train} train, {n_val} val, {n_test} test (total {n_train+n_val+n_test})"
    )

    # Datasets
    train_ds = VoxelDataset(
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
    val_ds = VoxelDataset(
        val_files,
        aug_mode="random",
        aug_perturb=False,
        preload=args.preload,
        console=console,
    )
    test_ds = VoxelDataset(
        test_files,
        aug_mode="random",
        aug_flip_x=False,
        aug_flip_y=False,
        aug_flip_z=False,
        aug_rot_x=False,
        aug_rot_y=False,
        aug_rot_z=False,
        aug_perturb=False,
        preload=args.preload,
        console=console,
    )

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=use_pin_memory,
        pin_memory=use_pin_memory,
    )

    # Class weights
    class_weights_base = parse_class_weights(args.class_weights)
    if class_weights_base is None:
        console.print("[bold]Class weights:[/bold] NONE (uniform)")

        def get_weight_tensor(_: torch.Tensor):
            return None

    else:
        console.print(f"[bold]Class weights:[/bold] {class_weights_base.tolist()}")
        console.print(
            f"[dim]  air={class_weights_base[0]:.4f}, "
            f"log={class_weights_base[1]:.4f}, "
            f"leaf={class_weights_base[2]:.4f}[/dim]"
        )
        class_weights_fp32 = class_weights_base.to(device=device, dtype=torch.float32)
        class_weights_fp16 = class_weights_fp32.to(dtype=torch.float16)
        try:
            class_weights_bf16 = class_weights_fp32.to(dtype=torch.bfloat16)
        except RuntimeError:
            class_weights_bf16 = None

        def get_weight_tensor(reference: torch.Tensor):
            dtype = reference.dtype
            if dtype == torch.float16:
                return class_weights_fp16
            if dtype == torch.bfloat16 and class_weights_bf16 is not None:
                return class_weights_bf16
            return class_weights_fp32

    console.print(
        f"[bold]Augmentation:[/bold] mode={args.aug_mode}, "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )
    console.print(f"[bold]VQ-VAE:[/bold] codebook_size={args.codebook_size}, latent_dim={args.latent_dim}, vq_beta={args.vq_beta}\n")

    # Model
    model = VQVAE3D(
        in_ch=3,
        out_ch=3,
        base=args.base,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        commitment_cost=0.25,  # Fixed commitment cost inside VQ module
    ).to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    console.print(
        f"[bold]Model parameters:[/bold] total={total_params:,}, "
        f"trainable={trainable_params:,}, non-trainable={non_trainable_params:,}"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = (
        torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    )

    if device.type == "cuda":
        if use_amp:
            console.print("[bold]AMP:[/bold] [green]ENABLED[/green]")
        else:
            console.print("[bold]AMP:[/bold] [yellow]DISABLED[/yellow]")
    else:
        console.print("[bold]AMP:[/bold] [dim]N/A[/dim]")

    os.makedirs(exp_dir, exist_ok=True)
    samples_dir = os.path.join(exp_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    start_epoch = 1
    best_val = math.inf
    training_history = []
    cumulative_time_offset = 0.0

    # Resume
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val = resume_checkpoint["best_val"]
        training_history = resume_checkpoint.get("training_history", [])
        cumulative_time_offset = resume_checkpoint.get("cumulative_time_secs", 0.0)

        if "rng_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["rng_state"])
        if "cuda_rng_state" in resume_checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state"])
        if "numpy_rng_state" in resume_checkpoint:
            np.random.set_state(resume_checkpoint["numpy_rng_state"])
        if "python_rng_state" in resume_checkpoint:
            random.setstate(resume_checkpoint["python_rng_state"])
        if scaler is not None and "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])

        console.print(
            f"[green]✓[/green] Resumed from epoch {resume_checkpoint['epoch']}, best_val={best_val:.4f}"
        )
        console.print(
            f"[cyan]Continuing training from epoch {start_epoch} to {args.epochs}[/cyan]\n"
        )

    # Prepare metadata paths
    csv1_path = os.path.join(exp_dir, f"training_history_{args.exp_name}.csv")
    csv2_path = os.path.join(exp_dir, f"experiment_metadata_{args.exp_name}.csv")
    csv3_path = os.path.join(exp_dir, f"experiment_metadata_flat_{args.exp_name}.csv")

    # Helper function for boolean to string
    def bool_to_str(v):
        return "TRUE" if v else "FALSE"

    # Prepare loss function descriptions
    ce_desc = (
        f"CrossEntropyLoss(weight=[{class_weights_base[0]:.2f},"
        f" {class_weights_base[1]:.2f}, {class_weights_base[2]:.2f}])"
        if class_weights_base is not None
        else "CrossEntropyLoss(weight=None)"
    )
    loss_formula = (
        f"Loss = {ce_desc} + {args.vq_beta} * VQ_Loss(codebook + commitment)"
    )
    current_script = (
        Path(__file__).name if "__file__" in globals() else "interactive_session"
    )

    # Create initial metadata (all fields that can be determined before training)
    # Note: start_epoch is set correctly after resume check above
    initial_metadata = {
        "exp_name": args.exp_name,
        "resumed_from": args.resume if args.resume else "None",
        "start_epoch": start_epoch,
        "end_epoch": args.epochs,
        "training_start_time": train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "samples_directory": samples_dir,
        "data_root": args.data_root,
        "out_dir": args.out_dir,
        "exp_dir": exp_dir,
        "script_name": current_script,
        "n_train_files": n_train,
        "n_val_files": n_val,
        "n_test_files": n_test,
        "n_total_files": n_train + n_val + n_test,
        "train_dataset_size": len(train_ds),
        "val_dataset_size": len(val_ds),
        "test_dataset_size": len(test_ds),
        "class_weights": args.class_weights,
        "vq_beta": args.vq_beta,
        "codebook_size": args.codebook_size,
        "commitment_cost": 0.25,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "workers": args.workers,
        "seed": args.seed,
        "force_cpu": bool_to_str(args.cpu),
        "device": str(device),
        "amp_enabled": bool_to_str(use_amp),
        "no_amp": bool_to_str(args.no_amp),
        "preload": bool_to_str(args.preload),
        "base": args.base,
        "latent_dim": args.latent_dim,
        "latent_spatial_size": "8x8x8",
        "latent_spatial_size_d": 8,
        "latent_total_elements": args.latent_dim * 8 * 8 * 8,
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "model_non_trainable_params": non_trainable_params,
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
        "save_every": args.save_every,
        "loss_function": loss_formula,
        "loss_reconstruction": ce_desc,
        "loss_vq": f"VQ_Loss(codebook + {0.25} * commitment)",
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

    # Save initial metadata (flat format) - only initial fields
    with open(csv3_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=initial_metadata.keys())
        writer.writeheader()
        writer.writerow(initial_metadata)
    console.print(
        f"[green]✓[/green] Created initial metadata (flat) file: [cyan]{csv3_path}[/cyan]"
    )

    # Create training_history.csv with header
    # If resuming, restore previous history from checkpoint if file doesn't exist
    if resume_checkpoint and os.path.exists(csv1_path):
        # File exists, will append new epochs
        console.print(
            f"[cyan]Training history file exists, will append new epochs: [cyan]{csv1_path}[/cyan]"
        )
    else:
        # Create new file with header
        fieldnames = [
            "epoch",
            "train_loss",
            "train_ce",
            "train_vq",
            "val_loss",
            "val_ce",
            "val_vq",
            "val_perplexity",
            "val_active_codes",
            "val_codebook_usage_rate",
            "val_latent_entropy",
            "val_overall_acc",
            "val_class_0_acc",
            "val_class_1_acc",
            "val_class_2_acc",
            "epoch_time_secs",
            "cumulative_time_secs",
            "is_best",
        ]
        with open(csv1_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # If resuming and file doesn't exist, restore previous history from checkpoint
            if resume_checkpoint and training_history:
                writer.writerows(training_history)
                console.print(
                    f"[cyan]Restored {len(training_history)} previous epochs from checkpoint[/cyan]"
                )
        console.print(
            f"[green]✓[/green] Created training history file: [cyan]{csv1_path}[/cyan]"
        )

    def save_checkpoint(epoch, is_best=False, is_last=False):
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "training_history": training_history,
            "cumulative_time_secs": cumulative_time_offset
            + (time.time() - global_t0),
            "args": vars(args),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
        if is_last:
            torch.save(checkpoint, last_checkpoint_path)

    remaining_epochs = args.epochs - start_epoch + 1
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

        overall_task = progress.add_task(
            "[cyan]Training", total=remaining_epochs + 1
        )

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_t0 = time.time()
            total_steps = len(train_loader) + len(val_loader)
            epoch_task = progress.add_task(
                f"[green]Epoch {epoch}/{args.epochs} - Training",
                total=total_steps,
            )

            # -------- Train --------
            model.train()
            running = 0.0
            running_ce = 0.0
            running_vq = 0.0

            if epoch == start_epoch:
                class_counts = torch.zeros(3, dtype=torch.long, device=device)

            for batch_idx, (onehot, labels) in enumerate(train_loader):
                onehot = onehot.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if epoch == start_epoch:
                    for c in range(3):
                        class_counts[c] += (labels == c).sum()

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp
                ):
                    logits, vq_loss, perplexity, _ = model(onehot)

                    if epoch == start_epoch and batch_idx == 0:
                        lf = logits.float().detach()
                        console.print(
                            f"[dim]Logits: mean={lf.mean():.4f}, std={lf.std():.4f}, "
                            f"min={lf.min():.4f}, max={lf.max():.4f} | "
                            f"VQ perplexity={perplexity.item():.2f}[/dim]"
                        )

                    weight_tensor = get_weight_tensor(logits)
                    ce = F.cross_entropy(
                        logits,
                        labels.long(),
                        weight=weight_tensor,
                    )
                    loss = ce + args.vq_beta * vq_loss

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_size = labels.size(0)
                running += loss.item() * batch_size
                running_ce += ce.item() * batch_size
                running_vq += vq_loss.item() * batch_size
                progress.update(epoch_task, advance=1)

            train_loss = running / len(train_loader.dataset)
            train_ce = running_ce / len(train_loader.dataset)
            train_vq = running_vq / len(train_loader.dataset)

            if epoch == start_epoch:
                total_voxels = class_counts.sum().item()
                if total_voxels > 0:
                    console.print(
                        "[dim]Class distribution (train, first epoch):[/dim]"
                    )
                    console.print(
                        f"[dim]  air:   {class_counts[0].item():,} "
                        f"({100*class_counts[0].item()/total_voxels:.1f}%)[/dim]"
                    )
                    console.print(
                        f"[dim]  log:   {class_counts[1].item():,} "
                        f"({100*class_counts[1].item()/total_voxels:.1f}%)[/dim]"
                    )
                    console.print(
                        f"[dim]  leaves:{class_counts[2].item():,} "
                        f"({100*class_counts[2].item()/total_voxels:.1f}%)[/dim]"
                    )

            # -------- Val --------
            progress.update(
                epoch_task,
                description=f"[yellow]Epoch {epoch}/{args.epochs} - Validation",
            )

            model.eval()
            running = 0.0
            running_ce = 0.0
            running_vq = 0.0
            running_perplexity = 0.0
            
            # 增量统计变量
            code_counts = np.zeros(args.codebook_size, dtype=np.int64)  # 用于 codebook usage
            position_counts = {}  # 用于 latent entropy: {(d,h,w): {code: count}}
            correct_predictions = np.zeros(3, dtype=np.int64)  # 每个类别的正确预测数
            total_predictions = np.zeros(3, dtype=np.int64)  # 每个类别的总数
            
            with torch.no_grad():
                for onehot, labels in val_loader:
                    onehot = onehot.to(device)
                    labels = labels.to(device)
                    logits, vq_loss, perplexity, indices = model(onehot)
                    weight_tensor = get_weight_tensor(logits)
                    ce = F.cross_entropy(
                        logits,
                        labels.long(),
                        weight=weight_tensor,
                    )
                    loss = ce + args.vq_beta * vq_loss
                    batch_size = labels.size(0)
                    running += loss.item() * batch_size
                    running_ce += ce.item() * batch_size
                    running_vq += vq_loss.item() * batch_size
                    running_perplexity += perplexity.item() * batch_size
                    
                    # 增量计算指标（避免内存爆炸）
                    # 1. Codebook usage: 累积 code counts
                    indices_flat = indices.cpu().view(-1).numpy()
                    batch_code_counts = np.bincount(indices_flat, minlength=args.codebook_size)
                    code_counts += batch_code_counts
                    
                    # 2. Latent entropy: 累积每个位置的 code 分布
                    indices_np = indices.cpu().numpy()  # [B, 8, 8, 8]
                    B, D, H, W = indices_np.shape
                    for d in range(D):
                        for h in range(H):
                            for w in range(W):
                                pos_key = (d, h, w)
                                if pos_key not in position_counts:
                                    position_counts[pos_key] = np.zeros(args.codebook_size, dtype=np.int64)
                                pos_codes = indices_np[:, d, h, w]  # [B]
                                pos_counts = np.bincount(pos_codes, minlength=args.codebook_size)
                                position_counts[pos_key] += pos_counts
                    
                    # 3. Per-class accuracy: 累积正确预测数
                    preds = logits.argmax(dim=1).cpu()  # [B, 32, 32, 32]
                    labels_cpu = labels.cpu()
                    for c in range(3):
                        mask = labels_cpu == c
                        if mask.any():
                            correct = (preds[mask] == labels_cpu[mask]).sum().item()
                            total = mask.sum().item()
                            correct_predictions[c] += correct
                            total_predictions[c] += total
                    
                    progress.update(epoch_task, advance=1)
            
            val_loss = running / len(val_loader.dataset)
            val_ce = running_ce / len(val_loader.dataset)
            val_vq = running_vq / len(val_loader.dataset)
            val_perplexity = running_perplexity / len(val_loader.dataset)
            
            # 计算最终指标
            # 1. Codebook usage
            active_codes = (code_counts > 0).sum()
            val_active_codes = active_codes
            val_codebook_usage_rate = active_codes / args.codebook_size
            
            # 2. Latent entropy
            entropies = []
            for pos_key, counts in position_counts.items():
                total = counts.sum()
                if total > 0:
                    probs = counts[counts > 0] / total
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(entropy)
            val_latent_entropy = np.mean(entropies) if entropies else 0.0
            
            # 3. Per-class accuracy
            val_overall_acc = correct_predictions.sum() / total_predictions.sum() if total_predictions.sum() > 0 else 0.0
            val_class_0_acc = correct_predictions[0] / total_predictions[0] if total_predictions[0] > 0 else 0.0
            val_class_1_acc = correct_predictions[1] / total_predictions[1] if total_predictions[1] > 0 else 0.0
            val_class_2_acc = correct_predictions[2] / total_predictions[2] if total_predictions[2] > 0 else 0.0
            progress.remove_task(epoch_task)

            epoch_secs = time.time() - epoch_t0
            cum_secs = cumulative_time_offset + (time.time() - global_t0)
            is_best = val_loss < best_val

            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce": train_ce,
                "train_vq": train_vq,
                "val_loss": val_loss,
                "val_ce": val_ce,
                "val_vq": val_vq,
                "val_perplexity": val_perplexity,
                "val_active_codes": val_active_codes,
                "val_codebook_usage_rate": val_codebook_usage_rate,
                "val_latent_entropy": val_latent_entropy,
                "val_overall_acc": val_overall_acc,
                "val_class_0_acc": val_class_0_acc,
                "val_class_1_acc": val_class_1_acc,
                "val_class_2_acc": val_class_2_acc,
                "epoch_time_secs": epoch_secs,
                "cumulative_time_secs": cum_secs,
                "is_best": "TRUE" if is_best else "FALSE",
            }
            training_history.append(history_row)

            # Immediately append to training_history.csv
            with open(csv1_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=history_row.keys())
                writer.writerow(history_row)

            if is_best:
                best_val = val_loss
                save_checkpoint(epoch, is_best=True)

            save_last_checkpoint = (epoch % args.save_every == 0) or (
                epoch == args.epochs
            )
            if save_last_checkpoint:
                save_checkpoint(epoch, is_last=True)

            best_marker = " | ★ Best!" if is_best else ""
            ckpt_marker = " | 💾 Saved" if save_last_checkpoint else ""
            progress.console.print(
                f"Epoch {epoch:03d}: train {train_loss:.4f} (ce={train_ce:.4f}, vq={train_vq:.4f}) | "
                f"val {val_loss:.4f} (ce={val_ce:.4f}, vq={val_vq:.4f}) | {fmt_secs(epoch_secs)}"
                f"{best_marker}{ckpt_marker}"
            )
            progress.console.print(
                f"  Metrics: perplexity={val_perplexity:.2f} | "
                f"active_codes={val_active_codes}/{args.codebook_size} ({val_codebook_usage_rate*100:.1f}%) | "
                f"latent_entropy={val_latent_entropy:.4f}"
            )
            progress.console.print(
                f"  Accuracy: overall={val_overall_acc*100:.2f}% | "
                f"air={val_class_0_acc*100:.2f}% | "
                f"log={val_class_1_acc*100:.2f}% | "
                f"leaves={val_class_2_acc*100:.2f}%"
            )

            # -------- Samples: reconstruction + random codebook samples --------
            if epoch % args.sample_every == 0:
                sample_task = progress.add_task(
                    "[blue]Generating samples", total=args.n_samples + 1
                )
                model.eval()
                with torch.no_grad():
                    # Reconstruction sample (for comparison)
                    for onehot, labels in val_loader:
                        onehot = onehot.to(device)
                        logits, _, _, indices = model(onehot)
                        rec = logits[0].detach().cpu()

                        # 儲存重建 voxel
                        save_volume_and_projections(
                            rec,
                            os.path.join(
                                samples_dir,
                                f"rec_e{epoch}_{args.exp_name}.npz",
                            ),
                            os.path.join(
                                samples_dir,
                                f"rec_e{epoch}_{args.exp_name}.png",
                            ),
                        )
                        # 順便把對應的 latent indices 也存起來
                        rec_indices = indices[0].cpu().numpy().astype(np.int16)
                        np.savez_compressed(
                            os.path.join(
                                samples_dir,
                                f"rec_e{epoch}_{args.exp_name}_codes.npz",
                            ),
                            indices=rec_indices,
                        )

                        progress.update(sample_task, advance=1)
                        break

                    # Prior samples: random codebook indices
                    for i in range(args.n_samples):
                        logits_prior, prior_indices = model.sample_random(
                            batch_size=1, device=device, spatial_size=(8, 8, 8)
                        )
                        vol_prior = logits_prior[0].detach().cpu()
                        save_volume_and_projections(
                            vol_prior,
                            os.path.join(
                                samples_dir,
                                f"sample_e{epoch}_{i}_{args.exp_name}.npz",
                            ),
                            os.path.join(
                                samples_dir,
                                f"sample_e{epoch}_{i}_{args.exp_name}.png",
                            ),
                        )
                        # 儲存 latent indices，之後你也可以拿來看 random 結構的 code 分佈
                        prior_idx_np = prior_indices[0].cpu().numpy().astype(np.int16)
                        np.savez_compressed(
                            os.path.join(
                                samples_dir,
                                f"sample_e{epoch}_{i}_{args.exp_name}_codes.npz",
                            ),
                            indices=prior_idx_np,
                        )
                        progress.update(sample_task, advance=1)
                progress.remove_task(sample_task)

            progress.update(overall_task, advance=1)

        # -------- Final Test --------
        best_model_path = best_checkpoint_path
        if os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(
                    best_model_path,
                    map_location=device,
                    weights_only=False,
                )["model"]
            )
        model.eval()

        running = 0.0
        test_task = progress.add_task(
            "[yellow]Final Test", total=len(test_loader)
        )
        with torch.no_grad():
            for onehot, labels in test_loader:
                onehot = onehot.to(device)
                labels = labels.to(device)
                logits, vq_loss, perplexity, _ = model(onehot)
                weight_tensor = get_weight_tensor(logits)
                ce = F.cross_entropy(
                    logits,
                    labels.long(),
                    weight=weight_tensor,
                )
                loss = ce + args.vq_beta * vq_loss
                running += loss.item() * labels.size(0)
                progress.update(test_task, advance=1)
        test_loss = running / len(test_loader.dataset)
        progress.remove_task(test_task)
        progress.update(overall_task, advance=1)
        progress.console.print(f"Final Test: test loss {test_loss:.4f}")

    # Clean last checkpoint
    last_checkpoint_removed = False
    if os.path.exists(last_checkpoint_path):
        try:
            os.remove(last_checkpoint_path)
            last_checkpoint_removed = True
            console.print(
                f"[cyan]Removed last checkpoint to reduce storage: {last_checkpoint_path}[/cyan]"
            )
        except OSError as e:
            console.print(
                f"[yellow]Warning:[/yellow] Failed to delete last checkpoint ({e})."
            )

    total_secs = cumulative_time_offset + (time.time() - global_t0)
    train_end_time = datetime.now()

    # Summary table
    final_table = Table(
        title="[bold cyan]Training Summary[/bold cyan]", box=box.ROUNDED
    )
    final_table.add_column("Metric", style="cyan", no_wrap=True)
    final_table.add_column("Value", style="magenta")
    final_table.add_row("Best Val Loss", f"{best_val:.6f}")
    final_table.add_row("Final Test Loss", f"{test_loss:.6f}")
    final_table.add_row("Total Runtime", fmt_secs(total_secs))
    final_table.add_row("Epochs Trained", f"{start_epoch} - {args.epochs}")
    final_table.add_row(
        "Started", train_start_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    final_table.add_row(
        "Completed", train_end_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    final_table.add_row(
        "Best Model",
        best_checkpoint_path if os.path.exists(best_checkpoint_path) else "None",
    )
    final_table.add_row(
        "Last Checkpoint",
        "Deleted after completion"
        if last_checkpoint_removed
        else last_checkpoint_path
        if os.path.exists(last_checkpoint_path)
        else "None",
    )
    console.print("\n", final_table, "\n")

    # Training history CSV is already being written incrementally during training
    console.print(
        f"[green]✓[/green] Training history saved incrementally to [cyan]{csv1_path}[/cyan]"
    )

    # Append final metadata fields to metadata files (append to end to maintain field order)
    final_metadata = {
        "training_end_time": train_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_checkpoint_path": "deleted_after_completion"
        if last_checkpoint_removed
        else (
            last_checkpoint_path
            if os.path.exists(last_checkpoint_path)
            else "not_available"
        ),
        "best_val_loss": best_val,
        "final_test_loss": test_loss,
        "total_training_time_secs": total_secs,
        "total_training_time_formatted": fmt_secs(total_secs),
    }

    # Append to key-value format metadata
    with open(csv2_path, "a", newline="") as f:
        writer = csv.writer(f)
        for k, v in final_metadata.items():
            writer.writerow([k, v])
    console.print(
        f"[green]✓[/green] Updated experiment metadata with final fields: [cyan]{csv2_path}[/cyan]"
    )

    # For flat format, we need to read existing, merge, and rewrite
    # But to maintain compatibility, we'll append new columns
    # Read existing flat metadata
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

    # Export latent indices if requested (after test, before summary)
    if args.export_latent_indices and args.latent_out_dir:
        console.print(f"\n[cyan]Exporting latent indices to {args.latent_out_dir}[/cyan]")
        train_latent_dir = os.path.join(args.latent_out_dir, "train")
        val_latent_dir = os.path.join(args.latent_out_dir, "val")
        test_latent_dir = os.path.join(args.latent_out_dir, "test")

        export_latent_indices_for_split(model, train_loader, device, train_latent_dir, "train", console=console)
        export_latent_indices_for_split(model, val_loader, device, val_latent_dir, "val", console=console)
        export_latent_indices_for_split(model, test_loader, device, test_latent_dir, "test", console=console)

    console.print(
        Panel.fit("[bold green]Training Complete! 🎉[/bold green]", border_style="green")
    )


# ----------------------
# Main
# ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Tree 3D VQ-VAE (8x8x8 discrete latent)")

    parser.add_argument("--data_root", type=str, required=False)
    parser.add_argument("--data_zip", type=str, required=False)
    parser.add_argument("--out_dir", type=str, required=False)
    parser.add_argument("--exp_name", type=str, required=False)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--base",
        type=int,
        default=64,
        help="Base channels for encoder/decoder",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Channels of spatial latent at 8x8x8",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=512,
        help="VQ-VAE codebook size",
    )
    parser.add_argument(
        "--vq_beta",
        type=float,
        default=1.0,
        help="Weight for VQ loss term (codebook + commitment)",
    )

    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--aug_mode",
        type=str,
        default="enumerate",
        choices=["enumerate", "random"],
    )
    parser.add_argument("--aug_rot_x", action="store_true")
    parser.add_argument("--aug_rot_y", action="store_true")
    parser.add_argument("--aug_rot_z", action="store_true")
    parser.add_argument("--aug_flip_x", action="store_true")
    parser.add_argument("--aug_flip_y", action="store_true")
    parser.add_argument("--aug_flip_z", action="store_true")
    parser.add_argument("--aug_perturb", action="store_true")
    parser.add_argument("--perturb_prob", type=float, default=0.01)

    parser.add_argument("--class_weights", type=str, default="none")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--preload", action="store_true")

    parser.add_argument(
        "--export_latent_indices",
        action="store_true",
        help="After training, export latent indices for train/val/test to NPZ files",
    )
    parser.add_argument(
        "--latent_out_dir",
        type=str,
        default=None,
        help="Output directory for latent indices NPZ files (default: exp_dir/latent_indices)",
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    # Resume logic
    resume_checkpoint = None
    if args.resume:
        import sys

        resume_path = args.resume
        if os.path.isdir(resume_path):
            last_files = glob(os.path.join(resume_path, "last_*.pt"))
            best_files = glob(os.path.join(resume_path, "best_*.pt"))
            if last_files:
                resume_path = last_files[0]
            elif best_files:
                resume_path = best_files[0]
            else:
                print(f"[✗] No checkpoint found in {args.resume}")
                raise SystemExit(1)
            args.resume = resume_path

        if not os.path.exists(args.resume):
            print(f"[✗] Checkpoint not found: {args.resume}")
            raise SystemExit(1)

        print(f"[→] Loading checkpoint: {args.resume}")
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        checkpoint_args = resume_checkpoint.get("args", {})

        explicitly_set = set()
        i = 1
        while i < len(sys.argv):
            a = sys.argv[i]
            if a.startswith("--"):
                name = a[2:].replace("-", "_")
                explicitly_set.add(name)
                if (
                    i + 1 < len(sys.argv)
                    and not sys.argv[i + 1].startswith("--")
                ):
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        merged = checkpoint_args.copy()
        for k in vars(args):
            if k in explicitly_set:
                merged[k] = getattr(args, k)
        for k, v in merged.items():
            setattr(args, k, v)

        print("[✓] Loaded params from checkpoint")
        print(f"    • data_root: {args.data_root}")
        print(f"    • out_dir: {args.out_dir}")
        print(f"    • exp_name: {args.exp_name}")
        print(
            f"    • epoch: {resume_checkpoint['epoch']} → {args.epochs}"
        )
        if explicitly_set - {"resume"}:
            print(
                f"    • Overridden: {', '.join(sorted(explicitly_set - {'resume'}))}"
            )
        print()

    if not args.data_root and not args.data_zip and not args.resume:
        parser.error(
            "Either --data_root or --data_zip is required (unless resuming)."
        )
    if args.data_root and args.data_zip:
        parser.error("Use only one of --data_root or --data_zip.")
    if not args.out_dir and not args.resume:
        parser.error(
            "--out_dir is required (unless resuming from checkpoint with it set)."
        )
    if not args.exp_name and not args.resume:
        parser.error(
            "--exp_name is required (unless resuming with it set)."
        )

    temp_dir_holder = []
    if args.data_zip:
        console = Console()
        extract_dir, temp_dir = extract_zip_to_temp(
            args.data_zip, console=console
        )
        args.data_root = extract_dir
        temp_dir_holder.append(temp_dir)

    seed_everything(args.seed)
    train(args, resume_checkpoint=resume_checkpoint)