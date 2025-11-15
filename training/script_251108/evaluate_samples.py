#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
評估 3D U-Net VAE 的隨機採樣生成質量。

從先驗分布 N(0,I) 採樣 latent code z，然後使用 decoder 生成樣本，
並計算統計指標（occupancy, class distribution, shape metrics 等）。

輸出：
- 生成的 .npz 檔案和投影圖片
- 統計指標 CSV/JSON
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# ---------------------------------------------------------------------------
# Model definitions (copied from training/evaluation scripts)
# ---------------------------------------------------------------------------


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


class ShallowEncoder3DUNetVAE(nn.Module):
    """Down: 32 -> 16 (single downsample)."""

    def __init__(self, in_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 4, stride=2, padding=1),  # 32 -> 16
            ResBlock3D(base * 2),
        )
        self.mu = nn.Conv3d(base * 2, latent_dim, 1)
        self.logvar = nn.Conv3d(base * 2, latent_dim, 1)

    def forward(self, x, return_skips: bool = True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        mu = self.mu(e2)
        logvar = self.logvar(e2)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        skips = (e1, e2) if return_skips else None
        return mu, logvar, skips


class ShallowDecoder3DUNetVAE(nn.Module):
    """Up: 16 -> 32 with configurable skip connections (0-2 levels)."""

    def __init__(self, out_ch=3, base=64, latent_dim=256, skip_levels: int = 2):
        super().__init__()
        if skip_levels < 0 or skip_levels > 2:
            raise ValueError("skip_levels must be between 0 and 2")
        self.skip_levels = int(skip_levels)
        self.use_skip_connections = self.skip_levels > 0

        self.rb_latent = ResBlock3D(latent_dim)

        up_in_channels = latent_dim + (base * 2 if self.skip_levels >= 1 else 0)
        self.up = nn.ConvTranspose3d(up_in_channels, base * 2, 4, stride=2, padding=1)  # 16 -> 32
        self.rb_up = ResBlock3D(base * 2)

        out_in_channels = base * 2 + (base if self.skip_levels >= 2 else 0)
        self.out_block = nn.Sequential(
            nn.Conv3d(out_in_channels, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, z, skips=None):
        if self.use_skip_connections:
            if skips is None or len(skips) != 2:
                raise ValueError("ShallowDecoder3DUNetVAE expects (e1,e2) skips when skip_levels > 0.")
            e1, e2 = skips
        else:
            e1 = e2 = None

        h = self.rb_latent(z)

        if self.skip_levels >= 1 and e2 is not None:
            h = torch.cat([h, e2], dim=1)

        h = self.up(h)
        h = self.rb_up(h)

        if self.skip_levels >= 2 and e1 is not None:
            h = torch.cat([h, e1], dim=1)

        h = self.out_block(h)
        logits = self.out(h)
        return logits


class ShallowUNet3DVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256, skip_levels: int = 2):
        super().__init__()
        if skip_levels < 0 or skip_levels > 2:
            raise ValueError("skip_levels must be between 0 and 2")
        self.skip_levels = int(skip_levels)
        self.use_skip_connections = self.skip_levels > 0

        self.encoder = ShallowEncoder3DUNetVAE(in_ch, base, latent_dim)
        self.decoder = ShallowDecoder3DUNetVAE(out_ch, base, latent_dim, skip_levels=self.skip_levels)

    def forward(self, x):
        mu, logvar, skips = self.encoder(x, return_skips=self.use_skip_connections)
        logits = self.decoder(mu, skips if self.use_skip_connections else None)
        return logits, mu, logvar


class Encoder3DUNetVAE(nn.Module):
    def __init__(self, in_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 4, stride=2, padding=1),
            ResBlock3D(base * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base * 2, base * 4, 4, stride=2, padding=1),
            ResBlock3D(base * 4),
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(base * 4, base * 8, 4, stride=2, padding=1),
            ResBlock3D(base * 8),
        )
        self.mu = nn.Conv3d(base * 8, latent_dim, 1)
        self.logvar = nn.Conv3d(base * 8, latent_dim, 1)

    def forward(self, x, return_skips: bool = True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        mu = self.mu(e4)
        logvar = self.logvar(e4)
        skips = (e1, e2, e3) if return_skips else None
        return mu, logvar, skips


class Decoder3DUNetVAE(nn.Module):
    def __init__(self, out_ch=3, base=64, latent_dim=256, skip_levels: int = 3):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")

        self.base = base

        self.up1 = nn.ConvTranspose3d(latent_dim, base * 8, 4, stride=2, padding=1)
        self.rb1 = ResBlock3D(base * 8)

        self.up2 = nn.ConvTranspose3d(base * 8 + base * 4, base * 4, 4, stride=2, padding=1)
        self.rb2 = ResBlock3D(base * 4)

        self.up3 = nn.ConvTranspose3d(base * 4 + base * 2, base * 2, 4, stride=2, padding=1)
        self.rb3 = ResBlock3D(base * 2)

        self.out_block = nn.Sequential(
            nn.Conv3d(base * 2 + base, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

        self.skip_gates = [0, 0, 0]
        self.set_skip_levels(skip_levels)

    def requires_skips(self) -> bool:
        return any(g > 0 for g in self.skip_gates)

    def set_skip_levels(self, levels: int):
        levels = int(max(0, min(3, levels)))
        gates = [0, 0, 0]
        for i in range(levels):
            gates[i] = 1
        self.skip_gates = gates
        self.skip_levels = levels
        self.use_skip_connections = self.requires_skips()

    def _concat_or_zeros(self, h, skip_feat, expected_ch: int, use_gate: bool):
        if use_gate and (skip_feat is not None):
            return torch.cat([h, skip_feat], dim=1)
        zeros = torch.zeros(
            h.size(0),
            expected_ch,
            h.size(2),
            h.size(3),
            h.size(4),
            device=h.device,
            dtype=h.dtype,
        )
        return torch.cat([h, zeros], dim=1)

    def forward(self, z, skips=None):
        e1 = e2 = e3 = None
        if skips is not None:
            e1, e2, e3 = skips

        h = self.up1(z)
        h = self.rb1(h)

        h = self._concat_or_zeros(
            h,
            e3,
            expected_ch=self.base * 4,
            use_gate=bool(self.skip_gates[0]),
        )

        h = self.up2(h)
        h = self.rb2(h)

        h = self._concat_or_zeros(
            h,
            e2,
            expected_ch=self.base * 2,
            use_gate=bool(self.skip_gates[1]),
        )

        h = self.up3(h)
        h = self.rb3(h)

        h = self._concat_or_zeros(
            h,
            e1,
            expected_ch=self.base,
            use_gate=bool(self.skip_gates[2]),
        )

        h = self.out_block(h)
        logits = self.out(h)
        return logits


class UNet3DVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256, skip_levels: int = 3):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.encoder = Encoder3DUNetVAE(in_ch, base, latent_dim)
        self.decoder = Decoder3DUNetVAE(out_ch, base, latent_dim, skip_levels=skip_levels)
        self.skip_levels = self.decoder.skip_levels
        self.use_skip_connections = self.decoder.use_skip_connections

    def set_skip_levels(self, levels: int):
        self.decoder.set_skip_levels(levels)
        self.skip_levels = self.decoder.skip_levels
        self.use_skip_connections = self.decoder.use_skip_connections

    def forward(self, x):
        need_skips = self.decoder.requires_skips()
        mu, logvar, skips = self.encoder(x, return_skips=need_skips)
        logits = self.decoder(mu, skips if need_skips else None)
        return logits, mu, logvar


class MidEncoder3DUNetVAE(nn.Module):
    """Down: 32 -> 16 -> 8 (spatial latent 8^3)."""

    def __init__(self, in_ch=3, base=64, latent_dim=256):
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
        self.mu = nn.Conv3d(base * 4, latent_dim, 1)
        self.logvar = nn.Conv3d(base * 4, latent_dim, 1)

    def forward(self, x, return_skips: bool = True):
        e1 = self.enc1(x)  # 32^3
        e2 = self.enc2(e1)  # 16^3
        e3 = self.enc3(e2)  # 8^3
        mu = self.mu(e3)
        logvar = self.logvar(e3)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        skips = (e1, e2, e3) if return_skips else None
        return mu, logvar, skips


class MidDecoder3DUNetVAE(nn.Module):
    """Up: 8 -> 16 -> 32 with optional skip connections (levels 0-3)."""

    def __init__(self, out_ch=3, base=64, latent_dim=256, skip_levels: int = 3):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.skip_levels = int(skip_levels)
        self.use_skip_connections = self.skip_levels > 0

        self.rb_latent = ResBlock3D(latent_dim)

        up1_in_channels = latent_dim + (base * 4 if self.skip_levels >= 1 else 0)
        self.up1 = nn.ConvTranspose3d(up1_in_channels, base * 4, 4, stride=2, padding=1)
        self.rb_up1 = ResBlock3D(base * 4)

        up2_in_channels = base * 4 + (base * 2 if self.skip_levels >= 2 else 0)
        self.up2 = nn.ConvTranspose3d(up2_in_channels, base * 2, 4, stride=2, padding=1)
        self.rb_up2 = ResBlock3D(base * 2)

        out_in_channels = base * 2 + (base if self.skip_levels >= 3 else 0)
        self.out_block = nn.Sequential(
            nn.Conv3d(out_in_channels, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, z, skips=None):
        if self.use_skip_connections:
            if skips is None or len(skips) != 3:
                raise ValueError("MidDecoder3DUNetVAE expects (e1,e2,e3) skips when skip_levels > 0.")
            e1, e2, e3 = skips
        else:
            e1 = e2 = e3 = None

        h = self.rb_latent(z)

        if self.skip_levels >= 1 and e3 is not None:
            h = torch.cat([h, e3], dim=1)

        h = self.up1(h)
        h = self.rb_up1(h)

        if self.skip_levels >= 2 and e2 is not None:
            h = torch.cat([h, e2], dim=1)

        h = self.up2(h)
        h = self.rb_up2(h)

        if self.skip_levels >= 3 and e1 is not None:
            h = torch.cat([h, e1], dim=1)

        h = self.out_block(h)
        logits = self.out(h)
        return logits


class MidUNet3DVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256, skip_levels: int = 3):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.skip_levels = int(skip_levels)
        self.use_skip_connections = self.skip_levels > 0
        self.encoder = MidEncoder3DUNetVAE(in_ch, base, latent_dim)
        self.decoder = MidDecoder3DUNetVAE(out_ch, base, latent_dim, skip_levels=self.skip_levels)

    def forward(self, x):
        mu, logvar, skips = self.encoder(x, return_skips=self.use_skip_connections)
        logits = self.decoder(mu, skips if self.use_skip_connections else None)
        return logits, mu, logvar


# ---------------------------------------------------------------------------
# Checkpoint loading and model detection
# ---------------------------------------------------------------------------


def detect_vae_variant(state_dict):
    if not isinstance(state_dict, dict):
        return "deep"
    keys = set(state_dict.keys())
    has_enc3 = any(k.startswith("encoder.enc3") for k in keys)
    has_enc4 = any(k.startswith("encoder.enc4") for k in keys)
    if not has_enc3:
        return "shallow"
    if has_enc3 and not has_enc4:
        return "mid"
    return "deep"


def parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(val)


def resolve_skip_levels(args, default: int = 3) -> int:
    if not isinstance(args, dict):
        return default

    candidates = [
        args.get("skip_levels_effective"),
        args.get("skip_levels"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass

    if "no_skip_connections" in args and parse_bool(args["no_skip_connections"]):
        return 0
    if "use_skip_connections" in args:
        return 3 if parse_bool(args["use_skip_connections"]) else 0

    return default


def load_model(checkpoint_path, device, use_amp=True):
    """Load model from checkpoint and return model, variant name, and parameters."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        args = ckpt.get("args", {}) or {}
        base = int(args.get("base", 64))
        latent = int(args.get("latent_dim", 256))
    else:
        state_dict = ckpt
        base = 64
        latent = 256

    vae_variant = detect_vae_variant(state_dict)
    default_skip = 3
    if vae_variant == "shallow":
        default_skip = 2
    elif vae_variant == "mid":
        default_skip = 3

    skip_levels = resolve_skip_levels(args, default=default_skip)
    if skip_levels is None:
        skip_levels = default_skip

    # Build model
    if vae_variant == "shallow":
        skip_levels = int(max(0, min(2, int(skip_levels))))
        model = ShallowUNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        model_variant = "vae_shallow16"
        latent_shape = (16, 16, 16)

    elif vae_variant == "mid":
        skip_levels = int(max(0, min(3, int(skip_levels))))
        model = MidUNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        model_variant = "vae_mid8"
        latent_shape = (8, 8, 8)

    else:
        skip_levels = int(max(0, min(3, int(skip_levels))))
        model = UNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        model_variant = "vae_deep4"
        latent_shape = (4, 4, 4)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if not use_amp:
        model = model.float()

    params = {
        "base": base,
        "latent_dim": latent,
        "skip_levels": skip_levels,
        "variant": model_variant,
        "latent_shape": latent_shape,
    }

    return model, params


# ---------------------------------------------------------------------------
# Sample generation and visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_sample_with_skips(model, z, device, base=64, batch_size=1):
    """Generate samples from latent code z, handling skip connections for prior sampling."""
    if model.use_skip_connections:
        # For generation from prior, we need to provide zero-filled skips
        # to match the decoder's expected skip tensor shapes
        # Deep decoder can handle None skips and will zero-fill automatically
        if isinstance(model, UNet3DVAE):
            # Deep decoder: pass None, it will handle zero-filling
            logits = model.decoder(z, skips=None)
        elif isinstance(model, ShallowUNet3DVAE):
            # Shallow: e1 at 32^3 (base), e2 at 16^3 (base*2)
            e1 = torch.zeros(batch_size, base, 32, 32, 32, device=device, dtype=z.dtype)
            e2 = torch.zeros(batch_size, base * 2, 16, 16, 16, device=device, dtype=z.dtype)
            skips = (e1, e2)
            logits = model.decoder(z, skips=skips)
        elif isinstance(model, MidUNet3DVAE):
            # Mid: e1 at 32^3 (base), e2 at 16^3 (base*2), e3 at 8^3 (base*4)
            e1 = torch.zeros(batch_size, base, 32, 32, 32, device=device, dtype=z.dtype)
            e2 = torch.zeros(batch_size, base * 2, 16, 16, 16, device=device, dtype=z.dtype)
            e3 = torch.zeros(batch_size, base * 4, 8, 8, 8, device=device, dtype=z.dtype)
            skips = (e1, e2, e3)
            logits = model.decoder(z, skips=skips)
        else:
            # Fallback: try without skips
            logits = model.decoder(z, skips=None)
    else:
        logits = model.decoder(z, skips=None)
    
    return logits


@torch.no_grad()
def save_volume_and_projections(vol_logits, out_npz, out_png):
    """
    vol_logits: [3,32,32,32] logits.
    Saves argmax labels npz + 3-view max projection PNG.
    """
    probs = F.softmax(vol_logits, dim=0)
    labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, labels)

    max_z = labels.max(axis=0)
    max_y = labels.max(axis=1)
    max_x = labels.max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(max_z, vmin=0, vmax=2, cmap="viridis")
    axes[0].set_title("MaxProj Z (Y,X)")
    axes[1].imshow(max_y, vmin=0, vmax=2, cmap="viridis")
    axes[1].set_title("MaxProj Y (Z,X)")
    axes[2].imshow(max_x, vmin=0, vmax=2, cmap="viridis")
    axes[2].set_title("MaxProj X (Z,Y)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def occupancy_ratio(labels):
    """Compute occupancy ratio (non-air voxels / total voxels)."""
    return float((labels != 0).sum()) / labels.size


def class_distribution(labels, num_classes=3):
    """Compute class distribution."""
    counts = np.bincount(labels.flatten(), minlength=num_classes)
    total = labels.size
    ratios = counts / total if total > 0 else np.zeros(num_classes)
    return counts.tolist(), [float(r) for r in ratios]


def aabb_spans(mask):
    """Compute bounding box dimensions and aspect ratios."""
    if not mask.any():
        return (0, 0, 0), (0.0, 0.0)
    coords = np.array(np.where(mask))
    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1)
    Lz, Ly, Lx = zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1
    aryx = Ly / Lx if Lx > 0 else 0.0
    azx = Lz / Lx if Lx > 0 else 0.0
    return (Lz, Ly, Lx), (aryx, azx)


def compute_sample_metrics(labels):
    """Compute comprehensive metrics for a generated sample."""
    metrics = {}

    # Occupancy
    metrics["occupancy"] = occupancy_ratio(labels)

    # Class distribution
    counts, ratios = class_distribution(labels)
    metrics["count_air"] = int(counts[0])
    metrics["count_log"] = int(counts[1])
    metrics["count_leaves"] = int(counts[2])
    metrics["ratio_air"] = ratios[0]
    metrics["ratio_log"] = ratios[1]
    metrics["ratio_leaves"] = ratios[2]

    # Non-air bounding box
    nonair_mask = labels != 0
    if nonair_mask.any():
        (Lz, Ly, Lx), (aryx, azx) = aabb_spans(nonair_mask)
        metrics["bbox_z"] = int(Lz)
        metrics["bbox_y"] = int(Ly)
        metrics["bbox_x"] = int(Lx)
        metrics["bbox_aryx"] = aryx
        metrics["bbox_azx"] = azx
        metrics["bbox_volume"] = int(Lz * Ly * Lx)
    else:
        metrics["bbox_z"] = 0
        metrics["bbox_y"] = 0
        metrics["bbox_x"] = 0
        metrics["bbox_aryx"] = 0.0
        metrics["bbox_azx"] = 0.0
        metrics["bbox_volume"] = 0

    # Per-class bounding boxes
    for class_idx, class_name in enumerate(["air", "log", "leaves"]):
        mask = labels == class_idx
        if mask.any():
            (Lz, Ly, Lx), (aryx, azx) = aabb_spans(mask)
            metrics[f"bbox_{class_name}_z"] = int(Lz)
            metrics[f"bbox_{class_name}_y"] = int(Ly)
            metrics[f"bbox_{class_name}_x"] = int(Lx)
            metrics[f"bbox_{class_name}_aryx"] = aryx
            metrics[f"bbox_{class_name}_azx"] = azx
        else:
            metrics[f"bbox_{class_name}_z"] = 0
            metrics[f"bbox_{class_name}_y"] = 0
            metrics[f"bbox_{class_name}_x"] = 0
            metrics[f"bbox_{class_name}_aryx"] = 0.0
            metrics[f"bbox_{class_name}_azx"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_samples(
    model_path,
    output_dir,
    device,
    n_samples=100,
    batch_size=8,
    seed=42,
    use_amp=True,
    save_individual=True,
    progress=None,
    task=None,
):
    """Generate samples and compute statistics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    model, model_params = load_model(model_path, device, use_amp=use_amp)
    latent_shape = model_params["latent_shape"]
    latent_dim = model_params["latent_dim"]

    # Generate samples
    all_metrics = []
    samples_dir = Path(output_dir)
    if save_individual:
        samples_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        batch_size_actual = batch_end - batch_start

        # Sample z from prior N(0, I)
        z = torch.randn(
            batch_size_actual,
            latent_dim,
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            device=device,
        )

        base = model_params["base"]
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = generate_sample_with_skips(model, z, device, base=base, batch_size=batch_size_actual)
        else:
            logits = generate_sample_with_skips(model, z, device, base=base, batch_size=batch_size_actual)

        # Process each sample in batch
        for i in range(batch_size_actual):
            sample_idx = batch_start + i
            sample_logits = logits[i : i + 1]  # [1, 3, 32, 32, 32]

            # Save individual sample if requested
            if save_individual:
                sample_npz = samples_dir / f"sample_{sample_idx:04d}.npz"
                sample_png = samples_dir / f"sample_{sample_idx:04d}.png"
                save_volume_and_projections(sample_logits[0], sample_npz, sample_png)

            # Compute metrics
            probs = F.softmax(sample_logits[0], dim=0)
            labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
            metrics = compute_sample_metrics(labels)
            metrics["sample_idx"] = sample_idx
            all_metrics.append(metrics)

        if progress is not None and task is not None:
            progress.update(task, advance=batch_size_actual)

    # Aggregate statistics
    aggregated = {}
    if all_metrics:
        # Mean and std for numeric metrics
        numeric_keys = [
            "occupancy",
            "ratio_air",
            "ratio_log",
            "ratio_leaves",
            "bbox_aryx",
            "bbox_azx",
            "bbox_volume",
            "bbox_log_aryx",
            "bbox_log_azx",
            "bbox_leaves_aryx",
            "bbox_leaves_azx",
        ]
        for key in numeric_keys:
            values = [m[key] for m in all_metrics]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))

        # Sums for counts
        count_keys = ["count_air", "count_log", "count_leaves"]
        for key in count_keys:
            aggregated[f"{key}_sum"] = int(sum(m[key] for m in all_metrics))
            aggregated[f"{key}_mean"] = float(np.mean([m[key] for m in all_metrics]))

        # Bounding box dimensions (means)
        bbox_keys = ["bbox_z", "bbox_y", "bbox_x"]
        for key in bbox_keys:
            values = [m[key] for m in all_metrics if m[key] > 0]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
            else:
                aggregated[f"{key}_mean"] = 0.0
                aggregated[f"{key}_std"] = 0.0

    return {
        "model": str(model_path),
        "model_params": model_params,
        "n_samples": n_samples,
        "seed": seed,
        "individual_metrics": all_metrics,
        "aggregated_stats": aggregated,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    console = Console()
    parser = argparse.ArgumentParser(description="Evaluate VAE sampling quality.")
    parser.add_argument("--models_dir", required=True, help="Directory containing checkpoints (.pt/.pth).")
    parser.add_argument("--out_dir", required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate per model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP for evaluation.")
    parser.add_argument(
        "--skip_patterns",
        nargs="*",
        default=["last"],
        help="Filename patterns to skip (case-insensitive). Default: ['last']",
    )
    parser.add_argument("--no_save_individual", action="store_true", help="Don't save individual sample files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        console.print(
            Panel.fit(
                f"[bold red]錯誤：模型目錄不存在或不是目錄[/bold red]\n"
                f"路徑: [yellow]{models_dir}[/yellow]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    model_files = sorted(list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth")))
    total_models_before_filter = len(model_files)

    skip_patterns = args.skip_patterns if args.skip_patterns else []
    if skip_patterns:
        skip_lower = [p.lower() for p in skip_patterns]
        model_files = [
            f for f in model_files if not any(pattern in f.name.lower() for pattern in skip_lower)
        ]
    filtered_count = total_models_before_filter - len(model_files)

    if not model_files:
        skip_info = f"（已過濾 {filtered_count} 個包含 {skip_patterns} 的檔案）" if skip_patterns else ""
        console.print(
            Panel.fit(
                f"[bold red]錯誤：模型目錄中沒有找到模型檔案[/bold red]\n"
                f"目錄: [yellow]{models_dir}[/yellow]\n"
                f"搜尋模式: *.pt, *.pth（遞迴搜尋）\n"
                f"{skip_info}",
                border_style="red",
            )
        )
        raise SystemExit(1)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_txt = out_dir / f"sample_eval_{ts}.txt"
    out_csv = out_dir / f"sample_eval_{ts}.csv"

    amp_status = "禁用" if not use_amp else "啟用"
    save_status = "是" if not args.no_save_individual else "否"
    filter_info = (
        f"（已過濾 {filtered_count} 個包含 {skip_patterns} 的檔案）" if filtered_count > 0 else ""
    )

    console.print(
        Panel.fit(
            f"[bold cyan]VAE 隨機採樣評估[/bold cyan]\n"
            f"模型數量: [yellow]{len(model_files)}[/yellow] {filter_info}\n"
            f"跳過模式: [yellow]{skip_patterns if skip_patterns else '無'}[/yellow]\n"
            f"每模型樣本數: [yellow]{args.n_samples}[/yellow]\n"
            f"批次大小: [yellow]{args.batch_size}[/yellow]\n"
            f"裝置: [yellow]{device}[/yellow]\n"
            f"CUDA AMP: [yellow]{amp_status}[/yellow]\n"
            f"保存個別樣本: [yellow]{save_status}[/yellow]\n"
            f"種子: [yellow]{args.seed}[/yellow]\n"
            f"輸出目錄: [cyan]{out_dir}[/cyan]",
            border_style="cyan",
        )
    )

    # CSV header
    header = [
        "model",
        # Model params
        "variant",
        "base",
        "latent_dim",
        "skip_levels",
        # Sample params
        "n_samples",
        "seed",
        # Aggregated stats
        "occupancy_mean",
        "occupancy_std",
        "occupancy_min",
        "occupancy_max",
        "ratio_air_mean",
        "ratio_air_std",
        "ratio_log_mean",
        "ratio_log_std",
        "ratio_leaves_mean",
        "ratio_leaves_std",
        "count_air_mean",
        "count_log_mean",
        "count_leaves_mean",
        "bbox_z_mean",
        "bbox_z_std",
        "bbox_y_mean",
        "bbox_y_std",
        "bbox_x_mean",
        "bbox_x_std",
        "bbox_aryx_mean",
        "bbox_aryx_std",
        "bbox_azx_mean",
        "bbox_azx_std",
        "bbox_volume_mean",
        "bbox_volume_std",
        # Error column
        "error",
    ]

    with out_txt.open("w", encoding="utf-8") as ft, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fc:
        writer = csv.writer(fc)
        writer.writerow(header)

        run_header = {
            "models_root": str(models_dir),
            "num_models": len(model_files),
            "num_models_filtered": filtered_count,
            "skip_patterns": skip_patterns,
            "n_samples_per_model": args.n_samples,
            "batch_size": args.batch_size,
            "device": str(device),
            "use_amp": use_amp,
            "save_individual": not args.no_save_individual,
            "seed": args.seed,
            "timestamp": ts,
        }
        console.print(json.dumps({"run_header": run_header}, ensure_ascii=False))
        ft.write(json.dumps({"run_header": run_header}, ensure_ascii=False) + "\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            models_task = progress.add_task(
                "[cyan]評估模型中...", total=len(model_files)
            )

            for idx, model_path in enumerate(model_files, start=1):
                model_name = model_path.name
                progress.update(
                    models_task,
                    description=f"[cyan]評估模型 {idx}/{len(model_files)}: {model_name}",
                )

                try:
                    # Create subdirectory for this model's samples
                    model_samples_dir = out_dir / "samples" / model_path.stem
                    model_samples_dir.mkdir(parents=True, exist_ok=True)

                    sample_task = progress.add_task(
                        "[dim]生成樣本...",
                        total=args.n_samples,
                        visible=args.n_samples > 10,
                    )

                    result = evaluate_samples(
                        model_path,
                        model_samples_dir,  # Pass model-specific directory
                        device,
                        n_samples=args.n_samples,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        use_amp=use_amp,
                        save_individual=not args.no_save_individual,
                        progress=progress,
                        task=sample_task,
                    )

                    progress.remove_task(sample_task)

                    params = result.get("model_params", {})
                    stats = result.get("aggregated_stats", {})

                    console.print(
                        f"[green]✓[/green] [{idx}/{len(model_files)}] {model_name}\n"
                        f"  [dim]佔用率={stats.get('occupancy_mean', 0):.4f}±{stats.get('occupancy_std', 0):.4f} "
                        f"Air={stats.get('ratio_air_mean', 0):.4f} "
                        f"Log={stats.get('ratio_log_mean', 0):.4f} "
                        f"Leaves={stats.get('ratio_leaves_mean', 0):.4f}[/dim]"
                    )

                    # Write JSON line
                    ft.write(json.dumps(result, ensure_ascii=False) + "\n")

                    # Write CSV row
                    row = [
                        result["model"],
                        # Model params
                        params.get("variant", ""),
                        params.get("base", ""),
                        params.get("latent_dim", ""),
                        params.get("skip_levels", ""),
                        # Sample params
                        result.get("n_samples", ""),
                        result.get("seed", ""),
                        # Stats
                        stats.get("occupancy_mean", ""),
                        stats.get("occupancy_std", ""),
                        stats.get("occupancy_min", ""),
                        stats.get("occupancy_max", ""),
                        stats.get("ratio_air_mean", ""),
                        stats.get("ratio_air_std", ""),
                        stats.get("ratio_log_mean", ""),
                        stats.get("ratio_log_std", ""),
                        stats.get("ratio_leaves_mean", ""),
                        stats.get("ratio_leaves_std", ""),
                        stats.get("count_air_mean", ""),
                        stats.get("count_log_mean", ""),
                        stats.get("count_leaves_mean", ""),
                        stats.get("bbox_z_mean", ""),
                        stats.get("bbox_z_std", ""),
                        stats.get("bbox_y_mean", ""),
                        stats.get("bbox_y_std", ""),
                        stats.get("bbox_x_mean", ""),
                        stats.get("bbox_x_std", ""),
                        stats.get("bbox_aryx_mean", ""),
                        stats.get("bbox_aryx_std", ""),
                        stats.get("bbox_azx_mean", ""),
                        stats.get("bbox_azx_std", ""),
                        stats.get("bbox_volume_mean", ""),
                        stats.get("bbox_volume_std", ""),
                        # Error
                        "",
                    ]
                    writer.writerow(row)

                except Exception as exc:
                    err = str(exc)
                    console.print(
                        f"[red]✗[/red] [{idx}/{len(model_files)}] {model_name}: [red]{err}[/red]"
                    )
                    ft.write(
                        json.dumps(
                            {"model": str(model_path), "error": err},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # Write error row
                    writer.writerow(
                        [str(model_path)] + [""] * (len(header) - 2) + [err]
                    )

                progress.update(models_task, advance=1)

    console.print(
        Panel.fit(
            f"[bold green]✓ 評估完成！[/bold green]\n"
            f"結果已保存至:\n"
            f"  [cyan]TXT: {out_txt}[/cyan]\n"
            f"  [cyan]CSV: {out_csv}[/cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

