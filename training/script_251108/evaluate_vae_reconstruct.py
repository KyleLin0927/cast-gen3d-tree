#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate 3D U-Net VAE checkpoints (.pt/.pth) on 32x32x32 voxel volumes.
Uses the deterministic mean of the posterior (mu) for reconstruction to avoid
stochastic variance during evaluation.

Metrics (3-class: 0=air, 1=log, 2=leaves):

For IoU / Dice / Precision / Recall / F1,統一輸出:
- *_mean: 三類 (air/log/leaves) macro mean
- *_nonair: 把 log+leaves 當一類 (binary, 相對 air)
- *_air, *_log, *_leaves: 各別類別

木頭相關額外指標 (從整體混淆矩陣計算):
- precision_log, recall_log, f1_log (同上，只是特別關心)
- log_overfill_ratio = FP_log / N_log
- log_on_air_ratio = FP_log_on_air / N_air
- log_on_leaves_ratio = FP_log_on_leaves / N_leaves
"""

import argparse
import csv
import json
import os
import zipfile
import tempfile
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Model definitions (copied from training script to match checkpoint layout)
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

        # Always allocate full channel capacity (3 skip levels)
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
        logits = self.decoder(mu, skips if need_skips else None)  # deterministic: use mean
        return logits, mu, logvar


# ---------------------------------------------------------------------------
# Shallow 16^3 bottleneck variant (single downsample)
# ---------------------------------------------------------------------------


class ShallowEncoder3DUNetVAE(nn.Module):
    """
    Down: 32 -> 16 (single downsample).
    Returns mu/logvar at 16^3 along with optional skip tensors.
    """

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
    """
    Up: 16 -> 32 with configurable skip connections (0-2 levels).
    """

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


# ---------------------------------------------------------------------------
# Mid 8^3 bottleneck variant (two downsamples)
# ---------------------------------------------------------------------------


class MidEncoder3DUNetVAE(nn.Module):
    """
    Down: 32 -> 16 -> 8 (spatial latent 8^3).
    """

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
    """
    Up: 8 -> 16 -> 32 with optional skip connections (levels 0-3).
    """

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
# VQ-VAE model definitions (8x8x8 discrete latent)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Checkpoint compatibility helpers
# ---------------------------------------------------------------------------


def _pad_tensor_channels(src: torch.Tensor, target_shape, dim: int, key: str):
    if src.shape == target_shape:
        return src

    if len(src.shape) != len(target_shape):
        raise RuntimeError(
            f"Cannot adapt parameter '{key}': rank mismatch {src.shape} vs {target_shape}"
        )

    # Validate that only the requested dim needs growth.
    for axis, (s, t) in enumerate(zip(src.shape, target_shape)):
        if axis == dim:
            if s > t:
                raise RuntimeError(
                    f"Cannot shrink parameter '{key}' along dim {dim}: {s} -> {t}"
                )
            continue
        if s != t:
            raise RuntimeError(
                f"Cannot adapt parameter '{key}': shape mismatch {src.shape} vs {target_shape}"
            )

    dst = src.new_zeros(target_shape)
    if dim == 0:
        dst[: src.shape[0], ...] = src
    elif dim == 1:
        dst[:, : src.shape[1], ...] = src
    else:
        raise RuntimeError(f"Unsupported padding dim {dim} for '{key}'")
    return dst


def upgrade_decoder_state_dict(state_dict: dict, model_state: dict):
    """
    Back-fill legacy decoder weights (without zero-fill skip channels) so that
    they match the current fixed-width skip architecture.
    """
    if not isinstance(state_dict, dict):
        return state_dict, []

    keys_to_adjust = [
        ("decoder.up2.weight", 0),
        ("decoder.up3.weight", 0),
        ("decoder.out_block.0.weight", 1),
    ]

    upgraded = OrderedDict(state_dict)
    adjustments = []

    for key, dim in keys_to_adjust:
        if key not in upgraded or key not in model_state:
            continue
        src = upgraded[key]
        target_shape = model_state[key].shape
        if src.shape == target_shape:
            continue
        padded = _pad_tensor_channels(src, target_shape, dim=dim, key=key)
        if padded.shape != target_shape:
            raise RuntimeError(
                f"Failed to adapt parameter '{key}' to target shape {target_shape}"
            )
        upgraded[key] = padded
        adjustments.append(key)

    return upgraded, adjustments


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def one_hot(labels, num_classes=3):
    oh = np.eye(num_classes, dtype=np.float32)[labels]
    return np.moveaxis(oh, -1, 0)


def compute_confusion_stats(pred, gt, num_classes=3):
    stats = []
    for c in range(num_classes):
        p, g = (pred == c), (gt == c)
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        stats.append((tp, fp, fn))
    return stats


def iou_from_stats(stats):
    return [tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0 for tp, fp, fn in stats]


def dice_from_stats(stats):
    return [
        (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 1.0
        for tp, fp, fn in stats
    ]


def prec_recall_f1_from_stats(stats):
    precisions, recalls, f1s = [], [], []
    for tp, fp, fn in stats:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return precisions, recalls, f1s


def occupancy_ratio(labels):
    return float((labels != 0).sum()) / labels.size


def aabb_spans(mask):
    if not mask.any():
        return (0, 0, 0), (0, 0)
    coords = np.array(np.where(mask))
    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1)
    Lz, Ly, Lx = zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1
    return (Lz, Ly, Lx), (Ly / Lx if Lx > 0 else 0, Lz / Lx if Lx > 0 else 0)


# ---------------------------------------------------------------------------
# Model parameter summaries
# ---------------------------------------------------------------------------


def extract_weight_stats(model, include_layer_details=False):
    weight_stats = {}
    all_weights = []
    layer_stats = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        arr = param.detach().cpu().numpy()
        all_weights.append(arr.reshape(-1))
        if include_layer_details:
            layer_stats[name] = {
                "shape": list(param.shape),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "numel": int(param.numel()),
            }

    if all_weights:
        concat = np.concatenate(all_weights)
        weight_stats = {
            "weight_mean": float(concat.mean()),
            "weight_std": float(concat.std()),
            "weight_min": float(concat.min()),
            "weight_max": float(concat.max()),
            "weight_abs_mean": float(np.abs(concat).mean()),
            "weight_abs_max": float(np.abs(concat).max()),
        }
        if include_layer_details:
            weight_stats["layer_details"] = layer_stats

    return weight_stats


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


def extract_model_params(model, ckpt, include_weight_stats=False, model_variant=None):
    params = {}

    if isinstance(ckpt, dict) and "args" in ckpt:
        args = ckpt["args"]
        params["base"] = args.get("base")
        params["latent_dim"] = args.get("latent_dim")
        params["lr"] = args.get("lr")
        params["batch_size"] = args.get("batch_size")
        params["epochs"] = args.get("epochs")
        params["kl_beta"] = args.get("kl_beta")
        params["checkpoint_epoch"] = ckpt.get("epoch")
        params["best_val_loss"] = ckpt.get("best_val")
        params["exp_name"] = args.get("exp_name")
        
        # VQ-VAE models don't have skip connections
        if model_variant == "vqvae" or model_variant == "vqvae_8x8x8":
            params["skip_levels"] = 0
            params["use_skip_connections"] = False
        else:
            # For VAE models, resolve skip levels from args
            skip_levels = resolve_skip_levels(args)
            if skip_levels is not None:
                params["skip_levels"] = skip_levels
            use_skip = args.get("use_skip_connections")
            if use_skip is not None:
                params["use_skip_connections"] = parse_bool(use_skip)
            elif skip_levels is not None:
                params["use_skip_connections"] = skip_levels > 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params["total_params"] = total_params
    params["trainable_params"] = trainable_params
    params["non_trainable_params"] = total_params - trainable_params
    params["model_size_mb"] = total_params * 4 / (1024 * 1024)  # float32

    if include_weight_stats:
        params["weight_stats"] = extract_weight_stats(model, include_layer_details=False)

    return params


def detect_vae_variant(state_dict):
    if not isinstance(state_dict, dict):
        return "deep"
    keys = set(state_dict.keys())
    
    # Check if it's a VQ-VAE model (has vq.embedding)
    has_vq = any(k.startswith("vq.embedding") for k in keys)
    if has_vq:
        return "vqvae"
    
    # VAE variants
    has_enc3 = any(k.startswith("encoder.enc3") for k in keys)
    has_enc4 = any(k.startswith("encoder.enc4") for k in keys)
    if not has_enc3:
        return "shallow"
    if has_enc3 and not has_enc4:
        return "mid"
    return "deep"


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_model(
    model_path,
    test_files,
    device,
    progress=None,
    task=None,
    use_amp=True,
    include_weight_stats=False,
):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        args = ckpt.get("args", {}) or {}
        base = int(args.get("base", 64))
    else:
        state_dict = ckpt
        base = 64

    vae_variant = detect_vae_variant(state_dict)
    compat_adjustments = []
    default_skip = 3
    if vae_variant == "shallow":
        default_skip = 2
    skip_levels = resolve_skip_levels(args, default=default_skip)
    if skip_levels is None:
        skip_levels = default_skip

    # Build model
    if vae_variant == "vqvae":
        # VQ-VAE model
        codebook_size = int(args.get("codebook_size", 512))
        commitment_cost = float(args.get("commitment_cost", 0.25))
        # VQ-VAE typically uses smaller latent_dim (default 64)
        latent = int(args.get("latent_dim", 64))
        model = VQVAE3D(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
        ).to(device)
        adapted_state_dict = OrderedDict(state_dict)
        model_variant = "vqvae_8x8x8"
    
    elif vae_variant == "shallow":
        # VAE models use default latent_dim=256
        latent = int(args.get("latent_dim", 256))
        clamped_skip = int(max(0, min(2, int(skip_levels))))
        if clamped_skip != skip_levels:
            compat_adjustments.append(
                f"clamped_skip_levels_to_{clamped_skip}_for_shallow_variant"
            )
        skip_levels = clamped_skip
        model = ShallowUNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        adapted_state_dict = OrderedDict(state_dict)
        model_variant = "vae_shallow16"

    elif vae_variant == "mid":
        # VAE models use default latent_dim=256
        latent = int(args.get("latent_dim", 256))
        clamped_skip = int(max(0, min(3, int(skip_levels))))
        if clamped_skip != skip_levels:
            compat_adjustments.append(
                f"clamped_skip_levels_to_{clamped_skip}_for_mid_variant"
            )
        skip_levels = clamped_skip
        model = MidUNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        adapted_state_dict = OrderedDict(state_dict)
        model_variant = "vae_mid8"

    else:
        # VAE models use default latent_dim=256
        latent = int(args.get("latent_dim", 256))
        clamped_skip = int(max(0, min(3, int(skip_levels))))
        if clamped_skip != skip_levels:
            compat_adjustments.append(
                f"clamped_skip_levels_to_{clamped_skip}_for_deep_variant"
            )
        skip_levels = clamped_skip
        model = UNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        model_state = model.state_dict()
        adapted_state_dict, padding_notes = upgrade_decoder_state_dict(
            state_dict, model_state
        )
        if padding_notes:
            compat_adjustments.extend(padding_notes)
        model_variant = "vae_deep4"

    compat_adjustments.append(f"detected_variant={model_variant}")
    model.load_state_dict(adapted_state_dict, strict=True)
    model.eval()

    if not use_amp:
        model = model.float()

    model_params = extract_model_params(model, ckpt, include_weight_stats=include_weight_stats, model_variant=model_variant)
    model_params["detected_variant"] = model_variant

    # Aggregators
    # agg[c] = [tp, fp, fn]
    agg = np.zeros((3, 3), dtype=np.int64)

    # non-air binary confusion
    nonair_tp = nonair_fp = nonair_fn = 0

    # occupancy
    occ_pred, occ_gt = [], []

    # bbox shape errors
    aryx_err, azx_err = [], []

    # for wood-specific stats
    total_air = total_log = total_leaves = 0
    total_pred_log = 0
    fp_log_on_air = 0
    fp_log_on_leaves = 0

    for idx, npz_path in enumerate(test_files):
        with np.load(npz_path, allow_pickle=False) as data:
            key = "arr_0" if "arr_0" in data else list(data.files)[0]
            arr = data[key]

        gt = arr.astype(np.uint8)
        x = torch.from_numpy(one_hot(gt)).unsqueeze(0).to(device)
        if not use_amp:
            x = x.float()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                output = model(x)
        else:
            output = model(x)
        
        # Handle different return values: VAE returns (logits, mu, logvar), VQ-VAE returns (logits, vq_loss, perplexity, indices)
        if len(output) == 4:
            logits, _, _, _ = output  # VQ-VAE: (logits, vq_loss, perplexity, indices)
        else:
            logits, _, _ = output  # VAE: (logits, mu, logvar)

        pred = torch.argmax(F.softmax(logits[0], dim=0), dim=0).cpu().numpy().astype(
            np.uint8
        )

        # Per-class confusion
        stats = compute_confusion_stats(pred, gt)
        agg += np.array(stats)

        # Non-air confusion (binary)
        pred_nonair = pred != 0
        gt_nonair = gt != 0
        nonair_tp += np.logical_and(pred_nonair, gt_nonair).sum()
        nonair_fp += np.logical_and(pred_nonair, ~gt_nonair).sum()
        nonair_fn += np.logical_and(~pred_nonair, gt_nonair).sum()

        # Occupancy
        occ_pred.append(occupancy_ratio(pred))
        occ_gt.append(occupancy_ratio(gt))

        # BBox aspect ratio errors (non-air)
        _, (aryx_p, azx_p) = aabb_spans(pred_nonair)
        _, (aryx_g, azx_g) = aabb_spans(gt_nonair)
        if aryx_g > 0:
            aryx_err.append(abs(aryx_p - aryx_g))
        if azx_g > 0:
            azx_err.append(abs(azx_p - azx_g))

        # Wood-specific counters
        total_air += (gt == 0).sum()
        total_log += (gt == 1).sum()
        total_leaves += (gt == 2).sum()
        total_pred_log += (pred == 1).sum()
        fp_log_on_air += np.logical_and(pred == 1, gt == 0).sum()
        fp_log_on_leaves += np.logical_and(pred == 1, gt == 2).sum()

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    # Per-class IoU, Dice
    iou = iou_from_stats(agg)
    dice = dice_from_stats(agg)

    # Per-class Precision / Recall / F1
    precisions, recalls, f1s = prec_recall_f1_from_stats(agg)

    # Macro means over 3 classes
    iou_mean = float(np.mean(iou))
    dice_mean = float(np.mean(dice))
    precision_mean = float(np.mean(precisions))
    recall_mean = float(np.mean(recalls))
    f1_mean = float(np.mean(f1s))

    # Non-air binary metrics
    if nonair_tp + nonair_fp + nonair_fn > 0:
        iou_nonair = nonair_tp / (nonair_tp + nonair_fp + nonair_fn)
        dice_nonair = (2 * nonair_tp) / (2 * nonair_tp + nonair_fp + nonair_fn)
    else:
        iou_nonair = 1.0
        dice_nonair = 1.0

    precision_nonair = (
        nonair_tp / (nonair_tp + nonair_fp) if (nonair_tp + nonair_fp) > 0 else 0.0
    )
    recall_nonair = (
        nonair_tp / (nonair_tp + nonair_fn) if (nonair_tp + nonair_fn) > 0 else 0.0
    )
    f1_nonair = (
        2 * precision_nonair * recall_nonair / (precision_nonair + recall_nonair)
        if (precision_nonair + recall_nonair) > 0
        else 0.0
    )

    # Wood-specific: class index 1
    tp_log, fp_log, fn_log = agg[1]
    precision_log = precisions[1]
    recall_log = recalls[1]
    f1_log = f1s[1]

    log_overfill_ratio = (fp_log / total_log) if total_log > 0 else 0.0
    log_on_air_ratio = (
        fp_log_on_air / total_air if total_air > 0 else 0.0
    )
    log_on_leaves_ratio = (
        fp_log_on_leaves / total_leaves if total_leaves > 0 else 0.0
    )

    # Occupancy / shape
    occ_pred_mean = float(np.mean(occ_pred))
    occ_gt_mean = float(np.mean(occ_gt))
    occ_err = float(abs(occ_pred_mean - occ_gt_mean))
    aryx_err_mean = float(np.mean(aryx_err)) if aryx_err else 0.0
    azx_err_mean = float(np.mean(azx_err)) if azx_err else 0.0

    return {
        "model": str(model_path),
        # IoU
        "iou_mean": iou_mean,
        "iou_nonair": float(iou_nonair),
        "iou_air": float(iou[0]),
        "iou_log": float(iou[1]),
        "iou_leaves": float(iou[2]),
        # Dice
        "dice_mean": dice_mean,
        "dice_nonair": float(dice_nonair),
        "dice_air": float(dice[0]),
        "dice_log": float(dice[1]),
        "dice_leaves": float(dice[2]),
        # Precision
        "precision_mean": precision_mean,
        "precision_nonair": float(precision_nonair),
        "precision_air": float(precisions[0]),
        "precision_log": float(precision_log),
        "precision_leaves": float(precisions[2]),
        # Recall
        "recall_mean": recall_mean,
        "recall_nonair": float(recall_nonair),
        "recall_air": float(recalls[0]),
        "recall_log": float(recall_log),
        "recall_leaves": float(recalls[2]),
        # F1
        "f1_mean": f1_mean,
        "f1_nonair": float(f1_nonair),
        "f1_air": float(f1s[0]),
        "f1_log": float(f1_log),
        "f1_leaves": float(f1s[2]),
        # Occupancy / shape
        "occ_pred": occ_pred_mean,
        "occ_gt": occ_gt_mean,
        "occ_err": occ_err,
        "aryx_err": aryx_err_mean,
        "azx_err": azx_err_mean,
        # Wood-specific error diagnostics
        "log_overfill_ratio": float(log_overfill_ratio),
        "log_on_air_ratio": float(log_on_air_ratio),
        "log_on_leaves_ratio": float(log_on_leaves_ratio),
        # Model meta
        "model_params": model_params,
        "compat_notes": compat_adjustments,
    }


# ---------------------------------------------------------------------------
# Zip file extraction helper
# ---------------------------------------------------------------------------


def extract_zip_to_temp(zip_path: str, console=None) -> tuple[str, tempfile.TemporaryDirectory]:
    """Extract zip file to a temporary directory and verify train/val/test."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    temp_dir = tempfile.TemporaryDirectory(prefix="eval_models_zip_")
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


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    console = Console()
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net VAE checkpoints.")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_dir", help="Directory with .npz test volumes.")
    data_group.add_argument("--data_zip", help="Zip file containing train/val/test directories with .npz files.")
    parser.add_argument("--models_dir", required=True, help="Directory containing checkpoints (.pt/.pth).")
    parser.add_argument("--out_dir", required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP for evaluation.")
    parser.add_argument(
        "--skip_patterns",
        nargs="*",
        default=["last"],
        help="Filename patterns to skip (case-insensitive). Default: ['last']",
    )
    parser.add_argument(
        "--include_weight_stats",
        action="store_true",
        help="Include overall weight statistics for each model.",
    )
    args = parser.parse_args()

    # Handle data source (directory or zip file)
    temp_dir_holder = []
    if args.data_zip:
        # Extract zip file to temporary directory
        zip_path = Path(args.data_zip).expanduser().resolve()
        extract_dir, temp_dir = extract_zip_to_temp(str(zip_path), console=console)
        test_dir = Path(extract_dir) / "test"
        temp_dir_holder.append(temp_dir)
    else:
        # Use provided directory
        test_dir = Path(args.data_dir).expanduser().resolve()
        if not test_dir.exists() or not test_dir.is_dir():
            console.print(
                Panel.fit(
                    f"[bold red]錯誤：測試資料目錄不存在或不是目錄[/bold red]\n"
                    f"路徑: [yellow]{test_dir}[/yellow]",
                    border_style="red",
                )
            )
            raise SystemExit(1)

    test_files = sorted(test_dir.glob("*.npz"))
    if not test_files:
        # Clean up temp directory before exiting
        if temp_dir_holder:
            for temp_dir in temp_dir_holder:
                try:
                    temp_dir.cleanup()
                except Exception:
                    pass
        console.print(
            Panel.fit(
                f"[bold red]錯誤：測試目錄中沒有 .npz 檔案[/bold red]\n"
                f"目錄: [yellow]{test_dir}[/yellow]",
                border_style="red",
            )
        )
        raise SystemExit(1)

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
    out_txt = out_dir / f"eval_unet_{ts}.txt"
    out_csv = out_dir / f"eval_unet_{ts}.csv"

    amp_status = "禁用" if not use_amp else "啟用"
    filter_info = (
        f"（已過濾 {filtered_count} 個包含 {skip_patterns} 的檔案）" if filtered_count > 0 else ""
    )
    console.print(
        Panel.fit(
            f"[bold cyan]3D U-Net VAE 模型評估[/bold cyan]\n"
            f"測試檔案數: [yellow]{len(test_files)}[/yellow]\n"
            f"模型數量: [yellow]{len(model_files)}[/yellow] {filter_info}\n"
            f"跳過模式: [yellow]{skip_patterns if skip_patterns else '無'}[/yellow]\n"
            f"裝置: [yellow]{device}[/yellow]\n"
            f"CUDA AMP: [yellow]{amp_status}[/yellow]\n"
            f"輸出目錄: [cyan]{out_dir}[/cyan]",
            border_style="cyan",
        )
    )

    # CSV header
    header = [
        "model",
        # IoU
        "iou_mean",
        "iou_nonair",
        "iou_air",
        "iou_log",
        "iou_leaves",
        # Dice
        "dice_mean",
        "dice_nonair",
        "dice_air",
        "dice_log",
        "dice_leaves",
        # Precision
        "precision_mean",
        "precision_nonair",
        "precision_air",
        "precision_log",
        "precision_leaves",
        # Recall
        "recall_mean",
        "recall_nonair",
        "recall_air",
        "recall_log",
        "recall_leaves",
        # F1
        "f1_mean",
        "f1_nonair",
        "f1_air",
        "f1_log",
        "f1_leaves",
        # Occupancy / shape
        "occ_pred",
        "occ_gt",
        "occ_err",
        "aryx_err",
        "azx_err",
        # Wood-specific diagnostics
        "log_overfill_ratio",
        "log_on_air_ratio",
        "log_on_leaves_ratio",
        # Model meta
        "base",
        "latent_dim",
        "skip_levels",
        "use_skip_connections",
        "total_params",
        "trainable_params",
        "model_size_mb",
        "lr",
        "batch_size",
        "epochs",
        "kl_beta",
        "checkpoint_epoch",
        "best_val_loss",
        # error (空字串表示成功)
        "error",
    ]

    with out_txt.open("w", encoding="utf-8") as ft, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fc:
        writer = csv.writer(fc)
        writer.writerow(header)

        run_header = {
            "data_source": "zip" if args.data_zip else "directory",
            "data_test_dir": str(test_dir),
            "data_zip_path": str(args.data_zip) if args.data_zip else None,
            "models_root": str(models_dir),
            "num_test_files": len(test_files),
            "num_models": len(model_files),
            "num_models_filtered": filtered_count,
            "skip_patterns": skip_patterns,
            "device": str(device),
            "use_amp": use_amp,
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
                    test_task = progress.add_task(
                        "[dim]處理測試檔案...",
                        total=len(test_files),
                        visible=len(test_files) > 10,
                    )

                    result = eval_model(
                        model_path,
                        test_files,
                        device,
                        progress=progress,
                        task=test_task,
                        use_amp=use_amp,
                        include_weight_stats=args.include_weight_stats,
                    )

                    progress.remove_task(test_task)

                    params = result.get("model_params", {})

                    console.print(
                        f"[green]✓[/green] [{idx}/{len(model_files)}] {model_name}\n"
                        f"  [dim]IoU_mean={result['iou_mean']:.4f} "
                        f"IoU_log={result['iou_log']:.4f} IoU_leaves={result['iou_leaves']:.4f} "
                        f"Prec_log={result['precision_log']:.4f} "
                        f"Overfill={result['log_overfill_ratio']:.3f}[/dim]"
                    )

                    compat_notes = result.get("compat_notes") or []
                    if compat_notes:
                        console.print(
                            f"[yellow]  相容性調整：{', '.join(compat_notes)}[/yellow]"
                        )

                    # Write JSON line
                    ft.write(json.dumps(result, ensure_ascii=False) + "\n")

                    row = [
                        result["model"],
                        # IoU
                        result["iou_mean"],
                        result["iou_nonair"],
                        result["iou_air"],
                        result["iou_log"],
                        result["iou_leaves"],
                        # Dice
                        result["dice_mean"],
                        result["dice_nonair"],
                        result["dice_air"],
                        result["dice_log"],
                        result["dice_leaves"],
                        # Precision
                        result["precision_mean"],
                        result["precision_nonair"],
                        result["precision_air"],
                        result["precision_log"],
                        result["precision_leaves"],
                        # Recall
                        result["recall_mean"],
                        result["recall_nonair"],
                        result["recall_air"],
                        result["recall_log"],
                        result["recall_leaves"],
                        # F1
                        result["f1_mean"],
                        result["f1_nonair"],
                        result["f1_air"],
                        result["f1_log"],
                        result["f1_leaves"],
                        # Occupancy / shape
                        result["occ_pred"],
                        result["occ_gt"],
                        result["occ_err"],
                        result["aryx_err"],
                        result["azx_err"],
                        # Wood-specific
                        result["log_overfill_ratio"],
                        result["log_on_air_ratio"],
                        result["log_on_leaves_ratio"],
                        # Meta
                        params.get("base", ""),
                        params.get("latent_dim", ""),
                        params.get("skip_levels", ""),
                        params.get("use_skip_connections", ""),
                        params.get("total_params", ""),
                        params.get("trainable_params", ""),
                        params.get("model_size_mb", ""),
                        params.get("lr", ""),
                        params.get("batch_size", ""),
                        params.get("epochs", ""),
                        params.get("kl_beta", ""),
                        params.get("checkpoint_epoch", ""),
                        params.get("best_val_loss", ""),
                        # error
                        "",
                    ]
                    writer.writerow(row)

                    # Optional pretty print of meta
                    if params:
                        lines = []
                        arch = []
                        if params.get("base") is not None:
                            arch.append(f"base={params['base']}")
                        if params.get("latent_dim") is not None:
                            arch.append(f"latent={params['latent_dim']}")
                        if params.get("skip_levels") is not None:
                            arch.append(f"skip_levels={params['skip_levels']}")
                        if arch:
                            lines.append(f"  [cyan]架構:[/cyan] {', '.join(arch)}")
                        if params.get("use_skip_connections") is not None:
                            skip_status = (
                                "啟用" if params["use_skip_connections"] else "停用"
                            )
                            lines.append(f"  [cyan]Skip 連接:[/cyan] {skip_status}")
                        if params.get("total_params") is not None:
                            total = params["total_params"]
                            trainable = params.get("trainable_params", total)
                            size_mb = params.get("model_size_mb", 0.0)
                            lines.append(
                                f"  [cyan]參數:[/cyan] 總計={total:,} (可訓練={trainable:,}), 大小={size_mb:.2f} MB"
                            )
                        training = []
                        if params.get("lr") is not None:
                            training.append(f"lr={params['lr']}")
                        if params.get("batch_size") is not None:
                            training.append(f"bs={params['batch_size']}")
                        if params.get("epochs") is not None:
                            training.append(f"epochs={params['epochs']}")
                        if params.get("kl_beta") is not None:
                            training.append(f"kl_beta={params['kl_beta']}")
                        if training:
                            lines.append(f"  [cyan]訓練:[/cyan] {', '.join(training)}")
                        if params.get("checkpoint_epoch") is not None:
                            lines.append(
                                f"  [cyan]檢查點:[/cyan] epoch={params['checkpoint_epoch']}"
                            )
                        if params.get("best_val_loss") is not None:
                            lines.append(
                                f"  [cyan]最佳驗證損失:[/cyan] {params['best_val_loss']:.4f}"
                            )
                        weight_stats = params.get("weight_stats", {})
                        if weight_stats:
                            ws = []
                            for key in [
                                "weight_mean",
                                "weight_std",
                                "weight_min",
                                "weight_max",
                            ]:
                                if key in weight_stats:
                                    ws.append(
                                        f"{key.split('_')[1]}={weight_stats[key]:.6f}"
                                    )
                            if ws:
                                lines.append(
                                    f"  [cyan]權重統計:[/cyan] {', '.join(ws)}"
                                )
                        if lines:
                            console.print("\n".join(lines))

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

                    # 填一列空的，最後一格放錯誤訊息
                    writer.writerow(
                        [str(model_path)]
                        + [""] * (len(header) - 2)
                        + [err]
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
    
    # Clean up temporary directory if zip file was used
    if temp_dir_holder:
        for temp_dir in temp_dir_holder:
            try:
                temp_dir.cleanup()
                console.print(f"[dim]Cleaned up temporary directory[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to clean up temporary directory: {e}[/yellow]")


if __name__ == "__main__":
    main()