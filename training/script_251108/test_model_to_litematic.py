#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_model_to_litematic.py

直接生成 litematic 檔案方便比較 raw vs rec。
將測試資料輸入模型進行推理，生成預測的 npz 檔案，並轉換為 Minecraft litematic 檔案。

功能：
1. 從指定資料夾讀取測試資料（.npz 檔案）
2. 使用指定的模型進行推理
3. 將預測結果保存為 .npz 檔案（可選）
4. 將預測結果轉換為 .litematic 檔案

轉換方式（體素 ID 到方塊）：
    - 0 = 空氣 (air)
    - 1 = 橡木木頭 (oak_wood)
    - 2 = 橡木樹葉 (oak_leaves)

用法：
  python test_model_to_litematic.py \
    --test_data_dir /path/to/test/data \
    --model_path /path/to/model.pt \
    --output_dir /path/to/output \
    [--device cuda] \
    [--save_npz] \
    [--no_amp]
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from litemapy import Schematic, Region, BlockState

console = Console()

# --- 模型定義（與 evaluate_models.py 相同）---
class ResBlock3D(nn.Module):
    def __init__(self, ch):
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


class ShallowEncoder3DUNetVAE(nn.Module):
    """僅下採樣一次的淺層 VAE Encoder（32 -> 16）。"""

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
    """淺層 VAE Decoder，16 -> 32，支援 0~2 層 skip。"""

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
                raise ValueError("ShallowDecoder 需要 (e1,e2) skip tensors。")
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
    """組合淺層 Encoder/Decoder 的 VAE。"""

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


class MidEncoder3DUNetVAE(nn.Module):
    """32→16→8 兩次下採樣的 VAE Encoder。"""

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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        mu = self.mu(e3)
        logvar = self.logvar(e3)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        skips = (e1, e2, e3) if return_skips else None
        return mu, logvar, skips


class MidDecoder3DUNetVAE(nn.Module):
    """8→16→32 的 VAE Decoder，支援 0~3 層 skip。"""

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
                raise ValueError("MidDecoder3DUNetVAE expects (e1,e2,e3) skip tensors.")
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
    """32→16→8 瓶頸的 VAE。"""

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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        B, C, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat, self.embedding.weight.t())
        )
        encodings_idx = torch.argmin(distances, dim=1)
        encodings_onehot = torch.zeros(encodings_idx.size(0), self.num_embeddings, device=z_e.device)
        encodings_onehot.scatter_(1, encodings_idx.unsqueeze(1), 1)

        z_q = torch.matmul(encodings_onehot, self.embedding.weight).view(B, D, H, W, C)
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        loss = self.beta * F.mse_loss(z_q.detach(), z_e) + F.mse_loss(z_q, z_e.detach())

        z_q_st = z_e + (z_q - z_e).detach()

        avg_probs = encodings_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encodings_idx = encodings_idx.view(B, D, H, W)
        return z_q_st, loss, perplexity, encodings_idx


class Encoder3DVQVAE(nn.Module):
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
        self.to_latent = nn.Conv3d(base * 8, latent_dim, 1)

    def forward(self, x, return_skips: bool = True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z_e = self.to_latent(e4)
        skips = (e1, e2, e3) if return_skips else None
        return z_e, skips


class Decoder3DVQVAE(nn.Module):
    def __init__(self, out_ch=3, base=64, latent_dim=256, skip_levels: int = 0):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.base = base
        self.skip_levels = int(skip_levels)
        self.use_skip_connections = self.skip_levels > 0

        self.up1 = nn.ConvTranspose3d(latent_dim, base * 8, 4, stride=2, padding=1)
        self.rb1 = ResBlock3D(base * 8)

        up2_in = base * 8 + (base * 4 if self.skip_levels >= 1 else 0)
        self.up2 = nn.ConvTranspose3d(up2_in, base * 4, 4, stride=2, padding=1)
        self.rb2 = ResBlock3D(base * 4)

        up3_in = base * 4 + (base * 2 if self.skip_levels >= 2 else 0)
        self.up3 = nn.ConvTranspose3d(up3_in, base * 2, 4, stride=2, padding=1)
        self.rb3 = ResBlock3D(base * 2)

        out_in = base * 2 + (base if self.skip_levels >= 3 else 0)
        self.out_block = nn.Sequential(
            nn.Conv3d(out_in, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, z_q, skips=None):
        if self.use_skip_connections:
            if skips is None:
                raise ValueError("Decoder expects skips when skip_levels > 0")
            e1, e2, e3 = skips
        else:
            e1 = e2 = e3 = None

        h = self.up1(z_q)
        h = self.rb1(h)

        if self.skip_levels >= 1:
            h = torch.cat([h, e3], dim=1)
        h = self.up2(h)
        h = self.rb2(h)

        if self.skip_levels >= 2:
            h = torch.cat([h, e2], dim=1)
        h = self.up3(h)
        h = self.rb3(h)

        if self.skip_levels >= 3:
            h = torch.cat([h, e1], dim=1)

        h = self.out_block(h)
        logits = self.out(h)
        return logits


class UNet3DVQVAE(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base=64,
        latent_dim=256,
        num_embeddings=512,
        vq_beta=0.25,
        skip_levels: int = 0,
    ):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.skip_levels = int(skip_levels)
        self.use_skip_connections = self.skip_levels > 0

        self.encoder = Encoder3DVQVAE(in_ch, base, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, beta=vq_beta)
        self.decoder = Decoder3DVQVAE(out_ch, base, latent_dim, skip_levels=self.skip_levels)

    def forward(self, x):
        z_e, skips = self.encoder(x, return_skips=self.use_skip_connections)
        z_q, vq_loss, perplexity, _ = self.vq(z_e)
        logits = self.decoder(z_q, skips if self.use_skip_connections else None)
        return logits, vq_loss, perplexity


def _pad_tensor_channels(src: torch.Tensor, target_shape, dim: int, key: str):
    if src.shape == target_shape:
        return src

    if len(src.shape) != len(target_shape):
        raise RuntimeError(
            f"Cannot adapt parameter '{key}': rank mismatch {src.shape} vs {target_shape}"
        )

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


def is_vq_checkpoint(args_dict, state_dict_keys):
    if isinstance(args_dict, dict):
        for key in ("model_variant", "vq_num_codes", "vq_beta", "vq_loss_weight"):
            if key in args_dict:
                if key == "model_variant":
                    if str(args_dict[key]).strip().lower() in {"vq", "vqvae", "vq-vae"}:
                        return True
                else:
                    return True
    return any(key.startswith("vq.") for key in state_dict_keys)


# --- 工具函數 ---
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


def one_hot(labels, num_classes=3):
    """將標籤轉換為 one-hot 編碼"""
    oh = np.eye(num_classes, dtype=np.float32)[labels]
    return np.moveaxis(oh, -1, 0)

def load_model(model_path, device, use_amp=True):
    """載入模型"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
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

    is_vq = is_vq_checkpoint(args, state_dict.keys())
    vae_variant = None if is_vq else detect_vae_variant(state_dict)
    compat_notes = []

    if is_vq:
        variant = "vq"
        default_skip = 0
        skip_levels = resolve_skip_levels(args, default=default_skip)
        if skip_levels is None:
            skip_levels = default_skip
        num_embeddings = 512
        beta = 0.25
        if isinstance(args, dict):
            num_embeddings = int(args.get("vq_num_codes", num_embeddings))
            beta = float(args.get("vq_beta", beta))
        if "vq.embedding.weight" in state_dict:
            emb_weight = state_dict["vq.embedding.weight"]
            num_embeddings = emb_weight.shape[0]
            latent = emb_weight.shape[1]
        model = UNet3DVQVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            num_embeddings=num_embeddings,
            vq_beta=beta,
            skip_levels=skip_levels,
        ).to(device)
        adapted_state_dict, _ = upgrade_decoder_state_dict(state_dict, model.state_dict())
    elif vae_variant == "shallow":
        variant = "shallow_vae"
        default_skip = 2
        skip_levels = resolve_skip_levels(args, default=default_skip)
        if skip_levels is None:
            skip_levels = default_skip
        clamped_skip = int(max(0, min(2, int(skip_levels))))
        if clamped_skip != skip_levels:
            compat_notes.append(f"clamped_skip_levels_to_{clamped_skip}")
        skip_levels = clamped_skip
        model = ShallowUNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        adapted_state_dict = OrderedDict(state_dict)
    elif vae_variant == "mid":
        variant = "mid_vae"
        default_skip = 3
        skip_levels = resolve_skip_levels(args, default=default_skip)
        if skip_levels is None:
            skip_levels = default_skip
        clamped_skip = int(max(0, min(3, int(skip_levels))))
        if clamped_skip != skip_levels:
            compat_notes.append(f"clamped_skip_levels_to_{clamped_skip}")
        skip_levels = clamped_skip
        model = MidUNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        adapted_state_dict = OrderedDict(state_dict)
    else:
        variant = "deep_vae"
        default_skip = 3
        skip_levels = resolve_skip_levels(args, default=default_skip)
        if skip_levels is None:
            skip_levels = default_skip
        clamped_skip = int(max(0, min(3, int(skip_levels))))
        if clamped_skip != skip_levels:
            compat_notes.append(f"clamped_skip_levels_to_{clamped_skip}")
        skip_levels = clamped_skip
        model = UNet3DVAE(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent,
            skip_levels=skip_levels,
        ).to(device)
        model_state = model.state_dict()
        adapted_state_dict, pad_notes = upgrade_decoder_state_dict(state_dict, model_state)
        if pad_notes:
            compat_notes.extend(pad_notes)

    model.load_state_dict(adapted_state_dict, strict=True)
    model.eval()

    if not use_amp:
        model = model.float()

    console.print(
        f"[dim]Detected model variant: {variant}, skip_levels={skip_levels}"
        + (f" ({', '.join(compat_notes)})" if compat_notes else "")
    )

    return model

def load_npz_array(npz_path: str) -> np.ndarray:
    """載入 npz 檔案中的陣列"""
    f = np.load(npz_path, allow_pickle=True)
    key = "data" if "data" in f.files else ("arr_0" if "arr_0" in f.files else f.files[0])
    arr = f[key]
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(npz_path)}: expected 3D array, got shape {arr.shape}")
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int8)
    return arr

def array_to_schematic(vox: np.ndarray, name: str) -> Schematic:
    """將體素陣列轉換為 litematic Schematic（參考 convert_npz_to_litematic.py）
    
    轉換方式：
        - 0 = 空氣 (air)
        - 1 = 橡木木頭 (oak_wood)
        - 2 = 橡木樹葉 (oak_leaves)
    """
    sx, sy, sz = vox.shape  # X, Y, Z
    # Region 只接受 6 個數字參數 (x, y, z, width, height, length)
    reg = Region(0, 0, 0, sx, sy, sz)  # 原點放置，大小即陣列尺寸

    # 三類方塊對應：0=空氣, 1=橡木木頭, 2=橡木樹葉
    AIR = BlockState("minecraft:air")
    OAK_WOOD = BlockState("minecraft:oak_wood")
    # 設定 persistent=true 以防止樹葉腐敗，distance 設為 1 以符合法規定屬性
    OAK_LEAVES = BlockState(
        "minecraft:oak_leaves",
        persistent="true",
        distance="1",
    )
    
    ID_TO_BLOCK = {
        0: AIR,        # 空氣
        1: OAK_WOOD,   # 橡木木頭
        2: OAK_LEAVES, # 橡木樹葉
    }

    # 寫方塊（使用 setblock）
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                bid = int(vox[x, y, z])
                block = ID_TO_BLOCK.get(bid, AIR)
                reg.setblock(x, y, z, block)

    # 用 as_schematic 直接封裝成 Schematic，並給名字
    schem = reg.as_schematic(name=name, author="inference_to_litematic", description="Generated from model inference")
    return schem

@torch.no_grad()
def inference_single(model, npz_path, device, use_amp=True):
    """對單個 npz 檔案進行推理，返回預測的體素陣列"""
    # 載入測試資料
    with np.load(npz_path, allow_pickle=False) as d:
        arr = d['arr_0'] if 'arr_0' in d else d[list(d.files)[0]]
    gt = arr.astype(np.uint8)
    
    # 轉換為 one-hot 編碼
    x = torch.from_numpy(one_hot(gt)).unsqueeze(0).to(device)
    if not use_amp:
        x = x.float()
    
    # 模型推理
    def _extract_logits(model_output):
        logits = model_output
        while isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits

    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast():
            logits = _extract_logits(model(x))
    else:
        logits = _extract_logits(model(x))

    if logits.dim() == 5:
        if logits.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 during inference, got batch dimension {logits.shape[0]}"
            )
        logits = logits[0]
    elif logits.dim() != 4:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    pred = torch.argmax(logits, dim=0).to(dtype=torch.uint8).cpu().numpy()
    return pred

def main():
    ap = argparse.ArgumentParser(
        description="將測試資料輸入模型進行推理，生成預測的 npz 檔案，並轉換為 Minecraft litematic 檔案"
    )
    ap.add_argument("--test_data_dir", required=True, help="測試資料資料夾（包含 .npz 檔案）")
    ap.add_argument("--model_path", required=True, help="模型檔案路徑（.pt 或 .pth）")
    ap.add_argument("--output_dir", required=True, help="輸出資料夾（將保存 .litematic 檔案）")
    ap.add_argument("--device", default="cuda", help="計算裝置：cuda 或 cpu（預設：cuda）")
    ap.add_argument("--save_npz", action="store_true", help="是否同時保存預測的 .npz 檔案")
    ap.add_argument("--no_amp", action="store_true", help="禁用 CUDA AMP（使用完整 float32 精度）")
    args = ap.parse_args()

    # 解析路徑
    test_data_dir = Path(args.test_data_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # 驗證輸入路徑
    if not test_data_dir.is_dir():
        console.print(f"[red]錯誤：測試資料資料夾不存在：{test_data_dir}[/red]")
        return
    
    if not model_path.is_file():
        console.print(f"[red]錯誤：模型檔案不存在：{model_path}[/red]")
        return

    # 收集所有 .npz 檔案
    test_files = sorted(test_data_dir.glob("*.npz"))
    if not test_files:
        console.print(f"[red]錯誤：在 {test_data_dir} 中找不到任何 .npz 檔案[/red]")
        return

    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npz:
        npz_output_dir = output_dir / "npz"
        npz_output_dir.mkdir(parents=True, exist_ok=True)

    # 設定裝置
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # 顯示啟動資訊
    console.print(f"[bold cyan]模型推理與轉換工具[/bold cyan]")
    console.print(f"測試資料資料夾：{test_data_dir}")
    console.print(f"模型檔案：{model_path}")
    console.print(f"輸出資料夾：{output_dir}")
    console.print(f"測試檔案數量：{len(test_files)}")
    console.print(f"計算裝置：{device}")
    console.print(f"CUDA AMP：{'啟用' if use_amp else '禁用'}")
    console.print(f"保存 .npz：{'是' if args.save_npz else '否'}")
    console.print()

    # 載入模型
    console.print(f"[cyan]載入模型中...[/cyan]")
    try:
        model = load_model(str(model_path), device, use_amp)
        console.print(f"[green]✓ 模型載入成功[/green]")
    except Exception as e:
        console.print(f"[red]錯誤：模型載入失敗：{e}[/red]")
        return

    # 處理所有測試檔案
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
        task = progress.add_task(
            "[cyan]處理測試檔案中...",
            total=len(test_files)
        )
        
        success_count = 0
        fail_count = 0
        
        for idx, test_file in enumerate(test_files, 1):
            # 更新進度條描述，顯示當前處理的檔案名
            file_name = test_file.name
            if len(file_name) > 40:
                file_name = file_name[:37] + "..."
            progress.update(
                task,
                description=f"[cyan]處理中 [{idx}/{len(test_files)}]: {file_name}"
            )
            
            try:
                # 推理
                pred = inference_single(model, str(test_file), device, use_amp)
                
                # 保存 .npz 檔案（如果需要）
                if args.save_npz:
                    npz_output_path = npz_output_dir / test_file.name
                    np.savez_compressed(str(npz_output_path), data=pred)
                
                # 轉換為 litematic
                base_name = test_file.stem
                schem = array_to_schematic(pred, name=base_name)
                litematic_path = output_dir / f"{base_name}.litematic"
                schem.save(str(litematic_path))
                
                success_count += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠ 處理失敗 {test_file.name}：{e}[/yellow]")
                fail_count += 1
            finally:
                progress.advance(task)

    console.print()
    console.print(f"[bold green]✅ 全部處理完成！[/bold green]")
    console.print(f"成功處理：{success_count} 個檔案")
    if fail_count > 0:
        console.print(f"[yellow]處理失敗：{fail_count} 個檔案[/yellow]")
    console.print(f"輸出資料夾：{output_dir}")
    if args.save_npz:
        console.print(f"NPZ 輸出資料夾：{npz_output_dir}")

if __name__ == "__main__":
    main()

