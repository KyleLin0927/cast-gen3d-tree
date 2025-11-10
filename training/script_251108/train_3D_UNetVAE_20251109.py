#!/usr/bin/env python3
"""
Minecraft 3D U-Net VAE (32x32x32 voxels, 3 classes: 0=air,1=oak_log,2=oak_leaves)

Now supports epoch-wise skip connection scheduling via --skip_schedule.

--skip_schedule format:
  "epoch:levels,epoch:levels,..."

Example:
  --skip_levels 3 --skip_schedule "1:3,9:2,17:1,33:0"

Meaning:
  - from epoch 1 use 3 skip levels (deepest 3: e3,e2,e1)
  - from epoch 9 use 2 levels  (e3,e2)
  - from epoch 17 use 1 level (e3)
  - from epoch 33 use 0 levels (no skip)

If --skip_schedule is not provided:
  - behaviour is the same as before:
    * if --no_skip_connections: no skips
    * else use fixed --skip_levels
"""

import argparse
import os
import math
import random
import time
import csv
import signal
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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
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

    temp_dir = tempfile.TemporaryDirectory(prefix='train_vae_zip_')
    extract_dir = temp_dir.name

    if console:
        console.print(f"[cyan]Extracting zip file: {zip_path}[/cyan]")
        console.print(f"[dim]Temporary extraction directory: {extract_dir}[/dim]")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        if console:
            console.print(f"[green]✓[/green] Extracted {len(zip_ref.namelist())} items from zip")

    train_dir = os.path.join(extract_dir, 'train')
    val_dir = os.path.join(extract_dir, 'val')
    test_dir = os.path.join(extract_dir, 'test')

    if not os.path.exists(train_dir):
        found_dirs = []
        for root, dirs, files in os.walk(extract_dir):
            if os.path.basename(root) in ['train', 'val', 'test']:
                found_dirs.append(root)
        if not found_dirs:
            raise ValueError("Zip file must contain train/val/test subdirectories.")
        if len(found_dirs) >= 3:
            common_parent = os.path.commonpath(found_dirs)
            extract_dir = common_parent
            train_dir = os.path.join(extract_dir, 'train')
            val_dir = os.path.join(extract_dir, 'val')
            test_dir = os.path.join(extract_dir, 'test')

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
        aug_mode: str = 'enumerate',
        aug_flip_x: bool = False,
        aug_flip_y: bool = False,
        aug_flip_z: bool = False,
        aug_rot_x: bool = False,
        aug_rot_y: bool = False,
        aug_rot_z: bool = False,
        aug_perturb: bool = False,
        perturb_prob: float = 0.01,
        preload: bool = False,
        console = None,
    ):
        from itertools import product
        self.files = files
        self.aug_mode = aug_mode
        self.rot_enabled = {'x': aug_rot_x, 'y': aug_rot_y, 'z': aug_rot_z}
        self.flip_enabled = {'x': aug_flip_x, 'y': aug_flip_y, 'z': aug_flip_z}
        self.aug_perturb = aug_perturb
        self.perturb_prob = perturb_prob
        self.preload = preload
        self.data_cache = None

        if aug_mode not in ('enumerate', 'random'):
            raise ValueError("aug_mode must be 'enumerate' or 'random'")

        if self.preload:
            self.data_cache = []
            if console:
                console.print(f"[cyan]Preloading {len(files)} files into RAM...[/cyan]")
            for i, path in enumerate(files):
                with np.load(path, allow_pickle=False) as data:
                    key = 'arr_0' if 'arr_0' in data else list(data.files)[0]
                    arr = data[key]
                assert arr.shape == (32, 32, 32), f"Expected (32,32,32), got {arr.shape} from {path}"
                self.data_cache.append(torch.from_numpy(arr.astype(np.int64)))
                if console and (i + 1) % 100 == 0:
                    console.print(f"  Loaded {i + 1}/{len(files)} files...")
            if console:
                mem_mb = len(files) * 32 * 32 * 32 * 8 / (1024**2)
                console.print(f"[green]✓[/green] Preloaded {len(files)} files (~{mem_mb:.1f} MB)")

        if aug_mode == 'enumerate':
            kx = list(range(4)) if self.rot_enabled['x'] else [0]
            ky = list(range(4)) if self.rot_enabled['y'] else [0]
            kz = list(range(4)) if self.rot_enabled['z'] else [0]
            fx = [0, 1] if self.flip_enabled['x'] else [0]
            fy = [0, 1] if self.flip_enabled['y'] else [0]
            fz = [0, 1] if self.flip_enabled['z'] else [0]
            self.combos = list(product(kx, ky, kz, fx, fy, fz))
        else:
            self.combos = None

    def __len__(self):
        if self.aug_mode == 'enumerate':
            return len(self.files) * len(self.combos)
        return len(self.files)

    @staticmethod
    def _one_hot(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
        return F.one_hot(labels.long(), num_classes=num_classes).permute(3, 0, 1, 2).float()

    @staticmethod
    def _apply_rot_flip(labels: torch.Tensor, kx: int, ky: int, kz: int, fx: int, fy: int, fz: int) -> torch.Tensor:
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
        kx = torch.randint(0, 4, (1,)).item() if self.rot_enabled['x'] else 0
        ky = torch.randint(0, 4, (1,)).item() if self.rot_enabled['y'] else 0
        kz = torch.randint(0, 4, (1,)).item() if self.rot_enabled['z'] else 0
        fx = int(torch.rand(()) < 0.5) if self.flip_enabled['x'] else 0
        fy = int(torch.rand(()) < 0.5) if self.flip_enabled['y'] else 0
        fz = int(torch.rand(()) < 0.5) if self.flip_enabled['z'] else 0
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
        if self.aug_mode == 'enumerate':
            file_idx = idx // len(self.combos)
            combo_idx = idx % len(self.combos)
            kx, ky, kz, fx, fy, fz = self.combos[combo_idx]
        else:
            file_idx = idx
            kx, ky, kz, fx, fy, fz = self._random_choice()

        if self.preload:
            labels = self.data_cache[file_idx].clone()
        else:
            path = self.files[file_idx]
            with np.load(path, allow_pickle=False) as data:
                key = 'arr_0' if 'arr_0' in data else list(data.files)[0]
                arr = data[key]
            assert arr.shape == (32, 32, 32), f"Expected (32,32,32), got {arr.shape} from {path}"
            labels = torch.from_numpy(arr.astype(np.int64))

        labels = self._apply_rot_flip(labels, kx, ky, kz, fx, fy, fz)
        if self.aug_perturb and self.perturb_prob > 0:
            labels = self._perturb(labels)

        onehot = self._one_hot(labels, 3)
        return onehot, labels

# ----------------------
# Model: 3D U-Net VAE
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

class Encoder3DUNetVAE(nn.Module):
    """
    Down: 32 -> 16 -> 8 -> 4
    Returns:
      mu, logvar: [B, latent_dim, 4,4,4]
      skips: (e1, e2, e3):
        e1: [B, base,   32,32,32]
        e2: [B, base*2, 16,16,16]
        e3: [B, base*4,  8, 8, 8]
    """
    def __init__(self, in_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base*2, 4, stride=2, padding=1),  # 32->16
            ResBlock3D(base*2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base*2, base*4, 4, stride=2, padding=1),  # 16->8
            ResBlock3D(base*4),
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(base*4, base*8, 4, stride=2, padding=1),  # 8->4
            ResBlock3D(base*8),
        )
        self.mu = nn.Conv3d(base*8, latent_dim, 1)
        self.logvar = nn.Conv3d(base*8, latent_dim, 1)

    def forward(self, x, return_skips: bool = True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        mu = self.mu(e4)
        logvar = self.logvar(e4)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        skips = (e1, e2, e3) if return_skips else None
        return mu, logvar, skips

class Decoder3DUNetVAE(nn.Module):
    """
    Up: 4 -> 8 -> 16 -> 32 with configurable skip connections.

    Design:
      - Always initialized with max skip capacity (3 levels).
      - Runtime controls the contribution of each skip via self.skip_gates (α factors):
          gate[0] -> scales e3 at 8^3
          gate[1] -> scales e2 at 16^3
          gate[2] -> scales e1 at 32^3
      - When α=0, we concat zeros instead of the skip feature so the ConvTranspose
        in_channels remain constant and training stays stable even if the skip
        configuration changes across epochs.
    """
    def __init__(self, out_ch=3, base=64, latent_dim=256, skip_levels: float = 3.0):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")

        self.base = base

        # Max-skip architecture (fixed channel sizes)
        self.up1 = nn.ConvTranspose3d(latent_dim, base * 8, 4, stride=2, padding=1)          # 4 -> 8
        self.rb1 = ResBlock3D(base * 8)

        self.up2 = nn.ConvTranspose3d(base * 8 + base * 4, base * 4, 4, stride=2, padding=1) # 8 -> 16
        self.rb2 = ResBlock3D(base * 4)

        self.up3 = nn.ConvTranspose3d(base * 4 + base * 2, base * 2, 4, stride=2, padding=1) # 16 -> 32
        self.rb3 = ResBlock3D(base * 2)

        self.out_block = nn.Sequential(
            nn.Conv3d(base * 2 + base, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

        # Initialize skip gating factors (deep-first: e3, e2, e1)
        self.skip_gates = [0.0, 0.0, 0.0]
        self.set_skip_levels(skip_levels)

    # Helper: should we compute / expect any skips?
    def requires_skips(self) -> bool:
        return any(g > 1e-6 for g in self.skip_gates)

    def set_skip_levels(self, levels: float):
        """
        levels: 0..3 (float), enables skips from deepest upwards:
          0.0 -> no skip
          1.0 -> only e3 (8^3)
          2.0 -> e3 + e2
          3.0 -> e3 + e2 + e1
          Values in-between (e.g. 1.5) apply soft gating (α * skip) to the next level.
        """
        try:
            levels = float(levels)
        except (TypeError, ValueError):
            levels = 0.0
        levels = max(0.0, min(3.0, levels))
        self.skip_levels = levels
        # deep-first gates: [e3, e2, e1]
        gates = [0.0, 0.0, 0.0]
        remaining = levels
        for i in range(3):
            if remaining <= 0.0:
                break
            take = min(1.0, remaining)
            gates[i] = take
            remaining -= take
        self.skip_gates = gates
        self.use_skip_connections = self.requires_skips()

    def _concat_or_zeros(self, h, skip_feat, expected_ch: int, alpha: float):
        """
        Concatenate alpha-scaled skip_feat (soft gate) if available,
        otherwise concatenate zeros with same spatial size and expected_ch.
        This keeps in_channels of subsequent convs fixed.
        """
        if alpha > 1e-6 and (skip_feat is not None):
            return torch.cat([h, skip_feat * alpha], dim=1)
        # concat zeros instead
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
            e1, e2, e3 = skips  # e1:32^3, e2:16^3, e3:8^3

        # 4 -> 8
        h = self.up1(z)
        h = self.rb1(h)

        # concat with e3 at 8^3
        h = self._concat_or_zeros(
            h,
            e3,
            expected_ch=self.base * 4,
            alpha=float(self.skip_gates[0]),
        )

        # 8 -> 16
        h = self.up2(h)
        h = self.rb2(h)

        # concat with e2 at 16^3
        h = self._concat_or_zeros(
            h,
            e2,
            expected_ch=self.base * 2,
            alpha=float(self.skip_gates[1]),
        )

        # 16 -> 32
        h = self.up3(h)
        h = self.rb3(h)

        # concat with e1 at 32^3
        h = self._concat_or_zeros(
            h,
            e1,
            expected_ch=self.base,
            alpha=float(self.skip_gates[2]),
        )

        h = self.out_block(h)
        logits = self.out(h)
        return logits

class UNet3DVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256, skip_levels: float = 3.0):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.encoder = Encoder3DUNetVAE(in_ch, base, latent_dim)
        self.decoder = Decoder3DUNetVAE(out_ch, base, latent_dim, skip_levels=skip_levels)
        self.skip_levels = self.decoder.skip_levels
        self.use_skip_connections = self.decoder.use_skip_connections

    def set_skip_levels(self, levels: float):
        self.decoder.set_skip_levels(levels)
        self.skip_levels = self.decoder.skip_levels
        self.use_skip_connections = self.decoder.use_skip_connections

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # only compute skips when needed
        need_skips = self.decoder.requires_skips()
        mu, logvar, skips = self.encoder(x, return_skips=need_skips)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, skips if need_skips else None)
        return logits, mu, logvar

# ----------------------
# Losses
# ----------------------

def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0,I)) over all non-batch dims."""
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1.0)

def parse_class_weights(arg: str):
    if arg is None or arg.strip().lower() == 'none':
        return None
    parts = [p.strip() for p in arg.split(',')]
    if len(parts) != 3:
        raise ValueError("--class_weights must have exactly 3 numbers or 'none'")
    vals = [float(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError("All class weights must be > 0")
    return torch.tensor(vals, dtype=torch.float32)

# ----------------------
# Skip schedule parser
# ----------------------

def parse_skip_schedule(schedule_str: str, total_epochs: int, base_levels: float):
    """
    Parse --skip_schedule like "1:3,9:2,17:1,33:0" into a sorted list:
      [(1,3), (9,2), (17,1), (33,0)]
    If first epoch not specified, prepend (1, base_levels).
    """
    if not schedule_str or schedule_str.strip().lower() in ('none', 'null'):
        return None

    items = []
    for part in schedule_str.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f"Invalid --skip_schedule segment: '{part}', expected 'epoch:levels'")
        e_str, l_str = part.split(':', 1)
        e = int(e_str)
        lv = float(l_str)
        if e < 1 or e > total_epochs:
            raise ValueError(f"--skip_schedule epoch {e} out of range 1..{total_epochs}")
        if lv < 0 or lv > 3:
            raise ValueError(f"--skip_schedule levels must be 0..3, got {lv}")
        items.append((e, lv))

    if not items:
        return None

    items.sort(key=lambda x: x[0])

    # ensure starting point
    if items[0][0] > 1:
        items.insert(0, (1, base_levels))

    return items

def get_skip_levels_for_epoch(epoch: int, schedule, default_levels: float):
    """
    Given parsed schedule, return skip_levels (float) to use at this epoch.
    If no schedule, return default_levels.
    """
    if not schedule:
        return default_levels
    cur = default_levels
    for e, lv in schedule:
        if epoch >= e:
            cur = lv
        else:
            break
    return cur

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
    axes[0].set_title('MaxProj Z (Y,X)')
    axes[1].imshow(max_y)
    axes[1].set_title('MaxProj Y (Z,X)')
    axes[2].imshow(max_x)
    axes[2].set_title('MaxProj X (Z,Y)')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

# ----------------------
# Training / Evaluation
# ----------------------

def train(args, resume_checkpoint=None):
    global_t0 = time.time()
    train_start_time = datetime.now()

    # Device
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    console = Console()

    if args.skip_levels < 0 or args.skip_levels > 3:
        console.print(Panel.fit(
            f"[bold red]ERROR: --skip_levels must be between 0 and 3 (got {args.skip_levels})[/bold red]",
            border_style="red"
        ))
        raise SystemExit(1)

    # Base effective skip (before schedule / flags)
    skip_levels_effective = 0.0 if args.no_skip_connections else float(args.skip_levels)
    args.skip_levels_effective = skip_levels_effective

    if args.no_skip_connections and args.skip_levels > 1e-6:
        console.print("[yellow]Warning:[/yellow] --no_skip_connections overrides --skip_levels; using 0 skip levels.")

    # Parse skip schedule (if any); disabled when no_skip_connections
    skip_schedule = None
    if not args.no_skip_connections and args.skip_schedule:
        skip_schedule = parse_skip_schedule(args.skip_schedule, args.epochs, skip_levels_effective)
        console.print(f"[bold]Skip schedule:[/bold] {skip_schedule}")
    elif args.skip_schedule and args.no_skip_connections:
        console.print("[yellow]Warning:[/yellow] --skip_schedule ignored because --no_skip_connections is set.")

    # Exp dir
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    best_checkpoint_path = os.path.join(exp_dir, f'best_{args.exp_name}.pt')
    last_checkpoint_path = os.path.join(exp_dir, f'last_{args.exp_name}.pt')
    final_checkpoint_path = os.path.join(exp_dir, f'final_{args.exp_name}.pt')

    if not resume_checkpoint:
        if os.path.exists(exp_dir) and os.listdir(exp_dir):
            console.print(Panel.fit(
                f"[bold red]ERROR: Experiment Directory Not Empty[/bold red]\n\n"
                f"[yellow]{exp_dir}[/yellow]\n"
                f"Use another --exp_name, clean directory, or use --resume.",
                border_style="red"
            ))
            raise SystemExit(1)

    # Header
    header_text = (
        f"[bold cyan]Minecraft 3D U-Net VAE Training[/bold cyan]\n"
        f"Experiment: [magenta]{args.exp_name}[/magenta]\n"
        f"Device: [yellow]{device}[/yellow]\n"
        f"Output: [cyan]{exp_dir}[/cyan]\n"
    )
    if resume_checkpoint:
        header_text += f"Mode: [yellow]RESUME from epoch {resume_checkpoint['epoch']}[/yellow]\n"
    header_text += f"Started: [green]{train_start_time.strftime('%Y-%m-%d %H:%M:%S')}[/green]"
    console.print(Panel.fit(header_text, border_style="cyan"))

    # Files
    train_files = sorted(glob(os.path.join(args.data_root, 'train', '*.npz')))
    val_files = sorted(glob(os.path.join(args.data_root, 'val', '*.npz')))
    test_files = sorted(glob(os.path.join(args.data_root, 'test', '*.npz')))

    assert len(train_files) > 0, "No train .npz found"
    assert len(val_files) > 0, "No val .npz found"
    assert len(test_files) > 0, "No test .npz found"

    n_train, n_val, n_test = len(train_files), len(val_files), len(test_files)
    console.print(f"\n[bold]Dataset:[/bold] {n_train} train, {n_val} val, {n_test} test (total {n_train+n_val+n_test})")

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
        aug_mode='random',
        aug_perturb=False,
        preload=args.preload,
        console=console,
    )
    test_ds = VoxelDataset(
        test_files,
        aug_mode='random',
        aug_flip_x=False, aug_flip_y=False, aug_flip_z=False,
        aug_rot_x=False, aug_rot_y=False, aug_rot_z=False,
        aug_perturb=False,
        preload=args.preload,
        console=console,
    )

    use_pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=use_pin_memory)

    # Class weights
    class_weights_base = parse_class_weights(args.class_weights)
    if class_weights_base is None:
        console.print("[bold]Class weights:[/bold] NONE (uniform)")
        def get_weight_tensor(_: torch.Tensor):
            return None
    else:
        console.print(f"[bold]Class weights:[/bold] {class_weights_base.tolist()}")
        console.print(f"[dim]  air={class_weights_base[0]:.4f}, log={class_weights_base[1]:.4f}, leaf={class_weights_base[2]:.4f}[/dim]")
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

    console.print(f"[bold]Augmentation:[/bold] mode={args.aug_mode}, "
                  f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Initial skip status (may be overridden by schedule per epoch)
    if skip_levels_effective <= 1e-6:
        skip_msg = "DISABLED (levels=0)"
    else:
        skip_msg = f"ENABLED (levels={skip_levels_effective:.3f})"
    console.print(f"[bold]Skip connections (initial):[/bold] {skip_msg}\n")

    # Model
    model = UNet3DVAE(
        in_ch=3,
        out_ch=3,
        base=args.base,
        latent_dim=args.latent_dim,
        skip_levels=skip_levels_effective,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    use_amp = (device.type == 'cuda') and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    if device.type == 'cuda':
        if use_amp:
            console.print(f"[bold]AMP:[/bold] [green]ENABLED[/green]")
        else:
            console.print(f"[bold]AMP:[/bold] [yellow]DISABLED[/yellow]")
    else:
        console.print(f"[bold]AMP:[/bold] [dim]N/A[/dim]")

    os.makedirs(exp_dir, exist_ok=True)
    samples_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    start_epoch = 1
    best_val = math.inf
    training_history = []
    cumulative_time_offset = 0.0

    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint['model'])
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        start_epoch = resume_checkpoint['epoch'] + 1
        best_val = resume_checkpoint['best_val']
        training_history = resume_checkpoint.get('training_history', [])
        cumulative_time_offset = resume_checkpoint.get('cumulative_time_secs', 0.0)

        if 'rng_state' in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint['rng_state'])
        if 'cuda_rng_state' in resume_checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_checkpoint['cuda_rng_state'])
        if 'numpy_rng_state' in resume_checkpoint:
            np.random.set_state(resume_checkpoint['numpy_rng_state'])
        if 'python_rng_state' in resume_checkpoint:
            random.setstate(resume_checkpoint['python_rng_state'])
        if scaler is not None and 'scaler' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler'])

        console.print(f"[green]✓[/green] Resumed from epoch {resume_checkpoint['epoch']}, best_val={best_val:.4f}")
        console.print(f"[cyan]Continuing training from epoch {start_epoch} to {args.epochs}[/cyan]\n")

    # Ctrl+C handling
    interrupted = {'flag': False, 'epoch': None}
    def signal_handler(signum, frame):
        console.print("\n[yellow]⚠ Ctrl+C detected! Saving checkpoint before exit...[/yellow]")
        interrupted['flag'] = True
    signal.signal(signal.SIGINT, signal_handler)

    def save_checkpoint(epoch, is_best=False, is_last=False, custom_path=None):
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val': best_val,
            'training_history': training_history,
            'cumulative_time_secs': cumulative_time_offset + (time.time() - global_t0),
            'args': vars(args),
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()

        saved_paths = []
        if is_best:
            path = best_checkpoint_path
            torch.save(checkpoint, path)
            saved_paths.append(path)
        if is_last:
            path = last_checkpoint_path
            torch.save(checkpoint, path)
            if epoch == args.epochs:
                torch.save(checkpoint, final_checkpoint_path)
            saved_paths.append(path)
        if custom_path:
            os.makedirs(os.path.dirname(custom_path), exist_ok=True)
            torch.save(checkpoint, custom_path)
            saved_paths.append(custom_path)
        return saved_paths[-1] if saved_paths else None

    remaining_epochs = args.epochs - start_epoch + 1
    last_skip_levels_used = None
    skip_transition_dir = os.path.join(exp_dir, 'skip_transitions')
    skip_transition_samples_dir = os.path.join(exp_dir, 'skip_transition_samples')

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

        overall_task = progress.add_task("[cyan]Training", total=remaining_epochs + 1)

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_t0 = time.time()
            total_steps = len(train_loader) + len(val_loader)
            epoch_task = progress.add_task(
                f"[green]Epoch {epoch}/{args.epochs} - Training",
                total=total_steps
            )

            # ---- Update skip config for this epoch (schedule) ----
            if args.no_skip_connections:
                cur_skip_levels = 0.0
            else:
                cur_skip_levels = get_skip_levels_for_epoch(
                    epoch,
                    skip_schedule,
                    skip_levels_effective
                )
            if (abs(cur_skip_levels - model.skip_levels) > 1e-6) and (not args.no_skip_connections or model.skip_levels > 1e-6):
                prev_levels = model.skip_levels
                transition_prefix = f"skip_transition_before_epoch_{epoch:03d}_prev_{prev_levels:.3f}_next_{cur_skip_levels:.3f}"
                transition_path = os.path.join(skip_transition_dir, f"{transition_prefix}.pt")
                save_checkpoint(
                    epoch=epoch - 1,
                    custom_path=transition_path
                )
                console.print(
                    f"[dim]Saved skip transition checkpoint: {transition_path}[/dim]"
                )
                # Save sample projections/volumes with current (pre-switch) skip configuration
                os.makedirs(skip_transition_samples_dir, exist_ok=True)
                prev_mode = model.training
                model.eval()
                with torch.no_grad():
                    for onehot_val, _ in val_loader:
                        onehot_val = onehot_val.to(device, non_blocking=True)
                        logits_val, mu_val, logvar_val = model(onehot_val)
                        rec = logits_val[0].detach().cpu()
                        rec_npz = os.path.join(skip_transition_samples_dir, f"{transition_prefix}_rec.npz")
                        rec_png = os.path.join(skip_transition_samples_dir, f"{transition_prefix}_rec.png")
                        save_volume_and_projections(
                            rec,
                            rec_npz,
                            rec_png
                        )
                        need_skips = model.decoder.requires_skips()
                        mu_b, logvar_b, skips = model.encoder(
                            onehot_val,
                            return_skips=need_skips
                        )
                        num_samples = min(args.n_samples, mu_b.shape[0])
                        for i in range(num_samples):
                            z = model.reparameterize(mu_b[i:i+1], logvar_b[i:i+1])
                            if need_skips and skips is not None:
                                skip_i = (
                                    skips[0][i:i+1],
                                    skips[1][i:i+1],
                                    skips[2][i:i+1],
                                )
                            else:
                                skip_i = None
                            logits_s = model.decoder(z, skips=skip_i)
                            vol_s = logits_s[0].detach().cpu()
                            sample_npz = os.path.join(skip_transition_samples_dir, f"{transition_prefix}_sample_{i}.npz")
                            sample_png = os.path.join(skip_transition_samples_dir, f"{transition_prefix}_sample_{i}.png")
                            save_volume_and_projections(
                                vol_s,
                                sample_npz,
                                sample_png
                            )
                        console.print(
                            f"[dim]Saved skip transition samples: {rec_npz} (+{num_samples} posterior samples)[/dim]"
                        )
                        break
                if prev_mode:
                    model.train()
            model.set_skip_levels(cur_skip_levels)

            if (last_skip_levels_used is None) or (not math.isclose(last_skip_levels_used, cur_skip_levels, rel_tol=1e-6, abs_tol=1e-6)):
                alpha_str = ", ".join(f"{a:.3f}" for a in model.decoder.skip_gates)
                console.print(
                    f"[cyan]Epoch {epoch}: skip_levels={cur_skip_levels:.3f} "
                    f"(alphas=[{alpha_str}])[/cyan]"
                )
                last_skip_levels_used = cur_skip_levels

            # ----------------- Train -----------------
            model.train()
            running = 0.0

            if epoch == start_epoch:
                class_counts = torch.zeros(3, dtype=torch.long, device=device)
                logits_stats_collected = False

            for batch_idx, (onehot, labels) in enumerate(train_loader):
                if interrupted['flag']:
                    break

                onehot = onehot.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if epoch == start_epoch:
                    for c in range(3):
                        class_counts[c] += (labels == c).sum()

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, mu, logvar = model(onehot)

                    if epoch == start_epoch and batch_idx == 0 and not logits_stats_collected:
                        lf = logits.float().detach()
                        console.print(
                            f"[dim]Logits: mean={lf.mean():.4f}, std={lf.std():.4f}, "
                            f"min={lf.min():.4f}, max={lf.max():.4f}[/dim]"
                        )
                        logits_stats_collected = True

                    weight_tensor = get_weight_tensor(logits)
                    ce = F.cross_entropy(
                        logits, labels.long(),
                        weight=weight_tensor
                    )
                    kl = kl_divergence(mu, logvar)
                    loss = ce + args.kl_beta * kl

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running += loss.item() * labels.size(0)
                progress.update(epoch_task, advance=1)

                if epoch == start_epoch and batch_idx == 0:
                    with torch.no_grad():
                        weight_tensor = get_weight_tensor(logits)
                        ce_only = F.cross_entropy(
                            logits, labels.long(),
                            weight=weight_tensor
                        )
                        kl_only = kl_divergence(mu, logvar)
                    console.print(
                        f"Debug: CE={ce_only.item():.4f}, KL={kl_only.item():.4f}, total={loss.item():.4f}"
                    )

            train_loss = running / len(train_loader.dataset)

            if epoch == start_epoch:
                total_voxels = class_counts.sum().item()
                if total_voxels > 0:
                    console.print(f"[dim]Class distribution (train, first epoch):[/dim]")
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

            # ----------------- Val -----------------
            progress.update(epoch_task, description=f"[yellow]Epoch {epoch}/{args.epochs} - Validation")

            if interrupted['flag']:
                progress.remove_task(epoch_task)
                val_loss = float('inf')
            else:
                model.eval()
                running = 0.0
                with torch.no_grad():
                    for onehot, labels in val_loader:
                        if interrupted['flag']:
                            break
                        onehot = onehot.to(device)
                        labels = labels.to(device)
                        logits, mu, logvar = model(onehot)
                        weight_tensor = get_weight_tensor(logits)
                        ce = F.cross_entropy(
                            logits, labels.long(),
                            weight=weight_tensor
                        )
                        kl = kl_divergence(mu, logvar)
                        loss = ce + args.kl_beta * kl
                        running += loss.item() * labels.size(0)
                        progress.update(epoch_task, advance=1)
                val_loss = running / len(val_loader.dataset)
                progress.remove_task(epoch_task)

            # Ctrl+C: save last full epoch
            if interrupted['flag']:
                interrupted['epoch'] = epoch
                save_epoch = epoch - 1
                console.print(f"\n[yellow]⚠ Interrupted during epoch {epoch}[/yellow]")
                console.print(f"[cyan]Saving checkpoint at last complete epoch {save_epoch}[/cyan]")
                cp = {
                    'epoch': save_epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val': best_val,
                    'training_history': training_history,
                    'cumulative_time_secs': cumulative_time_offset + (time.time() - global_t0),
                    'args': vars(args),
                    'rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'python_rng_state': random.getstate(),
                }
                if torch.cuda.is_available():
                    cp['cuda_rng_state'] = torch.cuda.get_rng_state_all()
                if scaler is not None:
                    cp['scaler'] = scaler.state_dict()
                torch.save(cp, last_checkpoint_path)
                raise KeyboardInterrupt("Training interrupted by user")

            epoch_secs = time.time() - epoch_t0
            cum_secs = cumulative_time_offset + (time.time() - global_t0)
            is_best = val_loss < best_val

            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time_secs': epoch_secs,
                'cumulative_time_secs': cum_secs,
                'is_best': 'TRUE' if is_best else 'FALSE',
            })

            if is_best:
                best_val = val_loss
                save_checkpoint(epoch, is_best=True)

            save_last_checkpoint = (epoch % args.save_every == 0) or (epoch == args.epochs)
            if save_last_checkpoint:
                save_checkpoint(epoch, is_last=True)

            best_marker = " | ★ Best!" if is_best else ""
            ckpt_marker = " | 💾 Saved" if save_last_checkpoint else ""
            progress.console.print(
                f"Epoch {epoch:03d}: train {train_loss:.6f} | val {val_loss:.6f} | {fmt_secs(epoch_secs)}"
                f"{best_marker}{ckpt_marker}"
            )

            # Samples
            if epoch % args.sample_every == 0:
                sample_task = progress.add_task(f"[blue]Generating samples", total=args.n_samples + 1)
                model.eval()
                with torch.no_grad():
                    # Reconstruction of first val batch
                    for onehot, labels in val_loader:
                        onehot = onehot.to(device)
                        logits, mu, logvar = model(onehot)
                        rec = logits[0].detach().cpu()
                        save_volume_and_projections(
                            rec,
                            os.path.join(samples_dir, f"rec_e{epoch}_{args.exp_name}.npz"),
                            os.path.join(samples_dir, f"rec_e{epoch}_{args.exp_name}.png"),
                        )
                        progress.update(sample_task, advance=1)

                        # Posterior samples: z ~ q(z|x)
                        need_skips = model.decoder.requires_skips()
                        mu_b, logvar_b, skips = model.encoder(
                            onehot,
                            return_skips=need_skips
                        )
                        for i in range(min(args.n_samples, mu_b.shape[0])):
                            z = model.reparameterize(mu_b[i:i+1], logvar_b[i:i+1])
                            if need_skips and skips is not None:
                                skip_i = (
                                    skips[0][i:i+1],
                                    skips[1][i:i+1],
                                    skips[2][i:i+1],
                                )
                            else:
                                skip_i = None
                            logits_s = model.decoder(z, skips=skip_i)
                            vol = logits_s[0].detach().cpu()
                            save_volume_and_projections(
                                vol,
                                os.path.join(samples_dir, f"sample_e{epoch}_{i}_{args.exp_name}.npz"),
                                os.path.join(samples_dir, f"sample_e{epoch}_{i}_{args.exp_name}.png"),
                            )
                            progress.update(sample_task, advance=1)
                        break
                progress.remove_task(sample_task)

            progress.update(overall_task, advance=1)

        # ----------------- Final Test -----------------
        best_model_path = best_checkpoint_path
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False)['model'])
        model.eval()

        running = 0.0
        test_task = progress.add_task("[yellow]Final Test", total=len(test_loader))
        with torch.no_grad():
            for onehot, labels in test_loader:
                onehot = onehot.to(device)
                labels = labels.to(device)
                logits, mu, logvar = model(onehot)
                weight_tensor = get_weight_tensor(logits)
                ce = F.cross_entropy(
                    logits, labels.long(),
                    weight=weight_tensor
                )
                kl = kl_divergence(mu, logvar)
                loss = ce + args.kl_beta * kl
                running += loss.item() * labels.size(0)
                progress.update(test_task, advance=1)
        test_loss = running / len(test_loader.dataset)
        progress.remove_task(test_task)
        progress.update(overall_task, advance=1)
        progress.console.print(f"Final Test: test loss {test_loss:.6f}")

    final_checkpoint_exists = os.path.exists(final_checkpoint_path)
    last_checkpoint_removed = False
    if os.path.exists(last_checkpoint_path):
        try:
            os.remove(last_checkpoint_path)
            last_checkpoint_removed = True
            console.print(f"[cyan]Removed last checkpoint to reduce storage: {last_checkpoint_path}[/cyan]")
        except OSError as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to delete last checkpoint ({e}).")

    total_secs = cumulative_time_offset + (time.time() - global_t0)
    train_end_time = datetime.now()

    # Summary table
    final_table = Table(title="[bold cyan]Training Summary[/bold cyan]", box=box.ROUNDED)
    final_table.add_column("Metric", style="cyan", no_wrap=True)
    final_table.add_column("Value", style="magenta")
    final_table.add_row("Best Val Loss", f"{best_val:.6f}")
    final_table.add_row("Final Test Loss", f"{test_loss:.6f}")
    final_table.add_row("Total Runtime", fmt_secs(total_secs))
    final_table.add_row("Epochs Trained", f"{start_epoch} - {args.epochs}")
    final_table.add_row("Started", train_start_time.strftime('%Y-%m-%d %H:%M:%S'))
    final_table.add_row("Completed", train_end_time.strftime('%Y-%m-%d %H:%M:%S'))
    final_table.add_row("Best Model", best_model_path if os.path.exists(best_model_path) else "None")
    final_table.add_row("Final Checkpoint", final_checkpoint_path if final_checkpoint_exists else "None")
    final_table.add_row(
        "Last Checkpoint",
        "Deleted after completion" if last_checkpoint_removed
        else last_checkpoint_path if os.path.exists(last_checkpoint_path)
        else "None"
    )
    console.print("\n", final_table, "\n")

    # CSV: training history
    csv1_path = os.path.join(exp_dir, f'training_history_{args.exp_name}.csv')
    with open(csv1_path, 'w', newline='') as f:
        fieldnames = ['epoch', 'train_loss', 'val_loss',
                      'epoch_time_secs', 'cumulative_time_secs', 'is_best']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_history)
    console.print(f"[green]✓[/green] Saved training history to [cyan]{csv1_path}[/cyan]")

    # CSV: metadata
    def bool_to_str(v): return 'TRUE' if v else 'FALSE'

    csv2_path = os.path.join(exp_dir, f'experiment_metadata_{args.exp_name}.csv')
    ce_desc = (f"CrossEntropyLoss(weight=[{class_weights_base[0]:.2f},"
               f" {class_weights_base[1]:.2f}, {class_weights_base[2]:.2f}])") if class_weights_base is not None \
              else "CrossEntropyLoss(weight=None)"
    loss_formula = f"Loss = {ce_desc} + {args.kl_beta} * KL_Divergence(mu, logvar)"

    metadata = {
        'exp_name': args.exp_name,
        'resumed_from': args.resume if args.resume else 'None',
        'start_epoch': start_epoch,
        'end_epoch': args.epochs,
        'training_start_time': train_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_end_time': train_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_model_path': best_checkpoint_path,
        'final_checkpoint_path': final_checkpoint_path if final_checkpoint_exists else 'not_available',
        'last_checkpoint_path': 'deleted_after_completion' if last_checkpoint_removed else (last_checkpoint_path if os.path.exists(last_checkpoint_path) else 'not_available'),
        'samples_directory': samples_dir,
        'data_root': args.data_root,
        'out_dir': args.out_dir,
        'exp_dir': exp_dir,
        'n_train_files': n_train,
        'n_val_files': n_val,
        'n_test_files': n_test,
        'n_total_files': n_train + n_val + n_test,
        'train_dataset_size': len(train_ds),
        'val_dataset_size': len(val_ds),
        'test_dataset_size': len(test_ds),
        'class_weights': args.class_weights,
        'kl_beta': args.kl_beta,
        'loss_kl_weight': args.kl_beta,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'workers': args.workers,
        'seed': args.seed,
        'force_cpu': bool_to_str(args.cpu),
        'device': str(device),
        'amp_enabled': bool_to_str(use_amp),
        'no_amp': bool_to_str(args.no_amp),
        'preload': bool_to_str(args.preload),
        'base': args.base,
        'latent_dim': args.latent_dim,
        'use_skip_connections': bool_to_str(any(alpha > 1e-6 for alpha in model.decoder.skip_gates)),
        'skip_levels': f"{model.skip_levels:.3f}",
        'skip_alphas_final': '[' + ', '.join(f"{alpha:.3f}" for alpha in model.decoder.skip_gates) + ']',
        'skip_schedule': args.skip_schedule if hasattr(args, 'skip_schedule') else 'None',
        'aug_mode': args.aug_mode,
        'aug_rot_x': bool_to_str(args.aug_rot_x),
        'aug_rot_y': bool_to_str(args.aug_rot_y),
        'aug_rot_z': bool_to_str(args.aug_rot_z),
        'aug_flip_x': bool_to_str(args.aug_flip_x),
        'aug_flip_y': bool_to_str(args.aug_flip_y),
        'aug_flip_z': bool_to_str(args.aug_flip_z),
        'aug_perturb': bool_to_str(args.aug_perturb),
        'perturb_prob': args.perturb_prob,
        'sample_every': args.sample_every,
        'n_samples': args.n_samples,
        'save_every': args.save_every,
        'best_val_loss': best_val,
        'final_test_loss': test_loss,
        'total_training_time_secs': total_secs,
        'total_training_time_formatted': fmt_secs(total_secs),
        'loss_function': loss_formula,
        'loss_reconstruction': ce_desc,
        'loss_kl_divergence': 'KL(q(z|x) || N(0,I)) (spatial)',
    }

    with open(csv2_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value'])
        for k, v in metadata.items():
            writer.writerow([k, v])
    console.print(f"[green]✓[/green] Saved experiment metadata to [cyan]{csv2_path}[/cyan]")

    csv3_path = os.path.join(exp_dir, f'experiment_metadata_flat_{args.exp_name}.csv')
    with open(csv3_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)
    console.print(f"[green]✓[/green] Saved experiment metadata (flat) to [cyan]{csv3_path}[/cyan]")

    console.print(Panel.fit("[bold green]Training Complete! 🎉[/bold green]", border_style="green"))

# ----------------------
# Main
# ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Tree 3D U-Net VAE")
    parser.add_argument('--data_root', type=str, required=False)
    parser.add_argument('--data_zip', type=str, required=False)
    parser.add_argument('--out_dir', type=str, required=False)
    parser.add_argument('--exp_name', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base', type=int, default=64,
                        help='Base channels for U-Net')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Channels of spatial latent at 4x4x4')

    parser.add_argument('--skip_levels', type=float, default=3.0,
                        help='Initial skip intensity (0..3, floats allowed) applied deep-first for skip connections')
    parser.add_argument('--skip_schedule', type=str, default=None,
                        help='Epoch schedule for skip levels, e.g. "1:3,9:2,17:1,33:0"')

    parser.add_argument('--kl_beta', type=float, default=1e-3)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--aug_mode', type=str, default='enumerate',
                        choices=['enumerate', 'random'])
    parser.add_argument('--aug_rot_x', action='store_true')
    parser.add_argument('--aug_rot_y', action='store_true')
    parser.add_argument('--aug_rot_z', action='store_true')
    parser.add_argument('--aug_flip_x', action='store_true')
    parser.add_argument('--aug_flip_y', action='store_true')
    parser.add_argument('--aug_flip_z', action='store_true')
    parser.add_argument('--aug_perturb', action='store_true')
    parser.add_argument('--perturb_prob', type=float, default=0.01)

    parser.add_argument('--class_weights', type=str, default='none')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--no_skip_connections', action='store_true',
                        help='Disable all decoder skip connections (overrides --skip_levels and --skip_schedule)')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    # If using data_zip, we need a console instance here
    temp_dir_holder = []
    if args.data_zip:
        console = Console()
        extract_dir, temp_dir = extract_zip_to_temp(args.data_zip, console=console)
        args.data_root = extract_dir
        temp_dir_holder.append(temp_dir)
        args.data_zip = None  # prevent double-specification checks downstream

    # Resume logic
    resume_checkpoint = None
    if args.resume:
        import sys
        resume_path = args.resume
        if os.path.isdir(resume_path):
            last_files = glob(os.path.join(resume_path, 'last_*.pt'))
            best_files = glob(os.path.join(resume_path, 'best_*.pt'))
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
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        checkpoint_args = resume_checkpoint.get('args', {})

        explicitly_set = set()
        i = 1
        while i < len(sys.argv):
            a = sys.argv[i]
            if a.startswith('--'):
                name = a[2:].replace('-', '_')
                explicitly_set.add(name)
                if i + 1 < len(sys.argv) and not sys.argv[i+1].startswith('--'):
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

        print(f"[✓] Loaded params from checkpoint")
        print(f"    • data_root: {args.data_root}")
        print(f"    • out_dir: {args.out_dir}")
        print(f"    • exp_name: {args.exp_name}")
        print(f"    • epoch: {resume_checkpoint['epoch']} → {args.epochs}")
        if explicitly_set - {'resume'}:
            print(f"    • Overridden: {', '.join(sorted(explicitly_set - {'resume'}))}")
        print()

    if not args.data_root and not args.data_zip and not resume_checkpoint:
        parser.error("Either --data_root or --data_zip is required (unless resuming).")
    if args.data_root and args.data_zip:
        parser.error("Use only one of --data_root or --data_zip.")
    if not args.out_dir and not resume_checkpoint:
        parser.error("--out_dir is required (unless resuming from checkpoint with it set).")
    if not args.exp_name and not resume_checkpoint:
        parser.error("--exp_name is required (unless resuming with it set).")

    seed_everything(args.seed)
    train(args, resume_checkpoint=resume_checkpoint)