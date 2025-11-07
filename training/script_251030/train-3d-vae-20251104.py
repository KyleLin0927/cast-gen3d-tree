#!/usr/bin/env python3
"""
Minecraft 3D VAE (32x32x32 voxels, 3 classes: 0=air,1=oak_log,2=oak_leaves)

- Input: dataset directory containing train/val/test subdirectories with .npz files
  (each .npz file should contain a 32x32x32 int8 array)
  OR zip file containing the same structure (automatically extracted to temp directory)
- Model: 3D VAE with categorical reconstruction (per-voxel logits over 3 classes)
- Output: CSV files and model in out_dir, samples in out_dir/samples/

Dataset Structure:
  data_root/
    ├── train/*.npz
    ├── val/*.npz
    └── test/*.npz

Output Structure:
  out_dir/
    └── exp_name/                                    # Experiment name directory
        ├── best_{exp_name}.pt                       # Best model checkpoint
        ├── training_history_{exp_name}.csv          # Train/val loss per epoch
        ├── experiment_metadata_{exp_name}.csv       # All parameters (vertical format)
        ├── experiment_metadata_flat_{exp_name}.csv  # All parameters (horizontal format)
        └── samples/                                 # All visualizations
            ├── rec_e5_{exp_name}.npz, rec_e5_{exp_name}.png   # Reconstructions
            └── sample_e5_0_{exp_name}.npz, ...                 # Generated samples

Augmentations (independent switches):
    * --aug_flip_x/--aug_flip_y/--aug_flip_z : mirror along each axis (2x each when enabled)
    * --aug_rot_x/--aug_rot_y/--aug_rot_z : 0/90/180/270° rotations around that axis (4x each when enabled)
    * --aug_perturb: small random label perturbations (configurable)
    * --aug_mode enumerate|random : enumerate = Cartesian product (static expansion), random = on-the-fly sampling

Performance:
    * --preload : Load all .npz files into RAM at startup (strongly recommended for small datasets)
                  Eliminates repeated file I/O and decompression overhead during training

Resume Training:
    * --resume PATH : Resume training from checkpoint file or directory
                      - File: ./runs/exp1/last_exp1.pt
                      - Directory: ./runs/exp1 (auto-finds last_*.pt or best_*.pt)
                      - Automatically loads ALL original training parameters from checkpoint
                      - Only explicitly specified command-line args will override checkpoint values
                      - Example: --resume ./runs/exp1 --epochs 100 (only epochs is overridden)
    * --save_every N : Save checkpoint every N epochs (default: 5)
    * Ctrl+C handling: Press Ctrl+C to interrupt training gracefully
                       - Detects interruption immediately during training
                       - Saves checkpoint at last completed epoch (incomplete epoch will be re-trained)
                       - Shows resume command for convenience
                       - Ensures no epoch progress is lost

Examples
--------
# 全開，512x 放大（enumerate）
python train-3d-autoencoder-20251102.py \
  --data_root ./npz_dataset \
  --out_dir ./runs \
  --exp_name vae_rotflip_512x \
  --epochs 50 --batch_size 32 \
  --aug_mode enumerate \
  --aug_rot_x --aug_rot_y --aug_rot_z \
  --aug_flip_x --aug_flip_y --aug_flip_z

# 只開旋轉Y -> 4x，啟用預載
python train-3d-autoencoder-20251102.py \
  --data_root ./npz_dataset \
  --out_dir ./runs \
  --exp_name vae_roty4 \
  --aug_mode enumerate --aug_rot_y \
  --preload

# 從中斷點繼續訓練（簡易方式：只需指定目錄和新的 epochs）
python train-3d-autoencoder-20251102.py \
  --resume ./runs/vae_roty4 \
  --epochs 100

# 或指定完整的 checkpoint 檔案路徑
python train-3d-autoencoder-20251102.py \
  --resume ./runs/vae_roty4/last_vae_roty4.pt \
  --epochs 100

# 覆蓋部分參數（其他參數從 checkpoint 讀取）
python train-3d-autoencoder-20251102.py \
  --resume ./runs/vae_roty4 \
  --epochs 200 \
  --lr 5e-4

# 訓練中按 Ctrl+C 會自動保存當前 epoch 的 checkpoint
# 顯示恢復命令後優雅退出

# 使用 zip 壓縮檔作為輸入（自動解壓縮）
python train-3d-vae-20251104.py \
  --data_zip ./dataset.zip \
  --out_dir ./runs \
  --exp_name vae_from_zip \
  --epochs 50 --batch_size 32
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
import shutil
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
    """Extract zip file to a temporary directory.
    
    Args:
        zip_path: Path to the zip file
        console: Optional Rich console for progress messages
        
    Returns:
        Tuple of (extracted_directory_path, TemporaryDirectory_object)
        The TemporaryDirectory object should be kept alive during training
        and will be cleaned up automatically when it goes out of scope.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")
    
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory(prefix='train_vae_zip_')
    extract_dir = temp_dir.name
    
    if console:
        console.print(f"[cyan]Extracting zip file: {zip_path}[/cyan]")
        console.print(f"[dim]Temporary extraction directory: {extract_dir}[/dim]")
    
    # Extract all files from zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
        # List extracted files for verification
        extracted_files = zip_ref.namelist()
        if console:
            console.print(f"[green]✓[/green] Extracted {len(extracted_files)} items from zip")
    
    # Verify that train/val/test subdirectories exist
    train_dir = os.path.join(extract_dir, 'train')
    val_dir = os.path.join(extract_dir, 'val')
    test_dir = os.path.join(extract_dir, 'test')
    
    # Check if structure is flat (files in root) or nested (train/val/test subdirs)
    # Handle both cases: zip might have train/val/test at root, or nested
    if not os.path.exists(train_dir):
        # Try to find train directory in subdirectories
        found_dirs = []
        for root, dirs, files in os.walk(extract_dir):
            if os.path.basename(root) in ['train', 'val', 'test']:
                found_dirs.append(root)
        
        if not found_dirs:
            raise ValueError(
                f"Zip file does not contain train/val/test subdirectories.\n"
                f"Expected structure: zip_root/train/, zip_root/val/, zip_root/test/\n"
                f"Found in zip: {os.listdir(extract_dir)}"
            )
        
        # If we found nested directories, use the parent as data_root
        # Find the common parent of train/val/test directories
        if len(found_dirs) >= 3:
            # Get common parent
            common_parent = os.path.commonpath(found_dirs)
            extract_dir = common_parent
            train_dir = os.path.join(extract_dir, 'train')
            val_dir = os.path.join(extract_dir, 'val')
            test_dir = os.path.join(extract_dir, 'test')
    
    # Final verification
    if not os.path.exists(train_dir):
        raise ValueError(f"train/ directory not found in extracted zip (checked: {train_dir})")
    if not os.path.exists(val_dir):
        raise ValueError(f"val/ directory not found in extracted zip (checked: {val_dir})")
    if not os.path.exists(test_dir):
        raise ValueError(f"test/ directory not found in extracted zip (checked: {test_dir})")
    
    if console:
        console.print(f"[green]✓[/green] Verified train/val/test structure in extracted directory")
    
    return extract_dir, temp_dir

# ----------------------
# Dataset & Augmentations
# ----------------------

class VoxelDataset(Dataset):
    """Loads 32x32x32 int8 npz files as class labels; returns one-hot float for encoder and labels for CE.

    Axis convention: raw array shape [Z, Y, X]. We'll convert to torch tensors as:
        - labels: LongTensor [Z, Y, X]
        - onehot: FloatTensor [C=3, Z, Y, X]

    Augmentation modes:
      - enumerate: Cartesian product over enabled rotations/flips.
        If you enable k rotation axes and m flip axes, expansion factor is 4^k * 2^m.
        E.g., all six → 4*4*4*2*2*2 = 512x. Only --aug_rot_x → 4x. Only --aug_flip_x → 2x.
      - random: on-the-fly; for each enabled rot axis, sample k∈{0,1,2,3}; for each enabled flip axis, sample {no flip, flip}.
    
    Preload mode:
      - If preload=True, all .npz files are loaded into RAM during __init__ (recommended for small datasets)
      - Avoids repeated file I/O and decompression overhead during training
    """

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

        # Preload all files into RAM if requested
        if self.preload:
            self.data_cache = []
            if console:
                console.print(f"[cyan]Preloading {len(files)} files into RAM...[/cyan]")
            for i, path in enumerate(files):
                with np.load(path, allow_pickle=False) as data:
                    arr = data['arr_0'] if 'arr_0' in data else data[list(data.files)[0]]
                assert arr.shape == (32, 32, 32), f"Expected (32,32,32), got {arr.shape} from {path}"
                self.data_cache.append(torch.from_numpy(arr.astype(np.int64)))
                if console and (i + 1) % 100 == 0:
                    console.print(f"  Loaded {i + 1}/{len(files)} files...")
            if console:
                mem_mb = len(files) * 32 * 32 * 32 * 8 / (1024**2)  # int64 = 8 bytes
                console.print(f"[green]✓[/green] Preloaded {len(files)} files (~{mem_mb:.1f} MB)")

        # Precompute transform combos for enumerate mode
        if aug_mode == 'enumerate':
            kx = list(range(4)) if self.rot_enabled['x'] else [0]
            ky = list(range(4)) if self.rot_enabled['y'] else [0]
            kz = list(range(4)) if self.rot_enabled['z'] else [0]
            fx = [0, 1] if self.flip_enabled['x'] else [0]
            fy = [0, 1] if self.flip_enabled['y'] else [0]
            fz = [0, 1] if self.flip_enabled['z'] else [0]
            # store as tuples (kx, ky, kz, fx, fy, fz)
            self.combos = list(product(kx, ky, kz, fx, fy, fz))
        else:
            self.combos = None  # decisions made per-sample

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
        # Rotations (0..3) around each axis
        if kx % 4:
            x = torch.rot90(x, k=int(kx) % 4, dims=(0, 1))  # rotate (Z,Y) -> rot around X
        if ky % 4:
            x = torch.rot90(x, k=int(ky) % 4, dims=(0, 2))  # rotate (Z,X) -> rot around Y
        if kz % 4:
            x = torch.rot90(x, k=int(kz) % 4, dims=(1, 2))  # rotate (Y,X) -> rot around Z
        # Flips (0=no, 1=yes)
        if fz:
            x = torch.flip(x, dims=[0])  # flip Z
        if fy:
            x = torch.flip(x, dims=[1])  # flip Y
        if fx:
            x = torch.flip(x, dims=[2])  # flip X
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
        # Slightly favor air to avoid overfilling
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

        # Load from cache or disk
        if self.preload:
            labels = self.data_cache[file_idx].clone()  # clone to avoid modifying cached data
        else:
            path = self.files[file_idx]
            with np.load(path, allow_pickle=False) as data:
                arr = data['arr_0'] if 'arr_0' in data else data[list(data.files)[0]]
            assert arr.shape == (32, 32, 32), f"Expected (32,32,32), got {arr.shape} from {path}"
            labels = torch.from_numpy(arr.astype(np.int64))  # [Z,Y,X]

        labels = self._apply_rot_flip(labels, kx, ky, kz, fx, fy, fz)
        if self.aug_perturb and self.perturb_prob > 0:
            labels = self._perturb(labels)

        onehot = self._one_hot(labels, 3)  # [3,Z,Y,X]
        return onehot, labels

# ----------------------
# Model: 3D VAE (UNet-ish Encoder/Decoder)
# ----------------------

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

class Encoder3D(nn.Module):
    def __init__(self, in_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.conv_in = nn.Conv3d(in_ch, base, 3, padding=1)
        self.down1 = nn.Sequential(ResBlock3D(base), nn.Conv3d(base, base*2, 4, stride=2, padding=1))   # 32->16
        self.down2 = nn.Sequential(ResBlock3D(base*2), nn.Conv3d(base*2, base*4, 4, stride=2, padding=1)) # 16->8
        self.down3 = nn.Sequential(ResBlock3D(base*4), nn.Conv3d(base*4, base*8, 4, stride=2, padding=1)) # 8->4
        self.mid = nn.Sequential(ResBlock3D(base*8), ResBlock3D(base*8), nn.GroupNorm(8, base*8), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mu = nn.Linear(base*8, latent_dim)
        self.logvar = nn.Linear(base*8, latent_dim)

    def forward(self, x):  # x: [B,3,32,32,32]
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.pool(h).flatten(1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

class Decoder3D(nn.Module):
    def __init__(self, out_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base*8)
        self.up0 = nn.Sequential(nn.Unflatten(1, (base*8, 1, 1, 1)))
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(base*8, base*4, 4, stride=2, padding=1),  # 1->2
            ResBlock3D(base*4),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base*4, base*2, 4, stride=2, padding=1),  # 2->4
            ResBlock3D(base*2),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(base*2, base, 4, stride=2, padding=1),    # 4->8
            ResBlock3D(base),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(base, base//2, 4, stride=2, padding=1),   # 8->16
            ResBlock3D(base//2),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(base//2, base//4, 4, stride=2, padding=1),# 16->32
            ResBlock3D(base//4),
        )
        self.out = nn.Conv3d(base//4, out_ch, 1)

    def forward(self, z):  # z: [B, latent]
        h = self.fc(z)
        h = self.up0(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        logits = self.out(h)  # [B,3,32,32,32]
        return logits

class VAE3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base, latent_dim)
        self.decoder = Decoder3D(out_ch, base, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

# ----------------------
# Losses
# ----------------------

def kl_divergence(mu, logvar):
    # KL(q(z|x) || N(0,I)) summed over latent dims, averaged over batch
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=1))

def parse_class_weights(arg: str):
    """Parse '--class_weights' like '1.0,0.6,0.7' or 'none'."""
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
# Training / Evaluation
# ----------------------

@torch.no_grad()
def save_volume_and_projections(vol_logits, out_npz, out_png):
    """vol_logits: [3,32,32,32] logits; save argmax labels and 3-view PNG."""
    import matplotlib.pyplot as plt
    probs = F.softmax(vol_logits, dim=0)
    labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)  # [Z,Y,X]
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, labels)

    # Simple projections (max over Z/Y/X)
    max_z = labels.max(axis=0)  # [Y,X]
    max_y = labels.max(axis=1)  # [Z,X]
    max_x = labels.max(axis=2)  # [Z,Y]

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

def train(args, resume_checkpoint=None):
    global_t0 = time.time()
    train_start_time = datetime.now()
    
    # Device selection with MPS support
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Initialize Rich console
    console = Console()
    
    # Create experiment directory under out_dir
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    
    # Check if experiment directory exists and is not empty (only for new training)
    if not resume_checkpoint:
        if os.path.exists(exp_dir):
            if os.listdir(exp_dir):  # Directory exists and is not empty
                console.print(Panel.fit(
                    f"[bold red]ERROR: Experiment Directory Not Empty[/bold red]\n\n"
                    f"The experiment directory already exists and contains files:\n"
                    f"[yellow]{exp_dir}[/yellow]\n\n"
                    f"To prevent data loss or confusion, training has been stopped.\n"
                    f"Please either:\n"
                    f"  • Choose a different experiment name (--exp_name)\n"
                    f"  • Delete or move the existing directory\n"
                    f"  • Clean the directory manually\n"
                    f"  • Use --resume to continue training from a checkpoint",
                    border_style="red"
                ))
                console.print(f"\n[red]Files in directory:[/red]")
                for item in os.listdir(exp_dir)[:10]:  # Show first 10 items
                    console.print(f"  - {item}")
                if len(os.listdir(exp_dir)) > 10:
                    console.print(f"  ... and {len(os.listdir(exp_dir)) - 10} more files")
                raise SystemExit(1)
    
    # Display training header
    header_text = (
        f"[bold cyan]Minecraft 3D VAE Training[/bold cyan]\n"
        f"Experiment: [magenta]{args.exp_name}[/magenta]\n"
        f"Device: [yellow]{device}[/yellow]\n"
        f"Output: [cyan]{exp_dir}[/cyan]\n"
    )
    if resume_checkpoint:
        header_text += f"Mode: [yellow]RESUME from epoch {resume_checkpoint['epoch']}[/yellow]\n"
    header_text += f"Started: [green]{train_start_time.strftime('%Y-%m-%d %H:%M:%S')}[/green]"
    
    console.print(Panel.fit(header_text, border_style="cyan"))

    # Collect files from train/val/test subdirectories
    train_files = sorted(glob(os.path.join(args.data_root, 'train', '*.npz')))
    val_files = sorted(glob(os.path.join(args.data_root, 'val', '*.npz')))
    test_files = sorted(glob(os.path.join(args.data_root, 'test', '*.npz')))
    
    assert len(train_files) > 0, f"No .npz found in {os.path.join(args.data_root, 'train')}"
    assert len(val_files) > 0, f"No .npz found in {os.path.join(args.data_root, 'val')}"
    assert len(test_files) > 0, f"No .npz found in {os.path.join(args.data_root, 'test')}"
    
    n_train = len(train_files)
    n_val = len(val_files)
    n_test = len(test_files)
    n_total = n_train + n_val + n_test
    
    # Display dataset info
    console.print(f"\n[bold]Dataset:[/bold] {n_train} train, {n_val} val, {n_test} test (total {n_total} files)")

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
    val_ds = VoxelDataset(val_files, aug_mode='random', aug_perturb=False, 
                         preload=args.preload, console=console)
    # Test dataset: explicitly disable all augmentations for reproducible results
    test_ds = VoxelDataset(
        test_files, 
        aug_mode='random',  # Mode doesn't matter when all aug flags are False
        aug_flip_x=False, aug_flip_y=False, aug_flip_z=False,
        aug_rot_x=False, aug_rot_y=False, aug_rot_z=False,
        aug_perturb=False,
        preload=args.preload, 
        console=console
    )

    # Only use pin_memory on CUDA devices
    use_pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=use_pin_memory)

    # ----- Class weights (manual) -----
    class_weights = parse_class_weights(args.class_weights)
    if class_weights is None:
        console.print("[bold]Class weights:[/bold] NONE (uniform)")
    else:
        # Explicitly show which weight corresponds to which class
        console.print(f"[bold]Class weights:[/bold] {class_weights.tolist()}")
        console.print(f"[dim]  • weight[0] = {class_weights[0]:.4f} → class 0 (air)[/dim]")
        console.print(f"[dim]  • weight[1] = {class_weights[1]:.4f} → class 1 (oak_log/wood)[/dim]")
        console.print(f"[dim]  • weight[2] = {class_weights[2]:.4f} → class 2 (oak_leaves)[/dim]")
    
    console.print(f"[bold]Augmentation:[/bold] mode={args.aug_mode}, "
                  f"Dataset size: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}\n")

    # Model
    model = VAE3D(in_ch=3, out_ch=3, base=args.base, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Use new API for GradScaler (torch >= 2.0)
    # Enable AMP for CUDA by default, but allow disabling via --no_amp flag
    # Disable for MPS (MPS has its own optimizations) and CPU
    use_amp = (device.type == 'cuda') and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None
    
    # Display AMP status
    if device.type == 'cuda':
        if use_amp:
            console.print(f"[bold]AMP:[/bold] [green]ENABLED[/green] (mixed precision training)")
        else:
            console.print(f"[bold]AMP:[/bold] [yellow]DISABLED[/yellow] (full precision training)")
    else:
        console.print(f"[bold]AMP:[/bold] [dim]N/A (not using CUDA)[/dim]")

    # Create experiment directory and subdirectories
    os.makedirs(exp_dir, exist_ok=True)
    samples_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Initialize training state
    start_epoch = 1
    best_val = math.inf
    training_history = []
    cumulative_time_offset = 0.0  # For resuming: add previous training time
    
    # Resume from checkpoint if provided
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint['model'])
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        start_epoch = resume_checkpoint['epoch'] + 1
        best_val = resume_checkpoint['best_val']
        training_history = resume_checkpoint.get('training_history', [])
        cumulative_time_offset = resume_checkpoint.get('cumulative_time_secs', 0.0)
        
        # Restore RNG states for reproducibility
        if 'rng_state' in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint['rng_state'])
        if 'cuda_rng_state' in resume_checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_checkpoint['cuda_rng_state'])
        if 'numpy_rng_state' in resume_checkpoint:
            np.random.set_state(resume_checkpoint['numpy_rng_state'])
        if 'python_rng_state' in resume_checkpoint:
            random.setstate(resume_checkpoint['python_rng_state'])
        
        # Restore scaler state if using AMP
        if scaler is not None and 'scaler' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler'])
        
        console.print(f"[green]✓[/green] Resumed from epoch {resume_checkpoint['epoch']}, best_val={best_val:.4f}")
        console.print(f"[cyan]Continuing training from epoch {start_epoch} to {args.epochs}[/cyan]\n")

    # Flag for handling Ctrl+C gracefully
    interrupted = {'flag': False, 'epoch': None}
    
    # Signal handler for Ctrl+C
    def signal_handler(signum, frame):
        console.print("\n[yellow]⚠ Ctrl+C detected! Saving checkpoint before exit...[/yellow]")
        interrupted['flag'] = True
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Helper function to save checkpoint
    def save_checkpoint(epoch, is_best=False, is_last=False):
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val': best_val,
            'training_history': training_history,
            'cumulative_time_secs': cumulative_time_offset + (time.time() - global_t0),
            'args': vars(args),
            # Save RNG states for reproducibility
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()
        
        if is_best:
            path = os.path.join(exp_dir, f'best_{args.exp_name}.pt')
            torch.save(checkpoint, path)
        if is_last:
            path = os.path.join(exp_dir, f'last_{args.exp_name}.pt')
            torch.save(checkpoint, path)
        return path if (is_best or is_last) else None

    # Create progress bar (remaining epochs + 1 for final test)
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
        
        overall_task = progress.add_task("[cyan]Training", total=remaining_epochs + 1)
        
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_t0 = time.time()
            
            # Create a single progress bar for both training and validation
            total_steps = len(train_loader) + len(val_loader)
            epoch_task = progress.add_task(
                f"[green]Epoch {epoch}/{args.epochs} - Training", 
                total=total_steps
            )
            
            # Training phase
            model.train()
            running = 0.0
            # Track class distribution and logits stats for debugging (first epoch only)
            if epoch == start_epoch:
                class_counts = torch.zeros(3, dtype=torch.long, device=device)
                logits_stats_collected = False  # Only collect stats from first batch
            
            for batch_idx, (onehot, labels) in enumerate(train_loader):
                # Check for interruption during training
                if interrupted['flag']:
                    break
                    
                onehot = onehot.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Track class distribution for debugging (first epoch only)
                if epoch == start_epoch:
                    for c in range(3):
                        class_counts[c] += (labels == c).sum().item()
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, mu, logvar = model(onehot)
                    
                    # Collect logits statistics for debugging (first epoch, first batch only)
                    if epoch == start_epoch and batch_idx == 0 and not logits_stats_collected:
                        # Convert to float32 for accurate statistics
                        logits_fp32 = logits.float().detach()
                        logits_mean = logits_fp32.mean().item()
                        logits_std = logits_fp32.std().item()
                        logits_min = logits_fp32.min().item()
                        logits_max = logits_fp32.max().item()
                        # Per-class logits stats (mean over spatial dimensions)
                        logits_per_class = logits_fp32.mean(dim=(2, 3, 4))  # [B, 3]
                        logits_per_class_mean = logits_per_class.mean(dim=0).cpu().tolist()  # [3]
                        
                        console.print(f"[dim]Logits statistics (first batch, epoch {epoch}):[/dim]")
                        console.print(f"[dim]  • Overall: mean={logits_mean:.4f}, std={logits_std:.4f}, min={logits_min:.4f}, max={logits_max:.4f}[/dim]")
                        console.print(f"[dim]  • Per class (mean logits):[/dim]")
                        console.print(f"[dim]    - class 0 (air): {logits_per_class_mean[0]:.4f}[/dim]")
                        console.print(f"[dim]    - class 1 (wood): {logits_per_class_mean[1]:.4f}[/dim]")
                        console.print(f"[dim]    - class 2 (leaves): {logits_per_class_mean[2]:.4f}[/dim]")
                        
                        # Check for potential underflow issues
                        if abs(logits_max) < 1.0 or abs(logits_min) < 1.0:
                            console.print(f"[yellow]⚠ Warning: Logits values are very small (max={logits_max:.4f}, min={logits_min:.4f})[/yellow]")
                            console.print(f"[yellow]  This might indicate precision issues, especially with AMP enabled.[/yellow]")
                            console.print(f"[yellow]  Try --no_amp to disable mixed precision training.[/yellow]")
                        
                        logits_stats_collected = True
                    
                    ce = F.cross_entropy(
                        logits, labels.long(),
                        weight=(class_weights.to(device) if class_weights is not None else None)
                    )
                    kl = kl_divergence(mu, logvar)
                    loss = ce + args.kl_beta * kl
                
                # Backward with or without AMP
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                running += loss.item() * labels.size(0)
                progress.update(epoch_task, advance=1)
            
            train_loss = running / len(train_loader.dataset)
            
            # Display class distribution for first epoch
            if epoch == start_epoch:
                total_voxels = class_counts.sum().item()
                if total_voxels > 0:
                    console.print(f"[dim]Class distribution (first batch):[/dim]")
                    console.print(f"[dim]  • class 0 (air): {class_counts[0].item():,} ({100*class_counts[0].item()/total_voxels:.1f}%)[/dim]")
                    console.print(f"[dim]  • class 1 (wood): {class_counts[1].item():,} ({100*class_counts[1].item()/total_voxels:.1f}%)[/dim]")
                    console.print(f"[dim]  • class 2 (leaves): {class_counts[2].item():,} ({100*class_counts[2].item()/total_voxels:.1f}%)[/dim]")
            
            # Switch to validation phase (update task description)
            progress.update(epoch_task, description=f"[yellow]Epoch {epoch}/{args.epochs} - Validation")

            # Validation phase (skip if interrupted during training)
            if interrupted['flag']:
                progress.remove_task(epoch_task)
                val_loss = float('inf')  # Mark as incomplete
            else:
                model.eval()
                running = 0.0
                with torch.no_grad():
                    for onehot, labels in val_loader:
                        # Check for interruption during validation
                        if interrupted['flag']:
                            break
                            
                        onehot = onehot.to(device)
                        labels = labels.to(device)
                        logits, mu, logvar = model(onehot)
                        ce = F.cross_entropy(
                            logits, labels.long(),
                            weight=(class_weights.to(device) if class_weights is not None else None)
                        )
                        kl = kl_divergence(mu, logvar)
                        loss = ce + args.kl_beta * kl
                        running += loss.item() * labels.size(0)
                        progress.update(epoch_task, advance=1)
                
                val_loss = running / len(val_loader.dataset)
                progress.remove_task(epoch_task)

            # Check for interruption after epoch completes
            if interrupted['flag']:
                interrupted['epoch'] = epoch
                
                # Since epoch was interrupted before completion, save the previous complete epoch
                # This ensures the incomplete epoch will be re-trained on resume
                save_epoch = epoch - 1
                
                console.print(f"\n[yellow]⚠ Training interrupted during epoch {epoch}[/yellow]")
                console.print(f"[cyan]Saving checkpoint at last complete epoch {save_epoch}...[/cyan]")
                console.print(f"[dim](Epoch {epoch} will be re-trained on resume)[/dim]")
                
                # Create a modified checkpoint with correct epoch number
                checkpoint = {
                    'epoch': save_epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val': best_val,
                    'training_history': training_history,  # Doesn't include incomplete epoch
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
                
                # Save checkpoint directly
                checkpoint_path = os.path.join(exp_dir, f'last_{args.exp_name}.pt')
                torch.save(checkpoint, checkpoint_path)
                
                console.print(f"[green]✓[/green] Checkpoint saved: last_{args.exp_name}.pt (epoch {save_epoch})")
                console.print(f"[cyan]To resume: python {os.path.basename(__file__)} --resume {exp_dir} --epochs {args.epochs}[/cyan]")
                raise KeyboardInterrupt("Training interrupted by user")
            
            epoch_secs = time.time() - epoch_t0
            cum_secs = cumulative_time_offset + (time.time() - global_t0)
            
            # Check if this is the best model
            is_best = val_loss < best_val
            
            # Record history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time_secs': epoch_secs,
                'cumulative_time_secs': cum_secs,
                'is_best': 'TRUE' if is_best else 'FALSE'
            })

            # Save best model
            if is_best:
                best_val = val_loss
                save_checkpoint(epoch, is_best=True)
            
            # Periodic checkpoint save (every N epochs)
            save_last_checkpoint = (epoch % args.save_every == 0) or (epoch == args.epochs)
            if save_last_checkpoint:
                save_checkpoint(epoch, is_last=True)
            
            # Display epoch summary
            best_marker = " | ★ Best!" if is_best else ""
            checkpoint_marker = " | 💾 Saved" if save_last_checkpoint else ""
            progress.console.print(
                f"Epoch {epoch:03d}: train {train_loss:.4f} | val {val_loss:.4f} | {fmt_secs(epoch_secs)}{best_marker}{checkpoint_marker}"
            )
            
            # Periodic samples
            if epoch % args.sample_every == 0:
                sample_task = progress.add_task(f"[blue]Generating samples", total=args.n_samples + 1)
                with torch.no_grad():
                    # Reconstruction of first val batch
                    for onehot, labels in val_loader:
                        onehot = onehot.to(device)
                        logits, mu, logvar = model(onehot)
                        rec = logits[0].detach().cpu()  # [3,32,32,32]
                        save_volume_and_projections(
                            rec,
                            os.path.join(samples_dir, f"rec_e{epoch}_{args.exp_name}.npz"),
                            os.path.join(samples_dir, f"rec_e{epoch}_{args.exp_name}.png"),
                        )
                        progress.update(sample_task, advance=1)
                        break
                    # Pure samples
                    z = torch.randn(args.n_samples, args.latent_dim, device=device)
                    logits = model.decoder(z)
                    for i in range(args.n_samples):
                        vol = logits[i].detach().cpu()
                        save_volume_and_projections(
                            vol,
                            os.path.join(samples_dir, f"sample_e{epoch}_{i}_{args.exp_name}.npz"),
                            os.path.join(samples_dir, f"sample_e{epoch}_{i}_{args.exp_name}.png"),
                        )
                        progress.update(sample_task, advance=1)
                progress.remove_task(sample_task)
            
            progress.update(overall_task, advance=1)
        
        # Final test (integrated into main progress bar)
        best_model_path = os.path.join(exp_dir, f'best_{args.exp_name}.pt')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=False)['model'])
        model.eval()
        running = 0.0
        
        test_task = progress.add_task("[yellow]Final Test", total=len(test_loader))
        with torch.no_grad():
            for onehot, labels in test_loader:
                onehot = onehot.to(device)
                labels = labels.to(device)
                logits, mu, logvar = model(onehot)
                ce = F.cross_entropy(
                    logits, labels.long(),
                    weight=(class_weights.to(device) if class_weights is not None else None)
                )
                kl = kl_divergence(mu, logvar)
                loss = ce + args.kl_beta * kl
                running += loss.item() * labels.size(0)
                progress.update(test_task, advance=1)
        
        test_loss = running / len(test_loader.dataset)
        progress.remove_task(test_task)
        progress.update(overall_task, advance=1)
        
        # Print final test result
        progress.console.print(f"Final Test: test loss {test_loss:.4f}")
    
    # Training complete, calculate total time
    total_secs = cumulative_time_offset + (time.time() - global_t0)
    train_end_time = datetime.now()
    
    # Display final results
    final_table = Table(title="[bold cyan]Training Summary[/bold cyan]", box=box.ROUNDED)
    final_table.add_column("Metric", style="cyan", no_wrap=True)
    final_table.add_column("Value", style="magenta")
    final_table.add_row("Best Val Loss", f"{best_val:.6f}")
    final_table.add_row("Final Test Loss", f"{test_loss:.6f}")
    final_table.add_row("Total Runtime", fmt_secs(total_secs))
    final_table.add_row("Epochs Trained", f"{start_epoch} - {args.epochs}")
    final_table.add_row("Started", train_start_time.strftime('%Y-%m-%d %H:%M:%S'))
    final_table.add_row("Completed", train_end_time.strftime('%Y-%m-%d %H:%M:%S'))
    final_table.add_row("Best Model", best_model_path)
    final_table.add_row("Last Checkpoint", os.path.join(exp_dir, f'last_{args.exp_name}.pt'))
    console.print("\n", final_table, "\n")
    
    # Save CSV 1: Training history (train/val/test losses)
    csv1_path = os.path.join(exp_dir, f'training_history_{args.exp_name}.csv')
    with open(csv1_path, 'w', newline='') as f:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'epoch_time_secs', 'cumulative_time_secs', 'is_best']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_history)
    console.print(f"[green]✓[/green] Saved training history to [cyan]{csv1_path}[/cyan]")
    
    # Save CSV 2: Experiment metadata and parameters
    csv2_path = os.path.join(exp_dir, f'experiment_metadata_{args.exp_name}.csv')
    
    # Construct loss function description
    if class_weights is not None:
        ce_desc = f"CrossEntropyLoss(weight=[{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}])"
    else:
        ce_desc = "CrossEntropyLoss(weight=None)"
    loss_formula = f"Loss = {ce_desc} + {args.kl_beta} * KL_Divergence(mu, logvar)"
    
    # Helper function to convert boolean to TRUE/FALSE for better Excel/Sheets compatibility
    def bool_to_str(val):
        return 'TRUE' if val else 'FALSE'
    
    metadata = {
        # Experiment info
        'exp_name': args.exp_name,
        'resumed_from': args.resume if args.resume else 'None',
        'start_epoch': start_epoch,
        'end_epoch': args.epochs,
        'training_start_time': train_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_end_time': train_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_model_path': best_model_path,
        'last_checkpoint_path': os.path.join(exp_dir, f'last_{args.exp_name}.pt'),
        'samples_directory': samples_dir,
        
        # Data info
        'data_root': args.data_root,
        'out_dir': args.out_dir,
        'exp_dir': exp_dir,
        'n_train_files': n_train,
        'n_val_files': n_val,
        'n_test_files': n_test,
        'n_total_files': n_total,
        'train_dataset_size': len(train_ds),
        'val_dataset_size': len(val_ds),
        'test_dataset_size': len(test_ds),
        
        # Loss function parameters (key parameters)
        'class_weights': args.class_weights,
        'kl_beta': args.kl_beta,
        'loss_kl_weight': args.kl_beta,
        
        # Training hyperparameters
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
        
        # Model architecture
        'base': args.base,
        'latent_dim': args.latent_dim,
        
        # Augmentation settings
        'aug_mode': args.aug_mode,
        'aug_rot_x': bool_to_str(args.aug_rot_x),
        'aug_rot_y': bool_to_str(args.aug_rot_y),
        'aug_rot_z': bool_to_str(args.aug_rot_z),
        'aug_flip_x': bool_to_str(args.aug_flip_x),
        'aug_flip_y': bool_to_str(args.aug_flip_y),
        'aug_flip_z': bool_to_str(args.aug_flip_z),
        'aug_perturb': bool_to_str(args.aug_perturb),
        'perturb_prob': args.perturb_prob,
        
        # Sampling settings
        'sample_every': args.sample_every,
        'n_samples': args.n_samples,
        
        # Checkpoint settings
        'save_every': args.save_every,
        
        # Training results
        'best_val_loss': best_val,
        'final_test_loss': test_loss,
        'total_training_time_secs': total_secs,
        'total_training_time_formatted': fmt_secs(total_secs),
        
        # Loss function details (at the end for flexibility)
        'loss_function': loss_formula,
        'loss_reconstruction': ce_desc,
        'loss_kl_divergence': 'KL(q(z|x) || N(0,I))',
    }
    with open(csv2_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value'])
        for key, value in metadata.items():
            writer.writerow([key, value])
    console.print(f"[green]✓[/green] Saved experiment metadata to [cyan]{csv2_path}[/cyan]")
    
    # Save CSV 3: Experiment metadata (flat format for easy comparison)
    csv3_path = os.path.join(exp_dir, f'experiment_metadata_flat_{args.exp_name}.csv')
    with open(csv3_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)
    console.print(f"[green]✓[/green] Saved experiment metadata (flat) to [cyan]{csv3_path}[/cyan]")
    
    console.print(Panel.fit(
        "[bold green]Training Complete! 🎉[/bold green]",
        border_style="green"
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Tree 3D VAE")
    parser.add_argument('--data_root', type=str, required=False, help='Directory with train/val/test subdirectories')
    parser.add_argument('--data_zip', type=str, required=False, help='Zip file containing train/val/test subdirectories (automatically extracted)')
    parser.add_argument('--out_dir', type=str, required=False, help='Base output directory')
    parser.add_argument('--exp_name', type=str, required=False, help='Experiment name (creates subdirectory in out_dir)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base', type=int, default=64, help='UNet base channels')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--kl_beta', type=float, default=1e-3, help='KL weight; raise for more regularization')
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Augmentation toggles & mode
    parser.add_argument('--aug_mode', type=str, default='enumerate', choices=['enumerate', 'random'],
                        help="Augmentation mode: 'enumerate' (Cartesian expansion) or 'random' (on-the-fly)")
    parser.add_argument('--aug_rot_x', action='store_true', help='Enable rotations 0/90/180/270° around X axis (YZ plane)')
    parser.add_argument('--aug_rot_y', action='store_true', help='Enable rotations 0/90/180/270° around Y axis (ZX plane)')
    parser.add_argument('--aug_rot_z', action='store_true', help='Enable rotations 0/90/180/270° around Z axis (XY plane)')
    parser.add_argument('--aug_flip_x', action='store_true', help='Enable mirror along X (flip X dimension)')
    parser.add_argument('--aug_flip_y', action='store_true', help='Enable mirror along Y (flip Y dimension)')
    parser.add_argument('--aug_flip_z', action='store_true', help='Enable mirror along Z (flip Z dimension)')
    parser.add_argument('--aug_perturb', action='store_true', help='Enable random label perturbation')
    parser.add_argument('--perturb_prob', type=float, default=0.01, help='Per-voxel perturb probability when enabled')

    # Manual class weights (air,log,leaf). Use 'none' to disable.
    parser.add_argument('--class_weights', type=str, default='none',
                        help="Comma-separated weights for classes [air,log,leaf], e.g. '1.0,2.0,2.0'. "
                             "Higher weight = higher penalty for misclassification. "
                             "Classes: 0=air, 1=oak_log/wood, 2=oak_leaves. "
                             "Use 'none' to disable. "
                             "Note: If model outputs are mostly empty (all air), try increasing weights for classes 1 and 2.")

    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable Automatic Mixed Precision (AMP) even on CUDA. '
                             'Useful for debugging precision issues when model outputs are mostly empty (all air). '
                             'AMP can cause underflow in logits/gradients for highly imbalanced classification.')
    parser.add_argument('--preload', action='store_true', 
                        help='Preload all .npz files into RAM (recommended for small datasets)')
    
    # Resume and checkpoint options
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training (e.g., ./runs/exp1/last_exp1.pt)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    args = parser.parse_args()
    
    # Handle resume logic: load parameters from checkpoint if not provided
    resume_checkpoint = None
    if args.resume:
        import sys
        
        # Support directory path: auto-find checkpoint file
        resume_path = args.resume
        if os.path.isdir(resume_path):
            # Try to find last_*.pt or best_*.pt in the directory
            last_files = glob(os.path.join(resume_path, 'last_*.pt'))
            best_files = glob(os.path.join(resume_path, 'best_*.pt'))
            
            if last_files:
                resume_path = last_files[0]
                print(f"[✓] Found checkpoint: {resume_path}")
            elif best_files:
                resume_path = best_files[0]
                print(f"[✓] Found checkpoint: {resume_path}")
            else:
                print(f"[✗] ERROR: No checkpoint files (last_*.pt or best_*.pt) found in {args.resume}")
                raise SystemExit(1)
            args.resume = resume_path
        
        # Load checkpoint to get original parameters
        if not os.path.exists(args.resume):
            print(f"[✗] ERROR: Checkpoint file not found: {args.resume}")
            raise SystemExit(1)
        
        print(f"[→] Loading checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        checkpoint_args = resume_checkpoint.get('args', {})
        
        # Determine which arguments were explicitly provided by the user
        # by checking sys.argv
        explicitly_set = set()
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith('--'):
                # Remove leading dashes and convert to underscore format
                arg_name = arg[2:].replace('-', '_')
                explicitly_set.add(arg_name)
                # Skip the next item if it's not a flag (i.e., it's a value)
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        # Start with checkpoint args as base
        merged_args = checkpoint_args.copy()
        
        # Override with explicitly provided command-line arguments
        for key in vars(args):
            if key in explicitly_set:
                merged_args[key] = getattr(args, key)
        
        # Apply merged args back to args namespace
        for key, value in merged_args.items():
            setattr(args, key, value)
        
        # Display loaded parameters and overrides
        print(f"[✓] Loaded parameters from checkpoint")
        print(f"    • data_root: {args.data_root}")
        print(f"    • out_dir: {args.out_dir}")
        print(f"    • exp_name: {args.exp_name}")
        print(f"    • epoch: {resume_checkpoint['epoch']} → continuing to {args.epochs}")
        
        if explicitly_set - {'resume'}:  # Show overrides (excluding --resume itself)
            overridden = explicitly_set - {'resume'}
            print(f"    • Overridden parameters: {', '.join(sorted(overridden))}")
        
        print()
    
    # Validate required parameters
    if not args.data_root and not args.data_zip:
        parser.error("Either --data_root or --data_zip is required (unless resuming from checkpoint)")
    if args.data_root and args.data_zip:
        parser.error("Cannot specify both --data_root and --data_zip. Use only one.")
    if not args.out_dir:
        parser.error("--out_dir is required (unless resuming from checkpoint)")
    if not args.exp_name:
        parser.error("--exp_name is required (unless resuming from checkpoint)")
    
    # Handle zip file extraction
    temp_dir_holder = []  # Keep reference to temp directory during training
    if args.data_zip:
        console = Console()
        extract_dir, temp_dir = extract_zip_to_temp(args.data_zip, console=console)
        args.data_root = extract_dir
        temp_dir_holder.append(temp_dir)  # Keep reference alive
        console.print(f"[green]✓[/green] Using extracted directory as data_root: {extract_dir}")
        console.print(f"[dim]Temporary directory will be cleaned up after training completes[/dim]\n")
    
    seed_everything(args.seed)
    train(args, resume_checkpoint=resume_checkpoint)
    
    # Note: temp_dir_holder will go out of scope after train() completes,
    # which will trigger cleanup of the temporary directory