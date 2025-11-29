#!/usr/bin/env python3
"""
Offline latent diagnostic tool for 3D VAE.

It measures:
  1. Pairwise distances
  2. Per-channel mean/std with collapse detection
  3. PCA scatter plot
  4. KL stats (if logvar available)
  5. Enhanced linear interpolation test (tree shape transition)
  6. Decoder(random noise) sanity check

Collapse Detection:
  - Channel variance collapse: Detects if all channels have very low variance (< 0.1)
  - Uniform variance collapse: Detects if all channels have similar variance (encoder not encoding diverse structure)
  - Interpolation collapse: Detects if linear interpolation between trees produces blob-like results

Supports:
  - Single model file: --vae_ckpt model.pt
  - Directory with models: --vae_ckpt ./models/ (recursively finds all .pt files)
  
Output:
  - For single file: placed in a directory named "latent" next to the model file.
  - For directory input: results are also organized in "latent_summary/<model_name>/" folder 
    at the first level of the input directory, containing all validation results.
"""

import argparse
import os
import shutil
from pathlib import Path
from glob import glob
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: pandas not available. CSV merging will be disabled. Install with: pip install pandas")
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.ndimage import label

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table

# Try to import from training scripts
UNet3DVAE = None
ShallowUNet3DVAE = None
MidUNet3DVAE = None
VoxelDataset = None

# Try to import from evaluate_models first (has all variants)
try:
    from evaluate_models import UNet3DVAE, ShallowUNet3DVAE, MidUNet3DVAE, VoxelDataset
except ImportError:
    # Fallback to training scripts
    try:
        from train_3D_UNetVAE_16latent import UNet3DVAE, VoxelDataset
    except ImportError:
        try:
            from train_3D_UNetVAE_20251108 import UNet3DVAE, VoxelDataset
        except ImportError:
            raise ImportError(
                "Could not find evaluate_models, train_3D_UNetVAE_16latent, or train_3D_UNetVAE_20251108. "
                "Please ensure the training script directories are accessible."
            )


def flatten_latent(z):
    return z.reshape(z.shape[0], -1)


def reparameterize(mu, logvar):
    """Reparameterization trick: z = mu + eps * sigma where eps ~ N(0,1)"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


@torch.no_grad()
def collect_latents(vae, loader, device, max_samples=300, progress=None, task=None, use_amp=True):
    """
    Collect latent vectors from the VAE encoder.
    
    Args:
        vae: VAE model
        loader: DataLoader (should have shuffle=True for random sampling)
        device: torch device
        max_samples: Maximum number of samples to collect
                     Note: Samples are selected sequentially from the shuffled DataLoader.
                     Since DataLoader has shuffle=True, each run will get different random samples.
        progress: Optional progress bar
        task: Optional progress task ID
        use_amp: Whether to use Automatic Mixed Precision (AMP) for inference.
                 Default: True (enabled for faster inference on CUDA devices)
    
    Returns:
        latents: Collected latent vectors [N, C, H, W, D]
        mus: Mean vectors [N, C, H, W, D]
        logvars: Log variance vectors [N, C, H, W, D]
    """
    latents = []
    mus = []
    logvars = []

    # Iterate through DataLoader and collect up to max_samples
    # The DataLoader should have shuffle=True to get random samples
    # Each batch has batch_size=1, so we collect one sample per iteration
    for i, (x, _) in enumerate(loader):
        if i >= max_samples:
            break
        x = x.to(device)
        
        # Use AMP if enabled and on CUDA
        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                # UNet3DVAE uses encoder() method, not encode()
                mu, logvar, _ = vae.encoder(x, return_skips=False)
                # Reparameterize to get z - use method if available, otherwise use function
                if hasattr(vae, 'reparameterize'):
                    z = vae.reparameterize(mu, logvar)
                else:
                    z = reparameterize(mu, logvar)
        else:
            # UNet3DVAE uses encoder() method, not encode()
            mu, logvar, _ = vae.encoder(x, return_skips=False)
            # Reparameterize to get z - use method if available, otherwise use function
            if hasattr(vae, 'reparameterize'):
                z = vae.reparameterize(mu, logvar)
            else:
                z = reparameterize(mu, logvar)

        latents.append(z.cpu())
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
        
        if progress and task is not None:
            progress.update(task, advance=1)

    latents = torch.cat(latents, dim=0)
    mus = torch.cat(mus, dim=0)
    logvars = torch.cat(logvars, dim=0)
    return latents, mus, logvars


def save_pca_plot(latents_flat, out_path):
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(latents_flat)

    plt.figure(figsize=(6, 6))
    plt.scatter(z2[:, 0], z2[:, 1], s=8, alpha=0.7)
    plt.title("Latent PCA scatter")
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_iou(img1, img2):
    """Compute Intersection over Union (IoU) between two binary segmentations.
    
    For multi-class segmentation, we compute IoU for each non-zero class
    and return the mean IoU across all classes.
    """
    # Convert to binary: non-zero voxels are foreground
    binary1 = (img1 > 0).astype(np.float32)
    binary2 = (img2 > 0).astype(np.float32)
    
    intersection = np.sum(binary1 * binary2)
    union = np.sum(np.maximum(binary1, binary2))
    
    if union == 0:
        return 1.0  # Both are empty, perfect match
    
    return float(intersection / union)


def compute_connected_components(img):
    """Compute the number of connected components in a 3D image.
    
    Uses 6-connectivity (face-connected neighbors) in 3D.
    Only face-adjacent voxels are considered connected (not edge or corner neighbors).
    """
    # Convert to binary: non-zero voxels are foreground
    binary = (img > 0).astype(np.int32)
    
    # Use 6-connectivity (face-connected) in 3D
    # Structure: center + 6 face neighbors (up, down, left, right, front, back)
    structure = np.zeros((3, 3, 3), dtype=np.int32)
    structure[1, 1, 1] = 1  # Center
    structure[0, 1, 1] = 1  # Up
    structure[2, 1, 1] = 1  # Down
    structure[1, 0, 1] = 1  # Left
    structure[1, 2, 1] = 1  # Right
    structure[1, 1, 0] = 1  # Front
    structure[1, 1, 2] = 1  # Back
    
    labeled, num_components = label(binary, structure=structure)
    
    return int(num_components)


def linear_interpolation(vae, z1, z2, out_path, device, steps=10, use_amp=True):
    """
    Enhanced linear interpolation test for tree shape transition.
    
    Tests if latent can linearly interpolate between two different trees.
    z(t) = (1-t)*z1 + t*z2
    
    If latent has representation power:
    - Tree shapes should gradually transition
    - Leaves and trunk positions should smoothly change
    
    If collapse:
    - All interpolations become blob
    - Almost no variation
    
    Returns metrics including:
    - Per-step IoU difference (most important for semantic shape continuity)
    - Connected Component continuity (topology continuity)
    - Mean Frame Diff % (quick screening metric)
    """
    zs = [(1 - t) * z1 + t * z2 for t in np.linspace(0, 1, steps)]
    
    # Decode all interpolations
    decoded_images = []
    for z in zs:
        with torch.no_grad():
            z_input = z.unsqueeze(0).to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = vae.decoder(z_input, skips=None)[0].cpu().detach().numpy()
            else:
                logits = vae.decoder(z_input, skips=None)[0].cpu().detach().numpy()
        decoded_images.append(logits.argmax(0))
    
    # ============================================================
    # No.1: Per-step IoU difference (最重要)
    # ============================================================
    # Compute IoU between consecutive frames
    iou_values = []
    iou_differences = []
    for i in range(len(decoded_images) - 1):
        iou = compute_iou(decoded_images[i], decoded_images[i + 1])
        iou_values.append(iou)
        # IoU difference: 1 - IoU (higher = more different/broken)
        iou_diff = 1.0 - iou
        iou_differences.append(iou_diff)
    
    mean_iou = np.mean(iou_values) if iou_values else 0.0
    mean_iou_diff = np.mean(iou_differences) if iou_differences else 0.0
    max_iou_diff = np.max(iou_differences) if iou_differences else 0.0
    std_iou_diff = np.std(iou_differences) if iou_differences else 0.0
    
    # ============================================================
    # No.2: Connected Component Continuity
    # ============================================================
    # Compute connected component count for each frame
    cc_counts = []
    cc_changes = []
    for i, img in enumerate(decoded_images):
        cc_count = compute_connected_components(img)
        cc_counts.append(cc_count)
        if i > 0:
            # Absolute change in CC count (detects sudden splits/merges)
            cc_change = abs(cc_count - cc_counts[i - 1])
            cc_changes.append(cc_change)
    
    mean_cc_count = np.mean(cc_counts) if cc_counts else 0.0
    mean_cc_change = np.mean(cc_changes) if cc_changes else 0.0
    max_cc_change = np.max(cc_changes) if cc_changes else 0
    std_cc_count = np.std(cc_counts) if cc_counts else 0.0
    
    # ============================================================
    # No.3: Mean Frame Diff (%) (already computed, keep for compatibility)
    # ============================================================
    # Compute pairwise differences between consecutive frames
    frame_diffs = []
    for i in range(len(decoded_images) - 1):
        diff = np.sum(decoded_images[i] != decoded_images[i + 1])
        frame_diffs.append(diff)
    
    mean_frame_diff = np.mean(frame_diffs) if frame_diffs else 0
    total_voxels = decoded_images[0].size
    mean_frame_diff_ratio = mean_frame_diff / total_voxels if total_voxels > 0 else 0.0
    
    # Create visualization with max projection along Z
    fig = plt.figure(figsize=(steps * 1.5, 3))
    for i, img in enumerate(decoded_images):
        # max projection along Z
        proj = img.max(axis=0)
        
        plt.subplot(1, steps, i + 1)
        plt.imshow(proj, cmap='viridis')
        plt.axis("off")
        plt.title(f"t={i/(steps-1):.2f}", fontsize=8)
    
    plt.suptitle(f"Linear Interpolation (IoU diff: {mean_iou_diff:.3f}, CC change: {mean_cc_change:.1f}, Frame diff: {mean_frame_diff_ratio*100:.2f}%)", 
                 fontsize=10, y=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return collapse indicator
    # If mean frame difference is very small (< 1% of voxels), likely collapse
    collapse_threshold = 0.01  # 1% of voxels
    is_collapsed = mean_frame_diff_ratio < collapse_threshold
    
    return {
        # No.1: IoU metrics (最重要)
        'mean_iou': float(mean_iou),
        'mean_iou_diff': float(mean_iou_diff),  # Per-step IoU difference (most important)
        'max_iou_diff': float(max_iou_diff),
        'std_iou_diff': float(std_iou_diff),
        
        # No.2: Connected Component metrics
        'mean_cc_count': float(mean_cc_count),
        'mean_cc_change': float(mean_cc_change),  # Connected Component continuity
        'max_cc_change': float(max_cc_change),
        'std_cc_count': float(std_cc_count),
        
        # No.3: Frame Diff metrics (for quick screening)
        'mean_frame_diff_ratio': float(mean_frame_diff_ratio),  # Mean Frame Diff (%)
        'mean_frame_diff': float(mean_frame_diff),
        'total_voxels': int(total_voxels),
        
        # Legacy collapse indicator
        'is_collapsed': is_collapsed,
        
        # Per-step details (for debugging/analysis)
        'iou_values': [float(x) for x in iou_values],
        'iou_differences': [float(x) for x in iou_differences],
        'cc_counts': [int(x) for x in cc_counts],
        'cc_changes': [int(x) for x in cc_changes],
    }


def detect_checkpoint_type(state_dict):
    """Detect if checkpoint is VAE or Diffusion model based on keys."""
    keys = list(state_dict.keys())
    
    # Diffusion model indicators
    has_diffusion_keys = any(
        k.startswith(("time_mlp", "in_conv", "down1", "mid", "upsample", "up1", "out_conv"))
        for k in keys
    )
    
    # VAE model indicators
    has_vae_keys = any(
        k.startswith(("encoder.", "decoder."))
        for k in keys
    )
    
    if has_diffusion_keys and not has_vae_keys:
        return "diffusion"
    elif has_vae_keys and not has_diffusion_keys:
        return "vae"
    else:
        return "unknown"


def detect_vae_variant(state_dict):
    """Detect VAE variant: shallow, mid, or deep based on encoder structure."""
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
    """Parse boolean value from various formats."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(val)


def resolve_skip_levels(args, default: int = 3) -> int:
    """Resolve skip_levels from args dict, with fallbacks."""
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


def find_model_files(input_path):
    """Find all .pt model files, recursively if input is a directory."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        if input_path.suffix == ".pt":
            return [str(input_path)]
        else:
            raise ValueError(f"Input file must be a .pt file, got: {input_path}")
    elif input_path.is_dir():
        # Directory - recursively find all .pt files
        pattern = str(input_path / "**" / "*.pt")
        files = glob(pattern, recursive=True)
        if not files:
            raise ValueError(f"No .pt files found in directory: {input_path}")
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def get_output_dir(model_path):
    """Generate output directory name based on model path."""
    model_path = Path(model_path)
    output_dir = model_path.parent / "latent"
    return str(output_dir)


def copy_results_to_latent_data(source_dir, target_dir, model_name, console):
    """Copy all validation results from source_dir to target_dir/latent_summary/model_name/"""
    source_path = Path(source_dir)
    target_path = Path(target_dir) / "latent_summary" / model_name
    
    if not source_path.exists():
        return False
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from source to target
    files_copied = 0
    for file_path in source_path.glob("*"):
        if file_path.is_file():
            shutil.copy2(file_path, target_path / file_path.name)
            files_copied += 1
    
    return files_copied > 0


def merge_csv_files(latent_data_dir, csv_filename, output_filename, console):
    """Merge all CSV files with the same name from different model directories."""
    if pd is None:
        console.print(f"[yellow]⚠[/yellow] pandas not available, skipping CSV merge for {csv_filename}")
        return False
    
    latent_data_path = Path(latent_data_dir)
    if not latent_data_path.exists():
        return False
    
    all_data = []
    model_names = []
    
    # Find all model directories and sort by name
    for model_dir in sorted(latent_data_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        csv_path = model_dir / csv_filename
        if csv_path.exists():
            try:
                # Try to read as CSV with header
                df = pd.read_csv(csv_path)
                # Add model name column
                df.insert(0, 'model_name', model_dir.name)
                all_data.append(df)
                model_names.append(model_dir.name)
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Failed to read {csv_path}: {e}")
                continue
    
    if not all_data:
        return False
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by model_name to ensure consistent ordering
    if 'model_name' in merged_df.columns:
        merged_df = merged_df.sort_values('model_name').reset_index(drop=True)
    
    # Save merged CSV
    output_path = latent_data_path / output_filename
    merged_df.to_csv(output_path, index=False)
    return True


def merge_channel_std_csv(latent_data_dir, csv_filename, output_filename, console):
    """
    Merge channel std CSV files with models as rows and channels as columns.
    
    Format:
    - Rows: model names
    - Columns: channel indices
    - Values: std values for each channel in each model
    """
    if pd is None:
        console.print(f"[yellow]⚠[/yellow] pandas not available, skipping channel std CSV merge")
        return False
    
    latent_data_path = Path(latent_data_dir)
    if not latent_data_path.exists():
        return False
    
    model_data = {}  # {model_name: {channel_idx: std_value}}
    all_channels = set()
    
    # Find all model directories
    for model_dir in sorted(latent_data_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        csv_path = model_dir / csv_filename
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                model_name = model_dir.name
                
                # Handle different CSV formats
                if 'channel_idx' in df.columns and 'std' in df.columns:
                    # New format with channel_idx and std columns
                    channel_data = dict(zip(df['channel_idx'], df['std']))
                elif 'channel_std' in df.columns:
                    # Old format: single column named 'channel_std'
                    channel_data = {i: val for i, val in enumerate(df['channel_std'])}
                elif len(df.columns) == 1:
                    # Single column without header or with different name
                    col_name = df.columns[0]
                    channel_data = {i: val for i, val in enumerate(df[col_name])}
                else:
                    # Try to find std column
                    std_col = None
                    for col in df.columns:
                        if 'std' in col.lower():
                            std_col = col
                            break
                    if std_col:
                        channel_data = {i: val for i, val in enumerate(df[std_col])}
                    else:
                        console.print(f"[yellow]⚠[/yellow] Could not parse {csv_path}, skipping")
                        continue
                
                model_data[model_name] = channel_data
                all_channels.update(channel_data.keys())
                
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Failed to read {csv_path}: {e}")
                continue
    
    if not model_data:
        return False
    
    # Create DataFrame: rows = models, columns = channels
    all_channels = sorted(all_channels)
    merged_data = {}
    
    for model_name in sorted(model_data.keys()):
        channel_data = model_data[model_name]
        # Fill in std values for each channel (use NaN if channel doesn't exist for this model)
        merged_data[model_name] = [channel_data.get(ch, np.nan) for ch in all_channels]
    
    # Create DataFrame with channels as index (rows) and models as columns, then transpose
    # First: rows = channels, columns = models
    # After transpose: rows = models, columns = channels
    column_names = [f'channel_{ch}' for ch in all_channels]
    merged_df = pd.DataFrame(merged_data, index=all_channels).T
    merged_df.columns = column_names
    merged_df.index.name = 'model_name'
    
    # Save merged CSV
    output_path = latent_data_path / output_filename
    merged_df.to_csv(output_path, index=True)
    return True


def combine_png_images(latent_data_dir, png_filename, output_filename, console, max_cols=4, max_images_per_file=9):
    """Combine multiple PNG images into a grid layout. If more than max_images_per_file, split into multiple files."""
    latent_data_path = Path(latent_data_dir)
    if not latent_data_path.exists():
        return False
    
    image_paths = []
    model_names = []
    
    # Find all PNG files with the same name from different model directories
    for model_dir in sorted(latent_data_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        png_path = model_dir / png_filename
        if png_path.exists():
            image_paths.append(png_path)
            model_names.append(model_dir.name)
    
    if not image_paths:
        return False
    
    n_images = len(image_paths)
    
    # Split into chunks if more than max_images_per_file
    if n_images > max_images_per_file:
        # Split into chunks of max_images_per_file
        num_files = (n_images + max_images_per_file - 1) // max_images_per_file
        
        # Get base filename without extension
        output_base = Path(output_filename)
        base_name = output_base.stem
        ext = output_base.suffix
        
        for file_idx in range(num_files):
            start_idx = file_idx * max_images_per_file
            end_idx = min(start_idx + max_images_per_file, n_images)
            
            chunk_paths = image_paths[start_idx:end_idx]
            chunk_names = model_names[start_idx:end_idx]
            
            # Calculate grid dimensions for this chunk (3x3 for 9 images)
            chunk_size = len(chunk_paths)
            n_cols_chunk = 3  # Use 3 columns for 9 images max
            n_rows_chunk = (chunk_size + n_cols_chunk - 1) // n_cols_chunk
            
            # Create figure with subplots
            fig = plt.figure(figsize=(n_cols_chunk * 4, n_rows_chunk * 4))
            gs = GridSpec(n_rows_chunk, n_cols_chunk, figure=fig, hspace=0.3, wspace=0.3)
            
            for idx, (img_path, model_name) in enumerate(zip(chunk_paths, chunk_names)):
                row = idx // n_cols_chunk
                col = idx % n_cols_chunk
                ax = fig.add_subplot(gs[row, col])
                
                try:
                    img = plt.imread(img_path)
                    # Handle different image formats (RGB, RGBA, grayscale)
                    if len(img.shape) == 2:
                        ax.imshow(img, cmap='gray')
                    else:
                        ax.imshow(img)
                    ax.set_title(model_name, fontsize=10, pad=5, wrap=True)
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading\n{model_name}", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=8, color='red')
                    ax.axis('off')
                    console.print(f"[yellow]⚠[/yellow] Failed to load {img_path}: {e}")
            
            # Save combined image with index
            output_path = latent_data_path / f"{base_name}_{file_idx + 1}{ext}"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return True
    else:
        # Original behavior for <= max_images_per_file images
        # Calculate grid dimensions
        n_cols = min(max_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, (img_path, model_name) in enumerate(zip(image_paths, model_names)):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            try:
                img = plt.imread(img_path)
                # Handle different image formats (RGB, RGBA, grayscale)
                if len(img.shape) == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                ax.set_title(model_name, fontsize=10, pad=5, wrap=True)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{model_name}", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=8, color='red')
                ax.axis('off')
                console.print(f"[yellow]⚠[/yellow] Failed to load {img_path}: {e}")
        
        # Save combined image
        output_path = latent_data_path / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True


def merge_kl_stats(latent_data_dir, console):
    """Merge all KL stats text files into a single CSV."""
    if pd is None:
        console.print(f"[yellow]⚠[/yellow] pandas not available, skipping KL stats merge")
        return False
    
    latent_data_path = Path(latent_data_dir)
    if not latent_data_path.exists():
        return False
    
    kl_data = []
    
    # Find all kl_stats.txt files
    for model_dir in sorted(latent_data_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        kl_path = model_dir / "kl_stats.txt"
        if kl_path.exists():
            try:
                with open(kl_path, 'r') as f:
                    content = f.read().strip()
                    # Extract KL value (format: mean_KL_per_element = 0.123456)
                    if '=' in content:
                        value_str = content.split('=')[1].strip()
                        try:
                            kl_value = float(value_str)
                            kl_data.append({
                                'model_name': model_dir.name,
                                'mean_KL_per_element': kl_value
                            })
                        except ValueError:
                            pass
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Failed to read {kl_path}: {e}")
                continue
    
    if not kl_data:
        return False
    
    # Create DataFrame and save
    df = pd.DataFrame(kl_data)
    output_path = latent_data_path / "merged_kl_stats.csv"
    df.to_csv(output_path, index=False)
    return True


def create_summary_reports(latent_data_dir, console):
    """Create merged summary reports from all model results."""
    console.print(f"[cyan]Creating summary reports...[/cyan]")
    
    # Merge CSV files
    if merge_csv_files(latent_data_dir, "latent_pairwise_stats.csv", 
                      "merged_pairwise_stats.csv", console):
        console.print(f"[green]✓[/green] Merged pairwise stats CSV")
    # Merge channel std with special format: rows=models, columns=channels
    if merge_channel_std_csv(latent_data_dir, "latent_channel_std.csv", 
                             "merged_channel_std.csv", console):
        console.print(f"[green]✓[/green] Merged channel std CSV (models as rows, channels as columns)")
    
    # Merge KL stats
    if merge_kl_stats(latent_data_dir, console):
        console.print(f"[green]✓[/green] Merged KL stats CSV")
    
    # Merge interpolation stats (IoU, CC, Frame Diff metrics)
    if merge_csv_files(latent_data_dir, "latent_interpolation_stats.csv", 
                      "merged_interpolation_stats.csv", console):
        console.print(f"[green]✓[/green] Merged interpolation stats CSV (IoU, CC, Frame Diff)")
    
    # Combine PNG images (will auto-split if > 9 images)
    latent_data_path = Path(latent_data_dir)
    
    # Count images for PCA
    pca_count = sum(1 for d in latent_data_path.iterdir() 
                    if d.is_dir() and (d / "latent_pca.png").exists())
    if combine_png_images(latent_data_dir, "latent_pca.png", 
                         "combined_pca.png", console, max_cols=4):
        if pca_count > 9:
            num_files = (pca_count + 8) // 9
            console.print(f"[green]✓[/green] Combined PCA images ({pca_count} models) → {num_files} file(s)")
        else:
            console.print(f"[green]✓[/green] Combined PCA images ({pca_count} models)")
    
    # Count images for interpolation
    interp_count = sum(1 for d in latent_data_path.iterdir() 
                      if d.is_dir() and (d / "latent_interpolation.png").exists())
    if combine_png_images(latent_data_dir, "latent_interpolation.png", 
                         "combined_interpolation.png", console, max_cols=3):
        if interp_count > 9:
            num_files = (interp_count + 8) // 9
            console.print(f"[green]✓[/green] Combined interpolation images ({interp_count} models) → {num_files} file(s)")
        else:
            console.print(f"[green]✓[/green] Combined interpolation images ({interp_count} models)")
    
    # Count images for decoder noise
    noise_count = sum(1 for d in latent_data_path.iterdir() 
                      if d.is_dir() and (d / "decoder_noise.png").exists())
    if combine_png_images(latent_data_dir, "decoder_noise.png", 
                         "combined_decoder_noise.png", console, max_cols=4):
        if noise_count > 9:
            num_files = (noise_count + 8) // 9
            console.print(f"[green]✓[/green] Combined decoder noise images ({noise_count} models) → {num_files} file(s)")
        else:
            console.print(f"[green]✓[/green] Combined decoder noise images ({noise_count} models)")
    
    console.print(f"[green]✓[/green] Summary reports created in [dim]{latent_data_dir}[/dim]")


def process_single_model(vae_ckpt_path, data_root, out_dir, max_samples, device, console, use_amp=True):
    """Process a single VAE model and generate diagnostics.
    
    Args:
        use_amp: Whether to use Automatic Mixed Precision (AMP) for inference.
                 Default: True (enabled for faster inference on CUDA devices)
    """
    vae_ckpt_path = Path(vae_ckpt_path)
    
    with console.status(f"[cyan]Loading model: {vae_ckpt_path.name}[/cyan]"):
        # -------------------------
        # Load VAE
        # -------------------------
        ckpt = torch.load(str(vae_ckpt_path), map_location="cpu", weights_only=False)
        
        # Check checkpoint type before attempting to load
        state_dict = ckpt.get("model", ckpt)
        if "model" in ckpt:
            ckpt_type = detect_checkpoint_type(ckpt["model"])
            if ckpt_type == "diffusion":
                console.print(f"[red]✗[/red] Skipping [yellow]{vae_ckpt_path.name}[/yellow] - Diffusion model (not VAE)")
                return False
            elif ckpt_type == "unknown":
                console.print(f"[yellow]⚠[/yellow] Could not determine checkpoint type for {vae_ckpt_path.name}")
        
        vae_args = ckpt.get("args", {})
        if not vae_args:
            console.print(f"[red]✗[/red] Skipping [yellow]{vae_ckpt_path.name}[/yellow] - Missing 'args' key")
            return False
        
        latent_dim = vae_args.get("latent_dim")
        base = vae_args.get("base")
        
        if latent_dim is None or base is None:
            console.print(f"[red]✗[/red] Skipping [yellow]{vae_ckpt_path.name}[/yellow] - Missing required args")
            return False

        # Detect VAE variant and select appropriate model class
        vae_variant = detect_vae_variant(state_dict)
        default_skip = 3
        if vae_variant == "shallow":
            default_skip = 2
        elif vae_variant == "mid":
            default_skip = 3
        
        skip_levels = resolve_skip_levels(vae_args, default=default_skip)
        
        # Clamp skip_levels based on variant
        if vae_variant == "shallow":
            skip_levels = int(max(0, min(2, int(skip_levels))))
        elif vae_variant == "mid":
            skip_levels = int(max(0, min(3, int(skip_levels))))
        else:  # deep
            skip_levels = int(max(0, min(3, int(skip_levels))))
        
        # Build appropriate model
        if vae_variant == "shallow" and ShallowUNet3DVAE is not None:
            console.print(f"[dim]Detected [cyan]{vae_variant}[/cyan] variant, using ShallowUNet3DVAE[/dim]")
            vae = ShallowUNet3DVAE(
                in_ch=3,
                out_ch=3,
                base=base,
                latent_dim=latent_dim,
                skip_levels=skip_levels,
            )
        elif vae_variant == "mid" and MidUNet3DVAE is not None:
            console.print(f"[dim]Detected [cyan]{vae_variant}[/cyan] variant, using MidUNet3DVAE[/dim]")
            vae = MidUNet3DVAE(
                in_ch=3,
                out_ch=3,
                base=base,
                latent_dim=latent_dim,
                skip_levels=skip_levels,
            )
        else:
            # Use standard UNet3DVAE for deep variant or fallback
            if UNet3DVAE is None:
                console.print(f"[red]✗[/red] UNet3DVAE class not available")
                return False
            if vae_variant != "deep":
                console.print(f"[yellow]⚠[/yellow] Variant [cyan]{vae_variant}[/cyan] detected but class not available, using UNet3DVAE as fallback[/yellow]")
            else:
                console.print(f"[dim]Detected [cyan]{vae_variant}[/cyan] variant, using UNet3DVAE[/dim]")
            vae = UNet3DVAE(
                in_ch=3,
                out_ch=3,
                base=base,
                latent_dim=latent_dim,
                skip_levels=skip_levels,
            )
        
        try:
            vae.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            console.print(f"[red]✗[/red] Failed to load [yellow]{vae_ckpt_path.name}[/yellow]: {str(e)[:100]}")
            return False
        
        vae.to(device)
        vae.eval()

    # -------------------------
    # Load Dataset
    # -------------------------
    train_dir = Path(data_root) / "train"
    if not train_dir.exists():
        console.print(f"[red]✗[/red] Data directory not found: {train_dir}")
        return False
    
    # Load all .npz files from train directory, sorted by filename
    files = sorted(train_dir.glob("*.npz"))
    if not files:
        console.print(f"[red]✗[/red] No .npz files found in {train_dir}")
        return False
    
    npz_files = [str(f) for f in files]
    ds = VoxelDataset(npz_files)
    # Create DataLoader with shuffle=True for random sampling
    # batch_size=1 means each iteration yields one sample
    # shuffle=True randomly shuffles the dataset each epoch, so we get different random samples
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    # -------------------------
    # Collect latents
    # -------------------------
    # Collect up to max_samples latent vectors from randomly shuffled dataset
    # The actual number collected may be less if dataset has fewer samples than max_samples
    z, mu, logvar = collect_latents(vae, loader, device, max_samples=max_samples, use_amp=use_amp)
    N, C, H, W, D = z.shape

    # Validate and create output directory
    out_dir_path = Path(out_dir)
    if out_dir_path.exists() and out_dir_path.is_file():
        raise ValueError(
            f"Output directory path points to an existing file: {out_dir}\n"
            f"Please specify a directory path, not a file path."
        )
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Failed to create output directory: {out_dir}\n"
            f"Error: {e}\n"
            f"Please ensure the path is valid and you have write permissions."
        )
    
    console.print(f"[green]✓[/green] Collected {N} latent vectors")

    # Flatten for statistics
    z_flat = flatten_latent(z.detach().numpy())

    # -------------------------
    # 1. Pairwise distance
    # -------------------------
    with console.status("[cyan]Computing pairwise distances...[/cyan]"):
        Dmat = pairwise_distances(z_flat)
        nonzero = Dmat[Dmat > 1e-8]

        stats = {
            "min": float(nonzero.min()),
            "max": float(nonzero.max()),
            "mean": float(nonzero.mean()),
            "median": float(np.median(nonzero)),
            "std": float(nonzero.std()),
        }

        np.savetxt(os.path.join(out_dir, "latent_pairwise_stats.csv"),
                   np.array([[stats[k] for k in stats]]),
                   fmt="%.6f",
                   header=",".join(stats.keys()))

    # -------------------------
    # 2. Per-channel statistics with collapse detection
    # -------------------------
    with console.status("[cyan]Computing per-channel statistics...[/cyan]"):
        z_np = z.detach().numpy()
        # Compute std per channel: std(dim=[0,2,3,4]) means std across batch, H, W, D
        std_per_chan = z_np.std(axis=(0, 2, 3, 4))
        
        # Collapse detection:
        # 1. If all channels have very low variance (< 0.1) → collapse
        # 2. If all channels have similar variance (low std of stds) → encoder not encoding meaningful structure
        mean_std = float(std_per_chan.mean())
        std_of_stds = float(std_per_chan.std())
        min_std = float(std_per_chan.min())
        max_std = float(std_per_chan.max())
        
        # Thresholds for collapse detection
        low_variance_threshold = 0.1  # If mean std < 0.1, likely collapse
        uniform_variance_threshold = 0.05  # If std_of_stds < 0.05, channels too similar
        
        is_collapsed_low_variance = mean_std < low_variance_threshold
        is_collapsed_uniform = std_of_stds < uniform_variance_threshold
        
        # Save detailed statistics
        stats_dict = {
            'channel_idx': np.arange(len(std_per_chan)),
            'std': std_per_chan
        }
        
        # Save CSV with channel indices
        if pd is not None:
            df = pd.DataFrame(stats_dict)
            df.to_csv(os.path.join(out_dir, "latent_channel_std.csv"), index=False)
        else:
            # Fallback: save as before
            np.savetxt(os.path.join(out_dir, "latent_channel_std.csv"),
                       std_per_chan,
                       fmt="%.6f",
                       header="channel_std")
        
        # Save collapse detection results
        collapse_info = {
            'mean_std': mean_std,
            'std_of_stds': std_of_stds,
            'min_std': min_std,
            'max_std': max_std,
            'is_collapsed_low_variance': is_collapsed_low_variance,
            'is_collapsed_uniform': is_collapsed_uniform,
            'num_channels': len(std_per_chan)
        }
        
        with open(os.path.join(out_dir, "channel_variance_stats.txt"), "w") as f:
            f.write(f"Channel Variance Statistics:\n")
            f.write(f"  Mean std across channels: {mean_std:.6f}\n")
            f.write(f"  Std of stds (variance diversity): {std_of_stds:.6f}\n")
            f.write(f"  Min std: {min_std:.6f}\n")
            f.write(f"  Max std: {max_std:.6f}\n")
            f.write(f"  Number of channels: {len(std_per_chan)}\n")
            f.write(f"\nCollapse Detection:\n")
            f.write(f"  Low variance collapse (mean_std < {low_variance_threshold}): {is_collapsed_low_variance}\n")
            f.write(f"  Uniform variance collapse (std_of_stds < {uniform_variance_threshold}): {is_collapsed_uniform}\n")
            
            if is_collapsed_low_variance:
                f.write(f"\n⚠️  WARNING: All channels have very low variance!\n")
                f.write(f"   This suggests latent collapse - encoder is not encoding meaningful structure.\n")
            if is_collapsed_uniform:
                f.write(f"\n⚠️  WARNING: All channels have similar variance!\n")
                f.write(f"   This suggests encoder is not encoding diverse meaningful structure.\n")
        
        # Print warnings to console
        if is_collapsed_low_variance:
            console.print(f"[yellow]⚠[/yellow] Channel variance collapse detected: mean_std={mean_std:.6f} < {low_variance_threshold}")
        if is_collapsed_uniform:
            console.print(f"[yellow]⚠[/yellow] Uniform channel variance detected: std_of_stds={std_of_stds:.6f} < {uniform_variance_threshold}")

    # -------------------------
    # 3. PCA visualization
    # -------------------------
    with console.status("[cyan]Generating PCA scatter plot...[/cyan]"):
        save_pca_plot(z_flat, os.path.join(out_dir, "latent_pca.png"))

    # -------------------------
    # 4. KL diagnostics
    # -------------------------
    with console.status("[cyan]Computing KL divergence...[/cyan]"):
        KL = -0.5 * (1 + logvar - mu**2 - logvar.exp())
        KL = KL.mean().item()

        with open(os.path.join(out_dir, "kl_stats.txt"), "w") as f:
            f.write(f"mean_KL_per_element = {KL:.6f}\n")

    # -------------------------
    # 5. Enhanced interpolation test (tree shape transition)
    # -------------------------
    with console.status("[cyan]Generating interpolation visualization...[/cyan]"):
        # Select two different tree latents (use first and a later one for diversity)
        z1 = z[0].cpu()
        # Try to find a more different tree (use one from middle/end of batch)
        z2_idx = min(N - 1, max(1, N // 2))
        z2 = z[z2_idx].cpu()
        
        interp_result = linear_interpolation(
            vae, z1, z2,
            os.path.join(out_dir, "latent_interpolation.png"),
            device,
            steps=10,
            use_amp=use_amp
        )
        
        # Save interpolation collapse detection results to text file
        with open(os.path.join(out_dir, "interpolation_stats.txt"), "w") as f:
            f.write(f"Linear Interpolation Statistics:\n")
            f.write(f"\n🥇 No.1: Per-step IoU difference (最重要):\n")
            f.write(f"  Mean IoU: {interp_result['mean_iou']:.6f}\n")
            f.write(f"  Mean IoU difference: {interp_result['mean_iou_diff']:.6f}\n")
            f.write(f"  Max IoU difference: {interp_result['max_iou_diff']:.6f}\n")
            f.write(f"  Std IoU difference: {interp_result['std_iou_diff']:.6f}\n")
            f.write(f"\n🥈 No.2: Connected Component Continuity:\n")
            f.write(f"  Mean CC count: {interp_result['mean_cc_count']:.2f}\n")
            f.write(f"  Mean CC change: {interp_result['mean_cc_change']:.2f}\n")
            f.write(f"  Max CC change: {interp_result['max_cc_change']}\n")
            f.write(f"  Std CC count: {interp_result['std_cc_count']:.2f}\n")
            f.write(f"\n🥉 No.3: Mean Frame Diff (%):\n")
            f.write(f"  Mean frame difference ratio: {interp_result['mean_frame_diff_ratio']:.6f} ({interp_result['mean_frame_diff_ratio']*100:.2f}%)\n")
            f.write(f"  Mean frame difference: {interp_result['mean_frame_diff']:.0f} voxels\n")
            f.write(f"  Total voxels: {interp_result['total_voxels']}\n")
            f.write(f"  Is collapsed: {interp_result['is_collapsed']}\n")
            
            if interp_result['is_collapsed']:
                f.write(f"\n⚠️  WARNING: Interpolation collapse detected!\n")
                f.write(f"   All interpolations are very similar (blob-like).\n")
                f.write(f"   This suggests latent space collapse - encoder is not encoding meaningful tree structure.\n")
            else:
                f.write(f"\n✓ Interpolation shows variation - latent space appears healthy.\n")
        
        # Save key metrics to CSV file
        interpolation_csv_data = {
            # No.1: IoU metrics (最重要)
            'mean_iou': [interp_result['mean_iou']],
            'mean_iou_diff': [interp_result['mean_iou_diff']],  # Per-step IoU difference (most important)
            'max_iou_diff': [interp_result['max_iou_diff']],
            'std_iou_diff': [interp_result['std_iou_diff']],
            
            # No.2: Connected Component metrics
            'mean_cc_count': [interp_result['mean_cc_count']],
            'mean_cc_change': [interp_result['mean_cc_change']],  # Connected Component continuity
            'max_cc_change': [interp_result['max_cc_change']],
            'std_cc_count': [interp_result['std_cc_count']],
            
            # No.3: Frame Diff metrics (for quick screening)
            'mean_frame_diff_ratio': [interp_result['mean_frame_diff_ratio']],  # Mean Frame Diff (%)
            'mean_frame_diff': [interp_result['mean_frame_diff']],
            'total_voxels': [interp_result['total_voxels']],
            
            # Legacy collapse indicator
            'is_collapsed': [interp_result['is_collapsed']],
        }
        
        if pd is not None:
            df_interp = pd.DataFrame(interpolation_csv_data)
            df_interp.to_csv(os.path.join(out_dir, "latent_interpolation_stats.csv"), index=False)
        else:
            # Fallback: save as text CSV
            csv_path = os.path.join(out_dir, "latent_interpolation_stats.csv")
            with open(csv_path, 'w') as f:
                # Write header
                f.write(",".join(interpolation_csv_data.keys()) + "\n")
                # Write values
                f.write(",".join([str(v[0]) for v in interpolation_csv_data.values()]) + "\n")
        
        # Print warnings to console
        if interp_result['is_collapsed']:
            console.print(f"[yellow]⚠[/yellow] Interpolation collapse detected: frame_diff_ratio={interp_result['mean_frame_diff_ratio']*100:.2f}% < 1%")
        
        # Print key metrics
        console.print(f"[dim]  IoU diff: {interp_result['mean_iou_diff']:.4f}, CC change: {interp_result['mean_cc_change']:.2f}, Frame diff: {interp_result['mean_frame_diff_ratio']*100:.2f}%[/dim]")

    # -------------------------
    # 6. Decoder(random noise)
    # -------------------------
    with console.status("[cyan]Testing decoder with random noise...[/cyan]"):
        z_rand = torch.randn_like(z1).unsqueeze(0).to(device)
        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = vae.decoder(z_rand, skips=None)[0].cpu().detach().numpy()
            else:
                logits = vae.decoder(z_rand, skips=None)[0].cpu().detach().numpy()
        proj = logits.argmax(0).max(axis=0)

        plt.imshow(proj)
        plt.title("Decoder(random noise) projection")
        plt.savefig(os.path.join(out_dir, "decoder_noise.png"))
        plt.close()

    return True


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()
    
    # Display AMP status
    use_amp = not args.no_amp
    amp_status = "啟用" if use_amp else "停用"
    amp_warning = ""
    if not use_amp:
        amp_warning = "\n[dim]AMP 已停用: 使用 float32 以獲得最高精度[/dim]"
    elif device.type == "cuda":
        amp_warning = "\n[dim]AMP 已啟用: 使用 float16 加速推理（CUDA）[/dim]"
    else:
        amp_warning = "\n[dim]AMP 僅在 CUDA 設備上有效[/dim]"
    
    console.print(Panel.fit(
        f"[bold cyan]VAE Latent Diagnostics Tool[/bold cyan]\n"
        f"Device: [yellow]{device}[/yellow]\n"
        f"AMP: [yellow]{amp_status}[/yellow]{amp_warning}",
        border_style="cyan"
    ))

    # Find all model files
    try:
        model_files = find_model_files(args.vae_ckpt)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return
    
    console.print(f"[bold]Found {len(model_files)} model file(s)[/bold]")
    
    if len(model_files) == 0:
        console.print("[red]No model files found![/red]")
        return
    
    # Check if input is a directory (for creating latent_summary folder)
    input_path = Path(args.vae_ckpt)
    is_directory_input = input_path.is_dir()
    base_dir = input_path if is_directory_input else input_path.parent
    
    # Process each model
    successful = 0
    failed = 0
    successful_models = []  # Track successful models for copying to latent_summary
    
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
            "[cyan]Processing models...",
            total=len(model_files)
        )
        
        for model_path in model_files:
            model_path = Path(model_path)
            
            # Generate output directory
            if args.out_dir:
                # If out_dir is specified, use it (for backward compatibility)
                out_dir_path = Path(args.out_dir)
                # Check if out_dir is a file path (e.g., ends with .pt)
                if out_dir_path.exists() and out_dir_path.is_file():
                    # If it's a file, create a directory with the same name (without extension)
                    out_dir = str(out_dir_path.parent / out_dir_path.stem)
                    console.print(f"[yellow]⚠[/yellow] --out_dir points to a file, using directory: [cyan]{out_dir}[/cyan]")
                elif out_dir_path.suffix in ['.pt', '.pth', '.ckpt']:
                    # If it has a checkpoint extension but doesn't exist, treat as directory name
                    out_dir = str(out_dir_path.parent / out_dir_path.stem)
                    console.print(f"[yellow]⚠[/yellow] --out_dir has file extension, using directory: [cyan]{out_dir}[/cyan]")
                else:
                    out_dir = args.out_dir
            else:
                # Auto-generate output directory next to model
                out_dir = get_output_dir(model_path)
            
            progress.update(task, description=f"[cyan]Processing: {model_path.name}[/cyan]")
            
            try:
                success = process_single_model(
                    model_path,
                    args.data_root,
                    out_dir,
                    args.max_samples,
                    device,
                    console,
                    use_amp=not args.no_amp
                )
                
                if success:
                    successful += 1
                    successful_models.append((model_path, out_dir))
                    console.print(f"[green]✓[/green] Completed: [cyan]{model_path.name}[/cyan] → [dim]{out_dir}[/dim]")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                console.print(f"[red]✗[/red] Error processing [yellow]{model_path.name}[/yellow]: {str(e)[:200]}")
            
            progress.advance(task)
    
    # If input was a directory, copy all results to latent_summary folder
    if is_directory_input and successful_models:
        console.print(f"\n[cyan]Organizing results into latent_summary folder...[/cyan]")
        with console.status("[cyan]Copying results to latent_summary...[/cyan]"):
            for model_path, out_dir in successful_models:
                model_name = model_path.stem
                if copy_results_to_latent_data(out_dir, base_dir, model_name, console):
                    console.print(f"[green]✓[/green] Copied results for [cyan]{model_name}[/cyan] to [dim]{base_dir / 'latent_summary' / model_name}[/dim]")
        
        # Create merged summary reports
        latent_summary_dir = base_dir / "latent_summary"
        if latent_summary_dir.exists():
            create_summary_reports(str(latent_summary_dir), console)
    
    # Summary
    console.print("\n" + "="*70)
    summary_table = Table(title="Processing Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]Successful[/green]", str(successful))
    summary_table.add_row("[red]Failed[/red]", str(failed))
    summary_table.add_row("[bold]Total[/bold]", str(len(model_files)))
    console.print(summary_table)
    if is_directory_input and successful > 0:
        console.print(f"\n[bold cyan]All results organized in:[/bold cyan] [dim]{base_dir / 'latent_summary'}[/dim]")
    console.print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VAE latent diagnostic tool. Supports single file or directory with recursive search."
    )
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        required=True,
        help="Path to VAE checkpoint file (.pt) or directory containing .pt files (recursive search)"
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing train/ subdirectory")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (if not specified, auto-generated as 'latent' next to each model)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=200, 
        help="Maximum number of samples to process. "
             "Samples are randomly selected from the training dataset (DataLoader with shuffle=True). "
             "Each run will select different random samples. "
             "If dataset has fewer samples than max_samples, all available samples will be used."
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP) for inference. "
             "By default, AMP is enabled for faster inference on CUDA devices. "
             "Use this flag to disable AMP and use float32 for maximum precision. "
             "Only effective on CUDA devices."
    )
    args = parser.parse_args()
    main(args)
