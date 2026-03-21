#!/usr/bin/env python3
"""
評估 16x16x16 Voxel Diffusion 模型

分層記錄評估數據：
- 層次 A：每個樣本的最終狀態指標（用於畫直方圖）
- 層次 B：整批樣本的統計摘要（用於報告）
- 層次 C：前幾個樣本的完整動力學軌跡（用於畫 divergence 圖）

輸出文件：
- simple_result.csv: 每個樣本的指標（層次 A）
- simple_summary.csv: 統計摘要（層次 B）
- dynamics_result.csv: 前幾個樣本的最終結果（層次 C）
- dynamics_summary.csv: 前幾個樣本的統計摘要（層次 C）
- dynamics_trace.csv: 前幾個樣本的完整軌跡（層次 C）
- dynamics_divergence_plot.png: 可視化圖表
- dynamics_timing_distribution_plot.png: t_emerge / t_lockin 分布圖

使用方式:
  python eval_16_voxel_diffusion.py --checkpoint <path/to/checkpoint.pt> --n_samples 100
"""

import argparse
import os
import math
import time
import random
import shlex
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import json

try:
    from scipy.ndimage import label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not installed. Component count will be disabled.")

import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add parent directory to path to import from training script
import sys
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import model and functions from training script
try:
    from unet_diffusion_16_voxel import (
        UNet3DDiffusion,
        BetaSchedule,
        sample_voxels,
        centered_to_onehot,
        compute_trunk_breakage,
        compute_occupancy_rates,
        compute_component_counts_26neighbor,
        compute_largest_log_component_ratio,
    )
except ImportError as e:
    print(f"[ERROR] Failed to import from training script: {e}")
    print("Make sure unet_diffusion_16_voxel.py is in the same directory.")
    sys.exit(1)


def compute_sample_metrics(labels: np.ndarray) -> Dict:
    """
    計算單個樣本的所有指標。
    
    Args:
        labels: [16, 16, 16] numpy array with class labels (0=air, 1=log, 2=leaf)
    
    Returns:
        dict with metrics:
            - ID: sample ID
            - Is_Main_Trunk_Broken: bool, True if main trunk is broken (no connected path from ground to top)
            - Is_Broken: bool, True if there are any disconnected wood components (not connected to ground)
            - Mass: total non-air voxels
            - Height: highest non-air voxel Y coordinate (from ground Y=0)
            - Log_Size: total log voxels
            - Leaf_Size: total leaf voxels
            - Base_Connected_Size: size of base-connected log component
            - Total_Log_Size: total log voxels (same as Log_Size)
            - Largest_Log_Ratio: ratio of largest log component
            - Occupancy_Non_Air: occupancy rate of non-air
            - Occupancy_Log: occupancy rate of log
            - Occupancy_Leaf: occupancy rate of leaf
            - Components_Non_Air: number of non-air components
            - Components_Log: number of log components
            - Components_Leaf: number of leaf components
    """
    # Trunk breakage analysis
    trunk_info = compute_trunk_breakage(labels, debug=False)
    
    # Occupancy rates
    occ_rates = compute_occupancy_rates(labels)
    
    # Component counts
    comp_counts = compute_component_counts_26neighbor(labels)
    
    # Largest log component ratio
    largest_log_ratio = compute_largest_log_component_ratio(labels)
    
    # Mass (total non-air voxels)
    mass = int((labels != 0).sum())
    
    # Height (highest non-air voxel, Y=0 is ground, so we find max Y)
    # labels shape is [Z, Y, X], so Y is at index 1
    non_air_coords = np.argwhere(labels != 0)
    if len(non_air_coords) > 0:
        max_y = non_air_coords[:, 1].max()  # Largest Y = highest point
        height = max_y + 1  # Convert to height from ground (Y=0 is ground, so height = max_y + 1)
    else:
        height = 0
    
    # Log and leaf sizes
    log_size = int((labels == 1).sum())
    leaf_size = int((labels == 2).sum())
    
    return {
        'Is_Main_Trunk_Broken': trunk_info['is_main_trunk_broken'],
        'Is_Broken': trunk_info['is_broken'],
        'Mass': mass,
        'Height': height,
        'Log_Size': log_size,
        'Leaf_Size': leaf_size,
        'Base_Connected_Size': trunk_info['base_connected_size'],
        'Total_Log_Size': trunk_info['total_wood_size'],
        'Largest_Log_Ratio': largest_log_ratio if largest_log_ratio >= 0 else -1.0,
        'Occupancy_Non_Air': occ_rates['non_air'],
        'Occupancy_Log': occ_rates['log'],
        'Occupancy_Leaf': occ_rates['leaf'],
        'Components_Non_Air': comp_counts['non_air'],
        'Components_Log': comp_counts['log'],
        'Components_Leaf': comp_counts['leaf'],
    }


def compute_dynamics_metrics(labels: np.ndarray, t_int: int) -> Dict:
    """
    計算動力學軌跡中的指標（每一步的狀態）。
    
    Args:
        labels: [16, 16, 16] numpy array with class labels
        t_int: current timestep
    
    Returns:
        dict with dynamics metrics
    """
    metrics = compute_sample_metrics(labels)
    metrics['t'] = t_int
    
    # Calculate Base_Connected_Ratio = Base_Connected_Size / Total_Log_Size
    total_log_size = metrics.get('Total_Log_Size', metrics.get('Log_Size', 0))
    if total_log_size > 0:
        base_connected_ratio = metrics['Base_Connected_Size'] / total_log_size
    else:
        base_connected_ratio = 0.0
    metrics['Base_Connected_Ratio'] = base_connected_ratio
    
    return metrics


def decode_probs_to_labels(probs: torch.Tensor, log_mask_threshold: Optional[float] = None) -> np.ndarray:
    """
    Decode class probabilities into discrete labels.
    
    Args:
        probs: [3, 16, 16, 16] class probabilities (air=0, log=1, leaf=2)
        log_mask_threshold: if provided, use threshold-based log mask decoding.
            - log if P(log) >= threshold
            - otherwise choose between air/leaf by argmax on [P(air), P(leaf)]
            if None, use full argmax over 3 classes.
    
    Returns:
        labels: [16, 16, 16] uint8 in {0,1,2}
    """
    if log_mask_threshold is None:
        return probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    probs_np = probs.detach().cpu().numpy()
    log_mask = probs_np[1] >= float(log_mask_threshold)
    
    # For non-log voxels, keep only air/leaf decision.
    non_log_mask = ~log_mask
    if np.any(non_log_mask):
        air_or_leaf = np.argmax(probs_np[[0, 2]], axis=0)  # 0->air, 1->leaf
        labels[non_log_mask] = np.where(air_or_leaf[non_log_mask] == 0, 0, 2)
    
    labels[log_mask] = 1
    return labels


def compute_t_emerge_and_t_lockin(sample_trace_data: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
    """
    計算 t_emerge 和 t_lockin。
    
    Args:
        sample_trace_data: List of trace data for a single sample, sorted by t (descending)
    
    Returns:
        Tuple of (t_emerge, t_lockin), both can be None if not found
    """
    # Sort by t descending (from high to low, i.e., 1000 -> 0)
    sorted_data = sorted(sample_trace_data, key=lambda x: x['t'], reverse=True)
    
    t_emerge = None
    t_lockin = None
    
    # Find t_emerge: base_connected_size > 10, consecutive two records, and log_size < 100
    for i in range(len(sorted_data) - 1):
        r1 = sorted_data[i]
        r2 = sorted_data[i + 1]
        
        log_size1 = r1.get('Log_Size', 0)
        log_size2 = r2.get('Log_Size', 0)
        base_connected1 = r1.get('Base_Connected_Size', 0)
        base_connected2 = r2.get('Base_Connected_Size', 0)
        
        # Check conditions: log_size < 100 and base_connected_size > 10 for both records
        if log_size1 < 100 and log_size2 < 100:
            if base_connected1 > 10 and base_connected2 > 10:
                # Use the earlier t (smaller t value, which is r2 since sorted descending)
                t_emerge = r2['t']
                break
    
    # Find t_lockin:
    # lock-in time = from this timestep onward, Is_Main_Trunk_Broken no longer flips
    if sorted_data:
        states = [bool(r.get('Is_Main_Trunk_Broken', False)) for r in sorted_data]
        last_flip_idx = None
        for i in range(len(states) - 1):
            if states[i] != states[i + 1]:
                last_flip_idx = i
        if last_flip_idx is None:
            # Never flips across tracked trajectory, lock-in starts at the earliest tracked timestep
            t_lockin = sorted_data[0]['t']
        else:
            # Flip happens between i and i+1; lock-in starts at i+1 (smaller t)
            t_lockin = sorted_data[last_flip_idx + 1]['t']
    
    return t_emerge, t_lockin


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    載入訓練好的模型。
    
    Args:
        checkpoint_path: path to .pt checkpoint file
        device: torch device
    
    Returns:
        (model, checkpoint_dict)
    """
    console = Console()
    console.print(f"[cyan]Loading checkpoint: {checkpoint_path}[/cyan]")
    
    # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
    # (checkpoint contains numpy objects and other non-weight data)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint args or use defaults
    args = checkpoint.get('args', {})
    base_channels = args.get('base_channels', 64)
    time_dim = args.get('time_dim', 128)
    
    # Create model
    model = UNet3DDiffusion(in_ch=3, base=base_channels, time_dim=time_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    console.print(f"[green]✓[/green] Model loaded: base_channels={base_channels}, time_dim={time_dim}")
    console.print(f"[green]✓[/green] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, checkpoint


def batch_evaluation(
    model: nn.Module,
    betas: BetaSchedule,
    device: torch.device,
    n_samples: int,
    n_steps: Optional[int] = None,
    batch_size: int = 10,
    use_amp: bool = False,
    output_dir: Optional[str] = None,
    exp_name: Optional[str] = None,
    save_projections: bool = True,
    save_npz: bool = False,
    log_mask_threshold: Optional[float] = None,
    console: Optional[Console] = None,
) -> Tuple[List[Dict], float]:
    """
    批量評估：生成 n_samples 個樣本，只記錄最終狀態。
    
    Args:
        model: UNet3DDiffusion model
        betas: BetaSchedule instance
        device: torch device
        n_samples: number of samples to generate
        n_steps: number of sampling steps (default: T)
        batch_size: number of samples to generate in parallel (default: 10)
        use_amp: whether to use mixed precision
        output_dir: output directory for saving projections (optional)
        exp_name: experiment name for projection titles (optional)
        save_projections: whether to save 3-view projections (default: True)
        save_npz: whether to save npz files for each sample (default: False)
        log_mask_threshold: decode with log mask threshold if set; otherwise use argmax
        console: rich console for progress
    
    Returns:
        Tuple of (List of metrics dicts, elapsed_time_seconds)
    """
    if console is None:
        console = Console()
    
    console.print(f"[bold]Running Batch Evaluation (N={n_samples}, batch_size={batch_size})...[/bold]")
    
    batch_metrics = []
    sample_counter = 0
    t0 = time.time()
    
    # Create projections directory if output_dir is provided and save_projections is enabled
    projections_dir = None
    if output_dir and save_projections:
        projections_dir = os.path.join(output_dir, "simple_projections")
        os.makedirs(projections_dir, exist_ok=True)
    
    # Create npz directory if output_dir is provided and save_npz is enabled
    npz_dir = None
    if output_dir and save_npz:
        npz_dir = os.path.join(output_dir, "simple_npz")
        os.makedirs(npz_dir, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as progress:
        task = progress.add_task("[cyan]Generating samples", total=n_samples)
        
        # Process in batches
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            
            # Generate batch of samples
            with torch.no_grad():
                x_0_batch = sample_voxels(
                    model,
                    betas,
                    shape=(current_batch_size, 3, 16, 16, 16),
                    device=device,
                    n_steps=n_steps,
                    use_amp=use_amp,
                    track_every=None,  # No tracking for batch evaluation
                    track_callback=None,
                )  # [current_batch_size, 3, 16, 16, 16] in [-1,1]
            
            # Process each sample in the batch
            for i in range(current_batch_size):
                # Convert to onehot-like and then to labels
                x_0_onehot = centered_to_onehot(x_0_batch[i])  # [3, 16, 16, 16] in [0,1]
                probs = F.softmax(x_0_onehot, dim=0)
                labels = decode_probs_to_labels(probs, log_mask_threshold=log_mask_threshold)  # [16, 16, 16]
                
                # Compute metrics
                metrics = compute_sample_metrics(labels)
                metrics['ID'] = f"{sample_counter+1:03d}"
                batch_metrics.append(metrics)
                
                # Save 3-view projection if output_dir is provided
                if projections_dir:
                    try:
                        projection_path = os.path.join(projections_dir, f"simple_sample_{sample_counter+1:03d}.png")
                        title_suffix = f" Sample {sample_counter+1}"
                        save_labels_and_projections(
                            labels,
                            projection_path,
                            title_suffix=title_suffix,
                            exp_name=exp_name if exp_name else None
                        )
                    except Exception as e:
                        console.print(f"[yellow]⚠[/yellow] Failed to save projection for sample {sample_counter+1}: {e}")
                
                # Save npz file if enabled
                if npz_dir:
                    try:
                        npz_path = os.path.join(npz_dir, f"simple_sample_{sample_counter+1:03d}.npz")
                        # Save labels and metrics (flatten metrics dict into separate arrays)
                        save_dict = {"labels": labels, "sample_id": f"{sample_counter+1:03d}"}
                        # Add each metric as a separate array
                        for key, value in metrics.items():
                            if isinstance(value, (int, float, bool)):
                                save_dict[f"metric_{key}"] = np.array([value])
                            elif isinstance(value, np.ndarray):
                                save_dict[f"metric_{key}"] = value
                            else:
                                save_dict[f"metric_{key}"] = np.array([str(value)])
                        np.savez(npz_path, **save_dict)
                    except Exception as e:
                        console.print(f"[yellow]⚠[/yellow] Failed to save npz for sample {sample_counter+1}: {e}")
                
                sample_counter += 1
                progress.update(task, advance=1)
    
    elapsed_time = time.time() - t0
    console.print(f"[green]✓[/green] Batch evaluation completed: {n_samples} samples in {fmt_secs(elapsed_time)}")
    if projections_dir:
        console.print(f"[green]✓[/green] Saved {n_samples} projections to: {projections_dir}")
    if npz_dir:
        console.print(f"[green]✓[/green] Saved {n_samples} npz files to: {npz_dir}")
    return batch_metrics, elapsed_time


def dynamics_evaluation(
    model: nn.Module,
    betas: BetaSchedule,
    device: torch.device,
    n_samples: int,
    n_steps: Optional[int] = None,
    track_every: int = 50,
    batch_size: int = 5,
    use_amp: bool = False,
    output_dir: Optional[str] = None,
    exp_name: Optional[str] = None,
    save_projections: bool = True,
    save_track_projections: bool = False,
    save_npz: bool = False,
    log_mask_threshold: Optional[float] = None,
    console: Optional[Console] = None,
) -> Tuple[List[Dict], float]:
    """
    動力學評估：生成 n_samples 個樣本，記錄每一步的狀態。
    
    Args:
        model: UNet3DDiffusion model
        betas: BetaSchedule instance
        device: torch device
        n_samples: number of samples to generate
        n_steps: number of sampling steps (default: T)
        track_every: track metrics every N steps
        batch_size: number of samples to generate in parallel (default: 5)
        use_amp: whether to use mixed precision
        output_dir: output directory for saving projections (optional)
        exp_name: experiment name for projection titles (optional)
        save_projections: whether to save 3-view projections for final states (default: True)
        save_track_projections: whether to save 3-view projections at each tracking step (default: False)
        save_npz: whether to save npz files for final states and tracking steps (default: False)
        log_mask_threshold: decode with log mask threshold if set; otherwise use argmax
        console: rich console for progress
    
    Returns:
        Tuple of (List of dynamics metrics dicts, elapsed_time_seconds)
    """
    if console is None:
        console = Console()
    
    console.print(f"[bold]Running Dynamics Analysis (N={n_samples}, batch_size={batch_size}, track_every={track_every})...[/bold]")
    
    trace_data = []
    global_sample_counter = 0  # Track global sample index across batches
    t0 = time.time()
    
    # Create projections directory if output_dir is provided and save_projections is enabled
    projections_dir = None
    if output_dir and save_projections:
        projections_dir = os.path.join(output_dir, "dynamics_projections")
        os.makedirs(projections_dir, exist_ok=True)
    
    # Create track projections directory if save_track_projections is enabled
    track_projections_dir = None
    if output_dir and save_track_projections:
        track_projections_dir = os.path.join(output_dir, "dynamics_track_projections")
        os.makedirs(track_projections_dir, exist_ok=True)
    
    # Create npz directories if save_npz is enabled
    npz_dir = None
    track_npz_dir = None
    if output_dir and save_npz:
        npz_dir = os.path.join(output_dir, "dynamics_npz")
        os.makedirs(npz_dir, exist_ok=True)
        track_npz_dir = os.path.join(output_dir, "dynamics_track_npz")
        os.makedirs(track_npz_dir, exist_ok=True)
    
    # Store final samples for projection saving
    final_samples = {}  # {sample_idx: labels}
    
    # Store final state metrics for CSV output
    final_metrics = []  # List of metrics dicts for final states
    
    # Track number of saved projections
    track_projection_count = 0
    track_npz_count = 0
    
    def make_track_callback(batch_offset: int):
        """Create a track callback function with batch offset."""
        def track_callback(sample_idx, step_idx, t_int, x_current, x0_hat):
            """Callback to track metrics at each step."""
            nonlocal track_projection_count, track_npz_count
            # sample_idx is relative to the batch, add batch_offset to get global index
            global_sample_idx = batch_offset + sample_idx
            
            # Use x0_hat (predicted clean state) for metrics
            x0hat_onehot = centered_to_onehot(x0_hat.unsqueeze(0))[0]  # [3, 16, 16, 16]
            probs = F.softmax(x0hat_onehot, dim=0)
            labels = decode_probs_to_labels(probs, log_mask_threshold=log_mask_threshold)
            
            # Compute metrics
            metrics = compute_dynamics_metrics(labels, t_int)
            metrics['sample_idx'] = global_sample_idx
            metrics['step_idx'] = step_idx
            trace_data.append(metrics)
            
            # Save projection at this tracking step if enabled
            if track_projections_dir:
                try:
                    projection_path = os.path.join(
                        track_projections_dir,
                        f"dynamics_sample_{global_sample_idx+1:03d}_step_{step_idx:04d}_t_{t_int:04d}.png"
                    )
                    title_suffix = f" Sample {global_sample_idx+1}, Step {step_idx}, t={t_int}"
                    save_labels_and_projections(
                        labels,
                        projection_path,
                        title_suffix=title_suffix,
                        exp_name=exp_name if exp_name else None
                    )
                    track_projection_count += 1
                except Exception as e:
                    if console:
                        console.print(f"[yellow]⚠[/yellow] Failed to save track projection for sample {global_sample_idx+1}, step {step_idx}: {e}")
            
            # Save npz file at this tracking step if enabled
            if track_npz_dir:
                try:
                    npz_path = os.path.join(
                        track_npz_dir,
                        f"dynamics_sample_{global_sample_idx+1:03d}_step_{step_idx:04d}_t_{t_int:04d}.npz"
                    )
                    # Save labels and metrics (flatten metrics dict into separate arrays)
                    save_dict = {
                        "labels": labels,
                        "sample_idx": np.array([global_sample_idx]),
                        "step_idx": np.array([step_idx]),
                        "t": np.array([t_int])
                    }
                    # Add each metric as a separate array
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, bool)):
                            save_dict[f"metric_{key}"] = np.array([value])
                        elif isinstance(value, np.ndarray):
                            save_dict[f"metric_{key}"] = value
                        else:
                            save_dict[f"metric_{key}"] = np.array([str(value)])
                    np.savez(npz_path, **save_dict)
                    track_npz_count += 1
                except Exception as e:
                    if console:
                        console.print(f"[yellow]⚠[/yellow] Failed to save track npz for sample {global_sample_idx+1}, step {step_idx}: {e}")
        
        return track_callback
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as progress:
        task = progress.add_task("[cyan]Generating samples with tracking", total=n_samples)
        
        # Process in batches
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            
            # Create callback with batch offset
            track_callback = make_track_callback(batch_start)
            
            with torch.no_grad():
                x_0_batch = sample_voxels(
                    model,
                    betas,
                    shape=(current_batch_size, 3, 16, 16, 16),
                    device=device,
                    n_steps=n_steps,
                    use_amp=use_amp,
                    track_every=track_every,
                    track_callback=track_callback,
                )
            
            # Save final state labels for projection and npz
            if projections_dir or npz_dir:
                for i in range(current_batch_size):
                    sample_idx = batch_start + i
                    x_0_onehot = centered_to_onehot(x_0_batch[i])  # [3, 16, 16, 16] in [0,1]
                    probs = F.softmax(x_0_onehot, dim=0)
                    labels = decode_probs_to_labels(probs, log_mask_threshold=log_mask_threshold)  # [16, 16, 16]
                    final_samples[sample_idx] = labels
            
            progress.update(task, advance=current_batch_size)
    
    # Save projections for final states
    if projections_dir and final_samples:
        for sample_idx, labels in final_samples.items():
            try:
                projection_path = os.path.join(projections_dir, f"dynamics_sample_{sample_idx+1:03d}.png")
                title_suffix = f" Dynamics Sample {sample_idx+1}"
                save_labels_and_projections(
                    labels,
                    projection_path,
                    title_suffix=title_suffix,
                    exp_name=exp_name if exp_name else None
                )
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Failed to save projection for dynamics sample {sample_idx+1}: {e}")
    
    # Compute and save final state metrics (sorted by sample_idx)
    # Group trace_data by sample_idx for t_emerge and t_lockin calculation
    trace_by_sample = {}
    for row in trace_data:
        sid = row['sample_idx']
        if sid not in trace_by_sample:
            trace_by_sample[sid] = []
        trace_by_sample[sid].append(row)
    
    for sample_idx in sorted(final_samples.keys()):
        labels = final_samples[sample_idx]
        try:
            # Compute metrics for final state
            metrics = compute_sample_metrics(labels)
            metrics['ID'] = f"{sample_idx+1:03d}"
            
            # Add Base_Connected_Ratio to final state metrics
            total_log_size = metrics.get('Total_Log_Size', metrics.get('Log_Size', 0))
            if total_log_size > 0:
                base_connected_ratio = metrics.get('Base_Connected_Size', 0) / total_log_size
            else:
                base_connected_ratio = 0.0
            metrics['Base_Connected_Ratio'] = base_connected_ratio
            
            # Calculate t_emerge and t_lockin from trace_data
            sample_trace = trace_by_sample.get(sample_idx, [])
            t_emerge, t_lockin = compute_t_emerge_and_t_lockin(sample_trace)
            metrics['t_emerge'] = t_emerge if t_emerge is not None else -1
            metrics['t_lockin'] = t_lockin if t_lockin is not None else -1
            
            final_metrics.append(metrics)
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to compute metrics for dynamics sample {sample_idx+1}: {e}")
    
    # Save npz files for final states
    if npz_dir and final_samples:
        for sample_idx, labels in final_samples.items():
            try:
                # Compute metrics for final state (already computed above, but need for npz)
                metrics = compute_sample_metrics(labels)
                npz_path = os.path.join(npz_dir, f"dynamics_sample_{sample_idx+1:03d}.npz")
                # Save labels and metrics (flatten metrics dict into separate arrays)
                save_dict = {
                    "labels": labels,
                    "sample_idx": np.array([sample_idx]),
                    "sample_id": f"{sample_idx+1:03d}"
                }
                # Add each metric as a separate array
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        save_dict[f"metric_{key}"] = np.array([value])
                    elif isinstance(value, np.ndarray):
                        save_dict[f"metric_{key}"] = value
                    else:
                        save_dict[f"metric_{key}"] = np.array([str(value)])
                np.savez(npz_path, **save_dict)
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Failed to save npz for dynamics sample {sample_idx+1}: {e}")
    
    elapsed_time = time.time() - t0
    console.print(f"[green]✓[/green] Dynamics evaluation completed: {len(trace_data)} tracking points in {fmt_secs(elapsed_time)}")
    if projections_dir:
        console.print(f"[green]✓[/green] Saved {len(final_samples)} final state projections to: {projections_dir}")
    if track_projections_dir:
        console.print(f"[green]✓[/green] Saved {track_projection_count} track step projections to: {track_projections_dir}")
    if npz_dir:
        console.print(f"[green]✓[/green] Saved {len(final_samples)} final state npz files to: {npz_dir}")
    if track_npz_dir:
        console.print(f"[green]✓[/green] Saved {track_npz_count} track step npz files to: {track_npz_dir}")
    return trace_data, elapsed_time, final_metrics


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


def save_batch_results(batch_metrics: List[Dict], output_dir: str, console: Optional[Console] = None):
    """
    保存批量評估結果到 CSV。
    
    Args:
        batch_metrics: List of metrics dicts
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "simple_result.csv")
    
    # Define fieldnames (ID first, then other metrics)
    fieldnames = ['ID'] + [k for k in batch_metrics[0].keys() if k != 'ID']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_metrics)
    
    console.print(f"[green]✓[/green] Saved batch results: {csv_path} ({len(batch_metrics)} samples)")


def save_dynamics_samples(dynamics_metrics: List[Dict], output_dir: str, console: Optional[Console] = None):
    """
    保存動力學追蹤樣本的最終結果到 CSV。
    
    Args:
        dynamics_metrics: List of metrics dicts for final states of dynamics samples
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    if not dynamics_metrics:
        console.print("[yellow]⚠[/yellow] No dynamics sample metrics to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dynamics_result.csv")
    
    # Define fieldnames with specific order (t_emerge and t_lockin after Is_Broken)
    base_fields = ['ID']
    metric_field_order = [
        'Is_Main_Trunk_Broken',
        'Is_Broken',
        't_emerge',  # After Is_Broken
        't_lockin',  # After Is_Broken
        'Mass',
        'Height',
        'Log_Size',
        'Leaf_Size',
        'Base_Connected_Size',
        'Base_Connected_Ratio',
        'Largest_Log_Ratio',
        'Occupancy_Non_Air',
        'Occupancy_Log',
        'Occupancy_Leaf',
        'Components_Non_Air',
        'Components_Log',
        'Components_Leaf',
    ]
    
    # Get all keys from data
    all_keys = set()
    for row in dynamics_metrics:
        all_keys.update(row.keys())
    
    # Build fieldnames: base fields + ordered metric fields + any remaining fields
    fieldnames = base_fields.copy()
    for field in metric_field_order:
        if field in all_keys:
            fieldnames.append(field)
    
    # Add any remaining fields that weren't in the ordered list
    remaining_fields = sorted([k for k in all_keys if k not in fieldnames])
    fieldnames.extend(remaining_fields)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dynamics_metrics)
    
    console.print(f"[green]✓[/green] Saved dynamics samples results: {csv_path} ({len(dynamics_metrics)} samples)")


def compute_summary_statistics(batch_metrics: List[Dict]) -> Dict:
    """
    計算統計摘要。
    
    Args:
        batch_metrics: List of metrics dicts
    
    Returns:
        dict with summary statistics
    """
    n = len(batch_metrics)
    if n == 0:
        return {}
    
    # Breakage rate (main trunk)
    breakage_rate = np.mean([1.0 if m['Is_Main_Trunk_Broken'] else 0.0 for m in batch_metrics]) * 100.0
    
    # Disconnected wood rate (any disconnected wood components)
    broken_rate = np.mean([1.0 if m['Is_Broken'] else 0.0 for m in batch_metrics]) * 100.0
    
    # Average mass
    avg_mass = np.mean([m['Mass'] for m in batch_metrics])
    std_mass = np.std([m['Mass'] for m in batch_metrics])
    
    # Success rate (not broken)
    success_rate = (1.0 - breakage_rate / 100.0) * 100.0
    
    # Average height
    avg_height = np.mean([m['Height'] for m in batch_metrics])
    std_height = np.std([m['Height'] for m in batch_metrics])
    
    # Average log size
    avg_log_size = np.mean([m['Log_Size'] for m in batch_metrics])
    std_log_size = np.std([m['Log_Size'] for m in batch_metrics])
    
    # Average leaf size
    avg_leaf_size = np.mean([m['Leaf_Size'] for m in batch_metrics])
    std_leaf_size = np.std([m['Leaf_Size'] for m in batch_metrics])
    
    # Average base connected size
    avg_base_connected = np.mean([m['Base_Connected_Size'] for m in batch_metrics])
    std_base_connected = np.std([m['Base_Connected_Size'] for m in batch_metrics])
    
    # Average base connected ratio (Base_Connected_Size / Total_Log_Size)
    base_connected_ratios = []
    for m in batch_metrics:
        total_log_size = m.get('Total_Log_Size', m.get('Log_Size', 0))
        if total_log_size > 0:
            ratio = m['Base_Connected_Size'] / total_log_size
            base_connected_ratios.append(ratio)
    avg_base_connected_ratio = np.mean(base_connected_ratios) if base_connected_ratios else 0.0
    std_base_connected_ratio = np.std(base_connected_ratios) if base_connected_ratios else 0.0
    
    # Average largest log ratio
    valid_ratios = [m['Largest_Log_Ratio'] for m in batch_metrics if m['Largest_Log_Ratio'] >= 0]
    avg_largest_log_ratio = np.mean(valid_ratios) if valid_ratios else -1.0
    
    # Average occupancy rates
    avg_occ_non_air = np.mean([m['Occupancy_Non_Air'] for m in batch_metrics])
    avg_occ_log = np.mean([m['Occupancy_Log'] for m in batch_metrics])
    avg_occ_leaf = np.mean([m['Occupancy_Leaf'] for m in batch_metrics])
    
    # Average component counts
    avg_comp_non_air = np.mean([m['Components_Non_Air'] for m in batch_metrics])
    avg_comp_log = np.mean([m['Components_Log'] for m in batch_metrics])
    avg_comp_leaf = np.mean([m['Components_Leaf'] for m in batch_metrics])
    
    return {
        'n_samples': n,
        'breakage_rate': breakage_rate,
        'broken_rate': broken_rate,
        'success_rate': success_rate,
        'avg_mass': avg_mass,
        'std_mass': std_mass,
        'avg_height': avg_height,
        'std_height': std_height,
        'avg_log_size': avg_log_size,
        'std_log_size': std_log_size,
        'avg_leaf_size': avg_leaf_size,
        'std_leaf_size': std_leaf_size,
        'avg_base_connected_size': avg_base_connected,
        'std_base_connected_size': std_base_connected,
        'avg_base_connected_ratio': avg_base_connected_ratio,
        'std_base_connected_ratio': std_base_connected_ratio,
        'avg_largest_log_ratio': avg_largest_log_ratio,
        'avg_occupancy_non_air': avg_occ_non_air,
        'avg_occupancy_log': avg_occ_log,
        'avg_occupancy_leaf': avg_occ_leaf,
        'avg_components_non_air': avg_comp_non_air,
        'avg_components_log': avg_comp_log,
        'avg_components_leaf': avg_comp_leaf,
    }


def compute_dynamics_summary_statistics(dynamics_metrics: List[Dict]) -> Dict:
    """
    計算 dynamics 最終狀態的統計摘要（基於 dynamics_result.csv 同一批資料）。
    
    Args:
        dynamics_metrics: List of metrics dicts for final states of dynamics samples
    
    Returns:
        dict with summary statistics
    """
    if not dynamics_metrics:
        return {}
    
    summary = compute_summary_statistics(dynamics_metrics)
    n = len(dynamics_metrics)
    
    # t_emerge/t_lockin statistics (-1 means not found)
    t_emerge_values = [m.get('t_emerge', -1) for m in dynamics_metrics if m.get('t_emerge', -1) >= 0]
    t_lockin_values = [m.get('t_lockin', -1) for m in dynamics_metrics if m.get('t_lockin', -1) >= 0]
    
    summary.update({
        'n_t_emerge_found': len(t_emerge_values),
        't_emerge_missing_rate': (1.0 - len(t_emerge_values) / n) if n > 0 else 0.0,
        'avg_t_emerge': float(np.mean(t_emerge_values)) if t_emerge_values else -1.0,
        'std_t_emerge': float(np.std(t_emerge_values)) if t_emerge_values else -1.0,
        'n_t_lockin_found': len(t_lockin_values),
        't_lockin_missing_rate': (1.0 - len(t_lockin_values) / n) if n > 0 else 0.0,
        'avg_t_lockin': float(np.mean(t_lockin_values)) if t_lockin_values else -1.0,
        'std_t_lockin': float(np.std(t_lockin_values)) if t_lockin_values else -1.0,
    })
    
    return summary


def save_summary(summary: Dict, output_dir: str, console: Optional[Console] = None):
    """
    保存統計摘要到 CSV 文件。
    
    Args:
        summary: summary statistics dict
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "simple_summary.csv")
    
    # Prepare CSV rows
    rows = [
        ["Metric", "Value"],
        ["Number of Samples", summary['n_samples']],
        ["", ""],  # Empty row for separation
        ["Breakage Analysis", ""],
        ["Main Trunk Breakage Rate (%)", f"{summary['breakage_rate']:.2f}"],
        ["Success Rate (%)", f"{summary['success_rate']:.2f}"],
        ["Any Wood Disconnection Rate (%)", f"{summary['broken_rate']:.2f}"],
        ["Average Base-Connected Ratio", f"{summary['avg_base_connected_ratio']:.4f}"],
        ["", ""],  # Empty row for separation
        ["Size Statistics", ""],
        ["Avg Mass (voxels)", f"{summary['avg_mass']:.1f}"],
        ["Std Mass (voxels)", f"{summary['std_mass']:.1f}"],
        ["Avg Height", f"{summary['avg_height']:.1f}"],
        ["Std Height", f"{summary['std_height']:.1f}"],
        ["Avg Log Size (voxels)", f"{summary['avg_log_size']:.1f}"],
        ["Std Log Size (voxels)", f"{summary['std_log_size']:.1f}"],
        ["Avg Leaf Size (voxels)", f"{summary['avg_leaf_size']:.1f}"],
        ["Std Leaf Size (voxels)", f"{summary['std_leaf_size']:.1f}"],
        ["Avg Base Connected Size (voxels)", f"{summary['avg_base_connected_size']:.1f}"],
        ["Std Base Connected Size (voxels)", f"{summary['std_base_connected_size']:.1f}"],
        ["Avg Base Connected Ratio", f"{summary['avg_base_connected_ratio']:.4f}"],
        ["Std Base Connected Ratio", f"{summary['std_base_connected_ratio']:.4f}"],
    ]
    
    # Add largest log ratio if available
    if summary['avg_largest_log_ratio'] >= 0:
        rows.append(["Avg Largest Log Ratio", f"{summary['avg_largest_log_ratio']:.4f}"])
    
    rows.extend([
        ["", ""],  # Empty row for separation
        ["Occupancy Rates", ""],
        ["Non-Air", f"{summary['avg_occupancy_non_air']:.4f}"],
        ["Log", f"{summary['avg_occupancy_log']:.4f}"],
        ["Leaf", f"{summary['avg_occupancy_leaf']:.4f}"],
        ["", ""],  # Empty row for separation
        ["Component Counts", ""],
        ["Non-Air", f"{summary['avg_components_non_air']:.2f}"],
        ["Log", f"{summary['avg_components_log']:.2f}"],
        ["Leaf", f"{summary['avg_components_leaf']:.2f}"],
    ])
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    console.print(f"[green]✓[/green] Saved summary: {csv_path}")


def save_dynamics_summary(summary: Dict, output_dir: str, console: Optional[Console] = None):
    """
    保存 dynamics 統計摘要到 CSV 文件。
    
    Args:
        summary: dynamics summary statistics dict
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    if not summary:
        console.print("[yellow]⚠[/yellow] No dynamics summary to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dynamics_summary.csv")
    
    rows = [
        ["Metric", "Value"],
        ["Number of Samples", summary['n_samples']],
        ["", ""],
        ["Breakage Analysis", ""],
        ["Main Trunk Breakage Rate (%)", f"{summary['breakage_rate']:.2f}"],
        ["Success Rate (%)", f"{summary['success_rate']:.2f}"],
        ["Any Wood Disconnection Rate (%)", f"{summary['broken_rate']:.2f}"],
        ["Average Base-Connected Ratio", f"{summary['avg_base_connected_ratio']:.4f}"],
        ["", ""],
        ["Size Statistics", ""],
        ["Avg Mass (voxels)", f"{summary['avg_mass']:.1f}"],
        ["Std Mass (voxels)", f"{summary['std_mass']:.1f}"],
        ["Avg Height", f"{summary['avg_height']:.1f}"],
        ["Std Height", f"{summary['std_height']:.1f}"],
        ["Avg Log Size (voxels)", f"{summary['avg_log_size']:.1f}"],
        ["Std Log Size (voxels)", f"{summary['std_log_size']:.1f}"],
        ["Avg Leaf Size (voxels)", f"{summary['avg_leaf_size']:.1f}"],
        ["Std Leaf Size (voxels)", f"{summary['std_leaf_size']:.1f}"],
        ["Avg Base Connected Size (voxels)", f"{summary['avg_base_connected_size']:.1f}"],
        ["Std Base Connected Size (voxels)", f"{summary['std_base_connected_size']:.1f}"],
        ["Avg Base Connected Ratio", f"{summary['avg_base_connected_ratio']:.4f}"],
        ["Std Base Connected Ratio", f"{summary['std_base_connected_ratio']:.4f}"],
        ["", ""],
        ["t_emerge/t_lockin Statistics", ""],
        ["t_emerge Found Samples", summary['n_t_emerge_found']],
        ["t_emerge Missing Rate", f"{summary['t_emerge_missing_rate']:.4f}"],
        ["Avg t_emerge", f"{summary['avg_t_emerge']:.2f}" if summary['avg_t_emerge'] >= 0 else "N/A"],
        ["Std t_emerge", f"{summary['std_t_emerge']:.2f}" if summary['std_t_emerge'] >= 0 else "N/A"],
        ["t_lockin Found Samples", summary['n_t_lockin_found']],
        ["t_lockin Missing Rate", f"{summary['t_lockin_missing_rate']:.4f}"],
        ["Avg t_lockin", f"{summary['avg_t_lockin']:.2f}" if summary['avg_t_lockin'] >= 0 else "N/A"],
        ["Std t_lockin", f"{summary['std_t_lockin']:.2f}" if summary['std_t_lockin'] >= 0 else "N/A"],
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    console.print(f"[green]✓[/green] Saved dynamics summary: {csv_path}")


def save_dynamics_trace(trace_data: List[Dict], output_dir: str, console: Optional[Console] = None):
    """
    保存動力學軌跡到 CSV。
    
    Args:
        trace_data: List of dynamics metrics dicts
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dynamics_trace.csv")
    
    if not trace_data:
        console.print("[yellow]⚠[/yellow] No trace data to save")
        return
    
    # Sort trace_data: first by sample_idx, then by step_idx
    sorted_trace_data = sorted(trace_data, key=lambda x: (x.get('sample_idx', 0), x.get('step_idx', 0)))
    
    # Convert sample_idx from 0-based to 1-based for CSV output (to match file naming)
    csv_data = []
    for row in sorted_trace_data:
        csv_row = row.copy()
        # Convert sample_idx from 0-based to 1-based
        if 'sample_idx' in csv_row:
            csv_row['sample_idx'] = csv_row['sample_idx'] + 1
        csv_data.append(csv_row)
    
    # Define fieldnames with specific order
    # Base fields first
    base_fields = ['sample_idx', 'step_idx', 't']
    
    # Define the order of metric fields (Base_Connected_Ratio before Largest_Log_Ratio)
    metric_field_order = [
        'Is_Main_Trunk_Broken',
        'Is_Broken',
        'Mass',
        'Height',
        'Log_Size',
        'Leaf_Size',
        'Base_Connected_Size',
        'Base_Connected_Ratio',  # Before Largest_Log_Ratio
        'Largest_Log_Ratio',
        'Occupancy_Non_Air',
        'Occupancy_Log',
        'Occupancy_Leaf',
        'Components_Non_Air',
        'Components_Log',
        'Components_Leaf',
    ]
    
    # Remove Total_Log_Size from data before saving
    for row in csv_data:
        if 'Total_Log_Size' in row:
            del row['Total_Log_Size']
    
    # Get all keys from data
    all_keys = set()
    for row in csv_data:
        all_keys.update(row.keys())
    
    # Build fieldnames: base fields + ordered metric fields + any remaining fields
    fieldnames = base_fields.copy()
    for field in metric_field_order:
        if field in all_keys:
            fieldnames.append(field)
    
    # Add any remaining fields that weren't in the ordered list
    remaining_fields = sorted([k for k in all_keys if k not in fieldnames])
    fieldnames.extend(remaining_fields)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    console.print(f"[green]✓[/green] Saved dynamics trace: {csv_path} ({len(trace_data)} tracking points)")


def plot_divergence(trace_data: List[Dict], output_dir: str, n_dynamics_samples: int = 5, console: Optional[Console] = None):
    """
    繪製 divergence 圖表：顯示不同樣本在採樣過程中的指標變化。
    
    Args:
        trace_data: List of dynamics metrics dicts
        output_dir: output directory
        n_dynamics_samples: number of dynamics samples (if > 10, only show 10 randomly selected samples)
        console: rich console
    """
    if console is None:
        console = Console()
    
    if not trace_data:
        console.print("[yellow]⚠[/yellow] No trace data to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "dynamics_divergence_plot.png")
    
    # Group by sample_idx
    samples = {}
    for row in trace_data:
        sid = row['sample_idx']
        if sid not in samples:
            samples[sid] = []
        samples[sid].append(row)
    
    # Sort each sample's data by t (timestep)
    for sid in samples:
        samples[sid].sort(key=lambda x: x['t'], reverse=True)  # t decreases: 1000 -> 0
    
    # Find T (maximum timestep) from trace_data
    T = max(row['t'] for row in trace_data) if trace_data else 1000
    t_threshold = int(T * 0.95)  # Start from 95% of T
    
    # Create figure with subplots (2 rows, 2 cols = 4 plots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Determine final state (is_main_trunk_broken) for each sample
    # This is used for averaging in Plot 3 and Plot 4
    sample_final_states = {}
    for sid in samples.keys():
        data = samples[sid]
        # Get final state (lowest t value, which is the final state)
        if data:
            final_data = min(data, key=lambda x: x['t'])
            sample_final_states[sid] = final_data.get('Is_Main_Trunk_Broken', False)
    
    # If n_dynamics_samples > 10, randomly select 10 samples to plot
    samples_to_plot = sorted(samples.keys())
    if n_dynamics_samples > 10:
        samples_to_plot = sorted(random.sample(list(samples.keys()), min(10, len(samples.keys()))))
    
    # Plot 1: Mass over time
    ax1 = axes[0, 0]
    for sid in samples_to_plot:
        data = samples[sid]
        ts = [r['t'] for r in data]
        masses = [r['Mass'] for r in data]
        ax1.plot(ts, masses, label=f'Sample {sid+1}', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Timestep (t)')
    ax1.set_ylabel('Mass (non-air voxels)')
    ax1.set_title('Mass Evolution During Sampling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # t decreases from 1000 to 0
    
    # Plot 2: Is_Main_Trunk_Broken over time (as fraction)
    ax2 = axes[0, 1]
    # Aggregate main trunk breakage rate across ALL dynamics samples per timestep
    total_counts_by_t = {}
    broken_counts_by_t = {}
    total_counts_broken_group_by_t = {}
    broken_counts_broken_group_by_t = {}
    total_counts_unbroken_group_by_t = {}
    broken_counts_unbroken_group_by_t = {}
    for sid, data in samples.items():
        is_final_broken = sample_final_states.get(sid, False)
        for r in data:
            t_val = r['t']
            total_counts_by_t[t_val] = total_counts_by_t.get(t_val, 0) + 1
            if r.get('Is_Main_Trunk_Broken', False):
                broken_counts_by_t[t_val] = broken_counts_by_t.get(t_val, 0) + 1
            else:
                broken_counts_by_t.setdefault(t_val, broken_counts_by_t.get(t_val, 0))
            
            # Split by final state group (broken vs unbroken)
            if is_final_broken:
                total_counts_broken_group_by_t[t_val] = total_counts_broken_group_by_t.get(t_val, 0) + 1
                if r.get('Is_Main_Trunk_Broken', False):
                    broken_counts_broken_group_by_t[t_val] = broken_counts_broken_group_by_t.get(t_val, 0) + 1
                else:
                    broken_counts_broken_group_by_t.setdefault(
                        t_val, broken_counts_broken_group_by_t.get(t_val, 0)
                    )
            else:
                total_counts_unbroken_group_by_t[t_val] = total_counts_unbroken_group_by_t.get(t_val, 0) + 1
                if r.get('Is_Main_Trunk_Broken', False):
                    broken_counts_unbroken_group_by_t[t_val] = broken_counts_unbroken_group_by_t.get(t_val, 0) + 1
                else:
                    broken_counts_unbroken_group_by_t.setdefault(
                        t_val, broken_counts_unbroken_group_by_t.get(t_val, 0)
                    )
    if total_counts_by_t:
        sorted_ts = sorted(total_counts_by_t.keys(), reverse=True)
        rates = []
        rates_broken_group = []
        rates_unbroken_group = []
        for t_val in sorted_ts:
            total = total_counts_by_t[t_val]
            broken = broken_counts_by_t.get(t_val, 0)
            rate = broken / total if total > 0 else 0.0
            rates.append(rate)
            
            total_b = total_counts_broken_group_by_t.get(t_val, 0)
            broken_b = broken_counts_broken_group_by_t.get(t_val, 0)
            rates_broken_group.append((broken_b / total_b) if total_b > 0 else np.nan)
            
            total_u = total_counts_unbroken_group_by_t.get(t_val, 0)
            broken_u = broken_counts_unbroken_group_by_t.get(t_val, 0)
            rates_unbroken_group.append((broken_u / total_u) if total_u > 0 else np.nan)
        
        ax2.plot(sorted_ts, rates, color='purple', linewidth=2.2, label='Breakage Rate (All)')
        ax2.plot(sorted_ts, rates_broken_group, color='red', linewidth=2.0, linestyle='--', label='Breakage Rate (Broken)')
        ax2.plot(sorted_ts, rates_unbroken_group, color='green', linewidth=2.0, linestyle='--', label='Breakage Rate (Unbroken)')
    ax2.set_xlabel('Timestep (t)')
    ax2.set_ylabel('Main Trunk Breakage Rate')
    ax2.set_title('Main Trunk Breakage Rate During Sampling (All Dynamics Samples)')
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Base Connected Ratio over time (from t=95% of T to 0)
    ax3 = axes[1, 0]
    
    # Collect all data for averaging (use all samples)
    all_ratios_by_t = {}  # {t: [ratios]}
    broken_ratios_by_t = {}  # {t: [ratios]}
    unbroken_ratios_by_t = {}  # {t: [ratios]}
    
    # First pass: collect all data for averaging
    for sid in sorted(samples.keys()):
        data = samples[sid]
        # Filter data: only show t <= t_threshold (from 95% of T to 0)
        filtered_data = [r for r in data if r['t'] <= t_threshold]
        if filtered_data:  # Only process if there's data after threshold
            ts = [r['t'] for r in filtered_data]
            # Calculate base_connected_ratio = Base_Connected_Size / Total_Log_Size
            base_connected_ratios = []
            for r in filtered_data:
                total_log_size = r.get('Total_Log_Size', r.get('Log_Size', 0))
                if total_log_size > 0:
                    ratio = r['Base_Connected_Size'] / total_log_size
                else:
                    ratio = 0.0
                base_connected_ratios.append(ratio)
            
            # Collect data for averaging (all samples)
            is_broken = sample_final_states.get(sid, False)
            for t, ratio in zip(ts, base_connected_ratios):
                if t not in all_ratios_by_t:
                    all_ratios_by_t[t] = []
                    broken_ratios_by_t[t] = []
                    unbroken_ratios_by_t[t] = []
                all_ratios_by_t[t].append(ratio)
                if is_broken:
                    broken_ratios_by_t[t].append(ratio)
                else:
                    unbroken_ratios_by_t[t].append(ratio)
    
    # Second pass: plot only selected samples
    for sid in samples_to_plot:
        data = samples[sid]
        # Filter data: only show t <= t_threshold (from 95% of T to 0)
        filtered_data = [r for r in data if r['t'] <= t_threshold]
        if filtered_data:  # Only plot if there's data after threshold
            ts = [r['t'] for r in filtered_data]
            # Calculate base_connected_ratio = Base_Connected_Size / Total_Log_Size
            base_connected_ratios = []
            for r in filtered_data:
                total_log_size = r.get('Total_Log_Size', r.get('Log_Size', 0))
                if total_log_size > 0:
                    ratio = r['Base_Connected_Size'] / total_log_size
                else:
                    ratio = 0.0
                base_connected_ratios.append(ratio)
            
            # Plot individual sample (only selected samples)
            ax3.plot(ts, base_connected_ratios, label=f'Sample {sid+1}', alpha=0.7, linewidth=1.5)
    
    # Calculate and plot averages
    if all_ratios_by_t:
        sorted_ts = sorted(all_ratios_by_t.keys(), reverse=True)
        avg_all = [np.mean(all_ratios_by_t[t]) for t in sorted_ts]
        avg_broken = [np.mean(broken_ratios_by_t[t]) if broken_ratios_by_t[t] else np.nan for t in sorted_ts]
        avg_unbroken = [np.mean(unbroken_ratios_by_t[t]) if unbroken_ratios_by_t[t] else np.nan for t in sorted_ts]
        
        ax3.plot(sorted_ts, avg_all, label='Avg (All)', color='black', linewidth=2.5, linestyle='-', alpha=0.9)
        ax3.plot(sorted_ts, avg_broken, label='Avg (Broken)', color='red', linewidth=2.5, linestyle='--', alpha=0.9)
        ax3.plot(sorted_ts, avg_unbroken, label='Avg (Unbroken)', color='green', linewidth=2.5, linestyle='--', alpha=0.9)
    
    ax3.set_xlabel('Timestep (t)')
    ax3.set_ylabel('Base Connected Ratio')
    ax3.set_title(f'Base-Connected Ratio Evolution (from t={t_threshold} to 0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    ax3.set_ylim([0, 1.1])  # Ratio should be between 0 and 1
    
    # Plot 4: Largest Log Ratio over time (from t=95% of T to 0)
    ax4 = axes[1, 1]
    
    # Collect all data for averaging (use all samples)
    all_ratios_by_t = {}  # {t: [ratios]}
    broken_ratios_by_t = {}  # {t: [ratios]}
    unbroken_ratios_by_t = {}  # {t: [ratios]}
    
    # First pass: collect all data for averaging
    for sid in sorted(samples.keys()):
        data = samples[sid]
        # Filter data: only show t <= t_threshold (from 95% of T to 0)
        filtered_data = [r for r in data if r['t'] <= t_threshold]
        if filtered_data:  # Only process if there's data after threshold
            # Filter out invalid ratios (< 0)
            largest_log_ratios = []
            valid_ts = []
            for r in filtered_data:
                ratio = r.get('Largest_Log_Ratio', -1.0)
                if ratio >= 0:
                    largest_log_ratios.append(ratio)
                    valid_ts.append(r['t'])
            if largest_log_ratios:  # Only process if there's valid data
                # Collect data for averaging (all samples)
                is_broken = sample_final_states.get(sid, False)
                for t, ratio in zip(valid_ts, largest_log_ratios):
                    if t not in all_ratios_by_t:
                        all_ratios_by_t[t] = []
                        broken_ratios_by_t[t] = []
                        unbroken_ratios_by_t[t] = []
                    all_ratios_by_t[t].append(ratio)
                    if is_broken:
                        broken_ratios_by_t[t].append(ratio)
                    else:
                        unbroken_ratios_by_t[t].append(ratio)
    
    # Second pass: plot only selected samples
    for sid in samples_to_plot:
        data = samples[sid]
        # Filter data: only show t <= t_threshold (from 95% of T to 0)
        filtered_data = [r for r in data if r['t'] <= t_threshold]
        if filtered_data:  # Only plot if there's data after threshold
            # Filter out invalid ratios (< 0)
            largest_log_ratios = []
            valid_ts = []
            for r in filtered_data:
                ratio = r.get('Largest_Log_Ratio', -1.0)
                if ratio >= 0:
                    largest_log_ratios.append(ratio)
                    valid_ts.append(r['t'])
            if largest_log_ratios:  # Only plot if there's valid data
                ax4.plot(valid_ts, largest_log_ratios, label=f'Sample {sid+1}', alpha=0.7, linewidth=1.5)
    
    # Calculate and plot averages
    if all_ratios_by_t:
        sorted_ts = sorted(all_ratios_by_t.keys(), reverse=True)
        avg_all = [np.mean(all_ratios_by_t[t]) for t in sorted_ts]
        avg_broken = [np.mean(broken_ratios_by_t[t]) if broken_ratios_by_t[t] else np.nan for t in sorted_ts]
        avg_unbroken = [np.mean(unbroken_ratios_by_t[t]) if unbroken_ratios_by_t[t] else np.nan for t in sorted_ts]
        
        ax4.plot(sorted_ts, avg_all, label='Avg (All)', color='black', linewidth=2.5, linestyle='-', alpha=0.9)
        ax4.plot(sorted_ts, avg_broken, label='Avg (Broken)', color='red', linewidth=2.5, linestyle='--', alpha=0.9)
        ax4.plot(sorted_ts, avg_unbroken, label='Avg (Unbroken)', color='green', linewidth=2.5, linestyle='--', alpha=0.9)
    
    ax4.set_xlabel('Timestep (t)')
    ax4.set_ylabel('Largest Log Ratio')
    ax4.set_title(f'Largest Log Component Ratio Evolution (from t={t_threshold} to 0)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    ax4.set_ylim([0, 1.1])  # Ratio should be between 0 and 1
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Saved divergence plot: {png_path}")


def plot_timing_distributions(trace_data: List[Dict], output_dir: str, console: Optional[Console] = None):
    """
    繪製 t_emerge 與 t_lockin 的分布圖，另存為獨立圖片。
    """
    if console is None:
        console = Console()
    
    if not trace_data:
        console.print("[yellow]⚠[/yellow] No trace data to plot timing distributions")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "dynamics_timing_distribution_plot.png")
    
    # Group by sample_idx
    samples = {}
    for row in trace_data:
        sid = row['sample_idx']
        if sid not in samples:
            samples[sid] = []
        samples[sid].append(row)
    
    # Determine final state (is_main_trunk_broken) for each sample
    sample_final_states = {}
    for sid in samples.keys():
        data = samples[sid]
        if data:
            final_data = min(data, key=lambda x: x['t'])
            sample_final_states[sid] = final_data.get('Is_Main_Trunk_Broken', False)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # Histogram 1: t_emerge
    ax1 = axes[0]
    t_emerge_broken = []
    t_emerge_unbroken = []
    for sid in sorted(samples.keys()):
        sample_trace = samples[sid]
        t_emerge, _ = compute_t_emerge_and_t_lockin(sample_trace)
        if t_emerge is not None:
            if sample_final_states.get(sid, False):
                t_emerge_broken.append(t_emerge)
            else:
                t_emerge_unbroken.append(t_emerge)
    
    if t_emerge_broken or t_emerge_unbroken:
        data_to_plot = []
        labels = []
        colors = []
        if t_emerge_broken:
            data_to_plot.append(t_emerge_broken)
            labels.append('Broken')
            colors.append('red')
        if t_emerge_unbroken:
            data_to_plot.append(t_emerge_unbroken)
            labels.append('Unbroken')
            colors.append('green')
        ax1.hist(data_to_plot, bins=30, edgecolor='black', alpha=0.65, label=labels, color=colors)
        if t_emerge_broken:
            mean_b = float(np.mean(t_emerge_broken))
            ax1.axvline(mean_b, color='red', linestyle='--', linewidth=2.0, label='Broken mean')
        if t_emerge_unbroken:
            mean_u = float(np.mean(t_emerge_unbroken))
            ax1.axvline(mean_u, color='green', linestyle='--', linewidth=2.0, label='Unbroken mean')
        ax1.set_xlabel('t_emerge (Timestep)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of t_emerge (Tree Emergence Time)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
    else:
        ax1.text(0.5, 0.5, 'No t_emerge data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Distribution of t_emerge (Tree Emergence Time)')
    
    # Histogram 2: t_lockin
    ax2 = axes[1]
    t_lockin_broken = []
    t_lockin_unbroken = []
    for sid in sorted(samples.keys()):
        sample_trace = samples[sid]
        _, t_lockin = compute_t_emerge_and_t_lockin(sample_trace)
        if t_lockin is not None:
            if sample_final_states.get(sid, False):
                t_lockin_broken.append(t_lockin)
            else:
                t_lockin_unbroken.append(t_lockin)
    
    if t_lockin_broken or t_lockin_unbroken:
        data_to_plot = []
        labels = []
        colors = []
        if t_lockin_broken:
            data_to_plot.append(t_lockin_broken)
            labels.append('Broken')
            colors.append('red')
        if t_lockin_unbroken:
            data_to_plot.append(t_lockin_unbroken)
            labels.append('Unbroken')
            colors.append('green')
        ax2.hist(data_to_plot, bins=30, edgecolor='black', alpha=0.65, label=labels, color=colors)
        if t_lockin_broken:
            mean_b = float(np.mean(t_lockin_broken))
            ax2.axvline(mean_b, color='red', linestyle='--', linewidth=2.0, label='Broken mean')
        if t_lockin_unbroken:
            mean_u = float(np.mean(t_lockin_unbroken))
            ax2.axvline(mean_u, color='green', linestyle='--', linewidth=2.0, label='Unbroken mean')
        ax2.set_xlabel('t_lockin (Timestep)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of t_lockin (Trunk State Lock-in Time)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()
    else:
        ax2.text(0.5, 0.5, 'No t_lockin data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distribution of t_lockin (Trunk State Lock-in Time)')
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓[/green] Saved timing distribution plot: {png_path}")


def fmt_secs(s: float) -> str:
    """Format seconds to human-readable string."""
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


def save_metadata(metadata: Dict, output_dir: str, console: Optional[Console] = None):
    """
    保存 metadata 到 CSV 文件（key-value 格式和 flat 格式）。
    
    Args:
        metadata: metadata dictionary
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Key-value format
    csv_kv_path = os.path.join(output_dir, "metadata.csv")
    with open(csv_kv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in metadata.items():
            writer.writerow([k, v])
    
    # Flat format
    csv_flat_path = os.path.join(output_dir, "metadata_flat.csv")
    with open(csv_flat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)
    
    console.print(f"[green]✓[/green] Saved metadata: [cyan]{csv_kv_path}[/cyan]")
    console.print(f"[green]✓[/green] Saved metadata (flat): [cyan]{csv_flat_path}[/cyan]")


def append_metadata(metadata: Dict, output_dir: str, console: Optional[Console] = None):
    """
    追加 metadata 到现有的 CSV 文件（key-value 格式）。
    同时更新 flat 格式文件。
    
    Args:
        metadata: metadata dictionary to append
        output_dir: output directory
        console: rich console
    """
    if console is None:
        console = Console()
    
    # Append to key-value format
    csv_kv_path = os.path.join(output_dir, "metadata.csv")
    if os.path.exists(csv_kv_path):
        with open(csv_kv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for k, v in metadata.items():
                writer.writerow([k, v])
    else:
        # If file doesn't exist, create it
        save_metadata(metadata, output_dir, console)
        return
    
    # Update flat format
    csv_flat_path = os.path.join(output_dir, "metadata_flat.csv")
    existing_flat_metadata = {}
    if os.path.exists(csv_flat_path):
        with open(csv_flat_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                row = next(reader, None)
                if row:
                    existing_flat_metadata = row
    
    # Merge with new metadata
    all_flat_metadata = {**existing_flat_metadata, **metadata}
    
    # Rewrite flat format with all fields
    with open(csv_flat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_flat_metadata.keys())
        writer.writeheader()
        writer.writerow(all_flat_metadata)
    
    console.print(f"[green]✓[/green] Updated metadata with additional fields")


def main():
    global_t0 = time.time()
    eval_start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description="Evaluate 16x16x16 Voxel Diffusion Model")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--base_channels", type=int, default=64, help="Base number of channels (if not in checkpoint)")
    parser.add_argument("--time_dim", type=int, default=128, help="Time embedding dimension (if not in checkpoint)")
    
    # Sampling
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples for batch evaluation")
    parser.add_argument("--n_dynamics_samples", type=int, default=5, help="Number of samples for dynamics analysis")
    parser.add_argument("--n_steps", type=int, default=None, help="Number of sampling steps (default: T from checkpoint)")
    parser.add_argument("--track_every", type=int, default=50, help="Track metrics every N steps during dynamics analysis")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for parallel generation (default: 10)")
    parser.add_argument("--dynamics_batch_size", type=int, default=None, help="Batch size for dynamics evaluation (default: same as --batch_size)")
    
    # Diffusion schedule
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion timesteps (if not in checkpoint)")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Beta schedule")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta start value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end value")
    
    # Output
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (used as output directory name, e.g., 'baseline_eval_v1')"
    )
    
    # Misc
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--no_projections", action="store_true", help="Disable saving 3-view projections (default: enabled)")
    parser.add_argument("--save_track_projections", action="store_true", help="Save 3-view projections at each tracking step during dynamics analysis (default: disabled)")
    parser.add_argument("--save_npz", action="store_true", help="Save npz files for each sample (default: disabled)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None, use random seed)")
    parser.add_argument(
        "--log_mask_threshold",
        type=float,
        default=None,
        help="Use threshold-based log mask decoding (e.g., 0.4/0.5/0.6). If not set, use argmax decoding."
    )
    
    args = parser.parse_args()
    
    # Set random seeds if seed is provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # Make cuDNN behavior deterministic when possible.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Enforce deterministic algorithms where PyTorch supports it.
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    console = Console()
    console.print("[bold]16x16x16 Voxel Diffusion Model Evaluation[/bold]\n")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Using device: {device}[/cyan]")
    if args.seed is not None:
        console.print(f"[cyan]Random seed: {args.seed}[/cyan]")
    if args.log_mask_threshold is None:
        console.print("[cyan]Label decoding: argmax[/cyan]")
    else:
        console.print(f"[cyan]Label decoding: log-mask threshold = {args.log_mask_threshold:.4f}[/cyan]")
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Get model info
    checkpoint_args = checkpoint.get('args', {})
    num_params = sum(p.numel() for p in model.parameters())
    base_channels = checkpoint_args.get('base_channels', args.base_channels)
    time_dim = checkpoint_args.get('time_dim', args.time_dim)
    checkpoint_epoch = checkpoint.get('epoch', 'unknown')
    checkpoint_best_val_loss = checkpoint.get('best_val_loss', 'unknown')
    
    # Get T and beta schedule from checkpoint args or use command line args
    T = checkpoint_args.get('T', args.T)
    beta_schedule = checkpoint_args.get('beta_schedule', args.beta_schedule)
    beta_start = checkpoint_args.get('beta_start', args.beta_start) if 'beta_start' in checkpoint_args else args.beta_start
    beta_end = checkpoint_args.get('beta_end', args.beta_end) if 'beta_end' in checkpoint_args else args.beta_end
    
    console.print(f"[cyan]Diffusion schedule: T={T}, schedule={beta_schedule}[/cyan]")
    
    # Create beta schedule
    betas = BetaSchedule(T=T, schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end).to(device)
    
    # Mixed precision
    use_amp = (device.type == "cuda") and (not args.no_amp)
    console.print(f"[cyan]Mixed precision: {use_amp}[/cyan]\n")
    
    # Use experiment name directly as output directory
    exp_output_dir = args.exp_name
    os.makedirs(exp_output_dir, exist_ok=True)
    console.print(f"[cyan]Experiment name: {args.exp_name}[/cyan]")
    console.print(f"[cyan]Output directory: {exp_output_dir}[/cyan]\n")
    
    # Helper function for boolean to string
    def bool_to_str(v):
        return "TRUE" if v else "FALSE"
    
    # Get script name
    current_script = Path(__file__).name if "__file__" in globals() else "interactive_session"
    invocation_command = get_invocation_command()
    
    # Prepare initial metadata
    initial_metadata = {
        "exp_name": args.exp_name,
        "evaluation_start_time": eval_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_cwd": os.getcwd(),
        "execution_command": invocation_command,
        "checkpoint_path": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_val_loss": checkpoint_best_val_loss,
        "output_directory": exp_output_dir,
        "script_name": current_script,
        "device": str(device),
        "seed": args.seed if args.seed is not None else "None",
        "amp_enabled": bool_to_str(use_amp),
        "no_amp": bool_to_str(args.no_amp),
        "base_channels": base_channels,
        "time_dim": time_dim,
        "model_total_params": num_params,
        "T": T,
        "beta_schedule": beta_schedule,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "n_samples": args.n_samples,
        "n_dynamics_samples": args.n_dynamics_samples,
        "n_steps": args.n_steps if args.n_steps is not None else T,
        "track_every": args.track_every,
        "batch_size": args.batch_size,
        "dynamics_batch_size": args.dynamics_batch_size if args.dynamics_batch_size is not None else args.batch_size,
        "save_projections": bool_to_str(not args.no_projections),
        "no_projections": bool_to_str(args.no_projections),
        "save_track_projections": bool_to_str(args.save_track_projections),
        "save_npz": bool_to_str(args.save_npz),
        "label_decoding_mode": "argmax" if args.log_mask_threshold is None else "log_mask_threshold",
        "log_mask_threshold": args.log_mask_threshold if args.log_mask_threshold is not None else "None",
        # New standardized output filenames
        "simple_eval_csv": os.path.join(exp_output_dir, "simple_result.csv"),
        "simple_summary_csv": os.path.join(exp_output_dir, "simple_summary.csv"),
        "dynamics_result_csv": os.path.join(exp_output_dir, "dynamics_result.csv"),
        "dynamics_summary_csv": os.path.join(exp_output_dir, "dynamics_summary.csv"),
        "dynamics_trace_csv": os.path.join(exp_output_dir, "dynamics_trace.csv"),
        "dynamics_divergence_plot_png": os.path.join(exp_output_dir, "dynamics_divergence_plot.png"),
        "dynamics_timing_distribution_plot_png": os.path.join(exp_output_dir, "dynamics_timing_distribution_plot.png"),
        # Backward-compat keys pointing to new paths
        "eval_batch_csv": os.path.join(exp_output_dir, "simple_result.csv"),
        "eval_summary_csv": os.path.join(exp_output_dir, "simple_summary.csv"),
        "divergence_plot_png": os.path.join(exp_output_dir, "dynamics_divergence_plot.png"),
    }
    
    # Save initial metadata
    save_metadata(initial_metadata, exp_output_dir, console)
    
    # ============================================
    # 層次 A & B: 批量評估（只記錄最終狀態）
    # ============================================
    console.print("[bold cyan]" + "="*60 + "[/bold cyan]")
    console.print("[bold cyan]Level A & B: Batch Evaluation[/bold cyan]")
    console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")
    
    # Determine batch sizes
    batch_eval_batch_size = args.batch_size
    dynamics_batch_size = args.dynamics_batch_size if args.dynamics_batch_size is not None else args.batch_size
    
    batch_metrics, batch_eval_time = batch_evaluation(
        model=model,
        betas=betas,
        device=device,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        batch_size=batch_eval_batch_size,
        use_amp=use_amp,
        output_dir=exp_output_dir,
        exp_name=args.exp_name,
        save_projections=not args.no_projections,
        save_npz=args.save_npz,
        log_mask_threshold=args.log_mask_threshold,
        console=console,
    )
    
    # Save batch results (Level A)
    save_batch_results(batch_metrics, exp_output_dir, console)
    
    # Compute and save summary (Level B)
    summary = compute_summary_statistics(batch_metrics)
    save_summary(summary, exp_output_dir, console)
    
    # Print summary to console
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Main Trunk Breakage Rate: {summary['breakage_rate']:.2f}%")
    console.print(f"  Broken Rate: {summary['broken_rate']:.2f}%")
    console.print(f"  Success Rate: {summary['success_rate']:.2f}%")
    console.print(f"  Avg Mass: {summary['avg_mass']:.1f} ± {summary['std_mass']:.1f} voxels")
    console.print(f"  Avg Height: {summary['avg_height']:.1f} ± {summary['std_height']:.1f}\n")
    
    # ============================================
    # 層次 C: 動力學分析（記錄每一步）
    # ============================================
    console.print("[bold cyan]" + "="*60 + "[/bold cyan]")
    console.print("[bold cyan]Level C: Dynamics Analysis[/bold cyan]")
    console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")
    
    trace_data, dynamics_eval_time, dynamics_final_metrics = dynamics_evaluation(
        model=model,
        betas=betas,
        device=device,
        n_samples=args.n_dynamics_samples,
        n_steps=args.n_steps,
        track_every=args.track_every,
        batch_size=dynamics_batch_size,
        use_amp=use_amp,
        output_dir=exp_output_dir,
        exp_name=args.exp_name,
        save_projections=not args.no_projections,
        save_track_projections=args.save_track_projections,
        save_npz=args.save_npz,
        log_mask_threshold=args.log_mask_threshold,
        console=console,
    )
    
    # Save dynamics samples final results (Level C - final states)
    save_dynamics_samples(dynamics_final_metrics, exp_output_dir, console)
    
    # Compute and save dynamics summary (Level C - summary)
    dynamics_summary = compute_dynamics_summary_statistics(dynamics_final_metrics)
    save_dynamics_summary(dynamics_summary, exp_output_dir, console)
    
    # Save dynamics trace (Level C)
    save_dynamics_trace(trace_data, exp_output_dir, console)
    
    # Plot divergence
    plot_divergence(trace_data, exp_output_dir, args.n_dynamics_samples, console)
    
    # Plot timing distributions (t_emerge / t_lockin)
    plot_timing_distributions(trace_data, exp_output_dir, console)
    
    # Compute total evaluation time
    total_secs = time.time() - global_t0
    eval_end_time = datetime.now()
    
    # Prepare final metadata (evaluation results)
    final_metadata = {
        "evaluation_end_time": eval_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluation_time_secs": total_secs,
        "total_evaluation_time_formatted": fmt_secs(total_secs),
        # Inference times
        "batch_evaluation_time_secs": batch_eval_time,
        "batch_evaluation_time_formatted": fmt_secs(batch_eval_time),
        "batch_evaluation_time_per_sample_secs": batch_eval_time / args.n_samples if args.n_samples > 0 else 0.0,
        "dynamics_evaluation_time_secs": dynamics_eval_time,
        "dynamics_evaluation_time_formatted": fmt_secs(dynamics_eval_time),
        "dynamics_evaluation_time_per_sample_secs": dynamics_eval_time / args.n_dynamics_samples if args.n_dynamics_samples > 0 else 0.0,
        # Summary statistics
        "breakage_rate": summary['breakage_rate'],
        "broken_rate": summary['broken_rate'],
        "success_rate": summary['success_rate'],
        "avg_mass": summary['avg_mass'],
        "std_mass": summary['std_mass'],
        "avg_height": summary['avg_height'],
        "std_height": summary['std_height'],
        "avg_log_size": summary['avg_log_size'],
        "std_log_size": summary['std_log_size'],
        "avg_leaf_size": summary['avg_leaf_size'],
        "std_leaf_size": summary['std_leaf_size'],
        "avg_base_connected_size": summary['avg_base_connected_size'],
        "std_base_connected_size": summary['std_base_connected_size'],
        "avg_base_connected_ratio": summary['avg_base_connected_ratio'],
        "std_base_connected_ratio": summary['std_base_connected_ratio'],
        "avg_largest_log_ratio": summary['avg_largest_log_ratio'] if summary['avg_largest_log_ratio'] >= 0 else -1.0,
        "avg_occupancy_non_air": summary['avg_occupancy_non_air'],
        "avg_occupancy_log": summary['avg_occupancy_log'],
        "avg_occupancy_leaf": summary['avg_occupancy_leaf'],
        "avg_components_non_air": summary['avg_components_non_air'],
        "avg_components_log": summary['avg_components_log'],
        "avg_components_leaf": summary['avg_components_leaf'],
        "n_dynamics_tracking_points": len(trace_data),
        # Dynamics final-state summary statistics
        "dynamics_n_samples": dynamics_summary.get('n_samples', 0),
        "dynamics_breakage_rate": dynamics_summary.get('breakage_rate', 0.0),
        "dynamics_broken_rate": dynamics_summary.get('broken_rate', 0.0),
        "dynamics_success_rate": dynamics_summary.get('success_rate', 0.0),
        "dynamics_avg_base_connected_ratio": dynamics_summary.get('avg_base_connected_ratio', 0.0),
        "dynamics_t_emerge_missing_rate": dynamics_summary.get('t_emerge_missing_rate', 0.0),
        "dynamics_t_lockin_missing_rate": dynamics_summary.get('t_lockin_missing_rate', 0.0),
    }
    
    # Append final metadata
    append_metadata(final_metadata, exp_output_dir, console)
    
    console.print("\n[bold green]Evaluation completed![/bold green]")
    console.print(f"Results saved to: {exp_output_dir}")
    console.print(f"Total evaluation time: {fmt_secs(total_secs)}")


if __name__ == "__main__":
    main()
