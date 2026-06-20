#!/usr/bin/env python3
"""
評估 16x16x16 Voxel Diffusion 模型

分層記錄評估數據：
- 層次 A：每個樣本的最終狀態指標（寫入 ``simple_label*.csv``，供後續分析）
- 層次 B：整批樣本的統計摘要（寫入 ``simple_label_summary.csv`` 與 metadata；不再產生獨立的 batch 摘要 CSV）
- 層次 C：前幾個樣本的完整動力學軌跡（寫入 ``dynamics_label_trace.csv``，並用於畫圖）

輸出文件：
- simple_label.csv / simple_label_summary.csv: 與 generate 腳本 ``sample_labels*.csv`` 相同欄位與格式
  （由 ``utils.export_csv`` 寫入；``source_name`` 相對於 ``--out_dir``，對應 ``simple_npz/`` 或 ``simple_projections/``）
- dynamics_label.csv / dynamics_label_summary.csv: 與 ``simple_label*.csv`` 相同欄位與格式（層次 C 最終態）；
  ``source_name`` 相對於 ``--out_dir``，對應 ``dynamics_npz/`` 或 ``dynamics_projections/``
- dynamics_label_trace.csv: 動力學軌跡每點一列，欄位為 label 格式（前綴 ``sample_idx, step_idx, t``，
  其餘與 ``sample_labels.csv`` 一致；``source_name`` 對應 ``dynamics_track_npz/`` 或 ``dynamics_track_projections/``）
- dynamics_divergence_plot.png: 可視化圖表
- plot_data_csv/: 與 dynamics_divergence_plot 四個子圖對應之原始數據 CSV（各一檔）
  - dynamics_divergence_subplot_01_mass.csv
  - dynamics_divergence_subplot_02_failure_rate.csv
  - dynamics_divergence_subplot_03_base_connected_ratio.csv
  - dynamics_divergence_subplot_04_largest_log_ratio.csv
- dynamics_timing_distribution_plot.png: t_emerge / t_lockin 分布圖
- plot_data_csv/dynamics_timing_distribution_subplot_01_t_emerge.csv（每樣本）
- plot_data_csv/dynamics_timing_distribution_subplot_01_t_emerge_histogram.csv（bins=30）
- plot_data_csv/dynamics_timing_distribution_subplot_02_t_lockin.csv（每樣本）
- plot_data_csv/dynamics_timing_distribution_subplot_02_t_lockin_histogram.csv（bins=30）
- dynamics_xt/: x_t 三視圖目錄（使用 --save_xt_projections 時生成）

（若啟用 --save_npz / 軌跡 npz：每檔僅含陣列鍵 ``voxel``；指標見對應 CSV。）

輸出目錄為 ``--out_dir``。圖表／投影圖的實驗標題（metadata 的 ``exp_name``）為 ``out_dir`` 路徑最後一層目錄名；若無法取得（例如根目錄 ``/``），則使用 ``eval_YYYYMMDD_HHMMSS``。

使用方式:
  python eval_diffusion_model.py --checkpoint <path/to/checkpoint.pt> --out_dir ./runs/my_eval --n_samples 100
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
import traceback

try:
    from scipy.ndimage import label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not installed. Component count will be disabled.")

import matplotlib
matplotlib.use("Agg")  # Headless backend for saving without display
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Add parent directory to path to import from training script
import sys
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import model and functions from training script
try:
    from train_unet_diffusion import (
        UNet3DDiffusion,
        BetaSchedule,
        centered_to_onehot,
    )
    from diffusion_sampling import (
        sample_guided_voxels,
        sample_ug_guided_voxels,
        sample_voxels,
    )
except ImportError as e:
    print(f"[ERROR] Failed to import from training script: {e}")
    print("Make sure unet_diffusion_16_voxel.py is in the same directory.")
    sys.exit(1)

from utils.export_csv import (
    sample_label_row_from_metrics,
    write_dynamics_label_trace_csv,
    write_sample_labels_csv,
    write_sample_labels_summary_csv,
)
from utils.voxel_label_projections import save_labels_and_projections
from utils.voxel_npz_io import save_voxel_npz
from utils.voxel_sample_metrics import (
    CAT_NEG_EASY,
    CAT_NEG_FLOAT,
    CAT_NEG_HARD,
    CAT_POSITIVE,
    compute_sample_metrics,
)


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


def is_success_sample(metrics: Dict, atol: float = 1e-6) -> bool:
    """
    Success criterion: Largest_Log_Ratio == 1 (within tolerance).
    Any value != 1 (or invalid <0) is treated as failure.
    """
    ratio = float(metrics.get("Largest_Log_Ratio", -1.0))
    return ratio >= 0.0 and bool(np.isclose(ratio, 1.0, atol=atol))


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


def load_scorer(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load TopologyScorer3D checkpoint used for guided sampling."""
    console = Console()
    console.print(f"[cyan]Loading scorer checkpoint: {checkpoint_path}[/cyan]")
    scorer_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    try:
        from train_scorer import TopologyScorer3D
    except ImportError as e:
        raise ImportError(
            "Failed to import `TopologyScorer3D` from `train_scorer.py`."
        ) from e

    scorer = TopologyScorer3D().to(device)
    if isinstance(scorer_ckpt, dict) and "model_state_dict" in scorer_ckpt:
        state_dict = scorer_ckpt["model_state_dict"]
    else:
        state_dict = scorer_ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Unsupported scorer checkpoint format. Expected dict-like state_dict, "
            f"but got type={type(scorer_ckpt).__name__}"
        )

    scorer.load_state_dict(state_dict, strict=True)
    scorer.eval()
    console.print(f"[green]✓[/green] Scorer loaded: {checkpoint_path}")
    if isinstance(scorer_ckpt, dict) and "epoch" in scorer_ckpt:
        console.print(
            f"[green]✓[/green] Scorer checkpoint epoch: {scorer_ckpt.get('epoch', 'unknown')}"
        )
    return scorer


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
    scorer_model: Optional[nn.Module] = None,
    guidance_scale: float = 50.0,
    t_start: int = 900,
    t_end: int = 400,
    guidance_lambda_ratio: float = 10.0,
    guidance_mode: str = "xt",
    ug_inject: str = "eps",
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
                if scorer_model is not None and guidance_mode == "ug":
                    x_0_batch = sample_ug_guided_voxels(
                        denoiser_model=model,
                        scorer_model=scorer_model,
                        betas=betas,
                        shape=(current_batch_size, 3, 16, 16, 16),
                        device=device,
                        guidance_scale=guidance_scale,
                        lambda_ratio=guidance_lambda_ratio,
                        t_start=t_start,
                        t_end=t_end,
                        inject=ug_inject,
                        n_steps=n_steps,
                        use_amp=use_amp,
                        track_every=None,
                        track_callback=None,
                    )
                elif scorer_model is not None:
                    x_0_batch = sample_guided_voxels(
                        denoiser_model=model,
                        scorer_model=scorer_model,
                        betas=betas,
                        shape=(current_batch_size, 3, 16, 16, 16),
                        device=device,
                        guidance_scale=guidance_scale,
                        lambda_ratio=guidance_lambda_ratio,
                        t_start=t_start,
                        t_end=t_end,
                        n_steps=n_steps,
                        use_amp=use_amp,
                        track_every=None,
                        track_callback=None,
                    )
                else:
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
                
                # Save npz file if enabled (single array key ``voxel``; metrics stay in CSV)
                if npz_dir:
                    try:
                        npz_path = os.path.join(npz_dir, f"simple_sample_{sample_counter+1:03d}.npz")
                        save_voxel_npz(npz_path, labels)
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
    save_xt_projections: bool = False,
    save_npz: bool = False,
    log_mask_threshold: Optional[float] = None,
    scorer_model: Optional[nn.Module] = None,
    guidance_scale: float = 50.0,
    t_start: int = 900,
    t_end: int = 400,
    guidance_lambda_ratio: float = 10.0,
    guidance_mode: str = "xt",
    ug_inject: str = "eps",
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
        save_xt_projections: whether to save 3-view projections of x_t (noisy state) at each tracking step (default: False)
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
    
    # Create x_t projections directory if save_xt_projections is enabled
    xt_projections_dir = None
    if output_dir and save_xt_projections:
        xt_projections_dir = os.path.abspath(os.path.join(output_dir, "dynamics_xt"))
        os.makedirs(xt_projections_dir, exist_ok=True)
        if console:
            console.print(f"[dim]x_t projections will be saved to: {xt_projections_dir}[/dim]")
    
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
    xt_projection_count = 0
    
    def make_track_callback(batch_offset: int):
        """Create a track callback function with batch offset."""
        def track_callback(sample_idx, step_idx, t_int, x_current, x0_hat):
            """Callback to track metrics at each step."""
            nonlocal track_projection_count, track_npz_count, xt_projection_count
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
            
            # Save npz at this tracking step (single array key ``voxel``)
            if track_npz_dir:
                try:
                    npz_path = os.path.join(
                        track_npz_dir,
                        f"dynamics_sample_{global_sample_idx+1:03d}_step_{step_idx:04d}_t_{t_int:04d}.npz"
                    )
                    save_voxel_npz(npz_path, labels)
                    track_npz_count += 1
                except Exception as e:
                    if console:
                        console.print(f"[yellow]⚠[/yellow] Failed to save track npz for sample {global_sample_idx+1}, step {step_idx}: {e}")
            
            # Save x_t (noisy state) 3-view projection if enabled
            if xt_projections_dir is not None:
                try:
                    xt_projection_path = os.path.join(
                        xt_projections_dir,
                        f"dynamics_sample_{global_sample_idx+1:03d}_step_{step_idx:04d}_t_{t_int:04d}_xt.png"
                    )
                    title_suffix = f" Sample {global_sample_idx+1}, Step {step_idx}, t={t_int} (x_t)"
                    _save_xt_projections_impl(
                        x_current,
                        xt_projection_path,
                        title_suffix=title_suffix,
                        exp_name=exp_name if exp_name else None
                    )
                    xt_projection_count += 1
                except Exception as e:
                    if console:
                        console.print(f"[yellow]⚠[/yellow] Failed to save x_t projection for sample {global_sample_idx+1}, step {step_idx}: {e}")
                        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
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
                if scorer_model is not None and guidance_mode == "ug":
                    x_0_batch = sample_ug_guided_voxels(
                        denoiser_model=model,
                        scorer_model=scorer_model,
                        betas=betas,
                        shape=(current_batch_size, 3, 16, 16, 16),
                        device=device,
                        guidance_scale=guidance_scale,
                        lambda_ratio=guidance_lambda_ratio,
                        t_start=t_start,
                        t_end=t_end,
                        inject=ug_inject,
                        n_steps=n_steps,
                        use_amp=use_amp,
                        track_every=track_every,
                        track_callback=track_callback,
                    )
                elif scorer_model is not None:
                    x_0_batch = sample_guided_voxels(
                        denoiser_model=model,
                        scorer_model=scorer_model,
                        betas=betas,
                        shape=(current_batch_size, 3, 16, 16, 16),
                        device=device,
                        guidance_scale=guidance_scale,
                        lambda_ratio=guidance_lambda_ratio,
                        t_start=t_start,
                        t_end=t_end,
                        n_steps=n_steps,
                        use_amp=use_amp,
                        track_every=track_every,
                        track_callback=track_callback,
                    )
                else:
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
            
            # Calculate t_emerge and t_lockin from trace_data
            sample_trace = trace_by_sample.get(sample_idx, [])
            t_emerge, t_lockin = compute_t_emerge_and_t_lockin(sample_trace)
            metrics['t_emerge'] = t_emerge if t_emerge is not None else -1
            metrics['t_lockin'] = t_lockin if t_lockin is not None else -1
            
            final_metrics.append(metrics)
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to compute metrics for dynamics sample {sample_idx+1}: {e}")
    
    # Save npz files for final states (single array key ``voxel``)
    if npz_dir and final_samples:
        for sample_idx, labels in final_samples.items():
            try:
                npz_path = os.path.join(npz_dir, f"dynamics_sample_{sample_idx+1:03d}.npz")
                save_voxel_npz(npz_path, labels)
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
    if xt_projections_dir:
        console.print(f"[green]✓[/green] Saved {xt_projection_count} x_t projections to: {xt_projections_dir}")
    return trace_data, elapsed_time, final_metrics


def _save_xt_projections_impl(x_t: torch.Tensor, out_png: str, title_suffix: str = "", exp_name: str = ""):
    """
    Save 3-view projections of x_t (noisy diffusion state).
    Converts centered x_t to pseudo-labels via argmax for visualization.
    Priority: wood (1) > leaf (2) > air (0)
    
    Args:
        x_t: [3, 16, 16, 16] tensor in centered form (values in [-1, 1])
        out_png: output png file path
        title_suffix: optional suffix to add to the figure title
        exp_name: experiment name to display as main title
    """
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    # Ensure tensor is float32 and contiguous (in case of AMP/GPU)
    x_t = x_t.detach().float().clone()
    # Convert centered to [0,1] then argmax to get pseudo-labels (noisy but viewable)
    x_onehot = centered_to_onehot(x_t.unsqueeze(0))[0]  # [3, 16, 16, 16] in [0,1]
    probs = F.softmax(x_onehot, dim=0)
    labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)  # [16, 16, 16]
    save_labels_and_projections(labels, out_png, title_suffix=title_suffix, exp_name=exp_name)


def save_simple_label_outputs(
    batch_metrics: List[Dict],
    output_dir: str,
    *,
    save_npz: bool,
    save_projections: bool,
    run_seed: Optional[int],
    console: Optional[Console] = None,
) -> None:
    """
    寫入 simple_label.csv / simple_label_summary.csv（欄位與 generate 的 sample_labels 一致）。
    """
    if console is None:
        console = Console()

    rows: List[Dict] = []
    for i, m in enumerate(batch_metrics):
        row = sample_label_row_from_metrics(m, sample_id=i + 1, run_seed=run_seed)
        if save_npz:
            row["source_name"] = f"simple_npz/simple_sample_{i + 1:03d}.npz"
        elif save_projections:
            row["source_name"] = f"simple_projections/simple_sample_{i + 1:03d}.png"
        else:
            row["source_name"] = ""
        rows.append(row)

    labels_path = os.path.join(output_dir, "simple_label.csv")
    summary_path = os.path.join(output_dir, "simple_label_summary.csv")
    write_sample_labels_csv(rows, labels_path)
    write_sample_labels_summary_csv(rows, summary_path)
    console.print(f"[green]✓[/green] Saved simple labels: {labels_path} ({len(rows)} samples)")
    console.print(f"[green]✓[/green] Saved simple label summary: {summary_path}")


def save_dynamics_label_outputs(
    dynamics_metrics: List[Dict],
    output_dir: str,
    *,
    save_npz: bool,
    save_projections: bool,
    run_seed: Optional[int],
    console: Optional[Console] = None,
) -> None:
    """
    寫入 dynamics_label.csv / dynamics_label_summary.csv（欄位與 simple_label / generate sample_labels 一致）。
    每列 ``id``（例如 ``1``）與 metrics 內 ``ID``（例如 ``001``）對齊檔名 ``dynamics_sample_001.*``。
    """
    if console is None:
        console = Console()

    rows: List[Dict] = []
    for m in dynamics_metrics:
        try:
            sid = int(str(m.get("ID", "0")).strip())
        except ValueError:
            sid = len(rows) + 1
        row = sample_label_row_from_metrics(m, sample_id=sid, run_seed=run_seed)
        if save_npz:
            row["source_name"] = f"dynamics_npz/dynamics_sample_{sid:03d}.npz"
        elif save_projections:
            row["source_name"] = f"dynamics_projections/dynamics_sample_{sid:03d}.png"
        else:
            row["source_name"] = ""
        rows.append(row)

    labels_path = os.path.join(output_dir, "dynamics_label.csv")
    summary_path = os.path.join(output_dir, "dynamics_label_summary.csv")
    write_sample_labels_csv(rows, labels_path)
    write_sample_labels_summary_csv(rows, summary_path)
    console.print(f"[green]✓[/green] Saved dynamics labels: {labels_path} ({len(rows)} samples)")
    console.print(f"[green]✓[/green] Saved dynamics label summary: {summary_path}")


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
        return {
            "n_samples": 0,
            "n_cat_positive": 0,
            "n_cat_neg_float": 0,
            "n_cat_neg_easy": 0,
            "n_cat_neg_hard": 0,
        }
    
    # Success/failure rate by largest_log_ratio==1.
    success_flags = [1.0 if is_success_sample(m) else 0.0 for m in batch_metrics]
    success_rate = np.mean(success_flags) * 100.0
    failure_rate = 100.0 - success_rate
    
    # Disconnected wood rate (any disconnected wood components)
    broken_rate = np.mean([1.0 if m['Is_Broken'] else 0.0 for m in batch_metrics]) * 100.0
    
    # Average mass
    avg_mass = np.mean([m['Mass'] for m in batch_metrics])
    std_mass = np.std([m['Mass'] for m in batch_metrics])
    
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

    cats = [m.get("Scorer_Category", "") for m in batch_metrics]
    n_cat_positive = sum(1 for x in cats if x == CAT_POSITIVE)
    n_cat_neg_float = sum(1 for x in cats if x == CAT_NEG_FLOAT)
    n_cat_neg_easy = sum(1 for x in cats if x == CAT_NEG_EASY)
    n_cat_neg_hard = sum(1 for x in cats if x == CAT_NEG_HARD)
    
    return {
        'n_samples': n,
        # Keep breakage_rate key for backward compatibility; now it means failure rate by LLR!=1.
        'breakage_rate': failure_rate,
        'failure_rate': failure_rate,
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
        'n_cat_positive': n_cat_positive,
        'n_cat_neg_float': n_cat_neg_float,
        'n_cat_neg_easy': n_cat_neg_easy,
        'n_cat_neg_hard': n_cat_neg_hard,
    }


def compute_dynamics_summary_statistics(dynamics_metrics: List[Dict]) -> Dict:
    """
    計算 dynamics 最終狀態的統計摘要（與 ``dynamics_label.csv`` 同一批 metrics dict）。
    
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


def _csv_float_cell(x: object) -> object:
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(xf):
        return ""
    return x


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
    plot_data_dir = os.path.join(output_dir, "plot_data_csv")
    os.makedirs(plot_data_dir, exist_ok=True)
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
    
    # Determine final success state by Largest_Log_Ratio==1 for each sample
    # This is used for averaging in Plot 3 and Plot 4
    sample_final_states = {}
    for sid in samples.keys():
        data = samples[sid]
        # Get final state (lowest t value, which is the final state)
        if data:
            final_data = min(data, key=lambda x: x['t'])
            sample_final_states[sid] = is_success_sample(final_data)
    
    # If n_dynamics_samples > 10, randomly select 10 samples to plot
    samples_to_plot = sorted(samples.keys())
    if n_dynamics_samples > 10:
        samples_to_plot = sorted(random.sample(list(samples.keys()), min(10, len(samples.keys()))))
    
    # Plot 1: Mass over time
    ax1 = axes[0, 0]
    mass_rows: List[Dict[str, object]] = []
    for sid in samples_to_plot:
        data = samples[sid]
        ts = [r['t'] for r in data]
        masses = [r['Mass'] for r in data]
        for t_i, m_i in zip(ts, masses):
            mass_rows.append({"timestep": t_i, "sample_idx": sid, "mass": m_i})
        ax1.plot(ts, masses, label=f'Sample {sid+1}', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Timestep (t)')
    ax1.set_ylabel('Mass (non-air voxels)')
    ax1.set_title('Mass Evolution During Sampling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # t decreases from 1000 to 0

    mass_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_01_mass.csv")
    with open(mass_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestep", "sample_idx", "mass"])
        w.writeheader()
        for row in mass_rows:
            w.writerow(row)
    
    # Plot 2: Failure rate (Largest_Log_Ratio != 1) over time
    ax2 = axes[0, 1]
    # Aggregate failure rate across ALL dynamics samples per timestep
    total_counts_by_t = {}
    failed_counts_by_t = {}
    total_counts_failed_group_by_t = {}
    failed_counts_failed_group_by_t = {}
    total_counts_success_group_by_t = {}
    failed_counts_success_group_by_t = {}
    for sid, data in samples.items():
        is_final_success = sample_final_states.get(sid, False)
        for r in data:
            t_val = r['t']
            total_counts_by_t[t_val] = total_counts_by_t.get(t_val, 0) + 1
            is_failed_now = not is_success_sample(r)
            if is_failed_now:
                failed_counts_by_t[t_val] = failed_counts_by_t.get(t_val, 0) + 1
            else:
                failed_counts_by_t.setdefault(t_val, failed_counts_by_t.get(t_val, 0))
            
            # Split by final state group (failed vs successful)
            if not is_final_success:
                total_counts_failed_group_by_t[t_val] = total_counts_failed_group_by_t.get(t_val, 0) + 1
                if is_failed_now:
                    failed_counts_failed_group_by_t[t_val] = failed_counts_failed_group_by_t.get(t_val, 0) + 1
                else:
                    failed_counts_failed_group_by_t.setdefault(
                        t_val, failed_counts_failed_group_by_t.get(t_val, 0)
                    )
            else:
                total_counts_success_group_by_t[t_val] = total_counts_success_group_by_t.get(t_val, 0) + 1
                if is_failed_now:
                    failed_counts_success_group_by_t[t_val] = failed_counts_success_group_by_t.get(t_val, 0) + 1
                else:
                    failed_counts_success_group_by_t.setdefault(
                        t_val, failed_counts_success_group_by_t.get(t_val, 0)
                    )
    if total_counts_by_t:
        sorted_ts = sorted(total_counts_by_t.keys(), reverse=True)
        rates = []
        rates_failed_group = []
        rates_success_group = []
        for t_val in sorted_ts:
            total = total_counts_by_t[t_val]
            failed = failed_counts_by_t.get(t_val, 0)
            rate = failed / total if total > 0 else 0.0
            rates.append(rate)
            
            total_f = total_counts_failed_group_by_t.get(t_val, 0)
            failed_f = failed_counts_failed_group_by_t.get(t_val, 0)
            rates_failed_group.append((failed_f / total_f) if total_f > 0 else np.nan)
            
            total_s = total_counts_success_group_by_t.get(t_val, 0)
            failed_s = failed_counts_success_group_by_t.get(t_val, 0)
            rates_success_group.append((failed_s / total_s) if total_s > 0 else np.nan)
        
        ax2.plot(sorted_ts, rates, color='purple', linewidth=2.2, label='Failure Rate (All)')
        ax2.plot(sorted_ts, rates_failed_group, color='red', linewidth=2.0, linestyle='--', label='Failure Rate (Failed Group)')
        ax2.plot(sorted_ts, rates_success_group, color='green', linewidth=2.0, linestyle='--', label='Failure Rate (Success Group)')

        fail_rows: List[Dict[str, object]] = []
        for t_val, rate, rf, rs in zip(
            sorted_ts, rates, rates_failed_group, rates_success_group
        ):
            total = total_counts_by_t[t_val]
            failed = failed_counts_by_t.get(t_val, 0)
            total_f = total_counts_failed_group_by_t.get(t_val, 0)
            failed_f = failed_counts_failed_group_by_t.get(t_val, 0)
            total_s = total_counts_success_group_by_t.get(t_val, 0)
            failed_s = failed_counts_success_group_by_t.get(t_val, 0)
            fail_rows.append(
                {
                    "timestep": t_val,
                    "n_total_tracked": total,
                    "n_failed_llr_neq_1": failed,
                    "failure_rate_all": _csv_float_cell(rate),
                    "n_total_final_failed_group": total_f,
                    "n_failed_llr_neq_1_final_failed_group": failed_f,
                    "failure_rate_final_failed_group": _csv_float_cell(rf),
                    "n_total_final_success_group": total_s,
                    "n_failed_llr_neq_1_final_success_group": failed_s,
                    "failure_rate_final_success_group": _csv_float_cell(rs),
                }
            )
        fail_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_02_failure_rate.csv")
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "timestep",
                    "n_total_tracked",
                    "n_failed_llr_neq_1",
                    "failure_rate_all",
                    "n_total_final_failed_group",
                    "n_failed_llr_neq_1_final_failed_group",
                    "failure_rate_final_failed_group",
                    "n_total_final_success_group",
                    "n_failed_llr_neq_1_final_success_group",
                    "failure_rate_final_success_group",
                ],
            )
            w.writeheader()
            w.writerows(fail_rows)
    else:
        fail_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_02_failure_rate.csv")
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "timestep",
                    "n_total_tracked",
                    "n_failed_llr_neq_1",
                    "failure_rate_all",
                    "n_total_final_failed_group",
                    "n_failed_llr_neq_1_final_failed_group",
                    "failure_rate_final_failed_group",
                    "n_total_final_success_group",
                    "n_failed_llr_neq_1_final_success_group",
                    "failure_rate_final_success_group",
                ],
            )
            w.writeheader()

    ax2.set_xlabel('Timestep (t)')
    ax2.set_ylabel('Failure Rate (Largest_Log_Ratio != 1)')
    ax2.set_title('Failure Rate During Sampling (All Dynamics Samples)')
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Base Connected Ratio over time (from t=95% of T to 0)
    ax3 = axes[1, 0]
    
    # Collect all data for averaging (use all samples)
    all_ratios_by_t = {}  # {t: [ratios]}
    failed_ratios_by_t = {}  # {t: [ratios]}
    success_ratios_by_t = {}  # {t: [ratios]}
    
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
            is_success = sample_final_states.get(sid, False)
            for t, ratio in zip(ts, base_connected_ratios):
                if t not in all_ratios_by_t:
                    all_ratios_by_t[t] = []
                    failed_ratios_by_t[t] = []
                    success_ratios_by_t[t] = []
                all_ratios_by_t[t].append(ratio)
                if not is_success:
                    failed_ratios_by_t[t].append(ratio)
                else:
                    success_ratios_by_t[t].append(ratio)
    
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
        avg_failed = [np.mean(failed_ratios_by_t[t]) if failed_ratios_by_t[t] else np.nan for t in sorted_ts]
        avg_success = [np.mean(success_ratios_by_t[t]) if success_ratios_by_t[t] else np.nan for t in sorted_ts]
        
        ax3.plot(sorted_ts, avg_all, label='Avg (All)', color='black', linewidth=2.5, linestyle='-', alpha=0.9)
        ax3.plot(sorted_ts, avg_failed, label='Avg (Failed)', color='red', linewidth=2.5, linestyle='--', alpha=0.9)
        ax3.plot(sorted_ts, avg_success, label='Avg (Success)', color='green', linewidth=2.5, linestyle='--', alpha=0.9)

        sample_bcr_by_t: Dict[int, Dict[int, float]] = {}
        for sid in samples_to_plot:
            fd = [r for r in samples[sid] if r["t"] <= t_threshold]
            d: Dict[int, float] = {}
            for r in fd:
                total_log_size = r.get("Total_Log_Size", r.get("Log_Size", 0))
                ratio = (
                    float(r["Base_Connected_Size"]) / float(total_log_size)
                    if total_log_size > 0
                    else 0.0
                )
                d[int(r["t"])] = ratio
            sample_bcr_by_t[sid] = d

        bcr_fieldnames = (
            ["timestep", "t_plot_threshold"]
            + [f"sample_{sid}_base_connected_ratio" for sid in samples_to_plot]
            + ["avg_all", "avg_failed", "avg_success"]
        )
        bcr_rows: List[Dict[str, object]] = []
        for t_i, a_all, a_f, a_s in zip(sorted_ts, avg_all, avg_failed, avg_success):
            row: Dict[str, object] = {
                "timestep": t_i,
                "t_plot_threshold": t_threshold,
                "avg_all": a_all,
                "avg_failed": _csv_float_cell(a_f),
                "avg_success": _csv_float_cell(a_s),
            }
            for sid in samples_to_plot:
                v = sample_bcr_by_t[sid].get(t_i)
                row[f"sample_{sid}_base_connected_ratio"] = "" if v is None else v
            bcr_rows.append(row)
        bcr_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_03_base_connected_ratio.csv")
        with open(bcr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=bcr_fieldnames)
            w.writeheader()
            w.writerows(bcr_rows)
    else:
        bcr_fieldnames = (
            ["timestep", "t_plot_threshold"]
            + [f"sample_{sid}_base_connected_ratio" for sid in samples_to_plot]
            + ["avg_all", "avg_failed", "avg_success"]
        )
        bcr_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_03_base_connected_ratio.csv")
        with open(bcr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=bcr_fieldnames)
            w.writeheader()
    
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
    failed_ratios_by_t = {}  # {t: [ratios]}
    success_ratios_by_t = {}  # {t: [ratios]}
    
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
                is_success = sample_final_states.get(sid, False)
                for t, ratio in zip(valid_ts, largest_log_ratios):
                    if t not in all_ratios_by_t:
                        all_ratios_by_t[t] = []
                        failed_ratios_by_t[t] = []
                        success_ratios_by_t[t] = []
                    all_ratios_by_t[t].append(ratio)
                    if not is_success:
                        failed_ratios_by_t[t].append(ratio)
                    else:
                        success_ratios_by_t[t].append(ratio)
    
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
        avg_failed = [np.mean(failed_ratios_by_t[t]) if failed_ratios_by_t[t] else np.nan for t in sorted_ts]
        avg_success = [np.mean(success_ratios_by_t[t]) if success_ratios_by_t[t] else np.nan for t in sorted_ts]
        
        ax4.plot(sorted_ts, avg_all, label='Avg (All)', color='black', linewidth=2.5, linestyle='-', alpha=0.9)
        ax4.plot(sorted_ts, avg_failed, label='Avg (Failed)', color='red', linewidth=2.5, linestyle='--', alpha=0.9)
        ax4.plot(sorted_ts, avg_success, label='Avg (Success)', color='green', linewidth=2.5, linestyle='--', alpha=0.9)

        sample_llr_by_t: Dict[int, Dict[int, float]] = {}
        for sid in samples_to_plot:
            fd = [r for r in samples[sid] if r["t"] <= t_threshold]
            d: Dict[int, float] = {}
            for r in fd:
                ratio = float(r.get("Largest_Log_Ratio", -1.0))
                if ratio >= 0.0:
                    d[int(r["t"])] = ratio
            sample_llr_by_t[sid] = d

        llr_fieldnames = (
            ["timestep", "t_plot_threshold"]
            + [f"sample_{sid}_largest_log_ratio" for sid in samples_to_plot]
            + ["avg_all", "avg_failed", "avg_success"]
        )
        llr_rows: List[Dict[str, object]] = []
        for t_i, a_all, a_f, a_s in zip(sorted_ts, avg_all, avg_failed, avg_success):
            row = {
                "timestep": t_i,
                "t_plot_threshold": t_threshold,
                "avg_all": a_all,
                "avg_failed": _csv_float_cell(a_f),
                "avg_success": _csv_float_cell(a_s),
            }
            for sid in samples_to_plot:
                v = sample_llr_by_t[sid].get(t_i)
                row[f"sample_{sid}_largest_log_ratio"] = "" if v is None else v
            llr_rows.append(row)
        llr_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_04_largest_log_ratio.csv")
        with open(llr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=llr_fieldnames)
            w.writeheader()
            w.writerows(llr_rows)
    else:
        llr_fieldnames = (
            ["timestep", "t_plot_threshold"]
            + [f"sample_{sid}_largest_log_ratio" for sid in samples_to_plot]
            + ["avg_all", "avg_failed", "avg_success"]
        )
        llr_csv = os.path.join(plot_data_dir, "dynamics_divergence_subplot_04_largest_log_ratio.csv")
        with open(llr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=llr_fieldnames)
            w.writeheader()
    
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
    console.print(f"[green]✓[/green] Divergence plot data CSV: [cyan]{plot_data_dir}/[/cyan]")


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
    plot_data_dir = os.path.join(output_dir, "plot_data_csv")
    os.makedirs(plot_data_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "dynamics_timing_distribution_plot.png")
    
    # Group by sample_idx
    samples = {}
    for row in trace_data:
        sid = row['sample_idx']
        if sid not in samples:
            samples[sid] = []
        samples[sid].append(row)
    
    # Determine final success state by Largest_Log_Ratio==1 for each sample
    sample_final_states = {}
    for sid in samples.keys():
        data = samples[sid]
        if data:
            final_data = min(data, key=lambda x: x['t'])
            sample_final_states[sid] = is_success_sample(final_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # Histogram 1: t_emerge
    ax1 = axes[0]
    t_emerge_failed = []
    t_emerge_success = []
    emerge_sample_rows: List[Dict[str, object]] = []
    for sid in sorted(samples.keys()):
        sample_trace = samples[sid]
        t_emerge, _ = compute_t_emerge_and_t_lockin(sample_trace)
        is_succ = bool(sample_final_states.get(sid, False))
        grp = "success" if is_succ else "failed"
        emerge_sample_rows.append(
            {
                "sample_idx": sid,
                "final_group": grp,
                "t_emerge": int(t_emerge) if t_emerge is not None else "",
                "found": 1 if t_emerge is not None else 0,
            }
        )
        if t_emerge is not None:
            if is_succ:
                t_emerge_success.append(t_emerge)
            else:
                t_emerge_failed.append(t_emerge)

    emerge_csv = os.path.join(plot_data_dir, "dynamics_timing_distribution_subplot_01_t_emerge.csv")
    with open(emerge_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "final_group", "t_emerge", "found"],
        )
        w.writeheader()
        w.writerows(emerge_sample_rows)

    def _write_timing_histogram_csv(
        path: str,
        vals_failed: List[int],
        vals_success: List[int],
        n_bins: int = 30,
    ) -> None:
        vf = np.asarray(vals_failed, dtype=np.float64)
        vs = np.asarray(vals_success, dtype=np.float64)
        parts = [a for a in (vf, vs) if a.size > 0]
        if not parts:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "bin_index",
                        "bin_left",
                        "bin_right",
                        "count_failed",
                        "count_success",
                    ],
                )
                w.writeheader()
            return
        combined = np.concatenate(parts)
        bin_edges = np.histogram_bin_edges(combined, bins=n_bins)
        nf, _ = np.histogram(vf, bins=bin_edges) if vf.size else np.zeros(len(bin_edges) - 1, dtype=np.int64)
        ns, _ = np.histogram(vs, bins=bin_edges) if vs.size else np.zeros(len(bin_edges) - 1, dtype=np.int64)
        rows_h: List[Dict[str, object]] = []
        for i in range(len(bin_edges) - 1):
            rows_h.append(
                {
                    "bin_index": i,
                    "bin_left": float(bin_edges[i]),
                    "bin_right": float(bin_edges[i + 1]),
                    "count_failed": int(nf[i]),
                    "count_success": int(ns[i]),
                }
            )
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "bin_index",
                    "bin_left",
                    "bin_right",
                    "count_failed",
                    "count_success",
                ],
            )
            w.writeheader()
            w.writerows(rows_h)

    emerge_hist_csv = os.path.join(
        plot_data_dir, "dynamics_timing_distribution_subplot_01_t_emerge_histogram.csv"
    )
    _write_timing_histogram_csv(emerge_hist_csv, t_emerge_failed, t_emerge_success, n_bins=30)
    
    if t_emerge_failed or t_emerge_success:
        data_to_plot = []
        labels = []
        colors = []
        if t_emerge_failed:
            data_to_plot.append(t_emerge_failed)
            labels.append('Failed')
            colors.append('red')
        if t_emerge_success:
            data_to_plot.append(t_emerge_success)
            labels.append('Success')
            colors.append('green')
        ax1.hist(data_to_plot, bins=30, edgecolor='black', alpha=0.65, label=labels, color=colors)
        if t_emerge_failed:
            mean_f = float(np.mean(t_emerge_failed))
            ax1.axvline(mean_f, color='red', linestyle='--', linewidth=2.0, label='Failed mean')
        if t_emerge_success:
            mean_s = float(np.mean(t_emerge_success))
            ax1.axvline(mean_s, color='green', linestyle='--', linewidth=2.0, label='Success mean')
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
    t_lockin_failed = []
    t_lockin_success = []
    lockin_sample_rows: List[Dict[str, object]] = []
    for sid in sorted(samples.keys()):
        sample_trace = samples[sid]
        _, t_lockin = compute_t_emerge_and_t_lockin(sample_trace)
        is_succ = bool(sample_final_states.get(sid, False))
        grp = "success" if is_succ else "failed"
        lockin_sample_rows.append(
            {
                "sample_idx": sid,
                "final_group": grp,
                "t_lockin": int(t_lockin) if t_lockin is not None else "",
                "found": 1 if t_lockin is not None else 0,
            }
        )
        if t_lockin is not None:
            if is_succ:
                t_lockin_success.append(t_lockin)
            else:
                t_lockin_failed.append(t_lockin)

    lockin_csv = os.path.join(plot_data_dir, "dynamics_timing_distribution_subplot_02_t_lockin.csv")
    with open(lockin_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "final_group", "t_lockin", "found"],
        )
        w.writeheader()
        w.writerows(lockin_sample_rows)

    lockin_hist_csv = os.path.join(
        plot_data_dir, "dynamics_timing_distribution_subplot_02_t_lockin_histogram.csv"
    )
    _write_timing_histogram_csv(lockin_hist_csv, t_lockin_failed, t_lockin_success, n_bins=30)
    
    if t_lockin_failed or t_lockin_success:
        data_to_plot = []
        labels = []
        colors = []
        if t_lockin_failed:
            data_to_plot.append(t_lockin_failed)
            labels.append('Failed')
            colors.append('red')
        if t_lockin_success:
            data_to_plot.append(t_lockin_success)
            labels.append('Success')
            colors.append('green')
        ax2.hist(data_to_plot, bins=30, edgecolor='black', alpha=0.65, label=labels, color=colors)
        if t_lockin_failed:
            mean_f = float(np.mean(t_lockin_failed))
            ax2.axvline(mean_f, color='red', linestyle='--', linewidth=2.0, label='Failed mean')
        if t_lockin_success:
            mean_s = float(np.mean(t_lockin_success))
            ax2.axvline(mean_s, color='green', linestyle='--', linewidth=2.0, label='Success mean')
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
    console.print(f"[green]✓[/green] Timing distribution plot data CSV: [cyan]{plot_data_dir}/[/cyan]")


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
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for all evaluation artifacts (created if missing). "
        "Last path component is used as the experiment label for figures and metadata; "
        "if missing, eval_YYYYMMDD_HHMMSS is used.",
    )
    
    # Misc
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--no_projections", action="store_true", help="Disable saving 3-view projections (default: enabled)")
    parser.add_argument("--save_track_projections", action="store_true", help="Save 3-view projections at each tracking step during dynamics analysis (default: disabled)")
    parser.add_argument("--save_xt_projections", action="store_true", help="Save 3-view projections of x_t (noisy diffusion state) at each tracking step to dynamics_xt/ (default: disabled)")
    parser.add_argument("--save_npz", action="store_true", help="Save npz files for each sample (default: disabled)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None, use random seed)")
    parser.add_argument(
        "--log_mask_threshold",
        type=float,
        default=None,
        help="Use threshold-based log mask decoding (e.g., 0.4/0.5/0.6). If not set, use argmax decoding."
    )
    parser.add_argument(
        "--scorer_ckpt",
        type=str,
        default=None,
        help="If set, use scorer-guided sampling (requires TopologyScorer3D checkpoint).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=50.0,
        help="Scorer guidance strength w (only used with --scorer_ckpt).",
    )
    parser.add_argument(
        "--lambda_ratio",
        type=float,
        default=10.0,
        help="Scorer guidance energy weight: minimize break - lambda_ratio * ratio (only used with --scorer_ckpt).",
    )
    parser.add_argument(
        "--t_start",
        type=int,
        default=900,
        help="Guidance start timestep (inclusive).",
    )
    parser.add_argument(
        "--t_end",
        type=int,
        default=400,
        help="Guidance end timestep (inclusive).",
    )
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="xt",
        choices=["xt", "ug"],
        help=(
            "Guided-sampling mode (with --scorer_ckpt): "
            "'xt'=scorer sees noisy x_t (Path-A, needs a --train_on xt scorer); "
            "'ug'=Universal Guidance, scorer sees Tweedie x_hat_0 (needs a --train_on x0 scorer)."
        ),
    )
    parser.add_argument(
        "--ug_inject",
        type=str,
        default="eps",
        choices=["eps", "x"],
        help=(
            "How UG injects the guidance gradient (with --guidance_mode ug): "
            "'eps'=forward guidance; 'x'=direct x_t update."
        ),
    )
    
    args = parser.parse_args()
    
    # Set random seeds if seed is provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            # Set CuBLAS workspace config early to reduce non-determinism warnings on CUDA>=10.2.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.manual_seed_all(args.seed)
            # Make cuDNN behavior deterministic when possible.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # For scorer-guided sampling we run backward inside sampling steps; some CUDA ops
        # (e.g. max_pool3d backward) are inherently non-deterministic and flood warnings.
        # Keep strict deterministic mode for unguided eval; relax for guided eval.
        if args.scorer_ckpt:
            torch.use_deterministic_algorithms(False)
        else:
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

    scorer_model: Optional[nn.Module] = None
    guidance_lambda_ratio = args.lambda_ratio
    if args.scorer_ckpt:
        scorer_model = load_scorer(args.scorer_ckpt, device)
        mode_desc = (
            f"UG (x_hat_0 route, inject={args.ug_inject}; needs a --train_on x0 scorer)"
            if args.guidance_mode == "ug"
            else "Path-A (noisy x_t; needs a --train_on xt scorer)"
        )
        console.print(
            f"[green]✓[/green] Guided sampling enabled [{mode_desc}]: scale={args.guidance_scale} "
            f"window t={args.t_start}..{args.t_end} (lambda_ratio={guidance_lambda_ratio})"
        )
    
    # Mixed precision
    use_amp = (device.type == "cuda") and (not args.no_amp)
    console.print(f"[cyan]Mixed precision: {use_amp}[/cyan]\n")
    
    exp_output_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(exp_output_dir, exist_ok=True)
    out_dir_leaf = Path(exp_output_dir).name
    if not out_dir_leaf:
        out_dir_leaf = f"eval_{eval_start_time.strftime('%Y%m%d_%H%M%S')}"
    console.print(f"[cyan]Output directory: {exp_output_dir}[/cyan]")
    console.print(
        f"[cyan]Experiment label (figures / metadata): {out_dir_leaf}[/cyan] "
        f"([dim]from last component of --out_dir, or eval_* timestamp if empty[/dim])\n"
    )
    
    # Helper function for boolean to string
    def bool_to_str(v):
        return "TRUE" if v else "FALSE"
    
    # Get script name
    current_script = Path(__file__).name if "__file__" in globals() else "interactive_session"
    invocation_command = get_invocation_command()
    
    # Prepare initial metadata
    initial_metadata = {
        "exp_name": out_dir_leaf,
        "out_dir_leaf": out_dir_leaf,
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
        "save_xt_projections": bool_to_str(args.save_xt_projections),
        "save_npz": bool_to_str(args.save_npz),
        "label_decoding_mode": "argmax" if args.log_mask_threshold is None else "log_mask_threshold",
        "log_mask_threshold": args.log_mask_threshold if args.log_mask_threshold is not None else "None",
        "scorer_ckpt": args.scorer_ckpt if args.scorer_ckpt else "None",
        "guidance_scale": f"{args.guidance_scale:.10g}",
        "guidance_t_start": str(args.t_start),
        "guidance_t_end": str(args.t_end),
        "guidance_lambda_ratio": f"{guidance_lambda_ratio:.10g}",
        # Standardized output filenames
        "simple_label_csv": os.path.join(exp_output_dir, "simple_label.csv"),
        "simple_label_summary_csv": os.path.join(exp_output_dir, "simple_label_summary.csv"),
        "dynamics_label_csv": os.path.join(exp_output_dir, "dynamics_label.csv"),
        "dynamics_label_summary_csv": os.path.join(exp_output_dir, "dynamics_label_summary.csv"),
        "dynamics_label_trace_csv": os.path.join(exp_output_dir, "dynamics_label_trace.csv"),
        "dynamics_divergence_plot_png": os.path.join(exp_output_dir, "dynamics_divergence_plot.png"),
        "plot_data_csv_dir": os.path.join(exp_output_dir, "plot_data_csv"),
        "dynamics_timing_distribution_plot_png": os.path.join(exp_output_dir, "dynamics_timing_distribution_plot.png"),
        "timing_plot_data_t_emerge_csv": os.path.join(
            exp_output_dir, "plot_data_csv", "dynamics_timing_distribution_subplot_01_t_emerge.csv"
        ),
        "timing_plot_data_t_emerge_histogram_csv": os.path.join(
            exp_output_dir, "plot_data_csv", "dynamics_timing_distribution_subplot_01_t_emerge_histogram.csv"
        ),
        "timing_plot_data_t_lockin_csv": os.path.join(
            exp_output_dir, "plot_data_csv", "dynamics_timing_distribution_subplot_02_t_lockin.csv"
        ),
        "timing_plot_data_t_lockin_histogram_csv": os.path.join(
            exp_output_dir, "plot_data_csv", "dynamics_timing_distribution_subplot_02_t_lockin_histogram.csv"
        ),
        "dynamics_xt_dir": os.path.join(exp_output_dir, "dynamics_xt"),
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
        exp_name=out_dir_leaf,
        save_projections=not args.no_projections,
        save_npz=args.save_npz,
        log_mask_threshold=args.log_mask_threshold,
        scorer_model=scorer_model,
        guidance_scale=args.guidance_scale,
        t_start=args.t_start,
        t_end=args.t_end,
        guidance_lambda_ratio=guidance_lambda_ratio,
        guidance_mode=args.guidance_mode,
        ug_inject=args.ug_inject,
        console=console,
    )
    
    # Label-format batch outputs (Level A)
    save_simple_label_outputs(
        batch_metrics,
        exp_output_dir,
        save_npz=args.save_npz,
        save_projections=not args.no_projections,
        run_seed=args.seed,
        console=console,
    )

    # Batch summary for console + metadata (Level B)
    summary = compute_summary_statistics(batch_metrics)
    
    # Print summary to console
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Failure Rate (Largest_Log_Ratio != 1): {summary['failure_rate']:.2f}%")
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
        exp_name=out_dir_leaf,
        save_projections=not args.no_projections,
        save_track_projections=args.save_track_projections,
        save_xt_projections=args.save_xt_projections,
        save_npz=args.save_npz,
        log_mask_threshold=args.log_mask_threshold,
        scorer_model=scorer_model,
        guidance_scale=args.guidance_scale,
        t_start=args.t_start,
        t_end=args.t_end,
        guidance_lambda_ratio=guidance_lambda_ratio,
        guidance_mode=args.guidance_mode,
        ug_inject=args.ug_inject,
        console=console,
    )
    
    # Dynamics final states as label CSV (Level C)
    save_dynamics_label_outputs(
        dynamics_final_metrics,
        exp_output_dir,
        save_npz=args.save_npz,
        save_projections=not args.no_projections,
        run_seed=args.seed,
        console=console,
    )

    # Dynamics summary for metadata (Level C)
    dynamics_summary = compute_dynamics_summary_statistics(dynamics_final_metrics)

    label_trace_path = os.path.join(exp_output_dir, "dynamics_label_trace.csv")
    write_dynamics_label_trace_csv(
        trace_data,
        label_trace_path,
        run_seed=args.seed,
        save_npz=args.save_npz,
        save_track_projections=args.save_track_projections,
    )
    console.print(f"[green]✓[/green] Saved dynamics label trace: {label_trace_path}")

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
        "failure_rate": summary['failure_rate'],
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
        "dynamics_failure_rate": dynamics_summary.get('failure_rate', 0.0),
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
