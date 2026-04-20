#!/usr/bin/env python3
"""
單純從 16x16x16 Voxel Diffusion 模型生成樣本（不跑評估指標、動力學、CSV 摘要）。

輸出（預設）：
- projections/{positive,neg_float,neg_easy,neg_hard}/：各類別三視圖 PNG
- npz/{...}/：同上，每檔僅含陣列鍵 ``voxel``（--no_npz 可關閉）
- sample_labels.csv: 每個樣本的 id、seed、分類與 compute_sample_metrics 之完整指標（見下方欄位）
- sample_labels_summary.csv: 全體樣本指標加總與平均、標準差等（與
  ``generate_16_voxel_diffusion.py`` 相同，由 ``utils.export_csv`` 產生）
- 分類定義：
  - neg_float：base_connected_size==0（連地板都沒碰到）
  - positive：base_connected_size>0，且全樹只有 1 個連通塊（log_components==1；Absolute Connectivity）
  - neg_hard：base_connected_size>0，且 log_components==2
  - neg_easy：base_connected_size>0，且 log_components>2
- metadata.csv / metadata_flat.csv：重現用參數
- generate_16_voxel_diffusion_bucket_snapshot_YYYYMMDD_HHMMSS.py：執行當下本腳本完整備份

sample_labels.csv 欄位（與 ``generate_16_voxel_diffusion.py`` 相同，順序如下）：
- id, seed, category（positive | neg_float | neg_easy | neg_hard）
- is_main_trunk_broken(0/1), is_broken(0/1), largest_log_ratio（無效時 -1）
- mass, height
- base_connected_ratio, base_connected_size, log_size, leaf_size
- occupancy_non_air, occupancy_log, occupancy_leaf
- components_non_air, components_log, components_leaf
- source_name：相對於 ``--out_dir`` 的 POSIX 路徑，優先 ``npz/<category>/<stem>.npz``；若 ``--no_npz`` 則為 ``projections/<category>/<stem>.png``；兩者皆關則為空字串
（以上對應 utils.voxel_sample_metrics.compute_sample_metrics 回傳之指標，外加 artifact 路徑）

專案目錄即 --out_dir：所有輸出寫入該路徑（會自動建立）。
實驗名稱與輸出檔名前綴（PNG/NPZ 的 ``<prefix>_<id>``）一律為 ``out_dir`` 路徑的最後一層目錄名稱（例如 ``--out_dir ./runs/exp_001`` → ``exp_001``）。

使用方式:
  python generate_16_voxel_diffusion_bucket.py --checkpoint path/to/model.pt --out_dir ./my_project --n_pos 50 --n_float 50 --n_easy 50 --n_hard 50
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from train_unet_diffusion import (
        BetaSchedule,
        UNet3DDiffusion,
        centered_to_onehot,
        sample_voxels,
    )
except ImportError as e:
    print(f"[ERROR] Failed to import from unet_diffusion_16_voxel: {e}")
    sys.exit(1)

from utils.voxel_label_projections import save_labels_and_projections
from utils.voxel_npz_io import save_voxel_npz
from utils.export_csv import (
    compute_sample_label_row,
    write_sample_labels_csv,
    write_sample_labels_summary_csv,
)
from utils.voxel_sample_metrics import (
    ALL_SCORER_CATEGORIES,
    CAT_NEG_EASY,
    CAT_NEG_FLOAT,
    CAT_NEG_HARD,
    CAT_POSITIVE,
)


def decode_probs_to_labels(
    probs: torch.Tensor, log_mask_threshold: Optional[float] = None
) -> np.ndarray:
    """與 eval_16_voxel_diffusion.py 相同：機率 → 離散標籤。"""
    if log_mask_threshold is None:
        return probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

    labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    probs_np = probs.detach().cpu().numpy()
    log_mask = probs_np[1] >= float(log_mask_threshold)
    non_log_mask = ~log_mask
    if np.any(non_log_mask):
        air_or_leaf = np.argmax(probs_np[[0, 2]], axis=0)
        labels[non_log_mask] = np.where(air_or_leaf[non_log_mask] == 0, 0, 2)
    labels[log_mask] = 1
    return labels


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    console = Console()
    console.print(f"[cyan]Loading checkpoint: {checkpoint_path}[/cyan]")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    base_channels = args.get("base_channels", 64)
    time_dim = args.get("time_dim", 128)
    model = UNet3DDiffusion(in_ch=3, base=base_channels, time_dim=time_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    console.print(
        f"[green]✓[/green] Model loaded: base_channels={base_channels}, time_dim={time_dim}"
    )
    console.print(f"[green]✓[/green] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    return model, checkpoint


def get_invocation_command() -> str:
    if not sys.argv:
        return ""
    exe = Path(sys.executable).name if sys.executable else "python"
    if exe.startswith("python"):
        exe = "python"
    return shlex.join([exe, *sys.argv])


def save_metadata(metadata: Dict, output_dir: str, console: Console) -> None:
    os.makedirs(output_dir, exist_ok=True)
    kv_path = os.path.join(output_dir, "metadata.csv")
    with open(kv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in metadata.items():
            w.writerow([k, v])
    flat_path = os.path.join(output_dir, "metadata_flat.csv")
    with open(flat_path, "w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=list(metadata.keys()))
        dw.writeheader()
        dw.writerow(metadata)
    console.print(f"[green]✓[/green] metadata: [cyan]{kv_path}[/cyan]")


def fmt_secs(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def generate_samples(
    model: nn.Module,
    betas: BetaSchedule,
    device: torch.device,
    target_counts: Dict[str, int],
    out_dir: str,
    batch_size: int = 10,
    n_steps: Optional[int] = None,
    use_amp: bool = False,
    save_projections: bool = True,
    save_npz: bool = False,
    log_mask_threshold: Optional[float] = None,
    filename_prefix: str = "sample",
    sample_verbose: bool = False,
    run_seed: Optional[int] = None,
    console: Optional[Console] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    if console is None:
        console = Console()

    targets: Dict[str, int] = {c: int(target_counts.get(c, 0)) for c in ALL_SCORER_CATEGORIES}
    for c, v in targets.items():
        if v < 0:
            raise ValueError(f"target_counts[{c}] must be >= 0, got {v}")
    target_total = int(sum(targets.values()))

    proj_roots: Dict[str, Optional[str]] = {c: None for c in ALL_SCORER_CATEGORIES}
    npz_roots: Dict[str, Optional[str]] = {c: None for c in ALL_SCORER_CATEGORIES}
    if save_projections:
        for c in ALL_SCORER_CATEGORIES:
            p = os.path.join(out_dir, "projections", c)
            os.makedirs(p, exist_ok=True)
            proj_roots[c] = p
    if save_npz:
        for c in ALL_SCORER_CATEGORIES:
            p = os.path.join(out_dir, "npz", c)
            os.makedirs(p, exist_ok=True)
            npz_roots[c] = p

    t0 = time.time()
    idx_attempt = 0
    label_rows: List[Dict[str, Any]] = []
    kept_counts: Dict[str, int] = {c: 0 for c in ALL_SCORER_CATEGORIES}
    logged_full: set[str] = set()

    def is_all_full() -> bool:
        return all(kept_counts[c] >= targets[c] for c in ALL_SCORER_CATEGORIES)

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
        task = progress.add_task("[cyan]Generating (bucket fill)", total=target_total)

        if target_total == 0:
            progress.update(task, completed=0)
        else:
            while not is_all_full():
                remaining_total = target_total - int(sum(kept_counts.values()))
                b = int(min(batch_size, max(1, remaining_total)))

                with torch.no_grad():
                    x_0 = sample_voxels(
                        model,
                        betas,
                        shape=(b, 3, 16, 16, 16),
                        device=device,
                        n_steps=n_steps,
                        use_amp=use_amp,
                        track_every=None,
                        track_callback=None,
                        verbose=sample_verbose,
                    )

                for i in range(b):
                    x_0_onehot = centered_to_onehot(x_0[i])
                    probs = F.softmax(x_0_onehot, dim=0)
                    labels = decode_probs_to_labels(
                        probs, log_mask_threshold=log_mask_threshold
                    )

                    idx_attempt += 1
                    sid = len(label_rows) + 1  # id for kept samples only
                    stem = f"{filename_prefix}_{sid:05d}"

                    row = compute_sample_label_row(
                        labels,
                        sample_id=sid,
                        run_seed=run_seed,
                    )
                    cat = row["category"]
                    if kept_counts[cat] >= targets[cat]:
                        # bucket full → skip record / outputs
                        if cat not in logged_full:
                            console.print(
                                f"[dim]Bucket full: {cat} ({kept_counts[cat]}/{targets[cat]}) — skipping further samples in this category[/dim]"
                            )
                            logged_full.add(cat)
                        continue

                    pr = proj_roots.get(cat)
                    if pr:
                        png_path = os.path.join(pr, f"{stem}.png")
                        save_labels_and_projections(
                            labels,
                            png_path,
                            exp_name=stem,
                        )

                    nz = npz_roots.get(cat)
                    if nz:
                        save_voxel_npz(os.path.join(nz, f"{stem}.npz"), labels)

                    if save_npz:
                        row["source_name"] = f"npz/{cat}/{stem}.npz"
                    elif save_projections and pr:
                        row["source_name"] = f"projections/{cat}/{stem}.png"
                    else:
                        row["source_name"] = ""

                    label_rows.append(row)
                    kept_counts[cat] += 1
                    progress.update(task, advance=1)

                    if kept_counts[cat] == targets[cat] and cat not in logged_full:
                        console.print(
                            f"[dim]Bucket reached target: {cat} ({kept_counts[cat]}/{targets[cat]})[/dim]"
                        )
                        logged_full.add(cat)

                    if is_all_full():
                        break

    csv_path = os.path.join(out_dir, "sample_labels.csv")
    write_sample_labels_csv(label_rows, csv_path)
    console.print(f"[green]✓[/green] Sample labels: {csv_path}")

    summary_path = os.path.join(out_dir, "sample_labels_summary.csv")
    write_sample_labels_summary_csv(
        label_rows,
        summary_path,
    )
    console.print(f"[green]✓[/green] Summary: {summary_path}")

    elapsed = time.time() - t0
    console.print(
        f"[green]✓[/green] Done: kept={len(label_rows)} (targets={target_total}) "
        f"in {fmt_secs(elapsed)} ({elapsed / max(len(label_rows), 1):.2f}s / kept-sample)"
    )
    if any(proj_roots.values()):
        console.print(
            "[green]✓[/green] Projections: [cyan]projections/[/cyan]"
            "{positive, neg_float, neg_easy, neg_hard}/"
        )
        for c in ALL_SCORER_CATEGORIES:
            if proj_roots[c]:
                console.print(f"    [dim]→[/dim] [cyan]projections/{c}/[/cyan]")
    if any(npz_roots.values()):
        console.print(
            "[green]✓[/green] NPZ: [cyan]npz/[/cyan]"
            "{positive, neg_float, neg_easy, neg_hard}/"
        )
        for c in ALL_SCORER_CATEGORIES:
            if npz_roots[c]:
                console.print(f"    [dim]→[/dim] [cyan]npz/{c}/[/cyan]")
    cat_counts = {c: sum(1 for r in label_rows if r.get("category") == c) for c in ALL_SCORER_CATEGORIES}
    console.print("[bold]Category counts:[/bold] " + ", ".join(f"{k}={v}" for k, v in cat_counts.items()))
    console.print("[bold]Category targets:[/bold] " + ", ".join(f"{k}={targets[k]}" for k in ALL_SCORER_CATEGORIES))
    console.print(f"[dim]Attempts sampled (including skipped due to full bucket): {idx_attempt}[/dim]")
    return elapsed, label_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 16³ voxel diffusion samples only (no eval metrics)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt")
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Project directory (created if missing); all outputs go here. "
            "Last path component is used as experiment name and output filename prefix."
        ),
    )
    parser.add_argument("--n_pos", type=int, default=0, help="Target number of positive samples")
    parser.add_argument("--n_float", type=int, default=0, help="Target number of neg_float samples")
    parser.add_argument("--n_easy", type=int, default=0, help="Target number of neg_easy samples")
    parser.add_argument("--n_hard", type=int, default=0, help="Target number of neg_hard samples")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=None, help="Sampling steps (default: T)")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "cosine"]
    )
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_projections", action="store_true", help="Skip PNG projections")
    parser.add_argument(
        "--no_npz",
        action="store_true",
        help="Do not save per-sample .npz under npz/ (default: save npz)",
    )
    parser.add_argument(
        "--log_mask_threshold",
        type=float,
        default=None,
        help="Log-mask decode threshold; omit for argmax",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sample_verbose",
        action="store_true",
        help="Per-step prints from sample_voxels (slow, noisy)",
    )
    args = parser.parse_args()
    t_start = datetime.now()

    out_dir_leaf = Path(args.out_dir).expanduser().resolve().name
    if not out_dir_leaf:
        out_dir_leaf = "sample"

    if args.seed is not None:
        import random

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    console = Console()
    console.print("[bold]16³ Voxel Diffusion — sample generation only[/bold]\n")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    console.print(f"[cyan]Device: {device}[/cyan]")
    use_amp = device.type == "cuda" and not args.no_amp
    console.print(f"[cyan]AMP: {use_amp}[/cyan]")
    save_npz = not args.no_npz
    console.print(f"[cyan]Save NPZ:[/cyan] {save_npz}")
    console.print(
        "[cyan]Targets:[/cyan] "
        f"positive={args.n_pos}, neg_float={args.n_float}, neg_easy={args.n_easy}, neg_hard={args.n_hard}"
    )

    model, checkpoint = load_model(args.checkpoint, device)
    ck_args = checkpoint.get("args", {})
    T = ck_args.get("T", args.T)
    beta_schedule = ck_args.get("beta_schedule", args.beta_schedule)
    beta_start = ck_args.get("beta_start", args.beta_start) if "beta_start" in ck_args else args.beta_start
    beta_end = ck_args.get("beta_end", args.beta_end) if "beta_end" in ck_args else args.beta_end

    betas = BetaSchedule(T=T, schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end).to(
        device
    )
    console.print(f"[cyan]Schedule: T={T}, {beta_schedule}[/cyan]")

    os.makedirs(args.out_dir, exist_ok=True)
    console.print(f"[cyan]Project directory:[/cyan] {os.path.abspath(args.out_dir)}")
    console.print(
        f"[cyan]Experiment name / filename prefix:[/cyan] {out_dir_leaf} "
        f"([dim]from last component of --out_dir[/dim])"
    )

    script_snapshot_path = ""
    if "__file__" in globals():
        sp = Path(__file__).resolve()
        snap_name = f"{sp.stem}_snapshot_{t_start.strftime('%Y%m%d_%H%M%S')}{sp.suffix}"
        script_snapshot_path = os.path.join(os.path.abspath(args.out_dir), snap_name)
        shutil.copy2(sp, script_snapshot_path)
        console.print(f"[green]✓[/green] Script snapshot: [cyan]{script_snapshot_path}[/cyan]")

    meta: Dict = {
        "run_start": t_start.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "out_dir": os.path.abspath(args.out_dir),
        "exp_name": out_dir_leaf,
        "out_dir_leaf": out_dir_leaf,
        "command": get_invocation_command(),
        "script_snapshot_py": script_snapshot_path or "None",
        "target_positive": int(args.n_pos),
        "target_neg_float": int(args.n_float),
        "target_neg_easy": int(args.n_easy),
        "target_neg_hard": int(args.n_hard),
        "target_total": int(args.n_pos + args.n_float + args.n_easy + args.n_hard),
        "batch_size": args.batch_size,
        "n_steps": args.n_steps if args.n_steps is not None else T,
        "T": T,
        "beta_schedule": beta_schedule,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "seed": args.seed if args.seed is not None else "None",
        "amp": str(use_amp),
        "log_mask_threshold": args.log_mask_threshold
        if args.log_mask_threshold is not None
        else "None",
        "save_projections": str(not args.no_projections),
        "save_npz": str(save_npz),
        "no_npz": str(args.no_npz),
        "filename_prefix": out_dir_leaf,
        "sample_labels_csv": os.path.join(os.path.abspath(args.out_dir), "sample_labels.csv"),
        "sample_labels_summary_csv": os.path.join(
            os.path.abspath(args.out_dir), "sample_labels_summary.csv"
        ),
        "projections_root": os.path.join(os.path.abspath(args.out_dir), "projections"),
        "npz_root": os.path.join(os.path.abspath(args.out_dir), "npz"),
        "projections_positive_dir": os.path.join(
            os.path.abspath(args.out_dir), "projections", CAT_POSITIVE
        ),
        "projections_neg_float_dir": os.path.join(
            os.path.abspath(args.out_dir), "projections", CAT_NEG_FLOAT
        ),
        "projections_neg_easy_dir": os.path.join(
            os.path.abspath(args.out_dir), "projections", CAT_NEG_EASY
        ),
        "projections_neg_hard_dir": os.path.join(
            os.path.abspath(args.out_dir), "projections", CAT_NEG_HARD
        ),
        "npz_positive_dir": os.path.join(os.path.abspath(args.out_dir), "npz", CAT_POSITIVE),
        "npz_neg_float_dir": os.path.join(
            os.path.abspath(args.out_dir), "npz", CAT_NEG_FLOAT
        ),
        "npz_neg_easy_dir": os.path.join(
            os.path.abspath(args.out_dir), "npz", CAT_NEG_EASY
        ),
        "npz_neg_hard_dir": os.path.join(
            os.path.abspath(args.out_dir), "npz", CAT_NEG_HARD
        ),
    }
    save_metadata(meta, args.out_dir, console)

    targets = {
        CAT_POSITIVE: int(args.n_pos),
        CAT_NEG_FLOAT: int(args.n_float),
        CAT_NEG_EASY: int(args.n_easy),
        CAT_NEG_HARD: int(args.n_hard),
    }
    elapsed, _ = generate_samples(
        model=model,
        betas=betas,
        device=device,
        target_counts=targets,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_amp=use_amp,
        save_projections=not args.no_projections,
        save_npz=save_npz,
        log_mask_threshold=args.log_mask_threshold,
        filename_prefix=out_dir_leaf,
        sample_verbose=args.sample_verbose,
        run_seed=args.seed,
        console=console,
    )

    meta_done = {
        "run_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_secs": elapsed,
        "elapsed_formatted": fmt_secs(elapsed),
    }
    kv_path = os.path.join(args.out_dir, "metadata.csv")
    with open(kv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in meta_done.items():
            w.writerow([k, v])

    flat = {**meta, **meta_done}
    with open(os.path.join(args.out_dir, "metadata_flat.csv"), "w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=list(flat.keys()))
        dw.writeheader()
        dw.writerow(flat)

    console.print(f"\n[bold green]Output:[/bold green] {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
