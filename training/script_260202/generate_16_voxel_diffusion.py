#!/usr/bin/env python3
"""
單純從 16x16x16 Voxel Diffusion 模型生成樣本（不跑評估指標、動力學、CSV 摘要）。

輸出（預設）：
- projections/{positive,neg_float,neg_easy,neg_hard}/：各類別三視圖 PNG
- npz/{...}/：同上，每檔僅含陣列鍵 ``voxel``（--no_npz 可關閉）
- sample_labels.csv: 每個樣本的 id、seed、樹幹/連通性標籤（見下方欄位）
- sample_labels_summary.csv: 全體樣本指標加總與平均、標準差等
- 分類定義（r = --hard_neg_llr_threshold）：
  - neg_float：base_connected_size==0（連地板都沒碰到）
  - positive：base_connected_size>0，且全樹只有 1 個連通塊（log_components==1；Absolute Connectivity）
  - neg_hard：base_connected_size>0，且 log_components==2
  - neg_easy：base_connected_size>0，且 log_components>2
- metadata.csv / metadata_flat.csv：重現用參數（含 r / hard_neg_llr_threshold）
- generate_16_voxel_diffusion_snapshot_YYYYMMDD_HHMMSS.py：執行當下本腳本完整備份

sample_labels.csv 欄位：
- id, seed, category（positive | neg_float | neg_easy | neg_hard）, is_main_trunk_broken(0/1),
  base_connected_ratio, base_connected_size, total_log_size, largest_log_ratio

專案目錄即 --out_dir：所有輸出寫入該路徑（會自動建立），無需另給實驗名稱。

使用方式:
  python generate_16_voxel_diffusion.py --checkpoint path/to/model.pt --out_dir ./my_project --exp_name exp_001 --n_samples 50 --basename sample
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
    from unet_diffusion_16_voxel import (
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
from utils.voxel_sample_metrics import (
    ALL_SCORER_CATEGORIES,
    CAT_NEG_EASY,
    CAT_NEG_FLOAT,
    CAT_NEG_HARD,
    CAT_POSITIVE,
    compute_sample_metrics,
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


def compute_sample_label_row(
    labels: np.ndarray,
    sample_id: int,
    run_seed: Optional[int],
    hard_neg_llr_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    由離散 labels 計算寫入 CSV 的一列（與 utils.voxel_sample_metrics.compute_sample_metrics 同一套指標）。
    category 仍依連通塊等規則由 compute_sample_metrics 決定；CSV 不寫 log_components。
    """
    m = compute_sample_metrics(labels, hard_neg_llr_threshold=hard_neg_llr_threshold)
    base_sz = int(m["Base_Connected_Size"])
    total_log = int(m["Total_Log_Size"])
    ratio = float(m["Base_Connected_Ratio"])
    llr_store = round(float(m["Largest_Log_Ratio"]), 6) if m["Largest_Log_Ratio"] >= 0 else -1.0
    broken = bool(m["Is_Main_Trunk_Broken"])

    return {
        "id": sample_id,
        "seed": "" if run_seed is None else int(run_seed),
        "category": m["Scorer_Category"],
        "is_main_trunk_broken": 1 if broken else 0,
        "base_connected_ratio": round(ratio, 6),
        "base_connected_size": base_sz,
        "total_log_size": total_log,
        "largest_log_ratio": llr_store,
    }


def write_sample_labels_csv(
    rows: List[Dict[str, Any]],
    path: str,
) -> None:
    fieldnames = [
        "id",
        "seed",
        "category",
        "is_main_trunk_broken",
        "base_connected_ratio",
        "base_connected_size",
        "total_log_size",
        "largest_log_ratio",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)


def write_sample_labels_summary_csv(
    rows: List[Dict[str, Any]],
    path: str,
    hard_neg_llr_threshold: float,
) -> None:
    """由 per-sample 列計算統計，寫成兩欄 CSV（Metric, Value）。"""
    lines: List[List[str]] = [["Metric", "Value"]]

    def add_section(title: str) -> None:
        lines.append(["", ""])
        lines.append([title, ""])

    n = len(rows)
    add_section("Overview")
    lines.append(["n_samples", str(n)])

    cats = [r.get("category", "") for r in rows]
    add_section("Category sample counts")
    lines.append(["positive (n_samples)", str(sum(1 for x in cats if x == CAT_POSITIVE))])
    lines.append(["negative floating (n_samples)", str(sum(1 for x in cats if x == CAT_NEG_FLOAT))])
    lines.append(["negative easy (n_samples)", str(sum(1 for x in cats if x == CAT_NEG_EASY))])
    lines.append(["negative hard (n_samples)", str(sum(1 for x in cats if x == CAT_NEG_HARD))])

    if n == 0:
        add_section("Scorer-style categories (scorer bucketing)")
        lines.append(["hard_neg_llr_threshold_r", f"{hard_neg_llr_threshold:.6f}"])
        lines.append(
            [
                "note",
                "positive (base_sz>0, log_components==1) | neg_float (base_sz==0) | neg_hard (base_sz>0, log_components==2) | neg_easy (base_sz>0, log_components>2)",
            ]
        )
        for c in ALL_SCORER_CATEGORIES:
            lines.append([f"count_{c}", "0"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(lines)
        return

    broken = np.array([int(r["is_main_trunk_broken"]) for r in rows], dtype=np.float64)
    br = np.array([float(r["base_connected_ratio"]) for r in rows])
    bcs = np.array([float(r["base_connected_size"]) for r in rows])
    tls = np.array([float(r["total_log_size"]) for r in rows])

    n_broken = int(broken.sum())
    n_intact = n - n_broken
    add_section("Main trunk (is_main_trunk_broken)")
    lines.append(["broken_count", str(n_broken)])
    lines.append(["intact_count", str(n_intact)])
    lines.append(["broken_rate_pct", f"{100.0 * n_broken / n:.4f}"])
    lines.append(["intact_rate_pct", f"{100.0 * n_intact / n:.4f}"])

    sum_tls = float(tls.sum())
    sum_bcs = float(bcs.sum())
    pooled_ratio = (sum_bcs / sum_tls) if sum_tls > 0 else 0.0
    add_section("Pooled over voxels (all samples)")
    lines.append(
        [
            "pooled_base_connected_ratio",
            f"{pooled_ratio:.6f}",
        ]
    )
    lines.append(
        [
            "pooled_base_connected_ratio_pct",
            f"{100.0 * pooled_ratio:.4f}",
        ]
    )
    lines.append(["note_pooled_ratio", "sum(base_connected_size) / sum(total_log_size)"])
    lines.append(["sum_total_log_size (voxels)", str(int(sum_tls))])
    lines.append(["sum_base_connected_size (voxels)", str(int(sum_bcs))])

    add_section("base_connected_ratio (per-sample)")
    lines.append(["mean", f"{float(br.mean()):.6f}"])
    lines.append(["std", f"{float(br.std()):.6f}"])
    lines.append(["min", f"{float(br.min()):.6f}"])
    lines.append(["max", f"{float(br.max()):.6f}"])

    add_section("base_connected_size")
    lines.append(["mean", f"{float(bcs.mean()):.4f}"])
    lines.append(["std", f"{float(bcs.std()):.4f}"])
    lines.append(["min", str(int(bcs.min()))])
    lines.append(["max", str(int(bcs.max()))])

    add_section("total_log_size")
    lines.append(["mean", f"{float(tls.mean()):.4f}"])
    lines.append(["std", f"{float(tls.std()):.4f}"])
    lines.append(["min", str(int(tls.min()))])
    lines.append(["max", str(int(tls.max()))])

    llr_all = np.array([float(r["largest_log_ratio"]) for r in rows])
    valid = llr_all >= 0.0

    add_section("largest_log_ratio")
    lines.append(["n_valid (>=0)", str(int(valid.sum()))])
    lines.append(["n_invalid (-1 / missing)", str(int((~valid).sum()))])
    if valid.any():
        v = llr_all[valid]
        lines.append(["mean (valid only)", f"{float(v.mean()):.6f}"])
        lines.append(["std (valid only)", f"{float(v.std()):.6f}"])
        lines.append(["min (valid only)", f"{float(v.min()):.6f}"])
        lines.append(["max (valid only)", f"{float(v.max()):.6f}"])
        lines.append(
            [
                "pct_samples_largest_log_ratio_ge_0.95",
                f"{100.0 * float(np.mean(v >= 0.95)):.4f}",
            ]
        )
        lines.append(
            [
                "pct_samples_largest_log_ratio_ge_0.99",
                f"{100.0 * float(np.mean(v >= 0.99)):.4f}",
            ]
        )
        lines.append(
            [
                f"pct_samples_largest_log_ratio_ge_r (r={hard_neg_llr_threshold:.6f})",
                f"{100.0 * float(np.mean(v >= hard_neg_llr_threshold)):.4f}",
            ]
        )
        lines.append(
            [
                "note_pct_ge_r",
                "among samples with valid largest_log_ratio only (aligns with neg_hard llr>=r)",
            ]
        )

    add_section("Scorer-style categories (scorer bucketing)")
    lines.append(["hard_neg_llr_threshold_r", f"{hard_neg_llr_threshold:.6f}"])
    lines.append(
        [
            "note",
            "positive | neg_float | neg_easy | neg_hard — see script docstring",
        ]
    )
    for c in ALL_SCORER_CATEGORIES:
        cnt = sum(1 for x in cats if x == c)
        lines.append([f"count_{c}", str(cnt)])
    if n > 0:
        for c in ALL_SCORER_CATEGORIES:
            cnt = sum(1 for x in cats if x == c)
            lines.append([f"pct_{c}_of_all", f"{100.0 * cnt / n:.4f}"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(lines)


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
    n_samples: int,
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
    hard_neg_llr_threshold: float = 0.5,
    console: Optional[Console] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    if console is None:
        console = Console()

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
    idx = 0
    label_rows: List[Dict[str, Any]] = []

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
        task = progress.add_task("[cyan]Generating", total=n_samples)

        for batch_start in range(0, n_samples, batch_size):
            b = min(batch_size, n_samples - batch_start)
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
                labels = decode_probs_to_labels(probs, log_mask_threshold=log_mask_threshold)
                sid = idx + 1
                stem = f"{filename_prefix}_{sid:05d}"

                row = compute_sample_label_row(
                    labels,
                    sample_id=sid,
                    run_seed=run_seed,
                    hard_neg_llr_threshold=hard_neg_llr_threshold,
                )
                cat = row["category"]
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

                label_rows.append(row)

                idx += 1
                progress.update(task, advance=1)

    csv_path = os.path.join(out_dir, "sample_labels.csv")
    write_sample_labels_csv(label_rows, csv_path)
    console.print(f"[green]✓[/green] Sample labels: {csv_path}")

    summary_path = os.path.join(out_dir, "sample_labels_summary.csv")
    write_sample_labels_summary_csv(
        label_rows,
        summary_path,
        hard_neg_llr_threshold=hard_neg_llr_threshold,
    )
    console.print(f"[green]✓[/green] Summary: {summary_path}")

    elapsed = time.time() - t0
    console.print(
        f"[green]✓[/green] Done: {n_samples} samples in {fmt_secs(elapsed)} "
        f"({elapsed / max(n_samples, 1):.2f}s / sample)"
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
    console.print(
        f"[cyan]hard_neg_llr_threshold r=[/cyan]{hard_neg_llr_threshold:.6f} "
        f"([dim]neg_hard: broken, base_sz>0, valid llr>=r[/dim])"
    )
    console.print("[bold]Category counts:[/bold] " + ", ".join(f"{k}={v}" for k, v in cat_counts.items()))
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
        help="Project directory (created if missing); all outputs go here",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Generation experiment name (for metadata / run tracking)",
    )
    parser.add_argument("--n_samples", type=int, default=32, help="Number of samples")
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
    # Output file naming: <basename>_<number>.(png|npz)
    # Keep --prefix for backward compatibility; prefer --basename going forward.
    parser.add_argument(
        "--basename",
        "--prefix",
        dest="prefix",
        type=str,
        default="sample",
        help="Output filename base. Files are named like <basename>_<number>.* (alias: --prefix)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sample_verbose",
        action="store_true",
        help="Per-step prints from sample_voxels (slow, noisy)",
    )
    parser.add_argument(
        "--hard_neg_llr_threshold",
        type=float,
        default=0.5,
        metavar="R",
        help=(
            "neg_hard: base_connected_size>0, log_components>1, valid largest_log_ratio >= R. "
            "neg_easy: base_connected_size>0, log_components>1, llr invalid or < R. "
            "neg_float: base_connected_size==0. Default R: 0.5"
        ),
    )

    args = parser.parse_args()
    t_start = datetime.now()

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
        f"[cyan]hard_neg_llr_threshold (r):[/cyan] {args.hard_neg_llr_threshold:.6f}"
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
    if args.exp_name:
        console.print(f"[cyan]Experiment name:[/cyan] {args.exp_name}")

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
        "exp_name": args.exp_name if args.exp_name else "None",
        "command": get_invocation_command(),
        "script_snapshot_py": script_snapshot_path or "None",
        "n_samples": args.n_samples,
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
        "basename": args.prefix,
        "prefix": args.prefix,  # backward-compatible key for existing parsers
        "sample_labels_csv": os.path.join(os.path.abspath(args.out_dir), "sample_labels.csv"),
        "sample_labels_summary_csv": os.path.join(
            os.path.abspath(args.out_dir), "sample_labels_summary.csv"
        ),
        # r：neg_hard 門檻 largest_log_ratio >= r（components>1 且接地後才會用到）
        "r": f"{args.hard_neg_llr_threshold:.10g}",
        "hard_neg_llr_threshold": f"{args.hard_neg_llr_threshold:.10g}",
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

    elapsed, _ = generate_samples(
        model=model,
        betas=betas,
        device=device,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_amp=use_amp,
        save_projections=not args.no_projections,
        save_npz=save_npz,
        log_mask_threshold=args.log_mask_threshold,
        filename_prefix=args.prefix,
        sample_verbose=args.sample_verbose,
        run_seed=args.seed,
        hard_neg_llr_threshold=args.hard_neg_llr_threshold,
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
