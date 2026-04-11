#!/usr/bin/env python3
"""
對資料夾內已存在的 voxel .npz 批次計算指標（預設遞迴掃描子資料夾內的 .npz）。

使用 utils.voxel_npz_io.load_voxel_npz 與 utils.voxel_sample_metrics.compute_sample_metrics。
輸出 sample_labels.csv、sample_labels_summary.csv；欄位與兩個 generate 腳本一致，**最後**一欄皆為 ``source_name``：此處為相對於 ``--npz_dir``（掃描根）的原始 .npz 路徑（POSIX 斜線；無法相對化時為檔名）。generate 腳本則為相對於 ``--out_dir`` 的 ``npz/...`` 或 ``projections/...``。

投影圖主標題（``exp_name`` 傳給 ``save_labels_and_projections``）為 ``--out_dir`` 解析後路徑的最後一層目錄名（預設 ``out_dir`` 與 ``--npz_dir`` 相同時即為該目錄名）；若無法取得則為 ``eval_YYYYMMDD_HHMMSS``。

使用方式:
  python eval_multi_npz.py --npz_dir /path/to/npz_root [--out_dir /path/to/out]
  python eval_multi_npz.py --npz_dir /path/to/npz_root --shallow  # 僅掃描該目錄，不含子資料夾
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")

from rich.console import Console

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from utils.voxel_label_projections import save_labels_and_projections
from utils.voxel_npz_io import load_voxel_npz
from utils.voxel_sample_metrics import (
    ALL_SCORER_CATEGORIES,
    CAT_NEG_EASY,
    CAT_NEG_FLOAT,
    CAT_NEG_HARD,
    CAT_POSITIVE,
    compute_sample_metrics,
)


def compute_sample_label_row(
    labels: np.ndarray,
    sample_id: int,
    run_seed: Optional[int],
    hard_neg_llr_threshold: float = 0.5,
) -> Dict[str, Any]:
    """與 generate_16_voxel_diffusion_bucket.compute_sample_label_row 相同。"""
    m = compute_sample_metrics(labels, hard_neg_llr_threshold=hard_neg_llr_threshold)
    llr = m["Largest_Log_Ratio"]
    llr_store = round(float(llr), 6) if llr >= 0 else -1.0

    return {
        "id": sample_id,
        "seed": "" if run_seed is None else int(run_seed),
        "category": m["Scorer_Category"],
        "is_main_trunk_broken": 1 if m["Is_Main_Trunk_Broken"] else 0,
        "is_broken": 1 if m["Is_Broken"] else 0,
        "mass": int(m["Mass"]),
        "height": int(m["Height"]),
        "log_size": int(m["Log_Size"]),
        "leaf_size": int(m["Leaf_Size"]),
        "base_connected_ratio": round(float(m["Base_Connected_Ratio"]), 6),
        "base_connected_size": int(m["Base_Connected_Size"]),
        "total_log_size": int(m["Total_Log_Size"]),
        "largest_log_ratio": llr_store,
        "occupancy_non_air": round(float(m["Occupancy_Non_Air"]), 6),
        "occupancy_log": round(float(m["Occupancy_Log"]), 6),
        "occupancy_leaf": round(float(m["Occupancy_Leaf"]), 6),
        "components_non_air": int(m["Components_Non_Air"]),
        "components_log": int(m["Components_Log"]),
        "components_leaf": int(m["Components_Leaf"]),
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
        "is_broken",
        "mass",
        "height",
        "log_size",
        "leaf_size",
        "base_connected_ratio",
        "base_connected_size",
        "total_log_size",
        "largest_log_ratio",
        "occupancy_non_air",
        "occupancy_log",
        "occupancy_leaf",
        "components_non_air",
        "components_log",
        "components_leaf",
        "source_name",
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
    """與 generate_16_voxel_diffusion_bucket.write_sample_labels_summary_csv 相同。"""
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

    broken_flag = np.array([int(r["is_broken"]) for r in rows], dtype=np.float64)
    n_bf = int(broken_flag.sum())
    add_section("Broken flag (is_broken)")
    lines.append(["broken_count", str(n_bf)])
    lines.append(["intact_count", str(n - n_bf)])
    lines.append(["broken_rate_pct", f"{100.0 * n_bf / n:.4f}"])
    lines.append(["intact_rate_pct", f"{100.0 * (n - n_bf) / n:.4f}"])

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

    mass = np.array([float(r["mass"]) for r in rows])
    add_section("mass")
    lines.append(["mean", f"{float(mass.mean()):.4f}"])
    lines.append(["std", f"{float(mass.std()):.4f}"])
    lines.append(["min", str(int(mass.min()))])
    lines.append(["max", str(int(mass.max()))])

    height = np.array([float(r["height"]) for r in rows])
    add_section("height")
    lines.append(["mean", f"{float(height.mean()):.4f}"])
    lines.append(["std", f"{float(height.std()):.4f}"])
    lines.append(["min", str(int(height.min()))])
    lines.append(["max", str(int(height.max()))])

    log_sz = np.array([float(r["log_size"]) for r in rows])
    add_section("log_size")
    lines.append(["mean", f"{float(log_sz.mean()):.4f}"])
    lines.append(["std", f"{float(log_sz.std()):.4f}"])
    lines.append(["min", str(int(log_sz.min()))])
    lines.append(["max", str(int(log_sz.max()))])

    leaf_sz = np.array([float(r["leaf_size"]) for r in rows])
    add_section("leaf_size")
    lines.append(["mean", f"{float(leaf_sz.mean()):.4f}"])
    lines.append(["std", f"{float(leaf_sz.std()):.4f}"])
    lines.append(["min", str(int(leaf_sz.min()))])
    lines.append(["max", str(int(leaf_sz.max()))])

    occ_na = np.array([float(r["occupancy_non_air"]) for r in rows])
    add_section("occupancy_non_air")
    lines.append(["mean", f"{float(occ_na.mean()):.6f}"])
    lines.append(["std", f"{float(occ_na.std()):.6f}"])
    lines.append(["min", f"{float(occ_na.min()):.6f}"])
    lines.append(["max", f"{float(occ_na.max()):.6f}"])

    occ_lg = np.array([float(r["occupancy_log"]) for r in rows])
    add_section("occupancy_log")
    lines.append(["mean", f"{float(occ_lg.mean()):.6f}"])
    lines.append(["std", f"{float(occ_lg.std()):.6f}"])
    lines.append(["min", f"{float(occ_lg.min()):.6f}"])
    lines.append(["max", f"{float(occ_lg.max()):.6f}"])

    occ_lf = np.array([float(r["occupancy_leaf"]) for r in rows])
    add_section("occupancy_leaf")
    lines.append(["mean", f"{float(occ_lf.mean()):.6f}"])
    lines.append(["std", f"{float(occ_lf.std()):.6f}"])
    lines.append(["min", f"{float(occ_lf.min()):.6f}"])
    lines.append(["max", f"{float(occ_lf.max()):.6f}"])

    comp_na = np.array([float(r["components_non_air"]) for r in rows])
    add_section("components_non_air")
    lines.append(["mean", f"{float(comp_na.mean()):.4f}"])
    lines.append(["std", f"{float(comp_na.std()):.4f}"])
    lines.append(["min", str(int(comp_na.min()))])
    lines.append(["max", str(int(comp_na.max()))])

    comp_lg = np.array([float(r["components_log"]) for r in rows])
    add_section("components_log")
    lines.append(["mean", f"{float(comp_lg.mean()):.4f}"])
    lines.append(["std", f"{float(comp_lg.std()):.4f}"])
    lines.append(["min", str(int(comp_lg.min()))])
    lines.append(["max", str(int(comp_lg.max()))])

    comp_lf = np.array([float(r["components_leaf"]) for r in rows])
    add_section("components_leaf")
    lines.append(["mean", f"{float(comp_lf.mean()):.4f}"])
    lines.append(["std", f"{float(comp_lf.std()):.4f}"])
    lines.append(["min", str(int(comp_lf.min()))])
    lines.append(["max", str(int(comp_lf.max()))])

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


def source_name_for_csv(file_path: Path, scan_root: Path) -> str:
    """路徑相對於掃描根目錄（含子資料夾內檔案時可區分路徑）；失敗則為檔名。"""
    root = scan_root.resolve()
    p = file_path.resolve()
    try:
        return str(p.relative_to(root).as_posix())
    except ValueError:
        return p.name


def discover_npz_paths(root: Path, recursive: bool) -> List[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    if recursive:
        paths = sorted(root.rglob("*.npz"))
    else:
        paths = sorted(p for p in root.iterdir() if p.suffix.lower() == ".npz" and p.is_file())
    return paths


def evaluate_folder(
    npz_paths: List[Path],
    *,
    scan_root: Path,
    expected_shape: Optional[Tuple[int, int, int]],
    run_seed: Optional[int],
    hard_neg_llr_threshold: float,
    save_projections: bool,
    projections_dir: Optional[str],
    projection_exp_name: str,
    console: Console,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(npz_paths):
        arr = load_voxel_npz(p)
        if arr.ndim != 3:
            raise ValueError(f"{p}: expected 3D array, got shape {arr.shape}")
        if expected_shape is not None and tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"{p}: expected shape {expected_shape}, got {tuple(arr.shape)} "
                f"(use --no_require_16_cube to skip)"
            )
        labels = np.asarray(arr)
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int64, copy=False)
        row = compute_sample_label_row(
            labels,
            sample_id=i + 1,
            run_seed=run_seed,
            hard_neg_llr_threshold=hard_neg_llr_threshold,
        )
        src = source_name_for_csv(p, scan_root)
        row["source_name"] = src
        rows.append(row)

        if save_projections and projections_dir:
            labels_u8 = labels.astype(np.uint8, copy=False)
            png_path = os.path.join(projections_dir, f"npz_eval_{i + 1:05d}.png")
            try:
                save_labels_and_projections(
                    labels_u8,
                    png_path,
                    title_suffix=f" {i + 1:05d} {row['category']}",
                    exp_name=projection_exp_name,
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠[/yellow] Failed to save projection for {src}: {e}"
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate voxel labels in a folder of .npz files; sample_labels.csv columns match "
            "generate_16_voxel_diffusion / generate_16_voxel_diffusion_bucket (including source_name)."
        ),
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        required=True,
        help="Directory containing .npz files (voxel array key per voxel_npz_io)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for CSVs (default: same as --npz_dir)",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="Only scan immediate directory (default: include all *.npz under subdirectories)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_require_16_cube",
        action="store_true",
        help="Do not require shape (16, 16, 16)",
    )
    parser.add_argument(
        "--hard_neg_llr_threshold",
        type=float,
        default=0.5,
        metavar="R",
        help="Same as generate_16_voxel_diffusion --hard_neg_llr_threshold (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, written in seed column for every row (for traceability; not used for RNG here)",
    )
    parser.add_argument(
        "--no_projections",
        action="store_true",
        help="Do not write projections/ three-view PNGs (default: write under --out_dir)",
    )
    parser.add_argument(
        "--projections_dir",
        type=str,
        default=None,
        help="Directory for PNGs (default: <out_dir>/projections)",
    )
    args = parser.parse_args()

    console = Console()
    npz_root = Path(args.npz_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else npz_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_abs = out_dir.resolve()

    save_projections = not args.no_projections
    projections_dir: Optional[str] = None
    if save_projections:
        projections_dir = (
            os.path.abspath(args.projections_dir)
            if args.projections_dir
            else os.path.join(str(out_dir_abs), "projections")
        )
        os.makedirs(projections_dir, exist_ok=True)

    projection_leaf = out_dir_abs.name
    if not projection_leaf:
        projection_leaf = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    projection_exp_name = projection_leaf
    if save_projections:
        console.print(
            f"[cyan]Projection figure title:[/cyan] {projection_exp_name} "
            f"([dim]from last component of --out_dir[/dim])"
        )

    expected: Optional[Tuple[int, int, int]] = None if args.no_require_16_cube else (16, 16, 16)

    t0 = time.time()
    scan_root = npz_root.resolve()
    paths = discover_npz_paths(npz_root, recursive=not args.shallow)
    if not paths:
        console.print(f"[yellow]No .npz files found under[/yellow] {scan_root}")
        rows: List[Dict[str, Any]] = []
    else:
        console.print(f"[cyan]Found {len(paths)} .npz file(s)[/cyan]")
        rows = evaluate_folder(
            paths,
            scan_root=scan_root,
            expected_shape=expected,
            run_seed=args.seed,
            hard_neg_llr_threshold=args.hard_neg_llr_threshold,
            save_projections=save_projections,
            projections_dir=projections_dir,
            projection_exp_name=projection_exp_name,
            console=console,
        )

    csv_labels = os.path.join(out_dir, "sample_labels.csv")
    csv_summary = os.path.join(out_dir, "sample_labels_summary.csv")

    write_sample_labels_csv(rows, csv_labels)
    write_sample_labels_summary_csv(
        rows,
        csv_summary,
        hard_neg_llr_threshold=args.hard_neg_llr_threshold,
    )

    elapsed = time.time() - t0
    console.print(f"[green]✓[/green] {csv_labels}")
    console.print(f"[green]✓[/green] {csv_summary}")
    if save_projections and projections_dir and paths:
        console.print(f"[green]✓[/green] Projections: [cyan]{projections_dir}[/cyan]")
    console.print(f"[dim]{elapsed:.2f}s[/dim]")


if __name__ == "__main__":
    main()
