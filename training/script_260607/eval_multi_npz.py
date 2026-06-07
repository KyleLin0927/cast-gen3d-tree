#!/usr/bin/env python3
"""
對資料夾內已存在的 voxel .npz 批次計算指標（預設遞迴掃描子資料夾內的 .npz）。

使用 utils.voxel_npz_io.load_voxel_npz；每列指標與 CSV 寫入由 ``utils.export_csv`` 統一產生。
輸出 sample_labels.csv、sample_labels_summary.csv；欄位與兩個 generate 腳本一致，**最後**一欄皆為 ``source_name``：此處為相對於 ``--npz_dir``（掃描根）的原始 .npz 路徑（POSIX 斜線；無法相對化時為檔名）。generate 腳本則為相對於 ``--out_dir`` 的 ``npz/...`` 或 ``projections/...``。

投影圖主標題（``exp_name`` 傳給 ``save_labels_and_projections``）為 ``--out_dir`` 解析後路徑的最後一層目錄名（預設 ``out_dir`` 與 ``--npz_dir`` 相同時即為該目錄名）；若無法取得則為 ``eval_YYYYMMDD_HHMMSS``。

使用方式:
  python eval_multi_npz.py --npz_dir /path/to/npz_root [--out_dir /path/to/out]
  python eval_multi_npz.py --npz_dir /path/to/npz_root --shallow  # 僅掃描該目錄，不含子資料夾
"""

from __future__ import annotations

import argparse
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

from utils.export_csv import (
    compute_sample_label_row,
    write_sample_labels_csv,
    write_sample_labels_summary_csv,
)
from utils.voxel_label_projections import save_labels_and_projections
from utils.voxel_npz_io import load_voxel_npz


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
    )

    elapsed = time.time() - t0
    console.print(f"[green]✓[/green] {csv_labels}")
    console.print(f"[green]✓[/green] {csv_summary}")
    if save_projections and projections_dir and paths:
        console.print(f"[green]✓[/green] Projections: [cyan]{projections_dir}[/cyan]")
    console.print(f"[dim]{elapsed:.2f}s[/dim]")


if __name__ == "__main__":
    main()
