#!/usr/bin/env python3
"""
Generate `sample_labels_summary.csv` from per-sample label rows.

（由 ``utils.export_csv`` 集中維護，供 generate / eval 腳本共用。）
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .voxel_sample_metrics import (
    ALL_SCORER_CATEGORIES,
    CAT_NEG_EASY,
    CAT_NEG_FLOAT,
    CAT_NEG_HARD,
    CAT_POSITIVE,
)


def write_sample_labels_summary_csv(
    rows: List[Dict[str, Any]],
    path: str,
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
    n_pos = sum(1 for x in cats if x == CAT_POSITIVE)
    n_neg_float = sum(1 for x in cats if x == CAT_NEG_FLOAT)
    n_neg_easy = sum(1 for x in cats if x == CAT_NEG_EASY)
    n_neg_hard = sum(1 for x in cats if x == CAT_NEG_HARD)
    denom = float(n) if n > 0 else 1.0
    lines.append(["positive (n_samples)", str(n_pos)])
    lines.append(["negative floating (n_samples)", str(n_neg_float)])
    lines.append(["negative easy (n_samples)", str(n_neg_easy)])
    lines.append(["negative hard (n_samples)", str(n_neg_hard)])
    lines.append(["positive (%)", f"{100.0 * n_pos / denom:.4f}" if n > 0 else "0.0000"])
    lines.append(
        ["negative floating (%)", f"{100.0 * n_neg_float / denom:.4f}" if n > 0 else "0.0000"]
    )
    lines.append(["negative easy (%)", f"{100.0 * n_neg_easy / denom:.4f}" if n > 0 else "0.0000"])
    lines.append(["negative hard (%)", f"{100.0 * n_neg_hard / denom:.4f}" if n > 0 else "0.0000"])

    if n == 0:
        add_section("Scorer-style categories (scorer bucketing)")
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

    llr_all = np.array([float(r["largest_log_ratio"]) for r in rows])
    valid = llr_all >= 0.0
    v_llr = llr_all[valid] if bool(valid.any()) else np.array([], dtype=np.float64)
    add_section("largest_log_ratio")
    lines.append(["n_valid (>=0)", str(int(valid.sum()))])
    lines.append(["n_invalid (-1 / missing)", str(int((~valid).sum()))])
    if v_llr.size > 0:
        lines.append(["mean (valid only)", f"{float(v_llr.mean()):.6f}"])
        lines.append(["std (valid only)", f"{float(v_llr.std()):.6f}"])
        lines.append(["min (valid only)", f"{float(v_llr.min()):.6f}"])
        lines.append(["max (valid only)", f"{float(v_llr.max()):.6f}"])
        lines.append(
            [
                "pct_samples_largest_log_ratio_ge_0.95",
                f"{100.0 * float(np.mean(v_llr >= 0.95)):.4f}",
            ]
        )
        lines.append(
            [
                "pct_samples_largest_log_ratio_ge_0.99",
                f"{100.0 * float(np.mean(v_llr >= 0.99)):.4f}",
            ]
        )

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

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(lines)
