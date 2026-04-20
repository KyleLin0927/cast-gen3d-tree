#!/usr/bin/env python3
"""
Per-run CSV helpers: per-sample label tables (e.g. ``sample_labels.csv`` /
``simple_label.csv``), per-timestep dynamics label traces (``dynamics_label_trace.csv``),
and two-column summaries (``sample_labels_summary.csv`` / ``simple_label_summary.csv``).

供 generate / eval 等腳本共用。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .voxel_sample_metrics import (
    ALL_SCORER_CATEGORIES,
    CAT_NEG_EASY,
    CAT_NEG_FLOAT,
    CAT_NEG_HARD,
    CAT_POSITIVE,
    compute_sample_metrics,
)

# Header order for ``sample_labels.csv`` (must match ``compute_sample_label_row`` keys + ``source_name``).
SAMPLE_LABELS_CSV_FIELDNAMES: Tuple[str, ...] = (
    "id",
    "seed",
    "category",
    "is_main_trunk_broken",
    "is_broken",
    "largest_log_ratio",
    "mass",
    "height",
    "base_connected_ratio",
    "base_connected_size",
    "log_size",
    "leaf_size",
    "occupancy_non_air",
    "occupancy_log",
    "occupancy_leaf",
    "components_non_air",
    "components_log",
    "components_leaf",
    "source_name",
)

# Per-timestep dynamics trace: same label columns as ``SAMPLE_LABELS_CSV_FIELDNAMES`` with leading
# ``sample_idx`` (1-based, 與 dynamics_trace.csv 一致), ``step_idx``, ``t``.
DYNAMICS_LABEL_TRACE_CSV_FIELDNAMES: Tuple[str, ...] = (
    "sample_idx",
    "step_idx",
    "t",
) + SAMPLE_LABELS_CSV_FIELDNAMES


def sample_label_row_from_metrics(
    m: Dict[str, Any],
    sample_id: int,
    run_seed: Optional[int],
) -> Dict[str, Any]:
    """
    由 ``compute_sample_metrics`` 回傳的 dict 組出與 ``compute_sample_label_row`` 相同鍵的一列
    （不含 ``source_name``；由呼叫端填入）。可用於已算過 metrics、手邊沒有 labels 陣列的情境。
    """
    llr = m["Largest_Log_Ratio"]
    llr_store = round(float(llr), 6) if llr >= 0 else -1.0

    return {
        "id": sample_id,
        "seed": "" if run_seed is None else int(run_seed),
        "category": m["Scorer_Category"],
        "is_main_trunk_broken": 1 if m["Is_Main_Trunk_Broken"] else 0,
        "is_broken": 1 if m["Is_Broken"] else 0,
        "largest_log_ratio": llr_store,
        "mass": int(m["Mass"]),
        "height": int(m["Height"]),
        "base_connected_ratio": round(float(m["Base_Connected_Ratio"]), 6),
        "base_connected_size": int(m["Base_Connected_Size"]),
        "log_size": int(m["Log_Size"]),
        "leaf_size": int(m["Leaf_Size"]),
        "occupancy_non_air": round(float(m["Occupancy_Non_Air"]), 6),
        "occupancy_log": round(float(m["Occupancy_Log"]), 6),
        "occupancy_leaf": round(float(m["Occupancy_Leaf"]), 6),
        "components_non_air": int(m["Components_Non_Air"]),
        "components_log": int(m["Components_Log"]),
        "components_leaf": int(m["Components_Leaf"]),
    }


def compute_sample_label_row(
    labels: np.ndarray,
    sample_id: int,
    run_seed: Optional[int],
) -> Dict[str, Any]:
    """
    由離散 labels 計算寫入 per-sample label CSV 的一列（不含 ``source_name``；由呼叫端填入）。
    """
    return sample_label_row_from_metrics(compute_sample_metrics(labels), sample_id, run_seed)


def write_sample_labels_csv(
    rows: List[Dict[str, Any]],
    path: str,
) -> None:
    """寫入 ``sample_labels.csv``；欄位順序見 ``SAMPLE_LABELS_CSV_FIELDNAMES``。"""
    fieldnames = list(SAMPLE_LABELS_CSV_FIELDNAMES)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)


def write_dynamics_label_trace_csv(
    trace_data: List[Dict[str, Any]],
    path: str,
    *,
    run_seed: Optional[int],
    save_npz: bool,
    save_track_projections: bool,
) -> None:
    """
    寫入 ``dynamics_label_trace.csv``：每個追蹤點一列，欄位為 ``DYNAMICS_LABEL_TRACE_CSV_FIELDNAMES``。

    ``trace_data`` 元素須與 ``eval_diffusion_model`` 軌跡列相同（含 0-based ``sample_idx``、
    ``step_idx``、``t`` 及 ``compute_sample_metrics`` 風格鍵名）。

    ``source_name``：若有寫出對應軌跡檔，為相對路徑
    ``dynamics_track_npz/dynamics_sample_XXX_step_YYYY_t_ZZZZ.npz`` 或
    ``dynamics_track_projections/...png``；否則空字串。
    """
    fieldnames = list(DYNAMICS_LABEL_TRACE_CSV_FIELDNAMES)
    skip = {"sample_idx", "step_idx", "t"}
    sorted_trace = sorted(trace_data, key=lambda x: (x.get("sample_idx", 0), x.get("step_idx", 0)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for tr in sorted_trace:
            sid0 = int(tr.get("sample_idx", 0))
            sid1 = sid0 + 1
            step_idx = int(tr.get("step_idx", 0))
            t_int = int(tr.get("t", 0))
            m = {k: v for k, v in tr.items() if k not in skip}
            base = sample_label_row_from_metrics(m, sample_id=sid1, run_seed=run_seed)
            if save_npz:
                src = f"dynamics_track_npz/dynamics_sample_{sid1:03d}_step_{step_idx:04d}_t_{t_int:04d}.npz"
            elif save_track_projections:
                src = f"dynamics_track_projections/dynamics_sample_{sid1:03d}_step_{step_idx:04d}_t_{t_int:04d}.png"
            else:
                src = ""
            row = {"sample_idx": sid1, "step_idx": step_idx, "t": t_int, **base, "source_name": src}
            w.writerow({k: row.get(k, "") for k in fieldnames})


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
