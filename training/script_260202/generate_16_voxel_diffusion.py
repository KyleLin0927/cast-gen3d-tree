#!/usr/bin/env python3
"""
單純從 16x16x16 Voxel Diffusion 模型生成樣本（不跑評估指標、動力學、CSV 摘要）。

輸出（預設）：
- projections/{positive,neg_float,neg_easy,neg_hard}/：各類別三視圖 PNG
- npz/{...}/：同上，每檔僅含陣列鍵 ``voxel``（--no_npz 可關閉）
- sample_labels.csv: 每個樣本的 id、seed、分類與 compute_sample_metrics 之完整指標（見下方欄位）
- sample_labels_summary.csv: 全體樣本指標加總與平均、標準差等；並含與
  ``eval_diffusion_model.compute_summary_statistics`` 對齊的扁平鍵（``avg_*``、成功率等），
  方便與 ``simple_summary.csv`` 或 aggregate 流程對照
- 分類定義：
  - neg_float：base_connected_size==0（連地板都沒碰到）
  - positive：base_connected_size>0，且全樹只有 1 個連通塊（log_components==1；Absolute Connectivity）
  - neg_hard：base_connected_size>0，且 log_components==2
  - neg_easy：base_connected_size>0，且 log_components>2
- metadata.csv / metadata_flat.csv：重現用參數
- generate_16_voxel_diffusion_snapshot_YYYYMMDD_HHMMSS.py：執行當下本腳本完整備份

sample_labels.csv 欄位：
- id, seed, category（positive | neg_float | neg_easy | neg_hard）
- is_main_trunk_broken(0/1), is_broken(0/1)
- mass, height, log_size, leaf_size
- base_connected_ratio, base_connected_size, total_log_size, largest_log_ratio（無效時 -1）
- occupancy_non_air, occupancy_log, occupancy_leaf
- components_non_air, components_log, components_leaf
- source_name：相對於 ``--out_dir`` 的 POSIX 路徑，優先對應寫出的 ``npz/<category>/<stem>.npz``；若 ``--no_npz`` 則為 ``projections/<category>/<stem>.png``；兩者皆關則為空字串
（以上對應 utils.voxel_sample_metrics.compute_sample_metrics 回傳之指標，外加 artifact 路徑）

專案目錄即 --out_dir：所有輸出寫入該路徑（會自動建立）。
實驗名稱與輸出檔名前綴（PNG/NPZ 的 ``<prefix>_<id>``）一律為 ``out_dir`` 路徑的最後一層目錄名稱（例如 ``--out_dir ./runs/exp_001`` → ``exp_001``）。

使用方式:
  python generate_16_voxel_diffusion.py --checkpoint path/to/model.pt --out_dir ./my_project --n_samples 50
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
) -> Dict[str, Any]:
    """
    由離散 labels 計算寫入 CSV 的一列；欄位涵蓋 compute_sample_metrics 目前回傳的所有指標。
    """
    m = compute_sample_metrics(labels)
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
    v_llr = llr_all[valid] if bool(valid.any()) else np.array([], dtype=np.float64)

    add_section("Flat summary keys (eval_diffusion_model.compute_summary_statistics)")
    succ_llr = np.array(
        [
            (float(r["largest_log_ratio"]) >= 0.0)
            and bool(np.isclose(float(r["largest_log_ratio"]), 1.0, atol=1e-6))
            for r in rows
        ],
        dtype=np.float64,
    )
    lines.append(["success_rate_pct_LLR_eq_1", f"{100.0 * float(succ_llr.mean()):.4f}"])
    lines.append(["failure_rate_pct_LLR_ne_1", f"{100.0 * float(1.0 - succ_llr.mean()):.4f}"])
    lines.append(["broken_rate_pct_Is_Broken", f"{100.0 * float(broken_flag.mean()):.4f}"])
    lines.append(["avg_mass", f"{float(mass.mean()):.6f}"])
    lines.append(["std_mass", f"{float(mass.std()):.6f}"])
    lines.append(["avg_height", f"{float(height.mean()):.6f}"])
    lines.append(["std_height", f"{float(height.std()):.6f}"])
    lines.append(["avg_log_size", f"{float(log_sz.mean()):.6f}"])
    lines.append(["std_log_size", f"{float(log_sz.std()):.6f}"])
    lines.append(["avg_leaf_size", f"{float(leaf_sz.mean()):.6f}"])
    lines.append(["std_leaf_size", f"{float(leaf_sz.std()):.6f}"])
    lines.append(["avg_base_connected_size", f"{float(bcs.mean()):.6f}"])
    lines.append(["std_base_connected_size", f"{float(bcs.std()):.6f}"])
    mask_tl = tls > 0
    lines.append(["n_samples_total_log_size_gt_0", str(int(mask_tl.sum()))])
    if bool(mask_tl.any()):
        br_over_log = (bcs[mask_tl] / tls[mask_tl]).astype(np.float64)
        lines.append(["avg_base_connected_ratio_total_log_gt_0", f"{float(br_over_log.mean()):.6f}"])
        lines.append(["std_base_connected_ratio_total_log_gt_0", f"{float(br_over_log.std()):.6f}"])
    else:
        lines.append(["avg_base_connected_ratio_total_log_gt_0", "0.000000"])
        lines.append(["std_base_connected_ratio_total_log_gt_0", "0.000000"])
    if v_llr.size > 0:
        lines.append(["avg_largest_log_ratio_valid_only", f"{float(v_llr.mean()):.6f}"])
        lines.append(["std_largest_log_ratio_valid_only", f"{float(v_llr.std()):.6f}"])
    else:
        lines.append(["avg_largest_log_ratio_valid_only", "-1.000000"])
        lines.append(["std_largest_log_ratio_valid_only", "0.000000"])
    lines.append(["avg_occupancy_non_air", f"{float(occ_na.mean()):.6f}"])
    lines.append(["std_occupancy_non_air", f"{float(occ_na.std()):.6f}"])
    lines.append(["avg_occupancy_log", f"{float(occ_lg.mean()):.6f}"])
    lines.append(["std_occupancy_log", f"{float(occ_lg.std()):.6f}"])
    lines.append(["avg_occupancy_leaf", f"{float(occ_lf.mean()):.6f}"])
    lines.append(["std_occupancy_leaf", f"{float(occ_lf.std()):.6f}"])
    lines.append(["avg_components_non_air", f"{float(comp_na.mean()):.6f}"])
    lines.append(["std_components_non_air", f"{float(comp_na.std()):.6f}"])
    lines.append(["avg_components_log", f"{float(comp_lg.mean()):.6f}"])
    lines.append(["std_components_log", f"{float(comp_lg.std()):.6f}"])
    lines.append(["avg_components_leaf", f"{float(comp_lf.mean()):.6f}"])
    lines.append(["std_components_leaf", f"{float(comp_lf.std()):.6f}"])
    lines.append(
        [
            "note_flat_keys",
            "mirrors eval_diffusion_model.compute_summary_statistics; "
            "base_connected_ratio here is mean over samples with total_log_size>0 only",
        ]
    )

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

    add_section("Scorer-style categories (scorer bucketing)")
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


def load_scorer(checkpoint_path: str, device: torch.device) -> nn.Module:
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
    # `train_scorer.py` saves in two possible formats:
    # 1) best/bundled checkpoint: {"model_state_dict": ..., "epoch": ..., ...}
    # 2) periodic checkpoint: plain state_dict (when using torch.save(model.state_dict(), ...))
    if isinstance(scorer_ckpt, dict) and "model_state_dict" in scorer_ckpt:
        state_dict = scorer_ckpt["model_state_dict"]
    else:
        state_dict = scorer_ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Unsupported scorer checkpoint format. Expected dict-like state_dict, "
            f"but got type={type(scorer_ckpt).__name__}"
        )

    # Strict load to catch accidental mismatch; adjust to strict=False only if you see missing keys.
    scorer.load_state_dict(state_dict, strict=True)
    scorer.eval()
    console.print(f"[green]✓[/green] Scorer loaded: {checkpoint_path}")
    if isinstance(scorer_ckpt, dict) and "epoch" in scorer_ckpt:
        console.print(
            f"[green]✓[/green] Scorer checkpoint epoch: {scorer_ckpt.get('epoch', 'unknown')}"
        )
    return scorer


@torch.no_grad()
def sample_guided_voxels(
    denoiser_model: nn.Module,
    scorer_model: nn.Module,
    betas: BetaSchedule,
    shape: Tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 50.0,
    lambda_ratio: float = 10.0,
    t_start: int = 900,
    t_end: int = 400,
    n_steps: Optional[int] = None,
    use_amp: bool = False,
) -> torch.Tensor:
    """
    Guided DDPM sampling where a scorer provides gradient-based guidance.

    Guidance is applied only for timesteps in [min(t_start,t_end), max(t_start,t_end)] (inclusive).
    """
    T = betas.T
    if n_steps is None:
        n_steps = T

    B, C, H, W, D = shape
    if (C, H, W, D) != (3, 16, 16, 16):
        raise ValueError(f"Expected shape=(B,3,16,16,16), got {shape}")

    guidance_lo = int(min(t_start, t_end))
    guidance_hi = int(max(t_start, t_end))
    guidance_lo = max(0, guidance_lo)
    guidance_hi = min(T - 1, guidance_hi)

    # x_T ~ N(0, I)
    x = torch.randn(shape, device=device)

    # Match `sample_voxels()` timestep selection behavior.
    if n_steps < T:
        timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, device=device)

    for t_int_tensor in timesteps:
        t_int = t_int_tensor.item() if isinstance(t_int_tensor, torch.Tensor) else int(t_int_tensor)
        t = torch.full((B,), t_int, device=device, dtype=torch.long)

        # ----------------------------
        # Scorer guidance intervention
        # ----------------------------
        if guidance_scale > 0.0 and guidance_lo <= t_int <= guidance_hi:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                pred_break_logits, pred_ratio = scorer_model(x, t)

                # Minimize break + maximize connectivity ratio.
                energy = pred_break_logits.sum() - lambda_ratio * pred_ratio.sum()
                grad = torch.autograd.grad(energy, x)[0]

                x = x - guidance_scale * grad
                x = x.detach()

        # ----------------------------
        # Standard DDPM reverse step
        # ----------------------------
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            eps_pred = denoiser_model(x, t)

        beta_t = betas.beta[t_int]
        alpha_t = betas.alpha[t_int]
        alpha_bar_t = betas.alpha_bar[t_int]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if t_int > 0:
            alpha_bar_prev = betas.alpha_bar[t_int - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        posterior_mean = coef1 * pred_x0 + coef2 * x

        if t_int > 0:
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x)
            x = posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            x = posterior_mean

    return x


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
    scorer_model: Optional[nn.Module] = None,
    guidance_scale: float = 50.0,
    t_start: int = 900,
    t_end: int = 400,
    guidance_lambda_ratio: float = 10.0,
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
                if scorer_model is not None:
                    x_0 = sample_guided_voxels(
                        denoiser_model=model,
                        scorer_model=scorer_model,
                        betas=betas,
                        shape=(b, 3, 16, 16, 16),
                        device=device,
                        guidance_scale=guidance_scale,
                        lambda_ratio=guidance_lambda_ratio,
                        t_start=t_start,
                        t_end=t_end,
                        n_steps=n_steps,
                        use_amp=use_amp,
                    )
                else:
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

                if save_npz:
                    row["source_name"] = f"npz/{cat}/{stem}.npz"
                elif save_projections and pr:
                    row["source_name"] = f"projections/{cat}/{stem}.png"
                else:
                    row["source_name"] = ""

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
        help=(
            "Project directory (created if missing); all outputs go here. "
            "Last path component is used as experiment name and output filename prefix."
        ),
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sample_verbose",
        action="store_true",
        help="Per-step prints from sample_voxels (slow, noisy)",
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

    scorer_model: Optional[nn.Module] = None
    guidance_lambda_ratio = args.lambda_ratio
    if args.scorer_ckpt:
        scorer_model = load_scorer(args.scorer_ckpt, device)
        console.print(
            f"[green]✓[/green] Guided sampling enabled: scale={args.guidance_scale} "
            f"window t={args.t_start}..{args.t_end} (lambda_ratio={guidance_lambda_ratio})"
        )

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
        "filename_prefix": out_dir_leaf,
        "sample_labels_csv": os.path.join(os.path.abspath(args.out_dir), "sample_labels.csv"),
        "sample_labels_summary_csv": os.path.join(
            os.path.abspath(args.out_dir), "sample_labels_summary.csv"
        ),
        "scorer_ckpt": args.scorer_ckpt if args.scorer_ckpt else "None",
        "guidance_scale": f"{args.guidance_scale:.10g}",
        "guidance_t_start": str(args.t_start),
        "guidance_t_end": str(args.t_end),
        "guidance_lambda_ratio": f"{guidance_lambda_ratio:.10g}",
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
        filename_prefix=out_dir_leaf,
        sample_verbose=args.sample_verbose,
        run_seed=args.seed,
        scorer_model=scorer_model,
        guidance_scale=args.guidance_scale,
        t_start=args.t_start,
        t_end=args.t_end,
        guidance_lambda_ratio=guidance_lambda_ratio,
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
