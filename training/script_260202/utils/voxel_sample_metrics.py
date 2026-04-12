#!/usr/bin/env python3
"""
單樣本 16³ voxel 評估指標（供 eval_16_voxel_diffusion、generate_16_voxel_diffusion 等腳本重用）。

依賴上一層目錄（與本腳本同層）的 unet_diffusion_16_voxel 中的連通性與佔用率輔助函數。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# utils/ 的上一層即 script_260202（unet_diffusion_16_voxel.py 所在目錄）
_script_260202 = Path(__file__).resolve().parents[1]
if str(_script_260202) not in sys.path:
    sys.path.insert(0, str(_script_260202))

try:
    from train_unet_diffusion import (
        compute_trunk_breakage,
        compute_occupancy_rates,
        compute_component_counts_26neighbor,
        compute_largest_log_component_ratio,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import from unet_diffusion_16_voxel: {e}. "
        "Ensure unet_diffusion_16_voxel.py is in the same directory."
    ) from e

# Scorer 分桶（與 generate_16_voxel_diffusion 一致）
CAT_POSITIVE = "positive"
CAT_NEG_FLOAT = "neg_float"
CAT_NEG_EASY = "neg_easy"
CAT_NEG_HARD = "neg_hard"
ALL_SCORER_CATEGORIES = (
    CAT_POSITIVE,
    CAT_NEG_FLOAT,
    CAT_NEG_EASY,
    CAT_NEG_HARD,
)


def classify_scorer_category(
    base_connected_size: int,
    log_components: int,
) -> str:
    """
    分類依 scorer 設定：
    1) neg_float: 連地板都沒碰到（base_connected_size==0）
    2) positive: 有碰到地板且全樹只有 1 個連通塊（log_components==1；Absolute Connectivity）
    3) neg_hard / neg_easy: 在 base_connected_size>0 且 log_components>1 時，
       log_components==2 為 neg_hard，否則為 neg_easy。
    """
    if base_connected_size == 0:
        return CAT_NEG_FLOAT
    if log_components == 1:
        return CAT_POSITIVE
    if log_components == 2:
        return CAT_NEG_HARD
    return CAT_NEG_EASY


def compute_sample_metrics(
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    計算單個樣本的所有指標。

    Args:
        labels: [16, 16, 16] numpy array with class labels (0=air, 1=log, 2=leaf)

    Returns:
        dict，鍵名含：
            - Is_Main_Trunk_Broken, Is_Broken
            - Mass, Height, Log_Size, Leaf_Size
            - Base_Connected_Size, Total_Log_Size, Base_Connected_Ratio
            - Largest_Log_Ratio
            - Scorer_Category: positive | neg_float | neg_easy | neg_hard
            - Occupancy_Non_Air, Occupancy_Log, Occupancy_Leaf
            - Components_Non_Air, Components_Log, Components_Leaf
        （ID 由呼叫端另行寫入，例如 eval 腳本。）
    """
    trunk_info = compute_trunk_breakage(labels, debug=False)
    occ_rates = compute_occupancy_rates(labels)
    comp_counts = compute_component_counts_26neighbor(labels)
    largest_log_ratio = compute_largest_log_component_ratio(labels)

    base_sz = int(trunk_info["base_connected_size"])
    total_log = int(trunk_info["total_wood_size"])
    base_connected_ratio = (float(base_sz) / float(total_log)) if total_log > 0 else 0.0

    log_components = int(comp_counts["log"])

    scorer_category = classify_scorer_category(
        base_connected_size=base_sz,
        log_components=log_components,
    )

    mass = int((labels != 0).sum())

    non_air_coords = np.argwhere(labels != 0)
    if len(non_air_coords) > 0:
        max_y = non_air_coords[:, 1].max()
        height = max_y + 1
    else:
        height = 0

    log_size = int((labels == 1).sum())
    leaf_size = int((labels == 2).sum())

    return {
        "Is_Main_Trunk_Broken": trunk_info["is_main_trunk_broken"],
        "Is_Broken": trunk_info["is_broken"],
        "Mass": mass,
        "Height": height,
        "Log_Size": log_size,
        "Leaf_Size": leaf_size,
        "Base_Connected_Size": base_sz,
        "Total_Log_Size": total_log,
        "Base_Connected_Ratio": base_connected_ratio,
        "Largest_Log_Ratio": largest_log_ratio if largest_log_ratio >= 0 else -1.0,
        "Scorer_Category": scorer_category,
        "Occupancy_Non_Air": occ_rates["non_air"],
        "Occupancy_Log": occ_rates["log"],
        "Occupancy_Leaf": occ_rates["leaf"],
        "Components_Non_Air": comp_counts["non_air"],
        "Components_Log": comp_counts["log"],
        "Components_Leaf": comp_counts["leaf"],
    }
