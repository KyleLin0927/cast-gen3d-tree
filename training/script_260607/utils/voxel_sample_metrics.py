#!/usr/bin/env python3
"""
單樣本 16³ voxel 評估指標（供 eval_16_voxel_diffusion、generate_16_voxel_diffusion 等腳本重用）。

依賴同層 ``train_unet_diffusion`` 中的連通性與佔用率輔助函數（請勿在 ``train_unet_diffusion``
載入過程中經由 ``utils`` 套件根 ``__init__`` 間接 import 本模組，否則會造成循環匯入）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from scipy.ndimage import label as _ndi_label

    _HAS_SCIPY_LOCAL = True
except Exception:  # pragma: no cover
    _HAS_SCIPY_LOCAL = False

_STRUCT_26 = np.ones((3, 3, 3), dtype=np.int32)

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
        f"Failed to import metric helpers from train_unet_diffusion: {e}. "
        "If this mentions a circular import, avoid importing utils.export_csv (or the whole "
        "utils package in a way that loads it) before train_unet_diffusion has finished loading."
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


# speckle 過濾門檻：分類連通度時，忽略小於此體素數的 wood 碎片。0＝關閉（行為與原本一致）。
# 由各入口腳本的 --min_component_voxels 旗標在 main() 開頭呼叫 set_min_component_voxels() 設定。
_MIN_COMPONENT_VOXELS = 0


def set_min_component_voxels(k: int) -> None:
    """設定 speckle 過濾門檻 k（忽略 < k 顆的 wood 連通塊）。0 = 關閉。"""
    global _MIN_COMPONENT_VOXELS
    try:
        _MIN_COMPONENT_VOXELS = max(0, int(k))
    except (TypeError, ValueError):
        _MIN_COMPONENT_VOXELS = 0


def get_min_component_voxels() -> int:
    """目前的 speckle 過濾門檻。"""
    return _MIN_COMPONENT_VOXELS


def _effective_log_components(labels: np.ndarray, min_voxels: int, raw_components: int) -> int:
    """
    忽略小於 ``min_voxels`` 顆的 wood 連通塊（碎片）後，剩下的「有效」連通塊數。
    ``min_voxels<=0`` 或無 scipy 時，直接回傳 ``raw_components``（即原始計數）。
    """
    if min_voxels <= 0 or not _HAS_SCIPY_LOCAL:
        return raw_components
    mask = (labels == 1).astype(np.int32)
    if int(mask.sum()) == 0:
        return 0
    lab, n = _ndi_label(mask, structure=_STRUCT_26)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]  # 去掉背景 (label 0)
    return int((sizes >= int(min_voxels)).sum())


def classify_scorer_category(
    log_components: int,
    total_log_size: int,
) -> str:
    """
    Category-agnostic 分類（不依賴地板 / trunk，適用任意 3D 物件，如 ShapeNet 椅子）：
    1) neg_float: 沒有任何 wood voxel（total_log_size==0；空結構 / 退化樣本）
    2) positive : wood 恰好形成 1 個連通塊（log_components==1；Absolute Connectivity）
    3) neg_hard : log_components==2
    4) neg_easy : log_components>=3

    註：保留原 4 個類別常數名（positive/neg_float/neg_easy/neg_hard），讓 generate 分桶與
    scorer 目錄結構不必更動；只是 neg_float 從「沒碰到地板」改為「完全沒有結構」。
    """
    if total_log_size == 0:
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
    計算單個樣本的所有指標（category-agnostic：連通度以「最大 wood 連通塊」與「連通塊數」
    定義，不依賴地板，適用任意解析度的立方體，如 16³ / 32³）。

    Args:
        labels: [D, H, W] numpy array with class labels (0=air, 1=log/occupancy, 2=leaf)；
                任意立方體尺寸皆可（ShapeNet 為 32³）。

    Returns:
        dict，鍵名含（為相容下游，鍵名沿用舊名，但語意改為 category-agnostic）：
            - Is_Main_Trunk_Broken (= 非單一連通塊), Is_Broken (= wood 斷成多塊)
            - Base_Connected_Size (= 最大連通塊大小), Base_Connected_Ratio (= 最大連通塊比例)
            - Is_Main_Trunk_Broken, Is_Broken
            - Mass, Height, Log_Size, Leaf_Size
            - Base_Connected_Size, Total_Log_Size, Base_Connected_Ratio
            - Largest_Log_Ratio
            - Log_AABB_Span_X, Log_AABB_Span_Y, Log_AABB_Span_Z
            - Log_AABB_Volume, Log_BBO (= Log_Size / Log_AABB_Volume)
            - Scorer_Category: positive | neg_float | neg_easy | neg_hard
            - Occupancy_Non_Air, Occupancy_Log, Occupancy_Leaf
            - Components_Non_Air, Components_Log, Components_Leaf
        （ID 由呼叫端另行寫入，例如 eval 腳本。）
    """
    occ_rates = compute_occupancy_rates(labels)
    comp_counts = compute_component_counts_26neighbor(labels)
    largest_log_ratio = compute_largest_log_component_ratio(labels)

    total_log = int((labels == 1).sum())
    log_components = int(comp_counts["log"])  # 原始連通塊數（含雜點），仍寫入 Components_Log

    # === Category-agnostic 連通度 ===
    # 以「最大 wood 連通塊」取代原本「連到地板(Y=0)的塊」。
    # largest_log_ratio = 最大連通塊 / 全部 wood，落在 [0,1]（無 wood 時 0；無 scipy 時 -1）。
    llr = largest_log_ratio if largest_log_ratio >= 0 else 0.0
    base_connected_ratio = llr                       # 重新定義：最大連通塊比例（scorer 的 y_ratio）
    base_sz = int(round(llr * total_log))            # 重新定義：最大連通塊大小

    # speckle 容忍：若 --min_component_voxels>0，分類時忽略小碎片，只看「有效連通塊數」。
    # 預設 0＝關閉，effective == log_components（行為不變）。Components_Log 仍記原始計數。
    effective_components = _effective_log_components(labels, _MIN_COMPONENT_VOXELS, log_components)

    is_connected = (total_log > 0 and effective_components == 1)  # 單一(有效)連通塊＝結構完整
    is_broken = bool(total_log > 0 and effective_components > 1)  # 斷成多塊
    is_main_trunk_broken = bool(not is_connected)                # 非單一連通塊即視為主結構斷裂

    scorer_category = classify_scorer_category(
        log_components=effective_components,
        total_log_size=total_log,
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

    log_coords = np.argwhere(labels == 1)
    if len(log_coords) > 0:
        min_coords = log_coords.min(axis=0)
        max_coords = log_coords.max(axis=0)
        log_aabb_dims = max_coords - min_coords + 1
        log_aabb_span_x = int(log_aabb_dims[0])
        log_aabb_span_y = int(log_aabb_dims[1])
        log_aabb_span_z = int(log_aabb_dims[2])
        log_aabb_volume = int(np.prod(log_aabb_dims))
        log_bbo = (float(log_size) / float(log_aabb_volume)) if log_aabb_volume > 0 else 0.0
    else:
        log_aabb_span_x = 0
        log_aabb_span_y = 0
        log_aabb_span_z = 0
        log_aabb_volume = 0
        log_bbo = 0.0

    return {
        "Is_Main_Trunk_Broken": is_main_trunk_broken,
        "Is_Broken": is_broken,
        "Mass": mass,
        "Height": height,
        "Log_Size": log_size,
        "Leaf_Size": leaf_size,
        "Base_Connected_Size": base_sz,
        "Total_Log_Size": total_log,
        "Base_Connected_Ratio": base_connected_ratio,
        "Largest_Log_Ratio": largest_log_ratio if largest_log_ratio >= 0 else -1.0,
        "Log_AABB_Span_X": log_aabb_span_x,
        "Log_AABB_Span_Y": log_aabb_span_y,
        "Log_AABB_Span_Z": log_aabb_span_z,
        "Log_AABB_Volume": log_aabb_volume,
        "Log_BBO": log_bbo,
        "Scorer_Category": scorer_category,
        "Occupancy_Non_Air": occ_rates["non_air"],
        "Occupancy_Log": occ_rates["log"],
        "Occupancy_Leaf": occ_rates["leaf"],
        "Components_Non_Air": comp_counts["non_air"],
        "Components_Log": comp_counts["log"],
        "Components_Leaf": comp_counts["leaf"],
    }
