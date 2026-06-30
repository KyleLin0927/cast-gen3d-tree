#!/usr/bin/env python3
"""
離散 voxel 標籤 [Z,Y,X] → 三視圖 PNG。

實體（label==1）以 26-connectivity 分連通塊後上色：最大塊一色、其餘警示色；
空氣維持背景色。priority max projection 沿射線取最高優先級，直接對應
largest_log_ratio / #components 指標（CSV 欄位名稱未改）。

於函式內 import pyplot，讓呼叫端可先執行 matplotlib.use("Agg")。
圖例與統計列字串由本模組集中定義，``voxel_orthoslices`` 共用同一套規則。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import label as ndimage_label

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 投影後的顯示類別（數值愈大，沿射線優先級愈高）
DISPLAY_AIR = 0
DISPLAY_LEAF = 1
DISPLAY_FRAGMENT = 2
DISPLAY_LARGEST = 3

# RGB in [0, 1]；最大 wood 用與原本 imshow(0/1/2) 預設 colormap 中 label==1 相近的黃色
WOOD_LARGEST_COLOR = np.array([0.94, 0.75, 0.19], dtype=np.float32)

DISPLAY_COLORS = np.array(
    [
        [0.10, 0.10, 0.12],  # air
        [0.50, 0.72, 0.42],  # leaf
        [0.92, 0.28, 0.22],  # other wood (warning)
        WOOD_LARGEST_COLOR,  # largest wood
    ],
    dtype=np.float32,
)

DISPLAY_PRIORITY = np.array([1, 2, 3, 4], dtype=np.uint8)

# --- Shared PNG UI (also used by voxel_orthoslices) ---
LEGEND_LABEL_MAIN_BODY = "main body"
LEGEND_LABEL_FRAGMENTS = "fragments"
LEGEND_LABEL_AIR = "air"
LEGEND_NCOL = 3
LEGEND_FONTSIZE = 8
STATS_SCIPY_MISSING = "connectivity coloring unavailable (scipy missing)"


def format_component_stats_line(n_comp: int, llr: float) -> str:
    """Suptitle stats for 26-connected component coloring (maps to CSV largest_log_ratio)."""
    if n_comp > 0:
        return f"components: {n_comp} | largest_part_ratio: {llr:.3f}"
    return "components: 0"


def make_component_legend_patches():
    """Legend patches: main body / fragments / air (label 2 omitted)."""
    from matplotlib.patches import Patch

    return [
        Patch(
            facecolor=DISPLAY_COLORS[DISPLAY_LARGEST],
            edgecolor="none",
            label=LEGEND_LABEL_MAIN_BODY,
        ),
        Patch(
            facecolor=DISPLAY_COLORS[DISPLAY_FRAGMENT],
            edgecolor="none",
            label=LEGEND_LABEL_FRAGMENTS,
        ),
        Patch(
            facecolor=DISPLAY_COLORS[DISPLAY_AIR],
            edgecolor="none",
            label=LEGEND_LABEL_AIR,
        ),
    ]


def add_component_legend(fig, *, bbox_to_anchor: tuple[float, float] = (0.5, -0.02)) -> None:
    """Attach the standard 3-item component legend below ``fig``."""
    fig.legend(
        handles=make_component_legend_patches(),
        loc="lower center",
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
    )


def _wood_component_volume(labels: np.ndarray) -> tuple[np.ndarray, int, float]:
    """
    將 wood voxel 分成最大塊 vs 其餘碎塊。

    Returns:
        voxel_class: [Z,Y,X] uint8，值為 DISPLAY_* 常數
        n_log_components: wood 連通塊數（0 表示無 wood）
        largest_log_ratio: 最大塊 / 全部 wood
    """
    voxel_class = np.zeros(labels.shape, dtype=np.uint8)
    voxel_class[labels == 2] = DISPLAY_LEAF

    wood_mask = labels == 1
    total_wood = int(wood_mask.sum())
    if total_wood == 0:
        return voxel_class, 0, 0.0

    structure = np.ones((3, 3, 3), dtype=np.int32)
    labeled, n_comp = ndimage_label(wood_mask.astype(np.int32), structure=structure)
    if n_comp == 0:
        return voxel_class, 0, 0.0

    sizes = np.array([(labeled == comp_id).sum() for comp_id in range(1, n_comp + 1)], dtype=np.int64)
    largest_id = int(1 + np.argmax(sizes))
    largest_size = int(sizes[largest_id - 1])
    llr = float(largest_size) / float(total_wood)

    voxel_class[wood_mask & (labeled == largest_id)] = DISPLAY_LARGEST
    voxel_class[wood_mask & (labeled != largest_id)] = DISPLAY_FRAGMENT
    return voxel_class, int(n_comp), llr


def _priority_max_class(voxel_class: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis 做 priority max，回傳該射線上最前景的 DISPLAY_* 類別。"""
    priority = DISPLAY_PRIORITY[voxel_class]
    idx = priority.argmax(axis=axis, keepdims=True)
    return np.take_along_axis(voxel_class, idx, axis=axis).squeeze(axis=axis)


def _class_to_rgb(projected_class: np.ndarray) -> np.ndarray:
    return DISPLAY_COLORS[projected_class]


def save_labels_and_projections(
    labels: np.ndarray,
    out_png: str,
    title_suffix: str = "",
    exp_name: str = "",
) -> None:
    """
    Save 3-view projections from discrete labels (no softmax).

    Solid voxels (label 1) are colored by 26-connected components: main body vs
    warning color for detached fragments. Air (0) uses a dark background. Ray priority:
    main body > fragments > air. Label 2 (if present) is drawn but omitted from the legend.

    Args:
        labels: [Z,Y,X] uint8 (or int), values in {0,1,2}
        out_png: output PNG path
        title_suffix: appended to each subplot title after Z/Y/X
        exp_name: if non-empty, shown as figure suptitle (stats line appended below)
    """
    import matplotlib.pyplot as plt

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)

    stats_line = ""
    if HAS_SCIPY:
        voxel_class, n_comp, llr = _wood_component_volume(labels)
        max_z = _priority_max_class(voxel_class, axis=0)
        max_y = _priority_max_class(voxel_class, axis=1)
        max_x = _priority_max_class(voxel_class, axis=2)
        views = (_class_to_rgb(max_z), _class_to_rgb(max_y), _class_to_rgb(max_x))
        stats_line = format_component_stats_line(n_comp, llr)
    else:
        # scipy 不可用時退回舊的 material-class 投影
        def legacy_priority_max(arr: np.ndarray, axis: int) -> np.ndarray:
            priority = np.where(arr == 1, 3, np.where(arr == 2, 2, 1))
            max_priority = priority.max(axis=axis)
            return np.where(max_priority == 3, 1, np.where(max_priority == 2, 2, 0))

        legacy = (
            legacy_priority_max(labels, axis=0),
            legacy_priority_max(labels, axis=1),
            legacy_priority_max(labels, axis=2),
        )
        legacy_cmap = np.array(
            [
                [0.10, 0.10, 0.12],
                WOOD_LARGEST_COLOR,
                [0.50, 0.72, 0.42],
            ],
            dtype=np.float32,
        )
        views = tuple(legacy_cmap[v] for v in legacy)
        stats_line = STATS_SCIPY_MISSING

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, rgb, axis_name in zip(axes, views, ("Z", "Y", "X")):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(axis_name + title_suffix)
        ax.axis("off")

    add_component_legend(fig)

    suptitle_parts = [p for p in (exp_name, stats_line) if p]
    if suptitle_parts:
        fig.suptitle("\n".join(suptitle_parts), fontsize=11, fontweight="bold", y=1.03)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
