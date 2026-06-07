#!/usr/bin/env python3
"""
離散 voxel 標籤 [Z,Y,X] → 三視圖 PNG（木 > 葉 > 空氣 的 priority max projection）。

於函式內 import pyplot，讓呼叫端可先執行 matplotlib.use("Agg")。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_labels_and_projections(
    labels: np.ndarray,
    out_png: str,
    title_suffix: str = "",
    exp_name: str = "",
) -> None:
    """
    Save 3-view projections from discrete labels (no softmax).
    Priority along ray: wood (1) > leaf (2) > air (0).

    Args:
        labels: [Z,Y,X] uint8 (or int), values in {0,1,2}
        out_png: output PNG path
        title_suffix: appended to each subplot title after Z/Y/X
        exp_name: if non-empty, shown as figure suptitle
    """
    import matplotlib.pyplot as plt

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    def priority_max(arr: np.ndarray, axis: int) -> np.ndarray:
        priority = np.where(arr == 1, 3, np.where(arr == 2, 2, 1))
        max_priority = priority.max(axis=axis)
        return np.where(max_priority == 3, 1, np.where(max_priority == 2, 2, 0))

    max_z = priority_max(labels, axis=0)
    max_y = priority_max(labels, axis=1)
    max_x = priority_max(labels, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(max_z)
    axes[0].set_title("Z" + title_suffix)
    axes[1].imshow(max_y)
    axes[1].set_title("Y" + title_suffix)
    axes[2].imshow(max_x)
    axes[2].set_title("X" + title_suffix)
    for ax in axes:
        ax.axis("off")
    if exp_name:
        fig.suptitle(exp_name, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
