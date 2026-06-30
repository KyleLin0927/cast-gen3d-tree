#!/usr/bin/env python3
"""
正交切片蒙太奇（orthoslice montage）：把單一 voxel 樣本沿某軸切成一排 2D 截面平鋪。

用途：分辨「中空的真實結構」 vs 「實心團塊（blob）」。三視圖（正交投影）會把深度壓掉、
把密度與結構混為一談，無法回答「這是椅子，還是 densify 出來的 blob」。逐層切片則直接
顯示截面：
  - 真椅子：低層是幾個小點（腳）、座面高度才填滿成一塊、椅背是一條線；多數截面「稀疏/細環」。
  - 實心 blob：每一層都是填滿的塊。

座標慣例（與 ``voxel_label_projections`` / ``voxel_sample_metrics`` 一致）：
  labels 形狀為 ``[Z, Y, X]``，其中 **Y（axis=1）為高度（上）**。
  因此預設沿 ``axis=1`` 切，得到一疊「水平截面（X–Z 平面）」，最適合看 腳→座→背 的結構。
  若想沿 Z 或 X 切，傳 ``axis=0`` 或 ``axis=2``。

著色與圖例規則與 ``voxel_label_projections`` 共用（``format_component_stats_line``、
``add_component_legend``）：
  - "component"（預設）：實體（label 1）以 26-connectivity 上色（main body 黃、fragments 紅）、
    air 深色。label 2 若存在會著色但不進圖例。浮空碎塊以警示色出現在對應層。
  - "occupancy"：純佔據（佔據亮 / air 深），最乾淨地讀密度。

於函式內 import pyplot，呼叫端可先執行 ``matplotlib.use("Agg")``。
本模組只依賴 numpy（+ 選用 scipy/matplotlib），不 import train_unet_diffusion，故不需 torch、
也不會觸發 voxel_sample_metrics 的循環匯入。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:  # 套件脈絡（from utils.voxel_orthoslices import ...）
    from .voxel_label_projections import (
        DISPLAY_AIR,
        DISPLAY_COLORS,
        HAS_SCIPY,
        STATS_SCIPY_MISSING,
        _wood_component_volume,
        add_component_legend,
        format_component_stats_line,
    )
    from .voxel_npz_io import load_voxel_npz
except ImportError:  # 允許 `python utils/voxel_orthoslices.py ...` 直接執行
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.voxel_label_projections import (
        DISPLAY_AIR,
        DISPLAY_COLORS,
        HAS_SCIPY,
        STATS_SCIPY_MISSING,
        _wood_component_volume,
        add_component_legend,
        format_component_stats_line,
    )
    from utils.voxel_npz_io import load_voxel_npz

_AXIS_NAME = {0: "Z", 1: "Y", 2: "X"}

# occupancy 模式：air 與 component 模式一致（共用 DISPLAY_COLORS）
_OCC_AIR = DISPLAY_COLORS[DISPLAY_AIR]
_OCC_FILLED = np.array([0.85, 0.85, 0.88], dtype=np.float32)


def _coerce_labels(arr: np.ndarray) -> np.ndarray:
    """接受 [Z,Y,X] 離散標籤；若給 4D（含 channel/機率）則沿最小軸 argmax 還原成標籤。"""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        ch_axis = int(np.argmin(arr.shape))  # channel 通常是最小維（如 3）
        return np.argmax(arr, axis=ch_axis).astype(np.uint8)
    raise ValueError(f"預期 3D [Z,Y,X] 標籤（或 4D 含 channel），但收到 shape={arr.shape}")


def _build_rgb_volume(labels: np.ndarray, color_mode: str):
    """回傳 (rgb_vol[Z,Y,X,3], use_component_legend, comp_info)。"""
    if color_mode not in ("component", "occupancy"):
        raise ValueError(f"color_mode 必須是 'component' 或 'occupancy'，收到 {color_mode!r}")
    if color_mode == "component" and HAS_SCIPY:
        voxel_class, n_comp, llr = _wood_component_volume(labels)
        return DISPLAY_COLORS[voxel_class], True, (n_comp, llr)
    occupied = labels != 0
    rgb_vol = np.where(occupied[..., None], _OCC_FILLED, _OCC_AIR).astype(np.float32)
    return rgb_vol, False, None


def _select_slice_indices(n, occ_per_slice, max_slices, skip_empty):
    """挑要畫哪些層：可略過全空層、可均勻取樣到 max_slices 張。"""
    idx = list(range(n))
    if skip_empty:
        nonempty = [i for i in idx if occ_per_slice[i] > 0]
        idx = nonempty if nonempty else idx  # 全空就照畫，避免空圖
    if max_slices is not None and len(idx) > max_slices:
        sel = np.linspace(0, len(idx) - 1, max_slices).round().astype(int)
        idx = [idx[i] for i in sorted(set(sel.tolist()))]
    return idx


def save_orthoslice_montage(
    labels: np.ndarray,
    out_png: str,
    *,
    axis: int = 1,
    color_mode: str = "component",
    max_slices: int | None = None,
    ncols: int | None = None,
    skip_empty: bool = False,
    annotate: bool = True,
    title_suffix: str = "",
    exp_name: str = "",
) -> None:
    """
    將單一 voxel 樣本沿 ``axis`` 切片，平鋪成蒙太奇 PNG。

    Args:
        labels: ``[Z,Y,X]`` 離散標籤（0=air, 1=solid, 2=optional）；亦接受 4D（含 channel，將自動 argmax）。
        out_png: 輸出 PNG 路徑。
        axis: 切片軸。0=Z、1=Y（高度，預設，得水平截面）、2=X。
        color_mode: "component"（連通塊上色，預設）或 "occupancy"（純佔據）。
        max_slices: 該軸層數過多時均勻取樣到這麼多張；None=全畫（32³ → 32 張）。
        ncols: 蒙太奇欄數；None 時自動（≤8 欄）。
        skip_empty: 略過完全沒有佔據的切片。
        annotate: 每張小圖標上「軸=index（該層佔據數）」。
        title_suffix / exp_name: 子圖標題附加字串 / 圖主標題。
    """
    import matplotlib.pyplot as plt

    if axis not in (0, 1, 2):
        raise ValueError(f"axis 必須是 0/1/2，收到 {axis}")

    labels = _coerce_labels(labels)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    rgb_vol, use_component_legend, comp_info = _build_rgb_volume(labels, color_mode)

    n = labels.shape[axis]
    occupied = labels != 0
    other_axes = tuple(a for a in (0, 1, 2) if a != axis)
    occ_per_slice = occupied.sum(axis=other_axes)  # 每層佔據數（沿其餘兩軸加總）

    idx = _select_slice_indices(n, occ_per_slice, max_slices, skip_empty)
    nsl = len(idx)

    if ncols is None:
        ncols = min(8, nsl)
    ncols = max(1, ncols)
    nrows = math.ceil(nsl / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.25, nrows * 1.25), squeeze=False
    )
    axis_name = _AXIS_NAME[axis]

    for cell, i in enumerate(idx):
        r, c = divmod(cell, ncols)
        ax = axes[r][c]
        sl = np.take(rgb_vol, i, axis=axis)  # [.,.,3]
        ax.imshow(sl, interpolation="nearest")
        if annotate:
            ax.set_title(f"{axis_name}={i} ({int(occ_per_slice[i])})", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for cell in range(nsl, nrows * ncols):  # 關掉多出來的空格子
        r, c = divmod(cell, ncols)
        axes[r][c].axis("off")

    if use_component_legend:
        add_component_legend(fig, bbox_to_anchor=(0.5, -0.01))

    occ = float(occupied.mean())
    stats_line = f"occupancy(non-air): {occ:.3f}"
    if comp_info is not None:
        n_comp, llr = comp_info
        stats_line += f" | {format_component_stats_line(n_comp, llr)}"
    elif color_mode == "component" and not HAS_SCIPY:
        stats_line += f" | {STATS_SCIPY_MISSING}"

    head = f"orthoslices along {axis_name} (n={nsl}/{n})"
    if title_suffix:
        head += title_suffix
    suptitle_parts = [p for p in (exp_name, head, stats_line) if p]
    fig.suptitle("\n".join(suptitle_parts), fontsize=10, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_orthoslice_montage_from_npz(npz_path, out_png: str, **kwargs) -> None:
    """便利包裝：從 ``.npz`` 載入 voxel 後產生切片蒙太奇（kwargs 同 save_orthoslice_montage）。"""
    labels = load_voxel_npz(npz_path)
    save_orthoslice_montage(labels, out_png, **kwargs)


def projection_png_to_orthoslice_png(projection_png: str) -> str:
    """
    由三視圖 PNG 路徑推導平行的 orthoslice 路徑。

    若路徑含 ``projections`` 目錄段，改為 ``orthoslices``；否則在同目錄加 ``_orthoslices`` 後綴。
    """
    p = Path(projection_png)
    if "projections" in p.parts:
        new_parts = ["orthoslices" if part == "projections" else part for part in p.parts]
        return str(Path(*new_parts))
    return str(p.with_name(f"{p.stem}_orthoslices{p.suffix}"))


def save_labels_projections_and_orthoslices(
    labels: np.ndarray,
    out_projection_png: str,
    *,
    title_suffix: str = "",
    exp_name: str = "",
    out_orthoslice_png: str | None = None,
) -> None:
    """
    一次寫入三視圖 PNG 與平行的 orthoslice 蒙太奇。

    ``out_orthoslice_png`` 省略時由 ``projection_png_to_orthoslice_png`` 推導。
    """
    try:
        from .voxel_label_projections import save_labels_and_projections
    except ImportError:
        from utils.voxel_label_projections import save_labels_and_projections

    save_labels_and_projections(
        labels,
        out_projection_png,
        title_suffix=title_suffix,
        exp_name=exp_name,
    )
    ortho_png = (
        out_orthoslice_png
        if out_orthoslice_png is not None
        else projection_png_to_orthoslice_png(out_projection_png)
    )
    save_orthoslice_montage(
        labels,
        ortho_png,
        title_suffix=title_suffix,
        exp_name=exp_name,
    )


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(description="voxel 正交切片蒙太奇（orthoslice montage）")
    ap.add_argument("npz", help="輸入 .npz（含 voxel 標籤）")
    ap.add_argument("out_png", help="輸出 PNG 路徑")
    ap.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                    help="切片軸：0=Z、1=Y（高度，預設）、2=X")
    ap.add_argument("--color_mode", default="component", choices=["component", "occupancy"])
    ap.add_argument("--max_slices", type=int, default=None)
    ap.add_argument("--ncols", type=int, default=None)
    ap.add_argument("--skip_empty", action="store_true")
    ap.add_argument("--no_annotate", action="store_true")
    args = ap.parse_args()

    save_orthoslice_montage_from_npz(
        args.npz, args.out_png,
        axis=args.axis, color_mode=args.color_mode,
        max_slices=args.max_slices, ncols=args.ncols,
        skip_empty=args.skip_empty, annotate=not args.no_annotate,
    )
    print(f"saved: {args.out_png}")
