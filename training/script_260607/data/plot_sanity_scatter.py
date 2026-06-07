#!/usr/bin/env python3
"""
從 aggregate_summaries 輸出的 CSV 畫 2D 散佈圖。

特色:
- X 軸可選 BBO 緊湊度或 Z 軸跨度
- Y 軸固定為連通成功率 (%)
- 點大小可由 log_size 控制
- 以雙邊界 (a, b) 畫紅色虛線
- a 左側 / b 右側自動塗淡紅警告區
- 圖檔格式由 --out_dir 副檔名決定 (png / pdf / svg ... 交給 matplotlib)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


X_AXIS_CHOICES = {
    "bbo": ("log_bbo__mean", "BBO Compactness (log_bbo__mean)"),
    "span_z": ("log_aabb_span_z__mean", "Span Z (log_aabb_span_z__mean)"),
}
Y_COLUMN = "broken_flag_is_broken__intact_rate_pct"
Y_LABEL = "Connectivity Success Rate (%)"
SIZE_COLUMN = "log_size__mean"
GUIDANCE_COLUMN = "guidance_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D scatter plot with sanity-check bounds."
    )
    parser.add_argument("--csv", required=True, help="輸入 CSV 路徑")
    parser.add_argument(
        "--out_dir",
        required=True,
        help="輸出圖片路徑，格式由副檔名決定 (例如 .png/.pdf/.svg)",
    )
    parser.add_argument(
        "--x_axis",
        choices=sorted(X_AXIS_CHOICES.keys()),
        default="bbo",
        help="X 軸欄位來源",
    )
    parser.add_argument("--limit_a", type=float, required=True, help="下限 a")
    parser.add_argument("--limit_b", type=float, required=True, help="上限 b")
    parser.add_argument("--label_a", default="-1 sigma", help="下限標線文字")
    parser.add_argument("--label_b", default="+1 sigma", help="上限標線文字")
    parser.add_argument(
        "--warn_side",
        choices=["outside", "left", "right"],
        default="outside",
        help="警告區塗色方向：outside=兩側, left=僅左側, right=僅右側",
    )
    parser.add_argument(
        "--size_scale",
        type=float,
        default=6.0,
        help="點大小縮放倍率 (最終 s = normalized_size * size_scale)",
    )
    parser.add_argument(
        "--mean_value",
        type=float,
        default=None,
        help="X 軸平均數位置。若提供，會畫一條平均數垂直線",
    )
    parser.add_argument(
        "--mean_label",
        default="Mean",
        help="平均數垂直線圖例文字",
    )
    parser.add_argument(
        "--meta_csv",
        default=None,
        help="可選，metadata CSV 路徑；提供後會依 guidance_scale 對樣本點上色",
    )
    return parser.parse_args()


def _normalize_sizes(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([40.0] * len(series), index=series.index)

    valid = numeric.dropna()
    vmin, vmax = valid.min(), valid.max()
    if vmax == vmin:
        normalized = pd.Series([1.0] * len(series), index=series.index)
    else:
        normalized = (numeric - vmin) / (vmax - vmin)
        normalized = normalized.fillna(0.5)

    return 30.0 + normalized * 170.0


def _find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def main() -> None:
    args = parse_args()

    output_path = Path(args.out_dir)
    if output_path.suffix == "":
        raise ValueError("`--out_dir` 需要有副檔名，例如 .png / .pdf / .svg")

    limit_min = min(args.limit_a, args.limit_b)
    limit_max = max(args.limit_a, args.limit_b)
    if limit_min == limit_max:
        raise ValueError("`--limit_a` 與 `--limit_b` 不能相同")

    df = pd.read_csv(args.csv, sep=None, engine="python")

    x_col, x_label = X_AXIS_CHOICES[args.x_axis]
    required = [x_col, Y_COLUMN, SIZE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位: {missing}")

    if args.meta_csv:
        meta_df = pd.read_csv(args.meta_csv, sep=None, engine="python")
        summary_key = _find_first_existing(
            list(df.columns),
            ["Experiment_Name", "exp_name", "source_dir", "filename_prefix"],
        )
        meta_key = _find_first_existing(
            list(meta_df.columns),
            ["exp_name", "Experiment_Name", "source_dir", "filename_prefix"],
        )
        if summary_key is None or meta_key is None:
            raise ValueError(
                "無法對齊 summary 與 meta。請確認兩者至少有一個可對齊欄位，例如 "
                "Experiment_Name / exp_name / source_dir / filename_prefix"
            )
        if GUIDANCE_COLUMN not in meta_df.columns:
            raise ValueError(f"`--meta_csv` 缺少必要欄位: {GUIDANCE_COLUMN}")

        merged = df.merge(
            meta_df[[meta_key, GUIDANCE_COLUMN]].rename(columns={meta_key: summary_key}),
            on=summary_key,
            how="left",
        )
        guidance_values = pd.to_numeric(merged[GUIDANCE_COLUMN], errors="coerce")
        df = merged
    else:
        guidance_values = None

    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[Y_COLUMN], errors="coerce")
    size_values = _normalize_sizes(df[SIZE_COLUMN]) * args.size_scale

    valid_mask = x_values.notna() & y_values.notna()
    if guidance_values is not None:
        valid_mask = valid_mask & guidance_values.notna()
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError("沒有可繪圖的有效資料列（X/Y 全為空或非數值）")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter_kwargs = {
        "x": x_values[valid_mask],
        "y": y_values[valid_mask],
        "s": size_values[valid_mask],
        "alpha": 0.8,
        "edgecolors": "white",
        "linewidths": 0.6,
        "zorder": 3,
        "label": f"Experiments (n={n_valid}, size=wood_size)",
    }
    if guidance_values is not None:
        scatter_kwargs["c"] = guidance_values[valid_mask]
        scatter_kwargs["cmap"] = "viridis"
    else:
        scatter_kwargs["color"] = "tab:blue"

    scatter = ax.scatter(**scatter_kwargs)
    if guidance_values is not None:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("guidance_scale")

    data_x_min = float(x_values[valid_mask].min())
    data_x_max = float(x_values[valid_mask].max())
    span = max(limit_max - limit_min, data_x_max - data_x_min, 1e-6)
    pad = 0.1 * span
    x_plot_min = min(data_x_min, limit_min) - pad
    x_plot_max = max(data_x_max, limit_max) + pad
    ax.set_xlim(x_plot_min, x_plot_max)

    ax.axvline(
        x=limit_min,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Lower Bound ({args.label_a})",
        zorder=2,
    )
    ax.axvline(
        x=limit_max,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Upper Bound ({args.label_b})",
        zorder=2,
    )

    if args.mean_value is not None:
        ax.axvline(
            x=args.mean_value,
            color="tab:blue",
            linestyle=":",
            linewidth=2,
            label=args.mean_label,
            zorder=2,
        )

    if args.warn_side in ("outside", "left"):
        ax.axvspan(
            x_plot_min,
            limit_min,
            color="red",
            alpha=0.1,
            zorder=1,
            label="Warning Area" if args.warn_side == "left" else None,
        )
    if args.warn_side in ("outside", "right"):
        ax.axvspan(
            limit_max,
            x_plot_max,
            color="red",
            alpha=0.1,
            zorder=1,
            label="Warning Area",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax.legend(loc="best", markerscale=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=220)
    plt.close(fig)
    print(f"[OK] 圖表已輸出: {output_path}")


if __name__ == "__main__":
    main()
