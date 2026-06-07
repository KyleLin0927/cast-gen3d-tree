#!/usr/bin/env python3
"""
畫「介入時間窗 × guidance_scale」的 broken rate 圖。

預設輸出:
- 2D Heatmap (可切換 grouped bar)

色彩指標定義:
- 取每個實驗的 broken rate 欄位 (預設 broken_flag_is_broken__broken_rate_pct)
- 在相同 (time_window, guidance_scale) 下取平均
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SUCCESS_COL = "broken_flag_is_broken__broken_rate_pct"
DEFAULT_GUIDANCE_COL = "guidance_scale"
DEFAULT_T_START_COL = "guidance_t_start"
DEFAULT_T_END_COL = "guidance_t_end"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time-window heatmap/grouped-bar for broken rate."
    )
    parser.add_argument("--summary_csv", required=True, help="aggregate_summaries 輸出 CSV")
    parser.add_argument("--meta_csv", required=True, help="aggregate_metadata 輸出 CSV")
    parser.add_argument("--out_dir", required=True, help="輸出圖片路徑（副檔名決定格式）")

    parser.add_argument(
        "--chart_type",
        choices=["heatmap", "bar"],
        default="heatmap",
        help="圖表形式",
    )
    parser.add_argument(
        "--success_col",
        default=DEFAULT_SUCCESS_COL,
        help="broken rate 欄位（百分比，0~100）",
    )
    parser.add_argument(
        "--exp_col_summary",
        default="Experiment_Name",
        help="summary 內的實驗名稱欄位",
    )
    parser.add_argument(
        "--exp_col_meta",
        default="exp_name",
        help="meta 內的實驗名稱欄位",
    )
    parser.add_argument(
        "--guidance_col",
        default=DEFAULT_GUIDANCE_COL,
        help="meta 內 guidance scale 欄位",
    )
    parser.add_argument(
        "--t_start_col",
        default=DEFAULT_T_START_COL,
        help="meta 內 time-window 起點欄位",
    )
    parser.add_argument(
        "--t_end_col",
        default=DEFAULT_T_END_COL,
        help="meta 內 time-window 終點欄位",
    )
    parser.add_argument(
        "--window_order",
        nargs="*",
        default=["1000-500", "900-400", "800-300", "700-200", "600-100", "500-0"],
        help="X 軸時間窗順序（可不填，會用預設）",
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="heatmap 格子顯示數值",
    )
    return parser.parse_args()


def _assert_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少欄位: {missing}")


def _plot_heatmap(pivot_df: pd.DataFrame, output: Path, annot: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    values = pivot_df.values.astype(float)
    im = ax.imshow(values, aspect="auto", cmap="inferno", vmin=0.0, vmax=100.0)

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_yticklabels([f"w={v:g}" for v in pivot_df.index])

    ax.set_xlabel("Intervention Time Window (t_start-t_end)")
    ax.set_ylabel("Guidance Scale (w)")

    if annot:
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                v = values[r, c]
                if np.isnan(v):
                    text = "-"
                else:
                    text = f"{v:.1f}"
                ax.text(c, r, text, ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Broken Rate (%)")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_grouped_bar(pivot_df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(pivot_df.columns))
    n_groups = max(len(pivot_df.index), 1)
    width = 0.8 / n_groups

    for idx, w in enumerate(pivot_df.index):
        offset = (idx - (n_groups - 1) / 2) * width
        y = pivot_df.loc[w].values.astype(float)
        ax.bar(x + offset, y, width=width, label=f"w={w:g}")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.columns, rotation=20, ha="right")
    ax.set_xlabel("Intervention Time Window (t_start-t_end)")
    ax.set_ylabel("Broken Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", ncol=2)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = Path(args.out_dir)
    if output_path.suffix == "":
        raise ValueError("`--out_dir` 需要副檔名，例如 .png / .pdf / .svg")

    summary_df = pd.read_csv(args.summary_csv, sep=None, engine="python")
    meta_df = pd.read_csv(args.meta_csv, sep=None, engine="python")

    _assert_columns(
        summary_df,
        [args.exp_col_summary, args.success_col],
        "summary_csv",
    )
    _assert_columns(
        meta_df,
        [args.exp_col_meta, args.guidance_col, args.t_start_col, args.t_end_col],
        "meta_csv",
    )

    merged = summary_df.merge(
        meta_df[
            [args.exp_col_meta, args.guidance_col, args.t_start_col, args.t_end_col]
        ].rename(columns={args.exp_col_meta: args.exp_col_summary}),
        on=args.exp_col_summary,
        how="inner",
    )
    if merged.empty:
        raise ValueError("summary 與 meta 合併後為空，請檢查實驗名稱欄位是否對齊")

    merged["success_rate"] = pd.to_numeric(merged[args.success_col], errors="coerce")
    merged["guidance_value"] = pd.to_numeric(merged[args.guidance_col], errors="coerce")
    merged["window_label"] = (
        pd.to_numeric(merged[args.t_start_col], errors="coerce").astype("Int64").astype(str)
        + "-"
        + pd.to_numeric(merged[args.t_end_col], errors="coerce").astype("Int64").astype(str)
    )

    merged["broken_rate"] = merged["success_rate"]

    plot_df = merged.dropna(subset=["guidance_value", "window_label", "broken_rate"])
    if plot_df.empty:
        raise ValueError("可繪圖資料為空，請確認欄位資料是否為數值")

    agg_df = (
        plot_df.groupby(["guidance_value", "window_label"], as_index=False)["broken_rate"]
        .mean()
    )

    pivot = agg_df.pivot(
        index="guidance_value",
        columns="window_label",
        values="broken_rate",
    )

    ordered_cols = [w for w in args.window_order if w in pivot.columns]
    extra_cols = [c for c in pivot.columns if c not in ordered_cols]
    pivot = pivot.reindex(columns=ordered_cols + sorted(extra_cols))
    pivot = pivot.sort_index()

    if args.chart_type == "heatmap":
        _plot_heatmap(pivot, output_path, annot=args.annot)
    else:
        _plot_grouped_bar(pivot, output_path)

    print(f"[OK] 圖表已輸出: {output_path}")


if __name__ == "__main__":
    main()
