#!/usr/bin/env python3
"""
第三張圖：樹葉 vs 樹幹 Trade-off（Dual-axis line chart）

用途:
- 固定時間窗 (預設 700-200)
- X 軸: guidance scale (w)，預設刻度 0 ~ 4.5（間隔 0.5）
- 左 Y 軸: 連通成功率 (%)
- 右 Y 軸: 平均樹葉數量 leaf_size__mean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SUCCESS_COL = "broken_flag_is_broken__intact_rate_pct"
DEFAULT_LEAF_COL = "leaf_size__mean"
# X 軸預設：0 ~ 4.5（間隔 0.5）
DEFAULT_GUIDANCE_ORDER = list(np.linspace(0.0, 4.5, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dual-axis trade-off: connectivity success vs leaf size."
    )
    parser.add_argument("--summary_csv", required=True, help="aggregate_summaries 輸出 CSV")
    parser.add_argument("--meta_csv", required=True, help="aggregate_metadata 輸出 CSV")
    parser.add_argument("--out_dir", required=True, help="輸出圖片路徑（副檔名決定格式）")

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
        default="guidance_scale",
        help="meta 內 guidance scale 欄位",
    )
    parser.add_argument(
        "--t_start_col",
        default="guidance_t_start",
        help="meta 內 time-window 起點欄位",
    )
    parser.add_argument(
        "--t_end_col",
        default="guidance_t_end",
        help="meta 內 time-window 終點欄位",
    )
    parser.add_argument(
        "--target_t_start",
        type=int,
        default=700,
        help="固定時間窗起點 (預設 700)",
    )
    parser.add_argument(
        "--target_t_end",
        type=int,
        default=200,
        help="固定時間窗終點 (預設 200)",
    )
    parser.add_argument(
        "--success_col",
        default=DEFAULT_SUCCESS_COL,
        help="左 Y 軸連通成功率欄位 (%)",
    )
    parser.add_argument(
        "--leaf_col",
        default=DEFAULT_LEAF_COL,
        help="右 Y 軸樹葉數量欄位",
    )
    parser.add_argument(
        "--guidance_order",
        nargs="*",
        type=float,
        default=None,
        help="X 軸 guidance scale 順序（未指定時為 0..4.5，間隔 0.5）",
    )
    return parser.parse_args()


def _assert_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少欄位: {missing}")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_dir)
    if out_path.suffix == "":
        raise ValueError("`--out_dir` 需要副檔名，例如 .png / .pdf / .svg")

    summary_df = pd.read_csv(args.summary_csv, sep=None, engine="python")
    meta_df = pd.read_csv(args.meta_csv, sep=None, engine="python")

    _assert_columns(
        summary_df,
        [args.exp_col_summary, args.success_col, args.leaf_col],
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

    t_start = pd.to_numeric(merged[args.t_start_col], errors="coerce")
    t_end = pd.to_numeric(merged[args.t_end_col], errors="coerce")
    filtered = merged[(t_start == args.target_t_start) & (t_end == args.target_t_end)].copy()
    if filtered.empty:
        raise ValueError(
            f"找不到時間窗 {args.target_t_start}-{args.target_t_end} 的資料，請確認 meta 內容"
        )

    filtered["guidance_value"] = pd.to_numeric(filtered[args.guidance_col], errors="coerce")
    filtered["success_rate"] = pd.to_numeric(filtered[args.success_col], errors="coerce")
    filtered["leaf_size"] = pd.to_numeric(filtered[args.leaf_col], errors="coerce")
    filtered = filtered.dropna(subset=["guidance_value", "success_rate", "leaf_size"])
    if filtered.empty:
        raise ValueError("可繪圖資料為空，請確認欄位都是有效數值")

    grouped = (
        filtered.groupby("guidance_value", as_index=False)[["success_rate", "leaf_size"]]
        .mean()
        .sort_values("guidance_value")
    )

    guidance_order = (
        args.guidance_order if args.guidance_order else DEFAULT_GUIDANCE_ORDER
    )
    order_df = pd.DataFrame({"guidance_value": guidance_order})
    grouped = order_df.merge(grouped, on="guidance_value", how="left")

    if not grouped["success_rate"].notna().any() or not grouped["leaf_size"].notna().any():
        raise ValueError("無任何 guidance 資料可繪圖，請檢查 guidance 與時間窗是否匹配")

    x = grouped["guidance_value"]
    y_success = grouped["success_rate"]
    y_leaf = grouped["leaf_size"]

    fig, ax_left = plt.subplots(figsize=(10, 5.5))
    ax_right = ax_left.twinx()

    ax_left.set_title(
        f"t window: {args.target_t_start}–{args.target_t_end}",
        fontsize=12,
    )

    line_success = ax_left.plot(
        x,
        y_success,
        color="tab:blue",
        marker="o",
        linewidth=2.2,
        label="Connectivity Success Rate (%)",
    )
    line_leaf = ax_right.plot(
        x,
        y_leaf,
        color="tab:green",
        marker="s",
        linewidth=2.2,
        label="Average Leaf Size",
    )

    ax_left.set_xlabel("Guidance Scale (w)")
    ax_left.set_ylabel("Connectivity Success Rate (%)", color="tab:blue")
    ax_right.set_ylabel("Average Leaf Size (leaf_size__mean)", color="tab:green")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_right.tick_params(axis="y", labelcolor="tab:green")
    ax_left.grid(True, linestyle=":", alpha=0.55)

    ax_left.set_xlim(min(guidance_order), max(guidance_order))
    ax_left.set_xticks(guidance_order)
    ax_left.set_xticklabels([f"{v:g}" for v in guidance_order])

    lines = line_success + line_leaf
    labels = [ln.get_label() for ln in lines]
    ax_left.legend(lines, labels, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] 圖表已輸出: {out_path}")


if __name__ == "__main__":
    main()
