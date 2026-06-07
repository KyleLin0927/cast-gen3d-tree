#!/usr/bin/env python3
"""
搜集並聚合所有的 sample_labels_summary.csv / sample_label_summary.csv

這支腳本會遞迴搜尋指定目錄下所有的 sample_labels_summary.csv /
sample_label_summary.csv，
將其兩欄式 (Metric, Value) 的資料轉置成單一橫列 (Row)，
並將所有實驗結果結合成一張超級大表 (aggregated_results.csv)，方便後續畫圖與分析。

解析規則（與 ``utils.export_csv.write_sample_labels_summary_csv`` 格式對齊）：

- 區段標題列為 ``Value`` 為空的列（例如 ``Metric=Overview``），其後列皆屬該區段。
- 輸出欄名為 ``{區段slug}__{指標slug}``，避免不同區段重複的 ``mean``、
  ``broken_count`` 等互相覆蓋，確保聚合後 **每一列指標** 都會出現在總表。

使用方式:
  python aggregate_summaries.py --search_dir ./generation --out_dir ./generation/all_experiments_summary.csv
  （--search_dir 為必填）
"""

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd

SUMMARY_BASENAMES = ("sample_labels_summary.csv", "sample_label_summary.csv")


def _slug_for_column(text: str, max_len: int = 96) -> str:
    """小寫、非英數改底線；長標題截斷（須與解析時一致）。"""
    t = str(text).strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        return "x"
    if len(t) > max_len:
        t = t[:max_len].rstrip("_")
    return t
def _coerce_cell_value(value) -> object:
    """將 Value 轉成 int / float / str；支援科學記號與無小數點的浮點字串。"""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    num = pd.to_numeric(s, errors="coerce")
    if pd.notna(num):
        fv = float(num)
        if fv.is_integer() and abs(fv) <= 2**53:
            return int(fv)
        return fv
    return s


def parse_summary_csv(filepath: str) -> dict:
    """
    解析單一的 summary CSV；依區段標題（Value 為空之列）組出唯一欄名，
    避免重複 Metric 互相覆蓋。
    """
    data = {}
    current_section_slug = ""
    try:
        # Auto-detect delimiter so both comma-CSV and tab-delimited files are accepted.
        df = pd.read_csv(filepath, sep=None, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        metric_col = next((c for c in df.columns if c.lower() == "metric"), None)
        value_col = next((c for c in df.columns if c.lower() == "value"), None)

        if metric_col is None or value_col is None:
            print(f"[警告] 檔案格式不符，略過: {filepath}")
            return data

        for _, row in df.iterrows():
            metric_raw = row[metric_col]
            value = row[value_col]
            if pd.isna(metric_raw):
                continue
            metric = str(metric_raw).strip()
            if metric == "" or metric.lower() == "nan":
                continue

            value_empty = pd.isna(value) or str(value).strip() == ""

            if value_empty:
                current_section_slug = _slug_for_column(metric)
                continue

            coerced = _coerce_cell_value(value)
            if coerced is None:
                continue

            if current_section_slug:
                key = f"{current_section_slug}__{_slug_for_column(metric)}"
            else:
                key = _slug_for_column(metric)
            data[key] = coerced
    except Exception as e:
        print(f"[錯誤] 無法解析 {filepath}: {e}")

    return data

def main():
    parser = argparse.ArgumentParser(description="Aggregate all sample_labels_summary.csv files.")
    parser.add_argument("--search_dir", type=str, required=True, help="要搜尋的根目錄（必填）")
    parser.add_argument("--out_dir", type=str, default="./all_experiments_summary.csv", help="輸出的總表路徑")
    args = parser.parse_args()

    summary_files = []
    for basename in SUMMARY_BASENAMES:
        search_pattern = os.path.join(args.search_dir, "**", basename)
        summary_files.extend(glob.glob(search_pattern, recursive=True))
    summary_files = sorted(set(summary_files))
    
    if not summary_files:
        names = " / ".join(SUMMARY_BASENAMES)
        print(f"在 {args.search_dir} 底下找不到任何 {names}！")
        return

    print(f"🔍 找到 {len(summary_files)} 個 summary 檔案，開始聚合...")

    all_experiments = []
    metric_columns_order = []
    seen_metric_columns = set()
    
    for filepath in sorted(summary_files):
        # 從路徑中萃取實驗名稱 (抓取 summary.csv 的上一層資料夾名稱)
        exp_name = Path(filepath).parent.name
        
        # 解析該檔案的指標
        metrics_dict = parse_summary_csv(filepath)
        
        if metrics_dict:
            # 將實驗名稱放在 Dictionary 的第一位
            row_data = {"Experiment_Name": exp_name}
            row_data.update(metrics_dict)
            all_experiments.append(row_data)

            # 欄位順序以原始 summary 檔內出現順序為準（先遇到先保留）
            for key in metrics_dict.keys():
                if key not in seen_metric_columns:
                    seen_metric_columns.add(key)
                    metric_columns_order.append(key)
            print(f"  ✓ 已處理: {exp_name}")

    if all_experiments:
        # 將 List of Dictionaries 轉為 DataFrame
        final_df = pd.DataFrame(all_experiments)
        
        # 欄位順序：Experiment_Name + 原始 summary 解析順序
        ordered_cols = ["Experiment_Name"] + [
            c for c in metric_columns_order if c in final_df.columns
        ]
        trailing_cols = [c for c in final_df.columns if c not in ordered_cols]
        final_df = final_df[ordered_cols + trailing_cols]

        # 輸出 CSV
        out_abs = os.path.abspath(args.out_dir)
        parent = os.path.dirname(out_abs)
        if parent:
            os.makedirs(parent, exist_ok=True)
        final_df.to_csv(args.out_dir, index=False, encoding="utf-8-sig")
        print(f"\n🎉 聚合完成！總表已儲存至: {args.out_dir}")
        print(f"現在你可以直接用這張表來畫 Trade-off 曲線了！")

if __name__ == "__main__":
    main()


