#!/usr/bin/env python3
"""
遞迴搜尋指定根目錄下所有子目錄中的 metadata_flat.csv，合併為單一 CSV。

各實驗目錄的寬表欄位可能不同，合併時採用欄位聯集（缺欄為空）。

使用方式:
  python aggregate_metadata_flat.py --search_dir ./runs --out_dir ./runs/all_metadata_flat.csv
  # --out_dir 也可傳「目錄」，會寫入該目錄下的 aggregated_metadata_flat.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate all metadata_flat.csv under a directory tree into one CSV."
    )
    parser.add_argument(
        "--search_dir",
        type=str,
        required=True,
        help="要遞迴搜尋的根目錄",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="aggregated_metadata_flat.csv",
        help="輸出 CSV 檔路徑；若為已存在的目錄則寫入 aggregated_metadata_flat.csv",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="source_dir",
        help="標註來源目錄的欄位名稱（相對於 --search_dir）",
    )
    args = parser.parse_args()

    root = Path(args.search_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"[錯誤] 不是有效目錄: {root}")
        return

    out_path = Path(args.out_dir).expanduser().resolve()
    if out_path.is_dir():
        out_path = out_path / "aggregated_metadata_flat.csv"

    files = sorted(root.rglob("metadata_flat.csv"))

    # 避免把輸出檔當成輸入（若輸出在搜尋樹內）
    files = [p for p in files if p.resolve() != out_path]

    if not files:
        print(f"在 {root} 底下找不到任何 metadata_flat.csv")
        return

    print(f"找到 {len(files)} 個 metadata_flat.csv，開始合併...")

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[警告] 無法讀取，略過 {path}: {e}")
            continue

        if df.empty:
            print(f"[警告] 無資料列，略過: {path}")
            continue

        rel = str(path.parent.relative_to(root))
        if rel == ".":
            rel = ""
        df = df.copy()
        df.insert(0, args.source_column, rel)
        frames.append(df)
        print(f"  ✓ {path.parent.relative_to(root) if path.parent != root else '.'}")

    if not frames:
        print("沒有任何可合併的列，結束。")
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n合併完成：{len(merged)} 列、{len(merged.columns)} 欄 → {out_path}")


if __name__ == "__main__":
    main()
