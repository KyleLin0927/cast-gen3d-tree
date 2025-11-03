#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from collections import Counter

# 顏色（若無 colorama 則不加色）
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    C_ENABLED = True
except Exception:
    C_ENABLED = False
    class _N: RESET_ALL = CYAN = YELLOW = GREEN = MAGENTA = ""
    class _S: BRIGHT = NORMAL = ""
    Fore = _N(); Style = _S()

try:
    from litemapy import Schematic
except ImportError:
    print("找不到 litemapy 套件。請先安裝：pip install litemapy", file=sys.stderr)
    sys.exit(1)


def count_blocks_in_litematic(file_path: Path) -> Counter:
    counts = Counter()
    try:
        schem = Schematic.load(str(file_path))
    except Exception as e:
        print(f"[警告] 無法載入 {file_path.name}: {e}", file=sys.stderr)
        return counts

    for reg in schem.regions.values():
        for x, y, z in reg.block_positions():
            block = reg[x, y, z]
            bid = block.id
            if bid != "minecraft:air":
                counts[bid] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="統計資料夾（含子資料夾）內所有 .litematic 檔案之方塊 ID 與數量（含百分比與彩色重點）")
    parser.add_argument("folder", help="含有 .litematic 檔案的資料夾路徑（會遞迴搜尋子資料夾）")
    parser.add_argument("-o", "--output", default="block_ids.txt", help="輸出檔案名稱 (預設: block_ids.txt)")
    parser.add_argument("--no-color", action="store_true", help="停用彩色輸出")
    args = parser.parse_args()

    if args.no_color:
        # 強制關閉顏色
        global C_ENABLED, Fore, Style
        C_ENABLED = False
        class _N: RESET_ALL = CYAN = YELLOW = GREEN = MAGENTA = ""
        class _S: BRIGHT = NORMAL = ""
        Fore = _N(); Style = _S()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"找不到資料夾：{folder}", file=sys.stderr)
        sys.exit(2)

    litematics = sorted(folder.rglob("*.litematic"))
    if not litematics:
        print(f"在 {folder} 及其子資料夾找不到任何 .litematic 檔案", file=sys.stderr)
        sys.exit(3)

    total_counts = Counter()
    for fp in litematics:
        total_counts.update(count_blocks_in_litematic(fp))

    # 輸出 block id 清單（txt 只含 ID）
    out_path = Path(args.output).expanduser().resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        for bid in sorted(total_counts.keys()):
            f.write(f"{bid}\n")

    # Console：僅四段內容（加上百分比與顏色）
    all_blocks = sum(total_counts.values())
    header = f"{Fore.CYAN}{Style.BRIGHT}====== 總結：所有檔案的方塊統計 ======{Style.NORMAL}{Fore.RESET}" if C_ENABLED else "====== 總結：所有檔案的方塊統計 ======"
    summary = (
        f"{Fore.YELLOW}{Style.BRIGHT}"
        f"檔案數：{len(litematics)}，不同方塊種類：{len(total_counts)}，非空氣方塊總數：{all_blocks}"
        f"{Style.NORMAL}{Fore.RESET}"
        if C_ENABLED else
        f"檔案數：{len(litematics)}，不同方塊種類：{len(total_counts)}，非空氣方塊總數：{all_blocks}"
    )
    print(header)
    print(summary)
    print()

    # 依數量遞減列出，前 3 名強調色
    sorted_items = sorted(total_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for idx, (bid, cnt) in enumerate(sorted_items, start=1):
        pct = (cnt / all_blocks * 100) if all_blocks else 0.0
        line = f"{bid}: {cnt} ({pct:.2f}%)"
        if C_ENABLED and idx <= 3:
            # 前 3 名加亮綠色
            line = f"{Fore.GREEN}{Style.BRIGHT}{line}{Style.NORMAL}{Fore.RESET}"
        print(line)

    print()
    tail = (
        f"{Fore.MAGENTA}{Style.BRIGHT}已輸出方塊 ID 清單：{out_path}{Style.NORMAL}{Fore.RESET}"
        if C_ENABLED else
        f"已輸出方塊 ID 清單：{out_path}"
    )
    print(tail)


if __name__ == "__main__":
    main()