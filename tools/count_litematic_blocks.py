#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import zipfile
import tempfile
import shutil
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

# 進度條（若無 rich 則不使用）
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
    RICH_ENABLED = True
    console = Console()
except Exception:
    RICH_ENABLED = False
    console = None


def safe_load_schematic(file_path: Path) -> Schematic:
    """
    安全地加載 Schematic，如果遇到缺失 Entities/TileEntities 鍵的錯誤，會在內存中修復（不保存文件）。
    本腳本不處理實體數據，只需要能夠成功加載文件即可。
    """
    try:
        # 先嘗試正常加載
        return Schematic.load(str(file_path))
    except KeyError as e:
        # 如果是 Entities 或 TileEntities 相關的 KeyError，在內存中修復
        if "Entities" in str(e) or "TileEntities" in str(e):
            try:
                import nbtlib
                from nbtlib import File
                
                # 讀取 NBT 文件（.litematic 文件是 gzipped 的）
                nbt_file = File.load(str(file_path), gzipped=True)
                
                # 檢查並修復每個 Region（在內存中）
                if "Regions" in nbt_file:
                    for region_name, region_data in nbt_file["Regions"].items():
                        if "Entities" not in region_data:
                            region_data["Entities"] = nbtlib.List([])
                        if "TileEntities" not in region_data:
                            region_data["TileEntities"] = nbtlib.List([])
                    
                    # 保存到臨時文件（不修改原始文件）
                    with tempfile.NamedTemporaryFile(suffix=".litematic", delete=False) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                    
                    try:
                        nbt_file.save(str(tmp_path), gzipped=True)
                        # 從臨時文件加載 Schematic
                        schem = Schematic.load(str(tmp_path))
                        # 刪除臨時文件
                        tmp_path.unlink()
                        return schem
                    except Exception:
                        # 如果失敗，清理臨時文件
                        if tmp_path.exists():
                            tmp_path.unlink()
                        raise
            except Exception as fix_err:
                # 如果修復失敗，重新拋出原始錯誤
                raise e from fix_err
        
        # 如果不是 Entities/TileEntities 相關的錯誤，直接拋出
        raise


def count_blocks_in_litematic(file_path: Path) -> Counter:
    counts = Counter()
    try:
        schem = safe_load_schematic(file_path)
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


def get_litematics_from_zip(zip_path: Path, temp_dir: Path) -> list[Path]:
    """從 zip 檔案中提取所有 .litematic 檔案到臨時目錄，返回檔案路徑列表"""
    litematics = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 提取所有 .litematic 檔案
            for name in zf.namelist():
                if name.endswith('.litematic'):
                    # 提取到臨時目錄
                    zf.extract(name, temp_dir)
                    litematics.append(temp_dir / name)
    except Exception as e:
        print(f"[警告] 無法讀取 zip 檔案 {zip_path.name}: {e}", file=sys.stderr)
    return litematics


def main():
    parser = argparse.ArgumentParser(description="統計資料夾（含子資料夾）或 zip 檔案內所有 .litematic 檔案之方塊 ID 與數量（含百分比與彩色重點）")
    parser.add_argument("--input_dir", help="含有 .litematic 檔案的資料夾路徑（會遞迴搜尋子資料夾）")
    parser.add_argument("--input_zip", help="含有 .litematic 檔案的 zip 檔案路徑")
    parser.add_argument("-o", "--output", default="block_ids.txt", help="輸出檔案名稱 (預設: block_ids.txt)")
    parser.add_argument("--no-color", action="store_true", help="停用彩色輸出")
    parser.add_argument("--no-progress", action="store_true", help="停用進度條")
    args = parser.parse_args()

    if args.no_color:
        # 強制關閉顏色
        global C_ENABLED, Fore, Style
        C_ENABLED = False
        class _N: RESET_ALL = CYAN = YELLOW = GREEN = MAGENTA = ""
        class _S: BRIGHT = NORMAL = ""
        Fore = _N(); Style = _S()

    # 檢查是否至少提供一個輸入
    if not args.input_dir and not args.input_zip:
        print("錯誤：必須提供 --input_dir 或 --input_zip 參數", file=sys.stderr)
        sys.exit(1)

    litematics = []
    temp_dir = None
    
    # 處理目錄輸入
    if args.input_dir:
        folder = Path(args.input_dir).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            print(f"找不到資料夾：{folder}", file=sys.stderr)
            sys.exit(2)
        dir_litematics = sorted(folder.rglob("*.litematic"))
        litematics.extend(dir_litematics)
    
    # 處理 zip 輸入（需要臨時目錄來存放提取的檔案）
    if args.input_zip:
        zip_path = Path(args.input_zip).expanduser().resolve()
        if not zip_path.exists() or not zip_path.is_file():
            print(f"找不到 zip 檔案：{zip_path}", file=sys.stderr)
            sys.exit(2)
        # 創建臨時目錄，在處理完所有檔案後才刪除
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        zip_litematics = get_litematics_from_zip(zip_path, temp_path)
        litematics.extend(zip_litematics)
    
    if not litematics:
        print("找不到任何 .litematic 檔案", file=sys.stderr)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(3)

    total_counts = Counter()
    try:
        if RICH_ENABLED and not args.no_progress:
            # 使用 rich 進度條
            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("處理檔案中", total=len(litematics))
                for fp in litematics:
                    total_counts.update(count_blocks_in_litematic(fp))
                    progress.update(task, advance=1, description=f"處理中: {fp.name}")
        else:
            # 沒有 rich 或禁用進度條時使用簡單的進度提示
            for idx, fp in enumerate(litematics, start=1):
                if not args.no_progress:
                    print(f"處理中 ({idx}/{len(litematics)}): {fp.name}", file=sys.stderr)
                total_counts.update(count_blocks_in_litematic(fp))
    finally:
        # 清理臨時目錄
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

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