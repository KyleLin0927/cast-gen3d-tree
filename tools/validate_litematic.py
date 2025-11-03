#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
驗證資料夾（含子資料夾）內所有 .litematic 檔案，將「格式損壞 / 不完整 / 有風險」的檔案列出並附上原因。
同時可檢查非空氣方塊佔比，超過門檻（預設 0.5）也視為風險。

安裝相依：
    pip install litemapy

用法：
    python validate_litematic_strict.py /path/to/folder --threshold 0.6 --quiet
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

console = Console()

# ---- 可調參數（必要時） ---------------------------------------------------------
EXPECTED_METADATA_KEYS = {"Author", "RegionCount"}  # 常見且實用的欄位（可自行增減）
# ---------------------------------------------------------------------------------

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def _compute_region_volume(reg) -> int:
    """嘗試用多種 API 取得 region 體積；失敗則拋出例外。"""
    # 常見 litemapy API：getvolume()
    for attr in ("getvolume", "get_volume"):
        if hasattr(reg, attr):
            return getattr(reg, attr)()
    # 有些版本可能提供 size 或 dimension
    for attr in ("size", "dimension", "shape"):
        if hasattr(reg, attr):
            s = getattr(reg, attr)
            # 允許 size 可能是 (x, y, z) 或具名屬性
            try:
                if isinstance(s, (tuple, list)) and len(s) == 3:
                    return int(s[0]) * int(s[1]) * int(s[2])
                # 嘗試具名屬性
                x = int(getattr(s, "x"))
                y = int(getattr(s, "y"))
                z = int(getattr(s, "z"))
                return x * y * z
            except Exception:
                pass
    raise RuntimeError("無法取得 region 體積（缺少 getvolume/size 資訊）")

def _compute_region_non_air(reg) -> int:
    """嘗試用多種 API 取得 region 的非空氣方塊數；失敗則拋出例外。"""
    # 常見 litemapy API：getblockcount()（通常指非空氣）
    for attr in ("getblockcount", "get_block_count", "count_non_air"):
        if hasattr(reg, attr):
            return getattr(reg, attr)()
    # 後備方案：遍歷 palette 或方塊迭代器（若有）
    # 註：不同版本 API 可能不同，以下盡量以 duck-typing 應付
    # 有些 region 可能提供 blocks() 或 iter_blocks()
    for attr in ("blocks", "iter_blocks", "iter_all_blocks"):
        if hasattr(reg, attr):
            it = getattr(reg, attr)()
            non_air = 0
            for b in it:
                # 嘗試判斷「是不是空氣」
                # 常見名稱 "minecraft:air", "air"
                try:
                    name = None
                    if hasattr(b, "name"):
                        name = b.name
                    elif isinstance(b, tuple) and b and hasattr(b[0], "name"):
                        name = b[0].name
                    if name is None:
                        # 無法判斷就保守計為非空氣（以免把有效方塊誤判為空氣）
                        non_air += 1
                    else:
                        if str(name).endswith(":air") or str(name) == "air":
                            pass
                        else:
                            non_air += 1
                except Exception:
                    # 有異常時保守計為非空氣
                    non_air += 1
            return non_air
    raise RuntimeError("無法取得非空氣方塊數（缺少 getblockcount/blocks 介面）")

def analyze_litematic(path: str, required_meta: set) -> Tuple[float, List[str]]:
    """
    回傳 (non_air_ratio, issues)
    - non_air_ratio: 若無法計算則回傳 -1
    - issues: 發現的問題列表（字串）
    """
    issues: List[str] = []
    schem = None

    # 1) 載入
    try:
        from litemapy import Schematic
    except ImportError:
        raise SystemExit("找不到 litemapy，請先安裝：pip install litemapy")

    try:
        schem = Schematic.load(path)
    except Exception as e:
        return -1.0, [f"unreadable: {e}"]

    # 2) 檢查 metadata（盡量用 duck-typing，避免不同版本差異）
    metadata = None
    for attr in ("metadata", "Metadata"):
        if hasattr(schem, attr):
            candidate = getattr(schem, attr)
            if isinstance(candidate, dict):
                metadata = candidate
                break
    if metadata is None:
        # 嘗試從 nbt root 抓
        try:
            nbt = getattr(schem, "nbt", None)
            if nbt and isinstance(nbt, dict) and "Metadata" in nbt and isinstance(nbt["Metadata"], dict):
                metadata = nbt["Metadata"]
        except Exception as e:
            issues.append(f"metadata_access_error: {e}")

    if metadata is None:
        issues.append("metadata_missing")
    else:
        missing = [k for k in required_meta if k not in metadata]
        if missing:
            issues.append(f"metadata_keys_missing: {','.join(missing)}")

    # 3) regions 結構
    regions = getattr(schem, "regions", None)
    if not isinstance(regions, dict) or len(regions) == 0:
        issues.append("regions_missing_or_empty")
        return -1.0, issues  # 沒 region 就無法算密度，直接回傳

    # 4) 計算密度（全檔案加總）
    total_volume = 0
    total_non_air = 0

    for name, reg in regions.items():
        try:
            vol = _compute_region_volume(reg)
            if vol <= 0:
                issues.append(f"region_zero_or_neg_volume:{_safe_str(name)}")
            total_volume += max(vol, 0)
        except Exception as e:
            issues.append(f"region_volume_error:{_safe_str(name)}:{e}")

        try:
            non_air = _compute_region_non_air(reg)
            if non_air < 0:
                issues.append(f"region_non_air_negative:{_safe_str(name)}")
            total_non_air += max(non_air, 0)
        except Exception as e:
            issues.append(f"region_non_air_error:{_safe_str(name)}:{e}")

    if total_volume <= 0:
        issues.append("total_volume_zero_or_invalid")
        return -1.0, issues

    non_air_ratio = total_non_air / float(total_volume)
    return non_air_ratio, issues

def main():
    parser = argparse.ArgumentParser(
        description="驗證 .litematic：列出格式損壞 / 不完整 / 有風險的檔案（含非空氣佔比門檻，遞迴搜尋所有子資料夾）"
    )
    parser.add_argument("folder", help="目標資料夾（會遞迴搜尋所有子資料夾）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="非空氣佔比門檻（0~1，小數；預設 0.5 代表 50%）")
    parser.add_argument("--quiet", action="store_true",
                        help="只輸出『有風險檔案』清單（精簡模式）")
    parser.add_argument("--list-all", action="store_true",
                        help="同時列出每個檔案的密度與問題摘要（非精簡）")
    args = parser.parse_args()

    folder_path = Path(args.folder).expanduser().resolve()
    if not folder_path.is_dir():
        console.print(f"[red]❌ 找不到資料夾：{folder_path}[/red]")
        sys.exit(1)

    # 遞迴收集所有 .litematic 檔案，並計算相對路徑
    candidates = []
    for fpath in sorted(folder_path.rglob("*.litematic")):
        if fpath.is_file():
            rel_path = fpath.relative_to(folder_path)
            candidates.append((str(rel_path), str(fpath)))
    
    if not candidates:
        console.print(f"[red]❌ 在 {folder_path} 及其子資料夾中找不到任何 .litematic 檔案[/red]")
        sys.exit(1)

    console.print(f"[cyan]找到 {len(candidates)} 個 .litematic 檔案，開始驗證...[/cyan]\n")

    risky: List[Tuple[str, float, List[str]]] = []
    all_results: List[Tuple[str, float, List[str]]] = []
    list_all_output = []  # 收集 --list-all 的輸出

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
        task = progress.add_task("[cyan]驗證中...", total=len(candidates))
        
        for rel_path, abs_path in candidates:
            ratio, issues = analyze_litematic(abs_path, EXPECTED_METADATA_KEYS)
            all_results.append((rel_path, ratio, issues))

            # 判定是否「有風險」
            is_risky = False
            reasons = list(issues)  # copy
            if ratio < 0:  # 無法計算密度（多半是壞掉/結構不完整）
                is_risky = True
                reasons.append("density_unavailable")
            elif ratio > args.threshold:
                is_risky = True
                reasons.append(f"non_air_ratio_above_threshold:{ratio:.2%}")

            if is_risky:
                risky.append((rel_path, ratio, reasons))

            if args.list_all and not args.quiet:
                # 收集輸出，不在迴圈內印（避免干擾進度條）
                if ratio >= 0:
                    list_all_output.append(f"{rel_path}: 非空氣佔比={ratio:.2%} | issues={';'.join(issues) if issues else 'none'}")
                else:
                    list_all_output.append(f"{rel_path}: 非空氣佔比=（無法計算） | issues={';'.join(issues) if issues else 'none'}")
            
            # 更新進度條
            risky_count = len(risky)
            progress.update(task, advance=1, description=f"[cyan]驗證中... (✓{len(candidates)-risky_count} ⚠{risky_count})")
    
    # 進度條結束後，輸出 --list-all 內容
    if list_all_output:
        console.print("\n[bold]詳細清單：[/bold]")
        for line in list_all_output:
            console.print(line)

    # 最終輸出：有風險清單
    console.print("\n[bold yellow]=== 有風險的 .litematic 檔案 ===[/bold yellow]")
    if not risky:
        console.print("[green]（沒有）[/green]")
    else:
        for fname, ratio, reasons in risky:
            ratio_txt = f"{ratio:.2%}" if ratio >= 0 else "N/A"
            console.print(f"[red]⚠[/red] {fname}\t[yellow]密度={ratio_txt}[/yellow]\t原因：{', '.join(reasons)}")

    if not args.quiet:
        # 總結
        console.print("\n[bold cyan]--- 統計 ---[/bold cyan]")
        console.print(f"總檔案數：[cyan]{len(candidates)}[/cyan]")
        console.print(f"有風險數：[yellow]{len(risky)}[/yellow]")
        if len(risky) == 0:
            console.print("[bold green]✓ 所有檔案都通過驗證！[/bold green]")

if __name__ == "__main__":
    main()