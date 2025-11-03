#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_litematic_to_npz.py

功能：
  將一個目錄（含子資料夾）下的所有 .litematic 檔案轉換為 .npz 格式，輸出到另一個目錄（保持相同資料夾結構）。

需求：
  pip install litemapy rich numpy
"""

import argparse
import os
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from litemapy import Schematic

console = Console()


def classify_block_id(block_id: str) -> int:
    """將方塊ID分類為 0=空氣, 1=原木, 2=樹葉"""
    bid = block_id.lower()
    if "air" in bid:
        return 0
    if "log" in bid or "stem" in bid or "wood" in bid:
        return 1
    if "leaves" in bid or "leaf" in bid:
        return 2
    return 0


def convert_litematic_to_voxel(path: str, size: int = 32) -> np.ndarray:
    """將單一 .litematic 轉為 numpy 立方體"""
    schem = Schematic.load(path)
    reg = list(schem.regions.values())[0]  # 假設每個檔案只有一個區域

    width, height, length = reg.width, reg.height, reg.length
    arr = np.zeros((width, height, length), dtype=np.int8)

    for x in range(width):
        for y in range(height):
            for z in range(length):
                block = reg.getblock(x + reg.min_x(), y + reg.min_y(), z + reg.min_z())
                arr[x, y, z] = classify_block_id(block.id)

    # 補零或裁切至指定尺寸
    if any(s < size for s in arr.shape):
        padded = np.zeros((size, size, size), dtype=np.int8)
        x0 = (size - arr.shape[0]) // 2
        y0 = (size - arr.shape[1]) // 2
        z0 = (size - arr.shape[2]) // 2
        padded[x0:x0 + arr.shape[0], y0:y0 + arr.shape[1], z0:z0 + arr.shape[2]] = arr
        arr = padded
    elif any(s > size for s in arr.shape):
        x0 = (arr.shape[0] - size) // 2
        y0 = (arr.shape[1] - size) // 2
        z0 = (arr.shape[2] - size) // 2
        arr = arr[x0:x0 + size, y0:y0 + size, z0:z0 + size]

    return arr


def main():
    parser = argparse.ArgumentParser(description="批次將 .litematic 轉換為 .npz（遞迴搜尋所有子資料夾並保持結構）")
    parser.add_argument("--input", required=True, help="輸入資料夾路徑（會遞迴搜尋所有子資料夾內的 .litematic）")
    parser.add_argument("--output", required=True, help="輸出資料夾路徑（會鏡像輸入資料夾的子資料夾結構）")
    parser.add_argument("--size", type=int, default=32, help="立方體尺寸，預設 32")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    
    if not input_path.is_dir():
        console.print(f"[red]找不到輸入資料夾：{input_path}[/red]")
        return
    
    # 遞迴收集所有 .litematic 檔案，並計算相對路徑
    files = []
    for fpath in sorted(input_path.rglob("*.litematic")):
        if fpath.is_file():
            rel_path = fpath.relative_to(input_path)
            files.append((rel_path, fpath))
    
    if not files:
        console.print(f"[red]在 {input_path} 及其子資料夾找不到任何 .litematic 檔案[/red]")
        return

    console.print(f"[green]共找到 {len(files)} 個 .litematic 檔案[/green]")
    console.print(f"[cyan]輸入資料夾：{input_path}[/cyan]")
    console.print(f"[cyan]輸出資料夾：{output_path}[/cyan]")
    console.print(f"[cyan]立方體尺寸：{args.size}x{args.size}x{args.size}[/cyan]")

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("轉換中", total=len(files))

        for rel_path, abs_path in files:
            try:
                voxel = convert_litematic_to_voxel(str(abs_path), size=args.size)
                
                # 計算輸出路徑（保持相對路徑結構）
                out_rel_path = rel_path.with_suffix(".npz")
                out_abs_path = output_path / out_rel_path
                
                # 確保輸出子目錄存在
                out_abs_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 儲存
                np.savez_compressed(str(out_abs_path), data=voxel)
            except Exception as e:
                console.print(f"[yellow]轉換失敗[/yellow] {rel_path}：{e}")
            finally:
                progress.advance(task)

    console.print(f"[bold green]✅ 全部轉換完成！[/bold green]\n輸出資料夾：{output_path}")


if __name__ == "__main__":
    main()