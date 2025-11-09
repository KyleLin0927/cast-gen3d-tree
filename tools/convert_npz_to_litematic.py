#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npz_to_litematic.py  (fixed for litemapy 0.11.0b0)

把 --input 資料夾（含子資料夾）內的 .npz 批次轉成 .litematic，輸出到 --output（保持相同資料夾結構）
需求: pip install litemapy numpy rich
"""

import argparse
import os
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from litemapy import Schematic, Region, BlockState  # Region(x,y,z,w,h,l), reg.as_schematic(), schem.save()

console = Console()

# 三類方塊對應：0=空氣, 1=橡木原木, 2=橡木樹葉
AIR = BlockState("minecraft:air")
OAK_LOG = BlockState("minecraft:oak_log")
# 設定 persistent=true 以防止樹葉腐敗，distance 設為 1 以符合法規定屬性
OAK_LEAVES = BlockState("minecraft:oak_leaves", {"persistent": "true", "distance": "1"})

ID_TO_BLOCK = {
    0: AIR,
    1: OAK_LOG,
    2: OAK_LEAVES,
}

def load_npz_array(npz_path: str) -> np.ndarray:
    f = np.load(npz_path, allow_pickle=True)
    key = "data" if "data" in f.files else f.files[0]
    arr = f[key]
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(npz_path)}: expected 3D array, got shape {arr.shape}")
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int8)
    return arr

def array_to_schematic(vox: np.ndarray, name: str) -> Schematic:
    sx, sy, sz = vox.shape  # X,Y,Z
    # Region 只接受 6 個數字參數 (x,y,z,width,height,length)
    reg = Region(0, 0, 0, sx, sy, sz)  # 原點放置，大小即陣列尺寸   [oai_citation:4‡litemapy.readthedocs.io](https://litemapy.readthedocs.io/en/v0.11.0b0/region.html)

    # 寫方塊（使用 setblock）
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                bid = int(vox[x, y, z])
                block = ID_TO_BLOCK.get(bid, AIR)
                reg.setblock(x, y, z, block)  # 官方示例中的 API   [oai_citation:5‡litemapy.readthedocs.io](https://litemapy.readthedocs.io/en/v0.7.2b0/setup.html?utm_source=chatgpt.com)

    # 用 as_schematic 直接封裝成 Schematic，並給名字
    schem = reg.as_schematic(name=name, author="npz_to_litematic", description="Converted from npz")  #  [oai_citation:6‡litemapy.readthedocs.io](https://litemapy.readthedocs.io/en/v0.11.0b0/region.html)
    return schem

def main():
    ap = argparse.ArgumentParser(description="Batch convert .npz (3-class voxels) to .litematic (遞迴搜尋所有子資料夾並保持結構)")
    ap.add_argument("--input", required=True, help="輸入資料夾（會遞迴搜尋所有子資料夾內的 .npz）")
    ap.add_argument("--output", required=True, help="輸出資料夾（會鏡像輸入資料夾的子資料夾結構）")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    
    if not input_path.is_dir():
        console.print(f"[red]找不到輸入資料夾：{input_path}[/red]")
        return
    
    # 遞迴收集所有 .npz 檔案，並計算相對路徑
    files = []
    for fpath in sorted(input_path.rglob("*.npz")):
        if fpath.is_file():
            rel_path = fpath.relative_to(input_path)
            files.append((rel_path, fpath))
    
    if not files:
        console.print(f"[red]在 {input_path} 及其子資料夾找不到任何 .npz 檔案[/red]")
        return

    console.print(f"[green]共找到 {len(files)} 個 .npz 檔案[/green]")
    console.print(f"[cyan]輸入資料夾：{input_path}[/cyan]")
    console.print(f"[cyan]輸出資料夾：{output_path}[/cyan]")

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
                vox = load_npz_array(str(abs_path))
                
                # 計算輸出路徑（保持相對路徑結構）
                out_rel_path = rel_path.with_suffix(".litematic")
                out_abs_path = output_path / out_rel_path
                
                # 確保輸出子目錄存在
                out_abs_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 轉換並儲存
                base_name = rel_path.stem
                schem = array_to_schematic(vox, name=base_name)
                schem.save(str(out_abs_path))  # 儲存檔案
            except Exception as e:
                console.print(f"[yellow]轉換失敗[/yellow] {rel_path}：{e}")
            finally:
                progress.advance(task)

    console.print(f"[bold green]✅ 全部轉換完成！[/bold green]\n輸出資料夾：{output_path}")

if __name__ == "__main__":
    main()