#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_litematic_to_npz.py

功能：
  將 .litematic 檔案轉換為 .npz 格式，支援從資料夾或 zip 檔案讀取，也可輸出為資料夾或 zip 檔案。

參數：
  --input:  輸入路徑（自動檢測：如果是 .zip 檔案則從 zip 讀取，否則視為資料夾）
  --output: 輸出路徑（如果以 .zip 結尾則輸出為 zip 檔案，否則為資料夾）
            - 輸出 zip 內部結構與輸入結構相同，不會添加額外的外層資料夾

需求：
  pip install litemapy rich numpy

重要說明：
- 本腳本「絕對不會修改原始文件」，所有操作都是只讀
- 如果遇到缺少 Entities/TileEntities 鍵的文件：
  * 會在內存中修復 NBT 數據（使用臨時文件）
  * 加載修復後的數據進行轉換
  * 處理完成後刪除臨時文件
  * 原始文件保持不變
- 本腳本不處理實體數據，只轉換方塊數據為 voxel 格式
"""

import argparse
import os
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
from litemapy import Schematic
import multiprocessing
import zipfile
from io import BytesIO
import tempfile

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


def safe_load_schematic(file_path: str) -> Schematic:
    """
    安全地加載 Schematic，如果遇到缺失 Entities/TileEntities 鍵的錯誤，會在內存中修復（不保存文件）。
    本腳本不處理實體數據，只需要能夠成功加載文件即可。
    """
    try:
        # 先嘗試正常加載
        return Schematic.load(file_path)
    except KeyError as e:
        # 如果是 Entities 或 TileEntities 相關的 KeyError，在內存中修復
        if "Entities" in str(e) or "TileEntities" in str(e):
            try:
                import nbtlib
                from nbtlib import File
                
                # 讀取 NBT 文件（.litematic 文件是 gzipped 的）
                nbt_file = File.load(file_path, gzipped=True)
                
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


def convert_litematic_to_voxel(path_or_bytes, size: int = 32) -> np.ndarray:
    """將單一 .litematic 轉為 numpy 立方體
    
    Args:
        path_or_bytes: 檔案路徑（str）或檔案內容（bytes）
        size: 立方體尺寸
    """
    if isinstance(path_or_bytes, bytes):
        # 從記憶體讀取：使用臨時檔案
        with tempfile.NamedTemporaryFile(suffix=".litematic", delete=True) as tmp_file:
            tmp_file.write(path_or_bytes)
            tmp_file.flush()
            schem = safe_load_schematic(tmp_file.name)
    else:
        # 從檔案路徑讀取
        schem = safe_load_schematic(path_or_bytes)
    
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


def process_single_file(args_tuple):
    """處理單一檔案（用於多進程）
    
    Args:
        args_tuple: (file_data, output_path, rel_path, size, output_zip)
            - file_data: 檔案路徑（str）或檔案內容（bytes）
            - output_path: 輸出路徑（Path 或 None，如果使用zip輸出）
            - rel_path: 相對路徑（str 或 Path）
            - size: 立方體尺寸
            - output_zip: 是否輸出到zip（bool）
    """
    file_data, output_path, rel_path, size, output_zip = args_tuple
    try:
        voxel = convert_litematic_to_voxel(file_data, size=size)
        
        if output_zip:
            # 輸出到zip：將npz數據保存到BytesIO
            # rel_path 是字符串（如 "folder1/folder2/file.litematic"）
            # 只需將 .litematic 改為 .npz
            if isinstance(rel_path, Path):
                rel_path_str = str(rel_path).replace('\\', '/')  # 確保使用正斜杠
            else:
                rel_path_str = rel_path
            out_rel_path_str = rel_path_str.replace('.litematic', '.npz')
            
            npz_buffer = BytesIO()
            np.savez_compressed(npz_buffer, data=voxel)
            npz_buffer.seek(0)
            return (True, rel_path_str, None, npz_buffer.getvalue(), out_rel_path_str)
        else:
            # 輸出到檔案系統
            # rel_path 可能是字符串或 Path，轉換為 Path
            if isinstance(rel_path, str):
                rel_path = Path(rel_path)
            out_rel_path = rel_path.with_suffix(".npz")
            out_abs_path = output_path / out_rel_path
            
            # 確保輸出子目錄存在
            out_abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 儲存
            np.savez_compressed(str(out_abs_path), data=voxel)
            return (True, str(rel_path), None, None, None)
    except Exception as e:
        rel_path_str = str(rel_path) if isinstance(rel_path, Path) else rel_path
        return (False, rel_path_str, str(e), None, None)


def main():
    parser = argparse.ArgumentParser(description="批次將 .litematic 轉換為 .npz（遞迴搜尋所有子資料夾並保持結構）")
    parser.add_argument("--input", required=True, help="輸入路徑（自動檢測：zip 檔案或資料夾）")
    parser.add_argument("--output", required=True, help="輸出路徑（如果以 .zip 結尾則輸出為 zip 檔案，否則為資料夾）")
    parser.add_argument("--size", type=int, default=32, help="立方體尺寸，預設 32")
    parser.add_argument("--workers", type=int, default=None, help="並行處理的進程數，預設為 CPU 核心數")
    args = parser.parse_args()

    # 處理輸入：自動檢測是 zip 檔案還是資料夾
    input_path = Path(args.input).expanduser().resolve()
    use_input_zip = False
    input_zip_base_name = None
    
    if input_path.is_file():
        # 檢查是否為 zip 檔案
        if input_path.suffix.lower() == '.zip':
            use_input_zip = True
            input_zip_base_name = input_path.stem  # 不含擴展名的文件名
        else:
            console.print(f"[red]錯誤：輸入檔案不是 zip 格式：{input_path}[/red]")
            return
    elif input_path.is_dir():
        use_input_zip = False
    else:
        console.print(f"[red]錯誤：找不到輸入路徑：{input_path}[/red]")
        return

    # 處理輸出：根據路徑是否以 .zip 結尾判斷
    output_path_str = args.output
    use_output_zip = output_path_str.lower().endswith('.zip')
    
    if use_output_zip:
        output_path = Path(output_path_str).expanduser().resolve()
    else:
        output_path = Path(output_path_str).expanduser().resolve()

    # 處理輸入
    files = []
    if use_input_zip:
        # 從 zip 讀取
        console.print(f"[dim]正在從 zip 讀取檔案...[/dim]")
        with zipfile.ZipFile(input_path, 'r') as zip_file:
            for zip_info in zip_file.namelist():
                if zip_info.endswith('.litematic') and not zip_info.endswith('/'):
                    # 保持 zip 中的原始路径（使用正斜杠，符合 ZIP 标准）
                    # 使用 Path 对象便于后续处理，但保留原始路径字符串用于输出
                    rel_path_str = zip_info  # 原始路径字符串（如 "folder1/folder2/file.litematic"）
                    rel_path = Path(zip_info)  # Path 对象用于处理
                    file_data = zip_file.read(zip_info)
                    files.append((rel_path_str, file_data))
        
        if not files:
            console.print(f"[red]在 zip 檔案中找不到任何 .litematic 檔案[/red]")
            return
    else:
        # 從資料夾讀取
        for fpath in sorted(input_path.rglob("*.litematic")):
            if fpath.is_file():
                rel_path = fpath.relative_to(input_path)
                files.append((rel_path, str(fpath)))
        
        if not files:
            console.print(f"[red]在 {input_path} 及其子資料夾找不到任何 .litematic 檔案[/red]")
            return

    # 確定 worker 數量
    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    
    console.print(f"[green]共找到 {len(files)} 個 .litematic 檔案[/green]")
    if use_input_zip:
        console.print(f"[cyan]輸入 zip：{input_path}[/cyan]")
    else:
        console.print(f"[cyan]輸入資料夾：{input_path}[/cyan]")
    if use_output_zip:
        console.print(f"[cyan]輸出 zip：{output_path}[/cyan]")
    else:
        console.print(f"[cyan]輸出資料夾：{output_path}[/cyan]")
    console.print(f"[cyan]立方體尺寸：{args.size}x{args.size}x{args.size}[/cyan]")
    console.print(f"[cyan]並行進程數：{num_workers}[/cyan]")

    # 處理輸出目錄（僅當不使用zip輸出時）
    if not use_output_zip:
        # 預先創建所有需要的輸出目錄
        console.print("[dim]正在預先創建輸出目錄結構...[/dim]")
        output_dirs = set()
        for rel_path, _ in files:
            # rel_path 可能是字符串（zip输入）或 Path（文件夹输入）
            if isinstance(rel_path, str):
                rel_path = Path(rel_path)
            out_rel_path = rel_path.with_suffix(".npz")
            out_abs_path = output_path / out_rel_path
            output_dirs.add(out_abs_path.parent)
        
        for out_dir in output_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]已創建 {len(output_dirs)} 個輸出目錄[/dim]")

    # 準備任務參數
    tasks = [(file_data, output_path, rel_path, args.size, use_output_zip) 
             for rel_path, file_data in files]
    
    failed_files = []
    success_count = 0
    output_results = {}  # 用於收集zip輸出的結果

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("轉換中", total=len(files))

        # 使用多進程並行處理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任務
            future_to_file = {executor.submit(process_single_file, task_args): task_args[2] 
                             for task_args in tasks}
            
            # 處理完成的任務
            for future in as_completed(future_to_file):
                success, rel_path, error, npz_data, out_rel_path_str = future.result()
                if success:
                    success_count += 1
                    if use_output_zip:
                        # 收集結果用於zip輸出，使用返回的輸出路徑（已處理好路徑分隔符）
                        output_results[out_rel_path_str] = npz_data
                else:
                    failed_files.append((rel_path, error))
                    console.print(f"[yellow]轉換失敗[/yellow] {rel_path}：{error}")
                
                progress.advance(task)

    # 如果使用zip輸出，將所有結果寫入zip
    if use_output_zip:
        # 驗證輸出路徑：如果路徑已存在，必須是文件，不能是目錄
        if output_path.exists():
            if output_path.is_dir():
                console.print(f"[red]錯誤：輸出 zip 路徑不能是目錄：{output_path}[/red]")
                console.print(f"[yellow]提示：請指定完整的文件路徑，例如：{output_path / 'output.zip'}[/yellow]")
                return
            # 如果已存在且是文件，可以覆蓋（zipfile 會處理）
        
        # 確保父目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[dim]正在將結果寫入 zip 檔案...[/dim]")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for rel_path_str, npz_data in output_results.items():
                # 直接使用輸入的結構，不添加額外的外層資料夾
                zip_file.writestr(rel_path_str, npz_data)
        console.print(f"[dim]已寫入 {len(output_results)} 個檔案到 zip[/dim]")

    # 輸出結果摘要
    console.print(f"\n[bold green]✅ 轉換完成！[/bold green]")
    console.print(f"[green]成功：{success_count} 個檔案[/green]")
    if failed_files:
        console.print(f"[yellow]失敗：{len(failed_files)} 個檔案[/yellow]")
    if use_output_zip:
        console.print(f"[cyan]輸出 zip：{output_path}[/cyan]")
    else:
        console.print(f"[cyan]輸出資料夾：{output_path}[/cyan]")


if __name__ == "__main__":
    main()