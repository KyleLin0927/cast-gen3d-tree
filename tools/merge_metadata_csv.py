#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合併模型目錄下的 metadata_flat CSV 檔案。

對於每個找到的 .pt 模型檔案，在同級目錄下尋找包含 "metadata_flat" 的 CSV 檔案，
並將所有 metadata 合併成一個 CSV 檔案。如果同一個目錄下有多個模型檔案，
它們會共用同一個 metadata CSV，但在合併的 CSV 中會為每個模型複製一行。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)


def find_metadata_csv(model_dir: Path) -> Path | None:
    """
    在指定目錄下尋找包含 "metadata_flat" 的 CSV 檔案。
    
    Args:
        model_dir: 模型檔案所在的目錄
        
    Returns:
        找到的 CSV 檔案路徑，如果沒找到則返回 None
    """
    csv_files = list(model_dir.glob("*metadata_flat*.csv"))
    if csv_files:
        # 如果有多個，返回第一個（按名稱排序）
        return sorted(csv_files)[0]
    return None


def read_metadata_csv(csv_path: Path) -> tuple[list[str], list[dict]]:
    """
    讀取 metadata CSV 檔案。
    
    Args:
        csv_path: CSV 檔案路徑
        
    Returns:
        (header, rows) 元組，header 是欄位名稱列表，rows 是資料行列表
    """
    header = []
    rows = []
    
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        raise RuntimeError(f"無法讀取 CSV 檔案 {csv_path}: {e}")
    
    return header, rows


def main():
    console = Console()
    parser = argparse.ArgumentParser(
        description="合併模型目錄下的 metadata_flat CSV 檔案。"
    )
    parser.add_argument(
        "--models_dir",
        required=True,
        help="包含模型檔案的目錄（會遞迴搜尋所有 .pt 檔案）",
    )
    parser.add_argument(
        "--output",
        default="metadata_summary.csv",
        help="輸出 CSV 檔案名稱（預設：metadata_summary.csv）",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        console.print(
            Panel.fit(
                f"[bold red]錯誤：模型目錄不存在或不是目錄[/bold red]\n"
                f"路徑: [yellow]{models_dir}[/yellow]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    # 遞迴搜尋所有 .pt 檔案
    model_files = sorted(list(models_dir.rglob("*.pt")))
    
    if not model_files:
        console.print(
            Panel.fit(
                f"[bold red]錯誤：在目錄中沒有找到 .pt 檔案[/bold red]\n"
                f"目錄: [yellow]{models_dir}[/yellow]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]合併 Metadata CSV[/bold cyan]\n"
            f"模型目錄: [yellow]{models_dir}[/yellow]\n"
            f"找到模型檔案數: [yellow]{len(model_files)}[/yellow]",
            border_style="cyan",
        )
    )

    # 按目錄分組模型檔案
    # dir_to_models: {目錄路徑: [模型檔案列表]}
    dir_to_models = defaultdict(list)
    for model_file in model_files:
        dir_to_models[model_file.parent].append(model_file)

    # 收集所有 metadata
    # metadata_map: {CSV路徑: (header, rows)}
    metadata_map = {}
    # model_metadata: [(模型檔案, CSV路徑)]
    model_metadata = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]搜尋 metadata CSV...",
            total=len(dir_to_models),
        )

        for model_dir, models in sorted(dir_to_models.items()):
            metadata_csv = find_metadata_csv(model_dir)
            
            if metadata_csv is None:
                console.print(
                    f"[yellow]警告：[/yellow] 目錄 [cyan]{model_dir}[/cyan] 下沒有找到 metadata_flat CSV，"
                    f"跳過 {len(models)} 個模型檔案"
                )
                progress.update(task, advance=1)
                continue

            # 讀取 metadata（如果還沒讀過）
            if metadata_csv not in metadata_map:
                try:
                    header, rows = read_metadata_csv(metadata_csv)
                    metadata_map[metadata_csv] = (header, rows)
                except Exception as e:
                    console.print(
                        f"[red]錯誤：[/red] 無法讀取 [cyan]{metadata_csv}[/cyan]: {e}"
                    )
                    progress.update(task, advance=1)
                    continue

            # 為每個模型檔案記錄對應的 metadata CSV
            for model_file in models:
                model_metadata.append((model_file, metadata_csv))

            progress.update(task, advance=1)

    if not model_metadata:
        console.print(
            Panel.fit(
                f"[bold red]錯誤：沒有找到任何有效的 metadata CSV 檔案[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    # 確定輸出檔案名稱
    output_path = Path(args.output).expanduser().resolve()
    
    # 如果輸出路徑是絕對路徑，直接使用；否則相對於 models_dir
    if output_path.is_absolute():
        output_csv = output_path
    else:
        output_csv = models_dir / args.output
    
    # 如果輸出路徑是一個已存在的目錄，在目錄下創建預設檔名
    if output_csv.exists() and output_csv.is_dir():
        output_csv = output_csv / "metadata_summary.csv"
        console.print(
            f"[yellow]提示：[/yellow] 輸出路徑是目錄，將在該目錄下創建 [cyan]{output_csv.name}[/cyan]"
        )
    
    # 如果輸出路徑沒有 .csv 後綴，自動添加
    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv.with_suffix(".csv")

    # 合併所有 metadata
    # 使用第一個找到的 CSV 的 header
    first_csv = model_metadata[0][1]
    header = metadata_map[first_csv][0]
    
    if not header:
        console.print(
            Panel.fit(
                f"[bold red]錯誤：無法取得 CSV header[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    # 添加模型檔案路徑欄位（相對於 models_dir）
    if "model_path" not in header:
        header = ["model_path"] + list(header)

    all_rows = []
    for model_file, metadata_csv in model_metadata:
        _, rows = metadata_map[metadata_csv]
        
        if not rows:
            # 如果 CSV 沒有資料行，創建一個空行
            row = {col: "" for col in header if col != "model_path"}
            row["model_path"] = str(model_file.relative_to(models_dir))
            all_rows.append(row)
        else:
            # 為每個資料行添加模型路徑
            for row in rows:
                new_row = dict(row)
                new_row["model_path"] = str(model_file.relative_to(models_dir))
                all_rows.append(new_row)

    # 寫入合併後的 CSV
    try:
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]錯誤：無法寫入輸出檔案[/bold red]\n"
                f"路徑: [yellow]{output_csv}[/yellow]\n"
                f"錯誤: {e}",
                border_style="red",
            )
        )
        raise SystemExit(1)

    console.print(
        Panel.fit(
            f"[bold green]✓ 合併完成！[/bold green]\n"
            f"處理的模型數量: [yellow]{len(model_metadata)}[/yellow]\n"
            f"合併的資料行數: [yellow]{len(all_rows)}[/yellow]\n"
            f"輸出檔案: [cyan]{output_csv}[/cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

