#!/usr/bin/env python3
"""
從 YAML 配置文件批次訓練 VAE
Batch Training VAE from YAML Configuration

使用方法:
    # 方法 1: 自動偵測（智能選擇）
    python batch_train_from_yaml.py
    
    # 方法 2: 指定配置文件
    python batch_train_from_yaml.py my_config.yaml

自動偵測規則:
    1. 優先使用 experiments_config.yaml（如果存在）
    2. 如果只有一個 .yaml/.yml 文件，自動使用
    3. 如果有多個文件，顯示選單讓用戶選擇
    4. 如果沒有文件，顯示錯誤提示
"""

import subprocess
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from glob import glob
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    import yaml
except ImportError:
    print("錯誤: 需要安裝 pyyaml")
    print("請執行: pip install pyyaml")
    sys.exit(1)

console = Console()

def find_yaml_configs():
    """尋找當前目錄下的 YAML 配置文件"""
    yaml_files = []
    for pattern in ['*.yaml', '*.yml']:
        yaml_files.extend(glob(pattern))
    return sorted(set(yaml_files))

def select_config_file():
    """自動選擇或讓用戶選擇配置文件"""
    # 如果命令列有指定，直接使用
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    # 尋找目錄下的 YAML 文件
    yaml_files = find_yaml_configs()
    
    # 沒有找到任何 YAML 文件
    if not yaml_files:
        console.print("[red]錯誤: 當前目錄下沒有找到 .yaml 或 .yml 配置文件[/red]")
        console.print("\n[yellow]請:[/yellow]")
        console.print("  1. 創建配置文件（如 experiments_config.yaml）")
        console.print("  2. 或使用: python batch_train_from_yaml.py <config_file.yaml>")
        sys.exit(1)
    
    # 優先使用 experiments_config.yaml（如果存在）
    if 'experiments_config.yaml' in yaml_files:
        console.print(f"[green]✓[/green] 自動使用: [cyan]experiments_config.yaml[/cyan]")
        return 'experiments_config.yaml'
    
    # 只有一個 YAML 文件，直接使用
    if len(yaml_files) == 1:
        console.print(f"[green]✓[/green] 自動使用: [cyan]{yaml_files[0]}[/cyan]")
        return yaml_files[0]
    
    # 多個 YAML 文件，讓用戶選擇
    console.print(Panel.fit(
        "[bold yellow]發現多個 YAML 配置文件[/bold yellow]\n"
        "請選擇要使用的配置文件:",
        border_style="yellow"
    ))
    
    table = Table(box=box.SIMPLE)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("檔案名稱", style="magenta")
    
    for idx, fname in enumerate(yaml_files, 1):
        table.add_row(str(idx), fname)
    
    console.print(table)
    
    while True:
        try:
            choice = input(f"\n請輸入編號 (1-{len(yaml_files)}) 或按 Ctrl+C 取消: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(yaml_files):
                selected = yaml_files[idx - 1]
                console.print(f"[green]✓[/green] 已選擇: [cyan]{selected}[/cyan]\n")
                return selected
            else:
                console.print(f"[red]請輸入 1 到 {len(yaml_files)} 之間的數字[/red]")
        except KeyboardInterrupt:
            console.print("\n[red]已取消[/red]")
            sys.exit(0)
        except ValueError:
            console.print("[red]請輸入有效的數字[/red]")

def load_config(config_path='experiments_config.yaml'):
    """載入 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        console.print(f"[red]錯誤: 找不到配置文件 {config_path}[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]錯誤: YAML 格式錯誤[/red]")
        console.print(str(e))
        sys.exit(1)

def merge_configs(base_config, exp_params):
    """合併基礎配置和實驗參數"""
    merged = base_config.copy()
    merged.update(exp_params)
    return merged

def build_command(params, exp_name_with_timestamp):
    """根據參數構建命令"""
    params = params.copy()
    params['exp_name'] = exp_name_with_timestamp
    
    cmd = ['python', 'train-3d-vae-20251104.py']
    
    for key, value in params.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    return cmd

def run_experiment(exp_config, base_config, exp_idx, total_exps):
    """執行單個實驗"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = exp_config['name']
    exp_name_with_timestamp = f"{exp_name}_{timestamp}"
    
    # 合併配置
    params = merge_configs(base_config, exp_config.get('params', {}))
    
    console.print(Panel.fit(
        f"[bold cyan]實驗 {exp_idx}/{total_exps}[/bold cyan]\n"
        f"名稱: [magenta]{exp_name}[/magenta]\n"
        f"說明: {exp_config.get('description', 'N/A')}\n"
        f"完整名稱: [yellow]{exp_name_with_timestamp}[/yellow]",
        border_style="cyan"
    ))
    
    cmd = build_command(params, exp_name_with_timestamp)
    console.print(f"[dim]指令: {' '.join(cmd)}[/dim]\n")
    
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        success = True
        console.print(f"\n[bold green]✓ 實驗 {exp_name} 完成！[/bold green]\n")
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code: {e.returncode}"
        console.print(f"\n[bold red]✗ 實驗 {exp_name} 失敗: {error_msg}[/bold red]\n")
    except Exception as e:
        error_msg = str(e)
        console.print(f"\n[bold red]✗ 實驗 {exp_name} 錯誤: {error_msg}[/bold red]\n")
    
    elapsed_time = time.time() - start_time
    
    return {
        'exp_name': exp_name,
        'exp_name_full': exp_name_with_timestamp,
        'description': exp_config.get('description', ''),
        'success': success,
        'error_msg': error_msg,
        'duration_secs': elapsed_time,
        'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
    }

def format_duration(seconds):
    """格式化時間"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"

def main():
    # 自動選擇配置文件
    config_path = select_config_file()
    
    console.print(Panel.fit(
        "[bold cyan]🚀 批次訓練系統啟動 (YAML 配置)[/bold cyan]\n"
        f"配置文件: [yellow]{config_path}[/yellow]",
        border_style="cyan"
    ))
    
    # 載入配置
    config = load_config(config_path)
    base_config = config.get('base_config', {})
    experiments = config.get('experiments', [])
    
    if not experiments:
        console.print("[red]錯誤: 配置文件中沒有定義實驗[/red]")
        sys.exit(1)
    
    console.print(f"\n[bold]基礎配置:[/bold]")
    for key, value in base_config.items():
        console.print(f"  {key}: [cyan]{value}[/cyan]")
    
    # 顯示實驗列表
    table = Table(title="\n實驗清單", box=box.ROUNDED)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("名稱", style="magenta")
    table.add_column("說明", style="white")
    
    for idx, exp in enumerate(experiments, 1):
        table.add_row(
            str(idx), 
            exp['name'], 
            exp.get('description', 'N/A')
        )
    
    console.print("\n", table, "\n")
    
    # 確認執行
    console.print(f"[yellow]準備執行 {len(experiments)} 個實驗，這可能需要很長時間...[/yellow]")
    console.print("[yellow]按 Ctrl+C 可以隨時中斷[/yellow]\n")
    
    try:
        input("按 Enter 繼續，或 Ctrl+C 取消...")
    except KeyboardInterrupt:
        console.print("\n[red]已取消[/red]")
        return
    
    console.print()
    
    # 執行所有實驗
    results = []
    batch_start_time = time.time()
    
    for idx, exp_config in enumerate(experiments, 1):
        try:
            result = run_experiment(exp_config, base_config, idx, len(experiments))
            results.append(result)
        except KeyboardInterrupt:
            console.print("\n[yellow]用戶中斷批次訓練[/yellow]")
            break
    
    batch_duration = time.time() - batch_start_time
    
    # 生成結果表格
    console.print("\n")
    summary_table = Table(title="[bold cyan]批次訓練結果總覽[/bold cyan]", box=box.DOUBLE)
    summary_table.add_column("實驗名稱", style="magenta")
    summary_table.add_column("狀態", justify="center")
    summary_table.add_column("耗時", style="cyan", justify="right")
    summary_table.add_column("說明", style="white")
    
    success_count = 0
    for result in results:
        if result['success']:
            status = "[green]✓ 成功[/green]"
            success_count += 1
        else:
            status = "[red]✗ 失敗[/red]"
        
        summary_table.add_row(
            result['exp_name'],
            status,
            format_duration(result['duration_secs']),
            result['description']
        )
    
    console.print(summary_table)
    
    # 保存結果到 CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'batch_training_summary_{timestamp}.csv'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['exp_name', 'exp_name_full', 'description', 'success', 'error_msg', 
                     'duration_secs', 'duration_formatted', 'start_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            result['duration_formatted'] = format_duration(result['duration_secs'])
            writer.writerow(result)
    
    # 最終總結
    console.print(Panel.fit(
        f"[bold green]批次訓練完成！[/bold green]\n\n"
        f"成功: [green]{success_count}[/green] / {len(results)}\n"
        f"失敗: [red]{len(results) - success_count}[/red] / {len(results)}\n"
        f"總耗時: [cyan]{format_duration(batch_duration)}[/cyan]\n"
        f"結果已保存至: [yellow]{csv_path}[/yellow]",
        border_style="green"
    ))

if __name__ == '__main__':
    main()

