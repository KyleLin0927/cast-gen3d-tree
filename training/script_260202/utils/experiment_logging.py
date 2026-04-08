"""
實驗紀錄共用工具：指令還原、metadata CSV、訓練腳本備份（固定檔名、每 out_dir 一份）。

慣例與同目錄下 eval_diffusion_model.py、generate_16_voxel_diffusion*.py 對齊：
- metadata.csv：多列 key-value（parameter, value），訓練結束可再 append
- metadata_flat.csv：單列寬表，append 時會讀入舊列再與新欄位合併後覆寫
"""
from __future__ import annotations

import csv
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Union


def get_invocation_command() -> str:
    """還原本次執行的 shell-safe 指令（與 eval_diffusion_model 相同邏輯）。"""
    if not sys.argv:
        return ""
    exe = Path(sys.executable).name if sys.executable else "python"
    if exe.startswith("python"):
        exe = "python"
    return shlex.join([exe, *sys.argv])


def save_metadata(metadata: Dict[str, Any], output_dir: str) -> None:
    """寫入 metadata.csv 與 metadata_flat.csv（若已存在則覆寫兩者）。"""
    os.makedirs(output_dir, exist_ok=True)
    kv_path = os.path.join(output_dir, "metadata.csv")
    with open(kv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in metadata.items():
            w.writerow([k, v])

    flat_path = os.path.join(output_dir, "metadata_flat.csv")
    with open(flat_path, "w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=list(metadata.keys()))
        dw.writeheader()
        dw.writerow(metadata)

    print(f"✓ metadata: {kv_path}")


def append_metadata(metadata: Dict[str, Any], output_dir: str) -> None:
    """
    在 metadata.csv 末尾追加列；將新欄位合併進既有 metadata_flat.csv 後整份覆寫。
    若尚無 metadata.csv，則等同 save_metadata(metadata)。
    """
    kv_path = os.path.join(output_dir, "metadata.csv")
    if not os.path.exists(kv_path):
        save_metadata(metadata, output_dir)
        return

    with open(kv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in metadata.items():
            w.writerow([k, v])

    flat_path = os.path.join(output_dir, "metadata_flat.csv")
    existing: Dict[str, Any] = {}
    if os.path.exists(flat_path):
        with open(flat_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                row = next(reader, None)
                if row:
                    existing = dict(row)

    merged = {**existing, **metadata}
    with open(flat_path, "w", newline="", encoding="utf-8") as f:
        dw = csv.DictWriter(f, fieldnames=list(merged.keys()))
        dw.writeheader()
        dw.writerow(merged)

    print(f"✓ Updated metadata_flat with appended fields: {output_dir}")


def copy_script_snapshot(
    script_path: Union[str, Path],
    out_dir: str,
) -> str:
    """
    將腳本複製到 out_dir，固定檔名 {stem}_snapshot{suffix}（同 out_dir 重跑會覆寫，僅保留一份）。
    回傳備份檔的絕對路徑字串。
    """
    sp = Path(script_path).resolve()
    snap_name = f"{sp.stem}_snapshot{sp.suffix}"
    dest = Path(out_dir).resolve() / snap_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sp, dest)
    print(f"✓ Script snapshot: {dest}")
    return str(dest)
