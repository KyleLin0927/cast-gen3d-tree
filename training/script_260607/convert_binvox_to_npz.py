#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_binvox_to_npz.py

把 ShapeNetVox32 的 ``.binvox``（32³ 二元佔據網格）批次轉成本專案用的 ``.npz``
（int8 標籤立方體），給 ``train_unet_diffusion.py --data_root`` 直接使用。

對齊 ``convert_litematic_to_npz.py`` 的輸出慣例：
  - npz 內唯一陣列鍵為 ``voxel``（int8）
  - 標籤 0=air, 1=wood(log), 2=leaf。binvox 是純佔據格，故 **occupancy → 1 (wood)**，
    **leaf (2) 一律留空**——與計畫「occupancy → wood channel、leaf 留空、in_channels=3 不改」一致。

輸入（ShapeNetVox32，三層結構 synset → model → 一個 binvox）：
  解壓後路徑為 ``<root>/<synset>/<model_id>/model.binvox``。
  ``--input`` 可指到 ShapeNetVox32 根、單一類別（如 ``.../03001627``）或任一上層；
  腳本一律遞迴抓所有 ``*.binvox``。可用 ``--synset 03001627`` 只挑椅子。

輸出（給 ``--data_root`` 直接吃）：
  ``<output>/train/<model_id>.npz``
  ``<output>/val/<model_id>.npz``
  ``<output>/test/<model_id>.npz``
  依 ``--val_ratio`` / ``--test_ratio`` 隨機切分（預設 80/10/10），``--seed`` 控制。
  注意：擴散訓練（train_unet_diffusion ``--data_root``）只讀 train/ 與 val/，**不會用到 test/**；
  test/ 在此是為了 (a) 與 ``--data_zip`` 路徑相容（該路徑要求 train/val/test 都存在），
  (b) 日後若要報生成分布指標（MMD/COV/1-NNA）可當真實參考集。設 ``--test_ratio 0`` 可不產生 test/。

可選 ``--n_preview N``：在 ``<output>/preview/`` 產生前 N 個樣本三視圖 PNG，給 Day-1 GATE
用眼睛檢查「腿/細結構在這個解析度還在嗎」。需要專案內 ``utils.voxel_label_projections``，
找不到就自動略過（不影響轉換）。

需求：``pip install numpy``（``rich`` 可選，無則退化為純文字進度）。

binvox 格式：ASCII header（``#binvox 1`` / ``dim`` / ``translate`` / ``scale`` / ``data``）後接
RLE body（value, count 交錯的位元組對）。儲存序為 x 最慢、z 次之、y 最快；本腳本轉成 (X, Y, Z)。
連通度指標與軸序無關，但統一軸序利於可視化與重現。

本腳本只讀輸入、只寫輸出，不修改任何來源檔。
"""

from __future__ import annotations

import argparse
import io
import multiprocessing
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---- rich（可選）----
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH = True
    _console = Console()
except Exception:  # pragma: no cover
    _HAS_RICH = False
    _console = None


def _print(msg: str) -> None:
    if _HAS_RICH and _console is not None:
        _console.print(msg)
    else:
        # 去掉 rich 標記
        import re

        print(re.sub(r"\[/?[^\]]*\]", "", msg))


# ==========================================
# binvox 解析
# ==========================================
def read_binvox_occupancy(path_or_bytes) -> np.ndarray:
    """讀單一 binvox，回傳 (X, Y, Z) 的 uint8 佔據陣列（值為 {0,1}）。"""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        f = io.BytesIO(path_or_bytes)
        close = True
    else:
        f = open(path_or_bytes, "rb")
        close = True
    try:
        header = f.readline().strip()
        if not header.startswith(b"#binvox"):
            raise ValueError(f"not a binvox file (header={header!r})")
        dims = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF in binvox header")
            s = line.strip()
            if s.startswith(b"dim"):
                dims = [int(x) for x in s.split()[1:]]
            elif s.startswith(b"translate"):
                pass
            elif s.startswith(b"scale"):
                pass
            elif s == b"data" or s.startswith(b"data"):
                break
        if not dims or len(dims) != 3:
            raise ValueError(f"bad dims in header: {dims}")
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    finally:
        if close:
            f.close()

    if raw.size % 2 != 0:
        raise ValueError("binvox RLE body has odd byte count")
    values = raw[0::2]
    counts = raw[1::2].astype(np.int64)
    total = int(counts.sum())
    expected = int(dims[0]) * int(dims[1]) * int(dims[2])
    if total != expected:
        raise ValueError(f"RLE expands to {total} voxels, expected {expected} (dims={dims})")

    data = np.repeat(values, counts).astype(np.uint8)
    # binvox 儲存序：x 最慢、z 次之、y 最快 → reshape 為 (X, Z, Y)
    data = data.reshape((dims[0], dims[1], dims[2]))
    # 轉成 (X, Y, Z)
    data = np.transpose(data, (0, 2, 1))
    return (data > 0).astype(np.uint8)


def fit_to_size(arr: np.ndarray, size: int) -> np.ndarray:
    """置中 pad 或 crop 到 (size, size, size)；若已是該尺寸則原樣回傳（轉 int8）。"""
    arr = arr.astype(np.int8)
    if arr.shape == (size, size, size):
        return arr
    out = np.zeros((size, size, size), dtype=np.int8)
    src_slices, dst_slices = [], []
    for n in arr.shape:
        if n >= size:
            start = (n - size) // 2
            src_slices.append(slice(start, start + size))
            dst_slices.append(slice(0, size))
        else:
            start = (size - n) // 2
            src_slices.append(slice(0, n))
            dst_slices.append(slice(start, start + n))
    out[dst_slices[0], dst_slices[1], dst_slices[2]] = arr[
        src_slices[0], src_slices[1], src_slices[2]
    ]
    return out


def convert_one(path_or_bytes, size: int = 32) -> np.ndarray:
    """單檔：binvox → (size³) int8 標籤立方體（occupancy→1）。"""
    occ = read_binvox_occupancy(path_or_bytes)
    return fit_to_size(occ, size)


# ==========================================
# 多進程 worker
# ==========================================
def _worker(task):
    src_path, out_path, size = task
    try:
        arr = convert_one(src_path, size=size)
        occ = int((arr > 0).sum())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), voxel=arr)
        return (True, str(src_path), None, occ)
    except Exception as e:  # noqa: BLE001
        return (False, str(src_path), repr(e), 0)


# ==========================================
# 預覽（GATE 用，可選）
# ==========================================
def _save_previews(npz_files, preview_dir: Path, n_preview: int) -> int:
    if n_preview <= 0:
        return 0
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from utils.voxel_label_projections import save_labels_and_projections
    except Exception as e:  # noqa: BLE001
        _print(f"[yellow]預覽略過：找不到 utils.voxel_label_projections（{e}）[/yellow]")
        return 0
    preview_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for p in npz_files[:n_preview]:
        try:
            with np.load(p) as z:
                labels = z["voxel"]
            stem = Path(p).stem
            save_labels_and_projections(labels, str(preview_dir / f"{stem}.png"), exp_name=stem)
            made += 1
        except Exception as e:  # noqa: BLE001
            _print(f"[yellow]預覽失敗 {p}: {e}[/yellow]")
    return made


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批次將 ShapeNetVox32 的 .binvox 轉為本專案 .npz（occupancy→wood，train/val 切分）。"
    )
    parser.add_argument("--input", required=True, help="輸入路徑（遞迴搜尋所有 *.binvox）")
    parser.add_argument("--output", required=True, help="輸出根目錄（會建立 train/ 與 val/）")
    parser.add_argument("--synset", default=None, help="只挑指定 synset id（如 03001627 椅子）；預設全收")
    parser.add_argument("--size", type=int, default=32, help="輸出立方體尺寸，預設 32（ShapeNetVox32 原生）")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="驗證集比例，預設 0.1")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="測試集比例，預設 0.1（設 0 則不產生 test/）")
    parser.add_argument("--seed", type=int, default=42, help="train/val/test 切分隨機種子")
    parser.add_argument("--workers", type=int, default=None, help="並行進程數，預設 CPU 核心數")
    parser.add_argument(
        "--limit", type=int, default=0, help="只處理前 N 個檔（>0 時生效）；給 Day-1 GATE 快速試跑用"
    )
    parser.add_argument(
        "--n_preview", type=int, default=0, help="另存前 N 個樣本三視圖 PNG 到 output/preview/（GATE 用）"
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        _print(f"[red]找不到輸入路徑：{input_path}[/red]")
        sys.exit(1)

    # 收集 binvox
    all_files = sorted(input_path.rglob("*.binvox"))
    if args.synset:
        syn = str(args.synset)
        all_files = [p for p in all_files if syn in p.parts]
    if not all_files:
        _print(f"[red]在 {input_path} 找不到任何 .binvox（synset 過濾={args.synset}）[/red]")
        sys.exit(1)
    if args.limit and args.limit > 0:
        all_files = all_files[: args.limit]

    # 輸出檔名用 model_id（binvox 的上一層資料夾名）；衝突時加序號
    def stem_for(p: Path) -> str:
        mid = p.parent.name
        if not mid or mid == input_path.name:
            mid = p.stem
        return mid

    seen: dict = {}
    stems = []
    for p in all_files:
        base = stem_for(p)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        stems.append(base)

    # train/val/test 切分（預設 80/10/10；--test_ratio 0 則只切 train/val）
    n = len(all_files)
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    val_ratio = max(0.0, float(args.val_ratio))
    test_ratio = max(0.0, float(args.test_ratio))
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if n >= 2:
        # 有比例的 split 至少 1 筆，且 train 至少留 1 筆
        if val_ratio > 0:
            n_val = max(1, n_val)
        if test_ratio > 0:
            n_test = max(1, n_test)
        while n_val + n_test > n - 1:
            if n_test > (1 if test_ratio > 0 else 0):
                n_test -= 1
            elif n_val > (1 if val_ratio > 0 else 0):
                n_val -= 1
            else:
                break
    else:
        n_val = 0
        n_test = 0
    val_idx = set(idx[:n_val])
    test_idx = set(idx[n_val:n_val + n_test])
    n_train = n - n_val - n_test

    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    dirs = {"train": train_dir, "val": val_dir, "test": test_dir}

    def _split_of(i: int) -> str:
        if i in val_idx:
            return "val"
        if i in test_idx:
            return "test"
        return "train"

    tasks = []
    for i, p in enumerate(all_files):
        sub = dirs[_split_of(i)]
        tasks.append((str(p), str(sub / f"{stems[i]}.npz"), int(args.size)))

    num_workers = args.workers if args.workers else multiprocessing.cpu_count()
    _print(f"[green]找到 {n} 個 .binvox[/green]（train={n_train}, val={n_val}, test={n_test}）")
    _print(f"[cyan]輸入：{input_path}[/cyan]")
    _print(
        f"[cyan]輸出：{output_path}  (size={args.size}, val_ratio={args.val_ratio}, "
        f"test_ratio={args.test_ratio}, seed={args.seed})[/cyan]"
    )
    _print(f"[cyan]並行進程數：{num_workers}[/cyan]")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    if n_test > 0:
        test_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    success = 0
    occ_list = []
    saved = {"train": [], "val": [], "test": []}

    def _record(res):
        nonlocal success
        ok, src, err, occ = res
        if ok:
            success += 1
            occ_list.append(occ)
        else:
            failed.append((src, err))
            _print(f"[yellow]轉換失敗[/yellow] {src}: {err}")

    if _HAS_RICH:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=_console,
        ) as progress:
            t = progress.add_task("轉換中", total=n)
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futs = {ex.submit(_worker, task): task for task in tasks}
                for fut in as_completed(futs):
                    _record(fut.result())
                    progress.advance(t)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(_worker, task): task for task in tasks}
            done = 0
            for fut in as_completed(futs):
                _record(fut.result())
                done += 1
                if done % 200 == 0:
                    _print(f"  {done}/{n}")

    # 收集輸出檔清單（給預覽用）
    for i, p in enumerate(all_files):
        sp = _split_of(i)
        fp = dirs[sp] / f"{stems[i]}.npz"
        if fp.exists():
            saved[sp].append(str(fp))

    _print(f"\n[bold green]✅ 完成[/bold green]：成功 {success} / {n}，失敗 {len(failed)}")
    if occ_list:
        occ_arr = np.array(occ_list, dtype=np.float64)
        vox_total = float(args.size**3)
        _print(
            "[bold]佔據體素統計（GATE 參考）[/bold]："
            f" mean={occ_arr.mean():.1f}, min={int(occ_arr.min())}, max={int(occ_arr.max())}"
            f"（佔比 mean={100.0 * occ_arr.mean() / vox_total:.2f}%）"
        )
        n_empty = int((occ_arr == 0).sum())
        if n_empty:
            _print(f"[yellow]注意：{n_empty} 個樣本佔據為 0（空立方體），可能 size 太小或檔案異常[/yellow]")
    _print(f"[cyan]train/ → {train_dir}[/cyan]")
    _print(f"[cyan]val/   → {val_dir}[/cyan]")
    if n_test > 0:
        _print(f"[cyan]test/  → {test_dir}[/cyan]")

    preview_src = saved["train"] or saved["val"] or saved["test"]
    made = _save_previews(preview_src, output_path / "preview", args.n_preview)
    if made:
        _print(f"[cyan]preview/ → {output_path / 'preview'}（{made} 張 PNG）[/cyan]")


if __name__ == "__main__":
    main()
