"""
訓練雙頭 Topology Scorer（加噪 voxel + 時間步）。

輸入（--data_zip）：
- 單一 zip，**根目錄**須有**恰好一個** ``.csv``（子目錄內的 ``.csv`` 一律忽略；多個根層 ``.csv`` 則報錯）。格式對齊 ``generate_16_voxel_diffusion_bucket.py`` 產出之 ``sample_labels.csv``（含 ``category``、``base_connected_ratio``、``id``、``source_name`` 等）。
- 目錄 ``gt``、``positive``、``neg_float``、``neg_easy``、``neg_hard``，內為 ``.npz``（鍵名優先 ``voxel``，見 ``utils.voxel_npz_io``）。
- ``source_name``：可為相對路徑（如 ``npz/<category>/<file>.npz`` …）；若**僅檔名**（無 ``/``），則嘗試 ``<category>/<檔名>``。**category 為 ``positive`` 時**會同時嘗試 ``positive/`` 與 ``gt/``（含 ``positive/…`` 路徑時亦會再試 ``gt/…`` 對應檔）。

輸出（--out_dir）：
- metadata.csv / metadata_flat.csv：參數與 run 起訖、耗時、checkpoint 列表（見 utils/experiment_logging.py）
- training_history_{run_label}.csv：每 epoch 一列（train/val 的 loss、bce、mse_ratio 與耗時），慣例比照 train_unet_diffusion 之 training_history_*.csv
- train_scorer_snapshot.py：實驗開始時備份本腳本一份（同 out_dir 重跑會覆寫）
- scorer_checkpoints/{run_label}_ep{N}.pt：依 --save_every 儲存；run_label 為 ``--out_dir`` 最後一層目錄名，若無法取得則為 ``train_YYYYMMDD_HHMMSS``（與 train_unet_diffusion 相同）
- scorer_checkpoints/best.pt：驗證集 val_loss 最低時覆寫（內含 ``model_state_dict``、``epoch``、``val_loss`` 等）

範例：
  python train_scorer.py --out_dir ./runs/scorer_001 --data_zip /path/samples_bundle.zip
"""
import argparse
import csv
import math
import os
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from tqdm import tqdm

# 載入原本的 BetaSchedule 用於加噪
from train_unet_diffusion import BetaSchedule, q_sample

from utils.experiment_logging import append_metadata, copy_script_snapshot, get_invocation_command, save_metadata
from utils.voxel_npz_io import load_voxel_npz

# 與 train_unet_diffusion.py 相同：每 epoch 追加一列
SCORER_HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss",
    "train_bce",
    "train_mse_ratio",
    "val_loss",
    "val_bce",
    "val_mse_ratio",
    "epoch_time_secs",
    "cumulative_time_secs",
]


# ==========================================
# 0. Data zip：解壓與路徑
# ==========================================
def _normalize_zip_relpath(source_name: str, *, strip_npz_prefix: bool) -> str:
    """將 CSV 的 source_name 轉成相對於 zip 解壓根目錄的路徑（POSIX segments）；可選擇去掉 ``npz/`` 前綴。"""
    s = source_name.replace("\\", "/").strip().lstrip("/")
    while s.startswith("./"):
        s = s[2:]
    if strip_npz_prefix and s.lower().startswith("npz/"):
        s = s[4:].lstrip("/")
    parts = [p for p in s.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise ValueError(f"invalid source_name (no '..' allowed): {source_name!r}")
    return "/".join(parts)


def _candidate_npz_paths(data_root: Path, source_name: str) -> List[Path]:
    """依序嘗試：去掉 npz/ 前綴（扁平 category 目錄）、再嘗試原始相對路徑（zip 內含 npz/ 子目錄時）。"""
    root = data_root.resolve()
    seen = set()
    out: List[Path] = []
    for strip in (True, False):
        rel = _normalize_zip_relpath(source_name, strip_npz_prefix=strip)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        p = (root / rel).resolve()
        if _is_under_root(root, p):
            out.append(p)
    return out


def _is_under_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _category_npz_subdirs(category: str) -> Tuple[str, ...]:
    """解析 npz 時使用的子目錄名；positive 同時對應生成樣本與 GT 目錄。"""
    cat = str(category).strip()
    if cat == "positive":
        return ("positive", "gt")
    return (cat,)


def _is_zip_root_entry(name: str) -> bool:
    """Zip 成員是否在壓縮檔根層（非任何子目錄）。"""
    n = name.replace("\\", "/").rstrip("/")
    return "/" not in n


def prepare_scorer_data_zip(zip_path: str) -> Tuple[pd.DataFrame, str]:
    """
    解壓 data zip 到暫存目錄，回傳 (DataFrame, extract_root)。
    zip **根目錄**必須恰好一個 .csv；子目錄內的 .csv 不計入；忽略 __MACOSX。
    """
    zp = Path(zip_path).expanduser().resolve()
    if not zp.is_file():
        raise FileNotFoundError(f"--data_zip not found: {zp}")

    with zipfile.ZipFile(zp, "r") as zf:
        csv_members: List[str] = []
        for name in zf.namelist():
            if name.startswith("__MACOSX/"):
                continue
            base = Path(name).name
            if base.startswith("._"):
                continue
            if name.lower().endswith(".csv") and _is_zip_root_entry(name):
                csv_members.append(name)
        if len(csv_members) != 1:
            raise ValueError(
                f"expected exactly one top-level .csv in zip (subdirs ignored), "
                f"found {len(csv_members)}: {csv_members}"
            )
        csv_member = csv_members[0]
        extract_root = tempfile.mkdtemp(prefix="train_scorer_zip_")
        zf.extractall(extract_root)

    extract_path = Path(extract_root)
    csv_disk = extract_path / csv_member
    if not csv_disk.is_file():
        raise FileNotFoundError(f"extracted CSV missing: {csv_disk}")

    df = pd.read_csv(csv_disk)
    for col in ("category", "base_connected_ratio", "id"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column {col!r}; have {list(df.columns)}")
    return df, extract_root


# ==========================================
# 1. Dataset: 動態加噪與標籤處理
# ==========================================
class VoxelScorerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        betas: BetaSchedule,
        t_range=(300, 950),
        train_on: str = "xt",
    ):
        if train_on not in ("xt", "x0"):
            raise ValueError(f"train_on must be 'xt' or 'x0', got {train_on!r}")
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self._root_path = Path(data_root).resolve()
        # DataLoader workers 在 CPU 上取樣；alpha_bar 固定放 CPU，與 q_sample 一致
        self.alpha_bar_cpu = betas.alpha_bar.detach().cpu().contiguous().clone()
        self.t_min, self.t_max = t_range
        # "xt"：加噪 voxel + 隨機時間步（noise-aware classifier，原 CAST Path-A）
        # "x0"：乾淨 voxel + t=0（clean-domain scorer，給 Universal Guidance 用）
        self.train_on = train_on

    def _resolve_npz_path(self, row) -> Path:
        cat = str(row["category"]).strip()
        sn = row["source_name"] if "source_name" in row else None
        if sn is not None and pd.notna(sn) and str(sn).strip() != "":
            s = str(sn).strip()
            tried: List[Path] = []
            seen: set = set()

            def _add(paths: List[Path]) -> None:
                for p in paths:
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        tried.append(p)

            _add(_candidate_npz_paths(Path(self.data_root), s))
            rel_norm = _normalize_zip_relpath(s, strip_npz_prefix=True)
            # positive/… 在 zip 內也可能放在 gt/…
            if rel_norm.startswith("positive/"):
                gt_rel = "gt/" + rel_norm[len("positive/") :]
                p_gt = (Path(self.data_root) / gt_rel).resolve()
                if _is_under_root(self._root_path, p_gt):
                    _add([p_gt])
            # CSV 僅給檔名（無路徑）時：<category>/<file>；positive 另試 gt/
            if rel_norm and "/" not in rel_norm:
                for sub in _category_npz_subdirs(cat):
                    if any(x in sub for x in ("/", "\\", "..")) or sub.startswith("."):
                        continue
                    p_cat = (Path(self.data_root) / sub / rel_norm).resolve()
                    if _is_under_root(self._root_path, p_cat):
                        _add([p_cat])

            for p in tried:
                if p.is_file():
                    return p
            raise FileNotFoundError(
                f"npz not found for source_name={sn!r}; tried: {[str(x) for x in tried]}"
            )
        if any(x in cat for x in ("/", "\\", "..")) or cat.startswith("."):
            raise ValueError(f"invalid category for fallback npz path: {cat!r}")
        stem = f"sample_{int(row['id']):05d}.npz"
        tried_fb: List[Path] = []
        for sub in _category_npz_subdirs(cat):
            if any(x in sub for x in ("/", "\\", "..")) or sub.startswith("."):
                continue
            path = (Path(self.data_root) / sub / stem).resolve()
            if not _is_under_root(self._root_path, path):
                raise ValueError(f"npz path escapes data root: {path}")
            tried_fb.append(path)
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"npz not found for id={row['id']!r} category={cat!r} (no source_name); tried: {[str(x) for x in tried_fb]}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cat = str(row["category"]).strip()

        npz_path = self._resolve_npz_path(row)

        labels = load_voxel_npz(npz_path)

        # 轉為 [-1, 1] 的連續特徵 [3, 16, 16, 16]
        x_0 = self._labels_to_centered(labels)

        # 依 train_on 決定 scorer 看到的輸入
        if self.train_on == "x0":
            # clean-domain scorer：不加噪、t 固定為 0，直接吃乾淨 voxel
            t_tensor = torch.tensor([0], dtype=torch.long)
            model_input = x_0
        else:
            # 動態抽樣時間步 t
            t_int = np.random.randint(self.t_min, self.t_max + 1)
            t_tensor = torch.tensor([t_int], dtype=torch.long)

            # 動態加噪：與 train_unet_diffusion.q_sample 一致（噪聲在函式內抽樣）
            x_t, _ = q_sample(
                x_0.unsqueeze(0),
                t_tensor,
                self.alpha_bar_cpu,
                x_0.device,
            )
            model_input = x_t.squeeze(0)

        # 準備雙頭標籤
        # y_break: gt / positive 為 0 (完美)，其餘為 1 (瑕疵)
        y_break_val = 0.0 if cat in ("positive", "gt") else 1.0
        y_break = torch.tensor([y_break_val], dtype=torch.float32)

        # y_ratio: 直接取 csv 中的 base_connected_ratio
        y_ratio = torch.tensor([row["base_connected_ratio"]], dtype=torch.float32)

        return model_input, t_tensor.squeeze(0), y_break, y_ratio

    def _labels_to_centered(self, labels: np.ndarray) -> torch.Tensor:
        t_labels = torch.from_numpy(labels).long()
        onehot = F.one_hot(t_labels, num_classes=3).float()  # [16,16,16,3]
        onehot = onehot.permute(3, 0, 1, 2)  # [3, 16, 16, 16]
        return (onehot * 2.0) - 1.0


# ==========================================
# 2. Model: 雙頭時間感知 3D CNN
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TopologyScorer3D(nn.Module):
    def __init__(self, in_channels=3, base_dim=32, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv1 = nn.Conv3d(in_channels, base_dim, kernel_size=3, padding=1)
        self.time_embed1 = nn.Linear(time_dim, base_dim)
        self.pool1 = nn.MaxPool3d(2)  # 16 -> 8

        self.conv2 = nn.Conv3d(base_dim, base_dim * 2, kernel_size=3, padding=1)
        self.time_embed2 = nn.Linear(time_dim, base_dim * 2)
        self.pool2 = nn.MaxPool3d(2)  # 8 -> 4

        self.conv3 = nn.Conv3d(base_dim * 2, base_dim * 4, kernel_size=3, padding=1)
        self.time_embed3 = nn.Linear(time_dim, base_dim * 4)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        hidden_dim = base_dim * 4

        # Head 1: 預測拓樸瑕疵 (接 BCEWithLogitsLoss，所以不加 Sigmoid)
        self.head_break = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Head 2: 預測連通比例 (接 MSELoss，加上 Sigmoid 確保輸出 0~1)
        self.head_ratio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h = self.conv1(x)
        h = h + self.time_embed1(t_emb).view(-1, h.shape[1], 1, 1, 1)
        h = F.gelu(self.pool1(h))

        h = self.conv2(h)
        h = h + self.time_embed2(t_emb).view(-1, h.shape[1], 1, 1, 1)
        h = F.gelu(self.pool2(h))

        h = self.conv3(h)
        h = h + self.time_embed3(t_emb).view(-1, h.shape[1], 1, 1, 1)
        h = F.gelu(h)

        h = self.global_pool(h).view(h.shape[0], -1)

        pred_break_logits = self.head_break(h)
        pred_ratio = self.head_ratio(h)
        return pred_break_logits, pred_ratio


def parse_args():
    p = argparse.ArgumentParser(description="Train topology scorer (dual-head 3D CNN).")
    p.add_argument(
        "--out_dir",
        type=str,
        default="./scorer_runs/default",
        help=(
            "實驗輸出目錄（metadata、腳本備份、checkpoint）。"
            "metadata 與 checkpoint 檔名前綴使用此路徑最後一層目錄名；"
            "若無法取得（例如根目錄 /）則使用 train_YYYYMMDD_HHMMSS。"
        ),
    )
    p.add_argument(
        "--data_zip",
        type=str,
        required=True,
        help="資料 zip：內含恰好一個 .csv 與 gt/positive/neg_float/neg_easy/neg_hard 等目錄之 .npz",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_ratio", type=float, default=10.0, help="MSE (ratio) loss 權重")
    p.add_argument("--t_min", type=int, default=300)
    p.add_argument("--t_max", type=int, default=950)
    p.add_argument(
        "--train_on",
        type=str,
        default="xt",
        choices=["xt", "x0"],
        help=(
            "scorer 訓練輸入：'xt'＝加噪 voxel + 隨機 t（原 noise-aware classifier，Path-A）；"
            "'x0'＝乾淨 voxel + t=0（Universal Guidance 用的 clean-domain scorer）。"
            "選 'x0' 時 --t_min/--t_max 不生效。"
        ),
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=10, help="每 N 個 epoch 存一次 checkpoint（0 表示只訓練不存）")
    p.add_argument("--beta_T", type=int, default=1000, help="BetaSchedule T（與 diffusion 訓練一致）")
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="驗證集比例（由同一 zip 資料 random_split；至少各保留 1 筆）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/val 切分用的隨機種子（僅影響 random_split）",
    )
    p.add_argument("--notes", type=str, default="", help="實驗備註（寫入 metadata）")
    return p.parse_args()


# ==========================================
# 3. Training Loop
# ==========================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    out_dir_leaf = Path(out_dir).name
    if not out_dir_leaf:
        out_dir_leaf = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Run label (metadata / checkpoint prefix): {out_dir_leaf} (from last component of --out_dir)")

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "scorer_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    t_start = datetime.now()
    script_snapshot_path = ""
    if "__file__" in globals():
        script_snapshot_path = copy_script_snapshot(__file__, out_dir)

    data_zip_abs = str(Path(args.data_zip).expanduser().resolve())
    labels_df, zip_extract_root = prepare_scorer_data_zip(data_zip_abs)
    print(f"✓ Data zip extracted to: {zip_extract_root} ({len(labels_df)} rows)")

    betas = BetaSchedule(T=args.beta_T, schedule=args.beta_schedule)
    dataset = VoxelScorerDataset(
        labels_df, zip_extract_root, betas, t_range=(args.t_min, args.t_max),
        train_on=args.train_on,
    )
    print(f"Scorer training input: train_on={args.train_on} "
          f"({'乾淨 x_0 + t=0 (UG clean scorer)' if args.train_on == 'x0' else '加噪 x_t + 隨機 t (Path-A)'})")
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError(f"Need at least 2 samples for train/val split, got {n_total}")
    n_val = int(round(n_total * float(args.val_ratio)))
    n_val = max(1, min(n_val, n_total - 1))
    n_train = n_total - n_val
    split_gen = torch.Generator().manual_seed(int(args.seed))
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f"Train/val split: n_train={n_train}, n_val={n_val} (val_ratio={args.val_ratio}, seed={args.seed})")

    history_csv_path = os.path.join(out_dir, f"training_history_{out_dir_leaf}.csv")

    meta: Dict = {
        "run_start": t_start.strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": out_dir_leaf,
        "out_dir_leaf": out_dir_leaf,
        "out_dir": out_dir,
        "command": get_invocation_command(),
        "script_snapshot_py": script_snapshot_path or "None",
        "training_history_csv": history_csv_path,
        "data_zip": data_zip_abs,
        "data_zip_extract_dir": zip_extract_root,
        "data_csv_rows": len(labels_df),
        "n_train": n_train,
        "n_val": n_val,
        "val_ratio": args.val_ratio,
        "split_seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_ratio": args.lambda_ratio,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "train_on": args.train_on,
        "num_workers": args.num_workers,
        "save_every": args.save_every,
        "beta_T": args.beta_T,
        "beta_schedule": args.beta_schedule,
        "device": str(device),
        "notes": args.notes or "None",
        "checkpoint_dir": ckpt_dir,
        "best_model_path": os.path.join(ckpt_dir, "best.pt"),
    }
    save_metadata(meta, out_dir)

    with open(history_csv_path, "w", newline="", encoding="utf-8") as hf:
        csv.DictWriter(hf, fieldnames=SCORER_HISTORY_FIELDNAMES).writeheader()
    print(f"✓ training history: {history_csv_path}")

    model = TopologyScorer3D().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion_break = nn.BCEWithLogitsLoss()
    criterion_ratio = nn.MSELoss()

    t_train0 = time.perf_counter()
    saved_ckpts = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_pt_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_mse = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x_t, t, y_break, y_ratio in pbar:
            x_t, t = x_t.to(device), t.to(device)
            y_break, y_ratio = y_break.to(device), y_ratio.to(device)

            optimizer.zero_grad()

            pred_break_logits, pred_ratio = model(x_t, t)

            loss_break = criterion_break(pred_break_logits, y_break)
            loss_ratio = criterion_ratio(pred_ratio, y_ratio)

            loss = loss_break + args.lambda_ratio * loss_ratio

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += loss_break.item()
            epoch_mse += loss_ratio.item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "BCE": f"{loss_break.item():.4f}",
                    "MSE": f"{loss_ratio.item():.4f}",
                }
            )

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_bce / n_batches
        avg_mse = epoch_mse / n_batches

        model.eval()
        val_loss_acc = 0.0
        val_bce_acc = 0.0
        val_mse_acc = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x_t, t, y_break, y_ratio in val_loader:
                x_t, t = x_t.to(device), t.to(device)
                y_break, y_ratio = y_break.to(device), y_ratio.to(device)
                pred_break_logits, pred_ratio = model(x_t, t)
                loss_break = criterion_break(pred_break_logits, y_break)
                loss_ratio = criterion_ratio(pred_ratio, y_ratio)
                loss = loss_break + args.lambda_ratio * loss_ratio
                val_loss_acc += loss.item()
                val_bce_acc += loss_break.item()
                val_mse_acc += loss_ratio.item()
                n_val_batches += 1
        avg_val_loss = val_loss_acc / max(n_val_batches, 1)
        avg_val_bce = val_bce_acc / max(n_val_batches, 1)
        avg_val_mse = val_mse_acc / max(n_val_batches, 1)

        epoch_secs = time.perf_counter() - epoch_t0
        cum_secs = time.perf_counter() - t_train0
        print(
            f"Epoch {epoch + 1} | train loss {avg_loss:.4f} (bce {avg_bce:.4f}, mse_r {avg_mse:.4f}) | "
            f"val loss {avg_val_loss:.4f} (bce {avg_val_bce:.4f}, mse_r {avg_val_mse:.4f})"
        )

        history_row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_bce": avg_bce,
            "train_mse_ratio": avg_mse,
            "val_loss": avg_val_loss,
            "val_bce": avg_val_bce,
            "val_mse_ratio": avg_val_mse,
            "epoch_time_secs": epoch_secs,
            "cumulative_time_secs": cum_secs,
        }
        try:
            with open(history_csv_path, "a", newline="", encoding="utf-8") as hf:
                csv.DictWriter(hf, fieldnames=SCORER_HISTORY_FIELDNAMES).writerow(history_row)
        except OSError as e:
            print(f"⚠ Failed to append training history: {e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "val_loss": avg_val_loss,
                    "val_bce": avg_val_bce,
                    "val_mse_ratio": avg_val_mse,
                    "model_state_dict": model.state_dict(),
                },
                best_pt_path,
            )
            print(f"✓ best.pt (epoch {best_epoch}, val_loss {best_val_loss:.6f})")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            fname = f"{out_dir_leaf}_ep{epoch + 1}.pt"
            ckpt_path = os.path.join(ckpt_dir, fname)
            torch.save(model.state_dict(), ckpt_path)
            saved_ckpts.append(ckpt_path)
            print(f"✓ checkpoint: {ckpt_path}")

    elapsed = time.perf_counter() - t_train0
    meta_done = {
        "run_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_secs": f"{elapsed:.4f}",
        "checkpoints_saved": ";".join(saved_ckpts) if saved_ckpts else "None",
        "best_checkpoint": best_pt_path if best_epoch >= 0 else "None",
        "best_epoch": str(best_epoch) if best_epoch >= 0 else "None",
        "best_val_loss": f"{best_val_loss:.8f}" if best_epoch >= 0 else "None",
    }
    append_metadata(meta_done, out_dir)


if __name__ == "__main__":
    main()
