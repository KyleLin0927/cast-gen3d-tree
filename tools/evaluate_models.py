#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate 3D VAE reconstructions on test set (.npz, 32x32x32, classes: 0=air,1=log,2=leaves).

Usage example:
  python evaluate_models.py \
    --data_dir /path/to/stage-4-split/test \
    --models_dir /path/to/runs \
    --out_dir /path/to/result/evaluation \
    --device cuda

Outputs:
- Console: compact summary per model
- TXT (JSON Lines) and CSV will be created under --out_dir with base name:
  eval_<YYYY-MM-DD_HH-MM-SS>.txt  and  eval_<YYYY-MM-DD_HH-MM-SS>.csv
"""

import argparse
import json
from pathlib import Path
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel

try:
    from scipy.ndimage import label as cc_label
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 允許 numpy 相關的全域變數以支援 PyTorch 2.6+ 的 weights_only 安全檢查
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except (ImportError, AttributeError):
    # 如果 numpy 版本不同或結構不同，fallback 到 weights_only=False
    pass


# --- Models ---
class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class Encoder3D(nn.Module):
    def __init__(self, in_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.conv_in = nn.Conv3d(in_ch, base, 3, padding=1)
        self.down1 = nn.Sequential(ResBlock3D(base), nn.Conv3d(base, base*2, 4, 2, 1))
        self.down2 = nn.Sequential(ResBlock3D(base*2), nn.Conv3d(base*2, base*4, 4, 2, 1))
        self.down3 = nn.Sequential(ResBlock3D(base*4), nn.Conv3d(base*4, base*8, 4, 2, 1))
        self.mid = nn.Sequential(ResBlock3D(base*8), ResBlock3D(base*8), nn.GroupNorm(8, base*8), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mu = nn.Linear(base*8, latent_dim)
        self.logvar = nn.Linear(base*8, latent_dim)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.pool(h).flatten(1)
        return self.mu(h), self.logvar(h)

class Decoder3D(nn.Module):
    def __init__(self, out_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base*8)
        self.up0 = nn.Sequential(nn.Unflatten(1, (base*8, 1, 1, 1)))
        self.up1 = nn.Sequential(nn.ConvTranspose3d(base*8, base*4, 4, 2, 1), ResBlock3D(base*4))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(base*4, base*2, 4, 2, 1), ResBlock3D(base*2))
        self.up3 = nn.Sequential(nn.ConvTranspose3d(base*2, base, 4, 2, 1), ResBlock3D(base))
        self.up4 = nn.Sequential(nn.ConvTranspose3d(base, base//2, 4, 2, 1), ResBlock3D(base//2))
        self.up5 = nn.Sequential(nn.ConvTranspose3d(base//2, base//4, 4, 2, 1), ResBlock3D(base//4))
        self.out = nn.Conv3d(base//4, out_ch, 1)

    def forward(self, z):
        h = self.fc(z)
        h = self.up0(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)
        return self.out(h)

class VAE3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256):
        super().__init__()  # ← 修正：一定要呼叫 Module.__init__()
        self.encoder = Encoder3D(in_ch, base, latent_dim)
        self.decoder = Decoder3D(out_ch, base, latent_dim)

    def forward(self, x):
        mu, _ = self.encoder(x)
        return self.decoder(mu)  # deterministic reconstruction


# --- Metrics ---
def one_hot(labels, num_classes=3):
    oh = np.eye(num_classes, dtype=np.float32)[labels]
    return np.moveaxis(oh, -1, 0)

def compute_confusion_stats(pred, gt, num_classes=3):
    stats = []
    for c in range(num_classes):
        p, g = (pred == c), (gt == c)
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        stats.append((tp, fp, fn))
    return stats

def iou_from_stats(stats): return [tp/(tp+fp+fn) if tp+fp+fn>0 else 1 for tp,fp,fn in stats]
def dice_from_stats(stats): return [(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn>0 else 1 for tp,fp,fn in stats]
def occupancy_ratio(labels): return float((labels != 0).sum()) / labels.size
def aabb_spans(mask):
    if not mask.any(): return (0,0,0),(0,0)
    coords = np.array(np.where(mask))
    zmin,ymin,xmin = coords.min(1); zmax,ymax,xmax = coords.max(1)
    Lz,Ly,Lx = zmax-zmin+1, ymax-ymin+1, xmax-xmin+1
    return (Lz,Ly,Lx),(Ly/Lx if Lx>0 else 0, Lz/Lx if Lx>0 else 0)
def connected_components_count(mask):
    if not _HAS_SCIPY: return -1
    _, n = cc_label(mask.astype(np.uint8), structure=np.ones((3,3,3),dtype=np.int8))
    return int(n)


# --- Core Eval ---
@torch.no_grad()
def eval_model(model_path, test_files, device, progress=None, task=None):
    # 使用 weights_only=False 以支援包含 numpy 物件的檢查點（PyTorch 2.6+ 相容性）
    # 這些檢查點來自受信任的訓練過程，因此可以安全載入
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
        args = ckpt.get('args', {})
        base, latent = int(args.get('base',64)), int(args.get('latent_dim',256))
    else:
        state_dict, base, latent = ckpt, 64, 256
    model = VAE3D(3,3,base,latent).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    agg = np.zeros((3,3),dtype=np.int64)
    occ_pred, occ_gt = [], []
    aryx_err, azx_err = [], []
    comps_nonair, comps_log = [], []

    for idx, f in enumerate(test_files):
        with np.load(f, allow_pickle=False) as d:
            arr = d['arr_0'] if 'arr_0' in d else d[list(d.files)[0]]
        gt = arr.astype(np.uint8)
        x = torch.from_numpy(one_hot(gt)).unsqueeze(0).to(device)
        pred = torch.argmax(F.softmax(model(x)[0],dim=0),dim=0).cpu().numpy().astype(np.uint8)

        agg += np.array(compute_confusion_stats(pred,gt))
        occ_pred.append(occupancy_ratio(pred)); occ_gt.append(occupancy_ratio(gt))

        _,(aryx_p,azx_p)=aabb_spans(pred!=0)
        _,(aryx_g,azx_g)=aabb_spans(gt!=0)
        if aryx_g>0: aryx_err.append(abs(aryx_p-aryx_g))
        if azx_g>0: azx_err.append(abs(azx_p-azx_g))

        c_all = connected_components_count(pred!=0)
        c_log = connected_components_count(pred==1)
        if c_all>=0: comps_nonair.append(c_all)
        if c_log>=0: comps_log.append(c_log)
        
        # 更新進度條（如果提供）
        if progress is not None and task is not None:
            progress.update(task, advance=1)

    iou, dice = iou_from_stats(agg), dice_from_stats(agg)
    conn = {}
    if comps_nonair: conn["non_air_mean"]=float(np.mean(comps_nonair))
    if comps_log: conn["log_mean"]=float(np.mean(comps_log))
    if not _HAS_SCIPY: conn["note"]="scipy not available"

    return {
        "model": str(model_path),
        "iou": [float(v) for v in iou],
        "dice": [float(v) for v in dice],
        "iou_mean": float(np.mean(iou)),
        "dice_mean": float(np.mean(dice)),
        "occ_pred": float(np.mean(occ_pred)),
        "occ_gt": float(np.mean(occ_gt)),
        "occ_err": float(abs(np.mean(occ_pred)-np.mean(occ_gt))),
        "aryx_err": float(np.mean(aryx_err)) if aryx_err else 0.0,
        "azx_err": float(np.mean(azx_err)) if azx_err else 0.0,
        "connectivity": conn
    }


# --- Main ---
def main():
    console = Console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to test directory with .npz files")
    ap.add_argument("--models_dir", required=True, help="Root dir containing model checkpoints (*.pt|*.pth)")
    ap.add_argument("--out_dir", required=True, help="Output directory; program will create eval_<DATE_TIME>.txt and .csv here")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    test_dir = Path(args.data_dir)
    assert test_dir.is_dir(), f"{test_dir} not found"
    test_files = sorted(test_dir.glob("*.npz"))
    assert test_files, f"No .npz in {test_dir}"

    model_files = sorted(list(Path(args.models_dir).rglob("*.pt")) + list(Path(args.models_dir).rglob("*.pth")))
    assert model_files, f"No models in {args.models_dir}"

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_txt = out_dir / f"eval_{ts}.txt"
    out_csv = out_dir / f"eval_{ts}.csv"

    # 顯示啟動資訊
    console.print(Panel.fit(
        f"[bold cyan]3D VAE 模型評估[/bold cyan]\n"
        f"測試檔案數: [yellow]{len(test_files)}[/yellow]\n"
        f"模型數量: [yellow]{len(model_files)}[/yellow]\n"
        f"裝置: [yellow]{device}[/yellow]\n"
        f"輸出目錄: [cyan]{out_dir}[/cyan]",
        border_style="cyan"
    ))

    with out_txt.open("w",encoding="utf-8") as ft, open(out_csv,"w",newline="",encoding="utf-8") as fc:
        writer = csv.writer(fc)
        writer.writerow(["model","iou_mean","dice_mean","iou_air","iou_log","iou_leaves",
                         "dice_air","dice_log","dice_leaves",
                         "occ_pred","occ_gt","occ_err","aryx_err","azx_err",
                         "non_air_comp","log_comp","notes"])

        run_header = {
            "data_test_dir": str(test_dir.resolve()),
            "models_root": str(Path(args.models_dir).resolve()),
            "num_test_files": len(test_files),
            "num_models": len(model_files),
            "device": str(device),
            "scipy_connectivity": _HAS_SCIPY,
            "timestamp": ts
        }
        console.print(json.dumps({"run_header": run_header}, ensure_ascii=False))
        ft.write(json.dumps({"run_header": run_header}, ensure_ascii=False) + "\n")

        # 使用 Rich 進度條評估所有模型
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            models_task = progress.add_task(
                f"[cyan]評估模型中...",
                total=len(model_files)
            )

            for idx, mp in enumerate(model_files, 1):
                model_name = mp.name
                progress.update(models_task, description=f"[cyan]評估模型 {idx}/{len(model_files)}: {model_name}")
                
                try:
                    # 為每個模型創建測試檔案進度條
                    test_task = progress.add_task(
                        f"[dim]處理測試檔案...",
                        total=len(test_files),
                        visible=len(test_files) > 10  # 只有測試檔案很多時才顯示
                    )
                    
                    res = eval_model(mp, test_files, device, progress, test_task)
                    
                    # 移除測試檔案進度條
                    progress.remove_task(test_task)
                    
                    iou, dice = res["iou"], res["dice"]
                    console.print(
                        f"[green]✓[/green] [{idx}/{len(model_files)}] {model_name}\n"
                        f"  [dim]IoU={res['iou_mean']:.4f} Dice={res['dice_mean']:.4f} OccErr={res['occ_err']:.4f}[/dim]"
                    )
                    ft.write(json.dumps(res, ensure_ascii=False) + "\n")

                    conn = res["connectivity"]
                    writer.writerow([
                        res["model"], res["iou_mean"], res["dice_mean"],
                        iou[0], iou[1], iou[2],
                        dice[0], dice[1], dice[2],
                        res["occ_pred"], res["occ_gt"], res["occ_err"],
                        res["aryx_err"], res["azx_err"],
                        conn.get("non_air_mean", ""), conn.get("log_mean", ""),
                        conn.get("note", "")
                    ])
                    
                except Exception as e:
                    err = str(e)
                    console.print(f"[red]✗[/red] [{idx}/{len(model_files)}] {model_name}: [red]{err}[/red]")
                    ft.write(json.dumps({"model": str(mp), "error": err}, ensure_ascii=False) + "\n")
                    writer.writerow([str(mp)] + [""] * 15 + [err])
                
                # 更新模型進度條
                progress.update(models_task, advance=1)

    console.print(Panel.fit(
        f"[bold green]✓ 評估完成！[/bold green]\n"
        f"結果已保存至:\n"
        f"  [cyan]TXT: {out_txt}[/cyan]\n"
        f"  [cyan]CSV: {out_csv}[/cyan]",
        border_style="green"
    ))

if __name__ == "__main__":
    main()