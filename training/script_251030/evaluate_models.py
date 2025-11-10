#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate 3D VAE reconstructions on test set (.npz, 32x32x32, classes: 0=air,1=log,2=leaves).

Usage example:
  python evaluate_models_display_para.py \
    --data_dir /path/to/stage-4-split/test \
    --models_dir /path/to/runs \
    --out_dir /path/to/result/evaluation \
    --device cuda \
    --no_amp \
    --skip_patterns last checkpoint \
    --include_weight_stats

Outputs:
- Console: compact summary per model with model parameters displayed
- TXT (JSON Lines) and CSV will be created under --out_dir with base name:
  eval_<YYYY-MM-DD_HH-MM-SS>.txt  and  eval_<YYYY-MM-DD_HH-MM-SS>.csv
  
The outputs include:
  - Evaluation metrics:
    * IoU, Dice (per class and mean)
    * Occupancy ratio and error
    * AABB span errors
    * Precision, Recall, F1 (non-air as positive class)
    * Connected components counts
  - Model parameters extracted from checkpoints:
    * Architecture: base, latent_dim
    * Parameter counts: total_params, trainable_params, model_size_mb
    * Training hyperparameters: lr, batch_size, epochs, kl_beta
    * Training state: checkpoint_epoch, best_val_loss
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

def compute_nonair_metrics(pred, gt):
    """計算 non-air（標籤 != 0）的 Precision、Recall、F1
    
    Args:
        pred: 預測標籤數組 (0=air, 1=log, 2=leaves)
        gt: 真實標籤數組 (0=air, 1=log, 2=leaves)
    
    Returns:
        (precision, recall, f1) 元組，如果無法計算則返回 (0, 0, 0)
    """
    # non-air 為陽性（pred != 0 和 gt != 0）
    pred_nonair = (pred != 0)
    gt_nonair = (gt != 0)
    
    tp = np.logical_and(pred_nonair, gt_nonair).sum()
    fp = np.logical_and(pred_nonair, ~gt_nonair).sum()
    fn = np.logical_and(~pred_nonair, gt_nonair).sum()
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(precision), float(recall), float(f1)
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


# --- Extract Model Weights Statistics ---
def extract_weight_stats(model, include_layer_details=False):
    """從模型中提取權重統計信息
    
    Args:
        model: PyTorch 模型
        include_layer_details: 是否包含每層的詳細統計信息
    
    Returns:
        包含權重統計信息的字典
    """
    weight_stats = {}
    all_weights = []
    layer_stats = {}
    
    # 遍歷所有參數
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 轉換為 numpy 進行統計
            weights_np = param.detach().cpu().numpy()
            all_weights.append(weights_np.flatten())
            
            if include_layer_details:
                layer_stats[name] = {
                    "shape": list(param.shape),
                    "mean": float(weights_np.mean()),
                    "std": float(weights_np.std()),
                    "min": float(weights_np.min()),
                    "max": float(weights_np.max()),
                    "numel": int(param.numel())
                }
    
    # 計算總體統計
    if all_weights:
        all_weights_flat = np.concatenate(all_weights)
        weight_stats = {
            "weight_mean": float(all_weights_flat.mean()),
            "weight_std": float(all_weights_flat.std()),
            "weight_min": float(all_weights_flat.min()),
            "weight_max": float(all_weights_flat.max()),
            "weight_abs_mean": float(np.abs(all_weights_flat).mean()),
            "weight_abs_max": float(np.abs(all_weights_flat).max()),
        }
        
        if include_layer_details:
            weight_stats["layer_details"] = layer_stats
    
    return weight_stats

# --- Extract Model Parameters ---
def extract_model_params(model, ckpt, include_weight_stats=False):
    """從檢查點和模型中提取參數信息"""
    params = {}
    
    # 從檢查點的 args 中提取訓練參數
    if isinstance(ckpt, dict) and 'args' in ckpt:
        args = ckpt['args']
        # 模型架構參數
        params['base'] = args.get('base', None)
        params['latent_dim'] = args.get('latent_dim', None)
        # 訓練超參數
        params['lr'] = args.get('lr', None)
        params['batch_size'] = args.get('batch_size', None)
        params['epochs'] = args.get('epochs', None)
        params['kl_beta'] = args.get('kl_beta', None)
        params['class_weights'] = args.get('class_weights', None)
        # 訓練狀態
        params['checkpoint_epoch'] = ckpt.get('epoch', None)
        params['best_val_loss'] = ckpt.get('best_val', None)
        # 其他可能的參數
        params['exp_name'] = args.get('exp_name', None)
        params['seed'] = args.get('seed', None)
        params['aug_mode'] = args.get('aug_mode', None)
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params['total_params'] = total_params
    params['trainable_params'] = trainable_params
    params['non_trainable_params'] = total_params - trainable_params
    
    # 計算模型大小（MB，假設 float32）
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    params['model_size_mb'] = model_size_mb
    
    # 提取權重統計信息（如果啟用）
    if include_weight_stats:
        weight_stats = extract_weight_stats(model, include_layer_details=False)
        params['weight_stats'] = weight_stats
    
    return params

# --- Core Eval ---
@torch.no_grad()
def eval_model(model_path, test_files, device, progress=None, task=None, use_amp=True, include_weight_stats=False):
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
    
    # 如果禁用 AMP，確保模型使用 float32 精度
    if not use_amp:
        model = model.float()
    
    # 提取模型參數信息（包括權重統計，如果啟用）
    model_params = extract_model_params(model, ckpt, include_weight_stats=include_weight_stats)

    agg = np.zeros((3,3),dtype=np.int64)
    occ_pred, occ_gt = [], []
    aryx_err, azx_err = [], []
    comps_nonair, comps_log = [], []
    precision_list, recall_list, f1_list = [], [], []

    for idx, f in enumerate(test_files):
        with np.load(f, allow_pickle=False) as d:
            arr = d['arr_0'] if 'arr_0' in d else d[list(d.files)[0]]
        gt = arr.astype(np.uint8)
        x = torch.from_numpy(one_hot(gt)).unsqueeze(0).to(device)
        # 如果禁用 AMP，確保輸入使用 float32
        if not use_amp:
            x = x.float()
        
        # 根據 use_amp 決定是否使用 autocast
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                pred = torch.argmax(F.softmax(model(x)[0],dim=0),dim=0).cpu().numpy().astype(np.uint8)
        else:
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
        
        # 計算 non-air 的 Precision、Recall、F1
        precision, recall, f1 = compute_nonair_metrics(pred, gt)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
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
        "precision_nonair": float(np.mean(precision_list)),
        "recall_nonair": float(np.mean(recall_list)),
        "f1_nonair": float(np.mean(f1_list)),
        "connectivity": conn,
        "model_params": model_params  # 添加模型參數信息
    }


# --- Main ---
def main():
    console = Console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to test directory with .npz files")
    ap.add_argument("--models_dir", required=True, help="Root dir containing model checkpoints (*.pt|*.pth)")
    ap.add_argument("--out_dir", required=True, help="Output directory; program will create eval_<DATE_TIME>.txt and .csv here")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP (Automatic Mixed Precision) to use full float32 precision for evaluation")
    ap.add_argument("--skip_patterns", nargs="*", default=["last"], 
                    help="File name patterns to skip (case-insensitive). Default: ['last']. Example: --skip_patterns last checkpoint")
    ap.add_argument("--include_weight_stats", action="store_true", 
                    help="Include weight statistics (mean, std, min, max) in model parameters")
    args = ap.parse_args()

    # 解析並驗證測試資料目錄
    test_dir = Path(args.data_dir).expanduser().resolve()
    if not test_dir.exists():
        console.print(Panel.fit(
            f"[bold red]錯誤：測試資料目錄不存在[/bold red]\n\n"
            f"提供的路徑: [yellow]{args.data_dir}[/yellow]\n"
            f"解析後絕對路徑: [yellow]{test_dir}[/yellow]\n"
            f"當前工作目錄: [dim]{Path.cwd()}[/dim]\n\n"
            f"請檢查路徑是否正確，或使用絕對路徑。",
            border_style="red"
        ))
        raise SystemExit(1)
    if not test_dir.is_dir():
        console.print(Panel.fit(
            f"[bold red]錯誤：指定的路徑不是目錄[/bold red]\n\n"
            f"路徑: [yellow]{test_dir}[/yellow]\n"
            f"請確認這是一個目錄而非檔案。",
            border_style="red"
        ))
        raise SystemExit(1)
    
    test_files = sorted(test_dir.glob("*.npz"))
    if not test_files:
        console.print(Panel.fit(
            f"[bold red]錯誤：測試目錄中沒有 .npz 檔案[/bold red]\n\n"
            f"目錄: [yellow]{test_dir}[/yellow]\n"
            f"請確認目錄中包含 .npz 格式的測試檔案。",
            border_style="red"
        ))
        raise SystemExit(1)

    # 解析並驗證模型目錄
    models_dir = Path(args.models_dir).expanduser().resolve()
    if not models_dir.exists():
        console.print(Panel.fit(
            f"[bold red]錯誤：模型目錄不存在[/bold red]\n\n"
            f"提供的路徑: [yellow]{args.models_dir}[/yellow]\n"
            f"解析後絕對路徑: [yellow]{models_dir}[/yellow]\n"
            f"當前工作目錄: [dim]{Path.cwd()}[/dim]\n\n"
            f"請檢查路徑是否正確，或使用絕對路徑。",
            border_style="red"
        ))
        raise SystemExit(1)
    if not models_dir.is_dir():
        console.print(Panel.fit(
            f"[bold red]錯誤：指定的路徑不是目錄[/bold red]\n\n"
            f"路徑: [yellow]{models_dir}[/yellow]\n"
            f"請確認這是一個目錄而非檔案。",
            border_style="red"
        ))
        raise SystemExit(1)
    
    model_files = sorted(list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth")))
    total_models_before_filter = len(model_files)
    
    # 根據參數過濾模型文件
    skip_patterns = args.skip_patterns if args.skip_patterns else []
    if skip_patterns:
        skip_patterns_lower = [p.lower() for p in skip_patterns]
        model_files = [f for f in model_files 
                      if not any(pattern in f.name.lower() for pattern in skip_patterns_lower)]
    
    filtered_count = total_models_before_filter - len(model_files)
    # skip_patterns 現在在整個函數中可用
    if not model_files:
        skip_info = f"過濾規則: 已跳過名稱包含 {skip_patterns} 的檔案\n" if skip_patterns else ""
        console.print(Panel.fit(
            f"[bold red]錯誤：模型目錄中沒有找到模型檔案[/bold red]\n\n"
            f"目錄: [yellow]{models_dir}[/yellow]\n"
            f"搜尋模式: *.pt, *.pth（遞迴搜尋所有子目錄）\n"
            f"{skip_info}"
            f"請確認目錄中包含模型檢查點檔案（.pt 或 .pth）。",
            border_style="red"
        ))
        raise SystemExit(1)

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp  # 如果指定了 --no_amp，則禁用 AMP

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_txt = out_dir / f"eval_{ts}.txt"
    out_csv = out_dir / f"eval_{ts}.csv"

    # 顯示啟動資訊
    amp_status = "禁用" if not use_amp else "啟用"
    if filtered_count > 0 and skip_patterns:
        filter_info = f"（已過濾 {filtered_count} 個包含 {skip_patterns} 的檔案）"
    else:
        filter_info = ""
    console.print(Panel.fit(
        f"[bold cyan]3D VAE 模型評估[/bold cyan]\n"
        f"測試檔案數: [yellow]{len(test_files)}[/yellow]\n"
        f"模型數量: [yellow]{len(model_files)}[/yellow] {filter_info}\n"
        f"跳過模式: [yellow]{skip_patterns if skip_patterns else '無'}[/yellow]\n"
        f"裝置: [yellow]{device}[/yellow]\n"
        f"CUDA AMP: [yellow]{amp_status}[/yellow]\n"
        f"輸出目錄: [cyan]{out_dir}[/cyan]",
        border_style="cyan"
    ))

    with out_txt.open("w",encoding="utf-8") as ft, open(out_csv,"w",newline="",encoding="utf-8") as fc:
        writer = csv.writer(fc)
        writer.writerow(["model","iou_mean","dice_mean","iou_air","iou_log","iou_leaves",
                         "dice_air","dice_log","dice_leaves",
                         "occ_pred","occ_gt","occ_err","aryx_err","azx_err",
                         "precision_nonair","recall_nonair","f1_nonair",
                         "non_air_comp","log_comp","notes",
                         "base","latent_dim","total_params","trainable_params","model_size_mb",
                         "lr","batch_size","epochs","kl_beta","checkpoint_epoch","best_val_loss"])

        run_header = {
            "data_test_dir": str(test_dir),
            "models_root": str(models_dir),
            "num_test_files": len(test_files),
            "num_models": len(model_files),
            "num_models_filtered": filtered_count,
            "skip_patterns": skip_patterns,
            "device": str(device),
            "use_amp": use_amp,
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
                    
                    res = eval_model(mp, test_files, device, progress, test_task, use_amp=use_amp, include_weight_stats=args.include_weight_stats)
                    
                    # 移除測試檔案進度條
                    progress.remove_task(test_task)
                    
                    iou, dice = res["iou"], res["dice"]
                    params = res.get("model_params", {})
                    
                    # 顯示評估結果
                    console.print(
                        f"[green]✓[/green] [{idx}/{len(model_files)}] {model_name}\n"
                        f"  [dim]IoU={res['iou_mean']:.4f} Dice={res['dice_mean']:.4f} OccErr={res['occ_err']:.4f}[/dim]\n"
                        f"  [dim]Precision={res['precision_nonair']:.4f} Recall={res['recall_nonair']:.4f} F1={res['f1_nonair']:.4f} (non-air)[/dim]"
                    )
                    
                    # 顯示模型參數（如果存在）
                    if params:
                        param_lines = []
                        # 架構參數
                        if params.get('base') is not None or params.get('latent_dim') is not None:
                            arch = []
                            if params.get('base') is not None:
                                arch.append(f"base={params['base']}")
                            if params.get('latent_dim') is not None:
                                arch.append(f"latent={params['latent_dim']}")
                            if arch:
                                param_lines.append(f"  [cyan]架構:[/cyan] {', '.join(arch)}")
                        
                        # 參數量
                        if params.get('total_params') is not None:
                            total = params['total_params']
                            trainable = params.get('trainable_params', total)
                            size_mb = params.get('model_size_mb', 0)
                            param_lines.append(f"  [cyan]參數:[/cyan] 總計={total:,} (可訓練={trainable:,}), 大小={size_mb:.2f} MB")
                        
                        # 訓練超參數
                        train_params = []
                        if params.get('lr') is not None:
                            train_params.append(f"lr={params['lr']}")
                        if params.get('batch_size') is not None:
                            train_params.append(f"bs={params['batch_size']}")
                        if params.get('epochs') is not None:
                            train_params.append(f"epochs={params['epochs']}")
                        if params.get('kl_beta') is not None:
                            train_params.append(f"kl_beta={params['kl_beta']}")
                        if train_params:
                            param_lines.append(f"  [cyan]訓練:[/cyan] {', '.join(train_params)}")
                        
                        # 訓練狀態
                        if params.get('checkpoint_epoch') is not None:
                            param_lines.append(f"  [cyan]檢查點:[/cyan] epoch={params['checkpoint_epoch']}")
                        if params.get('best_val_loss') is not None:
                            param_lines.append(f"  [cyan]最佳驗證損失:[/cyan] {params['best_val_loss']:.4f}")
                        
                        # 權重統計（如果啟用）
                        weight_stats = params.get('weight_stats', {})
                        if weight_stats:
                            weight_info = []
                            if 'weight_mean' in weight_stats:
                                weight_info.append(f"mean={weight_stats['weight_mean']:.6f}")
                            if 'weight_std' in weight_stats:
                                weight_info.append(f"std={weight_stats['weight_std']:.6f}")
                            if 'weight_min' in weight_stats:
                                weight_info.append(f"min={weight_stats['weight_min']:.6f}")
                            if 'weight_max' in weight_stats:
                                weight_info.append(f"max={weight_stats['weight_max']:.6f}")
                            if weight_info:
                                param_lines.append(f"  [cyan]權重統計:[/cyan] {', '.join(weight_info)}")
                        
                        if param_lines:
                            console.print("\n".join(param_lines))
                    
                    ft.write(json.dumps(res, ensure_ascii=False) + "\n")

                    conn = res["connectivity"]
                    params = res.get("model_params", {})
                    
                    # 構建CSV行，包含參數信息
                    row = [
                        res["model"], res["iou_mean"], res["dice_mean"],
                        iou[0], iou[1], iou[2],
                        dice[0], dice[1], dice[2],
                        res["occ_pred"], res["occ_gt"], res["occ_err"],
                        res["aryx_err"], res["azx_err"],
                        res["precision_nonair"], res["recall_nonair"], res["f1_nonair"],
                        conn.get("non_air_mean", ""), conn.get("log_mean", ""),
                        conn.get("note", "")
                    ]
                    
                    # 添加模型參數到CSV行
                    row.extend([
                        params.get("base", ""),
                        params.get("latent_dim", ""),
                        params.get("total_params", ""),
                        params.get("trainable_params", ""),
                        params.get("model_size_mb", ""),
                        params.get("lr", ""),
                        params.get("batch_size", ""),
                        params.get("epochs", ""),
                        params.get("kl_beta", ""),
                        params.get("checkpoint_epoch", ""),
                        params.get("best_val_loss", "")
                    ])
                    
                    writer.writerow(row)
                    
                except Exception as e:
                    err = str(e)
                    console.print(f"[red]✗[/red] [{idx}/{len(model_files)}] {model_name}: [red]{err}[/red]")
                    ft.write(json.dumps({"model": str(mp), "error": err}, ensure_ascii=False) + "\n")
                    # CSV行包含：model + 18個評估指標欄位 + 3個新指標（precision/recall/f1） + 11個參數欄位 = 33列
                    # 錯誤信息放在最後，所以是 model + 30個空欄位 + error
                    writer.writerow([str(mp)] + [""] * 30 + [err])
                
                # 更新模型進度條
                progress.update(models_task, advance=1)
        
        # 在文件末尾添加指標計算說明
        metrics_explanation = """
        
================================================================================
評估指標計算方法說明
================================================================================

1. IoU (Intersection over Union，交併比)
   計算方式：IoU = TP / (TP + FP + FN)
   說明：
   - TP (True Positive): 預測為該類別且實際為該類別
   - FP (False Positive): 預測為該類別但實際不是該類別
   - FN (False Negative): 實際為該類別但預測不是該類別
   - 分別計算 air、log、leaves 三個類別的 IoU
   - iou_mean: 三個類別 IoU 的平均值

2. Dice 係數 (Dice Coefficient)
   計算方式：Dice = 2*TP / (2*TP + FP + FN)
   說明：
   - 與 IoU 類似，但對重疊區域的權重不同
   - 分別計算 air、log、leaves 三個類別的 Dice
   - dice_mean: 三個類別 Dice 的平均值

3. Occupancy Ratio (佔用率)
   - occ_pred: 預測結果中非空氣（non-air）體素所佔比例
   - occ_gt: 真實標籤中非空氣體素所佔比例
   - occ_err: 預測佔用率與真實佔用率的絕對差值

4. AABB Span Errors (軸對齊邊界框誤差)
   - aryx_err: Y/X 軸比例誤差（預測與真實的 AABB 在 Y/X 比例上的絕對差值）
   - azx_err: Z/X 軸比例誤差（預測與真實的 AABB 在 Z/X 比例上的絕對差值）
   說明：
   - AABB 是包圍所有非空氣體素的最小軸對齊邊界框
   - 計算 AABB 的長寬高比例，用於評估形狀的相似性

5. Precision, Recall, F1 (精確率、召回率、F1分數)
   以 non-air（非空氣，即標籤 != 0）為陽性類別：
   - Precision (精確率) = TP / (TP + FP)
     說明：在所有預測為 non-air 的體素中，有多少是正確的
   - Recall (召回率) = TP / (TP + FN)
     說明：在所有實際為 non-air 的體素中，有多少被正確預測
   - F1 = 2 * (Precision * Recall) / (Precision + Recall)
     說明：Precision 和 Recall 的調和平均數，綜合評估性能
   其中：
   - TP: 預測為 non-air 且實際為 non-air
   - FP: 預測為 non-air 但實際為 air
   - FN: 實際為 non-air 但預測為 air

6. Connected Components (連通分量)
   - non_air_comp: 預測結果中所有 non-air 體素的連通分量數量
   - log_comp: 預測結果中 log（標籤=1）的連通分量數量
   說明：
   - 使用 3D 連通分量分析（3x3x3 鄰域）
   - 用於評估預測結果的結構完整性

7. 模型參數
   - base: 模型基礎通道數
   - latent_dim: 潛在空間維度
   - total_params: 模型總參數數量
   - trainable_params: 可訓練參數數量
   - model_size_mb: 模型大小（MB，假設 float32 精度）
   - lr: 學習率
   - batch_size: 批次大小
   - epochs: 訓練輪數
   - kl_beta: KL 散度權重係數
   - checkpoint_epoch: 檢查點保存時的訓練輪數
   - best_val_loss: 最佳驗證損失值

8. 權重統計（當使用 --include_weight_stats 時）
   - weight_mean: 所有權重的平均值
   - weight_std: 所有權重的標準差
   - weight_min: 所有權重的最小值
   - weight_max: 所有權重的最大值
   - weight_abs_mean: 所有權重絕對值的平均值
   - weight_abs_max: 所有權重絕對值的最大值

================================================================================
"""
        ft.write(metrics_explanation)
        
        # 在 CSV 文件末尾添加簡要說明（使用註釋格式）
        csv_explanation = [
            "",
            "# ========================================================================",
            "# 評估指標計算方法說明（詳細說明請參考 TXT 文件）",
            "# ========================================================================",
            "# IoU = TP / (TP + FP + FN) - 交併比，分別計算 air、log、leaves 三個類別",
            "# Dice = 2*TP / (2*TP + FP + FN) - Dice係數，分別計算三個類別",
            "# occ_pred/occ_gt: 預測/真實的非空氣體素佔用率，occ_err: 兩者的絕對差值",
            "# aryx_err/azx_err: AABB 的 Y/X 和 Z/X 軸比例誤差",
            "# Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2*P*R/(P+R) (以 non-air 為陽性)",
            "# non_air_comp/log_comp: 連通分量數量（3D，3x3x3 鄰域）",
            "# 模型參數: base, latent_dim, total_params, trainable_params, model_size_mb",
            "# 訓練參數: lr, batch_size, epochs, kl_beta, checkpoint_epoch, best_val_loss",
            "# 權重統計（--include_weight_stats）: weight_mean, std, min, max, abs_mean, abs_max",
            "# ========================================================================"
        ]
        for line in csv_explanation:
            fc.write(line + "\n")

    console.print(Panel.fit(
        f"[bold green]✓ 評估完成！[/bold green]\n"
        f"結果已保存至:\n"
        f"  [cyan]TXT: {out_txt}[/cyan]\n"
        f"  [cyan]CSV: {out_csv}[/cyan]",
        border_style="green"
    ))

if __name__ == "__main__":
    main()