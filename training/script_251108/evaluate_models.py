#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate 3D U-Net VAE checkpoints (.pt/.pth) on 32x32x32 voxel volumes.
Uses the deterministic mean of the posterior (mu) for reconstruction to avoid
stochastic variance during evaluation.
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# ---------------------------------------------------------------------------
# Model definitions (copied from training script to match checkpoint layout)
# ---------------------------------------------------------------------------


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
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


class Encoder3DUNetVAE(nn.Module):
    def __init__(self, in_ch=3, base=64, latent_dim=256):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 4, stride=2, padding=1),
            ResBlock3D(base * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base * 2, base * 4, 4, stride=2, padding=1),
            ResBlock3D(base * 4),
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(base * 4, base * 8, 4, stride=2, padding=1),
            ResBlock3D(base * 8),
        )
        self.mu = nn.Conv3d(base * 8, latent_dim, 1)
        self.logvar = nn.Conv3d(base * 8, latent_dim, 1)

    def forward(self, x, return_skips: bool = True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        mu = self.mu(e4)
        logvar = self.logvar(e4)
        skips = (e1, e2, e3) if return_skips else None
        return mu, logvar, skips


class Decoder3DUNetVAE(nn.Module):
    def __init__(self, out_ch=3, base=64, latent_dim=256, skip_levels: int = 3):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")

        self.base = base

        # Always allocate full channel capacity (3 skip levels)
        self.up1 = nn.ConvTranspose3d(latent_dim, base * 8, 4, stride=2, padding=1)
        self.rb1 = ResBlock3D(base * 8)

        self.up2 = nn.ConvTranspose3d(base * 8 + base * 4, base * 4, 4, stride=2, padding=1)
        self.rb2 = ResBlock3D(base * 4)

        self.up3 = nn.ConvTranspose3d(base * 4 + base * 2, base * 2, 4, stride=2, padding=1)
        self.rb3 = ResBlock3D(base * 2)

        self.out_block = nn.Sequential(
            nn.Conv3d(base * 2 + base, base, 3, padding=1),
            ResBlock3D(base),
        )
        self.out = nn.Conv3d(base, out_ch, 1)

        self.skip_gates = [0, 0, 0]
        self.set_skip_levels(skip_levels)

    def requires_skips(self) -> bool:
        return any(g > 0 for g in self.skip_gates)

    def set_skip_levels(self, levels: int):
        levels = int(max(0, min(3, levels)))
        gates = [0, 0, 0]
        for i in range(levels):
            gates[i] = 1
        self.skip_gates = gates
        self.skip_levels = levels
        self.use_skip_connections = self.requires_skips()

    def _concat_or_zeros(self, h, skip_feat, expected_ch: int, use_gate: bool):
        if use_gate and (skip_feat is not None):
            return torch.cat([h, skip_feat], dim=1)
        zeros = torch.zeros(
            h.size(0),
            expected_ch,
            h.size(2),
            h.size(3),
            h.size(4),
            device=h.device,
            dtype=h.dtype,
        )
        return torch.cat([h, zeros], dim=1)

    def forward(self, z, skips=None):
        e1 = e2 = e3 = None
        if skips is not None:
            e1, e2, e3 = skips

        h = self.up1(z)
        h = self.rb1(h)

        h = self._concat_or_zeros(
            h,
            e3,
            expected_ch=self.base * 4,
            use_gate=bool(self.skip_gates[0]),
        )

        h = self.up2(h)
        h = self.rb2(h)

        h = self._concat_or_zeros(
            h,
            e2,
            expected_ch=self.base * 2,
            use_gate=bool(self.skip_gates[1]),
        )

        h = self.up3(h)
        h = self.rb3(h)

        h = self._concat_or_zeros(
            h,
            e1,
            expected_ch=self.base,
            use_gate=bool(self.skip_gates[2]),
        )

        h = self.out_block(h)
        logits = self.out(h)
        return logits


class UNet3DVAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, latent_dim=256, skip_levels: int = 3):
        super().__init__()
        if skip_levels < 0 or skip_levels > 3:
            raise ValueError("skip_levels must be between 0 and 3")
        self.encoder = Encoder3DUNetVAE(in_ch, base, latent_dim)
        self.decoder = Decoder3DUNetVAE(out_ch, base, latent_dim, skip_levels=skip_levels)
        self.skip_levels = self.decoder.skip_levels
        self.use_skip_connections = self.decoder.use_skip_connections

    def set_skip_levels(self, levels: int):
        self.decoder.set_skip_levels(levels)
        self.skip_levels = self.decoder.skip_levels
        self.use_skip_connections = self.decoder.use_skip_connections

    def forward(self, x):
        need_skips = self.decoder.requires_skips()
        mu, logvar, skips = self.encoder(x, return_skips=need_skips)
        logits = self.decoder(mu, skips if need_skips else None)  # deterministic: use mean
        return logits, mu, logvar


# ---------------------------------------------------------------------------
# Metrics helpers (mostly reused from original evaluate_models.py)
# ---------------------------------------------------------------------------


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


def iou_from_stats(stats):
    return [tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0 for tp, fp, fn in stats]


def dice_from_stats(stats):
    return [(2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 1.0 for tp, fp, fn in stats]


def occupancy_ratio(labels):
    return float((labels != 0).sum()) / labels.size


def compute_nonair_metrics(pred, gt):
    pred_nonair = pred != 0
    gt_nonair = gt != 0

    tp = np.logical_and(pred_nonair, gt_nonair).sum()
    fp = np.logical_and(pred_nonair, ~gt_nonair).sum()
    fn = np.logical_and(~pred_nonair, gt_nonair).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(precision), float(recall), float(f1)


def aabb_spans(mask):
    if not mask.any():
        return (0, 0, 0), (0, 0)
    coords = np.array(np.where(mask))
    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1)
    Lz, Ly, Lx = zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1
    return (Lz, Ly, Lx), (Ly / Lx if Lx > 0 else 0, Lz / Lx if Lx > 0 else 0)


# ---------------------------------------------------------------------------
# Model parameter summaries
# ---------------------------------------------------------------------------


def extract_weight_stats(model, include_layer_details=False):
    weight_stats = {}
    all_weights = []
    layer_stats = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        arr = param.detach().cpu().numpy()
        all_weights.append(arr.reshape(-1))
        if include_layer_details:
            layer_stats[name] = {
                "shape": list(param.shape),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "numel": int(param.numel()),
            }

    if all_weights:
        concat = np.concatenate(all_weights)
        weight_stats = {
            "weight_mean": float(concat.mean()),
            "weight_std": float(concat.std()),
            "weight_min": float(concat.min()),
            "weight_max": float(concat.max()),
            "weight_abs_mean": float(np.abs(concat).mean()),
            "weight_abs_max": float(np.abs(concat).max()),
        }
        if include_layer_details:
            weight_stats["layer_details"] = layer_stats

    return weight_stats


def parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(val)


def resolve_skip_levels(args, default: int = 3) -> int:
    if not isinstance(args, dict):
        return default

    candidates = [
        args.get("skip_levels_effective"),
        args.get("skip_levels"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            pass

    if "no_skip_connections" in args and parse_bool(args["no_skip_connections"]):
        return 0
    if "use_skip_connections" in args:
        return 3 if parse_bool(args["use_skip_connections"]) else 0

    return default


def extract_model_params(model, ckpt, include_weight_stats=False):
    params = {}

    if isinstance(ckpt, dict) and "args" in ckpt:
        args = ckpt["args"]
        params["base"] = args.get("base")
        params["latent_dim"] = args.get("latent_dim")
        params["lr"] = args.get("lr")
        params["batch_size"] = args.get("batch_size")
        params["epochs"] = args.get("epochs")
        params["kl_beta"] = args.get("kl_beta")
        params["checkpoint_epoch"] = ckpt.get("epoch")
        params["best_val_loss"] = ckpt.get("best_val")
        params["exp_name"] = args.get("exp_name")
        skip_levels = resolve_skip_levels(args)
        if skip_levels is not None:
            params["skip_levels"] = skip_levels
        use_skip = args.get("use_skip_connections")
        if use_skip is not None:
            params["use_skip_connections"] = parse_bool(use_skip)
        elif skip_levels is not None:
            params["use_skip_connections"] = skip_levels > 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params["total_params"] = total_params
    params["trainable_params"] = trainable_params
    params["non_trainable_params"] = total_params - trainable_params
    params["model_size_mb"] = total_params * 4 / (1024 * 1024)  # float32

    if include_weight_stats:
        params["weight_stats"] = extract_weight_stats(model, include_layer_details=False)

    return params


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_model(
    model_path,
    test_files,
    device,
    progress=None,
    task=None,
    use_amp=True,
    include_weight_stats=False,
):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = {}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        args = ckpt.get("args", {}) or {}
        base = int(args.get("base", 64))
        latent = int(args.get("latent_dim", 256))
    else:
        state_dict = ckpt
        base = 64
        latent = 256
    skip_levels = resolve_skip_levels(args)
    if skip_levels is None:
        skip_levels = 3

    model = UNet3DVAE(in_ch=3, out_ch=3, base=base, latent_dim=latent, skip_levels=skip_levels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if not use_amp:
        model = model.float()

    model_params = extract_model_params(model, ckpt, include_weight_stats=include_weight_stats)

    agg = np.zeros((3, 3), dtype=np.int64)
    occ_pred, occ_gt = [], []
    aryx_err, azx_err = [], []
    precision_list, recall_list, f1_list = [], [], []

    for idx, npz_path in enumerate(test_files):
        with np.load(npz_path, allow_pickle=False) as data:
            key = "arr_0" if "arr_0" in data else list(data.files)[0]
            arr = data[key]

        gt = arr.astype(np.uint8)
        x = torch.from_numpy(one_hot(gt)).unsqueeze(0).to(device)
        if not use_amp:
            x = x.float()

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits, _, _ = model(x)
        else:
            logits, _, _ = model(x)

        pred = torch.argmax(F.softmax(logits[0], dim=0), dim=0).cpu().numpy().astype(np.uint8)

        agg += np.array(compute_confusion_stats(pred, gt))
        occ_pred.append(occupancy_ratio(pred))
        occ_gt.append(occupancy_ratio(gt))

        _, (aryx_p, azx_p) = aabb_spans(pred != 0)
        _, (aryx_g, azx_g) = aabb_spans(gt != 0)
        if aryx_g > 0:
            aryx_err.append(abs(aryx_p - aryx_g))
        if azx_g > 0:
            azx_err.append(abs(azx_p - azx_g))

        precision, recall, f1 = compute_nonair_metrics(pred, gt)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    iou = iou_from_stats(agg)
    dice = dice_from_stats(agg)

    return {
        "model": str(model_path),
        "iou": [float(v) for v in iou],
        "dice": [float(v) for v in dice],
        "iou_mean": float(np.mean(iou)),
        "dice_mean": float(np.mean(dice)),
        "occ_pred": float(np.mean(occ_pred)),
        "occ_gt": float(np.mean(occ_gt)),
        "occ_err": float(abs(np.mean(occ_pred) - np.mean(occ_gt))),
        "aryx_err": float(np.mean(aryx_err)) if aryx_err else 0.0,
        "azx_err": float(np.mean(azx_err)) if azx_err else 0.0,
        "precision_nonair": float(np.mean(precision_list)),
        "recall_nonair": float(np.mean(recall_list)),
        "f1_nonair": float(np.mean(f1_list)),
        "model_params": model_params,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    console = Console()
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net VAE checkpoints.")
    parser.add_argument("--data_dir", required=True, help="Directory with .npz test volumes.")
    parser.add_argument("--models_dir", required=True, help="Directory containing checkpoints (.pt/.pth).")
    parser.add_argument("--out_dir", required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP for evaluation.")
    parser.add_argument(
        "--skip_patterns",
        nargs="*",
        default=["last"],
        help="Filename patterns to skip (case-insensitive). Default: ['last']",
    )
    parser.add_argument(
        "--include_weight_stats",
        action="store_true",
        help="Include overall weight statistics for each model.",
    )
    args = parser.parse_args()

    test_dir = Path(args.data_dir).expanduser().resolve()
    if not test_dir.exists() or not test_dir.is_dir():
        console.print(
            Panel.fit(
                f"[bold red]錯誤：測試資料目錄不存在或不是目錄[/bold red]\n"
                f"路徑: [yellow]{test_dir}[/yellow]",
                border_style="red",
            )
        )
        raise SystemExit(1)

    test_files = sorted(test_dir.glob("*.npz"))
    if not test_files:
        console.print(
            Panel.fit(
                f"[bold red]錯誤：測試目錄中沒有 .npz 檔案[/bold red]\n"
                f"目錄: [yellow]{test_dir}[/yellow]",
                border_style="red",
            )
        )
        raise SystemExit(1)

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

    model_files = sorted(list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth")))
    total_models_before_filter = len(model_files)

    skip_patterns = args.skip_patterns if args.skip_patterns else []
    if skip_patterns:
        skip_lower = [p.lower() for p in skip_patterns]
        model_files = [
            f for f in model_files if not any(pattern in f.name.lower() for pattern in skip_lower)
        ]
    filtered_count = total_models_before_filter - len(model_files)

    if not model_files:
        skip_info = f"（已過濾 {filtered_count} 個包含 {skip_patterns} 的檔案）" if skip_patterns else ""
        console.print(
            Panel.fit(
                f"[bold red]錯誤：模型目錄中沒有找到模型檔案[/bold red]\n"
                f"目錄: [yellow]{models_dir}[/yellow]\n"
                f"搜尋模式: *.pt, *.pth（遞迴搜尋）\n"
                f"{skip_info}",
                border_style="red",
            )
        )
        raise SystemExit(1)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_txt = out_dir / f"eval_unet_{ts}.txt"
    out_csv = out_dir / f"eval_unet_{ts}.csv"

    amp_status = "禁用" if not use_amp else "啟用"
    filter_info = (
        f"（已過濾 {filtered_count} 個包含 {skip_patterns} 的檔案）" if filtered_count > 0 else ""
    )
    console.print(
        Panel.fit(
            f"[bold cyan]3D U-Net VAE 模型評估[/bold cyan]\n"
            f"測試檔案數: [yellow]{len(test_files)}[/yellow]\n"
            f"模型數量: [yellow]{len(model_files)}[/yellow] {filter_info}\n"
            f"跳過模式: [yellow]{skip_patterns if skip_patterns else '無'}[/yellow]\n"
            f"裝置: [yellow]{device}[/yellow]\n"
            f"CUDA AMP: [yellow]{amp_status}[/yellow]\n"
            f"輸出目錄: [cyan]{out_dir}[/cyan]",
            border_style="cyan",
        )
    )

    with out_txt.open("w", encoding="utf-8") as ft, out_csv.open("w", newline="", encoding="utf-8") as fc:
        writer = csv.writer(fc)
        writer.writerow(
            [
                "model",
                "iou_mean",
                "dice_mean",
                "iou_air",
                "iou_log",
                "iou_leaves",
                "dice_air",
                "dice_log",
                "dice_leaves",
                "occ_pred",
                "occ_gt",
                "occ_err",
                "aryx_err",
                "azx_err",
                "precision_nonair",
                "recall_nonair",
                "f1_nonair",
                "base",
                "latent_dim",
                "skip_levels",
                "use_skip_connections",
                "total_params",
                "trainable_params",
                "model_size_mb",
                "lr",
                "batch_size",
                "epochs",
                "kl_beta",
                "checkpoint_epoch",
                "best_val_loss",
            ]
        )

        run_header = {
            "data_test_dir": str(test_dir),
            "models_root": str(models_dir),
            "num_test_files": len(test_files),
            "num_models": len(model_files),
            "num_models_filtered": filtered_count,
            "skip_patterns": skip_patterns,
            "device": str(device),
            "use_amp": use_amp,
            "timestamp": ts,
        }
        console.print(json.dumps({"run_header": run_header}, ensure_ascii=False))
        ft.write(json.dumps({"run_header": run_header}, ensure_ascii=False) + "\n")

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
                "[cyan]評估模型中...", total=len(model_files)
            )

            for idx, model_path in enumerate(model_files, start=1):
                model_name = model_path.name
                progress.update(
                    models_task, description=f"[cyan]評估模型 {idx}/{len(model_files)}: {model_name}"
                )

                try:
                    test_task = progress.add_task(
                        "[dim]處理測試檔案...", total=len(test_files), visible=len(test_files) > 10
                    )

                    result = eval_model(
                        model_path,
                        test_files,
                        device,
                        progress=progress,
                        task=test_task,
                        use_amp=use_amp,
                        include_weight_stats=args.include_weight_stats,
                    )

                    progress.remove_task(test_task)

                    iou = result["iou"]
                    dice = result["dice"]
                    params = result.get("model_params", {})

                    console.print(
                        f"[green]✓[/green] [{idx}/{len(model_files)}] {model_name}\n"
                        f"  [dim]IoU={result['iou_mean']:.4f} Dice={result['dice_mean']:.4f} "
                        f"OccErr={result['occ_err']:.4f}[/dim]\n"
                        f"  [dim]Precision={result['precision_nonair']:.4f} "
                        f"Recall={result['recall_nonair']:.4f} "
                        f"F1={result['f1_nonair']:.4f} (non-air)[/dim]"
                    )

                    if params:
                        lines = []
                        arch = []
                        if params.get("base") is not None:
                            arch.append(f"base={params['base']}")
                        if params.get("latent_dim") is not None:
                            arch.append(f"latent={params['latent_dim']}")
                        if params.get("skip_levels") is not None:
                            arch.append(f"skip_levels={params['skip_levels']}")
                        if arch:
                            lines.append(f"  [cyan]架構:[/cyan] {', '.join(arch)}")
                        if params.get("use_skip_connections") is not None:
                            skip_status = "啟用" if params["use_skip_connections"] else "停用"
                            lines.append(f"  [cyan]Skip 連接:[/cyan] {skip_status}")

                        if params.get("total_params") is not None:
                            total = params["total_params"]
                            trainable = params.get("trainable_params", total)
                            size_mb = params.get("model_size_mb", 0.0)
                            lines.append(
                                f"  [cyan]參數:[/cyan] 總計={total:,} (可訓練={trainable:,}), 大小={size_mb:.2f} MB"
                            )

                        training = []
                        if params.get("lr") is not None:
                            training.append(f"lr={params['lr']}")
                        if params.get("batch_size") is not None:
                            training.append(f"bs={params['batch_size']}")
                        if params.get("epochs") is not None:
                            training.append(f"epochs={params['epochs']}")
                        if params.get("kl_beta") is not None:
                            training.append(f"kl_beta={params['kl_beta']}")
                        if training:
                            lines.append(f"  [cyan]訓練:[/cyan] {', '.join(training)}")

                        if params.get("checkpoint_epoch") is not None:
                            lines.append(f"  [cyan]檢查點:[/cyan] epoch={params['checkpoint_epoch']}")
                        if params.get("best_val_loss") is not None:
                            lines.append(
                                f"  [cyan]最佳驗證損失:[/cyan] {params['best_val_loss']:.4f}"
                            )

                        weight_stats = params.get("weight_stats", {})
                        if weight_stats:
                            ws = []
                            for key in ["weight_mean", "weight_std", "weight_min", "weight_max"]:
                                if key in weight_stats:
                                    ws.append(f"{key.split('_')[1]}={weight_stats[key]:.6f}")
                            if ws:
                                lines.append(f"  [cyan]權重統計:[/cyan] {', '.join(ws)}")

                        if lines:
                            console.print("\n".join(lines))

                    ft.write(json.dumps(result, ensure_ascii=False) + "\n")

                    row = [
                        result["model"],
                        result["iou_mean"],
                        result["dice_mean"],
                        iou[0],
                        iou[1],
                        iou[2],
                        dice[0],
                        dice[1],
                        dice[2],
                        result["occ_pred"],
                        result["occ_gt"],
                        result["occ_err"],
                        result["aryx_err"],
                        result["azx_err"],
                        result["precision_nonair"],
                        result["recall_nonair"],
                        result["f1_nonair"],
                        params.get("base", ""),
                        params.get("latent_dim", ""),
                        params.get("skip_levels", ""),
                        params.get("use_skip_connections", ""),
                        params.get("total_params", ""),
                        params.get("trainable_params", ""),
                        params.get("model_size_mb", ""),
                        params.get("lr", ""),
                        params.get("batch_size", ""),
                        params.get("epochs", ""),
                        params.get("kl_beta", ""),
                        params.get("checkpoint_epoch", ""),
                        params.get("best_val_loss", ""),
                    ]
                    writer.writerow(row)

                except Exception as exc:
                    err = str(exc)
                    console.print(f"[red]✗[/red] [{idx}/{len(model_files)}] {model_name}: [red]{err}[/red]")
                    ft.write(json.dumps({"model": str(model_path), "error": err}, ensure_ascii=False) + "\n")
                    writer.writerow([str(model_path)] + [""] * 26 + [err])

                progress.update(models_task, advance=1)

    console.print(
        Panel.fit(
            f"[bold green]✓ 評估完成！[/bold green]\n"
            f"結果已保存至:\n"
            f"  [cyan]TXT: {out_txt}[/cyan]\n"
            f"  [cyan]CSV: {out_csv}[/cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

