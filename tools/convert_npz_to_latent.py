import os
import argparse
import importlib.util
from glob import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# -----------------------------------------------------------------------------
# 使用方式 (Usage)
# -----------------------------------------------------------------------------
# 注意：
# - --model_script 必填：需提供定義 UNet3DVAE 的訓練腳本路徑
# - --vae_ckpt 可為單一 .pt 檔或包含 best_*.pt/last_*.pt 的資料夾
# - --out_dir 必須為資料夾（非 .pt 路徑）
#
# 參數說明 (Arguments)
# - --data_dir       輸入資料根目錄；需含 train/、val/、test/ 三個子資料夾，內放 .npz 標籤，
#                    shape=(32,32,32)、值為 0/1/2
# - --out_dir        輸出資料夾；會建立 train/、val/、test/，每筆輸出 latent .npy（[512, latent_dim]）
# - --vae_ckpt       VAE checkpoint（檔或資料夾）；資料夾時優先取 best_*.pt，其次 last_*.pt
# - --model_script   定義 UNet3DVAE 的 Python 腳本路徑（必填），用於重建模型結構
# - --cpu            強制使用 CPU（預設自動選擇 CUDA → MPS → CPU）
# - --aug_rot_x/y/z  啟用沿 X/Y/Z 軸的 90° 旋轉增強（僅 train split）
# - --aug_flip_x/y/z 啟用沿 X/Y/Z 軸的翻轉增強（僅 train split）
#
# 輸出格式
# - 使用 mu 作為 latent，將 [latent_dim, 8, 8, 8] 攤平成 [512, latent_dim]，儲存為 .npy 檔。
# ----------------------------------------------------------------------------- 

# ---------------------------
# Augmentation utilities
# ---------------------------
def apply_augmentations(voxel, cfg):
    aug_list = [voxel]

    # Rotations (90 degrees) in x/y/z directions
    if cfg.aug_rot_x:
        aug_list.append(torch.rot90(voxel, k=1, dims=(2, 3)))
        aug_list.append(torch.rot90(voxel, k=2, dims=(2, 3)))
        aug_list.append(torch.rot90(voxel, k=3, dims=(2, 3)))

    if cfg.aug_rot_y:
        aug_list.append(torch.rot90(voxel, k=1, dims=(1, 3)))
        aug_list.append(torch.rot90(voxel, k=2, dims=(1, 3)))
        aug_list.append(torch.rot90(voxel, k=3, dims=(1, 3)))

    if cfg.aug_rot_z:
        aug_list.append(torch.rot90(voxel, k=1, dims=(1, 2)))
        aug_list.append(torch.rot90(voxel, k=2, dims=(1, 2)))
        aug_list.append(torch.rot90(voxel, k=3, dims=(1, 2)))

    # Flips
    if cfg.aug_flip_x:
        aug_list.append(torch.flip(voxel, dims=[1]))
    if cfg.aug_flip_y:
        aug_list.append(torch.flip(voxel, dims=[2]))
    if cfg.aug_flip_z:
        aug_list.append(torch.flip(voxel, dims=[3]))

    # Ensure no duplicates (optional)
    unique_aug = []
    seen = set()
    for a in aug_list:
        key = a.numpy().tobytes()
        if key not in seen:
            seen.add(key)
            unique_aug.append(a)

    return unique_aug


# ---------------------------
# Dataset Loader (.npz labels -> onehot[3,32,32,32])
# ---------------------------
class LabelFolder(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = []
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Data split directory not found: {folder}")
        for f in os.listdir(folder):
            if f.endswith(".npz"):
                self.paths.append(os.path.join(folder, f))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as data:
            key = "arr_0" if "arr_0" in data else list(data.files)[0]
            labels = data[key]
        labels = torch.from_numpy(labels.astype(np.int64))  # [32,32,32]
        if labels.shape != (32, 32, 32):
            raise ValueError(f"Expected (32,32,32), got {labels.shape} from {path}")
        # one-hot -> [3,32,32,32], float32
        onehot = F.one_hot(labels.long(), num_classes=3).permute(3, 0, 1, 2).float()
        return onehot, os.path.basename(path)


# ---------------------------
# Save latent to disk
# ---------------------------
def save_latent(z, out_path):
    """
    z: tensor [512, latent_dim]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, z.cpu().numpy())

def _apply_rot_flip_labels(labels, cfg):
    """
    labels: torch.LongTensor [32,32,32] (Z,Y,X)
    Returns list of augmented label tensors (each [32,32,32]).
    """
    from itertools import product

    # Build Cartesian product of rotations and flips
    # Rotations: k in {0,1,2,3} if enabled else {0}
    kx_list = list(range(4)) if getattr(cfg, "aug_rot_x", False) else [0]
    ky_list = list(range(4)) if getattr(cfg, "aug_rot_y", False) else [0]
    kz_list = list(range(4)) if getattr(cfg, "aug_rot_z", False) else [0]
    # Flips: f in {0,1} if enabled else {0}
    fx_list = [0, 1] if getattr(cfg, "aug_flip_x", False) else [0]
    fy_list = [0, 1] if getattr(cfg, "aug_flip_y", False) else [0]
    fz_list = [0, 1] if getattr(cfg, "aug_flip_z", False) else [0]

    outputs = []
    for kx, ky, kz, fx, fy, fz in product(kx_list, ky_list, kz_list, fx_list, fy_list, fz_list):
        x = labels
        # Apply rotations (same convention as training dataset)
        if kx % 4:
            x = torch.rot90(x, k=int(kx) % 4, dims=(0, 1))  # X-axis: rotate Y<->Z
        if ky % 4:
            x = torch.rot90(x, k=int(ky) % 4, dims=(0, 2))  # Y-axis: rotate X<->Z
        if kz % 4:
            x = torch.rot90(x, k=int(kz) % 4, dims=(1, 2))  # Z-axis: rotate X<->Y
        # Apply flips (match train dataset: fz on dim 0, fy on dim 1, fx on dim 2)
        if fz:
            x = torch.flip(x, dims=[0])
        if fy:
            x = torch.flip(x, dims=[1])
        if fx:
            x = torch.flip(x, dims=[2])
        outputs.append((x, (int(kx), int(ky), int(kz), int(fx), int(fy), int(fz))))

    # Deduplicate to avoid redundant transforms
    unique, seen = [], set()
    for t, combo in outputs:
        key = t.numpy().tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((t, combo))
    return unique

def _labels_to_onehot(labels: torch.Tensor) -> torch.Tensor:
    # labels [32,32,32] -> onehot [3,32,32,32]
    return F.one_hot(labels.long(), num_classes=3).permute(3, 0, 1, 2).float()

def _resolve_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _resolve_checkpoint_path(ckpt_arg: str) -> str:
    if os.path.isdir(ckpt_arg):
        # Prefer best_*.pt then last_*.pt
        best = sorted(glob(os.path.join(ckpt_arg, "best_*.pt")))
        last = sorted(glob(os.path.join(ckpt_arg, "last_*.pt")))
        if best:
            return best[0]
        if last:
            return last[0]
        raise FileNotFoundError(f"No checkpoint files found in directory: {ckpt_arg}")
    if not os.path.exists(ckpt_arg):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_arg}")
    return ckpt_arg

def _import_unet3dvae(script_path: str):
    """
    Dynamically import UNet3DVAE from the training script.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Model script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("train_3d_unet_vae", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "UNet3DVAE"):
        raise AttributeError("UNet3DVAE not found in the provided training script.")
    return module.UNet3DVAE

def _build_model_from_checkpoint(ckpt_path: str, device: torch.device, model_script: str | None):
    """
    Load training checkpoint dict, reconstruct UNet3DVAE and load state_dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})

    # Default/fallbacks if absent
    base = int(args.get("base", 64))
    latent_dim = int(args.get("latent_dim", 256))
    skip_levels = int(args.get("skip_levels", 0 if args.get("no_skip_connections", False) else args.get("skip_levels", 0)))

    # Determine model script to import UNet3DVAE
    if model_script is None:
        raise ValueError("--model_script is required to locate UNet3DVAE definition.")
    UNet3DVAE = _import_unet3dvae(model_script)

    model = UNet3DVAE(
        in_ch=3,
        out_ch=3,
        base=base,
        latent_dim=latent_dim,
        skip_levels=skip_levels,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, latent_dim

# ---------------------------
# Main conversion process
# ---------------------------
def process_split(split_name, in_dir, out_dir, model, latent_dim, device, cfg):
    console = Console()
    console.print(f"\n[bold cyan]Processing split:[/bold cyan] {split_name}")

    ds = LabelFolder(in_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    n_inputs = len(ds)
    n_outputs = 0
    used_combos = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[green]Encoding latents ({split_name})", total=len(dl))

        for onehot, fname in dl:
            # onehot: [1,3,32,32,32]
            onehot = onehot.to(device)  # [1,3,D,H,W]

            # ----------------------
            # Augment only on TRAIN
            # ----------------------
            if split_name == "train":
                # reconstruct labels from onehot (argmax), apply label-space augs, convert back to onehot
                labels = onehot[0].argmax(dim=0)  # [32,32,32]
                aug_labels = _apply_rot_flip_labels(labels.cpu(), cfg)  # list of (labels_t, (kx,ky,kz,fx,fy,fz))
                augmented = [
                    (
                        _labels_to_onehot(lab_t).unsqueeze(0).to(device),
                        combo,
                    )
                    for (lab_t, combo) in aug_labels
                ]
            else:
                # no augmentation; mark combo as all zeros for identifiable naming
                augmented = [(onehot, (0, 0, 0, 0, 0, 0))]

            for idx, (aug, combo) in enumerate(augmented):

                # -----------------------------------
                # Encode: use mu as latent (NO sample)
                # -----------------------------------
                with torch.no_grad():
                    # Directly call encoder; do not use skip connections; returns [B,C,8,8,8]
                    mu, logvar, _ = model.encoder(aug, return_skips=False)
                    z = mu.squeeze(0)                  # [latent_dim, 8, 8, 8]

                # flatten to [512, latent_dim]
                z = z.permute(1, 2, 3, 0).reshape(-1, latent_dim)  # [512, latent_dim]

                # output path
                base = fname[0].replace(".npz","")
                kx, ky, kz, fx, fy, fz = combo
                suffix = f"aug_kx{kx}_ky{ky}_kz{kz}_fx{fx}_fy{fy}_fz{fz}"
                save_path = os.path.join(out_dir, split_name, f"{base}_{suffix}.npy")

                save_latent(z, save_path)

                n_outputs += 1
                used_combos.add(combo)

            progress.update(task, advance=1)
    console.print(f"[green]✓[/green] Finished split {split_name}")
    return {
        "split": split_name,
        "inputs": n_inputs,
        "outputs": n_outputs,
        "unique_combo_count": len(used_combos),
    }


# ---------------------------
# Entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--model_script",
        type=str,
        required=True,
        help="Path to a Python file that defines UNet3DVAE.",
    )

    parser.add_argument("--aug_rot_x", action="store_true")
    parser.add_argument("--aug_rot_y", action="store_true")
    parser.add_argument("--aug_rot_z", action="store_true")
    parser.add_argument("--aug_flip_x", action="store_true")
    parser.add_argument("--aug_flip_y", action="store_true")
    parser.add_argument("--aug_flip_z", action="store_true")

    cfg = parser.parse_args()

    # -----------------------
    # Validate out_dir (must be a directory path)
    # -----------------------
    console = Console()
    if cfg.out_dir.endswith(".pt"):
        console.print(Panel.fit("[bold red]--out_dir must be a directory, not a .pt file path.[/bold red]", border_style="red"))
        raise SystemExit(1)
    # Stop if target directory exists and is non-empty
    if os.path.isdir(cfg.out_dir):
        try:
            has_entries = any(os.scandir(cfg.out_dir))
        except PermissionError:
            console.print(Panel.fit(f"[bold red]Cannot access out_dir:[/bold red] {cfg.out_dir}", border_style="red"))
            raise SystemExit(1)
        if has_entries:
            console.print(Panel.fit(f"[bold red]Refusing to proceed:[/bold red] out_dir is not empty:\n[yellow]{cfg.out_dir}[/yellow]", border_style="red"))
            raise SystemExit(1)

    # -----------------------
    # Load VAE
    # -----------------------
    console.print("[bold]Loading VAE...[/bold]")
    device = _resolve_device(force_cpu=cfg.cpu)
    ckpt_path = _resolve_checkpoint_path(cfg.vae_ckpt)
    model, latent_dim = _build_model_from_checkpoint(ckpt_path, device, cfg.model_script)

    # -----------------------
    # Create output folders
    # -----------------------
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(cfg.out_dir, split), exist_ok=True)

    # -----------------------
    # Process three splits
    # -----------------------
    split_stats = []
    for split in ["train", "val", "test"]:
        stats = process_split(
            split,
            os.path.join(cfg.data_dir, split),
            cfg.out_dir,
            model,
            latent_dim,
            device,
            cfg
        )
        split_stats.append(stats)

    console.print("\n[bold green]All latent data generated successfully![/bold green]")
    # -----------------------
    # Write conversion metadata CSV
    # -----------------------
    out_csv = os.path.join(cfg.out_dir, "conversion_metadata.csv")
    from datetime import datetime
    import csv
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_str = str(device)
    # Try extract model args from checkpoint again
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        base = ckpt_args.get("base", "unknown")
        skip_levels = ckpt_args.get("skip_levels", 0 if ckpt_args.get("no_skip_connections", False) else ckpt_args.get("skip_levels", "unknown"))
        model_source = os.path.abspath(cfg.model_script)
    except Exception:
        base = "unknown"
        skip_levels = "unknown"
        model_source = os.path.abspath(cfg.model_script)

    total_inputs = sum(s["inputs"] for s in split_stats)
    total_outputs = sum(s["outputs"] for s in split_stats)

    rows = []
    # Global metadata
    rows.append(("timestamp", started_at))
    rows.append(("data_dir", os.path.abspath(cfg.data_dir)))
    rows.append(("out_dir", os.path.abspath(cfg.out_dir)))
    rows.append(("vae_ckpt_resolved", os.path.abspath(ckpt_path)))
    rows.append(("model_script", model_source))
    rows.append(("device", device_str))
    rows.append(("latent_dim", latent_dim))
    rows.append(("base", base))
    rows.append(("skip_levels", skip_levels))
    # Aug flags
    rows.append(("aug_rot_x", str(bool(cfg.aug_rot_x)).upper()))
    rows.append(("aug_rot_y", str(bool(cfg.aug_rot_y)).upper()))
    rows.append(("aug_rot_z", str(bool(cfg.aug_rot_z)).upper()))
    rows.append(("aug_flip_x", str(bool(cfg.aug_flip_x)).upper()))
    rows.append(("aug_flip_y", str(bool(cfg.aug_flip_y)).upper()))
    rows.append(("aug_flip_z", str(bool(cfg.aug_flip_z)).upper()))
    # Totals
    rows.append(("total_inputs", total_inputs))
    rows.append(("total_outputs", total_outputs))

    # Per-split stats
    for s in split_stats:
        prefix = s["split"]
        rows.append((f"{prefix}_inputs", s["inputs"]))
        rows.append((f"{prefix}_outputs", s["outputs"]))
        rows.append((f"{prefix}_unique_combo_count", s["unique_combo_count"]))

    try:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            for k, v in rows:
                writer.writerow([k, v])
        console.print(Panel.fit(f"[green]✓[/green] Saved conversion metadata to [cyan]{out_csv}[/cyan]", border_style="green"))
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Failed to write conversion metadata:[/bold red] {e}", border_style="red"))


if __name__ == "__main__":
    main()
