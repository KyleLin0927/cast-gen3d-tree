import os
import io
import argparse
import importlib.util
from glob import glob
import zipfile
import tempfile
import shutil

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
# - --model_script 必填：需提供定義 UNet3DVAE 或 VQVAE3D 的訓練腳本路徑
# - --vae_ckpt 可為單一 .pt 檔或包含 best_*.pt/last_*.pt 的資料夾
# - --out_dir 可為資料夾或 .zip 檔案路徑（若為 .zip，會將結果打包成壓縮檔）
#
# 參數說明 (Arguments)
# - --data_dir       輸入資料根目錄；需含 train/、val/、test/ 三個子資料夾，內放 .npz 標籤，
#                    shape=(32,32,32)、值為 0/1/2；或為 .zip 檔案（內部結構為 train/val/test/*.npz）
# - --out_dir        輸出路徑：
#                    - 若為資料夾：會建立 train/、val/、test/，每筆輸出 latent .npy
#                    - 若以 .zip 結尾：會將整個目錄結構打包成 zip 檔案
# - --vae_ckpt       VAE checkpoint（檔或資料夾）；資料夾時優先取 best_*.pt，其次 last_*.pt
# - --model_script   定義 UNet3DVAE 或 VQVAE3D 的 Python 腳本路徑（必填），用於重建模型結構
# - --cpu            強制使用 CPU（預設自動選擇 CUDA → MPS → CPU）
# - --aug_rot_x/y/z  啟用沿 X/Y/Z 軸的 90° 旋轉增強（僅 train split）
# - --aug_flip_x/y/z 啟用沿 X/Y/Z 軸的翻轉增強（僅 train split）
#
# 輸出格式
# - UNet3DVAE: 使用 mu 作為 latent，將 [latent_dim, 8, 8, 8] 攤平成 [512, latent_dim]，儲存為 .npy 檔。
# - VQVAE3D: 使用 codebook indices，將 [8, 8, 8] 攤平成 [512]，儲存為 .npy 檔。
# - 若 --out_dir 以 .zip 結尾，會將 train/val/test 目錄結構打包成 zip 檔案。
# - conversion_metadata.csv 會放在 zip 檔案外（與 zip 同目錄）或輸出資料夾內。
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
# 支援資料夾與 zip 兩種來源
# ---------------------------
class LabelFolder(torch.utils.data.Dataset):
    """
    從實體資料夾載入 .npz 檔。
    """

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


class ZipLabelFolder(torch.utils.data.Dataset):
    """
    從 zip 檔內的 train/val/test 子路徑載入 .npz 檔。
    不會事先解壓整個壓縮檔，只在需要時讀取檔案內容。
    """

    def __init__(self, zip_file: zipfile.ZipFile, split_name: str, base_prefix: str = ""):
        """
        zip_file: 已開啟的 ZipFile 物件（會在 main 中只開一次）
        split_name: "train" / "val" / "test"
        base_prefix: 若 zip 解壓後會多一層資料夾，例如 dataset_root/train/...，
                     則傳入 "dataset_root/"；若沒有額外層級則為空字串。
        """
        self.zip_file = zip_file
        self.split_name = split_name

        prefix = f"{base_prefix}{split_name}/"
        self.paths = [
            name
            for name in zip_file.namelist()
            if name.startswith(prefix) and name.endswith(".npz")
        ]
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path_in_zip = self.paths[idx]
        # 直接從 zip 讀 bytes，再用 BytesIO 包裝給 np.load，避免非 seekable 問題
        with self.zip_file.open(path_in_zip, "r") as f:
            data_bytes = f.read()
        with np.load(io.BytesIO(data_bytes), allow_pickle=False) as data:
            key = "arr_0" if "arr_0" in data else list(data.files)[0]
            labels = data[key]

        labels = torch.from_numpy(labels.astype(np.int64))  # [32,32,32]
        if labels.shape != (32, 32, 32):
            raise ValueError(
                f"Expected (32,32,32), got {labels.shape} from {path_in_zip}"
            )
        onehot = F.one_hot(labels.long(), num_classes=3).permute(3, 0, 1, 2).float()
        # 輸出檔名仍然只用 basename，行為與資料夾模式一致
        return onehot, os.path.basename(path_in_zip)


def _detect_zip_base_prefix(zf: zipfile.ZipFile) -> str:
    """
    嘗試自動偵測 zip 內部是否有一層共同根目錄，例如:
        dataset_root/train/...
        dataset_root/val/...
        dataset_root/test/...

    若存在上述結構，回傳 "dataset_root/" 作為 base_prefix；
    若 train/val/test 直接在根目錄下，則回傳空字串 ""。
    """
    names = zf.namelist()

    # 先檢查 train/val/test 是否直接在根目錄下
    has_root_train = any(name.startswith("train/") for name in names)
    has_root_val = any(name.startswith("val/") for name in names)
    has_root_test = any(name.startswith("test/") for name in names)
    if has_root_train and has_root_val and has_root_test:
        return ""

    # 否則嘗試找出共同的前綴資料夾
    candidate_prefixes = set()
    for name in names:
        idx = name.find("train/")
        if idx > 0:
            candidate_prefixes.add(name[:idx])

    for prefix in sorted(candidate_prefixes):
        has_train = any(n.startswith(f"{prefix}train/") for n in names)
        has_val = any(n.startswith(f"{prefix}val/") for n in names)
        has_test = any(n.startswith(f"{prefix}test/") for n in names)
        if has_train and has_val and has_test:
            return prefix

    # 若找不到，就當作結構不符合預期
    raise FileNotFoundError(
        "Could not find train/val/test structure inside zip file. "
        "Expected either train/val/test at root, or <root_dir>/train|val|test."
    )


# ---------------------------
# Save latent to disk
# ---------------------------
def save_latent(z, out_path):
    """
    z: tensor [512, latent_dim] for VAE or [512] for VQ-VAE
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

def _import_model_class(script_path: str):
    """
    Dynamically import UNet3DVAE or VQVAE3D from the training script.
    Returns (model_class, model_type) where model_type is 'vae' or 'vqvae'.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Model script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("train_model", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Try VQVAE3D first, then UNet3DVAE
    if hasattr(module, "VQVAE3D"):
        return module.VQVAE3D, "vqvae"
    elif hasattr(module, "UNet3DVAE"):
        return module.UNet3DVAE, "vae"
    else:
        raise AttributeError(
            "Neither VQVAE3D nor UNet3DVAE found in the provided training script. "
            "Please ensure the script defines one of these classes."
        )

def _build_model_from_checkpoint(ckpt_path: str, device: torch.device, model_script: str | None):
    """
    Load training checkpoint dict, reconstruct UNet3DVAE or VQVAE3D and load state_dict.
    Returns (model, model_type) where model_type is 'vae' or 'vqvae'.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})

    # Determine model script to import model class
    if model_script is None:
        raise ValueError("--model_script is required to locate model definition.")
    ModelClass, model_type = _import_model_class(model_script)

    # Default/fallbacks if absent
    base = int(args.get("base", 64))
    latent_dim = int(args.get("latent_dim", 256))

    if model_type == "vqvae":
        # VQ-VAE specific parameters
        codebook_size = int(args.get("codebook_size", 512))
        commitment_cost = float(args.get("commitment_cost", 0.25))
        
        model = ModelClass(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
        ).to(device)
    else:
        # VAE specific parameters
        skip_levels = int(args.get("skip_levels", 0 if args.get("no_skip_connections", False) else args.get("skip_levels", 0)))
        
        model = ModelClass(
            in_ch=3,
            out_ch=3,
            base=base,
            latent_dim=latent_dim,
            skip_levels=skip_levels,
        ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_type

# ---------------------------
# Main conversion process
# ---------------------------
def process_split(split_name, dataset, out_dir, model, model_type, device, cfg):
    console = Console()
    console.print(f"\n[bold cyan]Processing split:[/bold cyan] {split_name}")

    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    n_inputs = len(dataset)
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
                # Encode: VAE uses mu, VQ-VAE uses codebook indices
                # -----------------------------------
                with torch.no_grad():
                    if model_type == "vqvae":
                        # VQ-VAE: use encode_to_indices to get discrete codebook indices
                        indices = model.encode_to_indices(aug)  # [B, 8, 8, 8]
                        z = indices.squeeze(0)  # [8, 8, 8]
                        # flatten to [512]
                        z = z.reshape(-1)  # [512]
                    else:
                        # VAE: use mu as latent (NO sample)
                        mu, logvar, _ = model.encoder(aug, return_skips=False)
                        z = mu.squeeze(0)  # [latent_dim, 8, 8, 8]
                        # flatten to [512, latent_dim]
                        latent_dim = z.shape[0]
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

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory containing train/val/test subfolders with .npz files, "
        "or a .zip file whose internal structure is train/val/test/*.npz",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--model_script",
        type=str,
        required=True,
        help="Path to a Python file that defines UNet3DVAE or VQVAE3D.",
    )

    parser.add_argument("--aug_rot_x", action="store_true")
    parser.add_argument("--aug_rot_y", action="store_true")
    parser.add_argument("--aug_rot_z", action="store_true")
    parser.add_argument("--aug_flip_x", action="store_true")
    parser.add_argument("--aug_flip_y", action="store_true")
    parser.add_argument("--aug_flip_z", action="store_true")

    cfg = parser.parse_args()

    # -----------------------
    # Validate out_dir (can be directory or .zip file)
    # -----------------------
    console = Console()
    if cfg.out_dir.endswith(".pt"):
        console.print(Panel.fit("[bold red]--out_dir must be a directory or .zip file, not a .pt file path.[/bold red]", border_style="red"))
        raise SystemExit(1)
    
    # Check if output is zip file
    use_output_zip = cfg.out_dir.lower().endswith(".zip")
    temp_dir = None
    actual_out_dir = cfg.out_dir
    
    if use_output_zip:
        # Output is zip file
        output_zip_path = os.path.abspath(cfg.out_dir)
        if os.path.exists(output_zip_path):
            console.print(Panel.fit(f"[bold red]Output zip file already exists:[/bold red]\n[yellow]{output_zip_path}[/yellow]\nRefusing to proceed to avoid overwriting.", border_style="red"))
            raise SystemExit(1)
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="convert_npz_to_latent_")
        actual_out_dir = temp_dir
        console.print(f"[cyan]Output will be written to zip file:[/cyan] {output_zip_path}")
        console.print(f"[dim]Using temporary directory: {temp_dir}[/dim]")
    else:
        # Output is directory
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
    # 檢查 data_dir 型態（資料夾或 zip）
    # -----------------------
    is_zip = os.path.isfile(cfg.data_dir) and cfg.data_dir.lower().endswith(".zip")
    zip_file = None
    zip_base_prefix = ""

    if is_zip:
        # 僅開啟一次 zip，以避免重複 IO 開銷
        if not zipfile.is_zipfile(cfg.data_dir):
            console.print(
                Panel.fit(
                    f"[bold red]--data_dir is not a valid zip file:[/bold red] {cfg.data_dir}",
                    border_style="red",
                )
            )
            raise SystemExit(1)
        zip_file = zipfile.ZipFile(cfg.data_dir, "r")
        try:
            zip_base_prefix = _detect_zip_base_prefix(zip_file)
        except FileNotFoundError as e:
            console.print(
                Panel.fit(
                    f"[bold red]{str(e)}[/bold red]\n[yellow]{cfg.data_dir}[/yellow]",
                    border_style="red",
                )
            )
            zip_file.close()
            raise SystemExit(1)
    else:
        # 確認 train/val/test 目錄存在
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(cfg.data_dir, split)
            if not os.path.isdir(split_dir):
                console.print(
                    Panel.fit(
                        f"[bold red]Data split directory not found:[/bold red] {split_dir}",
                        border_style="red",
                    )
                )
                raise SystemExit(1)

    # -----------------------
    # Load VAE/VQ-VAE
    # -----------------------
    console.print("[bold]Loading model...[/bold]")
    device = _resolve_device(force_cpu=cfg.cpu)
    ckpt_path = _resolve_checkpoint_path(cfg.vae_ckpt)
    model, model_type = _build_model_from_checkpoint(
        ckpt_path, device, cfg.model_script
    )
    console.print(f"[green]✓[/green] Loaded {model_type.upper()} model")

    # -----------------------
    # Create output folders
    # -----------------------
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(actual_out_dir, split), exist_ok=True)

    # -----------------------
    # Process three splits
    # -----------------------
    split_stats = []
    try:
        for split in ["train", "val", "test"]:
            if is_zip:
                dataset = ZipLabelFolder(zip_file, split, base_prefix=zip_base_prefix)
            else:
                dataset = LabelFolder(os.path.join(cfg.data_dir, split))

            stats = process_split(
                split,
                dataset,
                actual_out_dir,
                model,
                model_type,
                device,
                cfg,
            )
            split_stats.append(stats)
    finally:
        if zip_file is not None:
            zip_file.close()

    console.print("\n[bold green]All latent data generated successfully![/bold green]")
    
    # -----------------------
    # Package to zip if needed
    # -----------------------
    if use_output_zip:
        console.print(f"[cyan]Packaging results into zip file...[/cyan]")
        output_zip_path = os.path.abspath(cfg.out_dir)
        try:
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Walk through temp directory and add all files
                for root, dirs, files in os.walk(actual_out_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate relative path from temp_dir
                        rel_path = os.path.relpath(file_path, actual_out_dir)
                        # Use forward slashes for zip (ZIP standard)
                        zip_rel_path = rel_path.replace(os.sep, '/')
                        zip_file.write(file_path, zip_rel_path)
            console.print(f"[green]✓[/green] Created zip file: [cyan]{output_zip_path}[/cyan]")
        except Exception as e:
            console.print(Panel.fit(f"[bold red]Failed to create zip file:[/bold red] {e}", border_style="red"))
            raise
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    console.print(f"[dim]Cleaned up temporary directory[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to clean up temporary directory {temp_dir}: {e}[/yellow]")
    
    # -----------------------
    # Write conversion metadata CSV (always outside zip)
    # -----------------------
    # If output is zip, save CSV next to the zip file
    if use_output_zip:
        output_zip_path = os.path.abspath(cfg.out_dir)
        csv_dir = os.path.dirname(output_zip_path)
        csv_name = os.path.splitext(os.path.basename(output_zip_path))[0] + "_conversion_metadata.csv"
        out_csv = os.path.join(csv_dir, csv_name)
    else:
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
        latent_dim = ckpt_args.get("latent_dim", "unknown")
        if model_type == "vqvae":
            codebook_size = ckpt_args.get("codebook_size", "unknown")
            commitment_cost = ckpt_args.get("commitment_cost", "unknown")
            skip_levels = "N/A (VQ-VAE)"
        else:
            skip_levels = ckpt_args.get("skip_levels", 0 if ckpt_args.get("no_skip_connections", False) else ckpt_args.get("skip_levels", "unknown"))
            codebook_size = "N/A (VAE)"
            commitment_cost = "N/A (VAE)"
        model_source = os.path.abspath(cfg.model_script)
    except Exception:
        base = "unknown"
        latent_dim = "unknown"
        skip_levels = "unknown"
        codebook_size = "unknown"
        commitment_cost = "unknown"
        model_source = os.path.abspath(cfg.model_script)

    total_inputs = sum(s["inputs"] for s in split_stats)
    total_outputs = sum(s["outputs"] for s in split_stats)

    rows = []
    # Global metadata
    rows.append(("timestamp", started_at))
    rows.append(("data_dir", os.path.abspath(cfg.data_dir)))
    rows.append(("out_dir", os.path.abspath(cfg.out_dir)))
    rows.append(("output_format", "zip" if use_output_zip else "directory"))
    rows.append(("vae_ckpt_resolved", os.path.abspath(ckpt_path)))
    rows.append(("model_script", model_source))
    rows.append(("model_type", model_type.upper()))
    rows.append(("device", device_str))
    rows.append(("base", base))
    rows.append(("latent_dim", latent_dim))
    if model_type == "vqvae":
        rows.append(("codebook_size", codebook_size))
        rows.append(("commitment_cost", commitment_cost))
        rows.append(("output_format", "[512] (discrete codebook indices)"))
    else:
        rows.append(("skip_levels", skip_levels))
        rows.append(("output_format", f"[512, {latent_dim}] (continuous latent vectors)"))
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
