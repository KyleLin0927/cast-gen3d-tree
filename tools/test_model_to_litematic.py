#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_model_to_litematic.py

直接生成 litematic 檔案方便比較 raw vs rec。
將測試資料輸入模型進行推理，生成預測的 npz 檔案，並轉換為 Minecraft litematic 檔案。

功能：
1. 從指定資料夾讀取測試資料（.npz 檔案）
2. 使用指定的模型進行推理
3. 將預測結果保存為 .npz 檔案（可選）
4. 將預測結果轉換為 .litematic 檔案

轉換方式（體素 ID 到方塊）：
    - 0 = 空氣 (air)
    - 1 = 橡木木頭 (oak_wood)
    - 2 = 橡木樹葉 (oak_leaves)

用法：
  python test_model_to_litematic.py \
    --test_data_dir /path/to/test/data \
    --model_path /path/to/model.pt \
    --output_dir /path/to/output \
    [--device cuda] \
    [--save_npz] \
    [--no_amp]
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from litemapy import Schematic, Region, BlockState

console = Console()

# --- 模型定義（與 evaluate_models.py 相同）---
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
        super().__init__()
        self.encoder = Encoder3D(in_ch, base, latent_dim)
        self.decoder = Decoder3D(out_ch, base, latent_dim)

    def forward(self, x):
        mu, _ = self.encoder(x)
        return self.decoder(mu)  # deterministic reconstruction


# --- 工具函數 ---
def one_hot(labels, num_classes=3):
    """將標籤轉換為 one-hot 編碼"""
    oh = np.eye(num_classes, dtype=np.float32)[labels]
    return np.moveaxis(oh, -1, 0)

def load_model(model_path, device, use_amp=True):
    """載入模型"""
    # 使用 weights_only=False 以支援包含 numpy 物件的檢查點（PyTorch 2.6+ 相容性）
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
        args = ckpt.get('args', {})
        base, latent = int(args.get('base', 64)), int(args.get('latent_dim', 256))
    else:
        state_dict, base, latent = ckpt, 64, 256
    
    model = VAE3D(3, 3, base, latent).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 如果禁用 AMP，確保模型使用 float32 精度
    if not use_amp:
        model = model.float()
    
    return model

def load_npz_array(npz_path: str) -> np.ndarray:
    """載入 npz 檔案中的陣列"""
    f = np.load(npz_path, allow_pickle=True)
    key = "data" if "data" in f.files else ("arr_0" if "arr_0" in f.files else f.files[0])
    arr = f[key]
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(npz_path)}: expected 3D array, got shape {arr.shape}")
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int8)
    return arr

def array_to_schematic(vox: np.ndarray, name: str) -> Schematic:
    """將體素陣列轉換為 litematic Schematic（參考 convert_npz_to_litematic.py）
    
    轉換方式：
        - 0 = 空氣 (air)
        - 1 = 橡木木頭 (oak_wood)
        - 2 = 橡木樹葉 (oak_leaves)
    """
    sx, sy, sz = vox.shape  # X, Y, Z
    # Region 只接受 6 個數字參數 (x, y, z, width, height, length)
    reg = Region(0, 0, 0, sx, sy, sz)  # 原點放置，大小即陣列尺寸

    # 三類方塊對應：0=空氣, 1=橡木木頭, 2=橡木樹葉
    AIR = BlockState("minecraft:air")
    OAK_WOOD = BlockState("minecraft:oak_wood")
    # 設定 persistent=true 以防止樹葉腐敗，distance 設為 1 以符合法規定屬性
    OAK_LEAVES = BlockState(
        "minecraft:oak_leaves",
        persistent="true",
        distance="1",
    )
    
    ID_TO_BLOCK = {
        0: AIR,        # 空氣
        1: OAK_WOOD,   # 橡木木頭
        2: OAK_LEAVES, # 橡木樹葉
    }

    # 寫方塊（使用 setblock）
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                bid = int(vox[x, y, z])
                block = ID_TO_BLOCK.get(bid, AIR)
                reg.setblock(x, y, z, block)

    # 用 as_schematic 直接封裝成 Schematic，並給名字
    schem = reg.as_schematic(name=name, author="inference_to_litematic", description="Generated from model inference")
    return schem

@torch.no_grad()
def inference_single(model, npz_path, device, use_amp=True):
    """對單個 npz 檔案進行推理，返回預測的體素陣列"""
    # 載入測試資料
    with np.load(npz_path, allow_pickle=False) as d:
        arr = d['arr_0'] if 'arr_0' in d else d[list(d.files)[0]]
    gt = arr.astype(np.uint8)
    
    # 轉換為 one-hot 編碼
    x = torch.from_numpy(one_hot(gt)).unsqueeze(0).to(device)
    if not use_amp:
        x = x.float()
    
    # 模型推理
    def _extract_logits(model_output):
        logits = model_output
        while isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits

    if use_amp and device.type == 'cuda':
        with torch.cuda.amp.autocast():
            logits = _extract_logits(model(x))
    else:
        logits = _extract_logits(model(x))

    if logits.dim() == 5:
        if logits.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 during inference, got batch dimension {logits.shape[0]}"
            )
        logits = logits[0]
    elif logits.dim() != 4:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    pred = torch.argmax(logits, dim=0).to(dtype=torch.uint8).cpu().numpy()
    return pred

def main():
    ap = argparse.ArgumentParser(
        description="將測試資料輸入模型進行推理，生成預測的 npz 檔案，並轉換為 Minecraft litematic 檔案"
    )
    ap.add_argument("--test_data_dir", required=True, help="測試資料資料夾（包含 .npz 檔案）")
    ap.add_argument("--model_path", required=True, help="模型檔案路徑（.pt 或 .pth）")
    ap.add_argument("--output_dir", required=True, help="輸出資料夾（將保存 .litematic 檔案）")
    ap.add_argument("--device", default="cuda", help="計算裝置：cuda 或 cpu（預設：cuda）")
    ap.add_argument("--save_npz", action="store_true", help="是否同時保存預測的 .npz 檔案")
    ap.add_argument("--no_amp", action="store_true", help="禁用 CUDA AMP（使用完整 float32 精度）")
    args = ap.parse_args()

    # 解析路徑
    test_data_dir = Path(args.test_data_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # 驗證輸入路徑
    if not test_data_dir.is_dir():
        console.print(f"[red]錯誤：測試資料資料夾不存在：{test_data_dir}[/red]")
        return
    
    if not model_path.is_file():
        console.print(f"[red]錯誤：模型檔案不存在：{model_path}[/red]")
        return

    # 收集所有 .npz 檔案
    test_files = sorted(test_data_dir.glob("*.npz"))
    if not test_files:
        console.print(f"[red]錯誤：在 {test_data_dir} 中找不到任何 .npz 檔案[/red]")
        return

    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_npz:
        npz_output_dir = output_dir / "npz"
        npz_output_dir.mkdir(parents=True, exist_ok=True)

    # 設定裝置
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    # 顯示啟動資訊
    console.print(f"[bold cyan]模型推理與轉換工具[/bold cyan]")
    console.print(f"測試資料資料夾：{test_data_dir}")
    console.print(f"模型檔案：{model_path}")
    console.print(f"輸出資料夾：{output_dir}")
    console.print(f"測試檔案數量：{len(test_files)}")
    console.print(f"計算裝置：{device}")
    console.print(f"CUDA AMP：{'啟用' if use_amp else '禁用'}")
    console.print(f"保存 .npz：{'是' if args.save_npz else '否'}")
    console.print()

    # 載入模型
    console.print(f"[cyan]載入模型中...[/cyan]")
    try:
        model = load_model(str(model_path), device, use_amp)
        console.print(f"[green]✓ 模型載入成功[/green]")
    except Exception as e:
        console.print(f"[red]錯誤：模型載入失敗：{e}[/red]")
        return

    # 處理所有測試檔案
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
        task = progress.add_task(
            "[cyan]處理測試檔案中...",
            total=len(test_files)
        )
        
        success_count = 0
        fail_count = 0
        
        for idx, test_file in enumerate(test_files, 1):
            # 更新進度條描述，顯示當前處理的檔案名
            file_name = test_file.name
            if len(file_name) > 40:
                file_name = file_name[:37] + "..."
            progress.update(
                task,
                description=f"[cyan]處理中 [{idx}/{len(test_files)}]: {file_name}"
            )
            
            try:
                # 推理
                pred = inference_single(model, str(test_file), device, use_amp)
                
                # 保存 .npz 檔案（如果需要）
                if args.save_npz:
                    npz_output_path = npz_output_dir / test_file.name
                    np.savez_compressed(str(npz_output_path), data=pred)
                
                # 轉換為 litematic
                base_name = test_file.stem
                schem = array_to_schematic(pred, name=base_name)
                litematic_path = output_dir / f"{base_name}.litematic"
                schem.save(str(litematic_path))
                
                success_count += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠ 處理失敗 {test_file.name}：{e}[/yellow]")
                fail_count += 1
            finally:
                progress.advance(task)

    console.print()
    console.print(f"[bold green]✅ 全部處理完成！[/bold green]")
    console.print(f"成功處理：{success_count} 個檔案")
    if fail_count > 0:
        console.print(f"[yellow]處理失敗：{fail_count} 個檔案[/yellow]")
    console.print(f"輸出資料夾：{output_dir}")
    if args.save_npz:
        console.print(f"NPZ 輸出資料夾：{npz_output_dir}")

if __name__ == "__main__":
    main()

