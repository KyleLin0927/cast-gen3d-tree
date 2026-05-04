# 🌳 CAST: Connectivity-Aware Sampling for Topology in 3D Voxel Diffusion

CAST 是一個以深度學習生成 Minecraft 樹木 Voxel 結構的研究型專案。嘗試以 VAE／VQ-VAE、Transformer Prior、Diffusion 為不同實驗階段重心，支援從資料前處理、模型訓練，到取樣生成與結果聚合分析的完整流程。目前主力是 diffusion 加上 Guidance 量化對於連通性之影響。

## 環境需求

- Python `3.11.8`
- macOS／Linux。
- 建議具備 NVIDIA GPU。

## 安裝與環境設定

### 1）建立虛擬環境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2）安裝依賴

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3）驗證安裝

```bash
python --version
python -c "import torch; print('torch:', torch.__version__)"
```
