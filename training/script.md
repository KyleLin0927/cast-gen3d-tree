# script_251030

傳統 3D ConvVAE 訓練腳本，主要特色：
1. 向量化潛在空間：Encoder 輸出 `mu/logvar` 向量，Decoder 從全連接層展開生成 32³ 體素。
2. ResBlock-based UNet-lite 架構：三層下採樣與逐層上採樣，但無 skip connection，重點在輕量化。
3. 支援資料增強：旋轉、翻轉、標籤擾動，以及 preload 至 RAM。
4. 範例輸出：每隔數 epoch 保存重建與先驗 `N(0,I)` 取樣，產出 `.npz` 與投影圖。
5. 中斷保護：Ctrl+C 時自動建立續訓 checkpoint。

# script_251108

3D UNet VAE 訓練腳本 (空間潛在表示)，延伸功能：
1. 空間 latent：Encoder 產生 `[B, latent_dim, 4,4,4]` 的 `mu/logvar`，Decoder 透過 ConvTranspose 恢復解析度。
2. 可調跳接：`--skip_levels` 或 `--no_skip_connections` 決定是否使用 UNet skip，loss 中 KL 以平均方式穩定梯度。
3. 分類權重：自動維護 fp32/fp16/bf16 三種張量，搭配 AMP 避免 dtype mismatch。
4. 取樣機制：除重建外，從後驗 `q(z|x)` 取樣確保與輸入一致；跳接啟用時會同步帶入 encoder skips。
5. 檔案管理：除了 best/last checkpoint，完成後自動刪除 last 以省空間，CSV 紀錄與 samples 結構與 ConvVAE 一致。

## train_3D_UNetVAE_20251108.py

首個 UNet 版本：固定 `--skip_levels` 為整訓期間的常數，可透過 `--no_skip_connections` 改為純 decoder；其餘流程與描述一致。

## train_3D_UNetVAE_20251109.py

增強版 UNet 腳本，新增功能：
1. Skip 調度：`--skip_schedule "epoch:levels,..."`
   - levels 可為 0~3 的浮點數，代表從最深層開始依序啟用 skip (支援半值作為 soft gate)。
   - 每次切換前自動存下 transition checkpoint 及樣本 (重建 + posterior samples) 以便比較。
2. 動態 skip gating：Decoder 維持最大通道數，實際效果由 `skip_gates` 控制，允許訓練途中平滑關閉 skip。
3. 終局檔案：訓練完成時額外輸出 `final_{exp}.pt`，同時保留 best checkpoint、刪除 last。
4. Metadata 擴充：紀錄 skip schedule、最終 gating、transition 資料等，利於後續分析。

> 提示：若需要進一步比較 20251108 與 20251109 的實作細節，可查閱 `Decoder3DUNetVAE` 與 skip 調度相關函式 (`parse_skip_schedule`, `get_skip_levels_for_epoch`)。*** End Patch