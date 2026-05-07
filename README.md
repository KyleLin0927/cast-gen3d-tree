# 🌳 CAST: Connectivity-Aware Sampling for Topology in 3D Voxel Diffusion

> Sampling-time guidance for topology preservation in 3D voxel diffusion.
> Improves connectivity success rate from **6.5% → 46%** on Minecraft tree volumes — without retraining the base denoiser.

<img width="900" alt="cast_comparison_11" src="https://github.com/user-attachments/assets/48f813df-9b32-42e5-9e2b-1beb6dfe93fe" />

*Left: baseline DDPM samples (broken trunks, floating leaves). Right: same model with connectivity-aware sampling-time guidance.*

---

## Key Results

Evaluated on 1,000 generated samples (16³ voxel grid):

| Method | Connectivity Success ↑ | Floating-Tree Failures ↓ |
|---|---|---|
| Baseline DDPM | 6.5% | 20.0% |
| **CAST (ours)** | **46.0%** | **0.3%** |

Operating point selected via 5-metric sanity check (log_size, AABB spans, BBO) keeping all metrics within ±1σ of the ground-truth distribution.

📄 [**Full PoC report**](https://www.notion.so/Connectivity-Aware-Sampling-for-Topology-CAST-PoC-32aaa15702ab8010b4ded7ec8110a79d) · 🤗 [**Pretrained checkpoints**](https://huggingface.co/jenkai-lin/cast-tree-voxel-diffusion)

---

## Method (TL;DR)

1. **Base model**: 3D voxel DDPM (16³, T=1000, cosine schedule) trained from scratch on 1,500 Minecraft-style tree volumes
2. **Connectivity scorer**: a separate 3D CNN trained with hard-negative mining on auto-labeled positive / floating / disconnected / fragmented structures
3. **Sampling-time guidance**: at each denoising step, add `∇ C(x_t)` to the standard DDPM update — no retraining or architectural change to the base denoiser

The key empirical finding: **connectivity failures form mid-sampling (t ≈ 800–600) and rarely self-recover**. Guidance applied only within an intermediate window (t = 700–200) outperforms either always-on or late-stage-only intervention.

<img width="1708" alt="minecraft-like-tree" src="https://github.com/user-attachments/assets/8cc758ed-622c-48e1-a922-5e39a583155d" />

---

## Quick Start

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tested on Python 3.11.8, macOS / Linux, NVIDIA GPU recommended.

### Generate samples (baseline, no guidance)

```bash
python sample.py --checkpoint /path/to/diffusion_best.pt
```

### Generate with CAST guidance

```bash
python sample.py \
  --checkpoint /path/to/diffusion_best.pt \
  --scorer_checkpoint /path/to/scorer_best.pt \
  --guidance_scale 2.0
```

Common flags (all have defaults):
- `--n_samples` — number of samples to generate (default `32`)
- `--out_dir` — output directory (default `./inference_outputs/run_001`)
- `--n_steps` — sampling steps (default `1000`)
- `--batch_size` — batch size (default `8`)
- `--guidance_t_start`, `--guidance_t_end` — intervention window (defaults reflect best operating point)

You can find the checkpint in Hugging Face page.

### Output structure

In `--out_dir`:
- `sample_labels.csv` — per-sample metrics
- `sample_labels_summary.csv`, `sample_summary.csv` — aggregated statistics
- `npz/{positive, neg_float, neg_easy, neg_hard}/` — raw voxel arrays per failure category
- `projections/{positive, neg_float, neg_easy, neg_hard}/` — three-view PNGs per sample

---

## Project History

CAST went through several abandoned directions before settling on diffusion + sampling-time guidance: an early VAE / VQ-VAE attempt that lost spatial fidelity at 16³ resolution, and a Transformer prior over discrete codes that struggled with the long-range dependencies needed for connected trunk structure. The current diffusion-based approach was selected after these dead-ends.

---

## Contact

Kyle Lin (林仁凱) — KyleLin0927@gmail.com · [GitHub](https://github.com/KyleLin0927)

---

## Citation

If you find this work useful for your research, please feel free to reach out for discussion.
