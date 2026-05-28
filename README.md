# 🌳 CAST: Connectivity-Aware Sampling for Topology in 3D Voxel Diffusion

> Training-free sampling-time guidance for topology preservation in 3D voxel diffusion.
> Improves connectivity success rate from **6.5% → 30%** on Minecraft tree volumes — no retraining of the base denoiser, no architectural change.

<img width="900" height="419" alt="baseline_vs_specific_window" src="https://github.com/user-attachments/assets/08e3cf4f-21d6-4f83-9a20-8afbcba1a18a" />


*Left: baseline DDPM samples (broken trunks, floating leaves). Right: same model with connectivity-aware sampling-time guidance.*

---

## Key Results

Evaluated on 1,000 generated samples (16³ voxel grid):

| Method | Connectivity Success ↑ |
|---|---|
| Baseline DDPM | 6.5% |
| **CAST (ours)** | **30.0%** (≈5× baseline) |

Operating point selected via a 5-metric sanity check (log_size, AABB spans, BBO) keeping all metrics within ±1σ of the ground-truth distribution — to filter out shortcut solutions where a single thick pillar trivially satisfies connectivity but no longer looks like a tree.

📄 [**Full PoC report**](https://www.notion.so/Connectivity-Aware-Sampling-for-Topology-CAST-PoC-32aaa15702ab8010b4ded7ec8110a79d) · 🤗 [**Pretrained checkpoints**](https://huggingface.co/jenkai-lin/cast-tree-voxel-diffusion)

---

## Three Main Findings

**1. Connectivity failure is a sampling-dynamics problem, not a model-capacity problem.** Successful and failed samples diverge between t = 800 and t = 600 (T = 1000), and once trunk connectivity breaks during this middle phase, baseline denoising almost never repairs it in later steps. Applying guidance over the window t = 800–300 raises connectivity success from 6.5% to 30%, showing that structural failure in 3D can be fixed by intervening on the sampling trajectory rather than by scaling the model.

**2. Guidance applied in a mid-sampling window passes both connectivity and naturalness sanity checks.** Intervening too early pushes the overall structure toward oversimplified, compressed forms; intervening too late doesn't give the diffusion prior enough time to pull samples back to the natural distribution. The intermediate window is the most stable operating point.

**3. A lightweight, on-demand scorer is a practical alternative to conditioning-based control when data is scarce.** CAST's scorer has only **346K parameters and trains in 145 seconds**, with no human annotation needed (auto-labeled from base-model samples). Instead of relying on a single general scorer (e.g., CLIP in DreamFusion) to carry all control objectives, control can be split into multiple narrow, domain-specific scorers, each lightweight and aligned directly with its target metric. Multi-objective control then becomes a composition problem.

<img width="1708" alt="minecraft-like-tree" src="https://github.com/user-attachments/assets/8cc758ed-622c-48e1-a922-5e39a583155d" />

---

## Method (TL;DR)

1. **Base model**: 3D voxel DDPM (16³, T=1000, cosine schedule) trained from scratch on 1,286 Minecraft-style tree volumes.
2. **Connectivity scorer**: a 346K-parameter 3D CNN trained with hard-negative mining on auto-labeled positive / floating / disconnected / fragmented structures. Training completes in 145 seconds on a single GPU.
3. **Sampling-time guidance**: at each denoising step within the window t = 800–300, add `∇ C(x_t)` to the standard DDPM update — no retraining, no architectural change to the base denoiser.

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
- `--guidance_t_start`, `--guidance_t_end` — intervention window (defaults reflect best operating point: t = 800–300)

Pretrained checkpoints are available on the [Hugging Face page](https://huggingface.co/jenkai-lin/cast-tree-voxel-diffusion).

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
