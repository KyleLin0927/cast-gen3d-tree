# 🌳 CAST: Connectivity-Aware Sampling for Topology in 3D Voxel Diffusion

> Training-free sampling-time guidance for topology preservation in 3D voxel diffusion.
> Improves connectivity success rate from **6.5% → 30%** on Minecraft tree volumes — no retraining of the base denoiser, no architectural change.

<img width="900" height="419" alt="baseline_vs_specific_window" src="https://github.com/user-attachments/assets/08e3cf4f-21d6-4f83-9a20-8afbcba1a18a" />


*Left: baseline DDPM samples (broken trunks, floating leaves). Right: same model with connectivity-aware sampling-time guidance.*

---

## Key Results

Connectivity success = fraction of generated samples whose trunk forms a **single connected component** *and* passes a geometric **sanity check** (so a thick single pillar that trivially satisfies connectivity but no longer looks like a tree does **not** count). 16³ voxel grid; each guided cell is the best group passing the sanity check.

| Connectivity success ↑ | Full-time guidance | **Windowed** (t = 800–300) |
|---|---|---|
| **Universal Guidance** (x̂₀, clean-domain) | 15% | **39%** |
| **Classifier Guidance** (xₜ, noise-aware) | 36% | 30% |

Baseline DDPM (no guidance): **6.5%**.

**The takeaway is the asymmetry, not a single number.** Windowing helps the clean-domain scorer a lot (UG: 15 → 39) but slightly *hurts* the noise-aware one (CG: 36 → 30). x̂₀ is only reliable in the middle of denoising (its early-step Tweedie estimate diverges), so restricting it to a mid window is exactly right; xₜ is noise-aware and can act across all timesteps, but pays for it with more shortcut solutions. **The value and form of intervention depend on which domain the scorer reads.**

> 📌 *Coming from the v2 report?* v2 reported the **30%** windowed (xₜ) result. v3 adds the UG × CG 2×2 above; the clean-domain scorer in the mid-window reaches **39%**. Both numbers are in the table — nothing was retracted, the picture was completed.

📄 [**Full PoC report**](https://drinkai.notion.site/Connectivity-Aware-Sampling-for-Topology-CAST-report-v2-36daa15702ab801f8291d71c2bc043c5) · 🤗 [**Pretrained checkpoints**](https://huggingface.co/jenkai-lin/cast-tree-voxel-diffusion)

---

## Three Main Findings

**1. Connectivity failure is a sampling-dynamics problem, not a model-capacity problem.** Successful and failed samples diverge between t = 800 and t = 600 (T = 1000), and once trunk connectivity breaks during this middle phase, baseline denoising almost never repairs it in later steps. Intervening over the window t = 800–300 raises connectivity success up to **39%** (and to 30% for the noise-aware scorer), showing that structural failure in 3D can be fixed by acting on the sampling *trajectory* rather than by scaling the model.

**2. Guidance applied in a mid-sampling window passes both connectivity and naturalness sanity checks.** Intervening too early pushes the overall structure toward oversimplified, compressed forms; intervening too late doesn't give the diffusion prior enough time to pull samples back to the natural distribution. The intermediate window is the most stable operating point — and *how much* the window helps depends on the scorer's operating domain (see Key Results).

**3. A lightweight, on-demand scorer is a practical alternative to conditioning-based control when data is scarce.** CAST's scorer has only **346K parameters (1.3% of the 26M base denoiser) and trains in 145 seconds** on a single GPU, with no human annotation — it is auto-labeled from the base model's own samples. Instead of relying on one general scorer (e.g., CLIP in DreamFusion) to carry every control objective, control can be split into multiple narrow, domain-specific scorers, each lightweight and aligned directly with its target metric. Multi-objective control then becomes a composition problem.

<img width="1708" alt="minecraft-like-tree" src="https://github.com/user-attachments/assets/8cc758ed-622c-48e1-a922-5e39a583155d" />

---

## Method (TL;DR)

1. **Base model** — a 3D voxel DDPM (16³, T = 1000, cosine schedule) trained from scratch on 1,286 Minecraft-style tree volumes. No conditioning.
2. **Connectivity scorer** — a 346K-parameter 3D CNN trained with hard-negative mining on auto-labeled `positive / floating / disconnected / fragmented` structures (labels come from the base model's own outputs via a BFS connectivity check). Training completes in ~145 seconds on a single GPU.
3. **Sampling-time guidance** — at each denoising step inside the window t = 800–300, nudge the standard DDPM update with the scorer gradient. Two routes are compared:
   - **xₜ (noise-aware, classifier-guidance style)** — the scorer reads the noisy state directly.
   - **x̂₀ (clean-domain, Universal-Guidance style)** — the scorer reads the Tweedie-estimated clean voxel.

   No retraining and no architectural change to the base denoiser in either route.

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

### Generate with guidance

CAST supports two sampling-time guidance routes. Pair each route with the matching scorer checkpoint (`--train_on xt` for CG, `--train_on x0` for UG). Pretrained checkpoints are on the [Hugging Face page](https://huggingface.co/jenkai-lin/cast-tree-voxel-diffusion).

**Classifier Guidance** (xₜ, noise-aware — windowed **30%** in Key Results):

```bash
python sample.py \
  --checkpoint /path/to/diffusion_best.pt \
  --scorer_checkpoint /path/to/scorer_xt_best.pt \
  --guidance_mode xt \
  --guidance_t_start 800 \
  --guidance_t_end 300
```

**Universal Guidance** (x̂₀, clean-domain — windowed **39%** in Key Results):

```bash
python sample.py \
  --checkpoint /path/to/diffusion_best.pt \
  --scorer_checkpoint /path/to/scorer_x0_best.pt \
  --guidance_mode ug \
  --ug_inject eps \
  --guidance_t_start 800 \
  --guidance_t_end 300
```

Common flags (all have defaults):
- `--n_samples` — number of samples to generate (default `32`)
- `--out_dir` — output directory (default `./inference_outputs/run_001`)
- `--n_steps` — sampling steps (default `1000`)
- `--batch_size` — batch size (default `8`)
- `--guidance_mode` — `xt` (classifier / noise-aware) or `ug` (Universal Guidance / clean x̂₀)
- `--ug_inject` — UG gradient injection: `eps` (default) or `x` (only with `--guidance_mode ug`)
- `--guidance_scale`, `--guidance_lambda_ratio` — guidance strength (tune per route; see report)
- `--guidance_t_start`, `--guidance_t_end` — intervention window (Key Results use **t = 800–300**; script default is 900–400)

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
