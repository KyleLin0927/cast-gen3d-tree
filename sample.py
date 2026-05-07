#!/usr/bin/env python3
"""
One-click inference script for 16^3 voxel diffusion checkpoints.

Default behavior:
- Generate samples from diffusion checkpoint
- Save per-sample NPZ and 3-view projection PNGs
- Save sample_labels.csv and sample_labels_summary.csv
- Also create sample_summary.csv as a convenience alias

Optional:
- Enable scorer guidance by providing --scorer_checkpoint
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from training.script_260202.generate_16_voxel_diffusion import (
    BetaSchedule,
    Console,
    fmt_secs,
    generate_samples,
    load_model,
    load_scorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click sample generation from diffusion checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Diffusion checkpoint (.pt)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./inference_outputs/run_001",
        help="Output directory for generated samples",
    )
    parser.add_argument("--n_samples", type=int, default=32, help="Number of generated samples")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--n_steps", type=int, default=1000, help="DDPM sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu (auto if omitted)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument(
        "--scorer_checkpoint",
        type=str,
        default=None,
        help="Optional scorer checkpoint for guidance sampling",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="Guidance strength (used only if --scorer_checkpoint is provided)",
    )
    parser.add_argument(
        "--guidance_lambda_ratio",
        type=float,
        default=1.0,
        help="Energy ratio for guidance (used only with --scorer_checkpoint)",
    )
    parser.add_argument("--guidance_t_start", type=int, default=900, help="Guidance start timestep")
    parser.add_argument("--guidance_t_end", type=int, default=400, help="Guidance end timestep")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    console = Console()

    setup_seed(int(args.seed))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = out_dir.name or "sample"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    console.print(f"[cyan]Device:[/cyan] {device} | [cyan]AMP:[/cyan] {use_amp}")

    model, checkpoint = load_model(str(Path(args.checkpoint).expanduser().resolve()), device)
    ck_args = checkpoint.get("args", {})
    T = int(ck_args.get("T", 1000))
    beta_schedule = ck_args.get("beta_schedule", "linear")
    beta_start = float(ck_args.get("beta_start", 1e-4))
    beta_end = float(ck_args.get("beta_end", 0.02))
    betas = BetaSchedule(
        T=T, schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end
    ).to(device)

    scorer_model: Optional[nn.Module] = None
    if args.scorer_checkpoint:
        scorer_model = load_scorer(str(Path(args.scorer_checkpoint).expanduser().resolve()), device)
        console.print(
            f"[green]Guidance ON[/green]: scale={args.guidance_scale}, "
            f"lambda_ratio={args.guidance_lambda_ratio}, "
            f"t=[{args.guidance_t_start}, {args.guidance_t_end}]"
        )
    else:
        console.print("[yellow]Guidance OFF[/yellow]: no --scorer_checkpoint")

    elapsed, _rows = generate_samples(
        model=model,
        betas=betas,
        device=device,
        n_samples=int(args.n_samples),
        out_dir=str(out_dir),
        batch_size=int(args.batch_size),
        n_steps=int(args.n_steps) if args.n_steps is not None else None,
        use_amp=use_amp,
        save_projections=True,
        save_npz=True,
        log_mask_threshold=None,
        filename_prefix=filename_prefix,
        sample_verbose=False,
        run_seed=int(args.seed),
        scorer_model=scorer_model,
        guidance_scale=float(args.guidance_scale),
        t_start=int(args.guidance_t_start),
        t_end=int(args.guidance_t_end),
        guidance_lambda_ratio=float(args.guidance_lambda_ratio),
        console=console,
    )

    summary_src = out_dir / "sample_labels_summary.csv"
    summary_alias = out_dir / "sample_summary.csv"
    if summary_src.exists():
        shutil.copy2(summary_src, summary_alias)
        console.print(f"[green]✓[/green] Summary alias: {summary_alias}")

    console.print(f"[bold green]Done[/bold green] in {fmt_secs(elapsed)}")
    console.print(f"[cyan]Output:[/cyan] {out_dir}")


if __name__ == "__main__":
    main()
