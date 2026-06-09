"""
Shared DDPM reverse sampling for 16³ voxel diffusion.

Used by train_unet_diffusion.py, generate_16_voxel_diffusion.py, and eval_diffusion_model.py.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def _predict_x0(x: torch.Tensor, eps_pred: torch.Tensor, t_int: int, betas) -> torch.Tensor:
    alpha_bar_t = betas.alpha_bar[t_int]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
    return (x - sqrt_one_minus_alpha_bar_t * eps_pred) / torch.sqrt(alpha_bar_t)


def _ddpm_posterior_step(
    x: torch.Tensor,
    eps_pred: torch.Tensor,
    t_int: int,
    betas,
    device: torch.device,
    *,
    clamp_x0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given x_t and predicted noise, compute pred_x0 and sample x_{t-1}.

    Returns:
        x_prev: next latent (x_{t-1}, or x_0 when t_int == 0)
        pred_x0: x_0 estimate (clamped to [-1, 1] when clamp_x0 is True)
    """
    pred_x0 = _predict_x0(x, eps_pred, t_int, betas)
    if clamp_x0:
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

    beta_t = betas.beta[t_int]
    alpha_t = betas.alpha[t_int]
    alpha_bar_t = betas.alpha_bar[t_int]

    if t_int > 0:
        alpha_bar_prev = betas.alpha_bar[t_int - 1]
    else:
        alpha_bar_prev = torch.tensor(1.0, device=device)

    coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    posterior_mean = coef1 * pred_x0 + coef2 * x

    if t_int > 0:
        posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        noise = torch.randn_like(x)
        x_prev = posterior_mean + torch.sqrt(posterior_var) * noise
    else:
        x_prev = posterior_mean

    return x_prev, pred_x0


@torch.no_grad()
def sample_voxels(
    model,
    betas,
    shape,
    device,
    n_steps=None,
    use_amp=False,
    track_every=None,
    track_callback=None,
    verbose: bool = True,
):
    """
    Reverse diffusion process: sample voxels from noise.

    Args:
        model: UNet3DDiffusion model
        betas: BetaSchedule instance
        shape: (B, C, H, W, D) where C=3 and H=W=D=N (cube; 16 for Minecraft, 32 for ShapeNet)
        device: torch device
        n_steps: number of sampling steps (default: T, can use fewer for speed)
        use_amp: whether to use mixed precision
        track_every: if not None, track metrics every N steps (calls track_callback)
        track_callback: callback(sample_idx, step_idx, t_int, x_current, x0_hat)
        verbose: if False, skip step-wise debug prints (useful for bulk generation)

    Returns:
        x_0: [B, C, H, W, D] sampled voxels in [-1,1] range
    """
    T = betas.T
    if n_steps is None:
        n_steps = T

    B, C, H, W, D = shape
    x = torch.randn(shape, device=device)

    if n_steps < T:
        timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, device=device)

    for i, t_int_tensor in enumerate(timesteps):
        t_int = t_int_tensor.item() if isinstance(t_int_tensor, torch.Tensor) else int(t_int_tensor)
        t = torch.full((B,), t_int, device=device, dtype=torch.long)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            eps_pred = model(x, t)

        pred_x0 = _predict_x0(x, eps_pred, t_int, betas)

        if verbose and i % 200 == 0:
            alpha_bar_t = betas.alpha_bar[t_int]
            sab = torch.sqrt(alpha_bar_t).item()
            eps_min = eps_pred.min().item()
            eps_max = eps_pred.max().item()
            eps_mean = eps_pred.mean().item()
            eps_std = eps_pred.std().item()
            pred_min = pred_x0.min().item()
            pred_max = pred_x0.max().item()
            pred_mean = pred_x0.mean().item()

            print(f"[Debug] Step {t_int}:")
            print(f"  sqrt(alpha_bar_t)={sab:.6e}")
            print(
                f"  eps_pred (U-Net輸出): range=[{eps_min:.2f}, {eps_max:.2f}], "
                f"mean={eps_mean:.2f}, std={eps_std:.2f}"
            )
            print(f"  pred_x0 (計算後): range=[{pred_min:.2f}, {pred_max:.2f}], mean={pred_mean:.2f}")

            if pred_max > 2.0 or pred_min < -2.0:
                print(
                    f"  [警報] pred_x0 數值飄移偵測！range=[{pred_min:.2f}, {pred_max:.2f}], "
                    f"mean={pred_mean:.2f}"
                )
            if abs(eps_mean) > 10.0 or eps_std > 50.0:
                print(f"  [警報] eps_pred 數值異常！mean={eps_mean:.2f}, std={eps_std:.2f}")

        pred_x0_before_clamp_min = pred_x0.min().item()
        pred_x0_before_clamp_max = pred_x0.max().item()
        pred_x0_after_clamp = pred_x0.clamp(-1.0, 1.0)
        pred_x0_after_clamp_min = pred_x0_after_clamp.min().item()
        pred_x0_after_clamp_max = pred_x0_after_clamp.max().item()

        if verbose and i % 200 == 0:
            was_clamped = (pred_x0_before_clamp_min < -1.0) or (pred_x0_before_clamp_max > 1.0)
            if was_clamped:
                print(
                    f"  [Clamp修正] pred_x0 已鉗制至: "
                    f"range=[{pred_x0_after_clamp_min:.2f}, {pred_x0_after_clamp_max:.2f}]"
                )

        x, pred_x0 = _ddpm_posterior_step(
            x, eps_pred, t_int, betas, device, clamp_x0=True
        )

        if track_every is not None and track_callback is not None and i % track_every == 0:
            for sample_idx in range(B):
                track_callback(sample_idx, i, t_int, x[sample_idx], pred_x0[sample_idx])

    return x


def sample_guided_voxels(
    denoiser_model: nn.Module,
    scorer_model: nn.Module,
    betas,
    shape: Tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 50.0,
    lambda_ratio: float = 10.0,
    t_start: int = 900,
    t_end: int = 400,
    n_steps: Optional[int] = None,
    use_amp: bool = False,
    track_every: Optional[int] = None,
    track_callback: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Guided DDPM sampling where a scorer provides gradient-based guidance.

    Not decorated with @torch.no_grad(): scorer guidance needs grad on x_t.
    Denoiser forward and DDPM posterior update run under torch.no_grad().

    Guidance is applied only for timesteps in [min(t_start,t_end), max(t_start,t_end)] (inclusive).
    """
    T = betas.T
    if n_steps is None:
        n_steps = T

    B, C, H, W, D = shape
    # 解析度無關：接受任意立方體 (B,3,N,N,N)，例如 16³（Minecraft）或 32³（ShapeNet）。
    if C != 3 or not (H == W == D):
        raise ValueError(f"Expected shape=(B,3,N,N,N) cubic with C=3, got {shape}")

    guidance_lo = int(min(t_start, t_end))
    guidance_hi = int(max(t_start, t_end))
    guidance_lo = max(0, guidance_lo)
    guidance_hi = min(T - 1, guidance_hi)

    with torch.no_grad():
        x = torch.randn(shape, device=device)

    if n_steps < T:
        timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, device=device)

    for i, t_int_tensor in enumerate(timesteps):
        t_int = t_int_tensor.item() if isinstance(t_int_tensor, torch.Tensor) else int(t_int_tensor)
        t = torch.full((B,), t_int, device=device, dtype=torch.long)

        if guidance_scale > 0.0 and guidance_lo <= t_int <= guidance_hi:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                pred_break_logits, pred_ratio = scorer_model(x, t)
                energy = pred_break_logits.sum() - lambda_ratio * pred_ratio.sum()
                grad = torch.autograd.grad(energy, x)[0]
                x = x - guidance_scale * grad
                x = x.detach()

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                eps_pred = denoiser_model(x, t)
            x, pred_x0 = _ddpm_posterior_step(x, eps_pred, t_int, betas, device)

        if track_every is not None and track_callback is not None and i % track_every == 0:
            for sample_idx in range(B):
                track_callback(sample_idx, i, t_int, x[sample_idx], pred_x0[sample_idx])

    return x


def sample_ug_guided_voxels(
    denoiser_model: nn.Module,
    scorer_model: nn.Module,
    betas,
    shape: Tuple[int, ...],
    device: torch.device,
    guidance_scale: float = 50.0,
    lambda_ratio: float = 10.0,
    t_start: int = 900,
    t_end: int = 400,
    inject: str = "eps",
    clamp_x0_for_scorer: bool = False,
    n_steps: Optional[int] = None,
    use_amp: bool = False,
    track_every: Optional[int] = None,
    track_callback: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Universal-Guidance-style sampling.

    Difference vs ``sample_guided_voxels`` (Path-A):
      - Path-A feeds the *noisy* x_t to the scorer (noise-aware classifier).
      - Here the scorer is evaluated on the *Tweedie clean estimate* x̂_0, so the
        guidance gradient flows THROUGH the denoiser (its Jacobian). This is the
        core "UG / x̂_0 route".

    The scorer is assumed to be a clean-domain scorer
    (``train_scorer.py --train_on x0``) and is queried at t=0.

    ``inject`` controls how the guidance gradient is injected:
      - "eps": forward universal guidance, eps_hat = eps + s*sqrt(1-abar_t)*grad_{x_t} loss
               (canonical UG; one denoiser forward + one backward per guided step).
      - "x"  : direct latent update, x_t <- x_t - s*grad_{x_t} loss
               (matches Path-A's injection, so the *only* changed variable vs
               Path-A is "scorer sees x_hat_0 instead of x_t" -- cleanest A/B.
               Costs an extra denoiser forward because x_t is changed before
               the posterior step.)
    where loss = break_logits.sum() - lambda_ratio * ratio.sum() evaluated on x_hat_0.

    Not decorated with @torch.no_grad(): guidance needs grad through the denoiser.
    Guidance is applied only for t in [min(t_start,t_end), max(t_start,t_end)].

    NOTE: x_hat_0 fed to the scorer is NOT clamped by default -- clamping zeroes the
    gradient wherever it saturates. Set ``clamp_x0_for_scorer=True`` to override.
    """
    if inject not in ("eps", "x"):
        raise ValueError(f"inject must be 'eps' or 'x', got {inject!r}")

    T = betas.T
    if n_steps is None:
        n_steps = T

    B, C, H, W, D = shape
    # 解析度無關：接受任意立方體 (B,3,N,N,N)，例如 16³（Minecraft）或 32³（ShapeNet）。
    if C != 3 or not (H == W == D):
        raise ValueError(f"Expected shape=(B,3,N,N,N) cubic with C=3, got {shape}")

    guidance_lo = max(0, int(min(t_start, t_end)))
    guidance_hi = min(T - 1, int(max(t_start, t_end)))

    with torch.no_grad():
        x = torch.randn(shape, device=device)

    if n_steps < T:
        timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)
    else:
        timesteps = torch.arange(T - 1, -1, -1, device=device)

    # clean-domain scorer is queried at t=0
    t0 = torch.zeros(B, device=device, dtype=torch.long)

    for i, t_int_tensor in enumerate(timesteps):
        t_int = t_int_tensor.item() if isinstance(t_int_tensor, torch.Tensor) else int(t_int_tensor)
        t = torch.full((B,), t_int, device=device, dtype=torch.long)

        apply_guidance = guidance_scale > 0.0 and guidance_lo <= t_int <= guidance_hi

        if apply_guidance:
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                # denoiser forward MUST carry grad: the guidance gradient flows
                # through the Tweedie estimate back to x_t.
                eps_pred = denoiser_model(x, t)
                x0_hat = _predict_x0(x, eps_pred, t_int, betas)
                if clamp_x0_for_scorer:
                    x0_hat = x0_hat.clamp(-1.0, 1.0)
                pred_break_logits, pred_ratio = scorer_model(x0_hat, t0)
                loss = pred_break_logits.sum() - lambda_ratio * pred_ratio.sum()
                grad = torch.autograd.grad(loss, x)[0]

            with torch.no_grad():
                if inject == "eps":
                    sqrt_one_minus_abar_t = torch.sqrt(1.0 - betas.alpha_bar[t_int])
                    eps_guided = (
                        eps_pred + guidance_scale * sqrt_one_minus_abar_t * grad
                    ).detach()
                    x = x.detach()
                    x, pred_x0 = _ddpm_posterior_step(x, eps_guided, t_int, betas, device)
                else:  # inject == "x"
                    x = (x - guidance_scale * grad).detach()
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        eps_clean = denoiser_model(x, t)
                    x, pred_x0 = _ddpm_posterior_step(x, eps_clean, t_int, betas, device)
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    eps_pred = denoiser_model(x, t)
                x, pred_x0 = _ddpm_posterior_step(x, eps_pred, t_int, betas, device)

        if track_every is not None and track_callback is not None and i % track_every == 0:
            for sample_idx in range(B):
                track_callback(sample_idx, i, t_int, x[sample_idx], pred_x0[sample_idx])

    return x
