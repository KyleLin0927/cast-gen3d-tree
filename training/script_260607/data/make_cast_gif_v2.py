"""
CAST: 3D Voxel Diffusion sampling dynamics — GIF assembly script (v2)
=====================================================================

Two presets supported:
  - baseline_vs_specific_window : Baseline (no guidance)  vs  CAST (windowed)
  - full_vs_specific_window     : Full-time guidance      vs  CAST (windowed)

Input:
    Each side (left/right) is a folder of PNG frames, one per timestep.
    File naming pattern:  dynamics_sample_NNN_step_SSSS_t_TTTT.png

Usage:
    # Preset 1: Baseline vs specific window
    python make_cast_gif_v2.py \
        --preset baseline_vs_specific_window \
        --left-dir  /path/to/baseline_frames \
        --right-dir /path/to/cast_frames \
        --sample 30 \
        --output /path/to/baseline_vs_specific_window.gif

    # Preset 2: Full-time vs specific window
    python make_cast_gif_v2.py \
        --preset full_vs_specific_window \
        --left-dir  /path/to/full_frames \
        --right-dir /path/to/cast_frames \
        --sample 30 \
        --output /path/to/full_vs_specific_window.gif

You can also override individual labels with --left-label / --right-label.
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ---------------------------------------------------------------------------
# Visual constants (rarely changed)
# ---------------------------------------------------------------------------
TARGET_WIDTH      = 900
TOP_TITLE_H       = 64     # was 56, room for larger title
SUBTITLE_H        = 34     # was 30, room for larger subtitle
COLUMN_LABEL_H    = 42     # was 36, room for larger header
PROGRESS_BAR_H    = 80     # was 72, room for larger guidance frame + t label
GUTTER            = 12
PADDING           = 16

FRAME_TOP_CROP    = 95
COMPRESS_INTERNAL_GAPS = True
GAP_KEEP_PX = 8

PLAYBACK_FPS              = 10
HOLD_FINAL_SEC            = 4
FAST_PHASE_T_BOUNDARY     = 2000   # FIX: was 2000 (bug, never triggered)
SLOW_PHASE_T_LO           = 800
SLOW_PHASE_T_HI           = 200

# Colors
BG_COLOR              = (245, 245, 247)
TITLE_COLOR           = (30, 30, 35)
SUBTITLE_COLOR        = (110, 110, 120)
COLUMN_LABEL_BG_BASE  = (235, 235, 240)
COLUMN_LABEL_BG_GUIDE = (220, 235, 220)
COLUMN_LABEL_BG_FULL  = (245, 220, 220)   # red-ish to signal "naive baseline of guidance"
COLUMN_LABEL_FG       = (40, 40, 50)
PROGRESS_TRACK        = (220, 220, 225)
PROGRESS_FILL         = (60, 110, 200)
GUIDANCE_REGION       = (255, 200, 100)
GUIDANCE_REGION_A     = 110
GUIDANCE_ACTIVE_BORDER = (180, 110, 30)   # warm brown — visible "guidance ON" frame
GUIDANCE_ACTIVE_BORDER_W = 5
FAILURE_BOX_COLOR     = (220, 60, 60)
SUCCESS_BOX_COLOR     = (50, 160, 80)
WARNING_BOX_COLOR     = (210, 130, 40)  # for "shortcut solution" final state

GIF_QUANTIZE_COLORS = 64
GIF_OPTIMIZE        = True

FNAME_RE = re.compile(
    r"dynamics_sample_(\d+)_step_(\d+)_t_(\d+)\.png", re.IGNORECASE
)

LABEL_STRIP_H = 38   # was 32, room for larger final-outcome pill text


# ---------------------------------------------------------------------------
# Side configuration: encapsulates everything per-column
# ---------------------------------------------------------------------------
@dataclass
class SideConfig:
    folder: Path
    label: str
    # Guidance window: (t_on, t_off) where t_on > t_off (DDPM convention).
    # None  → no guidance at all (e.g. baseline)
    # "full" → guidance ON for the entire trajectory
    # (a, b) → guidance ON during a >= t >= b
    guidance: Optional[object] = None
    # Header background tint
    header_bg: tuple = COLUMN_LABEL_BG_BASE
    # Final-frame outcome label & color
    final_label: str = ""
    final_color: tuple = FAILURE_BOX_COLOR


@dataclass
class GifConfig:
    left: SideConfig
    right: SideConfig
    sample_number: int
    output: Path
    title: str = "CAST: Connectivity-Aware Sampling for 3D Voxel Diffusion"
    subtitle_suffix: str = ""  # extra text appended to subtitle line


# ---------------------------------------------------------------------------
# Presets (the two GIFs we actually want)
# ---------------------------------------------------------------------------
def make_preset(
    name: str,
    left_dir: Path,
    right_dir: Path,
    sample: int,
    output: Path,
) -> GifConfig:
    if name == "baseline_vs_specific_window":
        left = SideConfig(
            folder=left_dir,
            label="Baseline (no guidance)",
            guidance=None,
            header_bg=COLUMN_LABEL_BG_BASE,
            final_label="Disconnected",
            final_color=FAILURE_BOX_COLOR,
        )
        right = SideConfig(
            folder=right_dir,
            label="+ CAST Guidance  (w=0.5, t=800-300)",
            guidance=(800, 300),
            header_bg=COLUMN_LABEL_BG_GUIDE,
            final_label="Connected & natural",
            final_color=SUCCESS_BOX_COLOR,
        )
        return GifConfig(left=left, right=right, sample_number=sample, output=output)

    if name == "full_vs_specific_window":
        left = SideConfig(
            folder=left_dir,
            label="Full-time guidance  (w=0.05, all t)",
            guidance="full",
            header_bg=COLUMN_LABEL_BG_FULL,
            final_label="Connected but less natural",
            final_color=WARNING_BOX_COLOR,
        )
        right = SideConfig(
            folder=right_dir,
            label="CAST Guidance  (w=0.5, t=800-300)",
            guidance=(800, 300),
            header_bg=COLUMN_LABEL_BG_GUIDE,
            final_label="Connected & natural",
            final_color=SUCCESS_BOX_COLOR,
        )
        return GifConfig(left=left, right=right, sample_number=sample, output=output)

    raise ValueError(f"Unknown preset: {name}")


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------
def discover_frames(folder: Path, sample_num: int) -> list[tuple[int, int, Path]]:
    if not folder.exists():
        raise FileNotFoundError(f"Frame folder not found: {folder}")
    found = []
    for p in folder.iterdir():
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        s_num = int(m.group(1))
        step = int(m.group(2))
        t = int(m.group(3))
        if s_num != sample_num:
            continue
        found.append((step, t, p))
    if not found:
        raise FileNotFoundError(
            f"No PNGs found for sample {sample_num} in {folder}"
        )
    found.sort(key=lambda x: x[0])
    return found


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------
def _compress_internal_gaps(img: Image.Image) -> Image.Image:
    arr = np.asarray(img)
    near_white = (arr >= 240).all(axis=2)
    col_is_gap = near_white.mean(axis=0) > 0.92
    if not col_is_gap.any():
        return img
    keep_mask = np.ones_like(col_is_gap, dtype=bool)
    i = 0
    W = len(col_is_gap)
    while i < W:
        if col_is_gap[i]:
            j = i
            while j < W and col_is_gap[j]:
                j += 1
            run_len = j - i
            is_edge = (i == 0) or (j == W)
            if not is_edge and run_len > GAP_KEEP_PX:
                drop_start = i + GAP_KEEP_PX // 2
                drop_end = j - GAP_KEEP_PX // 2
                keep_mask[drop_start:drop_end] = False
            i = j
        else:
            i += 1
    if keep_mask.all():
        return img
    return Image.fromarray(arr[:, keep_mask, :])


def load_and_crop(path: Path, target_w: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.crop((0, FRAME_TOP_CROP, w, h))
    if COMPRESS_INTERNAL_GAPS:
        img = _compress_internal_gaps(img)
    new_h = int(round(img.height * (target_w / img.width)))
    return img.resize((target_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def measure_text(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw, text, font, color, x_center, y_center):
    tw, th = measure_text(draw, text, font)
    draw.text((x_center - tw / 2, y_center - th / 2), text, font=font, fill=color)


# ---------------------------------------------------------------------------
# Progress bar with per-side guidance markers
# ---------------------------------------------------------------------------
def _guidance_is_active(guidance, t: int) -> bool:
    if guidance is None:
        return False
    if guidance == "full":
        return True
    a, b = guidance  # (t_on, t_off), a > b
    return b <= t <= a


def make_progress_bar(
    canvas_w: int,
    bar_h: int,
    t_value: int,
    t_max: int,
    left_guidance,
    right_guidance,
) -> Image.Image:
    """Progress bar with up to two stacked guidance window markers.

    Labeling logic:
      - If only one side has guidance, the marker is labeled "guidance window".
      - If both sides have guidance, markers are labeled "L" / "R" to disambiguate.
    """
    img = Image.new("RGB", (canvas_w, bar_h), BG_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")
    font_marker_label = load_font(13, bold=True)
    font_label = load_font(17, bold=True)

    bar_y = bar_h // 2 - 4
    bar_x_left = PADDING + 60
    bar_x_right = canvas_w - PADDING - 60
    bar_w = bar_x_right - bar_x_left

    # Track
    draw.rounded_rectangle(
        [bar_x_left, bar_y, bar_x_right, bar_y + 8],
        radius=4, fill=PROGRESS_TRACK,
    )

    sides_with_guidance = sum(g is not None for g in (left_guidance, right_guidance))
    use_LR_tags = sides_with_guidance >= 2

    # The progress bar (track) occupies vertical range [bar_y, bar_y+8].
    # Guidance frames wrap *around* this band so the viewer sees
    # "this portion of the timeline is being guided".

    def draw_guidance_region(guidance, ring_index: int, side_tag: str):
        """
        ring_index=0 → inner frame (sits closest to the track)
        ring_index=1 → outer frame (sits a few px further out)

        When only one side has guidance we always use ring_index=0 so the
        frame hugs the bar tightly.
        """
        if guidance is None:
            return
        if guidance == "full":
            gx1, gx2 = bar_x_left, bar_x_right
        else:
            a, b = guidance
            g_start_frac = (t_max - a) / t_max
            g_end_frac   = (t_max - b) / t_max
            gx1 = bar_x_left + int(bar_w * g_start_frac)
            gx2 = bar_x_left + int(bar_w * g_end_frac)

        active = _guidance_is_active(guidance, t_value)

        # Inner ring: tight around the 8-px track. Outer ring: a few px further out.
        pad = 4 + ring_index * 4   # 4 → 8 px
        frame_top = bar_y - pad
        frame_bot = bar_y + 8 + pad

        if active:
            outline = (*GUIDANCE_ACTIVE_BORDER, 255)
            outline_w = 3
        else:
            # Faded version of the same hue so OFF state is recognizable
            # as "the window outline" without being distracting.
            outline = (*GUIDANCE_ACTIVE_BORDER, 110)
            outline_w = 2

        draw.rounded_rectangle(
            [gx1, frame_top, gx2, frame_bot],
            radius=5,
            outline=outline,
            width=outline_w,
        )

        # ----- label placement -----
        # Place label above the outer edge of this ring.
        label_y = frame_top - 16
        label_color_on  = GUIDANCE_ACTIVE_BORDER
        label_color_off = (140, 110, 70)
        label_color = label_color_on if active else label_color_off

        if use_LR_tags:
            # Two-column case: tag at the left edge with L / R
            draw.text(
                (max(gx1 - 2, 4), label_y),
                side_tag, font=font_marker_label, fill=label_color,
            )
        else:
            # Single-guidance case: descriptive centered label
            label_text = "guidance ON" if active else "guidance window"
            ltw, _ = measure_text(draw, label_text, font_marker_label)
            frame_w = gx2 - gx1
            if frame_w > ltw + 10:
                draw.text(
                    (gx1 + (frame_w - ltw) // 2, label_y),
                    label_text, font=font_marker_label, fill=label_color,
                )
            else:
                draw.text(
                    (gx1, label_y),
                    label_text, font=font_marker_label, fill=label_color,
                )

    # Draw inner ring first (right side), then outer ring (left side).
    # When only one side has guidance, that side gets the inner ring.
    if use_LR_tags:
        draw_guidance_region(right_guidance, ring_index=0, side_tag="R")
        draw_guidance_region(left_guidance,  ring_index=1, side_tag="L")
    else:
        draw_guidance_region(right_guidance, ring_index=0, side_tag="R")
        draw_guidance_region(left_guidance,  ring_index=0, side_tag="L")

    # Fill (denoise progress: t goes from t_max -> 0)
    progress_frac = (t_max - t_value) / t_max
    fill_x = bar_x_left + int(bar_w * progress_frac)
    if fill_x > bar_x_left:
        draw.rounded_rectangle(
            [bar_x_left, bar_y, fill_x, bar_y + 8],
            radius=4, fill=PROGRESS_FILL,
        )

    # End labels (positioned vertically aligned with the bar mid-line,
    # outside the guidance frame so they remain readable when ON)
    end_label_font = load_font(15)
    _, elh = measure_text(draw, "noise", end_label_font)
    end_label_y = bar_y + (8 - elh) // 2 - 1
    draw.text((PADDING, end_label_y), "noise",
              font=end_label_font, fill=SUBTITLE_COLOR)
    draw.text((canvas_w - PADDING - 36, end_label_y), "final",
              font=end_label_font, fill=SUBTITLE_COLOR)

    # Current t value (placed below the guidance frame so they don't collide)
    t_label = f"t = {t_value}"
    tw, _ = measure_text(draw, t_label, font_label)
    draw.text(
        ((canvas_w - tw) // 2, bar_y + 26),
        t_label, font=font_label, fill=TITLE_COLOR,
    )

    return img


# ---------------------------------------------------------------------------
# Column header
# ---------------------------------------------------------------------------
def make_column_header(width: int, label: str, bg: tuple) -> Image.Image:
    img = Image.new("RGB", (width, COLUMN_LABEL_H), bg)
    draw = ImageDraw.Draw(img)
    # Pick the largest font size that still fits the label within the column.
    max_text_w = width - 16  # 8px padding on each side
    for size in (22, 20, 18, 16):
        font = load_font(size, bold=True)
        tw, _ = measure_text(draw, label, font)
        if tw <= max_text_w:
            break
    draw_centered_text(
        draw, label, font, COLUMN_LABEL_FG,
        width / 2, COLUMN_LABEL_H / 2,
    )
    return img


# ---------------------------------------------------------------------------
# Active guidance highlight (brown frame around currently-guided column)
# ---------------------------------------------------------------------------
def apply_active_border(img: Image.Image) -> Image.Image:
    """Draw a warm brown border *inside* the image to signal 'guidance ON'."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    bw = GUIDANCE_ACTIVE_BORDER_W
    draw.rectangle(
        [0, 0, w - 1, h - 1],
        outline=GUIDANCE_ACTIVE_BORDER,
        width=bw,
    )
    # Small "GUIDANCE ON" pill at the top-right corner of the image
    pill_text = "GUIDANCE ON"
    pill_font = load_font(13, bold=True)
    tw, th = measure_text(draw, pill_text, pill_font)
    pill_w = tw + 12
    pill_h = th + 6
    margin = 6
    x0 = w - pill_w - margin
    y0 = margin
    draw.rounded_rectangle(
        [x0, y0, x0 + pill_w, y0 + pill_h],
        radius=4, fill=GUIDANCE_ACTIVE_BORDER,
    )
    draw.text(
        (x0 + 6, y0 + (pill_h - th) // 2 - 1),
        pill_text, font=pill_font, fill=(255, 255, 255),
    )
    return out


# ---------------------------------------------------------------------------
# Final-frame outcome box
# ---------------------------------------------------------------------------
def overlay_box(img: Image.Image, color, label) -> Image.Image:
    w, h = img.size
    out = Image.new("RGB", (w, h + LABEL_STRIP_H), BG_COLOR)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    if color is None or label is None:
        return out
    border_w = 4
    draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=border_w)
    font = load_font(16, bold=True)
    tw, th = measure_text(draw, label, font)
    label_y_top = h + 4
    label_box_w = tw + 16
    label_box_h = LABEL_STRIP_H - 8
    x0 = (w - label_box_w) // 2
    draw.rounded_rectangle(
        [x0, label_y_top, x0 + label_box_w, label_y_top + label_box_h],
        radius=4, fill=color,
    )
    draw.text(
        (x0 + 8, label_y_top + (label_box_h - th) // 2 - 2),
        label, font=font, fill=(255, 255, 255),
    )
    return out


# ---------------------------------------------------------------------------
# Compose one frame
# ---------------------------------------------------------------------------
def compose_frame(
    left_img: Image.Image,
    right_img: Image.Image,
    t: int,
    is_final: bool,
    canvas_w: int,
    cfg: GifConfig,
    t_max: int,
) -> Image.Image:
    col_w = (canvas_w - 2 * PADDING - GUTTER) // 2

    def fit(img: Image.Image) -> Image.Image:
        new_h = int(round(img.height * (col_w / img.width)))
        return img.resize((col_w, new_h), Image.LANCZOS)

    left_fit  = fit(left_img)
    right_fit = fit(right_img)

    # Highlight currently-guided column with a brown active border
    if _guidance_is_active(cfg.left.guidance, t):
        left_fit = apply_active_border(left_fit)
    if _guidance_is_active(cfg.right.guidance, t):
        right_fit = apply_active_border(right_fit)

    if is_final:
        left_fit  = overlay_box(left_fit,  cfg.left.final_color,  cfg.left.final_label)
        right_fit = overlay_box(right_fit, cfg.right.final_color, cfg.right.final_label)
    else:
        left_fit  = overlay_box(left_fit,  None, None)
        right_fit = overlay_box(right_fit, None, None)

    frame_h = max(left_fit.height, right_fit.height)
    total_h = (
        TOP_TITLE_H + SUBTITLE_H
        + COLUMN_LABEL_H + frame_h
        + PROGRESS_BAR_H + PADDING
    )
    canvas = Image.new("RGB", (canvas_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Title
    title_font = load_font(24, bold=True)
    draw_centered_text(
        draw, cfg.title, title_font, TITLE_COLOR,
        canvas_w / 2, TOP_TITLE_H / 2 + 4,
    )
    # Subtitle
    subtitle_font = load_font(15)
    subtitle_parts = [
        "Same noise seed", "16³ voxels", "DDPM 1000 steps",
        f"Sample #{cfg.sample_number}",
    ]
    if cfg.subtitle_suffix:
        subtitle_parts.append(cfg.subtitle_suffix)
    subtitle = "  ·  ".join(subtitle_parts)
    draw_centered_text(
        draw, subtitle, subtitle_font, SUBTITLE_COLOR,
        canvas_w / 2, TOP_TITLE_H + SUBTITLE_H / 2,
    )

    # Column headers
    y = TOP_TITLE_H + SUBTITLE_H
    x_left = PADDING
    x_right = PADDING + col_w + GUTTER
    canvas.paste(
        make_column_header(col_w, cfg.left.label, cfg.left.header_bg),
        (x_left, y),
    )
    canvas.paste(
        make_column_header(col_w, cfg.right.label, cfg.right.header_bg),
        (x_right, y),
    )

    # Frame images
    y += COLUMN_LABEL_H
    canvas.paste(left_fit,  (x_left, y))
    canvas.paste(right_fit, (x_right, y))

    # Progress bar (with per-side guidance markers)
    y += frame_h + PADDING // 2
    bar = make_progress_bar(
        canvas_w, PROGRESS_BAR_H, t, t_max,
        left_guidance=cfg.left.guidance,
        right_guidance=cfg.right.guidance,
    )
    canvas.paste(bar, (0, y))

    return canvas


# ---------------------------------------------------------------------------
# Non-linear playback
# ---------------------------------------------------------------------------
def build_playback_schedule(t_values: list[int]) -> list[int]:
    schedule = []
    final_t = min(t_values)
    for idx, t in enumerate(t_values):
        if t == final_t:
            n_repeats = max(1, int(round(HOLD_FINAL_SEC * PLAYBACK_FPS)))
        elif t > FAST_PHASE_T_BOUNDARY:
            n_repeats = 0 if idx % 3 != 0 else 1
        elif SLOW_PHASE_T_LO >= t >= SLOW_PHASE_T_HI:
            n_repeats = 1
        else:
            n_repeats = 1
        schedule.extend([idx] * n_repeats)
    if not schedule:
        schedule = list(range(len(t_values)))
    return schedule


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_gif(cfg: GifConfig) -> None:
    print(f"[1/5] Discovering frames for sample {cfg.sample_number}...")
    left_frames  = discover_frames(cfg.left.folder,  cfg.sample_number)
    right_frames = discover_frames(cfg.right.folder, cfg.sample_number)
    print(f"    left  ({cfg.left.label}): {len(left_frames)} frames")
    print(f"    right ({cfg.right.label}): {len(right_frames)} frames")

    left_by_step  = {step: (t, p) for step, t, p in left_frames}
    right_by_step = {step: (t, p) for step, t, p in right_frames}
    common_steps = sorted(set(left_by_step) & set(right_by_step))
    if not common_steps:
        raise RuntimeError("No matching step values between left and right folders.")
    print(f"[2/5] {len(common_steps)} matching timesteps")

    print(f"[3/5] Loading and cropping frames...")
    col_w = (TARGET_WIDTH - 2 * PADDING - GUTTER) // 2
    left_imgs  = [load_and_crop(left_by_step[s][1],  col_w) for s in common_steps]
    right_imgs = [load_and_crop(right_by_step[s][1], col_w) for s in common_steps]
    t_values = [left_by_step[s][0] for s in common_steps]
    t_max = max(t_values)
    print(f"    t range: {min(t_values)} → {t_max}")

    print(f"[4/5] Composing canvas frames...")
    composed = []
    final_t = min(t_values)
    for i, t in enumerate(t_values):
        composed.append(
            compose_frame(
                left_imgs[i], right_imgs[i],
                t=t, is_final=(t == final_t),
                canvas_w=TARGET_WIDTH,
                cfg=cfg, t_max=t_max,
            )
        )

    print(f"[5/5] Building GIF with non-linear playback...")
    schedule = build_playback_schedule(t_values)
    gif_frames = [composed[i] for i in schedule]
    print(f"    total GIF frames: {len(gif_frames)}")
    print(f"    duration: ~{len(gif_frames) / PLAYBACK_FPS:.1f}s @ {PLAYBACK_FPS}fps")

    quantized = [
        f.quantize(colors=GIF_QUANTIZE_COLORS, method=Image.MEDIANCUT)
        for f in gif_frames
    ]

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        cfg.output,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / PLAYBACK_FPS),
        loop=0,
        optimize=GIF_OPTIMIZE,
        disposal=2,
    )

    size_kb = cfg.output.stat().st_size / 1024
    print(f"\n✓ Done: {cfg.output}  ({size_kb:.0f} KB)")
    if size_kb > 5000:
        print("  ⚠ File > 5 MB. Reduce TARGET_WIDTH, PLAYBACK_FPS, or GIF_QUANTIZE_COLORS.")


def parse_args() -> GifConfig:
    p = argparse.ArgumentParser(
        description="Build CAST comparison GIF (baseline_vs_specific_window or full_vs_specific_window)."
    )
    p.add_argument("--preset", required=True,
                   choices=["baseline_vs_specific_window", "full_vs_specific_window"],
                   help="Which comparison to render.")
    p.add_argument("--left-dir",  required=True, type=Path)
    p.add_argument("--right-dir", required=True, type=Path)
    p.add_argument("--sample", required=True, type=int,
                   help="Sample number (must exist in both folders, same seed).")
    p.add_argument("--output", required=True, type=Path)
    # Optional overrides
    p.add_argument("--left-label",  default=None)
    p.add_argument("--right-label", default=None)
    args = p.parse_args()

    cfg = make_preset(args.preset, args.left_dir, args.right_dir,
                      args.sample, args.output)
    if args.left_label:
        cfg.left.label  = args.left_label
    if args.right_label:
        cfg.right.label = args.right_label
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    build_gif(cfg)