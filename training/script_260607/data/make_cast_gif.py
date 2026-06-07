"""
CAST: Baseline vs Guided sampling dynamics — GIF assembly script
================================================================

Input:
    Two folders, each containing 50 PNG frames (one per timestep).
    File naming pattern (regex):  dynamics_sample_NNN_step_SSSS_t_TTTT.png

    Example:
        baseline/dynamics_sample_003_step_0000_t_0999.png
        baseline/dynamics_sample_003_step_0020_t_0979.png
        ...
        guided/dynamics_sample_003_step_0980_t_0019.png

Output:
    A single optimized GIF showing baseline vs guided side-by-side,
    with progress bar, guidance window marker, and final-frame highlights.

Usage:
    1. Set the two paths below (BASELINE_DIR, GUIDED_DIR)
    2. Adjust SAMPLE_NUMBER if needed
    3. Run:  python make_cast_gif.py

Tunables:
    - GUIDANCE_T_START / GUIDANCE_T_END : where the guidance was active
    - HOLD_FINAL_SEC                    : how long to pause on final frame
    - FAST_PHASE_T_BOUNDARY             : below this t, frames play fast
    - PLAYBACK_FPS                      : output GIF frame rate
"""

from __future__ import annotations
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — change these
# ---------------------------------------------------------------------------
BASELINE_DIR   = Path("/Users/kylelin/projects/MineTree/assets/comparison_baseline_and_guidance_gif/baseline_frames")   # folder with baseline PNGs
GUIDED_DIR     = Path("/Users/kylelin/projects/MineTree/assets/comparison_baseline_and_guidance_gif/guided_frames")     # folder with guided PNGs
SAMPLE_NUMBER  = 24                           # sample index to use
OUTPUT_GIF     = Path("/Users/kylelin/projects/MineTree/assets/comparison_baseline_and_guidance_gif/cast_comparison.gif")

# Guidance window (matches your PoC report's chosen schedule)
GUIDANCE_T_START = 700  # guidance turns ON
GUIDANCE_T_END   = 200  # guidance turns OFF

# Visual layout
TARGET_WIDTH      = 900    # final GIF width (px). Smaller => smaller file.
TOP_TITLE_H       = 56     # top banner height
SUBTITLE_H        = 30     # "same seed..." line
COLUMN_LABEL_H    = 36     # "Baseline" / "+ CAST Guidance" headers
PROGRESS_BAR_H    = 56     # bottom progress bar area
GUTTER            = 12     # gap between baseline and guided columns
PADDING           = 16

# Frame crop (remove the embedded title rows from your PNGs)
# Your PNGs have TWO title rows at the top:
#   1. Big title:  eval_baseline_0025_033_e0400_..._e100   (~50 px)
#   2. Per-view:   "Z Sample 3, Step 980, t=19" subtitles  (~45 px)
# We crop both (≈ 95 px). Adjust if your frames have different padding.
FRAME_TOP_CROP    = 95
# Optionally compress wide white gutters between the 3 subplots
# (matplotlib leaves gaps between Z/Y/X views). Set to False to disable.
COMPRESS_INTERNAL_GAPS = True
GAP_KEEP_PX = 8  # how many px of gap to keep between subplots

# Playback timing (non-linear)
PLAYBACK_FPS              = 10
HOLD_FINAL_SEC            = 4   # pause on final frame
FAST_PHASE_T_BOUNDARY     = 2000   # noisy early frames -> play 3x speed
SLOW_PHASE_T_LO           = 800   # critical window: play normal speed
SLOW_PHASE_T_HI           = 200

# Colors (RGB)
BG_COLOR          = (245, 245, 247)
TITLE_COLOR       = (30, 30, 35)
SUBTITLE_COLOR    = (110, 110, 120)
COLUMN_LABEL_BG_BASE  = (235, 235, 240)
COLUMN_LABEL_BG_GUIDE = (220, 235, 220)
COLUMN_LABEL_FG   = (40, 40, 50)
PROGRESS_TRACK    = (220, 220, 225)
PROGRESS_FILL     = (60, 110, 200)
GUIDANCE_REGION   = (255, 200, 100)   # marker on progress bar
GUIDANCE_REGION_A = 110                # alpha
FAILURE_BOX_COLOR = (220, 60, 60)
SUCCESS_BOX_COLOR = (50, 160, 80)

# Output
GIF_QUANTIZE_COLORS = 64               # palette size, lower => smaller file
GIF_OPTIMIZE        = True

# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Pick a reasonable font available on most Linux/Mac/Win systems."""
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

FNAME_RE = re.compile(
    r"dynamics_sample_(\d+)_step_(\d+)_t_(\d+)\.png", re.IGNORECASE
)

def discover_frames(folder: Path, sample_num: int) -> list[tuple[int, int, Path]]:
    """Return list of (step, t, path), sorted by step ascending."""
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
    found.sort(key=lambda x: x[0])  # by step
    return found

# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def _compress_internal_gaps(img: Image.Image) -> Image.Image:
    """
    Detect wide vertical strips of background (white-ish) between subplots
    and compress them down to GAP_KEEP_PX. Works on column means.
    """
    arr = np.asarray(img)
    # A "gap column" = column where pixels are mostly near-white (R,G,B >= 240)
    near_white = (arr >= 240).all(axis=2)              # (H, W) bool
    col_is_gap = near_white.mean(axis=0) > 0.92        # (W,) bool
    if not col_is_gap.any():
        return img

    # Find runs of consecutive gap columns
    keep_mask = np.ones_like(col_is_gap, dtype=bool)
    i = 0
    W = len(col_is_gap)
    while i < W:
        if col_is_gap[i]:
            j = i
            while j < W and col_is_gap[j]:
                j += 1
            run_len = j - i
            # Don't compress edge gaps (keep figure padding intact)
            is_edge = (i == 0) or (j == W)
            if not is_edge and run_len > GAP_KEEP_PX:
                # Drop the middle of this run, keep GAP_KEEP_PX columns
                drop_start = i + GAP_KEEP_PX // 2
                drop_end = j - GAP_KEEP_PX // 2
                keep_mask[drop_start:drop_end] = False
            i = j
        else:
            i += 1
    if keep_mask.all():
        return img
    new_arr = arr[:, keep_mask, :]
    return Image.fromarray(new_arr)


def load_and_crop(path: Path, target_w: int) -> Image.Image:
    """Load PNG, crop title row off, optionally compress gaps, resize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.crop((0, FRAME_TOP_CROP, w, h))
    if COMPRESS_INTERNAL_GAPS:
        img = _compress_internal_gaps(img)
    new_w = target_w
    new_h = int(round(img.height * (new_w / img.width)))
    return img.resize((new_w, new_h), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Composite layout
# ---------------------------------------------------------------------------

def measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw, text, font, color, x_center, y_center):
    tw, th = measure_text(draw, text, font)
    draw.text(
        (x_center - tw / 2, y_center - th / 2),
        text, font=font, fill=color,
    )


def make_progress_bar(
    canvas_w: int,
    bar_h: int,
    t_value: int,
    t_max: int = 999,
) -> Image.Image:
    """Draw progress bar with guidance window highlighted."""
    img = Image.new("RGB", (canvas_w, bar_h), BG_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")
    font_small = load_font(13)
    font_label = load_font(14, bold=True)

    # Bar geometry
    bar_y = bar_h // 2 - 4
    bar_x_left = PADDING + 60
    bar_x_right = canvas_w - PADDING - 60
    bar_w = bar_x_right - bar_x_left

    # Track
    draw.rounded_rectangle(
        [bar_x_left, bar_y, bar_x_right, bar_y + 8],
        radius=4, fill=PROGRESS_TRACK,
    )

    # Guidance region overlay (drawn first, behind fill)
    g_start_frac = (t_max - GUIDANCE_T_START) / t_max
    g_end_frac   = (t_max - GUIDANCE_T_END) / t_max
    gx1 = bar_x_left + int(bar_w * g_start_frac)
    gx2 = bar_x_left + int(bar_w * g_end_frac)
    draw.rounded_rectangle(
        [gx1, bar_y - 2, gx2, bar_y + 10],
        radius=4,
        fill=(*GUIDANCE_REGION, GUIDANCE_REGION_A),
    )

    # Fill (denoising progress: t goes from t_max -> 0)
    progress_frac = (t_max - t_value) / t_max
    fill_x = bar_x_left + int(bar_w * progress_frac)
    if fill_x > bar_x_left:
        draw.rounded_rectangle(
            [bar_x_left, bar_y, fill_x, bar_y + 8],
            radius=4, fill=PROGRESS_FILL,
        )

    # End labels
    draw.text(
        (PADDING, bar_y - 3), "noise",
        font=font_small, fill=SUBTITLE_COLOR,
    )
    draw.text(
        (canvas_w - PADDING - 30, bar_y - 3), "final",
        font=font_small, fill=SUBTITLE_COLOR,
    )

    # Current t value
    t_label = f"t = {t_value}"
    tw, th = measure_text(draw, t_label, font_label)
    draw.text(
        ((canvas_w - tw) // 2, bar_y + 14),
        t_label, font=font_label, fill=TITLE_COLOR,
    )

    # Guidance region label (shown when in window)
    if GUIDANCE_T_END <= t_value <= GUIDANCE_T_START:
        guidance_label = "guidance ON"
        gtw, _ = measure_text(draw, guidance_label, font_small)
        draw.text(
            (gx1 + (gx2 - gx1 - gtw) // 2, bar_y - 18),
            guidance_label,
            font=font_small,
            fill=(180, 110, 30),
        )

    return img


def make_column_header(width: int, label: str, is_guided: bool) -> Image.Image:
    bg = COLUMN_LABEL_BG_GUIDE if is_guided else COLUMN_LABEL_BG_BASE
    img = Image.new("RGB", (width, COLUMN_LABEL_H), bg)
    draw = ImageDraw.Draw(img)
    font = load_font(20, bold=True)
    draw_centered_text(
        draw, label, font, COLUMN_LABEL_FG,
        width / 2, COLUMN_LABEL_H / 2,
    )
    return img


LABEL_STRIP_H = 32  # always reserved below each frame image

def overlay_box(img: Image.Image, color: tuple | None, label: str | None) -> Image.Image:
    """
    Always returns img + label_strip_h pixels at the bottom, so all frames
    have identical height (no GIF jitter). When color/label given, draws
    a colored border around the image and a label pill in the strip.
    """
    w, h = img.size
    out = Image.new("RGB", (w, h + LABEL_STRIP_H), BG_COLOR)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    if color is None or label is None:
        return out
    border_w = 4
    draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=border_w)
    font = load_font(15, bold=True)
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


def compose_frame(
    base_img: Image.Image,
    guided_img: Image.Image,
    t: int,
    is_final: bool,
    canvas_w: int,
) -> Image.Image:
    """Stack everything into one canvas frame."""
    col_w = (canvas_w - 2 * PADDING - GUTTER) // 2

    # Resize each frame image to column width
    def fit(img: Image.Image) -> Image.Image:
        new_h = int(round(img.height * (col_w / img.width)))
        return img.resize((col_w, new_h), Image.LANCZOS)

    base_fit = fit(base_img)
    guided_fit = fit(guided_img)

    if is_final:
        base_fit = overlay_box(base_fit, FAILURE_BOX_COLOR, "Disconnected")
        guided_fit = overlay_box(guided_fit, SUCCESS_BOX_COLOR, "Connected")
    else:
        # Reserve the same vertical space so every frame has identical height
        base_fit = overlay_box(base_fit, None, None)
        guided_fit = overlay_box(guided_fit, None, None)

    frame_h = max(base_fit.height, guided_fit.height)

    total_h = (
        TOP_TITLE_H + SUBTITLE_H
        + COLUMN_LABEL_H + frame_h
        + PROGRESS_BAR_H + PADDING
    )
    canvas = Image.new("RGB", (canvas_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Title
    title_font = load_font(22, bold=True)
    draw_centered_text(
        draw,
        "CAST: Connectivity-Aware Sampling for 3D Voxel Diffusion",
        title_font, TITLE_COLOR,
        canvas_w / 2, TOP_TITLE_H / 2 + 4,
    )
    # Subtitle
    subtitle_font = load_font(13)
    draw_centered_text(
        draw,
        f"Same noise seed  ·  16³ voxels  ·  DDPM 1000 steps  ·  Sample #{SAMPLE_NUMBER}",
        subtitle_font, SUBTITLE_COLOR,
        canvas_w / 2, TOP_TITLE_H + SUBTITLE_H / 2,
    )

    # Column headers
    y = TOP_TITLE_H + SUBTITLE_H
    x_left = PADDING
    x_right = PADDING + col_w + GUTTER
    canvas.paste(make_column_header(col_w, "Baseline", False), (x_left, y))
    canvas.paste(make_column_header(col_w, "+ CAST Guidance", True), (x_right, y))

    # Frame images
    y += COLUMN_LABEL_H
    canvas.paste(base_fit, (x_left, y))
    canvas.paste(guided_fit, (x_right, y))

    # Progress bar
    y += frame_h + PADDING // 2
    bar = make_progress_bar(canvas_w, PROGRESS_BAR_H, t)
    canvas.paste(bar, (0, y))

    return canvas

# ---------------------------------------------------------------------------
# Non-linear playback timing
# ---------------------------------------------------------------------------

def build_playback_schedule(t_values: list[int]) -> list[int]:
    """Decide how many times to repeat each frame index, based on its t."""
    schedule = []
    for idx, t in enumerate(t_values):
        is_final = (t == min(t_values))
        if is_final:
            # Hold the last frame
            n_repeats = max(1, int(round(HOLD_FINAL_SEC * PLAYBACK_FPS)))
        elif t > FAST_PHASE_T_BOUNDARY:
            # Fast-forward through pure noise
            n_repeats = 0 if idx % 3 != 0 else 1
        elif SLOW_PHASE_T_LO >= t >= SLOW_PHASE_T_HI:
            # Critical window: normal speed (1 frame per slot)
            n_repeats = 1
        else:
            # Otherwise normal
            n_repeats = 1
        schedule.extend([idx] * n_repeats)
    if not schedule:
        schedule = list(range(len(t_values)))
    return schedule

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[1/5] Discovering frames...")
    base_frames = discover_frames(BASELINE_DIR, SAMPLE_NUMBER)
    guided_frames = discover_frames(GUIDED_DIR, SAMPLE_NUMBER)
    print(f"    baseline: {len(base_frames)} frames")
    print(f"    guided:   {len(guided_frames)} frames")

    # Pair frames by step (assumes same step grid)
    base_by_step = {step: (t, p) for step, t, p in base_frames}
    guided_by_step = {step: (t, p) for step, t, p in guided_frames}
    common_steps = sorted(set(base_by_step) & set(guided_by_step))
    if not common_steps:
        raise RuntimeError(
            "No matching step values between baseline and guided. "
            "Check your file naming."
        )
    print(f"[2/5] {len(common_steps)} matching timesteps")

    print(f"[3/5] Loading and cropping frames...")
    col_w = (TARGET_WIDTH - 2 * PADDING - GUTTER) // 2
    base_imgs = [load_and_crop(base_by_step[s][1], col_w) for s in common_steps]
    guided_imgs = [load_and_crop(guided_by_step[s][1], col_w) for s in common_steps]
    t_values = [base_by_step[s][0] for s in common_steps]

    print(f"[4/5] Composing canvas frames...")
    composed = []
    final_t = min(t_values)
    for i, t in enumerate(t_values):
        composed.append(
            compose_frame(
                base_imgs[i], guided_imgs[i],
                t=t, is_final=(t == final_t),
                canvas_w=TARGET_WIDTH,
            )
        )

    print(f"[5/5] Building GIF with non-linear playback...")
    schedule = build_playback_schedule(t_values)
    gif_frames = [composed[i] for i in schedule]
    print(f"    total GIF frames: {len(gif_frames)}")
    print(f"    duration: ~{len(gif_frames) / PLAYBACK_FPS:.1f}s @ {PLAYBACK_FPS}fps")

    # Quantize for smaller file
    quantized = [
        f.quantize(colors=GIF_QUANTIZE_COLORS, method=Image.MEDIANCUT)
        for f in gif_frames
    ]

    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / PLAYBACK_FPS),
        loop=0,
        optimize=GIF_OPTIMIZE,
        disposal=2,
    )

    size_kb = OUTPUT_GIF.stat().st_size / 1024
    print(f"\n✓ Done: {OUTPUT_GIF}  ({size_kb:.0f} KB)")
    if size_kb > 5000:
        print(
            "  ⚠ File > 5 MB. Try reducing TARGET_WIDTH, PLAYBACK_FPS, "
            "or GIF_QUANTIZE_COLORS."
        )


if __name__ == "__main__":
    main()
