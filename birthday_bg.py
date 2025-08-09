#!/usr/bin/env python3

import argparse
import os
from io import BytesIO
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from rembg import remove, new_session
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "rembg (and onnxruntime) are required. Install with: pip install rembg onnxruntime pillow numpy"
    ) from exc


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def remove_background_foreground(image: Image.Image) -> Image.Image:
    session = new_session()
    result = remove(image, session=session, alpha_matting=True)
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    # Fallback: if bytes are returned by this rembg version
    from io import BytesIO
    out_img = Image.open(BytesIO(result)).convert("RGBA")
    return out_img


def create_vertical_gradient(size: Tuple[int, int], start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> Image.Image:
    width, height = size
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        gradient[:, :, c] = np.linspace(start_rgb[c], end_rgb[c], height, dtype=np.uint8)[:, None]
    return Image.fromarray(gradient, mode="RGB").convert("RGBA")


def draw_confetti(draw: ImageDraw.ImageDraw, width: int, height: int, density: int = 250) -> None:
    rng = np.random.default_rng()
    colors = [
        (255, 99, 132, 200),  # pink
        (255, 205, 86, 200),  # yellow
        (75, 192, 192, 200),  # teal
        (54, 162, 235, 200),  # blue
        (153, 102, 255, 200),  # purple
        (255, 159, 64, 200),  # orange
    ]
    for _ in range(density):
        r = int(rng.integers(4, 12))
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        color = colors[int(rng.integers(0, len(colors)))]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_bunting(draw: ImageDraw.ImageDraw, width: int, top: int = 0, height: int = 70) -> None:
    colors = [
        (255, 105, 97, 255),
        (119, 221, 119, 255),
        (135, 206, 235, 255),
        (255, 179, 71, 255),
        (197, 179, 255, 255),
    ]
    num_flags = max(8, width // 120)
    spacing = width / num_flags
    rope_y = top + int(height * 0.25)
    draw.line([(0, rope_y), (width, rope_y)], fill=(120, 120, 120, 255), width=3)
    for i in range(num_flags + 1):
        x_center = int(i * spacing)
        color = colors[i % len(colors)]
        triangle = [
            (x_center - int(spacing * 0.4), rope_y),
            (x_center + int(spacing * 0.4), rope_y),
            (x_center, rope_y + int(height * 0.9)),
        ]
        draw.polygon(triangle, fill=color)


def safe_load_font(preferred: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    candidates = [preferred] if preferred else []
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def add_birthday_text(canvas: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    # Font size relative to image width
    font = safe_load_font(None, size=max(32, width // 12))

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = int(0.04 * width)
    x = (width - text_width) // 2
    y = padding

    # semi-transparent rounded rect background for readability
    rect_margin = int(0.5 * padding)
    rect = [
        x - rect_margin,
        y - rect_margin,
        x + text_width + rect_margin,
        y + text_height + rect_margin,
    ]
    radius = int(rect_margin * 0.8)
    try:
        draw.rounded_rectangle(rect, radius=radius, fill=(255, 255, 255, 180))
    except Exception:
        draw.rectangle(rect, fill=(255, 255, 255, 180))

    draw.text((x, y), text, font=font, fill=(40, 40, 40, 255))


def composite_subject_on_background(
    subject_rgba: Image.Image, bg: Image.Image, subject_scale: float = 0.55
) -> Image.Image:
    bg = bg.convert("RGBA")
    out = bg.copy()
    bg_w, bg_h = bg.size

    # Resize subject to fit into the frame nicely
    subj_w, subj_h = subject_rgba.size
    target_h = int(bg_h * subject_scale)
    scale = target_h / max(1, subj_h)
    new_size = (max(1, int(subj_w * scale)), max(1, int(subj_h * scale)))
    subject_resized = subject_rgba.resize(new_size, Image.LANCZOS)

    # Place near bottom center with some margin
    x = (bg_w - subject_resized.width) // 2
    y = int(bg_h - subject_resized.height - bg_h * 0.08)

    out.alpha_composite(subject_resized, dest=(x, y))
    return out


def build_generated_birthday_background(width: int, height: int) -> Image.Image:
    base = create_vertical_gradient((width, height), (255, 240, 248), (220, 245, 255))
    # Light blur to soften gradient
    base = base.filter(ImageFilter.GaussianBlur(radius=0.6))
    draw = ImageDraw.Draw(base)
    draw_bunting(draw, width, top=0, height=int(height * 0.12))
    draw_confetti(draw, width, height, density=max(150, (width * height) // (24_000)))
    return base


def load_or_create_background(bg_path: Optional[str], size: Tuple[int, int]) -> Image.Image:
    width, height = size
    if bg_path:
        bg = load_image(bg_path)
        return bg.resize((width, height), Image.LANCZOS)
    return build_generated_birthday_background(width, height)


def process(input_path: str, output_path: str, name: Optional[str], bg_path: Optional[str], size: Optional[str]) -> str:
    src = load_image(input_path)

    # Determine output canvas size
    if size:
        try:
            w_str, h_str = size.lower().split("x")
            canvas_size = (int(w_str), int(h_str))
        except Exception as exc:
            raise SystemExit("--size must be like 1080x1350") from exc
    else:
        # Use portrait-friendly canvas based on source aspect
        s_w, s_h = src.size
        aspect = s_w / max(1, s_h)
        if aspect >= 1:
            canvas_size = (1200, 900)
        else:
            canvas_size = (1080, 1440)

    subject = remove_background_foreground(src)
    background = load_or_create_background(bg_path, canvas_size)
    composed = composite_subject_on_background(subject, background, subject_scale=0.58)

    title = f"Happy Birthday{', ' + name + '!' if name else '!'}"
    add_birthday_text(composed, title)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    composed.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place a photo onto a festive birthday background.")
    parser.add_argument("input", help="Path to the input photo")
    parser.add_argument("output", help="Path to save the composited image (e.g., out.png)")
    parser.add_argument("--name", help="Name to show in the greeting", default=None)
    parser.add_argument("--bg", help="Optional custom background image path", default=None)
    parser.add_argument("--size", help="Canvas size like 1080x1350; defaults chosen by orientation", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = process(args.input, args.output, args.name, args.bg, args.size)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()