"""Slide rendering — paper-texture background with hand-lettered typography."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSETS_DIR = Path(__file__).parent / "assets"
TITLE_FONT_PATH = ASSETS_DIR / "PermanentMarker-Regular.ttf"
BODY_FONT_PATH = ASSETS_DIR / "Kalam-Regular.ttf"

INK_COLOR = (45, 38, 32)
ACCENT_COLOR = (183, 28, 28)

SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720


def generate_paper_texture(width: int = SLIDE_WIDTH, height: int = SLIDE_HEIGHT, seed: int = 42) -> Image.Image:
    """Off-white grain + soft vignette — a paper feel without an external image asset."""
    rng = np.random.default_rng(seed)
    base = np.array([246, 241, 228], dtype=np.float32)
    noise = rng.normal(0, 5, (height, width, 1)).astype(np.float32)
    arr = np.clip(base + noise, 0, 255)

    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width / 2, height / 2
    dist = np.sqrt(((xx - cx) / (width / 2)) ** 2 + ((yy - cy) / (height / 2)) ** 2)
    vignette = np.clip(1 - 0.14 * np.clip(dist - 0.75, 0, None) / 0.3, 0.86, 1.0)[..., None]
    arr = np.clip(arr * vignette, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    return img.filter(ImageFilter.GaussianBlur(0.5))


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def render_slide(title: str, bullets: list[str]) -> Image.Image:
    img = generate_paper_texture()
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(str(TITLE_FONT_PATH), 58)
    body_font = ImageFont.truetype(str(BODY_FONT_PATH), 34)

    margin_x = 90
    y = 80

    for line in _wrap_text(draw, title, title_font, SLIDE_WIDTH - 2 * margin_x):
        draw.text((margin_x, y), line, font=title_font, fill=ACCENT_COLOR)
        y += 74

    y += 10
    draw.line([(margin_x, y), (margin_x + 240, y)], fill=ACCENT_COLOR, width=6)
    y += 50

    for bullet in bullets:
        lines = _wrap_text(draw, bullet, body_font, SLIDE_WIDTH - 2 * margin_x - 50)
        draw.ellipse([margin_x, y + 14, margin_x + 14, y + 28], fill=INK_COLOR)
        bx = margin_x + 36
        for line in lines:
            draw.text((bx, y), line, font=body_font, fill=INK_COLOR)
            y += 46
        y += 16

    return img
