"""Layout and synthesis compositor for Pensync.

This module reconstructs digital text into a handwritten-looking page by
placing normalized character variants onto a blank canvas with baseline jitter,
spacing variance, and word wrapping. The compositor remains rule-based so the
generation process is interpretable and easy to validate in an academic report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _render_fallback_character(character: str, size: int = 64) -> np.ndarray:
    """Render a high-definition fallback glyph using TrueType fonts.
    
    Attempts to load a system TrueType font (Arial or alternatives) for smooth,
    anti-aliased rendering. Falls back gracefully to a scaled PIL default if fonts
    are unavailable.
    """

    canvas = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a TrueType font for high-quality rendering
    font = None
    font_size = int(size * 0.65)  # ~42 pixels for 64x64 canvas
    
    # Common TrueType font paths on Windows, macOS, Linux
    font_candidates = [
        "arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "C:\\Users\\rooma\\AppData\\Local\\Microsoft\\Windows\\Fonts\\arial.ttf",
    ]
    
    for font_path in font_candidates:
        try:
            if Path(font_path).exists():
                font = ImageFont.truetype(font_path, font_size)
                break
        except Exception:
            continue
    
    # Fallback: use default font if TrueType is unavailable
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            pass
    
    # Render text with proper positioning
    x_offset = int(size * 0.15)
    y_offset = int(size * 0.08)
    draw.text((x_offset, y_offset), character, fill=0, font=font)
    
    return np.array(canvas, dtype=np.uint8)


def _resolve_variant(char_db: Dict[str, List[np.ndarray]], character: str) -> np.ndarray:
    """Select a usable character variant from the database or a fallback glyph.
    
    Smart case fallback: If exact character not found, tries case-swapped variant
    (e.g., uppercase 'H' → lowercase 'h' if available) before resorting to
    synthetic PIL rendering. This solves EMNIST ByClass case confusion.
    """

    # Priority 1: Exact character match (preferred)
    direct_variants = char_db.get(character)
    if direct_variants:
        return direct_variants[int(np.random.randint(0, len(direct_variants)))]

    # Priority 2: Smart case fallback (uppercase ↔ lowercase from database)
    swapped_case = character.swapcase()
    swapped_variants = char_db.get(swapped_case)
    if swapped_variants:
        # Use the database variant BEFORE synthetic rendering
        return swapped_variants[int(np.random.randint(0, len(swapped_variants)))]

    # Priority 3: Pool any available variant (low fidelity but better than nothing)
    pooled_variants = [variant for variants in char_db.values() for variant in variants]
    if pooled_variants:
        return pooled_variants[int(np.random.randint(0, len(pooled_variants)))]

    # Priority 4: Synthetic fallback (only if database is empty)
    return _render_fallback_character(character)


def _extract_ink_mask(character_image: np.ndarray) -> np.ndarray:
    """Convert a character image into a floating ink mask in the range [0, 1]."""

    if character_image.ndim == 3:
        character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY)

    if character_image.mean() > 127.0:
        alpha = (255.0 - character_image.astype(np.float32)) / 255.0
    else:
        alpha = character_image.astype(np.float32) / 255.0

    return np.clip(alpha, 0.0, 1.0)


def _prepare_character_patch(character_image: np.ndarray, target_height: int = 56) -> tuple[np.ndarray, np.ndarray]:
    """Crop and resize a character patch using LANCZOS for HD anti-aliased scaling.
    
    Uses PIL's high-quality LANCZOS resampling to ensure smooth edges and prevent
    pixelation, especially important for synthesized output.
    """

    mask = _extract_ink_mask(character_image)
    foreground = np.argwhere(mask > 0.05)
    if foreground.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.float32)

    top, left = foreground.min(axis=0)
    bottom, right = foreground.max(axis=0) + 1
    cropped_image = character_image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]

    height, width = cropped_image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.float32)

    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))

    # Use PIL's LANCZOS for high-quality anti-aliased scaling
    # Convert image to PIL, resize, and convert back
    pil_image = Image.fromarray(cropped_image)
    pil_mask_image = Image.fromarray((cropped_mask * 255).astype(np.uint8))
    
    try:
        # Try LANCZOS (high-quality, available in Pillow 10.0+)
        resized_pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        resized_pil_mask = pil_mask_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    except AttributeError:
        # Fallback for older Pillow versions
        resized_pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        resized_pil_mask = pil_mask_image.resize((target_width, target_height), Image.LANCZOS)
    
    resized_image = np.array(resized_pil_image, dtype=np.uint8)
    resized_mask = np.array(resized_pil_mask, dtype=np.float32) / 255.0
    
    return resized_image, np.clip(resized_mask, 0.0, 1.0)


def _blend_character(page: np.ndarray, x: int, y: int, character_image: np.ndarray) -> int:
    """Blend one character into the page at the requested coordinates."""

    ink_color = int(np.random.randint(10, 35))
    character_patch, ink_mask = _prepare_character_patch(character_image)
    patch_height, patch_width = character_patch.shape[:2]

    page_height, page_width = page.shape[:2]
    if x >= page_width or y >= page_height:
        return 0

    available_width = min(patch_width, page_width - x)
    available_height = min(patch_height, page_height - y)
    if available_width <= 0 or available_height <= 0:
        return 0

    ink_mask = ink_mask[:available_height, :available_width]
    page_region = page[y : y + available_height, x : x + available_width].astype(np.float32)

    ink_layer = np.full_like(page_region, fill_value=float(ink_color))
    blended = page_region * (1.0 - ink_mask) + ink_layer * ink_mask
    page[y : y + available_height, x : x + available_width] = np.clip(blended, 0, 255).astype(np.uint8)
    return available_width


def generate_handwritten_page(
    text: str,
    char_db: Dict[str, List[np.ndarray]],
    page_size: tuple[int, int] = (1200, 1600),
) -> np.ndarray:
    """Reconstruct digital text into a handwritten page using a rule-based compositor.

    The page is modeled as a blank white canvas. Characters are placed left to
    right with baseline jitter, random spacing noise, and word wrapping. When a
    matching handwritten variant is available in ``char_db``, one is sampled at
    random to mimic the natural variation captured by K-Means clustering.

    Parameters
    ----------
    text:
        Digital text that should be laid out as a handwritten page.
    char_db:
        Dictionary mapping ASCII characters to lists of 64x64 extracted or
        synthesized character variants.
    page_size:
        Tuple of ``(width, height)`` for the output page.

    Returns
    -------
    np.ndarray
        A grayscale page image with handwritten-looking characters rendered on a
        white background.
    """

    page_width, page_height = page_size
    page = np.full((page_height, page_width), 255, dtype=np.uint8)

    left_margin = 60
    right_margin = 60
    top_margin = 80
    bottom_margin = 80
    line_height = 72
    space_advance = 24

    x = left_margin
    y = top_margin

    for character in text:
        if character == "\r":
            continue

        if character == "\n":
            x = left_margin
            y += line_height
            continue

        if y > page_height - bottom_margin:
            break

        if character == " ":
            x += int(max(12, space_advance + np.random.normal(loc=0.0, scale=4.0)))
            if x > page_width - right_margin:
                x = left_margin
                y += line_height
            continue

        if x > page_width - right_margin - 80:
            x = left_margin
            y += line_height

        selected_variant = _resolve_variant(char_db, character)
        baseline_jitter = int(np.random.randint(-3, 3))
        paste_y = max(0, y + baseline_jitter)
        pasted_width = _blend_character(page, x, paste_y, selected_variant)

        if pasted_width <= 0:
            x += int(max(10, space_advance + np.random.normal(loc=0.0, scale=3.0)))
            continue

        x += pasted_width + int(max(6, np.random.normal(loc=8.0, scale=3.0)))

    return page