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


def get_safe_label(char: str) -> str:
    """Return a filesystem-safe label for a character with explicit case separation."""

    if not char:
        return ""

    symbol = char[0]
    if symbol.isupper():
        return f"upper_{symbol}"
    if symbol.islower():
        return f"lower_{symbol}"
    if symbol.isdigit():
        return symbol
    return f"sym_{ord(symbol)}"


small_letters = "aceimnorsuvwxz"
ascender_letters = "bdfhklt"
descender_letters = "gjpqy"
tiny_punctuation = ".,"


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

    safe_character = get_safe_label(character)

    # Priority 1: Exact character match (preferred)
    direct_variants = char_db.get(safe_character)
    if direct_variants:
        return direct_variants[int(np.random.randint(0, len(direct_variants)))]

    # Priority 2: Smart case fallback (uppercase ↔ lowercase from database)
    swapped_case = character.swapcase()
    swapped_variants = char_db.get(get_safe_label(swapped_case))
    if swapped_variants:
        # Use the database variant BEFORE synthetic rendering
        return swapped_variants[int(np.random.randint(0, len(swapped_variants)))]

    # Priority 3: Synthetic fallback for the requested character — keep glyph identity
    # This ensures the generated page visually matches the input text even when the
    # user's database lacks that exact label. Pooling from unrelated variants can
    # produce the wrong letters (e.g. lots of 'B' or 'W'); so we prefer a clean
    # synthetic glyph first and only fall back to pooling as a last resort.
    try:
        return _render_fallback_character(character)
    except Exception:
        pass

    # Priority 4: Pool any available variant (low fidelity but better than nothing)
    pooled_variants = [variant for variants in char_db.values() for variant in variants]
    if pooled_variants:
        return pooled_variants[int(np.random.randint(0, len(pooled_variants)))]

    # If everything else fails, render a synthetic glyph (defensive)
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


def _thin_character_strokes(character_image: np.ndarray) -> np.ndarray:
    """Thin black ink slightly before mask generation while keeping white paper intact."""

    if character_image.ndim == 3:
        character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(character_image, kernel, iterations=1)


def _prepare_character_patch(
    character_image: np.ndarray,
    target_height: int = 56,
    thin_strokes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a character patch proportionally using LANCZOS for HD anti-aliased scaling.
    
    Uses PIL's high-quality LANCZOS resampling to ensure smooth edges and prevent
    pixelation, especially important for synthesized output.
    """

    if thin_strokes:
        character_image = _thin_character_strokes(character_image)

    if character_image.ndim != 2:
        raise ValueError("Character patches must be single-channel grayscale arrays.")

    height, width = character_image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.float32)

    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))

    # Use PIL's LANCZOS for high-quality anti-aliased scaling
    # Convert image to PIL, resize, and convert back
    pil_image = Image.fromarray(character_image)
    mask = _extract_ink_mask(character_image)
    pil_mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    
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


def _blend_character(
    page: np.ndarray,
    x: int,
    y: int,
    character_image: np.ndarray,
    target_height: int,
) -> int:
    """Blend one character into the page at the requested coordinates."""

    character_patch, ink_mask = _prepare_character_patch(character_image, target_height=target_height)
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

    blue_layer = np.zeros_like(page_region, dtype=np.float32)
    blue_layer[..., 2] = 139.0
    alpha = ink_mask[..., np.newaxis]
    blended = page_region * (1.0 - alpha) + blue_layer * alpha
    page[y : y + available_height, x : x + available_width] = np.clip(blended, 0, 255).astype(np.uint8)
    return available_width


def _get_typographic_settings(character: str, line_height: int) -> tuple[int, int]:
    """Return a per-character render height and vertical offset within the line box."""

    if not character or character == " " or character == "\n" or character == "\r":
        return line_height, 0

    if character in tiny_punctuation:
        target_height = max(1, int(round(line_height * 0.10)))
        y_offset = int(round(line_height * 0.90))
        return target_height, y_offset

    if character in small_letters:
        target_height = max(1, int(round(line_height * 0.45)))
        y_offset = int(round(line_height * 0.55))
        return target_height, y_offset

    if character in descender_letters:
        target_height = max(1, int(round(line_height * 0.65)))
        y_offset = int(round(line_height * 0.55))
        return target_height, y_offset

    if character in ascender_letters or character.isupper() or character.isdigit():
        return line_height, 0

    return line_height, 0


def _estimate_character_width(character: str, line_height: int) -> int:
    """Estimate the rendered width for one character at the current typographic scale."""

    if not character or character == "\r" or character == "\n":
        return 0

    if character == " ":
        return int(round(line_height * 0.4))

    target_height, _ = _get_typographic_settings(character, line_height)
    scale = target_height / float(line_height)
    base_width = line_height
    if character in tiny_punctuation:
        base_width = max(1, int(round(line_height * 0.18)))
    elif character in small_letters:
        base_width = max(1, int(round(line_height * 0.55)))
    elif character in descender_letters:
        base_width = max(1, int(round(line_height * 0.58)))
    elif character in ascender_letters or character.isupper() or character.isdigit():
        base_width = line_height
    else:
        base_width = max(1, int(round(line_height * 0.65)))

    return max(1, int(round(base_width * scale)))


def _estimate_word_width(word: str, line_height: int) -> int:
    """Estimate the width a word will occupy before drawing it."""

    if not word:
        return 0

    width = 0
    drawable_char_count = 0
    for character in word:
        if character in "\r\n":
            continue
        if character == " ":
            width += _estimate_character_width(character, line_height)
            continue
        width += _estimate_character_width(character, line_height)
        drawable_char_count += 1

    if drawable_char_count > 1:
        width += int(np.random.randint(2, 6)) * (drawable_char_count - 1)

    return width


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
    page = np.full((page_height, page_width, 3), 255, dtype=np.uint8)

    left_margin = 60
    right_margin = 60
    top_margin = 80
    bottom_margin = 80
    line_height = 58
    line_step = int(round(line_height * 1.5))
    space_advance = int(round(line_height * 0.42))

    current_x = left_margin
    current_y = top_margin

    words = text.split(" ")
    for word_index, word in enumerate(words):
        if current_y > page_height - bottom_margin:
            break

        if "\n" in word or "\r" in word:
            sublines = word.splitlines()
        else:
            sublines = [word]

        for subline_index, subline in enumerate(sublines):
            if subline_index > 0:
                current_x = left_margin
                current_y += line_step
                if current_y > page_height - bottom_margin:
                    break

            if not subline:
                continue

            word_width = _estimate_word_width(subline, line_height)
            if current_x + word_width > page_width - right_margin:
                current_x = left_margin
                current_y += line_step

            for character in subline:
                if character in "\r\n" or character == " ":
                    continue

                if current_y > page_height - bottom_margin:
                    break

                selected_variant = _resolve_variant(char_db, character)
                target_height, typographic_offset = _get_typographic_settings(character, line_height)
                baseline_jitter = int(np.random.randint(-3, 3))
                paste_y = max(0, current_y + typographic_offset + baseline_jitter)
                pasted_width = _blend_character(
                    page,
                    current_x,
                    paste_y,
                    selected_variant,
                    target_height=target_height,
                )

                if pasted_width <= 0:
                    current_x += int(np.random.randint(2, 6))
                    continue

                current_x += pasted_width + int(np.random.randint(2, 6))

            if current_y > page_height - bottom_margin:
                break

        if word_index < len(words) - 1:
            current_x += space_advance
            if current_x > page_width - right_margin:
                current_x = left_margin
                current_y += line_step

    return page