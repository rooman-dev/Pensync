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
import hashlib


# Cache for measured tight ink widths: key -> int
# Key format: "{safe_label}:{variant_index}:{line_height}". Cleared per-render.
_width_cache: Dict[str, int] = {}


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

    # Robust background detection: sample border/corner pixels to decide whether
    # the image background is light (white) or dark (black). Using the global
    # mean can flip for glyphs that occupy large area, producing inverted masks
    # where the page background becomes the foreground.
    h, w = character_image.shape[:2]
    corners = [
        character_image[0, 0],
        character_image[0, w - 1],
        character_image[h - 1, 0],
        character_image[h - 1, w - 1],
    ]
    bg_est = np.median(np.array(corners, dtype=np.float32))
    if bg_est > 127.0:
        # Background is light (white) -> ink is darker
        alpha = (255.0 - character_image.astype(np.float32)) / 255.0
    else:
        # Background is dark -> ink is lighter
        alpha = character_image.astype(np.float32) / 255.0

    return np.clip(alpha, 0.0, 1.0)


def _thin_character_strokes(character_image: np.ndarray) -> np.ndarray:
    """Thin black ink slightly before mask generation while keeping white paper intact."""

    if character_image.ndim == 3:
        character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(character_image, kernel, iterations=1)


def _adjust_stroke_width(resized_image: np.ndarray, mode: str = "thin", strength_ratio: float = 0.02) -> np.ndarray:
    """Adjust stroke width on a resized grayscale image using a kernel scaled to image size.

    mode: 'thin' uses erosion, 'thicken' uses dilation. strength_ratio controls kernel size
    as a fraction of the image height (e.g. 0.02 => kernel ~ 2% of height).
    """
    if resized_image.ndim == 3:
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized_image

    h = max(1, gray.shape[0])
    k = max(1, int(round(h * strength_ratio)))
    kernel = np.ones((k, k), np.uint8)

    if mode == "thin":
        adjusted = cv2.erode(gray, kernel, iterations=1)
    else:
        adjusted = cv2.dilate(gray, kernel, iterations=1)

    return adjusted


def _adjust_mask_width(resized_mask: np.ndarray, mode: str = "thin", strength_ratio: float = 0.02) -> np.ndarray:
    """Adjust the binary/float ink mask using morphological ops sized to image height.

    Operates on float masks in [0,1], converts to uint8 for morphology and back.
    """
    if resized_mask.ndim != 2:
        resized_mask = resized_mask[..., 0]

    h = max(1, resized_mask.shape[0])
    k = max(1, int(round(h * strength_ratio)))
    kernel = np.ones((k, k), np.uint8)

    mask_u8 = (resized_mask * 255.0).astype(np.uint8)
    try:
        if mode == "thin":
            adj = cv2.erode(mask_u8, kernel, iterations=1)
        else:
            adj = cv2.dilate(mask_u8, kernel, iterations=1)
        adj_f = adj.astype(np.float32) / 255.0
        return np.clip(adj_f, 0.0, 1.0)
    except Exception:
        return resized_mask


def _prepare_character_patch(
    character_image: np.ndarray,
    target_height: int = 56,
    thin_strokes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a character patch proportionally using LANCZOS for HD anti-aliased scaling.
    
    Uses PIL's high-quality LANCZOS resampling to ensure smooth edges and prevent
    pixelation, especially important for synthesized output.
    """

    if character_image.ndim != 2:
        raise ValueError("Character patches must be single-channel grayscale arrays.")

    # add small proportional padding to stabilize visual cap/x-height across variants
    height, width = character_image.shape[:2]
    pad = max(1, int(round(max(height, width) * 0.12)))
    # Use the image border median as the pad value so we don't force a white
    # background on images that originally had a dark background (e.g., processed
    # CNN inputs are padded with black). This prevents inverting the ink mask.
    corners = [
        character_image[0, 0],
        character_image[0, width - 1],
        character_image[height - 1, 0],
        character_image[height - 1, width - 1],
    ]
    pad_value = int(np.median(np.array(corners, dtype=np.float32)))
    padded = np.full((height + pad * 2, width + pad * 2), pad_value, dtype=character_image.dtype)
    padded[pad : pad + height, pad : pad + width] = character_image
    character_image = padded
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

    # After resizing, adjust stroke width on the mask (not the image) to keep
    # the alpha channel aligned with the pixel data. This avoids the wrong
    # blue-filled boxes caused by image/mask mismatch.
    try:
        resized_mask = _adjust_mask_width(resized_mask, mode=("thin" if thin_strokes else "thicken"), strength_ratio=0.02)
    except Exception:
        pass

    return resized_image, np.clip(resized_mask, 0.0, 1.0)


def _get_variant_from_db(char_db: Dict[str, List[np.ndarray]], character: str, variant_index: int | None) -> np.ndarray:
    """Return the specific variant from `char_db` if available, otherwise fallback."""
    safe = get_safe_label(character)
    variants = char_db.get(safe) if char_db is not None else None
    if variants and variant_index is not None and 0 <= variant_index < len(variants):
        return variants[variant_index]

    # If no valid index, try to resolve normally (may pool or synthesize)
    try:
        return _resolve_variant(char_db, character)
    except Exception:
        return _render_fallback_character(character)


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

    # Compute tight ink width for kerning: find first/last column with ink
    col_sums = np.sum(ink_mask, axis=0)
    cols = np.where(col_sums > 1e-3)[0]
    if cols.size:
        ink_width = int(cols[-1] - cols[0] + 1)
    else:
        ink_width = max(1, available_width)

    return ink_width


def _get_typographic_settings(character: str, line_height: int) -> tuple[int, int]:
    """Return a per-character render height and vertical offset within the line box."""

    if not character or character == " " or character == "\n" or character == "\r":
        return line_height, 0

    if character in tiny_punctuation:
        target_height = max(10, int(round(line_height * 0.18)))
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

    # Without access to a character variant database, fall back to heuristic.
    # The precise tight-ink advance requires a sampled variant from the `char_db`.
    # To preserve compatibility, this function keeps the old heuristic when no
    # `char_db` is provided. If a `char_db` is passed via keyword, we will
    # compute the tight ink width from an actual variant.
    return max(1, int(round(line_height * 0.5)))


def _measure_tight_ink_width_from_variant(
    character: str,
    line_height: int,
    char_db: Dict[str, List[np.ndarray]] | None,
    variant_index: int | None = None,
) -> int:
    """Measure tight ink width for `character` by sampling a variant from `char_db`.

    This resizes the variant using the same pipeline as rendering and returns
    the number of columns that contain ink — this matches the kerning used
    by `_blend_character` which bases advancement on actual ink extent.
    """
    if character == " " or character == "" or character in "\r\n":
        return int(round(line_height * 0.4))

    safe = get_safe_label(character)

    # Determine variant list and default index
    variant_list = []
    if char_db is not None:
        variant_list = char_db.get(safe, [])

    # If no variants, use synthetic single variant
    if not variant_list:
        variant_list = [_render_fallback_character(character, size=64)]

    # Use module-level cache for measured widths (keyed by variant index)
    global _width_cache

    # If a specific variant_index is provided, measure that variant only
    if variant_index is not None and variant_index >= 0 and variant_index < len(variant_list):
        key = f"{safe}:{variant_index}:{line_height}"
        if key in _width_cache:
            return _width_cache[key]

        variant = variant_list[variant_index]
        target_height, _ = _get_typographic_settings(character, line_height)
        try:
            _, mask = _prepare_character_patch(variant, target_height=target_height, thin_strokes=True)
            col_sums = np.sum(mask, axis=0)
            cols = np.where(col_sums > 1e-3)[0]
            if cols.size:
                ink_width = int(cols[-1] - cols[0] + 1)
            else:
                ink_width = max(1, mask.shape[1])
        except Exception:
            ink_width = max(1, int(round(line_height * 0.5)))

        _width_cache[key] = ink_width
        return ink_width

    # Otherwise, fall back to measuring the first available variant (compat)
    for idx, variant in enumerate(variant_list):
        key = f"{safe}:{idx}:{line_height}"
        if key in _width_cache:
            return _width_cache[key]

        target_height, _ = _get_typographic_settings(character, line_height)
        try:
            _, mask = _prepare_character_patch(variant, target_height=target_height, thin_strokes=True)
            col_sums = np.sum(mask, axis=0)
            cols = np.where(col_sums > 1e-3)[0]
            if cols.size:
                ink_width = int(cols[-1] - cols[0] + 1)
            else:
                ink_width = max(1, mask.shape[1])
        except Exception:
            ink_width = max(1, int(round(line_height * 0.5)))

        _width_cache[key] = ink_width
        return ink_width


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
        # Prefer tight ink-based advance if `char_db` is available at call site.
        # Backwards-compatible signature: if caller supplies `char_db` via closure
        # we won't have it here; the generate function will call the variant-aware
        # helper directly. For now use heuristic estimate.
        width += _estimate_character_width(character, line_height)
        drawable_char_count += 1

    if drawable_char_count > 1:
        # reduce default intra-word spacing when estimating widths
        width += int(np.random.randint(1, 4)) * (drawable_char_count - 1)

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

    # Clear width cache to avoid cross-request memory leaks and ensure
    # measurements reflect the current `char_db` and `line_height`.
    global _width_cache
    _width_cache.clear()

    left_margin = 60
    right_margin = 60
    top_margin = 80
    bottom_margin = 80
    line_height = 58
    line_step = int(round(line_height * 1.5))
    # Slightly reduce space advance to tighten inter-word spacing
    space_advance = int(round(line_height * 0.34))

    current_x = left_margin
    current_y = top_margin

    # Draw ruled notebook lines under the text before pasting characters.
    # Use black lines for higher contrast with tightened letter spacing.
    line_color = (0, 0, 0)
    first_line_y = int(round(top_margin + line_height))
    y = first_line_y
    while y < page_height - bottom_margin:
        cv2.line(page, (0, y), (page_width, y), line_color, thickness=2)
        y += line_step

    # Pre-select deterministic variant indices per character in the text so
    # measurement and pasting refer to the same exemplar.
    seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    variant_map: List[int | None] = []
    for ch in text:
        if ch in "\r\n " or ch == "":
            variant_map.append(None)
            continue
        safe = get_safe_label(ch)
        variants = char_db.get(safe) if char_db is not None else None
        if variants:
            variant_map.append(int(rng.randint(0, len(variants))))
        else:
            variant_map.append(-1)

    pos = 0
    text_len = len(text)
    while pos < text_len:
        if current_y > page_height - bottom_margin:
            break

        ch = text[pos]
        # Handle explicit newlines
        if ch == "\n" or ch == "\r":
            current_x = left_margin
            current_y += line_step
            pos += 1
            continue

        # Handle spaces
        if ch == " ":
            current_x += space_advance
            pos += 1
            continue

        # Collect a word from pos to next whitespace
        start = pos
        end = pos
        while end < text_len and text[end] not in " \r\n":
            end += 1

        # Measure word width using the preselected variants
        word_width = 0
        drawable_char_count_local = 0
        for j in range(start, end):
            ch2 = text[j]
            if ch2 in "\r\n":
                continue
            if ch2 == " ":
                word_width += int(round(line_height * 0.4))
                continue
            v_idx = variant_map[j]
            # pass None if v_idx is -1 to allow fallback synthetic measurement
            measured = _measure_tight_ink_width_from_variant(ch2, line_height, char_db, variant_index=(None if v_idx == -1 else v_idx))
            word_width += measured
            drawable_char_count_local += 1

        if drawable_char_count_local > 1:
            # reduce per-character extra spacing in words
            word_width += int(np.random.randint(1, 4)) * (drawable_char_count_local - 1)

        if current_x + word_width > page_width - right_margin:
            current_x = left_margin
            current_y += line_step

        # Paste each character using the preselected variant indices
        for j in range(start, end):
            character = text[j]
            if character in "\r\n" or character == " ":
                continue
            if current_y > page_height - bottom_margin:
                break

            v_idx = variant_map[j]
            if v_idx is None or v_idx == -1:
                selected_variant = _render_fallback_character(character)
            else:
                selected_variant = _get_variant_from_db(char_db, character, v_idx)

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
                current_x += 1
            else:
                # reduce random gap between adjacent characters (0 or 1 px)
                gap = int(np.random.randint(0, 2))
                current_x += pasted_width + gap

        pos = end

    return page