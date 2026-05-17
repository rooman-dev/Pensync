"""Generate synthetic handwriting sheets from .ttf fonts.

This utility creates large, widely spaced glyph grids that can be fed back into
the DIP pipeline to bootstrap handwriting profiles without manually writing
multiple pages.

Usage:
    python tools/generate_preset_sheets.py
    python tools/generate_preset_sheets.py --fonts-dir .

By default, the script scans the repository root for .ttf files and writes one
PNG sheet per font to the repository root as synthetic_sheet_<fontname>.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
PUNCTUATION = ".,-~*:'\"&[](){}<>!?/\\|_+=;@#$%^`"

# The compositor may synthesize these labels from verified samples, so we keep
# the preset sheet broad enough to exercise the same glyph space.
CHARACTER_SET = UPPERCASE + LOWERCASE + DIGITS + PUNCTUATION


def discover_fonts(fonts_dir: Path) -> List[Path]:
    """Return .ttf fonts from a directory in stable sorted order."""

    if not fonts_dir.exists():
        return []
    return sorted(path for path in fonts_dir.glob("*.ttf") if path.is_file())


def load_font(font_path: Path, font_size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at the requested size."""

    return ImageFont.truetype(str(font_path), font_size)


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Return a robust bounding box for a single glyph."""

    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox is None:
        return 0, 0, 0, 0
    return bbox


def render_sheet(font_path: Path, canvas_size: tuple[int, int] = (2400, 2400), font_size: int = 96) -> Image.Image:
    """Render one sheet per font using a wide grid of isolated glyphs."""

    canvas = Image.new("L", canvas_size, color=255)
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_path, font_size)

    width, height = canvas_size
    columns = 10
    rows = max(1, (len(CHARACTER_SET) + columns - 1) // columns)

    # Extremely wide cells so OpenCV contour detection won't merge adjacent glyphs.
    cell_width = width // columns
    cell_height = height // rows

    horizontal_padding = int(cell_width * 0.15)
    vertical_padding = int(cell_height * 0.18)

    for index, character in enumerate(CHARACTER_SET):
        row = index // columns
        column = index % columns

        cell_left = column * cell_width
        cell_top = row * cell_height
        cell_right = cell_left + cell_width
        cell_bottom = cell_top + cell_height

        bbox = text_bbox(draw, character, font)
        glyph_width = bbox[2] - bbox[0]
        glyph_height = bbox[3] - bbox[1]

        # Center each glyph in its own large cell with plenty of spacing.
        draw_x = cell_left + (cell_width - glyph_width) // 2 - bbox[0]
        draw_y = cell_top + (cell_height - glyph_height) // 2 - bbox[1]

        # Keep the glyph safely inside the cell even for wide punctuation.
        draw_x = max(cell_left + horizontal_padding, min(draw_x, cell_right - horizontal_padding))
        draw_y = max(cell_top + vertical_padding, min(draw_y, cell_bottom - vertical_padding))

        draw.text((draw_x, draw_y), character, fill=0, font=font)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic handwriting sheets from .ttf fonts.")
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory containing .ttf files. Defaults to the repository root.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        nargs=2,
        default=(2400, 2400),
        metavar=("WIDTH", "HEIGHT"),
        help="Output canvas size in pixels.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=96,
        help="Font size in points used to render the glyphs.",
    )
    args = parser.parse_args()

    fonts = discover_fonts(args.fonts_dir)
    if not fonts:
        raise SystemExit(f"No .ttf fonts found in {args.fonts_dir}")

    for font_path in fonts:
        sheet = render_sheet(font_path, canvas_size=tuple(args.canvas_size), font_size=args.font_size)
        output_name = f"synthetic_sheet_{font_path.stem}.png"
        output_path = PROJECT_ROOT / output_name
        sheet.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
