"""Import a handwriting sample sheet and populate the processed/verified dataset.

Usage:
  python tools/import_handwriting_sheet.py --image PATH_TO_IMAGE

The script will:
- Deskew and binarize the sheet using the existing DIP pipeline.
- Extract normalized 64x64 character crops and save them to `data/processed/`.
- Map the first 26 crops to A..Z and the next 26 to a..z by default and
  save them under `data/verified/<label>/` so the compositor can sample
  real exemplars for synthesis.

If the number of extracted crops doesn't match expectations the script will
place remaining crops under `data/verified/unlabeled/` so nothing is lost.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from preprocessing.dip_engine import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VERIFIED_DIR = PROJECT_ROOT / "data" / "verified"


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)


def save_array_as_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def default_label_order():
    upper = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    lower = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    return upper + lower


def import_sheet(image_path: Path, interactive: bool = False) -> None:
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    # Convert RGB -> BGR for OpenCV-friendly processing
    try:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        # If cv2 unavailable, fallback to using the RGB array directly
        bgr = rgb

    binary, contour_boxes, normalized_chars, angle = preprocess_image(bgr)

    ensure_dirs()

    print(f"Extracted {len(normalized_chars)} normalized character crops (deskew {angle:.2f} deg).")

    # Save to processed (char_0001.png ...)
    for idx, arr in enumerate(normalized_chars, start=1):
        fname = f"char_{idx:04d}.png"
        out_path = PROCESSED_DIR / fname
        save_array_as_png(arr, out_path)

    labels = default_label_order()
    mapped_count = min(len(normalized_chars), len(labels))

    # Map first N to labels A..Z a..z
    for i in range(mapped_count):
        label = labels[i]
        label_dir = VERIFIED_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        arr = normalized_chars[i]
        timestamp = int(time.time())
        save_array_as_png(arr, label_dir / f"{timestamp}_{i+1:03d}.png")

    # Remaining crops go to unlabeled
    if len(normalized_chars) > mapped_count:
        unlabeled_dir = VERIFIED_DIR / "unlabeled"
        unlabeled_dir.mkdir(parents=True, exist_ok=True)
        for j, arr in enumerate(normalized_chars[mapped_count:], start=1):
            save_array_as_png(arr, unlabeled_dir / f"{int(time.time())}_{j:03d}.png")

    print(f"Saved {mapped_count} crops to verified labels and {max(0, len(normalized_chars)-mapped_count)} to unlabeled.")
    print("You can now run the Streamlit app and the compositor will prefer these verified exemplars.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to handwriting sheet image (JPG/PNG)")
    parser.add_argument("--interactive", action="store_true", help="Ask before saving when counts mismatch")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    import_sheet(image_path, interactive=args.interactive)


if __name__ == "__main__":
    main()
