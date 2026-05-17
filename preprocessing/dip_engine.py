"""Digital image processing helpers for handwriting preprocessing.

This module implements the first stage of the Pensync pipeline:

1. Convert a scanned page to grayscale.
2. Deskew the page by estimating the dominant text baseline with Hough lines.
2. Apply Otsu's adaptive global thresholding to obtain a binary ink mask.
3. Extract connected character or stroke candidates through contour analysis.
4. Normalize each extracted character region into a fixed 64x64 grid.

The implementation intentionally stays classical and interpretable. Otsu's method is
preferred over a fixed threshold because scanned handwriting often contains uneven
paper illumination, shadows, and varying ink intensity. Otsu selects the threshold
that maximizes the separation between foreground and background pixel classes,
which makes the binarization more robust across sessions and devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourBox:
    """Axis-aligned bounding box describing one extracted contour.

    The box is represented by the top-left corner and its width/height. Using a
    simple bounding box is sufficient at this stage because the goal is not to
    classify a character yet, but to isolate candidate handwriting components for
    downstream normalization and model training.
    """

    x: int
    y: int
    width: int
    height: int
    area: int


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR image to grayscale.

    Grayscale conversion collapses color information into a single luminance
    channel, which is the correct input domain for classical thresholding because
    ink-vs-background separation depends primarily on intensity rather than hue.
    """

    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {image.shape}.")

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate an image around its center while preserving the original canvas.

    The rotation is used after baseline estimation so the handwriting becomes
    horizontally aligned. Keeping the same canvas size simplifies downstream
    contour extraction because all coordinates remain in a consistent page frame.
    """

    height, width = image.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def deskew_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew a handwriting image using the median angle of Hough line segments.

    The Hough Transform is useful here because handwriting pages usually contain
    many nearly horizontal strokes whose shared orientation reveals the page tilt.
    By extracting line segments from the inverted grayscale image and taking the
    median of their angles, we obtain a robust baseline estimate that is less
    sensitive to outlier strokes than a mean-based correction. The image is then
    rotated by the opposite angle so the text baseline approaches 0 degrees.

    Parameters
    ----------
    image:
        Input handwriting image in grayscale or color.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing the deskewed grayscale image and the applied rotation
        angle in degrees.
    """

    gray_image = _to_grayscale(image)
    inverted_gray = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted_gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=100,
        minLineLength=max(25, gray_image.shape[1] // 6),
        maxLineGap=12,
    )

    detected_angles: list[float] = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, line)
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0.0:
                continue

            angle_degrees = np.degrees(np.arctan2(dy, dx))
            if -45.0 <= angle_degrees <= 45.0:
                detected_angles.append(float(angle_degrees))

    rotation_angle = float(-np.median(detected_angles)) if detected_angles else 0.0
    deskewed_image = _rotate_image(gray_image, rotation_angle)
    return deskewed_image, rotation_angle


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """Binarize an image using Otsu's threshold selection.

    Otsu's algorithm searches for the threshold that minimizes the weighted
    intra-class variance of the background and foreground pixel groups. In plain
    terms, it finds a cut point that best separates paper from ink without needing
    a manually tuned constant threshold.

    Parameters
    ----------
    image:
        Input scanned page as a grayscale or color image.

    Returns
    -------
    np.ndarray
        A binary image where foreground handwriting pixels are white (255) and
        the background is black (0) after inversion.
    """

    gray_image = _to_grayscale(image)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def normalize_character(char_roi: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Fit a character region into a fixed square canvas without distorting it.

    Character normalization is performed by preserving aspect ratio during resize
    and then padding the remaining area with black pixels. This prevents shape
    distortion, which is important for both human-readable dataset generation and
    downstream statistical modeling because width-to-height proportions are part
    of handwriting style.

    Parameters
    ----------
    char_roi:
        A cropped binary or grayscale handwriting region.
    target_size:
        Output canvas size. The result is a square array of shape
        ``(target_size, target_size)``.

    Returns
    -------
    np.ndarray
        A centered, aspect-preserving character image padded with black pixels.
    """

    if char_roi.ndim != 2:
        raise ValueError("Character normalization expects a single-channel ROI.")

    if target_size <= 0:
        raise ValueError("target_size must be a positive integer.")

    roi = char_roi.copy()
    foreground = np.argwhere(roi > 0)
    if foreground.size == 0:
        return np.zeros((target_size, target_size), dtype=roi.dtype)

    top, left = foreground.min(axis=0)
    bottom, right = foreground.max(axis=0) + 1
    roi = roi[top:bottom, left:right]

    height, width = roi.shape[:2]
    scale = min(target_size / float(height), target_size / float(width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(roi, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((target_size, target_size), dtype=roi.dtype)
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def extract_normalized_characters(
    binary_image: np.ndarray,
    contour_boxes: List[ContourBox],
    target_size: int = 64,
) -> List[np.ndarray]:
    """Crop and normalize each extracted contour region.

    The contour boxes act as spatial anchors on the binarized page. Each crop is
    normalized independently so the downstream classifier sees a consistent input
    geometry even when the original handwriting contains different character
    widths, ascenders, or descenders.
    """

    normalized_characters: List[np.ndarray] = []
    for box in contour_boxes:
        char_roi = binary_image[box.y : box.y + box.height, box.x : box.x + box.width]
        normalized_characters.append(normalize_character(char_roi, target_size=target_size))

    return normalized_characters


def extract_contours(binary_image: np.ndarray, min_area: int = 25) -> List[ContourBox]:
    """Extract handwriting candidate regions from a binary image.

    Contours are useful because handwriting strokes form connected foreground
    regions after binarization. By filtering small areas, we suppress noise caused
    by scanner artifacts, dust, and isolated threshold speckles. The result is a
    compact set of bounding boxes that can later be normalized to fixed-size
    character patches.

    Parameters
    ----------
    binary_image:
        A binary image produced by :func:`otsu_binarize`.
    min_area:
        Minimum contour area required to keep a region. This acts as a simple
        statistical noise filter.

    Returns
    -------
    list[ContourBox]
        Bounding boxes for the retained contours, sorted top-to-bottom then
        left-to-right to preserve reading order.
    """

    if binary_image.ndim != 2:
        raise ValueError("Contour extraction expects a single-channel binary image.")

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_boxes: List[ContourBox] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        contour_boxes.append(
            ContourBox(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                area=area,
            )
        )

    contour_boxes.sort(key=lambda box: (box.y, box.x))
    return contour_boxes


def preprocess_image(
    image: np.ndarray,
    min_area: int = 25,
    target_size: int = 64,
) -> tuple[np.ndarray, List[ContourBox], List[np.ndarray], float]:
    """Run the base DIP preprocessing pipeline on a handwriting scan.

    This function combines the core mathematical steps needed for early Pensync
    processing. The binary output provides a clean foreground mask, while the
    contour list provides localized handwriting regions for later segmentation,
    normalization, and feature extraction.

    Parameters
    ----------
    image:
        Input scanned handwriting image.
    min_area:
        Minimum contour area used to suppress small noise components.
    target_size:
        Size of the square canvas used to normalize each extracted character.

    Returns
    -------
    tuple[np.ndarray, list[ContourBox], list[np.ndarray], float]
        A tuple containing the Otsu-binarized image and the extracted contour
        bounding boxes, the normalized character images, and the applied deskew
        angle in degrees.
    """

    deskewed_image, rotation_angle = deskew_image(image)
    binary_image = otsu_binarize(deskewed_image)
    contour_boxes = extract_contours(binary_image, min_area=min_area)
    normalized_characters = extract_normalized_characters(
        binary_image,
        contour_boxes,
        target_size=target_size,
    )
    return binary_image, contour_boxes, normalized_characters, rotation_angle
