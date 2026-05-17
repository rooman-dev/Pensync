"""Exploratory Data Analysis utilities for handwriting structural features.

These helpers convert contour-level geometry into statistical views that are
useful for understanding writing style variability before modeling.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


BoundingBox = Tuple[int, int, int, int]


def _valid_boxes(bounding_boxes: List[BoundingBox]) -> List[BoundingBox]:
    """Filter invalid boxes to avoid division and area artifacts in EDA plots."""

    valid: List[BoundingBox] = []
    for x, y, width, height in bounding_boxes:
        if width > 0 and height > 0:
            valid.append((x, y, width, height))
    return valid


def plot_aspect_ratios(bounding_boxes: List[BoundingBox]) -> Figure:
    """Plot the distribution of character aspect ratios from bounding boxes.

    The aspect ratio $r = h / w$ captures how tall or wide each extracted
    character component is. This is a useful structural statistic for handwriting
    analysis because script style, slant behavior, and pen motion can shift the
    ratio distribution across writers or sessions.

    Parameters
    ----------
    bounding_boxes:
        List of character region boxes represented as ``(x, y, w, h)`` tuples.

    Returns
    -------
    Figure
        A Matplotlib figure containing the aspect ratio distribution.
    """

    valid_boxes = _valid_boxes(bounding_boxes)
    figure, axis = plt.subplots(figsize=(8, 4.5))

    if not valid_boxes:
        axis.text(0.5, 0.5, "No valid bounding boxes available.", ha="center", va="center")
        axis.set_axis_off()
        return figure

    ratios = np.array([height / float(width) for _, _, width, height in valid_boxes], dtype=np.float64)
    axis.hist(ratios, bins=20, color="#1f77b4", alpha=0.85, edgecolor="white")
    axis.set_title("Character Aspect Ratio Distribution")
    axis.set_xlabel("Aspect Ratio (height / width)")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_character_areas(bounding_boxes: List[BoundingBox]) -> Figure:
    """Plot the distribution of character bounding box areas.

    The bounding box area $A = w \times h$ approximates character footprint size.
    Its distribution reflects variation in writing pressure, stroke spread, and
    character scale, which are meaningful for downstream clustering and style
    profiling.

    Parameters
    ----------
    bounding_boxes:
        List of character region boxes represented as ``(x, y, w, h)`` tuples.

    Returns
    -------
    Figure
        A Matplotlib figure containing the character area distribution.
    """

    valid_boxes = _valid_boxes(bounding_boxes)
    figure, axis = plt.subplots(figsize=(8, 4.5))

    if not valid_boxes:
        axis.text(0.5, 0.5, "No valid bounding boxes available.", ha="center", va="center")
        axis.set_axis_off()
        return figure

    areas = np.array([width * height for _, _, width, height in valid_boxes], dtype=np.float64)
    axis.hist(areas, bins=20, color="#2ca02c", alpha=0.85, edgecolor="white")
    axis.set_title("Character Area Distribution")
    axis.set_xlabel("Bounding Box Area (width x height)")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure