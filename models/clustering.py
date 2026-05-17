"""Style clustering utilities for Pensync character variants.

K-Means clustering is used to group normalized characters that share similar
structural appearance. This captures intra-class style variation (for example,
different ways of writing the same letter) and supports synthesis diversity so
generated handwriting avoids repetitive, robotic glyph reuse.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import KMeans


def cluster_character_variants(
    char_images: List[np.ndarray],
    n_clusters: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster normalized character images into style-variant groups.

    Each $64 \times 64$ image is flattened into a 1D feature vector and fed to
    K-Means. The algorithm partitions samples by minimizing within-cluster
    squared Euclidean distance, yielding compact groups of similarly shaped
    characters and their representative centroids. This helps solve the
    "robotic font" problem: instead of replaying one exact glyph repeatedly,
    Pensync can sample from several human-like variants of the same letter and
    keep generated handwriting visually natural.

    Parameters
    ----------
    char_images:
        List of normalized single-character images, expected at shape ``(64, 64)``.
    n_clusters:
        Number of style clusters to learn.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(labels, centers)`` where labels has shape ``(n_samples,)`` and
        centers has shape ``(n_clusters, 64, 64)``.
    """

    if not char_images:
        raise ValueError("char_images must contain at least one image.")

    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    flattened_images: List[np.ndarray] = []
    for image in char_images:
        if image.ndim != 2:
            raise ValueError("Each character image must be a 2D array.")
        if image.shape != (64, 64):
            raise ValueError("Each character image must have shape (64, 64).")
        flattened_images.append(image.astype(np.float32).reshape(-1) / 255.0)

    feature_matrix = np.vstack(flattened_images)
    effective_clusters = min(n_clusters, feature_matrix.shape[0])
    if effective_clusters <= 0:
        raise ValueError("n_clusters must be at least 1.")

    kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)
    center_images = (kmeans.cluster_centers_.reshape(effective_clusters, 64, 64) * 255.0).astype(np.uint8)
    return labels, center_images