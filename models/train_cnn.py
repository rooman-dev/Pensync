"""Offline EMNIST training script for Pensync's CNN character classifier.

This script trains the convolutional model on EMNIST byclass (62 classes) so
live Streamlit sessions can use a pre-trained recognizer without expensive
on-session model fitting.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
import tensorflow as tf
from emnist import clear_cached_data, extract_test_samples, extract_training_samples, get_cached_data_path
from typing import Callable

try:
    from models.cnn_classifier import build_cnn_model
except ImportError:
    from cnn_classifier import build_cnn_model


TARGET_IMAGE_SIZE = 64
NUM_CLASSES = 62
BATCH_SIZE = 256
EPOCHS = 5
VALIDATION_SPLIT = 0.1
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "emnist_cnn.keras"
NIST_EMNIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
    "Accept": "application/zip,*/*;q=0.8",
}


def _zip_internal_path(dataset_name: str, usage: str, component: str) -> str:
    """Build the archive path for a single EMNIST split component.

    EMNIST stores each split inside a ZIP file, with the actual IDX payload
    compressed again as gzip. Keeping this path construction explicit makes the
    loading logic easy to audit and aligns with the dataset's published layout.
    """

    if component == "images":
        dim = 3
    elif component == "labels":
        dim = 1
    else:
        raise ValueError(f"Unsupported component: {component}")

    return f"gzip/emnist-{dataset_name}-{usage}-{component}-idx{dim}-ubyte.gz"


def _parse_idx(data: bytes) -> np.ndarray:
    """Parse IDX-formatted binary data into a NumPy array.

    The EMNIST files follow the same IDX convention as MNIST. Parsing this
    format directly avoids extra dependencies while preserving the original
    dataset structure exactly.
    """

    if data[0] != 0 or data[1] != 0:
        raise ValueError("Data is not in IDX format.")

    data_type_code = data[2]
    dtype_map = {
        0x08: np.ubyte,
        0x09: np.byte,
        0x0B: np.int16,
        0x0C: np.int32,
        0x0D: np.float32,
        0x0E: np.float64,
    }
    data_type = dtype_map.get(data_type_code)
    if data_type is None:
        raise ValueError(f"Unsupported IDX data type code: {hex(data_type_code)}")

    dims = data[3]
    shape = []
    for dim_index in range(dims):
        offset = 4 * (dim_index + 1)
        dim_size = int(np.frombuffer(data[offset : offset + 4], dtype=">u4")[0])
        shape.append(dim_size)

    offset = 4 * (dims + 1)
    array = np.frombuffer(data[offset:], dtype=np.dtype(data_type).newbyteorder(">"))
    return array.reshape(tuple(shape))


def _extract_emnist_component(cache_path: Path, dataset_name: str, usage: str, component: str) -> np.ndarray:
    """Extract a single EMNIST component directly from the cached ZIP archive."""

    internal_path = _zip_internal_path(dataset_name, usage, component)
    with zipfile.ZipFile(cache_path) as zf:
        compressed_data = zf.read(internal_path)

    parsed = _parse_idx(gzip.decompress(compressed_data))
    if component == "images":
        return parsed.swapaxes(1, 2)
    return parsed

def _ensure_valid_emnist_cache() -> None:
    """Verify the EMNIST cache archive before attempting extraction.

    The EMNIST package only checks that the cache file exists and has nonzero
    size. If a download was interrupted or partially corrupted, ``zipfile`` will
    fail later with ``BadZipFile``. This guard detects that condition early and
    clears the cache so the package can redownload a clean archive.
    """

    cache_path = Path(get_cached_data_path())
    if cache_path.exists() and zipfile.is_zipfile(cache_path):
        return

    if cache_path.exists():
        print(f"      Detected corrupted EMNIST cache at {cache_path}; clearing it.")
        clear_cached_data()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_cache_path = cache_path.with_name(f"{cache_path.name}.partial")

    print(f"      Downloading EMNIST archive from NIST to {cache_path}...")
    with requests.get(NIST_EMNIST_ZIP_URL, headers=EMNIST_DOWNLOAD_HEADERS, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(temp_cache_path, "wb") as cache_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    cache_file.write(chunk)

    if not zipfile.is_zipfile(temp_cache_path):
        if temp_cache_path.exists():
            temp_cache_path.unlink()
        raise RuntimeError(
            "Downloaded EMNIST archive is not a valid ZIP file. Check network access to the NIST dataset endpoint."
        )

    if cache_path.exists():
        cache_path.unlink()
    temp_cache_path.replace(cache_path)
    print(f"      EMNIST archive cached at {cache_path}.")


def _load_split(dataset_name: str, usage: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one EMNIST split from the validated cache archive."""

    _ensure_valid_emnist_cache()
    cache_path = Path(get_cached_data_path())
    images = _extract_emnist_component(cache_path, dataset_name, usage, "images")
    labels = _extract_emnist_component(cache_path, dataset_name, usage, "labels")

    if len(images) != len(labels):
        raise RuntimeError("Extracted image and label arrays do not match in size.")

    return images, labels


def load_emnist_byclass() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the EMNIST byclass split for 62-way alphanumeric classification.

    The byclass split is selected because Pensync targets the full character set
    ``0-9, A-Z, a-z``. Using this dataset aligns the offline classifier's label
    space with the expected live handwriting extraction output.
    """

    print("[1/5] Loading EMNIST byclass training samples...")
    train_images, train_labels = _load_split("byclass", "train")
    print(f"      Training set loaded: images={train_images.shape}, labels={train_labels.shape}")

    print("[2/5] Loading EMNIST byclass test samples...")
    test_images, test_labels = _load_split("byclass", "test")
    print(f"      Test set loaded: images={test_images.shape}, labels={test_labels.shape}")
    return train_images, train_labels, test_images, test_labels


def _build_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    split_name: str,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset with fast batched resize and normalization.

    EMNIST images are $28 \times 28$, while Pensync consumes $64 \times 64$
    normalized glyphs from the DIP engine. This pipeline applies
    ``tf.image.resize`` in mini-batches so scaling remains efficient and avoids
    creating a very large fully materialized resized array in memory.
    """

    print(
        f"      Building {split_name} pipeline: resizing {len(images)} samples from 28x28 to {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}."
    )
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(min(len(images), 100_000), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)

    def _resize_and_normalize(batch_images: tf.Tensor, batch_labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        batch_images = tf.expand_dims(batch_images, axis=-1)
        batch_images = tf.image.resize(batch_images, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
        batch_images = tf.cast(batch_images, tf.float32) / 255.0
        return batch_images, batch_labels

    dataset = dataset.map(_resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _split_training_data(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training arrays into train/validation subsets.

    A fixed holdout split enables reliable monitoring of generalization during
    offline training and helps detect overfitting before model export.
    """

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    split_index = int(len(train_images) * (1.0 - validation_split))
    if split_index <= 0 or split_index >= len(train_images):
        raise ValueError("validation_split produced an invalid split index.")

    x_train = train_images[:split_index]
    y_train = train_labels[:split_index]
    x_val = train_images[split_index:]
    y_val = train_labels[split_index:]
    return x_train, y_train, x_val, y_val


def main() -> None:
    """Load EMNIST, train the CNN for 5 epochs, and save the trained model."""

    train_images, train_labels, test_images, test_labels = load_emnist_byclass()

    # Quick cv2 sanity check to confirm upscaling target shape for the pipeline.
    sanity_resized = cv2.resize(train_images[0], (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))
    print(f"[3/5] Resize sanity check via cv2: sample shape={sanity_resized.shape}")

    x_train, y_train, x_val, y_val = _split_training_data(
        train_images,
        train_labels,
        validation_split=VALIDATION_SPLIT,
    )
    print(
        f"      Split sizes: train={x_train.shape[0]}, val={x_val.shape[0]}, test={test_images.shape[0]}"
    )

    train_dataset = _build_dataset(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, split_name="train")
    val_dataset = _build_dataset(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False, split_name="validation")
    test_dataset = _build_dataset(test_images, test_labels, batch_size=BATCH_SIZE, shuffle=False, split_name="test")

    print("[4/5] Building and compiling CNN model...")
    model = build_cnn_model(input_shape=(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 1), num_classes=NUM_CLASSES)
    model.summary()

    print(f"[5/5] Training model for {EPOCHS} epochs...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
    )

    print("      Evaluating on EMNIST test split...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"      Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)
    print(f"Training complete. Model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()