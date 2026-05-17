"""CNN classifier architecture for Pensync character recognition.

This module provides a compact VGG-style convolutional network for classifying
normalized handwriting character images. Convolutional layers are used because
they learn translation-tolerant local features (edges, stroke junctions, loops),
which are exactly the visual primitives needed for handwritten character
recognition.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


CLASS_INDEX_TO_CHAR: tuple[str, ...] = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
MODEL_FILENAMES: tuple[str, ...] = ("emnist_cnn.keras", "emnist_cnn.h5")
MODELS_DIR = Path(__file__).resolve().parent


def build_cnn_model(
    input_shape: tuple[int, int, int] = (64, 64, 1),
    num_classes: int = 62,
) -> tf.keras.Model:
    """Build and compile a VGG-style CNN for character classification.

    The network follows a classical two-block convolution and pooling design.
    Early convolutional filters capture local stroke geometry while max pooling
    progressively compresses spatial dimensions and preserves salient structure.
    Dense layers then map these extracted visual features to class logits.

    Parameters
    ----------
    input_shape:
        Shape of one input sample as ``(height, width, channels)``.
    num_classes:
        Number of target character classes.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training.
    """

    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1 for classification.")

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="pensync_cnn_classifier",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    """Load the exported EMNIST CNN once and reuse it across Streamlit sessions.

    The trained model is kept on disk so live handwriting predictions do not pay
    the cost of rebuilding or retraining the network. Caching the loaded model is
    important in Streamlit because the script may rerun frequently while the user
    interacts with the interface.
    """

    for filename in MODEL_FILENAMES:
        model_path = MODELS_DIR / filename
        if model_path.exists():
            return tf.keras.models.load_model(model_path)

    raise FileNotFoundError(
        f"Could not find a trained CNN model in {MODELS_DIR}. Expected one of: {', '.join(MODEL_FILENAMES)}"
    )


def predict_character(model: tf.keras.Model, char_image: np.ndarray) -> str:
    """Predict the ASCII character represented by one normalized character image.

    During live synthesis sessions this function is called repeatedly, so the
    path is intentionally lightweight: shape validation, single-pass
    normalization, one forward pass, and direct index-to-character decoding.
    Convolutional classifiers are effective here because they retain spatial
    stroke structure while remaining robust to small local shifts.

    Parameters
    ----------
    model:
        A trained Keras classification model that outputs 62 class logits.
    char_image:
        One extracted character image, expected at shape ``(64, 64)`` or
        ``(64, 64, 1)``.

    Returns
    -------
    str
        The predicted character from the ordered EMNIST byclass mapping:
        ``0-9, A-Z, a-z``.
    """

    if char_image.ndim == 2:
        if char_image.shape != (64, 64):
            raise ValueError("char_image must have shape (64, 64) or (64, 64, 1).")
        network_input = char_image[np.newaxis, :, :, np.newaxis]
    elif char_image.ndim == 3 and char_image.shape == (64, 64, 1):
        network_input = char_image[np.newaxis, :, :, :]
    else:
        raise ValueError("char_image must have shape (64, 64) or (64, 64, 1).")

    network_input = network_input.astype(np.float32)
    if np.max(network_input) > 1.0:
        network_input /= 255.0

    probabilities = model.predict(network_input, verbose=0)
    predicted_index = int(np.argmax(probabilities, axis=1)[0])

    if predicted_index < 0 or predicted_index >= len(CLASS_INDEX_TO_CHAR):
        raise ValueError(f"Predicted class index {predicted_index} is outside the 62-class mapping.")

    return CLASS_INDEX_TO_CHAR[predicted_index]