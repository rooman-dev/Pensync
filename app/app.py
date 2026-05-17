"""Streamlit dashboard for Pensync Phase 1 data extraction.

This interface provides a lightweight visual layer over the classical DIP
pipeline. Users can upload a scanned handwriting image, inspect the processed
page with contour boxes, and review the first normalized character crops that
will be used for downstream style profiling and classification.
"""

from __future__ import annotations

import io
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import shutil
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.eda import plot_aspect_ratios, plot_character_areas
from generator.compositor import generate_handwritten_page
from preprocessing.dip_engine import ContourBox, normalize_character, preprocess_image
from models.cnn_classifier import load_model as load_cnn_model, predict_character

try:
    from models.clustering import cluster_character_variants
except Exception:
    cluster_character_variants = None


PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_VERIFIED_DIR = PROJECT_ROOT / "data" / "verified"


def _read_image_from_path(image_path: Path) -> np.ndarray:
    """Read an image from disk using OpenCV in a format suitable for analysis.

    The Streamlit uploader only provides an in-memory file-like object, so we
    first persist it to a temporary file and then reload it through OpenCV. This
    makes the processing path explicit and also mirrors how the pipeline will
    consume real scanned files in a production deployment.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image from {image_path}.")
    return image


def _annotate_contours(binary_image: np.ndarray, contour_boxes: List[ContourBox]) -> np.ndarray:
    """Overlay contour bounding boxes on a display image.

    The boxes are drawn over the binary page so that the extracted regions can be
    visually verified against the segmentation output. This is useful in an
    academic setting because it makes the contour extraction step auditable.
    """

    if binary_image.ndim == 2:
        annotated_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        annotated_image = binary_image.copy()

    for box in contour_boxes:
        cv2.rectangle(
            annotated_image,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            (0, 0, 255),
            1,
        )

    return annotated_image


def _load_uploaded_image(uploaded_bytes: bytes, filename: str) -> np.ndarray:
    """Persist an uploaded file to a temporary location and reload it safely.

    A named temporary file is used so OpenCV can access a real filesystem path.
    The file is removed in a ``finally`` block to ensure cleanup even when the
    pipeline raises an exception.
    """

    suffix = Path(filename).suffix.lower() or ".png"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_bytes)
            temp_path = Path(temp_file.name)

        return _read_image_from_path(temp_path)
    finally:
        if temp_path is not None and temp_path.exists():
            os.remove(temp_path)


def _render_character_grid(characters: List[np.ndarray], limit: int = 30) -> None:
    """Render a compact grid of normalized character crops.

    Showing the first few normalized crops helps validate that the segmentation
    and aspect-ratio preserving resize stage are behaving correctly before any
    training or statistical modeling is attempted.
    """

    if not characters:
        st.info("No contours were detected for normalization.")
        return

    st.subheader(f"First {min(limit, len(characters))} Normalized Characters")
    columns_per_row = 6
    selected_characters = characters[:limit]

    for row_start in range(0, len(selected_characters), columns_per_row):
        row_characters = selected_characters[row_start : row_start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column_index, character in enumerate(row_characters):
            with columns[column_index]:
                st.image(character, clamp=True, channels="GRAY", use_container_width=True)
                st.caption(f"Char {row_start + column_index + 1}")


def _next_dataset_index(dataset_dir: Path) -> int:
    """Return the next sequential filename index for dataset character images.

    Sequential indexing preserves append-only dataset growth across multiple user
    sessions and prevents accidental overwrites when repeated exports are done.
    """

    existing_indices: List[int] = []
    for image_path in dataset_dir.glob("char_*.png"):
        stem = image_path.stem
        suffix = stem.replace("char_", "", 1)
        if suffix.isdigit():
            existing_indices.append(int(suffix))

    return (max(existing_indices) + 1) if existing_indices else 1


def _save_characters_to_dataset(characters: List[np.ndarray], dataset_dir: Path) -> List[Path]:
    """Persist normalized character crops to the processed dataset directory.

    Each normalized sample is saved as a PNG file with a monotonic sequential
    name. PNG is used to avoid lossy compression artifacts that could distort
    stroke shape statistics used later in EDA and clustering.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)
    start_index = _next_dataset_index(dataset_dir)
    saved_paths: List[Path] = []

    for offset, character in enumerate(characters):
        file_index = start_index + offset
        output_path = dataset_dir / f"char_{file_index:04d}.png"
        write_ok = cv2.imwrite(str(output_path), character)
        if write_ok:
            saved_paths.append(output_path)

    return saved_paths


def _build_characters_zip(characters: List[np.ndarray]) -> bytes:
    """Create an in-memory ZIP archive containing extracted character images.

    In-memory zipping avoids temporary archive files on disk and provides a clean
    transport format for quick dataset download from the Streamlit interface.
    """

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, character in enumerate(characters, start=1):
            encoded_ok, encoded_image = cv2.imencode(".png", character)
            if not encoded_ok:
                continue
            archive.writestr(f"char_{index:04d}.png", encoded_image.tobytes())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def _to_bounding_box_tuples(contour_boxes: List[ContourBox]) -> List[Tuple[int, int, int, int]]:
    """Convert contour dataclass objects into tuple format for analytics helpers."""

    return [(box.x, box.y, box.width, box.height) for box in contour_boxes]


def _load_processed_character_paths(dataset_dir: Path) -> List[Path]:
    """Return available saved character images from the processed dataset."""

    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("char_*.png"))


def _load_image_from_path(image_path: Path) -> np.ndarray:
    """Load a grayscale character image from disk for live prediction."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}.")
    return image


def _sanitize_single_character_label(label: str, fallback: str) -> str:
    """Normalize a correction label to a single EMNIST character."""

    normalized_label = (label or "").strip()
    if not normalized_label:
        return fallback
    return normalized_label[0]


def _prepare_prediction_character(image: np.ndarray) -> np.ndarray:
    """Normalize a candidate character image into the 64x64 CNN input format.

    The same aspect-preserving padding logic used by the DIP pipeline is applied
    here so live predictions see the same geometric representation regardless of
    whether the image comes from the extracted dataset or a manually uploaded
    crop.
    """

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_character(image, target_size=64)


def _display_prediction_block(character_image: np.ndarray, prediction_text: str) -> None:
    """Render the live prediction result in a compact, user-friendly format."""

    preview_column, result_column = st.columns([1, 2])
    with preview_column:
        st.image(character_image, caption="Prediction Input", clamp=True, channels="GRAY", use_container_width=True)
    with result_column:
        st.metric("Predicted Character", prediction_text)
        st.caption("This prediction comes from the trained EMNIST byclass CNN classifier.")


def _build_character_database_from_verified_labels(verified_labels: dict[str, str]) -> dict[str, List[np.ndarray]]:
    """Rebuild the character database from user-verified labels in session state."""

    verified_character_db: dict[str, List[np.ndarray]] = {}

    for path_text, label in verified_labels.items():
        image_path = Path(path_text)
        if not image_path.exists() or not label:
            continue

        try:
            character_image = _prepare_prediction_character(_load_image_from_path(image_path))
        except Exception:
            continue

        verified_character_db.setdefault(label, []).append(character_image)

    return verified_character_db


def _build_character_database_from_processed_images(text: str) -> dict[str, List[np.ndarray]]:
    """Build a character database by predicting labels for saved character crops.

    This keeps the synthesis engine grounded in real extracted handwriting while
    still allowing the compositor to fall back gracefully if the trained model is
    not available yet. Predicted samples are grouped by character so the layout
    engine can sample stylistic variants during generation.
    """

    # If an in-session verified DB exists, prefer it (fast)
    verified_character_db = st.session_state.get("pensync_char_db")
    if isinstance(verified_character_db, dict) and verified_character_db:
        return verified_character_db

    character_db: dict[str, List[np.ndarray]] = {}
    allowed_characters = set(text) | set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    # 1) Load persisted verified samples from data/verified/<label>/*.png
    if PROCESSED_VERIFIED_DIR.exists():
        for label_dir in sorted(PROCESSED_VERIFIED_DIR.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in allowed_characters:
                continue
            for img_path in sorted(label_dir.glob("*.png")):
                try:
                    img = _prepare_prediction_character(_load_image_from_path(img_path))
                    character_db.setdefault(label, []).append(img)
                except Exception:
                    continue

    # 2) For labels not covered by verified samples, use model predictions on processed dataset
    try:
        model = load_cnn_model()
    except Exception:
        return character_db

    available_paths = _load_processed_character_paths(PROCESSED_DATASET_DIR)
    if not available_paths:
        return character_db

    for image_path in available_paths:
        try:
            character_image = _prepare_prediction_character(_load_image_from_path(image_path))
            predicted_character = predict_character(model, character_image)
            # Only add to db if that label isn't already satisfied by verified samples
            if predicted_character in allowed_characters and predicted_character not in character_db:
                character_db.setdefault(predicted_character, []).append(character_image)
        except Exception:
            continue

    return character_db


def _render_hitl_verification_grid() -> None:
    """Display saved characters with CNN predictions and user correction inputs."""

    available_paths = _load_processed_character_paths(PROCESSED_DATASET_DIR)
    if not available_paths:
        st.info("No saved characters were found in data/processed yet. Save extracted characters first.")
        return

    try:
        model = load_cnn_model()
    except Exception as exc:
        st.warning(f"CNN model could not be loaded for verification: {exc}")
        return

    verification_entries: List[dict[str, object]] = []
    for image_path in available_paths:
        try:
            original_image = _load_image_from_path(image_path)
            prediction_image = _prepare_prediction_character(original_image)
            predicted_label = predict_character(model, prediction_image)
            verification_entries.append(
                {
                    "path": image_path,
                    "image": prediction_image,
                    "predicted_label": predicted_label,
                    "widget_key": f"hitl_label_{image_path.stem}",
                }
            )
        except Exception:
            continue

    if not verification_entries:
        st.info("No readable character crops were available for verification.")
        return

    columns_per_row = 4
    for row_start in range(0, len(verification_entries), columns_per_row):
        row_entries = verification_entries[row_start : row_start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column_index, entry in enumerate(row_entries):
            with columns[column_index]:
                st.image(entry["image"], clamp=True, channels="GRAY", use_container_width=True)
                st.caption(f"CNN prediction: {entry['predicted_label']}")
                st.text_input(
                    "Correct label",
                    value=str(entry["predicted_label"]),
                    max_chars=1,
                    key=str(entry["widget_key"]),
                    label_visibility="collapsed",
                    help="Overtype the CNN label if it is wrong.",
                )
                st.caption(entry["path"].name)

    if st.button("Save Corrections", type="primary"):
        verified_labels: dict[str, str] = {}
        for entry in verification_entries:
            widget_key = str(entry["widget_key"])
            predicted_label = str(entry["predicted_label"])
            corrected_label = _sanitize_single_character_label(st.session_state.get(widget_key, predicted_label), predicted_label)
            verified_labels[str(entry["path"])] = corrected_label

        verified_character_db = _build_character_database_from_verified_labels(verified_labels)
        st.session_state["pensync_verified_labels"] = verified_labels
        st.session_state["pensync_char_db"] = verified_character_db
        st.success(f"Saved {len(verified_labels)} corrected labels into the compositor database.")

    if st.button("Save Verified Dataset"):
        # Move files into data/verified/<label>/ and update session DB
        saved_map: dict[str, str] = {}
        PROCESSED_VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
        for entry in verification_entries:
            widget_key = str(entry["widget_key"])
            predicted_label = str(entry["predicted_label"])
            corrected_label = _sanitize_single_character_label(st.session_state.get(widget_key, predicted_label), predicted_label)
            src_path: Path = entry["path"]
            dest_dir = PROCESSED_VERIFIED_DIR / corrected_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Ensure unique filename to avoid collisions
            dest_path = dest_dir / src_path.name
            if dest_path.exists():
                dest_path = dest_dir / f"{src_path.stem}_{uuid.uuid4().hex}{src_path.suffix}"
            try:
                shutil.move(str(src_path), str(dest_path))
                saved_map[str(dest_path)] = corrected_label
            except Exception as exc:
                st.warning(f"Failed to move {src_path.name}: {exc}")

        verified_character_db = _build_character_database_from_verified_labels(saved_map)
        st.session_state["pensync_verified_labels"] = saved_map
        st.session_state["pensync_char_db"] = verified_character_db
        st.success(f"Persisted {len(saved_map)} verified samples to {PROCESSED_VERIFIED_DIR.as_posix()}")


def _page_to_png_bytes(page: np.ndarray) -> bytes:
    """Encode a synthesized page as PNG bytes for download."""

    image = Image.fromarray(page)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def main() -> None:
    """Run the Pensync Phase 1 Streamlit interface."""

    st.set_page_config(page_title="Pensync: Phase 1 (Data Extraction)", layout="wide")
    st.title("Pensync: Phase 1 (Data Extraction)")
    st.write(
        "Upload a scanned handwriting image to deskew, binarize, segment, and normalize the character regions."
    )

    dip_tab, analytics_tab, ml_tab, synthesis_tab = st.tabs(
        ["DIP Extraction", "Data Science Analytics", "Machine Learning Core", "Phase 4: Synthesis Engine"]
    )

    uploaded_file = st.file_uploader("Upload a JPG or PNG handwriting image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        with dip_tab:
            st.info("Waiting for an upload.")
        with analytics_tab:
            st.info("Upload an image to generate handwriting analytics plots.")
            st.subheader("Style Variation Clustering")
            st.info(
                "K-Means groups structurally similar character variants so Pensync can reuse natural handwriting diversity instead of repeating one robotic glyph."
            )
        with ml_tab:
            st.subheader("CNN Architecture")
            try:
                model = load_cnn_model()
                summary_buffer = io.StringIO()
                model.summary(print_fn=lambda line: summary_buffer.write(f"{line}\n"))
                st.code(summary_buffer.getvalue(), language="text")
            except Exception as exc:
                st.warning(f"CNN model could not be loaded yet: {exc}")

            st.subheader("Live Prediction")
            st.info("Upload a 64x64 character image or use a random saved character from data/processed.")
            st.subheader("Style Clustering Demo")
            st.info(
                "K-Means groups structurally similar variants of the same character so synthesis can rotate among style-consistent exemplars instead of repeating one glyph."
            )

        with synthesis_tab:
            st.subheader("Layout & Synthesis Engine")
            synthesis_text = st.text_area(
                "Type the digital assignment or notes you want to synthesize",
                value="The quick brown fox jumps over the lazy dog.\nPensync converts digital text into handwritten form.",
                height=180,
            )

            if st.button("Generate Handwritten Document", type="primary"):
                try:
                    char_db = _build_character_database_from_processed_images(synthesis_text)
                    synthesized_page = generate_handwritten_page(synthesis_text, char_db)
                    st.session_state["pensync_synthesized_page"] = synthesized_page
                    st.success("Handwritten document generated successfully.")
                except Exception as exc:
                    st.error(f"Synthesis failed: {exc}")

            if "pensync_synthesized_page" in st.session_state:
                synthesized_page = st.session_state["pensync_synthesized_page"]
                st.image(synthesized_page, caption="Synthesized handwritten page", clamp=True, use_container_width=True)
                st.download_button(
                    label="Download Synthesized Page (PNG)",
                    data=_page_to_png_bytes(synthesized_page),
                    file_name="pensync_synthesized_page.png",
                    mime="image/png",
                )

            st.info(
                "The compositor uses rule-based line wrapping, baseline jitter, and random variant selection so the generated page looks like a real handwritten draft rather than a mechanical font render."
            )
        return

    try:
        uploaded_bytes = uploaded_file.getvalue()
        uploaded_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        analysis_image = _load_uploaded_image(uploaded_bytes, uploaded_file.name)
        binary_image, contour_boxes, normalized_characters, rotation_angle = preprocess_image(analysis_image)
        display_image = _annotate_contours(binary_image, contour_boxes)
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

        with dip_tab:
            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Original Upload")
                st.image(uploaded_image, use_container_width=True)

            with right_column:
                st.subheader("Processed Image with Bounding Boxes")
                st.image(display_image_rgb, use_container_width=True)
                st.caption(f"Deskew rotation applied: {rotation_angle:.2f} degrees")
                st.caption(f"Detected contour regions: {len(contour_boxes)}")

            _render_character_grid(normalized_characters, limit=30)

            st.subheader("Dataset Export")
            if not normalized_characters:
                st.info("No normalized characters available to save or download.")
            else:
                save_column, download_column = st.columns(2)
                with save_column:
                    if st.button("Save to Dataset", type="primary"):
                        saved_paths = _save_characters_to_dataset(normalized_characters, PROCESSED_DATASET_DIR)
                        st.success(
                            f"Saved {len(saved_paths)} character images to {PROCESSED_DATASET_DIR.as_posix()}."
                        )

                with download_column:
                    archive_bytes = _build_characters_zip(normalized_characters)
                    st.download_button(
                        label="Download Extracted Characters (.zip)",
                        data=archive_bytes,
                        file_name="pensync_extracted_characters.zip",
                        mime="application/zip",
                    )

        with analytics_tab:
            st.subheader("Structural Feature Distributions")
            if not contour_boxes:
                st.info("No contour regions were extracted, so analytics plots are unavailable.")
            else:
                bounding_boxes = _to_bounding_box_tuples(contour_boxes)
                aspect_ratio_figure = plot_aspect_ratios(bounding_boxes)
                st.pyplot(aspect_ratio_figure)
                plt.close(aspect_ratio_figure)

                area_figure = plot_character_areas(bounding_boxes)
                st.pyplot(area_figure)
                plt.close(area_figure)

            st.subheader("Style Variation Clustering")
            if not normalized_characters:
                st.info("No normalized characters available yet for clustering.")
            elif cluster_character_variants is None:
                st.warning("K-Means clustering is unavailable because scikit-learn is not installed in this environment.")
            elif len(normalized_characters) < 2:
                st.info("At least two character crops are needed to demonstrate clustering.")
            else:
                sample_count = min(len(normalized_characters), 18)
                cluster_count = min(3, sample_count)
                labels, centers = cluster_character_variants(normalized_characters[:sample_count], n_clusters=cluster_count)
                st.write(f"Clustered {sample_count} samples into {cluster_count} style groups.")
                cluster_columns = st.columns(cluster_count)
                for cluster_index in range(cluster_count):
                    with cluster_columns[cluster_index]:
                        st.caption(f"Cluster {cluster_index}")
                        st.image(centers[cluster_index], use_container_width=True, clamp=True, channels="GRAY")
                        member_count = int(np.sum(labels == cluster_index))
                        st.caption(f"Members: {member_count}")

        with ml_tab:
            st.subheader("CNN Architecture")
            try:
                model = load_cnn_model()
                summary_buffer = io.StringIO()
                model.summary(print_fn=lambda line: summary_buffer.write(f"{line}\n"))
                st.code(summary_buffer.getvalue(), language="text")
            except Exception as exc:
                st.warning(f"CNN model could not be initialized in this environment: {exc}")

            st.subheader("Human-in-the-Loop: Verify & Correct Labels")
            st.caption(
                "Review the CNN's guesses for saved character crops, correct any mistakes, and save the verified labels for synthesis."
            )
            _render_hitl_verification_grid()

            st.subheader("Live Prediction")
            prediction_source = st.radio(
                "Choose a prediction source",
                ["Upload a character image", "Use a random saved character"],
                horizontal=True,
            )

            prediction_character = None
            uploaded_character_file = None
            if prediction_source == "Upload a character image":
                uploaded_character_file = st.file_uploader(
                    "Upload a 64x64 character PNG/JPG",
                    type=["png", "jpg", "jpeg"],
                    key="ml_character_uploader",
                )
                if uploaded_character_file is not None:
                    prediction_character = _prepare_prediction_character(
                        np.array(Image.open(uploaded_character_file).convert("RGB"))
                    )
            else:
                if st.button("Grab Random Character From data/processed"):
                    available_paths = _load_processed_character_paths(PROCESSED_DATASET_DIR)
                    if not available_paths:
                        st.info("No saved characters were found in data/processed.")
                    else:
                        selected_path = random.choice(available_paths)
                        prediction_character = _load_image_from_path(selected_path)
                        st.session_state["pensync_random_character_path"] = str(selected_path)

                if "pensync_random_character_path" in st.session_state:
                    selected_path = Path(st.session_state["pensync_random_character_path"])
                    if selected_path.exists():
                        prediction_character = _prepare_prediction_character(_load_image_from_path(selected_path))

            if prediction_character is not None:
                try:
                    model = load_cnn_model()
                    predicted_character = predict_character(model, prediction_character)
                    _display_prediction_block(prediction_character, predicted_character)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            else:
                st.info("Provide a character image or select a saved sample to run live inference.")

            st.subheader("Style Clustering Demo")
            st.info(
                "K-Means will cluster visually similar character variants (same letter class) into style groups. "
                "During generation, selecting from these clusters helps maintain handwriting authenticity and avoids robotic repetition."
            )

    except Exception as exc:
        st.error(f"Unable to process the uploaded image: {exc}")


if __name__ == "__main__":
    main()