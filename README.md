# Pensync: AI-Assisted Personalized Handwriting Synthesis

Pensync is a four-phase academic handwriting pipeline that transforms scanned or user-provided handwriting into a personalized synthesis system. The project combines classical digital image processing, supervised character classification, active-learning correction, and rule-based composition to generate output that preserves the visual identity of the writer rather than collapsing into a generic font.

## Project Overview

The central problem addressed by Pensync is intraclass handwriting variance: the same writer may produce multiple visual forms of a letter depending on speed, pen pressure, context, or surrounding letters. Conventional OCR systems often normalize this variation away, while font-based synthesis produces robotic results that do not resemble the original handwriting style.

Pensync is designed to retain stylistic variation while remaining interpretable and auditable. It detects and normalizes handwriting regions, classifies characters with an EMNIST-trained CNN, allows human-in-the-loop correction, and then composes new handwritten pages by sampling verified character exemplars. The result is a personalized synthesis pipeline suitable for demonstrations, controlled experiments, and final-year project defense.

## Architecture

Pensync is organized into four phases:

### Phase 1: DIP Extraction

The Digital Image Processing stage prepares scanned handwriting for downstream use. It performs OpenCV-based deskewing, Otsu binarization, and morphological closing to improve segmentation quality, including the vertical merging of i/j dots with their stems. The contour extractor isolates candidate handwriting regions while preserving the original crisp binary crops for normalization.

### Phase 2: ML Classifier

The machine learning stage uses an EMNIST-trained VGG-style CNN for character recognition and auto-labeling. The classifier supports the standard alphanumeric set and provides confidence scores that are used by the verification workflow to prioritize uncertain samples.

### Phase 3: MLOps

The Streamlit dashboard acts as the human-in-the-loop operations layer. It supports active learning, confidence-based sorting, rapid correction workflows, and safe-labeling for punctuation and other special characters. Verified samples are persisted in a filesystem-safe format so that punctuation and other non-alphanumeric characters can be stored reliably on Windows OS.

### Phase 4: Compositor

The synthesis engine is a rule-based compositor that reconstructs text into a handwritten page. It uses K-Means variation sampling when multiple exemplars exist, applies word-level layout to avoid mid-word breaks, uses proportional typography scaling to preserve character aspect ratios, and performs PIL-based HD LANCZOS alpha-blending to preserve stroke realism and spacing variation. The compositor also respects verified samples and safe labels during lookup so that punctuation and special symbols are handled consistently.

## Quick Start

1. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install the dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch the Streamlit application:

```powershell
streamlit run app/app.py
```

4. Use the interface:

- Upload a handwriting sheet in the DIP tab to extract characters.
- Review and correct labels in the Human-in-the-Loop verification UI.
- Save verified samples so the compositor can reuse the writer's own style.
- Enter digital text in the synthesis tab to generate a handwritten page.

## Repository Notes

- `preprocessing/dip_engine.py` contains the segmentation and normalization pipeline.
- `models/train_cnn.py` provides the EMNIST training and loading workflow.
- `app/app.py` contains the Streamlit dashboard and human-in-the-loop tooling.
- `generator/compositor.py` renders synthesized pages from verified character exemplars.

## Design Goals

Pensync was built with three academic goals in mind:

1. Preserve handwriting identity across generated output.
2. Reduce robotic, font-like synthesis artifacts.
3. Keep the pipeline interpretable so each stage can be evaluated independently.

## Technology Stack

- Python
- OpenCV
- TensorFlow / Keras
- Streamlit
- scikit-learn
- NumPy
- Pillow

## Submission Summary

This repository is structured as a complete BSCS final-year project implementation, including preprocessing, classification, verification, synthesis, and a user-facing dashboard. The final system is intended for academic demonstration and controlled experimentation rather than large-scale production deployment.