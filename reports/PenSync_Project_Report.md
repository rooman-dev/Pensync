# PenSync — Project Report

Date: 2026-05-22

## Executive Summary

PenSync is a pipeline for high-fidelity synthetic handwriting generation and human-in-the-loop (HITL) verification. The project includes: preprocessing (DIP), analytics, an EMNIST-based classifier, a compositor that synthesizes pages using mask-first compositing, and a PyWebView desktop UI for rapid verification.

This report documents recent bug fixes (mask-based morphology), UI migration to PyWebView, verification queue APIs, and recommended next steps.

## Key Achievements

- Fixed compositor mask vs. image mismatch by applying morphological operations to the alpha mask only (removed blue filled-box artifacts).
- Implemented corner-median background estimation to avoid mask inversion on dense glyphs.
- Preserved LANCZOS resampling and used border-median padding to avoid color contamination.
- Tightened kerning via tight-ink width measurement and caching.
- Migrated UI from Streamlit to PyWebView (`ui/index.html`, `ui/styles.css`, `main_gui.py`).
- Implemented HITL Rapid-Fire verification: `get_verification_queue`, `resolve_verification` and keyboard-driven approve/reject (Y/N).
- Ensured case-safe filesystem routing for Windows by using `upper_`/`lower_`/`sym_<codepoint>` safe labels.

## Architecture Overview

- Image processing: OpenCV + Pillow (LANCZOS) + NumPy; mask-first compositing pipeline in `generator/compositor.py`.
- Classifier: EMNIST CNN (used for live predictions; current UI uses mock predictions until integrated).
- UI: PyWebView desktop shell with JS↔Python bridge (`main_gui.py` binds `PensyncAPI` to `window.pywebview.api`).
- Data directories: `data/pending`, `data/verified`, `data/rejected`.

## Notable Files Modified / Added (recent)

- `generator/compositor.py` — mask extraction, `_adjust_mask_width`, mask-only morphology, tight-ink measurement and caching.
- `preprocessing/dip_engine.py` — improved contour extraction (vertical closing) and normalization helpers.
- `app/app.py` — safe label handling for loader/persistence.
- `ui/index.html`, `ui/styles.css` — new desktop UI shell with dashboard and verification views.
- `main_gui.py` — PyWebView launcher and `PensyncAPI` (IPC methods: `get_dashboard_stats`, `get_verification_queue`, `resolve_verification`).

## How to Generate the PDF Report (this file)

A small helper script `tools/generate_pdf_report.py` is included to convert this markdown file into `reports/PenSync_Project_Report.pdf` using ReportLab. If ReportLab isn't installed, install it first:

```powershell
.venv\Scripts\Activate.ps1
pip install -r reports/requirements-report.txt
```

Then run:

```powershell
python tools/generate_pdf_report.py reports/PenSync_Project_Report.md reports/PenSync_Project_Report.pdf
```

## Next Steps / Roadmap

- Integrate the real EMNIST classifier into `get_verification_queue` (replace mock predictions).
- Export the model to ONNX and evaluate inference speed; provide an ONNX-runner fallback for the UI.
- Remove debug PNG artifacts from VCS and add CI checks to prevent large binary commits.
- Add a FAISS-based exemplar search for stylistic matching and GPU-accelerated compositing if throughput becomes limiting.

## Appendix

- Most recent pushed commit: `91b284f` (Compositor: mask-based morphology and UI migration changes).
- Relevant paths: `generator/compositor.py`, `preprocessing/dip_engine.py`, `main_gui.py`, `ui/index.html`, `ui/styles.css`.

