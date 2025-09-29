# s5p-gapfill-no2-so2

Code accompanying my thesis **“Sentinel-5P Spatio-Temporal Gap Filling for NO₂ and SO₂.”**  
⚠️ **Note:** This is *process code* from the research journey. It contains **redundant, exploratory, and experimental** scripts/notebooks kept for transparency and reproducibility.

---

## Overview
End-to-end workflow for gap-filling **Sentinel-5P (TROPOMI)** NO₂ and SO₂:
- QA-filtered daily mosaics and harmonization
- Feature engineering (spatial context, topography, land cover, meteorology, temporal lags)
- Models: **LightGBM**, **2D-CNN**, **3D-CNN**
- Evaluation and figure export used in the thesis

---

## Status & Caveats
- Many notebooks are **work-in-progress** and may overlap in functionality.
- Some paths are hard-coded from the thesis workflow; adjust to your environment.
- Large datasets and model weights are **not** tracked in Git.
