# Aedes-AI_Forecasting

Real-time, probabilistic forecasting of *Aedes aegypti* gravid female trap counts using the Aedes-AI neural network suite.

This repository contains code to (i) fine-tune a pre-trained Aedes-AI model on local weather, (ii) calibrate abundance estimates to observed trap counts near the forecast origin, and (iii) generate probabilistic 4 week-ahead forecasts under Poisson and Negative Binomial count assumptions.

> Related work: the original Aedes-AI model suite is hosted separately. See the Aedes-AI repository for background and model family details at 

---

## Repository layout

Top-level folders and files:

- `analysis/` — notebooks and/or scripts for evaluation, plots, and paper-quality figures
- `data/` — datasets and intermediate artifacts used by scripts (see notes below)
- `output/` — generated predictions, metrics, and figures
- `scripts/` — runnable scripts for preprocessing, training/fine-tuning, forecasting, and evaluation
- `utils/` — shared utilities (I/O, metrics, plotting helpers, etc.)
- `fpaths_config.json` — centralized file/path configuration for local runs (recommended first edit)

---

## Quickstart

### 1) Clone and create an environment
Create a Python environment appropriate for your setup (conda or venv). Then install dependencies.

- If the repo includes `requirements.txt` or `environment.yml`, use that file.
- Otherwise, install the packages required by the scripts you plan to run (see `scripts/` and `utils/`).

### 2) Configure paths
Update `fpaths_config.json` to point to your local data locations and desired output directories.

This repo is designed so you don’t have to hard-code paths inside scripts—keep them in the config file.

### 3) Run the pipeline
Typical workflow:

1. **Preprocess weather + trap data** (weekly aggregation, alignment, missingness handling)
2. **Fine-tune** a pre-trained Aedes-AI model on local weather
3. **Calibrate** abundance estimates to recent trap observations near each forecast origin
4. **Forecast** 1–4 weeks ahead using observed + (optionally imperfect) weather forecasts
5. **Evaluate** point accuracy and interval calibration/coverage
6. **Generate figures** for reports/manuscripts

See `scripts/` for runnable entry points and `analysis/` for figure generation and diagnostics.

---

## Citing

If you use this repository, please cite:

**Aedes-AI suite**
- Kinney, A. C., Current, S., & Lega, J. (2021). *Aedes-AI: Neural network models of mosquito abundance.* PLoS Computational Biology.

**Forecasting framework**
- Kinney, A. C., & Lega, J. (in preparation / preprint). *Real-time forecasting of mosquito trap counts with Aedes-AI neural networks.*

<details>
<summary>BibTeX (Aedes-AI)</summary>

```bibtex
@article{kinney2021aedesai,
  title={Aedes-AI: Neural network models of mosquito abundance},
  author={Kinney, Adrienne C. and Current, Sean and Lega, Joceline},
  journal={PLoS Computational Biology},
  year={2021}
}
