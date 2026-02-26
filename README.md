
# Weather Station Latitude Prediction


---

## Overview

Weather stations in the post-2020 dataset had their latitude values corrupted to `-9999` due to a data pipeline error. This project trains a machine learning model on pre-2021 labeled station data (**PS1**) to predict the latitude of post-2020 stations (**PS2**), and flags any PS2 stations that appear to be geographically outside the original PS1 collection region.

---

## Project Structure

```
.
├── data/
│   ├── PS1/                        # Labeled training stations (pre-2021, known latitude)
│   └── PS2/                        # Unlabeled prediction stations (post-2020, latitude = -9999)
├── sfl_latitude_prediction.ipynb   # Full analysis notebook (EDA → model → results)
├── run_pipeline.py                 # Headless Python script (same logic as notebook)
├── Dockerfile                      # Python 3.10 environment
├── run.sh                          # Build image + run container + save results
├── requirements.txt                # Pinned Python dependencies
└── README.md                       # This file
```

---

## Quickstart (Docker)

### Mac / Linux

Requires **bash** and **Docker Desktop** (or Docker Engine on Linux).

```bash
# 1. Place PS1 station files in ./data/PS1/
# 2. Place PS2 station files in ./data/PS2/
# 3. Run:
bash run.sh
```

On completion, `prediction_results.csv` will be written to the current directory.

**Optional — custom paths:**
```bash
bash run.sh --ps1 /path/to/PS1 --ps2 /path/to/PS2 --output my_results.csv
```

---

### Windows

Requires **Docker Desktop for Windows** (download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)). During installation, enable **"Use WSL 2 based engine"** when prompted. Open Docker Desktop and wait for **"Engine running"** in the bottom left before proceeding.

Windows does not support `run.sh` natively, so run the pipeline directly with `docker run`.

**Step 1 — Create data folders** (PowerShell):
```powershell
mkdir data\PS1
mkdir data\PS2
```
Copy your CSV station files into `data\PS1\` and `data\PS2\` respectively.

**Step 2 — Build the Docker image** (only needed once):
```powershell
docker build -t sfl_latitude_prediction .
```

**Step 3 — Run the pipeline:**

*PowerShell:*
```powershell
docker run --rm `
  -v "${PWD}\data\PS1:/app/data/PS1" `
  -v "${PWD}\data\PS2:/app/data/PS2" `
  -v "${PWD}:/app/output" `
  sfl_latitude_prediction `
  python run_pipeline.py `
    --ps1 /app/data/PS1 `
    --ps2 /app/data/PS2 `
    --output /app/output/prediction_results.csv
```

*Command Prompt (cmd.exe):*
```cmd
docker run --rm ^
  -v "%CD%\data\PS1:/app/data/PS1" ^
  -v "%CD%\data\PS2:/app/data/PS2" ^
  -v "%CD%:/app/output" ^
  sfl_latitude_prediction ^
  python run_pipeline.py ^
    --ps1 /app/data/PS1 ^
    --ps2 /app/data/PS2 ^
    --output /app/output/prediction_results.csv
```

> **Note:** Inside the container, paths always use forward slashes `/` even on Windows. The backtick `` ` `` is PowerShell's line continuation character; `^` is the equivalent in Command Prompt.

On completion, `prediction_results.csv` will appear in your project folder. Preview it with:
```powershell
Get-Content prediction_results.csv
```

**Optional — custom data paths (PowerShell):**
```powershell
docker run --rm `
  -v "C:\Users\YourName\Desktop\PS1_data:/app/data/PS1" `
  -v "C:\Users\YourName\Desktop\PS2_data:/app/data/PS2" `
  -v "${PWD}:/app/output" `
  sfl_latitude_prediction `
  python run_pipeline.py --ps1 /app/data/PS1 --ps2 /app/data/PS2 --output /app/output/prediction_results.csv
```

---

### Windows Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Docker Desktop isn't running — open it from the Start menu and wait for the engine to start |
| `Drive has not been shared` | Docker Desktop → Settings → Resources → File Sharing → add your drive (e.g. `C:\`) |
| `invalid reference format` | Check for extra spaces in the `-v` volume path arguments |
| `No CSV files found in PS1` | Confirm CSV files are directly inside `data\PS1\`, not in a subfolder |
| Container exits with no output | Remove `--rm`, re-run, then check logs: `docker logs <container_id>` |
| WSL 2 not installed | Open PowerShell as Administrator and run `wsl --install`, then restart your PC |
| Slow first run | Normal — Docker is downloading the base Python image (~150 MB). Subsequent runs are fast. |

---

## Quickstart (Local Python)

**Mac / Linux:**
```bash
pip install -r requirements.txt

python run_pipeline.py \
    --ps1 ./data/PS1 \
    --ps2 ./data/PS2 \
    --output prediction_results.csv
```

**Windows (PowerShell):**
```powershell
pip install -r requirements.txt

python run_pipeline.py `
    --ps1 .\data\PS1 `
    --ps2 .\data\PS2 `
    --output prediction_results.csv
```

To explore the full analysis interactively:

```bash
jupyter notebook sfl_latitude_prediction.ipynb
```

---

## Approach

### 1. Feature Engineering
Each station's hourly time-series is aggregated into a flat feature vector (~134 features) per station:

| Feature Group | Examples | Latitude Signal |
|---|---|---|
| Global statistics | mean, std, p10, p90 for all 17 variables | Baseline climate |
| **Seasonal amplitude** | `temp_seasonal_range`, `temp_seasonal_std` | **Strongest signal** — equatorial stations have near-zero seasonal swing |
| Monthly means | `temp_m01` … `temp_m12` for temp, humidity, radiation, pressure, dew point | Insolation pattern driven by sun angle |
| Diurnal temperature range | Mean daily max − min | Climate regime proxy |
| Metadata | Station height, data completeness fraction | Secondary context |

### 2. Model — Random Forest Regressor
A `scikit-learn` `Pipeline` composed of:
- **Median imputation** — handles the high fraction of missing measurements (originally `-9999` sentinels)
- **Random Forest Regressor** (200 trees, `max_features='sqrt'`, `min_samples_leaf=2`)

Random Forest was chosen over deep learning deliberately: it is robust to missing data, requires no distributional assumptions, provides interpretable feature importances, and performs well on tabular data with moderate sample sizes. Gradient Boosting (LightGBM/XGBoost) is the natural next step.

### 3. Validation
- **Leave-One-Out CV** when n_stations ≤ 30; **5-Fold CV** otherwise.
- Metrics reported: Mean Absolute Error (degrees) and R².
- Out-of-fold residual plots included in the notebook.

### 4. Out-of-Distribution (OOD) Detection
PS2 stations that are geographically novel (outside the PS1 region) are flagged using **Mahalanobis distance** on the top-20 most important features.

- A **χ² threshold at 99% confidence** provides the decision boundary.
- The covariance matrix is regularized (`+ 1e-6 · I`) to avoid singularity with small datasets.
- Stations whose Mahalanobis distance exceeds the threshold are marked `is_ood = True` in the output.

---

## Output Format

`prediction_results.csv` contains one row per PS2 station:

| Column | Description |
|---|---|
| `station_code` | Station identifier |
| `source_file` | Source CSV filename |
| `predicted_latitude` | Model-predicted latitude (degrees) |
| `mahalanobis_dist` | Distance from PS1 training distribution |
| `is_ood` | `True` if station is likely outside PS1 geographic region |

---

## Limitations & Next Steps

**Current limitations:**
- Hemisphere ambiguity: stations at ±20° may have mirror-image seasonal profiles. Adding a feature for which calendar month has peak temperature resolves this.
- Mahalanobis distance assumes a Gaussian training distribution — can be noisy with small n.
- Short time-series stations will have sparsely populated monthly bins, adding noise.

**Given more time:**
1. **More training data** — ingest all available PS1 files to reduce overfitting.
2. **Gradient Boosting** (LightGBM / XGBoost) — typically outperforms Random Forest on tabular data and handles missing values natively.
3. **Isolation Forest** for OOD detection — non-parametric, no Gaussian assumption required.
4. **Conformal prediction** — calibrated uncertainty intervals on latitude predictions.
5. **Hemisphere encoding** — feature indicating Northern vs. Southern Hemisphere seasonal peak to break sign ambiguity.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.2.2 | Data loading and wrangling |
| numpy | 1.26.4 | Numerical computation |
| scikit-learn | 1.5.0 | ML pipeline, Random Forest, imputation |
| scipy | 1.13.0 | Mahalanobis distance, chi-squared threshold |
| matplotlib | 3.9.0 | Visualization |
| seaborn | 0.13.2 | Statistical plots |
| jupyter | 1.0.0 | Notebook interface |

---

## References

- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
- Mahalanobis, P.C. (1936). *On the generalised distance in statistics*. Proceedings of the National Institute of Sciences of India.
- Liu, F.T., Ting, K.M., Zhou, Z-H. (2008). *Isolation Forest*. IEEE ICDM 2008.
- INMET — Instituto Nacional de Meteorologia (Brazilian weather station data source).
- Claude and ChatGPT for Coding and Documentation
