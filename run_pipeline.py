"""
run_pipeline.py
───────────────
SFL Scientific Take-Home Challenge
Weather Station Latitude Prediction Pipeline

This script replicates the full notebook logic as a standalone executable so
Docker can run it headlessly and produce prediction_results.csv.

Usage:
    python run_pipeline.py [--ps1 PATH] [--ps2 PATH] [--output FILE]

Environment variables (override defaults):
    PS1_DIR  — path to folder containing PS1 CSV files  (default: ./data/PS1)
    PS2_DIR  — path to folder containing PS2 CSV files  (default: ./data/PS2)
"""

import os
import glob
import argparse
import warnings
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
MISSING_SENTINEL = -9999

COL_MAP = {
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)":                       "precip",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)":  "pressure",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)":        "pressure_max",
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)":       "pressure_min",
    "RADIACAO GLOBAL (Kj/m²)":                                "radiation",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)":           "temp",
    "TEMPERATURA DO PONTO DE ORVALHO (°C)":                   "dew_point",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)":             "temp_max",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)":             "temp_min",
    "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)":       "dew_max",
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)":       "dew_min",
    "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)":               "humidity_max",
    "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)":               "humidity_min",
    "UMIDADE RELATIVA DO AR, HORARIA (%)":                    "humidity",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))":                   "wind_dir",
    "VENTO, RAJADA MAXIMA (m/s)":                             "wind_gust",
    "VENTO, VELOCIDADE HORARIA (m/s)":                        "wind_speed",
}
MEASUREMENT_COLS = list(COL_MAP.values())
META_COLS        = ["station_code", "latitude", "longitude", "source_file"]


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_station_file(filepath: str) -> pd.DataFrame:
    """Load a single station CSV; rename columns, parse datetime, replace sentinels."""
    df = pd.read_csv(filepath, index_col=0)
    df.rename(columns=COL_MAP, inplace=True)
    df["datetime"] = pd.to_datetime(
        df["Data"].astype(str) + " " + df["Hora"].astype(str), errors="coerce"
    )
    df["month"] = df["datetime"].dt.month
    df["hour"]  = df["datetime"].dt.hour
    df[MEASUREMENT_COLS] = df[MEASUREMENT_COLS].replace(MISSING_SENTINEL, np.nan)
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> dict:
    """
    Aggregate a station's time-series into a flat feature dictionary.
    Key features:
      - Global statistics (mean, std, p10, p90) for all measurement columns
      - Seasonal amplitude of temperature and radiation (latitude proxy)
      - Monthly means for temp, humidity, radiation, pressure, dew_point
      - Diurnal temperature range
      - Station metadata (height, data completeness)
    """
    feats: dict = {}

    # Global statistics
    for col in MEASUREMENT_COLS:
        s = df[col].dropna()
        if len(s) == 0:
            feats[f"{col}_mean"] = np.nan
            feats[f"{col}_std"]  = np.nan
            feats[f"{col}_p10"]  = np.nan
            feats[f"{col}_p90"]  = np.nan
        else:
            feats[f"{col}_mean"] = s.mean()
            feats[f"{col}_std"]  = s.std()
            feats[f"{col}_p10"]  = s.quantile(0.10)
            feats[f"{col}_p90"]  = s.quantile(0.90)

    # Seasonal amplitude (strongest latitude signal)
    monthly_temp = df.groupby("month")["temp"].mean()
    feats["temp_seasonal_range"] = monthly_temp.max() - monthly_temp.min()
    feats["temp_seasonal_std"]   = monthly_temp.std()

    monthly_rad = df.groupby("month")["radiation"].mean()
    feats["radiation_seasonal_range"] = monthly_rad.max() - monthly_rad.min()

    # Diurnal temperature range
    if df["datetime"].notna().any():
        tmp = df[["datetime", "temp"]].dropna()
        tmp = tmp.copy()
        tmp["date"] = tmp["datetime"].dt.date
        daily_range = tmp.groupby("date")["temp"].agg(lambda x: x.max() - x.min())
        feats["temp_diurnal_range_mean"] = daily_range.mean()
    else:
        feats["temp_diurnal_range_mean"] = np.nan

    # Monthly means (flattened)
    for col in ["temp", "humidity", "radiation", "pressure", "dew_point"]:
        monthly = df.groupby("month")[col].mean()
        for m in range(1, 13):
            feats[f"{col}_m{m:02d}"] = monthly.get(m, np.nan)

    # Metadata
    feats["height"]            = df["height"].iloc[0]
    feats["data_completeness"] = df[MEASUREMENT_COLS].notna().mean().mean()
    feats["station_code"]      = df["station_code"].iloc[0]
    feats["latitude"]          = df["latitude"].iloc[0]
    feats["longitude"]         = df["longitude"].iloc[0]

    return feats


def load_all_stations(data_dir: str, is_ps2: bool = False) -> pd.DataFrame:
    """Load all CSVs from a directory and return one-row-per-station feature DataFrame."""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        log.warning("No CSV files found in %s", data_dir)
        return pd.DataFrame()

    records = []
    for fp in sorted(files):
        try:
            df_s = load_station_file(fp)
            if is_ps2:
                df_s["latitude"] = np.nan
            feats = engineer_features(df_s)
            feats["source_file"] = os.path.basename(fp)
            records.append(feats)
        except Exception as exc:
            log.error("Failed to load %s: %s", fp, exc)

    result = pd.DataFrame(records)
    log.info("Loaded %d stations from %s", len(result), data_dir)
    return result


# ── Model ──────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Return a sklearn Pipeline: median imputation → Random Forest Regressor."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Fit pipeline with cross-validation reporting; return fitted pipeline."""
    pipeline = build_pipeline()
    n = len(X)

    if n > 1:
        cv = LeaveOneOut() if n <= 30 else KFold(n_splits=5, shuffle=True,
                                                  random_state=RANDOM_STATE)
        mae_scores = -cross_val_score(pipeline, X, y, cv=cv,
                                      scoring="neg_mean_absolute_error", n_jobs=-1)
        r2_scores  =  cross_val_score(pipeline, X, y, cv=cv,
                                      scoring="r2", n_jobs=-1)
        log.info("CV MAE : %.3f ± %.3f degrees", mae_scores.mean(), mae_scores.std())
        log.info("CV R²  : %.3f ± %.3f", r2_scores.mean(), r2_scores.std())
    else:
        log.warning("Only 1 training sample — skipping cross-validation.")

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    log.info("In-sample MAE : %.3f | R² : %.3f",
             mean_absolute_error(y, y_pred), r2_score(y, y_pred))
    return pipeline


# ── Anomaly Detection ──────────────────────────────────────────────────────────

def detect_ood(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    importances: pd.Series,
    top_n: int = 20,
    confidence: float = 0.99,
) -> tuple[np.ndarray, float]:
    """
    Compute Mahalanobis distance of each PS2 station from the PS1 distribution.
    Returns (distances array, threshold scalar).
    """
    top_feats = importances.nlargest(top_n).index.tolist()

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_tr = scaler.fit_transform(imputer.fit_transform(X_train[top_feats]))
    X_te = scaler.transform(imputer.transform(X_test[top_feats]))

    cov     = np.cov(X_tr.T) + np.eye(top_n) * 1e-6
    cov_inv = np.linalg.inv(cov)
    mu      = X_tr.mean(axis=0)

    distances = np.array([mahalanobis(row, mu, cov_inv) for row in X_te])
    threshold  = float(np.sqrt(chi2.ppf(confidence, df=top_n)))
    log.info("OOD threshold (χ² %.0f%%, k=%d): %.2f", confidence * 100, top_n, threshold)
    return distances, threshold


# ── Main ───────────────────────────────────────────────────────────────────────

def main(ps1_dir: str, ps2_dir: str, output_file: str) -> None:
    np.random.seed(RANDOM_STATE)

    # ── Load data ──────────────────────────────────────────────────────────────
    df_ps1 = load_all_stations(ps1_dir, is_ps2=False)
    df_ps2 = load_all_stations(ps2_dir, is_ps2=True)

    if df_ps1.empty:
        log.error("No PS1 training data found. Exiting.")
        return

    # ── Prepare training matrix ────────────────────────────────────────────────
    feature_cols = [c for c in df_ps1.columns if c not in META_COLS]
    df_train     = df_ps1.dropna(subset=["latitude"])
    X_train      = df_train[feature_cols]
    y_train      = df_train["latitude"].values

    log.info("Training on %d stations, %d features", len(X_train), len(feature_cols))

    # ── Train ──────────────────────────────────────────────────────────────────
    pipeline = train_and_evaluate(X_train, y_train)

    # Feature importances for OOD detection
    importances = pd.Series(
        pipeline.named_steps["model"].feature_importances_,
        index=feature_cols,
    )

    # ── Predict PS2 ───────────────────────────────────────────────────────────
    if df_ps2.empty:
        log.warning("No PS2 data. Writing empty results.")
        pd.DataFrame(columns=["station_code", "source_file",
                               "predicted_latitude", "mahalanobis_dist",
                               "is_ood"]).to_csv(output_file, index=False)
        return

    X_ps2           = df_ps2[feature_cols]
    lat_preds       = pipeline.predict(X_ps2)
    distances, thr  = detect_ood(X_train, X_ps2, importances)

    results = df_ps2[["station_code", "source_file"]].copy()
    results["predicted_latitude"] = lat_preds
    results["mahalanobis_dist"]   = distances
    results["is_ood"]             = distances > thr

    results.to_csv(output_file, index=False)
    log.info("Results written to %s", output_file)

    log.info("\n%s", results.to_string())
    ood_count = results["is_ood"].sum()
    log.info("%d / %d PS2 stations flagged as outside PS1 geographic region.",
             ood_count, len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latitude prediction pipeline")
    parser.add_argument("--ps1",    default=os.environ.get("PS1_DIR", "./data/PS1"))
    parser.add_argument("--ps2",    default=os.environ.get("PS2_DIR", "./data/PS2"))
    parser.add_argument("--output", default="prediction_results.csv")
    args = parser.parse_args()

    main(args.ps1, args.ps2, args.output)
