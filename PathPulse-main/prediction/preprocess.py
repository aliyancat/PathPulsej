"""
PathPulse — Data Preprocessing Module
======================================
Handles fetching the UCI Cleveland Heart Disease dataset, cleaning,
imputation, and feature scaling.  Exposes utilities that both the
training script (`train.py`) and the inference pipeline (`predict.py`)
consume so that the exact same transformations are applied consistently.

Dataset: UCI ML Repository — Heart Disease (Cleveland subset)
    303 records · 13 clinical features · binary target (0 / 1)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directory where fitted preprocessor artefacts are persisted
_MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"

# Paths for serialised preprocessor objects
IMPUTER_PATH: Path = _MODELS_DIR / "imputer.pkl"
SCALER_PATH: Path = _MODELS_DIR / "scaler.pkl"

# Columns that may contain missing values (coded as `?` or NaN in the raw
# UCI dataset) — the PRD specifies median imputation for these two.
COLS_TO_IMPUTE: List[str] = ["ca", "thal"]

# Continuous features that must be standardised (required for KNN; also
# applied globally so the Random Forest sees the same feature space).
CONTINUOUS_FEATURES: List[str] = [
    "age",
    "trestbps",   # resting blood pressure
    "chol",       # serum cholesterol
    "thalach",    # max heart rate achieved
    "oldpeak",    # ST depression
]

# Canonical ordered list of all 13 input features — the model was trained
# with columns in this exact order, so inference must match.
ALL_FEATURE_NAMES: List[str] = [
    "age",
    "sex",
    "cp",          # chest pain type (1-4)
    "trestbps",
    "chol",
    "fbs",         # fasting blood sugar > 120 mg/dl
    "restecg",     # resting ECG result
    "thalach",
    "exang",       # exercise-induced angina
    "oldpeak",
    "slope",       # slope of peak ST segment
    "ca",          # number of major vessels (0-3)
    "thal",        # thalassemia type
]

# Human-readable labels for the UI form (same order as ALL_FEATURE_NAMES)
FEATURE_LABELS: Dict[str, str] = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl",
    "restecg":  "Resting ECG Result",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "Slope of Peak ST Segment",
    "ca":       "Number of Major Vessels (0–3)",
    "thal":     "Thalassemia Type",
}

# Default (healthy-baseline) values used to pre-fill the Streamlit form
# so testers don't have to enter everything from scratch.
HEALTHY_DEFAULTS: Dict[str, Any] = {
    "age":      45,
    "sex":      1,        # 1 = male, 0 = female
    "cp":       0,        # 0 = typical angina
    "trestbps": 120,
    "chol":     200,
    "fbs":      0,        # 0 = false
    "restecg":  0,        # 0 = normal
    "thalach":  150,
    "exang":    0,        # 0 = no
    "oldpeak":  0.0,
    "slope":    1,        # 1 = flat
    "ca":       0,
    "thal":     2,        # 2 = normal
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def fetch_heart_disease_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch the Cleveland Heart Disease dataset from the UCI ML Repository.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 13 columns, shape (303, 13).
    y : pd.Series
        Binary target — 0 (no disease) / 1 (disease present).
        The raw dataset has values 0–4; we binarise: > 0 → 1.
    """
    heart_disease = fetch_ucirepo(id=45)  # ID 45 = Heart Disease

    X: pd.DataFrame = heart_disease.data.features.copy()
    y: pd.Series = heart_disease.data.targets.iloc[:, 0].copy()

    # Binarise target: 0 stays 0, anything > 0 becomes 1
    y = (y > 0).astype(int)
    y.name = "target"

    # Ensure column names are lowercase for consistency
    X.columns = [col.lower().strip() for col in X.columns]

    return X, y


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def fit_imputer(X: pd.DataFrame) -> SimpleImputer:
    """Fit a median imputer on the columns that may have missing values.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.

    Returns
    -------
    SimpleImputer
        Fitted imputer (median strategy).
    """
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X[COLS_TO_IMPUTE])
    return imputer


def apply_imputer(X: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """Apply a fitted imputer to fill missing values in `ca` and `thal`.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (may contain NaNs in ``ca`` / ``thal``).
    imputer : SimpleImputer
        Previously fitted imputer.

    Returns
    -------
    pd.DataFrame
        Copy of X with missing values filled.
    """
    X = X.copy()
    X[COLS_TO_IMPUTE] = imputer.transform(X[COLS_TO_IMPUTE])
    return X


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the continuous feature columns.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix (after imputation).

    Returns
    -------
    StandardScaler
        Fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(X[CONTINUOUS_FEATURES])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted scaler to standardise continuous features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (after imputation).
    scaler : StandardScaler
        Previously fitted scaler.

    Returns
    -------
    pd.DataFrame
        Copy of X with continuous columns standardised.
    """
    X = X.copy()
    X[CONTINUOUS_FEATURES] = scaler.transform(X[CONTINUOUS_FEATURES])
    return X


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_training_data(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    """Run the full preprocessing pipeline on training data.

    1. Fit & apply median imputer on ``ca``, ``thal``.
    2. Fit & apply StandardScaler on continuous features.
    3. Save fitted imputer and scaler to disk.

    Parameters
    ----------
    X : pd.DataFrame
        Raw training features.

    Returns
    -------
    X_processed : pd.DataFrame
        Cleaned and scaled feature matrix.
    imputer : SimpleImputer
        Fitted imputer (saved to disk).
    scaler : StandardScaler
        Fitted scaler (saved to disk).
    """
    # Ensure output directory exists
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — Imputation
    imputer: SimpleImputer = fit_imputer(X)
    X_clean: pd.DataFrame = apply_imputer(X, imputer)

    # Step 2 — Scaling
    scaler: StandardScaler = fit_scaler(X_clean)
    X_processed: pd.DataFrame = apply_scaler(X_clean, scaler)

    # Step 3 — Persist fitted objects
    save_preprocessor(imputer, IMPUTER_PATH)
    save_preprocessor(scaler, SCALER_PATH)

    return X_processed, imputer, scaler


def preprocess_new_input(
    patient_data: Dict[str, Any],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Transform a single patient record using pre-fitted preprocessors.

    This is the function called at inference time — it mirrors the exact
    same transformations that were applied during training.

    Parameters
    ----------
    patient_data : dict
        Mapping of feature name → value for one patient (13 keys).
    imputer : SimpleImputer
        Fitted imputer loaded from disk.
    scaler : StandardScaler
        Fitted scaler loaded from disk.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for model prediction.
    """
    # Build a single-row DataFrame in the canonical column order
    row: pd.DataFrame = pd.DataFrame([patient_data], columns=ALL_FEATURE_NAMES)

    # Apply imputation (in case user left ca/thal blank — unlikely via UI
    # but keeps the pipeline robust)
    row = apply_imputer(row, imputer)

    # Apply scaling
    row = apply_scaler(row, scaler)

    return row


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_preprocessor(obj: Any, path: Path) -> None:
    """Pickle a fitted preprocessor to disk.

    Parameters
    ----------
    obj : Any
        Scikit-learn transformer (imputer or scaler).
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_preprocessor(path: Path) -> Any:
    """Load a pickled preprocessor from disk.

    Parameters
    ----------
    path : Path
        File path to the serialised object.

    Returns
    -------
    Any
        The deserialised scikit-learn transformer.

    Raises
    ------
    FileNotFoundError
        If the artefact file does not exist at *path*.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor artefact not found at {path}. "
            "Run `python -m prediction.train` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
