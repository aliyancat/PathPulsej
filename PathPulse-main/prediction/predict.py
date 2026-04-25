"""
PathPulse — Inference Pipeline
===============================
Loads the serialised Random Forest model and pre-fitted preprocessors,
transforms new patient data, and returns a risk prediction with
confidence score and feature importance breakdown.

This module is called by the Streamlit UI (``app.py``).  All heavy I/O
(pickle loading) should be wrapped with ``@st.cache_resource`` at the
call site to avoid repeated disk reads.
"""

from __future__ import annotations

import pickle
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from prediction.preprocess import (
    ALL_FEATURE_NAMES,
    IMPUTER_PATH,
    SCALER_PATH,
    preprocess_new_input,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH: Path = _MODELS_DIR / "random_forest.pkl"


# ---------------------------------------------------------------------------
# Model & Preprocessor Loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(path: Optional[Path] = None) -> RandomForestClassifier:
    """Load the serialised Random Forest classifier from disk.

    Parameters
    ----------
    path : Path, optional
        Override path to the ``.pkl`` file.  Defaults to
        ``models/random_forest.pkl``.

    Returns
    -------
    RandomForestClassifier
        The trained model, ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    target: Path = path or MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Trained model not found at {target}.  "
            "Please run `python -m prediction.train` first to train "
            "and serialise the model."
        )
    with open(target, "rb") as f:
        model: RandomForestClassifier = pickle.load(f)
    return model


@st.cache_resource
def load_preprocessors(
    imputer_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> tuple[SimpleImputer, StandardScaler]:
    """Load the fitted imputer and scaler from disk.

    Parameters
    ----------
    imputer_path : Path, optional
        Override path for the imputer pickle.
    scaler_path : Path, optional
        Override path for the scaler pickle.

    Returns
    -------
    tuple[SimpleImputer, StandardScaler]
        Fitted imputer and scaler.

    Raises
    ------
    FileNotFoundError
        If either artefact file is missing.
    """
    imp_target: Path = imputer_path or IMPUTER_PATH
    scl_target: Path = scaler_path or SCALER_PATH

    for artefact_path in (imp_target, scl_target):
        if not artefact_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artefact not found at {artefact_path}.  "
                "Please run `python -m prediction.train` first."
            )

    with open(imp_target, "rb") as f:
        imputer: SimpleImputer = pickle.load(f)
    with open(scl_target, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    return imputer, scaler


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_risk(patient_data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a single patient record.

    Parameters
    ----------
    patient_data_dict : dict
        Mapping of feature name → raw value for one patient.
        Must contain all 13 keys defined in
        ``preprocess.ALL_FEATURE_NAMES``.

    Returns
    -------
    dict
        {
            "risk_class":          int,    # 0 = low risk, 1 = high risk
            "probability":         float,  # probability of class 1
            "feature_importance":  dict,   # feature name → importance score
        }

    Raises
    ------
    FileNotFoundError
        If the model or preprocessor files are missing.
    ValueError
        If required feature keys are missing from the input dict.
    """
    # --- Validate input keys -----------------------------------------------
    missing_keys: List[str] = [
        key for key in ALL_FEATURE_NAMES if key not in patient_data_dict
    ]
    if missing_keys:
        raise ValueError(
            f"Missing required features in patient data: {missing_keys}"
        )

    # --- Load artefacts ----------------------------------------------------
    model: RandomForestClassifier = load_model()
    imputer, scaler = load_preprocessors()

    # --- Preprocess the input (same transforms as training) ----------------
    X_input: pd.DataFrame = preprocess_new_input(
        patient_data=patient_data_dict,
        imputer=imputer,
        scaler=scaler,
    )

    # --- Predict -----------------------------------------------------------
    prediction: int = int(model.predict(X_input)[0])
    probabilities: np.ndarray = model.predict_proba(X_input)[0]
    prob_disease: float = round(float(probabilities[1]), 4)

    # --- Feature importances -----------------------------------------------
    importances: np.ndarray = model.feature_importances_
    feature_importance: Dict[str, float] = {
        name: round(float(imp), 4)
        for name, imp in zip(ALL_FEATURE_NAMES, importances)
    }

    return {
        "risk_class": prediction,
        "probability": prob_disease,
        "feature_importance": feature_importance,
    }
