"""
PathPulse — Model Training Script
==================================
Fetches the UCI Heart Disease dataset, preprocesses it, trains three
classifiers (Random Forest, KNN, Decision Tree), evaluates each on
Accuracy and Recall, and serialises the primary model (Random Forest)
along with the fitted preprocessors to the ``models/`` directory.

Usage
-----
    python -m prediction.train

Run this script **once** before launching the Streamlit app.  The
serialised artefacts are:
    models/random_forest.pkl   — trained Random Forest classifier
    models/imputer.pkl         — fitted SimpleImputer (median)
    models/scaler.pkl          — fitted StandardScaler
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Local imports
from prediction.preprocess import (
    ALL_FEATURE_NAMES,
    fetch_heart_disease_data,
    preprocess_training_data,
    apply_imputer,
    apply_scaler,
    load_preprocessor,
    IMPUTER_PATH,
    SCALER_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH: Path = _MODELS_DIR / "random_forest.pkl"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _build_models() -> Dict[str, Any]:
    """Instantiate the three classifiers specified in the PRD.

    Returns
    -------
    dict
        Mapping of model name → unfitted estimator instance.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }


def _evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute evaluation metrics for a trained model.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn classifier.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    dict
        Accuracy, Precision, Recall, and F1-Score (all rounded to 4 dp).
    """
    y_pred: np.ndarray = model.predict(X_test)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def _print_report(
    name: str,
    metrics: Dict[str, float],
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Pretty-print a single model's evaluation results to stdout.

    Parameters
    ----------
    name : str
        Human-readable model name.
    metrics : dict
        Dict returned by ``_evaluate_model``.
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}   << priority metric")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm: np.ndarray = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Path) -> None:
    """Pickle a trained model to disk.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn classifier.
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[OK]  Model saved -> {path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train() -> None:
    """Execute the full training pipeline.

    Steps
    -----
    1. Fetch the UCI Heart Disease dataset.
    2. Preprocess (impute + scale) and persist fitted transformers.
    3. Stratified 80/20 train-test split.
    4. Train Random Forest, KNN, and Decision Tree.
    5. Evaluate every model; print reports.
    6. Serialise the Random Forest to ``models/random_forest.pkl``.
    """
    print("=" * 50)
    print("  PathPulse -- Model Training Pipeline")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Step 1 — Fetch data
    # ------------------------------------------------------------------
    print("\n[>]  Fetching UCI Heart Disease dataset ...")
    X_raw, y = fetch_heart_disease_data()
    print(f"    Dataset shape: {X_raw.shape[0]} samples x {X_raw.shape[1]} features")
    print(f"    Target distribution: {dict(y.value_counts())}")

    # ------------------------------------------------------------------
    # Step 2 — Preprocess (fits + saves imputer & scaler)
    # ------------------------------------------------------------------
    print("\n[*]  Preprocessing (impute -> scale) ...")
    X_processed, imputer, scaler = preprocess_training_data(X_raw)
    print(f"    Imputer saved  -> {IMPUTER_PATH}")
    print(f"    Scaler saved   -> {SCALER_PATH}")

    # ------------------------------------------------------------------
    # Step 3 — Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n[*]  Split: {len(X_train)} train / {len(X_test)} test  "
          f"(stratified, random_state={RANDOM_STATE})")

    # ------------------------------------------------------------------
    # Step 4 — Train & evaluate all models
    # ------------------------------------------------------------------
    models: Dict[str, Any] = _build_models()
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        metrics: Dict[str, float] = _evaluate_model(model, X_test, y_test)
        results[name] = metrics

        y_pred: np.ndarray = model.predict(X_test)
        _print_report(name, metrics, y_test, y_pred)

    # ------------------------------------------------------------------
    # Step 5 — Summary comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Model Comparison Summary")
    print("=" * 50)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 55}")
    for name, m in results.items():
        marker = " << PRIMARY" if name == "Random Forest" else ""
        print(f"  {name:<25} {m['accuracy']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1_score']:>10.4f}{marker}")

    # ------------------------------------------------------------------
    # Step 6 — Serialise primary model
    # ------------------------------------------------------------------
    primary_model = models["Random Forest"]
    save_model(primary_model, MODEL_PATH)

    # Print feature importances
    importances: np.ndarray = primary_model.feature_importances_
    feature_names: List[str] = list(X_train.columns)
    sorted_indices: np.ndarray = np.argsort(importances)[::-1]

    print("\n[*]  Random Forest Feature Importances:")
    for idx in sorted_indices:
        bar = "#" * int(importances[idx] * 40)
        print(f"    {feature_names[idx]:<12} {importances[idx]:.4f}  {bar}")

    print("\n[OK]  Training complete!  Run `streamlit run app.py` to launch.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
