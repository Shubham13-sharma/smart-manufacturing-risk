"""ML pipeline: preprocessing → training → evaluation → model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset_from_csv

ARTIFACT_DIR = Path("artifacts")


# ── Pipeline builders ─────────────────────────────────────────────────────────


def _make_lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _make_rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ── Evaluation helper ─────────────────────────────────────────────────────────


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


# ── Main training entry-point ─────────────────────────────────────────────────


def train_and_save(dataset_path: str | Path) -> dict[str, Any]:
    """Train LR and RF pipelines, save the best one and return its metrics.

    Parameters
    ----------
    dataset_path:
        Path to the training CSV (project-native or AI4I-style schema).

    Returns
    -------
    dict with keys: model_name, metrics, artifact_dir
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset_from_csv(dataset_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Training dataset must contain a target column. "
            f"Expected '{TARGET_COLUMN}' or an alias like 'target', 'failure', 'machine_failure'."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Guard: need at least 2 classes
    if y.nunique() < 2:
        raise ValueError("Target column has fewer than 2 unique classes. Cannot train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates: dict[str, Pipeline] = {
        "LogisticRegression": _make_lr_pipeline(),
        "RandomForest": _make_rf_pipeline(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name: str = ""
    best_score: float = -1.0
    best_pipeline: Pipeline | None = None

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        mean_cv = float(np.mean(cv_scores))
        print(f"  {name:<25} CV F1 = {mean_cv:.4f}")
        if mean_cv > best_score:
            best_score = mean_cv
            best_name = name
            best_pipeline = pipe

    assert best_pipeline is not None
    metrics = evaluate(best_pipeline, X_test, y_test)
    metrics["cv_f1"] = round(best_score, 4)

    # Save artifacts
    joblib.dump(best_pipeline, ARTIFACT_DIR / "best_model.joblib")
    joblib.dump(FEATURE_COLUMNS, ARTIFACT_DIR / "feature_columns.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    print(f"\n✓ Best model: {best_name}  |  Test F1: {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")
    print(f"  Artifacts saved to: {ARTIFACT_DIR.resolve()}")

    return {"model_name": best_name, "metrics": metrics, "artifact_dir": str(ARTIFACT_DIR.resolve())}
