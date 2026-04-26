from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.downtime_risk.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_sample_dataset, load_dataset_from_csv


def load_or_create_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.exists():
        return load_dataset_from_csv(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_dataset()
    df.to_csv(dataset_path, index=False)
    return df


def train_and_select_model(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name = ""
    best_model = None
    best_cv = -1.0
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        if scores.mean() > best_cv:
            best_name = name
            best_model = pipeline
            best_cv = float(scores.mean())

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "cv_f1": best_cv,
    }
    return {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "metrics_json": json.dumps(metrics, indent=2),
    }


def train_and_save(dataset_path: Path) -> dict:
    import joblib

    df = load_or_create_dataset(Path(dataset_path))
    results = train_and_select_model(df)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["model"], artifacts_dir / "best_model.joblib")
    joblib.dump(results["feature_columns"], artifacts_dir / "feature_columns.joblib")
    (artifacts_dir / "metrics.json").write_text(results["metrics_json"], encoding="utf-8")
    return results

