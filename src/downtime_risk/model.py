from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.downtime_risk.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_sample_dataset, load_dataset_from_csv


def load_or_create_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.exists():
        return load_dataset_from_csv(dataset_path)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_dataset(num_rows=1200, random_state=42)
    df.to_csv(dataset_path, index=False)
    return df


def build_models() -> dict[str, Pipeline]:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURE_COLUMNS)]
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1200)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def train_and_select_model(df: pd.DataFrame) -> dict[str, object]:
    feature_columns = FEATURE_COLUMNS.copy()
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    best_model_name = ""
    best_model = None
    best_metrics = None
    best_f1 = -1.0

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_model = model
            best_metrics = metrics

    return {
        "model_name": best_model_name,
        "model": best_model,
        "metrics": best_metrics,
        "metrics_json": json.dumps(best_metrics, indent=2),
        "feature_columns": feature_columns,
    }
