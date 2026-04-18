import numpy as np
import pandas as pd
from typing import Any


FEATURE_COLUMNS = [
    "machine_temperature",
    "bearing_temperature",
    "vibration_level",
    "pressure",
    "runtime_hours",
    "load_percentage",
    "maintenance_delay_days",
    "error_log_count",
]

TARGET_COLUMN = "downtime_risk"

COLUMN_ALIASES = {
    "machine_temperature": [
        "machine_temperature",
        "process_temperature",
        "process temperature [k]",
        "process temperature",
        "temperature",
        "temp",
    ],
    "bearing_temperature": [
        "bearing_temperature",
        "air_temperature",
        "air temperature [k]",
        "air temperature",
        "bearing temp",
    ],
    "vibration_level": [
        "vibration_level",
        "vibration",
        "vibration amplitude",
        "torque [nm]",
        "torque",
    ],
    "pressure": [
        "pressure",
        "hydraulic_pressure",
        "process_pressure",
        "pressure_bar",
    ],
    "runtime_hours": [
        "runtime_hours",
        "runtime",
        "tool wear [min]",
        "tool_wear",
        "usage_hours",
    ],
    "load_percentage": [
        "load_percentage",
        "load",
        "load_percent",
        "rotational speed [rpm]",
        "rpm",
    ],
    "maintenance_delay_days": [
        "maintenance_delay_days",
        "maintenance_delay",
        "days_since_maintenance",
        "maintenance_gap",
    ],
    "error_log_count": [
        "error_log_count",
        "error_count",
        "error_logs",
        "failure_count",
        "fault_count",
    ],
    "downtime_risk": [
        "downtime_risk",
        "target",
        "label",
        "failure",
        "machine_failure",
        "machine failure",
    ],
}


def _normalize_column_name(column_name: str) -> str:
    return (
        column_name.strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _find_matching_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized_aliases = {_normalize_column_name(alias) for alias in aliases}
    for column in columns:
        if _normalize_column_name(column) in normalized_aliases:
            return column
    return None


def standardize_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    standardized = pd.DataFrame(index=df.index)
    source_columns = list(df.columns)

    for feature_name in FEATURE_COLUMNS:
        matched_column = _find_matching_column(source_columns, COLUMN_ALIASES[feature_name])
        if matched_column is not None:
            standardized[feature_name] = pd.to_numeric(df[matched_column], errors="coerce")

    if "machine_temperature" not in standardized.columns:
        raise ValueError("A machine or process temperature column is required.")

    if "bearing_temperature" not in standardized.columns:
        standardized["bearing_temperature"] = standardized["machine_temperature"] - 6

    if "vibration_level" not in standardized.columns:
        standardized["vibration_level"] = 2.5

    if "pressure" not in standardized.columns:
        standardized["pressure"] = 140.0

    if "runtime_hours" not in standardized.columns:
        standardized["runtime_hours"] = 1200

    if "load_percentage" not in standardized.columns:
        standardized["load_percentage"] = 70.0

    if "maintenance_delay_days" not in standardized.columns:
        standardized["maintenance_delay_days"] = 15

    if "error_log_count" not in standardized.columns:
        standardized["error_log_count"] = 0

    target_column = _find_matching_column(source_columns, COLUMN_ALIASES[TARGET_COLUMN])
    if target_column is not None:
        standardized[TARGET_COLUMN] = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
    elif {"twf", "hdf", "pwf", "osf", "rnf"}.issubset({_normalize_column_name(name) for name in source_columns}):
        normalized_lookup = {_normalize_column_name(name): name for name in source_columns}
        failure_columns = [normalized_lookup[key] for key in ["twf", "hdf", "pwf", "osf", "rnf"]]
        standardized[TARGET_COLUMN] = df[failure_columns].max(axis=1).astype(int)

    standardized = standardized[FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in standardized.columns else [])]
    return standardized


def load_dataset_from_csv(dataset_path: str | Any) -> pd.DataFrame:
    raw_df = pd.read_csv(dataset_path)
    standardized_df = standardize_dataset(raw_df)
    return standardized_df


def generate_sample_dataset(num_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame(
        {
            "machine_temperature": rng.normal(78, 10, num_rows).clip(35, 120),
            "bearing_temperature": rng.normal(82, 12, num_rows).clip(30, 140),
            "vibration_level": rng.normal(4.5, 1.8, num_rows).clip(0.1, 15),
            "pressure": rng.normal(145, 22, num_rows).clip(60, 250),
            "runtime_hours": rng.integers(50, 10000, num_rows),
            "load_percentage": rng.normal(70, 18, num_rows).clip(5, 120),
            "maintenance_delay_days": rng.integers(0, 180, num_rows),
            "error_log_count": rng.poisson(2.5, num_rows).clip(0, 20),
        }
    )

    risk_score = (
        0.030 * (df["machine_temperature"] - 70)
        + 0.028 * (df["bearing_temperature"] - 75)
        + 0.240 * (df["vibration_level"] - 3)
        + 0.012 * (df["pressure"] - 130)
        + 0.00025 * df["runtime_hours"]
        + 0.015 * (df["load_percentage"] - 60)
        + 0.022 * df["maintenance_delay_days"]
        + 0.280 * df["error_log_count"]
        - 4.8
    )

    probability = 1 / (1 + np.exp(-risk_score))
    df[TARGET_COLUMN] = rng.binomial(1, probability)

    missing_mask = rng.random(df.shape[0]) < 0.03
    df.loc[missing_mask, "pressure"] = np.nan

    return df
