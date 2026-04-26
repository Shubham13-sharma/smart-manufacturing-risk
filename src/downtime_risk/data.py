from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from io import StringIO


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

FEATURE_DEFAULTS = {
    "machine_temperature": 75.0,
    "bearing_temperature": 82.0,
    "vibration_level": 4.5,
    "pressure": 140.0,
    "runtime_hours": 2400.0,
    "load_percentage": 72.0,
    "maintenance_delay_days": 15.0,
    "error_log_count": 2.0,
}

COLUMN_ALIASES = {
    "machine_temperature": [
        "machine_temperature",
        "machine temperature",
        "process temperature [k]",
        "process_temperature",
        "process temperature",
        "temperature",
        "temp",
    ],
    "bearing_temperature": [
        "bearing_temperature",
        "bearing temperature",
        "air temperature [k]",
        "air_temperature",
        "air temperature",
        "bearing temp",
    ],
    "vibration_level": [
        "vibration_level",
        "vibration",
        "vibration level",
        "rotational speed [rpm]",
        "rotational_speed",
        "rpm",
    ],
    "pressure": [
        "pressure",
        "torque [nm]",
        "torque",
        "hydraulic pressure",
    ],
    "runtime_hours": [
        "runtime_hours",
        "runtime",
        "tool wear [min]",
        "tool_wear",
        "operating hours",
        "hours",
    ],
    "load_percentage": [
        "load_percentage",
        "load",
        "load percentage",
        "load_percent",
        "utilization",
    ],
    "maintenance_delay_days": [
        "maintenance_delay_days",
        "maintenance delay",
        "days since maintenance",
        "maintenance_delay",
    ],
    "error_log_count": [
        "error_log_count",
        "error logs",
        "error_count",
        "fault_count",
        "errors",
    ],
}

TARGET_ALIASES = [
    "downtime_risk",
    "machine failure",
    "machine_failure",
    "failure",
    "target",
    "class",
    "label",
]

MACHINE_LABEL_ALIASES = [
    "machine_label",
    "machine label",
    "machine id",
    "machine_id",
    "machine",
    "product id",
    "product_id",
    "udi",
    "asset_id",
    "asset id",
    "equipment_id",
    "equipment id",
]


def _normalise_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def _find_column(raw_df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {_normalise_name(col): col for col in raw_df.columns}
    for alias in aliases:
        found = lookup.get(_normalise_name(alias))
        if found is not None:
            return found
    return None


def _numeric_series(raw_df: pd.DataFrame, source_col: str | None, feature: str) -> pd.Series:
    if source_col and source_col in raw_df.columns:
        series = pd.to_numeric(raw_df[source_col], errors="coerce")
    else:
        series = pd.Series(np.nan, index=raw_df.index, dtype="float64")

    if feature == "bearing_temperature" and series.isna().all():
        machine_col = _find_column(raw_df, COLUMN_ALIASES["machine_temperature"])
        if machine_col:
            machine_temp = pd.to_numeric(raw_df[machine_col], errors="coerce")
            series = machine_temp + 6

    return series.fillna(FEATURE_DEFAULTS[feature]).astype(float)


def standardize_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert common manufacturing CSV formats into the app's feature schema."""
    if raw_df.empty:
        raise ValueError("Dataset is empty.")

    result = pd.DataFrame(index=raw_df.index)
    machine_label_col = _find_column(raw_df, MACHINE_LABEL_ALIASES)
    if machine_label_col:
        result["machine_label"] = raw_df[machine_label_col].astype(str).fillna("").replace("", pd.NA)
    else:
        result["machine_label"] = [f"MCH-{i+1:04d}" for i in range(len(raw_df))]

    for feature in FEATURE_COLUMNS:
        source_col = _find_column(raw_df, COLUMN_ALIASES[feature])
        result[feature] = _numeric_series(raw_df, source_col, feature)

    target_col = _find_column(raw_df, TARGET_ALIASES)
    if target_col:
        result[TARGET_COLUMN] = pd.to_numeric(raw_df[target_col], errors="coerce").fillna(0).astype(int)
    else:
        ai4i_failure_cols = [col for col in ["TWF", "HDF", "PWF", "OSF", "RNF"] if col in raw_df.columns]
        if ai4i_failure_cols:
            result[TARGET_COLUMN] = raw_df[ai4i_failure_cols].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1).astype(int)
        else:
            result[TARGET_COLUMN] = 0

    return result.reset_index(drop=True)


def standardize_dataset_with_mapping(
    raw_df: pd.DataFrame,
    column_mapping: dict[str, str | None],
    target_column: str | None = None,
    machine_label_column: str | None = None,
) -> pd.DataFrame:
    """Create model-ready data from user-selected CSV column mappings."""
    if raw_df.empty:
        raise ValueError("Dataset is empty.")

    result = pd.DataFrame(index=raw_df.index)
    if machine_label_column and machine_label_column in raw_df.columns:
        result["machine_label"] = raw_df[machine_label_column].astype(str).fillna("").replace("", pd.NA)
    else:
        guessed_label_col = _find_column(raw_df, MACHINE_LABEL_ALIASES)
        if guessed_label_col:
            result["machine_label"] = raw_df[guessed_label_col].astype(str).fillna("").replace("", pd.NA)
        else:
            result["machine_label"] = [f"MCH-{i+1:04d}" for i in range(len(raw_df))]

    for feature in FEATURE_COLUMNS:
        mapped_col = column_mapping.get(feature)
        result[feature] = _numeric_series(raw_df, mapped_col, feature)

    if target_column and target_column in raw_df.columns:
        result[TARGET_COLUMN] = pd.to_numeric(raw_df[target_column], errors="coerce").fillna(0).astype(int)
    else:
        result[TARGET_COLUMN] = 0

    return result.reset_index(drop=True)


def read_flexible_csv(dataset_path: str | Path | Any) -> pd.DataFrame:
    """Read CSV files that may contain preamble text, odd separators, or bad lines."""
    try:
        return pd.read_csv(dataset_path)
    except Exception:
        pass

    if hasattr(dataset_path, "seek"):
        dataset_path.seek(0)
    raw_text = dataset_path.read() if hasattr(dataset_path, "read") else Path(dataset_path).read_text(encoding="utf-8", errors="ignore")
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode("utf-8", errors="ignore")

    lines = raw_text.splitlines()
    header_index = 0
    best_score = -1
    for idx, line in enumerate(lines[:80]):
        comma_count = line.count(",")
        semicolon_count = line.count(";")
        tab_count = line.count("\t")
        score = max(comma_count, semicolon_count, tab_count)
        if score > best_score:
            best_score = score
            header_index = idx

    cleaned_text = "\n".join(lines[header_index:])
    for sep in [None, ",", ";", "\t", "|"]:
        try:
            frame = pd.read_csv(
                StringIO(cleaned_text),
                sep=sep,
                engine="python",
                on_bad_lines="skip",
            )
            if not frame.empty and len(frame.columns) > 1:
                return frame
        except Exception:
            continue

    raise ValueError("Could not read this dataset. Please upload a valid CSV file.")


def load_dataset_from_csv(dataset_path: str | Path | Any) -> pd.DataFrame:
    raw_df = read_flexible_csv(dataset_path)
    return standardize_dataset(raw_df)


def generate_sample_dataset(num_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    machine_temperature = rng.normal(75, 12, num_rows).clip(35, 120)
    bearing_temperature = (machine_temperature + rng.normal(7, 5, num_rows)).clip(30, 140)
    vibration_level = rng.gamma(2.2, 1.8, num_rows).clip(0, 15)
    pressure = rng.normal(145, 28, num_rows).clip(55, 250)
    runtime_hours = rng.integers(0, 10000, num_rows)
    load_percentage = rng.normal(70, 18, num_rows).clip(0, 120)
    maintenance_delay_days = rng.integers(0, 181, num_rows)
    error_log_count = rng.poisson(2.2, num_rows).clip(0, 20)

    risk_score = (
        0.018 * (machine_temperature - 75)
        + 0.026 * (bearing_temperature - 82)
        + 0.24 * (vibration_level - 4)
        + 0.004 * (runtime_hours - 3000) / 10
        + 0.018 * (load_percentage - 70)
        + 0.035 * (maintenance_delay_days - 20)
        + 0.42 * error_log_count
        + rng.normal(0, 1.2, num_rows)
    )
    probability = 1 / (1 + np.exp(-risk_score / 6))
    downtime_risk = (probability > np.quantile(probability, 0.68)).astype(int)

    return pd.DataFrame(
        {
            "machine_temperature": machine_temperature.round(2),
            "bearing_temperature": bearing_temperature.round(2),
            "vibration_level": vibration_level.round(2),
            "pressure": pressure.round(2),
            "runtime_hours": runtime_hours,
            "load_percentage": load_percentage.round(2),
            "maintenance_delay_days": maintenance_delay_days,
            "error_log_count": error_log_count,
            "downtime_risk": downtime_risk,
        }
    )
