"""Data loading and normalisation utilities.

Supports three input styles:
  1. Project-native column names.
  2. Common predictive-maintenance labels (target, failure, machine_failure …).
  3. AI4I-style columns (Process temperature [K], Torque [Nm] …).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import io

import numpy as np
import pandas as pd

# ── Canonical feature set ────────────────────────────────────────────────────
FEATURE_COLUMNS: list[str] = [
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

# ── Column mapping tables ─────────────────────────────────────────────────────

# AI4I / UCI predictive-maintenance style → project-native
_AI4I_MAP: dict[str, str] = {
    "air temperature [k]": "machine_temperature",
    "process temperature [k]": "bearing_temperature",
    "rotational speed [rpm]": "vibration_level",
    "torque [nm]": "pressure",
    "tool wear [min]": "runtime_hours",
}

# Common alternative names → project-native
_ALIAS_MAP: dict[str, str] = {
    # temperature
    "temp": "machine_temperature",
    "temperature": "machine_temperature",
    "air_temp": "machine_temperature",
    "machine_temp": "machine_temperature",
    "process_temp": "bearing_temperature",
    "bearing_temp": "bearing_temperature",
    # vibration
    "vibration": "vibration_level",
    "vib": "vibration_level",
    "rotational_speed": "vibration_level",
    # pressure / torque
    "torque": "pressure",
    # runtime
    "tool_wear": "runtime_hours",
    "runtime": "runtime_hours",
    "hours": "runtime_hours",
    # load
    "load": "load_percentage",
    "load_pct": "load_percentage",
    # maintenance
    "maintenance_delay": "maintenance_delay_days",
    "maint_delay": "maintenance_delay_days",
    "days_since_maintenance": "maintenance_delay_days",
    # errors
    "error_count": "error_log_count",
    "errors": "error_log_count",
    "error_logs": "error_log_count",
    "fault_count": "error_log_count",
}

# Target column aliases
_TARGET_ALIASES: list[str] = [
    "target",
    "failure",
    "machine_failure",
    "label",
    "fault",
    "breakdown",
    "downtime",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with columns renamed to the project-native schema."""
    df = df.copy()
    rename: dict[str, str] = {}

    for col in df.columns:
        low = col.strip().lower()
        if low in _AI4I_MAP:
            rename[col] = _AI4I_MAP[low]
        elif low in _ALIAS_MAP:
            rename[col] = _ALIAS_MAP[low]

    df = df.rename(columns=rename)

    # Normalise the target column
    for col in df.columns:
        if col.strip().lower() in _TARGET_ALIASES and TARGET_COLUMN not in df.columns:
            df = df.rename(columns={col: TARGET_COLUMN})
            break

    return df


def _fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing feature columns filled with column-median (or 0)."""
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_dataset_from_csv(
    source: Union[str, Path, io.BytesIO, io.StringIO],
) -> pd.DataFrame:
    """Load a CSV from *source*, normalise columns, and return a clean DataFrame.

    Raises
    ------
    ValueError
        When the CSV cannot be parsed or contains no recognisable feature data.
    """
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = _normalise_columns(df)

    # Check we have at least some useful columns
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available:
        raise ValueError(
            "No recognisable feature columns found. "
            "Expected columns like machine_temperature, vibration_level, "
            "or AI4I-style columns such as 'Air temperature [K]'."
        )

    df = _fill_missing_features(df)

    # Coerce feature columns to numeric, filling gaps with median
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Coerce target if present
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    return df
