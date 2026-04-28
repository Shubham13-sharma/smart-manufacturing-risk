from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

import pandas as pd


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


def _connect(config: DatabaseConfig):
    import mysql.connector

    kwargs = {
        "host": config.host,
        "port": int(config.port),
        "user": config.user,
        "password": config.password,
        "database": config.database,
        "connection_timeout": 12,
    }
    if "aivencloud.com" in str(config.host).lower():
        kwargs["ssl_disabled"] = False
    return mysql.connector.connect(**kwargs)


def test_connection(config: DatabaseConfig) -> tuple[bool, str]:
    try:
        conn = _connect(config)
        conn.close()
        return True, "Connected to MySQL successfully."
    except Exception as exc:
        return False, str(exc)


def _get_table_columns(cur, table_name: str) -> set[str]:
    cur.execute(f"SHOW COLUMNS FROM {table_name}")
    return {str(row[0]) for row in cur.fetchall()}


def _ensure_column(cur, table_name: str, column_name: str, definition: str) -> None:
    existing_columns = _get_table_columns(cur, table_name)
    if column_name not in existing_columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _ensure_varchar_width(cur, table_name: str, column_name: str, width: int) -> None:
    try:
        cur.execute(
            f"ALTER TABLE {table_name} MODIFY {column_name} VARCHAR({int(width)})"
        )
    except Exception:
        pass


def _get_column_info(cur, table_name: str, column_name: str) -> dict[str, str]:
    cur.execute(f"SHOW COLUMNS FROM {table_name} LIKE %s", (column_name,))
    row = cur.fetchone()
    if not row:
        return {}
    return {
        "field": str(row[0]),
        "type": str(row[1]).lower(),
        "null": str(row[2]).lower(),
        "key": str(row[3]).lower(),
        "default": "" if row[4] is None else str(row[4]),
        "extra": str(row[5]).lower(),
    }


def _run_id_mode(cur) -> str:
    prediction_info = _get_column_info(cur, "prediction_runs", "run_id")
    machine_info = _get_column_info(cur, "machine_predictions", "run_id")
    run_id_types = f"{prediction_info.get('type', '')} {machine_info.get('type', '')}"
    if "auto_increment" in prediction_info.get("extra", ""):
        return "auto_integer"
    if any(token in run_id_types for token in ("tinyint", "smallint", "mediumint", "int", "bigint")):
        return "manual_integer"
    return "text"


def _make_run_id(mode: str) -> str | int | None:
    if mode == "text":
        return uuid.uuid4().hex
    if mode == "manual_integer":
        return uuid.uuid4().int % 2_000_000_000
    return None


def _insert_dynamic(cur, table_name: str, values_by_column: dict[str, object]) -> None:
    available_columns = _get_table_columns(cur, table_name)
    columns = [column for column in values_by_column if column in available_columns]
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)
    values = tuple(values_by_column[column] for column in columns)
    cur.execute(
        f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
        values,
    )


def initialize_tables(config: DatabaseConfig) -> None:
    conn = _connect(config)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_runs (
            run_id VARCHAR(64) PRIMARY KEY,
            source_name VARCHAR(255),
            record_count INT NOT NULL DEFAULT 0,
            average_risk DOUBLE,
            high_risk_count INT NOT NULL DEFAULT 0,
            saved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_column(cur, "prediction_runs", "record_count", "INT NOT NULL DEFAULT 0")
    _ensure_column(cur, "prediction_runs", "average_risk", "DOUBLE")
    _ensure_column(cur, "prediction_runs", "high_risk_count", "INT NOT NULL DEFAULT 0")
    _ensure_column(cur, "prediction_runs", "saved_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP")
    _ensure_column(cur, "prediction_runs", "created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP")
    _ensure_varchar_width(cur, "prediction_runs", "run_id", 64)
    _ensure_varchar_width(cur, "prediction_runs", "source_name", 255)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS machine_predictions (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(64),
            machine_label VARCHAR(255),
            predicted_risk INT,
            risk_probability DOUBLE,
            recommendation TEXT,
            machine_temperature DOUBLE,
            bearing_temperature DOUBLE,
            vibration_level DOUBLE,
            pressure DOUBLE,
            runtime_hours DOUBLE,
            load_percentage DOUBLE,
            maintenance_delay_days DOUBLE,
            error_log_count DOUBLE,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_column(cur, "machine_predictions", "machine_label", "VARCHAR(255)")
    _ensure_column(cur, "machine_predictions", "predicted_risk", "INT")
    _ensure_column(cur, "machine_predictions", "risk_probability", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "recommendation", "TEXT")
    _ensure_column(cur, "machine_predictions", "machine_temperature", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "bearing_temperature", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "vibration_level", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "pressure", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "runtime_hours", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "load_percentage", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "maintenance_delay_days", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "error_log_count", "DOUBLE")
    _ensure_column(cur, "machine_predictions", "created_at", "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP")
    _ensure_varchar_width(cur, "machine_predictions", "run_id", 64)
    _ensure_varchar_width(cur, "machine_predictions", "machine_label", 255)
    for table, column in [
        ("prediction_runs", "saved_at"),
        ("prediction_runs", "created_at"),
        ("machine_predictions", "created_at"),
    ]:
        try:
            cur.execute(f"ALTER TABLE {table} MODIFY {column} TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            pass
    conn.commit()
    cur.close()
    conn.close()


def save_single_prediction(
    config: DatabaseConfig,
    input_df: pd.DataFrame,
    prediction: int,
    probability: float,
    recommendation: str,
    machine_label: str,
) -> str:
    df = input_df.copy()
    df["predicted_risk"] = prediction
    df["risk_probability"] = probability
    df["recommendation"] = recommendation
    df["machine_label"] = machine_label
    return save_batch_predictions(config, df, "single_machine_demo")


def _insert_prediction_batch(
    cur,
    scored_df: pd.DataFrame,
    source_name: str,
    run_id_mode: str,
) -> str:
    run_id = _make_run_id(run_id_mode)
    run_summary_values = (
        source_name,
        int(len(scored_df)),
        float(scored_df["risk_probability"].mean()) if len(scored_df) else 0.0,
        int(scored_df["predicted_risk"].sum()) if len(scored_df) else 0,
    )
    now = datetime.now()
    run_record = {
        "source_name": run_summary_values[0],
        "record_count": run_summary_values[1],
        "total_records": run_summary_values[1],
        "average_risk": run_summary_values[2],
        "high_risk_count": run_summary_values[3],
        "high_risk": run_summary_values[3],
        "created_at": now,
        "saved_at": now,
    }
    if run_id_mode == "auto_integer":
        _insert_dynamic(cur, "prediction_runs", run_record)
        run_id = cur.lastrowid
    else:
        run_record["run_id"] = run_id
        _insert_dynamic(cur, "prediction_runs", run_record)

    machine_columns = _get_table_columns(cur, "machine_predictions")
    machine_insert_columns = [
        column
        for column in [
            "run_id",
            "machine_label",
            "predicted_risk",
            "risk_probability",
            "recommendation",
            "machine_temperature",
            "bearing_temperature",
            "vibration_level",
            "pressure",
            "runtime_hours",
            "load_percentage",
            "maintenance_delay_days",
            "error_log_count",
            "created_at",
        ]
        if column in machine_columns
    ]
    insert_sql = (
        f"INSERT INTO machine_predictions ({', '.join(machine_insert_columns)}) "
        f"VALUES ({', '.join(['%s'] * len(machine_insert_columns))})"
    )
    rows = []
    for idx, row in scored_df.iterrows():
        machine_record = {
            "run_id": run_id,
            "machine_label": str(row.get("machine_label", f"MCH-{idx+1:04d}")),
            "predicted_risk": int(row.get("predicted_risk", 0)),
            "risk_probability": float(row.get("risk_probability", 0.0)),
            "recommendation": str(row.get("recommendation", "")),
            "machine_temperature": float(row.get("machine_temperature", 0.0)),
            "bearing_temperature": float(row.get("bearing_temperature", 0.0)),
            "vibration_level": float(row.get("vibration_level", 0.0)),
            "pressure": float(row.get("pressure", 0.0)),
            "runtime_hours": float(row.get("runtime_hours", 0.0)),
            "load_percentage": float(row.get("load_percentage", 0.0)),
            "maintenance_delay_days": float(row.get("maintenance_delay_days", 0.0)),
            "error_log_count": float(row.get("error_log_count", 0.0)),
            "created_at": now,
        }
        rows.append(tuple(machine_record[column] for column in machine_insert_columns))
    if rows:
        cur.executemany(insert_sql, rows)
    return str(run_id)


def _is_run_id_schema_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "run_id" in message and any(
        text in message
        for text in (
            "incorrect integer value",
            "data truncated",
            "out of range",
            "cannot be null",
        )
    )


def save_batch_predictions(config: DatabaseConfig, scored_df: pd.DataFrame, source_name: str) -> str:
    initialize_tables(config)
    conn = _connect(config)
    cur = conn.cursor()
    try:
        try:
            run_id = _insert_prediction_batch(cur, scored_df, source_name, _run_id_mode(cur))
        except Exception as exc:
            conn.rollback()
            if not _is_run_id_schema_error(exc):
                raise
            run_id = _insert_prediction_batch(cur, scored_df, source_name, "manual_integer")
        conn.commit()
        return str(run_id)
    finally:
        cur.close()
        conn.close()


def fetch_recent_predictions(config: DatabaseConfig, limit: int = 25) -> pd.DataFrame:
    initialize_tables(config)
    conn = _connect(config)
    query = """
        SELECT machine_label, predicted_risk, risk_probability, recommendation,
               machine_temperature, bearing_temperature, vibration_level, pressure,
               runtime_hours, load_percentage, maintenance_delay_days, error_log_count,
               created_at
        FROM machine_predictions
        ORDER BY created_at DESC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(int(limit),))
    conn.close()
    return df
