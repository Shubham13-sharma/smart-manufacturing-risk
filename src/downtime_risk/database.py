from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    import mysql.connector  # type: ignore
except ImportError:  # pragma: no cover
    mysql = None
else:
    mysql = mysql.connector


@dataclass
class DatabaseConfig:
    host: str = ""
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = "railway"

    @property
    def is_aiven(self) -> bool:
        return "aivencloud.com" in self.host.lower()

    def connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": int(self.port),
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "connection_timeout": 15,
        }
        if self.is_aiven:
            kwargs.update(
                {
                    "ssl_disabled": False,
                    "ssl_verify_cert": False,
                    "ssl_verify_identity": False,
                }
            )
        return kwargs


def _require_mysql() -> None:
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed.")


def _connect(cfg: DatabaseConfig):
    _require_mysql()
    return mysql.connect(**cfg.connect_kwargs())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _columns(conn, table_name: str) -> dict[str, str]:
    cur = conn.cursor()
    try:
        cur.execute(f"SHOW COLUMNS FROM {table_name}")
        return {row[0]: str(row[1]).lower() for row in cur.fetchall()}
    finally:
        cur.close()


def _ensure_column(conn, table_name: str, column_name: str, definition: str) -> None:
    cols = _columns(conn, table_name)
    if column_name not in cols:
        cur = conn.cursor()
        try:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
        finally:
            cur.close()


def _run_id_is_integer(cols: dict[str, str]) -> bool:
    return "run_id" in cols and any(token in cols["run_id"] for token in ["int", "bigint", "smallint"])


def test_connection(cfg: DatabaseConfig) -> tuple[bool, str]:
    try:
        conn = _connect(cfg)
        conn.close()
        return True, f"Connected to {cfg.host}:{cfg.port}/{cfg.database} successfully."
    except Exception as exc:
        return False, f"Connection failed: {exc}"


def initialize_tables(cfg: DatabaseConfig) -> None:
    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                run_id VARCHAR(36) NOT NULL PRIMARY KEY,
                source_name VARCHAR(255),
                total_records INT DEFAULT 0,
                record_count INT DEFAULT 0,
                high_risk INT DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS machine_predictions (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                run_id VARCHAR(36),
                machine_label VARCHAR(100),
                machine_temperature FLOAT,
                bearing_temperature FLOAT,
                vibration_level FLOAT,
                pressure FLOAT,
                runtime_hours FLOAT,
                load_percentage FLOAT,
                maintenance_delay_days FLOAT,
                error_log_count FLOAT,
                predicted_risk TINYINT DEFAULT 0,
                risk_probability FLOAT,
                recommendation TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        cur.close()

        _ensure_column(conn, "prediction_runs", "source_name", "VARCHAR(255)")
        _ensure_column(conn, "prediction_runs", "total_records", "INT DEFAULT 0")
        _ensure_column(conn, "prediction_runs", "record_count", "INT DEFAULT 0")
        _ensure_column(conn, "prediction_runs", "high_risk", "INT DEFAULT 0")
        _ensure_column(conn, "prediction_runs", "created_at", "DATETIME NULL DEFAULT CURRENT_TIMESTAMP")

        _ensure_column(conn, "machine_predictions", "machine_label", "VARCHAR(100)")
        _ensure_column(conn, "machine_predictions", "machine_temperature", "FLOAT")
        _ensure_column(conn, "machine_predictions", "bearing_temperature", "FLOAT")
        _ensure_column(conn, "machine_predictions", "vibration_level", "FLOAT")
        _ensure_column(conn, "machine_predictions", "pressure", "FLOAT")
        _ensure_column(conn, "machine_predictions", "runtime_hours", "FLOAT")
        _ensure_column(conn, "machine_predictions", "load_percentage", "FLOAT")
        _ensure_column(conn, "machine_predictions", "maintenance_delay_days", "FLOAT")
        _ensure_column(conn, "machine_predictions", "error_log_count", "FLOAT")
        _ensure_column(conn, "machine_predictions", "predicted_risk", "TINYINT DEFAULT 0")
        _ensure_column(conn, "machine_predictions", "risk_probability", "FLOAT")
        _ensure_column(conn, "machine_predictions", "recommendation", "TEXT")
        _ensure_column(conn, "machine_predictions", "created_at", "DATETIME NULL DEFAULT CURRENT_TIMESTAMP")
        conn.commit()
    finally:
        conn.close()


def _insert_run(conn, source_name: str, total_records: int, high_risk: int) -> str | int:
    run_cols = _columns(conn, "prediction_runs")
    now = _now()
    cur = conn.cursor()
    try:
        count_cols = [col for col in ["total_records", "record_count"] if col in run_cols]
        count_values = [total_records] * len(count_cols)
        if _run_id_is_integer(run_cols):
            columns = ["source_name", *count_cols, "high_risk", "created_at"]
            placeholders = ", ".join(["%s"] * len(columns))
            cur.execute(
                f"INSERT INTO prediction_runs ({', '.join(columns)}) VALUES ({placeholders})",
                (source_name, *count_values, high_risk, now),
            )
            return int(cur.lastrowid)

        run_id = str(uuid.uuid4())
        columns = ["run_id", "source_name", *count_cols, "high_risk", "created_at"]
        placeholders = ", ".join(["%s"] * len(columns))
        cur.execute(
            f"INSERT INTO prediction_runs ({', '.join(columns)}) VALUES ({placeholders})",
            (run_id, source_name, *count_values, high_risk, now),
        )
        return run_id
    finally:
        cur.close()


def _prediction_payload(
    run_id: str | int,
    row: pd.Series,
    predicted_risk: int,
    risk_probability: float,
    recommendation: str,
    machine_label: str,
) -> tuple[Any, ...]:
    return (
        run_id,
        machine_label,
        float(row.get("machine_temperature", 0) or 0),
        float(row.get("bearing_temperature", 0) or 0),
        float(row.get("vibration_level", 0) or 0),
        float(row.get("pressure", 0) or 0),
        float(row.get("runtime_hours", 0) or 0),
        float(row.get("load_percentage", 0) or 0),
        float(row.get("maintenance_delay_days", 0) or 0),
        float(row.get("error_log_count", 0) or 0),
        int(predicted_risk),
        float(risk_probability),
        recommendation,
        _now(),
    )


def save_single_prediction(
    cfg: DatabaseConfig,
    input_df: pd.DataFrame,
    prediction: int,
    probability: float,
    recommendation: str,
    machine_label: str,
) -> str:
    initialize_tables(cfg)
    conn = _connect(cfg)
    try:
        run_id = _insert_run(conn, "single_machine", 1, int(prediction == 1))
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO machine_predictions (
                run_id, machine_label, machine_temperature, bearing_temperature,
                vibration_level, pressure, runtime_hours, load_percentage,
                maintenance_delay_days, error_log_count, predicted_risk,
                risk_probability, recommendation, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            _prediction_payload(
                run_id,
                input_df.iloc[0],
                prediction,
                probability,
                recommendation,
                machine_label,
            ),
        )
        conn.commit()
        cur.close()
        return str(run_id)
    finally:
        conn.close()


def save_batch_predictions(
    cfg: DatabaseConfig,
    scored_df: pd.DataFrame,
    source_name: str,
) -> str:
    initialize_tables(cfg)
    conn = _connect(cfg)
    try:
        high_risk = int(scored_df["predicted_risk"].sum()) if "predicted_risk" in scored_df.columns else 0
        run_id = _insert_run(conn, source_name, len(scored_df), high_risk)
        rows = []
        for idx, row in scored_df.iterrows():
            pred = int(row.get("predicted_risk", 0))
            prob = float(row.get("risk_probability", 0) or 0)
            rec = (
                "Immediate maintenance inspection recommended"
                if pred == 1
                else "Continue monitoring under normal schedule"
            )
            rows.append(
                _prediction_payload(
                    run_id,
                    row,
                    pred,
                    prob,
                    rec,
                    str(row.get("machine_label", f"MACHINE-{idx}")),
                )
            )

        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO machine_predictions (
                run_id, machine_label, machine_temperature, bearing_temperature,
                vibration_level, pressure, runtime_hours, load_percentage,
                maintenance_delay_days, error_log_count, predicted_risk,
                risk_probability, recommendation, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        conn.commit()
        cur.close()
        return str(run_id)
    finally:
        conn.close()


def fetch_recent_predictions(cfg: DatabaseConfig, limit: int = 25) -> pd.DataFrame:
    initialize_tables(cfg)
    conn = _connect(cfg)
    try:
        pred_cols = _columns(conn, "machine_predictions")
        id_col = "id" if "id" in pred_cols else "prediction_id"
        time_col = "created_at" if "created_at" in pred_cols else "saved_at"
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
            SELECT
                mp.{id_col} AS id,
                mp.machine_label,
                mp.predicted_risk,
                mp.risk_probability,
                mp.recommendation,
                mp.{time_col} AS created_at,
                pr.source_name
            FROM machine_predictions mp
            LEFT JOIN prediction_runs pr ON mp.run_id = pr.run_id
            ORDER BY mp.{time_col} DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    finally:
        conn.close()
