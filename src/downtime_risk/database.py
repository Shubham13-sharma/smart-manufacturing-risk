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
    for table, column in [("prediction_runs", "saved_at"), ("machine_predictions", "created_at")]:
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


def save_batch_predictions(config: DatabaseConfig, scored_df: pd.DataFrame, source_name: str) -> str:
    initialize_tables(config)
    run_id = uuid.uuid4().hex
    conn = _connect(config)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO prediction_runs
        (run_id, source_name, record_count, average_risk, high_risk_count)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            run_id,
            source_name,
            int(len(scored_df)),
            float(scored_df["risk_probability"].mean()) if len(scored_df) else 0.0,
            int(scored_df["predicted_risk"].sum()) if len(scored_df) else 0,
        ),
    )
    insert_sql = """
        INSERT INTO machine_predictions
        (run_id, machine_label, predicted_risk, risk_probability, recommendation,
         machine_temperature, bearing_temperature, vibration_level, pressure,
         runtime_hours, load_percentage, maintenance_delay_days, error_log_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    rows = []
    for idx, row in scored_df.iterrows():
        rows.append(
            (
                run_id,
                str(row.get("machine_label", f"MCH-{idx+1:04d}")),
                int(row.get("predicted_risk", 0)),
                float(row.get("risk_probability", 0.0)),
                str(row.get("recommendation", "")),
                float(row.get("machine_temperature", 0.0)),
                float(row.get("bearing_temperature", 0.0)),
                float(row.get("vibration_level", 0.0)),
                float(row.get("pressure", 0.0)),
                float(row.get("runtime_hours", 0.0)),
                float(row.get("load_percentage", 0.0)),
                float(row.get("maintenance_delay_days", 0.0)),
                float(row.get("error_log_count", 0.0)),
            )
        )
    if rows:
        cur.executemany(insert_sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    return run_id


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
