from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:  # pragma: no cover
    mysql = None
    Error = Exception


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


def get_connection(config: DatabaseConfig):
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed.")

    return mysql.connector.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
    )


def test_connection(config: DatabaseConfig) -> tuple[bool, str]:
    try:
        connection = get_connection(config)
        connection.close()
        return True, "Database connection successful."
    except Error as exc:
        return False, str(exc)


def initialize_tables(config: DatabaseConfig) -> None:
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS prediction_runs (
            run_id INT AUTO_INCREMENT PRIMARY KEY,
            source_name VARCHAR(255) NOT NULL,
            record_count INT NOT NULL,
            created_at DATETIME NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS machine_predictions (
            prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id INT NULL,
            machine_label VARCHAR(100) NOT NULL,
            machine_temperature DECIMAL(10, 2) NULL,
            bearing_temperature DECIMAL(10, 2) NULL,
            vibration_level DECIMAL(10, 2) NULL,
            pressure DECIMAL(10, 2) NULL,
            runtime_hours DECIMAL(10, 2) NULL,
            load_percentage DECIMAL(10, 2) NULL,
            maintenance_delay_days DECIMAL(10, 2) NULL,
            error_log_count DECIMAL(10, 2) NULL,
            predicted_risk TINYINT NOT NULL,
            risk_probability DECIMAL(8, 6) NOT NULL,
            recommendation VARCHAR(255) NOT NULL,
            saved_at DATETIME NOT NULL,
            CONSTRAINT fk_prediction_run
                FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
                ON DELETE SET NULL
        )
        """,
    ]

    connection = get_connection(config)
    cursor = connection.cursor()
    try:
        for statement in ddl_statements:
            cursor.execute(statement)
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def save_single_prediction(
    config: DatabaseConfig,
    input_df: pd.DataFrame,
    prediction: int,
    probability: float,
    recommendation: str,
    machine_label: str,
) -> None:
    initialize_tables(config)

    payload = input_df.iloc[0].to_dict()
    connection = get_connection(config)
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO machine_predictions (
                machine_label,
                machine_temperature,
                bearing_temperature,
                vibration_level,
                pressure,
                runtime_hours,
                load_percentage,
                maintenance_delay_days,
                error_log_count,
                predicted_risk,
                risk_probability,
                recommendation,
                saved_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                machine_label,
                float(payload["machine_temperature"]),
                float(payload["bearing_temperature"]),
                float(payload["vibration_level"]),
                float(payload["pressure"]),
                float(payload["runtime_hours"]),
                float(payload["load_percentage"]),
                float(payload["maintenance_delay_days"]),
                float(payload["error_log_count"]),
                int(prediction),
                float(probability),
                recommendation,
                datetime.now(),
            ),
        )
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def save_batch_predictions(
    config: DatabaseConfig,
    scored_df: pd.DataFrame,
    source_name: str,
) -> int:
    initialize_tables(config)

    connection = get_connection(config)
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO prediction_runs (source_name, record_count, created_at)
            VALUES (%s, %s, %s)
            """,
            (source_name, int(len(scored_df)), datetime.now()),
        )
        run_id = cursor.lastrowid

        records = []
        for row_index, row in scored_df.iterrows():
            recommendation = (
                "Immediate maintenance inspection recommended"
                if int(row["predicted_risk"]) == 1
                else "Continue monitoring under normal schedule"
            )
            records.append(
                (
                    int(run_id),
                    f"Machine-{row_index + 1}",
                    _to_float(row, "machine_temperature"),
                    _to_float(row, "bearing_temperature"),
                    _to_float(row, "vibration_level"),
                    _to_float(row, "pressure"),
                    _to_float(row, "runtime_hours"),
                    _to_float(row, "load_percentage"),
                    _to_float(row, "maintenance_delay_days"),
                    _to_float(row, "error_log_count"),
                    int(row["predicted_risk"]),
                    float(row["risk_probability"]),
                    recommendation,
                    datetime.now(),
                )
            )

        cursor.executemany(
            """
            INSERT INTO machine_predictions (
                run_id,
                machine_label,
                machine_temperature,
                bearing_temperature,
                vibration_level,
                pressure,
                runtime_hours,
                load_percentage,
                maintenance_delay_days,
                error_log_count,
                predicted_risk,
                risk_probability,
                recommendation,
                saved_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            records,
        )
        connection.commit()
        return int(run_id)
    finally:
        cursor.close()
        connection.close()


def fetch_recent_predictions(config: DatabaseConfig, limit: int = 20) -> pd.DataFrame:
    initialize_tables(config)

    connection = get_connection(config)
    query = """
        SELECT
            prediction_id,
            run_id,
            machine_label,
            predicted_risk,
            risk_probability,
            recommendation,
            saved_at
        FROM machine_predictions
        ORDER BY saved_at DESC
        LIMIT %s
    """
    try:
        return pd.read_sql(query, connection, params=(limit,))
    finally:
        connection.close()


def _to_float(row: pd.Series, column_name: str) -> float | None:
    value = row.get(column_name)
    if pd.isna(value):
        return None
    return float(value)
