"""MySQL storage layer — Aiven cloud compatible.

Aiven MySQL requires SSL. This module automatically enables SSL for any
host containing 'aivencloud.com', and gracefully falls back to non-SSL
for local connections.

All public functions raise exceptions on failure so the Streamlit
dashboard can catch and display user-friendly error messages.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

try:
    import mysql.connector  # type: ignore
    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False


# ── Config dataclass ──────────────────────────────────────────────────────────


@dataclass
class DatabaseConfig:
    host: str = ""
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = "defaultdb"

    @property
    def _is_aiven(self) -> bool:
        return "aivencloud.com" in self.host.lower()

    def as_connect_kwargs(self) -> dict:
        kwargs: dict = {
            "host":               self.host,
            "port":               self.port,
            "user":               self.user,
            "password":           self.password,
            "database":           self.database,
            "connection_timeout": 10,
        }
        if self._is_aiven:
            kwargs["ssl_disabled"]        = False
            kwargs["ssl_verify_cert"]     = False
            kwargs["ssl_verify_identity"] = False
        return kwargs


# ── Internal helpers ──────────────────────────────────────────────────────────


def _ensure_mysql_installed() -> None:
    if not _MYSQL_AVAILABLE:
        raise RuntimeError(
            "mysql-connector-python is not installed. "
            "Run: pip install mysql-connector-python"
        )


def _connect(cfg: DatabaseConfig):
    _ensure_mysql_installed()
    return mysql.connector.connect(**cfg.as_connect_kwargs())


# ── Public API ────────────────────────────────────────────────────────────────


def test_connection(cfg: DatabaseConfig) -> tuple[bool, str]:
    """Return (success, message)."""
    _ensure_mysql_installed()
    try:
        conn = _connect(cfg)
        conn.close()
        label = "Aiven cloud" if cfg._is_aiven else cfg.host
        return True, f"Connected to {label}:{cfg.port}/{cfg.database} successfully."
    except Exception as exc:
        return False, f"Connection failed: {exc}"


def initialize_tables(cfg: DatabaseConfig) -> None:
    """Create the prediction tables if they do not already exist."""
    _ensure_mysql_installed()

    ddl_runs = """
    CREATE TABLE IF NOT EXISTS prediction_runs (
        run_id        VARCHAR(36)   NOT NULL PRIMARY KEY,
        source_name   VARCHAR(255),
        total_records INT           DEFAULT 0,
        high_risk     INT           DEFAULT 0,
        created_at    DATETIME      NOT NULL
    )
    """

    ddl_preds = """
    CREATE TABLE IF NOT EXISTS machine_predictions (
        id                      BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
        run_id                  VARCHAR(36),
        machine_label           VARCHAR(100),
        machine_temperature     FLOAT,
        bearing_temperature     FLOAT,
        vibration_level         FLOAT,
        pressure                FLOAT,
        runtime_hours           FLOAT,
        load_percentage         FLOAT,
        maintenance_delay_days  FLOAT,
        error_log_count         FLOAT,
        predicted_risk          TINYINT       DEFAULT 0,
        risk_probability        FLOAT,
        recommendation          TEXT,
        created_at              DATETIME      NOT NULL,
        CONSTRAINT fk_run FOREIGN KEY (run_id)
            REFERENCES prediction_runs (run_id)
            ON DELETE CASCADE
    )
    """

    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(ddl_runs)
        cur.execute(ddl_preds)
        conn.commit()
    finally:
        conn.close()


def save_single_prediction(
    cfg: DatabaseConfig,
    input_df: pd.DataFrame,
    prediction: int,
    probability: float,
    recommendation: str,
    machine_label: str,
) -> str:
    """Insert one prediction record. Returns run_id."""
    _ensure_mysql_installed()
    run_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row    = input_df.iloc[0].to_dict()

    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prediction_runs "
            "(run_id, source_name, total_records, high_risk, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (run_id, "single_machine", 1, int(prediction == 1), now),
        )
        cur.execute(
            """
            INSERT INTO machine_predictions
              (run_id, machine_label,
               machine_temperature, bearing_temperature, vibration_level,
               pressure, runtime_hours, load_percentage,
               maintenance_delay_days, error_log_count,
               predicted_risk, risk_probability, recommendation, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id, machine_label,
                float(row.get("machine_temperature", 0)),
                float(row.get("bearing_temperature", 0)),
                float(row.get("vibration_level", 0)),
                float(row.get("pressure", 0)),
                float(row.get("runtime_hours", 0)),
                float(row.get("load_percentage", 0)),
                float(row.get("maintenance_delay_days", 0)),
                float(row.get("error_log_count", 0)),
                int(prediction), float(probability), recommendation, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return run_id


def save_batch_predictions(
    cfg: DatabaseConfig,
    scored_df: pd.DataFrame,
    source_name: str,
) -> str:
    """Insert all rows of a scored DataFrame. Returns run_id."""
    _ensure_mysql_installed()
    run_id    = str(uuid.uuid4())
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    total     = len(scored_df)
    high_risk = int(scored_df["predicted_risk"].sum()) if "predicted_risk" in scored_df.columns else 0

    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO prediction_runs "
            "(run_id, source_name, total_records, high_risk, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (run_id, source_name, total, high_risk, now),
        )

        insert_sql = """
            INSERT INTO machine_predictions
              (run_id, machine_label,
               machine_temperature, bearing_temperature, vibration_level,
               pressure, runtime_hours, load_percentage,
               maintenance_delay_days, error_log_count,
               predicted_risk, risk_probability, recommendation, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        batch = []
        for idx, row in scored_df.iterrows():
            pred = int(row.get("predicted_risk", 0))
            prob = float(row.get("risk_probability", 0.0))
            rec  = (
                "Immediate maintenance inspection recommended"
                if pred == 1
                else "Continue monitoring under normal schedule"
            )
            batch.append((
                run_id,
                str(row.get("machine_label", f"MACHINE-{idx}")),
                float(row.get("machine_temperature", 0)),
                float(row.get("bearing_temperature", 0)),
                float(row.get("vibration_level", 0)),
                float(row.get("pressure", 0)),
                float(row.get("runtime_hours", 0)),
                float(row.get("load_percentage", 0)),
                float(row.get("maintenance_delay_days", 0)),
                float(row.get("error_log_count", 0)),
                pred, prob, rec, now,
            ))

        cur.executemany(insert_sql, batch)
        conn.commit()
    finally:
        conn.close()

    return run_id


def fetch_recent_predictions(cfg: DatabaseConfig, limit: int = 25) -> pd.DataFrame:
    """Return the most recent *limit* predictions as a DataFrame."""
    _ensure_mysql_installed()
    sql = """
        SELECT mp.id, mp.machine_label, mp.predicted_risk,
               mp.risk_probability, mp.recommendation,
               mp.created_at, pr.source_name
        FROM machine_predictions mp
        JOIN prediction_runs pr ON mp.run_id = pr.run_id
        ORDER BY mp.created_at DESC
        LIMIT %s
    """
    conn = _connect(cfg)
    try:
        df = pd.read_sql(sql, conn, params=(limit,))
    finally:
        conn.close()
    return df
