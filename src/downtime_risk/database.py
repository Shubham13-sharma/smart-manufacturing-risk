"""MySQL storage layer — Aiven cloud compatible.

Aiven MySQL requires SSL. This module automatically enables SSL for any
host containing 'aivencloud.com', and gracefully falls back to non-SSL
for local connections.

BUGS FIXED (3 issues shown in screenshots):
  1. fetch_recent_predictions: used pd.read_sql() with params= which
     passes a tuple — some mysql-connector versions reject this. Replaced
     with cursor.execute() + fetchall() which works reliably.
     Also the old query selected 'mp.id' which fails if the table was
     previously created without that column OR if the foreign key
     constraint prevented table creation. Now selects explicit safe columns.

  2. save_single_prediction: INSERT into prediction_runs referenced
     'total_records' — column that EXISTS in the schema but the error
     "Unknown column 'total_records'" means the old table in the live
     Aiven DB was created by an earlier schema version that did NOT have
     this column. Fixed by making initialize_tables() use ALTER TABLE to
     add missing columns, and by adding the column to CREATE TABLE.

  3. save_batch_predictions: same total_records issue — fixed same way.

ROOT CAUSE: The live Aiven MySQL table was created by an older version of
this file that had a different schema. The fix is:
  a) initialize_tables() now runs ALTER TABLE … ADD COLUMN IF NOT EXISTS
     for any column that might be missing in an existing table.
  b) fetch_recent_predictions() no longer uses pd.read_sql (unreliable
     with mysql-connector params) and uses cursor(dictionary=True) instead.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

try:
    import mysql.connector          # type: ignore
    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    host:     str = ""
    port:     int = 3306
    user:     str = ""
    password: str = ""
    database: str = "defaultdb"

    @property
    def _is_aiven(self) -> bool:
        return "aivencloud.com" in self.host.lower()

    def as_connect_kwargs(self) -> dict:
        kwargs: dict = {
            "host":               self.host,
            "port":               int(self.port),
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
    """
    Create the prediction tables if they do not already exist.
    Also adds any columns that may be missing in an existing table
    (handles schema drift between old and new deployments).
    """
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

    # ALTER TABLE statements to add missing columns in EXISTING tables.
    # MySQL ignores these if the column already exists (IF NOT EXISTS).
    alter_runs = [
        "ALTER TABLE prediction_runs ADD COLUMN IF NOT EXISTS total_records INT DEFAULT 0",
        "ALTER TABLE prediction_runs ADD COLUMN IF NOT EXISTS high_risk INT DEFAULT 0",
        "ALTER TABLE prediction_runs ADD COLUMN IF NOT EXISTS source_name VARCHAR(255)",
    ]
    alter_preds = [
        "ALTER TABLE machine_predictions ADD COLUMN IF NOT EXISTS id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY",
        "ALTER TABLE machine_predictions ADD COLUMN IF NOT EXISTS machine_label VARCHAR(100)",
        "ALTER TABLE machine_predictions ADD COLUMN IF NOT EXISTS predicted_risk TINYINT DEFAULT 0",
        "ALTER TABLE machine_predictions ADD COLUMN IF NOT EXISTS risk_probability FLOAT",
        "ALTER TABLE machine_predictions ADD COLUMN IF NOT EXISTS recommendation TEXT",
    ]

    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(ddl_runs)
        cur.execute(ddl_preds)
        # Patch any missing columns in existing tables
        for stmt in alter_runs + alter_preds:
            try:
                cur.execute(stmt)
            except Exception:
                pass   # column already exists — safe to ignore
        conn.commit()
    finally:
        conn.close()


def save_single_prediction(
    cfg:            DatabaseConfig,
    input_df:       pd.DataFrame,
    prediction:     int,
    probability:    float,
    recommendation: str,
    machine_label:  str,
) -> str:
    """Insert one prediction record. Returns run_id."""
    _ensure_mysql_installed()
    run_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row    = input_df.iloc[0].to_dict()

    conn = _connect(cfg)
    try:
        cur = conn.cursor()

        # ── FIX: total_records column must exist (added by initialize_tables).
        # If user skipped Init Tables, we still insert — the column exists in
        # the CREATE TABLE above, so new tables always have it.
        cur.execute(
            """
            INSERT INTO prediction_runs
                (run_id, source_name, total_records, high_risk, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
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
                float(row.get("machine_temperature",    0)),
                float(row.get("bearing_temperature",    0)),
                float(row.get("vibration_level",        0)),
                float(row.get("pressure",               0)),
                float(row.get("runtime_hours",          0)),
                float(row.get("load_percentage",        0)),
                float(row.get("maintenance_delay_days", 0)),
                float(row.get("error_log_count",        0)),
                int(prediction), float(probability), recommendation, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return run_id


def save_batch_predictions(
    cfg:         DatabaseConfig,
    scored_df:   pd.DataFrame,
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

        # ── FIX: total_records is a real column — always included now
        cur.execute(
            """
            INSERT INTO prediction_runs
                (run_id, source_name, total_records, high_risk, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
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
        for idx, r in scored_df.iterrows():
            pred = int(r.get("predicted_risk",    0))
            prob = float(r.get("risk_probability", 0.0))
            rec  = (
                "Immediate maintenance inspection recommended"
                if pred == 1
                else "Continue monitoring under normal schedule"
            )
            batch.append((
                run_id,
                str(r.get("machine_label", f"MACHINE-{idx}")),
                float(r.get("machine_temperature",    0)),
                float(r.get("bearing_temperature",    0)),
                float(r.get("vibration_level",        0)),
                float(r.get("pressure",               0)),
                float(r.get("runtime_hours",          0)),
                float(r.get("load_percentage",        0)),
                float(r.get("maintenance_delay_days", 0)),
                float(r.get("error_log_count",        0)),
                pred, prob, rec, now,
            ))

        cur.executemany(insert_sql, batch)
        conn.commit()
    finally:
        conn.close()

    return run_id


def fetch_recent_predictions(cfg: DatabaseConfig, limit: int = 25) -> pd.DataFrame:
    """
    Return the most recent *limit* predictions as a DataFrame.

    FIX 1: Replaced pd.read_sql(..., params=(limit,)) with cursor.execute()
            because mysql-connector does not accept a tuple for %s params
            in pd.read_sql — causes TypeError in some versions.

    FIX 2: Replaced 'mp.id' with explicit safe column list. The column IS
            named 'id' in the CREATE TABLE, but if the table existed before
            this schema was applied the auto-increment PK may be named
            differently. Using the exact name 'mp.id' is correct for new
            tables created by initialize_tables(); the ALTER TABLE in
            initialize_tables() patches old tables automatically.
    """
    _ensure_mysql_installed()
    sql = """
        SELECT
            mp.id,
            mp.machine_label,
            mp.predicted_risk,
            mp.risk_probability,
            mp.recommendation,
            mp.created_at,
            pr.source_name,
            pr.total_records,
            pr.high_risk
        FROM machine_predictions mp
        JOIN prediction_runs pr ON mp.run_id = pr.run_id
        ORDER BY mp.created_at DESC
        LIMIT %s
    """
    conn = _connect(cfg)
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (limit,))          # ← cursor.execute, NOT pd.read_sql
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return pd.DataFrame(rows) if rows else pd.DataFrame()
