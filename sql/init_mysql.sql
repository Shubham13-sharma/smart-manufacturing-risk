-- ============================================================
-- Smart Manufacturing — Aiven MySQL initialisation script
-- Run this in MySQL Workbench or via the "Init Tables" button
-- ============================================================

-- prediction_runs: one row per save session
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id        VARCHAR(36)   NOT NULL PRIMARY KEY,
    source_name   VARCHAR(255),
    total_records INT           DEFAULT 0,
    high_risk     INT           DEFAULT 0,
    created_at    DATETIME      NOT NULL
);

-- machine_predictions: one row per machine
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
);

-- Patch columns that may be missing in tables created by older schema:
ALTER TABLE prediction_runs
    ADD COLUMN IF NOT EXISTS total_records INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS high_risk     INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS source_name   VARCHAR(255);

ALTER TABLE machine_predictions
    ADD COLUMN IF NOT EXISTS machine_label   VARCHAR(100),
    ADD COLUMN IF NOT EXISTS predicted_risk  TINYINT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS risk_probability FLOAT,
    ADD COLUMN IF NOT EXISTS recommendation  TEXT;
