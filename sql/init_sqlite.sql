CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id TEXT PRIMARY KEY,
    source_name TEXT,
    total_records INTEGER NOT NULL DEFAULT 0,
    record_count INTEGER NOT NULL DEFAULT 0,
    average_risk REAL,
    high_risk INTEGER NOT NULL DEFAULT 0,
    high_risk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS machine_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    machine_label TEXT,
    machine_temperature REAL,
    bearing_temperature REAL,
    vibration_level REAL,
    pressure REAL,
    runtime_hours REAL,
    load_percentage REAL,
    maintenance_delay_days REAL,
    error_log_count REAL,
    predicted_risk INTEGER DEFAULT 0,
    risk_probability REAL,
    recommendation TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_machine_predictions_run_id
    ON machine_predictions(run_id);

CREATE INDEX IF NOT EXISTS idx_machine_predictions_created_at
    ON machine_predictions(created_at);
