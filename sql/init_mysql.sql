CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id VARCHAR(64) NOT NULL PRIMARY KEY,
    source_name VARCHAR(255),
    total_records INT DEFAULT 0,
    record_count INT DEFAULT 0,
    average_risk DOUBLE,
    high_risk INT DEFAULT 0,
    high_risk_count INT DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    saved_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS machine_predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(64),
    machine_label VARCHAR(255),
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
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_run
        FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_run_id ON machine_predictions(run_id);
CREATE INDEX idx_created_at ON machine_predictions(created_at);

ALTER TABLE prediction_runs
    MODIFY COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE prediction_runs
    MODIFY COLUMN saved_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE machine_predictions
    MODIFY COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP;

DESCRIBE prediction_runs;
DESCRIBE machine_predictions;
