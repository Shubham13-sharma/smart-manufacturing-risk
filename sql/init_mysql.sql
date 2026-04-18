CREATE DATABASE IF NOT EXISTS smart_manufacturing;
USE smart_manufacturing;

CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id INT AUTO_INCREMENT PRIMARY KEY,
    source_name VARCHAR(255) NOT NULL,
    record_count INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

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
    saved_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_run
        FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
        ON DELETE SET NULL
);

SELECT * FROM prediction_runs;
SELECT * FROM machine_predictions;
