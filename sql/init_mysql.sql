-- =========================================
-- CLEAN & RECREATE DATABASE TABLES
-- =========================================

-- Drop old incorrect tables (VERY IMPORTANT)
DROP TABLE IF EXISTS machine_predictions;
DROP TABLE IF EXISTS prediction_runs;

-- =========================================
-- TABLE 1: prediction_runs
-- =========================================
CREATE TABLE prediction_runs (
run_id VARCHAR(36) PRIMARY KEY,
source_name VARCHAR(255),
total_records INT DEFAULT 0,
high_risk INT DEFAULT 0,
created_at DATETIME NOT NULL
);

-- =========================================
-- TABLE 2: machine_predictions
-- =========================================
CREATE TABLE machine_predictions (
id BIGINT AUTO_INCREMENT PRIMARY KEY,
run_id VARCHAR(36),

```
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

created_at DATETIME NOT NULL,

CONSTRAINT fk_run
    FOREIGN KEY (run_id)
    REFERENCES prediction_runs(run_id)
    ON DELETE CASCADE
```

);

-- =========================================
-- OPTIONAL: INDEX FOR PERFORMANCE
-- =========================================
CREATE INDEX idx_run_id ON machine_predictions(run_id);
CREATE INDEX idx_created_at ON machine_predictions(created_at);
