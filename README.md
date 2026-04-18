# Smart Manufacturing Downtime Risk Classification

This project predicts machine downtime risk before a failure happens. It is designed as an interview-ready Industry 4.0 mini-project with an end-to-end ML pipeline, real dataset ingestion, a presentation-ready dashboard, and MySQL-backed prediction storage.

## Problem Statement

In manufacturing, machine breakdowns stop production and increase cost. The goal of this project is to classify whether a machine is at high downtime risk based on sensor and operational signals.

Output:

- `0` -> No risk
- `1` -> High risk

## Features Used

The standardized feature set includes these manufacturing signals:

- `machine_temperature`
- `bearing_temperature`
- `vibration_level`
- `pressure`
- `runtime_hours`
- `load_percentage`
- `maintenance_delay_days`
- `error_log_count`

## Project Structure

```text
F:\HCL PROJECT
|-- app.py
|-- README.md
|-- requirements.txt
|-- scripts
|   |-- generate_sample_data.py
|   `-- train_model.py
`-- src
    `-- downtime_risk
        |-- __init__.py
        |-- data.py
        |-- model.py
        `-- predict.py
```

## ML Pipeline

1. Data preprocessing
   - handle missing values with median imputation
   - scale numeric features
2. Model training
   - Logistic Regression
   - Random Forest
3. Model selection
   - choose the best model using F1 score
4. Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1 score

## Formula You Can Explain

Logistic Regression estimates failure probability as:

`P(Y=1|X) = 1 / (1 + e^-(w0 + w1x1 + ... + wnxn))`

## How To Run

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate sample manufacturing data

```powershell
python scripts\generate_sample_data.py
```

This creates `data\manufacturing_downtime_sample.csv`.

### 3. Train the model

```powershell
python scripts\train_model.py
```

Artifacts saved in `artifacts\`:

- `best_model.joblib`
- `feature_columns.joblib`
- `metrics.json`

### 4. Run the dashboard

```powershell
streamlit run app.py
```

## Real Dataset Loader

You can now train or analyze the project using a real CSV dataset.

Supported input styles include:

- project-native columns such as `machine_temperature`, `vibration_level`, `error_log_count`
- common predictive maintenance labels such as `target`, `failure`, `machine_failure`
- AI4I-style fields such as `Process temperature [K]`, `Air temperature [K]`, `Torque [Nm]`, `Tool wear [min]`

Example training command with a real dataset:

```powershell
python scripts\train_model.py --dataset data\your_real_dataset.csv
```

In the dashboard, choose `Upload real dataset` in the sidebar and upload a CSV for scoring and visual analysis.

## Dashboard Highlights

The Streamlit dashboard now includes:

- executive KPI cards
- live single-machine prediction
- risk distribution view
- top risky machines ranking
- failure trend visualization
- correlation heatmap for sensor analysis
- MySQL database console for saving predictions
- batch save option for scored datasets

## MySQL Workbench Integration

This project can save predictions into MySQL so you can show both frontend and backend integration during your internship review.

Important:

- for local testing, you can still use your own MySQL server
- for deployment on Streamlit Community Cloud, use a remote MySQL host
- do not use `localhost` after deployment because it points to the cloud container, not your laptop

### 1. Open MySQL Workbench

Run the SQL script from:

`sql\init_mysql.sql`

This will create:

- database: `smart_manufacturing`
- table: `prediction_runs`
- table: `machine_predictions`

### 2. Install Python dependencies

```powershell
py -3 -m pip install -r requirements.txt
```

### 3. Train the model

```powershell
py -3 scripts\train_model.py --dataset data\real_dataset.csv
```

### 4. Launch the app

```powershell
py -3 -m streamlit run app.py
```

### 5. Save data to MySQL from the dashboard

Inside the sidebar:

- enter `host`, `port`, `user`, `password`, and `database`
- click `Test DB`
- click `Init Tables`

Inside `Prediction Studio`:

- click `Save Current Prediction` to store one record
- click `Save Full Dataset Batch` to store all scored rows

## Remote MySQL For Deployment

For Streamlit Community Cloud, store your database credentials in secrets instead of hardcoding them.

Create a local file based on:

`\.streamlit\secrets.toml.example`

Example:

```toml
[mysql]
host = "your-remote-mysql-host"
port = 3306
user = "your_mysql_user"
password = "your_mysql_password"
database = "smart_manufacturing"
```

When deploying, paste the same contents into the `Secrets` box under Streamlit Community Cloud `Advanced settings`.

## Standout Extensions

You can present these as next steps:

- Real-time prediction from live sensor feeds
- Alert system when risk crosses a threshold
- Power BI or Streamlit dashboard for plant managers
- Integration with maintenance scheduling systems

## Interview Summary

You can describe the project in one line like this:

"We built a classification system that uses machine sensor data and operational history to predict downtime risk early, helping factories reduce unplanned stoppages and maintenance cost."
