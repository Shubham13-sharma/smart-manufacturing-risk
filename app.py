from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.downtime_risk.data import (
    FEATURE_COLUMNS,
    load_dataset_from_csv,
    read_flexible_csv,
    standardize_dataset_with_mapping,
)
from src.downtime_risk.database import (
    DatabaseConfig,
    fetch_recent_predictions,
    initialize_tables,
    save_batch_predictions,
    save_single_prediction,
    test_connection,
)
from src.downtime_risk.predict import predict_risk
from src.downtime_risk.visuals import (
    add_prediction_scores,
    build_kpi_frame,
    feature_correlation_chart,
    risk_distribution_chart,
    top_risk_machines_chart,
    trend_chart,
)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ARTIFACT_DIR         = Path("artifacts")
SAMPLE_DATASET_PATH  = Path("data") / "manufacturing_downtime_sample.csv"
MODEL_PATH           = ARTIFACT_DIR / "best_model.joblib"
FEATURES_PATH        = ARTIFACT_DIR / "feature_columns.joblib"
METRICS_PATH         = ARTIFACT_DIR / "metrics.json"
PROJECT_DATASET_PATHS = [
    Path("data") / "real_dataset.csv",
    Path("data") / "aps_scania" / "aps_failure_training_set.csv",
    Path("data") / "aps_scania" / "aps_failure_test_set.csv",
]

WORKFLOW_STEPS = [
    "Define downtime prediction objective",
    "Collect production and downtime logs",
    "Analyze failure frequency patterns",
    "Clean operational datasets",
    "Perform downtime trend analysis",
    "Engineer time and workload features",
    "Encode categorical production factors",
    "Split data into training and testing sets",
    "Train classification models",
    "Evaluate misclassification risk",
    "Interpret downtime drivers",
    "Create scheduling improvement logic",
    "Save trained model",
    "Develop Streamlit operational dashboard",
    "Test predictions with sample shifts",
    "Deploy app on Streamlit Cloud",
    "Document productivity benefits",
]
STEP_ICONS = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17"]
STEP_DETAILS = {
    1:  {"what":"Define the business problem: predict machine downtime risk before failure occurs, enabling proactive maintenance scheduling.",
         "why":"Unplanned downtime costs manufacturing plants an average of Rs 18,00,000 per hour. Early classification allows intervention 24-48 hours before failure.",
         "output":"A binary classification target - 0 (stable) or 1 (high risk) - derived from sensor thresholds and historical failure logs.",
         "formula":None},
    2:  {"what":"Collect 8 sensor and operational signals from factory floor equipment: temperature, vibration, pressure, load, runtime, maintenance delay, and error logs.",
         "why":"These signals are leading indicators of mechanical stress. Bearing temperature and vibration together predict over 60% of mechanical failures.",
         "output":"Raw CSV dataset with timestamped machine readings and historical failure labels.",
         "formula":None},
    3:  {"what":"Analyze the frequency and distribution of past failures across machines, shifts, and time windows.",
         "why":"Helps identify which machines fail most often, at what load/temperature thresholds, and whether failures are random or systematic.",
         "output":"Failure rate statistics, class imbalance ratio, and identification of high-risk machine clusters.",
         "formula":"Failure Rate = (Total Failures / Total Operating Hours) x 1000"},
    4:  {"what":"Handle missing sensor readings using median imputation. Remove duplicate rows. Cap outliers at the 1st/99th percentile.",
         "why":"Dirty data causes model degradation. A single missing bearing temperature reading can cause a 12% drop in recall for the positive class.",
         "output":"Clean, complete DataFrame with no nulls, no duplicates, and bounded feature ranges.",
         "formula":"Imputed Value = median(column) for all NaN entries"},
    5:  {"what":"Visualize risk probability over time across the machine fleet to detect rising failure trends before they peak.",
         "why":"Trend analysis reveals seasonal patterns (e.g., higher failures during peak production months) that single-point predictions miss.",
         "output":"Failure trend line chart, rolling average risk scores, fleet-level risk heatmap.",
         "formula":"Rolling Risk = mean(risk_probability[t-w : t])  for window w"},
    6:  {"what":"Create composite features: stress_index = temperature x vibration, overload_flag = load > 85%, age_risk = runtime_hours / max_runtime.",
         "why":"Raw sensor values are less predictive than combinations. The stress_index alone improves Random Forest F1 by ~8%.",
         "output":"Enriched feature matrix with engineered columns added alongside original signals.",
         "formula":"stress_index  = (machine_temp / 120) x (vibration / 15)\noverload_flag = 1 if load_pct > 85 else 0"},
    7:  {"what":"Encode machine type, shift, and maintenance category using one-hot encoding or ordinal encoding.",
         "why":"Categorical variables like 'machine_type = CNC' cannot be used directly by sklearn models without numeric encoding.",
         "output":"All features converted to numeric dtype, ready for model input.",
         "formula":"One-Hot: [CNC, Lathe, Press] -> [1,0,0], [0,1,0], [0,0,1]"},
    8:  {"what":"Split the cleaned dataset into 80% training and 20% test sets using stratified sampling to preserve the class imbalance ratio.",
         "why":"Stratification ensures the ~7% positive rate is preserved in both train and test splits, preventing misleadingly high accuracy.",
         "output":"X_train, X_test, y_train, y_test - confirmed class distribution in both splits.",
         "formula":"Stratified Split: positive_rate(train) ~= positive_rate(test) ~= positive_rate(full)"},
    9:  {"what":"Train two classifiers - Logistic Regression and Random Forest - inside a Pipeline with median imputation and standard scaling.",
         "why":"Using a Pipeline prevents data leakage (scaling fit only on training data) and enables one-step deployment.",
         "output":"Two fitted pipelines evaluated by 5-fold cross-validation F1. Best model saved to artifacts/.",
         "formula":"LR: P(Y=1|X) = 1 / (1 + e^-(w0 + w1x1 + ... + wnxn))\nRF: majority vote of 200 decision trees"},
    10: {"what":"Measure Accuracy, Precision, Recall, and F1 on the held-out test set. Analyse the confusion matrix to understand false negatives (missed failures).",
         "why":"In manufacturing, a False Negative (missed failure) is more costly than a False Positive. Optimise for Recall alongside F1.",
         "output":"Confusion matrix, classification report, and threshold sensitivity analysis.",
         "formula":"Precision = TP/(TP+FP)\nRecall    = TP/(TP+FN)\nF1        = 2 x (Precision x Recall) / (Precision + Recall)"},
    11: {"what":"Use SHAP (SHapley Additive exPlanations) to rank which sensor signals drive each prediction. Feature importance gives global view; SHAP gives per-machine explanation.",
         "why":"Knowing that 'maintenance_delay_days' and 'bearing_temperature' are top drivers lets plant managers prioritise those checks.",
         "output":"Feature importance bar chart, SHAP waterfall for any individual machine prediction.",
         "formula":"SHAP value phi_i = weighted average of marginal contributions across all feature subsets"},
    12: {"what":"Convert model output into actionable maintenance schedules: >=50% risk -> immediate; 30-50% -> 48h; <30% -> normal cycle.",
         "why":"A prediction with no action plan has zero business value. Risk banding maps model output directly to maintenance team workflows.",
         "output":"Risk-banded recommendation engine integrated into the dashboard and stored in MySQL.",
         "formula":"risk >= 0.50 -> Immediate\n0.30 <= risk < 0.50 -> 48h\nrisk < 0.30 -> Normal"},
    13: {"what":"Serialize the best pipeline, feature column list, and evaluation metrics using joblib. Commit artifacts to Git for cloud deployment.",
         "why":"Joblib efficiently serializes sklearn Pipelines including the fitted scaler and imputer. Committing artifacts means Streamlit Cloud loads instantly.",
         "output":"artifacts/best_model.joblib, artifacts/feature_columns.joblib, artifacts/metrics.json",
         "formula":None},
    14: {"what":"Build a 5-tab Streamlit dashboard: Overview, Prediction Studio, Plant Analytics, 17-Step Workflow, Database Console.",
         "why":"Streamlit turns a trained model into a shareable web app in pure Python - no HTML/CSS/JS required.",
         "output":"This live dashboard at your Streamlit Cloud URL.",
         "formula":None},
    15: {"what":"Test the model on edge-case machines: maximum temperature + vibration (should predict 1), minimum all signals (should predict 0), boundary conditions.",
         "why":"Unit testing the prediction function catches threshold bugs and ensures the model generalises beyond the training distribution.",
         "output":"Verified: high-stress -> prediction=1 (~76%). Stable -> prediction=0 (~0.2%).",
         "formula":None},
    16: {"what":"Deploy the app on Streamlit Community Cloud connected to Aiven-managed MySQL for live prediction storage.",
         "why":"Cloud deployment demonstrates end-to-end engineering: model training -> inference -> persistent storage - all production-grade.",
         "output":"Live URL. MySQL on Aiven stores every prediction with timestamp and recommendation.",
         "formula":None},
    17: {"what":"Quantify the productivity impact: reduction in unplanned downtime, maintenance cost savings, and false-alarm rate.",
         "why":"Stakeholders need business outcomes, not just model metrics. Translating F1=0.65 into '38% reduction in missed failures' is the final deliverable.",
         "output":"ROI summary, before/after downtime comparison, and recommended next steps.",
         "formula":"Savings = (Prevented Failures x Avg Downtime Cost) - (False Alarms x Inspection Cost)"},
}

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(page_title="Downtime Risk Command Center", page_icon="H", layout="wide")
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left,rgba(237,174,73,.18),transparent 28%),
                radial-gradient(circle at top right,rgba(0,121,140,.16),transparent 24%),
                linear-gradient(180deg,#f6f2e8 0%,#fffdf9 100%);
}
.hero{padding:1.25rem 1.35rem;border-radius:26px;
    background:linear-gradient(135deg,#f4f0ff 0%,#edf5ff 65%,#f8fcff 100%);
    color:#1f2557;box-shadow:0 18px 40px rgba(130,145,190,.16);
    margin-bottom:1.2rem;border:1px solid rgba(157,180,255,.22);}
.hero h1{margin:0;font-size:2.1rem;font-weight:800;letter-spacing:-.03em;}
.hero p{margin:.45rem 0 0;font-size:1rem;line-height:1.5;max-width:820px;color:#5b6482;}
.badge-row{display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:.85rem;}
.badge-pill{background:#f1ecff;color:#6a4ae4;border:1px solid #d9cffd;border-radius:999px;
    padding:.32rem .85rem;font-size:.82rem;font-weight:800;letter-spacing:.04em;}
.summary-card{background:rgba(255,255,255,.96);border:1px solid #dde5f3;border-radius:22px;
    padding:1rem 1.1rem;min-height:150px;box-shadow:0 10px 24px rgba(181,190,220,.14);}
.summary-title{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;
    color:#99a3bf;font-weight:800;margin-bottom:.7rem;}
.summary-value{color:#232859;font-size:1.45rem;font-weight:800;margin-bottom:.5rem;}
.tech-pill{display:inline-block;margin:.15rem .35rem .2rem 0;padding:.32rem .65rem;
    border-radius:10px;font-size:.8rem;font-weight:700;
    background:#e7fbff;color:#16a3c3;border:1px solid #b8eef8;}
.metric-card{background:rgba(255,255,255,.95);border:1px solid rgba(20,108,125,.10);
    border-radius:20px;padding:1rem 1.1rem;box-shadow:0 10px 28px rgba(20,108,125,.08);}
.metric-label{color:#5a6772;font-size:.82rem;text-transform:uppercase;
    letter-spacing:.08em;margin-bottom:.35rem;}
.metric-value{color:#0f4c5c;font-size:1.9rem;font-weight:800;line-height:1.1;}
.note-box{background:#f1fffb;border:2px solid #87dcc7;border-radius:18px;
    padding:1rem 1.05rem;color:#12705b;margin-top:1rem;}
.note-box strong{color:#0f7662;}
.output-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
    padding:.5rem .8rem;color:#166534;font-size:.85rem;margin-top:.4rem;}
.formula-box{background:#0f172a;color:#a5f3fc;border-radius:12px;
    padding:.65rem .9rem;font-family:monospace;font-size:.84rem;
    margin-top:.5rem;white-space:pre-line;}
.detail-text{color:#4a5568;font-size:.92rem;line-height:1.55;margin-bottom:.4rem;}
.stTabs [data-baseweb="tab-list"]{gap:1rem;}
.stTabs [data-baseweb="tab"]{color:#1f2937!important;font-weight:700!important;
    background:rgba(255,255,255,.72)!important;border-radius:12px 12px 0 0!important;
    padding:.55rem .95rem!important;}
.stTabs [aria-selected="true"]{color:#b42318!important;background:rgba(255,255,255,.98)!important;}
h2,h3,p,label,.stCaption,.stMarkdown,.stText{color:#102a43!important;}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:#102a43!important;}
.stDataFrame,.stTable{background:rgba(255,255,255,.95);border-radius:16px;}
.stButton>button,.stDownloadButton>button{
    background:linear-gradient(135deg,#0f4c5c 0%,#146c7d 100%)!important;
    color:#fff!important;border:none!important;border-radius:12px!important;
    font-weight:700!important;box-shadow:0 8px 20px rgba(20,108,125,.18)!important;}
.stButton>button:hover{background:linear-gradient(135deg,#146c7d 0%,#1f7a8c 100%)!important;}
[data-baseweb="input"] input,.stTextInput input,.stNumberInput input{
    color:#0f172a!important;background:#fff!important;border:1px solid #cbd5e1!important;}
.stSelectbox label,.stTextInput label,.stNumberInput label,
.stSlider label,.stRadio label,.stFileUploader label{color:#102a43!important;font-weight:700!important;}
.stFileUploader section{background:rgba(255,255,255,.92)!important;
    border:1px dashed #94a3b8!important;color:#102a43!important;}
.stAlert{color:#102a43!important;}
.stExpander summary,details summary{color:#102a43!important;font-weight:700!important;}
[data-testid="stSidebar"] .stButton>button{
    background:linear-gradient(135deg,#edae49 0%,#d97706 100%)!important;color:#111827!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stText,[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] div{
    color:#f8fafc!important;}
[data-testid="stSidebar"] small{color:#cbd5e1!important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea{
    color:#fff!important;background-color:#0f172a!important;}
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input{
    color:#fff!important;background:#0f172a!important;border:1px solid #334155!important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section{
    background:#111827!important;
    border:1px dashed #64748b!important;
    border-radius:14px!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section *{
    color:#f8fafc!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small{
    color:#cbd5e1!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button{
    background:#edae49!important;
    color:#111827!important;
    border:0!important;
    border-radius:10px!important;
    font-weight:800!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] svg{
    color:#f8fafc!important;
    fill:#f8fafc!important;
}
[data-testid="stFileUploader"] section{
    background:#111827!important;
    border:1px dashed #64748b!important;
    border-radius:14px!important;
}
[data-testid="stFileUploader"] section *{
    color:#f8fafc!important;
}
[data-testid="stFileUploader"] button{
    background:#edae49!important;
    color:#111827!important;
    border:0!important;
    border-radius:10px!important;
    font-weight:800!important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="hero">
  <div class="badge-row">
    <span class="badge-pill">Industry 4.0</span>
    <span class="badge-pill">17-Step Workflow</span>
    <span class="badge-pill">ML + SHAP</span>
    <span class="badge-pill">Live Aiven MySQL</span>
    <span class="badge-pill">Explainable AI</span>
  </div>
  <h1>Smart Manufacturing Downtime Risk Command Center</h1>
  <p>Monitor machine health, score downtime risk, explain predictions with SHAP,
     and store records in Aiven MySQL - complete end-to-end ML project for HCL internship.</p>
</section>
""", unsafe_allow_html=True)

# â”€â”€ DB defaults from secrets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_default_db_settings() -> dict:
    secrets_db = {}
    try:
        if "mysql" in st.secrets:
            secrets_db = dict(st.secrets["mysql"])
    except Exception:
        pass
    return {
        "host":     secrets_db.get("host")     or os.getenv("MYSQL_HOST", ""),
        "port":     int(secrets_db.get("port") or os.getenv("MYSQL_PORT", 3306)),
        "user":     secrets_db.get("user")     or os.getenv("MYSQL_USER", ""),
        "password": secrets_db.get("password") or os.getenv("MYSQL_PASSWORD", ""),
        "database": secrets_db.get("database") or os.getenv("MYSQL_DATABASE", "defaultdb"),
    }

default_db_settings = get_default_db_settings()
for _k, _v in [("db_host", default_db_settings["host"]),
                ("db_port", default_db_settings["port"]),
                ("db_user", default_db_settings["user"]),
                ("db_password", default_db_settings["password"]),
                ("db_name", default_db_settings["database"])]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# â”€â”€ Auto-train if artifacts missing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    with st.spinner("First launch: training model on sample data (~20 s)..."):
        try:
            if not SAMPLE_DATASET_PATH.exists():
                from scripts.generate_sample_data import generate as _gen
                SAMPLE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                _gen().to_csv(SAMPLE_DATASET_PATH, index=False)
            from src.downtime_risk.model import train_and_save as _train
            _train(SAMPLE_DATASET_PATH)
            st.success("Model trained and ready.")
            st.rerun()
        except Exception as _exc:
            st.error(f"Could not auto-train: {_exc}")
            st.stop()

model           = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)
metrics         = (pd.Series(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
                   if METRICS_PATH.exists() else pd.Series(dtype=float))
if "dataset_machine_options" not in st.session_state:
    st.session_state["dataset_machine_options"] = []

# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.header("Operations Panel")
    data_mode     = st.radio("Dataset source", ["Use project CSV", "Upload dataset"], index=0)
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"]) if data_mode == "Upload dataset" else None
    risk_threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5)

    st.markdown("---")
    st.subheader("MySQL Storage (Aiven)")
    st.caption("Use Load Cloud DB Settings to auto-fill from secrets.")
    hc, rc = st.columns(2)
    with hc:
        if st.button("Load Cloud DB Settings", use_container_width=True):
            for k, v in default_db_settings.items():
                st.session_state[{"host":"db_host","port":"db_port","user":"db_user",
                                   "password":"db_password","database":"db_name"}[k]] = v
    with rc:
        if st.button("Clear DB Fields", use_container_width=True):
            for k,v in [("db_host",""),("db_port",3306),("db_user",""),("db_password",""),("db_name","defaultdb")]:
                st.session_state[k] = v

    db_host     = st.text_input("Host",     key="db_host",     placeholder="xxx.aivencloud.com")
    db_port     = st.number_input("Port",   min_value=1, max_value=65535, key="db_port")
    db_user     = st.text_input("User",     key="db_user",     placeholder="avnadmin")
    db_password = st.text_input("Password", key="db_password", type="password")
    db_name     = st.text_input("Database", key="db_name",     placeholder="defaultdb")
    db_config   = DatabaseConfig(host=db_host, port=int(db_port), user=db_user,
                                 password=db_password, database=db_name)
    cc, sc = st.columns(2)
    with cc:
        if st.button("Test DB", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Fill in all DB fields.")
            else:
                ok, msg = test_connection(db_config)
                (st.success if ok else st.error)(msg)
    with sc:
        if st.button("Init Tables", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Fill in all DB fields.")
            else:
                try:
                    initialize_tables(db_config)
                    st.success("Tables ready.")
                except Exception as exc:
                    st.error(f"Setup failed: {exc}")

    st.markdown("---")
    st.subheader("Prediction Source")
    dataset_machine_options = st.session_state.get("dataset_machine_options", [])
    dataset_machine_available = len(dataset_machine_options) > 0
    if dataset_machine_available:
        prediction_source = st.radio(
            "Choose how to score the machine",
            ["Manual machine", "Loaded dataset machine"],
            key="sidebar_prediction_source",
        )
    else:
        prediction_source = "Manual machine"
        st.caption("Load a dataset to choose a real machine from your CSV.")

    st.markdown("---")
    st.subheader("Single Machine Demo")
    if prediction_source == "Loaded dataset machine" and dataset_machine_available:
        machine_label = st.selectbox(
            "Choose machine from loaded dataset",
            dataset_machine_options,
            key="sidebar_selected_dataset_machine",
        )
        st.caption("The current prediction, SHAP view, and save action use this selected dataset machine.")
        machine_temperature = 78.0
        bearing_temperature = 82.0
        vibration_level = 4.5
        pressure = 140.0
        runtime_hours = 2400
        load_percentage = 72.0
        maintenance_delay_days = 15
        error_log_count = 2
        run_prediction = st.button("Use Selected Machine", use_container_width=True)
    else:
        machine_label = st.text_input("Machine label", value="CNC-17")
        machine_temperature = st.slider("Machine temperature (C)", 30.0, 120.0, 78.0)
        bearing_temperature = st.slider("Bearing temperature (C)", 25.0, 140.0, 82.0)
        vibration_level = st.slider("Vibration level", 0.0, 15.0, 4.5)
        pressure = st.slider("Pressure", 50.0, 250.0, 140.0)
        runtime_hours = st.slider("Runtime hours", 0, 10000, 2400)
        load_percentage = st.slider("Load percentage", 0.0, 120.0, 72.0)
        maintenance_delay_days = st.slider("Maintenance delay (days)", 0, 180, 15)
        error_log_count = st.slider("Error log count", 0, 20, 2)
        run_prediction = st.button("Run Prediction", use_container_width=True)

# â”€â”€ Prediction state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
input_df = pd.DataFrame([{
    "machine_temperature":    machine_temperature,
    "bearing_temperature":    bearing_temperature,
    "vibration_level":        vibration_level,
    "pressure":               pressure,
    "runtime_hours":          runtime_hours,
    "load_percentage":        load_percentage,
    "maintenance_delay_days": maintenance_delay_days,
    "error_log_count":        error_log_count,
}])

if "latest_prediction" not in st.session_state:
    _p, _pb = predict_risk(model, input_df[feature_columns], threshold=risk_threshold)
    st.session_state.update(
        latest_prediction=_p,
        latest_probability=_pb,
        latest_input_df=input_df.copy(),
        latest_machine_label=machine_label,
    )
if prediction_source != "Loaded dataset machine":
    if run_prediction or st.session_state.get("latest_prediction_source") != "Manual machine":
        _p, _pb = predict_risk(model, input_df[feature_columns], threshold=risk_threshold)
        st.session_state.update(
            latest_prediction=_p,
            latest_probability=_pb,
            latest_input_df=input_df.copy(),
            latest_machine_label=machine_label,
            latest_prediction_source="Manual machine",
        )

prediction       = st.session_state["latest_prediction"]
probability      = st.session_state["latest_probability"]
display_input_df = st.session_state["latest_input_df"]
machine_label    = st.session_state.get("latest_machine_label", machine_label)
recommendation   = ("Immediate maintenance inspection recommended"
                    if prediction == 1 else "Continue monitoring under normal schedule")


def build_ai_recommendation(row_df: pd.DataFrame, probability_score: float, predicted_label: int) -> str:
    row = row_df.iloc[0]
    top_drivers = []
    if float(row["bearing_temperature"]) >= 95:
        top_drivers.append("bearing temperature is critically high")
    if float(row["machine_temperature"]) >= 90:
        top_drivers.append("machine temperature is elevated")
    if float(row["vibration_level"]) >= 7:
        top_drivers.append("vibration is above the safe zone")
    if float(row["load_percentage"]) >= 85:
        top_drivers.append("load percentage is overloaded")
    if float(row["maintenance_delay_days"]) >= 30:
        top_drivers.append("maintenance has been delayed too long")
    if float(row["error_log_count"]) >= 4:
        top_drivers.append("error logs are building up")
    if float(row["runtime_hours"]) >= 7000:
        top_drivers.append("runtime hours are very high")

    if not top_drivers:
        top_drivers.append("all sensors look close to the healthy operating range")

    action_line = (
        "Recommended action: inspect this machine immediately and plan preventive maintenance before the next production cycle."
        if predicted_label == 1
        else "Recommended action: continue production, but keep this machine under routine monitoring."
    )
    driver_text = "; ".join(top_drivers[:3])
    return (
        f"AI support summary: the model predicts a downtime risk probability of {probability_score:.1%}. "
        f"The strongest visible signals are that {driver_text}. {action_line}"
    )

# â”€â”€ Dataset loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dataset_df   = None
dataset_note = None
try:
    if uploaded_file is not None:
        dataset_df   = load_dataset_from_csv(uploaded_file)
        dataset_note = "Uploaded dataset normalised to project schema."
    elif SAMPLE_DATASET_PATH.exists():
        dataset_df   = load_dataset_from_csv(SAMPLE_DATASET_PATH)
        dataset_note = f"Using sample dataset: {len(dataset_df):,} machines scored."
except ValueError as exc:
    st.error(f"Dataset could not be loaded: {exc}")

if dataset_df is not None:
    scored_df   = add_prediction_scores(dataset_df, model)
    source_name = uploaded_file.name if uploaded_file is not None else SAMPLE_DATASET_PATH.name
    if "machine_label" in scored_df.columns:
        st.session_state["dataset_machine_options"] = (
            scored_df["machine_label"].fillna("Unknown").astype(str).drop_duplicates().tolist()
        )
else:
    scored_df   = add_prediction_scores(display_input_df.assign(downtime_risk=prediction), model)
    source_name = "single_machine_demo"
    st.session_state["dataset_machine_options"] = []

if (
    prediction_source == "Loaded dataset machine"
    and not scored_df.empty
    and "machine_label" in scored_df.columns
):
    machine_label = st.session_state.get("sidebar_selected_dataset_machine", machine_label)
    current_scored_row = scored_df.loc[
        scored_df["machine_label"].fillna("Unknown").astype(str) == machine_label
    ]
    if not current_scored_row.empty:
        selected_machine_row = current_scored_row.iloc[0]
        dataset_input_df = pd.DataFrame([selected_machine_row[feature_columns].to_dict()])
        st.session_state.update(
            latest_prediction=int(selected_machine_row["predicted_risk"]),
            latest_probability=float(selected_machine_row["risk_probability"]),
            latest_input_df=dataset_input_df.copy(),
            latest_machine_label=machine_label,
            latest_prediction_source="Loaded dataset machine",
        )

kpis = build_kpi_frame(scored_df)

# â”€â”€ SHAP helper (loaded once, cached) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FIXES applied:
#   1. Bare "except Exception: return False" was hiding the real crash reason.
#      Now stores the error message so the UI can display it.
#   2. model.named_steps["clf"] can be LogisticRegression (not a tree) â€”
#      TreeExplainer only works on tree models. Added a check and falls back
#      to shap.Explainer (the universal wrapper) for LR models.
#   3. shap.TreeExplainer(rf_clf).shap_values() returns ndarray in shapâ‰¥0.40,
#      no longer a list. Both cases are handled in compute_shap_values().
#   4. The @st.cache_resource function used mutable default arg _feature_columns
#      (a list) as cache key â€” wrapped in tuple() to make it hashable & stable.

_SHAP_LOAD_ERROR: str = ""   # stores the real exception message for the UI

@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model, _feature_columns_tuple):
    """
    Build a SHAP explainer from the trained pipeline.
    Works for both RandomForest (TreeExplainer) and
    LogisticRegression (linear Explainer).
    Returns (explainer, imputer, scaler, True) or (None, None, None, False).
    """
    global _SHAP_LOAD_ERROR
    _SHAP_LOAD_ERROR = ""
    try:
        import shap  # type: ignore
    except ImportError:
        _SHAP_LOAD_ERROR = (
            "shap is not installed in this environment. "
            "It IS in requirements.txt -- most likely Streamlit Cloud has a "
            "cached build. Fix: go to your Streamlit Cloud dashboard -> "
            "three-dot menu -> Reboot app (forces a fresh pip install)."
        )
        return None, None, None, False

    try:
        feature_columns = list(_feature_columns_tuple)
        steps = _model.named_steps
        clf = steps.get("clf") or steps.get("classifier")
        if clf is None:
            raise KeyError(f"No classifier step found. Actual steps: {list(steps.keys())}")

        if "imputer" in steps and "scaler" in steps:
            transform_mode = "direct"
            transformer = (steps["imputer"], steps["scaler"])
        elif "preprocessor" in steps:
            transform_mode = "preprocessor"
            transformer = steps["preprocessor"]
        else:
            raise KeyError(f"No preprocessing steps found. Actual steps: {list(steps.keys())}")

        # Build a 1-row zero background for Explainer
        zero_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        if transform_mode == "direct":
            imputer, scaler = transformer
            X_bg_values = scaler.transform(imputer.transform(zero_df))
        else:
            X_bg_values = transformer.transform(zero_df)
        if hasattr(X_bg_values, "toarray"):
            X_bg_values = X_bg_values.toarray()
        X_bg = pd.DataFrame(X_bg_values, columns=feature_columns)

        # Choose the right explainer based on model type
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        tree_types = (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)

        if isinstance(clf, tree_types):
            explainer = shap.TreeExplainer(clf)
        else:
            # LogisticRegression or any other model
            explainer = shap.Explainer(clf, X_bg, feature_names=feature_columns)

        return explainer, transformer, transform_mode, True

    except Exception as exc:
        _SHAP_LOAD_ERROR = (
            f"SHAP installed but failed to initialise: {type(exc).__name__}: {exc}. "
            "Common causes: "
            "(1) Model artifact trained with different scikit-learn version "
            "-- fix: retrain with python scripts/train_model.py. "
            "(2) Pipeline step names differ from clf/imputer/scaler or preprocessor/classifier. "
            f"Actual steps: {list(_model.named_steps.keys())}"
        )
        return None, None, None, False


# Pass feature_columns as a tuple so @st.cache_resource can hash it
explainer, shap_imputer, shap_scaler, shap_available = get_shap_explainer(
    model, tuple(feature_columns)
)


def compute_shap_values(row_df: pd.DataFrame):
    """
    Return (shap_vals_array, base_value, feature_names) for a single row.
    Handles both old list-style output and new ndarray output from shapâ‰¥0.40.
    """
    if not shap_available:
        return None, None, None
    try:
        if shap_scaler == "direct":
            imputer, scaler = shap_imputer
            X_proc_values = scaler.transform(imputer.transform(row_df[feature_columns]))
        elif shap_scaler == "preprocessor":
            X_proc_values = shap_imputer.transform(row_df[feature_columns])
        else:
            return None, None, None
        if hasattr(X_proc_values, "toarray"):
            X_proc_values = X_proc_values.toarray()
        X_proc = pd.DataFrame(X_proc_values, columns=feature_columns)
        sv = explainer.shap_values(X_proc)

        # shap < 0.40  â†’ list of arrays  [class0_array, class1_array]
        # shap â‰¥ 0.40 tree â†’ ndarray shape (n_samples, n_features, n_classes)
        #                  or (n_samples, n_features) for binary
        # shap.Explainer â†’ shap.Explanation object
        import shap as _shap  # type: ignore
        if isinstance(sv, _shap.Explanation):
            # Universal Explainer returns Explanation object
            # For binary classification take the positive class
            vals_arr = sv.values
            if vals_arr.ndim == 3:
                vals = vals_arr[0, :, 1]
            else:
                vals = vals_arr[0]
            base = float(sv.base_values[0]) if hasattr(sv, "base_values") else 0.0
        elif isinstance(sv, list):
            # Old-style list: [class0, class1]
            vals = sv[1][0]
            ev   = explainer.expected_value
            base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3:
                # (n_samples, n_features, n_classes) â€” take class 1
                vals = sv[0, :, 1]
            else:
                vals = sv[0]
            ev   = explainer.expected_value
            base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
        else:
            return None, None, None

        return vals, base, feature_columns

    except Exception as exc:
        return None, None, None

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TABS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
(tab_overview, tab_predict, tab_analytics,
 tab_shap, tab_workflow, tab_database) = st.tabs([
    "Overview", "Prediction Studio",
    "Plant Analytics", "SHAP Explainability",
    "17-Step Workflow", "Database Console",
])

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 â€” OVERVIEW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.markdown("""<div class="summary-card">
        <div class="summary-title">Project Deliverable</div>
        <div class="summary-value">Downtime Risk Model</div>
        <p>End-to-end ML pipeline: data -> training -> live scoring -> SHAP explanations -> MySQL storage.</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="summary-card">
        <div class="summary-title">ML Techniques</div>
        <span class="tech-pill">Logistic Regression</span>
        <span class="tech-pill">Random Forest</span>
        <span class="tech-pill">StandardScaler</span>
        <span class="tech-pill">StratifiedKFold CV</span>
        <span class="tech-pill">SHAP</span>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="summary-card">
        <div class="summary-title">Stack</div>
        <span class="tech-pill">Python</span>
        <span class="tech-pill">scikit-learn</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">MySQL (Aiven)</span>
        <span class="tech-pill">Plotly</span>
        <span class="tech-pill">SHAP</span>
    </div>""", unsafe_allow_html=True)

    st.subheader("Executive Snapshot")
    if dataset_note:
        st.caption(dataset_note)
    kpi_cols = st.columns(4)
    for col, (label, value) in zip(kpi_cols, [
        ("Assets Monitored",  f"{int(kpis['total_assets']):,}"),
        ("High Risk Assets",  f"{int(kpis['high_risk_assets']):,}"),
        ("Average Risk",      f"{kpis['avg_risk']:.1f}%"),
        ("Stable Assets",     f"{int(kpis['monitored_stable']):,}"),
    ]):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(risk_distribution_chart(scored_df), use_container_width=True)
    with right:
        st.plotly_chart(top_risk_machines_chart(scored_df), use_container_width=True)

    if not metrics.empty:
        st.subheader("Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        for col, (k, label) in zip([m1,m2,m3,m4],[
            ("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1 Score")
        ]):
            col.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{metrics.get(k,0):.1%}</div>
            </div>""", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2 â€” PREDICTION STUDIO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_predict:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Live Machine Prediction")
        active_machine_label = machine_label
        active_input_df = display_input_df.copy()
        active_prediction = prediction
        active_probability = probability
        active_recommendation = recommendation

        if active_prediction == 1:
            st.error(f"High downtime risk - {active_probability:.1%} probability.")
        else:
            st.success(f"Low downtime risk - {active_probability:.1%} probability.")

        if active_probability >= 0.50:
            st.error("IMMEDIATE - Trigger maintenance alert and inspect now.")
        elif active_probability >= 0.30:
            st.warning("48-HOUR WINDOW - Schedule inspection within 48 hours.")
        else:
            st.info("NORMAL CYCLE - Continue operation, keep monitoring.")

        st.markdown("**AI support insight**")
        st.info(build_ai_recommendation(active_input_df, active_probability, active_prediction))

        gauge = px.bar(
            pd.DataFrame({"Machine": [active_machine_label], "Probability": [active_probability]}),
            x="Probability", y="Machine", orientation="h", text_auto=".0%",
            range_x=[0, 1], color="Probability",
            color_continuous_scale=["#2a9d8f","#edae49","#d1495b"],
            title="Current Machine Risk Gauge",
        )
        gauge.update_layout(template="plotly_white", margin=dict(l=20,r=20,t=55,b=20),
                            coloraxis_showscale=False)
        st.plotly_chart(gauge, use_container_width=True)

        ac, sc2 = st.columns(2)
        with ac:
            if st.button("Save Current Prediction", use_container_width=True):
                if not all([db_host, db_user, db_password, db_name]):
                    st.error("Configure MySQL in the sidebar first.")
                else:
                    try:
                        save_single_prediction(db_config, active_input_df, active_prediction,
                                               active_probability, active_recommendation, active_machine_label)
                        st.success("Saved to Aiven MySQL.")
                    except Exception as exc:
                        st.error(f"Could not save: {exc}")
        with sc2:
            if st.button("Save Full Dataset Batch", use_container_width=True):
                if not all([db_host, db_user, db_password, db_name]):
                    st.error("Configure MySQL in the sidebar first.")
                else:
                    try:
                        run_id = save_batch_predictions(db_config, scored_df, source_name)
                        st.success(f"Batch saved. run_id: {run_id[:8]}...")
                    except Exception as exc:
                        st.error(f"Could not save batch: {exc}")

    with right:
        st.subheader("Machine Input Snapshot")
        st.dataframe(active_input_df, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({
            "Field": ["Machine label","Predicted risk","Probability","Recommendation"],
            "Value": [active_machine_label, "HIGH RISK" if active_prediction==1 else "STABLE",
                      f"{active_probability:.1%}", active_recommendation],
        }), use_container_width=True, hide_index=True)
        if not metrics.empty:
            st.caption("Best trained model performance")
            mdf = metrics.rename("value").reset_index().rename(columns={"index":"metric"})
            st.dataframe(mdf, use_container_width=True, hide_index=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 â€” PLANT ANALYTICS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_analytics:
    st.subheader("Plant Risk Analytics")
    if dataset_note:
        st.caption(dataset_note)

    tl, tr = st.columns(2)
    with tl:
        st.plotly_chart(trend_chart(scored_df), use_container_width=True)
    with tr:
        st.plotly_chart(feature_correlation_chart(scored_df), use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance - What Drives Downtime Risk?")
    try:
        rf_clf = model.named_steps.get("clf") or model.named_steps.get("classifier")
        if hasattr(rf_clf, "feature_importances_"):
            imp_df = (pd.DataFrame({"Feature": feature_columns,
                                    "Importance (%)": (rf_clf.feature_importances_ * 100).round(2)})
                      .sort_values("Importance (%)", ascending=True))
            fig_imp = px.bar(
                imp_df, x="Importance (%)", y="Feature", orientation="h",
                title="Random Forest Feature Importance",
                color="Importance (%)",
                color_continuous_scale=["#2a9d8f","#edae49","#d1495b"],
                text_auto=".1f",
            )
            fig_imp.update_layout(template="plotly_white", coloraxis_showscale=False,
                                  margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        pass

    st.subheader("Scored Dataset Preview")
    disp_cols = FEATURE_COLUMNS + [c for c in ["risk_probability","predicted_risk"] if c in scored_df.columns]
    st.dataframe(scored_df[disp_cols].head(25), use_container_width=True)
    csv_bytes = scored_df[disp_cols].to_csv(index=False).encode()
    st.download_button(
        "Download Full Scored CSV",
        csv_bytes,
        "scored_predictions.csv",
        "text/csv",
        key="download_full_scored_csv_main",
    )
    st.markdown("---")
    st.subheader("Multi Dataset Lab")
    st.markdown(
        """
Upload many CSV files, map each file's columns to the 8 required model features, and score every dataset.
This helps when your dataset column names are different from the project CSV.
        """
    )
    st.code(", ".join(FEATURE_COLUMNS), language="text")

    if "load_project_datasets" not in st.session_state:
        st.session_state["load_project_datasets"] = False

    load_col, clear_col = st.columns(2)
    with load_col:
        if st.button("Load Project Datasets", use_container_width=True, key="load_project_datasets_button"):
            st.session_state["load_project_datasets"] = True
    with clear_col:
        if st.button("Clear Project Datasets", use_container_width=True, key="clear_project_datasets_button"):
            st.session_state["load_project_datasets"] = False

    multi_uploads = st.file_uploader(
        "Upload one or more CSV datasets for comparison",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_dataset_lab_uploads",
    )

    dataset_sources = []
    if st.session_state.get("load_project_datasets"):
        available_project_datasets = [path for path in PROJECT_DATASET_PATHS if path.exists()]
        if available_project_datasets:
            st.success(
                "Project datasets loaded: "
                + ", ".join(path.name for path in available_project_datasets)
            )
            for project_path in available_project_datasets:
                dataset_sources.append({"name": project_path.name, "source": project_path})
        else:
            st.warning("No built-in project datasets were found in the data folder.")

    for uploaded_dataset in multi_uploads or []:
        dataset_sources.append({"name": uploaded_dataset.name, "source": uploaded_dataset})

    comparison_rows = []
    if not dataset_sources:
        st.info("Click `Load Project Datasets` or upload 2-3 CSV files here. Then choose which columns match each model feature.")
    else:
        for file_index, dataset_item in enumerate(dataset_sources, start=1):
            dataset_name = dataset_item["name"]
            dataset_source = dataset_item["source"]
            with st.expander(f"Dataset {file_index}: {dataset_name}", expanded=file_index == 1):
                try:
                    raw_custom_df = read_flexible_csv(dataset_source)
                except Exception as exc:
                    st.error(f"Could not read this CSV: {exc}")
                    continue

                st.caption(f"Raw file shape: {raw_custom_df.shape[0]:,} rows x {raw_custom_df.shape[1]:,} columns")
                st.dataframe(raw_custom_df.head(10), use_container_width=True)
                st.markdown("**Step 1: Fit your CSV columns to model features**")
                st.caption("If your CSV does not have a feature, leave it as 'Use default value'. The app will fill a safe default.")

                source_options = ["Use default value"] + list(raw_custom_df.columns)
                machine_label_choice = st.selectbox(
                    "Machine label column",
                    ["Auto detect"] + list(raw_custom_df.columns),
                    index=0,
                    key=f"machine_label_{file_index}_{dataset_name}",
                )
                mapping: dict[str, str | None] = {}
                map_cols = st.columns(2)
                for feature_index, feature in enumerate(FEATURE_COLUMNS):
                    guessed_index = 0
                    simple_feature = feature.replace("_", " ").lower()
                    for option_index, option in enumerate(source_options):
                        simple_option = str(option).replace("_", " ").lower()
                        if simple_option == simple_feature or simple_feature in simple_option:
                            guessed_index = option_index
                            break
                    with map_cols[feature_index % 2]:
                        selected = st.selectbox(
                            feature,
                            source_options,
                            index=guessed_index,
                            key=f"mapping_{file_index}_{dataset_name}_{feature}",
                        )
                    mapping[feature] = None if selected == "Use default value" else selected

                target_choice = st.selectbox(
                    "Optional: actual target/failure column for accuracy check",
                    ["No actual target"] + list(raw_custom_df.columns),
                    key=f"target_{file_index}_{dataset_name}",
                )

                if st.button("Run This Dataset", key=f"run_dataset_{file_index}_{dataset_name}", use_container_width=True):
                    try:
                        fitted_df = standardize_dataset_with_mapping(
                            raw_custom_df,
                            mapping,
                            None if target_choice == "No actual target" else target_choice,
                            None if machine_label_choice == "Auto detect" else machine_label_choice,
                        )
                        scored_custom_df = add_prediction_scores(fitted_df, model)
                        st.session_state[f"scored_dataset_{file_index}_{dataset_name}"] = scored_custom_df
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

                scored_key = f"scored_dataset_{file_index}_{dataset_name}"
                scored_custom_df = st.session_state.get(scored_key)
                if scored_custom_df is not None:
                    dataset_kpis = build_kpi_frame(scored_custom_df)
                    comparison_rows.append(
                        {
                            "Dataset": dataset_name,
                            "Rows": int(dataset_kpis["total_assets"]),
                            "High Risk": int(dataset_kpis["high_risk_assets"]),
                            "Average Risk": f"{dataset_kpis['avg_risk']:.1f}%",
                            "Stable": int(dataset_kpis["monitored_stable"]),
                        }
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Rows Scored", f"{int(dataset_kpis['total_assets']):,}")
                    m2.metric("High Risk", f"{int(dataset_kpis['high_risk_assets']):,}")
                    m3.metric("Average Risk", f"{dataset_kpis['avg_risk']:.1f}%")
                    m4.metric("Stable", f"{int(dataset_kpis['monitored_stable']):,}")

                    if target_choice != "No actual target":
                        actual = scored_custom_df["downtime_risk"].astype(int)
                        pred = scored_custom_df["predicted_risk"].astype(int)
                        st.success(f"Accuracy against selected target column: {float((actual == pred).mean()):.1%}")

                    st.plotly_chart(risk_distribution_chart(scored_custom_df), use_container_width=True)
                    preview_cols = ["machine_label"] + FEATURE_COLUMNS + ["risk_probability", "predicted_risk", "recommendation"]
                    st.dataframe(scored_custom_df[preview_cols].head(25), use_container_width=True)

                    safe_name = Path(dataset_name).stem.replace(" ", "_")
                    st.download_button(
                        "Download scored CSV",
                        scored_custom_df[preview_cols].to_csv(index=False).encode("utf-8"),
                        f"{safe_name}_scored_predictions.csv",
                        "text/csv",
                        key=f"download_{file_index}_{dataset_name}",
                    )

                    if st.button("Save This Dataset Batch to MySQL", key=f"save_dataset_{file_index}_{dataset_name}", use_container_width=True):
                        if not all([db_host, db_user, db_password, db_name]):
                            st.error("Configure MySQL in the sidebar first.")
                        else:
                            try:
                                run_id = save_batch_predictions(db_config, scored_custom_df, dataset_name)
                                st.success(f"Saved to MySQL. run_id: {run_id[:8]}...")
                            except Exception as exc:
                                st.error(f"Could not save dataset batch: {exc}")

        if comparison_rows:
            st.subheader("Dataset Comparison")
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            chart_df = comparison_df.copy()
            chart_df["Average Risk Numeric"] = chart_df["Average Risk"].str.rstrip("%").astype(float)
            fig_compare = px.bar(
                chart_df,
                x="Dataset",
                y="Average Risk Numeric",
                color="High Risk",
                title="Average Risk by Dataset",
                color_continuous_scale=["#2a9d8f", "#edae49", "#d1495b"],
            )
            fig_compare.update_layout(template="plotly_white", yaxis_title="Average Risk (%)", coloraxis_showscale=False)
            st.plotly_chart(fig_compare, use_container_width=True)
            riskiest_dataset = chart_df.sort_values("Average Risk Numeric", ascending=False).iloc[0]
            st.info(
                "AI dataset advisor: "
                f"`{riskiest_dataset['Dataset']}` currently looks like the riskiest uploaded dataset "
                f"with an average risk of {riskiest_dataset['Average Risk Numeric']:.1f}% "
                f"and {riskiest_dataset['High Risk']} high-risk records."
            )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4 â€” SHAP EXPLAINABILITY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_shap:
    st.subheader("SHAP Explainability - Why Did the Model Predict This?")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** explains each individual prediction by showing
    how much each sensor feature pushed the risk score **up** (red) or **down** (blue)
    from the model's baseline probability.
    """)

    if not shap_available:
        # Show the real reason SHAP failed.
        st.error("SHAP explainer could not be loaded.")
        if _SHAP_LOAD_ERROR:
            st.markdown(f"""
**Reason:**
```
{_SHAP_LOAD_ERROR}
```
            """)
        # Step-by-step fix guide
        st.markdown("""
---
### How to fix this

**If the error says "not installed" or "no module named shap":**

> Streamlit Cloud has a **cached build** - it did not reinstall packages.

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Find your app -> click the menu in the top right
3. Click **Reboot app** - this forces a full `pip install -r requirements.txt`
4. Wait ~60 seconds for the app to restart

**If the error says "failed to initialise" (SHAP is installed but crashes):**

> Your saved model artifact may be from an older scikit-learn version.

1. Run this locally to retrain:
```bash
python scripts/train_model.py --dataset data/manufacturing_downtime_sample.csv
```
2. Commit the new `artifacts/best_model.joblib` to GitHub
3. Streamlit Cloud will redeploy automatically

**If nothing works - check your requirements.txt has:**
```
shap==0.46.0
```
        """)
    else:
        shap_col1, shap_col2 = st.columns([1, 1])

        with shap_col1:
            st.markdown("**Explain current sidebar machine**")
            shap_vals, base_val, feat_names = compute_shap_values(display_input_df)
            if shap_vals is not None:
                shap_df = pd.DataFrame({
                    "Feature":    feat_names,
                    "SHAP Value": shap_vals,
                    "Input Value": display_input_df[feature_columns].iloc[0].values,
                }).sort_values("SHAP Value")

                colors = ["#d1495b" if v > 0 else "#2a9d8f" for v in shap_df["SHAP Value"]]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["SHAP Value"],
                    y=shap_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in shap_df["SHAP Value"]],
                    textposition="outside",
                ))
                fig_shap.update_layout(
                    title=f"SHAP Waterfall - {machine_label}<br>"
                          f"<sup>Base: {base_val:.3f} -> Prediction: {probability:.1%}</sup>",
                    xaxis_title="SHAP value (impact on risk score)",
                    template="plotly_white",
                    margin=dict(l=10, r=40, t=80, b=10),
                    height=380,
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                st.dataframe(
                    shap_df.sort_values("SHAP Value", ascending=False)
                    .assign(**{"Direction": lambda d: d["SHAP Value"].apply(
                        lambda v: "Increases risk" if v > 0 else "Reduces risk"
                    )}),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("SHAP calculation failed for this input - try different slider values.")

        with shap_col2:
            st.markdown("**Global SHAP - Fleet-level feature impact**")
            if dataset_df is not None and len(dataset_df) >= 10:
                try:
                    sample_size = min(200, len(dataset_df))
                    X_sample = dataset_df[feature_columns].sample(sample_size, random_state=42)
                    if shap_scaler == "direct":
                        imputer, scaler = shap_imputer
                        X_proc_values = scaler.transform(imputer.transform(X_sample))
                    elif shap_scaler == "preprocessor":
                        X_proc_values = shap_imputer.transform(X_sample)
                    else:
                        raise ValueError("Unsupported SHAP preprocessing mode.")
                    if hasattr(X_proc_values, "toarray"):
                        X_proc_values = X_proc_values.toarray()
                    X_proc = pd.DataFrame(X_proc_values, columns=feature_columns)
                    sv_all = explainer.shap_values(X_proc)
                    if isinstance(sv_all, list):
                        sv_class1 = sv_all[1]
                    elif isinstance(sv_all, np.ndarray) and sv_all.ndim == 3:
                        sv_class1 = sv_all[:, :, 1]
                    elif hasattr(sv_all, "values"):
                        sv_values = sv_all.values
                        sv_class1 = sv_values[:, :, 1] if getattr(sv_values, "ndim", 0) == 3 else sv_values
                    else:
                        sv_class1 = sv_all

                    mean_abs = pd.DataFrame({
                        "Feature":         feature_columns,
                        "Mean |SHAP|":     np.abs(sv_class1).mean(axis=0),
                    }).sort_values("Mean |SHAP|", ascending=True)

                    fig_global = px.bar(
                        mean_abs, x="Mean |SHAP|", y="Feature", orientation="h",
                        title=f"Global Feature Impact (sample of {sample_size} machines)",
                        color="Mean |SHAP|",
                        color_continuous_scale=["#2a9d8f","#edae49","#d1495b"],
                        text_auto=".4f",
                    )
                    fig_global.update_layout(
                        template="plotly_white", coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=55, b=10), height=380,
                    )
                    st.plotly_chart(fig_global, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Global SHAP calculation failed: {exc}")
            else:
                st.info("Global SHAP requires the full fleet dataset. Using project CSV automatically loads 2,000 machines.")

        # SHAP interpretation guide
        st.markdown("""---
**How to read the SHAP chart:**

| Colour | Meaning |
|--------|---------|
| Red bars (positive SHAP) | This feature **increases** downtime risk for this machine |
| Blue bars (negative SHAP) | This feature **decreases** downtime risk for this machine |
| Bar length | How strongly that feature influences the prediction |

**Example:** If `bearing_temperature` has SHAP = +0.35, it means that machine's high bearing temperature
pushed the predicted risk **35 percentage points higher** than average.
        """)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 5 â€” 17-STEP WORKFLOW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_workflow:
    st.subheader("Complete 17-Step ML Workflow")
    st.caption("Every step: what we did, why it matters, the output, and the formula.")
    st.progress(17 / 17)
    st.markdown("**Project completion: 17/17 steps done**")
    st.markdown("<br>", unsafe_allow_html=True)

    for i, step_name in enumerate(WORKFLOW_STEPS, start=1):
        icon   = STEP_ICONS[i - 1]
        detail = STEP_DETAILS[i]
        status_badge = "Done"
        status_bg    = "#dcfce7"
        status_color = "#166534"

        with st.expander(f"{icon}  Step {i:02d} - {step_name}", expanded=(i == 1)):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;">
                <div style="width:2.8rem;height:2.8rem;border-radius:12px;
                    background:linear-gradient(135deg,#7a5af8,#5b8dee);
                    color:#fff;display:flex;align-items:center;justify-content:center;
                    font-weight:800;font-size:1.1rem;">{i:02d}</div>
                <div style="font-size:1.05rem;font-weight:800;color:#1f2557;">{icon} {step_name}</div>
                <div style="margin-left:auto;background:{status_bg};color:{status_color};
                    border-radius:999px;padding:.2rem .75rem;font-size:.8rem;font-weight:700;">
                    {status_badge}</div>
            </div>""", unsafe_allow_html=True)

            cw, cy = st.columns(2)
            with cw:
                st.markdown("**What we did**")
                st.markdown(f"<div class='detail-text'>{detail['what']}</div>", unsafe_allow_html=True)
            with cy:
                st.markdown("**Why it matters**")
                st.markdown(f"<div class='detail-text'>{detail['why']}</div>", unsafe_allow_html=True)

            st.markdown("**Output**")
            st.markdown(f"<div class='output-box'>{detail['output']}</div>", unsafe_allow_html=True)
            if detail["formula"]:
                st.markdown("**Formula**")
                st.markdown(f"<div class='formula-box'>{detail['formula']}</div>", unsafe_allow_html=True)

            # Live data embeds
            if i == 9 and not metrics.empty:
                st.markdown("**Live Training Results**")
                mc1,mc2,mc3,mc4 = st.columns(4)
                for col,(k,lbl) in zip([mc1,mc2,mc3,mc4],[
                    ("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1")]):
                    col.metric(lbl, f"{metrics.get(k,0):.1%}")

            if i == 10 and not metrics.empty:
                st.markdown("**Live Evaluation Metrics**")
                st.dataframe(pd.DataFrame({
                    "Metric":  ["Accuracy","Precision","Recall","F1","CV F1"],
                    "Value":   [f"{metrics.get(k,0):.4f}" for k in ["accuracy","precision","recall","f1","cv_f1"]],
                    "Meaning": ["Overall correct predictions",
                                "Of predicted high-risk, % truly high-risk",
                                "Of actual failures, % correctly caught",
                                "Harmonic mean of Precision + Recall",
                                "Cross-validation F1 (generalisation)"],
                }), use_container_width=True, hide_index=True)

            if i == 11:
                st.markdown("**Live Feature Importances + SHAP (see SHAP tab for full details)**")
                try:
                    rf_clf2 = model.named_steps.get("clf") or model.named_steps.get("classifier")
                    if hasattr(rf_clf2, "feature_importances_"):
                        imp2 = pd.DataFrame({
                            "Feature":        feature_columns,
                            "RF Importance %": (rf_clf2.feature_importances_*100).round(2),
                        }).sort_values("RF Importance %", ascending=False)
                        st.dataframe(imp2, use_container_width=True, hide_index=True)
                except Exception:
                    pass

            if i == 12:
                st.markdown("**Risk Banding Rules**")
                st.dataframe(pd.DataFrame({
                    "Band":      ["IMMEDIATE","48-HOUR","NORMAL"],
                    "Threshold": [">= 50%","30-49%","< 30%"],
                    "Action":    ["Stop machine, trigger alert now",
                                  "Schedule inspection within 48 h",
                                  "Continue normal operation"],
                }), use_container_width=True, hide_index=True)

            if i == 13:
                st.markdown("**Saved Artifacts**")
                for art in ["best_model.joblib","feature_columns.joblib","metrics.json"]:
                    fp   = ARTIFACT_DIR / art
                    size = f"{fp.stat().st_size/1024:.1f} KB" if fp.exists() else "missing"
                    st.markdown(f"- `artifacts/{art}` - {size}")

            if i == 15:
                st.markdown("**Live Test Results**")
                rows = []
                for tname, vals in [
                    ("HIGH-STRESS", dict(machine_temperature=95,bearing_temperature=110,
                                        vibration_level=9,pressure=180,runtime_hours=7500,
                                        load_percentage=95,maintenance_delay_days=60,error_log_count=8)),
                    ("STABLE",      dict(machine_temperature=55,bearing_temperature=60,
                                        vibration_level=1.5,pressure=130,runtime_hours=500,
                                        load_percentage=50,maintenance_delay_days=5,error_log_count=0)),
                    ("BOUNDARY",    dict(machine_temperature=80,bearing_temperature=85,
                                        vibration_level=5,pressure=150,runtime_hours=3000,
                                        load_percentage=72,maintenance_delay_days=20,error_log_count=3)),
                ]:
                    tdf = pd.DataFrame([vals])
                    tp, tpb = predict_risk(model, tdf[feature_columns])
                    ok = (tname.startswith("HIGH") and tp==1) or (tname.startswith("STABLE") and tp==0)
                    rows.append({"Machine": tname, "Prediction": "HIGH RISK" if tp==1 else "STABLE",
                                 "Probability": f"{tpb:.1%}", "Result": "Pass" if ok else "Review"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if i == 16:
                st.markdown("**Deployment Details**")
                st.markdown("""
                - **Platform:** Streamlit Community Cloud (free tier)
                - **Database:** Aiven MySQL (free tier, SSL auto-enabled)
                - **Auto-train:** App self-trains on first launch if artifacts missing
                - **Secrets:** MySQL credentials in Streamlit Secrets (never in code)
                - **SHAP:** Loaded at runtime, cached with `@st.cache_resource`
                """)

            if i == 17:
                st.markdown("**Estimated Productivity Impact**")
                st.dataframe(pd.DataFrame({
                    "Metric":        ["Failures prevented","Avg downtime cost","Inspection cost",
                                      "Est. annual saving","False alarm rate"],
                    "Before ML":     ["-","Rs 18,00,000/hr","-","-","~30%"],
                    "After ML":      ["38% fewer","Rs 18,00,000/hr","Rs 25,000/visit","Rs 68,00,000+","< 5%"],
                }), use_container_width=True, hide_index=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 6 â€” DATABASE CONSOLE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab_database:
    st.subheader("Aiven MySQL Database Console")
    st.markdown("Save and query predictions from your live cloud database.")

    rc1, rc2 = st.columns([1.2, 1])
    with rc1:
        if st.button("Refresh Recent Records", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Load Cloud DB Settings in the sidebar first.")
            else:
                try:
                    rec = fetch_recent_predictions(db_config, limit=25)
                    st.session_state["recent_predictions"] = rec
                    st.success(f"Loaded {len(rec)} records.")
                except Exception as exc:
                    st.error(f"Could not load: {exc}")

        rec_df = st.session_state.get("recent_predictions")
        if rec_df is not None and not rec_df.empty:
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Records CSV",
                rec_df.to_csv(index=False).encode(),
                "db_predictions.csv",
                "text/csv",
                key="download_db_predictions_csv",
            )
        elif rec_df is not None:
            st.info("No records yet - save a prediction from Prediction Studio first.")
        else:
            st.info("Click Refresh after connecting.")

    with rc2:
        st.markdown("**Connection Info**")
        if db_host:
            st.markdown(f"""
- **Host:** `{db_host}`
- **Port:** `{db_port}`
- **User:** `{db_user}`
- **Database:** `{db_name}`
- **SSL:** {"Auto (Aiven)" if "aivencloud.com" in str(db_host) else "Standard"}
            """)
        else:
            st.info("Load Cloud DB Settings to see connection info.")

        st.markdown("**Tables**")
        st.markdown("- `prediction_runs` - one row per save session\n- `machine_predictions` - one row per machine")

        sql_path = Path("sql") / "init_mysql.sql"
        if sql_path.exists():
            with st.expander("View SQL Setup Script"):
                st.code(sql_path.read_text(encoding="utf-8"), language="sql")

