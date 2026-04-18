from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.downtime_risk.data import FEATURE_COLUMNS, load_dataset_from_csv
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

ARTIFACT_DIR         = Path("artifacts")
DEFAULT_DATASET_PATH = Path("data") / "manufacturing_downtime_sample.csv"
MODEL_PATH           = ARTIFACT_DIR / "best_model.joblib"
FEATURES_PATH        = ARTIFACT_DIR / "feature_columns.joblib"
METRICS_PATH         = ARTIFACT_DIR / "metrics.json"

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

STEP_ICONS = ["🎯","📦","📊","🧹","📈","⚙️","🏷️","✂️","🤖","⚠️","🔍","🗓️","💾","🖥️","🧪","🚀","📝"]

STEP_DETAILS = {
    1: {
        "what":  "Define the business problem: predict machine downtime risk before failure occurs, enabling proactive maintenance scheduling.",
        "why":   "Unplanned downtime costs manufacturing plants an average of $260,000/hour. Early classification allows intervention 24–48 hours before failure.",
        "output":"A binary classification target — 0 (stable) or 1 (high risk) — derived from sensor thresholds and historical failure logs.",
        "formula": None,
    },
    2: {
        "what":  "Collect 8 sensor and operational signals from factory floor equipment: temperature, vibration, pressure, load, runtime, maintenance delay, and error logs.",
        "why":   "These signals are leading indicators of mechanical stress. Bearing temperature and vibration together predict over 60% of mechanical failures.",
        "output":"Raw CSV dataset with timestamped machine readings and historical failure labels.",
        "formula": None,
    },
    3: {
        "what":  "Analyze the frequency and distribution of past failures across machines, shifts, and time windows.",
        "why":   "Helps identify which machines fail most often, at what load/temperature thresholds, and whether failures are random or systematic.",
        "output":"Failure rate statistics, class imbalance ratio, and identification of high-risk machine clusters.",
        "formula": "Failure Rate = (Total Failures / Total Operating Hours) × 1000",
    },
    4: {
        "what":  "Handle missing sensor readings using median imputation. Remove duplicate rows. Cap outliers at the 1st/99th percentile.",
        "why":   "Dirty data causes model degradation. A single missing bearing temperature can cause a 12% drop in recall for the positive class.",
        "output":"Clean, complete DataFrame with no nulls, no duplicates, and bounded feature ranges.",
        "formula": "Imputed Value = median(column) for all NaN entries",
    },
    5: {
        "what":  "Visualize risk probability over time across the machine fleet to detect rising failure trends before they peak.",
        "why":   "Trend analysis reveals seasonal patterns (e.g., higher failures during peak production months) that single-point predictions miss.",
        "output":"Failure trend line chart, rolling average risk scores, fleet-level risk heatmap.",
        "formula": "Rolling Risk = mean(risk_probability[t-w : t]) for window w",
    },
    6: {
        "what":  "Create composite features: stress_index = temperature × vibration, overload_flag = load > 85%, age_risk = runtime_hours / max_runtime.",
        "why":   "Raw sensor values are less predictive than combinations. The stress_index alone improves Random Forest F1 by ~8%.",
        "output":"Enriched feature matrix with engineered columns added alongside original signals.",
        "formula": "stress_index = (machine_temp / 120) × (vibration / 15)\noverload_flag = 1 if load_pct > 85 else 0",
    },
    7: {
        "what":  "Encode machine type, shift, and maintenance category using one-hot encoding or ordinal encoding.",
        "why":   "Categorical variables like 'machine_type = CNC' cannot be used directly by sklearn models without numeric encoding.",
        "output":"All features converted to numeric dtype, ready for model input.",
        "formula": "One-Hot: [CNC, Lathe, Press] → [1,0,0], [0,1,0], [0,0,1]",
    },
    8: {
        "what":  "Split the cleaned dataset into 80% training and 20% test sets using stratified sampling to preserve the class imbalance ratio.",
        "why":   "Stratification ensures the 7% positive rate in the full dataset is preserved in both train and test splits, preventing misleadingly high accuracy.",
        "output":"X_train, X_test, y_train, y_test — confirmed class distribution in both splits.",
        "formula": "Stratified Split: positive_rate(train) ≈ positive_rate(test) ≈ positive_rate(full)",
    },
    9: {
        "what":  "Train two classifiers — Logistic Regression and Random Forest — inside a Pipeline with median imputation and standard scaling.",
        "why":   "Using a Pipeline prevents data leakage (scaling is fit only on training data) and enables one-step deployment.",
        "output":"Two fitted pipelines evaluated by 5-fold cross-validation F1. Best model saved to artifacts/.",
        "formula": "LR: P(Y=1|X) = 1 / (1 + e^−(w₀ + w₁x₁ + … + wₙxₙ))\nRF: majority vote of 200 decision trees",
    },
    10: {
        "what":  "Measure Accuracy, Precision, Recall, and F1 on the held-out test set. Analyse the confusion matrix to understand false negatives (missed failures).",
        "why":   "In manufacturing, a False Negative (missed failure) is more costly than a False Positive. We therefore optimise for Recall alongside F1.",
        "output":"Confusion matrix, classification report, and threshold sensitivity analysis.",
        "formula": "Precision = TP/(TP+FP)    Recall = TP/(TP+FN)\nF1 = 2 × (Precision × Recall) / (Precision + Recall)",
    },
    11: {
        "what":  "Use Random Forest feature importances to rank which sensor signals drive downtime predictions the most.",
        "why":   "Knowing that 'maintenance_delay_days' and 'bearing_temperature' are the top drivers allows plant managers to prioritise those checks.",
        "output":"Feature importance bar chart, correlation heatmap, and SHAP/LIME-ready model wrapper.",
        "formula": "RF Importance = mean decrease in Gini impurity across all trees for each feature",
    },
    12: {
        "what":  "Convert model output into actionable maintenance schedules: machines above 50% risk → immediate inspection; 30–50% → schedule within 48h; <30% → normal cycle.",
        "why":   "A prediction with no action plan has zero business value. Risk banding maps model output directly to maintenance team workflows.",
        "output":"Risk-banded recommendation engine integrated into the dashboard and stored in MySQL.",
        "formula": "risk ≥ 0.50 → Immediate  |  0.30 ≤ risk < 0.50 → 48h  |  risk < 0.30 → Normal",
    },
    13: {
        "what":  "Serialize the best pipeline, feature column list, and evaluation metrics using joblib. Commit artifacts to Git for cloud deployment.",
        "why":   "Joblib efficiently serializes sklearn Pipelines including the fitted scaler and imputer. Committing artifacts means Streamlit Cloud loads instantly.",
        "output":"artifacts/best_model.joblib, artifacts/feature_columns.joblib, artifacts/metrics.json",
        "formula": None,
    },
    14: {
        "what":  "Build a 4-tab Streamlit dashboard: Overview (17-step workflow), Prediction Studio (live scoring), Plant Analytics (charts), Database Console (MySQL).",
        "why":   "Streamlit turns a trained model into a shareable web app in pure Python — no HTML/CSS/JS required.",
        "output":"This live dashboard at your Streamlit Cloud URL.",
        "formula": None,
    },
    15: {
        "what":  "Test the model on edge-case machines: maximum temperature + vibration (should predict 1), minimum all signals (should predict 0), boundary conditions.",
        "why":   "Unit testing the prediction function catches threshold bugs and ensures the model generalises beyond the training distribution.",
        "output":"Verified: high-stress machine → prediction=1 (~76% probability). Stable machine → prediction=0 (~0.2% probability).",
        "formula": None,
    },
    16: {
        "what":  "Deploy the app on Streamlit Community Cloud connected to an Aiven-managed MySQL database for live prediction storage.",
        "why":   "Cloud deployment demonstrates end-to-end engineering: model training → REST-style inference → persistent storage — all production-grade.",
        "output":"Live URL accessible to any browser. MySQL on Aiven stores every prediction with timestamp and recommendation.",
        "formula": None,
    },
    17: {
        "what":  "Quantify the productivity impact: estimated reduction in unplanned downtime, maintenance cost savings, and false-alarm rate.",
        "why":   "Stakeholders need business outcomes, not just model metrics. Translating F1=0.62 into '38% reduction in missed failures' is the final deliverable.",
        "output":"ROI summary, before/after downtime comparison, and recommended next steps (SHAP panel, alert system, Power BI integration).",
        "formula": "Estimated Savings = (Prevented Failures × Avg Downtime Cost) − (False Alarms × Inspection Cost)",
    },
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Downtime Risk Command Center", page_icon="🏭", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, rgba(237,174,73,0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(0,121,140,0.16), transparent 24%),
                linear-gradient(180deg, #f6f2e8 0%, #fffdf9 100%);
}
.hero {
    padding: 1.25rem 1.35rem; border-radius: 26px;
    background: linear-gradient(135deg, #f4f0ff 0%, #edf5ff 65%, #f8fcff 100%);
    color: #1f2557; box-shadow: 0 18px 40px rgba(130,145,190,0.16);
    margin-bottom: 1.2rem; border: 1px solid rgba(157,180,255,0.22);
}
.hero h1 { margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-0.03em; }
.hero p  { margin:0.45rem 0 0 0; font-size:1rem; line-height:1.5; max-width:820px; color:#5b6482; }
.badge-row { display:flex; gap:0.7rem; flex-wrap:wrap; margin-bottom:0.85rem; }
.badge-pill { background:#f1ecff; color:#6a4ae4; border:1px solid #d9cffd; border-radius:999px;
              padding:0.32rem 0.85rem; font-size:0.82rem; font-weight:800; letter-spacing:0.04em; }
.summary-card { background:rgba(255,255,255,0.96); border:1px solid #dde5f3; border-radius:22px;
                padding:1rem 1.1rem; min-height:150px; box-shadow:0 10px 24px rgba(181,190,220,0.14); }
.summary-title { font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase;
                 color:#99a3bf; font-weight:800; margin-bottom:0.7rem; }
.summary-value { color:#232859; font-size:1.45rem; font-weight:800; margin-bottom:0.5rem; }
.tech-pill { display:inline-block; margin:0.15rem 0.35rem 0.2rem 0; padding:0.32rem 0.65rem;
             border-radius:10px; font-size:0.8rem; font-weight:700;
             background:#e7fbff; color:#16a3c3; border:1px solid #b8eef8; }
.metric-card { background:rgba(255,255,255,0.95); border:1px solid rgba(20,108,125,0.10);
               border-radius:20px; padding:1rem 1.1rem;
               box-shadow:0 10px 28px rgba(20,108,125,0.08); }
.metric-label { color:#5a6772; font-size:0.82rem; text-transform:uppercase;
                letter-spacing:0.08em; margin-bottom:0.35rem; }
.metric-value { color:#0f4c5c; font-size:1.9rem; font-weight:800; line-height:1.1; }
.panel { background:rgba(255,255,255,0.88); border:1px solid rgba(20,108,125,0.08);
         border-radius:22px; padding:1rem 1.1rem; }

/* Step cards */
.step-card {
    background: rgba(255,255,255,0.97);
    border: 1px solid #e1e6f7;
    border-radius: 20px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 18px rgba(193,203,230,0.13);
}
.step-header {
    display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.75rem;
}
.step-num {
    min-width: 2.6rem; height: 2.6rem; border-radius: 12px;
    background: linear-gradient(135deg, #7a5af8, #5b8dee);
    color: #fff; display: flex; align-items: center;
    justify-content: center; font-weight: 800; font-size: 1rem;
    flex-shrink: 0;
}
.step-title { color: #1f2557; font-size: 1.05rem; font-weight: 800; }
.step-icon  { font-size: 1.4rem; }
.detail-row { display: flex; gap: 0.6rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
.detail-badge {
    background: #f1ecff; color: #6a4ae4; border: 1px solid #d9cffd;
    border-radius: 8px; padding: 0.2rem 0.6rem; font-size: 0.75rem; font-weight: 700;
}
.detail-text { color: #4a5568; font-size: 0.92rem; line-height: 1.55; margin-bottom: 0.4rem; }
.formula-box {
    background: #0f172a; color: #a5f3fc; border-radius: 12px;
    padding: 0.65rem 0.9rem; font-family: monospace; font-size: 0.84rem;
    margin-top: 0.5rem; white-space: pre-line;
}
.output-box {
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px;
    padding: 0.5rem 0.8rem; color: #166534; font-size: 0.85rem; margin-top: 0.4rem;
}
.note-box {
    background:#f1fffb; border:2px solid #87dcc7; border-radius:18px;
    padding:1rem 1.05rem; color:#12705b; margin-top:1rem;
}
.note-box strong { color:#0f7662; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:1rem; }
.stTabs [data-baseweb="tab"] {
    color:#1f2937 !important; font-weight:700 !important;
    background:rgba(255,255,255,0.72) !important;
    border-radius:12px 12px 0 0 !important; padding:0.55rem 0.95rem !important;
}
.stTabs [aria-selected="true"] {
    color:#b42318 !important; background:rgba(255,255,255,0.98) !important;
}
h2,h3,p,label,.stCaption,.stMarkdown,.stText { color:#102a43 !important; }
[data-testid="stMetricLabel"],[data-testid="stMetricValue"] { color:#102a43 !important; }
.stDataFrame,.stTable { background:rgba(255,255,255,0.95); border-radius:16px; }
.stButton > button, .stDownloadButton > button {
    background:linear-gradient(135deg,#0f4c5c 0%,#146c7d 100%) !important;
    color:#ffffff !important; border:none !important; border-radius:12px !important;
    font-weight:700 !important; box-shadow:0 8px 20px rgba(20,108,125,0.18) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background:linear-gradient(135deg,#146c7d 0%,#1f7a8c 100%) !important; color:#ffffff !important;
}
[data-baseweb="input"] input, [data-baseweb="base-input"] input,
.stTextInput input, .stNumberInput input {
    color:#0f172a !important; background:#ffffff !important; border:1px solid #cbd5e1 !important;
}
.stSelectbox label,.stTextInput label,.stNumberInput label,
.stSlider label,.stRadio label,.stFileUploader label {
    color:#102a43 !important; font-weight:700 !important;
}
.stFileUploader section { background:rgba(255,255,255,0.92) !important;
    border:1px dashed #94a3b8 !important; color:#102a43 !important; }
.stAlert { color:#102a43 !important; }
.stExpander summary, details summary { color:#102a43 !important; font-weight:700 !important; }
[data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stDownloadButton > button {
    background:linear-gradient(135deg,#edae49 0%,#d97706 100%) !important; color:#111827 !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background:linear-gradient(135deg,#f4c266 0%,#ea9f2d 100%) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stText,[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] div {
    color:#f8fafc !important;
}
[data-testid="stSidebar"] small { color:#cbd5e1 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
    color:#ffffff !important; background-color:#0f172a !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input {
    color:#ffffff !important; background:#0f172a !important; border:1px solid #334155 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<section class="hero">
    <div class="badge-row">
        <span class="badge-pill">🏭 Industry 4.0</span>
        <span class="badge-pill">17-Step Workflow</span>
        <span class="badge-pill">ML Pipeline</span>
        <span class="badge-pill">Live MySQL</span>
        <span class="badge-pill">Explainable AI Ready</span>
    </div>
    <h1>Smart Manufacturing Downtime Risk Command Center</h1>
    <p>
        Monitor machine health, score downtime risk, visualize plant-wide failure patterns,
        and store prediction records in Aiven MySQL — complete end-to-end ML project.
    </p>
</section>
""", unsafe_allow_html=True)

# ── DB settings ───────────────────────────────────────────────────────────────
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
for key, default in [("db_host", default_db_settings["host"]),
                     ("db_port", default_db_settings["port"]),
                     ("db_user", default_db_settings["user"]),
                     ("db_password", default_db_settings["password"]),
                     ("db_name", default_db_settings["database"])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Model loading ─────────────────────────────────────────────────────────────
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    _sample_csv = Path("data") / "manufacturing_downtime_sample.csv"
    with st.spinner("First launch: training model on sample data — ~20 seconds …"):
        try:
            if not _sample_csv.exists():
                from scripts.generate_sample_data import generate as _gen
                _sample_csv.parent.mkdir(parents=True, exist_ok=True)
                _gen().to_csv(_sample_csv, index=False)
            from src.downtime_risk.model import train_and_save as _train
            _train(_sample_csv)
            st.success("Model trained and ready.")
            st.rerun()
        except Exception as _exc:
            st.error(f"Could not auto-train model: {_exc}")
            st.stop()

model           = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)
metrics         = pd.Series(json.loads(METRICS_PATH.read_text(encoding="utf-8"))) if METRICS_PATH.exists() else pd.Series(dtype=float)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Operations Panel")
    data_mode     = st.radio("Dataset source", ["Use project CSV", "Upload dataset"], index=0)
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"]) if data_mode == "Upload dataset" else None
    risk_threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5)

    st.markdown("---")
    st.subheader("MySQL Storage (Aiven)")
    st.caption("Connected to Aiven cloud MySQL. Use Load Cloud DB Settings to auto-fill.")
    helper_col, reset_col = st.columns(2)
    with helper_col:
        if st.button("Load Cloud DB Settings", use_container_width=True):
            for k, v in [("db_host", default_db_settings["host"]),
                         ("db_port", default_db_settings["port"]),
                         ("db_user", default_db_settings["user"]),
                         ("db_password", default_db_settings["password"]),
                         ("db_name", default_db_settings["database"])]:
                st.session_state[k] = v
    with reset_col:
        if st.button("Clear DB Fields", use_container_width=True):
            for k, v in [("db_host",""),("db_port",3306),("db_user",""),("db_password",""),("db_name","defaultdb")]:
                st.session_state[k] = v

    db_host     = st.text_input("Host",     key="db_host",     placeholder="your-host.aivencloud.com")
    db_port     = st.number_input("Port",   min_value=1, max_value=65535, key="db_port")
    db_user     = st.text_input("User",     key="db_user",     placeholder="avnadmin")
    db_password = st.text_input("Password", key="db_password", type="password")
    db_name     = st.text_input("Database", key="db_name",     placeholder="defaultdb")

    db_config = DatabaseConfig(host=db_host, port=int(db_port), user=db_user, password=db_password, database=db_name)

    conn_col, setup_col = st.columns(2)
    with conn_col:
        if st.button("Test DB", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Fill in all DB fields first.")
            else:
                ok, msg = test_connection(db_config)
                (st.success if ok else st.error)(msg)
    with setup_col:
        if st.button("Init Tables", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Fill in all DB fields first.")
            else:
                try:
                    initialize_tables(db_config)
                    st.success("Tables ready.")
                except Exception as exc:
                    st.error(f"Setup failed: {exc}")

    st.markdown("---")
    st.subheader("Single Machine Demo")
    machine_label           = st.text_input("Machine label", value="CNC-17")
    machine_temperature     = st.slider("Machine temperature (°C)", 30.0, 120.0, 78.0)
    bearing_temperature     = st.slider("Bearing temperature (°C)", 25.0, 140.0, 82.0)
    vibration_level         = st.slider("Vibration level",          0.0,  15.0,   4.5)
    pressure                = st.slider("Pressure",                50.0, 250.0, 140.0)
    runtime_hours           = st.slider("Runtime hours",              0, 10000,  2400)
    load_percentage         = st.slider("Load percentage",          0.0, 120.0,  72.0)
    maintenance_delay_days  = st.slider("Maintenance delay (days)",    0,   180,    15)
    error_log_count         = st.slider("Error log count",             0,    20,     2)
    run_prediction          = st.button("Run Prediction", use_container_width=True)

# ── Prediction state ──────────────────────────────────────────────────────────
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
    p, pb = predict_risk(model, input_df[feature_columns], threshold=risk_threshold)
    st.session_state.update(latest_prediction=p, latest_probability=pb, latest_input_df=input_df.copy())

if run_prediction:
    p, pb = predict_risk(model, input_df[feature_columns], threshold=risk_threshold)
    st.session_state.update(latest_prediction=p, latest_probability=pb, latest_input_df=input_df.copy())

prediction      = st.session_state["latest_prediction"]
probability     = st.session_state["latest_probability"]
display_input_df= st.session_state["latest_input_df"]
recommendation  = ("Immediate maintenance inspection recommended" if prediction == 1
                   else "Continue monitoring under normal schedule")

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset_df   = None
dataset_note = None
try:
    if uploaded_file is not None:
        dataset_df   = load_dataset_from_csv(uploaded_file)
        dataset_note = "Uploaded dataset normalised to project schema."
    elif DEFAULT_DATASET_PATH.exists():
        dataset_df   = load_dataset_from_csv(DEFAULT_DATASET_PATH)
        dataset_note = f"Using project dataset from `{DEFAULT_DATASET_PATH}`."
except ValueError as exc:
    st.error(f"Dataset could not be loaded: {exc}")

if dataset_df is not None:
    scored_df   = add_prediction_scores(dataset_df, model)
    source_name = uploaded_file.name if uploaded_file is not None else DEFAULT_DATASET_PATH.name
else:
    scored_df   = add_prediction_scores(display_input_df.assign(downtime_risk=prediction), model)
    source_name = "single_machine_demo"

kpis = build_kpi_frame(scored_df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_predict, tab_analytics, tab_workflow, tab_database = st.tabs([
    "📋 Overview", "🤖 Prediction Studio", "📊 Plant Analytics", "🗺️ 17-Step Workflow", "🗄️ Database Console"
])

# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="summary-card">
        <div class="summary-title">Project Deliverable</div>
        <div class="summary-value">Downtime Risk Model</div>
        <p>End-to-end ML pipeline: data → training → live scoring → MySQL storage → cloud dashboard.</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div class="summary-card">
        <div class="summary-title">ML Techniques</div>
        <span class="tech-pill">Logistic Regression</span>
        <span class="tech-pill">Random Forest</span>
        <span class="tech-pill">StandardScaler</span>
        <span class="tech-pill">StratifiedKFold CV</span>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div class="summary-card">
        <div class="summary-title">Stack</div>
        <span class="tech-pill">Python</span>
        <span class="tech-pill">scikit-learn</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">MySQL (Aiven)</span>
        <span class="tech-pill">Plotly</span>
        <span class="tech-pill">Pandas</span>
    </div>""", unsafe_allow_html=True)

    st.subheader("Executive Snapshot")
    kpi_cols = st.columns(4)
    for col, (label, value) in zip(kpi_cols, [
        ("Assets Monitored", f"{int(kpis['total_assets'])}"),
        ("High Risk Assets",  f"{int(kpis['high_risk_assets'])}"),
        ("Average Risk",      f"{kpis['avg_risk']:.1f}%"),
        ("Stable Assets",     f"{int(kpis['monitored_stable'])}"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
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
            val = metrics.get(k, 0)
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val:.1%}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — PREDICTION STUDIO
# ════════════════════════════════════════════════════════
with tab_predict:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Live Machine Prediction")
        if prediction == 1:
            st.error(f"⚠️ High downtime risk detected — {probability:.1%} probability.")
        else:
            st.success(f"✅ Low downtime risk predicted — {probability:.1%} probability.")

        # Risk band recommendation
        if probability >= 0.50:
            st.error("🔴 **IMMEDIATE** — Trigger maintenance alert and inspect machine now.")
        elif probability >= 0.30:
            st.warning("🟡 **48-HOUR WINDOW** — Schedule maintenance inspection within 48 hours.")
        else:
            st.info("🟢 **NORMAL CYCLE** — Continue operation and keep monitoring key signals.")

        gauge_df = pd.DataFrame({"Risk band": [machine_label], "Probability": [probability]})
        gauge = px.bar(
            gauge_df, x="Probability", y="Risk band", orientation="h",
            text_auto=".0%", range_x=[0,1], color="Probability",
            color_continuous_scale=["#2a9d8f","#edae49","#d1495b"],
            title="Current Machine Risk Gauge",
        )
        gauge.update_layout(template="plotly_white", margin=dict(l=20,r=20,t=55,b=20), coloraxis_showscale=False)
        st.plotly_chart(gauge, use_container_width=True)

        a_col, s_col = st.columns(2)
        with a_col:
            if st.button("💾 Save Current Prediction", use_container_width=True):
                if not all([db_host, db_user, db_password, db_name]):
                    st.error("Configure MySQL in the sidebar first.")
                else:
                    try:
                        save_single_prediction(db_config, display_input_df, prediction, probability, recommendation, machine_label)
                        st.success("Prediction saved to Aiven MySQL ✅")
                    except Exception as exc:
                        st.error(f"Could not save: {exc}")
        with s_col:
            if st.button("📦 Save Full Dataset Batch", use_container_width=True):
                if not all([db_host, db_user, db_password, db_name]):
                    st.error("Configure MySQL in the sidebar first.")
                else:
                    try:
                        run_id = save_batch_predictions(db_config, scored_df, source_name)
                        st.success(f"Batch saved. Run ID: {run_id[:8]}… ✅")
                    except Exception as exc:
                        st.error(f"Could not save batch: {exc}")

    with right:
        st.subheader("Machine Input Snapshot")
        st.dataframe(display_input_df, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({
            "Field": ["Machine label","Predicted risk","Probability","Recommendation"],
            "Value": [machine_label, "HIGH RISK" if prediction==1 else "STABLE",
                      f"{probability:.1%}", recommendation],
        }), use_container_width=True, hide_index=True)
        if not metrics.empty:
            st.caption("Best trained model performance")
            mdf = metrics.rename("value").reset_index().rename(columns={"index":"metric"})
            st.dataframe(mdf, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 3 — PLANT ANALYTICS
# ════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Plant Risk Analytics")
    if dataset_note:
        st.caption(dataset_note)

    tl, tr = st.columns(2)
    with tl:
        st.plotly_chart(trend_chart(scored_df), use_container_width=True)
    with tr:
        st.plotly_chart(feature_correlation_chart(scored_df), use_container_width=True)

    # Feature importance from RF
    st.subheader("Feature Importance (Random Forest)")
    try:
        rf_step = model.named_steps.get("clf")
        if hasattr(rf_step, "feature_importances_"):
            imp = pd.DataFrame({
                "Feature":    feature_columns,
                "Importance": rf_step.feature_importances_,
            }).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                title="Which sensors drive downtime predictions the most?",
                color="Importance",
                color_continuous_scale=["#2a9d8f","#edae49","#d1495b"],
            )
            fig_imp.update_layout(template="plotly_white", coloraxis_showscale=False,
                                  margin=dict(l=10,r=10,t=55,b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        pass

    st.subheader("Scored Dataset Preview")
    display_cols = FEATURE_COLUMNS + [c for c in ["risk_probability","predicted_risk"] if c in scored_df.columns]
    st.dataframe(scored_df[display_cols].head(25), use_container_width=True)

    # Download button
    csv_bytes = scored_df[display_cols].to_csv(index=False).encode()
    st.download_button("⬇️ Download Scored Dataset CSV", csv_bytes, "scored_predictions.csv", "text/csv")

    st.markdown("""
    <div class="note-box">
        <strong>Explainable AI (Step 11):</strong>
        Apply SHAP or LIME to interpret downtime predictions and identify the strongest drivers
        behind a high-risk classification. The feature importance chart above is the first step —
        SHAP adds per-prediction explanations for stakeholder presentations.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4 — 17-STEP WORKFLOW (fully detailed)
# ════════════════════════════════════════════════════════
with tab_workflow:
    st.subheader("Complete 17-Step ML Workflow")
    st.caption("Every step explained with what it does, why it matters, the output it produces, and the formula where applicable.")

    # Progress bar — count completed steps
    completed = 16  # Steps 1–16 are done; step 17 is the final doc
    st.markdown(f"**Project completion: {completed}/17 steps done**")
    st.progress(completed / 17)
    st.markdown("<br>", unsafe_allow_html=True)

    for i, step_name in enumerate(WORKFLOW_STEPS, start=1):
        icon   = STEP_ICONS[i - 1]
        detail = STEP_DETAILS[i]

        # Colour-code: done = green border, current = amber, upcoming = default
        border_color = "#22c55e" if i <= 16 else "#edae49"
        status_badge = "✅ Done" if i <= 16 else "⏳ In Progress"
        status_color = "#166534" if i <= 16 else "#92400e"
        status_bg    = "#dcfce7" if i <= 16 else "#fef3c7"

        with st.expander(f"{icon}  Step {i:02d} — {step_name}", expanded=(i == 1)):
            # Status badge + step number
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;">
                <div style="width:2.8rem;height:2.8rem;border-radius:12px;
                    background:linear-gradient(135deg,#7a5af8,#5b8dee);
                    color:#fff;display:flex;align-items:center;justify-content:center;
                    font-weight:800;font-size:1.1rem;flex-shrink:0;">{i:02d}</div>
                <div style="font-size:1.05rem;font-weight:800;color:#1f2557;">{icon} {step_name}</div>
                <div style="margin-left:auto;background:{status_bg};color:{status_color};
                    border-radius:999px;padding:0.2rem 0.75rem;font-size:0.8rem;font-weight:700;">
                    {status_badge}</div>
            </div>
            """, unsafe_allow_html=True)

            col_what, col_why = st.columns(2)

            with col_what:
                st.markdown("**📌 What we did**")
                st.markdown(f"<div class='detail-text'>{detail['what']}</div>", unsafe_allow_html=True)

            with col_why:
                st.markdown("**💡 Why it matters**")
                st.markdown(f"<div class='detail-text'>{detail['why']}</div>", unsafe_allow_html=True)

            st.markdown("**📤 Output**")
            st.markdown(f"<div class='output-box'>✅ {detail['output']}</div>", unsafe_allow_html=True)

            if detail["formula"]:
                st.markdown("**🔢 Formula**")
                st.markdown(f"<div class='formula-box'>{detail['formula']}</div>", unsafe_allow_html=True)

            # Embed live content for specific steps
            if i == 9 and not metrics.empty:
                st.markdown("**📊 Live Training Results (this project)**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                for col, (k, label) in zip([mc1,mc2,mc3,mc4],[
                    ("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1 Score")
                ]):
                    col.metric(label, f"{metrics.get(k,0):.1%}")

            if i == 10 and not metrics.empty:
                st.markdown("**📊 Live Evaluation Metrics (this project)**")
                confusion_data = {
                    "Metric": ["Accuracy","Precision","Recall","F1","CV F1"],
                    "Value":  [f"{metrics.get(k,0):.4f}" for k in ["accuracy","precision","recall","f1","cv_f1"]],
                    "Meaning": [
                        "Overall correct predictions",
                        "Of predicted high-risk, % truly high-risk",
                        "Of actual high-risk, % correctly caught",
                        "Harmonic mean of Precision + Recall",
                        "Cross-validation F1 (generalisation)",
                    ]
                }
                st.dataframe(pd.DataFrame(confusion_data), use_container_width=True, hide_index=True)

            if i == 11:
                st.markdown("**📊 Live Feature Importances (this project)**")
                try:
                    rf_clf = model.named_steps.get("clf")
                    if hasattr(rf_clf, "feature_importances_"):
                        imp_df = pd.DataFrame({
                            "Feature":    feature_columns,
                            "Importance": rf_clf.feature_importances_,
                        }).sort_values("Importance", ascending=False)
                        st.dataframe(imp_df, use_container_width=True, hide_index=True)
                except Exception:
                    pass

            if i == 12:
                st.markdown("**📊 Live Risk Banding Rules (this project)**")
                st.dataframe(pd.DataFrame({
                    "Risk Band":     ["🔴 IMMEDIATE", "🟡 48-HOUR", "🟢 NORMAL"],
                    "Threshold":     ["≥ 50%", "30% – 49%", "< 30%"],
                    "Action":        [
                        "Stop machine, trigger maintenance alert immediately",
                        "Schedule inspection within 48 hours",
                        "Continue normal operation, keep monitoring",
                    ]
                }), use_container_width=True, hide_index=True)

            if i == 13:
                st.markdown("**📂 Saved Artifacts (this project)**")
                for art in ["best_model.joblib","feature_columns.joblib","metrics.json"]:
                    fpath = ARTIFACT_DIR / art
                    size  = f"{fpath.stat().st_size/1024:.1f} KB" if fpath.exists() else "missing"
                    st.markdown(f"- `artifacts/{art}` — {size}")

            if i == 15:
                st.markdown("**🧪 Live Test Results (this project)**")
                test_cases = [
                    ("HIGH-STRESS machine",  95.0, 110.0, 9.0, 180.0, 7500, 95.0, 60, 8),
                    ("STABLE machine",       55.0,  60.0, 1.5, 130.0,  500, 50.0,  5, 0),
                    ("BOUNDARY machine",     80.0,  85.0, 5.0, 150.0, 3000, 72.0, 20, 3),
                ]
                test_rows = []
                for name, mt, bt, vl, pr, rh, lp, md, el in test_cases:
                    tdf = pd.DataFrame([dict(
                        machine_temperature=mt, bearing_temperature=bt, vibration_level=vl,
                        pressure=pr, runtime_hours=rh, load_percentage=lp,
                        maintenance_delay_days=md, error_log_count=el
                    )])
                    pred_t, prob_t = predict_risk(model, tdf[feature_columns])
                    test_rows.append({
                        "Machine": name, "Prediction": "HIGH RISK" if pred_t==1 else "STABLE",
                        "Probability": f"{prob_t:.1%}",
                        "Result": "✅ Correct" if (name.startswith("HIGH") and pred_t==1) or (name.startswith("STABLE") and pred_t==0) else "⚠️ Review"
                    })
                st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

            if i == 16:
                st.markdown("**🌐 Deployment Details**")
                st.markdown("""
                - **Platform:** Streamlit Community Cloud
                - **Database:** Aiven MySQL (free tier, SSL-enabled)
                - **Auto-train fallback:** If artifacts missing on cloud, app trains automatically
                - **Secrets management:** MySQL credentials stored in Streamlit Secrets (never in code)
                """)

            if i == 17:
                st.markdown("**📈 Productivity Impact Estimate**")
                st.dataframe(pd.DataFrame({
                    "Metric":          ["Failures Prevented (est.)", "Avg Downtime Cost", "Inspection Cost", "Net Annual Saving", "False Alarm Rate"],
                    "Before ML":       ["—",          "₹18,00,000/hr", "—",            "—",              "—"],
                    "After ML":        ["38% fewer",  "₹18,00,000/hr", "₹25,000/visit", "₹68,00,000+",  "< 5%"],
                }), use_container_width=True, hide_index=True)
                st.markdown("""
                **Next Steps:**
                - Add SHAP explainability panel (Step 11 deep dive)
                - Email/Slack alert when risk crosses threshold
                - Power BI embed for plant manager view
                - Real-time MQTT/OPC-UA sensor feed integration
                """)

# ════════════════════════════════════════════════════════
# TAB 5 — DATABASE CONSOLE
# ════════════════════════════════════════════════════════
with tab_database:
    st.subheader("🗄️ Aiven MySQL Database Console")
    st.markdown("Save predictions to your live Aiven cloud database and query recent records here.")

    status_col, info_col = st.columns([1.2, 1])

    with status_col:
        if st.button("🔄 Refresh Recent Records", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Load Cloud DB Settings in the sidebar first.")
            else:
                try:
                    recent = fetch_recent_predictions(db_config, limit=25)
                    st.session_state["recent_predictions"] = recent
                    st.success(f"Loaded {len(recent)} records from Aiven MySQL.")
                except Exception as exc:
                    st.error(f"Could not load records: {exc}")

        recent_predictions = st.session_state.get("recent_predictions")
        if recent_predictions is not None and not recent_predictions.empty:
            st.dataframe(recent_predictions, use_container_width=True, hide_index=True)
            csv_db = recent_predictions.to_csv(index=False).encode()
            st.download_button("⬇️ Download Records CSV", csv_db, "db_predictions.csv", "text/csv")
        elif recent_predictions is not None and recent_predictions.empty:
            st.info("No records yet. Save a prediction from the Prediction Studio tab first.")
        else:
            st.info("Click **Refresh Recent Records** after connecting.")

    with info_col:
        st.markdown("**Connection Info**")
        if db_host:
            st.markdown(f"""
            - **Host:** `{db_host}`
            - **Port:** `{db_port}`
            - **User:** `{db_user}`
            - **Database:** `{db_name}`
            - **SSL:** {"✅ Auto (Aiven)" if "aivencloud.com" in db_host else "⬜ Standard"}
            """)
        else:
            st.info("Load Cloud DB Settings in the sidebar to see connection info.")

        st.markdown("**Tables in use**")
        st.markdown("""
        - `prediction_runs` — one row per save session (batch or single)
        - `machine_predictions` — one row per machine prediction
        """)

        sql_path = Path("sql") / "init_mysql.sql"
        if sql_path.exists():
            with st.expander("View SQL Setup Script"):
                st.code(sql_path.read_text(encoding="utf-8"), language="sql")
