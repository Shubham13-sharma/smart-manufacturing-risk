from pathlib import Path
import json
import os

import joblib
import pandas as pd
import plotly.express as px
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


ARTIFACT_DIR = Path("artifacts")
DEFAULT_DATASET_PATH = Path("data") / "real_dataset.csv"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
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


def get_default_db_settings() -> dict[str, object]:
    secrets_db = {}
    if "mysql" in st.secrets:
        secrets_db = dict(st.secrets["mysql"])

    return {
        "host": secrets_db.get("host") or os.getenv("MYSQL_HOST", ""),
        "port": int(secrets_db.get("port") or os.getenv("MYSQL_PORT", 3306)),
        "user": secrets_db.get("user") or os.getenv("MYSQL_USER", ""),
        "password": secrets_db.get("password") or os.getenv("MYSQL_PASSWORD", ""),
        "database": secrets_db.get("database") or os.getenv("MYSQL_DATABASE", "smart_manufacturing"),
    }


default_db_settings = get_default_db_settings()


st.set_page_config(page_title="Downtime Risk Command Center", page_icon="F", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(237, 174, 73, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 121, 140, 0.16), transparent 24%),
            linear-gradient(180deg, #f6f2e8 0%, #fffdf9 100%);
    }
    .hero {
        padding: 1.25rem 1.35rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #f4f0ff 0%, #edf5ff 65%, #f8fcff 100%);
        color: #1f2557;
        box-shadow: 0 18px 40px rgba(130, 145, 190, 0.16);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(157, 180, 255, 0.22);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.45rem 0 0 0;
        font-size: 1rem;
        line-height: 1.5;
        max-width: 820px;
        color: #5b6482;
    }
    .badge-row {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-bottom: 0.85rem;
    }
    .badge-pill {
        background: #f1ecff;
        color: #6a4ae4;
        border: 1px solid #d9cffd;
        border-radius: 999px;
        padding: 0.32rem 0.85rem;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
    .summary-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dde5f3;
        border-radius: 22px;
        padding: 1rem 1.1rem;
        min-height: 150px;
        box-shadow: 0 10px 24px rgba(181, 190, 220, 0.14);
    }
    .summary-title {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #99a3bf;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }
    .summary-value {
        color: #232859;
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .tech-pill {
        display: inline-block;
        margin: 0.15rem 0.35rem 0.2rem 0;
        padding: 0.32rem 0.65rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 700;
        background: #e7fbff;
        color: #16a3c3;
        border: 1px solid #b8eef8;
    }
    .workflow-wrap {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid #e0e7f5;
        border-radius: 24px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }
    .workflow-header {
        color: #8d97b2;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.85rem;
    }
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background: #fafbff;
        border: 1px solid #e1e6f7;
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 8px 18px rgba(193, 203, 230, 0.12);
    }
    .step-id {
        min-width: 2.2rem;
        height: 2.2rem;
        border-radius: 10px;
        background: #f2edff;
        color: #7a5af8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
    }
    .step-text {
        color: #52607a;
        font-size: 1rem;
        font-weight: 600;
    }
    .note-box {
        background: #f1fffb;
        border: 2px solid #87dcc7;
        border-radius: 18px;
        padding: 1rem 1.05rem;
        color: #12705b;
        margin-top: 1rem;
    }
    .note-box strong {
        color: #0f7662;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(20, 108, 125, 0.10);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 28px rgba(20, 108, 125, 0.08);
    }
    .metric-label {
        color: #5a6772;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #0f4c5c;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .panel {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(20, 108, 125, 0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1f2937 !important;
        font-weight: 700 !important;
        background: rgba(255, 255, 255, 0.72) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 0.55rem 0.95rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #b42318 !important;
        background: rgba(255, 255, 255, 0.98) !important;
    }
    h2, h3, p, label, .stCaption, .stMarkdown, .stText {
        color: #102a43 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #102a43 !important;
    }
    .stDataFrame, .stTable {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
    }
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0f4c5c 0%, #146c7d 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 20px rgba(20, 108, 125, 0.18) !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #146c7d 0%, #1f7a8c 100%) !important;
        color: #ffffff !important;
    }
    .stButton > button:focus,
    .stDownloadButton > button:focus {
        color: #ffffff !important;
        box-shadow: 0 0 0 0.2rem rgba(20, 108, 125, 0.25) !important;
    }
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    [data-baseweb="textarea"] textarea,
    .stTextInput input,
    .stNumberInput input {
        color: #0f172a !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label,
    .stFileUploader label {
        color: #102a43 !important;
        font-weight: 700 !important;
    }
    .stFileUploader section {
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px dashed #94a3b8 !important;
        color: #102a43 !important;
    }
    .stAlert {
        color: #102a43 !important;
    }
    .stExpander summary,
    details summary {
        color: #102a43 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        background: linear-gradient(135deg, #edae49 0%, #d97706 100%) !important;
        color: #111827 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #f4c266 0%, #ea9f2d 100%) !important;
        color: #111827 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] small {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [data-baseweb="slider"] * {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #ffffff !important;
        background-color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="base-input"] input {
        color: #ffffff !important;
        background: #0f172a !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] .stFileUploader section,
    [data-testid="stSidebar"] .stAlert {
        color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="badge-row">
            <span class="badge-pill">#17</span>
            <span class="badge-pill">Manufacturing</span>
            <span class="badge-pill">Explainable AI Ready</span>
        </div>
        <h1>Smart Manufacturing Downtime Risk Command Center</h1>
        <p>
            Monitor machine health, score downtime risk, visualize plant-wide failure patterns,
            and store prediction records in MySQL for internship-grade project presentation.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.warning("Model artifacts not found. Run `py -3 scripts\\train_model.py --dataset data\\real_dataset.csv` first.")
    st.stop()


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)
metrics = pd.Series(json.loads(METRICS_PATH.read_text(encoding="utf-8"))) if METRICS_PATH.exists() else pd.Series(dtype=float)

with st.sidebar:
    st.header("Operations Panel")
    data_mode = st.radio("Dataset source", ["Use project CSV", "Upload dataset"], index=0)
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"]) if data_mode == "Upload dataset" else None
    risk_threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5)

    st.markdown("---")
    st.subheader("MySQL Storage")
    st.caption("Use a remote MySQL host for deployment. `localhost` is not supported on Streamlit Cloud.")
    db_host = st.text_input("Host", value=str(default_db_settings["host"]), placeholder="your-remote-mysql-host")
    db_port = st.number_input("Port", min_value=1, max_value=65535, value=int(default_db_settings["port"]))
    db_user = st.text_input("User", value=str(default_db_settings["user"]), placeholder="db_username")
    db_password = st.text_input("Password", value=str(default_db_settings["password"]), type="password")
    db_name = st.text_input("Database", value=str(default_db_settings["database"]), placeholder="smart_manufacturing")

    db_config = DatabaseConfig(
        host=db_host,
        port=int(db_port),
        user=db_user,
        password=db_password,
        database=db_name,
    )

    connection_col, setup_col = st.columns(2)
    with connection_col:
        if st.button("Test DB", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Enter your remote MySQL host, user, password, and database first.")
            else:
                ok, message = test_connection(db_config)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
    with setup_col:
        if st.button("Init Tables", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Enter your remote MySQL host, user, password, and database first.")
            else:
                try:
                    initialize_tables(db_config)
                    st.success("Tables are ready in MySQL.")
                except Exception as exc:
                    st.error(f"Database setup failed: {exc}")

    st.markdown("---")
    st.subheader("Single Machine Demo")
    machine_label = st.text_input("Machine label", value="CNC-17")
    machine_temperature = st.slider("Machine temperature (deg C)", 30.0, 120.0, 78.0)
    bearing_temperature = st.slider("Bearing temperature (deg C)", 25.0, 140.0, 82.0)
    vibration_level = st.slider("Vibration level", 0.0, 15.0, 4.5)
    pressure = st.slider("Pressure", 50.0, 250.0, 140.0)
    runtime_hours = st.slider("Runtime hours", 0, 10000, 2400)
    load_percentage = st.slider("Load percentage", 0.0, 120.0, 72.0)
    maintenance_delay_days = st.slider("Maintenance delay (days)", 0, 180, 15)
    error_log_count = st.slider("Error log count", 0, 20, 2)
    run_prediction = st.button("Run Prediction", use_container_width=True)


input_df = pd.DataFrame(
    [
        {
            "machine_temperature": machine_temperature,
            "bearing_temperature": bearing_temperature,
            "vibration_level": vibration_level,
            "pressure": pressure,
            "runtime_hours": runtime_hours,
            "load_percentage": load_percentage,
            "maintenance_delay_days": maintenance_delay_days,
            "error_log_count": error_log_count,
        }
    ]
)

if "latest_prediction" not in st.session_state:
    initial_prediction, initial_probability = predict_risk(
        model, input_df[feature_columns], threshold=risk_threshold
    )
    st.session_state["latest_prediction"] = initial_prediction
    st.session_state["latest_probability"] = initial_probability
    st.session_state["latest_input_df"] = input_df.copy()

if run_prediction:
    latest_prediction, latest_probability = predict_risk(
        model, input_df[feature_columns], threshold=risk_threshold
    )
    st.session_state["latest_prediction"] = latest_prediction
    st.session_state["latest_probability"] = latest_probability
    st.session_state["latest_input_df"] = input_df.copy()

prediction = st.session_state["latest_prediction"]
probability = st.session_state["latest_probability"]
display_input_df = st.session_state["latest_input_df"]
recommendation = (
    "Immediate maintenance inspection recommended"
    if prediction == 1
    else "Continue monitoring under normal schedule"
)

dataset_df = None
dataset_note = None

try:
    if uploaded_file is not None:
        dataset_df = load_dataset_from_csv(uploaded_file)
        dataset_note = "Uploaded dataset normalized to the project schema."
    elif DEFAULT_DATASET_PATH.exists():
        dataset_df = load_dataset_from_csv(DEFAULT_DATASET_PATH)
        dataset_note = f"Using project dataset from `{DEFAULT_DATASET_PATH}`."
except ValueError as exc:
    st.error(f"Dataset could not be loaded: {exc}")

if dataset_df is not None:
    scored_df = add_prediction_scores(dataset_df, model)
    source_name = uploaded_file.name if uploaded_file is not None else DEFAULT_DATASET_PATH.name
else:
    scored_df = add_prediction_scores(display_input_df.assign(downtime_risk=prediction), model)
    source_name = "single_machine_demo"

kpis = build_kpi_frame(scored_df)
workflow_html = "".join(
    [
        f"""
        <div class="workflow-step">
            <div class="step-id">{index:02d}</div>
            <div class="step-text">{step}</div>
        </div>
        """
        for index, step in enumerate(WORKFLOW_STEPS, start=1)
    ]
)

tab_overview, tab_predict, tab_analytics, tab_database = st.tabs(
    ["Overview", "Prediction Studio", "Plant Analytics", "Database Console"]
)

with tab_overview:
    summary_cols = st.columns(3)
    summary_cols[0].markdown(
        """
        <div class="summary-card">
            <div class="summary-title">Deliverables</div>
            <div class="summary-value">Downtime Risk Model</div>
            <p>Operational dashboard, prediction workflow, alert-ready risk scoring, and SQL storage.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_cols[1].markdown(
        """
        <div class="summary-card">
            <div class="summary-title">Prerequisites</div>
            <span class="tech-pill">Decision Trees</span>
            <span class="tech-pill">Random Forest</span>
            <span class="tech-pill">Classification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_cols[2].markdown(
        """
        <div class="summary-card">
            <div class="summary-title">Key Techniques</div>
            <span class="tech-pill">Failure Pattern Analysis</span>
            <span class="tech-pill">Classification</span>
            <span class="tech-pill">SHAP / LIME</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Executive Snapshot")
    kpi_columns = st.columns(4)
    kpi_items = [
        ("Assets Monitored", f"{int(kpis['total_assets'])}"),
        ("High Risk Assets", f"{int(kpis['high_risk_assets'])}"),
        ("Average Risk", f"{kpis['avg_risk']:.1f}%"),
        ("Stable Assets", f"{int(kpis['monitored_stable'])}"),
    ]
    for column, (label, value) in zip(kpi_columns, kpi_items):
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(risk_distribution_chart(scored_df), use_container_width=True)
    with right:
        st.plotly_chart(top_risk_machines_chart(scored_df), use_container_width=True)

    st.markdown(
        f"""
        <div class="workflow-wrap">
            <div class="workflow-header">Step-By-Step Workflow • 17 Steps</div>
            {workflow_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_predict:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Live Machine Prediction")
        if prediction == 1:
            st.error(f"High downtime risk detected with {probability:.1%} probability.")
        else:
            st.success(f"Low downtime risk predicted with {probability:.1%} probability.")

        if probability >= risk_threshold:
            st.warning("Recommended action: trigger maintenance alert and inspect the machine.")
        else:
            st.info("Recommended action: continue operation and keep monitoring key signals.")

        gauge_df = pd.DataFrame({"Risk band": [machine_label], "Probability": [probability]})
        gauge = px.bar(
            gauge_df,
            x="Probability",
            y="Risk band",
            orientation="h",
            text_auto=".0%",
            range_x=[0, 1],
            color="Probability",
            color_continuous_scale=["#2a9d8f", "#edae49", "#d1495b"],
            title="Current Machine Risk Gauge",
        )
        gauge.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=55, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(gauge, use_container_width=True)

        action_col, save_col = st.columns(2)
        with action_col:
            if st.button("Save Current Prediction", use_container_width=True):
                if not all([db_host, db_user, db_password, db_name]):
                    st.error("Configure your remote MySQL database in the sidebar first.")
                else:
                    try:
                        save_single_prediction(
                            db_config,
                            display_input_df,
                            prediction,
                            probability,
                            recommendation,
                            machine_label,
                        )
                        st.success("Current prediction saved to MySQL.")
                    except Exception as exc:
                        st.error(f"Could not save prediction: {exc}")
        with save_col:
            if st.button("Save Full Dataset Batch", use_container_width=True):
                if not all([db_host, db_user, db_password, db_name]):
                    st.error("Configure your remote MySQL database in the sidebar first.")
                else:
                    try:
                        run_id = save_batch_predictions(db_config, scored_df, source_name)
                        st.success(f"Dataset batch saved to MySQL with run id {run_id}.")
                    except Exception as exc:
                        st.error(f"Could not save batch: {exc}")

    with right:
        st.subheader("Machine Input Snapshot")
        st.dataframe(display_input_df, use_container_width=True, hide_index=True)
        decision_df = pd.DataFrame(
            {
                "Field": ["Machine label", "Predicted risk", "Probability", "Recommendation"],
                "Value": [machine_label, prediction, f"{probability:.1%}", recommendation],
            }
        )
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        if not metrics.empty:
            st.caption("Best trained model performance")
            metrics_df = metrics.rename("value").reset_index().rename(columns={"index": "metric"})
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab_analytics:
    st.subheader("Plant Risk Analytics")
    if dataset_note:
        st.caption(dataset_note)

    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(trend_chart(scored_df), use_container_width=True)
    with top_right:
        st.plotly_chart(feature_correlation_chart(scored_df), use_container_width=True)

    st.subheader("Scored Dataset Preview")
    display_columns = FEATURE_COLUMNS + [
        column for column in ["risk_probability", "predicted_risk"] if column in scored_df.columns
    ]
    st.dataframe(scored_df[display_columns].head(25), use_container_width=True)
    st.markdown(
        """
        <div class="note-box">
            <strong>Explainable AI Requirement:</strong>
            Apply SHAP or LIME to interpret downtime predictions and identify the strongest drivers
            behind a high-risk machine classification during your final presentation.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_database:
    st.subheader("Database Console")
    st.markdown(
        """
        Use the buttons in the sidebar or Prediction Studio to connect and save records into MySQL.
        Run the SQL file from `sql/init_mysql.sql` in MySQL Workbench before saving if your database is not ready yet.
        """
    )

    recent_col, sql_col = st.columns([1.2, 1])
    with recent_col:
        if st.button("Refresh Recent Records", use_container_width=True):
            if not all([db_host, db_user, db_password, db_name]):
                st.error("Configure your remote MySQL database in the sidebar first.")
            else:
                try:
                    recent_predictions = fetch_recent_predictions(db_config, limit=25)
                    st.session_state["recent_predictions"] = recent_predictions
                except Exception as exc:
                    st.error(f"Could not load saved records: {exc}")

        recent_predictions = st.session_state.get("recent_predictions")
        if recent_predictions is not None:
            st.dataframe(recent_predictions, use_container_width=True, hide_index=True)
        else:
            st.info("No records loaded yet. Click `Refresh Recent Records` after connecting to MySQL.")

    with sql_col:
        sql_path = Path("sql") / "init_mysql.sql"
        if sql_path.exists():
            st.caption("MySQL Workbench SQL setup")
            st.code(sql_path.read_text(encoding="utf-8"), language="sql")
