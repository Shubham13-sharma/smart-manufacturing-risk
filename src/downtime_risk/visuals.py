"""Chart and KPI helpers — fixed for single-row and multi-row DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

from .data import FEATURE_COLUMNS
from .predict import predict_risk


# ── Scoring ───────────────────────────────────────────────────────────────────

def add_prediction_scores(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    try:
        proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        df["risk_probability"] = (proba * 100).round(1)
        df["predicted_risk"]   = (proba >= 0.5).astype(int)
    except Exception:
        df["risk_probability"] = 0.0
        df["predicted_risk"]   = 0
    if "machine_label" not in df.columns:
        df["machine_label"] = [f"MACHINE-{i+1:03d}" for i in range(len(df))]
    return df


# ── KPIs ──────────────────────────────────────────────────────────────────────

def build_kpi_frame(scored_df: pd.DataFrame) -> dict[str, float]:
    total     = len(scored_df)
    high_risk = int(scored_df["predicted_risk"].sum()) if "predicted_risk" in scored_df.columns else 0
    avg_risk  = float(scored_df["risk_probability"].mean()) if "risk_probability" in scored_df.columns else 0.0
    return {
        "total_assets":    total,
        "high_risk_assets": high_risk,
        "avg_risk":        avg_risk,
        "monitored_stable": total - high_risk,
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def risk_distribution_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Pie chart — High Risk vs Stable. Works with any number of rows."""
    if "predicted_risk" not in scored_df.columns or len(scored_df) < 2:
        # Not enough data — show placeholder
        fig = go.Figure(go.Pie(
            labels=["Stable", "High Risk"],
            values=[1, 0],
            hole=0.42,
            marker_colors=["#2a9d8f", "#d1495b"],
        ))
        fig.update_layout(
            title="Risk Distribution (load full dataset for live data)",
            template="plotly_white",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        return fig

    counts = scored_df["predicted_risk"].value_counts()
    labels = [("High Risk" if k == 1 else "Stable") for k in counts.index]
    colors = [("#d1495b" if k == 1 else "#2a9d8f") for k in counts.index]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.42,
        marker_colors=colors,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Fleet Risk Distribution",
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def top_risk_machines_chart(scored_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top N highest-risk machines."""
    if "risk_probability" not in scored_df.columns or len(scored_df) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Top Risk Machines (load full dataset for live data)",
            template="plotly_white",
        )
        return fig

    label_col = "machine_label" if "machine_label" in scored_df.columns else None
    if label_col is None:
        scored_df = scored_df.copy()
        scored_df["machine_label"] = [f"MACHINE-{i+1:03d}" for i in range(len(scored_df))]

    top = (
        scored_df[["machine_label", "risk_probability"]]
        .nlargest(top_n, "risk_probability")
        .reset_index(drop=True)
    )

    fig = px.bar(
        top,
        x="risk_probability",
        y="machine_label",
        orientation="h",
        title=f"Top {min(top_n, len(top))} Highest Risk Machines",
        labels={"risk_probability": "Risk Probability (%)", "machine_label": "Machine"},
        color="risk_probability",
        color_continuous_scale=["#2a9d8f", "#edae49", "#d1495b"],
        range_x=[0, 100],
        text_auto=".1f",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def trend_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Risk probability trend across the fleet with rolling average."""
    if "risk_probability" not in scored_df.columns or len(scored_df) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Failure Risk Trend (load full dataset for live data)",
            template="plotly_white",
        )
        return fig

    sample   = scored_df["risk_probability"].reset_index(drop=True)
    window   = max(5, len(sample) // 40)
    smoothed = sample.rolling(window, min_periods=1).mean()
    x_vals   = list(range(len(sample)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=sample.tolist(),
        mode="markers",
        name="Risk score",
        marker=dict(color="#edae49", size=3, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=smoothed.tolist(),
        mode="lines",
        name="Rolling average",
        line=dict(color="#d1495b", width=2.5),
    ))
    fig.update_layout(
        template="plotly_white",
        title="Failure Risk Trend Across Fleet",
        xaxis_title="Machine index",
        yaxis_title="Risk probability (%)",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def feature_correlation_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Heatmap of Pearson correlation between sensor features and risk probability."""
    cols = [c for c in FEATURE_COLUMNS if c in scored_df.columns]
    if "risk_probability" in scored_df.columns:
        cols = cols + ["risk_probability"]

    if len(cols) < 3 or len(scored_df) < 5:
        fig = go.Figure()
        fig.update_layout(
            title="Correlation Heatmap (load full dataset for live data)",
            template="plotly_white",
        )
        return fig

    corr = scored_df[cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Sensor Feature Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig
