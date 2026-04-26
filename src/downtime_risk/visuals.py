from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from src.downtime_risk.data import FEATURE_COLUMNS


def add_prediction_scores(df: pd.DataFrame, model) -> pd.DataFrame:
    scored = df.copy()
    scored["risk_probability"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scored["predicted_risk"] = (scored["risk_probability"] >= 0.5).astype(int)
    scored["machine_label"] = scored.get("machine_label", pd.Series([f"MCH-{i+1:04d}" for i in range(len(scored))]))
    scored["recommendation"] = np.select(
        [
            scored["risk_probability"] >= 0.5,
            scored["risk_probability"] >= 0.3,
        ],
        [
            "Immediate maintenance inspection recommended",
            "Schedule inspection within 48 hours",
        ],
        default="Continue monitoring under normal schedule",
    )
    return scored


def build_kpi_frame(scored_df: pd.DataFrame) -> pd.Series:
    total_assets = len(scored_df)
    high_risk_assets = int((scored_df["predicted_risk"] == 1).sum()) if total_assets else 0
    avg_risk = float(scored_df["risk_probability"].mean() * 100) if total_assets else 0.0
    return pd.Series(
        {
            "total_assets": total_assets,
            "high_risk_assets": high_risk_assets,
            "avg_risk": avg_risk,
            "monitored_stable": total_assets - high_risk_assets,
        }
    )


def risk_distribution_chart(scored_df: pd.DataFrame):
    fig = px.histogram(
        scored_df,
        x="risk_probability",
        nbins=30,
        title="Risk Probability Distribution",
        color_discrete_sequence=["#0f4c5c"],
    )
    fig.update_layout(template="plotly_white", xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=55, b=10))
    return fig


def top_risk_machines_chart(scored_df: pd.DataFrame):
    top_df = scored_df.nlargest(min(10, len(scored_df)), "risk_probability")
    fig = px.bar(
        top_df,
        x="risk_probability",
        y="machine_label",
        orientation="h",
        title="Top Risk Machines",
        color="risk_probability",
        color_continuous_scale=["#2a9d8f", "#edae49", "#d1495b"],
    )
    fig.update_layout(template="plotly_white", coloraxis_showscale=False, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=55, b=10))
    return fig


def trend_chart(scored_df: pd.DataFrame):
    trend_df = scored_df.reset_index().rename(columns={"index": "record"})
    trend_df["rolling_average"] = trend_df["risk_probability"].rolling(30, min_periods=1).mean()
    fig = px.line(trend_df, x="record", y="rolling_average", title="Rolling Failure Risk Trend")
    fig.add_scatter(x=trend_df["record"], y=trend_df["risk_probability"], mode="markers", name="Risk score", opacity=0.35)
    fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=55, b=10))
    return fig


def feature_correlation_chart(scored_df: pd.DataFrame):
    corr = scored_df[FEATURE_COLUMNS + ["risk_probability"]].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#2a9d8f", "#f4f1de", "#d1495b"],
        title="Sensor Correlation Heatmap",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10))
    return fig

