import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.downtime_risk.data import FEATURE_COLUMNS


def add_prediction_scores(df: pd.DataFrame, model) -> pd.DataFrame:
    scored_df = df.copy()
    scored_df["risk_probability"] = model.predict_proba(scored_df[FEATURE_COLUMNS])[:, 1]
    scored_df["predicted_risk"] = (scored_df["risk_probability"] >= 0.5).astype(int)
    return scored_df


def build_kpi_frame(scored_df: pd.DataFrame) -> dict[str, float]:
    total_assets = float(len(scored_df))
    high_risk_assets = float(scored_df["predicted_risk"].sum())
    avg_risk = float(scored_df["risk_probability"].mean() * 100)
    monitored_stable = total_assets - high_risk_assets
    return {
        "total_assets": total_assets,
        "high_risk_assets": high_risk_assets,
        "avg_risk": avg_risk,
        "monitored_stable": monitored_stable,
    }


def risk_distribution_chart(scored_df: pd.DataFrame):
    return px.histogram(
        scored_df,
        x="risk_probability",
        nbins=20,
        color_discrete_sequence=["#d1495b"],
        title="Downtime Risk Distribution",
        labels={"risk_probability": "Predicted downtime probability"},
    ).update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
    )


def feature_correlation_chart(df: pd.DataFrame):
    corr_df = df[FEATURE_COLUMNS].corr(numeric_only=True)
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale=["#f7f4ea", "#f2cc8f", "#d1495b", "#7f5539"],
        title="Sensor Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20))
    return fig


def top_risk_machines_chart(scored_df: pd.DataFrame):
    top_df = scored_df.nlargest(10, "risk_probability").reset_index().rename(columns={"index": "machine_id"})
    top_df["machine_id"] = top_df["machine_id"].astype(str)
    fig = px.bar(
        top_df,
        x="risk_probability",
        y="machine_id",
        orientation="h",
        color="risk_probability",
        color_continuous_scale=["#f2cc8f", "#d1495b"],
        title="Top 10 Machines by Risk Score",
        labels={"risk_probability": "Risk probability", "machine_id": "Machine row"},
    )
    fig.update_layout(
        template="plotly_white",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=20, r=20, t=55, b=20),
        coloraxis_showscale=False,
    )
    return fig


def trend_chart(scored_df: pd.DataFrame):
    trend_df = scored_df.reset_index().rename(columns={"index": "sequence"})
    rolling = trend_df["risk_probability"].rolling(window=25, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df["sequence"],
            y=trend_df["risk_probability"],
            mode="markers",
            marker=dict(color="#edae49", size=6, opacity=0.45),
            name="Risk score",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["sequence"],
            y=rolling,
            mode="lines",
            line=dict(color="#00798c", width=3),
            name="Rolling average",
        )
    )
    fig.update_layout(
        title="Failure Trend View",
        xaxis_title="Machine sample sequence",
        yaxis_title="Predicted downtime probability",
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig
