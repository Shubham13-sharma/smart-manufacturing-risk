"""Prediction utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_risk(
    model: Pipeline,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[int, float]:
    """Return (binary_prediction, probability) for a single-row DataFrame.

    Parameters
    ----------
    model:
        Trained sklearn Pipeline with a ``predict_proba`` method.
    X:
        DataFrame with the expected feature columns (single row for live demo,
        multiple rows accepted – only the first row's result is returned).
    threshold:
        Probability cutoff for classifying as high risk (1).

    Returns
    -------
    (prediction, probability)
        prediction  – 0 (no risk) or 1 (high risk)
        probability – probability of class 1 (float in [0, 1])
    """
    if X.empty:
        return 0, 0.0

    proba = model.predict_proba(X)[0]

    # Handle models trained with only one class (edge case)
    if len(proba) == 1:
        prob_high = float(proba[0])
    else:
        prob_high = float(proba[1])

    prediction = int(prob_high >= threshold)
    return prediction, round(prob_high, 4)
