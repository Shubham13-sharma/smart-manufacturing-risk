import pandas as pd


def predict_risk(model, input_df: pd.DataFrame, threshold: float = 0.5) -> tuple[int, float]:
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= threshold)
    return prediction, probability
