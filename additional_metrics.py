from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def mape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined for zero actual counts")
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def smape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    denominator = np.abs(actual) + np.abs(forecast)
    if np.any(denominator == 0):
        raise ValueError("sMAPE is undefined where actual and forecast sums are zero")
    return float(
        np.mean(200 * np.abs(actual - forecast) / denominator)
    )


def mase(actual, forecast, training_actual, seasonality=12):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    training_actual = np.asarray(training_actual)

    if len(training_actual) <= seasonality:
        raise ValueError("Training history is too short for the requested seasonal MASE")

    seasonal_naive_errors = np.abs(
        training_actual[seasonality:] - training_actual[:-seasonality]
    )
    scale = np.mean(seasonal_naive_errors)
    if scale == 0:
        raise ValueError("MASE scale is zero")

    return float(np.mean(np.abs(actual - forecast)) / scale)


forecasts = pd.read_csv(
    OUT / "test_forecasts.csv",
    parse_dates=["date"],
)
actual = forecasts["actual"].to_numpy()

processed = pd.read_csv(
    ROOT / "data" / "processed" / "rj_monthly_dengue.csv",
    parse_dates=["date"],
)
training_actual = processed.iloc[:-24]["cases"].to_numpy()

rows = []
for model in [
    column for column in forecasts.columns
    if column not in ["date", "actual"]
]:
    prediction = forecasts[model].to_numpy()
    rows.append({
        "model": model,
        "MAPE (%)": mape(actual, prediction),
        "MASE (seasonal naive scale)": mase(
            actual,
            prediction,
            training_actual,
            seasonality=12,
        ),
        "sMAPE (%)": smape(actual, prediction),
    })

metrics = pd.DataFrame(rows).sort_values("MASE (seasonal naive scale)")
metrics.round(4).to_csv(OUT / "additional_metrics.csv", index=False)
print(metrics.round(4).to_string(index=False))
