from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "rj_monthly_dengue.csv"
OUT = ROOT / "outputs"

SEASONAL_PERIOD = 12
FIT_END = pd.Timestamp("2014-12-01")
VALIDATION_START = pd.Timestamp("2015-01-01")
VALIDATION_END = pd.Timestamp("2017-12-01")
TEST_START = pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2019-12-01")

ORDER = (2, 1, 2)
SEASONAL_ORDER = (0, 1, 1, 12)
LEGACY_WEIGHT = 0.60


def rmse(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return float(np.mean(np.abs(actual - forecast)))


def mape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def fit_rolling_forecasts(log_cases, cases, period_start, period_end):
    period_dates = pd.date_range(period_start, period_end, freq="MS")
    sarima_forecasts = []

    for date in period_dates:
        expanded_end = log_cases.index.get_loc(date)
        expanded_log = log_cases.iloc[:expanded_end]
        model = SARIMAX(
            expanded_log,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False, maxiter=500)
        log_forecast = float(np.asarray(model.forecast(steps=1))[0])
        sarima_forecasts.append(np.expm1(log_forecast))

    seasonal_forecasts = np.array([
        cases.loc[date - pd.DateOffset(years=1)]
        for date in period_dates
    ])

    actual = cases.loc[period_dates].to_numpy()
    sarima_forecasts = np.clip(sarima_forecasts, 0, None)
    return actual, sarima_forecasts, seasonal_forecasts


df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").asfreq("MS")
cases = df["cases"]
log_cases = np.log1p(cases)

# Hard boundary: weight search receives only 2001-2014 fit data and 2015-2017 validation data.
fit_cases = cases.loc[:FIT_END]
fit_log_cases = log_cases.loc[:FIT_END]
validation_cases = cases.loc[VALIDATION_START:VALIDATION_END]
assert validation_cases.index.min() == VALIDATION_START
assert validation_cases.index.max() == VALIDATION_END
assert validation_cases.index.min() > fit_cases.index.max()
assert cases.loc[TEST_START:TEST_END].index.min() == TEST_START
assert not set(validation_cases.index).intersection(
    cases.loc[TEST_START:TEST_END].index
)

# Refit rolling SARIMA and append observed validation values as they become available.
validation_actual, validation_sarima, validation_seasonal = fit_rolling_forecasts(
    log_cases,
    cases,
    VALIDATION_START,
    VALIDATION_END,
)

validation_curve = []
weights = np.round(np.arange(0, 1.0001, 0.05), 2)
for weight in weights:
    blend = weight * validation_sarima + (1 - weight) * validation_seasonal
    validation_curve.append({
        "sarima_weight": weight,
        "seasonal_weight": round(1 - weight, 2),
        "RMSE": rmse(validation_actual, blend),
        "MAE": mae(validation_actual, blend),
        "MAPE (%)": mape(validation_actual, blend),
    })

validation_curve = pd.DataFrame(validation_curve)
validation_curve.to_csv(OUT / "blend_validation_curve.csv", index=False)

# MAPE is the requested selection metric because it matches the project's prior SARIMA target.
chosen_row = validation_curve.sort_values("MAPE (%)").iloc[0]
chosen_weight = float(chosen_row["sarima_weight"])
legacy_row = validation_curve[
    np.isclose(validation_curve["sarima_weight"], LEGACY_WEIGHT)
].iloc[0]

# Test set is untouched until the validation search and chosen weight are finalized.
test_actual, test_sarima, test_seasonal = fit_rolling_forecasts(
    log_cases,
    cases,
    TEST_START,
    TEST_END,
)
chosen_test_forecast = chosen_weight * test_sarima + (1 - chosen_weight) * test_seasonal
legacy_test_forecast = LEGACY_WEIGHT * test_sarima + (1 - LEGACY_WEIGHT) * test_seasonal

test_comparison = pd.DataFrame([
    {
        "weight": f"{chosen_weight:.2f}/" + f"{1 - chosen_weight:.2f}",
        "source": "chosen on 2015-2017 validation",
        "RMSE": rmse(test_actual, chosen_test_forecast),
        "MAE": mae(test_actual, chosen_test_forecast),
        "MAPE (%)": mape(test_actual, chosen_test_forecast),
    },
    {
        "weight": "0.60/0.40",
        "source": "legacy weight",
        "RMSE": rmse(test_actual, legacy_test_forecast),
        "MAE": mae(test_actual, legacy_test_forecast),
        "MAPE (%)": mape(test_actual, legacy_test_forecast),
    },
])
test_comparison.to_csv(OUT / "blend_test_comparison.csv", index=False)

print(validation_curve.round(3).to_string(index=False))
print("\nChosen on validation:")
print(chosen_row.round(3).to_string())
print("\nLegacy 60/40 validation:")
print(legacy_row.round(3).to_string())
print("\nTest set evaluated once:")
print(test_comparison.round(3).to_string(index=False))
