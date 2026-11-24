from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "rj_monthly_dengue.csv"
OUT = ROOT / "outputs"

TEST_MONTHS = 24
LAGS = 12
VALIDATION_START = pd.Timestamp("2015-01-01")
VALIDATION_END = pd.Timestamp("2017-12-01")
TEST_START = pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2019-12-01")

SARIMA_ORDER = (2, 1, 2)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 12)
XGBOOST_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 1.0,
}


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mae(actual, forecast):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def mape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def sarima_rolling(log_cases, dates):
    forecasts = []
    for date in dates:
        position = log_cases.index.get_loc(date)
        expanded = log_cases.iloc[:position]
        model = SARIMAX(
            expanded,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False, maxiter=500)
        log_forecast = float(np.asarray(model.forecast(steps=1))[0])
        forecasts.append(np.expm1(log_forecast))
    return np.clip(forecasts, 0, None)


def xgboost_forecast(train_log, dates):
    rows = []
    for i in range(LAGS, len(train_log)):
        row = {f"lag_{lag}": train_log.iloc[i - lag] for lag in range(1, LAGS + 1)}
        row["month"] = train_log.index[i].month
        row["rolling_mean_3"] = train_log.iloc[i - 3:i].mean()
        row["target"] = train_log.iloc[i]
        rows.append(row)

    training = pd.DataFrame(rows, index=train_log.index[LAGS:])
    features = [column for column in training if column != "target"]
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **XGBOOST_PARAMS,
    )
    model.fit(training[features], training["target"])

    history = list(train_log.iloc[-LAGS:].to_numpy())
    forecasts = []
    for date in dates:
        row = {f"lag_{lag}": history[-lag] for lag in range(1, LAGS + 1)}
        row["month"] = date.month
        row["rolling_mean_3"] = np.mean(history[-3:])
        log_forecast = float(model.predict(pd.DataFrame([row], columns=features))[0])
        forecasts.append(log_forecast)
        history.append(log_forecast)

    return np.expm1(np.asarray(forecasts)).clip(min=0)


df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").asfreq("MS")
cases = df["cases"]
log_cases = np.log1p(cases)

validation_dates = pd.date_range(VALIDATION_START, VALIDATION_END, freq="MS")
test_dates = pd.date_range(TEST_START, TEST_END, freq="MS")
validation_actual = cases.loc[validation_dates]
test_actual = cases.loc[test_dates]

# Validation begins after the fit window. The test set is never read during selection.
assert validation_dates.min() > pd.Timestamp("2014-12-01")
assert set(validation_dates).isdisjoint(set(test_dates))

validation_sarima = sarima_rolling(log_cases, validation_dates)
validation_xgboost = xgboost_forecast(
    log_cases.loc[:VALIDATION_END].iloc[:-len(validation_dates)],
    validation_dates,
)

validation_predictions = np.column_stack([
    validation_sarima,
    validation_xgboost,
])

# Search XGBoost/SARIMA weights on validation RMSE only.
weight_curve = []
for xgb_weight in np.round(np.arange(0, 1.0001, 0.05), 2):
    sarima_weight = 1 - xgb_weight
    forecast = xgb_weight * validation_xgboost + sarima_weight * validation_sarima
    weight_curve.append({
        "xgboost_weight": xgb_weight,
        "sarima_weight": round(sarima_weight, 2),
        "RMSE": rmse(validation_actual, forecast),
        "MAE": mae(validation_actual, forecast),
        "MAPE (%)": mape(validation_actual, forecast),
    })

weight_curve = pd.DataFrame(weight_curve).sort_values("xgboost_weight")
chosen_weight_row = weight_curve.sort_values("RMSE").iloc[0]
chosen_xgb_weight = float(chosen_weight_row["xgboost_weight"])

# Fit linear stacking coefficients using only validation predictions.
stacker = LinearRegression()
stacker.fit(validation_predictions, validation_actual.to_numpy())
stack_weights = dict(
    sarima=stacker.coef_[0],
    xgboost=stacker.coef_[1],
)

# Test set is used once here, after weight and stacking coefficients are fixed.
test_sarima = sarima_rolling(log_cases, test_dates)
test_xgboost = xgboost_forecast(train_log=log_cases.loc[:VALIDATION_END], dates=test_dates)

test_predictions = np.column_stack([test_sarima, test_xgboost])
chosen_weight_forecast = (
    chosen_xgb_weight * test_xgboost
    + (1 - chosen_xgb_weight) * test_sarima
)
stack_forecast = np.clip(stacker.predict(test_predictions), 0, None)

results = pd.DataFrame([
    {
        "model": "tuned SARIMA",
        "RMSE": rmse(test_actual, test_sarima),
        "MAE": mae(test_actual, test_sarima),
        "MAPE (%)": mape(test_actual, test_sarima),
    },
    {
        "model": "tuned XGBoost",
        "RMSE": rmse(test_actual, test_xgboost),
        "MAE": mae(test_actual, test_xgboost),
        "MAPE (%)": mape(test_actual, test_xgboost),
    },
    {
        "model": f"weighted average ({chosen_xgb_weight:.2f} XGB / {1 - chosen_xgb_weight:.2f} SARIMA)",
        "RMSE": rmse(test_actual, chosen_weight_forecast),
        "MAE": mae(test_actual, chosen_weight_forecast),
        "MAPE (%)": mape(test_actual, chosen_weight_forecast),
    },
    {
        "model": "linear stacking",
        "RMSE": rmse(test_actual, stack_forecast),
        "MAE": mae(test_actual, stack_forecast),
        "MAPE (%)": mape(test_actual, stack_forecast),
    },
])
results.to_csv(OUT / "ensemble_test_results.csv", index=False)
weight_curve.to_csv(OUT / "ensemble_validation_weights.csv", index=False)

test_forecasts = pd.DataFrame({
    "date": test_dates,
    "actual": test_actual.to_numpy(),
    "sarima": test_sarima,
    "xgboost": test_xgboost,
    "weighted_average": chosen_weight_forecast,
    "linear_stacking": stack_forecast,
})
test_forecasts.to_csv(OUT / "ensemble_test_forecasts.csv", index=False)

print(weight_curve.round(4).to_string(index=False))
print("\nChosen validation weighted average:")
print(chosen_weight_row.round(4).to_string())
print("\nStacking coefficients:")
print("SARIMA:", round(stacker.coef_[0], 6))
print("XGBoost:", round(stacker.coef_[1], 6))
print("Intercept:", round(stacker.intercept_, 6))
print("\nTest results:")
print(results.round(3).to_string(index=False))
