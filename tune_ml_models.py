from itertools import product
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "rj_monthly_dengue.csv"
OUT = ROOT / "outputs"

TEST_MONTHS = 24
FOLD_MONTHS = 12
LAGS = 12


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


def xgboost_recursive_forecast(
    train_log,
    forecast_dates,
    params,
):
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
        **params,
    )
    model.fit(training[features], training["target"])

    history = list(train_log.iloc[-LAGS:].to_numpy())
    log_forecasts = []
    for date in forecast_dates:
        row = {f"lag_{lag}": history[-lag] for lag in range(1, LAGS + 1)}
        row["month"] = date.month
        row["rolling_mean_3"] = np.mean(history[-3:])
        prediction = float(model.predict(pd.DataFrame([row], columns=features))[0])
        log_forecasts.append(prediction)
        history.append(prediction)

    return np.expm1(np.asarray(log_forecasts)).clip(min=0)


def prophet_recursive_forecast(
    train_log,
    forecast_dates,
    changepoint_prior_scale,
    seasonality_prior_scale,
):
    frame = pd.DataFrame({
        "ds": train_log.index,
        "y": train_log.to_numpy(),
    })
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    model.fit(frame)
    future = model.make_future_dataframe(
        periods=len(forecast_dates),
        freq="MS",
    )
    log_forecast = model.predict(future).tail(len(forecast_dates))["yhat"].to_numpy()
    return np.expm1(log_forecast).clip(min=0)


df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").asfreq("MS")
cases = df["cases"]
log_cases = np.log1p(cases)

train = cases.iloc[:-TEST_MONTHS]
test = cases.iloc[-TEST_MONTHS:]
train_log = log_cases.iloc[:-TEST_MONTHS]

# Three expanding-window folds on training data only.
# Each fold is 12 months, and the fit portion grows by one year per fold.
fold_starts = [len(train) - 36, len(train) - 24, len(train) - 12]
xgboost_cv = []
prophet_cv = []

xgboost_grid = {
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [300, 500],
    "subsample": [0.8, 1.0],
}
xgboost_candidates = [
    dict(zip(xgboost_grid, values))
    for values in product(*xgboost_grid.values())
]

prophet_grid = [
    (change_scale, season_scale)
    for change_scale in [0.05, 0.1, 0.5]
    for season_scale in [1.0, 5.0, 10.0]
]

print("Starting expanding-window time-series CV...")

for params in xgboost_candidates:
    fold_errors = []
    for fold_start in fold_starts:
        fit_log = train_log.iloc[:fold_start]
        valid_log = train_log.iloc[fold_start:fold_start + FOLD_MONTHS]
        valid_actual = train.iloc[fold_start:fold_start + FOLD_MONTHS]

        assert valid_actual.index.min() > fit_log.index.max()
        forecast = xgboost_recursive_forecast(
            fit_log,
            valid_log.index,
            params,
        )
        fold_errors.append(rmse(valid_actual, forecast))

    xgboost_cv.append({
        **params,
        "mean_cv_RMSE": float(np.mean(fold_errors)),
    })

xgboost_cv = pd.DataFrame(xgboost_cv).sort_values("mean_cv_RMSE")
best_xgboost_params = xgboost_cv.iloc[0].to_dict()
best_xgboost_params.pop("mean_cv_RMSE")
best_xgboost_params["max_depth"] = int(best_xgboost_params["max_depth"])
best_xgboost_params["n_estimators"] = int(best_xgboost_params["n_estimators"])

for changepoint_scale, seasonality_scale in prophet_grid:
    fold_errors = []
    for fold_start in fold_starts:
        fit_log = train_log.iloc[:fold_start]
        valid_log = train_log.iloc[fold_start:fold_start + FOLD_MONTHS]
        valid_actual = train.iloc[fold_start:fold_start + FOLD_MONTHS]

        assert valid_actual.index.min() > fit_log.index.max()
        forecast = prophet_recursive_forecast(
            fit_log,
            valid_log.index,
            changepoint_scale,
            seasonality_scale,
        )
        fold_errors.append(rmse(valid_actual, forecast))

    prophet_cv.append({
        "changepoint_prior_scale": changepoint_scale,
        "seasonality_prior_scale": seasonality_scale,
        "mean_cv_RMSE": float(np.mean(fold_errors)),
    })

prophet_cv = pd.DataFrame(prophet_cv).sort_values("mean_cv_RMSE")
best_prophet_params = prophet_cv.iloc[0].to_dict()
best_prophet_mean = float(best_prophet_params.pop("mean_cv_RMSE"))

xgboost_cv.to_csv(OUT / "xgboost_cv_results.csv", index=False)
prophet_cv.to_csv(OUT / "prophet_cv_results.csv", index=False)

print("\nXGBoost CV ranking:")
print(xgboost_cv.head(10).round(4).to_string(index=False))

print("\nProphet CV ranking:")
print(prophet_cv.head(10).round(4).to_string(index=False))

print("\nBest XGBoost params:")
print(best_xgboost_params)
print("Mean CV RMSE:", round(float(xgboost_cv.iloc[0]["mean_cv_RMSE"]), 3))

print("\nBest Prophet params:")
print(best_prophet_params)
print("Mean CV RMSE:", round(best_prophet_mean, 3))

# Refit each winning model once on the full 2001-2017 training period.
xgboost_test_forecast = xgboost_recursive_forecast(
    train_log,
    test.index,
    best_xgboost_params,
)

prophet_test_forecast = prophet_recursive_forecast(
    train_log,
    test.index,
    float(best_prophet_params["changepoint_prior_scale"]),
    float(best_prophet_params["seasonality_prior_scale"]),
)

test_comparison = pd.DataFrame([
    {
        "model": "xgboost",
        "RMSE": rmse(test, xgboost_test_forecast),
        "MAE": mae(test, xgboost_test_forecast),
        "MAPE (%)": mape(test, xgboost_test_forecast),
        "parameters": str({
            key: best_xgboost_params[key]
            for key in ["max_depth", "learning_rate", "n_estimators", "subsample"]
        }),
    },
    {
        "model": "prophet",
        "RMSE": rmse(test, prophet_test_forecast),
        "MAE": mae(test, prophet_test_forecast),
        "MAPE (%)": mape(test, prophet_test_forecast),
        "parameters": str(best_prophet_params),
    },
])
test_comparison.to_csv(OUT / "ml_cv_test_results.csv", index=False)

pd.DataFrame({
    "date": test.index,
    "actual": test.to_numpy(),
    "xgboost": xgboost_test_forecast,
    "prophet": prophet_test_forecast,
}).to_csv(OUT / "ml_cv_test_forecasts.csv", index=False)

print("\nTest set evaluated once after CV:")
print(test_comparison.round(3).to_string(index=False))
