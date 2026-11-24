from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "rj_monthly_dengue.csv"
OUTPUT_PATH = ROOT / "outputs"
TEST_MONTHS = 24
SEASONAL_PERIOD = 12


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mae(actual, forecast):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def mape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def make_xgboost_forecast(train_log, test_dates, recursive=True):
    rows = []
    for i in range(SEASONAL_PERIOD, len(train_log)):
        row = {f"lag_{lag}": train_log.iloc[i - lag] for lag in range(1, 13)}
        row["month"] = train_log.index[i].month
        row["rolling_mean_3"] = train_log.iloc[i - 3:i].mean()
        row["target"] = train_log.iloc[i]
        rows.append(row)

    training = pd.DataFrame(rows, index=train_log.index[SEASONAL_PERIOD:])
    features = [column for column in training if column != "target"]
    model = XGBRegressor(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(training[features], training["target"])

    history = list(train_log.iloc[-SEASONAL_PERIOD:].to_numpy())
    forecast = []
    for forecast_date in test_dates:
        row = {f"lag_{lag}": history[-lag] for lag in range(1, 13)}
        row["month"] = forecast_date.month
        row["rolling_mean_3"] = np.mean(history[-3:])
        prediction = float(model.predict(pd.DataFrame([row], columns=features))[0])
        forecast.append(prediction)
        history.append(prediction)

    return np.expm1(np.asarray(forecast)).clip(min=0)


def prophet_forecast(train_log, test_dates, changepoint_prior_scale):
    frame = pd.DataFrame({"ds": train_log.index, "y": train_log.to_numpy()})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(frame)
    future = model.make_future_dataframe(periods=len(test_dates), freq="MS")
    log_forecast = model.predict(future).tail(len(test_dates))["yhat"].to_numpy()
    return np.expm1(log_forecast).clip(min=0)


df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").asfreq("MS")
cases = df["cases"]
log_cases = np.log1p(cases)
train = cases.iloc[:-TEST_MONTHS]
test = cases.iloc[-TEST_MONTHS:]
train_log = log_cases.iloc[:-TEST_MONTHS]

multi = pd.DataFrame({"actual": test})
rolling = pd.DataFrame({"actual": test})

# Baselines
multi["naive"] = train.iloc[-1]
multi["seasonal_naive"] = [
    train.iloc[-SEASONAL_PERIOD + (i % SEASONAL_PERIOD)]
    for i in range(TEST_MONTHS)
]

rolling["naive"] = [
    cases.iloc[len(train) + i - 1] for i in range(TEST_MONTHS)
]
rolling["seasonal_naive"] = [
    cases.iloc[len(train) + i - SEASONAL_PERIOD]
    for i in range(TEST_MONTHS)
]

# ARIMA
arima_order = (3, 1, 2)
arima_multi_result = SARIMAX(
    train_log,
    order=arima_order,
    trend="n",
    enforce_stationarity=True,
    enforce_invertibility=True,
).fit(disp=False, maxiter=500)
arima_multi_log = arima_multi_result.get_forecast(steps=TEST_MONTHS).predicted_mean
multi["arima"] = np.expm1(np.asarray(arima_multi_log)).clip(min=0)

arima_rolling = []
for step in range(TEST_MONTHS):
    expanded_log = log_cases.iloc[: len(train_log) + step]
    result = SARIMAX(
        expanded_log,
        order=arima_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False, maxiter=500)
    arima_rolling.append(
        np.expm1(float(np.asarray(result.forecast(steps=1))[0]))
    )
rolling["arima"] = np.clip(arima_rolling, 0, None)

# SARIMA
sarima_order = (2, 1, 2)
sarima_seasonal_order = (0, 1, 1, 12)
sarima_multi_result = SARIMAX(
    train_log,
    order=sarima_order,
    seasonal_order=sarima_seasonal_order,
    trend="n",
    enforce_stationarity=True,
    enforce_invertibility=True,
).fit(disp=False, maxiter=500)
sarima_multi_log = sarima_multi_result.get_forecast(steps=TEST_MONTHS).predicted_mean
multi["sarima"] = np.expm1(np.asarray(sarima_multi_log)).clip(min=0)

sarima_rolling = []
for step in range(TEST_MONTHS):
    expanded_log = log_cases.iloc[: len(train_log) + step]
    result = SARIMAX(
        expanded_log,
        order=sarima_order,
        seasonal_order=sarima_seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False, maxiter=500)
    sarima_rolling.append(
        np.expm1(float(np.asarray(result.forecast(steps=1))[0]))
    )
rolling["sarima"] = np.clip(sarima_rolling, 0, None)

# Holt-Winters
hw_add_multi = ExponentialSmoothing(
    train_log, trend=None, seasonal="add", seasonal_periods=12
).fit(optimized=True)
hw_damped_multi = ExponentialSmoothing(
    train_log, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12
).fit(optimized=True)
hw_mul_multi = ExponentialSmoothing(
    train, trend=None, seasonal="mul", seasonal_periods=12
).fit(optimized=True)
multi["hw_add_log"] = np.clip(
    np.expm1(hw_add_multi.forecast(TEST_MONTHS).to_numpy()), 0, None
)
multi["hw_damped_log"] = np.clip(
    np.expm1(hw_damped_multi.forecast(TEST_MONTHS).to_numpy()), 0, None
)
multi["hw_mul_raw"] = np.clip(
    hw_mul_multi.forecast(TEST_MONTHS).to_numpy(), 0, None
)

for name, kind in [
    ("hw_add_log", "add_log"),
    ("hw_damped_log", "damped_log"),
    ("hw_mul_raw", "mul_raw"),
]:
    forecasts = []
    for step in range(TEST_MONTHS):
        end = len(train) + step
        if kind == "add_log":
            result = ExponentialSmoothing(
                log_cases.iloc[:end], trend=None, seasonal="add", seasonal_periods=12
            ).fit(optimized=True)
            forecasts.append(np.expm1(result.forecast(1).iloc[0]))
        elif kind == "damped_log":
            result = ExponentialSmoothing(
                log_cases.iloc[:end],
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=12,
            ).fit(optimized=True)
            forecasts.append(np.expm1(result.forecast(1).iloc[0]))
        else:
            result = ExponentialSmoothing(
                cases.iloc[:end], trend=None, seasonal="mul", seasonal_periods=12
            ).fit(optimized=True)
            forecasts.append(result.forecast(1).iloc[0])
    rolling[name] = np.clip(forecasts, 0, None)

# XGBoost
multi["xgboost"] = make_xgboost_forecast(train_log, test.index)

xgboost_rolling = []
for step in range(TEST_MONTHS):
    expanded_log = log_cases.iloc[: len(train_log) + step]
    next_date = test.index[step]
    prediction = make_xgboost_forecast(
        expanded_log, pd.DatetimeIndex([next_date])
    )[0]
    xgboost_rolling.append(prediction)
rolling["xgboost"] = np.asarray(xgboost_rolling)

# Prophet: tune on 2016-2017, then fit the selected scale in both protocols
prophet_validation_train = train_log.iloc[:-24]
prophet_validation_actual = cases.iloc[-48:-24]
prophet_scores = {}
for scale in [0.01, 0.05, 0.1, 0.3, 0.5]:
    validation_log_forecast = prophet_forecast(
        prophet_validation_train, prophet_validation_actual.index, scale
    )
    prophet_scores[scale] = mape(
        prophet_validation_actual,
        validation_log_forecast,
    )
prophet_scale = min(prophet_scores, key=prophet_scores.get)

multi["prophet"] = prophet_forecast(train_log, test.index, prophet_scale)

prophet_rolling = []
for step in range(TEST_MONTHS):
    expanded_log = log_cases.iloc[: len(train_log) + step]
    next_date = test.index[step]
    prophet_rolling.append(
        prophet_forecast(
            expanded_log,
            pd.DatetimeIndex([next_date]),
            prophet_scale,
        )[0]
    )
rolling["prophet"] = np.asarray(prophet_rolling)

# Save model forecasts for both protocols
multi.round(2).to_csv(OUTPUT_PATH / "test_forecasts_multistep.csv", index_label="date")
rolling.round(2).to_csv(OUTPUT_PATH / "test_forecasts_rolling.csv", index_label="date")

# One comparison table with RMSE, MAE, and MAPE
rows = []
for model in [column for column in multi.columns if column != "actual"]:
    rows.append({
        "model": model,
        "multi_RMSE": rmse(multi["actual"], multi[model]),
        "multi_MAE": mae(multi["actual"], multi[model]),
        "multi_MAPE (%)": mape(multi["actual"], multi[model]),
        "rolling_RMSE": rmse(rolling["actual"], rolling[model]),
        "rolling_MAE": mae(rolling["actual"], rolling[model]),
        "rolling_MAPE (%)": mape(rolling["actual"], rolling[model]),
    })

comparison = pd.DataFrame(rows).sort_values("rolling_RMSE").reset_index(drop=True)
comparison.round(3).to_csv(OUTPUT_PATH / "protocol_comparison.csv", index=False)
print(comparison.round(3).to_string(index=False))
