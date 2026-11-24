from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_PATH = ROOT / "data" / "processed" / "rj_monthly_dengue.csv"
RAW_PATH = ROOT / "data" / "raw" / "data_2000_2019.csv"

df = pd.read_csv(
    OUT / "ensemble_test_forecasts.csv",
    parse_dates=["date"],
)
df = df.set_index("date").sort_index()

# "Best overall" model: rolling SARIMA, with the lowest RMSE and MAE.
df["forecast"] = df["sarima"]
df["signed_error"] = df["forecast"] - df["actual"]
df["absolute_error"] = df["signed_error"].abs()
df["percentage_error"] = df["signed_error"] / df["actual"] * 100

by_month = df.groupby(df.index.month).agg(
    mean_absolute_error=("absolute_error", "mean"),
    mean_signed_error=("signed_error", "mean"),
    mean_actual=("actual", "mean"),
    mean_forecast=("forecast", "mean"),
)
month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
by_month.index = [month_names[i - 1] for i in by_month.index]

by_year = df.groupby(df.index.year).agg(
    mean_absolute_error=("absolute_error", "mean"),
    mean_signed_error=("signed_error", "mean"),
    mean_actual=("actual", "mean"),
    mean_forecast=("forecast", "mean"),
    rmse=("signed_error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
)

spring = df.loc["2019-03-01":"2019-05-01"]
peak_actual = df["actual"].idxmax()
peak_forecast = df.loc["2019-03-01":"2019-07-01", "forecast"].idxmax()

monthly_peak_rows = df.loc["2019-03-01":"2019-05-01"]
peak_underestimate = float(
    df.loc[peak_actual, "actual"] - df.loc[peak_actual, "forecast"]
)

by_month.to_csv(OUT / "error_by_calendar_month.csv")
by_year.to_csv(OUT / "error_by_year.csv")

# Plot actual vs rolling SARIMA and highlight early-2019 rise.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df["actual"], color="black", linewidth=2.5, marker="o",
        label="Actual")
ax.plot(df.index, df["forecast"], color="tab:red", linewidth=2,
        label="Rolling tuned SARIMA")
ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-04-01"),
           color="orange", alpha=0.12, label="Early-2019 rise")
ax.scatter([peak_actual], [df.loc[peak_actual, "actual"]],
           color="black", s=70, zorder=4)
ax.annotate(
    f"Actual peak: {df.loc[peak_actual, 'actual']:,.0f}",
    xy=(peak_actual, df.loc[peak_actual, "actual"]),
    xytext=(peak_actual - pd.DateOffset(months=5),
            df.loc[peak_actual, "actual"] - 800),
    arrowprops={"arrowstyle": "->", "color": "black"},
)
ax.set_ylabel("Dengue cases")
ax.set_title("Rolling tuned SARIMA: actual vs predicted, 2018–2019")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figures" / "09_error_actual_vs_sarima.png", dpi=180)
plt.close(fig)

# Train XGBoost on the full training period and report gain feature importance.
dengue = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    index_col="date",
).sort_index()
log_cases = np.log1p(dengue["cases"])

raw = pd.read_csv(RAW_PATH)
rio = raw[(raw["state_code"] == 33) & (raw["year"] >= 2001)].copy()
rio["mean_temperature"] = (rio["tmax"] + rio["tmin"]) / 2
temperature = (
    rio.groupby(["year", "month"], as_index=False)["mean_temperature"].mean()
)
temperature["date"] = pd.to_datetime(
    dict(year=temperature["year"], month=temperature["month"], day=1)
)
temperature = temperature.set_index("date")[["mean_temperature"]]

covariates = dengue[["population"]].join(temperature, how="left")
features = pd.DataFrame(index=dengue.index)
features["month"] = dengue.index.month
for lag in range(1, 13):
    features[f"lag_{lag}"] = log_cases.shift(lag)
features["rolling_mean_3"] = log_cases.shift(1).rolling(3).mean()
for column in covariates.columns:
    for lag in [1, 2, 3]:
        features[f"{column}_lag_{lag}"] = covariates[column].shift(lag)
features["target"] = log_cases

train = features.iloc[:-24].dropna()
xgb = XGBRegressor(
    max_depth=2,
    learning_rate=0.03,
    n_estimators=300,
    subsample=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
feature_columns = [column for column in train if column != "target"]
xgb.fit(train[feature_columns], train["target"])

importance = pd.DataFrame({
    "feature": feature_columns,
    "gain": xgb.feature_importances_,
}).sort_values("gain", ascending=False)
importance["gain_percent"] = importance["gain"] / importance["gain"].sum() * 100
importance.head(25).to_csv(OUT / "xgboost_feature_importance.csv", index=False)

print("\nAbsolute error by calendar month:")
print(by_month.round(1).to_string())
print("\nAbsolute error by year:")
print(by_year.round(1).to_string())
print("\nPeak actual:", peak_actual.strftime("%b %Y"),
      round(df.loc[peak_actual, "actual"], 1))
print("Forecast at actual peak:", round(df.loc[peak_actual, "forecast"], 1))
print("Peak underestimate:", round(peak_underestimate, 1))
print("\nTop XGBoost features:")
print(importance.head(10).round(4).to_string(index=False))
