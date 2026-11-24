import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "data_2000_2019.csv"
DENGUE_PATH = ROOT / "data" / "processed" / "rj_monthly_dengue.csv"
OUT = ROOT / "outputs"


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mae(actual, forecast):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def mape(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def load_population():
    return pd.read_csv(
        DENGUE_PATH,
        parse_dates=["date"],
        index_col="date",
    )[["population"]].sort_index()


def load_climate(climate_path):
    """
    Load a monthly Rio climate CSV.

    Expected columns:
      date, mean_temperature, total_precipitation

    If station-level INMET data is supplied, this function also accepts
    station_date, mean_temperature, precipitation_mm and aggregates by month.
    """
    climate = pd.read_csv(climate_path)

    date_column = "date"
    if "station_date" in climate.columns:
        date_column = "station_date"
    elif "data" in climate.columns:
        date_column = "data"

    temperature_column = None
    for candidate in ["mean_temperature", "tmean", "temperature", "temperatura_media"]:
        if candidate in climate.columns:
            temperature_column = candidate
            break

    precipitation_column = None
    for candidate in [
        "total_precipitation",
        "precipitation_mm",
        "precipitation",
        "precip_total",
        "precipitacao",
    ]:
        if candidate in climate.columns:
            precipitation_column = candidate
            break

    if date_column not in climate.columns:
        raise ValueError("Climate file needs a date/station_date/data column")

    climate["date"] = pd.to_datetime(climate[date_column], errors="coerce")
    climate = climate.dropna(subset=["date"])
    climate = climate.set_index("date")

    monthly = pd.DataFrame(index=pd.date_range(
        climate.index.min(), climate.index.max(), freq="MS"
    ))
    if temperature_column:
        monthly["mean_temperature"] = climate[temperature_column].resample("MS").mean()
    if precipitation_column:
        monthly["total_precipitation"] = climate[precipitation_column].resample("MS").sum()

    if monthly.empty or len(monthly.columns) == 0:
        raise ValueError(
            "Climate file needs mean_temperature/tmean/temperature and/or "
            "total_precipitation/precipitation_mm/precipitation"
        )
    return monthly.sort_index()


def build_dataset(climate_path=None):
    dengue = pd.read_csv(
        DENGUE_PATH,
        parse_dates=["date"],
        index_col="date",
    )[["cases", "population"]].sort_index()

    dataset = dengue.copy()

    if climate_path:
        dataset = dataset.join(load_climate(climate_path), how="left")
    else:
        raw = pd.read_csv(RAW_PATH)
        rio = raw[(raw["state_code"] == 33) & (raw["year"] >= 2001)]
        temperature = (
            rio.assign(
                mean_temperature=(rio["tmax"].astype(float) + rio["tmin"].astype(float)) / 2
            )
            .groupby(["year", "month"], as_index=False)["mean_temperature"]
            .mean()
        )
        temperature["date"] = pd.to_datetime(
            dict(year=temperature["year"], month=temperature["month"], day=1)
        )
        temperature = temperature.set_index("date")[["mean_temperature"]].sort_index()
        dataset = dataset.join(temperature, how="left")

    return dataset


def create_features(dataset, lags, covariate_columns=None):
    cases = np.log1p(dataset["cases"])
    feature_columns = ["month"]
    if covariate_columns is None:
        covariate_columns = [
            column
            for column in ["mean_temperature", "total_precipitation", "population"]
            if column in dataset.columns
        ]

    feature_data = pd.DataFrame(index=dataset.index)
    feature_data["target"] = cases
    feature_data["month"] = dataset.index.month

    for lag in range(1, 13):
        feature_data[f"lag_{lag}"] = cases.shift(lag)
    for column in covariate_columns:
        for lag in lags:
            feature_data[f"{column}_lag_{lag}"] = dataset[column].shift(lag)

    feature_data["rolling_mean_3"] = cases.shift(1).rolling(3).mean()
    return feature_data, covariate_columns


def fit_predict_xgboost(train_features, test_dates, params, covariate_columns):
    train_end = train_features.index.max()
    feature_columns = [
        column
        for column in train_features.columns
        if column not in ["target"]
    ]

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(train_features[feature_columns], train_features["target"])

    # Recursive XGBoost cannot produce future covariates. This implementation
    # uses rolling one-step forecasts and known/observed covariate history.
    return model, feature_columns


def recursive_xgboost_full(dataset, test_dates, params, covariates):
    cases = np.log1p(dataset["cases"])
    covariate_columns = [
        column
        for column in ["mean_temperature", "total_precipitation", "population"]
        if column in covariates
    ]
    train_features = create_features(dataset.loc[: test_dates.min() - pd.offsets.MonthBegin()],
                                     [1, 2, 3],
                                     covariate_columns)[0]
    train_features = train_features.dropna()

    feature_columns = [column for column in train_features if column != "target"]
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(train_features[feature_columns], train_features["target"])

    history_log = cases.copy()
    train_end = train_features.index.max()
    known_covariates = dataset[covariate_columns].loc[:train_end].copy()
    predictions = []

    for date in test_dates:
        row = {"month": date.month}
        for lag in range(1, 13):
            row[f"lag_{lag}"] = history_log.loc[:date - pd.DateOffset(months=lag)].iloc[-1]
        row["rolling_mean_3"] = history_log.loc[:date - pd.DateOffset(months=1)].tail(3).mean()

        for column in covariate_columns:
            for lag in [1, 2, 3]:
                source_date = date - pd.DateOffset(months=lag)
                if source_date in known_covariates.index:
                    row[f"{column}_lag_{lag}"] = known_covariates.loc[source_date, column]
                else:
                    proxy_date = source_date - pd.DateOffset(years=1)
                    if proxy_date in known_covariates.index:
                        row[f"{column}_lag_{lag}"] = known_covariates.loc[proxy_date, column]
                    else:
                        row[f"{column}_lag_{lag}"] = known_covariates[column].iloc[-lag]

        prediction = float(model.predict(pd.DataFrame([row], columns=feature_columns))[0])
        predictions.append(np.expm1(prediction))
        history_log.loc[date] = prediction

    return np.clip(predictions, 0, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--climate",
        type=Path,
        default=None,
        help="CSV with date and mean_temperature/total_precipitation",
    )
    args = parser.parse_args()
    climate_path = args.climate if args.climate else None

    dataset = build_dataset(climate_path)
    test_dates = pd.date_range("2018-01-01", "2019-12-01", freq="MS")

    params = {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "subsample": 1.0,
    }

    baseline_covariates = []
    baseline_forecast = recursive_xgboost_full(
        dataset[["cases", "population"]],
        test_dates,
        params,
        baseline_covariates,
    )

    covariates = [column for column in dataset.columns if column in [
        "mean_temperature",
        "total_precipitation",
        "population",
    ]]
    covariate_forecast = recursive_xgboost_full(
        dataset,
        test_dates,
        params,
        covariates,
    )

    actual = dataset.loc[test_dates, "cases"].to_numpy()
    results = pd.DataFrame([
        {
            "model": "XGBoost without climate",
            "covariates": "lags 1-12 only",
            "RMSE": rmse(actual, baseline_forecast),
            "MAE": mae(actual, baseline_forecast),
            "MAPE (%)": mape(actual, baseline_forecast),
        },
        {
            "model": "XGBoost with climate",
            "covariates": ", ".join(covariates),
            "RMSE": rmse(actual, covariate_forecast),
            "MAE": mae(actual, covariate_forecast),
            "MAPE (%)": mape(actual, covariate_forecast),
        },
    ])
    results.to_csv(OUT / "xgboost_climate_comparison.csv", index=False)
    print(results.round(3).to_string(index=False))
    print(
        "\nCovariates used:",
        covariates,
        "\nPopulation source: existing processed RJ population column.",
        "\nExternal precipitation: supply --climate data/climate/rj_monthly.csv",
        "with date, mean_temperature, and total_precipitation columns.",
    )


if __name__ == "__main__":
    main()
