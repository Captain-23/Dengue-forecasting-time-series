# Phase 2: Classical and machine-learning forecasting of monthly dengue cases, Rio de Janeiro state

Forecasts monthly dengue cases for the state of Rio de Janeiro with classical time-series models (naive baselines, ARIMA, SARIMA, Holt-Winters), XGBoost, and Prophet, then compares them with RMSE, MAE and MAPE on a 24-month hold-out (Jan 2018 to Dec 2019).

## Data

Place the raw file at `data/raw/data_2000_2019.csv` (not included in this folder).

- Monthly dengue cases for 558 IBGE micro-regions in Brazil, 2000 to 2019.
- Rio de Janeiro state = `state_code` 33 (18 micro-regions), aggregated to one monthly series.
- 2000 has no dengue counts for any micro-region, so the study period is January 2001 to December 2019 (228 months).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter as the notebook kernel.

## Run order

Run each notebook top to bottom (Restart and Run All). Each notebook depends on the output of the one before it.

1. `notebooks/01_preprocessing_stationarity.ipynb` writes `data/processed/rj_monthly_dengue.csv` and the stationarity tests.
2. `notebooks/02_classical_models.ipynb` fits the models and writes the forecasts and model logs.
3. `notebooks/03_evaluation_comparison.ipynb` computes the metrics and the comparison figures.

The `.py` files contain the same code in percent format (`# %%` marks each cell). They are for copying and version control. The `.ipynb` files are the executed notebooks.

## Folder layout

```
dengue_phase2/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # place data_2000_2019.csv here
│   └── processed/rj_monthly_dengue.csv
├── notebooks/
│   ├── 01_preprocessing_stationarity.ipynb / .py
│   ├── 02_classical_models.ipynb / .py
│   └── 03_evaluation_comparison.ipynb / .py
└── outputs/
    ├── stationarity_tests.csv
    ├── test_forecasts.csv
    ├── model_selection_log.csv
    ├── holtwinters_summary.csv
    ├── sarima_intervals.csv
    ├── metrics.csv
    ├── models/                  # ARIMA and SARIMA full summaries
    └── figures/                 # all plots used in the presentation
```

## Method in brief

- **Transform:** log(cases + 1) for ARIMA, SARIMA and Holt-Winters additive, to stabilise the epidemic swings. Holt-Winters multiplicative uses raw counts.
- **Split:** chronological. Train Jan 2001 to Dec 2017 (204 months), test Jan 2018 to Dec 2019 (24 months).
- **Differencing:** seasonal D = 1 (period 12) for SARIMA. Regular d = 1 for ARIMA and SARIMA.
- **Model selection:** among candidates whose residuals pass the Ljung-Box test (lags 12 and 24, burn-in excluded), pick the lowest AIC. AIC is only compared within one differencing order.
- **Evaluation:** RMSE, MAE and MAPE on the test window, all on the case-count scale. Each model is fitted once and forecasts the whole window, with no refitting.

## Two-protocol evaluation

Run `python evaluation_protocols.py` to generate a consistent comparison for every model.

- **Multi-step:** fit once on the January 2001–December 2017 training set and forecast all 24 test months using only training data.
- **One-step rolling:** refit at the start of each test month and append the actual observed value after each forecast.
- All metrics are computed after applying `expm1(log1p(x))`, using RMSE, MAE, and MAPE on original monthly case counts.

The numbers differ because the rolling protocol receives each new actual month and can correct drift or adjust for the 2019 outbreak, while the multi-step protocol accumulates forecast errors from one fixed origin. The multi-step result is fairer for honest 24-month planning or when no future observations will be available. The rolling result is fairer for operational monthly deployment, where the latest actual is available before producing the next month’s forecast.

## SARIMA/seasonal-naive blend validation

Run `python validation_blend.py` to search SARIMA weights from 0 to 1 in 0.05 increments.

- Fit window: January 2001–December 2014
- Validation window: January 2015–December 2017
- Test window: January 2018–December 2019
- Selection metric: MAPE
- Validation forecasts are rolling one-step SARIMA forecasts, with each observed validation month appended before the next forecast.
- The script asserts that the validation and test date sets do not overlap, and the test set is not read until after the weight search has selected and stored a weight.

The validation search is reported in `outputs/blend_validation_curve.csv`. The chosen weight and the legacy 60/40 weight are compared once on the test set in `outputs/blend_test_comparison.csv`.

## Expanding-window tuning for XGBoost and Prophet

Run `python tune_ml_models.py` to tune the machine-learning models using the training period only.

- Cross-validation: three expanding-window folds
- Validation block: 12 months per fold
- Folds: 2015–2016–2017 validation blocks after fitting on the preceding training history
- XGBoost grid: depth 2–3, learning rate 0.03–0.05, 300–500 trees, and subsample 0.8–1.0
- Prophet grid: changepoint prior 0.05–0.5 and seasonality prior 1–10
- CV metric: RMSE on the original monthly case scale after log inversion
- Final step: refit each winning model once on all 2001–2017 training data and evaluate once on 2018–2019

CV rankings are saved in `outputs/xgboost_cv_results.csv` and `outputs/prophet_cv_results.csv`. Final test metrics and forecasts are saved in `outputs/ml_cv_test_results.csv` and `outputs/ml_cv_test_forecasts.csv`.

## SARIMA/XGBoost ensemble

Run `python build_ensemble.py` to generate validation-selected ensemble results.

- Validation period: January 2015–December 2017
- Test period: January 2018–December 2019
- Weighted-average search: XGBoost weights from 0 to 1 in 0.05 increments
- Selection metric: validation RMSE
- Chosen weight: 15% tuned XGBoost / 85% rolling SARIMA
- Stacking model: linear regression fitted on the two validation predictions
- The test set is not accessed until both validation-selected choices are fixed

The weighted average improves test MAPE from 25.86% for rolling SARIMA to 22.54%, although it does not improve SARIMA's lower test RMSE or MAE. Linear stacking overfits the 2015–2017 prediction relationship and does not improve the test metrics.

## Monthly climate covariates

Run `python climate_xgboost.py` to compare XGBoost with and without covariates.

The existing raw file provides Rio de Janeiro `tmax` and `tmin`, from which the script constructs mean temperature. It does not provide precipitation. For temperature and precipitation, download Rio station or state observations from [INMET](https://portal.inmet.gov.br/) and state/municipal population series from [IBGE](https://www.ibge.gov.br/). The script currently uses the population column already present in the processed dengue data.

Create a monthly climate CSV in this format:

```text
date,mean_temperature,total_precipitation
2001-01-01,27.4,245.0
2001-02-01,27.1,198.3
```

Then run:

```bash
python climate_xgboost.py --climate data/climate/rj_monthly.csv
```

The loader also accepts INMET-style `station_date` plus `mean_temperature`/`precipitation_mm`, or Portuguese `data` plus `temperatura_media`/`precipitacao` columns. Temperature is aggregated to a monthly mean and precipitation to a monthly sum. Population is taken from the processed monthly series.

All covariates are shifted by 1, 2, and 3 months before training. The 2018–2019 comparison uses only covariate values available at each forecast origin: for missing future covariates, it applies a prior-year proxy for the lagged month. For operational deployment, current-month weather is not usable with lags 1–3 unless it is observed and treated as a known value. Population estimates or projections for future periods should use a year-ahead estimate or remain an unavailable covariate rather than using test-period population.

Current comparison using the existing temperature proxy and population:

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| XGBoost without covariates | 2,854 | 1,600 | 70.89% |
| XGBoost with population and mean temperature | 2,769 | 1,500 | 61.74% |

## Test-period error analysis

Run `python error_analysis.py` to produce:

- `outputs/error_by_calendar_month.csv`
- `outputs/error_by_year.csv`
- `outputs/xgboost_feature_importance.csv`
- `outputs/figures/09_error_actual_vs_sarima.png`

The primary forecast is the rolling tuned SARIMA, selected because it has the lowest test RMSE and MAE. It underestimates the May 2019 peak of 10,022 cases by 2,877 cases. The largest monthly absolute errors occur in March–May 2019, while 2018 is substantially more accurate.

Plain-language implications:

- Hospital surge planning should not rely on a point forecast alone during March–May: the model can miss a rapidly rising outbreak.
- An early-warning threshold may be reached when the forecast and the most recent observed month begin to rise together. Use a 2019-style sensitivity analysis rather than a fixed historical average.
- February–March are the clearest early-rise months for the dengue surveillance calendar, so weekly or monthly refreshes are important.
- XGBoost gain indicates that lagged cases, especially lag 1, dominate its prediction; recent case momentum matters more than the month indicator.
- A 1-month lagged temperature term also ranks highly. This supports adding lagged weather to outbreak monitoring, but population and temperature should not be supplied from future test values in deployment.
