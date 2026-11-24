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
