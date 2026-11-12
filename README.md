# Phase 2: Classical forecasting of monthly dengue cases, Rio de Janeiro state

Forecasts monthly dengue cases for the state of Rio de Janeiro with classical time-series models (naive baselines, ARIMA, SARIMA, Holt-Winters), then compares them with RMSE, MAE and MAPE on a 24-month hold-out (Jan 2018 to Dec 2019).

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
