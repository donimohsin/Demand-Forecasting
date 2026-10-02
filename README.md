# 📦 Supply Chain Demand Forecasting

SKU-level **28-day demand forecasts** for a multi-store FMCG retailer, with **prediction intervals**,
**explainability** and an interactive dashboard. The model uses promotions, prices, holidays and sales history
to beat the traditional moving-average approach.

**Developed by Mohammad Mohsin Doni**

---

## Business problem
A retailer's moving-average forecasts ignore promotions, holidays and price changes, causing stockouts on
popular items and excess stock on slow movers. This project replaces them with an ML forecaster that:

- forecasts every store x SKU up to **28 days ahead** (multi-horizon)
- uses **known-in-advance inputs**: promo calendar, prices, holidays, calendar features
- outputs **P10 / P50 / P90** so planners can choose a service level (probabilistic forecast)
- explains what drives demand (permutation importance)
- is evaluated against realistic baselines using **WMAPE**

## Results (56-day backtest on held-out data)

| Method | WMAPE | MAE | RMSE |
|---|---|---|---|
| 28-day moving average (legacy) | 22.7% | 7.07 | 10.18 |
| Seasonal naive | 26.0% | 8.10 | 11.78 |
| **Gradient boosting (this project)** | **16.0%** | **4.99** | **6.99** |

- **29.3% WMAPE reduction** vs. the best baseline (target was 15-20%)
- 80% prediction interval coverage: **77.2%**
- Top demand drivers: recent sales level (28d / 7d means), SKU, day of week, promotions

> These numbers come from a **synthetic dataset** built to behave like retail data (weekly and yearly seasonality,
> trend, holidays, promotions, price effects, noise). Re-run on real data (M5 / Favorita) to get real-world figures.

## How it works

```
sales.csv ──► feature engineering ──► Gradient Boosting (Poisson loss)  ──► P50 forecast
 (store x SKU     lags >= 28 days        + 2 quantile models (P10, P90)  ──► prediction interval
  x day)          rolling stats          trained on 120 series            ──► backtest + metrics
                  promo / price /                                          ──► permutation importance
                  holiday / calendar                                       ──► Streamlit dashboard
```

**Key design choice:** every sales-derived feature (lags, rolling means) is at least 28 days old. One global
model can therefore forecast *any* day in the next 28 days directly, with no error-compounding recursion and no
data leakage. Promo, price and holiday features are known in advance, so they can be used for future days.

## Project structure
```
demand-forecasting/
├── app.py                 # Streamlit dashboard
├── src/
│   ├── generate_data.py   # synthetic retail data (history + 28 future days)
│   ├── features.py        # lag / rolling / calendar / holiday / price features, WMAPE
│   └── train.py           # backtest, baselines, quantile models, importance, forecast
├── data/                  # sales.csv (generated)
├── outputs/               # metrics, backtest, forecast, importance (generated)
├── models/                # saved models (generated)
├── requirements.txt
└── Dockerfile
```

## Getting started
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python src/generate_data.py      # creates data/sales.csv
python src/train.py              # trains, backtests, forecasts, writes outputs/
streamlit run app.py
```
Docker: `docker build -t demand-forecast . && docker run -p 8501:8501 demand-forecast`

## Using your own data
Provide `data/sales.csv` with columns:
`date, store, sku, category, price, promo, holiday, units` (units = NaN for future days you want forecast,
with promo/price/holiday filled in for those days). Then run `python src/train.py`.

## Retraining
Re-run `python src/train.py` whenever new sales arrive (daily/weekly). It can be scheduled with Windows Task
Scheduler or cron. The by-week accuracy chart shows when performance drifts and retraining is needed.

## Dashboard
- **Forecast:** history, backtest, forecast and 80% band per store/SKU, with promo days highlighted
- **Accuracy:** WMAPE by category and by week, model vs. baselines
- **Drivers:** feature importance
- **Table:** full forecast with CSV download

*(Add screenshots here.)*

## Limitations
- Synthetic data; real data brings stockouts (censored demand), new products and price competition
- Intervals are slightly narrow (77% vs. 80% target); calibrate with conformal prediction for production
- Single model family; no deep learning

## Future improvements
- Temporal Fusion Transformer (`pytorch-forecasting`) on a GPU (Colab/Kaggle), compared against this baseline
- SHAP explanations, web-search-trend and macroeconomic features
- Conformal calibration, hierarchical reconciliation (store -> region -> total)
- FastAPI endpoint, drift monitoring (Evidently AI), automated retraining pipeline

## Author
**Mohammad Mohsin Doni**, developed as part of an AI internship project.
