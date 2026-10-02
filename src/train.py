"""Train + backtest + forecast.  Run: python src/train.py
1) Backtest on the last 56 days vs. two baselines (what a traditional system would do)
2) Refit on ALL history and forecast the next 28 days with P10 / P50 / P90 bands
3) Permutation importance for explainability"""
import json, os, sys, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.inspection import permutation_importance
from src.features import build_features, FEATURES, CAT_FEATURES, wmape

TEST_DAYS = 56
for folder in ("outputs", "models"):
    os.makedirs(folder, exist_ok=True)
df = pd.read_csv("data/sales.csv", parse_dates=["date"])
df = build_features(df)
hist = df.dropna(subset=["units"]).dropna(subset=["lag_28"])
last = hist["date"].max()
test_start = last - pd.Timedelta(days=TEST_DAYS - 1)
train, test = hist[hist["date"] < test_start], hist[hist["date"] >= test_start]
print(f"train rows {len(train):,} | test rows {len(test):,} | test {test_start.date()} -> {last.date()}")

def make(loss="poisson", q=None):
    kw = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40,
              l2_regularization=1.0, random_state=42, categorical_features=CAT_FEATURES)
    return HGB(loss="quantile", quantile=q, **kw) if loss == "quantile" else HGB(loss=loss, **kw)

# ---- 1. Backtest
model = make().fit(train[FEATURES], train["units"])
p50 = np.clip(model.predict(test[FEATURES]), 0, None)
q10 = make("quantile", .1).fit(train[FEATURES], train["units"]).predict(test[FEATURES])
q90 = make("quantile", .9).fit(train[FEATURES], train["units"]).predict(test[FEATURES])
q10, q90 = np.minimum(np.clip(q10, 0, None), p50), np.maximum(q90, p50)

y = test["units"].values
base_naive = test["lag_28"].fillna(test["rmean_28"]).values        # seasonal naive (same weekday, 4 weeks ago)
base_ma = test["rmean_28"].values                                     # 28-day moving average (legacy system)
res = {}
for name, p in [("moving_average_28d", base_ma), ("seasonal_naive", base_naive), ("gradient_boosting", p50)]:
    res[name] = {"WMAPE": round(wmape(y, p), 4), "MAE": round(float(np.abs(y - p).mean()), 3),
                 "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 3)}
best_base = min(res["moving_average_28d"]["WMAPE"], res["seasonal_naive"]["WMAPE"])
res["wmape_reduction_vs_best_baseline_pct"] = round(100 * (best_base - res["gradient_boosting"]["WMAPE"]) / best_base, 1)
res["wmape_reduction_vs_moving_average_pct"] = round(100 * (res["moving_average_28d"]["WMAPE"] - res["gradient_boosting"]["WMAPE"]) / res["moving_average_28d"]["WMAPE"], 1)
res["interval_80pct_coverage"] = round(float(((y >= q10) & (y <= q90)).mean()), 3)
res["test_period"] = f"{test_start.date()} to {last.date()}"
print(json.dumps(res, indent=2))

bt = test[["date", "store", "sku", "category", "units"]].copy()
bt["pred"], bt["p10"], bt["p90"], bt["baseline_ma"], bt["baseline_naive"] = p50, q10, q90, base_ma, base_naive
bt.to_csv("outputs/backtest.csv", index=False)

# per-category + weekly accuracy
def acc(g):
    return pd.Series({"model": wmape(g.units, g.pred), "baseline": wmape(g.units, g.baseline_ma)})
bt.groupby("category")[["units", "pred", "baseline_ma"]].apply(acc).round(4).to_csv("outputs/by_category.csv")
bt["week"] = bt["date"].dt.to_period("W").dt.start_time
bt.groupby("week")[["units", "pred", "baseline_ma"]].apply(acc).round(4).to_csv("outputs/by_week.csv")

# ---- 3. Explainability (permutation importance on a sample of the test set)
samp = test.sample(min(8000, len(test)), random_state=1)
pi = permutation_importance(model, samp[FEATURES], samp["units"], n_repeats=3, random_state=1,
                            scoring="neg_mean_absolute_error", n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
imp.to_csv("outputs/feature_importance.csv", index=False)
print("Top drivers:", ", ".join(imp.feature.head(5)))

# ---- 2. Refit on all history, forecast next 28 days
final = {"p50": make().fit(hist[FEATURES], hist["units"]),
         "p10": make("quantile", .1).fit(hist[FEATURES], hist["units"]),
         "p90": make("quantile", .9).fit(hist[FEATURES], hist["units"])}
fut = df[df["units"].isna()].copy()
fc = fut[["date", "store", "sku", "category", "promo", "price"]].copy()
fc["forecast"] = np.clip(final["p50"].predict(fut[FEATURES]), 0, None)
fc["p10"] = np.minimum(np.clip(final["p10"].predict(fut[FEATURES]), 0, None), fc["forecast"])
fc["p90"] = np.maximum(final["p90"].predict(fut[FEATURES]), fc["forecast"])
fc.to_csv("outputs/forecast.csv", index=False)
joblib.dump({"models": final, "features": FEATURES}, "models/forecaster.joblib")
res["forecast_horizon"] = f"{fut.date.min().date()} to {fut.date.max().date()}"
json.dump(res, open("outputs/metrics.json", "w"), indent=2)
print("Saved outputs/ and models/. Forecast rows:", len(fc))
