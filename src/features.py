"""Feature engineering. All sales-derived features use lags >= MIN_LAG (28 days), so ONE global model
can forecast any day in the next 28 days without recursion or leakage. Promo / price / holiday /
calendar features are known in advance and carry the short-term signal."""
import numpy as np, pandas as pd

MIN_LAG = 28
KEYS = ["store", "sku"]
CATS = ["store", "sku", "category"]

def build_features(df):
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True).copy()
    g = df.groupby(KEYS)["units"]
    for lag in (28, 35, 42, 56, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(MIN_LAG)
    grp = [df["store"], df["sku"]]
    for w in (7, 28, 91):
        df[f"rmean_{w}"] = shifted.groupby(grp).transform(lambda x: x.rolling(w, min_periods=1).mean())
    df["rstd_28"] = shifted.groupby(grp).transform(lambda x: x.rolling(28, min_periods=7).std())
    # calendar
    d = df["date"]
    df["dow"], df["month"], df["dom"] = d.dt.dayofweek, d.dt.month, d.dt.day
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # holiday proximity (known in advance)
    hol_dates = np.sort(df.loc[df["holiday"] == 1, "date"].unique())
    dd = d.values
    nxt = np.searchsorted(hol_dates, dd, side="left")
    prv = np.searchsorted(hol_dates, dd, side="right") - 1
    to_next = np.where(nxt < len(hol_dates), (hol_dates[np.minimum(nxt, len(hol_dates) - 1)] - dd) / np.timedelta64(1, "D"), 30)
    since = np.where(prv >= 0, (dd - hol_dates[np.maximum(prv, 0)]) / np.timedelta64(1, "D"), 30)
    df["days_to_holiday"], df["days_since_holiday"] = np.clip(to_next, 0, 30), np.clip(since, 0, 30)
    # price features
    med = df.groupby(KEYS)["price"].transform("median")
    df["price_ratio"] = df["price"] / med
    for c in CATS:
        df[c + "_id"] = df[c].astype("category").cat.codes
    return df

FEATURES = (["lag_28", "lag_35", "lag_42", "lag_56", "lag_364", "rmean_7", "rmean_28", "rmean_91", "rstd_28",
             "dow", "month", "dom", "weekofyear", "is_weekend", "holiday", "days_to_holiday",
             "days_since_holiday", "promo", "price", "price_ratio"] + [c + "_id" for c in CATS])
CAT_FEATURES = [c + "_id" for c in CATS]

def wmape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum())
