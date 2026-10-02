"""Generate a realistic synthetic FMCG retail dataset: daily sales per store x SKU with weekly/yearly
seasonality, trend, holidays, promotions and price changes.  Also writes 28 future days that contain
only the *known-in-advance* inputs (promo plan, price, holidays) with units = NaN.
Swap in a real dataset (M5, Favorita) by producing a CSV with the same columns."""
import numpy as np, pandas as pd

HOLIDAYS = ["01-01", "01-26", "02-14", "05-01", "08-15", "10-02", "10-31", "11-11", "12-25", "12-31"]
CATEGORIES = {"Beverages": 8, "Snacks": 8, "Dairy": 8, "Household": 8, "PersonalCare": 8}
STORES = {"Store_A": 1.0, "Store_B": 1.4, "Store_C": 0.7}

def generate(history_days=1096, future_days=28, end_date="2026-09-30", seed=7):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(end - pd.Timedelta(days=history_days - 1), periods=history_days + future_days)
    t = np.arange(len(dates))
    hol = pd.Series(dates.strftime("%m-%d")).isin(HOLIDAYS).values.astype(int)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values
    rows = []
    for cat, n in CATEGORIES.items():
        wk_amp = rng.uniform(0.05, 0.35)                       # weekend effect
        season_amp, season_phase = rng.uniform(0.05, 0.3), rng.uniform(0, 2 * np.pi)
        hol_up = rng.uniform(0.15, 0.6)
        for k in range(n):
            sku = f"{cat[:3].upper()}_{k+1:02d}"
            base = rng.lognormal(3.0, 0.5)
            price0 = rng.uniform(20, 300)
            trend = rng.uniform(-0.0001, 0.0004)
            for store, mult in STORES.items():
                # promotion windows every 40-90 days, 4-9 days long, planned in advance
                promo = np.zeros(len(dates), int)
                disc = np.zeros(len(dates))
                d = int(rng.integers(5, 40))
                while d < len(dates):
                    ln = int(rng.integers(4, 10))
                    promo[d:d + ln] = 1
                    disc[d:d + ln] = rng.uniform(0.10, 0.25)
                    d += int(rng.integers(40, 90))
                price = price0 * (1 - disc) * (1 + 0.0002 * t)
                mean = (base * mult * (1 + trend * t)
                        * (1 + wk_amp * (dow >= 5))
                        * (1 + season_amp * np.sin(2 * np.pi * doy / 365 + season_phase))
                        * (1 + 1.8 * disc)                               # promo uplift tied to discount
                        * (1 + hol_up * hol)
                        * rng.lognormal(0, 0.08, len(dates)))
                units = rng.poisson(mean).astype(float)
                units[history_days:] = np.nan                            # future: unknown
                rows.append(pd.DataFrame({"date": dates, "store": store, "sku": sku, "category": cat,
                                          "price": price.round(2), "promo": promo, "holiday": hol,
                                          "units": units}))
    return pd.concat(rows, ignore_index=True)

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/sales.csv", index=False)
    print(df.shape, "| series:", df.groupby(["store", "sku"]).ngroups,
          "| history to", df.dropna(subset=["units"]).date.max().date(), "| future to", df.date.max().date())
