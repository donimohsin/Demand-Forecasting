"""Streamlit dashboard.  Run: streamlit run app.py   (after: python src/generate_data.py && python src/train.py)"""
import json, os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demand Forecasting", page_icon="📦", layout="wide")
st.title("📦 Supply Chain Demand Forecasting")
st.caption("28-day SKU-level forecasts with prediction intervals, explainability and accuracy tracking")

if not os.path.exists("outputs/forecast.csv"):
    st.error("No results found. Run `python src/generate_data.py` and then `python src/train.py` first.")
    st.stop()

@st.cache_data
def load():
    sales = pd.read_csv("data/sales.csv", parse_dates=["date"]).dropna(subset=["units"])
    fc = pd.read_csv("outputs/forecast.csv", parse_dates=["date"])
    bt = pd.read_csv("outputs/backtest.csv", parse_dates=["date"])
    return (sales, fc, bt, json.load(open("outputs/metrics.json")),
            pd.read_csv("outputs/by_category.csv"), pd.read_csv("outputs/by_week.csv", parse_dates=["week"]),
            pd.read_csv("outputs/feature_importance.csv"))
sales, fc, bt, m, by_cat, by_week, imp = load()

# ---- KPI row
gb, base = m["gradient_boosting"], m["moving_average_28d"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model WMAPE", f"{gb['WMAPE']:.1%}")
c2.metric("Moving-average baseline", f"{base['WMAPE']:.1%}")
c3.metric("Error reduction", f"{m['wmape_reduction_vs_moving_average_pct']:.1f}%")
c4.metric("80% interval coverage", f"{m['interval_80pct_coverage']:.0%}")
st.caption(f"Backtest period: {m['test_period']}  |  Forecast horizon: {m['forecast_horizon']}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🎯 Accuracy", "🔍 What drives demand", "📋 Forecast table"])

with tab1:
    a, b, c = st.columns(3)
    store = a.selectbox("Store", sorted(fc["store"].unique()))
    cat = b.selectbox("Category", sorted(fc["category"].unique()))
    sku = c.selectbox("SKU", sorted(fc.loc[fc["category"] == cat, "sku"].unique()))
    h = sales[(sales.store == store) & (sales.sku == sku)].tail(90)
    f = fc[(fc.store == store) & (fc.sku == sku)]
    t = bt[(bt.store == store) & (bt.sku == sku)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(h["date"], h["units"], color="#444", lw=1.2, label="Actual sales")
    ax.plot(t["date"], t["pred"], color="#f39c12", lw=1, ls="--", label="Backtest prediction")
    ax.fill_between(f["date"], f["p10"], f["p90"], color="#3b82f6", alpha=0.2, label="80% interval")
    ax.plot(f["date"], f["forecast"], color="#2563eb", lw=2, label="Forecast")
    promo_days = f[f["promo"] == 1]["date"]
    for d in promo_days:
        ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#22c55e", alpha=0.15)
    ax.set_ylabel("Units per day"); ax.legend(loc="upper left", ncol=4, fontsize=8); ax.grid(alpha=.25)
    fig.autofmt_xdate(); st.pyplot(fig)
    if len(promo_days):
        st.caption("Green bands = planned promotion days (a known input to the model).")
    d1, d2, d3 = st.columns(3)
    d1.metric("Expected units (next 28 days)", f"{f['forecast'].sum():,.0f}")
    d2.metric("Plan for high case (P90)", f"{f['p90'].sum():,.0f}")
    d3.metric("Low case (P10)", f"{f['p10'].sum():,.0f}")
    st.info("💡 Stocking to the P90 line protects against stockouts; stocking to the P10 line minimises excess "
            "inventory. Pick the percentile that matches the cost of running out vs. holding stock.")

with tab2:
    l, r = st.columns(2)
    with l:
        st.subheader("WMAPE by category (lower is better)")
        st.bar_chart(by_cat.set_index("category")[["model", "baseline"]])
    with r:
        st.subheader("WMAPE by week")
        st.line_chart(by_week.set_index("week")[["model", "baseline"]])
    st.subheader("Model vs. baselines")
    st.dataframe(pd.DataFrame({k: m[k] for k in ["moving_average_28d", "seasonal_naive", "gradient_boosting"]}).T,
                 use_container_width=True)

with tab3:
    st.subheader("Feature importance (permutation importance, increase in MAE when shuffled)")
    top = imp.head(15).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(top["feature"], top["importance"], color="#2563eb"); ax2.grid(alpha=.25, axis="x")
    st.pyplot(fig2)
    st.write("Recent sales levels set the baseline; promotions, weekday, holidays and price then explain the "
             "short-term swings. This gives the supply-chain team a reason behind every number.")

with tab4:
    st.dataframe(f.round(1), use_container_width=True, height=400)
    st.download_button("Download full forecast (CSV)", fc.to_csv(index=False), "forecast.csv", "text/csv")
