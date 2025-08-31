# app.py
import streamlit as st
from modules.logger import get_logger
from modules.data_loader import load_all
from modules.api_clients import fetch_weather_forecast, fetch_electricity_prices
from modules.features import create_time_features, rolling_features, make_supervised
from modules.forecasting import train_model, predict_horizon
from modules.optimizer import optimize_schedule_pulp, schedule_to_profile
from modules.viz import plot_price_curve, plot_hourly_comparison
from modules.db import init_db, log_run, get_runs
import pandas as pd
import os
from datetime import datetime

logger = get_logger("seo_ui")
init_db()

st.set_page_config(page_title="Smart Energy Optimizer Pro", layout="wide")
st.title("? Smart Energy Optimizer — Pro (API + ML + Optimization)")

st.sidebar.header("Data & APIs")
use_sample = st.sidebar.checkbox("Use bundled sample data (recommended)", True)
if use_sample:
    baseline_path = "data/baseline_fixed_load.csv"
    appliances_path = "data/appliances.csv"
    prices_path = "data/tou_prices.csv"
    history_path = "data/hourly_sample_usage.csv"
else:
    baseline_path = st.sidebar.file_uploader("Baseline CSV", type=["csv"])
    appliances_path = st.sidebar.file_uploader("Appliances CSV", type=["csv"])
    prices_path = st.sidebar.file_uploader("TOU prices CSV", type=["csv"])
    history_path = st.sidebar.file_uploader("Historical usage CSV", type=["csv"])

try:
    baseline_df, appliances_df, prices_df, history_df = load_all(baseline_path, appliances_path, prices_path, history_path)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

st.subheader("TOU Price Curve")
st.plotly_chart(plot_price_curve(prices_df), use_container_width=True)

# Optional: fetch weather for forecasting if API key is set
st.sidebar.markdown("### External Data (optional)")
if st.sidebar.button("Fetch weather (if API key present)"):
    # default lat/lon example (New York)
    lat = st.sidebar.number_input("lat", value=40.7128)
    lon = st.sidebar.number_input("lon", value=-74.0060)
    w = fetch_weather_forecast(lat, lon)
    if w is None:
        st.sidebar.info("No weather data available (check OPENWEATHER_API_KEY or connection).")
    else:
        st.sidebar.success("Fetched weather data (see logs).")

# Forecasting panel
with st.expander("Forecast hourly demand (optional)"):
    if history_df is None:
        st.info("No historical usage provided; please upload history CSV to train model.")
    else:
        df = create_time_features(history_df)
        df = rolling_features(df)
        df = make_supervised(df)
        # pick features automatically
        feature_cols = [c for c in df.columns if c.startswith(('hour','dayofweek','is_weekend','roll_mean_','lag_'))]
        st.write("Features used:", feature_cols[:10], "...")
        res = train_model(df, feature_cols, target='kwh', test_frac=0.2, use_xgb=False)
        st.write(f"Trained. MAE: {res['mae']:.3f}, RMSE: {res['rmse']:.3f}")
        model = res['model']
        future = predict_horizon(model, df.tail(72), feature_cols, horizon=24)
        st.line_chart(future.set_index('datetime')['kwh'])

# Optimization
st.subheader("Optimize today's flexible appliance schedule")
opt_schedule = optimize_schedule_pulp(appliances_df, baseline_df, prices_df)
optimized_profile = schedule_to_profile(baseline_df, appliances_df, opt_schedule)

# Build naive schedule for baseline (earliest start)
naive_sched = {}
for _, r in appliances_df.iterrows():
    if r['flexible'] == 1:
        est = int(r['earliest_start'])
        dur = int(r['duration_hours'])
        naive_sched[r['name']] = [(est + i) % 24 for i in range(dur)]
naive_profile = schedule_to_profile(baseline_df, appliances_df, naive_sched)

def compute_cost(profile_df, prices_df):
    pm = dict(zip(prices_df['hour'], prices_df['price_per_kwh']))
    return float((profile_df['total_kwh'] * profile_df['hour'].map(pm)).sum())

baseline_cost = compute_cost(naive_profile, prices_df)
optimized_cost = compute_cost(optimized_profile, prices_df)
savings = baseline_cost - optimized_cost

c1, c2, c3 = st.columns(3)
c1.metric("Baseline cost (day)", f"${baseline_cost:.2f}")
c2.metric("Optimized cost (day)", f"${optimized_cost:.2f}")
c3.metric("Savings/day", f"${savings:.2f}")

st.plotly_chart(plot_hourly_comparison(naive_profile, optimized_profile), use_container_width=True)

# show schedule
st.subheader("Optimized schedule")
rows = []
for k,v in opt_schedule.items():
    rows.append({"appliance":k, "hours":", ".join(map(str, v)) if v else "—"})
st.table(pd.DataFrame(rows))

# Log run
if st.button("Log this run"):
    run_time = datetime.utcnow().isoformat()
    from modules.db import log_run
    log_run(run_time, baseline_cost, optimized_cost, savings)
    st.success("Run logged.")
    st.dataframe(get_runs())

st.sidebar.header("Developer tools")
if st.sidebar.checkbox("Show raw data"):
    st.sidebar.write("Appliances")
    st.sidebar.write(appliances_df)
    st.sidebar.write("Baseline")
    st.sidebar.write(baseline_df)
    st.sidebar.write("Prices")
    st.sidebar.write(prices_df)
