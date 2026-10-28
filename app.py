import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --- CONFIGURATION & CONSTANTS ---
LAT = 51.0447
LON = -114.0719

FEATURE_NAMES = [
    "Calgary_Fire_Hall_Headquarters_temperature_2m",
    "Calgary_Fire_Hall_Headquarters_shortwave_radiation",
    "Calgary_Fire_Hall_Headquarters_direct_normal_irradiance",
    "Calgary_Fire_Hall_Headquarters_diffuse_radiation",
    "Calgary_Fire_Hall_Headquarters_precipitation",
    "Calgary_Fire_Hall_Headquarters_cloud_cover",
]

API_VARIABLES = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "precipitation",
    "cloud_cover",
]


# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    """Loads the trained model from the local directory."""
    try:
        with open("best_knn_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            "⚠️ 'best_knn_model.pkl' not found! Please check that your model file sits in this exact workspace folder."
        )
        return None


@st.cache_data(ttl=600)
def fetch_weather_forecast():
    """Fetches a 15-day weather forecast from Open-Meteo API with explicit JSON parsing validation."""
    url = "https://open-meteo.com"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(API_VARIABLES),
        "timezone": "auto",
        "forecast_days": 15
    }

    try:
        response = requests.get(url, params=params)

        # Guard against Cloudflare walls, service outages, or API rate blocks
        if response.status_code != 200:
            return None

        # Verify that content actually exists and isn't empty space
        if not response.text or not response.text.strip():
            return None

        data = response.json()
        if "hourly" not in data:
            return None

        hourly_data = data["hourly"]
        df = pd.DataFrame(hourly_data)
        df["time"] = pd.to_datetime(df["time"])

        rename_dict = {
            api_var: feat_var
            for api_var, feat_var in zip(API_VARIABLES, FEATURE_NAMES)
        }
        df = df.rename(columns=rename_dict)
        return df

    except Exception:
        # Silently fail here so the layout guard below can intercept the null response cleanly
        return None


# --- MAIN APP LAYOUT (FASTERO KPI THEME) ---
st.set_page_config(page_title="Calgary Power Performance Overview", layout="wide")

st.title("📈 Calgary Power Performance Overview")
st.caption(
    f"Live model projections for Calgary Fire Hall Headquarters (Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')})"
)
st.markdown("---")

# 1. Load Model Assets
model = load_model()
if model is None:
    st.stop()  # Halt execution cleanly if the model object is broken or missing

# 2. Fetch API Forecast Data
df_forecast = fetch_weather_forecast()

# CRITICAL SECURITY GUARD: If the dataframe failed to load, break here gracefully instead of crashing
if df_forecast is None:
    st.warning(
        "🛑 **Unable to fetch live forecast data from Open-Meteo.** "
        "This usually happens when hitting temporary public IP rate limits. "
    )
    st.info("💡 **How to resolve:** Wait 60 seconds and refresh your browser page to re-establish the connection.")
    st.stop()  # Safely kills the execution thread before model.predict runs

# --- CALCULATE FORECAST METRICS (Only runs if data is completely valid) ---
now_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
target_time_2h = now_hour + timedelta(hours=2)

row_current = df_forecast[df_forecast["time"] == now_hour]
row_2h = df_forecast[df_forecast["time"] == target_time_2h]

# Generate model predictions
df_forecast["predicted_kWh"] = model.predict(df_forecast[FEATURE_NAMES])
df_forecast["predicted_kWh"] = df_forecast["predicted_kWh"].clip(lower=0)

# --- KPI METRIC CARD ROW ---
a, b, c = st.columns(3)

# Card 1: 2-Hour Ahead Target Forecast
if not row_2h.empty:
    pred_2h = row_2h["predicted_kWh"].values[0]
    if not row_current.empty:
        pred_now = row_current["predicted_kWh"].values[0]
        delta_val = pred_2h - pred_now
        delta_str = f"{delta_val:+.2f} kWh vs now"
    else:
        delta_str = None

    a.metric(
        label=f"Output Forecast (+2h at {target_time_2h.strftime('%H:%M')})",
        value=f"{pred_2h:.2f} kWh",
        delta=delta_str,
    )
else:
    a.metric(label="Output Forecast (+2h)", value="N/A")

# Card 2: Next 24 Hours Cumulative Prediction
df_24h = df_forecast[
    df_forecast["time"].between(now_hour, now_hour + timedelta(hours=24))
]
total_24h = df_24h["predicted_kWh"].sum()
b.metric(label="Next 24h Cumulative", value=f"{total_24h:,.2f} kWh")

# Card 3: 15-Day Peak Daily Output
df_forecast["date"] = df_forecast["time"].dt.date
df_daily = df_forecast.groupby("date")["predicted_kWh"].sum().reset_index()

tomorrow = datetime.now().date() + timedelta(days=1)
end_filter = tomorrow + timedelta(days=15)
df_daily_15 = df_daily[
    (df_daily["date"] >= tomorrow) & (df_daily["date"] < end_filter)
]

if not df_daily_15.empty:
    peak_row = df_daily_15.loc[df_daily_15["predicted_kWh"].idxmax()]
    c.metric(
        label=f"Projected Daily Peak ({peak_row['date'].strftime('%d %b')})",
        value=f"{peak_row['predicted_kWh']:.2f} kWh",
    )
else:
    c.metric(label="Projected Daily Peak", value="N/A")

st.markdown("---")

# --- VISUALIZATION SPLIT LAYOUT ---
st.subheader("📅 15-Day Daily Production Trend")

df_chart_data = df_daily_15.rename(
    columns={"date": "Timeline", "predicted_kWh": "Total Energy (kWh)"}
)
st.line_chart(data=df_chart_data, x="Timeline", y="Total Energy (kWh)")

with st.expander("🔍 View Itemized Daily Breakdown Data Table", expanded=False):
    st.dataframe(
        df_chart_data.style.format({"Total Energy (kWh)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )