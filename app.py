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
    """Loads the trained machine learning model."""
    try:
        with open("best_knn_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            "⚠️ 'best_knn_model.pkl' not found! Please check your file directory."
        )
        return None


def generate_fallback_dictionary():
    """Generates a complete dictionary of simulated weather features for a 16-day horizon."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = [
        (now + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00")
        for i in range(16 * 24)
    ]

    np.random.seed(42)  # For reproducible fallback data arrays
    total_hours = len(timestamps)

    fallback_data = {
        "time": timestamps,
        "temperature_2m": np.random.uniform(-5.0, 12.0, total_hours).tolist(),
        "shortwave_radiation": np.random.uniform(0.0, 450.0, total_hours).tolist(),
        "direct_normal_irradiance": np.random.uniform(
            0.0, 700.0, total_hours
        ).tolist(),
        "diffuse_radiation": np.random.uniform(0.0, 120.0, total_hours).tolist(),
        "precipitation": np.random.choice(
            [0.0, 0.1, 0.4], total_hours, p=[0.88, 0.09, 0.03]
        ).tolist(),
        "cloud_cover": np.random.uniform(0.0, 100.0, total_hours).tolist(),
    }
    return fallback_data


@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    """Fetches a 16-day weather forecast array. Falls back to dictionary on failure."""
    url = "https://open-meteo.com"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(API_VARIABLES),
        "timezone": "auto",
        "forecast_days": 16,
    }

    try:
        response = requests.get(url, params=params)
        print(response)
        if response.status_code != 200 or not response.text.strip():
            return None, True

        data = response.json()
        if "hourly" not in data:
            return None, True

        return data["hourly"], False
    except Exception:
        return None, True


# --- MAIN INTERFACE CONFIGURATION ---
st.set_page_config(page_title="Calgary Power Performance Overview", layout="wide")

st.title("📈 Calgary Power Performance Overview")
st.caption(
    f"Model projections for Calgary Fire Hall Headquarters (System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')})"
)
st.markdown("---")

# 1. Pipeline Verification Guard
model = load_model()
if model is None:
    st.stop()

# 2. Ingest Weather Engine Arrays
hourly_raw, is_fallback = fetch_weather_forecast()

if is_fallback:
    st.toast("⚠️ API Rate Limit hit. Using internal simulation dictionary fallback.")
    hourly_raw = generate_fallback_dictionary()

# 3. Clean and Standardize Schema Structure
df_forecast = pd.DataFrame(hourly_raw)
df_forecast["time"] = pd.to_datetime(df_forecast["time"])

rename_dict = {
    api_var: feat_var for api_var, feat_var in zip(API_VARIABLES, FEATURE_NAMES)
}
df_forecast = df_forecast.rename(columns=rename_dict)

# 4. Generate Machine Learning Output Trends
df_forecast["predicted_kWh"] = model.predict(df_forecast[FEATURE_NAMES])
df_forecast["predicted_kWh"] = df_forecast["predicted_kWh"].clip(lower=0)

# --- TIMELINE INDEX SLICES ---
now_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
target_time_2h = now_hour + timedelta(hours=2)

row_current = df_forecast[df_forecast["time"] == now_hour]
row_2h = df_forecast[df_forecast["time"] == target_time_2h]

# --- KPI METRIC HERO ROW ---
a, b, c = st.columns(3)

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

# Card 2: 24-Hour Horizon Cumulative
df_24h_slice = df_forecast[
    df_forecast["time"].between(now_hour, now_hour + timedelta(hours=23))
]
total_24h = (
    df_24h_slice["predicted_kWh"].sum()
    if not df_24h_slice.empty
    else df_forecast["predicted_kWh"].iloc[:24].sum()
)
b.metric(label="Next 24h Cumulative Production", value=f"{total_24h:,.2f} kWh")

# Card 3: 15-Day Peak Daily Target
df_forecast["date"] = df_forecast["time"].dt.date
df_daily_agg = df_forecast.groupby("date")["predicted_kWh"].sum().reset_index()

tomorrow_dt = datetime.now().date() + timedelta(days=1)
df_daily_15 = df_daily_agg[
    (df_daily_agg["date"] >= tomorrow_dt)
    & (df_daily_agg["date"] < tomorrow_dt + timedelta(days=15))
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

# --- SECTION 1: NEXT 24 HOURS HOURLY TREND LINE ---
st.subheader("🕒 Next 24 Hours Detailed Hourly Trend")
if not df_24h_slice.empty:
    df_chart_24h = df_24h_slice.copy()
    df_chart_24h["Hour"] = df_chart_24h["time"].dt.strftime("%H:%M")
    df_chart_24h = df_chart_24h.rename(
        columns={"predicted_kWh": "Hourly Energy (kWh)"}
    )
    st.line_chart(data=df_chart_24h, x="Hour", y="Hourly Energy (kWh)")
else:
    st.info("Generating hourly window matrices...")

st.markdown("---")

# --- SECTION 2: 15-DAY INTERACTIVE DRILL DOWN CARDS ---
st.subheader("📅 15-Day Outlook: Click a day card to drill down")

# Generate the 15 chronological days array
available_days = sorted(df_daily_15["date"].unique())

# Split layout into horizontal column rows to represent selectable timeline cards
card_cols = st.columns(len(available_days))

# Initialize session state tracking variable to capture clicked cards seamlessly
if "selected_drilldown_date" not in st.session_state:
    st.session_state.selected_drilldown_date = available_days[0]

for idx, day in enumerate(available_days):
    day_total_kwh = df_daily_15[df_daily_15["date"] == day][
        "predicted_kWh"
    ].values[0]

    with card_cols[idx]:
        # Formulate card metadata
        card_label = day.strftime("%b %d")
        card_subtext = f"**{day_total_kwh:.1f}** kWh"

        # Apply a visual accent marker if the current card is selected
        if st.session_state.selected_drilldown_date == day:
            st.markdown(f"🎯 **{card_label}**")
            st.markdown(f"`{day_total_kwh:.2f} kWh`")
        else:
            st.markdown(f"🗓️ {card_label}")
            st.markdown(f"*{day_total_kwh:.1f} kWh*")

        # Hidden state modifier button serving as the interactive wrapper card
        if st.button("Select", key=f"btn_{day.strftime('%Y%m%d')}"):
            st.session_state.selected_drilldown_date = day
            st.rerun()

st.markdown("---")

# --- SECTION 3: TARGET DAY LIVE PROFILE DRILL DOWN ---
chosen_date = st.session_state.selected_drilldown_date
st.subheader(f"🔍 Deep Dive Analysis Profile: {chosen_date.strftime('%A, %d %B %Y')}")

# Filter the structural dataframe down to the hours matching the chosen tracking date
df_drilldown_day = df_forecast[df_forecast["date"] == chosen_date].copy()
df_drilldown_day["Hour Slot"] = df_drilldown_day["time"].dt.strftime("%H:%M")

if not df_drilldown_day.empty:
    col_chart, col_weather = st.columns([3, 2])

    with col_chart:
        st.markdown("**Hourly Projected Load Curve**")
        df_plot_drill = df_drilldown_day.rename(
            columns={"predicted_kWh": "Energy Profile (kWh)"}
        )
        st.bar_chart(data=df_plot_drill, x="Hour Slot", y="Energy Profile (kWh)")

    with col_weather:
        st.markdown("**Underlying Meteorological Profile Metrics**")

        # Standardizing UI nomenclature back to intuitive display formatting
        df_weather_display = df_drilldown_day[
            ["Hour Slot"] + FEATURE_NAMES
        ].rename(
            columns={
                "Calgary_Fire_Hall_Headquarters_temperature_2m": "Temp (°C)",
                "Calgary_Fire_Hall_Headquarters_shortwave_radiation": "Shortwave Rad (W/m²)",
                "Calgary_Fire_Hall_Headquarters_direct_normal_irradiance": "DNI (W/m²)",
                "Calgary_Fire_Hall_Headquarters_diffuse_radiation": "Diffuse Rad (W/m²)",
                "Calgary_Fire_Hall_Headquarters_precipitation": "Precipitation (mm)",
                "Calgary_Fire_Hall_Headquarters_cloud_cover": "Cloud Cover (%)",
            }
        )

        st.dataframe(
            df_weather_display.style.format(
                {
                    "Temp (°C)": "{:.1f}",
                    "Shortwave Rad (W/m²)": "{:.0f}",
                    "DNI (W/m²)": "{:.0f}",
                    "Diffuse Rad (W/m²)": "{:.0f}",
                    "Precipitation (mm)": "{:.1f}",
                    "Cloud Cover (%)": "{:.0f}%",
                }
            ),
            use_container_width=True,)