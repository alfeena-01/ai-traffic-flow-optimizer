import os
import json
import pandas as pd
import joblib
import streamlit as st
import pydeck as pdk

# --- Resolve project root ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data")

# --- Load pre-trained model ---
model = joblib.load(MODEL_PATH)

st.title("🚦 AI Traffic Flow Optimizer Dashboard")

# --- Load datasets ---
traffic_data = pd.read_csv(os.path.join(DATA_PATH, "traffic_sensor_data.csv"))
weather_data = pd.read_csv(os.path.join(DATA_PATH, "weather_conditions.csv"))

# --- Load sensor locations ---
with open(os.path.join(DATA_PATH, "sensor_locations.json")) as f:
    sensor_data = json.load(f)

# IMPORTANT: check what "sensors" contains
# If it's already a list of dicts with lat/lon:
locations = pd.DataFrame(sensor_data["sensors"])

# If lat/lon are named differently (e.g. "lat","lng"), rename them:
if "lat" in locations.columns and "lng" in locations.columns:
    locations.rename(columns={"lat": "latitude", "lng": "longitude"}, inplace=True)

# --- Merge traffic + weather + locations ---
traffic_data["timestamp"] = pd.to_datetime(traffic_data["timestamp"])
weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
merged = traffic_data.merge(weather_data, on="timestamp", how="left")
traffic = merged.merge(locations, on="sensor_id")

# --- Compute defaults for sidebar ---
avg_speed = merged["average_speed_kmh"].mean()
avg_precip = merged["precipitation_mm"].mean()
avg_visib = merged["visibility_km"].mean()
lag1 = merged["vehicle_count"].iloc[-1]
lag2 = merged["vehicle_count"].iloc[-2]

# --- Sidebar inputs for prediction ---
hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
is_weekend = 1 if day_of_week in [5, 6] else 0

speed = st.sidebar.number_input("Average Speed (km/h)", value=float(avg_speed))
precipitation = st.sidebar.number_input("Precipitation (mm)", value=float(avg_precip))
visibility = st.sidebar.number_input("Visibility (km)", value=float(avg_visib))

# --- Build input row ---
sample = pd.DataFrame({
    "hour": [hour],
    "day_of_week": [day_of_week],
    "is_weekend": [is_weekend],
    "average_speed_kmh": [speed],
    "precipitation_mm": [precipitation],
    "visibility_km": [visibility],
    "vehicle_count_lag1": [lag1],
    "vehicle_count_lag2": [lag2]
})

# --- Prediction ---
prediction = model.predict(sample)[0]

st.subheader("🔮 Traffic Prediction")
st.write(f"**Hour:** {hour}:00")
st.write(f"**Day of Week:** {day_of_week} ({'Weekend' if is_weekend else 'Weekday'})")
st.write(f"**Predicted Vehicle Count:** {prediction:.0f}")

# --- Heatmap ---
st.title("🌍 Traffic Heatmap")

traffic["hour"] = traffic["timestamp"].dt.hour
selected_hour = st.sidebar.slider("Select Hour for Heatmap", 0, 23, 8)
traffic_filtered = traffic[traffic["hour"] == selected_hour]

# Only proceed if lat/lon exist
if "latitude" in traffic_filtered.columns and "longitude" in traffic_filtered.columns:
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=traffic_filtered.dropna(subset=["latitude", "longitude"]),
        get_position=["longitude", "latitude"],
        get_weight="vehicle_count",
        radiusPixels=60,
    )

    view_state = pdk.ViewState(
        latitude=traffic_filtered["latitude"].mean(),
        longitude=traffic_filtered["longitude"].mean(),
        zoom=11,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state))
else:
    st.warning("⚠️ No latitude/longitude data found in sensor_locations.json. Please add coordinates.")
