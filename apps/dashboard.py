import os
import json
import pandas as pd
import joblib
import streamlit as st
import pydeck as pdk

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="AI Traffic Flow Optimizer", layout="wide")
st.title("🚦 AI Traffic Flow Optimizer Dashboard")

def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        return None

def load_csv(path):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to read CSV file {path}: {exc}")
        return pd.DataFrame()

def load_json(path):
    if not os.path.exists(path):
        st.error(f"JSON file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Unable to read JSON file {path}: {exc}")
        return {}

model = load_model(MODEL_PATH)
traffic_data = load_csv(os.path.join(DATA_PATH, "traffic_sensor_data.csv"))
weather_data = load_csv(os.path.join(DATA_PATH, "weather_conditions.csv"))
sensor_data = load_json(os.path.join(DATA_PATH, "sensor_locations.json"))

if traffic_data.empty or weather_data.empty:
    st.stop()

locations = pd.json_normalize(sensor_data.get("sensors", []))
if not locations.empty:
    locations.rename(
        columns={
            "coordinates.latitude": "latitude",
            "coordinates.longitude": "longitude",
            "location_name": "location",
        },
        inplace=True,
    )
else:
    st.warning("No sensor locations found in sensor_locations.json.")
    locations = pd.DataFrame(columns=["sensor_id", "latitude", "longitude", "location"])

traffic_data["timestamp"] = pd.to_datetime(traffic_data["timestamp"], errors="coerce")
weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"], errors="coerce")

merged = traffic_data.merge(weather_data, on="timestamp", how="left")
traffic = merged.merge(locations, on="sensor_id", how="left")

avg_speed = merged["average_speed_kmh"].mean(skipna=True) if "average_speed_kmh" in merged else 0.0
avg_precip = merged["precipitation_mm"].mean(skipna=True) if "precipitation_mm" in merged else 0.0
avg_visib = merged["visibility_km"].mean(skipna=True) if "visibility_km" in merged else 0.0

lag1 = merged["vehicle_count"].iloc[-1] if len(merged) >= 1 and "vehicle_count" in merged else 0
lag2 = merged["vehicle_count"].iloc[-2] if len(merged) >= 2 and "vehicle_count" in merged else lag1

hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
is_weekend = 1 if day_of_week in [5, 6] else 0

speed = st.sidebar.number_input("Average Speed (km/h)", value=float(avg_speed))
precipitation = st.sidebar.number_input("Precipitation (mm)", value=float(avg_precip))
visibility = st.sidebar.number_input("Visibility (km)", value=float(avg_visib))

sample = pd.DataFrame(
    {
        "hour": [hour],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "average_speed_kmh": [speed],
        "precipitation_mm": [precipitation],
        "visibility_km": [visibility],
        "vehicle_count_lag1": [lag1],
        "vehicle_count_lag2": [lag2],
    }
)

st.subheader("🔮 Traffic Prediction")
st.write(f"**Hour:** {hour}:00")
st.write(f"**Day of Week:** {day_of_week} ({'Weekend' if is_weekend else 'Weekday'})")

if model is not None:
    try:
        prediction = model.predict(sample)[0]
        st.write(f"**Predicted Vehicle Count:** {prediction:.0f}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
else:
    st.warning("No model loaded. Prediction is unavailable.")

st.title("🌍 Traffic Heatmap")

traffic["hour"] = traffic["timestamp"].dt.hour
selected_hour = st.sidebar.slider("Select Hour for Heatmap", 0, 23, 8)

traffic_filtered = traffic[
    (traffic["hour"] == selected_hour)
    & traffic["latitude"].notna()
    & traffic["longitude"].notna()
]

if not traffic_filtered.empty:
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=traffic_filtered,
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
    st.warning(
        "⚠️ No latitude/longitude data found for the selected hour. Please check sensor_locations.json and data files."
    )


use_predictions = st.sidebar.checkbox("Show Predicted Counts on Heatmap")

if use_predictions:
    traffic["predicted_count"] = model.predict(
        traffic[["hour","day_of_week","is_weekend",
                 "average_speed_kmh","precipitation_mm","visibility_km",
                 "vehicle_count_lag1","vehicle_count_lag2"]]
    )
    weight_col = "predicted_count"
else:
    weight_col = "vehicle_count"

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=traffic_filtered.dropna(subset=["latitude","longitude"]),
    get_position=["longitude","latitude"],
    get_weight=weight_col,
    radiusPixels=60,
)
selected_date = st.sidebar.date_input("Select Date", traffic["timestamp"].dt.date.min())
traffic_filtered = traffic[traffic["timestamp"].dt.date == selected_date]
