import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model
model = joblib.load("../models/traffic_model.pkl")

st.title("🚦 AI Traffic Flow Optimizer Dashboard")

# Sidebar inputs
hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
is_weekend = 1 if day_of_week in [5,6] else 0
speed = st.sidebar.number_input("Average Speed (km/h)", value=40)
precipitation = st.sidebar.number_input("Precipitation (mm)", value=0.0)
visibility = st.sidebar.number_input("Visibility (km)", value=10.0)

# Build input row
sample = pd.DataFrame({
    "hour": [hour],
    "day_of_week": [day_of_week],
    "is_weekend": [is_weekend],
    "average_speed_kmh": [speed],
    "precipitation_mm": [precipitation],
    "visibility_km": [visibility],
    "vehicle_count_lag1": [100],  # placeholder
    "vehicle_count_lag2": [95]
})

prediction = model.predict(sample)[0]

st.subheader("🔮 Traffic Prediction")
st.write(f"Predicted Vehicle Count: {prediction:.0f}")

st.subheader("📊 Model Metrics")
st.write(f"MAE: {model.metrics_['mae']:.2f}")
st.write(f"R2 Score: {model.metrics_['r2']:.2f}")