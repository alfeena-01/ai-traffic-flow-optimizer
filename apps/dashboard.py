import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Title
st.title("🚦 AI Traffic Flow Optimizer Dashboard")

# Sidebar: Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload Traffic CSV", type=["csv"])
if uploaded_file:
    traffic = pd.read_csv(uploaded_file)
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"])

    # Feature engineering
    traffic["hour"] = traffic["timestamp"].dt.hour
    traffic["day_of_week"] = traffic["timestamp"].dt.dayofweek
    traffic["is_weekend"] = traffic["day_of_week"].isin([5,6]).astype(int)
    traffic["vehicle_count_lag1"] = traffic["vehicle_count"].shift(1)
    traffic["vehicle_count_lag2"] = traffic["vehicle_count"].shift(2)
    traffic = traffic.dropna()

    # Features and target
    X = traffic[["hour","day_of_week","is_weekend","average_speed_kmh",
                 "precipitation_mm","visibility_km",
                 "vehicle_count_lag1","vehicle_count_lag2"]]
    y = traffic["vehicle_count"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = gb_model.score(X_test, y_test)

    st.subheader("📊 Model Performance")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.3f}")

    # Plot Predicted vs Actual
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.3, color="purple")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Vehicle Count")
    ax.set_ylabel("Predicted Vehicle Count")
    ax.set_title("Predicted vs Actual Traffic Volume")
    st.pyplot(fig)

    # User input prediction
    st.sidebar.subheader("🔮 Predict Traffic")
    hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
    speed = st.sidebar.number_input("Average Speed (km/h)", value=40)
    precipitation = st.sidebar.number_input("Precipitation (mm)", value=0.0)
    visibility = st.sidebar.number_input("Visibility (km)", value=10.0)

    # Build input row
    sample = pd.DataFrame({
        "hour": [hour],
        "day_of_week": [0],  # default Monday
        "is_weekend": [0],
        "average_speed_kmh": [speed],
        "precipitation_mm": [precipitation],
        "visibility_km": [visibility],
        "vehicle_count_lag1": [traffic["vehicle_count"].iloc[-1]],
        "vehicle_count_lag2": [traffic["vehicle_count"].iloc[-2]]
    })

    prediction = gb_model.predict(sample)[0]
    st.sidebar.write(f"**Predicted Vehicle Count:** {prediction:.0f}")
else:
    st.info("Please upload a traffic dataset to begin.")

import joblib
model = joblib.load("../models/traffic_model.pkl")
