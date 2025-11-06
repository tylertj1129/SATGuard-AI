# main.py - SatGuard AI v3.1: LIVE ML FAILURE PREDICTION
import streamlit as st
import pandas as pd
import numpy as np
from skyfield.api import load, wgs84
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="SatGuard AI", layout="wide")
st.title("SatGuard AI")
st.caption("Upload telemetry → Get failure prediction in seconds")

# === Upload or Generate Data ===
uploaded = st.file_uploader("Drop CSV (temperature, battery_level, gyro_drift)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Telemetry loaded")
else:
    st.info("Demo mode — generating failure data...")
    data = []
    for i, minute in enumerate([0, 10, 20, 30, 40, 50]):
        if minute >= 40:
            temp = 35 + np.random.normal(0, 3)
            battery = 85 - 2 * (minute - 40) + np.random.normal(0, 0.5)
            gyro = 0.15 + 0.03 * (minute - 40) + np.random.normal(0, 0.02)
        else:
            temp = 22 + 10 * np.sin(minute/60) + np.random.normal(0, 1.5)
            battery = 98 - 0.008 * minute + np.random.normal(0, 0.3)
            gyro = 0.01 * minute + np.random.normal(0, 0.05)
        data.append({
            'temperature': round(temp, 2),
            'battery_level': round(battery, 2),
            'gyro_drift': round(gyro, 4)
        })
    df = pd.DataFrame(data)

# === ML: Isolation Forest ===
features = df[['temperature', 'battery_level', 'gyro_drift']].values
model = IsolationForest(contamination=0.3, random_state=42)
model.fit(features)
scores = model.decision_function(features)
pred = model.predict(features)

# RUL Estimate
drift_rate = df['gyro_drift'].diff().mean()
rul = 999
if drift_rate > 0:
    rul = max(0, (0.8 - df['gyro_drift'].iloc[-1]) / drift_rate) * 10

# === Dashboard ===
col1, col2 = st.columns([2,1])
with col1:
    st.line_chart(df[['temperature', 'battery_level', 'gyro_drift']], height=400)
with col2:
    st.metric("Anomaly Score", f"{scores[-1]:.2f}")
    st.metric("RUL", f"{rul:.1f} days")

if pred[-1] == -1:
    st.error("FAILURE IMMINENT")
    st.warning(f"Reaction wheel dies in **{rul:.1f} days**")
else:
    st.success("Healthy")

st.download_button("Export", df.to_csv(index=False), "satguard_report.csv")