# main.py - SatGuard AI v2.0: ML Anomaly Detection
from skyfield.api import load, wgs84
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest

# === 1. Load real CubeSat orbits ===
stations_url = "https://celestrak.org/NORAD/elements/cubesat.txt"
satellites = load.tle_file(stations_url)
print(f"Loaded {len(satellites)} CubeSats")

sat = satellites[0]
ts = load.timescale()

# === 2. Generate 6 time points ===
hours = [0, 0, 0, 0, 0, 0]
minutes = [0, 10, 20, 30, 40, 50]
times = ts.utc(2025, 11, 6, hours, minutes, 0)

# === 3. Generate fake telemetry ===
data = []
for t in times:
    pos = sat.at(t)
    lat, lon = wgs84.latlon_of(pos)[:2]

    hour = t.utc.hour
    minute = t.utc.minute

   # INJECT FAILURE AFTER 40 MINUTES
    if minute >= 40:
        gyro_drift = 0.15 + 0.03 * (minute - 40) + np.random.normal(0, 0.02)
        battery = 85 - 2 * (minute - 40) + np.random.normal(0, 0.5)
        temp = 35 + np.random.normal(0, 3)
    else:
        gyro_drift = 0.01 * minute + np.random.normal(0, 0.05)
        battery = 98 - 0.008 * minute + np.random.normal(0, 0.3)
        temp = 22 + 10 * np.sin(hour * 0.5) + np.random.normal(0, 1.5)

    data.append({
        'time': t.utc_iso(),
        'lat': round(lat.degrees, 4),
        'lon': round(lon.degrees, 4),
        'temperature': round(temp, 2),
        'battery_level': round(battery, 2),
        'gyro_drift': round(gyro_drift, 4)
    })

df = pd.DataFrame(data)
df.to_csv("satguard_telemetry.csv", index=False)
print("Telemetry saved")

# === 4. Dashboard ===
st.title("SatGuard AI v2.0")
st.write("ML-powered satellite failure prediction")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Using live-generated data")

st.line_chart(df[['temperature', 'battery_level', 'gyro_drift']], height=300)

# === 5. ML Anomaly Detection ===
features = df[['temperature', 'battery_level', 'gyro_drift']].values
train_data = features[:4]
test_data = features[4:]

model = IsolationForest(contamination=0.3, random_state=42)
model.fit(train_data)
anomaly_scores = model.decision_function(test_data)
prediction = model.predict(test_data)

drift_rate = df['gyro_drift'].diff().mean()
rul_days = 999
if drift_rate > 0:
    rul_days = max(0, (0.8 - df['gyro_drift'].iloc[-1]) / drift_rate) * 10

st.subheader("AI Report")
if len(prediction) > 0 and prediction[-1] == -1:
    st.error(f"ANOMALY (Score: {anomaly_scores[-1]:.2f})")
    st.warning(f"Failure in **{rul_days:.1f} days**")
else:
    st.success("Healthy")

st.download_button("Download", df.to_csv(index=False), "satguard_telemetry.csv")