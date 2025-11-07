# --------------------------------------------------------------
# SatGuard AI v4.1 – Robust Persistent Learning
# --------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="SatGuard AI", layout="wide")
st.title("SatGuard AI")
st.caption("Upload telemetry → AI predicts failure. Learns from every upload.")

# ------------------------------------------------------------------
# 1. PERSISTENT STORAGE (historical_data.csv)
# ------------------------------------------------------------------
HISTORY_FILE = Path("historical_data.csv")

# Initialize with correct columns if file doesn't exist
if not HISTORY_FILE.exists():
    pd.DataFrame(columns=["temperature", "battery_level", "gyro_drift"]).to_csv(
        HISTORY_FILE, index=False
    )

# ------------------------------------------------------------------
# 2. LOAD CURRENT DATA (uploaded or demo)
# ------------------------------------------------------------------
uploaded = st.file_uploader(
    "Drop CSV (temperature, battery_level, gyro_drift)", type="csv"
)

if uploaded:
    df = pd.read_csv(uploaded)
    required = ["temperature", "battery_level", "gyro_drift"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    df = df[required].copy()
    source = "your upload"
else:
    st.info("Demo mode – generating failing satellite…")
    minutes = [0, 10, 20, 30, 40, 50]
    rows = []
    for m in minutes:
        if m >= 40:
            temp = 35 + np.random.normal(0, 3)
            batt = 85 - 2 * (m - 40) + np.random.normal(0, 0.5)
            gyro = 0.15 + 0.03 * (m - 40) + np.random.normal(0, 0.02)
        else:
            temp = 22 + 10 * np.sin(m / 60) + np.random.normal(0, 1.5)
            batt = 98 - 0.008 * m + np.random.normal(0, 0.3)
            gyro = 0.01 * m + np.random.normal(0, 0.05)
        rows.append({
            "temperature": round(temp, 2),
            "battery_level": round(batt, 2),
            "gyro_drift": round(gyro, 4),
        })
    df = pd.DataFrame(rows)
    source = "demo"

st.success(f"Loaded {len(df)} rows from {source}")

# ------------------------------------------------------------------
# 3. APPEND NEW DATA TO HISTORY (only if uploaded)
# ------------------------------------------------------------------
if uploaded:
    hist_df = pd.read_csv(HISTORY_FILE)
    # Avoid duplicates
    df_to_append = df[~df.isin(hist_df).all(axis=1)]
    if not df_to_append.empty:
        df_to_append.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        st.info(f"Added {len(df_to_append)} new rows to model memory.")
    else:
        st.info("No new data — using existing knowledge.")
else:
    st.info("Demo data not saved (upload a real file to train).")

# ------------------------------------------------------------------
# 4. LOAD ALL HISTORICAL DATA FOR TRAINING
# ------------------------------------------------------------------
historical = pd.read_csv(HISTORY_FILE)

# CRITICAL: Ensure we have at least 2 rows to train
if len(historical) < 2:
    st.warning("Not enough data to train yet. Using current session only.")
    X_train = df[["temperature", "battery_level", "gyro_drift"]].values
else:
    X_train = historical[["temperature", "battery_level", "gyro_drift"]].values

# ------------------------------------------------------------------
# 5. TRAIN ISOLATION FOREST (only if enough data)
# ------------------------------------------------------------------
if len(X_train) >= 2:
    model = IsolationForest(contamination=min(0.3, 1/len(X_train)), random_state=42)
    model.fit(X_train)
    st.success(f"Model trained on {len(X_train)} total points.")
else:
    st.warning("Not enough data — using simple threshold rules.")
    model = None

# ------------------------------------------------------------------
# 6. PREDICT ON CURRENT DATA
# ------------------------------------------------------------------
X_current = df[["temperature", "battery_level", "gyro_drift"]].values

if model is not None:
    scores = model.decision_function(X_current)
    pred = model.predict(X_current)
else:
    # Fallback: flag if gyro_drift > 0.8
    scores = np.where(X_current[:, 2] > 0.8, -1.0, 0.0)
    pred = np.where(X_current[:, 2] > 0.8, -1, 1)

# ------------------------------------------------------------------
# 7. RUL CALCULATION
# ------------------------------------------------------------------
drift_rate = df["gyro_drift"].diff().mean()
current_drift = df["gyro_drift"].iloc[-1]
THRESHOLD = 0.80

if drift_rate > 0 and current_drift < THRESHOLD:
    rul_days = max(0.1, (THRESHOLD - current_drift) / drift_rate) * 10
elif current_drift >= THRESHOLD:
    rul_days = 0.0
else:
    rul_days = 999.0

# ------------------------------------------------------------------
# 8. DASHBOARD
# ------------------------------------------------------------------
c1, c2 = st.columns([2, 1])
with c1:
    st.line_chart(df[["temperature", "battery_level", "gyro_drift"]], height=400)
with c2:
    st.metric("Anomaly Score", f"{scores[-1]:.3f}")
    st.metric("RUL", f"{rul_days:.1f} days")

if pred[-1] == -1:
    st.error("FAILURE IMMINENT")
    if rul_days == 0:
        st.warning("Reaction wheel **already failed**.")
    else:
        st.warning(f"Failure in **{rul_days:.1f} days**.")
else:
    st.success("Healthy")

# ------------------------------------------------------------------
# 9. EXPORT
# ------------------------------------------------------------------
report = df.copy()
report["anomaly_score"] = scores
report["prediction"] = pred
st.download_button("Export Report", report.to_csv(index=False), "satguard_report.csv")