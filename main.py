# --------------------------------------------------------------
# SatGuard AI v4 – ML-powered satellite failure predictor
# --------------------------------------------------------------
#  • Upload a CSV → instant anomaly score + Remaining Useful Life
#  • All uploads are saved → model learns from every satellite
#  • Demo mode generates a failing satellite for quick testing
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# 1. CONFIGURATION & PERSISTENT STORAGE
# ------------------------------------------------------------------
st.set_page_config(page_title="SatGuard AI", layout="wide")

# Where we keep every telemetry row ever uploaded
HISTORY_FILE = Path("historical_data.csv")
# Create an empty file the first time the app starts
if not HISTORY_FILE.exists():
    pd.DataFrame(columns=["temperature", "battery_level", "gyro_drift"]).to_csv(
        HISTORY_FILE, index=False
    )

# ------------------------------------------------------------------
# 2. UI – FILE UPLOADER
# ------------------------------------------------------------------
st.title("SatGuard AI")
st.caption(
    "Upload telemetry (temperature, battery_level, gyro_drift) → get a failure forecast."
)

uploaded = st.file_uploader(
    "Drop a CSV file", type="csv", help="Columns: temperature, battery_level, gyro_drift"
)

# ------------------------------------------------------------------
# 3. LOAD DATA (uploaded OR demo)
# ------------------------------------------------------------------
if uploaded:
    # ---- 3a. User-supplied data -------------------------------------------------
    df = pd.read_csv(uploaded)

    # Basic validation – keep only the three required columns
    required = ["temperature", "battery_level", "gyro_drift"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    df = df[required].copy()
    source = "your upload"
else:
    # ---- 3b. Demo data (6 points, failure in the last 2) -----------------------
    st.info("Demo mode – generating a failing satellite…")
    minutes = [0, 10, 20, 30, 40, 50]
    rows = []
    for minute in minutes:
        if minute >= 40:                                     # <-- failure zone
            temp = 35 + np.random.normal(0, 3)
            batt = 85 - 2 * (minute - 40) + np.random.normal(0, 0.5)
            gyro = 0.15 + 0.03 * (minute - 40) + np.random.normal(0, 0.02)
        else:                                                # <-- normal zone
            temp = 22 + 10 * np.sin(minute / 60) + np.random.normal(0, 1.5)
            batt = 98 - 0.008 * minute + np.random.normal(0, 0.3)
            gyro = 0.01 * minute + np.random.normal(0, 0.05)

        rows.append(
            {
                "temperature": round(temp, 2),
                "battery_level": round(batt, 2),
                "gyro_drift": round(gyro, 4),
            }
        )
    df = pd.DataFrame(rows)
    source = "demo"

st.success(f"Loaded {len(df)} rows from {source}")

# ------------------------------------------------------------------
# 4. PERSIST NEW DATA (so the model keeps learning)
# ------------------------------------------------------------------
if uploaded:
    # Append only the new rows (avoid duplicates if the same file is re-uploaded)
    existing = pd.read_csv(HISTORY_FILE)
    df_to_append = df[~df.isin(existing).all(axis=1)]
    if not df_to_append.empty:
        df_to_append.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        st.info(f"Appended {len(df_to_append)} new rows to the knowledge base.")

# Load *all* historic data for training
historical = pd.read_csv(HISTORY_FILE)
X_all = historical[["temperature", "battery_level", "gyro_drift"]].values

# ------------------------------------------------------------------
# 5. TRAIN ISOLATION FOREST ON *EVERYTHING* WE HAVE
# ------------------------------------------------------------------
# contamination = expected fraction of anomalies (tune later)
model = IsolationForest(contamination=0.3, random_state=42, n_jobs=-1)
model.fit(X_all)

# ------------------------------------------------------------------
# 6. PREDICT ON THE *CURRENT* dataset (uploaded or demo)
# ------------------------------------------------------------------
X_current = df[["temperature", "battery_level", "gyro_drift"]].values
anomaly_scores = model.decision_function(X_current)   # lower = more anomalous
predictions = model.predict(X_current)                # -1 = anomaly, 1 = normal

# ------------------------------------------------------------------
# 7. REMAINING USEFUL LIFE (RUL) – simple drift-based estimate
# ------------------------------------------------------------------
drift_rate = df["gyro_drift"].diff().mean()          # average change per sample
current_drift = df["gyro_drift"].iloc[-1]

FAILURE_THRESHOLD = 0.80                              # gyro drift beyond this = dead

if drift_rate > 0 and current_drift < FAILURE_THRESHOLD:
    rul_days = max(0.1, (FAILURE_THRESHOLD - current_drift) / drift_rate) * 10
elif current_drift >= FAILURE_THRESHOLD:
    rul_days = 0.0                                    # already past the line
else:
    rul_days = 999.0                                   # no clear trend

# ------------------------------------------------------------------
# 8. DASHBOARD
# ------------------------------------------------------------------
col_chart, col_metrics = st.columns([2, 1])

with col_chart:
    st.line_chart(
        df[["temperature", "battery_level", "gyro_drift"]], height=400
    )

with col_metrics:
    st.metric("Latest Anomaly Score", f"{anomaly_scores[-1]:.3f}")
    st.metric("RUL (days)", f"{rul_days:.1f}")

# Final alert
if predictions[-1] == -1:
    st.error("FAILURE IMMINENT")
    if rul_days <= 0:
        st.warning("Reaction wheel is **already beyond safe limits**.")
    else:
        st.warning(f"Reaction wheel dies in **{rul_days:.1f} days**.")
else:
    st.success("Healthy")

# ------------------------------------------------------------------
# 9. EXPORT CURRENT REPORT
# ------------------------------------------------------------------
csv_report = df.copy()
csv_report["anomaly_score"] = anomaly_scores
csv_report["prediction"] = predictions
st.download_button(
    "Export Full Report",
    data=csv_report.to_csv(index=False),
    file_name="satguard_report.csv",
    mime="text/csv",
)

# ------------------------------------------------------------------
# END OF FILE
# ------------------------------------------------------------------