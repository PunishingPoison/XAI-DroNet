import streamlit as st
import pandas as pd
import json
import os
import time

st.set_page_config(page_title="XAI-DroNet Dashboard", layout="wide")

st.title("🚁 XAI-DroNet Live Dashboard")

# Paths
OUTPUT_DIR = "outputs/airsim"
NAV_LOG_FILE = os.path.join(OUTPUT_DIR, "autonomous_navigation.jsonl")
EVAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "evaluation_report.json")
LATEST_FRAME = os.path.join(OUTPUT_DIR, "latest_frame.jpg")
LATEST_GRADCAM = os.path.join(OUTPUT_DIR, "latest_gradcam.jpg")

# Layout
st.subheader("Live Camera Feeds")
cam_col1, cam_col2 = st.columns(2)
with cam_col1:
    frame_placeholder = st.empty()
with cam_col2:
    gradcam_placeholder = st.empty()

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Telemetry History")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Evaluation Metrics")
    metrics_placeholder = st.empty()

st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 0.5, 5.0, 1.0)

def load_telemetry():
    if not os.path.exists(NAV_LOG_FILE):
        return pd.DataFrame()
    data = []
    try:
        with open(NAV_LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception:
        pass
    return pd.DataFrame(data)

def load_metrics():
    if not os.path.exists(EVAL_REPORT_FILE):
        return {}
    try:
        with open(EVAL_REPORT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def update_dashboard():
    # Update Images
    if os.path.exists(LATEST_FRAME):
        try:
            frame_placeholder.image(LATEST_FRAME, caption="Raw Camera Feed", use_container_width=True, channels="BGR")
        except Exception:
            pass
    else:
        frame_placeholder.info("Waiting for camera feed...")

    if os.path.exists(LATEST_GRADCAM):
        try:
            gradcam_placeholder.image(LATEST_GRADCAM, caption="Grad-CAM (Collision)", use_container_width=True, channels="BGR")
        except Exception:
            pass
    else:
        gradcam_placeholder.info("Waiting for Grad-CAM...")

    df = load_telemetry()
    if not df.empty:
        df = df.tail(100) # Only show last 100 steps
        
        # Plot Steering and Collision Probability
        chart_data = pd.DataFrame({
            'Steering': df['steering'].values,
            'Collision Probability': df['collision_probability'].values
        }, index=df['step_index'].values)
        
        with chart_placeholder.container():
            st.line_chart(chart_data)
    else:
        chart_placeholder.info("No telemetry data found yet.")
        
    metrics = load_metrics()
    with metrics_placeholder.container():
        if metrics:
            st.metric("Total Steps", metrics.get("total_steps", 0))
            st.metric("Collisions", metrics.get("collisions", 0))
            st.metric("Total Distance", f"{metrics.get('total_distance', 0.0):.2f} m")
            st.metric("Steering Variance", f"{metrics.get('steering_variance', 0.0):.4f}")
        else:
            st.info("No evaluation report available yet.")

if auto_refresh:
    while True:
        update_dashboard()
        time.sleep(refresh_rate)
        st.rerun()
else:
    update_dashboard()
