import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import plotly.graph_objects as go
from datetime import datetime


# --- 1. Rebuild the Model Architecture ---
class LoginBehaviorModel(nn.Module):
    def __init__(self):
        super(LoginBehaviorModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# --- 2. Load the Artifacts ---
@st.cache_resource
def load_components():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model = LoginBehaviorModel()
    model.load_state_dict(torch.load('login_model_weights.pth', weights_only=True))
    model.eval()
    return scaler, model


scaler, model = load_components()

# --- 3. Initialize Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []


# --- 4. Plotly Gauge Chart Function ---
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': "%", 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Threat Level", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#00cc96"},  # Green / Safe
                {'range': [30, 70], 'color': "#FFA15A"},  # Orange / Warning
                {'range': [70, 100], 'color': "#EF553B"}  # Red / Danger
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# --- 5. The Streamlit UI ---
st.set_page_config(page_title="CyberOps | AI Monitor", page_icon="🛡️", layout="wide")

# Custom CSS for a sleeker look
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ CyberOps: AI Login Threat Monitor")
st.markdown("Live telemetry analysis powered by PyTorch Deep Learning.")

st.divider()

# Layout: 3 Columns
col1, col2, col3 = st.columns([1, 1, 1.5])

with col1:
    st.subheader("📍 Geospatial & Time")
    distance = st.number_input("Distance from last login (km)", min_value=0.0, value=5.0, step=10.0)
    time_min = st.number_input("Time since last login (min)", min_value=0.0, value=720.0, step=10.0)

with col2:
    st.subheader("💻 Device & Authentication")
    failed_attempts = st.number_input("Failed attempts (24h)", min_value=0, value=0, step=1)
    device_status = st.selectbox("Device Recognition", options=["Known Device", "Unknown/New Device"])
    is_unknown_device = 1 if device_status == "Unknown/New Device" else 0

with col3:
    st.subheader("⚙️ System Command")
    analyze_btn = st.button("🚨 Analyze Telemetry", type="primary", use_container_width=True)

    # Placeholder for the gauge chart
    chart_placeholder = st.empty()

# --- Prediction Logic ---
if analyze_btn:
    # Inference
    raw_data = [[distance, time_min, failed_attempts, is_unknown_device]]
    scaled_data = scaler.transform(raw_data)
    tensor_data = torch.FloatTensor(scaled_data)

    with torch.no_grad():
        score = model(tensor_data).item()

    threat_percent = score * 100
    is_suspicious = score > 0.5

    # Update Chart
    chart_placeholder.plotly_chart(create_gauge(score), use_container_width=True)

    # Explainable AI Results
    st.subheader("🧠 Neural Network Diagnosis")
    if is_suspicious:
        st.error(f"🛑 VERDICT: LOGIN BLOCKED. High probability of account compromise.")
        if distance > 500 and time_min < 60:
            st.warning("⚠️ **Impossible Travel:** Geographic movement exceeds physical limits.")
        if failed_attempts >= 3:
            st.warning("⚠️ **Brute Force:** Multiple failed authentications detected.")
        if is_unknown_device == 1:
            st.warning("⚠️ **Hardware Anomaly:** Unrecognized device footprint.")
        status = "Blocked 🛑"
    else:
        st.success(f"✅ VERDICT: LOGIN APPROVED. Behavior matches baseline.")
        status = "Approved ✅"

    # Save to history
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"**[{timestamp}]** Dist: {distance}km | Time: {time_min}m | Fails: {failed_attempts} | Status: {status} ({threat_percent:.1f}%)"

    # Add to beginning of list so newest is on top
    st.session_state.history.insert(0, log_entry)

    # Keep only the last 5 logs
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()

# --- History Log ---
if st.session_state.history:
    st.divider()
    st.subheader("📜 Recent Telemetry Logs")
    for log in st.session_state.history:
        st.markdown(log)