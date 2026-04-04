# 🛡️ Suspicious Login Behavior Detection System

An end-to-end Machine Learning pipeline and interactive cybersecurity dashboard built with PyTorch and Streamlit. 

## 🚀 Project Overview
This project simulates, detects, and explains suspicious login attempts (like brute force attacks and session hijacking) using a custom Feed-Forward Neural Network. It features an Explainable AI (XAI) engine that translates mathematical threat scores into human-readable security alerts.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (Multi-Layer Perceptron with Dropout)
* **Data Processing:** Scikit-Learn (StandardScaler), Pandas, NumPy
* **Frontend UI:** Streamlit, Plotly (Interactive Gauges)

## 🧠 Features
* **Custom Synthetic Dataset:** Engineered features including geographic distance, time elapsed, device footprint, and authentication failures.
* **Explainable AI:** Does not just block users—it provides the exact reasoning (e.g., "Impossible Travel Detected").
* **Live Telemetry Dashboard:** A real-time UI with visual threat gauges and session history logging.

## 💻 How to Run Locally
1. Clone this repository.
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`
