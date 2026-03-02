import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Custom CSS for clean look
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
        }
        h1 {
            text-align: center;
        }
        .result-box {
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            text-align: center;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to estimate churn probability.")

st.markdown("---")

# Inputs
tenure = st.slider("Tenure (months)", 0, 120, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 2000.0, 75.0)
contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# Auto calculate total charges
total_charges = tenure * monthly_charges
st.caption(f"Estimated Total Charges: {total_charges:.2f}")

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

st.markdown("---")

if st.button("Predict Churn"):

    features = np.array([
        [tenure, monthly_charges, total_charges, contract_map[contract]]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    # Risk level
    if probability < 0.4:
        risk_color = "#4CAF50"
        risk_text = "Low Risk"
    elif probability < 0.7:
        risk_color = "#FF9800"
        risk_text = "Medium Risk"
    else:
        risk_color = "#F44336"
        risk_text = "High Risk"

    st.markdown(
        f"""
        <div class="result-box" style="background-color:{risk_color}20; border:1px solid {risk_color};">
            <b>Churn Probability:</b> {probability*100:.2f}% <br>
            <b>Risk Level:</b> {risk_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(probability))

st.markdown("---")
st.caption("Built by KADIYALA SURYATEJA | ANNAMACHARYA UNIVERSITY")
