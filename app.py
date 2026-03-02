import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict customer churn probability using Machine Learning</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, step=1)
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=2000.0,
        step=10.0
    )

with col2:
    auto_calc = st.checkbox("Auto-calculate Total Charges (tenure × monthly charges)")
    
    if auto_calc:
        total_charges = tenure * monthly_charges
        st.info(f"Total Charges: {total_charges:.2f}")
    else:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=50000.0,
            step=100.0
        )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

# Contract mapping
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

st.markdown("---")

if st.button("🚀 Predict Churn"):

    features = np.array([
        [tenure, monthly_charges, total_charges, contract_map[contract]]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("📈 Prediction Result")

    # Risk Level Logic
    if probability < 0.4:
        risk_level = "Low Risk 🟢"
    elif probability < 0.7:
        risk_level = "Medium Risk 🟡"
    else:
        risk_level = "High Risk 🔴"

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.markdown(f"### 🔢 Churn Probability: {probability*100:.2f}%")
    st.markdown(f"### 🚦 Risk Level: {risk_level}")

    st.progress(float(probability))

st.markdown("---")
st.markdown(
    "<center><b>Built by KADIYALA SURYATEJA</b> | ANNAMACHARYA UNIVERSITY</center>",
    unsafe_allow_html=True
)
