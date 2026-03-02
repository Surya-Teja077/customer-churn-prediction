import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom Styling
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.metric-card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 📊 Customer Churn Analytics Dashboard")
st.markdown("Predict churn probability and assess customer risk profile.")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Customer Inputs")

tenure = st.sidebar.slider("Tenure (months)", 0, 120, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 2000.0, 75.0)
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

total_charges = tenure * monthly_charges

st.sidebar.markdown(f"**Estimated Total Charges:** {total_charges:.2f}")

st.sidebar.markdown("---")

if st.sidebar.button("Predict Churn"):

    features = np.array([
        [tenure, monthly_charges, total_charges, contract_map[contract]]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    # Risk classification
    if probability < 0.4:
        risk = "Low Risk 🟢"
    elif probability < 0.7:
        risk = "Medium Risk 🟡"
    else:
        risk = "High Risk 🔴"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability*100:.2f}%")

    with col2:
        st.metric("Risk Level", risk)

    with col3:
        result_text = "Likely to CHURN" if prediction[0] == 1 else "Likely to STAY"
        st.metric("Prediction", result_text)

    st.markdown("---")

    st.subheader("Probability Gauge")
    st.progress(float(probability))

st.markdown("---")
st.caption("Built by KADIYALA SURYATEJA | ANNAMACHARYA UNIVERSITY")
