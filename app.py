import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details below:")

# Inputs
tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)
contract = st.selectbox("Contract Type (0 = Month-to-month, 1 = One year, 2 = Two year)", [0, 1, 2])

if st.button("Predict Churn"):
    features = np.array([[tenure, monthly_charges, total_charges, contract]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer is likely to STAY ✅")

    st.markdown(f"### 🔢 Churn Probability: **{probability*100:.2f}%**")

    st.progress(float(probability))

st.markdown("---")
st.caption("Built by KADIYALA SURYATEJA | ANNAMACHARYA UNIVERSITY")
