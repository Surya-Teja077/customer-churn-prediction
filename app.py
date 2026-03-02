import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)
contract = st.selectbox("Contract Type", [0, 1, 2])

if st.button("Predict Churn"):
    features = np.array([[tenure, monthly_charges, total_charges, contract]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer is likely to STAY ✅")