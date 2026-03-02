import streamlit as st
import joblib
import numpy as np

st.title("Customer Churn Prediction")

st.write("Enter Customer Details")

tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)

if st.button("Predict"):
    st.success("Model prediction will appear here after model integration.")