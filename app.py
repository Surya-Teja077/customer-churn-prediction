import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom Dark Theme Styling
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.big-number {
    font-size: 80px;
    font-weight: bold;
    text-align: center;
}
.sub-text {
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Churn Intelligence Dashboard")
st.markdown("AI-powered customer churn risk prediction system")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Customer Profile")

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

if st.sidebar.button("Run Prediction"):

    features = np.array([
        [tenure, monthly_charges, total_charges, contract_map[contract]]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]

    # Risk classification
    if probability < 0.4:
        risk = "Low Risk"
        color = "#00FFAA"
    elif probability < 0.7:
        risk = "Medium Risk"
        color = "#FFD700"
    else:
        risk = "High Risk"
        color = "#FF4B4B"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prediction",
            "Likely to CHURN" if prediction[0] == 1 else "Likely to STAY"
        )
        st.metric("Risk Level", risk)

    with col2:
        st.markdown(
            f"""
            <div class="big-number" style="color:{color};">
                {probability*100:.2f}%
            </div>
            <div class="sub-text">
                Churn Probability
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Feature Importance Section
    st.subheader("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features_names = ["Tenure", "Monthly Charges", "Total Charges", "Contract"]

        df_importance = pd.DataFrame({
            "Feature": features_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(df_importance["Feature"], df_importance["Importance"])
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

st.markdown("---")
st.caption("Built by KADIYALA SURYATEJA | ANNAMACHARYA UNIVERSITY")
