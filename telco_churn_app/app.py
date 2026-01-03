import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model & scaler
# ===============================
model = joblib.load("telco_churn_app/churn_model.pkl")
scaler = joblib.load("telco_churn_app/scaler.pkl")


st.title("📊 Telco Customer Churn Prediction")

st.write("Enter customer details to predict churn")

# ===============================
# User Inputs
# ===============================
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
dependents = st.selectbox("Dependents", [0, 1])
phone_service = st.selectbox("Phone Service", [0, 1])
multiple_lines = st.selectbox("Multiple Lines", [0, 1])
internet_service = st.selectbox("Internet Service", [0, 1])
online_security = st.selectbox("Online Security", [0, 1])
online_backup = st.selectbox("Online Backup", [0, 1])
device_protection = st.selectbox("Device Protection", [0, 1])
tech_support = st.selectbox("Tech Support", [0, 1])
streaming_tv = st.selectbox("Streaming TV", [0, 1])
streaming_movies = st.selectbox("Streaming Movies", [0, 1])
contract = st.selectbox("Contract", [0, 1, 2])
paperless_billing = st.selectbox("Paperless Billing", [0, 1])
payment_method = st.selectbox("Payment Method", [0, 1, 2, 3])

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
tenure_months = st.number_input("Tenure (Months)", min_value=0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# ===============================
# Prediction
# ===============================
if st.button("Predict Churn"):

    # ---- RAW data for scaler (MUST MATCH TRAINING) ----
    raw_df = pd.DataFrame([{
        "tenure_months": tenure_months,
        "total_charges": total_charges
    }])

    # ---- Scale ----
    scaled_vals = scaler.transform(raw_df)

    # ---- Final input for model (EXACT feature names) ----
    final_input = pd.DataFrame([{
        'Senior Citizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'Phone Service': phone_service,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Online Security': online_security,
        'Online Backup': online_backup,
        'Device Protection': device_protection,
        'Tech Support': tech_support,
        'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_movies,
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Payment Method': payment_method,
        'Monthly Charges': monthly_charges,
        'Tenure Months Scaled': scaled_vals[0][0],
        'Total Charges Scaled': scaled_vals[0][1]
    }])

    # ---- Predict ----
    prediction = model.predict(final_input)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")
