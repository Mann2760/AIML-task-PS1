import streamlit as st
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("telco_churn_app/churn_model.pkl")
scaler = joblib.load("telco_churn_app/scaler.pkl")

st.title("Telco Churn Prediction App")

# ---- INPUTS (numeric / already encoded) ----
senior = st.number_input("Senior Citizen (0 or 1)", 0, 1)
partner = st.number_input("Partner (0 or 1)", 0, 1)
dependents = st.number_input("Dependents (0 or 1)", 0, 1)
phone = st.number_input("Phone Service (0 or 1)", 0, 1)
multiple = st.number_input("Multiple Lines (0 or 1)", 0, 1)
internet = st.number_input("Internet Service (encoded)", 0)
security = st.number_input("Online Security (0 or 1)", 0, 1)
backup = st.number_input("Online Backup (0 or 1)", 0, 1)
device = st.number_input("Device Protection (0 or 1)", 0, 1)
tech = st.number_input("Tech Support (0 or 1)", 0, 1)
tv = st.number_input("Streaming TV (0 or 1)", 0, 1)
movies = st.number_input("Streaming Movies (0 or 1)", 0, 1)
contract = st.number_input("Contract (encoded)", 0)
paperless = st.number_input("Paperless Billing (0 or 1)", 0, 1)
payment = st.number_input("Payment Method (encoded)", 0)

tenure = st.number_input("Tenure (Months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):
    df = pd.DataFrame([[ 
        senior, partner, dependents, phone, multiple,
        internet, security, backup, device, tech,
        tv, movies, contract, paperless, payment,
        monthly, 'Tenure Months Scaled': tenure,
    'Total Charges Scaled': total
    ]], columns=[
        'Senior Citizen','Partner','Dependents','Phone Service',
        'Multiple Lines','Internet Service','Online Security',
        'Online Backup','Device Protection','Tech Support',
        'Streaming TV','Streaming Movies','Contract',
        'Paperless Billing','Payment Method',
        'Monthly Charges','Tenure',
        'Total'
    ])

    # Scale only what you scaled in training
    df[['tenure', 'TotalCharges']] = scaler.transform(
    df[['tenure', 'TotalCharges']]
)



    prob = model.predict_proba(df)[0][1]
    st.success(f"Churn Probability: {round(prob*100,2)}%")
