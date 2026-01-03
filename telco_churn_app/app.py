import streamlit as st
import pandas as pd
import joblib
voting_clf = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("AI‑Powered Telco Churn Prediction App")
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["Yes", "No"])
backup = st.selectbox("Online Backup", ["Yes", "No"])
device = st.selectbox("Device Protection", ["Yes", "No"])
tech = st.selectbox("Tech Support", ["Yes", "No"])
tv = st.selectbox("Streaming TV", ["Yes", "No"])
movies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly = st.number_input("Monthly Charges", 0.0)
tenure = st.number_input("Tenure (Months)", 0)
total = st.number_input("Total Charges", 0.0)
if st.button("Predict Churn"):

    df = pd.DataFrame([{
        'Senior Citizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'Phone Service': phone,
        'Multiple Lines': multiple,
        'Internet Service': internet,
        'Online Security': security,
        'Online Backup': backup,
        'Device Protection': device,
        'Tech Support': tech,
        'Streaming TV': tv,
        'Streaming Movies': movies,
        'Contract': contract,
        'Paperless Billing': paperless,
        'Payment Method': payment,
        'Monthly Charges': monthly,
        'Tenure Months Scaled': tenure,
        'Total Charges Scaled': total
    }])
    # Encode categorical columns
    
    # Scale numeric columns
    df[['Tenure Months Scaled', 'Total Charges Scaled']] = scaler.transform(
        df[['Tenure Months Scaled', 'Total Charges Scaled']]
    )
    prob = voting_clf.predict_proba(df)[0][1]

    st.success(f"Churn Probability: {round(prob*100, 2)}%")

    if prob > 0.7:
        st.error("🔴 High Risk – Offer discount & priority support")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk – Retention campaign")
    else:
        st.success("🟢 Low Risk – Upsell opportunity")
