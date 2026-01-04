import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model & scaler
# ===============================
@st.cache_resource
def load_models():
    try:
        model = joblib.load("telco_churn_app/churn_model.pkl")
        scaler = joblib.load("telco_churn_app/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Place `churn_model.pkl` and `scaler.pkl` in `telco_churn_app/` folder.")
        st.stop()

model, scaler = load_models()

st.title("📊 Telco Customer Churn Prediction")
st.markdown("### Enter customer details below 👇")

# ===============================
# User Inputs (All in columns for better layout)
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", [0, 1])
    dependents = st.selectbox("Dependents", [0, 1])
    phone_service = st.selectbox("Phone Service", [0, 1])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", [0, 1, 2])  # Fixed: added 2
    internet_service = st.selectbox("Internet Service", [0, 1, 2])  # Fixed: added 2
    online_security = st.selectbox("Online Security", [0, 1, 2])  # Fixed: added 2
    online_backup = st.selectbox("Online Backup", [0, 1, 2])  # Fixed: added 2

with col3:
    device_protection = st.selectbox("Device Protection", [0, 1, 2])  # Fixed: added 2
    tech_support = st.selectbox("Tech Support", [0, 1, 2])  # Fixed: added 2
    streaming_tv = st.selectbox("Streaming TV", [0, 1, 2])  # Fixed: added 2
    streaming_movies = st.selectbox("Streaming Movies", [0, 1, 2])  # Fixed: added 2

col4, col5, col6 = st.columns(3)
with col4:
    contract = st.selectbox("Contract", [1, 2, 3])  # Fixed: 1,2,3
    paperless_billing = st.selectbox("Paperless Billing", [0, 1])
    payment_method = st.selectbox("Payment Method", [1, 2, 3, 4])  # Fixed: 1-4

with col5:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    tenure_months = st.number_input("Tenure (Months)", min_value=0, value=36)

with col6:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

# ===============================
# BIG PREDICT BUTTON
# ===============================
if st.button("🔮 **PREDICT CHURN**", type="primary", use_container_width=True):
    
    # ---- FIX: Correct column names for MinMaxScaler ----
    raw_df = pd.DataFrame([{
        "Tenure Months": tenure_months,      # ✅ Fixed: was "tenure_months"
        "Total Charges": total_charges       # ✅ Fixed: was "total_charges"
    }])

    # ---- Scale ----
    scaled_vals = scaler.transform(raw_df)

    # ---- Final input for model ----
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
    probability = model.predict_proba(final_input)[0][1] * 100

    # ===============================
    # DIRECT RESULT - NO DELAY
    # ===============================
    st.markdown("---")
    st.markdown("### 🎯 **PREDICTION RESULT**")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    if prediction == 1:
        with col1:
            st.markdown("### 🚨 **CHURN**")
        with col2:
            st.error(f"**Customer WILL CHURN**")
            st.warning(f"Churn Probability: **{probability:.1f}%**")
            st.info("💡 **Action:** Offer retention discount immediately!")
    else:
        with col1:
            st.markdown("### ✅ **STAY**")
        with col2:
            st.success(f"**Customer WILL STAY**")
            st.info(f"Retention Probability: **{100-probability:.1f}%**")
            st.info("🎉 **Action:** Continue excellent service!")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.json(final_input.iloc[0].to_dict())
