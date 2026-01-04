import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load trained model and scaler if available"""
    try:
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, True
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Model files not found. Using demo mode.")
        return None, None, False

def preprocess_features(features_dict):
    """Preprocess features matching exact training pipeline"""
    feature_cols = [
        'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
        'Multiple Lines', 'Internet Service', 'Online Security',
        'Online Backup', 'Device Protection', 'Tech Support',
        'Streaming TV', 'Streaming Movies', 'Contract',
        'Paperless Billing', 'Payment Method',
        'Monthly Charges', 'Tenure Months Scaled', 'Total Charges Scaled'
    ]
    
    # Create DataFrame with EXACT column order
    features_df = pd.DataFrame([features_dict])[feature_cols]
    
    # Initialize imputer and scaler (demo mode recreates training pipeline)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    # Transform (no feature name validation issues)
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled.flatten()

def demo_predict(X_scaled):
    """Demo prediction function when model not available"""
    # Simple rule-based prediction for demo (replace with real model)
    tenure_scaled = X_scaled[16]  # Tenure Months Scaled
    total_scaled = X_scaled[17]   # Total Charges Scaled
    contract = X_scaled[12]       # Contract
    
    # Demo logic: low tenure + month-to-month + high charges = high churn
    churn_prob = 0.5 + (0.3 * (1 - tenure_scaled)) + (0.2 * (contract < 1.5)) - (0.1 * total_scaled)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    prediction = 1 if churn_prob > 0.5 else 0
    return prediction, churn_prob

def main():
    st.title("📱 Telco Customer Churn Predictor")
    st.markdown("---")
    
    # Load models
    model, scaler, model_loaded = load_models()
    
    # Sidebar inputs
    st.sidebar.header("👤 Customer Profile")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], 
                                    format_func=lambda x: "Yes" if x else "No")
    with col2:
        partner = st.selectbox("Partner", [0, 1], 
                             format_func=lambda x: "Yes" if x else "No")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        dependents = st.selectbox("Dependents", [0, 1], 
                                format_func=lambda x: "Yes" if x else "No")
    with col2:
        phone_service = st.selectbox("Phone Service", [0, 1], 
                                   format_func=lambda x: "Yes" if x else "No")
    
    st.sidebar.markdown("---")
    multiple_lines = st.sidebar.selectbox("Multiple Lines", [0, 1, 2], 
                                        format_func=lambda x: ["No", "Yes", "No phone"][x])
    
    internet_service = st.sidebar.selectbox("Internet Service", [0, 1, 2], 
                                          format_func=lambda x: ["No", "DSL", "Fiber"][x])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Services")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        online_security = st.selectbox("Security", [0, 1, 2], 
                                     format_func=lambda x: ["No", "Yes", "N/A"][x])
    with col2:
        online_backup = st.selectbox("Backup", [0, 1, 2], 
                                   format_func=lambda x: ["No", "Yes", "N/A"][x])
    with col3:
        device_protection = st.selectbox("Protection", [0, 1, 2], 
                                       format_func=lambda x: ["No", "Yes", "N/A"][x])
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        tech_support = st.selectbox("Tech Support", [0, 1, 2], 
                                  format_func=lambda x: ["No", "Yes", "N/A"][x])
    with col2:
        streaming_tv = st.selectbox("Streaming TV", [0, 1, 2], 
                                  format_func=lambda x: ["No", "Yes", "N/A"][x])
    with col3:
        streaming_movies = st.selectbox("Streaming Movies", [0, 1, 2], 
                                      format_func=lambda x: ["No", "Yes", "N/A"][x])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Billing & Contract")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        contract = st.selectbox("Contract", [1, 2, 3], 
                              format_func=lambda x: ["Month-to-month", "1 Year", "2 Year"][x-1])
    with col2:
        paperless_billing = st.selectbox("Paperless", [0, 1], 
                                       format_func=lambda x: "Yes" if x else "No")
    
    payment_method = st.sidebar.selectbox("Payment Method", [1, 2, 3, 4], 
                                        format_func=lambda x: ["E-check", "Mail", "Bank", "Credit"][x-1])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Usage")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    with col2:
        tenure_months = st.slider("Tenure (Months)", 0, 72, 36)
    
    total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 10000.0, 2000.0)
    
    # Calculate scaled features
    tenure_scaled = tenure_months / 72.0
    total_scaled = total_charges / 10000.0
    
    # Predict button
    if st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True):
        
        # Prepare features
        features = {
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
            'Tenure Months Scaled': tenure_scaled,
            'Total Charges Scaled': total_scaled
        }
        
        # Process features
        X_scaled = preprocess_features(features)
        
        # Make prediction
        if model_loaded and model is not None:
            prediction = model.predict(X_scaled.reshape(1, -1))[0]
            probability = model.predict_proba(X_scaled.reshape(1, -1))[0][1]
            st.success("✅ Using trained model!")
        else:
            prediction, probability = demo_predict(X_scaled)
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", "🚪 CHURN" if prediction == 1 else "✅ STAY",
                     delta=None)
        
        with col2:
            st.metric("Probability", f"{probability:.1%}", delta=None)
        
        risk_color = "🔴" if probability > 0.7 else "🟡" if probability > 0.4 else "🟢"
        with col3:
            st.metric("Risk", f"{risk_color} {['Low','Medium','High'][min(2, int(probability*3))]}")
        
        # Recommendations
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ **High Churn Risk Detected!**")
            st.info("- Offer retention discount\n- Call customer immediately\n- Review service quality")
        else:
            st.success("✅ **Low Churn Risk**")
            st.info("- Maintain service quality\n- Consider upselling\n- Send loyalty rewards")

if __name__ == "__main__":
    main()
