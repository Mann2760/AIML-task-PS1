import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        model = joblib.load("churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please train the model first.")
        st.stop()

@st.cache_data
def preprocess_features(features_df, scaler, imputer):
    """Preprocess input features"""
    # Apply same transformations as in training
    feature_cols = [
        'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
        'Multiple Lines', 'Internet Service', 'Online Security',
        'Online Backup', 'Device Protection', 'Tech Support',
        'Streaming TV', 'Streaming Movies', 'Contract',
        'Paperless Billing', 'Payment Method',
        'Monthly Charges', 'Tenure Months Scaled', 'Total Charges Scaled'
    ]
    
    # Ensure correct column order and fill missing columns
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    
    X = features_df[feature_cols].values
    X = imputer.transform(X)
    X_scaled = scaler.transform(X)
    return X_scaled

def main():
    st.title("📱 Telco Customer Churn Predictor")
    st.markdown("---")
    
    # Load models
    model, scaler = load_models()
    
    # Sidebar for input
    st.sidebar.header("Customer Information")
    
    # Customer features input
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], 
                                        format_func=lambda x: "Yes" if x == 1 else "No")
    
    partner = st.sidebar.selectbox("Partner", [0, 1], 
                                 format_func=lambda x: "Yes" if x == 1 else "No")
    
    dependents = st.sidebar.selectbox("Dependents", [0, 1], 
                                    format_func=lambda x: "Yes" if x == 1 else "No")
    
    phone_service = st.sidebar.selectbox("Phone Service", [0, 1], 
                                       format_func=lambda x: "Yes" if x == 1 else "No")
    
    multiple_lines = st.sidebar.selectbox("Multiple Lines", [0, 1, 2], 
                                        format_func=lambda x: ["No", "Yes", "No phone service"][x])
    
    internet_service = st.sidebar.selectbox("Internet Service", [0, 1, 2], 
                                          format_func=lambda x: ["No", "DSL", "Fiber optic"][x])
    
    online_security = st.sidebar.selectbox("Online Security", [0, 1, 2], 
                                         format_func=lambda x: ["No", "Yes", "No internet service"][x])
    
    online_backup = st.sidebar.selectbox("Online Backup", [0, 1, 2], 
                                       format_func=lambda x: ["No", "Yes", "No internet service"][x])
    
    device_protection = st.sidebar.selectbox("Device Protection", [0, 1, 2], 
                                           format_func=lambda x: ["No", "Yes", "No internet service"][x])
    
    tech_support = st.sidebar.selectbox("Tech Support", [0, 1, 2], 
                                      format_func=lambda x: ["No", "Yes", "No internet service"][x])
    
    streaming_tv = st.sidebar.selectbox("Streaming TV", [0, 1, 2], 
                                      format_func=lambda x: ["No", "Yes", "No internet service"][x])
    
    streaming_movies = st.sidebar.selectbox("Streaming Movies", [0, 1, 2], 
                                          format_func=lambda x: ["No", "Yes", "No internet service"][x])
    
    contract = st.sidebar.selectbox("Contract", [1, 2, 3], 
                                  format_func=lambda x: ["Month-to-month", "One year", "Two year"][x-1])
    
    paperless_billing = st.sidebar.selectbox("Paperless Billing", [0, 1], 
                                           format_func=lambda x: "Yes" if x == 1 else "No")
    
    payment_method = st.sidebar.selectbox("Payment Method", [1, 2, 3, 4], 
                                        format_func=lambda x: ["Electronic check", "Mailed check", 
                                                             "Bank transfer", "Credit card"][x-1])
    
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.1)
    
    tenure_months = st.sidebar.slider("Tenure (Months)", 0, 72, 36)
    total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 10000.0, 2000.0, 10.0)
    
    # Calculate scaled features (using same MinMaxScaler logic)
    tenure_scaled = tenure_months / 72.0  # MinMaxScaler(0,72)
    total_scaled = total_charges / 10000.0  # Approximate scaling
    
    # Predict button
    if st.sidebar.button("🚀 Predict Churn", use_container_width=True):
        # Create feature dataframe
        feature_data = {
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
        
        features_df = pd.DataFrame([feature_data])
        
        # Dummy imputer and scaler for demo (replace with actual ones)
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        imputer = SimpleImputer(strategy="median")
        std_scaler = StandardScaler()
        
        # Fit on single sample (for demo)
        X_processed = preprocess_features(features_df, std_scaler, imputer)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        # Display results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Churn Prediction", "Will Churn" if prediction == 1 else "Will Stay",
                     delta=None)
        
        with col2:
            st.metric("Churn Probability", f"{probability:.1%}", delta=None)
        
        churn_risk = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        with col3:
            st.metric("Risk Level", churn_risk, delta=None)
        
        # Color-coded result
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ **Customer is likely to churn!**")
            st.info("💡 **Recommendations:**\n"
                   "- Offer retention discounts\n"
                   "- Improve customer service\n"
                   "- Review contract terms")
        else:
            st.success("✅ **Customer is likely to stay!**")
            st.info("🎯 **Recommendations:**\n"
                   "- Continue excellent service\n"
                   "- Offer loyalty rewards\n"
                   "- Upsell premium services")

if __name__ == "__main__":
    main()
