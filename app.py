import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Title
st.title("🏥 Heart Disease Prediction App")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("C:\\Users\\prabh\\OneDrive\\Documents\\logistic regrection\\logistic_regression_model.pkl")
    return model

# Load the dataset to get feature names
@st.cache_resource
def load_data():
    df = pd.read_csv("C:\\Users\\prabh\\AppData\\Local\\Packages\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\\LocalState\\sessions\\D801DBC6E7E33F867C05ADEE00200191C6592666\\transfers\\2026-04\\heart.csv.xls")
    return df

try:
    model = load_model()
    df = load_data()
    
    # Get feature names (excluding target and age_group)
    feature_names = [col for col in df.columns if col not in ['target', 'age_group']]
    
    st.sidebar.header("📋 Patient Information")
    
    # Create input fields for each feature
    user_input = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographic & Health Metrics")
        for i, feature in enumerate(feature_names[:len(feature_names)//2]):
            if feature in df.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                user_input[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=0.1
                )
    
    with col2:
        st.subheader("Additional Metrics")
        for i, feature in enumerate(feature_names[len(feature_names)//2:]):
            if feature in df.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                user_input[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=0.1
                )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Predict", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([user_input])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK")
                st.metric("Risk Level", "High")
            else:
                st.success("✅ LOW RISK")
                st.metric("Risk Level", "Low")
        
        with col2:
            st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")
        
        with col3:
            st.metric("Prediction Class", f"Class {prediction}")
        
        st.markdown("---")
        
        # Show probability breakdown
        st.subheader("📈 Probability Breakdown")
        prob_df = pd.DataFrame({
            'Class': ['No Disease (0)', 'Disease (1)'],
            'Probability': prediction_proba
        })
        st.bar_chart(prob_df.set_index('Class'))
        
        # Show input summary
        st.subheader("📝 Patient Input Summary")
        st.dataframe(input_data.T, use_container_width=True)

except FileNotFoundError as e:
    st.error(f"❌ Error: {e}")
    st.info(st.info(r"Make sure the model file is saved at: C:\Users\prabh\OneDrive\Documents\logistic regrection\logistic_regression_model.pkl")
)
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.info("Please check your inputs and try again.")