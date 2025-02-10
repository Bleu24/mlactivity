import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
@st.cache_resource
def load_model_and_scaler():
    model = load_model('heart_failure.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("Heart Failure Risk Prediction")
st.write("This app predicts the likelihood of heart failure based on clinical parameters.")

# Sidebar input fields
st.sidebar.header("Patient Data Input")
age = st.sidebar.slider('Age', 20, 100, value=50)
anaemia = st.sidebar.selectbox('Anaemia', ['No', 'Yes'])
creatinine_phosphokinase = st.sidebar.slider('CPK (mcg/L)', 20, 8000, value=200)
diabetes = st.sidebar.selectbox('Diabetes', ['No', 'Yes'])
ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 10, 80, value=50)
high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', ['No', 'Yes'])
platelets = st.sidebar.slider('Platelets (k/mL)', 100000, 500000, value=250000, step=1000)
serum_creatinine = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.5, 5.0, value=1.0, step=0.1)
serum_sodium = st.sidebar.slider('Serum Sodium (mEq/L)', 110, 150, value=135)
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
smoking = st.sidebar.selectbox('Smoking', ['No', 'Yes'])
time = st.sidebar.slider('Follow-up Period (days)', 4, 300, value=100)

# Convert categorical inputs to numerical
anaemia = 1 if anaemia == 'Yes' else 0
diabetes = 1 if diabetes == 'Yes' else 0
high_blood_pressure = 1 if high_blood_pressure == 'Yes' else 0
sex = 1 if sex == 'Male' else 0
smoking = 1 if smoking == 'Yes' else 0

# Prepare input data
input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                        sex, smoking, time]])
normalized_input = scaler.transform(input_data)

# Make prediction
prediction = model.predict(normalized_input)[0][0]
risk_category = "High Risk" if prediction > 0.5 else "Low Risk"

# Display prediction
st.subheader("Prediction")
st.write(f"Risk Score: **{prediction:.2f}**")
st.write(f"Risk Category: **{risk_category}**")

st.info("Adjust the patient data on the sidebar to see updated predictions.")
