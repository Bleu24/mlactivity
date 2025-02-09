import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('titanic_model.h5')

# App title
st.title("Titanic Survival Prediction")
st.write("Predict the survival of a Titanic passenger based on their details.")

# Input form for passenger details
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=0)
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
siblings_spouses = st.sidebar.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parents_children = st.sidebar.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, value=7.25)

# Preprocess inputs
sex = 0 if sex == "Male" else 1  # Encode 'Sex'
features = np.array([[pclass, sex, age, siblings_spouses, parents_children, fare]])

# Prediction button
if st.sidebar.button("Predict Survival"):
    prediction = model.predict(features)
    survival_probability = prediction[0][0]
    st.write(f"Survival Probability: {survival_probability:.2f}")

    if survival_probability >= 0.5:
        st.success("The passenger is likely to survive. 🌟")
    else:
        st.error("The passenger is unlikely to survive. ⚠️")
