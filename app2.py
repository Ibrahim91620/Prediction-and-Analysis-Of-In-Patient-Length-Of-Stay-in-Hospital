import streamlit as st
import pandas as pd
import joblib  # For loading the trained model
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained logistic regression model and scaler
model = joblib.load("length_of_stay_model.pkl")  # Ensure this file exists
scaler = joblib.load("scaler.pkl")  # Ensure the scaler used during training is available

st.title("Hospital Length of Stay Prediction")


st.sidebar.header("Enter Patient Details")

rcount = st.sidebar.number_input("Readmission Count", min_value=0, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

asthma = st.sidebar.selectbox("Asthma", ["Yes", "No"])
irondef = st.sidebar.selectbox("Iron Deficiency", ["Yes", "No"])
pneum = st.sidebar.selectbox("Pneumonia", ["Yes", "No"])
psychologicaldisordermajor = st.sidebar.selectbox("Psychological Disorder Major", ["Yes", "No"])
depress = st.sidebar.selectbox("Depression", ["Yes", "No"])
malnutrition = st.sidebar.selectbox("Malnutrition", ["Yes", "No"])
hemo = st.sidebar.number_input("Hemoglobin", min_value=0.0, step=0.1)

sodium = st.sidebar.number_input("Sodium", min_value=0.0, step=0.1)
glucose = st.sidebar.number_input("Glucose", min_value=0.0, step=0.1)
creatinine = st.sidebar.number_input("Creatinine", min_value=0.0, step=0.1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, step=0.1)
pulse = st.sidebar.number_input("Pulse", min_value=0, step=1)
respiration = st.sidebar.number_input("Respiration Rate", min_value=0, step=1)

# Convert categorical inputs to numerical values
gender = 1 if gender == "Male" else 0

asthma = 1 if asthma == "Yes" else 0
irondef = 1 if irondef == "Yes" else 0
pneum = 1 if pneum == "Yes" else 0
psychologicaldisordermajor = 1 if psychologicaldisordermajor == "Yes" else 0
depress = 1 if depress == "Yes" else 0
malnutrition = 1 if malnutrition == "Yes" else 0

# Create a DataFrame for model input
input_data = pd.DataFrame({
    "rcount": [rcount],
    "gender": [gender],

    "asthma": [asthma],
    "irondef": [irondef],
    "pneum": [pneum],
    "psychologicaldisordermajor": [psychologicaldisordermajor],
    "depress": [depress],
    "malnutrition": [malnutrition],
    "hemo": [hemo],
    
    "sodium": [sodium],
    "glucose": [glucose],
    "creatinine": [creatinine],
    "bmi": [bmi],
    "pulse": [pulse],
    "respiration": [respiration]
})

# Apply feature scaling
scaled_input = scaler.transform(input_data)

# Prediction Button
if st.sidebar.button("Predict Length of Stay"):
    prediction = model.predict(scaled_input)
    st.write(f"Predicted Length of Stay: {prediction[0]} days")
