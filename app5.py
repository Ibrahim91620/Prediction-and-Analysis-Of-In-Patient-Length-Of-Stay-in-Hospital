import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained AdaBoost model
model = joblib.load("adaboost.pkl")

# Streamlit app title
st.title("Hospital Length of Stay Prediction")

# Collect user input
gender = st.selectbox("Gender", ["Male", "Female"])
dialysis = st.selectbox("Dialysis Renal End Stage", ["No", "Yes"])
asthma = st.selectbox("Asthma", ["No", "Yes"])
irondef = st.selectbox("Iron Deficiency", ["No", "Yes"])
pneum = st.selectbox("Pneumonia", ["No", "Yes"])
psych_disorder = st.selectbox("Psychological Disorder Major", ["No", "Yes"])
depress = st.selectbox("Depression", ["No", "Yes"])
malnutrition = st.selectbox("Malnutrition", ["No", "Yes"])
hemo = st.selectbox("Hemodialysis", ["No", "Yes"])

hematocrit = st.number_input("Hematocrit", min_value=0.0, max_value=100.0, step=0.1)
sodium = st.number_input("Sodium", min_value=100.0, max_value=200.0, step=0.1)
glucose = st.number_input("Glucose", min_value=50.0, max_value=500.0, step=0.1)
creatinine = st.number_input("Creatinine", min_value=0.1, max_value=10.0, step=0.1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
pulse = st.number_input("Pulse Rate", min_value=30, max_value=200, step=1)
respiration = st.number_input("Respiration Rate", min_value=10, max_value=40, step=1)
rcount = st.number_input("Red Blood Cell Count", min_value=0, max_value=10, step=1)

# Convert categorical inputs to binary
binary_mapping = {"No": 0, "Yes": 1, "Male": 0, "Female": 1}
features = [
    rcount, binary_mapping[gender], binary_mapping[dialysis], binary_mapping[asthma],
    binary_mapping[irondef], binary_mapping[pneum], binary_mapping[psych_disorder],
    binary_mapping[depress], binary_mapping[malnutrition], binary_mapping[hemo],
    hematocrit, sodium, glucose, creatinine, bmi, pulse, respiration
]

# Define feature names
feature_names = [
    "rcount", "gender", "dialysisrenalendstage", "asthma", "irondef", "pneum", 
    "psychologicaldisordermajor", "depress", "malnutrition", "hemo",
    "hematocrit", "sodium", "glucose", "creatinine", "bmi", "pulse", "respiration"
]

# Predict on button click
if st.button("Predict Length of Stay"):
    input_data = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(input_data)
    st.success(f"Predicted Length of Stay: {prediction[0]} days")
