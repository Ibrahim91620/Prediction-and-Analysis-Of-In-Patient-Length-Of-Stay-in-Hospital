import streamlit as st
import pickle
import numpy as np

# Load the trained AdaBoost model
model_path= "adaboost.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit App Title
st.title("AdaBoost Prediction App")
st.write("Enter the patient details to predict the outcome.")

# Input Fields
rcount = st.number_input("RCount", min_value=0, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
dialysisrenalendstage = st.selectbox("Dialysis Renal End Stage", [0, 1])
asthma = st.selectbox("Asthma", [0, 1])
irondef = st.selectbox("Iron Deficiency", [0, 1])
pneum = st.selectbox("Pneumonia", [0, 1])
psychologicaldisordermajor = st.selectbox("Psychological Disorder Major", [0, 1])
depress = st.selectbox("Depression", [0, 1])
malnutrition = st.selectbox("Malnutrition", [0, 1])
hemo = st.selectbox("Hemodialysis", [0, 1])
hematocrit = st.number_input("Hematocrit", min_value=0.0)
sodium = st.number_input("Sodium", min_value=0.0)
glucose = st.number_input("Glucose", min_value=0.0)
creatinine = st.number_input("Creatinine", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
pulse = st.number_input("Pulse", min_value=0)
respiration = st.number_input("Respiration", min_value=0)

# Convert categorical values
gender = 1 if gender == "Male" else 0

# Prediction Button
if st.button("Predict"):
    features = np.array([
        rcount, gender, dialysisrenalendstage, asthma, irondef, pneum,
        psychologicaldisordermajor, depress, malnutrition, hemo, hematocrit,
        sodium, glucose, creatinine, bmi, pulse, respiration
    ]).reshape(1, -1)
    
    prediction = model.predict(features)
    st.write(f"Predicted Outcome: {prediction[0]}")