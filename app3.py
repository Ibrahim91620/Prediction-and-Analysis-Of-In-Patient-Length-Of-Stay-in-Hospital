import streamlit as st
import pickle
import numpy as np

# Load the trained Decision Tree model
model_path = "decision_tree_los1.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Title of the Streamlit app
st.title("Hospital Length of Stay Prediction")

# User input fields
rcount = st.selectbox("Readmission Count", ["0", "1", "2", "3", "4", "5+"])
gender = st.selectbox("Gender", ["Male", "Female"])
dialysisrenalendstage = st.selectbox("Dialysis Renal End Stage", [0, 1])
asthma = st.selectbox("Asthma", [0, 1])
irondef = st.selectbox("Iron Deficiency", [0, 1])
pneum = st.selectbox("Pneumonia", [0, 1])
psychologicaldisordermajor = st.selectbox("Psychological Disorder Major", [0, 1])
depress = st.selectbox("Depression", [0, 1])
malnutrition = st.selectbox("Malnutrition", [0, 1])
hemo = st.number_input("Hemoglobin", min_value=0.0, format="%.2f")
hematocrit = st.number_input("Hematocrit", min_value=0.0, format="%.2f")
sodium = st.number_input("Sodium", min_value=0.0, format="%.2f")
glucose = st.number_input("Glucose", min_value=0.0, format="%.2f")
creatinine = st.number_input("Creatinine", min_value=0.0, format="%.2f")
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
pulse = st.number_input("Pulse", min_value=0.0, format="%.2f")
respiration = st.number_input("Respiration Rate", min_value=0.0, format="%.2f")

# Convert categorical inputs to numerical values
rcount_mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5+": 5}
rcount = rcount_mapping[rcount]
gender = 1 if gender == "Male" else 0

# Make prediction if the user clicks the button
if st.button("Predict Length of Stay"):
    input_features = np.array([
        rcount, gender, dialysisrenalendstage, asthma, irondef, pneum,
        psychologicaldisordermajor, depress, malnutrition, hemo, hematocrit,
        sodium, glucose, creatinine, bmi, pulse, respiration
    ]).reshape(1, -1)
    
    

    
    prediction = model.predict(input_features)[0]
    st.success(f"Predicted Length of Stay: in range form Type-{prediction}")
    st.subheader("Length of Stay Categories")
st.table({"Category": ["Short Stay Type-0", "Medium Stay Type-1", "Long Stay Type-2"], "Days": ["0-3", "4-7", "8+"]})