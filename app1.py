import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_model():
    model = joblib.load("length_of_stay_model.pkl")  # Ensure correct model filename
    scaler = joblib.load("scaler.pkl")  # Ensure correct scaler is loaded
    return model, scaler

def preprocess_input(data, scaler):
    # Define the expected feature order (exactly as in training)
    expected_features = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 
                         'pneum', 'psychologicaldisordermajor', 'depress', 'malnutrition', 
                         'hemo', 'hematocrit', 'sodium', 'glucose', 'creatinine', 
                         'bmi', 'pulse', 'respiration']

    # Define categorical mappings
    category_mappings = {
        'gender': {'Male': 1, 'Female': 0},
        'dialysisrenalendstage': {'Yes': 1, 'No': 0},
        'asthma': {'Yes': 1, 'No': 0},
        'irondef': {'Yes': 1, 'No': 0},
        'pneum': {'Yes': 1, 'No': 0},
        'psychologicaldisordermajor': {'Yes': 1, 'No': 0},
        'depress': {'Yes': 1, 'No': 0},
        'malnutrition': {'Yes': 1, 'No': 0},
        'hemo': {'Yes': 1, 'No': 0},
    }

    # Convert categorical values to numeric
    for col, mapping in category_mappings.items():
        data[col] = data[col].map(mapping)

    # Ensure all expected features are present and in order
    data = data.reindex(columns=expected_features)

    # Apply scaling to numerical features
    numeric_cols = ['rcount', 'hematocrit', 'sodium', 'glucose', 'creatinine', 'bmi', 'pulse', 'respiration']
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    return data


def main():
    st.title("Hospital Length of Stay Prediction")
    st.write("Enter patient details to predict the length of stay.")

    # Input fields for selected features
    rcount = st.number_input("Readmission Count", min_value=0, max_value=10, value=0)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    dialysisrenalendstage = st.selectbox("Dialysis/Renal End Stage", ['Yes', 'No'])
    asthma = st.selectbox("Asthma", ['Yes', 'No'])
    irondef = st.selectbox("Iron Deficiency", ['Yes', 'No'])
    pneum = st.selectbox("Pneumonia", ['Yes', 'No'])
    psychologicaldisordermajor = st.selectbox("Psychological Disorder (Major)", ['Yes', 'No'])
    depress = st.selectbox("Depression", ['Yes', 'No'])
    malnutrition = st.selectbox("Malnutrition", ['Yes', 'No'])
    hemo = st.selectbox("Hemodialysis", ['Yes', 'No'])
    hematocrit = st.number_input("Hematocrit Level", min_value=0.0, max_value=100.0, value=40.0)
    sodium = st.number_input("Sodium Level", min_value=100.0, max_value=200.0, value=140.0)
    glucose = st.number_input("Glucose Level", min_value=50.0, max_value=500.0, value=100.0)
    creatinine = st.number_input("Creatinine Level", min_value=0.1, max_value=10.0, value=1.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    pulse = st.number_input("Pulse Rate", min_value=30, max_value=200, value=70)
    respiration = st.number_input("Respiration Rate", min_value=10, max_value=50, value=20)

    # Create dataframe from inputs
    input_data = pd.DataFrame([[rcount, gender, dialysisrenalendstage, asthma, irondef, pneum, 
                                psychologicaldisordermajor, depress, malnutrition, hemo, 
                                hematocrit, sodium, glucose, creatinine, bmi, pulse, respiration]], 
                               columns=['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 
                                        'pneum', 'psychologicaldisordermajor', 'depress', 'malnutrition', 
                                        'hemo', 'hematocrit', 'sodium', 'glucose', 'creatinine', 
                                        'bmi', 'pulse', 'respiration'])
    
    if st.button("Predict Length of Stay"):
        model, scaler = load_model()
        processed_data = preprocess_input(input_data, scaler)
        prediction = model.predict(processed_data)
        st.success(f"Predicted Length of Stay: {prediction[0]} days")

if __name__ == "__main__":
    main()
