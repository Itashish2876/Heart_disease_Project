import streamlit as st
import numpy as np
import pickle

# Load Model and Scaler (make sure these files exist in your working directory)
Model = pickle.load(open("Heart_disease_Model.pkl", "rb"))
scaler = pickle.load(open("Scaler.pkl", "rb"))

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f5, #e1e5ea);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2E3B55;
    }
    label {
        font-weight: bold;
    }
    .stButton>button {
        color: white;
        background-color: #007BFF;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title(" Heart Disease Prediction Model")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Choose Gender (1=Male, 0=Female):", ['Male', 'Female'])
    age = st.number_input("Enter Age:", min_value=0)
    currentSmoker = st.selectbox("Is Current Smoker? (1=Yes, 0=No):", ['Yes', 'No'])
    cigsPerDay = st.number_input("Cigarettes per Day:", min_value=0.0)
    BPMeds = st.selectbox("On Blood Pressure Medication? (1=Yes, 0=No):", ['Yes', 'No'])
    prevalentStroke = st.selectbox("Had a Stroke? (1=Yes, 0=No):", ['Yes', 'No'])
    prevalentHyp = st.selectbox("Has Hypertension? (1=Yes, 0=No):", ['Yes', 'No'])

with col2:
    diabetes = st.selectbox("Has Diabetes? (1=Yes, 0=No):", ['Yes', 'No'])
    totChol = st.number_input("Total Cholesterol Level:", min_value=0.0)
    sysBP = st.number_input("Systolic Blood Pressure:", min_value=0.0)
    diaBP = st.number_input("Diastolic Blood Pressure:", min_value=0.0)
    BMI = st.number_input("Body Mass Index (BMI):", min_value=0.0)
    heartRate = st.number_input("Heart Rate:", min_value=0.0)
    glucose = st.number_input("Glucose Level:", min_value=0.0)

# Helper function to encode and predict
def predict_heart_disease(gender, age, currentSmoker, cigsPerDay, BPMeds, 
                          prevalentStroke, prevalentHyp, diabetes, totChol, 
                          sysBP, diaBP, BMI, heartRate, glucose):
    
    # Fixed encoding for categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    currentSmoker_encoded = 1 if currentSmoker == 'Yes' else 0
    BPMeds_encoded = 1 if BPMeds == 'Yes' else 0
    prevalentStroke_encoded = 1 if prevalentStroke == 'Yes' else 0
    prevalentHyp_encoded = 1 if prevalentHyp == 'Yes' else 0
    diabetes_encoded = 1 if diabetes == 'Yes' else 0

    # Prepare feature array (ensure correct order)
    features = np.array([[gender_encoded, age, currentSmoker_encoded, cigsPerDay, 
                          BPMeds_encoded, prevalentStroke_encoded, prevalentHyp_encoded, 
                          diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    
    # Scale the features using pre-fitted scaler
    scaled_features = scaler.transform(features)

    # Predict using the model
    prediction = Model.predict(scaled_features)

    return prediction[0]

# Prediction button
if st.button('Predict'):
    result = predict_heart_disease(gender, age, currentSmoker, cigsPerDay, BPMeds, 
                                   prevalentStroke, prevalentHyp, diabetes, totChol, 
                                   sysBP, diaBP, BMI, heartRate, glucose)
    
    if result == 1:
        st.success("⚠️ Person **has a heart disease**.")
    else:
        st.success("✅ Person **does not have a heart disease**.")

st.markdown("</div>", unsafe_allow_html=True)
