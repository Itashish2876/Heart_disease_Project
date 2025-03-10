import streamlit as st
import numpy as np
import pickle

# Custom CSS for better layout and colors
st.markdown(
    '''
    <style>
        .main {
            background-color: #f0f2f5;
            padding: 20px;
            border-radius: 12px;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
        }
    </style>
    ''', unsafe_allow_html=True
)


st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🫀 Heart Disease Prediction Model")

# Load Model and Scaler with Error Handling
try:
    model = pickle.load(open("Heart_disease_Model.pkl", "rb"))
    scaler = pickle.load(open("Scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check the files.")
    st.stop()

# Input fields (with basic validation ranges provided in sidebar)
st.sidebar.title("ℹ️ Input Guidelines")
st.sidebar.write("**Age:** 32 to 70 years")
st.sidebar.write("**Cholesterol:** 100 to 700 mg/dL")
st.sidebar.write("**Systolic Blood Pressure:** 83 to 300 mmHg")
st.sidebar.write("**Heart Rate:** 44 to 150 mmHg")
st.sidebar.write("**Glucose:** 40 to 400 mg/dL")
st.sidebar.write("**BMI:** 15 to 60")

col1, col2 = st.columns(2)

glucose = st.number_input("Glucose Level:", min_value=40.0, max_value=400.0, value=100.0, step=1.0)
    
with col1:
    age = st.number_input("Enter Age:", min_value=32, max_value=70, value=40, step=1)
    gender = st.selectbox("Select Gender:", ['Male', 'Female'])
    currentSmoker = st.selectbox("Is Current Smoker?", ['Yes', 'No'])
    BPMeds = st.selectbox("On Blood Pressure Medication?", ['Yes', 'No'])
    prevalentStroke = st.selectbox("Had a Stroke?", ['Yes', 'No'])
   

with col2:
    diabetes = st.selectbox("Has Diabetes?", ['Yes', 'No'])
    totChol = st.number_input("Total Cholesterol Level:", min_value=100.0, max_value=700.0, value=200.0, step=1.0)
    sysBP = st.number_input("Systolic Blood Pressure:", min_value=83.0, max_value=300.0, value=120.0, step=1.0)
    BMI = st.number_input("Body Mass Index (BMI):", min_value=15.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate:", min_value=44, max_value=150, value=70, step=1)
    
# Helper function to encode and predict
def predict_heart_disease(gender, age, currentSmoker,  BPMeds, 
                          prevalentStroke, diabetes, totChol, 
                          sysBP,  BMI, heartRate, glucose):
    
    # Fixed encoding for categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    currentSmoker_encoded = 1 if currentSmoker == 'Yes' else 0
    BPMeds_encoded = 1 if BPMeds == 'Yes' else 0
    prevalentStroke_encoded = 1 if prevalentStroke == 'Yes' else 0
    diabetes_encoded = 1 if diabetes == 'Yes' else 0

   # Prepare feature array (ensure correct order)
    features = np.array([[
        gender_encoded, age, currentSmoker_encoded, 
        BPMeds_encoded, prevalentStroke_encoded, 
        diabetes_encoded, totChol, sysBP, BMI, heartRate, glucose
    ]])

   # Scale the features using pre-fitted scaler
    scaled_features = scaler.transform(features)

   # Predict using the model
    prediction = model.predict(scaled_features)
    return prediction[0]

# Predict button with results block
if st.button('🔎 Predict'):
    result = predict_heart_disease(gender, age, currentSmoker, BPMeds, 
                                   prevalentStroke, diabetes, totChol, 
                                   sysBP, BMI, heartRate, glucose)

    st.markdown("---")    # <- This adds a nice separator line
    if result == 1:
        st.error("⚠️ **Prediction Result:** Person is at risk of developing heart disease.")
    else:
        st.success("✅ **Prediction Result:** Person is not at risk of developing heart disease.")
    
    st.markdown("Prediction is based on risk factors like age, cholesterol, blood pressure, and smoking status.")

st.markdown("</div>", unsafe_allow_html=True)
