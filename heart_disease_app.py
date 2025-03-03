import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f5, #e1e5ea); /* Light gradient background */
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white panel */
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333; /* Dark gray headings */
        font-family: 'Arial', sans-serif;
    }
    label {
        font-weight: bold;
    }
    .stButton>button {
        color: #fff;
        background-color: #007BFF; /* Blue button */
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# Load Model And Data  --------------------------------------
Model = pickle.load(open("Heart_Disease_Model.pkl","rb"))
scaler = pickle.load(open("Heart_Disease_Scaler.pkl","rb"))



label_encoder = LabelEncoder()

# Lets Create  Main Streamlit Web App ----------------------------------------

# Main content area
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("HeartDisease Prediction Regression Model")
col1,col2 = st.columns(2)

with col1 :
    

    gender =  st.selectbox("Choose the Gender As 1 for Male 0 for Female :", options = ['Male' , ' Female'])
    age = st.number_input("Enter the Age:", min_value=0)    

    currentSmoker =st.selectbox("Choose Person is currentSmoker :" , options = ['Yes' , ' No'])
    cigsPerDay  = st.number_input("Enter the No. of cigarettes smoked per day ::", min_value=0.0)

    BPMeds  = st.selectbox("Choose Person is on Blood Pressure Medication :", options = ['Yes' , ' No'])

    prevalentStroke = st.selectbox("Choose Person had a prevalentStroke or not :" , options = ['Yes' , ' No'])
    prevalentHyp = st.selectbox("Choose Person has HyperTension or not :" , options = ['Yes' , ' No'])   
with col2 :
    
    diabetes  = st.selectbox("Choose the Person has diabetes or not :"  , options = ['Yes' , ' No'])

    totChol = st.number_input("Enter the Total Cholestrol level:", min_value=0.0)

    sysBP  = st.number_input("Enter the Systolic Blood Pressure :", min_value=0.0)

    diaBP = st.number_input("Enter the Diastolic Blood Pressure :", min_value=0.0)

    BMI  = st.number_input("Enter the Body Mass Index :", min_value=0.0)

    heartRate = st.number_input("Enter the Heart Rate :", min_value=0.0)

    glucose  = st.number_input("Enter the Glucose Level :", min_value=0.0)



# Helper Function ---------------------------------------------

    # First we define the function for columns in which we are working 
def predictive(gender, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, 
               sysBP, diaBP, BMI, heartRate, glucose):
    
    ## Encode the categorical columns -----------------
    gender_encoded = label_encoder.fit_transform([gender])[0] 
    currentSmoker_encoded = label_encoder.fit_transform([currentSmoker])[0] 
    BPMeds_encoded = label_encoder.fit_transform([BPMeds])[0] 
    prevalentStroke_encoded = label_encoder.fit_transform([prevalentStroke])[0] 
    prevalentHyp_encoded = label_encoder.fit_transform([prevalentHyp])[0]
    diabetes_encoded = label_encoder.fit_transform([diabetes])[0] 
    
    ## Prepare Features Array 
    features = np.array([[gender_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded ,prevalentHyp_encoded ,
          diabetes_encoded , totChol, sysBP, diaBP, BMI, heartRate, glucose]])       # Then we convert are columns to 2d array 
    print(features )
    ## Scalling
    scaled_features = scaler.transform(features)            # Then we do transform features columns 
    print(scaled_features)
    ## Predict by model
    result = Model.predict(scaled_features)                  # Atlast we finally predict the model by logisicRegression 
    print(result)
    return result[0]



# Predict Button ------------------------------------------------------
if st.button('Predict'):
    result =predictive(gender, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
    if result == 1:
        st.write("Person has a heart disease :")
    else:
        st.write("Person does  not has a  heart disease :")

st.markdown("</div>", unsafe_allow_html=True)



 