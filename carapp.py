import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load XGBoost model
@st.cache_resource
def load_xgb_model():
    with open('xgboost_model_car.pkl', 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Car Price Prediction App using XGBoost")

# Dropdown fields for car features
brand = st.selectbox("Select Car Brand", ['Brand1', 'Brand2', 'Brand3'])  # Replace with actual brand list
model = st.selectbox("Select Car Model", ['Model1', 'Model2', 'Model3'])  # Replace with actual model list
year = st.selectbox("Select Car Year", list(range(1900, 2026)))  # Dropdown with years from 1900 to 2025
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
fuel_type = st.selectbox("Select Fuel Type", ["Petrol", "Diesel", "Electric"])
transmission = st.selectbox("Select Transmission", ["Manual", "Automatic"])
mileage = st.number_input("Mileage (km)", min_value=0, step=100)
doors = st.selectbox("Select Number of Doors", [2, 3, 4, 5])  # Dropdown for number of doors
owner_count = st.selectbox("Select Owner Count", [1, 2, 3, 4])  # Dropdown for number of previous owners

# Prepare input data
input_data = pd.DataFrame([[brand, model, year, engine_size, fuel_type, transmission, mileage, doors, owner_count]],
                          columns=['Brand', 'Model', 'Year', 'Engine_Size', 'Fuel_Type', 'Transmission', 'Mileage', 'Doors', 'Owner_Count'])

# Feature encoding using LabelEncoder
label_encoder = LabelEncoder()
input_data['Brand'] = label_encoder.fit_transform(input_data['Brand'])
input_data['Model'] = label_encoder.fit_transform(input_data['Model'])
input_data['Fuel_Type'] = label_encoder.fit_transform(input_data['Fuel_Type'])
input_data['Transmission'] = label_encoder.fit_transform(input_data['Transmission'])

# Load model
model = load_xgb_model()

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.write(f"The predicted car price is: ${prediction[0]:,.2f}")
