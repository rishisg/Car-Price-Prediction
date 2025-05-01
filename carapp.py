# Step 8: Streamlit App Development
# Now, we can create the Streamlit app (carapp.py) to allow the user to input car details and predict the price.
# Streamlit App (carapp.py):

import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load models
@st.cache_resource
def load_rf_model():
    with open('random_forest_model_car.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_xgb_model():
    with open('xgboost_model_car.pkl', 'rb') as f:
        return pickle.load(f)

# Streamlit app
st.title("Car Price Prediction App")

# Model selection
model_option = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

# Input fields
brand = st.text_input("Brand")
model = st.text_input("Model")
year = st.number_input("Year", min_value=1900, max_value=2025, step=1)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
mileage = st.number_input("Mileage (km)", min_value=0, step=100)
doors = st.number_input("Doors", min_value=1, max_value=5, step=1)
owner_count = st.number_input("Owner Count", min_value=1, step=1)

# Prepare input data
input_data = pd.DataFrame([[brand, model, year, engine_size, fuel_type, transmission, mileage, doors, owner_count]],
                          columns=['Brand', 'Model', 'Year', 'Engine_Size', 'Fuel_Type', 'Transmission', 'Mileage', 'Doors', 'Owner_Count'])

# Feature encoding
label_encoder = LabelEncoder()
input_data['Brand'] = label_encoder.fit_transform(input_data['Brand'])
input_data['Model'] = label_encoder.fit_transform(input_data['Model'])
input_data['Fuel_Type'] = label_encoder.fit_transform(input_data['Fuel_Type'])
input_data['Transmission'] = label_encoder.fit_transform(input_data['Transmission'])

# Prediction
if st.button("Predict Price"):
    if model_option == "Random Forest":
        model = load_rf_model()
    else:
        model = load_xgb_model()

    prediction = model.predict(input_data)

    st.write(f"The predicted car price is: ${prediction[0]:,.2f}")
