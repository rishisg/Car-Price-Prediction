import pickle
import streamlit as st
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
brands = ['Kia', 'Chevrolet', 'Mercedes', 'Audi', 'Volkswagen', 'Toyota', 'Honda', 'BMW', 'Hyundai', 'Ford']
models = {
    'Kia': ['Rio', 'Sportage', 'Optima'],
    'Chevrolet': ['Malibu', 'Equinox', 'Impala'],
    'Mercedes': ['GLA', 'GLC', 'E-Class'],
    'Audi': ['Q5', 'A3', 'A4'],
    'Volkswagen': ['Golf', 'Tiguan', 'Passat'],
    'Toyota': ['Camry', 'RAV4', 'Corolla'],
    'Honda': ['Civic', 'CR-V', 'Accord'],
    'BMW': ['5 Series', 'X5', '3 Series'],
    'Hyundai': ['Elantra', 'Sonata', 'Tucson'],
    'Ford': ['Explorer', 'Fiesta']
}

# User input fields
brand = st.selectbox("Select Car Brand", brands)
model = st.selectbox("Select Car Model", models[brand])  # Models change based on the selected brand
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
