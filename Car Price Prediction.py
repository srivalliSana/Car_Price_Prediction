import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load the trained pipeline
# ----------------------------
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("Car Price Prediction.pkl")  # must match your notebook's filename
    except FileNotFoundError:
        st.error("❌ Model file not found. Make sure 'Car Price Prediction.pkl' is in the same folder as this app.")
        return None

pipeline = load_pipeline()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🚗 Car Price Prediction App")

st.write("Enter car details below to predict its selling price.")

# Input fields
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.number_input("Number of Previous Owners", min_value=0, max_value=5, step=1)

# Prepare input for pipeline
input_data = pd.DataFrame([{
    "Year": year,
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Fuel_Type": fuel_type,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Owner": owner
}])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Price"):
    if pipeline:
        prediction = pipeline.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: {prediction:.2f} Lakhs")
