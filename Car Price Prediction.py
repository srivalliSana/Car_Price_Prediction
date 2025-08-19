import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load pipeline
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("Car Price Prediction.pkl")

model = load_model()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")

st.markdown("Enter the car details below and get the predicted selling price.")

# -----------------------
# User Inputs
# -----------------------
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0)
    kms_driven = st.number_input("Kms Driven", min_value=0, value=50000)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    car_name = st.text_input("Car Name", "Swift")

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "Year": [year],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_type],
        "Seller_Type": [seller_type],
        "Transmission": [transmission],
        "Car_Name": [car_name]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Selling Price: **{prediction:.2f} Lakhs**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------
# Dataset Preview (Optional)
# -----------------------
with st.expander("📂 About this Model"):
    st.write("This model was trained on `car data.csv` using a pipeline with preprocessing + regression.")
    st.write("It automatically handles categorical features (Fuel Type, Seller Type, Transmission, Car Name).")
