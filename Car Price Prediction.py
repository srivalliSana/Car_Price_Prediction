import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the model
try:
    model = pickle.load(open("car_price_model.pkl", "rb"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Load the dataset
try:
    df = pd.read_csv("car_data.csv")
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# Ensure consistent data types
df['Car_Name'] = df['Car_Name'].astype(str)

# App layout
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")

# Sidebar navigation
tabs = ["Home", "Data Preview", "Feature Importance", "Predict Price"]
selected_tab = st.sidebar.radio("Navigate", tabs)

# Home Tab
if selected_tab == "Home":
    st.header("Welcome!")
    st.markdown("""
        This app predicts the **selling price** of a used car based on its features.
        Navigate through the tabs to explore data, understand feature importance, and make predictions.
    """)

# Data Preview Tab
elif selected_tab == "Data Preview":
    st.header("📊 Dataset Preview")
    st.dataframe(df, use_container_width=True)

# Feature Importance Tab
elif selected_tab == "Feature Importance":
    st.header("📈 Feature Importance")
    try:
        importances = model.feature_importances_
        feature_names = df.drop(columns=["Selling_Price"]).columns.tolist()

        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            st.bar_chart(fi_df.set_index('Feature'))
            st.dataframe(fi_df, use_container_width=True)
        else:
            st.error("Mismatch between model features and dataset columns.")
    except AttributeError:
        st.warning("Feature importance not available for this model.")

# Predict Price Tab
elif selected_tab == "Predict Price":
    st.header("🔮 Predict Car Selling Price")

    # Input fields
    car_name = st.selectbox("Car Name", df['Car_Name'].unique())
    year = st.slider("Year of Purchase", 1990, 2025, 2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Count", [0, 1, 2, 3])

    # Feature engineering
    car_age = 2025 - year
    fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
    fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
    seller_type_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    input_data = np.array([[present_price, kms_driven, owner, car_age,
                            fuel_type_diesel, fuel_type_petrol,
                            seller_type_individual, transmission_manual]])

    # Prediction
    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Selling Price: ₹ {prediction:.2f} lakhs")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
