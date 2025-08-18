import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------
# Load model & data
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")  # your trained model file

@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")  # your dataset

model = load_model()
df = load_data()

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("🚗 Car Price App")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset", "Predict", "Model Performance"]
)

# -----------------------
# Pages
# -----------------------

if page == "Home":
    st.title("🚘 Car Price Prediction App")
    st.write("Welcome! Use the sidebar to navigate through the app.")

elif page == "Dataset":
    st.header("📂 Dataset Preview")
    st.dataframe(df.head())

elif page == "Predict":
    st.header("🎯 Car Price Prediction")

    # User inputs
    year = st.number_input("Year", min_value=1990, max_value=2025, step=1, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
    driven_kms = st.number_input("Driven KMs", min_value=0, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [0, 1, 2, 3])

    # Encoding (must match training)
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    selling_map = {"Dealer": 0, "Individual": 1}
    transmission_map = {"Manual": 0, "Automatic": 1}

    input_data = {
        "Year": [year],
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Fuel_Type": [fuel_map[fuel_type]],
        "Selling_type": [selling_map[selling_type]],
        "Transmission": [transmission_map[transmission]],
        "Owner": [owner]
    }

    input_df = pd.DataFrame(input_data)

    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Predicted Selling Price: {prediction:.2f} lakhs")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Performance":
    st.header("📉 Model Performance")

    # Prepare features & target
    X = df.drop(["Selling_Price", "Car_Name"], axis=1).copy()
    y = df["Selling_Price"]

    # Encode
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    selling_map = {"Dealer": 0, "Individual": 1}
    transmission_map = {"Manual": 0, "Automatic": 1}

    X["Fuel_Type"] = X["Fuel_Type"].map(fuel_map)
    X["Selling_type"] = X["Selling_type"].map(selling_map)
    X["Transmission"] = X["Transmission"].map(transmission_map)

    # Predict
    y_pred = model.predict(X)

    # Metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Squared Error:** {mse:.3f}")

    # Plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
