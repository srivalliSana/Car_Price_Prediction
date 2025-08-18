import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------
# Load model & data safely
# -----------------------
MODEL_FILE = "Car Price Prediction.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        st.warning("⚠️ Model file not found. Using fallback dummy model.")
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        dummy_df = pd.DataFrame({
            "Year": [2015, 2016],
            "Present_Price": [5, 7],
            "Driven_kms": [50000, 30000],
            "Fuel_Type": [0, 1],
            "Selling_type": [0, 1],
            "Transmission": [0, 1],
            "Owner": [0, 0],
            "Selling_Price": [3.5, 5.0]
        })
        X = dummy_df.drop("Selling_Price", axis=1)
        y = dummy_df["Selling_Price"]
        return LinearRegression().fit(X, y)


@st.cache_data
def load_data():
    if os.path.exists("car data.csv"):
        return pd.read_csv("car data.csv")
    else:
        st.warning("⚠️ Dataset not found. Using fallback data.")
        return pd.DataFrame({
            "Car_Name": ["Swift", "Innova", "City"],
            "Year": [2015, 2017, 2018],
            "Present_Price": [5.5, 12.0, 9.5],
            "Driven_kms": [50000, 30000, 20000],
            "Fuel_Type": ["Petrol", "Diesel", "Petrol"],
            "Selling_type": ["Dealer", "Individual", "Dealer"],
            "Transmission": ["Manual", "Automatic", "Manual"],
            "Owner": [0, 0, 0],
            "Selling_Price": [3.5, 7.0, 5.5]
        })

model = load_model()
df = load_data()

# Encoding maps
fuel_dict = {"Petrol": 0, "Diesel": 1, "CNG": 2}
selling_dict = {"Dealer": 0, "Individual": 1}
trans_dict = {"Manual": 0, "Automatic": 1}

def encode_data(dataframe):
    df_copy = dataframe.copy()
    if "Fuel_Type" in df_copy:
        df_copy["Fuel_Type"] = df_copy["Fuel_Type"].map(fuel_dict)
    if "Selling_type" in df_copy:
        df_copy["Selling_type"] = df_copy["Selling_type"].map(selling_dict)
    if "Transmission" in df_copy:
        df_copy["Transmission"] = df_copy["Transmission"].map(trans_dict)
    return df_copy

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
    st.write("Welcome! Use the sidebar to explore the app.")

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

    input_df = pd.DataFrame([{
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_dict[fuel_type],
        "Selling_type": selling_dict[selling_type],
        "Transmission": trans_dict[transmission],
        "Owner": owner
    }])

    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Predicted Selling Price: {prediction:.2f} lakhs")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Performance":
    st.header("📊 Model Performance")

    try:
        df_encoded = encode_data(df)
        X = df_encoded.drop(["Selling_Price", "Car_Name"], axis=1)
        y = df_encoded["Selling_Price"]

        preds = model.predict(X)
        r2 = r2_score(y, preds)
        mse = mean_squared_error(y, preds)

        st.write(f"**R² Score:** {r2:.2f}")
        st.write(f"**MSE:** {mse:.2f}")

        fig, ax = plt.subplots()
        sns.scatterplot(x=y, y=preds, ax=ax)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not evaluate model: {e}")
