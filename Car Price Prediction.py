import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# Load dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("car data.csv")

df = load_data()

# =========================
# Load trained model
# =========================
@st.cache_resource
def load_model():
    model_path = "car_price_prediction.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# =========================
# Prepare training features reference
# =========================
# Drop target column if exists
if "Selling_Price" in df.columns:
    X = pd.get_dummies(df.drop(["Selling_Price"], axis=1), drop_first=True)
else:
    X = pd.get_dummies(df, drop_first=True)

training_features = X.columns  # all feature names used in training

# =========================
# Helper: prepare input for prediction
# =========================
def prepare_features(user_input: pd.DataFrame) -> pd.DataFrame:
    input_df = pd.get_dummies(user_input, drop_first=True)
    # Align columns to training features
    input_df = input_df.reindex(columns=training_features, fill_value=0)
    return input_df

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("🚗 Car Price App")
page = st.sidebar.radio(
    "Navigate",
    ["📂 Dataset Preview", "📊 Data Analysis", "📈 Model Performance", "🎯 Car Price Prediction"]
)

# =========================
# Pages
# =========================

# ---- Dataset Preview ----
if page == "📂 Dataset Preview":
    st.title("📂 Dataset Preview")
    st.write(df.head())

# ---- Data Analysis ----
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    st.write("Basic statistics:")
    st.write(df.describe())

    st.write("Fuel Type Distribution:")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Fuel_Type", ax=ax)
    st.pyplot(fig)

    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---- Model Performance ----
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    if model is None:
        st.error("⚠️ Model file not found.")
    else:
        try:
            X = pd.get_dummies(df.drop(["Selling_Price"], axis=1), drop_first=True)
            y = df["Selling_Price"]
            preds = model.predict(X)
            r2 = r2_score(y, preds)
            rmse = np.sqrt(mean_squared_error(y, preds))
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")
            fig, ax = plt.subplots()
            ax.scatter(y, preds, alpha=0.6)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not evaluate model: {e}")

# ---- Car Price Prediction ----
elif page == "🎯 Car Price Prediction":
    st.title("🎯 Car Price Prediction")

    st.subheader("Enter Car Details:")

    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
    kms_driven = st.number_input("KMs Driven", min_value=0, value=50000)
    fuel_type = st.selectbox("Fuel Type", df["Fuel_Type"].unique())
    selling_type = st.selectbox("Selling Type", df["Selling_type"].unique())
    transmission = st.selectbox("Transmission", df["Transmission"].unique())
    owner = st.selectbox("Owner", sorted(df["Owner"].unique()))

    user_input = pd.DataFrame({
        "Year": [year],
        "Present_Price": [present_price],
        "Driven_kms": [kms_driven],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner]
    })

    if st.button("Predict Price"):
        if model is None:
            st.error("⚠️ Model file not found.")
        else:
            try:
                features = prepare_features(user_input)
                prediction = model.predict(features)[0]
                st.success(f"💰 Estimated Selling Price: {prediction:.2f} lakhs")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
