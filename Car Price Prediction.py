import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# Load model and dataset
# =========================
model = pickle.load(open("car_price_prediction.pkl", "rb"))
data = pd.read_csv("car data.csv")

# =========================
# Custom CSS Navbar
# =========================
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: space-around;
        background-color: #0d6efd;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .navbar a {
        text-decoration: none;
        color: white;
        font-weight: bold;
        padding: 8px 15px;
        border-radius: 8px;
        transition: 0.3s;
        font-size: 18px;
    }
    .navbar a:hover {
        background-color: #084298;
    }
    .active {
        background-color: #084298;
    }
    </style>
""", unsafe_allow_html=True)

# Navbar items
# Navbar items
nav_items = {
    "Home": "🏠",
    "Dataset": "📊",
    "EDA": "📈",
    "Predict": "🎯",
    "Model Performance": "📉",
    "About": "ℹ️"
}

if "page" not in st.session_state:
    st.session_state.page = "Home"

nav_html = '<div class="navbar">'
for page, icon in nav_items.items():
    if st.session_state.page == page:
        nav_html += f'<a class="active" href="?page={page}">{icon} {page}</a>'
    else:
        nav_html += f'<a href="?page={page}">{icon} {page}</a>'
nav_html += "</div>"
st.markdown(nav_html, unsafe_allow_html=True)

# ✅ Updated method
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]


# =========================
# Pages
# =========================

# --- Home ---
if st.session_state.page == "Home":
    st.title("🚗 Car Price Prediction App")
    st.write("Welcome! Use the navigation bar above to explore the app.")

# --- Dataset ---
elif st.session_state.page == "Dataset":
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.subheader("🔎 Data Info")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    st.write("Columns:", list(data.columns))

# --- EDA ---
elif st.session_state.page == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    # Distribution of Selling Price
    fig, ax = plt.subplots()
    sns.histplot(data["Selling_Price"], kde=True, ax=ax)
    ax.set_title("Selling Price Distribution")
    st.pyplot(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# --- Predict ---
elif st.session_state.page == "Predict":
    st.header("🎯 Car Price Prediction")

    # User inputs
    year = st.number_input("Year", min_value=1990, max_value=2025, step=1, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
    driven_kms = st.number_input("Driven KMs", min_value=0, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [0, 1, 2, 3])

    # Encoding categorical variables (must match training!)
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


# --- Model Performance ---
elif st.session_state.page == "Model Performance":
    st.header("📉 Model Performance")

    # Separate features and target
    X = df.drop(["Selling_Price", "Car_Name"], axis=1).copy()
    y = df["Selling_Price"]

    # Encode categorical variables
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    selling_map = {"Dealer": 0, "Individual": 1}
    transmission_map = {"Manual": 0, "Automatic": 1}

    X["Fuel_Type"] = X["Fuel_Type"].map(fuel_map)
    X["Selling_type"] = X["Selling_type"].map(selling_map)
    X["Transmission"] = X["Transmission"].map(transmission_map)

    # Predictions
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Squared Error:** {mse:.3f}")

    # Plot actual vs predicted
    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


# --- About ---
elif st.session_state.page == "About":
    st.subheader("ℹ️ About this Project")
    st.write("""
    This Car Price Prediction App was built using:
    - Streamlit (Frontend + Dashboard)
    - Machine Learning (Regression Model)
    - Exploratory Data Analysis (EDA)

    🔹 You can explore the dataset, visualize trends, 
    predict car prices, and evaluate model performance.
    """)
