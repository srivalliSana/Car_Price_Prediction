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

query_params = st.experimental_get_query_params()
if "page" in query_params:
    st.session_state.page = query_params["page"][0]

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
    st.subheader("🎯 Predict Car Price")

    car_name = st.text_input("Car Name", "Hyundai i20")  # kept for consistency
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
    kms_driven = st.number_input("Driven kms", min_value=0, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Selling type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [0, 1, 2, 3])

    input_df = pd.DataFrame([{
        "Car_Name": car_name,
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": kms_driven,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Selling Price: **{round(prediction, 2)} lakhs**")

# --- Model Performance ---
elif st.session_state.page == "Model Performance":
    st.subheader("📉 Model Performance")

    X = data.drop("Selling_Price", axis=1)
    y = data["Selling_Price"]

    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    # Actual vs Predicted plot
    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.6)
    ax.plot([0, max(y)], [0, max(y)], 'r--')
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
