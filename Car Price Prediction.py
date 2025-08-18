import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# Load model and dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("car data.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("car_price_prediction.pkl", "rb"))

df = load_data()
model = load_model()

st.set_page_config(page_title="Car Price Prediction App", layout="wide")

# =========================
# Custom CSS Navbar
# =========================
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: center;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-item {
        margin: 0 20px;
        text-align: center;
        cursor: pointer;
        color: #555;
        font-size: 16px;
        text-decoration: none;
    }
    .nav-item:hover {
        color: #000;
    }
    .nav-icon {
        font-size: 22px;
        display: block;
    }
    .active {
        color: #4CAF50;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Navbar with icons (using emojis for simplicity)
nav_items = {
    "Dataset": "📊",
    "EDA": "📈",
    "Features": "⚡",
    "Predict": "💰"
}

# Store selection in session_state
if "page" not in st.session_state:
    st.session_state.page = "Dataset"

# Render Navbar
cols = st.columns(len(nav_items))
for i, (page, icon) in enumerate(nav_items.items()):
    if cols[i].button(f"{icon}\n{page}"):
        st.session_state.page = page

st.title("🚗 Car Price Prediction App")

# =========================
# Dataset Overview
# =========================
if st.session_state.page == "Dataset":
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df.head())

    st.write("**Shape of dataset:**", df.shape)
    st.write("**Column Info:**")
    st.write(df.dtypes)

    st.write("**Summary Statistics:**")
    st.write(df.describe())

# =========================
# Exploratory Data Analysis
# =========================
elif st.session_state.page == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    # Correlation heatmap
    st.write("### 🔥 Correlation Heatmap")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())

    # Distribution of Selling Price
    st.write("### 💰 Distribution of Selling Price")
    plt.figure(figsize=(8,5))
    sns.histplot(df['Selling_Price'], kde=True, bins=30, color="blue")
    st.pyplot(plt.gcf())

    # Categorical analysis
    cat_col = st.selectbox("Select Categorical Column to Analyze", df.select_dtypes("object").columns)
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x=cat_col, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# =========================
# Feature Importance
# =========================
elif st.session_state.page == "Features":
    st.subheader("⚡ Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = df.drop("Selling_Price", axis=1).columns
        imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x="Importance", y="Feature", data=imp_df, palette="magma")
        st.pyplot(plt.gcf())
    else:
        st.warning("Model does not provide feature importance.")

# =========================
# Price Prediction
# =========================
elif st.session_state.page == "Predict":
    st.subheader("🎯 Predict Car Price")

    # Inputs
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
    kms_driven = st.number_input("Kms Driven", min_value=0, step=500)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [0,1,2,3])

    # Encode categorical variables (adjust mapping based on your notebook preprocessing)
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 0, "Automatic": 1}

    features = np.array([[present_price, kms_driven, owner, year, 
                          fuel_map[fuel_type], seller_map[seller_type], trans_map[transmission]]])

    if st.button("Predict Price"):
        prediction = model.predict(features)[0]
        st.success(f"💰 Predicted Selling Price: **{round(prediction, 2)} lakhs**")

# =========================
# Sidebar Model Performance
# =========================
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Model Performance"):
    X = df.drop("Selling_Price", axis=1, errors="ignore")
    y = df["Selling_Price"]
    y_pred = model.predict(X)

    st.sidebar.write("📈 R² Score:", round(r2_score(y, y_pred), 3))
    st.sidebar.write("📉 RMSE:", round(np.sqrt(mean_squared_error(y, y_pred)), 3))
