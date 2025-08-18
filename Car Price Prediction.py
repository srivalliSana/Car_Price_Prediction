import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("car data.csv")

@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction.pkl")

df = load_data()
model = load_model()

# Tabs
tabs = st.tabs(["🏠 Home", "📊 Data Trends", "🤖 Feature Importance", "🔮 Prediction"])

# ----- HOME TAB -----
with tabs[0]:
    st.title("Car Price Prediction App 🚗")
    st.markdown("""
        <style>
        .big-font {font-size:22px ; font-weight:600;}
        </style>
        <div class="big-font">Welcome! Explore, analyze and predict used car prices.<br>
        Use the tabs above for Data Trends and Price Prediction.</div>
    """, unsafe_allow_html=True)
    st.image('https://images.unsplash.com/photo-1485463613374-7c9c9233c1a5?auto=format&fit=crop&w=800&q=80', use_container_width=True)
    st.write("This app allows you to:")
    st.markdown("- **View trends** in the car dataset")
    st.markdown("- **Predict selling price** for any car")
    st.markdown("Start by exploring the Data Trends, or jump straight to Prediction!")

# ----- DATA TRENDS TAB -----
with tabs[1]:
    st.header("📊 Car Data Trends")

    st.subheader("Top 10 Popular Car Models")
    top_cars = df['Car_Name'].value_counts().head(10)
    st.bar_chart(top_cars)

    st.subheader("Average Selling Price by Fuel Type")
    avg_price_fuel = df.groupby('Fuel_Type')['Selling_Price'].mean().round(2)
    st.table(avg_price_fuel)

    st.subheader("Yearly Trends in Average Selling Price")
    fig, ax = plt.subplots()
    yearly_price = df.groupby('Year')['Selling_Price'].mean()
    ax.plot(yearly_price.index, yearly_price.values, marker='o', linestyle='-', color='green')
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Selling Price (Lakh)")
    ax.set_title("Yearly Average Selling Price")
    st.pyplot(fig)

    st.subheader("Distribution of Selling Prices")
    fig, ax = plt.subplots()
    sns.histplot(df["Selling_Price"], bins=30, color='skyblue', ax=ax)
    ax.set_title("Selling Price Distribution")
    st.pyplot(fig)

# ----- FEATURE IMPORTANCE TAB -----
with tabs[2]:
    st.header("🤖 Feature Importance")

    try:
        feature_names = df.drop(columns=["Selling_Price"]).columns.tolist()
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False)
        st.bar_chart(fi_df.set_index('Feature'))
        st.dataframe(fi_df, use_container_width=True)
    except AttributeError:
        st.warning("Feature importance is not available for this model type.")

# ----- PREDICTION TAB -----
with tabs[3]:
    st.header("🔮 Predict Your Car's Selling Price")

    with st.form(key='predict_form'):
        col1, col2 = st.columns(2)
        car_name = col1.selectbox('Car Name', sorted(df['Car_Name'].unique()))
        year = col2.number_input('Year', min_value=1990, max_value=2025, value=2018)
        present_price = col1.number_input('Present Price (Lakh)', min_value=0.0, value=5.0)
        driven_kms = col2.number_input('Driven Kms', min_value=0, value=10000)
        fuel_type = col1.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
        selling_type = col2.selectbox('Seller Type', ['Dealer', 'Individual'])
        transmission = col1.selectbox('Transmission', ['Manual', 'Automatic'])
        owner = col2.selectbox('Owner', [0, 1, 2, 3])

        submit = st.form_submit_button('✅ Predict Price')

    if submit:
        input_dict = {
            'Car_Name': car_name,
            'Year': year,
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Fuel_Type': fuel_type,
            'Selling_type': selling_type,
            'Transmission': transmission,
            'Owner': owner
        }
        input_df = pd.DataFrame([input_dict])

        # Reminder: Apply preprocessing if needed
        try:
            sell_price = model.predict(input_df)[0]
            st.success(f"Estimated Selling Price: ₹ {sell_price:.2f} Lakh")
            st.info("Note: Prediction accuracy depends on feature processing and model quality.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
