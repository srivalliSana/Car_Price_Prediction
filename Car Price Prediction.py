import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("car data.csv")

@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction.pkl")

df = load_data()
model = load_model()

tabs = st.tabs(["🏠 Home", "📊 Data Trends", "🤖 Feature Importance", "🔮 Prediction"])

with tabs[0]:
    st.title("Car Price Prediction App 🚗")
    st.markdown("""
        <style>
        .big-font {font-size:22px; font-weight:600;}
        </style>
        <div class="big-font">Welcome! Explore, analyze and predict used car prices.<br>
        Use the tabs above for Data Trends and Price Prediction.
        </div>
    """, unsafe_allow_html=True)
    st.image('https://images.unsplash.com/photo-1485463613374-7c9c9233dbd9?auto=format&fit=crop&w=800&q=80', use_container_width=True)
    st.write("This app allows you to:")
    st.markdown("- View trends in the dataset")
    st.markdown("- Predict selling prices for used cars")

with tabs[1]:
    st.header("Car Data Trends")
    st.subheader("Top 10 Popular Car Models")
    top_cars = df['Car_Name'].value_counts().head(10)
    st.bar_chart(top_cars)

    st.subheader("Average Selling Price by Fuel Type")
    avg_price_fuel = df.groupby('Fuel_Type')['Selling_Price'].mean()
    st.table(avg_price_fuel)

    st.subheader("Yearly Trends of Average Selling Prices")
    fig, ax = plt.subplots()
    yearly_avg = df.groupby('Year')['Selling_Price'].mean()
    ax.plot(yearly_avg.index, yearly_avg.values, marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Selling Price (Lakh)")
    st.pyplot(fig)

    st.subheader("Distribution of Selling Prices")
    fig, ax = plt.subplots()
    sns.histplot(df["Selling_Price"], bins=30, color='skyblue', ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.header("Feature Importance")
    try:
        # Get the expected number of features from model input or training process
        # Here assuming you have a list or pipeline used at training; adjust accordingly
        feature_names = list(df.drop(columns=['Selling_Price']).columns)

        importances = model.feature_importances_
        
        # If lengths mismatch, trim or warn
        if len(feature_names) != len(importances):
            st.warning(f"Feature names ({len(feature_names)}) and importances ({len(importances)}) length mismatch. Showing truncated results.")
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]

        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        st.bar_chart(fi_df.set_index('Feature'))
        st.dataframe(fi_df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not display feature importance: {e}")

with tabs[3]:
    st.header("Predict Selling Price")

    with st.form(key='predict_form'):
        col1, col2 = st.columns(2)
        car_name = col1.selectbox('Car Name', sorted(df['Car_Name'].unique()))
        year = col2.number_input('Year', min_value=1990, max_value=2025, value=2015)
        present_price = col1.number_input('Present Price (Lakh)', min_value=0.0, value=5.0)
        driven_kms = col2.number_input('Driven Kms', min_value=0, value=15000)
        fuel_type = col1.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
        selling_type = col2.selectbox('Seller Type', ['Dealer', 'Individual'])
        transmission = col1.selectbox('Transmission', ['Manual', 'Automatic'])
        owner = col2.selectbox('Number of Owners', [0, 1, 2, 3])

        submitted = st.form_submit_button('Predict Price')

    if submitted:
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

        # Note: If your model needs preprocessing (encoding/scaling), prepare accordingly.
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakh")
