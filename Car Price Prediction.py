import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

tabs = st.tabs(["🏠 Home", "📈 Data Overview", "📊 Data Analysis", "🤖 Feature Importance", "🔮 Prediction", "📝 About"])

# HOME TAB
with tabs[0]:
    st.title("Car Price Prediction App 🚗")
    st.image('https://images.unsplash.com/photo-1485463613374-7c9c9233c1a5?auto=format&fit=crop&w=800&q=80', use_column_width=True)
    st.markdown("""
    Welcome to your complete car price prediction platform!
    Explore data, analyze features, view model insights, and predict prices easily.
    """)

# DATA OVERVIEW TAB
with tabs[1]:
    st.header("📈 Data Overview")
    st.dataframe(df, use_container_width=True)
    st.write("Filter or search the table above for details.")
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

# DATA ANALYSIS TAB
with tabs[2]:
    st.header("📊 Data Analysis")

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='RdYlBu', ax=ax)
    st.pyplot(fig)

    st.subheader("Boxplot: Selling Price by Fuel Type")
    fig, ax = plt.subplots()
    sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Scatterplot: Present Price vs. Selling Price")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Present_Price", y="Selling_Price", hue="Fuel_Type", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Driven Kms")
    fig, ax = plt.subplots()
    sns.histplot(df["Driven_kms"], bins=30, color="orange", ax=ax)
    st.pyplot(fig)

# FEATURE IMPORTANCE TAB
with tabs[3]:
    st.header("🤖 Feature Importance")
    # If using a sklearn tree-based model
    if hasattr(model, 'feature_importances_'):
        feature_names = ['Year', 'Present_Price', 'Driven_kms', 'Owner']  # Update if using encoding/pipeline!
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        st.bar_chart(fi_df.set_index('Feature'))
        st.write(fi_df.sort_values('Importance', ascending=False))
    else:
        st.info("Feature importance plot is only available for tree-based models.")

# PREDICTION TAB
with tabs[4]:
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
        sell_price = model.predict(input_df)
        st.success(f"Estimated Selling Price: ₹ {sell_price:.2f} Lakh")

        st.subheader("Residual Plot Example")
        pred = model.predict(df[input_df.columns])
        residuals = df['Selling_Price'] - pred
        fig, ax = plt.subplots()
        ax.scatter(pred, residuals, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Predicted Selling Price")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        st.pyplot(fig)

        # Download result option
        st.download_button(
            label="Download this Prediction as CSV",
            data=input_df.assign(Predicted_Selling_Price=sell_price).to_csv(index=False),
            file_name="car_prediction.csv",
            mime="text/csv"
        )

# ABOUT TAB
with tabs[5]:
    st.header("📝 About This Project")
    st.write("""
    This app predicts used car prices based on features like year, present price, mileage, fuel, and more.
    Built with Streamlit, pandas, scikit-learn, seaborn, and matplotlib.
    Dataset: Provided by user (`car data.csv`)
    Model: Trained ML model (`car_price_prediction.pkl`)
    """)
    st.write("For queries or feedback, contact: *your-email@example.com*")
