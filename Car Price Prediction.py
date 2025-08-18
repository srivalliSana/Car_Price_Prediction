import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Load data & model
@st.cache_data
def load_data():
    df = pd.read_csv("C://Users//srivalli sana//Downloads//car data.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction.pkl")

df = load_data()
model = load_model()

# Navigation bar
tabs = st.tabs(["🏠 Home", "📊 Data Trends", "🔮 Prediction"])

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
    st.image('https://images.unsplash.com/photo-1485463613374-7c9c9233c1a5?auto=format&fit=crop&w=800&q=80', use_column_width=True)
    st.write("This app allows you to:")
    st.markdown("- **View trends** in the car dataset")
    st.markdown("- **Predict selling price** for any car")
    st.markdown("Start by exploring the Data Trends, or jump straight to Prediction!")

# ----- DATA TRENDS TAB -----
with tabs[1]:
    st.header("📊 Car Data Trends")
    # Basic stats
    st.subheader("Top 10 Popular Car Models")
    top_cars = df['Car_Name'].value_counts().head(10)
    st.bar_chart(top_cars)

    st.subheader("Average Selling Price by Fuel Type")
    avg_price_fuel = df.groupby('Fuel_Type')['Selling_Price'].mean()
    st.table(avg_price_fuel)

    st.subheader("Yearly Trends in Average Selling Price")
    fig, ax = plt.subplots()
    yearly_price = df.groupby('Year')['Selling_Price'].mean()
    ax.plot(yearly_price.index, yearly_price.values, marker='o')
    ax.set_xlabel("Year"); ax.set_ylabel("Avg Selling Price (Lakh)");
    st.pyplot(fig)

    st.subheader("Distribution of Selling Prices")
    fig, ax = plt.subplots()
    sns.histplot(df["Selling_Price"], bins=30, color='skyblue', ax=ax)
    st.pyplot(fig)

# ----- PREDICTION TAB -----
with tabs[2]:
    st.header("🔮 Predict Your Car's Selling Price")

    # User input form
    st.subheader("Enter Car Details:")
    form = st.form(key='predict_form')
    col1, col2 = form.columns(2)
    car_name = col1.selectbox('Car Name', sorted(df['Car_Name'].unique()))
    year = col2.number_input('Year', min_value=1990, max_value=2025, value=2018)
    present_price = col1.number_input('Present Price (Lakh)', min_value=0.0, value=5.0)
    driven_kms = col2.number_input('Driven Kms', min_value=0, value=10000)
    fuel_type = col1.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    selling_type = col2.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = col1.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = col2.selectbox('Owner', [0, 1, 2, 3])
    submit = form.form_submit_button('✅ Predict Price')

    if submit:
        # Feature engineering (replicate what your model expects!)
        input_dict = {'Car_Name': car_name,
                      'Year': year,
                      'Present_Price': present_price,
                      'Driven_kms': driven_kms,
                      'Fuel_Type': fuel_type,
                      'Selling_type': selling_type,
                      'Transmission': transmission,
                      'Owner': owner}
        # If your model expects encoded inputs, use your pipeline or manual encoding here
        input_df = pd.DataFrame([input_dict])
        sell_price = model.predict(input_df)[0]
        st.success(f"Estimated Selling Price: ₹ {sell_price:.2f} Lakh")

        st.info("Note: Prediction accuracy depends on feature processing and your model quality.")

---

## How does this app work?

- Uses **streamlit tabs** (with icons: Home, Data Trends, Prediction).
- Each tab is a different part of the app—no dropdown/select menu needed for navigation.
- Home page introduces your app. Data Trends shows charts, tables, and analysis on your dataset. Prediction lets users enter car details and get predicted price.
- Visually appealing and interactive.

---

## How to write advanced Streamlit navigation apps

1. **Use tabs or multipage (`st.tabs()`, not only sidebar) for a modern navigation feel**
2. **Add icons/emojis** to tab names for friendly UX
3. **Split each function (intro, analysis, prediction) into its own tab/page**
4. **Include charts, tables, and images to enrich user experience**
5. **Always keep your input features and model preprocessing consistent**
6. **Make your interface interactive and clean**

---

Let me know if you want to add more pages (like uploading a CSV, batch predictions, feature explanations, etc.) or customize the visual theme!
