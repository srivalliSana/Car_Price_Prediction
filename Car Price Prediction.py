import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("car data.csv")

@st.cache_resource
def load_pipeline():
    # This pipeline includes all preprocessing steps + trained model
    return joblib.load("car_price_pipeline.pkl")

df = load_data()
pipeline = load_pipeline()

tabs = st.tabs([
    "🏠 Home",
    "🔍 Data Explorer",
    "📊 Visual Analysis",
    "📈 Advanced Analysis",
    "🤖 Feature Importance",
    "🧪 Model Evaluation",
    "🔮 Prediction"
])

# HOME
with tabs[0]:
    st.title("Car Price Prediction App 🚗")
    st.image(
      'https://images.unsplash.com/photo-1485463613374-7c9c9233c1a5?auto=format&fit=crop&w=800&q=80',
      use_container_width=True)
    st.markdown(
        "Explore, analyze, and predict used car prices easily through this interactive app."
    )

# DATA EXPLORER
with tabs[1]:
    st.header("🔍 Data Explorer")
    try:
        st.dataframe(df, use_container_width=True)
        brands = st.multiselect("Filter by Car Brand:", sorted(df['Car_Name'].unique()))
        years = st.slider("Year range:", int(df['Year'].min()), int(df['Year'].max()),
                          value=(int(df['Year'].min()), int(df['Year'].max())))
        fuels = st.multiselect("Fuel Type:", sorted(df['Fuel_Type'].unique()))

        filtered_df = df.copy()
        if brands:
            filtered_df = filtered_df[filtered_df['Car_Name'].isin(brands)]
        filtered_df = filtered_df[(filtered_df['Year'] >= years[0]) & (filtered_df['Year'] <= years[1])]
        if fuels:
            filtered_df = filtered_df[filtered_df['Fuel_Type'].isin(fuels)]

        st.write(f"Filtered data has {len(filtered_df)} rows.")
        st.dataframe(filtered_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying filtered data: {e}")

# VISUAL ANALYSIS
with tabs[2]:
    st.header("📊 Visual Analysis")
    try:
        fig, ax = plt.subplots()
        sns.boxplot(x='Fuel_Type', y='Selling_Price', hue='Selling_type', data=df, ax=ax)
        st.subheader("Selling Price by Fuel Type & Seller Type")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering boxplot: {e}")

    try:
        fig, ax = plt.subplots()
        sns.scatterplot(x='Present_Price', y='Selling_Price', hue='Fuel_Type', style='Transmission', data=df, ax=ax)
        st.subheader("Present Price vs Selling Price by Fuel Type & Transmission")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering scatterplot: {e}")

    try:
        fig, ax = plt.subplots()
        sns.histplot(df['Selling_Price'], bins=30, kde=True, color='skyblue', ax=ax)
        st.subheader("Selling Price Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering selling price histogram: {e}")

    try:
        fig, ax = plt.subplots()
        sns.histplot(df['Driven_kms'], bins=30, color='darkorange', ax=ax)
        st.subheader("Driven Kms Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering driven kms histogram: {e}")

# ADVANCED ANALYSIS
with tabs[3]:
    st.header("📈 Advanced Analysis")
    try:
        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.subheader("Correlation Matrix")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering correlation heatmap: {e}")

    try:
        st.subheader("Top 5 correlations with Selling Price")
        corr_target = corr['Selling_Price'].drop('Selling_Price').abs().sort_values(ascending=False).head(5)
        st.bar_chart(corr_target)
    except Exception as e:
        st.error(f"Error showing top correlations: {e}")

    try:
        Q1 = df['Selling_Price'].quantile(0.25)
        Q3 = df['Selling_Price'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['Selling_Price'] < Q1 - 1.5 * IQR) | (df['Selling_Price'] > Q3 + 1.5 * IQR)]

        fig, ax = plt.subplots()
        ax.scatter(df['Present_Price'], df['Selling_Price'], alpha=0.5, label='Normal')
        ax.scatter(outliers['Present_Price'], outliers['Selling_Price'], color='red', label='Outliers')
        ax.set_xlabel("Present Price")
        ax.set_ylabel("Selling Price")
        ax.legend()
        st.subheader("Outlier Identification in Price vs Present Price")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering outlier detection: {e}")

    try:
        cluster_data = df[['Present_Price','Selling_Price']].dropna()
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(cluster_data)
        cluster_data['Cluster'] = clusters

        fig, ax = plt.subplots()
        sns.scatterplot(x='Present_Price', y='Selling_Price', hue='Cluster', data=cluster_data, palette='Set1', ax=ax)
        st.subheader("KMeans Clustering on Present Price & Selling Price")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering clustering: {e}")

# FEATURE IMPORTANCE
with tabs[4]:
    st.header("🤖 Feature Importance")
    try:
        if hasattr(pipeline, "feature_importances_"):
            features = list(df.drop(columns=['Selling_Price']).columns)
            importances = pipeline.feature_importances_
            if len(features) != len(importances):
                min_len = min(len(features), len(importances))
                features = features[:min_len]
                importances = importances[:min_len]
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            st.bar_chart(fi_df.set_index('Feature'))
            st.dataframe(fi_df, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.error(f"Error showing feature importances: {e}")

# MODEL EVALUATION
with tabs[5]:
    st.header("🧪 Model Evaluation")
    try:
        X = df.drop(columns=['Selling_Price'])
        y = df['Selling_Price']
        preds = pipeline.predict(X)
        st.metric("R² Score", f"{r2_score(y, preds):.3f}")
        st.metric("Mean Absolute Error", f"{mean_absolute_error(y, preds):.3f}")
        st.metric("Root Mean Squared Error", f"{mean_squared_error(y, preds, squared=False):.3f}")

        fig, ax = plt.subplots()
        ax.scatter(y, preds, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Actual Selling Price")
        ax.set_ylabel("Predicted Selling Price")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        residuals = y - preds
        sns.histplot(residuals, bins=30, kde=True, color="purple", ax=ax)
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")

# PREDICTION TAB
with tabs[6]:
    st.header("🔮 Predict Selling Price")
    with st.form(key="predict_form"):
        col1, col2 = st.columns(2)
        car_name = col1.selectbox("Car Name", sorted(df['Car_Name'].unique()))
        year = col2.number_input("Year", min_value=1990, max_value=2025, value=2017)
        present_price = col1.number_input("Present Price (Lakh)", min_value=0.0, value=5.0)
        driven_kms = col2.number_input("Driven Kms", min_value=0, value=10000)
        fuel_type = col1.selectbox("Fuel Type", sorted(df['Fuel_Type'].unique()))
        selling_type = col2.selectbox("Seller Type", sorted(df['Selling_type'].unique()))
        transmission = col1.selectbox("Transmission", sorted(df['Transmission'].unique()))
        owner = col2.selectbox("Owner", sorted(df['Owner'].unique()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        user_input = pd.DataFrame([{
            'Car_Name': car_name,
            'Year': year,
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Fuel_Type': fuel_type,
            'Selling_type': selling_type,
            'Transmission': transmission,
            'Owner': owner
        }])
        try:
            pred_price = pipeline.predict(user_input)[0]
            st.success(f"Estimated Selling Price: ₹ {pred_price:.2f} Lakh")
        except Exception as e:
            st.error(f"Prediction failed. Confirm your input matches training data format. Error: {e}")
