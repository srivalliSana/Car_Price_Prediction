import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("car data.csv")

@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction.pkl")

df = load_data()
model = load_model()

tabs = st.tabs([
    "🏠 Home",
    "🔍 Data Explorer",
    "📊 Visual Analysis",
    "📈 Advanced Analysis",
    "🤖 Feature Importance",
    "🧪 Model Evaluation",
    "🔮 Prediction"
])

with tabs[0]:
    st.title("Car Price Prediction App 🚗")
    st.image('https://images.unsplash.com/photo-1485463613374-7c9c9233c1a5?auto=format&fit=crop&w=800&q=80', use_container_width=True)
    st.markdown("Explore, analyze, and predict used car prices. All tabs are error-safe.")

with tabs[1]:
    st.header("🔍 Data Explorer")
    try:
        st.dataframe(df, use_container_width=True)
        brands = st.multiselect("Car Brand", sorted(df['Car_Name'].unique()))
        years = st.slider("Year Range:", int(df['Year'].min()), int(df['Year'].max()),
                         (int(df['Year'].min()), int(df['Year'].max())))
        fuel = st.multiselect("Fuel Type", df['Fuel_Type'].unique())
        filtered = df.copy()
        if brands: filtered = filtered[filtered["Car_Name"].isin(brands)]
        if years: filtered = filtered[(filtered["Year"] >= years) & (filtered["Year"] <= years[1])]
        if fuel: filtered = filtered[filtered["Fuel_Type"].isin(fuel)]
        st.write(f"Filtered rows: {len(filtered)}")
        st.dataframe(filtered, use_container_width=True)
    except Exception as ex:
        st.error(f"Could not filter or show data: {ex}")

with tabs[2]:
    st.header("📊 Visual Analysis")
    try:
        st.subheader("Selling Price by Fuel Type & Seller")
        fig, ax = plt.subplots()
        sns.boxplot(x="Fuel_Type", y="Selling_Price", hue="Selling_type", data=df, ax=ax)
        st.pyplot(fig)
    except Exception as ex:
        st.info(f"Boxplot error: {ex}")

    try:
        st.subheader("Present Price vs. Selling Price Scatter (By Transmission)")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Present_Price", y="Selling_Price", style="Transmission", hue="Fuel_Type", data=df, ax=ax)
        st.pyplot(fig)
    except Exception as ex:
        st.info(f"Scatterplot error: {ex}")

    try:
        st.subheader("Selling Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Selling_Price"], bins=30, kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)
    except Exception as ex:
        st.info(f"Histogram error: {ex}")

    try:
        st.subheader("Driven Kms Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Driven_kms"], bins=40, color="darkorange", ax=ax)
        st.pyplot(fig)
    except Exception as ex:
        st.info(f"Histogram (kms) error: {ex}")

with tabs[3]:
    st.header("📈 Advanced Analysis")
    try:
        st.subheader("Correlation Matrix")
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.subheader("Top 5 Feature Correlations with Selling Price")
        corr_target = corr['Selling_Price'].drop('Selling_Price').sort_values(key=abs, ascending=False).head(5)
        st.bar_chart(corr_target)
    except Exception as ex:
        st.info(f"Correlation error: {ex}")

    try:
        st.subheader("Outlier Detection: Selling Price vs. Present Price")
        Q1 = df['Selling_Price'].quantile(0.25)
        Q3 = df['Selling_Price'].quantile(0.75)
        outlier_mask = (df['Selling_Price'] < Q1 - 1.5*(Q3-Q1)) | (df['Selling_Price'] > Q3 + 1.5*(Q3-Q1))
        fig, ax = plt.subplots()
        ax.scatter(df[~outlier_mask]['Present_Price'], df[~outlier_mask]['Selling_Price'], label='Normal', alpha=0.6)
        ax.scatter(df[outlier_mask]['Present_Price'], df[outlier_mask]['Selling_Price'], color='red', label='Outliers', alpha=0.7)
        ax.legend(); ax.set_xlabel("Present Price"); ax.set_ylabel("Selling Price")
        st.pyplot(fig)
    except Exception as ex:
        st.info(f"Outlier plot error: {ex}")

    try:
        from sklearn.cluster import KMeans
        st.subheader("KMeans Clustering (2D: Present Price vs. Selling Price)")
        cluster_data = df[['Present_Price', 'Selling_Price']].dropna()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(cluster_data)
        cluster_data['cluster'] = kmeans.labels_
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=cluster_data, x='Present_Price', y='Selling_Price',
            hue='cluster', palette='Set1', ax=ax
        )
        st.pyplot(fig)
    except Exception as ex:
        st.info(f"KMeans cluster plot error: {ex}")

with tabs[4]:
    st.header("🤖 Feature Importance")
    try:
        if hasattr(model, "feature_importances_"):
            possible_features = [c for c in df.columns if c != 'Selling_Price']
            imp = model.feature_importances_
            if len(imp) != len(possible_features):
                min_len = min(len(imp), len(possible_features))
                feature_names, imp = possible_features[:min_len], imp[:min_len]
            else:
                feature_names = possible_features
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values('Importance', ascending=False)
            st.bar_chart(imp_df.set_index('Feature'))
            st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Feature importances only for tree-based models.")
    except Exception as e:
        st.error(f"Feature importance error: {e}")

with tabs[5]:
    st.header("🧪 Model Evaluation")
    try:
        X = df.drop(columns=["Selling_Price"])
        y = df['Selling_Price']
        pred = model.predict(X)
        st.metric("R²", r2_score(y, pred))
        st.metric("MAE", mean_absolute_error(y, pred))
        st.metric("RMSE", mean_squared_error(y, pred, squared=False))

        fig, ax = plt.subplots()
        ax.scatter(y, pred, alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title("Actual vs. Predicted Selling Price")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        residuals = y - pred
        sns.histplot(residuals, bins=30, kde=True, color="slateblue", ax=ax)
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Evaluation error: check that model input matches training features! Details: {e}")

with tabs[6]:
    st.header("🔮 Predict Selling Price")
    with st.form(key='predict_form'):
        col1, col2 = st.columns(2)
        car_name = col1.selectbox('Car Name', sorted(df['Car_Name'].unique()))
        year = col2.number_input('Year', min_value=1990, max_value=2025, value=2018)
        present_price = col1.number_input('Present Price (Lakh)', min_value=0.0, value=5.0)
        driven_kms = col2.number_input('Driven Kms', min_value=0, value=10000)
        fuel_type = col1.selectbox('Fuel Type', sorted(df['Fuel_Type'].unique()))
        selling_type = col2.selectbox('Seller Type', sorted(df['Selling_type'].unique()))
        transmission = col1.selectbox('Transmission', sorted(df['Transmission'].unique()))
        owner = col2.selectbox('Owner', sorted(df['Owner'].unique()))
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
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakh")
        except Exception as ex:
            st.error(f"Prediction error: {ex}")
