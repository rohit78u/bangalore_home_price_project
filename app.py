# -------------------------------------------
# 🏠 Bangalore Home Price Prediction & Dashboard App
# -------------------------------------------

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.express as px
import os

# -------------------------------
# Setup and Configuration
# -------------------------------
st.set_page_config(page_title="🏙️ Bangalore Home Price App", page_icon="🏠", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Predict Price", "📊 Data Dashboard"])

# Base path for file access
base_path = os.path.dirname(__file__)

# -------------------------------
# Load Model and Column Data
# -------------------------------
model_path = os.path.join(base_path, "notebooks", "banglore_home_prices_model.pickle")
columns_path = os.path.join(base_path, "notebooks", "columns.json")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(columns_path, "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract locations (skip first 3 columns)
locations = data_columns[3:]
locations.sort()


# -------------------------------
# Price Prediction Function
# -------------------------------
def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)


# -------------------------------
# 🏠 PAGE 1: Prediction Interface
# -------------------------------
if page == "🏠 Predict Price":
    st.title("🏠 Bangalore Home Price Prediction App")
    st.markdown("Estimate **house prices in Bangalore** based on area, bedrooms, bathrooms, and location.")

    # Input columns
    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, step=50, value=1000)
        bath = st.slider("Number of Bathrooms", 1, 10, 2)

    with col2:
        bhk = st.slider("Number of BHK", 1, 10, 2)
        location = st.selectbox("Select Location", locations)

    # Predict Button
    if st.button("Predict Price"):
        result = predict_price(location, sqft, bath, bhk)
        st.success(f"💰 Estimated Price: ₹ {result} Lakhs")

    # Footer
    st.markdown("---")
    st.caption("Made with ❤️ by Rohit, Prasad, Praneeth & Kushal using Streamlit & Scikit-learn")


# -------------------------------
# 📊 PAGE 2: Data Dashboard
# -------------------------------
# -------------------------------
# 📊 PAGE 2: Data Dashboard
# -------------------------------
elif page == "📊 Data Dashboard":
    st.title("📊 Bangalore Real Estate Data Insights")

    # Load dataset
    data_path = os.path.join(base_path, "data", "Bengaluru_House_Data.csv")
    if not os.path.exists(data_path):
        st.error("❌ Data file not found. Please ensure 'Bengaluru_House_Data.csv' is in the 'data' folder.")
    else:
        df = pd.read_csv(data_path)

        # ✅ Create bhk column from 'size' (e.g., "2 BHK" → 2)
        if 'size' in df.columns:
            df['bhk'] = df['size'].apply(
                lambda x: int(str(x).split(' ')[0]) if isinstance(x, str) and str(x)[0].isdigit() else None
            )

        # ✅ Clean data
        df = df.dropna(subset=["price", "total_sqft", "bhk", "bath", "location"])

        # --- Convert total_sqft safely to numeric ---
        def convert_sqft_to_num(x):
            try:
                if isinstance(x, str):
                    x = x.replace(',', '').strip()
                    # Handle ranges like "2100 - 2850"
                    if '-' in x:
                        tokens = x.split('-')
                        return (float(tokens[0]) + float(tokens[1])) / 2
                    # Handle "34.46Sq. Meter" or similar
                    val = ''.join([ch for ch in x if ch.isdigit() or ch == '.'])
                    return float(val) if val else None
                return float(x)
            except:
                return None

        df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
        df = df.dropna(subset=["total_sqft"])

        # ✅ Create Dashboard Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Summary Overview",
            "📍 Location Analysis",
            "🧱 BHK Trends",
            "💰 Sqft vs Price"
        ])

        # -------- TAB 1 --------
        with tab1:
            st.subheader("📈 Overall Market Summary")

            total_listings = len(df)
            avg_price = round(df["price"].mean(), 2)
            avg_sqft = round(df["total_sqft"].mean(), 2)
            avg_bhk = round(df["bhk"].mean(), 1)
            avg_bath = round(df["bath"].mean(), 1)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🏠 Total Listings", f"{total_listings:,}")
            col2.metric("💰 Avg Price (Lakhs)", f"{avg_price}")
            col3.metric("📐 Avg Area (sqft)", f"{avg_sqft}")
            col4.metric("🛏️ Avg BHK", f"{avg_bhk}")
            col5.metric("🚿 Avg Bathrooms", f"{avg_bath}")

            st.markdown("#### 📊 Price Distribution Overview")
            fig = px.histogram(df, x="price", nbins=50, color_discrete_sequence=["#4C9AFF"])
            st.plotly_chart(fig, use_container_width=True)

        # -------- TAB 2 --------
        with tab2:
            st.subheader("Top 20 Locations by Average Price")
            avg_price = df.groupby("location")["price"].mean().reset_index().sort_values("price", ascending=False)
            fig1 = px.bar(avg_price.head(20), x="location", y="price", color="price",
                          title="🏙️ Average Price per Location")
            st.plotly_chart(fig1, use_container_width=True)

        # -------- TAB 3 --------
        with tab3:
            st.subheader("Price Distribution Across BHK Types")
            fig2 = px.box(df, x="bhk", y="price", color="bhk", title="🏘️ BHK vs Price Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        # -------- TAB 4 --------
        with tab4:
            st.subheader("Sqft vs Price (Interactive Scatter Plot)")
            selected_loc = st.selectbox("Select Location to Explore", sorted(df["location"].unique()))
            loc_data = df[df["location"] == selected_loc]
            fig3 = px.scatter(loc_data, x="total_sqft", y="price", color="bhk", size="bath",
                              title=f"💰 Price vs Sqft in {selected_loc}", hover_data=["bhk", "bath"])
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.caption("📈 Dashboard generated with Plotly & Streamlit")
