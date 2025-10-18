# -------------------------------
# 🏠 Bangalore Home Price Prediction App
# -------------------------------

import streamlit as st
import pickle
import json
import numpy as np
import os

# -------------------------------
# Load model and columns
# -------------------------------
# Adjusted paths for GitHub/Streamlit
base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "notebooks", "banglore_home_prices_model.pickle")
columns_path = os.path.join(base_path, "notebooks", "columns.json")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load columns info
with open(columns_path, "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract locations
locations = data_columns[3:]  # first 3: sqft, bath, bhk
locations.sort()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🏠 Bangalore Home Price Prediction", page_icon="🏙️")

st.title("🏠 Bangalore Home Price Prediction App")
st.markdown("Estimate **house prices in Bangalore** based on area, bedrooms, bathrooms, and location.")

# Collect user input
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, step=50, value=1000)
    bath = st.slider("Number of Bathrooms", 1, 10, 2)

with col2:
    bhk = st.slider("Number of BHK", 1, 10, 2)
    location = st.selectbox("Select Location", locations)

# -------------------------------
# Prediction Function
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
# Predict Button
# -------------------------------
if st.button("Predict Price"):
    result = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {result} Lakhs")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Made with ❤️ by Batman using Streamlit & scikit-learn")

