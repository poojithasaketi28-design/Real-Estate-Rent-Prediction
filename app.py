import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# LOAD MODEL & SCALER
# =============================
try:
    model = pickle.load(open("real_estate_rf_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'real_estate_rf_model.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Real Estate Rent Prediction", layout="centered")

st.title("🏠 Real Estate Rent Prediction")
st.write("Enter property details to predict estimated Rent / Price")

# =============================
# USER INPUT SECTION
# (Change fields according to your dataset)
# =============================

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Number of Bedrooms", 1, 10, 2)
    bathrooms = st.number_input("Number of Bathrooms", 1, 10, 2)

with col2:
    area = st.number_input("Total Area (sqft)", 100, 10000, 1200)

# Convert categorical to numeric
# No categorical variables

# =============================
# Create DataFrame
# =============================
input_data = pd.DataFrame({
    "Bedroom": [bedrooms],
    "Bathroom": [bathrooms],
    "sqft": [area]
})

st.write("### Input Data Preview")
st.dataframe(input_data)

# =============================
# Prediction Button
# =============================
if st.button("Predict Rent / Price"):
    try:
        # The model expects the data as is (no scaling in training)
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Rent / Price: ₹ {round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}. Please check your inputs or model compatibility.")