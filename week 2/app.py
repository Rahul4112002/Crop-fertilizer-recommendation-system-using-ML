import streamlit as st
import pandas as pd
import pickle

# Load models
crop_model_path = "crop_recommendation_model.pkl"
fertilizer_model_path = "fertilizer_recommendation_model.pkl"

with open(crop_model_path, 'rb') as file:
    crop_model = pickle.load(file)

with open(fertilizer_model_path, 'rb') as file:
    fertilizer_model = pickle.load(file)

# App title
st.title("🌱 Intelligent Crop & Fertilizer Recommendation System")
st.write("Enter your soil and environmental details to get personalized crop and fertilizer suggestions.")

# Sidebar inputs
st.sidebar.header("Input Parameters")
N = st.sidebar.slider("Nitrogen (N) level:", 0, 200, 50)
P = st.sidebar.slider("Phosphorus (P) level:", 0, 200, 50)
K = st.sidebar.slider("Potassium (K) level:", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C):", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%):", 0, 100, 50)
ph = st.sidebar.slider("Soil pH Level:", 0.0, 14.0, 7.0)
rainfall = st.sidebar.slider("Rainfall (mm):", 0, 300, 100)

# Crop type input
st.sidebar.subheader("Crop Type")
crop_type = st.sidebar.selectbox(
    "Select Crop Type:",
    ["cotton", "ground nuts", "maize", "millets", "oil seeds", "others"]
)

# One-hot encoding for crop type
crop_type_encoded = [0] * 6
crop_types = ["cotton", "ground nuts", "maize", "millets", "oil seeds", "others"]
if crop_type in crop_types:
    crop_type_encoded[crop_types.index(crop_type)] = 1

# Combine all inputs
features = [N, P, K, temperature, humidity, ph, rainfall] + crop_type_encoded
feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"] + [f"crop type_{ct}" for ct in crop_types]
input_data = pd.DataFrame([features], columns=feature_names)

# Crop Recommendation
st.subheader("🌾 Recommended Crop")
try:
    crop_prediction = crop_model.predict(input_data)[0]
    st.success(f"**{crop_prediction}**")
except ValueError as e:
    st.error(f"Error in crop recommendation: {e}")

# Fertilizer Recommendation
st.subheader("🧪 Recommended Fertilizer")
try:
    fertilizer_prediction = fertilizer_model.predict(input_data)[0]
    st.success(f"**{fertilizer_prediction}**")
except ValueError as e:
    st.error(f"Error in fertilizer recommendation: {e}")

# Footer
st.markdown("Made with ❤️ for farmers to enhance agricultural productivity.")
