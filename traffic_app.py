import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import pytz

# ⬅️ Ensure this is the first Streamlit command
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

# Load model and label encoder
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# Get current IST date and time
ist = pytz.timezone('Asia/Kolkata')
current_datetime = datetime.now(ist)

# Extract relevant time features
current_hour = current_datetime.hour
current_day = current_datetime.day
current_weekday = current_datetime.strftime('%A')  # e.g., 'Sunday'
current_month = current_datetime.month
is_weekend = 1 if current_weekday in ["Saturday", "Sunday"] else 0

# Display information
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown(f"📅 **Date:** {current_day}/{current_month}")
st.markdown(f"📆 **Weekday:** {current_weekday}")
st.markdown(f"🕒 **Current Hour (IST):** {current_hour}")

# Let user select the Junction
junction = st.selectbox("Select Junction", [1, 2, 3, 4], format_func=lambda x: f"Junction {x}")

# Predict button
if st.button("Predict Traffic Level"):
    input_data = pd.DataFrame([{
        "Junction": junction,
        "Hour": current_hour,
        "Day": current_day,
        "Weekday": current_datetime.weekday(),  # Numeric format for model
        "Month": current_month,
        "IsWeekend": is_weekend
    }])

    prediction = model.predict(input_data)
    traffic_level = le.inverse_transform(prediction)[0]

    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari**")
