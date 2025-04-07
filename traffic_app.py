import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import pytz

# Load model and label encoder
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Junction mapping
junctions = {
    'Hebbal Junction': 0,
    'KR Puram Junction': 1,
    'Nayandahalli Junction': 2,
    'Nagawara Junction': 3,
    'Konnur High Road': 4
}

# Helper function to extract features
def extract_features(junction, date, hour):
    date = pd.to_datetime(date)
    day = date.day
    month = date.month
    weekday_num = date.weekday()
    quarter = date.quarter
    is_month_start = int(date.is_month_start)
    is_month_end = int(date.is_month_end)
    is_weekend = int(weekday_num >= 5)
    is_weekend_morning = int(is_weekend and 6 <= hour <= 9)

    if 5 <= hour < 12:
        part_of_day = "Morning"
    elif 12 <= hour < 17:
        part_of_day = "Afternoon"
    elif 17 <= hour < 21:
        part_of_day = "Evening"
    else:
        part_of_day = "Night"

    return {
        "Junction": junctions[junction],
        "Hour": hour,
        "Day": day,
        "Weekday": weekday_num,
        "Month": month,
        "IsWeekend": is_weekend,
        "PartOfDay": part_of_day,
        "IsMonthStart": is_month_start,
        "IsMonthEnd": is_month_end,
        "Quarter": quarter,
        "IsWeekendMorning": is_weekend_morning
    }

# Streamlit page setup
st.set_page_config(page_title="Real-Time Traffic Predictor", page_icon="🚦", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>🚦 Real-Time Traffic Level Predictor</h1>", unsafe_allow_html=True)
st.markdown("#### Predict traffic levels for any date, time, and location in Bengaluru.")

# Input selections
junction = st.selectbox("Select Junction", list(junctions.keys()))

# Get current IST datetime
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)

date = st.date_input("Select Date", now.date())
hour = st.slider("Select Hour (0–23)", min_value=0, max_value=23, value=now.hour)

# Show selected date/time
weekday_name = date.strftime("%A")
formatted_date = date.strftime("%d %B %Y")
st.markdown(f"📅 **Selected Date:** {formatted_date} ({weekday_name})")
st.markdown(f"🕒 **Selected Hour:** {hour:02d}:00")

# Prediction form
with st.form("predict_form"):
    submitted = st.form_submit_button("Predict Traffic Level")
    show_debug = st.checkbox("Show model input data (debug mode)")

    if submitted:
        features = extract_features(junction, date, hour)
        input_data = pd.DataFrame([features])

        # Reorder to match model
        input_data.columns = model.get_booster().feature_names

        if show_debug:
            st.write("🔍 Input Features Sent to Model:", input_data)
            st.write("📌 Model Expected Features:", model.get_booster().feature_names)

        prediction = model.predict(input_data)
        traffic_level = le.inverse_transform(prediction)[0]

        st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

# Footer
st.markdown("<br><hr><center>Created by Nivethakumari</center>", unsafe_allow_html=True)
