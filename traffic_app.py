import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# This must be the first Streamlit command
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

# Junction Mapping
junction_map = {
    "Hebbal Junction": 1,
    "Nagawara Junction": 2,
    "Silk Board": 3,
    "Electronic City": 4
}

# Title and description
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Predict traffic levels for any date, time, and location in Bengaluru.")

# Select junction
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
junction = junction_map[junction_name]

# User inputs: date and hour
selected_date = st.date_input("Select Date")
selected_hour = st.slider("Select Hour (0-23)", 0, 23, datetime.now().hour)

# Extract base features
day = selected_date.day
month = selected_date.month
weekday_name = selected_date.strftime("%A")
weekday_num = selected_date.weekday()
is_weekend = 1 if weekday_name in ["Saturday", "Sunday"] else 0

# Additional engineered features
def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 0  # Morning
    elif 12 <= hour < 17:
        return 1  # Afternoon
    elif 17 <= hour < 21:
        return 2  # Evening
    else:
        return 3  # Night

part_of_day = get_part_of_day(selected_hour)
is_month_start = 1 if day <= 5 else 0
is_month_end = 1 if day >= 26 else 0
is_weekend_morning = 1 if is_weekend and selected_hour < 12 else 0
quarter = (month - 1) // 3 + 1

# Show selected info
formatted_date = selected_date.strftime("%d %B %Y")
st.markdown(f"📅 **Selected Date:** {formatted_date} ({weekday_name})")
st.markdown(f"🕒 **Selected Hour:** {selected_hour}:00")

# Predict button
if st.button("Predict Traffic Level"):
    input_data = pd.DataFrame([{
        "Junction": junction,
        "Hour": selected_hour,
        "Day": day,
        "Weekday": weekday_num,
        "Month": month,
        "IsWeekend": is_weekend,
        "PartOfDay": part_of_day,
        "IsMonthStart": is_month_start,
        "IsMonthEnd": is_month_end,
        "IsWeekendMorning": is_weekend_morning,
        "Quarter": quarter
    }])

    prediction = model.predict(input_data)
    traffic_level = le.inverse_transform(prediction)[0]

    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari**")
