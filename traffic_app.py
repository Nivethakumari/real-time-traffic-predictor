import streamlit as st

# ✅ This must be the FIRST Streamlit command!
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

import pickle
import pandas as pd
import datetime

# Load the model and label encoder
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("This app predicts the traffic level based on current time and location.")

# Junction Mapping
junction_map = {
    "Hebbal Junction": 1,
    "Nagawara Junction": 2,
    "Silk Board": 3,
    "Electronic City": 4
}

# User selects a readable junction name
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
junction = junction_map[junction_name]

# Get current date and time values
now = datetime.datetime.now()
hour = now.hour
day = now.day
weekday = now.strftime("%A")  # e.g., 'Sunday'
month = now.month
is_weekend = 1 if weekday in ['Saturday', 'Sunday'] else 0

# Display the auto-filled info
st.write(f"🕒 Current Hour: {hour}")
st.write(f"📅 Date: {now.strftime('%d %B %Y')} ({weekday})")

# Predict
if st.button("Predict Traffic Level"):
    input_data = pd.DataFrame([{
        "Junction": junction,
        "Hour": hour,
        "Day": day,
        "Weekday": now.weekday(),  # model needs number 0–6
        "Month": month,
        "IsWeekend": is_weekend
    }])

    prediction = model.predict(input_data)
    traffic_level = le.inverse_transform(prediction)[0]
    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

# Footer
st.markdown("---")
st.markdown("Created by **Nivethakumari**")
