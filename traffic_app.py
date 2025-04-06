import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import pytz

# ✅ Must be the first Streamlit command
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
now = datetime.now(ist)

# Extract features
hour = now.hour
day = now.day
month = now.month
year = now.year
weekday_str = now.strftime("%A")
weekday_num = now.weekday()
is_weekend = 1 if weekday_str in ['Saturday', 'Sunday'] else 0

# Junction name mapping
junction_map = {
    "Hebbal Junction": 1,
    "Nagawara Junction": 2,
    "Silk Board": 3,
    "Electronic City": 4
}

# App interface
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("This app predicts the traffic level based on the current time and junction location.")

# Display current datetime info
st.markdown(f"📅 **Date:** {day} {now.strftime('%B')} {year} ({weekday_str})")
st.markdown(f"🕒 **Current Hour (IST):** {hour}")

# Junction selection
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
junction = junction_map[junction_name]

# Predict button
if st.button("Predict Traffic Level"):
    input_df = pd.DataFrame([{
        "Junction": junction,
        "Hour": hour,
        "Day": day,
        "Weekday": weekday_num,
        "Month": month,
        "IsWeekend": is_weekend
    }])
    prediction = model.predict(input_df)
    traffic_level = le.inverse_transform(prediction)[0]
    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari**")
