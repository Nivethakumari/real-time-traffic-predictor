import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load the model and label encoder
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# Page layout
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")
st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Use this app to predict traffic level based on time and location!")

# Input fields
junction = st.selectbox("Select Junction", [1, 2, 3, 4])
now = datetime.now()
hour = st.slider("Hour", 0, 23, now.hour)
day = now.day
weekday = now.weekday()
month = now.month
is_weekend = 1 if weekday in [5, 6] else 0

# Display current time context
st.write(f"📅 **Date:** {now.date()} | 🕒 **Time:** {hour}:00")
st.write(f"🛣️ **Junction:** {junction} | 🗓️ **Day:** {day}, **Weekday:** {weekday}, **Month:** {month}, **Weekend:** {'Yes' if is_weekend else 'No'}")

# Predict button
if st.button("Predict Traffic Level"):
    input_df = pd.DataFrame([[junction, hour, day, weekday, month, is_weekend]],
                            columns=["Junction", "Hour", "Day", "Weekday", "Month", "IsWeekend"])
    prediction_encoded = model.predict(input_df)[0]
    prediction = le.inverse_transform([prediction_encoded])[0]
    st.success(f"✅ Predicted Traffic Level: **{prediction}**")
