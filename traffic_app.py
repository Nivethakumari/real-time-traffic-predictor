import streamlit as st
import pickle
import calendar
import datetime

# Set page config first
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

st.title("🚦 Real-Time Traffic Level Predictor")

# Input fields
junction = st.selectbox("Select Junction", [1, 2, 3, 4])
hour = st.slider("Hour of Day (24hr)", 0, 23, 12)

# Get day name from dropdown
day_name = st.selectbox("Day of the Week", list(calendar.day_name))
day = list(calendar.day_name).index(day_name)

# Get weekday and weekend info
weekday = day  # for compatibility
is_weekend = 1 if day in [5, 6] else 0  # 5 = Saturday, 6 = Sunday

month = st.slider("Month", 1, 12, datetime.datetime.now().month)

# Predict button
if st.button("Predict Traffic Level"):
    input_data = {
        "Junction": junction,
        "Hour": hour,
        "Day": day,
        "Weekday": weekday,
        "Month": month,
        "IsWeekend": is_weekend
    }

    prediction = model.predict([list(input_data.values())])[0]
    traffic_level = le.inverse_transform([prediction])[0]
    
    st.success(f"Predicted Traffic Level: **{traffic_level}**")
