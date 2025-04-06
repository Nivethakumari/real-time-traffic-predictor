import streamlit as st
import pickle
import datetime
import calendar

# Set the page config
st.set_page_config(page_title="Traffic Level Predictor", layout="centered")

# Load model and label encoder
@st.cache_data
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

st.title("🚦 Real-Time Traffic Level Predictor")
st.markdown("Predict traffic levels at a junction based on the time and date.")

# Select Junction
junction = st.selectbox("Select Junction", [1, 2, 3, 4])

# Select Hour
hour = st.slider("Select Hour (0 - 23)", 0, 23)

# Get current date info
now = datetime.datetime.now()
day = now.weekday()  # 0 = Monday, 6 = Sunday
day_name = calendar.day_name[day]
month = now.month
is_weekend = 1 if day >= 5 else 0  # Saturday = 5, Sunday = 6

st.markdown(f"**Today:** {day_name}, **Month:** {month}")

# Prepare input data
input_data = {
    "Junction": junction,
    "Hour": hour,
    "Day": day,
    "Weekday": day,
    "Month": month,
    "IsWeekend": is_weekend
}

# Predict
if st.button("Predict Traffic Level"):
    pred = model.predict([list(input_data.values())])
    traffic_level = le.inverse_transform(pred)[0]
    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")
