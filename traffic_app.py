import streamlit as st
import datetime
import joblib

# Load the trained model (make sure this path is correct)
model = joblib.load("traffic_model.pkl")

# App title
st.markdown("<h1 style='text-align: center;'>🚦 Real-Time Traffic Level Predictor</h1>", unsafe_allow_html=True)
st.write("Use this app to predict traffic level based on time and location!")

# Select junction
junction = st.selectbox("Select Junction", [1, 2, 3, 4])

# Get current date and time
now = datetime.datetime.now()
day = now.day
hour = now.hour
month = now.month
weekday_index = now.weekday()  # Monday = 0, Sunday = 6
weekday_name = now.strftime("%A")
is_weekend = "Yes" if weekday_index >= 5 else "No"

# Show current info
st.write(f"📅 **Date:** {now.date()} | 🕒 **Time:** {now.strftime('%H:%M')}")
st.write(f"📍 **Junction:** {junction} | 📅 **Day:** {day}, **Weekday:** {weekday_name}, **Month:** {month}, **Weekend:** {is_weekend}")

# Predict button
if st.button("Predict Traffic Level"):
    # Prepare the features for prediction
    features = [[junction, hour, weekday_index, month]]
    prediction = model.predict(features)

    # Decode traffic level (if your model returns encoded labels)
    traffic_level = prediction[0].capitalize()  # Assuming model returns 'low', 'medium', or 'high'

    # Show result
    st.success(f"✅ Predicted Traffic Level: **{traffic_level}**")
