import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Set page configuration
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

# User Inputs
junction_name = st.selectbox("Select Junction", list(junction_map.keys()))
junction = junction_map[junction_name]

selected_date = st.date_input("Select Date")
selected_hour = st.slider("Select Hour (0-23)", 0, 23, datetime.now().hour)

# Feature Engineering (only those used in training)
day = selected_date.day
month = selected_date.month
weekday_name = selected_date.strftime("%A")
weekday_num = selected_date.weekday()
is_weekend = 1 if weekday_name in ["Saturday", "Sunday"] else 0

# Display selected info
formatted_date = selected_date.strftime("%d %B %Y")
st.markdown(f"📅 **Selected Date:** {formatted_date} ({weekday_name})")
st.markdown(f"🕒 **Selected Hour:** {selected_hour}:00")

# Debug checkbox
debug = st.checkbox("Show model input data")

# Predict button
if st.button("Predict Traffic Level"):
    input_data = pd.DataFrame([{
        "Junction": junction,
        "Hour": selected_hour,
        "Day": day,
        "Weekday": weekday_num,
        "Month": month,
        "IsWeekend": is_weekend
    }])

    if debug:
        st.subheader("🧪 Model Input Data")
        st.write(input_data)
        st.write("🧩 Expected Feature Names by Model:")
        st.write(model.get_booster().feature_names)
        st.write("🧪 Input Data Columns:")
        st.write(input_data.columns.tolist())

    # Prediction
    prediction = model.predict(input_data)
    traffic_level = le.inverse_transform(prediction)[0]

    # Output
    st.success(f"🚗 Predicted Traffic Level: **{traffic_level}**")

    # Additional Info for Specific Junctions
    if junction_name == "Hebbal Junction":
        st.markdown("🔍 Note: Predictions for **Hebbal** are made relative to its usual traffic levels. Even low vehicle counts may result in 'High' labels due to local patterns.")
    elif junction_name == "Nagawara Junction":
        st.markdown("🔍 Note: **Nagawara** tends to show 'Medium' traffic predictions due to its traffic distribution.")
    elif junction_name == "Electronic City":
        st.markdown("🔍 Note: **Electronic City** usually has lighter traffic, so predictions may skew towards 'Low'.")

# Footer
st.markdown("---")
st.markdown("👩‍💻 Created by **Nivethakumari & Dharshini Shree**")
