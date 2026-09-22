import streamlit as st
import pandas as pd
import joblib
import base64

def play_sound(file_path):
    with open(file_path, "rb") as f:
        audio = f.read()

    audio_base64 = base64.b64encode(audio).decode()

    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)

# Load the saved model
model_data = joblib.load("student_model.pkl")

model = model_data["model"]
columns = model_data["columns"]


# Page title
st.title("🎓 Student Performance Predictor")

st.write("Enter the student's information below to predict the result.")


# User inputs
study_time = st.number_input(
    "Study Time (hours per week)",
    min_value=0,
    max_value=50,
    value=10
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=60
)

assignments = st.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=20,
    value=10
)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5
)


# Prediction
if st.button("Predict Performance"):

    sample = pd.DataFrame([{
        "study_time": study_time,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments": assignments,
        "absences": absences
    }])

    # Make sure columns are in the same order as training
    sample = sample.reindex(columns=columns)

    result = model.predict(sample)[0]

    if result == 1:
        st.success("✅ Prediction: PASS")
        st.balloons()
        play_sound("sounds/faahh.mp3")
    else:
        st.error("❌ Prediction: FAIL")
        play_sound("sounds/sad.mp3")