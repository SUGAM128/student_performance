from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model_data = joblib.load("student_model.pkl")

model = model_data["model"]
columns = model_data["columns"]

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        study_time = float(request.form["study_time"])
        attendance = float(request.form["attendance"])
        previous_score = float(request.form["previous_score"])
        assignments = float(request.form["assignments"])
        absences = float(request.form["absences"])

        sample = pd.DataFrame([{
            "study_time": study_time,
            "attendance": attendance,
            "previous_score": previous_score,
            "assignments": assignments,
            "absences": absences
        }])

        result = model.predict(sample)[0]

        if result == 1:
            prediction = "PASS"
        else:
            prediction = "FAIL"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)