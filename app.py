from flask import Flask, render_template, request
import os
import csv
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# 1) Auto‑train the model if model.pkl is not found (first deploy on Render)
# ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print("⚙️  model.pkl not found – training model on server…")
    # Try to import a function `train_and_save_model()` from train_model.py
    try:
        from train_model import train_and_save_model
        train_and_save_model()               # generates model.pkl
    except ImportError:
        # Fallback: run the script directly
        import subprocess, sys
        subprocess.check_call([sys.executable, "train_model.py"])

# Load the freshly‑created (or existing) model
model = joblib.load(MODEL_PATH)

# ────────────────────────────────────────────────────────────────────────────
# 2) Flask setup
# ────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

def calculate_bmi(height, weight):
    bmiv = weight / ((height / 100) ** 2)
    return round(bmiv, 1)

def get_bmi_status(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"

def get_bmi_advice(bmi):
    if bmi < 18.5:
        return "You're underweight. Increase nutritious calories and consider resistance training."
    if bmi < 25:
        return "Healthy weight! Maintain balanced diet and regular activity."
    return "You're overweight. Focus on calorie control and increased physical activity."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            name    = request.form["name"]
            age     = int(request.form["age"])
            gender  = request.form["gender"].lower()
            height  = float(request.form["height"])
            weight  = float(request.form["weight"])
            goal    = request.form["goal"].lower()

            if any(v <= 0 for v in (age, height, weight)):
                raise ValueError("Age, height, and weight must be positive numbers.")

            bmi        = calculate_bmi(height, weight)
            bmi_status = get_bmi_status(bmi)
            advice     = get_bmi_advice(bmi)
            calories   = round(10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161))
            water      = round(weight * 0.033, 2)

            gender_num = 1 if gender == "male" else 0
            goal_map   = {"lose": 0, "maintain": 1, "gain": 2}
            goal_num   = goal_map.get(goal, 1)

            predicted_weight = round(
                model.predict([[age, height, weight, gender_num, goal_num]])[0], 2
            )

            # Save history ----------------------------------------------------
            new_row = [name, age, gender, height, weight, goal, bmi,
                       calories, water, advice, predicted_weight]
            file_exists = os.path.isfile("history.csv")
            with open("history.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Name", "Age", "Gender", "Height", "Weight",
                                     "Goal", "BMI", "Calories", "Water",
                                     "Advice", "Predicted_Weight"])
                writer.writerow(new_row)

            result = {
                "name": name,
                "bmi": bmi,
                "bmi_status": bmi_status,
                "calories": calories,
                "water": water,
                "advice": advice,
                "predicted_weight": predicted_weight,
            }
            return render_template("index.html", result=result)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

@app.route("/history")
def history():
    records = []
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        records = df.values.tolist()

        # Light chart
        plt.figure(figsize=(10, 5))
        df.groupby("Name")["Weight"].last().plot(kind="bar", color="tomato")
        plt.ylabel("Weight (kg)", color="black")
        plt.title("Latest Recorded Weight by User", color="black")
        plt.xticks(rotation=45, color="black")
        plt.yticks(color="black")
        plt.tight_layout()
        plt.savefig("static/weight_chart_light.png")
        plt.close()

        # Dark chart
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 5))
        df.groupby("Name")["Weight"].last().plot(kind="bar", color="skyblue")
        plt.ylabel("Weight (kg)", color="white")
        plt.title("Latest Recorded Weight by User", color="white")
        plt.xticks(rotation=45, color="white")
        plt.yticks(color="white")
        plt.tight_layout()
        plt.savefig("static/weight_chart_dark.png")
        plt.close()

    return render_template("history.html", records=records)

# ────────────────────────────────────────────────────────────────────────────
# 3) Bind to Render‑provided PORT
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
