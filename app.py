from flask import Flask, render_template, request
import joblib
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
model = joblib.load("model.pkl")

def calculate_bmi(height, weight):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 1)

def get_bmi_status(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_goal_advice(goal, bmi):
    if bmi < 18.5:
        return "You're underweight. Increase healthy calories and consult a dietitian."
    elif 18.5 <= bmi <= 24.9:
        return "You have a healthy weight. Keep up your balanced habits!"
    elif bmi >= 25:
        return "You're overweight. Focus on reducing calories and staying active."
    return "Maintain a healthy lifestyle."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            name = request.form["name"]
            age = int(request.form["age"])
            gender = request.form["gender"].lower()
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            goal = request.form["goal"].lower()

            if age <= 0 or height <= 0 or weight <= 0:
                raise ValueError("All values must be positive.")

            bmi = calculate_bmi(height, weight)
            bmi_status = get_bmi_status(bmi)
            calories = round(10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161))
            water = round(weight * 0.033, 2)
            advice = get_goal_advice(goal, bmi)

            gender_num = 1 if gender == "male" else 0
            goal_map = {"lose": 0, "maintain": 1, "gain": 2}
            goal_num = goal_map.get(goal, 1)

            features = [[age, height, weight, gender_num, goal_num]]
            predicted_weight = round(model.predict(features)[0], 2)

            # Save to CSV
            file_exists = os.path.isfile("history.csv")
            with open("history.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Name", "Age", "Gender", "Height", "Weight", "Goal", "BMI", "Calories", "Water", "Advice", "Predicted_Weight"])
                writer.writerow([name, age, gender, height, weight, goal, bmi, calories, water, advice, predicted_weight])

            result = {
                "name": name,
                "bmi": bmi,
                "bmi_status": bmi_status,
                "calories": calories,
                "water": water,
                "advice": advice,
                "predicted_weight": predicted_weight
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

        # Light mode chart
        plt.figure(figsize=(10, 5))
        df.groupby("Name")["Weight"].last().plot(kind='bar', color='tomato')
        plt.title("Latest Recorded Weight by User", color='black')
        plt.ylabel("Weight (kg)", color='black')
        plt.xticks(color='black', rotation=45)
        plt.yticks(color='black')
        plt.tight_layout()
        plt.savefig("static/weight_chart_light.png")
        plt.close()

        # Dark mode chart
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 5))
        df.groupby("Name")["Weight"].last().plot(kind='bar', color='skyblue')
        plt.title("Latest Recorded Weight by User", color='white')
        plt.ylabel("Weight (kg)", color='white')
        plt.xticks(color='white', rotation=45)
        plt.yticks(color='white')
        plt.tight_layout()
        plt.savefig("static/weight_chart_dark.png")
        plt.close()

    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
