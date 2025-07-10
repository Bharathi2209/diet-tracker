from flask import Flask, render_template, request, send_file
from sklearn.linear_model import LinearRegression
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Dummy training data
model = LinearRegression()
X = [
    [25, 170, 60, 0, 1],
    [30, 160, 70, 1, 0],
    [22, 180, 75, 2, 1],
    [40, 165, 68, 1, 0],
    [28, 172, 62, 0, 1]
]
y = [65, 70, 78, 68, 64]
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        name = request.form["name"]
        age = int(request.form["age"])
        gender = request.form["gender"]
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        goal = request.form["goal"]

        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)
        if bmi < 18.5:
            bmi_status = "Underweight"
        elif 18.5 <= bmi < 25:
            bmi_status = "Normal"
        elif 25 <= bmi < 30:
            bmi_status = "Overweight"
        else:
            bmi_status = "Obese"

        if gender == "male":
            calories = round(10 * weight + 6.25 * height - 5 * age + 5)
        else:
            calories = round(10 * weight + 6.25 * height - 5 * age - 161)

        if goal == "lose":
            calories -= 500
            advice = "Focus on a calorie deficit with high protein intake."
            goal_code = 0
        elif goal == "maintain":
            advice = "Maintain balanced diet and regular activity."
            goal_code = 1
        else:
            calories += 300
            advice = "Add calorie surplus with strength training."
            goal_code = 2

        water = round(weight * 0.033, 2)
        gender_code = 1 if gender == "male" else 0
        predicted_weight = round(model.predict([[age, height, weight, goal_code, gender_code]])[0], 2)

        result = {
            "name": name,
            "bmi": bmi,
            "bmi_status": bmi_status,
            "calories": calories,
            "water": water,
            "advice": advice,
            "predicted_weight": predicted_weight
        }

        file_exists = os.path.isfile("history.csv")
        with open("history.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Name", "Age", "Gender", "Height", "Weight", "Goal", "BMI", "Calories", "Water", "Advice", "Predicted_Weight"])
            writer.writerow([name, age, gender, height, weight, goal, bmi, calories, water, advice, predicted_weight])

    return render_template("index.html", result=result)

@app.route("/history")
def history():
    records = []
    header = []
    if os.path.isfile("history.csv"):
        with open("history.csv", newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            records = list(reader)
    return render_template("history.html", header=header, records=records)

@app.route("/chart")
def chart():
    if not os.path.isfile("history.csv"):
        return "No data available"

    names, weights, predicted, calories, bmis = [], [], [], [], []

    with open("history.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            names.append(row[0])
            weights.append(float(row[4]))
            predicted.append(float(row[10]))
            calories.append(float(row[7]))
            bmis.append(float(row[6]))

    # Weight Bar Chart
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, weights, width, label='Actual')
    plt.bar(x + width/2, predicted, width, label='Predicted')
    plt.xticks(x, names, rotation=30)
    plt.ylabel('Weight (kg)')
    plt.title('Actual vs Predicted Weight')
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/chart_weight.png")
    plt.close()

    # Calorie Bar Chart
    plt.figure(figsize=(10, 5))
    plt.bar(names, calories, color='orange')
    plt.xticks(rotation=30)
    plt.ylabel('Calories')
    plt.title('Recommended Daily Calories')
    plt.tight_layout()
    plt.savefig("static/chart_calories.png")
    plt.close()

    # BMI Bar Chart
    plt.figure(figsize=(10, 5))
    plt.bar(names, bmis, color='green')
    plt.xticks(rotation=30)
    plt.ylabel('BMI')
    plt.title('BMI Distribution')
    plt.tight_layout()
    plt.savefig("static/chart_bmi.png")
    plt.close()

    return "Bar charts generated."

if __name__ == "__main__":
    app.run(debug=True)
