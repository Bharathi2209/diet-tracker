from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os, joblib, matplotlib
matplotlib.use("Agg")            # disable GUI backend
import matplotlib.pyplot as plt
import pandas as pd

# ───────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────
DB_PATH = "sqlite:///diet_tracker.db"
MODEL_PATH = "model.pkl"

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DB_PATH
db = SQLAlchemy(app)
with app.app_context():
    db.create_all()


# ───────────────────────────────────────────────────────────────
# DB Model
# ───────────────────────────────────────────────────────────────
class Record(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(50))
    age         = db.Column(db.Integer)
    gender      = db.Column(db.String(6))
    height      = db.Column(db.Float)
    weight      = db.Column(db.Float)
    goal        = db.Column(db.String(10))
    diabetes    = db.Column(db.Boolean)
    bmi         = db.Column(db.Float)
    calories    = db.Column(db.Integer)
    water       = db.Column(db.Float)
    advice      = db.Column(db.String(300))
    predicted   = db.Column(db.Float)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()


# ───────────────────────────────────────────────────────────────
# ML model – auto‑train if missing
# ───────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    from train_model import train_and_save_model
    train_and_save_model()

model = joblib.load(MODEL_PATH)

# ───────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────
def bmi_value(h_cm, w_kg):
    return round(w_kg / ((h_cm / 100) ** 2), 1)

def bmi_status(bmi):
    if bmi < 18.5:   return "Underweight"
    if bmi < 25:     return "Normal"
    if bmi < 30:     return "Overweight"
    return "Obese"

def advice_text(bmi, diabetes):
    if diabetes:
        if bmi < 18.5:
            return ("You're underweight and diabetic. Add nutritious "
                    "calories (nuts, paneer, pulses) and monitor glucose.")
        if bmi < 25:
            return ("Weight is normal. Emphasise low‑GI carbs, split meals, "
                    "and track sugar regularly.")
        return ("Over‑weight with diabetes – reduce refined carbs, "
                "increase fiber & daily brisk walk 30 min.")
    else:
        if bmi < 18.5:
            return "Under‑weight. Increase healthy calories & strength training."
        if bmi < 25:
            return "Healthy weight. Maintain balanced diet & exercise."
        return "Over‑weight. Reduce calories and increase activity."

LOW_GI_LIST = [
    "Oats / Barley porridge", "Whole‑wheat roti", "Brown rice / Quinoa",
    "Leafy greens & beans", "Lentils / Chickpeas", "Apple / Berries (small serving)"
]

# ───────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ---------- Collect form data ----------
        name     = request.form["name"].strip().title()
        age      = int(request.form["age"])
        gender   = request.form["gender"].lower()
        height   = float(request.form["height"])
        weight   = float(request.form["weight"])
        goal     = request.form["goal"].lower()
        diabetes = request.form["diabetes"].lower() == "yes"

        # ---------- Calculations ----------
        bmi   = bmi_value(height, weight)
        bmis  = bmi_status(bmi)
        cal   = round(10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161))
        water = round(weight * 0.033, 2)
        adv   = advice_text(bmi, diabetes)

        gender_num = 1 if gender == "male" else 0
        goal_num   = {"lose":0, "maintain":1, "gain":2}.get(goal,1)
        pred       = round(model.predict([[age, height, weight, gender_num, goal_num]])[0], 2)

        # ---------- Save to DB ----------
        rec = Record(name=name, age=age, gender=gender, height=height,
                     weight=weight, goal=goal, diabetes=diabetes, bmi=bmi,
                     calories=cal, water=water, advice=adv, predicted=pred)
        db.session.add(rec)
        db.session.commit()

        # ---------- Pass to template ----------
        return render_template(
            "index.html",
            result={
                "name": name, "bmi": bmi, "bmi_status": bmis,
                "calories": cal, "water": water, "advice": adv,
                "predicted": pred, "diabetes": diabetes,
                "food": LOW_GI_LIST if diabetes else []
            }
        )

    return render_template("index.html")

@app.route("/history")
def history():
    data = Record.query.order_by(Record.created_at.desc()).all()
    if data:
        df = pd.DataFrame([r.__dict__ for r in data]).drop(columns="_sa_instance_state")
        # generate/update charts
        plt.figure(figsize=(10,4))
        df.groupby("name")["weight"].last().plot(kind="bar", color="tab:orange")
        plt.ylabel("Weight (kg)"); plt.tight_layout()
        plt.savefig("static/weight_chart_light.png"); plt.close()

        plt.style.use("dark_background")
        plt.figure(figsize=(10,4))
        df.groupby("name")["weight"].last().plot(kind="bar", color="skyblue")
        plt.ylabel("Weight (kg)"); plt.tight_layout()
        plt.savefig("static/weight_chart_dark.png"); plt.close()

    return render_template("history.html", records=data)

# ───────────────────────────────────────────────────────────────
# Run – use Render‑provided PORT if available
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
