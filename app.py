from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os, joblib, matplotlib
matplotlib.use("Agg")                # head‑less backend for server
import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DB_URI  = "sqlite:///diet_tracker.db"
MODEL_PKL = "model.pkl"

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
db = SQLAlchemy(app)

# ──────────────────────────────────────────────────────────────
# Database Model
# ──────────────────────────────────────────────────────────────
class Record(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    name      = db.Column(db.String(50))
    age       = db.Column(db.Integer)
    gender    = db.Column(db.String(6))
    height    = db.Column(db.Float)
    weight    = db.Column(db.Float)
    goal      = db.Column(db.String(10))
    diabetes  = db.Column(db.Boolean)
    bmi       = db.Column(db.Float)
    calories  = db.Column(db.Integer)
    water     = db.Column(db.Float)
    advice    = db.Column(db.String(300))
    predicted = db.Column(db.Float)
    created   = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ──────────────────────────────────────────────────────────────
# Auto‑train model if missing
# ──────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PKL):
    print("⚙️  No model.pkl found – training on server…")
    from train_model import train_and_save_model
    train_and_save_model()

model = joblib.load(MODEL_PKL)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def bmi_val(h_cm, w_kg):
    return round(w_kg / ((h_cm / 100) ** 2), 1)

def bmi_status(b):
    return ("Underweight" if b < 18.5 else
            "Normal"      if b < 25   else
            "Overweight"  if b < 30   else
            "Obese")

def advice_text(bmi, diabetic):
    if diabetic:
        if bmi < 18.5: return ("Underweight & diabetic – add nutritious "
                               "calories (nuts / pulses) & monitor glucose.")
        if bmi < 25:   return ("Weight normal – emphasise low‑GI carbs, "
                               "small frequent meals, regular sugar checks.")
        return ("Overweight & diabetic – cut refined carbs, "
                "increase fibre & daily 30‑min walk.")
    else:
        if bmi < 18.5: return "Under‑weight. Increase healthy calories & strength training."
        if bmi < 25:   return "Healthy weight. Maintain balanced diet & exercise."
        return "Over‑weight. Reduce calories and increase activity."

LOW_GI = ["Oats & barley", "Whole‑wheat roti", "Brown rice / Quinoa",
          "Leafy greens", "Lentils / Beans", "Apple / Berries"]

# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Form values
        name     = request.form["name"].strip().title()
        age      = int(request.form["age"])
        gender   = request.form["gender"].lower()
        height   = float(request.form["height"])
        weight   = float(request.form["weight"])
        goal     = request.form["goal"].lower()
        diabetes = request.form["diabetes"].lower() == "yes"

        bmi  = bmi_val(height, weight)
        bmis = bmi_status(bmi)
        cal  = round(10*weight + 6.25*height - 5*age + (5 if gender=="male" else -161))
        wat  = round(weight * 0.033, 2)
        adv  = advice_text(bmi, diabetes)

        gender_num = 1 if gender == "male" else 0
        goal_num   = {"lose":0, "maintain":1, "gain":2}.get(goal,1)
        pred = round(model.predict([[age, height, weight, gender_num, goal_num]])[0], 2)

        # Save to DB
        row = Record(name=name, age=age, gender=gender, height=height,
                     weight=weight, goal=goal, diabetes=diabetes, bmi=bmi,
                     calories=cal, water=wat, advice=adv, predicted=pred)
        db.session.add(row)
        db.session.commit()

        return render_template("index.html",
            result=dict(name=name, bmi=bmi, bmi_status=bmis,
                        calories=cal, water=wat, advice=adv,
                        predicted=pred, diabetes=diabetes,
                        foods=LOW_GI if diabetes else [])
        )

    return render_template("index.html")

@app.route("/history")
def history():
    rows = Record.query.order_by(Record.created.desc()).all()
    if rows:
        df = pd.DataFrame([r.__dict__ for r in rows]).drop(columns="_sa_instance_state")
        plt.figure(figsize=(10,4))
        df.groupby("name")["weight"].last().plot(kind="bar", color="tab:orange")
        plt.tight_layout(); plt.savefig("static/weight_chart_light.png"); plt.close()

        plt.style.use("dark_background")
        plt.figure(figsize=(10,4))
        df.groupby("name")["weight"].last().plot(kind="bar", color="skyblue")
        plt.tight_layout(); plt.savefig("static/weight_chart_dark.png"); plt.close()

    return render_template("history.html", records=rows)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
