"""
train_model.py
──────────────
Trains a simple Linear Regression model on a small sample
dataset and saves it to 'model.pkl'.

The main entry point `train_and_save_model()` is imported
by app.py when model.pkl is missing on the server.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train_and_save_model():
    # ── Sample dataset ─────────────────────────────────────────
    data = {
        "age":    [20, 25, 30, 22, 28, 35, 40, 26, 32, 45],
        "height": [160, 170, 175, 165, 180, 168, 172, 174, 169, 178],
        "weight": [50, 65, 70, 55, 80, 60, 75, 68, 66, 85],
        "gender": ["Female", "MALE", "Male", "FEMALE", "male",
                   "female", "MALE", "Male", "female", "MALE"],
        "goal":   ["Gain", "lose", "Maintain", "GAIN", "Lose",
                   "Maintain", "LOSE", "gain", "maintain", "GAIN"],
        "future_weight": [55, 60, 70, 60, 75, 60, 70, 73, 66, 90],
    }
    df = pd.DataFrame(data)

    # preprocess
    df = df[df["age"] > 0]
    df["gender"] = df["gender"].str.lower().map({"male": 1, "female": 0})
    df["goal"]   = df["goal"].str.lower().map({"lose": 0, "maintain": 1, "gain": 2})
    df.dropna(inplace=True)

    X = df[["age", "height", "weight", "gender", "goal"]]
    y = df["future_weight"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("✅  model.pkl generated on server")


# Allow `python train_model.py` to run standalone.
if __name__ == "__main__":
    train_and_save_model()
