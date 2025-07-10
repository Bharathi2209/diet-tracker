import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    "age": [20, 25, 30, 22, 28, 35, 40, 26, 32, 45],
    "height": [160, 170, 175, 165, 180, 168, 172, 174, 169, 178],
    "weight": [50, 65, 70, 55, 80, 60, 75, 68, 66, 85],
    "gender": ["Female", "MALE", "Male", "FEMALE", "male", "female", "MALE", "Male", "female", "MALE"],
    "goal": ["Gain", "lose", "Maintain", "GAIN", "Lose", "Maintain", "LOSE", "gain", "maintain", "GAIN"],
    "future_weight": [55, 60, 70, 60, 75, 60, 70, 73, 66, 90]
}

df = pd.DataFrame(data)

# Filter valid positive age
df = df[df["age"] > 0]

# Normalize gender and goal (case-insensitive)
df["gender"] = df["gender"].str.lower().map({"male": 1, "female": 0})
df["goal"] = df["goal"].str.lower().map({"lose": 0, "maintain": 1, "gain": 2})

# Drop bad/unmapped rows
df.dropna(inplace=True)

# Features and target
X = df[["age", "height", "weight", "gender", "goal"]]
y = df["future_weight"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
