# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dummy training data
data = pd.DataFrame({
    'age': [20, 25, 30, 35, 40],
    'height': [160, 165, 170, 175, 180],
    'weight': [50, 60, 70, 80, 90],
    'gender': ['male', 'female', 'male', 'female', 'male'],
    'goal': ['lose', 'maintain', 'gain', 'lose', 'gain'],
    'future_weight': [48, 60, 75, 78, 95]
})

# One-hot encode categorical values
data = pd.get_dummies(data, columns=['gender', 'goal'])

# Train model
X = data.drop('future_weight', axis=1)
y = data['future_weight']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
