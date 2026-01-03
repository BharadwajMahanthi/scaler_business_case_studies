from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import os
from .schemas import InsuranceInput, PredictionOutput

app = FastAPI(
title="Insurance Premium Prediction API",
description="API for estimating health insurance premiums based on health profile.",
version="1.0.0"
)

# Load artifacts
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../app/random_forest_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../app/scaler.pkl')

try:
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
model = None
scaler = None
print("Warning: Model artifacts not found. API will not likely function correctly.")

@app.get("/")
def read_root():
return {"message": "Welcome to the Insurance Premium Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict_premium(data: InsuranceInput):
if not model or not scaler:
raise HTTPException(status_code=503, detail="Model service unavailable")

# BMI Calculation
bmi = data.Weight / ((data.Height / 100) ** 2)

# Prepare DataFrame
input_dict = data.dict()
input_dict['BMI'] = bmi

# Feature ordering must match training
# Expected columns based on app.py logic
df = pd.DataFrame([input_dict])

# Selecting columns in correct order if needed, or relying on DF creation order
# Training columns: 'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI'
ordered_cols = [
'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases',
'Height', 'Weight', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI'
]
df = df[ordered_cols]

# Scale numerical features
numerical_cols = ['Age', 'Height', 'Weight', 'BMI', 'NumberOfMajorSurgeries']
df[numerical_cols] = scaler.transform(df[numerical_cols])

# Predict
try:
prediction = model.predict(df)[0]
except Exception as e:
raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

return {
"premium_price": round(prediction, 2),
"currency": "EUR/USD/INR", # Keeping generic as dataset unit wasn't specified as strict currency
"confidence_interval": {
"lower": round(prediction - 1.96 * 5000, 2),
"upper": round(prediction + 1.96 * 5000, 2)
}
}
