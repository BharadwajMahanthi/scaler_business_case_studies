import pytest
import pandas as pd
import numpy as np
import joblib
import os

@pytest.fixture
def model():
    model_path = os.path.join(os.path.dirname(__file__), '../app/random_forest_model.pkl')
    if not os.path.exists(model_path):
        pytest.skip("Model file not found. Skipping model tests.")
    return joblib.load(model_path)

@pytest.fixture
def scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), '../app/scaler.pkl')
    if not os.path.exists(scaler_path):
        pytest.skip("Scaler file not found. Skipping scaler tests.")
    return joblib.load(scaler_path)

def test_model_loading(model):
    """Test that the model loads correctly and is a valid sklearn estimator."""
    assert model is not None
    assert hasattr(model, 'predict')

def test_prediction_shape(model, scaler):
    """Test that the model returns a prediction for a valid input vector."""
    # Create a dummy input matching the training schema order:
    # Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, 
    # Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries, BMI
    
    # Raw values
    age = 30
    diabetes = 0
    bp = 0
    transplants = 0
    chronic = 0
    height = 175
    weight = 70
    allergies = 0
    cancer = 0
    surgeries = 0
    
    bmi = weight / ((height / 100) ** 2)
    
    input_df = pd.DataFrame([{
        'Age': age,
        'Diabetes': diabetes,
        'BloodPressureProblems': bp,
        'AnyTransplants': transplants,
        'AnyChronicDiseases': chronic,
        'Height': height,
        'Weight': weight,
        'KnownAllergies': allergies,
        'HistoryOfCancerInFamily': cancer,
        'NumberOfMajorSurgeries': surgeries,
        'BMI': bmi
    }])
    
    # Important: The scaler expects specific numerical columns in a specific order if it was fitted that way.
    # Based on app.py: numerical_cols = ['Age', 'Height', 'Weight', 'BMI', 'NumberOfMajorSurgeries']
    numerical_cols = ['Age', 'Height', 'Weight', 'BMI', 'NumberOfMajorSurgeries']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    prediction = model.predict(input_df)
    
    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.float64, np.float32))
    assert prediction[0] > 0
