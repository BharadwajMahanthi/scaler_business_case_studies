import pytest
import pandas as pd
import numpy as np

def calculate_bmi(weight_kg, height_cm):
    """Refactored logic from app.py for testing."""
    return weight_kg / ((height_cm / 100) ** 2)

def test_bmi_calculation():
    """Test BMI calculation accuracy."""
    weight = 70
    height = 175
    expected_bmi = 70 / (1.75 ** 2) # ~22.86
    
    calculated = calculate_bmi(weight, height)
    
    assert np.isclose(calculated, expected_bmi, atol=0.01)

def test_bmi_edge_cases():
    """Test BMI with extreme but valid inputs."""
    # Minimum realistic
    bmi_min = calculate_bmi(40, 150)
    assert bmi_min > 0
    
    # Maximum realistic
    bmi_max = calculate_bmi(150, 150)
    assert bmi_max > 0
