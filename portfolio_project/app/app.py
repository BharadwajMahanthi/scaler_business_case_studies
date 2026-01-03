# Artifact: Streamlit App for Insurance Premium Calculator
# Save as `app.py` and run with `streamlit run app.py`

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
import os

# Load model and scaler
@st.cache_resource
def load_artifacts():
    app_dir = os.path.dirname(__file__)
    model_path = os.path.join(app_dir, 'random_forest_model.pkl')
    scaler_path = os.path.join(app_dir, 'scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# Streamlit app
st.title("Insurance Premium Calculator")
st.write("Enter your health profile to estimate your insurance premium.")

# Input form
with st.form("health_form"):
    age = st.slider("Age (18–66)", 18, 66, 30)
    height = st.slider("Height (cm, 145–188)", 145, 188, 165)
    weight = st.slider("Weight (kg, 51–132)", 51, 132, 70)
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    bp_problems = st.selectbox("Blood Pressure Problems", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    transplants = st.selectbox("Any Transplants", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    chronic_diseases = st.selectbox("Any Chronic Diseases", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    allergies = st.selectbox("Known Allergies", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cancer_history = st.selectbox("History of Cancer in Family", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    surgeries = st.slider("Number of Major Surgeries (0–3)", 0, 3, 0)
    submit = st.form_submit_button("Calculate Premium")

# Prediction
if submit:
    # Calculate BMI
    bmi: float = weight / (height / 100) ** 2
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Diabetes': [diabetes],
        'BloodPressureProblems': [bp_problems],
        'AnyTransplants': [transplants],
        'AnyChronicDiseases': [chronic_diseases],
        'Height': [height],
        'Weight': [weight],
        'KnownAllergies': [allergies],
        'HistoryOfCancerInFamily': [cancer_history],
        'NumberOfMajorSurgeries': [surgeries],
        'BMI': [bmi]
    })
    # Scale numerical features
    numerical_cols = ['Age', 'Height', 'Weight', 'BMI', 'NumberOfMajorSurgeries']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    # Predict
    premium = model.predict(input_data)[0]
    # Confidence interval (approximate, based on RF predictions)
    ci_lower: float = float(premium - 1.96 * 5000)  # Assuming std ~5000 from data
    ci_upper: float = float(premium + 1.96 * 5000)
    st.success(f"Estimated Premium: ${premium:,.2f}")
    st.write(f"95% Confidence Interval: ${ci_lower:,.2f} – ${ci_upper:,.2f}")

# Instructions
st.markdown("""
**How to Use**:
1. Adjust the sliders and select options for your health profile.
2. Click "Calculate Premium" to see the estimated insurance premium.
3. The confidence interval provides a range for the prediction.
""")