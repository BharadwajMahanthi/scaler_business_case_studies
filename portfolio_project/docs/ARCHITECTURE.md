# System Architecture

## Overview

The Insurance Premium Prediction system is designed as a lightweight, interactive web application powered by machine learning. It follows a simple client-server logic where the "client" is the Streamlit UI and the "server" logic is embedded within the Python application connecting to a pre-trained model.

## Components

### 1. User Interface (Streamlit)

- **Role**: Input collection and visualization.
- **Features**:
  - Dynamic sliders and dropdowns for health metrics.
  - Real-time BMI calculation feedback.
  - Result display with confidence intervals.

### 2. Processing Layer (Python/Pandas)

- **Input Transformation**: Converts raw user inputs into the format required by the model.
- **Feature Engineering**:
  - `BMI = Weight_kg / (Height_m)^2`
- **Scaling**: Applies standard scaling (`StandardScaler`) using the saved scalar object to normalize numerical inputs (Age, Height, Weight, BMI, Surgeries).

### 3. Inference Engine (Scikit-Learn)

- **Model**: Random Forest Regressor (`random_forest_model.pkl`).
- **Operation**: Loads model into memory on startup; performs inference on processed single-row dataframes.

### 4. Data Layer

- **Model Artifacts**: Serialized `.pkl` files stored locally (`app/`).
- **Dataset**: CSV files stored in `data/` for retraining (offline).

## Data Flow

1. User enters data (Age, Height, Weight, etc.) in Web UI.
2. App calculates BMI and assembles a raw feature vector.
3. App applies `scaler.pkl` to normalize numerical features.
4. Normalized vector is passed to `random_forest_model.pkl`.
5. Model returns predicted Premium Price.
6. App calculates ±95% CI bounds.
7. Result is rendered to the user.

## Deployment Strategy

- **Containerization**: Application is Dockerized for consistency across environments (Dev/Prod).
- **Hosting**: Can be deployed to Streamlit Cloud, AWS EC2, or any container orchestration platform.
