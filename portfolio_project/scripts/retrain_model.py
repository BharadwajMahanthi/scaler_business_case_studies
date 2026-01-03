import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def retrain():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '../data/insurance.csv')
    df = pd.read_csv(data_path)

    # Feature Engineering
    print("Calculating BMI...")
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # Defined Features matching App
    features = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants', 
                'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies', 
                'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI']
    
    target = 'PremiumPrice'

    X = df[features]
    y = df[target]

    # Scaling
    # App expects scaler to work on these columns only
    numerical_cols = ['Age', 'Height', 'Weight', 'BMI', 'NumberOfMajorSurgeries']
    
    print("Fitting Scaler...")
    scaler = StandardScaler()
    # We fit only on numerical columns
    scaler.fit(X[numerical_cols])
    
    # Transform X for training
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.transform(X[numerical_cols])

    # Model Training - Upgrading to XGBoost with Tuning
    print("Training XGBoost with Hyperparameter Tuning...")
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    # Initial XGBoost Regressor
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Simple Parameter Grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Grid Search
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='r2')
    grid_search.fit(X_scaled, y)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score (R2): {grid_search.best_score_}")

    model = grid_search.best_estimator_

    # Saving Artifacts
    output_dir = os.path.join(os.path.dirname(__file__), '../app')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'random_forest_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving scaler to {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    
    print("Done. Artifacts updated to match current sklearn version.")

if __name__ == "__main__":
    retrain()
