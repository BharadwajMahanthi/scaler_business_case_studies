# Insurance Premium Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning-powered web application that estimates health insurance premiums based on individual health demographics. Built with Python, Scikit-Learn, and Streamlit.

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/insurance-premium-prediction.git
cd insurance-premium-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app/app.py
```

Access the application at `http://localhost:8501`.

## 📂 Project Structure

```
├── app/
│   ├── app.py                 # Streamlit application entry point
│   ├── random_forest_model.pkl # Trained model artifact
│   └── scaler.pkl             # Feature scaling artifact
├── data/
│   ├── insurance.csv          # Raw dataset
│   └── ...                    # Processed data
├── docs/
│   ├── ARCHITECTURE.md        # System design overview
│   └── MODEL_CARD.md          # Detailed model documentation
├── images/                    # Visualization assets
├── notebooks/
│   └── Insurance_Cost_Prediction.ipynb # Model training & analysis
├── requirements.txt           # Project dependencies
├── LICENSE
└── README.md
```

## 📊 Key Features

- **Interactive UI**: Real-time premium estimation using slider and dropdown inputs.
- **Robust Model**: Random Forest Regressor with ~89% R² score on test data.
- **Confidence Intervals**: Provides 95% confidence bounds for price estimates.
- **Exploratory Analysis**: Comprehensive EDA proving key correlations (Age, BMI, Smoking).

## 🧠 Model Performance

The predictive model uses a Random Forest Regressor trained on 986 records.

| Metric   | Score      |
| -------- | ---------- |
| R² Score | **0.89**   |
| RMSE     | **$2,128** |
| MAE      | **$1,014** |

Top drivers of premium cost:

1. **Age**: Primary factor (~65% importance)
2. **Transplants**: Major cost driver
3. **Chronic Diseases**: Significant impact

_For full model details, limitations, and biases, see the [Model Card](docs/MODEL_CARD.md)._

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Engine**: Scikit-Learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📈 Future Roadmap

- [ ] Docker container support
- [ ] API endpoint (FastAPI) for programmatic access
- [ ] SHAP values for individual prediction explainability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
