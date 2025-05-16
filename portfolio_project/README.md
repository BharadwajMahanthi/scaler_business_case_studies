
# Insurance Cost Prediction Project

## Problem Statement

Insurance companies need to accurately predict health insurance premiums to set competitive and fair prices while managing risk effectively. Traditional methods often rely on broad actuarial tables, which may not account for individual health nuances. This project leverages machine learning to predict `PremiumPrice` for individuals using the `insurance.csv` dataset, which contains 986 records and 11 features: `Age`, `Diabetes`, `BloodPressureProblems`, `AnyTransplants`, `AnyChronicDiseases`, `Height`, `Weight`, `KnownAllergies`, `HistoryOfCancerInFamily`, `NumberOfMajorSurgeries`, and `PremiumPrice`. The goal is to enhance precision in pricing, improve risk assessment, and enable personalized offerings.

 **Target Metric** :

* Minimize prediction errors (RMSE, MAE) for `PremiumPrice`.
* Maximize explained variance (R²) to ensure the model captures key patterns in the data.

## Steps Taken

### 1. Exploratory Data Analysis (EDA)

* **Dataset Overview** :
* 986 rows, 11 columns, no missing values.
* `PremiumPrice` ranges from 15,000 to 40,000 (mean ~24,000, std ~6500).
* Added `BMI` as a feature: `BMI = Weight / (Height/100)²`.
* **Distributions** :
* Numerical: `Age` (18–66, uniform), `Height` (145–188 cm, normal), `Weight` (51–132 kg, right-skewed), `BMI` (15–50, right-skewed), `NumberOfMajorSurgeries` (0–3, most at 0–1), `PremiumPrice` (peaks at 15,000 and 25,000).
* Categorical: ~50% have `Diabetes` or `BloodPressureProblems`; ~20% have `AnyTransplants` or `HistoryOfCancerInFamily`.
* **Correlations** :
* `Age` vs. `PremiumPrice`: 0.7 (strong positive).
* `NumberOfMajorSurgeries` vs. `PremiumPrice`: 0.26 (moderate).
* `BMI` vs. `PremiumPrice`: 0.1 (weak).
* **Outliers** :
* `PremiumPrice`: 6 values at 39,000–40,000.
* `BMI`: 22 values >42 (e.g., 48.89).
* **Boxplots** :
* Higher premiums for `Diabetes`=1, `AnyChronicDiseases`=1, `AnyTransplants`=1.
* `KnownAllergies` and `HistoryOfCancerInFamily` have minimal impact.

 **Visualizations** : See `images/` folder for plots (`numerical_distributions.png`, `categorical_distributions.png`, `correlation_heatmap.png`, `boxplots_by_categorical.png`).

### 2. Hypothesis Testing

* **H1: Diabetes vs. PremiumPrice** :
* T-test: p=0.0167 (<0.05).
* Result: Premiums are significantly higher for diabetic individuals (mean ~24,500 vs. ~23,800).
* **H2: Chronic Diseases vs. PremiumPrice** :
* T-test: p<0.0001.
* Result: Premiums are significantly higher for individuals with chronic diseases (mean ~25,500 vs. ~23,600).
* **H3: NumberOfMajorSurgeries vs. PremiumPrice** :
* ANOVA: p<0.0001.
* Result: Premiums differ significantly by number of major surgeries (e.g., 0: ~22,800, 3: ~28,300).

### 3. Machine Learning Modeling

* **Preprocessing** :
* Added `BMI` feature.
* Scaled numerical features (`Age`, `Height`, `Weight`, `BMI`, `NumberOfMajorSurgeries`) using `StandardScaler`.
* No encoding needed (categorical features are binary).
* **Models** :
* Linear Regression (baseline).
* Random Forest (tree-based, captures non-linearities).
* XGBoost (gradient boosting, high performance).
* **Evaluation Metrics** :
* RMSE, MAE (prediction errors).
* R² (explained variance).
* 5-fold cross-validation for stability.
* **Performance** (on test set, 20% of data):
  * **Linear Regression** : R² 0.7136, RMSE 3494.41, MAE 2586.18.
  * **Random Forest** : R² 0.8938, RMSE 2128.25, MAE 1014.29 (best model).
  * **XGBoost** : R² 0.8271, RMSE 2715.05, MAE 1341.17.
* **Cross-Validation** :
* Random Forest: CV R² Mean 0.7862, CV R² Std 0.0678 (most stable).
* XGBoost: CV R² Mean 0.7623, CV R² Std 0.0767.
* Linear Regression: CV R² Mean 0.6316, CV R² Std 0.0846.
* **Feature Importance** (Random Forest):
  * `Age`: 0.65 (top predictor, aligns with correlation of 0.7).
  * `AnyTransplants`: 0.10.
  * `AnyChronicDiseases`: 0.08.
  * `NumberOfMajorSurgeries`: 0.06.
  * `BMI`: 0.03 (aligns with weak correlation of 0.1).
  * See `images/feature_importance.png` for the plot.

 **Insights** :

* `Age` is the dominant predictor of premiums, followed by health conditions like `AnyTransplants` and `AnyChronicDiseases`.
* Random Forest captures non-linear relationships effectively, outperforming Linear Regression and XGBoost.
* The model’s accuracy (MAE ~4.2% of mean premium) supports precise pricing.

### 4. Deployment

* **Streamlit App** : Built a web-based calculator (`app.py`) to estimate premiums based on user inputs.
* **Functionality** :
* Users input health data (e.g., `Age`, `Diabetes`, `Height`, `Weight`) via sliders and dropdowns.
* The app calculates `BMI`, scales features, and predicts `PremiumPrice` using the trained Random Forest model.
* Displays the predicted premium with a 95% confidence interval (approximated as ±1.96 * 5000).
* **Steps to Run Locally** :

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Access at `http://localhost:8501`, input data, and view the predicted premium.

* **Files** :
* `app.py`: Streamlit app code.
* `random_forest_model.pkl`: Trained Random Forest model.
* `scaler.pkl`: Scaler for numerical features.
* **Optional Deployment** : Can be deployed to Streamlit Cloud for a live demo (requires a Streamlit account and GitHub repository).

## Final Scores Achieved

* **Best Model (Random Forest)** :
* **R²** : 0.8938 (explains 89.38% of variance in `PremiumPrice`).
* **RMSE** : 2128.25 (~8.9% of mean premium, ~33% of std).
* **MAE** : 1014.29 (~4.2% of mean premium).
* **Cross-Validation** : CV R² Mean 0.7862, CV R² Std 0.0678 (stable performance).

## Insights and Recommendations

* **Insights** :
* `Age` is the primary driver of premiums (correlation 0.7, feature importance 0.65), with older individuals paying more (e.g., 50+ age group: ~28,000 vs. 18–29: ~15,000).
* Health conditions like `AnyTransplants` (mean premium ~27,300 vs. ~23,600) and `AnyChronicDiseases` (mean ~25,500 vs. ~23,600) significantly increase costs.
* `BMI` has a weaker impact (correlation 0.1), but outliers (>42) indicate high-risk profiles.
* **Recommendations** :
* **Dynamic Pricing** : Adjust premiums based on `Age`, `AnyTransplants`, and `AnyChronicDiseases` for fairer pricing.
* **Wellness Programs** : Offer programs for diabetic and high-BMI individuals to reduce risk and premiums over time.
* **Risk Profiling** : Use the model to identify high-risk profiles (e.g., older individuals with transplants) for targeted interventions.
* **Customer Engagement** : Leverage insights for personalized marketing (e.g., tailored advice for high-risk groups).

## Additional Notes

* The project includes a Tableau dashboard for visualization (to be published on Tableau Public).
* A 2000-word technical blog and a 5-minute demo video are in progress as part of the submission requirements.
* All code and visualizations are documented in `notebooks/Insurance_Cost_Prediction.ipynb`.
