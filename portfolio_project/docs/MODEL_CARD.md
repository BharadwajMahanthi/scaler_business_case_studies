# Model Card: Insurance Premium Predictor

## Model Details

- **Model Type**: Random Forest Regressor
- **Framework**: Scikit-Learn
- **Version**: 1.0.0
- **License**: MIT
- **Inputs**: 10 Features (Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries)
- **Target**: PremiumPrice (Continuous)

## Intended Use

- **Primary Use Case**: Estimating health insurance premiums for individuals based on health demographics and history.
- **Target Users**: Insurance agents, risk assessors, and individuals seeking premium estimates.
- **Out of Scope**: Medical diagnosis, definitive actuarial calculation (should be used as an estimate only).

## Training Data

- **Dataset**: `insurance.csv` (986 records)
- **Preprocessing**:
  - Calculated BMI from Height and Weight.
  - Standard scaling applied to numerical features.
- **Train/Test Split**: 80/20 random split.

## Performance Metrics

The model was evaluated on a hold-out test set (20% of data).

| Metric       | Score   | Interpretation                       |
| ------------ | ------- | ------------------------------------ |
| **R² Score** | 0.8938  | Explains ~89% of variance in price.  |
| **MAE**      | 1014.29 | Average prediction error is ~$1,014. |
| **RMSE**     | 2128.25 | Root Mean Squared Error.             |

## Feature Importance

The Random Forest model identified the following as top predictors:

1. **Age** (~65% importance)
2. **AnyTransplants** (~10%)
3. **AnyChronicDiseases** (~8%)
4. **NumberOfMajorSurgeries** (~6%)
5. **BMI** (~3%)

## Limitations & Biases

- **Data Size**: Trained on a relatively small dataset (<1000 records).
- **Demographics**: May not generalize to populations outside the training distribution.
- **BMI Bias**: BMI is a simplified health metric and may not capture muscle mass vs. fat composition.
- **Historical Bias**: Predictions reflect historical pricing patterns which may contain systemic biases.

## Ethical Considerations

- **Fairness**: Age is a dominant factor; ensure this aligns with local regulations regarding age-based pricing.
- **Privacy**: No PII (Personally Identifiable Information) is used/stored by the model during inference.
