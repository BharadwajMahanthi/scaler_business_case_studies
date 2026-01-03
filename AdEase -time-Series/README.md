# AdEase Time Series Analysis

## 📊 Project Overview

This project focuses on **Time Series Forecasting** for "AdEase", analyzing daily traffic data for various online campaigns (represented as pages). The goal is to understand traffic patterns, evaluate the impact of marketing interventions (Exogenous variables), and forecast future views to optimize ad inventory planning.

## 📂 Files Description

- **`AdEase_Solution.ipynb`**: The main Jupyter Notebook containing:
  - Data Preprocessing and Cleaning.
  - Exploratory Data Analysis (EDA) of top performing campaigns.
  - Analysis of the "Campaign Active" exogenous vs. traffic correlation.
  - **ARIMA** (AutoRegressive Integrated Moving Average) Forecasting model implementation.
- **`train_1.csv`**: Daily traffic dataset.

  - Rows: Different Campaigns/Pages.
  - Columns: Daily view counts from 2015-07-01 to 2016-12-31.

- **`Exog_Campaign_eng`**: Binary exogenous variable file.
  - `1`: Campaign Active / Special Event.
  - `0`: Normal Day.

## 🚀 Key Insights & Methodology

1. **Top Campaign Identification**: We identified the highest traffic pages to focus our analysis on high-impact inventory.
2. **Campaign Impact**: We analyze whether the presence of the exogenous "Campaign" flag significantly alters daily traffic.
3. **Forecasting**: An ARIMA model serves as the baseline to predict future traffic, allowing AdEase to estimate inventory availability.

## 🛠️ Usage

1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels
   ```
2. Launch the analysis:
   ```bash
   jupyter notebook AdEase_Solution.ipynb
   ```
