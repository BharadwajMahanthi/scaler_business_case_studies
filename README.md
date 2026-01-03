# Scaler Business Case Studies & Projects

Welcome to the **Scaler Business Case Studies** repository. This collection features a diverse range of Data Science and Machine Learning projects, covering Key areas such as Regression, Natural Language Processing (NLP), Feature Engineering, and Time Series Analysis.

Each project focuses on solving a specific business problem using real-world data and advanced analytical techniques.

## 📂 Repository Contents

| Project                                                                               | Domain               | Description                                                         | Key Tech                                           |
| :------------------------------------------------------------------------------------ | :------------------- | :------------------------------------------------------------------ | :------------------------------------------------- |
| **[Insurance Premium Prediction](#1-insurance-premium-prediction-portfolio-project)** | Regression / Web App | End-to-end ML app to estimate health insurance costs.               | Python, Streamlit, Scikit-learn, Random Forest     |
| **[FlipItNews](#2-flipitnews-nlp-news-classification)**                               | NLP / Classification | Automated categorization of news articles into 5 topics.            | NLP, TF-IDF, Naive Bayes, Random Forest            |
| **[Twitter NER](#3-twitter-ner-named-entity-recognition)**                            | NLP / Deep Learning  | Named Entity Recognition on Twitter data using BERT.                | BERT, Transformers, PyTorch/TensorFlow             |
| **[Jamboree Education](#4-jamboree-education-admission-prediction)**                  | Linear Regression    | Analyzing factors influencing graduate admission chances.           | Statsmodels, Linear Regression, Hypothesis Testing |
| **[Delhivery](#5-delhivery-feature-engineering)**                                     | Feature Engineering  | Data processing and feature engineering for logistics optimization. | Pandas, EDA, Data Cleaning                         |
| **[AdEase](#6-adease-time-series-analysis)**                                          | Time Series          | Forecasting traffic & measuring campaign impact.                    | ARIMA, Time Series Analysis                        |

---

## 🚀 Project Details

### 1. Insurance Premium Prediction (Portfolio Project)

A production-ready Machine Learning web application that predicts individual health insurance premiums based on demographics (Age, BMI, Smoking status, etc.).

- **Key Features**:
  - Interactive Streamlit UI.
  - Real-time predictions with confidence intervals.
  - deployable structure with `app.py`.
- **Location**: `portfolio_project/` and Root Directory.
- **Run the App**:
  ```bash
  streamlit run app.py
  ```

### 2. FlipItNews: NLP News Classification

A comprehensive NLP solution attempting to classify news articles into **Politics, Technology, Sports, Business, and Entertainment**.

- **Key Techniques**: Text Preprocessing (Tokenization, Lemmatization), TF-IDF vs Bag of Words, Multi-class Classification.
- **Location**: `NLPFlipItNews/`
- **Notebook**: `FlipItNews_Solution.ipynb`

### 3. Twitter NER: Named Entity Recognition

Deep learning project focused on extracting entities (like Person, Location, Organization) from Twitter text data.

- **Key Techniques**: BERT (Bidirectional Encoder Representations from Transformers), Token Classification.
- **Location**: `Datasets-TwitterNERNLP/`
- **Notebook**: `Twitter_NER_Solution.ipynb`

### 4. Jamboree Education: Admission Prediction

A statistical analysis case study to understand the probability of admission into IVY league colleges based on GRE, TOEFL, CGPA, and other metrics.

- **Key Techniques**: Linear Regression, Multicollinearity Check (VIF), Residual Analysis, Homoscedasticity.
- **Location**: `Jamboree_Education_cs/`
- **Notebook**: `Jamboree Education - Linear Regression.ipynb`

### 5. Delhivery: Feature Engineering

A deep dive into raw logistics data to clean, process, and engineer features useful for predicting delivery time and reliability.

- **Key Techniques**: Handling missing values, Outlier detection, Categorical encoding, Feature creation.
- **Location**: `delhivery_casestudy/`
- **Notebook**: `bcs_del.ipynb`

### 6. AdEase: Time Series Analysis

A time-series forecasting project for online ad campaigns.

- **Key Techniques**: ARIMA Forecasting, Seasonality Analysis, Exogenous Variable Impact.
- **Location**: `AdEase -time-Series/`
- **Notebook**: `AdEase_Solution.ipynb`

---

## 🛠️ Installation & Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/scaler_business_case_studies.git
   ```

2. **Install Common Dependencies**:
   Most projects rely on standard data science libraries.

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit statsmodels nltk transformers
   ```

3. **Explore a Project**:
   Navigate to the specific directory and launch Jupyter Notebook:
   ```bash
   cd NLPFlipItNews
   jupyter notebook FlipItNews_Solution.ipynb
   ```

## 🤝 Contributing

Feel free to open issues or submit pull requests if you have suggestions for improving the models or analysis.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
