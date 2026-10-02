# Financial Fraud Detection

## Project Overview

Financial fraud is a major challenge for financial institutions and digital payment systems because fraudulent transactions can result in financial losses and customer risk.

This project develops a machine learning and data analytics solution to identify potentially fraudulent financial transactions, analyze fraud patterns, assign transaction-level risk levels, and present the results through an interactive Streamlit dashboard.

The project follows an end-to-end analytical workflow covering data assessment, data preprocessing, exploratory data analysis, feature engineering, machine learning model development, model evaluation, fraud risk analysis, and dashboard development.

## Project Objectives

- Identify potentially fraudulent financial transactions using machine learning techniques.
- Perform data quality assessment, preprocessing, and feature engineering on transaction data.
- Analyze transaction patterns and key factors associated with fraudulent activity.
- Develop and evaluate supervised machine learning models for fraud classification.
- Generate transaction-level fraud probabilities and risk categories.
- Analyze high-risk transactions to support fraud investigation and monitoring.
- Develop an interactive Streamlit dashboard for fraud analytics and risk visualization.
- Present model performance using appropriate classification metrics rather than relying only on accuracy.

## Dataset Overview

The project uses the `financial_fraud_detection_dataset.csv` dataset containing 5,000 financial transactions and 14 original columns.

### Dataset Characteristics

- **Total transactions:** 5,000
- **Total features:** 13 predictor/identifier fields
- **Target variable:** `Fraudulent`
- **Fraudulent transactions:** 482
- **Non-fraudulent transactions:** 4,518
- **Fraud rate:** 9.64%
- **Missing values:** None
- **Duplicate rows:** None

### Main Variables

| Variable | Description |
|---|---|
| `Transaction_ID` | Unique transaction identifier |
| `Customer_ID` | Customer identifier |
| `Transaction_Date` | Date and time of the transaction |
| `Transaction_Amount` | Transaction value |
| `Merchant_Category` | Category of merchant |
| `Payment_Method` | Payment method used |
| `Device_Type` | Device used for the transaction |
| `Location` | Transaction location |
| `Is_International` | Indicates whether the transaction is international |
| `Previous_Transactions` | Number of previous transactions |
| `Average_Spend` | Customer's average spending |
| `Account_Age_Days` | Age of the customer account in days |
| `Suspicious_Keyword` | Indicator of a suspicious keyword |
| `Fraudulent` | Target variable indicating fraud |

The dataset was selected because it provides a manageable transaction volume, a meaningful fraud rate, and categorical and numerical variables suitable for fraud-pattern analysis and machine learning.

## Project Methodology

The project follows an end-to-end data analytics and machine learning workflow:

1. **Dataset Assessment** — Examined dataset structure, data types, missing values, duplicate records, class distribution, and variable characteristics.
2. **Data Quality & Cleaning** — Validated transaction dates, checked data consistency, and assessed potential data-quality issues.
3. **Exploratory Data Analysis** — Analyzed fraud patterns across transaction characteristics, merchant categories, payment methods, devices, locations, international transactions, suspicious keywords, and time-based features.
4. **Feature Engineering** — Extracted transaction hour, day, day of week, month, and a night-transaction indicator from the transaction date.
5. **Preprocessing** — Removed identifier fields from the initial model feature set, separated numerical and categorical variables, standardized numerical features, and applied one-hot encoding to categorical variables.
6. **Train-Test Split** — Used a stratified 80:20 train-test split to preserve the fraud/non-fraud class distribution.
7. **Machine Learning** — Developed Logistic Regression, Decision Tree, Random Forest, and XGBoost classification models.
8. **Model Evaluation** — Evaluated models using Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and confusion matrices.
9. **Threshold Analysis** — Evaluated classification thresholds using a validation dataset before applying selected thresholds to the held-out test set.
10. **Fraud Risk Analysis** — Used Random Forest fraud probabilities to classify test transactions into Low, Medium, High, and Critical risk levels.
11. **Dashboard Development** — Developed an interactive Streamlit dashboard presenting fraud KPIs, trends, fraud indicators, risk distributions, high-risk transactions, and model performance.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to identify patterns and characteristics associated with fraudulent transactions.

### Key Observations

- Transactions containing a **Suspicious Keyword** had a fraud rate of approximately **47.31%**, compared with **7.57%** for transactions without one.
- **International transactions** had a fraud rate of approximately **36.96%**, compared with **7.00%** for domestic transactions.
- **Night transactions** had a fraud rate of approximately **25.61%**, compared with **4.47%** during non-night periods.
- Fraud rates across merchant categories ranged from approximately **7.54% to 11.43%**.
- Fraud rates across payment methods ranged from approximately **8.20% to 10.41%**.
- Fraud rates across device types ranged from approximately **9.16% to 10.24%**.
- Fraud rates across locations ranged from approximately **8.58% to 10.53%**.
- Fraud activity was relatively elevated during the **00:00–05:59** period.
- Monthly fraud rates varied over the observed period, with **October** showing the highest monthly fraud rate in the dataset.
- Transaction amount distributions showed substantial overlap between fraudulent and non-fraudulent transactions, indicating that transaction amount alone was not a strong fraud discriminator.

These findings were used to guide feature engineering, model development, and fraud risk analysis.

## Feature Engineering & Preprocessing

The transaction data was prepared using a leakage-aware preprocessing workflow.

### Feature Engineering

The following time-based features were derived from `Transaction_Date`:

- `Transaction_Hour`
- `Transaction_Day`
- `Transaction_DayOfWeek`
- `Transaction_Month`
- `Night_Transaction`

`Transaction_ID` and `Customer_ID` were excluded from the initial model feature set because they are identifiers rather than meaningful transactional predictors, while the original identifiers were retained for analysis and dashboard use.

### Preprocessing

- Numerical features were standardized using `StandardScaler`.
- Categorical features were converted using `OneHotEncoder`.
- Unknown categorical values were handled using `handle_unknown="ignore"`.
- A `ColumnTransformer` was used to apply the appropriate preprocessing to numerical and categorical features.
- The data was divided using an **80:20 stratified train-test split**.
- The training set contained **4,000 transactions**, while the test set contained **1,000 transactions**.
- Class imbalance was addressed using class-weighted model configurations as the primary approach.
- SMOTE was considered as a controlled training experiment and was not applied to the held-out test data.

The resulting preprocessing pipeline transformed the original model inputs into a consistent feature matrix suitable for machine learning.

## Machine Learning Models

Four supervised machine learning algorithms were developed and evaluated for fraudulent transaction classification:

| Model | Description |
|---|---|
| Logistic Regression | A linear classification model used as a baseline for fraud classification. |
| Decision Tree | A tree-based model that learns rule-based transaction patterns. |
| Random Forest | An ensemble of decision trees designed to capture nonlinear relationships and improve predictive robustness. |
| XGBoost | A gradient-boosting algorithm that builds an ensemble of sequential decision trees. |

Class imbalance was considered during model development using class-weighted configurations and `scale_pos_weight` for XGBoost.

The models were trained using the processed training data and evaluated on the held-out test dataset.

## Model Evaluation

Because the dataset is imbalanced, model performance was evaluated using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC rather than relying only on accuracy.

### Default Threshold Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.743 | 0.2720 | 1.0000 | 0.4276 | 0.8879 | 0.4185 |
| Decision Tree | 0.876 | 0.2812 | 0.1875 | 0.2250 | 0.5683 | 0.1307 |
| Random Forest | 0.887 | 0.3951 | 0.3333 | 0.3616 | 0.8928 | 0.4251 |
| XGBoost | 0.862 | 0.3220 | 0.3958 | 0.3551 | 0.8734 | 0.3359 |

### Threshold Analysis

Classification thresholds were evaluated using a validation split rather than selecting thresholds directly from the held-out test set.

The selected thresholds were subsequently applied to the untouched test set:

| Model | Selected Threshold | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.75 | 0.3613 | 0.4479 | 0.4000 |
| Decision Tree | 0.10 | 0.2812 | 0.1875 | 0.2250 |
| Random Forest | 0.35 | 0.3261 | 0.7812 | 0.4601 |
| XGBoost | 0.10 | 0.2769 | 0.8854 | 0.4218 |

The threshold analysis demonstrates that changing the classification threshold can materially change the balance between precision and recall, which is particularly important when the cost of missed fraudulent transactions and false alerts differs.

## Fraud Risk Analysis

The Random Forest model was used to generate fraud probabilities for the held-out test transactions, which were then grouped into four risk levels.

### Risk Classification

| Risk Level | Fraud Probability Range |
|---|---:|
| Low | 0.00–0.20 |
| Medium | 0.20–0.50 |
| High | 0.50–0.75 |
| Critical | 0.75–1.00 |

### Risk Distribution

| Risk Level | Transactions | Share of Test Set | Observed Fraud Rate |
|---|---:|---:|---:|
| Low | 649 | 64.9% | 0.15% |
| Medium | 270 | 27.0% | 23.33% |
| High | 79 | 7.9% | 39.24% |
| Critical | 2 | 0.2% | 50.00% |

The observed fraud rate increases substantially across the risk categories in the held-out test data, providing a useful basis for prioritizing transactions for further review.

The dashboard also provides a high-risk transaction view containing transaction characteristics, fraud probability, and assigned risk level.

## Streamlit Dashboard

An interactive Streamlit dashboard was developed to present fraud analytics and model-based risk insights in a user-friendly interface.

### Dashboard Components

- **KPI Cards**
  - Total Transactions
  - Fraud Transactions
  - Fraud Rate
  - Fraudulent Transaction Amount

- **Fraud Analytics**
  - Monthly fraud-rate trend
  - Transaction fraud distribution
  - Fraud rate by merchant category
  - Fraud rate by payment method
  - Fraud indicators by international transaction status and suspicious keyword

- **Risk Analytics**
  - Model-based fraud risk distribution
  - Observed fraud rate by risk level
  - High-risk transaction review
  - Interactive risk-level filtering

- **Model Performance**
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - PR-AUC
  - Classification threshold

The dashboard is designed as an analytical and demonstration interface using the project's historical test data and model outputs rather than as a live production fraud-monitoring system.

## Project Structure

```text
Financial-Fraud-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_assessment.ipynb
│   ├── 02_feature_engineering_preprocessing.ipynb
│   ├── 03_fraud_detection_modeling.ipynb
│   └── 04_fraud_risk_analysis.ipynb
│
├── reports/
│   ├── feature_analysis/
│   ├── default_threshold_model_metrics.csv
│   ├── final_model_metrics.csv
│   ├── final_confusion_matrices.csv
│   ├── validation_threshold_results.csv
│   ├── risk_level_summary.csv
│   ├── risk_level_performance.csv
│   ├── fraud_distribution_by_risk.csv
│   └── high_risk_transactions.csv
│
├── models/
├── src/
├── requirements.txt
├── README.md
├── .gitignore
└── run.py

## Technologies Used

### Programming & Analysis
- Python
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn
- Plotly

### Machine Learning
- Scikit-learn
- XGBoost
- Imbalanced-learn

### Dashboard
- Streamlit

### Development Environment
- VS Code
- Jupyter Notebook
- Git & GitHub

### Data & Model Utilities
- OpenPyXL
- Joblib

## How to Run the Project
### 1. Clone the Repository

```bash
git clone https://github.com/chinnu835/financial-fraud-detection.git
cd financial-fraud-detection




## Key Findings

The project identified several important patterns related to financial fraud:

- Suspicious keyword transactions showed a significantly higher observed fraud rate compared with normal transactions.
- International transactions demonstrated a higher fraud occurrence compared with domestic transactions.
- Fraud risk increased consistently across the model-generated risk categories from Low to Critical.
- Machine learning models were able to generate transaction-level fraud probabilities and classify potentially risky transactions.
- Random Forest provided strong ranking capability for fraud risk analysis, enabling the creation of operational risk categories.
- The Streamlit dashboard provides an interactive way to explore fraud trends, risk levels, high-risk transactions, and model performance.

## Future Enhancements

## Author