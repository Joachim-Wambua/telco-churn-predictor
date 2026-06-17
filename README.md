# Credit Card Fraud Detection

A machine learning project for detecting customer churn using the Telco Customer Churn dataset.

## Dataset

- **Source:** `data/Telco-Customer-Churn.csv`
- **Task:** Binary classification (`Churn`: Yes / No)

## Exploratory Data Analysis

**Dataset:** 7,043 rows × 21 columns — 3 numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`), 1 binary numeric (`SeniorCitizen`), and 16 categorical features.

**Key findings:**

- **Class imbalance:** Churn rate is 26.5% (1,869 vs 5,174). Accuracy is a misleading metric — use precision, recall, and F1.
- **Contract type dominates:** Month-to-month customers churn at 42.7% vs 11.3% (one-year) and 2.8% (two-year) — the strongest single churn signal.
- **Tenure is the top numeric predictor:** Churned customers average ~18 months tenure vs ~38 months for retained customers (corr ≈ −0.35).
- **`tenure` and `TotalCharges` are collinear** (corr ≈ 0.83) — include only one in linear models to avoid multicollinearity.
- **11 missing values in `TotalCharges`** disguised as blank strings for `tenure == 0` customers — impute as 0 or drop before modeling.
