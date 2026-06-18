# Telco Customer Churn — Binary Classification

Predicting whether a telecom customer will churn (leave the service) based on account
features such as contract type, tenure, and monthly charges.

## Dataset

- **Source:** IBM Telco Customer Churn dataset (Kaggle) — `data/Telco-Customer-Churn.csv`
- **Task:** Binary classification — target column: `Churn` (Yes / No)
- **Size:** 7,043 customers × 21 features

## Metrics

Do not use accuracy as a primary metric — a majority-class baseline already reaches ~73.5%.

| Metric | Target | Notes |
|---|---|---|
| ROC-AUC | > 0.80 | Primary metric |
| F1-score | — | On minority class (`Churn=Yes`) |
| Precision | — | On minority class (`Churn=Yes`) |
| Recall | — | On minority class (`Churn=Yes`) |

**Class imbalance:** ~73.5% No / ~26.5% Yes — moderate imbalance.
Preferred handling: `class_weight='balanced'` or threshold tuning (try 0.3–0.4).
SMOTE is not the default approach at this imbalance ratio.

## Exploratory Data Analysis

**Dataset:** 7,043 rows × 21 columns — 3 numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`), 1 binary numeric (`SeniorCitizen`), and 16 categorical features.

**Key findings:**

- **Class imbalance:** Churn rate is 26.5% (1,869 vs 5,174). Accuracy is a misleading metric — use precision, recall, and F1.
- **Contract type dominates:** Month-to-month customers churn at 42.7% vs 11.3% (one-year) and 2.8% (two-year) — the strongest single churn signal.
- **Tenure is the top numeric predictor:** Churned customers average ~18 months tenure vs ~38 months for retained customers (corr ≈ −0.35).
- **`tenure` and `TotalCharges` are collinear** (corr ≈ 0.83) — include only one in linear models to avoid multicollinearity.
- **11 missing values in `TotalCharges`** disguised as blank strings for `tenure == 0` customers — impute as 0 or drop before modeling.
