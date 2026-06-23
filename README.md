# ChurnScope — Telecom Customer Retention Intelligence

> End-to-end ML pipeline predicting telecom customer churn with XGBoost, Optuna tuning, and MLflow tracking — deployed as an interactive Streamlit portfolio app.

**Live Demo:** [To Be Added](https://teleco-churn-predictor.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Tech Stack](#tech-stack)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Feature Engineering](#feature-engineering)
8. [Key Decisions & Lessons](#key-decisions--lessons)
9. [File Structure](#file-structure)

---

## Project Overview

### The Problem

Telecommunication companies spend significantly more acquiring new customers than retaining existing ones, yet they have limited ability to intervene before a customer cancels their subscription. The challenge is identifying which customers are at risk of churning(leaving) *before* they leave, early enough that a retention offer can still work.

### End User

This system is built for a **Telecommunications Customer Retention team**: customer success managers who run outreach campaigns, and data analysts who monitor churn risk. The model surfaces high-risk customers so the team can prioritise who to call, what to offer, and when.

### The Data

**IBM Telco Customer Churn dataset** (publicly available on Kaggle):
- **7,043 rows**, 21 raw columns
- Binary target: `Churn` (Yes = 26.5%, No = 73.5%) i.e Class Imbalance is present though not extreme 
- Features span customer demographics, account info (contract type, payment method, tenure), and subscribed services (phone, internet, security add-ons, streaming)
- 11 rows dropped during cleaning: customers with `tenure=0` had blank-string `TotalCharges` that couldn't be coerced to numeric — a data entry artifact, not meaningful missing data

### What the Model Outputs

For each customer, the model returns a **churn probability** (0–1). At a 0.50 threshold:
- Flags **4 in 5 actual churners** (80.7% recall)
- Operates at **ROC-AUC 0.8403** on held-out test data

The Streamlit app also surfaces a risk-factor breakdown explaining *why* a given score is high or low, making the prediction interpretable to non-technical users.

### Key Design Decision

**Recall over precision as the primary business metric.** A missed churner is lost lifetime value. A false alarm is a courtesy retention call. The asymmetric cost of these errors justifies optimising recall — and it drove every modelling choice, from the imbalance strategy (`scale_pos_weight` rather than SMOTE) to the final model selection (XGBoost won the tiebreaker over Gradient Boosting on recall: 76.5% vs 53.7%, despite equal CV ROC-AUC).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│  Raw CSV          DataLoader          Cleaner                   │
│  (Kaggle)  ──►   + QualityGate  ──►  (11 rows  ──►             │
│  21 cols          8 checks            dropped)                  │
│                                                                 │
│          Feature                  Feature                       │
│   ──►    Engineering   ──►        Selection     ──►             │
│          (+11 features)           (corr ≥ 0.95,                 │
│                                   near-zero var)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL PIPELINE                            │
│                                                                 │
│  Baseline          Model               Optuna                   │
│  LogReg    ──►    Comparison   ──►    Tuning       ──►          │
│  (ROC-AUC          RF / GBM /         30 trials                 │
│   0.8387)          XGBoost            5-fold CV                 │
│                    5-fold CV          TPE sampler               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRACKING & SERVING                           │
│                                                                 │
│  MLflow            Production          Streamlit                │
│  Tracking  ──►    Model           ──►  App          ──►  User   │
│  SQLite            production_         4-page                   │
│  backend           model.pkl          portfolio                 │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow in brief:**
1. `Telco-Customer-Churn.csv` → `src/data/loader.py` reads, `src/data/cleaner.py` cleans → `data/cleaned.csv`
2. `src/features/engineering.py` adds 11 features → `src/features/run_features.py` applies selection → `data/features.csv`
3. `src/models/baseline.py` establishes LogReg reference
4. `src/models/compare_models.py` benchmarks RF, GBM, XGBoost with 5-fold stratified CV
5. `src/models/tuning.py` runs 30 Optuna trials on XGBoost → `models/best_params.json`
6. `src/models/run_training.py` logs both runs to MLflow → `models/production_model.pkl`
7. `app/streamlit_app.py` loads the production model and serves the portfolio app

---

## Results

### Model Comparison

| Model | CV ROC-AUC | Test ROC-AUC | Test Recall | Test F1 | Train Time |
|---|---|---|---|---|---|
| **XGBoost Tuned (Winner)** | **0.8513 ±0.0048** | **0.8403** | **0.8075** | **0.6176** | **1.05s** |
| Gradient Boosting | 0.8402 ±0.0046 | 0.8309 | 0.5374 | 0.5702 | 20.18s |
| XGBoost (untuned) | 0.8310 | 0.8310 | 0.7647 | 0.6131 | — |
| Random Forest | 0.8302 ±0.0065 | 0.8140 | 0.4813 | 0.5430 | 14.81s |
| Logistic Regression (Baseline) | — | 0.8387 | 0.5508 | 0.5928 | 0.23s |

_Evaluation: 80/20 stratified train/test split, `random_state=42`. CV on training set only. All models use `StandardScaler` + `OneHotEncoder` in a scikit-learn `Pipeline`._

### Improvement vs Baseline

| Metric | Baseline (LogReg) | Winner (XGBoost Tuned) | Delta |
|---|---|---|---|
| ROC-AUC | 0.8387 | 0.8403 | +0.0016 |
| Recall | 0.5508 | 0.8075 | **+25.7 pp** |
| F1 | 0.5928 | 0.6176 | +0.0248 |
| Precision | 0.6417 | 0.5000 | −0.1417 |

> **The headline win is recall, not AUC.** The tuned XGBoost catches 80.7% of actual churners vs the baseline's 55.1% — a 25.7 percentage-point improvement. The precision trade-off is acceptable: a false alarm in a retention campaign costs a phone call, not a customer.

### Hyperparameter Search Space (Optuna)

30 trials using TPE sampler, 5-fold stratified CV, optimising `roc_auc`:

| Parameter | Search Range |
|---|---|
| `n_estimators` | 100 – 500 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 1e-3 – 0.3 (log scale) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha` | 1e-8 – 10.0 (log scale) |
| `reg_lambda` | 1e-8 – 10.0 (log scale) |
| `min_child_weight` | 1 – 10 |
| `scale_pos_weight` | Fixed at 2.76 (neg/pos ratio) |

Best configuration stored in `models/best_params.json`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **pandas** | Data loading, cleaning, feature construction |
| **scikit-learn** | Preprocessing pipelines, LogReg, RF, GBM, cross-validation, metrics |
| **XGBoost** | Production model — regularised gradient boosting with native imbalance handling |
| **Optuna** | Bayesian hyperparameter search with TPE sampler |
| **MLflow 3.x** | Experiment tracking, metric logging, artifact storage (SQLite backend) |
| **Plotly** | Interactive charts: bar, box, heatmap, ROC curve, gauge |
| **Streamlit** | 4-page interactive portfolio web app |
| **pytest** | Unit and integration tests across data, features, and model layers |
| **Great Expectations** | Schema and data quality contracts |
| **FastAPI + Uvicorn** | REST API scaffold for future inference endpoint |
| **Docker + Compose** | Containerised deployment with volume mounts for data and models |
| **joblib** | Serialisation of sklearn pipelines to `.pkl` |

---

## Setup & Installation

### Prerequisites

- Python 3.9+ (developed on 3.12)
- pip
- Docker (optional, for containerised run)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Joachim-Wambua/telco-churn-predictor.git
cd telco-churn-predictor

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project as an editable package
#    (enables `from src.data.loader import ...` imports from any directory)
pip install -e .
```

### Dataset

Download the IBM Telco Customer Churn dataset from Kaggle and place it at:

```
data/Telco-Customer-Churn.csv
```

The app runs in **demo mode** without the dataset (synthetic data is generated on the fly), but the training pipeline requires the real file.

---

## How to Run

### Full Training Pipeline

Run steps in order — each step reads the output of the previous one.

```bash
# Step 1: Inspect raw data (optional — prints shape, dtypes, missing values)
python src/data/loader.py

# Step 2: Clean data → data/cleaned.csv
python src/data/cleaner.py

# Step 3: Feature engineering + selection → data/features.csv
python src/features/run_features.py

# Step 4: Baseline (Logistic Regression) → models/baseline.pkl
python src/models/baseline.py

# Step 5: Model comparison, RF / GBM / XGBoost, 5-fold CV → models/*.pkl
python src/models/compare_models.py

# Step 6: Optuna tuning, 30 trials → models/best_params.json + tuned_model.pkl
python src/models/tuning.py

# Step 7: Final training + MLflow logging → models/production_model.pkl
python src/models/run_training.py
```

Expected total time on a modern laptop: ~3–5 minutes (Optuna dominates).

### Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The app has four pages:

| Page | Content |
|---|---|
| Overview | KPI cards, dataset summary, tech badges |
| Explore Data | Churn by contract, feature distributions, correlation heatmap |
| Model Results | Comparison table, feature importance, confusion matrix, ROC curve, live predictor |
| How I Built This | Architecture diagram, build timeline, key technical decisions |

### MLflow UI

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Open http://localhost:5000. The `telco-customer-churn` experiment contains two registered runs: `baseline` (LogReg) and `tuned_xgboost`, each with full parameter and metric history plus joblib artifact attachments.

### Docker

```bash
# Build image and start container
docker compose up --build

# Open http://localhost:8501

# Stop
docker compose down
```

`docker-compose.yml` mounts `./data` and `./models` as volumes so a pre-trained model is loaded at runtime without rebuilding the image.

### Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Individual modules
pytest tests/test_data_quality.py -v   # null rates, schema checks, row counts
pytest tests/test_features.py -v       # engineering correctness, selection logic
pytest tests/test_model.py -v          # loading, prediction shape, probability validity
```

`test_model.py` requires `models/production_model.pkl`. Run the full training pipeline first or these tests will fail with a clear assertion error pointing to the missing file.

---

## Feature Engineering

11 features were engineered across three categories in `src/features/engineering.py`, then filtered through a two-pass selection: correlation ≥ 0.95 (upper-triangle walk, keeping the first of any collinear pair) and near-zero variance (cutoff = 1% of median variance, using median to prevent `TotalCharges` from inflating the threshold).

### Domain-Specific Features (Churn Drivers)

| Feature | Type | Business Rationale |
|---|---|---|
| `is_month_to_month` | Binary | Zero switching cost. EDA churn rate: **42.7%** vs 11.3% (one year) and 2.8% (two year). Strongest single signal. |
| `num_addon_services` | Count (0–6) | Each add-on (OnlineSecurity, Backup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies) raises the practical cost of leaving. |
| `has_fiber_optic` | Binary | Fiber customers face stronger competition from cable and 5G alternatives. Dataset churn rate: **41.9%** vs 18.9% DSL. |
| `uses_auto_payment` | Binary | Automatic billing (bank transfer or credit card) puts payment on autopilot — customers rarely review their subscription actively. |
| `is_new_customer` | Binary | Tenure ≤ 3 months is the highest-risk onboarding window, before habits are formed. |

### Statistical Transformations

| Feature | Formula | Rationale |
|---|---|---|
| `avg_monthly_spend` | `TotalCharges / (tenure + 1)` | Raw `TotalCharges` is nearly perfectly collinear with `tenure × MonthlyCharges` (corr ≈ 0.83). This ratio captures price experience without duplicating the tenure signal. |
| `tenure_stage` | Buckets: 0 (≤12 mo), 1 (13–36 mo), 2 (>36 mo) | The churn–tenure relationship is non-linear: risk drops sharply after year one, again after year three. Bucket encoding captures this without imposing linearity. |
| `monthly_charges_vs_median` | `MonthlyCharges − median(MonthlyCharges)` | Customers paying well above the median have stronger financial motivation to price-compare. Those below feel less pressure to switch. |

### Interaction Features

| Feature | Formula | Rationale |
|---|---|---|
| `high_charge_x_no_contract` | `MonthlyCharges × is_month_to_month` | Isolates the highest-risk profile: expensive + no lock-in. Neither factor alone captures it — a cheap month-to-month customer stays out of inertia; a pricey two-year customer can't easily leave. |
| `stickiness_score` | `tenure × num_addon_services` | Every additional month deepens habits; every add-on raises switching effort. The product captures compound retention on both dimensions simultaneously. |
| `charge_per_service` | `MonthlyCharges / (num_addon_services + 1)` | Value-perception proxy. A high ratio flags customers paying a lot relative to what they subscribe to — a dissatisfaction signal. |

### Feature Importance (XGBoost Gain — Top 10)

| Rank | Feature | Importance (gain) |
|---|---|---|
| 1 | `is_month_to_month` | 0.2243 |
| 2 | `Contract_Month-to-month` (OHE) | 0.2194 |
| 3 | `high_charge_x_no_contract` | 0.0734 |
| 4 | `tenure` | 0.0523 |
| 5 | `stickiness_score` | 0.0481 |
| 6 | `MonthlyCharges` | 0.0452 |
| 7 | `charge_per_service` | 0.0398 |
| 8 | `TotalCharges` | 0.0341 |
| 9 | `PaymentMethod_Electronic check` (OHE) | 0.0287 |
| 10 | `InternetService_Fiber optic` (OHE) | 0.0265 |

The dominance of contract-related features confirms the EDA finding: **contract type is the primary lever for churn risk**, accounting for over 44% of model gain between the raw feature and its one-hot encoding.

---

## Key Decisions & Lessons

- **`scale_pos_weight` over SMOTE for class imbalance.** SMOTE generates synthetic minority samples, which can leak information across the train/test boundary and doesn't translate to real inference. Setting `scale_pos_weight = 2.76` (the negative/positive ratio in the training set) tells XGBoost to cost-weight each `Churn=Yes` example more heavily during training — no artificial data, no boundary leakage, and the weight is directly interpretable. At a 26.5% positive rate the imbalance was mild enough that weighting alone was sufficient.

- **Recall as the tiebreaker, not AUC.** Gradient Boosting and XGBoost tied on CV ROC-AUC (both ≈ 0.8402). The deciding factor was test-set recall: 76.5% (XGBoost) vs 53.7% (GBM). In a retention campaign, a missed churner represents lost lifetime value. A false positive costs a phone call. That asymmetry makes recall the right tiebreaker, not a metric to optimise incidentally.

- **Failure: `variances.mean()` in feature selection eliminated all useful features.** The first implementation of `select_features` used `variances.mean()` as the low-variance cutoff. `TotalCharges` has variance ~400,000, which inflated the mean to ~3,965 — high enough that `tenure`, `MonthlyCharges`, and every engineered feature was flagged as "near-zero variance" and dropped. Switching to `variances.median()` (~9) produced a sensible cutoff. **Lesson: always audit the scale of your features before using an aggregate as a threshold. `mean` and `median` can differ by orders of magnitude when one feature dominates.**

- **MLflow 3.x requires explicit SQLite backend and `skops_trusted_types`.** MLflow 3.x deprecated the default filesystem tracking backend (`mlruns/`). The fix was `mlflow.set_tracking_uri("sqlite:///mlflow.db")`. XGBoost artifacts additionally required `skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]` in `log_model()` — MLflow 3.x introduced a security model requiring explicit trust declarations for third-party objects inside sklearn estimators.

- **Optuna found a counterintuitive optimum: shallow trees, low learning rate.** Conventional XGBoost intuition on tabular data suggests moderate depth (4–6) and learning rate 0.05–0.1. The best Optuna trial settled on `max_depth=3` with a lower-than-default learning rate and higher regularisation. This makes sense for a dataset of ~5,600 training rows: shallower trees with stronger regularisation prevent overfitting on relatively small data. It's a configuration that wouldn't have been reached by manual grid search over "sensible" defaults.

---

## File Structure

```
telco-churn-predictor/
│
├── .github/
│   └── workflows/              # CI pipeline (GitHub Actions)
│
├── .streamlit/
│   └── config.toml             # Streamlit theme and server settings
│
├── app/
│   ├── streamlit_app.py        # 4-page portfolio app (Overview, EDA, Results, Process)
│   └── generate_data.py        # Generates synthetic demo data for the app
│
├── data/
│   ├── Telco-Customer-Churn.csv    # Raw dataset (download separately from Kaggle)
│   ├── cleaned.csv                 # Output of cleaner.py (7,032 rows)
│   ├── features.csv                # Output of run_features.py (engineered + selected)
│   ├── model_results.json          # Serialised model comparison metrics
│   └── predictions.csv             # Test-set predictions for confusion matrix display
│
├── models/
│   ├── baseline.pkl            # Logistic Regression sklearn pipeline
│   ├── random_forest.pkl       # Random Forest sklearn pipeline
│   ├── gradient_boosting.pkl   # Gradient Boosting sklearn pipeline
│   ├── xgboost.pkl             # XGBoost sklearn pipeline (untuned)
│   ├── tuned_model.pkl         # XGBoost sklearn pipeline (Optuna-tuned)
│   ├── production_model.pkl    # Production model loaded by Streamlit app
│   └── best_params.json        # Best hyperparameters found by Optuna
│
├── notebooks/
│   └── eda.ipynb               # 8-section EDA: distributions, correlations, churn segments
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # CSV reader, shape/dtype/missing value summaries
│   │   ├── cleaner.py          # Dtype coercion, null handling, deduplication pipeline
│   │   └── quality.py          # 8-point quality gate (null rates, schema, value ranges)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py      # create_features() and select_features()
│   │   └── run_features.py     # CLI: cleaned.csv → features.csv
│   │
│   └── models/
│       ├── __init__.py
│       ├── baseline.py         # Logistic Regression baseline → models/baseline.pkl
│       ├── compare_models.py   # RF / GBM / XGBoost 5-fold CV comparison
│       ├── tuning.py           # Optuna study (30 trials) → best_params.json
│       └── run_training.py     # MLflow logging + production model export
│
├── tests/
│   ├── conftest.py             # Shared pytest fixtures (features_df, raw_df)
│   ├── test_data_quality.py    # Quality gate: null rates, schema, row counts
│   ├── test_features.py        # Feature engineering correctness and selection logic
│   └── test_model.py           # Model: loading, prediction shape, probability validity
│
├── config.yaml                 # Project config: target col, imbalance strategy, metrics
├── Dockerfile                  # python:3.9-slim image, installs deps, runs Streamlit
├── docker-compose.yml          # Mounts data/ and models/ volumes, exposes :8501
├── mlflow.db                   # SQLite MLflow backend (runs, metrics, artifact refs)
├── mlruns/                     # MLflow run artifact storage
├── requirements.txt            # Python dependencies
├── setup.py                    # Editable install for src/ package
└── README.md                   # This file
```

---

## Contact

**Joachim Wambua** — Data Scientist  
kimwambua96@gmail.com  
[GitHub](https://github.com/Joachim-Wambua) · [Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
