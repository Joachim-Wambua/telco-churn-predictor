import sys
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR      = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = DATA_DIR / "features.csv"

TARGET_COL   = "Churn"
POS_LABEL    = "Yes"
DROP_COLS    = ["customerID"]
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# Baseline reference from src/models/baseline.py (LogisticRegression defaults)
BASELINE_ROC_AUC = 0.8387

# Why each model fits this problem — printed with the winner summary
_MODEL_RATIONALE = {
    "Random Forest": (
        "Parallel ensemble of decorrelated trees. "
        "Handles the mix of binary flags (is_month_to_month, has_fiber_optic) and "
        "continuous features (tenure, MonthlyCharges) without scaling. "
        "class_weight='balanced' corrects for the 26.5%/73.5% split. "
        "Feature importance gives Telco teams a ranked list of business drivers."
    ),
    "Gradient Boosting": (
        "Sequential ensemble — each tree corrects the residual errors of the previous one. "
        "Captures non-linear interactions (e.g. new customers on fiber optic paying above "
        "median) that logistic regression misses. Consistently strong on tabular classification "
        "problems where the signal is spread across many weak predictors."
    ),
    "XGBoost": (
        "Regularised gradient boosting with L1/L2 penalties to prevent overfitting on the "
        "engineered features. scale_pos_weight explicitly up-weights the minority class (Churn=Yes), "
        "improving recall — critical for Telco teams who need to catch churners before they leave, "
        "not just maximise overall accuracy."
    ),
}


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols     = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def _evaluate_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    # y_test is 0/1 encoded; positive class is 1
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def run_comparison() -> pd.DataFrame:
    print(f"Loading  {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    # Encode target to 0/1 — required by XGBoost; harmless for sklearn estimators.
    y = (df[TARGET_COL] == POS_LABEL).astype(int)
    pos_rate = y.mean()
    print(f"  Shape : {df.shape[0]:,} rows x {X.shape[1]} features  |  positive rate: {pos_rate:.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # XGBoost scale_pos_weight: ratio of negative to positive samples in training set.
    # Tells the model to treat each Churn=Yes example as ~2.8x more important.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = round(neg_count / pos_count, 2)
    print(f"  XGBoost scale_pos_weight = {spw}  ({neg_count} No / {pos_count} Yes in train)\n")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            verbosity=0,
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, estimator in models.items():
        print(f"[{name}]")
        t0 = time.time()

        pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(X_train)),
            ("model", estimator),
        ])

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV ROC-AUC : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        pipeline.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        m = _evaluate_test(pipeline, X_test, y_test)
        print(f"  Test       : accuracy={m['accuracy']}  precision={m['precision']}  "
              f"recall={m['recall']}  f1={m['f1']}  roc_auc={m['roc_auc']}")

        save_path = MODELS_DIR / f"{_slug(name)}.pkl"
        joblib.dump(pipeline, save_path)
        print(f"  Saved -> {save_path}  ({elapsed}s)\n")

        rows.append({
            "model":        name,
            "cv_mean":      round(cv_scores.mean(), 4),
            "cv_std":       round(cv_scores.std(), 4),
            "test_roc_auc": m["roc_auc"],
            "test_f1":      m["f1"],
            "test_recall":  m["recall"],
            "train_time_s": elapsed,
        })

    results = (
        pd.DataFrame(rows)
        .sort_values(["cv_mean", "test_recall"], ascending=False)
        .reset_index(drop=True)
    )
    return results


def _print_table(df: pd.DataFrame) -> None:
    col_w = {col: max(len(col), df[col].astype(str).str.len().max()) for col in df.columns}
    header = "  ".join(col.ljust(col_w[col]) for col in df.columns)
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print("  ".join(str(v).ljust(col_w[c]) for c, v in row.items()))
    print(sep)
    # Baseline reference row
    baseline_row = {
        "model": "Logistic Regression (baseline)",
        "cv_mean": "--",
        "cv_std": "--",
        "test_roc_auc": BASELINE_ROC_AUC,
        "test_f1": "0.5928",
        "test_recall": "0.5508",
        "train_time_s": "0.23",
    }
    print("  ".join(str(baseline_row[c]).ljust(col_w[c]) for c in df.columns))


def _print_winner(df: pd.DataFrame) -> None:
    winner = df.iloc[0]
    delta  = round(float(winner["test_roc_auc"]) - BASELINE_ROC_AUC, 4)
    sign   = "+" if delta >= 0 else ""

    print(f"\nBest model   : {winner['model']}")
    print(f"CV ROC-AUC   : {winner['cv_mean']} +/- {winner['cv_std']}")
    print(f"Test ROC-AUC : {winner['test_roc_auc']}  ({sign}{delta} vs baseline {BASELINE_ROC_AUC})")
    print(f"Test Recall  : {winner['test_recall']}  (proportion of actual churners caught)")
    print(f"\nWhy this model fits:")
    print(f"  {_MODEL_RATIONALE.get(winner['model'], '')}")


if __name__ == "__main__":
    results = run_comparison()

    print("=" * 72)
    print("MODEL COMPARISON  (sorted by CV ROC-AUC, baseline appended for reference)")
    print("=" * 72)
    _print_table(results)
    _print_winner(results)
