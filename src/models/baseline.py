import sys
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

FEATURES_PATH = DATA_DIR   / "features.csv"
OUTPUT_PATH   = MODELS_DIR / "baseline.pkl"

TARGET_COL = "Churn"
DROP_COLS  = ["customerID"]


def _detect_task(y: pd.Series) -> str:
    return "classification" if (y.dtype == object or y.nunique() <= 10) else "regression"


def _build_pipeline(X: pd.DataFrame, task: str) -> Pipeline:
    numeric_cols     = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    # max_iter raised from default 100 to avoid ConvergenceWarning on this dataset;
    # all other hyperparameters are sklearn defaults (this is the baseline).
    estimator = (
        LogisticRegression(max_iter=1000)
        if task == "classification"
        else LinearRegression()
    )

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _eval_classification(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    classes   = pipeline.named_steps["model"].classes_
    # Use the minority class as the positive label — for Churn that's "Yes".
    pos_label = y_test.value_counts().idxmin()
    pos_idx   = list(classes).index(pos_label)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, pos_idx]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score((y_test == pos_label).astype(int), y_prob), 4),
    }


def _eval_regression(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_test, y_pred) ** 0.5), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }


def run_baseline(
    features_path: Path = FEATURES_PATH,
    output_path: Path   = OUTPUT_PATH,
    target_col: str     = TARGET_COL,
    test_size: float    = 0.2,
    random_state: int   = 42,
) -> dict:
    t0 = time.time()

    print(f"Loading  {features_path}")
    df = pd.read_csv(features_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} cols")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    task = _detect_task(y)
    print(f"  Task  : {task}  |  target='{target_col}'  |  classes={sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task == "classification" else None,
    )
    print(f"\nSplit    train={len(X_train):,}  test={len(X_test):,}  (stratified)")

    print("Training ...")
    pipeline = _build_pipeline(X_train, task)
    pipeline.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.2f}s")

    print("\nEvaluation on test set:")
    metrics = (
        _eval_classification(pipeline, X_test, y_test)
        if task == "classification"
        else _eval_regression(pipeline, X_test, y_test)
    )
    col_w = max(len(k) for k in metrics) + 2
    for name, val in metrics.items():
        print(f"  {name:<{col_w}}{val}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\nSaved -> {output_path}")
    print(f"Done in {time.time() - t0:.2f}s")

    return metrics


if __name__ == "__main__":
    run_baseline()
