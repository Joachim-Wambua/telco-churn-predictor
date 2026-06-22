import json
import logging
import sys
import tempfile
import time
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "data"
MODELS_DIR    = ROOT / "models"

FEATURES_PATH    = DATA_DIR   / "features.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
PRODUCTION_PATH  = MODELS_DIR / "production_model.pkl"

TARGET_COL   = "Churn"
POS_LABEL    = "Yes"
DROP_COLS    = ["customerID"]
RANDOM_STATE = 42
TEST_SIZE    = 0.2
EXPERIMENT    = "telco-customer-churn"
MLFLOW_DB     = ROOT / "mlflow.db"   # SQLite backend; file store is deprecated in MLflow 3.x

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Preprocessing ─────────────────────────────────────────────────────────────

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


# ── Model configs ──────────────────────────────────────────────────────────────

def _build_configs(y_train: pd.Series) -> list[dict]:
    """Return one config dict per model run. Built after the split so
    scale_pos_weight is computed from the actual training labels."""
    spw = round((y_train == 0).sum() / (y_train == 1).sum(), 4)

    best_params = json.loads(BEST_PARAMS_PATH.read_text())

    return [
        {
            "run_name":  "baseline",
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "mlflow_params": {
                "model_type":    "LogisticRegression",
                "max_iter":      1000,
                "class_weight":  "none",
                "random_state":  RANDOM_STATE,
            },
        },
        {
            "run_name":  "tuned_xgboost",
            "estimator": XGBClassifier(**best_params),
            "mlflow_params": {
                "model_type": "XGBClassifier",
                # log the tuned params verbatim (skip internal-only keys)
                **{k: v for k, v in best_params.items()
                   if k not in ("eval_metric", "verbosity")},
            },
        },
    ]


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, prefix: str) -> dict:
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    return {
        f"{prefix}accuracy":  round(accuracy_score(y, y_pred), 4),
        f"{prefix}precision": round(precision_score(y, y_pred, pos_label=1, zero_division=0), 4),
        f"{prefix}recall":    round(recall_score(y, y_pred, pos_label=1, zero_division=0), 4),
        f"{prefix}f1":        round(f1_score(y, y_pred, pos_label=1, zero_division=0), 4),
        f"{prefix}roc_auc":   round(roc_auc_score(y, y_prob), 4),
    }


# ── Main training loop ─────────────────────────────────────────────────────────

def run_training() -> None:
    t_total = time.time()

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == POS_LABEL).astype(int)
    logger.info("Shape: %d rows x %d features | positive rate: %.1f%%",
                len(df), X.shape[1], y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT)
    logger.info("MLflow experiment: '%s'  db: %s", EXPERIMENT, MLFLOW_DB)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    configs  = _build_configs(y_train)
    results  = []
    best_pipeline = None

    for cfg in configs:
        run_name = cfg["run_name"]
        logger.info("── Starting run: %s", run_name)
        t_run = time.time()

        with mlflow.start_run(run_name=run_name):

            # ── Train ──────────────────────────────────────────────────────
            pipeline = Pipeline([
                ("preprocessor", _build_preprocessor(X_train)),
                ("model",        cfg["estimator"]),
            ])
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - t_run, 2)

            # ── Metrics ────────────────────────────────────────────────────
            train_metrics = _compute_metrics(pipeline, X_train, y_train, "train_")
            test_metrics  = _compute_metrics(pipeline, X_test,  y_test,  "test_")

            # ── Log to MLflow ──────────────────────────────────────────────
            mlflow.set_tag("model_name", run_name)
            mlflow.log_params({**cfg["mlflow_params"], "train_time_s": train_time})
            mlflow.log_metrics({**train_metrics, **test_metrics})

            # Log sklearn pipeline as a native MLflow model artifact.
            # skops_trusted_types is required in MLflow 3.x when the pipeline
            # contains XGBoost estimators (skops security model requires explicit trust).
            mlflow.sklearn.log_model(
                pipeline,
                name="pipeline",
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )

            # Also persist the pipeline as a joblib file and attach it
            with tempfile.TemporaryDirectory() as tmp:
                pkl_path = Path(tmp) / f"{run_name}.pkl"
                joblib.dump(pipeline, pkl_path)
                mlflow.log_artifact(str(pkl_path), artifact_path="joblib")

            run_id = mlflow.active_run().info.run_id

        # ── Console summary ────────────────────────────────────────────────
        test_auc = test_metrics["test_roc_auc"]
        test_f1  = test_metrics["test_f1"]
        test_rec = test_metrics["test_recall"]
        logger.info(
            "%s | test roc_auc=%.4f  f1=%.4f  recall=%.4f  [%s]  (%.2fs)",
            run_name, test_auc, test_f1, test_rec, run_id[:8], train_time,
        )

        results.append({
            "run_name":       run_name,
            "run_id":         run_id[:8],
            "test_roc_auc":   test_auc,
            "test_f1":        test_f1,
            "test_recall":    test_rec,
            "train_time_s":   train_time,
        })

        # Keep the tuned XGBoost as the production model
        if run_name == "tuned_xgboost":
            best_pipeline = pipeline

    # ── Persist production model ───────────────────────────────────────────────
    if best_pipeline is not None:
        joblib.dump(best_pipeline, PRODUCTION_PATH)
        logger.info("Production model saved -> %s", PRODUCTION_PATH)

    # ── Final table ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 66)
    print("RUN SUMMARY")
    print("=" * 66)
    col_w = {c: max(len(c), results_df[c].astype(str).str.len().max()) for c in results_df.columns}
    header = "  ".join(c.ljust(col_w[c]) for c in results_df.columns)
    print(header)
    print("-" * len(header))
    for _, row in results_df.iterrows():
        print("  ".join(str(v).ljust(col_w[c]) for c, v in row.items()))
    print("=" * 66)
    print(f"\nMLflow UI  ->  mlflow server --backend-store-uri sqlite:///{MLFLOW_DB} --host 127.0.0.1 --port 5000")
    print(f"               then open  http://localhost:5000")
    print(f"\nTotal time : {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    run_training()
