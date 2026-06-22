import json
import logging
import sys
import time
import joblib
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
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
FEATURES_PATH = DATA_DIR   / "features.csv"
PARAMS_PATH   = MODELS_DIR / "best_params.json"
MODEL_PATH    = MODELS_DIR / "tuned_model.pkl"

TARGET_COL   = "Churn"
POS_LABEL    = "Yes"
DROP_COLS    = ["customerID"]
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
N_TRIALS     = 30

# Untuned XGBoost scores from compare_models.py — printed as reference at the end
_UNTUNED = {"roc_auc": 0.8310, "f1": 0.6131, "recall": 0.7647, "precision": 0.5116}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial Optuna noise


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


def _make_objective(X_train, y_train, preprocessor, cv, spw):
    """Return an Optuna objective that closes over training data and the CV splitter."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                **params,
                scale_pos_weight=spw,
                random_state=RANDOM_STATE,
                eval_metric="auc",
                verbosity=0,
            )),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mean_auc, std_auc = scores.mean(), scores.std()

        logger.info(
            "Trial %3d | CV=%.4f+/-%.4f | "
            "n_est=%3d  depth=%d  lr=%.4f  sub=%.2f  col=%.2f  "
            "alpha=%.1e  lambda=%.1e  mcw=%d",
            trial.number, mean_auc, std_auc,
            params["n_estimators"], params["max_depth"], params["learning_rate"],
            params["subsample"], params["colsample_bytree"],
            params["reg_alpha"], params["reg_lambda"], params["min_child_weight"],
        )

        return mean_auc

    return objective


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def run_tuning() -> dict:
    t_start = time.time()

    # ── Data ─────────────────────────────────────────────────────────────────
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

    spw = round((y_train == 0).sum() / (y_train == 1).sum(), 4)
    logger.info("scale_pos_weight fixed at %.2f", spw)

    # ── Optuna study ──────────────────────────────────────────────────────────
    preprocessor = _build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logger.info("Starting Optuna study — %d trials, %d-fold CV, metric=roc_auc", N_TRIALS, CV_FOLDS)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        _make_objective(X_train, y_train, preprocessor, cv, spw),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best_trial  = study.best_trial
    best_cv_auc = round(best_trial.value, 4)
    best_params = {
        **best_trial.params,
        "scale_pos_weight": spw,
        "random_state":     RANDOM_STATE,
        "eval_metric":      "auc",
        "verbosity":        0,
    }

    logger.info("Best trial: #%d | CV ROC-AUC=%.4f", best_trial.number, best_cv_auc)
    logger.info("Best hyperparameters: %s", best_trial.params)

    # ── Save best params ──────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Convert numpy scalars to native Python types for JSON serialisation
    serialisable = {
        k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
        for k, v in best_params.items()
    }
    with open(PARAMS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Best params saved -> %s", PARAMS_PATH)

    # ── Final model ───────────────────────────────────────────────────────────
    logger.info("Training final model on full training set ...")
    t_fit = time.time()
    final_pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(X_train)),
        ("model", XGBClassifier(**best_params)),
    ])
    final_pipeline.fit(X_train, y_train)
    logger.info("Trained in %.2fs", time.time() - t_fit)

    metrics = _evaluate(final_pipeline, X_test, y_test)

    joblib.dump(final_pipeline, MODEL_PATH)
    logger.info("Tuned model saved -> %s", MODEL_PATH)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("TUNING RESULTS")
    print("=" * 58)
    print(f"  Trials run       : {N_TRIALS}")
    print(f"  Best trial       : #{best_trial.number}")
    print(f"  Best CV ROC-AUC  : {best_cv_auc}")
    print()

    col_w = max(len(k) for k in metrics) + 2
    header = f"  {'Metric':<{col_w}}  {'Tuned':>8}  {'Untuned':>8}  {'Delta':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k, v in metrics.items():
        untuned = _UNTUNED.get(k, "--")
        delta   = f"{v - untuned:+.4f}" if isinstance(untuned, float) else "  --"
        print(f"  {k:<{col_w}}  {v:>8}  {str(untuned):>8}  {delta:>8}")

    print()
    print("  Best hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"    {k:<20} {v}")

    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    print("=" * 58)

    return metrics


if __name__ == "__main__":
    run_tuning()
