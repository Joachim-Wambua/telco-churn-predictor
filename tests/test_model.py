from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

TARGET_COL = "Churn"
DROP_COLS  = ["customerID"]


@pytest.fixture(scope="module")
def model():
    path = MODELS_DIR / "production_model.pkl"
    assert path.exists(), f"Model not found at {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def X_sample(features_df):
    df = features_df.drop(columns=[c for c in DROP_COLS + [TARGET_COL] if c in features_df.columns])
    return df.head(20)


def test_model_loads(model):
    assert model is not None


def test_model_has_predict_method(model):
    assert callable(getattr(model, "predict", None))


def test_model_has_predict_proba_method(model):
    assert callable(getattr(model, "predict_proba", None))


def test_model_predict_returns_correct_shape(model, X_sample):
    preds = model.predict(X_sample)
    assert len(preds) == len(X_sample)


def test_model_predict_proba_shape(model, X_sample):
    proba = model.predict_proba(X_sample)
    assert proba.shape == (len(X_sample), 2), (
        f"Expected shape ({len(X_sample)}, 2), got {proba.shape}"
    )


def test_model_predict_proba_in_range(model, X_sample):
    proba = model.predict_proba(X_sample)
    assert (proba >= 0).all(), "Probabilities must be >= 0"
    assert (proba <= 1).all(), "Probabilities must be <= 1"


def test_model_predict_proba_sums_to_one(model, X_sample):
    proba = model.predict_proba(X_sample)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
                               err_msg="Each row's probabilities must sum to 1")


def test_model_predictions_are_valid_classes(model, X_sample):
    preds = model.predict(X_sample)
    valid_classes = set(model.classes_)
    assert set(preds) <= valid_classes, (
        f"Unexpected prediction values: {set(preds) - valid_classes}"
    )


def test_model_predicts_both_classes(model, X_sample):
    # With 20 samples from a real dataset the model should predict at least one
    # positive (Churn=Yes) and one negative — a degenerate model would always
    # predict the majority class.
    preds = model.predict(X_sample)
    assert len(set(preds)) > 1, (
        f"Model predicts only one class ({set(preds)}) across 20 samples"
    )
