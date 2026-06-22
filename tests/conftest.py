import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"


@pytest.fixture(scope="session")
def cleaned_df():
    return pd.read_csv(DATA_DIR / "cleaned.csv")


@pytest.fixture(scope="session")
def features_df():
    return pd.read_csv(DATA_DIR / "features.csv")


@pytest.fixture(scope="session")
def minimal_raw_df():
    """Minimal DataFrame with all columns required by create_features."""
    return pd.DataFrame({
        "tenure":           [1, 12, 24, 48, 60],
        "MonthlyCharges":   [29.85, 56.95, 42.30, 79.65, 104.80],
        "TotalCharges":     [29.85, 683.40, 1015.20, 3823.20, 6288.00],
        "Contract":         ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month"],
        "InternetService":  ["DSL", "Fiber optic", "No", "Fiber optic", "DSL"],
        "PaymentMethod":    [
            "Electronic check",
            "Bank transfer (automatic)",
            "Mailed check",
            "Credit card (automatic)",
            "Electronic check",
        ],
        "OnlineSecurity":   ["No", "Yes", "No internet service", "Yes", "No"],
        "OnlineBackup":     ["Yes", "No", "No internet service", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No internet service", "No", "No"],
        "TechSupport":      ["No", "No", "No internet service", "Yes", "No"],
        "StreamingTV":      ["No", "Yes", "No internet service", "No", "No"],
        "StreamingMovies":  ["No", "Yes", "No internet service", "Yes", "No"],
    })
