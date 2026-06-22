import pandas as pd
import pytest

from src.features.engineering import create_features

# Columns added by create_features — update if the function changes.
ENGINEERED_COLUMNS = [
    "is_month_to_month",
    "num_addon_services",
    "has_fiber_optic",
    "uses_auto_payment",
    "is_new_customer",
    "avg_monthly_spend",
    "tenure_stage",
    "monthly_charges_vs_median",
    "high_charge_x_no_contract",
    "stickiness_score",
    "charge_per_service",
]


@pytest.fixture(scope="module")
def featured_df(minimal_raw_df):
    return create_features(minimal_raw_df)


def test_feature_engineering_adds_expected_columns(minimal_raw_df, featured_df):
    new_cols = [c for c in featured_df.columns if c not in minimal_raw_df.columns]
    assert set(new_cols) == set(ENGINEERED_COLUMNS), (
        f"Unexpected columns: {set(new_cols) - set(ENGINEERED_COLUMNS)}\n"
        f"Missing columns:    {set(ENGINEERED_COLUMNS) - set(new_cols)}"
    )


def test_feature_engineering_total_column_count(minimal_raw_df, featured_df):
    expected = len(minimal_raw_df.columns) + len(ENGINEERED_COLUMNS)
    assert featured_df.shape[1] == expected


def test_no_nan_in_engineered_columns(featured_df):
    nan_cols = [c for c in ENGINEERED_COLUMNS if featured_df[c].isna().any()]
    assert nan_cols == [], f"NaN found in engineered columns: {nan_cols}"


def test_binary_features_are_zero_or_one(featured_df):
    binary_cols = [
        "is_month_to_month",
        "has_fiber_optic",
        "uses_auto_payment",
        "is_new_customer",
    ]
    for col in binary_cols:
        unique_vals = set(featured_df[col].unique())
        assert unique_vals <= {0, 1}, f"'{col}' contains non-binary values: {unique_vals}"


def test_num_addon_services_range(featured_df):
    col = featured_df["num_addon_services"]
    assert col.min() >= 0, "num_addon_services must be non-negative"
    assert col.max() <= 6, "num_addon_services cannot exceed 6 (number of addon service columns)"


def test_tenure_stage_values(featured_df):
    valid = {0, 1, 2}
    actual = set(featured_df["tenure_stage"].unique())
    assert actual <= valid, f"tenure_stage contains unexpected values: {actual - valid}"


def test_avg_monthly_spend_nonnegative(featured_df):
    assert (featured_df["avg_monthly_spend"] >= 0).all(), (
        "avg_monthly_spend must be non-negative (TotalCharges / (tenure + 1))"
    )


def test_charge_per_service_positive(featured_df):
    assert (featured_df["charge_per_service"] > 0).all(), (
        "charge_per_service must be positive (MonthlyCharges / (num_addon_services + 1))"
    )


def test_is_new_customer_matches_tenure(featured_df):
    # Customers with tenure <= 3 must be flagged as new.
    mask_new = featured_df["tenure"] <= 3
    assert (featured_df.loc[mask_new, "is_new_customer"] == 1).all()
    assert (featured_df.loc[~mask_new, "is_new_customer"] == 0).all()


def test_stickiness_score_is_product(featured_df):
    expected = featured_df["tenure"] * featured_df["num_addon_services"]
    pd.testing.assert_series_equal(
        featured_df["stickiness_score"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
