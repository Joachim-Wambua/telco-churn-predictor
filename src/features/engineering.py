import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.cleaner import clean_data
from src.data.loader import DATA_DIR, load_data

_ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

_AUTO_PAYMENT_METHODS = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges arrives as a blank string for tenure==0 customers in raw data;
    # coerce here so this function is safe to call on either raw or cleaned input.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # =========================================================================
    # Category 1: Domain-specific (Telco churn drivers)
    # =========================================================================

    # Month-to-month customers have zero switching cost — they can leave at any
    # billing cycle. EDA showed 42.7% churn rate vs 2.8% for two-year contracts,
    # making this the single strongest churn signal in the dataset.
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # Each add-on service raises the practical cost of leaving: a churning customer
    # must find, subscribe to, and re-configure every service they drop. More
    # services = higher switching cost = lower churn probability.
    addon_flags = [
        df[col].isin(["Yes"]).astype(int)
        for col in _ADDON_SERVICES
        if col in df.columns
    ]
    df["num_addon_services"] = sum(addon_flags) if addon_flags else pd.Series(0, index=df.index)

    # Fiber optic customers pay a premium and face stronger competition from cable
    # and 5G home internet alternatives, making them more price-sensitive and
    # more likely to shop around when dissatisfied.
    df["has_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)

    # Automatic payment customers are in "set and forget" mode — they rarely
    # review their subscription actively, which suppresses churn.
    df["uses_auto_payment"] = df["PaymentMethod"].isin(_AUTO_PAYMENT_METHODS).astype(int)

    # The first 3 months are the highest-risk onboarding window: customers who
    # haven't yet built service habits or integrated the product into their
    # routine are far more likely to cancel early.
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

    # =========================================================================
    # Category 2: Statistical transformations
    # =========================================================================

    # Normalised spend per month of tenure. Raw TotalCharges is nearly perfectly
    # collinear with tenure * MonthlyCharges (corr ≈ 0.83), so including it
    # directly risks multicollinearity. This ratio captures effective price
    # experience without duplicating the tenure signal.
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Tenure split into lifecycle stages: new (≤ 12 months), developing (13–36),
    # and mature (> 36). The churn–tenure relationship is non-linear — risk drops
    # sharply after the first year and again past three years.
    df["tenure_stage"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 36, float("inf")],
        labels=[0, 1, 2],
    ).astype(float).astype(int)

    # Deviation from median monthly charge. Customers paying well above the median
    # are more likely to price-compare with competitors; those below are in
    # lower-cost plans with less financial motivation to switch.
    df["monthly_charges_vs_median"] = df["MonthlyCharges"] - df["MonthlyCharges"].median()

    # =========================================================================
    # Category 3: Interaction features (products / ratios)
    # =========================================================================

    # High charges × no contract commitment = maximum churn risk. Neither alone
    # captures the combined pressure: a cheap month-to-month customer may stay
    # out of inertia; an expensive two-year customer can't easily leave. The
    # product isolates the high-cost, low-friction churner.
    df["high_charge_x_no_contract"] = df["MonthlyCharges"] * df["is_month_to_month"]

    # Long tenure × many services = compound stickiness. Every additional month
    # deepens habits; every additional service raises switching effort. Their
    # product captures customers who are hard to lose on both dimensions at once.
    df["stickiness_score"] = df["tenure"] * df["num_addon_services"]

    # Monthly charge per subscribed service — a value-perception proxy. A high
    # ratio means the customer pays a lot relative to what they use, signalling
    # potential dissatisfaction with price-to-value.
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_addon_services"] + 1)

    return df


if __name__ == "__main__":
    raw_df = load_data(DATA_DIR / "Telco-Customer-Churn.csv")
    cleaned_df, _ = clean_data(raw_df, target_column="Churn")

    featured_df = create_features(cleaned_df)

    new_feature_cols = featured_df.columns[cleaned_df.shape[1]:].tolist()
    print(f"Original columns    : {cleaned_df.shape[1]}")
    print(f"Engineered features : {len(new_feature_cols)}")
    print(f"Total columns       : {featured_df.shape[1]}")
    print("\nNew features:")
    for col in new_feature_cols:
        print(f"  {col}")
    print("\nSample (first 3 rows, new features only):")
    print(featured_df[new_feature_cols].head(3).to_string())
