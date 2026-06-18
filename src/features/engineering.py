import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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


_CORR_THRESHOLD = 0.95
_VARIANCE_THRESHOLD_RATIO = 0.01  # drop features whose variance < 1% of overall variance


def select_features(
    df: pd.DataFrame,
    corr_threshold: float = _CORR_THRESHOLD,
    variance_threshold_ratio: float = _VARIANCE_THRESHOLD_RATIO,
) -> tuple[list[str], pd.DataFrame]:
    """Return (selected_feature_names, reduced_df) after dropping collinear and near-zero-variance features."""
    numeric_df = df.select_dtypes(include="number")
    candidates = numeric_df.columns.tolist()

    dropped_corr: dict[str, str] = {}
    dropped_var: dict[str, float] = {}

    # --- Pass 1: collinearity ---
    # Walk the upper triangle of the correlation matrix. When two features exceed
    # the threshold, drop the second one (later column) and keep the first. This
    # preserves the feature that appeared earlier, which for this dataset means
    # original columns are preferred over engineered ones.
    corr_matrix = numeric_df.corr().abs()
    keep = list(candidates)
    for i, col_a in enumerate(candidates):
        if col_a not in keep:
            continue
        for col_b in candidates[i + 1:]:
            if col_b not in keep:
                continue
            if corr_matrix.loc[col_a, col_b] > corr_threshold:
                keep.remove(col_b)
                dropped_corr[col_b] = col_a
                logger.info(
                    "Dropped '%s' (corr=%.3f with '%s' > threshold %.2f)",
                    col_b, corr_matrix.loc[col_a, col_b], col_a, corr_threshold,
                )

    # --- Pass 2: near-zero variance ---
    # A feature whose variance is a tiny fraction of the dataset's overall numeric
    # spread carries almost no discriminating information. The threshold is relative
    # so it scales automatically when features are on different units.
    variances = numeric_df[keep].var()
    # Use median rather than mean so that one high-scale feature (e.g. TotalCharges
    # with variance ~400k) cannot inflate the cutoff and incorrectly kill lower-scale
    # but perfectly meaningful features like tenure or MonthlyCharges.
    overall_variance = variances.median()
    variance_cutoff = variance_threshold_ratio * overall_variance
    low_var_cols = variances[variances < variance_cutoff].index.tolist()
    for col in low_var_cols:
        keep.remove(col)
        dropped_var[col] = float(variances[col])
        logger.info(
            "Dropped '%s' (variance=%.6f < cutoff %.6f)",
            col, variances[col], variance_cutoff,
        )

    # Rebuild df with non-numeric columns + surviving numeric columns, preserving
    # original column order as closely as possible.
    non_numeric_cols = [c for c in df.columns if c not in numeric_df.columns]
    selected = [c for c in df.columns if c in non_numeric_cols or c in keep]

    logger.info(
        "Feature selection complete — kept %d / %d numeric features "
        "(%d dropped for collinearity, %d for low variance)",
        len(keep), len(candidates), len(dropped_corr), len(dropped_var),
    )

    return keep, df[selected]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

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

    print("\n--- Feature Selection ---")
    selected_cols, reduced_df = select_features(featured_df)
    dropped = [c for c in featured_df.select_dtypes(include="number").columns if c not in selected_cols]
    print(f"Numeric features before : {featured_df.select_dtypes(include='number').shape[1]}")
    print(f"Numeric features after  : {len(selected_cols)}")
    print(f"Dropped ({len(dropped)})            : {dropped}")
    print(f"Reduced dataframe shape : {reduced_df.shape}")
