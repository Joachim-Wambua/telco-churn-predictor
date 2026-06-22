"""
ChurnScope — Telecom Customer Retention Intelligence
app/streamlit_app.py

A 4-page portfolio app showcasing an end-to-end churn prediction pipeline:
  Page 1 · Project Overview
  Page 2 · Explore the Data
  Page 3 · Model Results
  Page 4 · How I Built This

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))

# ── Page config — must be the first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="ChurnScope | Telecom Retention Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
PRIMARY   = "#1E40AF"
ACCENT    = "#3B82F6"
SUCCESS   = "#059669"
DANGER    = "#DC2626"
AMBER     = "#D97706"
MUTED     = "#6B7280"

CHURN_PALETTE = {"Retained": ACCENT, "Churned": DANGER}

TECH_STACK = [
    ("Python 3.12", "#3776AB"),
    ("scikit-learn", "#F7931E"),
    ("XGBoost",      "#189A0C"),
    ("Optuna",       "#5764C6"),
    ("MLflow",       "#0194E2"),
    ("Pandas",       "#150458"),
    ("Plotly",       "#3F4F75"),
    ("Streamlit",    "#FF4B4B"),
]

PLOTLY_THEME = "plotly_white"

# Hardcoded results used when data/model_results.json hasn't been generated yet
_DEMO_MODEL_RESULTS = [
    {"name": "Logistic Regression", "label": "Baseline",
     "cv_mean": None,   "cv_std": None,
     "test_accuracy": 0.7989, "test_precision": 0.6417, "test_recall": 0.5508,
     "test_f1": 0.5928, "test_roc_auc": 0.8387, "train_time_s": 0.23, "winner": False},
    {"name": "Random Forest",       "label": "Candidate",
     "cv_mean": 0.8302, "cv_std": 0.0065,
     "test_accuracy": 0.7846, "test_precision": 0.6228, "test_recall": 0.4813,
     "test_f1": 0.5430, "test_roc_auc": 0.8140, "train_time_s": 14.81, "winner": False},
    {"name": "Gradient Boosting",   "label": "Candidate",
     "cv_mean": 0.8402, "cv_std": 0.0046,
     "test_accuracy": 0.7846, "test_precision": 0.6073, "test_recall": 0.5374,
     "test_f1": 0.5702, "test_roc_auc": 0.8309, "train_time_s": 20.18, "winner": False},
    {"name": "XGBoost (Tuned)",     "label": "Winner",
     "cv_mean": 0.8513, "cv_std": 0.0048,
     "test_accuracy": 0.7342, "test_precision": 0.5000, "test_recall": 0.8075,
     "test_f1": 0.6176, "test_roc_auc": 0.8403, "train_time_s": 1.05,  "winner": True},
]

# ── CSS ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hero */
.cs-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
    padding: 2.75rem 2.5rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 1.75rem;
    box-shadow: 0 8px 32px rgba(30,64,175,0.25);
}
.cs-hero h1 {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin: 0 0 0.4rem;
}
.cs-hero .subtitle {
    font-size: 1.15rem;
    opacity: 0.8;
    letter-spacing: 0.5px;
    margin: 0;
}
.cs-hero .tagline {
    font-size: 0.9rem;
    opacity: 0.6;
    margin-top: 0.6rem;
    font-style: italic;
}

/* Section labels */
.cs-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.3rem;
}
.cs-section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

/* Callout boxes */
.cs-callout {
    padding: 1rem 1.25rem;
    border-radius: 0 10px 10px 0;
    margin: 0.5rem 0;
    line-height: 1.6;
}
.cs-callout-blue   { background:#eff6ff; border-left:4px solid #3b82f6; }
.cs-callout-green  { background:#f0fdf4; border-left:4px solid #10b981; }
.cs-callout-amber  { background:#fffbeb; border-left:4px solid #f59e0b; }
.cs-callout-red    { background:#fef2f2; border-left:4px solid #ef4444; }

/* Tech badges */
.cs-badge-row { display:flex; flex-wrap:wrap; gap:8px; margin:0.5rem 0 1.5rem; }
.cs-badge {
    display:inline-block;
    padding:5px 14px;
    border-radius:20px;
    font-size:13px;
    font-weight:600;
    color:white;
    letter-spacing:0.3px;
}

/* Comparison cards (winner rationale) */
.cs-win-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center;
    height: 100%;
}
.cs-win-card .metric { font-size: 2rem; font-weight: 800; }
.cs-win-card .label  { color: #6b7280; font-size: 0.85rem; margin-top: 0.25rem; }

/* Timeline */
.cs-tl-item {
    border-left: 3px solid #3b82f6;
    padding: 0.75rem 1.25rem;
    margin: 0.4rem 0;
    background: #f8fafc;
    border-radius: 0 10px 10px 0;
}
.cs-tl-day   { font-size: 0.75rem; font-weight: 700; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.8px; }
.cs-tl-title { font-weight: 700; color: #0f172a; margin: 0.1rem 0; }
.cs-tl-desc  { color: #6b7280; font-size: 0.88rem; }

/* Decision items */
.cs-decision {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.4rem 0;
    border: 1px solid #e2e8f0;
}
.cs-decision strong { color: #1e40af; }

/* Footer */
.cs-footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    color: #9ca3af;
    font-size: 0.82rem;
    border-top: 1px solid #e5e7eb;
    margin-top: 3rem;
}
</style>
"""

# ── Demo data generators ────────────────────────────────────────────────────────

def _make_demo_raw_data() -> pd.DataFrame:
    """Synthetic IBM Telco-like dataset matching known column schema and distributions."""
    rng = np.random.default_rng(42)
    n   = 7032

    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
    )
    churn_prob = np.where(
        contracts == "Month-to-month", 0.427,
        np.where(contracts == "One year", 0.113, 0.028),
    )
    churn   = rng.random(n) < churn_prob
    tenure  = np.clip(
        np.where(churn, rng.exponential(20, n), rng.exponential(40, n)).astype(int),
        0, 72,
    )
    monthly = rng.normal(65, 30, n).clip(18, 120).round(2)
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    payment  = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.22, 0.22, 0.22],
    )
    yn = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1 - p])  # noqa: E731
    return pd.DataFrame({
        "customerID":       [f"DEMO-{i:04d}" for i in range(n)],
        "gender":           rng.choice(["Male", "Female"], n),
        "SeniorCitizen":    rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":          yn(0.48),
        "Dependents":       yn(0.30),
        "tenure":           tenure,
        "PhoneService":     yn(0.90),
        "MultipleLines":    rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService":  internet,
        "OnlineSecurity":   rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.50, 0.21]),
        "OnlineBackup":     rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.21]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.21]),
        "TechSupport":      rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.21]),
        "StreamingTV":      rng.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.21]),
        "StreamingMovies":  rng.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.40, 0.21]),
        "Contract":         contracts,
        "PaperlessBilling": yn(0.59),
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly,
        "TotalCharges":     (monthly * tenure).round(2),
        "Churn":            np.where(churn, "Yes", "No"),
        "_is_demo":         True,
    })


def _make_demo_predictions() -> pd.DataFrame:
    """Synthetic test-set predictions whose confusion matrix matches known model metrics.

    Targets: 1,407 rows | 374 churners | recall=0.807 | precision=0.50 | ROC-AUC≈0.84
    """
    rng   = np.random.default_rng(42)
    # True positives (302), false negatives (72), true negatives (731), false positives (302)
    tp    = np.clip(rng.beta(7, 2.5, 302) * 0.5 + 0.5,  0.50, 1.0)
    fn    = np.clip(rng.beta(2.5, 6,    72) * 0.5,       0.0,  0.49)
    tn    = np.clip(rng.beta(2,   7,   731) * 0.5,       0.0,  0.49)
    fp    = np.clip(rng.beta(3,   5,   302) * 0.4 + 0.5, 0.50, 0.90)

    probs   = np.concatenate([tp, fn, tn, fp])
    actuals = np.concatenate([
        np.ones(302),  np.ones(72),    # churners
        np.zeros(731), np.zeros(302),  # retained
    ]).astype(int)

    idx = rng.permutation(len(probs))
    probs, actuals = probs[idx], actuals[idx]
    return pd.DataFrame({
        "actual":      actuals,
        "predicted":   (probs >= 0.5).astype(int),
        "probability": np.round(probs, 4),
        "_is_demo":    True,
    })


def _demo_feature_importance() -> pd.DataFrame:
    """Hardcoded top-15 importances from the actual trained XGBoost model."""
    return pd.DataFrame([
        ("is_month_to_month",              0.2243),
        ("Contract_Month-to-month",        0.2194),
        ("high_charge_x_no_contract",      0.0734),
        ("tenure",                         0.0523),
        ("stickiness_score",               0.0481),
        ("MonthlyCharges",                 0.0452),
        ("charge_per_service",             0.0398),
        ("TotalCharges",                   0.0341),
        ("PaymentMethod_Electronic check", 0.0287),
        ("InternetService_Fiber optic",    0.0265),
        ("Contract_Two year",              0.0198),
        ("num_addon_services",             0.0184),
        ("has_fiber_optic",                0.0171),
        ("tenure_stage",                   0.0158),
        ("Contract_One year",              0.0142),
    ], columns=["feature", "importance"])


def _heuristic_predict(
    contract: str, tenure: int, monthly_charges: float,
    internet_service: str, payment_method: str,
    senior_citizen: bool, partner: bool, num_addons: int,
) -> float:
    """Logistic-score fallback when production_model.pkl is unavailable.
    Coefficients derived from EDA churn rates and known model feature importances."""
    score  = -1.8
    score += 1.30 * (contract == "Month-to-month")
    score += 0.90 * (tenure <= 3)
    score += 0.40 * (3 < tenure <= 12)
    score += 0.70 * (internet_service == "Fiber optic")
    score += 0.50 * (monthly_charges > 70)
    score += 0.60 * (num_addons == 0)
    score += 0.40 * ("automatic" not in payment_method)
    score += 0.30 * int(senior_citizen)
    return float(1.0 / (1.0 + np.exp(-score)))


# ── Data loading ────────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data() -> pd.DataFrame:
    p = DATA_DIR / "Telco-Customer-Churn.csv"
    if not p.exists():
        return _make_demo_raw_data()
    df = pd.read_csv(p)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna(subset=["TotalCharges"])


@st.cache_data
def load_features() -> pd.DataFrame | None:
    p = DATA_DIR / "features.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_predictions() -> pd.DataFrame:
    p = DATA_DIR / "predictions.csv"
    return pd.read_csv(p) if p.exists() else _make_demo_predictions()


@st.cache_data
def load_model_results() -> list[dict]:
    p = DATA_DIR / "model_results.json"
    return json.loads(p.read_text())["models"] if p.exists() else _DEMO_MODEL_RESULTS


@st.cache_resource
def load_model():
    p = MODELS_DIR / "production_model.pkl"
    return joblib.load(p) if p.exists() else None


# ── Feature importance helper ───────────────────────────────────────────────────

def get_feature_importance(pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model        = pipeline.named_steps["model"]

    raw_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    def _clean(name: str) -> str:
        for prefix in ("num__", "cat__"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    df = pd.DataFrame({"feature": [_clean(n) for n in raw_names], "importance": importances})
    return df.nlargest(15, "importance").reset_index(drop=True)


# ── Input builder for "try it yourself" ────────────────────────────────────────

def build_prediction_row(
    contract: str, tenure: int, monthly_charges: float,
    internet_service: str, payment_method: str,
    senior_citizen: bool, partner: bool, num_addons: int,
) -> pd.DataFrame:
    is_mtm   = int(contract == "Month-to-month")
    is_fiber = int(internet_service == "Fiber optic")
    is_auto  = int("automatic" in payment_method)
    is_new   = int(tenure <= 3)
    t_stage  = 0 if tenure <= 12 else (1 if tenure <= 36 else 2)
    total_ch = monthly_charges * max(tenure, 1)

    return pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": int(senior_citizen),
        "Partner": "Yes" if partner else "No", "Dependents": "No",
        "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_ch,
        "is_month_to_month": is_mtm, "num_addon_services": num_addons,
        "has_fiber_optic": is_fiber, "uses_auto_payment": is_auto,
        "is_new_customer": is_new, "tenure_stage": t_stage,
        "high_charge_x_no_contract": monthly_charges * is_mtm,
        "stickiness_score": tenure * num_addons,
        "charge_per_service": monthly_charges / (num_addons + 1),
    }])


# ── Page 1 · Overview ─────────────────────────────────────────────────────────

def page_overview():
    st.markdown("""
    <div class="cs-hero">
        <h1>📡 ChurnScope</h1>
        <p class="subtitle">Telecom Customer Retention Intelligence</p>
        <p class="tagline">End-to-end ML pipeline · IBM Telco Churn · XGBoost + Optuna + MLflow</p>
    </div>
    """, unsafe_allow_html=True)

    # ── What this project does ─────────────────────────────────────────────────
    st.markdown("""
    <div class="cs-section-label">About</div>
    <div class="cs-section-title">What this project does</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ChurnScope predicts which telecom customers are likely to cancel their service,
    using a production-grade machine learning pipeline trained on IBM's Telco Customer Churn dataset.
    The system applies feature engineering, systematic model selection, and Optuna-tuned XGBoost
    to achieve **84% ROC-AUC** — catching **80.7% of actual churners** on held-out test data.
    Results are tracked end-to-end in MLflow, and the trained model powers this live portfolio demo.
    """)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cs-section-label">Results at a glance</div>
    <div class="cs-section-title">Key Numbers</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers analyzed",   "7,032",  help="IBM Telco dataset after removing 11 rows with blank TotalCharges")
    c2.metric("Features engineered",  "11",     help="Across 3 categories: domain flags, statistical, and interaction terms")
    c3.metric("Test ROC-AUC",         "0.8403", delta="+0.0016 vs baseline",  delta_color="normal")
    c4.metric("Test Recall (Churn=1)", "80.7%", delta="+25.7pp vs baseline",  delta_color="normal")

    st.caption("Recall improvement is the headline metric — catching churners before they leave is the business goal.")

    # ── Quick-facts callout ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="cs-callout cs-callout-blue">
        <strong>Dataset</strong> — IBM Telco Customer Churn (Kaggle)<br>
        7,043 rows &times; 21 columns &nbsp;|&nbsp; Binary target: Churn (Yes/No)<br>
        Class imbalance: 73.5% No / 26.5% Yes
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="cs-callout cs-callout-green">
        <strong>Primary metric</strong> — ROC-AUC (target &gt; 0.80)<br>
        Secondary: F1, Recall, Precision on the Churn=Yes class<br>
        Strategy: <code>scale_pos_weight</code> in XGBoost, not SMOTE
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="cs-callout cs-callout-amber">
        <strong>Tuning</strong> — Optuna TPE sampler, 30 trials, 5-fold stratified CV<br>
        Best trial #22 &nbsp;|&nbsp; CV ROC-AUC: 0.8513<br>
        8 hyperparameters searched (n_estimators, depth, LR, regularization)
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="cs-callout cs-callout-red">
        <strong>Experiment tracking</strong> — MLflow 3.14 with SQLite backend<br>
        2 registered runs: baseline LogReg + tuned XGBoost<br>
        Artifacts: sklearn pipelines, joblib files, full metric history
        </div>
        """, unsafe_allow_html=True)

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cs-section-label">Stack</div>
    <div class="cs-section-title">Technologies Used</div>
    """, unsafe_allow_html=True)

    badges_html = '<div class="cs-badge-row">'
    for tech, color in TECH_STACK:
        badges_html += f'<span class="cs-badge" style="background:{color};">{tech}</span>'
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)


# ── Page 2 · Explore the Data ─────────────────────────────────────────────────

def page_eda():
    st.markdown("""
    <div class="cs-section-label">Dataset · 7,032 rows · 21 raw features</div>
    <div class="cs-section-title">Explore the Data</div>
    """, unsafe_allow_html=True)

    raw = load_raw_data()
    if raw.get("_is_demo", pd.Series([False])).any():
        st.caption("Demo data — place `Telco-Customer-Churn.csv` in `data/` for real charts.")
    raw["Churn_label"] = raw["Churn"].map({"No": "Retained", "Yes": "Churned"})

    # ── Row 1: Churn distribution + Churn by contract ─────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        dist = raw["Churn_label"].value_counts().reset_index()
        dist.columns = ["Status", "Count"]
        dist["Rate"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
        fig = px.bar(
            dist, x="Status", y="Count", color="Status",
            color_discrete_map=CHURN_PALETTE,
            text=dist["Rate"].astype(str) + "%",
            title="Overall Churn Distribution",
            template=PLOTLY_THEME,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=340, yaxis_title="Customers",
                          margin=dict(t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        contract_churn = (
            raw.groupby("Contract")["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index()
        )
        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        fig2 = px.bar(
            contract_churn, x="Contract", y="Churn Rate (%)",
            color="Churn Rate (%)", color_continuous_scale=["#bfdbfe", "#1e40af"],
            text=contract_churn["Churn Rate (%)"].round(1).astype(str) + "%",
            title="Churn Rate by Contract Type",
            template=PLOTLY_THEME,
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=340,
                           margin=dict(t=45, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Feature selector + Correlation heatmap ─────────────────────────
    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "SeniorCitizen"]

    with col3:
        selected = st.selectbox("Select a feature to compare by churn status", numeric_cols)
        fig3 = px.box(
            raw, x="Churn_label", y=selected, color="Churn_label",
            color_discrete_map=CHURN_PALETTE,
            title=f"{selected} — Retained vs. Churned",
            template=PLOTLY_THEME,
        )
        fig3.update_layout(showlegend=False, height=360,
                           xaxis_title="", margin=dict(t=45, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        feats_df = load_features()
        if feats_df is not None:
            corr_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            extra     = ["is_month_to_month", "num_addon_services", "stickiness_score",
                         "charge_per_service", "has_fiber_optic"]
            corr_df   = feats_df[corr_cols + extra].corr().round(2)
            fig4 = px.imshow(
                corr_df,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=True,
                title="Feature Correlation (selected)",
                template=PLOTLY_THEME,
            )
            fig4.update_layout(height=380, margin=dict(t=45, b=10))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info(
                "Correlation heatmap uses engineered features. "
                "Run `python src/features/run_features.py` to generate `data/features.csv`."
            )

    # ── Key findings ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="cs-section-label">Findings</div>
    <div class="cs-section-title">What the Data Tells Us</div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("""
        <div class="cs-callout cs-callout-red">
        <strong>Contract type is the strongest signal.</strong><br>
        Month-to-month customers churn at <strong>42.7%</strong> vs 11.3% (1-year) and
        2.8% (2-year). Customers on long-term contracts are structurally retained.
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="cs-callout cs-callout-amber">
        <strong>Fiber optic &amp; electronic check are co-risk factors.</strong><br>
        Fiber optic customers churn at ~41.9% vs 18.9% for DSL. Electronic check users
        churn at 45.3% — likely because the payment method requires active effort to cancel.
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown("""
        <div class="cs-callout cs-callout-green">
        <strong>Tenure is a strong protective factor.</strong><br>
        Customers in their first year have the highest churn rate (&gt;35%).
        After 3 years, churn drops below 10%. Retaining new customers past month 12
        dramatically reduces lifetime churn risk.
        </div>
        """, unsafe_allow_html=True)


# ── Page 3 · Model Results ────────────────────────────────────────────────────

def page_models():
    st.markdown("""
    <div class="cs-section-label">Experiment Results</div>
    <div class="cs-section-title">Model Comparison &amp; Performance</div>
    """, unsafe_allow_html=True)

    results    = load_model_results()
    preds      = load_predictions()
    model      = load_model()
    _demo_preds = preds.get("_is_demo", pd.Series([False])).any() if preds is not None else False
    _demo_model = model is None

    # ── Comparison table ──────────────────────────────────────────────────────
    rows = []
    for m in results:
        rows.append({
            "Model":          m["name"],
            "Role":           m["label"],
            "CV ROC-AUC":     f'{m["cv_mean"]:.4f} ±{m["cv_std"]:.4f}' if m["cv_mean"] else "—",
            "Test ROC-AUC":   m["test_roc_auc"],
            "Test Recall":    m["test_recall"],
            "Test F1":        m["test_f1"],
            "Train Time (s)": m["train_time_s"],
        })

    df_cmp = pd.DataFrame(rows)

    def _highlight_winner(row):
        style = [""] * len(row)
        if row["Role"] == "Winner":
            style = ["background-color:#d1fae5; font-weight:600"] * len(row)
        return style

    styled = (
        df_cmp.style
        .apply(_highlight_winner, axis=1)
        .format({"Test ROC-AUC": "{:.4f}", "Test Recall": "{:.4f}", "Test F1": "{:.4f}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Winner rationale ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cs-section-label">Why XGBoost Won</div>
    <div class="cs-section-title">Winner Analysis</div>
    """, unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown("""
        <div class="cs-win-card">
          <div class="metric" style="color:#059669;">80.7%</div>
          <div class="label">Test Recall (Churn=Yes)</div>
          <hr style="border:none;border-top:1px solid #e5e7eb;margin:0.75rem 0;">
          <small>Catches 4 in 5 churners — the highest among all candidates.
          In Telco, a missed churner is lost revenue; a false alarm is a courtesy call.</small>
        </div>
        """, unsafe_allow_html=True)
    with wc2:
        st.markdown("""
        <div class="cs-win-card">
          <div class="metric" style="color:#1e40af;">0.8513</div>
          <div class="label">Best CV ROC-AUC (30 Optuna trials)</div>
          <hr style="border:none;border-top:1px solid #e5e7eb;margin:0.75rem 0;">
          <small>Highest cross-validated AUC after Optuna tuning.
          TPE sampler found optimal depth-3, low learning-rate configuration
          that generalises well.</small>
        </div>
        """, unsafe_allow_html=True)
    with wc3:
        st.markdown("""
        <div class="cs-win-card">
          <div class="metric" style="color:#d97706;">1.05s</div>
          <div class="label">Training Time (12× faster than GBM)</div>
          <hr style="border:none;border-top:1px solid #e5e7eb;margin:0.75rem 0;">
          <small><code>scale_pos_weight=2.76</code> handles class imbalance natively — no
          SMOTE oversampling needed. Fast iteration during Optuna search was a key advantage.</small>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance + Confusion matrix ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fi_col, cm_col = st.columns([1, 1])

    with fi_col:
        st.markdown("**Top 15 Feature Importances**")
        fi_df = get_feature_importance(model) if model else _demo_feature_importance()
        if _demo_model:
            st.caption("Demo importances from actual model run — train the model for live extraction.")
        fig_fi = px.bar(
            fi_df[::-1], x="importance", y="feature", orientation="h",
            color="importance",
            color_continuous_scale=["#bfdbfe", "#1e40af"],
            title="XGBoost Feature Importance (gain)",
            template=PLOTLY_THEME,
        )
        fig_fi.update_layout(
            coloraxis_showscale=False, height=450,
            yaxis_title="", xaxis_title="Importance (gain)",
            margin=dict(t=45, b=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with cm_col:
        st.markdown("**Confusion Matrix — Test Set**")
        if _demo_preds:
            st.caption("Demo predictions — run `python app/generate_data.py` for actual model results.")
        cm_arr = confusion_matrix(preds["actual"], preds["predicted"])
        labels = ["Retained (0)", "Churned (1)"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=labels, y=labels,
            text=[[str(v) for v in row] for row in cm_arr],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            colorscale=[[0, "#bfdbfe"], [1, "#1e40af"]],
            showscale=False,
        ))
        fig_cm.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template=PLOTLY_THEME,
            height=350,
            margin=dict(t=45, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    if preds is not None:  # always True now; demo data guaranteed
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(preds["actual"], preds["probability"])
        auc_score   = roc_auc_score(preds["actual"], preds["probability"])

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"XGBoost Tuned (AUC = {auc_score:.4f})",
            line=dict(color=ACCENT, width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random baseline",
            line=dict(color=MUTED, width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template=PLOTLY_THEME,
            height=340,
            legend=dict(x=0.55, y=0.1),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Try it yourself ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="cs-section-label">Live Demo</div>
    <div class="cs-section-title">Try It Yourself</div>
    """, unsafe_allow_html=True)
    if _demo_model:
        st.caption(
            "Demo mode — using a heuristic predictor. "
            "Run `python src/models/run_training.py` to enable the live XGBoost model."
        )
    else:
        st.caption("Adjust the customer profile below and click Predict — the production XGBoost model runs in real time.")

    with st.form("prediction_form"):
        f1, f2, f3 = st.columns(3)

        with f1:
            contract     = st.selectbox("Contract Type",
                                        ["Month-to-month", "One year", "Two year"])
            tenure       = st.slider("Tenure (months)", 0, 72, 12)
            senior       = st.checkbox("Senior Citizen")

        with f2:
            monthly_ch   = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=1.0)
            internet     = st.selectbox("Internet Service",
                                        ["Fiber optic", "DSL", "No"])
            partner      = st.checkbox("Has Partner")

        with f3:
            payment      = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ])
            num_addons   = st.slider("Add-on Services (0–6)", 0, 6, 2,
                                     help="OnlineSecurity, Backup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies")

        submitted = st.form_submit_button("Predict Churn Risk", type="primary", use_container_width=True)

    if submitted:
        if model:
            row  = build_prediction_row(contract, tenure, monthly_ch, internet,
                                        payment, senior, partner, num_addons)
            prob = float(model.predict_proba(row)[0, 1])
        else:
            prob = _heuristic_predict(contract, tenure, monthly_ch, internet,
                                      payment, senior, partner, num_addons)

        risk_label = "HIGH" if prob > 0.55 else ("MEDIUM" if prob > 0.30 else "LOW")
        risk_color = DANGER if prob > 0.55 else (AMBER if prob > 0.30 else SUCCESS)

        res1, res2 = st.columns([1, 1])

        with res1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 44, "color": risk_color}},
                delta={"reference": 26.5, "suffix": "pp vs base rate",
                       "increasing": {"color": DANGER}, "decreasing": {"color": SUCCESS}},
                title={"text": f"Churn Probability — <b>{risk_label} RISK</b>",
                       "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar":  {"color": risk_color, "thickness": 0.25},
                    "steps": [
                        {"range": [0,  30], "color": "#d1fae5"},
                        {"range": [30, 55], "color": "#fef3c7"},
                        {"range": [55, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#374151", "width": 2},
                        "thickness": 0.75,
                        "value": 26.5,
                    },
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=30, b=0, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res2:
            st.markdown("**Risk factor breakdown**")
            risk_items    = []
            protect_items = []

            if contract == "Month-to-month":
                risk_items.append("Month-to-month contract (42.7% avg churn rate)")
            else:
                protect_items.append(f"{contract} contract — low structural churn")

            if tenure <= 3:
                risk_items.append("New customer (<= 3 months) — highest-risk period")
            elif tenure > 36:
                protect_items.append(f"{tenure}-month tenure — well past the churn window")

            if internet == "Fiber optic":
                risk_items.append("Fiber optic service (41.9% churn vs 18.9% DSL)")

            if monthly_ch > 70:
                risk_items.append(f"High monthly charges (${monthly_ch:.0f} > $70 threshold)")

            if num_addons == 0:
                risk_items.append("No add-on services — low switching cost")
            elif num_addons >= 3:
                protect_items.append(f"{num_addons} add-on services — high switching cost")

            if "automatic" not in payment:
                risk_items.append("Non-automatic payment — easier to actively cancel")
            else:
                protect_items.append("Automatic payment — passive billing reduces churn")

            for item in risk_items:
                st.markdown(f"""
                <div class="cs-callout cs-callout-red" style="padding:0.5rem 1rem; margin:0.3rem 0; font-size:0.88rem;">
                &#9888; {item}
                </div>""", unsafe_allow_html=True)

            for item in protect_items:
                st.markdown(f"""
                <div class="cs-callout cs-callout-green" style="padding:0.5rem 1rem; margin:0.3rem 0; font-size:0.88rem;">
                &#10003; {item}
                </div>""", unsafe_allow_html=True)


# ── Page 4 · How I Built This ─────────────────────────────────────────────────

def page_process():
    st.markdown("""
    <div class="cs-section-label">Engineering</div>
    <div class="cs-section-title">How I Built This</div>
    """, unsafe_allow_html=True)

    # ── Architecture diagram ───────────────────────────────────────────────────
    st.markdown("**Pipeline Architecture**")
    st.graphviz_chart("""
    digraph pipeline {
        rankdir=LR
        node [shape=box style="filled,rounded" fontname="Arial" fontsize=11 margin="0.2,0.1"]
        edge [fontsize=10 color="#94a3b8"]

        A [label="Raw CSV\\n(Kaggle)" fillcolor="#dbeafe" color="#3b82f6"]
        B [label="DataLoader\\n+ QualityGate" fillcolor="#ede9fe" color="#7c3aed"]
        C [label="Cleaner\\n(11 rows dropped)" fillcolor="#ede9fe" color="#7c3aed"]
        D [label="Feature\\nEngineering\\n(+11 features)" fillcolor="#ede9fe" color="#7c3aed"]
        E [label="Feature\\nSelection\\n(corr + variance)" fillcolor="#ede9fe" color="#7c3aed"]
        F [label="Baseline\\nLogReg" fillcolor="#fef9c3" color="#ca8a04"]
        G [label="3-Model\\nComparison\\n(RF, GBM, XGB)" fillcolor="#fef9c3" color="#ca8a04"]
        H [label="Optuna\\nTuning\\n(30 trials)" fillcolor="#fee2e2" color="#dc2626"]
        I [label="MLflow\\nTracking" fillcolor="#fce7f3" color="#db2777"]
        J [label="Production\\nModel" fillcolor="#d1fae5" color="#059669"]
        K [label="Streamlit\\nApp" fillcolor="#dbeafe" color="#2563eb"]

        A->B->C->D->E->F->G->H->I->J->K
    }
    """)

    # ── Build timeline ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cs-section-label">Timeline</div>
    <div class="cs-section-title">Build Timeline</div>
    """, unsafe_allow_html=True)

    timeline = [
        ("Day 1", "Data Foundation",
         "CSV loader, 8-point quality gate, cleaning pipeline. Identified 11 blank-string TotalCharges rows (tenure=0) and dropped them cleanly."),
        ("Day 2", "Exploratory Data Analysis",
         "8-section EDA notebook: univariate distributions, bivariate vs churn, correlation matrix, class imbalance analysis, and segment-level churn rates by Contract, Payment, Internet."),
        ("Day 3", "Feature Engineering",
         "11 new features across 3 categories: domain flags (is_month_to_month, has_fiber_optic, uses_auto_payment), statistical features (charge_per_service, tenure_stage), and interaction terms (stickiness_score, high_charge_x_no_contract)."),
        ("Day 4", "Model Selection",
         "Baseline LogReg (ROC-AUC 0.8387). Compared Random Forest, Gradient Boosting, and XGBoost with 5-fold stratified CV. XGBoost tied on CV AUC but dominated on recall (76.5% vs 53.7%) — selected as the candidate."),
        ("Day 5", "Hyperparameter Tuning",
         "30 Optuna trials with TPE sampler over 8 hyperparameters. Best trial #22 improved CV AUC from 0.8402 to 0.8513 and recall from 76.5% to 80.7% on the test set."),
        ("Day 6", "ML Experiment Tracking",
         "MLflow 3.14 with SQLite backend (file store deprecated in v3). Logged params, train/test metrics, sklearn pipeline artifacts, and joblib files for both runs. Production model exported."),
        ("Day 7", "Portfolio App",
         "This Streamlit app — 4-page interactive portfolio with live predictions, feature importance, confusion matrix, ROC curve, and a real-time prediction form."),
    ]

    for day, title, desc in timeline:
        st.markdown(f"""
        <div class="cs-tl-item">
          <div class="cs-tl-day">{day}</div>
          <div class="cs-tl-title">{title}</div>
          <div class="cs-tl-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key technical decisions ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cs-section-label">Engineering Choices</div>
    <div class="cs-section-title">Key Technical Decisions</div>
    """, unsafe_allow_html=True)

    decisions = [
        (
            "Median variance (not mean) in feature selection",
            "The first implementation used <code>variances.mean()</code> as the threshold — "
            "TotalCharges variance (~400K) inflated the mean to 3,965, which eliminated tenure, "
            "MonthlyCharges, and all useful features. Switching to <code>variances.median()</code> "
            "(~9) gave a sensible threshold that only dropped zero-variance noise."
        ),
        (
            "scale_pos_weight instead of SMOTE for class imbalance",
            "SMOTE generates synthetic minority samples, which can leak information across the "
            "train/test boundary and doesn't translate naturally to inference on real data. "
            "<code>scale_pos_weight = 2.76</code> (neg/pos ratio) tells XGBoost to cost-weight "
            "each Churn=Yes example more heavily during training — cleaner, faster, and more interpretable."
        ),
        (
            "Recall over precision as the tie-breaker",
            "XGBoost and Gradient Boosting tied on CV ROC-AUC (0.8402). The tiebreaker was recall "
            "on the test set: 76.5% vs 53.7%. In Telco retention, a missed churner = lost LTV. "
            "A false positive = an unnecessary retention call. The asymmetric cost justifies "
            "optimizing recall."
        ),
        (
            "MLflow SQLite backend (not file store)",
            "MLflow 3.x deprecated the default filesystem tracking backend. The fix was "
            "<code>mlflow.set_tracking_uri('sqlite:///mlflow.db')</code>. XGBoost artifacts "
            "also required <code>skops_trusted_types</code> in <code>log_model()</code> due to "
            "MLflow 3.x's new security model for sklearn estimators."
        ),
    ]

    for title, body in decisions:
        st.markdown(f"""
        <div class="cs-decision">
          <strong>{title}</strong>
          <p style="margin:0.4rem 0 0; color:#374151; font-size:0.9rem; line-height:1.6;">{body}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Links ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    link_col1, link_col2, _ = st.columns([1, 1, 2])
    with link_col1:
        st.markdown("""
        <a href="https://github.com/Joachim-Wambua" target="_blank"
           style="display:inline-block; background:#0f172a; color:white;
                  padding:10px 20px; border-radius:8px; text-decoration:none;
                  font-weight:600; font-size:0.9rem;">
           GitHub Profile
        </a>
        """, unsafe_allow_html=True)
    with link_col2:
        st.markdown("""
        <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
           target="_blank"
           style="display:inline-block; background:#20beff; color:white;
                  padding:10px 20px; border-radius:8px; text-decoration:none;
                  font-weight:600; font-size:0.9rem;">
           Dataset (Kaggle)
        </a>
        """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📡 ChurnScope")
        st.caption("Telecom Customer Retention Intelligence")
        st.divider()

        page = st.radio(
            "Navigate",
            options=["Overview", "Explore Data", "Model Results", "How I Built This"],
            format_func=lambda x: {
                "Overview":          "🏠  Overview",
                "Explore Data":      "🔍  Explore Data",
                "Model Results":     "📊  Model Results",
                "How I Built This":  "🛠️  How I Built This",
            }[x],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("**Joachim Wambua**")
        st.caption("Data Scientist")
        st.caption("kimwambua96@gmail.com")
        st.markdown(
            '<a href="https://github.com/Joachim-Wambua" target="_blank">GitHub</a>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("IBM Telco Churn · 7,032 rows · XGBoost")
        st.caption("ROC-AUC 0.8403 · Recall 80.7%")

    # ── Route to page ─────────────────────────────────────────────────────────
    if page == "Overview":
        page_overview()
    elif page == "Explore Data":
        page_eda()
    elif page == "Model Results":
        page_models()
    else:
        page_process()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cs-footer">
        Built by <strong>Joachim Wambua</strong> &nbsp;·&nbsp;
        Stack: Python · scikit-learn · XGBoost · Optuna · MLflow · Streamlit &nbsp;·&nbsp;
        Dataset: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
                    target="_blank" style="color:#3b82f6;">IBM Telco Customer Churn</a>
    </div>
    """, unsafe_allow_html=True)


main()
