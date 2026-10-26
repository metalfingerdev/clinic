"""Dataset schema, loading, cleaning and encoding for loan_data.csv."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "loan_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_loan_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_analysis.md"

TARGET = "loan_status"
# loan_status: 0 = rejected, 1 = approved (index = class label).
CLASS_NAMES = ["Rejected", "Approved"]

# Category order defines the integer code used by the model and by LIME.
# Education is listed in its natural order so its codes are ordinal.
CATEGORIES = {
    "person_gender": ["female", "male"],
    "person_education": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

# Column order as in the CSV; the model and LIME both rely on it.
FEATURES = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORIES]
CATEGORICAL_INDICES = [FEATURES.index(f) for f in CATEGORIES]

FEATURE_LABELS = {
    "person_age": "Age",
    "person_gender": "Gender",
    "person_education": "Education",
    "person_income": "Annual income",
    "person_emp_exp": "Employment experience (yrs)",
    "person_home_ownership": "Home ownership",
    "loan_amnt": "Loan amount",
    "loan_intent": "Loan intent",
    "loan_int_rate": "Interest rate (%)",
    "loan_percent_income": "Loan / income ratio",
    "cb_person_cred_hist_length": "Credit history (yrs)",
    "credit_score": "Credit score",
    "previous_loan_defaults_on_file": "Previous loan default",
}

MAX_PLAUSIBLE_AGE = 100


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with impossible ages (e.g. 144); every other row is kept."""
    return df[df["person_age"] <= MAX_PLAUSIBLE_AGE].reset_index(drop=True)


def loan_percent_income(loan_amnt: float, person_income: float) -> float:
    """Derived feature, defined in the dataset as loan / income rounded to 2 dp."""
    return round(loan_amnt / person_income, 2)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix with categoricals replaced by integer codes."""
    out = df[FEATURES].copy()
    for col, categories in CATEGORIES.items():
        codes = pd.Categorical(out[col], categories=categories).codes
        if (codes < 0).any():
            unknown = sorted(set(out[col][codes < 0]))
            raise ValueError(f"Unknown value(s) for {col}: {unknown}")
        out[col] = codes
    return out.astype(float)
