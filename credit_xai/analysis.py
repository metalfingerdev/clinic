"""Feature analysis of loan_data.csv, written to reports/feature_analysis.md.

Run with:  python -m credit_xai.analysis
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from credit_xai.data import (
    CATEGORIES,
    FEATURES,
    MAX_PLAUSIBLE_AGE,
    NUMERIC_FEATURES,
    REPORT_PATH,
    TARGET,
    encode,
    load_raw,
)


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def analyze(df: pd.DataFrame) -> str:
    y = df[TARGET]
    approval_rate = y.mean()
    sections = ["# Feature Analysis: loan_data.csv", ""]

    sections += [
        "## Overview",
        f"- Rows: {len(df):,}  |  Features: {len(FEATURES)} "
        f"({len(NUMERIC_FEATURES)} numeric, {len(CATEGORIES)} categorical)  |  Target: `{TARGET}`",
        f"- Missing values: {int(df.isna().sum().sum())}  |  Duplicate rows: {int(df.duplicated().sum())}",
        f"- Class balance: {approval_rate:.1%} approved (1), {1 - approval_rate:.1%} rejected (0)",
        "",
    ]

    implausible_age = int((df["person_age"] > MAX_PLAUSIBLE_AGE).sum())
    derived = (df["loan_amnt"] / df["person_income"]).round(2)
    pct_match = ((derived - df["loan_percent_income"]).abs() <= 0.011).mean()
    sections += [
        "## Data quality",
        f"- `person_age` > {MAX_PLAUSIBLE_AGE}: {implausible_age} rows "
        f"(max {df['person_age'].max():.0f}); these are dropped before training.",
        f"- `person_income` is heavily right-skewed: median {df['person_income'].median():,.0f}, "
        f"max {df['person_income'].max():,.0f}. Tree models are insensitive to this, so it is kept as-is.",
        f"- `loan_percent_income` equals `round(loan_amnt / person_income, 2)` in {pct_match:.1%} of rows, "
        "so the app derives it instead of asking for it.",
        "",
    ]

    numeric = df[NUMERIC_FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
    by_class = df.groupby(TARGET)[NUMERIC_FEATURES].mean().T
    numeric["mean | rejected"] = by_class[0]
    numeric["mean | approved"] = by_class[1]
    numeric["corr w/ target"] = df[NUMERIC_FEATURES].corrwith(y)
    numeric.index.name = "feature"
    sections += ["## Numeric features", md_table(numeric.sort_values("corr w/ target")), ""]

    sections.append("## Categorical features (approval rate per category)")
    for col in CATEGORIES:
        stats = df.groupby(col)[TARGET].agg(count="count", approval_rate="mean")
        stats["count"] = stats["count"].astype(str)
        sections += [f"### `{col}`", md_table(stats.sort_values("approval_rate")), ""]

    mi = mutual_info_classif(
        encode(df),
        y,
        discrete_features=[f in CATEGORIES for f in FEATURES],
        random_state=42,
    )
    ranking = pd.DataFrame({"mutual_information": mi}, index=pd.Index(FEATURES, name="feature"))
    sections += [
        "## Feature ranking (mutual information with target)",
        md_table(ranking.sort_values("mutual_information", ascending=False), "{:.4f}"),
        "",
    ]

    prior_default = df.groupby("previous_loan_defaults_on_file")[TARGET].mean()
    renters = df.groupby("person_home_ownership")[TARGET].mean()
    sections += [
        "## Key observations",
        f"- **Previous default is a hard rule:** applicants with a default on file are approved "
        f"{prior_default['Yes']:.1%} of the time vs {prior_default['No']:.1%} without one.",
        "- **Loan burden and pricing dominate the remaining signal:** `loan_percent_income` and "
        "`loan_int_rate` are the most correlated numeric features, followed by `person_income` "
        "(lower income goes with more approvals).",
        f"- **Home ownership matters:** renters are approved {renters['RENT']:.1%} of the time vs "
        f"{renters['OWN']:.1%} for owners and {renters['MORTGAGE']:.1%} for mortgage holders.",
        "- **No signal from demographics or credit score:** gender, education, age and `credit_score` "
        "have near-zero correlation/mutual information with the target.",
        "- **Caveat on label semantics:** the dataset documents `1 = approved`, yet higher loan-to-income "
        "ratios and interest rates go with *more* approvals. The set appears to be derived from a "
        "credit-risk dataset where 1 meant default, so read the explanations as \"what this "
        "dataset rewards\", not as lending policy.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    report = analyze(load_raw())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
