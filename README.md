# Explainable AI Framework for Credit Risk Assessment and Loan Approval Prediction

An XGBoost classifier trained on `loan_data.csv` predicts whether a loan application is
**approved** or **rejected**. A Streamlit frontend shows the decision and uses **LIME**
to explain each prediction locally.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Usage

```bash
.venv/bin/python -m credit_xai.analysis   # feature analysis -> reports/feature_analysis.md
.venv/bin/python -m credit_xai.train      # train XGBoost   -> models/xgb_loan_model.joblib
.venv/bin/streamlit run app.py            # interactive frontend
```

## Project layout

| Path | Purpose |
|---|---|
| `loan_data.csv` | Dataset: 45,000 applications, 13 features, target `loan_status` (1 = approved, 0 = rejected) |
| `credit_xai/data.py` | Schema, cleaning (drops ages > 100), categorical encoding |
| `credit_xai/analysis.py` | Feature analysis report |
| `credit_xai/train.py` | XGBoost training, test-set evaluation, model + metrics export |
| `credit_xai/explain.py` | LIME tabular explainer and prediction wrapper |
| `app.py` | Streamlit UI: applicant form, decision, LIME visualisations |
| `models/` | Trained model bundle and `metrics.json` (generated) |
| `reports/` | Feature analysis report (generated) |

## How it works

- **Encoding:** categorical features are mapped to integer codes (education in its natural
  order). The same encoded matrix feeds XGBoost and LIME, so LIME can perturb categories
  as categories and label them by name (e.g. `Home ownership=RENT`).
- **Derived feature:** `loan_percent_income` equals `round(loan_amnt / person_income, 2)` in
  the data, so the app computes it instead of asking for it.
- **Model:** `XGBClassifier` with early stopping on a validation split, then evaluated on a
  stratified 20% held-out test set (accuracy ≈ 0.94, ROC AUC ≈ 0.98).
- **Explanations:** LIME samples perturbed applicants around the input, scores them with the
  model, and fits a weighted linear surrogate. The app renders LIME's bar chart, its native
  interactive HTML view, and a weights table. The surrogate's local R² is shown as a fit
  indicator.

## Dataset caveats

- `previous_loan_defaults_on_file = Yes` means rejection in 100% of rows, so it dominates
  the model.
- Higher loan-to-income ratios, higher interest rates and renting go with *more* approvals
  in this dataset. The labels follow the dataset's documentation (1 = approved), but these
  patterns look like default risk. Explanations describe what this dataset rewards, not
  sound lending policy.
- Gender, education, age and credit score carry almost no signal.
