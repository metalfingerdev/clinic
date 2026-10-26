# Feature Analysis: loan_data.csv

## Overview
- Rows: 45,000  |  Features: 13 (8 numeric, 5 categorical)  |  Target: `loan_status`
- Missing values: 0  |  Duplicate rows: 0
- Class balance: 22.2% approved (1), 77.8% rejected (0)

## Data quality
- `person_age` > 100: 7 rows (max 144); these are dropped before training.
- `person_income` is heavily right-skewed: median 67,048, max 7,200,766. Tree models are insensitive to this, so it is kept as-is.
- `loan_percent_income` equals `round(loan_amnt / person_income, 2)` in 100.0% of rows, so the app derives it instead of asking for it.

## Numeric features
| feature | mean | std | min | 50% | max | mean | rejected | mean | approved | corr w/ target |
|---|---|---|---|---|---|---|---|---|
| person_income | 80,319.053 | 80,422.499 | 8,000.000 | 67,048.000 | 7,200,766.000 | 86,157.041 | 59,886.097 | -0.136 |
| person_age | 27.764 | 6.045 | 20.000 | 26.000 | 144.000 | 27.834 | 27.521 | -0.021 |
| person_emp_exp | 5.410 | 6.064 | 0.000 | 4.000 | 125.000 | 5.477 | 5.178 | -0.020 |
| cb_person_cred_hist_length | 5.867 | 3.880 | 2.000 | 4.000 | 30.000 | 5.898 | 5.760 | -0.015 |
| credit_score | 632.609 | 50.436 | 390.000 | 640.000 | 850.000 | 632.815 | 631.887 | -0.008 |
| loan_amnt | 9,583.158 | 6,314.887 | 500.000 | 8,000.000 | 35,000.000 | 9,219.577 | 10,855.690 | 0.108 |
| loan_int_rate | 11.007 | 2.979 | 5.420 | 11.010 | 20.000 | 10.478 | 12.857 | 0.332 |
| loan_percent_income | 0.140 | 0.087 | 0.000 | 0.120 | 0.660 | 0.122 | 0.203 | 0.385 |

## Categorical features (approval rate per category)
### `person_gender`
| person_gender | count | approval_rate |
|---|---|---|
| male | 24841 | 0.222 |
| female | 20159 | 0.222 |

### `person_education`
| person_education | count | approval_rate |
|---|---|---|
| Master | 6980 | 0.218 |
| Associate | 12028 | 0.220 |
| High School | 11972 | 0.223 |
| Bachelor | 13399 | 0.225 |
| Doctorate | 621 | 0.229 |

### `person_home_ownership`
| person_home_ownership | count | approval_rate |
|---|---|---|
| OWN | 2951 | 0.075 |
| MORTGAGE | 18489 | 0.116 |
| RENT | 23443 | 0.324 |
| OTHER | 117 | 0.333 |

### `loan_intent`
| loan_intent | count | approval_rate |
|---|---|---|
| VENTURE | 7819 | 0.144 |
| EDUCATION | 9153 | 0.170 |
| PERSONAL | 7552 | 0.201 |
| HOMEIMPROVEMENT | 4783 | 0.263 |
| MEDICAL | 8548 | 0.278 |
| DEBTCONSOLIDATION | 7145 | 0.303 |

### `previous_loan_defaults_on_file`
| previous_loan_defaults_on_file | count | approval_rate |
|---|---|---|
| Yes | 22858 | 0.000 |
| No | 22142 | 0.452 |

## Feature ranking (mutual information with target)
| feature | mutual_information |
|---|---|
| previous_loan_defaults_on_file | 0.1910 |
| person_income | 0.0911 |
| loan_percent_income | 0.0831 |
| loan_int_rate | 0.0704 |
| person_home_ownership | 0.0350 |
| loan_amnt | 0.0162 |
| loan_intent | 0.0102 |
| person_emp_exp | 0.0020 |
| cb_person_cred_hist_length | 0.0019 |
| credit_score | 0.0016 |
| person_age | 0.0010 |
| person_education | 0.0000 |
| person_gender | 0.0000 |

## Key observations
- **Previous default is a hard rule:** applicants with a default on file are approved 0.0% of the time vs 45.2% without one.
- **Loan burden and pricing dominate the remaining signal:** `loan_percent_income` and `loan_int_rate` are the most correlated numeric features, followed by `person_income` (lower income goes with more approvals).
- **Home ownership matters:** renters are approved 32.4% of the time vs 7.5% for owners and 11.6% for mortgage holders.
- **No signal from demographics or credit score:** gender, education, age and `credit_score` have near-zero correlation/mutual information with the target.
- **Caveat on label semantics:** the dataset documents `1 = approved`, yet higher loan-to-income ratios and interest rates go with *more* approvals. The set appears to be derived from a credit-risk dataset where 1 meant default, so read the explanations as "what this dataset rewards", not as lending policy.
