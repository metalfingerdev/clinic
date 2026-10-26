"""LIME local explanations for the XGBoost loan model."""

import numpy as np
import pandas as pd
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer

from credit_xai.data import CATEGORICAL_INDICES, CATEGORIES, CLASS_NAMES, FEATURE_LABELS, FEATURES

APPROVED = CLASS_NAMES.index("Approved")


def build_explainer(background: np.ndarray) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=background,
        feature_names=[FEATURE_LABELS[f] for f in FEATURES],
        class_names=CLASS_NAMES,
        categorical_features=CATEGORICAL_INDICES,
        categorical_names={FEATURES.index(f): cats for f, cats in CATEGORIES.items()},
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def predict_fn(model):
    """Wrap the model so LIME's perturbed numpy rows keep their feature names."""

    def predict_proba(rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(rows, columns=FEATURES))

    return predict_proba


def explain(
    explainer: LimeTabularExplainer,
    model,
    encoded_row: np.ndarray,
    num_features: int = 10,
    num_samples: int = 5000,
) -> Explanation:
    """Explain one encoded applicant; weights are toward the "Approved" class."""
    return explainer.explain_instance(
        encoded_row,
        predict_fn(model),
        num_features=num_features,
        num_samples=num_samples,
        labels=(APPROVED,),
    )
