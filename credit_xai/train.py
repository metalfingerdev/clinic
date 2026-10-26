"""Train the XGBoost loan approval model and save it to models/.

Run with:  python -m credit_xai.train
"""

import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from credit_xai.data import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    TARGET,
    clean,
    encode,
    load_raw,
)

RANDOM_STATE = 42
THRESHOLD = 0.5


def train():
    df = clean(load_raw())
    X, y = encode(df), df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # Hold out part of the training set for early stopping; the test set stays untouched.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "best_iteration": int(model.best_iteration),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance_gain": dict(
            sorted(
                zip(FEATURES, map(float, model.feature_importances_)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The training matrix is stored as LIME's background data so the app
    # does not need the raw CSV at runtime.
    joblib.dump(
        {
            "model": model,
            "background": X_train.to_numpy(),
            "threshold": THRESHOLD,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    metrics = train()
    print(f"Best iteration: {metrics['best_iteration']}")
    for name in ("accuracy", "roc_auc", "precision", "recall", "f1"):
        print(f"{name:>10}: {metrics[name]:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])
    print("Top features (gain):")
    for feat, gain in list(metrics["feature_importance_gain"].items())[:5]:
        print(f"  {feat:<32} {gain:.3f}")
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
