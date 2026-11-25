
import argparse
import json
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train(data_dir="data/processed", n_estimators=100):
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "target" not in train.columns or "target" not in test.columns:
        raise ValueError("Both datasets must contain a 'target' column")

    X_tr = train.drop(columns=["target"])
    y_tr = train["target"]
    X_te = test.drop(columns=["target"])
    y_te = test["target"]

    if list(X_tr.columns) != list(X_te.columns):
        raise ValueError("Training and test feature columns do not match")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    return model, {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(
            precision_score(y_te, pred, zero_division=0),
            4,
        ),
        "recall": round(
            recall_score(y_te, pred, zero_division=0),
            4,
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--model-out", default="models/model.joblib")
    p.add_argument("--metrics-out", default="models/metrics.json")
    a = p.parse_args()

    model, metrics = train(a.data_dir, a.n_estimators)

    os.makedirs(os.path.dirname(a.model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.metrics_out) or ".", exist_ok=True)

    joblib.dump(model, a.model_out)
    with open(a.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", a.model_out)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
