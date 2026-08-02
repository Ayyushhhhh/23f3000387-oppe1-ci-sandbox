import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURE_COLS = ["rolling_avg_10", "volume_sum_10", "stock_name"]
TARGET_COL = "target"


def main():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ.get("MODEL_NAME", "stock_movement_predictor")
    model_alias = os.environ.get("MODEL_ALIAS", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    test = pd.read_csv("data/test.csv")
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "model_uri": model_uri,
        "n_test_rows": len(test),
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs("ci_out", exist_ok=True)
    with open("ci_out/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_metrics = {k: v for k, v in metrics.items() if k in ("accuracy", "precision", "recall", "f1")}
    plt.figure(figsize=(5, 4))
    plt.bar(plot_metrics.keys(), plot_metrics.values(), color="#4C72B0")
    plt.ylim(0, 1)
    plt.title(f"CI eval - {model_uri}")
    plt.tight_layout()
    plt.savefig("ci_out/metrics.png")

    test_with_preds = test.copy()
    test_with_preds["prediction"] = preds
    test_with_preds.to_csv("ci_out/predictions.csv", index=False)


if __name__ == "__main__":
    main()
