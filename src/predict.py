"""Reusable customer-segment prediction function."""

from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model_pipeline.pkl"
FEATURES = [
    "RecencyDays",
    "PurchaseFrequency",
    "TotalSpending",
    "AverageOrderValue",
    "TotalQuantity",
    "AverageItemsPerInvoice",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run: python src/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_customer_segment(customer_data):
    missing = [f for f in FEATURES if f not in customer_data]
    if missing:
        raise ValueError(f"Missing customer features: {missing}")

    model_bundle = load_model()
    pipeline = model_bundle["pipeline"]
    encoder = model_bundle["label_encoder"]

    row = pd.DataFrame(
        [{feature: float(customer_data[feature]) for feature in FEATURES}],
        columns=FEATURES,
    )
    predicted_code = int(pipeline.predict(row)[0])
    result = {
        "predicted_segment": str(encoder.inverse_transform([predicted_code])[0]),
        "model_name": "Saved selected classifier",
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        result["class_probabilities"] = {
            str(encoder.inverse_transform([int(code)])[0]): float(prob)
            for code, prob in zip(pipeline.named_steps["classifier"].classes_, probabilities)
        }
        result["confidence_for_predicted_class"] = float(max(probabilities))
    return result
