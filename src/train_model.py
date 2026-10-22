"""Train and compare seven traditional ML classification models."""

from pathlib import Path
import json
import warnings
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import (
    load_or_build_customer_features, NUMERIC_FEATURES, TARGET, SEGMENT_ORDER
)

warnings.filterwarnings("ignore", category=FutureWarning)
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = MODEL_DIR / "best_model_pipeline.pkl"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
INFO_PATH = RESULTS_DIR / "model_info.json"
TEST_METRICS_PATH = RESULTS_DIR / "final_test_metrics.json"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
    ], remainder="drop")


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, class_weight="balanced", min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "SVC": SVC(
            kernel="rbf", C=1.0, class_weight="balanced",
            probability=True, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", n_jobs=-1, random_state=42
        ),
    }


def evaluate(y_true, predictions, labels):
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision_macro": float(precision_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "Recall_macro": float(recall_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "F1_macro": float(f1_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "F1_weighted": float(f1_score(
            y_true, predictions, average="weighted", zero_division=0
        )),
    }


def main():
    customer_df = load_or_build_customer_features()
    customer_df = customer_df.dropna(subset=NUMERIC_FEATURES + [TARGET]).copy()

    X = customer_df[NUMERIC_FEATURES].astype(float)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(customer_df[TARGET].astype(str))
    class_names = list(label_encoder.classes_)

    if len(class_names) < 2:
        raise ValueError("At least two customer segments are needed to train.")

    # Stratified 70/15/15 split. Validation is used for model selection;
    # test is held aside until the selected model is final.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15 / 0.85,
        stratify=y_train_val, random_state=42
    )

    print("Customer records:", len(customer_df))
    print("Class labels:", class_names)
    print("Split sizes:", len(X_train), "train,", len(X_val), "validation,", len(X_test), "test")
    print("\nClass distribution:\n", customer_df[TARGET].value_counts())

    results = []
    fitted = {}
    models = build_models()

    # Compute balanced weights from training labels for XGBoost only.
    xgb_sample_weights = compute_sample_weight("balanced", y_train)

    for name, classifier in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ])

        if name == "XGBoost":
            pipeline.fit(X_train, y_train, classifier__sample_weight=xgb_sample_weights)
        else:
            pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)
        metrics = evaluate(y_val, predictions, class_names)
        metrics["Model"] = name
        results.append(metrics)
        fitted[name] = pipeline
        print(f"Validation macro F1: {metrics['F1_macro']:.4f}")

    comparison = pd.DataFrame(results).sort_values(
        ["F1_macro", "Recall_macro", "Precision_macro"],
        ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    selected_name = str(comparison.iloc[0]["Model"])
    selected_pipeline = fitted[selected_name]
    print("\nSelected model based on validation macro F1:", selected_name)

    # One final test evaluation after model selection.
    test_predictions = selected_pipeline.predict(X_test)
    test_metrics = evaluate(y_test, test_predictions, class_names)
    test_metrics["confusion_matrix"] = confusion_matrix(
        y_test, test_predictions, labels=list(range(len(class_names)))
    ).tolist()
    test_metrics["class_names"] = class_names
    test_metrics["classification_report"] = classification_report(
        y_test, test_predictions, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )

    # Save a complete pipeline plus encoder so app predictions use the
    # same target-label mapping.
    bundle = {
        "pipeline": selected_pipeline,
        "label_encoder": label_encoder,
        "features": NUMERIC_FEATURES,
    }
    joblib.dump(bundle, MODEL_PATH)

    info = {
        "selected_model": selected_name,
        "selection_metric": "Validation macro F1-score",
        "features": NUMERIC_FEATURES,
        "target": TARGET,
        "class_names": class_names,
        "label_definition": (
            "RFM-derived categories: High Value for RFM score >= 12; "
            "Regular for score >= 8 and < 12; Low Engagement otherwise."
        ),
        "dataset_is_real_transaction_data": True,
        "train_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
    }
    INFO_PATH.write_text(json.dumps(info, indent=4), encoding="utf-8")
    TEST_METRICS_PATH.write_text(json.dumps(test_metrics, indent=4), encoding="utf-8")
    pd.DataFrame({
        "Actual": label_encoder.inverse_transform(y_test),
        "Predicted": label_encoder.inverse_transform(test_predictions),
    }).to_csv(TEST_PREDICTIONS_PATH, index=False)

    print("\nValidation model comparison:\n", comparison.to_string(index=False))
    print("\nFinal test metrics:\n", json.dumps(
        {k: v for k, v in test_metrics.items()
         if k not in ["confusion_matrix", "classification_report"]},
        indent=2
    ))
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(np.array(test_metrics["confusion_matrix"]))
    print("\nSaved model:", MODEL_PATH)
    print("Saved comparison:", COMPARISON_PATH)
    print("Saved test metrics:", TEST_METRICS_PATH)


if __name__ == "__main__":
    main()
