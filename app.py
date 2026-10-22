"""Streamlit app with Dashboard, Segment Prediction, and Model Comparison."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_utils import (
    load_or_build_customer_features, NUMERIC_FEATURES, TARGET, SEGMENT_ORDER
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "models" / "best_model_pipeline.pkl"
COMPARISON_PATH = RESULTS / "model_comparison.csv"
INFO_PATH = RESULTS / "model_info.json"
METRICS_PATH = RESULTS / "final_test_metrics.json"
PREDICTIONS_PATH = RESULTS / "test_predictions.csv"

st.set_page_config(page_title="Customer Segmentation ML", page_icon="🛍️", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def get_customers():
    return load_or_build_customer_features()


@st.cache_resource
def get_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model missing. Run: python src/train_model.py")
    return joblib.load(MODEL_PATH)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Results file missing: {path.name}. Run training first.")
    return json.loads(path.read_text(encoding="utf-8"))


def dashboard(df):
    st.header("Customer Segmentation Dashboard")
    st.write("Explore customer purchasing behaviour from Online Retail II transactions.")

    total = len(df)
    counts = df[TARGET].value_counts()
    high = int(counts.get("High Value", 0))
    regular = int(counts.get("Regular", 0))
    low = int(counts.get("Low Engagement", 0))

    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{total:,}")
    b.metric("High Value", f"{high:,}")
    c.metric("Regular", f"{regular:,}")
    d.metric("Low Engagement", f"{low:,}")

    st.subheader("Customer Segment Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x=TARGET, order=SEGMENT_ORDER, ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Customers")
    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Total Spending Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["TotalSpending"], bins=35, kde=True, ax=ax)
        ax.set_xlabel("Total Spending")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Purchase Frequency by Segment")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y="PurchaseFrequency", order=SEGMENT_ORDER, ax=ax)
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "Segments are generated from transparent RFM rules for this educational project. "
        "The classifier learns to reproduce these rule-derived labels; these are not "
        "independently verified business categories."
    )


def prediction_page(df):
    st.header("Predict Customer Segment")
    st.write("Enter customer purchasing statistics to predict a segment.")

    try:
        bundle = get_model_bundle()
        info = read_json(INFO_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    defaults = {
        feature: float(df[feature].median())
        for feature in NUMERIC_FEATURES
    }

    with st.form("customer_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            recency = st.number_input(
                "Recency (days since last purchase)", min_value=0.0,
                value=max(0.0, defaults["RecencyDays"]), step=1.0
            )
            frequency = st.number_input(
                "Purchase Frequency (unique invoices)", min_value=1.0,
                value=max(1.0, defaults["PurchaseFrequency"]), step=1.0
            )
            spending = st.number_input(
                "Total Spending", min_value=0.0,
                value=max(0.0, defaults["TotalSpending"]), step=10.0
            )
        with col2:
            average_order = st.number_input(
                "Average Order Value", min_value=0.0,
                value=max(0.0, defaults["AverageOrderValue"]), step=5.0
            )
            quantity = st.number_input(
                "Total Quantity Purchased", min_value=0.0,
                value=max(0.0, defaults["TotalQuantity"]), step=1.0
            )
            average_items = st.number_input(
                "Average Items per Invoice", min_value=0.0,
                value=max(0.0, defaults["AverageItemsPerInvoice"]), step=1.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Segment", type="primary", use_container_width=True
        )

    if submitted:
        row = pd.DataFrame([{
            "RecencyDays": recency,
            "PurchaseFrequency": frequency,
            "TotalSpending": spending,
            "AverageOrderValue": average_order,
            "TotalQuantity": quantity,
            "AverageItemsPerInvoice": average_items,
        }], columns=NUMERIC_FEATURES)

        try:
            pipeline = bundle["pipeline"]
            encoder = bundle["label_encoder"]
            code = int(pipeline.predict(row)[0])
            segment = str(encoder.inverse_transform([code])[0])
            st.subheader("Prediction Result")
            if segment == "High Value":
                st.success(f"Predicted Segment: {segment}")
            elif segment == "Regular":
                st.info(f"Predicted Segment: {segment}")
            else:
                st.warning(f"Predicted Segment: {segment}")

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(row)[0]
                st.write("Estimated model probabilities")
                probability_df = pd.DataFrame({
                    "Customer Segment": encoder.inverse_transform(
                        [int(c) for c in pipeline.named_steps["classifier"].classes_]
                    ),
                    "Estimated Probability": probs,
                }).sort_values("Estimated Probability", ascending=False)
                st.dataframe(
                    probability_df.style.format({"Estimated Probability": "{:.2%}"}),
                    use_container_width=True, hide_index=True
                )
                st.caption(
                    "These are model-estimated probabilities for rule-derived labels, "
                    "not guaranteed probabilities of future customer behaviour."
                )
            st.caption(f"Selected model: {info['selected_model']}")
        except (ValueError, KeyError, AttributeError) as exc:
            st.error(f"Prediction failed: {exc}")

    st.warning(
        "Educational demonstration only. Customer segments depend on the RFM rules "
        "used to create the labels and should not be treated as verified customer identities."
    )


def model_comparison_page():
    st.header("Model Comparison")
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
        info = read_json(INFO_PATH)
        metrics = read_json(METRICS_PATH)
        predictions = pd.read_csv(PREDICTIONS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    st.subheader("Algorithm Performance (Validation Set)")
    cols = [
        "Model", "Accuracy", "Precision_macro", "Recall_macro",
        "F1_macro", "F1_weighted"
    ]
    st.dataframe(
        comparison[cols].style.format({
            "Accuracy": "{:.4f}", "Precision_macro": "{:.4f}",
            "Recall_macro": "{:.4f}", "F1_macro": "{:.4f}",
            "F1_weighted": "{:.4f}",
        }),
        use_container_width=True, hide_index=True
    )

    st.subheader("Validation Macro F1-Score")
    sorted_df = comparison.sort_values("F1_macro", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_df["Model"], sorted_df["F1_macro"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1-score")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Selected Model")
    st.write(f"**Model:** {info['selected_model']}")
    st.write(f"**Selection metric:** {info['selection_metric']}")

    st.subheader("Final Test Metrics")
    metric_names = ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "F1_weighted"]
    cols = st.columns(len(metric_names))
    for col, name in zip(cols, metric_names):
        col.metric(name.replace("_", " "), f"{metrics[name]:.4f}")

    st.subheader("Confusion Matrix (Test Set)")
    labels = info["class_names"]
    matrix = confusion_matrix(
        predictions["Actual"], predictions["Predicted"], labels=labels
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Segment")
    ax.set_ylabel("Actual Segment")
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        ["Dashboard", "Segment Prediction", "Model Comparison"]
    )
    try:
        df = get_customers()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    if page == "Dashboard":
        dashboard(df)
    elif page == "Segment Prediction":
        prediction_page(df)
    else:
        model_comparison_page()


if __name__ == "__main__":
    main()
