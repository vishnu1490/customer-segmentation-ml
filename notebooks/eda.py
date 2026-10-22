"""Exploratory data analysis. All figures are saved; none are popped up."""

from pathlib import Path
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save figures without opening GUI windows.
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_or_build_customer_features, NUMERIC_FEATURES, TARGET

OUTPUT_DIR = ROOT / "results" / "eda"
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run EDA and save all charts without pop-up windows."
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-read the Excel workbook and rebuild customer features."
    )
    args = parser.parse_args()

    customers = load_or_build_customer_features(force_rebuild=args.rebuild)
    print("Customer-level dataset shape:", customers.shape)
    print("\\nCustomer feature columns:", customers.columns.tolist())
    print("\\nCustomer feature data types:\\n", customers.dtypes)
    print("\\nCustomer feature statistics:\\n", customers[NUMERIC_FEATURES].describe().T)
    print("\\nMissing values:\\n", customers.isna().sum())
    print("Duplicate customer rows:", customers.duplicated().sum())
    print("\\nCustomer segment distribution:\\n", customers[TARGET].value_counts(dropna=False))
    print("\\nCustomer segment percentages:\\n",
          (customers[TARGET].value_counts(normalize=True) * 100).round(2))

    # 1. Customer segment counts
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=customers, x=TARGET, hue=TARGET, order=["Low Engagement", "Regular", "High Value"], legend=False, ax=ax)
    ax.set_title("Customer Segment Distribution")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    save_fig(fig, "customer_segment_distribution.png")

    # 2. Feature distributions
    for feature in NUMERIC_FEATURES:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=customers, x=feature, bins=35, kde=True, color="steelblue", ax=ax)
        ax.set_title(f"Distribution of {feature}")
        save_fig(fig, f"distribution_{feature.lower()}.png")

    # 3. Feature comparisons by category
    for feature in NUMERIC_FEATURES:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(
            data=customers, x=TARGET, y=feature,
            order=["Low Engagement", "Regular", "High Value"], ax=ax
        )
        ax.set_title(f"{feature} by Customer Segment")
        ax.tick_params(axis="x", rotation=15)
        save_fig(fig, f"segment_comparison_{feature.lower()}.png")

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(customers[NUMERIC_FEATURES].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Customer Features")
    save_fig(fig, "correlation_heatmap.png")

    # 5. Outlier boxplots
    long_df = customers[NUMERIC_FEATURES].melt(
        var_name="Feature", value_name="Value"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=long_df, x="Feature", y="Value", ax=ax)
    ax.set_title("Outlier Analysis of Customer Features")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "outlier_boxplots.png")

    # 6. Pairwise feature relationships using a sample for readability.
    sample = customers.sample(n=min(500, len(customers)), random_state=42)
    grid = sns.pairplot(
        sample[NUMERIC_FEATURES + [TARGET]],
        hue=TARGET, diag_kind="hist",
        plot_kws={"alpha": 0.45, "s": 15}
    )
    grid.fig.suptitle("Relationships Among Customer Features", y=1.02)
    grid.savefig(OUTPUT_DIR / "feature_relationships.png", dpi=130, bbox_inches="tight")
    plt.close(grid.fig)

    print(f"\nEDA finished. All figures saved in: {OUTPUT_DIR}")
    print("No chart windows were opened.")


if __name__ == "__main__":
    main()
