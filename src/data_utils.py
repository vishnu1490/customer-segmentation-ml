"""Shared data loading and customer feature engineering utilities."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "online_retail_ii.xlsx"
FEATURES_PATH = ROOT / "results" / "customer_features.csv"

NUMERIC_FEATURES = [
    "RecencyDays",
    "PurchaseFrequency",
    "TotalSpending",
    "AverageOrderValue",
    "TotalQuantity",
    "AverageItemsPerInvoice",
]
TARGET = "CustomerSegment"
SEGMENT_ORDER = ["Low Engagement", "Regular", "High Value"]


def load_transactions():
    """Load and combine both sheets from the supplied Online Retail II workbook."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place online_retail_ii.xlsx in the project's data folder."
        )

    try:
        sheets = pd.read_excel(DATA_PATH, sheet_name=None, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            "Could not read the Excel dataset. Ensure openpyxl is installed "
            "and data/online_retail_ii.xlsx is a valid workbook."
        ) from exc

    frames = [frame for frame in sheets.values() if not frame.empty]
    if not frames:
        raise ValueError("The Excel workbook contains no transaction rows.")

    df = pd.concat(frames, ignore_index=True)
    required = [
        "Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Remove rows without a customer/date and remove returns/cancellations
    # from the purchase-behaviour features used in this educational project.
    df = df.dropna(subset=["Customer ID", "InvoiceDate", "Quantity", "Price"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    if df.empty:
        raise ValueError("No valid positive-quantity purchase rows remain.")

    df["Customer ID"] = df["Customer ID"].astype("int64")
    df["LineRevenue"] = df["Quantity"] * df["Price"]
    return df


def build_customer_features(transactions):
    """Aggregate transaction lines into one row per customer and define RFM labels."""
    df = transactions.copy()

    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    customer = df.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        PurchaseFrequency=("Invoice", "nunique"),
        TotalSpending=("LineRevenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        InvoiceLineCount=("Invoice", "size"),
    ).reset_index()

    customer["RecencyDays"] = (
        reference_date - customer["LastPurchase"].dt.normalize()
    ).dt.days.clip(lower=0)

    customer["AverageOrderValue"] = (
        customer["TotalSpending"] / customer["PurchaseFrequency"].clip(lower=1)
    )
    customer["AverageItemsPerInvoice"] = (
        customer["TotalQuantity"] / customer["PurchaseFrequency"].clip(lower=1)
    )

    # Labels are created with a transparent RFM rule. Recency is reversed:
    # a more recent customer receives a higher recency score.
    # duplicates='drop' handles tied values in small or unusual datasets.
    customer["R_Score"] = (
        pd.qcut(
            customer["RecencyDays"].rank(method="first"),
            5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
    )
    customer["F_Score"] = (
        pd.qcut(
            customer["PurchaseFrequency"].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
    )
    customer["M_Score"] = (
        pd.qcut(
            customer["TotalSpending"].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
    )

    customer["RFM_Score"] = (
        customer["R_Score"] + customer["F_Score"] + customer["M_Score"]
    )

    # Three easy-to-explain labels for supervised classification.
    # These labels are rule-derived, not independent human annotations.
    customer[TARGET] = np.select(
        [
            customer["RFM_Score"] >= 12,
            customer["RFM_Score"] >= 8,
        ],
        [
            "High Value",
            "Regular",
        ],
        default="Low Engagement",
    )

    customer = customer.replace([np.inf, -np.inf], np.nan)
    customer = customer.dropna(subset=NUMERIC_FEATURES + [TARGET]).copy()
    customer[TARGET] = pd.Categorical(
        customer[TARGET], categories=SEGMENT_ORDER
    )
    return customer


def load_or_build_customer_features(force_rebuild=False):
    """Read cached engineered data, or create it from the provided workbook."""
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if FEATURES_PATH.exists() and not force_rebuild:
        customer = pd.read_csv(FEATURES_PATH)
        needed = NUMERIC_FEATURES + [TARGET]
        if all(col in customer.columns for col in needed):
            customer[TARGET] = pd.Categorical(
                customer[TARGET], categories=SEGMENT_ORDER
            )
            return customer

    transactions = load_transactions()
    customer = build_customer_features(transactions)
    customer.to_csv(FEATURES_PATH, index=False)
    return customer
