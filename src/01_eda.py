"""
01_eda.py — Exploratory Data Analysis
======================================
Dataset : UCI Heart Disease (Cleveland)
Goal    : Understand the data before modelling — distributions,
          correlations, missing values, class balance.

Run from project root:
    python src/01_eda.py
All figures are saved to reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DATA_PATH   = "data/heart.csv"
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── column names (UCI specification) ────────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Download once; use cached CSV thereafter."""
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset …")
        df = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values="?")
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved to {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH, na_values="?")
    return df


# ── 2. Inspect ───────────────────────────────────────────────────────────────
def inspect(df: pd.DataFrame) -> None:
    print("\n── Shape ──────────────────────────────")
    print(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

    print("\n── First 5 rows ───────────────────────")
    print(df.head())

    print("\n── Data types ─────────────────────────")
    print(df.dtypes)

    print("\n── Missing values ─────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n── Descriptive statistics ─────────────")
    print(df.describe().round(2))


# ── 3. Target engineering ────────────────────────────────────────────────────
def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Original target: 0 = no disease, 1-4 = varying severity.
    We map to 3 risk levels that are clinically intuitive and
    easy to explain in a viva:
        0      → Low  (0)
        1-2    → Medium (1)
        3-4    → High (2)
    """
    def map_risk(v):
        if v == 0:
            return 0   # Low
        elif v <= 2:
            return 1   # Medium
        else:
            return 2   # High

    df = df.copy()
    df["risk"] = df["target"].apply(map_risk)
    df.drop(columns=["target"], inplace=True)
    return df


# ── 4. Plot: class distribution ──────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    labels = {0: "Low", 1: "Medium", 2: "High"}
    counts = df["risk"].value_counts().sort_index()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([labels[i] for i in counts.index], counts.values, color=colors, width=0.5)

    # add count labels on bars
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Risk Level Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_class_distribution.png", dpi=150)
    plt.close()
    print("Saved: 01_class_distribution.png")


# ── 5. Plot: feature distributions ───────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> None:
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    palette = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}

    for i, col in enumerate(numeric):
        ax = axes[i]
        for risk, color in palette.items():
            subset = df[df["risk"] == risk][col].dropna()
            ax.hist(subset, bins=15, alpha=0.6, color=color,
                    label=risk_labels[risk], edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # hide unused subplot
    axes[-1].set_visible(False)

    fig.suptitle("Feature Distributions by Risk Level", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_feature_distributions.png")


# ── 6. Plot: correlation heatmap ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # show lower triangle only

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: 03_correlation_heatmap.png")


# ── 7. Plot: categorical features ────────────────────────────────────────────
def plot_categorical_features(df: pd.DataFrame) -> None:
    """
    Categorical columns and their natural labels.
    We use countplots split by risk level.
    """
    cat_cols = {
        "sex":     {0: "Female", 1: "Male"},
        "cp":      {0: "Typical", 1: "Atypical", 2: "Non-anginal", 3: "Asymptomatic"},
        "fbs":     {0: "FBS ≤ 120", 1: "FBS > 120"},
        "exang":   {0: "No", 1: "Yes"},
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    palette = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    for i, (col, labels) in enumerate(cat_cols.items()):
        ax = axes[i]
        plot_df = df[[col, "risk"]].copy()
        plot_df[col] = plot_df[col].map(labels)
        plot_df["risk"] = plot_df["risk"].map(risk_labels)

        sns.countplot(data=plot_df, x=col, hue="risk", ax=ax,
                      palette={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
        ax.set_title(col.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Risk", fontsize=8)

    fig.suptitle("Categorical Features by Risk Level", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_categorical_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_categorical_features.png")


# ── 8. Missing value summary ─────────────────────────────────────────────────
def print_missing_summary(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    summary = summary[summary["Missing Count"] > 0]

    if summary.empty:
        print("\n✓ No missing values in this dataset after loading.")
    else:
        print("\n── Missing value summary ───────────────")
        print(summary)
        print("\nStrategy: median imputation for numeric, mode for categorical.")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    inspect(df)
    df = engineer_target(df)
    print_missing_summary(df)

    print("\nGenerating plots …")
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_features(df)

    print("\n✓ EDA complete. Figures saved to reports/figures/")
    print("Next step → run: python src/02_preprocess_and_train.py")
