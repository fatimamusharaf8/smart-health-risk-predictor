"""
02_preprocess_and_train.py — Preprocessing + Model Training
============================================================
What this script does, in order:
  1. Load the raw UCI Heart Disease CSV
  2. Engineer the 3-class risk target (Low / Medium / High)
  3. Handle missing values (median imputation)
  4. Encode categorical variables
  5. Split into train / test (80/20, stratified)
  6. Scale numeric features with StandardScaler
  7. Train three models:
       a) Logistic Regression (baseline, interpretable)
       b) Random Forest       (ensemble, handles non-linearity)
       c) XGBoost             (gradient boosting, often best)
  8. Evaluate all three and print a comparison table
  9. Save the best model + preprocessor to models/

Run from project root:
    python src/02_preprocess_and_train.py
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/heart.csv"
MODELS_DIR  = "models"
FIGURES_DIR = "reports/figures"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

# ── features ─────────────────────────────────────────────────────────────────
# Numeric features → impute + scale
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical features → already encoded as integers in this dataset,
# but we treat them as categorical so we don't scale them
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, na_values="?")
    if "target" not in df.columns:
        df.columns = COLUMNS
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TARGET ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Why 3 classes?
      — Binary (disease/no disease) is common but less clinically actionable.
      — 3 classes (Low/Medium/High) give more nuance and make the
        classification task more interesting for a portfolio project.

    Mapping:
      0     → 0 (Low risk)
      1, 2  → 1 (Medium risk)
      3, 4  → 2 (High risk)
    """
    def _map(v):
        if   v == 0: return 0
        elif v <= 2: return 1
        else:        return 2

    df = df.copy()
    df["risk"] = df["target"].apply(_map)
    df.drop(columns=["target"], inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """
    Returns:
        X_train, X_test, y_train, y_test, scaler

    Steps:
      a) Separate features and target
      b) Median imputation for missing values in numeric cols
      c) Mode  imputation for missing values in categorical cols
      d) Train/test split (stratified to preserve class balance)
      e) StandardScaler on numeric features only (fit on train, transform both)

    Why StandardScaler?
      Logistic Regression is sensitive to feature magnitudes.
      Tree-based models don't need scaling, but it doesn't hurt them,
      so applying it uniformly keeps the pipeline consistent.
    """
    X = df[ALL_FEATURES].copy()
    y = df["risk"].copy()

    # ── imputation ─────────────────────────────────────────────────────────
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X[NUMERIC_FEATURES]     = num_imputer.fit_transform(X[NUMERIC_FEATURES])
    X[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(X[CATEGORICAL_FEATURES])

    # ── train / test split (80 / 20) ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── scaling (numeric features only) ───────────────────────────────────
    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES]  = scaler.transform(X_test[NUMERIC_FEATURES])

    print(f"Train size : {X_train.shape[0]} rows")
    print(f"Test  size : {X_test.shape[0]}  rows")
    print(f"Class dist : {dict(y_train.value_counts().sort_index())}")

    # save imputers & scaler for inference
    joblib.dump(num_imputer, f"{MODELS_DIR}/num_imputer.pkl")
    joblib.dump(cat_imputer, f"{MODELS_DIR}/cat_imputer.pkl")
    joblib.dump(scaler,      f"{MODELS_DIR}/scaler.pkl")

    return X_train, X_test, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_models() -> dict:
    """
    Three models, each with a clear rationale:

    Logistic Regression
      + Highly interpretable (coefficients map directly to feature impact)
      + Great baseline; fast to train
      − Assumes linear decision boundary → may underfit complex patterns

    Random Forest
      + Handles non-linear relationships without feature engineering
      + Naturally resistant to overfitting via averaging many trees
      + Provides feature importance scores
      − Less interpretable than LR; slower inference

    XGBoost
      + State-of-the-art on tabular data; regularised to prevent overfitting
      + Built-in handling of missing values (less important here, but good habit)
      + Often achieves best accuracy on medical datasets
      − More hyperparameters; slightly more complex to explain
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="mlogloss",  # suppresses a warning
            verbosity=0
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """
    Train, predict, and compute all metrics.
    Returns a flat dict of results for comparison.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 5-fold CV accuracy on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")

    results = {
        "model":       name,
        "accuracy":    round(accuracy_score(y_test, y_pred),                               4),
        "precision":   round(precision_score(y_test, y_pred, average="macro",  zero_division=0), 4),
        "recall":      round(recall_score   (y_test, y_pred, average="macro",  zero_division=0), 4),
        "f1_macro":    round(f1_score       (y_test, y_pred, average="macro",  zero_division=0), 4),
        "cv_f1_mean":  round(cv_scores.mean(),                                              4),
        "cv_f1_std":   round(cv_scores.std(),                                               4),
    }

    print(f"\n── {name} ─────────────────────────────")
    print(f"  Accuracy  : {results['accuracy']}")
    print(f"  Precision : {results['precision']}")
    print(f"  Recall    : {results['recall']}")
    print(f"  F1 (macro): {results['f1_macro']}")
    print(f"  CV F1     : {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Low", "Medium", "High"], zero_division=0))

    return results, model, y_pred


# ── confusion matrix plot ────────────────────────────────────────────────────
def plot_confusion_matrix(name: str, y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{FIGURES_DIR}/cm_{safe_name}.png", dpi=150)
    plt.close()
    print(f"Saved: cm_{safe_name}.png")


# ── model comparison bar chart ───────────────────────────────────────────────
def plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_macro"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for ax, metric in zip(axes, metrics):
        vals = comparison_df.set_index("model")[metric]
        bars = ax.bar(vals.index, vals.values, color=colors, width=0.5)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(vals.index, rotation=20, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/05_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(model, model_name: str) -> None:
    """
    Works for tree-based models (Random Forest, XGBoost).
    Shows which features drive the predictions the most.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} has no feature_importances_ — skipping.")
        return

    importances = pd.Series(model.feature_importances_, index=ALL_FEATURES)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if i >= len(importances) - 3 else "#3498db"
              for i in range(len(importances))]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"{FIGURES_DIR}/fi_{safe_name}.png", dpi=150)
    plt.close()
    print(f"Saved: fi_{safe_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
def save_best_model(comparison_df: pd.DataFrame, trained_models: dict) -> str:
    """
    Select the model with the highest macro F1 score.
    Save it along with:
      - feature names  (so the API knows the input order)
      - label mapping  (0→Low, 1→Medium, 2→High)
      - comparison results (for reporting)
    """
    best_row   = comparison_df.loc[comparison_df["f1_macro"].idxmax()]
    best_name  = best_row["model"]
    best_model = trained_models[best_name]

    print(f"\n✓ Best model: {best_name} (F1 macro = {best_row['f1_macro']})")

    joblib.dump(best_model,       f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(ALL_FEATURES,     f"{MODELS_DIR}/feature_names.pkl")
    joblib.dump(NUMERIC_FEATURES, f"{MODELS_DIR}/numeric_features.pkl")
    joblib.dump(CATEGORICAL_FEATURES, f"{MODELS_DIR}/categorical_features.pkl")

    # save metadata as JSON (human-readable)
    metadata = {
        "best_model":     best_name,
        "features":       ALL_FEATURES,
        "numeric":        NUMERIC_FEATURES,
        "categorical":    CATEGORICAL_FEATURES,
        "label_map":      {"0": "Low", "1": "Medium", "2": "High"},
        "metrics": {
            "accuracy":  best_row["accuracy"],
            "f1_macro":  best_row["f1_macro"],
        }
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved artifacts to {MODELS_DIR}/")
    return best_name


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Health Risk Predictor — Training Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Loading data …")
    df = load_data()

    # 2. Target
    print("[2/7] Engineering target …")
    df = engineer_target(df)

    # 3. Preprocess
    print("[3/7] Preprocessing …")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # 4. Train & evaluate all models
    print("\n[4/7] Training and evaluating models …")
    models      = get_models()
    results_list = []
    trained_models = {}
    predictions    = {}

    for name, model in models.items():
        res, trained, y_pred = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results_list.append(res)
        trained_models[name] = trained
        predictions[name]    = y_pred

    # 5. Compare
    comparison_df = pd.DataFrame(results_list)
    print("\n── Model Comparison Summary ───────────────────────────────")
    print(comparison_df.to_string(index=False))

    # 6. Plots
    print("\n[5/7] Generating plots …")
    for name, y_pred in predictions.items():
        plot_confusion_matrix(name, y_test, y_pred)

    plot_model_comparison(comparison_df)

    for name, model in trained_models.items():
        plot_feature_importance(model, name)

    # 7. Save
    print("\n[6/7] Saving best model …")
    best = save_best_model(comparison_df, trained_models)

    print("\n[7/7] Done.")
    print(f"\n✓ Best model '{best}' saved to {MODELS_DIR}/best_model.pkl")
    print("Next step → run: python src/03_tune.py")
