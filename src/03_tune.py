"""
03_tune.py — Hyperparameter Tuning
====================================
We tune the best model (usually XGBoost or Random Forest) using GridSearchCV.

Why GridSearchCV?
  — Exhaustively searches every combination in the grid
  — Uses cross-validation internally (5-fold) → more honest than a single split
  — Controlled and reproducible

Run from project root:
    python src/03_tune.py
"""

import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import f1_score, classification_report
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH   = "data/heart.csv"
MODELS_DIR  = "models"
FIGURES_DIR = "reports/figures"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

NUMERIC_FEATURES     = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_prepare():
    """Reproduce the same preprocessing as training."""
    df = pd.read_csv(DATA_PATH, na_values="?")
    if "target" not in df.columns:
        df.columns = COLUMNS

    # target
    def map_risk(v):
        if v == 0: return 0
        elif v <= 2: return 1
        else: return 2

    df["risk"] = df["target"].apply(map_risk)
    df.drop(columns=["target"], inplace=True)

    X = df[ALL_FEATURES].copy()
    y = df["risk"].copy()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[NUMERIC_FEATURES]     = num_imputer.fit_transform(X[NUMERIC_FEATURES])
    X[CATEGORICAL_FEATURES] = cat_imputer.fit_transform(X[CATEGORICAL_FEATURES])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES]  = scaler.transform(X_test[NUMERIC_FEATURES])

    return X_train, X_test, y_train, y_test


def get_param_grid(model_name: str) -> tuple:
    """
    Parameter grids are intentionally modest to keep training time
    under a minute on a standard laptop.

    For a viva, you can explain each parameter:
      n_estimators  : number of trees / boosting rounds
      max_depth     : how deep each tree can grow (controls overfitting)
      learning_rate : step size for gradient updates (XGBoost)
      subsample     : fraction of rows used per tree (reduces overfitting)
    """
    if model_name in ("XGBoost", "xgboost"):
        model = XGBClassifier(
            random_state=42, eval_metric="mlogloss", verbosity=0
        )
        param_grid = {
            "n_estimators":  [100, 200],
            "max_depth":     [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample":     [0.8, 1.0],
        }
    else:
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators":       [100, 200, 300],
            "max_depth":          [None, 5, 10],
            "min_samples_split":  [2, 5],
            "min_samples_leaf":   [1, 2],
        }
    return model, param_grid


def run_grid_search(model, param_grid, X_train, y_train):
    print(f"\nRunning GridSearchCV with {len(X_train)} training samples …")
    print(f"Grid size: {pd.DataFrame([param_grid]).apply(lambda c: len(c[0])).prod()} combinations × 5 folds")

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs


def plot_before_after(before_f1: float, after_f1: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Before tuning", "After tuning"],
                  [before_f1, after_f1],
                  color=["#95a5a6", "#2ecc71"], width=0.4)
    for bar, val in zip(bars, [before_f1, after_f1]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 Score (test set)")
    ax.set_title("Hyperparameter Tuning — Before vs After", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/06_tuning_comparison.png", dpi=150)
    plt.close()
    print("Saved: 06_tuning_comparison.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Health Risk Predictor — Hyperparameter Tuning")
    print("=" * 60)

    # load metadata to find which model was best
    with open(f"{MODELS_DIR}/metadata.json") as f:
        meta = json.load(f)
    best_model_name = meta["best_model"]
    before_f1 = meta["metrics"]["f1_macro"]

    print(f"\nTuning: {best_model_name}  (baseline F1 = {before_f1})")

    X_train, X_test, y_train, y_test = load_and_prepare()
    model, param_grid = get_param_grid(best_model_name)

    # grid search
    gs = run_grid_search(model, param_grid, X_train, y_train)

    print(f"\n── Best parameters ─────────────────────")
    for k, v in gs.best_params_.items():
        print(f"  {k}: {v}")
    print(f"  CV F1 (mean): {gs.best_score_:.4f}")

    # evaluate on held-out test set
    y_pred = gs.best_estimator_.predict(X_test)
    after_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n── Test set performance ─────────────────")
    print(f"  Before tuning F1 : {before_f1:.4f}")
    print(f"  After  tuning F1 : {after_f1:.4f}")
    delta = after_f1 - before_f1
    print(f"  Improvement      : {delta:+.4f}")
    print("\nClassification Report (tuned model):")
    print(classification_report(y_test, y_pred,
          target_names=["Low", "Medium", "High"], zero_division=0))

    # save tuned model (overwrite best_model.pkl)
    joblib.dump(gs.best_estimator_, f"{MODELS_DIR}/best_model.pkl")
    print(f"\n✓ Tuned model saved to {MODELS_DIR}/best_model.pkl")

    # update metadata
    meta["best_params"] = gs.best_params_
    meta["metrics"]["f1_macro_tuned"] = round(after_f1, 4)
    meta["metrics"]["f1_macro_before_tuning"] = before_f1
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # visualise improvement
    plot_before_after(before_f1, after_f1)

    print("\n✓ Tuning complete.")
    print("Next step → run: uvicorn api.main:app --reload")
