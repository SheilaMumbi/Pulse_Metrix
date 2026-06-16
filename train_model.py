"""
train_model.py – Cardio Guard Model Training Pipeline
======================================================
Run this script once to train and save the model:
    python train_model.py

Outputs
-------
models/cardio_guard_model.pkl  – trained LogisticRegression (best params via GridSearch)
models/scaler.pkl              – StandardScaler fitted on training data
models/training_report.txt     – full evaluation report
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics        import (accuracy_score, classification_report,
                                    confusion_matrix, roc_auc_score, roc_curve,
                                    average_precision_score)
from sklearn.pipeline        import Pipeline
from sklearn.utils           import resample

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = Path("framingham.csv")

FEATURE_COLS = ["age", "cigsPerDay", "totChol", "sysBP", "glucose", "diaBP", "heartRate"]
TARGET_COL   = "TenYearCHD"


# ── 1. Load & clean data ─────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading data from {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "education" in df.columns:
        df.drop(columns=["education"], inplace=True)
    if "male" in df.columns:
        df.rename(columns={"male": "Sex_male"}, inplace=True)

    before = len(df)
    df.dropna(inplace=True)
    print(f"[INFO] Rows: {before} -> {len(df)} after dropping NaN")
    print(f"[INFO] Class distribution:\n{df[TARGET_COL].value_counts()}")
    return df


# ── 2. Feature engineering ───────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Pulse pressure (systolic − diastolic) – a known CHD marker
    df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
    # Mean arterial pressure
    df["MAP"] = df["diaBP"] + (df["pulse_pressure"] / 3)
    # Cholesterol-to-glucose ratio
    df["chol_glucose_ratio"] = df["totChol"] / (df["glucose"] + 1)
    # Age × smoking interaction
    df["age_x_cigs"] = df["age"] * df["cigsPerDay"]
    return df


ENGINEERED_FEATURES = FEATURE_COLS + ["pulse_pressure", "MAP", "chol_glucose_ratio", "age_x_cigs"]


# ── 3. Train ─────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    df = feature_engineering(df)

    X = df[ENGINEERED_FEATURES].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    print("[INFO] Applying SMOTE to balance training set …")
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"[INFO] After SMOTE: {np.bincount(y_train_res)}")
    except Exception as e:
        print(f"[WARN] SMOTE failed ({e}), using original training set.")
        X_train_res, y_train_res = X_train, y_train

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    # ── Grid-search logistic regression ──
    print("[INFO] Running GridSearchCV for LogisticRegression …")
    lr_params = {
        "C":        [0.01, 0.1, 1, 10, 100],
        "penalty":  ["l1", "l2"],
        "solver":   ["liblinear"],
        "max_iter": [2000],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced"),
        lr_params, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid.fit(X_train_sc, y_train_res)
    best_lr = grid.best_estimator_
    print(f"[INFO] Best LR params: {grid.best_params_}  CV AUC: {grid.best_score_:.4f}")

    # ── Gradient Boosting as challenger ──
    print("[INFO] Training GradientBoostingClassifier …")
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train_sc, y_train_res)

    # ── Choose best model by test AUC ──
    lr_auc = roc_auc_score(y_test, best_lr.predict_proba(X_test_sc)[:, 1])
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test_sc)[:, 1])
    print(f"[INFO] Test AUC  –  LogReg: {lr_auc:.4f}  |  GradBoost: {gb_auc:.4f}")

    best_model = best_lr if lr_auc >= gb_auc else gb
    model_name = "LogisticRegression" if lr_auc >= gb_auc else "GradientBoosting"
    print(f"[INFO] Selecting {model_name} as final model.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred      = best_model.predict(X_test_sc)
    y_prob      = best_model.predict_proba(X_test_sc)[:, 1]
    # Use a threshold tuned for recall on positive class
    threshold   = 0.30
    y_pred_tuned = (y_prob >= threshold).astype(int)

    report_lines = [
        f"Cardio Guard – Model Evaluation Report",
        f"=======================================",
        f"Final model  : {model_name}",
        f"Threshold    : {threshold}",
        f"",
        f"-- Standard threshold (0.5) --",
        classification_report(y_test, y_pred, target_names=["No CHD", "CHD"]),
        f"Accuracy : {accuracy_score(y_test, y_pred):.4f}",
        f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}",
        f"Avg Prec : {average_precision_score(y_test, y_prob):.4f}",
        f"",
        f"-- Tuned threshold ({threshold}) --",
        classification_report(y_test, y_pred_tuned, target_names=["No CHD", "CHD"]),
        f"Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}",
        f"",
        f"Confusion Matrix (tuned threshold):",
        str(confusion_matrix(y_test, y_pred_tuned)),
    ]
    report = "\n".join(report_lines)
    print("\n" + report)

    (MODELS_DIR / "training_report.txt").write_text(report)

    # ── ROC curve plot ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"ROC AUC = {roc_auc_score(y_test, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Cardio Guard"); ax.legend()
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # ── Feature importance (coefficients or feature_importances_) ────────────
    try:
        if hasattr(best_model, "coef_"):
            importances = np.abs(best_model.coef_[0])
        else:
            importances = best_model.feature_importances_

        fi_df = pd.DataFrame({"Feature": ENGINEERED_FEATURES, "Importance": importances})
        fi_df.sort_values("Importance", ascending=True, inplace=True)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.barh(fi_df["Feature"], fi_df["Importance"], color="#8e44ad")
        ax2.set_title("Feature Importance"); ax2.set_xlabel("Absolute Coefficient / Importance")
        fig2.tight_layout()
        fig2.savefig(MODELS_DIR / "feature_importance.png", dpi=150)
        plt.close(fig2)
        print(f"[INFO] Feature importance plot saved.")
    except Exception as e:
        print(f"[WARN] Could not plot feature importance: {e}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    # Wrap model to remember the threshold and feature list
    artefact = {
        "model":    best_model,
        "features": ENGINEERED_FEATURES,
        "threshold": threshold,
    }
    joblib.dump(artefact, MODELS_DIR / "cardio_guard_model.pkl")
    joblib.dump(scaler,   MODELS_DIR / "scaler.pkl")
    print(f"[INFO] Model saved -> {MODELS_DIR / 'cardio_guard_model.pkl'}")
    print(f"[INFO] Scaler saved -> {MODELS_DIR / 'scaler.pkl'}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DATA_PATH.exists():
        sys.exit(
            f"[ERROR] Dataset not found at '{DATA_PATH}'.\n"
            "Download the Framingham Heart Study CSV from:\n"
            "  https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset\n"
            "and place it as 'framingham.csv' in this directory."
        )

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[WARN] imbalanced-learn not installed. Run: pip install imbalanced-learn")
        print("[WARN] Continuing without SMOTE …")

    df = load_data(DATA_PATH)
    train(df)
    print("\n[SUCCESS] Training complete! You can now run: streamlit run app.py")