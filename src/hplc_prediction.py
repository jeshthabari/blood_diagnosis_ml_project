# =====================================
# HPLC EARLY vs FULL MODEL COMPARISON
# With Proper Data Cleaning
# =====================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# --------------------------------------------------
# 1. Project Root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hplc_augmented_with_ids.csv"
RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)

print("Project root:", PROJECT_ROOT)

# --------------------------------------------------
# 2. Load Data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nOriginal Diagnosis distribution:")
print(df["Diagnosis"].value_counts())

# --------------------------------------------------
# 3. CLEANING & TYPE CONVERSION
# --------------------------------------------------

# ---- Convert Age from "12 years" → 12 ----
df["Age"] = df["Age"].astype(str).str.extract(r'(\d+)')
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# ---- Convert binary symptom columns if needed ----
binary_columns = ["Weekness", "Jaundice"]

for col in binary_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower()
        df[col] = df[col].apply(lambda x: 1 if x in ["yes", "1", "true"] else 0)

# --------------------------------------------------
# 4. Convert Diagnosis to Binary
# --------------------------------------------------

# 0 = Normal
# 1 = Any thalassemia

df["Diagnosis"] = df["Diagnosis"].astype(str).str.lower()
df["Diagnosis"] = df["Diagnosis"].apply(
    lambda x: 0 if x == "normal" else 1
)

print("\nBinary Diagnosis distribution:")
print(df["Diagnosis"].value_counts())

# --------------------------------------------------
# 5. Drop Non-Numeric / Non-Useful Columns
# --------------------------------------------------

drop_cols = [
    "Sl No",
    "Gender",
    "Religion",
    "Present District",
    "patient_id"
]

X_full = df.drop(columns=drop_cols + ["Diagnosis"])
y = df["Diagnosis"]

# Ensure all features numeric
X_full = X_full.apply(pd.to_numeric, errors="coerce")

# Fill any numeric NaN values with median
X_full = X_full.fillna(X_full.median())

# --------------------------------------------------
# 6. Define Early Model (Remove Dominant Biomarkers)
# --------------------------------------------------

dominant_markers = [
    "HbA2", "HbA0", "HbF",
    "RBC", "HB", "MCV", "MCH", "MCHC", "RDWcv"
]

early_features = [col for col in X_full.columns if col not in dominant_markers]

X_early = X_full[early_features]

print("\nFull feature count:", X_full.shape[1])
print("Early feature count:", X_early.shape[1])

# --------------------------------------------------
# 7. Train/Test Split
# --------------------------------------------------

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_early = X_train_full[early_features]
X_test_early = X_test_full[early_features]

# --------------------------------------------------
# 8. Train FULL Diagnostic Model
# --------------------------------------------------

full_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

full_model.fit(X_train_full, y_train)

# --------------------------------------------------
# 9. Train EARLY Screening Model
# --------------------------------------------------

early_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

early_model.fit(X_train_early, y_train)

# --------------------------------------------------
# 10. Evaluate Models
# --------------------------------------------------

# FULL MODEL
full_probs = full_model.predict_proba(X_test_full)[:, 1]
full_preds = full_model.predict(X_test_full)

full_auc = roc_auc_score(y_test, full_probs)
full_acc = accuracy_score(y_test, full_preds)

# EARLY MODEL
early_probs = early_model.predict_proba(X_test_early)[:, 1]
early_preds = early_model.predict(X_test_early)

early_auc = roc_auc_score(y_test, early_probs)
early_acc = accuracy_score(y_test, early_preds)

# --------------------------------------------------
# 11. Print Results
# --------------------------------------------------

print("\n===== FULL DIAGNOSTIC MODEL =====")
print("AUC:", full_auc)
print("Accuracy:", full_acc)
print(classification_report(y_test, full_preds))

print("\n===== EARLY SCREENING MODEL =====")
print("AUC:", early_auc)
print("Accuracy:", early_acc)
print(classification_report(y_test, early_preds))

print("\nPerformance Gain (AUC):", full_auc - early_auc)

# --------------------------------------------------
# 12. Save Results
# --------------------------------------------------

comparison_df = pd.DataFrame({
    "Model": ["Full", "Early"],
    "AUC": [full_auc, early_auc],
    "Accuracy": [full_acc, early_acc],
    "AUC_Gain": [0, full_auc - early_auc]
})

comparison_df.to_csv(
    RESULTS_PATH / "hplc_model_comparison.csv",
    index=False
)

print("\nResults saved to results/hplc_model_comparison.csv")
print("HPLC analysis completed successfully.")
importances = pd.Series(
    full_model.feature_importances_,
    index=X_train_full.columns
).sort_values(ascending=False)

print("\nTop 10 Full Model Features:")
print(importances.head(10))