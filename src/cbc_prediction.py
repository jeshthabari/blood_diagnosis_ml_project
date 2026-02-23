import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cbc_augmented_with_ids.csv")
df = pd.read_csv(DATA_PATH)

# Ensure target exists
if "target" not in df.columns:
    df["Label_clean"] = df["Label"].astype(str).str.lower().str.strip()
    df["target"] = df["Label_clean"].apply(lambda x: 0 if "normal" in x else 1)

# Keep numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("target")

# =========================
# FINAL MODEL (Diagnostic)
# =========================
X_final = df[numeric_cols].fillna(df[numeric_cols].median())
y = df["target"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_final, y, stratify=y, test_size=0.2, random_state=42
)

final_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_f, y_train_f)

final_probs = final_model.predict_proba(X_test_f)[:,1]
final_auc = roc_auc_score(y_test_f, final_probs)
final_acc = accuracy_score(y_test_f, final_model.predict(X_test_f))

print("\n===== FINAL DIAGNOSTIC MODEL =====")
print("AUC:", final_auc)
print("Accuracy:", final_acc)
print(classification_report(y_test_f, final_model.predict(X_test_f)))

# =========================
# EARLY MODEL (Remove M_Protein)
# =========================
early_features = [c for c in numeric_cols if c != "M_Protein"]

X_early = df[early_features].fillna(df[early_features].median())

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_early, y, stratify=y, test_size=0.2, random_state=42
)

early_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

early_model.fit(X_train_e, y_train_e)

early_probs = early_model.predict_proba(X_test_e)[:,1]
early_auc = roc_auc_score(y_test_e, early_probs)
early_acc = accuracy_score(y_test_e, early_model.predict(X_test_e))

print("\n===== EARLY RISK MODEL =====")
print("AUC:", early_auc)
print("Accuracy:", early_acc)
print(classification_report(y_test_e, early_model.predict(X_test_e)))

# =========================
# FEATURE IMPORTANCE (Early Model)
# =========================
importances = pd.Series(
    early_model.feature_importances_,
    index=early_features
).sort_values(ascending=False)

print("\nTop 10 Early Predictors:")
print(importances.head(10))

# =========================
# SAVE RESULTS
# =========================
results = {
    "Final_AUC": final_auc,
    "Final_Accuracy": final_acc,
    "Early_AUC": early_auc,
    "Early_Accuracy": early_acc
}

pd.DataFrame([results]).to_csv("cbc_model_comparison.csv", index=False)