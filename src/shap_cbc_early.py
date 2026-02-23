# =====================================
# SHAP ANALYSIS - CBC EARLY RISK MODEL
# Thalassemia Project
# =====================================

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# 1. Project Root (Portable)
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "cbc_augmented_with_ids.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_random_forest.pkl"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)

print("Project root:", PROJECT_ROOT)

# --------------------------------------------------
# 2. Load Data (Corrected)
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Drop non-feature columns
drop_cols = []

if "patient_id" in df.columns:
    drop_cols.append("patient_id")

if "Garbage_Category" in df.columns:
    drop_cols.append("Garbage_Category")

df = df.drop(columns=drop_cols)

# Define target
TARGET_COLUMN = "Label"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("Final feature columns:", X.columns.tolist())
# --------------------------------------------------
# 3. Load Model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# --------------------------------------------------
# 4. Compute SHAP
# --------------------------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Binary classification
shap_vals = shap_values[1]

# --------------------------------------------------
# 5. Global Importance
# --------------------------------------------------

mean_shap = np.abs(shap_vals).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_SHAP": mean_shap
}).sort_values(by="Mean_SHAP", ascending=False)

print("\nTop 15 SHAP Features:")
print(importance_df.head(15))

importance_df.to_csv(
    RESULTS_PATH / "cbc_early_shap_importance.csv",
    index=False
)

# --------------------------------------------------
# 6. Noise vs Biological Analysis
# --------------------------------------------------

noise_cols = [col for col in X_test.columns if "Random_Noise" in col]
bio_cols = [col for col in X_test.columns if col not in noise_cols]

noise_importance = importance_df[
    importance_df["Feature"].isin(noise_cols)
]["Mean_SHAP"].sum()

bio_importance = importance_df[
    importance_df["Feature"].isin(bio_cols)
]["Mean_SHAP"].sum()

noise_percentage = (
    noise_importance /
    (noise_importance + bio_importance)
) * 100

print("\nNoise Contribution %:", noise_percentage)

summary_df = pd.DataFrame({
    "Biological_SHAP": [bio_importance],
    "Noise_SHAP": [noise_importance],
    "Noise_Percentage": [noise_percentage]
})

summary_df.to_csv(
    RESULTS_PATH / "cbc_early_noise_analysis.csv",
    index=False
)

# --------------------------------------------------
# 7. SHAP Summary Plot
# --------------------------------------------------

plt.figure()
shap.summary_plot(shap_vals, X_test, show=False)
plt.tight_layout()
plt.savefig(
    RESULTS_PATH / "cbc_early_shap_summary.png",
    dpi=300
)
plt.close()

print("SHAP summary plot saved.")

# --------------------------------------------------
# 8. Permutation Noise Test
# --------------------------------------------------

baseline_auc = roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:, 1]
)

print("Baseline AUC:", baseline_auc)

perm_results = []

for col in noise_cols:
    X_temp = X_test.copy()
    X_temp[col] = np.random.permutation(X_temp[col])

    auc = roc_auc_score(
        y_test,
        model.predict_proba(X_temp)[:, 1]
    )

    perm_results.append({
        "Feature": col,
        "Permuted_AUC": auc,
        "AUC_Drop": baseline_auc - auc
    })

perm_df = pd.DataFrame(perm_results)

perm_df.to_csv(
    RESULTS_PATH / "cbc_early_noise_permutation.csv",
    index=False
)

print("Permutation test completed.")
print("SHAP analysis finished.")