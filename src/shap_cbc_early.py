# =====================================
# SHAP ANALYSIS - CBC EARLY RISK MODEL
# Supports Binary + Multi-Class
# =====================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# 1. Project Root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cbc_augmented_with_ids.csv"
RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)

print("Project root:", PROJECT_ROOT)

# --------------------------------------------------
# 2. Load Data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

drop_cols = []
if "patient_id" in df.columns:
    drop_cols.append("patient_id")
if "Garbage_Category" in df.columns:
    drop_cols.append("Garbage_Category")

df = df.drop(columns=drop_cols)

TARGET_COLUMN = "Label"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Convert to binary
# 0 = Normal
# 1 = Disease (any non-zero class)

y = y.apply(lambda x: 0 if x == 0 else 1)

print("\nConverted to Binary Classification")
print("New class distribution:")
print(y.value_counts())
print("Unique labels:", y.unique())

num_classes = len(np.unique(y))
print("Number of classes:", num_classes)

# --------------------------------------------------
# 3. Train/Test Split
# --------------------------------------------------

if y.value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# --------------------------------------------------
# 4. Train Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully.")

# --------------------------------------------------
# 5. Compute AUC (Robust Version)
# --------------------------------------------------

probs = model.predict_proba(X_test)
classes = model.classes_

if len(classes) == 2:
    baseline_auc = roc_auc_score(
        y_test,
        probs[:, 1]
    )
else:
    baseline_auc = roc_auc_score(
        y_test,
        probs,
        labels=classes,
        multi_class="ovr"
    )

print("Baseline AUC:", baseline_auc)

# --------------------------------------------------
# 6. SHAP Analysis
# --------------------------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For multi-class, average absolute SHAP across classes
if isinstance(shap_values, list):
    shap_vals = np.mean(
        [np.abs(class_vals) for class_vals in shap_values],
        axis=0
    )
else:
    shap_vals = shap_values

# --------------------------------------------------
# 7. Global Importance
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
# 8. Noise Contribution
# --------------------------------------------------

noise_cols = [col for col in X_test.columns if "Random_Noise" in col]
bio_cols = [col for col in X_test.columns if col not in noise_cols]

noise_importance = importance_df[
    importance_df["Feature"].isin(noise_cols)
]["Mean_SHAP"].sum()

bio_importance = importance_df[
    importance_df["Feature"].isin(bio_cols)
]["Mean_SHAP"].sum()

noise_percentage = (noise_importance / (noise_importance + bio_importance)) * 100

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
# 9. SHAP Summary Plot
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
# 10. Permutation Noise Test
# --------------------------------------------------

perm_results = []

for col in noise_cols:
    X_temp = X_test.copy()
    X_temp[col] = np.random.permutation(X_temp[col])

    temp_probs = model.predict_proba(X_temp)

    if len(classes) == 2:
        auc = roc_auc_score(
            y_test,
            temp_probs[:, 1]
        )
    else:
        auc = roc_auc_score(
            y_test,
            temp_probs,
            labels=classes,
            multi_class="ovr"
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
print("SHAP analysis finished successfully.")