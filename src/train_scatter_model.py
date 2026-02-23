# src/train_scatter_model.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, auc, classification_report)

from ood_detector import fit_mahalanobis, detect_ood

# ========== Config ==========
DATA_PATH = "data/scatter_coordinates.csv"   # change if you want to use generated hard file
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 50

# ========== Load ==========
df = pd.read_csv(DATA_PATH)
print("Scatter labels (sample):", df["Label"].unique())

# ========== Binary mapping ==========
df["Label_clean"] = df["Label"].astype(str).str.strip().str.lower()
df["target"] = df["Label_clean"].apply(lambda x: 0 if x == "normal" else 1)
print("Binary counts:\n", df["target"].value_counts())

# ========== Features ==========
# prefer original coords, but include rotated/scaled if present
candidate_features = ["X", "Y", "X_rot", "Y_rot", "X_scaled", "Y_scaled", "Light_Intensity", "Cell_Size", "M_Protein"]
feature_cols = [c for c in candidate_features if c in df.columns]
if len(feature_cols) == 0:
    raise ValueError("No expected scatter numeric columns found. Check column names.")

X = df[feature_cols].copy()
y = df["target"].copy()

# ==========================================================
#            SCATTER FEATURE CORRELATION
# ==========================================================

corr = X.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
plt.title("Scatter Feature Correlation Matrix")
plt.tight_layout()

corr_path = os.path.join(RESULTS_DIR, "scatter_feature_correlation.png")
plt.savefig(corr_path, dpi=300)
plt.close()

print("Saved scatter correlation heatmap →", corr_path)

# fill missing numerics with median
for c in feature_cols:
    if X[c].isnull().any():
        X[c].fillna(X[c].median(), inplace=True)

# ========== Train/test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ==========================================================
#            RULE BASELINE — SCATTER
# ==========================================================

def scatter_rule_predict(X):

    preds = []

    for _, row in X.iterrows():

        if (
            ("M_Protein" in row and row["M_Protein"] > 1.5) or
            ("Cell_Size" in row and row["Cell_Size"] > X["Cell_Size"].median()) or
            ("Light_Intensity" in row and row["Light_Intensity"] > X["Light_Intensity"].median())
        ):
            preds.append(1)
        else:
            preds.append(0)

    return np.array(preds)


rule_preds = scatter_rule_predict(X_test)

print("\n===== Scatter Rule Baseline =====")
print("Acc:", accuracy_score(y_test, rule_preds))
print("Prec:", precision_score(y_test, rule_preds, zero_division=0))
print("Rec:", recall_score(y_test, rule_preds, zero_division=0))
print("F1:", f1_score(y_test, rule_preds, zero_division=0))

# ========== Models ==========
models = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE)
}

results = []
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # get probability or score
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = np.nan
    pr_auc = np.nan
    if y_score is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except Exception:
            roc_auc = np.nan
        try:
            p_vals, r_vals, _ = precision_recall_curve(y_test, y_score)
            pr_auc = auc(r_vals, p_vals)
        except Exception:
            pr_auc = np.nan

    # Save model
    model_file = os.path.join(RESULTS_DIR, f"scatter_model_{name}.pkl")
    joblib.dump(model, model_file)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Pred Normal","Pred Abnormal"],
                yticklabels=["True Normal","True Abnormal"])
    plt.title(f"Scatter Confusion Matrix - {name}")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"scatter_confmat_{name}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ROC & PR
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1],'k--', alpha=0.5)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"Scatter ROC - {name}")
        plt.legend(loc="lower right")
        roc_path = os.path.join(RESULTS_DIR, f"scatter_roc_{name}.png")
        plt.tight_layout(); plt.savefig(roc_path, dpi=300); plt.close()

        plt.figure(figsize=(6,5))
        plt.plot(r_vals, p_vals, label=f"PR AUC = {pr_auc:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Scatter PR - {name}")
        pr_path = os.path.join(RESULTS_DIR, f"scatter_pr_{name}.png")
        plt.tight_layout(); plt.savefig(pr_path, dpi=300); plt.close()

    print(f"{name} — Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f} ROC_AUC:{roc_auc}")
    results.append({
        "model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "model_file": model_file, "confusion_matrix": cm_path
    })

# Save results summary
summary_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
summary_df.to_csv(os.path.join(RESULTS_DIR, "scatter_model_results.csv"), index=False)
print("\nSaved Scatter summary to", os.path.join(RESULTS_DIR, "scatter_model_results.csv"))

# ==========================================================
#           FEATURE IMPORTANCE — SCATTER
# ==========================================================

print("\nGenerating Scatter Feature Importance...")

for name, model in models.items():

    importance = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", None)
        if clf is not None and hasattr(clf, "coef_"):
            importance = np.abs(clf.coef_[0])

    if importance is not None:

        idx = np.argsort(importance)

        plt.figure(figsize=(6,4))
        plt.barh(range(len(idx)), importance[idx])
        plt.yticks(range(len(idx)), [feature_cols[i] for i in idx])
        plt.title(f"Scatter Importance — {name}")

        save_path = os.path.join(
            RESULTS_DIR,
            f"scatter_feature_importance_{name}.png"
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print("Saved →", save_path)

# ========== OOD detection (Mahalanobis) ==========
print("\nRunning OOD detection for Scatter (Mahalanobis on scaled features)...")
from sklearn.preprocessing import StandardScaler
scaler_for_ood = StandardScaler()
X_train_scaled = scaler_for_ood.fit_transform(X_train)
X_test_scaled = scaler_for_ood.transform(X_test)

mu, inv_cov, threshold = fit_mahalanobis(X_train_scaled)
flags, distances = detect_ood(X_test_scaled, mu, inv_cov, threshold)
print("Scatter OOD flagged:", flags.sum(), "/", len(flags))

# Save OOD report
ood_df = pd.DataFrame({
    "distance": distances,
    "is_ood": flags
})
ood_df.to_csv(os.path.join(RESULTS_DIR, "scatter_ood_report.csv"), index=False)
print("Saved Scatter OOD report to", os.path.join(RESULTS_DIR, "scatter_ood_report.csv"))
# ==========================================================
#                PREDICTIVE UNCERTAINTY BLOCK
#        (Paste at the END of train_cbc_model.py AND
#                  train_scatter_model.py)
# ==========================================================

from uncertainty_utils import (
    load_models,
    ensemble_proba,
    predictive_entropy_from_probas,
    choose_thresholds,
    decide_accept
)
import os
import numpy as np

print("\n======================================")
print(" Running Predictive Uncertainty")
print("======================================\n")

# ----------------------------------------------------------
# 1. LOAD YOUR MODEL FILES
# ----------------------------------------------------------
# Make sure these names match the actual saved files in /results/
model_files = [
    os.path.join("results", f"scatter_model_{name}.pkl")
    for name in [
        "LogisticRegression",
        "RandomForest",
        "SVM_RBF",
        "AdaBoost",
        "ExtraTrees"
    ]
]


models = load_models(model_files)

# ----------------------------------------------------------
# 2. CHOOSE THE INPUT MATRIX
# ----------------------------------------------------------
# If you used a scaler, X_train_scaled/X_test_scaled should be used instead.
X_train_unc = X_train
X_test_unc = X_test

# ----------------------------------------------------------
# 3. COMPUTE ENSEMBLE PROBABILITIES
# ----------------------------------------------------------
mean_p, std_p, all_probas = ensemble_proba(models, X_test_unc)

# entropy across models
entropy_p = predictive_entropy_from_probas(all_probas)

# thresholds learned from training distribution
mean_p_train, std_p_train, train_probas = ensemble_proba(models, X_train_unc)
entropy_train = predictive_entropy_from_probas(train_probas)

std_thresh, ent_thresh = choose_thresholds(std_p_train, entropy_train)

print(f"STD Threshold: {std_thresh:.4f}")
print(f"Entropy Threshold: {ent_thresh:.4f}")

# ----------------------------------------------------------
# 4. COMBINE WITH MAHALANOBIS OOD FLAGS
# ----------------------------------------------------------
# 'flags' should already exist from your OOD detector block
maha_flags_unc = flags   # rename if needed

# ----------------------------------------------------------
# 5. MAKE FINAL DECISION
# ----------------------------------------------------------
decisions = decide_accept(
    mean_p, std_p, entropy_p, maha_flags_unc,
    std_thresh, ent_thresh,
    p_confident=0.65
)

# ----------------------------------------------------------
# 6. SAVE UNCERTAINTY REPORT
# ----------------------------------------------------------
unc_df = pd.DataFrame({
    "mean_proba": mean_p,
    "std_proba": std_p,
    "entropy": entropy_p,
    "mahalanobis_ood": maha_flags_unc,
    "final_decision": decisions
})

unc_report_path = os.path.join("results", "scatter_uncertainty_report.csv")


unc_df.to_csv(unc_report_path, index=False)

print(f"Saved uncertainty report to {unc_report_path}")
print("Uncertainty block completed.\n")
