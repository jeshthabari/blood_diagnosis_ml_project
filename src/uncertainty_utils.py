# src/uncertainty_utils.py
import numpy as np
import joblib
from scipy.stats import entropy

def load_models(model_paths):
    """Load models from list of paths (joblib files). Returns list of models."""
    return [joblib.load(p) for p in model_paths]

def ensemble_proba(models, X):
    """
    Given a list of models (sklearn-like with predict_proba), return:
      mean_proba: shape (n_samples,) probability of positive class
      std_proba:  shape (n_samples,) stddev across ensemble for positive class
      all_probas: shape (n_models, n_samples)
    """
    probas = []
    for m in models:
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[:, 1]
        else:
            # fallback: try decision_function then logisticize
            try:
                df = m.decision_function(X)
                p = 1 / (1 + np.exp(-df))
            except Exception:
                raise ValueError("Model has neither predict_proba nor decision_function")
        probas.append(p)
    probas = np.vstack(probas)                # shape: (n_models, n_samples)
    mean_proba = probas.mean(axis=0)
    std_proba = probas.std(axis=0)
    return mean_proba, std_proba, probas

def predictive_entropy_from_probas(probas):
    """
    probas: (n_models, n_samples) or (n_samples,) for mean probs
    If given mean probs (1D), compute binary entropy.
    If given all model probas (2D), compute mean entropy across models.
    """
    if probas.ndim == 1:
        p = np.clip(probas, 1e-10, 1-1e-10)
        return -(p * np.log2(p) + (1-p) * np.log2(1-p))
    else:
        # compute each model's entropy and average
        ent = []
        for row in probas:
            pr = np.clip(row, 1e-10, 1-1e-10)
            ent.append(-(pr * np.log2(pr) + (1-pr) * np.log2(1-pr)))
        ent = np.vstack(ent)   # (n_models, n_samples)
        return ent.mean(axis=0)

def choose_thresholds(std_vals_train, entropy_train, std_pct=95, ent_pct=95):
    """
    Choose thresholds based on training (or validation) distributions.
    Default: 95th percentile.
    Returns (std_thresh, entropy_thresh)
    """
    std_thresh = np.percentile(std_vals_train, std_pct)
    ent_thresh = np.percentile(entropy_train, ent_pct)
    return std_thresh, ent_thresh

def decide_accept(mean_p, std_p, entropy_p, maha_flag, 
                  std_thresh, ent_thresh, p_confident=0.6):
    """
    Decision rule (vectorized):
      - If mahalanobis OOD flag True => "REJECT_OOD"
      - Else if std_p > std_thresh or entropy_p > ent_thresh => "REJECT_UNCERTAIN"
      - Else if mean_p between (1-p_confident, p_confident) => "REJECT_LOW_CONF"
      - Else return class (0/1) predicted by threshold 0.5
    Returns array of decisions (strings or ints).
    """
    decisions = np.empty(len(mean_p), dtype=object)
    for i in range(len(mean_p)):
        if maha_flag[i]:
            decisions[i] = "REJECT_OOD"
            continue
        if std_p[i] > std_thresh or entropy_p[i] > ent_thresh:
            decisions[i] = "REJECT_UNCERTAIN"
            continue
        if (mean_p[i] < p_confident) and (mean_p[i] > (1 - p_confident)):
            decisions[i] = "REJECT_LOW_CONF"
            continue
        decisions[i] = int(mean_p[i] >= 0.5)
    return decisions
