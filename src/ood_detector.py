# src/ood_detector.py
import numpy as np
import pandas as pd
from scipy.linalg import eigh

def fit_mahalanobis(X_train, regularize=1e-6):
    """
    Fit Mahalanobis detector on numeric training data (numpy array or DataFrame).
    Returns: mu (mean vector), inv_cov (inverse covariance), threshold (99th percentile on train)
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    # Ensure finite
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.mean(X_train, axis=0)
    Xc = X_train - mu
    cov = np.cov(Xc, rowvar=False)
    cov_reg = cov + np.eye(cov.shape[0]) * regularize
    eigvals, eigvecs = eigh(cov_reg)
    eigvals_clipped = np.clip(eigvals, regularize, None)
    inv_cov = (eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T)
    d2_train = mahalanobis_sq(X_train, mu, inv_cov)
    d_train = np.sqrt(d2_train)
    threshold = np.percentile(d_train, 99)  # conservative default
    return mu, inv_cov, threshold

def mahalanobis_sq(X, mu, inv_cov):
    """
    Vectorized squared Mahalanobis distance.
    X: (n_samples, n_features)
    mu: (n_features,)
    inv_cov: (n_features, n_features)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    Xc = X - mu
    left = Xc.dot(inv_cov)
    m2 = np.einsum('ij,ij->i', left, Xc)
    return m2

def detect_ood(X, mu, inv_cov, threshold):
    """
    Returns (flags, distances) where flags is boolean array (True = OOD)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    d2 = mahalanobis_sq(X, mu, inv_cov)
    d = np.sqrt(d2)
    flags = d > threshold
    return flags, d
