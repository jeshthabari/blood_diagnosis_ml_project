# src/generate_extended_datasets.py
# Generates two extended datasets (50,000 rows each) from existing cbc_features.csv and scatter_coordinates.csv
# Run from project root:
#   python src/generate_extended_datasets.py

import pandas as pd
import numpy as np
import os
np.random.seed(50)

# Paths (relative to project root)
DATA_DIR = os.path.join("data")
cbc_path = os.path.join(DATA_DIR, "cbc_features.csv")
scatter_path = os.path.join(DATA_DIR, "scatter_coordinates.csv")

# Output paths
cbc_out = os.path.join(DATA_DIR, "cbc_extended_50000.csv")
scatter_out = os.path.join(DATA_DIR, "scatter_extended_50000.csv")

# Safety checks
if not os.path.exists(cbc_path):
    raise FileNotFoundError(f"File not found: {cbc_path}")
if not os.path.exists(scatter_path):
    raise FileNotFoundError(f"File not found: {scatter_path}")

cbc = pd.read_csv(cbc_path)
scatter = pd.read_csv(scatter_path)

def extend_dataset(df, target_n=50000, label_col='Label', noise_scale=0.12, outlier_frac=0.01):
    # numeric columns (robust)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    # fallback: try to infer numeric-like cols
    if len(numeric_cols) == 0:
        for c in df.columns:
            try:
                pd.to_numeric(df[c])
                numeric_cols.append(c)
            except Exception:
                pass
        numeric_cols = [c for c in numeric_cols if c != label_col]

    n = len(df)
    repeats = int(np.ceil(target_n / n))
    parts = []
    for r in range(repeats):
        sampled = df.sample(n=n, replace=True, random_state=50 + r).reset_index(drop=True)
        for col in numeric_cols:
            col_std = sampled[col].std(skipna=True)
            if np.isnan(col_std) or col_std == 0:
                noise = np.random.normal(0, 1e-6, size=len(sampled))
            else:
                # vary scale slightly per repeat to create more realistic spread
                scale = noise_scale * (1 + (r % 4) * 0.02)
                noise = np.random.normal(0, col_std * scale, size=len(sampled))
            sampled[col] = sampled[col].astype(float) + noise
            # clip negative where not sensible
            sampled[col] = sampled[col].where(sampled[col] >= 0, sampled[col].abs())
        parts.append(sampled)

    extended = pd.concat(parts, ignore_index=True).sample(n=target_n, random_state=50).reset_index(drop=True)

    # add outliers (large shifts) in a small fraction
    out_n = max(1, int(outlier_frac * target_n))
    out_idx = np.random.choice(range(target_n), size=out_n, replace=False)
    for col in numeric_cols:
        col_std = extended[col].std() if not np.isnan(extended[col].std()) else 1.0
        large_noise = np.random.normal(0, col_std * 5, size=out_n)
        extended.loc[out_idx, col] = extended.loc[out_idx, col] + large_noise
        extended[col] = extended[col].where(extended[col] >= 0, extended[col].abs())

    # scatter-specific ambiguity: nudge some X,Y toward other-class centroids (if present)
    if {'X','Y', label_col}.issubset(set(extended.columns)) and {'X','Y'}.issubset(set(df.columns)):
        try:
            centroids = df.groupby(label_col)[['X','Y']].mean()
            unique_labels = centroids.index.tolist()
            amb_frac = 0.02
            amb_n = int(amb_frac * target_n)
            amb_idx = np.random.choice(range(target_n), size=amb_n, replace=False)
            for idx in amb_idx:
                cur_lab = extended.at[idx, label_col]
                other_labels = [l for l in unique_labels if l != cur_lab]
                if other_labels:
                    target_lab = np.random.choice(other_labels)
                    frac = np.random.uniform(0.25, 0.6)
                    extended.at[idx,'X'] = extended.at[idx,'X'] * (1-frac) + centroids.loc[target_lab,'X'] * frac
                    extended.at[idx,'Y'] = extended.at[idx,'Y'] * (1-frac) + centroids.loc[target_lab,'Y'] * frac
        except Exception:
            pass

    # round integer-like columns same as original
    for col in numeric_cols:
        try:
            if all((df[col].dropna() - df[col].dropna().astype(int)).abs() < 1e-8):
                extended[col] = extended[col].round().astype(int)
        except Exception:
            pass

    return extended

print("Extending CBC dataset...")
cbc_ext = extend_dataset(cbc, target_n=50000, label_col='Label', noise_scale=0.12, outlier_frac=0.01)
cbc_ext.to_csv(cbc_out, index=False)
print("Saved:", cbc_out, cbc_ext.shape)
print("Label counts (CBC):")
print(cbc_ext['Label'].value_counts())

print("\nExtending Scatter dataset...")
scatter_ext = extend_dataset(scatter, target_n=50000, label_col='Label', noise_scale=0.12, outlier_frac=0.015)
scatter_ext.to_csv(scatter_out, index=False)
print("Saved:", scatter_out, scatter_ext.shape)
print("Label counts (Scatter):")
print(scatter_ext['Label'].value_counts())

print("\nFinished. Files created in 'data/' folder.")
