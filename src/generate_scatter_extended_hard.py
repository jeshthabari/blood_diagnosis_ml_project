# src/generate_scatter_extended_hard.py
# Generates extremely difficult scatter dataset: 50,000 rows with heavy noise,
# rotations, cluster blending, synthetic points, garbage columns & label noise.

import pandas as pd
import numpy as np
import os

np.random.seed(77)

DATA_DIR = "data"
scatter_path = os.path.join(DATA_DIR, "scatter_coordinates.csv")
scatter_out = os.path.join(DATA_DIR, "scatter_extended_hard_50000.csv")

scatter = pd.read_csv(scatter_path)

# -------------------------------------------
# 1) Massive + aggressive scatter augmentation
# -------------------------------------------
def extend_scatter_hard(df, target_n=50000, label_col="Label"):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    # STEP A: Expand base dataset heavily
    reps = int(np.ceil(target_n / len(df)))
    expanded = pd.concat(
        [df.sample(frac=1, replace=True) for _ in range(reps)],
        ignore_index=True
    )
    expanded = expanded.sample(target_n, random_state=77).reset_index(drop=True)

    # STEP B: Inject strong noise into X, Y, Light_Intensity, Cell_Size
    for col in numeric_cols:
        std = expanded[col].std() or 1
        # Mix normal noise + occasional large spikes
        noise = np.random.normal(0, std * np.random.uniform(0.4, 1.0), size=target_n)
        expanded[col] += noise
        expanded[col] = expanded[col].abs()

    # STEP C: Create 20–45% highly ambiguous points
    amb_frac = np.random.uniform(0.20, 0.45)
    amb_n = int(amb_frac * target_n)
    amb_idx = np.random.choice(target_n, amb_n, replace=False)
    for col in numeric_cols:
        std = expanded[col].std() or 1
        expanded.loc[amb_idx, col] += np.random.normal(0, std * 1.5, size=amb_n)

    # STEP D: Heavy geometric distortions to scatter coordinates
    if "X" in df.columns and "Y" in df.columns:
        X = expanded["X"].values
        Y = expanded["Y"].values

        # ---- Rotation by a random angle ----
        theta = np.random.uniform(0, np.pi)
        rot_x = X * np.cos(theta) - Y * np.sin(theta)
        rot_y = X * np.sin(theta) + Y * np.cos(theta)
        expanded["X_rot"] = rot_x
        expanded["Y_rot"] = rot_y

        # ---- Mirrored noise cloud ----
        expanded["X_inv"] = -X + np.random.normal(0, 1.0, target_n)
        expanded["Y_inv"] = -Y + np.random.normal(0, 1.0, target_n)

        # ---- Scaled distortions ----
        scale = np.random.uniform(0.7, 1.3)
        expanded["X_scaled"] = X * scale + np.random.normal(0, 0.3, target_n)
        expanded["Y_scaled"] = Y * scale + np.random.normal(0, 0.3, target_n)

    # STEP E: Add garbage random numeric columns
    for i in range(5):
        expanded[f"Noise_{i}"] = np.random.normal(0, 1, target_n)

    # STEP F: Add garbage categorical columns
    expanded["Category_Garbage"] = np.random.choice(["A", "B", "C", "D"], size=target_n)

    # STEP G: 10% label noise to make classification realistically messy
    label_values = expanded[label_col].unique()
    noise_idx = np.random.choice(target_n, size=int(0.10 * target_n), replace=False)
    expanded.loc[noise_idx, label_col] = np.random.choice(label_values, len(noise_idx))

    return expanded


print("\n🔧 Generating EXTREME scatter dataset (50,000 rows)...")
scatter_hard = extend_scatter_hard(scatter, target_n=50000, label_col="Label")
scatter_hard.to_csv(scatter_out, index=False)
print("✅ Saved:", scatter_out)
print("Shape:", scatter_hard.shape)
print("Label counts:\n", scatter_hard["Label"].value_counts())
print("\n🎉 Done! This dataset will be VERY hard for your models.\n")
