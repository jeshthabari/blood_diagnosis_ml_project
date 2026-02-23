import pandas as pd
import numpy as np
import os

np.random.seed(42)

DATA_DIR = "data"
cbc_path = os.path.join(DATA_DIR, "cbc_features.csv")
scatter_path = os.path.join(DATA_DIR, "scatter_coordinates.csv")

cbc_out = os.path.join(DATA_DIR, "cbc_extended_hard_50000.csv")
scatter_out = os.path.join(DATA_DIR, "scatter_extended_hard_50000.csv")

cbc = pd.read_csv(cbc_path)
scatter = pd.read_csv(scatter_path)


def generate_hard_noise(df, target_n=50000, label_col="Label"):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    # STEP 1: Expand dataset massively
    reps = int(np.ceil(target_n / len(df)))
    extended = pd.concat([df.sample(frac=1, replace=True) for _ in range(reps)], ignore_index=True)
    extended = extended.sample(target_n).reset_index(drop=True)

    # STEP 2: Strong noise injection
    for col in numeric_cols:
        std = extended[col].std() or 1
        extended[col] += np.random.normal(0, std * 0.4, size=target_n)   # heavy noise
        extended[col] = np.abs(extended[col])                            # no negatives

    # STEP 3: Ambiguous points (20–40%)
    amb_frac = np.random.uniform(0.2, 0.4)
    amb_idx = np.random.choice(target_n, size=int(amb_frac*target_n), replace=False)

    for col in numeric_cols:
        std = extended[col].std() or 1
        extended.loc[amb_idx, col] += np.random.normal(0, std * 1.2, len(amb_idx))  # extremely confusing noise

    # STEP 4: Add useless random columns
    for i in range(5):
        extended[f"Random_Noise_{i}"] = np.random.normal(0, 1, target_n)

    # STEP 5: Add random categorical garbage
    extended["Garbage_Category"] = np.random.choice(["A", "B", "C", "D"], size=target_n)

    # STEP 6: 5% label noise (flip labels incorrectly)
    label_values = extended[label_col].unique()
    noise_idx = np.random.choice(target_n, size=int(0.05 * target_n), replace=False)
    extended.loc[noise_idx, label_col] = np.random.choice(label_values, len(noise_idx))

    return extended


# SPECIAL SCATTER MODIFICATIONS
def distort_scatter(df):
    df = df.copy()

    if "X" in df.columns and "Y" in df.columns:
        X = df["X"].values
        Y = df["Y"].values

        # Random rotation
        theta = np.random.uniform(0, np.pi)
        rot_x = X * np.cos(theta) - Y * np.sin(theta)
        rot_y = X * np.sin(theta) + Y * np.cos(theta)

        df["X_rot"] = rot_x
        df["Y_rot"] = rot_y

        # Fake mirrored points
        df["X_mirror"] = -X + np.random.normal(0, 0.5, len(df))
        df["Y_mirror"] = -Y + np.random.normal(0, 0.5, len(df))

    return df


print("Generating HARD CBC dataset...")
cbc_hard = generate_hard_noise(cbc, target_n=50000, label_col="Label")
cbc_hard.to_csv(cbc_out, index=False)
print("Saved:", cbc_out)

print("Generating HARD SCATTER dataset...")
scatter_hard = generate_hard_noise(scatter, target_n=50000, label_col="Label")
scatter_hard = distort_scatter(scatter_hard)
scatter_hard.to_csv(scatter_out, index=False)
print("Saved:", scatter_out)

print("\nYour hard datasets are ready!")
