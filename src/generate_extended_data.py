import pandas as pd
import numpy as np

# Load your original dataset
df = pd.read_csv("HPLC data.csv")

# How many total rows you want
target_size = 50000
multiplier = int(np.ceil(target_size / len(df)))

# Identify numeric columns
num_cols = ['HbA0', 'HbA2', 'HbF', 'S-Window', 'Unknown',
            'RBC', 'HB', 'MCV', 'MCH', 'MCHC', 'RDWcv']

# Generate noisy copies
frames = [df]
for i in range(multiplier - 1):
    noisy = df.copy()
    for col in num_cols:
        noise = np.random.normal(0, df[col].std() * 0.1, size=len(df))
        noisy[col] = np.clip(df[col] + noise, a_min=0, a_max=None)
    frames.append(noisy)

# Combine and shuffle
df_extended = pd.concat(frames, ignore_index=True).sample(n=target_size, random_state=50).reset_index(drop=True)

# Add small random jitter to age
if 'Age' in df_extended.columns:
    df_extended['Age'] = df_extended['Age'].astype(str)
    df_extended['Age'] = df_extended['Age'].apply(lambda x: str(int(np.clip(np.random.normal(35, 15), 1, 80))) + " yrs")

# Save
df_extended.to_csv("HPLC_data_extended.csv", index=False)
print("✅ Extended dataset saved as 'HPLC_data_extended.csv' with", len(df_extended), "rows")
