import pandas as pd
import matplotlib
import os

print("Loading CSV...")
# low_memory=False prevents DtypeWarning if mixed types exist across chunks
DATASET_NAME = '6HAR_Processed_Features_100hz_4_second_window.csv'
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
DATASET_PATH = os.environ.get('HAR_DATASET') or os.path.join(
    _REPO_ROOT, 'datasets', DATASET_NAME
)
df = pd.read_csv(DATASET_PATH, low_memory=False)

initial_rows = len(df)

# Create a boolean mask: True for any row where *any* column contains 'running' (case-insensitive)
mask = df.astype(str).apply(lambda col: col.str.contains('running', case=False, na=False)).any(axis=1)

# Keep only the rows where the mask is False (i.e., drop the ones with 'running')
df_cleaned = df[~mask]

dropped_count = initial_rows - len(df_cleaned)

# Save back to CSV
df_cleaned.to_csv('2HAR_No_Running_Master_Data.csv', index=False)

print(f"Done! Dropped {dropped_count} rows. Cleaned file saved to data_cleaned.csv")
