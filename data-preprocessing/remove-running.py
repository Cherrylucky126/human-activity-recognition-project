import pandas as pd
import matplotlib

print("Loading CSV...")
# low_memory=False prevents DtypeWarning if mixed types exist across chunks
df = pd.read_csv('raw_data', low_memory=False)

initial_rows = len(df)

# Create a boolean mask: True for any row where *any* column contains 'running' (case-insensitive)
mask = df.astype(str).apply(lambda col: col.str.contains('running', case=False, na=False)).any(axis=1)

# Keep only the rows where the mask is False (i.e., drop the ones with 'running')
df_cleaned = df[~mask]

dropped_count = initial_rows - len(df_cleaned)

# Save back to CSV
df_cleaned.to_csv('no_running_master_data.csv', index=False)

print(f"Done! Dropped {dropped_count} rows. Cleaned file saved to data_cleaned.csv")
