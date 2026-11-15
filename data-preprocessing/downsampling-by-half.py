import pandas as pd
import os
print("Loading filtered dataset...")
DATASET_NAME = '5HAR_Filtered_Dataset.csv'
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
DATASET_PATH = os.environ.get('HAR_DATASET') or os.path.join(
    _REPO_ROOT, 'datasets', DATASET_NAME
)
df = pd.read_csv(DATASET_PATH, low_memory=False)

group_cols = ['Participant', 'Activity_Label', 'Trial_Number']

print("Downsampling trials from 100 Hz to 50 Hz...")
downsampled_chunks = []

# Group by trials to downsample each trial independently without mixing boundaries
for keys, group in df.groupby(group_cols):
    g = group.reset_index(drop=True)
    
    # Take every 2nd row (reduces sample rate by half: 100 Hz -> 50 Hz)
    g_downsampled = g.iloc[::2].copy()
    
    downsampled_chunks.append(g_downsampled)

# Recombine all downsampled trials
df_downsampled = pd.concat(downsampled_chunks, ignore_index=True)

# Save the downsampled dataset
output_path = r'5.1HAR_Downsampled_50Hz.csv'
df_downsampled.to_csv(output_path, index=False)

print(f"Success! Downsampled dataset saved to: {output_path}")
print(f"Original rows: {len(df)} | Downsampled rows: {len(df_downsampled)}")
