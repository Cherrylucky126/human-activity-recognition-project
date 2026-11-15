import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
print("Loading CSV...")
DATASET_NAME = '6HAR_Processed_Features_100hz_4_second_window.csv'
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
DATASET_PATH = os.environ.get('HAR_DATASET') or os.path.join(
    _REPO_ROOT, 'datasets', DATASET_NAME
)
df = pd.read_csv(DATASET_PATH, low_memory=False)

group_cols = ['Participant', 'Activity_Label', 'Trial_Number']
imu_sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

# --- CONFIGURATION PARAMETERS ---
fs = 100.0          # Sampling frequency (100 Hz)
imu_cutoff = 5.0    # IMU low-pass cutoff frequency (Hz)
ir_lowcut = 0.5     # IR bandpass lower cutoff (Hz)
ir_highcut = 5.0    # IR bandpass upper cutoff (Hz)
order = 4           # Butterworth filter order

# --- FILTER FUNCTIONS ---
def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)

# --- PROCESSING TRIAL-BY-TRIAL VIA LOOP ---
print("Filtering IMU and IR sensor data trial-by-trial...")
processed_chunks = []

for keys, group in df.groupby(group_cols):
    g = group.copy()
    
    # 1. IMU Low-Pass Filter
    for col in imu_sensor_cols:
        if col in g.columns:
            g[col] = butter_lowpass(g[col].values, imu_cutoff, fs, order)

    # 2. IR Bandpass Filter & Z-Score Normalization
    if 'IR_Value' in g.columns:
        filtered_ir = butter_bandpass(g['IR_Value'].values, ir_lowcut, ir_highcut, fs, order)
        mean_val = np.mean(filtered_ir)
        std_val = np.std(filtered_ir)
        g['IR_Value'] = (filtered_ir - mean_val) / (std_val + 1e-8)
        
    processed_chunks.append(g)

# Combine all processed trials back together
df_filtered = pd.concat(processed_chunks, ignore_index=True)

# Save the filtered dataset
output_path = '5HAR_Filtered_Dataset.csv'
df_filtered.to_csv(output_path, index=False)
print(f"Success! Filtered dataset saved with all columns to: {output_path}")
