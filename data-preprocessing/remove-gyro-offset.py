import pandas as pd
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

# Replace these values with the exact offsets you calculated from your flat recording file
gyro_offsets = {
    'gx': -0.019267,  # Example offset value for GyroX
    'gy': 0.008196, # Example offset value for GyroY
    'gz': -0.022916   # Example offset value for GyroZ
}

# Subtract the bias from the gyroscope columns
df['gx'] = df['gx'] - gyro_offsets['gx']
df['gy'] = df['gy'] - gyro_offsets['gy']
df['gz'] = df['gz'] - gyro_offsets['gz']

# Save the fully calibrated dataset
df.to_csv('4.2HAR_Fully_Calibrated.csv', index=False)
print("Gyroscope bias successfully subtracted and saved!")
