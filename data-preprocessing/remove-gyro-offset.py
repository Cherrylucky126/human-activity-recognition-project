import pandas as pd

print("Loading CSV...")
df = pd.read_csv(r'C:\All Python Files\SynologyDrive\Pandas Practice\no_running_with_id.csv', low_memory=False)

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
df.to_csv(r'no_running_fully_calibrated.csv', index=False)
print("Gyroscope bias successfully subtracted and saved!")
