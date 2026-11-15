import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
# --- Configuration ---
DATASET_NAME = '5HAR_Filtered_Dataset.csv'
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
DATASET_PATH = os.environ.get('HAR_DATASET') or os.path.join(
    _REPO_ROOT, 'datasets', DATASET_NAME
)

SAMPLING_FREQ = 100  # Match your dataset's sampling frequency (e.g., 50 or 25)
WINDOW_SIZE_SEC = 4  # 4-second window size for stable IR frequency bins
WINDOW_SIZE = int(SAMPLING_FREQ * WINDOW_SIZE_SEC)  
OVERLAP = 0.5  
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))

def extract_features_from_window(window_imu, window_ir, fs):
    """Extracts IMU per-axis, magnitude, spectral features, and precise IR cardiac metrics."""
    features = {}
    axis_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
# 1. IMU Per-Axis Features (Fixed with zero-centering for dominant frequency)
    for i, axis in enumerate(axis_names):
        signal = window_imu[:, i]
        features[f'{axis}_mean'] = np.mean(signal)
        features[f'{axis}_std'] = np.std(signal)
        features[f'{axis}_rms'] = np.sqrt(np.mean(signal**2))
        
        # FFT features (Demean signal so 0 Hz gravity offset doesn't steal the argmax peak)
        signal_centered = signal - np.mean(signal)
        fft_vals = np.fft.rfft(signal_centered)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
        fft_mag = np.abs(fft_vals)
        
        features[f'{axis}_dominant_freq'] = fft_freqs[np.argmax(fft_mag)] if len(fft_mag) > 0 else 0
        features[f'{axis}_energy'] = np.sum(fft_mag**2)

    # 2. Magnitude & Correlation Features (with gravity offset removed for dominant frequency)
    accel_data = window_imu[:, :3]
    gyro_data = window_imu[:, 3:]
    
    accel_mag = np.linalg.norm(accel_data, axis=1)
    gyro_mag = np.linalg.norm(gyro_data, axis=1)
    
    features['accel_mag_mean'] = np.mean(accel_mag)
    features['accel_mag_std'] = np.std(accel_mag)
    features['accel_mag_max'] = np.max(accel_mag)
    features['accel_spectral_energy'] = np.sum(np.abs(np.fft.rfft(accel_mag - np.mean(accel_mag)))**2)
    
    # Demean accel_mag so the 0 Hz gravity DC component doesn't steal the argmax peak
    accel_mag_centered = accel_mag - np.mean(accel_mag)
    accel_fft = np.fft.rfft(accel_mag_centered)
    accel_freqs = np.fft.rfftfreq(len(accel_mag), d=1.0/fs)
    features['accel_dominant_freq'] = accel_freqs[np.argmax(np.abs(accel_fft))] if len(accel_fft) > 0 else 0
    
    features['gyro_mag_mean'] = np.mean(gyro_mag)
    features['gyro_mag_std'] = np.std(gyro_mag)
    features['axis_corr_xy'] = np.corrcoef(window_imu[:, 0], window_imu[:, 1])[0, 1] if len(window_imu) > 1 else 0
    features['SMA'] = np.sum(np.abs(accel_data)) / len(accel_data)

    # 3. Precise PPG / IR Cardiac Features
    if window_ir is not None and len(window_ir) > 0:
        features['ir_ac_variance'] = np.var(window_ir)
        features['ir_p2p_amplitude'] = np.ptp(window_ir) # Peak-to-peak amplitude range
        
        # FFT-based dominant frequency (Robust BPM calculation)
        fft_ir = np.fft.rfft(window_ir)
        freqs_ir = np.fft.rfftfreq(len(window_ir), d=1.0/fs)
        mag_ir = np.abs(fft_ir)
        
        valid_idx = np.where((freqs_ir >= 0.75) & (freqs_ir <= 3.0)) # 45 to 180 BPM cardiac band
        if len(valid_idx[0]) > 0:
            dom_ir_freq = freqs_ir[valid_idx][np.argmax(mag_ir[valid_idx])]
            features['rolling_bpm'] = dom_ir_freq * 60.0
        else:
            features['rolling_bpm'] = 0.0
            
        # Inter-Beat Interval (IBI) standard deviation via constrained peak detection
        peaks, _ = find_peaks(window_ir, distance=int(fs * 0.4), prominence=0.2)
        if len(peaks) > 1:
            ibi_intervals = np.diff(peaks) / fs # time differences between beats in seconds
            features['ibi_std'] = np.std(ibi_intervals)
        else:
            features['ibi_std'] = 0.0
    else:
        features['ir_ac_variance'] = 0.0
        features['ir_p2p_amplitude'] = 0.0
        features['rolling_bpm'] = 0.0
        features['ibi_std'] = 0.0

    return features

def main():
    print("1. Loading already-filtered dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    has_ir = 'IR_Value' in df.columns
    
    # Optional Z-score normalization on IR to ensure consistent peak amplitude scaling
    if has_ir:
        ir_vals = df['IR_Value'].values
        df['IR_Value'] = (ir_vals - np.mean(ir_vals)) / (np.std(ir_vals) + 1e-8)

    print("2. Segmenting data within strict trial boundaries and extracting features...")
    feature_list = []
    group_cols = ['Participant', 'Activity_Label', 'Trial_Number']
    
    for keys, group in df.groupby(group_cols):
        participant, activity, trial = keys
        group_data = group.reset_index(drop=True)
        
        imu_chunk = group_data[sensor_cols].values
        ir_chunk = group_data['IR_Value'].values if has_ir else None
        
        # Slide window strictly within this specific trial to avoid boundary contamination
        for start in range(0, len(group_data) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            
            window_imu = imu_chunk[start:end]
            window_ir = ir_chunk[start:end] if ir_chunk is not None else None
            
            feats = extract_features_from_window(window_imu, window_ir, SAMPLING_FREQ)
            feats['Participant'] = participant
            feats['Activity_Label'] = activity
            feats['Trial_Number'] = trial
            
            feature_list.append(feats)
        
    feature_df = pd.DataFrame(feature_list)

    print("3. Applying rolling median smoothing to rolling_bpm per trial...")
    # Smooth rolling_bpm trial-by-trial to eliminate isolated motion noise spikes
    smoothed_chunks = []
    for keys, group in feature_df.groupby(group_cols):
        g = group.copy()
        if 'rolling_bpm' in g.columns:
            g['rolling_bpm'] = g['rolling_bpm'].rolling(window=3, center=True, min_periods=1).median()
        smoothed_chunks.append(g)
    
    feature_df_smoothed = pd.concat(smoothed_chunks, ignore_index=True)

    print("4. Saving processed and smoothed feature dataset...")
    output_filename = '6HAR_Processed_Features_100hz_4_second_window.csv'
    feature_df_smoothed.to_csv(output_filename, index=False)
    
    print(f"Successfully generated '{output_filename}' with {len(feature_df_smoothed)} feature rows and {len(feature_df_smoothed.columns)} columns!")

if __name__ == '__main__':
    main()
