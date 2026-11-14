import serial
import numpy as np
import pandas as pd
import joblib
import time
from scipy.signal import find_peaks
import warnings

# Suppress sklearn and numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
COM_PORT = 'COM8'  # Your incoming Bluetooth port
BAUD_RATE = 115200
SAMPLING_FREQ = 100
WINDOW_SIZE = 400  # 4 seconds at 100Hz

print("Loading model and scaler...")
scaler = joblib.load("har_scaler_dt.pkl")
model = joblib.load("har_decision_tree.pkl")
print("Decision Tree model loaded successfully!")

def extract_features_from_window(window_imu, window_ir, fs):
    """Exact feature extraction matching your master dataset pipeline."""
    features = []
    axis_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    # 1. IMU Per-Axis Features
    for i, axis in enumerate(axis_names):
        signal = window_imu[:, i]
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms_val = np.sqrt(np.mean(signal**2))
        
        # FFT features with zero-centering
        signal_centered = signal - mean_val
        fft_vals = np.fft.rfft(signal_centered)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
        fft_mag = np.abs(fft_vals)
        
        dom_freq = fft_freqs[np.argmax(fft_mag)] if len(fft_mag) > 0 else 0.0
        energy = np.sum(fft_mag**2)

        features.extend([mean_val, std_val, rms_val, dom_freq, energy])

    # 2. Magnitude & Correlation Features
    accel_data = window_imu[:, :3]
    gyro_data = window_imu[:, 3:]
    
    accel_mag = np.linalg.norm(accel_data, axis=1)
    gyro_mag = np.linalg.norm(gyro_data, axis=1)
    
    accel_mag_mean = np.mean(accel_mag)
    accel_mag_std = np.std(accel_mag)
    accel_mag_max = np.max(accel_mag)
    accel_spectral_energy = np.sum(np.abs(np.fft.rfft(accel_mag - accel_mag_mean))**2)
    
    accel_mag_centered = accel_mag - accel_mag_mean
    accel_fft = np.fft.rfft(accel_mag_centered)
    accel_freqs = np.fft.rfftfreq(len(accel_mag), d=1.0/fs)
    accel_dominant_freq = accel_freqs[np.argmax(np.abs(accel_fft))] if len(accel_fft) > 0 else 0.0
    
    features.extend([accel_mag_mean, accel_mag_std, accel_mag_max, accel_spectral_energy, accel_dominant_freq])
    
    gyro_mag_mean = np.mean(gyro_mag)
    gyro_mag_std = np.std(gyro_mag)
    features.extend([gyro_mag_mean, gyro_mag_std])
    
    axis_corr_xy = np.corrcoef(window_imu[:, 0], window_imu[:, 1])[0, 1] if len(window_imu) > 1 else 0.0
    if np.isnan(axis_corr_xy): axis_corr_xy = 0.0
    features.append(axis_corr_xy)
    
    sma = np.sum(np.abs(accel_data)) / len(accel_data)
    features.append(sma)

    # 3. Precise PPG / IR Cardiac Features
    if window_ir is not None and len(window_ir) > 0:
        ir_ac_variance = np.var(window_ir)
        ir_p2p_amplitude = np.ptp(window_ir)
        
        fft_ir = np.fft.rfft(window_ir)
        freqs_ir = np.fft.rfftfreq(len(window_ir), d=1.0/fs)
        mag_ir = np.abs(fft_ir)
        
        valid_idx = np.where((freqs_ir >= 0.75) & (freqs_ir <= 3.0))
        if len(valid_idx[0]) > 0:
            dom_ir_freq = freqs_ir[valid_idx][np.argmax(mag_ir[valid_idx])]
            rolling_bpm = dom_ir_freq * 60.0
        else:
            rolling_bpm = 0.0
            
        peaks, _ = find_peaks(window_ir, distance=int(fs * 0.4), prominence=0.2)
        if len(peaks) > 1:
            ibi_intervals = np.diff(peaks) / fs
            ibi_std = np.std(ibi_intervals)
        else:
            ibi_std = 0.0
    else:
        ir_ac_variance = 0.0
        ir_p2p_amplitude = 0.0
        rolling_bpm = 0.0
        ibi_std = 0.0

    features.extend([ir_ac_variance, ir_p2p_amplitude, rolling_bpm, ibi_std])
    return features

def main():
    print(f"Connecting to {COM_PORT}...")
    try:
        bt_serial = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1, dsrdtr=False, rtscts=False)
        bt_serial.reset_input_buffer()
        print("Connected! Listening for sensor data...")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    buffer = []
    print("Entering read loop. Monitoring incoming stream...")

    while True:
        try:
            line_bytes = bt_serial.readline()
            if not line_bytes:
                continue
            
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line: 
                continue
            
            # Parse CSV values (Expects 7 values: ax, ay, az, gx, gy, gz, ir)
            vals = [float(x) for x in line.split(',')]
            if len(vals) == 7:
                buffer.append(vals)
                if len(buffer) % 100 == 0:
                    print(f"Buffer progress: {len(buffer)}/{WINDOW_SIZE} samples...")

            # Once we hit or exceed 400 samples
            if len(buffer) >= WINDOW_SIZE:
                window_data = np.array(buffer[:WINDOW_SIZE])
                # Slide window forward by 50% (200 samples)
                buffer = buffer[int(WINDOW_SIZE * 0.5):]  
                
                window_imu = window_data[:, :6]
                window_ir = window_data[:, 6]
                
                window_ir = (window_ir - np.mean(window_ir)) / (np.std(window_ir) + 1e-8)
                
                raw_features = extract_features_from_window(window_imu, window_ir, SAMPLING_FREQ)
                features_array = np.array(raw_features).reshape(1, -1)
                
                scaled_features = scaler.transform(features_array)
                scaled_features = np.nan_to_num(scaled_features, nan=0.0)
                
                pred_val = model.predict(scaled_features)[0]
                pred_label = str(pred_val).upper()
                
                probs = model.predict_proba(scaled_features)[0]
                confidence = np.max(probs)
                
                print(f"[{time.strftime('%H:%M:%S')}] >>> PREDICTED: {pred_label} (Confidence: {confidence:.2f}) <<<")

        except KeyboardInterrupt:
            print("\nStopping...")
            bt_serial.close()
            break
        except Exception as e:
            print(f"CRITICAL ERROR IN LOOP: {e}")

if __name__ == "__main__":
    main()
