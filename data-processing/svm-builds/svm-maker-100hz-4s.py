import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Configuration ---
DATASET_PATH = r'C:\All Python Files\SynologyDrive\Pandas Practice\HAR_Processed_Features_100hz_4_second_window.csv'

print("1. Loading processed feature dataset...")
df = pd.read_csv(DATASET_PATH)

# Separate features (X) and target labels (y)
meta_cols = ['Participant', 'Activity_Label', 'Trial_Number']
feature_cols = [col for col in df.columns if col not in meta_cols]

X = df[feature_cols]
y = df['Activity_Label']

print(f"Dataset loaded! Total feature samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
print(f"Target activities: {y.unique()}")

# 2. Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Fitting StandardScaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("4. Training Support Vector Machine (SVM) with probability calibration...")
# Wrap base SVC with CalibratedClassifierCV to enable predict_proba() warning-free
base_svc = SVC(kernel='rbf', C=1.0, random_state=42)
model = CalibratedClassifierCV(base_svc, method='sigmoid', ensemble=False)
model.fit(X_train_scaled, y_train)

# 5. Evaluate performance on the test set
print("5. Evaluating model performance...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- Test Accuracy: {acc * 100:.2f}% ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save scaler and trained SVM model files
print("6. Saving scaler and SVM model files...")
joblib.dump(scaler, 'har_scaler_svm.pkl')
joblib.dump(model, 'har_svm.pkl')

print("Successfully exported 'har_scaler.pkl' and 'har_svm.pkl' to your working directory!")
