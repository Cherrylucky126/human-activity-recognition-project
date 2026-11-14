import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv(r'C:\All Python Files\SynologyDrive\Pandas Practice\HAR_Processed_Features_100hz_4_second_window.csv')

# 2. Separate Features (X) and Labels (y)
metadata_cols = ['Participant', 'Activity_Label', 'Trial_Number']
X = df.drop(columns=[col for col in metadata_cols if col in df.columns])
y = df['Activity_Label']

# 3. Train/Val/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# 4. Scale features (Important if you use scaler stats on the ESP32)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Decision Tree
dt_classifier = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_classifier.fit(X_train_scaled, y_train)

# Evaluate
test_acc = accuracy_score(y_test, dt_classifier.predict(X_test_scaled))
print(f"Decision Tree Test Accuracy: {test_acc:.4f}")

# Save model and scaler parameters
joblib.dump(dt_classifier, "har_decision_tree.pkl")
joblib.dump(scaler, "har_scaler.pkl")
print("Model and scaler saved successfully!")
