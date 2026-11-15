import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Configuration ---
# The processed feature dataset. It is NOT in the repository: it is the
# participants' recordings, and it is large. Get it from the project Drive
# folder linked in the top-level README.md and save it as
#
#     data/HAR_Processed_Features_100hz_4_second_window.csv
#
# or set the HAR_DATASET environment variable to wherever you keep it.
#
# This used to be a single absolute path on one machine, which meant nobody
# else could re-run the training at all.
import os

DATASET_NAME = 'HAR_Processed_Features_100hz_4_second_window.csv'
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
DATASET_PATH = os.environ.get('HAR_DATASET') or os.path.join(
    _REPO_ROOT, 'data', DATASET_NAME
)

if not os.path.exists(DATASET_PATH):
    raise SystemExit(f"""
Dataset not found:
    {DATASET_PATH}

Download {DATASET_NAME} from the Drive folder
linked in README.md and save it to data/, or set HAR_DATASET to its location.
""")

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

print("4. Training Logistic Regression Classifier...")
# max_iter is increased to ensure convergence on complex feature sets
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate performance on the test set
print("5. Evaluating model performance...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n--- Test Accuracy: {acc * 100:.2f}% ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# --- 5b. Leave-One-Subject-Out (LOSO) Evaluation Snippet ---
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
loso_y_true, loso_y_pred, fold_accs = [], [], {}

for train_idx, test_idx in logo.split(X, y, df['Participant']):
  subj = df['Participant'].iloc[test_idx].iloc[0]

  fold_scaler = StandardScaler()
  X_tr = fold_scaler.fit_transform(X.iloc[train_idx])
  X_te = fold_scaler.transform(X.iloc[test_idx])

  fold_model = clone(model)  # automatically copies your model + hyperparameters
  fold_model.fit(X_tr, y.iloc[train_idx])
  preds = fold_model.predict(X_te)

  fold_accs[subj] = accuracy_score(y.iloc[test_idx], preds)
  loso_y_true.extend(y.iloc[test_idx])
  loso_y_pred.extend(preds)

print(
    f'\n--- LOSO Mean Accuracy: {np.mean(list(fold_accs.values())) * 100:.2f}%'
    f' (± {np.std(list(fold_accs.values())) * 100:.2f}%) ---'
)
for subj, f_acc in fold_accs.items():
  print(f'  Held-out {subj}: {f_acc * 100:.2f}%')

print('\nLOSO Classification Report:')
print(classification_report(loso_y_true, loso_y_pred))
# 6. Save scaler and trained Logistic Regression model files
print("6. Saving scaler and Logistic Regression model files...")
#joblib.dump(scaler, 'har_scaler.lr.pkl')
#joblib.dump(model, 'har_logistic_regression.pkl')

print("Successfully exported 'har_scaler.pkl' and 'har_logistic_regression.pkl' to your working directory!")
