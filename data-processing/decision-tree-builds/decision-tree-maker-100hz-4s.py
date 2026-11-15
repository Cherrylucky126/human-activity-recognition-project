import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Update this path to match where your feature extraction script saved the CSV
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

DATASET_NAME = r"C:\All Python Files\SynologyDrive\Pandas Practice\HAR_Processed_Features_100hz_4_second_window.csv"
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
# Drop non-feature metadata columns safely
meta_cols = ['Participant', 'Activity_Label', 'Trial_Number']
feature_cols = [col for col in df.columns if col not in meta_cols]

X = df[feature_cols]
y = df['Activity_Label']

print(f"Dataset loaded! Total feature samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
print(f"Target activities: {y.unique()}")

# 2. Train/Test Split (Stratified to ensure equal class representation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Fitting StandardScaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("4. Training Decision Tree Classifier...")
# You can adjust max_depth (e.g., 10 to 20) if you notice overfitting
model = DecisionTreeClassifier(random_state=42, max_depth=15)
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
# --- REPLACE YOUR LOGGING BLOCK WITH THIS ---
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# 1. Capture timestamp and original model name
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = type(model).__name__

# 2. Convert classification reports to DataFrames and add identifier columns
test_rep_dict = classification_report(y_test, y_pred, output_dict=True)
loso_rep_dict = classification_report(
    loso_y_true, loso_y_pred, output_dict=True
)

df_test_report = pd.DataFrame(test_rep_dict).transpose().reset_index()
df_test_report = df_test_report.rename(columns={"index": "Class_Or_Average"})
df_test_report.insert(0, "Timestamp", current_time)
df_test_report.insert(1, "Model", model_name)

df_loso_report = pd.DataFrame(loso_rep_dict).transpose().reset_index()
df_loso_report = df_loso_report.rename(columns={"index": "Class_Or_Average"})
df_loso_report.insert(0, "Timestamp", current_time)
df_loso_report.insert(1, "Model", model_name)

# 3. Build current run summary row
summary_data = {
    "Timestamp": [current_time],
    "Model": [model_name],
    "Test Accuracy (%)": [acc * 100],
    "LOSO Mean Accuracy (%)": [np.mean(list(fold_accs.values())) * 100],
    "LOSO Std (%)": [np.std(list(fold_accs.values())) * 100],
}

for subj, f_acc in fold_accs.items():
  summary_data[f"Held-out {subj} (%)"] = [f_acc * 100]

df_new_summary = pd.DataFrame(summary_data)
excel_file = "model_evaluation_master.xlsx"


# 4. Helper function to safely append new data to historical sheets
def append_to_sheet(file_path, sheet_name, df_new):
  if os.path.exists(file_path):
    try:
      df_existing = pd.read_excel(file_path, sheet_name=sheet_name)
      return pd.concat([df_existing, df_new], ignore_index=True)
    except Exception:
      return df_new
  return df_new


df_all_summaries = append_to_sheet(
    excel_file, "All_Runs_Summary", df_new_summary
)
df_all_test_reports = append_to_sheet(
    excel_file, "All_Test_Reports", df_test_report
)
df_all_loso_reports = append_to_sheet(
    excel_file, "All_LOSO_Reports", df_loso_report
)

# 5. Store raw confusion matrices cleanly so each run can be arranged side-by-side
class_labels = sorted(list(set(y_test) | set(loso_y_true)))


def make_cm_records(y_true, y_pred_vals, eval_type):
  cm = confusion_matrix(y_true, y_pred_vals, labels=class_labels)
  df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
  df_cm.index.name = "True_Class"
  df_cm = df_cm.reset_index()
  df_cm.insert(0, "Timestamp", current_time)
  df_cm.insert(1, "Model", model_name)
  df_cm.insert(2, "Eval_Type", eval_type)
  return df_cm


df_cm_new = pd.concat(
    [
        make_cm_records(y_test, y_pred, "Split"),
        make_cm_records(loso_y_true, loso_y_pred, "LOSO"),
    ],
    ignore_index=True,
)

df_all_cm_raw = append_to_sheet(excel_file, "All_CM_Raw", df_cm_new)


# 6. Build the side-by-side grid layout matching your spreadsheet spacing
def build_side_by_side_cm_grid(df_raw):
  num_classes = len(class_labels)
  block_width = num_classes + 1  # 1 label col + N predicted class cols
  spacer_cols = 1  # 1 empty column between models (matches your screenshot)

  # Preserve exact order models were run
  runs = list(df_raw[["Timestamp", "Model"]].drop_duplicates().itertuples(index=False, name=None))
  total_cols = 1 + len(runs) * (block_width + spacer_cols)

  # Rows: Header + Split block + 1 blank row + LOSO header + LOSO block
  total_rows = (1 + 1 + num_classes) + 1 + (1 + 1 + num_classes)
  grid = [["" for _ in range(total_cols)] for _ in range(total_rows)]

  split_start_row = 0
  loso_start_row = num_classes + 3  # 1 blank row separating Split and LOSO

  grid[split_start_row][0] = "Split"
  grid[loso_start_row][0] = "LOSO"

  for idx, (ts, m_name) in enumerate(runs):
    col_offset = 1 + idx * (block_width + spacer_cols)

    # Model name header
    grid[split_start_row][col_offset] = m_name
    grid[loso_start_row][col_offset] = m_name

    # Column headers (Actual \ Pred + class names)
    grid[split_start_row + 1][col_offset] = "Actual \\ Pred"
    grid[loso_start_row + 1][col_offset] = "Actual \\ Pred"
    for c_i, lbl in enumerate(class_labels):
      grid[split_start_row + 1][col_offset + 1 + c_i] = lbl
      grid[loso_start_row + 1][col_offset + 1 + c_i] = lbl

    # Fill Split & LOSO rows
    for eval_type, base_row in [("Split", split_start_row + 2), ("LOSO", loso_start_row + 2)]:
      sub = df_raw[
          (df_raw["Timestamp"] == ts)
          & (df_raw["Model"] == m_name)
          & (df_raw["Eval_Type"] == eval_type)
      ].set_index("True_Class")

      for r_i, lbl in enumerate(class_labels):
        grid[base_row + r_i][col_offset] = lbl
        for c_i, pred_lbl in enumerate(class_labels):
          grid[base_row + r_i][col_offset + 1 + c_i] = int(sub.loc[lbl, pred_lbl])

  return pd.DataFrame(grid)


df_cm_layout = build_side_by_side_cm_grid(df_all_cm_raw)

# 7. Save all accumulated history back into the master Excel file
with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
  df_all_summaries.to_excel(writer, sheet_name="All_Runs_Summary", index=False)
  df_all_test_reports.to_excel(writer, sheet_name="All_Test_Reports", index=False)
  df_all_loso_reports.to_excel(writer, sheet_name="All_LOSO_Reports", index=False)
  df_all_cm_raw.to_excel(writer, sheet_name="All_CM_Raw", index=False)
  df_cm_layout.to_excel(
      writer, sheet_name="All_CM_Layout", index=False, header=False
  )

print(
    f"\n--- Successfully appended all metrics, reports, and side-by-side"
    f" confusion matrices to '{excel_file}' ---"
)

# 6. Save scaler and trained model files
print("6. Saving scaler and model files...")
#joblib.dump(scaler, 'har_scaler_dt_100hz_4s.pkl')
#joblib.dump(model, 'har_decision_tree_100hz_4s.pkl')

print("Successfully exported 'har_scaler.pkl' and 'har_decision_tree.pkl' to your working directory!")
