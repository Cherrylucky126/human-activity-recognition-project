# Datasets

Every CSV the pipeline reads or writes, in order, from the raw recordings to
the feature table the models are trained on.

Unlike the `data/` folder this replaces, the files **are** in the repository.
They are stored with Git LFS, so a plain clone gives you seven 130-byte
pointer files instead of the data — see [Getting the actual files](#getting-the-actual-files)
below before you run anything.

## What is here

| # | File | Size | What it is |
|---|---|---|---|
| 1 | `1HAR_Raw_Master_Dataset.csv` | 112 MB | Everything the wrist unit sent, concatenated. One row per sample at 100 Hz: `ax ay az gx gy gz`, `IR_Value`, and the `Participant` / `Activity_Label` / `Trial_Number` labels. |
| 2 | `2HAR_No_Running_Master_Data.csv` | 98 MB | The same, with every `running` row dropped (`remove-running.py`). This is why the models know six activities and not seven. |
| 3 | `3HAR_ID_Maker.csv` | 93 MB | The participant and trial IDs put into their final form. No committed script produces this one — it was made outside the repository. |
| 4.1 | `4.1ESP32_MPU6050_Calibration_Data.csv` | 369 KB | A flat, still recording used to measure the gyroscope's zero offset. Not training data: it exists so the bias in step 4.2 can be calculated. |
| 4.2 | `4.2HAR_Fully_Calibrated.csv` | 114 MB | Step 3 with that gyroscope bias subtracted (`remove-gyro-offset.py`). |
| 5 | `5HAR_Filtered_Dataset.csv` | 192 MB | Band-pass and low-pass filtered, trial by trial so no filter runs across a boundary (`data-filtration.py`). |
| 6 | `6HAR_Processed_Features_100hz_4_second_window.csv` | 3.8 MB | **The one the training scripts use.** 4-second windows at 100 Hz, 50% overlap: 5,217 rows × 43 feature columns, plus `Participant`, `Activity_Label`, `Trial_Number`. |

`5.1HAR_Downsampled_50Hz.csv`, from `downsampling-by-half.py`, is not here.
Nothing downstream reads it, and the models are all fitted at 100 Hz.

### What is in the feature table

5 participants, coded `1`–`5`. 6 activities — downstairs, laying, sitting,
standing, upstairs, walking — with 867–870 windows each, so the classes are
balanced to within half a percent. 3 trials per participant per activity.

The 43 features are 5 per axis for all six IMU axes (mean, std, RMS,
dominant frequency, energy), 8 across the axes (accelerometer magnitude
mean/std/max, spectral energy, dominant frequency, gyroscope magnitude
mean/std, X–Y correlation), SMA, and 4 from the PPG channel
(`ir_ac_variance`, `ir_p2p_amplitude`, `rolling_bpm`, `ibi_std`).

Three of those four PPG features cannot be reproduced exactly live — see
[`../docs/known-limitations.md`](../docs/known-limitations.md).

## Getting the actual files

The CSVs are Git LFS objects. Without LFS you get the pointer files, and
`pd.read_csv` then fails on a 130-byte file with an error that looks like a
corrupt dataset rather than a missing one.

```
git lfs install          # once per machine
git lfs pull             # fetches all seven - about 600 MB
```

600 MB is most of a free GitHub account's 1 GB of LFS storage, and **every
pull counts against 1 GB of bandwidth a month, shared by the whole team**.
Four people each fetching all seven files exceeds it, and LFS then refuses
downloads until the quota resets or a data pack is bought. So fetch the one
file you actually need:

```
git lfs pull --include="datasets/6HAR_Processed_Features_100hz_4_second_window.csv"
```

That is 3.8 MB, and it is the only file any training script reads.

A copy also lives in the project Drive folder linked from the top-level
[`README.md`](../README.md), which costs no LFS bandwidth at all.

## Pointing a script somewhere else

Every script in `data-preprocessing/` and `data-processing/` looks in this
folder by default. To keep a file elsewhere, set `HAR_DATASET` to its full
path:

```
set HAR_DATASET=D:\har\features.csv        # Windows
export HAR_DATASET=/data/har/features.csv  # Linux / macOS
```

A script run without the file prints where it looked and what to fetch,
rather than a `FileNotFoundError` on someone else's absolute path.

## Two things to know before re-running the pipeline

**The stage inputs do not currently match this table.** `remove-running.py`,
`remove-gyro-offset.py` and `data-filtration.py` all have `DATASET_NAME` set
to file 6, the pipeline's *output*, rather than to the file that stage is
meant to consume (1, 3 and 4.2 respectively). Run as they stand, they read
the finished feature table and write raw-shaped output over the stage names
above. The table here describes the intended order; the scripts are
Cherrylucky126's and are flagged for her, not changed here.

**Stage outputs are written to the current directory**, not to this folder,
because each script calls `to_csv('<name>.csv')` with a bare filename. Run
from the repository root, a regenerated file lands at the root rather than
beside its inputs. Move it in by hand, or run from inside `datasets/`.

## Why the data is in the repository at all, and what that means

Participants are numbered, not named: the only identifier in any of these
files is `Participant` as `1`–`5`, and the consent forms stay in the Drive
folder, out of version control.

The repository is private. If it is ever made public, these seven files go
with it, including every raw sample of five people's movement and pulse —
LFS history is not removed by deleting a file in a later commit. Anonymised
or not, publishing it is a decision for the whole team and the participants,
not a side effect of flipping a repository setting.
