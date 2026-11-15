# Dataset

The training data is not in the repository. It is participant recordings,
and it is large.

## What goes here

```
data/HAR_Processed_Features_100hz_4_second_window.csv
```

The processed feature table: one row per 4-second window at 100 Hz, 43
feature columns plus `Participant`, `Activity_Label` and `Trial_Number`.

## Where to get it

The project Drive folder, linked from the top-level `README.md`.

## Using it from somewhere else

Every training script in `data-processing/` looks here by default. To keep
the file elsewhere, set `HAR_DATASET` to its full path:

```
set HAR_DATASET=D:\har\features.csv        # Windows
export HAR_DATASET=/data/har/features.csv  # Linux / macOS
```

A script run without the file prints where it looked and what to download,
rather than a bare `FileNotFoundError` on someone else's absolute path.

## Why it is not committed

It is recordings of named participants. Consent forms cover use within the
project; publishing the raw data in a repository is a different thing, and
not one this project asked them for. The Drive folder keeps access
deliberate.
