# Human Activity Recognition — wrist unit

A wrist-worn ESP32 with an MPU6050 IMU and a MAX30102 pulse sensor, a
machine-learning pipeline that recognises six activities from it, and a live
dashboard that shows what the device is doing right now.

## Layout

```
human-activity-recognition-project/
├── firmware/            Arduino / C++ sketches for the ESP32
│   ├── arduino-ide-data-collection.cpp    the collector: IMU + PPG over Bluetooth
│   └── arduino-ide-imu-calibration.cpp    gyroscope offset calibration
├── data-preprocessing/  filtering, downsampling, feature extraction
├── data-processing/     model training, and the trained .pkl files
├── dashboard/           the live Dash dashboard  (see dashboard/README.md)
├── datasets/            every pipeline CSV, in order  (Git LFS)
├── docs/                plan, schedule, known limitations
└── README.md
```

## Running the dashboard

```
cd dashboard
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:8050>. It runs with no hardware attached: a
simulator stands in for the device. To read the real unit instead, pass the
port it is paired on:

```
HAR_SERIAL_PORT=COM7 python app.py
```

Full instructions, and what every number on the page does and does not mean,
are in [`dashboard/README.md`](dashboard/README.md).

## Re-training the models

The training scripts live in `data-processing/*-builds/`. They need the
processed feature table, `datasets/6HAR_Processed_Features_100hz_4_second_window.csv`.
It is in the repository, but as a Git LFS object, so a plain clone leaves a
130-byte pointer in its place:

```
git lfs install
git lfs pull --include="datasets/6HAR_Processed_Features_100hz_4_second_window.csv"
```

Fetch that one file, not all seven: together they are about 600 MB against a free
account's 1 GB of monthly LFS bandwidth for the whole team.
[`datasets/README.md`](datasets/README.md) explains every file, the order
they are made in, and what to check before re-running the pipeline.

## Before quoting a result

[`docs/known-limitations.md`](docs/known-limitations.md) lists what this
system does not do and where the numbers are optimistic, with suggested
wording for the report. Worth reading before writing up.

## Data

The CSVs are in [`datasets/`](datasets/README.md). The consent forms, and a
copy of the data that costs no LFS bandwidth, are in the project Drive
folder:
<https://drive.google.com/drive/folders/1tDXUz5hRw2sz_EQjFG9E-jbHLzcRiPIT?usp=drive_link>

Participants appear only as `1`–`5`. The repository is private, and
`datasets/README.md` says what making it public would publish.
