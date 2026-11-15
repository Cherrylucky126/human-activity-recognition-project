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
├── data/                where the training CSV goes  (not committed)
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
processed feature table, which is not in this repository — see
[`data/README.md`](data/README.md) for where to get it and where to put it.

## Before quoting a result

[`docs/known-limitations.md`](docs/known-limitations.md) lists what this
system does not do and where the numbers are optimistic, with suggested
wording for the report. Worth reading before writing up.

## Data

Recordings, consent forms and the processed dataset:
<https://drive.google.com/drive/folders/1tDXUz5hRw2sz_EQjFG9E-jbHLzcRiPIT?usp=drive_link>
