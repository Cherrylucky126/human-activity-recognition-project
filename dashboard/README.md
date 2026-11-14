# Live dashboard

A Dash app that shows what the wrist unit is doing, right now: the activity
the classifier thinks it is seeing, how confident it is, the raw sensor
channels, heart rate, a step count, and an alert if it detects a fall.

It runs with no hardware attached. A simulator stands in for the device, so
the dashboard can be built, reviewed and demonstrated on a laptop.

## Running it

```
cd dashboard
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:8050>.

To read the real device instead of the simulator, give it the port the
wrist unit is paired on:

```
HAR_SERIAL_PORT=COM7 python app.py        # Windows
HAR_SERIAL_PORT=/dev/rfcomm0 python app.py # Linux
```

The unit sends over Bluetooth Classic, which pairs as an outgoing serial
port, so the port name is the only thing that differs from a cable. If the
port will not open, the dashboard says so and starts in simulation rather
than refusing to run.

## What is on the page

| Region | Shows |
|---|---|
| Header | Current activity, confidence, clock |
| Status strip | Battery, link state, **measured** sampling rate, heart rate |
| Sensors | Latest accelerometer, gyroscope and IR values, step count |
| Activity probability | Every class's probability, and a timeline of recent predictions |
| Charts | Accelerometer and gyroscope over the last 10 seconds |
| Banner | Fall alert, latched until dismissed |

With the simulator running there is also a control bar: pick an activity, or
press **Simulate a fall** to raise the alert. It is only built when the
source is the simulator, and labels itself, so a demonstration can never be
mistaken for a recording.

## How it fits together

```
datasource.py ──→ live.py ──→ app.py ──→ the page
 (or serialsource.py)   │
                        ├──→ buffer.py       400-sample windows
                        ├──→ inference.py    ──→ data-preprocessing/feature-extraction
                        │                    ──→ data-processing/*.pkl
                        ├──→ falldetection.py
                        ├──→ pedometer.py
                        └──→ vitals.py
```

| File | Does |
|---|---|
| `config.py` | Every constant: 100 Hz, 4 s windows, 50% overlap |
| `datasource.py` | The `SensorSource` interface, and the simulator |
| `serialsource.py` | The same interface, reading the real device |
| `buffer.py` | Keeps recent samples; cuts 400-sample windows, each handed out once |
| `inference.py` | Features (imported from the project) → scaler → model |
| `falldetection.py` | Free fall → impact → stillness, as a causal rule |
| `pedometer.py` | Step counting from accelerometer magnitude |
| `vitals.py` | Heart rate from the PPG channel |
| `live.py` | Owns one of each of the above; every callback reads from it |
| `figures.py` / `timeline.py` | The charts and the prediction strip |
| `layout.py` | The page's shape and its element ids |
| `app.py` | Wiring and callbacks only |

Every module except `app.py` and `layout.py` runs its own self-test:

```
python buffer.py        python falldetection.py   python vitals.py
python datasource.py    python pedometer.py       python figures.py
python inference.py     python timeline.py        python serialsource.py
python live.py
```

They need no hardware and no browser, and they print what they measured
rather than only passing.

## Things to know before trusting a number on this screen

These are real, they change what the display means, and most of them are
not fixable inside the dashboard.

**Fall detection will not fire on real recordings as the firmware
stands.** The collector sketch never writes register `0x1C`, so the
MPU6050 stays at its default ±2 g range. A real fall impact is several g,
so impacts clip at about 2 g and the detector's 2.5 g threshold is never
reached. Either the firmware selects ±8 g — which invalidates every
trained model, all fitted on ±2 g data — or the threshold drops below 2 g
and the alert becomes less specific. `FallDetector` already accepts every
threshold as a keyword argument, so the second route needs no code change.
This is a team decision, not one the dashboard should make quietly.

**The random forest does not load.** `har_random_forest_100hz_4s.pkl`
raises `KeyError: 13` under scikit-learn 1.9.1, having been written by a
different version. It is the model the submission names. The dashboard
falls back to the next one that loads and prints which model is running in
the footer, so a demo cannot silently run a different model than the report
describes.

**There is no "running" class.** The models know six: downstairs, laying,
sitting, standing, upstairs, walking. The repository has a `remove-running`
preprocessing step, so this looks deliberate. Anyone running is classified
as something else, confidently.

**Three of the 43 features are approximated.** IR is z-scored across the
whole dataset before training, which a live system cannot reproduce — there
is no "whole dataset" while data is still arriving. `ir_ac_variance`,
`ir_p2p_amplitude` and `ibi_std` are therefore scaled from a trailing 60
seconds instead. The exact fix is two numbers: put the training set's IR
mean and standard deviation into `TRAINING_IR_MEAN` and `TRAINING_IR_STD`
in `inference.py` and the live features match the training ones exactly.
The other 40 features are unaffected.

**Heart rate is not a medical measurement.** It is an indicative figure
from an optical sensor on a wrist strap, and the page says so. It also
cannot use the pipeline's `rolling_bpm`, which smooths with a centred
median — that needs samples from after the moment it is describing.
`vitals.py` estimates the same quantity from the past only, so the numbers
differ from the offline ones by design.

**Battery reads "not reported".** Nothing in this project measures the
cell: the unit runs off an HT7833 with no divider into an ADC pin. The
readout stays on the page showing the truth rather than being removed, so
the gap stays visible.

**The step count is from a wrist.** Gesturing while seated adds steps;
walking with a hand in a pocket loses them. A fall impact also registers as
one step. Counting only while the classifier says "walking" would fix most
of this, at the cost of coupling the pedometer to the model.

**The sampling rate shown is measured, never assumed.** If it does not read
100, the windows reaching the classifier are not the shape the models were
trained on, and the predictions are wrong even though nothing looks broken.
That readout is the first thing to check.

## Dependencies

`requirements.txt`. The versions this was developed and tested against, on
Python 3.14:

| Package | Version |
|---|---|
| dash | 4.5.0 |
| plotly | 7.1.0 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| pyserial | 3.5 |

`scikit-learn` is the one that matters: the `.pkl` files must be unpickled
by a version compatible with the one that wrote them, which is exactly what
the random forest currently fails on.
