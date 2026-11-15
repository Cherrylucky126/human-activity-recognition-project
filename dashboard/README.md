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
| Status strip | Link state, **measured** sampling rate, heart rate |
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

**The fall threshold is 1.8 g, not the literature's 2.5 g.** The
collector sketch never writes register `0x1C`, so the MPU6050 stays at its
default ±2 g range and every reading is clipped at 2 g before it leaves the
device. A real fall impact is several g but arrives as 2 g, which made the
2.5 g threshold unreachable on any real recording — detection worked in
simulation and could never have fired on the wrist unit.

The team chose on 15 November to lower the threshold rather than set `0x1C`
to ±8 g, because the ±8 g fix invalidates every trained model (all fitted on
±2 g data) and there was no time to re-collect and re-train before the 17th.
The cost is specificity: 1.8 g sits inside the range of ordinary vigorous
movement. What holds the false-alarm rate down is that the impact stage never
fires alone — an alert still needs a 30–150 ms free-fall dip below 0.6 g
before it and a second of stillness after. Walking peaks at 1.60 g in the
self-test against the 1.80 g threshold, so the margin is 0.2 g and is the
first thing to re-check against real recordings. If the firmware is ever
fixed to ±8 g, raise the threshold back to 2.5 g in the same change.

**The random forest loads, and is the model in use.** It was re-saved
under scikit-learn 1.9.1 on 15 November, which fixed the `KeyError: 13` it
used to raise. The dashboard still falls back to the next model that loads
if anything goes wrong with it, and still prints the model actually running
in the footer, so a demo cannot silently use a different model than the
report describes.

**The feature file is found whichever way it is spelled.** It has been
`feature-extraction`, and is now `feature-extraction.py`. `inference.py`
tries both, plus `feature_extraction.py`, and a self-test fails if a
spelling exists that the loader would not find. This is not hypothetical:
the rename on 15 November silently stopped the dashboard loading any model
at all, and the page still looked healthy while predicting nothing.

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

**There is no battery readout, and no gesture readout.** Nothing in this
project measures the cell — the unit runs off an HT7833 with no divider into
an ADC pin — and no gesture classifier was built. Both rows were removed in
PR 16 rather than left showing "not reported" and "not implemented": a strip
of readouts is easier to trust when every row on it is a real measurement.
`live.py` still has `battery_label()` and the gesture label, with their
self-tests, so restoring either row is one `_readout` in `layout.py` and one
`Output` in `app.py`.

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
