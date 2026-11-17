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

```powershell
$env:HAR_SERIAL_PORT = "COM7"    # PowerShell, which is what Windows opens
python app.py
```

```bash
HAR_SERIAL_PORT=/dev/rfcomm0 python app.py    # Linux, or Git Bash
```

Use the PowerShell form on Windows. `HAR_SERIAL_PORT=COM7 python app.py` is
bash syntax; PowerShell rejects it outright, which is the first thing that
goes wrong for everyone who copies the Linux line.

The unit sends over Bluetooth Classic, which pairs as an outgoing serial
port, so the port name is the only thing that differs from a cable. If the
port will not open, the dashboard says so and starts in simulation rather
than refusing to run.

Full pairing steps are under **Connecting the wrist unit** below.

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
| `replay.py` | The device's wire format, replayed through the whole stack |
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
python live.py          python replay.py
```

They need no hardware and no browser, and they print what they measured
rather than only passing.

## Connecting the wrist unit

The dashboard reads the device when `HAR_SERIAL_PORT` names a port, and the
simulator when it does not. That is the whole switch, and it is decided once
at startup: there is no simulator process to stop, because the simulator is
an object inside this app.

**1. Power the unit.** The sketch starts advertising at boot and streams as
soon as something connects. There is no start command to send.

**2. Pair it.** Settings > Bluetooth & devices > Add device > Bluetooth, and
pick `ESP32_HAR_Classic`. If a PIN is wanted, try `1234` or `0000`.

**3. Find the port.**

```powershell
python serialsource.py --ports
```

Windows creates **two** ports per paired device, Incoming and Outgoing.
Only the Outgoing one carries data; the Incoming one opens quite happily
and then stays silent forever. If the listing is empty, the unit is not
paired or not powered.

**4. Free the port.** Close the Arduino IDE's Serial Monitor. Windows gives
a port to one program at a time and whoever asks second is refused.

**5. Run it.**

```powershell
$env:HAR_SERIAL_PORT = "COM7"
python app.py
```

**6. Check the page agrees.** Three readings, in order of how much they
tell you:

| | Simulator | Device |
|---|---|---|
| Demo control bar | present | **gone** |
| Connection readout | `live (simulated)` | `live (device)` |
| Sampling rate | exactly 100 Hz | around 100 Hz, wobbling |

The demo bar is the reliable one: it is only ever built when the source is
the simulator.

### The unit in the team's hands sends 52 Hz, and that invalidates predictions

Read this before demonstrating on the real device.

The wrist unit does not run the sketch committed in `firmware/`. It sends
seven fields instead of eight, with no sample counter, and it sends them at
about **52 Hz** rather than 100. Both formats are read correctly, so sensor
values and charts are live and right.

Activity classification is **not** valid on it. `config.py` cuts windows of
400 samples because 400 samples is 4 seconds at 100 Hz; at 52 Hz the same
400 samples cover 7.7 seconds, and every model was fitted on 4. The model
will keep returning confident labels regardless - on a motionless desk it
read "laying, 62%".

The other three readouts *are* corrected, as of 17 November. Heart rate,
the step count and fall detection are all timing measurements, and all
three were built with the configured 100 Hz rather than the real one. An
earlier version of this section said only classification was affected,
which was wrong: at 52 Hz the heart rate read about twice the true pulse
and still passed its 40-200 bpm plausibility check, the pedometer enforced
half its intended gap between steps, and the fall detector measured a
100 ms free fall as 192 ms - past its own maximum, so the sequence was
discarded and **falls stopped being detected at all**.

`live.py` now measures the rate and re-tunes those three to it. The footer
says so when it happens (`timings re-tuned to 52 Hz`). The classifier is
deliberately left alone, because nothing in the dashboard can refit a model
to a different window length.

Check the sampling rate readout before trusting anything on the page. If it
does not say 100, the predictions are meaningless no matter how reasonable
they look. Flashing the committed 100 Hz sketch onto the unit fixes it, and
nothing in the dashboard or the models needs to change.

Full detail in [`docs/known-limitations.md`](../docs/known-limitations.md),
section 2b.

### When it will not connect

The dashboard never refuses to start over a bad port. It falls back to the
simulator, prints the reason, and now says so on the page itself:

```
live (simulated, COM99 would not open)
```

That readout exists because the failure is otherwise invisible: a page
reading `live (simulated)` looks completely healthy, and the only clue is a
line in a terminal nobody is watching. A demonstration given on simulated
data by accident is the thing worth engineering against.

The connection readout also reports a lossy link once data is flowing, for
example `live (device, 40 dropped)`. Dropped samples mean gaps in the
windows reaching the classifier, which nothing else on the page would show.

`$env:` persists for the whole PowerShell window. To go back to the
simulator in the same window, clear it first with
`$env:HAR_SERIAL_PORT = ""`, or the dashboard will keep trying the device
and keep falling back.

### Testing the device path without a device

```
python replay.py
```

`replay.py` generates lines in the sketch's exact wire format, applies the
hardware's limits to them, and drives the whole dashboard off them through
a fake serial port. It checks that predictions, steps, fall alerts and
dropped-packet reporting all come out of the far end.

It is a harder test than the simulator, because it reproduces three things
the simulator cannot: the accelerometer clipping at 2 g per axis, the
gyroscope clipping at 250 deg/s, and every float being rounded to the two
decimals `%.2f` prints. The simulator's own docstring notes that its falls
are cleaner than recorded ones; this is where that gap gets closed.

## Things to know before trusting a number on this screen

These are real, they change what the display means, and most of them are
not fixable inside the dashboard.

**The fall threshold is 1.8 g, not the literature's 2.5 g.** The
collector sketch never writes register `0x1C`, so the MPU6050 stays at its
default ±2 g range and **each axis** is clipped at 2 g before it leaves the
device. A real fall impact is several g, and whatever it was, no single
channel reports more than 2.

Note the per-axis part, because the first version of this paragraph got it
wrong and the error is worth not repeating. The detector works on the
vector magnitude, √(ax²+ay²+az²), which is *not* capped at 2 g: three axes
at 2 g give 3.46 g. `replay.py` measures 2.86 g at impact on clipped data.

So the honest statement is that the ceiling depends on orientation. An
impact landing square on one axis gives 2.00 g and a 2.5 g threshold can
never fire; one landing across two or three axes reaches 2.8–3.5 g and it
can. A 2.5 g threshold would therefore catch some falls and be physically
incapable of catching others, depending on how the wrist happened to be
turned — which is a good reason to lower it, but not the reason recorded
on 15 November.

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

**The model is chosen on the page.** The footer has a picker listing every
trained model; the SVM is the default. The name beside it is read from the
model that actually loaded, not from the picker, so if a switch fails the
two disagree and the footer is the one telling the truth. The saved SVM is
a `CalibratedClassifierCV` wrapping the support vector classifier, which is
what gives it the calibrated `predict_proba` the probability bars need.

**The footer reports failed windows.** `predict()` swallows an exception
and keeps the previous prediction on screen, so a classifier failing on
every window would otherwise look like one that is working and sure of
itself - same label, full confidence, clock still ticking.

**The random forest loads.** It was re-saved
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
