"""HAR wrist unit - live dashboard.

Wiring only: this module builds the Dash app, attaches the layout from
``layout.py``, and holds the callbacks that push live values into it. All
shared state - the sensor source, the sample buffer, the fall detector -
lives in ``live.py`` as the single ``LIVE`` object.

Run it with::

    cd dashboard
    python app.py

then open http://127.0.0.1:8050.

Progress: PR 5 of 14. The page is live: the clock runs, the link status and
the measured sampling rate are real. Activity, heart rate, the sensor
readouts and the graphs are still placeholders and are filled by PRs 6-11.

How the callbacks are arranged
------------------------------
Every callback begins with ``LIVE.poll()``. That reads whatever the device
has sent since the last call and files it into the shared buffer. It is
deliberately called from every callback rather than one designated pumping
callback, because Dash makes no promise about the order it runs callbacks
in; ``poll()`` is lock-protected, so several callers in the same tick
between them collect every sample exactly once.
"""

import time

from dash import Dash, Input, Output, callback, ctx, no_update

from config import APP_TITLE, DISPLAY_WINDOW_SEC, HOST, PORT

TIMELINE_SECONDS = 60.0
"""How much prediction history the timeline strip shows.

A minute is about 30 windows: enough to see whether the classifier is
steady, short enough that each block stays wide enough to hover.
"""
import timeline
from figures import line_figure, probability_figure
from layout import Ids, build_layout
from live import LIVE

app = Dash(__name__, title=APP_TITLE)

# The demo controls drive the simulator, so they are only built when the
# source actually is the simulator. In front of a real device they would do
# nothing while implying the opposite.
app.layout = build_layout(show_demo_controls=LIVE.source_kind == "simulated")

# Dash warns about callbacks targeting ids that do not exist. Every id in
# layout.Ids is present from PR 4 on, so nothing is suppressed here; if a
# later PR needs an id created at runtime, set suppress_callback_exceptions
# above rather than inventing hidden divs.


@callback(
    Output(Ids.CLOCK, "children"),
    Output(Ids.CONNECTION, "children"),
    Output(Ids.SAMPLE_RATE, "children"),
    Output(Ids.HEART_RATE, "children"),
    Output(Ids.ACTIVITY, "children"),
    Output(Ids.CONFIDENCE, "children"),
    Output(Ids.MODEL_NAME, "children"),
    Input(Ids.TICK, "n_intervals"),
)
def update_status(_n_intervals):
    """Clock, link state and measured sampling rate, ten times a second.

    The sampling rate shown is **measured from the arriving timestamps**,
    never the configured 100 Hz. That distinction is the whole point of the
    readout: if the firmware is sending at a different rate than the models
    were trained on, this is the one place on the page that will say so,
    and the predictions would otherwise be quietly wrong rather than
    visibly broken.

    Gesture and battery are deliberately not outputs here. The device
    reports neither, so both could only ever display a placeholder, and a
    row of readouts is more trustworthy when every row on it is a real
    measurement. ``live.py`` still computes both and still tests them, so
    restoring either is one Output here and one ``_readout`` in
    ``layout.py``.
    """
    LIVE.poll()
    activity, confidence, _gesture = LIVE.activity_labels()
    return (
        _wall_clock(),
        LIVE.connection_label(),
        LIVE.sample_rate_label(),
        LIVE.heart_rate_label(),
        activity,
        confidence,
        LIVE.model_label(),
    )


@callback(
    Output(Ids.ACCEL_VALUES, "children"),
    Output(Ids.GYRO_VALUES, "children"),
    Output(Ids.IR_VALUE, "children"),
    Output(Ids.STEPS, "children"),
    Input(Ids.TICK, "n_intervals"),
)
def update_sensors(_n_intervals):
    """The raw sensor panel and the step count.

    All four values are read in one call, so the panel cannot show three
    numbers from one instant and a fourth from the next - which is exactly
    the kind of inconsistency nobody notices on screen but which makes a
    screenshot in the report wrong.

    The step count is maintained in ``live.poll()`` rather than here. If it
    were counted from whatever this callback happened to see, a dropped
    browser frame would silently lose steps.
    """
    LIVE.poll()
    return LIVE.sensor_labels()


@callback(
    Output(Ids.ACCEL_GRAPH, "figure"),
    Output(Ids.GYRO_GRAPH, "figure"),
    Input(Ids.TICK, "n_intervals"),
)
def update_graphs(_n_intervals):
    """The two live line charts.

    Both are built from one read of the buffer, so the accelerometer and
    gyroscope charts always cover the same ten seconds. Reading twice
    would let them drift a frame apart, which looks like a lag between the
    sensors that is not there.
    """
    LIVE.poll()
    history = LIVE.buffer.get_recent_seconds(DISPLAY_WINDOW_SEC)
    axes = ("X", "Y", "Z")
    return (
        line_figure(history, ("ax", "ay", "az"), axes, "g"),
        line_figure(history, ("gx", "gy", "gz"), axes, "deg/s", y_zero_line=True),
    )


@callback(
    Output(Ids.PROBABILITY_BARS, "figure"),
    Output(Ids.PREDICTION_TIMELINE, "children"),
    Input(Ids.TICK, "n_intervals"),
)
def update_classifier_panel(_n_intervals):
    """What the model thinks now, and what it has been thinking.

    The bars show the current window's probabilities; the strip below
    shows the recent run of predictions. The strip is the one that answers
    "is this steady or is it flickering?", which the big activity label
    cannot.
    """
    LIVE.poll()
    classes = LIVE.classifier.classes
    history = LIVE.prediction_history()
    latest = history[-1][1] if history else None

    figure = (
        probability_figure(latest.probabilities, classes)
        if latest is not None
        else probability_figure(None, classes)
    )
    return figure, timeline.build(history, classes, seconds=TIMELINE_SECONDS)


@callback(
    Output(Ids.FALL_BANNER, "className"),
    Output(Ids.FALL_DETAIL, "children"),
    Input(Ids.TICK, "n_intervals"),
    Input(Ids.FALL_DISMISS, "n_clicks"),
)
def update_fall_banner(_n_intervals, _n_clicks):
    """Raise, describe and lower the fall alert.

    One callback rather than two, because both the tick and the Dismiss
    button write the same className. Two callbacks with the same output
    is an error in Dash, and the version that works - the tick quietly
    overwriting the dismissal a tenth of a second later - would be worse:
    a banner that refuses to go away.

    The alert latches in the detector until dismissed. A fall that
    appeared for one frame while nobody was looking at the screen would
    not be an alert.
    """
    if ctx.triggered_id == Ids.FALL_DISMISS:
        LIVE.dismiss_fall()
        return "fall-banner hidden", ""

    LIVE.poll()
    active, detail = LIVE.fall_state()
    if not active:
        return "fall-banner hidden", ""
    return "fall-banner", detail


@callback(
    Output(Ids.DEMO_ACTIVITY, "value"),
    Input(Ids.DEMO_ACTIVITY, "value"),
    Input(Ids.DEMO_FALL, "n_clicks"),
    prevent_initial_call=True,
)
def drive_simulator(activity, _n_clicks):
    """Act on the demo controls.

    Only reachable when the layout was built with them, which only happens
    with a simulated source. ``LIVE`` guards the calls anyway, so a real
    device attached mid-session cannot turn this into an exception inside
    a callback.

    The dropdown's own value is the output so the callback has somewhere
    to write; selecting an activity leaves it unchanged, and the fall
    button does not disturb it.
    """
    if ctx.triggered_id == Ids.DEMO_FALL:
        LIVE.trigger_demo_fall()
        return no_update

    if activity:
        LIVE.set_demo_activity(activity)
    return no_update


def _wall_clock():
    """Local time as hh:mm:ss.

    Wall-clock time, not the sample timestamps: this readout answers "is
    this screen frozen?", which a monotonic clock cannot.
    """
    import time

    return time.strftime("%H:%M:%S")


if __name__ == "__main__":
    LIVE.start()

    # The reloader is switched off when a real device is attached, and this
    # is not a preference.
    #
    # Dash's debug mode runs Werkzeug's reloader, which imports this module
    # in a parent process and then again in a child. Two imports means two
    # LiveState objects and two attempts to open the serial port. A serial
    # port belongs to one process at a time, so the first import takes it
    # and the second is refused - and the second is the one that serves the
    # page. The result is a dashboard reading the simulator while the
    # device sits there connected, which is exactly the confusion the
    # connection readout was added to expose.
    #
    # With the simulator there is no shared resource and the reloader is
    # genuinely useful, so it stays on.
    reading_device = LIVE.source_kind == "device"
    app.run(
        debug=not reading_device,
        use_reloader=not reading_device,
        host=HOST,
        port=PORT,
    )
