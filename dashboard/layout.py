"""Page layout for the live dashboard.

PR 4 of 14: the shell only. Every region on the finished dashboard exists here
as an empty placeholder, so the PRs that follow can fill one region each
without rearranging the page.

Why a separate module: ``app.py`` stays small and is only about wiring, while
the shape of the page lives here. Later PRs add callbacks in ``app.py`` that
target the element ids in :class:`Ids`.

The ids are the contract between this file and every later PR. Treat them as
fixed: renaming one silently breaks whichever callback was writing to it,
because Dash resolves these by string at runtime, not at import.
"""

from dash import dcc, html

from config import APP_TITLE, DISPLAY_WINDOW_SEC, TICK_INTERVAL_MS
from figures import empty_figure as _blank_figure


class Ids:
    """Element ids, referenced by the callbacks added in later PRs."""

    # The clock that drives every live update (PR 3).
    TICK = "tick"

    # Header (PR 5).
    ACTIVITY = "activity"
    CONFIDENCE = "confidence"
    CLOCK = "clock"
    GESTURE = "gesture"

    # Status strip (PR 6).
    BATTERY = "battery"
    CONNECTION = "connection"
    SAMPLE_RATE = "sample-rate"
    HEART_RATE = "heart-rate"

    # Sensor panel and pedometer (PR 7).
    ACCEL_VALUES = "accel-values"
    GYRO_VALUES = "gyro-values"
    IR_VALUE = "ir-value"
    STEPS = "steps"

    # Graphs (PR 8).
    ACCEL_GRAPH = "accel-graph"
    GYRO_GRAPH = "gyro-graph"

    # Classifier output (PR 11).
    PROBABILITY_BARS = "probability-bars"
    PREDICTION_TIMELINE = "prediction-timeline"

    # Fall alert (PR 9 wires the banner; the rule itself is PR 12).
    FALL_BANNER = "fall-banner"
    FALL_DISMISS = "fall-dismiss"
    FALL_DETAIL = "fall-detail"

    # Demo controls (PR 9). Present only when the source is the simulator,
    # so they can never appear in front of a real recording.
    DEMO_ACTIVITY = "demo-activity"
    DEMO_FALL = "demo-fall"


PLACEHOLDER = "—"
"""Shown wherever a value has no data yet, so empty regions look deliberate."""


def _card(title, children, class_name="card"):
    """One titled panel.

    Every region on the page is a card, which is what keeps the grid tidy: the
    CSS sizes cards, not their contents.
    """
    return html.Section(
        className=class_name,
        children=[html.H2(title, className="card-title"), *children],
    )


def _readout(label, element_id, unit=""):
    """A label above a big value, for the status strip and sensor panel."""
    return html.Div(
        className="readout",
        children=[
            html.Span(label, className="readout-label"),
            html.Div(
                className="readout-value",
                children=[
                    html.Span(PLACEHOLDER, id=element_id),
                    html.Span(unit, className="readout-unit") if unit else None,
                ],
            ),
        ],
    )


def _demo_bar():
    """Controls for driving the simulator during a demonstration.

    A fall banner that cannot be made to appear is a banner nobody has
    checked. These make the alert, and every activity, reproducible in
    front of an audience without anyone having to throw the device at the
    floor.

    The bar labels itself as simulated. Someone walking past the screen
    should not be able to mistake a demonstration for a recording.
    """
    return html.Div(
        className="demo-bar",
        children=[
            html.Span("Simulator", className="demo-tag"),
            dcc.Dropdown(
                id=Ids.DEMO_ACTIVITY,
                options=[
                    {"label": name.replace("_", " "), "value": name}
                    for name in DEMO_ACTIVITIES
                ],
                value="sitting",
                clearable=False,
                className="demo-select",
            ),
            html.Button(
                "Simulate a fall",
                id=Ids.DEMO_FALL,
                className="demo-button",
                n_clicks=0,
            ),
        ],
    )


DEMO_ACTIVITIES = (
    "sitting",
    "standing",
    "lying",
    "walking",
    "running",
    "walking_upstairs",
    "walking_downstairs",
)
"""The activities the simulator can play, for the demo control.

Kept as a plain tuple here rather than imported from ``datasource``, so
that building the page never depends on a sensor module being importable.
``app.py`` checks the two agree.
"""


def build_layout(show_demo_controls=False):
    """Build the whole page.

    Parameters
    ----------
    show_demo_controls:
        Add the activity selector and the "Simulate a fall" button. These
        drive the simulator, so ``app.py`` only passes True when the source
        actually is the simulator - they can never appear in front of a
        real recording, where they would do nothing and imply the opposite.
    """
    return html.Div(
        className="page",
        children=[
            # Drives every live update.
            dcc.Interval(id=Ids.TICK, interval=TICK_INTERVAL_MS, n_intervals=0),
            # Hidden until a fall is confirmed. It sits at the top of the
            # page because an alert nobody notices is not an alert.
            html.Div(
                id=Ids.FALL_BANNER,
                className="fall-banner hidden",
                children=[
                    html.Div(
                        children=[
                            html.Strong("Fall detected"),
                            html.Span(
                                "",
                                id=Ids.FALL_DETAIL,
                                className="fall-detail",
                            ),
                        ]
                    ),
                    html.Button("Dismiss", id=Ids.FALL_DISMISS, className="dismiss"),
                ],
            ),
            _demo_bar() if show_demo_controls else html.Div(),
            # Header: the one thing a person glances at.
            html.Header(
                className="header",
                children=[
                    html.Div(
                        className="header-main",
                        children=[
                            html.Span("Current activity", className="readout-label"),
                            html.H1(PLACEHOLDER, id=Ids.ACTIVITY, className="activity"),
                        ],
                    ),
                    html.Div(
                        className="header-side",
                        children=[
                            _readout("Confidence", Ids.CONFIDENCE, "%"),
                            _readout("Gesture", Ids.GESTURE),
                            _readout("Time", Ids.CLOCK),
                        ],
                    ),
                ],
            ),
            # Status strip: is the thing actually working?
            html.Div(
                className="status-strip",
                children=[
                    # No unit on the battery readout: the device sends no
                    # battery level at all, so this shows "not reported"
                    # rather than a number, and "not reported %" would be
                    # nonsense. See live.battery_label().
                    _readout("Battery", Ids.BATTERY),
                    _readout("Connection", Ids.CONNECTION),
                    _readout("Sampling rate", Ids.SAMPLE_RATE, "Hz"),
                    _readout("Heart rate", Ids.HEART_RATE, "bpm"),
                ],
            ),
            html.Div(
                className="grid",
                children=[
                    _card(
                        "Sensors",
                        [
                            html.Div(
                                className="sensor-grid",
                                children=[
                                    _readout("Accelerometer (g)", Ids.ACCEL_VALUES),
                                    _readout("Gyroscope (deg/s)", Ids.GYRO_VALUES),
                                    _readout("IR", Ids.IR_VALUE),
                                    _readout("Steps", Ids.STEPS),
                                ],
                            )
                        ],
                    ),
                    _card(
                        "Activity probability",
                        [
                            dcc.Graph(
                                id=Ids.PROBABILITY_BARS,
                                figure=_blank_figure("Waiting for a prediction"),
                                config={"displayModeBar": False},
                                className="graph short",
                            ),
                            html.Div(
                                id=Ids.PREDICTION_TIMELINE,
                                className="timeline",
                                children=html.Span(
                                    f"Last {DISPLAY_WINDOW_SEC * 6} s of predictions",
                                    className="muted",
                                ),
                            ),
                        ],
                    ),
                    _card(
                        f"Accelerometer — last {DISPLAY_WINDOW_SEC} s",
                        [
                            dcc.Graph(
                                id=Ids.ACCEL_GRAPH,
                                figure=_blank_figure("Waiting for sensor data"),
                                config={"displayModeBar": False},
                                className="graph",
                            )
                        ],
                        class_name="card wide",
                    ),
                    _card(
                        f"Gyroscope — last {DISPLAY_WINDOW_SEC} s",
                        [
                            dcc.Graph(
                                id=Ids.GYRO_GRAPH,
                                figure=_blank_figure("Waiting for sensor data"),
                                config={"displayModeBar": False},
                                className="graph",
                            )
                        ],
                        class_name="card wide",
                    ),
                ],
            ),
            # The heart-rate caveat is on the page itself, not only in the
            # report. Anyone reading a number off this screen during the
            # demo should see, in the same glance, that it is not a
            # clinical measurement.
            html.Footer(
                className="footer",
                children=[
                    html.Span(APP_TITLE, className="muted"),
                    html.Span(
                        "Heart rate is an indicative signal from an optical "
                        "sensor, not a medical measurement.",
                        className="muted caveat",
                    ),
                ],
            ),
        ],
    )
