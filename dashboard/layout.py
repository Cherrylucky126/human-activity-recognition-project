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

from config import (
    APP_TITLE,
    DISPLAY_WINDOW_SEC,
    TICK_INTERVAL_MS,
    pretty_label,
)
from figures import empty_figure as _blank_figure
from inference import model_names


class Ids:
    """Element ids, referenced by the callbacks added in later PRs."""

    # The clock that drives every live update (PR 3).
    TICK = "tick"

    # Header (PR 5).
    ACTIVITY = "activity"
    CONFIDENCE = "confidence"
    CLOCK = "clock"

    # No GESTURE or BATTERY id. Both readouts were removed in PR 16: the
    # device reports neither, so both could only ever show a placeholder.
    # live.py still has gesture_label() and battery_label() with their
    # self-tests, so putting either readout back is adding one line here and
    # one Output in app.py - see live.battery_label().

    # Status strip (PR 6).
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

    # Classifier (PR 10).
    MODEL_NAME = "model-name"

    # Classifier output (PR 11).
    PROBABILITY_BARS = "probability-bars"
    PREDICTION_TIMELINE = "prediction-timeline"

    # Fall alert (PR 9 wires the banner; the rule itself is PR 12).
    FALL_BANNER = "fall-banner"
    FALL_DISMISS = "fall-dismiss"
    DEMO_BAR = "demo-bar"
    STATUS_DOT = "status-dot"
    MODEL_SELECT = "model-select"
    RECONNECT = "reconnect"
    RECONNECT_STATUS = "reconnect-status"
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


def _model_picker():
    """A dropdown for choosing which trained model classifies.

    Sits beside the footer's model name, which is written by the status
    callback from whichever model is *actually* loaded. So the label is
    not an echo of this dropdown: if a model fails to load, the dropdown
    shows what was asked for and the label shows what is running, and the
    two disagreeing is the signal.

    Every model in the fallback chain is offered. The list comes from
    ``inference.MODEL_CANDIDATES`` rather than being typed again here, so
    adding a model in one place puts it on the page.
    """
    return html.Div(
        className="model-picker",
        children=[
            # No label here: the footer already reads "Model: SVM" two
            # items to the left, and the dropdown's own placeholder says
            # "Switch model". A third "Model" was just noise.
            dcc.Dropdown(
                id=Ids.MODEL_SELECT,
                # The value stays the model's own name - it is what
                # inference matches on - while the label is formatted.
                options=[
                    {"label": pretty_label(n), "value": n} for n in model_names()
                ],
                value=None,
                placeholder="switch model",
                clearable=False,
                searchable=False,
                className="demo-select model-select",
            ),
        ],
    )


def _reconnect_control():
    """A button that hunts for the device again, and a line saying what happened.

    This sits in the status strip, beside the readouts that say something is
    wrong, because that is where someone looks when the rate is bad or the
    link has died. The two failures this hardware actually has - the unit
    drops its Bluetooth link, and it comes back on a different COM port
    after re-pairing - both leave the app running and useless, and both are
    fixed by looking for the device again rather than restarting the app.

    It is always built, never only for the device: the most useful moment
    to press it is when the dashboard fell back to the simulator because
    nothing was connected at startup.
    """
    return html.Div(
        className="reconnect",
        children=[
            html.Button(
                "Reconnect",
                id=Ids.RECONNECT,
                className="demo-button",
                n_clicks=0,
                title="Look for the wrist unit again, on any port",
            ),
            html.Span("", id=Ids.RECONNECT_STATUS, className="reconnect-status"),
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
        id=Ids.DEMO_BAR,
        className="demo-bar",
        children=[
            html.Span("Simulator", className="demo-tag"),
            dcc.Dropdown(
                id=Ids.DEMO_ACTIVITY,
                options=[
                    {"label": pretty_label(name), "value": name}
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
            # Masthead: what this page is, and whether it is alive.
            #
            # The page used to open straight onto the activity label,
            # which reads well on a laptop and badly on a projector,
            # where the audience sees a word with no frame around it.
            html.Div(
                className="masthead",
                children=[
                    html.Div(
                        className="masthead-title",
                        children=[
                            # The dot's colour is set by the status
                            # callback from the live connection state, so
                            # the one thing visible from the back of a
                            # room is whether data is arriving.
                            html.Span(id=Ids.STATUS_DOT, className="dot dot-waiting"),
                            html.Span(APP_TITLE, className="masthead-name"),
                        ],
                    ),
                    html.Span(
                        "Team 2 · Applied Computer Science Lab · SRH Heidelberg",
                        className="masthead-sub",
                    ),
                ],
            ),
            # Always built; the status callback shows or hides it.
            #
            # It used to be built only when the source was the simulator,
            # decided once at startup. Reconnect can change the source
            # afterwards, which left the simulator's controls on screen
            # beside a real device - the precise confusion the bar's label
            # exists to prevent. Deciding it every tick means the bar
            # follows whatever is really feeding the page.
            _demo_bar(),
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
                            _readout("Time", Ids.CLOCK),
                        ],
                    ),
                ],
            ),
            # Status strip: is the thing actually working?
            html.Div(
                className="status-strip",
                children=[
                    _readout("Connection", Ids.CONNECTION),
                    _readout("Sampling rate", Ids.SAMPLE_RATE, "Hz"),
                    _readout("Heart rate", Ids.HEART_RATE, "bpm"),
                    _reconnect_control(),
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
                    # Which model is running, named on screen. The
                    # dashboard falls back when a model will not load, and
                    # a demo silently running one model while the report
                    # describes another is caught at the viva, not before.
                    html.Span(PLACEHOLDER, id=Ids.MODEL_NAME, className="muted"),
                    _model_picker(),
                    html.Span(
                        "Heart rate is an indicative signal from an optical "
                        "sensor, not a medical measurement.",
                        className="muted caveat",
                    ),
                ],
            ),
        ],
    )
