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

    # Fall alert (PR 12).
    FALL_BANNER = "fall-banner"
    FALL_DISMISS = "fall-dismiss"


PLACEHOLDER = "—"
"""Shown wherever a value has no data yet, so empty regions look deliberate."""


def _blank_figure(message="No data yet"):
    """An empty, theme-neutral plot.

    Plotly's default template paints a white canvas with visible axes, which
    fights the dark theme and makes an empty dashboard look broken rather than
    unfinished. Until real figures arrive in PR 8 and PR 11, every graph shows
    this: transparent background, no axes, one muted label.

    The label colour is a mid grey chosen to stay readable on both the light
    and the dark card background, since a figure cannot read CSS variables.
    """
    return {
        "data": [],
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": message,
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "font": {"color": "#9aa3af", "size": 13},
                }
            ],
        },
    }


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


def build_layout():
    """Build the whole page.

    Returns a Dash component tree. Nothing here is live: the interval ticks,
    but no callback is listening to it until PR 3.
    """
    return html.Div(
        className="page",
        children=[
            # Drives every live update from PR 3 onward.
            dcc.Interval(id=Ids.TICK, interval=TICK_INTERVAL_MS, n_intervals=0),
            # Hidden until a fall is confirmed (PR 12). It sits at the top of
            # the page because an alert nobody notices is not an alert.
            html.Div(
                id=Ids.FALL_BANNER,
                className="fall-banner hidden",
                children=[
                    html.Strong("Fall detected"),
                    html.Button("Dismiss", id=Ids.FALL_DISMISS, className="dismiss"),
                ],
            ),
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
                    _readout("Battery", Ids.BATTERY, "%"),
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
            html.Footer(
                className="footer",
                children=html.Span(
                    f"{APP_TITLE} · layout shell, PR 4 of 14 · no live data yet",
                    className="muted",
                ),
            ),
        ],
    )
