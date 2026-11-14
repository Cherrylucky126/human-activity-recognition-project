"""The live line charts.

PR 8 of 14.

What is being drawn
-------------------
Two charts, each three series: the accelerometer axes in g and the
gyroscope axes in deg/s, over the last ten seconds. Separate charts rather
than one chart with two y-axes - g and deg/s differ by two orders of
magnitude, and a second y-axis lets a reader infer crossings and
correlations between the two that are pure artefacts of how the axes were
scaled.

Time runs right-to-left toward zero: the right edge is now, and a point at
-4 happened four seconds ago. Absolute clock time on the axis would make
every tick label change on every frame and tell the reader nothing they
want.

The colours
-----------
``SERIES_COLORS`` is three slots from a palette validated for colour-vision
deficiency and for contrast, checked against both the light and the dark
card background this page actually uses. One set serves both themes
deliberately: a figure is built on the server and cannot read a CSS
variable, so a theme-specific palette would mean guessing which theme the
browser is in and being wrong half the time.

Colour is never the only thing carrying identity. Every chart has a legend
*and* each line is labelled at its right-hand end, which is also what the
palette's contrast rule requires for the lighter hues.

Why the data is thinned
-----------------------
Ten seconds at 100 Hz is 1000 points per series, 3000 per chart, redrawn
ten times a second. That is a few hundred kilobytes a second of JSON for
detail finer than a screen pixel. ``MAX_POINTS`` keeps every nth sample so
a chart carries at most ~500 points per series, which at a ten-second width
is still several points per pixel column.
"""

from __future__ import annotations

SERIES_COLORS = ("#3987e5", "#d95926", "#199e70")
"""Blue, orange, green: slots 1-3 of the reference categorical palette.

Validated with the palette checker against both surfaces this page uses
(``#ffffff`` light card, ``#171a20`` dark card), all pairs: lightness band,
chroma floor, colour-vision separation, normal-vision separation and
contrast all pass in both modes. Do not substitute by eye - re-run the
checker.
"""

CLASS_COLORS = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # green
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # dark green
)
"""One colour per activity class, for the probability bars and timeline.

Six slots of the same reference palette, validated on both card surfaces
for adjacent pairs: lightness, chroma, colour-vision separation,
normal-vision separation and contrast all pass in light and dark.

Assigned to classes in the model's own fixed class order, never by rank.
If colour followed "most likely" then every bar would change colour each
time the prediction changed, and the timeline would repaint its history.
A class keeps its colour for as long as the dashboard is open.
"""


def class_color(label, classes):
    """The colour for one class, by its position in the model's class list."""
    try:
        index = list(classes).index(label)
    except ValueError:
        return AXIS_TEXT
    return CLASS_COLORS[index % len(CLASS_COLORS)]


GRID_COLOR = "rgba(128,128,128,0.16)"
"""Grid lines, recessive by design and theme-neutral.

Mid grey at low alpha reads as a faint line on a white card and on a dark
one, so the chart needs no knowledge of the active theme.
"""

AXIS_TEXT = "#9aa3af"
"""Axis and label ink: muted, never a series colour.

Labels and values wear text ink; the coloured line beside a label is what
carries identity.
"""

MAX_POINTS = 500
"""Most points kept per series after thinning."""

LINE_WIDTH = 2


def _thin(values, stride):
    """Keep every ``stride``-th value, always including the most recent one.

    The last sample matters more than the grid being even: the right-hand
    end of the line is where the eye goes, and it carries the direct label.
    """
    if stride <= 1:
        return list(values)
    kept = values[::stride]
    if values and kept and kept[-1] is not values[-1]:
        kept = list(kept) + [values[-1]]
    return list(kept)


def line_figure(history, channels, labels, unit, *, y_zero_line=False):
    """Build one live line chart.

    Parameters
    ----------
    history:
        A ``buffer.ChannelHistory``.
    channels:
        Channel keys to draw, in order, for example ``("ax", "ay", "az")``.
    labels:
        Human names for those channels, same order, used in the legend and
        the end-of-line labels.
    unit:
        Axis title, for example ``"g"``.
    y_zero_line:
        Draw a reference line at zero. On for the gyroscope, where zero
        means "not rotating" and is worth seeing; off for the
        accelerometer, where the resting value is 1 g, not 0, and a line
        at zero would suggest a baseline that is not there.
    """
    timestamps = list(history.timestamps)
    if not timestamps:
        return empty_figure("Waiting for sensor data")

    latest = timestamps[-1]
    stride = max(1, len(timestamps) // MAX_POINTS)
    x = [t - latest for t in _thin(timestamps, stride)]

    traces = []
    annotations = []
    for index, (channel, label) in enumerate(zip(channels, labels)):
        y = _thin(list(history[channel]), stride)
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": label,
                "x": x,
                "y": y,
                "line": {"color": color, "width": LINE_WIDTH},
                "hovertemplate": f"{label}: %{{y:.2f}} {unit}<extra></extra>",
            }
        )
        if y:
            # Direct label at the line's right-hand end, in text ink. With
            # three lines a reader should not have to go to the legend and
            # back to tell which is which.
            annotations.append(
                {
                    "x": x[-1],
                    "y": y[-1],
                    "text": f" {label}",
                    "showarrow": False,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "font": {"color": AXIS_TEXT, "size": 11},
                }
            )

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 46, "r": 52, "t": 8, "b": 28},
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1.0,
            "font": {"color": AXIS_TEXT, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "annotations": annotations,
        # Crosshair plus one tooltip listing all three series at that
        # instant, which is the question a reader actually has: what were
        # all the axes doing at this moment?
        "hovermode": "x unified",
        "xaxis": {
            "title": {"text": "seconds ago", "font": {"color": AXIS_TEXT, "size": 11}},
            "gridcolor": GRID_COLOR,
            "zeroline": False,
            "linecolor": GRID_COLOR,
            "tickfont": {"color": AXIS_TEXT, "size": 10},
            "fixedrange": True,
        },
        "yaxis": {
            "title": {"text": unit, "font": {"color": AXIS_TEXT, "size": 11}},
            "gridcolor": GRID_COLOR,
            "zeroline": y_zero_line,
            "zerolinecolor": GRID_COLOR,
            "linecolor": GRID_COLOR,
            "tickfont": {"color": AXIS_TEXT, "size": 10},
            "fixedrange": True,
        },
        # Keeps the browser from resetting the view on every redraw.
        "uirevision": "live",
    }
    return {"data": traces, "layout": layout}


def probability_figure(probabilities, classes):
    """Horizontal bars, one per activity class.

    Class order is fixed - the model's own order, reversed so the first
    class sits at the top - and never sorted by probability. Sorting by
    value makes bars swap places every two seconds, so reading the chart
    means re-finding the row you were looking at. A fixed order lets the
    eye stay on one row and watch it move.

    Each bar carries its own percentage as a label, which is what the
    palette's contrast rule asks for and also what makes the difference
    between 0.48 and 0.52 legible at all.
    """
    if not probabilities or not classes:
        return empty_figure("Waiting for a prediction")

    ordered = [c for c in classes if c in probabilities]
    if not ordered:
        return empty_figure("Waiting for a prediction")

    # Plotly draws a horizontal bar chart bottom-up, so reversing puts the
    # model's first class at the top of the panel.
    rows = list(reversed(ordered))
    values = [probabilities[c] * 100 for c in rows]
    colors = [class_color(c, classes) for c in rows]

    return {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": values,
                "y": [c.replace("_", " ") for c in rows],
                "marker": {"color": colors},
                "text": [f"{v:.0f}%" for v in values],
                "textposition": "outside",
                "textfont": {"color": AXIS_TEXT, "size": 11},
                "cliponaxis": False,
                "hovertemplate": "%{y}: %{x:.1f}%<extra></extra>",
                "width": 0.62,
            }
        ],
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 86, "r": 44, "t": 4, "b": 20},
            "showlegend": False,
            "bargap": 0.3,
            "xaxis": {
                # Fixed 0-100. An auto-scaled axis would make a 3% bar
                # look like a confident answer whenever nothing else
                # scored.
                "range": [0, 100],
                "gridcolor": GRID_COLOR,
                "zeroline": False,
                "tickfont": {"color": AXIS_TEXT, "size": 10},
                "ticksuffix": "%",
                "fixedrange": True,
            },
            "yaxis": {
                "tickfont": {"color": AXIS_TEXT, "size": 11},
                "gridcolor": "rgba(0,0,0,0)",
                "zeroline": False,
                "fixedrange": True,
            },
            "uirevision": "probabilities",
        },
    }


def empty_figure(message="No data yet"):
    """A blank, theme-neutral plot carrying one muted message.

    Plotly's default template paints a white canvas with visible axes,
    which fights the dark theme and makes a dashboard that is merely
    waiting look broken instead.
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
                    "font": {"color": AXIS_TEXT, "size": 13},
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# Self-test: python figures.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import time

    try:
        from buffer import SampleBuffer
        from config import SENSOR_COLUMNS
        from datasource import SimulatedSource
    except ImportError:  # pragma: no cover
        from dashboard.buffer import SampleBuffer
        from dashboard.config import SENSOR_COLUMNS
        from dashboard.datasource import SimulatedSource

    print("figures.py self-test")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    buf = SampleBuffer()
    src = SimulatedSource(activity="walking", seed=5)
    src.start()
    elapsed = 0.0
    while elapsed < 12.0:
        time.sleep(0.05)
        elapsed += 0.05
        buf.extend(s.as_dict() for s in src.read())
    src.stop()

    history = buf.get_recent_seconds(10.0)
    fig = line_figure(history, ("ax", "ay", "az"), ("X", "Y", "Z"), "g")

    check("three series, one per axis", len(fig["data"]) == 3, f"{len(fig['data'])}")

    check(
        "series use the validated palette, in order",
        [t["line"]["color"] for t in fig["data"]] == list(SERIES_COLORS),
    )

    counts = [len(t["x"]) for t in fig["data"]]
    check(
        f"thinned to at most {MAX_POINTS + 1} points per series",
        all(c <= MAX_POINTS + 1 for c in counts) and len(set(counts)) == 1,
        f"{counts[0]} points each",
    )

    xs = fig["data"][0]["x"]
    check(
        "time runs backwards to zero at the right edge",
        xs[-1] == 0 and xs[0] < 0 and all(a <= b for a, b in zip(xs, xs[1:])),
        f"{xs[0]:.1f} s .. {xs[-1]:.1f} s",
    )

    check(
        "the newest sample survives thinning",
        abs(fig["data"][0]["y"][-1] - history["ax"][-1]) < 1e-9,
    )

    check(
        "a legend is shown and every line is labelled at its end",
        fig["layout"]["showlegend"] is True
        and len(fig["layout"]["annotations"]) == 3,
    )

    check(
        "labels wear text ink, never a series colour",
        all(
            a["font"]["color"] == AXIS_TEXT for a in fig["layout"]["annotations"]
        )
        and fig["layout"]["legend"]["font"]["color"] == AXIS_TEXT,
    )

    check(
        "one y-axis only - no second scale",
        not any(k.startswith("yaxis2") for k in fig["layout"]),
    )

    check(
        "backgrounds are transparent so the card theme shows through",
        fig["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
        and fig["layout"]["plot_bgcolor"] == "rgba(0,0,0,0)",
    )

    check(
        "hover gives a crosshair reading of all series at one instant",
        fig["layout"]["hovermode"] == "x unified",
    )

    gyro = line_figure(history, ("gx", "gy", "gz"), ("X", "Y", "Z"), "deg/s", y_zero_line=True)
    check(
        "the gyroscope chart draws a zero reference, the accelerometer does not",
        gyro["layout"]["yaxis"]["zeroline"] is True
        and fig["layout"]["yaxis"]["zeroline"] is False,
    )

    empty = SampleBuffer().get_recent_seconds(10.0)
    blank = line_figure(empty, ("ax", "ay", "az"), ("X", "Y", "Z"), "g")
    check(
        "an empty buffer gives the waiting message, not an empty chart frame",
        blank["data"] == [] and "Waiting" in blank["layout"]["annotations"][0]["text"],
    )

    check(
        "every channel the charts ask for exists in the buffer",
        all(c in history.channels for c in SENSOR_COLUMNS),
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
