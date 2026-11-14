"""The strip of recent predictions.

PR 11 of 14.

A row of coloured blocks, one per classified window, oldest on the left.
It answers a question the big activity label cannot: *is the classifier
steady, or is it flickering between two answers?* A label reading
"walking" tells you nothing about whether it read "upstairs" half a second
ago. On a demonstration that distinction is most of what a viewer wants to
know.

Built as plain HTML rather than a chart. It is a row of blocks with no
axes and no scale; a plotting library would add a figure, a margin and a
redraw cost for something CSS does in one line. It also means each block
can carry a tooltip of its own.

Colour matches the probability bars exactly - the same class has the same
colour in both panels, assigned from the model's class order and never
from rank, so a class keeps its colour the whole session.

Colour is not the only encoding. Underneath the strip is a legend of the
classes actually present, and every block has a ``title`` giving its
activity, confidence and clock time on hover.
"""

from __future__ import annotations

import time

from dash import html

try:
    from figures import class_color
except ImportError:  # pragma: no cover
    from dashboard.figures import class_color

LOW_CONFIDENCE = 0.5
"""Below this the block is drawn faded.

A prediction the model is unsure of should not look identical to one it
is certain about. Fading is a second channel on top of colour, so the
doubt survives being printed in greyscale in the report.
"""


def build(predictions, classes, seconds=60.0):
    """Return the timeline's children for the given predictions.

    Parameters
    ----------
    predictions:
        Pairs of ``(wall_clock_time, Prediction)``, oldest first.
    classes:
        The model's class list, which fixes the colour assignment.
    seconds:
        How far back to show.
    """
    if not predictions:
        return html.Span(
            f"Last {seconds:.0f} s of predictions", className="muted"
        )

    cutoff = time.time() - seconds
    recent = [(t, p) for t, p in predictions if t >= cutoff]
    if not recent:
        recent = list(predictions)[-1:]

    blocks = []
    present = []
    for when, prediction in recent:
        label = prediction.label
        if label not in present:
            present.append(label)

        confident = prediction.confidence == prediction.confidence  # not NaN
        faded = confident and prediction.confidence < LOW_CONFIDENCE

        percent = f"{prediction.percent}%" if confident else "confidence unknown"
        clock = time.strftime("%H:%M:%S", time.localtime(when))

        blocks.append(
            html.Span(
                className="tl-block" + (" tl-faded" if faded else ""),
                title=f"{label.replace('_', ' ')} · {percent} · {clock}",
                style={"backgroundColor": class_color(label, classes)},
            )
        )

    # Legend of the classes that actually appear, in the model's order, so
    # identity never rests on colour alone.
    legend = [
        html.Span(
            className="tl-key",
            children=[
                html.Span(
                    className="tl-swatch",
                    style={"backgroundColor": class_color(name, classes)},
                ),
                html.Span(name.replace("_", " ")),
            ],
        )
        for name in classes
        if name in present
    ]

    return [
        html.Div(blocks, className="tl-track"),
        html.Div(legend, className="tl-legend"),
    ]


# ---------------------------------------------------------------------------
# Self-test: python timeline.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    from dataclasses import dataclass

    print("timeline.py self-test")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    @dataclass
    class FakePrediction:
        label: str
        confidence: float

        @property
        def percent(self):
            return round(self.confidence * 100)

    classes = ["downstairs", "laying", "sitting", "standing", "upstairs", "walking"]

    empty = build([], classes)
    check(
        "no predictions gives a caption, not an empty bar",
        hasattr(empty, "className") and empty.className == "muted",
    )

    now = time.time()
    history = [
        (now - 30, FakePrediction("sitting", 0.91)),
        (now - 20, FakePrediction("walking", 0.44)),
        (now - 10, FakePrediction("walking", 0.88)),
        (now - 2, FakePrediction("standing", 0.77)),
    ]
    track, legend = build(history, classes, seconds=60)

    check("one block per prediction", len(track.children) == 4, f"{len(track.children)}")

    check(
        "each class keeps its own colour",
        track.children[1].style["backgroundColor"]
        == track.children[2].style["backgroundColor"]
        and track.children[0].style["backgroundColor"]
        != track.children[1].style["backgroundColor"],
    )

    check(
        "a low-confidence prediction is faded as well as coloured",
        "tl-faded" in track.children[1].className
        and "tl-faded" not in track.children[2].className,
    )

    check(
        "every block names its activity, confidence and time on hover",
        all("%" in b.title and ":" in b.title for b in track.children),
        track.children[0].title,
    )

    check(
        "the legend lists only the classes present, in the model's order",
        [k.children[1].children for k in legend.children]
        == ["sitting", "standing", "walking"],
        str([k.children[1].children for k in legend.children]),
    )

    old = [(now - 600, FakePrediction("laying", 0.99))]
    track_old, _ = build(old, classes, seconds=60)
    check(
        "everything older than the window still shows the most recent one",
        len(track_old.children) == 1,
    )

    colors = {
        c: class_color(c, classes) for c in classes
    }
    check(
        "all six classes get distinct colours",
        len(set(colors.values())) == 6,
        ", ".join(f"{c}={v}" for c, v in list(colors.items())[:3]) + ", ...",
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
