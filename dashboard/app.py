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

from dash import Dash, Input, Output, callback

from config import APP_TITLE, HOST, PORT
from layout import Ids, build_layout
from live import LIVE

app = Dash(__name__, title=APP_TITLE)
app.layout = build_layout()

# Dash warns about callbacks targeting ids that do not exist. Every id in
# layout.Ids is present from PR 4 on, so nothing is suppressed here; if a
# later PR needs an id created at runtime, set suppress_callback_exceptions
# above rather than inventing hidden divs.


@callback(
    Output(Ids.CLOCK, "children"),
    Output(Ids.CONNECTION, "children"),
    Output(Ids.SAMPLE_RATE, "children"),
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
    """
    LIVE.poll()
    return (
        _wall_clock(),
        LIVE.connection_label(),
        LIVE.sample_rate_label(),
    )


def _wall_clock():
    """Local time as hh:mm:ss.

    Wall-clock time, not the sample timestamps: this readout answers "is
    this screen frozen?", which a monotonic clock cannot.
    """
    import time

    return time.strftime("%H:%M:%S")


if __name__ == "__main__":
    LIVE.start()
    app.run(debug=True, host=HOST, port=PORT)
