"""The live state the whole page shares.

PR 5 of 14: the first PR where the dashboard actually moves.

What this file is for
---------------------
Three modules already exist and none of them knows about the others:

* ``datasource.py`` produces samples
* ``buffer.py`` stores them and cuts them into analysis windows
* ``falldetection.py`` watches them for a fall

Something has to own one instance of each and pump samples from the first
into the other two. That is this file, and that is all it does. It holds no
Dash components and imports no Dash code, so it can be driven from a plain
Python prompt - which is how the self-test at the bottom works.

Why a single shared object
--------------------------
A Dash app answers each browser request in whatever worker thread is free.
If every callback built its own source and buffer, each panel would show a
different slice of a different stream. So there is exactly one
:class:`LiveState`, created once when ``app.py`` is imported, and every
callback reads from it.

Why ``poll()`` is safe to call from anywhere
--------------------------------------------
Every callback calls ``LIVE.poll()`` before reading. That looks wasteful,
but it is the simplest correct arrangement: ``poll()`` moves whatever has
arrived into the shared buffer under a lock, so two callbacks polling in
the same tick between them collect every sample exactly once, in order,
and neither can see a half-written buffer. The alternative - one designated
pumping callback - would need Dash to guarantee callback ordering, which it
does not.

What this file deliberately does not do
---------------------------------------
No feature extraction, no model, no training. Window objects are handed
out by the buffer and the classifier work starts from them in PR 10.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

try:  # running from inside dashboard/, as app.py does
    from buffer import SampleBuffer
    from config import SAMPLING_FREQ_HZ
    from datasource import SensorSource, SimulatedSource
    from falldetection import FallDetector
    from inference import ActivityClassifier
    from pedometer import StepCounter
    from vitals import HeartRateEstimator
except ImportError:  # pragma: no cover - imported as a package instead
    from dashboard.buffer import SampleBuffer
    from dashboard.config import SAMPLING_FREQ_HZ
    from dashboard.datasource import SensorSource, SimulatedSource
    from dashboard.falldetection import FallDetector
    from dashboard.inference import ActivityClassifier
    from dashboard.pedometer import StepCounter
    from dashboard.vitals import HeartRateEstimator


STALE_AFTER_S = 0.5
"""No samples for this long and the link is reported as stalled.

At 100 Hz, half a second is 50 missing samples - far past a scheduling
hiccup, and still quick enough that someone unplugging the device sees the
status change while they are still looking at the screen.
"""

HEART_WINDOW_S = 8.0
"""How much PPG history the heart rate estimate looks at.

Long enough for a 40 BPM pulse to repeat five times, short enough that a
real change in heart rate shows up while someone is still watching.
"""

HEART_UPDATE_INTERVAL_S = 1.0
"""How often the heart rate is re-estimated.

Not a performance limit - the estimate is cheap. The smoothing inside
``vitals.HeartRateEstimator`` is tuned for roughly this spacing, so
feeding it ten times a second would make the readout much laggier than
its smoothing constant suggests.
"""

PREDICTION_HISTORY = 2000
"""How many predictions to remember for the timeline.

One window every two seconds, so this is over an hour of continuous use.
Cheap enough to keep every one rather than thinning and having the
timeline disagree with what was actually predicted.
"""

PLACEHOLDER = "--"
"""Shown for any value that is genuinely not known yet.

Never a plausible-looking stand-in. A sampling rate of "100 Hz" displayed
because nothing has been measured yet is worse than useless: it is the
number someone would quote in the report.
"""


def default_source() -> SensorSource:
    """The simulator, unless a serial port is named in the environment.

        HAR_SERIAL_PORT=COM7 python app.py

    An environment variable rather than a control on the page, because
    which device the dashboard is reading is not something a viewer should
    be able to change by clicking, and because the choice has to be made
    before the first sample arrives.

    If the port will not open, this falls back to the simulator and leaves
    the reason in the returned object's ``last_error``. Refusing to start
    would leave someone with a dead page and a traceback in a terminal
    they may not be looking at; coming up in simulation, clearly labelled
    "simulated" in the status strip, at least shows them a working
    dashboard while they sort the port out.
    """
    port = os.environ.get("HAR_SERIAL_PORT", "").strip()
    if not port:
        return SimulatedSource()

    try:
        from serialsource import SerialSource
    except ImportError:  # pragma: no cover
        from dashboard.serialsource import SerialSource

    try:
        source = SerialSource(port)
        source.start()
        return source
    except Exception as exc:  # noqa: BLE001
        print(f"[live] could not open {port}: {exc}; falling back to the simulator")
        return SimulatedSource()


def _describe_age(seconds: float) -> str:
    """How long ago, in words, for the alert line.

    Seconds while that is still meaningful, then minutes. A banner saying
    "412 s ago" makes the reader do arithmetic to learn that the fall was
    a while back.
    """
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.0f} h"


class LiveState:
    """One sensor source, one buffer, one fall detector, and a lock.

    Parameters
    ----------
    source:
        Any :class:`datasource.SensorSource`. Defaults to the simulator, so
        the dashboard runs with no hardware attached. PR 13 passes a serial
        source here instead and nothing else in this file changes.
    """

    def __init__(self, source: SensorSource | None = None) -> None:
        self.source: SensorSource = source if source is not None else default_source()
        self.buffer = SampleBuffer()
        self.detector = FallDetector()
        self.heart = HeartRateEstimator(SAMPLING_FREQ_HZ)
        self.steps = StepCounter(SAMPLING_FREQ_HZ)
        self.classifier = ActivityClassifier()

        # Every prediction made, newest last, for the timeline strip. One
        # window every two seconds, so an hour of running costs 1800
        # entries - small enough to keep in full rather than sampled.
        self.predictions: deque = deque(maxlen=PREDICTION_HISTORY)

        self._lock = threading.RLock()
        self._running = False
        self._last_sample_at: float | None = None
        self._last_error: str | None = None
        self._polls = 0
        self._last_heart_update: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the source. Safe to call more than once."""
        with self._lock:
            if not self._running:
                self.source.start()
                self._running = True

    def stop(self) -> None:
        """Stop the source. Safe to call more than once."""
        with self._lock:
            if self._running:
                self.source.stop()
                self._running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # -- the pump ----------------------------------------------------------

    def poll(self) -> int:
        """Move everything that has arrived into the buffer and detector.

        Returns the number of samples stored. Starts the source on first
        call, so a callback never has to remember to.

        A source that raises is caught rather than allowed to kill the
        callback: the dashboard then shows the link as failed, which is the
        behaviour you want when someone pulls the USB cable mid-demo. The
        message is kept in ``last_error`` for the status strip.
        """
        with self._lock:
            if not self._running:
                self.start()

            self._polls += 1
            try:
                samples = self.source.read()
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                self._last_error = f"{type(exc).__name__}: {exc}"
                return 0

            if not samples:
                return 0

            readings = [s.as_dict() for s in samples]
            stored = self.buffer.extend(readings)
            self.detector.update_batch(readings)
            self.steps.update_batch(readings)
            self.classifier.note_ir(r["IR_Value"] for r in readings)

            # Classify every window the buffer has finished, inside the
            # same lock. Done in a callback instead, two callbacks in one
            # tick would each drain part of the queue and each would see
            # only half the predictions.
            for window in self.buffer.drain_windows():
                prediction = self.classifier.predict(window)
                if prediction is not None:
                    self.predictions.append((time.time(), prediction))

            self._last_sample_at = time.monotonic()
            self._last_error = None
            return stored

    # -- what the panels ask about -----------------------------------------

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def source_kind(self) -> str:
        """``"simulated"`` or ``"device"``, for the status strip."""
        return "simulated" if isinstance(self.source, SimulatedSource) else "device"

    @property
    def seconds_since_sample(self) -> float | None:
        with self._lock:
            if self._last_sample_at is None:
                return None
            return time.monotonic() - self._last_sample_at

    def connection_label(self) -> str:
        """One short phrase for the connection readout.

        Four states worth telling apart, because they need different
        reactions from whoever is standing in front of the screen:
        something is broken, nothing has arrived yet, data is flowing, or
        it was flowing and stopped.
        """
        with self._lock:
            if self._last_error is not None:
                return "error"
            if self._last_sample_at is None:
                return "waiting"
            if time.monotonic() - self._last_sample_at <= STALE_AFTER_S:
                return f"live ({self.source_kind})"
            return "stalled"

    def sample_rate_label(self) -> str:
        """The *measured* rate, or ``--``.

        This is the measured value, never the configured one. If the
        firmware is actually sending 50 Hz while ``config.py`` says 100,
        this readout is the only place on the page that will say so.
        """
        rate = self.buffer.measured_sampling_rate()
        if rate is None:
            return PLACEHOLDER
        return f"{rate:.0f}"

    def heart_rate_label(self) -> str:
        """Heart rate in BPM, or ``--``.

        Re-estimated about once a second rather than on every tick. Not for
        speed - the estimate is cheap - but because the smoothing inside
        the estimator is tuned for roughly that spacing. Feeding it ten
        times a second would make the readout far laggier than it looks.

        **Not a medical measurement.** See ``vitals.py``.
        """
        now = time.monotonic()
        with self._lock:
            due = now - self._last_heart_update >= HEART_UPDATE_INTERVAL_S
            if due:
                self._last_heart_update = now

        if due:
            history = self.buffer.get_recent_seconds(HEART_WINDOW_S)
            if not history.is_empty:
                self.heart.estimate(history["IR_Value"])

        bpm = self.heart.bpm
        return PLACEHOLDER if bpm is None else f"{bpm:.0f}"

    def sensor_labels(self) -> tuple[str, str, str, str]:
        """The four sensor readouts: accelerometer, gyroscope, IR, steps.

        All four come from one call so the panel can never show three
        values from one instant and a fourth from the next.

        Returned as formatted strings rather than numbers because the
        formatting is part of the reading: two decimals on accelerometer
        in g, none on gyroscope in deg/s, and a thousands separator on the
        raw IR count, which runs to six figures and is unreadable without
        one.
        """
        latest = self.buffer.latest_sample()
        if latest is None:
            return (PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, str(self.steps.steps))

        accel = "  ".join(f"{latest[axis]:+.2f}" for axis in ("ax", "ay", "az"))
        gyro = "  ".join(f"{latest[axis]:+.0f}" for axis in ("gx", "gy", "gz"))
        ir = f"{latest['IR_Value']:,.0f}"
        return accel, gyro, ir, str(self.steps.steps)

    def activity_labels(self) -> tuple[str, str, str]:
        """Current activity, confidence, and the gesture readout.

        Returned together so the header cannot show an activity from one
        window beside a confidence from another - which would be invisible
        on screen and wrong in a screenshot.

        When no model could be loaded the activity reads "unavailable"
        and the connection to why is in ``classifier.unavailable_reason``.
        Showing the last activity it managed before the model broke would
        be worse than showing nothing.
        """
        gesture = "not implemented"

        if not self.classifier.ready:
            return "unavailable", PLACEHOLDER, gesture

        with self._lock:
            latest = self.predictions[-1][1] if self.predictions else None

        if latest is None:
            return "waiting", PLACEHOLDER, gesture

        label = latest.label.replace("_", " ")
        if latest.confidence != latest.confidence:  # NaN: no probabilities
            return label, PLACEHOLDER, gesture
        return label, str(latest.percent), gesture

    def prediction_history(self) -> list:
        """A snapshot copy of every remembered prediction, oldest first.

        A copy, because the caller iterates it while ``poll()`` may be
        appending from another Dash worker thread, and a deque mutated
        mid-iteration raises.
        """
        with self._lock:
            return list(self.predictions)

    def model_label(self) -> str:
        """Which model is actually running, for the footer.

        Named on screen because the dashboard falls back when one will not
        load. A demonstration that silently runs the decision tree while
        the report describes the random forest is a problem nobody catches
        until the viva.
        """
        if not self.classifier.ready:
            return f"no model: {self.classifier.unavailable_reason}"
        return f"model: {self.classifier.name}"

    def fall_state(self) -> tuple[bool, str]:
        """Whether the alert is up, and one line describing the event.

        The detail line exists because "Fall detected" alone invites the
        question "says what?" - during a demonstration and afterwards in
        the report. The event carries the numbers that triggered it, so
        they are put on screen: when, how hard, how long the free fall
        lasted.
        """
        active = self.detector.alert_active
        event = self.detector.last_event
        if not active or event is None:
            return active, ""

        # The event's own summary carries the stream's monotonic
        # timestamps, which run to six figures and mean nothing to a
        # reader. Rebuild the line from the same fields in terms a person
        # watching the screen can check: how long ago, and how hard.
        parts = [f"{event.peak_impact_g:.1f} g impact"]
        parts.append(f"{event.free_fall_duration_s * 1000:.0f} ms free fall")

        latest = self.buffer.latest_timestamp
        if latest is not None:
            ago = latest - event.detected_at
            if ago >= 0:
                parts.append(f"{_describe_age(ago)} ago")

        return True, ", ".join(parts)

    def dismiss_fall(self) -> None:
        """Lower the alert. The event history is kept."""
        self.detector.dismiss()

    # -- demo controls (simulator only) ------------------------------------

    def set_demo_activity(self, name: str) -> bool:
        """Ask a simulated source to play ``name``. False if not simulated.

        Guarded rather than assumed: with a real device attached this does
        nothing and says so, instead of raising in a callback halfway
        through a demonstration.
        """
        setter = getattr(self.source, "set_activity", None)
        if setter is None:
            return False
        setter(name)
        return True

    def trigger_demo_fall(self) -> bool:
        """Ask a simulated source to play a fall. False if not simulated."""
        trigger = getattr(self.source, "trigger_fall", None)
        if trigger is None:
            return False
        trigger()
        return True

    def battery_label(self) -> str:
        """Always ``not reported`` - the device does not send a battery level.

        There is no battery telemetry anywhere in this project: the wrist
        unit runs off an HT7833 regulator with no voltage divider into an
        ADC pin, and the firmware directory is empty, so nothing is
        measuring the cell.

        The readout is kept on the page, showing the truth, rather than
        removed. A missing battery panel looks like a dashboard that forgot
        about battery; a panel saying "not reported" is a standing question
        for whoever owns the firmware, visible every time anyone opens the
        page. Inventing a percentage here would be the one genuinely
        dishonest thing this dashboard could do, because it is the one
        number nobody could check by looking at the person wearing it.
        """
        return "not reported"

    def status(self) -> dict:
        """Everything above in one dict, for the self-test and for debugging."""
        with self._lock:
            return {
                "running": self._running,
                "source": self.source_kind,
                "connection": self.connection_label(),
                "sample_rate": self.sample_rate_label(),
                "buffered": len(self.buffer),
                "polls": self._polls,
                "last_error": self._last_error,
                "fall_alert": self.detector.alert_active,
            }


LIVE = LiveState()
"""The one instance. ``app.py`` imports this; callbacks use it directly."""


# ---------------------------------------------------------------------------
# Self-test: python live.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    print("live.py self-test")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    state = LiveState()

    check(
        "nothing is known before the first poll",
        state.connection_label() == "waiting"
        and state.sample_rate_label() == PLACEHOLDER,
        f"connection={state.connection_label()!r} rate={state.sample_rate_label()!r}",
    )

    state.poll()
    time.sleep(0.25)
    stored = state.poll()
    check(
        "poll() starts the source and stores samples",
        state.is_running and stored > 0,
        f"{stored} samples",
    )

    check(
        "the link reads as live",
        state.connection_label().startswith("live"),
        state.connection_label(),
    )

    # Let enough samples through for a rate measurement to settle.
    for _ in range(12):
        time.sleep(0.05)
        state.poll()
    rate = state.sample_rate_label()
    check(
        "measured rate is near 100 Hz",
        rate != PLACEHOLDER and 90 <= float(rate) <= 110,
        f"{rate} Hz",
    )

    # Two threads polling at once must not lose or duplicate a sample.
    before = state.buffer.stats.samples_received
    errors = []

    def hammer():
        try:
            for _ in range(40):
                state.poll()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    for t in threads:
        t.join()
    state.poll()
    gained = state.buffer.stats.samples_received - before
    check(
        "four threads polling together: no error, no out-of-order sample",
        not errors and state.buffer.stats.samples_out_of_order == 0,
        f"{gained} samples gained, {len(errors)} errors",
    )

    # A source that breaks must not take the callback down with it.
    class BrokenSource(SimulatedSource):
        def read(self):
            raise OSError("device disconnected")

    broken = LiveState(source=BrokenSource())
    broken.poll()
    check(
        "a failing source is reported, not raised",
        broken.connection_label() == "error"
        and "device disconnected" in (broken.last_error or ""),
        broken.connection_label(),
    )

    # A source that stops sending must stop reading as live.
    state._last_sample_at = time.monotonic() - (STALE_AFTER_S + 0.1)
    check(
        "a silent link reads as stalled",
        state.connection_label() == "stalled",
        state.connection_label(),
    )

    fresh = LiveState()
    check(
        "heart rate is unknown before any data arrives",
        fresh.heart_rate_label() == PLACEHOLDER,
        fresh.heart_rate_label(),
    )

    check(
        "battery is reported as unavailable, never as a number",
        LiveState().battery_label() == "not reported",
    )

    keys = set(LiveState().status())
    check(
        "status() carries what the panels need",
        {"running", "connection", "sample_rate", "buffered", "fall_alert"} <= keys,
        ", ".join(sorted(keys)),
    )

    state.stop()
    check("stop() stops the source", not state.is_running)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
