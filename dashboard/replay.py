"""Replaying the device's wire format, with no device attached.

PR 21 of 21.

Why this file exists
--------------------
``serialsource.py`` already self-tests: it proves that a line from the
sketch parses into a :class:`datasource.Sample`. What it does not prove is
that those samples then survive the rest of the dashboard - the buffer, the
43 features, the scaler, the model, the fall detector, the pedometer.

That seam is where a demonstration breaks. Everything upstream of it has
been exercised for weeks against the simulator; the serial path has been
exercised against a fake port in isolation. The join between them has never
run end to end, because running it needed the hardware.

So this module replays the wire format *through the whole stack*. It builds
lines in the sketch's exact ``printf`` format, hands them to a
:class:`serialsource.SerialSource` through a fake port, drives a real
:class:`live.LiveState`, and checks that predictions, steps and fall alerts
come out of the other end.

What makes it worth more than the simulator
--------------------------------------------
The simulator produces clean Python floats. The device cannot. Three things
happen to a reading on its way down the wire, and this module does all
three:

1. **The accelerometer clips at 2 g.** The sketch never writes register
   ``0x1C``, so the MPU6050 stays in its default range. A 3.9 g fall impact
   leaves the device as 2.00 g.
2. **The gyroscope clips at 250 deg/s**, where the simulator runs to 500.
3. **Every float is rounded to two decimals**, because the sketch prints
   ``%.2f``.

Which means this harness answers a question nothing else in the project
answers: *does fall detection still fire when the impact has been flattened
to 2 g?* It is the direct test of the 15 November threshold decision, and
it can be run today, without the wrist unit.

Running it::

    cd dashboard
    python replay.py

One private call, declared
--------------------------
``SimulatedSource._generate_sample()`` is called directly to produce samples
as fast as the CPU allows. The public ``read()`` returns only what is due by
the wall clock, so generating 25 seconds of data through it would take 25
seconds. The private method advances the simulator's own index and never
reads the clock, so the same signal comes out in a few milliseconds. The
simulator's lock is held around the loop, as that method requires.
"""

from __future__ import annotations

import math
import time

try:
    from datasource import SimulatedSource
    from serialsource import (
        ACCEL_FULL_SCALE_G,
        GYRO_FULL_SCALE_DPS,
        SerialSource,
    )
except ImportError:  # pragma: no cover - import plumbing
    from dashboard.datasource import SimulatedSource
    from dashboard.serialsource import (
        ACCEL_FULL_SCALE_G,
        GYRO_FULL_SCALE_DPS,
        SerialSource,
    )

__all__ = [
    "encode",
    "generate_lines",
    "ReplayPort",
    "replay_source",
]


def _clip(value: float, limit: float) -> float:
    """What the sensor can actually report, rather than what happened."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def encode(sample, counter: int) -> str:
    """One :class:`datasource.Sample` as the sketch would print it.

    The format is copied from the collector sketch::

        ESP_BT.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lu\\n",
                      sampleCounter++, ax, ay, az, gx, gy, gz, irValue);

    so eight fields: the counter, six motion channels at two decimals, and
    the IR count as an unsigned integer. Clipping happens here rather than
    in the generator because it is a property of the *wire*, not of the
    movement: the arm really did accelerate at 3.9 g, and the device really
    did report 2.00.
    """
    ax = _clip(sample.ax, ACCEL_FULL_SCALE_G)
    ay = _clip(sample.ay, ACCEL_FULL_SCALE_G)
    az = _clip(sample.az, ACCEL_FULL_SCALE_G)
    gx = _clip(sample.gx, GYRO_FULL_SCALE_DPS)
    gy = _clip(sample.gy, GYRO_FULL_SCALE_DPS)
    gz = _clip(sample.gz, GYRO_FULL_SCALE_DPS)
    ir = max(0, int(round(sample.IR_Value)))
    return (
        f"{counter},{ax:.2f},{ay:.2f},{az:.2f},"
        f"{gx:.2f},{gy:.2f},{gz:.2f},{ir}\n"
    )


def generate_lines(
    activity: str = "walking",
    seconds: float = 25.0,
    *,
    fall_at: float | None = None,
    start_counter: int = 0,
    drop_at: float | None = None,
    drop_count: int = 12,
    sampling_freq_hz: int = 100,
) -> list[str]:
    """Build a run of wire-format lines for one activity.

    Parameters
    ----------
    activity:
        Any name the simulator knows: walking, sitting, standing, laying,
        upstairs, downstairs.
    fall_at:
        Seconds into the run at which to inject a scripted fall.
    drop_at:
        Seconds into the run at which to skip ``drop_count`` counter
        values, as a dropped Bluetooth packet would. The lines are missing
        but the counter still advances, which is exactly what the device
        does and what ``SerialSource`` detects.
    """
    source = SimulatedSource()
    source.set_activity(activity)
    source.start()

    total = int(seconds * sampling_freq_hz)
    fall_index = None if fall_at is None else int(fall_at * sampling_freq_hz)
    drop_index = None if drop_at is None else int(drop_at * sampling_freq_hz)

    lines: list[str] = []
    counter = start_counter
    skipped = 0

    try:
        with source._lock:  # the generator below requires the lock held
            for i in range(total):
                if fall_index is not None and i == fall_index:
                    source._fall_elapsed = 0.0

                sample = source._generate_sample()

                if (
                    drop_index is not None
                    and drop_index <= i < drop_index + drop_count
                ):
                    # The counter advances; the line never arrives.
                    counter += 1
                    skipped += 1
                    continue

                lines.append(encode(sample, counter))
                counter += 1
    finally:
        source.stop()

    return lines


class ReplayPort:
    """A stand-in for a pyserial port that reads back a list of lines.

    ``SerialSource`` only ever calls ``readline()`` and ``close()``, so
    those are the only two methods a fake port needs. Once the lines run
    out it returns empty bytes after a short pause, the way a real port
    does when the device has gone quiet - never raising, so the reader
    thread stays alive and the dashboard shows "stalled" rather than
    "error".
    """

    def __init__(self, lines) -> None:
        self._lines = list(lines)
        self.closed = False
        self.served = 0

    def readline(self) -> bytes:
        if self._lines:
            self.served += 1
            return self._lines.pop(0).encode("ascii")
        time.sleep(0.005)
        return b""

    def close(self) -> None:
        self.closed = True

    @property
    def exhausted(self) -> bool:
        return not self._lines


def replay_source(lines, port: str = "REPLAY") -> SerialSource:
    """A :class:`serialsource.SerialSource` fed by ``lines``.

    Drop-in for the real thing: ``LiveState(replay_source(...))`` drives
    the entire dashboard off recorded wire traffic.
    """
    return SerialSource(port, serial_factory=lambda: ReplayPort(lines))


# ---------------------------------------------------------------------------
# Self-test: python replay.py
# ---------------------------------------------------------------------------

def _drain(live, timeout_s: float = 20.0) -> int:
    """Poll until the replay stops producing, the way the page would."""
    stored = 0
    idle = 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        got = live.poll()
        stored += got
        idle = idle + 1 if got == 0 else 0
        if idle >= 25 and stored > 0:
            break
        time.sleep(0.01)
    return stored


def _selftest() -> int:
    try:
        from live import LiveState
        from buffer import SampleBuffer
        from config import STEP_SIZE, WINDOW_SIZE
    except ImportError:  # pragma: no cover
        from dashboard.live import LiveState
        from dashboard.buffer import SampleBuffer
        from dashboard.config import STEP_SIZE, WINDOW_SIZE

    print("replay.py self-test")
    print("  (no hardware needed: the sketch's wire format, through the")
    print("   whole dashboard - buffer, model, fall detector, pedometer)")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    # -- the wire format ----------------------------------------------------
    lines = generate_lines("walking", seconds=2.0)
    first = lines[0].strip().split(",")
    check(
        "a generated line has the sketch's eight fields",
        len(first) == 8,
        lines[0].strip(),
    )
    check(
        "motion channels carry two decimals, as %.2f prints them",
        all(len(f.split(".")[-1]) == 2 for f in first[1:7]),
    )
    check(
        "the IR field is an integer, as %lu prints it",
        first[7].isdigit(),
        f"IR={first[7]}",
    )

    # -- the hardware ceilings ----------------------------------------------
    fall_lines = generate_lines("walking", seconds=6.0, fall_at=3.0)
    peak = max(
        max(abs(float(p)) for p in ln.split(",")[1:4]) for ln in fall_lines
    )
    check(
        "a fall impact is clipped to the hardware's 2 g ceiling",
        abs(peak - ACCEL_FULL_SCALE_G) < 1e-9,
        f"peak {peak:.2f} g (the simulator alone reaches ~4 g)",
    )
    gyro_peak = max(
        max(abs(float(p)) for p in ln.split(",")[4:7]) for ln in fall_lines
    )
    check(
        "the gyroscope stays inside its 250 deg/s range",
        gyro_peak <= GYRO_FULL_SCALE_DPS + 1e-9,
        f"peak {gyro_peak:.1f} deg/s",
    )

    # The clipping is PER AXIS. The detector works on the vector magnitude,
    # which is therefore not capped at 2 g at all - see the note printed at
    # the end of this self-test.
    svm_peak = max(
        math.sqrt(sum(float(p) ** 2 for p in ln.split(",")[1:4]))
        for ln in fall_lines
    )
    check(
        "the magnitude the detector sees exceeds the per-axis ceiling",
        svm_peak > ACCEL_FULL_SCALE_G,
        f"SVM peak {svm_peak:.2f} g from axes each capped at "
        f"{ACCEL_FULL_SCALE_G:.1f} g",
    )

    # -- the whole stack, on device-grade data ------------------------------
    walk = generate_lines("walking", seconds=22.0, fall_at=16.0)
    live = LiveState(replay_source(walk))

    # Give this one a bigger buffer than the dashboard uses, before it
    # starts.
    #
    # A replay port hands over its whole stream as fast as readline() is
    # called, so 22 seconds of samples land in a fraction of a second. The
    # real buffer holds 10 seconds, so it correctly throws away the oldest
    # windows before the consumer can classify them - between three and
    # five of the ten, depending on how the reader thread happens to be
    # scheduled. That is the ring buffer working as designed, not a fault,
    # but it made this check assert against a number that moved from run
    # to run and fail roughly one time in three.
    #
    # Holding the whole replay means the count is determined by the
    # pipeline rather than by thread timing. What the real buffer does
    # under overflow is covered by buffer.py's own tests.
    live.buffer = SampleBuffer(capacity=len(walk) + WINDOW_SIZE)
    live.start()
    stored = _drain(live)

    check(
        "the dashboard reads the replay as a device, not a simulator",
        live.source_kind == "device",
        f"source_kind={live.source_kind!r}",
    )
    check(
        "every line reached the buffer",
        stored >= len(walk) - 5,
        f"{stored} of {len(walk)} samples stored",
    )

    rate = live.buffer.measured_sampling_rate()
    check(
        "the measured rate comes out at 100 Hz",
        rate is not None and abs(rate - 100.0) < 1.0,
        f"{rate:.1f} Hz" if rate else "no rate",
    )

    history = live.prediction_history()
    expected_windows = 1 + (len(walk) - WINDOW_SIZE) // STEP_SIZE
    check(
        "the classifier produced predictions from wire data",
        len(history) == expected_windows,
        f"{len(history)} of {expected_windows} windows classified",
    )
    check(
        "no window was discarded, so the count is the pipeline's, not the clock's",
        live.buffer.stats.windows_skipped == 0,
        f"{live.buffer.stats.windows_skipped} skipped",
    )
    if history:
        labels = {p.label for _, p in history}
        known = set(live.classifier.classes)
        check(
            "every prediction is one of the six trained classes",
            labels <= known,
            ", ".join(sorted(labels)),
        )

    steps = live.steps.steps
    check(
        "the pedometer counted steps while walking",
        steps > 0,
        f"{steps} steps in 22 s",
    )

    fired, detail = live.fall_state()
    check(
        "FALL DETECTION FIRES ON A 2 g-CLIPPED IMPACT",
        fired,
        detail or "no alert",
    )
    live.stop()

    # -- a dropped Bluetooth packet -----------------------------------------
    gappy = generate_lines("sitting", seconds=6.0, drop_at=3.0, drop_count=12)
    source = replay_source(gappy)
    live2 = LiveState(source)
    live2.start()
    _drain(live2, timeout_s=10.0)
    check(
        "a dropped packet is reported, not silently absorbed",
        source.dropped_samples >= 10,
        f"{source.dropped_samples} samples missing",
    )
    check(
        "the link stays up after the gap",
        source.last_error is None,
    )
    live2.stop()

    print()
    print("  A correction this harness turned up, worth reading:")
    print()
    print("  The MPU6050 clips EACH AXIS at 2 g, but the fall detector works")
    print("  on the vector magnitude, sqrt(ax^2 + ay^2 + az^2). Three axes at")
    print(f"  2 g give {math.sqrt(3) * ACCEL_FULL_SCALE_G:.2f} g, not 2 g. The impact above measured")
    print(f"  {svm_peak:.2f} g AFTER clipping, because it lands across two axes.")
    print()
    print("  So '2.5 g is unreachable on this hardware' is not right. What is")
    print("  true: the ceiling depends on how the fall is oriented - 2.00 g if")
    print("  the impact lands square on one axis, up to 3.46 g across three.")
    print("  A 2.5 g threshold would catch some falls and be impossible for")
    print("  others, which is still a good reason to lower it, but it is a")
    print("  different reason from the one written down on 15 November.")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
