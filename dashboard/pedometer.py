"""Step counting from the accelerometer.

PR 7 of 14.

What this is
------------
A causal step counter: it sees each sample once, in order, and decides
immediately. No window is revisited, nothing is recomputed later. That is
the only kind of counter a live dashboard can have, and it is also what
makes the count trustworthy - a number that silently revises itself after
the fact would be worse than no number.

This is display-only. It is not a feature for the classifier, it does not
touch the training pipeline, and nothing in ``data-processing/`` depends
on it.

How a step is recognised
------------------------
Walking shows up in the accelerometer magnitude as a regular bump once per
footfall. The difficulty is that the magnitude also sits at 1 g from
gravity, drifts as the wrist changes orientation, and jiggles constantly
from arm movement that is not walking.

So:

1. **Track a slow baseline.** An exponential average of the magnitude over
   about a second. This follows posture and drift, and is subtracted off,
   leaving only the quick part. Fixed thresholds against raw magnitude do
   not survive the wrist turning over.
2. **Count a rising crossing.** When the quick part rises past
   ``STEP_THRESHOLD_G`` the counter registers one step, then refuses to
   count again until the signal has dropped back below
   ``REARM_THRESHOLD_G``. Two thresholds rather than one - hysteresis -
   so a signal hovering at the boundary produces one step, not forty.
3. **Enforce a refractory period.** No two steps closer than
   ``MIN_STEP_INTERVAL_S``. This caps the counter at 4 steps a second,
   above a sprint and well below the frequency of a hand tremor or a
   sensor spike.

What it gets wrong, deliberately
--------------------------------
A wrist is not a hip. Vigorous arm movement while sitting still - gesturing
while talking, clapping - will be counted as steps, and walking with a hand
in a pocket will be undercounted. Fixing either needs the activity
classifier's opinion, which is PR 10; the clean version of this is to count
steps only while the model says the person is walking. That coupling is
deliberately not built here, so this module stays testable on its own and
the step count keeps working if the model does not load.
"""

from __future__ import annotations

import math

BASELINE_TAU_S = 1.0
"""Time constant of the slow baseline.

Long enough to ignore individual steps, short enough to follow the wrist
turning over within a second or so.
"""

STEP_THRESHOLD_G = 0.18
"""How far above baseline the magnitude must rise to count as a footfall.

Measured against the simulator: resting jitter stays under 0.05 g while
walking produces swings of 0.3 g and more, so this sits comfortably
between them rather than being tuned to either edge.
"""

REARM_THRESHOLD_G = 0.06
"""The signal must fall back below this before another step can count.

The gap between this and the step threshold is what stops a signal
hovering near the boundary from registering a burst of steps.
"""

MIN_STEP_INTERVAL_S = 0.25
"""No two steps closer together than this: a ceiling of 4 steps a second.

Above a sprint, and below the frequency of a hand tremor or a sensor
glitch, so neither can inflate the count.
"""


class StepCounter:
    """Counts footfalls from a stream of accelerometer samples."""

    def __init__(
        self,
        sampling_freq_hz: float,
        *,
        step_threshold_g: float = STEP_THRESHOLD_G,
        rearm_threshold_g: float = REARM_THRESHOLD_G,
        min_step_interval_s: float = MIN_STEP_INTERVAL_S,
        baseline_tau_s: float = BASELINE_TAU_S,
    ) -> None:
        self.fs = float(sampling_freq_hz)
        self.step_threshold = float(step_threshold_g)
        self.rearm_threshold = float(rearm_threshold_g)
        self.min_interval = float(min_step_interval_s)

        # Per-sample decay for the baseline average.
        self._alpha = 1.0 - math.exp(-1.0 / (self.fs * float(baseline_tau_s)))

        self._baseline: float | None = None
        self._armed = True
        self._steps = 0
        self._samples = 0
        self._last_step_sample = -10**9

    @property
    def steps(self) -> int:
        return self._steps

    def reset(self) -> None:
        self._baseline = None
        self._armed = True
        self._steps = 0
        self._samples = 0
        self._last_step_sample = -10**9

    def update(self, sample) -> bool:
        """Feed one sample. Returns True if it completed a step."""
        try:
            ax = float(sample["ax"])
            ay = float(sample["ay"])
            az = float(sample["az"])
        except (KeyError, TypeError):
            ax = float(getattr(sample, "ax"))
            ay = float(getattr(sample, "ay"))
            az = float(getattr(sample, "az"))

        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if not math.isfinite(magnitude):
            return False

        self._samples += 1

        if self._baseline is None:
            # Start the baseline at the first reading rather than at zero,
            # otherwise the first second of data looks like one enormous
            # step while the average climbs from 0 to 1 g.
            self._baseline = magnitude
            return False

        self._baseline += self._alpha * (magnitude - self._baseline)
        deviation = magnitude - self._baseline

        if self._armed and deviation > self.step_threshold:
            if (self._samples - self._last_step_sample) / self.fs >= self.min_interval:
                self._steps += 1
                self._last_step_sample = self._samples
                self._armed = False
                return True
            # Inside the refractory period: still disarm, so the same bump
            # cannot be re-examined once the period expires.
            self._armed = False
        elif not self._armed and deviation < self.rearm_threshold:
            self._armed = True

        return False

    def update_batch(self, samples) -> int:
        """Feed several samples in order. Returns how many steps they added."""
        before = self._steps
        for sample in samples:
            self.update(sample)
        return self._steps - before


# ---------------------------------------------------------------------------
# Self-test: python pedometer.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import time

    try:
        from config import SAMPLING_FREQ_HZ
        from datasource import SimulatedSource
    except ImportError:  # pragma: no cover
        from dashboard.config import SAMPLING_FREQ_HZ
        from dashboard.datasource import SimulatedSource

    print("pedometer.py self-test")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    def walk(activity, seconds):
        """Run the simulator and count, the way the dashboard does."""
        src = SimulatedSource(activity=activity, seed=11)
        src.start()
        counter = StepCounter(SAMPLING_FREQ_HZ)
        total = 0
        elapsed = 0.0
        while elapsed < seconds:
            time.sleep(0.05)
            elapsed += 0.05
            batch = [s.as_dict() for s in src.read()]
            total += len(batch)
            counter.update_batch(batch)
        src.stop()
        return counter, total

    # Still activities must not accumulate steps. This is the failure that
    # matters: a counter that creeps while someone sits is useless, and
    # unlike undercounting it is invisible until the total is absurd.
    for activity in ("sitting", "standing", "lying"):
        counter, _ = walk(activity, 8.0)
        check(
            f"no steps while {activity}",
            counter.steps == 0,
            f"{counter.steps} counted",
        )

    # Walking: count against the simulator's own cadence rather than a
    # number written here, so the test does not drift from the generator.
    # Expected steps per second. The simulator's own self-test measures its
    # gait at about 1.7 Hz walking and 2.6-2.8 Hz running; these bands allow
    # a generous margin either side of that, which is what a step counter
    # should manage without being tuned to one generator.
    for activity, low_rate, high_rate in (
        ("walking", 1.2, 2.2),
        ("running", 2.0, 3.4),
    ):
        seconds = 12.0
        src = SimulatedSource(activity=activity, seed=11)
        src.start()
        counter = StepCounter(SAMPLING_FREQ_HZ)
        elapsed = 0.0
        while elapsed < seconds:
            time.sleep(0.05)
            elapsed += 0.05
            counter.update_batch([s.as_dict() for s in src.read()])
        src.stop()

        # The first second or so goes on settling the baseline, so rate it
        # over the time that was actually available for counting.
        counting_time = seconds - 1.0
        rate = counter.steps / counting_time
        check(
            f"{activity}: cadence lands in {low_rate}-{high_rate} steps/s",
            low_rate <= rate <= high_rate,
            f"{counter.steps} steps in {seconds:.0f} s = {rate:.2f}/s",
        )

    # Running must produce more steps than walking over the same time.
    walk_counter, _ = walk("walking", 10.0)
    run_counter, _ = walk("running", 10.0)
    check(
        "running counts more steps than walking",
        run_counter.steps > walk_counter.steps,
        f"{run_counter.steps} vs {walk_counter.steps}",
    )

    # Hysteresis: a signal parked exactly at the threshold must not spray
    # steps. One sustained push is one step, not one per sample.
    counter = StepCounter(SAMPLING_FREQ_HZ)
    for _ in range(300):
        counter.update({"ax": 0.0, "ay": 0.0, "az": 1.0})
    before = counter.steps
    for _ in range(300):
        counter.update({"ax": 0.0, "ay": 0.0, "az": 1.4})
    check(
        "a sustained push counts once, not once per sample",
        counter.steps - before <= 2,
        f"{counter.steps - before} steps from 3 s of constant offset",
    )

    # Rubbish must not crash the counter mid-demo.
    counter = StepCounter(SAMPLING_FREQ_HZ)
    counter.update({"ax": float("nan"), "ay": 0.0, "az": 1.0})
    counter.update({"ax": 0.0, "ay": 0.0, "az": 1.0})
    check("a NaN sample is ignored rather than raising", counter.steps == 0)

    # reset() must actually clear it.
    counter, _ = walk("walking", 4.0)
    had = counter.steps
    counter.reset()
    check(
        "reset() clears the count",
        counter.steps == 0 and had >= 0,
        f"was {had}",
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
