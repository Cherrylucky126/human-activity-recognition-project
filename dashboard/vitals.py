"""Live heart rate from the PPG channel.

PR 6 of 14.

Read this first
---------------
**This is not a medical measurement.** It is an indicative number from a
consumer optical sensor on a wrist strap, shown so the dashboard has
something to say about the PPG channel. The literature review is explicit
about this and so is the dashboard footer. Do not quote it as a vital sign
in the report.

Why this file exists at all
---------------------------
The training pipeline already computes a heart rate feature
(``rolling_bpm`` in ``data-preprocessing/feature-extraction``), so it is
fair to ask why the dashboard does not simply call that.

It cannot, and the reason is worth understanding rather than working
around. That function smooths with a **centred** median: to produce the
value at time *t* it looks at windows on both sides of *t*, including the
future. Offline, over a finished recording, that is free and it is the
right thing to do. Live, the future has not happened yet. The same is true
of the IR z-score there, which is computed over the whole dataset at once.

So this module estimates the same quantity from only the past. The numbers
will not match hers exactly, and they are not supposed to: a causal
estimate is always a little noisier and a little later than a centred one.
What matters is that this one can exist on a live stream at all.

How it works
------------
1. **Remove the slow part.** The raw IR count sits around 85,000 and
   drifts with skin contact and strap pressure. Subtracting a short moving
   average leaves only the fast part - the pulse, plus motion.
2. **Ask how fast it repeats.** Autocorrelation over the shifts that
   correspond to 40-200 BPM: at what delay does this window most resemble
   itself? Counting individual beats instead sounds simpler, but every
   real pulse has a second smaller bump (the dicrotic notch) and the
   sensor adds spikes of its own, so a peak counter needs thresholds that
   have to be retuned for every strap tightness. Asking about repetition
   sidesteps all of it: a doubled bump shifts the answer not at all.
3. **Check that it actually repeats.** The height of that peak, relative
   to the signal's own energy, says how periodic the window really is.
   Below a confidence of 0.30, no number is shown. This is the single
   thing that stops the dashboard reporting a runner's step rate as a
   pulse.

When it refuses to answer
-------------------------
``estimate()`` returns the previous reading - or ``None`` if there has
never been one - whenever the window does not support a new answer. The
dashboard shows ``--`` for that. A wrong heart rate displayed confidently
is worse than no heart rate, because nobody can tell the difference by
looking at it.

During running the motion artefact in the PPG channel is larger than the
pulse itself and sits at step frequency. Expect this module to be
uncertain there. That is honest behaviour for a wrist optical sensor, not
a bug to tune away.
"""

from __future__ import annotations

MIN_BPM = 40.0
MAX_BPM = 200.0
"""Outside this range the estimate is refused rather than displayed.

40 BPM is below a resting athlete; 200 is above a maximal heart rate for
anyone in this study. A value outside it means the analysis has locked
onto something that is not a pulse.
"""

MIN_SECONDS = 6.0
"""Shorter than this and a slow pulse has not repeated enough times for
its period to stand out."""

MIN_CONFIDENCE = 0.30
"""How strongly the window must repeat before a number is shown.

The best autocorrelation peak divided by the signal's own energy: 1.0
would be a perfect repeat, 0.0 none at all. Because it is a ratio it does
not depend on pulse amplitude, which is exactly the quantity that changes
with how tightly the strap is worn.
"""

SMOOTHING = 0.25
"""Weight of each new estimate in the displayed value.

Low enough that the readout is steady, high enough that a real change
arrives within a few seconds.
"""

BASELINE_SECONDS = 0.75
"""Width of the moving average that is subtracted off.

Longer than one beat at 200 BPM, shorter than one at 40, so it removes
drift without removing the pulse along with it.
"""

WORKING_RATE_HZ = 25.0
"""The IR channel is averaged down to this before analysis.

A pulse never exceeds 3.5 Hz, so 25 Hz is far above what Nyquist asks for.
Averaging blocks of four samples also suppresses high-frequency sensor
noise, and cuts the autocorrelation to a few thousand operations - small
enough to run ten times a second without the page stuttering.
"""


class HeartRateEstimator:
    """Turns a window of IR counts into a BPM reading, or into ``None``.

    Holds one value between calls: the smoothed reading. Everything else is
    recomputed from the window handed in, so the estimator can be fed out
    of order or restarted without going wrong.
    """

    def __init__(self, sampling_freq_hz: float) -> None:
        self.fs = float(sampling_freq_hz)
        self._smoothed: float | None = None
        self._last_raw: float | None = None
        self._last_confidence: float = 0.0
        self._refusals = 0
        self._accepted = 0

    def reset(self) -> None:
        self._smoothed = None
        self._last_raw = None
        self._last_confidence = 0.0

    @property
    def bpm(self) -> float | None:
        """The smoothed reading, or ``None`` if there isn't one yet."""
        return self._smoothed

    @property
    def confidence(self) -> float:
        """How periodic the most recently accepted window was, 0.0 to 1.0."""
        return self._last_confidence

    @property
    def refusals(self) -> int:
        return self._refusals

    @property
    def accepted(self) -> int:
        return self._accepted

    def set_sampling_rate(self, sampling_freq_hz: float) -> None:
        """Re-tune to the measured rate.

        Everything rate-dependent is derived inside :meth:`estimate` from
        ``self.fs``, so this is the whole change. It matters more here
        than anywhere: a period misread by the ratio of the two rates
        becomes a BPM wrong by that same ratio, and 2x a resting pulse is
        about 130 - high, but not high enough for the 40-200 plausibility
        check to reject it.
        """
        fs = float(sampling_freq_hz)
        if fs <= 0:
            raise ValueError("sampling_freq_hz must be positive")
        self.fs = fs

    def estimate(self, ir_values) -> float | None:
        """Estimate BPM from a window of raw IR counts, most recent last.

        An unusable window never wipes out a reading that was already
        good - it just stops updating it.
        """
        values = list(ir_values)
        if len(values) < MIN_SECONDS * self.fs:
            self._refusals += 1
            return self._smoothed

        working, fs = _downsample(values, self.fs, WORKING_RATE_HZ)
        detrended = _remove_baseline(working, int(BASELINE_SECONDS * fs))

        period, confidence = _dominant_period(detrended, fs, MIN_BPM, MAX_BPM)
        if period is None or confidence < MIN_CONFIDENCE:
            self._refusals += 1
            return self._smoothed

        raw = 60.0 / period
        if not (MIN_BPM <= raw <= MAX_BPM):
            self._refusals += 1
            return self._smoothed

        self._accepted += 1
        self._last_raw = raw
        self._last_confidence = confidence
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed += SMOOTHING * (raw - self._smoothed)
        return self._smoothed


def _downsample(values, fs_in, fs_target):
    """Average blocks of samples down to roughly ``fs_target``."""
    factor = max(1, int(round(fs_in / fs_target)))
    if factor == 1:
        return list(values), fs_in
    out = []
    for i in range(0, len(values) - factor + 1, factor):
        block = values[i : i + factor]
        out.append(sum(block) / factor)
    return out, fs_in / factor


def _remove_baseline(values, width):
    """Subtract a moving average, leaving only the fast part.

    The average is taken over samples on both sides of each point, but all
    of them are inside the window already in hand, which is entirely in the
    past. No future sample is involved - unlike the offline version this
    replaces.
    """
    width = max(3, int(width))
    n = len(values)
    half = width // 2
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = values[i] - sum(values[lo:hi]) / (hi - lo)
    return out


def _dominant_period(x, fs, min_bpm, max_bpm):
    """Period in seconds of the strongest repetition, and how strong it is.

    Returns ``(None, 0.0)`` when the window cannot support an answer.
    """
    n = len(x)
    if n < 8:
        return None, 0.0

    energy = sum(v * v for v in x) / n
    if energy <= 0:
        return None, 0.0

    lo = max(1, int(fs * 60.0 / max_bpm))
    hi = min(n // 2, int(fs * 60.0 / min_bpm) + 1)
    if hi <= lo + 1:
        return None, 0.0

    scores = []
    for lag in range(lo, hi + 1):
        overlap = n - lag
        total = 0.0
        for i in range(overlap):
            total += x[i] * x[i + lag]
        scores.append(total / overlap)

    best_i = max(range(len(scores)), key=scores.__getitem__)
    best_lag = lo + best_i
    best = scores[best_i]
    if best <= 0:
        return None, 0.0

    # Parabolic interpolation against the neighbouring lags, so the answer
    # is not quantised to whole samples. At 25 Hz one sample of lag is
    # worth several BPM, which would show as a readout that jumps between
    # fixed values instead of drifting.
    refined = float(best_lag)
    if 0 < best_i < len(scores) - 1:
        a, b, c = scores[best_i - 1], scores[best_i], scores[best_i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            shift = 0.5 * (a - c) / denom
            if -1.0 < shift < 1.0:
                refined = best_lag + shift

    return refined / fs, best / energy


# ---------------------------------------------------------------------------
# Self-test: python vitals.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import random
    import time

    try:
        from config import SAMPLING_FREQ_HZ
        from datasource import SimulatedSource
    except ImportError:  # pragma: no cover
        from dashboard.config import SAMPLING_FREQ_HZ
        from dashboard.datasource import SimulatedSource

    print("vitals.py self-test")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    window = int(8 * SAMPLING_FREQ_HZ)

    def collect(activity, bpm, seconds):
        """Gather IR from the simulator, and its true heart rate at the end.

        The simulator generates from the monotonic clock, so this has to
        pass real time rather than ask for a block of samples at once.

        It returns the *ground truth at the end of the run*, not the value
        that was asked for. The simulator eases heart rate toward a new
        target with an 8 s time constant, so over a short recording the
        requested figure is simply not what the PPG channel is carrying.
        Testing against the request rather than the truth would be testing
        the simulator's lag, not this estimator.
        """
        src = SimulatedSource(activity=activity, seed=7)
        if bpm is not None:
            src.set_heart_rate(bpm)
        src.start()
        ir = []
        elapsed = 0.0
        while elapsed < seconds:
            time.sleep(0.05)
            elapsed += 0.05
            ir.extend(s.IR_Value for s in src.read())
        truth = src.heart_rate_bpm
        src.stop()
        return ir, truth

    def run_stream(ir):
        est = HeartRateEstimator(SAMPLING_FREQ_HZ)
        for end in range(window, len(ir) + 1, int(SAMPLING_FREQ_HZ)):
            est.estimate(ir[end - window : end])
        return est

    # Heart rate is driven by the activity, so each activity is a steady
    # rate to recover. These are the accuracy checks: compare the estimate
    # against the rate the simulator says it is actually generating.
    for label, activity, forced, seconds in (
        ("sitting", "sitting", None, 14.0),
        ("standing", "standing", None, 14.0),
    ):
        ir, truth = collect(activity, forced, seconds)
        est = run_stream(ir)
        got = est.bpm
        check(
            f"pulse recovered, {label} (true {truth:.0f} BPM)",
            got is not None and abs(got - truth) <= 8,
            f"estimated {got:.1f}, confidence {est.confidence:.2f}"
            if got
            else "no estimate",
        )

    # A forced-high rate decays back toward the activity's resting rate, so
    # it sweeps rather than holding still. An 8 s window averaged over a
    # sweep cannot equal the instantaneous value at the end of it, and
    # asserting that it does would only be measuring the estimator's lag.
    # What must be true is that the readout clearly rises above resting.
    ir, truth = collect("sitting", 130, 10.0)
    est = run_stream(ir)
    got = est.bpm
    check(
        "a raised heart rate reads clearly above resting",
        got is not None and got > 85 and got >= truth - 2,
        f"estimated {got:.1f} while decaying through {truth:.0f} BPM"
        if got
        else "no estimate",
    )

    est = HeartRateEstimator(SAMPLING_FREQ_HZ)
    check(
        "a 2 s window is refused",
        est.estimate([85000.0] * 200) is None,
    )

    est = HeartRateEstimator(SAMPLING_FREQ_HZ)
    check(
        "a flat signal gives no reading",
        est.estimate([85000.0] * 1000) is None,
    )

    rng = random.Random(3)
    est = HeartRateEstimator(SAMPLING_FREQ_HZ)
    noise = [85000 + rng.gauss(0, 400) for _ in range(1200)]
    check(
        "random noise is refused",
        est.estimate(noise) is None,
        f"{est.refusals} refusal(s)",
    )

    ir, _ = collect("sitting", None, 14.0)
    est = run_stream(ir)
    held = est.bpm
    est.estimate([85000.0] * 50)
    check(
        "a bad window holds the last good reading rather than blanking it",
        held is not None and est.bpm == held,
        f"{held:.1f} BPM held" if held else "none",
    )

    ir, truth = collect("walking", None, 20.0)
    est = run_stream(ir)
    got = est.bpm
    check(
        f"pulse still recovered while walking (true {truth:.0f} BPM)",
        got is not None and abs(got - truth) <= 12,
        f"estimated {got:.1f}" if got else "no estimate",
    )

    ir, _ = collect("running", None, 16.0)
    est = run_stream(ir)
    print(
        f"  --   running: estimate "
        f"{'none' if est.bpm is None else round(est.bpm)}, "
        f"{est.refusals} refusals / {est.accepted} accepted "
        f"- motion artefact dominates here; informational, not pass/fail"
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
