"""Threshold-based fall detection for the live HAR dashboard.

Why this file is not a machine-learning model
---------------------------------------------
The rest of the project classifies activities with models trained in
``data-processing/``. Those models are useless for falls, for a very simple
reason: nobody fell over while the training data was being recorded. A
classifier can only ever report the classes it was shown, so asking it "is this
a fall?" is asking a question it has never seen an example of. It would answer
with whichever ordinary activity happens to look closest, which is worse than
not answering at all.

So fall detection here is a *rule*, written out by hand, with every number
visible at the top of the file where it can be argued about and tuned. The rule
comes from the project's literature review:

    Bagala, F. et al. (2012), "Evaluation of Accelerometer-Based Fall Detection
    Algorithms on Real-World Falls", PLOS ONE 7(5): e37062.

That paper compares thirteen published algorithms against real falls recorded
from real people, and the family of algorithms that survives contact with
reality all look for the same three-part *signature*, in this order:

    1. free-fall   - the body stops being supported, so the accelerometer
                     briefly reads almost nothing,
    2. impact      - the body hits the floor and is violently decelerated,
                     so the accelerometer reads a sharp spike,
    3. stillness   - the person is on the floor and, for a second or so,
                     is not moving much.

The ordering is the whole point. Each stage on its own is something ordinary
life produces dozens of times a day. A sharp spike alone is setting a glass
down. A quiet second alone is sitting at a desk. A moment of free-fall alone is
a hop, or an arm swing at the bottom of its arc. It is the three of them, in
that sequence, inside a window of a couple of seconds, that is hard to produce
by accident.

The physics, stage by stage
---------------------------
An accelerometer does not measure "acceleration" in the everyday sense. It
measures *proper acceleration*: the force per unit mass that the sensor's
casing is applying to its internal mass. Sitting on a table, the table pushes
up on the sensor to stop it falling, and the sensor reports that push as
1 g upwards. This is why a still accelerometer reads 1 g and not 0 g.

We collapse the three axes into one number, the signal vector magnitude:

    SVM = sqrt(ax**2 + ay**2 + az**2)

Taking the magnitude throws away orientation, which is exactly what we want:
a wrist unit is in a different orientation every second, and the rule must not
care which way round the hand happens to be.

* Stage 1, free-fall -> SVM near zero. If nothing is holding the sensor up,
  nothing is pushing on it, so there is nothing to measure. Everything inside
  the sensor is falling together. This is the same reason astronauts float:
  not an absence of gravity, an absence of support. A falling person is not a
  clean 0 g - the arm is still attached to a body that is rotating, and the
  wrist clips furniture on the way down - so the literature uses a threshold
  comfortably below 1 g rather than a test for zero.

* Stage 2, impact -> SVM spikes well above 1 g. The floor stops the body over
  a very short distance. Stopping anything quickly takes a large force, and
  force per unit mass is precisely what the sensor reports. A few centimetres
  of give in flesh and carpet is enough to turn a modest landing speed into
  several g for a few tens of milliseconds.

* Stage 3, stillness -> SVM back to roughly 1 g, and *staying* there. Once
  the person is on the floor the floor supports them again, so gravity's 1 g
  reappears. What distinguishes this from ordinary standing still is that it
  arrives immediately after stages 1 and 2. This stage is what rejects
  running: a runner produces free-fall between strides and an impact at every
  footfall, but never a quiet second, because the next stride is already
  coming.

Design properties this module is required to have
-------------------------------------------------
* **Causal.** Every decision is made from samples already seen. Nothing looks
  ahead, because in the live dashboard there is no ahead yet. Concretely, that
  means the detector cannot "look back" at a buffer and re-decide; it is a
  state machine that is pushed one sample at a time and remembers only a
  handful of counters.
* **Latching.** Once a fall is confirmed the alert stays up until something
  explicitly dismisses it. A person lying on the floor is in no position to
  clear a notification, and an alert that times out by itself is an alert that
  nobody sees.
* **Self-contained.** It takes plain dicts, not another module's class, and it
  imports nothing but the standard library and ``config.SAMPLING_FREQ_HZ``.
  No dash, no plotly, no numpy - so it can be unit-tested, or reused in the
  firmware-side tooling, on a machine with none of those installed.

Usage
-----
::

    from falldetection import FallDetector

    detector = FallDetector()
    for sample in stream:                  # dicts: timestamp, ax..gz, IR_Value
        if detector.update(sample):
            print("FALL", detector.last_event)
    if detector.alert_active:
        ...                                # paint the dashboard red
    detector.dismiss()                     # when a human acknowledges it

Self-test::

    cd dashboard
    python falldetection.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

# ``config`` is imported rather than hard-coding 100 Hz: if the device is ever
# re-flashed at a different rate, every duration below re-derives itself. The
# dashboard is run from inside its own folder (``python app.py``), so the flat
# import is the normal case; the package-qualified form is a fallback for when
# something imports this module as ``dashboard.falldetection`` from the repo
# root.
try:  # pragma: no cover - import plumbing, not logic
    from config import SAMPLING_FREQ_HZ
except ImportError:  # pragma: no cover
    from dashboard.config import SAMPLING_FREQ_HZ


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
# EVERY number the rule depends on lives in this block, with its unit and its
# source. These are literature figures, not measurements from this device, so
# they are starting points: the person who first straps the real wrist unit on
# should expect to adjust them. The suggested order of tuning is
#   1. IMPACT_THRESHOLD_G          (easiest to measure: drop-test the unit)
#   2. FREE_FALL_MAX_DURATION_S    (the jump-versus-fall knob, see below)
#   3. STILLNESS_TOLERANCE_G       (depends on how noisy the unit really is)
# Nothing else in this file hard-codes a threshold or a duration, so changing a
# value here is the whole change.

# Reference reading of a still accelerometer, in g. Not a tuning knob - it is
# what "supported by the floor" means. 1 g is gravity's worth of support force.
GRAVITY_G = 1.0

# --- Stage 1: free-fall ----------------------------------------------------
# Units: g (multiples of 9.81 m/s**2).
# Below this SVM, the wrist is treated as unsupported. Bagala et al. (2012)
# and the algorithms it reviews use free-fall thresholds in the 0.4-0.7 g
# band; the project literature review quotes ~0.6 g. Lower = fewer false
# alarms but misses falls that are cushioned or partly broken by an arm.
FREE_FALL_THRESHOLD_G = 0.6

# Units: seconds. The dip must last at least this long. A single stray sample
# under 0.6 g is sensor noise or one hard jolt of an arm swing, not a fall.
# 30 ms = 3 samples at 100 Hz.
FREE_FALL_MIN_DURATION_S = 0.030

# Units: seconds. The dip must ALSO not last longer than this. This is the
# single most important false-alarm guard in the file, and the least obvious.
# A real fall is a rotation about the hips, not a drop: the wrist is only
# genuinely unsupported for a few tens of milliseconds, which is why the
# literature range is 30-100 ms. A JUMP, by contrast, is true free-fall for
# the whole airborne phase - even a modest 10 cm hop is ~280 ms - and a jump
# otherwise produces a textbook fall signature (free-fall, landing impact,
# then standing still). Capping the dip at 150 ms keeps the full literature
# range with margin while rejecting all but the very smallest hops. Raise this
# and jumps start alarming; lower it and slow, cushioned falls are missed.
FREE_FALL_MAX_DURATION_S = 0.150

# --- Stage 2: impact -------------------------------------------------------
# Units: g. SVM must spike above this for the impact stage to be satisfied.
#
# The literature figure is ~2.5 g: the algorithms surveyed by Bagala et al.
# use 2-3 g at the waist, and a wrist sees *higher* peaks than a waist for the
# same fall, being a light limb at the end of a lever. On the sensor data this
# project actually collects, that figure cannot be reached.
#
# The collector sketch never writes MPU6050 register 0x1C, so the accelerometer
# stays on its default +/-2 g range. EACH AXIS is clipped at 2 g before it
# leaves the device.
#
# Per axis, which is the part this comment originally got wrong. The rule below
# works on the vector magnitude, and three axes clipped at 2 g still give
# sqrt(3) * 2 = 3.46 g. replay.py measures 2.86 g at impact on clipped data.
#
# So 2.5 g is reachable for some orientations and not for others: an impact
# landing square on one axis gives exactly 2.00 g and could never cross it,
# one landing across two or three axes reaches 2.8-3.5 g and would. A 2.5 g
# threshold is not impossible here - it is orientation-dependent, which for a
# fall alarm is arguably worse, because it fails silently for a whole class of
# falls. That is the real argument for lowering it; see docs/known-limitations.
#
# 1.8 g is the team's decision (15 Nov), taken over the alternative of setting
# 0x1C to +/-8 g: that is the better fix in principle, but it invalidates every
# trained model, all of which were fitted on +/-2 g data, and there is no time
# to re-collect and re-train before the 17th.
#
# The cost is specificity. 1.8 g sits inside the range of ordinary vigorous
# movement, so stage 2 alone is weaker evidence than it was. What keeps the
# false-alarm rate down is that stage 2 never fires alone: an alert still needs
# a 30-150 ms free-fall dip below 0.6 g first, and a second of stillness after.
# Hard knocks and sit-downs are rejected at stage 1, as the self-tests check.
#
# If the firmware is ever fixed to +/-8 g, raise this back to 2.5 g in the same
# change - the two settings belong together.
IMPACT_THRESHOLD_G = 1.8

# Units: seconds. How long after the dip ends we keep waiting for that spike.
# Physically the floor arrives within a few tens of milliseconds of the
# free-fall ending; 300 ms is generous slack for a fall that bounces off a
# chair on the way down.
IMPACT_SEARCH_WINDOW_S = 0.300

# Units: seconds. Grace period straight after the impact, during which the
# signal is allowed to ring and rebound without counting against stillness.
# Flesh, clothing and the strap all oscillate for a moment after a hit.
POST_IMPACT_SETTLE_S = 0.150

# --- Stage 3: post-fall stillness -----------------------------------------
# Units: seconds. How long the quiet has to last. "About a second" is the
# figure from the literature review. Longer = more confident, but delays the
# alert by exactly that much, which matters when the point is summoning help.
STILLNESS_DURATION_S = 1.000

# Units: g. How far SVM may stray from 1 g and still count as "quiet". A
# sample outside 1 g +/- this is counted as movement. Should be a few times
# the unit's resting noise; 0.25 g is loose enough to tolerate breathing and
# a shaking hand, tight enough that walking (which swings several tenths of a
# g) fails immediately.
STILLNESS_TOLERANCE_G = 0.25

# Unitless fraction of the stillness window that is allowed to be "movement"
# before the stage is rejected. A person who has fallen may groan, roll, or
# try to get up; demanding a perfectly flat second would throw away real
# falls. 0.20 of 1 s = up to 200 ms of movement tolerated.
STILLNESS_ACTIVE_ALLOWANCE = 0.20

# --- Whole-sequence guard --------------------------------------------------
# Units: seconds. Hard ceiling on the time from the start of the free-fall dip
# to the confirmation of the alert. Three stages that each happen eventually
# are a coincidence; three stages inside two and a half seconds are a fall.
# Must stay larger than the sum of the stage budgets above
# (0.150 + 0.300 + 0.150 + 1.000 = 1.600 s) or falls become impossible to
# confirm.
MAX_SEQUENCE_DURATION_S = 2.500

# Accepted spellings of the timestamp field, tried in this order. The upstream
# data source is not written yet, so this module refuses to depend on one
# spelling of one key in one other file. Timestamps are used only for
# *reporting* when the fall happened (see ``_timestamp_of``); none of the
# decision logic depends on them.
TIMESTAMP_KEYS = ("timestamp", "t", "time", "ts")


def _duration_to_samples(seconds: float, sampling_freq_hz: float) -> int:
    """Convert a duration in seconds into a whole number of samples.

    At 100 Hz each sample covers 10 ms, so 30 ms is 3 samples. Rounds to the
    nearest sample and never returns less than 1: a duration of "one sample"
    is the shortest thing a sampled signal can express.
    """
    return max(1, round(seconds * sampling_freq_hz))


def signal_vector_magnitude(sample: Mapping[str, Any]) -> float:
    """Return sqrt(ax**2 + ay**2 + az**2) for one sample, in g.

    This is the length of the acceleration vector, so it answers "how hard is
    this thing being pushed, in total?" while ignoring which direction the
    push points. A unit lying still on a table reads about 1.0 whichever face
    is down, which is the property that makes the rule orientation-free.
    """
    ax = float(sample["ax"])
    ay = float(sample["ay"])
    az = float(sample["az"])
    return math.sqrt(ax * ax + ay * ay + az * az)


class FallStage(str, Enum):
    """Where the detector currently is in the three-stage signature.

    Exposed so the dashboard can show progress ("impact seen, waiting to see
    if they get up") rather than only a final yes/no, and so that whoever
    tunes the thresholds can watch which stage keeps rejecting their test
    falls.
    """

    IDLE = "idle"                    # nothing interesting; watching for a dip
    FREE_FALL = "free_fall"          # stage 1 running, SVM is under threshold
    AWAIT_IMPACT = "await_impact"    # dip was valid, now hoping for the spike
    SETTLING = "settling"            # spike seen, letting the ringing die down
    STILLNESS = "stillness"          # stage 3 running, counting quiet samples


@dataclass(frozen=True)
class FallEvent:
    """Everything known about one confirmed fall.

    Kept as a small immutable record so the dashboard can show something
    specific ("fall at t=12.4 s, 3.1 g impact") instead of a bare red light.
    """

    detected_at: float            # timestamp of the confirming sample, seconds
    impact_at: float              # timestamp of the impact spike, seconds
    peak_impact_g: float          # largest SVM seen during impact + settling, g
    free_fall_min_g: float        # lowest SVM seen during the dip, g
    free_fall_duration_s: float   # how long the dip lasted, seconds
    sample_index: int             # how many samples the detector had seen

    def summary(self) -> str:
        """One human-readable line, for a dashboard banner or a log."""
        return (
            f"fall at t={self.detected_at:.2f}s "
            f"(impact {self.peak_impact_g:.2f} g at t={self.impact_at:.2f}s, "
            f"free-fall {self.free_fall_duration_s * 1000:.0f} ms "
            f"to {self.free_fall_min_g:.2f} g)"
        )


class FallDetector:
    """Causal, latching, three-stage fall detector fed one sample at a time.

    How to think about it
    ---------------------
    It is a *state machine*. Picture a token sitting on one of five squares
    (:class:`FallStage`). Each new sample moves the token, or leaves it where
    it is, based only on that sample and a few counters. There is no buffer of
    history to re-examine and no peeking at future samples, which is what
    makes it usable live - the dashboard can hand it the newest reading the
    instant it arrives.

    The journey to an alert is::

        IDLE         --SVM drops below 0.6 g-------------> FREE_FALL
        FREE_FALL    --dip ended, lasted 30-150 ms------> AWAIT_IMPACT
        AWAIT_IMPACT --SVM spikes past 1.8 g in 300 ms--> SETTLING
        SETTLING     --150 ms of ringing allowed to pass-> STILLNESS
        STILLNESS    --1 s with <=20% movement----------> FALL CONFIRMED, IDLE

    Any failed condition sends the token straight back to IDLE, and the whole
    journey must finish within :data:`MAX_SEQUENCE_DURATION_S`.

    All stage lengths are counted in *samples*, derived from the sampling rate,
    rather than from the stream's timestamps. That is deliberate: a serial link
    that stutters produces late timestamps, and a rule built on timestamps
    would silently change its mind about how long 30 ms is. The trade-off is
    the opposite failure - if the link *drops* samples, the stages become
    longer in wall-clock terms than they look here.

    Thresholds can be overridden per instance (handy for tuning against a
    recording without editing the module), but the module-level constants are
    the defaults and the documentation.
    """

    def __init__(
        self,
        sampling_freq_hz: float = SAMPLING_FREQ_HZ,
        *,
        free_fall_threshold_g: float = FREE_FALL_THRESHOLD_G,
        free_fall_min_duration_s: float = FREE_FALL_MIN_DURATION_S,
        free_fall_max_duration_s: float = FREE_FALL_MAX_DURATION_S,
        impact_threshold_g: float = IMPACT_THRESHOLD_G,
        impact_search_window_s: float = IMPACT_SEARCH_WINDOW_S,
        post_impact_settle_s: float = POST_IMPACT_SETTLE_S,
        stillness_duration_s: float = STILLNESS_DURATION_S,
        stillness_tolerance_g: float = STILLNESS_TOLERANCE_G,
        stillness_active_allowance: float = STILLNESS_ACTIVE_ALLOWANCE,
        max_sequence_duration_s: float = MAX_SEQUENCE_DURATION_S,
    ) -> None:
        if sampling_freq_hz <= 0:
            raise ValueError("sampling_freq_hz must be positive")

        self.sampling_freq_hz = float(sampling_freq_hz)
        self.sample_period_s = 1.0 / self.sampling_freq_hz

        # Thresholds in g, kept as given.
        self.free_fall_threshold_g = free_fall_threshold_g
        self.impact_threshold_g = impact_threshold_g
        self.stillness_tolerance_g = stillness_tolerance_g

        # The durations in seconds are the real parameters; the sample
        # counts below are only their expression at one sampling rate.
        # Keeping the seconds means the detector can be re-tuned when the
        # device turns out to be sending at a different rate, instead of
        # measuring a 100 ms free fall as 190 ms and rejecting it.
        self._durations_s = {
            "free_fall_min": free_fall_min_duration_s,
            "free_fall_max": free_fall_max_duration_s,
            "impact_search": impact_search_window_s,
            "settle": post_impact_settle_s,
            "stillness": stillness_duration_s,
            "max_sequence": max_sequence_duration_s,
        }
        self._stillness_active_allowance = stillness_active_allowance
        self._recompute_sample_counts()

        if self.free_fall_max_samples < self.free_fall_min_samples:
            raise ValueError(
                "free_fall_max_duration_s must be >= free_fall_min_duration_s"
            )

        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def _recompute_sample_counts(self) -> None:
        """Express every duration in samples at the current rate."""
        to_samples = _duration_to_samples
        fs = self.sampling_freq_hz
        d = self._durations_s
        self.free_fall_min_samples = to_samples(d["free_fall_min"], fs)
        self.free_fall_max_samples = to_samples(d["free_fall_max"], fs)
        self.impact_search_samples = to_samples(d["impact_search"], fs)
        self.settle_samples = to_samples(d["settle"], fs)
        self.stillness_samples = to_samples(d["stillness"], fs)
        self.max_sequence_samples = to_samples(d["max_sequence"], fs)
        self.stillness_active_allowance_samples = int(
            self._stillness_active_allowance * self.stillness_samples
        )

    def set_sampling_rate(self, sampling_freq_hz: float) -> None:
        """Re-tune every duration to the rate samples really arrive at.

        The thresholds in g are untouched - they are physics, not timing.
        Only the windows change, which is what goes wrong when the device
        does not send at the rate the detector assumed: at 52 Hz a
        detector tuned for 100 Hz sees every duration as roughly double,
        so a genuine free fall runs past ``free_fall_max_samples`` and the
        sequence is discarded before the impact stage is ever reached.

        Any part-finished sequence is abandoned, because its stage
        boundaries were measured against the old counts and comparing them
        to the new ones would mix two timebases. A latched alert survives:
        it records something that already happened.
        """
        fs = float(sampling_freq_hz)
        if fs <= 0:
            raise ValueError("sampling_freq_hz must be positive")
        self.sampling_freq_hz = fs
        self.sample_period_s = 1.0 / fs
        self._recompute_sample_counts()
        self._clear_sequence()

    def reset(self) -> None:
        """Clear everything: the latched alert, the state machine and the log.

        Use this when the stream restarts (new device, new session), where
        carrying counters across a discontinuity would be meaningless. To
        acknowledge an alert while leaving the detector's history intact, use
        :meth:`dismiss` instead.
        """
        self._alert_active = False
        self._last_event: Optional[FallEvent] = None
        self._fall_count = 0
        self._samples_seen = 0
        self._last_timestamp = 0.0
        self._clear_sequence()

    def dismiss(self) -> None:
        """Acknowledge and lower the latched alert.

        The alert does not clear itself: that is the entire point of a latch.
        Something external - a carer pressing a button on the dashboard - has
        to say "seen it". The record of the fall survives in
        :attr:`last_event` and :attr:`fall_count`, so the event is not lost
        from the history just because the light went out.
        """
        self._alert_active = False

    def _clear_sequence(self) -> None:
        """Send the token back to IDLE and forget the partial signature."""
        self._state = FallStage.IDLE
        self._sequence_samples = 0        # samples since the dip began
        self._free_fall_samples = 0
        self._free_fall_min_g = math.inf
        self._await_impact_samples = 0
        self._settle_samples_seen = 0
        self._peak_impact_g = 0.0
        self._impact_timestamp = 0.0
        self._stillness_samples_seen = 0
        self._stillness_active_samples = 0
        # True while we are waiting for SVM to climb back above the free-fall
        # threshold after rejecting an over-long dip. Without this, a 280 ms
        # jump would be rejected at 150 ms and then immediately start counting
        # a brand-new dip from sample 16, manufacturing a valid-looking 130 ms
        # dip out of the tail of the one we just threw away.
        self._await_recovery = False

    # -- read-only view ----------------------------------------------------
    @property
    def alert_active(self) -> bool:
        """True from the moment a fall is confirmed until :meth:`dismiss`."""
        return self._alert_active

    # Alias: callers reach for both names, and a fall alert is important
    # enough that guessing the attribute should not be a silent AttributeError.
    @property
    def fall_detected(self) -> bool:
        """Alias of :attr:`alert_active`."""
        return self._alert_active

    @property
    def state(self) -> FallStage:
        """Which stage of the signature the detector is currently in."""
        return self._state

    @property
    def last_event(self) -> Optional[FallEvent]:
        """The most recent confirmed fall, or None if there has never been one."""
        return self._last_event

    @property
    def fall_count(self) -> int:
        """How many falls have been confirmed since the last :meth:`reset`."""
        return self._fall_count

    def status(self) -> dict:
        """Flat dict of the detector's state, for a dashboard callback.

        Returning a plain dict (rather than the objects themselves) keeps the
        Dash side free of imports from this module beyond the class itself,
        and is trivially JSON-serialisable into a ``dcc.Store``.
        """
        event = self._last_event
        return {
            "alert_active": self._alert_active,
            "stage": self._state.value,
            "fall_count": self._fall_count,
            "samples_seen": self._samples_seen,
            "detected_at": event.detected_at if event else None,
            "peak_impact_g": event.peak_impact_g if event else None,
            "summary": event.summary() if event else None,
        }

    # -- the state machine -------------------------------------------------
    def update(self, sample: Mapping[str, Any]) -> bool:
        """Feed one sample in; return True only on the sample that confirms a fall.

        The return value is an *edge*, not a level: it is True exactly once per
        fall, on the sample where stage 3 completes. Use it to fire a one-shot
        action (log it, send a message). For "is the alert up right now?", read
        :attr:`alert_active`, which is a level and stays raised.

        ``sample`` is any mapping with ``ax``, ``ay``, ``az`` in g and
        (optionally) a timestamp in seconds. ``gx``..``gz`` and ``IR_Value``
        are accepted and ignored - see the module notes on why the rule is
        accelerometer-only.
        """
        self._samples_seen += 1
        timestamp = self._timestamp_of(sample)
        self._last_timestamp = timestamp
        svm = signal_vector_magnitude(sample)
        return self._step(svm, timestamp)

    def update_batch(self, samples: Iterable[Mapping[str, Any]]) -> bool:
        """Feed several samples in order; return True if any confirmed a fall.

        Convenience for a dashboard tick that drains a queue. Samples are
        still processed strictly one at a time and in order, so a batch behaves
        identically to the same samples fed individually - batching is never a
        way of looking ahead.
        """
        confirmed = False
        for sample in samples:
            if self.update(sample):
                confirmed = True
        return confirmed

    def _timestamp_of(self, sample: Mapping[str, Any]) -> float:
        """Pull a timestamp out of a sample, or synthesise one from the count.

        A missing timestamp must not break detection, because nothing in the
        rule depends on it; we only need *some* number to report back. Falling
        back to ``samples_seen / rate`` gives a sensible seconds-since-start.
        """
        for key in TIMESTAMP_KEYS:
            if key in sample:
                try:
                    return float(sample[key])
                except (TypeError, ValueError):
                    break
        return self._samples_seen * self.sample_period_s

    def _step(self, svm: float, timestamp: float) -> bool:
        """Advance the machine by one sample; True if this sample confirms a fall.

        Written as a flat if-chain with deliberate fall-through: a sample that
        ends the free-fall dip may *also* be the impact spike (the signal can
        go from 0.3 g to 3 g in a single 10 ms step), so after promoting
        FREE_FALL -> AWAIT_IMPACT we test the very same sample for impact
        rather than throwing it away.
        """
        in_free_fall = svm < self.free_fall_threshold_g

        # --- IDLE: watching for the start of a dip ------------------------
        if self._state is FallStage.IDLE:
            if self._await_recovery:
                # Still below threshold from a dip we already rejected. Only
                # re-arm once the signal has genuinely come back up.
                if not in_free_fall:
                    self._await_recovery = False
                return False
            if in_free_fall:
                self._state = FallStage.FREE_FALL
                self._sequence_samples = 1
                self._free_fall_samples = 1
                self._free_fall_min_g = svm
            return False

        # Any non-IDLE state is part of a candidate signature, so the overall
        # clock is running.
        self._sequence_samples += 1
        if self._sequence_samples > self.max_sequence_samples:
            self._clear_sequence()
            return False

        # --- Stage 1: free-fall -------------------------------------------
        if self._state is FallStage.FREE_FALL:
            if in_free_fall:
                self._free_fall_samples += 1
                self._free_fall_min_g = min(self._free_fall_min_g, svm)
                if self._free_fall_samples > self.free_fall_max_samples:
                    # Too long to be a fall - this is a jump, a dropped
                    # device, or the unit being thrown. Reject, and wait for
                    # the signal to recover before arming again.
                    self._clear_sequence()
                    self._await_recovery = True
                return False
            # The dip has just ended. Was it long enough to be real?
            if self._free_fall_samples < self.free_fall_min_samples:
                self._clear_sequence()   # one or two stray samples: noise
                return False
            self._state = FallStage.AWAIT_IMPACT
            self._await_impact_samples = 0
            # fall through: this same sample might be the impact

        # --- Stage 2: impact ----------------------------------------------
        if self._state is FallStage.AWAIT_IMPACT:
            self._await_impact_samples += 1
            if svm >= self.impact_threshold_g:
                self._state = FallStage.SETTLING
                self._settle_samples_seen = 0
                self._peak_impact_g = svm
                self._impact_timestamp = timestamp
                return False
            if self._await_impact_samples >= self.impact_search_samples:
                # Free-fall with no floor at the end of it. Most likely an arm
                # swing, or a fall that was caught.
                self._clear_sequence()
            return False

        # --- Between stages: let the impact ring out ----------------------
        if self._state is FallStage.SETTLING:
            # Keep tracking the peak: the true maximum of a spike usually
            # arrives a sample or two after it first crosses the threshold.
            self._peak_impact_g = max(self._peak_impact_g, svm)
            self._settle_samples_seen += 1
            if self._settle_samples_seen >= self.settle_samples:
                self._state = FallStage.STILLNESS
                self._stillness_samples_seen = 0
                self._stillness_active_samples = 0
            return False

        # --- Stage 3: post-fall stillness ---------------------------------
        if self._state is FallStage.STILLNESS:
            self._stillness_samples_seen += 1
            if abs(svm - GRAVITY_G) > self.stillness_tolerance_g:
                self._stillness_active_samples += 1
                if (
                    self._stillness_active_samples
                    > self.stillness_active_allowance_samples
                ):
                    # They are still moving: this was vigorous activity, not a
                    # fall. This is the branch that rejects running.
                    self._clear_sequence()
                    return False
            if self._stillness_samples_seen >= self.stillness_samples:
                self._confirm_fall(timestamp)
                return True
            return False

        return False

    def _confirm_fall(self, timestamp: float) -> None:
        """Raise the latch, record the event, and re-arm the state machine."""
        self._last_event = FallEvent(
            detected_at=timestamp,
            impact_at=self._impact_timestamp,
            peak_impact_g=self._peak_impact_g,
            free_fall_min_g=self._free_fall_min_g,
            free_fall_duration_s=self._free_fall_samples * self.sample_period_s,
            sample_index=self._samples_seen,
        )
        self._fall_count += 1
        self._alert_active = True
        # Back to IDLE so a second fall can still be detected while the first
        # alert is up; the latch itself is untouched by _clear_sequence.
        self._clear_sequence()


# ===========================================================================
# Self-test
# ===========================================================================
# Everything below here exists only to prove the rule separates real falls
# from ordinary movement, using synthetic 100 Hz signals built from the
# physics described at the top of the file. Run it with:
#
#     cd dashboard
#     python falldetection.py
#
# Synthetic signals are not a substitute for testing against the real device -
# they are a regression net, so that tuning a threshold cannot quietly break
# the walking or running cases.

def _make_sample(index: int, az: float, ax: float = 0.0, ay: float = 0.0,
                 fs: float = SAMPLING_FREQ_HZ) -> dict:
    """Build one sample dict of the shape the live data source will produce."""
    return {
        "timestamp": index / fs,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "IR_Value": 0,
    }


def _rng(seed: int):
    """Deterministic small-noise generator, so the tests never flake."""
    import random

    rand = random.Random(seed)
    return lambda scale=0.02: rand.uniform(-scale, scale)


def _still(n: int, start: int, noise, level: float = 1.0) -> list:
    """``n`` samples of a supported, motionless unit: SVM ~= 1 g."""
    return [_make_sample(start + i, level + noise(), noise(0.01), noise(0.01))
            for i in range(n)]


def _periodic(n: int, start: int, freq_hz: float, amp: float,
              harmonic: float, lateral: float, noise) -> list:
    """``n`` samples of rhythmic gait: 1 g plus a swing at ``freq_hz``.

    Walking and running both look like this at the wrist - a roughly
    sinusoidal swing about 1 g, with a second harmonic from the footfall. The
    only difference is cadence and amplitude, which is the point: the rule
    must not separate them from a fall by amplitude alone.
    """
    out = []
    for i in range(n):
        t = i / SAMPLING_FREQ_HZ
        phase = 2.0 * math.pi * freq_hz * t
        az = 1.0 + amp * math.sin(phase) + harmonic * math.sin(2.0 * phase)
        ax = lateral * math.sin(phase + 0.7)
        out.append(_make_sample(start + i, az + noise(), ax + noise(), noise()))
    return out


def _fall(noise, dip_ms: int = 60, dip_g: float = 0.30,
          peak_g: float = 3.0, still_ms: int = 1200,
          pre_ms: int = 1000) -> list:
    """A textbook fall: quiet, a dip below 0.6 g, a spike, then a quiet second."""
    fs = SAMPLING_FREQ_HZ
    out = _still(int(pre_ms * fs / 1000), 0, noise)
    i = len(out)

    for _ in range(int(dip_ms * fs / 1000)):          # stage 1
        out.append(_make_sample(i, dip_g + noise(0.03)))
        i += 1

    # Stage 2: a short asymmetric spike. The rise is fast (the floor is hard),
    # the decay is slower (flesh and strap ring down).
    for level in (1.2, 2.2, peak_g, peak_g * 0.8, 1.5, 1.1):
        out.append(_make_sample(i, level + noise(0.03)))
        i += 1

    out.extend(_still(int(still_ms * fs / 1000), i, noise))  # stage 3
    return out


def _spike_only(noise, peak_g: float = 3.0) -> list:
    """Sitting down hard / banging the desk: an impact with no free-fall first."""
    fs = SAMPLING_FREQ_HZ
    out = _still(fs, 0, noise)
    i = len(out)
    for level in (1.4, 2.0, peak_g, peak_g * 0.7, 1.3, 1.0):
        out.append(_make_sample(i, level + noise(0.03)))
        i += 1
    out.extend(_still(int(1.5 * fs), i, noise))
    return out


def _free_fall_only(noise) -> list:
    """An arm swing, or a fall that was caught: a dip with no floor after it."""
    fs = SAMPLING_FREQ_HZ
    out = _still(fs, 0, noise)
    i = len(out)
    for _ in range(6):                      # 60 ms under 0.6 g
        out.append(_make_sample(i, 0.30 + noise(0.03)))
        i += 1
    for level in (0.7, 0.9, 1.1, 1.3, 1.2, 1.05):   # gentle recovery, no spike
        out.append(_make_sample(i, level + noise(0.03)))
        i += 1
    out.extend(_still(int(1.5 * fs), i, noise))
    return out


def _jump(noise) -> list:
    """A hop: genuine free-fall, a genuine landing impact, then standing still.

    The nastiest false positive there is - every stage of the signature is
    present. Only the *length* of the airborne phase gives it away.
    """
    fs = SAMPLING_FREQ_HZ
    out = _still(fs, 0, noise)
    i = len(out)
    for _ in range(int(0.28 * fs)):         # 280 ms airborne
        out.append(_make_sample(i, 0.10 + noise(0.03)))
        i += 1
    for level in (1.5, 2.8, 3.4, 2.5, 1.4, 1.05):   # landing
        out.append(_make_sample(i, level + noise(0.03)))
        i += 1
    out.extend(_still(int(1.5 * fs), i, noise))
    return out


def _set_device_down(noise) -> list:
    """Putting the unit briskly on a table: shallow dip, a knock, then stillness.

    The hand decelerates the unit on the way down, so it is never truly
    unsupported and the dip stops short of 0.6 g.
    """
    fs = SAMPLING_FREQ_HZ
    out = _still(fs, 0, noise)
    i = len(out)
    for _ in range(int(0.10 * fs)):         # 100 ms at 0.75 g: lowered, not dropped
        out.append(_make_sample(i, 0.75 + noise(0.03)))
        i += 1
    for level in (1.6, 2.8, 2.2, 1.3, 1.0):  # knock on the table
        out.append(_make_sample(i, level + noise(0.03)))
        i += 1
    out.extend(_still(int(1.5 * fs), i, noise))
    return out


def _run_case(name: str, samples: list, expect_fall: bool,
              must_reach: Optional[FallStage] = None) -> None:
    """Feed one synthetic signal to a fresh detector and assert the outcome.

    ``must_reach`` asserts that the signal actually exercised a given stage.
    Without it a "no false alarm" test can pass for an embarrassing reason -
    a signal so tame it never left IDLE proves nothing about the rule. Naming
    the stage that is expected to do the rejecting means a future threshold
    change cannot make these tests pass by accident.
    """
    detector = FallDetector()
    confirmed = False
    stages_seen = set()
    for sample in samples:                 # one at a time: exercises update()
        if detector.update(sample):
            confirmed = True
        stages_seen.add(detector.state)

    svms = [signal_vector_magnitude(s) for s in samples]
    shape = f"SVM {min(svms):.2f}-{max(svms):.2f} g over {len(samples)} samples"

    if must_reach is not None:
        assert must_reach in stages_seen, (
            f"{name}: expected the signal to reach stage {must_reach.value}, "
            f"but it only reached {sorted(s.value for s in stages_seen)} - "
            f"this test is no longer testing what it claims to"
        )

    if expect_fall:
        assert confirmed, f"{name}: expected a fall, detector saw none ({shape})"
        assert detector.alert_active, f"{name}: fall confirmed but alert not latched"
        assert detector.last_event is not None, f"{name}: no event recorded"
        detail = detector.last_event.summary()
    else:
        assert not confirmed, (
            f"{name}: FALSE ALARM - {shape}; "
            f"{detector.last_event.summary() if detector.last_event else ''}"
        )
        assert not detector.alert_active, f"{name}: alert latched without a fall"
        detail = f"no alert ({shape})"

    print(f"  PASS  {name:<26} {detail}")


def _test_latching() -> None:
    """The alert must survive quiet time and only clear when told to."""
    noise = _rng(99)
    detector = FallDetector()
    assert not detector.alert_active, "a fresh detector must not be alarming"

    assert detector.update_batch(_fall(noise)), "latch test: fall not detected"
    assert detector.alert_active, "latch test: alert should be raised"
    event = detector.last_event
    assert event is not None and event.peak_impact_g >= IMPACT_THRESHOLD_G

    # Minutes of calm must not clear it: a person on the floor cannot press
    # the button, and the dashboard must still be red when help arrives.
    detector.update_batch(_still(SAMPLING_FREQ_HZ * 5, 10_000, noise))
    assert detector.alert_active, "latch test: alert cleared itself over time"
    assert detector.state is FallStage.IDLE, "latch test: machine should be re-armed"

    detector.dismiss()
    assert not detector.alert_active, "latch test: dismiss() did not clear the alert"
    # The history must outlive the acknowledgement.
    assert detector.fall_count == 1, "latch test: fall count lost on dismiss"
    assert detector.last_event is event, "latch test: event record lost on dismiss"

    detector.reset()
    assert detector.fall_count == 0 and detector.last_event is None, (
        "latch test: reset() should wipe the history as well as the alert"
    )
    print(f"  PASS  {'latch / dismiss / reset':<26} "
          f"alert held through 5 s of calm, cleared on request")


def _test_stream_handling() -> None:
    """Batching and missing timestamps must not change any decision.

    The dashboard will drain a queue on a timer, so it hands over a clump of
    samples at a time, and the clumps are whatever size the serial link
    produced. That must give bit-for-bit the same answer as feeding samples
    individually, otherwise the alert would depend on network timing.
    """
    signal = _fall(_rng(21))

    one_at_a_time = FallDetector()
    for sample in signal:
        one_at_a_time.update(sample)

    in_chunks = FallDetector()
    for start in range(0, len(signal), 7):       # 7: a deliberately awkward size
        in_chunks.update_batch(signal[start:start + 7])

    assert one_at_a_time.alert_active and in_chunks.alert_active, (
        "stream test: the fall must be detected both ways"
    )
    assert (one_at_a_time.last_event.sample_index
            == in_chunks.last_event.sample_index), (
        "stream test: batching changed WHEN the fall was confirmed"
    )

    # Nothing in the rule reads the clock, so a stream with no timestamp field
    # at all must still work; only the reported time is synthesised.
    stripped = [{k: v for k, v in s.items() if k != "timestamp"} for s in signal]
    assert FallDetector().update_batch(stripped), (
        "stream test: detection must not require a timestamp field"
    )
    print(f"  PASS  {'batching / no timestamps':<26} "
          f"chunks of 7 and timestamp-less samples give identical decisions")


def _test_fall_variants() -> None:
    """The whole literature range of free-fall durations must be detected.

    A rule that only fires for the one synthetic fall written above would be
    worthless. These sweep the 30-100 ms range quoted by the literature (plus
    the 150 ms cap) and a barely-over-threshold impact.
    """
    for dip_ms, peak_g in ((30, 3.0), (40, 2.6), (60, 3.0),
                           (100, 4.5), (150, 3.0)):
        detector = FallDetector()
        assert detector.update_batch(
            _fall(_rng(22), dip_ms=dip_ms, peak_g=peak_g)
        ), f"variant test: missed a fall with a {dip_ms} ms dip and {peak_g} g impact"
    print(f"  PASS  {'fall variants':<26} "
          f"dips of 30-150 ms and impacts of 2.6-4.5 g all detected")


def _test_clipped_impact() -> None:
    """A fall as the real device actually reports it: clipped at +/-2 g.

    This is the case the 1.8 g threshold exists for, and the one the old
    2.5 g threshold could never pass. The firmware leaves the MPU6050 on its
    default +/-2 g range, so however hard the floor is hit, nothing above 2 g
    ever leaves the device. Here every axis is clamped the way the sensor
    clamps it, and the fall must still be found.

    If someone later raises IMPACT_THRESHOLD_G back above 2 g without also
    fixing register 0x1C in the firmware, this test is what fails.
    """
    def clamp(sample: dict) -> dict:
        out = dict(sample)
        for axis in ("ax", "ay", "az"):
            out[axis] = max(-2.0, min(2.0, out[axis]))
        return out

    signal = [clamp(sample) for sample in _fall(_rng(31), peak_g=6.0)]

    peak = max(math.sqrt(s["ax"] ** 2 + s["ay"] ** 2 + s["az"] ** 2)
               for s in signal)
    assert peak <= 2.0 + 1e-9, (
        f"clipping test: the clamped signal still peaks at {peak:.2f} g, "
        "so it is not testing what it claims to"
    )

    detector = FallDetector()
    assert detector.update_batch(signal), (
        f"clipping test: a 6 g fall clipped to {peak:.2f} g was missed - "
        f"IMPACT_THRESHOLD_G is {IMPACT_THRESHOLD_G} g, which a +/-2 g "
        "accelerometer can never report"
    )
    print(f"  PASS  {'impact clipped at 2 g':<26} "
          f"a 6 g fall clipped to {peak:.2f} g is still detected")


def _test_rate_independence() -> None:
    """The rule must still work if the device is re-flashed at another rate.

    Nothing below is hard-coded to 100 Hz: every duration is turned into a
    sample count at construction time, so doubling the rate doubles the counts
    and the rule still means the same number of milliseconds.

    Note the one wrinkle: a sample count has to be a whole number, so at low
    rates short durations round. At 50 Hz, 30 ms is 1.5 samples and becomes 2
    (= 40 ms), which is why the check below compares 100 Hz with 200 Hz, where
    the arithmetic is exact. Anyone running the unit below ~50 Hz should know
    that the free-fall window gets coarse.
    """
    base = FallDetector(sampling_freq_hz=100)
    fast = FallDetector(sampling_freq_hz=200)
    assert base.free_fall_min_samples == 3, "30 ms at 100 Hz is 3 samples"
    assert fast.free_fall_min_samples == 2 * base.free_fall_min_samples
    assert fast.free_fall_max_samples == 2 * base.free_fall_max_samples
    assert fast.stillness_samples == 2 * base.stillness_samples
    assert fast.impact_search_samples == 2 * base.impact_search_samples
    assert fast.max_sequence_samples == 2 * base.max_sequence_samples
    print(f"  PASS  {'durations track the rate':<26} "
          f"{SAMPLING_FREQ_HZ} Hz: {FallDetector().stillness_samples} samples "
          f"of stillness; 200 Hz: {fast.stillness_samples}")


def _self_test() -> None:
    print("falldetection.py self-test")
    print(f"\nSynthetic signals at {SAMPLING_FREQ_HZ} Hz. Rule: free-fall "
          f"<{FREE_FALL_THRESHOLD_G} g for "
          f"{FREE_FALL_MIN_DURATION_S * 1000:.0f}-"
          f"{FREE_FALL_MAX_DURATION_S * 1000:.0f} ms, then impact "
          f">{IMPACT_THRESHOLD_G} g, then {STILLNESS_DURATION_S:.1f} s within "
          f"+/-{STILLNESS_TOLERANCE_G} g of 1 g.\n")

    # The `must_reach` argument records *why* each case is rejected, which is
    # as much the point as the rejection itself.
    _run_case("real fall", _fall(_rng(1)), expect_fall=True,
              must_reach=FallStage.STILLNESS)
    # Walking swings the wrist below 0.6 g at the bottom of the arc, so it does
    # enter stage 1 - it is stopped by the absence of a 1.8 g impact.
    _run_case("walking, 1.7 Hz",
              _periodic(600, 0, 1.7, 0.45, 0.20, 0.25, _rng(2)),
              expect_fall=False, must_reach=FallStage.AWAIT_IMPACT)
    # Running clears stages 1 AND 2 on every stride - free-fall between
    # footfalls, then a hard heel strike. It is stage 3 that saves us, because
    # the next stride arrives long before a quiet second does.
    _run_case("running, 2.8 Hz",
              _periodic(900, 0, 2.8, 1.55, 0.45, 0.60, _rng(3)),
              expect_fall=False, must_reach=FallStage.STILLNESS)
    _run_case("sitting still", _still(SAMPLING_FREQ_HZ * 5, 0, _rng(4)),
              expect_fall=False)
    # Rejected at stage 1: a hard landing with nothing before it.
    _run_case("impact, no free-fall", _spike_only(_rng(5)), expect_fall=False)
    # Rejected at stage 2: the 300 ms impact window expires.
    _run_case("free-fall, no impact", _free_fall_only(_rng(6)),
              expect_fall=False, must_reach=FallStage.AWAIT_IMPACT)
    # Rejected at stage 1 by the MAXIMUM free-fall duration: 280 ms airborne
    # is too long to be a fall.
    _run_case("jumping", _jump(_rng(7)), expect_fall=False,
              must_reach=FallStage.FREE_FALL)
    # Rejected at stage 1 by the threshold: a lowered hand never reaches 0.6 g.
    _run_case("device set down", _set_device_down(_rng(8)), expect_fall=False)

    _test_latching()
    _test_fall_variants()
    _test_clipped_impact()
    _test_stream_handling()
    _test_rate_independence()

    print("\nAll checks passed.")


if __name__ == "__main__":
    _self_test()
