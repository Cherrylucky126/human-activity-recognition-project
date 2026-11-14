"""Data-source layer for the HAR wrist unit dashboard.

The hardware
------------
The wrist unit is an ESP32 carrying two sensors:

* an **MPU6050** inertial measurement unit, which reports linear acceleration
  on three axes (``ax``, ``ay``, ``az``, in *g*, where 1 g = 9.81 m/s^2) and
  rotation rate on three axes (``gx``, ``gy``, ``gz``, in degrees per second);
* a **MAX30102** photoplethysmography (PPG) sensor, which shines infrared
  light into the skin and counts how much comes back (``IR_Value``, a raw
  18-bit ADC count). Blood volume in the wrist rises and falls with every
  heartbeat, so that count wobbles at the pulse rate.

The board streams those seven numbers to the PC over USB serial.

Why this file exists
--------------------
The dashboard has to run on a laptop with **no board plugged in** - during
development, on the demo machine, in a lecture hall. So every consumer of
sensor data talks to the abstract :class:`SensorSource` interface, and a
simulator (:class:`SimulatedSource`) stands in for the hardware. When the
real serial reader is written it becomes another :class:`SensorSource`
subclass and nothing downstream changes.

What this file deliberately does NOT do
---------------------------------------
No windowing, no feature extraction, no model inference, no heart-rate
algorithm, no fall *detection*. This module only *produces samples*. The
processing layers are someone else's files. It also imports neither Dash nor
Plotly nor numpy, so it can be imported and tested on a bare Python install.

Quick start
-----------
::

    from datasource import SimulatedSource

    src = SimulatedSource()
    src.start()
    src.set_activity("walking")
    samples = src.read()          # every ~100 ms; returns ~10 Sample objects
    src.trigger_fall()            # inject a scripted fall for testing
    src.stop()

Self-test::

    cd dashboard
    python datasource.py

Units, at a glance
------------------
=============  =========================================================
``t``          seconds, from :func:`time.monotonic` (never goes backwards)
``ax ay az``   g (gravity included - a resting sensor reads ~1 g total)
``gx gy gz``   degrees per second
``IR_Value``   raw PPG counts (unitless ADC counts, ~0 .. 262143)
=============  =========================================================
"""

from __future__ import annotations

import math
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

# The dashboard is run as a flat folder of scripts (``python app.py`` from
# inside ``dashboard/``), so the plain import is the normal case. The relative
# import is a fallback in case someone later turns the folder into a package.
try:  # pragma: no cover - import plumbing
    from config import (
        BUFFER_SAMPLES,
        IR_COLUMN,
        SAMPLING_FREQ_HZ,
        SENSOR_COLUMNS,
    )
except ImportError:  # pragma: no cover - import plumbing
    from .config import (  # type: ignore[no-redef]
        BUFFER_SAMPLES,
        IR_COLUMN,
        SAMPLING_FREQ_HZ,
        SENSOR_COLUMNS,
    )

__all__ = [
    "Sample",
    "SensorSource",
    "SimulatedSource",
    "ActivityProfile",
    "ACTIVITIES",
    "FALL_FREEFALL_G",
    "FALL_FREEFALL_SEC",
    "FALL_IMPACT_PEAK_G",
    "FALL_IMPACT_SEC",
    "FALL_SETTLE_SEC",
    "FALL_STILL_SEC",
    "FALL_TOTAL_SEC",
]

TWO_PI = 2.0 * math.pi

# --- Sensor hardware limits -------------------------------------------------
# An MPU6050 is configured for a fixed measurement range; anything stronger
# than that clips. We model +/-8 g and +/-500 deg/s, which are the ranges that
# comfortably cover walking, running and a fall impact without saturating.
ACCEL_RANGE_G = 8.0
GYRO_RANGE_DPS = 500.0

# A MAX30102 reading is an 18-bit unsigned ADC count, so it cannot go below 0
# or above 2**18 - 1.
IR_MAX_COUNT = 262143

# --- PPG (heart-rate) signal shape ------------------------------------------
# A raw PPG trace is a large steady "DC" level - light bouncing off skin,
# bone and non-pulsing tissue - with a tiny "AC" ripple riding on it that is
# the actual heartbeat. The ripple is typically well under 2% of the DC level,
# which is why naive plotting of IR_Value looks like a flat line.
PPG_DC_COUNTS = 85_000.0      # baseline reflected light
PPG_AC_COUNTS = 950.0         # peak cardiac ripple (~1.1% of DC)
PPG_NOISE_COUNTS = 70.0       # electrical + photon shot noise, 1 sigma
PPG_RESP_COUNTS = 420.0       # slow baseline rise/fall with breathing

# Physiological heart-rate limits. The band 0.75-3.0 Hz (45-180 BPM) is what
# downstream heart-rate code is expected to search, so the simulator never
# produces a pulse outside it.
HR_MIN_BPM = 45.0
HR_MAX_BPM = 180.0

# How quickly simulated state follows a change of activity (seconds).
PROFILE_BLEND_SEC = 0.6   # body mechanics change over about half a second
CADENCE_TAU_SEC = 1.0     # step rate settles into its new rhythm
HR_TAU_SEC = 8.0          # heart rate lags exertion by several seconds
STARTLE_TAU_SEC = 15.0    # adrenaline after a fall decays slowly

# --- Scripted fall timeline -------------------------------------------------
# These are the numbers the fall *detector* (written by another team member)
# has to catch, so they are module constants rather than magic numbers buried
# in a function - the detector's thresholds can be sanity-checked against
# them.
#
#   1. FREE FALL   the wrist stops being supported, so the accelerometer
#                  briefly reads far less than 1 g (in true free fall it would
#                  read 0 g; a real tumbling wrist reads a few tenths).
#   2. IMPACT      the body hits the floor and the sensor sees a short, very
#                  large spike, with high-frequency ringing on top.
#   3. SETTLE      a couple of small bounces as limbs come to rest.
#   4. STILLNESS   the person is lying on the floor and not moving - this is
#                  what separates a fall from merely sitting down hard.
FALL_FREEFALL_SEC = 0.070     # inside the 30-100 ms window a detector expects
FALL_FREEFALL_G = 0.32        # well below the usual 0.6 g free-fall threshold
FALL_IMPACT_SEC = 0.085
FALL_IMPACT_PEAK_G = 3.9      # well above the usual 2.5 g impact threshold;
                              # the ringing on top can push the measured peak
                              # to roughly 4.4 g on some runs
FALL_SETTLE_SEC = 0.195
FALL_STILL_SEC = 1.050        # "about a second of near-stillness"
FALL_TOTAL_SEC = (
    FALL_FREEFALL_SEC + FALL_IMPACT_SEC + FALL_SETTLE_SEC + FALL_STILL_SEC
)
FALL_STILL_NOISE_G = 0.006    # how quiet "near-stillness" is
FALL_PPG_ARTEFACT = 6_000.0   # the PPG sensor is thrown off its contact point
FALL_HR_STARTLE_BPM = 25.0    # heart rate jumps after the shock


def _clamp(value: float, low: float, high: float) -> float:
    """Keep ``value`` inside ``[low, high]`` (used for sensor saturation)."""
    return low if value < low else (high if value > high else value)


def _unit(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Scale a 3-vector to length 1.

    Used for gravity directions: an activity profile says *which way* gravity
    points through the device, and this makes sure its magnitude is exactly
    1 g rather than whatever the hand-written numbers happened to add up to.
    """
    x, y, z = vec
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / norm, y / norm, z / norm)


def _lerp(a: float, b: float, m: float) -> float:
    """Linear blend: ``m=0`` gives ``a``, ``m=1`` gives ``b``."""
    return a + (b - a) * m


def _lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    m: float,
) -> tuple[float, float, float]:
    """Blend two 3-vectors component by component."""
    return (_lerp(a[0], b[0], m), _lerp(a[1], b[1], m), _lerp(a[2], b[2], m))


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Sample:
    """A single instant of sensor data - one row of the eventual dataset.

    At 100 Hz, one of these is produced every 10 ms. Field names match
    :data:`config.SENSOR_COLUMNS` and :data:`config.IR_COLUMN` exactly, so
    ``[s.as_dict() for s in samples]`` can be handed straight to
    ``pandas.DataFrame`` and the columns will line up with what the training
    code used.

    Attributes
    ----------
    t:
        Timestamp in seconds on the :func:`time.monotonic` clock. Monotonic
        means it only ever moves forward, even if the PC's wall clock is
        changed or the machine crosses a daylight-saving boundary - so
        differences between timestamps are always trustworthy.
    ax, ay, az:
        Acceleration in g, *including gravity*. A device lying perfectly still
        does not read zero; it reads 1 g spread across whichever axes are
        tilted towards the floor.
    gx, gy, gz:
        Rotation rate in degrees per second about each axis. Zero when the
        wrist is not turning, regardless of orientation.
    IR_Value:
        Raw infrared count from the PPG sensor. Big number, small wobble: the
        heartbeat is the wobble, not the number.
    """

    t: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    IR_Value: float

    def as_dict(self) -> dict[str, float]:
        """Return the sample as a plain dict keyed by the config column names.

        Key order is ``t`` followed by :data:`config.SENSOR_COLUMNS` then
        :data:`config.IR_COLUMN`.
        """
        return {
            "t": self.t,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "gx": self.gx,
            "gy": self.gy,
            "gz": self.gz,
            IR_COLUMN: self.IR_Value,
        }

    @property
    def accel_magnitude(self) -> float:
        """Length of the acceleration vector, in g.

        This is the orientation-independent "how hard is this thing being
        shaken" number: about 1.0 when still (gravity only), below 0.6 in free
        fall, above 2.5 on a hard impact. Handy for plots and for eyeballing
        the fall signature.
        """
        return math.sqrt(self.ax ** 2 + self.ay ** 2 + self.az ** 2)

    @property
    def gyro_magnitude(self) -> float:
        """Length of the rotation-rate vector, in degrees per second."""
        return math.sqrt(self.gx ** 2 + self.gy ** 2 + self.gz ** 2)


# ---------------------------------------------------------------------------
# The interface every data source implements
# ---------------------------------------------------------------------------
class SensorSource(ABC):
    """Abstract base class: anything that can hand the dashboard samples.

    Two implementations are expected to exist:

    * :class:`SimulatedSource` - synthetic data, always available.
    * a serial reader for the real ESP32 - someone else's file; it subclasses
      this and nothing in the dashboard needs to know which one it got.

    The contract, in full:

    * :meth:`start` opens the source (or starts the simulated clock).
    * :meth:`read` is called repeatedly - roughly every 100 ms by the Dash
      interval callback - and returns **everything that arrived since the
      previous call**, oldest first. It never blocks waiting for data and
      returns an empty list when nothing new has arrived.
    * :meth:`stop` closes the source. Calling it twice is harmless.
    * :attr:`sample_rate_hz` reports the nominal rate, for code that needs to
      convert sample counts into seconds.

    It also works as a context manager::

        with SimulatedSource() as src:
            samples = src.read()
    """

    @property
    @abstractmethod
    def sample_rate_hz(self) -> float:
        """Nominal samples per second per channel (100 Hz for this project)."""

    @abstractmethod
    def start(self) -> None:
        """Begin producing data. Safe to call on an already-started source."""

    @abstractmethod
    def stop(self) -> None:
        """Stop producing data and release anything that was opened."""

    @abstractmethod
    def read(self) -> list[Sample]:
        """Return all samples that arrived since the last call, oldest first.

        Returns an empty list if nothing new has arrived, or if the source has
        not been started. Must not block.
        """

    @property
    def is_running(self) -> bool:
        """Whether :meth:`start` has been called and :meth:`stop` has not."""
        return getattr(self, "_running", False)

    def __enter__(self) -> "SensorSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# What each activity looks like to a wrist sensor
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ActivityProfile:
    """The recipe for one activity's sensor signature.

    Every activity is modelled the same way: a constant gravity vector (which
    way is down through the device), a periodic body rhythm (steps, or slow
    postural sway when standing still), random sensor noise, and a heart rate.
    Only the numbers change between activities.

    Attributes
    ----------
    gravity:
        Unit vector times 1 g, i.e. where the 1 g of gravity lands on the
        three axes for this posture. Sitting with the forearm flat puts it on
        ``az``; standing with the arm hanging puts it on ``-ay``; lying with
        the wrist on its side puts it on ``ax``.
    cadence_range_hz:
        Cycles per second of the dominant body rhythm. For gait this is the
        step rate (two steps per stride, ~1.6-1.9 Hz for a walk). For static
        postures it is slow postural sway, a few tenths of a hertz.
    accel_amp_g:
        Peak acceleration swing per axis, in g, on top of gravity.
    gyro_amp_dps:
        Peak rotation-rate swing per axis, in degrees per second. The wrist
        rotates a lot during arm swing, which is why these numbers are large
        compared with the accelerometer's.
    harmonic2:
        How much energy sits at *twice* the cadence. Real gait is not a clean
        sine wave: each step has a sharp heel-strike, which shows up as a
        second harmonic. Stair descent has the most.
    accel_noise_g, gyro_noise_dps:
        Standard deviation of the random sensor noise. The MPU6050's own noise
        floor is a few milli-g; the larger values for running represent real
        soft-tissue jiggle rather than electrical noise.
    heart_rate_bpm:
        Steady-state heart rate for this activity. The simulator eases towards
        it rather than jumping.
    ppg_motion_counts:
        Peak motion artefact in the PPG channel. Movement shifts the sensor
        against the skin, which corrupts the optical signal at the *step*
        frequency - the classic reason wrist heart-rate readings go wrong
        while running.
    intensity:
        0.0 (lying still) to 1.0 (running). Drives breathing rate and the
        drop in pulse amplitude when the sensor is being jostled.
    description:
        Plain-language note for anyone reading the dashboard code.
    """

    gravity: tuple[float, float, float]
    cadence_range_hz: tuple[float, float]
    accel_amp_g: tuple[float, float, float]
    gyro_amp_dps: tuple[float, float, float]
    harmonic2: float
    accel_noise_g: float
    gyro_noise_dps: float
    heart_rate_bpm: float
    ppg_motion_counts: float
    intensity: float
    description: str = ""


#: Every activity the simulator can produce, keyed by canonical name.
#:
#: The three static postures differ only in which axis carries gravity - that
#: is genuinely the whole difference a wrist accelerometer sees between
#: sitting, standing and lying, and it is why classifiers confuse them when
#: orientation drifts.
ACTIVITIES: dict[str, ActivityProfile] = {
    "sitting": ActivityProfile(
        gravity=(0.02, 0.08, 0.99),
        cadence_range_hz=(0.20, 0.35),
        accel_amp_g=(0.010, 0.008, 0.012),
        gyro_amp_dps=(0.8, 0.6, 0.8),
        harmonic2=0.15,
        accel_noise_g=0.012,
        gyro_noise_dps=0.45,
        heart_rate_bpm=72.0,
        ppg_motion_counts=60.0,
        intensity=0.02,
        description="Forearm resting roughly flat; gravity almost all on az.",
    ),
    "standing": ActivityProfile(
        gravity=(0.05, -0.97, 0.12),
        cadence_range_hz=(0.25, 0.40),
        accel_amp_g=(0.025, 0.020, 0.030),
        gyro_amp_dps=(1.8, 1.4, 1.6),
        harmonic2=0.20,
        accel_noise_g=0.013,
        gyro_noise_dps=0.50,
        heart_rate_bpm=78.0,
        ppg_motion_counts=120.0,
        intensity=0.08,
        description="Arm hanging, forearm vertical; gravity mostly on -ay, "
                    "with slow postural sway.",
    ),
    "lying": ActivityProfile(
        gravity=(0.98, 0.06, -0.10),
        cadence_range_hz=(0.20, 0.30),
        accel_amp_g=(0.008, 0.010, 0.008),
        gyro_amp_dps=(0.5, 0.5, 0.5),
        harmonic2=0.10,
        accel_noise_g=0.010,
        gyro_noise_dps=0.40,
        heart_rate_bpm=62.0,
        ppg_motion_counts=40.0,
        intensity=0.0,
        description="Wrist on its side at rest; gravity mostly on ax. The "
                    "only movement is breathing.",
    ),
    "walking": ActivityProfile(
        gravity=(0.08, -0.95, 0.15),
        cadence_range_hz=(1.60, 1.90),
        accel_amp_g=(0.35, 0.45, 0.30),
        gyro_amp_dps=(70.0, 45.0, 55.0),
        harmonic2=0.45,
        accel_noise_g=0.030,
        gyro_noise_dps=2.0,
        heart_rate_bpm=100.0,
        ppg_motion_counts=500.0,
        intensity=0.45,
        description="Arm swing at ~1.6-1.9 steps per second, with a clear "
                    "heel-strike second harmonic.",
    ),
    "running": ActivityProfile(
        gravity=(0.10, -0.94, 0.18),
        cadence_range_hz=(2.50, 3.00),
        accel_amp_g=(1.30, 1.90, 1.10),
        gyro_amp_dps=(240.0, 180.0, 200.0),
        harmonic2=0.50,
        accel_noise_g=0.080,
        gyro_noise_dps=6.0,
        heart_rate_bpm=150.0,
        ppg_motion_counts=1800.0,
        intensity=1.0,
        description="2.5-3 steps per second with several times the walking "
                    "amplitude; the PPG channel is badly corrupted.",
    ),
    "walking_upstairs": ActivityProfile(
        gravity=(0.12, -0.92, 0.24),
        cadence_range_hz=(1.30, 1.60),
        accel_amp_g=(0.45, 0.60, 0.40),
        gyro_amp_dps=(85.0, 60.0, 70.0),
        harmonic2=0.40,
        accel_noise_g=0.035,
        gyro_noise_dps=2.5,
        heart_rate_bpm=118.0,
        ppg_motion_counts=700.0,
        intensity=0.62,
        description="Slower cadence than level walking but more effort: "
                    "bigger swings and a much higher heart rate.",
    ),
    "walking_downstairs": ActivityProfile(
        gravity=(0.10, -0.93, 0.20),
        cadence_range_hz=(1.50, 1.80),
        accel_amp_g=(0.55, 0.70, 0.45),
        gyro_amp_dps=(95.0, 70.0, 80.0),
        harmonic2=0.60,
        accel_noise_g=0.040,
        gyro_noise_dps=3.0,
        heart_rate_bpm=105.0,
        ppg_motion_counts=650.0,
        intensity=0.52,
        description="Faster and harder-landing than going up - the strongest "
                    "second harmonic of any activity - but less cardiac load.",
    ),
}

#: Friendly spellings accepted by :meth:`SimulatedSource.set_activity`.
ACTIVITY_ALIASES: dict[str, str] = {
    "sit": "sitting",
    "seated": "sitting",
    "stand": "standing",
    "lie": "lying",
    "lay": "lying",
    "laying": "lying",
    "lying_down": "lying",
    "walk": "walking",
    "run": "running",
    "jog": "running",
    "jogging": "running",
    "upstairs": "walking_upstairs",
    "stairs_up": "walking_upstairs",
    "downstairs": "walking_downstairs",
    "stairs_down": "walking_downstairs",
}


def _canonical_activity(name: str) -> str:
    """Turn user-ish spellings into a key of :data:`ACTIVITIES`.

    Accepts any case and uses spaces, hyphens or underscores interchangeably,
    so ``"Walking Upstairs"``, ``"walking-upstairs"`` and ``"upstairs"`` all
    mean the same thing. Raises :class:`ValueError` listing the valid names if
    it cannot work out what was meant.
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = ACTIVITY_ALIASES.get(key, key)
    if key not in ACTIVITIES:
        valid = ", ".join(sorted(ACTIVITIES))
        raise ValueError(f"unknown activity {name!r}; choose one of: {valid}")
    return key


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class SimulatedSource(SensorSource):
    """Synthetic 100 Hz wrist-sensor data, generated in real time.

    How the timing works
    --------------------
    There is no background thread. The simulator remembers when it was started
    and how many samples it has handed out; when :meth:`read` is called it
    works out how many samples *should* have arrived by now and generates
    exactly that many on the spot. Called every 100 ms at 100 Hz, that is
    about 10 samples per call. This keeps the module single-threaded and
    deterministic, and means a slow dashboard callback cannot desynchronise
    the timestamps from the clock.

    If the consumer stalls for a long time (a breakpoint, a blocked UI), the
    backlog is capped at :data:`config.BUFFER_SAMPLES` and the older samples
    are dropped rather than returned in one enormous list - the same thing a
    real serial buffer does when it overflows. :attr:`dropped_samples` counts
    them.

    How the signals are built
    -------------------------
    Each sample is the sum of a few physically meaningful pieces:

    * **gravity** - a constant vector set by the posture, which is how an
      accelerometer tells sitting from standing from lying;
    * **body rhythm** - a periodic wave at the step cadence plus its second
      harmonic (heel-strike), scaled by the activity's amplitudes;
    * **sensor noise** - Gaussian, at a level in the right ballpark for an
      MPU6050;
    * **PPG** - a large steady baseline, a slow breathing wobble, the cardiac
      pulse in the 0.75-3 Hz band, motion artefact at the step frequency that
      grows with activity intensity, and photon/electrical noise.

    Switching activity with :meth:`set_activity` blends between profiles over
    about half a second instead of jumping, and heart rate eases towards its
    new value over several seconds, as a real one does.

    Parameters
    ----------
    activity:
        Which activity to start in. Default ``"sitting"``.
    sample_rate_hz:
        Samples per second. Defaults to :data:`config.SAMPLING_FREQ_HZ`
        (100 Hz) and should normally be left alone - the trained models expect
        100 Hz data.
    seed:
        Seed for the internal random generator. Pass an int for repeatable
        output in tests; leave as ``None`` for fresh randomness each run.
    max_backlog_samples:
        How many samples may queue up before the oldest are dropped. Defaults
        to :data:`config.BUFFER_SAMPLES` (10 s of data).
    """

    def __init__(
        self,
        activity: str = "sitting",
        sample_rate_hz: float = SAMPLING_FREQ_HZ,
        seed: int | None = None,
        max_backlog_samples: int = BUFFER_SAMPLES,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        self._fs = float(sample_rate_hz)
        self._dt = 1.0 / self._fs
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._max_backlog = max(1, int(max_backlog_samples))

        # Clock state.
        self._running = False
        self._t0 = 0.0
        self._index = 0           # samples generated since start()
        self._dropped = 0

        # Activity state. ``_profile`` is where we are heading, ``_prev`` is
        # where we came from, ``_blend`` walks 0 -> 1 between them.
        name = _canonical_activity(activity)
        self._activity = name
        self._profile = ACTIVITIES[name]
        self._prev_profile = ACTIVITIES[name]
        self._blend = 1.0

        # Oscillator phases, in radians. Keeping phase (rather than computing
        # sin(2*pi*f*t) from absolute time) means the waveform stays
        # continuous even when the frequency changes mid-stride.
        self._step_phase = self._rng.uniform(0.0, TWO_PI)
        self._pulse_phase = self._rng.uniform(0.0, TWO_PI)
        self._resp_phase = self._rng.uniform(0.0, TWO_PI)

        self._cadence = sum(self._profile.cadence_range_hz) / 2.0
        self._cadence_target = self._pick_cadence(self._profile)
        self._amp_mod = 1.0       # slow stride-to-stride amplitude variation
        self._ppg_drift = 0.0     # slow PPG baseline wander

        self._hr = self._profile.heart_rate_bpm
        self._hr_startle = 0.0

        # Scripted-fall state.
        self._fall_elapsed: float | None = None
        self._fall_pre_dir = _unit(self._profile.gravity)
        self._fall_post_dir = _unit(ACTIVITIES["lying"].gravity)

    # -- SensorSource interface --------------------------------------------
    @property
    def sample_rate_hz(self) -> float:
        """Nominal sample rate in hertz (100 Hz unless overridden)."""
        return self._fs

    def start(self) -> None:
        """Start the simulated clock.

        Resets the sample counter and timestamp origin, so a restart does not
        appear downstream as one giant gap in the data. Calling it on an
        already-running source simply restarts the clock.
        """
        with self._lock:
            self._t0 = time.monotonic()
            self._index = 0
            self._dropped = 0
            self._running = True

    def stop(self) -> None:
        """Stop producing samples. Harmless to call more than once."""
        with self._lock:
            self._running = False

    def read(self) -> list[Sample]:
        """Return every sample due since the previous call, oldest first.

        Empty list if the source is stopped or if less than one sample period
        has passed. Never blocks.
        """
        with self._lock:
            if not self._running:
                return []

            elapsed = time.monotonic() - self._t0
            due = int(elapsed * self._fs) - self._index
            if due <= 0:
                return []

            if due > self._max_backlog:
                # The consumer fell behind. Throw away the oldest backlog so
                # the dashboard shows "now" instead of replaying history.
                skip = due - self._max_backlog
                self._index += skip
                self._dropped += skip
                due = self._max_backlog

            return [self._generate_sample() for _ in range(due)]

    # -- simulator controls -------------------------------------------------
    @property
    def activity(self) -> str:
        """Canonical name of the activity currently being simulated.

        This is ground truth - useful for checking a classifier's output
        during development, since the real hardware obviously cannot tell you
        what the wearer is doing.
        """
        return self._activity

    @staticmethod
    def available_activities() -> tuple[str, ...]:
        """Names of every activity this source can simulate, sorted."""
        return tuple(sorted(ACTIVITIES))

    def set_activity(self, name: str) -> None:
        """Switch the simulated activity at runtime.

        Accepts any spelling :func:`_canonical_activity` understands, e.g.
        ``"walking"``, ``"Walking Upstairs"``, ``"downstairs"``. The change is
        blended in over about :data:`PROFILE_BLEND_SEC` seconds, and a fresh
        step cadence is drawn from the new activity's range, so switching
        looks like a person changing what they are doing rather than a jump
        cut.

        Raises
        ------
        ValueError
            If the name is not a known activity.
        """
        key = _canonical_activity(name)
        with self._lock:
            if key == self._activity:
                return
            # Freeze whatever mixture is on screen right now and blend from
            # there, so repeated quick switches stay continuous.
            self._prev_profile = self._effective_profile()
            self._profile = ACTIVITIES[key]
            self._activity = key
            self._blend = 0.0
            self._cadence_target = self._pick_cadence(self._profile)

    def set_heart_rate(self, bpm: float) -> None:
        """Force the simulated heart rate to ``bpm`` immediately.

        For exercising heart-rate panels and alarms without waiting for the
        activity to raise the rate naturally. Clamped to the physiological
        45-180 BPM band the simulator guarantees; the value then drifts back
        towards the current activity's resting rate.
        """
        with self._lock:
            self._hr = _clamp(float(bpm), HR_MIN_BPM, HR_MAX_BPM)

    @property
    def heart_rate_bpm(self) -> float:
        """The heart rate currently being written into ``IR_Value``.

        Ground truth for testing heart-rate extraction: whatever that code
        recovers from the PPG waveform should land near this number.
        """
        return self._hr

    @property
    def dropped_samples(self) -> int:
        """Samples discarded because the consumer could not keep up."""
        return self._dropped

    def trigger_fall(self, post_activity: str = "lying") -> float:
        """Inject one scripted fall, starting with the very next sample.

        The accelerometer signature is the textbook one a detector looks for,
        in four stages (exact numbers are the ``FALL_*`` module constants):

        1. **free fall**, :data:`FALL_FREEFALL_SEC` (70 ms, inside the
           30-100 ms a detector expects): total acceleration drops to about
           :data:`FALL_FREEFALL_G` (0.32 g), far below the usual 0.6 g
           threshold, while the wrist tumbles;
        2. **impact**, :data:`FALL_IMPACT_SEC` (85 ms): a spike peaking around
           :data:`FALL_IMPACT_PEAK_G` (3.9 g, up to ~4.4 g once the ringing
           is added), well above the usual 2.5 g threshold;
        3. **settle**, :data:`FALL_SETTLE_SEC` (195 ms): a couple of small
           decaying bounces;
        4. **stillness**, :data:`FALL_STILL_SEC` (1.05 s): gravity on the
           new, lying-down axis and almost no movement at all.

        The PPG channel also takes a large artefact (the sensor is thrown off
        its contact point) and the heart rate jumps by
        :data:`FALL_HR_STARTLE_BPM`, then decays - a real startle response.

        The posture is switched to ``post_activity`` (``"lying"`` by default,
        because the person is now on the floor) immediately, so that when the
        script ends the signal continues as someone lying still. Pass another
        activity name if a test needs them to get up again.

        Calling this while a fall is already playing restarts the script.

        Returns
        -------
        float
            How long the scripted fall lasts, in seconds
            (:data:`FALL_TOTAL_SEC`, about 1.4 s), so a caller knows how long
            to keep collecting.
        """
        key = _canonical_activity(post_activity)
        with self._lock:
            effective = self._effective_profile()
            # Free fall starts from whatever orientation the wrist is in now,
            # and ends in the new lying-down orientation.
            self._fall_pre_dir = _unit(effective.gravity)
            post = ACTIVITIES[key]
            self._fall_post_dir = _unit(
                (
                    post.gravity[0] + self._rng.gauss(0.0, 0.08),
                    post.gravity[1] + self._rng.gauss(0.0, 0.08),
                    post.gravity[2] + self._rng.gauss(0.0, 0.08),
                )
            )
            # Switch the posture now, with no blending: the scripted waveform
            # overrides the accelerometer and gyroscope for the whole fall, so
            # by the time the script ends the body is already "lying".
            self._profile = post
            self._prev_profile = post
            self._activity = key
            self._blend = 1.0
            self._cadence_target = self._pick_cadence(post)
            self._cadence = self._cadence_target
            self._hr_startle = FALL_HR_STARTLE_BPM
            self._fall_elapsed = 0.0
        return FALL_TOTAL_SEC

    @property
    def is_falling(self) -> bool:
        """True while a scripted fall is still playing out."""
        return self._fall_elapsed is not None

    # -- internals ----------------------------------------------------------
    def _pick_cadence(self, profile: ActivityProfile) -> float:
        """Draw a step rate from this activity's plausible range.

        Real people do not all walk at exactly the same cadence, so each
        switch picks a new one inside the activity's band and the simulator
        then drifts gently around it.
        """
        low, high = profile.cadence_range_hz
        return self._rng.uniform(low, high)

    def _effective_profile(self) -> ActivityProfile:
        """The profile actually in force right now.

        While :attr:`_blend` is below 1 this is a mixture of the previous and
        the new activity, which is what stops a switch from looking like a cut
        in the signal.
        """
        m = self._blend
        if m >= 1.0:
            return self._profile
        a, b = self._prev_profile, self._profile
        return replace(
            b,
            gravity=_lerp3(a.gravity, b.gravity, m),
            cadence_range_hz=(
                _lerp(a.cadence_range_hz[0], b.cadence_range_hz[0], m),
                _lerp(a.cadence_range_hz[1], b.cadence_range_hz[1], m),
            ),
            accel_amp_g=_lerp3(a.accel_amp_g, b.accel_amp_g, m),
            gyro_amp_dps=_lerp3(a.gyro_amp_dps, b.gyro_amp_dps, m),
            harmonic2=_lerp(a.harmonic2, b.harmonic2, m),
            accel_noise_g=_lerp(a.accel_noise_g, b.accel_noise_g, m),
            gyro_noise_dps=_lerp(a.gyro_noise_dps, b.gyro_noise_dps, m),
            heart_rate_bpm=_lerp(a.heart_rate_bpm, b.heart_rate_bpm, m),
            ppg_motion_counts=_lerp(
                a.ppg_motion_counts, b.ppg_motion_counts, m
            ),
            intensity=_lerp(a.intensity, b.intensity, m),
        )

    @staticmethod
    def _gait_wave(phase: float, harmonic2: float) -> float:
        """One cycle of a gait-like wave, scaled to roughly +/-1.

        A pure sine looks nothing like walking. Adding a component at twice
        the frequency produces the asymmetric "swing then heel-strike" shape
        that real wrist accelerometers show, and it is also what makes the
        frequency-domain features in the training set look the way they do.
        """
        raw = math.sin(phase) + harmonic2 * math.sin(2.0 * phase + 0.6)
        return raw / (1.0 + harmonic2)

    @staticmethod
    def _pulse_wave(phase: float) -> float:
        """One cardiac cycle of a PPG waveform, scaled to roughly +/-1.

        A real pulse wave has a fast systolic upstroke, then a dip (the
        dicrotic notch, where the aortic valve closes), then a smaller
        diastolic bump. Three harmonics are enough to look convincing and,
        importantly, all the energy still sits at and just above the
        fundamental heart rate, which is what heart-rate code searches for.
        """
        return (
            math.sin(phase)
            + 0.33 * math.sin(2.0 * phase - 0.9)
            + 0.12 * math.sin(3.0 * phase - 1.8)
        ) / 1.45

    def _generate_sample(self) -> Sample:
        """Build the next single :class:`Sample`. Caller holds the lock."""
        dt = self._dt
        t = self._t0 + self._index * dt
        self._index += 1

        # --- advance the slow state machines ------------------------------
        if self._blend < 1.0:
            self._blend = min(1.0, self._blend + dt / PROFILE_BLEND_SEC)
        p = self._effective_profile()

        # Step rate drifts towards its target with a little random wander, so
        # the rhythm is never metronomic.
        self._cadence += (
            (self._cadence_target - self._cadence) * dt / CADENCE_TAU_SEC
            + self._rng.gauss(0.0, 0.004)
        )
        self._cadence = _clamp(self._cadence, 0.05, 4.0)
        self._step_phase = (
            self._step_phase + TWO_PI * self._cadence * dt
        ) % TWO_PI

        # Stride-to-stride amplitude variation (people are not machines).
        self._amp_mod += (1.0 - self._amp_mod) * dt * 2.0 + self._rng.gauss(
            0.0, 0.010
        )
        self._amp_mod = _clamp(self._amp_mod, 0.80, 1.20)

        # Breathing: faster when working harder (12 breaths/min at rest,
        # ~27 at a run).
        resp_hz = 0.20 + 0.25 * p.intensity
        self._resp_phase = (self._resp_phase + TWO_PI * resp_hz * dt) % TWO_PI

        # Heart rate eases towards the activity's rate, plus any startle left
        # over from a fall.
        self._hr_startle *= math.exp(-dt / STARTLE_TAU_SEC)
        hr_target = _clamp(
            p.heart_rate_bpm + self._hr_startle, HR_MIN_BPM, HR_MAX_BPM
        )
        self._hr += (hr_target - self._hr) * dt / HR_TAU_SEC
        self._hr = _clamp(self._hr, HR_MIN_BPM, HR_MAX_BPM)

        # Respiratory sinus arrhythmia: the heart speeds up slightly on each
        # in-breath. Small, but it is why real HR traces are never flat.
        inst_hr = _clamp(
            self._hr * (1.0 + 0.03 * math.sin(self._resp_phase)),
            HR_MIN_BPM,
            HR_MAX_BPM,
        )
        self._pulse_phase = (
            self._pulse_phase + TWO_PI * (inst_hr / 60.0) * dt
        ) % TWO_PI

        # --- inertial channels --------------------------------------------
        if self._fall_elapsed is not None:
            accel, gyro, fall_artefact = self._fall_waveform(self._fall_elapsed)
            self._fall_elapsed += dt
            if self._fall_elapsed > FALL_TOTAL_SEC:
                self._fall_elapsed = None
        else:
            accel, gyro = self._motion_waveform(p)
            fall_artefact = 0.0

        ax, ay, az = (_clamp(v, -ACCEL_RANGE_G, ACCEL_RANGE_G) for v in accel)
        gx, gy, gz = (_clamp(v, -GYRO_RANGE_DPS, GYRO_RANGE_DPS) for v in gyro)

        ir = self._ppg_value(p, fall_artefact)

        return Sample(
            t=t, ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz, IR_Value=ir
        )

    def _motion_waveform(
        self, p: ActivityProfile
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Accelerometer and gyroscope values for ordinary activity.

        Accelerometer = gravity (posture) + body rhythm (gait or sway) +
        noise. The gyroscope has no gravity term - a still sensor reads zero
        rotation whichever way up it is - so it is just rhythm + noise, which
        is exactly why the gyro channels are good at telling moving activities
        apart and useless for telling sitting from standing.

        The three axes are given different phase offsets because the wrist
        does not accelerate along all axes at the same instant during a
        stride.
        """
        grav = _unit(p.gravity)
        amp = p.accel_amp_g
        gamp = p.gyro_amp_dps
        h2 = p.harmonic2
        ph = self._step_phase
        mod = self._amp_mod

        accel_offsets = (0.0, 1.9, 3.4)
        gyro_offsets = (0.7, 2.6, 4.5)

        accel = tuple(
            grav[i]
            + amp[i] * mod * self._gait_wave(ph + accel_offsets[i], h2)
            + self._rng.gauss(0.0, p.accel_noise_g)
            for i in range(3)
        )
        gyro = tuple(
            gamp[i] * mod * self._gait_wave(ph + gyro_offsets[i], h2 * 0.7)
            + self._rng.gauss(0.0, p.gyro_noise_dps)
            for i in range(3)
        )
        return accel, gyro  # type: ignore[return-value]

    def _fall_waveform(
        self, te: float
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        """The scripted fall, as a function of time since it started.

        Returns ``(accel_g, gyro_dps, ppg_artefact_counts)``.

        The trick is to generate a *magnitude* profile (how hard the sensor is
        being accelerated) and a *direction* that rotates from the pre-fall
        posture to the lying-down posture, then multiply them. That way the
        total-acceleration trace a detector computes has exactly the intended
        dip-then-spike shape no matter which way the device was facing.
        """
        t1 = FALL_FREEFALL_SEC
        t2 = t1 + FALL_IMPACT_SEC
        t3 = t2 + FALL_SETTLE_SEC

        if te < t1:
            # STAGE 1 - free fall. Support is gone, so the proof mass inside
            # the accelerometer stops being pushed and the reading collapses
            # towards 0 g. A short ramp at the start keeps the signal
            # continuous rather than teleporting down.
            ramp = min(1.0, te / 0.012)
            mag = _lerp(1.0, FALL_FREEFALL_G, ramp) + self._rng.gauss(0.0, 0.02)
            direction = self._fall_pre_dir
            # Tumbling: the wrist rotates freely while unsupported.
            tumble = TWO_PI * 3.0 * te
            gyro = (
                130.0 * math.sin(tumble) + self._rng.gauss(0.0, 6.0),
                -95.0 * math.cos(tumble * 1.3) + self._rng.gauss(0.0, 6.0),
                110.0 * math.sin(tumble * 0.8 + 1.0) + self._rng.gauss(0.0, 6.0),
            )
            artefact = FALL_PPG_ARTEFACT * 0.4

        elif te < t2:
            # STAGE 2 - impact. The body hits the floor: a short, violent
            # spike, shaped like half a sine so it starts and ends smoothly,
            # with fast ringing from the structure of the device and the arm.
            u = (te - t1) / FALL_IMPACT_SEC
            base = _lerp(FALL_FREEFALL_G, 1.0, u)
            shock = (FALL_IMPACT_PEAK_G - 1.0) * math.sin(math.pi * u)
            ring = 1.0 + 0.18 * math.sin(TWO_PI * 55.0 * (te - t1))
            mag = base + shock * ring + self._rng.gauss(0.0, 0.05)
            direction = _unit(
                _lerp3(self._fall_pre_dir, self._fall_post_dir, u * 0.8)
            )
            spike = math.sin(math.pi * u)
            gyro = (
                300.0 * spike + self._rng.gauss(0.0, 12.0),
                -220.0 * spike + self._rng.gauss(0.0, 12.0),
                260.0 * spike + self._rng.gauss(0.0, 12.0),
            )
            artefact = FALL_PPG_ARTEFACT

        elif te < t3:
            # STAGE 3 - settle. Limbs and clothing come to rest: a decaying
            # wobble around 1 g while the orientation finishes rotating.
            u = (te - t2) / FALL_SETTLE_SEC
            decay = math.exp(-u * 6.0)
            mag = (
                1.0
                + 0.30 * decay * math.sin(TWO_PI * 9.0 * (te - t2))
                + self._rng.gauss(0.0, 0.02)
            )
            direction = _unit(
                _lerp3(self._fall_pre_dir, self._fall_post_dir, 0.8 + 0.2 * u)
            )
            gyro = tuple(
                80.0 * decay * math.sin(TWO_PI * 7.0 * (te - t2) + k)
                + self._rng.gauss(0.0, 2.0)
                for k in (0.0, 2.1, 4.2)
            )
            artefact = FALL_PPG_ARTEFACT * 0.5 * decay

        else:
            # STAGE 4 - stillness. The person is on the floor and not moving.
            # This quiet stretch is what distinguishes a fall from sitting
            # down heavily or from dropping the device and picking it up.
            mag = 1.0 + self._rng.gauss(0.0, FALL_STILL_NOISE_G)
            direction = self._fall_post_dir
            gyro = tuple(self._rng.gauss(0.0, 0.8) for _ in range(3))
            artefact = FALL_PPG_ARTEFACT * 0.05

        accel = (
            direction[0] * mag + self._rng.gauss(0.0, 0.004),
            direction[1] * mag + self._rng.gauss(0.0, 0.004),
            direction[2] * mag + self._rng.gauss(0.0, 0.004),
        )
        return accel, gyro, artefact  # type: ignore[return-value]

    def _ppg_value(self, p: ActivityProfile, fall_artefact: float) -> float:
        """One raw ``IR_Value`` count from the PPG sensor.

        Built from four pieces, largest first:

        1. **baseline (DC)**: ~85 000 counts of light that comes back off
           skin, bone and non-pulsing tissue, plus a slow drift and a wobble
           with each breath (the chest moving changes venous return).
        2. **cardiac pulse (AC)**: about 1% of the baseline. Its frequency is
           the current heart rate, always inside 0.75-3.0 Hz (45-180 BPM), so
           downstream heart-rate code has a real signal to find. The amplitude
           drops a little during vigorous activity because sensor contact gets
           worse.
        3. **motion artefact** at the *step* frequency, growing with activity
           intensity: the sensor shifting against the skin. During running it
           is larger than the pulse itself, which is exactly the problem real
           wrist monitors have. It is deliberately kept at a level where a
           decent algorithm can still recover the heart rate rather than being
           hopeless.
        4. **noise**: a few tens of counts.

        The result is clamped to the sensor's 18-bit range.
        """
        # 1. baseline
        self._ppg_drift += self._rng.gauss(0.0, 8.0)
        self._ppg_drift = _clamp(self._ppg_drift, -1500.0, 1500.0)
        baseline = (
            PPG_DC_COUNTS
            + self._ppg_drift
            + PPG_RESP_COUNTS * math.sin(self._resp_phase)
        )

        # 2. cardiac pulse
        ac_amp = PPG_AC_COUNTS * (1.0 - 0.25 * p.intensity)
        cardiac = ac_amp * self._pulse_wave(self._pulse_phase)

        # 3. motion artefact, at the step frequency (plus its harmonic), so
        #    it is distinguishable from the pulse unless cadence and heart
        #    rate happen to coincide.
        motion = p.ppg_motion_counts * (
            math.sin(self._step_phase + 0.4)
            + 0.3 * math.sin(2.0 * self._step_phase)
        )
        if fall_artefact:
            motion += fall_artefact * math.sin(self._step_phase * 3.0 + 1.1)

        # 4. noise
        noise = self._rng.gauss(0.0, PPG_NOISE_COUNTS)

        return _clamp(
            baseline + cardiac + motion + noise, 0.0, float(IR_MAX_COUNT)
        )


# ---------------------------------------------------------------------------
# Self-test: python datasource.py
# ---------------------------------------------------------------------------
# Everything below this line exists only to prove the module works when it is
# run directly. The little rate estimator is a crude check that the signals
# land in the right frequency band - it is NOT the project's heart-rate or
# fall-detection algorithm, which live in other people's files.
def _selftest_dominant_rate(
    series: list[float], fs: float, lo_hz: float, hi_hz: float
) -> float:
    """Rough dominant frequency of ``series`` by autocorrelation (test only).

    Removes the mean, then asks "which shift lines the signal up with itself
    best?". The best shift is one period, so frequency = fs / lag. Only lags
    inside ``[1/hi_hz, 1/lo_hz]`` are considered. Scores are divided by the
    full length rather than by the number of overlapping terms, which stops
    long lags (few terms, noisy) from winning by accident.
    """
    n = len(series)
    if n < 8:
        return 0.0
    mean = sum(series) / n
    x = [v - mean for v in series]
    lo_lag = max(1, int(fs / hi_hz))
    hi_lag = min(n - 2, int(fs / lo_hz))
    best_lag, best_score = 0, -1e30
    for lag in range(lo_lag, hi_lag + 1):
        score = sum(x[i] * x[i + lag] for i in range(n - lag)) / n
        if score > best_score:
            best_lag, best_score = lag, score
    return fs / best_lag if best_lag else 0.0


def _collect(src: SimulatedSource, seconds: float) -> list[Sample]:
    """Poll ``read()`` every 10 ms for ``seconds``, the way the UI would."""
    out: list[Sample] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        out.extend(src.read())
        time.sleep(0.010)
    out.extend(src.read())
    return out


def _sparkline(values: list[float], lo: float, hi: float) -> str:
    """Draw a list of numbers as a one-line bar chart (no plotting library)."""
    glyphs = " .:-=+*#%@"
    span = (hi - lo) or 1.0
    return "".join(
        glyphs[
            min(
                len(glyphs) - 1,
                max(0, int((v - lo) / span * (len(glyphs) - 1))),
            )
        ]
        for v in values
    )


def _selftest() -> int:
    """Run the checks described in the module docstring. Returns an exit code."""
    failures: list[str] = []

    print("=" * 72)
    print("datasource.py self-test - simulated HAR wrist unit")
    print("=" * 72)
    print(f"nominal rate   : {SAMPLING_FREQ_HZ} Hz")
    print(f"activities     : {', '.join(SimulatedSource.available_activities())}")
    print(f"sample fields  : t, {', '.join(SENSOR_COLUMNS)}, {IR_COLUMN}")

    src = SimulatedSource(activity="sitting", seed=7)

    # --- 1. measured sample rate -------------------------------------------
    print("\n--- 1. sample rate over ~2 s, read() polled every 100 ms -------")
    src.start()
    counts: list[int] = []
    t_start = time.monotonic()
    total = 0
    while time.monotonic() - t_start < 2.0:
        time.sleep(0.100)
        n = len(src.read())
        counts.append(n)
        total += n
    elapsed = time.monotonic() - t_start
    measured = total / elapsed
    print(f"samples        : {total} in {elapsed:.3f} s")
    print(f"measured rate  : {measured:.2f} Hz")
    print(
        f"per read()     : min {min(counts)}, max {max(counts)}, "
        f"mean {total / len(counts):.1f} (expect ~10)"
    )
    print(f"dropped        : {src.dropped_samples}")
    if not 95.0 <= measured <= 105.0:
        failures.append(f"sample rate {measured:.2f} Hz is not ~100 Hz")

    # --- 2. a few samples --------------------------------------------------
    print("\n--- 2. a few raw samples (sitting) -----------------------------")
    batch = _collect(src, 0.2)
    print(
        f"{'t (s)':>8} {'ax':>7} {'ay':>7} {'az':>7} {'gx':>7} {'gy':>7} "
        f"{'gz':>7} {'IR_Value':>10} {'|a| g':>7}"
    )
    t0 = batch[0].t
    for s in batch[:6]:
        print(
            f"{s.t - t0:8.3f} {s.ax:7.3f} {s.ay:7.3f} {s.az:7.3f} "
            f"{s.gx:7.2f} {s.gy:7.2f} {s.gz:7.2f} {s.IR_Value:10.1f} "
            f"{s.accel_magnitude:7.3f}"
        )
    print("as_dict() keys :", list(batch[0].as_dict()))

    # --- 3. static postures ------------------------------------------------
    print("\n--- 3. static postures: ~1 g of gravity, little movement -------")
    print(
        f"{'activity':<20}{'mean |a| (g)':>14}{'gravity axis':>18}"
        f"{'gyro max (dps)':>16}"
    )
    for name in ("sitting", "standing", "lying"):
        src.set_activity(name)
        data = _collect(src, 0.9)[-50:]
        mean_mag = sum(s.accel_magnitude for s in data) / len(data)
        axis_means = {
            "ax": sum(s.ax for s in data) / len(data),
            "ay": sum(s.ay for s in data) / len(data),
            "az": sum(s.az for s in data) / len(data),
        }
        axis = max(axis_means, key=lambda k: abs(axis_means[k]))
        gyro_max = max(s.gyro_magnitude for s in data)
        label = f"{axis} = {axis_means[axis]:+.2f}"
        print(f"{name:<20}{mean_mag:>14.3f}{label:>18}{gyro_max:>16.1f}")
        if not 0.90 <= mean_mag <= 1.10:
            failures.append(f"{name}: mean |a| {mean_mag:.3f} g is not ~1 g")
        if gyro_max > 20.0:
            failures.append(f"{name}: gyro {gyro_max:.1f} dps is not near-static")

    # --- 4. moving activities ----------------------------------------------
    print("\n--- 4. moving activities: cadence and amplitude ----------------")
    print(
        f"{'activity':<20}{'cadence (Hz)':>14}{'|a| range (g)':>18}"
        f"{'gyro max (dps)':>16}"
    )
    expected = {
        "walking": (1.50, 2.00),
        "running": (2.40, 3.10),
        "walking_upstairs": (1.20, 1.70),
        "walking_downstairs": (1.40, 1.90),
    }
    for name, (lo, hi) in expected.items():
        src.set_activity(name)
        # Throw away the first couple of seconds: the step rate eases from
        # the previous activity's cadence into the new one (CADENCE_TAU_SEC),
        # so measuring immediately would average the transition.
        _collect(src, 3.0)
        data = _collect(src, 3.0)[-250:]
        mags = [s.accel_magnitude for s in data]
        cadence = _selftest_dominant_rate(mags, src.sample_rate_hz, 1.0, 3.5)
        gyro_max = max(s.gyro_magnitude for s in data)
        rng_label = f"{min(mags):.2f} .. {max(mags):.2f}"
        print(f"{name:<20}{cadence:>14.2f}{rng_label:>18}{gyro_max:>16.1f}")
        if not lo <= cadence <= hi:
            failures.append(
                f"{name}: cadence {cadence:.2f} Hz outside {lo}-{hi} Hz"
            )

    # --- 5. PPG pulse ------------------------------------------------------
    print("\n--- 5. PPG channel carries a pulse in the 0.75-3 Hz band -------")
    src.set_activity("sitting")
    # Pin the heart rate so the check does not straddle the slow decay from
    # the running section above (HR_TAU_SEC is 8 s).
    src.set_heart_rate(72.0)
    _collect(src, 1.0)
    data = _collect(src, 6.0)
    ir = [s.IR_Value for s in data]
    pulse_hz = _selftest_dominant_rate(ir, src.sample_rate_hz, 0.75, 3.0)
    dc = sum(ir) / len(ir)
    print(f"IR baseline    : {dc:,.0f} counts (peak-to-peak {max(ir) - min(ir):,.0f})")
    print(f"detected pulse : {pulse_hz:.2f} Hz = {pulse_hz * 60:.0f} BPM")
    print(f"simulated HR   : {src.heart_rate_bpm:.0f} BPM (ground truth)")
    if not 0.75 <= pulse_hz <= 3.0:
        failures.append(f"PPG pulse {pulse_hz:.2f} Hz outside 0.75-3 Hz band")
    if abs(pulse_hz * 60 - src.heart_rate_bpm) > 12.0:
        failures.append(
            f"PPG pulse {pulse_hz * 60:.0f} BPM far from simulated "
            f"{src.heart_rate_bpm:.0f} BPM"
        )

    # --- 6. scripted fall --------------------------------------------------
    print("\n--- 6. trigger_fall(): dip below 0.6 g, then spike above 2.5 g -")
    src.set_activity("walking")
    _collect(src, 1.5)
    duration = src.trigger_fall()
    fall_data = _collect(src, duration + 0.6)
    mags = [s.accel_magnitude for s in fall_data]
    fs = src.sample_rate_hz

    peak_i = max(range(len(mags)), key=lambda i: mags[i])
    # The dip is the quiet stretch immediately before the spike.
    dip_end = peak_i
    while dip_end > 0 and mags[dip_end] >= 0.6:
        dip_end -= 1
    dip_start = dip_end
    while dip_start > 0 and mags[dip_start - 1] < 0.6:
        dip_start -= 1
    dip_len_ms = (dip_end - dip_start + 1) / fs * 1000.0
    dip_min = min(mags[dip_start : dip_end + 1]) if dip_end >= dip_start else 9.9

    still_from = peak_i + int(0.35 * fs)
    still_to = min(len(mags), peak_i + int(1.35 * fs))
    still = mags[still_from:still_to]
    still_mean = sum(still) / len(still)
    still_dev = max(abs(v - 1.0) for v in still)
    gyro_still = max(s.gyro_magnitude for s in fall_data[still_from:still_to])

    print(f"free-fall dip  : min {dip_min:.3f} g, {dip_len_ms:.0f} ms below 0.6 g")
    print(f"impact peak    : {max(mags):.2f} g")
    print(
        f"post-fall still: mean |a| {still_mean:.3f} g, "
        f"max deviation {still_dev:.3f} g, gyro max {gyro_still:.1f} dps"
    )
    print(f"activity after : {src.activity!r} (is_falling={src.is_falling})")

    lo_i = max(0, peak_i - 25)
    hi_i = min(len(mags), peak_i + 115)
    window = mags[lo_i:hi_i]
    print(
        f"\n|a| trace, {(hi_i - lo_i) / fs * 1000:.0f} ms around the impact "
        f"(blank = 0 g, '@' = {max(window):.1f} g):"
    )
    print("  " + _sparkline(window, 0.0, max(window)))
    print(
        "  "
        + "".join("F" if v < 0.6 else ("I" if v > 2.5 else ".") for v in window)
        + "   F=free-fall  I=impact  .=normal"
    )

    if not dip_min < 0.6:
        failures.append(f"free-fall dip {dip_min:.3f} g did not go below 0.6 g")
    if not 30.0 <= dip_len_ms <= 100.0:
        failures.append(f"free-fall lasted {dip_len_ms:.0f} ms, want 30-100 ms")
    if not max(mags) > 2.5:
        failures.append(f"impact peak {max(mags):.2f} g did not exceed 2.5 g")
    if still_dev > 0.15 or gyro_still > 15.0:
        failures.append("post-impact stretch is not near-still")

    src.stop()
    if src.read():
        failures.append("read() returned data after stop()")

    # --- verdict -----------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        print(f"FAIL - {len(failures)} problem(s):")
        for problem in failures:
            print(f"  * {problem}")
        return 1
    print("PASS - all checks met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
