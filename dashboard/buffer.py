"""Sample buffer for the live HAR dashboard.

What problem does this file solve?
---------------------------------
The wrist unit sends one reading ("sample") every 1/100 s. The Dash page wakes
up ten times a second, grabs whatever samples arrived since last time, and
needs two different things from them:

1. **Recent history for the graphs.** The live line charts show the last ten
   seconds, one line per channel. They want plain Python lists.
2. **Complete analysis windows for the classifier.** The models in
   ``data-processing/`` were trained on 4-second windows (400 samples) that
   advance 2 seconds (200 samples) at a time, so they must be fed the same
   shape. They want numpy arrays.

:class:`SampleBuffer` below does both, and nothing else. It stores samples and
hands out windows. It does **not** compute features and it does **not** run a
model - that is the next person's job, and it starts from the window this file
produces. This module also never imports ``dash`` or ``plotly``, so you can
import it and poke at it from a plain Python prompt.

The two ideas worth understanding
---------------------------------
**A ring buffer.** We only ever care about the most recent ``BUFFER_SAMPLES``
(1000) samples. A ring buffer is a container with a fixed size: when it is full
and you add one more item, the oldest item falls off the other end
automatically. We use :class:`collections.deque` with ``maxlen`` for this, so
appending is cheap no matter how long the dashboard has been running - nothing
is re-allocated and the history is never copied. That matters, because this
code runs ten times a second for as long as the dashboard is open.

**Absolute sample numbers.** Every sample that is ever accepted gets a number:
0, 1, 2, 3, ... forever. These numbers keep counting up even after a sample has
fallen out of the ring buffer. The buffer remembers which absolute number the
*next* analysis window should start at (``_next_window_start``). When it hands
a window out, it moves that marker forward by ``STEP_SIZE``. That one marker is
the whole trick behind "each window is handed out exactly once": a window is
only built from the marker position, and the marker only ever moves forward.

Typical use from a Dash callback::

    from buffer import SampleBuffer

    buf = SampleBuffer()                      # created once, at import time

    # ... ten times a second: ...
    buf.extend(new_samples_from_the_device)   # an iterable of plain dicts

    history = buf.get_recent_seconds(10)      # for the graphs
    fig.add_scatter(x=history.timestamps, y=history["ax"], name="ax")

    for window in buf.drain_windows():        # usually 0 or 1 of them
        imu = window.imu                      # numpy array, shape (400, 6)
        ir = window.ir                        # numpy array, shape (400,)
        # -> hand imu/ir to the feature-extraction code

    rate = buf.measured_sampling_rate()       # e.g. 99.7, the *real* rate

Glossary
--------
sample
    One timestamped reading: a timestamp in seconds plus the six IMU channels
    (``ax ay az gx gy gz``) and ``IR_Value``.
window
    400 consecutive samples (4 seconds) handed to the classifier as one unit.
step
    How far the window start moves between windows: 200 samples (2 seconds),
    which is the 50 % overlap the models were trained with.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from config import (
    BUFFER_SAMPLES,
    IR_COLUMN,
    SAMPLING_FREQ_HZ,
    SENSOR_COLUMNS,
    STEP_SIZE,
    WINDOW_SIZE,
)

__all__ = [
    "SampleBuffer",
    "AnalysisWindow",
    "ChannelHistory",
    "BufferStats",
    "TIMESTAMP_KEYS",
]

# Field names we are willing to read the timestamp from (first one present
# wins). The firmware notes and the simulator have each used a different name at
# some point, so rather than pick a fight we accept any of them. Pass
# ``timestamp_key="whatever"`` to SampleBuffer if your source uses another name.
TIMESTAMP_KEYS: tuple[str, ...] = ("timestamp", "t", "time", "ts", "time_s")

# How much later than expected a sample may arrive before we call it a "gap".
# At 100 Hz samples are 0.01 s apart, so with the default factor of 2.0
# anything more than 0.02 s after its predecessor counts as a gap.
DEFAULT_GAP_FACTOR = 2.0

# The nominal (promised, assumed) time between samples, in seconds: 0.01 s.
NOMINAL_PERIOD_SEC = 1.0 / SAMPLING_FREQ_HZ


def _read_field(sample: Any, key: str) -> Any:
    """Pull ``key`` out of ``sample``, whether it is a dict or an object.

    We promised not to depend on any other module's sample class, so a plain
    ``dict`` is the supported input::

        {"timestamp": 12.34, "ax": 0.1, ..., "IR_Value": 812}

    But if the data source ever hands us an object with attributes instead (a
    dataclass, a namedtuple, a pandas row), reading the attribute works too, so
    nothing breaks. Returns ``None`` when the field is simply not there; the
    caller decides whether that is fatal.
    """
    if isinstance(sample, Mapping):
        if key in sample:
            return sample[key]
        return None
    return getattr(sample, key, None)


def _as_float(value: Any, key: str) -> float:
    """Convert one field to ``float``, with an error message a human can read."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sample field {key!r} is {value!r}, which is not a number"
        ) from exc


@dataclass(frozen=True)
class ChannelHistory:
    """Recent history, shaped for a line chart.

    Returned by the ``get_recent_*`` methods.

    Attributes
    ----------
    timestamps:
        The device timestamps in seconds, oldest first. Use this as the x-axis.
    channels:
        One list of y-values per channel, same length and same order as
        ``timestamps``. The keys are the six names in ``SENSOR_COLUMNS`` plus
        ``IR_COLUMN`` ("IR_Value").

    ``history["ax"]`` is shorthand for ``history.channels["ax"]``, so plotting
    reads nicely::

        fig.add_scatter(x=history.timestamps, y=history["ax"])
    """

    timestamps: list[float]
    channels: dict[str, list[float]]

    def __getitem__(self, channel: str) -> list[float]:
        """Allow ``history["ax"]`` instead of ``history.channels["ax"]``."""
        return self.channels[channel]

    def __contains__(self, channel: str) -> bool:
        return channel in self.channels

    @property
    def n(self) -> int:
        """Number of samples in this slice of history."""
        return len(self.timestamps)

    @property
    def seconds(self) -> float:
        """Time span covered: last timestamp minus first (0.0 if under two)."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to plot yet."""
        return not self.timestamps


@dataclass(frozen=True, eq=False)
class AnalysisWindow:
    """One complete 4-second window, ready for feature extraction.

    This is the hand-off point between this file and the feature/model code.
    Everything in here describes exactly 400 consecutive samples.

    Attributes
    ----------
    index:
        Which window this is, counting from 0 for the first one this buffer
        produced. Useful for logging, and for proving no window repeats.
    imu:
        ``numpy`` array of shape ``(WINDOW_SIZE, 6)`` = ``(400, 6)``, dtype
        float64. The columns are in **exactly** ``SENSOR_COLUMNS`` order:
        column 0 is ``ax``, 1 ``ay``, 2 ``az``, 3 ``gx``, 4 ``gy``, 5 ``gz``.
        The feature code indexes these positionally, so the order is not
        negotiable.
    ir:
        ``numpy`` array of shape ``(WINDOW_SIZE,)`` = ``(400,)`` holding
        ``IR_Value`` for the same 400 samples, in the same order.
    timestamps:
        ``numpy`` array of shape ``(400,)``: the device timestamp of each
        sample, increasing.
    start_index, end_index:
        Absolute sample numbers. ``start_index`` is the number of the first
        sample in the window; ``end_index`` is one past the last, so
        ``end_index - start_index == 400``, like a Python slice.
    gap_count:
        How many times inside this window a sample arrived later than expected
        (see ``DEFAULT_GAP_FACTOR``). ``0`` means a clean window.
    max_dt:
        The largest time step between two neighbouring samples in the window.
        At a healthy 100 Hz this is about 0.01; a big value means the stream
        stalled part-way through.

    A window with ``gap_count > 0`` is still handed over - dropping it silently
    would make the dashboard mysteriously stop predicting. Use
    :meth:`is_contiguous` if you would rather not classify a window with a hole
    in it.
    """

    index: int
    imu: np.ndarray
    ir: np.ndarray
    timestamps: np.ndarray
    start_index: int
    end_index: int
    gap_count: int
    max_dt: float

    @property
    def start_time(self) -> float:
        """Device timestamp of the first sample in the window."""
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        """Device timestamp of the last sample in the window."""
        return float(self.timestamps[-1])

    @property
    def duration(self) -> float:
        """Seconds between the first and last sample (about 3.99 at 100 Hz)."""
        return self.end_time - self.start_time

    @property
    def measured_rate(self) -> float:
        """Sampling rate measured across this window alone, in Hz.

        ``nan`` if the timestamps in the window do not advance at all.
        """
        span = self.duration
        if span <= 0.0:
            return float("nan")
        return (len(self.timestamps) - 1) / span

    def is_contiguous(self, gap_factor: float = DEFAULT_GAP_FACTOR) -> bool:
        """True when no sample in the window arrived suspiciously late.

        ``gap_factor`` is a multiple of the nominal 0.01 s spacing, so the
        default of 2.0 tolerates anything up to 0.02 s between samples.
        """
        if math.isnan(self.max_dt):
            return True
        return self.max_dt <= gap_factor * NOMINAL_PERIOD_SEC

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"AnalysisWindow(index={self.index}, "
            f"samples={self.start_index}..{self.end_index}, "
            f"imu={self.imu.shape}, ir={self.ir.shape}, "
            f"gaps={self.gap_count})"
        )


@dataclass
class BufferStats:
    """Plain counters about everything the buffer has seen.

    Handy for a "stream health" panel on the dashboard, and for debugging.
    ``samples_received`` counts accepted samples only; the other counters record
    what we threw away and why, so a misbehaving stream shows up as a number on
    screen instead of as silence.
    """

    samples_received: int = 0
    samples_out_of_order: int = 0
    samples_rejected_invalid: int = 0
    windows_emitted: int = 0
    windows_skipped: int = 0
    gaps_seen: int = 0

    def as_dict(self) -> dict[str, int]:
        """The counters as a plain dict, for merging into a status panel."""
        return dict(vars(self))


class SampleBuffer:
    """A ring buffer of recent samples that also hands out analysis windows.

    Create one of these when the dashboard starts and keep using it; it is
    designed to live for the whole life of the process.

    Parameters
    ----------
    capacity:
        How many samples to keep for the graphs. Defaults to ``BUFFER_SAMPLES``
        (1000 = 10 s at 100 Hz). Older samples fall off the back
        automatically. Must be at least ``window_size``, otherwise a full
        window could never be assembled.
    window_size:
        Samples per analysis window. Defaults to ``WINDOW_SIZE`` (400). Do not
        change this for the real dashboard - the trained models expect 400.
    step_size:
        Samples between the start of one window and the start of the next.
        Defaults to ``STEP_SIZE`` (200 = 50 % overlap), again to match training.
    timestamp_key:
        Force a specific timestamp field name instead of trying each of
        :data:`TIMESTAMP_KEYS` in turn.
    gap_factor:
        How late a sample may be before it counts as a gap, as a multiple of
        the nominal sample spacing. Default 2.0, i.e. 0.02 s at 100 Hz.
    reject_out_of_order:
        When ``True`` (the default) a sample whose timestamp is not newer than
        the previous sample's is refused and counted, instead of being stored.
        See :meth:`append` for why.

    Thread safety
    -------------
    Dash can run callbacks on several threads, and two threads both pulling
    windows is exactly how a window would get handed out twice. Every public
    method here takes an internal re-entrant lock, so one buffer can be shared.
    """

    def __init__(
        self,
        capacity: int = BUFFER_SAMPLES,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        *,
        timestamp_key: str | None = None,
        gap_factor: float = DEFAULT_GAP_FACTOR,
        reject_out_of_order: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if capacity < window_size:
            raise ValueError(
                f"capacity ({capacity}) must be at least window_size "
                f"({window_size}), or a full window could never be collected"
            )

        self.capacity = int(capacity)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.timestamp_key = timestamp_key
        self.gap_factor = float(gap_factor)
        self.reject_out_of_order = bool(reject_out_of_order)

        # The ring buffer itself. Each entry is one tuple of plain floats:
        #   (timestamp, ax, ay, az, gx, gy, gz, IR_Value)
        # A tuple rather than a dict keeps memory small and makes building the
        # numpy window a single conversion. ``maxlen`` is what makes this a
        # *ring* buffer: appending to a full deque drops the oldest entry in
        # constant time, with no copying and no re-allocation.
        self._samples: deque[tuple[float, ...]] = deque(maxlen=self.capacity)

        # Time step between each stored sample and the one before it. Kept the
        # same length as ``_samples`` and in step with it, so reporting gaps on
        # a window costs nothing extra at read time. The very first sample has
        # no predecessor, so its entry is ``nan``.
        self._deltas: deque[float] = deque(maxlen=self.capacity)

        # Absolute sample number of the first sample still held in _samples.
        self._oldest_index = 0
        # Absolute sample number the next analysis window should start at.
        self._next_window_start = 0
        # Timestamp of the most recently accepted sample (None before the first
        # one), used for the ordering check and for measuring gaps.
        self._last_timestamp: float | None = None

        self._window_counter = 0
        self.stats = BufferStats()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(self, sample: Any) -> bool:
        """Add one sample. Returns ``True`` if it was stored.

        ``sample`` is normally a plain dict::

            {"timestamp": 101.23, "ax": 0.01, "ay": -0.98, "az": 0.11,
             "gx": 0.3, "gy": 0.0, "gz": -0.2, "IR_Value": 812.0}

        The timestamp may be under any of the names in :data:`TIMESTAMP_KEYS`.

        A sample is refused (returning ``False`` and bumping a counter in
        :attr:`stats`) in two cases:

        * **Out of order.** Its timestamp is not newer than the previous
          sample's. Everything downstream - the windows, the graphs, the
          measured rate - assumes time runs forwards, and a repeated or stale
          timestamp is almost always a duplicated or retransmitted packet. We
          would rather drop one sample and count it than put a backwards step
          into a window the classifier will be judged on. Pass
          ``reject_out_of_order=False`` to keep such samples instead; they are
          counted either way.
        * **Not a finite number.** A value that is ``NaN`` or infinite would
          poison every feature computed from the window it lands in, so the
          sample is dropped and counted.

        A *missing* field is a programming error rather than a stream hiccup, so
        a sample with no timestamp or a missing channel raises ``ValueError``
        instead of being quietly skipped.
        """
        row = self._normalise(sample)
        with self._lock:
            if row is None:
                # NaN or infinity somewhere in the sample.
                self.stats.samples_rejected_invalid += 1
                return False

            timestamp = row[0]

            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                self.stats.samples_out_of_order += 1
                if self.reject_out_of_order:
                    return False

            if self._last_timestamp is None:
                delta = float("nan")
            else:
                delta = timestamp - self._last_timestamp
                if delta > self.gap_factor * NOMINAL_PERIOD_SEC:
                    self.stats.gaps_seen += 1

            # If the deque is already full this append silently drops the
            # oldest sample, so the absolute number of the oldest sample we
            # still hold moves up by one.
            if len(self._samples) == self.capacity:
                self._oldest_index += 1
            self._samples.append(row)
            self._deltas.append(delta)

            self._last_timestamp = timestamp
            self.stats.samples_received += 1
            return True

    def extend(self, samples: Iterable[Any]) -> int:
        """Add many samples at once; returns how many were actually stored.

        This is what the Dash tick callback calls with whatever the data source
        collected during the last 100 ms.
        """
        stored = 0
        with self._lock:
            for sample in samples:
                if self.append(sample):
                    stored += 1
        return stored

    def clear(self) -> None:
        """Throw everything away and reset every counter.

        Use this when the device is unplugged and reconnected: the new
        timestamps may start again from zero, which would otherwise look like
        one enormous backwards jump and get every new sample rejected as out of
        order.
        """
        with self._lock:
            self._samples.clear()
            self._deltas.clear()
            self._oldest_index = 0
            self._next_window_start = 0
            self._last_timestamp = None
            self._window_counter = 0
            self.stats = BufferStats()

    def _normalise(self, sample: Any) -> tuple[float, ...] | None:
        """Turn one incoming sample into a tuple of 8 floats.

        Order of the result: timestamp first, then the six IMU channels in
        ``SENSOR_COLUMNS`` order, then ``IR_Value``. Storing it in this fixed
        order is what lets :meth:`next_window` build the ``(400, 6)`` array
        without looking up any names.

        Returns ``None`` when a value is not finite (the caller drops and counts
        the sample); raises ``ValueError`` when a field is missing or is not a
        number at all.
        """
        if self.timestamp_key is not None:
            timestamp = _read_field(sample, self.timestamp_key)
            if timestamp is None:
                raise ValueError(
                    f"sample has no {self.timestamp_key!r} field: {sample!r}"
                )
        else:
            timestamp = None
            for key in TIMESTAMP_KEYS:
                timestamp = _read_field(sample, key)
                if timestamp is not None:
                    break
            if timestamp is None:
                raise ValueError(
                    "sample has no timestamp field (looked for "
                    f"{', '.join(TIMESTAMP_KEYS)}): {sample!r}"
                )

        values = [_as_float(timestamp, "timestamp")]
        for column in SENSOR_COLUMNS:
            raw = _read_field(sample, column)
            if raw is None:
                raise ValueError(f"sample has no {column!r} field: {sample!r}")
            values.append(_as_float(raw, column))
        raw_ir = _read_field(sample, IR_COLUMN)
        if raw_ir is None:
            raise ValueError(f"sample has no {IR_COLUMN!r} field: {sample!r}")
        values.append(_as_float(raw_ir, IR_COLUMN))

        if not all(math.isfinite(v) for v in values):
            return None
        return tuple(values)

    # ------------------------------------------------------------------
    # Reading: history for the graphs
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """How many samples are held right now (never more than ``capacity``)."""
        return len(self._samples)

    @property
    def is_full(self) -> bool:
        """True once the ring buffer holds a full ``capacity`` of samples."""
        return len(self._samples) == self.capacity

    @property
    def total_received(self) -> int:
        """Count of samples ever accepted.

        This is also the absolute sample number the *next* accepted sample will
        be given, since numbering starts at 0.
        """
        return self.stats.samples_received

    @property
    def latest_timestamp(self) -> float | None:
        """Timestamp of the newest sample, or ``None`` if nothing has arrived."""
        return self._last_timestamp

    def latest_sample(self) -> dict[str, float] | None:
        """The newest sample as a dict, or ``None`` if the buffer is empty.

        Useful for a "current reading" readout beside the graphs.
        """
        with self._lock:
            if not self._samples:
                return None
            return self._row_to_dict(self._samples[-1])

    @staticmethod
    def _row_to_dict(row: Sequence[float]) -> dict[str, float]:
        """Turn one stored tuple back into a named dict."""
        out: dict[str, float] = {"timestamp": row[0]}
        for offset, column in enumerate(SENSOR_COLUMNS, start=1):
            out[column] = row[offset]
        out[IR_COLUMN] = row[1 + len(SENSOR_COLUMNS)]
        return out

    def get_recent_samples(self, n: int | None = None) -> ChannelHistory:
        """Return the last ``n`` samples as per-channel lists for plotting.

        ``n=None`` means "everything you have". Asking for more than the buffer
        holds is not an error - you simply get what there is, so the graphs work
        from the very first tick instead of waiting for the buffer to fill.
        """
        with self._lock:
            held = len(self._samples)
            if n is None:
                count = held
            else:
                if n < 0:
                    raise ValueError("n must be >= 0")
                count = min(int(n), held)

            if count == 0:
                return ChannelHistory(
                    timestamps=[],
                    channels={c: [] for c in (*SENSOR_COLUMNS, IR_COLUMN)},
                )

            # A deque cannot be sliced, so copy it to a list and take the last
            # ``count`` rows. That copy is at most ``capacity`` (1000) rows and
            # happens on read, not on the hot append path.
            rows = list(self._samples)[held - count :]

        timestamps = [row[0] for row in rows]
        channels: dict[str, list[float]] = {}
        for offset, column in enumerate(SENSOR_COLUMNS, start=1):
            channels[column] = [row[offset] for row in rows]
        ir_offset = 1 + len(SENSOR_COLUMNS)
        channels[IR_COLUMN] = [row[ir_offset] for row in rows]
        return ChannelHistory(timestamps=timestamps, channels=channels)

    def get_recent_seconds(self, seconds: float) -> ChannelHistory:
        """Return the samples from the last ``seconds`` seconds, for plotting.

        The cut is made on the *device* timestamps rather than on a sample
        count, so if the stream stalled you get fewer points instead of a
        window that secretly reaches further back in time than the axis claims.
        """
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        with self._lock:
            if not self._samples:
                return self.get_recent_samples(0)
            cutoff = self._samples[-1][0] - float(seconds)
            # Samples are stored in time order, so walk back from the newest end
            # and stop at the first sample older than the cutoff.
            count = 0
            for row in reversed(self._samples):
                if row[0] < cutoff:
                    break
                count += 1
            return self.get_recent_samples(count)

    # ------------------------------------------------------------------
    # Reading: analysis windows for the classifier
    # ------------------------------------------------------------------

    @property
    def pending_windows(self) -> int:
        """How many complete windows are waiting to be handed out right now.

        Normally 0 or 1: at 100 Hz a new window becomes ready every 2 seconds,
        while the dashboard asks ten times a second.
        """
        with self._lock:
            ready = self.total_received - self._next_window_start - self.window_size
            if ready < 0:
                return 0
            return ready // self.step_size + 1

    def next_window(self) -> AnalysisWindow | None:
        """Hand out the next analysis window, or ``None`` if none is ready.

        "Ready" means ``window_size`` samples have arrived at and after
        ``_next_window_start``. Each call that returns a window moves that
        marker forward by ``step_size``, so:

        * **no window is ever handed out twice** - a window is only built from
          the marker position, and the marker only ever moves forward;
        * **no window is skipped** - the marker advances by exactly one step at
          a time, so it lands on every start position in turn.

        The one exception is a caller that stops asking for so long that the
        window it was due has already fallen out of the ring buffer (about 6
        seconds of not polling, with the default sizes). That data is gone and
        cannot be reconstructed, so the marker jumps forward to the oldest start
        position that *is* still available - staying on the same 2-second grid -
        and ``stats.windows_skipped`` records how many windows were lost. Loud
        rather than silent, which is the point.

        Returns ``None`` while fewer than ``window_size`` samples have arrived,
        which is the normal state during the first four seconds after startup.
        """
        with self._lock:
            self._catch_up_marker()

            start = self._next_window_start
            if self.total_received - start < self.window_size:
                return None  # not enough samples yet

            offset = start - self._oldest_index
            rows = list(self._samples)[offset : offset + self.window_size]
            deltas = list(self._deltas)[offset : offset + self.window_size]

            # One conversion for the whole window: shape (400, 8).
            block = np.asarray(rows, dtype=np.float64)
            timestamps = block[:, 0].copy()
            # Columns 1..6 are already in SENSOR_COLUMNS order, because
            # _normalise stored them that way. ascontiguousarray hands the
            # feature code a clean, C-contiguous (400, 6) array of its own
            # rather than a view onto this block.
            imu = np.ascontiguousarray(block[:, 1 : 1 + len(SENSOR_COLUMNS)])
            ir = block[:, 1 + len(SENSOR_COLUMNS)].copy()

            # The first delta in the window describes the step from the sample
            # *before* the window, which is not this window's business, so skip
            # it. ``nan`` deltas (the very first sample ever) are skipped too.
            inner = [d for d in deltas[1:] if not math.isnan(d)]
            max_dt = max(inner) if inner else float("nan")
            threshold = self.gap_factor * NOMINAL_PERIOD_SEC
            gap_count = sum(1 for d in inner if d > threshold)

            window = AnalysisWindow(
                index=self._window_counter,
                imu=imu,
                ir=ir,
                timestamps=timestamps,
                start_index=start,
                end_index=start + self.window_size,
                gap_count=gap_count,
                max_dt=max_dt,
            )

            self._window_counter += 1
            self.stats.windows_emitted += 1
            self._next_window_start = start + self.step_size
            return window

    def drain_windows(self, max_windows: int | None = None) -> list[AnalysisWindow]:
        """Hand out every window that is ready, oldest first.

        This is the method a Dash tick callback wants: it usually returns an
        empty list, and every couple of seconds returns one window. Pass
        ``max_windows`` to cap how much work a single tick does, so a burst of
        buffered samples cannot stall the page; whatever is left stays pending
        and comes back on the next call.
        """
        windows: list[AnalysisWindow] = []
        with self._lock:
            while max_windows is None or len(windows) < max_windows:
                window = self.next_window()
                if window is None:
                    break
                windows.append(window)
        return windows

    def iter_windows(self) -> Iterator[AnalysisWindow]:
        """Generator form of :meth:`drain_windows`, for ``for`` loops.

        Stops as soon as no further window is ready. Note that each window is
        consumed as you iterate, exactly as with :meth:`next_window`.
        """
        while True:
            window = self.next_window()
            if window is None:
                return
            yield window

    def _catch_up_marker(self) -> None:
        """Move the window marker forward if its samples have already expired.

        Only does anything when the caller fell far behind. Keeps the marker on
        the ``step_size`` grid, so window boundaries stay where the training
        pipeline would have put them.
        """
        if self._next_window_start >= self._oldest_index:
            return
        behind = self._oldest_index - self._next_window_start
        steps = -(-behind // self.step_size)  # ceiling division
        self._next_window_start += steps * self.step_size
        self.stats.windows_skipped += steps

    # ------------------------------------------------------------------
    # The real sampling rate
    # ------------------------------------------------------------------

    def measured_sampling_rate(self, n: int | None = None) -> float | None:
        """Measure the actual sampling rate in Hz from the stored timestamps.

        The device *promises* ``SAMPLING_FREQ_HZ`` (100 Hz) but may deliver 97,
        or 103, or stutter. The dashboard should show the truth, so this counts
        the samples in a recent stretch and divides by the time they really
        took::

            rate = (number_of_samples - 1) / (last_timestamp - first_timestamp)

        (The ``- 1`` is because *n* samples have only *n - 1* gaps between them.
        Forgetting it is the classic off-by-one here.)

        Parameters
        ----------
        n:
            How many of the most recent samples to measure over. The default is
            two seconds' worth (200 samples), or everything held if that is
            less. A short stretch reacts quickly; a long one is steadier.

        Returns
        -------
        float or None
            The measured rate, or ``None`` when it cannot be computed: fewer
            than two samples held, or timestamps that do not advance at all.
            ``None`` means "no measurement yet" - show it as "--", do not fall
            back to 100 and tell the user a number nobody measured.
        """
        with self._lock:
            held = len(self._samples)
            if held < 2:
                return None
            if n is None:
                count = min(held, 2 * SAMPLING_FREQ_HZ)
            else:
                count = min(int(n), held)
            if count < 2:
                return None
            rows = list(self._samples)[held - count :]
            span = rows[-1][0] - rows[0][0]
            if span <= 0:
                return None
            return (len(rows) - 1) / span

    def status(self) -> dict[str, Any]:
        """A snapshot dict for a status panel or a log line.

        Everything a human needs in order to see whether the stream is healthy:
        how full the buffer is, the measured rate against the nominal one, how
        many windows have gone out, and the counts of everything that went
        wrong.
        """
        with self._lock:
            rate = self.measured_sampling_rate()
            history = self.get_recent_samples()
            return {
                "buffered_samples": len(self._samples),
                "capacity": self.capacity,
                "buffered_seconds": history.seconds,
                "total_received": self.total_received,
                "measured_rate_hz": rate,
                "nominal_rate_hz": SAMPLING_FREQ_HZ,
                "pending_windows": self.pending_windows,
                "next_window_start": self._next_window_start,
                **self.stats.as_dict(),
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"SampleBuffer(held={len(self._samples)}/{self.capacity}, "
            f"received={self.total_received}, "
            f"windows_emitted={self.stats.windows_emitted})"
        )


# ======================================================================
# Self-test: run ``python buffer.py`` from inside the dashboard folder.
# ======================================================================

def _make_sample(i: int, t: float) -> dict[str, float]:
    """Build one fake sample.

    Each channel gets a value that encodes both the sample number and *which
    channel it is*: ``ax = i + 0.1``, ``ay = i + 0.2``, ... ``gz = i + 0.6``,
    ``IR_Value = i + 0.7``. So if a window ever puts the columns in the wrong
    order, the fractional part gives it away immediately.
    """
    return {
        "timestamp": t,
        "ax": i + 0.1,
        "ay": i + 0.2,
        "az": i + 0.3,
        "gx": i + 0.4,
        "gy": i + 0.5,
        "gz": i + 0.6,
        IR_COLUMN: i + 0.7,
    }


def _feed(
    buf: SampleBuffer,
    count: int,
    rate: float = SAMPLING_FREQ_HZ,
    start_index: int = 0,
    start_time: float = 0.0,
) -> float:
    """Feed ``count`` evenly spaced samples; returns the next free timestamp."""
    period = 1.0 / rate
    for k in range(count):
        buf.append(_make_sample(start_index + k, start_time + k * period))
    return start_time + count * period


def _check_ring_buffer_caps() -> None:
    buf = SampleBuffer()
    _feed(buf, 2500)
    assert len(buf) == BUFFER_SAMPLES, len(buf)
    assert buf.is_full
    assert buf.total_received == 2500
    newest = buf.latest_sample()
    assert newest is not None and round(newest["ax"] - 2499, 6) == 0.1, newest
    # The oldest sample held must be exactly capacity-1 behind the newest.
    history = buf.get_recent_samples()
    assert history.n == BUFFER_SAMPLES
    assert round(history["ax"][0] - (2500 - BUFFER_SAMPLES), 6) == 0.1
    print(f"  ok  ring buffer caps at {BUFFER_SAMPLES} samples")


def _check_history_for_graphs() -> None:
    buf = SampleBuffer()
    _feed(buf, 500)

    hist = buf.get_recent_samples(300)
    assert hist.n == 300
    assert len(hist.timestamps) == 300
    assert list(hist.channels) == [*SENSOR_COLUMNS, IR_COLUMN], list(hist.channels)
    for column in (*SENSOR_COLUMNS, IR_COLUMN):
        assert len(hist[column]) == 300
    assert all(isinstance(v, float) for v in hist["ax"][:5])
    # Oldest first, matching the timestamps.
    assert hist.timestamps[0] < hist.timestamps[-1]
    assert round(hist["ax"][0] - 200, 6) == 0.1
    assert round(hist["gz"][-1] - 499, 6) == 0.6
    assert abs(hist.seconds - 2.99) < 1e-6, hist.seconds

    # Asking for more than is held gives what there is, not an error.
    assert buf.get_recent_samples(99999).n == 500
    assert buf.get_recent_samples(0).n == 0
    assert SampleBuffer().get_recent_samples(10).is_empty

    secs = buf.get_recent_seconds(2.0)
    assert secs.n in (200, 201), secs.n  # 2 s at 100 Hz, both ends included
    assert abs(secs.seconds - 2.0) < 0.02, secs.seconds
    print("  ok  per-channel history for the graphs (counts, order, timestamps)")


def _check_window_shape_and_column_order() -> None:
    buf = SampleBuffer()
    _feed(buf, WINDOW_SIZE - 1)
    assert buf.next_window() is None, "a window appeared before 400 samples"
    assert buf.pending_windows == 0
    _feed(
        buf,
        1,
        start_index=WINDOW_SIZE - 1,
        start_time=(WINDOW_SIZE - 1) / SAMPLING_FREQ_HZ,
    )

    window = buf.next_window()
    assert window is not None, "no window after exactly 400 samples"
    assert window.imu.shape == (WINDOW_SIZE, 6), window.imu.shape
    assert window.ir.shape == (WINDOW_SIZE,), window.ir.shape
    assert window.timestamps.shape == (WINDOW_SIZE,)
    assert window.imu.dtype == np.float64 and window.ir.dtype == np.float64
    assert window.start_index == 0 and window.end_index == WINDOW_SIZE
    assert window.index == 0

    # Column order: the fractional part identifies the channel, the integer
    # part identifies the sample. Checked for every row and every column.
    for row in range(WINDOW_SIZE):
        for col in range(6):
            expected = row + (col + 1) / 10
            assert abs(window.imu[row, col] - expected) < 1e-9, (
                f"imu[{row}, {col}] is {window.imu[row, col]}, expected "
                f"{expected} - columns are not in SENSOR_COLUMNS order"
            )
        assert abs(window.ir[row] - (row + 0.7)) < 1e-9
    assert SENSOR_COLUMNS == ["ax", "ay", "az", "gx", "gy", "gz"]
    assert window.is_contiguous() and window.gap_count == 0
    assert abs(window.measured_rate - SAMPLING_FREQ_HZ) < 1e-6
    print("  ok  window shapes (400, 6) and (400,), columns in SENSOR_COLUMNS order")


def _check_windows_never_repeat_or_skip() -> None:
    """The important one: stream in dribs and drabs, like the real dashboard."""
    buf = SampleBuffer()
    total = 5000
    collected: list[AnalysisWindow] = []

    # 10 samples per tick = 100 ms of data at 100 Hz, drained every tick.
    index = 0
    time_s = 0.0
    while index < total:
        time_s = _feed(buf, 10, start_index=index, start_time=time_s)
        index += 10
        collected.extend(buf.drain_windows())

    expected_count = (total - WINDOW_SIZE) // STEP_SIZE + 1
    assert len(collected) == expected_count, (len(collected), expected_count)

    starts = [w.start_index for w in collected]
    assert starts == list(range(0, total - WINDOW_SIZE + 1, STEP_SIZE)), starts[:5]
    assert len(set(starts)) == len(starts), "a window was handed out twice"
    for earlier, later in zip(starts, starts[1:]):
        assert later - earlier == STEP_SIZE, (earlier, later)
    assert [w.index for w in collected] == list(range(expected_count))
    assert buf.stats.windows_emitted == expected_count
    assert buf.stats.windows_skipped == 0

    # And the contents are the samples we think they are, not just the labels.
    for window in collected:
        first = window.start_index
        assert abs(window.imu[0, 0] - (first + 0.1)) < 1e-9
        assert abs(window.imu[-1, 5] - (first + WINDOW_SIZE - 1 + 0.6)) < 1e-9
        assert abs(window.ir[0] - (first + 0.7)) < 1e-9
        assert window.imu.shape == (WINDOW_SIZE, 6)

    # Asking again right away gives nothing back: no window is ever re-served.
    assert buf.next_window() is None
    assert buf.drain_windows() == []
    print(
        f"  ok  {expected_count} windows, {STEP_SIZE} samples apart, "
        "none repeated, none skipped"
    )


def _check_slow_consumer_is_reported() -> None:
    """If nobody drains for ages, the lost windows are counted, not faked."""
    buf = SampleBuffer()
    _feed(buf, 3000)  # 30 s of data into a 10 s buffer, nothing drained
    windows = buf.drain_windows()
    starts = [w.start_index for w in windows]
    assert buf.stats.windows_skipped > 0, "falling behind went unnoticed"
    assert len(set(starts)) == len(starts)
    assert starts == sorted(starts)
    # Everything handed over must come from data we actually still hold.
    oldest_held = 3000 - BUFFER_SAMPLES
    assert min(starts) >= oldest_held, (min(starts), oldest_held)
    # Starts stay on the training grid (multiples of STEP_SIZE from 0).
    assert all(s % STEP_SIZE == 0 for s in starts), starts
    for window in windows:
        assert window.imu.shape == (WINDOW_SIZE, 6)
        assert abs(window.imu[0, 0] - (window.start_index + 0.1)) < 1e-9
    print(
        f"  ok  slow consumer: {buf.stats.windows_skipped} expired windows "
        "counted in stats.windows_skipped"
    )


def _check_measured_rate() -> None:
    for fed_rate in (100.0, 97.3, 103.8, 50.0):
        buf = SampleBuffer()
        _feed(buf, 600, rate=fed_rate)
        measured = buf.measured_sampling_rate()
        assert measured is not None
        assert abs(measured - fed_rate) < 0.05, (fed_rate, measured)
    # Not enough data to measure -> None, never a made-up 100.
    empty = SampleBuffer()
    assert empty.measured_sampling_rate() is None
    empty.append(_make_sample(0, 0.0))
    assert empty.measured_sampling_rate() is None
    # Short and long measuring stretches both work.
    buf = SampleBuffer()
    _feed(buf, 600, rate=99.0)
    short = buf.measured_sampling_rate(50)
    long = buf.measured_sampling_rate(600)
    assert short is not None and abs(short - 99.0) < 0.05, short
    assert long is not None and abs(long - 99.0) < 0.05, long
    print(
        "  ok  measured sampling rate matches the rate fed in (and is None "
        "when unmeasurable)"
    )


def _check_gap_handling() -> None:
    buf = SampleBuffer()
    t = _feed(buf, 200)
    t += 1.0  # a one-second hole in the stream
    _feed(buf, 300, start_index=200, start_time=t)
    assert buf.stats.gaps_seen == 1, buf.stats.gaps_seen
    window = buf.next_window()
    assert window is not None
    assert window.gap_count == 1, window.gap_count
    assert abs(window.max_dt - 1.01) < 1e-6, window.max_dt
    assert not window.is_contiguous()
    assert window.imu.shape == (WINDOW_SIZE, 6)  # still a usable window
    # The samples either side of the gap are still in the right places.
    assert abs(window.imu[199, 0] - 199.1) < 1e-9
    assert abs(window.imu[200, 0] - 200.1) < 1e-9
    print("  ok  gap in the stream: counted, flagged on the window, data intact")


def _check_out_of_order_and_bad_values() -> None:
    buf = SampleBuffer()
    _feed(buf, 100)
    held = len(buf)

    assert buf.append(_make_sample(100, 0.5)) is False  # stale timestamp
    assert buf.append(_make_sample(101, 0.99)) is False  # equal to the last one
    assert len(buf) == held, "an out-of-order sample got stored"
    assert buf.stats.samples_out_of_order == 2
    assert buf.total_received == 100

    bad = _make_sample(102, 1.5)
    bad["ay"] = float("nan")
    assert buf.append(bad) is False
    assert buf.stats.samples_rejected_invalid == 1
    assert len(buf) == held

    # A good sample after the bad ones still goes in.
    assert buf.append(_make_sample(103, 1.6)) is True
    assert len(buf) == held + 1

    # Missing fields are a bug, not a stream hiccup: they raise.
    missing_gz = {k: v for k, v in _make_sample(1, 2.0).items() if k != "gz"}
    for broken in ({"ax": 1.0}, missing_gz):
        try:
            buf.append(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing field accepted: {broken!r}")

    # Alternative timestamp key names are accepted.
    other = SampleBuffer()
    sample = _make_sample(0, 7.0)
    sample["t"] = sample.pop("timestamp")
    assert other.append(sample) is True
    print("  ok  out-of-order, NaN and malformed samples handled")


def _check_short_buffer_and_reset() -> None:
    buf = SampleBuffer()
    assert len(buf) == 0 and buf.next_window() is None
    assert buf.latest_sample() is None
    assert buf.measured_sampling_rate() is None
    assert buf.status()["buffered_samples"] == 0

    _feed(buf, 399)
    assert buf.next_window() is None, "window emitted with only 399 samples"
    assert buf.pending_windows == 0
    _feed(buf, 1, start_index=399, start_time=3.99)
    assert buf.pending_windows == 1
    assert buf.next_window() is not None

    buf.clear()
    assert len(buf) == 0
    assert buf.total_received == 0
    assert buf.next_window() is None
    assert buf.stats.windows_emitted == 0
    # After clear(), timestamps may start over without everything being refused.
    _feed(buf, 400)
    assert buf.next_window() is not None
    print("  ok  under-full buffer, pending_windows, and clear()")


def _check_status_panel() -> None:
    buf = SampleBuffer()
    _feed(buf, 1200, rate=98.5)
    buf.drain_windows()
    status = buf.status()
    for key in (
        "buffered_samples",
        "capacity",
        "buffered_seconds",
        "total_received",
        "measured_rate_hz",
        "nominal_rate_hz",
        "pending_windows",
        "windows_emitted",
        "windows_skipped",
        "gaps_seen",
        "samples_out_of_order",
        "samples_rejected_invalid",
    ):
        assert key in status, key
    assert status["buffered_samples"] == BUFFER_SAMPLES
    assert abs(status["measured_rate_hz"] - 98.5) < 0.05
    assert status["nominal_rate_hz"] == SAMPLING_FREQ_HZ
    print("  ok  status() snapshot for the dashboard panel")


def _self_test() -> None:
    print("buffer.py self-test")
    print(
        f"  config: {SAMPLING_FREQ_HZ} Hz, window {WINDOW_SIZE}, "
        f"step {STEP_SIZE}, buffer {BUFFER_SAMPLES}"
    )
    _check_ring_buffer_caps()
    _check_history_for_graphs()
    _check_window_shape_and_column_order()
    _check_windows_never_repeat_or_skip()
    _check_slow_consumer_is_reported()
    _check_measured_rate()
    _check_gap_handling()
    _check_out_of_order_and_bad_values()
    _check_short_buffer_and_reset()
    _check_status_panel()
    print("all checks passed")


if __name__ == "__main__":
    _self_test()
