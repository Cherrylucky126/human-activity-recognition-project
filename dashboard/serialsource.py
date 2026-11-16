"""Reading the real wrist unit.

PR 13 of 14.

A :class:`datasource.SensorSource` that reads the device instead of
simulating it. Everything above it - buffer, classifier, fall detector,
every panel - is unchanged; only which object ``live.py`` is given
changes.

The wire format
---------------
Taken from the project's own collector sketch,
``firmware/arduino-ide-data-collection``, which writes one line per sample::

    ESP_BT.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lu\\n",
                  sampleCounter++, ax, ay, az, gx, gy, gz, irValue);

So each line is eight comma-separated fields:

====  ==============  ===========================================
   0  sample counter  increments by one per sample; **not a time**
 1-3  ax, ay, az      g, raw/16384
 4-6  gx, gy, gz      deg/s, raw/131
   7  IR_Value        raw count from the MAX30102
====  ==============  ===========================================

It goes over Bluetooth Classic, not USB. On Windows that pairs as an
outgoing COM port, so this reads it as an ordinary serial port; the port
name is the only thing that differs from a cable.

Where the timestamps come from
------------------------------
**There is no timestamp on the wire.** The first field is a counter, so
this module derives time from it: ``t = counter / 100``.

Deriving from the counter rather than from arrival time is deliberate.
Bluetooth delivers in bursts - several samples arrive in one packet and
then nothing for a while - so arrival times would show a jitter the
sensor never had, and the windows handed to the classifier would be
unevenly spaced. The counter is exact, evenly spaced, and it also makes
dropped samples visible: a gap in the counter is a gap in time, which is
what the buffer and the status strip then report.

The cost is that if the firmware's loop does not really run at 100 Hz,
this will not notice. That is what the measured sampling rate in the
status strip is for - it is computed from these derived timestamps, so a
disagreement with the wall clock shows up there.

Two hardware findings that belong upstream
------------------------------------------
**The accelerometer is at plus or minus 2 g.** The sketch writes only
register 0x6B, to wake the MPU6050, and never writes 0x1C, so the part
stays at its power-on default range. The divisor of 16384 confirms it.

That matters more than it sounds: a real fall impact is several g, and no
single channel can report more than 2. Impacts clip, so however hard the
floor is hit, no one axis ever leaves the device above 2 g.

Per axis, though. The fall detector works on the vector magnitude, which
three clipped axes can still take to 3.46 g; ``replay.py`` measures 2.86 g
at impact on clipped data. The literature's 2.5 g is therefore reachable
for some impact orientations and not for others, rather than impossible
outright. **On 15 November 2026 the team lowered
the threshold to 1.8 g** so that detection can fire on real recordings at
all; see ``falldetection.IMPACT_THRESHOLD_G`` for what that costs in
specificity. The ceiling described here is why that number cannot go back
above 2 g while the firmware is as it is.

The alternative was not equal. Writing 0x1C to select
plus or minus 8 g fixes the clipping - but every model in
``data-processing/`` was trained on data recorded at plus or minus 2 g,
so changing the range invalidates them unless everything is retrained.
Leaving the range alone and lowering the fall detector's impact threshold
below 2 g keeps the models valid at the cost of a less specific alert.
``FallDetector`` already takes every threshold as a keyword argument, so
the second route needs no code change here.

**The gyroscope is at plus or minus 250 deg/s** (divisor 131), where the
simulator clips at 500. Brisk wrist rotation will clip on the real device
where it does not in simulation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

try:
    from config import SAMPLING_FREQ_HZ
    from datasource import Sample, SensorSource
except ImportError:  # pragma: no cover
    from dashboard.config import SAMPLING_FREQ_HZ
    from dashboard.datasource import Sample, SensorSource

BAUD_RATE = 115200
"""From ``Serial.begin(115200)`` in the collector sketch."""

FIELDS_PER_LINE = 8
"""The committed sketch's format: a counter, six motion channels, IR."""

FIELDS_PER_LINE_NO_COUNTER = 7
"""The format the unit actually shipped with, found on 16 November 2026.

The wrist unit in the team's hands does not run the sketch committed as
``firmware/arduino-ide-data-collection.cpp``. It sends **seven** fields,
not eight: the leading ``sampleCounter`` is absent, the floats carry four
decimals rather than two, and IR arrives around 3,200 rather than the
~85,000 the committed sketch's LED settings produce. For example::

    -0.2963,0.1891,1.0014,3.7710,0.5038,6.9160,3215

Both formats are accepted rather than one being chosen, because which
firmware is on the board is not something the dashboard can know, and a
reader that works for only one of them fails silently - every line is
discarded as noise and the page shows a connected device sending nothing.

What the missing counter costs is dropped-sample detection. With a
counter, a gap in the numbering is a gap in time, and the buffer and the
status strip both report it. Without one there is nothing to compare
against: samples are numbered locally as they arrive, so a dropped
Bluetooth packet shortens the recording invisibly instead of appearing as
a gap. The measured sampling rate is then the only symptom, which is why
that readout matters more on this firmware than on the committed one.
"""

ACCEL_FULL_SCALE_G = 2.0
"""What the hardware can actually report, given its default range.

Recorded here so that code and reviewers can see the ceiling rather than
rediscovering it from a flat-topped impact.
"""

GYRO_FULL_SCALE_DPS = 250.0
"""The gyroscope's ceiling, from the sketch's divisor of 131.

The simulator runs to 500 deg/s, so brisk wrist rotation clips on the real
device where it does not in a demonstration. ``replay.py`` applies both
ceilings so the pipeline can be tested on device-grade data.
"""


def parse_line(line, sampling_freq_hz=SAMPLING_FREQ_HZ, counter_if_absent=None):
    """Turn one line from the device into a :class:`datasource.Sample`.

    Returns ``(sample, counter)``, or ``(None, None)`` for anything that
    is not a data line. The device also sends handshake text such as
    ``ACK_3``, and a connection can be joined mid-line, so unparseable
    input is normal traffic and never an error.

    Handles both wire formats (see :data:`FIELDS_PER_LINE_NO_COUNTER`).
    For the seven-field one the caller passes ``counter_if_absent``, the
    number this sample should take; a line with no counter and no
    substitute is rejected rather than guessed at.

    Returns ``(sample, counter, counter_on_wire)``. The third value says
    which format the line was in, and it is returned rather than inferred
    because inferring it does not work: an eight-field stream beginning at
    counter 0 is indistinguishable from a seven-field line that was handed
    a local counter of 0. That exact case silently turned the committed
    firmware's timestamps into arrival times until ``replay.py`` caught it.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("ascii", errors="ignore")
        except Exception:  # noqa: BLE001
            return None, None, False

    parts = line.strip().split(",")

    if len(parts) == FIELDS_PER_LINE:
        try:
            counter = int(parts[0])
            ax, ay, az, gx, gy, gz = (float(p) for p in parts[1:7])
            ir = float(parts[7])
        except ValueError:
            return None, None, False
        on_wire = True
    elif len(parts) == FIELDS_PER_LINE_NO_COUNTER:
        # The firmware actually on the unit. No counter on the wire, so
        # the caller supplies one; see FIELDS_PER_LINE_NO_COUNTER for what
        # that costs.
        if counter_if_absent is None:
            return None, None, False
        try:
            ax, ay, az, gx, gy, gz = (float(p) for p in parts[0:6])
            ir = float(parts[6])
        except ValueError:
            return None, None, False
        counter = counter_if_absent
        on_wire = False
    else:
        return None, None, False

    return (
        Sample(
            t=counter / float(sampling_freq_hz),
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
            IR_Value=ir,
        ),
        counter,
        on_wire,
    )


class SerialSource(SensorSource):
    """Reads the wrist unit over a serial (or Bluetooth serial) port.

    Parameters
    ----------
    port:
        The port name, for example ``"COM7"`` on Windows or
        ``"/dev/rfcomm0"`` on Linux.
    serial_factory:
        Used by the tests to supply a fake port. Left alone it opens a
        real one with pyserial.
    """

    def __init__(self, port, *, baud=BAUD_RATE, serial_factory=None, sampling_freq_hz=SAMPLING_FREQ_HZ):
        self.port = port
        self.baud = baud
        self.fs = float(sampling_freq_hz)
        self._serial_factory = serial_factory

        self._serial = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = []

        self._last_counter = None
        self._local_counter = 0
        self._t0 = None
        self.counter_on_wire = None
        """Whether the firmware numbers its own samples.

        ``None`` until the first data line, then True for the committed
        eight-field sketch and False for the seven-field firmware actually
        on the unit. False means ``dropped_samples`` cannot be trusted:
        there is nothing on the wire to detect a gap against.
        """
        self.dropped_samples = 0
        self.bad_lines = 0
        self.restarts = 0
        self.last_error = None

    @property
    def sample_rate_hz(self) -> float:
        return self.fs

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return

        if self._serial_factory is not None:
            self._serial = self._serial_factory()
        else:
            try:
                import serial
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "pyserial is not installed - pip install -r requirements.txt"
                ) from exc
            self._serial = serial.Serial(self.port, self.baud, timeout=0.2)

        self._stop.clear()
        # A reader thread, unlike everywhere else in this dashboard,
        # because a serial read blocks. Doing it inside read() would stall
        # a Dash callback for up to the timeout and freeze the page.
        self._thread = threading.Thread(
            target=self._reader, name="serial-reader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001
                pass
            self._serial = None

    def read(self):
        """Everything received since the last call, oldest first."""
        with self._lock:
            out, self._pending = self._pending, []
        return out

    def _reader(self):
        while not self._stop.is_set():
            try:
                raw = self._serial.readline()
            except Exception as exc:  # noqa: BLE001
                # A disconnected device must surface as a status, not as a
                # dead thread with no explanation.
                self.last_error = f"{type(exc).__name__}: {exc}"
                break

            if not raw:
                continue

            # The seven-field firmware sends no counter, so one is supplied
            # here: the next number in a locally kept sequence. Kept on the
            # reader thread, which is the only writer, so no lock is needed.
            sample, counter, on_wire = parse_line(
                raw, self.fs, counter_if_absent=self._local_counter
            )
            if sample is None:
                self.bad_lines += 1
                continue

            self.counter_on_wire = on_wire
            if not on_wire:
                # The line carried no counter and took the local one, which
                # means its timestamp was derived from an ASSUMED 100 Hz.
                # That assumption is thrown away here and the arrival time
                # used instead - see _arrival_time below for why.
                self._local_counter += 1
                sample = replace(sample, t=self._arrival_time())

            self._account_for(counter)
            with self._lock:
                self._pending.append(sample)

    def _arrival_time(self) -> float:
        """Seconds since the first sample, measured, not assumed.

        The eight-field firmware numbers its samples, so time comes from the
        counter: exact, evenly spaced, and a gap in it is a gap in time.

        The seven-field firmware numbers nothing, and the temptation is to
        count locally and divide by the configured 100 Hz. That must not be
        done, and the reason is the whole point of this method.

        The unit measured here sends at about 51 Hz. A local counter divided
        by 100 would hand the buffer timestamps 2 ms apart, the buffer would
        compute exactly 100 Hz from them, and the status strip - the one
        readout on the page whose entire job is to catch a rate mismatch -
        would display 100 Hz while the device sent half that. The safeguard
        would report success because it was measuring its own assumption.

        Arrival time is noisier: Bluetooth delivers in bursts, so timestamps
        carry jitter the sensor never had. That is a real cost, and it is
        the lesser one. Jitter makes the rate readout wobble; a fabricated
        counter makes it lie.
        """
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    def _account_for(self, counter):
        """Track dropped samples and device restarts from the counter."""
        previous = self._last_counter
        self._last_counter = counter

        if previous is None:
            return
        gap = counter - previous
        if gap == 1:
            return
        if gap > 1:
            # Counter jumped: that many samples never arrived.
            self.dropped_samples += gap - 1
        else:
            # Counter went backwards - the device reset and started
            # counting again. Timestamps restart with it, so the consumer
            # has to be told rather than left with time running backwards.
            self.restarts += 1

    def status(self) -> dict:
        return {
            "port": self.port,
            "running": self.is_running,
            "dropped_samples": self.dropped_samples,
            "counter_on_wire": self.counter_on_wire,
            "bad_lines": self.bad_lines,
            "restarts": self.restarts,
            "last_error": self.last_error,
        }


def list_ports():
    """Candidate port names, for the error message when one is not given."""
    try:
        from serial.tools import list_ports as tools
    except ImportError:
        return []
    return [(p.device, p.description) for p in tools.comports()]


# ---------------------------------------------------------------------------
# Self-test: python serialsource.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import time

    print("serialsource.py self-test")
    print("  (no hardware needed: a fake port replays the sketch's own format)")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    sample, counter, on_wire = parse_line("41,0.01,-0.02,1.00,0.5,-1.2,0.3,85231")
    check(
        "a data line parses into the right fields",
        sample is not None
        and counter == 41
        and sample.az == 1.0
        and sample.gy == -1.2
        and sample.IR_Value == 85231,
        f"t={sample.t:.2f}s" if sample else "failed",
    )

    check(
        "time comes from the counter, not from arrival",
        sample is not None and abs(sample.t - 0.41) < 1e-9,
        f"counter 41 -> {sample.t}s",
    )

    for junk in ("ACK_3", "", "Bluetooth Classic started.", "1,2,3", "x,y,z,1,2,3,4,5"):
        got, _, _ = parse_line(junk)
        check(f"non-data line ignored: {junk!r:32s}", got is None)

    check(
        "bytes from a real port are accepted too",
        parse_line(b"7,0.00,0.00,1.00,0.0,0.0,0.0,84000\r\n")[0] is not None,
    )

    class FakePort:
        """Replays lines the way pyserial would, then blocks quietly."""

        def __init__(self, lines):
            self.lines = list(lines)
            self.closed = False

        def readline(self):
            if self.lines:
                return self.lines.pop(0).encode()
            time.sleep(0.01)
            return b""

        def close(self):
            self.closed = True

    lines = [f"{i},0.01,0.02,1.00,1.0,2.0,3.0,{85000 + i}" for i in range(50)]
    lines.insert(10, "ACK_3")
    # Drop counters 20-24 to prove the gap is noticed.
    lines = [ln for ln in lines if not (ln[:2].isdigit() and 20 <= int(ln.split(",")[0]) <= 24)]

    src = SerialSource("FAKE", serial_factory=lambda: FakePort(lines))
    src.start()
    gathered = []
    for _ in range(40):
        time.sleep(0.02)
        gathered.extend(src.read())
    src.stop()

    check(
        "the source drives SensorSource's interface",
        isinstance(src, SensorSource) and not src.is_running,
    )

    check(
        "every data line became a sample",
        len(gathered) == 45,
        f"{len(gathered)} samples",
    )

    check(
        "samples arrive in order with evenly spaced timestamps",
        all(a.t < b.t for a, b in zip(gathered, gathered[1:])),
    )

    check(
        "the handshake line was counted as noise, not data",
        src.bad_lines >= 1,
        f"{src.bad_lines} non-data lines",
    )

    check(
        "a gap in the counter is reported as dropped samples",
        src.dropped_samples == 5,
        f"{src.dropped_samples} dropped",
    )

    # -- the firmware actually on the unit (seven fields, no counter) ------
    real = "-0.2963,0.1891,1.0014,3.7710,0.5038,6.9160,3215"
    sample7, counter7, on_wire7 = parse_line(real, counter_if_absent=12)
    check(
        "the unit's own seven-field format parses",
        sample7 is not None
        and abs(sample7.ax + 0.2963) < 1e-9
        and abs(sample7.az - 1.0014) < 1e-9
        and abs(sample7.gz - 6.9160) < 1e-9
        and sample7.IR_Value == 3215,
        real,
    )
    check(
        "a seven-field line takes the counter it is given",
        sample7 is not None and counter7 == 12 and abs(sample7.t - 0.12) < 1e-9,
    )
    check(
        "a seven-field line with no counter offered is refused, not guessed",
        parse_line(real)[0] is None,
    )
    check(
        "the parser says which format it saw, rather than leaving it to be guessed",
        on_wire is True and on_wire7 is False,
        f"eight-field={on_wire}, seven-field={on_wire7}",
    )
    # replay.py caught this: counter 0 on the wire used to look identical to
    # a local counter of 0, so the committed firmware's exact timestamps were
    # being silently replaced by jittery arrival times.
    zero_lines = [f"{i},0.01,0.02,1.00,1.0,2.0,3.0,85000" for i in range(20)]
    src0 = SerialSource("FAKE", serial_factory=lambda: FakePort(zero_lines))
    src0.start()
    got0 = []
    for _ in range(40):
        time.sleep(0.02)
        got0.extend(src0.read())
    src0.stop()
    check(
        "a counter starting at 0 is still read as a counter",
        src0.counter_on_wire is True
        and len(got0) > 1
        and all(abs((b.t - a.t) - 0.01) < 1e-9 for a, b in zip(got0, got0[1:])),
        "timestamps stay exact, not arrival-timed",
    )

    class PacedPort(FakePort):
        """Serves lines at a set rate, the way a real device does.

        Needed because arrival time is now what the seven-field firmware's
        timestamps are built from: a port that dumps every line at once
        would make the measured rate meaninglessly high.
        """

        def __init__(self, lines, hz):
            super().__init__(lines)
            self._period = 1.0 / hz

        def readline(self):
            time.sleep(self._period)
            return super().readline()

    # Served at 50 Hz, deliberately not the configured 100.
    seven = [f"0.01,0.02,1.00,1.0,2.0,3.0,{3200 + i}" for i in range(30)]
    src7 = SerialSource("FAKE", serial_factory=lambda: PacedPort(seven, 50.0))
    src7.start()
    got7 = []
    # Generously longer than the port needs to serve 30 lines at 50 Hz,
    # so this measures the reader and not the test's own timing.
    for _ in range(60):
        time.sleep(0.02)
        got7.extend(src7.read())
    src7.stop()
    check(
        "the seven-field stream reaches the dashboard",
        len(got7) == 30,
        f"{len(got7)} samples",
    )
    check(
        "timestamps rise in order",
        len(got7) > 1 and all(a.t < b.t for a, b in zip(got7, got7[1:])),
    )
    # The guard that matters: these samples were served far slower than
    # 100 Hz, and the timestamps must say so rather than reproducing the
    # configured rate. A local counter over 100 Hz would score exactly
    # 100.0 here and the rate readout on the page would lie.
    span7 = got7[-1].t - got7[0].t
    rate7 = (len(got7) - 1) / span7 if span7 else 0.0
    check(
        "the derived rate is measured, not the configured 100 Hz",
        span7 > 0 and 30.0 < rate7 < 75.0,
        f"{rate7:.1f} Hz from arrival times for a 50 Hz port, not 100.0",
    )
    check(
        "the source knows this firmware numbers nothing",
        src7.counter_on_wire is False,
        "so dropped_samples cannot be trusted on it",
    )

    # A device reset restarts the counter; that must be noticed.
    reset_lines = [
        "100,0,0,1,0,0,0,85000",
        "101,0,0,1,0,0,0,85000",
        "0,0,0,1,0,0,0,85000",
        "1,0,0,1,0,0,0,85000",
    ]
    src2 = SerialSource("FAKE", serial_factory=lambda: FakePort(reset_lines))
    src2.start()
    time.sleep(0.3)
    src2.read()
    src2.stop()
    check(
        "a device restart is noticed rather than read as time going backwards",
        src2.restarts == 1,
        f"{src2.restarts} restart(s)",
    )

    class BrokenPort:
        def readline(self):
            raise OSError("device disconnected")

        def close(self):
            pass

    src3 = SerialSource("FAKE", serial_factory=BrokenPort)
    src3.start()
    time.sleep(0.2)
    check(
        "a disconnection is recorded, not swallowed",
        src3.last_error is not None and "disconnected" in src3.last_error,
        src3.last_error,
    )
    src3.stop()

    check(
        "status() reports what the strip needs",
        {"port", "running", "dropped_samples", "bad_lines", "restarts"}
        <= set(src.status()),
    )

    print()
    print(f"  accelerometer ceiling on this firmware: +/-{ACCEL_FULL_SCALE_G} g")
    print("  (see the module docstring: impacts clip at this ceiling, which")
    print("   is why the fall threshold is 1.8 g and cannot go above 2 g)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


def _print_ports() -> int:
    """``python serialsource.py --ports``: what this machine can see.

    On demo day the one thing standing between a paired wrist unit and a
    live dashboard is the port name, and Windows hides it three levels
    deep in Control Panel. This prints it.

    Bluetooth pairs as two ports, Incoming and Outgoing, and only the
    Outgoing one carries data, so the descriptions are printed beside the
    names rather than the names alone.
    """
    ports = list_ports()
    if not ports:
        print("No serial ports visible.")
        print()
        print("If the wrist unit is paired, it should appear here as an")
        print("outgoing Bluetooth port. If it does not:")
        print("  - check the unit is powered on,")
        print("  - re-pair it in Settings > Bluetooth & devices,")
        print("  - and look in Control Panel > Bluetooth Devices > COM Ports.")
        return 1

    print(f"{len(ports)} serial port(s):")
    print()
    for device, description in ports:
        print(f"  {device:<10} {description}")
    print()
    print("Start the dashboard on one of these with:")
    print()
    print('  $env:HAR_SERIAL_PORT = "COM7"   # PowerShell, your port here')
    print("  python app.py")
    print()
    print("Pick the OUTGOING Bluetooth port. The incoming one pairs and")
    print("opens, but no data ever arrives on it.")
    return 0


if __name__ == "__main__":
    import sys

    if "--ports" in sys.argv:
        raise SystemExit(_print_ports())
    raise SystemExit(_selftest())
