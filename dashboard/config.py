"""Pipeline constants for the live dashboard.

These values are read off the committed training code, not chosen here:

  - ``data-preprocessing/feature-extraction`` sets the sampling rate, the
    window length and the overlap.
  - ``data-processing/*-builds/*-maker-100hz-4s.py`` set the metadata columns
    that are dropped before training.

If the dashboard windows the signal differently from the training code, the
models are handed features they were never fitted on. They will still return a
prediction, and it will be wrong, with nothing on screen looking broken. So
these constants must stay in step with the training code.
"""

APP_TITLE = "HAR Wrist Unit — Live Dashboard"

# --- Sampling and windowing -------------------------------------------------
# NOTE: 100 Hz and a 4 s window, NOT the 50 Hz / 2 s in the earlier project
# documents. The trained models in data-processing/ are all built at 100 Hz
# with 4-second windows.
SAMPLING_FREQ_HZ = 100
WINDOW_SIZE_SEC = 4
WINDOW_SIZE = SAMPLING_FREQ_HZ * WINDOW_SIZE_SEC          # 400 samples
OVERLAP = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))              # 200 samples

# --- Channels ---------------------------------------------------------------
# Order matters: the feature code indexes these positionally.
SENSOR_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
IR_COLUMN = "IR_Value"

# --- Display ----------------------------------------------------------------
DISPLAY_WINDOW_SEC = 10
BUFFER_SAMPLES = SAMPLING_FREQ_HZ * DISPLAY_WINDOW_SEC    # 1000 samples
TICK_INTERVAL_MS = 100

# --- Server -----------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8050
