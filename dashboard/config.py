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

# --- Display formatting -----------------------------------------------------
# Presentation only. The model's class names are its own - "downstairs",
# "k-nearest neighbours" - and nothing here may change the strings used to
# look a class up, index a probability or name a file. These helpers are
# applied at the edge, where a value becomes text on screen, so the data
# path keeps the model's spelling and the page keeps the reader's.

DISPLAY_ACRONYMS = frozenset({"svm", "ir", "bpm", "hz", "knn", "imu", "ppg"})
"""Words that are initialisms, and stay fully capitalised."""


def pretty_label(text) -> str:
    """Title-case a name for display: ``downstairs`` -> ``Downstairs``.

    Only ever raises the first letter of each word; the rest of a word is
    left exactly as it was, so a name that already carries meaningful
    capitals keeps them. Underscores become spaces, hyphenated words are
    capitalised either side of the hyphen ("k-nearest" -> "K-Nearest"),
    and anything in :data:`DISPLAY_ACRONYMS` is upper-cased whole.

    ``str.title()`` is deliberately not used: it lowercases the rest of
    each word, which turns "SVM" into "Svm".
    """
    if text is None:
        return ""
    words = str(text).replace("_", " ").split()
    out = []
    for word in words:
        if word.lower() in DISPLAY_ACRONYMS:
            out.append(word.upper())
        elif "-" in word:
            out.append("-".join(_cap(part) for part in word.split("-")))
        else:
            out.append(_cap(word))
    return " ".join(out)


def sentence_case(text) -> str:
    """Raise the first letter only: ``live (device)`` -> ``Live (device)``.

    For status phrases, where title-casing every word would read like a
    headline rather than a state.
    """
    if text is None:
        return ""
    text = str(text)
    return text[:1].upper() + text[1:] if text else text


def _cap(word: str) -> str:
    """First letter up, rest untouched."""
    return word[:1].upper() + word[1:] if word else word


# --- Server -----------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 8050
