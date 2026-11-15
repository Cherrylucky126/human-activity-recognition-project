"""Running the trained classifier on live windows.

PR 10 of 14.

This file is the seam between the dashboard and the machine-learning work
in ``data-processing/``. Everything here is about *using* what was trained;
nothing here trains, retrains or alters a model, and no feature is computed
by this file. The feature code is imported from the project's own
``data-preprocessing/feature-extraction`` and called unchanged - if the
dashboard computed its own features they would drift from the ones the
models were fitted on, and the predictions would quietly stop meaning
anything.

The order of operations, and why each step is there
---------------------------------------------------
1. A 400-sample window arrives from ``buffer.py`` - 4 seconds, the same
   shape the models were trained on.
2. The project's own feature function turns it into 43 named features.
3. Those are arranged **in the order the scaler declares**, not in
   dictionary order. The scaler carries ``feature_names_in_``, so the
   correct order is read from the artefact rather than assumed. Getting
   this wrong would not raise: it would feed the gyroscope's energy into
   the slot trained for the accelerometer's, and still return a confident
   answer.
4. ``scaler.transform`` then ``model.predict_proba``. Skipping the scaler
   is the classic way to get a model that runs and is wrong - the
   distance-based ones especially.

Three things about this that are not clean, and are not mine to fix
--------------------------------------------------------------------
These are recorded here because they change what the numbers on screen
mean. They belong to the data-processing side of the project.

**1. The random forest will not load.** ``har_random_forest_100hz_4s.pkl``
raises ``KeyError: 13`` when unpickled under scikit-learn 1.9.1 - it was
written by a different version. It is the model the submission checklist
names, so the dashboard falls back to the next one that loads and says on
screen which model it is actually running.

**2. The IR channel was z-scored over the whole dataset before training.**
That happens in the feature file's ``main()``, outside the window
function: mean and standard deviation are taken across every recording at
once. Live, there is no "every recording". Three features depend on that
scaling - ``ir_ac_variance``, ``ir_p2p_amplitude``, and ``ibi_std``
through a peak-prominence threshold expressed in z-score units - so fed
raw IR counts of around 85,000 they come out meaningless.

This file approximates the training-time scaling from a trailing window of
recent IR, which is the best a causal system can do. It is an
approximation, and the remaining 40 features are unaffected. The exact fix
is two numbers: the dataset's IR mean and standard deviation. Drop them
into ``TRAINING_IR_MEAN`` and ``TRAINING_IR_STD`` and the live features
become identical to the training ones.

**3. There is no "running" class.** The models were trained on six:
downstairs, laying, sitting, standing, upstairs, walking. The repository
contains a ``remove-running`` preprocessing step, so this looks deliberate.
Someone running will therefore always be classified as something else,
confidently. Nothing here can fix that; it is worth knowing before the
demonstration.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from dataclasses import dataclass

TRAINING_IR_MEAN: float | None = None
TRAINING_IR_STD: float | None = None
"""The dataset-wide IR mean and standard deviation used during training.

Both ``None`` means they are not known, and the IR features are scaled
from recent history instead - see the module docstring. Filling these in
with the two numbers from the training run makes the live IR features
exactly match the ones the models were fitted on.
"""

IR_REFERENCE_SECONDS = 60.0
"""How much IR history the fallback scaling is estimated from.

Long enough to average over many beats and over posture changes, short
enough to follow a strap being re-tightened.
"""

# Tried in this order; the first that loads wins. Random forest first
# because it is the model the submission names.
MODEL_CANDIDATES = (
    ("random forest", "random-forest-builds/har_random_forest_100hz_4s.pkl",
     "random-forest-builds/har_scaler_rf_100hz_4s.pkl"),
    ("SVM", "svm-builds/har_svm_100hz_4s.pkl",
     "svm-builds/har_scaler_svm_100hz_4s.pkl"),
    ("logistic regression", "logistic-regression-builds/har_logistic_regression_100hz_4s.pkl",
     "logistic-regression-builds/har_scaler_lr_100hz_4s.pkl"),
    ("k-nearest neighbours", "knn-builds/har_knn_100hz_4s.pkl",
     "knn-builds/har_scaler_knn_100hz_4s.pkl"),
    ("decision tree", "decision-tree-builds/har_decision_tree_100hz-4s.pkl",
     "decision-tree-builds/har_scaler_rf.pkl"),
)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(_HERE, os.pardir, "data-processing")
# The feature-extraction file has been spelled three ways during the project:
# no extension at all, then `feature-extraction.py`, and `feature_extraction.py`
# is the importable name it would get if it is ever tidied up again. All three
# are tried, in the order a reader would expect, because a rename in
# data-preprocessing/ must never be what stops the dashboard predicting. The
# first one that exists wins; FEATURE_MODULE_PATH keeps naming the one in use
# so the rest of the module and the error messages are unchanged.
FEATURE_MODULE_NAMES = (
    "feature_extraction.py",
    "feature-extraction.py",
    "feature-extraction",
)
FEATURE_MODULE_DIR = os.path.join(_HERE, os.pardir, "data-preprocessing")


def feature_module_path():
    """Return the first spelling of the feature file that exists.

    Falls back to the last candidate when none is present, so the caller
    reports a missing file by a sensible name instead of raising here.
    """
    for name in FEATURE_MODULE_NAMES:
        candidate = os.path.normpath(os.path.join(FEATURE_MODULE_DIR, name))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(
        os.path.join(FEATURE_MODULE_DIR, FEATURE_MODULE_NAMES[-1])
    )


FEATURE_MODULE_PATH = feature_module_path()


@dataclass(frozen=True)
class Prediction:
    """One classified window."""

    label: str
    confidence: float
    probabilities: dict
    window_index: int = -1

    @property
    def percent(self) -> int:
        return round(self.confidence * 100)


def load_feature_extractor():
    """Import the project's feature function and return it.

    The file is loaded directly by path rather than imported, because for
    most of this project it had no ``.py`` extension and ``import`` could
    not find it. It is now ``feature-extraction.py``, which a plain import
    still cannot take (the hyphen is not a valid identifier), so loading by
    path remains the right mechanism and also means the dashboard never has
    to care which spelling is on disk - see FEATURE_MODULE_NAMES.

    The file is read and executed exactly as written, so the dashboard
    computes features with the same code the training run did.

    Returns ``None`` if the file is missing or will not execute, so the
    dashboard can come up and report the problem instead of refusing to
    start.
    """
    path = feature_module_path()
    if not os.path.exists(path):
        return None
    try:
        loader = importlib.machinery.SourceFileLoader("har_featex", path)
        spec = importlib.util.spec_from_loader("har_featex", loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return getattr(module, "extract_features_from_window", None)
    except Exception:  # noqa: BLE001 - a broken import must not stop the page
        return None


def load_model():
    """Load the first model and scaler that will unpickle.

    Returns ``(name, model, scaler, notes)``. ``model`` is ``None`` when
    nothing loaded, and ``notes`` lists what was tried and why each failed,
    so the reason reaches the screen rather than only a terminal.
    """
    notes = []
    try:
        import joblib
    except ImportError:
        return None, None, None, ["joblib is not installed - pip install -r requirements.txt"]

    for name, model_rel, scaler_rel in MODEL_CANDIDATES:
        model_path = os.path.normpath(os.path.join(MODELS_ROOT, model_rel))
        scaler_path = os.path.normpath(os.path.join(MODELS_ROOT, scaler_rel))
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            notes.append(f"{name}: file missing")
            continue
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        return name, model, scaler, notes

    return None, None, None, notes


class ActivityClassifier:
    """Turns analysis windows into predictions, or explains why it cannot."""

    def __init__(self) -> None:
        self.extract = load_feature_extractor()
        self.name, self.model, self.scaler, self.notes = load_model()
        self._ir_history: list[float] = []
        self._last: Prediction | None = None
        self._errors = 0

        self.feature_order = []
        if self.scaler is not None:
            names = getattr(self.scaler, "feature_names_in_", None)
            if names is not None:
                self.feature_order = [str(n) for n in names]

    @property
    def ready(self) -> bool:
        return (
            self.extract is not None
            and self.model is not None
            and self.scaler is not None
            and bool(self.feature_order)
        )

    @property
    def unavailable_reason(self) -> str:
        """One line for the screen when no prediction is possible."""
        if self.extract is None:
            return "feature extraction code not found"
        if self.model is None:
            first = self.notes[0] if self.notes else "no model loaded"
            return f"no model loaded ({first})"
        if not self.feature_order:
            return "scaler does not record its feature names"
        return ""

    @property
    def classes(self) -> list:
        if self.model is None:
            return []
        return [str(c) for c in getattr(self.model, "classes_", [])]

    @property
    def last(self) -> Prediction | None:
        return self._last

    def note_ir(self, values) -> None:
        """Keep recent IR for the fallback scaling. Cheap; call it per poll."""
        if TRAINING_IR_MEAN is not None and TRAINING_IR_STD:
            return
        self._ir_history.extend(float(v) for v in values)
        limit = int(IR_REFERENCE_SECONDS * 100)
        if len(self._ir_history) > limit:
            del self._ir_history[: len(self._ir_history) - limit]

    def _scale_ir(self, window_ir):
        """Put IR on the scale the models were trained with.

        Uses the training constants when they are known. Otherwise
        standardises against recent history - an approximation of the
        dataset-wide z-score, and the closest a live system can get.
        """
        import numpy as np

        values = np.asarray(window_ir, dtype=float)
        if TRAINING_IR_MEAN is not None and TRAINING_IR_STD:
            return (values - TRAINING_IR_MEAN) / (TRAINING_IR_STD + 1e-8)

        reference = self._ir_history if len(self._ir_history) >= 200 else list(values)
        reference = np.asarray(reference, dtype=float)
        mean = float(reference.mean())
        std = float(reference.std())
        return (values - mean) / (std + 1e-8)

    def predict(self, window) -> Prediction | None:
        """Classify one ``buffer.AnalysisWindow``.

        Returns ``None`` when no prediction could be made. A failure here
        is recorded and swallowed: one bad window must not take down the
        page, and the previous prediction stays on screen rather than the
        display flickering to blank and back.
        """
        if not self.ready:
            return None

        try:
            import numpy as np

            features = self.extract(window.imu, self._scale_ir(window.ir), 100)
            row = [float(features[name]) for name in self.feature_order]
            vector = np.asarray([row], dtype=float)
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

            scaled = self.scaler.transform(vector)
            classes = self.classes

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(scaled)[0]
                best = int(probabilities.argmax())
                label = classes[best]
                confidence = float(probabilities[best])
                probs = {c: float(p) for c, p in zip(classes, probabilities)}
            else:
                # Some classifiers have no probabilities. Reporting 1.0
                # would be a lie, so the bar chart gets one filled bar and
                # the confidence reads as unknown.
                label = str(self.model.predict(scaled)[0])
                confidence = float("nan")
                probs = {c: (1.0 if c == label else 0.0) for c in classes}

            prediction = Prediction(
                label=label,
                confidence=confidence,
                probabilities=probs,
                window_index=getattr(window, "index", -1),
            )
            self._last = prediction
            return prediction
        except Exception:  # noqa: BLE001
            self._errors += 1
            return None


# ---------------------------------------------------------------------------
# Self-test: python inference.py
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import time

    try:
        from buffer import SampleBuffer
        from datasource import SimulatedSource
    except ImportError:  # pragma: no cover
        from dashboard.buffer import SampleBuffer
        from dashboard.datasource import SimulatedSource

    print("inference.py self-test")
    failures = []

    def check(label, condition, detail=""):
        mark = "ok  " if condition else "FAIL"
        print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
        if not condition:
            failures.append(label)

    clf = ActivityClassifier()

    check(
        "the project's own feature function was imported, not reimplemented",
        clf.extract is not None,
        os.path.basename(feature_module_path()),
    )

    # Regression guard. On 15 Nov the feature file was renamed from
    # `feature-extraction` to `feature-extraction.py` - a fix that had been
    # asked for - and the dashboard silently stopped loading any model at all,
    # because the path here was pinned to the old spelling. The page still came
    # up and still looked healthy; it just never predicted. Renaming a file in
    # someone else's directory must not be able to do that again.
    _renames_ok = True
    _real = feature_module_path()
    for _name in FEATURE_MODULE_NAMES:
        _candidate = os.path.normpath(os.path.join(FEATURE_MODULE_DIR, _name))
        if os.path.exists(_candidate) and _candidate != _real:
            _renames_ok = False
    check(
        "the feature file is found whichever way it is spelled",
        os.path.exists(_real) and _renames_ok,
        f"{os.path.basename(_real)} of {len(FEATURE_MODULE_NAMES)} spellings tried",
    )

    check(
        "a model and its scaler loaded",
        clf.ready,
        f"using the {clf.name}" if clf.ready else clf.unavailable_reason,
    )

    if clf.notes:
        print("  --   models that would not load:")
        for note in clf.notes:
            print(f"         {note}")

    if not clf.ready:
        print("\nFAILED: no model available")
        return 1

    check(
        "feature order is taken from the scaler, not assumed",
        len(clf.feature_order) == 43,
        f"{len(clf.feature_order)} features",
    )

    # Feed real simulated motion through the whole chain.
    src = SimulatedSource(activity="walking", seed=3)
    src.start()
    buf = SampleBuffer()
    elapsed = 0.0
    while elapsed < 7.0:
        time.sleep(0.05)
        elapsed += 0.05
        readings = [s.as_dict() for s in src.read()]
        buf.extend(readings)
        clf.note_ir(r["IR_Value"] for r in readings)
    src.stop()

    windows = buf.drain_windows()
    check("the buffer produced windows to classify", len(windows) >= 1, f"{len(windows)}")

    prediction = clf.predict(windows[0])
    check("a window classifies without raising", prediction is not None)

    if prediction is None:
        print("\nFAILED: prediction failed")
        return 1

    check(
        "the label is one of the model's own classes",
        prediction.label in clf.classes,
        f"{prediction.label} at {prediction.percent}%",
    )

    total = sum(prediction.probabilities.values())
    check(
        "probabilities cover every class and sum to 1",
        len(prediction.probabilities) == len(clf.classes) and abs(total - 1.0) < 1e-6,
        f"{len(prediction.probabilities)} classes, sum {total:.4f}",
    )

    check(
        "there is no 'running' class - anyone running will be misread",
        "running" not in clf.classes,
        ", ".join(clf.classes),
    )

    # Every remaining window must classify too - no window shape surprises.
    ok = all(clf.predict(w) is not None for w in windows)
    check(f"all {len(windows)} windows classify", ok)

    # A broken window must be survived, not fatal.
    class Broken:
        imu = "not an array"
        ir = None
        index = 99

    before = clf._errors
    check(
        "a malformed window is survived, not raised",
        clf.predict(Broken()) is None and clf._errors == before + 1,
    )

    check(
        "the last good prediction survives a failure",
        clf.last is not None and clf.last.label in clf.classes,
    )

    print()
    print(f"  model in use: {clf.name}")
    print(f"  classes     : {', '.join(clf.classes)}")
    if TRAINING_IR_MEAN is None:
        print("  IR scaling  : approximated from recent history "
              "(set TRAINING_IR_MEAN/STD for an exact match)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
