"""Bug hunt: ``inference_notes`` never reach the gamfit Python API — the cr/cs/sz
data-support cap (and every other note) is silently dropped at the FFI boundary.

The cr/cs/sz cap fix (#1541, #1542) records a human-readable note whenever it
reduces ``k`` to the covariate's distinct-value count, or degrades a binary
covariate to the linear marginal. The fix's own contract says (``term_builder.rs``):

    "inference_notes records any reduction so the user sees that k was capped
     (mgcv emits a warning in the same situation)."

That is true for the CLI (``run_fit.rs`` calls ``print_inference_summary``), but
**false for gamfit** — the interface the original bug was filed through. The
Python fit path (``fit_dataset_impl`` in ``crates/gam-pyffi``) consumes
``materialized.request`` and throws away ``materialized.inference_notes``: the
notes appear nowhere in ``crates/gam-pyffi``. So a gamfit user whose ``k`` is
silently halved — or whose cubic-regression smooth is silently demoted to a
straight line — gets ZERO signal, while mgcv would loudly ``warning()``. Silent
basis reduction is a statistical-software footgun.

Expected (mgcv parity):
  * a fit that caps/degrades a cr/cs/sz basis emits a Python ``warnings.warn``
    (category ``gamfit.GamInferenceWarning``) carrying the note text; and
  * the note is retrievable after the fact via ``model.notes``.
  * a clean fit (no reduction) emits no inference warning and has empty notes.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _ternary_data(seed: int = 1542) -> dict:
    rng = np.random.default_rng(seed)
    n = 600
    x = rng.integers(0, 3, n).astype(float)  # 3 distinct values
    y = 0.5 * x + rng.normal(0.0, 0.3, n)
    return {"x": x.tolist(), "y": y.tolist()}


def _binary_data(seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)
    n = 600
    x = rng.integers(0, 2, n).astype(float)  # 2 distinct values
    y = 2.0 * x + rng.normal(0.0, 0.3, n)
    return {"x": x.tolist(), "y": y.tolist()}


def test_gamfit_exposes_inference_warning_category() -> None:
    # The public warning category must exist and be a UserWarning subclass so it
    # is visible by default (mgcv warnings are not silenced by default either).
    assert issubclass(gamfit.GamInferenceWarning, UserWarning)


def test_cr_cap_emits_warning_and_exposes_note_on_model() -> None:
    d = _ternary_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gamfit.fit(d, "y ~ s(x, bs='cr', k=10)")
    notes = [str(w.message) for w in caught if issubclass(w.category, gamfit.GamInferenceWarning)]
    assert notes, "no GamInferenceWarning emitted for a capped cr basis"
    joined = " ".join(notes).lower()
    assert "reduced" in joined or "cap" in joined, f"warning lacks cap wording: {notes}"
    assert "distinct" in joined, f"warning lacks the distinct-value reason: {notes}"
    # ...and the same note is retrievable from the fitted model after the fact.
    model_notes = list(model.notes)
    assert any(("reduced" in n.lower() or "cap" in n.lower()) for n in model_notes), (
        f"model.notes does not carry the cap note: {model_notes}"
    )


def test_binary_degradation_warns() -> None:
    d = _binary_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gamfit.fit(d, "y ~ s(x, bs='cr', k=10)")
    notes = " ".join(
        str(w.message) for w in caught if issubclass(w.category, gamfit.GamInferenceWarning)
    ).lower()
    assert "degrad" in notes or "linear" in notes, (
        f"binary degradation not surfaced as a warning: {notes!r}"
    )
    assert any("degrad" in n.lower() or "linear" in n.lower() for n in model.notes)


def test_clean_fit_has_no_inference_notes() -> None:
    # A continuous covariate with plenty of distinct values is never capped, so
    # there must be no inference warning and model.notes is empty. (Guards
    # against a regression that warns spuriously on every fit.)
    rng = np.random.default_rng(7)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, n)
    d = {"x": x.tolist(), "y": y.tolist()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gamfit.fit(d, "y ~ s(x, bs='cr', k=10)")
    inference_warnings = [
        w for w in caught if issubclass(w.category, gamfit.GamInferenceWarning)
    ]
    assert not inference_warnings, f"spurious inference warning on clean fit: {inference_warnings}"
    assert list(model.notes) == [], f"clean fit should have no notes, got {model.notes}"
