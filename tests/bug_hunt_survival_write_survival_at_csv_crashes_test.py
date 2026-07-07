"""Bug hunt: ``SurvivalPrediction.write_survival_at_csv`` is completely
non-functional — it raises unconditionally on any fitted survival model,
while the sibling ``survival_at`` on the *same* object works fine.

A fitted survival model exposes a survival surface ``S(t | x)``. The public
:class:`SurvivalPrediction` object returned by ``model.predict(frame)`` offers
two ways to read that surface at a set of query times:

* ``pred.survival_at(times)`` -> an in-memory ``(n_rows, n_times)`` array, and
* ``pred.write_survival_at_csv(path, times)`` -> the same numbers streamed to a
  CSV (documented for large person x time grids that do not fit in memory).

The first works. The second raises **before writing anything**, via two stacked
defects in ``gamfit/_survival.py::write_survival_at_csv`` (the FFI-surface arm,
lines ~537-556):

1. ``left_value, right_value = _extrapolation_for("survival")`` unpacks a
   3-tuple ``(1.0, None, 0.0)`` into two names -> ``ValueError: too many values
   to unpack (expected 2)``. Every other caller of ``_extrapolation_for``
   (lines 303, 377) correctly unpacks three values ``(left, right, inf)``; this
   site was missed when the ``inf_value`` field was added (#965).

2. Even with (1) fixed, the call then forwards ``left_value``/``right_value`` as
   two extra positional arguments to the Rust ``write_survival_csv``, whose
   ``#[pyfunction]`` signature
   (``crates/gam-pyffi/src/io/survival_surface_io.rs:414``) accepts only
   ``(path, grid, surface, times, id_column, row_ids, people_chunk,
   time_grid_chunk)`` -> ``TypeError: write_survival_csv() takes 8 positional
   arguments but 10 were given``. The Python wrapper was extended to thread an
   extrapolation law into the writer but the Rust function was never updated to
   accept it (an incomplete refactor; the comment at line 540 even says "The
   Rust CSV writer used to hardcode (1.0, 0.0)").

The net effect: ``write_survival_at_csv`` has no working code path. The method
is public, documented, and reachable directly from ``model.predict(...)`` on
every survival model.

This test fits an ordinary right-censored survival model, predicts to obtain a
``SurvivalPrediction``, verifies ``survival_at`` works (the sanity anchor), then
asserts that ``write_survival_at_csv`` writes a valid CSV whose ``survival``
column matches ``survival_at`` for the same query times. When both defects are
fixed (unpack three values; align the Python call with the Rust signature — or
extend the Rust signature to honour the extrapolation law), the CSV is emitted
and this test passes without edits.

Related: #965 (survival ``inf_value`` extrapolation field), #1595 (survival
extrapolation past the grid).
"""

from __future__ import annotations

import csv
import importlib
import tempfile
from pathlib import Path
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _survival_data(seed: int = 314, n: int = 300) -> dict[str, Any]:
    """Ordinary right-censored survival with a single covariate effect."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    lam = np.exp(0.5 * x)
    event_time = rng.exponential(1.0 / lam)
    censor_time = rng.exponential(5.0, n)
    exit_t = np.maximum(np.minimum(event_time, censor_time), 0.11)
    event = (event_time <= censor_time).astype(int)
    return {
        "entry": np.zeros(n),
        "exit": exit_t,
        "event": event,
        "x": x,
    }


def test_write_survival_at_csv_emits_valid_csv() -> None:
    model = gamfit.fit(
        _survival_data(), "Surv(entry, exit, event) ~ s(x)", family="survival"
    )
    pred = model.predict(
        {"entry": [0, 0], "exit": [1, 1], "event": [1, 1], "x": [-0.5, 0.5]}
    )

    query_times = [0.5, 1.0, 2.0]

    # Sanity anchor: the in-memory reader works on this exact object, so the
    # surface exists and is well-posed. Shape is (n_rows, n_times).
    in_memory = np.asarray(pred.survival_at(query_times), dtype=float)
    assert in_memory.shape == (2, len(query_times))
    assert np.all(np.isfinite(in_memory))

    # The CSV writer must produce the same numbers, not raise.
    out_path = Path(tempfile.mkdtemp()) / "survival_at.csv"
    returned = pred.write_survival_at_csv(str(out_path), query_times)

    assert returned == str(out_path)
    assert out_path.exists() and out_path.stat().st_size > 0, (
        "write_survival_at_csv must create a non-empty CSV; "
        f"returned={returned!r}"
    )

    rows = list(csv.reader(out_path.open()))
    header = rows[0]
    assert "survival" in header and "time" in header
    data_rows = rows[1:]
    # 2 people x 3 query times.
    assert len(data_rows) == 2 * len(query_times)

    surv_idx = header.index("survival")
    surv_values = np.array([float(r[surv_idx]) for r in data_rows], dtype=float)
    # Survival probabilities live in [0, 1].
    assert np.all(surv_values >= -1e-9) and np.all(surv_values <= 1.0 + 1e-9)

    # And they must agree with survival_at for the same covariates/times
    # (both read the same fitted surface). CSV is row-major over (person, time).
    assert np.allclose(surv_values, in_memory.reshape(-1), atol=1e-8), (
        "write_survival_at_csv values must match survival_at; "
        f"csv={surv_values.tolist()} in_memory={in_memory.reshape(-1).tolist()}"
    )
