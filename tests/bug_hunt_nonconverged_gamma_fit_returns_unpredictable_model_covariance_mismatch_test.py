"""Bug hunt: ``gamfit.fit(..., family="gamma")`` on an ordinary multi-smooth
model can return a ``Model`` object that is a **landmine** — every downstream
consumer (``predict``, ``summary``, ``save``→``load``) then aborts with an
*internal-consistency* error the fit itself never raised:

    GamError: Invalid input: UnifiedFitResult inference conditional covariance
              must match top-level covariance_conditional

The fit succeeds (returns a ``Model``, no exception, and ``save`` even writes
the model to disk), but the returned model violates the engine's own
``UnifiedFitResult::validate()`` invariant
(``crates/gam-solve/src/model_types/result_types.rs`` lines ~1652-1662), which
requires the inference block's ``beta_covariance`` to be *bit-for-bit equal*
(``**cov == *top``) to the top-level ``covariance_conditional``. Because
``validate()`` runs on every predict / summary / load, the model is unusable
from the moment it is handed back.

Root cause read: these are the non-converged / indefinite-Hessian gamma fits
(``[INDEF-HESS]`` + ``edf#1788`` railed-smoothing recovery). Gamma routes
through the custom-family "robust never-fail" path
(``crates/gam-custom-family/src/fit.rs``), which on an indefinite Hessian falls
back to a sampled/inflated covariance and reassigns the top-level
``covariance_conditional`` (``fit.rs`` ~1814 / ~1848,
``cov.mapv_inplace(|v| v * inflation)``) **without** updating the paired
``inference.beta_covariance`` to match — so the two copies diverge and the
exact-equality invariant trips. The never-fail guarantee returns the model
anyway. (This is the covariance-mismatch sibling of the EDF-collapse symptom
fixed in #1788 on the same non-converged multi-smooth gamma class.)

Contract asserted here (well-posed and family-agnostic): **if ``fit`` returns a
``Model`` without raising, that model must be usable** — ``predict`` on the
training frame must not raise an internal-consistency error and must yield a
finite response mean per row. Seeds 0, 4 and 16 (with the data-generating code
below, n=150, ``y ~ s(x1)+s(x2)``, ``family="gamma"``) deterministically
produce the landmine today; the loop below sweeps seeds 0..19 and asserts every
returned model predicts. It currently fails (multiple returned models raise the
covariance-mismatch ``GamError`` at predict time); once the fit keeps the two
covariance blocks consistent (or refuses to return a model it cannot validate),
every returned model predicts and the test passes without edits.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _make_frame(seed: int, n: int = 150) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    mu = np.exp(0.5 + 1.5 * np.sin(2.0 * np.pi * x1) + np.cos(3.0 * x2))
    y = rng.gamma(2.0, mu / 2.0)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    # Fits emit a large volume of solver diagnostics on stdout/stderr; keep the
    # test output readable without swallowing exceptions.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return fn(*args, **kwargs)


def test_returned_gamma_model_is_predictable() -> None:
    formula = "y ~ s(x1) + s(x2)"
    failures: list[str] = []
    n_models = 0

    for seed in range(20):
        frame = _make_frame(seed)
        try:
            model = _quiet(gamfit.fit, frame, formula, family="gamma")
        except Exception:  # noqa: BLE001 - a raised fit is not this bug
            # The bug is specifically about a *returned* (accepted) model, so a
            # fit that legitimately raises is out of scope for this contract.
            continue

        n_models += 1
        # `fit` handed us a Model with no exception: it must be usable.
        try:
            out = _quiet(model.predict, frame)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"seed={seed}: predict raised {type(exc).__name__}: {exc}")
            continue

        # A bare point prediction is a 1-D response-scale ndarray.
        means = np.asarray(out, dtype=float).reshape(-1)
        if means.shape[0] != len(frame):
            failures.append(
                f"seed={seed}: predict returned {means.shape[0]} rows for {len(frame)} inputs"
            )
        elif not np.all(np.isfinite(means)):
            failures.append(f"seed={seed}: predict returned non-finite means")

    assert n_models > 0, "expected at least one gamma fit to return a Model"
    assert not failures, (
        "fit() returned Model objects that are not usable — a returned model "
        "must predict without an internal-consistency error:\n  "
        + "\n  ".join(failures)
    )
