"""One convergence-fuzz rep in a fresh process: one (case, family, n).

Usage: python -m convergence_fuzz.worker CASE FAMILY N   (``bench/`` on the path)

Prints exactly one ``RESULT {json}`` line on stdout. The driver (``run.py``)
launches it through ``pygam_compare.run.run_isolated``: pinned single-thread
environment, process-tree RSS polling, and a scratch working directory with
only ``bench/`` on ``PYTHONPATH``, so the installed gamfit wheel is imported
rather than the source tree's ``./gamfit``.

The rep fits the case's additive model twice:

``fit``
    the default fit of the drawn training table;
``refit``
    the same model on the same rows after a row permutation and a positive
    affine map ``x -> a x + b`` of every covariate. A B-spline basis on the
    data range with data-driven knots is equivariant under both, and a
    derivative penalty only rescales, so the refit's REML/LAML optimum is the
    same function with the same criterion value. The refit starts the outer
    search from the engine's own seed for the rescaled problem; a criterion
    that differs beyond the certificate's tolerance means one of the two
    searches stopped short of the optimum or the optimum is not unique.

Every phase that raises is recorded with its exception type and message; the
rep never retries or relaxes anything.
"""

from __future__ import annotations

import importlib
import json
import resource
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from convergence_fuzz import dgp

INTERVAL_LEVEL = 0.95
# Interval finiteness is checked on this many held-out rows; interval speed is
# the pygam_compare harness's job, not this one's.
INTERVAL_ROWS = 500
REFIT_SEED_OFFSET = 7_919
# Families whose default prediction, the posterior mean E[exp(eta) | data],
# can exceed the float64 range with every input finite.
LOG_LINK_FAMILIES = ("poisson",)
LOG_DBL_MAX = float(np.log(np.finfo(np.float64).max))


def _cpu() -> float:
    return time.process_time()


def _summary_fields(model: Any) -> dict[str, Any]:
    summ = model.summary()
    conv = summ.convergence
    out: dict[str, Any] = {
        "reml_score": summ.reml_score,
        "raw_reml_score": summ.raw_reml_score,
        "reml_score_unavailable": summ.reml_score_unavailable,
        "edf": summ.edf_total,
        "lambdas": [float(v) for v in summ.lambdas],
        "convergence": json.loads(json.dumps(conv, default=str)),
        "certified": None if conv is None else bool(conv.get("certified")),
        "notes": list(summ.notes),
    }
    return out


def _overflows_exactly(
    model: Any, test: dict[str, np.ndarray], pred: np.ndarray
) -> bool:
    """Whether every non-finite log-link posterior mean is the correctly
    rounded value of a number beyond float64.

    Under the conditional Gaussian posterior of ``eta`` the log-link posterior
    mean is exactly ``exp(eta + Var(eta) / 2)``. Both pieces come from the
    fit's own affine design and conditional covariance, the same pair the
    engine prices the prediction from, so a far extrapolation whose exact
    value is past ``DBL_MAX`` must round to ``+inf``; anything else that is
    non-finite is a defect.
    """
    bad = ~np.isfinite(pred)
    if not np.all(np.isposinf(pred[bad])):
        return False
    design = model.design_matrix({k: v[bad] for k, v in test.items()})
    if design.covariance_conditional is None:
        return False
    eta = design.offset + design.matrix @ design.coefficients
    var = np.einsum(
        "ij,jk,ik->i",
        design.eta_gradient,
        design.covariance_conditional,
        design.eta_gradient,
    )
    return bool(np.all(eta + 0.5 * var > LOG_DBL_MAX))


def _refit_table(
    train: dict[str, np.ndarray], names: list[str], seed: tuple[int, ...]
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train["y"].size)
    out: dict[str, np.ndarray] = {"y": train["y"][perm]}
    for nm in names:
        a = float(10.0 ** rng.uniform(-1.0, 1.0))
        b = float(rng.normal(0.0, 10.0))
        out[nm] = a * train[nm][perm] + b
    return out


def run(case: int, family: str, n: int) -> dict[str, Any]:
    data = dgp.draw(case, family, n)
    spec = data.spec
    out: dict[str, Any] = {
        "formula": spec.formula,
        "p": spec.p,
        "shape_mix": spec.describe(),
        "spec": dgp.spec_json(spec),
        "unique_counts": [int(np.unique(data.train[nm]).size) for nm in spec.names],
    }
    errors: dict[str, str] = {}
    error_types: dict[str, str] = {}

    def phase(name: str, fn: Callable[[], Any]) -> Any:
        t = _cpu()
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - every raise is a finding
            errors[name] = f"{type(exc).__name__}: {exc}"[:2000]
            errors[f"{name}_traceback"] = traceback.format_exc(limit=6)[-3000:]
            error_types[name] = type(exc).__name__
            return None
        out[f"{name}_cpu_s"] = _cpu() - t
        return value

    gamfit: Any = phase("import", lambda: importlib.import_module("gamfit"))
    if gamfit is not None:
        out["lib_version"] = str(gamfit.__version__)
        model = phase(
            "fit", lambda: gamfit.fit(data.train, spec.formula, family=family)
        )
        if model is not None:
            info = phase("summary", lambda: _summary_fields(model))
            if info is not None:
                out.update(info)
            pred = phase(
                "predict",
                lambda: np.asarray(model.predict(data.test), dtype=float).reshape(-1),
            )
            if pred is not None:
                out["pred_finite"] = bool(np.all(np.isfinite(pred)))
                if out["pred_finite"]:
                    out["rmse_mu"] = float(np.sqrt(np.mean((pred - data.mu_test) ** 2)))
                elif family in LOG_LINK_FAMILIES:
                    exact = phase(
                        "predict_exact",
                        lambda: _overflows_exactly(model, data.test, pred),
                    )
                    if exact is not None:
                        out["pred_nonfinite_exact"] = exact
            head = {k: v[:INTERVAL_ROWS] for k, v in data.test.items()}
            iv = phase(
                "interval",
                lambda: model.predict(
                    head, interval=INTERVAL_LEVEL, return_type="dict"
                ),
            )
            if iv is not None:
                lo = np.asarray(iv["posterior_mean_lower"], dtype=float)
                hi = np.asarray(iv["posterior_mean_upper"], dtype=float)
                ok = np.isfinite(lo) & np.isfinite(hi)
                out["interval_finite"] = bool(np.all(ok))
                if out["interval_finite"]:
                    mu = data.mu_test[: lo.size]
                    out["coverage"] = float(np.mean((mu >= lo) & (mu <= hi)))
        refit_train = _refit_table(
            data.train, spec.names, (dgp.ROOT_SEED, case, n, REFIT_SEED_OFFSET)
        )
        refit = phase(
            "refit", lambda: gamfit.fit(refit_train, spec.formula, family=family)
        )
        if refit is not None:
            rinfo = phase("refit_summary", lambda: _summary_fields(refit))
            if rinfo is not None:
                out["refit"] = rinfo
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out["peak_rss_mb"] = usage.ru_maxrss / 1024.0
    out["cpu_user_s"] = usage.ru_utime
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
        out["error_types"] = error_types
    return out


def _finite_json(value: Any) -> Any:
    """NaN/Inf become the string they are: JSON has none, and a silent
    ``null`` would hide a non-finite certificate field from the triage."""
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    case, family, n = int(argv[0]), argv[1], int(float(argv[2]))
    out = run(case, family, n)
    print("RESULT " + json.dumps(_finite_json(out), default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
