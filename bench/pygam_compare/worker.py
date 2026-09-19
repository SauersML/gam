"""One benchmark rep: one library, one (family, n, design, seed) cell, fresh process.

Usage: worker.py LIB FAMILY N DESIGN SEED

  LIB     gamfit | pygam | pygam_gs
  FAMILY  gaussian | binomial | poisson
  DESIGN  p<k> (additive in k covariates, e.g. p1, p3, p5, p20) | te | fz<case>

A ``fz<case>`` design is a convergence-fuzz case (``fuzz_terms.py``): a seeded
term structure — tensor, ``ti``, ``by=``, factor, random-effect, cyclic, 2-D
isotropic, shape-constrained and concurvity terms — fitted with gamfit only;
see :func:`run_fuzz`.

Prints exactly one ``RESULT {json}`` line on stdout. The driver (``run.py``)
launches this script with a pinned thread environment and a scratch working
directory, so the installed gamfit wheel is imported rather than the source
tree's ``./gamfit`` (which has no compiled ``_rust``).

Phases are timed with both ``perf_counter`` (wall) and ``process_time`` (CPU of
this process, all threads). CPU is the primary speed metric: on a loaded or
shared host wall time measures the neighbours as much as the library.

  import    import of the library (numpy/scipy already imported)
  fit       one cold fit (the first fit in the process)
  fit_warm  a second fit of the same data in the same process: the per-fit cost
            once imports, lazy initialisation and caches are paid
  pred      point prediction on ``n`` fresh rows
  interval  95% interval prediction on the same rows

Held-out accuracy is computed on ``n`` fresh rows drawn with ``seed + 1000``,
against both the true mean (``rmse_mu``, ``coverage``) and the drawn response
(``deviance``, ``logscore``). A phase that raises is recorded under ``errors``
and makes the rep's status ``error``; the metrics of the phases that did run
are still reported.
"""

from __future__ import annotations

import importlib
import json
import re
import resource
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

if __package__:
    from . import fuzz_terms
else:  # run as a script by run.py: this directory is sys.path[0]
    import fuzz_terms  # type: ignore[no-redef]

LIBS = ("gamfit", "pygam", "pygam_gs")
FAMILIES = ("gaussian", "binomial", "poisson")
DESIGNS = ("p1", "p5", "p20", "te")
INTERVAL_LEVEL = 0.95
TEST_SEED_OFFSET = 1000
# The fuzz interval phase checks that intervals are finite on this many
# held-out rows; interval speed is measured by the core plans, not here.
FUZZ_INTERVAL_ROWS = 200

FloatArray = NDArray[np.float64]


def design_width(design: str) -> int | None:
    """Covariate count of an additive design ``p<k>``; ``None`` otherwise."""
    match = re.fullmatch(r"p([1-9][0-9]*)", design)
    return None if match is None else int(match.group(1))


def make_data(
    n: int, design: str, family: str, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Draw ``(X, y, mu)``: covariates, response, true response-scale mean.

    Same generators as the pyGAM audit (bench/pygam_audit/speed/worker.py) so
    numbers stay comparable with the audit's speed.md tables.
    """
    rng = np.random.default_rng(seed)
    if design == "te":
        X = rng.uniform(0.0, 1.0, (n, 2))
        eta = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    elif design_width(design) is not None:
        p = design_width(design)
        X = rng.uniform(0.0, 1.0, (n, p))
        eta = np.zeros(n)
        for j in range(p):
            eta += np.sin(2 * np.pi * X[:, j] + j) / np.sqrt(p)
    else:
        raise ValueError(f"unknown design {design!r}; expected p<k> or te")
    if family == "gaussian":
        mu = eta
        y = eta + rng.normal(0.0, 0.5, n)
    elif family == "binomial":
        mu = special.expit(1.5 * eta)
        y = (rng.uniform(size=n) < mu).astype(float)
    elif family == "poisson":
        mu = np.exp(0.5 + 0.7 * eta)
        y = rng.poisson(mu).astype(float)
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    return X, y, mu


def mean_deviance(family: str, y: FloatArray, mu: FloatArray) -> float:
    """Mean unit deviance of held-out ``y`` at predicted mean ``mu``."""
    if family == "gaussian":
        return float(np.mean((y - mu) ** 2))
    if family == "binomial":
        unit = special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu))
        return float(np.mean(2 * unit))
    unit = special.xlogy(y, y / mu) - (y - mu)
    return float(np.mean(2 * unit))


def mean_logscore(
    family: str, y: FloatArray, mu: FloatArray, predictive_sd: FloatArray | None
) -> float:
    """Mean negative log predictive density of held-out ``y`` (lower is better).

    Binomial and Poisson are scored at the predicted mean. Gaussian needs a
    predictive scale: the library's own 95% prediction interval gives it as
    ``(upper - lower) / (2 z_0.975)``, which prices both the fitted scale and
    the posterior variance of the mean.
    """
    if family == "binomial":
        return float(-np.mean(special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu)))
    if family == "poisson":
        return float(-np.mean(stats.poisson.logpmf(y, mu)))
    if predictive_sd is None:
        raise ValueError("gaussian logscore needs a predictive sd")
    return float(-np.mean(stats.norm.logpdf(y, loc=mu, scale=predictive_sd)))


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def rss_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Timer:
    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = time.process_time()

    def stop(self) -> tuple[float, float]:
        return time.perf_counter() - self.wall, time.process_time() - self.cpu


class Adapter:
    """Library-specific fit / predict calls behind one interface."""

    version: str

    def fit(self, X: FloatArray, y: FloatArray) -> None:
        raise NotImplementedError

    def predict(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def interval(
        self, X: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        """Return (lower, upper) of the 95% interval for the mean, and the
        Gaussian predictive sd (``None`` for other families)."""
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        raise NotImplementedError


class GamfitAdapter(Adapter):
    def __init__(self, family: str, design: str, p: int) -> None:
        self.gamfit: Any = importlib.import_module("gamfit")
        self.version = str(self.gamfit.__version__)
        self.family = family
        self.names = [f"x{j}" for j in range(p)]
        if design == "te":
            self.formula = "y ~ te(x0, x1)"
        else:
            self.formula = "y ~ " + " + ".join(f"s({nm})" for nm in self.names)
        self.model: Any = None

    def _table(self, X: FloatArray) -> dict[str, FloatArray]:
        return {nm: X[:, j] for j, nm in enumerate(self.names)}

    def fit(self, X: FloatArray, y: FloatArray) -> None:
        data = self._table(X)
        data["y"] = y
        self.model = self.gamfit.fit(data, self.formula, family=self.family)

    def predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.model.predict(self._table(X)), dtype=float).reshape(-1)

    def interval(
        self, X: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        gaussian = self.family == "gaussian"
        res = self.model.predict(
            self._table(X),
            interval=INTERVAL_LEVEL,
            observation_interval=gaussian,
            return_type="dict",
        )
        lo = np.asarray(res["posterior_mean_lower"], dtype=float)
        hi = np.asarray(res["posterior_mean_upper"], dtype=float)
        sd = None
        if gaussian:
            olo = np.asarray(res["observation_lower"], dtype=float)
            ohi = np.asarray(res["observation_upper"], dtype=float)
            sd = (ohi - olo) / (2 * stats.norm.ppf(0.5 + INTERVAL_LEVEL / 2))
        return lo, hi, sd

    def model_info(self) -> dict[str, Any]:
        summ = self.model.summary()
        conv = getattr(summ, "convergence", None)
        return {
            "edf": None if summ.edf_total is None else float(summ.edf_total),
            "ncoef": None if summ.coefficients is None else len(summ.coefficients),
            "iterations": summ.iterations,
            "convergence": json.loads(json.dumps(conv, default=str)),
        }


class PygamAdapter(Adapter):
    def __init__(self, family: str, design: str, p: int, gridsearch: bool) -> None:
        # pyGAM ships no type information; its API is used as Any.
        pygam: Any = importlib.import_module("pygam")
        self.pygam = pygam
        self.version = str(pygam.__version__)
        self.family = family
        if design == "te":
            self.terms: Any = pygam.te(0, 1)
        else:
            self.terms = pygam.s(0)
            for j in range(1, p):
                self.terms = self.terms + pygam.s(j)
        self.dist, self.link = {
            "gaussian": ("normal", "identity"),
            "binomial": ("binomial", "logit"),
            "poisson": ("poisson", "log"),
        }[family]
        self.gridsearch = gridsearch
        self.model: Any = None

    def fit(self, X: FloatArray, y: FloatArray) -> None:
        if self.family == "gaussian":
            # LinearGAM is GAM(normal, identity) plus prediction_intervals,
            # which the Gaussian log score needs.
            g = self.pygam.LinearGAM(self.terms)
        else:
            g = self.pygam.GAM(self.terms, distribution=self.dist, link=self.link)
        if self.gridsearch:
            g.gridsearch(X, y, progress=False)
        else:
            g.fit(X, y)
        self.model = g

    def predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1)

    def interval(
        self, X: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        ci = np.asarray(
            self.model.confidence_intervals(X, width=INTERVAL_LEVEL), dtype=float
        )
        sd = None
        if self.family == "gaussian":
            pi = np.asarray(
                self.model.prediction_intervals(X, width=INTERVAL_LEVEL), dtype=float
            )
            sd = (pi[:, 1] - pi[:, 0]) / (2 * stats.norm.ppf(0.5 + INTERVAL_LEVEL / 2))
        return ci[:, 0], ci[:, 1], sd

    def model_info(self) -> dict[str, Any]:
        return {
            "edf": float(self.model.statistics_["edof"]),
            "ncoef": int(len(self.model.coef_)),
            "lam": [float(v) for v in _flatten(self.model.lam)],
        }


def run_fuzz(family: str, n: int, design: str, seed: int) -> dict[str, Any]:
    """One convergence-fuzz rep (design ``fz<case>``, gamfit only).

    Same phases as :func:`run`, plus what the fuzz triage needs: the formula,
    the certificate verdict, whether every prediction and interval came back
    finite, and the exception type of a phase that raised, so a typed
    build-time refusal can be told apart from a solver failure.
    """
    data = fuzz_terms.draw(int(design[2:]), n, family, seed)
    out: dict[str, Any] = {"base_rss_mb": rss_peak_mb(), "formula": data.formula}
    errors: dict[str, str] = {}
    error_types: dict[str, str] = {}

    def phase(name: str, fn: Callable[[], Any]) -> Any:
        t = Timer()
        try:
            value = fn()
        except Exception as exc:
            errors[name] = traceback.format_exc(limit=4)[-2000:]
            error_types[name] = type(exc).__name__
            return None
        out[f"{name}_s"], out[f"{name}_cpu_s"] = t.stop()
        return value

    gamfit: Any = phase("import", lambda: importlib.import_module("gamfit"))
    model: Any = None
    if gamfit is not None:
        out["lib_version"] = str(gamfit.__version__)
        train = fuzz_terms.as_frame(data.train, data.categorical)
        model = phase("fit", lambda: gamfit.fit(train, data.formula, family=family))
    if model is not None:
        out["rss_after_fit_mb"] = rss_peak_mb()
        test = fuzz_terms.as_frame(data.test, data.categorical)
        head = fuzz_terms.as_frame(
            {k: v[:FUZZ_INTERVAL_ROWS] for k, v in data.test.items()},
            data.categorical,
        )
        pred = phase(
            "pred", lambda: np.asarray(model.predict(test), dtype=float).reshape(-1)
        )
        iv = phase(
            "interval",
            lambda: model.predict(head, interval=INTERVAL_LEVEL, return_type="dict"),
        )
        summ = phase("info", model.summary)
        if summ is not None:
            conv = getattr(summ, "convergence", None)
            out["convergence"] = json.loads(json.dumps(conv, default=str))
            out["certified"] = None if conv is None else bool(conv.get("certified"))
            out["edf"] = None if summ.edf_total is None else float(summ.edf_total)
        if pred is not None:
            out["pred_finite"] = bool(np.all(np.isfinite(pred)))
            if out["pred_finite"]:
                out["rmse_mu"] = float(np.sqrt(np.mean((pred - data.mu_test) ** 2)))
        if iv is not None:
            lo = np.asarray(iv["posterior_mean_lower"], dtype=float)
            hi = np.asarray(iv["posterior_mean_upper"], dtype=float)
            finite = np.isfinite(lo) & np.isfinite(hi)
            out["interval_finite"] = bool(np.all(finite))
            if out["interval_finite"]:
                mu = data.mu_test[: lo.shape[0]]
                out["coverage"] = float(np.mean((mu >= lo) & (mu <= hi)))
    out["peak_rss_mb"] = rss_peak_mb()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out["cpu_user_s"] = usage.ru_utime
    out["cpu_sys_s"] = usage.ru_stime
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
        out["error_types"] = error_types
    return out


def run(lib: str, family: str, n: int, design: str, seed: int) -> dict[str, Any]:
    if lib not in LIBS:
        raise ValueError(f"unknown lib {lib!r}; expected one of {LIBS}")
    if fuzz_terms.is_fuzz_design(design):
        if lib != "gamfit":
            raise ValueError(f"fuzz design {design!r} is gamfit-only, got lib {lib!r}")
        return run_fuzz(family, n, design, seed)
    X, y, _ = make_data(n, design, family, seed)
    Xt, yt, mut = make_data(n, design, family, seed + TEST_SEED_OFFSET)
    out: dict[str, Any] = {"base_rss_mb": rss_peak_mb()}
    errors: dict[str, str] = {}

    def phase(name: str, fn: Callable[[], Any], timed: bool = True) -> Any:
        t = Timer()
        try:
            value = fn()
        except Exception:
            errors[name] = traceback.format_exc(limit=4)[-2000:]
            return None
        if timed:
            out[f"{name}_s"], out[f"{name}_cpu_s"] = t.stop()
        return value

    def make_adapter() -> Adapter:
        if lib == "gamfit":
            return GamfitAdapter(family, design, X.shape[1])
        return PygamAdapter(family, design, X.shape[1], gridsearch=lib == "pygam_gs")

    adapter: Adapter | None = phase("import", make_adapter)
    if adapter is not None:
        out["lib_version"] = adapter.version
        out["after_import_rss_mb"] = rss_peak_mb()
        phase("fit", lambda: adapter.fit(X, y))
    if adapter is not None and "fit" not in errors:
        phase("fit_warm", lambda: adapter.fit(X, y))
    if adapter is not None and not {"fit", "fit_warm"} & errors.keys():
        out["rss_after_fit_mb"] = rss_peak_mb()
        pred: FloatArray | None = phase("pred", lambda: adapter.predict(Xt))
        iv = phase("interval", lambda: adapter.interval(Xt))
        info = phase("info", adapter.model_info, timed=False)
        if info is not None:
            out.update(info)
        if pred is not None:
            out["rmse_mu"] = float(np.sqrt(np.mean((pred - mut) ** 2)))
            out["rmse_y"] = float(np.sqrt(np.mean((pred - yt) ** 2)))
            out["deviance"] = mean_deviance(family, yt, pred)
        if iv is not None:
            lo, hi, sd = iv
            out["coverage"] = float(np.mean((mut >= lo) & (mut <= hi)))
            out["ci_width"] = float(np.mean(hi - lo))
            if pred is not None:
                out["logscore"] = phase(
                    "logscore", lambda: mean_logscore(family, yt, pred, sd), timed=False
                )
        elif pred is not None and family != "gaussian":
            out["logscore"] = mean_logscore(family, yt, pred, None)
    out["peak_rss_mb"] = rss_peak_mb()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out["cpu_user_s"] = usage.ru_utime
    out["cpu_sys_s"] = usage.ru_stime
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    lib, family, n, design, seed = (
        argv[0],
        argv[1],
        int(float(argv[2])),
        argv[3],
        int(argv[4]),
    )
    out = run(lib, family, n, design, seed)
    # Non-finite floats become null: JSON has no NaN/Inf, and a report that
    # silently parses "NaN" would hide a broken metric.
    nonfinite = [
        k for k, v in out.items() if isinstance(v, float) and not np.isfinite(v)
    ]
    clean = {k: (None if k in nonfinite else v) for k, v in out.items()}
    if nonfinite:
        clean["nonfinite"] = nonfinite
    print("RESULT " + json.dumps(clean, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
