"""One benchmark rep: one library, one (family, n, design, seed) cell, fresh process.

Usage: worker.py LIB FAMILY N DESIGN SEED

  LIB     gamfit | pygam | pygam_gs
  FAMILY  gaussian | binomial | poisson, or a count family of the
          ``count_sweep`` plan (see ``COUNT_FAMILIES``)
  DESIGN  p1 | p5 | p20 | te

Prints exactly one ``RESULT {json}`` line on stdout. The driver (``run.py``)
launches this script with a pinned thread environment and a scratch working
directory, so the installed gamfit wheel is imported rather than the source
tree's ``./gamfit`` (which has no compiled ``_rust``).

Phases are timed with both ``perf_counter`` (wall) and ``process_time`` (CPU of
this process, all threads). CPU is the primary speed metric: on a loaded or
shared host wall time measures the neighbours as much as the library.

  import    import of the library (numpy/scipy already imported)
  fit       one cold fit (the first fit in the process)
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
import resource
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

LIBS = ("gamfit", "pygam", "pygam_gs")
FAMILIES = ("gaussian", "binomial", "poisson")
# Count families of the ``count_sweep`` plan. ``poisson_*`` put the mean count at
# 0.3, 5 and 500 (``mu = level * exp(0.7 eta)``); ``poisson_exposure`` is the
# 0.3 rate times a log-uniform exposure in [0.5, 50], fitted with the log
# exposure as an offset; ``negbin`` is NB2 at ``theta = 2`` around mean 5, fitted
# with theta estimated; ``tweedie`` is the compound Poisson-gamma at ``p = 1.5``,
# ``phi = 1`` around mean 5, fitted with phi estimated (gamfit does not estimate
# the power: profiling it is a derivative-free search SPEC.md forbids).
COUNT_FAMILIES = (
    "poisson_lo",
    "poisson_mid",
    "poisson_hi",
    "poisson_exposure",
    "negbin",
    "tweedie",
)
ALL_FAMILIES = FAMILIES + COUNT_FAMILIES
POISSON_LEVELS = {"poisson_lo": 0.3, "poisson_mid": 5.0, "poisson_hi": 500.0}
COUNT_SLOPE = 0.7
EXPOSURE_RATE = 0.3
EXPOSURE_RANGE = (0.5, 50.0)
NEGBIN_MEAN, NEGBIN_THETA = 5.0, 2.0
TWEEDIE_MEAN, TWEEDIE_P, TWEEDIE_PHI = 5.0, 1.5, 1.0
# pyGAM has no negative-binomial or Tweedie distribution; those cells run gamfit
# alone and report absolute times and the certification rate.
PYGAM_UNSUPPORTED = frozenset({"negbin", "tweedie"})
DESIGNS = ("p1", "p5", "p20", "te")
INTERVAL_LEVEL = 0.95
TEST_SEED_OFFSET = 1000

FloatArray = NDArray[np.float64]


def supports(lib: str, family: str) -> bool:
    return lib == "gamfit" or family not in PYGAM_UNSUPPORTED


def _tweedie_draw(
    rng: np.random.Generator, mu: FloatArray, p: float, phi: float
) -> FloatArray:
    """Compound Poisson-gamma draw with mean ``mu`` and variance ``phi mu^p``."""
    rate = mu ** (2 - p) / (phi * (2 - p))
    shape = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    counts = rng.poisson(rate)
    y = np.zeros_like(mu)
    hit = counts > 0
    y[hit] = rng.gamma(shape * counts[hit], scale[hit])
    return y


def make_data(
    n: int, design: str, family: str, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    """Draw ``(X, y, mu, offset)``: covariates, response, true response-scale
    mean, and the log-exposure offset (``None`` for families without one).

    Same generators as the pyGAM audit (bench/pygam_audit/speed/worker.py) so
    numbers stay comparable with the audit's speed.md tables.
    """
    rng = np.random.default_rng(seed)
    if design == "te":
        X = rng.uniform(0.0, 1.0, (n, 2))
        eta = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])
    elif design in DESIGNS:
        p = int(design[1:])
        X = rng.uniform(0.0, 1.0, (n, p))
        eta = np.zeros(n)
        for j in range(p):
            eta += np.sin(2 * np.pi * X[:, j] + j) / np.sqrt(p)
    else:
        raise ValueError(f"unknown design {design!r}; expected one of {DESIGNS}")
    if family == "gaussian":
        mu = eta
        y = eta + rng.normal(0.0, 0.5, n)
    elif family == "binomial":
        mu = special.expit(1.5 * eta)
        y = (rng.uniform(size=n) < mu).astype(float)
    elif family == "poisson":
        mu = np.exp(0.5 + 0.7 * eta)
        y = rng.poisson(mu).astype(float)
    elif family in POISSON_LEVELS:
        mu = POISSON_LEVELS[family] * np.exp(COUNT_SLOPE * eta)
        y = rng.poisson(mu).astype(float)
    elif family == "poisson_exposure":
        lo, hi = EXPOSURE_RANGE
        offset = rng.uniform(np.log(lo), np.log(hi), n)
        mu = EXPOSURE_RATE * np.exp(offset + COUNT_SLOPE * eta)
        y = rng.poisson(mu).astype(float)
        return X, y, mu, offset
    elif family == "negbin":
        mu = NEGBIN_MEAN * np.exp(COUNT_SLOPE * eta)
        y = rng.negative_binomial(NEGBIN_THETA, NEGBIN_THETA / (NEGBIN_THETA + mu))
        y = y.astype(float)
    elif family == "tweedie":
        mu = TWEEDIE_MEAN * np.exp(COUNT_SLOPE * eta)
        y = _tweedie_draw(rng, mu, TWEEDIE_P, TWEEDIE_PHI)
    else:
        raise ValueError(f"unknown family {family!r}; expected one of {ALL_FAMILIES}")
    return X, y, mu, None


def mean_deviance(family: str, y: FloatArray, mu: FloatArray) -> float:
    """Mean unit deviance of held-out ``y`` at predicted mean ``mu``.

    Negative binomial and Tweedie are scored at the generating ``theta`` / ``p``,
    so the score ranks predicted means only, the same way for every library.
    """
    if family == "gaussian":
        return float(np.mean((y - mu) ** 2))
    if family == "binomial":
        unit = special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu))
        return float(np.mean(2 * unit))
    if family == "negbin":
        t = NEGBIN_THETA
        unit = special.xlogy(y, y / mu) - special.xlogy(y + t, (y + t) / (mu + t))
        return float(np.mean(2 * unit))
    if family == "tweedie":
        p = TWEEDIE_P
        unit = (
            y ** (2 - p) / ((1 - p) * (2 - p))
            - y * mu ** (1 - p) / (1 - p)
            + mu ** (2 - p) / (2 - p)
        )
        return float(np.mean(2 * unit))
    unit = special.xlogy(y, y / mu) - (y - mu)
    return float(np.mean(2 * unit))


def mean_logscore(
    family: str, y: FloatArray, mu: FloatArray, predictive_sd: FloatArray | None
) -> float | None:
    """Mean negative log predictive density of held-out ``y`` (lower is better).

    Binomial and the Poisson families are scored at the predicted mean, the
    negative binomial at the predicted mean and the generating theta. Gaussian
    needs a predictive scale: the library's own 95% prediction interval gives
    it as ``(upper - lower) / (2 z_0.975)``, which prices both the fitted scale
    and the posterior variance of the mean. The Tweedie density has no closed
    form, so its log score is not reported.
    """
    if family == "binomial":
        return float(-np.mean(special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu)))
    if family == "poisson" or family in POISSON_LEVELS or family == "poisson_exposure":
        return float(-np.mean(stats.poisson.logpmf(y, mu)))
    if family == "negbin":
        t = NEGBIN_THETA
        return float(-np.mean(stats.nbinom.logpmf(y, t, t / (t + mu))))
    if family == "tweedie":
        return None
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

    def fit(self, X: FloatArray, y: FloatArray, offset: FloatArray | None) -> None:
        raise NotImplementedError

    def predict(self, X: FloatArray, offset: FloatArray | None) -> FloatArray:
        raise NotImplementedError

    def interval(
        self, X: FloatArray, offset: FloatArray | None
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        """Return (lower, upper) of the 95% interval for the mean, and the
        Gaussian predictive sd (``None`` for other families). ``offset`` is the
        log exposure of the rows, or ``None``."""
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        raise NotImplementedError


GAMFIT_FAMILY = {
    "negbin": "negative-binomial",
    "tweedie": f"tweedie(p={TWEEDIE_P})",
    **{f: "poisson" for f in (*POISSON_LEVELS, "poisson_exposure")},
}
OFFSET_COLUMN = "log_exposure"


class GamfitAdapter(Adapter):
    def __init__(self, family: str, design: str, p: int) -> None:
        self.gamfit: Any = importlib.import_module("gamfit")
        self.version = str(self.gamfit.__version__)
        self.family = family
        self.gamfit_family = GAMFIT_FAMILY.get(family, family)
        self.names = [f"x{j}" for j in range(p)]
        if design == "te":
            self.formula = "y ~ te(x0, x1)"
        else:
            self.formula = "y ~ " + " + ".join(f"s({nm})" for nm in self.names)
        self.model: Any = None

    def _table(self, X: FloatArray, offset: FloatArray | None) -> dict[str, FloatArray]:
        table = {nm: X[:, j] for j, nm in enumerate(self.names)}
        if offset is not None:
            table[OFFSET_COLUMN] = offset
        return table

    def fit(self, X: FloatArray, y: FloatArray, offset: FloatArray | None) -> None:
        data = self._table(X, offset)
        data["y"] = y
        self.model = self.gamfit.fit(
            data,
            self.formula,
            family=self.gamfit_family,
            offset=None if offset is None else OFFSET_COLUMN,
        )

    def predict(self, X: FloatArray, offset: FloatArray | None) -> FloatArray:
        pred = self.model.predict(self._table(X, offset))
        return np.asarray(pred, dtype=float).reshape(-1)

    def interval(
        self, X: FloatArray, offset: FloatArray | None
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        gaussian = self.family == "gaussian"
        res = self.model.predict(
            self._table(X, offset),
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
        if family in PYGAM_UNSUPPORTED:
            raise ValueError(f"pyGAM has no {family!r} distribution")
        self.dist, self.link = {
            "gaussian": ("normal", "identity"),
            "binomial": ("binomial", "logit"),
        }.get(family, ("poisson", "log"))
        self.gridsearch = gridsearch
        self.model: Any = None

    def fit(self, X: FloatArray, y: FloatArray, offset: FloatArray | None) -> None:
        if self.family == "gaussian":
            # LinearGAM is GAM(normal, identity) plus prediction_intervals,
            # which the Gaussian log score needs.
            g = self.pygam.LinearGAM(self.terms)
        else:
            g = self.pygam.GAM(self.terms, distribution=self.dist, link=self.link)
        extra: dict[str, FloatArray] = {}
        if offset is not None:
            # pyGAM has no offset. PoissonGAM's ``exposure`` is the rate
            # ``y / E`` fitted with weights ``E``, which is the Poisson
            # likelihood with offset ``log E``; it is spelled out here because
            # ``PoissonGAM.gridsearch`` passes the weights on positionally as
            # the exposure and fits ``y / E^2`` (pyGAM 0.12.0).
            exposure = np.exp(offset)
            y = y / exposure
            extra = {"weights": exposure}
        if self.gridsearch:
            g.gridsearch(X, y, progress=False, **extra)
        else:
            g.fit(X, y, **extra)
        self.model = g

    def predict(self, X: FloatArray, offset: FloatArray | None) -> FloatArray:
        pred = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        return pred if offset is None else pred * np.exp(offset)

    def interval(
        self, X: FloatArray, offset: FloatArray | None
    ) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        ci = np.asarray(
            self.model.confidence_intervals(X, width=INTERVAL_LEVEL), dtype=float
        )
        if offset is not None:
            ci = ci * np.exp(offset)[:, None]
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


def run(lib: str, family: str, n: int, design: str, seed: int) -> dict[str, Any]:
    if lib not in LIBS:
        raise ValueError(f"unknown lib {lib!r}; expected one of {LIBS}")
    X, y, _, off = make_data(n, design, family, seed)
    Xt, yt, mut, offt = make_data(n, design, family, seed + TEST_SEED_OFFSET)
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
        phase("fit", lambda: adapter.fit(X, y, off))
    if adapter is not None and "fit" not in errors:
        out["rss_after_fit_mb"] = rss_peak_mb()
        pred: FloatArray | None = phase("pred", lambda: adapter.predict(Xt, offt))
        iv = phase("interval", lambda: adapter.interval(Xt, offt))
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
