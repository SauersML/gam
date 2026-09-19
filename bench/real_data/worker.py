"""One leaderboard rep: one library, one dataset, one CV fold, fresh process.

Usage: worker.py LIB DATASET FOLD

  LIB      gamfit | gamfit_auto | pygam | pygam_gs
  DATASET  a name in ``datasets.REGISTRY`` (already ``prepare``d by run.py)
  FOLD     0 .. FOLDS-1: the held-out fold of the fixed-seed split

``gamfit`` fits the dataset's formula; ``gamfit_auto`` fits ``y ~ .`` on the
same columns, with the factor columns passed as strings so the automatic rule
sees them as categorical; ``pygam`` fits the same terms at pyGAM's default
smoothing and ``pygam_gs`` runs ``gridsearch`` (the fair comparator: pyGAM's
own way of choosing smoothing).

Prints exactly one ``RESULT {json}`` line. Held-out metrics are on the fold
the fit never saw: mean unit deviance (prior-weighted where the dataset has
weights), RMSE, and for Gaussian data the coverage and width of the
library's 95% prediction interval for a new observation, plus the log score
that interval implies. A phase that raises is recorded under ``errors`` and
makes the status ``error``.
"""

from __future__ import annotations

import importlib
import json
import resource
import sys
import time
import traceback
import warnings
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

if __package__:
    from . import datasets
else:  # run as a script by run.py, with this directory on sys.path
    import datasets  # type: ignore[no-redef]

LIBS = ("gamfit", "gamfit_auto", "pygam", "pygam_gs")
FOLDS = 5
SPLIT_SEED = 20240501
INTERVAL_LEVEL = 0.95

FloatArray = NDArray[np.float64]

PYGAM_FAMILY = {
    "gaussian": ("normal", "identity"),
    "binomial": ("binomial", "logit"),
    "poisson": ("poisson", "log"),
    "gamma": ("gamma", "log"),
}


def fold_ids(y: FloatArray, family: str, folds: int = FOLDS) -> NDArray[np.int64]:
    """Fixed-seed fold labels; binomial splits are stratified on the response.

    Rows are shuffled with ``SPLIT_SEED`` and dealt round-robin into folds.
    For binomial data the shuffled rows are first stably grouped by response
    class, so every fold (and every training set) keeps the class balance: a
    rare-event dataset such as ``default`` cannot draw a fold without events.
    """
    order = np.random.default_rng(SPLIT_SEED).permutation(len(y))
    if family == "binomial":
        order = order[np.argsort(np.round(y[order]), kind="stable")]
    ids = np.empty(len(y), dtype=np.int64)
    ids[order] = np.arange(len(y)) % folds
    return ids


def mean_deviance(family: str, y: FloatArray, mu: FloatArray, w: FloatArray) -> float:
    """Prior-weighted mean unit deviance of held-out ``y`` at predicted mean ``mu``."""
    if family == "gaussian":
        unit = (y - mu) ** 2
    elif family == "binomial":
        unit = 2 * (special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu)))
    elif family == "poisson":
        unit = 2 * (special.xlogy(y, y / mu) - (y - mu))
    elif family == "gamma":
        unit = 2 * (-np.log(y / mu) + (y - mu) / mu)
    else:
        raise ValueError(f"no deviance for family {family!r}")
    return float(np.sum(w * unit) / np.sum(w))


def rss_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


class Adapter:
    version: str

    def fit(self, rows: NDArray[np.bool_]) -> None:
        raise NotImplementedError

    def predict(self, rows: NDArray[np.bool_]) -> FloatArray:
        raise NotImplementedError

    def observation_interval(self, rows: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        raise NotImplementedError


class GamfitAdapter(Adapter):
    def __init__(self, ds: datasets.Dataset, data: dict[str, FloatArray], auto: bool) -> None:
        self.gamfit: Any = importlib.import_module("gamfit")
        self.version = str(self.gamfit.__version__)
        self.ds = ds
        self.data = data
        self.auto = auto
        self.formula = "y ~ ." if auto else ds.formula
        self.weights = "weights" if "weights" in data else None
        self.model: Any = None

    def _table(self, rows: NDArray[np.bool_], with_y: bool) -> Any:
        cols = list(self.ds.columns)
        if with_y:
            cols.append("y")
            if self.weights:
                cols.append(self.weights)
        table: dict[str, Any] = {c: self.data[c][rows] for c in cols}
        if self.auto:
            import pandas as pd

            # Level codes as strings: the automatic rule reads a string column
            # as categorical, which is what the explicit formula's factor() says.
            for c in self.ds.factor_columns:
                table[c] = np.array([f"L{int(v)}" for v in table[c]], dtype=object)
            return pd.DataFrame(table)
        return table

    def fit(self, rows: NDArray[np.bool_]) -> None:
        kw: dict[str, Any] = {"family": self.ds.family}
        if self.weights:
            kw["weights"] = self.weights
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the auto rule reports its formula as a warning
            self.model = self.gamfit.fit(self._table(rows, True), self.formula, **kw)

    def predict(self, rows: NDArray[np.bool_]) -> FloatArray:
        return np.asarray(self.model.predict(self._table(rows, False)), dtype=float).reshape(-1)

    def observation_interval(self, rows: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
        res = self.model.predict(
            self._table(rows, False),
            interval=INTERVAL_LEVEL,
            observation_interval=True,
            return_type="dict",
        )
        return (
            np.asarray(res["observation_lower"], dtype=float),
            np.asarray(res["observation_upper"], dtype=float),
        )

    def model_info(self) -> dict[str, Any]:
        summ = self.model.summary()
        conv = getattr(summ, "convergence", None)
        info: dict[str, Any] = {
            "edf": None if summ.edf_total is None else float(summ.edf_total),
            "ncoef": None if summ.coefficients is None else len(summ.coefficients),
            "iterations": summ.iterations,
            "convergence": json.loads(json.dumps(conv, default=str)),
        }
        if self.auto:
            info["auto_formula"] = str(getattr(self.model, "formula", ""))
        return info


class PygamAdapter(Adapter):
    def __init__(
        self, ds: datasets.Dataset, data: dict[str, FloatArray], gridsearch: bool
    ) -> None:
        pygam: Any = importlib.import_module("pygam")  # untyped: used as Any
        self.pygam = pygam
        self.version = str(pygam.__version__)
        self.ds = ds
        self.data = data
        self.cols = list(ds.columns)
        index = {c: j for j, c in enumerate(self.cols)}
        terms: Any = ds.terms[0].pygam(index)
        for t in ds.terms[1:]:
            terms = terms + t.pygam(index)
        self.terms = terms
        self.gridsearch = gridsearch
        self.model: Any = None
        self.warnings: list[str] = []

    def _X(self, rows: NDArray[np.bool_]) -> FloatArray:
        return np.column_stack([self.data[c][rows] for c in self.cols])

    def fit(self, rows: NDArray[np.bool_]) -> None:
        if self.ds.family == "gaussian":
            # LinearGAM is GAM(normal, identity) plus prediction_intervals.
            g = self.pygam.LinearGAM(self.terms)
        else:
            dist, link = PYGAM_FAMILY[self.ds.family]
            g = self.pygam.GAM(self.terms, distribution=dist, link=link)
        X, y = self._X(rows), self.data["y"][rows]
        w = self.data["weights"][rows] if "weights" in self.data else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self.gridsearch:
                g.gridsearch(X, y, weights=w, progress=False)
            else:
                g.fit(X, y, weights=w)
        self.warnings = sorted({str(c.message)[:200] for c in caught})
        self.model = g

    def predict(self, rows: NDArray[np.bool_]) -> FloatArray:
        return np.asarray(self.model.predict(self._X(rows)), dtype=float).reshape(-1)

    def observation_interval(self, rows: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
        pi = np.asarray(
            self.model.prediction_intervals(self._X(rows), width=INTERVAL_LEVEL), dtype=float
        )
        return pi[:, 0], pi[:, 1]

    def model_info(self) -> dict[str, Any]:
        return {
            "edf": float(self.model.statistics_["edof"]),
            "ncoef": int(len(self.model.coef_)),
            "lam": [float(v) for v in _flatten(self.model.lam)],
            "pygam_warnings": self.warnings,
            "pygam_not_converged": any("converge" in m for m in self.warnings),
        }


def run(lib: str, name: str, fold: int) -> dict[str, Any]:
    if lib not in LIBS:
        raise ValueError(f"unknown lib {lib!r}; expected one of {LIBS}")
    ds = datasets.REGISTRY[name]
    data = datasets.load(name)
    ids = fold_ids(data["y"], ds.family)
    train, test = ids != fold, ids == fold
    y_test = data["y"][test]
    w_test = data["weights"][test] if "weights" in data else np.ones(int(test.sum()))
    out: dict[str, Any] = {
        "n": int(len(data["y"])),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "base_rss_mb": rss_peak_mb(),
    }
    errors: dict[str, str] = {}

    def phase(key: str, fn: Callable[[], Any], timed: bool = True) -> Any:
        t_wall, t_cpu = time.perf_counter(), time.process_time()
        try:
            value = fn()
        except Exception:
            errors[key] = traceback.format_exc(limit=6)[-3000:]
            return None
        if timed:
            out[f"{key}_s"] = time.perf_counter() - t_wall
            out[f"{key}_cpu_s"] = time.process_time() - t_cpu
        return value

    def make_adapter() -> Adapter:
        if lib.startswith("gamfit"):
            return GamfitAdapter(ds, data, auto=lib == "gamfit_auto")
        return PygamAdapter(ds, data, gridsearch=lib == "pygam_gs")

    adapter: Adapter | None = phase("import", make_adapter)
    if adapter is not None:
        out["lib_version"] = adapter.version
        out["after_import_rss_mb"] = rss_peak_mb()
        phase("fit", lambda: adapter.fit(train))
    if adapter is not None and "fit" not in errors:
        out["rss_after_fit_mb"] = rss_peak_mb()
        info = phase("info", adapter.model_info, timed=False)
        if info is not None:
            out.update(info)
        pred: FloatArray | None = phase("pred", lambda: adapter.predict(test))
        if pred is not None:
            out["deviance"] = mean_deviance(ds.family, y_test, pred, w_test)
            out["rmse"] = float(np.sqrt(np.sum(w_test * (pred - y_test) ** 2) / np.sum(w_test)))
        if ds.family == "gaussian":
            iv = phase("interval", lambda: adapter.observation_interval(test))
            if iv is not None:
                lo, hi = iv
                out["coverage"] = float(np.mean((y_test >= lo) & (y_test <= hi)))
                out["pi_width"] = float(np.mean(hi - lo))
                if pred is not None:
                    sd = (hi - lo) / (2 * stats.norm.ppf(0.5 + INTERVAL_LEVEL / 2))
                    out["logscore"] = float(-np.mean(stats.norm.logpdf(y_test, pred, sd)))
    out["peak_rss_mb"] = rss_peak_mb()
    out["status"] = "error" if errors else "ok"
    if errors:
        out["errors"] = errors
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    out = run(argv[0], argv[1], int(argv[2]))
    # Non-finite floats become null (JSON has no NaN/Inf) and are listed, so a
    # broken metric is visible rather than silently parsed.
    nonfinite = [k for k, v in out.items() if isinstance(v, float) and not np.isfinite(v)]
    clean = {k: (None if k in nonfinite else v) for k, v in out.items()}
    if nonfinite:
        clean["nonfinite"] = nonfinite
    print("RESULT " + json.dumps(clean, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
