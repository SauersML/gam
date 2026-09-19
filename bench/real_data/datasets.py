"""Real-data registry for the gamfit vs pyGAM leaderboard.

Every dataset is fetched at run time from a public URL into a cache directory
(``$GAMFIT_REAL_DATA_CACHE``, default ``~/.cache/gamfit-bench/real_data``) and
verified against the SHA-256 recorded here before it is used. Nothing is
vendored: a checksum mismatch is an error, never a silent re-download.

Each dataset carries one model specification, a tuple of terms that renders to
both a gamfit formula and the equivalent pyGAM ``TermList``, so the two
libraries fit the same model. ``prepare`` turns the raw download into a
model-ready ``.npz`` (named numeric columns, response, optional prior weights)
that the per-rep worker loads without importing anything else.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

CACHE = Path(
    os.environ.get(
        "GAMFIT_REAL_DATA_CACHE", Path.home() / ".cache" / "gamfit-bench" / "real_data"
    )
)
PYGAM_TAG_URL = "https://raw.githubusercontent.com/dswah/pyGAM/v0.12.0/pygam/datasets/{}.csv"
RDATASETS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/{}.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/{}.zip"


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Term:
    """One model term, rendered identically for gamfit and pyGAM."""

    kind: str  # "s" | "f" | "te"
    cols: tuple[str, ...]
    k: int | None = None

    def gamfit(self) -> str:
        if self.kind == "f":
            return f"factor({self.cols[0]})"
        k = "" if self.k is None else f", k={self.k}"
        return f"{self.kind}({', '.join(self.cols)}{k})"

    def pygam(self, index: dict[str, int]) -> Any:
        import pygam

        idx = [index[c] for c in self.cols]
        kw = {} if self.k is None else {"n_splines": self.k}
        return {"s": pygam.s, "f": pygam.f, "te": pygam.te}[self.kind](*idx, **kw)


def S(col: str, k: int | None = None) -> Term:
    return Term("s", (col,), k)


def F(col: str) -> Term:
    return Term("f", (col,))


def TE(a: str, b: str) -> Term:
    return Term("te", (a, b))


@dataclass(frozen=True)
class Dataset:
    name: str
    group: str
    family: str
    terms: tuple[Term, ...]
    sources: tuple[Source, ...]
    build: Callable[[dict[str, Path]], dict[str, np.ndarray]]
    note: str = ""

    @property
    def formula(self) -> str:
        return "y ~ " + " + ".join(t.gamfit() for t in self.terms)

    @property
    def columns(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for t in self.terms:
            for c in t.cols:
                seen.setdefault(c, None)
        return tuple(seen)

    @property
    def factor_columns(self) -> tuple[str, ...]:
        return tuple(t.cols[0] for t in self.terms if t.kind == "f")


# ---------------------------------------------------------------------------
# fetch + checksum
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(src: Source) -> Path:
    """Download ``src`` into the cache once and verify its checksum."""
    raw = CACHE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / src.filename
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(src.url, tmp)
        tmp.rename(dest)
    got = _sha256(dest)
    if got != src.sha256:
        raise RuntimeError(
            f"checksum mismatch for {src.filename} from {src.url}: "
            f"expected {src.sha256}, got {got}. Delete {dest} to re-fetch, and "
            "update the registry only after checking the upstream change."
        )
    return dest


def prepared_path(ds: Dataset) -> Path:
    return CACHE / "prepared" / f"{ds.name}.npz"


def prepare(ds: Dataset) -> Path:
    """Fetch, verify and build ``ds`` into a model-ready ``.npz``."""
    paths = {s.filename: fetch(s) for s in ds.sources}
    arrays = ds.build(paths)
    missing = {"y", *ds.columns} - arrays.keys()
    if missing:
        raise RuntimeError(f"{ds.name}: build produced no column(s) {sorted(missing)}")
    n = len(arrays["y"])
    for k, v in arrays.items():
        if len(v) != n or not np.all(np.isfinite(v)):
            raise RuntimeError(f"{ds.name}: column {k} is ragged or non-finite")
    out = prepared_path(ds)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v, dtype=float) for k, v in arrays.items()})
    return out


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(prepared_path(REGISTRY[name])) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _pd() -> Any:
    import pandas as pd

    return pd


def _codes(values: Any) -> np.ndarray:
    """Integer level codes (sorted level order) of a categorical column."""
    return np.unique(np.asarray(values).astype(str), return_inverse=True)[1].astype(float)


def _zip_member(path: Path, member: str) -> io.BytesIO:
    with zipfile.ZipFile(path) as z:
        return io.BytesIO(z.read(member))


def _pygam_csv(name: str, sha: str) -> Source:
    return Source(f"pygam_{name}.csv", PYGAM_TAG_URL.format(name), sha)


def _hist_counts(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    # pyGAM's own coal / faithful loaders: counts per equal-width bin at the
    # bin midpoints.
    y, edges = np.histogram(values, bins=bins)
    return edges[:-1] + np.diff(edges) / 2, y.astype(float)


def _b_mcycle(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_mcycle.csv"], index_col=0)
    return {"times": d["times"].to_numpy(float), "y": d["accel"].to_numpy(float)}


def _b_coal(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_coal.csv"], index_col=0)
    x, y = _hist_counts(d.to_numpy(float).ravel(), 150)
    return {"year": x, "y": y}


def _b_faithful(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_faithful.csv"], index_col=0)
    x, y = _hist_counts(d["eruptions"].to_numpy(float), 200)
    return {"eruptions": x, "y": y}


def _b_wage(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_wage.csv"], index_col=0)
    return {
        "year": d["year"].to_numpy(float),
        "age": d["age"].to_numpy(float),
        "education": _codes(d["education"]),
        "y": d["wage"].to_numpy(float),
    }


def _b_trees(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_trees.csv"], index_col=0)
    return {
        "girth": d["Girth"].to_numpy(float),
        "height": d["Height"].to_numpy(float),
        "y": d["Volume"].to_numpy(float),
    }


def _b_default(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_default.csv"], index_col=0)
    return {
        "student": (d["student"] == "Yes").to_numpy(float),
        "balance": d["balance"].to_numpy(float),
        "income": d["income"].to_numpy(float),
        "y": (d["default"] == "Yes").to_numpy(float),
    }


def _b_cake(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_cake.csv"], index_col=0)
    return {
        "recipe": _codes(d["recipe"]),
        "replicate": d["replicate"].to_numpy(float),
        "temperature": d["temperature"].to_numpy(float),
        "y": d["angle"].to_numpy(float),
    }


def _b_hepatitis(p: dict[str, Path]) -> dict[str, np.ndarray]:
    # Seroprevalence by age: a binomial response (positives out of ``total``
    # tested), fitted as the proportion with prior weight ``total``. Ages with
    # nobody tested carry no information and are dropped, as pyGAM's loader does.
    d = _pd().read_csv(p["pygam_hepatitis_A_bulgaria.csv"]).astype(float)
    d = d[d["total"] > 0]
    return {
        "age": d["age"].to_numpy(float),
        "y": (d["hepatitis_A_positive"] / d["total"]).to_numpy(float),
        "weights": d["total"].to_numpy(float),
    }


def _b_head(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_head_circumference.csv"], index_col=0)
    return {"age": d["age"].to_numpy(float), "y": d["head"].to_numpy(float)}


def _b_chicago(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["pygam_chicago.csv"], index_col=0)
    d = d[["time", "tmpd", "pm10median", "o3median", "death"]].dropna()
    return {
        "time": d["time"].to_numpy(float),
        "tmpd": d["tmpd"].to_numpy(float),
        "pm10": d["pm10median"].to_numpy(float),
        "o3": d["o3median"].to_numpy(float),
        "y": d["death"].to_numpy(float),
    }


def _b_airquality(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["r_airquality.csv"]).dropna(subset=["Ozone", "Solar.R"])
    return {
        "solar": d["Solar.R"].to_numpy(float),
        "wind": d["Wind"].to_numpy(float),
        "temp": d["Temp"].to_numpy(float),
        "y": d["Ozone"].to_numpy(float),
    }


def _b_co2(p: dict[str, Path]) -> dict[str, np.ndarray]:
    # Mauna Loa monthly CO2 (R ``datasets::co2``): long-run trend + seasonal cycle.
    d = _pd().read_csv(p["r_co2.csv"])
    t = d["time"].to_numpy(float)
    return {
        "time": t,
        "month": np.round((t - np.floor(t)) * 12.0) + 1.0,
        "y": d["value"].to_numpy(float),
    }


def _b_boston(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(p["r_Boston.csv"])
    cols = ["lstat", "rm", "crim", "dis", "nox", "ptratio", "age", "tax"]
    out = {c: d[c].to_numpy(float) for c in cols}
    out["chas"] = d["chas"].to_numpy(float)
    out["y"] = np.log(d["medv"].to_numpy(float))
    return out


def _b_gamsim1(_: dict[str, Path]) -> dict[str, np.ndarray]:
    # mgcv::gamSim(1): f0 = 2 sin(pi x0), f1 = exp(2 x1), f2 = the bimodal
    # beta-like bump, x3 pure noise; n = 400, scale 2 (mgcv's defaults).
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, (400, 4))
    f = (
        2 * np.sin(np.pi * X[:, 0])
        + np.exp(2 * X[:, 1])
        + 0.2 * X[:, 2] ** 11 * (10 * (1 - X[:, 2])) ** 6
        + 10 * (10 * X[:, 2]) ** 3 * (1 - X[:, 2]) ** 10
    )
    out = {f"x{j}": X[:, j] for j in range(4)}
    out["y"] = f + rng.normal(0.0, 2.0, 400)
    return out


def _b_bike(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_csv(_zip_member(p["uci_bike.zip"], "hour.csv"))
    out = {c: d[c].to_numpy(float) for c in ("hr", "temp", "hum", "windspeed")}
    for c in ("season", "yr", "weathersit", "workingday"):
        out[c] = d[c].to_numpy(float)
    out["y"] = d["cnt"].to_numpy(float)
    return out


def _b_california(p: dict[str, Path]) -> dict[str, np.ndarray]:
    # The StatLib file behind sklearn's fetch_california_housing, with
    # sklearn's per-household derived features.
    with tarfile.open(p["cal_housing.tgz"]) as tf:
        member = tf.extractfile("CaliforniaHousing/cal_housing.data")
        assert member is not None
        a = np.loadtxt(member, delimiter=",")
    lon, lat, age, rooms, beds, pop, households, inc, value = a.T
    return {
        "medinc": inc,
        "houseage": age,
        "averooms": rooms / households,
        "avebedrms": beds / households,
        "population": pop,
        "aveoccup": pop / households,
        "latitude": lat,
        "longitude": lon,
        "y": np.log(value),
    }


def _b_abalone(p: dict[str, Path]) -> dict[str, np.ndarray]:
    names = ["sex", "length", "diameter", "height", "whole", "shucked", "viscera", "shell", "rings"]
    d = _pd().read_csv(_zip_member(p["uci_abalone.zip"], "abalone.data"), header=None, names=names)
    out = {c: d[c].to_numpy(float) for c in names[1:-1]}
    out["sex"] = _codes(d["sex"])
    out["y"] = d["rings"].to_numpy(float)
    return out


def _b_concrete(p: dict[str, Path]) -> dict[str, np.ndarray]:
    d = _pd().read_excel(_zip_member(p["uci_concrete.zip"], "Concrete_Data.xls"))
    names = ["cement", "slag", "ash", "water", "superplastic", "coarse", "fine", "age"]
    a = d.to_numpy(float)
    out = {c: a[:, j] for j, c in enumerate(names)}
    out["y"] = a[:, 8]
    return out


WINE_FEATURES = (
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_so2", "total_so2", "density", "ph", "sulphates", "alcohol",
)  # fmt: skip


def _b_wine(colour: str) -> Callable[[dict[str, Path]], dict[str, np.ndarray]]:
    def build(p: dict[str, Path]) -> dict[str, np.ndarray]:
        member = _zip_member(p["uci_wine.zip"], f"winequality-{colour}.csv")
        a = _pd().read_csv(member, sep=";").to_numpy(float)
        out = {c: a[:, j] for j, c in enumerate(WINE_FEATURES)}
        out["y"] = a[:, 11]
        return out

    return build


def _b_adult(p: dict[str, Path]) -> dict[str, np.ndarray]:
    names = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income",
    ]  # fmt: skip
    d = _pd().read_csv(
        _zip_member(p["uci_adult.zip"], "adult.data"),
        header=None,
        names=names,
        skipinitialspace=True,
    )
    out = {c: d[c].to_numpy(float) for c in ("age", "education_num", "hours_per_week")}
    # Capital gains / losses are zero for most people and span five decades
    # otherwise; an analyst smooths them on the log1p scale.
    out["log_capital_gain"] = np.log1p(d["capital_gain"].to_numpy(float))
    out["log_capital_loss"] = np.log1p(d["capital_loss"].to_numpy(float))
    for c in ("sex", "marital_status", "occupation", "relationship", "race", "workclass"):
        out[c] = _codes(d[c])  # "?" (unknown) is its own level
    out["y"] = (d["income"] == ">50K").to_numpy(float)
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_R = RDATASETS_URL
_CHICAGO = _pygam_csv("chicago", "fbfa13beecfef0fc7b0a75a0f5cdecd5a0ef22a7751811a7a19512bbb6585773")
_WINE = Source("uci_wine.zip", UCI_URL.format("186/wine+quality"),
               "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4")  # fmt: skip

DATASETS: tuple[Dataset, ...] = (
    # ---- pyGAM's own datasets (pygam.datasets, CSVs from the v0.12.0 tag) ----
    Dataset("mcycle", "pygam", "gaussian", (S("times"),),
            (_pygam_csv("mcycle", "1b3d02a8d8a3e86a265e2d22a906d77c9f0cac48bd57453da4033b1aa90e8b2e"),),
            _b_mcycle),
    Dataset("coal", "pygam", "poisson", (S("year"),),
            (_pygam_csv("coal", "dbb0db57e53f42949d9fb290558967c384fc8bccdb533b85ad9a893065e14261"),),
            _b_coal, "disaster counts in 150 yearly bins (pyGAM's loader)"),
    Dataset("faithful", "pygam", "poisson", (S("eruptions"),),
            (_pygam_csv("faithful", "6675d052dd7495645ac2146290c8ca69e6c1d0795723288d578d2cb2c6c9cda3"),),
            _b_faithful, "eruption-length counts in 200 bins (pyGAM's loader)"),
    Dataset("wage", "pygam", "gaussian", (S("year"), S("age"), F("education")),
            (_pygam_csv("wage", "f1cff965ab86ca950aac5f269554fdba5b772cf458314ef3d037c4ab86dbc2ea"),),
            _b_wage, "year has 7 distinct values"),
    Dataset("trees", "pygam", "gamma", (S("girth"), S("height")),
            (_pygam_csv("trees", "c347a9dddb57ba054d5e65b0a0da9310f7a22d1de9a8b3bf24402f4e54411d54"),),
            _b_trees, "n = 31"),
    Dataset("default", "pygam", "binomial", (F("student"), S("balance"), S("income")),
            (_pygam_csv("default", "2e1f857c3d9924c79f36f46db646bf96b5f34bbb50c1549f71036047d5fd0c6e"),),
            _b_default, "3.3% positives"),
    Dataset("cake", "pygam", "gaussian", (F("recipe"), F("replicate"), S("temperature")),
            (_pygam_csv("cake", "2dd68ce7a10ca65dace02b85f17faf4269fed7430e03c39c63212261543fd642"),),
            _b_cake, "temperature has 6 distinct values"),
    Dataset("hepatitis", "pygam", "binomial", (S("age"),),
            (_pygam_csv("hepatitis_A_bulgaria",
                        "7941c8e138f2db3aae31e597f95c461af9932ff5b4b4528f47a0aca33eb43908"),),
            _b_hepatitis, "positives / tested, prior weight = tested"),
    Dataset("head_circumference", "pygam", "gaussian", (S("age"),),
            (_pygam_csv("head_circumference",
                        "5fdd0327eede22270559cd57b3829e7f56c1ae00bdc7f6945ad7eb06ec9b2e31"),),
            _b_head),
    Dataset("chicago", "pygam", "poisson", (S("time"), S("tmpd"), S("pm10"), S("o3")),
            (_CHICAGO,), _b_chicago),
    Dataset("chicago_docs", "pygam", "poisson", (S("time", k=200), TE("o3", "tmpd"), S("pm10")),
            (_CHICAGO,), _b_chicago, "pyGAM's documented chicago model"),
    # ---- classic R / mgcv examples (Rdatasets CSVs) ----
    Dataset("airquality", "R classic", "gamma", (S("solar"), S("wind"), S("temp")),
            (Source("r_airquality.csv", _R.format("datasets/airquality"),
                    "65d2c4afd976c169af9bb0bd97e9e78e1e8a185f1b52e2e3153e30f90c7fb5f8"),),
            _b_airquality, "complete Ozone / Solar.R rows"),
    Dataset("co2", "R classic", "gaussian", (S("time"), S("month")),
            (Source("r_co2.csv", _R.format("datasets/CO2"),
                    "16f9baa6f3986f31d5b4585d14f5479471a719906638626570ea96faab038807"),),
            _b_co2, "Mauna Loa monthly CO2 (datasets::co2)"),
    Dataset("boston", "R classic", "gaussian",
            (S("lstat"), S("rm"), S("crim"), S("dis"), S("nox"), S("ptratio"), S("age"),
             S("tax"), F("chas")),
            (Source("r_Boston.csv", _R.format("MASS/Boston"),
                    "654ae93c04416defb2b3752951a7f4357d5951c84c02371b80b48a4492337f86"),),
            _b_boston, "log(medv)"),
    Dataset("gamsim1", "R classic", "gaussian", (S("x0"), S("x1"), S("x2"), S("x3")),
            (), _b_gamsim1, "mgcv::gamSim(1) re-created, n = 400, fixed seed"),
    # ---- UCI ----
    Dataset("bike_hour", "UCI", "poisson",
            (S("hr"), S("temp"), S("hum"), S("windspeed"), F("season"), F("yr"),
             F("weathersit"), F("workingday")),
            (Source("uci_bike.zip", UCI_URL.format("275/bike+sharing+dataset"),
                    "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"),),
            _b_bike, "hourly rental counts"),
    Dataset("california", "UCI", "gaussian",
            (S("medinc"), S("houseage"), S("averooms"), S("avebedrms"), S("population"),
             S("aveoccup"), TE("latitude", "longitude")),
            (Source("cal_housing.tgz", "https://ndownloader.figshare.com/files/5976036",
                    "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"),),
            _b_california, "log median house value"),
    Dataset("abalone", "UCI", "poisson",
            (F("sex"), S("length"), S("diameter"), S("height"), S("whole"), S("shucked"),
             S("viscera"), S("shell")),
            (Source("uci_abalone.zip", UCI_URL.format("1/abalone"),
                    "755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4"),),
            _b_abalone, "rings"),
    Dataset("concrete", "UCI", "gaussian",
            (S("cement"), S("slag"), S("ash"), S("water"), S("superplastic"), S("coarse"),
             S("fine"), S("age")),
            (Source("uci_concrete.zip", UCI_URL.format("165/concrete+compressive+strength"),
                    "dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7"),),
            _b_concrete, "compressive strength"),
    Dataset("wine_red", "UCI", "gaussian", tuple(S(c) for c in WINE_FEATURES), (_WINE,),
            _b_wine("red")),
    Dataset("wine_white", "UCI", "gaussian", tuple(S(c) for c in WINE_FEATURES), (_WINE,),
            _b_wine("white")),
    Dataset("adult", "UCI", "binomial",
            (S("age"), S("education_num"), S("hours_per_week"), S("log_capital_gain"),
             S("log_capital_loss"), F("sex"), F("marital_status"), F("occupation"),
             F("relationship"), F("race"), F("workclass")),
            (Source("uci_adult.zip", UCI_URL.format("2/adult"),
                    "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"),),
            _b_adult, "income > 50K (adult.data)"),
)  # fmt: skip

REGISTRY: dict[str, Dataset] = {d.name: d for d in DATASETS}
