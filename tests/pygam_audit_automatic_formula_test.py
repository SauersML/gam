"""pyGAM audit F6: the automatic formula is built from the data schema in Rust.

``GAMRegressor().fit(X, y)`` with no formula (pyGAM's ``LinearGAM().fit(X, y)``)
used to be impossible: ``formula`` was a required constructor argument and the
formula grammar had no way to say "every remaining column". The automatic
formula is now the ``.`` term, expanded by ONE Rust rule
(``gam_terms::inference::automatic_formula``) that the Python fit, the sklearn
wrapper and ``gam fit data.csv "y ~ ."`` all call, so they cannot disagree.

Each column maps to a penalized term whose null space is itself penalized, so a
column carrying no signal is shrunk to (near) zero effective degrees of freedom.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit.sklearn import GAMClassifier, GAMRegressor

# The fitted formula the schema rule implies for `_mixed_frame`, in column
# order: numeric (>= 3 distinct values) -> s(); pandas category and string ->
# factor(); bool and two-valued numeric -> linear; constant -> dropped.
_EXPECTED = "y ~ s(x) + factor(grp) + factor(city) + flag + treated + s(noise)"


def _mixed_frame(n: int = 240, seed: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    grp = pd.Categorical(rng.choice(["a", "b", "c"], n))
    city = rng.choice(["north", "south", "east", "west"], n).astype(object)
    flag = rng.random(n) < 0.5
    treated = rng.integers(0, 2, n).astype(float)
    noise = rng.uniform(-1.0, 1.0, n)
    level = {"a": 0.0, "b": 0.5, "c": -0.5}
    y = (
        np.sin(2.0 * np.pi * x)
        + np.array([level[g] for g in grp])
        + 0.4 * treated
        + rng.normal(0.0, 0.3, n)
    )
    X = pd.DataFrame(
        {
            "x": x,
            "grp": grp,
            "city": city,
            "flag": flag,
            "constant": np.full(n, 2.5),
            "treated": treated,
            "noise": noise,
        }
    )
    return X, y


def _gam_binary() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (
        os.environ.get("GAM_BIN"),
        repo_root / "target" / "release" / "gam",
        repo_root / "target" / "debug" / "gam",
        shutil.which("gam"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise AssertionError(
        "no `gam` CLI binary found (GAM_BIN, target/release/gam, target/debug/gam, PATH)"
    )


def test_regressor_without_formula_builds_the_schema_formula_in_rust() -> None:
    X, y = _mixed_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg = GAMRegressor().fit(X, y)
    assert reg.formula_ == _EXPECTED
    assert reg.model_.formula == _EXPECTED
    assert list(reg.feature_names_in_) == list(X.columns)
    # The constant column is dropped, and the drop is announced, not silent.
    messages = [str(w.message) for w in caught]
    assert any("'constant'" in m and "single value" in m for m in messages), messages
    assert any(_EXPECTED in m for m in messages), messages
    assert np.all(np.isfinite(reg.predict(X)))


def test_dot_formula_is_identical_through_python_and_the_cli(tmp_path: Path) -> None:
    X, y = _mixed_frame()
    frame = X.assign(y=y)
    python_model = gamfit.fit(frame, "y ~ .")

    # The CSV carries what the Python boundary hands Rust: booleans as 0/1
    # (`_tables.stringify_cell`), categories and strings as their labels.
    data_path = tmp_path / "mixed.csv"
    csv_frame = frame.assign(flag=frame["flag"].astype(int), grp=frame["grp"].astype(str))
    csv_frame.to_csv(data_path, index=False, quoting=csv.QUOTE_MINIMAL)
    model_path = tmp_path / "mixed.json"
    completed = subprocess.run(
        [_gam_binary(), "fit", str(data_path), "y ~ .", "--out", str(model_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert _EXPECTED in completed.stderr, completed.stderr
    cli_formula = json.loads(model_path.read_text(encoding="utf-8"))["payload"]["formula"]

    assert python_model.formula == _EXPECTED
    assert cli_formula == python_model.formula


def test_explicit_terms_combine_with_the_dot() -> None:
    X, y = _mixed_frame()
    frame = X.assign(y=y)
    model = gamfit.fit(frame, "y ~ s(x, k=12) + .")
    assert model.formula == (
        "y ~ s(x, k=12) + factor(grp) + factor(city) + flag + treated + s(noise)"
    )


def test_pure_noise_column_is_shrunk_to_near_zero_edf() -> None:
    rng = np.random.default_rng(11)
    n = 800
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "z": rng.uniform(0.0, 1.0, n)})
    y = np.sin(5.0 * X["x"].to_numpy()) + rng.normal(0.0, 0.3, n)
    reg = GAMRegressor().fit(X, y)
    assert reg.formula_ == "y ~ s(x) + s(z)"
    terms = {t["name"]: float(t["edf"]) for t in reg.model_.summary().smooth_terms}
    z_edf = next(edf for name, edf in terms.items() if "z" in name)
    x_edf = next(edf for name, edf in terms.items() if "x" in name)
    assert z_edf < 0.5, terms
    assert x_edf > 2.0, terms


def test_numpy_features_get_the_automatic_formula() -> None:
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.uniform(0.0, 1.0, 200), rng.integers(0, 2, 200)])
    y = np.cos(3.0 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0.0, 0.2, 200)
    reg = GAMRegressor().fit(X, y)
    assert reg.formula_ == "y ~ s(x0) + x1"


def test_classifier_without_formula_uses_the_automatic_formula() -> None:
    rng = np.random.default_rng(12)
    n = 400
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "z": rng.uniform(0.0, 1.0, n)})
    labels = np.where(X["x"].to_numpy() + rng.normal(0.0, 0.2, n) > 0.5, "yes", "no")
    clf = GAMClassifier(family="binomial").fit(X, labels)
    assert clf.formula_ == "y ~ s(x) + s(z)"
    assert list(clf.classes_) == ["no", "yes"]
    assert clf.score(X, labels) > 0.8


def test_automatic_formula_needs_a_response() -> None:
    X, _ = _mixed_frame(n=40)
    with pytest.raises(ValueError, match="pass y"):
        GAMRegressor().fit(X)
