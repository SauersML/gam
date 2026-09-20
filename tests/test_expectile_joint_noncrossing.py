"""Joint non-crossing multi-level expectile fits (pyGAM audit F5b).

``family="expectile"`` with ``expectile_tau=[0.1, 0.5, 0.9]`` is ONE
location-scale fit whose level curves ``mu(x) + c_tau * sigma(x)`` are ordered
at every covariate value, by construction. Separately fitted levels are free to
cross; the fixture below is one where they do. The same request through the
``gam`` CLI (``--expectile-tau 0.1,0.5,0.9``) must produce the same curves, and
a CLI-saved model must predict the same curves from Python.
"""

import csv
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

import gamfit

LEVELS = [0.1, 0.5, 0.9]
FORMULA = "y ~ s(x)"


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
    # No skip (SPEC 16): an unbuilt CLI is a real gap and must read red.
    raise AssertionError(
        "no `gam` CLI binary found (GAM_BIN, target/release/gam, target/debug/gam, PATH)"
    )


def _heteroscedastic_data(n: int = 600, seed: int = 20260919) -> dict[str, list[float]]:
    rng = random.Random(seed)
    xs = [i / (n - 1) for i in range(n)]
    ys = [
        1.0 + math.sin(3.0 * x) + (0.02 + 1.2 * (1.0 - x) ** 2) * rng.gauss(0.0, 1.0)
        for x in xs
    ]
    return {"x": xs, "y": ys}


def _dense_grid() -> dict[str, list[float]]:
    # The data range [0, 1] extended by a quarter on each side.
    xs = [-0.25 + 1.5 * i / 600 for i in range(601)]
    return {"x": xs, "y": [0.0] * len(xs)}


def test_joint_expectile_curves_are_ordered_where_separate_fits_cross() -> None:
    data = _heteroscedastic_data()
    grid = _dense_grid()

    separate = np.column_stack(
        [
            np.asarray(
                gamfit.fit(data, FORMULA, family="expectile", expectile_tau=tau).predict(grid)
            ).reshape(-1)
            for tau in LEVELS
        ]
    )
    assert np.any(np.diff(separate, axis=1) <= 0.0), (
        "fixture must be one where independently fitted adjacent expectiles cross"
    )

    joint = np.asarray(
        gamfit.fit(data, FORMULA, family="expectile", expectile_tau=LEVELS).predict(grid)
    )
    assert joint.shape == (len(grid["x"]), len(LEVELS))
    gaps = np.diff(joint, axis=1)
    assert np.all(gaps > 0.0), f"joint expectile curves cross; smallest gap {gaps.min():.3e}"


def test_one_level_list_is_the_scalar_expectile_fit() -> None:
    data = _heteroscedastic_data(n=300, seed=7)
    grid = _dense_grid()
    scalar = np.asarray(
        gamfit.fit(data, FORMULA, family="expectile", expectile_tau=0.9).predict(grid)
    )
    listed = np.asarray(
        gamfit.fit(data, FORMULA, family="expectile", expectile_tau=[0.9]).predict(grid)
    )
    assert scalar.shape == listed.shape == (len(grid["x"]),)
    np.testing.assert_array_equal(listed, scalar)


def test_joint_expectile_survives_save_and_load(tmp_path: Path) -> None:
    data = _heteroscedastic_data(n=300, seed=11)
    grid = _dense_grid()
    model = gamfit.fit(data, FORMULA, family="expectile", expectile_tau=LEVELS)
    path = tmp_path / "joint.gam"
    model.save(path)
    np.testing.assert_array_equal(
        np.asarray(gamfit.load(path).predict(grid)), np.asarray(model.predict(grid))
    )


def test_joint_expectile_cli_and_python_agree(tmp_path: Path) -> None:
    data = _heteroscedastic_data(n=300, seed=13)
    grid = _dense_grid()
    data_path = tmp_path / "train.csv"
    grid_path = tmp_path / "grid.csv"
    for path, table in ((data_path, data), (grid_path, grid)):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y"])
            writer.writerows(zip(table["x"], table["y"]))

    gam = _gam_binary()
    model_path = tmp_path / "joint.gam"
    pred_path = tmp_path / "pred.csv"
    subprocess.run(
        [
            gam,
            "fit",
            "--family",
            "expectile",
            "--expectile-tau",
            ",".join(str(tau) for tau in LEVELS),
            "--out",
            str(model_path),
            str(data_path),
            FORMULA,
        ],
        check=True,
    )
    subprocess.run(
        [gam, "predict", "--out", str(pred_path), str(model_path), str(grid_path)],
        check=True,
    )
    with pred_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = [f"expectile_{tau}" for tau in LEVELS]
    cli = np.array([[float(row[name]) for name in columns] for row in rows])

    python = np.asarray(
        gamfit.fit(data, FORMULA, family="expectile", expectile_tau=LEVELS).predict(grid)
    )
    cli_model_in_python = np.asarray(gamfit.load(model_path).predict(grid))
    assert cli.shape == python.shape == (len(grid["x"]), len(LEVELS))
    np.testing.assert_allclose(cli, python, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cli_model_in_python, cli, rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(cli, axis=1) > 0.0)
