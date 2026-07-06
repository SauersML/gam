import csv
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest


def gam_bin():
    candidates = [
        os.environ.get("GAM_BIN"),
        "target/release/gam",
        "target/debug/gam",
        shutil.which("gam"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    pytest.skip("gam binary not built")


def write_data(tmp_path):
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    poisson = tmp_path / "poisson.csv"
    gaussian = tmp_path / "gaussian.csv"
    pd.DataFrame(
        {"y": rng.poisson(np.exp(0.3 + np.sin(2 * np.pi * x))), "x": x}
    ).to_csv(poisson, index=False)
    pd.DataFrame(
        {"y": np.sin(2 * np.pi * x) + rng.normal(0.0, 0.3, n), "x": x}
    ).to_csv(gaussian, index=False)
    return poisson, gaussian


def run_gam(*args):
    subprocess.run([gam_bin(), *map(str, args)], check=True)


def header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def test_gaussian_reference_honours_uncertainty_flag(tmp_path):
    _, gaussian = write_data(tmp_path)
    model = tmp_path / "gaussian.gam"
    no_uncertainty = tmp_path / "gaussian-no.csv"
    yes_uncertainty = tmp_path / "gaussian-yes.csv"

    run_gam("fit", gaussian, "y ~ s(x)", "--family", "gaussian", "--out", model, "-q")
    run_gam("predict", model, gaussian, "--out", no_uncertainty, "-q")
    run_gam("predict", model, gaussian, "--out", yes_uncertainty, "--uncertainty", "-q")

    assert header(no_uncertainty) == ["eta", "mean"]
    assert header(yes_uncertainty) == [
        "eta",
        "mean",
        "std_error",
        "mean_lower",
        "mean_upper",
    ]


def test_poisson_predict_without_uncertainty_must_not_emit_band_columns(tmp_path):
    poisson, _ = write_data(tmp_path)
    model = tmp_path / "poisson.gam"
    no_uncertainty = tmp_path / "poisson-no.csv"

    run_gam("fit", poisson, "y ~ s(x)", "--family", "poisson-log", "--out", model, "-q")
    run_gam("predict", model, poisson, "--out", no_uncertainty, "-q")

    assert header(no_uncertainty) == ["eta", "mean"]


def test_uncertainty_flag_is_not_a_noop_for_poisson(tmp_path):
    poisson, _ = write_data(tmp_path)
    model = tmp_path / "poisson.gam"
    no_uncertainty = tmp_path / "poisson-no.csv"
    yes_uncertainty = tmp_path / "poisson-yes.csv"

    run_gam("fit", poisson, "y ~ s(x)", "--family", "poisson-log", "--out", model, "-q")
    run_gam("predict", model, poisson, "--out", no_uncertainty, "-q")
    run_gam("predict", model, poisson, "--out", yes_uncertainty, "--uncertainty", "-q")

    assert header(no_uncertainty) == ["eta", "mean"]
    assert header(yes_uncertainty) == [
        "eta",
        "mean",
        "std_error",
        "mean_lower",
        "mean_upper",
    ]
