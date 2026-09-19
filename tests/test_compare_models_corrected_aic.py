"""Model comparison ranks on the smoothing-corrected AIC, in one Rust function.

pyGAM audit d11 (bench/pygam_audit, slop.md G2): ``gamfit.compare_models`` and
``Model.evidence`` ranked fits on an uncorrected ``-2*loglik + 2*edf`` that the
Python FFI assembled from summary fields. It counted no scale parameter and no
Wood-Pya-Saefken correction for having estimated the smoothing parameters, so on
``y ~ s(x)`` against ``y ~ s(x) + s(z)`` with ``z`` pure noise it preferred the
noise model in 9 of 20 fixed-seed replicates -- a coin flip.

The ranking is now ``compare_saved_models`` in Rust: it reads
``aic_corrected = -2*loglik + 2*(edf_corrected + scale_dof)`` from the fitted
summary, and both ``gamfit.compare_models`` and ``gam compare`` print its
serialized result.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import gamfit

D11_REPS = 20
D11_SEED = 7
D11_N = 200


def _d11_reps():
    rng = np.random.default_rng(D11_SEED)
    for _ in range(D11_REPS):
        x = rng.uniform(0.0, 1.0, D11_N)
        z = rng.uniform(0.0, 1.0, D11_N)
        y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.5, D11_N)
        yield {"x": x, "z": z, "y": y}


def test_d11_corrected_aic_prefers_the_true_model() -> None:
    true_wins = 0
    for data in _d11_reps():
        truth = gamfit.fit(data, "y ~ s(x)")
        noise = gamfit.fit(data, "y ~ s(x) + s(z)")
        comparison = gamfit.compare_models([truth, noise], names=["truth", "noise"])
        assert comparison["criterion"] == "aic_corrected"
        true_wins += comparison["winner"] == "truth"
    # Measured: the true model wins 17 of 20 replicates on this seed (the
    # uncorrected ranking it replaces managed 11 of 20).
    assert true_wins >= 15, (
        f"compare_models preferred y ~ s(x) over y ~ s(x) + s(z) (z pure noise) "
        f"in only {true_wins}/{D11_REPS} fixed-seed replicates"
    )


def test_ranking_rows_are_the_summary_information_criteria() -> None:
    data = next(_d11_reps())
    truth = gamfit.fit(data, "y ~ s(x)")
    noise = gamfit.fit(data, "y ~ s(x) + s(z)")
    comparison = gamfit.compare_models([truth, noise], names=["truth", "noise"])
    rows = {row["name"]: row for row in comparison["ranking"]}
    best = min(
        truth.summary().aic_corrected, noise.summary().aic_corrected
    )
    for name, model in (("truth", truth), ("noise", noise)):
        summary = model.summary()
        # Gaussian: the profiled scale is one more estimated parameter.
        assert summary.scale_dof == 1.0
        assert summary.aic_conditional == pytest.approx(
            -2.0 * summary.log_likelihood + 2.0 * (summary.edf_total + 1.0)
        )
        assert summary.edf_corrected >= summary.edf_total
        assert summary.aic_corrected == pytest.approx(
            -2.0 * summary.log_likelihood + 2.0 * (summary.edf_corrected + 1.0)
        )
        row = rows[name]
        assert row["aic_corrected"] == summary.aic_corrected
        assert row["aic_conditional"] == summary.aic_conditional
        assert row["edf_corrected"] == summary.edf_corrected
        assert row["edf_conditional"] == summary.edf_total
        assert row["delta_aic"] == pytest.approx(summary.aic_corrected - best)
        assert row["evidence_ratio"] == pytest.approx(math.exp(0.5 * row["delta_aic"]))
    # The pairwise ratio is the same criterion as the table.
    assert math.log(truth.evidence_ratio_vs(noise)) == pytest.approx(
        0.5 * (rows["noise"]["aic_corrected"] - rows["truth"]["aic_corrected"])
    )


def _binary_data(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    return {"x": x, "z": z, "eta": np.sin(2.0 * np.pi * x), "rng": rng}


@pytest.mark.parametrize(
    "family, draw, scale_dof",
    [
        ("binomial", lambda eta, rng: rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))), 0.0),
        ("poisson", lambda eta, rng: rng.poisson(np.exp(eta)), 0.0),
        ("gamma", lambda eta, rng: rng.gamma(4.0, np.exp(eta) / 4.0), 1.0),
    ],
)
def test_corrected_aic_is_formed_for_non_gaussian_families(family, draw, scale_dof) -> None:
    base = _binary_data(11)
    data = {
        "x": base["x"],
        "z": base["z"],
        "y": draw(base["eta"], base["rng"]).astype(float),
    }
    model = gamfit.fit(data, "y ~ s(x) + s(z)", family=family)
    summary = model.summary()
    assert summary.aic_corrected_unavailable is None
    assert summary.scale_dof == scale_dof
    assert summary.edf_corrected >= summary.edf_total
    assert summary.aic_corrected == pytest.approx(
        -2.0 * summary.log_likelihood + 2.0 * (summary.edf_corrected + scale_dof)
    )
    comparison = gamfit.compare_models([model], names=[family])
    assert comparison["ranking"][0]["aic_corrected"] == summary.aic_corrected


def test_corrected_aic_is_formed_beyond_four_smoothing_parameters() -> None:
    rng = np.random.default_rng(5)
    n = 400
    data = {f"x{j}": rng.uniform(0.0, 1.0, n) for j in range(1, 6)}
    data["y"] = (
        np.sin(2.0 * np.pi * data["x1"]) + data["x2"] ** 2 + rng.normal(0.0, 0.5, n)
    )
    model = gamfit.fit(data, "y ~ s(x1) + s(x2) + s(x3) + s(x4) + s(x5)")
    summary = model.summary()
    # Five double-penalized smooths carry ten smoothing parameters.
    assert len(summary.lambdas) > 4
    assert summary.aic_corrected_unavailable is None
    assert summary.edf_corrected > summary.edf_total
    assert np.isfinite(summary.aic_corrected)


def test_compare_models_refuses_a_fit_without_corrected_aic() -> None:
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    data = {"x": x, "y": np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, n)}
    dense = gamfit.fit(data, "y ~ s(x)")
    scan = gamfit.fit(data, "y ~ s(x)", double_penalty=False)
    reason = scan.summary().aic_corrected_unavailable
    assert reason is not None and scan.summary().aic_corrected is None
    # Never a fallback to the conditional AIC: the refusal carries the reason.
    with pytest.raises(ValueError, match="spline-scan"):
        gamfit.compare_models([dense, scan], names=["dense", "scan"])
    with pytest.raises(ValueError, match="spline-scan"):
        dense.evidence_ratio_vs(scan)


def test_compare_models_takes_models_not_summary_mappings() -> None:
    with pytest.raises(TypeError, match="gamfit.Model"):
        gamfit.compare_models([{"aic_corrected": 1.0}], names=["mapping"])


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
    # No skip: an unbuilt CLI is a real gap in this parity check.
    raise AssertionError(
        "no `gam` CLI binary found (GAM_BIN, target/release/gam, target/debug/gam, PATH)"
    )


def test_cli_compare_prints_the_python_comparison(tmp_path: Path) -> None:
    data = next(_d11_reps())
    truth = gamfit.fit(data, "y ~ s(x)")
    noise = gamfit.fit(data, "y ~ s(x) + s(z)")
    truth_path = tmp_path / "truth.gam"
    noise_path = tmp_path / "noise.gam"
    truth.save(truth_path)
    noise.save(noise_path)

    python_result = gamfit.compare_models([truth, noise], names=["truth", "noise"])
    completed = subprocess.run(
        [
            _gam_binary(),
            "compare",
            str(truth_path),
            str(noise_path),
            "--names",
            "truth",
            "noise",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_result = json.loads(completed.stdout)
    # One Rust function behind both front ends: the documents are identical.
    assert cli_result == python_result
