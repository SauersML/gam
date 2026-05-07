"""End-to-end PGS pipeline tests through the Python library.

These tests mirror the three-stage pipeline that the deleted
``examples/pgs_calibration_pipeline.py`` script exercised, but as proper
pytest cases against the ``gam`` Python binding:

* Stage 1 — conditional Gaussianization of ``PGS`` on the PC manifold.
* Stage 2a — Bernoulli marginal-slope (probit) on the calibrated z.
* Survival risk transfer — calibrated PGS risk on the left-truncated outcome.

Unlike the per-stage smoke tests in ``tests/test_python_api.py`` (which
fit and predict on a single dataset), these tests split the synthetic
biobank into disjoint train/test halves, fit every stage on train only,
and check downstream quality on the held-out test rows. This exercises
the deployment path: a calibration learned on training data must transfer
to unseen samples, and the downstream binary / survival checks must keep
their headline accuracy on out-of-sample rows.
"""

from __future__ import annotations
import typing

import math

import pytest

pytest.importorskip("gam._rust")

import numpy as np
import pandas as pd

import gam
from gam.pgs import PgsCalibration


PC_COLUMNS = ["pc1", "pc2", "pc3", "pc4"]


def _require_extension() -> None:
    if not gam.build_info().get("available"):
        pytest.skip("rust extension not built")


def _pc_duchon(centers: int = 6) -> str:
    return (
        f"duchon(pc1, pc2, pc3, pc4, centers={centers}, "
        "order=0, power=2, length_scale=1)"
    )


def _split_train_test(
    df: pd.DataFrame, test_fraction: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic 50/50 split that preserves the input row order."""
    n = len(df)
    cut = n - int(round(n * test_fraction))
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    y = y_true[order]
    pos = float(y.sum())
    neg = float(y.shape[0] - pos)
    if pos == 0.0 or neg == 0.0:
        return float("nan")
    ranks = np.arange(1, y.shape[0] + 1, dtype=float)
    return float((ranks[y > 0.5].sum() - pos * (pos + 1.0) / 2.0) / (pos * neg))


def _c_index(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    for i in range(times.shape[0]):
        if events[i] < 0.5:
            continue
        for j in range(times.shape[0]):
            if j == i or times[j] <= times[i]:
                continue
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1.0
            elif math.isclose(float(risk[i]), float(risk[j])):
                concordant += 0.5
    return concordant / comparable if comparable else float("nan")


def test_e2e_stage1_calibration_transfers_to_heldout_test(
    synthetic_biobank_factory: typing.Any,
) -> None:
    """Calibrate on train, transform test, check held-out z is ~ N(0, 1).

    Stage 1's contract is that ``h(PGS | PCs) ~ N(0, 1)`` after the map is
    fit. The strong form of that contract is *transfer*: a calibration
    learned on a training cohort must produce z-scores that are still
    approximately standard normal (and uncorrelated with the PCs) on a
    held-out cohort. Without that, the downstream marginal-slope models
    cannot trust ``pgs_ctn_z`` at deployment time.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=20, n=256)
    train, test = _split_train_test(df)

    calibration = PgsCalibration(
        pgs_column="PGS",
        pc_columns=PC_COLUMNS,
        duchon_centers=5,
        out_column="pgs_ctn_z",
    ).fit(train)

    test_cal = calibration.transform(test)
    z_test = np.asarray(test_cal["pgs_ctn_z"].to_numpy(), dtype=float)

    assert z_test.shape == (len(test),)
    assert np.all(np.isfinite(z_test))

    z_mean = float(z_test.mean())
    z_std = float(z_test.std(ddof=0))
    assert abs(z_mean) < 0.30, f"held-out z mean drifted: {z_mean:+.3f}"
    assert 0.50 < z_std < 1.75, f"held-out z std out of band: {z_std:.3f}"

    # Held-out z must be approximately uncorrelated with each PC; the
    # Gaussianization map purges PC structure on train and that should
    # transfer to test up to small-sample noise.
    for pc in PC_COLUMNS:
        rho = float(np.corrcoef(z_test, test[pc].to_numpy())[0, 1])
        assert abs(rho) < 0.40, f"corr(z, {pc}) on test = {rho:+.3f}"


def test_e2e_stage1_then_stage2a_binary_holdout_auc(
    synthetic_biobank_factory: typing.Any,
) -> None:
    """Stage 1 → Stage 2a end-to-end: train, predict on held-out, AUC > 0.6.

    The synthetic biobank generates ``disease`` from a probit of
    ``0.5 * PGS + 0.1 * pc1`` (see ``tests/conftest.py``). After Stage 1
    folds the PC dependence into ``pgs_ctn_z``, a Bernoulli marginal-slope
    fit with a constant marginal slope should comfortably discriminate
    cases from controls on a held-out half. AUC > 0.60 is a loose floor;
    the data-generating signal supports stronger performance at this
    sample size, but seed-to-seed variability makes a tighter threshold
    flaky.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=21, n=256)
    train, test = _split_train_test(df)

    calib = gam.fit(
        train,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    train["pgs_ctn_z"] = np.asarray(calib.predict(train), dtype=float)
    test["pgs_ctn_z"] = np.asarray(calib.predict(test), dtype=float)

    model = gam.fit(
        train,
        "disease ~ z",
        family="bernoulli-marginal-slope",
        link="probit",
        scale_dimensions=True,
        z_column="pgs_ctn_z",
        logslope_formula="1",
    )

    pred = model.predict(test, return_type="dict")
    probs = np.asarray(pred["mean"], dtype=float)
    assert probs.shape == (len(test),)
    assert np.all(np.isfinite(probs))
    assert np.all((probs > 0.0) & (probs < 1.0))

    auc = _auc(test["disease"].to_numpy(), probs)
    assert math.isfinite(auc), "held-out AUC is NaN — both classes required"
    assert auc > 0.60, f"held-out AUC = {auc:.3f} below 0.60 floor"


def test_e2e_stage1_calibrated_pgs_survival_holdout_cindex(
    synthetic_biobank_factory: typing.Any,
) -> None:
    """Stage 1 survival transfer: train, score held-out, C-index > 0.55.

    The synthetic biobank's hazard is ``lam = exp(-1.2 - 0.3 * PGS)``, so
    higher PGS ⇒ lower hazard ⇒ longer survival. After Stage 1 folds PC
    structure into ``pgs_ctn_z``, the calibrated score itself should
    discriminate short- from long-lived held-out samples. C-index > 0.55 is
    the loose floor; the data-generating signal is moderate, and a thin
    biobank sample inflates seed variance.
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=22, n=256)
    train, test = _split_train_test(df)

    calib = gam.fit(
        train,
        f"PGS ~ {_pc_duchon()}",
        transformation_normal=True,
        scale_dimensions=True,
    )
    test["pgs_ctn_z"] = np.asarray(calib.predict(test), dtype=float)

    risk = -test["pgs_ctn_z"].to_numpy(dtype=float)
    cindex = _c_index(
        test["age_exit"].to_numpy(),
        test["event"].to_numpy(),
        risk,
    )
    assert math.isfinite(cindex), "C-index is NaN — no comparable pairs"
    assert cindex > 0.55, f"held-out C-index = {cindex:.3f} below 0.55 floor"


def test_e2e_pipeline_save_and_reload_predicts_identically(
    synthetic_biobank_factory: typing.Any, tmp_path: typing.Any
) -> None:
    """Roundtrip the full pipeline (calibration + binary model) through disk.

    A deployment-time concern: after Stage 1 calibration is fit on train,
    serialized to disk, and reloaded, predictions on the held-out test
    set must match the in-memory model bit-for-bit. Same for the Stage 2a
    Bernoulli marginal-slope model. This guards against silent state loss
    in ``save`` / ``load`` (e.g. missing marginal-slope metadata).
    """
    _require_extension()
    df = synthetic_biobank_factory(seed=23, n=256)
    train, test = _split_train_test(df)

    calibration = PgsCalibration(
        pgs_column="PGS",
        pc_columns=PC_COLUMNS,
        duchon_centers=5,
        out_column="pgs_ctn_z",
    ).fit(train)
    calib_path = tmp_path / "stage1.pgs_calibration"
    calibration.save(calib_path)
    reloaded_calib = PgsCalibration.load(calib_path)

    z_inmem = np.asarray(
        calibration.transform(test)["pgs_ctn_z"].to_numpy(), dtype=float
    )
    z_disk = np.asarray(
        reloaded_calib.transform(test)["pgs_ctn_z"].to_numpy(), dtype=float
    )
    np.testing.assert_allclose(z_inmem, z_disk, rtol=0.0, atol=1e-10)

    train["pgs_ctn_z"] = np.asarray(
        calibration.transform(train)["pgs_ctn_z"].to_numpy(), dtype=float
    )
    test["pgs_ctn_z"] = z_inmem

    model = gam.fit(
        train,
        "disease ~ z",
        family="bernoulli-marginal-slope",
        link="probit",
        scale_dimensions=True,
        z_column="pgs_ctn_z",
        logslope_formula="1",
    )

    model_path = tmp_path / "stage2a.gam"
    model.save(model_path)
    reloaded_model = gam.load(model_path)

    probs_inmem = np.asarray(
        model.predict(test, return_type="dict")["mean"], dtype=float
    )
    probs_disk = np.asarray(
        reloaded_model.predict(test, return_type="dict")["mean"], dtype=float
    )
    np.testing.assert_allclose(probs_inmem, probs_disk, rtol=0.0, atol=1e-10)
