"""Bug hunt: the LAST factor level of a ``s(x, by=<factor>)`` smooth has its
per-term EDF (and ref_df / chi_sq / p_value) reported as 0 / NaN in
``Model.summary()`` even though that group's smooth is genuinely fitted.

A factor ``by=`` smooth expands into one centred smooth per level (see
``src/terms/term_builder.rs:644-659``, the ``ColumnKindTag::Categorical`` arm,
which pushes one ``SmoothBasisSpec::ByVariable`` block per ``(level_bits,
level_label)``). With ``L`` levels the model therefore carries ``L`` smooth
terms, each estimating that group's curve. Their effective degrees of freedom is
``tr(F)`` restricted to the term's coefficient block, where ``F`` is the
whole-model influence/hat operator
(``src/model_types/result_types.rs::per_term_edf``).

Observed: for any number of levels, the per-level EDFs reported by
``summary().smooth_terms`` are correct for every level *except the last*, whose
EDF, ref_df, chi_sq and p_value all come back ``0`` / ``NaN`` — as if the group's
smooth had been shrunk to a flat line. It was not: predictions for that group
recover its (strongly non-linear) curve, and the missing EDF is exactly the gap
between ``edf_total`` and the sum of the reported per-level EDFs. The per-term
EDF / smooth-test window for the last term runs off the end of the bookkeeping
(an off-by-one in the smooth-block column / penalty cursor accounting, the
``smooth_start`` / ``penalty_cursor`` walk in ``src/main/model_summary.rs:198``
and the ``per_term_edf`` fallbacks at ``result_types.rs:1860-1882``), so the
genuinely-fitted last group is silently reported as carrying zero complexity and
no significance.

This is a *summary / introspection* defect, not a fit defect: the point
predictions for the last group are correct. But a user reading ``summary()``
sees one group's smooth as identically zero EDF with a NaN significance test,
which is wrong and misleading.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _fit_by_factor_smooth(n_levels, seed=7, n_per_level=500):
    """Fit ``y ~ s(x, by=grp)`` where every group carries an equally strong,
    equally wiggly (phase-shifted) sinusoid — so by construction every group's
    true smooth is strongly non-linear and its honest EDF is well above 1."""
    rng = np.random.default_rng(seed)
    n = n_per_level * n_levels
    x = rng.uniform(0.0, 1.0, n)
    labels = [f"g{i}" for i in range(n_levels)]
    g = rng.choice(labels, n)
    phase = {lab: i for i, lab in enumerate(labels)}
    mean = np.sin(2.0 * np.pi * x + np.array([phase[gi] for gi in g]))
    y = mean + 0.15 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "grp": g, "y": y})
    return gamfit.fit(df, "y~s(x,by=grp)", family="gaussian")


def _by_level_edf_rows(model):
    frame = model.summary().smooth_terms_frame()
    mask = frame["name"].astype(str).str.startswith("s(x,by=grp):")
    return frame[mask]


def test_every_by_factor_level_reports_positive_edf():
    """Every group's by-smooth carries a strongly non-linear signal, so every
    per-level EDF must be comfortably above 1. The last level currently reports
    exactly 0.0 (its smooth EDF is silently dropped)."""
    for n_levels in (2, 3, 5):
        model = _fit_by_factor_smooth(n_levels)
        rows = _by_level_edf_rows(model)
        assert len(rows) == n_levels, (
            f"expected {n_levels} by-level smooth terms, got {len(rows)}: "
            f"{list(rows['name'])}"
        )
        edfs = rows["edf"].to_numpy(dtype=float)
        # All groups are exchangeable strong sinusoids; the genuine per-level EDF
        # is ~10-20 here. None should come back at (or near) zero.
        assert edfs.min() > 1.0, (
            f"n_levels={n_levels}: a by-factor level reports EDF "
            f"{edfs.min():.4f} (rounded set {np.round(edfs, 3).tolist()}). Every "
            "group is genuinely fitted with a strongly non-linear smooth, so no "
            "per-level EDF should be ~0 — the last level's EDF is being dropped "
            "by the summary per-term accounting."
        )


def test_by_factor_per_level_edf_accounts_for_model_total():
    """The reported per-level EDFs must add up to (about) the model's total EDF.
    The dropped last-level EDF makes them fall short by that level's worth of
    degrees of freedom."""
    model = _fit_by_factor_smooth(5)
    summary = model.summary()
    rows = _by_level_edf_rows(model)
    edf_sum = float(rows["edf"].to_numpy(dtype=float).sum())
    edf_total = float(summary.edf_total)
    # The five centred by-smooths supply essentially all of the model's EDF
    # (only the intercept and an unpenalised treatment-coded factor main effect
    # sit outside them). A whole missing level leaves a double-digit shortfall.
    assert edf_sum >= edf_total - 5.0, (
        f"sum of per-level by-smooth EDFs {edf_sum:.3f} falls far short of the "
        f"model total EDF {edf_total:.3f}: a by-factor level's EDF is missing "
        f"from summary().smooth_terms (per-level EDFs: "
        f"{np.round(rows['edf'].to_numpy(dtype=float), 3).tolist()})."
    )


def test_by_factor_last_level_significance_is_not_nan():
    """A genuinely-fitted group's smooth significance test must produce a real
    statistic, not NaN. The last level currently returns chi_sq / p_value NaN
    because its EDF was dropped to 0."""
    model = _fit_by_factor_smooth(5)
    rows = _by_level_edf_rows(model)
    assert len(rows) == 5, f"expected 5 by-level smooth terms, got {len(rows)}"
    chi_sq = rows["chi_sq"].to_numpy(dtype=float)
    assert np.isfinite(chi_sq).all(), (
        "a by-factor level's smooth significance test is NaN "
        f"(chi_sq values {chi_sq.tolist()}): the last group is genuinely fitted "
        "but is reported with zero EDF and no significance."
    )
