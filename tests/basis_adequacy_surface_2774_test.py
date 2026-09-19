"""gam#2774 — the Python surface of the per-smooth basis-adequacy check.

The engine-side statistic and its size/power are pinned in Rust
(``crates/gam-terms/src/inference/basis_adequacy.rs`` and
``tests/basis_adequacy_confounded_exposure_2774.rs``). What this file pins is
the part a gamfit user actually touches, and the part the issue says was
missing: a fit that cannot represent what it was asked to model must SAY SO,
through the ordinary surfaces, without the caller knowing to look.

Four independent angles on the same fix, so a regression is caught even if one
assertion drifts:

1. ``gamfit.fit`` raises a :class:`~gamfit.errors.GamInferenceWarning` at fit time.
   This is the channel the user cannot miss; everything else requires them to
   go looking.
2. ``summary().basis_checks`` carries the evidence with NO data and no refit —
   the persisted-payload path, which is what a saved model can offer.
3. ``Model.basis_check(data)`` recomputes it from the training rows and agrees
   with the persisted verdict.
4. The same design with an ADEQUATE basis warns about nothing. Without this a
   diagnostic that fires unconditionally would pass every other check here.

The fixture is a reduced version of the filed one: a null exposure correlated
with 16 population PCs, adjusted by ``duchon(pc1..pc16, centers=24)``.

Two fixture choices are measured rather than assumed:

* ``scale_dimensions=True``, matching the issue's own repeated-calibration
  table. It is not cosmetic. The 16 PCs carry geometrically decreasing scales
  (3.0 down to 0.45), so with scaling OFF an isotropic kernel concentrates its
  resolution on the high-variance axes — exactly where the simulated effect
  lives — and 24 centers get much closer to the surface. With it ON the kernel
  spreads over 13 axes that carry nothing. One seed at ``n = 3000``: lack-of-fit
  ``p = 0.43`` off against ``p = 6e-11`` on.
* ``n = 12000``. The statistic's excess over its reference d.f. grows linearly
  in ``n``; at ``n = 3000`` with scaling on it is already ~107 against ``r = 75``,
  so 12000 puts the flagged arm orders of magnitude clear of the threshold
  rather than a factor of a few, which is what an arm gated at ``p < 1e-3``
  needs from a single seed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
gamfit = pytest.importorskip("gamfit")

PC_DIMENSION = 16
CENTERS = 24
N_ROWS = 12000
SEED = 20260820

#: The engine's own fit-time note level, Bonferroni-corrected over the terms
#: that produced a verdict. Asserting against the same threshold the note uses
#: keeps this file from drifting away from the behaviour it is describing.
NOTE_LEVEL = 1.0e-3


def _logistic(value):
    return 1.0 / (1.0 + np.exp(-value))


def _confounded_frame(effect: str) -> pd.DataFrame:
    """The filed DGP: exposure correlated with the PCs, outcome with a PC effect
    and **no exposure effect conditional on the PCs**.

    ``effect="rotated_curved"`` is the filed alternative — nonlinear, rotated
    and anisotropic, hence outside anything seven kernel columns in 16
    dimensions can reach. ``effect="linear"`` lies exactly inside the Duchon's
    own unpenalized linear null space, so the basis is adequate by
    construction; it is the negative control.
    """
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(N_ROWS, PC_DIMENSION))
    pcs = z * np.geomspace(3.0, 0.45, PC_DIMENSION)
    frequency = _logistic(-0.7 + 0.75 * z[:, 0] - 0.55 * z[:, 1] + 0.25 * z[:, 2])
    dosage = rng.binomial(2, frequency).astype(float)
    if effect == "rotated_curved":
        u = (z[:, 0] + z[:, 1]) / math.sqrt(2.0)
        v = (z[:, 0] - z[:, 1]) / math.sqrt(2.0)
        pc_effect = 0.70 * np.sin(1.6 * u) + 0.38 * (v * v - 1.0)
    elif effect == "linear":
        pc_effect = 0.6 * z[:, 0] - 0.4 * z[:, 1] + 0.3 * z[:, 2]
    else:  # pragma: no cover - guards a typo in a parametrization
        raise ValueError(effect)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _logistic(mid + pc_effect).mean() < 0.10:
            lo = mid
        else:
            hi = mid
    y = rng.binomial(1, _logistic(0.5 * (lo + hi) + pc_effect)).astype(float)
    frame = pd.DataFrame({"y": y, "dosage": dosage})
    for column in range(PC_DIMENSION):
        frame[f"pc{column + 1}"] = pcs[:, column]
    return frame


def _formula() -> str:
    covariates = ", ".join(f"pc{j}" for j in range(1, PC_DIMENSION + 1))
    return f"y ~ dosage + duchon({covariates}, centers={CENTERS})"


def _fit(effect: str, scale_dimensions: bool = True):
    """Fit the fixture.

    ``scale_dimensions`` is NOT cosmetic here, and it is measured rather than
    assumed: the 16 PCs carry geometrically decreasing scales, so with scaling
    OFF an isotropic kernel spends its resolution on the high-variance axes —
    exactly where the simulated effect lives — and 24 centers reach the surface.
    With it ON every axis is equalized, the kernel spreads over 13 axes that
    carry nothing, and the basis genuinely cannot reach it. One seed at
    ``n = 3000``: lack-of-fit ``p = 0.43`` off against ``p = 6e-11`` on. The
    issue's repeated-calibration table is ``scale_dimensions=True``, which is
    what these fixtures use.
    """
    frame = _confounded_frame(effect)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gamfit.fit(
            frame, _formula(), family="binomial", scale_dimensions=scale_dimensions
        )
    return frame, model, caught


@pytest.fixture(scope="module")
def underfitted():
    return _fit("rotated_curved")


@pytest.fixture(scope="module")
def adequate():
    return _fit("linear")


def _adequacy_warnings(caught) -> list[str]:
    return [
        str(record.message)
        for record in caught
        if issubclass(record.category, gamfit.errors.GamInferenceWarning)
        and "basis adequacy" in str(record.message)
    ]


def test_underfitted_smooth_warns_at_fit_time(underfitted):
    """The channel a user cannot miss.

    Everything else in this file requires the caller to go looking; the filed
    complaint is precisely that a caller who does not go looking receives
    ``certified=True`` and a very confident false association.
    """
    _, _, caught = underfitted
    messages = _adequacy_warnings(caught)
    assert messages, "an underfitted smooth must raise a GamInferenceWarning at fit time"
    message = messages[0]
    assert "duchon" in message, message
    # The advisory has to name the remedy and disclaim the certificate, not just
    # announce a p-value.
    assert "larger basis" in message, message
    assert "convergence certificate covers the optimizer only" in message, message


def test_summary_carries_the_evidence_without_data(underfitted):
    """The persisted-payload path: a saved model can answer this with no rows.

    The score is a function of converged IRLS row state a saved model does not
    carry, so if the fit did not persist its verdict there is nothing a
    data-free ``summary()`` could report.
    """
    _, model, _ = underfitted
    summary = model.summary()
    assert summary.convergence["certified"] is True, (
        "the filed fit converges and certifies — that is the whole complaint"
    )
    checks = summary.basis_checks
    assert len(checks) == 1, checks
    row = checks[0]
    assert row["provenance"] == "radial_enrichment"
    assert row["p_value"] < NOTE_LEVEL, row
    # ``nullspace_dim`` is the JOINT null space — the directions no penalty
    # touches — and it is 0 here because a Duchon term is DOUBLE penalized (the
    # RKHS curvature Gram plus a complementary trend ridge on the polynomial
    # block). That is precisely why an EDF reading misleads on this term: the
    # polynomial block is d+1 = 17 of the columns and is only WEAKLY penalized,
    # so it carries most of the EDF and makes the fit look near-saturated while
    # the lack-of-fit test says the span is the problem.
    assert row["nullspace_dim"] == 0, row
    assert 0.0 < row["edf"] <= row["basis_dim"] + 1e-6, row
    assert row["enrichment_rank"] > row["basis_dim"], (
        "the alternative must carry more resolution than the fitted basis",
        row,
    )


def test_basis_check_recomputes_and_agrees(underfitted):
    """The data-carrying path agrees with the persisted verdict.

    ``basis_check`` refits at the frozen spec, so this is not a tautology: it
    re-derives the IRLS row state the fit-time report was computed from. The two
    must reach the same conclusion, or one of them is measuring a different
    model.
    """
    frame, model, _ = underfitted
    recomputed = model.basis_check(frame)
    assert len(recomputed) == 1, recomputed
    row = recomputed[0]
    persisted = model.summary().basis_checks[0]
    assert row["provenance"] == persisted["provenance"] == "radial_enrichment"
    assert row["basis_dim"] == persisted["basis_dim"]
    assert row["nullspace_dim"] == persisted["nullspace_dim"]
    assert row["edf"] == pytest.approx(persisted["edf"], rel=1e-6)
    assert row["p_value"] < NOTE_LEVEL, row
    # Same fit, same rows, same frozen spec: the statistic should land in the
    # same place rather than merely on the same side of the threshold.
    assert row["statistic"] == pytest.approx(persisted["statistic"], rel=1e-6), (
        row,
        persisted,
    )


def test_adequate_basis_warns_about_nothing(adequate):
    """The negative control, and the reason the rest of this file is not vacuous.

    Same model, same basis, same ``n`` — a truth that lives exactly inside the
    smooth's own unpenalized null space. A diagnostic that fires here would fire
    on essentially every GAM, and a warning that always fires is not a warning.
    """
    frame, model, caught = adequate
    assert not _adequacy_warnings(caught), _adequacy_warnings(caught)
    checks = model.summary().basis_checks
    assert len(checks) == 1, checks
    assert checks[0]["provenance"] == "radial_enrichment"
    assert checks[0]["p_value"] > NOTE_LEVEL, checks[0]
    assert model.basis_check(frame)[0]["p_value"] > NOTE_LEVEL


def test_a_model_without_smooths_reports_no_rows():
    """A purely parametric fit has nothing to check, and says so with an empty
    table rather than a fabricated verdict."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=400)
    y = rng.binomial(1, _logistic(0.3 * x)).astype(float)
    frame = pd.DataFrame({"y": y, "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamfit.fit(frame, "y ~ x", family="binomial")
    assert model.summary().basis_checks == []
    assert model.basis_check(frame) == []
