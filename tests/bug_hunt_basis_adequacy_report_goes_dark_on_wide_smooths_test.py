"""Bug hunt: the #2774 basis-adequacy check never runs on a 1-D ``s()`` whose
basis has 18 or more coefficient columns. Every such fit reports
``provenance='statistic_unavailable'`` and ``p_value=None`` -- on every dataset,
at every ``n``, adequate or not.

``docs/diagnostics.md:70``:

    That is what `basis_check` measures, and what every fit now measures for
    itself

and ``docs/diagnostics.md:100-101``:

    `p_value` is present exactly when a test ran, so "adequate" and
    "not measured" are never confusable.

Measured (``y = sin(30*pi*x) + N(0, 0.1)``, a truth no basis in this ladder can
represent; ``summary().basis_checks[0]``):

    basis_dim   enrichment_dim   enrichment_rank   p_value
        9             36              8            9.6e-05
       11             44              6            7.9e-03
       13             52              5            1.3e-02
       15             60              3            6.3e-01
       17             68              1            1.8e-01
       19             76              0            None   <- statistic_unavailable
       24             96              0            None
       29            116              0            None
       39            156              0            None

The enrichment gets WIDER at every rung (36 -> 156 columns) while the reference
d.f. the test manages to use falls MONOTONICALLY to zero. Past ``basis_dim=18``
nothing survives and the report goes dark. The cliff does not move with ``n``
(measured identical at n=1000, 4000, 8000) or with the data (measured identical
on a ``sin(2*pi*x)`` truth the basis represents perfectly), so it is a property
of the width alone.

Mechanism. ``crates/gam-terms/src/inference/basis_adequacy.rs:145``:

    const ESTIMABLE_DIRECTION_FLOOR: f64 = 1.0e-9;

is applied at lines 336-346 to the eigenvalues of the RESIDUALIZED enrichment
Gram ``V = Z~^T W_F Z~`` (``Z~ = Z - X (X^T W_H X)^-1 X^T W_H Z``), but scaled by
``energy_scale``, the largest diagonal of the RAW ``Z^T W_F Z`` (line ~136: "The
floor is taken relative to the enrichment's own weighted energy scale ... rather
than to `V`'s largest eigenvalue"). Those two quantities move in opposite
directions as the enrichment widens: ``energy_scale`` grows with the center count
while the residual spectrum of a smooth 1-D radial kernel decays geometrically
once the design already spans its coarse directions. Reproducing the same linear
algebra outside the engine on the ladder above (design from
``Model.design_matrix``, enrichment from ``gamfit.basis.duchon_basis`` at
``enrichment_width`` equal-mass centers) recovers the shipped ranks exactly and
shows the crossing:

    basis_dim=9   energy=1.3e+04  floor=1.3e-05  lambda_max(V)=3.3e-03  rank 7
    basis_dim=14  energy=1.9e+04  floor=1.9e-05  lambda_max(V)=1.1e-04  rank 4
    basis_dim=19  energy=2.6e+04  floor=2.6e-05  lambda_max(V)=1.0e-05  rank 0
    basis_dim=24  energy=3.2e+04  floor=3.2e-05  lambda_max(V)=1.8e-06  rank 0

At ``basis_dim=19`` the LARGEST eigenvalue of the residualized Gram is already
below the floor, so ``rank == 0`` (line 347) returns ``None`` and the driver
records ``StatisticUnavailable``
(``crates/gam-models/src/fit_orchestration/drivers/basis_adequacy.rs:584``).
Nothing numerically degenerate has happened: the residual spectrum decays
smoothly through the floor with no gap, and an ordinary rank tolerance taken
against ``V``'s own largest eigenvalue (``lambda_max * q * eps``) retains
``q - rank(X)`` directions at every rung.

Observed: the diagnostic that "every fit now measures for itself" silently does
not run for any 1-D smooth with 18+ columns.

Expected: a 1-D smooth wide enough to be worth checking still gets a verdict.
The enrichment is built four times wider than the realized basis precisely so
there is new resolution to test against; the reference d.f. must reflect that
width instead of collapsing as it grows.

Related: the same truncation costs the test its power at the rungs where it does
still report (``bug_hunt_basis_adequacy_reference_df_collapses_as_the_enrichment_widens_test.py``).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

#: Coefficient widths at and above the observed cliff. ``s(x, k=K)`` realizes
#: ``K - 1`` columns after the sum-to-zero constraint.
_WIDE_K = [20, 25, 30, 40]


def _frame(n: int, cycles: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    truth = np.sin(2.0 * cycles * np.pi * x)
    return {"x": x, "y": truth + 0.1 * rng.standard_normal(n)}


def _check_row(data: dict[str, Any], formula: str) -> dict[str, Any]:
    model = gamfit.fit(data, formula, family="gaussian")
    rows = model.summary().basis_checks
    assert len(rows) == 1, f"{formula}: expected one basis-adequacy row, got {rows!r}"
    return dict(rows[0])


def test_control_a_narrow_one_dimensional_smooth_still_gets_a_verdict() -> None:
    """Green today: the report works at the default width, so the failures below
    are about width, not about the fixture or the surface."""
    row = _check_row(_frame(4000, 15.0, 11), "y ~ s(x, k=10)")
    assert row["basis_dim"] == 9
    assert row["provenance"] == "radial_enrichment"
    assert row["p_value"] is not None


def test_control_a_two_dimensional_smooth_gets_a_verdict_at_a_far_greater_width() -> None:
    """Green today: ``te(x, z)`` realizes 48 columns -- more than twice the 1-D
    cliff -- and still reports. The cliff is not a budget or a cost cap."""
    rng = np.random.default_rng(21)
    n = 3000
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    data = {
        "x": x,
        "z": z,
        "y": np.sin(8.0 * np.pi * x) + np.cos(6.0 * np.pi * z) + 0.3 * rng.standard_normal(n),
    }
    row = _check_row(data, "y ~ te(x, z)")
    assert row["basis_dim"] > max(_WIDE_K)
    assert row["provenance"] == "radial_enrichment"
    assert row["p_value"] is not None


@pytest.mark.parametrize("k", _WIDE_K)
def test_wide_one_dimensional_smooth_reports_a_verdict_on_inadequate_data(k: int) -> None:
    row = _check_row(_frame(4000, 15.0, 11), f"y ~ s(x, k={k})")
    assert row["provenance"] == "radial_enrichment", (
        f"s(x, k={k}) realized {row['basis_dim']} columns and the basis-adequacy "
        f"check never ran: provenance={row['provenance']!r}, "
        f"enrichment_dim={row['enrichment_dim']!r}, "
        f"enrichment_rank={row['enrichment_rank']!r}"
    )
    assert row["p_value"] is not None


@pytest.mark.parametrize("k", _WIDE_K)
def test_wide_one_dimensional_smooth_reports_a_verdict_on_adequate_data(k: int) -> None:
    """The report must be able to say "adequate", not only "inadequate": a
    low-frequency truth that these bases represent exactly is dark too."""
    row = _check_row(_frame(4000, 1.0, 11), f"y ~ s(x, k={k})")
    assert row["provenance"] == "radial_enrichment", (
        f"s(x, k={k}) on a truth its basis represents exactly still produced no "
        f"verdict: provenance={row['provenance']!r}"
    )
    assert row["p_value"] is not None


@pytest.mark.parametrize("n", [1000, 8000])
def test_the_cliff_does_not_move_with_the_row_count(n: int) -> None:
    """Power grows linearly in ``n``; a missing verdict that is identical at
    n=1000 and n=8000 is not a power problem."""
    row = _check_row(_frame(n, 15.0, 12), "y ~ s(x, k=25)")
    assert row["provenance"] == "radial_enrichment", (
        f"n={n}: s(x, k=25) produced no verdict ({row['provenance']!r})"
    )
    assert row["p_value"] is not None
