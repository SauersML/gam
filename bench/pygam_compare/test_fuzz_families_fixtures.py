"""Fixtures from the family/link convergence fuzzer.

Each cell below failed before its root-cause fix and must now be clean under
:func:`failure_cause`:

* gamma(inverse): the dispersion profile evaluated the shape and phi at
  ``exp(eta)`` whatever the link, so on the inverse link the fitted scale sat
  hundreds of standard errors off the truth.
* binomial-trials: the fully-normalized log-likelihood required ``fl(w * y)``
  to be an exact integer, which most stored proportions ``k / w`` miss by an
  ulp (``439 / 1000 * 1000 = 439.00000000000006``), so every
  almost-deterministic binomial refused. main now evaluates the continuous
  normalizer ``ln C(w, w y)`` for any finite positive weight; the cells stay
  as its regression fixtures.
The inverse-gaussian canonical-link fixes on this branch (a dimensionless
P-IRLS KKT certificate, working-weight seed and ``rho`` domain, and a
resolution-relative penalty eigenvalue floor) have their regression tests in
the Rust crates; end-to-end equivariance under ``y -> c y`` is still blocked by
the lambda-search dispersion frozen at the unit-carrying ``rho = 0`` anchor,
which the dispersion-estimation lanes own.

A failing cell reruns in isolation with
``python worker.py gamfit FAMILY N DESIGN SEED``.
"""

from __future__ import annotations

import pytest

from .fuzz_families import failure_cause, fuzz_design, run

FIXTURES: tuple[tuple[str, int, str, int], ...] = (
    ("gamma(inverse)", 500, "base", 0),
    ("binomial-trials(logit)", 50, "lowdisp", 0),
    ("binomial-trials(cloglog)", 500, "lowdisp", 0),
)


@pytest.mark.parametrize(("label", "n", "regime", "seed"), FIXTURES)
def test_fuzz_fixture_is_clean(label: str, n: int, regime: str, seed: int) -> None:
    design = fuzz_design(regime)
    record = run(label, n, design, seed) | {
        "family": label,
        "n": n,
        "design": design,
        "seed": seed,
    }
    assert failure_cause(record) is None, record.get("error_head")

