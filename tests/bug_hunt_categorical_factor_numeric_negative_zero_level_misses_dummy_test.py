"""Bug hunt #2146: a numeric ``factor(g)`` categorical main effect gated each
level dummy by raw IEEE-754 bits, so the numerically-identical group ``-0.0``
missed the interned ``0.0`` dummy and ``predict(g=-0.0)`` collapsed onto the
grand-mean intercept — silently (``check()`` reported ``ok=True``).

Root cause: ``realized_design_column`` (and the derivative / factor-by paths)
zeroed the design entry whenever ``v.to_bits() != level_bits``. ``to_bits()`` is
a *bit* identity: ``+0.0`` is ``0x0000_0000_0000_0000`` and ``-0.0`` is
``0x8000_0000_0000_0000``, so the ``-0.0`` row failed every level comparison and
kept only the intercept, even though IEEE-754 guarantees ``-0.0 == 0.0``.

This is NOT an unseen-level case (cf. #2102/#2137): ``-0.0`` *is* the seen level
``0.0``, so the contract is to reproduce group-0's prediction, not to raise and
not to average. The fix routes both the interned level set and the per-row gate
through ``gam_data::canonical_level_bits`` (``-0.0 → +0.0``), so the two spellings
of zero name one dummy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _pred(model, g_value: float) -> float:
    out = model.predict(pd.DataFrame({"g": [g_value]}))
    return float(np.asarray(out).ravel()[0])


def _make(seed: int = 20240705, n: int = 1500):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n).astype(float)  # numeric levels 0.0, 1.0, 2.0
    means = {0: 4.0, 1: -2.0, 2: 3.0}  # group 0 far from grand mean
    y = np.array([means[int(v)] for v in g]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "g": g})


def test_factor_negative_zero_level_matches_positive_zero_dummy() -> None:
    m = gamfit.fit(_make(), "y ~ factor(g)")

    assert 0.0 == -0.0  # sanity: IEEE-754 numeric equality

    pos = _pred(m, 0.0)
    neg = _pred(m, -0.0)

    # -0.0 and 0.0 name the same real number: the predictions must be identical.
    assert neg == pos, f"predict(g=-0.0)={neg} != predict(g=0.0)={pos}"

    # And the shared value must be the group-0 fitted mean (≈ 4), NOT the
    # grand-mean intercept (≈ 1.66) that the missed-dummy bug returned.
    assert abs(pos - 4.0) < 0.6, f"group-0 prediction {pos} not near 4.0"


def test_factor_negative_zero_check_is_not_silently_ok() -> None:
    """The signed-zero row is the seen level 0.0, so ``check`` should accept it
    AND the prediction must be correct — the failure mode was a silent
    ``ok=True`` masking a wrong (intercept-only) prediction."""
    m = gamfit.fit(_make(), "y ~ factor(g)")
    report = m.check(pd.DataFrame({"g": [-0.0]}))
    assert getattr(report, "ok", True) is True
    # ok=True is only honest if the prediction is actually right:
    assert _pred(m, -0.0) == _pred(m, 0.0)


def test_factor_computed_negative_zero_from_arithmetic() -> None:
    """``-0.0`` produced by ordinary float arithmetic (``-1.0 * 0.0``) at
    predict time must resolve to the same dummy as a literal ``0.0``."""
    m = gamfit.fit(_make(), "y ~ factor(g)")
    computed_neg_zero = -1.0 * 0.0
    assert np.signbit(computed_neg_zero)  # genuinely -0.0
    assert _pred(m, computed_neg_zero) == _pred(m, 0.0)


def test_factor_training_column_mixing_signed_zero_is_one_dummy() -> None:
    """A training column that mixes ``0.0`` and ``-0.0`` for the physically same
    group must intern as a SINGLE dummy, not split into two spurious levels —
    the fit-side face of the same signed-zero keying bug."""
    rng = np.random.default_rng(5)
    per = 200
    g0 = np.where(rng.random(per) < 0.5, -0.0, 0.0)  # same physical group 0
    g = np.concatenate([g0, np.full(per, 1.0), np.full(per, 2.0)])
    y = np.concatenate(
        [np.full(per, 4.0), np.full(per, -2.0), np.full(per, 3.0)]
    ) + rng.normal(0, 0.2, g.size)
    m = gamfit.fit(pd.DataFrame({"y": y, "g": g}), "y ~ factor(g)")

    assert _pred(m, -0.0) == _pred(m, 0.0)
    assert abs(_pred(m, 0.0) - 4.0) < 0.6, "group-0 mean not recovered from mixed ±0.0"
