"""Bug hunt #2145: a numeric random-effect group column keyed by raw IEEE-754
bits splits ``0.0`` and ``-0.0`` into two groups, so ``predict(g=-0.0)`` drops
the random effect and collapses onto the population mean.

Root cause: ``build_random_effect_block`` interned each observed level with
``f64::to_bits()`` and looked each prediction row up the same way. ``to_bits()``
is a *bit* identity, not the *numeric* equality IEEE-754 defines: ``+0.0`` is
``0x0000_0000_0000_0000`` and ``-0.0`` is ``0x8000_0000_0000_0000``, yet
``0.0 == -0.0``. A ``-0.0`` prediction row therefore matched no fitted column,
the random effect contributed zero, and the prediction silently returned the
fixed-effect intercept (≈ grand mean) instead of the learned group-0 deviation.

Fix: intern and look up every level through ``gam_data::canonical_level_bits``,
which maps ``-0.0 → +0.0`` (and any NaN payload to one key) while leaving every
ordinary finite value bit-stable. ``predict(g=-0.0) == predict(g=0.0)`` then
holds exactly, because ``-0.0`` and ``0.0`` name the same group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _pred(model, g_value: float) -> float:
    out = model.predict(pd.DataFrame({"g": [g_value]}))
    return float(np.asarray(out).ravel()[0])


def test_random_effect_negative_zero_matches_positive_zero_group() -> None:
    rng = np.random.default_rng(20240705)
    n_groups, per = 12, 40
    dev = np.linspace(-3.0, 3.0, n_groups)
    dev[0] = 3.0  # group 0 sits +3 above the grand mean
    g = np.repeat(np.arange(n_groups, dtype=float), per)  # FLOAT group column
    y = 5.0 + dev[g.astype(int)] + rng.normal(0, 0.25, g.size)

    m = gamfit.fit(pd.DataFrame({"y": y, "g": g}), 'y ~ s(g, bs="re")')

    assert 0.0 == -0.0  # sanity: IEEE-754 numeric equality

    pos = _pred(m, 0.0)
    neg = _pred(m, -0.0)

    # -0.0 and 0.0 name the same real number, so the two predictions must be
    # bit-for-bit identical — not merely close.
    assert neg == pos, f"predict(g=-0.0)={neg} != predict(g=0.0)={pos}"

    # And the shared prediction must be the learned group-0 deviation (≈ 8),
    # NOT the population mean (≈ 5.5) that the dropped-effect bug returned.
    assert abs(pos - 8.0) < 0.6, f"group-0 prediction {pos} not near 8.0"


def test_random_effect_computed_negative_zero_from_arithmetic() -> None:
    """``-0.0`` arising from ordinary float arithmetic (``-1.0 * 0.0``) at
    predict time must resolve to the same group as a literal ``0.0``."""
    rng = np.random.default_rng(11)
    per = 50
    g = np.repeat(np.array([0.0, 1.0, 2.0]), per)
    means = {0.0: 6.0, 1.0: -1.0, 2.0: 2.0}
    y = np.array([means[v] for v in g]) + rng.normal(0, 0.2, g.size)
    m = gamfit.fit(pd.DataFrame({"y": y, "g": g}), 'y ~ s(g, bs="re")')

    computed_neg_zero = -1.0 * 0.0  # bit-pattern -0.0
    assert np.signbit(computed_neg_zero)  # really is -0.0
    assert _pred(m, computed_neg_zero) == _pred(m, 0.0)


def test_random_effect_training_column_mixing_signed_zero_is_one_group() -> None:
    """A training column that mixes ``0.0`` and ``-0.0`` for the physically same
    group must fit as a single group, not two spurious levels."""
    rng = np.random.default_rng(7)
    per = 60
    # Half of group-0's rows carry -0.0, half carry +0.0 — same physical group.
    g0 = np.where(rng.random(per) < 0.5, -0.0, 0.0)
    g = np.concatenate([g0, np.full(per, 1.0), np.full(per, 2.0)])
    base = {0.0: 5.0, 1.0: -2.0, 2.0: 3.0}
    y = np.concatenate(
        [
            np.full(per, base[0.0]),
            np.full(per, base[1.0]),
            np.full(per, base[2.0]),
        ]
    ) + rng.normal(0, 0.2, g.size)
    m = gamfit.fit(pd.DataFrame({"y": y, "g": g}), 'y ~ s(g, bs="re")')

    # Both signed-zero spellings predict the same fitted group-0 mean.
    assert _pred(m, -0.0) == _pred(m, 0.0)
    assert abs(_pred(m, 0.0) - 5.0) < 0.8
