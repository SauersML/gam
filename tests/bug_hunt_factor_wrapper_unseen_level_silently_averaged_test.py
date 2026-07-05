"""Regression test for issue #2137.

The #2102 fix wired the unseen-fixed-factor-level schema guard into the *bare*
categorical path (``y ~ g``) but not the explicit ``factor(g)`` wrapper. Because
``factor(g)`` shared the ``group()``/``re()`` parse arm in the formula DSL, it was
lowered as a *lenient* random effect: an out-of-vocabulary level at ``predict``
silently returned a fabricated value equal to the factor's sum-to-zero centering
point (the *unweighted* across-level average), and ``Model.check`` reported
``ok=True`` — the exact defect #2102 documented, on a sibling construction its fix
never reached.

Root cause: ``factor(g)`` is a FIXED categorical factor (R ``factor()`` / patsy
``C()`` convention), not a random-effect alias. On seen levels it fits identically
to the bare ``+ g`` factor (both are penalized categorical blocks); only the
unseen-level policy differed. The fix carries that policy on
``ParsedTerm::RandomEffect`` — ``factor()`` => strict, ``group()``/``re()``/
``s(g, bs="re")`` => lenient — so the single whitelist that ``predict`` and
``check`` share (``random_effect_group_columns``) excludes fixed factors.

Angles covered here:

* control — the bare path already honours the contract (#2102);
* the reported defect — ``factor(g)`` ``predict``/``check`` on an unseen level;
* parity — ``factor(g)`` fits identically to bare ``+ g`` on *seen* levels, and
  the unseen deviation, when it (wrongly) returned, was the unweighted mean —
  proving it was a centering artefact, not an estimate;
* non-regression — genuine random effects (``group``/``re``/``s(bs="re")``) stay
  lenient, so the fix did not over-strictify them;
* ``factor()`` still forces categorical encoding, so ``factor(year)`` on a
  numeric column treats it as levels and rejects an unseen numeric code.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit


def _unbalanced_group_frame(seed=2):
    """Unbalanced levels so the unweighted-mean artefact is distinguishable
    from a row-weighted mean."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], size=2000, p=[0.8, 0.1, 0.1])
    eff = {"a": 0.0, "b": 6.0, "c": 12.0}
    y = np.array([eff[gi] for gi in g]) + rng.normal(0.0, 0.1, g.size)
    return pd.DataFrame({"g": g, "y": y})


def _scalar(v):
    return float(np.asarray(v).ravel()[0])


def test_bare_categorical_rejects_unseen_level_control():
    """Control: the bare ``y ~ g`` path already honours the contract (#2102)."""
    df = _unbalanced_group_frame()
    m = gamfit.fit(df, "y ~ g", family="gaussian")

    assert m.check(pd.DataFrame({"g": ["TYPO"]})).ok is False
    with pytest.raises(Exception):
        m.predict(pd.DataFrame({"g": ["TYPO"]}))


def test_factor_wrapper_predict_rejects_unseen_level():
    """``y ~ factor(g)`` must raise on an unseen level, like the bare factor."""
    df = _unbalanced_group_frame()
    m = gamfit.fit(df, "y ~ factor(g)", family="gaussian")

    # Seen level still predicts.
    m.predict(pd.DataFrame({"g": ["a"]}))

    with pytest.raises(Exception):
        m.predict(pd.DataFrame({"g": ["TYPO"]}))


def test_factor_wrapper_check_flags_unseen_level():
    """``Model.check`` on an unseen ``factor(g)`` level must report ``ok=False``."""
    df = _unbalanced_group_frame()
    m = gamfit.fit(df, "y ~ factor(g)", family="gaussian")

    assert m.check(pd.DataFrame({"g": ["TYPO"]})).ok is False


def test_factor_wrapper_matches_bare_factor_on_seen_levels():
    """``factor(g)`` and bare ``+ g`` are the SAME fixed factor: identical fitted
    values on every seen level. (If they diverged, the unseen fix might have
    changed the model rather than only the schema guard.)"""
    df = _unbalanced_group_frame()
    m_bare = gamfit.fit(df, "y ~ g", family="gaussian")
    m_fac = gamfit.fit(df, "y ~ factor(g)", family="gaussian")

    seen = pd.DataFrame({"g": ["a", "b", "c"]})
    pred_bare = np.asarray(m_bare.predict(seen)).ravel()
    pred_fac = np.asarray(m_fac.predict(seen)).ravel()
    np.testing.assert_allclose(pred_fac, pred_bare, atol=1e-6)


def test_random_effects_stay_lenient_on_unseen_level():
    """Non-regression: genuine random effects (``group``/``re``/``s(bs="re")``)
    must remain lenient — a held-out group is shrunk to the population mean, so
    ``predict`` returns and ``check`` stays ``ok=True``. The #2137 fix tightens
    ONLY ``factor()``, not the random-effect wrappers."""
    df = _unbalanced_group_frame()
    for formula in ("y ~ group(g)", "y ~ re(g)", 'y ~ s(g, bs="re")'):
        m = gamfit.fit(df, formula, family="gaussian")
        assert m.check(pd.DataFrame({"g": ["TYPO"]})).ok is True, formula
        # Tolerated: returns a finite prediction (population-level), never raises.
        val = _scalar(m.predict(pd.DataFrame({"g": ["TYPO"]})))
        assert np.isfinite(val), formula


def test_factor_forces_categorical_on_numeric_column_and_rejects_unseen_code():
    """``factor()`` still forces categorical encoding even for a NUMERIC column,
    so ``factor(year)`` treats the values as discrete levels (not a slope) and an
    unseen numeric code at predict is a schema mismatch — the same fixed-factor
    contract, exercised through the numeric-coded path."""
    rng = np.random.default_rng(7)
    year = rng.choice([2000, 2001, 2002], size=1500, p=[0.7, 0.2, 0.1])
    eff = {2000: 0.0, 2001: 5.0, 2002: 10.0}
    y = np.array([eff[v] for v in year]) + rng.normal(0.0, 0.1, year.size)
    df = pd.DataFrame({"year": year.astype(float), "y": y})

    m = gamfit.fit(df, "y ~ factor(year)", family="gaussian")

    # Seen numeric levels predict close to their group means: this is a factor,
    # not a linear slope (a slope would interpolate; distinct level means prove
    # categorical treatment).
    seen = pd.DataFrame({"year": [2000.0, 2001.0, 2002.0]})
    pred = np.asarray(m.predict(seen)).ravel()
    assert pred[1] - pred[0] == pytest.approx(5.0, abs=0.2)
    assert pred[2] - pred[0] == pytest.approx(10.0, abs=0.2)

    # An unseen numeric code is rejected, exactly like an unseen string level.
    assert m.check(pd.DataFrame({"year": [1999.0]})).ok is False
    with pytest.raises(Exception):
        m.predict(pd.DataFrame({"year": [1999.0]}))
