"""Bug hunt #2137 (sibling of #2102): ``factor(g)`` — a FIXED categorical main
effect — silently mapped an out-of-vocabulary level to the unweighted
across-level average at ``predict``, and ``Model.check(...)`` reported
``ok=True``.

Root cause: ``factor(g)`` shared the ``group()``/``re()`` parse arm in
``formula_dsl`` and was lowered as a *lenient* random effect
(``lenient_unseen: true``), so an unseen level at predict collapsed onto the
factor's sum-to-zero centering point instead of raising. ``factor()`` is a fixed
categorical factor (R ``factor()`` / patsy ``C()`` convention), not a
random-effect alias: like a bare ``+ g`` categorical main effect (#2102), an
unseen level is a schema mismatch that must raise. The unseen policy is now
carried on ``ParsedTerm::RandomEffect`` and set by the wrapper the user wrote —
``factor()`` strict, ``group()``/``re()``/``s(bs="re")`` lenient — so seen-level
fits stay identical while only the held-out-level policy differs.

``docs/exceptions.md`` requires that an unseen categorical level either raise
from ``predict`` or be reported by ``check`` as a non-``ok`` issue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit


def _make(seed: int = 2) -> pd.DataFrame:
    """Unbalanced 3-level factor so the fabricated across-level average is
    distinguishable from any row-weighted mean (the #2137 repro)."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], size=2000, p=[0.8, 0.1, 0.1])
    eff = {"a": 0.0, "b": 6.0, "c": 12.0}
    y = np.array([eff[gi] for gi in g]) + rng.normal(0, 0.1, g.size)
    return pd.DataFrame({"g": g, "y": y})


def test_factor_wrapper_predict_raises_on_unseen_level() -> None:
    """``predict`` on an out-of-vocabulary level of a fixed ``factor(g)`` must
    raise — NOT silently return the centering point. The strict categorical
    encode raises the same ``GamError`` (``unseen level '…' in categorical
    column '…'``) that the bare ``+ g`` path has raised since #2102, so the two
    fixed-factor spellings are now consistent."""
    m = gamfit.fit(_make(), "y ~ factor(g)")
    with pytest.raises(gamfit.GamError) as exc:
        m.predict(pd.DataFrame({"g": ["z"]}))
    assert "unseen level" in str(exc.value)
    assert "g" in str(exc.value)


def test_factor_wrapper_matches_bare_categorical_on_unseen_level() -> None:
    """The crux of #2137: ``factor(g)`` and a bare ``+ g`` fixed categorical main
    effect must handle an unseen level identically — both raise from ``predict``
    and both flag it in ``check``. Before the fix ``factor(g)`` silently averaged
    while bare ``+ g`` (fixed by #2102) correctly rejected."""
    df = _make()
    unseen = pd.DataFrame({"g": ["z"]})
    for formula in ("y ~ g", "y ~ factor(g)"):
        m = gamfit.fit(df, formula)
        with pytest.raises(gamfit.GamError):
            m.predict(unseen)
        assert m.check(unseen).ok is False, f"{formula} must flag the unseen level"


def test_factor_wrapper_check_flags_unseen_level() -> None:
    """``Model.check`` must report the unseen level as a non-``ok`` issue rather
    than the silent ``ok=True`` the lenient-random-effect lowering returned."""
    m = gamfit.fit(_make(), "y ~ factor(g)")
    report = m.check(pd.DataFrame({"g": ["z"]}))
    assert report.ok is False, "check() must flag an unseen fixed-factor level"


def test_group_wrapper_stays_lenient_on_unseen_level() -> None:
    """The scoping guard: an EXPLICIT random effect ``group(g)`` deliberately
    tolerates a held-out group (shrunk to the population mean). It must NOT
    inherit ``factor()``'s strictness — predict succeeds and check stays ``ok``."""
    df = _make()
    m = gamfit.fit(df, "y ~ group(g)")

    # Lenient: no raise on the unseen level.
    out = m.predict(pd.DataFrame({"g": ["z"]}))
    pred = float(np.asarray(out).ravel()[0])
    assert np.isfinite(pred)

    # A held-out group carries zero random-effect deviation, so it lands on the
    # sum-to-zero centering point — the across-level average of the seen effects
    # {a:0, b:6, c:12}, i.e. ≈ 6. This "silently return the centering point"
    # behavior is exactly what #2137 forbids for a FIXED factor(g) but is the
    # documented, deliberate policy for a genuine random effect.
    assert abs(pred - 6.0) < 1.5, (
        f"held-out random-effect group prediction {pred} should sit on the "
        f"across-level centering point (~6)"
    )
    assert m.check(pd.DataFrame({"g": ["z"]})).ok is True


def test_factor_and_group_agree_on_seen_levels() -> None:
    """The fix must ONLY change the unseen-level policy: on seen levels
    ``factor(g)`` and ``group(g)`` share the penalized-categorical block and so
    must predict identically."""
    df = _make()
    m_factor = gamfit.fit(df, "y ~ factor(g)")
    m_group = gamfit.fit(df, "y ~ group(g)")

    seen = pd.DataFrame({"g": ["a", "b", "c"]})
    pf = np.asarray(m_factor.predict(seen)).ravel()
    pg = np.asarray(m_group.predict(seen)).ravel()
    np.testing.assert_allclose(pf, pg, rtol=1e-6, atol=1e-6)
