"""Bug hunt: ``link(type=flexible(loglog))`` / ``flexible(cauchit)`` on a
binomial fit is accepted by the formula/validation layer, then crashes deep in
the solver with an opaque ``IntegrationError`` — while the sibling
``flexible(cloglog)`` / ``flexible(logit)`` / ``flexible(probit)`` fit fine.

``loglog`` and ``cauchit`` are advertised, implemented binomial links (their
plain forms were un-gated in #2104, and their inverse-link derivative jets live
in ``crates/gam-solve/src/mixture_link.rs``). Wrapping one in ``flexible(...)``
should therefore either fit (the joint score-warp / "wiggle" solver handles it)
or be rejected *up front* with a clear link/configuration error. Instead:

* the parse-time gate ``linkname_supports_joint_wiggle``
  (``crates/gam-terms/src/inference/formula_dsl.rs:1499``) admits every link
  except ``sas``/``beta-logistic``, so ``flexible(loglog)`` / ``flexible(cauchit)``
  pass validation, but
* the actual wiggle solver only implements ``{logit, probit, cloglog}``
  (``binomial_inverse_link_supports_joint_wiggle``, same file ~line 1558,
  enforced in ``crates/gam-models/src/gamlss/builders.rs``), so the fit then
  aborts far downstream with
  ``IntegrationError: optimize_external_design requires a supported standard
  GLM family/link``.

The three gates that decide "is this link supported" disagree, and the
permissive one runs first — so an advertised link config is accepted and then
blows up with an internal integration error instead of being handled cleanly.
A dedicated ``UnsupportedLinkError`` type even exists for exactly this situation
but is not used here.

This test pins the graceful-handling contract in a way that is robust to either
reasonable fix (teach the wiggle solver these links, or reject them up front):
``flexible(cloglog)`` must keep fitting (positive control / setup sanity), and
``flexible(loglog)`` / ``flexible(cauchit)`` must EITHER fit and predict a valid
probability OR raise a clean link/configuration rejection — never the opaque
``IntegrationError`` from deep inside the solver.

Related: #2104 (plain loglog/cauchit binomial links were wrongly rejected — the
opposite polarity of this gate mismatch).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

# Reasonable, non-opaque outcomes if a link truly cannot be fit with a flexible
# warp: an up-front, actionable link/configuration rejection.
_CLEAN_REJECTIONS = (
    gamfit.UnsupportedLinkError,
    gamfit.InvalidConfigurationError,
    gamfit.InvalidSpecificationError,
    gamfit.FormulaError,
)


def _binomial_frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    prob = 1.0 / (1.0 + np.exp(-(0.5 * x)))
    y = (rng.uniform(size=n) < prob).astype(float)
    return pd.DataFrame({"x": x, "y": y})


def test_flexible_cloglog_positive_control_fits() -> None:
    # Setup sanity: the flexible-link machinery works for a supported link.
    df = _binomial_frame()
    model = gamfit.fit(df, "y ~ x + link(type=flexible(cloglog))", family="binomial")
    mean = np.asarray(
        model.predict(pd.DataFrame({"x": [0.0]}), return_type="pandas")["mean"],
        dtype=float,
    )
    assert np.all((mean >= 0.0) & (mean <= 1.0))


@pytest.mark.parametrize("link", ["flexible(loglog)", "flexible(cauchit)"])
def test_flexible_advertised_link_is_handled_gracefully(link: str) -> None:
    df = _binomial_frame()
    try:
        model = gamfit.fit(df, f"y ~ x + link(type={link})", family="binomial")
    except _CLEAN_REJECTIONS:
        # Acceptable: a clear, up-front "this link isn't supported here" error.
        return
    except Exception as exc:  # noqa: BLE001 - we specifically indict the opaque crash
        pytest.fail(
            f"binomial fit with link={link} crashed with "
            f"{type(exc).__name__} deep in the solver instead of fitting or "
            f"raising a clean link/configuration error: {exc}"
        )

    # If it did fit, it must produce a valid probability prediction.
    mean = np.asarray(
        model.predict(pd.DataFrame({"x": [0.0]}), return_type="pandas")["mean"],
        dtype=float,
    )
    assert np.all((mean >= 0.0) & (mean <= 1.0))
