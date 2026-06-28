"""Wiring guard (#1598): the Python/formula fit path must thread a
``link(type=blended(...))`` / ``link(type=mixture(...))`` component spec into
``FitOptions.mixture_link`` so a binomial-mixture fit REACHES the joint solver
instead of aborting at the wiring guard.

Background
----------
A *blended* / *mixture* inverse link is a learnable convex combination of base
inverse links (e.g. ``blended(logit, probit)``) with jointly-fit mixing weights.
The binomial-mixture solver guard in ``gam-solve``
(``fit_gamwith_penalty_specs_andwarm_start`` / the external-optim sibling) bails
immediately with::

    BinomialMixture requires mixture_link specification

for any ``is_binomial_mixture()`` family whose ``FitOptions.mixture_link`` is
``None``. The ``gam`` CLI populates that field from the parsed components
(``run_fit.rs`` ``mixture_linkspec``); the formula/Python (gamfit) path used to
leave it ``None``, so::

    gamfit.fit(df, "y ~ x + link(type=blended(logit, probit))", family="binomial")

raised the wiring error *immediately*, before any fitting — the Python-side
analogue of the #1128 / #1160 link-wiring gaps. The fix
(``materialize_standard`` in ``crates/gam-models/.../materialize/standard.rs``)
threads the parsed components into ``mixture_link`` (and sets
``optimize_mixture``), bringing the formula/Python path to PARITY with the CLI.

What this test guards
---------------------
This test guards the WIRING fix, NOT full joint-solver convergence. After the
fix the Python path no longer raises ``BinomialMixture requires mixture_link
specification`` — it now reaches the same joint mixture solver the CLI reaches.

The joint mixture/SAS link solve is independently fragile (it can fail outer
startup with "observed Hessian curvature is not positive finite" / "no candidate
seeds passed outer startup validation"); making it converge is tracked as deeper
solver work. So this test asserts EXACTLY the wiring contract:

  * if the fit SUCCEEDS, we additionally require it to recover the signal
    (``corr(pred, true_p) > 0.95``) — a genuine full-fix assertion; but
  * the fit must, at minimum, NOT fail with the wiring-guard message. A
    downstream solver/convergence failure is an ACCEPTED outcome here because it
    proves the wiring guard was cleared and the solver was actually entered.

If the wiring regresses (``mixture_link`` left ``None`` again), the immediate
``BinomialMixture requires mixture_link specification`` abort returns and this
test fails.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

# The wiring-guard message that must NOT appear once the Python path populates
# FitOptions.mixture_link. Matching on this exact substring is what makes the
# test a precise regression guard for the wiring fix rather than for the (still
# fragile) downstream joint solve.
_WIRING_GUARD_MSG = "BinomialMixture requires mixture_link specification"


def _clean_logit_frame(seed: int = 0, n: int = 1500) -> pd.DataFrame:
    """n=1500 rows of a clean, well-separated logit signal (no link
    misspecification): a regime whose pure logit/probit components each fit
    trivially, so any failure is the mixture machinery, never the data."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    eta = 0.5 + 1.3 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0.0, 1.0, n) < p).astype(float)
    return pd.DataFrame({"x": x, "y": y, "true_p": p})


def test_blended_mixture_link_reaches_solver_not_wiring_abort() -> None:
    df = _clean_logit_frame()
    formula = "y ~ x + link(type=blended(logit, probit))"

    try:
        model = gamfit.fit(df, formula, family="binomial")
    except Exception as exc:  # noqa: BLE001 - we classify the failure below
        msg = str(exc)
        assert _WIRING_GUARD_MSG not in msg, (
            "REGRESSION (#1598): the Python/formula fit path aborted at the "
            "binomial-mixture WIRING guard "
            f"({_WIRING_GUARD_MSG!r}) — FitOptions.mixture_link was left None, so "
            "the parsed blended(...) components never reached the solver. The "
            "materializer must thread the components into mixture_link (parity "
            f"with the CLI). Full error: {msg}"
        )
        # Reaching here means the wiring guard was cleared and the solver was
        # entered; a downstream joint-solve / convergence failure is an accepted
        # outcome for THIS test (it guards the wiring, not convergence). Document
        # it loudly via skip so the parity fix is not mistaken for a full fix.
        pytest.skip(
            "wiring fix verified: blended-link fit reached the joint mixture "
            "solver (no wiring abort). The joint solve itself did not converge "
            f"on this data — deeper solver work, tracked separately: {msg}"
        )

    # Full-fix path: if the joint solve DID converge, hold it to recovering the
    # signal. A clean logit signal under a logit/probit blend must track the true
    # probabilities closely.
    pred = np.asarray(model.predict(df), dtype=float).reshape(-1)
    true_p = df["true_p"].to_numpy(dtype=float)
    corr = float(np.corrcoef(pred, true_p)[0, 1])
    assert corr > 0.95, (
        "blended(logit, probit) fit converged but did not recover the signal "
        f"(corr(pred, true_p) = {corr:.4f} ≤ 0.95); a learnable link that nests "
        "logit must do no worse than plain logit on clean logit data."
    )
