"""Bug hunt #1602: the fitted affine design must reproduce the engine's
``linear_predictor`` for every family — not only the Gaussian identity link.

Root cause (see ``scratchpad/issue_1602_coef_predictor.md``): the exported
coefficients (``summary().coefficients``) are the penalized MLE / posterior mode
``β̂``, so ``offset + matrix @ coefficients == η̂`` exactly and linearly.
But for curved inverse links (``prediction_uses_posterior_mean() == true``) the
wiggle-free posterior-mean predict path reported a **bias-corrected** linear
predictor ``η̂_BC = X @ (β̂ + b̂)`` with ``b̂ = H⁻¹S(β̂−μ)`` the O(1/n)
frequentist bias-correction vector (``crates/gam-predict/src/standard.rs`` →
``predict_gam_posterior_mean_from_backendwith_bc``). That broke the documented
``docs/predictions.md`` affine-design contract and the
``offset + posterior.samples @ X.T`` recipe — by
exactly ``X @ b̂`` (1.5–4 % of the lp range) for Poisson/Gamma ``log`` and
binomial ``logit``/``probit``, while staying exact for the identity link (whose
plug-in path deliberately sets ``apply_bias_correction: false``).

The fix drops the bias correction from the reported ``linear_predictor`` so the
identity holds for all links (and so the posterior-mean integral is correctly
centered on the conditional posterior mode ``X β̂``).

Angles covered:

* **The defining identity, all links** — ``max|design_matrix @ coef −
  linear_predictor|`` must be at the floating-point floor relative to the
  linear-predictor range for Poisson, Gamma, binomial-logit, binomial-probit
  (the four failing cases), with Gaussian-identity as the always-passing control.
* **Documented ``samples @ X.T`` recipe** — the mean of ``posterior.samples @
  X.T`` is centered at the reported ``linear_predictor`` (same point estimate),
  not offset by a hidden bias-correction shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _make(family: str, link: str | None, seed: int = 0, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = 1.2 * np.sin(2 * np.pi * x)
    if family == "gaussian":
        y = eta + rng.normal(0, 0.2, n)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.5 + eta)).astype(float)
    elif family == "gamma":
        y = rng.gamma(4.0, np.exp(0.5 + eta) / 4.0)
    elif family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(family)
    return pd.DataFrame({"x": x, "y": y})


def _fit(df: pd.DataFrame, family: str, link: str | None):
    kw = {"family": family}
    if link is not None:
        kw["link"] = link
    return gamfit.fit(df, "y ~ s(x)", **kw)


def _coef(model) -> np.ndarray:
    return np.asarray(
        [row["estimate"] for row in model.summary().coefficients], dtype=float
    )


def _engine_eta(model, df: pd.DataFrame) -> np.ndarray:
    out = model.predict(df, interval=0.95, return_type="dict")
    return np.asarray(out["linear_predictor"], dtype=float)


CASES = [
    ("gaussian", None),
    ("poisson", None),
    ("gamma", None),
    ("binomial", "logit"),
    ("binomial", "probit"),
]


def test_design_matrix_coef_reproduces_linear_predictor_all_links() -> None:
    """``design_matrix @ coef == linear_predictor`` to fp floor for every link.

    Before the fix the four non-identity links drifted 1.5–4 % of the lp range;
    Gaussian identity was always exact.
    """
    for family, link in CASES:
        df = _make(family, link)
        model = _fit(df, family, link)
        affine = model.design_matrix(df)
        x_design = np.asarray(affine.matrix, dtype=float)
        coef = _coef(model)
        eta_engine = _engine_eta(model, df)

        assert isinstance(affine, gamfit.AffineDesign)
        assert affine.coefficient_frame == "full"
        assert affine.coefficient_slice == slice(0, coef.shape[0])
        assert x_design.shape[1] == coef.shape[0], (
            f"{family}/{link}: design has {x_design.shape[1]} cols but "
            f"{coef.shape[0]} coefficients"
        )
        np.testing.assert_allclose(affine.coefficients, coef, rtol=0.0, atol=0.0)
        eta_from_coef = affine.offset + x_design @ affine.coefficients
        lp_range = float(np.ptp(eta_engine))
        max_abs = float(np.max(np.abs(eta_from_coef - eta_engine)))
        rel = max_abs / lp_range if lp_range > 0 else max_abs
        # Pure linear composition X@β̂ vs the engine's reported η — must agree to
        # the floating-point floor (matmul round-off), << the old 1.5–4 % gap.
        assert rel < 1e-9, (
            f"{family}/{link}: max|design@coef - linear_predictor|={max_abs:.3e} "
            f"({rel:.2%} of lp range {lp_range:.3f}) — coef no longer reproduces "
            f"the engine linear predictor"
        )


def test_posterior_samples_recipe_centered_on_linear_predictor() -> None:
    """The documented ``posterior.samples @ X.T`` recipe is centered on the
    reported ``linear_predictor`` (same point estimate), for a curved link."""
    df = _make("poisson", None)
    model = _fit(df, "poisson", None)
    affine = model.design_matrix(df)
    x_design = np.asarray(affine.matrix, dtype=float)
    eta_engine = _engine_eta(model, df)

    posterior = model.sample(df, samples=4000, seed=11)
    samples = np.asarray(posterior.samples, dtype=float)  # (n_draws, n_coef)
    eta_draws = affine.offset + samples @ x_design.T       # (n_draws, n_rows)
    eta_mean = eta_draws.mean(axis=0)

    lp_range = float(np.ptp(eta_engine))
    # Monte-Carlo mean of the recipe must track the reported linear predictor;
    # the residual is sampling noise (~1/sqrt(n_draws)), not the deterministic
    # bias-correction shift (~3 % of lp range) that this issue removed.
    max_abs = float(np.max(np.abs(eta_mean - eta_engine)))
    assert max_abs / lp_range < 0.02, (
        f"posterior.samples @ X.T mean drifts {max_abs:.3e} "
        f"({max_abs / lp_range:.2%} of lp range) from linear_predictor"
    )
