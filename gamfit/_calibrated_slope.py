"""Calibrated marginal-slope chain: the CTN Stage-1 → marginal-slope Stage-2
cross-fitted, Neyman-orthogonal score-calibration workflow (closes #461).

This module is a *thin* marshaller. It carries the Stage-1 conditional
transformation-normal (CTN) recipe from a user's :func:`gamfit.fit` call down
to the Rust core as a single ``ctn_stage1`` JSON object on the existing
``fit_table`` payload. ALL numerics — fitting Stage-1 per fold, computing the
out-of-fold latent score ``z``, the score-influence Jacobian ``∂z/∂θ₁``, and the
absorbed leakage-projection block — live in Rust
(``gam::solver::workflow::crossfit_score_calibration`` and
``gam::families::marginal_slope_orthogonal``). Nothing here computes a transform,
a fold split, or a Jacobian; supplying the recipe is the sole, magic-by-default
auto-enable signal for the orthogonalized path (design §5).

The Stage-2 model (``family="bernoulli-marginal-slope"`` or a survival
marginal-slope ``survival_likelihood="marginal-slope"``) is requested exactly as
before. When the recipe is present, the Rust materializer refits the CTN on each
fold's complement, evaluates the held-out ``z`` and ``J``, replaces the in-sample
score with its out-of-fold value, and installs the realized leakage-projection
block — so the fitted slope surface ``β(x)`` is insensitive to Stage-1
calibration error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CtnStage1:
    """Stage-1 conditional transformation-normal recipe for a calibrated
    marginal-slope chain.

    Passing this to :func:`gamfit.fit` on a marginal-slope model (Bernoulli or
    survival) auto-enables the cross-fitted, Neyman-orthogonal score
    calibration of #461: the Rust core fits the CTN ``h(Y|x) ~ N(0,1)`` per
    fold, produces an out-of-fold latent score ``z = Φ⁻¹(PIT)`` that replaces the
    raw in-sample score, and absorbs the Stage-1 score-influence directions so
    the estimated slope surface ``β(x)`` does not inherit Stage-1 miscalibration.

    There is no separate boolean to "turn on" orthogonalization — supplying the
    recipe *is* the request (magic by default). Omit it (and supply a raw
    ``z_column`` instead) to keep the legacy free-warp ``score_warp`` fallback,
    which can only defend the x-free leakage column.

    Parameters
    ----------
    response:
        Stage-1 response column name — the ``y`` the CTN transforms into the
        latent score. This is the column you would have fitted a standalone
        ``transformation_normal=True`` model on in the old two-call workflow.
    covariates:
        Stage-1 covariate-side formula right-hand side (e.g.
        ``"s(pc1) + s(pc2)"``). Built into the CTN covariate basis exactly as
        the standalone CTN fit would, so each per-fold refit reproduces the
        original Stage-1 covariate geometry. Pass the right-hand side only — no
        ``~`` and no response symbol.
    response_degree, response_num_internal_knots, response_penalty_order,
    response_extra_penalty_orders, double_penalty:
        Optional overrides for the CTN response-direction basis / penalty.
        Leave as ``None`` to use the Rust ``TransformationNormalConfig``
        defaults (degree 3, 10 interior knots, 2nd-difference penalty, extra
        order ``[1]``, double penalty on). Any subset may be set; unset fields
        fall back to the Rust default.
    weights:
        Optional Stage-1 observation-weight column name.
    offset:
        Optional Stage-1 offset column name.
    """

    response: str
    covariates: str
    response_degree: int | None = None
    response_num_internal_knots: int | None = None
    response_penalty_order: int | None = None
    response_extra_penalty_orders: tuple[int, ...] | None = None
    double_penalty: bool | None = None
    weights: str | None = None
    offset: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.response, str) or not self.response.strip():
            raise ValueError("CtnStage1.response must be a non-empty column name")
        if not isinstance(self.covariates, str) or not self.covariates.strip():
            raise ValueError(
                "CtnStage1.covariates must be a non-empty covariate formula right-hand side"
            )
        if "~" in self.covariates:
            raise ValueError(
                "CtnStage1.covariates is a right-hand side only; pass 's(pc1) + s(pc2)', "
                "not 'score ~ s(pc1) + s(pc2)'"
            )

    def to_rust_recipe(self) -> dict[str, Any]:
        """Serialize to the ``ctn_stage1`` JSON object consumed by the Rust
        ``PyFitConfig`` deserializer (a one-to-one mirror of the core
        ``CtnStage1Recipe`` struct). Only set CTN-config overrides are emitted;
        omitted ones let the Rust ``TransformationNormalConfig`` default fill in.
        """

        config: dict[str, Any] = {}
        if self.response_degree is not None:
            config["response_degree"] = int(self.response_degree)
        if self.response_num_internal_knots is not None:
            config["response_num_internal_knots"] = int(self.response_num_internal_knots)
        if self.response_penalty_order is not None:
            config["response_penalty_order"] = int(self.response_penalty_order)
        if self.response_extra_penalty_orders is not None:
            config["response_extra_penalty_orders"] = [
                int(order) for order in self.response_extra_penalty_orders
            ]
        if self.double_penalty is not None:
            config["double_penalty"] = bool(self.double_penalty)

        recipe: dict[str, Any] = {
            "response_column": self.response.strip(),
            "covariate_formula_rhs": self.covariates.strip(),
        }
        if config:
            recipe["config"] = config
        if self.weights is not None:
            recipe["weight_column"] = self.weights
        if self.offset is not None:
            recipe["offset_column"] = self.offset
        return recipe


def normalize_ctn_stage1(value: Any) -> CtnStage1 | None:
    """Coerce the ``transformation_normal_stage1`` argument to a
    :class:`CtnStage1` (or ``None``).

    Accepts a :class:`CtnStage1` directly, or a mapping with ``response`` /
    ``covariates`` keys (plus any optional CTN-config / weights / offset keys)
    for callers who prefer a plain dict. Anything else is a type error.
    """

    if value is None:
        return None
    if isinstance(value, CtnStage1):
        return value
    if isinstance(value, Mapping):
        # Build directly; CtnStage1.__post_init__ validates required keys.
        try:
            return CtnStage1(**value)
        except TypeError as exc:
            raise TypeError(
                "transformation_normal_stage1 mapping has unexpected keys; "
                "supported keys are 'response', 'covariates', 'response_degree', "
                "'response_num_internal_knots', 'response_penalty_order', "
                "'response_extra_penalty_orders', 'double_penalty', 'weights', 'offset'"
            ) from exc
    raise TypeError(
        "transformation_normal_stage1 must be a gamfit.CtnStage1 or a mapping of its fields"
    )
