//! #913: dispersion-channel GAMLSS location-scale families.
//!
//! Extracted from `gamlss.rs` (issue #780); this module now owns the
//! dispersion-channel joint-curvature corrections.

use super::weighted_design_products::{mirror_upper_to_lower, xt_diag_x_design, xt_diag_y_design};
// Concrete `Order2` algebra methods live on the shared `JetField` supertrait;
// keep it in scope alongside `JetScalar` so the row programs resolve them.
use super::{
    BlockwiseTermFitResult, GamlssLambdaLayout, LOCATION_SCALE_N_OUTPUTS,
    LocationScaleFamilyBuilder, build_location_scale_block, fit_location_scale_terms,
    identity_penalty, solve_penalizedweighted_projection, spatial_length_scale_term_indices,
};
use crate::block_layout::block_count::validate_block_count;
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
};
use crate::gamlss::GamlssError;
use crate::model_types::UnifiedFitResult;
use gam_linalg::matrix::LinearOperator;
use gam_math::jet_scalar::JetScalar;
use gam_math::nested_dual::JetField;
use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
    get_spatial_length_scale, spatial_term_uses_per_axis_psi,
};
use ndarray::{Array1, Array2, s};
use statrs::function::gamma::ln_gamma;

// ============================================================================
// #913: dispersion-channel GAMLSS location-scale families.
//
// `noise_formula` (a second linear predictor on the dispersion channel) was
// wired only for Gaussian/Binomial location-scale and the survival families.
// The genuine-dispersion mean families — NegativeBinomial, Gamma, Beta and
// Tweedie — were mean-only with a single scalar dispersion. This module adds a
// SINGLE generic two-block family that routes all four through the existing
// blockwise REML engine and the shared `LocationScaleFamilyBuilder` /
// `fit_location_scale_terms` plumbing, so the κ-coordinate assembly, warm
// start, shrinkage-penalised scale block and result extraction are reused
// verbatim. A family is added by supplying only its per-row log-likelihood and
// the (mean, log-precision) working sets — everything else is shared.
//
// Block layout: block 0 = mean predictor (η_μ, log link for NB/Gamma/Tweedie,
// logit for Beta); block 1 = log-precision predictor (η_d). The dispersion
// channel models log(precision) uniformly — `θ` for NegativeBinomial, the
// shape `ν` for Gamma, `φ` for Beta, and `1/φ` for Tweedie — so a larger η_d
// always means *less* dispersion, matching the Gaussian/Binomial convention
// where η_logσ smaller ⇒ tighter. With no `noise_formula` the log-precision
// block is a single intercept and the fit reduces to the scalar-dispersion
// model.
//
// NB2 with `(μ, θ)` and the exponential-dispersion members here with
// `(μ, φ)` are Fisher-orthogonal in their standard mean/dispersion
// parameterizations: Gamma uses shape `ν = 1/φ`, and Tweedie models
// `log(1/φ)`, so those precision-channel transforms preserve zero expected
// mean/dispersion cross information. Beta is the exception in this module's
// mean/precision parameterization. For `Beta(μφ, (1−μ)φ)`,
//
//   I_{μ,φ} = φ · (μ ψ'(μφ) − (1−μ) ψ'((1−μ)φ)),
//
// so in predictor coordinates `(η_μ = logit μ, η_φ = log φ)` the Fisher cross
// block is
//
//   I_{η_μ,η_φ} = μ(1−μ) φ² · (μ ψ'(μφ) − (1−μ) ψ'((1−μ)φ)),
//
// which is generically nonzero. Block-cyclic Fisher-scoring IRLS is still a
// valid block coordinate solve for the point estimate, but joint-curvature
// consumers (`log|H|`, coefficient covariance, posterior draws) must receive
// Beta's off-diagonal coefficient block. Smoothing-parameter selection still
// runs through the engine's first-order (gradient-only) outer path: the family
// declines the dense outer Hessian capability because its working weights
// couple the two blocks (`W_μ` depends on the precision and vice-versa), which
// the block-local diagonal-drift hook cannot represent exactly.
// ============================================================================

/// The genuine-dispersion mean family whose precision (overdispersion) channel
/// can carry a second `noise_formula` linear predictor (issue #913).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DispersionFamilyKind {
    /// NB2: `Var = μ + μ²/θ`; the precision channel models `log θ`.
    NegativeBinomial,
    /// Gamma with `Var = μ²/ν`; the precision channel models `log ν` (shape).
    Gamma,
    /// Beta(μφ, (1−μ)φ) with a logit mean link; the precision channel models
    /// `log φ`.
    Beta,
    /// Tweedie compound Poisson–Gamma with `Var = φ μ^p`, fixed power `p`; the
    /// precision channel models `log(1/φ)`. The per-row density uses the
    /// saddlepoint (Nelder–Pregibon) approximation for `y > 0` and the exact
    /// point mass at `y = 0`; this is the standard tractable Tweedie ML
    /// surface (an exact-series φ-derivative is the remaining hard sub-item of
    /// #913).
    Tweedie { p: f64 },
}

impl DispersionFamilyKind {
    pub const fn family_tag(self) -> &'static str {
        match self {
            DispersionFamilyKind::NegativeBinomial => FAMILY_NEGBIN_LOCATION_SCALE,
            DispersionFamilyKind::Gamma => FAMILY_GAMMA_LOCATION_SCALE,
            DispersionFamilyKind::Beta => FAMILY_BETA_LOCATION_SCALE,
            DispersionFamilyKind::Tweedie { .. } => FAMILY_TWEEDIE_LOCATION_SCALE,
        }
    }

    /// The mean link is logit for Beta (a probability mean) and log otherwise.
    pub(crate) const fn mean_is_logit(self) -> bool {
        matches!(self, DispersionFamilyKind::Beta)
    }

    /// The mean inverse link this dispersion family fits on: log for
    /// NegativeBinomial / Gamma / Tweedie, logit for Beta. Single source of
    /// truth shared by the CLI and FFI save paths so the persisted
    /// `base_link` never diverges from the fitted channel.
    pub fn base_link(self) -> gam_problem::InverseLink {
        use gam_problem::{InverseLink, StandardLink};
        if self.mean_is_logit() {
            InverseLink::Standard(StandardLink::Logit)
        } else {
            InverseLink::Standard(StandardLink::Log)
        }
    }

    /// The family's canonical [`LikelihoodSpec`] (mean response × mean link).
    /// The overdispersion parameter is estimated by the log-precision channel,
    /// so the response-family placeholder parameters (`phi`, `theta`) mirror
    /// the [`resolve_family`](crate::fit_orchestration::materialize::resolve_family) defaults
    /// and are not consumed as fixed values at predict time. This is the single
    /// source of truth for the persisted location-scale likelihood so the CLI
    /// and FFI save paths cannot diverge.
    pub fn likelihood_spec(self) -> gam_problem::LikelihoodSpec {
        use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
        let response = match self {
            DispersionFamilyKind::NegativeBinomial => ResponseFamily::NegativeBinomial {
                theta: 1.0,
                theta_fixed: false,
            },
            DispersionFamilyKind::Gamma => ResponseFamily::Gamma,
            DispersionFamilyKind::Beta => ResponseFamily::Beta { phi: 1.0 },
            DispersionFamilyKind::Tweedie { p } => ResponseFamily::Tweedie { p },
        };
        let link = if self.mean_is_logit() {
            InverseLink::Standard(StandardLink::Logit)
        } else {
            InverseLink::Standard(StandardLink::Log)
        };
        LikelihoodSpec::new(response, link)
    }
}

pub const FAMILY_NEGBIN_LOCATION_SCALE: &str = "negbin-location-scale";
pub const FAMILY_GAMMA_LOCATION_SCALE: &str = "gamma-location-scale";
pub const FAMILY_BETA_LOCATION_SCALE: &str = "beta-location-scale";
pub const FAMILY_TWEEDIE_LOCATION_SCALE: &str = "tweedie-location-scale";

/// Row count above which the per-row dispersion-kernel map fans out across
/// rayon workers (only when not already running on a worker, to avoid nested
/// oversubscription). Below it the serial map beats the fork/join overhead.
/// Mirrors the row-chunk guard in
/// [`row_coeff_operator`](super::gaussian::row_coeff_operator).
const DISPERSION_PARALLEL_ROW_THRESHOLD: usize = 1024;

/// Per-row working quantities for both channels at the current `(η_μ, η_d)`.
pub(super) struct DispersionRowKernel {
    pub(super) loglik: f64,
    pub(super) mean_weight: f64,
    pub(super) mean_response: f64,
    pub(super) disp_weight: f64,
    pub(super) disp_response: f64,
}

#[inline]
fn dispersion_geometry_error(row: usize, quantity: &'static str, eta: f64, value: f64) -> String {
    GamlssError::RowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
    .into()
}

/// Certify the exact open parameter domain used by the row towers.  The domain
/// is defined by representability of the linked distribution parameters, not
/// by an arbitrary predictor box.
fn validate_dispersion_row_geometry_inputs(
    kind: DispersionFamilyKind,
    row: usize,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> Result<(), String> {
    if !eta_mu.is_finite() || !eta_d.is_finite() {
        return Err(GamlssError::NonFinite {
            reason: format!(
                "{} requires finite predictors at row {row}; eta_mu={eta_mu}, eta_d={eta_d}",
                kind.family_tag()
            ),
        }
        .into());
    }
    if !prior_weight.is_finite() || prior_weight < 0.0 {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "{} requires finite non-negative prior weights; weight[{row}]={prior_weight}",
                kind.family_tag()
            ),
        }
        .into());
    }
    if prior_weight == 0.0 {
        return Ok(());
    }
    let (support_ok, support) = match kind {
        DispersionFamilyKind::NegativeBinomial => (
            yi.is_finite() && yi >= 0.0 && yi.fract() == 0.0,
            "a finite non-negative integer",
        ),
        DispersionFamilyKind::Gamma => (yi.is_finite() && yi > 0.0, "finite and > 0"),
        DispersionFamilyKind::Beta => (
            yi.is_finite() && yi > 0.0 && yi < 1.0,
            "finite and strictly inside (0, 1)",
        ),
        DispersionFamilyKind::Tweedie { p } => (
            yi.is_finite() && yi >= 0.0 && p.is_finite() && p > 1.0 && p < 2.0,
            "finite and >= 0 with power strictly inside (1, 2)",
        ),
    };
    if !support_ok {
        return Err(GamlssError::InvalidInput {
            reason: format!(
                "{} response outside support at row {row}: y={yi} (requires {support})",
                kind.family_tag()
            ),
        }
        .into());
    }

    let require_positive = |quantity, eta, value: f64| {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(dispersion_geometry_error(row, quantity, eta, value))
        }
    };
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = eta_mu.exp();
            let theta = eta_d.exp();
            require_positive("negative-binomial mean exp(eta_mu)", eta_mu, mu)?;
            require_positive("negative-binomial precision exp(eta_d)", eta_d, theta)
        }
        DispersionFamilyKind::Gamma => {
            require_positive("Gamma mean exp(eta_mu)", eta_mu, eta_mu.exp())?;
            require_positive("Gamma precision exp(eta_d)", eta_d, eta_d.exp())
        }
        DispersionFamilyKind::Beta => {
            let mu = gam_linalg::utils::stable_logistic(eta_mu);
            if !mu.is_finite() || mu <= 0.0 || mu >= 1.0 {
                return Err(dispersion_geometry_error(
                    row,
                    "Beta mean logistic(eta_mu) in the open unit interval",
                    eta_mu,
                    mu,
                ));
            }
            let phi = eta_d.exp();
            require_positive("Beta precision exp(eta_d)", eta_d, phi)?;
            require_positive("Beta first shape mu*phi", eta_mu, mu * phi)?;
            require_positive("Beta second shape (1-mu)*phi", eta_mu, (1.0 - mu) * phi)
        }
        DispersionFamilyKind::Tweedie { .. } => {
            require_positive("Tweedie mean exp(eta_mu)", eta_mu, eta_mu.exp())?;
            require_positive("Tweedie dispersion exp(-eta_d)", eta_d, (-eta_d).exp())
        }
    }
}

fn validate_dispersion_row_kernel_output(
    row: usize,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
    output: &DispersionRowKernel,
) -> Result<(), String> {
    if prior_weight == 0.0 {
        return Ok(());
    }
    for (quantity, eta, value, strictly_positive) in [
        (
            "dispersion-family row log likelihood",
            eta_mu,
            output.loglik,
            false,
        ),
        (
            "dispersion-family mean working weight",
            eta_mu,
            output.mean_weight,
            true,
        ),
        (
            "dispersion-family mean working response",
            eta_mu,
            output.mean_response,
            false,
        ),
        (
            "dispersion-family precision working weight",
            eta_d,
            output.disp_weight,
            true,
        ),
        (
            "dispersion-family precision working response",
            eta_d,
            output.disp_response,
            false,
        ),
    ] {
        if !value.is_finite() || (strictly_positive && value <= 0.0) {
            return Err(dispersion_geometry_error(row, quantity, eta, value));
        }
    }
    Ok(())
}

#[cfg(test)]
mod test_support {
    use super::*;

    /// Test-oracle NB2 row NLL over a generic [`JetScalar<2>`], seeded on the
    /// natural parameters `(μ, θ)`.
    #[inline]
    pub(super) fn dispersion_nb_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
        yi: f64,
        mu_value: f64,
        theta_value: f64,
        wi: f64,
    ) -> S {
        let mu = S::variable(mu_value, 0);
        let theta = S::variable(theta_value, 1);
        let tpm = theta.add(&mu);
        // (theta + yi).ln_gamma() - theta.ln_gamma() - ln_gamma(yi+1)
        //   + theta*theta.ln() - theta*tpm.ln() + mu.ln()*yi - tpm.ln()*yi
        let loglik = theta
            .add(&S::constant(yi))
            .ln_gamma()
            .sub(&theta.ln_gamma())
            .sub(&S::constant(ln_gamma(yi + 1.0)))
            .add(&theta.mul(&theta.ln()))
            .sub(&theta.mul(&tpm.ln()))
            .add(&mu.ln().scale(yi))
            .sub(&tpm.ln().scale(yi));
        loglik.scale(-wi)
    }

    /// Test-oracle Gamma row NLL over a generic [`JetScalar<2>`], seeded on
    /// `(μ, ν)`.
    #[inline]
    pub(super) fn dispersion_gamma_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
        yi: f64,
        y_pos: f64,
        mu_value: f64,
        nu_value: f64,
        wi: f64,
    ) -> S {
        let mu = S::variable(mu_value, 0);
        let nu = S::variable(nu_value, 1);
        // nu*nu.ln() - nu*mu.ln() - nu.ln_gamma() + (nu-1)*y_pos.ln() - nu*(mu.recip()*yi)
        let loglik = nu
            .mul(&nu.ln())
            .sub(&nu.mul(&mu.ln()))
            .sub(&nu.ln_gamma())
            .add(&nu.sub(&S::constant(1.0)).scale(y_pos.ln()))
            .sub(&nu.mul(&mu.recip().scale(yi)));
        loglik.scale(-wi)
    }

    /// Test-oracle Beta row NLL over a generic [`JetScalar<2>`], seeded on
    /// `(μ, φ)`.
    #[inline]
    pub(super) fn dispersion_beta_nll_generic<S: gam_math::jet_scalar::JetScalar<2>>(
        yi: f64,
        mu_value: f64,
        phi_value: f64,
        wi: f64,
    ) -> S {
        let mu = S::variable(mu_value, 0);
        let phi = S::variable(phi_value, 1);
        let one_minus_mu = S::constant(1.0).sub(&mu);
        let yc = yi;
        let a = mu.mul(&phi);
        let b = one_minus_mu.mul(&phi);
        // phi.ln_gamma() - a.ln_gamma() - b.ln_gamma()
        //   + (a-1)*yc.ln() + (b-1)*(1-yc).ln()
        let loglik = phi
            .ln_gamma()
            .sub(&a.ln_gamma())
            .sub(&b.ln_gamma())
            .add(&a.sub(&S::constant(1.0)).scale(yc.ln()))
            .add(&b.sub(&S::constant(1.0)).scale((-yc).ln_1p()));
        loglik.scale(-wi)
    }

    /// #1591 jet-prune oracle: full `Order2<2>` (value/grad/Hessian) NB2 row NLL.
    ///
    /// Production no longer consumes the mean (`μ`-axis) derivative channels of
    /// this tower — the NB mean block is Fisher-orthogonal and hand-written
    /// exactly in [`dispersion_row_kernel`] — so the hot path uses the pruned
    /// single-axis [`dispersion_nb_disp_order2`] instead. This `K=2` form
    /// survives only as the dense-`Tower4<2>` oracle pin
    /// (`order2_matches_dense_tower_all_channels`).
    #[inline]
    pub(super) fn dispersion_nb_nll_order2(
        yi: f64,
        mu_value: f64,
        theta_value: f64,
        wi: f64,
    ) -> gam_math::jet_scalar::Order2<2> {
        type O2 = gam_math::jet_scalar::Order2<2>;

        let mu = O2::variable(mu_value, 0);
        let theta = O2::variable(theta_value, 1);
        let tpm = theta.add(&mu);
        let theta_plus_y = theta.add(&O2::constant(yi));
        let loglik = order2_ln_gamma(&theta_plus_y)
            .sub(&order2_ln_gamma(&theta))
            .sub(&O2::constant(ln_gamma(yi + 1.0)))
            .add(&theta.mul(&theta.ln()))
            .sub(&theta.mul(&tpm.ln()))
            .add(&mu.ln().scale(yi))
            .sub(&tpm.ln().scale(yi));
        loglik.scale(-wi)
    }

    /// #1591 jet-prune oracle: full `Order2<2>` Gamma row NLL. As with NB, the
    /// mean axis is unused in production (hand-written, Fisher-orthogonal); the
    /// hot path uses the single-axis [`dispersion_gamma_disp_order2`]. Kept only
    /// as the dense-tower oracle pin.
    #[inline]
    pub(super) fn dispersion_gamma_nll_order2(
        yi: f64,
        y_pos: f64,
        mu_value: f64,
        nu_value: f64,
        wi: f64,
    ) -> gam_math::jet_scalar::Order2<2> {
        type O2 = gam_math::jet_scalar::Order2<2>;

        let mu = O2::variable(mu_value, 0);
        let nu = O2::variable(nu_value, 1);
        let loglik = nu
            .mul(&nu.ln())
            .sub(&nu.mul(&mu.ln()))
            .sub(&order2_ln_gamma(&nu))
            .add(&nu.sub(&O2::constant(1.0)).scale(y_pos.ln()))
            .sub(&nu.mul(&mu.recip().scale(yi)));
        loglik.scale(-wi)
    }
}

/// Production `Order2<2>` Beta row NLL (value/grad/Hessian hot path; the cross
/// channel `h()[0][1]` feeds the Beta observed cross weight).
#[inline]
pub(crate) fn dispersion_beta_nll_order2(
    yi: f64,
    mu_value: f64,
    phi_value: f64,
    wi: f64,
) -> gam_math::jet_scalar::Order2<2> {
    type O2 = gam_math::jet_scalar::Order2<2>;

    let mu = O2::variable(mu_value, 0);
    let phi = O2::variable(phi_value, 1);
    let one_minus_mu = O2::constant(1.0).sub(&mu);
    let yc = yi;
    let a = mu.mul(&phi);
    let b = one_minus_mu.mul(&phi);
    let loglik = order2_ln_gamma(&phi)
        .sub(&order2_ln_gamma(&a))
        .sub(&order2_ln_gamma(&b))
        .add(&a.sub(&O2::constant(1.0)).scale(yc.ln()))
        .add(&b.sub(&O2::constant(1.0)).scale((-yc).ln_1p()));
    loglik.scale(-wi)
}

#[inline]
fn order2_ln_gamma<const K: usize>(
    x: &gam_math::jet_scalar::Order2<K>,
) -> gam_math::jet_scalar::Order2<K> {
    gam_math::jet_scalar::Order2(
        x.0.compose_unary(gam_math::jet_tower::ln_gamma_derivative_stack_order2(x.0.v)),
    )
}

// ============================================================================
// #1591 jet-prune: single-axis (`K=1`) dispersion-channel towers.
//
// For NegativeBinomial / Gamma / Tweedie the production row kernel consumes ONLY
// the dispersion-axis derivatives (`g[disp]`, `h[disp][disp]`) and the value;
// the mean block is Fisher-orthogonal and assembled in closed form. Seeding the
// mean as a CONSTANT and the dispersion parameter as the SOLE jet variable
// therefore yields a tower whose `(value, g[0], h[0][0])` are `to_bits`-
// identical to the consumed `(value, g[1], h[1][1])` of the old `Order2<2>`
// tower — the mean seed only ever populated the now-discarded `g[mean]` /
// `h[mean][·]` channels (Leibniz/Faà-di-Bruno never read the dispersion-axis
// channels off the mean seed). Collapsing `K=2 → K=1` quarters the Hessian
// tensor (1 entry vs 4) and halves the gradient, with no change to any consumed
// float bit. The `ln_gamma` derivative stacks are unchanged (the irreducible
// transcendental cost), so this trims the rational composition, not the special
// functions.
// ============================================================================

// `dispersion_nb_disp_order1` / `dispersion_nb_disp_order2` — the pruned
// `Order1<1>` / `Order2<1>` NB2 dispersion oracle pins used only by
// `prune_towers_*` — now live in the `#[cfg(test)] mod tests` below, next to
// their sole consumer (they are genuinely test-support, not production —
// the real NB2 dispersion row kernel below computes score/curvature in
// closed form via `digamma`/`nb_log_precision_fisher_jensen`, never through
// either jet tower — so they belong in a `#[cfg(test)]` mod rather than
// carrying `allow(dead_code)`).

/// Pruned single-axis Gamma dispersion tower: `ν` is the sole jet variable
/// (axis 0), `μ` a constant. Consumed channels match
/// `dispersion_gamma_nll_order2` index-1 bit-for-bit.
#[inline]
pub(crate) fn dispersion_gamma_disp_order2(
    yi: f64,
    y_pos: f64,
    mu_value: f64,
    nu_value: f64,
    wi: f64,
) -> gam_math::jet_scalar::Order2<1> {
    type O1 = gam_math::jet_scalar::Order2<1>;

    let mu = O1::constant(mu_value);
    let nu = O1::variable(nu_value, 0);
    let loglik = nu
        .mul(&nu.ln())
        .sub(&nu.mul(&mu.ln()))
        .sub(&order2_ln_gamma(&nu))
        .add(&nu.sub(&O1::constant(1.0)).scale(y_pos.ln()))
        .sub(&nu.mul(&mu.recip().scale(yi)));
    loglik.scale(-wi)
}

/// Pruned single-axis Tweedie dispersion tower seeded on the predictor `η_d`
/// (axis 0), with `η_μ` a constant (so `μ = exp(η_μ)` carries no jet). The
/// `φ = exp(−η_d)` chain and its nonlinear `∂²φ/∂η_d²` curvature are carried
/// exactly as in `dispersion_tweedie_nll_generic`; `value`/`g[0]`/`h[0][0]`
/// match that program's `value`/`g[1]`/`h[1][1]` bit-for-bit.
#[inline]
pub(crate) fn dispersion_tweedie_disp_order2(
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    p: f64,
    wi: f64,
) -> gam_math::jet_scalar::Order2<1> {
    type O1 = gam_math::jet_scalar::Order2<1>;

    let one_minus_p = 1.0 - p;
    let two_minus_p = 2.0 - p;
    let mu = O1::constant(eta_mu).exp();
    let phi = O1::variable(eta_d, 0).scale(-1.0).exp();
    if yi > 0.0 {
        let dev = mu
            .powf(two_minus_p)
            .scale(1.0 / two_minus_p)
            .sub(&mu.powf(one_minus_p).scale(yi / one_minus_p))
            .add(&O1::constant(
                yi.powf(two_minus_p) / (one_minus_p * two_minus_p),
            ))
            .scale(2.0);
        let loglik = dev
            .mul(&phi.recip().scale(-0.5))
            .sub(&phi.scale(2.0 * std::f64::consts::PI).ln().scale(0.5))
            .sub(&O1::constant(0.5 * p * yi.ln()));
        loglik.scale(-wi)
    } else {
        let c = mu.powf(two_minus_p).scale(1.0 / two_minus_p);
        let loglik = c.mul(&phi.recip()).scale(-1.0);
        loglik.scale(-wi)
    }
}

// ============================================================================
// #1591 jet-prune: value-only (`K=0`) row negative-log-likelihood.
//
// `log_likelihood_only` reads ONLY `row.loglik = -tower.value()`; the full row
// kernel it used to call evaluated every dispersion tower's gradient AND Hessian
// — including the digamma/trigamma derivative stacks — purely to discard them.
// These functions evaluate the SAME value-channel program in plain `f64`, so
// they are `to_bits`-identical to `-tower.value()` (the jet value channel is the
// naive scalar evaluation: `mul.v = a.v*b.v`, `compose.v = stack[0]`), while
// touching only `ln_gamma` (stack slot 0) and never the digamma/trigamma slots.
// On a per-row loglik that is the dominant transcendental saving.
// ============================================================================

/// NB2 row log-likelihood, evaluated through stable log shares so `mu+theta`
/// is never formed and may mathematically exceed `f64::MAX`.
#[inline]
fn dispersion_nb_loglik(yi: f64, mu: f64, theta: f64, wi: f64) -> f64 {
    let log_theta_share = log_positive_share(theta, mu);
    let log_mu_share = log_positive_share(mu, theta);
    let s = ln_gamma(theta + yi) - ln_gamma(theta) - ln_gamma(yi + 1.0)
        + theta * log_theta_share
        + yi * log_mu_share;
    -(s * -wi)
}

/// `log(numerator / (numerator + other))` without forming the potentially
/// overflowing sum or subtracting nearly equal logarithms.
#[inline]
fn log_positive_share(numerator: f64, other: f64) -> f64 {
    if numerator >= other {
        -(other / numerator).ln_1p()
    } else {
        let ratio = numerator / other;
        numerator.ln() - other.ln() - ratio.ln_1p()
    }
}

#[inline]
fn positive_share(numerator: f64, other: f64) -> f64 {
    if numerator >= other {
        1.0 / (1.0 + other / numerator)
    } else {
        let ratio = numerator / other;
        ratio / (1.0 + ratio)
    }
}

/// Jensen NB precision Fisher information already transformed to log-precision
/// coordinates, `theta^2 I_theta`.  For large theta, expand
/// `trigamma(x)-1/x` after the transformation so the representable O(1)
/// result is never obtained by subtracting underflowed O(theta^-2) terms.
#[inline]
fn nb_log_precision_fisher_jensen(mu: f64, theta: f64) -> f64 {
    let r = positive_share(theta, mu);
    let q = positive_share(mu, theta);
    if theta <= 32.0 {
        let total = theta + mu;
        let remainder_theta = gam_math::jet_tower::trigamma(theta) - theta.recip();
        let remainder_total = gam_math::jet_tower::trigamma(total) - total.recip();
        return theta * theta * (remainder_theta - remainder_total);
    }
    let one_minus_r2 = q * (1.0 + r);
    let r2 = r * r;
    let one_minus_r3 = q * (1.0 + r + r2);
    let r4 = r2 * r2;
    let one_minus_r5 = q * (1.0 + r + r2 + r2 * r + r4);
    let r6 = r4 * r2;
    let one_minus_r7 = q * (1.0 + r + r2 + r2 * r + r4 + r4 * r + r6);
    let inv = theta.recip();
    let inv2 = inv * inv;
    0.5 * one_minus_r2 + (inv / 6.0) * one_minus_r3 - (inv * inv2 / 30.0) * one_minus_r5
        + (inv * inv2 * inv2 / 42.0) * one_minus_r7
}

/// Gamma row log-likelihood, plain `f64`, bit-identical to
/// `-dispersion_gamma_disp_order2(..).value()`.
#[inline]
fn dispersion_gamma_loglik(yi: f64, y_pos: f64, mu: f64, nu: f64, wi: f64) -> f64 {
    // NB: the jet forms `μ.recip().scale(yi)` = `(1/μ)·yᵢ` (reciprocal then
    // multiply), NOT `yᵢ/μ` (single divide) — these differ in the last bit, so
    // the value path must reproduce the reciprocal-then-multiply exactly.
    let s = nu * nu.ln() - nu * mu.ln() - ln_gamma(nu) + (nu - 1.0) * y_pos.ln()
        - nu * ((1.0 / mu) * yi);
    -(s * -wi)
}

/// Beta row log-likelihood, plain `f64`, bit-identical to
/// `-dispersion_beta_nll_order2(..).value()`.
#[inline]
fn dispersion_beta_loglik(yi: f64, mu: f64, phi: f64, wi: f64) -> f64 {
    let one_minus_mu = 1.0 - mu;
    let yc = yi;
    let a = mu * phi;
    let b = one_minus_mu * phi;
    let s =
        ln_gamma(phi) - ln_gamma(a) - ln_gamma(b) + (a - 1.0) * yc.ln() + (b - 1.0) * (-yc).ln_1p();
    -(s * -wi)
}

/// Tweedie row log-likelihood, plain `f64`, bit-identical to
/// `-dispersion_tweedie_disp_order2(..).value()` (both density branches).
#[inline]
fn dispersion_tweedie_loglik(yi: f64, eta_mu: f64, eta_d: f64, p: f64, wi: f64) -> f64 {
    let one_minus_p = 1.0 - p;
    let two_minus_p = 2.0 - p;
    let mu = eta_mu.exp();
    let phi = (-eta_d).exp();
    let s = if yi > 0.0 {
        let dev = (mu.powf(two_minus_p) * (1.0 / two_minus_p)
            - mu.powf(one_minus_p) * (yi / one_minus_p)
            + yi.powf(two_minus_p) / (one_minus_p * two_minus_p))
            * 2.0;
        dev * ((1.0 / phi) * -0.5)
            - (phi * (2.0 * std::f64::consts::PI)).ln() * 0.5
            - 0.5 * p * yi.ln()
    } else {
        let c = mu.powf(two_minus_p) * (1.0 / two_minus_p);
        (c * (1.0 / phi)) * -1.0
    };
    -(s * -wi)
}

/// Value-only row negative log-likelihood for one observation — the pruned hot
/// path for [`CustomFamily::log_likelihood_only`]. Mirrors the exact-link
/// preamble of [`dispersion_row_kernel`] exactly, then evaluates ONLY the value
/// channel (no gradient/Hessian, no digamma/trigamma). Returns `row.loglik`
/// `to_bits`-identically.
#[inline]
pub(crate) fn dispersion_row_loglik(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> f64 {
    // Zero-weight rows are excluded from the likelihood entirely (and exempt
    // from the boundary support validation), so their row term must be an
    // exact 0 rather than `0 · (±inf)` = NaN.
    if prior_weight <= 0.0 {
        return 0.0;
    }
    let wi = prior_weight;
    let em = eta_mu;
    let ed = eta_d;
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp();
            let theta = ed.exp();
            dispersion_nb_loglik(yi, mu, theta, wi)
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp();
            let nu = ed.exp();
            let y_pos = yi;
            dispersion_gamma_loglik(yi, y_pos, mu, nu, wi)
        }
        DispersionFamilyKind::Beta => {
            let mu = gam_linalg::utils::stable_logistic(em);
            let phi = ed.exp();
            dispersion_beta_loglik(yi, mu, phi, wi)
        }
        DispersionFamilyKind::Tweedie { p } => dispersion_tweedie_loglik(yi, em, ed, p, wi),
    }
}

/// Observed η-space row NLL tower: both exact predictors are jet variables
/// (`η_μ` axis 0, `η_d` axis 1) and the full mean-link / precision-link
/// chains are carried by the jet algebra, so `h()` is the exact per-row
/// OBSERVED Hessian in `(η_μ, η_d)` — including the inverse-link
/// second-derivative terms and the mean/dispersion cross curvature that the
/// expected (Fisher) working weights do not represent. Example: Gamma with
/// log links at `y = 4, μ = 2, ν = 3` has exact per-row `∂²NLL/∂η_μ² =
/// νy/μ = 6` and `∂²NLL/∂η_μ∂η_ν = ν(1 − y/μ) = −3`, where the Fisher
/// working weights give `ν = 3` and `0`.
pub(crate) fn dispersion_eta_nll_order2(
    kind: DispersionFamilyKind,
    yi: f64,
    em: f64,
    ed: f64,
    wi: f64,
) -> gam_math::jet_scalar::Order2<2> {
    type O2 = gam_math::jet_scalar::Order2<2>;
    let eta_mu = O2::variable(em, 0);
    let eta_d = O2::variable(ed, 1);
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            // The NB log-likelihood below is written directly in the linear
            // predictors (log-scale) via `log_total`, so the mean `exp(eta_mu)`
            // is never materialized here (unlike the Gamma arm).
            let theta = eta_d.exp();
            let theta_plus_y = theta.add(&O2::constant(yi));
            let log_total = if em >= ed {
                eta_mu.add(&eta_d.sub(&eta_mu).exp().add(&O2::constant(1.0)).ln())
            } else {
                eta_d.add(&eta_mu.sub(&eta_d).exp().add(&O2::constant(1.0)).ln())
            };
            let loglik = order2_ln_gamma(&theta_plus_y)
                .sub(&order2_ln_gamma(&theta))
                .sub(&O2::constant(ln_gamma(yi + 1.0)))
                .add(&theta.mul(&eta_d.sub(&log_total)))
                .add(&eta_mu.sub(&log_total).scale(yi));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Gamma => {
            let mu = eta_mu.exp();
            let nu = eta_d.exp();
            let y_pos = yi;
            let loglik = nu
                .mul(&nu.ln())
                .sub(&nu.mul(&mu.ln()))
                .sub(&order2_ln_gamma(&nu))
                .add(&nu.sub(&O2::constant(1.0)).scale(y_pos.ln()))
                .sub(&nu.mul(&mu.recip().scale(yi)));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Beta => {
            let mu = eta_mu.scale(-1.0).exp().add(&O2::constant(1.0)).recip();
            let phi = eta_d.exp();
            let one_minus_mu = O2::constant(1.0).sub(&mu);
            let yc = yi;
            let a = mu.mul(&phi);
            let b = one_minus_mu.mul(&phi);
            let loglik = order2_ln_gamma(&phi)
                .sub(&order2_ln_gamma(&a))
                .sub(&order2_ln_gamma(&b))
                .add(&a.sub(&O2::constant(1.0)).scale(yc.ln()))
                .add(&b.sub(&O2::constant(1.0)).scale((-yc).ln_1p()));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Tweedie { p } => {
            let one_minus_p = 1.0 - p;
            let two_minus_p = 2.0 - p;
            let mu = eta_mu.exp();
            let phi = eta_d.scale(-1.0).exp();
            if yi > 0.0 {
                let dev = mu
                    .powf(two_minus_p)
                    .scale(1.0 / two_minus_p)
                    .sub(&mu.powf(one_minus_p).scale(yi / one_minus_p))
                    .add(&O2::constant(
                        yi.powf(two_minus_p) / (one_minus_p * two_minus_p),
                    ))
                    .scale(2.0);
                let loglik = dev
                    .mul(&phi.recip().scale(-0.5))
                    .sub(&phi.scale(2.0 * std::f64::consts::PI).ln().scale(0.5))
                    .sub(&O2::constant(0.5 * p * yi.ln()));
                loglik.scale(-wi)
            } else {
                let c = mu.powf(two_minus_p).scale(1.0 / two_minus_p);
                let loglik = c.mul(&phi.recip()).scale(-1.0);
                loglik.scale(-wi)
            }
        }
    }
}

/// Per-row observed `(∂²NLL/∂η_μ², ∂²NLL/∂η_μ∂η_d, ∂²NLL/∂η_d²)` weights for
/// the exact joint Hessian at the supplied predictors.
pub(crate) fn dispersion_row_observed_hessian_weights(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> (f64, f64, f64) {
    if prior_weight <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let tower = dispersion_eta_nll_order2(kind, yi, eta_mu, eta_d, prior_weight);
    let h = tower.h();
    (h[0][0], h[0][1], h[1][1])
}

/// Exact row-local geometry consumed by saved-model case deletion.
///
/// The score is the gradient of the weighted negative log-likelihood in the
/// affine coordinates `(eta_mu, eta_d)`.  `observed_hessian` is its observed
/// Hessian, not a Fisher working-weight surrogate and not the outer product of
/// the score.  Keeping those two objects separate is essential for ALO: the
/// observed Hessian controls the deletion denominator, while the score outer
/// product controls the sandwich variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispersionAloRowGeometry {
    pub nll_score: [f64; 2],
    pub observed_hessian: [[f64; 2]; 2],
}

/// Replay the exact fitted row likelihood in its two affine predictor
/// coordinates for saved-model ALO.
///
/// This is intentionally a thin public boundary over the same order-two jet
/// program used by the fitter, so diagnostics cannot drift onto a second,
/// hand-maintained approximation of the dispersion likelihood.
pub fn dispersion_alo_row_geometry(
    kind: DispersionFamilyKind,
    row: usize,
    y: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> Result<DispersionAloRowGeometry, String> {
    validate_dispersion_row_geometry_inputs(kind, row, y, eta_mu, eta_d, prior_weight)?;
    if prior_weight == 0.0 {
        return Ok(DispersionAloRowGeometry {
            nll_score: [0.0; 2],
            observed_hessian: [[0.0; 2]; 2],
        });
    }
    let tower = dispersion_eta_nll_order2(kind, y, eta_mu, eta_d, prior_weight);
    let (_, gradient, hessian) = tower.into_channels();
    let geometry = DispersionAloRowGeometry {
        nll_score: gradient,
        observed_hessian: hessian,
    };
    if geometry
        .nll_score
        .iter()
        .chain(geometry.observed_hessian.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(GamlssError::RowGeometryUnrepresentable {
            row,
            quantity: "dispersion-family ALO row geometry",
            eta: eta_mu,
            value: f64::NAN,
        }
        .into());
    }
    Ok(geometry)
}

#[inline]
pub(crate) fn tower_score_info<const K: usize>(
    tower: &gam_math::jet_scalar::Order2<K>,
    idx: usize,
    wi: f64,
) -> (f64, f64) {
    if wi == 0.0 {
        (0.0, 0.0)
    } else {
        (-tower.g()[idx] / wi, tower.h()[idx][idx] / wi)
    }
}

/// Evaluate the row log-likelihood and the (mean, log-precision) Fisher-scoring
/// working sets for one observation. `eta_mu`/`eta_d` already include any
/// per-channel offset (they are the block predictors). `prior_weight` is the
/// observation's prior weight.
pub(super) fn dispersion_row_kernel(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> DispersionRowKernel {
    let em = eta_mu;
    let ed = eta_d;
    // Zero-weight rows are excluded from the likelihood (and exempt from the
    // boundary support validation): return exact zeros rather than letting
    // `0 · (±inf)` poison the objective sum.
    if prior_weight <= 0.0 {
        return DispersionRowKernel {
            loglik: 0.0,
            mean_weight: 0.0,
            mean_response: em,
            disp_weight: 0.0,
            disp_response: ed,
        };
    }
    let wi = prior_weight;
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp();
            let theta = ed.exp(); // precision (size)
            let loglik = dispersion_nb_loglik(yi, mu, theta, wi);
            let mean_eta_information = if mu >= theta {
                theta / (1.0 + theta / mu)
            } else {
                mu / (1.0 + mu / theta)
            };
            let mean_weight = wi * mean_eta_information;
            let mean_response = em + (yi - mu) / mu;
            // Dispersion (log-θ) IRLS curvature: use the EXPECTED (Fisher)
            // information in θ, not the per-row OBSERVED Hessian channel
            // (`_info_theta_observed`). The NB2 log-likelihood is strongly
            // non-quadratic in θ: `−∂²ℓ/∂θ²` carries the row-specific term
            // `ψ′(θ+y)` and goes NEGATIVE for every row whose count sits below
            // its current fitted precision (overestimated size / underestimated
            // overdispersion). Far from the optimum a majority of rows can be
            // negative, so the assembled block curvature `Xᵀdiag(w)X` loses
            // positive-definiteness; replacing each negative row by an
            // arbitrary epsilon then divides the exact score by
            // ~0 in the working response, producing O(1e12) IRLS targets that
            // make the dispersion block step explode and the inner block-cyclic
            // solve stall (never reaching KKT within the cycle budget — the
            // `nb` location-scale `IntegrationError`, gam#1606). The mean block
            // already uses its closed-form expected info `θ/(μ(θ+μ))`; the
            // dispersion block must do the same.
            //
            // The Fisher information in θ has the closed form
            //   I(θ) = ψ′(θ) − E[ψ′(θ+Y)] − 1/θ + 1/(θ+μ),
            // whose only costly piece is the per-row infinite expectation
            // `E[ψ′(θ+Y)]`. Replacing it with the Jensen plug-in `ψ′(θ+μ)`
            // (valid because ψ′ is convex, so this is a tight lower bound on the
            // expectation) gives a per-row, sum-free, STRICTLY POSITIVE
            // curvature
            //   I_θ ≈ ψ′(θ) − ψ′(θ+μ) − 1/θ + 1/(θ+μ) > 0  for all (μ,θ),
            // since ψ′ is strictly decreasing. The working RESPONSE still
            // carries the EXACT score `s_theta` (= ∂ℓ/∂θ from the tower), so the
            // penalized stationary point (score = 0) is byte-unchanged — this is
            // Fisher scoring, which only re-conditions the inner solve and never
            // shifts the optimum. The observed channel `_info_theta_observed` is no
            // longer consumed for the weight.
            // #1591-follow-up: scalar `trigamma` (== `trigamma_derivative_stack
            // (·)[0]` bit-for-bit) evaluates ONLY ψ′; the old `[0]`-index form
            // built the full order-1..5 polygamma stack and discarded four of
            // five per call (8 wasted polygamma evaluations per NB2 row).
            let theta_fraction = if theta >= mu {
                (mu / theta - yi / theta) / (1.0 + mu / theta)
            } else {
                (1.0 - yi / mu) / (1.0 + theta / mu)
            };
            let score_theta = gam_math::jet_tower::digamma(theta + yi)
                - gam_math::jet_tower::digamma(theta)
                + log_positive_share(theta, mu)
                + theta_fraction;
            let score_eta = theta * score_theta;
            let eta_information = nb_log_precision_fisher_jensen(mu, theta);
            let disp_weight = wi * eta_information;
            let disp_response = ed + score_eta / eta_information;
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp();
            let nu = ed.exp(); // precision = shape ν
            let tower = dispersion_gamma_disp_order2(yi, yi, mu, nu, wi);
            let (s_nu, info_nu_raw) = tower_score_info(&tower, 0, wi);
            let loglik = -tower.value();
            let mean_weight = wi * nu;
            let mean_response = em + (yi - mu) / mu;
            let disp_weight = wi * nu * nu * info_nu_raw;
            let disp_response = ed + s_nu / (nu * info_nu_raw);
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Beta => {
            // logit mean link.
            let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em);
            let mu = logit.mu;
            let phi = ed.exp(); // precision
            let q = logit.d1;
            let tower = dispersion_beta_nll_order2(yi, mu, phi, wi);
            let (score_mu, _) = tower_score_info(&tower, 0, wi);
            let (s_phi, _) = tower_score_info(&tower, 1, wi);
            let loglik = -tower.value();
            let a = mu * phi;
            let b = (1.0 - mu) * phi;
            let tri_a = gam_math::jet_tower::trigamma(a);
            let tri_b = gam_math::jet_tower::trigamma(b);
            let tri_phi = gam_math::jet_tower::trigamma(phi);
            let info_mu = phi * phi * (tri_a + tri_b);
            let one_minus_mu = 1.0 - mu;
            let info_phi = mu * mu * tri_a + one_minus_mu * one_minus_mu * tri_b - tri_phi;
            let mean_weight = wi * q * q * info_mu;
            let mean_response = em + score_mu / (q * info_mu);
            let disp_weight = wi * phi * phi * info_phi;
            let disp_response = ed + s_phi / (phi * info_phi);
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Tweedie { p } => {
            let mu = em.exp();
            // Precision channel models log(1/φ) ⇒ φ = exp(−η_d).
            let phi = (-ed).exp();
            let two_minus_p = 2.0 - p;
            // Mean channel: the quasi-score `(y−μ)/μ` and Fisher weight
            // `μ^{2−p}/φ` are simple closed forms (and the mean block is
            // Fisher-orthogonal to the dispersion block in this
            // parameterization), so they stay hand-written exactly as the
            // NB/Gamma mean arms do.
            let mean_weight = wi * mu.powf(two_minus_p) / phi;
            let mean_response = em + (yi - mu) / mu;
            // Dispersion channel: the η_d-space score and OBSERVED information
            // come straight off the single-expression tower seeded on `η_d`
            // (#932), so the saddlepoint/point-mass branch split, the
            // `φ = exp(−η_d)` chain and its nonlinear `∂²φ/∂η_d²` curvature
            // correction are all mechanically carried — no per-branch
            // `s_phi`/`s_eta`/`curvature_eta` hand calculus. #1591: only the
            // η_d axis is consumed, so the tower is the pruned single-axis
            // `Order2<1>` (`η_μ` enters as a constant).
            let tower = dispersion_tweedie_disp_order2(yi, em, ed, p, wi);
            let loglik = -tower.value();
            // η_d-space score and observed information off the tower, via the
            // same helper the NB/Gamma/Beta arms use (returns `(0, 0)` when the
            // prior weight is zero, so the row stays excluded below).
            let (s_eta, info_eta_raw) = tower_score_info(&tower, 0, wi);
            let curvature_eta = if yi > 0.0 { 0.5 } else { info_eta_raw };
            let disp_weight = wi * curvature_eta;
            let disp_response = ed + s_eta / curvature_eta;
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
    }
}

/// Two-block GAMLSS family for the genuine-dispersion mean families (#913).
#[derive(Clone)]
pub(crate) struct DispersionGlmLocationScaleFamily {
    pub(crate) kind: DispersionFamilyKind,
    pub(crate) y: Array1<f64>,
    pub(crate) weights: Array1<f64>,
}

impl DispersionGlmLocationScaleFamily {
    pub(crate) const BLOCK_MEAN: usize = 0;
    pub(crate) const BLOCK_DISP: usize = 1;
}

impl CustomFamily for DispersionGlmLocationScaleFamily {
    // Preserve the pre-gam#1395 behavior: the trait default flipped to OFF (the
    // flat-prior exact-Newton objective carries no Jeffreys term), so families
    // that historically armed the term by default opt back in explicitly.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        validate_block_count::<GamlssError>(self.kind.family_tag(), 2, block_states.len())?;
        let eta_mu = &block_states[Self::BLOCK_MEAN].eta;
        let eta_d = &block_states[Self::BLOCK_DISP].eta;
        let n = self.y.len();
        if eta_mu.len() != n || eta_d.len() != n || self.weights.len() != n {
            return Err(format!(
                "{} row-count mismatch: y={n}, eta_mu={}, eta_d={}, weights={}",
                self.kind.family_tag(),
                eta_mu.len(),
                eta_d.len(),
                self.weights.len()
            ));
        }
        for i in 0..n {
            validate_dispersion_row_geometry_inputs(
                self.kind,
                i,
                self.y[i],
                eta_mu[i],
                eta_d[i],
                self.weights[i],
            )?;
        }
        // `dispersion_row_kernel` is a pure, row-independent map — each row reads
        // only `y[i]`/`eta_mu[i]`/`eta_d[i]`/`weights[i]` and writes nothing
        // shared — and it is transcendental-heavy (per-row digamma/trigamma
        // derivative stacks), so the per-row evaluation is embarrassingly
        // row-parallel. Materialize the per-row kernels (in parallel for large
        // `n` when not already on a rayon worker; mirrors the
        // `row_coeff_operator` guard), then reduce SERIALLY in index order so
        // the log-likelihood sum is bit-identical to the old serial loop — no
        // float reassociation. The reduction touches no transcendentals, so the
        // parallel kernel map captures essentially all the savings.
        let kernels: Vec<DispersionRowKernel> =
            if rayon::current_thread_index().is_none() && n > DISPERSION_PARALLEL_ROW_THRESHOLD {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        dispersion_row_kernel(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                        )
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        dispersion_row_kernel(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                        )
                    })
                    .collect()
            };

        // The objective is the honest sum: with support/weight validation at
        // the public boundary and zero-weight rows short-circuited in the
        // kernel, a non-finite row term means the likelihood genuinely
        // diverges at this (β_μ, β_d) — silently dropping such rows would
        // evaluate a different dataset's objective.
        let mut log_likelihood = 0.0;
        for (i, row) in kernels.iter().enumerate() {
            validate_dispersion_row_kernel_output(i, eta_mu[i], eta_d[i], self.weights[i], row)?;
            log_likelihood += row.loglik;
            if !log_likelihood.is_finite() {
                return Err(dispersion_geometry_error(
                    i,
                    "dispersion-family cumulative log likelihood",
                    eta_mu[i],
                    log_likelihood,
                ));
            }
        }
        let mean_weights = Array1::from_iter(kernels.iter().map(|row| row.mean_weight));
        let mean_response = Array1::from_iter(kernels.iter().map(|row| row.mean_response));
        let disp_weights = Array1::from_iter(kernels.iter().map(|row| row.disp_weight));
        let disp_response = Array1::from_iter(kernels.iter().map(|row| row.disp_response));
        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::diagonal_checked(mean_response, mean_weights)?,
                BlockWorkingSet::diagonal_checked(disp_response, disp_weights)?,
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        validate_block_count::<GamlssError>(self.kind.family_tag(), 2, block_states.len())?;
        let eta_mu = &block_states[Self::BLOCK_MEAN].eta;
        let eta_d = &block_states[Self::BLOCK_DISP].eta;
        let n = self.y.len();
        if eta_mu.len() != n || eta_d.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{} log-likelihood row-count mismatch: y={n}, eta_mu={}, eta_d={}, weights={}",
                    self.kind.family_tag(),
                    eta_mu.len(),
                    eta_d.len(),
                    self.weights.len()
                ),
            }
            .into());
        }
        for i in 0..n {
            validate_dispersion_row_geometry_inputs(
                self.kind,
                i,
                self.y[i],
                eta_mu[i],
                eta_d[i],
                self.weights[i],
            )?;
        }
        // #1591 prune: the objective needs only the row log-likelihood, so each
        // row evaluates the value channel alone (`to_bits`-identical to
        // `dispersion_row_kernel(..).loglik`), skipping every gradient/Hessian
        // and digamma/trigamma derivative-stack evaluation. That value-only map
        // is still a pure, row-independent per-row `ln_gamma` evaluation, so it
        // is row-parallel; fan it out (large `n`, off a rayon worker) into a
        // per-row buffer, then sum SERIALLY in index order to keep the objective
        // bit-identical to the serial loop (no float reassociation).
        let per_row: Vec<f64> =
            if rayon::current_thread_index().is_none() && n > DISPERSION_PARALLEL_ROW_THRESHOLD {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        dispersion_row_loglik(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                        )
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        dispersion_row_loglik(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                        )
                    })
                    .collect()
            };
        // Honest sum — see `evaluate`: non-finite row terms signal genuine
        // divergence and must reach the caller, not be silently dropped.
        let mut ll = 0.0;
        for (i, loglik) in per_row.into_iter().enumerate() {
            if !loglik.is_finite() {
                return Err(dispersion_geometry_error(
                    i,
                    "dispersion-family row log likelihood",
                    eta_mu[i],
                    loglik,
                ));
            }
            ll += loglik;
            if !ll.is_finite() {
                return Err(dispersion_geometry_error(
                    i,
                    "dispersion-family cumulative log likelihood",
                    eta_mu[i],
                    ll,
                ));
            }
        }
        Ok(ll)
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        crate::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    /// Exact joint coefficient-space Hessian `H_L = -∇²log L` in flattened
    /// `[mean | log-precision]` block order.
    ///
    /// All four members assemble `Xᵀ diag(W) X` blocks from the per-row
    /// OBSERVED η-space second derivatives
    /// (`dispersion_row_observed_hessian_weights`): the full mean-link and
    /// precision-link chains, the inverse-link second-derivative terms, and
    /// the mean/dispersion cross curvature are all carried exactly by the
    /// `Order2<2>` jet tower. This is deliberately NOT the Fisher-scoring
    /// working-weight matrix that `evaluate` returns for the inner IRLS —
    /// expected information is a legitimate inner-solve preconditioner (the
    /// working response keeps the exact score, so the optimum is unchanged),
    /// but LAML/REML log-determinants, Jeffreys corrections, EDF, and the
    /// joint posterior covariance all require the observed Hessian. The
    /// Fisher-orthogonal members (NB2 / Gamma / Tweedie) have EXPECTED cross
    /// information zero, yet their per-row observed cross curvature is
    /// nonzero (Gamma at `y=4, μ=2, ν=3`: `∂²NLL/∂η_μ∂η_ν = −3`), so the
    /// assembled `H_L` is genuinely coupled for every member.
    ///
    /// Returning this dense `H_L` — rather than `None` — is what lets the
    /// multi-block outer-REML path (`build_joint_hessian_closures` →
    /// `joint_outer_evaluate`) and the joint posterior covariance
    /// (`compute_joint_covariance`) run for these families instead of failing
    /// the "multi-block families must provide a joint outer path" gate and
    /// silently escalating to a degraded ρ-seed fit with no covariance/EDF
    /// (gam#1119).
    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>(self.kind.family_tag(), 2, block_states.len())?;
        if specs.len() != 2 {
            return Err(format!(
                "{} exact joint Hessian expects 2 specs, got {}",
                self.kind.family_tag(),
                specs.len()
            ));
        }
        let eta_mu = &block_states[Self::BLOCK_MEAN].eta;
        let eta_d = &block_states[Self::BLOCK_DISP].eta;
        let n = self.y.len();
        if eta_mu.len() != n || eta_d.len() != n || self.weights.len() != n {
            return Err(format!(
                "{} exact joint Hessian row-count mismatch: y={n}, eta_mu={}, eta_d={}, weights={}",
                self.kind.family_tag(),
                eta_mu.len(),
                eta_d.len(),
                self.weights.len()
            ));
        }
        for i in 0..n {
            validate_dispersion_row_geometry_inputs(
                self.kind,
                i,
                self.y[i],
                eta_mu[i],
                eta_d[i],
                self.weights[i],
            )?;
        }

        // Per-row observed `(∂²/∂η_μ², ∂²/∂η_μ∂η_d, ∂²/∂η_d²)` weights — one
        // full `Order2<2>` η-space tower per row. Row-independent, so fan it
        // out for large `n` (off a rayon worker) into a per-row buffer —
        // index-ordered, no reduction, so byte-identical to the serial map.
        let observed: Vec<(f64, f64, f64)> =
            if rayon::current_thread_index().is_none() && n > DISPERSION_PARALLEL_ROW_THRESHOLD {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        dispersion_row_observed_hessian_weights(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                        )
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        dispersion_row_observed_hessian_weights(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                        )
                    })
                    .collect()
            };
        for (i, &(h_mm, h_md, h_dd)) in observed.iter().enumerate() {
            for (quantity, eta, value) in [
                ("dispersion-family observed mean curvature", eta_mu[i], h_mm),
                (
                    "dispersion-family observed cross curvature",
                    eta_mu[i],
                    h_md,
                ),
                (
                    "dispersion-family observed precision curvature",
                    eta_d[i],
                    h_dd,
                ),
            ] {
                if !value.is_finite() {
                    return Err(dispersion_geometry_error(i, quantity, eta, value));
                }
            }
        }
        let mean_weights = Array1::from_shape_fn(n, |i| observed[i].0);
        let cross_weights = Array1::from_shape_fn(n, |i| observed[i].1);
        let disp_weights = Array1::from_shape_fn(n, |i| observed[i].2);
        let mean_spec = &specs[Self::BLOCK_MEAN];
        let disp_spec = &specs[Self::BLOCK_DISP];
        if mean_spec.design.nrows() != n || disp_spec.design.nrows() != n {
            return Err(format!(
                "{} exact joint Hessian design row mismatch: y={n}, mean rows={}, precision rows={}",
                self.kind.family_tag(),
                mean_spec.design.nrows(),
                disp_spec.design.nrows()
            ));
        }
        let p_mean = mean_spec.design.ncols();
        let p_disp = disp_spec.design.ncols();
        if block_states[Self::BLOCK_MEAN].beta.len() != p_mean
            || block_states[Self::BLOCK_DISP].beta.len() != p_disp
        {
            return Err(format!(
                "{} exact joint Hessian beta/design mismatch: mean beta {} vs cols {}, precision beta {} vs cols {}",
                self.kind.family_tag(),
                block_states[Self::BLOCK_MEAN].beta.len(),
                p_mean,
                block_states[Self::BLOCK_DISP].beta.len(),
                p_disp
            ));
        }

        let h_mean = xt_diag_x_design(&mean_spec.design, &mean_weights)?;
        let h_cross = xt_diag_y_design(&mean_spec.design, &cross_weights, &disp_spec.design)?;
        let h_disp = xt_diag_x_design(&disp_spec.design, &disp_weights)?;
        let total = p_mean + p_disp;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..p_mean, 0..p_mean]).assign(&h_mean);
        h.slice_mut(s![0..p_mean, p_mean..total]).assign(&h_cross);
        h.slice_mut(s![p_mean..total, p_mean..total])
            .assign(&h_disp);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    /// The joint likelihood Hessian is NOT block-diagonal for any member:
    /// even the Fisher-orthogonal parameterizations — NB2 `(μ, θ)`, Gamma
    /// shape `ν`, Tweedie `log(1/φ)` — have zero EXPECTED cross information
    /// but nonzero per-row OBSERVED cross curvature `∂²NLL/∂η_μ∂η_d`
    /// (Gamma at `y=4, μ=2, ν=3` has `ν(1−y/μ) = −3`). The former
    /// `uncoupled = true` shortcut for these members made the outer calculus
    /// consume a block-diagonal matrix as if it were the exact Hessian.
    /// The explicit-joint-Hessian marker below is what routes the outer
    /// dispatch to the trusted coupled override instead (gam#1119).
    fn likelihood_blocks_uncoupled(&self) -> bool {
        false
    }

    /// `exact_newton_joint_hessian_with_specs` above returns the true coupled
    /// observed joint Hessian for every member, so mark it explicit for the
    /// outer-REML trust dispatch.
    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// The mean and precision working weights couple across both blocks, which
    /// the block-local diagonal drift hook cannot represent, so decline the
    /// dense outer Hessian capability whenever the actual two-block (or
    /// larger) geometry is in play; a degenerate single-block probe — there
    /// is no cross-block coupling to reject — keeps the trait default's
    /// availability verdict.
    ///
    /// The override still validates the block-spec slice it is handed (the
    /// same consistency check the trait default's assertion bottoms out in)
    /// so a malformed probe is reported here rather than downstream.
    fn outer_hyper_hessian_dense_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert!(
            crate::custom_family::validate_blockspec_consistency(specs).is_ok(),
            "DispersionGlmLocationScale outer hyper-Hessian dense availability: \
             inconsistent parameter block specs"
        );
        specs.len() < 2
    }
}

/// Term spec consumed by [`fit_dispersion_glm_location_scale_terms`]; mirrors
/// [`GaussianLocationScaleTermSpec`](super::GaussianLocationScaleTermSpec) with
/// the dispersion channel in place of the Gaussian log-σ channel.
pub struct DispersionGlmLocationScaleTermSpec {
    pub kind: DispersionFamilyKind,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub log_dispspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
    pub log_disp_offset: Array1<f64>,
}

pub(crate) struct DispersionGlmLocationScaleTermBuilder {
    pub(crate) kind: DispersionFamilyKind,
    pub(crate) y: Array1<f64>,
    pub(crate) weights: Array1<f64>,
    pub(crate) meanspec: TermCollectionSpec,
    pub(crate) noisespec: TermCollectionSpec,
    pub(crate) mean_offset: Array1<f64>,
    pub(crate) noise_offset: Array1<f64>,
}

/// Warm start for a dispersion location-scale fit: project a link-transformed
/// response onto the mean block and seed the log-precision block at a constant
/// (precision ≈ 1) baseline. The block-cyclic IRLS then refines both jointly.
pub(crate) fn dispersion_location_scale_warm_start(
    kind: DispersionFamilyKind,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    mean_block: &ParameterBlockSpec,
    disp_block: &ParameterBlockSpec,
    mean_beta_hint: Option<&Array1<f64>>,
    disp_beta_hint: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let ridge_floor = 1e-10;
    let mean_beta = if let Some(beta) = mean_beta_hint {
        beta.clone()
    } else {
        let target = Array1::from_shape_fn(y.len(), |i| {
            if kind.mean_is_logit() {
                let yi = y[i].clamp(1e-3, 1.0 - 1e-3);
                (yi / (1.0 - yi)).ln()
            } else {
                // log mean link; the +0.1 keeps zero counts finite.
                (y[i].max(0.0) + 0.1).ln()
            }
        });
        solve_penalizedweighted_projection(
            &mean_block.design,
            &mean_block.offset,
            &target,
            weights,
            &mean_block.penalties,
            &mean_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    let disp_beta = if let Some(beta) = disp_beta_hint {
        beta.clone()
    } else {
        // Seed the precision block from a smoothed method-of-moments surface
        // rather than the old flat η_d=0 constant.  A single observation cannot
        // identify its own variance, but for the Fisher-orthogonal dispersion
        // members the residual-squared moment contains the correct first-order
        // signal:
        //
        //   Gamma:   Var(Y)=μ²/ν              ⇒ log ν     ≈ log(μ²/e²)
        //   NB2:     Var(Y)=μ+μ²/θ            ⇒ log θ     ≈ log(μ²/(e²-μ))
        //   Tweedie: Var(Y)=φ μ^p, η_d=log1/φ ⇒ η_d       ≈ log(μ^p/e²)
        //
        // The targets are deliberately conservative (finite residual floor,
        // precision cap, and no fixture-specific constants): they only give the
        // block-cyclic likelihood solve a correctly-signed non-flat starting
        // surface, while the final estimate is still the penalized joint MLE.
        let mean_eta = mean_block.design.apply(&mean_beta) + &mean_block.offset;
        let target = Array1::from_shape_fn(y.len(), |i| {
            dispersion_moment_log_precision_seed(kind, y[i], mean_eta[i])
        });
        solve_penalizedweighted_projection(
            &disp_block.design,
            &disp_block.offset,
            &target,
            weights,
            &disp_block.penalties,
            &disp_block.initial_log_lambdas,
            ridge_floor,
        )?
    };
    Ok((mean_beta, disp_beta))
}

#[inline]
fn dispersion_moment_log_precision_seed(kind: DispersionFamilyKind, yi: f64, eta_mu: f64) -> f64 {
    const LOG_PRECISION_FLOOR: f64 = -10.0;
    const LOG_PRECISION_CEILING: f64 = 10.0;
    let em = eta_mu;
    let raw = match kind {
        DispersionFamilyKind::Beta => {
            // Beta's mean and precision scores are not Fisher-orthogonal in
            // the (logit μ, log φ) parameterization.  Per-row residual moments
            // therefore make a poor block-cyclic seed: an outlying y near 0/1
            // can imply a near-zero φ and pull the coupled mean block onto the
            // boundary before the joint likelihood has had a chance to settle.
            // Keep the neutral precision seed for this one coupled member; the
            // exact Beta cross-Hessian below still drives the joint solve and
            // covariance with the coherent two-block likelihood geometry.
            0.0
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp().max(1e-12);
            let e2 = (yi - mu).powi(2).max(1e-8 * mu * mu);
            (mu * mu / e2).max(1e-6).ln()
        }
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp().max(1e-12);
            let e2 = (yi - mu).powi(2);
            let excess = (e2 - mu).max(1e-6 * (mu + mu * mu));
            (mu * mu / excess).max(1e-6).ln()
        }
        DispersionFamilyKind::Tweedie { p } => {
            let mu = em.exp().max(1e-12);
            let e2 = (yi - mu).powi(2).max(1e-8 * mu.powf(p));
            (mu.powf(p) / e2).max(1e-6).ln()
        }
    };
    raw.clamp(LOG_PRECISION_FLOOR, LOG_PRECISION_CEILING)
}

impl LocationScaleFamilyBuilder for DispersionGlmLocationScaleTermBuilder {
    type Family = DispersionGlmLocationScaleFamily;

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
    }

    fn noise_penalty_count(&self, noise_design: &TermCollectionDesign) -> usize {
        // Mirror the Gaussian/Binomial scale block: a full-span shrinkage
        // penalty pins the log-precision nullspace so REML does not optimise
        // the dispersion smoothing on a flat surface.
        noise_design.penalties.len() + 1
    }

    fn build_blocks(
        &self,
        theta: &Array1<f64>,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
        mean_beta_hint: Option<Array1<f64>>,
        noise_beta_hint: Option<Array1<f64>>,
    ) -> Result<Vec<ParameterBlockSpec>, String> {
        let layout = GamlssLambdaLayout::two_block(
            mean_design.penalties.len(),
            self.noise_penalty_count(noise_design),
        );
        layout.validate_theta_len(theta.len(), "dispersion location-scale")?;

        let mean_offset = mean_design
            .compose_offset(self.mean_offset.view(), "dispersion location-scale mean")
            .map_err(|error| error.to_string())?;
        let noise_offset = noise_design
            .compose_offset(
                self.noise_offset.view(),
                "dispersion location-scale log-precision",
            )
            .map_err(|error| error.to_string())?;
        let mut meanspec = build_location_scale_block(
            "mu",
            mean_design.design.clone(),
            mean_offset,
            mean_design.penalties_as_penalty_matrix(),
            mean_design.nullspace_dims.clone(),
            layout.mean_from(theta),
            mean_beta_hint,
            0,
            LOCATION_SCALE_N_OUTPUTS,
            "DispersionLocationScale::build_blocks: mu",
        )?;

        let p_disp = noise_design.design.ncols();
        let mut disp_penalties = noise_design.penalties_as_penalty_matrix();
        disp_penalties.push(PenaltyMatrix::Dense(identity_penalty(p_disp)));
        let mut disp_nullspace = noise_design.nullspace_dims.clone();
        disp_nullspace.push(0);
        let mut dispspec = build_location_scale_block(
            "log_precision",
            noise_design.design.clone(),
            noise_offset,
            disp_penalties,
            disp_nullspace,
            layout.noise_from(theta),
            noise_beta_hint,
            1,
            LOCATION_SCALE_N_OUTPUTS,
            "DispersionLocationScale::build_blocks: log_precision",
        )?;

        if meanspec.initial_beta.is_none() || dispspec.initial_beta.is_none() {
            let (mean_beta0, disp_beta0) = dispersion_location_scale_warm_start(
                self.kind,
                &self.y,
                &self.weights,
                &meanspec,
                &dispspec,
                meanspec.initial_beta.as_ref(),
                dispspec.initial_beta.as_ref(),
            )?;
            if meanspec.initial_beta.is_none() {
                meanspec.initial_beta = Some(mean_beta0);
            }
            if dispspec.initial_beta.is_none() {
                dispspec.initial_beta = Some(disp_beta0);
            }
        }

        Ok(vec![meanspec, dispspec])
    }

    fn build_family(
        &self,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Self::Family {
        // The family stores y/weights/kind directly and does not need the
        // designs at construction time, but the row geometry of the offered
        // designs is the only cross-check that ties this family back to the
        // builder's data — assert it before handing the family to the engine
        // so a misaligned design surfaces here rather than downstream in the
        // inner solver.
        assert_eq!(
            mean_design.design.nrows(),
            self.y.len(),
            "DispersionGlmLocationScale::build_family: mean design row count must match y"
        );
        assert_eq!(
            noise_design.design.nrows(),
            self.y.len(),
            "DispersionGlmLocationScale::build_family: noise design row count must match y"
        );
        DispersionGlmLocationScaleFamily {
            kind: self.kind,
            y: self.y.clone(),
            weights: self.weights.clone(),
        }
    }

    fn extract_primary_betas(
        &self,
        fit: &UnifiedFitResult,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mean_beta = fit
            .block_states
            .get(DispersionGlmLocationScaleFamily::BLOCK_MEAN)
            .ok_or_else(|| "missing dispersion mean block state".to_string())?
            .beta
            .clone();
        let disp_beta = fit
            .block_states
            .get(DispersionGlmLocationScaleFamily::BLOCK_DISP)
            .ok_or_else(|| "missing dispersion log-precision block state".to_string())?
            .beta
            .clone();
        Ok((mean_beta, disp_beta))
    }

    fn build_psiderivative_blocks(
        &self,
        data: ndarray::ArrayView2<'_, f64>,
        meanspec: &TermCollectionSpec,
        noisespec: &TermCollectionSpec,
        mean_design: &TermCollectionDesign,
        noise_design: &TermCollectionDesign,
    ) -> Result<Vec<Vec<CustomFamilyBlockPsiDerivative>>, String> {
        // The dispersion location-scale families do not expose the complete
        // coupled higher-order calculus needed for analytic spatial psi
        // derivatives. The public fit boundary rejects enabled κ/ψ requests;
        // if a future caller bypasses that boundary, return a real diagnostic
        // rather than a sentinel. Include the exact data/design shape so the
        // invalid call is diagnosable from the error string alone.
        Err(format!(
            "dispersion location-scale ({:?}) does not implement analytic spatial \
             psi derivatives; the κ/ψ joint optimizer must be explicitly disabled before \
             this builder is consulted. Called with data {n_rows}×{n_cols}, mean \
             spec (linear={mean_lin}, random={mean_re}, smooth={mean_sm}), noise \
             spec (linear={noise_lin}, random={noise_re}, smooth={noise_sm}), \
             mean design cols={mean_p}, noise design cols={noise_p}",
            self.kind,
            n_rows = data.nrows(),
            n_cols = data.ncols(),
            mean_lin = meanspec.linear_terms.len(),
            mean_re = meanspec.random_effect_terms.len(),
            mean_sm = meanspec.smooth_terms.len(),
            noise_lin = noisespec.linear_terms.len(),
            noise_re = noisespec.random_effect_terms.len(),
            noise_sm = noisespec.smooth_terms.len(),
            mean_p = mean_design.design.ncols(),
            noise_p = noise_design.design.ncols(),
        ))
    }
}

/// Validate family support and prior weights at the public boundary.
///
/// The row kernels evaluate the requested likelihood verbatim; they do not
/// winsorize out-of-range Beta responses, floor nonpositive Gamma responses,
/// zero negative Tweedie responses, accept noninteger negative-binomial
/// counts, or clamp negative weights — all of those silently fit a DIFFERENT
/// dataset than the one supplied. Invalid rows must therefore be rejected
/// here. Rows with an exactly-zero prior weight are exempt from the response
/// support check (they are excluded from the likelihood entirely), which is
/// the supported way to carry deliberately masked observations.
fn validate_dispersion_family_data(
    kind: DispersionFamilyKind,
    y: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<(), String> {
    if y.len() != weights.len() {
        return Err(format!(
            "{}: response/weights length mismatch: y={}, weights={}",
            kind.family_tag(),
            y.len(),
            weights.len()
        ));
    }
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(format!(
                "{}: prior weights must be finite and non-negative; got weights[{i}] = {w}",
                kind.family_tag()
            ));
        }
    }
    for (i, &yi) in y.iter().enumerate() {
        if weights[i] == 0.0 {
            continue;
        }
        let (ok, requirement) = match kind {
            DispersionFamilyKind::NegativeBinomial => (
                yi.is_finite() && yi >= 0.0 && yi.fract() == 0.0,
                "a finite non-negative integer count",
            ),
            DispersionFamilyKind::Gamma => (yi.is_finite() && yi > 0.0, "finite and > 0"),
            DispersionFamilyKind::Beta => (
                yi.is_finite() && yi > 0.0 && yi < 1.0,
                "finite and strictly inside (0, 1)",
            ),
            DispersionFamilyKind::Tweedie { .. } => {
                (yi.is_finite() && yi >= 0.0, "finite and >= 0")
            }
        };
        if !ok {
            return Err(format!(
                "{}: response outside family support at row {i}: y = {yi} (must be {requirement}; \
                 set the row's prior weight to 0 to exclude it)",
                kind.family_tag()
            ));
        }
    }
    Ok(())
}

/// Reject a spatial-hyperparameter request that this coupled family cannot
/// differentiate exactly.
///
/// The shared spatial bridge can provide exact design/penalty jets
/// (`X_psi`, `S_psi`, and their second derivatives), but the dispersion
/// likelihood's observed two-block Hessian also moves through both fitted
/// predictors. An exact profiled LAML gradient therefore additionally needs
/// the coupled `D_beta H` and `D_beta H_psi` contractions. This family does not
/// expose those higher-order row jets yet. Silently switching `enabled` off
/// changes the requested model; exposing the existing typed configuration
/// error makes fixed geometry an explicit caller choice instead.
fn validate_dispersion_spatial_hyperparameter_request(
    kind: DispersionFamilyKind,
    meanspec: &TermCollectionSpec,
    log_dispspec: &TermCollectionSpec,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<(), GamlssError> {
    if !kappa_options.enabled {
        return Ok(());
    }

    let unfrozen_terms = |spec: &TermCollectionSpec| -> Vec<usize> {
        spatial_length_scale_term_indices(spec)
            .into_iter()
            .filter(|&idx| {
                // On the incoming (pre-build) spec, `0.0` is the Matérn
                // auto-initialization sentinel, not a user-locked scale.
                // A positive scalar scale freezes only an isotropic axis;
                // per-axis psi coordinates remain an optimization request.
                let scalar_scale_is_locked = get_spatial_length_scale(spec, idx)
                    .is_some_and(|scale| scale.is_finite() && scale > 0.0)
                    && !spatial_term_uses_per_axis_psi(spec, idx);
                !scalar_scale_is_locked
            })
            .collect()
    };
    let mean_terms = unfrozen_terms(meanspec);
    let log_disp_terms = unfrozen_terms(log_dispspec);
    if mean_terms.is_empty() && log_disp_terms.is_empty() {
        return Ok(());
    }
    let term_names = |spec: &TermCollectionSpec, indices: &[usize]| -> Vec<String> {
        indices
            .iter()
            .filter_map(|&idx| spec.smooth_terms.get(idx).map(|term| term.name.clone()))
            .collect()
    };
    Err(GamlssError::UnsupportedConfiguration {
        reason: format!(
            "dispersion location-scale ({kind:?}) cannot optimize spatial hyperparameters: \
             exact coupled D_beta H and D_beta H_psi derivatives are unavailable for \
             unfrozen spatial terms (mean={:?}, log_precision={:?}). Supply locked spatial \
             geometry or explicitly set spatial length-scale optimization enabled=false; the \
             fitter will not silently freeze a requested spatial optimization",
            term_names(meanspec, &mean_terms),
            term_names(log_dispspec, &log_disp_terms),
        ),
    })
}

/// Fit a dispersion-channel GAMLSS location-scale model (#913). All four
/// genuine-dispersion mean families share this single entry; the per-family
/// likelihood lives in [`dispersion_row_kernel`].
pub fn fit_dispersion_glm_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: DispersionGlmLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, String> {
    if let DispersionFamilyKind::Tweedie { p } = spec.kind {
        if !(p.is_finite() && p > 1.0 && p < 2.0) {
            return Err(format!(
                "Tweedie location-scale requires a variance power strictly in (1, 2); got p={p}"
            ));
        }
    }
    validate_dispersion_family_data(spec.kind, &spec.y, &spec.weights)?;
    validate_dispersion_spatial_hyperparameter_request(
        spec.kind,
        &spec.meanspec,
        &spec.log_dispspec,
        kappa_options,
    )?;
    // A dispersion location-scale model is an inherently *predictable* model:
    // posterior-mean prediction (the response-scale predict path the CLI/FFI
    // drive) needs the joint `(β_μ, β_d)` posterior covariance, and so does the
    // reported total EDF / coefficient SEs. The block-diagonal joint Hessian is
    // always assembled here (`exact_newton_joint_hessian_with_specs` →
    // `compute_joint_covariance`, which for this family's `RidgedQuadraticReml`
    // outer objective uses the never-erroring SPD-retry → positive-part
    // pseudo-inverse), so we can — and must — request the covariance
    // unconditionally rather than leaving `covariance_conditional = None`
    // whenever the outer optimizer happens to *converge* (the only family-
    // independent reason NB sometimes populated covariance was that it escalated
    // into the never-fail posterior-sampling rung, while a cleanly-converged
    // Gamma/Tweedie fit took the `!options.compute_covariance ⇒ None` early
    // return and stranded its covariance/EDF — gam#1119). Forcing the flag here
    // makes all four genuine-dispersion mean families assemble the joint
    // covariance + EDF deterministically, exactly as a predictable model
    // requires.
    let mut options = options.clone();
    options.compute_covariance = true;
    fit_location_scale_terms(
        data,
        DispersionGlmLocationScaleTermBuilder {
            kind: spec.kind,
            y: spec.y,
            weights: spec.weights,
            meanspec: spec.meanspec,
            noisespec: spec.log_dispspec,
            mean_offset: spec.mean_offset,
            noise_offset: spec.log_disp_offset,
        },
        &options,
        kappa_options,
    )
}

#[cfg(test)]
mod tests {
    use super::test_support::{dispersion_gamma_nll_order2, dispersion_nb_nll_order2};
    use super::*;
    use crate::gamlss::test_support::dispersion_tweedie_nll_generic;
    use gam_math::nested_dual::JetField;

    #[test]
    fn saved_alo_gamma_row_geometry_matches_closed_form_and_keeps_meat_distinct() {
        let y = 4.0;
        let mu: f64 = 2.0;
        let nu: f64 = 3.0;
        let weight = 1.7;
        let geometry = dispersion_alo_row_geometry(
            DispersionFamilyKind::Gamma,
            0,
            y,
            mu.ln(),
            nu.ln(),
            weight,
        )
        .expect("Gamma row geometry must be representable");

        let ratio = y / mu;
        let a = gam_math::jet_tower::digamma(nu) - nu.ln() - 1.0 + mu.ln() - y.ln() + ratio;
        let expected_score = [weight * nu * (1.0 - ratio), weight * nu * a];
        let expected_hessian = [
            [weight * nu * ratio, weight * nu * (1.0 - ratio)],
            [
                weight * nu * (1.0 - ratio),
                weight * nu * (a + nu * gam_math::jet_tower::trigamma(nu) - 1.0),
            ],
        ];
        for coordinate in 0..2 {
            assert_close(
                "Gamma ALO score",
                geometry.nll_score[coordinate],
                expected_score[coordinate],
                2e-12,
            );
            for other in 0..2 {
                assert_close(
                    "Gamma ALO observed Hessian",
                    geometry.observed_hessian[coordinate][other],
                    expected_hessian[coordinate][other],
                    2e-12,
                );
            }
        }

        let score_meat = [
            [
                expected_score[0] * expected_score[0],
                expected_score[0] * expected_score[1],
            ],
            [
                expected_score[1] * expected_score[0],
                expected_score[1] * expected_score[1],
            ],
        ];
        assert_ne!(
            geometry.observed_hessian, score_meat,
            "the deletion curvature must not be replaced by score covariance"
        );
    }

    /// Order-≤1 `ln Γ` compose: only the value (`d[0] = lnΓ`) and first-derivative
    /// (`d[1] = ψ`) stack slots are consumed by an [`Order1`] jet, so we evaluate
    /// ONLY `lnΓ` and `ψ` — never `ψ′` (trigamma). This is the value/gradient
    /// twin of [`order2_ln_gamma`]; its `(value, g)` channels are bit-identical to
    /// that function's order-≤1 channels because [`Order1`] runs the same Leibniz /
    /// Faà-di-Bruno value+gradient float ops as [`Order2`] (doc on `Order1`).
    #[inline]
    fn order1_ln_gamma<const K: usize>(
        x: &gam_math::jet_scalar::Order1<K>,
    ) -> gam_math::jet_scalar::Order1<K> {
        x.compose_unary([
            ln_gamma(x.v),
            gam_math::jet_tower::digamma(x.v),
            0.0,
            0.0,
            0.0,
        ])
    }

    /// Value+gradient-only NB2 dispersion tower: `θ` is the sole jet variable
    /// (axis 0), `μ` a constant. `value`/`g[0]` reproduce the consumed
    /// `value`/`g[0]` of [`dispersion_nb_disp_order2`] bit-for-bit, but as an
    /// [`Order1`] jet it never evaluates the trigamma (`ψ′`) that the discarded
    /// observed-Hessian channel would need. `h[0][0]` is pure discarded work —
    /// dropping it here removes two `ψ′` evaluations (at `θ+y` and `θ`) per
    /// evaluation on top of the tensor-shrink.
    #[inline]
    fn dispersion_nb_disp_order1(
        yi: f64,
        mu_value: f64,
        theta_value: f64,
        wi: f64,
    ) -> gam_math::jet_scalar::Order1<1> {
        type O1 = gam_math::jet_scalar::Order1<1>;

        let mu = O1::constant(mu_value);
        let theta = O1::variable(theta_value, 0);
        let tpm = theta.add(&mu);
        let theta_plus_y = theta.add(&O1::constant(yi));
        let loglik = order1_ln_gamma(&theta_plus_y)
            .sub(&order1_ln_gamma(&theta))
            .sub(&O1::constant(ln_gamma(yi + 1.0)))
            .add(&theta.mul(&theta.ln()))
            .sub(&theta.mul(&tpm.ln()))
            .add(&mu.ln().scale(yi))
            .sub(&tpm.ln().scale(yi));
        loglik.scale(-wi)
    }

    /// Pruned single-axis NB2 dispersion tower: `θ` is the sole jet variable
    /// (axis 0), `μ` a constant. `value`/`g[0]`/`h[0][0]` reproduce the consumed
    /// `value`/`g[1]`/`h[1][1]` of `dispersion_nb_nll_order2` bit-for-bit. The
    /// `Order2` oracle pin for `prune_towers_match_dense_all_channels`, and the
    /// `Order1` oracle target for `dispersion_nb_disp_order1` above.
    #[inline]
    fn dispersion_nb_disp_order2(
        yi: f64,
        mu_value: f64,
        theta_value: f64,
        wi: f64,
    ) -> gam_math::jet_scalar::Order2<1> {
        use gam_math::jet_scalar::JetScalar;
        use statrs::function::gamma::ln_gamma;
        type O1 = gam_math::jet_scalar::Order2<1>;

        let mu = O1::constant(mu_value);
        let theta = O1::variable(theta_value, 0);
        let tpm = theta.add(&mu);
        let theta_plus_y = theta.add(&O1::constant(yi));
        let loglik = order2_ln_gamma(&theta_plus_y)
            .sub(&order2_ln_gamma(&theta))
            .sub(&O1::constant(ln_gamma(yi + 1.0)))
            .add(&theta.mul(&theta.ln()))
            .sub(&theta.mul(&tpm.ln()))
            .add(&mu.ln().scale(yi))
            .sub(&tpm.ln().scale(yi));
        loglik.scale(-wi)
    }

    pub(crate) fn beta_fisher_cross_info_mu_phi(mu: f64, phi: f64) -> f64 {
        let a = mu * phi;
        let b = (1.0 - mu) * phi;
        phi * (mu * gam_math::jet_tower::trigamma_derivative_stack(a)[0]
            - (1.0 - mu) * gam_math::jet_tower::trigamma_derivative_stack(b)[0])
    }

    pub(crate) fn assert_close(label: &str, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "{label}: got {got:.12e}, want {want:.12e}, |diff|={:.3e}",
            (got - want).abs()
        );
    }

    #[test]
    fn spatial_hyperparameter_request_is_a_typed_error_until_explicitly_frozen() {
        let locked_meanspec = crate::gamlss::tests::simple_matern_term_collection(&[0, 1], 0.6);
        let mut meanspec = locked_meanspec.clone();
        let gam_terms::smooth::SmoothBasisSpec::Matern { spec, .. } =
            &mut meanspec.smooth_terms[0].basis
        else {
            panic!("test fixture must contain a Matérn term");
        };
        spec.aniso_log_scales = Some(vec![0.0, 0.0]);
        let log_dispspec = crate::gamlss::tests::empty_term_collection();
        let enabled = SpatialLengthScaleOptimizationOptions::default();

        let error = validate_dispersion_spatial_hyperparameter_request(
            DispersionFamilyKind::Gamma,
            &meanspec,
            &log_dispspec,
            &enabled,
        )
        .expect_err("enabled dispersion spatial optimization must be rejected");
        assert!(matches!(
            error,
            GamlssError::UnsupportedConfiguration { .. }
        ));

        let n = 8;
        let public_error = match fit_dispersion_glm_location_scale_terms(
            Array2::zeros((n, 2)).view(),
            DispersionGlmLocationScaleTermSpec {
                kind: DispersionFamilyKind::Gamma,
                y: Array1::from_elem(n, 1.0),
                weights: Array1::from_elem(n, 1.0),
                meanspec: meanspec.clone(),
                log_dispspec: log_dispspec.clone(),
                mean_offset: Array1::zeros(n),
                log_disp_offset: Array1::zeros(n),
            },
            &BlockwiseFitOptions::default(),
            &enabled,
        ) {
            Ok(_) => panic!("public fit must not silently freeze spatial optimization"),
            Err(error) => error,
        };
        assert!(public_error.contains("will not silently freeze"));

        validate_dispersion_spatial_hyperparameter_request(
            DispersionFamilyKind::Gamma,
            &locked_meanspec,
            &log_dispspec,
            &enabled,
        )
        .expect("a caller-supplied locked spatial scale is explicit frozen geometry");

        let auto_meanspec = crate::gamlss::tests::simple_matern_term_collection(&[0, 1], 0.0);
        assert!(matches!(
            validate_dispersion_spatial_hyperparameter_request(
                DispersionFamilyKind::Gamma,
                &auto_meanspec,
                &log_dispspec,
                &enabled,
            ),
            Err(GamlssError::UnsupportedConfiguration { .. })
        ));

        let mut frozen = enabled;
        frozen.enabled = false;
        validate_dispersion_spatial_hyperparameter_request(
            DispersionFamilyKind::Gamma,
            &meanspec,
            &log_dispspec,
            &frozen,
        )
        .expect("an explicit frozen-geometry request is supported");

        validate_dispersion_spatial_hyperparameter_request(
            DispersionFamilyKind::Gamma,
            &log_dispspec,
            &log_dispspec,
            &SpatialLengthScaleOptimizationOptions::default(),
        )
        .expect("enabled spatial optimization is irrelevant without spatial coordinates");
    }

    #[test]
    pub(crate) fn beta_tower_mixed_channel_matches_cross_information_formula() {
        let mu = 0.1;
        let phi = 10.0;
        let a = mu * phi;
        let b = (1.0 - mu) * phi;
        let digamma_a = gam_math::jet_tower::digamma_derivative_stack(a)[0];
        let digamma_b = gam_math::jet_tower::digamma_derivative_stack(b)[0];
        let score_neutral_y = 1.0 / (1.0 + (-(digamma_a - digamma_b)).exp());

        let tower = dispersion_beta_nll_order2(score_neutral_y, mu, phi, 1.0);
        let trigamma_a = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        let trigamma_b = gam_math::jet_tower::trigamma_derivative_stack(b)[0];
        let analytic = phi * (mu * trigamma_a - (1.0 - mu) * trigamma_b);
        let helper = beta_fisher_cross_info_mu_phi(mu, phi);

        assert!(
            analytic > 0.58,
            "audit example should have visibly nonzero cross information, got {analytic}"
        );
        assert_close("helper cross information", helper, analytic, 1e-12);
        assert_close("tower mixed channel", tower.h()[0][1], analytic, 1e-8);

        // η-space chain: ∂²NLL/∂η_μ∂η_d = q·φ·f_μφ with q = dμ/dη_μ (the
        // cross entry carries no ∂²μ/∂η² term because q is η_d-free).
        let q = mu * (1.0 - mu);
        let em = (mu / (1.0 - mu)).ln();
        let ed = phi.ln();
        let eta_tower =
            dispersion_eta_nll_order2(DispersionFamilyKind::Beta, score_neutral_y, em, ed, 1.0);
        assert_close(
            "eta-scale observed cross curvature",
            eta_tower.h()[0][1],
            q * phi * analytic,
            1e-8,
        );
    }

    /// #932 oracle: the production `Order2<2>` evaluation of each dispersion
    /// row NLL must reproduce, channel-for-channel (value/grad/Hessian), the
    /// dense `Tower4<2>` evaluation of the same row expression.
    #[test]
    pub(crate) fn order2_matches_dense_tower_all_channels() {
        use gam_math::jet_scalar::Order2;
        use gam_math::jet_tower::Tower4;

        fn check_o2_vs_tower4(label: &str, o2: Order2<2>, t4: Tower4<2>) {
            let band = |a: f64, b: f64| 1e-9 + 1e-9 * a.abs().max(b.abs());
            assert!(
                (o2.value() - t4.v).abs() <= band(o2.value(), t4.v),
                "{label} value: {} vs {}",
                o2.value(),
                t4.v
            );
            for a in 0..2 {
                assert!(
                    (o2.g()[a] - t4.g[a]).abs() <= band(o2.g()[a], t4.g[a]),
                    "{label} grad[{a}]: {} vs {}",
                    o2.g()[a],
                    t4.g[a]
                );
                for b in 0..2 {
                    assert!(
                        (o2.h()[a][b] - t4.h[a][b]).abs() <= band(o2.h()[a][b], t4.h[a][b]),
                        "{label} hess[{a}][{b}]: {} vs {}",
                        o2.h()[a][b],
                        t4.h[a][b]
                    );
                }
            }
        }

        let wi = 1.7_f64;
        // NB2: (μ, θ).
        for &(yi, mu, theta) in &[(0.0, 1.2, 3.0), (4.0, 2.5, 0.7), (10.0, 0.6, 5.0)] {
            check_o2_vs_tower4(
                "nb",
                dispersion_nb_nll_order2(yi, mu, theta, wi),
                test_support::dispersion_nb_nll_generic::<Tower4<2>>(yi, mu, theta, wi),
            );
        }
        // Gamma: (μ, ν).
        for &(yi, mu, nu) in &[
            (0.5_f64, 1.1_f64, 2.0_f64),
            (3.0, 4.0, 0.9),
            (1.0, 0.3, 6.0),
        ] {
            let y_pos = yi.max(1e-300);
            check_o2_vs_tower4(
                "gamma",
                dispersion_gamma_nll_order2(yi, y_pos, mu, nu, wi),
                test_support::dispersion_gamma_nll_generic::<Tower4<2>>(yi, y_pos, mu, nu, wi),
            );
        }
        // Beta: (μ, φ).
        for &(yi, mu, phi) in &[(0.3, 0.4, 5.0), (0.9, 0.6, 12.0), (0.01, 0.2, 3.0)] {
            check_o2_vs_tower4(
                "beta",
                dispersion_beta_nll_order2(yi, mu, phi, wi),
                test_support::dispersion_beta_nll_generic::<Tower4<2>>(yi, mu, phi, wi),
            );
        }
        // Tweedie: (η_μ, η_d), both density branches.
        for &(yi, eta_mu, eta_d, p) in &[
            (0.0, 0.4, -0.3, 1.5),
            (2.5, -0.2, 0.5, 1.3),
            (0.0, 1.0, 0.1, 1.7),
            (5.0, 0.7, -0.6, 1.6),
        ] {
            check_o2_vs_tower4(
                "tweedie",
                dispersion_tweedie_nll_generic::<Order2<2>>(yi, eta_mu, eta_d, p, wi),
                dispersion_tweedie_nll_generic::<Tower4<2>>(yi, eta_mu, eta_d, p, wi),
            );
        }
    }

    /// #1591 prune oracle: the pruned single-axis (`K=1`) dispersion towers
    /// reproduce, `to_bits`-exactly, the CONSUMED channels (`value`, dispersion-
    /// axis `g`/`h`) of the full `Order2<2>` towers — across ≥2000 randomized
    /// rows per family (both Tweedie density branches). This is the bit-identity guarantee that the K-prune changes no
    /// observable float.
    #[test]
    pub(crate) fn pruned_disp_towers_bit_identical_to_full_order2() {
        use gam_math::jet_scalar::Order2;

        // Deterministic LCG so the sweep is reproducible without an rng dep.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let bits = |x: f64| x.to_bits();

        let n_per = 600; // 600 rows × 4 families (Tweedie ×2 branches) > 2000.
        for _ in 0..n_per {
            let wi = 0.25 + 3.0 * next();
            let yi_count = (next() * 12.0).floor();

            // NB: full O2<2> seeds (μ, θ); pruned seeds θ only.
            {
                let mu = (0.05 + 4.0 * next()).max(1e-300);
                let theta = (0.05 + 6.0 * next()).max(1e-12);
                let full = dispersion_nb_nll_order2(yi_count, mu, theta, wi);
                let prn = dispersion_nb_disp_order2(yi_count, mu, theta, wi);
                assert_eq!(bits(full.value()), bits(prn.value()), "nb value");
                assert_eq!(bits(full.g()[1]), bits(prn.g()[0]), "nb grad");
                assert_eq!(bits(full.h()[1][1]), bits(prn.h()[0][0]), "nb hess");
                // Value+gradient-only production tower (`Order1`, trigamma-free):
                // its consumed `value`/`g[0]` must match the `Order2` form
                // bit-for-bit (the observed Hessian it drops is unused by the NB2
                // Fisher-scoring row kernel).
                let prn1 = dispersion_nb_disp_order1(yi_count, mu, theta, wi);
                assert_eq!(bits(prn.value()), bits(prn1.value()), "nb order1 value");
                assert_eq!(bits(prn.g()[0]), bits(prn1.g()[0]), "nb order1 grad");
                // value-only path == -tower.value(), bit-for-bit.
                assert_close(
                    "nb stable value-only",
                    dispersion_nb_loglik(yi_count, mu, theta, wi),
                    -prn.value(),
                    1e-12,
                );
            }
            // Gamma: seeds (μ, ν) / ν.
            {
                let mu = (0.05 + 4.0 * next()).max(1e-300);
                let nu = (0.05 + 6.0 * next()).max(1e-12);
                let yi = 0.01 + 8.0 * next();
                let y_pos = yi.max(1e-300);
                let full = dispersion_gamma_nll_order2(yi, y_pos, mu, nu, wi);
                let prn = dispersion_gamma_disp_order2(yi, y_pos, mu, nu, wi);
                assert_eq!(bits(full.value()), bits(prn.value()), "gamma value");
                assert_eq!(bits(full.g()[1]), bits(prn.g()[0]), "gamma grad");
                assert_eq!(bits(full.h()[1][1]), bits(prn.h()[0][0]), "gamma hess");
                assert_eq!(
                    bits(dispersion_gamma_loglik(yi, y_pos, mu, nu, wi)),
                    bits(-prn.value()),
                    "gamma value-only"
                );
            }
            // Beta value-only path vs full K=2 tower value.
            {
                let mu = (1e-6 + (1.0 - 2e-6) * next()).clamp(1e-12, 1.0 - 1e-12);
                let phi = (0.05 + 20.0 * next()).max(1e-12);
                let yi = next();
                let full = dispersion_beta_nll_order2(yi, mu, phi, wi);
                assert_eq!(
                    bits(dispersion_beta_loglik(yi, mu, phi, wi)),
                    bits(-full.value()),
                    "beta value-only"
                );
            }
            // Tweedie: seeds (η_μ, η_d) / η_d, both density branches.
            for &(yi, eta_mu, eta_d, p) in &[
                (
                    0.0_f64,
                    -4.0 + 8.0 * next(),
                    -4.0 + 8.0 * next(),
                    1.1 + 0.8 * next(),
                ),
                (
                    0.01 + 9.0 * next(),
                    -4.0 + 8.0 * next(),
                    -4.0 + 8.0 * next(),
                    1.1 + 0.8 * next(),
                ),
                (3.0, -8.0, 8.0, 1.5),
            ] {
                let em = eta_mu;
                let ed = eta_d;
                let full = dispersion_tweedie_nll_generic::<Order2<2>>(yi, em, ed, p, wi);
                let prn = dispersion_tweedie_disp_order2(yi, em, ed, p, wi);
                assert_eq!(bits(full.value()), bits(prn.value()), "tweedie value");
                assert_eq!(bits(full.g()[1]), bits(prn.g()[0]), "tweedie grad");
                assert_eq!(bits(full.h()[1][1]), bits(prn.h()[0][0]), "tweedie hess");
                assert_eq!(
                    bits(dispersion_tweedie_loglik(yi, em, ed, p, wi)),
                    bits(-prn.value()),
                    "tweedie value-only"
                );
            }
        }
    }

    /// Audit finding 34 pin: the exact joint Hessian consumes OBSERVED
    /// per-row η-space curvature, not expected (Fisher) information. Gamma
    /// with log links at `y = 4, μ = 2, ν = 3` has closed-form per-row NLL
    /// second derivatives `∂²/∂η_μ² = νy/μ = 6` and `∂²/∂η_μ∂η_ν =
    /// ν(1 − y/μ) = −3`; the Fisher weights are `ν = 3` and `0`.
    #[test]
    pub(crate) fn observed_eta_hessian_matches_gamma_closed_form() {
        let (yi, mu, nu): (f64, f64, f64) = (4.0, 2.0, 3.0);
        let (h_mm, h_md, h_dd) = dispersion_row_observed_hessian_weights(
            DispersionFamilyKind::Gamma,
            yi,
            mu.ln(),
            nu.ln(),
            1.0,
        );
        assert_close("gamma observed d2/d_eta_mu2", h_mm, nu * yi / mu, 1e-10);
        assert_close(
            "gamma observed cross d2/d_eta_mu d_eta_nu",
            h_md,
            nu * (1.0 - yi / mu),
            1e-10,
        );
        // ∂²NLL/∂η_ν² = ν²(ψ′(ν) − 1/ν) + [ν(lnμ − lnν − 1 + ψ(ν) − ln y + y/μ)]·(−1)…
        // pin against a central finite difference of the value channel instead
        // of a second hand derivation.
        let nll =
            |ed: f64| -dispersion_row_loglik(DispersionFamilyKind::Gamma, yi, mu.ln(), ed, 1.0);
        let h = 1e-5;
        let ed0 = nu.ln();
        let fd = (nll(ed0 + h) - 2.0 * nll(ed0) + nll(ed0 - h)) / (h * h);
        assert_close("gamma observed d2/d_eta_nu2 (FD)", h_dd, fd, 1e-4);
    }

    /// Finite predictors beyond the former arbitrary clamp remain on the exact
    /// likelihood surface and retain their score and curvature.
    #[test]
    pub(crate) fn observed_eta_hessian_is_exact_beyond_former_clamp() {
        let (h_mm, h_md, h_dd) = dispersion_row_observed_hessian_weights(
            DispersionFamilyKind::Gamma,
            4.0,
            35.0,
            0.5,
            1.0,
        );
        assert!(h_mm > 0.0);
        assert!(h_md.is_finite());
        assert!(h_dd != 0.0);
        let kernel = dispersion_row_kernel(DispersionFamilyKind::Gamma, 4.0, 35.0, 0.5, 1.0);
        assert!(kernel.mean_weight > 0.0);
        assert_ne!(kernel.mean_response, 35.0);
        assert!(kernel.disp_weight > 0.0);
    }

    #[test]
    fn negative_binomial_balanced_ratios_and_precision_information_keep_tail_geometry() {
        let huge = 1.0e200_f64;
        let kernel = dispersion_row_kernel(
            DispersionFamilyKind::NegativeBinomial,
            1.0,
            huge.ln(),
            huge.ln(),
            1.0,
        );
        assert!(kernel.loglik.is_finite());
        assert!(kernel.mean_weight.is_finite());
        assert!((kernel.mean_weight / huge - 0.5).abs() <= 8.0 * f64::EPSILON);
        assert!(kernel.disp_weight.is_finite() && kernel.disp_weight > 0.0);

        let eta_info = nb_log_precision_fisher_jensen(1.0, 1.0e17);
        assert!(eta_info.is_finite() && eta_info > 0.0);
        assert!((eta_info * 1.0e17 - 1.0).abs() < 1.0e-12);

        let log_share = log_positive_share((-700.0_f64).exp(), 700.0_f64.exp());
        assert!(log_share.is_finite());
        assert!((log_share + 1400.0).abs() < 1.0e-12);
    }

    /// Speed-path guard (#932): `evaluate` / `log_likelihood_only` materialize
    /// the row-kernel map in parallel for large `n`, then reduce SERIALLY in
    /// index order. This pins the parallel output (log-likelihood + both
    /// blocks' working response/weight vectors) to a hand-rolled serial
    /// reference so CI catches any reassociation or row-misindex regression.
    /// `n` sits well above `DISPERSION_PARALLEL_ROW_THRESHOLD`, and the test
    /// runs on the main thread (not a rayon worker), so the parallel branch is
    /// the one exercised. Because the reduction order is preserved the match is
    /// in fact bit-exact; the `1e-9` band is the contract floor.
    #[test]
    pub(crate) fn parallel_evaluate_matches_serial_reference() {
        let n = DISPERSION_PARALLEL_ROW_THRESHOLD * 3 + 7;
        // Deterministic LCG row data (no rng dependency).
        let mut state: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        for kind in [
            DispersionFamilyKind::NegativeBinomial,
            DispersionFamilyKind::Gamma,
            DispersionFamilyKind::Beta,
            DispersionFamilyKind::Tweedie { p: 1.5 },
        ] {
            let y = Array1::from_shape_fn(n, |_| match kind {
                DispersionFamilyKind::Beta => 1e-3 + (1.0 - 2e-3) * next(),
                DispersionFamilyKind::NegativeBinomial => (next() * 12.0).floor(),
                _ => 0.05 + 8.0 * next(),
            });
            let weights = Array1::from_shape_fn(n, |_| 0.25 + 2.0 * next());
            let eta_mu = Array1::from_shape_fn(n, |_| -1.0 + 2.0 * next());
            let eta_d = Array1::from_shape_fn(n, |_| -1.0 + 2.0 * next());

            let family = DispersionGlmLocationScaleFamily {
                kind,
                y: y.clone(),
                weights: weights.clone(),
            };
            let states = vec![
                ParameterBlockState {
                    beta: Array1::zeros(0),
                    eta: eta_mu.clone(),
                },
                ParameterBlockState {
                    beta: Array1::zeros(0),
                    eta: eta_d.clone(),
                },
            ];

            // Serial reference, computed exactly as the pre-parallel loop did.
            let mut ll_ref = 0.0;
            let mut mw_ref = Array1::<f64>::zeros(n);
            let mut mr_ref = Array1::<f64>::zeros(n);
            let mut dw_ref = Array1::<f64>::zeros(n);
            let mut dr_ref = Array1::<f64>::zeros(n);
            for i in 0..n {
                let row = dispersion_row_kernel(kind, y[i], eta_mu[i], eta_d[i], weights[i]);
                ll_ref += row.loglik;
                mw_ref[i] = row.mean_weight;
                mr_ref[i] = row.mean_response;
                dw_ref[i] = row.disp_weight;
                dr_ref[i] = row.disp_response;
            }

            let eval = family.evaluate(&states).expect("parallel evaluate");
            assert_close(
                &format!("{kind:?} evaluate log-likelihood"),
                eval.log_likelihood,
                ll_ref,
                1e-9,
            );

            let BlockWorkingSet::Diagonal {
                working_response: mr,
                working_weights: mw,
            } = &eval.blockworking_sets[0]
            else {
                panic!("mean block not diagonal");
            };
            let BlockWorkingSet::Diagonal {
                working_response: dr,
                working_weights: dw,
            } = &eval.blockworking_sets[1]
            else {
                panic!("dispersion block not diagonal");
            };
            for i in 0..n {
                assert_close("mean weight", mw[i], mw_ref[i], 1e-9);
                assert_close("mean response", mr[i], mr_ref[i], 1e-9);
                assert_close("disp weight", dw[i], dw_ref[i], 1e-9);
                assert_close("disp response", dr[i], dr_ref[i], 1e-9);
            }

            // `log_likelihood_only` takes the same parallel-then-serial-sum
            // path; its value-only kernel is bit-identical to evaluate's loglik.
            let ll_only = family
                .log_likelihood_only(&states)
                .expect("parallel log_likelihood_only");
            assert_close(
                &format!("{kind:?} log_likelihood_only"),
                ll_only,
                ll_ref,
                1e-9,
            );
        }
    }
}
