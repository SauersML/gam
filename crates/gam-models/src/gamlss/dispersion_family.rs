//! #913: dispersion-channel GAMLSS location-scale families.
//!
//! Extracted from `gamlss.rs` (issue #780); this module now owns the
//! dispersion-channel joint-curvature corrections.

use super::weighted_design_products::{mirror_upper_to_lower, xt_diag_x_design, xt_diag_y_design};
use super::{
    BlockwiseTermFitResult, GamlssLambdaLayout, LOCATION_SCALE_N_OUTPUTS,
    LocationScaleFamilyBuilder, build_location_scale_block, fit_location_scale_terms,
    input_failure, solve_penalizedweighted_projection, spatial_length_scale_term_indices,
};
use crate::fit_orchestration::FitFailure;
use crate::block_layout::block_count::validate_block_count;
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
};
use crate::gamlss::GamlssError;
use crate::model_types::UnifiedFitResult;
use gam_linalg::matrix::LinearOperator;
use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
    get_spatial_length_scale, spatial_term_uses_per_axis_psi,
};
use gam_row_macros::row_program;
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

    /// The family's canonical `LikelihoodSpec` (mean response × mean link).
    /// The overdispersion parameter is estimated by the log-precision channel,
    /// so the response-family placeholder parameters (`phi`, `theta`) mirror
    /// the `resolve_family` defaults
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
///
/// Work bound (#2469): result-invariant. Each of its four uses maps every row
/// through the same row function on either side of it and collects the per-row
/// values in index order, so everything downstream sees the identical `Vec`
/// (`parallel_evaluate_matches_serial_reference` exercises the parallel side).
const DISPERSION_PARALLEL_ROW_THRESHOLD: usize = 1024;

/// Per-row working quantities for both channels at the current `(η_μ, η_d)`.
pub(super) struct DispersionRowKernel {
    pub(super) loglik: f64,
    pub(super) mean_weight: f64,
    pub(super) mean_response: f64,
    pub(super) disp_weight: f64,
    pub(super) disp_response: f64,
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
            Err(GamlssError::row_geometry_unrepresentable(row, quantity, eta, value))
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
                return Err(GamlssError::row_geometry_unrepresentable(
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
            return Err(GamlssError::row_geometry_unrepresentable(row, quantity, eta, value));
        }
    }
    Ok(())
}

#[cfg(test)]
mod test_support {
    use super::*;
    use crate::gamlss::test_support::order2_ln_gamma;
    use gam_math::jet_scalar::JetScalar;
    use gam_math::nested_dual::JetField;

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
    /// The NB row kernel reads its scores from the negative binomial row program
    /// in [`dispersion_row_kernel`], so this `K=2` form survives only as the
    /// dense-`Tower4<2>` oracle pin (`order2_matches_dense_tower_all_channels`).
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
    /// row kernel reads the Gamma scores from its row program, so this form is
    /// kept only as the dense-tower oracle pin.
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

    /// Full `Order2<2>` Beta row NLL seeded on `(μ, φ)`. The row kernel reads
    /// the Beta score from its row program; this tower is its oracle
    /// (`row_kernel_closed_form_dispersion_channels_match_the_towers`) and the
    /// dense-tower pin's subject.
    #[inline]
    pub(super) fn dispersion_beta_nll_order2(
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

    /// Pruned single-axis Gamma dispersion tower: `ν` is the sole jet variable
    /// (axis 0), `μ` a constant. Consumed channels match
    /// `dispersion_gamma_nll_order2` index-1 bit-for-bit, and the row kernel's
    /// row-program dispersion score and closed-form information match this tower.
    #[inline]
    pub(super) fn dispersion_gamma_disp_order2(
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
    pub(super) fn dispersion_tweedie_disp_order2(
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
/// `trigamma_theta` is `ψ′(θ)`, which the row's score stack already carries.
#[inline]
fn nb_log_precision_fisher_jensen(mu: f64, theta: f64, trigamma_theta: f64) -> f64 {
    let r = positive_share(theta, mu);
    let q = positive_share(mu, theta);
    if theta <= 32.0 {
        let total = theta + mu;
        let remainder_theta = trigamma_theta - theta.recip();
        let remainder_total = gam_math::special::trigamma(total) - total.recip();
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

// ============================================================================
// Dispersion-channel row derivatives (#932).
//
// Each member's row log-likelihood is declared once, as a `row_program!` in the
// local predictor coordinates `(δ_μ, δ_d)` about the row's `(η_μ, η_d)`. Its
// order-2 surface is the row's score and observed Hessian and its contracted
// third surface is that Hessian's directional derivative, so the link chains,
// the mean/precision cross curvature and every product-rule term are generated
// from the declaration. The caller supplies only one-variable derivative stacks
// at the row: `ln Γ` through tetragamma, softplus for the negative binomial log
// shares, the logistic mean link, the Tweedie mean's power terms `e^{(2−p)t}` and
// `e^{(1−p)t}` with their coefficients, and `e^t` at `t = 0`. Each argument's
// polygamma entries come from one walk of the recurrence
// (`gam_math::special::polygamma_stack`), which divides once per step for every
// order where the per-order scalars divide once each.
//
// A supplied value that enters the result only through `add` or `scale` reaches
// the value channel and nothing else. The production stacks supply zero for those
// values (every `ln Γ` value, the negative binomial `−ln q`, the Tweedie density
// normalizer), and the row log-likelihood comes from the plain-f64 functions
// above. Values that multiply a jet (the negative binomial `−ln r`, the Beta mean,
// the Tweedie deviance terms) are always supplied. The programs emit through
// third order, whose surfaces read stack entries through the third, so every
// fourth entry is zero.
// ============================================================================

// NB2: ℓ = ln Γ(θ + y) − ln Γ(θ) − ln Γ(y + 1) + θ ln r + y ln q, with
// q = μ/(μ + θ) and r = θ/(μ + θ). In `x = η_μ − η_d`, `ln r = −softplus(x)` and
// `ln q = −softplus(−x)`, whose stacks are the stable shares `q`, `r` and `qr`, so
// `μ + θ` is never formed.
row_program! {
    fn negative_binomial_row_program(
        delta_mu,
        delta_d;
        theta,
        count,
        ln_gamma_count,
        ln_gamma_total,
        digamma_total,
        trigamma_total,
        tetragamma_total,
        ln_gamma_theta,
        digamma_theta,
        trigamma_theta,
        tetragamma_theta,
        neg_log_theta_share,
        neg_log_mu_share,
        mu_share,
        theta_share
    )
    emit [order2, third];
    leaves {
        unit_exponential => supplied,
        ln_gamma_at_total => supplied,
        ln_gamma_at_theta => supplied,
        softplus_at_log_ratio => supplied,
        softplus_at_negative_log_ratio => supplied,
    }
    witnesses [];
    {
        let precision_ratio = compose(unit_exponential, delta_d, 1.0, 1.0, 1.0, 1.0, 1.0);
        let precision = scale(precision_ratio, theta);
        let total = add_constant(precision, count);
        let ln_gamma_total_jet = compose(
            ln_gamma_at_total,
            total,
            ln_gamma_total,
            digamma_total,
            trigamma_total,
            tetragamma_total,
            0.0
        );
        let ln_gamma_theta_jet = compose(
            ln_gamma_at_theta,
            precision,
            ln_gamma_theta,
            digamma_theta,
            trigamma_theta,
            tetragamma_theta,
            0.0
        );
        let spread = add(delta_mu, neg(delta_d));
        let theta_gap = compose(
            softplus_at_log_ratio,
            spread,
            neg_log_theta_share,
            mu_share,
            mu_share * theta_share,
            mu_share * theta_share * (theta_share - mu_share),
            0.0
        );
        let reverse_spread = neg(spread);
        let mu_gap = compose(
            softplus_at_negative_log_ratio,
            reverse_spread,
            neg_log_mu_share,
            theta_share,
            mu_share * theta_share,
            mu_share * theta_share * (mu_share - theta_share),
            0.0
        );
        let ln_gamma_ratio = add(ln_gamma_total_jet, neg(ln_gamma_theta_jet));
        let log_shares = add(mul(precision, theta_gap), scale(mu_gap, count));
        return add_constant(add(ln_gamma_ratio, neg(log_shares)), -ln_gamma_count);
    }
}

// Gamma: ℓ = ν (η_d − η_μ) − ln Γ(ν) + (ν − 1) ln y − ν y e^{−η_μ}, with
// ν = e^{η_d}; `response_ratio` is `y/μ` at the row.
row_program! {
    fn gamma_row_program(
        delta_mu,
        delta_d;
        shape,
        log_shape_ratio,
        log_response,
        response_ratio,
        ln_gamma_shape,
        digamma_shape,
        trigamma_shape,
        tetragamma_shape
    )
    emit [order2, third];
    leaves {
        unit_exponential => supplied,
        ln_gamma_at_shape => supplied,
    }
    witnesses [];
    {
        let shape_ratio = compose(unit_exponential, delta_d, 1.0, 1.0, 1.0, 1.0, 1.0);
        let precision = scale(shape_ratio, shape);
        let log_ratio = add_constant(add(delta_d, neg(delta_mu)), log_shape_ratio);
        let reverse_delta_mu = neg(delta_mu);
        let inverse_mean_ratio =
            compose(unit_exponential, reverse_delta_mu, 1.0, 1.0, 1.0, 1.0, 1.0);
        let ln_gamma_jet = compose(
            ln_gamma_at_shape,
            precision,
            ln_gamma_shape,
            digamma_shape,
            trigamma_shape,
            tetragamma_shape,
            0.0
        );
        let scaled_response = scale(inverse_mean_ratio, response_ratio);
        let kernel = add(mul(precision, log_ratio), neg(ln_gamma_jet));
        let response = add(scale(precision, log_response), neg(mul(precision, scaled_response)));
        return add_constant(add(kernel, response), -log_response);
    }
}

// Beta(μφ, (1 − μ)φ): ℓ = ln Γ(φ) − ln Γ(a) − ln Γ(b) + (a − 1) ln y
// + (b − 1) ln(1 − y), with a = μφ, b = (1 − μ)φ, μ = logistic(η_μ), φ = e^{η_d}.
row_program! {
    fn beta_row_program(
        delta_mu,
        delta_d;
        precision,
        mean,
        mean_first,
        mean_second,
        mean_third,
        log_response,
        log_complement,
        ln_gamma_precision,
        digamma_precision,
        trigamma_precision,
        tetragamma_precision,
        ln_gamma_first_shape,
        digamma_first_shape,
        trigamma_first_shape,
        tetragamma_first_shape,
        ln_gamma_second_shape,
        digamma_second_shape,
        trigamma_second_shape,
        tetragamma_second_shape
    )
    emit [order2, third];
    leaves {
        unit_exponential => supplied,
        logistic => supplied,
        ln_gamma_at_precision => supplied,
        ln_gamma_at_first_shape => supplied,
        ln_gamma_at_second_shape => supplied,
    }
    witnesses [];
    {
        let mean_jet = compose(logistic, delta_mu, mean, mean_first, mean_second, mean_third, 0.0);
        let complement = add_constant(neg(mean_jet), 1.0);
        let precision_ratio = compose(unit_exponential, delta_d, 1.0, 1.0, 1.0, 1.0, 1.0);
        let precision_jet = scale(precision_ratio, precision);
        let first_shape = mul(mean_jet, precision_jet);
        let second_shape = mul(complement, precision_jet);
        let ln_gamma_precision_jet = compose(
            ln_gamma_at_precision,
            precision_jet,
            ln_gamma_precision,
            digamma_precision,
            trigamma_precision,
            tetragamma_precision,
            0.0
        );
        let ln_gamma_first_jet = compose(
            ln_gamma_at_first_shape,
            first_shape,
            ln_gamma_first_shape,
            digamma_first_shape,
            trigamma_first_shape,
            tetragamma_first_shape,
            0.0
        );
        let ln_gamma_second_jet = compose(
            ln_gamma_at_second_shape,
            second_shape,
            ln_gamma_second_shape,
            digamma_second_shape,
            trigamma_second_shape,
            tetragamma_second_shape,
            0.0
        );
        let normalizer = add(
            ln_gamma_precision_jet,
            neg(add(ln_gamma_first_jet, ln_gamma_second_jet))
        );
        let response = add(scale(first_shape, log_response), scale(second_shape, log_complement));
        return add_constant(add(normalizer, response), -(log_response + log_complement));
    }
}

// Tweedie, positive y (saddlepoint density): ℓ = −½ κ dev + ½ η_d − ½ ln 2π
// − ½ p ln y, with κ = 1/φ = e^{η_d} and
// dev = 2 (μ^{2−p}/(2−p) − y μ^{1−p}/(1−p) + y^{2−p}/((1−p)(2−p))). About the
// row, ½ dev = A e^{(2−p)δ_μ} − B e^{(1−p)δ_μ} + C with A = μ^{2−p}/(2−p),
// B = y μ^{1−p}/(1−p) and C = y^{2−p}/((1−p)(2−p)). The caller supplies the two
// exponential terms' stacks `A (2−p)^k` and `B (1−p)^k`; `deviance_offset` is C
// and `log_normalizer` is `½ η_d − ½ ln 2π − ½ p ln y` at the row.
row_program! {
    fn tweedie_positive_row_program(
        delta_mu,
        delta_d;
        kappa,
        mean_term,
        mean_first,
        mean_second,
        mean_third,
        response_term,
        response_first,
        response_second,
        response_third,
        deviance_offset,
        log_normalizer
    )
    emit [order2, third];
    leaves {
        unit_exponential => supplied,
        mean_power => supplied,
        response_power => supplied,
    }
    witnesses [];
    {
        let precision_ratio = compose(unit_exponential, delta_d, 1.0, 1.0, 1.0, 1.0, 1.0);
        let precision = scale(precision_ratio, kappa);
        let mean_jet = compose(
            mean_power,
            delta_mu,
            mean_term,
            mean_first,
            mean_second,
            mean_third,
            0.0
        );
        let response_jet = compose(
            response_power,
            delta_mu,
            response_term,
            response_first,
            response_second,
            response_third,
            0.0
        );
        let half_deviance = add_constant(add(mean_jet, neg(response_jet)), deviance_offset);
        let density = add(neg(mul(precision, half_deviance)), scale(delta_d, 0.5));
        return add_constant(density, log_normalizer);
    }
}

// Tweedie, y = 0 (exact point mass): ℓ = −κ μ^{2−p}/(2−p), κ = e^{η_d}. About the
// row μ^{2−p}/(2−p) = A e^{(2−p)δ_μ}, and the caller supplies the stack `A (2−p)^k`.
row_program! {
    fn tweedie_zero_row_program(
        delta_mu,
        delta_d;
        kappa,
        mean_term,
        mean_first,
        mean_second,
        mean_third
    )
    emit [order2, third];
    leaves {
        unit_exponential => supplied,
        mean_power => supplied,
    }
    witnesses [];
    {
        let precision_ratio = compose(unit_exponential, delta_d, 1.0, 1.0, 1.0, 1.0, 1.0);
        let precision = scale(precision_ratio, kappa);
        let mean_jet = compose(
            mean_power,
            delta_mu,
            mean_term,
            mean_first,
            mean_second,
            mean_third,
            0.0
        );
        return neg(mul(precision, mean_jet));
    }
}

/// One row's supplied stacks for its member's row program, through derivative
/// `order`. Polygamma entries above `order` are zero, and so are the values that
/// reach only the value channel (see the section note). A surface of order `k`
/// reads the entries through `k`.
#[derive(Clone, Copy)]
enum DispersionRowStacks {
    NegativeBinomial {
        theta: f64,
        count: f64,
        ln_gamma_count: f64,
        ln_gamma_total: f64,
        digamma_total: f64,
        trigamma_total: f64,
        tetragamma_total: f64,
        ln_gamma_theta: f64,
        digamma_theta: f64,
        trigamma_theta: f64,
        tetragamma_theta: f64,
        neg_log_theta_share: f64,
        neg_log_mu_share: f64,
        mu_share: f64,
        theta_share: f64,
    },
    Gamma {
        shape: f64,
        log_shape_ratio: f64,
        log_response: f64,
        response_ratio: f64,
        ln_gamma_shape: f64,
        digamma_shape: f64,
        trigamma_shape: f64,
        tetragamma_shape: f64,
    },
    Beta {
        precision: f64,
        mean: f64,
        mean_first: f64,
        mean_second: f64,
        mean_third: f64,
        log_response: f64,
        log_complement: f64,
        ln_gamma_precision: f64,
        digamma_precision: f64,
        trigamma_precision: f64,
        tetragamma_precision: f64,
        ln_gamma_first_shape: f64,
        digamma_first_shape: f64,
        trigamma_first_shape: f64,
        tetragamma_first_shape: f64,
        ln_gamma_second_shape: f64,
        digamma_second_shape: f64,
        trigamma_second_shape: f64,
        tetragamma_second_shape: f64,
    },
    TweediePositive {
        kappa: f64,
        mean_term: f64,
        mean_first: f64,
        mean_second: f64,
        mean_third: f64,
        response_term: f64,
        response_first: f64,
        response_second: f64,
        response_third: f64,
        deviance_offset: f64,
        log_normalizer: f64,
    },
    TweedieZero {
        kappa: f64,
        mean_term: f64,
        mean_first: f64,
        mean_second: f64,
        mean_third: f64,
    },
}

impl DispersionRowStacks {
    #[inline(always)]
    fn at(kind: DispersionFamilyKind, yi: f64, em: f64, ed: f64, order: usize) -> Self {
        use gam_math::special::polygamma_stack;
        match kind {
            DispersionFamilyKind::NegativeBinomial => {
                let mu = em.exp();
                let theta = ed.exp();
                Self::negative_binomial(
                    yi,
                    mu,
                    theta,
                    polygamma_stack(theta + yi, order),
                    polygamma_stack(theta, order),
                )
            }
            DispersionFamilyKind::Gamma => {
                let nu = ed.exp();
                Self::gamma(yi, em, ed, nu, polygamma_stack(nu, order))
            }
            DispersionFamilyKind::Beta => {
                let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em);
                let phi = ed.exp();
                Self::beta(
                    yi,
                    &logit,
                    phi,
                    polygamma_stack(phi, order),
                    polygamma_stack(logit.mu * phi, order),
                    polygamma_stack((1.0 - logit.mu) * phi, order),
                )
            }
            DispersionFamilyKind::Tweedie { p } => Self::tweedie(yi, p, em.exp(), ed.exp()),
        }
    }

    /// `total` and `precision` are the polygamma stacks at `θ + y` and `θ`.
    #[inline(always)]
    fn negative_binomial(
        yi: f64,
        mu: f64,
        theta: f64,
        total: [f64; 5],
        precision: [f64; 5],
    ) -> Self {
        let [digamma_total, trigamma_total, tetragamma_total, ..] = total;
        let [digamma_theta, trigamma_theta, tetragamma_theta, ..] = precision;
        Self::NegativeBinomial {
            theta,
            count: yi,
            ln_gamma_count: 0.0,
            ln_gamma_total: 0.0,
            digamma_total,
            trigamma_total,
            tetragamma_total,
            ln_gamma_theta: 0.0,
            digamma_theta,
            trigamma_theta,
            tetragamma_theta,
            neg_log_theta_share: -log_positive_share(theta, mu),
            neg_log_mu_share: 0.0,
            mu_share: positive_share(mu, theta),
            theta_share: positive_share(theta, mu),
        }
    }

    /// `shape` is the polygamma stack at `ν = e^{η_d}`.
    #[inline(always)]
    fn gamma(yi: f64, em: f64, ed: f64, nu: f64, shape: [f64; 5]) -> Self {
        let [digamma_shape, trigamma_shape, tetragamma_shape, ..] = shape;
        Self::Gamma {
            shape: nu,
            log_shape_ratio: ed - em,
            log_response: yi.ln(),
            response_ratio: (1.0 / em.exp()) * yi,
            ln_gamma_shape: 0.0,
            digamma_shape,
            trigamma_shape,
            tetragamma_shape,
        }
    }

    /// `precision`, `first_shape` and `second_shape` are the polygamma stacks at
    /// `φ`, `μφ` and `(1 − μ)φ`.
    #[inline(always)]
    fn beta(
        yi: f64,
        logit: &gam_solve::mixture_link::LogitJet5,
        phi: f64,
        precision: [f64; 5],
        first_shape: [f64; 5],
        second_shape: [f64; 5],
    ) -> Self {
        let [digamma_precision, trigamma_precision, tetragamma_precision, ..] = precision;
        let [digamma_first_shape, trigamma_first_shape, tetragamma_first_shape, ..] = first_shape;
        let [digamma_second_shape, trigamma_second_shape, tetragamma_second_shape, ..] =
            second_shape;
        Self::Beta {
            precision: phi,
            mean: logit.mu,
            mean_first: logit.d1,
            mean_second: logit.d2,
            mean_third: logit.d3,
            log_response: yi.ln(),
            log_complement: (-yi).ln_1p(),
            ln_gamma_precision: 0.0,
            digamma_precision,
            trigamma_precision,
            tetragamma_precision,
            ln_gamma_first_shape: 0.0,
            digamma_first_shape,
            trigamma_first_shape,
            tetragamma_first_shape,
            ln_gamma_second_shape: 0.0,
            digamma_second_shape,
            trigamma_second_shape,
            tetragamma_second_shape,
        }
    }

    /// The deviance's power terms are formed as `dispersion_tweedie_loglik` forms
    /// them, so a row kernel that evaluates both shares their divisions.
    #[inline(always)]
    fn tweedie(yi: f64, p: f64, mu: f64, kappa: f64) -> Self {
        let two_minus_p = 2.0 - p;
        let mean_power_two = mu.powf(two_minus_p);
        let mean_term = mean_power_two * (1.0 / two_minus_p);
        let mean_second = mean_power_two * two_minus_p;
        let mean_third = mean_second * two_minus_p;
        if yi > 0.0 {
            let one_minus_p = 1.0 - p;
            let mean_power_one = mu.powf(one_minus_p);
            let response_first = yi * mean_power_one;
            let response_second = response_first * one_minus_p;
            Self::TweediePositive {
                kappa,
                mean_term,
                mean_first: mean_power_two,
                mean_second,
                mean_third,
                response_term: mean_power_one * (yi / one_minus_p),
                response_first,
                response_second,
                response_third: response_second * one_minus_p,
                deviance_offset: yi.powf(two_minus_p) / (one_minus_p * two_minus_p),
                log_normalizer: 0.0,
            }
        } else {
            Self::TweedieZero {
                kappa,
                mean_term,
                mean_first: mean_power_two,
                mean_second,
                mean_third,
            }
        }
    }

    /// Value, score and observed Hessian of the row log-likelihood in
    /// `(η_μ, η_d)`: the member's order-2 surface at `δ = 0`.
    #[inline(always)]
    fn order2(self) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        let (value, gradient, hessian, []) = match self {
            Self::NegativeBinomial {
                theta,
                count,
                ln_gamma_count,
                ln_gamma_total,
                digamma_total,
                trigamma_total,
                tetragamma_total,
                ln_gamma_theta,
                digamma_theta,
                trigamma_theta,
                tetragamma_theta,
                neg_log_theta_share,
                neg_log_mu_share,
                mu_share,
                theta_share,
            } => negative_binomial_row_program_order2(
                0.0,
                0.0,
                theta,
                count,
                ln_gamma_count,
                ln_gamma_total,
                digamma_total,
                trigamma_total,
                tetragamma_total,
                ln_gamma_theta,
                digamma_theta,
                trigamma_theta,
                tetragamma_theta,
                neg_log_theta_share,
                neg_log_mu_share,
                mu_share,
                theta_share,
            ),
            Self::Gamma {
                shape,
                log_shape_ratio,
                log_response,
                response_ratio,
                ln_gamma_shape,
                digamma_shape,
                trigamma_shape,
                tetragamma_shape,
            } => gamma_row_program_order2(
                0.0,
                0.0,
                shape,
                log_shape_ratio,
                log_response,
                response_ratio,
                ln_gamma_shape,
                digamma_shape,
                trigamma_shape,
                tetragamma_shape,
            ),
            Self::Beta {
                precision,
                mean,
                mean_first,
                mean_second,
                mean_third,
                log_response,
                log_complement,
                ln_gamma_precision,
                digamma_precision,
                trigamma_precision,
                tetragamma_precision,
                ln_gamma_first_shape,
                digamma_first_shape,
                trigamma_first_shape,
                tetragamma_first_shape,
                ln_gamma_second_shape,
                digamma_second_shape,
                trigamma_second_shape,
                tetragamma_second_shape,
            } => beta_row_program_order2(
                0.0,
                0.0,
                precision,
                mean,
                mean_first,
                mean_second,
                mean_third,
                log_response,
                log_complement,
                ln_gamma_precision,
                digamma_precision,
                trigamma_precision,
                tetragamma_precision,
                ln_gamma_first_shape,
                digamma_first_shape,
                trigamma_first_shape,
                tetragamma_first_shape,
                ln_gamma_second_shape,
                digamma_second_shape,
                trigamma_second_shape,
                tetragamma_second_shape,
            ),
            Self::TweediePositive {
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
                response_term,
                response_first,
                response_second,
                response_third,
                deviance_offset,
                log_normalizer,
            } => tweedie_positive_row_program_order2(
                0.0,
                0.0,
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
                response_term,
                response_first,
                response_second,
                response_third,
                deviance_offset,
                log_normalizer,
            ),
            Self::TweedieZero {
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
            } => tweedie_zero_row_program_order2(
                0.0,
                0.0,
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
            ),
        };
        (value, gradient, hessian)
    }

    /// The row log-likelihood's third derivative contracted along `direction`,
    /// `Σ_c ℓ_abc u_c` in `(η_μ, η_d)`: the member's contracted third surface.
    #[inline(always)]
    fn third_contracted(self, direction: &[f64; 2]) -> [[f64; 2]; 2] {
        match self {
            Self::NegativeBinomial {
                theta,
                count,
                ln_gamma_count,
                ln_gamma_total,
                digamma_total,
                trigamma_total,
                tetragamma_total,
                ln_gamma_theta,
                digamma_theta,
                trigamma_theta,
                tetragamma_theta,
                neg_log_theta_share,
                neg_log_mu_share,
                mu_share,
                theta_share,
            } => negative_binomial_row_program_third_contracted(
                0.0,
                0.0,
                theta,
                count,
                ln_gamma_count,
                ln_gamma_total,
                digamma_total,
                trigamma_total,
                tetragamma_total,
                ln_gamma_theta,
                digamma_theta,
                trigamma_theta,
                tetragamma_theta,
                neg_log_theta_share,
                neg_log_mu_share,
                mu_share,
                theta_share,
                direction,
            ),
            Self::Gamma {
                shape,
                log_shape_ratio,
                log_response,
                response_ratio,
                ln_gamma_shape,
                digamma_shape,
                trigamma_shape,
                tetragamma_shape,
            } => gamma_row_program_third_contracted(
                0.0,
                0.0,
                shape,
                log_shape_ratio,
                log_response,
                response_ratio,
                ln_gamma_shape,
                digamma_shape,
                trigamma_shape,
                tetragamma_shape,
                direction,
            ),
            Self::Beta {
                precision,
                mean,
                mean_first,
                mean_second,
                mean_third,
                log_response,
                log_complement,
                ln_gamma_precision,
                digamma_precision,
                trigamma_precision,
                tetragamma_precision,
                ln_gamma_first_shape,
                digamma_first_shape,
                trigamma_first_shape,
                tetragamma_first_shape,
                ln_gamma_second_shape,
                digamma_second_shape,
                trigamma_second_shape,
                tetragamma_second_shape,
            } => beta_row_program_third_contracted(
                0.0,
                0.0,
                precision,
                mean,
                mean_first,
                mean_second,
                mean_third,
                log_response,
                log_complement,
                ln_gamma_precision,
                digamma_precision,
                trigamma_precision,
                tetragamma_precision,
                ln_gamma_first_shape,
                digamma_first_shape,
                trigamma_first_shape,
                tetragamma_first_shape,
                ln_gamma_second_shape,
                digamma_second_shape,
                trigamma_second_shape,
                tetragamma_second_shape,
                direction,
            ),
            Self::TweediePositive {
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
                response_term,
                response_first,
                response_second,
                response_third,
                deviance_offset,
                log_normalizer,
            } => tweedie_positive_row_program_third_contracted(
                0.0,
                0.0,
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
                response_term,
                response_first,
                response_second,
                response_third,
                deviance_offset,
                log_normalizer,
                direction,
            ),
            Self::TweedieZero {
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
            } => tweedie_zero_row_program_third_contracted(
                0.0,
                0.0,
                kappa,
                mean_term,
                mean_first,
                mean_second,
                mean_third,
                direction,
            ),
        }
    }
}

/// Per-row log-likelihood derivatives in the predictor coordinates `(η_μ, η_d)`
/// through second order, `([ℓ_μ, ℓ_d], [ℓ_μμ, ℓ_μd, ℓ_dd])`, from the member's row
/// program.
///
/// The mean-link and precision-link chains, the inverse-link second-derivative
/// terms and the mean/dispersion cross curvature are all included, so `−w` times
/// this is the exact per-row OBSERVED Hessian rather than the expected (Fisher)
/// working weights. Example: Gamma with log links at `y = 4, μ = 2, ν = 3` has
/// `∂²NLL/∂η_μ² = νy/μ = 6` and `∂²NLL/∂η_μ∂η_ν = ν(1 − y/μ) = −3`, where the
/// Fisher working weights give `ν = 3` and `0`. The oracle is
/// `crate::gamlss::test_support::dispersion_eta_nll_order2`.
#[inline]
fn dispersion_eta_loglik_second(
    kind: DispersionFamilyKind,
    yi: f64,
    em: f64,
    ed: f64,
) -> ([f64; 2], [f64; 3]) {
    let (_, gradient, hessian) = DispersionRowStacks::at(kind, yi, em, ed, 2).order2();
    (gradient, [hessian[0][0], hessian[0][1], hessian[1][1]])
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
    let (_, [l_mm, l_md, l_dd]) = dispersion_eta_loglik_second(kind, yi, eta_mu, eta_d);
    (
        -prior_weight * l_mm,
        -prior_weight * l_md,
        -prior_weight * l_dd,
    )
}

/// Per-row directional derivative of the observed η-space Hessian channels
/// `(∂²NLL/∂η_μ², ∂²NLL/∂η_μ∂η_d, ∂²NLL/∂η_d²)` along the per-row η-motion
/// `(du_mu, du_d)`: the member's contracted third surface. The oracle is
/// `crate::gamlss::test_support::dispersion_eta_nll_order3`.
pub(crate) fn dispersion_row_observed_hessian_directional(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
    du_mu: f64,
    du_d: f64,
) -> (f64, f64, f64) {
    if prior_weight <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let drift =
        DispersionRowStacks::at(kind, yi, eta_mu, eta_d, 3).third_contracted(&[du_mu, du_d]);
    let scale = -prior_weight;
    (
        scale * drift[0][0],
        scale * drift[0][1],
        scale * drift[1][1],
    )
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
/// This is intentionally a thin public boundary over the same row-program
/// derivatives the fitter's observed Hessian uses, so diagnostics cannot drift
/// onto a second approximation of the dispersion likelihood.
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
    let ([l_m, l_d], [l_mm, l_md, l_dd]) = dispersion_eta_loglik_second(kind, y, eta_mu, eta_d);
    let scale = -prior_weight;
    let geometry = DispersionAloRowGeometry {
        nll_score: [scale * l_m, scale * l_d],
        observed_hessian: [[scale * l_mm, scale * l_md], [scale * l_md, scale * l_dd]],
    };
    if geometry
        .nll_score
        .iter()
        .chain(geometry.observed_hessian.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(GamlssError::row_geometry_unrepresentable(row, "dispersion-family ALO row geometry", eta_mu, f64::NAN));
    }
    Ok(geometry)
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
            // The score reads ψ at θ + y and θ, and the precision information
            // below reads ψ′ at θ, so θ's stack carries both from one recurrence.
            // The score reads stack entries through the first.
            let theta_stack = gam_math::special::polygamma_stack(theta, 2);
            let [score_mu, score_eta] = DispersionRowStacks::negative_binomial(
                yi,
                mu,
                theta,
                gam_math::special::polygamma_stack(theta + yi, 1),
                theta_stack,
            )
            .order2()
            .1;
            let mean_weight = wi * mean_eta_information;
            let mean_response = em + score_mu / mean_eta_information;
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
            // carries the EXACT score (the row program's `∂ℓ/∂η_d`), so the
            // penalized stationary point (score = 0) is byte-unchanged — this is
            // Fisher scoring, which only re-conditions the inner solve and never
            // shifts the optimum. The observed channel `_info_theta_observed` is no
            // longer consumed for the weight.
            // #1591-follow-up: the information reads ψ′(θ) off θ's score stack and
            // evaluates only ψ′(θ+μ) itself; an earlier form built the full
            // order-1..5 polygamma stack, read index 0 and discarded four of five
            // per call (8 wasted polygamma evaluations per NB2 row).
            let eta_information = nb_log_precision_fisher_jensen(mu, theta, theta_stack[1]);
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
            let loglik = dispersion_gamma_loglik(yi, yi, mu, nu, wi);
            // The shape information −ℓ_νν = ψ′(ν) − 1/ν is positive for ν > 0 and
            // free of y, so it is the Fisher information too; the mean channel's
            // is ν. The scores are the row program's gradient in (η_μ, η_d).
            // One stack at ν carries the score's ψ and the information's ψ′.
            let nu_stack = gam_math::special::polygamma_stack(nu, 2);
            let [score_mu, score_eta] =
                DispersionRowStacks::gamma(yi, em, ed, nu, nu_stack).order2().1;
            let info_nu = nu_stack[1] - nu.recip();
            let mean_weight = wi * nu;
            let mean_response = em + score_mu / nu;
            let disp_weight = wi * nu * nu * info_nu;
            let disp_response = ed + score_eta / (nu * nu * info_nu);
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
            let loglik = dispersion_beta_loglik(yi, mu, phi, wi);
            let one_minus_mu = 1.0 - mu;
            let a = mu * phi;
            let b = one_minus_mu * phi;
            // Fisher information of Beta(a, b) with a = μφ and b = (1 − μ)φ:
            //   I_μμ = φ² (ψ′(a) + ψ′(b)),  I_φφ = μ² ψ′(a) + (1 − μ)² ψ′(b) − ψ′(φ),
            // carried into (η_μ, η_d) by dμ/dη_μ = q and dφ/dη_d = φ. The scores
            // are the row program's gradient in (η_μ, η_d).
            // One stack at each of φ, a and b carries the score's ψ and the
            // information's ψ′.
            let phi_stack = gam_math::special::polygamma_stack(phi, 2);
            let a_stack = gam_math::special::polygamma_stack(a, 2);
            let b_stack = gam_math::special::polygamma_stack(b, 2);
            let [score_mu, score_eta] =
                DispersionRowStacks::beta(yi, &logit, phi, phi_stack, a_stack, b_stack)
                    .order2()
                    .1;
            let tri_a = a_stack[1];
            let tri_b = b_stack[1];
            let tri_phi = phi_stack[1];
            let info_mu = phi * phi * (tri_a + tri_b);
            let info_phi = mu * mu * tri_a + one_minus_mu * one_minus_mu * tri_b - tri_phi;
            let mean_weight = wi * q * q * info_mu;
            let mean_response = em + score_mu / (q * q * info_mu);
            let disp_weight = wi * phi * phi * info_phi;
            let disp_response = ed + score_eta / (phi * phi * info_phi);
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
            let loglik = dispersion_tweedie_loglik(yi, em, ed, p, wi);
            // κ = 1/φ is the reciprocal the log-likelihood forms, and the stacks form
            // the deviance's power terms as it does, so the scores (the row
            // program's gradient in (η_μ, η_d)) share its divisions.
            let kappa = 1.0 / phi;
            let [score_mu, score_eta] = DispersionRowStacks::tweedie(yi, p, mu, kappa).order2().1;
            // Mean channel: the Fisher weight `μ^{2−p}/φ` (the mean block is
            // Fisher-orthogonal to the dispersion block in this parameterization).
            let mean_information = mu.powf(two_minus_p) * kappa;
            let mean_weight = wi * mean_information;
            let mean_response = em + score_mu / mean_information;
            // Dispersion channel in η_d, where φ = exp(−η_d). Positive y
            // (saddlepoint density ℓ = −dev/(2φ) − ½ ln(2πφ) − ½ p ln y) keeps the
            // constant curvature ½. The point mass at y = 0 (ℓ = −c/φ with
            // c = μ^{2−p}/(2−p)) uses its observed information c/φ.
            let curvature_eta = if yi > 0.0 {
                0.5
            } else {
                mu.powf(two_minus_p) * (1.0 / two_minus_p) * kappa
            };
            let disp_weight = wi * curvature_eta;
            let disp_response = ed + score_eta / curvature_eta;
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
    /// Whether this member's Jeffreys/Firth prior is armed. A fit arms it only
    /// on the unarmed fit's own evidence, through
    /// `fit_custom_family_arming_on_evidence` (#979).
    pub(crate) jeffreys_armed: bool,
}

impl DispersionGlmLocationScaleFamily {
    pub(crate) const BLOCK_MEAN: usize = 0;
    pub(crate) const BLOCK_DISP: usize = 1;
}

impl crate::custom_family::JeffreysArming for DispersionGlmLocationScaleFamily {
    fn with_jeffreys_armed(
        &self,
        evidence: Option<&gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    ) -> Self {
        Self {
            jeffreys_armed: evidence.is_some(),
            ..self.clone()
        }
    }
}

impl CustomFamily for DispersionGlmLocationScaleFamily {
    // The self-limiting Jeffreys/Firth curvature bounds a coefficient the data do
    // not, but it is armed only when the unarmed fit proves it is needed (#979).
    fn joint_jeffreys_term_required(&self) -> bool {
        self.jeffreys_armed
    }

    /// The unscaled family deviance `2·Σ wᵢ d(yᵢ, μ̂ᵢ; θ̂ᵢ)` evaluated row by row
    /// through the SAME per-row oracle the standard PIRLS path reports from,
    /// with each row's fitted precision channel (`θᵢ`, `φᵢ`) supplied where the
    /// unit deviance depends on it (negative-binomial, beta) and none where it
    /// does not (gamma, Tweedie). One definition, two paths (#2786).
    fn classical_deviance(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
        validate_block_count::<GamlssError>(self.kind.family_tag(), 2, block_states.len())?;
        let eta_mu = &block_states[Self::BLOCK_MEAN].eta;
        let eta_d = &block_states[Self::BLOCK_DISP].eta;
        let n = self.y.len();
        if eta_mu.len() != n || eta_d.len() != n || self.weights.len() != n {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "{} deviance row-count mismatch: y={n}, eta_mu={}, eta_d={}, weights={}",
                    self.kind.family_tag(),
                    eta_mu.len(),
                    eta_d.len(),
                    self.weights.len()
                ),
            }
            .into());
        }
        let mut half = 0.0_f64;
        for i in 0..n {
            let (response, link) = match self.kind {
                DispersionFamilyKind::NegativeBinomial => (
                    ResponseFamily::NegativeBinomial {
                        theta: eta_d[i].exp(),
                        theta_fixed: true,
                    },
                    StandardLink::Log,
                ),
                DispersionFamilyKind::Gamma => (ResponseFamily::Gamma, StandardLink::Log),
                DispersionFamilyKind::Beta => (
                    ResponseFamily::Beta {
                        phi: eta_d[i].exp(),
                    },
                    StandardLink::Logit,
                ),
                DispersionFamilyKind::Tweedie { p } => {
                    (ResponseFamily::Tweedie { p }, StandardLink::Log)
                }
            };
            let inverse_link = InverseLink::Standard(link);
            let likelihood = gam_spec::GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                response,
                inverse_link.clone(),
            ));
            let row = gam_solve::pirls::deviance_eta_row_with_log_measure_scale(
                i,
                self.y[i],
                eta_mu[i],
                &likelihood,
                &inverse_link,
                self.weights[i],
                0.0,
            )
            .map_err(|error| error.to_string())?;
            half += row.half_deviance;
        }
        if !half.is_finite() {
            return Err(format!(
                "{} classical deviance is non-finite ({half})",
                self.kind.family_tag()
            ));
        }
        Ok(Some(2.0 * half))
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
                return Err(GamlssError::row_geometry_unrepresentable(
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
                return Err(GamlssError::row_geometry_unrepresentable(
                    i,
                    "dispersion-family row log likelihood",
                    eta_mu[i],
                    loglik,
                ));
            }
            ll += loglik;
            if !ll.is_finite() {
                return Err(GamlssError::row_geometry_unrepresentable(
                    i,
                    "dispersion-family cumulative log likelihood",
                    eta_mu[i],
                    ll,
                ));
            }
        }
        Ok(ll)
    }

    /// Exact joint coefficient-space Hessian `H_L = -∇²log L` in flattened
    /// `[mean | log-precision]` block order.
    ///
    /// All four members assemble `Xᵀ diag(W) X` blocks from the per-row
    /// OBSERVED η-space second derivatives
    /// (`dispersion_row_observed_hessian_weights`): the full mean-link and
    /// precision-link chains, the inverse-link second-derivative terms, and
    /// the mean/dispersion cross curvature are all carried exactly by the
    /// member's row program. This is deliberately NOT the Fisher-scoring
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

        // Per-row observed `(∂²/∂η_μ², ∂²/∂η_μ∂η_d, ∂²/∂η_d²)` weights, one
        // row-program second-order evaluation each. Row-independent, so fan it
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
                    return Err(GamlssError::row_geometry_unrepresentable(i, quantity, eta, value));
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

    /// Exact β-directional derivative of the observed joint Hessian,
    /// `D_β H_L[u]`, assembled row-wise from each member's contracted third
    /// row-program surface (`dispersion_row_observed_hessian_directional`): with per-row η-motion
    /// `du_μ = X_μ u_μ`, `du_d = X_d u_d`, each Hessian channel drifts by the
    /// exact tensor contraction `dW_ab = Σ_c (∂³NLL/∂η_a∂η_b∂η_c) du_c`, and
    /// the blocks are the same `Xᵀ diag(dW) X` grams the Hessian itself uses.
    ///
    /// Supplying this hook (instead of the previous silent `None`) is
    /// load-bearing twice over:
    ///  * the inner Firth/Jeffreys term `joint_jeffreys_term` builds its
    ///    `∇Φ`/`H_Φ` from `Hdot[e_k]`; with `None` it degrades to `(Φ, 0, 0)`,
    ///    so the inner merit contains a β-dependent `−Φ` the KKT gradient
    ///    cannot see — the objective↔gradient desync behind the flat-residual
    ///    inner stall (and, post rail-face certification, the λ=∞ null-model
    ///    collapse) on Beta/NB/Tweedie dispersion location-scale fits (#1561);
    ///  * the outer profiled-Laplace mode-response correction
    ///    (`dot H_k = A_k + D_β H_L[u_k]`) consumes the same object.
    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>(self.kind.family_tag(), 2, block_states.len())?;
        if specs.len() != 2 {
            return Err(format!(
                "{} joint Hessian directional derivative expects 2 specs, got {}",
                self.kind.family_tag(),
                specs.len()
            ));
        }
        let eta_mu = &block_states[Self::BLOCK_MEAN].eta;
        let eta_d = &block_states[Self::BLOCK_DISP].eta;
        let n = self.y.len();
        if eta_mu.len() != n || eta_d.len() != n || self.weights.len() != n {
            return Err(format!(
                "{} joint Hessian directional derivative row-count mismatch: y={n}, eta_mu={}, eta_d={}, weights={}",
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
        let mean_spec = &specs[Self::BLOCK_MEAN];
        let disp_spec = &specs[Self::BLOCK_DISP];
        if mean_spec.design.nrows() != n || disp_spec.design.nrows() != n {
            return Err(format!(
                "{} joint Hessian directional derivative design row mismatch: y={n}, mean rows={}, precision rows={}",
                self.kind.family_tag(),
                mean_spec.design.nrows(),
                disp_spec.design.nrows()
            ));
        }
        let p_mean = mean_spec.design.ncols();
        let p_disp = disp_spec.design.ncols();
        if d_beta_flat.len() != p_mean + p_disp {
            return Err(format!(
                "{} joint Hessian directional derivative direction length mismatch: got {}, expected {}",
                self.kind.family_tag(),
                d_beta_flat.len(),
                p_mean + p_disp
            ));
        }
        let u_mu = d_beta_flat.slice(s![0..p_mean]).to_owned();
        let u_d = d_beta_flat.slice(s![p_mean..p_mean + p_disp]).to_owned();
        // η-motion of the direction: the offset is β-independent, so
        // `dη_b = X_b u_b` exactly.
        let du_mu = mean_spec.design.apply(&u_mu);
        let du_d = disp_spec.design.apply(&u_d);
        let directional: Vec<(f64, f64, f64)> =
            if rayon::current_thread_index().is_none() && n > DISPERSION_PARALLEL_ROW_THRESHOLD {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        dispersion_row_observed_hessian_directional(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                            du_mu[i],
                            du_d[i],
                        )
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        dispersion_row_observed_hessian_directional(
                            self.kind,
                            self.y[i],
                            eta_mu[i],
                            eta_d[i],
                            self.weights[i],
                            du_mu[i],
                            du_d[i],
                        )
                    })
                    .collect()
            };
        for (i, &(d_mm, d_md, d_dd)) in directional.iter().enumerate() {
            for (quantity, eta, value) in [
                (
                    "dispersion-family directional mean curvature drift",
                    eta_mu[i],
                    d_mm,
                ),
                (
                    "dispersion-family directional cross curvature drift",
                    eta_mu[i],
                    d_md,
                ),
                (
                    "dispersion-family directional precision curvature drift",
                    eta_d[i],
                    d_dd,
                ),
            ] {
                if !value.is_finite() {
                    return Err(GamlssError::row_geometry_unrepresentable(i, quantity, eta, value));
                }
            }
        }
        let mean_drift = Array1::from_shape_fn(n, |i| directional[i].0);
        let cross_drift = Array1::from_shape_fn(n, |i| directional[i].1);
        let disp_drift = Array1::from_shape_fn(n, |i| directional[i].2);
        let dh_mean = xt_diag_x_design(&mean_spec.design, &mean_drift)?;
        let dh_cross = xt_diag_y_design(&mean_spec.design, &cross_drift, &disp_spec.design)?;
        let dh_disp = xt_diag_x_design(&disp_spec.design, &disp_drift)?;
        let total = p_mean + p_disp;
        let mut dh = Array2::<f64>::zeros((total, total));
        dh.slice_mut(s![0..p_mean, 0..p_mean]).assign(&dh_mean);
        dh.slice_mut(s![0..p_mean, p_mean..total]).assign(&dh_cross);
        dh.slice_mut(s![p_mean..total, p_mean..total])
            .assign(&dh_disp);
        mirror_upper_to_lower(&mut dh);
        Ok(Some(dh))
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

/// Warm start for a dispersion location-scale fit: each channel's target is the
/// iteratively reweighted least-squares working response of its link, projected
/// onto the channel's block; the block-cyclic IRLS then refines both jointly.
///
/// Mean. The working response `g(m) + (y − m)·g'(m)` evaluated at the saturated
/// fit `m = y` is the link transform `g(y)` itself. For the logit mean that covers
/// every admissible response (`0 < y < 1`), and for the log mean every positive
/// response. A zero count, where `m = y` lies outside the log link's domain, takes
/// the working response at the pooled mean `μ̄` instead, `ln μ̄ + (0 − μ̄)/μ̄`. No
/// response needs a floor or a clamp.
///
/// Precision. Smyth's double GLM: `dᵢ` is the moment statistic whose expectation is
/// the member's dispersion `ϕ` at the seeded mean (`e²/μ²` Gamma, `(e² − μ)/μ²`
/// negative binomial, `e²/μ^p` Tweedie), pooled to `ϕ̄`, and the log-link working
/// response `ln ϕ̄ + (dᵢ − ϕ̄)/ϕ̄` is negated onto the log-precision scale. It is
/// linear in `dᵢ`, so a row whose excess moment is zero or negative gives a finite
/// target rather than a pole, and no box is needed; a pooled dispersion that is not
/// positive leaves the precision with no finite seed and is refused. Beta's mean
/// and precision scores are not Fisher-orthogonal, so an outlying row near 0 or 1
/// would pull the coupled mean before the joint likelihood settles; it keeps a
/// constant seed at the pooled moment's precision (`e²/(μ(1 − μ))` estimates
/// `1/(1 + φ)`), refused outside `0 < ϕ̄ < 1`.
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
    let tag = kind.family_tag();
    // Rows with zero prior weight are exempt from the support check, so they enter
    // no moment, and their targets take the pooled value so they stay finite.
    let mut weight_sum = 0.0_f64;
    let mut weighted_response = 0.0_f64;
    for (&yi, &wi) in y.iter().zip(weights.iter()) {
        if wi > 0.0 {
            weight_sum += wi;
            weighted_response += wi * yi;
        }
    }
    if !(weight_sum > 0.0) {
        return Err(format!(
            "{tag}: the warm start needs positive total prior weight; got {weight_sum}"
        ));
    }
    let pooled_mean = weighted_response / weight_sum;
    let mean_beta = if let Some(beta) = mean_beta_hint {
        beta.clone()
    } else {
        let target = if kind.mean_is_logit() {
            let pooled_logit = (pooled_mean / (1.0 - pooled_mean)).ln();
            Array1::from_shape_fn(y.len(), |i| {
                if weights[i] > 0.0 {
                    (y[i] / (1.0 - y[i])).ln()
                } else {
                    pooled_logit
                }
            })
        } else {
            if !(pooled_mean > 0.0) {
                return Err(format!(
                    "{tag}: the weighted mean response is {pooled_mean}, so the log mean has no \
                     finite seed"
                ));
            }
            let log_pooled_mean = pooled_mean.ln();
            // The working response at the pooled mean for a zero count.
            let zero_count_target = log_pooled_mean + (0.0 - pooled_mean) / pooled_mean;
            Array1::from_shape_fn(y.len(), |i| {
                if !(weights[i] > 0.0) {
                    log_pooled_mean
                } else if y[i] > 0.0 {
                    y[i].ln()
                } else {
                    zero_count_target
                }
            })
        };
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
        let mean_eta = mean_block.design.apply(&mean_beta) + &mean_block.offset;
        let mut moment = Array1::<f64>::zeros(y.len());
        let mut weighted_moment = 0.0_f64;
        for i in 0..y.len() {
            if !(weights[i] > 0.0) {
                continue;
            }
            let mu = if kind.mean_is_logit() {
                gam_linalg::utils::stable_logistic(mean_eta[i])
            } else {
                mean_eta[i].exp()
            };
            let e2 = (y[i] - mu) * (y[i] - mu);
            let d = match kind {
                DispersionFamilyKind::NegativeBinomial => (e2 - mu) / (mu * mu),
                DispersionFamilyKind::Gamma => e2 / (mu * mu),
                DispersionFamilyKind::Tweedie { p } => e2 / mu.powf(p),
                DispersionFamilyKind::Beta => e2 / (mu * (1.0 - mu)),
            };
            if !d.is_finite() {
                return Err(format!(
                    "{tag}: the dispersion moment is not representable at row {i} (seeded mean \
                     {mu:e}, response {})",
                    y[i]
                ));
            }
            moment[i] = d;
            weighted_moment += weights[i] * d;
        }
        let pooled_dispersion = weighted_moment / weight_sum;
        let target = match kind {
            DispersionFamilyKind::Beta => {
                if !(pooled_dispersion > 0.0 && pooled_dispersion < 1.0) {
                    return Err(format!(
                        "{tag}: the pooled moment e²/(μ(1 − μ)) about the seeded mean is \
                         {pooled_dispersion}, outside (0, 1), so the Beta precision has no \
                         finite seed"
                    ));
                }
                Array1::from_elem(y.len(), (1.0 / pooled_dispersion - 1.0).ln())
            }
            _ => {
                if !(pooled_dispersion > 0.0) {
                    return Err(format!(
                        "{tag}: the pooled dispersion moment about the seeded mean is \
                         {pooled_dispersion}; with no variance beyond the member's variance \
                         function the precision has no finite seed"
                    ));
                }
                let log_pooled = pooled_dispersion.ln();
                Array1::from_shape_fn(y.len(), |i| {
                    if weights[i] > 0.0 {
                        -(log_pooled + (moment[i] - pooled_dispersion) / pooled_dispersion)
                    } else {
                        -log_pooled
                    }
                })
            }
        };
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

impl LocationScaleFamilyBuilder for DispersionGlmLocationScaleTermBuilder {
    type Family = DispersionGlmLocationScaleFamily;

    fn fit_blocks(
        &self,
        family: &Self::Family,
        blocks: &[crate::custom_family::ParameterBlockSpec],
        options: &crate::custom_family::BlockwiseFitOptions,
    ) -> Result<UnifiedFitResult, FitFailure> {
        crate::custom_family::fit_custom_family_arming_on_evidence(family, blocks, options)
            .map_err(FitFailure::from)
    }

    fn meanspec(&self) -> &TermCollectionSpec {
        &self.meanspec
    }

    fn noisespec(&self) -> &TermCollectionSpec {
        &self.noisespec
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

        // SPEC-5: the log-precision block is penalized by its formula-native
        // function-space penalties only (a smooth term carries its own
        // REML-selected function-metric null-space shrinkage when
        // `double_penalty=true`, the default). The previous full-span
        // `identity_penalty` ridge was a basis-dependent coefficient-space prior
        // that double-penalized the range space and shrank the fitted dispersion
        // surface toward its coordinate origin — the exact over-shrinkage the
        // Gaussian location-scale path dropped in `de5599435` (#1561). Mirroring
        // that path here removes the extra REML smoothing coordinate the ridge
        // introduced, so the coupled inner solve no longer optimizes the
        // dispersion smoothing against a phantom full-span penalty.
        let disp_penalties = noise_design.penalties_as_penalty_matrix();
        let disp_nullspace = noise_design.nullspace_dims.clone();
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
            jeffreys_armed: true,
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
/// likelihood lives in `dispersion_row_kernel`.
pub fn fit_dispersion_glm_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: DispersionGlmLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BlockwiseTermFitResult, FitFailure> {
    if let DispersionFamilyKind::Tweedie { p } = spec.kind {
        if !(p.is_finite() && p > 1.0 && p < 2.0) {
            return Err(input_failure(format!(
                "Tweedie location-scale requires a variance power strictly in (1, 2); got p={p}"
            )));
        }
    }
    // Both validators refuse only what the caller supplied (#2937).
    validate_dispersion_family_data(spec.kind, &spec.y, &spec.weights).map_err(input_failure)?;
    validate_dispersion_spatial_hyperparameter_request(
        spec.kind,
        &spec.meanspec,
        &spec.log_dispspec,
        kappa_options,
    )
    .map_err(|err| input_failure(err.to_string()))?;
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
mod tests_row_program_932;

#[cfg(test)]
mod tests {
    use super::test_support::{
        dispersion_beta_nll_order2, dispersion_gamma_disp_order2, dispersion_gamma_nll_order2,
        dispersion_nb_nll_order2, dispersion_tweedie_disp_order2,
    };
    use super::*;
    use crate::gamlss::test_support::{
        dispersion_eta_nll_order2, dispersion_eta_nll_order3, dispersion_tweedie_nll_generic,
        order2_ln_gamma,
    };
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
        let a = gam_math::special::digamma(nu) - nu.ln() - 1.0 + mu.ln() - y.ln() + ratio;
        let expected_score = [weight * nu * (1.0 - ratio), weight * nu * a];
        let expected_hessian = [
            [weight * nu * ratio, weight * nu * (1.0 - ratio)],
            [
                weight * nu * (1.0 - ratio),
                weight * nu * (a + nu * gam_math::special::trigamma(nu) - 1.0),
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

    /// Pruned single-axis NB2 dispersion tower: `θ` is the sole jet variable
    /// (axis 0), `μ` a constant. `value`/`g[0]`/`h[0][0]` reproduce the consumed
    /// `value`/`g[1]`/`h[1][1]` of `dispersion_nb_nll_order2` bit-for-bit. The
    /// `Order2` oracle pin for `prune_towers_match_dense_all_channels`.
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
        assert!(public_error.to_string().contains("will not silently freeze"));
        assert_eq!(
            public_error.category(),
            gam_problem::FailureCategory::Input,
            "a refused spatial request is the caller's configuration (#2937)"
        );

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
        // The kernel takes LOG-space inputs and re-exponentiates: its internal
        // precision is `huge.ln().exp()`, which differs from `huge` by the
        // exp∘ln round-trip (~|ln huge|·EPSILON ≈ 460·EPSILON relative at
        // η≈460), so dividing by the pre-log `huge` injects ~50·EPSILON of pure
        // round-trip noise. Reference the precision the kernel actually sees:
        // with μ==θ, `mean_weight = θ/(1+θ/μ) = θ/2` is exact (a bit-for-bit
        // exponent decrement), so `mean_weight / precision == 0.5` exactly and
        // the tight tolerance stays a genuine balanced-tail geometry guard.
        let precision = huge.ln().exp();
        assert!((kernel.mean_weight / precision - 0.5).abs() <= 8.0 * f64::EPSILON);
        assert!(kernel.disp_weight.is_finite() && kernel.disp_weight > 0.0);

        let eta_info =
            nb_log_precision_fisher_jensen(1.0, 1.0e17, gam_math::special::trigamma(1.0e17));
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
                jeffreys_armed: true,
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

    /// #932: the row kernel reads the Gamma, Beta and Tweedie dispersion scores
    /// from the row programs and forms their information in closed form. Its
    /// working sets must match the ones the pruned jet towers produce, across
    /// randomized rows and both Tweedie density branches.
    #[test]
    fn row_kernel_closed_form_dispersion_channels_match_the_towers() {
        let mut state: u64 = 0x2901_D15C_0A11_0001;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let close = |label: &str, hand: f64, tower: f64| {
            let band = 1e-10 * (1.0 + hand.abs().max(tower.abs()));
            assert!(
                (hand - tower).abs() <= band,
                "{label}: row program {hand:.17e} vs tower {tower:.17e}"
            );
        };
        for _ in 0..500 {
            let wi = 0.25 + 3.0 * next();
            let em = -3.0 + 6.0 * next();
            let ed = -3.0 + 6.0 * next();

            let yi = 0.01 + 8.0 * next();
            let row = dispersion_row_kernel(DispersionFamilyKind::Gamma, yi, em, ed, wi);
            let (mu, nu) = (em.exp(), ed.exp());
            let tower = dispersion_gamma_disp_order2(yi, yi, mu, nu, wi);
            let s_nu = -tower.g()[0] / wi;
            let info_nu = tower.h()[0][0] / wi;
            close("gamma loglik", row.loglik, -tower.value());
            close("gamma disp weight", row.disp_weight, wi * nu * nu * info_nu);
            close(
                "gamma disp response",
                row.disp_response,
                ed + s_nu / (nu * info_nu),
            );

            let yi = 0.005 + 0.99 * next();
            let em_beta = -2.5 + 5.0 * next();
            let ed_beta = -1.0 + 4.0 * next();
            let row = dispersion_row_kernel(DispersionFamilyKind::Beta, yi, em_beta, ed_beta, wi);
            let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em_beta);
            let (mu, phi) = (logit.mu, ed_beta.exp());
            let tower = dispersion_beta_nll_order2(yi, mu, phi, wi);
            let score_mu = -tower.g()[0] / wi;
            let s_phi = -tower.g()[1] / wi;
            let tri_a = gam_math::special::trigamma(mu * phi);
            let tri_b = gam_math::special::trigamma((1.0 - mu) * phi);
            let info_mu = phi * phi * (tri_a + tri_b);
            let info_phi = mu * mu * tri_a + (1.0 - mu) * (1.0 - mu) * tri_b
                - gam_math::special::trigamma(phi);
            close("beta loglik", row.loglik, -tower.value());
            close(
                "beta mean response",
                row.mean_response,
                em_beta + score_mu / (logit.d1 * info_mu),
            );
            close(
                "beta disp response",
                row.disp_response,
                ed_beta + s_phi / (phi * info_phi),
            );

            let p = 1.1 + 0.8 * next();
            for yi in [0.0, 0.01 + 9.0 * next()] {
                let row = dispersion_row_kernel(DispersionFamilyKind::Tweedie { p }, yi, em, ed, wi);
                let tower = dispersion_tweedie_disp_order2(yi, em, ed, p, wi);
                let s_eta = -tower.g()[0] / wi;
                let curvature_eta = if yi > 0.0 {
                    0.5
                } else {
                    tower.h()[0][0] / wi
                };
                close("tweedie loglik", row.loglik, -tower.value());
                close("tweedie disp weight", row.disp_weight, wi * curvature_eta);
                close(
                    "tweedie disp response",
                    row.disp_response,
                    ed + s_eta / curvature_eta,
                );
            }
        }
    }

    /// #932: the observed η-space Hessian, its directional derivative and the
    /// saved-model ALO geometry come from each member's row program. They must
    /// match the independent jet towers, on randomized rows of every member and
    /// both Tweedie density branches.
    #[test]
    fn eta_space_row_program_derivatives_match_the_towers() {
        let mut state: u64 = 0x2901_E7A5_0A11_0002;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let close = |label: &str, hand: f64, tower: f64| {
            let band = 1e-9 * (1.0 + hand.abs().max(tower.abs()));
            assert!(
                (hand - tower).abs() <= band,
                "{label}: row program {hand:.17e} vs tower {tower:.17e}"
            );
        };
        for _ in 0..400 {
            let wi = 0.25 + 3.0 * next();
            let em = -2.5 + 5.0 * next();
            let ed = -2.0 + 4.0 * next();
            let du_mu = -1.0 + 2.0 * next();
            let du_d = -1.0 + 2.0 * next();
            let p = 1.1 + 0.8 * next();
            let rows = [
                (DispersionFamilyKind::NegativeBinomial, (next() * 12.0).floor()),
                (DispersionFamilyKind::Gamma, 0.01 + 8.0 * next()),
                (DispersionFamilyKind::Beta, 0.005 + 0.99 * next()),
                (DispersionFamilyKind::Tweedie { p }, 0.0),
                (DispersionFamilyKind::Tweedie { p }, 0.01 + 9.0 * next()),
            ];
            for (kind, yi) in rows {
                let label = format!("{kind:?} y={yi} em={em} ed={ed}");
                let tower2 = dispersion_eta_nll_order2(kind, yi, em, ed, wi);
                let g2 = tower2.g();
                let h2 = tower2.h();
                let (h_mm, h_md, h_dd) =
                    dispersion_row_observed_hessian_weights(kind, yi, em, ed, wi);
                close(&format!("{label} h_mm"), h_mm, h2[0][0]);
                close(&format!("{label} h_md"), h_md, h2[0][1]);
                close(&format!("{label} h_dd"), h_dd, h2[1][1]);
                let geometry = dispersion_alo_row_geometry(kind, 0, yi, em, ed, wi)
                    .expect("finite dispersion row geometry");
                close(&format!("{label} score_mu"), geometry.nll_score[0], g2[0]);
                close(&format!("{label} score_d"), geometry.nll_score[1], g2[1]);
                close(
                    &format!("{label} alo cross"),
                    geometry.observed_hessian[1][0],
                    h2[1][0],
                );
                let t3 = dispersion_eta_nll_order3(kind, yi, em, ed, wi).t3;
                let (d_mm, d_md, d_dd) =
                    dispersion_row_observed_hessian_directional(kind, yi, em, ed, wi, du_mu, du_d);
                close(
                    &format!("{label} dH_mm"),
                    d_mm,
                    t3[0][0][0] * du_mu + t3[0][0][1] * du_d,
                );
                close(
                    &format!("{label} dH_md"),
                    d_md,
                    t3[0][1][0] * du_mu + t3[0][1][1] * du_d,
                );
                close(
                    &format!("{label} dH_dd"),
                    d_dd,
                    t3[1][1][0] * du_mu + t3[1][1][1] * du_d,
                );
            }
        }
    }
}
