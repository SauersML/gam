//! #913: dispersion-channel GAMLSS location-scale families.
//!
//! Extracted from `gamlss.rs` (issue #780); this module now owns the
//! dispersion-channel joint-curvature corrections.

use super::weighted_design_products::{mirror_upper_to_lower, xt_diag_x_design, xt_diag_y_design};
use super::{
    BlockwiseTermFitResult, GamlssLambdaLayout, LOCATION_SCALE_N_OUTPUTS,
    LocationScaleFamilyBuilder, build_location_scale_block, fit_location_scale_terms,
    identity_penalty, solve_penalizedweighted_projection,
};
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
};
use crate::families::block_layout::block_count::validate_block_count;
use crate::gamlss::GamlssError;
use crate::matrix::LinearOperator;
use crate::model_types::UnifiedFitResult;
use crate::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
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
    pub fn base_link(self) -> crate::types::InverseLink {
        use crate::types::{InverseLink, StandardLink};
        if self.mean_is_logit() {
            InverseLink::Standard(StandardLink::Logit)
        } else {
            InverseLink::Standard(StandardLink::Log)
        }
    }

    /// The family's canonical [`LikelihoodSpec`] (mean response × mean link).
    /// The overdispersion parameter is estimated by the log-precision channel,
    /// so the response-family placeholder parameters (`phi`, `theta`) mirror
    /// the [`resolve_family`](crate::solver::fit_orchestration::resolve_family) defaults
    /// and are not consumed as fixed values at predict time. This is the single
    /// source of truth for the persisted location-scale likelihood so the CLI
    /// and FFI save paths cannot diverge.
    pub fn likelihood_spec(self) -> crate::types::LikelihoodSpec {
        use crate::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
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

/// `η` magnitude clamp shared by both channels (mirrors PIRLS `ETA_CLAMP`):
/// keeps `exp(η)` and the logit jet away from overflow while staying in the
/// smooth interior of every link.
pub(super) const DISPERSION_ETA_CLAMP: f64 = 30.0;
/// Floor for a per-row IRLS working weight / curvature so the block normal
/// equations stay positive-definite. The working *response* always carries the
/// exact score, so the stationary point (penalised score = 0) is independent
/// of this floor; it only conditions the inner solve.
pub(super) const DISPERSION_MIN_CURVATURE: f64 = 1e-12;

/// Per-row working quantities for both channels at the current `(η_μ, η_d)`.
pub(super) struct DispersionRowKernel {
    pub(super) loglik: f64,
    pub(super) mean_weight: f64,
    pub(super) mean_response: f64,
    pub(super) disp_weight: f64,
    pub(super) disp_response: f64,
}

#[inline]
pub(crate) fn dispersion_nb_nll_tower(
    yi: f64,
    mu_value: f64,
    theta_value: f64,
    wi: f64,
) -> crate::families::jet_tower::Tower4<2> {
    use crate::families::jet_tower::Tower4;
    let mu = Tower4::<2>::variable(mu_value, 0);
    let theta = Tower4::<2>::variable(theta_value, 1);
    let tpm = theta + mu;
    let loglik = (theta + yi).ln_gamma() - theta.ln_gamma() - ln_gamma(yi + 1.0)
        + theta * theta.ln()
        - theta * tpm.ln()
        + mu.ln() * yi
        - tpm.ln() * yi;
    -loglik * wi
}

#[inline]
pub(crate) fn dispersion_gamma_nll_tower(
    yi: f64,
    y_pos: f64,
    mu_value: f64,
    nu_value: f64,
    wi: f64,
) -> crate::families::jet_tower::Tower4<2> {
    use crate::families::jet_tower::Tower4;
    let mu = Tower4::<2>::variable(mu_value, 0);
    let nu = Tower4::<2>::variable(nu_value, 1);
    let loglik = nu * nu.ln() - nu * mu.ln() - nu.ln_gamma() + (nu - 1.0) * y_pos.ln()
        - nu * (mu.recip() * yi);
    -loglik * wi
}

#[inline]
pub(crate) fn dispersion_beta_nll_tower(
    yi: f64,
    mu_value: f64,
    phi_value: f64,
    wi: f64,
) -> crate::families::jet_tower::Tower4<2> {
    use crate::families::jet_tower::Tower4;
    let mu = Tower4::<2>::variable(mu_value, 0);
    let phi = Tower4::<2>::variable(phi_value, 1);
    let one_minus_mu = Tower4::<2>::constant(1.0) - mu;
    let yc = yi.clamp(1e-12, 1.0 - 1e-12);
    let a = mu * phi;
    let b = one_minus_mu * phi;
    let loglik = phi.ln_gamma() - a.ln_gamma() - b.ln_gamma()
        + (a - 1.0) * yc.ln()
        + (b - 1.0) * (1.0 - yc).ln();
    -loglik * wi
}

/// Tweedie compound Poisson–Gamma row NLL written ONCE over `Tower4<2>`, seeded
/// directly on the PREDICTOR primaries `(η_μ, η_d)` (#932).
///
/// Unlike the NB/Gamma/Beta towers — which seed on the natural parameters and
/// let the caller apply the precision→η chain via the Fisher-orthogonal
/// `precision²·info` shortcut — this tower carries `μ = exp(η_μ)` and
/// `φ = exp(−η_d)` INSIDE the program, so `tower.g[1]` / `tower.h[1][1]` are
/// the η_d-space score and OBSERVED information directly, with the nonlinear
/// `∂²φ/∂η_d²` chain correction the hand path documented (the `y = 0` branch's
/// `2c/φ − c/φ = c/φ` cancellation) mechanically carried rather than re-derived.
///
/// Both density branches are smooth in `(η_μ, η_d)`:
/// * `y > 0` — the Nelder–Pregibon saddlepoint density
///   `ℓ = w·[ −dev/(2φ) − ½ln(2πφ) − ½p·ln y ]` with the unit deviance
///   `dev = 2·[ y^{2−p}/((1−p)(2−p)) − y·μ^{1−p}/(1−p) + μ^{2−p}/(2−p) ]`.
/// * `y = 0` — the exact compound-Poisson point mass
///   `ℓ = w·[ −μ^{2−p}/(φ(2−p)) ]`.
///
/// `μ` and `φ` enter the deviance only through `powf` and a `recip`, whose
/// `[f64; 5]` derivative stacks the tower owns, so no primitive is re-derived:
/// only the Leibniz/Faà-di-Bruno composition is mechanized.
#[inline]
pub(crate) fn dispersion_tweedie_nll_tower(
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    p: f64,
    wi: f64,
) -> crate::families::jet_tower::Tower4<2> {
    use crate::families::jet_tower::Tower4;
    let one_minus_p = 1.0 - p;
    let two_minus_p = 2.0 - p;
    // μ = exp(η_μ), φ = exp(−η_d): the natural parameters as jets in the
    // predictor primaries, so the whole derivative tower is in η-space.
    let mu = Tower4::<2>::variable(eta_mu, 0).exp();
    let phi = (Tower4::<2>::variable(eta_d, 1) * (-1.0)).exp();
    if yi > 0.0 {
        let dev = (mu.powf(two_minus_p) * (1.0 / two_minus_p)
            - mu.powf(one_minus_p) * (yi / one_minus_p)
            + Tower4::<2>::constant(yi.powf(two_minus_p) / (one_minus_p * two_minus_p)))
            * 2.0;
        let loglik = dev * (phi.recip() * (-0.5))
            - (phi * (2.0 * std::f64::consts::PI)).ln() * 0.5
            - Tower4::<2>::constant(0.5 * p * yi.ln());
        loglik * (-wi)
    } else {
        // Exact point mass P(Y=0) = exp(−μ^{2−p}/(φ(2−p))).
        let c = mu.powf(two_minus_p) * (1.0 / two_minus_p);
        let loglik = c * phi.recip() * (-1.0);
        loglik * (-wi)
    }
}

#[inline]
pub(crate) fn beta_observed_cross_weight_eta(yi: f64, mu: f64, phi: f64, wi: f64) -> f64 {
    let q = (mu * (1.0 - mu)).max(1e-12);
    let tower = dispersion_beta_nll_tower(yi, mu, phi, wi);
    q * phi * tower.h[0][1]
}

#[inline]
pub(crate) fn dispersion_row_cross_weight(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> f64 {
    let wi = prior_weight.max(0.0);
    if wi == 0.0 {
        return 0.0;
    }
    let em = eta_mu.clamp(-DISPERSION_ETA_CLAMP, DISPERSION_ETA_CLAMP);
    let ed = eta_d.clamp(-DISPERSION_ETA_CLAMP, DISPERSION_ETA_CLAMP);
    match kind {
        DispersionFamilyKind::Beta => {
            let mu = (1.0 / (1.0 + (-em).exp())).clamp(1e-12, 1.0 - 1e-12);
            let phi = ed.exp().max(1e-12);
            beta_observed_cross_weight_eta(yi, mu, phi, wi)
        }
        DispersionFamilyKind::NegativeBinomial
        | DispersionFamilyKind::Gamma
        | DispersionFamilyKind::Tweedie { .. } => 0.0,
    }
}

#[inline]
pub(crate) fn tower_score_info(
    tower: &crate::families::jet_tower::Tower4<2>,
    idx: usize,
    wi: f64,
) -> (f64, f64) {
    if wi == 0.0 {
        (0.0, 0.0)
    } else {
        (-tower.g[idx] / wi, tower.h[idx][idx] / wi)
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
    let wi = prior_weight.max(0.0);
    let em = eta_mu.clamp(-DISPERSION_ETA_CLAMP, DISPERSION_ETA_CLAMP);
    let ed = eta_d.clamp(-DISPERSION_ETA_CLAMP, DISPERSION_ETA_CLAMP);
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp().max(1e-300);
            let theta = ed.exp().max(1e-12); // precision (size)
            let tpm = theta + mu;
            let tower = dispersion_nb_nll_tower(yi, mu, theta, wi);
            let (s_theta, info_theta_raw) = tower_score_info(&tower, 1, wi);
            let loglik = -tower.v;
            let info_mu = if wi == 0.0 {
                DISPERSION_MIN_CURVATURE
            } else {
                (theta / (mu * tpm)).max(DISPERSION_MIN_CURVATURE)
            };
            let score_mu = theta * (yi - mu) / (mu * tpm);
            let mean_weight = wi * mu * mu * info_mu;
            let mean_response = em + score_mu / (mu * info_mu);
            let info_theta = info_theta_raw;
            let info_pos = info_theta.max(DISPERSION_MIN_CURVATURE);
            let disp_weight = wi * theta * theta * info_pos;
            let disp_response = ed + s_theta / (theta * info_pos);
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp().max(1e-300);
            let nu = ed.exp().max(1e-12); // precision = shape ν
            let y_pos = yi.max(1e-300);
            let tower = dispersion_gamma_nll_tower(yi, y_pos, mu, nu, wi);
            let (s_nu, info_nu_raw) = tower_score_info(&tower, 1, wi);
            let loglik = -tower.v;
            let info_mu = if wi == 0.0 {
                DISPERSION_MIN_CURVATURE
            } else {
                (nu / (mu * mu)).max(DISPERSION_MIN_CURVATURE)
            };
            let score_mu = nu * (yi - mu) / (mu * mu);
            let mean_weight = wi * mu * mu * info_mu;
            let mean_response = em + score_mu / (mu * info_mu);
            let info_nu = info_nu_raw.max(DISPERSION_MIN_CURVATURE);
            let disp_weight = wi * nu * nu * info_nu;
            let disp_response = ed + s_nu / (nu * info_nu);
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
            let mu = (1.0 / (1.0 + (-em).exp())).clamp(1e-12, 1.0 - 1e-12);
            let phi = ed.exp().max(1e-12); // precision
            let q = (mu * (1.0 - mu)).max(1e-12); // dμ/dη
            let tower = dispersion_beta_nll_tower(yi, mu, phi, wi);
            let (score_mu, info_mu_raw) = tower_score_info(&tower, 0, wi);
            let (s_phi, info_phi_raw) = tower_score_info(&tower, 1, wi);
            let loglik = -tower.v;
            let info_mu = info_mu_raw.max(DISPERSION_MIN_CURVATURE);
            let mean_weight = wi * q * q * info_mu;
            let mean_response = em + score_mu / (q * info_mu);
            let info_phi = info_phi_raw.max(DISPERSION_MIN_CURVATURE);
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
            let mu = em.exp().max(1e-300);
            // Precision channel models log(1/φ) ⇒ φ = exp(−η_d).
            let phi = (-ed).exp().max(1e-12);
            let two_minus_p = 2.0 - p;
            // Mean channel: the quasi-score `(y−μ)/μ` and Fisher weight
            // `μ^{2−p}/φ` are simple closed forms (and the mean block is
            // Fisher-orthogonal to the dispersion block in this
            // parameterization), so they stay hand-written exactly as the
            // NB/Gamma mean arms do.
            let mean_weight = wi * mu.powf(two_minus_p) / phi;
            let mean_response = em + (yi - mu) / mu;
            // Dispersion channel: the η_d-space score and OBSERVED information
            // come straight off the single-expression `Tower4<2>` seeded on
            // `(η_μ, η_d)` (#932), so the saddlepoint/point-mass branch split,
            // the `φ = exp(−η_d)` chain and its nonlinear `∂²φ/∂η_d²` curvature
            // correction are all mechanically carried — no per-branch
            // `s_phi`/`s_eta`/`curvature_eta` hand calculus.
            let tower = dispersion_tweedie_nll_tower(yi, em, ed, p, wi);
            let loglik = -tower.v;
            // η_d-space score and observed information off the tower, via the
            // same helper the NB/Gamma/Beta arms use (returns `(0, 0)` when the
            // prior weight is zero, so the row stays excluded below).
            let (s_eta, info_eta_raw) = tower_score_info(&tower, 1, wi);
            let curvature_eta = if wi == 0.0 {
                DISPERSION_MIN_CURVATURE
            } else {
                info_eta_raw.max(DISPERSION_MIN_CURVATURE)
            };
            // The working response divides by this per-row curvature so the
            // prior weight cancels (and a zero-prior-weight row stays excluded
            // via `disp_weight = 0`).
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
        let mut log_likelihood = 0.0;
        let mut mean_weights = Array1::<f64>::zeros(n);
        let mut mean_response = Array1::<f64>::zeros(n);
        let mut disp_weights = Array1::<f64>::zeros(n);
        let mut disp_response = Array1::<f64>::zeros(n);
        for i in 0..n {
            let row =
                dispersion_row_kernel(self.kind, self.y[i], eta_mu[i], eta_d[i], self.weights[i]);
            if row.loglik.is_finite() {
                log_likelihood += row.loglik;
            }
            mean_weights[i] = row.mean_weight.max(0.0);
            mean_response[i] = row.mean_response;
            disp_weights[i] = row.disp_weight.max(0.0);
            disp_response[i] = row.disp_response;
        }
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
        let mut ll = 0.0;
        for i in 0..self.y.len() {
            let row =
                dispersion_row_kernel(self.kind, self.y[i], eta_mu[i], eta_d[i], self.weights[i]);
            if row.loglik.is_finite() {
                ll += row.loglik;
            }
        }
        Ok(ll)
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        crate::families::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    /// Exact joint coefficient-space Hessian `H_L = -∇²log L` in flattened
    /// `[mean | log-precision]` block order.
    ///
    /// All four members assemble the same `Xᵀ diag(W) X` blocks; the cross
    /// block is the per-row mixed weight `dispersion_row_cross_weight`. Beta
    /// carries a genuinely nonzero (η_μ, η_φ) cross weight; the Fisher-
    /// orthogonal members (NegativeBinomial / Gamma / Tweedie) report a zero
    /// cross weight, so this returns their exact *block-diagonal* joint
    /// Hessian. Returning that block-diagonal `H_L` — rather than `None` —
    /// is what lets the multi-block outer-REML path (`build_joint_hessian_
    /// closures` → `joint_outer_evaluate`) and the joint posterior covariance
    /// (`compute_joint_covariance`) run for these families instead of failing
    /// the "multi-block families must provide a joint outer path" gate and
    /// silently escalating to a degraded ρ-seed fit with no covariance/EDF
    /// (gam#1119). The orthogonal members additionally declare
    /// `likelihood_blocks_uncoupled() = true` so the directional-derivative
    /// and Jeffreys dispatch route through the block-diagonal-exact fallback
    /// rather than rejecting the structurally-uncoupled Hessian.
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

        let eval = self.evaluate(block_states)?;
        let BlockWorkingSet::Diagonal {
            working_weights: mean_weights,
            ..
        } = &eval.blockworking_sets[Self::BLOCK_MEAN]
        else {
            return Err(format!(
                "{} dispersion mean block did not return diagonal weights",
                self.kind.family_tag()
            ));
        };
        let BlockWorkingSet::Diagonal {
            working_weights: disp_weights,
            ..
        } = &eval.blockworking_sets[Self::BLOCK_DISP]
        else {
            return Err(format!(
                "{} dispersion precision block did not return diagonal weights",
                self.kind.family_tag()
            ));
        };

        let cross_weights = Array1::from_shape_fn(n, |i| {
            dispersion_row_cross_weight(self.kind, self.y[i], eta_mu[i], eta_d[i], self.weights[i])
        });
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

        let h_mean = xt_diag_x_design(&mean_spec.design, mean_weights)?;
        let h_cross = xt_diag_y_design(&mean_spec.design, &cross_weights, &disp_spec.design)?;
        let h_disp = xt_diag_x_design(&disp_spec.design, disp_weights)?;
        let total = p_mean + p_disp;
        let mut h = Array2::<f64>::zeros((total, total));
        h.slice_mut(s![0..p_mean, 0..p_mean]).assign(&h_mean);
        h.slice_mut(s![0..p_mean, p_mean..total]).assign(&h_cross);
        h.slice_mut(s![p_mean..total, p_mean..total])
            .assign(&h_disp);
        mirror_upper_to_lower(&mut h);
        Ok(Some(h))
    }

    /// Whether the joint likelihood Hessian is block-diagonal in the
    /// `[mean | log-precision]` coefficient vector.
    ///
    /// `Beta(μφ, (1−μ)φ)` carries a genuinely nonzero `(η_μ, η_φ)` Fisher
    /// cross block (see the module header), so its blocks are coupled. The
    /// remaining members are Fisher-orthogonal in their mean/precision
    /// parameterizations — NB2 `(μ, θ)`, Gamma shape `ν = 1/φ`, Tweedie
    /// `log(1/φ)` — so `∂²L/∂β_μ∂β_d = 0` and the joint Hessian is exactly
    /// block-diagonal. Declaring that here lets the trait's directional-
    /// derivative / Jeffreys dispatch accept the block-diagonal joint Hessian
    /// via the working-set-exact fallback instead of rejecting it as an
    /// untrusted structurally-uncoupled override (which would strand the
    /// outer-REML gradient with a "dH unavailable" error, gam#1119).
    fn likelihood_blocks_uncoupled(&self) -> bool {
        !matches!(self.kind, DispersionFamilyKind::Beta)
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
    let em = eta_mu.clamp(-DISPERSION_ETA_CLAMP, DISPERSION_ETA_CLAMP);
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

        let mut meanspec = build_location_scale_block(
            "mu",
            mean_design.design.clone(),
            self.mean_offset.clone(),
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
            self.noise_offset.clone(),
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
        // The dispersion location-scale families have no closed-form analytic
        // spatial psi derivatives, and `fit_dispersion_glm_location_scale_terms`
        // disables the κ/ψ joint optimizer before the engine ever asks. If we
        // do get called (for example by a future caller that forgets the
        // disable), return a real diagnostic rather than a sentinel — emit the
        // exact data and design shape that was passed in so the bug is
        // diagnosable from the error string alone.
        Err(format!(
            "dispersion location-scale ({:?}) does not implement analytic spatial \
             psi derivatives; the κ/ψ joint optimizer must be disabled before \
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
    // The κ/ψ anisotropic-kernel joint optimizer needs analytic psi
    // derivatives this family does not provide; disable it so the engine runs
    // the full ρ REML directly via `fit_custom_family` (1-D and tensor smooth
    // penalties λ are still REML-selected).
    let mut kappa = kappa_options.clone();
    kappa.enabled = false;
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
        &kappa,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn beta_fisher_cross_info_mu_phi(mu: f64, phi: f64) -> f64 {
        let a = mu * phi;
        let b = (1.0 - mu) * phi;
        phi * (mu * crate::families::jet_tower::trigamma_derivative_stack(a)[0]
            - (1.0 - mu) * crate::families::jet_tower::trigamma_derivative_stack(b)[0])
    }

    pub(crate) fn assert_close(label: &str, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "{label}: got {got:.12e}, want {want:.12e}, |diff|={:.3e}",
            (got - want).abs()
        );
    }

    #[test]
    pub(crate) fn beta_tower_mixed_channel_matches_cross_information_formula() {
        let mu = 0.1;
        let phi = 10.0;
        let a = mu * phi;
        let b = (1.0 - mu) * phi;
        let digamma_a = crate::families::jet_tower::digamma_derivative_stack(a)[0];
        let digamma_b = crate::families::jet_tower::digamma_derivative_stack(b)[0];
        let score_neutral_y = 1.0 / (1.0 + (-(digamma_a - digamma_b)).exp());

        let tower = dispersion_beta_nll_tower(score_neutral_y, mu, phi, 1.0);
        let trigamma_a = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        let trigamma_b = crate::families::jet_tower::trigamma_derivative_stack(b)[0];
        let analytic = phi * (mu * trigamma_a - (1.0 - mu) * trigamma_b);
        let helper = beta_fisher_cross_info_mu_phi(mu, phi);

        assert!(
            analytic > 0.58,
            "audit example should have visibly nonzero cross information, got {analytic}"
        );
        assert_close("helper cross information", helper, analytic, 1e-12);
        assert_close("tower mixed channel", tower.h[0][1], analytic, 1e-8);

        let q = mu * (1.0 - mu);
        let eta_cross = beta_observed_cross_weight_eta(score_neutral_y, mu, phi, 1.0);
        assert_close(
            "eta-scale cross weight",
            eta_cross,
            q * phi * analytic,
            1e-8,
        );
    }

    #[test]
    pub(crate) fn orthogonal_dispersion_families_report_zero_cross_weight() {
        let cases = [
            DispersionFamilyKind::NegativeBinomial,
            DispersionFamilyKind::Gamma,
            DispersionFamilyKind::Tweedie { p: 1.5 },
        ];
        for kind in cases {
            let got = dispersion_row_cross_weight(kind, 1.25, 0.2, -0.3, 2.0);
            assert_close(kind.family_tag(), got, 0.0, 1e-12);
        }
    }
}
