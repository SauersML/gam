//! NUTS Sampler using general-mcmc
//!
//! This module provides NUTS (No-U-Turn Sampler) for honest uncertainty
//! quantification after PIRLS convergence.
//!
//! # Design
//!
//! Since general-mcmc's NUTS uses an identity mass matrix, we whiten the
//! parameter space using the Cholesky decomposition of the inverse Hessian:
//!
//! - Transform: β = μ + L @ z  (where L L^T = H^{-1})
//! - The whitened space has unit covariance, so NUTS mixes efficiently
//! - Samples are un-transformed back to the original space
//!
//! # Analytical Gradients
//!
//! We override `unnorm_logp_and_grad` to compute gradients analytically using
//! ndarray, avoiding burn's autodiff overhead. The gradient computation mirrors
//! the true log-posterior gradient (not the PIRLS working gradient).
//!
//! # Memory Efficiency
//!
//! Large data (design matrix, response, etc.) is wrapped in `Arc` to allow
//! sharing across chains without duplication when general-mcmc clones the target.

use crate::gpu_polya_gamma::{PgSeed, PolyaGammaBatchInput};
use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_ata_into, fast_atv, fast_av_into};
use gam_linalg::matrix::DesignMatrix;
use gam_linalg::triangular::back_substitution_lower_transpose_guarded_into;
use gam_models::wiggle::monotone_wiggle_basis_with_derivative_order;
use gam_problem::types::{
    InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink, is_valid_tweedie_power,
};
use gam_solve::estimate::reml::FirthDenseOperator;
use gam_solve::estimate::reml::penalty_logdet::PenaltyPseudologdet;
use gam_solve::estimate::{
    EstimationError, UnifiedFitResult, validate_explicit_dense_hessian_for_whitening,
};
use gam_solve::mixture_link::{
    InverseLinkKernel, LinkParamPartials, inverse_link_jet_for_inverse_link, softmax_last_fixedzero,
};
use gam_terms::construction::CanonicalPenalty;
use general_mcmc::generic_hmc::HamiltonianTarget;
pub use general_mcmc::generic_nuts::NUTSMassMatrixConfig;
use general_mcmc::generic_nuts::{GenericNUTS, MassMatrixAdaptation};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Binomial families whose inverse link has a Fisher-weight jet
/// (`fisher_weight_jet5`) support the Jeffreys/Firth term. This is the
/// link-general set shared with the REML/PIRLS Firth operator; the canonical
/// logit case is unchanged.
#[inline]
fn likelihood_spec_supports_firth(spec: &LikelihoodSpec) -> bool {
    spec.supports_firth()
}

/// Inverse link to evaluate the Fisher working weight with for the Jeffreys
/// term. Returns `None` for unsupported specs.
#[inline]
fn likelihood_spec_jeffreys_link(spec: &LikelihoodSpec) -> Option<InverseLink> {
    if likelihood_spec_supports_firth(spec) {
        Some(spec.link.clone())
    } else {
        None
    }
}

/// Typed error variants for the HMC / NUTS sampling module.
///
/// External-facing helpers in this module continue to return
/// `Result<_, String>`; this enum is materialized internally and converted
/// at the public boundary via `.map_err(String::from)` so that the error
/// text remains byte-identical to the previous `format!` output.
#[derive(Debug, Clone)]
pub enum HmcError {
    /// Sampler state (penalty / Hessian / mode / posterior values) contains
    /// NaN or Inf where finiteness is required.
    NonFiniteState { reason: String },
    /// Configuration value (e.g. `target_accept`, unit-weight requirement)
    /// is out of range or otherwise invalid.
    InvalidConfig { reason: String },
    /// Dimensions of the supplied matrices / vectors are inconsistent.
    DimensionMismatch { reason: String },
    /// Firth/Jeffreys correction was requested for a family that does not
    /// support it.
    FirthUnsupported { reason: String },
    /// Inverse-link state does not match the requested likelihood family in
    /// the joint (β, ρ) sampler.
    LinkMismatch { reason: String },
    /// Likelihood family is not implemented in the current sampling path.
    UnsupportedFamily { reason: String },
    /// Sampling produced no usable output (empty kept set, non-finite
    /// summary statistic, etc.).
    SamplingFailed { reason: String },
}

impl fmt::Display for HmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HmcError::NonFiniteState { reason }
            | HmcError::InvalidConfig { reason }
            | HmcError::DimensionMismatch { reason }
            | HmcError::FirthUnsupported { reason }
            | HmcError::LinkMismatch { reason }
            | HmcError::UnsupportedFamily { reason }
            | HmcError::SamplingFailed { reason } => f.write_str(reason),
        }
    }
}

impl From<HmcError> for String {
    fn from(err: HmcError) -> String {
        err.to_string()
    }
}

/// Upper bound on the autocorrelation lag summed in the effective-sample-size
/// estimate. The Geyer initial-positive-sequence sum normally self-truncates
/// long before this, but a hard cap bounds the `O(n·lag)` work for very long
/// chains where the autocorrelation tail is numerical noise.
const MAX_AUTOCORRELATION_LAG: usize = 1000;

/// Floor on the lag-0 autocovariance (chain variance) used as the denominator in
/// the autocorrelation ratios, guarding against division by zero for a chain
/// that is numerically constant.
const AUTOCOVARIANCE_FLOOR: f64 = 1e-16;

/// Compute split-chain R-hat and ESS using the Gelman-Rubin diagnostic.
///
/// This is the standard split-chain formulation (no rank normalization).
/// Returns (max_rhat, min_ess) across dimensions.
pub(crate) fn compute_split_rhat_and_ess(samples: &Array3<f64>) -> (f64, f64) {
    let n_chains = samples.shape()[0];
    let n_samples = samples.shape()[1];
    let dim = samples.shape()[2];

    if n_chains < 2 || n_samples < 4 {
        return (1.0, n_chains as f64 * n_samples as f64 * 0.5);
    }

    // Split each chain in half to detect non-stationarity
    let half = n_samples / 2;
    let n_split_chains = n_chains * 2;
    let n_split_samples = half;

    let mut max_rhat = 0.0f64;
    let mut min_ess = f64::INFINITY;

    #[inline]
    fn splitvalue(
        samples: &Array3<f64>,
        n_chains: usize,
        half: usize,
        dim: usize,
        sc: usize,
        t: usize,
    ) -> f64 {
        let chain = sc % n_chains;
        if sc < n_chains {
            samples[[chain, t, dim]]
        } else {
            samples[[chain, half + t, dim]]
        }
    }

    fn ess_from_split_dimension(
        samples: &Array3<f64>,
        n_chains: usize,
        half: usize,
        dim: usize,
    ) -> f64 {
        let m = n_chains * 2;
        let n = half;
        if m == 0 || n < 4 {
            return (m * n).max(1) as f64;
        }

        let mut means = vec![0.0_f64; m];
        let mut gamma0 = vec![0.0_f64; m];
        for sc in 0..m {
            let mut sum = 0.0;
            for t in 0..n {
                sum += splitvalue(samples, n_chains, half, dim, sc, t);
            }
            let mean = sum / n as f64;
            means[sc] = mean;
            let mut g0 = 0.0;
            for t in 0..n {
                let d = splitvalue(samples, n_chains, half, dim, sc, t) - mean;
                g0 += d * d;
            }
            gamma0[sc] = (g0 / n as f64).max(AUTOCOVARIANCE_FLOOR);
        }

        let max_lag = (n - 1).min(MAX_AUTOCORRELATION_LAG);
        let mut tau = 1.0_f64;
        let mut lag = 1usize;
        while lag < max_lag {
            let mut pair = 0.0_f64;
            for l in [lag, lag + 1] {
                if l > max_lag {
                    continue;
                }
                let mut rho_l = 0.0;
                for sc in 0..m {
                    let mu = means[sc];
                    let mut cov = 0.0;
                    let denom = (n - l) as f64;
                    for t in 0..(n - l) {
                        let x0 = splitvalue(samples, n_chains, half, dim, sc, t);
                        let x1 = splitvalue(samples, n_chains, half, dim, sc, t + l);
                        cov += (x0 - mu) * (x1 - mu);
                    }
                    cov /= denom;
                    rho_l += cov / gamma0[sc];
                }
                rho_l /= m as f64;
                pair += rho_l;
            }
            if !pair.is_finite() || pair <= 0.0 {
                break;
            }
            tau += 2.0 * pair;
            lag += 2;
        }
        if !tau.is_finite() || tau <= 0.0 {
            return 1.0;
        }
        let total = (m * n) as f64;
        (total / tau).clamp(1.0, total)
    }

    let mut chain_means = vec![0.0_f64; n_split_chains];
    let mut chainvars = vec![0.0_f64; n_split_chains];
    for d in 0..dim {
        for chain in 0..n_chains {
            // First half
            let mut sum1 = 0.0;
            for i in 0..half {
                sum1 += samples[[chain, i, d]];
            }
            let mean1 = sum1 / half as f64;
            let mut var1 = 0.0;
            for i in 0..half {
                let diff = samples[[chain, i, d]] - mean1;
                var1 += diff * diff;
            }
            var1 /= (half - 1).max(1) as f64;
            let first_idx = chain;
            chain_means[first_idx] = mean1;
            chainvars[first_idx] = var1;

            // Second half
            let mut sum2 = 0.0;
            for i in half..(2 * half) {
                sum2 += samples[[chain, i, d]];
            }
            let mean2 = sum2 / half as f64;
            let mut var2 = 0.0;
            for i in half..(2 * half) {
                let diff = samples[[chain, i, d]] - mean2;
                var2 += diff * diff;
            }
            var2 /= (half - 1).max(1) as f64;
            let second_idx = n_chains + chain;
            chain_means[second_idx] = mean2;
            chainvars[second_idx] = var2;
        }

        // Within-chain variance W
        let w: f64 = chainvars.iter().copied().sum::<f64>() / n_split_chains as f64;

        // Between-chain variance B
        let overall_mean: f64 = chain_means.iter().copied().sum::<f64>() / n_split_chains as f64;
        let b: f64 = chain_means
            .iter()
            .map(|m| (m - overall_mean).powi(2))
            .sum::<f64>()
            * n_split_samples as f64
            / (n_split_chains - 1) as f64;

        // Estimated variance
        let var_hat = (n_split_samples as f64 - 1.0) / n_split_samples as f64 * w
            + b / n_split_samples as f64;

        // R-hat
        let rhat_d = if w > 1e-10 { (var_hat / w).sqrt() } else { 1.0 };
        max_rhat = max_rhat.max(rhat_d);

        // Real ESS via split-chain autocorrelation with Geyer IPS truncation.
        let ess_d = ess_from_split_dimension(samples, n_chains, half, d);
        min_ess = min_ess.min(ess_d);
    }

    (max_rhat, min_ess.max(1.0))
}

/// Solve L^T * X = I where L is lower triangular.
///
/// Returns X = L^{-T} (the inverse transpose of L).
///
/// This is the correct way to compute the whitening transform matrix:
/// Given H = L L^T (Cholesky), we need W where W W^T = H^{-1}
/// Since H^{-1} = L^{-T} L^{-1}, we have W = L^{-T}.
///
/// Implementation strategy (math-equivalent to back-substitution on L^T):
/// We compute L^{-1} column-wise via forward substitution on L, then the
/// result is `L^{-1}` transposed. Forward-substituting column `c` of L^{-1}
/// uses `L`'s rows (which are contiguous in row-major `Array2`), giving
/// stride-1 inner loops instead of the strided `l[[j, i]]` (column-major
/// access pattern) and double-indexed writes of the original. We also
/// exploit the triangular structure of `L^{-1}` (entries above the diagonal
/// are zero), skipping ~half of the inner work compared to the previous
/// version which traversed `i = (0..dim).rev()` for every column.
///
/// Total cost: ~dim^3 / 6 multiply-adds (down from dim^3 / 2), with all
/// inner loops on contiguous slices.
fn solve_upper_triangular_transpose(l: &Array2<f64>, dim: usize) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((dim, dim));
    if dim == 0 {
        return result;
    }

    // Pull contiguous row slice access from L (row-major standard layout).
    // Falls back to a one-time owned copy if `l` is not standard-layout
    // (e.g. a transposed view); both branches feed the same inner loop.
    let l_owned;
    let l_rows: &[f64] = if let Some(s) = l.as_slice() {
        s
    } else {
        l_owned = l.to_owned();
        l_owned
            .as_slice()
            .expect("owned standard-layout Array2 has contiguous storage")
    };

    // Scratch column for L^{-1}[:, col]; reused across columns.
    let mut y = vec![0.0_f64; dim];

    for col in 0..dim {
        // Forward-substitute L * y = e_col. y[i] = 0 for i < col.
        // Diagonal term:
        let d_col = l_rows[col * dim + col];
        let inv_d_col = if d_col.abs() > 1e-15 {
            1.0 / d_col
        } else {
            0.0
        };
        y[col] = inv_d_col;

        // Below-diagonal entries: y[i] = -(sum_{j=col..i} L[i,j] * y[j]) / L[i,i].
        // Each inner loop is a stride-1 dot product on row `i` of L (contiguous).
        for i in (col + 1)..dim {
            let row_off = i * dim;
            let l_row = &l_rows[row_off + col..row_off + i];
            let y_seg = &y[col..i];
            // Both operands are contiguous slices of equal length; the loop
            // is a straight-line stride-1 reduction the optimizer can
            // auto-vectorize.
            let mut sum = 0.0_f64;
            for k in 0..l_row.len() {
                sum += l_row[k] * y_seg[k];
            }
            let d = l_rows[row_off + i];
            y[i] = if d.abs() > 1e-15 { -sum / d } else { 0.0 };
        }

        // Write the column into result transposed: result[col, i] = y[i] for i >= col.
        // result[i, col] is left at zero for i < col (upper-triangular L^{-T}).
        // That matches `result[col, i]` filling row `col` from column `col` rightward.
        let res_row_start = col * dim + col;
        let res_row = &mut result.as_slice_mut().expect("owned Array2 contiguous")
            [res_row_start..res_row_start + (dim - col)];
        for (k, slot) in res_row.iter_mut().enumerate() {
            *slot = y[col + k];
        }

        // Clear scratch positions we wrote, so the next column starts clean above.
        for slot in &mut y[col..dim] {
            *slot = 0.0;
        }
    }

    result
}

struct WhiteningTransform {
    chol: Array2<f64>,
    chol_t: Array2<f64>,
}

fn hessian_whitening_transform(
    hessian: ArrayView2<f64>,
    dim: usize,
    cov_scale: f64,
    cholesky_error_prefix: &str,
) -> Result<WhiteningTransform, String> {
    let hessian_owned = hessian.to_owned();
    let chol_factor = hessian_owned
        .cholesky(Side::Lower)
        .map_err(|e| format!("{cholesky_error_prefix}: {:?}", e))?;
    let l_h = chol_factor.lower_triangular();
    let mut chol = solve_upper_triangular_transpose(&l_h, dim);
    let sqrt_cov_scale = cov_scale.max(0.0).sqrt();
    if (sqrt_cov_scale - 1.0).abs() > 0.0 {
        chol.mapv_inplace(|v| v * sqrt_cov_scale);
    }
    let chol_t = chol.t().to_owned();
    Ok(WhiteningTransform { chol, chol_t })
}

/// Shared data for NUTS posterior (wrapped in Arc to prevent cloning).
///
/// This struct holds read-only data that is shared across all chains.
/// Using Arc prevents memory explosion when general-mcmc clones the target.
#[derive(Clone)]
struct SharedData {
    /// Design matrix X [n_samples, dim]
    x: Arc<Array2<f64>>,
    /// Response vector y [n_samples]
    y: Arc<Array1<f64>>,
    /// Observation/case weights [n_samples]
    weights: Arc<Array1<f64>>,
    /// MAP estimate (mode) μ [dim]
    mode: Arc<Array1<f64>>,
    /// Fixed additive offset on the linear predictor: η = Xβ + offset
    /// [n_samples]. `None` when the model was fit without an offset (the common
    /// case), avoiding a per-step O(n) add of zeros. The offset shifts η only —
    /// it is constant in β, so ∂η/∂β = X is unchanged and no gradient,
    /// Hessian, or penalty term is affected. Dropping it (the historical
    /// behaviour) silently sampled the wrong posterior for any `--offset-column`
    /// fit (#882).
    offset: Option<Arc<Array1<f64>>>,
    /// Auxiliary log-link family parameter: Gamma shape, Tweedie power, or NB theta.
    gamma_shape: f64,
    /// Dispersion parameter φ (Gaussian: σ²; Gamma: 1/shape; `Known(1.0)` for
    /// fixed-scale families). Consumed **only** by the likelihood adapters that
    /// carry the dispersion in the data term itself: the profiled-Gaussian
    /// log-likelihood and its gradient multiply through by `1/φ`, and the
    /// Tweedie quasi-likelihood folds `1/φ` into its weight. It does NOT drive
    /// the whitening or penalty scaling — those use the `cov_scale` invariant
    /// (`NutsFamily::coefficient_covariance_scale`), which is `1.0` for Gamma
    /// even though `φ ≠ 1`, because Gamma's dispersion already lives inside the
    /// working weight (the `shape` factor in `gamma_log_logp_and_grad`). See
    /// `inference::dispersion_cov` for the ownership invariants.
    dispersion: gam_solve::model_types::Dispersion,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
}

thread_local! {
    static NUTS_RESIDUAL_SCRATCH: RefCell<Array1<f64>> = RefCell::new(Array1::zeros(0));
}

/// Whitened log-posterior target with analytical gradients.
///
/// Uses Arc for shared data to prevent memory explosion when cloned for chains.
/// Uses faer for numerically stable Cholesky decomposition.
/// Family mode for NUTS log-likelihood computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NutsFamily {
    Gaussian,
    BinomialLogit,
    BinomialProbit,
    BinomialCLogLog,
    PoissonLog,
    TweedieLog,
    NegativeBinomialLog,
    GammaLog,
}

impl NutsFamily {
    #[inline]
    fn likelihood_spec(self) -> LikelihoodSpec {
        match self {
            Self::Gaussian => LikelihoodSpec {
                response: ResponseFamily::Gaussian,
                link: InverseLink::Standard(StandardLink::Identity),
            },
            Self::BinomialLogit => LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Logit),
            },
            Self::BinomialProbit => LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Probit),
            },
            Self::BinomialCLogLog => LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::CLogLog),
            },
            Self::PoissonLog => LikelihoodSpec {
                response: ResponseFamily::Poisson,
                link: InverseLink::Standard(StandardLink::Log),
            },
            Self::TweedieLog => LikelihoodSpec {
                response: ResponseFamily::Tweedie { p: 1.5 },
                link: InverseLink::Standard(StandardLink::Log),
            },
            Self::NegativeBinomialLog => LikelihoodSpec {
                response: ResponseFamily::NegativeBinomial {
                    theta: 1.0,
                    theta_fixed: false,
                },
                link: InverseLink::Standard(StandardLink::Log),
            },
            Self::GammaLog => LikelihoodSpec {
                response: ResponseFamily::Gamma,
                link: InverseLink::Standard(StandardLink::Log),
            },
        }
    }

    /// Coefficient-covariance scale for the whitened NUTS target — the
    /// NUTS-family counterpart of
    /// [`gam_problem::types::GlmLikelihoodSpec::coefficient_covariance_scale`] (#679).
    ///
    /// The sampler must reproduce the posterior `N(mode, Vb)` with
    /// `Vb = scale · H⁻¹`, where `H = XᵀWX + S_λ` is the stored penalized
    /// Hessian (penalty `S_λ` added **unscaled**). The returned `scale` is:
    ///
    /// * `profiled_gaussian_phi` (= σ̂²) for the **profiled Gaussian** identity
    ///   model, whose working weight is scale-free (`W = priorweights`), so the
    ///   stored `H` omits the dispersion and `Vb = σ̂²·H⁻¹`. The NUTS Gaussian
    ///   log-likelihood is structurally this profiled form: scale-free
    ///   residuals multiplied by `1/φ` (see `gaussian_logp_and_grad_into`).
    /// * `1.0` for every **weight-carries-dispersion** family (Gamma, Tweedie,
    ///   Negative-Binomial, and the fixed-scale Poisson/Binomial). Their
    ///   working weight already folds in the reciprocal dispersion / full
    ///   Fisher information — for Gamma-log the shape `ν = 1/φ` is baked into
    ///   the likelihood score `∂ℓ/∂η = ν·(y/μ − 1)` — so the stored `H` is
    ///   already the true penalized Hessian and `Vb = H⁻¹`. Multiplying by the
    ///   dispersion again double-counts it and shrinks every posterior SD by
    ///   `√dispersion` — exactly the Gamma-log defect addressed in #680.
    ///
    /// This single scalar governs BOTH the whitening preconditioner
    /// (`L Lᵀ = scale·H⁻¹`, so `L` is scaled by `√scale`) and the target's
    /// penalty weight (`penalty_scale = 1/scale`), keeping the sampled
    /// posterior, its whitening metric, and the Wald `Vb` of #679 mutually
    /// consistent. Crucially it does NOT key off the statistical dispersion
    /// `φ`: Gamma carries `φ = 1/shape ≠ 1` yet still has `scale = 1`, because
    /// that `φ` already lives inside `W`.
    #[inline]
    fn coefficient_covariance_scale(self, profiled_gaussian_phi: f64) -> f64 {
        match self {
            NutsFamily::Gaussian => profiled_gaussian_phi,
            _ => 1.0,
        }
    }
}

/// Whitened-coordinate target for the No-U-Turn HMC sampler.
///
/// The posterior over β is reparameterized via `β = L z` where `L Lᵀ = H⁻¹`
/// (Cholesky factor of the inverse posterior Hessian at the MAP), so that
/// in `z`-coordinates the local curvature is approximately the identity.
/// The struct holds the shared design, the whitening factor `L` and its
/// transpose (for gradient chain-rule pull-back `∇_z = Lᵀ ∇_β`), the
/// family-specific log-likelihood adapter, and a precomputed
/// `M = Lᵀ S L` so the smoothing penalty `−½ βᵀSβ` becomes the cheap
/// quadratic `−½ zᵀMz` inside the leapfrog hot loop.  Optionally adds
/// the identifiable-subspace Firth/Jeffreys term to keep posterior modes
/// away from infinity under separation.
pub struct NutsPosterior {
    /// Shared read-only data (Arc prevents duplication)
    data: SharedData,
    /// Transform: L where L L^T = H^{-1} (computed from Hessian)
    /// This is the inverse-transpose of the Cholesky of H.
    chol: Array2<f64>,
    /// L^T for gradient chain rule: ∇z = L^T @ ∇_β
    chol_t: Array2<f64>,
    /// Family for log-likelihood computation
    nuts_family: NutsFamily,
    /// Whether to add the identifiable-subspace Jeffreys/Firth term to the
    /// target
    firth_enabled: bool,
    /// Precomputed whitened-penalty operator `M = L^T S L` (dim×dim, symmetric
    /// positive-semidefinite). The penalty term in z-coordinates is
    ///   −0.5 βᵀSβ = −[c0 + (Lᵀ S μ)ᵀ z + 0.5 zᵀ M z],
    /// so its z-gradient is just `−(L^T S μ + M z)` — no per-step `S·β` matvec
    /// or `L^T·∇_β penalty` map is needed.
    penalty_z_quad: Array2<f64>,
    /// Precomputed `Lᵀ S μ` (length dim) — z-space gradient contribution from
    /// the linear-in-z portion of the penalty.
    penalty_z_lin: Array1<f64>,
    /// Precomputed `0.5 μᵀ S μ` (scalar) — constant term of the penalty.
    penalty_z_const: f64,
    /// Coefficient-covariance scale `cov_scale` (#679/#680 invariant): the
    /// `Vb = cov_scale·H⁻¹` multiplier. `σ̂²` for profiled Gaussian, `1.0` for
    /// every weight-carries-dispersion family. Drives both the whitening
    /// (`L Lᵀ = cov_scale·H⁻¹`) and the target penalty weight
    /// (`penalty_scale = 1/cov_scale`).
    cov_scale: f64,
}

impl NutsPosterior {
    /// Creates a new posterior target from ndarray data.
    ///
    /// # Arguments
    /// * `x` - Design matrix [n_samples, dim]
    /// * `y` - Response vector [n_samples]
    /// * `weights` - Observation/case weights [n_samples]
    /// * `penalty_matrix` - Combined penalty S [dim, dim]
    /// * `mode` - MAP estimate μ [dim]
    /// * `hessian` - Hessian H [dim, dim] (NOT the inverse!)
    /// * `nuts_family` - Family for log-likelihood computation
    ///
    /// # Numerical Stability
    /// Accepts the Hessian directly and computes L = (chol(H))^{-T} via
    /// triangular solves, which is more stable than explicitly inverting H.
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_matrix: ArrayView2<f64>,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        nuts_family: NutsFamily,
        gamma_shape: f64,
        dispersion: gam_solve::model_types::Dispersion,
        firth_enabled: bool,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let dim = x.ncols();

        // Validate inputs are finite
        if !penalty_matrix.iter().all(|x| x.is_finite()) {
            return Err(HmcError::NonFiniteState {
                reason: "Penalty matrix contains NaN or Inf values".to_string(),
            }
            .into());
        }
        if !hessian.iter().all(|x| x.is_finite()) {
            return Err(HmcError::NonFiniteState {
                reason: "Hessian matrix contains NaN or Inf values".to_string(),
            }
            .into());
        }
        if !mode.iter().all(|x| x.is_finite()) {
            return Err(HmcError::NonFiniteState {
                reason: "Mode vector contains NaN or Inf values".to_string(),
            }
            .into());
        }

        validate_firth_support(nuts_family, firth_enabled).map_err(String::from)?;
        if nuts_family.likelihood_spec().is_binomial() {
            validate_binary_responses("binomial NUTS", &y, &weights).map_err(String::from)?;
        }
        if matches!(nuts_family, NutsFamily::NegativeBinomialLog) {
            validate_count_responses("negative-binomial NUTS", &y, &weights)
                .map_err(String::from)?;
        }

        // Whitening metric: `L Lᵀ` must equal the posterior covariance the
        // sampler reproduces, `Vb = cov_scale · H⁻¹` (#679/#680 invariant), so
        // scale `L` by `√cov_scale`. Only the profiled-Gaussian model carries a
        // non-unit scale (σ̂² = `dispersion.phi()`); every weight-carries-
        // dispersion family (Gamma/Tweedie/NB) already folds its dispersion into
        // the stored `H`, so `cov_scale == 1` and this is a no-op. This replaces
        // a previous `sqrt_phi()` multiply that wrongly scaled Gamma (and any
        // φ-bearing family) by `√φ`, mis-preconditioning against `φ·H⁻¹`.
        let cov_scale = nuts_family.coefficient_covariance_scale(dispersion.phi());
        let whitening = hessian_whitening_transform(
            hessian,
            dim,
            cov_scale,
            "Hessian Cholesky decomposition failed",
        )?;
        let chol = whitening.chol;
        let chol_t = whitening.chol_t;

        // Precompute the whitened penalty operator and constants so that the
        // penalty contribution to logp/grad becomes a single symv against z.
        // Math identity (β = μ + L z, L L^T = H^{-1}):
        //   0.5 β^T S β = 0.5 μ^T S μ + (L^T S μ)^T z + 0.5 z^T (L^T S L) z
        // and ∇_z [0.5 β^T S β] = L^T S μ + (L^T S L) z.
        // This replaces three matvecs per leapfrog step (S·β, L·z used only
        // for that purpose, and L^T·∇_β penalty) with one dim×dim symv.
        let penalty_owned = penalty_matrix.to_owned();
        let mode_owned = mode.to_owned();
        let s_mu = penalty_owned.dot(&mode_owned);
        let penalty_z_const = 0.5 * mode_owned.dot(&s_mu);
        let penalty_z_lin = chol_t.dot(&s_mu);
        // M = L^T S L = chol_t · (S · chol). Computed in two GEMMs at
        // construction time only.
        let s_chol = penalty_owned.dot(&chol);
        let penalty_z_quad = chol_t.dot(&s_chol);

        let data = SharedData {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            mode: Arc::new(mode_owned),
            offset: None,
            gamma_shape,
            dispersion,
            n_samples,
            dim,
        };

        Ok(Self {
            data,
            chol,
            chol_t,
            nuts_family,
            firth_enabled,
            penalty_z_quad,
            penalty_z_lin,
            penalty_z_const,
            cov_scale,
        })
    }

    /// Attach a fixed additive offset to the linear predictor: η = Xβ + offset.
    ///
    /// The offset is constant in β, so the whitening geometry (`chol`), penalty
    /// operators, and stored Hessian are all unchanged — only η (and hence the
    /// per-observation working residual / mean) shifts. The fitted `mode` and
    /// `hessian` handed to [`Self::new`] already correspond to the offset-trained
    /// fit, so this only needs to restore the offset to the likelihood
    /// evaluation. Returns an error if the offset length disagrees with the data
    /// or carries non-finite entries.
    fn with_offset(mut self, offset: ArrayView1<f64>) -> Result<Self, String> {
        if offset.len() != self.data.n_samples {
            return Err(HmcError::DimensionMismatch {
                reason: format!(
                    "NUTS offset length {} does not match {} observations",
                    offset.len(),
                    self.data.n_samples
                ),
            }
            .into());
        }
        if !offset.iter().all(|v| v.is_finite()) {
            return Err(HmcError::NonFiniteState {
                reason: "NUTS offset contains NaN or Inf values".to_string(),
            }
            .into());
        }
        self.data.offset = Some(Arc::new(offset.to_owned()));
        Ok(self)
    }

    fn compute_logp_and_grad_nd_into(
        &self,
        z: &Array1<f64>,
        residual: &mut Array1<f64>,
        grad: &mut Array1<f64>,
    ) -> f64 {
        // === Step 1: Transform z (whitened) -> β (original) ===
        // β = μ + L @ z
        let beta = self.data.mode.as_ref() + &self.chol.dot(z);

        // === Step 2: Compute η = X @ β (+ offset) ===
        let mut eta = gam_linalg::faer_ndarray::fast_av(self.data.x.as_ref(), &beta);
        if let Some(offset) = self.data.offset.as_ref() {
            eta += offset.as_ref();
        }

        // === Step 3: Compute log-likelihood and gradient ===
        let (ll, mut grad_ll_beta) = self.family_logp_and_grad_into(&eta, residual);

        let mut firth_logdet = 0.0;
        if self.firth_enabled {
            match firth_jeffreys_logp_and_grad(&self.nuts_family.likelihood_spec(), &self.data, &eta)
            {
                Ok((value, grad_beta_firth)) => {
                    firth_logdet = value;
                    grad_ll_beta += &grad_beta_firth;
                }
                Err(err) => {
                    log::warn!(
                        "[NUTS/Firth] Jeffreys target became invalid at the current state: {}",
                        err
                    );
                    grad.fill(0.0);
                    return f64::NEG_INFINITY;
                }
            }
        }

        // === Step 4: Penalty in z-coordinates (precomputed; see `new`) ===
        //   −0.5 βᵀ S β  =  −[c0 + lᵀ z + 0.5 zᵀ M z]
        //   ∇_z (−0.5 βᵀ S β) = −(l + M z)
        // where l = L^T S μ, M = L^T S L, c0 = 0.5 μᵀ S μ.
        // This single dim×dim symmetric matvec replaces both the per-step
        // S·β multiply and the L^T·∇_β penalty chain-rule multiply, and lets
        // the penalty value, β-gradient and chain rule fuse into one pass.
        //
        // Penalty weight in the un-whitened β-target
        // `log p(β) = loglik(β) − penalty_scale · ½ βᵀSβ`. The invariant is
        // `Vb = cov_scale · H⁻¹` with `H = XᵀWX + S` (penalty added unscaled),
        // so the target curvature must equal `Vb⁻¹ = H/cov_scale`. The
        // likelihood already supplies `−∇²ℓ = (data Fisher info)/cov_scale`
        // (explicitly `/σ²` for profiled Gaussian, implicitly via the working
        // weight / the `shape ≡ 1/φ` baked into `gamma_log_logp_and_grad` for
        // the dispersion-carrying families), so the penalty must match it:
        //   penalty_scale = 1/cov_scale.
        // That is `1/σ²` for profiled Gaussian and exactly `1.0` for
        // Gamma/Tweedie/NB/Poisson/Binomial. The previous code used
        // `dispersion.inv_phi()` for GammaLog (= shape = 1/φ ≠ 1), which
        // double-counted the dispersion in the sampled posterior (#680); the
        // statistical dispersion `φ` is NOT `1/cov_scale` for Gamma because it
        // already lives inside `W`. Mirrors `LinkWigglePosterior`.
        let penalty_scale = 1.0 / self.cov_scale.max(1e-300);
        let mz = self.penalty_z_quad.dot(z);
        let lin_term = self.penalty_z_lin.dot(z);
        let quad_term = 0.5 * z.dot(&mz);
        let penalty = penalty_scale * (self.penalty_z_const + lin_term + quad_term);

        // === Step 5: z-space gradient ===
        // ∇z log p = L^T ∇_β ℓ  −  penalty_scale · (l + M z)
        fast_av_into(&self.chol_t, &grad_ll_beta, grad);
        // gradz -= penalty_scale · (penalty_z_lin + M z); fused parallel update.
        let lin_view = self.penalty_z_lin.view();
        ndarray::Zip::from(grad)
            .and(&lin_view)
            .and(&mz)
            .par_for_each(|g, &l, &m| {
                *g -= penalty_scale * (l + m);
            });

        ll + firth_logdet - penalty
    }

    fn family_logp_and_grad_into(
        &self,
        eta: &Array1<f64>,
        residual: &mut Array1<f64>,
    ) -> (f64, Array1<f64>) {
        nuts_family_logp_and_grad_into(self.nuts_family, &self.data, eta, residual)
    }

    /// Get the Cholesky factor L for un-whitening samples
    pub fn chol(&self) -> &Array2<f64> {
        &self.chol
    }

    /// Get the mode
    pub fn mode(&self) -> &Array1<f64> {
        &self.data.mode
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.data.dim
    }
}

const HALF_LOG_2PI: f64 = 0.918_938_533_204_672_7;

#[inline]
fn standard_normal_log_pdf(x: f64) -> f64 {
    -0.5 * x * x - HALF_LOG_2PI
}

/// Stable log Φ(x) for the standard normal CDF.
#[inline]
fn log_ndtr(x: f64) -> f64 {
    let arg = -x * std::f64::consts::FRAC_1_SQRT_2;
    let erfc_val = statrs::function::erf::erfc(arg);
    if erfc_val > 0.0 {
        erfc_val.ln() - std::f64::consts::LN_2
    } else {
        -0.5 * x * x - (-x).ln() - HALF_LOG_2PI
    }
}

#[inline]
fn validate_firth_support(family: NutsFamily, firth_enabled: bool) -> Result<(), HmcError> {
    let spec = family.likelihood_spec();
    if firth_enabled && !likelihood_spec_supports_firth(&spec) {
        return Err(HmcError::FirthUnsupported {
            reason: format!(
                "NUTS with Firth requires a Binomial inverse link with a Fisher-weight jet; {} does not support it",
                spec.pretty_name()
            ),
        });
    }
    Ok::<(), _>(())
}

#[inline]
fn validate_firth_likelihood_support(
    likelihood: &LikelihoodSpec,
    firth_enabled: bool,
) -> Result<(), HmcError> {
    if firth_enabled && !likelihood_spec_supports_firth(likelihood) {
        return Err(HmcError::FirthUnsupported {
            reason: format!(
                "Joint HMC with Firth requires a Binomial inverse link with a Fisher-weight jet; {} does not support it",
                likelihood.pretty_name()
            ),
        });
    }
    Ok::<(), _>(())
}

#[inline]
fn valid_count_response(y: f64) -> bool {
    y.is_finite() && y >= 0.0 && (y - y.round()).abs() <= 1e-9
}

fn validate_count_responses(
    family: &str,
    y: &ArrayView1<'_, f64>,
    weights: &ArrayView1<'_, f64>,
) -> Result<(), HmcError> {
    for (i, (&yi, &wi)) in y.iter().zip(weights.iter()).enumerate() {
        if wi > 0.0 && !valid_count_response(yi) {
            return Err(HmcError::InvalidConfig {
                reason: format!(
                    "{family} response must be a finite non-negative integer at positive-weight row {i}; got {yi}"
                ),
            });
        }
    }
    Ok(())
}

fn validate_binary_responses(
    family: &str,
    y: &ArrayView1<'_, f64>,
    weights: &ArrayView1<'_, f64>,
) -> Result<(), HmcError> {
    for (i, (&yi, &wi)) in y.iter().zip(weights.iter()).enumerate() {
        if wi > 0.0 && !(yi == 0.0 || yi == 1.0) {
            return Err(HmcError::InvalidConfig {
                reason: format!(
                    "{family} response must be exactly 0 or 1 at positive-weight row {i}; got {yi}"
                ),
            });
        }
    }
    Ok(())
}

/// Compute the identifiable-subspace Jeffreys/Firth contribution and its
/// β-gradient.
///
/// HMC uses the same `FirthDenseOperator` as the REML exact-gradient path.
/// The operator owns the reduced identifiable Fisher factorization, the
/// Jeffreys log-determinant, and the analytic β-gradient.
///
/// Takes the full `LikelihoodSpec` — not a `NutsFamily` — because the
/// Jeffreys determinant is built from the *inverse link's* Fisher-weight
/// jet: at η = 0 the logit weight is 1/4 while probit's is 2/π, so
/// collapsing every binomial link to logit produces the wrong determinant
/// and gradient for probit / cloglog / adaptive (SAS, mixture) links.
fn firth_jeffreys_logp_and_grad(
    likelihood: &LikelihoodSpec,
    data: &SharedData,
    eta: &Array1<f64>,
) -> Result<(f64, Array1<f64>), HmcError> {
    if eta.len() != data.n_samples {
        return Err(HmcError::DimensionMismatch {
            reason: format!(
                "Firth Jeffreys term eta length {} != number of samples {}",
                eta.len(),
                data.n_samples
            ),
        });
    }
    if data.dim == 0 || data.n_samples == 0 {
        return Ok((0.0, Array1::zeros(data.dim)));
    }
    validate_firth_likelihood_support(likelihood, true)?;
    if data.weights.iter().all(|w| *w == 0.0) {
        return Ok((0.0, Array1::zeros(data.dim)));
    }

    let jeffreys_link =
        likelihood_spec_jeffreys_link(likelihood).ok_or_else(|| HmcError::FirthUnsupported {
            reason: format!(
                "Firth Jeffreys term has no Fisher-weight jet for {}",
                likelihood.pretty_name()
            ),
        })?;
    let op = if data.weights.iter().all(|&w| w == 1.0) {
        FirthDenseOperator::build_for_link(&jeffreys_link, data.x.as_ref(), eta)
    } else {
        FirthDenseOperator::build_with_observation_weights_for_link(
            &jeffreys_link,
            data.x.as_ref(),
            eta,
            data.weights.view(),
        )
    }
    .map_err(|e| HmcError::SamplingFailed {
        reason: format!("Firth Jeffreys operator failed: {e}"),
    })?;
    Ok(op.jeffreys_logdet_and_beta_gradient())
}

// ============================================================================
// Shared family log-likelihood helpers
// ============================================================================
//
// Freestanding functions for computing ℓ(y|β) and ∇_β ℓ for each supported
// family. Used by both `NutsPosterior` (fixed-ρ β-only sampling) and
// `JointBetaRhoPosterior` (joint β+ρ sampling).

fn nuts_family_logp_and_grad_into(
    family: NutsFamily,
    data: &SharedData,
    eta: &Array1<f64>,
    residual: &mut Array1<f64>,
) -> (f64, Array1<f64>) {
    match family {
        NutsFamily::BinomialLogit => logit_logp_and_grad_into(data, eta, residual),
        NutsFamily::BinomialProbit => probit_logp_and_grad_into(data, eta, residual),
        NutsFamily::BinomialCLogLog => cloglog_logp_and_grad_into(data, eta, residual),
        NutsFamily::Gaussian => gaussian_logp_and_grad_into(data, eta, residual),
        NutsFamily::PoissonLog => poisson_log_logp_and_grad(data, eta),
        // Family mapping: TweedieLog stores variance power p in data.gamma_shape.
        // Its dispersion phi stays in data.dispersion, matching REML scale ownership.
        NutsFamily::TweedieLog => tweedie_log_quasilogp_and_grad(data, eta, data.gamma_shape),
        NutsFamily::NegativeBinomialLog => {
            // Family mapping: NegativeBinomialLog stores theta in data.gamma_shape.
            // NB has unit REML scale; theta is never sourced from fixed_phi.
            negative_binomial_log_logp_and_grad(data, eta, data.gamma_shape)
        }
        NutsFamily::GammaLog => gamma_log_logp_and_grad(data, eta),
    }
}

#[derive(Clone, Debug)]
struct BinomialLinkTerms {
    log_mu: f64,
    log1m_mu: f64,
    dlog_mu_deta: f64,
    dlog1m_mu_deta: f64,
    dmu_dlink: Vec<f64>,
}

#[inline]
fn log_terms_from_mu_and_dmu(
    mu: f64,
    dmu_deta: f64,
    dmu_dlink: Vec<f64>,
) -> Result<BinomialLinkTerms, String> {
    if !(mu.is_finite() && (0.0..=1.0).contains(&mu) && dmu_deta.is_finite()) {
        return Err(format!(
            "binomial inverse link returned invalid mu/deta derivative: mu={mu}, dmu_deta={dmu_deta}"
        ));
    }
    let log_mu = if mu == 0.0 {
        f64::NEG_INFINITY
    } else {
        mu.ln()
    };
    let one_minus_mu = 1.0 - mu;
    let log1m_mu = if one_minus_mu == 0.0 {
        f64::NEG_INFINITY
    } else {
        one_minus_mu.ln()
    };
    let dlog_mu_deta = if mu == 0.0 {
        f64::INFINITY.copysign(dmu_deta)
    } else {
        dmu_deta / mu
    };
    let dlog1m_mu_deta = if one_minus_mu == 0.0 {
        f64::NEG_INFINITY.copysign(dmu_deta)
    } else {
        -dmu_deta / one_minus_mu
    };
    Ok(BinomialLinkTerms {
        log_mu,
        log1m_mu,
        dlog_mu_deta,
        dlog1m_mu_deta,
        dmu_dlink,
    })
}

#[inline]
fn binomial_link_terms(
    inverse_link: &InverseLink,
    eta: f64,
    n_link_params: usize,
) -> Result<BinomialLinkTerms, String> {
    let jet =
        inverse_link_jet_for_inverse_link(inverse_link, eta).map_err(|err| err.to_string())?;
    let mut dmu_dlink = vec![0.0; n_link_params];
    if n_link_params > 0 {
        match inverse_link
            .param_partials(eta)
            .map_err(|err| err.to_string())?
        {
            Some(LinkParamPartials::Sas(partials)) => {
                if n_link_params != 2 {
                    return Err(format!(
                        "SAS/Beta-Logistic link parameter dimension mismatch: expected 2, got {n_link_params}"
                    ));
                }
                dmu_dlink[0] = partials.djet_depsilon.mu;
                dmu_dlink[1] = partials.djet_dlog_delta.mu;
            }
            Some(LinkParamPartials::Mixture(partials)) => {
                if partials.djet_drho.len() != n_link_params {
                    return Err(format!(
                        "mixture link parameter dimension mismatch: expected {}, got {n_link_params}",
                        partials.djet_drho.len()
                    ));
                }
                for (slot, partial) in dmu_dlink.iter_mut().zip(partials.djet_drho.iter()) {
                    *slot = partial.mu;
                }
            }
            None => {
                return Err(format!(
                    "joint HMC expected {n_link_params} adaptive link parameters, but the inverse link exposes none"
                ));
            }
        }
    }
    log_terms_from_mu_and_dmu(jet.mu, jet.d1, dmu_dlink)
}

fn joint_binomial_logp_grad_and_link_grad(
    inverse_link: &InverseLink,
    data: &SharedData,
    eta: &Array1<f64>,
    n_link_params: usize,
) -> Result<(f64, Array1<f64>, Array1<f64>), String> {
    let n = data.n_samples;
    // Per-row: compute stable log-tail terms and derivatives without endpoint
    // clamping. Positive-weight responses were validated as Bernoulli before
    // target construction, so each row selects exactly one log branch.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let per_row: Result<Vec<(f64, f64, Vec<f64>)>, String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let y_i = data.y[i];
            let w_i = data.weights[i];
            if w_i <= 0.0 {
                return Ok((0.0, 0.0, vec![0.0; n_link_params]));
            }
            let terms = binomial_link_terms(inverse_link, eta[i], n_link_params)?;
            if y_i == 1.0 {
                let inv_mu = terms.log_mu.exp().recip();
                let log_mu = terms.log_mu;
                let dlog_mu_deta = terms.dlog_mu_deta;
                let grad_link = terms
                    .dmu_dlink
                    .into_iter()
                    .map(|dmu| w_i * dmu * inv_mu)
                    .collect();
                Ok((w_i * log_mu, w_i * dlog_mu_deta, grad_link))
            } else if y_i == 0.0 {
                let inv_one_minus_mu = terms.log1m_mu.exp().recip();
                let log1m_mu = terms.log1m_mu;
                let dlog1m_mu_deta = terms.dlog1m_mu_deta;
                let grad_link = terms
                    .dmu_dlink
                    .into_iter()
                    .map(|dmu| -w_i * dmu * inv_one_minus_mu)
                    .collect();
                Ok((w_i * log1m_mu, w_i * dlog1m_mu_deta, grad_link))
            } else {
                Err(format!(
                    "binomial joint HMC response must be exactly 0 or 1 after validation; got {y_i}"
                ))
            }
        })
        .collect();
    let per_row = per_row?;
    let mut residual = Array1::<f64>::zeros(n);
    let mut grad_link = Array1::<f64>::zeros(n_link_params);
    let mut ll = 0.0;
    for (i, (ll_i, residual_i, grad_link_i)) in per_row.into_iter().enumerate() {
        ll += ll_i;
        residual[i] = residual_i;
        for (slot, value) in grad_link.iter_mut().zip(grad_link_i.iter()) {
            *slot += *value;
        }
    }

    Ok((ll, fast_atv(&data.x, &residual), grad_link))
}

fn joint_binomial_logp_and_grad(
    likelihood: &LikelihoodSpec,
    data: &SharedData,
    eta: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    if !matches!(likelihood.response, ResponseFamily::Binomial) {
        return Err(HmcError::UnsupportedFamily {
            reason: format!(
                "{} is not a binomial joint-HMC family",
                likelihood.pretty_name()
            ),
        }
        .into());
    }
    match &likelihood.link {
        InverseLink::Standard(StandardLink::Logit) => Ok(logit_logp_and_grad(data, eta)),
        InverseLink::Standard(StandardLink::Probit) => Ok(probit_logp_and_grad(data, eta)),
        InverseLink::Standard(StandardLink::CLogLog) => Ok(cloglog_logp_and_grad(data, eta)),
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {
            let (ll, grad_beta, _) =
                joint_binomial_logp_grad_and_link_grad(&likelihood.link, data, eta, 0)?;
            Ok((ll, grad_beta))
        }
        InverseLink::Standard(_) => Err(HmcError::UnsupportedFamily {
            reason: format!(
                "{} is not a binomial joint-HMC family",
                likelihood.pretty_name()
            ),
        }
        .into()),
    }
}

fn joint_family_logp_grad_and_link_grad(
    likelihood: &LikelihoodSpec,
    data: &SharedData,
    eta: &Array1<f64>,
    n_link_params: usize,
) -> Result<(f64, Array1<f64>, Array1<f64>), String> {
    match (&likelihood.response, &likelihood.link) {
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
            let (ll, grad) = logit_logp_and_grad(data, eta);
            Ok((ll, grad, Array1::zeros(n_link_params)))
        }
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
            let (ll, grad) = probit_logp_and_grad(data, eta);
            Ok((ll, grad, Array1::zeros(n_link_params)))
        }
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
            let (ll, grad) = cloglog_logp_and_grad(data, eta);
            Ok((ll, grad, Array1::zeros(n_link_params)))
        }
        (
            ResponseFamily::Binomial,
            InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_),
        ) => joint_binomial_logp_grad_and_link_grad(&likelihood.link, data, eta, n_link_params),
        _ => {
            let (ll, grad) = joint_family_logp_and_grad(likelihood, data, eta)?;
            Ok((ll, grad, Array1::zeros(n_link_params)))
        }
    }
}

fn joint_family_logp_and_grad(
    likelihood: &LikelihoodSpec,
    data: &SharedData,
    eta: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    match &likelihood.response {
        ResponseFamily::Binomial => joint_binomial_logp_and_grad(likelihood, data, eta),
        ResponseFamily::Gaussian => Ok(gaussian_logp_and_grad(data, eta)),
        ResponseFamily::Poisson => Ok(poisson_log_logp_and_grad(data, eta)),
        ResponseFamily::Tweedie { p } => {
            // Family mapping: Tweedie payload p is the variance power.
            // Its dispersion phi stays in data.dispersion, matching REML.
            let p = *p;
            if !is_valid_tweedie_power(p) {
                return Err(HmcError::InvalidConfig {
                    reason: format!(
                        "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                    ),
                }
                .into());
            }
            Ok(tweedie_log_quasilogp_and_grad(data, eta, p))
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            // Family mapping: NegativeBinomial payload theta is overdispersion.
            // NB keeps unit REML scale and never reads fixed_phi for theta.
            Ok(negative_binomial_log_logp_and_grad(data, eta, *theta))
        }
        ResponseFamily::Beta { .. } => Err(HmcError::UnsupportedFamily {
            reason: "Joint HMC fallback is not implemented for BetaLogit".to_string(),
        }
        .into()),
        ResponseFamily::Gamma => Ok(gamma_log_logp_and_grad(data, eta)),
        ResponseFamily::RoystonParmar => Err(HmcError::UnsupportedFamily {
            reason: "Joint HMC fallback is not implemented for RoystonParmar".to_string(),
        }
        .into()),
    }
}

/// Logistic regression log-likelihood and gradient.
///
/// log p(y|η) = y·η − log(1 + exp(η)), gradient = X'(w ⊙ (y − μ))
fn logit_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let mut residual = Array1::<f64>::zeros(data.n_samples);
    logit_logp_and_grad_into(data, eta, &mut residual)
}

fn logit_logp_and_grad_into(
    data: &SharedData,
    eta: &Array1<f64>,
    residual: &mut Array1<f64>,
) -> (f64, Array1<f64>) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    assert_eq!(residual.len(), n);
    // Per-row independent: write residual entry into a pre-allocated buffer and
    // reduce the ll contribution in parallel — avoids materialising a
    // Vec<(f64, f64)> and the serial scatter that follows.
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let eta_i = eta[i];
            let y_i = data.y[i];
            let w_i = data.weights[i];
            let mu = gam_linalg::utils::stable_logistic(eta_i);
            *slot = w_i * (y_i - mu);
            w_i * (y_i * eta_i - gam_linalg::utils::stable_softplus(eta_i))
        })
        .sum();

    let grad_ll = fast_atv(data.x.as_ref(), &*residual);
    (ll, grad_ll)
}

/// Probit regression log-likelihood and gradient.
///
/// log p(y|η) = Σ [y·log Φ(η) + (1-y)·log(1-Φ(η))],
/// gradient_i = w_i · [y_i · φ(η_i)/Φ(η_i) − (1-y_i) · φ(η_i)/(1−Φ(η_i))]
///
/// Uses erfc-based log Φ for numerical stability.
fn probit_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let mut residual = Array1::<f64>::zeros(data.n_samples);
    probit_logp_and_grad_into(data, eta, &mut residual)
}

fn probit_logp_and_grad_into(
    data: &SharedData,
    eta: &Array1<f64>,
    residual: &mut Array1<f64>,
) -> (f64, Array1<f64>) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    assert_eq!(residual.len(), n);
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let eta_i = eta[i];
            let y_i = data.y[i];
            let w_i = data.weights[i];
            let log_phi_pos = log_ndtr(eta_i);
            let log_phi_neg = log_ndtr(-eta_i);
            let log_phi_val = standard_normal_log_pdf(eta_i);
            let ratio_pos = (log_phi_val - log_phi_pos).exp();
            let ratio_neg = (log_phi_val - log_phi_neg).exp();
            let grad_i = y_i * ratio_pos - (1.0 - y_i) * ratio_neg;
            *slot = w_i * grad_i;
            w_i * (y_i * log_phi_pos + (1.0 - y_i) * log_phi_neg)
        })
        .sum();

    let grad_ll = fast_atv(data.x.as_ref(), &*residual);
    (ll, grad_ll)
}

/// Complementary log-log regression log-likelihood and gradient.
///
/// CLogLog link: μ = 1 − exp(−exp(η))
/// log p(y|η) = Σ [y·log(1−exp(−exp(η))) + (1−y)·(−exp(η))]
/// gradient_i = w_i · [y_i · exp(η_i)·exp(−exp(η_i)) / (1−exp(−exp(η_i))) − (1−y_i)·exp(η_i)]
#[inline]
fn cloglog_bernoulli_logp_and_residual(eta: f64, y: f64) -> Result<(f64, f64), EstimationError> {
    if !eta.is_finite() {
        gam_problem::bail_invalid_estim!("cloglog eta must be finite; got {eta}");
    }
    let exp_eta = eta.exp();
    // Deep left tail: exp(η) underflows to zero below η ≈ −745, but the exact
    // limits are log μ = log(1 − exp(−e^η)) → η and d(log μ)/dη → 1, so the
    // valid finite log-density is preserved instead of degenerating to
    // log(0) = −∞. (Previously any η outside an arbitrary ±700 window was
    // declared impossible, truncating the sampled posterior.)
    if exp_eta == 0.0 {
        let ll_i = y * eta; // (1 − y)·(−exp_eta) is exactly 0 here
        let residual_i = y; // grad_log_mu → 1, (1 − y)·exp_eta → 0
        return Ok((ll_i, residual_i));
    }
    // log_mu = log(1 - exp(-exp_eta)); exp_eta > 0 here, so this is exactly
    // the canonical cancellation-free log1mexp (single source of truth).
    let log_mu = crate::probability::log1mexp_positive(exp_eta);
    let log_one_minus_mu = -exp_eta;
    let grad_log_mu = (eta - exp_eta - log_mu).exp();
    let ll_i = y * log_mu + (1.0 - y) * log_one_minus_mu;
    let residual_i = y * grad_log_mu - (1.0 - y) * exp_eta;
    Ok((ll_i, residual_i))
}

fn cloglog_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let mut residual = Array1::<f64>::zeros(data.n_samples);
    cloglog_logp_and_grad_into(data, eta, &mut residual)
}

fn cloglog_logp_and_grad_into(
    data: &SharedData,
    eta: &Array1<f64>,
    residual: &mut Array1<f64>,
) -> (f64, Array1<f64>) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    assert_eq!(residual.len(), n);
    if eta.iter().any(|&eta_i| !eta_i.is_finite()) {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let y_i = data.y[i];
            let w_i = data.weights[i];
            let (ll_i, residual_i) =
                cloglog_bernoulli_logp_and_residual(eta[i], y_i).expect("validated cloglog eta");
            *slot = w_i * residual_i;
            w_i * ll_i
        })
        .sum();

    // A finite η can still exhaust binary64 (y = 0 with exp(η) overflowing):
    // that is a genuine log-density underflow, rejected as −∞ with a zero
    // gradient — not an a-priori support window.
    if !ll.is_finite() {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let grad_ll = fast_atv(data.x.as_ref(), &*residual);
    (ll, grad_ll)
}

/// Gaussian log-likelihood and gradient.
///
/// log p(y|η) = −½ (w/φ)·(y − η)²,  gradient = (1/φ)·X'(w ⊙ (y − η))
///
/// Both the log-likelihood and its β-gradient are scaled by `1/φ` so that
/// the working likelihood matches the φ-scaled posterior covariance the
/// HMC whitening transform targets. With `φ == 1` (the only value
/// passed by the pre-refactor call sites) this collapses to the original
/// `−½ w·(y − η)²` expression; with an estimated dispersion (the
/// `Dispersion::Estimated(σ²)` branch) it removes the silent unit-σ
/// approximation the Gaussian NUTS log-density used previously.
fn gaussian_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let mut weighted_residual = Array1::<f64>::zeros(data.n_samples);
    gaussian_logp_and_grad_into(data, eta, &mut weighted_residual)
}

fn gaussian_logp_and_grad_into(
    data: &SharedData,
    eta: &Array1<f64>,
    weighted_residual: &mut Array1<f64>,
) -> (f64, Array1<f64>) {
    use gam_problem::dispersion_cov::DispersionExt as _;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    let inv_phi = data.dispersion.inv_phi();
    assert_eq!(weighted_residual.len(), n);
    // Per-row: residual = y - η, weighted_residual = (w/φ)·residual,
    // ll contribution = -0.5·(w/φ)·residual². All independent across rows.
    let ll: f64 = weighted_residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let residual = data.y[i] - eta[i];
            let w_i = data.weights[i];
            let scaled = w_i * inv_phi;
            *slot = scaled * residual;
            -0.5 * scaled * residual * residual
        })
        .sum();

    let grad_ll = fast_atv(data.x.as_ref(), &*weighted_residual);
    (ll, grad_ll)
}

/// Poisson(log) log-likelihood and gradient.
///
/// log p(y|η) = y·η − exp(η), gradient = X'(w ⊙ (y − μ))
fn poisson_log_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    // Only non-finite η invalidates the target. Any finite η has a valid
    // log-density (e.g. y = 0, η = −701 gives ℓ = −exp(−701) ≈ 0); the old
    // ±700 window declared such points impossible and truncated the
    // posterior. Genuine binary64 exhaustion (exp(η) overflowing against a
    // positive count) is caught after the sum below.
    if eta.iter().any(|&eta_i| !eta_i.is_finite()) {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let mut residual = Array1::<f64>::zeros(n);
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let eta_i = eta[i];
            let mu_i = eta_i.exp();
            let y_i = data.y[i];
            let w_i = data.weights[i];
            *slot = w_i * (y_i - mu_i);
            w_i * (y_i * eta_i - mu_i)
        })
        .sum();

    if !ll.is_finite() {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

fn tweedie_log_quasilogp_and_grad(
    data: &SharedData,
    eta: &Array1<f64>,
    p: f64,
) -> (f64, Array1<f64>) {
    use gam_problem::dispersion_cov::DispersionExt as _;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    // Family mapping: Tweedie p is the variant payload; phi is data.dispersion.
    // Invalid payloads invalidate the target instead of falling back to p=1.5.
    if !is_valid_tweedie_power(p) {
        return (f64::NAN, Array1::from_elem(data.dim, f64::NAN));
    }
    // Finite η is always in-support for the quasi-likelihood; the old ±700
    // window artificially truncated the posterior. Binary64 exhaustion is
    // caught after the sum.
    if eta.iter().any(|&eta_i| !eta_i.is_finite()) {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let inv_phi = data.dispersion.inv_phi();
    let mut residual = Array1::<f64>::zeros(n);
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let eta_i = eta[i];
            let mu_i = eta_i.exp().max(1e-300);
            let y_i = data.y[i];
            let w_i = data.weights[i] * inv_phi;
            *slot = w_i * (y_i - mu_i) * mu_i.powf(1.0 - p);
            let qll = y_i * mu_i.powf(1.0 - p) / (1.0 - p) - mu_i.powf(2.0 - p) / (2.0 - p);
            w_i * qll
        })
        .sum();

    if !ll.is_finite() {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

fn negative_binomial_log_logp_and_grad(
    data: &SharedData,
    eta: &Array1<f64>,
    theta: f64,
) -> (f64, Array1<f64>) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    if !(theta.is_finite() && theta > 0.0)
        || eta.iter().any(|&eta_i| !eta_i.is_finite())
        || data
            .y
            .iter()
            .zip(data.weights.iter())
            .any(|(&y_i, &w_i)| w_i > 0.0 && !valid_count_response(y_i))
    {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let mut residual = Array1::<f64>::zeros(n);
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let eta_i = eta[i];
            let mu_i = eta_i.exp().max(1e-12);
            let y_i = data.y[i];
            let w_i = data.weights[i];
            if w_i <= 0.0 {
                *slot = 0.0;
                return 0.0;
            }
            let log_mu_term = if y_i > 0.0 { y_i * mu_i.ln() } else { 0.0 };
            *slot = w_i * theta * (y_i - mu_i) / (theta + mu_i);
            w_i * (statrs::function::gamma::ln_gamma(y_i + theta)
                - statrs::function::gamma::ln_gamma(theta)
                - statrs::function::gamma::ln_gamma(y_i + 1.0)
                + theta * (theta.ln() - (theta + mu_i).ln())
                + log_mu_term
                - y_i * (theta + mu_i).ln())
        })
        .sum();

    if !ll.is_finite() {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

fn gamma_log_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
    let n = data.n_samples;
    if eta.iter().any(|&eta_i| !eta_i.is_finite()) {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let shape = data.gamma_shape.max(1e-10);
    // Hoist shape-only constants out of the per-sample loop: ln Γ(shape) and
    // shape · ln(shape) are independent of i, so previously each sample paid
    // an extra `ln_gamma` and `ln` plus a multiply. n is typically large-scale-
    // scale, so this collapses Θ(n) gamma-function evaluations to one.
    let shape_ln_shape = shape * shape.ln();
    let log_gamma_shape = statrs::function::gamma::ln_gamma(shape);
    let shape_minus_one = shape - 1.0;
    let mut residual = Array1::<f64>::zeros(n);
    let ll: f64 = residual
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .map(|(i, slot)| {
            let eta_i = eta[i];
            let mu_i = eta_i.exp();
            let y_i = data.y[i];
            let w_i = data.weights[i];
            let ll_i = w_i
                * (shape_ln_shape - log_gamma_shape - shape * eta_i
                    + shape_minus_one * y_i.max(1e-12).ln()
                    - shape * y_i / mu_i);
            *slot = w_i * shape * (y_i / mu_i - 1.0);
            ll_i
        })
        .sum();

    if !ll.is_finite() {
        return (f64::NEG_INFINITY, Array1::zeros(data.dim));
    }
    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

#[cfg(test)]
mod tests {
    use super::{
        FamilyNutsInputs, GlmFlatInputs, JointBetaRhoInputs, JointBetaRhoPosterior,
        LinkWigglePosterior, LinkWiggleSplineArtifacts, NutsConfig, NutsFamily, NutsPosterior,
        NutsResult, SharedData, cloglog_bernoulli_logp_and_residual, firth_jeffreys_logp_and_grad,
        joint_family_logp_and_grad, laplace_directional_cubic_diagnostic,
        laplace_skewness_threshold, laplace_trustworthiness_from_skewness,
        run_joint_beta_rho_sampling, run_logit_polya_gamma_gibbs,
        run_nuts_sampling_flattened_family,
    };
    use gam_linalg::matrix::DesignMatrix;
    use gam_models::survival::{PenaltyBlocks, SurvivalMonotonicityPenalty, SurvivalSpec};
    use gam_problem::types::{
        InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LogLikelihoodNormalization,
        ResponseFamily, RhoPrior, StandardLink,
    };
    use gam_solve::estimate::{
        BlockRole, FitGeometry, FitInference, FittedBlock, FittedLinkState, UnifiedFitResult,
        UnifiedFitResultParts,
    };
    use gam_terms::construction::CanonicalPenalty;
    use general_mcmc::generic_hmc::HamiltonianTarget;
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    #[test]
    fn posterior_interval_uses_shared_linear_quantiles() {
        let result = NutsResult {
            samples: array![[0.0], [1.0], [2.0], [3.0]],
            posterior_mean: array![1.5],
            posterior_std: array![1.0],
            rhat: 1.0,
            ess: 4.0,
            converged: true,
        };

        let (lower, upper) = result.posterior_interval_of(|row| row[0], 25.0, 75.0);

        assert!((lower - 0.75).abs() < 1e-12, "lower = {lower}");
        assert!((upper - 2.25).abs() < 1e-12, "upper = {upper}");
    }

    impl NutsPosterior {
        /// Test-only allocation wrapper around `compute_logp_and_grad_nd_into`.
        pub(super) fn compute_logp_and_grad_nd(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
            let mut residual = Array1::<f64>::zeros(self.data.n_samples);
            let mut grad = Array1::<f64>::zeros(z.len());
            let logp = self.compute_logp_and_grad_nd_into(z, &mut residual, &mut grad);
            (logp, grad)
        }
    }

    impl LinkWigglePosterior {
        /// Test-only allocation wrapper around `compute_logp_and_grad_into`.
        pub(super) fn compute_logp_and_grad(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
            let dim = self.p_base + self.p_link;
            let mut grad = Array1::<f64>::zeros(dim);
            let logp = self.compute_logp_and_grad_into(z, &mut grad);
            (logp, grad)
        }
    }

    impl JointBetaRhoPosterior {
        /// Test-only allocation wrapper around `compute_joint_logp_and_grad_into`.
        pub(super) fn compute_joint_logp_and_grad(
            &self,
            params: &Array1<f64>,
        ) -> (f64, Array1<f64>) {
            let total_dim = self.n_beta + self.n_rho + self.n_link_params;
            let mut grad = Array1::<f64>::zeros(total_dim);
            let logp = self.compute_joint_logp_and_grad_into(params, &mut grad);
            (logp, grad)
        }
    }

    fn hmc_test_fit(
        blocks: Vec<FittedBlock>,
        inference: Option<FitInference>,
        geometry: Option<FitGeometry>,
    ) -> UnifiedFitResult {
        let lambdas = Array1::zeros(0);
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks,
            log_lambdas: lambdas.clone(),
            lambdas,
            likelihood_family: Some(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: -1.0,
            deviance: 2.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 1,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference,
            fitted_link: FittedLinkState::Standard(None),
            geometry,
            block_states: Vec::new(),
            pirls_status: gam_solve::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: Default::default(),
            inner_cycles: 0,
        })
        .expect("valid HMC handoff test fit")
    }

    #[test]
    fn hmc_whitening_consumes_standard_fit_inference_hessian() {
        let hessian = array![[2.0, 0.1], [0.1, 1.6]];
        let fit = hmc_test_fit(
            vec![FittedBlock {
                beta: array![0.05, -0.1],
                role: BlockRole::Mean,
                edf: 2.0,
                lambdas: Array1::zeros(0),
            }],
            Some(FitInference {
                edf_by_block: vec![],
                penalty_block_trace: vec![],
                edf_total: 2.0,
                smoothing_correction: None,
                penalized_hessian: hessian.clone().into(),
                working_weights: array![1.0, 1.0, 1.0],
                working_response: array![0.0, 0.1, -0.2],
                reparam_qs: None,
                dispersion: gam_solve::estimate::Dispersion::Known(1.0),
                beta_covariance: None,
                beta_standard_errors: None,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                weighted_gram: None,
                bias_correction_beta: None,
                bias_correction_jacobian: None,
            }),
            None,
        );

        let explicit = super::explicit_fit_hessian_for_whitening(&fit, 2, "standard fit")
            .expect("standard fit exports explicit Hessian");
        assert_eq!(explicit, &hessian);

        let x = array![[1.0, 0.0], [1.0, 0.5], [1.0, -0.5]];
        let y = array![0.0, 0.2, -0.1];
        let weights = Array1::ones(3);
        let penalty = Array2::eye(2);
        NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            fit.beta.view(),
            explicit.view(),
            NutsFamily::Gaussian,
            1.0,
            gam_solve::estimate::Dispersion::Known(1.0),
            false,
        )
        .expect("HMC target whitens with upstream Hessian");
    }

    #[test]
    fn hmc_whitening_consumes_blockwise_geometry_hessian() {
        let hessian = array![[3.0, 0.2], [0.2, 2.0]];
        let fit = hmc_test_fit(
            vec![
                FittedBlock {
                    beta: array![0.1],
                    role: BlockRole::Location,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: array![-0.2],
                    role: BlockRole::Scale,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            None,
            Some(FitGeometry {
                penalized_hessian: hessian.clone().into(),
                working_weights: array![1.0, 0.8],
                working_response: array![0.0, 0.1],
            }),
        );

        let explicit = super::explicit_fit_hessian_for_whitening(&fit, 2, "blockwise fit")
            .expect("blockwise fit exports materialized Hessian");
        assert_eq!(explicit, &hessian);
    }

    #[test]
    fn hmc_whitening_rejects_covariance_only_fit_without_synthesizing_hessian() {
        let fit = UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: array![0.0],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            }],
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: -1.0,
            deviance: 2.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 1,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: Some(array![[0.5]]),
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: gam_solve::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: Default::default(),
            inner_cycles: 0,
        })
        .expect("covariance-only fit can exist for prediction");

        let err = super::explicit_fit_hessian_for_whitening(&fit, 1, "covariance-only fit")
            .expect_err("HMC must not invert covariance as a Hessian fallback");
        assert!(
            err.contains("missing an explicit penalized Hessian"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn log1pexp_is_finite_for_extreme_eta() {
        assert!(gam_linalg::utils::stable_softplus(1000.0).is_finite());
        assert!(gam_linalg::utils::stable_softplus(-1000.0).is_finite());
        assert!((gam_linalg::utils::stable_softplus(-1000.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_stable_behaves_at_extremes() {
        let hi = gam_linalg::utils::stable_logistic(1000.0);
        let lo = gam_linalg::utils::stable_logistic(-1000.0);
        assert!((1.0 - 1e-12..=1.0).contains(&hi));
        assert!((0.0..=1e-12).contains(&lo));
    }

    #[test]
    fn cloglog_log_mu_uses_complementary_loglog_inverse_link() {
        let eta = -1.0_f64;
        let (ll_y1, residual_y1) =
            cloglog_bernoulli_logp_and_residual(eta, 1.0).expect("valid eta");
        let expected = (1.0 - (-eta.exp()).exp()).ln();
        let wrong_log_one_minus_exp_eta = (1.0 - eta.exp()).ln();

        assert!((ll_y1 - expected).abs() < 1e-14);
        assert!((ll_y1 - wrong_log_one_minus_exp_eta).abs() > 0.5);

        let eps = 1e-6;
        let (lp, _) = cloglog_bernoulli_logp_and_residual(eta + eps, 1.0).expect("valid eta");
        let (lm, _) = cloglog_bernoulli_logp_and_residual(eta - eps, 1.0).expect("valid eta");
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (residual_y1 - fd).abs() < 1e-9,
            "cloglog residual is not the derivative of log μ: analytic={residual_y1}, fd={fd}"
        );
    }

    #[test]
    fn finite_eta_beyond_the_old_support_window_keeps_its_valid_log_density() {
        // A Poisson row with y = 0 at η = −701 has log-likelihood
        // −exp(−701) ≈ 0 — a perfectly valid, essentially maximal density.
        // The old hard-coded ±700 window declared it impossible (−∞),
        // truncating the sampled posterior at an arbitrary boundary.
        let data = SharedData {
            x: Arc::new(array![[1.0]]),
            y: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            mode: Arc::new(array![0.0]),
            offset: None,
            gamma_shape: 1.0,
            dispersion: gam_solve::model_types::Dispersion::Known(1.0),
            n_samples: 1,
            dim: 1,
        };
        let eta = array![-701.0];
        let (ll, grad) = super::poisson_log_logp_and_grad(&data, &eta);
        assert!(
            ll.is_finite() && ll.abs() < 1e-300,
            "Poisson y=0, eta=-701 must keep its ~0 log-density, got {ll}"
        );
        assert!(grad[0].is_finite());

        // Deep cloglog left tail: exp(η) underflows below η ≈ −745, but the
        // exact limits are log μ → η and d(log μ)/dη → 1.
        let (ll_tail, res_tail) =
            cloglog_bernoulli_logp_and_residual(-750.0, 1.0).expect("finite eta is valid");
        assert!(
            (ll_tail - (-750.0)).abs() < 1e-9,
            "cloglog log-density must approach eta in the deep left tail, got {ll_tail}"
        );
        assert!((res_tail - 1.0).abs() < 1e-9, "residual limit is 1");

        // Genuine binary64 exhaustion (y > 0 against an overflowing mean) is
        // still rejected as −∞ with a zero gradient.
        let data_pos = SharedData {
            y: Arc::new(array![3.0]),
            ..data
        };
        let (ll_of, grad_of) = super::poisson_log_logp_and_grad(&data_pos, &array![710.0]);
        assert_eq!(ll_of, f64::NEG_INFINITY);
        assert_eq!(grad_of[0], 0.0);
    }

    #[test]
    fn link_wiggle_posterior_whitening_uses_supplied_explicit_joint_hessian() {
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![0.0, 1.0, 1.0];
        let weights = Array1::ones(3);
        let penalty_base = Array2::zeros((1, 1));
        let penalty_link = Array2::zeros((1, 1));
        let mode_beta = array![0.2];
        let mode_theta = array![0.05];
        let hessian = array![[4.0, 1.0], [1.0, 3.0]];
        let spline = LinkWiggleSplineArtifacts {
            knot_range: (-1.0, 1.0),
            knot_vector: Array1::from_vec(vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
            degree: 2,
        };

        let posterior = LinkWigglePosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty_base.view(),
            penalty_link.view(),
            mode_beta.view(),
            mode_theta.view(),
            hessian.view(),
            spline,
            NutsFamily::BinomialLogit,
            1.0,
        )
        .expect("link-wiggle posterior should accept explicit SPD joint Hessian");

        let reconstructed_cov = posterior.chol().dot(&posterior.chol().t());
        let eye_from_hessian = hessian.dot(&reconstructed_cov);
        for r in 0..2 {
            for c in 0..2 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (eye_from_hessian[[r, c]] - expected).abs() < 1e-10,
                    "whitening did not use the supplied explicit joint Hessian at ({r},{c}): got {} expected {}",
                    eye_from_hessian[[r, c]],
                    expected
                );
            }
        }
    }

    #[test]
    fn link_wiggle_cloglog_gradient_matches_its_log_likelihood() {
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let weights = array![1.0, 1.2, 0.8, 1.4];
        let penalty_base = Array2::zeros((1, 1));
        let penalty_link = Array2::zeros((1, 1));
        let mode_beta = array![-0.8];
        let mode_theta = array![0.04];
        let hessian = Array2::eye(2);
        let spline = LinkWiggleSplineArtifacts {
            knot_range: (-1.5, 0.5),
            knot_vector: Array1::from_vec(vec![-1.5, -1.5, -1.5, 0.5, 0.5, 0.5]),
            degree: 2,
        };

        let posterior = LinkWigglePosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty_base.view(),
            penalty_link.view(),
            mode_beta.view(),
            mode_theta.view(),
            hessian.view(),
            spline,
            NutsFamily::BinomialCLogLog,
            1.0,
        )
        .expect("cloglog link-wiggle posterior");

        let z = array![0.2, -0.03];
        let (_, grad) = posterior.compute_logp_and_grad(&z);
        let eps = 1e-6;
        for j in 0..z.len() {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let (lp, _) = posterior.compute_logp_and_grad(&z_plus);
            let (lm, _) = posterior.compute_logp_and_grad(&z_minus);
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (grad[j] - fd).abs() < 1e-6,
                "link-wiggle cloglog gradient mismatch at {j}: analytic={}, fd={}",
                grad[j],
                fd
            );
        }
    }

    #[test]
    fn nuts_logitgradient_matches_finite_difference() {
        let x = array![[1.0, -0.5], [0.2, 0.7], [-1.0, 0.3], [0.5, -1.2]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.5, 0.8, 1.2];
        let penalty = array![[0.4, 0.0], [0.0, 0.6]];
        let mode = array![0.1, -0.2];
        let hessian = array![[2.0, 0.2], [0.2, 1.7]]; // SPD

        let posterior = NutsPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::BinomialLogit,
            1.0,
            gam_solve::estimate::Dispersion::Known(1.0),
            true,
        )
        .expect("posterior");

        let z = array![0.15, -0.35];
        let (_, grad) = posterior.compute_logp_and_grad_nd(&z);

        let eps = 1e-6;
        for j in 0..z.len() {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let (lp, _) = posterior.compute_logp_and_grad_nd(&z_plus);
            let (lm, _) = posterior.compute_logp_and_grad_nd(&z_minus);
            let fd = (lp - lm) / (2.0 * eps);
            assert_eq!(
                grad[j].signum(),
                fd.signum(),
                "gradient sign mismatch at {}: analytic={}, fd={}",
                j,
                grad[j],
                fd
            );
            assert!(
                (grad[j] - fd).abs() < 1e-5,
                "gradient mismatch at {}: analytic={}, fd={}",
                j,
                grad[j],
                fd
            );
        }
    }

    #[test]
    fn gamma_log_logp_and_grad_uses_fitted_shape() {
        let x = array![[1.0_f64], [1.0_f64]];
        let y = array![1.5_f64, 2.5_f64];
        let weights = array![1.0_f64, 2.0_f64];
        let eta = array![0.2_f64, 0.4_f64];
        let shape = 3.5_f64;
        let data = SharedData {
            x: Arc::new(x.clone()),
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            mode: Arc::new(Array1::zeros(1)),
            offset: None,
            gamma_shape: shape,
            dispersion: gam_solve::estimate::Dispersion::Known(1.0),
            n_samples: x.nrows(),
            dim: x.ncols(),
        };

        let (ll, grad) = super::gamma_log_logp_and_grad(&data, &eta);

        let mut expected_ll = 0.0;
        let mut expected_score = 0.0;
        for i in 0..eta.len() {
            let mu = eta[i].exp();
            expected_ll += weights[i]
                * (shape * shape.ln() - statrs::function::gamma::ln_gamma(shape) - shape * eta[i]
                    + (shape - 1.0) * y[i].ln()
                    - shape * y[i] / mu);
            expected_score += weights[i] * shape * (y[i] / mu - 1.0);
        }

        assert!((ll - expected_ll).abs() < 1e-12);
        assert_eq!(grad.len(), 1);
        assert!((grad[0] - expected_score).abs() < 1e-12);
    }

    /// Gamma observed information at the mode, `Xᵀ diag(w·ν·y/μ) X`, where the
    /// per-point curvature `w·ν·y/μ` is exactly `−∂/∂η` of the analytic score
    /// slot `w·ν·(y/μ − 1)` used by `gamma_log_logp_and_grad`.
    fn gamma_log_observed_information(
        x: &Array2<f64>,
        mode: &Array1<f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        shape: f64,
    ) -> Array2<f64> {
        let p = x.ncols();
        let eta = x.dot(mode);
        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..x.nrows() {
            let mu = eta[i].exp();
            let wt = weights[i] * shape * y[i] / mu;
            for a in 0..p {
                for b in 0..p {
                    h[[a, b]] += wt * x[[i, a]] * x[[i, b]];
                }
            }
        }
        h
    }

    /// Regression for #680: the whitened GammaLog NUTS target must reproduce
    /// the #679 coefficient-covariance contract `Vb = H⁻¹` (scale `1.0`), NOT
    /// the dispersion-double-counted `(1/ν)(XᵀΛX + S)⁻¹`.
    ///
    /// We set the stored Hessian to the *true* penalized curvature of the
    /// target at the mode, `H = Xᵀ diag(w·ν·y/μ) X + S` (Gamma observed
    /// information + the penalty added **unscaled** — exactly the #679 `H`).
    /// The whitened target's curvature in z at the mode is `Lᵀ Hβ L`. The fix
    /// makes `L Lᵀ = H⁻¹` and `Hβ = H`, so this is the identity. The pre-fix
    /// code scaled the penalty by `ν` and the whitening by `√φ`, turning the
    /// z-curvature into `φ·(I + (ν−1)·L_H⁻¹ S L_H⁻ᵀ) ≠ I` (for ν=4 the
    /// diagonal collapses toward ~0.25, never 1).
    #[test]
    fn gamma_log_nuts_target_curvature_matches_unscaled_hessian_issue_680() {
        let x = array![[1.0, -0.7], [1.0, 0.3], [1.0, 1.1], [1.0, -0.2], [1.0, 0.8],];
        let mode = array![0.4_f64, -0.6_f64];
        let y = array![1.2_f64, 0.7, 2.3, 0.9, 1.6];
        let weights = array![1.0_f64, 1.5, 0.8, 1.2, 1.0];
        // ν = 1/φ = 4 ⇒ φ = 0.25: a large, easily-detectable double-count.
        let shape = 4.0_f64;
        let p = x.ncols();

        let h_data = gamma_log_observed_information(&x, &mode, &y, &weights, shape);
        // A genuine PD smoothing penalty so the ×ν double-count is detectable.
        let s = array![[0.5_f64, 0.1], [0.1, 0.9]];
        let hessian = &h_data + &s;

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            s.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::GammaLog,
            shape,
            gam_solve::estimate::Dispersion::Estimated(1.0 / shape),
            false,
        )
        .expect("GammaLog NUTS target builds");

        // z-space precision at the mode (z = 0) via central differences of the
        // analytic gradient: `−∂(∇_z logp)/∂z = Lᵀ Hβ L`. Correct value: I.
        let eps = 1e-6;
        let z0 = Array1::<f64>::zeros(p);
        let mut hz = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let mut zp = z0.clone();
            let mut zm = z0.clone();
            zp[j] += eps;
            zm[j] -= eps;
            let (_, gp) = target.compute_logp_and_grad_nd(&zp);
            let (_, gm) = target.compute_logp_and_grad_nd(&zm);
            for a in 0..p {
                hz[[a, j]] = -(gp[a] - gm[a]) / (2.0 * eps);
            }
        }

        for a in 0..p {
            for b in 0..p {
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (hz[[a, b]] - expected).abs() < 1e-4,
                    "z-curvature[{a},{b}] = {} (expected {expected}); a non-identity \
                     value means the GammaLog target re-introduced the #680 dispersion \
                     double-count (penalty ×ν and/or whitening ×√φ)",
                    hz[[a, b]]
                );
            }
        }
        // Trace = p (identity) rejects the φ-scaled `φ·tr(...)` signature.
        let trace: f64 = (0..p).map(|i| hz[[i, i]]).sum();
        assert!(
            (trace - p as f64).abs() < 1e-3,
            "z-curvature trace {trace} ≠ {p}: dispersion double-count signature"
        );
    }

    /// Regression for #680 (whitening half, isolated): for a weight-carries-
    /// dispersion family the whitening must satisfy `L Lᵀ = H⁻¹` — i.e.
    /// `cov_scale = 1` — so the sampler whitens against the same `H⁻¹` it
    /// targets. The pre-fix Gamma path scaled `L` by `√φ`, giving
    /// `L Lᵀ = φ·H⁻¹` and `chol·cholᵀ·H = φ·I ≠ I`.
    #[test]
    fn gamma_log_nuts_whitening_targets_unscaled_inverse_hessian_issue_680() {
        let x = array![[1.0, -0.4], [1.0, 0.6], [1.0, 0.1], [1.0, 1.3]];
        let mode = array![0.2_f64, 0.3_f64];
        let y = array![0.8_f64, 1.7, 1.1, 2.2];
        let weights = array![1.0_f64, 1.0, 1.5, 0.7];
        let shape = 6.25_f64; // φ = 0.16
        let p = x.ncols();
        let s = array![[0.3_f64, 0.0], [0.0, 0.7]];
        let hessian = &gamma_log_observed_information(&x, &mode, &y, &weights, shape) + &s;

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            s.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::GammaLog,
            shape,
            gam_solve::estimate::Dispersion::Estimated(1.0 / shape),
            false,
        )
        .expect("GammaLog NUTS target builds");

        // chol = L with L Lᵀ = H⁻¹  ⇒  (L Lᵀ) H = I.
        let l = target.chol();
        let llt = l.dot(&l.t());
        let prod = llt.dot(&hessian);
        for a in 0..p {
            for b in 0..p {
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (prod[[a, b]] - expected).abs() < 1e-8,
                    "L Lᵀ H[{a},{b}] = {} (expected {expected}); a φ·I result means \
                     the Gamma whitening still scales by √φ (#680)",
                    prod[[a, b]]
                );
            }
        }
    }

    #[test]
    fn firth_jeffreys_logit_is_finite_for_rank_deficient_design() {
        let x = array![
            [1.0, -0.5, 1.0],
            [1.0, 0.3, 1.0],
            [1.0, 0.8, 1.0],
            [1.0, -1.2, 1.0],
        ];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let weights = array![1.0, 2.0, 0.5, 1.5];
        let eta = array![0.2, -0.1, 0.4, -0.3];

        let data = SharedData {
            x: Arc::new(x.clone()),
            y: Arc::new(y),
            weights: Arc::new(weights.clone()),
            mode: Arc::new(Array1::zeros(x.ncols())),
            offset: None,
            gamma_shape: 1.0,
            dispersion: gam_solve::estimate::Dispersion::Known(1.0),
            n_samples: x.nrows(),
            dim: x.ncols(),
        };

        let (value, grad) = firth_jeffreys_logp_and_grad(
            &NutsFamily::BinomialLogit.likelihood_spec(),
            &data,
            &eta,
        )
        .expect("firth");

        assert!(value.is_finite());
        assert_eq!(grad.len(), x.ncols());
        assert!(grad.iter().all(|v| v.is_finite()));

        // The Jeffreys term is link-general: at the same eta the probit
        // Fisher weight differs from logit (2/pi vs 1/4 at eta = 0), so the
        // determinants must differ — a hard-coded logit correction would
        // make these equal (finding 19, #2245).
        let (value_probit, grad_probit) = firth_jeffreys_logp_and_grad(
            &NutsFamily::BinomialProbit.likelihood_spec(),
            &data,
            &eta,
        )
        .expect("probit firth");
        assert!(value_probit.is_finite());
        assert!(
            (value_probit - value).abs() > 1e-6,
            "probit and logit Jeffreys log-determinants must differ: {value_probit} vs {value}"
        );
        assert!(grad_probit.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn logit_pg_gibbs_returns_finite_samples() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let cfg = NutsConfig {
            n_samples: 30,
            nwarmup: 30,
            n_chains: 2,
            target_accept: 0.8,
            seed: 123,
        };
        let out = run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &cfg,
        )
        .expect("pg gibbs should run");
        assert_eq!(out.samples.ncols(), 2);
        assert_eq!(out.samples.nrows(), cfg.n_samples * cfg.n_chains);
        assert!(out.samples.iter().all(|v| v.is_finite()));
        assert!(out.posterior_mean.iter().all(|v| v.is_finite()));
        assert!(out.posterior_std.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn family_pg_dispatch_rejects_non_bernoulli_response() {
        let x = array![[1.0], [1.0]];
        let y = array![2.0, 0.0];
        let w = array![1.0, 1.0];
        let penalty = array![[0.1]];
        let mode = array![0.0];
        let non_spd_hessian = array![[0.0]];
        let cfg = NutsConfig {
            n_samples: 1,
            nwarmup: 1,
            n_chains: 1,
            target_accept: 0.8,
            seed: 321,
        };

        let result = run_nuts_sampling_flattened_family(
            LikelihoodSpec::binomial_logit(),
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: non_spd_hessian.view(),
                gamma_shape: None,
                dispersion: gam_solve::model_types::Dispersion::Known(1.0),
                firth_bias_reduction: false,
                offset: None,
            }),
            &cfg,
        );

        let err = result.err().expect("PG dispatch should reject count rows");
        assert!(
            err.contains("response must be exactly 0 or 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn family_dispatch_uses_pg_gibbs_for_standard_logit() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let non_spdhessian = array![[0.0, 0.0], [0.0, 0.0]];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 456,
        };
        let out = run_nuts_sampling_flattened_family(
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Logit),
            },
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: non_spdhessian.view(),
                gamma_shape: None,
                dispersion: gam_solve::estimate::Dispersion::Known(1.0),
                firth_bias_reduction: false,
                offset: None,
            }),
            &cfg,
        )
        .expect("dispatch should use PG Gibbs and not require Hessian factorization");
        assert_eq!(out.samples.nrows(), cfg.n_samples * cfg.n_chains);
        assert!(out.samples.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn family_dispatch_routes_probit_to_nuts_path() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let non_spdhessian = array![[0.0, 0.0], [0.0, 0.0]];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 654,
        };

        let err = match run_nuts_sampling_flattened_family(
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Probit),
            },
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: non_spdhessian.view(),
                gamma_shape: None,
                dispersion: gam_solve::estimate::Dispersion::Known(1.0),
                firth_bias_reduction: false,
                offset: None,
            }),
            &cfg,
        ) {
            Ok(_) => panic!("non-SPD Hessian should fail after probit routes to the NUTS path"),
            Err(err) => err,
        };

        assert!(
            err.contains("Hessian Cholesky decomposition failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn family_dispatch_rejects_nonbinomial_firth_family() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 2.0, 0.0, 3.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let hessian = array![[1.5, 0.1], [0.1, 1.2]];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 111,
        };

        let err = match run_nuts_sampling_flattened_family(
            LikelihoodSpec {
                response: ResponseFamily::Poisson,
                link: InverseLink::Standard(StandardLink::Log),
            },
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: hessian.view(),
                gamma_shape: None,
                dispersion: gam_solve::estimate::Dispersion::Known(1.0),
                firth_bias_reduction: true,
                offset: None,
            }),
            &cfg,
        ) {
            Ok(_) => panic!("Poisson Firth should be rejected explicitly"),
            Err(err) => err,
        };

        assert!(
            err.contains(
                "NUTS with Firth requires a Binomial inverse link with a Fisher-weight jet"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn run_nuts_sampling_rejects_invalid_target_accept() {
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![0.5, -0.5, 1.0];
        let weights = array![1.0, 1.0, 1.0];
        let penalty = array![[0.25]];
        let mode = array![0.0];
        let hessian = array![[1.25]];
        let cfg = NutsConfig {
            n_samples: 10,
            nwarmup: 10,
            n_chains: 1,
            target_accept: 1.0,
            seed: 222,
        };

        let err = super::run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::Gaussian,
            1.0,
            gam_solve::estimate::Dispersion::Known(1.0),
            false,
            None,
            &cfg,
        )
        .expect_err("invalid target_accept should be rejected before sampling");

        assert!(
            err.contains("target_accept must be finite and lie in (0, 1)"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn run_nuts_sampling_rejects_zero_or_too_few_samples() {
        // Issue #399: `samples=0` (and `samples` in {1, 2, 3}) reached the
        // engine and panicked across the FFI boundary in `general-mcmc`'s
        // `.expect(...)` (empty stack / "split R-hat and ESS require at least 2
        // split chains and 2 draws per split chain"). The up-front guard must
        // reject anything below the split-R-hat-defined minimum of 4 draws with
        // a clean typed error *before* the sampler is constructed.
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![0.5, -0.5, 1.0];
        let weights = array![1.0, 1.0, 1.0];
        let penalty = array![[0.25]];
        let mode = array![0.0];
        let hessian = array![[1.25]];

        for bad_samples in [0usize, 1, 2, 3] {
            let cfg = NutsConfig {
                n_samples: bad_samples,
                nwarmup: 10,
                n_chains: 2,
                target_accept: 0.8,
                seed: 222,
            };

            let err = super::run_nuts_sampling(
                x.view(),
                y.view(),
                weights.view(),
                penalty.view(),
                mode.view(),
                hessian.view(),
                NutsFamily::Gaussian,
                1.0,
                gam_solve::estimate::Dispersion::Known(1.0),
                false,
                None,
                &cfg,
            )
            .expect_err("too-few samples must be rejected before sampling");

            assert!(
                err.contains("n_samples must be >= 4"),
                "n_samples={bad_samples} gave unexpected error: {err}"
            );
        }
    }

    #[test]
    fn polya_gamma_gibbs_rejects_degenerate_counts_but_accepts_single_chain() {
        // Issue #399 (missed path): the canonical unit-weight Bernoulli-logit
        // GAM auto-selects the hand-rolled Pólya-Gamma Gibbs sampler, NOT the
        // general-mcmc NUTS engine. Pre-fix that path never validated
        // n_samples/n_chains, so `chains=0` / `samples=0` silently returned a
        // degenerate empty `(0, p)` posterior instead of the typed error the
        // NUTS path raised — a divergent contract on one public API. Assert PG
        // now rejects the degenerate counts up front, and (mirroring NUTS)
        // still accepts a single chain.
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let weights = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.25]];
        let mode = array![0.0];

        let zero_chain_cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 10,
            n_chains: 0,
            target_accept: 0.8,
            seed: 7,
        };
        let err = super::run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            &zero_chain_cfg,
        )
        .expect_err("PG Gibbs must reject zero chains up front, not return an empty posterior");
        assert!(
            err.contains("n_chains must be >= 1"),
            "PG n_chains=0 gave unexpected error: {err}"
        );

        let zero_sample_cfg = NutsConfig {
            n_samples: 0,
            nwarmup: 10,
            n_chains: 2,
            target_accept: 0.8,
            seed: 7,
        };
        let err = super::run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            &zero_sample_cfg,
        )
        .expect_err("PG Gibbs must reject zero samples up front, not return an empty posterior");
        assert!(
            err.contains("n_samples must be >= 4"),
            "PG n_samples=0 gave unexpected error: {err}"
        );

        let single_chain_cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 10,
            n_chains: 1,
            target_accept: 0.8,
            seed: 7,
        };
        let result = super::run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            &single_chain_cfg,
        )
        .expect("PG Gibbs must accept a single chain and return draws");
        assert_eq!(
            result.samples.nrows(),
            20,
            "single-chain PG run should return all 20 requested draws"
        );
    }

    #[test]
    fn run_nuts_sampling_rejects_zero_chains_but_accepts_single_chain() {
        // Issue #399: only `chains=0` is degenerate — it produces an empty
        // initial-position vector and panics in `ndarray::stack`, so it must be
        // rejected up front with a typed error.
        //
        // A *single* chain, by contrast, is a supported, tested configuration
        // (`tests/test_sample_seed_is_reproducible.py`,
        // `tests/test_posterior_save_no_extension_roundtrip.py`,
        // `tests/test_penalty_sampling_survival_diagnostics_regressions.py` all
        // sample with `chains=1`): the engine splits each chain in half, so one
        // chain still yields the two split-chains the R-hat path needs, and
        // `compute_split_rhat_and_ess` early-returns gracefully for
        // `n_chains < 2`. The original #399 fix wrongly raised the floor to 2
        // and regressed those tests; this asserts `chains=1` *returns draws*.
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![0.5, -0.5, 1.0];
        let weights = array![1.0, 1.0, 1.0];
        let penalty = array![[0.25]];
        let mode = array![0.0];
        let hessian = array![[1.25]];

        let zero_chain_cfg = NutsConfig {
            n_samples: 50,
            nwarmup: 10,
            n_chains: 0,
            target_accept: 0.8,
            seed: 222,
        };
        let err = super::run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::Gaussian,
            1.0,
            gam_solve::estimate::Dispersion::Known(1.0),
            false,
            None,
            &zero_chain_cfg,
        )
        .expect_err("zero chains must be rejected before sampling");
        assert!(
            err.contains("n_chains must be >= 1"),
            "n_chains=0 gave unexpected error: {err}"
        );

        let single_chain_cfg = NutsConfig {
            n_samples: 50,
            nwarmup: 10,
            n_chains: 1,
            target_accept: 0.8,
            seed: 222,
        };
        let result = super::run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::Gaussian,
            1.0,
            gam_solve::estimate::Dispersion::Known(1.0),
            false,
            None,
            &single_chain_cfg,
        )
        .expect("a single chain is a supported configuration and must return draws");
        assert_eq!(
            result.samples.nrows(),
            50,
            "single-chain run should return all 50 requested draws"
        );
    }

    #[test]
    fn joint_hmc_boundary_rejects_nonbinomial_firth_family() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 2.0, 0.0, 3.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let hessian = array![[1.5, 0.1], [0.1, 1.2]];
        let penalty_root = array![[0.4, 0.0], [0.0, 0.6]];
        let mode = array![0.0, 0.0];
        let rho_mode = array![0.0];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 111,
        };

        let inputs = JointBetaRhoInputs {
            x: x.view(),
            y: y.view(),
            weights: w.view(),
            likelihood: LikelihoodSpec {
                response: ResponseFamily::Poisson,
                link: InverseLink::Standard(StandardLink::Log),
            },
            gamma_shape: None,
            mode: mode.view(),
            hessian: hessian.view(),
            penalty_roots: vec![CanonicalPenalty::from_dense_root(
                penalty_root.clone(),
                penalty_root.ncols(),
            )],
            rho_mode: rho_mode.view(),
            rho_prior: RhoPrior::default(),
            firth_bias_reduction: true,
            trigger_skewness: 0.75,
        };

        let err = match run_joint_beta_rho_sampling(&inputs, &cfg) {
            Ok(_) => panic!("Poisson joint HMC Firth should be rejected explicitly"),
            Err(err) => err,
        };

        assert!(
            err.contains(
                "Joint HMC with Firth requires a Binomial inverse link with a Fisher-weight jet"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn joint_hmc_uses_combined_penalty_logdet_for_overlapping_penalties() {
        let x = array![[0.0, 0.0]];
        let y = array![0.0];
        let w = array![0.0];
        let mode = array![0.0, 0.0];
        let hessian = array![[1.0, 0.0], [0.0, 1.0]];
        let rho_mode = array![0.0, 0.0];
        let penalty_1 = array![[1.0, 0.0], [0.0, 1.0]];
        let penalty_2 = array![[2.0_f64.sqrt(), 0.0], [0.0, 1.0]];
        let target = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![
                CanonicalPenalty::from_dense_root(penalty_1, 2),
                CanonicalPenalty::from_dense_root(penalty_2, 2),
            ],
            rho_mode.view(),
            LikelihoodSpec {
                response: ResponseFamily::Gaussian,
                link: InverseLink::Standard(StandardLink::Identity),
            },
            None,
            RhoPrior::Flat,
            false,
        )
        .expect("joint target");

        let params = array![0.0, 0.0, 0.0, 0.0];
        let (_, grad) = target.compute_joint_logp_and_grad(&params);
        assert!(
            (grad[2] - 5.0 / 12.0).abs() < 1.0e-10,
            "expected overlapping-penalty gradient 5/12, got {}",
            grad[2]
        );
        assert!(
            (grad[3] - 7.0 / 12.0).abs() < 1.0e-10,
            "expected overlapping-penalty gradient 7/12, got {}",
            grad[3]
        );
    }

    #[test]
    fn joint_hmc_target_does_not_depend_on_rho_mode_when_prior_is_fixed() {
        let x = array![[0.0]];
        let y = array![0.0];
        let w = array![0.0];
        let mode = array![0.0];
        let hessian = array![[1.0]];
        let penalty = CanonicalPenalty::from_dense_root(array![[1.0]], 1);
        let prior = RhoPrior::Normal {
            mean: 0.25,
            sd: 1.7,
        };

        let target_a = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![penalty.clone()],
            array![0.0].view(),
            LikelihoodSpec {
                response: ResponseFamily::Gaussian,
                link: InverseLink::Standard(StandardLink::Identity),
            },
            None,
            prior.clone(),
            false,
        )
        .expect("target a");
        let target_b = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![penalty],
            array![2.5].view(),
            LikelihoodSpec {
                response: ResponseFamily::Gaussian,
                link: InverseLink::Standard(StandardLink::Identity),
            },
            None,
            prior,
            false,
        )
        .expect("target b");

        let params = array![0.0, -0.4];
        let (lp_a, grad_a) = target_a.compute_joint_logp_and_grad(&params);
        let (lp_b, grad_b) = target_b.compute_joint_logp_and_grad(&params);
        assert!((lp_a - lp_b).abs() < 1.0e-12);
        for i in 0..grad_a.len() {
            assert!(
                (grad_a[i] - grad_b[i]).abs() < 1.0e-12,
                "rho_mode leaked into target gradient at {}: {} vs {}",
                i,
                grad_a[i],
                grad_b[i]
            );
        }
    }

    #[test]
    fn joint_hmc_binomial_sas_uses_runtime_link_state() {
        let x = array![[1.0], [1.0]];
        let y = array![1.0, 0.0];
        let weights = array![1.0, 1.0];
        let eta = array![0.3, -0.2];
        let sas_state =
            gam_solve::mixture_link::state_from_sasspec(gam_problem::types::SasLinkSpec {
                initial_epsilon: 0.4,
                initial_log_delta: -0.2,
            })
            .expect("sas state");
        let data = SharedData {
            x: Arc::new(x),
            y: Arc::new(y),
            weights: Arc::new(weights),
            mode: Arc::new(Array1::zeros(1)),
            offset: None,
            gamma_shape: 1.0,
            dispersion: gam_solve::estimate::Dispersion::Known(1.0),
            n_samples: 2,
            dim: 1,
        };

        let (ll_sas, _) = joint_family_logp_and_grad(
            &LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Sas(sas_state),
            },
            &data,
            &eta,
        )
        .expect("sas joint logp");
        let (ll_logit, _) = joint_family_logp_and_grad(
            &LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Logit),
            },
            &data,
            &eta,
        )
        .expect("logit joint logp");

        assert!(
            (ll_sas - ll_logit).abs() > 1.0e-6,
            "adaptive SAS link should not collapse to the logit likelihood"
        );
    }

    #[test]
    fn directional_cubic_diagnostic_is_rotation_invariant_for_hessian_eigenvectors() {
        let x = array![[1.0, 0.5], [-0.3, 1.4], [0.8, -1.1]];
        let c = array![0.7, -0.5, 0.2];
        let h = array![[4.0, 0.0], [0.0, 1.0]];
        let theta = std::f64::consts::FRAC_PI_4;
        let q = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()],];
        let x_rot = x.dot(&q);
        let h_rot = q.t().dot(&h).dot(&q);

        let (base_max, base_vals) = laplace_directional_cubic_diagnostic(
            &h,
            &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x)),
            &c,
            true,
        )
        .expect("base diagnostic");
        let (rot_max, rot_vals) = laplace_directional_cubic_diagnostic(
            &h_rot,
            &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_rot)),
            &c,
            true,
        )
        .expect("rotated diagnostic");

        let mut base_abs: Vec<f64> = base_vals.iter().map(|v| v.abs()).collect();
        let mut rot_abs: Vec<f64> = rot_vals.iter().map(|v| v.abs()).collect();
        base_abs.sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));
        rot_abs.sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));

        assert!((base_max - rot_max).abs() < 1.0e-10);
        for i in 0..base_abs.len() {
            assert!(
                (base_abs[i] - rot_abs[i]).abs() < 1.0e-10,
                "directional diagnostic changed under rotation at {}: {} vs {}",
                i,
                base_abs[i],
                rot_abs[i]
            );
        }
    }

    /// Verify that joint HMC and REML compute identical penalty logdet
    /// derivatives for the same penalty system. This catches any divergence
    /// between the two code paths.
    #[test]
    fn joint_hmc_penalty_logdet_agrees_with_reml_path() {
        use gam_solve::estimate::reml::penalty_logdet::PenaltyPseudologdet;

        // Two overlapping 3x3 penalties with non-trivial lambdas.
        let root_1 = array![[1.0, 0.5, 0.0], [0.0, 0.8, 0.3]];
        let root_2 = array![[0.0, 0.7, 0.0], [0.0, 0.0, 1.2]];
        let cp1 = CanonicalPenalty::from_dense_root(root_1, 3);
        let cp2 = CanonicalPenalty::from_dense_root(root_2, 3);
        let lambdas = [2.5_f64, 0.8];
        let penalties = [cp1.clone(), cp2.clone()];

        // REML path: PenaltyPseudologdet directly.
        let pld =
            PenaltyPseudologdet::from_penalties(&penalties, &lambdas, 0.0, 3).expect("reml pld");
        let reml_value = pld.value();
        let (reml_d1, reml_d2) = pld.rho_derivatives_from_penalties(&penalties, &lambdas);

        // Joint HMC path: build a JointBetaRhoPosterior and extract the
        // penalty logdet contribution. We isolate it by using zero data
        // (so likelihood = 0, penalty quadratic = 0) and Flat rho prior.
        let x = Array2::<f64>::zeros((1, 3));
        let y = array![0.0];
        let w = array![0.0];
        let mode = Array1::<f64>::zeros(3);
        let hessian = Array2::<f64>::eye(3);
        let rho = Array1::from_vec(lambdas.iter().map(|l| l.ln()).collect());
        let target = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![cp1, cp2],
            rho.view(),
            LikelihoodSpec {
                response: ResponseFamily::Gaussian,
                link: InverseLink::Standard(StandardLink::Identity),
            },
            None,
            RhoPrior::Flat,
            false,
        )
        .expect("joint target");

        // Evaluate at beta=0, rho=ln(lambdas).
        let mut params = Array1::<f64>::zeros(3 + 2);
        params[3] = rho[0];
        params[4] = rho[1];
        let (logp, grad) = target.compute_joint_logp_and_grad(&params);

        // logp should be 0.5 * reml_value (likelihood=0, prior=0, quadratic=0).
        assert!(
            (logp - 0.5 * reml_value).abs() < 1.0e-8,
            "joint HMC logdet value {} vs REML 0.5*{} = {}",
            logp,
            reml_value,
            0.5 * reml_value,
        );

        // grad[3..5] should be 0.5 * reml_d1.
        for k in 0..2 {
            assert!(
                (grad[3 + k] - 0.5 * reml_d1[k]).abs() < 1.0e-8,
                "joint HMC logdet gradient[{}] = {} vs REML 0.5*{} = {}",
                k,
                grad[3 + k],
                reml_d1[k],
                0.5 * reml_d1[k],
            );
        }

        // Sanity: second derivatives are available from REML but not directly
        // from a single HMC gradient call; just verify they're symmetric.
        assert!(
            (reml_d2[[0, 1]] - reml_d2[[1, 0]]).abs() < 1.0e-12,
            "REML penalty logdet Hessian not symmetric"
        );
    }

    /// Verify the family-gating invariant: every LikelihoodSpec that
    /// joint_family_logp_and_grad accepts produces a result (not an error
    /// about missing implementation). Every family it rejects returns an
    /// explicit error. No family is silently remapped to a different one.
    #[test]
    fn joint_hmc_family_gating_never_remaps() {
        let data = SharedData {
            x: Arc::new(array![[1.0], [1.0]]),
            y: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 1.0]),
            mode: Arc::new(Array1::zeros(1)),
            offset: None,
            gamma_shape: 1.0,
            dispersion: gam_solve::estimate::Dispersion::Known(1.0),
            n_samples: 2,
            dim: 1,
        };
        let eta = array![0.1, -0.1];

        // These families must succeed with their own inverse link.
        let accepted = [
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Logit),
            },
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::Probit),
            },
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Standard(StandardLink::CLogLog),
            },
            LikelihoodSpec {
                response: ResponseFamily::Gaussian,
                link: InverseLink::Standard(StandardLink::Identity),
            },
            LikelihoodSpec {
                response: ResponseFamily::Poisson,
                link: InverseLink::Standard(StandardLink::Log),
            },
            LikelihoodSpec {
                response: ResponseFamily::Gamma,
                link: InverseLink::Standard(StandardLink::Log),
            },
        ];
        for spec in &accepted {
            let result = joint_family_logp_and_grad(spec, &data, &eta);
            assert!(
                result.is_ok(),
                "spec {:?} should be accepted but got error: {:?}",
                spec,
                result.err(),
            );
        }

        // SAS/BetaLogistic/Mixture must succeed with their real link state,
        // NOT be remapped to logit.
        let sas_state =
            gam_solve::mixture_link::state_from_sasspec(gam_problem::types::SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            })
            .expect("sas state");
        let adaptive = [
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::Sas(sas_state),
            },
            LikelihoodSpec {
                response: ResponseFamily::Binomial,
                link: InverseLink::BetaLogistic(
                    gam_solve::mixture_link::state_from_sasspec(gam_problem::types::SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .expect("bl state"),
                ),
            },
        ];
        for spec in &adaptive {
            let result = joint_family_logp_and_grad(spec, &data, &eta);
            assert!(
                result.is_ok(),
                "adaptive spec {:?} should be accepted with its real link",
                spec,
            );
        }

        // RoystonParmar must be explicitly rejected (not silently remapped).
        let rp_result = joint_family_logp_and_grad(
            &LikelihoodSpec {
                response: ResponseFamily::RoystonParmar,
                link: InverseLink::Standard(StandardLink::Logit),
            },
            &data,
            &eta,
        );
        assert!(
            rp_result.is_err(),
            "RoystonParmar should be rejected, not silently accepted"
        );
    }

    /// The power-iteration refinement should find non-Gaussianity at least
    /// as large as the eigenvector-only pass (it's a supremum search).
    #[test]
    fn directional_cubic_power_iteration_finds_larger_or_equal_skewness() {
        // Construct a design where the maximum |gamma| occurs off-axis.
        // A single row with asymmetric structure makes the cubic form
        // peak between eigenvectors.
        let x = array![
            [2.0, 1.0],
            [-1.0, 2.0],
            [0.5, -0.5],
            [1.5, 0.3],
            [-0.8, 1.7],
        ];
        let c = array![1.0, -0.5, 0.3, -0.7, 0.4];
        let h = array![[3.0, 1.0], [1.0, 2.0]];

        let (max_val, eigenvector_vals) = laplace_directional_cubic_diagnostic(
            &h,
            &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x)),
            &c,
            true,
        )
        .expect("diagnostic");

        // max_val should be >= max of eigenvector-only values.
        let eig_max = eigenvector_vals
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_val >= eig_max - 1.0e-12,
            "power iteration result {} should be >= eigenvector max {}",
            max_val,
            eig_max,
        );
    }

    #[test]
    fn laplace_trustworthiness_is_block_local_and_threshold_shrinks_with_n() {
        // Two directions: one nearly Gaussian (tiny skewness), one strongly
        // skewed. The adaptive verdict must flag ONLY the skewed direction —
        // this is the block-local behavior #784 requires (keep cheap Laplace
        // where the Gaussian summary holds, correct only the curvature-heavy
        // block).
        let skew = array![0.01, 0.9];

        // At a modest effective sample size the skewed direction dominates the
        // Laplace floor and must be flagged; the near-Gaussian one must not.
        let verdict = laplace_trustworthiness_from_skewness(&skew, 100.0);
        assert_eq!(
            verdict.untrustworthy_directions,
            vec![1],
            "only the strongly-skewed direction should be flagged (block-local)",
        );
        assert!(verdict.fallback_required());
        assert!((verdict.max_abs_skewness - 0.9).abs() < 1e-12);

        // The threshold must SHRINK as n grows (Laplace gets stricter): a
        // direction tolerated at small n becomes untrustworthy at large n,
        // because the Gaussian floor it must beat is O(1/n).
        let t_small = laplace_skewness_threshold(25.0);
        let t_large = laplace_skewness_threshold(10_000.0);
        assert!(
            t_large < t_small,
            "validity threshold must tighten with sample size: {t_large} !< {t_small}",
        );

        // Degenerate / empty curvature support => everything trustworthy
        // (nothing for the Gaussian summary to be wrong about).
        let none = laplace_trustworthiness_from_skewness(&skew, 0.0);
        assert!(!none.fallback_required());
        assert!(none.threshold.is_infinite());
    }

    /// Synthetic block-excess oracle: an anharmonicity `ΔF(t) = a·Σ_k t_k⁴`
    /// whose per-direction strength carries unit ρ-sensitivity, so
    /// `∂ΔF/∂ρ_k = a·t_k⁴`. `a = 0` is a pure Gaussian block (exactly zero
    /// excess and zero ρ-gradient — the consistency anchor); `a > 0` is the
    /// quartic correction oracle the importance sampler is checked against.
    struct AnharmonicBlock {
        lambdas: Array1<f64>,
        a: f64,
    }
    impl super::BlockExcessTarget for AnharmonicBlock {
        fn block_dim(&self) -> usize {
            self.lambdas.len()
        }
        fn rho_dim(&self) -> usize {
            self.lambdas.len()
        }
        fn block_curvatures(&self) -> &Array1<f64> {
            &self.lambdas
        }
        fn excess(&self, t: &Array1<f64>) -> f64 {
            self.a * t.iter().map(|&x| x.powi(4)).sum::<f64>()
        }
        fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64> {
            t.mapv(|x| self.a * x.powi(4))
        }
        fn displaced_neg_score(&self, t: &Array1<f64>) -> Array1<f64> {
            // The synthetic oracle has no observation rows: its ΔF carries no
            // deviance channel, so the per-row score moment is empty and the
            // (b)–(d) channel assembly contracts against nothing.
            assert_eq!(t.len(), self.block_dim(), "displacement dim mismatch");
            Array1::zeros(0)
        }
        fn base_neg_score(&self) -> Array1<f64> {
            Array1::zeros(0)
        }
    }

    #[test]
    fn block_sampled_marginal_is_zero_for_gaussian_block() {
        // A purely Gaussian block has ΔF ≡ 0, so the sampled correction (the
        // log-ratio of true to Laplace block free energy) must be exactly 0,
        // with a zero ρ-gradient. This is the consistency anchor: where the
        // Gaussian summary holds, the fallback is a no-op.
        let target = AnharmonicBlock {
            lambdas: array![2.0, 0.5],
            a: 0.0,
        };
        let out = super::block_sampled_marginal_correction(&target).expect("correction");
        assert!(
            out.value.abs() < 1e-12,
            "Gaussian block value {}",
            out.value
        );
        assert!(out.rho_gradient.iter().all(|&g| g.abs() < 1e-12));
        assert!(out.n_draws > 0);
    }

    #[test]
    fn block_sampled_marginal_recovers_analytic_quartic_correction() {
        // 1-D block with a quartic excess ΔF(t) = a t⁴ (a small positive
        // anharmonicity). Then exp(Δ_b) = E_{t~N(0,1/λ)}[exp(−a t⁴)], a known
        // 1-D integral the IS estimator must recover. We check the sampled Δ_b
        // matches a high-accuracy deterministic quadrature of the same
        // expectation, and that Δ_b < 0 (an added quartic penalty makes the
        // true block mass *smaller* than the Gaussian's).
        let lambda = 3.0_f64;
        let a = 0.05_f64;
        let target = AnharmonicBlock {
            lambdas: array![lambda],
            a,
        };
        let out = super::block_sampled_marginal_correction(&target).expect("correction");

        // Deterministic reference: Δ_b = log E_{t~N(0,1/λ)}[exp(−a t⁴)] via a
        // fine trapezoid rule over the Gaussian density.
        let sigma = (1.0 / lambda).sqrt();
        let steps = 20_001;
        let lo = -8.0 * sigma;
        let hi = 8.0 * sigma;
        let h = (hi - lo) / (steps as f64 - 1.0);
        let mut integral = 0.0_f64;
        for i in 0..steps {
            let tt = lo + h * i as f64;
            let gauss = (-(tt * tt) / (2.0 * sigma * sigma)).exp()
                / (sigma * (2.0 * std::f64::consts::PI).sqrt());
            let w = if i == 0 || i == steps - 1 { 0.5 } else { 1.0 };
            integral += w * gauss * (-a * tt.powi(4)).exp() * h;
        }
        let reference = integral.ln();
        assert!(
            (out.value - reference).abs() < 5e-3,
            "sampled Δ_b {} vs reference {}",
            out.value,
            reference,
        );
        assert!(out.value < 0.0, "quartic penalty must shrink block mass");
    }

    /// A block target whose excess and per-row score are driven by real design
    /// matvecs `s = X·(V_b·t)` — the SAME structure as the production
    /// `Gam784BlockTarget` — so it can compute those matvecs either serially
    /// (one `fast_av` per draw) or batched (one GEMM over all draws), toggled by
    /// `batched`. The two must yield a bit-for-bit (to FP-reassociation
    /// tolerance) identical correction: that is exactly the #1082 batching
    /// contract — GEMM changes HOW the matvec is computed, never WHAT.
    struct MatvecBlock {
        lambdas: Array1<f64>,
        x: Array2<f64>,
        v_b: Array2<f64>,
        y: Array1<f64>,
        batched: bool,
    }
    impl MatvecBlock {
        fn s_of(&self, t: &Array1<f64>) -> Array1<f64> {
            let delta = self.v_b.dot(t);
            gam_linalg::faer_ndarray::fast_av(&self.x, &delta)
        }
        // A smooth, finite, family-like excess + per-row score built from `s`.
        fn excess_and_ngs(&self, s: &Array1<f64>) -> (f64, Array1<f64>) {
            let mut excess = 0.0;
            let mut ngs = Array1::<f64>::zeros(s.len());
            for i in 0..s.len() {
                let mu = (self.y[i] + s[i]).tanh();
                excess += 0.5 * s[i] * s[i] - 0.1 * mu;
                ngs[i] = mu - self.y[i];
            }
            (excess, ngs)
        }
    }
    impl super::BlockExcessTarget for MatvecBlock {
        fn block_dim(&self) -> usize {
            self.lambdas.len()
        }
        fn rho_dim(&self) -> usize {
            self.lambdas.len()
        }
        fn block_curvatures(&self) -> &Array1<f64> {
            &self.lambdas
        }
        fn excess(&self, t: &Array1<f64>) -> f64 {
            self.excess_and_ngs(&self.s_of(t)).0
        }
        fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64> {
            t.mapv(|x| 0.01 * x)
        }
        fn displaced_neg_score(&self, t: &Array1<f64>) -> Array1<f64> {
            self.excess_and_ngs(&self.s_of(t)).1
        }
        fn base_neg_score(&self) -> Array1<f64> {
            self.excess_and_ngs(&self.s_of(&Array1::zeros(self.block_dim())))
                .1
        }
        fn excess_with_displaced_neg_score_batch(
            &self,
            draws: &Array2<f64>,
        ) -> Vec<(f64, Option<Array1<f64>>)> {
            if !self.batched {
                // Serial reference: per-column, exactly the default path.
                let mut out = Vec::with_capacity(draws.ncols());
                let mut t = Array1::<f64>::zeros(draws.nrows());
                for s in 0..draws.ncols() {
                    t.assign(&draws.column(s));
                    out.push(self.excess_with_displaced_neg_score(&t));
                }
                return out;
            }
            // Batched: Δ = V_b·T then S = X·Δ as two GEMMs, then per-column.
            let delta_all = gam_linalg::faer_ndarray::fast_ab(&self.v_b, draws);
            let s_all = gam_linalg::faer_ndarray::fast_ab(&self.x, &delta_all);
            (0..draws.ncols())
                .map(|c| {
                    let (e, ngs) = self.excess_and_ngs(&s_all.column(c).to_owned());
                    if e.is_finite() {
                        (e, Some(ngs))
                    } else {
                        (e, None)
                    }
                })
                .collect()
        }
    }

    #[test]
    fn block_sampled_marginal_batched_matches_serial_matvec() {
        // Real design / block-frame matvecs, large enough that the GEMM path is
        // actually taken (n, p ≥ faer threshold). The batched override must give
        // the same correction value, ρ-gradient, and moments as the serial path.
        let n = 80usize;
        let p = 40usize;
        let m = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[(i, j)] = ((i * 7 + j * 13) % 11) as f64 * 0.05 - 0.25;
            }
        }
        let mut v_b = Array2::<f64>::zeros((p, m));
        for i in 0..p {
            for r in 0..m {
                v_b[(i, r)] = ((i * 3 + r * 5) % 7) as f64 * 0.1 - 0.3;
            }
        }
        let y: Array1<f64> = (0..n).map(|i| ((i % 5) as f64) * 0.2).collect();
        let lambdas = array![2.0, 1.0, 0.5];

        let serial = super::block_sampled_marginal_correction(&MatvecBlock {
            lambdas: lambdas.clone(),
            x: x.clone(),
            v_b: v_b.clone(),
            y: y.clone(),
            batched: false,
        })
        .expect("serial");
        let batched = super::block_sampled_marginal_correction(&MatvecBlock {
            lambdas,
            x,
            v_b,
            y,
            batched: true,
        })
        .expect("batched");

        assert_eq!(serial.n_draws, batched.n_draws);
        assert!(
            (serial.value - batched.value).abs() <= 1e-10 * (1.0 + serial.value.abs()),
            "value serial {} vs batched {}",
            serial.value,
            batched.value
        );
        for k in 0..serial.rho_gradient.len() {
            assert!(
                (serial.rho_gradient[k] - batched.rho_gradient[k]).abs()
                    <= 1e-10 * (1.0 + serial.rho_gradient[k].abs()),
                "rho_gradient[{k}] serial {} vs batched {}",
                serial.rho_gradient[k],
                batched.rho_gradient[k]
            );
        }
        let ms = serial.moments.expect("serial moments");
        let mb = batched.moments.expect("batched moments");
        for (a, b) in ms.e_t.iter().zip(mb.e_t.iter()) {
            assert!((a - b).abs() <= 1e-10 * (1.0 + a.abs()), "e_t {a} vs {b}");
        }
        for (a, b) in ms.e_neg_score.iter().zip(mb.e_neg_score.iter()) {
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + a.abs()),
                "e_neg_score {a} vs {b}"
            );
        }
        for (a, b) in ms.e_t_neg_score.iter().zip(mb.e_t_neg_score.iter()) {
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + a.abs()),
                "e_t_neg_score {a} vs {b}"
            );
        }
    }

    #[test]
    fn logit_pg_rao_blackwell_returns_finite_terms() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let roots = vec![array![[0.2_f64.sqrt(), 0.0], [0.0, 0.4_f64.sqrt()]]];
        let cfg = NutsConfig {
            n_samples: 30,
            nwarmup: 30,
            n_chains: 2,
            target_accept: 0.8,
            seed: 789,
        };

        let rb = super::estimate_logit_pg_rao_blackwell_terms(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &roots,
            &cfg,
        )
        .expect("rao-blackwell PG should run");

        assert_eq!(rb.len(), 1);
        assert!(rb[0].is_finite());
        assert!(rb[0] >= 0.0);
    }

    #[test]
    fn logit_pg_rao_blackwell_rejects_non_bernoulli_response() {
        let x = array![[1.0], [1.0]];
        let y = array![0.25, 1.0];
        let w = array![1.0, 1.0];
        let penalty = array![[0.1]];
        let mode = array![0.0];
        let roots = vec![array![[0.1_f64.sqrt()]]];
        let cfg = NutsConfig {
            n_samples: 1,
            nwarmup: 1,
            n_chains: 1,
            target_accept: 0.8,
            seed: 654,
        };

        let result = super::estimate_logit_pg_rao_blackwell_terms(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &roots,
            &cfg,
        );

        let err = result
            .err()
            .expect("PG Rao-Blackwell should reject proportion rows");
        assert!(
            err.contains("response must be exactly 0 or 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn logit_pg_rao_blackwell_matches_beta_quadratic_moment_sanity() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let roots = vec![array![[0.2_f64.sqrt(), 0.0], [0.0, 0.4_f64.sqrt()]]];
        let cfg = NutsConfig {
            n_samples: 120,
            nwarmup: 80,
            n_chains: 2,
            target_accept: 0.8,
            seed: 901,
        };

        let gibbs = run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &cfg,
        )
        .expect("pg gibbs should run");
        let mc_quad = gibbs
            .samples
            .rows()
            .into_iter()
            .map(|beta| {
                let sb = penalty.dot(&beta.to_owned());
                beta.dot(&sb)
            })
            .sum::<f64>()
            / (gibbs.samples.nrows() as f64);

        let rb = super::estimate_logit_pg_rao_blackwell_terms(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &roots,
            &cfg,
        )
        .expect("rao-blackwell PG should run");

        let diff = (rb[0] - mc_quad).abs();
        assert!(
            diff < 0.35,
            "Rao-Blackwell vs beta-moment mismatch too large: rb={}, mc={}, diff={}",
            rb[0],
            mc_quad,
            diff
        );
    }

    #[test]
    fn survival_hmc_structural_monotonic_returns_finitevalues() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.2]];
        let x_exit = array![[1.0, 0.6]];
        let x_derivative = array![[0.0, 1.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);

        let posterior = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            penalties,
            monotonicity,
            SurvivalSpec::Net,
            true,
            2,
            mode.view(),
            hessian.view(),
        )
        .expect("construct survival posterior");

        let position = array![0.0, 0.0];
        let mut grad = Array1::<f64>::zeros(2);
        let logp = HamiltonianTarget::logp_and_grad(&posterior, &position, &mut grad);
        assert!(logp.is_finite());
        assert!(grad.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn survival_hmc_structural_monotonic_differs_from_linear_geometry() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.2, 0.1]];
        let x_exit = array![[0.6, 0.3]];
        let x_derivative = array![[1.0, 0.0]];
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);
        let z = array![std::f64::consts::LN_2, 0.0];

        let posterior_linear = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            PenaltyBlocks::new(Vec::new()),
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct linear posterior");
        let mut grad_linear = Array1::<f64>::zeros(2);
        HamiltonianTarget::logp_and_grad(&posterior_linear, &z, &mut grad_linear);

        let posterior_struct = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            PenaltyBlocks::new(Vec::new()),
            monotonicity,
            SurvivalSpec::Net,
            true,
            2,
            mode.view(),
            hessian.view(),
        )
        .expect("construct structural posterior");
        let mut grad_struct = Array1::<f64>::zeros(2);
        HamiltonianTarget::logp_and_grad(&posterior_struct, &z, &mut grad_struct);

        assert!(
            (grad_struct[0] - grad_linear[0]).abs() > 1e-6,
            "expected structural and linear fallback gradients to differ"
        );
        assert!(grad_struct[0].is_finite());
        assert!(grad_linear[0].is_finite());
    }

    #[test]
    fn survival_hmc_fallback_barrier_rejects_offsets_below_monotonicity_threshold() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.0]];
        // Zero derivative design so derivative_offset_exit drives d_eta/dt.
        let x_derivative = array![[0.0, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);
        let z = array![0.0, 0.0];

        let posterior_no_offset = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![0.0].view()),
            penalties.clone(),
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior without derivative offset");
        let mut grad_no_offset = Array1::<f64>::zeros(2);
        let logp_no_offset =
            HamiltonianTarget::logp_and_grad(&posterior_no_offset, &z, &mut grad_no_offset);

        let posteriorwith_offset = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![2.0].view()),
            penalties,
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior with derivative offset");
        let mut gradwith_offset = Array1::<f64>::zeros(2);
        let logpwith_offset =
            HamiltonianTarget::logp_and_grad(&posteriorwith_offset, &z, &mut gradwith_offset);

        assert!(!logp_no_offset.is_finite());
        assert!(!logpwith_offset.is_finite());
        assert!(grad_no_offset.iter().all(|v| *v == 0.0));
        assert!(gradwith_offset.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn survival_hmc_fallback_barrier_becomes_finite_once_offset_clears_guard() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.0]];
        let x_derivative = array![[0.0, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);
        let z = array![0.0, 0.0];

        let posterior_below_guard = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![2.0].view()),
            penalties.clone(),
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior below derivative guard");
        let mut grad_below_guard = Array1::<f64>::zeros(2);
        let logp_below_guard =
            HamiltonianTarget::logp_and_grad(&posterior_below_guard, &z, &mut grad_below_guard);

        let posterior_above_guard = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![3.1].view()),
            penalties,
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior above derivative guard");
        let mut grad_above_guard = Array1::<f64>::zeros(2);
        let logp_above_guard =
            HamiltonianTarget::logp_and_grad(&posterior_above_guard, &z, &mut grad_above_guard);

        assert!(!logp_below_guard.is_finite());
        assert!(logp_above_guard.is_finite());
        assert!(grad_below_guard.iter().all(|v| *v == 0.0));
        assert!(grad_above_guard.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn survival_hmc_structural_monotonic_handles_sparse_multirow_geometry() {
        let age_entry = array![1.0, 1.2];
        let age_exit = array![2.0, 2.4];
        let event_target = array![1u8, 1u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[0.1, 0.0, 0.2], [0.2, 0.1, 0.2]];
        let x_exit = array![[0.4, 0.2, 0.3], [0.6, 0.1, 0.3]];
        // First row constrains only column 0, second row constrains columns 0 and 1.
        let x_derivative = array![[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 3.0 };
        let mode = array![4.0, 2.0, 0.0];
        let hessian = Array2::<f64>::eye(3);
        let z = array![0.05, -0.1, 0.15];

        let posterior = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            PenaltyBlocks::new(Vec::new()),
            monotonicity,
            SurvivalSpec::Net,
            true,
            2,
            mode.view(),
            hessian.view(),
        )
        .expect("construct structural posterior");

        let mut grad = Array1::<f64>::zeros(3);
        let logp = HamiltonianTarget::logp_and_grad(&posterior, &z, &mut grad);
        assert!(logp.is_finite());
        assert!(grad.iter().all(|v| v.is_finite()));
    }
}

/// Implement HamiltonianTarget for NUTS with analytical gradients.
impl HamiltonianTarget<Array1<f64>> for NutsPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        NUTS_RESIDUAL_SCRATCH.with(|scratch| {
            let mut residual = scratch.borrow_mut();
            if residual.len() != self.data.n_samples {
                *residual = Array1::<f64>::zeros(self.data.n_samples);
            }
            self.compute_logp_and_grad_nd_into(position, &mut residual, grad)
        })
    }
}

/// Configuration for NUTS sampling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NutsConfig {
    /// Number of samples to collect (after warmup)
    pub n_samples: usize,
    /// Number of warmup samples to discard
    pub nwarmup: usize,
    /// Number of parallel chains
    pub n_chains: usize,
    /// Target acceptance probability (0.6-0.9 recommended)
    pub target_accept: f64,
    /// Seed for deterministic chain initialization
    #[serde(default = "default_nuts_seed")]
    pub seed: u64,
}

fn default_nuts_seed() -> u64 {
    42
}

fn validate_nuts_target_accept(target_accept: f64) -> Result<(), HmcError> {
    if target_accept.is_finite() && target_accept > 0.0 && target_accept < 1.0 {
        Ok(())
    } else {
        Err(HmcError::InvalidConfig {
            reason: format!(
                "NUTS target_accept must be finite and lie in (0, 1), got {target_accept}"
            ),
        })
    }
}

/// Minimum number of post-warmup draws per chain that keeps the split-R-hat /
/// ESS machinery well-defined. Each chain is split in half for the
/// Gelman-Rubin diagnostic (`compute_split_rhat_and_ess` and the engine's own
/// run-stats path), so both halves need at least two draws, i.e. four draws
/// total. Below this the engine `.expect(...)` calls (empty-stack / "split
/// R-hat and ESS require at least 2 split chains and 2 draws per split chain")
/// panic across the FFI boundary instead of returning a typed error.
const MIN_NUTS_SAMPLES: usize = 4;

/// Minimum number of parallel chains. With zero chains the engine receives an
/// empty initial-position vector and panics in `ndarray::stack` (and the
/// Laplace fallback would produce an empty `(0, p)` posterior). A *single*
/// chain is well-defined and is a supported, tested configuration: the engine
/// splits each chain in half for the diagnostic, so one chain still yields the
/// two split-chains the R-hat path needs, and `compute_split_rhat_and_ess`
/// gracefully early-returns for `n_chains < 2`. We therefore only reject the
/// genuinely-degenerate `n_chains == 0`.
const MIN_NUTS_CHAINS: usize = 1;

/// Validate the draw / chain counts of a NUTS configuration up front, mirroring
/// `validate_nuts_target_accept`, so that out-of-range values surface as a typed
/// `HmcError::InvalidConfig` *before* the sampling engine is constructed rather
/// than as a panic caught at the FFI boundary.
fn validate_nuts_draws(config: &NutsConfig) -> Result<(), HmcError> {
    if config.n_chains < MIN_NUTS_CHAINS {
        return Err(HmcError::InvalidConfig {
            reason: format!(
                "NUTS n_chains must be >= {MIN_NUTS_CHAINS}; with zero chains the \
                 sampler has no initial positions to run, got {}",
                config.n_chains
            ),
        });
    }
    if config.n_samples < MIN_NUTS_SAMPLES {
        return Err(HmcError::InvalidConfig {
            reason: format!(
                "NUTS n_samples must be >= {MIN_NUTS_SAMPLES} so split-R-hat / ESS \
                 diagnostics are defined, got {}",
                config.n_samples
            ),
        });
    }
    Ok(())
}

/// Full up-front validation of a NUTS configuration shared by every sampling
/// entry point (dense NUTS, link-wiggle, joint (β, ρ), survival, the
/// auto-selected Pólya-Gamma Gibbs path, and the Laplace-Gaussian fallback).
pub(crate) fn validate_nuts_config(config: &NutsConfig) -> Result<(), HmcError> {
    validate_nuts_target_accept(config.target_accept)?;
    validate_nuts_draws(config)?;
    Ok(())
}

#[inline]
fn splitmix64(x: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(x)
}

#[inline]
fn chain_stream_seed(seed: u64, chain: usize, stream: u64) -> u64 {
    splitmix64(seed ^ stream ^ ((chain as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)))
}

#[inline]
fn nuts_transition_seed(seed: u64, stream: u64) -> u64 {
    splitmix64(seed ^ stream ^ 0xA24B_AED4_963E_E407)
}

#[inline]
fn gibbs_pg_seed(seed: u64, chain: usize, stream: u64, iter: usize) -> u64 {
    chain_stream_seed(
        seed,
        chain,
        stream ^ ((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
}

fn draw_logit_pg1_omega(
    shapes: ArrayView1<'_, u32>,
    tilts: ArrayView1<'_, f64>,
    seed: u64,
    out: &mut Array1<f64>,
) -> Result<(), String> {
    if out.len() != tilts.len() {
        return Err(HmcError::DimensionMismatch {
            reason: "draw_logit_pg1_omega: output length mismatch".to_string(),
        }
        .into());
    }
    let draws = crate::gpu_polya_gamma::draw_batch(PolyaGammaBatchInput {
        shapes,
        tilts,
        seed: PgSeed(seed),
    })?;
    out.assign(&draws);
    out.mapv_inplace(|v| v.max(1.0e-12));
    Ok(())
}

/// Parameter dimension above which the posterior is treated as "high-dimensional"
/// for the purpose of the more conservative sampler heuristics below: a higher
/// target-acceptance floor (smaller leapfrog steps) and stronger mass-matrix
/// regularization. The boundary matches the `dense_max_dim` cap at which the
/// engine stops attempting dense mass-matrix adaptation.
const HIGH_DIM_THRESHOLD: usize = 50;

/// Target-acceptance floor enforced for high-dimensional posteriors
/// (`dim > HIGH_DIM_THRESHOLD`). NUTS efficiency degrades faster with too-large
/// steps in high dimensions, so we refuse to honor a requested accept below this.
const HIGH_DIM_TARGET_ACCEPT_FLOOR: f64 = 0.92;
/// Target-acceptance floor for low-dimensional posteriors.
const LOW_DIM_TARGET_ACCEPT_FLOOR: f64 = 0.90;
/// Upper bound on the effective target acceptance. Pushing target accept toward
/// 1 collapses the step size and stalls mixing, so we cap the requested value.
const MAX_TARGET_ACCEPT: f64 = 0.95;

/// Minimum warmup length below which mass-matrix adaptation is disabled: the
/// windowed (Stan-style) adaptation schedule needs enough warmup iterations to
/// populate its initial / terminal buffers, otherwise the estimated metric is
/// noise. With fewer warmup steps the sampler runs on the identity metric.
const MIN_WARMUP_FOR_MASS_ADAPT: usize = 80;

/// Largest parameter dimension for which the engine attempts *dense* mass-matrix
/// adaptation; above this it falls back to a diagonal metric (an `O(p²)` dense
/// metric is neither affordable nor reliably estimable from limited warmup).
const DENSE_MASS_MATRIX_MAX_DIM: usize = 75;

/// Mass-matrix ridge (added to the diagonal of the estimated metric) for the
/// general (mean-family) sampler. The high-dimensional value is larger because
/// the warmup metric estimate is noisier relative to its scale as `p` grows.
const MASS_REGULARIZE_HIGH_DIM: f64 = 0.14;
const MASS_REGULARIZE_LOW_DIM: f64 = 0.10;
/// Mass-matrix ridge for survival posteriors, which are frequently skewed by
/// censoring / rare events and so warrant a heavier ridge than the mean family.
const SURVIVAL_MASS_REGULARIZE_HIGH_DIM: f64 = 0.18;
const SURVIVAL_MASS_REGULARIZE_LOW_DIM: f64 = 0.12;

/// Jitter added during mass-matrix inversion to keep the metric strictly
/// positive-definite against round-off in the warmup covariance estimate.
const MASS_MATRIX_JITTER: f64 = 1e-5;

#[inline]
fn robust_target_accept(requested: f64, dim: usize) -> f64 {
    let floor = if dim > HIGH_DIM_THRESHOLD {
        HIGH_DIM_TARGET_ACCEPT_FLOOR
    } else {
        LOW_DIM_TARGET_ACCEPT_FLOOR
    };
    requested.max(floor).min(MAX_TARGET_ACCEPT)
}

fn jittered_initial_positions(
    config: &NutsConfig,
    dim: usize,
    scale: f64,
    stream: u64,
) -> Vec<Array1<f64>> {
    (0..config.n_chains)
        .map(|chain| {
            let mut rng = StdRng::seed_from_u64(chain_stream_seed(config.seed, chain, stream));
            Array1::from_shape_fn(dim, |_| sample_standard_normal(&mut rng) * scale)
        })
        .collect()
}

fn robust_mass_matrix_config(dim: usize, nwarmup: usize) -> NUTSMassMatrixConfig {
    if nwarmup < MIN_WARMUP_FOR_MASS_ADAPT {
        return NUTSMassMatrixConfig::disabled();
    }
    let start_buffer = (nwarmup / 8).clamp(35, 180);
    let end_buffer = (nwarmup / 5).clamp(50, 250);
    let initial_window = (nwarmup / 20).clamp(10, 60);
    NUTSMassMatrixConfig {
        adaptation: MassMatrixAdaptation::Diagonal,
        start_buffer,
        end_buffer,
        initial_window,
        regularize: if dim > HIGH_DIM_THRESHOLD {
            MASS_REGULARIZE_HIGH_DIM
        } else {
            MASS_REGULARIZE_LOW_DIM
        },
        jitter: MASS_MATRIX_JITTER,
        dense_max_dim: DENSE_MASS_MATRIX_MAX_DIM,
    }
}

fn robust_survival_mass_matrix_config(dim: usize, nwarmup: usize) -> NUTSMassMatrixConfig {
    if nwarmup < MIN_WARMUP_FOR_MASS_ADAPT {
        return NUTSMassMatrixConfig::disabled();
    }
    // Survival posteriors with censoring/rare events are often skewed; this
    // configuration uses diagonal adaptation.
    let start_buffer = (nwarmup / 7).clamp(40, 200);
    let end_buffer = (nwarmup / 4).clamp(60, 280);
    let initial_window = (nwarmup / 20).clamp(10, 60);
    NUTSMassMatrixConfig {
        adaptation: MassMatrixAdaptation::Diagonal,
        start_buffer,
        end_buffer,
        initial_window,
        regularize: if dim > HIGH_DIM_THRESHOLD {
            SURVIVAL_MASS_REGULARIZE_HIGH_DIM
        } else {
            SURVIVAL_MASS_REGULARIZE_LOW_DIM
        },
        jitter: MASS_MATRIX_JITTER,
        dense_max_dim: DENSE_MASS_MATRIX_MAX_DIM,
    }
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            nwarmup: 500,
            n_chains: 4,
            target_accept: 0.9,
            seed: 42,
        }
    }
}

impl NutsConfig {
    /// Create a config with sample counts tuned for the model dimension.
    ///
    /// Higher dimensions need more samples because:
    /// - ESS decreases with dimension (autocorrelation grows)
    /// - Split R-hat needs enough samples per chain to be meaningful
    ///
    /// Rule of thumb: target 100 effective samples per parameter.
    pub fn for_dimension(n_params: usize) -> Self {
        // ESS ≈ n_samples / (1 + 2τ) where τ ≈ sqrt(dim) for well-tuned NUTS
        let effective_autocorr = (n_params as f64).sqrt().max(1.0);

        // Target: at least 100 effective samples per parameter
        let target_ess = 100 * n_params;

        // Samples needed = ESS * (1 + 2τ), with 1.5x safety factor
        let raw_samples = (target_ess as f64 * (1.0 + 2.0 * effective_autocorr) * 1.5) as usize;

        // Clamp to reasonable range [500, 10000]
        let n_samples = raw_samples.clamp(500, 10_000);

        // Warmup ≈ samples (standard practice for adaptation)
        let nwarmup = n_samples;

        // More chains for higher dims (better R-hat estimation)
        let n_chains = if n_params > 50 { 4 } else { 2 };

        Self {
            n_samples,
            nwarmup,
            n_chains,
            target_accept: 0.9,
            seed: 42,
        }
    }
}

/// Result of NUTS sampling.
#[derive(Clone, Debug)]
pub struct NutsResult {
    /// Coefficient samples in ORIGINAL space: shape (n_total_samples, n_coeffs)
    pub samples: Array2<f64>,
    /// Posterior mean
    pub posterior_mean: Array1<f64>,
    /// Posterior standard deviation
    pub posterior_std: Array1<f64>,
    /// R-hat convergence diagnostic
    pub rhat: f64,
    /// Effective sample size
    pub ess: f64,
    /// Whether sampling converged (R-hat < 1.1)
    pub converged: bool,
}

#[derive(Clone, Copy)]
struct NutsConvergenceThresholds {
    max_rhat: f64,
    min_ess: Option<f64>,
}

impl NutsConvergenceThresholds {
    #[inline]
    fn converged(self, rhat: f64, ess: f64) -> bool {
        let rhat_ok = rhat < self.max_rhat;
        match self.min_ess {
            Some(min_ess) => rhat_ok && ess > min_ess,
            None => rhat_ok,
        }
    }
}

fn run_whitened_nuts_samples<Target>(
    target: Target,
    initial_positions: Vec<Array1<f64>>,
    config: &NutsConfig,
    dim: usize,
    mass_cfg: NUTSMassMatrixConfig,
    transition_seed_stream: u64,
    sampling_error_label: &str,
) -> Result<(Array3<f64>, String), String>
where
    Target: HamiltonianTarget<Array1<f64>> + Sync + Send,
{
    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial_positions,
        robust_target_accept(config.target_accept, dim),
        mass_cfg,
    )
    .set_seed(nuts_transition_seed(config.seed, transition_seed_stream));

    let (samples_array, run_stats) = sampler
        .run_progress(config.n_samples, config.nwarmup)
        .map_err(|e| format!("{sampling_error_label}: {e}"))?;
    Ok((samples_array, run_stats.to_string()))
}

fn unwhiten_samples(
    samples_array: &Array3<f64>,
    mode: &Array1<f64>,
    chol: &Array2<f64>,
    dim: usize,
    z_start: usize,
) -> Array2<f64> {
    let shape = samples_array.shape();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let mut samples = Array2::<f64>::zeros((total_samples, dim));
    let mut z_buffer = Array1::<f64>::zeros(dim);
    for chain in 0..n_chains {
        for sample_i in 0..n_samples_out {
            let zview = samples_array.slice(ndarray::s![chain, sample_i, z_start..z_start + dim]);
            z_buffer.assign(&zview);
            let beta = mode + &chol.dot(&z_buffer);
            let sample_idx = chain * n_samples_out + sample_i;
            samples.row_mut(sample_idx).assign(&beta);
        }
    }

    samples
}

fn summarize_unwhitened_nuts_samples(
    samples: Array2<f64>,
    samples_array: &Array3<f64>,
    empty_mean: Array1<f64>,
    convergence: NutsConvergenceThresholds,
) -> NutsResult {
    let posterior_mean = samples.mean_axis(Axis(0)).unwrap_or(empty_mean);
    let posterior_std = samples.std_axis(Axis(0), 0.0);
    let (rhat, ess) = compute_split_rhat_and_ess(samples_array);
    let converged = convergence.converged(rhat, ess);

    NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    }
}

fn run_whitened_nuts_result<Target>(
    target: Target,
    mode: &Array1<f64>,
    chol: &Array2<f64>,
    initial_positions: Vec<Array1<f64>>,
    config: &NutsConfig,
    dim: usize,
    mass_cfg: NUTSMassMatrixConfig,
    transition_seed_stream: u64,
    sampling_error_label: &str,
    empty_mean: Array1<f64>,
    convergence: NutsConvergenceThresholds,
) -> Result<(NutsResult, String), String>
where
    Target: HamiltonianTarget<Array1<f64>> + Sync + Send,
{
    let (samples_array, run_stats) = run_whitened_nuts_samples(
        target,
        initial_positions,
        config,
        dim,
        mass_cfg,
        transition_seed_stream,
        sampling_error_label,
    )?;
    let samples = unwhiten_samples(&samples_array, mode, chol, dim, 0);
    let result =
        summarize_unwhitened_nuts_samples(samples, &samples_array, empty_mean, convergence);
    Ok((result, run_stats))
}

impl NutsResult {
    /// Computes the posterior mean of a function applied to coefficients.
    /// Returns 0.0 if samples is empty to avoid divide-by-zero.
    pub fn posterior_mean_of<F>(&self, f: F) -> f64
    where
        F: Fn(ArrayView1<f64>) -> f64 + Sync,
    {
        let n = self.samples.nrows();
        if n == 0 {
            return 0.0;
        }
        // Posterior mean of a sample-function: parallel reduction over rows.
        // `f: Fn(ArrayView1) -> f64` is shared-access so safe across threads.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let sum: f64 = (0..n).into_par_iter().map(|i| f(self.samples.row(i))).sum();
        sum / n as f64
    }

    /// Computes percentiles of a function applied to coefficients.
    pub fn posterior_interval_of<F>(&self, f: F, lower_pct: f64, upper_pct: f64) -> (f64, f64)
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let n = self.samples.nrows();
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut values: Vec<f64> = (0..n).map(|i| f(self.samples.row(i))).collect();
        values.sort_by(f64::total_cmp);

        (
            gam_math::quantile::quantile_from_sorted(&values, lower_pct / 100.0),
            gam_math::quantile::quantile_from_sorted(&values, upper_pct / 100.0),
        )
    }
}

#[inline]
fn sample_standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Runs a Pólya-Gamma Gibbs sampler for Bernoulli-logit models.
///
/// This sampler is gradient-free: each iteration alternates
/// 1) ω_i | β, y ~ PG(1, x_i^T β), and
/// 2) β | ω, y ~ N(Q^{-1} b, Q^{-1}), with Q = S + X^T diag(ω) X, b = X^T(y - 1/2).
///
/// For weighted data, this implementation is defined for weights ≈ 1.0 because it
/// samples PG(1,·) latent variables.
pub fn run_logit_polya_gamma_gibbs(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || weights.len() != n {
        return Err(HmcError::DimensionMismatch {
            reason: "run_logit_polya_gamma_gibbs: input length mismatch".to_string(),
        }
        .into());
    }
    if mode.len() != p || penalty_matrix.nrows() != p || penalty_matrix.ncols() != p {
        return Err(HmcError::DimensionMismatch {
            reason: "run_logit_polya_gamma_gibbs: coefficient/penalty dimension mismatch"
                .to_string(),
        }
        .into());
    }
    if !weights.iter().all(|w| (*w - 1.0).abs() <= 1e-10) {
        return Err(HmcError::InvalidConfig {
            reason: "run_logit_polya_gamma_gibbs requires unit weights (PG(1,·)); use NUTS for non-unit weights".to_string(),
        }
        .into());
    }
    validate_binary_responses("run_logit_polya_gamma_gibbs", &y, &weights).map_err(String::from)?;
    // Issue #399: the auto-selected PG-Gibbs path is reached for the canonical
    // unit-weight Bernoulli-logit GAM. Without this guard, `n_chains == 0` /
    // `n_samples == 0` would not panic but silently return a degenerate empty
    // `(0, p)` posterior, diverging from the typed error the NUTS path raises
    // for the same inputs. Route it through the shared validator so every
    // `Model.sample` surface rejects degenerate draw/chain counts identically.
    validate_nuts_config(config).map_err(String::from)?;

    let n_iter = config.nwarmup + config.n_samples;

    // b = X^T (y - 1/2), constant across iterations.
    let kappa = y.mapv(|v| v - 0.5);
    let rhs_b = fast_atv(&x, &kappa);

    let mut samples_array = Array3::<f64>::zeros((config.n_chains, config.n_samples, p));
    let mut eta = Array1::<f64>::zeros(n);
    let mut omega = Array1::<f64>::ones(n);
    let pg_shapes = Array1::<u32>::from_elem(n, 1);
    let mut xw = x.to_owned();
    let mut xt_omega_x = Array2::<f64>::zeros((p, p));
    let penalty = penalty_matrix.to_owned();
    let mut q = Array2::<f64>::zeros((p, p));
    let mut mean = Array1::<f64>::zeros(p);
    let mut z = Array1::<f64>::zeros(p);
    let mut noise = Array1::<f64>::zeros(p);

    for chain in 0..config.n_chains {
        let mut init_rng =
            StdRng::seed_from_u64(chain_stream_seed(config.seed, chain, 0xB3C4_5A1F_8E9D_7632));
        let mut draw_rng =
            StdRng::seed_from_u64(chain_stream_seed(config.seed, chain, 0x17A9_26D5_4C1B_E083));
        let mut beta = mode.to_owned();
        // Small jitter so chains are not perfectly coupled.
        for j in 0..p {
            beta[j] += 0.05 * sample_standard_normal(&mut init_rng);
        }

        for iter in 0..n_iter {
            eta.assign(&gam_linalg::faer_ndarray::fast_av(&x, &beta));
            draw_logit_pg1_omega(
                pg_shapes.view(),
                eta.view(),
                gibbs_pg_seed(config.seed, chain, 0x4D94_DF4E_5D72_81AB, iter),
                &mut omega,
            )?;

            // Build Xweighted = diag(sqrt(ω)) X and compute X^T Ω X via faer GEMM.
            // Per-row scaling is fully independent across rows.
            ndarray::Zip::indexed(xw.rows_mut())
                .and(x.rows())
                .and(&omega)
                .par_for_each(|_idx, mut xw_row, x_row, omega_i| {
                    let s = omega_i.sqrt();
                    for j in 0..p {
                        xw_row[j] = x_row[j] * s;
                    }
                });
            fast_ata_into(&xw, &mut xt_omega_x);

            q.assign(&penalty);
            q += &xt_omega_x;

            // β | ω,y ~ N(Q^{-1} b, Q^{-1})
            let factor = q
                .cholesky(Side::Lower)
                .map_err(|e| format!("PG Gibbs failed to factor Q: {:?}", e))?;
            mean.assign(&factor.solvevec(&rhs_b));

            for j in 0..p {
                z[j] = sample_standard_normal(&mut draw_rng);
            }
            let l = factor.lower_triangular();
            back_substitution_lower_transpose_guarded_into(&l, &z, &mut noise);
            beta.assign(&(&mean + &noise));

            if iter >= config.nwarmup {
                let keep_idx = iter - config.nwarmup;
                samples_array
                    .slice_mut(ndarray::s![chain, keep_idx, ..])
                    .assign(&beta);
            }
        }
    }

    let total_samples = config.n_chains * config.n_samples;
    let mut samples = Array2::<f64>::zeros((total_samples, p));
    for chain in 0..config.n_chains {
        for s in 0..config.n_samples {
            let idx = chain * config.n_samples + s;
            samples
                .row_mut(idx)
                .assign(&samples_array.slice(ndarray::s![chain, s, ..]));
        }
    }

    let posterior_mean = samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(p));
    let posterior_std = samples.std_axis(Axis(0), 0.0);
    let (rhat, ess) = if config.n_chains >= 2 && config.n_samples >= 4 {
        compute_split_rhat_and_ess(&samples_array)
    } else {
        (1.0, (total_samples as f64) * 0.5)
    };
    let converged = rhat < 1.1 && ess > 100.0;

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
}

/// Estimate E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ] with PG Gibbs + Rao-Blackwellization.
///
/// For each retained Gibbs state ω:
///   Q = S + Xᵀ diag(ω) X,  μ = Q^{-1} Xᵀ(y-1/2),
/// and with S_k = R_kᵀ R_k:
///   tr(S_k Q^{-1}) + μᵀ S_k μ
/// = tr(R_k Q^{-1} R_kᵀ) + ||R_k μ||².
///
/// Returns one expectation per penalty block k, averaged over retained draws.
pub fn estimate_logit_pg_rao_blackwell_terms(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    penalty_roots: &[Array2<f64>],
    config: &NutsConfig,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || weights.len() != n {
        return Err(HmcError::DimensionMismatch {
            reason: "estimate_logit_pg_rao_blackwell_terms: input length mismatch".to_string(),
        }
        .into());
    }
    if mode.len() != p || penalty_matrix.nrows() != p || penalty_matrix.ncols() != p {
        return Err(HmcError::DimensionMismatch {
            reason: "estimate_logit_pg_rao_blackwell_terms: coefficient/penalty dimension mismatch"
                .to_string(),
        }
        .into());
    }
    if !weights.iter().all(|w| (*w - 1.0).abs() <= 1e-10) {
        return Err(HmcError::InvalidConfig {
            reason: "estimate_logit_pg_rao_blackwell_terms requires unit weights (PG(1,·))"
                .to_string(),
        }
        .into());
    }
    validate_binary_responses("estimate_logit_pg_rao_blackwell_terms", &y, &weights)
        .map_err(String::from)?;
    if penalty_roots.iter().any(|r| r.ncols() != p) {
        return Err(HmcError::DimensionMismatch {
            reason: "estimate_logit_pg_rao_blackwell_terms: root width mismatch".to_string(),
        }
        .into());
    }
    // Precompute transposed root blocks once:
    //   R_k^T is the RHS used for batched solves Q X = R_k^T.
    let penalty_roots_t: Vec<Array2<f64>> =
        penalty_roots.iter().map(|r| r.t().to_owned()).collect();

    let n_iter = config.nwarmup + config.n_samples;

    // Logistic PG identity uses kappa_i = y_i - 1/2 so that
    // b = X^T kappa in the Gaussian conditional for beta|omega.
    let kappa = y.mapv(|v| v - 0.5);
    let rhs_b = fast_atv(&x, &kappa);

    let penalty = penalty_matrix.to_owned();
    let mut eta = Array1::<f64>::zeros(n);
    let mut omega = Array1::<f64>::ones(n);
    let pg_shapes = Array1::<u32>::from_elem(n, 1);
    let mut xw = x.to_owned();
    let mut xt_omega_x = Array2::<f64>::zeros((p, p));
    let mut q = Array2::<f64>::zeros((p, p));
    let mut mean = Array1::<f64>::zeros(p);
    let mut rb_sum = Array1::<f64>::zeros(penalty_roots.len());
    let mut z = Array1::<f64>::zeros(p);
    let mut noise = Array1::<f64>::zeros(p);

    let mut kept = 0usize;
    for chain in 0..config.n_chains {
        let mut init_rng =
            StdRng::seed_from_u64(chain_stream_seed(config.seed, chain, 0x28F0_7B65_1A4D_C93E));
        let mut draw_rng =
            StdRng::seed_from_u64(chain_stream_seed(config.seed, chain, 0xC642_6E35_B5A9_1D80));
        let mut beta = mode.to_owned();
        for j in 0..p {
            beta[j] += 0.05 * sample_standard_normal(&mut init_rng);
        }

        for iter in 0..n_iter {
            eta.assign(&gam_linalg::faer_ndarray::fast_av(&x, &beta));
            draw_logit_pg1_omega(
                pg_shapes.view(),
                eta.view(),
                gibbs_pg_seed(config.seed, chain, 0x83F1_56C9_A7E0_2D4B, iter),
                &mut omega,
            )?;

            ndarray::Zip::from(xw.rows_mut())
                .and(x.rows())
                .and(&omega)
                .par_for_each(|mut xw_row, x_row, &omega_i| {
                    let s = omega_i.sqrt();
                    for j in 0..p {
                        xw_row[j] = x_row[j] * s;
                    }
                });
            fast_ata_into(&xw, &mut xt_omega_x);

            // Conditional precision:
            //   Q = S + X^T diag(omega) X.
            q.assign(&penalty);
            q += &xt_omega_x;

            let factor = q
                .cholesky(Side::Lower)
                .map_err(|e| format!("PG Rao-Blackwell failed to factor Q: {:?}", e))?;
            // Conditional mean:
            //   mu = Q^{-1} b,  b = X^T(y - 1/2).
            mean.assign(&factor.solvevec(&rhs_b));

            // Draw beta for the next Gibbs state.
            for j in 0..p {
                z[j] = sample_standard_normal(&mut draw_rng);
            }
            let l = factor.lower_triangular();
            back_substitution_lower_transpose_guarded_into(&l, &z, &mut noise);
            beta.assign(&(&mean + &noise));

            if iter < config.nwarmup {
                continue;
            }
            kept += 1;

            for (k, r_k) in penalty_roots.iter().enumerate() {
                if r_k.nrows() == 0 {
                    continue;
                }

                // mu^T S_k mu via root form S_k = R_k^T R_k.
                let rmu = r_k.dot(&mean);
                let mu_quad = rmu.dot(&rmu);

                // Batched trace solve:
                //   V_k = Q^{-1} R_k^T  (single multi-RHS solve)
                // then tr(R_k Q^{-1} R_k^T) = <R_k, V_k^T>_F.
                let solved_mat = factor.solve_mat(&penalty_roots_t[k]); // (p, r_k)
                let solved_t = solved_mat.t();
                let mut trace_term = 0.0_f64;
                for (&a, &b) in r_k.iter().zip(solved_t.iter()) {
                    trace_term += a * b;
                }

                rb_sum[k] += trace_term + mu_quad;
            }
        }
    }

    if kept == 0 {
        return Err(HmcError::SamplingFailed {
            reason: "estimate_logit_pg_rao_blackwell_terms: no retained samples".to_string(),
        }
        .into());
    }
    let out = rb_sum.mapv(|v| v / (kept as f64));
    if !out.iter().all(|v| v.is_finite()) {
        return Err(HmcError::NonFiniteState {
            reason: "estimate_logit_pg_rao_blackwell_terms: non-finite expectation".to_string(),
        }
        .into());
    }
    Ok(out)
}

/// Runs NUTS sampling using general-mcmc with whitened parameter space.
///
/// # Arguments
/// * `x` - Design matrix [n_samples, dim]
/// * `y` - Response vector [n_samples]
/// * `weights` - Observation/case weights [n_samples]
/// * `penalty_matrix` - Combined penalty S [dim, dim]
/// * `mode` - MAP estimate μ [dim]
/// * `hessian` - Penalized Hessian H [dim, dim] (NOT the inverse!)
/// * `nuts_family` - Family for log-likelihood computation
/// * `firth_bias_reduction` - Whether Firth bias reduction was used in training
/// * `config` - NUTS configuration
pub(crate) fn run_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    nuts_family: NutsFamily,
    gamma_shape: f64,
    dispersion: gam_solve::model_types::Dispersion,
    firth_bias_reduction: bool,
    offset: Option<ArrayView1<f64>>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    validate_firth_support(nuts_family, firth_bias_reduction).map_err(String::from)?;
    validate_nuts_config(config).map_err(String::from)?;
    if nuts_family == NutsFamily::TweedieLog && !is_valid_tweedie_power(gamma_shape) {
        return Err(format!(
            "Tweedie variance power must be finite and strictly between 1 and 2; got {gamma_shape}"
        ));
    }
    let dim = mode.len();

    // Create posterior target with analytical gradients. When Firth is enabled,
    // this target includes the identifiable-subspace Jeffreys term.
    let target = NutsPosterior::new(
        x,
        y,
        weights,
        penalty_matrix,
        mode,
        hessian,
        nuts_family,
        gamma_shape,
        dispersion,
        firth_bias_reduction,
    )?;
    let target = match offset {
        Some(offset) => target.with_offset(offset)?,
        None => target,
    };

    // Get Cholesky factor for un-whitening samples later
    let chol = target.chol().clone();
    let mode_arr = target.mode().clone();

    let initial_positions = jittered_initial_positions(config, dim, 0.1, 0x0F65_83B2_BC71_4D9E);
    let mass_cfg = robust_mass_matrix_config(dim, config.nwarmup);
    let (result, run_stats) = run_whitened_nuts_result(
        target,
        &mode_arr,
        &chol,
        initial_positions,
        config,
        dim,
        mass_cfg,
        0xF1D3_C2B5_A697_804E,
        "NUTS sampling failed",
        Array1::zeros(dim),
        NutsConvergenceThresholds {
            max_rhat: 1.1,
            min_ess: Some(100.0),
        },
    )?;
    log::info!("NUTS sampling complete: {}", run_stats);

    Ok(result)
}

/// Terminal never-fail Gaussian-posterior sampling target.
///
/// This is the bottom rung of the solver's geometry-driven escalation ladder.
/// When the outer smoothing optimizer cannot certify convergence on a custom
/// (BMS / general) family — typically because Strong-Wolfe stalls on an
/// indefinite or non-smooth LAML objective — the driver no longer dead-ends
/// with an `Err`. Instead it lands here: the *same* penalized objective's
/// curvature (its penalized joint Hessian `H = −∇²log L + Σ_k λ_k S_k`,
/// augmented with the proper (unconditional) Jeffreys/PC term)
/// is used as the precision of a proper Gaussian posterior `N(β̂, H⁻¹)` about
/// the best mode `β̂` the inner solve reached. Sampling a multivariate normal
/// cannot fail: in the worst case (a poorly conditioned `H`) the intervals come
/// out honestly wider, which is the intended "magic for all users" behavior —
/// a finite point with calibrated SEs instead of a hard error.
///
/// The target is expressed in the whitened space `z` (`β = β̂ + L z`,
/// `L Lᵀ = H⁻¹`), where the posterior is the standard normal `N(0, I)`. Its
/// log-density and gradient are then exactly `logp(z) = −½ zᵀz`,
/// `∇ = −z` — a smooth, globally coercive target with no failure mode. The
/// `chol` factor un-whitens draws back to coefficient space, identically to
/// the `NutsPosterior` whitening contract above.
struct GaussianModeTarget;

impl HamiltonianTarget<Array1<f64>> for GaussianModeTarget {
    #[inline]
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        // Standard-normal target in whitened coordinates: logp = -0.5 zᵀz,
        // ∇ = -z. The whitening `L` (built from the penalized Hessian) carries
        // all of the posterior geometry, so the sampler itself only ever sees a
        // unit-covariance Gaussian — which is why this rung cannot stall.
        let mut quad = 0.0;
        for (g, &zi) in grad.iter_mut().zip(position.iter()) {
            *g = -zi;
            quad += zi * zi;
        }
        -0.5 * quad
    }
}

/// Sample the proper Gaussian posterior `N(mode, H⁻¹)` defined by a mode and a
/// (penalized, Jeffreys-augmented) SPD precision `hessian`.
///
/// This is the terminal, never-fail rung of the outer-optimizer escalation:
/// it consumes the same penalized-objective curvature the inner machinery
/// already computed and returns an honest posterior summary. It returns `Err`
/// only for a *structurally* impossible request (dimension mismatch, a Hessian
/// that is not even positive-definite after symmetrization, a degenerate
/// config) — never for "did not converge", which is precisely the dead-end this
/// path exists to remove.
///
/// `hessian` must be the SPD penalized joint Hessian at `mode` (e.g. from
/// `compute_joint_geometry`). It is symmetrized defensively and Cholesky-
/// factored to build the whitening `L` with `L Lᵀ = H⁻¹`.
pub fn sample_gaussian_mode_posterior(
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    config: &NutsConfig,
) -> Result<GaussianModePosterior, String> {
    validate_nuts_config(config).map_err(String::from)?;
    let dim = mode.len();
    if hessian.nrows() != dim || hessian.ncols() != dim {
        return Err(format!(
            "Gaussian-posterior fallback: hessian shape {:?} does not match mode dim {dim}",
            hessian.dim()
        ));
    }
    if dim == 0 {
        return Err("Gaussian-posterior fallback: zero-dimensional posterior".to_string());
    }

    // Symmetrize defensively (the assembled joint Hessian may carry
    // floating-point asymmetry from directional-callback construction) and add
    // a tiny jitter on the diagonal so a Hessian that is SPD-up-to-roundoff at a
    // boundary optimum still factors. The jitter only ever *widens* the
    // posterior, consistent with the honest-interval guarantee.
    let mut h = hessian.to_owned();
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = avg;
            h[[j, i]] = avg;
        }
    }
    let diag_scale = (0..dim).map(|i| h[[i, i]].abs()).fold(0.0_f64, f64::max);
    let jitter = (diag_scale * 1e-10).max(1e-12);
    for i in 0..dim {
        h[[i, i]] += jitter;
    }

    let mode_owned = mode.to_owned();
    let whitening = hessian_whitening_transform(
        h.view(),
        dim,
        1.0,
        "Gaussian-posterior fallback Cholesky failed",
    )?;
    let chol = whitening.chol;
    let target = GaussianModeTarget;
    let initial_positions = jittered_initial_positions(config, dim, 0.1, 0x51A6_2C73_90E4_1DBF);
    let mass_cfg = robust_mass_matrix_config(dim, config.nwarmup);
    let (result, run_stats) = run_whitened_nuts_result(
        target,
        &mode_owned,
        &chol,
        initial_positions,
        config,
        dim,
        mass_cfg,
        0x7C19_5A3E_82D6_44B1,
        "Gaussian-posterior fallback NUTS sampling failed",
        mode_owned.clone(),
        NutsConvergenceThresholds {
            max_rhat: 1.1,
            min_ess: None,
        },
    )?;
    log::info!(
        "never-fail Gaussian-posterior fallback: sampling complete dim={dim} {}",
        run_stats
    );

    Ok(GaussianModePosterior {
        samples: result.samples,
        posterior_mean: result.posterior_mean,
        posterior_std: result.posterior_std,
        rhat: result.rhat,
        ess: result.ess,
    })
}

/// Penalty subtracted from the log-density when the `ρ`-criterion closure
/// reports an infeasible / non-finite point during Tier-2 `ρ`-posterior NUTS
/// (#938). The fallback density is the whitened standard normal shifted down by
/// this constant, so the sampler sees a smooth, coercive pull back toward the
/// feasible region around `ρ̂` instead of a `-inf` cliff.
const RHO_NUTS_INFEASIBLE_LOGP_PENALTY: f64 = 1.0e8;

/// Tier-2 of the exact marginal-smoothing inference stack (#938): the whitened
/// `ρ`-criterion Hamiltonian target.
///
/// This reuses the module's β-level whitening design ONE LEVEL UP: the target
/// log-density is `logp(ρ) = −(criterion(ρ) − criterion(ρ̂))` — i.e.
/// `π(ρ|y) ∝ exp(−LAML(ρ))`, the exact profiled criterion the outer optimizer
/// minimizes — expressed in the whitened coordinates `ρ = ρ̂ + L z` with
/// `L Lᵀ = H_ρ⁻¹` built from the exact outer Hessian at `ρ̂`. The gradient is
/// the caller's exact profiled `ρ`-gradient pushed through the chain rule:
/// `∇_z logp = −Lᵀ ∇_ρ criterion`.
///
/// The criterion closure is `FnMut` (each evaluation is one warm inner profile
/// solve with interior caches), so it is serialized behind a `Mutex`; chains
/// take turns evaluating, which also keeps the inner warm-start trajectory
/// coherent.
struct WhitenedRhoCriterionTarget<F> {
    /// `ρ ↦ (criterion(ρ), ∇_ρ criterion(ρ))`; `None` marks an infeasible point.
    criterion_and_grad: Mutex<F>,
    /// `ρ̂`, the converged smoothing parameters (the whitening center).
    mode: Array1<f64>,
    /// `L` with `L Lᵀ = H_ρ⁻¹`: maps whitened `z` to `ρ = ρ̂ + L z`.
    chol: Array2<f64>,
    /// `Lᵀ`, for the gradient chain rule.
    chol_t: Array2<f64>,
    /// `criterion(ρ̂)`, subtracted for numerical stability (cancels in MCMC).
    cost_hat: f64,
}

impl<F> HamiltonianTarget<Array1<f64>> for WhitenedRhoCriterionTarget<F>
where
    F: FnMut(&Array1<f64>) -> Option<(f64, Array1<f64>)> + Send,
{
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let rho = &self.mode + &self.chol.dot(position);
        let eval = {
            let mut criterion = self
                .criterion_and_grad
                .lock()
                .expect("rho-criterion mutex poisoned");
            (*criterion)(&rho)
        };
        match eval {
            Some((cost, g))
                if cost.is_finite()
                    && g.len() == position.len()
                    && g.iter().all(|v| v.is_finite()) =>
            {
                let grad_z = self.chol_t.dot(&g);
                for (gi, &v) in grad.iter_mut().zip(grad_z.iter()) {
                    *gi = -v;
                }
                -(cost - self.cost_hat)
            }
            _ => {
                // Infeasible criterion: smooth coercive fallback toward ρ̂.
                let mut quad = 0.0;
                for (gi, &zi) in grad.iter_mut().zip(position.iter()) {
                    *gi = -zi;
                    quad += zi * zi;
                }
                -0.5 * quad - RHO_NUTS_INFEASIBLE_LOGP_PENALTY
            }
        }
    }
}

/// Run NUTS over the smoothing parameters `ρ` with the exact profiled criterion
/// and gradient (#938 Tier 2).
///
/// * `rho_hat` — converged `ρ̂` (the whitening center and chain seed).
/// * `outer_hessian` — exact outer Hessian `H_ρ` at `ρ̂` (symmetrized and
///   jittered defensively, then Cholesky-factored for the whitening).
/// * `criterion_and_grad` — `ρ ↦ (LAML(ρ), ∇_ρ LAML(ρ))`, both exact; `None`
///   for infeasible `ρ`. Each call is one warm inner profile solve.
/// * `config` — sampler configuration; determinism comes from `config.seed`
///   through the same splitmix64 chain/transition streams as every other NUTS
///   entry point (no clock, no global RNG).
///
/// Returns draws in the ORIGINAL `ρ` space (un-whitened), with split-R̂/ESS
/// diagnostics.
pub fn run_rho_criterion_nuts<F>(
    rho_hat: ArrayView1<f64>,
    outer_hessian: ArrayView2<f64>,
    mut criterion_and_grad: F,
    config: &NutsConfig,
) -> Result<NutsResult, String>
where
    F: FnMut(&Array1<f64>) -> Option<(f64, Array1<f64>)> + Send,
{
    validate_nuts_config(config).map_err(String::from)?;
    let dim = rho_hat.len();
    if dim == 0 {
        return Err("rho-posterior NUTS: zero-dimensional rho".to_string());
    }
    if outer_hessian.nrows() != dim || outer_hessian.ncols() != dim {
        return Err(format!(
            "rho-posterior NUTS: outer Hessian shape {:?} does not match rho dim {dim}",
            outer_hessian.dim()
        ));
    }

    // Symmetrize + jitter the exact outer Hessian so a boundary optimum that is
    // SPD-up-to-roundoff still factors; jitter only widens the proposal metric.
    let mut h = outer_hessian.to_owned();
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
            h[[i, j]] = avg;
            h[[j, i]] = avg;
        }
    }
    let diag_scale = (0..dim).map(|i| h[[i, i]].abs()).fold(0.0_f64, f64::max);
    let jitter = (diag_scale * 1e-10).max(1e-12);
    for i in 0..dim {
        h[[i, i]] += jitter;
    }

    let mode = rho_hat.to_owned();
    let whitening = hessian_whitening_transform(
        h.view(),
        dim,
        1.0,
        "rho-posterior NUTS: outer-Hessian Cholesky failed",
    )?;

    let cost_hat = match criterion_and_grad(&mode) {
        Some((cost, _)) if cost.is_finite() => cost,
        _ => {
            return Err(
                "rho-posterior NUTS: criterion is infeasible at rho_hat itself".to_string(),
            );
        }
    };

    let chol = whitening.chol;
    let target = WhitenedRhoCriterionTarget {
        criterion_and_grad: Mutex::new(criterion_and_grad),
        mode: mode.clone(),
        chol: chol.clone(),
        chol_t: whitening.chol_t,
        cost_hat,
    };
    let initial_positions = jittered_initial_positions(config, dim, 0.1, 0x3D8A_91C4_E27B_5F60);
    // The rho target is already whitened by the exact outer Hessian at rho_hat,
    // so the local mass matrix in z-space is identity. Re-adapting a diagonal or
    // dense metric during warmup would spend expensive profile solves estimating
    // curvature we have already supplied analytically.
    let mass_cfg = NUTSMassMatrixConfig::disabled();
    let (result, run_stats) = run_whitened_nuts_result(
        target,
        &mode,
        &chol,
        initial_positions,
        config,
        dim,
        mass_cfg,
        0x6B42_E9A1_05D7_C83F,
        "rho-posterior NUTS sampling failed",
        mode.clone(),
        NutsConvergenceThresholds {
            max_rhat: 1.1,
            min_ess: None,
        },
    )?;
    log::info!("rho-posterior NUTS (#938 tier 2): sampling complete dim={dim} {run_stats}");
    Ok(result)
}

/// Flattened numeric inputs for GLM-family NUTS sampling.
pub struct GlmFlatInputs<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub penalty_matrix: ArrayView2<'a, f64>,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
    pub gamma_shape: Option<f64>,
    /// Dispersion parameter φ used to scale the likelihood and the
    /// whitening Cholesky. For fixed-scale families (Binomial, Poisson)
    /// this is `Dispersion::Known(1.0)` and has no numerical effect;
    /// for Gaussian / Gamma it carries the estimated `phi` so that the
    /// sampler targets the φ-scaled posterior covariance `Vb = φ·H⁻¹`.
    /// See `inference::dispersion_cov` for the ownership invariants.
    pub dispersion: gam_solve::model_types::Dispersion,
    pub firth_bias_reduction: bool,
    /// Fixed additive offset on the linear predictor (η = Xβ + offset), or
    /// `None` for an offset-free fit. Carried so posterior sampling targets the
    /// same η the model was fit and predicts on; omitting it sampled the wrong
    /// posterior for any `--offset-column` model (#882).
    pub offset: Option<ArrayView1<'a, f64>>,
}

/// Flat survival inputs for engine-facing HMC APIs.
pub struct SurvivalFlatInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub weights: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
    pub eta_offset_entry: Option<ArrayView1<'a, f64>>,
    pub eta_offset_exit: Option<ArrayView1<'a, f64>>,
    pub derivative_offset_exit: Option<ArrayView1<'a, f64>>,
}

/// Flattened numeric inputs for Royston-Parmar NUTS sampling.
pub struct SurvivalNutsInputs<'a> {
    pub flat: SurvivalFlatInputs<'a>,
    pub penalties: gam_models::survival::PenaltyBlocks,
    pub monotonicity: gam_models::survival::SurvivalMonotonicityPenalty,
    pub spec: gam_models::survival::SurvivalSpec,
    pub structurally_monotonic: bool,
    pub structural_time_columns: usize,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
}

/// Family-dispatched flattened NUTS inputs.
pub enum FamilyNutsInputs<'a> {
    Glm(GlmFlatInputs<'a>),
    Survival(Box<SurvivalNutsInputs<'a>>),
}

/// Return the explicit fitted penalized Hessian used for HMC/NUTS whitening.
///
/// This is the only supported upstream-to-HMC curvature handoff: callers must
/// pass a dense Hessian (or an already materialized exact operator stored as a
/// dense Hessian) exported by the fitter. We deliberately do not synthesize a
/// numerical Hessian and do not invert `beta_covariance` as a compatibility
/// fallback, because either path can silently whiten against curvature that the
/// upstream fit never certified.
pub fn explicit_fit_hessian_for_whitening<'a>(
    fit: &'a UnifiedFitResult,
    expected_dim: usize,
    label: &str,
) -> Result<&'a Array2<f64>, String> {
    let hessian = fit.penalized_hessian().ok_or_else(|| {
        format!(
            "{label}: fit result is missing an explicit penalized Hessian for HMC/NUTS whitening"
        )
    })?;
    validate_explicit_dense_hessian_for_whitening(
        &format!("{label} penalized Hessian"),
        hessian,
        expected_dim,
    )
    .map_err(|err| err.to_string())?;
    Ok(hessian)
}

/// Family-agnostic flattened NUTS entrypoint across all supported likelihood families.
pub fn run_nuts_sampling_flattened_family(
    likelihood: LikelihoodSpec,
    inputs: FamilyNutsInputs<'_>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    if let FamilyNutsInputs::Glm(glm) = &inputs
        && glm.firth_bias_reduction
        && !likelihood_spec_supports_firth(&likelihood)
    {
        return Err(HmcError::FirthUnsupported {
            reason: format!(
                "NUTS with Firth requires a Binomial inverse link with a Fisher-weight jet; {} does not support it",
                likelihood.pretty_name()
            ),
        }
        .into());
    }

    match (likelihood.response.clone(), likelihood.link.clone(), inputs) {
        (
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
            FamilyNutsInputs::Glm(glm),
        ) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::Gaussian,
            1.0,
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
            FamilyNutsInputs::Glm(glm),
        ) => {
            // Auto-select PG Gibbs when assumptions hold; otherwise fall back to NUTS.
            // This gives gradient-free posterior draws for standard Bernoulli logit GAMs.
            // The Pólya-Gamma augmentation here assumes η = Xβ (no offset); an
            // offset model routes to NUTS, which carries the offset through
            // `glm.offset` (#882). PG-with-offset is a valid but separate scheme
            // we deliberately do not duplicate.
            if !glm.firth_bias_reduction
                && glm.offset.is_none()
                && glm.weights.iter().all(|w| (*w - 1.0).abs() <= 1e-10)
            {
                run_logit_polya_gamma_gibbs(
                    glm.x,
                    glm.y,
                    glm.weights,
                    glm.penalty_matrix,
                    glm.mode,
                    config,
                )
            } else {
                run_nuts_sampling(
                    glm.x,
                    glm.y,
                    glm.weights,
                    glm.penalty_matrix,
                    glm.mode,
                    glm.hessian,
                    NutsFamily::BinomialLogit,
                    1.0,
                    glm.dispersion,
                    glm.firth_bias_reduction,
                    glm.offset,
                    config,
                )
            }
        }
        (
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
            FamilyNutsInputs::Glm(glm),
        ) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::BinomialProbit,
            1.0,
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
            FamilyNutsInputs::Glm(glm),
        ) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::BinomialCLogLog,
            1.0,
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (
            ResponseFamily::Binomial,
            InverseLink::LatentCLogLog(_),
            FamilyNutsInputs::Glm(glm),
        ) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::BinomialCLogLog,
            1.0,
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (ResponseFamily::Binomial, InverseLink::Mixture(_), FamilyNutsInputs::Glm(_)) => Err(
            "BinomialMixture NUTS is not implemented yet; use fit_gam/predict_gam for blended inverse-link models"
                .to_string(),
        ),
        (ResponseFamily::Binomial, InverseLink::Sas(_), FamilyNutsInputs::Glm(_)) => Err(
            "BinomialSas NUTS is not implemented yet; use fit_gam/predict_gam for SAS-link models"
                .to_string(),
        ),
        (ResponseFamily::Binomial, InverseLink::BetaLogistic(_), FamilyNutsInputs::Glm(_)) => Err(
            "BinomialBetaLogistic NUTS is not implemented yet; use fit_gam/predict_gam for beta-logistic-link models"
                .to_string(),
        ),
        (ResponseFamily::Binomial, InverseLink::Standard(_), FamilyNutsInputs::Glm(_)) => Err(
            "NUTS sampling is not implemented for this binomial inverse link".to_string(),
        ),
        (ResponseFamily::RoystonParmar, _, FamilyNutsInputs::Survival(survival)) => {
            survival_hmc::run_survival_nuts_sampling(
                survival.flat.age_entry,
                survival.flat.age_exit,
                survival.flat.event_target,
                survival.flat.event_competing,
                survival.flat.weights,
                survival.flat.x_entry,
                survival.flat.x_exit,
                survival.flat.x_derivative,
                survival.flat.eta_offset_entry,
                survival.flat.eta_offset_exit,
                survival.flat.derivative_offset_exit,
                survival.penalties,
                survival.monotonicity,
                survival.spec,
                survival.structurally_monotonic,
                survival.structural_time_columns,
                survival.mode,
                survival.hessian,
                config,
            )
        }
        (ResponseFamily::RoystonParmar, _, FamilyNutsInputs::Glm(_)) => Err(
            "RoystonParmar family requires FamilyNutsInputs::Survival flattened inputs".to_string(),
        ),
        (_, _, FamilyNutsInputs::Survival(_)) => Err(
            "Survival flattened inputs are only valid for the Royston-Parmar response family"
                .to_string(),
        ),
        (ResponseFamily::Poisson, _, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::PoissonLog,
            1.0,
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (ResponseFamily::Tweedie { p }, _, FamilyNutsInputs::Glm(glm)) => {
            // Family mapping: Tweedie payload p is passed through the family-parameter slot.
            // The Tweedie dispersion phi remains in glm.dispersion, matching REML.
            if !is_valid_tweedie_power(p) {
                return Err(format!(
                    "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                ));
            }
            run_nuts_sampling(
                glm.x,
                glm.y,
                glm.weights,
                glm.penalty_matrix,
                glm.mode,
                glm.hessian,
                NutsFamily::TweedieLog,
                p,
                glm.dispersion,
                glm.firth_bias_reduction,
                glm.offset,
                config,
            )
        }
        (ResponseFamily::NegativeBinomial { theta, .. }, _, FamilyNutsInputs::Glm(glm)) => {
            // Family mapping: NegativeBinomial payload theta is passed through the family slot.
            // NB dispersion scale is unit; theta is not derived from fixed_phi.
            run_nuts_sampling(
                glm.x,
                glm.y,
                glm.weights,
                glm.penalty_matrix,
                glm.mode,
                glm.hessian,
                NutsFamily::NegativeBinomialLog,
                theta,
                glm.dispersion,
                glm.firth_bias_reduction,
                glm.offset,
                config,
            )
        }
        (ResponseFamily::Beta { .. }, _, FamilyNutsInputs::Glm(_)) => Err(
            "NUTS sampling is not implemented for beta-regression logit".to_string(),
        ),
        (ResponseFamily::Gamma, _, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::GammaLog,
            glm.gamma_shape.unwrap_or(1.0),
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (ResponseFamily::Gaussian, _, FamilyNutsInputs::Glm(_)) => Err(
            "NUTS sampling is only implemented for Gaussian with identity link".to_string(),
        ),
    }
}

// ============================================================================
// Joint (β, θ) Link-Wiggle HMC
// ============================================================================
//
// NUTS sampling over the joint parameter space [β_eta; β_wiggle] for models
// with a structurally monotone I-spline link wiggle. The wiggle introduces a
// nonlinear coupling:
//
//   η(β_eta, β_wiggle) = q₀(β_eta) + B(q₀(β_eta)) · β_wiggle
//
// where B is the shared monotone wiggle basis evaluated at the base linear
// predictor q₀ = X · β_eta. The gradient of log p(y|β_eta, β_wiggle) w.r.t.
// β_eta picks up a chain-rule factor g'(q₀) = 1 + B'(q₀) · β_wiggle / range_width
// from the dependence of B on q₀.
//
// Whitening uses the Cholesky of the joint Hessian at the mode, exactly as for
// the standard NutsPosterior. C^1 linear extension outside the training knot
// range prevents basis evaluation discontinuities.

/// Fixed spline artifacts for link-wiggle posterior sampling.
#[derive(Clone)]
pub struct LinkWiggleSplineArtifacts {
    /// Knot range (min, max) from training (in standardized [0,1] space of q₀)
    pub knot_range: (f64, f64),
    /// Full knot vector for the shared monotone I-spline basis
    pub knot_vector: Array1<f64>,
    /// I-spline degree
    pub degree: usize,
}

/// Whitened log-posterior target for joint (β_eta, β_wiggle) with analytical gradients.
#[derive(Clone)]
pub struct LinkWigglePosterior {
    /// Main design matrix X (n × p_main)
    x: Arc<Array2<f64>>,
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    /// Penalty for main coefficients (p_main × p_main)
    penalty_base: Arc<Array2<f64>>,
    /// Penalty for wiggle coefficients (p_wiggle × p_wiggle)
    penalty_link: Arc<Array2<f64>>,
    mode_beta: Arc<Array1<f64>>,
    mode_theta: Arc<Array1<f64>>,
    spline: LinkWiggleSplineArtifacts,
    /// L where LL^T = H^{-1} (joint Hessian)
    chol: Array2<f64>,
    /// L^T for gradient chain rule
    chol_t: Array2<f64>,
    p_base: usize,
    p_link: usize,
    n_samples: usize,
    nuts_family: NutsFamily,
    /// Family-specific noise parameter: Gaussian sigma or Gamma shape.
    scale: f64,
    /// Coefficient-covariance scale `cov_scale` (#679/#680 invariant): the
    /// `Vb = cov_scale·H⁻¹` multiplier driving both the whitening
    /// (`L Lᵀ = cov_scale·H⁻¹`) and the target penalty weight
    /// (`penalty_scale = 1/cov_scale`). `σ²` for profiled Gaussian, `1.0` for
    /// every weight-carries-dispersion family (Gamma/Tweedie/NB).
    cov_scale: f64,
}

impl LinkWigglePosterior {
    /// Standardize q₀ values to [0,1] range using training knot bounds.
    #[inline]
    fn standardized_z(&self, u: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let (min_u, max_u) = self.spline.knot_range;
        let rw = (max_u - min_u).max(1e-6);
        let z_raw: Array1<f64> = u.mapv(|v| (v - min_u) / rw);
        let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));
        (z_raw, z_c, rw)
    }

    /// Creates a new link-wiggle posterior target.
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_base: ArrayView2<f64>,
        penalty_link: ArrayView2<f64>,
        mode_beta: ArrayView1<f64>,
        mode_theta: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        spline: LinkWiggleSplineArtifacts,
        nuts_family: NutsFamily,
        scale: f64,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let p_base = x.ncols();
        let p_link = mode_theta.len();
        let dim = p_base + p_link;
        if hessian.nrows() != dim || hessian.ncols() != dim {
            return Err(HmcError::DimensionMismatch {
                reason: format!(
                    "LinkWigglePosterior: Hessian dim mismatch: {}x{} vs expected {}x{}",
                    hessian.nrows(),
                    hessian.ncols(),
                    dim,
                    dim,
                ),
            }
            .into());
        }
        if nuts_family.likelihood_spec().is_binomial() {
            validate_binary_responses("binomial link-wiggle NUTS", &y, &weights)
                .map_err(String::from)?;
        }
        if matches!(nuts_family, NutsFamily::NegativeBinomialLog) {
            validate_count_responses("negative-binomial link-wiggle NUTS", &y, &weights)
                .map_err(String::from)?;
        }
        // Whitening metric `L Lᵀ = cov_scale · H⁻¹` (#679/#680 invariant), so
        // scale `L` by `√cov_scale`. For the link-wiggle joint target `scale`
        // is σ (Gaussian), so the profiled-Gaussian covariance scale is
        // `cov_scale = σ²`. Every other family folds its dispersion into the
        // working weight / the `shape`/`theta` already inside its
        // log-likelihood, so `cov_scale = 1` and this is a no-op. The previous
        // Gamma branch scaled `L` by `1/√shape = √φ`, mis-preconditioning the
        // sampler against `φ·H⁻¹` instead of the correct `H⁻¹` (#680).
        let cov_scale = match nuts_family {
            NutsFamily::Gaussian => scale * scale,
            _ => 1.0,
        };
        let whitening = hessian_whitening_transform(
            hessian,
            dim,
            cov_scale,
            "LinkWigglePosterior Cholesky failed",
        )?;
        let chol = whitening.chol;
        let chol_t = whitening.chol_t;
        Ok(Self {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            penalty_base: Arc::new(penalty_base.to_owned()),
            penalty_link: Arc::new(penalty_link.to_owned()),
            mode_beta: Arc::new(mode_beta.to_owned()),
            mode_theta: Arc::new(mode_theta.to_owned()),
            spline,
            chol,
            chol_t,
            p_base,
            p_link,
            n_samples,
            nuts_family,
            scale,
            cov_scale,
        })
    }

    /// Evaluate the wiggle basis and compute η = q₀ + B(q₀)·θ with C^1 linear extension.
    fn evaluate_link(&self, u: &Array1<f64>, theta: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        let n = u.len();
        if theta.is_empty() {
            return (Array2::zeros((n, 0)), u.clone());
        }

        let (z_raw, z_c, _) = self.standardized_z(u);
        let Ok(mut basis) = monotone_wiggle_basis_with_derivative_order(
            z_c.view(),
            &self.spline.knot_vector,
            self.spline.degree,
            0,
        ) else {
            return (Array2::zeros((n, theta.len())), u.clone());
        };
        if basis.ncols() != theta.len() {
            return (Array2::zeros((n, theta.len())), u.clone());
        }

        // C^1 linear extension outside [0, 1]:
        // B_ext(z_raw) = B(z_c) + (z_raw - z_c) * B'(z_c)
        let mut needs_ext = false;
        for i in 0..n {
            if (z_raw[i] - z_c[i]).abs() > 1e-12 {
                needs_ext = true;
                break;
            }
        }
        if needs_ext
            && let Ok(b_prime) = monotone_wiggle_basis_with_derivative_order(
                z_c.view(),
                &self.spline.knot_vector,
                self.spline.degree,
                1,
            )
        {
            for i in 0..n {
                let dz = z_raw[i] - z_c[i];
                if dz.abs() <= 1e-12 {
                    continue;
                }
                for j in 0..basis.ncols().min(b_prime.ncols()) {
                    basis[[i, j]] += dz * b_prime[[i, j]];
                }
            }
        }
        (
            basis.clone(),
            u + &gam_linalg::faer_ndarray::fast_av(&basis, theta),
        )
    }

    /// Compute dη/dq₀ = 1 + B'(q₀)·θ / range_width (chain-rule factor for β_eta gradient).
    fn compute_g_prime(&self, u: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
        let n = u.len();
        let mut g = Array1::<f64>::ones(n);
        let (_, z_c, rw) = self.standardized_z(u);
        if theta.is_empty() {
            return g;
        }

        let Ok(b_prime_constrained) = monotone_wiggle_basis_with_derivative_order(
            z_c.view(),
            &self.spline.knot_vector,
            self.spline.degree,
            1,
        ) else {
            return g;
        };
        if b_prime_constrained.ncols() != theta.len() {
            return g;
        }
        let dwiggle_dz = gam_linalg::faer_ndarray::fast_av(&b_prime_constrained, theta);
        ndarray::Zip::from(&mut g)
            .and(&dwiggle_dz)
            .par_for_each(|gi, &dw| *gi = 1.0 + dw / rw);
        g
    }

    fn compute_logp_and_grad_into(&self, z: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let dim = self.p_base + self.p_link;

        // Un-whiten: q = mode + L·z
        let mut mode = Array1::<f64>::zeros(dim);
        mode.slice_mut(ndarray::s![0..self.p_base])
            .assign(&self.mode_beta);
        mode.slice_mut(ndarray::s![self.p_base..])
            .assign(&self.mode_theta);
        let q = &mode + &self.chol.dot(z);
        let beta = q.slice(ndarray::s![0..self.p_base]).to_owned();
        let theta = q.slice(ndarray::s![self.p_base..]).to_owned();

        // Compute η = q₀ + B(q₀)·θ where q₀ = X·β
        let u = gam_linalg::faer_ndarray::fast_av(self.x.as_ref(), &beta);
        let (bwiggle, eta) = self.evaluate_link(&u, &theta);

        // Log-likelihood and residuals via family dispatch
        let ll;
        let mut residual = Array1::<f64>::zeros(self.n_samples);
        match self.nuts_family {
            NutsFamily::Gaussian => {
                let inv_scale_sq = 1.0 / (self.scale * self.scale).max(1e-10);
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let r = self.y[i] - eta[i];
                    let w = self.weights[i];
                    ll_acc -= 0.5 * w * r * r * inv_scale_sq;
                    residual[i] = w * r * inv_scale_sq;
                }
                ll = ll_acc;
            }
            NutsFamily::BinomialLogit => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    ll_acc += w_i * (y_i * eta_i - gam_linalg::utils::stable_softplus(eta_i));
                    let mu = gam_linalg::utils::stable_logistic(eta_i);
                    residual[i] = w_i * (y_i - mu);
                }
                ll = ll_acc;
            }
            NutsFamily::BinomialProbit => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let log_phi_pos = log_ndtr(eta_i);
                    let log_phi_neg = log_ndtr(-eta_i);
                    ll_acc += w_i * (y_i * log_phi_pos + (1.0 - y_i) * log_phi_neg);
                    let log_phi = standard_normal_log_pdf(eta_i);
                    let ratio_pos = (log_phi - log_phi_pos).exp();
                    let ratio_neg = (log_phi - log_phi_neg).exp();
                    residual[i] = w_i * (y_i * ratio_pos - (1.0 - y_i) * ratio_neg);
                }
                ll = ll_acc;
            }
            NutsFamily::BinomialCLogLog => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    if !eta_i.is_finite() {
                        grad.fill(0.0);
                        return f64::NEG_INFINITY;
                    }
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let (ll_i, residual_i) = match cloglog_bernoulli_logp_and_residual(eta_i, y_i) {
                        Ok(values) => values,
                        Err(_) => {
                            grad.fill(0.0);
                            return f64::NEG_INFINITY;
                        }
                    };
                    ll_acc += w_i * ll_i;
                    residual[i] = w_i * residual_i;
                }
                ll = ll_acc;
            }
            NutsFamily::PoissonLog => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    // Only non-finite η invalidates the target: any finite η
                    // has a valid Poisson log-density (the old ±30 window
                    // declared e.g. y = 0, η = −31 impossible, pinning the
                    // sampled posterior to an arbitrary boundary). Genuine
                    // binary64 exhaustion is caught after the family match.
                    if !eta_i.is_finite() {
                        grad.fill(0.0);
                        return f64::NEG_INFINITY;
                    }
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let mu = eta_i.exp();
                    ll_acc += w_i * (y_i * eta_i - mu);
                    residual[i] = w_i * (y_i - mu);
                }
                ll = ll_acc;
            }
            NutsFamily::TweedieLog => {
                let mut ll_acc = 0.0;
                // Family mapping: Tweedie scale carries payload p; phi is not stored here.
                // Invalid p makes the link-wiggle target invalid instead of defaulting.
                if !is_valid_tweedie_power(self.scale) {
                    grad.fill(0.0);
                    return f64::NEG_INFINITY;
                }
                let p = self.scale;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    if !eta_i.is_finite() {
                        grad.fill(0.0);
                        return f64::NEG_INFINITY;
                    }
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let mu = eta_i.exp().max(1e-300);
                    ll_acc +=
                        w_i * (y_i * mu.powf(1.0 - p) / (1.0 - p) - mu.powf(2.0 - p) / (2.0 - p));
                    residual[i] = w_i * (y_i - mu) * mu.powf(1.0 - p);
                }
                ll = ll_acc;
            }
            NutsFamily::NegativeBinomialLog => {
                let mut ll_acc = 0.0;
                // Family mapping: NegativeBinomial scale carries payload theta.
                // Invalid theta makes the link-wiggle target invalid instead of clamping.
                if !(self.scale.is_finite() && self.scale > 0.0) {
                    grad.fill(0.0);
                    return f64::NEG_INFINITY;
                }
                let theta = self.scale;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    if !eta_i.is_finite() {
                        grad.fill(0.0);
                        return f64::NEG_INFINITY;
                    }
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    if w_i <= 0.0 {
                        residual[i] = 0.0;
                        continue;
                    }
                    let mu = eta_i.exp().max(1e-12);
                    let log_mu_term = if y_i > 0.0 { y_i * mu.ln() } else { 0.0 };
                    ll_acc += w_i
                        * (statrs::function::gamma::ln_gamma(y_i + theta)
                            - statrs::function::gamma::ln_gamma(theta)
                            - statrs::function::gamma::ln_gamma(y_i + 1.0)
                            + theta * (theta.ln() - (theta + mu).ln())
                            + log_mu_term
                            - y_i * (theta + mu).ln());
                    residual[i] = w_i * theta * (y_i - mu) / (theta + mu);
                }
                ll = ll_acc;
            }
            NutsFamily::GammaLog => {
                let mut ll_acc = 0.0;
                let shape = self.scale.max(1e-10);
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    if !eta_i.is_finite() {
                        grad.fill(0.0);
                        return f64::NEG_INFINITY;
                    }
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let mu = eta_i.exp();
                    ll_acc += w_i * shape * (-y_i / mu - eta_i);
                    residual[i] = w_i * shape * (y_i / mu - 1.0);
                }
                ll = ll_acc;
            }
        }

        // A finite η can still exhaust binary64 (overflowing exp against a
        // positive response): genuine log-density underflow is rejected as −∞
        // with a zero gradient — never via an a-priori η support window.
        if !ll.is_finite() {
            grad.fill(0.0);
            return f64::NEG_INFINITY;
        }

        // Penalty weight = 1/cov_scale (#679/#680 invariant), matching the
        // factor the likelihood already carries so the prior and likelihood
        // live on the same scale and the MAP-anchored target curvature equals
        // `Vb⁻¹ = H/cov_scale`. The Gaussian block above multiplies through by
        // `1/σ²`, so `penalty_scale = 1/σ²`. The Gamma block carries an explicit
        // `shape = 1/φ` factor in its score (`w_i·shape·(y/μ − 1)`) — that is
        // the *data* Fisher information, already folded into the working
        // weight, so the penalty must stay UNSCALED (`cov_scale = 1`,
        // `penalty_scale = 1`). The previous code used `penalty_scale = shape`
        // for Gamma, double-counting the dispersion in the sampled posterior
        // and shrinking every posterior SD by `√φ` (#680). Tweedie/NB/Poisson/
        // Binomial are unit-scale and unchanged.
        let penalty_scale = 1.0 / self.cov_scale.max(1e-300);

        // Gradient w.r.t. θ (wiggle): ∂ℓ/∂θ = B(q₀)^T · residual − S_link · θ
        let s_link_theta = self.penalty_link.dot(&theta);
        let grad_theta = &fast_atv(&bwiggle, &residual) - &(&s_link_theta * penalty_scale);

        // Gradient w.r.t. β_eta: ∂ℓ/∂β = X^T · (residual ⊙ g'(q₀)) − S_base · β
        // where g'(q₀) = dη/dq₀ is the chain-rule factor
        let g_prime = self.compute_g_prime(&u, &theta);
        let r_scaled: Array1<f64> = residual
            .iter()
            .zip(g_prime.iter())
            .map(|(&r, &g)| r * g)
            .collect();
        let s_base_beta = self.penalty_base.dot(&beta);
        let grad_beta = &fast_atv(&self.x, &r_scaled) - &(&s_base_beta * penalty_scale);

        // Penalty (also φ-scaled for Gaussian; see `penalty_scale` above).
        let penalty =
            penalty_scale * (0.5 * beta.dot(&s_base_beta) + 0.5 * theta.dot(&s_link_theta));

        // Assemble joint gradient and transform to whitened space
        let mut grad_q = Array1::<f64>::zeros(dim);
        grad_q
            .slice_mut(ndarray::s![0..self.p_base])
            .assign(&grad_beta);
        grad_q
            .slice_mut(ndarray::s![self.p_base..])
            .assign(&grad_theta);
        fast_av_into(&self.chol_t, &grad_q, grad);
        ll - penalty
    }

    /// Get the Cholesky factor L for un-whitening samples.
    pub fn chol(&self) -> &Array2<f64> {
        &self.chol
    }

    /// Get the mode [β_eta; β_wiggle].
    pub fn mode_joint(&self) -> Array1<f64> {
        let dim = self.p_base + self.p_link;
        let mut mode = Array1::<f64>::zeros(dim);
        mode.slice_mut(ndarray::s![0..self.p_base])
            .assign(&self.mode_beta);
        mode.slice_mut(ndarray::s![self.p_base..])
            .assign(&self.mode_theta);
        mode
    }
}

impl HamiltonianTarget<Array1<f64>> for LinkWigglePosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        self.compute_logp_and_grad_into(position, grad)
    }
}

/// Runs NUTS sampling for joint (β_eta, β_wiggle) in a link-wiggle model.
pub fn run_link_wiggle_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_base: ArrayView2<f64>,
    penalty_link: ArrayView2<f64>,
    mode_beta: ArrayView1<f64>,
    mode_theta: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    spline: LinkWiggleSplineArtifacts,
    nuts_family: NutsFamily,
    scale: f64,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    validate_nuts_config(config).map_err(String::from)?;
    let dim = mode_beta.len() + mode_theta.len();
    let target = LinkWigglePosterior::new(
        x,
        y,
        weights,
        penalty_base,
        penalty_link,
        mode_beta,
        mode_theta,
        hessian,
        spline,
        nuts_family,
        scale,
    )?;
    let chol = target.chol().clone();
    let mode_arr = target.mode_joint();

    let initial_positions = jittered_initial_positions(config, dim, 0.1, 0x8C48_0F65_3A2B_D917);

    let mass_cfg = robust_mass_matrix_config(dim, config.nwarmup);
    let (result, run_stats) = run_whitened_nuts_result(
        target,
        &mode_arr,
        &chol,
        initial_positions,
        config,
        dim,
        mass_cfg,
        0x2E31_A4B6_C908_F57D,
        "Link-wiggle NUTS sampling failed",
        Array1::zeros(dim),
        NutsConvergenceThresholds {
            max_rhat: 1.1,
            min_ess: Some(100.0),
        },
    )?;
    log::info!("Link-wiggle NUTS sampling complete: {}", run_stats);

    Ok(result)
}

// ============================================================================
// Joint (β, ρ) HMC for Skewed Posteriors
// ============================================================================
//
// When the Laplace approximation to the marginal likelihood is unreliable
// (high posterior skewness), we bypass LAML entirely and sample from the
// joint posterior p(β, ρ | y) ∝ p(y|β) p(β|ρ) p(ρ).
//
// The joint log-posterior is:
//   log p(β, ρ | y) = ℓ(y|β) + Φ(β) [if Firth]
//                    - 0.5 β'S(ρ)β + 0.5 log|S(ρ)|_+ + log p(ρ) + const
//
// Gradients:
//   ∇_β: ∇_β ℓ + ∇_β Φ(β) [if Firth] - S(ρ) β
//   ∂/∂ρ_k: -0.5 λ_k β'S_k β + 0.5 tr(S_+⁻¹ A_k) + ∂log p(ρ)/∂ρ_k
//
// This completely avoids the Laplace approximation. When Firth bias reduction
// is active, the sampled target also includes the Jeffreys term Φ(β) in
// addition to the smoothing-parameter prior.

/// Directional cubic non-Gaussianity diagnostic for the Laplace approximation.
///
/// For each positive-curvature Hessian eigenpair `(lambda_r, v_r)`, this computes
///
///   gamma_r = T[v_r, v_r, v_r] / lambda_r^(3/2)
///            = Σ_i c_i (x_i^T v_r)^3 / lambda_r^(3/2),
///
/// and reports `max_r |gamma_r|`. This is invariant to arbitrary coordinate
/// relabeling and uses the full directional cubic contraction rather than only
/// diagonal tensor entries.
/// `refine_supremum` controls Phase 2, the cubic power-iteration that sharpens
/// the returned scalar `max_abs` toward the true supremum of `|γ(u)|` over the
/// H-unit sphere (which can exceed the per-eigenvector maximum). That scalar is
/// the ONLY thing Phase 2 affects — the per-direction `directional` vector,
/// which drives [`laplace_trustworthiness_from_skewness`]'s direction selection
/// AND its own internally-recomputed `max_abs_skewness`, comes entirely from
/// Phase 1. The #784 block-local REML correction
/// (`block_local_sampled_correction`) consumes `directional` and uses `max_abs`
/// only for a `> 0` finiteness guard that Phase 1 already satisfies, so it
/// passes `false` and skips Phase 2's multi-probe O(probes·iters·np) refinement
/// on every inner evaluation. Diagnostic callers that report the true supremum
/// pass `true`.
pub fn laplace_directional_cubic_diagnostic(
    hessian: &Array2<f64>,
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    refine_supremum: bool,
) -> Result<(f64, Array1<f64>), String> {
    let p = hessian.nrows();
    if p == 0 || hessian.ncols() != p {
        return Ok((0.0, Array1::zeros(0)));
    }

    let sym_h = (hessian + &hessian.t()) * 0.5;
    let (evals, evecs) = sym_h
        .eigh(Side::Lower)
        .map_err(|e| format!("directional cubic diagnostic eigendecomposition failed: {e}"))?;
    let max_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_eval * 1.0e-12).max(1.0e-14);
    let mut directional = Array1::<f64>::zeros(p);
    let mut max_abs = 0.0_f64;

    // Build the whitening transform L^{-1} where H = L L^T, so that
    // the standardized cubic along whitened direction u is:
    //   gamma(u) = T[L^{-T}u, L^{-T}u, L^{-T}u]  for ||u||=1
    // Eigenvector directions v_r satisfy u_r = lambda_r^{1/2} v_r (after
    // appropriate normalization), so gamma_r = T[v_r,v_r,v_r] / lambda_r^{3/2}.

    // Phase 1: evaluate gamma_r for all positive-curvature eigenvectors.
    for r in 0..p {
        let lambda = evals[r];
        if lambda <= tol {
            continue;
        }
        let v = evecs.column(r);
        let gamma = directional_cubic_contraction(design, c_weights, &v) / lambda.powf(1.5);
        directional[r] = if gamma.is_finite() { gamma } else { 0.0 };
        max_abs = max_abs.max(directional[r].abs());
    }

    // Phase 2: power-iteration refinement in whitened space.
    //
    // The supremum of |gamma(u)| over ||u||_H=1 can exceed the max over
    // eigenvectors. We approximate it with a few rounds of cubic power
    // iteration: given current direction v, the gradient of T[v,v,v] w.r.t.
    // v on the H-unit sphere is 3 T[·,v,v] projected onto the tangent space.
    // Since T[·,v,v] = X^T diag(c_i (x_i^T v)^2) which is a matrix-vector
    // product, each iteration is O(np).
    //
    // We seed from the eigenvector with largest |gamma_r| and also from a
    // few random probe directions.
    if refine_supremum && p >= 2 {
        // Build H^{-1/2} columns for whitening: H^{-1/2} = V diag(1/sqrt(lam)) V^T
        // We need it to map whitened u -> original v = H^{-1/2} u, and
        // H^{1/2} to project back: H^{1/2} v = V diag(sqrt(lam)) V^T v.
        let positive_mask: Vec<bool> = evals.iter().map(|&ev| ev > tol).collect();
        let n_pos = positive_mask.iter().filter(|&&m| m).count();
        if n_pos >= 2 {
            let max_abs_from_probes = cubic_power_iteration_refinement(
                design,
                c_weights,
                &evals,
                &evecs,
                &positive_mask,
                n_pos,
            );
            if max_abs_from_probes > max_abs {
                max_abs = max_abs_from_probes;
            }
        }
    }

    Ok((max_abs, directional))
}

/// Compute T[v,v,v] = Σ_i c_i (x_i^T v)^3 for a given direction v.
fn directional_cubic_contraction(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    v: &ArrayView1<f64>,
) -> f64 {
    match design.as_sparse() {
        Some(x_sparse) => {
            let (symbolic, values) = x_sparse.as_ref().parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            let mut row_scores = vec![0.0_f64; x_sparse.nrows()];
            for col in 0..x_sparse.ncols() {
                let coeff = v[col];
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    row_scores[row_idx[ptr]] += values[ptr] * coeff;
                }
            }
            let mut cubic = 0.0_f64;
            for i in 0..row_scores.len().min(c_weights.len()) {
                cubic += c_weights[i] * row_scores[i].powi(3);
            }
            cubic
        }
        None => {
            let x_dense = design.to_dense_cow();
            let x_dense = x_dense.as_ref();
            let mut cubic = 0.0_f64;
            for i in 0..x_dense.nrows().min(c_weights.len()) {
                let proj = x_dense.row(i).dot(v);
                cubic += c_weights[i] * proj.powi(3);
            }
            cubic
        }
    }
}

/// Compute the gradient of T[v,v,v] w.r.t. v:  3 X^T diag(c_i (x_i^T v)^2) 1.
/// More precisely: ∂/∂v T[v,v,v] = 3 Σ_i c_i (x_i^T v)^2 x_i.
fn directional_cubic_gradient(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    let p = v.len();
    match design.as_sparse() {
        Some(x_sparse) => {
            let (symbolic, values) = x_sparse.as_ref().parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            let n = x_sparse.nrows();
            let mut row_scores = vec![0.0_f64; n];
            for col in 0..x_sparse.ncols() {
                let coeff = v[col];
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    row_scores[row_idx[ptr]] += values[ptr] * coeff;
                }
            }
            // quadratic weights: 3 c_i (x_i^T v)^2
            let mut quad_weights = vec![0.0_f64; n];
            for i in 0..n.min(c_weights.len()) {
                quad_weights[i] = 3.0 * c_weights[i] * row_scores[i] * row_scores[i];
            }
            // X^T quad_weights
            let mut grad = Array1::<f64>::zeros(p);
            for col in 0..x_sparse.ncols() {
                let mut acc = 0.0_f64;
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    acc += values[ptr] * quad_weights[row_idx[ptr]];
                }
                grad[col] = acc;
            }
            grad
        }
        None => {
            let x_dense = design.to_dense_cow();
            let x_dense = x_dense.as_ref();
            let n = x_dense.nrows();
            let mut grad = Array1::<f64>::zeros(p);
            for i in 0..n.min(c_weights.len()) {
                let proj = x_dense.row(i).dot(v);
                let w = 3.0 * c_weights[i] * proj * proj;
                // scaled_add works with any ArrayBase reference.
                let row = x_dense.row(i);
                for j in 0..p {
                    grad[j] += w * row[j];
                }
            }
            grad
        }
    }
}

/// Power-iteration refinement for the supremum of |gamma(u)| over ||u||_H = 1.
///
/// Seeds from the best eigenvector direction plus deterministic probe
/// directions constructed from pairs of eigenvectors. Runs a few Riemannian
/// gradient ascent steps on the whitened unit sphere.
fn cubic_power_iteration_refinement(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    positive_mask: &[bool],
    n_pos: usize,
) -> f64 {
    let p = evals.len();
    let max_probes = 8;
    let max_iters = 5;

    // Helper: convert whitened u -> original v = Σ_r (u_r / sqrt(lam_r)) * evec_r
    // (only over positive eigenspace).
    let to_original = |u: &Array1<f64>| -> Array1<f64> {
        let mut v = Array1::<f64>::zeros(p);
        let mut idx = 0;
        for r in 0..p {
            if positive_mask[r] {
                let scale = u[idx] / evals[r].sqrt();
                let col = evecs.column(r);
                for j in 0..p {
                    v[j] += scale * col[j];
                }
                idx += 1;
            }
        }
        v
    };

    // Helper: project original-space vector to whitened: u_j = sqrt(lam_r) (evec_r^T g)
    let to_whitened = |g: &Array1<f64>| -> Array1<f64> {
        let mut u = Array1::<f64>::zeros(n_pos);
        let mut idx = 0;
        for r in 0..p {
            if positive_mask[r] {
                u[idx] = evals[r].sqrt() * evecs.column(r).dot(g);
                idx += 1;
            }
        }
        u
    };

    // Evaluate |gamma(u)| for whitened direction u.
    let eval_gamma = |u: &Array1<f64>| -> f64 {
        let norm = u.dot(u).sqrt();
        if norm < 1e-30 {
            return 0.0;
        }
        let u_normed: Array1<f64> = u / norm;
        let v = to_original(&u_normed);
        // gamma = T[v,v,v] since v already has ||v||_H = 1
        let cubic = directional_cubic_contraction(design, c_weights, &v.view());
        if cubic.is_finite() { cubic.abs() } else { 0.0 }
    };

    // One step of Riemannian gradient ascent on the whitened sphere for |T[v,v,v]|.
    let refine_step = |u: &Array1<f64>| -> Array1<f64> {
        let norm = u.dot(u).sqrt();
        if norm < 1e-30 {
            return u.clone();
        }
        let u_normed: Array1<f64> = u / norm;
        let v = to_original(&u_normed);
        // Gradient of T[v,v,v] w.r.t. v in original space
        let grad_v = directional_cubic_gradient(design, c_weights, &v);
        // Map to whitened space
        let mut grad_u = to_whitened(&grad_v);
        // Project onto tangent plane of sphere: grad - (grad . u) u
        let dot = grad_u.dot(&u_normed);
        grad_u.scaled_add(-dot, &u_normed);
        // Sign: we want to maximize |T|, so follow sign(T) * grad
        let cubic_val = directional_cubic_contraction(design, c_weights, &v.view());
        let sign = if cubic_val >= 0.0 { 1.0 } else { -1.0 };
        let step_size = 0.3;
        let mut u_new = &u_normed + &(&grad_u * (sign * step_size));
        let new_norm = u_new.dot(&u_new).sqrt();
        if new_norm > 1e-30 {
            u_new /= new_norm;
        }
        u_new
    };

    let mut best = 0.0_f64;

    // Build seed directions:
    // (a) The eigenvector with largest |gamma_r| (already computed by caller,
    //     but we re-derive the whitened form here).
    // (b) Deterministic probe directions from pairs of top eigenvectors:
    //     (e_i + e_j) / sqrt(2) and (e_i - e_j) / sqrt(2) in whitened space.
    let mut seeds: Vec<Array1<f64>> = Vec::with_capacity(max_probes);

    // Seed (a): each eigenvector is a standard basis vector in whitened space.
    // Find the one with largest |gamma|.
    let mut best_eig_idx = 0;
    let mut best_eig_gamma = 0.0_f64;
    for j in 0..n_pos {
        let mut u = Array1::<f64>::zeros(n_pos);
        u[j] = 1.0;
        let g = eval_gamma(&u);
        if g > best_eig_gamma {
            best_eig_gamma = g;
            best_eig_idx = j;
        }
    }
    best = best.max(best_eig_gamma);
    let mut u_best = Array1::<f64>::zeros(n_pos);
    u_best[best_eig_idx] = 1.0;
    seeds.push(u_best);

    // Seed (b): pairwise combinations of the top few eigenvectors.
    let n_top = n_pos.min(4);
    for i in 0..n_top {
        for j in (i + 1)..n_top {
            if seeds.len() >= max_probes {
                break;
            }
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let mut u_plus = Array1::<f64>::zeros(n_pos);
            u_plus[i] = inv_sqrt2;
            u_plus[j] = inv_sqrt2;
            seeds.push(u_plus);
            if seeds.len() < max_probes {
                let mut u_minus = Array1::<f64>::zeros(n_pos);
                u_minus[i] = inv_sqrt2;
                u_minus[j] = -inv_sqrt2;
                seeds.push(u_minus);
            }
        }
    }

    // Run power iteration from each seed.
    for seed in &seeds {
        let mut u = seed.clone();
        for _ in 0..max_iters {
            u = refine_step(&u);
        }
        let g = eval_gamma(&u);
        best = best.max(g);
    }

    best
}

// ───────────────── #1521 laplace-sampler contract re-exports ─────────────────
//
// The neutral DATA carriers + the caller-supplied [`BlockExcessTarget`]
// evaluator + the pure threshold math were contract-downed to the neutral
// `gam-problem` crate (#1521) so gam-solve (whose `Gam784BlockTarget`
// IMPLEMENTS `BlockExcessTarget`) and this gam-inference-tier sampler share one
// set of types without an SCC edge. The COMPUTATION (NUTS, importance sampling,
// the directional-cubic eigen diagnostic) stays UP in this module and
// constructs these types under their original names via this re-export.
pub use gam_problem::laplace_sampler_contract::{
    BlockExcessTarget, BlockSampledMarginal, BlockSampledMoments, GaussianModePosterior,
    LaplaceTrustworthiness, laplace_skewness_threshold, laplace_trustworthiness_from_skewness,
};

/// Monolith (gam-inference-tier) implementor of the contract-downed
/// [`LaplaceMarginalSampler`](gam_problem::laplace_sampler_contract::LaplaceMarginalSampler):
/// wraps the `hmc_io` directional-cubic eigen diagnostic and the
/// importance-sampled #784 block correction. Registered at process init via
/// `gam_problem::laplace_sampler_contract::set_laplace_marginal_sampler`.
pub struct HmcIoLaplaceMarginalSampler;

impl gam_problem::laplace_sampler_contract::LaplaceMarginalSampler for HmcIoLaplaceMarginalSampler {
    fn directional_cubic_diagnostic(
        &self,
        hessian: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
        refine_supremum: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        laplace_directional_cubic_diagnostic(hessian, design, c_weights, refine_supremum)
    }

    fn block_sampled_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
    ) -> Result<BlockSampledMarginal, String> {
        block_sampled_marginal_correction(target)
    }
}

/// Monolith (gam-inference-tier) implementor of the contract-downed
/// [`GaussianModePosteriorSampler`](gam_problem::laplace_sampler_contract::GaussianModePosteriorSampler):
/// the never-fail Gaussian mode-posterior rung. Builds the NUTS config from the
/// problem dimension internally (so `NutsConfig` never crosses the contract)
/// and wraps `hmc_io::sample_gaussian_mode_posterior`. Registered at process
/// init via `gam_problem::laplace_sampler_contract::set_gaussian_mode_posterior_sampler`.
pub struct HmcIoGaussianModePosteriorSampler;

impl gam_problem::laplace_sampler_contract::GaussianModePosteriorSampler
    for HmcIoGaussianModePosteriorSampler
{
    fn sample_gaussian_mode_posterior(
        &self,
        mode: ArrayView1<f64>,
        precision: ArrayView2<f64>,
    ) -> Result<GaussianModePosterior, String> {
        let config = NutsConfig::for_dimension(mode.len());
        sample_gaussian_mode_posterior(mode, precision, &config)
    }
}

/// Auto-derive the number of importance draws for the block-local sampled
/// marginalization from the block dimension.  MAGIC: more directions need more
/// draws to control the importance-weight variance, but the block is small by
/// construction (only the curvature-heavy directions), so this stays cheap.
/// No CLI flag.
fn block_sampling_draws(block_dim: usize) -> usize {
    // Base budget plus a per-direction allowance; capped so a pathological
    // block can never make a single inner evaluation explode.
    const BASE: usize = 256;
    const PER_DIM: usize = 256;
    const CAP: usize = 4096;
    (BASE + PER_DIM * block_dim).min(CAP)
}

/// Estimate the block-local sampled marginal correction `Δ_b` and its
/// ρ-gradient by importance sampling against the local Laplace Gaussian
/// (issue #784).
///
/// # Math
///
/// Draw `t_s ~ q = N(0, diag(1/λ_r))` (the local Laplace Gaussian in the block
/// subspace; whitened draws `z_s ~ N(0, I)` give `t_{s,r} = z_{s,r}/√λ_r`).
/// With the non-Gaussian remainder `ΔF` defined on [`BlockExcessTarget`],
///
///   exp(Δ_b) = E_q[ exp(−ΔF(t)) ]  ⇒  Δ_b = log mean_s exp(−ΔF(t_s)),
///
/// computed via a numerically-stable log-mean-exp.  The ρ-gradient follows
/// from differentiating `Δ_b = log E_q[e^{−ΔF}]` (the `q`-Gaussian normalizer
/// `½Σ log(2π/λ_r)` cancels against `A_Lap`, leaving only the `ΔF` channel):
///
///   ∂Δ_b/∂ρ_k = E_p[ −∂ΔF/∂ρ_k ],   p ∝ q·e^{−ΔF},
///
/// i.e. the self-normalized importance-weighted average of `−∂ΔF/∂ρ_k` over the
/// same draws.  Because value and gradient come from one set of draws and one
/// target, they are mutually consistent — the contract the outer REML needs.
///
/// Determinism: draws come from a fixed-seed RNG so the inner evaluation is a
/// pure function of `(β̂, H, ρ)` and the outer optimizer sees a smooth,
/// reproducible objective rather than Monte-Carlo jitter across evaluations.
pub fn block_sampled_marginal_correction<T: BlockExcessTarget + ?Sized>(
    target: &T,
) -> Result<BlockSampledMarginal, String> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let m = target.block_dim();
    let k = target.rho_dim();
    if m == 0 {
        return Ok(BlockSampledMarginal {
            value: 0.0,
            rho_gradient: Array1::zeros(k),
            importance_ess: 0.0,
            n_draws: 0,
            moments: None,
        });
    }
    let lambdas = target.block_curvatures();
    if lambdas.len() != m {
        return Err(format!(
            "block_sampled_marginal_correction: block_curvatures len {} != block_dim {m}",
            lambdas.len()
        ));
    }
    let inv_sqrt_lambda: Array1<f64> = lambdas.mapv(|l| {
        if l > 0.0 {
            1.0 / l.sqrt()
        } else {
            // A non-positive block curvature means the mode is not a strict
            // minimum in this direction; the Laplace Gaussian is undefined
            // there. Reject rather than fabricate a correction.
            f64::NAN
        }
    });
    if inv_sqrt_lambda.iter().any(|v| !v.is_finite()) {
        return Err(
            "block_sampled_marginal_correction: non-positive block curvature (mode is not a \
             strict local minimum in a sampled direction)"
                .to_string(),
        );
    }

    let n_draws = block_sampling_draws(m);
    // ρ-invariant fixed seed → deterministic AND smooth-in-ρ objective.
    //
    // The doc comment above promises "the outer optimizer sees a smooth,
    // reproducible objective rather than Monte-Carlo jitter across
    // evaluations." That smoothness holds only if the importance draws
    // `z_s` themselves do NOT depend on ρ — ρ may enter the estimator
    // only through the per-sample importance weights `exp(−ΔF(t_s))` and
    // the rescaling `t_s = z_s / √λ_r`, both of which are continuous in
    // ρ for fixed `z_s`. A seed mixed from `λ_r = exp(ρ_k)` (or any
    // other ρ-dependent quantity such as the H-eigenvalues) permutes
    // `z_s` for every ρ probe, so the FD `(F(ρ+h) − F(ρ−h))/2h`
    // identity fails by O(MC_stdev/h) — exactly the order-10²–10³ FD
    // blow-up observed in the iso-κ Duchon binomial FD probes — and
    // every outer trust-region step lands on a different random face of
    // the objective. Mix only the (ρ-invariant) block / outer dimensions
    // so different problems still get independent streams.
    let mut seed_bits: u64 = 0x9E37_79B9_7F4A_7C15;
    seed_bits ^= (m as u64).rotate_left(17);
    seed_bits = seed_bits.wrapping_mul(0x1000_0000_01B3);
    seed_bits ^= (k as u64).rotate_left(31);
    seed_bits = seed_bits.wrapping_mul(0x1000_0000_01B3);
    let mut rng = StdRng::seed_from_u64(seed_bits);

    // Streaming, numerically-stable accumulation of the log-mean-exp value,
    // the explicit gradient channel `E_p[−∂ΔF/∂ρ]`, AND the gradient-channel
    // moments `E_p[t]`, `E_p[t tᵀ]`, `E_p[ngs]`, `E_p[t ⊗ ngs]` needed by the
    // exact (b)–(d) channel assembly (gradient exactness contract above).
    // Weights are kept relative to a running maximum log-weight: whenever a
    // new maximum arrives, every accumulator is rescaled by
    // `exp(max_old − max_new) ≤ 1`, so each per-draw relative weight is ≤ 1
    // and the sums never overflow. Infeasible / divergent draws contribute
    // zero weight rather than poisoning the estimate.
    let n_obs = target.base_neg_score().len();
    let mut max_lw = f64::NEG_INFINITY;
    let mut sum_w = 0.0_f64;
    let mut sum_w2 = 0.0_f64;
    let mut grad_acc = Array1::<f64>::zeros(k);
    let mut e_t_acc = Array1::<f64>::zeros(m);
    let mut e_tt_acc = Array2::<f64>::zeros((m, m));
    let mut e_ngs_acc = Array1::<f64>::zeros(n_obs);
    let mut e_t_ngs_acc = Array2::<f64>::zeros((n_obs, m));

    // Pre-generate ALL whitened draws into the columns of `draws` (m × n_draws)
    // in the EXACT same RNG order as the serial loop (draw 0: r=0..m, draw 1:
    // r=0..m, …). The per-draw design matvec `s = X_t·(V_b·t_s)` is then batched
    // into two BLAS-3 products over all columns at once (the #1082 hot path),
    // instead of n_draws separate BLAS-2 matvecs — the draws, seed, budget, and
    // importance weights are byte-for-byte unchanged; only the matvecs are
    // reassociated into a GEMM.
    let mut draws = Array2::<f64>::zeros((m, n_draws));
    for s in 0..n_draws {
        let mut col = draws.column_mut(s);
        for r in 0..m {
            let z = sample_standard_normal(&mut rng);
            col[r] = z * inv_sqrt_lambda[r];
        }
    }
    let batched = target.excess_with_displaced_neg_score_batch(&draws);

    let mut t = Array1::<f64>::zeros(m);
    for (sidx, (excess, displaced_ngs)) in batched.into_iter().enumerate() {
        t.assign(&draws.column(sidx));
        if !excess.is_finite() {
            continue;
        }
        let Some(ngs) = displaced_ngs else {
            // A finite excess always carries a score; absence means infeasible.
            continue;
        };
        let lw = -excess;
        if lw > max_lw {
            // exp(−∞ − lw) = 0 zeroes the (empty) accumulators on the first
            // feasible draw, so no special-casing is needed.
            let rescale = (max_lw - lw).exp();
            sum_w *= rescale;
            sum_w2 *= rescale * rescale;
            grad_acc *= rescale;
            e_t_acc *= rescale;
            e_tt_acc *= rescale;
            e_ngs_acc *= rescale;
            e_t_ngs_acc *= rescale;
            max_lw = lw;
        }
        let w = (lw - max_lw).exp();
        sum_w += w;
        sum_w2 += w * w;
        // Explicit channel: −∂ΔF/∂ρ.
        grad_acc.scaled_add(-w, &target.excess_rho_gradient(&t));
        // Moment channels (score already computed in the fused call above).
        if ngs.len() != n_obs {
            return Err(format!(
                "block_sampled_marginal_correction: displaced_neg_score len {} != {n_obs}",
                ngs.len()
            ));
        }
        e_t_acc.scaled_add(w, &t);
        e_ngs_acc.scaled_add(w, &ngs);
        for r in 0..m {
            let wt_r = w * t[r];
            for q in 0..m {
                e_tt_acc[(q, r)] += wt_r * t[q];
            }
            e_t_ngs_acc.column_mut(r).scaled_add(wt_r, &ngs);
        }
    }
    if !max_lw.is_finite() {
        return Err(
            "block_sampled_marginal_correction: all importance draws were infeasible".to_string(),
        );
    }
    let value = max_lw + (sum_w / n_draws as f64).ln();
    // Self-normalized importance-weighted gradient E_p[−∂ΔF/∂ρ] and moments.
    let (rho_gradient, moments) = if sum_w > 0.0 {
        (
            grad_acc / sum_w,
            Some(BlockSampledMoments {
                e_t: e_t_acc / sum_w,
                e_tt: e_tt_acc / sum_w,
                e_neg_score: e_ngs_acc / sum_w,
                e_t_neg_score: e_t_ngs_acc / sum_w,
            }),
        )
    } else {
        (Array1::zeros(k), None)
    };
    // Kish effective sample size of the importance weights.
    let importance_ess = if sum_w2 > 0.0 {
        (sum_w * sum_w) / sum_w2
    } else {
        0.0
    };

    if !value.is_finite() || rho_gradient.iter().any(|v| !v.is_finite()) {
        return Err(
            "block_sampled_marginal_correction: produced a non-finite correction or gradient"
                .to_string(),
        );
    }
    if let Some(mo) = moments.as_ref()
        && (mo.e_t.iter().any(|v| !v.is_finite())
            || mo.e_tt.iter().any(|v| !v.is_finite())
            || mo.e_neg_score.iter().any(|v| !v.is_finite())
            || mo.e_t_neg_score.iter().any(|v| !v.is_finite()))
    {
        return Err(
            "block_sampled_marginal_correction: produced non-finite gradient-channel moments"
                .to_string(),
        );
    }

    Ok(BlockSampledMarginal {
        value,
        rho_gradient,
        importance_ess,
        n_draws,
        moments,
    })
}

/// Result of joint (β, ρ) sampling.
#[derive(Clone, Debug)]
pub struct JointBetaRhoResult {
    /// Coefficient samples: shape (n_total_samples, n_beta)
    pub beta_samples: Array2<f64>,
    /// Log-smoothing parameter samples: shape (n_total_samples, n_rho)
    pub rho_samples: Array2<f64>,
    /// Posterior mean of β
    pub beta_mean: Array1<f64>,
    /// Adaptive inverse-link parameter samples: shape (n_total_samples, n_link_params)
    pub link_param_samples: Array2<f64>,
    /// Posterior mean of adaptive inverse-link parameters
    pub link_param_mean: Array1<f64>,
    /// Posterior mean of ρ
    pub rho_mean: Array1<f64>,
    /// R-hat diagnostic
    pub rhat: f64,
    /// Effective sample size
    pub ess: f64,
    /// Whether sampling converged
    pub converged: bool,
    /// Max skewness that triggered this sampling
    pub trigger_skewness: f64,
}

/// Joint (β, ρ) posterior target for NUTS.
///
/// Samples from p(β, ρ | y) ∝ p(y|β) p(β|ρ) p(ρ) directly,
/// completely bypassing the Laplace approximation.
///
/// The parameter vector is [z_β; ρ] where z_β = L⁻¹(β - μ) is the
/// whitened β, ρ is the raw log-smoothing parameters, and adaptive inverse-link
/// parameters follow when the binomial link has fitted shape/mixing parameters.
struct JointBetaRhoPosterior {
    data: SharedData,
    /// L where LL' = H⁻¹ (whitening for β block)
    chol: Array2<f64>,
    /// L' for chain rule
    chol_t: Array2<f64>,
    /// Joint likelihood specification (response + parameterized link).
    likelihood: LikelihoodSpec,
    /// Dimension of β
    n_beta: usize,
    /// Dimension of ρ
    n_rho: usize,
    /// Dimension of adaptive inverse-link parameters
    n_link_params: usize,
    /// LAML-converged adaptive inverse-link parameters (used only to initialize chains)
    link_param_mode: Array1<f64>,
    /// Canonical penalties in the transformed basis.
    penalty_canonical: Vec<gam_terms::construction::CanonicalPenalty>,
    /// Fixed prior on rho used by the sampled target.
    rho_prior: RhoPrior,
    /// LAML-converged ρ (used only to initialize chains)
    rho_mode: Array1<f64>,
    /// Whether to add the identifiable-subspace Jeffreys/Firth term to the
    /// target
    firth_enabled: bool,
    /// One-deep cache for the structural penalty pseudo-logdet and its
    /// ρ-gradient. NUTS tree-doubling and U-turn checks repeatedly evaluate
    /// the joint log-posterior at the same `rho` bytes, so a single-slot
    /// cache keyed on the exact f64 bit pattern of `rho` avoids redundant
    /// SVD/eigendecompositions inside `PenaltyPseudologdet::from_penalties`.
    /// `Mutex` (not `RefCell`) because chains share the target via
    /// `Arc<Target>` and run in parallel via rayon.
    penalty_logdet_cache: Mutex<Option<(u64, f64, Array1<f64>)>>,
}

impl JointBetaRhoPosterior {
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        penalty_canonical: Vec<gam_terms::construction::CanonicalPenalty>,
        rho_mode: ArrayView1<f64>,
        likelihood: LikelihoodSpec,
        gamma_shape: Option<f64>,
        dispersion: gam_solve::model_types::Dispersion,
        offset: Option<ArrayView1<f64>>,
        rho_prior: RhoPrior,
        firth_enabled: bool,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let n_beta = x.ncols();
        let n_rho = penalty_canonical.len();

        if let Some(offset) = offset.as_ref() {
            if offset.len() != n_samples {
                return Err(HmcError::DimensionMismatch {
                    reason: format!(
                        "Joint HMC offset length {} does not match {} observations",
                        offset.len(),
                        n_samples
                    ),
                }
                .into());
            }
            if !offset.iter().all(|v| v.is_finite()) {
                return Err(HmcError::NonFiniteState {
                    reason: "Joint HMC offset contains NaN or Inf values".to_string(),
                }
                .into());
            }
        }

        if rho_mode.len() != n_rho {
            return Err(HmcError::DimensionMismatch {
                reason: format!(
                    "rho_mode length {} != penalty count {}",
                    rho_mode.len(),
                    n_rho
                ),
            }
            .into());
        }

        match (&likelihood.response, &likelihood.link) {
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {}
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {}
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {}
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_)) => {}
            (ResponseFamily::Binomial, InverseLink::Sas(_)) => {}
            (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => {}
            (ResponseFamily::Binomial, InverseLink::Mixture(_)) => {}
            (ResponseFamily::Binomial, InverseLink::Standard(other)) => {
                return Err(HmcError::LinkMismatch {
                    reason: format!(
                        "Joint HMC binomial response requires a binomial-compatible inverse link; got {:?}",
                        other
                    ),
                }
                .into());
            }
            (ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity)) => {}
            (ResponseFamily::Gaussian, _) => {
                return Err(HmcError::LinkMismatch {
                    reason: "Joint HMC Gaussian requires an identity inverse link".to_string(),
                }
                .into());
            }
            (
                ResponseFamily::Poisson
                | ResponseFamily::Tweedie { .. }
                | ResponseFamily::NegativeBinomial { .. }
                | ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            ) => {}
            (
                ResponseFamily::Poisson
                | ResponseFamily::Tweedie { .. }
                | ResponseFamily::NegativeBinomial { .. }
                | ResponseFamily::Gamma,
                _,
            ) => {
                return Err(HmcError::LinkMismatch {
                    reason: "Joint HMC log-link family requires a log inverse link".to_string(),
                }
                .into());
            }
            (ResponseFamily::Beta { .. }, InverseLink::Standard(StandardLink::Logit)) => {}
            (ResponseFamily::Beta { .. }, _) => {
                return Err(HmcError::LinkMismatch {
                    reason: "Joint HMC Beta requires a logit inverse link".to_string(),
                }
                .into());
            }
            (ResponseFamily::RoystonParmar, _) => {
                return Err(HmcError::UnsupportedFamily {
                    reason: "Joint HMC fallback is not implemented for RoystonParmar".to_string(),
                }
                .into());
            }
        }

        validate_firth_likelihood_support(&likelihood, firth_enabled).map_err(String::from)?;
        if matches!(likelihood.response, ResponseFamily::NegativeBinomial { .. }) {
            validate_count_responses("negative-binomial joint HMC", &y, &weights)
                .map_err(String::from)?;
        }
        if likelihood.is_binomial() {
            validate_binary_responses("binomial joint HMC", &y, &weights).map_err(String::from)?;
        }

        let whitening = hessian_whitening_transform(
            hessian,
            n_beta,
            1.0,
            "Joint HMC: Hessian Cholesky failed",
        )?;
        let chol = whitening.chol;
        let chol_t = whitening.chol_t;

        let data = SharedData {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            mode: Arc::new(mode.to_owned()),
            // The fit's offset and dispersion are fixed likelihood state: the
            // joint target must retain both or it samples a different model —
            // hard-coding φ = 1 gave a Gaussian fit with σ² = 4 four times its
            // true likelihood curvature, and dropping the offset shifted every
            // offset model's posterior (finding 18, #2245). The family kernels
            // read `data.dispersion` (Gaussian 1/φ scaling, Tweedie 1/φ
            // quasi-weight); families whose working weight already carries the
            // dispersion pass `Known(1.0)` from the caller.
            offset: offset.map(|o| Arc::new(o.to_owned())),
            gamma_shape: gamma_shape.unwrap_or(1.0),
            dispersion,
            n_samples,
            dim: n_beta,
        };
        let link_param_mode = Self::link_param_mode(&likelihood.link);

        Ok(Self {
            data,
            chol,
            chol_t,
            likelihood,
            n_beta,
            n_rho,
            n_link_params: link_param_mode.len(),
            link_param_mode,
            penalty_canonical,
            rho_prior,
            rho_mode: rho_mode.to_owned(),
            firth_enabled,
            penalty_logdet_cache: Mutex::new(None),
        })
    }

    /// FNV-1a hash over the raw f64 bit pattern of `rho`.
    ///
    /// NUTS leapfrog / tree-doubling / U-turn checks revisit identical
    /// position vectors byte-for-byte, so exact-equality on `to_bits()`
    /// captures the dominant repetition pattern without any tolerance.
    #[inline]
    fn hash_rho(rho: ndarray::ArrayView1<f64>) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in rho.iter() {
            h ^= x.to_bits();
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn link_param_mode(inverse_link: &InverseLink) -> Array1<f64> {
        match inverse_link {
            InverseLink::Sas(state) | InverseLink::BetaLogistic(state) => {
                Array1::from_vec(vec![state.epsilon, state.log_delta])
            }
            InverseLink::Mixture(state) => state.rho.clone(),
            InverseLink::Standard(_) | InverseLink::LatentCLogLog(_) => Array1::zeros(0),
        }
    }

    fn inverse_link_with_params(
        &self,
        link_params: ndarray::ArrayView1<'_, f64>,
    ) -> Result<InverseLink, String> {
        match &self.likelihood.link {
            InverseLink::Sas(_) => {
                if link_params.len() != 2 {
                    return Err(format!(
                        "SAS link parameter length must be 2, got {}",
                        link_params.len()
                    ));
                }
                Ok(InverseLink::Sas(
                    gam_solve::mixture_link::sas_link_state_from_raw(
                        link_params[0],
                        link_params[1],
                    )?,
                ))
            }
            InverseLink::BetaLogistic(_) => {
                if link_params.len() != 2 || !link_params.iter().all(|v| v.is_finite()) {
                    return Err(
                        "Beta-Logistic link parameters must be finite with length 2".to_string()
                    );
                }
                Ok(InverseLink::BetaLogistic(
                    gam_problem::types::SasLinkState {
                        epsilon: link_params[0],
                        log_delta: link_params[1],
                        delta: link_params[1].exp(),
                    },
                ))
            }
            InverseLink::Mixture(state) => {
                let rho = link_params.to_owned();
                Ok(InverseLink::Mixture(gam_problem::types::MixtureLinkState {
                    components: state.components.clone(),
                    pi: softmax_last_fixedzero(&rho),
                    rho,
                }))
            }
            InverseLink::Standard(_) | InverseLink::LatentCLogLog(_) => {
                Ok(self.likelihood.link.clone())
            }
        }
    }

    /// Compute the joint log-posterior and gradient.
    ///
    /// The joint log-posterior is:
    ///   log p(β, ρ | y) = ℓ(y|β) + ½ log|I(β)| [if Firth]
    ///                    − ½β'S(ρ)β + ½ log|S(ρ)|₊ + log p(ρ) + const
    ///
    /// This is NOT the REML/LAML objective (which integrates out β). Here β is
    /// an explicit parameter being sampled, evaluated at arbitrary values — not
    /// just at the mode β̂(ρ).
    ///
    /// Parameter vector layout: [z_β (whitened, length n_beta); ρ (length n_rho);
    /// adaptive inverse-link params (length n_link_params)]
    fn compute_joint_logp_and_grad_into(
        &self,
        params: &Array1<f64>,
        out_grad: &mut Array1<f64>,
    ) -> f64 {
        let n_beta = self.n_beta;
        let n_rho = self.n_rho;
        let n_link_params = self.n_link_params;

        // Split parameter vector — keep as views to avoid two per-step
        // `to_owned()` allocations of size n_beta and n_rho.
        let z = params.slice(ndarray::s![..n_beta]);
        let rho = params.slice(ndarray::s![n_beta..n_beta + n_rho]);
        let link_params = params.slice(ndarray::s![n_beta + n_rho..]);
        let lambdas: Array1<f64> = rho.mapv(f64::exp);

        let inverse_link = match self.inverse_link_with_params(link_params) {
            Ok(link) => link,
            Err(err) => {
                log::warn!(
                    "[Joint HMC] adaptive inverse-link parameters are invalid: {}",
                    err
                );
                out_grad.fill(0.0);
                return f64::NEG_INFINITY;
            }
        };

        // Un-whiten: β = μ + L z
        let beta = self.data.mode.as_ref() + &self.chol.dot(&z);

        // η = X β (+ fixed fit-time offset)
        let mut eta = gam_linalg::faer_ndarray::fast_av(self.data.x.as_ref(), &beta);
        if let Some(offset) = self.data.offset.as_ref() {
            eta += offset.as_ref();
        }

        // ---- Log-likelihood ℓ(y|β) and ∇_β ℓ ----
        let step_likelihood = LikelihoodSpec {
            response: self.likelihood.response.clone(),
            link: inverse_link,
        };
        let (ll, mut grad_ll_beta, grad_link) = match joint_family_logp_grad_and_link_grad(
            &step_likelihood,
            &self.data,
            &eta,
            n_link_params,
        ) {
            Ok(value) => value,
            Err(err) => {
                log::warn!(
                    "[Joint HMC] likelihood target became invalid at the current state: {}",
                    err
                );
                out_grad.fill(0.0);
                return f64::NEG_INFINITY;
            }
        };

        let mut firth_logdet = 0.0;
        if self.firth_enabled {
            // The Jeffreys determinant must use the *sampled* inverse link's
            // Fisher-weight jet — the same `step_likelihood` (with the current
            // adaptive link parameters) the log-likelihood above was evaluated
            // under. Hard-coding logit gave probit/cloglog/SAS/mixture targets
            // the wrong determinant and gradient (finding 19, #2245).
            match firth_jeffreys_logp_and_grad(&step_likelihood, &self.data, &eta) {
                Ok((value, grad_beta_firth)) => {
                    firth_logdet = value;
                    grad_ll_beta += &grad_beta_firth;
                }
                Err(err) => {
                    log::warn!(
                        "[Joint HMC/Firth] Jeffreys target became invalid at the current state: {}",
                        err
                    );
                    out_grad.fill(0.0);
                    return f64::NEG_INFINITY;
                }
            }
        }

        // ---- Penalty: -0.5 β'S(ρ)β ----
        // S(ρ) = Σ_k λ_k S_k where S_k = R_k'R_k (precomputed in penalty_matrices).
        // Uses penalty_roots for the efficient ||R_k β||² form.
        let mut penalty_val = 0.0;
        let mut s_beta = Array1::<f64>::zeros(n_beta);
        let mut grad_rho = Array1::<f64>::zeros(n_rho);

        // Reuse one max-rank scratch buffer for r_beta = R_k · β_block across
        // all penalty blocks instead of allocating a fresh Array1 per block
        // per HMC step.
        let max_rank = self
            .penalty_canonical
            .iter()
            .map(|cp| cp.rank())
            .max()
            .unwrap_or(0);
        let mut r_beta_scratch = Array1::<f64>::zeros(max_rank);

        for (k, cp) in self.penalty_canonical.iter().enumerate() {
            // Block-local quadratic: β'S_k β via root
            let r = &cp.col_range;
            let beta_block = beta.slice(ndarray::s![r.start..r.end]);
            let rank_k = cp.rank();
            gam_linalg::faer_ndarray::fast_av_view_into(
                &cp.root,
                &beta_block,
                r_beta_scratch.slice_mut(ndarray::s![..rank_k]),
            );
            let r_beta = r_beta_scratch.slice(ndarray::s![..rank_k]);
            let quad_k = r_beta.dot(&r_beta);
            penalty_val += 0.5 * lambdas[k] * quad_k;

            // Accumulate S(ρ)β for β-gradient — block-local
            for a in 0..cp.block_dim() {
                let val: f64 = (0..rank_k).map(|row| cp.root[[row, a]] * r_beta[row]).sum();
                s_beta[r.start + a] += lambdas[k] * val;
            }

            // ρ_k gradient from penalty
            grad_rho[k] = -0.5 * lambdas[k] * quad_k;
        }

        // ---- Structural penalty log-determinant: +0.5 log|S(ρ)|₊ and ρ-derivatives ----
        //
        // One-deep cache keyed on the exact f64 bits of `rho`: NUTS tree
        // doubling revisits identical positions byte-for-byte, so an
        // exact-equality cache eliminates the dominant SVD/eigendecomp
        // cost in `PenaltyPseudologdet::from_penalties` across leapfrog
        // half-steps.
        let log_det_s = if self.penalty_canonical.is_empty() {
            0.0
        } else {
            let rho_hash = Self::hash_rho(rho);
            let cached = self.penalty_logdet_cache.lock().ok().and_then(|guard| {
                guard.as_ref().and_then(|(h, v, g)| {
                    if *h == rho_hash && g.len() == n_rho {
                        for k in 0..n_rho {
                            grad_rho[k] += 0.5 * g[k];
                        }
                        Some(*v)
                    } else {
                        None
                    }
                })
            });
            if let Some(hit) = cached {
                hit
            } else {
                match PenaltyPseudologdet::from_penalties(
                    &self.penalty_canonical,
                    lambdas.as_slice().unwrap_or(&[]),
                    0.0,
                    n_beta,
                ) {
                    Ok(pld) => {
                        let (det1, _) = pld.rho_derivatives_from_penalties(
                            &self.penalty_canonical,
                            lambdas.as_slice().unwrap_or(&[]),
                        );
                        let value = pld.value();
                        if let Ok(mut guard) = self.penalty_logdet_cache.lock() {
                            *guard = Some((rho_hash, value, det1.clone()));
                        }
                        for k in 0..n_rho {
                            grad_rho[k] += 0.5 * det1[k];
                        }
                        value
                    }
                    Err(err) => {
                        log::warn!(
                            "[Joint HMC] structural penalty logdet became invalid at the current state: {}",
                            err
                        );
                        out_grad.fill(0.0);
                        return f64::NEG_INFINITY;
                    }
                }
            }
        };

        // ---- Prior on ρ ----
        let mut rho_prior = 0.0;
        match &self.rho_prior {
            RhoPrior::Flat => {}
            RhoPrior::Normal { mean, sd } => {
                let inv_var = 1.0 / (*sd * *sd);
                for k in 0..n_rho {
                    let d = rho[k] - *mean;
                    rho_prior -= 0.5 * inv_var * d * d;
                    grad_rho[k] -= inv_var * d;
                }
            }
            RhoPrior::GammaPrecision { shape, rate } => {
                for k in 0..n_rho {
                    let lambda = rho[k].exp();
                    // Density over sampled rho includes the e^rho Jacobian (Gamma is on lambda = e^rho).
                    rho_prior += *shape * rho[k] - *rate * lambda;
                    grad_rho[k] += *shape - *rate * lambda;
                }
            }
            RhoPrior::PenalizedComplexity { upper, tail_prob } => {
                if !pc_prior_params_valid(*upper, *tail_prob) {
                    out_grad.fill(0.0);
                    return f64::NEG_INFINITY;
                }
                let theta = -tail_prob.ln() / *upper;
                for k in 0..n_rho {
                    // log p(ρ) = const − ρ/2 − θ exp(−ρ/2).
                    let e = (-0.5 * rho[k]).exp();
                    rho_prior += -0.5 * rho[k] - theta * e;
                    grad_rho[k] += -0.5 + 0.5 * theta * e;
                }
            }
            RhoPrior::Independent(priors) => {
                if priors.len() != n_rho {
                    out_grad.fill(0.0);
                    return f64::NEG_INFINITY;
                }
                for k in 0..n_rho {
                    match &priors[k] {
                        RhoPrior::Flat => {}
                        RhoPrior::Normal { mean, sd } => {
                            let inv_var = 1.0 / (*sd * *sd);
                            let d = rho[k] - *mean;
                            rho_prior -= 0.5 * inv_var * d * d;
                            grad_rho[k] -= inv_var * d;
                        }
                        RhoPrior::GammaPrecision { shape, rate } => {
                            let lambda = rho[k].exp();
                            // Density over sampled rho includes the e^rho Jacobian (Gamma is on lambda = e^rho).
                            rho_prior += *shape * rho[k] - *rate * lambda;
                            grad_rho[k] += *shape - *rate * lambda;
                        }
                        RhoPrior::PenalizedComplexity { upper, tail_prob } => {
                            if !pc_prior_params_valid(*upper, *tail_prob) {
                                out_grad.fill(0.0);
                                return f64::NEG_INFINITY;
                            }
                            let theta = -tail_prob.ln() / *upper;
                            let e = (-0.5 * rho[k]).exp();
                            rho_prior += -0.5 * rho[k] - theta * e;
                            grad_rho[k] += -0.5 + 0.5 * theta * e;
                        }
                        RhoPrior::Independent(_) => {
                            out_grad.fill(0.0);
                            return f64::NEG_INFINITY;
                        }
                    }
                }
            }
        }

        // ---- Assemble ----
        let logp = ll + firth_logdet - penalty_val + 0.5 * log_det_s + rho_prior;

        // β-gradient in original space: ∇_β ℓ - S(ρ)β
        let grad_beta = &grad_ll_beta - &s_beta;

        // Combined gradient: [∇_z; ∇_ρ; ∇_link]
        gam_linalg::faer_ndarray::fast_av_view_into(
            &self.chol_t,
            &grad_beta,
            out_grad.slice_mut(ndarray::s![..n_beta]),
        );
        out_grad
            .slice_mut(ndarray::s![n_beta..n_beta + n_rho])
            .assign(&grad_rho);
        out_grad
            .slice_mut(ndarray::s![n_beta + n_rho..])
            .assign(&grad_link);

        logp
    }
}

/// Penalized-complexity hyperparameters are usable iff `upper` is finite and
/// strictly positive and `tail_prob` is a probability in the open `(0, 1)`.
/// Mirrors the validation in the shared `rho_prior_eval` engine; an invalid
/// configuration repels the sampler (`-∞` potential) rather than producing a
/// non-finite gradient.
fn pc_prior_params_valid(upper: f64, tail_prob: f64) -> bool {
    upper.is_finite() && upper > 0.0 && tail_prob.is_finite() && tail_prob > 0.0 && tail_prob < 1.0
}

impl HamiltonianTarget<Array1<f64>> for JointBetaRhoPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        self.compute_joint_logp_and_grad_into(position, grad)
    }
}

/// Inputs for joint (β, ρ) sampling.
pub struct JointBetaRhoInputs<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub likelihood: LikelihoodSpec,
    pub gamma_shape: Option<f64>,
    /// Fitted dispersion φ, exactly as the flat NUTS path carries it: the
    /// estimated σ² for a profiled Gaussian and the Tweedie φ; `Known(1.0)`
    /// for families whose working weight already folds the dispersion in.
    /// The joint target is the fitted model's posterior only if this matches
    /// the fit.
    pub dispersion: gam_solve::model_types::Dispersion,
    /// Fixed additive offset on the linear predictor (η = Xβ + offset), or
    /// `None` for an offset-free fit. Dropping a fit-time offset shifts the
    /// sampled posterior of every offset model.
    pub offset: Option<ArrayView1<'a, f64>>,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
    pub penalty_roots: Vec<CanonicalPenalty>,
    pub rho_mode: ArrayView1<'a, f64>,
    pub rho_prior: RhoPrior,
    pub firth_bias_reduction: bool,
    /// Max posterior skewness that triggered this sampling
    pub trigger_skewness: f64,
}

/// Run joint (β, ρ) NUTS sampling.
///
/// This is the automatic fallback when the Laplace approximation has high
/// skewness. It samples from the true joint posterior, completely bypassing
/// the Laplace approximation for smoothing parameter selection.
pub fn run_joint_beta_rho_sampling(
    inputs: &JointBetaRhoInputs<'_>,
    config: &NutsConfig,
) -> Result<JointBetaRhoResult, String> {
    validate_firth_likelihood_support(&inputs.likelihood, inputs.firth_bias_reduction)
        .map_err(String::from)?;
    validate_nuts_config(config).map_err(String::from)?;
    let n_beta = inputs.mode.len();
    let n_rho = inputs.penalty_roots.len();
    let n_link_params = JointBetaRhoPosterior::link_param_mode(&inputs.likelihood.link).len();
    let total_dim = n_beta + n_rho + n_link_params;

    log::info!(
        "[Joint HMC] Sampling (β, ρ, link) jointly: {} β-params + {} ρ-params + {} link-params = {} total (triggered by skewness {:.3})",
        n_beta,
        n_rho,
        n_link_params,
        total_dim,
        inputs.trigger_skewness,
    );

    let target = JointBetaRhoPosterior::new(
        inputs.x,
        inputs.y,
        inputs.weights,
        inputs.mode,
        inputs.hessian,
        inputs.penalty_roots.clone(),
        inputs.rho_mode,
        inputs.likelihood.clone(),
        inputs.gamma_shape,
        inputs.rho_prior.clone(),
        inputs.firth_bias_reduction,
    )?;

    let chol = target.chol.clone();
    let mode_arr = target.data.mode.clone();
    let rho_mode = target.rho_mode.clone();
    let link_param_mode = target.link_param_mode.clone();

    // Initialize chains: z_β at 0 (= mode), ρ at rho_mode, link params at fitted state.
    let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
        .map(|chain| {
            let mut rng =
                StdRng::seed_from_u64(chain_stream_seed(config.seed, chain, 0x9B51_6E37_F2D0_A48C));
            let mut pos = Array1::<f64>::zeros(total_dim);
            // Small jitter for β (whitened space)
            for j in 0..n_beta {
                pos[j] = sample_standard_normal(&mut rng) * 0.1;
            }
            // Small jitter for ρ around mode
            for k in 0..n_rho {
                pos[n_beta + k] = rho_mode[k] + sample_standard_normal(&mut rng) * 0.2;
            }
            // Small jitter for adaptive link parameters around fitted state
            for k in 0..n_link_params {
                pos[n_beta + n_rho + k] =
                    link_param_mode[k] + sample_standard_normal(&mut rng) * 0.05;
            }
            pos
        })
        .collect();

    // Keep warmup covariance phase-local: diagonal windows are less likely to
    // encode cross-block covariance from a transient mode switch.
    let mass_cfg = robust_mass_matrix_config(total_dim, config.nwarmup);

    let (samples_array, run_stats) = run_whitened_nuts_samples(
        target,
        initial_positions,
        config,
        total_dim,
        mass_cfg,
        0x63AF_175B_D820_C94E,
        "Joint (β,ρ) NUTS sampling failed",
    )?;
    log::info!("[Joint HMC] Sampling complete: {}", run_stats);

    // Unpack samples
    let shape = samples_array.shape();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let beta_samples = unwhiten_samples(&samples_array, mode_arr.as_ref(), &chol, n_beta, 0);
    let mut rho_samples = Array2::<f64>::zeros((total_samples, n_rho));
    let mut link_param_samples = Array2::<f64>::zeros((total_samples, n_link_params));

    for chain in 0..n_chains {
        for sample_i in 0..n_samples_out {
            let sample_idx = chain * n_samples_out + sample_i;
            let zview = samples_array.slice(ndarray::s![chain, sample_i, ..]);

            // ρ and adaptive link parameters are stored directly
            let rho_slice = zview.slice(ndarray::s![n_beta..n_beta + n_rho]);
            rho_samples.row_mut(sample_idx).assign(&rho_slice);
            let link_slice = zview.slice(ndarray::s![n_beta + n_rho..]);
            link_param_samples.row_mut(sample_idx).assign(&link_slice);
        }
    }

    let beta_mean = beta_samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_beta));
    let rho_mean = rho_samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_rho));
    let link_param_mean = link_param_samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_link_params));

    let (rhat, ess) = compute_split_rhat_and_ess(&samples_array);

    let converged = NutsConvergenceThresholds {
        max_rhat: 1.1,
        min_ess: Some(50.0),
    }
    .converged(rhat, ess);
    if !converged {
        log::warn!(
            "[Joint HMC] Convergence warning: R-hat={:.3}, ESS={:.1}",
            rhat,
            ess,
        );
    }

    Ok(JointBetaRhoResult {
        beta_samples,
        rho_samples,
        beta_mean,
        link_param_samples,
        link_param_mean,
        rho_mean,
        rhat,
        ess,
        converged,
        trigger_skewness: inputs.trigger_skewness,
    })
}

// ============================================================================
// Survival Model HMC Support
// ============================================================================

mod survival_hmc {
    use super::*;
    use gam_models::survival::{
        PenaltyBlocks, SurvivalEngineInputs, SurvivalMonotonicityPenalty, SurvivalSpec,
        WorkingModelSurvival,
    };

    /// Shared data for survival NUTS posterior (wrapped in Arc to prevent cloning).
    #[derive(Clone)]
    struct SharedSurvivalData {
        /// Exact survival model in original spline coordinates.
        base_model: Arc<WorkingModelSurvival>,
        /// MAP estimate in coefficient coordinates.
        mode: Arc<Array1<f64>>,
    }

    /// Whitened log-posterior target for survival models with analytical gradients.
    #[derive(Clone)]
    pub struct SurvivalPosterior {
        /// Shared read-only data (Arc prevents duplication)
        data: SharedSurvivalData,
        /// Transform: L where L L^T = H^{-1}
        chol: Array2<f64>,
        /// L^T for gradient chain rule: ∇z = L^T @ ∇_β
        chol_t: Array2<f64>,
    }

    impl SurvivalPosterior {
        /// Creates a new survival posterior target.
        pub fn new(
            age_entry: ArrayView1<'_, f64>,
            age_exit: ArrayView1<'_, f64>,
            event_target: ArrayView1<'_, u8>,
            event_competing: ArrayView1<'_, u8>,
            sampleweight: ArrayView1<'_, f64>,
            x_entry: ArrayView2<'_, f64>,
            x_exit: ArrayView2<'_, f64>,
            x_derivative: ArrayView2<'_, f64>,
            offset_eta_entry: Option<ArrayView1<'_, f64>>,
            offset_eta_exit: Option<ArrayView1<'_, f64>>,
            offset_derivative_exit: Option<ArrayView1<'_, f64>>,
            penalties: PenaltyBlocks,
            monotonicity: SurvivalMonotonicityPenalty,
            spec: SurvivalSpec,
            structurally_monotonic: bool,
            structural_time_columns: usize,
            mode: ArrayView1<f64>,
            hessian: ArrayView2<f64>,
        ) -> Result<Self, String> {
            let n = age_entry.len();
            let off_eta_entry = offset_eta_entry
                .map(|v| v.to_owned())
                .unwrap_or_else(|| Array1::zeros(n));
            let off_eta_exit = offset_eta_exit
                .map(|v| v.to_owned())
                .unwrap_or_else(|| Array1::zeros(n));
            let off_deriv_exit = offset_derivative_exit
                .map(|v| v.to_owned())
                .unwrap_or_else(|| Array1::zeros(n));

            let mut base_model = WorkingModelSurvival::from_engine_inputswith_offsets(
                SurvivalEngineInputs {
                    age_entry,
                    age_exit,
                    event_target,
                    event_competing,
                    sampleweight,
                    x_entry,
                    x_exit,
                    x_derivative,
                    monotonicity_constraint_rows: None,
                    monotonicity_constraint_offsets: None,
                },
                Some(gam_models::survival::SurvivalBaselineOffsets {
                    eta_entry: off_eta_entry.view(),
                    eta_exit: off_eta_exit.view(),
                    derivative_exit: off_deriv_exit.view(),
                }),
                penalties,
                monotonicity,
                spec,
            )
            .map_err(|e| format!("Survival state construction failed: {:?}", e))?;
            if structurally_monotonic {
                base_model
                    .set_structural_monotonicity(true, structural_time_columns)
                    .map_err(|e| {
                        format!("Failed to enable structural monotonicity in survival HMC: {e}")
                    })?;
            }

            let sampler_mode = mode.to_owned();
            let dim = sampler_mode.len();

            let whitening = hessian_whitening_transform(
                hessian,
                dim,
                1.0,
                "Hessian Cholesky decomposition failed",
            )?;
            let chol = whitening.chol;
            let chol_t = whitening.chol_t;

            let data = SharedSurvivalData {
                base_model: Arc::new(base_model),
                mode: Arc::new(sampler_mode),
            };

            Ok(Self { data, chol, chol_t })
        }

        fn compute_logp_and_grad_into(
            &self,
            z: &Array1<f64>,
            grad: &mut Array1<f64>,
        ) -> Result<f64, String> {
            let sampler_position = self.data.mode.as_ref() + &self.chol.dot(z);
            let state = self
                .data
                .base_model
                .update_state(&sampler_position)
                .map_err(|e| format!("Survival state update failed: {:?}", e))?;
            let logp = state.log_likelihood - state.penalty_term;
            let grad_beta = state.gradient.mapv(|g| -g);
            fast_av_into(&self.chol_t, &grad_beta, grad);
            Ok(logp)
        }

        /// Get the Cholesky factor L for un-whitening samples
        pub fn chol(&self) -> &Array2<f64> {
            &self.chol
        }

        /// Get the mode
        pub fn mode(&self) -> &Array1<f64> {
            &self.data.mode
        }
    }

    impl HamiltonianTarget<Array1<f64>> for SurvivalPosterior {
        fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
            match self.compute_logp_and_grad_into(position, grad) {
                Ok(logp) => logp,
                Err(e) => {
                    log::warn!("Survival posterior evaluation failed: {}", e);
                    grad.fill(0.0);
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Runs NUTS sampling for survival models with whitened parameter space.
    pub(crate) fn run_survival_nuts_sampling(
        age_entry: ArrayView1<'_, f64>,
        age_exit: ArrayView1<'_, f64>,
        event_target: ArrayView1<'_, u8>,
        event_competing: ArrayView1<'_, u8>,
        sampleweight: ArrayView1<'_, f64>,
        x_entry: ArrayView2<'_, f64>,
        x_exit: ArrayView2<'_, f64>,
        x_derivative: ArrayView2<'_, f64>,
        eta_offset_entry: Option<ArrayView1<'_, f64>>,
        eta_offset_exit: Option<ArrayView1<'_, f64>>,
        derivative_offset_exit: Option<ArrayView1<'_, f64>>,
        penalties: PenaltyBlocks,
        monotonicity: SurvivalMonotonicityPenalty,
        spec: SurvivalSpec,
        structurally_monotonic: bool,
        structural_time_columns: usize,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        config: &NutsConfig,
    ) -> Result<NutsResult, String> {
        validate_nuts_config(config).map_err(String::from)?;
        // Create posterior target
        let target = SurvivalPosterior::new(
            age_entry,
            age_exit,
            event_target,
            event_competing,
            sampleweight,
            x_entry,
            x_exit,
            x_derivative,
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            penalties,
            monotonicity,
            spec,
            structurally_monotonic,
            structural_time_columns,
            mode,
            hessian,
        )?;

        // Get Cholesky factor for un-whitening samples later
        let chol = target.chol().clone();
        let mode_arr = target.mode().clone();
        let dim = mode_arr.len();

        let initial_positions = jittered_initial_positions(config, dim, 0.1, 0xEC2D_7A9B_4051_F638);

        let mass_cfg = robust_survival_mass_matrix_config(dim, config.nwarmup);
        let (result, run_stats) = run_whitened_nuts_result(
            target,
            &mode_arr,
            &chol,
            initial_positions,
            config,
            dim,
            mass_cfg,
            0x731B_60D4_AE52_9C8F,
            "NUTS sampling failed",
            Array1::zeros(dim),
            NutsConvergenceThresholds {
                max_rhat: 1.1,
                min_ess: None,
            },
        )?;

        log::info!("Survival NUTS sampling complete: {}", run_stats);

        Ok(result)
    }
}

/// Engine-facing flattened survival NUTS entrypoint.
pub fn run_survival_nuts_sampling_flattened<'a>(
    flat: SurvivalFlatInputs<'a>,
    penalties: gam_models::survival::PenaltyBlocks,
    monotonicity: gam_models::survival::SurvivalMonotonicityPenalty,
    spec: gam_models::survival::SurvivalSpec,
    structurally_monotonic: bool,
    structural_time_columns: usize,
    mode: ArrayView1<'a, f64>,
    hessian: ArrayView2<'a, f64>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    run_nuts_sampling_flattened_family(
        LikelihoodSpec {
            response: ResponseFamily::RoystonParmar,
            link: InverseLink::Standard(StandardLink::Identity),
        },
        FamilyNutsInputs::Survival(Box::new(SurvivalNutsInputs {
            flat,
            penalties,
            monotonicity,
            spec,
            structurally_monotonic,
            structural_time_columns,
            mode,
            hessian,
        })),
        config,
    )
}
