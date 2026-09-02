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
use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerEigh, fast_ab, fast_ata_into, fast_atv, fast_av, fast_av_into,
};
use gam_linalg::matrix::DesignMatrix;
use gam_linalg::triangular::back_substitution_lower_transpose_guarded_into;
use gam_problem::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodScaleMetadata, LikelihoodSpec,
    ResolvedLikelihoodScale, ResponseFamily, RhoPrior, StandardLink, is_valid_tweedie_power,
};
use gam_solve::estimate::reml::FirthDenseOperator;
use gam_solve::estimate::{UnifiedFitResult, validate_explicit_dense_hessian_for_whitening};
use gam_solve::model_types::InferenceCovarianceMode;
use gam_terms::construction::CanonicalPenalty;
use general_mcmc::generic_hmc::HamiltonianTarget;
pub use general_mcmc::generic_nuts::NUTSMassMatrixConfig;
use general_mcmc::generic_nuts::{GenericNUTS, MassMatrixAdaptation};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};
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
    if !(cov_scale.is_finite() && cov_scale > 0.0) {
        return Err(format!(
            "whitening covariance scale must be finite and strictly positive, got {cov_scale}"
        ));
    }
    let hessian_owned = hessian.to_owned();
    gam_linalg::utils::certified_spd_factorize(&hessian_owned, cholesky_error_prefix)
        .map_err(|error| error.to_string())?;
    let chol_factor = hessian_owned
        .cholesky(Side::Lower)
        .map_err(|e| format!("{cholesky_error_prefix}: {:?}", e))?;
    let l_h = chol_factor.lower_triangular();
    let mut chol = solve_upper_triangular_transpose(&l_h, dim);
    let sqrt_cov_scale = cov_scale.sqrt();
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
    /// Fully resolved family, link, and scale metadata consumed by the exact
    /// shared PIRLS/HMC row oracle. Keeping one typed object prevents Gamma
    /// shape, Tweedie power, NB theta, and response dispersion from sharing an
    /// ambiguous scalar slot.
    likelihood: GlmLikelihoodSpec,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
}

thread_local! {
    static NUTS_RESIDUAL_SCRATCH: RefCell<Array1<f64>> = RefCell::new(Array1::zeros(0));
}

/// Resolve and certify the scale metadata consumed by an HMC target.
///
/// The fitted likelihood remains the source of the covariance-scale contract,
/// while the returned target likelihood makes the actual fixed data-term scale
/// explicit. In particular, a profiled Gaussian target is evaluated at its
/// fitted `phi` and a Gamma supplied as fixed dispersion is converted to its
/// exact reciprocal shape. No family parameter is inferred from an unrelated
/// slot and no unit default exists.
fn resolve_hmc_likelihood(
    likelihood: GlmLikelihoodSpec,
    dispersion: gam_solve::model_types::Dispersion,
) -> Result<(GlmLikelihoodSpec, f64), HmcError> {
    let resolved_scale = likelihood
        .resolved_scale()
        .map_err(|error| HmcError::InvalidConfig {
            reason: format!("HMC likelihood scale metadata is unresolved: {error}"),
        })?;
    let phi = dispersion.phi();
    let inv_phi = dispersion
        .reciprocal()
        .map_err(|error| HmcError::InvalidConfig {
            reason: format!("HMC likelihood requires a finite positive dispersion: {error}"),
        })?;

    if matches!(resolved_scale, ResolvedLikelihoodScale::ProfiledGaussian) {
        if !dispersion.is_estimated() {
            return Err(HmcError::InvalidConfig {
                reason: "profiled-Gaussian HMC requires an estimated fitted dispersion".to_string(),
            });
        }
    } else {
        // The standard-deviation argument is consulted only by the profiled
        // Gaussian branch handled above. Every resolved non-profiled family
        // derives its response dispersion entirely from typed metadata.
        let expected = gam_solve::estimate::dispersion_from_likelihood(&likelihood, None).map_err(
            |error| HmcError::InvalidConfig {
                reason: format!("HMC likelihood scale metadata is inconsistent: {error}"),
            },
        )?;
        if expected.phi().to_bits() != phi.to_bits()
            || expected.is_estimated() != dispersion.is_estimated()
        {
            return Err(HmcError::InvalidConfig {
                reason: format!(
                    "HMC dispersion {dispersion:?} disagrees with likelihood metadata dispersion {expected:?}"
                ),
            });
        }
    }

    match (&likelihood.spec.response, &likelihood.spec.link) {
        (ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity))
        | (ResponseFamily::Gamma, InverseLink::Standard(StandardLink::Log))
        | (ResponseFamily::Poisson, InverseLink::Standard(StandardLink::Log))
        | (ResponseFamily::Tweedie { .. }, InverseLink::Standard(StandardLink::Log))
        | (ResponseFamily::NegativeBinomial { .. }, InverseLink::Standard(StandardLink::Log))
        | (ResponseFamily::Beta { .. }, InverseLink::Standard(StandardLink::Logit))
        | (ResponseFamily::Binomial, _) => {}
        (family, link) => {
            return Err(HmcError::LinkMismatch {
                reason: format!(
                    "HMC response family {} is incompatible with inverse link {link:?}",
                    family.name()
                ),
            });
        }
    }

    let cov_scale = likelihood
        .coefficient_covariance_scale(phi)
        .map_err(|error| HmcError::InvalidConfig {
            reason: format!("HMC coefficient-covariance scale is unresolved: {error}"),
        })?;
    if !(cov_scale.is_finite() && cov_scale > 0.0) {
        return Err(HmcError::InvalidConfig {
            reason: format!(
                "HMC coefficient-covariance scale must be finite and positive, got {cov_scale}"
            ),
        });
    }

    let mut target = likelihood;
    target.scale = match (&target.spec.response, target.scale) {
        (ResponseFamily::Gaussian, _) => LikelihoodScaleMetadata::FixedDispersion { phi },
        (ResponseFamily::Gamma, LikelihoodScaleMetadata::FixedDispersion { .. }) => {
            LikelihoodScaleMetadata::FixedGammaShape { shape: inv_phi }
        }
        (_, scale) => scale,
    };
    Ok((target, cov_scale))
}

fn validate_hmc_arrays(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    context: &str,
) -> Result<(), HmcError> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || weights.len() != n {
        return Err(HmcError::DimensionMismatch {
            reason: format!(
                "{context}: row mismatch X={n}, y={}, weights={}",
                y.len(),
                weights.len()
            ),
        });
    }
    if mode.len() != p
        || penalty.nrows() != p
        || penalty.ncols() != p
        || hessian.nrows() != p
        || hessian.ncols() != p
    {
        return Err(HmcError::DimensionMismatch {
            reason: format!(
                "{context}: coefficient geometry mismatch X columns={p}, mode={}, penalty={:?}, hessian={:?}",
                mode.len(),
                penalty.dim(),
                hessian.dim(),
            ),
        });
    }
    for (name, values) in [
        ("design", x.iter()),
        ("penalty", penalty.iter()),
        ("hessian", hessian.iter()),
    ] {
        if let Some((index, value)) = values.enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(HmcError::NonFiniteState {
                reason: format!(
                    "{context}: {name} has non-finite entry {value} at flat index {index}"
                ),
            });
        }
    }
    if let Some((index, value)) = mode
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(HmcError::NonFiniteState {
            reason: format!("{context}: mode has non-finite entry {value} at index {index}"),
        });
    }
    if let Some((index, weight)) = weights
        .iter()
        .enumerate()
        .find(|(_, weight)| !(weight.is_finite() && **weight >= 0.0))
    {
        return Err(HmcError::InvalidConfig {
            reason: format!(
                "{context}: observation weight at row {index} must be finite and non-negative, got {weight}"
            ),
        });
    }
    for row in 0..p {
        for col in (row + 1)..p {
            if penalty[[row, col]] != penalty[[col, row]] {
                return Err(HmcError::InvalidConfig {
                    reason: format!(
                        "{context}: penalty must be exactly symmetric; ({row},{col})={} but ({col},{row})={}",
                        penalty[[row, col]],
                        penalty[[col, row]],
                    ),
                });
            }
        }
    }
    Ok(())
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
    /// * `y` - Response vector \[n_samples\]
    /// * `weights` - Observation/case weights \[n_samples\]
    /// * `penalty_matrix` - Combined penalty S [dim, dim]
    /// * `mode` - MAP estimate μ \[dim\]
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
        likelihood: GlmLikelihoodSpec,
        dispersion: gam_solve::model_types::Dispersion,
        offset: Option<ArrayView1<f64>>,
        firth_enabled: bool,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let dim = x.ncols();

        validate_hmc_arrays(x, y, weights, penalty_matrix, mode, hessian, "NUTS")
            .map_err(String::from)?;
        let (likelihood, cov_scale) =
            resolve_hmc_likelihood(likelihood, dispersion).map_err(String::from)?;
        if let Some(offset) = offset.as_ref() {
            if offset.len() != n_samples {
                return Err(HmcError::DimensionMismatch {
                    reason: format!(
                        "NUTS offset length {} does not match {n_samples} observations",
                        offset.len()
                    ),
                }
                .into());
            }
            if let Some((row, value)) = offset
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(HmcError::NonFiniteState {
                    reason: format!("NUTS offset has non-finite value {value} at row {row}"),
                }
                .into());
            }
        }
        validate_firth_likelihood_support(&likelihood.spec, firth_enabled).map_err(String::from)?;
        if likelihood.spec.is_binomial() {
            validate_binary_responses("binomial NUTS", &y, &weights).map_err(String::from)?;
        }
        if matches!(
            likelihood.spec.response,
            ResponseFamily::NegativeBinomial { .. }
        ) {
            validate_count_responses("negative-binomial NUTS", &y, &weights)
                .map_err(String::from)?;
        }
        let mut eta_at_mode = x.dot(&mode);
        if let Some(offset) = offset.as_ref() {
            eta_at_mode += offset;
        }
        let mut score_at_mode = Array1::zeros(n_samples);
        gam_solve::pirls::eta_log_likelihood_value_and_score_into(
            y,
            &eta_at_mode,
            &likelihood,
            &likelihood.spec.link,
            weights,
            &mut score_at_mode,
        )
        .map_err(|error| format!("NUTS likelihood is invalid at the fitted mode: {error}"))?;

        // Whitening metric: `L Lᵀ` must equal the posterior covariance the
        // sampler reproduces, `Vb = cov_scale · H⁻¹` (#679/#680 invariant), so
        // scale `L` by `√cov_scale`. Only the profiled-Gaussian model carries a
        // non-unit scale (σ̂² = `dispersion.phi()`); every weight-carries-
        // dispersion family (Gamma/Tweedie/NB) already folds its dispersion into
        // the stored `H`, so `cov_scale == 1` and this is a no-op. This replaces
        // a previous `sqrt_phi()` multiply that wrongly scaled Gamma (and any
        // φ-bearing family) by `√φ`, mis-preconditioning against `φ·H⁻¹`.
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
            offset: offset.map(|values| Arc::new(values.to_owned())),
            likelihood,
            n_samples,
            dim,
        };

        Ok(Self {
            data,
            chol,
            chol_t,
            firth_enabled,
            penalty_z_quad,
            penalty_z_lin,
            penalty_z_const,
            cov_scale,
        })
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
        let (ll, mut grad_ll_beta) = match self.family_logp_and_grad_into(&eta, residual) {
            Ok(value) => value,
            Err(error) => {
                log::warn!("[NUTS] likelihood target is unrepresentable: {error}");
                grad.fill(0.0);
                return f64::NEG_INFINITY;
            }
        };

        let mut firth_logdet = 0.0;
        if self.firth_enabled {
            match firth_jeffreys_logp_and_grad(&self.data.likelihood.spec, &self.data, &eta) {
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
        // weight / the `shape ≡ 1/φ` encoded by the resolved Gamma metadata for
        // the dispersion-carrying families), so the penalty must match it:
        //   penalty_scale = 1/cov_scale.
        // That is `1/σ²` for profiled Gaussian and exactly `1.0` for
        // Gamma/Tweedie/NB/Poisson/Binomial. The previous code used
        // the response-dispersion reciprocal for GammaLog (= shape = 1/φ ≠ 1), which
        // double-counted the dispersion in the sampled posterior (#680); the
        // statistical dispersion `φ` is NOT `1/cov_scale` for Gamma because it
        // already lives inside `W`.
        let penalty_scale = 1.0 / self.cov_scale;
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
    ) -> Result<(f64, Array1<f64>), String> {
        exact_glm_logp_and_grad_into(&self.data, eta, residual)
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

/// Wrap the workspace count-response contract in this crate's error type.
///
/// The predicate, the row scan and the message all belong to
/// [`gam_solve::pirls::certify_count_responses`]. This crate previously carried
/// its own copy of all three, and the copy's integrality test was a `1e-9`
/// tolerance against the canonical exact `y == y.round()` -- so a response of
/// `3.0 + 5e-10` was admitted for joint HMC and refused by P-IRLS on the same
/// data and family, while both reported the identical message text.
fn validate_count_responses(
    family: &str,
    y: &ArrayView1<'_, f64>,
    weights: &ArrayView1<'_, f64>,
) -> Result<(), HmcError> {
    gam_solve::pirls::certify_count_responses(y, weights, family)
        .map_err(|reason| HmcError::InvalidConfig { reason })
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

fn exact_glm_logp_and_grad_for_likelihood_into(
    likelihood: &GlmLikelihoodSpec,
    data: &SharedData,
    eta: &Array1<f64>,
    residual: &mut Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    let value = gam_solve::pirls::eta_log_likelihood_value_and_score_into(
        data.y.view(),
        eta,
        likelihood,
        &likelihood.spec.link,
        data.weights.view(),
        residual,
    )
    .map_err(|error| error.to_string())?;
    Ok((value, fast_atv(data.x.as_ref(), residual)))
}

fn exact_glm_logp_and_grad_into(
    data: &SharedData,
    eta: &Array1<f64>,
    residual: &mut Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    exact_glm_logp_and_grad_for_likelihood_into(&data.likelihood, data, eta, residual)
}

fn default_nuts_seed() -> u64 {
    42
}

#[cfg(test)]
mod tests {

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
        PoissonLog,
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
                Self::PoissonLog => LikelihoodSpec {
                    response: ResponseFamily::Poisson,
                    link: InverseLink::Standard(StandardLink::Log),
                },
                Self::GammaLog => LikelihoodSpec {
                    response: ResponseFamily::Gamma,
                    link: InverseLink::Standard(StandardLink::Log),
                },
            }
        }
    }

    use super::{FamilyNutsInputs, GlmFlatInputs, NutsConfig, NutsPosterior, NutsResult, SharedData, exact_glm_logp_and_grad_into, firth_jeffreys_logp_and_grad, laplace_directional_cubic_diagnostic, laplace_skewness_threshold, laplace_trustworthiness_from_skewness, run_logit_polya_gamma_gibbs, run_nuts_sampling_flattened_family};
    use gam_linalg::matrix::DesignMatrix;
    use gam_models::survival::{PenaltyBlocks, SurvivalMonotonicityPenalty, SurvivalSpec};
    use gam_problem::types::{GlmLikelihoodSpec, InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LogLikelihoodNormalization, ResponseFamily, StandardLink};
    use gam_solve::estimate::{
        BlockRole, FitGeometry, FitInference, FittedBlock, FittedLinkState, UnifiedFitResult,
        UnifiedFitResultParts,
    };
    use general_mcmc::generic_hmc::HamiltonianTarget;
    use crate::sample::PosteriorSampler;
    use gam_solve::model_types::InferenceCovarianceMode;
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
            sampler: PosteriorSampler::Nuts,
            covariance: InferenceCovarianceMode::Conditional,
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

    fn nuts_test_likelihood(family: NutsFamily, parameter: f64) -> GlmLikelihoodSpec {
        let spec = family.likelihood_spec();
        let scale = match family {
            NutsFamily::Gaussian => LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            NutsFamily::GammaLog => {
                LikelihoodScaleMetadata::EstimatedGammaShape { shape: parameter }
            }
            NutsFamily::BinomialLogit | NutsFamily::BinomialProbit | NutsFamily::PoissonLog => {
                LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 }
            }
        };
        GlmLikelihoodSpec { spec, scale }
    }

    fn exact_eta_geometry(
        likelihood: &GlmLikelihoodSpec,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        eta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>), String> {
        let mut score = Array1::zeros(eta.len());
        let value = gam_solve::pirls::eta_log_likelihood_value_and_score_into(
            y.view(),
            eta,
            likelihood,
            &likelihood.spec.link,
            weights.view(),
            &mut score,
        )
        .map_err(|error| error.to_string())?;
        Ok((value, score))
    }

    fn hmc_test_fit(
        blocks: Vec<FittedBlock>,
        inference: Option<FitInference>,
        geometry: Option<FitGeometry>,
    ) -> UnifiedFitResult {
        let lambdas = Array1::zeros(0);
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks,
            training_sample_size: 16,
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
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            // Fixed-fit semantics (outer_iterations = 0): these hand-built
            // fixtures carry no analytic stationarity certificate, and the
            // #2255 assembly gate correctly refuses an
            // iterations-ran-but-uncertified state. The fixtures never
            // exercised an outer search; declaring them fixed-ρ fits states
            // what they actually are.
            outer_iterations: 0,
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
                smoothing_correction_method: None,
                smoothing_correction_first_order: None,
                smoothing_correction_method_first_order: None,
                penalized_hessian: hessian.clone().into(),
                reparam_qs: None,
                dispersion: gam_solve::estimate::Dispersion::UNIT,
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
            nuts_test_likelihood(NutsFamily::Gaussian, 1.0),
            gam_solve::estimate::Dispersion::UNIT,
            None,
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
                coefficient_gauge: gam_problem::Gauge::identity(&[1, 1]),
                penalized_hessian: hessian.clone().into(),
                constrained_posterior: None,
                working: None,
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
            training_sample_size: 16,
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
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            // Fixed-fit semantics (outer_iterations = 0): these hand-built
            // fixtures carry no analytic stationarity certificate, and the
            // #2255 assembly gate correctly refuses an
            // iterations-ran-but-uncertified state. The fixtures never
            // exercised an outer search; declaring them fixed-ρ fits states
            // what they actually are.
            outer_iterations: 0,
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
    fn exact_hmc_family_tails_are_finite_when_the_surface_is_representable() {
        let tail_cases = [
            (
                GlmLikelihoodSpec::canonical(LikelihoodSpec::poisson_log()),
                array![0.0],
                array![-1000.0],
            ),
            (
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::Probit),
                )),
                array![1.0],
                array![-30.0],
            ),
            (
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Standard(StandardLink::CLogLog),
                )),
                array![1.0],
                array![-1000.0],
            ),
            (
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(
                        ResponseFamily::Gamma,
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    scale: LikelihoodScaleMetadata::FixedGammaShape { shape: 1.0 },
                },
                array![1.0],
                array![1.0e308],
            ),
            (
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(
                        ResponseFamily::NegativeBinomial {
                            theta: 1.0,
                            theta_fixed: true,
                        },
                        InverseLink::Standard(StandardLink::Log),
                    ),
                    scale: LikelihoodScaleMetadata::FixedNegBinTheta { theta: 1.0 },
                },
                array![2.0],
                array![1.0e308],
            ),
            (
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(
                        ResponseFamily::Beta { phi: 8.0 },
                        InverseLink::Standard(StandardLink::Logit),
                    ),
                    scale: LikelihoodScaleMetadata::EstimatedBetaPhi { phi: 8.0 },
                },
                array![0.2],
                array![-1000.0],
            ),
        ];
        for (likelihood, y, eta) in tail_cases {
            let (value, score) = exact_eta_geometry(&likelihood, &y, &array![1.0], &eta)
                .unwrap_or_else(|error| {
                    panic!(
                        "{} tail should be representable: {error}",
                        likelihood.spec.pretty_name()
                    )
                });
            assert!(value.is_finite());
            assert!(score[0].is_finite());
        }
    }

    #[test]
    fn hmc_scale_resolution_rejects_inconsistent_metadata_without_defaults() {
        let inconsistent_nb = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::NegativeBinomial {
                    theta: 2.0,
                    theta_fixed: true,
                },
                InverseLink::Standard(StandardLink::Log),
            ),
            scale: LikelihoodScaleMetadata::FixedNegBinTheta { theta: 3.0 },
        };
        assert!(
            super::resolve_hmc_likelihood(
                inconsistent_nb,
                gam_solve::model_types::Dispersion::UNIT,
            )
            .is_err()
        );

        let unresolved_gamma = GlmLikelihoodSpec {
            spec: LikelihoodSpec::new(
                ResponseFamily::Gamma,
                InverseLink::Standard(StandardLink::Log),
            ),
            scale: LikelihoodScaleMetadata::Unspecified,
        };
        assert!(
            super::resolve_hmc_likelihood(
                unresolved_gamma,
                gam_solve::model_types::Dispersion::UNIT,
            )
            .is_err()
        );
    }

    #[test]
    fn cloglog_log_mu_uses_complementary_loglog_inverse_link() {
        let eta = -1.0_f64;
        let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
        ));
        let (ll_y1, score) =
            exact_eta_geometry(&likelihood, &array![1.0], &array![1.0], &array![eta])
                .expect("valid eta");
        let residual_y1 = score[0];
        let expected = (1.0 - (-eta.exp()).exp()).ln();
        let wrong_log_one_minus_exp_eta = (1.0 - eta.exp()).ln();

        assert!((ll_y1 - expected).abs() < 1e-14);
        assert!((ll_y1 - wrong_log_one_minus_exp_eta).abs() > 0.5);

        let eps = 1e-6;
        let (lp, _) =
            exact_eta_geometry(&likelihood, &array![1.0], &array![1.0], &array![eta + eps])
                .expect("valid eta");
        let (lm, _) =
            exact_eta_geometry(&likelihood, &array![1.0], &array![1.0], &array![eta - eps])
                .expect("valid eta");
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
            likelihood: nuts_test_likelihood(NutsFamily::PoissonLog, 1.0),
            n_samples: 1,
            dim: 1,
        };
        let eta = array![-701.0];
        let mut eta_score = Array1::zeros(1);
        let (ll, grad) = exact_glm_logp_and_grad_into(&data, &eta, &mut eta_score)
            .expect("representable Poisson tail");
        assert!(
            ll.is_finite() && ll.abs() < 1e-300,
            "Poisson y=0, eta=-701 must keep its ~0 log-density, got {ll}"
        );
        assert!(grad[0].is_finite());

        // Deep cloglog left tail: exp(η) underflows below η ≈ −745, but the
        // exact limits are log μ → η and d(log μ)/dη → 1.
        let cloglog = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
        ));
        let (ll_tail, score_tail) =
            exact_eta_geometry(&cloglog, &array![1.0], &array![1.0], &array![-750.0])
                .expect("finite eta is valid");
        let res_tail = score_tail[0];
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
        let mut overflow_score = Array1::zeros(1);
        assert!(
            exact_glm_logp_and_grad_into(&data_pos, &array![710.0], &mut overflow_score).is_err(),
            "unrepresentable Poisson tail must fail atomically"
        );
    }

    /// #2245 finding 16: saved-model sampling must reconstruct the fitted
    /// *weighted* likelihood, not a unit-weight one. The intercept-only
    /// Bernoulli with `(y, w) = (1, 100), (0, 1)` has weighted score
    /// `dℓ/dη = 100·(1 − μ) − 1·μ = 100 − 101·μ`, which vanishes at
    /// `μ = 100/101`, i.e. the weighted mode `η* = log 100`. Reconstructing the
    /// target with `weights = ones` (the historical bug) instead centres it at
    /// `η = 0`. Pinning the weighted kernel here guards the `saved_prior_weights`
    /// plumbing in `sample.rs` against a silent regression to the unweighted
    /// posterior.
    #[test]
    fn weighted_bernoulli_target_is_centered_at_the_weighted_mode() {
        let data = SharedData {
            x: Arc::new(array![[1.0], [1.0]]),
            y: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![100.0, 1.0]),
            mode: Arc::new(array![0.0]),
            offset: None,
            likelihood: nuts_test_likelihood(NutsFamily::BinomialLogit, 1.0),
            n_samples: 2,
            dim: 1,
        };
        // At the weighted MLE η* = log 100 the score is (numerically) zero.
        let eta_star = 100.0_f64.ln();
        let mut score_star = Array1::zeros(2);
        let (_, grad_star) =
            exact_glm_logp_and_grad_into(&data, &array![eta_star, eta_star], &mut score_star)
                .expect("weighted logit geometry");
        assert!(
            grad_star[0].abs() < 1e-9,
            "weighted Bernoulli score must vanish at log 100, got {}",
            grad_star[0]
        );
        // At the *unweighted* mode η = 0 the weighted score is 100 − 101·0.5 =
        // 49.5 ≫ 0: the unit-weight reconstruction targets the wrong posterior.
        let mut score_zero = Array1::zeros(2);
        let (_, grad_zero) =
            exact_glm_logp_and_grad_into(&data, &array![0.0, 0.0], &mut score_zero)
                .expect("weighted logit geometry");
        assert!(
            (grad_zero[0] - 49.5).abs() < 1e-9,
            "unit-weight point must carry a large positive weighted score, got {}",
            grad_zero[0]
        );
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
            nuts_test_likelihood(NutsFamily::BinomialLogit, 1.0),
            gam_solve::estimate::Dispersion::UNIT,
            None,
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
            likelihood: nuts_test_likelihood(NutsFamily::GammaLog, shape),
            n_samples: x.nrows(),
            dim: x.ncols(),
        };

        let mut eta_score = Array1::zeros(eta.len());
        let (ll, grad) =
            exact_glm_logp_and_grad_into(&data, &eta, &mut eta_score).expect("Gamma geometry");

        let mut expected_ll = 0.0;
        let mut expected_score = 0.0;
        for i in 0..eta.len() {
            let mu = eta[i].exp();
            let ratio = y[i] / mu;
            expected_ll -= weights[i] * shape * (ratio - 1.0 - ratio.ln());
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
            nuts_test_likelihood(NutsFamily::GammaLog, shape),
            gam_solve::estimate::Dispersion::estimated(1.0 / shape).unwrap(),
            None,
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
            nuts_test_likelihood(NutsFamily::GammaLog, shape),
            gam_solve::estimate::Dispersion::estimated(1.0 / shape).unwrap(),
            None,
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
            likelihood: nuts_test_likelihood(NutsFamily::BinomialLogit, 1.0),
            n_samples: x.nrows(),
            dim: x.ncols(),
        };

        let (value, grad) =
            firth_jeffreys_logp_and_grad(&NutsFamily::BinomialLogit.likelihood_spec(), &data, &eta)
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
        assert!(
            out.posterior_std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "every sampled coefficient must have positive posterior spread"
        );
        let eta = x.dot(&out.posterior_mean);
        let posterior_nll: f64 = eta
            .iter()
            .zip(y.iter())
            .map(|(&eta_i, &y_i)| gam_linalg::utils::stable_softplus(eta_i) - y_i * eta_i)
            .sum();
        let zero_nll = x.nrows() as f64 * std::f64::consts::LN_2;
        assert!(
            posterior_nll < zero_nll,
            "posterior mean must discriminate the planted binary responses: {posterior_nll} !< {zero_nll}"
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
                likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
                dispersion: gam_solve::estimate::Dispersion::UNIT,
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
                likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
                dispersion: gam_solve::estimate::Dispersion::UNIT,
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
                likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
                dispersion: gam_solve::estimate::Dispersion::UNIT,
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
            nuts_test_likelihood(NutsFamily::Gaussian, 1.0),
            gam_solve::estimate::Dispersion::UNIT,
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
                nuts_test_likelihood(NutsFamily::Gaussian, 1.0),
                gam_solve::estimate::Dispersion::UNIT,
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
            nuts_test_likelihood(NutsFamily::Gaussian, 1.0),
            gam_solve::estimate::Dispersion::UNIT,
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
            nuts_test_likelihood(NutsFamily::Gaussian, 1.0),
            gam_solve::estimate::Dispersion::UNIT,
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

    /// The batched contraction must return, direction for direction, what the
    /// single-direction contraction returns.
    ///
    /// Batching is a pure performance change, so the only thing that can go
    /// wrong is arithmetic: a transposed direction matrix, a row panel that
    /// drops or double-counts observations, or a sparse scatter that misroutes
    /// a nonzero. Each of those produces a WRONG cubic rather than a slow one,
    /// and the caller only compares it against a threshold — so a silent error
    /// here shows up as a correction that engages when it should not, or vice
    /// versa. Both storage arms are checked against the same reference, on a
    /// design tall enough to cross the row-panel boundary.
    #[test]
    fn batched_directional_cubics_match_the_single_direction_contraction() {
        use super::{directional_cubic_contraction, directional_cubic_contractions};
        use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};

        let n = 37;
        let p = 5;
        // Deterministic, well-conditioned, and NOT symmetric in any way that
        // would let a transposition slip through unnoticed.
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i as f64) * 0.37 + (j as f64) * 1.13).sin() * (1.0 + j as f64 * 0.25)
        });
        let c = Array1::from_shape_fn(n, |i| 0.6 - 0.05 * (i as f64) + ((i % 3) as f64) * 0.4);
        let directions = Array2::from_shape_fn((p, 4), |(j, r)| {
            ((j as f64) * 0.91 - (r as f64) * 0.44).cos()
        });

        // A sparse twin of the same matrix: identical entries, different
        // storage, so both arms must land on the same numbers.
        use faer::sparse::{SparseColMat, Triplet};
        let mut triplets = Vec::new();
        for i in 0..n {
            for j in 0..p {
                if x[[i, j]] != 0.0 {
                    triplets.push(Triplet::new(i, j, x[[i, j]]));
                }
            }
        }
        let dense = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
        let sparse = DesignMatrix::Sparse(gam_linalg::matrix::SparseDesignMatrix::new(
            SparseColMat::try_new_from_triplets(n, p, &triplets).expect("sparse twin"),
        ));

        for (label, design) in [("dense", &dense), ("sparse", &sparse)] {
            let batched = directional_cubic_contractions(design, &c, &directions.view());
            for r in 0..directions.ncols() {
                let reference =
                    directional_cubic_contraction(design, &c, &directions.column(r).view());
                assert!(
                    (batched[r] - reference).abs() <= 1.0e-9 * reference.abs().max(1.0),
                    "{label} arm disagreed on direction {r}: batched {} vs reference {}",
                    batched[r],
                    reference
                );
            }
        }
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
        fn displaced_neg_score(&self, t: &Array1<f64>) -> Result<Array1<f64>, String> {
            // The synthetic oracle has no observation rows: its ΔF carries no
            // deviance channel, so the per-row score moment is empty and the
            // (b)–(d) channel assembly contracts against nothing.
            assert_eq!(t.len(), self.block_dim(), "displacement dim mismatch");
            Ok(Array1::zeros(0))
        }
        fn base_neg_score(&self) -> Result<Array1<f64>, String> {
            Ok(Array1::zeros(0))
        }
    }

    #[test]
    fn block_quadrature_marginal_is_zero_for_gaussian_block() {
        // A purely Gaussian block has ΔF ≡ 0, so the quadrature correction (the
        // log-ratio of true to Laplace block free energy) must be exactly 0,
        // with a zero ρ-gradient. This is the consistency anchor: where the
        // Gaussian summary holds, the fallback is a no-op.
        let target = AnharmonicBlock {
            lambdas: array![2.0, 0.5],
            a: 0.0,
        };
        let out = super::block_quadrature_marginal_correction(&target).expect("correction");
        assert!(
            out.value.abs() < 1e-12,
            "Gaussian block value {}",
            out.value
        );
        assert!(out.rho_gradient.iter().all(|&g| g.abs() < 1e-12));
        assert!(out.node_count > 0);
        assert_eq!(out.quadrature_error, 0.0);
    }

    #[test]
    fn block_quadrature_marginal_recovers_analytic_quartic_correction() {
        // 1-D block with a quartic excess ΔF(t) = a t⁴ (a small positive
        // anharmonicity). Then exp(Δ_b) = E_{t~N(0,1/λ)}[exp(−a t⁴)], a known
        // 1-D integral the deterministic rule must recover. We check Δ_b
        // matches a high-accuracy deterministic quadrature of the same
        // expectation, and that Δ_b < 0 (an added quartic penalty makes the
        // true block mass *smaller* than the Gaussian's).
        let lambda = 3.0_f64;
        let a = 0.05_f64;
        let target = AnharmonicBlock {
            lambdas: array![lambda],
            a,
        };
        let out = super::block_quadrature_marginal_correction(&target).expect("correction");

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
            "quadrature Δ_b {} vs reference {}",
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
        fn displaced_neg_score(&self, t: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(self.excess_and_ngs(&self.s_of(t)).1)
        }
        fn base_neg_score(&self) -> Result<Array1<f64>, String> {
            Ok(self
                .excess_and_ngs(&self.s_of(&Array1::zeros(self.block_dim())))
                .1)
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
    fn block_quadrature_marginal_batched_matches_serial_matvec() {
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

        let serial = super::block_quadrature_marginal_correction(&MatvecBlock {
            lambdas: lambdas.clone(),
            x: x.clone(),
            v_b: v_b.clone(),
            y: y.clone(),
            batched: false,
        })
        .expect("serial");
        let batched = super::block_quadrature_marginal_correction(&MatvecBlock {
            lambdas,
            x,
            v_b,
            y,
            batched: true,
        })
        .expect("batched");

        assert_eq!(serial.node_count, batched.node_count);
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
        let h = 1e-6;
        for axis in 0..z.len() {
            let mut plus = z.clone();
            let mut minus = z.clone();
            plus[axis] += h;
            minus[axis] -= h;
            let mut plus_grad = Array1::<f64>::zeros(3);
            let mut minus_grad = Array1::<f64>::zeros(3);
            let plus_logp = HamiltonianTarget::logp_and_grad(&posterior, &plus, &mut plus_grad);
            let minus_logp = HamiltonianTarget::logp_and_grad(&posterior, &minus, &mut minus_grad);
            let finite_difference = (plus_logp - minus_logp) / (2.0 * h);
            assert!(
                (grad[axis] - finite_difference).abs() <= 2e-5 * finite_difference.abs().max(1.0),
                "structural sparse HMC gradient[{axis}]: analytic={}, finite_difference={finite_difference}",
                grad[axis]
            );
        }
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

/// The sampler that actually produced a [`NutsResult`].
///
/// Every constructor of a draw set stamps its own variant here, so the public
/// `method` badge is a property of the draws and can never be re-derived from
/// the model class by a downstream consumer (gam#2778: the Python badge keyed
/// on `predict_model_class()` and reported `"nuts"` for the Gaussian
/// closed-form and Pólya-Gamma routes, presenting the Laplace path's
/// constant `rhat = 1.0` / `ess = n_draws` as measured MCMC diagnostics).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PosteriorSampler {
    /// No-U-Turn HMC on the exact posterior; `rhat` / `ess` are measured.
    Nuts,
    /// Pólya-Gamma Gibbs on the exact Bernoulli-logit posterior; `rhat` /
    /// `ess` are measured.
    PolyaGammaGibbs,
    /// Independent draws from the Gaussian (Laplace) posterior
    /// approximation, including its exact rejection-truncated and
    /// latent-chart forms; `rhat = 1.0` and `ess = n_draws` hold by
    /// construction and diagnose nothing.
    Laplace,
    /// Reflective HMC on the inequality-truncated Gaussian posterior
    /// approximation; `rhat` / `ess` are measured.
    TruncatedLaplaceHmc,
}

impl PosteriorSampler {
    /// The public badge (`PosteriorSamples.method` in Python).
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nuts => "nuts",
            Self::PolyaGammaGibbs => "polya-gamma",
            Self::Laplace => "laplace",
            Self::TruncatedLaplaceHmc => "truncated-laplace",
        }
    }

    /// Whether the draws target the model's exact posterior rather than a
    /// Gaussian approximation of it. Only the MCMC routes on the exact
    /// likelihood qualify; every Laplace form is an approximation, however
    /// its draws are produced.
    pub const fn targets_exact_posterior(self) -> bool {
        match self {
            Self::Nuts | Self::PolyaGammaGibbs => true,
            Self::Laplace | Self::TruncatedLaplaceHmc => false,
        }
    }
}

/// Result of posterior sampling.
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
    /// Which sampler produced the draws.
    pub sampler: PosteriorSampler,
    /// Which coefficient covariance the draws describe. MCMC on the exact
    /// likelihood is conditional on the fitted smoothing parameters; the
    /// Laplace path draws from the fit's PUBLISHED covariance, which is the
    /// smoothing-corrected `Vp` whenever the fit carries one (gam#2777).
    pub covariance: InferenceCovarianceMode,
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
        sampler: PosteriorSampler::Nuts,
        covariance: InferenceCovarianceMode::Conditional,
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
        // Posterior mean of a sample-function: deterministic parallel reduction over rows.
        // `f: Fn(ArrayView1) -> f64` is shared-access so safe across threads.
        let sum: f64 = gam_linalg::pairwise_reduce::par_pairwise_sum(n, |i| f(self.samples.row(i)));
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
    // Box-Muller requires U1 in the open interval (0, 1). Reject the single
    // exactly-zero lattice point instead of projecting an interval of valid
    // uniforms onto an arbitrary floor, which creates an atom and truncates
    // the normal tail.
    let u1 = loop {
        let draw = rng.random::<f64>();
        if draw > 0.0 {
            break draw;
        }
    };
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
            ndarray::Zip::from(xw.rows_mut())
                .and(x.rows())
                .and(&omega)
                .par_for_each(|mut xw_row, x_row, omega_i| {
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
        sampler: PosteriorSampler::PolyaGammaGibbs,
        covariance: InferenceCovarianceMode::Conditional,
    })
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
/// * `likelihood` - Exact family, inverse-link, and fitted scale metadata
/// * `firth_bias_reduction` - Whether Firth bias reduction was used in training
/// * `config` - NUTS configuration
pub(crate) fn run_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    likelihood: GlmLikelihoodSpec,
    dispersion: gam_solve::model_types::Dispersion,
    firth_bias_reduction: bool,
    offset: Option<ArrayView1<f64>>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    validate_firth_likelihood_support(&likelihood.spec, firth_bias_reduction)
        .map_err(String::from)?;
    validate_nuts_config(config).map_err(String::from)?;
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
        likelihood,
        dispersion,
        offset,
        firth_bias_reduction,
    )?;

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
/// * `outer_hessian` — exact finite symmetric positive-definite outer Hessian
///   `H_ρ` at `ρ̂`, factored without perturbation for whitening.
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

    let mode = rho_hat.to_owned();
    let whitening = hessian_whitening_transform(
        outer_hessian,
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
    /// Fitted scale metadata paired with the `LikelihoodSpec` passed to the
    /// flattened entry point. Family parameters must agree exactly with their
    /// metadata; construction never supplies a unit/shape default.
    pub likelihood_scale: LikelihoodScaleMetadata,
    /// Dispersion parameter φ used to scale the likelihood and the
    /// whitening Cholesky. For fixed-scale families (Binomial, Poisson)
    /// this is exact unit dispersion and has no numerical effect;
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
    let resolved_glm_likelihood = match &inputs {
        FamilyNutsInputs::Glm(glm) => Some(GlmLikelihoodSpec {
            spec: likelihood.clone(),
            scale: glm.likelihood_scale,
        }),
        FamilyNutsInputs::Survival(_) => None,
    };
    if let (Some(resolved), FamilyNutsInputs::Glm(glm)) =
        (resolved_glm_likelihood.as_ref(), &inputs)
    {
        // Validate family/scale ownership before dispatch, including the PG
        // branch that does not construct a NutsPosterior.
        resolve_hmc_likelihood(resolved.clone(), glm.dispersion).map_err(String::from)?;
    }
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
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
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
                    resolved_glm_likelihood
                        .clone()
                        .expect("GLM match arm has resolved likelihood"),
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
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
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
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
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
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
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
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
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
                resolved_glm_likelihood
                    .clone()
                    .expect("GLM match arm has resolved likelihood"),
                glm.dispersion,
                glm.firth_bias_reduction,
                glm.offset,
                config,
            )
        }
        (ResponseFamily::NegativeBinomial { .. }, _, FamilyNutsInputs::Glm(glm)) => {
            // Family mapping: NegativeBinomial payload theta is passed through the family slot.
            // NB dispersion scale is unit; theta is not derived from fixed_phi.
            run_nuts_sampling(
                glm.x,
                glm.y,
                glm.weights,
                glm.penalty_matrix,
                glm.mode,
                glm.hessian,
                resolved_glm_likelihood
                    .clone()
                    .expect("GLM match arm has resolved likelihood"),
                glm.dispersion,
                glm.firth_bias_reduction,
                glm.offset,
                config,
            )
        }
        (
            ResponseFamily::Beta { .. },
            InverseLink::Standard(StandardLink::Logit),
            FamilyNutsInputs::Glm(glm),
        ) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
            glm.dispersion,
            glm.firth_bias_reduction,
            glm.offset,
            config,
        ),
        (ResponseFamily::Beta { .. }, _, FamilyNutsInputs::Glm(_)) => {
            Err("beta-regression NUTS requires the logit inverse link".to_string())
        }
        (ResponseFamily::Gamma, _, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            resolved_glm_likelihood
                .clone()
                .expect("GLM match arm has resolved likelihood"),
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
    //
    // Every direction here contracts the SAME design against a different
    // eigenvector, so the whole phase is one `X V` product. Issuing it as p
    // independent GEMVs re-streamed `X` from memory p times and made this the
    // single largest cost in a fit profile; batching it hands faer a GEMM that
    // reuses each row of `X` across all directions at once.
    let positive: Vec<usize> = (0..p).filter(|&r| evals[r] > tol).collect();
    if !positive.is_empty() {
        let mut directions = Array2::<f64>::zeros((p, positive.len()));
        for (slot, &r) in positive.iter().enumerate() {
            directions.column_mut(slot).assign(&evecs.column(r));
        }
        let cubics = directional_cubic_contractions(design, c_weights, &directions.view());
        for (slot, &r) in positive.iter().enumerate() {
            let gamma = cubics[slot] / evals[r].powf(1.5);
            directional[r] = if gamma.is_finite() { gamma } else { 0.0 };
            max_abs = max_abs.max(directional[r].abs());
        }
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

/// Row-panel height for the batched contraction, chosen so one panel of
/// projections stays inside a few MiB regardless of how many directions are
/// batched: `rows × k ≲ 2^21` doubles (16 MiB).
const CUBIC_PANEL_DOUBLES: usize = 1 << 21;

/// Compute `T[v_r,v_r,v_r] = Σ_i c_i (x_iᵀ v_r)³` for EVERY column `v_r` of
/// `directions` (p × k) in one pass.
///
/// The single-direction [`directional_cubic_contraction`] forms `X v`, so
/// calling it once per direction forms `X v_1, …, X v_k` — which is the GEMM
/// `X V` spelled as k separate GEMVs. The diagnostic's phase 1 does exactly
/// that over every positive-curvature eigenvector, so the whole O(n·p²) step
/// was running at BLAS-2 intensity: each GEMV re-streams all of `X` from
/// memory to reuse a single vector. Forming the product once lets the rows of
/// `X` be reused across all k directions while they are in cache, which is the
/// entire difference between a memory-bound and a compute-bound kernel.
///
/// Rows are processed in panels so the intermediate never scales with `n·k`.
fn directional_cubic_contractions(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    directions: &ArrayView2<f64>,
) -> Array1<f64> {
    let k = directions.ncols();
    let mut cubics = Array1::<f64>::zeros(k);
    if k == 0 {
        return cubics;
    }
    match design.as_sparse() {
        Some(x_sparse) => {
            // One structural pass over the CSC nonzeros scatters into all k
            // projection columns at once, instead of k passes that each walk
            // the same index arrays.
            let (symbolic, values) = x_sparse.as_ref().parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            let rows = x_sparse.nrows().min(c_weights.len());
            if rows == 0 {
                return cubics;
            }
            let panel = (CUBIC_PANEL_DOUBLES / k).clamp(1, rows);
            let mut start = 0;
            while start < rows {
                let stop = (start + panel).min(rows);
                let mut projections = Array2::<f64>::zeros((stop - start, k));
                for col in 0..x_sparse.ncols() {
                    let coeffs = directions.row(col);
                    for ptr in col_ptr[col]..col_ptr[col + 1] {
                        let row = row_idx[ptr];
                        if row < start || row >= stop {
                            continue;
                        }
                        let value = values[ptr];
                        let mut target = projections.row_mut(row - start);
                        for r in 0..k {
                            target[r] += value * coeffs[r];
                        }
                    }
                }
                for (offset, i) in (start..stop).enumerate() {
                    let weight = c_weights[i];
                    let row = projections.row(offset);
                    for r in 0..k {
                        cubics[r] += weight * row[r].powi(3);
                    }
                }
                start = stop;
            }
        }
        None => {
            let x_dense = design.to_dense_cow();
            let x_dense = x_dense.as_ref();
            let rows = x_dense.nrows().min(c_weights.len());
            if rows == 0 {
                return cubics;
            }
            let panel = (CUBIC_PANEL_DOUBLES / k).clamp(1, rows);
            let mut start = 0;
            while start < rows {
                let stop = (start + panel).min(rows);
                let projections = fast_ab(&x_dense.slice(s![start..stop, ..]), directions);
                for (offset, i) in (start..stop).enumerate() {
                    let weight = c_weights[i];
                    let row = projections.row(offset);
                    for r in 0..k {
                        cubics[r] += weight * row[r].powi(3);
                    }
                }
                start = stop;
            }
        }
    }
    for value in cubics.iter_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    cubics
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
            let rows = x_dense.nrows().min(c_weights.len());
            if rows == 0 {
                return 0.0;
            }
            // `x_i · v` for every row IS `X v`. Issuing it as `rows` separate
            // 1-D dots leaves ndarray on its scalar `dot_generic` fallback —
            // this crate builds ndarray without the `blas` feature, so its
            // `dot` never reaches a GEMV kernel. A profile of a temporal fit
            // put 44% of total runtime in that one symbol, called from here
            // and from `directional_cubic_gradient` under the power iteration
            // below. One faer GEMV does the same arithmetic against the SIMD
            // microkernels. The sparse arm above already batches this way.
            let projections = fast_av(&x_dense.slice(s![..rows, ..]), v);
            let mut cubic = 0.0_f64;
            for i in 0..rows {
                cubic += c_weights[i] * projections[i].powi(3);
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
            let rows = x_dense.nrows().min(c_weights.len());
            if rows == 0 {
                return Array1::<f64>::zeros(p);
            }
            // Same two products the sparse arm above forms explicitly:
            // `X v` for the projections, then `Xᵀ w` for the gradient. Written
            // row-at-a-time this was a scalar 1-D dot plus a hand-rolled
            // `grad += w · row` inner loop, both of which show up in a fit
            // profile (`dot_generic` and `scaled_add`, together the single
            // largest cost in a temporal fit). Two faer GEMVs replace the
            // whole nest.
            let x_rows = x_dense.slice(s![..rows, ..]);
            let projections = fast_av(&x_rows, v);
            let mut quad_weights = Array1::<f64>::zeros(rows);
            for i in 0..rows {
                quad_weights[i] = 3.0 * c_weights[i] * projections[i] * projections[i];
            }
            fast_atv(&x_rows, &quad_weights)
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
    BLOCK_GH_MAX_DIM, BlockExcessTarget, BlockQuadratureMarginal, BlockQuadratureMoments,
    LaplaceTrustworthiness, laplace_skewness_threshold,
    laplace_trustworthiness_from_skewness,
};

/// Monolith (gam-inference-tier) implementor of the contract-downed
/// [`LaplaceMarginalCorrector`](gam_problem::laplace_sampler_contract::LaplaceMarginalCorrector):
/// wraps the `hmc_io` directional-cubic eigen diagnostic and the
/// deterministic #784 block correction.
pub struct HmcIoLaplaceMarginalCorrector;

impl gam_problem::laplace_sampler_contract::LaplaceMarginalCorrector
    for HmcIoLaplaceMarginalCorrector
{
    fn directional_cubic_diagnostic(
        &self,
        hessian: &Array2<f64>,
        design: &DesignMatrix,
        c_weights: &Array1<f64>,
        refine_supremum: bool,
    ) -> Result<(f64, Array1<f64>), String> {
        laplace_directional_cubic_diagnostic(hessian, design, c_weights, refine_supremum)
    }

    fn block_quadrature_marginal_correction(
        &self,
        target: &dyn BlockExcessTarget,
    ) -> Result<BlockQuadratureMarginal, String> {
        block_quadrature_marginal_correction(target)
    }
}

/// Evaluate the block-local marginal correction `Δ_b` and its ρ-gradient by
/// deterministic Gauss-Hermite quadrature against the local Laplace Gaussian
/// (issue #784).
///
/// # Math
///
/// Integrate `t ~ q = N(0, diag(1/λ_r))` (the local Laplace Gaussian in the
/// block subspace; standard-normal nodes `z_s` give `t_{s,r}=z_{s,r}/√λ_r`).
/// With the non-Gaussian remainder `ΔF` defined on [`BlockExcessTarget`],
///
///   exp(Δ_b) = E_q[ exp(−ΔF(t)) ],
///
/// computed via a numerically-stable weighted log-sum-exp. The ρ-gradient follows
/// from differentiating `Δ_b = log E_q[e^{−ΔF}]` (the `q`-Gaussian normalizer
/// `½Σ log(2π/λ_r)` cancels against `A_Lap`, leaving only the `ΔF` channel):
///
///   ∂Δ_b/∂ρ_k = E_p[ −∂ΔF/∂ρ_k ],   p ∝ q·e^{−ΔF},
///
/// i.e. the normalized quadrature average of `−∂ΔF/∂ρ_k`. Because value,
/// gradient, and all envelope moments come from the same nodes and target, they
/// are mutually consistent — the contract the outer REML needs.
///
/// The five-node rule is exact for standard-normal polynomials through degree
/// nine. A separate three-node rule (degree five) supplies a deterministic
/// rule-difference estimate for the realized non-polynomial integrand; the
/// caller admits the correction only when that difference resolves both `Δ_b`
/// and the `O(1/n_eff)` Laplace floor.
/// Streaming log-sum-exp of `log_terms` in the order given, accumulated
/// against a running maximum exactly as the block-quadrature loop below
/// accumulates its scalar weight (`sum_w *= exp(max_old − max_new)` on a new
/// maximum, then `sum_w += exp(lw − max)`). Returns `−∞` for an empty
/// sequence.
///
/// The operation order is the contract, not an implementation detail: the
/// correction is the self-normalised `log Σ wᵢe^{−ΔFᵢ} − log Σ wᵢ`, and when
/// `ΔF ≡ 0` the two reductions run over bit-identical terms in the same
/// order, so they cancel exactly and a Gaussian block reports a correction of
/// exactly `0` with a paired-rule error of exactly `0`. Normalising with the
/// literal `1` the rule's weights sum to in exact arithmetic would instead
/// report the rule's own weight-normalisation roundoff (≈1e-16, and different
/// for the five-node and three-node rules) as a quadrature error.
fn streaming_log_sum_exp(log_terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut max_lw = f64::NEG_INFINITY;
    let mut sum_w = 0.0_f64;
    for lw in log_terms {
        if lw > max_lw {
            sum_w *= (max_lw - lw).exp();
            max_lw = lw;
        }
        sum_w += (lw - max_lw).exp();
    }
    max_lw + sum_w.ln()
}

pub fn block_quadrature_marginal_correction<T: BlockExcessTarget + ?Sized>(
    target: &T,
) -> Result<BlockQuadratureMarginal, String> {
    let m = target.block_dim();
    let k = target.rho_dim();
    if m == 0 {
        return Ok(BlockQuadratureMarginal {
            value: 0.0,
            rho_gradient: Array1::zeros(k),
            quadrature_error: 0.0,
            node_count: 0,
            moments: None,
        });
    }
    let lambdas = target.block_curvatures();
    if lambdas.len() != m {
        return Err(format!(
            "block_quadrature_marginal_correction: block_curvatures len {} != block_dim {m}",
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
            "block_quadrature_marginal_correction: non-positive block curvature (mode is not a \
             strict local minimum in an integrated direction)"
                .to_string(),
        );
    }
    if m > BLOCK_GH_MAX_DIM {
        return Err(format!(
            "block-local Gauss-Hermite correction supports at most {BLOCK_GH_MAX_DIM} \
             curvature-heavy directions, got {m}"
        ));
    }

    let fine_rule = crate::rho_posterior::standard_normal_gh_rule(5)
        .expect("the five-node standard-normal Gauss-Hermite rule is built in");
    let mut fine_nodes = Vec::new();
    crate::rho_posterior::enumerate_gh_product(
        m,
        fine_rule,
        0,
        &mut Array1::zeros(m),
        0.0,
        &mut fine_nodes,
    );
    let node_count = fine_nodes.len();

    // Streaming, numerically-stable accumulation of the weighted log-sum-exp value,
    // the explicit gradient channel `E_p[−∂ΔF/∂ρ]`, AND the gradient-channel
    // moments `E_p[t]`, `E_p[t tᵀ]`, `E_p[ngs]`, `E_p[t ⊗ ngs]` needed by the
    // exact (b)–(d) channel assembly (gradient exactness contract above).
    // Weights are kept relative to a running maximum log-weight: whenever a
    // new maximum arrives, every accumulator is rescaled by
    // `exp(max_old − max_new) ≤ 1`, so each per-draw relative weight is ≤ 1
    // and the sums never overflow. Infeasible / divergent draws contribute
    // zero weight rather than poisoning the estimate.
    let n_obs = target.base_neg_score()?.len();
    let mut max_lw = f64::NEG_INFINITY;
    let mut sum_w = 0.0_f64;
    let mut grad_acc = Array1::<f64>::zeros(k);
    let mut e_t_acc = Array1::<f64>::zeros(m);
    let mut e_tt_acc = Array2::<f64>::zeros((m, m));
    let mut e_ngs_acc = Array1::<f64>::zeros(n_obs);
    let mut e_t_ngs_acc = Array2::<f64>::zeros((n_obs, m));

    // Materialize all transformed quadrature nodes into the columns of `draws`
    // (`m × n_draws`). The per-node design matvec `s = X_t·(V_b·t_s)` is batched
    // into two BLAS-3 products over all columns at once (the #1082 hot path),
    // instead of separate BLAS-2 matvecs.
    let mut draws = Array2::<f64>::zeros((m, node_count));
    for (s, (z, _)) in fine_nodes.iter().enumerate() {
        for r in 0..m {
            draws[(r, s)] = z[r] * inv_sqrt_lambda[r];
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
        let lw = fine_nodes[sidx].1 - excess;
        if lw > max_lw {
            // exp(−∞ − lw) = 0 zeroes the (empty) accumulators on the first
            // feasible draw, so no special-casing is needed.
            let rescale = (max_lw - lw).exp();
            sum_w *= rescale;
            grad_acc *= rescale;
            e_t_acc *= rescale;
            e_tt_acc *= rescale;
            e_ngs_acc *= rescale;
            e_t_ngs_acc *= rescale;
            max_lw = lw;
        }
        let w = (lw - max_lw).exp();
        sum_w += w;
        // Explicit channel: −∂ΔF/∂ρ.
        grad_acc.scaled_add(-w, &target.excess_rho_gradient(&t));
        // Moment channels (score already computed in the fused call above).
        if ngs.len() != n_obs {
            return Err(format!(
                "block_quadrature_marginal_correction: displaced_neg_score len {} != {n_obs}",
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
            "block_quadrature_marginal_correction: all fine quadrature nodes were infeasible"
                .to_string(),
        );
    }
    // Self-normalised value `log Σ wᵢe^{−ΔFᵢ} − log Σ wᵢ`: the normaliser is
    // the same streaming reduction over the same product log-weights, so the
    // rule's weight-normalisation roundoff cancels rather than being reported
    // as part of the correction (see `streaming_log_sum_exp`).
    let fine_log_norm = streaming_log_sum_exp(fine_nodes.iter().map(|(_, log_w)| *log_w));
    let value = (max_lw + sum_w.ln()) - fine_log_norm;
    // Self-normalized importance-weighted gradient E_p[−∂ΔF/∂ρ] and moments.
    let (rho_gradient, moments) = if sum_w > 0.0 {
        (
            grad_acc / sum_w,
            Some(BlockQuadratureMoments {
                e_t: e_t_acc / sum_w,
                e_tt: e_tt_acc / sum_w,
                e_neg_score: e_ngs_acc / sum_w,
                e_t_neg_score: e_t_ngs_acc / sum_w,
            }),
        )
    } else {
        (Array1::zeros(k), None)
    };
    // Paired-rule error estimate: repeat only the scalar integral with the
    // three-node (degree-five) product rule. The fine/coarse difference is in
    // the same log-marginal units as Δ_b and is deterministic across rho.
    let coarse_rule = crate::rho_posterior::standard_normal_gh_rule(3)
        .expect("the three-node standard-normal Gauss-Hermite rule is built in");
    let mut coarse_nodes = Vec::new();
    crate::rho_posterior::enumerate_gh_product(
        m,
        coarse_rule,
        0,
        &mut Array1::zeros(m),
        0.0,
        &mut coarse_nodes,
    );
    let mut coarse_draws = Array2::<f64>::zeros((m, coarse_nodes.len()));
    for (s, (z, _)) in coarse_nodes.iter().enumerate() {
        for r in 0..m {
            coarse_draws[(r, s)] = z[r] * inv_sqrt_lambda[r];
        }
    }
    let coarse_values = target.excess_batch(&coarse_draws);
    // The coarse estimate is the same self-normalised reduction as the fine
    // one (infeasible nodes contribute zero weight to the numerator and keep
    // their weight in the normaliser, exactly as in the fine loop).
    let coarse_log_numerator = streaming_log_sum_exp(
        coarse_values
            .iter()
            .zip(&coarse_nodes)
            .filter(|(excess, _)| excess.is_finite())
            .map(|(excess, (_, log_w))| log_w - excess),
    );
    if !coarse_log_numerator.is_finite() {
        return Err(
            "block_quadrature_marginal_correction: every coarse quadrature node was infeasible"
                .to_string(),
        );
    }
    let coarse_log_norm = streaming_log_sum_exp(coarse_nodes.iter().map(|(_, log_w)| *log_w));
    let coarse_value = coarse_log_numerator - coarse_log_norm;
    let quadrature_error = (value - coarse_value).abs();

    if !value.is_finite() || rho_gradient.iter().any(|v| !v.is_finite()) {
        return Err(
            "block_quadrature_marginal_correction: produced a non-finite correction or gradient"
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
            "block_quadrature_marginal_correction: produced non-finite gradient-channel moments"
                .to_string(),
        );
    }

    Ok(BlockQuadratureMarginal {
        value,
        rho_gradient,
        quadrature_error,
        node_count,
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

/// Inputs for joint (β, ρ) sampling.
pub struct JointBetaRhoInputs<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub likelihood: GlmLikelihoodSpec,
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
