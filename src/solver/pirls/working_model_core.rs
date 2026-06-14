
// Re-export public API that lived in the original flat file
use convergence::effective_kkt_tolerance;

use damping::{
    add_scaled_diagonal_to_upper_sparse, compute_lm_d2, update_scaled_diagonal_in_place,
};

pub use edf::StablePLSResult;

use edf::{
    calculate_edf_from_sparse_factor, calculate_edf_with_penalty,
    calculate_edfwithworkspace_from_factor, calculate_edfwithworkspace_with_penalty,
};

use log_link_working_state::ETA_CLAMP;

/// The canonical PIRLS numeric floors live in [`log_link_working_state`]; this
/// re-export gives every family in the crate one shared `MIN_WEIGHT` so the
/// weighted normal equations stay well posed with a single retunable value.
pub(crate) use log_link_working_state::MIN_WEIGHT;

use penalty::{
    KroneckerQsTransform, PirlsPenalty, WorkingCoordinateDesign, WorkingReparamTransform,
    attach_penalty_shift,
};

use pls_solver::solve_penalized_least_squares_implicit;

pub use pls_solver::{GaussianFixedCache, SparseXtwxPrecomputed};

pub use reweight::runworking_model_pirls;

pub(crate) use state::array1_l2_norm;

pub use state::{
    AdaptiveKktTolerance, ExportedLaplaceCurvature, FirthDiagnostics, HessianCurvatureKind,
    PirlsCoordinateFrame, PirlsLinearSolvePath, PirlsResult, PirlsStatus,
    WorkingModelIterationInfo, WorkingModelPirlsResult, WorkingState,
};


const GAMMA_SHAPE_MIN: f64 = 1e-8;

const GAMMA_SHAPE_MAX: f64 = 1e12;

const GAMMA_SHAPE_TARGET_TOL: f64 = 1e-12;


/// Saturation threshold for `|η|` diagnostics at inner P-IRLS iterates.
///
/// This value no longer rejects otherwise finite step candidates. Stable
/// likelihood code owns tail arithmetic; this threshold only helps the rescue
/// logic classify a stalled fit pinned deep in a separated/saturated tail.
pub(super) const PIRLS_ETA_ABS_CAP: f64 = 40.0;


#[inline]
fn gamma_shape_score(shape: f64, target: f64) -> f64 {
    shape.ln() - digamma(shape) - target
}


fn estimate_gamma_shape_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    const EPS: f64 = 1e-12;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (weighted_target, total_weight) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            let yi = y[i].max(EPS);
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(EPS);
            let ratio = yi / mui;
            (wi * (ratio - ratio.ln() - 1.0), wi)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(t1, w1), (t2, w2)| (t1 + t2, w1 + w2),
        );

    if total_weight <= 0.0 {
        return 1.0;
    }

    let target = (weighted_target / total_weight).max(0.0);
    if target <= GAMMA_SHAPE_TARGET_TOL {
        return GAMMA_SHAPE_MAX;
    }

    let discriminant = (target - 3.0) * (target - 3.0) + 24.0 * target;
    let approx = ((3.0 - target) + discriminant.sqrt()) / (12.0 * target);
    let mut lo = GAMMA_SHAPE_MIN;
    let mut hi = approx.max(1.0);

    while hi < GAMMA_SHAPE_MAX && gamma_shape_score(hi, target) > 0.0 {
        hi = (hi * 2.0).min(GAMMA_SHAPE_MAX);
    }
    if gamma_shape_score(hi, target) > 0.0 {
        return GAMMA_SHAPE_MAX;
    }

    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if gamma_shape_score(mid, target) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= GAMMA_SHAPE_TARGET_TOL * hi.max(1.0) {
            break;
        }
    }

    0.5 * (lo + hi)
}


/// Method-of-moments estimate of the Beta-regression precision `phi` from the
/// current linear predictor `eta` (logit link).
///
/// For a Beta GLM `Var(y_i) = mu_i(1-mu_i)/(1+phi)`, so the standardized Pearson
/// residual `s_i = (y_i - mu_i)^2 / (mu_i(1-mu_i))` has `E[s_i] = 1/(1+phi)`.
/// Equating the prior-weighted average of `s_i` to its expectation gives
/// `1 + phi = Σ w_i / Σ w_i s_i`, i.e. `phi = (Σ w_i / Σ w_i s_i) - 1`. This is
/// the standard moment estimator betareg uses to initialize / cross-check the
/// joint MLE; iterating mean-fit → phi-estimate → refit across the outer
/// smoothing-parameter loop drives it to the joint optimum. The estimate is
/// clamped to a wide, strictly-positive admissible band so a transient
/// near-degenerate residual sum cannot push `phi` non-positive or to infinity.
fn estimate_beta_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    const PHI_MIN: f64 = 1e-3;
    const PHI_MAX: f64 = 1e6;
    const MU_EPS: f64 = 1e-9;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (weighted_pearson, total_weight) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            // Logit inverse link with a small guard so the variance denominator
            // mu(1-mu) stays strictly positive at the boundaries.
            let mui = (1.0 / (1.0 + (-eta[i].clamp(-ETA_CLAMP, ETA_CLAMP)).exp()))
                .clamp(MU_EPS, 1.0 - MU_EPS);
            let var_unit = mui * (1.0 - mui);
            let resid = y[i] - mui;
            (wi * resid * resid / var_unit, wi)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(p1, w1), (p2, w2)| (p1 + p2, w1 + w2),
        );

    if total_weight <= 0.0 || weighted_pearson <= 0.0 {
        return 1.0;
    }
    let one_plus_phi = (total_weight / weighted_pearson).max(1.0 + PHI_MIN);
    (one_plus_phi - 1.0).clamp(PHI_MIN, PHI_MAX)
}


/// Pearson moment estimate of the Tweedie dispersion `phi` from the current
/// linear predictor `eta` (log link, `mu = exp(eta)`).
///
/// A Tweedie response has `Var(yᵢ) = phi · V(μᵢ) / wᵢ` with unit variance
/// function `V(μ) = μ^p` and prior weight `wᵢ`, so the prior-weighted Pearson
/// statistic `Σ wᵢ (yᵢ − μᵢ)² / μᵢ^p` has expectation `phi · (Σwᵢ − edf)`.
/// Equating it to its expectation and normalising by the total prior weight
/// gives the moment estimator
///
/// ```text
/// phî = Σ wᵢ (yᵢ − μᵢ)² / μᵢ^p   /   Σ wᵢ.
/// ```
///
/// This is the standard Pearson dispersion estimator (statsmodels' Tweedie and
/// mgcv's fixed-`p` `Tweedie()` use the same statistic). We normalise by `Σwᵢ`
/// rather than the residual df `Σwᵢ − edf` to match the sibling Gamma-shape /
/// Beta-precision moment estimators in this module, which also estimate at the
/// converged η without an edf correction; the `O(edf/n)` difference is far
/// below statistical resolution at any `n` for which a Tweedie fit is
/// meaningful, and the iterate-to-self-consistency contract (reported `phi` ==
/// `estimate_tweedie_phi_from_eta(final_eta)`) is what the covariance scale and
/// the prediction SE both consume. Threading `phî` into the working weight
/// `prior·μ^{2−p}/phi` is what makes `SE(η̂) ∝ √phi` (issue #771); freezing
/// `phi = 1` made every Tweedie SE / interval / generate draw ignore the data's
/// dispersion. The estimate is clamped to a wide strictly-positive band so a
/// transient degenerate residual sum cannot push `phi` non-positive or
/// non-finite.
fn estimate_tweedie_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    p: f64,
) -> f64 {
    const PHI_MIN: f64 = 1e-6;
    const PHI_MAX: f64 = 1e12;
    const MU_EPS: f64 = 1e-300;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (weighted_pearson, total_weight) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(MU_EPS);
            let resid = y[i] - mui;
            // Unit variance function V(mu) = mu^p with the dispersion factored
            // out; the prior-weighted Pearson contribution is wᵢ·resid²/V(μᵢ).
            let var_unit = mui.powf(p).max(MU_EPS);
            (wi * resid * resid / var_unit, wi)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(p1, w1), (p2, w2)| (p1 + p2, w1 + w2),
        );

    if total_weight <= 0.0 || !weighted_pearson.is_finite() || weighted_pearson <= 0.0 {
        return 1.0;
    }
    (weighted_pearson / total_weight).clamp(PHI_MIN, PHI_MAX)
}


/// Admissible band for the estimated Negative-Binomial overdispersion `theta`.
/// `THETA_MIN` caps the heaviest overdispersion the estimator will report;
/// `THETA_MAX` is the effective Poisson limit (`Var → mu`) used when the data is
/// equi- or under-dispersed and the ML score has no finite interior root.
const NEGBIN_THETA_MIN: f64 = 1e-3;

const NEGBIN_THETA_MAX: f64 = 1e6;


/// Prior-weighted Negative-Binomial `theta` ML score and observed information at
/// a single `theta`, evaluated at the log-link mean `mu = exp(eta)`.
///
/// For the NB2 log-likelihood
/// `ℓ = Σ wᵢ[lnΓ(yᵢ+θ) − lnΓ(θ) − lnΓ(yᵢ+1) + θ(lnθ − ln(θ+μᵢ)) + yᵢ ln μᵢ
///        − yᵢ ln(θ+μᵢ)]`,
/// the score and (negative second-derivative) observed information in `θ` are
/// ```text
/// S(θ) = Σ wᵢ[ ψ(yᵢ+θ) − ψ(θ) + lnθ + 1 − ln(θ+μᵢ) − (yᵢ+θ)/(μᵢ+θ) ]
/// I(θ) = Σ wᵢ[ −ψ'(yᵢ+θ) + ψ'(θ) − 1/θ + 2/(μᵢ+θ) − (yᵢ+θ)/(μᵢ+θ)² ]
/// ```
/// — the exact statistics MASS `glm.nb`/`theta.ml` Newton-iterates. Both sums
/// share one pass over the rows.
fn negbin_theta_score_and_info(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> (f64, f64) {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let psi_theta = digamma(theta);
    let trigamma_theta = trigamma(theta);
    let ln_theta = theta.ln();
    let inv_theta = 1.0 / theta;
    let (score, info) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            let yi = y[i];
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(1e-300);
            let theta_plus_mu = theta + mui;
            let theta_plus_y = theta + yi;
            let s = digamma(yi + theta) - psi_theta + ln_theta + 1.0
                - theta_plus_mu.ln()
                - theta_plus_y / theta_plus_mu;
            let info_row = -trigamma(yi + theta) + trigamma_theta - inv_theta + 2.0 / theta_plus_mu
                - theta_plus_y / (theta_plus_mu * theta_plus_mu);
            (wi * s, wi * info_row)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(s1, i1), (s2, i2)| (s1 + s2, i1 + i2),
        );
    (score, info)
}


/// Maximum-likelihood estimate of the Negative-Binomial overdispersion `theta`
/// from the current linear predictor `eta` (log link, `mu = exp(eta)`).
///
/// NB2 has `Var(yᵢ) = μᵢ + μᵢ²/θ` — `θ` is a genuine free parameter that, unlike
/// the dispersion scales of Gamma/Tweedie/Beta, lives inside the *variance
/// function*: it enters the IRLS working weight `W = μθ/(θ+μ)` (the full NB2
/// Fisher information), so threading `θ̂` into the weight is what makes the
/// coefficient/η SEs respond to the data's overdispersion (issue #802 — a frozen
/// `θ = 1` left every SE/interval/`generate` draw ignoring it). The seed `θ`
/// carried on the family variant does not enter here, so the converged estimate
/// is seed-independent.
///
/// We solve the ML score `S(θ) = 0` (the same statistic MASS `glm.nb` uses).
/// `S` is strictly decreasing on `(0, ∞)` with `S(0⁺) = +∞`, so an interior root
/// exists iff `S(THETA_MAX) < 0` (the data is overdispersed); when the data is
/// equi- or under-dispersed `S` stays positive and the MLE diverges toward the
/// Poisson limit, which we report as the clamp `THETA_MAX`. The root is found by
/// safeguarded Newton (Newton step on the analytic `(S, I)`, bisection fallback
/// whenever a step leaves the maintained sign-bracket), seeded from the method-
/// of-moments overdispersion `μ̄/(D−1)` with `D` the Poisson-Pearson ratio. This
/// converges quadratically near the root in a handful of `O(n)` passes, matching
/// the sibling Gamma-shape/Beta-φ converged-η estimators in this module.
fn estimate_negbin_theta_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // Method-of-moments seed from the Poisson-Pearson overdispersion ratio
    // `D = Σ wᵢ (yᵢ−μᵢ)²/μᵢ / Σ wᵢ`. With `Var/μ = 1 + μ/θ`, matching the
    // weighted-mean `μ̄` gives `θ₀ = μ̄/(D−1)`; if `D ≤ 1` the data is not
    // overdispersed and we start at the Poisson-limit clamp.
    let (wsum, wmu, wpearson) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64, 0.0_f64);
            }
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(1e-300);
            let resid = y[i] - mui;
            (wi, wi * mui, wi * resid * resid / mui)
        })
        .reduce(
            || (0.0_f64, 0.0_f64, 0.0_f64),
            |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
        );
    if wsum <= 0.0 {
        return 1.0;
    }
    let mu_bar = wmu / wsum;
    let pearson_ratio = wpearson / wsum;
    let mut theta = if pearson_ratio > 1.0 + 1e-6 {
        (mu_bar / (pearson_ratio - 1.0)).clamp(NEGBIN_THETA_MIN, NEGBIN_THETA_MAX)
    } else {
        // Not overdispersed at this η: the score stays positive throughout, so
        // the MLE is the Poisson-limit clamp. Probe it directly below.
        NEGBIN_THETA_MAX
    };

    // If even at THETA_MAX the score is non-negative, the data carries no
    // resolvable overdispersion and the MLE is the Poisson limit.
    let (score_hi, _) = negbin_theta_score_and_info(y, eta, priorweights, NEGBIN_THETA_MAX);
    if !score_hi.is_finite() {
        return 1.0;
    }
    if score_hi >= 0.0 {
        return NEGBIN_THETA_MAX;
    }
    // The interior root is bracketed by (lo, hi) with S(lo) > 0, S(hi) < 0.
    let (score_lo, _) = negbin_theta_score_and_info(y, eta, priorweights, NEGBIN_THETA_MIN);
    if !score_lo.is_finite() || score_lo <= 0.0 {
        // Degenerate: no sign change in the admissible band. Fall back to the
        // heaviest-overdispersion clamp (S(0⁺)=+∞ guarantees this is the side
        // the root would lie toward if one existed numerically).
        return NEGBIN_THETA_MIN;
    }
    let mut lo = NEGBIN_THETA_MIN;
    let mut hi = NEGBIN_THETA_MAX;
    theta = theta.clamp(lo, hi);

    const MAX_NEWTON_ITERS: usize = 100;
    const REL_TOL: f64 = 1e-10;
    for _ in 0..MAX_NEWTON_ITERS {
        let (score, info) = negbin_theta_score_and_info(y, eta, priorweights, theta);
        if !score.is_finite() {
            break;
        }
        // Maintain the sign-bracket: S decreasing ⇒ S>0 on the low side.
        if score > 0.0 {
            lo = theta;
        } else {
            hi = theta;
        }
        // Safeguarded Newton: `S` decreasing ⇒ `I = −S' > 0`; the Newton step is
        // `θ + S/I`. Take it only when it stays strictly inside the bracket,
        // otherwise bisect.
        let next = if info.is_finite() && info > 0.0 {
            let candidate = theta + score / info;
            if candidate > lo && candidate < hi {
                candidate
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        if (next - theta).abs() <= REL_TOL * theta.max(1.0) {
            theta = next;
            break;
        }
        theta = next;
    }
    theta.clamp(NEGBIN_THETA_MIN, NEGBIN_THETA_MAX)
}


#[derive(Clone, Debug)]
pub struct SparsePirlsDecision {
    pub path: PirlsLinearSolvePath,
    pub reason: &'static str,
    pub p: usize,
    pub nnz_x: usize,
    pub nnz_xtwx_symbolic: Option<usize>,
    pub nnz_s_lambda: usize,
    pub nnz_h_est: Option<usize>,
    pub density_h_est: Option<f64>,
}


fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "na".to_string())
}


fn fmt_opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "na".to_string())
}


impl SparsePirlsDecision {
    fn path_str(&self) -> &'static str {
        match self.path {
            PirlsLinearSolvePath::DenseTransformed => "dense_transformed",
            PirlsLinearSolvePath::SparseNative => "sparse_native",
        }
    }

    fn format_fields(&self, path: &str) -> String {
        format!(
            "path={path} reason={} p={} nnz_x={} nnz_xtwx_symbolic={} nnz_s_lambda={} nnz_h_est={} density_h_est={}",
            self.reason,
            self.p,
            self.nnz_x,
            fmt_opt_usize(self.nnz_xtwx_symbolic),
            self.nnz_s_lambda,
            fmt_opt_usize(self.nnz_h_est),
            fmt_opt_f64(self.density_h_est),
        )
    }

    fn log_once(&self) {
        let path = self.path_str();
        let key = self.format_fields(path);
        let repetition_count = pirls_decision_repetition_count(key.clone());
        if repetition_count == 1 {
            log::debug!("[pirls-path] {key}");
            return;
        }

        if should_log_pirls_decision_summary(repetition_count) {
            log::debug!(
                "[pirls-path] repeated path={} reason={} count={} (suppressing identical decisions)",
                path,
                self.reason,
                repetition_count,
            );
        }
    }
}


fn pirls_decision_repetition_count(log_key: String) -> usize {
    static PIRLS_DECISION_LOG_COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    let counts = PIRLS_DECISION_LOG_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut counts = counts.lock().expect("pirls decision log counter poisoned");
    let count = counts.entry(log_key).or_insert(0);
    *count += 1;
    *count
}


fn should_log_pirls_decision_summary(repetition_count: usize) -> bool {
    repetition_count > 1 && repetition_count.is_power_of_two()
}


const SPARSE_NATIVE_MAX_H_DENSITY: f64 = 0.30;


#[derive(Clone, Debug)]
struct SparsePenaltyPattern {
    upper_triplets: Vec<(usize, usize, f64)>,
    nnz_upper: usize,
}


impl SparsePenaltyPattern {
    fn from_dense_upper(matrix: &Array2<f64>, tol: f64) -> Self {
        let p = matrix.nrows().min(matrix.ncols());
        let mut upper_triplets = Vec::new();
        for col in 0..p {
            for row in 0..=col {
                let value = matrix[[row, col]];
                if value.abs() > tol {
                    upper_triplets.push((row, col, value));
                }
            }
        }
        let nnz_upper = upper_triplets.len();
        Self {
            upper_triplets,
            nnz_upper,
        }
    }
}


#[derive(Clone, Debug)]
pub(crate) struct SparsePenalizedSystemStats {
    pub(crate) nnz_xtwx_symbolic: usize,
    pub(crate) nnz_s_lambda_upper: usize,
    pub(crate) nnz_h_upper: usize,
    pub(crate) density_upper: f64,
}


// Phase 2 sparse-native PIRLS will reuse this cache for symbolic structure and
// repeated numeric assembly of H = X'WX + S_lambda + ridge I.
//
// This is the natural insertion point for any future selected-inversion /
// Takahashi trace backend. In original spline coefficient order, the assembled
// penalized system can remain sparse/banded, so exact traces like
// tr(H^{-1} S_k) can be computed from a sparse factorization without ever
// materializing a dense inverse. That is not true after the REML
// reparameterization rotates the problem into the dense Qs basis.
//
// Algebra:
//   H = X'WX + sum_k lambda_k S_k + delta I
// and the REML/LAML first-order trace terms have the form
//   T_k = tr(H^{-1} S_k).
// Since tr(AB) = sum_ij A_ij B_ji, for symmetric sparse S_k we only need
// inverse entries on the support of S_k:
//   T_k = sum_{(i,j) in nz(S_k), i>=j} (2 - 1{i=j}) (H^{-1})_{ij} (S_k)_{ij}.
// Takahashi/selected inversion exploits exactly this fact. Given a sparse
// Cholesky-type factorization H = LDL', it computes only those entries of
// H^{-1} that lie on the filled graph of L, which contains the structural
// nonzeros needed for spline penalties. For banded spline systems with
// half-bandwidth b, the work scales like sum_j |N(j)|^2 = O(p b^2) instead of
// dense O(p^3), where N(j) is the subdiagonal nonzero pattern of column j of L.
struct SparsePenalizedSystemCache {
    xtwx_cache: SparseXtWxCache,
    penalty_pattern: SparsePenaltyPattern,
    h_upper_symbolic: SymbolicSparseColMat<usize>,
    h_uppervalues: Vec<f64>,
    h_upper_col_ptr: Vec<usize>,
    h_upperrow_idx: Vec<usize>,
    p: usize,
}


impl SparsePenalizedSystemCache {
    fn new(
        x: &SparseColMat<usize, f64>,
        penalty_pattern: SparsePenaltyPattern,
    ) -> Result<Self, EstimationError> {
        let xtwx_cache = SparseXtWxCache::new(x)?;
        let p = x.ncols();
        let h_upper_symbolic = build_penalized_symbolic(
            p,
            xtwx_cache.xtwx_symbolic.col_ptr(),
            xtwx_cache.xtwx_symbolic.row_idx(),
            &penalty_pattern.upper_triplets,
        )?;
        let h_uppervalues = vec![0.0; h_upper_symbolic.row_idx().len()];
        Ok(Self {
            xtwx_cache,
            penalty_pattern,
            h_upper_col_ptr: h_upper_symbolic.col_ptr().to_vec(),
            h_upperrow_idx: h_upper_symbolic.row_idx().to_vec(),
            h_upper_symbolic,
            h_uppervalues,
            p,
        })
    }

    fn matches(
        &self,
        x: &SparseColMat<usize, f64>,
        penalty_pattern: &SparsePenaltyPattern,
    ) -> bool {
        self.xtwx_cache.matches(x)
            && self.penalty_pattern.nnz_upper == penalty_pattern.nnz_upper
            && self.penalty_pattern.upper_triplets == penalty_pattern.upper_triplets
    }

    fn stats(&self) -> SparsePenalizedSystemStats {
        let upper_total = self.p.saturating_mul(self.p + 1) / 2;
        SparsePenalizedSystemStats {
            nnz_xtwx_symbolic: self.xtwx_cache.xtwx_symbolic.row_idx().len(),
            nnz_s_lambda_upper: self.penalty_pattern.nnz_upper,
            nnz_h_upper: self.h_upper_symbolic.row_idx().len(),
            density_upper: if upper_total == 0 {
                0.0
            } else {
                self.h_upper_symbolic.row_idx().len() as f64 / upper_total as f64
            },
        }
    }

    fn assemble_upper(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        ridge: f64,
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        if weights.len() != self.xtwx_cache.nrows {
            crate::bail_invalid_estim!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.xtwx_cache.nrows
            );
        }
        // Gaussian-Identity fast path: when the caller has pre-built the
        // `XᵀWX` numerical values (weights are constant across the outer
        // loop), install them into the inner cache and skip the SpGEMM.
        // We verify symbolic-pattern equivalence first; on mismatch we
        // fall back to the regular per-call recomputation rather than
        // installing values keyed to a different sparsity layout.
        let use_precomputed = match precomputed_xtwx {
            Some(pre) => {
                let col_ptr_ok =
                    pre.xtwx_symbolic_col_ptr.as_slice() == self.xtwx_cache.xtwx_symbolic.col_ptr();
                let row_idx_ok =
                    pre.xtwx_symbolic_row_idx.as_slice() == self.xtwx_cache.xtwx_symbolic.row_idx();
                let values_ok = pre.xtwxvalues.len() == self.xtwx_cache.xtwxvalues.len();
                if col_ptr_ok && row_idx_ok && values_ok {
                    self.xtwx_cache.xtwxvalues.copy_from_slice(&pre.xtwxvalues);
                    true
                } else {
                    log::warn!(
                        "[sparse-xtwx-cache] precomputed XᵀWX pattern mismatch; \
                         falling back to per-call recompute"
                    );
                    false
                }
            }
            None => false,
        };
        if !use_precomputed {
            self.xtwx_cache.compute_numeric(x, weights)?;
        }
        self.h_uppervalues.fill(0.0);

        let mut cursor = self.h_upper_col_ptr[..self.p].to_vec();

        let xtwx_col_ptr = self.xtwx_cache.xtwx_symbolic.col_ptr();
        let xtwxrow_idx = self.xtwx_cache.xtwx_symbolic.row_idx();
        for col in 0..self.p {
            let start = xtwx_col_ptr[col];
            let end = xtwx_col_ptr[col + 1];
            for idx in start..end {
                let row = xtwxrow_idx[idx];
                if row <= col {
                    let cursor_idx = &mut cursor[col];
                    while *cursor_idx < self.h_upper_col_ptr[col + 1]
                        && self.h_upperrow_idx[*cursor_idx] < row
                    {
                        *cursor_idx += 1;
                    }
                    if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                        || self.h_upperrow_idx[*cursor_idx] != row
                    {
                        crate::bail_invalid_estim!("penalized symbolic pattern missing XtWX entry");
                    }
                    self.h_uppervalues[*cursor_idx] += self.xtwx_cache.xtwxvalues[idx];
                }
            }
        }

        cursor.copy_from_slice(&self.h_upper_col_ptr[..self.p]);
        for &(row, col, value) in &self.penalty_pattern.upper_triplets {
            let cursor_idx = &mut cursor[col];
            while *cursor_idx < self.h_upper_col_ptr[col + 1]
                && self.h_upperrow_idx[*cursor_idx] < row
            {
                *cursor_idx += 1;
            }
            if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                || self.h_upperrow_idx[*cursor_idx] != row
            {
                crate::bail_invalid_estim!("penalized symbolic pattern missing penalty entry");
            }
            self.h_uppervalues[*cursor_idx] += value;
        }

        if ridge > 0.0 {
            cursor.copy_from_slice(&self.h_upper_col_ptr[..self.p]);
            for col in 0..self.p {
                let cursor_idx = &mut cursor[col];
                while *cursor_idx < self.h_upper_col_ptr[col + 1]
                    && self.h_upperrow_idx[*cursor_idx] < col
                {
                    *cursor_idx += 1;
                }
                if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                    || self.h_upperrow_idx[*cursor_idx] != col
                {
                    crate::bail_invalid_estim!("penalized symbolic pattern missing diagonal entry");
                }
                self.h_uppervalues[*cursor_idx] += ridge;
            }
        }

        Ok(SparseColMat::new(
            self.h_upper_symbolic.clone(),
            self.h_uppervalues.clone(),
        ))
    }
}


fn build_penalized_symbolic(
    p: usize,
    xtwx_col_ptr: &[usize],
    xtwxrow_idx: &[usize],
    penalty_triplets: &[(usize, usize, f64)],
) -> Result<SymbolicSparseColMat<usize>, EstimationError> {
    let mut cols: Vec<BTreeSet<usize>> = (0..p).map(|_| BTreeSet::new()).collect();
    for col in 0..p {
        cols[col].insert(col);
        let start = xtwx_col_ptr[col];
        let end = xtwx_col_ptr[col + 1];
        for &row in &xtwxrow_idx[start..end] {
            if row <= col {
                cols[col].insert(row);
            }
        }
    }
    for &(row, col, _) in penalty_triplets {
        if row > col || col >= p {
            crate::bail_invalid_estim!(
                "penalty sparse pattern must be upper-triangular within bounds"
            );
        }
        cols[col].insert(row);
    }

    let mut col_ptr = Vec::with_capacity(p + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for rows in cols {
        row_idx.extend(rows.into_iter());
        col_ptr.push(row_idx.len());
    }
    // `cols` has exactly p BTreeSet columns. Draining them into CSC order
    // gives p+1 monotone col_ptr entries ending at row_idx.len(), and each
    // per-column row slice is sorted and duplicate-free. Every inserted row
    // satisfies row <= col < p: diagonal and XᵀWX entries are inserted only
    // for the current column's upper triangle, and penalty triplets were
    // checked above.
    // SAFETY: the generated col_ptr length, monotonicity, terminal nnz,
    // sorted per-column rows, absence of duplicates, and row bounds are
    // exactly the CSC invariants skipped by new_unchecked.
    Ok(unsafe { SymbolicSparseColMat::new_unchecked(p, p, col_ptr, None, row_idx) })
}


pub trait WorkingModel {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError>;

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        curvature_kind: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        assert!(core::mem::size_of_val(&curvature_kind) > 0);
        self.update(beta)
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, curvature)
    }

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        arr: &Array1<f64>,
        linear_predictor: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(std::mem::size_of_val(linear_predictor) > 0);
        self.update_candidate(beta, curvature)
            .map(CandidateEvaluation::Full)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        false
    }
}


/// Result of a cheap LM-candidate screen: penalized objective + arithmetic
/// finiteness, without the gradient/Hessian needed for an accepted step.
#[derive(Debug, Clone)]
pub struct CandidateScreen {
    pub penalized_objective: f64,
    pub deviance: f64,
    pub penalty_term: f64,
    pub arithmetic_finite: bool,
}


/// Outcome of `WorkingModel::screen_candidate`: either a cheap screen result
/// (LM loop must upgrade with `update_with_curvature` on acceptance) or the
/// full state when screening was not applicable.
pub enum CandidateEvaluation {
    Screen(CandidateScreen),
    Full(WorkingState),
}


impl CandidateEvaluation {
    #[inline]
    fn penalized_objective(&self, firth_bias_reduction: bool) -> f64 {
        match self {
            Self::Screen(s) => s.penalized_objective,
            Self::Full(state) => {
                let mut value = state.deviance + state.penalty_term;
                if firth_bias_reduction && let Some(j) = state.jeffreys_logdet() {
                    value -= 2.0 * j;
                }
                value
            }
        }
    }

    #[inline]
    fn arithmetic_finite(&self) -> bool {
        match self {
            Self::Screen(s) => s.arithmetic_finite,
            Self::Full(state) => state.gradient.iter().all(|g| g.is_finite()),
        }
    }

    #[inline]
    fn into_full(self) -> Option<WorkingState> {
        match self {
            Self::Full(state) => Some(state),
            Self::Screen(_) => None,
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PirlsAcceptedStateCacheKey {
    curvature: HessianCurvatureKind,
    firth_active: bool,
    beta_bits: Vec<u64>,
    arrow_latent_bits: Option<Vec<u64>>,
}


impl PirlsAcceptedStateCacheKey {
    fn requested(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        Self::new(beta, curvature, options.firth_bias_reduction, options)
    }

    fn accepted(
        beta: &Coefficients,
        state: &WorkingState,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        Self::new(
            beta,
            state.hessian_curvature,
            matches!(state.firth, FirthDiagnostics::Active { .. }),
            options,
        )
    }

    fn new(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        firth_active: bool,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        let arrow_latent_bits = options.arrow_schur.as_ref().map(|arrow_cfg| {
            arrow_cfg.snapshot_t.as_ref()()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        });
        Self {
            curvature,
            firth_active,
            beta_bits: beta.as_ref().iter().map(|value| value.to_bits()).collect(),
            arrow_latent_bits,
        }
    }
}


/// Uncertainty inputs for integrated (GHQ) IRLS updates.
#[derive(Clone, Copy)]
pub(crate) struct IntegratedWorkingInput<'a> {
    pub quadctx: &'a crate::quadrature::QuadratureContext,
    pub se: ArrayView1<'a, f64>,
    pub mixture_link_state: Option<&'a MixtureLinkState>,
    pub sas_link_state: Option<&'a SasLinkState>,
}


pub struct WorkingDerivativeBuffersMut<'a> {
    c: &'a mut Array1<f64>,
    d: &'a mut Array1<f64>,
    dmu_deta: &'a mut Array1<f64>,
    d2mu_deta2: &'a mut Array1<f64>,
    d3mu_deta3: &'a mut Array1<f64>,
}


/// Contiguous mutable views of the three core working buffers (`mu`, `weights`,
/// `z`) shared by every PIRLS working-state writer.
pub(super) struct WorkingSlices<'a> {
    pub mu: &'a mut [f64],
    pub weights: &'a mut [f64],
    pub z: &'a mut [f64],
}


/// Contiguous mutable views of the Newton derivative/curvature buffers
/// (`c`, `d`, `dmu/deta` jet) shared by the full-derivative PIRLS writers.
pub(super) struct WorkingDerivSlices<'a> {
    pub c: &'a mut [f64],
    pub d: &'a mut [f64],
    pub dmu: &'a mut [f64],
    pub d2: &'a mut [f64],
    pub d3: &'a mut [f64],
}


/// Canonical "contiguous-or-panic" unpacking of the three core working buffers.
///
/// Single source of truth for the contiguity contract and panic messages that
/// every working-state writer relies on; every writer routes through this.
#[inline]
pub(super) fn working_slices<'a>(
    mu: &'a mut Array1<f64>,
    weights: &'a mut Array1<f64>,
    z: &'a mut Array1<f64>,
) -> WorkingSlices<'a> {
    WorkingSlices {
        mu: mu.as_slice_mut().expect("mu must be contiguous"),
        weights: weights.as_slice_mut().expect("weights must be contiguous"),
        z: z.as_slice_mut().expect("z must be contiguous"),
    }
}


/// Canonical "contiguous-or-panic" unpacking of the Newton derivative buffers.
///
/// Single source of truth for the contiguity contract and panic messages of the
/// `c`/`d`/`dmu`/`d2`/`d3` curvature buffers; every full-derivative writer routes
/// through this.
#[inline]
pub(super) fn working_deriv_slices<'a>(
    derivs: &'a mut WorkingDerivativeBuffersMut<'_>,
) -> WorkingDerivSlices<'a> {
    WorkingDerivSlices {
        c: derivs.c.as_slice_mut().expect("c must be contiguous"),
        d: derivs.d.as_slice_mut().expect("d must be contiguous"),
        dmu: derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous"),
        d2: derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous"),
        d3: derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous"),
    }
}


#[derive(Clone, Copy)]
struct WorkingBernoulliGeometry {
    mu: f64,
    weight: f64,
    z: f64,
    c: f64,
    d: f64,
}


/// Shared likelihood interface used by PIRLS working updates.
///
/// This keeps the update/deviance math in one place so engine-level likelihoods
/// and higher-level wrappers (custom family, GAMLSS warm starts) can share a
/// consistent implementation.
pub(crate) trait WorkingLikelihood {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError>;

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError>;
}


impl WorkingLikelihood for GlmLikelihoodSpec {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError> {
        match (&self.spec.response, &self.spec.link, integrated.is_some()) {
            (ResponseFamily::Binomial, _, true) => {
                let integ = integrated.unwrap();
                update_glmvectors_integrated_by_family(
                    integ.quadctx,
                    y,
                    eta,
                    integ.se,
                    &self.spec,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                    integ.mixture_link_state,
                    integ.sas_link_state,
                )?;
                Ok(())
            }
            (ResponseFamily::Binomial, link, false) => {
                if matches!(link, InverseLink::Mixture(_)) {
                    crate::bail_invalid_estim!(
                        "BinomialMixture IRLS update requires explicit mixture link state"
                            .to_string(),
                    );
                }
                update_glmvectors(
                    y,
                    eta,
                    &self.spec.link,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Gaussian, _, _) => {
                update_glmvectors(
                    y,
                    eta,
                    &InverseLink::Standard(StandardLink::Identity),
                    priorweights,
                    mu,
                    weights,
                    z,
                    None,
                )?;
                // For Gaussian identity, the canonical IRLS working weight is
                //     w_i = prior_i * (dmu/deta)^2 / Var(Y_i | mu_i) = prior_i / phi.
                // When the scale metadata explicitly fixes phi (rather than
                // profiling sigma out), the working weights must include 1/phi
                // so that PIRLS minimises the scaled deviance / scaled negative
                // log-likelihood that the calibrator and downstream variance
                // calculations expect. `ProfiledGaussian` returns `None` here,
                // preserving the historical "weights == prior" behaviour for
                // the default profiled case.
                if let Some(phi) = self.scale.fixed_phi() {
                    if !(phi.is_finite() && phi > 0.0) {
                        crate::bail_invalid_estim!(
                            "Gaussian fixed dispersion phi must be finite and positive (got {})",
                            phi
                        );
                    }
                    if phi != 1.0 {
                        let inv_phi = 1.0 / phi;
                        weights.mapv_inplace(|w| w * inv_phi);
                    }
                }
                Ok(())
            }
            (ResponseFamily::Poisson, _, _) => {
                write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives);
                Ok(())
            }
            (ResponseFamily::Tweedie { p }, _, _) => {
                let p = *p;
                write_tweedie_log_working_state(
                    y,
                    eta,
                    priorweights,
                    p,
                    fixed_glm_dispersion(self),
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::NegativeBinomial { theta, .. }, _, _) => {
                let theta = *theta;
                write_negative_binomial_log_working_state(
                    y,
                    eta,
                    priorweights,
                    theta,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Beta { phi }, _, _) => {
                let phi = *phi;
                write_beta_logit_working_state(
                    y,
                    eta,
                    priorweights,
                    phi,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Gamma, _, _) => {
                write_gamma_log_working_state(
                    y,
                    eta,
                    priorweights,
                    self.gamma_shape().unwrap_or(1.0),
                    mu,
                    weights,
                    z,
                    derivatives,
                );
                Ok(())
            }
            (ResponseFamily::RoystonParmar, _, _) => Err(EstimationError::InvalidInput(
                "RoystonParmar is survival-specific and not a GLM IRLS family".to_string(),
            )),
        }
    }

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError> {
        if matches!(self.spec.response, ResponseFamily::Tweedie { .. }) {
            validate_tweedie_responses(&y, &priorweights)?;
        }
        Ok(calculate_deviance(y, mu, self, priorweights))
    }
}


// Suggestion #6: Preallocate and reuse iteration workspaces
pub struct PirlsWorkspace {
    // Common IRLS buffers. Only O(n) state is kept persistently; any
    // design-weighted n x p scratch must be streamed through bounded chunks.
    pub wz: Array1<f64>,
    pub eta_buf: Array1<f64>,
    // Stage 2/4 assembly (use max needed sizes)
    pub scaled_matrix: Array2<f64>,    // (<= p + ebrows) x p
    pub final_aug_matrix: Array2<f64>, // (<= p + erows) x p
    // Stage 5 RHS buffers
    pub rhs_full: Array1<f64>, // length <= p + erows
    // Gradient check helpers
    pub working_residual: Array1<f64>,
    pub weighted_residual: Array1<f64>,
    // Step-halving direction (XΔβ)
    pub delta_eta: Array1<f64>,
    // Preallocated buffer for GEMV results (length p)
    pub vec_buf_p: Array1<f64>,
    // Cached sparse penalized-system workspace for sparse-native solve eligibility/assembly.
    sparse_penalized_system_cache: Option<SparsePenalizedSystemCache>,
    // Factorization scratch (avoid per-iteration allocation)
    pub factorization_scratch: MemBuffer,
    // Permutation buffers for LDLT
    pub perm: Vec<usize>,
    pub perm_inv: Vec<usize>,
    // Buffer for in-place factorization (preserves original Hessian in WorkingState)
    pub factorization_matrix: Array2<f64>,
    // Buffer for sparse matrix scaling (avoid per-iteration allocation)
    pub weighted_xvalues: Vec<f64>,
    // Dense chunk buffer for streaming X'WX assembly on very large n.
    pub weighted_x_chunk: Array2<f64>,
    // Reusable p×p buffer for Hessian assembly (avoids per-iteration allocation).
    pub hessian_buf: Array2<f64>,
    // Reusable n-length buffer for X*β matvec (avoids per-iteration allocation in update).
    pub matvec_buf: Array1<f64>,
}


impl PirlsWorkspace {
    pub fn new(n: usize, p: usize, idx: usize, idx2: usize) -> Self {
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
        // Stage buffers are allocated lazily: historically these were pre-sized to
        // worst-case dimensions, which inflates memory when many PIRLS workspaces
        // exist concurrently (e.g. parallel REML evals).
        // The active code paths resize-on-demand where needed.

        PirlsWorkspace {
            wz: Array1::zeros(n),
            eta_buf: Array1::zeros(n),
            scaled_matrix: Array2::zeros((0, 0).f()),
            final_aug_matrix: Array2::zeros((0, 0).f()),
            rhs_full: Array1::zeros(0),
            working_residual: Array1::zeros(n),
            weighted_residual: Array1::zeros(n),
            delta_eta: Array1::zeros(n),
            vec_buf_p: Array1::zeros(p),
            sparse_penalized_system_cache: None,
            // Keep scratch minimal at init; grow only if/when a factorization path
            // needs it.
            factorization_scratch: {
                let par = faer::Par::Seq;
                let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                    1,
                    par,
                    Spec::new(<LltParams as Auto<f64>>::auto()),
                );
                MemBuffer::new(req)
            },
            perm: vec![0; p],
            perm_inv: vec![0; p],
            factorization_matrix: Array2::zeros((0, 0)),
            weighted_xvalues: Vec::new(),
            weighted_x_chunk: Array2::zeros((0, 0).f()),
            hessian_buf: Array2::zeros((0, 0).f()),
            matvec_buf: Array1::zeros(n),
        }
    }

    pub(super) fn add_dense_xtwx_signed(
        weights: &Array1<f64>,
        weighted_x_scratch: &mut Array2<f64>,
        x: &Array2<f64>,
        out: &mut Array2<f64>,
    ) {
        *out = crate::solver::estimate::reml::assembly::xt_diag_x_dense_into(
            x,
            weights,
            weighted_x_scratch,
        );
    }

    /// Ensure the sparse penalty cache is populated and consistent with `x` and `s_lambda`.
    fn ensure_sparse_penalty_cache(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<(), EstimationError> {
        let penalty_pattern = SparsePenaltyPattern::from_dense_upper(s_lambda, 1e-12);
        let rebuild = match self.sparse_penalized_system_cache.as_ref() {
            Some(cache) => !cache.matches(x, &penalty_pattern),
            None => true,
        };
        if rebuild {
            self.sparse_penalized_system_cache =
                Some(SparsePenalizedSystemCache::new(x, penalty_pattern)?);
        }
        Ok(())
    }

    pub(crate) fn sparse_penalized_system_stats(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<SparsePenalizedSystemStats, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        Ok(self.sparse_penalized_system_cache.as_ref().unwrap().stats())
    }

    // Phase 2 hook: numeric sparse penalized-system assembly in original coordinates.
    pub(super) fn assemble_sparse_penalized_hessian(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
        ridge: f64,
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        self.sparse_penalized_system_cache
            .as_mut()
            .unwrap()
            .assemble_upper(x, weights, ridge, precomputed_xtwx)
    }
}


#[derive(Clone, Debug)]
pub struct WorkingModelPirlsOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
    pub max_step_halving: usize,
    pub min_step_size: f64,
    pub firth_bias_reduction: bool,
    /// Optional lower bounds on coefficients (same coordinate system as `beta`).
    /// Use `-inf` for unconstrained entries.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional linear inequality constraints in current coefficient coordinates:
    ///   A * beta >= b.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    /// Optional warm-start hint for the Levenberg-Marquardt damping
    /// coefficient. When set, the inner solver seeds `λ_LM` to this
    /// value instead of the default `1e-6`. Clamped on consumption to
    /// `[1e-6, 1e-3]` so a stale or pathological hint cannot poison the
    /// solve: the upper bound costs at most three damping halvings
    /// versus the cold default, which is dwarfed by the savings when
    /// the hint is informative.
    ///
    /// Used by `execute_pirls_if_needed` (in `solver::reml::runtime`)
    /// to persist the converged λ across consecutive PIRLS calls in a
    /// single REML outer optimization, so the inner Newton does not
    /// have to rediscover problem-specific damping at every accepted
    /// outer iterate.
    pub initial_lm_lambda: Option<f64>,
    /// Enable the Transtrum-Sethna geodesic-acceleration second-order
    /// correction on each accepted Levenberg-Marquardt step. When true,
    /// after the standard LM direction `δp = −(H + λ_lm·diag(H))⁻¹ g`
    /// is computed and accepted by the LM gain test, the solver computes
    /// a finite-difference estimate of the directional second derivative
    /// of the gradient along `δp`, solves a *second* linear system with
    /// the same (already-factored) Hessian, and adds the correction
    /// `δp₂` to the step only if `‖δp₂‖ ≤ α‖δp‖` (the Transtrum-Sethna
    /// 2011 acceptance criterion, α = 0.75 here). The correction costs
    /// two extra full `WorkingModel::update` calls per accepted step
    /// (for the FD evaluations); it is most useful for fits whose
    /// penalized Hessian is near-singular (latent-coordinate fits,
    /// near-collinear bases). Default `false`; opt-in until validated
    /// across the broader family of likelihoods and penalties.
    pub geodesic_acceleration: bool,
    /// Optional arrow-Schur structured-inner-solve descriptor.
    ///
    /// When `Some`, every accepted LM Newton step inside the inner loop
    /// is computed by the per-observation arrow-Schur path
    /// ([`crate::solver::arrow_schur::ArrowSchurSystem`]) instead of the
    /// β-only `solve_newton_direction_dense`. When `None`, the existing
    /// β-only path is used unchanged (back-compat: every existing call
    /// site that does not opt in is unaffected).
    ///
    /// **Scope note.** This wires the *inner* Gauss–Newton step. The REML
    /// outer-loop gradient w.r.t. `t` (which carries a shared `Schur⁻¹`
    /// factor) is a separate plumbing change owned by the REML driver and is
    /// **not** handled here.
    pub arrow_schur: Option<ArrowSchurInnerConfig>,
}


/// Per-iteration arrow-Schur builder hook.
///
/// The driver supplies a closure that, given the current `β` iterate,
/// returns a freshly-populated [`crate::solver::arrow_schur::ArrowSchurSystem`]
/// — i.e. the per-row `H_tt^(i)`, `H_tβ^(i)`, `g_t^(i)` blocks and the
/// β-block `H_ββ`, `g_β`. The driver owns the assembly because the
/// per-row Jacobians depend on the latent-coord term's basis (Duchon,
/// Sphere, …) and the analytic-penalty contributions depend on the
/// registry the outer-fit configuration owns. PIRLS only knows how to
/// *solve* the bordered system once it has been assembled.
#[derive(Clone)]
pub struct ArrowSchurInnerConfig {
    /// Number of latent rows `N`.
    pub n_rows: usize,
    /// Latent dimensionality `d`.
    pub latent_dim: usize,
    /// β dimensionality `K` (must match the inner Hessian dimension).
    pub n_beta: usize,
    /// Closure that builds the bordered system at the current `β` and
    /// current latent `t` (the latter held externally by the driver, e.g.
    /// in a `LatentCoordValues` registered alongside the working model).
    /// Returning `None` signals "fall back to the β-only path for this
    /// iteration" — useful for the seeding sweep before `t` has been
    /// initialized.
    pub build: std::sync::Arc<
        dyn Fn(&Array1<f64>) -> Option<crate::solver::arrow_schur::ArrowSchurSystem> + Send + Sync,
    >,
    /// BA Schur solve mode. `None` selects Direct for `K <= 2000` and
    /// InexactPCG above, following "Bundle Adjustment in the Large".
    pub solver_mode: Option<crate::solver::arrow_schur::ArrowSolverMode>,
    /// When set, assemble the reduced dense Schur block in row chunks.
    pub streaming_chunk_size: Option<usize>,
    /// Steihaug trust-region radius for the reduced shared step. This ports
    /// the Ceres/BA trust-region guard while retaining PIRLS's LM damping.
    pub trust_region_radius: f64,
    /// Optional β-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// When `Some`, the PIRLS driver calls
    /// [`crate::solver::arrow_schur::ArrowSchurSystem::set_block_offsets`] on
    /// every system returned by the `build` closure, wiring the block-Jacobi
    /// path without requiring each family's closure to call it manually.
    ///
    /// Derive from `ParameterBlockSpec` slices via
    /// [`crate::families::custom_family::block_offsets_from_specs`].  When
    /// `None`, the preconditioner falls back to scalar-diagonal Jacobi (the
    /// pre-#287 behaviour); when `Some([])` (empty slice), the same fallback
    /// applies.
    pub block_offsets: Option<Arc<[std::ops::Range<usize>]>>,
    /// Callback that the inner solver invokes after each LM-attempted
    /// joint step to write the latent tangent increment back into the
    /// driver's `LatentCoordValues` via that latent's update rule
    /// (`retract_flat_delta` for manifold latents). `delta_t` is the flat
    /// row-major increment of length `n_rows * latent_dim`.
    pub apply_delta_t: std::sync::Arc<dyn Fn(&Array1<f64>) + Send + Sync>,
    /// Snapshot the driver's latent field before an LM trial step mutates it.
    pub snapshot_t: std::sync::Arc<dyn Fn() -> Array1<f64> + Send + Sync>,
    /// Restore a snapshot produced by [`Self::snapshot_t`] after any rejected
    /// LM trial. Accepted trials deliberately do not call this hook: β and t
    /// commit together.
    pub restore_t: std::sync::Arc<dyn Fn(&Array1<f64>) + Send + Sync>,
}


impl std::fmt::Debug for ArrowSchurInnerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowSchurInnerConfig")
            .field("n_rows", &self.n_rows)
            .field("latent_dim", &self.latent_dim)
            .field("n_beta", &self.n_beta)
            .field("solver_mode", &self.solver_mode)
            .field("streaming_chunk_size", &self.streaming_chunk_size)
            .field("trust_region_radius", &self.trust_region_radius)
            .field(
                "block_offsets",
                &self.block_offsets.as_ref().map(|o| o.len()),
            )
            .finish_non_exhaustive()
    }
}


fn restore_arrow_latent_if_needed(
    options: &WorkingModelPirlsOptions,
    snapshot: Option<Array1<f64>>,
) {
    if let (Some(arrow_cfg), Some(snapshot)) = (options.arrow_schur.as_ref(), snapshot) {
        arrow_cfg.restore_t.as_ref()(&snapshot);
    }
}


pub(super) fn restore_pending_arrow_latent_if_needed(
    options: &WorkingModelPirlsOptions,
    pending_snapshot: &mut Option<Array1<f64>>,
) {
    restore_arrow_latent_if_needed(options, pending_snapshot.take());
}


pub(super) fn commit_pending_arrow_latent(pending_snapshot: &mut Option<Array1<f64>>) {
    drop(pending_snapshot.take());
}


// Fixed stabilization ridge for PIRLS/PLS. `penalty_term` carries this as
// ridge * ||beta||^2 (equivalently 0.5 * ridge * ||beta||^2 in the
// 0.5 * (deviance + penalty_term) objective), and it is constant w.r.t. rho.
//
// Math note:
//   Objective: V(ρ) includes log|H(ρ)| with H(ρ) = X' W X + S_λ(ρ) + δ I.
//   If δ = δ(ρ) is adaptive, V(ρ) is only piecewise-smooth and ∂V/∂ρ ignores
//   ∂δ/∂ρ, causing a mismatch between the optimized surface and the analytic
//   derivative surface. Using a fixed δ makes V(ρ) smooth and the standard
//   envelope-theorem gradient valid:
//     dV/dρ_k = 0.5 λ_k βᵀ S_k β + 0.5 λ_k tr(H^{-1} S_k) - 0.5 det1[k].
pub(super) const FIXED_STABILIZATION_RIDGE: f64 = 1e-8;


pub(super) struct GamWorkingModel<'a> {
    x_original: DesignMatrix,
    coordinate_design: WorkingCoordinateDesign,
    offset: Array1<f64>,
    y: ArrayView1<'a, f64>,
    priorweights: ArrayView1<'a, f64>,
    penalty: PirlsPenalty,
    workspace: PirlsWorkspace,
    likelihood: GlmLikelihoodSpec,
    link_kind: InverseLink,
    firth_bias_reduction: bool,
    lastmu: Array1<f64>,
    lastweights: Array1<f64>,
    lastz: Array1<f64>,
    last_c: Array1<f64>,
    last_d: Array1<f64>,
    lasthessian_weights: Array1<f64>,
    lasthessian_c: Array1<f64>,
    lasthessian_d: Array1<f64>,
    lasthessian_curvature: HessianCurvatureKind,
    last_dmu_deta: Array1<f64>,
    last_d2mu_deta2: Array1<f64>,
    last_d3mu_deta3: Array1<f64>,
    last_penalty_term: f64,
    x_original_csr: Option<SparseRowMat<usize, f64>>,
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses integrated family-dispatched working updates.
    covariate_se: Option<Array1<f64>>,
    /// Whether the Gamma dispersion shape has been estimated and frozen for the
    /// duration of this inner P-IRLS solve. The shape (= 1/φ) is a nuisance
    /// scale that multiplies both the working weight (`w = shape·prior`) and the
    /// reported deviance (`2·shape·Σ wᵢ dᵢ`). Re-estimating it per inner Newton/LM
    /// iterate moves the product φ·λ that the penalized argmin β̂ depends on, so
    /// the LM gain ratio compares two different objectives and the solve stalls.
    /// The shape is therefore estimated once from the warm-start η on the first
    /// curvature build and held fixed; it refreshes naturally across *outer*
    /// iterations because a fresh `GamWorkingModel` is built per inner solve.
    /// See issue #511 (regression of #359).
    gamma_shape_locked: bool,
    /// Whether the Beta-regression precision `phi` has been estimated and frozen
    /// for the duration of this inner P-IRLS solve. Like the Gamma shape, `phi`
    /// is a nuisance scale entering the working weight `w ∝ (1+phi)` and the
    /// variance `Var(y)=mu(1-mu)/(1+phi)`; re-estimating it per Newton/LM iterate
    /// moves the penalized argmin, so it is estimated once from the warm-start η
    /// and held fixed within the inner solve, refreshing across outer iterations
    /// (a fresh working model is built per inner solve). Issue #567.
    beta_phi_locked: bool,
    /// Whether the Tweedie dispersion `phi` has been estimated and frozen for the
    /// duration of this inner P-IRLS solve. Like the Gamma shape, `phi` is a
    /// nuisance scale entering only the working weight (`prior·μ^{2−p}/phi`) and
    /// not the working response, so re-estimating it per Newton/LM iterate would
    /// move the product `φ·λ` the penalized argmin β̂ depends on and stall the LM
    /// gain ratio. It is therefore estimated once from the warm-start η and held
    /// fixed within the inner solve, refreshing across outer iterations (a fresh
    /// working model is built per inner solve). Issue #771.
    tweedie_phi_locked: bool,
    /// Whether the Negative-Binomial overdispersion `theta` has been estimated
    /// and frozen for the duration of this inner P-IRLS solve. `theta` enters the
    /// working weight `W = μθ/(θ+μ)` (the NB2 Fisher information) and the working
    /// response, so — like the Beta precision, and unlike the scale-free Gamma
    /// shape — re-estimating it per Newton/LM iterate would move the penalized
    /// argmin β̂ and stall the LM gain ratio. It is therefore estimated once from
    /// the warm-start η and held fixed within the inner solve, refreshing across
    /// outer iterations (a fresh working model is built per inner solve). The
    /// converged-η joint refresh in `loop_driver` re-arms this lock so the
    /// reported `theta` is exactly the ML estimate at the reported η. Issue #802.
    negbin_theta_locked: bool,
    quadctx: crate::quadrature::QuadratureContext,
    /// Frozen-weight first-Fisher-step data-fit Gram `XᵀWX` (#1111 / #1033
    /// mechanism (c)), in the same *original* (conditioned `x_fit`) frame
    /// `penalized_hessian` forms `compute_xtwx_blas(self.x_original, ...)` in,
    /// i.e. BEFORE any Qs conjugation. When present it serves the FIRST
    /// Fisher-scoring iteration's `XᵀWX` n-free, eliding the dominant
    /// O(N·p²) weighted cross-product on a large-n GLM ψ-trial. Consumed at
    /// most once per inner solve (the first `penalized_hessian` build at the
    /// warm β); later iterations restream the true moving `W`.
    glm_first_step_gram: Option<Array2<f64>>,
    /// Set once the frozen-W first-step Gram has been consumed, so subsequent
    /// inner iterations restream `XᵀWX` from the (moving) working weights.
    glm_first_step_gram_consumed: bool,
}


pub(super) struct GamModelFinalState {
    likelihood: GlmLikelihoodSpec,
    coordinate_frame: PirlsCoordinateFrame,
    finalmu: Array1<f64>,
    finalweights: Array1<f64>,
    scoreweights: Array1<f64>,
    finalz: Array1<f64>,
    final_c: Array1<f64>,
    final_d: Array1<f64>,
    final_dmu_deta: Array1<f64>,
    final_d2mu_deta2: Array1<f64>,
    final_d3mu_deta3: Array1<f64>,
    penalty_term: f64,
}


impl<'a> GamWorkingModel<'a> {
    fn new(
        x_transformed: Option<DesignMatrix>,
        x_original: DesignMatrix,
        coordinate_frame: PirlsCoordinateFrame,
        offset: ArrayView1<f64>,
        y: ArrayView1<'a, f64>,
        priorweights: ArrayView1<'a, f64>,
        penalty: PirlsPenalty,
        workspace: PirlsWorkspace,
        likelihood: GlmLikelihoodSpec,
        link_kind: InverseLink,
        firth_bias_reduction: bool,
        transform: Option<WorkingReparamTransform>,
        quadctx: crate::quadrature::QuadratureContext,
        glm_first_step_gram: Option<Array2<f64>>,
    ) -> Self {
        let coordinate_design = match coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => {
                WorkingCoordinateDesign::OriginalSparseNative
            }
            PirlsCoordinateFrame::TransformedQs => {
                if let Some(x_transformed) = x_transformed {
                    WorkingCoordinateDesign::TransformedExplicit {
                        x_csr: x_transformed.to_csr_cache(),
                        x_transformed,
                    }
                } else {
                    WorkingCoordinateDesign::TransformedImplicit {
                        transform: transform.expect(
                            "TransformedQs PIRLS coordinate frame requires either x_transformed or qs",
                        ),
                    }
                }
            }
        };
        let x_original_csr = x_original.to_csr_cache();
        let n = match &coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => x_original.nrows(),
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.nrows()
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => x_original.nrows(),
        };
        GamWorkingModel {
            x_original,
            coordinate_design,
            offset: offset.to_owned(),
            y,
            priorweights,
            penalty,
            workspace,
            likelihood,
            link_kind,
            firth_bias_reduction,
            lastmu: Array1::zeros(n),
            lastweights: Array1::zeros(n),
            lastz: Array1::zeros(n),
            last_c: Array1::zeros(n),
            last_d: Array1::zeros(n),
            lasthessian_weights: Array1::zeros(n),
            lasthessian_c: Array1::zeros(n),
            lasthessian_d: Array1::zeros(n),
            lasthessian_curvature: HessianCurvatureKind::Fisher,
            last_dmu_deta: Array1::zeros(n),
            last_d2mu_deta2: Array1::zeros(n),
            last_d3mu_deta3: Array1::zeros(n),
            last_penalty_term: 0.0,
            x_original_csr,
            covariate_se: None,
            gamma_shape_locked: false,
            beta_phi_locked: false,
            tweedie_phi_locked: false,
            negbin_theta_locked: false,
            quadctx,
            glm_first_step_gram,
            glm_first_step_gram_consumed: false,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the working model uses uncertainty-aware IRLS updates.
    fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
    }

    /// Convert the working model into its final state for outer REML consumption.
    ///
    /// The `finalweights` field is set to `lasthessian_weights`, which are the
    /// **observed-information** weights (for non-canonical links) or Fisher weights
    /// (for canonical links where observed = Fisher). These flow into the outer
    /// REML H = X'W_obs X + S, ensuring log|H| uses the correct Laplace curvature.
    /// See response.md Section 3 for the mathematical justification.
    fn into_final_state(self) -> GamModelFinalState {
        let GamWorkingModel {
            coordinate_design,
            lastmu,
            lastweights,
            lastz,
            last_c: _,
            last_d: _,
            lasthessian_weights,
            lasthessian_c,
            lasthessian_d,
            last_dmu_deta,
            last_d2mu_deta2,
            last_d3mu_deta3,
            last_penalty_term,
            ..
        } = self;
        let coordinate_frame = match coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                PirlsCoordinateFrame::OriginalSparseNative
            }
            WorkingCoordinateDesign::TransformedExplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
        };
        GamModelFinalState {
            likelihood: self.likelihood.clone(),
            coordinate_frame,
            finalmu: lastmu,
            finalweights: lasthessian_weights,
            scoreweights: lastweights,
            finalz: lastz,
            final_c: lasthessian_c,
            final_d: lasthessian_d,
            final_dmu_deta: last_dmu_deta,
            final_d2mu_deta2: last_d2mu_deta2,
            final_d3mu_deta3: last_d3mu_deta3,
            penalty_term: last_penalty_term,
        }
    }

    /// Compute X_transformed * β into a pre-allocated buffer, avoiding
    /// per-iteration allocation in the dense case.
    fn transformed_matvec_into(&self, beta: &Coefficients, out: &mut Array1<f64>) {
        self.transformed_matvec_array_into(beta.as_ref(), out);
    }

    /// View-based sibling of `transformed_matvec_into` that operates on a raw
    /// `&Array1<f64>` to avoid wrapping (and cloning into) `Coefficients` on
    /// hot LM-screen paths.
    fn transformed_matvec_array_into(&self, beta: &Array1<f64>, out: &mut Array1<f64>) {
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                if let Some(dense) = x_transformed.as_dense() {
                    fast_av_into(dense, beta, out);
                    return;
                }
                out.assign(&x_transformed.matrixvectormultiply(beta));
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                // Composed: X · (Qs · beta).  Qs·beta is p-dim (cheap),
                // then write X·(Qs·beta) directly into out when X is dense.
                let beta_orig = transform.apply(beta);
                if let Some(dense) = self.x_original.as_dense() {
                    fast_av_into(dense, &beta_orig, out);
                } else {
                    out.assign(&self.x_original.apply(&beta_orig));
                }
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                out.assign(&self.x_original.matrixvectormultiply(beta));
            }
        }
    }

    fn transformed_transpose_matvec(&self, vec: &Array1<f64>) -> Array1<f64> {
        match &self.coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                self.x_original.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtv = self.x_original.transpose_vector_multiply(vec);
                transform.apply_transpose(&xtv)
            }
        }
    }

    /// Compute X^T W X via the shared dense assembly path.
    /// Falls back to the scalar loop for sparse matrices.
    fn compute_xtwx_blas(
        workspace: &mut PirlsWorkspace,
        design: &DesignMatrix,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        match design {
            // Only the materialized arm can use the shared dense assembly path.
            // Lazy operator-backed dense designs (TPS/Matern at large scale)
            // cannot be densified; fall through to the operator XᵀWX path.
            DesignMatrix::Dense(x) if x.is_materialized_dense() => {
                let p = x.ncols();
                let x_dense = x.to_dense_arc();
                // Reuse workspace hessian buffer to avoid per-iteration allocation.
                if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                    workspace.hessian_buf = Array2::zeros((p, p).f());
                } else {
                    workspace.hessian_buf.fill(0.0);
                }
                if crate::gpu::cuda_selected() {
                    return crate::solver::gpu::pirls_gpu::weighted_crossprod_gpu(
                        x_dense.view(),
                        weights.view(),
                    )
                    .map_err(EstimationError::InvalidInput);
                }
                crate::gpu::log_backend_inventory_once();
                // DenseXtWX has no compiled vendor backend on this path; the
                // workload-size predicate is computed only for diagnostic
                // logging via the `decide` reason channel.
                let gpu_decision = crate::gpu::decide(
                    crate::gpu::GpuKernel::DenseXtWX,
                    crate::gpu::GpuEligibility::BackendNotCompiled,
                );
                gpu_decision
                    .require_supported()
                    .map_err(EstimationError::InvalidInput)?;
                gpu_decision.log();
                if weights.iter().any(|&w| w < 0.0) {
                    // Observed-information assembly may have signed row
                    // weights.  Use Xᵀ(WX) exactly; never sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                } else {
                    // All weights are non-negative; the shared dense helper
                    // computes Xᵀ·diag(w)·X directly without sqrt/clip.
                    PirlsWorkspace::add_dense_xtwx_signed(
                        weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                    );
                }
                // Move the buffer out instead of cloning — saves O(p²) memcpy.
                // Next call will reallocate (same cost as the existing zero-fill).
                Ok(std::mem::take(&mut workspace.hessian_buf))
            }
            // Observed-Hessian assembly: working weights may be signed
            // (binomial + cloglog, Gamma + identity, etc.). Route through the
            // signed-Gram API so the CSC / sparse-accumulator paths preserve
            // sign instead of silently clipping negative-curvature mass.
            _ => crate::matrix::xt_diag_x_signed(
                design,
                crate::matrix::SignedWeightsView::from_array(weights),
            )
            .map(|h| h.to_dense())
            .map_err(EstimationError::InvalidInput),
        }
    }

    fn penalized_hessian(&mut self, weights: &Array1<f64>) -> Result<Array2<f64>, EstimationError> {
        // #1111 / #1033 mechanism (c): the frozen-weight first-Fisher-step Gram
        // `XᵀWX` (in the original / `x_fit` conditioned frame) serves the FIRST
        // Fisher-scoring iteration n-free, eliding the dominant O(N·p²) weighted
        // cross-product on a large-n GLM ψ-trial. It is only correct for the
        // first build at the warm β with FISHER curvature (the frozen tensor was
        // assembled from the canonical Fisher weights), and only in the two
        // original-frame coordinate designs (TransformedImplicit conjugates the
        // original-frame Gram afterward; OriginalSparseNative is already in that
        // frame). For TransformedExplicit the streamed Gram lives in the Qs frame
        // the tensor was not built in, so that variant always restreams. Every
        // later iteration restreams the true (moving) `W`, so the converged β̂ is
        // unchanged — only the first Gram build is skipped.
        let use_frozen_first_step = !self.glm_first_step_gram_consumed
            && self.glm_first_step_gram.is_some()
            && self.lasthessian_curvature == HessianCurvatureKind::Fisher
            && !matches!(
                self.coordinate_design,
                WorkingCoordinateDesign::TransformedExplicit { .. }
            );
        if use_frozen_first_step {
            // Take the cached original-frame Gram exactly once.
            let xtwx = self
                .glm_first_step_gram
                .take()
                .expect("frozen first-step Gram present by the guard above");
            self.glm_first_step_gram_consumed = true;
            log::debug!(
                "[frozen-glm-gram] serving first Fisher-step XᵀWX n-free (p={})",
                xtwx.nrows()
            );
            return match &self.coordinate_design {
                WorkingCoordinateDesign::TransformedImplicit { transform } => {
                    let mut h = transform.conjugate_matrix(&xtwx);
                    self.penalty.add_to_hessian(&mut h);
                    Ok(h)
                }
                WorkingCoordinateDesign::OriginalSparseNative => {
                    let mut h = xtwx;
                    self.penalty.add_to_hessian(&mut h);
                    Ok(h)
                }
                WorkingCoordinateDesign::TransformedExplicit { .. } => {
                    // Excluded from `use_frozen_first_step` by the guard above
                    // (the frozen Gram lives in the original frame the explicit
                    // transform was not built in). A clean error rather than a
                    // panic if a future refactor ever lets this state through.
                    Err(EstimationError::InvalidInput(
                        "frozen first-step Gram path reached with TransformedExplicit \
                         coordinate design, which the gate excludes"
                            .to_string(),
                    ))
                }
            };
        }
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                let mut h = Self::compute_xtwx_blas(&mut self.workspace, x_transformed, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtwx = Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                let mut h = transform.conjugate_matrix(&xtwx);
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                let mut h =
                    Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
        }
    }

    fn supports_observed_hessian_curvature(&self) -> bool {
        supports_observed_hessian_curvature_for_likelihood(&self.likelihood, &self.link_kind)
    }

    /// Compute the Hessian-side weight arrays (w, c, d) for the requested curvature kind.
    ///
    /// When `requested == Observed` and the link supports it, returns the
    /// **observed-information** weights including the residual-dependent correction:
    ///   W_obs = W_Fisher - (y - mu) * B,  B = (h'' V - h'^2 V') / (phi V^2)
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// For canonical links (for example logit-Binomial and log-Poisson), B = 0
    /// so observed = Fisher. Gamma-log is non-canonical and therefore needs its
    /// own observed-information correction.
    ///
    /// These arrays serve dual purpose:
    /// 1. **Inner iteration**: They define the Newton system H*delta = -g.
    ///    Fisher scoring (using W_Fisher) is also valid here since any convergent
    ///    algorithm finds the same mode.
    /// 2. **Outer REML**: They define the Laplace Hessian H_obs = X'W_obs X + S.
    ///    The outer log|H| and trace terms MUST use observed information for the
    ///    exact Laplace approximation. See response.md Section 3.
    fn update_hessian_curvature_arrays(
        &mut self,
        requested: HessianCurvatureKind,
    ) -> Result<HessianCurvatureKind, EstimationError> {
        if requested == HessianCurvatureKind::Fisher || !self.supports_observed_hessian_curvature()
        {
            self.lasthessian_weights.assign(&self.lastweights);
            self.lasthessian_c.assign(&self.last_c);
            self.lasthessian_d.assign(&self.last_d);
            return Ok(HessianCurvatureKind::Fisher);
        }

        compute_observed_hessian_curvature_arrays_into(
            &self.likelihood,
            &self.link_kind,
            &self.workspace.eta_buf,
            self.y,
            &self.lastweights,
            self.priorweights,
            &mut self.lasthessian_weights,
            &mut self.lasthessian_c,
            &mut self.lasthessian_d,
        )?;
        Ok(HessianCurvatureKind::Observed)
    }

    fn sparse_penalized_hessian(
        &mut self,
        weights: &Array1<f64>,
        ridge: f64,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        let x_sparse = self.x_original.as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse-native PIRLS requires a sparse original design".to_string(),
            )
        })?;
        let PirlsPenalty::Dense { s_transformed, .. } = &self.penalty else {
            crate::bail_invalid_estim!(
                "sparse-native PIRLS requires a dense transformed penalty matrix"
            );
        };
        self.workspace.assemble_sparse_penalized_hessian(
            x_sparse,
            weights,
            s_transformed,
            ridge,
            None,
        )
    }

    /// LM-screen helper: evaluates a candidate β by reusing the previous
    /// `current_eta` plus a single design-matrix matvec `X·δ`, then runs the
    /// inverse-link only far enough to recover μ, w, z and the deviance.
    /// No Hessian assembly, no derivative buffers, no Jeffreys logdet.
    ///
    /// The LM loop calls `update_with_curvature` to upgrade the screen to a
    /// full `WorkingState` only when the screen is accepted. Rejected LM
    /// candidates therefore skip the O(np²) curvature build entirely.
    fn screen_candidate_from_direction(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
    ) -> Result<CandidateScreen, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.delta_eta.len() != n {
            self.workspace.delta_eta = Array1::zeros(n);
        }

        // Compute δη = X·direction once into the workspace, then assemble
        // η_cand = η_current + δη in parallel.
        let mut delta_eta = std::mem::take(&mut self.workspace.delta_eta);
        // Avoid wrapping/cloning `direction` into a `Coefficients` newtype just
        // to satisfy the &Coefficients overload — the view-based sibling
        // performs the identical matvec without the per-LM-attempt clone.
        self.transformed_matvec_array_into(direction, &mut delta_eta);
        Zip::from(&mut self.workspace.eta_buf)
            .and(current_eta.as_ref())
            .and(&delta_eta)
            .par_for_each(|eta, &base, &d| *eta = base + d);
        self.workspace.delta_eta = delta_eta;

        // NB: the Gamma dispersion shape is deliberately NOT re-estimated here.
        // This screen only evaluates a *trial* β to feed the LM gain-ratio
        // accept/reject test, whose predicted reduction comes from the gradient
        // and Hessian built (at the current shape) by the last accepted
        // `update_with_curvature`. Re-estimating the shape per trial — and per
        // halving attempt — silently changes the objective the screen reports
        // (deviance = 2·shape·Σ wᵢ dᵢ) relative to that predicted reduction, so
        // the gain ratio compares two different objectives, every step is
        // rejected, λ_LM runs to its ceiling, and the inner solve stalls with a
        // large residual gradient ("LM step search exhausted"). The shape is a
        // nuisance scale that must stay fixed within an inner Newton/LM step; it
        // is updated once per *accepted* iterate in `update_with_curvature`
        // (block-coordinate β | shape), exactly as mgcv holds the scale fixed
        // through the inner P-IRLS solve. See issue #511 (regression of #359).
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        None,
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        None,
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    None,
                )?;
            }
        }

        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &self.lastmu, self.priorweights)?;
        let penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let penalized_objective = deviance + penalty_term;
        let arithmetic_finite = penalized_objective.is_finite()
            && self.workspace.eta_buf.iter().all(|v| v.is_finite())
            && self.lastmu.iter().all(|v| v.is_finite())
            && self.lastweights.iter().all(|v| v.is_finite());
        Ok(CandidateScreen {
            penalized_objective,
            deviance,
            penalty_term,
            arithmetic_finite,
        })
    }
}


impl<'a> WorkingModel for GamWorkingModel<'a> {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
    }

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        requested_curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        let mut matvec_tmp = std::mem::take(&mut self.workspace.matvec_buf);
        self.transformed_matvec_into(beta, &mut matvec_tmp);
        self.workspace.eta_buf.assign(&self.offset);
        self.workspace.eta_buf += &matvec_tmp;
        self.workspace.matvec_buf = matvec_tmp;

        // Estimate the Gamma dispersion shape once from the warm-start η and
        // freeze it for the remainder of this inner solve. Holding the shape
        // fixed keeps the product φ·λ constant, so the penalized argmin β̂ is a
        // stationary target and the LM gain ratio stays consistent across trial
        // and accepted iterates. The shape refreshes across outer iterations
        // because a fresh model is built per inner solve. See issue #511.
        if self.likelihood.scale.gamma_shape_is_estimated() && !self.gamma_shape_locked {
            let shape =
                estimate_gamma_shape_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.clone().with_gamma_shape(shape);
            self.gamma_shape_locked = true;
        }

        // Estimate the Beta precision φ once from the warm-start η and freeze it
        // for this inner solve (issue #567). φ enters the IRLS weights and the
        // variance `Var(y)=mu(1-mu)/(1+φ)`; holding it fixed within the inner
        // solve keeps the penalized argmin β̂ stationary (mirroring the Gamma
        // shape lock above), and it refreshes across outer iterations as a fresh
        // working model is built per inner solve. With φ pinned at the seed of 1
        // the mean smooth was over-penalized / under-fit on precise data.
        if self.likelihood.scale.beta_phi_is_estimated() && !self.beta_phi_locked {
            let phi =
                estimate_beta_phi_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.clone().with_beta_phi(phi);
            self.beta_phi_locked = true;
        }

        // Estimate the Tweedie dispersion φ once from the warm-start η and freeze
        // it for this inner solve (issue #771). φ enters the IRLS weight
        // `prior·μ^{2−p}/φ` (and so the covariance Vb = H⁻¹, giving SE ∝ √φ);
        // holding it fixed within the inner solve keeps the product φ·λ — hence
        // the penalized argmin β̂ — a stationary LM target (mirroring the Gamma
        // shape and Beta φ locks above), and it refreshes across outer iterations
        // as a fresh working model is built per inner solve.
        if self.likelihood.scale.tweedie_phi_is_estimated() && !self.tweedie_phi_locked {
            if let ResponseFamily::Tweedie { p } = self.likelihood.spec.response {
                let phi = estimate_tweedie_phi_from_eta(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    p,
                );
                self.likelihood = self.likelihood.clone().with_tweedie_phi(phi);
                self.tweedie_phi_locked = true;
            }
        }

        // Estimate the Negative-Binomial overdispersion `theta` once from the
        // warm-start η and freeze it for this inner solve (issue #802). `theta`
        // enters the working weight `W = μθ/(θ+μ)` (the NB2 Fisher information)
        // and the working response, so holding it fixed within the inner solve
        // keeps the penalized argmin β̂ a stationary LM target (mirroring the Beta
        // φ lock above); it refreshes across outer iterations as a fresh working
        // model is built per inner solve. With `theta` frozen at the seed every
        // coefficient/η SE ignored the data's overdispersion.
        if self.likelihood.scale.negbin_theta_is_estimated() && !self.negbin_theta_locked {
            let theta =
                estimate_negbin_theta_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.clone().with_negbin_theta(theta);
            self.negbin_theta_locked = true;
        }

        // Use integrated (GHQ) likelihood if per-observation SE is available.
        // This coherently accounts for uncertainty in the base prediction.
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::LatentCLogLog(_) | InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    Some(WorkingDerivativeBuffersMut {
                        c: &mut self.last_c,
                        d: &mut self.last_d,
                        dmu_deta: &mut self.last_dmu_deta,
                        d2mu_deta2: &mut self.last_d2mu_deta2,
                        d3mu_deta3: &mut self.last_d3mu_deta3,
                    }),
                )?;
            }
        }
        let mut firth = FirthDiagnostics::Inactive;
        if self.firth_bias_reduction {
            if !inverse_link_has_fisher_weight_jet(&self.link_kind) {
                crate::bail_invalid_estim!(
                    "Firth/Jeffreys PIRLS requested for unsupported inverse link {:?}",
                    self.link_kind
                );
            }
            // IMPORTANT: Jeffreys/Firth bias reduction must be computed in the
            // *same coefficient basis* as the inner objective being optimized by PIRLS.
            //
            // The working response (z) and the coefficients β are in the transformed
            // basis when a reparameterization is used. The Jeffreys term is the
            // identifiable-subspace Fisher logdet evaluated on a canonical
            // orthonormal basis of the transformed design column space,
            // not a raw-coordinate logdet. Its PIRLS hat-diagonal adjustment must
            // therefore be computed from that same transformed-design Fisher
            // matrix, otherwise the inner objective and the outer LAML
            // derivatives disagree.
            //
            // This mismatch is subtle but severe: it leaves the analytic gradient
            // differentiating a *different* objective than the one PIRLS actually
            // solved, and the gradient check fails catastrophically.
            //
            // Rule: use X_transformed if available; fall back to X_original only
            // when PIRLS is operating directly in the original basis.
            let (hat_diag, jeffreys_logdet, firth_score_shift) = match &self.coordinate_design {
                WorkingCoordinateDesign::TransformedExplicit {
                    x_transformed,
                    x_csr,
                } => {
                    if x_transformed.as_sparse().is_some() {
                        let csr = x_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "missing CSR cache for sparse transformed design".to_string(),
                            )
                        })?;
                        compute_jeffreys_pirls_diagnostics_sparse(
                            &self.link_kind,
                            csr,
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    } else {
                        let x_dense_cow = x_transformed.to_dense_cow();
                        compute_jeffreys_pirls_diagnostics(
                            &self.link_kind,
                            x_dense_cow.view(),
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    }
                }
                WorkingCoordinateDesign::TransformedImplicit { transform } => {
                    // Jeffreys/Firth MUST use a consistent basis. TransformedImplicit
                    // stores s_transformed in the Qs basis, so we need X in that
                    // same basis.  Materialize X·Qs on demand (Firth models are
                    // typically small clinical logistic regressions).
                    let x_t_dense =
                        fast_ab(&self.x_original.to_dense(), &transform.materialize_dense());
                    compute_jeffreys_pirls_diagnostics(
                        &self.link_kind,
                        x_t_dense.view(),
                        self.workspace.eta_buf.view(),
                        self.priorweights,
                    )?
                }
                WorkingCoordinateDesign::OriginalSparseNative => {
                    // s_transformed is in original coords here (qs = I).
                    if self.x_original.as_sparse().is_some() {
                        let csr = self.x_original_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "missing CSR cache for sparse original design".to_string(),
                            )
                        })?;
                        compute_jeffreys_pirls_diagnostics_sparse(
                            &self.link_kind,
                            csr,
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    } else {
                        let x_dense = self
                            .x_original
                            .try_to_dense_arc(
                                "Firth diagnostics require dense access to the original design",
                            )
                            .map_err(EstimationError::InvalidInput)?;
                        compute_jeffreys_pirls_diagnostics(
                            &self.link_kind,
                            x_dense.view(),
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    }
                }
            };
            firth = FirthDiagnostics::Active {
                jeffreys_logdet,
                hat_diag: hat_diag.clone(),
            };
            // Apply the link-general Firth working-response shift `Δ_i` built by
            // the operator (`½ (w'_i/w_i) h_diag_i`). PIRLS then solves
            // `Xᵀ W (z* − η) = 0`, so the Firth term it adds to the score is
            // `Σ_i w_i Δ_i x_i = ½ Σ_i w'_i h_diag_i x_i = ∂Φ/∂β` — exactly the
            // Jeffreys score the outer REML differentiates. For the canonical
            // logit `Δ_i` equals the historical `h_i (½ − μ_i)/w_i`; for probit /
            // cloglog it carries the correct non-canonical `w'_i/w_i` instead of
            // the logit-pinned `(½ − μ_i)`, so the inner mode and the outer
            // objective no longer disagree.
            ndarray::Zip::from(&mut self.lastz)
                .and(&firth_score_shift)
                .and(&self.lastweights)
                .par_for_each(|zi, &delta_i, &wi| {
                    if wi > 0.0 {
                        *zi += delta_i;
                    }
                });
        }

        let z = &self.lastz;
        // Fused single-pass: compute weighted_residual = (eta - z) * w
        // and working_residual = eta - z simultaneously, avoiding two
        // separate O(n) passes and an intermediate copy.
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&mut self.workspace.working_residual)
            .and(&self.workspace.eta_buf)
            .and(z)
            .and(&self.lastweights)
            .par_for_each(|wr, r, &eta, &zi, &wi| {
                let residual = eta - zi;
                *r = residual;
                *wr = residual * wi;
            });
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        // Score norm ||X' (weighted residual)||_2 — captured before adding the
        // penalty contribution so the natural gradient scale can be assembled
        // for the scale-invariant convergence certificate.
        let score_norm = array1_l2_norm(&gradient);
        let s_beta = self.penalty.shifted_gradient(beta.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        gradient += &s_beta;
        let hessian_curvature = self.update_hessian_curvature_arrays(requested_curvature)?;
        self.lasthessian_curvature = hessian_curvature;

        // Build solver-side weights in the reusable n-buffer: apply a
        // per-observation SPD floor so the Newton linear system is
        // well-conditioned, without contaminating the model weights stored in
        // `lasthessian_weights`.
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        solver_hessian_weights_into(
            &self.lasthessian_weights,
            &self.lastweights,
            &mut self.workspace.matvec_buf,
        );
        let solver_weights = std::mem::take(&mut self.workspace.matvec_buf);

        let (penalized_hessian, sparsehessian, ridge_used) = if matches!(
            self.coordinate_design,
            WorkingCoordinateDesign::OriginalSparseNative
        ) {
            // The SPD-check factor is discarded here: the downstream consumer
            // is the LM Newton step, which always factorizes
            // (H + loop_lambda · I) with a non-zero loop_lambda (initial value
            // 1e-6), so it sees a different matrix.
            let (h_sparse, _factor, ridge_used) =
                ensure_sparse_positive_definitewithridge(|ridge| {
                    self.sparse_penalized_hessian(&solver_weights, ridge)
                })?;
            (Array2::zeros((0, 0)), Some(h_sparse), ridge_used)
        } else {
            let mut penalized_hessian = self.penalized_hessian(&solver_weights)?;
            assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", 1e-8);
            let ridge_used = ensure_positive_definitewithridge(
                &mut penalized_hessian,
                "PIRLS penalized Hessian",
            )?;
            (penalized_hessian, None, ridge_used)
        };
        self.workspace.matvec_buf = solver_weights;

        // Match the stabilized Hessian used by the outer LAML objective.
        // If a ridge is needed, we treat it as an explicit penalty term:
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β - 0.5 * ridge * ||β||²
        //
        // This keeps the PIRLS fixed point aligned with the stabilized Hessian
        // that drives log|H| and the implicit-gradient correction.
        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &self.lastmu, self.priorweights)?;
        let log_likelihood = calculate_loglikelihood_omitting_constants(
            self.y,
            &self.lastmu,
            &self.likelihood,
            self.priorweights,
        );

        let mut penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let mut ridge_grad_norm = 0.0;
        if ridge_used > 0.0 {
            let ridge_penalty = ridge_used * beta.as_ref().dot(beta.as_ref());
            penalty_term += ridge_penalty;
            gradient.zip_mut_with(beta.as_ref(), |g, &b| *g += ridge_used * b);
            ridge_grad_norm = ridge_used * array1_l2_norm(beta.as_ref());
        }

        self.last_penalty_term = penalty_term;
        let gradient_natural_scale = score_norm + s_beta_norm + ridge_grad_norm;

        Ok(WorkingState {
            eta: LinearPredictor::new(std::mem::replace(
                &mut self.workspace.eta_buf,
                Array1::zeros(0),
            )),
            gradient,
            hessian: match sparsehessian {
                Some(h_sparse) => crate::linalg::matrix::SymmetricMatrix::Sparse(h_sparse),
                None => crate::linalg::matrix::SymmetricMatrix::Dense(penalized_hessian),
            },

            log_likelihood,
            deviance,
            penalty_term,
            firth,
            ridge_used,
            hessian_curvature,
            gradient_natural_scale,
        })
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        if !self.firth_bias_reduction {
            return self.update_with_curvature(beta, curvature);
        }
        let firth_enabled = self.firth_bias_reduction;
        self.firth_bias_reduction = false;
        let result = self.update_with_curvature(beta, curvature);
        self.firth_bias_reduction = firth_enabled;
        result
    }

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        if self.firth_bias_reduction {
            return self
                .update_candidate(beta, curvature)
                .map(CandidateEvaluation::Full);
        }
        self.screen_candidate_from_direction(beta, direction, current_eta)
            .map(CandidateEvaluation::Screen)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        self.supports_observed_hessian_curvature()
    }
}


// Cutoff between the dense outer-product backend and sparse SpGEMM. At p=1024
// the dense p×p output buffer is 8 MiB — L3-resident on most current targets
// and small enough that per-thread copies used during parallel reduction stay
// within an order of magnitude of the cache hierarchy.
const DENSE_OUTER_MAX_P: usize = 1024;


// Estimated FLOP threshold below which spawning rayon workers for the dense
// outer-product path costs more than the work itself. Calibrated to cover
// rayon's per-task overhead (microseconds) plus the cost of zeroing one dense
// buffer per worker; below this, everything stays on the calling thread.
const DENSE_OUTER_PARALLEL_FLOP_THRESHOLD: u64 = 100_000;


/// Backend selection for sparse-design XᵀWX assembly.
///
/// XᵀWX = Σᵢ wᵢ · xᵢ xᵢᵀ. The matrix is symmetric, so only the upper triangle
/// needs to be computed; the only consumer (`assemble_upper`) filters to
/// row ≤ col. Two backends trade off in opposite memory regimes:
///
/// * **Dense outer-product** (small p): allocate a dense p×p buffer and
///   accumulate one rank-1 update per data row. Per-row work is nnz(xᵢ)² —
///   for B-spline-style designs this dominates SpGEMM by orders of magnitude.
///
/// * **Sparse SpGEMM** (large p): faer's symbolic + numeric pipeline. Avoids
///   the dense p×p buffer when it would no longer be cache-resident.
enum XtWxBackend {
    Dense(DenseOuterState),
    Sparse(SparseSpGemmState),
}


/// State for the dense outer-product backend.
///
/// `xtwx_dense` is row-major p×p; the inner loop fills only the upper triangle
/// (j ≤ k), exploiting faer's CSC convention that row indices within each
/// column are stored in ascending order. Lower-triangle entries are left at
/// zero — they are written through the scatter to `xtwxvalues` but never read,
/// because `assemble_upper` filters to row ≤ col.
///
/// `thread_buffers` is bounded at exactly `rayon::current_num_threads()` and
/// reused across PIRLS iterations, so allocation cost is amortized across the
/// entire fit rather than paid per call.
struct DenseOuterState {
    xtwx_dense: Array2<f64>,
    thread_buffers: Vec<Array2<f64>>,
}


/// State for the sparse-SpGEMM backend (faer numeric matmul scratch and the
/// pre-scaled (√W)·X factors that feed it).
///
/// `sqrt_weights` caches `√wᵢ` for each finite nonnegative PIRLS working
/// weight row of X. Without it, the right-factor loop would recompute the same
/// sqrt once per nonzero of X (each row weight gets read by every column that
/// has a nonzero in that row), so for an n=400 K · avg-nnz-per-row=10 design
/// that's 4 M sqrts per PIRLS iteration. Precomputing once collapses that to n
/// sqrts and the inner loop becomes a pure multiply.
///
/// This is deliberately separate from REML/Firth's fixed
/// `observation_weight_sqrt` handling in `solver/reml/firth.rs`: this cache
/// materializes the current working-weight Gram factors, while Firth stores
/// case-weight roots so reduced designs can later be mapped back with
/// reciprocal roots.
struct SparseSpGemmState {
    wxvalues: Vec<f64>,
    wx_tvalues: Vec<f64>,
    sqrt_weights: Vec<f64>,
    info: SparseMatMulInfo,
    scratch: MemBuffer,
    par: Par,
}


pub(crate) struct SparseXtWxCache {
    xtwx_symbolic: SymbolicSparseColMat<usize>,
    xtwxvalues: Vec<f64>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    x_col_ptr: Vec<usize>,
    xrow_idx: Vec<usize>,
    /// CSC of Xᵀ. In CSC, column i of Xᵀ stores the nonzeros of row i of X,
    /// so this doubles as a CSR view of X for row-by-row access in the
    /// dense-outer path.
    x_t_csc: SparseColMat<usize, f64>,
    backend: XtWxBackend,
}


impl SparseXtWxCache {
    fn new(x: &SparseColMat<usize, f64>) -> Result<Self, EstimationError> {
        // For X^T X where X is CSC: X^T is a SparseRowMat, which we need to
        // convert to CSC format for the matmul API.
        let x_t_csc =
            x.as_ref().transpose().to_col_major().map_err(|_| {
                EstimationError::InvalidInput("failed to transpose to CSC".to_string())
            })?;
        let (xtwx_symbolic, info) = sparse_sparse_matmul_symbolic(x_t_csc.symbolic(), x.symbolic())
            .map_err(|_| {
                EstimationError::InvalidInput("failed to build symbolic XtWX cache".to_string())
            })?;
        let xtwxvalues = vec![0.0; xtwx_symbolic.row_idx().len()];

        let backend = if x.ncols() <= DENSE_OUTER_MAX_P {
            XtWxBackend::Dense(DenseOuterState {
                xtwx_dense: Array2::<f64>::zeros((x.ncols(), x.ncols())),
                thread_buffers: Vec::new(),
            })
        } else {
            // SpGEMM scratch is sized for a fixed parallelism handle, so we
            // capture it once at construction; `get_global_parallelism()` is
            // stable for the lifetime of the process.
            let par = get_global_parallelism();
            let scratch = MemBuffer::new(sparse_sparse_matmul_numeric_scratch::<usize, f64>(
                xtwx_symbolic.as_ref(),
                par,
            ));
            XtWxBackend::Sparse(SparseSpGemmState {
                wxvalues: vec![0.0; x.val().len()],
                wx_tvalues: vec![0.0; x_t_csc.val().len()],
                sqrt_weights: vec![0.0; x.nrows()],
                info,
                scratch,
                par,
            })
        };

        Ok(Self {
            xtwx_symbolic,
            xtwxvalues,
            nrows: x.nrows(),
            ncols: x.ncols(),
            nnz: x.val().len(),
            x_col_ptr: x.symbolic().col_ptr().to_vec(),
            xrow_idx: x.symbolic().row_idx().to_vec(),
            x_t_csc,
            backend,
        })
    }

    fn matches(&self, x: &SparseColMat<usize, f64>) -> bool {
        if self.nrows != x.nrows() || self.ncols != x.ncols() || self.nnz != x.val().len() {
            return false;
        }
        let sym = x.symbolic();
        self.x_col_ptr.as_slice() == sym.col_ptr() && self.xrow_idx.as_slice() == sym.row_idx()
    }

    fn compute_numeric(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        if weights.len() != self.nrows {
            crate::bail_invalid_estim!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.nrows
            );
        }

        match &mut self.backend {
            XtWxBackend::Dense(state) => {
                state.compute(self.x_t_csc.as_ref(), weights, self.nrows, self.ncols);
                // Scatter the upper triangle of `xtwx_dense` into the
                // symbolic XᵀX pattern. The pattern stores both halves of
                // the symmetric product, but `assemble_upper` (the sole
                // consumer) reads only entries with row ≤ col, so writing
                // the lower half would be wasted work. The unwritten
                // lower-triangle entries of `xtwxvalues` start at zero
                // (from `vec![0.0; …]` at construction) and remain zero
                // throughout this cache's lifetime, since the dense outer
                // product never writes to lower-triangle positions either.
                let col_ptr = self.xtwx_symbolic.col_ptr();
                let row_idx = self.xtwx_symbolic.row_idx();
                let dense = &state.xtwx_dense;
                for col in 0..self.ncols {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        if row <= col {
                            self.xtwxvalues[idx] = dense[[row, col]];
                        }
                    }
                }
            }
            XtWxBackend::Sparse(state) => state.compute(
                x,
                self.x_t_csc.as_ref(),
                weights,
                self.ncols,
                self.xtwx_symbolic.as_ref(),
                &mut self.xtwxvalues,
            ),
        }

        Ok(())
    }
}


impl DenseOuterState {
    /// Compute the upper triangle of XᵀWX = Σᵢ wᵢ · xᵢ xᵢᵀ into
    /// `self.xtwx_dense`.
    ///
    /// Decides serial vs parallel from a cost model on total estimated FLOPs
    /// and the number of available rayon workers. In parallel mode each
    /// worker accumulates into a thread-local p×p buffer (allocated once and
    /// reused across calls); the workers are summed into `xtwx_dense` in
    /// place, preserving its allocation rather than replacing it with a
    /// freshly-allocated reduction result.
    fn compute(
        &mut self,
        x_t: SparseColMatRef<'_, usize, f64>,
        weights: &Array1<f64>,
        n: usize,
        p: usize,
    ) {
        assert_eq!(self.xtwx_dense.dim(), (p, p));
        self.xtwx_dense.fill(0.0);
        if n == 0 || p == 0 {
            return;
        }
        let xtwx_start = std::time::Instant::now();

        // Cost model: per-row outer-product is nnz(xᵢ)². With avg_nnz ≈
        // nnz_total / n, total work ≈ nnz_total² / n. For designs with
        // uniform row support (e.g. B-splines) this proxy is tight; for
        // mixed-support designs it is an order-of-magnitude estimate, which
        // is all we need to gate parallel spawn.
        let nnz_total = x_t.symbolic().row_idx().len() as u64;
        let work = nnz_total
            .saturating_mul(nnz_total)
            .checked_div(n as u64)
            .unwrap_or(u64::MAX);
        let n_threads = rayon::current_num_threads();
        let parallelize = n_threads > 1 && work >= DENSE_OUTER_PARALLEL_FLOP_THRESHOLD;

        if !parallelize {
            accumulate_outer_upper(&mut self.xtwx_dense, x_t, weights, 0..n);
            log::info!(
                "[STAGE] PIRLS dense XᵀWX assembly (serial) n={} p={} flops~{} elapsed={:.3}s",
                n,
                p,
                (n as u64).saturating_mul((p as u64).saturating_mul(p as u64)),
                xtwx_start.elapsed().as_secs_f64(),
            );
            return;
        }

        // Bounded thread allocation: exactly `n_threads` p×p buffers, one
        // per worker, reused across calls.
        if self.thread_buffers.len() != n_threads {
            self.thread_buffers
                .resize_with(n_threads, || Array2::<f64>::zeros((p, p)));
        }
        let chunk = n.div_ceil(n_threads);
        self.thread_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(t, buf)| {
                buf.fill(0.0);
                let start = t * chunk;
                let end = (start + chunk).min(n);
                if start < end {
                    accumulate_outer_upper(buf, x_t, weights, start..end);
                }
            });

        // Reduce per-thread buffers into the cached output. The += preserves
        // `xtwx_dense`'s storage; we never reallocate it.
        for buf in &self.thread_buffers {
            self.xtwx_dense += buf;
        }
        log::info!(
            "[STAGE] PIRLS dense XᵀWX assembly (parallel, threads={}) n={} p={} flops~{} elapsed={:.3}s",
            rayon::current_num_threads(),
            n,
            p,
            (n as u64).saturating_mul((p as u64).saturating_mul(p as u64)),
            xtwx_start.elapsed().as_secs_f64(),
        );
    }
}


impl SparseSpGemmState {
    /// Compute XᵀWX into the symbolic-pattern array `xtwxvalues` via faer's
    /// sparse-sparse matmul: XᵀWX = (√W·X)ᵀ · (√W·X).
    fn compute(
        &mut self,
        x: &SparseColMat<usize, f64>,
        x_t: SparseColMatRef<'_, usize, f64>,
        weights: &Array1<f64>,
        p: usize,
        xtwx_symbolic: SymbolicSparseColMatRef<'_, usize>,
        xtwxvalues: &mut [f64],
    ) {
        let n = x_t.ncols();
        assert_eq!(weights.len(), n);
        assert_eq!(self.sqrt_weights.len(), n);

        assert!(
            weights.iter().all(|&w| w.is_finite() && w >= 0.0),
            "SparseSpGemmState::compute requires finite nonnegative PIRLS weights"
        );
        // Cache √w once per row so the inner loops can multiply
        // without repeated sqrt calls. Single owning slice avoids ndarray
        // bounds checks in the hot loops below.
        let sqrt_w = self.sqrt_weights.as_mut_slice();
        for (dst, &w) in sqrt_w.iter_mut().zip(weights.iter()) {
            *dst = w.sqrt();
        }
        let sqrt_w: &[f64] = sqrt_w;

        let x_ref = x.as_ref();
        // Right factor: √W · X, stored in X's CSC sparsity pattern.
        for col in 0..p {
            let rows = x_ref.row_idx_of_col_raw(col);
            let xvals = x_ref.val_of_col(col);
            let range = x_ref.col_range(col);
            let dst = &mut self.wxvalues[range];
            for ((d, &s), row) in dst.iter_mut().zip(xvals.iter()).zip(rows.iter()) {
                *d = s * sqrt_w[row.unbound()];
            }
        }
        // Left factor: (√W · X)ᵀ in X^T's CSC sparsity pattern. X^T's columns
        // correspond to rows of X, so each column scales by √w_row — read
        // straight from the cached slice with no per-column sqrt.
        for col in 0..n {
            let w = sqrt_w[col];
            let xvals = x_t.val_of_col(col);
            let range = x_t.col_range(col);
            let dst = &mut self.wx_tvalues[range];
            for (d, &s) in dst.iter_mut().zip(xvals.iter()) {
                *d = s * w;
            }
        }

        let wx_ref = SparseColMatRef::new(x.symbolic(), &self.wxvalues[..]);
        let wx_t_ref = SparseColMatRef::new(x_t.symbolic(), &self.wx_tvalues[..]);
        let stack = MemStack::new(&mut self.scratch);
        let xtwxmut = SparseColMatMut::new(xtwx_symbolic, xtwxvalues);
        sparse_sparse_matmul_numeric(
            xtwxmut,
            Accum::Replace,
            wx_t_ref,
            wx_ref,
            1.0,
            &self.info,
            self.par,
            stack,
        );
    }
}


/// Accumulate the upper triangle of Σᵢ wᵢ · xᵢ xᵢᵀ over `rows` into `acc`.
///
/// `x_t` is Xᵀ in CSC: column i lists the nonzero columns of row i of X.
/// Faer's CSC convention stores these in ascending order, so iterating
/// `jj < kk` over per-row index pairs gives `j ≤ k` and only ever writes
/// to `acc[[j, k]]` with `j ≤ k` (the upper triangle, including the
/// diagonal at `jj == kk`).
///
/// Inner-loop layout: `acc` is row-major p×p, so row j lives in the
/// contiguous slice `acc_data[j·p .. (j+1)·p]`. We reborrow that slice once
/// per outer-product step — cheaper than ndarray's `row_mut(j).as_slice_mut()`
/// because it skips the per-call stride-validation and contiguity check.
#[inline]
fn accumulate_outer_upper(
    acc: &mut Array2<f64>,
    x_t: SparseColMatRef<'_, usize, f64>,
    weights: &Array1<f64>,
    rows: std::ops::Range<usize>,
) {
    assert_eq!(acc.nrows(), acc.ncols());
    let p = acc.ncols();
    let acc_data = acc
        .as_slice_mut()
        .expect("dense XᵀWX accumulator is row-major and contiguous");

    for i in rows {
        // Sparse PIRLS precompute deliberately clips to Fisher-style
        // nonnegative weights before the row outer product. The shared REML
        // dense helper preserves signed observed-Hessian weights exactly, so
        // routing this sparse path through it would change curvature semantics.
        let w_i = weights[i].max(0.0);
        if w_i == 0.0 {
            continue;
        }
        let cols = x_t.row_idx_of_col_raw(i);
        let vals = x_t.val_of_col(i);
        let nnz_i = cols.len();
        for jj in 0..nnz_i {
            let j = cols[jj].unbound();
            let wvj = w_i * vals[jj];
            let row = &mut acc_data[j * p..j * p + p];
            for kk in jj..nnz_i {
                let k = cols[kk].unbound();
                row[k] += wvj * vals[kk];
            }
        }
    }
}


pub(super) fn compute_jeffreys_pirls_diagnostics_sparse(
    link: &InverseLink,
    x_design_csr: &SparseRowMat<usize, f64>,
    eta: ArrayView1<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64, Array1<f64>), EstimationError> {
    let n = x_design_csr.nrows();
    let p = x_design_csr.ncols();
    let mut x_dense = Array2::<f64>::zeros((n, p));
    let xview = x_design_csr.as_ref();
    for i in 0..n {
        let vals = xview.val_of_row(i);
        let cols = xview.col_idx_of_row_raw(i);
        if cols.len() != vals.len() {
            crate::bail_invalid_estim!(
                "sparse row structure mismatch: column/value lengths differ"
            );
        }
        for (idx, &col) in cols.iter().enumerate() {
            x_dense[[i, col.unbound()]] = vals[idx];
        }
    }
    compute_jeffreys_pirls_diagnostics(link, x_dense.view(), eta, observation_weights)
}


pub(super) fn compute_jeffreys_pirls_diagnostics(
    link: &InverseLink,
    x_design: ArrayView2<f64>,
    eta: ArrayView1<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64, Array1<f64>), EstimationError> {
    // PIRLS must use the same identifiable-subspace Jeffreys functional as the
    // outer REML code:
    //   Φ(β) = 0.5 log|Xᵀ W(η) X|_+.
    // The operator below is the single source of truth for the Jeffreys scalar
    // value, the PIRLS hat-diagonal, AND the working-response score shift the
    // inner solve applies. The Fisher working weight `W(η)` is evaluated for the
    // resolved inverse link; `StandardLink::Logit` reproduces the released logit
    // diagnostics exactly while non-canonical links (probit, cloglog) get the
    // correct link-general shift instead of the logit-pinned `(½ − μ)` term.
    let op = FirthDenseOperator::build_with_observation_weights_for_link(
        link,
        &x_design.to_owned(),
        &eta.to_owned(),
        observation_weights,
    )?;
    Ok((
        op.pirls_hat_diag(),
        op.jeffreys_logdet(),
        op.pirls_firth_score_shift(),
    ))
}


fn ensure_positive_definitewithridge(
    hess: &mut Array2<f64>,
    label: &str,
) -> Result<f64, EstimationError> {
    let ridge = if FIXED_STABILIZATION_RIDGE > 0.0 {
        FIXED_STABILIZATION_RIDGE
    } else {
        0.0
    };

    if hess.cholesky(Side::Lower).is_ok() {
        return Ok(0.0);
    }

    if ridge > 0.0 {
        for i in 0..hess.nrows() {
            hess[[i, i]] += ridge;
        }

        if hess.cholesky(Side::Lower).is_ok() {
            log::debug!("{} stabilized with fixed ridge {:.1e}.", label, ridge);
            return Ok(ridge);
        }
    }

    if let Ok((evals, _)) = hess.eigh(Side::Lower) {
        let min_eig = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        return Err(EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: min_eig,
        });
    }
    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: f64::NEG_INFINITY,
    })
}


pub(super) fn solve_newton_direction_dense(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    solve_newton_direction_dense_with_factor(hessian, gradient, direction_out).map(|_| ())
}


pub(super) fn solve_direction_with_dense_factor(
    factor: &FaerSymmetricFactor,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) {
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }
    direction_out.assign(gradient);
    let mut rhsview = array1_to_col_matmut(direction_out);
    factor.solve_in_place(rhsview.as_mut());
    direction_out.mapv_inplace(|v| -v);
}


/// Fixes the audit-revised geodesic-acceleration note: expose the dense
/// factor so the optional second-order correction can reuse it instead of
/// refactorizing the same Hessian.
pub(super) fn solve_newton_direction_dense_with_factor(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<Option<FaerSymmetricFactor>, EstimationError> {
    let dense_solve_start = std::time::Instant::now();
    let p = hessian.nrows();
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    if crate::gpu::cuda_selected() {
        let rhs = Array2::from_shape_vec((p, 1), gradient.to_vec()).map_err(|e| {
            EstimationError::InvalidInput(format!("CUDA PIRLS RHS layout failed: {e}"))
        })?;
        let (solved, _) =
            crate::solver::gpu::pirls_gpu::cholesky_solve_gpu(hessian.view(), rhs.view())
                .map_err(EstimationError::InvalidInput)?;
        direction_out.assign(&solved.column(0));
        direction_out.mapv_inplace(|v| -v);
        if array_is_finite(direction_out) {
            log::info!(
                "[STAGE] PIRLS dense newton solve backend=CUDA p={} flops~{} elapsed={:.3}s route=\"cuSOLVER potrf/potrs\"",
                p,
                (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
                dense_solve_start.elapsed().as_secs_f64(),
            );
            return Ok(None);
        }
    }

    let cpu_route = String::from("CPU stable solver");

    let factor = StableSolver::new("pirls newton direction")
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    solve_direction_with_dense_factor(&factor, gradient, direction_out);

    // Validate: bare Cholesky on a near-singular H produces huge spurious
    // step magnitudes in the null direction. If `‖H·δ + g‖∞ / (1+‖g‖∞)` is
    // not small the H is rank-deficient (eigenvalue below floating-point
    // resolution); fall through to the rank-revealing pseudoinverse path
    // which projects rhs onto range(H) before inverting and zeroes the
    // null-direction component of δ. This is the same arithmetic the
    // outer IFT correction uses via penalty_subspace_trace.
    let validation_residual = {
        let h_delta = hessian.dot(direction_out);
        h_delta
            .iter()
            .zip(gradient.iter())
            .map(|(h, g)| (h + g).abs())
            .fold(0.0_f64, f64::max)
    };
    let g_inf = gradient.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let rel = validation_residual / (1.0 + g_inf);
    if !rel.is_finite() || rel > 1.0e-3 {
        // Construct rhs = -gradient (note the gradient is the un-negated
        // ∇f at β; the Newton equation is H·δ = -g) and reach for the
        // pseudoinverse path. `solve_with_pseudoinverse_fallback` handles
        // its own ridge retries and falls back to truncated-eigh
        // pseudoinverse if Cholesky residual is high.
        let rhs = gradient.mapv(|v| -v);
        if let Some(pseudo) = StableSolver::new("pirls newton direction (pseudoinverse fallback)")
            .solve_with_pseudoinverse_fallback(hessian, &rhs, 1.0e-10, 1.0e-3, 1.0e-10)
        {
            direction_out.assign(&pseudo);
            log::info!(
                "[STAGE] PIRLS dense newton solve backend=CPU p={} elapsed={:.3}s route=\"{} + pseudoinverse fallback (rel={:.3e} > 1e-3)\"",
                p,
                dense_solve_start.elapsed().as_secs_f64(),
                cpu_route,
                rel,
            );
            return Ok(Some(factor));
        }
    }
    if array_is_finite(direction_out) {
        log::info!(
            "[STAGE] PIRLS dense newton solve backend=CPU p={} flops~{} elapsed={:.3}s route=\"{}\"",
            p,
            (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
            dense_solve_start.elapsed().as_secs_f64(),
            cpu_route,
        );
        return Ok(Some(factor));
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed {
            context: "PIRLS dense newton solve exhausted",
        },
    ))
}


/// Solve the Newton direction implicitly via PCG against an operator-form
/// Hessian. Bypasses materialization of the `p × p` Hessian when at least one
/// penalty is operator-form and `p` is large enough that the implicit-matvec
/// cost amortizes against avoiding a dense Cholesky.
///
/// `apply_xtwx`: closure computing `(X^T W X) v`.
/// `xtwx_diag`: diagonal of `X^T W X`, used in the Jacobi preconditioner.
/// `dense_penalties`: pairs `(λ_k, S_k)` for penalties whose dense matrix is
/// the only available representation; their contribution to `H v` is computed
/// as `λ_k · S_k.dot(v)` and their diagonal contribution to the preconditioner
/// is `λ_k · diag(S_k)`.
/// `op_penalties`: pairs `(λ_k, op)` for penalties carrying a `PenaltyOp`
/// handle; their contribution to `H v` is `λ_k · op.matvec(v)` and their
/// diagonal is `λ_k · op.diag()`.
/// `ridge`: nonnegative ridge added to the Hessian diagonal for stabilization.
///
/// On success the negated solution `−H⁻¹ g` is written into `direction_out`,
/// matching the sign convention of `solve_newton_direction_dense`.
pub fn solve_newton_direction_implicit<F>(
    apply_xtwx: F,
    xtwx_diag: ArrayView1<'_, f64>,
    dense_penalties: &[(f64, &Array2<f64>)],
    op_penalties: &[(f64, &dyn crate::terms::penalty_op::PenaltyOp)],
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
    ridge: f64,
    rel_tol: f64,
    max_iter: usize,
) -> Result<(), EstimationError>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = gradient.len();
    if xtwx_diag.len() != p {
        crate::bail_invalid_estim!(
            "solve_newton_direction_implicit: xtwx_diag length {} != gradient length {}",
            xtwx_diag.len(),
            p
        );
    }
    for (_, s) in dense_penalties.iter() {
        if s.nrows() != p || s.ncols() != p {
            crate::bail_invalid_estim!(
                "solve_newton_direction_implicit: dense penalty dim {}×{} != p={}",
                s.nrows(),
                s.ncols(),
                p
            );
        }
    }
    for (_, op) in op_penalties.iter() {
        if op.dim() != p {
            crate::bail_invalid_estim!(
                "solve_newton_direction_implicit: op penalty dim {} != p={}",
                op.dim(),
                p
            );
        }
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }

    let pcg_start = std::time::Instant::now();

    let mut precond_diag = xtwx_diag.to_owned();
    if ridge > 0.0 {
        precond_diag.mapv_inplace(|d| d + ridge);
    }
    for (lambda, s) in dense_penalties.iter() {
        if *lambda == 0.0 {
            continue;
        }
        for i in 0..p {
            precond_diag[i] += *lambda * s[[i, i]];
        }
    }
    for (lambda, op) in op_penalties.iter() {
        if *lambda == 0.0 {
            continue;
        }
        let d = op.diag();
        for i in 0..p {
            precond_diag[i] += *lambda * d[i];
        }
    }

    // SAFETY: `apply_xtwx`, `dense_penalties`, and `op_penalties` are passed
    // by reference into the closure. The PCG closure runs synchronously within
    // this function, so the borrows live for the duration of the call.
    let apply_h = |v: &Array1<f64>| -> Array1<f64> {
        let mut hv = apply_xtwx(v);
        if ridge > 0.0 {
            hv.zip_mut_with(v, |h, &x| *h += ridge * x);
        }
        for (lambda, s) in dense_penalties.iter() {
            if *lambda == 0.0 {
                continue;
            }
            let sv = fast_av(s, v);
            hv.scaled_add(*lambda, &sv);
        }
        for (lambda, op) in op_penalties.iter() {
            if *lambda == 0.0 {
                continue;
            }
            let mut sv = Array1::<f64>::zeros(p);
            op.matvec(v.view(), sv.view_mut());
            hv.scaled_add(*lambda, &sv);
        }
        hv
    };

    let solution =
        crate::linalg::utils::solve_spd_pcg(apply_h, gradient, &precond_diag, rel_tol, max_iter)
            .ok_or(EstimationError::LinearSystemSolveFailed(
                FaerLinalgError::FactorizationFailed {
                    context: "PIRLS implicit PCG solve exhausted",
                },
            ))?;

    direction_out.assign(&solution);
    direction_out.mapv_inplace(|v| -v);
    if !array_is_finite(direction_out) {
        return Err(EstimationError::LinearSystemSolveFailed(
            FaerLinalgError::FactorizationFailed {
                context: "PIRLS implicit PCG non-finite direction",
            },
        ));
    }
    log::info!(
        "[STAGE] PIRLS implicit (PCG) newton solve p={} dense_pens={} op_pens={} elapsed={:.3}s",
        p,
        dense_penalties.len(),
        op_penalties.len(),
        pcg_start.elapsed().as_secs_f64(),
    );
    Ok(())
}


pub(super) fn project_coefficients_to_lower_bounds(
    beta: &mut Array1<f64>,
    lower_bounds: &Array1<f64>,
) {
    for i in 0..beta.len() {
        let lb = lower_bounds[i];
        if lb.is_finite() && beta[i] < lb {
            beta[i] = lb;
        }
    }
}


/// Compute the projected gradient norm for bound-constrained optimization.
///
/// At a constrained optimum, gradient components for variables at their lower
/// bound that point into the infeasible direction (gradient > 0 for minimization)
/// are KKT multipliers, not convergence defects.  Zeroing them gives the
/// standard "projected gradient" used to test stationarity.
/// Relative and absolute tolerances for deciding when a coefficient sits "at"
/// its lower bound (an active box constraint). A coefficient is active when its
/// slack is below `ACTIVE_BOUND_REL_TOL * scale + ACTIVE_BOUND_ABS_TOL`; the
/// absolute term keeps genuinely-near-zero bounded coefficients (e.g. I-spline
/// time coefficients pinned around 1e-6) from being treated as interior. Both
/// the projected-gradient norm and the active-set classifier must use the same
/// band so KKT diagnostics and the working set agree.
const ACTIVE_BOUND_REL_TOL: f64 = 1e-6;

const ACTIVE_BOUND_ABS_TOL: f64 = 1e-10;


pub(super) fn projected_gradient_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
) -> f64 {
    let Some(lb) = lower_bounds else {
        return gradient.dot(gradient).sqrt();
    };
    let mut sum_sq = 0.0;
    for i in 0..gradient.len() {
        let g = gradient[i];
        if lb[i].is_finite() && g > 0.0 {
            // Use a relative+absolute tolerance so near-bound coefficients
            // (e.g. I-spline time coefficients at 1e-6) are recognized as
            // active.  At a KKT point the gradient into the infeasible region
            // is a multiplier, not a convergence defect.
            let slack = beta[i] - lb[i];
            let scale = beta[i].abs().max(lb[i].abs()).max(1.0);
            let tol = ACTIVE_BOUND_REL_TOL * scale + ACTIVE_BOUND_ABS_TOL;
            if slack < tol {
                continue;
            }
        }
        sum_sq += g * g;
    }
    sum_sq.sqrt()
}


/// "Soft" P-IRLS acceptance reasons — fits that did not certify strict KKT
/// stationarity but that the post-loop rescue would still classify as
/// `StalledAtValidMinimum`. Evaluating them per-iter (gated by a streak)
/// lets the loop exit at the iteration that first meets the criterion
/// instead of grinding to `MaxIterations` only to be rescued with the
/// same conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PirlsSoftAccept {
    /// Projected gradient inside the 10× near-stationary band AND the
    /// progress signal has plateaued at `tol · objective_scale` (or, in
    /// the LM-rejection context, at the much tighter `1e-12 · |Φ|` model
    /// noise floor — see [`SoftAcceptProgress`]). The standard
    /// "good-enough plateau" rescue, and the only branch that fires
    /// when no LM step was accepted.
    NearStationaryPlateau,
    /// `max|η|` is pinned against [`PIRLS_ETA_ABS_CAP`] AND the deviance
    /// has plateaued. Same saturated-boundary class as separated binomial
    /// fits: extra Newton work only re-tries the clipped boundary. Only
    /// meaningful when a step was actually taken — the LM-rejection
    /// context skips this branch.
    BoundarySaturation,
    /// Projected gradient is small *relative to the objective magnitude*
    /// (not just the dimension scale) AND the deviance has plateaued
    /// strictly (×0.1 floor) AND is non-decreasing. This is the
    /// per-observation rescue for large-scale GLMs where ‖g‖ scales
    /// with √n and the absolute KKT test becomes systematically too
    /// tight even when the fit is functionally converged. Like
    /// [`PirlsSoftAccept::BoundarySaturation`], this is only meaningful
    /// when a step was actually taken.
    RelativeBandPlateau,
}


/// Source of the "is the fit still moving?" signal handed to
/// [`pirls_soft_acceptance`]. There are two contexts in which we need to
/// decide whether a fit should be accepted as a soft minimum:
///
/// - [`SoftAcceptProgress::Realized`] — a step was accepted (per-iter
///   path) or the loop has run out of iterations (post-loop rescue). We
///   know the realized change in penalized deviance and can compare it
///   directly against the standard `tol · objective_scale` plateau band.
///   All three [`PirlsSoftAccept`] branches are eligible.
///
/// - [`SoftAcceptProgress::Predicted`] — no LM candidate step survived
///   screening, so there is no realized Δdev to test. Instead, the
///   model's *predicted* reduction from the unaccepted step (`predicted
///   = -(g·d + ½ d·H·d)`) is compared against the much tighter model
///   noise floor `1e-12 · max(|Φ|, 1)`. This preserves the historical
///   LM-rejection acceptance criterion exactly: only the
///   near-stationary-plateau branch is eligible (saturated-η and
///   relative-band tests both rely on a realized deviance change and
///   would widen acceptance if applied with `predicted=0`).
#[derive(Clone, Copy, Debug)]
pub(super) enum SoftAcceptProgress {
    /// Realized change in penalized deviance from the most recent
    /// accepted step (per-iter) or final accepted step (post-loop).
    Realized { dev_change: f64 },
    /// Predicted reduction `-(g·d + ½ d·H·d)` from the unaccepted LM
    /// candidate step, paired with the current penalized objective so
    /// the helper can scale the model noise floor consistently with the
    /// LM-rejection branch's historical `1e-12 · max(|Φ|, 1)` cutoff.
    Predicted {
        predicted_reduction: f64,
        current_penalized: f64,
    },
}


/// Evaluate every "soft" acceptance criterion that the post-loop rescue
/// applies to a fit which has hit `MaxIterations`. Returns the first
/// matching reason, or `None` if no criterion fires.
///
/// Three call sites share this helper:
///
/// 1. **Per-iter** (after an accepted step) — gated on a 2-iter plateau
///    streak so a single noisy step that briefly satisfies the band
///    can't trigger an early exit. All three branches are eligible.
/// 2. **Post-loop rescue** (MaxIterations hit) — accepts immediately;
///    all three branches are eligible.
/// 3. **LM-rejection** (no candidate step survived screening) — accepts
///    immediately, but only the [`PirlsSoftAccept::NearStationaryPlateau`]
///    branch is eligible, with the tighter model noise floor that the
///    historical LM-rejection check used. Saturated-η and relative-band
///    tests need a realized Δdev and are skipped.
///
/// Sharing the helper guarantees the three acceptance contexts stay in
/// lockstep — anything accepted post-loop is also a candidate for
/// early-exit, and the LM-rejection branch accepts exactly the same set
/// of states it accepted before unification.
#[inline]
pub(super) fn pirls_soft_acceptance(
    state: &WorkingState,
    projected_grad: f64,
    progress: SoftAcceptProgress,
    max_abs_eta: f64,
    progress_tol: f64,
    kkt_tol: f64,
) -> Option<PirlsSoftAccept> {
    let objective_scale = state.deviance.abs().max(state.penalty_term.abs()).max(1.0);
    // Progress tests stay on the fixed PIRLS tolerance; only KKT stationarity uses kkt_tol.
    let scaled_dev_tol = progress_tol * objective_scale;

    // Near-stationary plateau is eligible in every context. The only
    // thing that varies is which "is the fit still moving?" signal we
    // compare against which floor.
    let near_stationary_plateau = match progress {
        SoftAcceptProgress::Realized { dev_change } => {
            state.near_stationary_kkt(projected_grad, kkt_tol) && dev_change.abs() < scaled_dev_tol
        }
        SoftAcceptProgress::Predicted {
            predicted_reduction,
            current_penalized,
        } => {
            // Historical LM-rejection floor: model-predicted reduction
            // below `1e-12 · max(|Φ|, 1)` is indistinguishable from
            // numerical noise on the quadratic model. Keep this exact
            // formula — it is strictly tighter than `tol · scaled_dev_tol`
            // for the standard tol=1e-6, so the unified helper does not
            // widen the LM-rejection acceptance set.
            let reduction_noise_floor = current_penalized.abs().max(1.0) * 1e-12;
            state.near_stationary_kkt(projected_grad, kkt_tol)
                && predicted_reduction.abs() <= reduction_noise_floor
        }
    };
    if near_stationary_plateau {
        return Some(PirlsSoftAccept::NearStationaryPlateau);
    }

    // The remaining branches both require a realized Δdev to be
    // meaningful: η-cap saturation tests "did the step move and yet η
    // stayed pinned at the cap?", and the relative-band plateau tests a
    // signed, magnitude-bounded Δdev. Substituting `predicted=0` would
    // trivially satisfy both with zero diagnostic value and would widen
    // the LM-rejection acceptance set, so they are gated on a Realized
    // progress signal.
    let dev_change = match progress {
        SoftAcceptProgress::Realized { dev_change } => dev_change,
        SoftAcceptProgress::Predicted { .. } => return None,
    };

    if max_abs_eta >= PIRLS_ETA_ABS_CAP * (1.0 - 1e-12) && dev_change.abs() < scaled_dev_tol {
        return Some(PirlsSoftAccept::BoundarySaturation);
    }

    if projected_grad <= progress_tol.max(1e-6) * objective_scale
        && dev_change.abs() < scaled_dev_tol * 0.1
        && dev_change >= 0.0
    {
        return Some(PirlsSoftAccept::RelativeBandPlateau);
    }

    None
}


pub(super) fn constrained_stationarity_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> f64 {
    // `gradient`, `beta`, and `linear_constraints` are all represented in the
    // current PIRLS coefficient basis (raw sparse-native or Qs-transformed).
    // At an active inequality, the raw gradient can carry a valid KKT
    // multiplier, so convergence must use the full KKT residual in that same
    // frame rather than the unprojected gradient norm.
    if let Some(constraints) = linear_constraints {
        let kkt = compute_constraint_kkt_diagnostics(beta, gradient, constraints);
        return kkt
            .primal_feasibility
            .max(kkt.dual_feasibility)
            .max(kkt.complementarity)
            .max(kkt.stationarity);
    }
    projected_gradient_norm(gradient, beta, lower_bounds)
}


fn count_dense_upper_nnz(matrix: &Array2<f64>, tol: f64) -> usize {
    let p = matrix.nrows().min(matrix.ncols());
    let mut nnz = 0usize;
    for col in 0..p {
        for row in 0..=col {
            if matrix[[row, col]].abs() > tol {
                nnz += 1;
            }
        }
    }
    nnz
}


fn estimate_sparse_native_decision(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    let p = x_original.ncols();
    let nnz_s_lambda = count_dense_upper_nnz(s_lambda, 1e-12);
    let dense_reject = |reason: &'static str, nnz_x: usize| SparsePirlsDecision {
        path: PirlsLinearSolvePath::DenseTransformed,
        reason,
        p,
        nnz_x,
        nnz_xtwx_symbolic: None,
        nnz_s_lambda,
        nnz_h_est: None,
        density_h_est: None,
    };

    // Constrained solves require the dense active-set / projected Newton machinery.
    let has_finite_lower_bounds = coefficient_lower_bounds
        .map(|lb| lb.iter().any(|bound| bound.is_finite()))
        .unwrap_or(false);
    if has_finite_lower_bounds || linear_constraints_original.is_some() {
        return dense_reject("constraints_present", 0);
    }

    let x_sparse = if let Some(sparse) = x_original.as_sparse() {
        sparse
    } else {
        // Count nonzeros via chunks so operator-backed dense designs
        // (e.g. lazy ScaleDeviationOperator) participate in this diagnostic
        // path without forcing a full materialization.
        let row_chunk_start = std::time::Instant::now();
        let n = x_original.nrows();
        let chunk = row_chunk_for_byte_budget(n, x_original.ncols());
        let mut nnz: usize = 0;
        let mut chunks_processed = 0usize;
        if chunk > 0 && n > 0 {
            let mut start = 0;
            while start < n {
                let end = (start + chunk).min(n);
                chunks_processed += 1;
                match x_original.try_row_chunk(start..end) {
                    Ok(rows) => {
                        nnz = nnz.saturating_add(rows.iter().filter(|v| v.abs() > 1e-12).count());
                    }
                    Err(_) => {
                        nnz = nnz.saturating_add((end - start).saturating_mul(x_original.ncols()));
                    }
                }
                start = end;
            }
        }
        log::info!(
            "[STAGE] PIRLS row-chunk generation chunks={} n={} p={} nnz={} elapsed={:.3}s",
            chunks_processed,
            n,
            x_original.ncols(),
            nnz,
            row_chunk_start.elapsed().as_secs_f64(),
        );
        return dense_reject("design_not_sparse", nnz);
    };
    let nnz_x = x_sparse.val().len();
    match workspace.sparse_penalized_system_stats(x_sparse, s_lambda) {
        Ok(stats) => SparsePirlsDecision {
            path: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                PirlsLinearSolvePath::SparseNative
            } else {
                PirlsLinearSolvePath::DenseTransformed
            },
            reason: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                "sparse_native_eligible"
            } else {
                "penalized_hessian_too_dense"
            },
            p,
            nnz_x,
            nnz_xtwx_symbolic: Some(stats.nnz_xtwx_symbolic),
            nnz_s_lambda: stats.nnz_s_lambda_upper,
            nnz_h_est: Some(stats.nnz_h_upper),
            density_h_est: Some(stats.density_upper),
        },
        Err(_) => dense_reject("sparse_stats_failed", nnz_x),
    }
}


pub(super) fn should_use_sparse_native_pirls(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    estimate_sparse_native_decision(
        workspace,
        x_original,
        s_lambda,
        coefficient_lower_bounds,
        linear_constraints_original,
    )
}


pub(crate) fn sparse_reml_penalized_hessian(
    workspace: &mut PirlsWorkspace,
    x: &SparseColMat<usize, f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
    precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    workspace.assemble_sparse_penalized_hessian(x, weights, s_lambda, ridge, precomputed_xtwx)
}


/// Assemble a sparse SPD Hessian with adaptive diagonal ridge, returning the
/// matrix, its successful Cholesky factor, and the ridge that was needed.
///
/// Returning the factor avoids the previous double-factorization where the SPD
/// check would factor the matrix and discard the factor, then the caller would
/// immediately call `factorize_sparse_spd` again on the same matrix to solve.
pub(super) fn ensure_sparse_positive_definitewithridge<F>(
    mut assemble: F,
) -> Result<
    (
        SparseColMat<usize, f64>,
        crate::linalg::sparse_exact::SparseExactFactor,
        f64,
    ),
    EstimationError,
>
where
    F: FnMut(f64) -> Result<SparseColMat<usize, f64>, EstimationError>,
{
    // Step 1 — genuine round-off stabilization. A symmetric Hessian assembled
    // from `XᵀWX + S_λ` is mathematically PSD; the only reason an exact-arithmetic
    // PSD matrix fails a Cholesky is floating-point round-off in the assembly,
    // which a fixed tiny nugget on the diagonal cures. This is the principled,
    // scale-free first attempt and the common case.
    let h0 = assemble(0.0)?;
    if let Ok(factor) = factorize_sparse_spd(&h0) {
        return Ok((h0, factor, 0.0));
    }
    let h_eps = assemble(FIXED_STABILIZATION_RIDGE)?;
    if let Ok(factor) = factorize_sparse_spd(&h_eps) {
        return Ok((h_eps, factor, FIXED_STABILIZATION_RIDGE));
    }

    // Step 2 — the matrix is genuinely non-PD (rank-deficiency, wrong-sign
    // curvature, or weight underflow in the Hessian assembly), not mere
    // round-off. Rather than escalate a magic ridge by powers of ten until it
    // happens to factorize — which silently perturbs the exported curvature by
    // an unknown amount — we SURFACE the conditioning problem and set the ridge
    // DIRECTLY from a rigorous spectral bound.
    //
    // Gershgorin's circle theorem gives a guaranteed lower bound on the smallest
    // eigenvalue: λ_min(H) ≥ min_i ( H_ii − Σ_{j≠i} |H_ij| ). Adding a diagonal
    // ridge τ shifts the whole spectrum up by τ, so choosing
    //
    //     τ = (margin·scale) − gershgorin_lower_bound
    //
    // guarantees the Gershgorin lower bound of `H + τ·I` is `≥ margin·scale > 0`,
    // hence the shifted matrix is provably SPD. This costs ONE bound pass
    // (O(nnz)) and ONE factorization instead of geometric trial-and-error, and
    // the ridge is tied to the actual most-negative curvature rather than a
    // timeout-shaped iteration count.
    let (gershgorin_min, diag_scale) = gershgorin_min_eig_lower_bound(&h_eps);

    // Round-off margin relative to the matrix scale: enough to clear the gap
    // between the (conservative) Gershgorin bound and the pivoting tolerance of
    // the sparse Cholesky, without over-regularizing.
    let scale = diag_scale.max(1.0);
    let margin = FIXED_STABILIZATION_RIDGE * scale;
    let direct_ridge = (margin - gershgorin_min).max(FIXED_STABILIZATION_RIDGE);

    log::warn!(
        "sparse penalized Hessian is not positive definite (Gershgorin λ_min ≥ {:.3e}, \
         diag scale {:.3e}); regularizing curvature with direct ridge {:.3e}. Exported \
         curvature/SEs are stabilized, not exact — investigate rank-deficiency or weight \
         underflow in the Hessian assembly.",
        gershgorin_min,
        scale,
        direct_ridge,
    );

    // The Gershgorin-derived ridge is provably sufficient; the only reason it
    // could still fail is a degenerate non-symmetric / non-finite assembly. We
    // allow a single conservative doubling to absorb residual pivot round-off,
    // then fail loud rather than silently shipping a heavily-ridged surrogate.
    for ridge in [direct_ridge, direct_ridge * 2.0] {
        let h = assemble(ridge)?;
        if let Ok(factor) = factorize_sparse_spd(&h) {
            return Ok((h, factor, ridge));
        }
    }

    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: gershgorin_min,
    })
}

/// Rigorous lower bound on the smallest eigenvalue of a symmetric sparse matrix
/// via Gershgorin's circle theorem, plus the largest |diagonal| as a scale.
///
/// Returns `(λ_min_lower_bound, diag_scale)`. The bound is storage-agnostic:
/// off-diagonal magnitudes are added to the radius of both endpoints, so
/// upper-only, lower-only, and full-symmetric storage all yield a valid (and at
/// worst conservative) lower bound — it never over-claims positive-definiteness.
fn gershgorin_min_eig_lower_bound(h: &SparseColMat<usize, f64>) -> (f64, f64) {
    let n = h.ncols();
    let mut diag = vec![0.0_f64; n];
    let mut radius = vec![0.0_f64; n];
    let (symbolic, values) = h.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..n {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for idx in start..end {
            let row = row_idx[idx];
            let value = values[idx];
            if row == col {
                diag[col] += value;
            } else {
                let a = value.abs();
                radius[row] += a;
                radius[col] += a;
            }
        }
    }
    let mut min_bound = f64::INFINITY;
    let mut diag_scale = 0.0_f64;
    for i in 0..n {
        min_bound = min_bound.min(diag[i] - radius[i]);
        diag_scale = diag_scale.max(diag[i].abs());
    }
    if !min_bound.is_finite() {
        min_bound = f64::NEG_INFINITY;
    }
    (min_bound, diag_scale)
}


fn solve_subsystem_direction(
    h_sub: ndarray::ArrayView2<f64>,
    g_sub: ndarray::ArrayView1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let n = g_sub.len();
    if out.len() != n {
        *out = Array1::zeros(n);
    }
    // Try direct factorization first.
    if let Ok(factor) = StableSolver::new("pirls bounded subsystem").factorize_any(&h_sub) {
        out.assign(&g_sub);
        let mut rhs = array1_to_col_matmut(out);
        factor.solve_in_place(rhs.as_mut());
        out.mapv_inplace(|v| -v);
        if array_is_finite(out) {
            return Ok(());
        }
    }
    // Factorization failed or produced non-finite values — the reduced Hessian
    // is singular or nearly so (common on underdetermined problems).  Add a
    // diagonal ridge and retry with geometrically increasing strength.
    let diag_scale = (0..n)
        .map(|i| h_sub[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mut tau = 1e-8 * diag_scale;
    let mut h_reg = h_sub.to_owned();
    for _ in 0..12 {
        for i in 0..n {
            h_reg[[i, i]] = h_sub[[i, i]] + tau;
        }
        if let Ok(factor) = StableSolver::new("pirls bounded subsystem ridge").factorize(&h_reg) {
            out.assign(&g_sub);
            let mut rhs = array1_to_col_matmut(out);
            factor.solve_in_place(rhs.as_mut());
            out.mapv_inplace(|v| -v);
            if array_is_finite(out) {
                return Ok(());
            }
        }
        tau *= 10.0;
    }
    // All ridge attempts failed — fall back to steepest descent on the
    // free subspace: d = -g / ||g||, scaled to a conservative step.
    let gnorm = g_sub.dot(&g_sub).sqrt();
    if gnorm > 0.0 {
        let scale = 1.0 / gnorm.max(diag_scale);
        for i in 0..n {
            out[i] = -g_sub[i] * scale;
        }
        return Ok(());
    }
    // Zero gradient — already at optimum on this subspace.
    out.fill(0.0);
    Ok(())
}


pub(super) fn linear_constraints_from_lower_bounds(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}


pub(super) fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    active_set::compute_constraint_kkt_diagnostics(beta, gradient, constraints)
}


/// Select which active bound-constraint to release in the primal active-set
/// QP loop, or `None` when KKT is satisfied (no negative multiplier).
///
/// `use_blands` switches between two pivoting rules with the same KKT-test
/// semantics but different anti-cycling guarantees:
///
/// - `false` — **worst-violation**: release the constraint with the most
///   negative multiplier `λ_i = g_i + (H d)_i`. Greedy and fast on
///   non-degenerate problems but can cycle when several constraints have
///   multipliers near zero of comparable magnitude.
/// - `true` — **Bland's rule**: release the *lowest-index* constraint with a
///   strictly-negative multiplier (using a scale-aware deadband to ignore
///   pure round-off). This is the textbook anti-cycling choice — combined
///   with Bland-compatible tie-breaking on entering, it guarantees the
///   active-set sequence visits each vertex at most once and so terminates
///   in finitely many pivots.
pub(super) fn select_active_set_release(
    gradient: &Array1<f64>,
    hd: &Array1<f64>,
    active_idx: &[usize],
    use_blands: bool,
) -> Option<usize> {
    if use_blands {
        for &i in active_idx {
            let lambda_i = gradient[i] + hd[i];
            let scale = gradient[i].abs().max(hd[i].abs()).max(1.0);
            let tol = 64.0 * f64::EPSILON * scale;
            if lambda_i < -tol {
                return Some(i);
            }
        }
        None
    } else {
        let mut worst = 0.0_f64;
        let mut idx = None;
        for &i in active_idx {
            let lambda_i = gradient[i] + hd[i];
            if lambda_i < worst {
                worst = lambda_i;
                idx = Some(i);
            }
        }
        idx
    }
}


pub(crate) fn solve_newton_directionwith_lower_bounds(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: &Array1<f64>,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    // Bound-constrained Newton step on the local quadratic model:
    //
    //   min_d  g^T d + 0.5 d^T H d
    //   s.t.   beta + d >= l
    //
    // KKT conditions for active bounds A:
    //   d_A = 0,
    //   H_FF d_F = -g_F,
    //   lambda_A = g_A + (H d)_A >= 0.
    //
    // We solve the free subsystem, enforce primal feasibility by clipping to the
    // first boundary hit, then enforce dual feasibility by releasing active bounds
    // with negative multipliers. This is the standard primal active-set loop for
    // strictly convex box QPs.
    let p = gradient.len();
    if lower_bounds.len() != p || beta.len() != p {
        crate::bail_invalid_estim!(
            "lower-bound size mismatch: beta={}, gradient={}, bounds={}",
            beta.len(),
            gradient.len(),
            lower_bounds.len()
        );
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    direction_out.fill(0.0);

    // Fast path: if unconstrained Newton step is already feasible for all lower
    // bounds, it is the exact constrained minimizer (strict convex quadratic).
    let has_active_hint = active_hint
        .as_ref()
        .map(|hint| !hint.is_empty())
        .unwrap_or(false);
    if !has_active_hint && solve_newton_direction_dense(hessian, gradient, direction_out).is_ok() {
        let mut feasible = true;
        for i in 0..p {
            let lb = lower_bounds[i];
            if lb.is_finite() && beta[i] + direction_out[i] < lb {
                feasible = false;
                break;
            }
        }
        if feasible {
            return Ok(());
        }
    }

    let mut active = vec![false; p];
    if let Some(hint) = active_hint.as_ref() {
        for &idx in hint.iter() {
            if idx < p {
                active[idx] = true;
            }
        }
    }
    for i in 0..p {
        let lb = lower_bounds[i];
        if lb.is_finite() && gradient[i] > 0.0 {
            // Use a relative+absolute tolerance matching projected_gradient_norm
            // so coefficients near the bound (e.g. I-spline at 1e-6) with positive
            // gradient (KKT multiplier) are correctly identified as active.
            let scale = beta[i].abs().max(lb.abs()).max(1.0);
            let tol = ACTIVE_BOUND_REL_TOL * scale + ACTIVE_BOUND_ABS_TOL;
            if beta[i] <= lb + tol {
                active[i] = true;
            }
        }
    }

    // Hybrid pivoting: worst-violation gives faster average convergence on
    // non-degenerate problems but can cycle at degenerate vertices (multiple
    // active constraints with multipliers near zero, ping-ponging activate/
    // release of the same coordinate). After a worst-violation grace period
    // we switch to Bland's lowest-index rule, which monotonically orders the
    // active-set sequence visited and therefore terminates in finitely many
    // additional pivots. Entering already uses Bland-compatible tie-breaking
    // (smallest α_hit, ties broken by ascending free-index iteration order
    // because `boundary_hit_step_fraction` requires `step < current_step_limit`
    // strictly), so the leaving rule is the only place anti-cycling has to
    // be enforced.
    const BLANDS_RULE_GRACE: usize = 2;
    let blands_threshold = BLANDS_RULE_GRACE * (p + 1);
    let max_iters = 8 * (p + 1);
    let mut d_free = Array1::<f64>::zeros(p);
    // Reusable hoisted buffers for the free-block Newton subsystem; sliced down
    // to the current `n_free` each iteration to avoid reallocating the p×p
    // block and length-p prefix on every active-set pivot.
    let mut h_ff_buf = Array2::<f64>::zeros((p, p));
    let mut g_f_buf = Array1::<f64>::zeros(p);
    for it in 0..max_iters {
        let use_blands = it >= blands_threshold;
        let free_idx: Vec<usize> = (0..p).filter(|&i| !active[i]).collect();
        let active_idx: Vec<usize> = (0..p).filter(|&i| active[i]).collect();
        direction_out.fill(0.0);
        for &i in &active_idx {
            let lb = lower_bounds[i];
            if lb.is_finite() {
                direction_out[i] = lb - beta[i];
            }
        }
        if free_idx.is_empty() {
            let hd = fast_av(hessian, direction_out);
            if let Some(idx) = select_active_set_release(gradient, &hd, &active_idx, use_blands) {
                active[idx] = false;
                continue;
            }
            if let Some(hint) = active_hint {
                hint.clear();
                hint.extend((0..p).filter(|&i| active[i]));
            }
            return Ok(());
        }

        let n_free = free_idx.len();
        // Reuse hoisted top-left n_free×n_free block and length-n_free prefix.
        {
            let mut h_ff = h_ff_buf.slice_mut(ndarray::s![..n_free, ..n_free]);
            let mut g_f = g_f_buf.slice_mut(ndarray::s![..n_free]);
            for (ii, &i) in free_idx.iter().enumerate() {
                let mut gi = gradient[i];
                for &j in &active_idx {
                    gi += hessian[[i, j]] * direction_out[j];
                }
                g_f[ii] = gi;
                for (jj, &j) in free_idx.iter().enumerate() {
                    h_ff[[ii, jj]] = hessian[[i, j]];
                }
            }
        }
        solve_subsystem_direction(
            h_ff_buf.slice(ndarray::s![..n_free, ..n_free]),
            g_f_buf.slice(ndarray::s![..n_free]),
            &mut d_free,
        )?;
        for (ii, &i) in free_idx.iter().enumerate() {
            direction_out[i] = d_free[ii];
        }

        // Enforce primal feasibility for bound-constrained coefficients.
        let mut hit_idx: Option<usize> = None;
        let mut best_alpha = 1.0_f64;
        for &i in &free_idx {
            let lb = lower_bounds[i];
            if !lb.is_finite() {
                continue;
            }
            let slack = beta[i] - lb;
            let di = direction_out[i];
            if let Some(alpha_i) = boundary_hit_step_fraction(slack, di, best_alpha) {
                best_alpha = alpha_i;
                hit_idx = Some(i);
            }
        }
        if let Some(i_hit) = hit_idx {
            for i in 0..p {
                direction_out[i] *= best_alpha;
            }
            active[i_hit] = true;
            continue;
        }

        // Dual feasibility on active constraints:
        // λ_i = g_i + (H d)_i must be >= 0 for all active lower bounds.
        let hd = fast_av(hessian, direction_out);
        if let Some(idx) = select_active_set_release(gradient, &hd, &active_idx, use_blands) {
            active[idx] = false;
            continue;
        }

        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend((0..p).filter(|&i| active[i]));
        }
        return Ok(());
    }

    // Active-set loop did not converge — fall back to a projected gradient
    // step.  This is always feasible and gives a descent direction, letting the
    // outer LM loop decide whether to accept or increase damping.
    let gnorm = gradient.dot(gradient).sqrt();
    if gnorm > 0.0 {
        let diag_scale = (0..p)
            .map(|i| hessian[[i, i]].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let step_scale = 1.0 / diag_scale;
        for i in 0..p {
            let di = -gradient[i] * step_scale;
            let lb = lower_bounds[i];
            if lb.is_finite() && beta[i] + di < lb {
                direction_out[i] = lb - beta[i];
            } else {
                direction_out[i] = di;
            }
        }
    } else {
        direction_out.fill(0.0);
    }
    if let Some(hint) = active_hint {
        hint.clear();
    }
    Ok(())
}


/// Reduce a constraint matrix to full row rank using column-pivoted QR on A^T.
///
/// Given k constraint rows in R^p, computes the numerical row rank r via
/// pivoted QR of A^T (p × k) with a tolerance scaled to `eps · max(k, p) ·
/// |R₀₀|`, and retains only the r pivot rows.  Dropped rows have their
/// group membership merged into the most-aligned kept row so that the
/// active-set QP can still release the underlying original constraints via
/// multiplier signs.
///
/// This is a shared numerical primitive used by both the PIRLS and
/// custom-family active-set solvers.
pub(super) fn solve_newton_directionwith_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    active_set::solve_newton_direction_with_linear_constraints(
        hessian,
        gradient,
        beta,
        constraints,
        direction_out,
        active_hint,
    )
}


// loop_driver owns: default_beta_guess_external, solve_intercept_for_prevalence,
// assemble_pirls_result, detect_logit_instability, stack_lambdaweighted_penalty_root_canonical,
// build_sparse_native_reparam_result, build_diagonal_penalty_from_kronecker, canonical_prior_shift,
// canonical_prior_mean_aggregate, PirlsProblem, PenaltyConfig, fit_model_for_fixed_rho,
// fit_model_for_fixed_rho_with_adaptive_kkt, PirlsConfig, make_reparam_operator,
// build_transformed_lower_bound_constraints*, build_transformed_linear_constraints*,
// merge_linear_constraints, sparse_from_denseview.
use loop_driver::assert_symmetric_tol;

pub(crate) use loop_driver::fit_model_for_fixed_rho_with_adaptive_kkt;

pub use loop_driver::{PenaltyConfig, PirlsConfig, PirlsProblem, fit_model_for_fixed_rho};


#[inline]
pub(super) fn standard_inverse_link_jet(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<MixtureInverseLinkJet, EstimationError> {
    crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta)
}


#[inline]
fn bernoulli_logit_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: crate::mixture_link::LogitJet5,
    zero_on_nonsmooth: bool,
) -> WorkingBernoulliGeometry {
    let fisher = jet.d1;
    let nonsmooth = eta_raw != eta_used || !fisher.is_finite() || fisher < 0.0;
    let (c, d) = if nonsmooth && zero_on_nonsmooth {
        (0.0, 0.0)
    } else {
        (priorweight * jet.d2, priorweight * jet.d3)
    };
    WorkingBernoulliGeometry {
        mu: jet.mu,
        weight: priorweight * fisher,
        z: bernoulli_exact_working_response(eta_used, y, jet.mu, jet.d1),
        c,
        d,
    }
}


/// Compute working IRLS geometry for a single Bernoulli observation.
///
/// This helper returns the exact statistical working state. It does not floor
/// the Fisher mass or the working response for solver conditioning; doing so
/// would change the model rather than just the Newton system.
///
/// The weight returned is the **Fisher** (expected information) weight
/// W_F = h'(η)² / V(μ). The c and d fields are likewise the Fisher
/// derivatives c_F = dW_F/dη and d_F = d²W_F/dη².
///
/// NOTE: For non-canonical links (probit, cloglog, SAS, mixture), the
/// observed weight differs:
///   W_obs = W_F − (y−μ) · B,  B = (h''V − h'²V') / V²
/// The observed c/d include residual-dependent corrections. PIRLS keeps
/// these Fisher carriers for the score-side RHS `X'W(z-eta) - S beta`,
/// while the Newton/Laplace Hessian side may switch to the observed,
/// clamped curvature surface. The accepted Hessian-side c/d arrays are
/// stored separately in `PirlsResult::solve_c_array` / `solve_d_array`
/// and consumed directly by the REML/LAML exact-derivative code.
#[inline]
fn bernoulli_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: MixtureInverseLinkJet,
) -> WorkingBernoulliGeometry {
    let mu = jet.mu;
    let v = mu * (1.0 - mu);
    let n0 = jet.d1 * jet.d1;
    let fisher = if v.is_finite() && v > 0.0 {
        n0 / v
    } else {
        0.0
    };
    let nonsmooth =
        eta_raw != eta_used || !v.is_finite() || v <= 0.0 || !fisher.is_finite() || fisher < 0.0;
    let (c, d) = if nonsmooth {
        (0.0, 0.0)
    } else {
        let v1 = jet.d1 * (1.0 - 2.0 * mu);
        let v2 = jet.d2 * (1.0 - 2.0 * mu) - 2.0 * jet.d1 * jet.d1;
        let n1 = 2.0 * jet.d1 * jet.d2;
        let n2 = 2.0 * (jet.d2 * jet.d2 + jet.d1 * jet.d3);
        let numer1 = n1 * v - n0 * v1;
        let c = priorweight * numer1 / (v * v);
        let d = priorweight * ((n2 * v - n0 * v2) / (v * v) - 2.0 * numer1 * v1 / (v * v * v));
        (c, d)
    };
    WorkingBernoulliGeometry {
        mu,
        weight: priorweight * fisher,
        z: bernoulli_exact_working_response(eta_used, y, mu, jet.d1),
        c,
        d,
    }
}


#[inline]
fn bernoulli_exact_working_response(eta: f64, y: f64, mu: f64, dmu_deta: f64) -> f64 {
    // Preserve the exact IRLS score carrier W(z-eta) = y-mu whenever the link
    // jet is finite. Numerical conditioning belongs in the linear solve, not in
    // the Bernoulli likelihood geometry.
    if dmu_deta.is_finite() && dmu_deta > 0.0 {
        let delta = (y - mu) / dmu_deta;
        if delta.is_finite() {
            return eta + delta;
        }
    }
    eta
}


#[inline]
fn write_identityworking_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    mu.assign(eta);
    weights.assign(&priorweights);
    z.assign(&y);
    if let Some(derivs) = derivatives {
        derivs.c.fill(0.0);
        derivs.d.fill(0.0);
        derivs.dmu_deta.fill(1.0);
        derivs.d2mu_deta2.fill(0.0);
        derivs.d3mu_deta3.fill(0.0);
    }
}


/// Working state for Poisson with a log link.
///
/// `V(mu) = mu`, so the Fisher weight is `prior * mu` and the canonical-link
/// curvature buffers both equal the working weight.
#[inline]
fn write_poisson_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::PoissonIdentity,
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: 1.0,
                d_ratio: 1.0,
            },
            floor_weight: true,
            zero_mu_jet_on_clamp: false,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
}


/// Working state for Gamma(shape = k) with a log link.
///
/// With `mu = exp(eta)` and `V(mu) = mu^2`, the Fisher weight is the
/// prior/sample weight scaled by the fixed Gamma shape, independent of `eta`;
/// the weight is therefore written unfloored and the curvature buffers vanish.
#[inline]
fn write_gamma_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    shape: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::Constant { factor: shape },
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: 0.0,
                d_ratio: 0.0,
            },
            floor_weight: false,
            zero_mu_jet_on_clamp: false,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
}


pub const BETA_MU_EPS: f64 = 1.0e-12;


#[inline]
fn tweedie_log_weight_mu_power(mu: f64, p: f64) -> f64 {
    // Match the 1e-300 MIN_DEVIANCE floor used by the REML deviance path:
    // smaller positive mu values are below a non-degenerate f64 likelihood
    // contribution, but flooring here keeps mu^(2-p) away from underflow.
    mu.max(1.0e-300).powf(2.0 - p)
}


#[inline]
fn valid_negbin_theta(theta: f64) -> bool {
    theta.is_finite() && theta > 0.0
}


#[inline]
fn valid_count_response(y: f64) -> bool {
    y.is_finite() && y >= 0.0 && (y - y.round()).abs() <= 1e-9
}


fn validate_count_responses(
    y: &ArrayView1<'_, f64>,
    priorweights: &ArrayView1<'_, f64>,
    family: &str,
) -> Result<(), EstimationError> {
    for (i, (&yi, &wi)) in y.iter().zip(priorweights.iter()).enumerate() {
        if wi > 0.0 && !valid_count_response(yi) {
            crate::bail_invalid_estim!(
                "{family} response must be a finite non-negative integer at positive-weight row {i}; got {yi}"
            );
        }
    }
    Ok(())
}


#[inline]
fn valid_beta_phi(phi: f64) -> bool {
    phi.is_finite() && phi > 0.0
}


#[inline]
fn valid_beta_response(y: f64) -> bool {
    y.is_finite() && y > 0.0 && y < 1.0
}


fn validate_beta_responses(
    y: &ArrayView1<'_, f64>,
    priorweights: &ArrayView1<'_, f64>,
) -> Result<(), EstimationError> {
    for (i, (&yi, &wi)) in y.iter().zip(priorweights.iter()).enumerate() {
        if wi > 0.0 && !valid_beta_response(yi) {
            crate::bail_invalid_estim!(
                "beta-regression response must be finite and strictly inside (0, 1) at positive-weight row {i}; got {yi}"
            );
        }
    }
    Ok(())
}


#[inline]
fn valid_tweedie_response(y: f64) -> bool {
    y.is_finite() && y >= 0.0
}


fn validate_tweedie_responses(
    y: &ArrayView1<'_, f64>,
    priorweights: &ArrayView1<'_, f64>,
) -> Result<(), EstimationError> {
    for (i, (&yi, &wi)) in y.iter().zip(priorweights.iter()).enumerate() {
        if wi > 0.0 && !valid_tweedie_response(yi) {
            crate::bail_invalid_estim!(
                "Tweedie response must be finite and non-negative at positive-weight row {i}; got {yi}"
            );
        }
    }
    Ok(())
}


#[inline]
fn safe_beta_mu(mu: f64) -> f64 {
    mu.clamp(BETA_MU_EPS, 1.0 - BETA_MU_EPS)
}


#[inline]
fn trigamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
        - inv2 * inv2 * inv2 * inv2 * inv / 30.0
}


#[inline]
fn polygamma2(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc -= 2.0 / (x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    acc - inv2 - inv3 - 0.5 * inv2 * inv2 + inv3 * inv3 / 6.0 - inv2 * inv3 * inv3 / 6.0
        + 0.3 * inv2 * inv2 * inv3 * inv3
        - 5.0 * inv2 * inv2 * inv2 * inv3 * inv3 / 6.0
}


#[inline]
fn polygamma3(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 6.0 / (x * x * x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv4 = inv2 * inv2;
    acc + 2.0 * inv3 + 3.0 * inv4 + 2.0 * inv4 * inv - inv4 * inv3 + 4.0 * inv4 * inv3 * inv2 / 3.0
        - 3.0 * inv4 * inv3 * inv4
        + 10.0 * inv4 * inv4 * inv4 * inv
}


#[inline]
fn beta_logit_working_curvature_eta_derivatives(
    prior_weight: f64,
    phi: f64,
    mu: f64,
    q: f64,
    a: f64,
    b: f64,
    trigamma_sum: f64,
) -> (f64, f64) {
    let q_prime = q * (1.0 - 2.0 * mu);
    let q_double_prime = q * (1.0 - 2.0 * mu) * (1.0 - 2.0 * mu) - 2.0 * q * q;
    let psi2_diff = polygamma2(a) - polygamma2(b);
    let psi3_sum = polygamma3(a) + polygamma3(b);
    let phi_sq = phi * phi;
    let q_sq = q * q;
    let c = prior_weight * phi_sq * (2.0 * q * q_prime * trigamma_sum + q_sq * phi * q * psi2_diff);
    let d = prior_weight
        * phi_sq
        * (2.0 * (q_prime * q_prime + q * q_double_prime) * trigamma_sum
            + 4.0 * q * q_prime * phi * q * psi2_diff
            + q_sq * (phi * q_prime * psi2_diff + phi_sq * q_sq * psi3_sum));
    (c, d)
}


/// Working state for Tweedie with a log link.
///
/// With `mu = exp(eta)`, `V(mu) = phi * mu^p`, and `g'(mu) = 1 / mu`, the Fisher
/// working weight is `mu^(2-p) / phi`, scaled by prior weight. The `mu`-jet must
/// be zeroed when `eta` is clamped because the fractional power makes the local
/// jet unreliable there. Parameter ranges and responses are validated up front.
#[inline]
fn write_tweedie_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    p: f64,
    phi: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if !is_valid_tweedie_power(p) {
        crate::bail_invalid_estim!(
            "Tweedie variance power must be finite and strictly between 1 and 2; got {p}",
            p = p
        );
    }
    if !(phi.is_finite() && phi > 0.0) {
        crate::bail_invalid_estim!(
            "Tweedie dispersion phi must be finite and > 0; got {phi}",
            phi = phi
        );
    }
    validate_tweedie_responses(&y, &priorweights)?;
    let exponent = 2.0 - p;
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::TweediePower { p, phi },
            curvature: log_link_working_state::WorkingCurvature::Proportional {
                c_ratio: exponent,
                d_ratio: exponent * exponent,
            },
            floor_weight: true,
            zero_mu_jet_on_clamp: true,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
    Ok(())
}


/// Working state for NB(mu, theta) with a log link and fixed theta.
///
/// The size parameter is treated as a fixed hyperparameter for this GLM stack;
/// no theta profiling or REML update is performed here. The Fisher weight is
/// `mu * theta / (theta + mu)`, written in the numerically-stable branch form
/// that avoids cancellation for very small or very large `mu / theta`.
#[inline]
fn write_negative_binomial_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    theta: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if !valid_negbin_theta(theta) {
        crate::bail_invalid_estim!(
            "negative-binomial theta must be finite and > 0; got {theta}",
            theta = theta
        );
    }
    validate_count_responses(&y, &priorweights, "negative-binomial")?;
    log_link_working_state::write_log_link_working_state(
        &log_link_working_state::LogLinkRule {
            weight: log_link_working_state::WorkingWeight::NegativeBinomial { theta },
            curvature: log_link_working_state::WorkingCurvature::NegativeBinomial { theta },
            floor_weight: true,
            zero_mu_jet_on_clamp: false,
        },
        y,
        eta,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    );
    Ok(())
}


/// Working state for Beta(mu * phi, (1 - mu) * phi) with a logit link.
#[inline]
fn write_beta_logit_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    phi: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if !valid_beta_phi(phi) {
        crate::bail_invalid_estim!("beta-regression phi must be finite and > 0; got {phi}");
    }
    validate_beta_responses(&y, &priorweights)?;
    if let Some(mut derivs) = derivatives {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        let WorkingDerivSlices {
            c: c_s,
            d: d_s,
            dmu: dmu_s,
            d2: d2_s,
            d3: d3_s,
        } = working_deriv_slices(&mut derivs);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                    let jet = logit_inverse_link_jet5(eta_i);
                    let mu_i = safe_beta_mu(jet.mu);
                    let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                    let yi = y[i];
                    let a = (mu_i * phi).max(BETA_MU_EPS);
                    let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                    let score_mu = phi * (digamma(b) - digamma(a) + yi.ln() - (1.0 - yi).ln());
                    let trigamma_sum = trigamma(a) + trigamma(b);
                    let info_mu = phi * phi * trigamma_sum;
                    let prior_weight = priorweights[i].max(0.0);
                    let raw_weight = prior_weight * q * q * info_mu;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *mu_o = mu_i;
                    *w_o = if raw_weight > 0.0 {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        0.0
                    };
                    *z_o = eta_i + score_mu / (q * info_mu).max(MIN_WEIGHT);
                    *dmu_o = q;
                    *d2_o = q * (1.0 - 2.0 * mu_i);
                    *d3_o = q * (1.0 - 6.0 * q);
                    if floor_active || eta_raw != eta_i {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        let (c_i, d_i) = beta_logit_working_curvature_eta_derivatives(
                            prior_weight,
                            phi,
                            mu_i,
                            q,
                            a,
                            b,
                            trigamma_sum,
                        );
                        *c_o = c_i;
                        *d_o = d_i;
                    }
                },
            );
    } else {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
                let jet = logit_inverse_link_jet5(eta_i);
                let mu_i = safe_beta_mu(jet.mu);
                let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                let yi = y[i];
                let a = (mu_i * phi).max(BETA_MU_EPS);
                let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                let score_mu = phi * (digamma(b) - digamma(a) + yi.ln() - (1.0 - yi).ln());
                let info_mu = phi * phi * (trigamma(a) + trigamma(b));
                let raw_weight = priorweights[i].max(0.0) * q * q * info_mu;
                *mu_o = mu_i;
                *w_o = if raw_weight > 0.0 {
                    raw_weight.max(MIN_WEIGHT)
                } else {
                    0.0
                };
                *z_o = eta_i + score_mu / (q * info_mu).max(MIN_WEIGHT);
            });
    }
    Ok(())
}


/// Zero-allocation update of GLM working vectors using pre-allocated buffers.
#[inline]
pub fn update_glmvectors(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    let link = inverse_link.link_function();

    // Fast vectorized path for pure logit (most common binomial link).
    // Avoids per-element function dispatch; structured for SIMD auto-vectorization.
    if matches!(link, LinkFunction::Logit)
        && inverse_link.mixture_state().is_none()
        && inverse_link.sas_state().is_none()
    {
        if let Some(mut derivs) = derivatives {
            let WorkingSlices {
                mu: mu_s,
                weights: weights_s,
                z: z_s,
            } = working_slices(mu, weights, z);
            let WorkingDerivSlices {
                c: c_s,
                d: d_s,
                dmu: dmu_s,
                d2: d2_s,
                d3: d3_s,
            } = working_deriv_slices(&mut derivs);
            mu_s.par_iter_mut()
                .zip(weights_s.par_iter_mut())
                .zip(z_s.par_iter_mut())
                .zip(c_s.par_iter_mut())
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .for_each(
                    |(i, (((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o))| {
                        let eta_raw = eta[i];
                        let eta_c = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                        let jet = logit_inverse_link_jet5(eta_c);
                        let geom = bernoulli_logit_geometry_from_jet(
                            eta_raw,
                            eta_c,
                            y[i],
                            priorweights[i],
                            jet,
                            true,
                        );
                        *mu_o = geom.mu;
                        *w_o = geom.weight;
                        *z_o = geom.z;
                        *c_o = geom.c;
                        *d_o = geom.d;
                        *dmu_o = jet.d1;
                        *d2_o = jet.d2;
                        *d3_o = jet.d3;
                    },
                );
        } else {
            let WorkingSlices {
                mu: mu_s,
                weights: weights_s,
                z: z_s,
            } = working_slices(mu, weights, z);
            mu_s.par_iter_mut()
                .zip(weights_s.par_iter_mut())
                .zip(z_s.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((mu_o, w_o), z_o))| {
                    let eta_raw = eta[i];
                    let eta_c = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                    let jet = logit_inverse_link_jet5(eta_c);
                    let geom = bernoulli_logit_geometry_from_jet(
                        eta_raw,
                        eta_c,
                        y[i],
                        priorweights[i],
                        jet,
                        true,
                    );
                    *mu_o = geom.mu;
                    *w_o = geom.weight;
                    *z_o = geom.z;
                });
        }
        return Ok(());
    }

    match link {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => {
            // On logit geometry, freeze higher η-derivatives in nonsmooth
            // regions so PIRLS and outer derivative code differentiate the
            // same piecewise-smooth surface.
            let zero_on_nonsmooth = matches!(link, LinkFunction::Logit);
            if let Some(mut derivs) = derivatives {
                let WorkingSlices {
                    mu: mu_s,
                    weights: weights_s,
                    z: z_s,
                } = working_slices(mu, weights, z);
                let WorkingDerivSlices {
                    c: c_s,
                    d: d_s,
                    dmu: dmu_s,
                    d2: d2_s,
                    d3: d3_s,
                } = working_deriv_slices(&mut derivs);
                mu_s.par_iter_mut()
                    .zip(weights_s.par_iter_mut())
                    .zip(z_s.par_iter_mut())
                    .zip(c_s.par_iter_mut())
                    .zip(d_s.par_iter_mut())
                    .zip(dmu_s.par_iter_mut())
                    .zip(d2_s.par_iter_mut())
                    .zip(d3_s.par_iter_mut())
                    .enumerate()
                    .try_for_each(
                        |(
                            i,
                            (((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o),
                        )|
                         -> Result<(), EstimationError> {
                            let eta_used = eta_for_observed_hessian_jet(inverse_link, eta[i]);
                            if matches!(link, LinkFunction::Logit) {
                                let jet = logit_inverse_link_jet5(eta_used);
                                let geom = bernoulli_logit_geometry_from_jet(
                                    eta[i],
                                    eta_used,
                                    y[i],
                                    priorweights[i],
                                    jet,
                                    zero_on_nonsmooth,
                                );
                                *mu_o = geom.mu;
                                *w_o = geom.weight;
                                *z_o = geom.z;
                                *c_o = geom.c;
                                *d_o = geom.d;
                                *dmu_o = jet.d1;
                                *d2_o = jet.d2;
                                *d3_o = jet.d3;
                            } else {
                                let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                                let geom = bernoulli_geometry_from_jet(
                                    eta[i],
                                    eta_used,
                                    y[i],
                                    priorweights[i],
                                    jet,
                                );
                                *mu_o = geom.mu;
                                *w_o = geom.weight;
                                *z_o = geom.z;
                                *c_o = geom.c;
                                *d_o = geom.d;
                                *dmu_o = jet.d1;
                                *d2_o = jet.d2;
                                *d3_o = jet.d3;
                            }
                            Ok(())
                        },
                    )?;
            } else {
                let WorkingSlices {
                    mu: mu_s,
                    weights: weights_s,
                    z: z_s,
                } = working_slices(mu, weights, z);
                mu_s.par_iter_mut()
                    .zip(weights_s.par_iter_mut())
                    .zip(z_s.par_iter_mut())
                    .enumerate()
                    .try_for_each(|(i, ((mu_o, w_o), z_o))| -> Result<(), EstimationError> {
                        let eta_used = eta_for_observed_hessian_jet(inverse_link, eta[i]);
                        if matches!(link, LinkFunction::Logit) {
                            let jet = logit_inverse_link_jet5(eta_used);
                            let geom = bernoulli_logit_geometry_from_jet(
                                eta[i],
                                eta_used,
                                y[i],
                                priorweights[i],
                                jet,
                                zero_on_nonsmooth,
                            );
                            *mu_o = geom.mu;
                            *w_o = geom.weight;
                            *z_o = geom.z;
                        } else {
                            let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                            let geom = bernoulli_geometry_from_jet(
                                eta[i],
                                eta_used,
                                y[i],
                                priorweights[i],
                                jet,
                            );
                            *mu_o = geom.mu;
                            *w_o = geom.weight;
                            *z_o = geom.z;
                        }
                        Ok(())
                    })?;
            }
            Ok(())
        }
        LinkFunction::Identity => {
            write_identityworking_state(y, eta, priorweights, mu, weights, z, derivatives);
            Ok(())
        }
        LinkFunction::Log => {
            write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives);
            Ok(())
        }
    }
}


/// Family-dispatched GLM vector update helper.
#[inline]
pub fn update_glmvectors_by_family(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    likelihood.irls_update(y, eta, priorweights, mu, weights, z, None, None)
}


fn integrated_inverse_link_from_family(
    spec: &LikelihoodSpec,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<InverseLink, EstimationError> {
    match (&spec.response, &spec.link) {
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit))
        | (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit))
        | (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
            Ok(spec.link.clone())
        }
        (ResponseFamily::Binomial, InverseLink::Sas(_)) => {
            let state = sas_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Integrated BinomialSas update requires explicit SasLinkState".to_string(),
                )
            })?;
            Ok(InverseLink::Sas(*state))
        }
        (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => {
            let state = sas_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Integrated BinomialBetaLogistic update requires explicit SasLinkState"
                        .to_string(),
                )
            })?;
            Ok(InverseLink::BetaLogistic(*state))
        }
        (ResponseFamily::Binomial, InverseLink::Mixture(_)) => {
            let state = mixture_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Integrated BinomialMixture update requires explicit MixtureLinkState"
                        .to_string(),
                )
            })?;
            Ok(InverseLink::Mixture(state.clone()))
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "Integrated link-runtime update is not supported for likelihood (response={:?}, link={:?})",
            spec.response, spec.link
        ))),
    }
}


/// Updates Bernoulli-family GLM working vectors using an integrated
/// (uncertainty-aware) inverse-link runtime.
///
/// For the calibrator, we model:
///   μᵢ = E[σ(ηᵢ + ε)] where ε ~ N(0, SEᵢ²)
///
/// This integrates out uncertainty in the base prediction, giving a coherent
/// probabilistic treatment of measurement error. The effect is that steep
/// calibration adjustments are automatically attenuated when SE is high.
///
/// Uses the general IRLS formula (not canonical shortcut):
///   weight = prior × (dμ/dη)² / (μ(1-μ))
///   z = η + (y - μ) / (dμ/dη)
///
/// Derivation of the integrated quantities:
/// Let the uncertain latent predictor at row i be
///   eta_tilde_i = eta_i + eps_i,   eps_i ~ N(0, se_i^2).
/// Then the integrated mean used by PIRLS is
///   mu_i = E[g^{-1}(eta_tilde_i)].
/// Because the Gaussian family is a location family,
///   dmu_i / deta_i
///   = d/deta_i E[g^{-1}(eta_i + eps_i)]
///   = E[(g^{-1})'(eta_i + eps_i)].
/// That derivative is the exact object needed in the general GLM scoring update:
///   W_i = prior_i * (dmu_i/deta_i)^2 / Var(Y_i | mu_i),
///   z_i = eta_i + (y_i - mu_i) / (dmu_i/deta_i).
/// So any future exact link-specific replacement only needs to preserve the
/// contract
///   (eta_i, se_i) -> (mu_i, dmu_i/deta_i),
/// and the rest of the PIRLS machinery remains unchanged.
///
/// Why this matters for performance:
/// This helper runs inside the inner PIRLS loop, so any per-row integration cost
/// is multiplied by both the sample count and the number of IRLS iterations.
/// GHQ is robust, but it means repeated evaluation of quadrature nodes in a hot
/// path that can dominate calibrator or measurement-error fits.
///
/// Link-specific exact replacements:
/// - Probit:
///     E[Phi(eta + eps)] = Phi(eta / sqrt(1 + sigma^2))
///   exactly, with equally simple derivative. Integrated probit updates should
///   never need GHQ once they are routed through a dedicated family dispatch.
/// - Logit:
///   logistic-normal moments admit exact convergent Faddeeva / erfcx series,
///   which are the natural replacement for the GHQ calls below.
/// - Cloglog:
///   the mean is the complement of the lognormal Laplace transform and has
///   exact non-GHQ representations (Gamma / erfc / asymptotic series), which
///   is also relevant to survival transforms of the form exp(-exp(eta)).
///
/// This is the canonical integrated PIRLS update for binomial-style inverse
/// links. The runtime `InverseLink` carries the exact link state, so callers do
/// not have to thread `family + optional SAS/Mixture state` separately. Family
///-level integrated updates should reconstruct an `InverseLink` and delegate
/// here.
#[inline]
pub fn update_glmvectors_integrated_for_link(
    quadctx: &crate::quadrature::QuadratureContext,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    se: ArrayView1<f64>,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    let link = inverse_link.link_function();
    if !matches!(
        inverse_link,
        InverseLink::Standard(StandardLink::Logit)
            | InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    ) {
        crate::bail_invalid_estim!(
            "Integrated link-runtime update is not supported for inverse link {:?}",
            inverse_link
        );
    }
    if let Some(mut derivs) = derivatives {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        let WorkingDerivSlices {
            c: c_s,
            d: d_s,
            dmu: dmu_s,
            d2: d2_s,
            d3: d3_s,
        } = working_deriv_slices(&mut derivs);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .enumerate()
            .try_for_each(
                |(i, (((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o))|
                 -> Result<(), EstimationError> {
                    let jet = if let InverseLink::LatentCLogLog(state) = inverse_link {
                        crate::families::lognormal_kernel::latent_cloglog_inverse_link_jet(
                            quadctx,
                            eta[i],
                            se[i].hypot(state.latent_sd),
                        )?
                    } else if matches!(inverse_link, InverseLink::Standard(StandardLink::Logit)) {
                        crate::quadrature::integrated_logit_inverse_link_jet_pirls(
                            quadctx, eta[i], se[i],
                        )?
                    } else {
                        crate::quadrature::integrated_inverse_link_jetwith_state(
                            quadctx,
                            link,
                            eta[i],
                            se[i],
                            inverse_link.mixture_state(),
                            inverse_link.sas_state(),
                        )?
                    };
                    let local_jet = MixtureInverseLinkJet {
                        mu: jet.mean,
                        d1: jet.d1,
                        d2: jet.d2,
                        d3: jet.d3,
                    };
                    let e = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
                    let geom = bernoulli_geometry_from_jet(
                        eta[i],
                        e,
                        y[i],
                        priorweights[i],
                        local_jet,
                    );
                    *mu_o = geom.mu;
                    *w_o = geom.weight;
                    *z_o = geom.z;
                    *c_o = geom.c;
                    *d_o = geom.d;
                    *dmu_o = local_jet.d1;
                    *d2_o = local_jet.d2;
                    *d3_o = local_jet.d3;
                    Ok(())
                },
            )?;
    } else {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .try_for_each(|(i, ((mu_o, w_o), z_o))| -> Result<(), EstimationError> {
                let jet = if let InverseLink::LatentCLogLog(state) = inverse_link {
                    crate::families::lognormal_kernel::latent_cloglog_inverse_link_jet(
                        quadctx,
                        eta[i],
                        se[i].hypot(state.latent_sd),
                    )?
                } else if matches!(inverse_link, InverseLink::Standard(StandardLink::Logit)) {
                    crate::quadrature::integrated_logit_inverse_link_jet_pirls(
                        quadctx, eta[i], se[i],
                    )?
                } else {
                    crate::quadrature::integrated_inverse_link_jetwith_state(
                        quadctx,
                        link,
                        eta[i],
                        se[i],
                        inverse_link.mixture_state(),
                        inverse_link.sas_state(),
                    )?
                };
                let local_jet = MixtureInverseLinkJet {
                    mu: jet.mean,
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                };
                let e = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
                let geom = bernoulli_geometry_from_jet(eta[i], e, y[i], priorweights[i], local_jet);
                *mu_o = geom.mu;
                *w_o = geom.weight;
                *z_o = geom.z;
                Ok(())
            })?;
    }
    Ok(())
}


/// Family-dispatched integrated GLM vector update helper.
///
/// This is the adapter from structural likelihood families onto the canonical
/// link-runtime implementation above. It keeps existing family-based call sites
/// working while making the `InverseLink` path authoritative.
///
/// This remains the intended dispatch point for eliminating GHQ link-by-link:
/// - `BinomialProbit` uses the exact Gaussian-probit convolution identity,
/// - `BinomialLogit` uses the best validated exact/special-function path and
///   otherwise falls back,
/// - `BinomialCLogLog` uses the plug-in / Taylor / Miles / Gamma ladder.
///
/// The important architectural point is that each family-specific exact path
/// only needs to provide:
///   1. the integrated mean
///        mu_i = E[g^{-1}(eta_i + eps_i)]
///   2. the integrated derivative
///        dmu_i / deta_i = E[(g^{-1})'(eta_i + eps_i)].
/// Once those are available, the general IRLS weight and working-response
/// formulas above remain unchanged. That makes this dispatch site the natural
/// place to swap GHQ out for exact link-specific mathematics without touching
/// the rest of the PIRLS update logic.
///
/// Keeping the dispatch here avoids contaminating the general PIRLS machinery
/// with link-specific special-function code and lets each family choose the
/// mathematically correct integration strategy.
#[inline]
pub fn update_glmvectors_integrated_by_family(
    quadctx: &crate::quadrature::QuadratureContext,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    se: ArrayView1<f64>,
    spec: &LikelihoodSpec,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<(), EstimationError> {
    let inverse_link =
        integrated_inverse_link_from_family(spec, mixture_link_state, sas_link_state)?;
    update_glmvectors_integrated_for_link(
        quadctx,
        y,
        eta,
        se,
        &inverse_link,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    )
}


/// Compute first/second eta derivatives of the PIRLS working curvature W(eta),
/// consistent with the clamped working-geometry rules used by
/// `update_glmvectors`.
///
/// Math note:
/// - In the smooth interior (no clamps/floors active), `c[i]` and `d[i]` are
///   classical derivatives of the diagonal PIRLS curvature W_i(eta):
///     c_i = dW_i/dη_i,  d_i = d²W_i/dη_i².
/// - For canonical GLM families, these are the per-observation carriers of
///   higher likelihood derivatives (`-ℓ'''(η_i)` and `-ℓ''''(η_i)`) expressed
///   through the working-curvature map W(η).
/// - They are load-bearing in exact outer derivatives:
///   `c` enters dH/dρ (outer gradient), and `d` enters d²H/dρ² (outer Hessian).
/// - When hard clamps activate, the update map is piecewise and no longer C².
///   Setting c_i=d_i=0 is a practical subgradient-like choice to avoid unstable
///   explosive derivatives at the kink.
pub(crate) fn computeworkingweight_derivatives_from_eta(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
) -> Result<
    (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ),
    EstimationError,
> {
    let n = eta.len();
    let mut c = Array1::<f64>::zeros(n);
    let mut d = Array1::<f64>::zeros(n);
    let mut dmu_deta = Array1::<f64>::zeros(n);
    let mut d2mu_deta2 = Array1::<f64>::zeros(n);
    let mut d3mu_deta3 = Array1::<f64>::zeros(n);
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            dmu_deta.fill(1.0);
        }
        ResponseFamily::Poisson => {
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::PoissonIdentity,
                    curvature: log_link_working_state::WorkingCurvature::Proportional {
                        c_ratio: 1.0,
                        d_ratio: 1.0,
                    },
                    floor_weight: true,
                    zero_mu_jet_on_clamp: false,
                },
                inverse_link,
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) {
                crate::bail_invalid_estim!(
                    "Tweedie variance power must be finite and strictly between 1 and 2; got {p}",
                    p = p
                );
            }
            if !(phi.is_finite() && phi > 0.0) {
                crate::bail_invalid_estim!(
                    "Tweedie dispersion phi must be finite and > 0; got {phi}",
                    phi = phi
                );
            }
            let exponent = 2.0 - p;
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::TweediePower { p, phi },
                    curvature: log_link_working_state::WorkingCurvature::Proportional {
                        c_ratio: exponent,
                        d_ratio: exponent * exponent,
                    },
                    floor_weight: true,
                    zero_mu_jet_on_clamp: true,
                },
                inverse_link,
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            if !valid_negbin_theta(theta) {
                crate::bail_invalid_estim!(
                    "negative-binomial theta must be finite and > 0; got {theta}",
                    theta = theta
                );
            }
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::NegativeBinomial { theta },
                    curvature: log_link_working_state::WorkingCurvature::NegativeBinomial { theta },
                    floor_weight: true,
                    zero_mu_jet_on_clamp: false,
                },
                inverse_link,
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !valid_beta_phi(phi) {
                crate::bail_invalid_estim!("beta-regression phi must be finite and > 0; got {phi}");
            }
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                    let jet = logit_inverse_link_jet5(eta_i);
                    let mu_i = safe_beta_mu(jet.mu);
                    let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                    let a = (mu_i * phi).max(BETA_MU_EPS);
                    let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                    let trigamma_sum = trigamma(a) + trigamma(b);
                    let prior_weight = priorweights[i].max(0.0);
                    let raw_weight = prior_weight * q * q * phi * phi * trigamma_sum;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    if floor_active || eta_raw != eta_i {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        let (c_i, d_i) = beta_logit_working_curvature_eta_derivatives(
                            prior_weight,
                            phi,
                            mu_i,
                            q,
                            a,
                            b,
                            trigamma_sum,
                        );
                        *c_o = c_i;
                        *d_o = d_i;
                    }
                    *dmu_o = q;
                    *d2_o = q * (1.0 - 2.0 * mu_i);
                    *d3_o = q * (1.0 - 6.0 * q);
                });
        }
        ResponseFamily::Gamma => {
            // The Gamma log-link Fisher weight is independent of η, so the
            // working-curvature carriers `c`/`d` vanish identically (the kernel
            // returns `(0, 0)`); only the link jet is written here.
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::Constant { factor: 1.0 },
                    curvature: log_link_working_state::WorkingCurvature::Proportional {
                        c_ratio: 0.0,
                        d_ratio: 0.0,
                    },
                    floor_weight: false,
                    zero_mu_jet_on_clamp: false,
                },
                inverse_link,
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::Binomial => {
            let link = inverse_link.link_function();
            // On logit geometry, freeze higher η-derivatives in nonsmooth
            // regions so PIRLS and outer derivative code differentiate the
            // same piecewise-smooth surface.
            let zero_on_nonsmooth = matches!(link, LinkFunction::Logit);
            // Five independent per-row writes: same parallelization shape as
            // `update_glmvectors` above. Note the `jet.mu` argument is reused
            // here as the response (matching the original serial code) — this
            // is the score-derivative path where y is replaced by mu so the
            // (y - mu) residual term vanishes by construction.
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .try_for_each(
                    |(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| -> Result<(), EstimationError> {
                        let eta_used = match link {
                            LinkFunction::Logit => eta[i].clamp(-ETA_CLAMP, ETA_CLAMP),
                            LinkFunction::Probit
                            | LinkFunction::CLogLog
                            | LinkFunction::Sas
                            | LinkFunction::BetaLogistic => eta[i].clamp(-30.0, 30.0),
                            LinkFunction::Log => eta[i].clamp(-ETA_CLAMP, ETA_CLAMP),
                            LinkFunction::Identity => eta[i],
                        };
                        if matches!(link, LinkFunction::Logit) {
                            let jet = logit_inverse_link_jet5(eta_used);
                            let geom = bernoulli_logit_geometry_from_jet(
                                eta[i],
                                eta_used,
                                jet.mu,
                                priorweights[i],
                                jet,
                                zero_on_nonsmooth,
                            );
                            *c_o = geom.c;
                            *d_o = geom.d;
                            *dmu_o = jet.d1;
                            *d2_o = jet.d2;
                            *d3_o = jet.d3;
                        } else {
                            let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                            let geom = bernoulli_geometry_from_jet(
                                eta[i],
                                eta_used,
                                jet.mu,
                                priorweights[i],
                                jet,
                            );
                            *c_o = geom.c;
                            *d_o = geom.d;
                            *dmu_o = jet.d1;
                            *d2_o = jet.d2;
                            *d3_o = jet.d3;
                        }
                        Ok(())
                    },
                )?;
        }
        ResponseFamily::RoystonParmar => {
            crate::bail_invalid_estim!(
                "RoystonParmar is survival-specific and not a GLM IRLS family"
            );
        }
    }
    Ok((c, d, dmu_deta, d2mu_deta2, d3mu_deta3))
}


// General noncanonical observed-information weight corrections
//
// For an exponential-dispersion family with noncanonical link g, where
// h(η) = g⁻¹(η) is the inverse link and μ = h(η):
//
// Notation (all evaluated at a single observation):
//   h₁ = h'(η),  h₂ = h''(η),  h₃ = h'''(η),  h₄ = h''''(η)
//   V  = V(μ),   V₁ = V'(μ),   V₂ = V''(μ),    V₃ = V'''(μ)
//   φ  = dispersion parameter
//   pw = prior weight for this observation
//
// Fisher (expected) weight and its first two η-derivatives:
//   w_F = h₁² / (φV)
//   c_F = (2 h₁ h₂ V − h₁³ V₁) / (φ V²)
//   d_F = ∂c_F/∂η   (derived below)
//
// The observed weight subtracts a (y−μ)-dependent correction:
//   B   = (h₂ V − h₁² V₁) / (φ V²)
//   w_obs = w_F − (y−μ) · B
//
// First η-derivative of B:
//   B_η = (h₃ V² − 3 h₁ h₂ V V₁ − h₁³ V V₂ + 2 h₁³ V₁²) / (φ V³)
//
// Observed c (∂w_obs/∂η):
//   c_obs = c_F + h₁·B − (y−μ)·B_η
//
// Second η-derivative of B:
//   B_ηη = ∂B_η/∂η  (full expression in code below)
//
// Observed d (∂²w_obs/∂η²):
//   d_obs = d_F + h₂·B + 2 h₁·B_η − (y−μ)·B_ηη
//
// This function unifies all per-link hardcoded c/d computations: given the
// inverse-link jet (h₁…h₄) and the variance-function jet (V…V₃), it returns
// (w_obs, c_obs, d_obs) without any family- or link-specific dispatch.

/// Variance-function jet evaluated at μ: V(μ), V'(μ), V''(μ), V'''(μ), V''''(μ).
#[derive(Clone, Copy, Debug)]
pub struct VarianceJet {
    pub v: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
}


impl VarianceJet {
    /// Lower floor on μ before evaluating power-law variance functions, so that
    /// `μ^(p−k)` derivatives stay finite as μ → 0 instead of producing inf/NaN.
    const VARIANCE_MU_FLOOR: f64 = 1e-10;

    /// Bernoulli / binomial variance V(μ) = μ(1−μ).
    #[inline]
    pub fn bernoulli(mu: f64) -> Self {
        Self {
            v: mu * (1.0 - mu),
            v1: 1.0 - 2.0 * mu,
            v2: -2.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Poisson variance V(μ) = μ.
    #[inline]
    pub fn poisson(mu: f64) -> Self {
        Self {
            v: mu,
            v1: 1.0,
            v2: 0.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Gamma variance V(μ) = μ².
    #[inline]
    pub fn gamma(mu: f64) -> Self {
        Self {
            v: mu * mu,
            v1: 2.0 * mu,
            v2: 2.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Tweedie variance V(μ) = μ^p.
    #[inline]
    pub fn tweedie(mu: f64, p: f64) -> Self {
        let mu = mu.max(Self::VARIANCE_MU_FLOOR);
        Self {
            v: mu.powf(p),
            v1: p * mu.powf(p - 1.0),
            v2: p * (p - 1.0) * mu.powf(p - 2.0),
            v3: p * (p - 1.0) * (p - 2.0) * mu.powf(p - 3.0),
            v4: p * (p - 1.0) * (p - 2.0) * (p - 3.0) * mu.powf(p - 4.0),
        }
    }

    /// Negative-binomial variance V(μ) = μ + μ² / theta.
    #[inline]
    pub fn negative_binomial(mu: f64, theta: f64) -> Self {
        let mu = mu.max(Self::VARIANCE_MU_FLOOR);
        let inv_theta = if valid_negbin_theta(theta) {
            1.0 / theta
        } else {
            f64::NAN
        };
        Self {
            v: mu + mu * mu * inv_theta,
            v1: 1.0 + 2.0 * mu * inv_theta,
            v2: 2.0 * inv_theta,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Gaussian (identity) variance V(μ) = 1.
    #[inline]
    pub fn gaussian() -> Self {
        Self {
            v: 1.0,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
            v4: 0.0,
        }
    }

    /// Binomial(n, p) variance V(p) = p(1−p), identical to Bernoulli.
    ///
    /// The trial count `n` enters as a prior-weight multiplier, not through
    /// the variance function itself.
    #[inline]
    pub fn binomial_n(mu: f64) -> Self {
        // V(μ) = μ(1−μ), same jet as Bernoulli
        Self::bernoulli(mu)
    }

    /// Beta-regression variance V(μ) = μ(1−μ)/(1+φ).
    #[inline]
    pub fn beta(mu: f64, phi: f64) -> Self {
        let scale = 1.0 / (1.0 + phi.max(1e-12));
        let base = Self::bernoulli(mu);
        Self {
            v: base.v * scale,
            v1: base.v1 * scale,
            v2: base.v2 * scale,
            v3: 0.0,
            v4: 0.0,
        }
    }
}


const OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC: f64 = 1e-6;

const OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR: f64 = 1e-12;


/// Returns the per-row floor `max(fisher · 1e-6, 1e-12)` used by PIRLS to
/// stabilize the observed-information Hessian H = X' W X + S. Saturated
/// rows where W_obs ≤ floor were silently raised to `floor` when PIRLS
/// built the inner Hessian; outer REML/LAML derivatives must use the
/// **same** floored W to keep `H` and `dH/dψ` on one surface.
///
/// This is the single source of truth for the floor formula. Both the
/// inner solver (`solver_hessian_weights_into`) and the outer derivative
/// path (`outer_hessian_curvature_arrays`) route through this helper so
/// the inner-stabilized H and the outer dH/dψ cannot drift apart.
#[inline]
pub fn solver_hessian_weight_floor(fisher_weight: f64) -> f64 {
    (fisher_weight.max(0.0) * OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC)
        .max(OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR)
}


/// Build the (W, c, d) triple that matches PIRLS's stabilized H = X' W X + S.
///
/// PIRLS internally uses `W[i] = max(W_obs[i], floor(W_F[i]))` to keep H PD,
/// but `pirls_result.finalweights` stores the **unfloored** observed weights.
/// Reusing those directly in `∂H/∂ψ = X_τ' W X + … + X' diag(c · X_τ β̂) X`
/// produces an operator that disagrees with `H` at every saturated row — a
/// 5%-Frobenius bias that `tr(G_ε(H) · op)` amplifies by O(1/σ_min(H)),
/// driving the analytic gradient off by orders of magnitude.
///
/// This helper returns the floored W, plus c and d masked to zero wherever
/// the floor is active (so `∂W/∂η` is zero on the constant-floor branch).
pub fn outer_hessian_curvature_arrays(
    hessian_weights: crate::matrix::SignedWeightsView<'_>,
    fisher_weights: crate::matrix::PsdWeightsView<'_>,
    c_array: &Array1<f64>,
    d_array: &Array1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let hessian_view = hessian_weights.view();
    let fisher_view = fisher_weights.view();
    let n = hessian_view.len();
    let mut w_out = Array1::<f64>::zeros(n);
    let mut c_out = Array1::<f64>::zeros(n);
    let mut d_out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let floor = solver_hessian_weight_floor(fisher_view[i]);
        let w = hessian_view[i];
        let clamp_active = eta_clamp_active(inverse_link, eta[i]);
        let w_below_floor = !(w.is_finite() && w > floor);
        if w_below_floor {
            w_out[i] = floor;
            c_out[i] = 0.0;
            d_out[i] = 0.0;
        } else if clamp_active {
            w_out[i] = w;
            c_out[i] = 0.0;
            d_out[i] = 0.0;
        } else {
            w_out[i] = w;
            c_out[i] = c_array[i];
            d_out[i] = d_array[i];
        }
    }
    (w_out, c_out, d_out)
}


#[inline]
fn fixed_glm_dispersion(likelihood: &GlmLikelihoodSpec) -> f64 {
    likelihood.fixed_phi().unwrap_or(1.0)
}


#[inline]
pub fn weight_family_for_glm_likelihood(likelihood: &GlmLikelihoodSpec) -> WeightFamily {
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => WeightFamily::Gaussian,
        ResponseFamily::Poisson => WeightFamily::Poisson,
        ResponseFamily::Tweedie { p } => WeightFamily::Tweedie { p: *p },
        ResponseFamily::NegativeBinomial { theta, .. } => {
            WeightFamily::NegativeBinomial { theta: *theta }
        }
        ResponseFamily::Beta { phi } => WeightFamily::Beta { phi: *phi },
        ResponseFamily::Gamma => WeightFamily::Gamma,
        ResponseFamily::Binomial => WeightFamily::Binomial,
        ResponseFamily::RoystonParmar => WeightFamily::Gaussian,
    }
}


#[inline]
fn weight_link_for_inverse_link(inverse_link: &InverseLink) -> WeightLink {
    match inverse_link {
        InverseLink::Standard(StandardLink::Identity) => WeightLink::Identity,
        InverseLink::Standard(StandardLink::Log) => WeightLink::Log,
        InverseLink::Standard(StandardLink::Logit) => WeightLink::Logit,
        InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => WeightLink::Other,
    }
}


#[inline]
fn supports_observed_hessian_curvature_for_likelihood(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
) -> bool {
    let spec = &likelihood.spec;
    if matches!(spec.response, ResponseFamily::NegativeBinomial { .. }) {
        return matches!(inverse_link, InverseLink::Standard(StandardLink::Log));
    }
    if matches!(spec.response, ResponseFamily::Gamma) {
        return true;
    }
    if !matches!(spec.response, ResponseFamily::Binomial) {
        return false;
    }
    matches!(
        spec.link,
        InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
}


#[inline]
fn eta_for_observed_hessian_jet(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        // Why: canonical links keep V(mu) representable across the full f64 eta range; only guard against inf.
        InverseLink::Standard(StandardLink::Logit | StandardLink::Log) => {
            eta.clamp(-ETA_CLAMP, ETA_CLAMP)
        }
        InverseLink::Standard(StandardLink::Identity) => eta,
        // Why: probit mu=Phi(eta) saturates to 1.0 in f64 by |eta|~8.3; +/-6 keeps V=mu(1-mu) ~ 1e-9 representable.
        InverseLink::Standard(StandardLink::Probit) => eta.clamp(-6.0, 6.0),
        // Why: cloglog has mu~exp(eta) for eta<<0 (underflows below ~-23) and 1-mu~exp(-exp(eta)) collapses by eta=3.
        InverseLink::Standard(StandardLink::CLogLog) | InverseLink::LatentCLogLog(_) => {
            eta.clamp(-23.0, 3.0)
        }
        // Why: SAS / beta-logistic / mixture compose logistic-like sigmoids that saturate by |eta|~20 (logistic(20)~1-2e-9).
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) | InverseLink::Mixture(_) => {
            eta.clamp(-20.0, 20.0)
        }
    }
}


/// Returns true at rows where PIRLS clamped η (so the observed-info weights
/// were computed at the clamped value, making `∂W/∂η` zero w.r.t. the
/// **unclamped** η).  Outer REML/LAML derivative formulas must mask `c_obs`
/// and `d_obs` to zero on these rows or the analytic ∂H/∂ψ disagrees with
/// the H whose log-det we differentiate.
#[inline]
pub fn eta_clamp_active(inverse_link: &InverseLink, eta: f64) -> bool {
    let clamped = eta_for_observed_hessian_jet(inverse_link, eta);
    clamped != eta
}


/// Build solver-conditioned weights from the exact hessian weights.
///
/// The returned array applies a solver-only floor per observation so the
/// Newton linear system X'W X + S stays numerically usable. This floor is
/// purely a linear-algebra concern: the exact statistical weights stored in
/// `lasthessian_weights` / `finalweights` are not affected.
fn solver_hessian_weights_into(
    hessian_weights: &Array1<f64>,
    fisher_weights: &Array1<f64>,
    out: &mut Array1<f64>,
) {
    if out.len() != hessian_weights.len() {
        *out = Array1::<f64>::zeros(hessian_weights.len());
    }
    ndarray::Zip::from(out)
        .and(hessian_weights)
        .and(fisher_weights)
        .par_for_each(|o, &w, &fw| {
            let floor = solver_hessian_weight_floor(fw);
            *o = if w.is_finite() && w > floor { w } else { floor };
        });
}


/// Compute vectorised observed-information curvature arrays (w_obs, c_obs, d_obs)
/// for the Hessian surface at the mode.
///
/// This function is the primary entry point for obtaining the observed weights
/// that flow into the outer REML/LAML Hessian H_obs = X' W_obs X + S. The
/// observed corrections include residual-dependent terms that vanish for
/// canonical links but are nonzero for probit, cloglog, SAS, mixture, Gamma-log,
/// and other flexible links.
///
/// The output arrays are:
/// - `hessian_weights`: W_obs per observation (exact; solver floor applied separately).
/// - `hessian_c`: c_obs = dW_obs/deta per observation (for outer gradient C[v]).
/// - `hessian_d`: d_obs = d^2W_obs/deta^2 per observation (for outer Hessian Q[v_k,v_l]).
///
/// See `observed_weight_noncanonical` for the per-observation formulas and
/// response.md Section 3 for the mathematical justification of why observed
/// (not Fisher) information is required.
fn compute_observed_hessian_curvature_arrays_into(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    hessian_weights: &mut Array1<f64>,
    hessian_c: &mut Array1<f64>,
    hessian_d: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    assert!(supports_observed_hessian_curvature_for_likelihood(
        likelihood,
        inverse_link
    ));
    let n = eta.len();
    if hessian_weights.len() != n {
        *hessian_weights = Array1::<f64>::zeros(n);
    }
    if hessian_c.len() != n {
        *hessian_c = Array1::<f64>::zeros(n);
    }
    if hessian_d.len() != n {
        *hessian_d = Array1::<f64>::zeros(n);
    }

    let weight_family = weight_family_for_glm_likelihood(likelihood);
    let weight_link = weight_link_for_inverse_link(inverse_link);
    let phi = fixed_glm_dispersion(likelihood);

    // Parallel per-row weight assembly. At large scale (n = 320k) this loop
    // dominates non-canonical paths because each row independently evaluates
    // inverse-link jets and residual-dependent observed curvature. Write
    // directly into reusable output slices rather than collecting row tuples,
    // which removes an O(n) temporary allocation on every PIRLS update.
    hessian_weights
        .as_slice_mut()
        .expect("hessian weights must be contiguous")
        .par_iter_mut()
        .zip(
            hessian_c
                .as_slice_mut()
                .expect("hessian c must be contiguous")
                .par_iter_mut(),
        )
        .zip(
            hessian_d
                .as_slice_mut()
                .expect("hessian d must be contiguous")
                .par_iter_mut(),
        )
        .enumerate()
        .try_for_each(|(i, ((w_out, c_out), d_out))| -> Result<(), EstimationError> {
            let eta_used = eta_for_observed_hessian_jet(inverse_link, eta[i]);
            // Why: closed-form observed_weight_noncanonical requires (mu, d1..d3, h4) at one consistent eta;
            // mixing PIRLS-state jets at unclamped eta with h4 at eta_used produced 0/0 in phi_v* divisions,
            // surfacing as: "observed Hessian curvature is not positive finite at row N: observed=NaN, fisher=0".
            let jet =
                crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta_used)?;
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                inverse_link, eta_used,
            )?;
            let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
                weight_family,
                weight_link,
                eta_used,
                y[i],
                jet.mu,
                phi,
                priorweights[i].max(0.0),
                jet,
                h4,
            );
            let fisher_weight = fisher_weights[i].max(0.0);
            if !(w_obs.is_finite() && w_obs > 0.0) {
                crate::bail_invalid_estim!(
                    "observed Hessian curvature is not positive finite at row {i}: observed={w_obs}, fisher={fisher_weight}"
                );
            }
            if !c_obs.is_finite() || !d_obs.is_finite() {
                crate::bail_invalid_estim!(
                    "observed Hessian curvature derivatives are non-finite at row {i}: c={c_obs}, d={d_obs}"
                );
            }
            *w_out = w_obs;
            *c_out = c_obs;
            *d_out = d_obs;
            Ok(())
        })
}


pub(crate) fn compute_observed_hessian_curvature_arrays(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
    let n = eta.len();
    let mut hessian_weights = Array1::<f64>::zeros(n);
    let mut hessian_c = Array1::<f64>::zeros(n);
    let mut hessian_d = Array1::<f64>::zeros(n);
    compute_observed_hessian_curvature_arrays_into(
        likelihood,
        inverse_link,
        eta,
        y,
        fisher_weights,
        priorweights,
        &mut hessian_weights,
        &mut hessian_c,
        &mut hessian_d,
    )?;
    Ok((hessian_weights, hessian_c, hessian_d))
}


/// Per-observation observed-information weights and their first two
/// eta-derivatives for a general exponential-dispersion family with a
/// noncanonical link.
///
/// The observed weight differs from the Fisher (expected) weight by a
/// residual-dependent correction (see response.md Section 3):
///
///   W_obs = W_Fisher - (y - mu) * B
///   B = (h'' V - h'^2 V') / (phi V^2)
///
///   c_obs = c_Fisher + h' * B - (y - mu) * B_eta
///   d_obs = d_Fisher + h'' * B + 2*h' * B_eta - (y - mu) * B_etaeta
///
/// For canonical links (for example logit-Binomial and log-Poisson), B = 0
/// so observed = Fisher and no correction is needed.
///
/// These observed quantities are required for:
/// 1. The outer REML/LAML Hessian H_obs = X' W_obs X + S (log|H| term).
/// 2. The outer gradient's C[v] correction (uses c_obs).
/// 3. The outer Hessian's Q[v_k, v_l] correction (uses d_obs).
///
/// Using Fisher weights in the outer REML would yield a PQL-type surrogate
/// rather than the exact Laplace approximation.
///
/// # Arguments
/// * `y`   -- response value
/// * `mu`  -- fitted mean h(eta)
/// * `h1`...`h4` -- inverse-link derivatives h'(eta) ... h''''(eta)
/// * `vj`  -- variance-function jet (V, V', V'', V''') evaluated at mu
/// * `phi` -- dispersion parameter (1.0 for Bernoulli/Poisson)
/// * `pw`  -- prior weight for this observation
///
/// # Returns
/// `(w_obs, c_obs, d_obs)` -- the observed weight and its first two
/// eta-derivatives, all pre-multiplied by `pw`.
#[inline]
pub fn observed_weight_noncanonical(
    y: f64,
    mu: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> (f64, f64, f64) {
    let VarianceJet {
        v,
        v1,
        v2,
        v3,
        v4: _,
    } = vj;
    let phi_v = phi * v;
    let phi_v2 = phi * v * v;
    let phi_v3 = phi * v * v * v;

    // ---- Fisher weight and derivatives ----
    let h1_sq = h1 * h1;
    let w_f = h1_sq / phi_v;

    // c_F = (2 h₁ h₂ V − h₁³ V₁) / (φ V²)
    let n0 = h1_sq; // numerator of w_F
    let n1 = 2.0 * h1 * h2; // ∂(h₁²)/∂η
    let n2 = 2.0 * (h2 * h2 + h1 * h3); // ∂²(h₁²)/∂η²
    let vd1 = h1 * v1; // ∂V/∂η = V'·h'
    let vd2 = h2 * v1 + h1_sq * v2; // ∂²V/∂η²

    let c_f = (n1 * v - n0 * vd1) / phi_v2;

    // d_F = ∂c_F/∂η via quotient rule on c_F = (n1·v − n0·vd1) / (φ·v²)
    // numerator of c_F and its η-derivative (cross terms cancel):
    let numer_cf = n1 * v - n0 * vd1;
    let dnumer_cf = n2 * v - n0 * vd2;
    let d_f = (dnumer_cf * v - 2.0 * numer_cf * vd1) / (phi_v3);

    // ---- Observed correction term B and its η-derivatives ----
    // B = (h₂ V − h₁² V₁) / (φ V²)
    let b_num = h2 * v - h1_sq * v1;
    let b = b_num / phi_v2;

    // B_η = (h₃ V² − 3 h₁ h₂ V V₁ − h₁³ V V₂ + 2 h₁³ V₁²) / (φ V³)
    let b_eta_num =
        h3 * v * v - 3.0 * h1 * h2 * v * v1 - h1_sq * h1 * v * v2 + 2.0 * h1_sq * h1 * v1 * v1;
    let b_eta = b_eta_num / phi_v3;

    // B_ηη = ∂B_η/∂η.
    //
    // We differentiate b_eta_num / (φ V³) using the quotient rule.
    //
    // Numerator derivative of b_eta_num w.r.t. η, using chain rule ∂/∂η = h₁·∂/∂μ
    // for the V-dependent parts:
    //
    //   ∂/∂η [h₃ V²]               = h₄ V² + 2 h₃ V h₁ V₁
    //   ∂/∂η [3 h₁ h₂ V V₁]        = 3(h₂² + h₁ h₃)V V₁ + 3 h₁ h₂(h₁ V₁² + V h₁ V₂)
    //   ∂/∂η [h₁³ V V₂]            = 3 h₁² h₂ V V₂ + h₁³(h₁ V₁ V₂ + V h₁ V₃)
    //   ∂/∂η [2 h₁³ V₁²]           = 6 h₁² h₂ V₁² + 4 h₁³ V₁ h₁ V₂
    //                                = 6 h₁² h₂ V₁² + 4 h1_sq * h1_sq * v1 * v2
    //
    // Denominator derivative: ∂/∂η [φ V³] = 3 φ V² h₁ V₁.

    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    let db_eta_num = h4 * v * v + 2.0 * h3 * v * h1 * v1
        - 3.0 * (h2 * h2 + h1 * h3) * v * v1
        - 3.0 * h1 * h2 * (h1 * v1 * v1 + v * h1 * v2)
        - 3.0 * h1_sq * h2 * v * v2
        - h1_cu * (h1 * v1 * v2 + v * h1 * v3)
        + 6.0 * h1_sq * h2 * v1 * v1
        + 4.0 * h1_qu * v1 * v2;

    let phi_v4 = phi_v3 * v;
    let b_etaeta = (db_eta_num * v - 3.0 * b_eta_num * h1 * v1) / phi_v4;

    // ---- Assemble observed quantities ----
    let resid = y - mu;

    let w_obs = w_f - resid * b;
    let c_obs = c_f + h1 * b - resid * b_eta;
    let d_obs = d_f + h2 * b + 2.0 * h1 * b_eta - resid * b_etaeta;

    (pw * w_obs, pw * c_obs, pw * d_obs)
}


/// Per-observation third η-derivative of the observed-information weight,
/// `e_obs := ∂³W_obs/∂η³`, for a general exponential-dispersion family with
/// any (canonical or non-canonical) link.
///
/// Closed-form derivation:
///   Define `T(η) := h₁(η)/(φ V(μ(η)))`. Then
///   * Fisher weight `W_F = h₁ · T`
///   * Observed correction `B = T'`, so `B_η = T''`, `B_ηη = T'''`,
///     `B_ηηη = T''''`
///   * `W_obs = W_F − (y−μ) · T'`
///
/// Differentiating three times:
///   `∂³W_obs/∂η³ = W_F''' + h₃·T' + 3 h₂·T'' + 3 h₁·T''' − (y−μ)·T''''`
///
/// `T` is computed via Leibniz on `T·Q = h₁` with `Q = φV`; `W_F` via
/// Leibniz on `W_F·1 = h₁·T` (product rule).
///
/// All inverse-link derivatives `h₁..h₅` and variance-function derivatives
/// `V..V₄` are required as inputs. Caller supplies them.
///
/// Returns `pw * e_obs` (pre-multiplied by the prior weight) so the result
/// scales identically to `(w_obs, c_obs, d_obs)` from
/// `observed_weight_noncanonical`.
#[inline]
pub fn e_obs_from_jets(
    y: f64,
    mu: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    h5: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> f64 {
    let VarianceJet { v, v1, v2, v3, v4 } = vj;
    let q = phi * v;

    // Q = φV and its η-derivatives.
    //   Q'    = φ V₁ h₁
    //   Q''   = φ (V₁ h₂ + V₂ h₁²)
    //   Q'''  = φ (V₁ h₃ + 3 V₂ h₁ h₂ + V₃ h₁³)
    //   Q'''' = φ (V₁ h₄ + 4 V₂ h₁ h₃ + 3 V₂ h₂² + 6 V₃ h₁² h₂ + V₄ h₁⁴)
    let h1_sq = h1 * h1;
    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    let q1 = phi * v1 * h1;
    let q2 = phi * (v1 * h2 + v2 * h1_sq);
    let q3 = phi * (v1 * h3 + 3.0 * v2 * h1 * h2 + v3 * h1_cu);
    let q4 = phi
        * (v1 * h4 + 4.0 * v2 * h1 * h3 + 3.0 * v2 * h2 * h2 + 6.0 * v3 * h1_sq * h2 + v4 * h1_qu);

    // T = h₁/Q and T', T'', T''', T'''' via Leibniz on T·Q = h₁.
    //   T'    = (h₂  − T·Q')/Q
    //   T''   = (h₃  − 2 T'·Q' − T·Q'')/Q
    //   T'''  = (h₄  − 3 T''·Q' − 3 T'·Q'' − T·Q''')/Q
    //   T'''' = (h₅  − 4 T'''·Q' − 6 T''·Q'' − 4 T'·Q''' − T·Q'''')/Q
    let t0 = h1 / q;
    let t1 = (h2 - t0 * q1) / q;
    let t2 = (h3 - 2.0 * t1 * q1 - t0 * q2) / q;
    let t3 = (h4 - 3.0 * t2 * q1 - 3.0 * t1 * q2 - t0 * q3) / q;
    let t4 = (h5 - 4.0 * t3 * q1 - 6.0 * t2 * q2 - 4.0 * t1 * q3 - t0 * q4) / q;

    // Fisher weight derivatives via product rule on W_F = h₁·T.
    //   W_F^(0) = h₁ T
    //   W_F^(1) = h₁ T₁ + h₂ T
    //   W_F^(2) = h₁ T₂ + 2 h₂ T₁ + h₃ T
    //   W_F^(3) = h₁ T₃ + 3 h₂ T₂ + 3 h₃ T₁ + h₄ T
    let w_f3 = h1 * t3 + 3.0 * h2 * t2 + 3.0 * h3 * t1 + h4 * t0;

    // Observed third derivative: differentiate W_obs = W_F − (y−μ)·T₁ thrice.
    // (resid)' = −h₁, so iterating product rule yields
    //   ∂³((y−μ)·T₁)/∂η³ = −h₃·T₁ − 3 h₂·T₂ − 3 h₁·T₃ + (y−μ)·T₄
    let resid = y - mu;
    let e_obs = w_f3 + h3 * t1 + 3.0 * h2 * t2 + 3.0 * h1 * t3 - resid * t4;

    pw * e_obs
}


// Direct (closed-form) observed-information weights for specific family-link
// combinations.  These avoid the overhead of the generic noncanonical formula
// when the algebra simplifies.

/// Gaussian family with log link: y ~ N(μ, φ), μ = exp(η).
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight `pw`.
///
/// ```text
/// w_obs = ω μ(2μ − y) / φ
/// c_obs = ω μ(4μ − y) / φ
/// d_obs = ω μ(8μ − y) / φ
/// ```
#[inline]
pub fn observed_weight_gaussian_log(y: f64, mu: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let inv_phi = pw / phi;
    let w = inv_phi * mu * (2.0 * mu - y);
    let c = inv_phi * mu * (4.0 * mu - y);
    let d = inv_phi * mu * (8.0 * mu - y);
    (w, c, d)
}


/// Gaussian family with inverse link: y ~ N(μ, φ), μ = 1/η.
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight `pw`.
///
/// ```text
/// w_obs = ω (3 − 2ηy) / (φ η⁴)
/// c_obs = 6ω (ηy − 2) / (φ η⁵)
/// d_obs = 12ω (5 − 2ηy) / (φ η⁶)
/// ```
#[inline]
pub fn observed_weight_gaussian_inverse(y: f64, eta: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let eta2 = eta * eta;
    let eta4 = eta2 * eta2;
    let eta5 = eta4 * eta;
    let eta6 = eta4 * eta2;
    let ey = eta * y;
    let inv_phi = pw / phi;
    let w = inv_phi * (3.0 - 2.0 * ey) / eta4;
    let c = inv_phi * 6.0 * (ey - 2.0) / eta5;
    let d = inv_phi * 12.0 * (5.0 - 2.0 * ey) / eta6;
    (w, c, d)
}


#[inline]
fn observed_weight_binomial_logit_from_jet(
    n_trials: f64,
    jet: MixtureInverseLinkJet,
    pw: f64,
) -> (f64, f64, f64) {
    let scale = pw * n_trials;
    (scale * jet.d1, scale * jet.d2, scale * jet.d3)
}


/// Family tag for the observed-information weight dispatch.
///
/// This is a simplified family tag that identifies the variance function,
/// independent of the link function. It is used by [`observed_weight_dispatch`]
/// to select closed-form weight specializations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFamily {
    Gaussian,
    Binomial,
    Poisson,
    Tweedie { p: f64 },
    NegativeBinomial { theta: f64 },
    Beta { phi: f64 },
    Gamma,
}


/// Link tag for the observed-information weight dispatch.
///
/// Identifies the link function for selecting closed-form weight
/// specializations in [`observed_weight_dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLink {
    Identity,
    Log,
    Logit,
    Inverse,
    /// Any other link — falls back to the generic noncanonical formula.
    Other,
}


#[inline]
pub fn variance_jet_for_weight_family(family: WeightFamily, mu: f64) -> VarianceJet {
    match family {
        WeightFamily::Gaussian => VarianceJet::gaussian(),
        WeightFamily::Binomial => VarianceJet::binomial_n(mu),
        WeightFamily::Poisson => VarianceJet::poisson(mu),
        WeightFamily::Tweedie { p } => VarianceJet::tweedie(mu, p),
        WeightFamily::NegativeBinomial { theta } => VarianceJet::negative_binomial(mu, theta),
        WeightFamily::Beta { phi } => VarianceJet::beta(mu, phi),
        WeightFamily::Gamma => VarianceJet::gamma(mu),
    }
}


/// Dispatch to closed-form observed-information weights for known family-link
/// combinations, falling back to the generic noncanonical formula.
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight.
///
/// For the `Binomial + Logit` case, `n_trials` is passed as `phi` (dispersion
/// slot is unused for binomial) and the prior weight controls the
/// observation-level scaling. For all other cases, `phi` is the dispersion
/// parameter.
///
/// `jet` and `h4` are the inverse-link derivatives used by the generic
/// noncanonical fallback path. They may be zero for the specialized paths.
pub fn observed_weight_dispatch(
    family: WeightFamily,
    link: WeightLink,
    eta: f64,
    y: f64,
    mu: f64,
    phi: f64,
    prior_weight: f64,
    jet: MixtureInverseLinkJet,
    h4: f64,
) -> (f64, f64, f64) {
    match (family, link) {
        (WeightFamily::Gaussian, WeightLink::Log) => {
            observed_weight_gaussian_log(y, mu, phi, prior_weight)
        }
        (WeightFamily::Gaussian, WeightLink::Inverse) => {
            observed_weight_gaussian_inverse(y, eta, phi, prior_weight)
        }
        (WeightFamily::Binomial, WeightLink::Logit) => {
            observed_weight_binomial_logit_from_jet(1.0, jet, prior_weight)
        }
        _ => {
            // Generic noncanonical path via the full variance-function jet.
            let vj = variance_jet_for_weight_family(family, mu);
            observed_weight_noncanonical(y, mu, jet.d1, jet.d2, jet.d3, h4, vj, phi, prior_weight)
        }
    }
}


#[derive(Clone)]
pub enum DirectionalWorkingCurvature {
    /// Directional derivative of the PIRLS curvature when the working
    /// curvature is diagonal in observation space:
    ///   W_τ = diag(w_τ).
    Diagonal(Array1<f64>),
}


pub fn directionalworking_curvature_from_c_array(
    c_array: &Array1<f64>,
    hessian_weights: &Array1<f64>,
    eta_direction: &Array1<f64>,
) -> DirectionalWorkingCurvature {
    let mut w_direction = c_array * eta_direction;
    for i in 0..w_direction.len() {
        if hessian_weights[i] <= 0.0 || !w_direction[i].is_finite() {
            w_direction[i] = 0.0;
        }
    }
    DirectionalWorkingCurvature::Diagonal(w_direction)
}


/// Floor/ceiling for binomial mu before taking `ln(mu)` / `ln(1 - mu)`.
/// Matches the precedent in families/lognormal_kernel.rs (1e-12) so that
/// saturating inverse links (probit, cloglog, logit at large |eta|) cannot
/// produce -inf in the deviance or log-likelihood reductions.
const BINOMIAL_MU_EPS: f64 = 1e-12;


/// Clamp `mu` away from 0 and 1 so `mu.ln()` and `(1 - mu).ln()` are finite.
/// Centralized to keep deviance and log-likelihood symmetric — both must use
/// the same floor or the log-lik / deviance identity drifts near saturation.
#[inline]
fn safe_mu_for_binomial(mu: f64) -> f64 {
    mu.clamp(BINOMIAL_MU_EPS, 1.0 - BINOMIAL_MU_EPS)
}


#[inline]
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}


#[inline]
fn log_gamma_stirling_correction(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    inv / 12.0 - inv * inv2 / 360.0 + inv * inv2 * inv2 / 1260.0
}


#[inline]
fn log_gamma_large_ratio(base: f64, delta: f64) -> f64 {
    let ratio = delta / base;
    delta * base.ln() + (base + delta - 0.5) * ratio.ln_1p() - delta
        + log_gamma_stirling_correction(base + delta)
        - log_gamma_stirling_correction(base)
}


#[inline]
fn beta_log_normalizer(a: f64, b: f64, sum: f64) -> f64 {
    let direct = ln_gamma(sum) - ln_gamma(a) - ln_gamma(b);
    if direct.is_finite() {
        return direct;
    }
    let small = a.min(b);
    let large = a.max(b);
    if small < 8.0 {
        return log_gamma_large_ratio(large, small) - ln_gamma(small);
    }
    -xlogy(a, a / sum) - xlogy(b, b / sum)
        + 0.5 * (a.ln() + b.ln() - sum.ln() - (2.0 * std::f64::consts::PI).ln())
        + log_gamma_stirling_correction(sum)
        - log_gamma_stirling_correction(a)
        - log_gamma_stirling_correction(b)
}


#[inline]
fn poisson_unit_deviance(yi: f64, mui_c: f64) -> f64 {
    xlogy(yi, yi / mui_c) - (yi - mui_c)
}


#[inline]
fn gamma_unit_deviance(yi_c: f64, mui_c: f64) -> f64 {
    let ratio = yi_c / mui_c;
    ratio - 1.0 - ratio.ln()
}


#[inline]
fn tweedie_unit_deviance(yi: f64, mui_c: f64, p: f64) -> f64 {
    if !is_valid_tweedie_power(p) {
        f64::NAN
    } else if !valid_tweedie_response(yi) {
        f64::NAN
    } else if yi == 0.0 {
        mui_c.powf(2.0 - p) / (2.0 - p)
    } else {
        yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - yi * mui_c.powf(1.0 - p) / (1.0 - p)
            + mui_c.powf(2.0 - p) / (2.0 - p)
    }
}


#[inline]
fn negative_binomial_unit_deviance(yi: f64, mui_c: f64, theta: f64) -> f64 {
    if !valid_negbin_theta(theta) || !valid_count_response(yi) {
        return f64::NAN;
    }
    let y_term = xlogy(yi, (yi * (theta + mui_c)) / (mui_c * (theta + yi)));
    let theta_term = theta * ((theta + mui_c) / (theta + yi)).ln();
    theta_term + y_term
}


#[inline]
fn beta_loglikelihood_full_unit(yi: f64, mui: f64, phi: f64) -> f64 {
    if !valid_beta_phi(phi) || !valid_beta_response(yi) {
        return f64::NAN;
    }
    let mui_c = safe_beta_mu(mui);
    let a = (mui_c * phi).max(BETA_MU_EPS);
    let b = ((1.0 - mui_c) * phi).max(BETA_MU_EPS);
    beta_log_normalizer(a, b, phi) + phi * xlogy(mui_c, yi) + phi * xlogy(1.0 - mui_c, 1.0 - yi)
        - yi.ln()
        - (1.0 - yi).ln()
}


#[inline]
fn beta_unit_deviance(yi: f64, mui: f64, phi: f64) -> f64 {
    if !valid_beta_response(yi) {
        return f64::NAN;
    }
    beta_loglikelihood_full_unit(yi, yi, phi) - beta_loglikelihood_full_unit(yi, mui, phi)
}


#[inline]
pub fn calculate_deviance(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    const EPS: f64 = 1e-8;
    // Match the μ floor used by the shared PIRLS log-link working-state engine
    // (`MIN_MU = 1e-10` in `log_link_working_state`) so deviance / weights
    // stay self-consistent when the linear predictor saturates.
    const MU_FLOOR: f64 = 1e-10;
    match &likelihood.spec.response {
        ResponseFamily::Binomial => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total_residual: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    // Inverse links (probit, cloglog, logit) can saturate to
                    // exactly 0 or 1 in finite precision; clamp before ln so
                    // the deviance sum stays finite. Uses the same floor as
                    // the log-likelihood site below to keep the two reductions
                    // self-consistent.
                    let mui_c = safe_mu_for_binomial(mu[i]);
                    let wi = priorweights[i];
                    let term1 = if yi > EPS {
                        yi * (yi.ln() - mui_c.ln())
                    } else {
                        0.0
                    };
                    let term2 = if yi < 1.0 - EPS {
                        (1.0 - yi) * ((1.0 - yi).ln() - (1.0 - mui_c).ln())
                    } else {
                        0.0
                    };
                    wi * (term1 + term2)
                })
                .sum();
            2.0 * total_residual
        }
        ResponseFamily::Gaussian => {
            // Scaled Gaussian deviance is sum(prior_i * (y_i - mu_i)^2 / phi).
            // The default `ProfiledGaussian` metadata reports no fixed phi and
            // we keep the historical unscaled form (phi == 1) so that profiled
            // sigma fits remain unchanged. When the caller fixes phi explicitly
            // we divide by it so the deviance lines up with the IRLS working
            // weights (`prior_i / phi`) and with the canonical exponential-
            // family scaled deviance used elsewhere.
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            if !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            let raw: f64 = ndarray::Zip::from(y)
                .and(mu)
                .and(priorweights)
                .map_collect(|&yi, &mui, &wi| wi * (yi - mui) * (yi - mui))
                .sum();
            raw / phi
        }
        ResponseFamily::Poisson => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * poisson_unit_deviance(yi, mui_c)
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            if validate_tweedie_responses(&y, &priorweights).is_err() {
                return f64::NAN;
            }
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * tweedie_unit_deviance(yi, mui_c, p) / phi
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * negative_binomial_unit_deviance(yi, mui_c, theta)
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !valid_beta_phi(phi) {
                return f64::NAN;
            }
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| priorweights[i] * beta_unit_deviance(y[i], mu[i], phi))
                .sum();
            2.0 * total
        }
        ResponseFamily::Gamma => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * shape * gamma_unit_deviance(yi_c, mui_c)
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::RoystonParmar => f64::NAN,
    }
}


#[inline]
/// Per-observation log-likelihood (with the same family-specific constants
/// dropped as [`calculate_loglikelihood_omitting_constants`]) evaluated at the
/// supplied fitted means `mu`.
///
/// This is the single source of truth for the per-row likelihood kernel: the
/// scalar aggregate sums this vector, and the model-comparison machinery
/// (`crate::inference::model_comparison`) evaluates it at ALO-corrected means
/// to form pointwise predictive densities for PSIS-LOO. Because the same
/// family-independent constants are omitted in every evaluation, the dropped
/// constants cancel exactly in any *difference* of log-likelihoods — paired
/// Δelpd between two fits on the same response, and the self-normalized PSIS
/// importance ratios — so the omission is harmless for comparison channels.
///
/// For the deviance-parameterized families (Tweedie, Gamma) the per-row value
/// is `-0.5 ·` the per-row scaled unit deviance, matching the aggregate exactly
/// row by row.
pub fn pointwise_loglikelihood_omitting_constants(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> Array1<f64> {
    // Same μ floor as PIRLS log-link working-state writers; see note in
    // `calculate_deviance` above.
    const MU_FLOOR: f64 = 1e-10;
    const EPS: f64 = 1e-8;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    let values: Vec<f64> = match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            // Gaussian log-likelihood (constants dropped) is
            //     -0.5 * prior_i * (y_i - mu_i)^2 / phi.
            // `ProfiledGaussian` returns no fixed phi and falls back to phi=1,
            // preserving the historical profiled-sigma behaviour. A caller that
            // fixes phi gets the scaled form that matches the IRLS weights and
            // the scaled deviance in `calculate_deviance`.
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            if !(phi.is_finite() && phi > 0.0) {
                return Array1::from_elem(n, f64::NAN);
            }
            let inv_phi = 1.0 / phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let resid = y[i] - mu[i];
                    -0.5 * priorweights[i] * resid * resid * inv_phi
                })
                .collect()
        }
        ResponseFamily::Binomial => (0..n)
            .into_par_iter()
            .map(|i| {
                // Share the deviance helper so both reductions floor mu at
                // the same epsilon — otherwise the deviance / log-lik identity
                // drifts whenever the link saturates.
                let mui_c = safe_mu_for_binomial(mu[i]);
                priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .collect(),
        ResponseFamily::Poisson => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i].max(MU_FLOOR);
                let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
                priorweights[i] * (log_term - mui_c)
            })
            .collect(),
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return Array1::from_elem(n, f64::NAN);
            }
            if validate_tweedie_responses(&y, &priorweights).is_err() {
                return Array1::from_elem(n, f64::NAN);
            }
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    -priorweights[i] * tweedie_unit_deviance(yi, mui_c, p) / phi
                })
                .collect()
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_negbin_theta(theta) {
                        return f64::NAN;
                    }
                    let yi = y[i];
                    if !valid_count_response(yi) {
                        return f64::NAN;
                    }
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i]
                        * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                            + theta * (theta.ln() - (theta + mui_c).ln())
                            + xlogy(yi, mui_c)
                            - yi * (theta + mui_c).ln())
                })
                .collect()
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_beta_phi(phi) {
                        return f64::NAN;
                    }
                    priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
                })
                .collect()
        }
        ResponseFamily::Gamma => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(MU_FLOOR);
                    -priorweights[i] * shape * gamma_unit_deviance(yi_c, mui_c)
                })
                .collect()
        }
        ResponseFamily::RoystonParmar => vec![f64::NAN; n],
    };
    Array1::from_vec(values)
}


pub(crate) fn calculate_loglikelihood_omitting_constants(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    // Same μ floor as PIRLS log-link working-state writers; see note in
    // `calculate_deviance` above.
    const MU_FLOOR: f64 = 1e-10;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            // Gaussian log-likelihood (constants dropped) is
            //     -0.5 * prior_i * (y_i - mu_i)^2 / phi.
            // `ProfiledGaussian` returns no fixed phi and falls back to phi=1,
            // preserving the historical profiled-sigma behaviour. A caller that
            // fixes phi gets the scaled form that matches the IRLS weights and
            // the scaled deviance in `calculate_deviance`.
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            if !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            let inv_phi = 1.0 / phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let resid = y[i] - mu[i];
                    -0.5 * priorweights[i] * resid * resid * inv_phi
                })
                .sum()
        }
        ResponseFamily::Binomial => (0..n)
            .into_par_iter()
            .map(|i| {
                // Share the deviance helper so both reductions floor mu at
                // the same epsilon — otherwise the deviance / log-lik identity
                // drifts whenever the link saturates.
                let mui_c = safe_mu_for_binomial(mu[i]);
                priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .sum(),
        ResponseFamily::Poisson => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i].max(MU_FLOOR);
                let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
                priorweights[i] * (log_term - mui_c)
            })
            .sum(),
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            -0.5 * calculate_deviance(y, mu, likelihood, priorweights)
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_negbin_theta(theta) {
                        return f64::NAN;
                    }
                    let yi = y[i];
                    if !valid_count_response(yi) {
                        return f64::NAN;
                    }
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i]
                        * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                            + theta * (theta.ln() - (theta + mui_c).ln())
                            + xlogy(yi, mui_c)
                            - yi * (theta + mui_c).ln())
                })
                .sum()
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_beta_phi(phi) {
                        return f64::NAN;
                    }
                    priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
                })
                .sum()
        }
        ResponseFamily::Gamma => {
            // REML/LAML outer objective: use the scaled-deviance form
            //   ℓ = −½ D(y, μ) = −Σ wᵢ · shape · d(yᵢ, μᵢ)
            // (with `shape = 1/φ` folded into the deviance), exactly as the
            // Tweedie branch above. This is the mgcv convention: the outer
            // objective only needs the β-dependent part of the log-likelihood
            // plus the penalty/log-determinant terms; the saturated-likelihood
            // normalizing constants `shape·ln(shape) − lnΓ(shape) − shape − ln y`
            // are independent of β (hence of the outer derivative under the
            // fixed-dispersion handling Gamma is routed through) and are
            // intentionally dropped.
            //
            // Using the full saturated form here is what made the Gamma outer
            // cost non-finite: the per-iterate shape estimate saturates to
            // `GAMMA_SHAPE_MAX = 1e12` whenever the working fit drives the unit
            // deviance toward zero (the common high-dispersion / CV≈1 case),
            // and `shape·ln(shape) − lnΓ(shape)` evaluated at 1e12 across n rows
            // overflows. The scaled-deviance form carries no such term: the
            // bounded unit deviance keeps the product `shape · d(y, μ)` finite
            // even as the shape grows, so the seed screen no longer rejects
            // every ρ candidate. See issue #359.
            -0.5 * calculate_deviance(y, mu, likelihood, priorweights)
        }
        ResponseFamily::RoystonParmar => f64::NAN,
    }
}


// ---------------------------------------------------------------------------
// Piece 5: structured low-rank weight in the inner solve.
//
// External Fisher-Rao / behavioral metrics arrive shaped as `W = D + U Vᵀ`
// with `U, V` tall-skinny (rank r ≪ n). These siblings to the diagonal-W
// PIRLS kernels add the rank-r correction without touching the existing
// `compute_xtwx_blas` / `penalized_hessian` call sites used by Piece 1's
// Newton-direction hooks. The metric is supplied by the caller; this
// module never estimates a covariance internally.
//
// Composition with the existing signed-Gram API:
// - The diagonal part flows through `xt_diag_x_signed` / `xt_diag_x_psd`
//   exactly as before. When `LowRankWeight::is_rank_zero()` the path is
//   bit-identical to the legacy diagonal flow.
// - The low-rank correction is `(XᵀU)(VᵀX)`, a `p × p` outer product of
//   tall-skinny projections — dimension `p × p`, never `n × n`.
// - Cholesky-friendly factorisation uses the parameter-space Woodbury
//   identity: factor `A = XᵀDX + S` once (the existing dense / sparse
//   path), then solve the small `r × r` capacitance system.
// ---------------------------------------------------------------------------

use crate::linalg::low_rank_weight::LowRankWeight;


/// `Xᵀ W X` for a low-rank-corrected weight, where the diagonal part is
/// assembled by the **existing** signed-Gram kernels and the rank-r
/// correction is added in place via [`LowRankWeight::add_low_rank_xtwx_correction`].
///
/// This is the new sibling of `GamWorkingModel::compute_xtwx_blas`; it is
/// a free function (not a method on `GamWorkingModel`) so it can be reused
/// for backward passes through downstream models without holding a borrow
/// on a working-model instance.
///
/// Rank-0 fast path: returns the legacy diagonal-W Gram unchanged.
pub fn compute_xtwx_low_rank(
    workspace: &mut PirlsWorkspace,
    design: &DesignMatrix,
    weight: &LowRankWeight<'_>,
) -> Result<Array2<f64>, EstimationError> {
    // Diagonal part: reuse the diagonal-W BLAS / sparse path verbatim.
    let diag_owned = weight.diag.to_owned();
    let mut xtwx = GamWorkingModel::compute_xtwx_blas(workspace, design, &diag_owned)?;
    if weight.is_rank_zero() {
        return Ok(xtwx);
    }
    weight
        .add_low_rank_xtwx_correction(design, &mut xtwx)
        .map_err(EstimationError::InvalidInput)?;
    Ok(xtwx)
}


/// `Xᵀ W y` for a low-rank-corrected weight. Used in the right-hand side
/// of the weighted-LS normal equation `(XᵀWX + S) β = XᵀWz`. Rank-0 fast
/// path coincides with `design.compute_xtwy(&d, &y)`.
pub fn compute_xtwy_low_rank(
    design: &DesignMatrix,
    weight: &LowRankWeight<'_>,
    y: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    weight
        .xtw_y(design, y.view())
        .map_err(EstimationError::InvalidInput)
}


/// Dense multi-output block Fisher assembly for latent / coupled GLM fits.
///
/// Given `X` with shape `(N, K)` and per-row output Fisher blocks `W_i`
/// with shape `(N, P, P)`, this returns the coupled coefficient Hessian
/// ordered as output-major coefficients: `a*K + i`.
///
/// `H[a*K+i, b*K+j] = Σ_n row_weight[n] * X[n,i] * W[n,a,b] * X[n,j]`.
/// When `row_weights` is `None`, all row weights are one.
pub fn dense_block_xtwx(
    design: ArrayView2<'_, f64>,
    fisher_blocks: ArrayView3<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let shape = fisher_blocks.shape();
    if shape.len() != 3 || shape[0] != n || shape[1] != shape[2] {
        crate::bail_invalid_estim!(
            "dense block Fisher shape mismatch: expected ({n}, p, p), got {shape:?}"
        );
    }
    if let Some(w) = row_weights.as_ref() {
        if w.len() != n {
            crate::bail_invalid_estim!(
                "dense block row weight length mismatch: expected {n}, got {}",
                w.len()
            );
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            crate::bail_invalid_estim!("dense block row weights must be finite and non-negative");
        }
    }
    let p_out = shape[1];
    let dim = k * p_out;
    // Coupled multi-output Gram `Σ_row (W_row ⊗ x_row x_rowᵀ)` of dimension
    // `(M·k) × (M·k)`. For the multinomial softmax family this `X^T W X` is
    // rebuilt at every inner Newton cycle of every outer smoothing-parameter
    // trial, so its `O(n · M² · k²)` accumulation is the dominant inner cost
    // (#722). The per-row contributions are an independent sum, so fan the row
    // loop across the rayon pool with per-thread dense accumulators reduced by
    // addition — the arithmetic is identical to the serial accumulation,
    // bit-for-bit up to the associativity of the row partition.
    //
    // Finiteness is validated up front in a cheap `O(n · M²)` parallel scan so
    // the hot accumulation stays branch-light and the error is reported with
    // the offending `(row, a, b)` index, preserving the serial contract.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let nonfinite = (0..n)
        .into_par_iter()
        .filter_map(|row| {
            let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
            for a in 0..p_out {
                for b in 0..p_out {
                    if !(rw * fisher_blocks[[row, a, b]]).is_finite() {
                        return Some((row, a, b));
                    }
                }
            }
            None
        })
        .min();
    if let Some((row, a, b)) = nonfinite {
        crate::bail_invalid_estim!("dense block Fisher entry ({row},{a},{b}) is not finite");
    }
    let mut out = (0..n)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros((dim, dim)),
            |mut acc, row| {
                let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
                for a in 0..p_out {
                    for b in 0..p_out {
                        let wab = rw * fisher_blocks[[row, a, b]];
                        if wab == 0.0 {
                            continue;
                        }
                        let row_a = a * k;
                        let row_b = b * k;
                        for i in 0..k {
                            let xi = design[[row, i]];
                            if xi == 0.0 {
                                continue;
                            }
                            let scaled = wab * xi;
                            for j in 0..k {
                                acc[[row_a + i, row_b + j]] += scaled * design[[row, j]];
                            }
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || Array2::<f64>::zeros((dim, dim)),
            |mut a, b| {
                a += &b;
                a
            },
        );
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    Ok(out)
}


/// Dense multi-output block right-hand side `X^T W Y`, using the same
/// output-major coefficient ordering as [`dense_block_xtwx`].
pub fn dense_block_xtwy(
    design: ArrayView2<'_, f64>,
    fisher_blocks: ArrayView3<'_, f64>,
    response: ArrayView2<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let shape = fisher_blocks.shape();
    if shape.len() != 3 || shape[0] != n || shape[1] != shape[2] {
        crate::bail_invalid_estim!(
            "dense block Fisher shape mismatch: expected ({n}, p, p), got {shape:?}"
        );
    }
    let p_out = shape[1];
    if response.dim() != (n, p_out) {
        crate::bail_invalid_estim!(
            "dense block response shape mismatch: expected ({n}, {p_out}), got {}x{}",
            response.nrows(),
            response.ncols()
        );
    }
    if let Some(w) = row_weights.as_ref()
        && w.len() != n
    {
        crate::bail_invalid_estim!(
            "dense block row weight length mismatch: expected {n}, got {}",
            w.len()
        );
    }
    let mut out = Array1::<f64>::zeros(k * p_out);
    for row in 0..n {
        let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
        for a in 0..p_out {
            let mut wy = 0.0_f64;
            for b in 0..p_out {
                let wab = rw * fisher_blocks[[row, a, b]];
                if !wab.is_finite() {
                    crate::bail_invalid_estim!(
                        "dense block Fisher entry ({row},{a},{b}) is not finite"
                    );
                }
                wy += wab * response[[row, b]];
            }
            for i in 0..k {
                out[a * k + i] += design[[row, i]] * wy;
            }
        }
    }
    Ok(out)
}


/// Build the small `r × r` capacitance for the parameter-space Woodbury
/// solve `(A + Û V̂ᵀ)⁻¹ b`, where `A = XᵀDX + S` has already been factored
/// by the caller and `a_inv_uhat = A⁻¹ Û` came out of `r` back-solves
/// against that factor. The returned matrix is `I_r + V̂ᵀ A⁻¹ Û`, the
/// system the caller inverts (Cholesky for symmetric metrics, dense LU
/// otherwise) to apply the low-rank correction to the Newton direction.
pub fn woodbury_gram_capacitance(
    a_inv_uhat: &Array2<f64>,
    vhat: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    LowRankWeight::gram_capacitance(a_inv_uhat, vhat).map_err(EstimationError::InvalidInput)
}


#[cfg(test)]
mod low_rank_weight_pirls_tests {
    use super::{
        DesignMatrix, LowRankWeight, PirlsWorkspace, compute_xtwx_low_rank, compute_xtwy_low_rank,
        woodbury_gram_capacitance,
    };
    use crate::linalg::matrix::{LinearOperator, SignedWeightsView};
    use ndarray::{Array2, array};

    fn tiny_design() -> DesignMatrix {
        let x = array![
            [1.0, 0.5, -0.2],
            [0.3, 1.2, 0.4],
            [-0.1, 0.7, 1.0],
            [0.6, -0.3, 0.8],
            [0.2, 0.9, -0.5],
        ];
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x))
    }

    #[test]
    fn xtwx_low_rank_matches_diagonal_when_rank_zero() {
        let design = tiny_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = Array2::<f64>::zeros((5, 0));
        let v = Array2::<f64>::zeros((5, 0));
        let weight = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let mut ws = PirlsWorkspace::new(5, 3, 0, 0);
        let got = compute_xtwx_low_rank(&mut ws, &design, &weight).unwrap();
        let want = design
            .xt_diag_x_signed_op(SignedWeightsView::from_array(&d))
            .unwrap();
        let diff = (&got - &want).mapv(f64::abs).sum();
        assert!(diff < 1e-12, "rank-0 path diverged from diagonal: {}", diff);
    }

    #[test]
    fn xtwy_low_rank_matches_dense_reference() {
        let design = tiny_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let weight = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let y = array![0.7, -1.2, 0.3, 0.9, -0.4];
        let got = compute_xtwy_low_rank(&design, &weight, &y).unwrap();

        let xdense = design.as_dense().unwrap().to_owned();
        let mut w = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            w[[i, i]] = d[i];
        }
        w += &u.dot(&v.t());
        let want = xdense.t().dot(&w.dot(&y));
        let diff: f64 = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-10, "xtwy_low_rank diverged: {}", diff);
    }

    #[test]
    fn woodbury_capacitance_is_well_formed() {
        let uhat = array![[0.5, 0.1], [-0.2, 0.7], [0.3, -0.4]];
        let vhat = array![[0.1, 0.2], [0.6, -0.1], [-0.3, 0.4]];
        let cap = woodbury_gram_capacitance(&uhat, &vhat).unwrap();
        let want = {
            let mut m = vhat.t().dot(&uhat);
            for k in 0..2 {
                m[[k, k]] += 1.0;
            }
            m
        };
        let diff: f64 = cap
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-12);
    }
}
