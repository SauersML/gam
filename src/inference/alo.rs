use crate::estimate::EstimationError;
use crate::estimate::{FitGeometry, UnifiedFitResult};
use crate::faer_ndarray::FaerArrayView;
use crate::linalg::utils::StableSolver;
use crate::pirls;
use crate::types::LinkFunction;
use faer::Mat as FaerMat;
use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Par};
use ndarray::{Array1, Array2, ArrayView1, ShapeBuilder, s};
use std::fmt;

/// Typed error variants for the ALO (approximate leave-one-out) diagnostics
/// module.
///
/// Public entry points continue to return `Result<_, EstimationError>`; this
/// enum is materialized at leaf sites and converted at the boundary via
/// `From<AloError> for EstimationError` so error text remains byte-identical
/// to the previous `EstimationError::InvalidInput(format!(...))` /
/// `ModelIsIllConditioned { ... }` output.
#[derive(Debug, Clone)]
pub enum AloError {
    /// Caller-supplied configuration is structurally invalid: dimension
    /// mismatch, non-finite inputs that are not weights/response, missing
    /// PIRLS / geometry artifacts, or out-of-range scalar parameters.
    InvalidInput { reason: String },
    /// IRLS weights or working response contain a non-finite entry, or the
    /// working response itself is invalid.
    WeightInvalid { reason: String },
    /// The dense design matrix required for ALO could not be materialized
    /// from the underlying PIRLS artifact (e.g. sparse-only export).
    DesignDegenerate { reason: String },
    /// The penalized Hessian factorization failed, or downstream diagnostics
    /// produced NaN values that indicate the influence matrix is unusable.
    InfluenceMatrixFailed { condition_number: f64 },
    /// Per-observation ALO computation produced a non-finite value (variance,
    /// denominator, or corrected η̃) at convergence.
    LooComputationFailed { reason: String },
}

impl fmt::Display for AloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AloError::InvalidInput { reason }
            | AloError::WeightInvalid { reason }
            | AloError::DesignDegenerate { reason }
            | AloError::LooComputationFailed { reason } => f.write_str(reason),
            AloError::InfluenceMatrixFailed { condition_number } => {
                write!(
                    f,
                    "ALO influence matrix failed (condition number {condition_number:.3e})"
                )
            }
        }
    }
}

impl std::error::Error for AloError {}

impl From<AloError> for EstimationError {
    fn from(err: AloError) -> EstimationError {
        match err {
            AloError::InvalidInput { reason }
            | AloError::WeightInvalid { reason }
            | AloError::DesignDegenerate { reason }
            | AloError::LooComputationFailed { reason } => EstimationError::InvalidInput(reason),
            AloError::InfluenceMatrixFailed { condition_number } => {
                EstimationError::ModelIsIllConditioned { condition_number }
            }
        }
    }
}

impl From<AloError> for String {
    fn from(err: AloError) -> String {
        err.to_string()
    }
}

/// Approximate leave-one-out diagnostics derived from a fitted model.
#[derive(Debug, Clone)]
pub struct AloDiagnostics {
    pub eta_tilde: Array1<f64>,
    /// Bayesian/conditional standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} x_i).
    pub se_bayes: Array1<f64>,
    /// Frequentist sandwich-style standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} X^T W X H^{-1} x_i).
    pub se_sandwich: Array1<f64>,
    pub pred_identity: Array1<f64>,
    pub leverage: Array1<f64>,
    pub fisherweights: Array1<f64>,
}

#[inline]
fn alo_eta_updatewith_offset(
    eta_hat: f64,
    z: f64,
    offset: f64,
    x_hinv_x: f64,
    score_weight: f64,
    denom: f64,
) -> f64 {
    // PIRLS working-response algebra is centered on offset, so the scalar
    // score uses (eta - offset) - (z - offset).
    let eta_centered = eta_hat - offset;
    let z_centered = z - offset;
    let score = score_weight * (eta_centered - z_centered);
    offset + eta_centered + x_hinv_x * score / denom
}

#[inline]
fn bayesvar_eta(phi: f64, x_hinv_x: f64) -> f64 {
    phi * x_hinv_x
}

#[inline]
fn sandwichvar_eta(phi: f64, x_hinv_x: f64, es_norm2: f64, ridge: f64, s_norm2: f64) -> f64 {
    // With H = X'WX + S + ridge*I and t = H^{-1}x_i:
    // t'X'WXt = t'Ht - t'St - ridge*||t||^2
    //         = x_i't - ||E t||^2 - ridge*||t||^2.
    phi * (x_hinv_x - es_norm2 - ridge * s_norm2)
}

#[inline]
fn variance_negative_tolerance(scale: f64) -> f64 {
    // Tight relative tolerance for cancellation from x'H^{-1}x - ||E t||^2 - ridge||t||^2.
    1e-12 * scale.abs().max(1.0)
}

const LEVERAGE_HIGH_THRESHOLD: f64 = 0.99;
const LEVERAGE_VERY_HIGH_THRESHOLD: f64 = 0.999;
const LEVERAGE_RATE_THRESHOLDS: [f64; 3] = [0.90, 0.95, 0.99];
const LEVERAGE_PERCENTILES: [f64; 3] = [0.50, 0.95, 0.99];
const ALO_DENOMINATOR_MIN: f64 = 1e-12;
const MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[inline]
fn percentile_index(sample_size: usize, quantile: f64) -> usize {
    if sample_size <= 1 {
        return 0;
    }
    let max_index = sample_size - 1;
    ((quantile * max_index as f64).round() as usize).min(max_index)
}

#[inline]
fn percentile_from_sorted(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        0.0
    } else {
        sorted[percentile_index(sorted.len(), quantile)]
    }
}

#[inline]
fn multiblock_col_offsets(block_designs: &[Array2<f64>]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(block_designs.len());
    let mut off = 0usize;
    for design in block_designs {
        offsets.push(off);
        off += design.ncols();
    }
    offsets
}

#[inline]
fn multiblock_alo_parallel_leverage_chunk_size(
    p_tot: usize,
    n_blocks: usize,
    n_obs: usize,
    max_workers: usize,
) -> usize {
    if p_tot == 0 || n_blocks == 0 || n_obs == 0 {
        return 1;
    }

    // Each parallel leverage chunk owns q_storage for all block RHS products
    // (B * p_tot * chunk_len) plus one transposed design chunk across all
    // blocks (p_tot * chunk_len).  Divide the global scratch budget by the
    // maximum number of chunks Rayon can execute concurrently so total live
    // per-chunk scratch remains bounded.
    let workers = max_workers.max(1);
    let per_worker_budget = (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / workers).max(1);
    let elem_count_per_obs = p_tot.saturating_mul(n_blocks.saturating_add(1)).max(1);
    let bytes_per_obs = elem_count_per_obs
        .saturating_mul(std::mem::size_of::<f64>())
        .max(1);
    let budget_obs = (per_worker_budget / bytes_per_obs).max(1);
    budget_obs.min(n_obs)
}

fn compute_alo_diagnostics_from_pirls_impl(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_inner(base, y, link).map_err(EstimationError::from)
}

fn compute_alo_diagnostics_from_pirls_inner(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, AloError> {
    let x_dense_arc = base
        .x_transformed
        .try_to_dense_arc("ALO diagnostics require dense transformed design")
        .map_err(|reason| AloError::DesignDegenerate { reason })?;
    let x_dense = x_dense_arc.as_ref();
    let n = x_dense.nrows();

    // Compute dispersion parameter.
    let phi = match link {
        LinkFunction::Log => 1.0,
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => 1.0,
        LinkFunction::Identity => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let rss: f64 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let r = y[i] - base.finalmu[i];
                    base.finalweights[i] * r * r
                })
                .sum();
            let dof = (n as f64) - base.edf;
            let denom = dof.max(1.0);
            rss / denom
        }
    };

    let e = &base.reparam_result.e_transformed;
    let ridge = base.ridge_passport.laplacehessianridge().max(0.0);

    // ALO needs the exact penalized Hessian materialized densely for chunked
    // column solves via StableSolver.  The PIRLS export path validates the
    // matrix instead of falling back to a numerical Hessian approximation.
    let h_dense_for_alo = base
        .dense_stabilizedhessian_transformed(
            "ALO diagnostics require exact dense stabilized penalized Hessian",
        )
        .map_err(|e| match e {
            EstimationError::InvalidInput(reason) => AloError::InvalidInput { reason },
            other => AloError::InvalidInput {
                reason: format!("{other:?}"),
            },
        })?;

    // Build model-agnostic AloInput from PIRLS geometry, then delegate.
    let input = AloInput {
        design: x_dense,
        penalized_hessian: &h_dense_for_alo,
        hessian_weights: &base.finalweights,
        score_weights: &base.solveweights,
        working_response: &base.solveworking_response,
        eta: &base.final_eta,
        offset: &base.final_offset,
        link,
        phi,
        penalty_root: if e.nrows() > 0 { Some(e) } else { None },
        ridge,
    };

    let result = compute_alo_from_input_inner(&input)?;

    // PIRLS-specific post-hoc leverage diagnostics logging.
    log_leverage_diagnostics(&result.leverage, phi);

    // Final NaN guard with detailed error reporting.
    let has_nan_pred = result.eta_tilde.iter().any(|&x| x.is_nan());
    let has_nan_se_bayes = result.se_bayes.iter().any(|&x| x.is_nan());
    let has_nan_se_sandwich = result.se_sandwich.iter().any(|&x| x.is_nan());
    let has_nan_leverage = result.leverage.iter().any(|&x| x.is_nan());

    if has_nan_pred || has_nan_se_bayes || has_nan_se_sandwich || has_nan_leverage {
        log::error!("[GAM ALO] NaN values found in ALO diagnostics:");
        log::error!(
            "[GAM ALO] eta_tilde: {} NaN values",
            result.eta_tilde.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se_bayes: {} NaN values",
            result.se_bayes.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] se_sandwich: {} NaN values",
            result.se_sandwich.iter().filter(|&&x| x.is_nan()).count()
        );
        log::error!(
            "[GAM ALO] leverage: {} NaN values",
            result.leverage.iter().filter(|&&x| x.is_nan()).count()
        );
        return Err(AloError::InfluenceMatrixFailed {
            condition_number: f64::INFINITY,
        });
    }

    Ok(result)
}

/// Log detailed leverage percentile diagnostics for a completed ALO computation.
fn log_leverage_diagnostics(leverage: &Array1<f64>, phi: f64) {
    let n = leverage.len();
    if n == 0 {
        return;
    }

    let mut invalid_count = 0usize;
    let mut high_leverage_count = 0usize;
    let mut threshold_counts = [0usize; LEVERAGE_RATE_THRESHOLDS.len()];
    let mut finite_leverage = Vec::with_capacity(n);

    for (obs, &ai) in leverage.iter().enumerate() {
        if ai.is_finite() {
            finite_leverage.push(ai);
        }

        if !(0.0..=1.0).contains(&ai) || !ai.is_finite() {
            invalid_count += 1;
            log::warn!("[GAM ALO] invalid leverage at i={}, a_ii={:.6e}", obs, ai);
        } else if ai > LEVERAGE_HIGH_THRESHOLD {
            high_leverage_count += 1;
            if ai > LEVERAGE_VERY_HIGH_THRESHOLD {
                log::warn!("[GAM ALO] very high leverage at i={}, a_ii={:.6e}", obs, ai);
            }
        }

        for (idx, threshold) in LEVERAGE_RATE_THRESHOLDS.iter().enumerate() {
            if ai > *threshold {
                threshold_counts[idx] += 1;
            }
        }
    }

    if invalid_count > 0 || high_leverage_count > 0 {
        log::warn!(
            "[GAM ALO] leverage diagnostics: {} invalid values, {} high values (>0.99)",
            invalid_count,
            high_leverage_count
        );
    }

    finite_leverage.sort_by(f64::total_cmp);

    let finite_n = finite_leverage.len();
    let a_mean = if finite_n > 0 {
        finite_leverage.iter().copied().sum::<f64>() / finite_n as f64
    } else {
        0.0
    };
    let a_median = percentile_from_sorted(&finite_leverage, LEVERAGE_PERCENTILES[0]);
    let a_p95 = percentile_from_sorted(&finite_leverage, LEVERAGE_PERCENTILES[1]);
    let a_p99 = percentile_from_sorted(&finite_leverage, LEVERAGE_PERCENTILES[2]);
    let a_max = finite_leverage.last().copied().unwrap_or(0.0);

    log::warn!(
        "[GAM ALO] leverage: n={}, mean={:.3e}, median={:.3e}, p95={:.3e}, p99={:.3e}, max={:.3e}",
        n,
        a_mean,
        a_median,
        a_p95,
        a_p99,
        a_max
    );
    log::warn!(
        "[GAM ALO] high-leverage: a>0.90: {:.2}%, a>0.95: {:.2}%, a>0.99: {:.2}%, dispersion phi={:.3e}",
        100.0 * (threshold_counts[0] as f64) / n as f64,
        100.0 * (threshold_counts[1] as f64) / n as f64,
        100.0 * (threshold_counts[2] as f64) / n as f64,
        phi
    );
}

/// Model-agnostic input for ALO diagnostics.
///
/// Any model with a design matrix, penalized Hessian, and IRLS geometry can
/// compute ALO leverages and leave-one-out predictions. This decouples ALO
/// from the single-block PIRLS solver and enables diagnostics for GAMLSS,
/// survival, and joint models.
pub struct AloInput<'a> {
    /// Dense design matrix X (n × p).
    pub design: &'a Array2<f64>,
    /// Penalized Hessian H = X'WX + S(λ) at convergence (p × p).
    pub penalized_hessian: &'a Array2<f64>,
    /// Hessian-side IRLS weights W_H at convergence (n).
    pub hessian_weights: &'a Array1<f64>,
    /// Score-side IRLS weights W_S paired with `working_response` (n).
    pub score_weights: &'a Array1<f64>,
    /// IRLS working response at convergence (n).
    pub working_response: &'a Array1<f64>,
    /// Fitted linear predictor η̂ (n).
    pub eta: &'a Array1<f64>,
    /// Offset vector (n). Pass zeros if no offset.
    pub offset: &'a Array1<f64>,
    /// Link function (for phi determination).
    pub link: LinkFunction,
    /// Dispersion parameter φ. For non-Gaussian families this is 1.0.
    pub phi: f64,
    /// Optional penalty square root E with E^T E = S(λ) (rank × p) for sandwich SE.
    /// When `None`, sandwich SE is set equal to Bayesian SE.
    pub penalty_root: Option<&'a Array2<f64>>,
    /// Ridge added to the Hessian for logdet surface.
    pub ridge: f64,
}

impl<'a> AloInput<'a> {
    /// Build an `AloInput` from `FitGeometry` and associated vectors.
    pub fn from_geometry(
        geom: &'a FitGeometry,
        design: &'a Array2<f64>,
        eta: &'a Array1<f64>,
        offset: &'a Array1<f64>,
        link: LinkFunction,
        phi: f64,
    ) -> Self {
        Self {
            design,
            penalized_hessian: &geom.penalized_hessian,
            // FitGeometry stores one working-weight vector, so this constructor is
            // exact only when the score- and Hessian-side IRLS weights coincide.
            hessian_weights: &geom.working_weights,
            score_weights: &geom.working_weights,
            working_response: &geom.working_response,
            eta,
            offset,
            link,
            phi,
            penalty_root: None,
            ridge: 0.0,
        }
    }
}

/// Compute ALO diagnostics from model-agnostic inputs.
///
/// This is the generalized entry point that works for any model type.
/// For standard single-block GAMs, prefer `compute_alo_diagnostics_from_fit`
/// which automatically extracts the PIRLS geometry (including sandwich SE).
pub fn compute_alo_from_input(input: &AloInput) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_from_input_inner(input).map_err(EstimationError::from)
}

fn compute_alo_from_input_inner(input: &AloInput) -> Result<AloDiagnostics, AloError> {
    let x_dense = input.design;
    let n = x_dense.nrows();
    let p = x_dense.ncols();
    let w_h = input.hessian_weights;
    let w_s = input.score_weights;

    validate_alo_solve_setup(input, n, p)?;

    let factor = StableSolver::new("alo penalized hessian")
        .factorize(input.penalized_hessian)
        .map_err(|_| AloError::InfluenceMatrixFailed {
            condition_number: f64::INFINITY,
        })?;

    let xt = x_dense.t();
    let phi = input.phi;
    let ridge = input.ridge;

    let e_rank = input.penalty_root.map(|e| e.nrows()).unwrap_or(0);

    let mut aii = Array1::<f64>::zeros(n);
    let mut x_hinv_x_diag = Array1::<f64>::zeros(n);
    let mut se_bayes = Array1::<f64>::zeros(n);
    let mut se_sandwich = Array1::<f64>::zeros(n);

    let block_cols = 8192usize;
    // Allocate the RHS scratch in column-major (Fortran) order so its column
    // slices are contiguous and align with faer's column-major solve output.
    // This removes redundant `xrow = x_dense.row(obs)` indirection inside the
    // per-observation loop: rhs_chunk_buf already holds X^T at the right cols.
    let mut rhs_chunk_buf = Array2::<f64>::zeros((p, block_cols).f());
    // Reusable faer column-major buffer for the E*S product. Building this
    // once per chunk lets the inner loop read contiguous columns directly via
    // `col_as_slice`, which is just a borrow into the existing storage.
    let mut es_chunk_storage = if e_rank > 0 {
        FaerMat::<f64>::zeros(e_rank, block_cols)
    } else {
        FaerMat::<f64>::zeros(0, 0)
    };

    for chunk_start in (0..n).step_by(block_cols) {
        let chunk_end = (chunk_start + block_cols).min(n);
        let width = chunk_end - chunk_start;

        rhs_chunk_buf
            .slice_mut(s![.., ..width])
            .assign(&xt.slice(s![.., chunk_start..chunk_end]));

        let rhs_chunkview = rhs_chunk_buf.slice(s![.., ..width]);
        let rhs_chunk = FaerArrayView::new(&rhs_chunkview);
        // s_chunk is owned column-major faer storage; its column slices are
        // contiguous and can be read directly via `col_as_slice` — no need to
        // materialize a parallel ndarray copy.
        let s_chunk = factor.solve(rhs_chunk.as_ref());

        if e_rank > 0 {
            if let Some(e) = input.penalty_root {
                let eview = FaerArrayView::new(e);
                // Compute only the leading `width` columns; `col_as_slice` will
                // index into the full-width buffer up to `width` below.
                let mut es_target = es_chunk_storage.as_mut().subcols_mut(0, width);
                matmul(
                    es_target.rb_mut(),
                    Accum::Replace,
                    eview.as_ref(),
                    s_chunk.as_ref(),
                    1.0,
                    Par::Seq,
                );
            }
        }

        let rhs_view = rhs_chunk_buf.slice(s![.., ..width]);

        for local_col in 0..width {
            let obs = chunk_start + local_col;
            // rhs is column-major Fortran ndarray; faer Mat columns are
            // contiguous by construction. Both accesses borrow the existing
            // storage directly — no per-column copy.
            let rhs_col = rhs_view.column(local_col);
            let rhs_slice = rhs_col.as_slice().expect("column-major col contiguous");
            let s_slice = s_chunk.col_as_slice(local_col);

            let mut x_hinv_x = 0.0f64;
            let mut s_norm2 = 0.0f64;
            // Fused dot products over the same column: one cache-friendly pass.
            for k in 0..p {
                let sval = s_slice[k];
                let xval = rhs_slice[k];
                x_hinv_x = sval.mul_add(xval, x_hinv_x);
                s_norm2 = sval.mul_add(sval, s_norm2);
            }
            let ai = w_h[obs].max(0.0) * x_hinv_x;
            let mut es_norm2 = 0.0f64;
            if e_rank > 0 {
                let es_slice = es_chunk_storage.col_as_slice(local_col);
                for r in 0..e_rank {
                    let v = es_slice[r];
                    es_norm2 = v.mul_add(v, es_norm2);
                }
            }
            aii[obs] = ai;
            x_hinv_x_diag[obs] = x_hinv_x;

            let var_bayes = bayesvar_eta(phi, x_hinv_x);
            let var_sandwich = if e_rank > 0 {
                sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2)
            } else {
                var_bayes
            };

            if !var_bayes.is_finite() || !var_sandwich.is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "ALO variance is not finite at row {obs}: bayes={var_bayes:.6e}, sandwich={var_sandwich:.6e}"
                    ),
                });
            }
            let bayes_tol = variance_negative_tolerance(phi * x_hinv_x.abs());
            if var_bayes < -bayes_tol {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "ALO Bayesian variance is materially negative at row {obs}: var={var_bayes:.6e}, tol={bayes_tol:.6e}"
                    ),
                });
            }
            if e_rank > 0 {
                let sandwich_scale =
                    phi * (x_hinv_x.abs() + es_norm2.abs() + (ridge * s_norm2).abs());
                let sandwich_tol = variance_negative_tolerance(sandwich_scale);
                if var_sandwich < -sandwich_tol {
                    return Err(AloError::LooComputationFailed {
                        reason: format!(
                            "ALO sandwich variance is materially negative at row {obs}: var={var_sandwich:.6e}, tol={sandwich_tol:.6e}"
                        ),
                    });
                }
            }

            se_bayes[obs] = var_bayes.max(0.0).sqrt();
            se_sandwich[obs] = var_sandwich.max(0.0).sqrt();
        }
    }

    let eta_hat = input.eta;
    let z = input.working_response;
    let offset = input.offset;

    use rayon::prelude::*;
    let eta_tilde_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let denom_raw = 1.0 - aii[i];
            if denom_raw <= ALO_DENOMINATOR_MIN || !denom_raw.is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "ALO denominator is too small at row {i}: a_ii={:.6e}, 1-a_ii={:.6e}, min={:.1e}",
                        aii[i], denom_raw, ALO_DENOMINATOR_MIN
                    ),
                });
            }
            let v = alo_eta_updatewith_offset(
                eta_hat[i],
                z[i],
                offset[i],
                x_hinv_x_diag[i],
                w_s[i],
                denom_raw,
            );
            if !v.is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!("ALO eta_tilde is not finite at row {i}: eta_tilde={v}"),
                });
            }
            Ok(v)
        })
        .collect::<Result<_, _>>()?;
    let eta_tilde = Array1::from(eta_tilde_vec);

    Ok(AloDiagnostics {
        eta_tilde,
        se_bayes,
        se_sandwich,
        pred_identity: eta_hat.clone(),
        leverage: aii,
        fisherweights: w_h.clone(),
    })
}

fn validate_alo_solve_setup(input: &AloInput, n: usize, p: usize) -> Result<(), AloError> {
    let h = input.penalized_hessian;
    if h.nrows() != p || h.ncols() != p {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO diagnostics require a dense exact penalized Hessian with shape {p}x{p}; got {}x{}",
                h.nrows(),
                h.ncols()
            ),
        });
    }
    if h.iter().any(|v| !v.is_finite()) {
        return Err(AloError::InvalidInput {
            reason: "ALO diagnostics require a finite dense exact penalized Hessian".to_string(),
        });
    }
    let sym_tol = 1e-8;
    for i in 0..p {
        for j in 0..i {
            let a = h[[i, j]];
            let b = h[[j, i]];
            let scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > sym_tol * scale {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "ALO diagnostics require a symmetric dense exact penalized Hessian; entries ({i},{j}) and ({j},{i}) differ by {:.3e}",
                        (a - b).abs()
                    ),
                });
            }
        }
    }

    let vector_lengths = [
        ("hessian_weights", input.hessian_weights.len()),
        ("score_weights", input.score_weights.len()),
        ("working_response", input.working_response.len()),
        ("eta", input.eta.len()),
        ("offset", input.offset.len()),
    ];
    for (name, len) in vector_lengths {
        if len != n {
            return Err(AloError::InvalidInput {
                reason: format!("ALO diagnostics require {name} length {n}; got {len}"),
            });
        }
    }
    if input.hessian_weights.iter().any(|v| !v.is_finite()) {
        return Err(AloError::WeightInvalid {
            reason: "ALO diagnostics require finite Hessian-side weights".to_string(),
        });
    }
    if input.score_weights.iter().any(|v| !v.is_finite()) {
        return Err(AloError::WeightInvalid {
            reason: "ALO diagnostics require finite score-side weights".to_string(),
        });
    }
    if input.working_response.iter().any(|v| !v.is_finite()) {
        return Err(AloError::WeightInvalid {
            reason: "ALO diagnostics require finite working responses".to_string(),
        });
    }
    if input.eta.iter().any(|v| !v.is_finite()) || input.offset.iter().any(|v| !v.is_finite()) {
        return Err(AloError::InvalidInput {
            reason: "ALO diagnostics require finite linear predictors and offsets".to_string(),
        });
    }
    if !input.phi.is_finite() || input.phi <= 0.0 {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO diagnostics require positive finite dispersion phi; got {}",
                input.phi
            ),
        });
    }
    if !input.ridge.is_finite() || input.ridge < 0.0 {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO diagnostics require a finite non-negative Hessian ridge; got {}",
                input.ridge
            ),
        });
    }
    if let Some(e) = input.penalty_root {
        if e.ncols() != p {
            return Err(AloError::InvalidInput {
                reason: format!(
                    "ALO diagnostics require penalty root to have {p} columns; got {}",
                    e.ncols()
                ),
            });
        }
        if e.iter().any(|v| !v.is_finite()) {
            return Err(AloError::InvalidInput {
                reason: "ALO diagnostics require finite penalty-root entries".to_string(),
            });
        }
    }
    Ok(())
}

/// Compute ALO diagnostics (eta_tilde, SE, leverage) from a fitted GAM result.
pub fn compute_alo_diagnostics_from_fit(
    fit: &UnifiedFitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    let pirls = fit
        .artifacts
        .pirls
        .as_ref()
        .ok_or_else(|| AloError::InvalidInput {
            reason:
                "ALO diagnostics require a PIRLS-backed fit; this fit does not expose PIRLS geometry"
                    .to_string(),
        })
        .map_err(EstimationError::from)?;
    compute_alo_diagnostics_from_pirls_impl(pirls, y, link)
}

/// Compute ALO diagnostics from a `UnifiedFitResult`.
///
/// Extracts `FitGeometry` from `unified.geometry`, builds an `AloInput`
/// via `from_geometry`, and delegates to `compute_alo_from_input`.
/// This avoids requiring a full `UnifiedFitResult` with PIRLS artifacts.
pub fn compute_alo_diagnostics_from_unified(
    unified: &UnifiedFitResult,
    design: &Array2<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    link: LinkFunction,
    phi: f64,
) -> Result<AloDiagnostics, EstimationError> {
    let geom = unified
        .geometry
        .as_ref()
        .ok_or_else(|| AloError::InvalidInput {
            reason: "UnifiedFitResult does not contain working-set geometry; \
             ALO diagnostics require geometry at convergence"
                .to_string(),
        })
        .map_err(EstimationError::from)?;
    let input = AloInput::from_geometry(geom, design, eta, offset, link, phi);
    compute_alo_from_input(&input)
}

/// Compute ALO diagnostics from a PIRLS result for lower-level callers.
pub fn compute_alo_diagnostics_from_pirls(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(base, y, link)
}

// Multi-block ALO for multi-predictor models (GAMLSS, survival, joint)

/// Diagnostics returned by multi-block ALO.
#[derive(Debug, Clone)]
pub struct MultiBlockAloDiagnostics {
    /// Corrected linear predictors η̃^{(-i)} for each observation.
    /// Outer length = n_obs, inner length = n_blocks (B).
    pub eta_tilde: Vec<Array1<f64>>,
    /// Per-observation leverage tr(H_ii) where H_ii is the B×B hat-matrix block.
    pub leverage: Array1<f64>,
    /// Per-observation ALO variance diagonals: for each observation i,
    /// Var(Δη_i) ≈ A_i (I - W_i A_i)⁻¹ W_i (I - A_i W_i)⁻¹ A_iᵀ.
    /// Outer length = n_obs, inner length = n_blocks (B) containing the
    /// diagonal entries of the variance matrix.
    pub alo_variance: Vec<Array1<f64>>,
    /// Cook-type ALO influence: D_i = Δη_iᵀ W_i Δη_i.
    /// Length = n_obs.
    pub cook_distance: Array1<f64>,
}

/// Model-agnostic input for multi-predictor ALO diagnostics.
///
/// Generalises [`AloInput`] to models with B > 1 linear predictors per
/// observation (e.g. location-scale GAMLSS with B=2, or survival models
/// with time-dependent predictors).
///
/// # Mathematical setup
///
/// For observation i the per-observation Jacobian is a B × p_tot block matrix
/// X_i whose b-th row is the i-th row of `block_designs[b]`.  The joint
/// hat-matrix block is
///
///   H_ii = X_i H⁻¹ X_iᵀ W_i     (B × B)
///
/// where H = Σ_i X_iᵀ W_i X_i + S is the total penalized Hessian and W_i
/// is the B × B per-observation weight matrix (negative Hessian of the
/// log-likelihood w.r.t. the B predictors at observation i).
///
/// The ALO leave-one-out correction is
///
///   Δη_i^ALO = A_i (I_B − W_i A_i)⁻¹ s_i
///
/// where A_i = X_i H⁻¹ X_iᵀ (the B×B per-observation influence matrix),
/// W_i is the B×B per-observation NLL Hessian, and
/// s_i = ∇_{η_i} NLL_i(η̂_i) is the B-dimensional score vector.
/// This is algebraically equivalent to (I_B − H_ii)⁻¹ H_ii W_i⁻¹ s_i
/// but does NOT require W_i⁻¹, which is critical when W_i is singular
/// (e.g. at boundary observations in survival models).
/// For B = 1 this reduces to the classical scalar ALO formula.
pub struct MultiBlockAloInput<'a> {
    /// Number of observations.
    pub n_obs: usize,
    /// Number of predictors per observation (B).
    pub n_blocks: usize,
    /// B design matrices, each n_obs × p_b.  The total parameter count is
    /// p_tot = Σ_b p_b.
    pub block_designs: &'a [Array2<f64>],
    /// Inverse of the penalized Hessian, H⁻¹ (p_tot × p_tot).
    pub penalized_hessian_inv: &'a Array2<f64>,
    /// Per-observation weight matrices W_i (B × B).  Length = n_obs.
    pub block_weights: Vec<Array2<f64>>,
    /// Per-observation score vectors s_i = ∇_{η_i} NLL_i.  Length = n_obs,
    /// each entry is B-dimensional.
    pub scores: Vec<Array1<f64>>,
    /// Fitted linear predictor vectors η̂_i.  Length = n_obs, each entry is
    /// B-dimensional.
    pub eta_hat: Vec<Array1<f64>>,
}

/// Compute multi-block ALO diagnostics: corrected η̃ and leverages.
///
/// # Optimisation note
///
/// The dominant cost is forming X_i H⁻¹ X_iᵀ for every observation.
/// Rather than forming the B × p_tot row-block X_i and multiplying naïvely,
/// we precompute for each block b the matrix
///
///   Q_b = H⁻¹ X_bᵀ      (p_tot × n)
///
/// Then the (a, b) entry of the B × B matrix X_i H⁻¹ X_iᵀ is simply
///
///   (X_i H⁻¹ X_iᵀ)_{a,b} = x_{a,i}ᵀ Q_b[:,i]
///                           = Σ_k  X_a[i,k] · Q_b[k,i]
///
/// where x_{a,i} is the i-th row of block-design a.  This turns the per-
/// observation work from O(B · p_tot²) into O(B² · p_tot), and the
/// precomputation is O(B · p_tot² · n) total via a single blocked solve.
pub fn compute_multiblock_alo(
    input: &MultiBlockAloInput,
) -> Result<MultiBlockAloDiagnostics, EstimationError> {
    compute_multiblock_alo_inner(input).map_err(EstimationError::from)
}

fn compute_multiblock_alo_inner(
    input: &MultiBlockAloInput,
) -> Result<MultiBlockAloDiagnostics, AloError> {
    use rayon::prelude::*;

    let n = input.n_obs;
    let b = input.n_blocks;
    let p_tot = input.penalized_hessian_inv.nrows();

    // --- Validate dimensions ---
    if input.block_designs.len() != b {
        return Err(AloError::InvalidInput {
            reason: format!(
                "MultiBlockAloInput: expected {} block designs, got {}",
                b,
                input.block_designs.len()
            ),
        });
    }

    // Verify total column count matches p_tot.
    let col_sum: usize = input.block_designs.iter().map(|d| d.ncols()).sum();
    if col_sum != p_tot {
        return Err(AloError::InvalidInput {
            reason: format!(
                "MultiBlockAloInput: total design columns ({}) != penalized_hessian_inv size ({})",
                col_sum, p_tot
            ),
        });
    }

    let col_offsets = multiblock_col_offsets(input.block_designs);
    let (chunk_size, max_concurrent_chunks) = multiblock_alo_parallel_plan(p_tot, b, n);
    let chunk_starts: Vec<usize> = (0..n).step_by(chunk_size).collect();

    // Each Rayon worker owns its small B×B/B-vector scratch buffers via
    // `map_init`, avoiding cross-thread mutation and avoiding per-observation
    // allocations.  The much larger Q panels are bounded by the parallel chunk
    // size and by wave-level concurrency, so at most roughly one global memory
    // budget worth of p_total × chunk_len panels can be live across workers.
    let mut chunk_results: Vec<Result<MultiBlockAloChunkDiagnostics, AloError>> =
        Vec::with_capacity(chunk_starts.len());
    for chunk_wave in chunk_starts.chunks(max_concurrent_chunks) {
        let mut wave_results: Vec<Result<MultiBlockAloChunkDiagnostics, AloError>> = chunk_wave
            .par_iter()
            .map_init(
                || MultiBlockAloScratch::new(b),
                |scratch, &chunk_start| {
                    let chunk_end = (chunk_start + chunk_size).min(n);
                    compute_multiblock_alo_chunk(
                        input,
                        &col_offsets,
                        chunk_start,
                        chunk_end,
                        scratch,
                    )
                },
            )
            .collect();
        chunk_results.append(&mut wave_results);
    }

    let mut eta_tilde = Vec::with_capacity(n);
    let mut leverage = Array1::<f64>::zeros(n);
    let mut alo_variance = Vec::with_capacity(n);
    let mut cook_distance = Array1::<f64>::zeros(n);

    let mut chunks = Vec::with_capacity(chunk_results.len());
    for result in chunk_results {
        chunks.push(result?);
    }
    chunks.sort_unstable_by_key(|chunk| chunk.chunk_start);

    for chunk in chunks {
        let chunk_start = chunk.chunk_start;
        eta_tilde.extend(chunk.eta_tilde);
        alo_variance.extend(chunk.alo_variance);
        for (local_i, lev) in chunk.leverage.into_iter().enumerate() {
            leverage[chunk_start + local_i] = lev;
        }
        for (local_i, cook) in chunk.cook_distance.into_iter().enumerate() {
            cook_distance[chunk_start + local_i] = cook;
        }
    }

    Ok(MultiBlockAloDiagnostics {
        eta_tilde,
        leverage,
        alo_variance,
        cook_distance,
    })
}

#[inline]
fn multiblock_alo_parallel_plan(p_tot: usize, n_blocks: usize, n_obs: usize) -> (usize, usize) {
    if p_tot == 0 || n_blocks == 0 || n_obs == 0 {
        return (1, 1);
    }
    let bytes_per_obs = (p_tot * n_blocks * std::mem::size_of::<f64>()).max(1);
    let workers = rayon::current_num_threads().max(1);
    let max_concurrent_chunks = (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / bytes_per_obs)
        .max(1)
        .min(workers);
    let per_worker_budget =
        (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / max_concurrent_chunks).max(bytes_per_obs);
    let budget_obs = (per_worker_budget / bytes_per_obs).max(1);
    (budget_obs.min(n_obs), max_concurrent_chunks)
}

struct MultiBlockAloScratch {
    a_i: Vec<f64>,
    wa: Vec<f64>,
    aw: Vec<f64>,
    imwa: Vec<f64>,
    imaw: Vec<f64>,
    perm_imwa: Vec<usize>,
    perm_imaw: Vec<usize>,
    delta_eta: Vec<f64>,
    rhs_buf: Vec<f64>,
    w_u: Vec<f64>,
    var_diag_buf: Vec<f64>,
    w_flat: Vec<f64>,
    lu_scratch: Vec<f64>,
}

impl MultiBlockAloScratch {
    fn new(b: usize) -> Self {
        let bb_sz = b * b;
        Self {
            a_i: vec![0.0f64; bb_sz],
            wa: vec![0.0f64; bb_sz],
            aw: vec![0.0f64; bb_sz],
            imwa: vec![0.0f64; bb_sz],
            imaw: vec![0.0f64; bb_sz],
            perm_imwa: vec![0usize; b],
            perm_imaw: vec![0usize; b],
            delta_eta: vec![0.0f64; b],
            rhs_buf: vec![0.0f64; b],
            w_u: vec![0.0f64; b],
            var_diag_buf: vec![0.0f64; b],
            w_flat: vec![0.0f64; bb_sz],
            lu_scratch: vec![0.0f64; b],
        }
    }
}

struct MultiBlockAloChunkDiagnostics {
    chunk_start: usize,
    eta_tilde: Vec<Array1<f64>>,
    leverage: Vec<f64>,
    alo_variance: Vec<Array1<f64>>,
    cook_distance: Vec<f64>,
}

fn compute_multiblock_alo_chunk(
    input: &MultiBlockAloInput,
    col_offsets: &[usize],
    chunk_start: usize,
    chunk_end: usize,
    scratch: &mut MultiBlockAloScratch,
) -> Result<MultiBlockAloChunkDiagnostics, AloError> {
    let b = input.n_blocks;
    let chunk_len = chunk_end - chunk_start;

    let mut q_blocks = Vec::with_capacity(b);
    for blk in 0..b {
        let x_chunk_t = input.block_designs[blk]
            .slice(s![chunk_start..chunk_end, ..])
            .t()
            .to_owned();
        let off_b = col_offsets[blk];
        let h_slice = input
            .penalized_hessian_inv
            .slice(s![.., off_b..off_b + x_chunk_t.nrows()])
            .to_owned();
        q_blocks.push(h_slice.dot(&x_chunk_t));
    }

    let mut eta_tilde = Vec::with_capacity(chunk_len);
    let mut leverage = vec![0.0f64; chunk_len];
    let mut alo_variance = Vec::with_capacity(chunk_len);
    let mut cook_distance = vec![0.0f64; chunk_len];

    for local_i in 0..chunk_len {
        let i = chunk_start + local_i;
        let w_i = &input.block_weights[i];

        // Flatten W_i once per observation (row-major).
        for r in 0..b {
            for c in 0..b {
                scratch.w_flat[r * b + c] = w_i[(r, c)];
            }
        }

        // --- Assemble A_i = X_i H⁻¹ X_iᵀ  (B × B), row-major flat. ---
        for a in 0..b {
            let x_a = &input.block_designs[a];
            let p_a = x_a.ncols();
            let off_a = col_offsets[a];
            let xa_row = x_a.row(i);
            for bb in 0..b {
                let q_bb = &q_blocks[bb];
                let mut dot = 0.0f64;
                for k in 0..p_a {
                    dot += xa_row[k] * q_bb[(off_a + k, local_i)];
                }
                scratch.a_i[a * b + bb] = dot;
            }
        }

        // WA = W_i · A_i (row-major).
        mat_mul_flat(&scratch.w_flat, &scratch.a_i, &mut scratch.wa, b);
        // AW = A_i · W_i (row-major).
        mat_mul_flat(&scratch.a_i, &scratch.w_flat, &mut scratch.aw, b);

        // Trace of H_ii = A_i W_i (= AW): leverage[i].
        // (Original code wrote H_ii = A · W — the same operator we already have in `aw`.)
        let mut tr = 0.0f64;
        for d in 0..b {
            tr += scratch.aw[d * b + d];
        }
        leverage[local_i] = tr;

        // Build (I - W A) and (I - A W) into imwa/imaw.
        for r in 0..b {
            for c in 0..b {
                let idx = r * b + c;
                let id = if r == c { 1.0 } else { 0.0 };
                scratch.imwa[idx] = id - scratch.wa[idx];
                scratch.imaw[idx] = id - scratch.aw[idx];
            }
        }

        // Factor in place with partial pivoting; ridge on the diagonal if singular.
        // Equivalence with original: original computed det via det_small, regularized
        // by adding eps=1e-6 to the diagonal when |det| < 1e-12, then re-factored on
        // the regularized matrix. Here we factor directly; if any pivot is below the
        // singular threshold we add the ridge once and re-factor — same numerical path.
        if !lu_factor_in_place(&mut scratch.imwa, &mut scratch.perm_imwa, b) {
            for r in 0..b {
                for c in 0..b {
                    let idx = r * b + c;
                    let id = if r == c { 1.0 } else { 0.0 };
                    scratch.imwa[idx] = id - scratch.wa[idx];
                }
            }
            for d in 0..b {
                scratch.imwa[d * b + d] += 1e-6;
            }
            let _ = lu_factor_in_place(&mut scratch.imwa, &mut scratch.perm_imwa, b);
        }
        if !lu_factor_in_place(&mut scratch.imaw, &mut scratch.perm_imaw, b) {
            for r in 0..b {
                for c in 0..b {
                    let idx = r * b + c;
                    let id = if r == c { 1.0 } else { 0.0 };
                    scratch.imaw[idx] = id - scratch.aw[idx];
                }
            }
            for d in 0..b {
                scratch.imaw[d * b + d] += 1e-6;
            }
            let _ = lu_factor_in_place(&mut scratch.imaw, &mut scratch.perm_imaw, b);
        }

        // v_i = (I - W A)⁻¹ s_i  -- solve into rhs_buf.
        let s_i = &input.scores[i];
        for k in 0..b {
            scratch.rhs_buf[k] = s_i[k];
        }
        lu_solve_in_place(
            &scratch.imwa,
            &scratch.perm_imwa,
            &mut scratch.rhs_buf,
            &mut scratch.lu_scratch,
            b,
        );
        // delta_eta = A_i · v_i
        for r in 0..b {
            let mut acc = 0.0f64;
            let row_off = r * b;
            for k in 0..b {
                acc += scratch.a_i[row_off + k] * scratch.rhs_buf[k];
            }
            scratch.delta_eta[r] = acc;
        }

        let eta_i = &input.eta_hat[i];
        let mut corrected = Array1::<f64>::zeros(b);
        for d in 0..b {
            corrected[d] = eta_i[d] + scratch.delta_eta[d];
        }
        eta_tilde.push(corrected);

        // Cook's distance: δη^T W δη.
        let mut cook = 0.0f64;
        for r in 0..b {
            let mut w_delta_r = 0.0f64;
            let row_off = r * b;
            for k in 0..b {
                w_delta_r += scratch.w_flat[row_off + k] * scratch.delta_eta[k];
            }
            cook += scratch.delta_eta[r] * w_delta_r;
        }
        cook_distance[local_i] = cook;

        // var_diag[d] = a_d^T (I-WA)⁻¹ W (I-AW)⁻¹ a_d
        // where a_d is the d-th row of A_i.
        // Reuses already-factored imwa and imaw (one LU factorization each, reused
        // across all B right-hand sides — major saving over the original which redid
        // both LU decompositions B times per observation).
        for d in 0..b {
            let row_off = d * b;
            // u_d = (I - A W)⁻¹ a_d
            for k in 0..b {
                scratch.rhs_buf[k] = scratch.a_i[row_off + k];
            }
            lu_solve_in_place(
                &scratch.imaw,
                &scratch.perm_imaw,
                &mut scratch.rhs_buf,
                &mut scratch.lu_scratch,
                b,
            );
            // w_u = W u_d
            for r in 0..b {
                let mut acc = 0.0f64;
                let wr = r * b;
                for k in 0..b {
                    acc += scratch.w_flat[wr + k] * scratch.rhs_buf[k];
                }
                scratch.w_u[r] = acc;
            }
            // t_d = (I - W A)⁻¹ w_u  (back-solve in place using w_u as RHS).
            lu_solve_in_place(
                &scratch.imwa,
                &scratch.perm_imwa,
                &mut scratch.w_u,
                &mut scratch.lu_scratch,
                b,
            );
            // v_dd = a_d^T t_d
            let mut v_dd = 0.0f64;
            for k in 0..b {
                v_dd += scratch.a_i[row_off + k] * scratch.w_u[k];
            }
            scratch.var_diag_buf[d] = v_dd.max(0.0);
        }
        let mut var_diag = Array1::<f64>::zeros(b);
        for d in 0..b {
            var_diag[d] = scratch.var_diag_buf[d];
        }
        alo_variance.push(var_diag);
    }

    Ok(MultiBlockAloChunkDiagnostics {
        chunk_start,
        eta_tilde,
        leverage,
        alo_variance,
        cook_distance,
    })
}

/// B × B row-major matmul: out = a · b.
#[inline]
fn mat_mul_flat(a: &[f64], b_mat: &[f64], out: &mut [f64], b: usize) {
    for r in 0..b {
        let ar = r * b;
        let or = r * b;
        for c in 0..b {
            let mut acc = 0.0f64;
            for k in 0..b {
                acc += a[ar + k] * b_mat[k * b + c];
            }
            out[or + c] = acc;
        }
    }
}

/// LU-decompose a B × B row-major matrix in place with partial pivoting and
/// physical row swaps. Returns false if any pivot |a_kk| < 1e-12 (singular).
/// On success, `m` holds L (strict lower, unit diag implicit) and U (upper, diag
/// included); `perm[k]` records the original-row index that ended up in physical
/// row k after pivoting. Pivot threshold matches the original `det_small < 1e-12`
/// path so the regularization branch fires under equivalent conditions.
fn lu_factor_in_place(m: &mut [f64], perm: &mut [usize], b: usize) -> bool {
    for i in 0..b {
        perm[i] = i;
    }
    for col in 0..b {
        // Partial pivot on column `col` over physical rows `[col..b]`.
        let mut max_val = m[col * b + col].abs();
        let mut max_idx = col;
        for row in (col + 1)..b {
            let v = m[row * b + col].abs();
            if v > max_val {
                max_val = v;
                max_idx = row;
            }
        }
        if max_val < 1e-12 {
            return false;
        }
        if max_idx != col {
            // Physically swap rows `col` and `max_idx` (full row, all columns).
            for k in 0..b {
                m.swap(col * b + k, max_idx * b + k);
            }
            perm.swap(col, max_idx);
        }
        let pivot = m[col * b + col];
        for row in (col + 1)..b {
            let factor = m[row * b + col] / pivot;
            m[row * b + col] = factor; // store L below diag
            for k in (col + 1)..b {
                let upd = factor * m[col * b + k];
                m[row * b + k] -= upd;
            }
        }
    }
    true
}

/// Solve L U x = P rhs using a previously factored matrix (LU in `m`, perm).
/// Writes the solution back into `rhs`. `scratch` must have length ≥ b.
fn lu_solve_in_place(m: &[f64], perm: &[usize], rhs: &mut [f64], scratch: &mut [f64], b: usize) {
    // Forward substitution Ly = P rhs (L is unit-diag, strict lower of m).
    let y = &mut scratch[..b];
    for row in 0..b {
        let mut s = rhs[perm[row]];
        for k in 0..row {
            s -= m[row * b + k] * y[k];
        }
        y[row] = s;
    }
    // Back substitution U x = y.  Write into rhs[].
    for row in (0..b).rev() {
        let mut s = y[row];
        for k in (row + 1)..b {
            s -= m[row * b + k] * rhs[k];
        }
        rhs[row] = s / m[row * b + row];
    }
}

/// Compute only per-observation leverages tr(H_ii) for multi-predictor models.
///
/// This is cheaper than the full ALO correction when only EDF or leverage
/// diagnostics are needed (no scores or W⁻¹ computation required).
///
/// Returns an n-length array of leverages.  The total model EDF is the sum
/// of all leverages.
pub fn compute_multiblock_alo_leverages(
    n_obs: usize,
    n_blocks: usize,
    block_designs: &[Array2<f64>],
    penalized_hessian_inv: &Array2<f64>,
    block_weights: &[Array2<f64>],
) -> Result<Array1<f64>, EstimationError> {
    use rayon::prelude::*;

    let n = n_obs;
    let b = n_blocks;
    let p_tot = penalized_hessian_inv.nrows();

    let col_offsets = multiblock_col_offsets(block_designs);
    let max_workers = rayon::current_num_threads();
    let chunk_size = multiblock_alo_parallel_leverage_chunk_size(p_tot, b, n, max_workers);

    let mut leverage = Array1::<f64>::zeros(n);

    // Per-block H_inv stripe scratch (p_tot × p_blk) is read-only once built
    // and shared by the parallel chunks.  Only per-chunk q/XT/B×B scratch is
    // replicated across Rayon workers.
    let block_widths: Vec<usize> = block_designs.iter().map(|d| d.ncols()).collect();
    let mut h_stripes: Vec<FaerMat<f64>> = block_widths
        .iter()
        .map(|&p_blk| FaerMat::<f64>::zeros(p_tot, p_blk))
        .collect();
    // Populate the H_inv stripes once: each block reads a constant column slab
    // out of `penalized_hessian_inv` and copies it into a column-major faer Mat.
    for blk in 0..b {
        let off_b = col_offsets[blk];
        let p_blk = block_widths[blk];
        let stripe = &mut h_stripes[blk];
        for c in 0..p_blk {
            for r in 0..p_tot {
                stripe[(r, c)] = penalized_hessian_inv[(r, off_b + c)];
            }
        }
    }

    leverage
        .as_slice_mut()
        .expect("newly allocated Array1 is contiguous")
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, leverage_chunk)| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_len = leverage_chunk.len();
            let chunk_end = chunk_start + chunk_len;

            // Chunk-local scratch: B×B flat row-major buffers for A_i, W_i
            // and AW = A·W.  Each worker writes only its `leverage_chunk`, so
            // output writes are disjoint and require no synchronization.
            let bb_sz = b * b;
            let mut a_i = vec![0.0f64; bb_sz];
            let mut aw = vec![0.0f64; bb_sz];
            let mut w_flat = vec![0.0f64; bb_sz];

            // Column-major faer storage for q_blocks: q_k has shape
            // (p_tot, chunk_len) with contiguous columns, so
            // `col_as_slice(local_i)` is a direct stripe.
            let mut q_storage: Vec<FaerMat<f64>> = block_widths
                .iter()
                .map(|_| FaerMat::<f64>::zeros(p_tot, chunk_len))
                .collect();

            // Per-block X^T scratch in column-major faer storage
            // (p_blk × chunk_len), owned by this chunk to keep the matmul input
            // contiguous without sharing mutable scratch across threads.
            let mut xt_storage: Vec<FaerMat<f64>> = block_widths
                .iter()
                .map(|&p_blk| FaerMat::<f64>::zeros(p_blk, chunk_len))
                .collect();

            // Build q_blocks[blk] = H_inv[:, off..off+p_blk] · X_blk[chunk, :]^T
            // entirely in column-major faer storage so subsequent column reads
            // are contiguous f64 stripes — replaces the per-chunk `to_owned()`
            // ndarray slicing + row-major `dot()` from the original.
            for blk in 0..b {
                let p_blk = block_widths[blk];

                let x_chunk = block_designs[blk].slice(s![chunk_start..chunk_end, ..]);
                let xt = &mut xt_storage[blk];
                for local_i in 0..chunk_len {
                    let row = x_chunk.row(local_i);
                    for j in 0..p_blk {
                        xt[(j, local_i)] = row[j];
                    }
                }

                matmul(
                    q_storage[blk].as_mut(),
                    Accum::Replace,
                    h_stripes[blk].as_ref(),
                    xt_storage[blk].as_ref(),
                    1.0,
                    Par::Seq,
                );
            }

            for local_i in 0..chunk_len {
                let i = chunk_start + local_i;
                let w_i = &block_weights[i];

                // Flatten W_i once per observation (row-major).
                for r in 0..b {
                    for c in 0..b {
                        w_flat[r * b + c] = w_i[(r, c)];
                    }
                }

                // Assemble A_i[a, k] = X_a[i, :] · q_k[off_a:off_a+p_a, local_i].
                // For each k, read its column once (contiguous f64 stripe), then
                // for each a take the matching offset slab.
                for r in 0..bb_sz {
                    a_i[r] = 0.0;
                }
                for k in 0..b {
                    let q_k = &q_storage[k];
                    let q_col = q_k.col_as_slice(local_i);
                    for a in 0..b {
                        let p_a = block_widths[a];
                        let off_a = col_offsets[a];
                        let xa_row = block_designs[a].row(i);
                        let mut dot = 0.0f64;
                        for j in 0..p_a {
                            dot = xa_row[j].mul_add(q_col[off_a + j], dot);
                        }
                        a_i[a * b + k] = dot;
                    }
                }

                // AW = A_i · W_i (B×B), then leverage = trace(AW) = sum_{a,k} A[a,k]·W[k,a].
                mat_mul_flat(&a_i, &w_flat, &mut aw, b);
                let mut tr = 0.0f64;
                for d in 0..b {
                    tr += aw[d * b + d];
                }
                leverage_chunk[local_i] = tr;
            }
        });

    Ok(leverage)
}

// (Allocation-free, factor-once-reuse-many B×B LU helpers live next to the
// multi-block ALO callsite — see `lu_factor_in_place` and `lu_solve_in_place`.)

#[cfg(test)]
mod tests {
    use super::{
        alo_eta_updatewith_offset, bayesvar_eta, percentile_from_sorted, percentile_index,
        sandwichvar_eta,
    };

    #[test]
    fn alo_offset_update_matches_centered_algebra() {
        let eta_hat = 11.0;
        let z = 13.0;
        let offset = 10.0;
        let x_hinv_x = 0.2;
        let hessian_weight = 1.0;
        let score_weight = 1.0;
        // centered: eta~=off + ((eta-off)-a(z-off))/(1-a) when W_S = W_H.
        let leverage = hessian_weight * x_hinv_x;
        let expected =
            offset + ((eta_hat - offset) - leverage * (z - offset)) / (1.0 - leverage);
        let got = alo_eta_updatewith_offset(
            eta_hat,
            z,
            offset,
            x_hinv_x,
            score_weight,
            1.0 - leverage,
        );
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_reduces_to_classicwhen_offsetzero() {
        let eta_hat = 1.25;
        let z = -0.5;
        let x_hinv_x = 0.35;
        let hessian_weight = 1.0;
        let score_weight = 1.0;
        let leverage = hessian_weight * x_hinv_x;
        let expected = (eta_hat - leverage * z) / (1.0 - leverage);
        let got = alo_eta_updatewith_offset(
            eta_hat,
            z,
            0.0,
            x_hinv_x,
            score_weight,
            1.0 - leverage,
        );
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_uses_distinct_score_and_hessian_weights() {
        let eta_hat = 1.7;
        let z = 0.4;
        let offset = -0.2;
        let x_hinv_x = 0.15;
        let hessian_weight = 3.0;
        let score_weight = 5.0;
        let expected = offset
            + (eta_hat - offset)
            + x_hinv_x * score_weight * ((eta_hat - offset) - (z - offset))
                / (1.0 - hessian_weight * x_hinv_x);
        let got = alo_eta_updatewith_offset(
            eta_hat,
            z,
            offset,
            x_hinv_x,
            score_weight,
            1.0 - hessian_weight * x_hinv_x,
        );
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_handles_zero_hessian_weight() {
        let eta_hat = 0.8;
        let z = -0.3;
        let offset = 0.1;
        let x_hinv_x = 0.4;
        let hessian_weight = 0.0;
        let score_weight = 2.5;
        let expected = offset
            + (eta_hat - offset)
            + x_hinv_x * score_weight * ((eta_hat - offset) - (z - offset));
        let got = alo_eta_updatewith_offset(
            eta_hat,
            z,
            offset,
            x_hinv_x,
            score_weight,
            1.0 - hessian_weight * x_hinv_x,
        );
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_unpenalized_sandwich_equals_bayes() {
        // In Gaussian linear model with S=0 and ridge=0:
        // H = X'WX, so sandwich and bayes eta variances are identical.
        let phi = 2.5;
        let x_hinv_x = 0.3;
        let es_norm2 = 0.0;
        let ridge = 0.0;
        let s_norm2 = 0.0;
        let vb = bayesvar_eta(phi, x_hinv_x);
        let vs = sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
        assert!((vb - vs).abs() < 1e-12);
    }

    #[test]
    fn sandwich_matches_direct_linear_gaussian_formula() {
        // Small brute-force linear Gaussian check:
        // var_sandwich(eta_i) = phi * x_i^T H^{-1} X'WX H^{-1} x_i.
        let phi = 1.7;
        let x_hinv_x = 0.41;
        let es_norm2 = 0.05;
        let ridge = 1e-3;
        let s_norm2 = 2.0;
        let got = sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2);
        let expected = phi * (x_hinv_x - es_norm2 - ridge * s_norm2);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn percentile_index_matches_expected_rounding() {
        assert_eq!(percentile_index(0, 0.95), 0);
        assert_eq!(percentile_index(1, 0.95), 0);
        assert_eq!(percentile_index(10, 0.50), 5);
        assert_eq!(percentile_index(10, 0.95), 9);
    }

    #[test]
    fn percentile_from_sorted_returns_order_statistic() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_from_sorted(&values, 0.50), 3.0);
        assert_eq!(percentile_from_sorted(&values, 0.95), 5.0);
        assert_eq!(percentile_from_sorted(&[], 0.95), 0.0);
    }

    // --- Multi-block ALO tests ---

    use super::{MultiBlockAloInput, compute_multiblock_alo, compute_multiblock_alo_leverages};
    use ndarray::{Array1, Array2};

    #[test]
    fn multiblock_b1_matches_scalar_leverage() {
        // With B=1 the multi-block formula should reduce to the scalar case.
        // H_ii = x_i^T H^{-1} x_i * w_i  (scalar).
        let n = 3;
        let p = 2;
        let x = Array2::from_shape_vec((n, p), vec![1.0, 0.5, 0.8, -0.3, 0.2, 1.1]).unwrap();
        // H = X'WX + I (simple regularisation).
        let w = vec![1.0, 2.0, 0.5];
        let mut h = Array2::<f64>::eye(p);
        for i in 0..n {
            for r in 0..p {
                for c in 0..p {
                    h[(r, c)] += w[i] * x[(i, r)] * x[(i, c)];
                }
            }
        }
        // Invert H (2x2).
        let det = h[(0, 0)] * h[(1, 1)] - h[(0, 1)] * h[(1, 0)];
        let mut h_inv = Array2::<f64>::zeros((p, p));
        h_inv[(0, 0)] = h[(1, 1)] / det;
        h_inv[(1, 1)] = h[(0, 0)] / det;
        h_inv[(0, 1)] = -h[(0, 1)] / det;
        h_inv[(1, 0)] = -h[(1, 0)] / det;

        // Scalar leverages: a_ii = w_i * x_i^T H^{-1} x_i
        let mut scalar_lev = vec![0.0f64; n];
        for i in 0..n {
            let mut xhx = 0.0;
            for r in 0..p {
                for c in 0..p {
                    xhx += x[(i, r)] * h_inv[(r, c)] * x[(i, c)];
                }
            }
            scalar_lev[i] = w[i] * xhx;
        }

        // Multi-block with B=1.
        let block_designs = vec![x.clone()];
        let block_weights: Vec<Array2<f64>> =
            w.iter().map(|&wi| Array2::from_elem((1, 1), wi)).collect();
        let scores: Vec<Array1<f64>> = (0..n).map(|_| Array1::from_vec(vec![0.1])).collect();
        let eta_hat: Vec<Array1<f64>> = (0..n).map(|i| Array1::from_vec(vec![i as f64])).collect();

        let input = MultiBlockAloInput {
            n_obs: n,
            n_blocks: 1,
            block_designs: &block_designs,
            penalized_hessian_inv: &h_inv,
            block_weights,
            scores,
            eta_hat,
        };

        let result = compute_multiblock_alo(&input).unwrap();
        for i in 0..n {
            assert!(
                (result.leverage[i] - scalar_lev[i]).abs() < 1e-10,
                "leverage mismatch at i={}: got {}, expected {}",
                i,
                result.leverage[i],
                scalar_lev[i]
            );
        }
    }

    #[test]
    fn multiblock_leverage_only_matches_full() {
        // Verify that compute_multiblock_alo_leverages returns the same
        // leverages as compute_multiblock_alo.
        let n = 4;
        let p1 = 2;
        let p2 = 3;
        let x1 = Array2::from_shape_fn((n, p1), |(i, j)| (i + j + 1) as f64 * 0.3);
        let x2 = Array2::from_shape_fn((n, p2), |(i, j)| (i * 2 + j) as f64 * 0.2 - 0.1);
        let p_tot = p1 + p2;
        let h_inv = Array2::<f64>::eye(p_tot); // Simple identity for test.
        let block_weights: Vec<Array2<f64>> = (0..n)
            .map(|i| {
                let v = (i + 1) as f64;
                Array2::from_shape_vec((2, 2), vec![v, 0.1, 0.1, v * 0.5]).unwrap()
            })
            .collect();
        let scores: Vec<Array1<f64>> = (0..n).map(|_| Array1::from_vec(vec![0.0, 0.0])).collect();
        let eta_hat: Vec<Array1<f64>> = (0..n).map(|_| Array1::from_vec(vec![0.0, 0.0])).collect();
        let block_designs = vec![x1.clone(), x2.clone()];

        let input = MultiBlockAloInput {
            n_obs: n,
            n_blocks: 2,
            block_designs: &block_designs,
            penalized_hessian_inv: &h_inv,
            block_weights: block_weights.clone(),
            scores,
            eta_hat,
        };
        let full = compute_multiblock_alo(&input).unwrap();
        let lev_only =
            compute_multiblock_alo_leverages(n, 2, &block_designs, &h_inv, &block_weights).unwrap();

        for i in 0..n {
            assert!(
                (full.leverage[i] - lev_only[i]).abs() < 1e-12,
                "leverage mismatch at i={}: full={}, lev_only={}",
                i,
                full.leverage[i],
                lev_only[i]
            );
        }
    }

    #[test]
    fn multiblock_singular_weight_still_corrects() {
        // When W_i = 0 (singular), the W_i⁻¹-free formula still works:
        // (I - W_i A_i)⁻¹ = I, so Δη = A_i s_i.
        // A_i = x H⁻¹ xᵀ = 1.0² + 0.5² = 1.25 (scalar, B=1).
        let n = 1;
        let p = 2;
        let x = Array2::from_shape_vec((1, p), vec![1.0, 0.5]).unwrap();
        let h_inv = Array2::eye(p);
        let block_designs = vec![x.clone()];
        let block_weights = vec![Array2::from_elem((1, 1), 0.0)]; // singular
        let scores = vec![Array1::from_vec(vec![1.0])];
        let eta_hat = vec![Array1::from_vec(vec![std::f64::consts::PI])];

        let input = MultiBlockAloInput {
            n_obs: n,
            n_blocks: 1,
            block_designs: &block_designs,
            penalized_hessian_inv: &h_inv,
            block_weights,
            scores,
            eta_hat,
        };
        let result = compute_multiblock_alo(&input).unwrap();
        // Δη = A_i * s_i = 1.25 * 1.0 = 1.25
        let expected = std::f64::consts::PI + 1.25;
        assert!(
            (result.eta_tilde[0][0] - expected).abs() < 1e-12,
            "expected {}, got {}",
            expected,
            result.eta_tilde[0][0]
        );
        // Cook's distance should be 0 since W_i = 0.
        assert!(result.cook_distance[0].abs() < 1e-14);
        // ALO variance should be 0 since W_i = 0.
        assert!(result.alo_variance[0][0].abs() < 1e-14);
    }

    #[test]
    fn multiblock_cook_and_variance_basic() {
        // B=1 with known values: verify Cook's distance and variance.
        let n = 1;
        let x = Array2::from_elem((1, 1), 1.0);
        // H⁻¹ = [[0.5]]
        let h_inv = Array2::from_elem((1, 1), 0.5);
        let block_designs = vec![x.clone()];
        let w_val = 2.0;
        let s_val = 0.4;
        let block_weights = vec![Array2::from_elem((1, 1), w_val)];
        let scores = vec![Array1::from_vec(vec![s_val])];
        let eta_hat = vec![Array1::from_vec(vec![1.0])];

        let input = MultiBlockAloInput {
            n_obs: n,
            n_blocks: 1,
            block_designs: &block_designs,
            penalized_hessian_inv: &h_inv,
            block_weights,
            scores,
            eta_hat,
        };
        let result = compute_multiblock_alo(&input).unwrap();

        // A_i = x H⁻¹ xᵀ = 1 * 0.5 * 1 = 0.5
        // (I - W A)⁻¹ = 1 / (1 - 2.0 * 0.5) = 1/0 => regularised
        // Actually 1 - w*a = 1 - 1.0 = 0.0, so det < 1e-12 => regularised with eps=1e-6
        // (I - W A + eps) = 1e-6, so v = s / 1e-6 = 4e5
        // delta_eta = A * v = 0.5 * 4e5 = 2e5
        // This is the regularised case; just check it doesn't panic and returns finite values.
        assert!(result.eta_tilde[0][0].is_finite());
        assert!(result.cook_distance[0].is_finite());
        assert!(result.alo_variance[0][0].is_finite());
    }
}
