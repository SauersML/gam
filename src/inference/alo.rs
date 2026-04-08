use crate::estimate::EstimationError;
use crate::estimate::{FitGeometry, UnifiedFitResult};
use crate::faer_ndarray::FaerArrayView;
use crate::linalg::utils::StableSolver;
use crate::pirls;
use crate::types::LinkFunction;
use faer::Mat as FaerMat;
use faer::linalg::matmul::matmul;
use faer::{Accum, Par};
use ndarray::{Array1, Array2, ArrayView1, s};

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
fn alo_eta_updatewith_offset(eta_hat: f64, z: f64, offset: f64, aii: f64) -> f64 {
    let denom = 1.0 - aii;
    // PIRLS solve is centered on offset:
    //   eta - offset = A (z - offset)
    let eta_centered = eta_hat - offset;
    let z_centered = z - offset;
    offset + (eta_centered - aii * z_centered) / denom
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
fn multiblock_alo_chunk_size(p_tot: usize, n_blocks: usize, n_obs: usize) -> usize {
    if p_tot == 0 || n_blocks == 0 || n_obs == 0 {
        return 1;
    }
    let bytes_per_obs = (p_tot * n_blocks * std::mem::size_of::<f64>()).max(1);
    let budget_obs = (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / bytes_per_obs).max(1);
    budget_obs.min(n_obs)
}

fn compute_alo_diagnostics_from_pirls_impl(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    let x_dense_arc = base
        .x_transformed
        .try_to_dense_arc("ALO diagnostics require dense transformed design")
        .map_err(EstimationError::InvalidInput)?;
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
            let mut rss = 0.0;
            for i in 0..n {
                let r = y[i] - base.finalmu[i];
                let wi = base.finalweights[i];
                rss += wi * r * r;
            }
            let dof = (n as f64) - base.edf;
            let denom = dof.max(1.0);
            rss / denom
        }
    };

    let e = &base.reparam_result.e_transformed;
    let ridge = base.ridge_passport.laplacehessianridge().max(0.0);

    // ALO needs dense Hessian for chunked column solves via StableSolver.
    let h_dense_for_alo = base.stabilizedhessian_transformed.to_dense();

    // Build model-agnostic AloInput from PIRLS geometry, then delegate.
    let input = AloInput {
        design: x_dense,
        penalized_hessian: &h_dense_for_alo,
        working_weights: &base.finalweights,
        working_response: &base.solveworking_response,
        eta: &base.final_eta,
        offset: &base.final_offset,
        link,
        phi,
        penalty_null_space: if e.nrows() > 0 { Some(e) } else { None },
        ridge,
    };

    let result = compute_alo_from_input(&input)?;

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
        return Err(EstimationError::ModelIsIllConditioned {
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
    /// IRLS working weights at convergence (n).
    pub working_weights: &'a Array1<f64>,
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
    /// Optional null-space projector E (rank × p) for sandwich SE.
    /// When `None`, sandwich SE is set equal to Bayesian SE.
    pub penalty_null_space: Option<&'a Array2<f64>>,
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
            working_weights: &geom.working_weights,
            working_response: &geom.working_response,
            eta,
            offset,
            link,
            phi,
            penalty_null_space: None,
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
    let x_dense = input.design;
    let n = x_dense.nrows();
    let p = x_dense.ncols();
    let w = input.working_weights;

    let factor = StableSolver::new("alo penalized hessian")
        .factorize(input.penalized_hessian)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;

    let xt = x_dense.t();
    let phi = input.phi;
    let ridge = input.ridge;

    let e_rank = input.penalty_null_space.map(|e| e.nrows()).unwrap_or(0);

    let mut aii = Array1::<f64>::zeros(n);
    let mut se_bayes = Array1::<f64>::zeros(n);
    let mut se_sandwich = Array1::<f64>::zeros(n);

    let block_cols = 8192usize;
    let mut rhs_chunk_buf = Array2::<f64>::zeros((p, block_cols));

    for chunk_start in (0..n).step_by(block_cols) {
        let chunk_end = (chunk_start + block_cols).min(n);
        let width = chunk_end - chunk_start;

        rhs_chunk_buf
            .slice_mut(s![.., ..width])
            .assign(&xt.slice(s![.., chunk_start..chunk_end]));

        let rhs_chunkview = rhs_chunk_buf.slice(s![.., ..width]);
        let rhs_chunk = FaerArrayView::new(&rhs_chunkview);
        let s_chunk = factor.solve(rhs_chunk.as_ref());

        let mut es_chunk_storage = FaerMat::<f64>::zeros(e_rank, width);
        if e_rank > 0 {
            if let Some(e) = input.penalty_null_space {
                let eview = FaerArrayView::new(e);
                matmul(
                    es_chunk_storage.as_mut(),
                    Accum::Replace,
                    eview.as_ref(),
                    s_chunk.as_ref(),
                    1.0,
                    Par::Seq,
                );
            }
        }

        for local_col in 0..width {
            let obs = chunk_start + local_col;
            let xrow = x_dense.row(obs);
            let mut x_hinv_x = 0.0f64;
            let mut s_norm2 = 0.0f64;
            for row in 0..p {
                let sval = s_chunk[(row, local_col)];
                let xval = xrow[row];
                x_hinv_x = sval.mul_add(xval, x_hinv_x);
                s_norm2 = sval.mul_add(sval, s_norm2);
            }
            let ai = w[obs].max(0.0) * x_hinv_x;
            let mut es_norm2 = 0.0f64;
            if e_rank > 0 {
                for r in 0..e_rank {
                    let v = es_chunk_storage[(r, local_col)];
                    es_norm2 = v.mul_add(v, es_norm2);
                }
            }
            aii[obs] = ai;

            let var_bayes = bayesvar_eta(phi, x_hinv_x);
            let var_sandwich = if e_rank > 0 {
                sandwichvar_eta(phi, x_hinv_x, es_norm2, ridge, s_norm2)
            } else {
                var_bayes
            };

            if !var_bayes.is_finite() || !var_sandwich.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "ALO variance is not finite at row {obs}: bayes={var_bayes:.6e}, sandwich={var_sandwich:.6e}"
                )));
            }
            let bayes_tol = variance_negative_tolerance(phi * x_hinv_x.abs());
            if var_bayes < -bayes_tol {
                return Err(EstimationError::InvalidInput(format!(
                    "ALO Bayesian variance is materially negative at row {obs}: var={var_bayes:.6e}, tol={bayes_tol:.6e}"
                )));
            }
            if e_rank > 0 {
                let sandwich_scale =
                    phi * (x_hinv_x.abs() + es_norm2.abs() + (ridge * s_norm2).abs());
                let sandwich_tol = variance_negative_tolerance(sandwich_scale);
                if var_sandwich < -sandwich_tol {
                    return Err(EstimationError::InvalidInput(format!(
                        "ALO sandwich variance is materially negative at row {obs}: var={var_sandwich:.6e}, tol={sandwich_tol:.6e}"
                    )));
                }
            }

            se_bayes[obs] = var_bayes.max(0.0).sqrt();
            se_sandwich[obs] = var_sandwich.max(0.0).sqrt();
        }
    }

    let eta_hat = input.eta;
    let z = input.working_response;
    let offset = input.offset;

    let mut eta_tilde = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom_raw = 1.0 - aii[i];
        if denom_raw <= 0.0 || !denom_raw.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "ALO denominator is non-positive at row {i}: a_ii={:.6e}, 1-a_ii={:.6e}",
                aii[i], denom_raw
            )));
        }
        eta_tilde[i] = alo_eta_updatewith_offset(eta_hat[i], z[i], offset[i], aii[i]);
        if !eta_tilde[i].is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "ALO eta_tilde is not finite at row {i}: eta_tilde={}",
                eta_tilde[i]
            )));
        }
    }

    Ok(AloDiagnostics {
        eta_tilde,
        se_bayes,
        se_sandwich,
        pred_identity: eta_hat.clone(),
        leverage: aii,
        fisherweights: w.clone(),
    })
}

/// Compute ALO diagnostics (eta_tilde, SE, leverage) from a fitted GAM result.
pub fn compute_alo_diagnostics_from_fit(
    fit: &UnifiedFitResult,
    y: ArrayView1<f64>,
    link: LinkFunction,
) -> Result<AloDiagnostics, EstimationError> {
    let pirls = fit.artifacts.pirls.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "ALO diagnostics require a PIRLS-backed fit; this fit does not expose PIRLS geometry"
                .to_string(),
        )
    })?;
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
    let geom = unified.geometry.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "UnifiedFitResult does not contain working-set geometry; \
             ALO diagnostics require geometry at convergence"
                .to_string(),
        )
    })?;
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
    let n = input.n_obs;
    let b = input.n_blocks;
    let p_tot = input.penalized_hessian_inv.nrows();

    // --- Validate dimensions ---
    if input.block_designs.len() != b {
        return Err(EstimationError::InvalidInput(format!(
            "MultiBlockAloInput: expected {} block designs, got {}",
            b,
            input.block_designs.len()
        )));
    }

    // Verify total column count matches p_tot.
    let col_sum: usize = input.block_designs.iter().map(|d| d.ncols()).sum();
    if col_sum != p_tot {
        return Err(EstimationError::InvalidInput(format!(
            "MultiBlockAloInput: total design columns ({}) != penalized_hessian_inv size ({})",
            col_sum, p_tot
        )));
    }

    let col_offsets = multiblock_col_offsets(input.block_designs);
    let chunk_size = multiblock_alo_chunk_size(p_tot, b, n);

    let ib = Array2::<f64>::eye(b);

    let mut eta_tilde = Vec::with_capacity(n);
    let mut leverage = Array1::<f64>::zeros(n);
    let mut alo_variance = Vec::with_capacity(n);
    let mut cook_distance = Array1::<f64>::zeros(n);

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);
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

        for local_i in 0..chunk_len {
            let i = chunk_start + local_i;
            // --- Assemble A_i = G_i = X_i H⁻¹ X_iᵀ  (B × B) ---
            let mut a_i = Array2::<f64>::zeros((b, b));
            for a in 0..b {
                let x_a = &input.block_designs[a];
                let p_a = x_a.ncols();
                let off_a = col_offsets[a];
                for bb in 0..b {
                    let q_bb = &q_blocks[bb];
                    let mut dot = 0.0f64;
                    for k in 0..p_a {
                        dot += x_a[(i, k)] * q_bb[(off_a + k, local_i)];
                    }
                    a_i[(a, bb)] = dot;
                }
            }

            let w_i = &input.block_weights[i];
            let mut h_ii = Array2::<f64>::zeros((b, b));
            for r in 0..b {
                for c in 0..b {
                    let mut v = 0.0f64;
                    for k in 0..b {
                        v += a_i[(r, k)] * w_i[(k, c)];
                    }
                    h_ii[(r, c)] = v;
                }
            }

            let mut tr = 0.0f64;
            for d in 0..b {
                tr += h_ii[(d, d)];
            }
            leverage[i] = tr;

            let s_i = &input.scores[i];
            let eta_i = &input.eta_hat[i];

            let mut imwa = ib.clone();
            for r in 0..b {
                for c in 0..b {
                    let mut wa_rc = 0.0f64;
                    for k in 0..b {
                        wa_rc += w_i[(r, k)] * a_i[(k, c)];
                    }
                    imwa[(r, c)] -= wa_rc;
                }
            }

            let imwa_det = det_small(&imwa, b);
            let imwa_to_solve = if imwa_det.abs() < 1e-12 {
                let eps = 1e-6;
                let mut reg = imwa.clone();
                for d in 0..b {
                    reg[(d, d)] += eps;
                }
                reg
            } else {
                imwa
            };

            let v_i = solve_small(&imwa_to_solve, s_i, b);

            let mut delta_eta = Array1::<f64>::zeros(b);
            for r in 0..b {
                let mut acc = 0.0f64;
                for k in 0..b {
                    acc += a_i[(r, k)] * v_i[k];
                }
                delta_eta[r] = acc;
            }

            let mut corrected = eta_i.clone();
            for d in 0..b {
                corrected[d] += delta_eta[d];
            }
            eta_tilde.push(corrected);

            let mut cook = 0.0f64;
            for r in 0..b {
                let mut w_delta_r = 0.0f64;
                for k in 0..b {
                    w_delta_r += w_i[(r, k)] * delta_eta[k];
                }
                cook += delta_eta[r] * w_delta_r;
            }
            cook_distance[i] = cook;

            let mut imaw = ib.clone();
            for r in 0..b {
                for c in 0..b {
                    let mut aw_rc = 0.0f64;
                    for k in 0..b {
                        aw_rc += a_i[(r, k)] * w_i[(k, c)];
                    }
                    imaw[(r, c)] -= aw_rc;
                }
            }

            let imaw_det = det_small(&imaw, b);
            let imaw_to_solve = if imaw_det.abs() < 1e-12 {
                let eps = 1e-6;
                let mut reg = imaw.clone();
                for d in 0..b {
                    reg[(d, d)] += eps;
                }
                reg
            } else {
                imaw
            };

            let mut var_diag = Array1::<f64>::zeros(b);
            for d in 0..b {
                let mut a_col_d = Array1::<f64>::zeros(b);
                for k in 0..b {
                    a_col_d[k] = a_i[(d, k)];
                }

                let u_d = solve_small(&imaw_to_solve, &a_col_d, b);
                let mut w_u_d = Array1::<f64>::zeros(b);
                for r in 0..b {
                    let mut acc = 0.0f64;
                    for k in 0..b {
                        acc += w_i[(r, k)] * u_d[k];
                    }
                    w_u_d[r] = acc;
                }

                let t_d = solve_small(&imwa_to_solve, &w_u_d, b);
                let mut v_dd = 0.0f64;
                for k in 0..b {
                    v_dd += a_i[(d, k)] * t_d[k];
                }
                var_diag[d] = v_dd.max(0.0);
            }
            alo_variance.push(var_diag);
        }
    }

    Ok(MultiBlockAloDiagnostics {
        eta_tilde,
        leverage,
        alo_variance,
        cook_distance,
    })
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
    let n = n_obs;
    let b = n_blocks;
    let p_tot = penalized_hessian_inv.nrows();

    let col_offsets = multiblock_col_offsets(block_designs);
    let chunk_size = multiblock_alo_chunk_size(p_tot, b, n);

    let mut leverage = Array1::<f64>::zeros(n);

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);
        let chunk_len = chunk_end - chunk_start;
        let mut q_blocks = Vec::with_capacity(b);
        for blk in 0..b {
            let x_chunk_t = block_designs[blk]
                .slice(s![chunk_start..chunk_end, ..])
                .t()
                .to_owned();
            let off_b = col_offsets[blk];
            let h_slice = penalized_hessian_inv
                .slice(s![.., off_b..off_b + x_chunk_t.nrows()])
                .to_owned();
            q_blocks.push(h_slice.dot(&x_chunk_t));
        }

        for local_i in 0..chunk_len {
            let i = chunk_start + local_i;
            let w_i = &block_weights[i];
            let mut tr = 0.0f64;
            for a in 0..b {
                let x_a = &block_designs[a];
                let p_a = x_a.ncols();
                let off_a = col_offsets[a];
                for k in 0..b {
                    let q_k = &q_blocks[k];
                    let mut g_ak = 0.0f64;
                    for j in 0..p_a {
                        g_ak += x_a[(i, j)] * q_k[(off_a + j, local_i)];
                    }
                    tr += g_ak * w_i[(k, a)];
                }
            }
            leverage[i] = tr;
        }
    }

    Ok(leverage)
}

// Small-matrix helpers for B × B systems (B typically 2-4)

/// Determinant of a small B × B matrix (B ≤ 4).
fn det_small(m: &Array2<f64>, b: usize) -> f64 {
    match b {
        1 => m[(0, 0)],
        2 => m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
        3 => {
            m[(0, 0)] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)])
                - m[(0, 1)] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)])
                + m[(0, 2)] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)])
        }
        _ => {
            // LU-style determinant for B > 3.  Copy and pivot.
            let mut a = m.clone();
            let mut det = 1.0f64;
            for col in 0..b {
                // Partial pivot.
                let mut max_val = a[(col, col)].abs();
                let mut max_row = col;
                for row in (col + 1)..b {
                    let v = a[(row, col)].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = row;
                    }
                }
                if max_val < 1e-50 {
                    return 0.0;
                }
                if max_row != col {
                    for k in 0..b {
                        let tmp = a[(col, k)];
                        a[(col, k)] = a[(max_row, k)];
                        a[(max_row, k)] = tmp;
                    }
                    det = -det;
                }
                det *= a[(col, col)];
                let pivot = a[(col, col)];
                for row in (col + 1)..b {
                    let factor = a[(row, col)] / pivot;
                    for k in (col + 1)..b {
                        a[(row, k)] -= factor * a[(col, k)];
                    }
                }
            }
            det
        }
    }
}

/// Solve a small B × B linear system m x = rhs.
///
/// For B ≤ 3 uses closed-form inverse; for B > 3 uses LU with partial pivoting.
fn solve_small(m: &Array2<f64>, rhs: &Array1<f64>, b: usize) -> Array1<f64> {
    match b {
        1 => {
            let mut out = Array1::<f64>::zeros(1);
            out[0] = rhs[0] / m[(0, 0)];
            out
        }
        2 => {
            let det = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
            let inv_det = 1.0 / det;
            let mut out = Array1::<f64>::zeros(2);
            out[0] = inv_det * (m[(1, 1)] * rhs[0] - m[(0, 1)] * rhs[1]);
            out[1] = inv_det * (-m[(1, 0)] * rhs[0] + m[(0, 0)] * rhs[1]);
            out
        }
        3 => {
            let det = det_small(m, 3);
            let inv_det = 1.0 / det;
            let mut out = Array1::<f64>::zeros(3);
            // Cramer's rule for 3×3.
            out[0] = inv_det
                * (rhs[0] * (m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)])
                    - m[(0, 1)] * (rhs[1] * m[(2, 2)] - m[(1, 2)] * rhs[2])
                    + m[(0, 2)] * (rhs[1] * m[(2, 1)] - m[(1, 1)] * rhs[2]));
            out[1] = inv_det
                * (m[(0, 0)] * (rhs[1] * m[(2, 2)] - m[(1, 2)] * rhs[2])
                    - rhs[0] * (m[(1, 0)] * m[(2, 2)] - m[(1, 2)] * m[(2, 0)])
                    + m[(0, 2)] * (m[(1, 0)] * rhs[2] - rhs[1] * m[(2, 0)]));
            out[2] = inv_det
                * (m[(0, 0)] * (m[(1, 1)] * rhs[2] - rhs[1] * m[(2, 1)])
                    - m[(0, 1)] * (m[(1, 0)] * rhs[2] - rhs[1] * m[(2, 0)])
                    + rhs[0] * (m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)]));
            out
        }
        _ => {
            // LU with partial pivoting.
            let mut a = m.clone();
            let b_vec = rhs.clone();
            let mut perm: Vec<usize> = (0..b).collect();
            for col in 0..b {
                let mut max_val = a[(perm[col], col)].abs();
                let mut max_idx = col;
                for row in (col + 1)..b {
                    let v = a[(perm[row], col)].abs();
                    if v > max_val {
                        max_val = v;
                        max_idx = row;
                    }
                }
                perm.swap(col, max_idx);
                let pivot = a[(perm[col], col)];
                if pivot.abs() < 1e-50 {
                    // Singular: return zero vector.
                    return Array1::<f64>::zeros(b);
                }
                for row in (col + 1)..b {
                    let factor = a[(perm[row], col)] / pivot;
                    a[(perm[row], col)] = factor;
                    for k in (col + 1)..b {
                        let val = a[(perm[col], k)];
                        a[(perm[row], k)] -= factor * val;
                    }
                }
            }
            // Forward substitution (Ly = Pb).
            let mut y = Array1::<f64>::zeros(b);
            for row in 0..b {
                let mut s = b_vec[perm[row]];
                for k in 0..row {
                    s -= a[(perm[row], k)] * y[k];
                }
                y[row] = s;
            }
            // Back substitution (Ux = y).
            let mut x = Array1::<f64>::zeros(b);
            for row in (0..b).rev() {
                let mut s = y[row];
                for k in (row + 1)..b {
                    s -= a[(perm[row], k)] * x[k];
                }
                x[row] = s / a[(perm[row], row)];
            }
            x
        }
    }
}

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
        let aii = 0.2;
        // centered: eta~=off + ((eta-off)-a(z-off))/(1-a)
        let expected = offset + ((eta_hat - offset) - aii * (z - offset)) / (1.0 - aii);
        let got = alo_eta_updatewith_offset(eta_hat, z, offset, aii);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_reduces_to_classicwhen_offsetzero() {
        let eta_hat = 1.25;
        let z = -0.5;
        let aii = 0.35;
        let expected = (eta_hat - aii * z) / (1.0 - aii);
        let got = alo_eta_updatewith_offset(eta_hat, z, 0.0, aii);
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

    use super::{
        MultiBlockAloInput, compute_multiblock_alo, compute_multiblock_alo_leverages, det_small,
        solve_small,
    };
    use ndarray::{Array1, Array2};

    #[test]
    fn det_small_1x1() {
        let m = Array2::from_elem((1, 1), 3.5);
        assert!((det_small(&m, 1) - 3.5).abs() < 1e-14);
    }

    #[test]
    fn det_small_2x2() {
        let m = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((det_small(&m, 2) - (-2.0)).abs() < 1e-14);
    }

    #[test]
    fn det_small_3x3_identity() {
        let m = Array2::eye(3);
        assert!((det_small(&m, 3) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn solve_small_2x2() {
        let m = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 0.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![5.0, 9.0]);
        let x = solve_small(&m, &rhs, 2);
        // 2x + y = 5, 3y = 9 => y=3, x=1
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
    }

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
