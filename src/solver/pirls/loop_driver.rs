//! Outer driver for a single fixed-ρ PIRLS fit.
//!
//! Owns:
//! - `fit_model_for_fixed_rho` and `fit_model_for_fixed_rho_with_adaptive_kkt`
//!   — build the working model, run the inner LM loop, assemble the final result.
//! - `PirlsProblem`, `PenaltyConfig`, `PirlsConfig` — the configuration types.
//! - Helper functions exclusive to the fixed-ρ fitting path: constraint
//!   transformation, sparse-native decision, reparam materialisation, prior
//!   shift assembly, initial-β guess, Gaussian short-circuit assembly, etc.
//! - The two GPU dispatch blocks (Stage 3.3) that call into
//!   `crate::solver::gpu::pirls_dispatch_wire`.

use super::{
    FIXED_STABILIZATION_RIDGE, GamWorkingModel, PirlsWorkspace, SparsePirlsDecision,
    WorkingModel, WorkingModelPirlsOptions, WorkingReparamTransform,
    attach_penalty_shift, should_use_sparse_native_pirls, solve_penalized_least_squares_implicit,
    runworking_model_pirls, standard_inverse_link_jet,
    // edf helpers
    calculate_edf_with_penalty, calculate_edfwithworkspace_with_penalty,
    // state re-exports
    AdaptiveKktTolerance, ExportedLaplaceCurvature, FirthDiagnostics, HessianCurvatureKind,
    PirlsCoordinateFrame, PirlsLinearSolvePath, PirlsResult, PirlsStatus,
    WorkingModelIterationInfo, WorkingModelPirlsResult, WorkingState,
    LinearInequalityConstraints,
    // pls_solver types
    GaussianFixedCache, SparseXtwxPrecomputed,
    // penalty types
    KroneckerQsTransform, PirlsPenalty,
    // misc helpers
    array1_l2_norm, inf_norm, compute_constraint_kkt_diagnostics,
    // compute functions
    calculate_deviance, calculate_loglikelihood_omitting_constants,
    computeworkingweight_derivatives_from_eta,
    // moved into mod.rs but referenced from this driver:
    ArrowSchurInnerConfig, GamModelFinalState, project_coefficients_to_lower_bounds,
};
use super::convergence::effective_kkt_tolerance;
use super::gpu_dispatch::{try_gaussian_pls_gpu, try_pirls_loop_gpu};
use crate::linalg::faer_ndarray::fast_ab;
use crate::probability::standard_normal_quantile;
use crate::construction::{
    EngineDims, KroneckerReparamResult, ReparamResult,
    create_balanced_penalty_root_from_canonical,
    kronecker_reparameterization_engine, stable_reparameterization_engine_canonical,
};
use crate::estimate::EstimationError;
use crate::matrix::{DesignMatrix, LinearOperator, ReparamOperator, SymmetricMatrix};
use crate::solver::active_set;
use crate::types::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LinearPredictor, LinkFunction,
    LogSmoothingParamsView, MixtureLinkState, RidgePassport, RidgePolicy,
    SasLinkState, StandardLink,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::borrow::Cow;
use std::sync::Arc;
use faer::sparse::{SparseColMat, Triplet};

pub(super) fn default_beta_guess_external(
    p: usize,
    link_function: LinkFunction,
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Array1<f64> {
    let mut beta = Array1::<f64>::zeros(p);
    let intercept_col = 0usize;
    match link_function {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => {
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                let prevalence =
                    ((weighted_sum + 0.5) / (totalweight + 1.0)).clamp(1e-6, 1.0 - 1e-6);
                beta[intercept_col] = match link_function {
                    LinkFunction::Logit => (prevalence / (1.0 - prevalence)).ln(),
                    LinkFunction::Probit => {
                        standard_normal_quantile(prevalence).unwrap_or_else(|_| {
                            // `prevalence` is clamped to (0, 1); this fallback is
                            // only for defensive robustness under non-finite upstream inputs.
                            (prevalence / (1.0 - prevalence)).ln()
                        })
                    }
                    LinkFunction::CLogLog => (-(1.0 - prevalence).ln()).ln(),
                    LinkFunction::Sas => solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or_else(|| {
                        standard_normal_quantile(prevalence)
                            .unwrap_or_else(|_| (prevalence / (1.0 - prevalence)).ln())
                    }),
                    LinkFunction::BetaLogistic => solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or_else(|| {
                        standard_normal_quantile(prevalence)
                            .unwrap_or_else(|_| (prevalence / (1.0 - prevalence)).ln())
                    }),
                    // Outer arm guard already filtered out Log/Identity; fall
                    // back to the canonical logit transform for defensive safety
                    // if these are ever reached unexpectedly.
                    LinkFunction::Log | LinkFunction::Identity => {
                        (prevalence / (1.0 - prevalence)).ln()
                    }
                };
                if mixture_link_state.is_some() {
                    beta[intercept_col] = solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or(beta[intercept_col]);
                }
            }
        }
        LinkFunction::Identity => {
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                beta[intercept_col] = weighted_sum / totalweight;
            }
        }
        LinkFunction::Log => {
            // For log link, intercept = ln(weighted mean of y)
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                let mean_y = (weighted_sum / totalweight).max(1e-10);
                beta[intercept_col] = mean_y.ln();
            }
        }
    }
    beta
}

pub(super) fn solve_intercept_for_prevalence(
    link_function: LinkFunction,
    prevalence: f64,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Option<f64> {
    #[inline]
    fn f_eta(
        link_function: LinkFunction,
        eta: f64,
        prevalence: f64,
        mixture_link_state: Option<&MixtureLinkState>,
        sas_link_state: Option<&SasLinkState>,
    ) -> f64 {
        let inverse_link = if let Some(state) = mixture_link_state {
            InverseLink::Mixture(state.clone())
        } else if let Some(state) = sas_link_state {
            match link_function {
                LinkFunction::BetaLogistic => InverseLink::BetaLogistic(*state),
                _ => InverseLink::Sas(*state),
            }
        } else {
            // SAFETY: when `sas_link_state` is None, `solve_intercept_for_prevalence`
            // is only invoked with the five legal `StandardLink` variants (the
            // dispatch site at pirls.rs:4203 routes Sas/BetaLogistic into the
            // Some branch above with state).
            InverseLink::Standard(StandardLink::try_from(link_function).expect(
                "state-bearing link reached state-less arm in solve_intercept_for_prevalence",
            ))
        };
        standard_inverse_link_jet(&inverse_link, eta)
            .map(|jet| jet.mu - prevalence)
            .unwrap_or(f64::NAN)
    }

    let mut lo = -40.0;
    let mut hi = 40.0;
    let mut f_lo = f_eta(
        link_function,
        lo,
        prevalence,
        mixture_link_state,
        sas_link_state,
    );
    let mut f_hi = f_eta(
        link_function,
        hi,
        prevalence,
        mixture_link_state,
        sas_link_state,
    );
    if !(f_lo.is_finite() && f_hi.is_finite()) {
        return None;
    }
    for _ in 0..8 {
        if f_lo <= 0.0 && f_hi >= 0.0 {
            break;
        }
        lo *= 2.0;
        hi *= 2.0;
        f_lo = f_eta(
            link_function,
            lo,
            prevalence,
            mixture_link_state,
            sas_link_state,
        );
        f_hi = f_eta(
            link_function,
            hi,
            prevalence,
            mixture_link_state,
            sas_link_state,
        );
        if !(f_lo.is_finite() && f_hi.is_finite()) {
            return None;
        }
    }
    if f_lo > 0.0 {
        return Some(lo);
    }
    if f_hi < 0.0 {
        return Some(hi);
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f_eta(
            link_function,
            mid,
            prevalence,
            mixture_link_state,
            sas_link_state,
        );
        if !f_mid.is_finite() {
            return None;
        }
        if f_mid > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(0.5 * (lo + hi))
}





pub(super) fn assemble_pirls_result(
    working_summary: &WorkingModelPirlsResult,
    likelihood: GlmLikelihoodSpec,
    offset: ArrayView1<'_, f64>,
    penalized_hessian_transformed: SymmetricMatrix,
    stabilizedhessian_transformed: SymmetricMatrix,
    edf: f64,
    penalty_term: f64,
    finalmu: &Array1<f64>,
    finalweights: &Array1<f64>,
    scoreweights: &Array1<f64>,
    finalz: &Array1<f64>,
    final_c: &Array1<f64>,
    final_d: &Array1<f64>,
    final_dmu_deta: &Array1<f64>,
    final_d2mu_deta2: &Array1<f64>,
    final_d3mu_deta3: &Array1<f64>,
    status: PirlsStatus,
    reparam_result: ReparamResult,
    x_transformed: DesignMatrix,
    coordinate_frame: PirlsCoordinateFrame,
    linear_constraints_transformed: Option<LinearInequalityConstraints>,
) -> PirlsResult {
    let final_eta_arr = working_summary.state.eta.as_ref().clone();
    PirlsResult {
        likelihood,
        beta_transformed: working_summary.beta.clone(),
        penalized_hessian_transformed,
        stabilizedhessian_transformed,
        ridge_passport: RidgePassport::scaled_identity(
            working_summary.state.ridge_used,
            RidgePolicy::explicit_stabilization_full(),
        ),
        ridge_used: working_summary.state.ridge_used,
        deviance: working_summary.state.deviance,
        edf,
        stable_penalty_term: penalty_term,
        firth: working_summary.state.firth.clone(),
        finalweights: finalweights.clone(),
        final_offset: offset.to_owned(),
        final_eta: final_eta_arr,
        finalmu: finalmu.clone(),
        solveweights: scoreweights.clone(),
        solveworking_response: finalz.clone(),
        solvemu: finalmu.clone(),
        solve_dmu_deta: final_dmu_deta.clone(),
        solve_d2mu_deta2: final_d2mu_deta2.clone(),
        solve_d3mu_deta3: final_d3mu_deta3.clone(),
        solve_c_array: final_c.clone(),
        solve_d_array: final_d.clone(),
        derivatives_unsupported: false,
        status,
        iteration: working_summary.iterations,
        max_abs_eta: working_summary.max_abs_eta,
        lastgradient_norm: working_summary.lastgradient_norm,
        gradient_natural_scale: working_summary.state.gradient_natural_scale,
        last_deviance_change: working_summary.last_deviance_change,
        last_step_halving: working_summary.last_step_halving,
        hessian_curvature: working_summary.state.hessian_curvature,
        exported_laplace_curvature: working_summary.exported_laplace_curvature.clone(),
        final_lm_lambda: working_summary.final_lm_lambda,
        final_accept_rho: working_summary.final_accept_rho,
        constraint_kkt: working_summary.constraint_kkt.clone(),
        linear_constraints_transformed,
        reparam_result,
        x_transformed,
        coordinate_frame,
        cache_compacted: false,
        min_penalized_deviance: working_summary.min_penalized_deviance,
    }
}

pub(super) fn detect_logit_instability(
    link: LinkFunction,
    has_penalty: bool,
    firth_active: bool,
    summary: &WorkingModelPirlsResult,
    finalmu: &Array1<f64>,
    finalweights: &Array1<f64>,
    y: ArrayView1<'_, f64>,
) -> bool {
    if link != LinkFunction::Logit || firth_active {
        return false;
    }

    let n = y.len() as f64;
    if n == 0.0 {
        return false;
    }

    let max_abs_eta = summary.max_abs_eta;
    let sat_fraction = {
        const SAT_EPS: f64 = 1e-3;
        finalmu
            .iter()
            .filter(|&&m| m <= SAT_EPS || m >= 1.0 - SAT_EPS)
            .count() as f64
            / n
    };

    let weight_collapse_fraction = {
        const WEIGHT_EPS: f64 = 1e-8;
        finalweights
            .iter()
            .filter(|&&w| w <= WEIGHT_EPS || !w.is_finite())
            .count() as f64
            / n
    };

    let beta_norm = summary.beta.as_ref().dot(summary.beta.as_ref()).sqrt();
    let dev_per_sample = summary.state.deviance / n;

    let mut has_pos = false;
    let mut has_neg = false;
    let mut min_eta_pos = f64::INFINITY;
    let mut max_eta_neg = f64::NEG_INFINITY;
    for (eta_i, &yi) in summary.state.eta.iter().zip(y.iter()) {
        if yi > 0.5 {
            has_pos = true;
            if *eta_i < min_eta_pos {
                min_eta_pos = *eta_i;
            }
        } else {
            has_neg = true;
            if *eta_i > max_eta_neg {
                max_eta_neg = *eta_i;
            }
        }
    }
    let order_separated = has_pos && has_neg && (min_eta_pos - max_eta_neg) > 1e-3;

    let classic_signals =
        max_abs_eta > 30.0 || sat_fraction > 0.98 || dev_per_sample < 1e-3 || beta_norm > 1e4;

    if !has_penalty {
        return classic_signals || order_separated;
    }

    let severe_saturation = sat_fraction > 0.995 && max_abs_eta > 30.0;
    let weights_collapsed = weight_collapse_fraction > 0.98;
    let dev_extremely_small = dev_per_sample < 1e-6;

    order_separated || severe_saturation || weights_collapsed || dev_extremely_small
}

/// Stack λ-weighted penalty roots from canonical penalties into a single
/// `total_rank × p` matrix for PIRLS. Each block-local root is embedded
/// into the full column space on-the-fly.
pub(super) fn stack_lambdaweighted_penalty_root_canonical(
    penalties: &[crate::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> Array2<f64> {
    let totalrows: usize = penalties.iter().map(|cp| cp.rank()).sum();
    if totalrows == 0 {
        return Array2::zeros((0, p));
    }
    let mut e = Array2::<f64>::zeros((totalrows, p));
    let mut row_start = 0usize;
    for (k, cp) in penalties.iter().enumerate() {
        let rows = cp.rank();
        if rows == 0 {
            continue;
        }
        let scale = lambdas.get(k).copied().unwrap_or(0.0).max(0.0).sqrt();
        if scale != 0.0 {
            // Embed block-local root (rank × block_dim) into full width (rank × p).
            let r = &cp.col_range;
            for row in 0..rows {
                for col in 0..cp.block_dim() {
                    e[[row_start + row, r.start + col]] = scale * cp.root[[row, col]];
                }
            }
        }
        row_start += rows;
    }
    e
}

pub(super) fn build_sparse_native_reparam_result(
    base: ReparamResult,
    penalties: &[crate::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> ReparamResult {
    // Assemble weighted penalty sum block-locally.
    let mut s_original = Array2::<f64>::zeros((p, p));
    for (k, cp) in penalties.iter().enumerate() {
        let lambda_k = lambdas.get(k).copied().unwrap_or(0.0);
        if lambda_k != 0.0 {
            cp.accumulate_weighted(&mut s_original, lambda_k);
        }
    }
    let u_original = if base.u_truncated.nrows() == p {
        fast_ab(&base.qs, &base.u_truncated)
    } else {
        Array2::<f64>::eye(p)
    };
    // In the sparse-native path, qs = I, so the penalties are already in the
    // right coordinate frame. We keep them as-is in canonical_transformed.
    let canonical_transformed: Vec<crate::construction::CanonicalPenalty> = penalties.to_vec();
    ReparamResult {
        penalty_shrinkage_ridge: base.penalty_shrinkage_ridge,
        s_transformed: s_original,
        log_det: base.log_det,
        det1: base.det1,
        qs: Array2::<f64>::eye(p),
        canonical_transformed,
        e_transformed: stack_lambdaweighted_penalty_root_canonical(penalties, lambdas, p),
        u_truncated: u_original,
    }
}

pub(super) fn build_diagonal_penalty_from_kronecker(
    kron_result: &KroneckerReparamResult,
    lambdas: &[f64],
) -> PirlsPenalty {
    let d = kron_result.marginal_dims.len();
    let p: usize = kron_result.marginal_dims.iter().copied().product();
    let mut diag = Array1::<f64>::zeros(p);
    let mut positive_indices = Vec::new();

    let mut multi_idx = vec![0usize; d];
    let mut flat = 0usize;
    loop {
        let mut sigma = kron_result.penalty_shrinkage_ridge;
        for k in 0..d {
            sigma += lambdas[k] * kron_result.marginal_eigenvalues[k][multi_idx[k]];
        }
        if kron_result.has_double_penalty && lambdas.len() > d {
            sigma += lambdas[d];
        }
        diag[flat] = sigma;
        if sigma > 0.0 {
            positive_indices.push(flat);
        }
        flat += 1;

        let mut carry = true;
        for dim in (0..d).rev() {
            if carry {
                multi_idx[dim] += 1;
                if multi_idx[dim] < kron_result.marginal_dims[dim] {
                    carry = false;
                } else {
                    multi_idx[dim] = 0;
                }
            }
        }
        if carry {
            break;
        }
    }

    PirlsPenalty::Diagonal {
        diag,
        positive_indices,
        linear_shift: Array1::zeros(p),
        constant_shift: 0.0,
        prior_mean_target: Array1::zeros(p),
    }
}

pub(super) fn canonical_prior_shift(
    penalties: &[crate::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> (Array1<f64>, f64) {
    let mut linear = Array1::<f64>::zeros(p);
    let mut constant = 0.0;
    for (idx, cp) in penalties.iter().enumerate() {
        let Some(&lambda) = lambdas.get(idx) else {
            continue;
        };
        if lambda == 0.0 {
            continue;
        }
        linear += &cp.prior_linear_shift(lambda);
        constant += cp.prior_constant_shift(lambda);
    }
    (linear, constant)
}

/// Aggregate prior-mean target across canonical penalty blocks: the sum of
/// each block's `full_width_prior_mean()`. Used by the PIRLS solve sites
/// that add a fixed stabilization ridge `δI` to the penalized Hessian — they
/// must also add `δ · prior_mean_target` to the RHS to keep `β = μ` recovery
/// exact when the data carries no information (X'WX = 0). Equivalent to
/// `canonical_prior_shift` with all λ = 1 and dropping `S_k` from the linear
/// piece (i.e., raw μ rather than `S_k μ`). Returned in the *original*
/// coordinates; callers transform if needed.
pub(super) fn canonical_prior_mean_aggregate(
    penalties: &[crate::construction::CanonicalPenalty],
    p: usize,
) -> Array1<f64> {
    let mut mean = Array1::<f64>::zeros(p);
    for cp in penalties {
        mean += &cp.full_width_prior_mean();
    }
    mean
}


pub struct PirlsProblem<'a, X> {
    pub x: X,
    pub offset: ArrayView1<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub priorweights: ArrayView1<'a, f64>,
    pub covariate_se: Option<ArrayView1<'a, f64>>,
    /// When set, the inner PLS solver reuses the precomputed `XᵀWX` and
    /// `XᵀW(y − offset)` in *original* coordinates instead of streaming the
    /// O(N·p²) GEMM and the O(N·p) matvec on every outer REML iteration.
    ///
    /// Valid only when the family is Gaussian + Identity link, prior weights
    /// are constant across outer iterations (always true in the REML outer
    /// loop), no Firth bias reduction, and no inequality / lower-bound
    /// constraints (matching the existing Identity short-circuit at
    /// `pirls.rs:6237`). The penalty `λ·S` is still added per-λ on top of
    /// the cached `XᵀWX`.
    pub gaussian_fixed_cache: Option<&'a GaussianFixedCache>,
}

pub struct PenaltyConfig<'a> {
    /// Block-local canonical penalties with precomputed roots and spectral data.
    /// This is the single canonical penalty representation — no full-width
    /// `rank × p` roots are stored. When the reparameterization engine needs
    /// full-width roots, they are derived on-the-fly from these block-local roots.
    pub canonical_penalties: &'a [crate::construction::CanonicalPenalty],
    pub balanced_penalty_root: Option<&'a Array2<f64>>,
    pub reparam_invariant: Option<&'a crate::construction::ReparamInvariant>,
    pub p: usize,
    pub coefficient_lower_bounds: Option<&'a Array1<f64>>,
    pub linear_constraints_original: Option<&'a LinearInequalityConstraints>,
    /// Relative shrinkage floor for eigenvalues of the penalized block.
    /// If `Some(epsilon)`, a rho-independent ridge of `epsilon * max_balanced_eigenvalue`
    /// is added to prevent barely-penalized directions from causing pathological
    /// non-Gaussianity in the posterior. Typical value: `1e-6`. `None` disables.
    pub penalty_shrinkage_floor: Option<f64>,
    /// When set, the penalties have Kronecker (tensor-product) structure.
    /// The reparameterization engine will use factored Qs = U_1 ⊗ ... ⊗ U_d
    /// instead of eigendecomposing the full p×p balanced penalty.
    pub kronecker_factored: Option<&'a crate::basis::KroneckerFactoredBasis>,
}

/// P-IRLS solver that follows mgcv's architecture exactly
///
/// This function implements the complete algorithm from mgcv's gam.fit3 function
/// for fitting a GAM model with a fixed set of smoothing parameters:
///
/// - Perform stable reparameterization ONCE at the beginning (mgcv's gam.reparam)
/// - Transform the design matrix into this stable basis
/// - Extract a single penalty square root from the transformed penalty
/// - Run the P-IRLS loop entirely in the transformed basis
/// - Transform the coefficients back to the original basis only when returning
/// - Reuse a cached balanced penalty root when available to avoid repeated eigendecompositions
///
/// This architecture ensures optimal numerical stability throughout the entire
/// fitting process by working in a well-conditioned parameter space.
pub fn fit_model_for_fixed_rho<'a, X: Into<DesignMatrix> + Clone>(
    rho: LogSmoothingParamsView<'_>,
    problem: PirlsProblem<'a, X>,
    penalty: PenaltyConfig<'_>,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    fit_model_for_fixed_rho_with_adaptive_kkt(rho, problem, penalty, config, warm_start_beta, None)
}

pub(crate) fn fit_model_for_fixed_rho_with_adaptive_kkt<'a, X: Into<DesignMatrix> + Clone>(
    rho: LogSmoothingParamsView<'_>,
    problem: PirlsProblem<'a, X>,
    penalty: PenaltyConfig<'_>,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
    adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let PirlsProblem {
        x,
        offset,
        y,
        priorweights,
        covariate_se,
        gaussian_fixed_cache,
    } = problem;
    let quadctx = crate::quadrature::QuadratureContext::new();
    let lambdas = rho.exp();
    let lambdas_slice = lambdas.as_slice_memory_order().ok_or_else(|| {
        EstimationError::InvalidInput("non-contiguous lambda storage".to_string())
    })?;

    let likelihood = &config.likelihood;
    let link_function = config.link_function();

    use crate::construction::{
        EngineDims, create_balanced_penalty_root_from_canonical,
        stable_reparameterization_engine_canonical,
    };

    let eb_cow: Cow<'_, Array2<f64>> = if let Some(precomputed) = penalty.balanced_penalty_root {
        Cow::Borrowed(precomputed)
    } else {
        Cow::Owned(create_balanced_penalty_root_from_canonical(
            penalty.canonical_penalties,
            penalty.p,
        )?)
    };
    let eb: &Array2<f64> = eb_cow.as_ref();

    // Build a cheap weighted penalty sum for the sparse-native decision
    // WITHOUT running the expensive eigendecomposition engine.
    // The full reparameterization is deferred until we know which path we need.
    let cheap_s_lambda: Option<Array2<f64>> = if penalty.kronecker_factored.is_none() {
        let mut s = Array2::<f64>::zeros((penalty.p, penalty.p));
        for (k, cp) in penalty.canonical_penalties.iter().enumerate() {
            let lam = lambdas_slice.get(k).copied().unwrap_or(0.0);
            if lam != 0.0 {
                cp.accumulate_weighted(&mut s, lam);
            }
        }
        Some(s)
    } else {
        None
    };
    let kronecker_runtime = if let Some(kron) = penalty.kronecker_factored {
        let kron_result = crate::construction::kronecker_reparameterization_engine(
            &kron.marginal_designs,
            &kron.marginal_penalties,
            &kron.marginal_dims,
            lambdas_slice,
            kron.has_double_penalty,
            penalty.penalty_shrinkage_floor,
        )?;
        let transform = Arc::new(KroneckerQsTransform::new(&kron_result));
        let penalty_diag = build_diagonal_penalty_from_kronecker(&kron_result, lambdas_slice);
        Some((kron_result, transform, penalty_diag))
    } else {
        None
    };
    // Constraint transformation is deferred until after the sparse-native
    // decision, because the dense reparameterization engine (which provides Qs)
    // is now run lazily.  Kronecker constraints can be built eagerly since
    // the Kronecker transform is already available.
    let kronecker_constraints = if let Some((_, transform, _)) = kronecker_runtime.as_ref() {
        let tb = build_transformed_lower_bound_constraints_with_transform(
            &WorkingReparamTransform::Kronecker(Arc::clone(transform)),
            penalty.coefficient_lower_bounds,
        );
        let tl = build_transformed_linear_constraints_with_transform(
            &WorkingReparamTransform::Kronecker(Arc::clone(transform)),
            penalty.linear_constraints_original,
        );
        Some(merge_linear_constraints(tb, tl))
    } else {
        None
    };

    let x_original: DesignMatrix = x.into();
    // Auto-detect sparse structure in dense designs so the sparse-native path
    // can engage for structurally sparse models that happen to be stored dense.
    let x_original = {
        let auto_sparse = x_original
            .as_dense()
            .and_then(|dense| sparse_from_denseview(dense.view()));
        auto_sparse.unwrap_or(x_original)
    };
    let ebrows = eb.nrows();
    let erows = if let Some((_, _, penalty_diag)) = kronecker_runtime.as_ref() {
        penalty_diag.rank()
    } else {
        // Compute penalty root rank cheaply from canonical penalties.
        penalty
            .canonical_penalties
            .iter()
            .map(|cp| cp.rank())
            .sum::<usize>()
    };
    let mut workspace = PirlsWorkspace::new(x_original.nrows(), x_original.ncols(), ebrows, erows);
    let solver_decision = if let Some((_, _, _)) = kronecker_runtime.as_ref() {
        SparsePirlsDecision {
            path: PirlsLinearSolvePath::DenseTransformed,
            reason: "kronecker_runtime",
            p: x_original.ncols(),
            nnz_x: 0,
            nnz_xtwx_symbolic: None,
            nnz_s_lambda: 0,
            nnz_h_est: None,
            density_h_est: None,
        }
    } else {
        should_use_sparse_native_pirls(
            &mut workspace,
            &x_original,
            cheap_s_lambda
                .as_ref()
                .expect("cheap_s_lambda should be present outside Kronecker path"),
            penalty.coefficient_lower_bounds,
            penalty.linear_constraints_original,
        )
    };
    solver_decision.log_once();

    let use_sparse_native = matches!(solver_decision.path, PirlsLinearSolvePath::SparseNative);

    // Run the expensive eigendecomposition engine ONLY for the dense-transformed
    // path. Sparse-native fits skip this entirely during the PIRLS solve.
    let dense_reparam_result = if !use_sparse_native && penalty.kronecker_factored.is_none() {
        Some(stable_reparameterization_engine_canonical(
            penalty.canonical_penalties,
            lambdas_slice,
            EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
            penalty.reparam_invariant,
            penalty.penalty_shrinkage_floor,
        )?)
    } else {
        None
    };
    let qs_arc = dense_reparam_result
        .as_ref()
        .map(|reparam_result| Arc::new(reparam_result.qs.clone()));
    let transform_active = if let Some((_, transform, _)) = kronecker_runtime.as_ref() {
        Some(WorkingReparamTransform::Kronecker(Arc::clone(transform)))
    } else if use_sparse_native {
        None
    } else {
        Some(WorkingReparamTransform::Dense(Arc::clone(
            qs_arc
                .as_ref()
                .expect("dense Qs should exist for non-Kronecker transformed path"),
        )))
    };
    let mut penalty_active = if let Some((_, _, penalty_diag)) = kronecker_runtime.as_ref() {
        penalty_diag.clone()
    } else if use_sparse_native {
        // Build sparse-native penalty directly from canonical penalties.
        // No dense eigendecomposition needed for the PIRLS solve itself.
        let s_lambda = cheap_s_lambda
            .as_ref()
            .expect("cheap_s_lambda should be present for sparse-native path")
            .clone();
        let e_root = stack_lambdaweighted_penalty_root_canonical(
            penalty.canonical_penalties,
            lambdas_slice,
            penalty.p,
        );
        PirlsPenalty::Dense {
            s_transformed: s_lambda,
            e_transformed: e_root,
            linear_shift: Array1::zeros(penalty.p),
            constant_shift: 0.0,
            prior_mean_target: Array1::zeros(penalty.p),
        }
    } else {
        let dense = dense_reparam_result
            .as_ref()
            .expect("dense reparam result should be present outside Kronecker path");
        PirlsPenalty::Dense {
            s_transformed: dense.s_transformed.clone(),
            e_transformed: dense.e_transformed.clone(),
            linear_shift: Array1::zeros(penalty.p),
            constant_shift: 0.0,
            prior_mean_target: Array1::zeros(penalty.p),
        }
    };
    let (shift_original, shift_constant) =
        canonical_prior_shift(penalty.canonical_penalties, lambdas_slice, penalty.p);
    let shift_active = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(&shift_original))
        .unwrap_or(shift_original);
    let prior_mean_original =
        canonical_prior_mean_aggregate(penalty.canonical_penalties, penalty.p);
    let prior_mean_active = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(&prior_mean_original))
        .unwrap_or(prior_mean_original);
    attach_penalty_shift(
        &mut penalty_active,
        shift_active,
        shift_constant,
        prior_mean_active,
    );
    // Build transformed constraints now that dense_reparam_result is available.
    let linear_constraints = if let Some(kc) = kronecker_constraints {
        kc
    } else if let Some(reparam) = dense_reparam_result.as_ref() {
        let tb = build_transformed_lower_bound_constraints(
            &reparam.qs,
            penalty.coefficient_lower_bounds,
        );
        let tl =
            build_transformed_linear_constraints(&reparam.qs, penalty.linear_constraints_original);
        merge_linear_constraints(tb, tl)
    } else {
        // Sparse-native without dense reparam: constraints stay in original
        // coordinates (identity Qs).  Use an identity matrix of appropriate size.
        let p = penalty.p;
        let qs_identity = Array2::<f64>::eye(p);
        let tb = build_transformed_lower_bound_constraints(
            &qs_identity,
            penalty.coefficient_lower_bounds,
        );
        let tl =
            build_transformed_linear_constraints(&qs_identity, penalty.linear_constraints_original);
        merge_linear_constraints(tb, tl)
    };

    let coordinate_frame = if use_sparse_native {
        PirlsCoordinateFrame::OriginalSparseNative
    } else {
        PirlsCoordinateFrame::TransformedQs
    };
    let materialize_final_reparam_result = || -> Result<ReparamResult, EstimationError> {
        if let Some((kron_result, _, _)) = kronecker_runtime.as_ref() {
            let rs_list: Vec<Array2<f64>> = penalty
                .canonical_penalties
                .iter()
                .map(|cp| cp.full_width_root())
                .collect();
            kron_result.materialize_dense_artifact_result(&rs_list, lambdas_slice, penalty.p)
        } else if use_sparse_native {
            // Sparse-native path: run the eigendecomposition engine now (deferred
            // from the PIRLS solve) to produce the REML-required log-determinant
            // and derivative quantities, then override with identity Qs.
            let base = stable_reparameterization_engine_canonical(
                penalty.canonical_penalties,
                lambdas_slice,
                EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
                penalty.reparam_invariant,
                penalty.penalty_shrinkage_floor,
            )?;
            Ok(build_sparse_native_reparam_result(
                base,
                penalty.canonical_penalties,
                lambdas_slice,
                penalty.p,
            ))
        } else {
            Ok(dense_reparam_result
                .as_ref()
                .expect("dense reparam result should be present outside Kronecker path")
                .clone())
        }
    };

    // Stage 3.3-GI: GPU exact PLS dispatch — see gpu_dispatch::try_gaussian_pls_gpu.
    if let Some(result) = try_gaussian_pls_gpu(
        link_function,
        config,
        penalty.coefficient_lower_bounds,
        penalty.linear_constraints_original,
        gaussian_fixed_cache,
        &penalty_active,
        &qs_arc,
        &x_original,
        use_sparse_native,
        penalty.p,
        || materialize_final_reparam_result(),
        y,
        priorweights,
        offset,
        coordinate_frame,
        &linear_constraints,
    ) {
        return result;
    }

    if matches!(link_function, LinkFunction::Identity) {
        // Apply the Gaussian-Identity fixed-data cache only when every
        // precondition for the short-circuit's exact reuse holds: the family
        // really is Gaussian (z = y), there is no Firth bias-reduction term,
        // no coefficient lower bounds, and no linear inequality constraints
        // — anything that would change the right-hand side or the system
        // beyond the additive penalty would invalidate the cache.
        let cache_eligible = gaussian_fixed_cache.is_some()
            && likelihood.spec.is_gaussian_identity()
            && !config.firth_bias_reduction
            && penalty.coefficient_lower_bounds.is_none()
            && penalty.linear_constraints_original.is_none();
        let cache_for_solve = if cache_eligible {
            gaussian_fixed_cache
        } else {
            None
        };
        let (pls_result, _) = solve_penalized_least_squares_implicit(
            &x_original,
            transform_active.as_ref(),
            y,
            priorweights,
            offset,
            &penalty_active,
            &mut workspace,
            y,
            link_function,
            cache_for_solve,
        )?;

        let beta_transformed = pls_result.beta;
        let penalized_hessian = pls_result.penalized_hessian;
        let edf = pls_result.edf;
        let baseridge = pls_result.ridge_used;

        let priorweights_owned = priorweights.to_owned();
        // eta = offset + X Qs beta (composed, no materialization)
        let qbeta = transform_active
            .as_ref()
            .map(|transform| transform.apply(beta_transformed.as_ref()))
            .unwrap_or_else(|| beta_transformed.as_ref().clone());
        let mut eta = offset.to_owned();
        eta += &x_original.apply(&qbeta);
        let final_eta = eta.clone();
        let finalmu = eta.clone();
        let finalz = y.to_owned();

        let mut weighted_residual = finalmu.clone();
        weighted_residual -= &finalz;
        weighted_residual *= &priorweights_owned;
        // gradient = Qs^T X^T (w * residual) (composed)
        let xt_wr = x_original.apply_transpose(&weighted_residual);
        let gradient_data = transform_active
            .as_ref()
            .map(|transform| transform.apply_transpose(&xt_wr))
            .unwrap_or(xt_wr);
        let score_norm = array1_l2_norm(&gradient_data);
        let s_beta = penalty_active.shifted_gradient(beta_transformed.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        let mut gradient = gradient_data;
        gradient += &s_beta;
        let mut penalty_term = penalty_active.shifted_quadratic(beta_transformed.as_ref());
        let deviance = calculate_deviance(y, &finalmu, likelihood, priorweights);
        let ridge_used = baseridge;
        let stabilizedhessian = if ridge_used > 0.0 {
            penalized_hessian
                .addridge(ridge_used)
                .map_err(|e| EstimationError::InvalidInput(format!("ridge addition failed: {e}")))?
        } else {
            penalized_hessian.clone()
        };
        let mut ridge_grad_norm = 0.0;
        if ridge_used > 0.0 {
            let ridge_penalty =
                ridge_used * beta_transformed.as_ref().dot(beta_transformed.as_ref());
            penalty_term += ridge_penalty;
            gradient += &beta_transformed.as_ref().mapv(|v| ridge_used * v);
            ridge_grad_norm = ridge_used * array1_l2_norm(beta_transformed.as_ref());
        }

        let gradient_norm = array1_l2_norm(&gradient);
        let max_abs_eta = inf_norm(finalmu.iter().copied());
        let log_likelihood =
            calculate_loglikelihood_omitting_constants(y, &finalmu, likelihood, priorweights);

        let working_state = WorkingState {
            eta: LinearPredictor::new(finalmu.clone()),
            gradient: gradient.clone(),
            hessian: penalized_hessian.clone(),

            log_likelihood,
            deviance,
            penalty_term,
            firth: FirthDiagnostics::Inactive,
            ridge_used,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: score_norm + s_beta_norm + ridge_grad_norm,
        };

        let zero_iter_penalized = deviance + penalty_term;
        let working_summary = WorkingModelPirlsResult {
            beta: beta_transformed.clone(),
            state: working_state,
            status: PirlsStatus::Converged,
            iterations: 1,
            lastgradient_norm: gradient_norm,
            last_deviance_change: 0.0,
            last_step_size: 1.0,
            last_step_halving: 0,
            max_abs_eta,
            constraint_kkt: linear_constraints.as_ref().map(|lin| {
                compute_constraint_kkt_diagnostics(beta_transformed.as_ref(), &gradient, lin)
            }),
            min_penalized_deviance: if zero_iter_penalized.is_finite() {
                zero_iter_penalized
            } else {
                f64::INFINITY
            },
            // Zero-iteration synthesis: no LM damping was exercised, so
            // hand the next solve the cold default.
            final_lm_lambda: 1e-6,
            // Zero-iteration synthesis: no LM gain ratio was measured.
            final_accept_rho: None,
            // Zero-iteration synthesis assembles the Hessian with prior
            // weights only; no observed-information re-evaluation has
            // happened. Label honestly as a Fisher-type surrogate so
            // outer Laplace consumers see the truth.
            exported_laplace_curvature: ExportedLaplaceCurvature::ExpectedInformationSurrogate,
        };

        let (solve_c_array, solve_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                &config.likelihood,
                &config.link_kind,
                &final_eta,
                priorweights_owned.view(),
            )?;
        let reparam_result = materialize_final_reparam_result()?;
        let qs_arc_final = Arc::new(reparam_result.qs.clone());
        let pirls_result = PirlsResult {
            likelihood: config.likelihood.clone(),
            beta_transformed,
            penalized_hessian_transformed: penalized_hessian,
            stabilizedhessian_transformed: stabilizedhessian,
            ridge_passport: RidgePassport::scaled_identity(
                ridge_used,
                RidgePolicy::explicit_stabilization_full(),
            ),
            ridge_used,
            deviance,
            edf,
            stable_penalty_term: penalty_term,
            firth: FirthDiagnostics::Inactive,
            finalweights: priorweights_owned.clone(),
            final_offset: offset.to_owned(),
            final_eta: final_eta.clone(),
            finalmu: finalmu.clone(),
            solveweights: priorweights_owned,
            solveworking_response: finalz.clone(),
            solvemu: finalmu.clone(),
            solve_dmu_deta,
            solve_d2mu_deta2,
            solve_d3mu_deta3,
            solve_c_array,
            solve_d_array,
            derivatives_unsupported: false,
            status: PirlsStatus::Converged,
            iteration: 1,
            max_abs_eta,
            lastgradient_norm: gradient_norm,
            gradient_natural_scale: score_norm + s_beta_norm + ridge_grad_norm,
            last_deviance_change: 0.0,
            last_step_halving: 0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            exported_laplace_curvature: working_summary.exported_laplace_curvature.clone(),
            final_lm_lambda: working_summary.final_lm_lambda,
            final_accept_rho: working_summary.final_accept_rho,
            constraint_kkt: working_summary.constraint_kkt.clone(),
            linear_constraints_transformed: linear_constraints.clone(),
            reparam_result,
            x_transformed: make_reparam_operator(&x_original, &qs_arc_final, use_sparse_native),
            coordinate_frame,
            cache_compacted: false,
            min_penalized_deviance: working_summary.min_penalized_deviance,
        };

        return Ok((pirls_result, working_summary));
    }

    let x_original_for_result = x_original.clone();
    let mut working_model = GamWorkingModel::new(
        None, // No pre-materialized x_transformed: use implicit Qs composition
        x_original.clone(),
        coordinate_frame,
        offset,
        y,
        priorweights,
        penalty_active.clone(),
        workspace,
        config.likelihood.clone(),
        config.link_kind.clone(),
        config.firth_bias_reduction
            && matches!(
                &config.link_kind,
                InverseLink::Standard(StandardLink::Logit)
            ),
        transform_active.clone(),
        quadctx,
    );

    // Apply integrated (GHQ) likelihood if per-observation SE is provided.
    // This is used by the calibrator to coherently account for base prediction uncertainty.
    if let Some(se) = covariate_se {
        working_model = working_model.with_covariate_se(se.to_owned());
    }

    let mut beta_guess_original = warm_start_beta
        .filter(|beta| beta.len() == penalty.p)
        .map(|beta| beta.to_owned())
        .unwrap_or_else(|| {
            Coefficients::new(default_beta_guess_external(
                penalty.p,
                link_function,
                y,
                priorweights,
                config.link_kind.mixture_state(),
                config.link_kind.sas_state(),
            ))
        });
    if let Some(lb) = penalty.coefficient_lower_bounds {
        project_coefficients_to_lower_bounds(&mut beta_guess_original.0, lb);
    }
    let initial_beta = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(beta_guess_original.as_ref()))
        .unwrap_or_else(|| beta_guess_original.as_ref().clone());
    let initial_beta = if let Some(constraints) = linear_constraints.as_ref() {
        let current_violation = constraints
            .a
            .dot(&initial_beta)
            .iter()
            .zip(constraints.b.iter())
            .map(|(lhs, rhs)| (rhs - lhs).max(0.0))
            .fold(0.0_f64, f64::max);
        if current_violation > 1e-8 {
            active_set::feasible_point_for_linear_constraints(constraints, initial_beta.len())
                .unwrap_or(initial_beta)
        } else {
            initial_beta
        }
    } else {
        initial_beta
    };
    let firth_active = config.firth_bias_reduction && matches!(link_function, LinkFunction::Logit);
    let base_max_step_halving = if firth_active { 60 } else { 30 };
    let options = WorkingModelPirlsOptions {
        // Firth logit fits often need more inner iterations to settle.
        max_iterations: if firth_active {
            config.max_iterations.max(200)
        } else {
            config.max_iterations
        },
        convergence_tolerance: config.convergence_tolerance,
        adaptive_kkt_tolerance,
        // LM step-halving is a per-iteration damping retry budget; it is
        // independent of the total outer-iteration cap. Tying the two
        // together collapsed step halving to 3 under seed screening (where
        // max_iterations is intentionally capped low), turning recoverable
        // damping into spurious failures.
        max_step_halving: base_max_step_halving,
        min_step_size: if firth_active { 1e-12 } else { 1e-10 },
        firth_bias_reduction: firth_active,
        coefficient_lower_bounds: None,
        linear_constraints: linear_constraints.clone(),
        initial_lm_lambda: config.initial_lm_lambda,
        geodesic_acceleration: config.geodesic_acceleration,
        arrow_schur: config.arrow_schur.clone(),
    };

    let mut iteration_logger = |info: &WorkingModelIterationInfo| {
        log::debug!(
            "[PIRLS] iter {:>3} | deviance {:.6e} | |grad| {:.3e} | step {:.3e} (halving {})",
            info.iteration,
            info.deviance,
            info.gradient_norm,
            info.step_size,
            info.step_halving
        );
    };

    // Stage 3.3 GPU PIRLS-loop dispatch — see gpu_dispatch::try_pirls_loop_gpu.
    if let Some(result) = try_pirls_loop_gpu(
        config,
        &penalty_active,
        kronecker_runtime.is_none(),
        use_sparse_native,
        &linear_constraints,
        &x_original,
        &qs_arc,
        penalty.p,
        &x_original_for_result,
        || materialize_final_reparam_result(),
        y,
        priorweights,
        offset,
        &initial_beta,
        link_function,
        coordinate_frame,
    ) {
        return result;
    }

    let mut working_summary = runworking_model_pirls(
        &mut working_model,
        Coefficients::new(initial_beta),
        &options,
        &mut iteration_logger,
    )?;

    // Extract workspace before consuming working_model so we can reuse
    // the pre-allocated buffers in calculate_edfwithworkspace_with_penalty.
    // into_final_state() drops the workspace field anyway (it uses `..` in
    // its destructure); we replace it with a zero-sized stub to satisfy the
    // borrow checker, then keep the real workspace alive for the EDF call.
    let mut saved_workspace = std::mem::replace(
        &mut working_model.workspace,
        PirlsWorkspace::new(0, 0, 0, 0),
    );
    let final_state = working_model.into_final_state();
    let GamModelFinalState {
        likelihood: final_likelihood,
        coordinate_frame,
        finalmu,
        finalweights,
        scoreweights,
        finalz,
        final_c,
        final_d,
        final_dmu_deta,
        final_d2mu_deta2,
        final_d3mu_deta3,
        penalty_term,
        ..
    } = final_state;

    // Preserve the Hessian as-is (sparse or dense) — no densification.
    // P-IRLS already folded any stabilization ridge directly into the Hessian.
    // Keep that exact matrix so outer LAML derivatives stay consistent:
    // H_eff = X'W_H X + S_λ + ridge I (if ridge_used > 0).
    let penalized_hessian_transformed = working_summary.state.hessian.clone();
    let stabilizedhessian_transformed = penalized_hessian_transformed.clone();
    // Use the workspace-backed variant for the dense path to reuse the
    // `final_aug_matrix` allocation; the sparse path still allocates
    // internally because no pre-computed factor is available at this site.
    let mut edf = if let Some(dense_h) = penalized_hessian_transformed.as_dense() {
        calculate_edfwithworkspace_with_penalty(dense_h, &penalty_active, &mut saved_workspace)?
    } else {
        calculate_edf_with_penalty(&penalized_hessian_transformed, &penalty_active)?
    };
    if !edf.is_finite() || edf.is_nan() {
        let p = penalized_hessian_transformed.ncols() as f64;
        let r = penalty_active.rank() as f64;
        edf = (p - r).max(0.0);
    }

    // Outer rescue: a fit that hit max-iterations may still be a usable
    // minimum if progress has effectively stopped (deviance plateaued or
    // step size collapsed to the floor) AND the projected gradient is in
    // the near-stationary band under the scale-invariant certificate.
    // Same logic for non-Firth and Firth paths; firth_active just gates
    // the second pass.
    let stalled_at_valid_minimum = |summary: &WorkingModelPirlsResult| -> bool {
        let dev_scale = summary.state.deviance.abs().max(1.0);
        // Progress plateau uses the fixed solver tolerance; only the KKT band below adapts.
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        let progress_stopped =
            summary.last_deviance_change.abs() <= dev_tol || summary.last_step_size <= step_floor;
        let near_stationary = summary
            .state
            .near_stationary_kkt(summary.lastgradient_norm, effective_kkt_tolerance(&options));
        progress_stopped && near_stationary
    };

    let mut status = working_summary.status;
    if status.is_failed_max_iterations() && stalled_at_valid_minimum(&working_summary) {
        status = PirlsStatus::StalledAtValidMinimum;
        working_summary.status = status;
    }
    if status.is_failed_max_iterations()
        && firth_active
        && stalled_at_valid_minimum(&working_summary)
    {
        // Firth-adjusted fits can stall; accept under the same dual-criterion
        // near-stationary band.
        status = PirlsStatus::StalledAtValidMinimum;
        working_summary.status = status;
    }
    let has_penalty = penalty_active.rank() > 0;
    let firth_active = options.firth_bias_reduction;
    if detect_logit_instability(
        link_function,
        has_penalty,
        firth_active,
        &working_summary,
        &finalmu,
        &finalweights,
        y,
    ) {
        status = PirlsStatus::Unstable;
        working_summary.status = status;
    }

    // Store a lazy ReparamOperator instead of materializing X·Qs.
    // Consumers that truly need dense access can call .to_dense() on demand.
    let reparam_result_final = materialize_final_reparam_result()?;
    let qs_arc_final = Arc::new(reparam_result_final.qs.clone());
    let x_transformed_final =
        make_reparam_operator(&x_original_for_result, &qs_arc_final, use_sparse_native);

    let pirls_result = assemble_pirls_result(
        &working_summary,
        final_likelihood,
        offset,
        penalized_hessian_transformed,
        stabilizedhessian_transformed,
        edf,
        penalty_term,
        &finalmu,
        &finalweights,
        &scoreweights,
        &finalz,
        &final_c,
        &final_d,
        &final_dmu_deta,
        &final_d2mu_deta2,
        &final_d3mu_deta3,
        status,
        reparam_result_final,
        x_transformed_final,
        coordinate_frame,
        linear_constraints,
    );

    Ok((pirls_result, working_summary))
}

#[derive(Clone)]
pub struct PirlsConfig {
    pub likelihood: GlmLikelihoodSpec,
    pub link_kind: InverseLink,
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub firth_bias_reduction: bool,
    /// Optional warm-start hint for `WorkingModelPirlsOptions::initial_lm_lambda`.
    /// Forwarded directly when `fit_model_for_fixed_rho` builds its
    /// internal options. See the field doc on `WorkingModelPirlsOptions`
    /// for the seeding semantics.
    pub initial_lm_lambda: Option<f64>,
    /// Enable the Transtrum-Sethna geodesic-acceleration second-order
    /// correction on each accepted LM step. Forwarded to
    /// `WorkingModelPirlsOptions::geodesic_acceleration`; see that
    /// field's doc for the full semantics and cost model. Default
    /// `false`; opt-in until validated.
    pub geodesic_acceleration: bool,
    /// Optional arrow-Schur structured-inner-solve descriptor. When
    /// `Some`, forwarded to `WorkingModelPirlsOptions::arrow_schur` so
    /// each accepted LM step is solved by the per-observation
    /// arrow-Schur path
    /// ([`crate::solver::arrow_schur::ArrowSchurSystem`]). When `None`
    /// (the default), the existing β-only path is used unchanged.
    ///
    /// See `proposals/latent_coord.md` for the design and the math
    /// audit caveats; see [`ArrowSchurInnerConfig`] for the closure
    /// contract.
    pub arrow_schur: Option<ArrowSchurInnerConfig>,
}

impl PirlsConfig {
    #[inline]
    pub fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }
}

#[inline]
pub(super) fn max_symmetric_asymmetry(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows().min(matrix.ncols());
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
            if diff > max_asym {
                max_asym = diff;
            }
        }
    }
    max_asym
}

#[inline]
pub(super) fn assert_symmetric_tol(matrix: &Array2<f64>, label: &str, tol: f64) {
    let max_asym = max_symmetric_asymmetry(matrix);
    assert!(
        max_asym <= tol,
        "{} asymmetry too large: {:.3e} (tol {:.3e})",
        label,
        max_asym,
        tol
    );
}

/// Build a DesignMatrix wrapping a lazy ReparamOperator (or the original for sparse-native).
pub(super) fn make_reparam_operator(
    x_original: &DesignMatrix,
    qs_arc: &Arc<Array2<f64>>,
    use_sparse_native: bool,
) -> DesignMatrix {
    if use_sparse_native {
        x_original.clone()
    } else {
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
            ReparamOperator::new(x_original.clone(), Arc::clone(qs_arc)),
        )))
    }
}

// solve_penalized_least_squares_implicit lives in pls_solver (imported above).

pub(super) fn build_transformed_lower_bound_constraints(
    qs: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Option<LinearInequalityConstraints> {
    let lb = coefficient_lower_bounds?;
    if lb.len() != qs.nrows() {
        return None;
    }
    let activerows: Vec<usize> = (0..lb.len()).filter(|&i| lb[i].is_finite()).collect();
    if activerows.is_empty() {
        return None;
    }
    let mut a = Array2::<f64>::zeros((activerows.len(), qs.ncols()));
    let mut b = Array1::<f64>::zeros(activerows.len());
    for (r, &idx) in activerows.iter().enumerate() {
        a.row_mut(r).assign(&qs.row(idx));
        b[r] = lb[idx];
    }
    Some(LinearInequalityConstraints::from_paired(a, b))
}

pub(super) fn build_transformed_lower_bound_constraints_with_transform(
    transform: &WorkingReparamTransform,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Option<LinearInequalityConstraints> {
    let lb = coefficient_lower_bounds?;
    let p = match transform {
        WorkingReparamTransform::Dense(qs) => qs.nrows(),
        WorkingReparamTransform::Kronecker(kron) => kron.p,
    };
    if lb.len() != p {
        return None;
    }
    let activerows: Vec<usize> = (0..lb.len()).filter(|&i| lb[i].is_finite()).collect();
    if activerows.is_empty() {
        return None;
    }
    let mut a = Array2::<f64>::zeros((activerows.len(), p));
    let mut b = Array1::<f64>::zeros(activerows.len());
    for (r, &idx) in activerows.iter().enumerate() {
        let mut basis = Array1::<f64>::zeros(p);
        basis[idx] = 1.0;
        let row = transform.apply_transpose(&basis);
        a.row_mut(r).assign(&row);
        b[r] = lb[idx];
    }
    Some(LinearInequalityConstraints::from_paired(a, b))
}

pub(super) fn build_transformed_linear_constraints(
    qs: &Array2<f64>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    let lc = linear_constraints?;
    if lc.a.ncols() != qs.nrows() {
        return None;
    }
    Some(LinearInequalityConstraints::from_paired(
        lc.a.dot(qs),
        lc.b.clone(),
    ))
}

pub(super) fn build_transformed_linear_constraints_with_transform(
    transform: &WorkingReparamTransform,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    let lc = linear_constraints?;
    let p = match transform {
        WorkingReparamTransform::Dense(qs) => qs.nrows(),
        WorkingReparamTransform::Kronecker(kron) => kron.p,
    };
    if lc.a.ncols() != p {
        return None;
    }
    let mut a = Array2::<f64>::zeros((lc.a.nrows(), p));
    for row in 0..lc.a.nrows() {
        let transformed = transform.apply_transpose(&lc.a.row(row).to_owned());
        a.row_mut(row).assign(&transformed);
    }
    Some(LinearInequalityConstraints { a, b: lc.b.clone() })
}

pub(super) fn merge_linear_constraints(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    match (first, second) {
        (None, None) => None,
        (Some(c), None) | (None, Some(c)) => Some(c),
        (Some(c1), Some(c2)) => {
            if c1.a.ncols() != c2.a.ncols() {
                return None;
            }
            let rows = c1.a.nrows() + c2.a.nrows();
            let cols = c1.a.ncols();
            let mut a = Array2::<f64>::zeros((rows, cols));
            a.slice_mut(s![0..c1.a.nrows(), ..]).assign(&c1.a);
            a.slice_mut(s![c1.a.nrows()..rows, ..]).assign(&c2.a);
            let mut b = Array1::<f64>::zeros(rows);
            b.slice_mut(s![0..c1.b.len()]).assign(&c1.b);
            b.slice_mut(s![c1.b.len()..rows]).assign(&c2.b);
            Some(LinearInequalityConstraints { a, b })
        }
    }
}

pub(super) fn sparse_from_denseview(x: ArrayView2<f64>) -> Option<DesignMatrix> {
    let nrows = x.nrows();
    let ncols = x.ncols();
    if nrows == 0 || ncols == 0 {
        return None;
    }
    // Narrow matrices are faster in dense form; avoid any sparsity scan overhead.
    if ncols <= 32 {
        return None;
    }

    const ZERO_EPS: f64 = 1e-12;
    let total = nrows.saturating_mul(ncols);
    if total == 0 {
        return None;
    }
    // If a matrix exceeds this nnz count it is too dense for sparse path; bail early.
    let sparse_nnz_limit = ((total as f64) * 0.20).floor() as usize;
    let mut nnz = 0usize;
    for &val in x.iter() {
        if val.abs() > ZERO_EPS {
            nnz += 1;
            if nnz > sparse_nnz_limit {
                return None;
            }
        }
    }
    let mut triplets = Vec::with_capacity(nnz);
    for (row_idx, row) in x.outer_iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            if val.abs() > ZERO_EPS {
                triplets.push(Triplet::new(row_idx, col_idx, val));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .ok()
        .map(DesignMatrix::from)
}
