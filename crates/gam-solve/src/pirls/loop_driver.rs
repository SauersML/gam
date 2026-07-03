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
//!   `crate::gpu::pirls_dispatch_wire`.

use super::{
    // state re-exports
    AdaptiveKktTolerance,
    ExportedLaplaceCurvature,
    FirthDiagnostics,
    GamWorkingModel,
    HessianCurvatureKind,
    // penalty types
    KroneckerQsTransform,
    LinearInequalityConstraints,
    PirlsCoordinateFrame,
    PirlsLinearSolvePath,
    PirlsPenalty,
    PirlsResult,
    PirlsStatus,
    PirlsWorkspace,
    SparsePirlsDecision,
    WorkingModelIterationInfo,
    WorkingModelPirlsOptions,
    WorkingModelPirlsResult,
    WorkingReparamTransform,
    WorkingState,
    // misc helpers
    array1_l2_norm,
    attach_penalty_shift,
    // compute functions
    calculate_deviance,
    // edf helpers
    calculate_edf_with_penalty,
    calculate_edfwithworkspace_with_penalty,
    calculate_loglikelihood,
    compute_constraint_kkt_diagnostics,
    computeworkingweight_derivatives_from_eta,
    inf_norm,
    runworking_model_pirls,
    should_use_sparse_native_pirls,
    solve_penalized_least_squares_implicit,
    standard_inverse_link_jet,
};
use super::{
    ArrowSchurInnerConfig, GamModelFinalState, effective_kkt_tolerance,
    project_coefficients_to_lower_bounds,
};
use crate::active_set;
use crate::estimate::EstimationError;
use crate::gpu::pirls_host_dispatch::{try_gaussian_pls_gpu, try_pirls_loop_gpu};
use crate::mixture_link::inverse_link_has_fisher_weight_jet;
use faer::sparse::{SparseColMat, Triplet};
use gam_linalg::faer_ndarray::fast_ab;
use gam_linalg::matrix::{DesignMatrix, LinearOperator, ReparamOperator, SymmetricMatrix};
use gam_math::probability::standard_normal_quantile;
use gam_problem::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LinearPredictor, LinkFunction,
    LogSmoothingParamsView, MixtureLinkState, ResponseFamily, RidgePassport, RidgePolicy,
    SasLinkState, StandardLink,
};
use gam_terms::construction::{KroneckerReparamResult, ReparamResult};
use ndarray::{ArcArray1, Array1, Array2, ArrayView1, ArrayView2, s};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// #1868 deterministic n-independence instrument.
///
/// Process-global accumulator of the number of length-`n` row-element touches
/// (array allocations / row-wise scans) performed by the Gaussian
/// zero-iteration inner synthesis on the **#1033 n-free κ-trial skip path**
/// (`row_prediction_is_stale`). On that path the outer criterion, gradient and
/// inner solve are all served from the k-space ψ-Gram sufficient statistics, so
/// the architectural invariant (#1033: "each hyperparameter trial touches only
/// k×k objects") requires this counter to stay FLAT — it must not grow with `n`.
/// A value that scales with `n` is exactly the #1868 O(n)-per-callback
/// regression (the stale-row lane re-materialising `offset`/`y`/`weights` and
/// the constant working-weight derivative arrays per trial instead of sharing
/// the once-built frozen row bundle).
///
/// This is the *deterministic* replacement for the old wall-clock
/// per-callback-ratio gate (#1868 / #2055): the same invariant, read as an exact
/// integer in milliseconds at small `n` instead of a noisy timing ratio that
/// needed a multi-hour 320k sweep to surface. Monotonic; callers snapshot the
/// value before and after the κ-trial phase and assert on the delta.
pub(crate) static NFREE_SKIP_ROW_ELEMENT_TOUCHES: AtomicU64 = AtomicU64::new(0);

/// Record `elems` length-`n` row-element touches on the n-free κ-trial skip
/// path (see [`NFREE_SKIP_ROW_ELEMENT_TOUCHES`]). Called at each length-`n`
/// materialisation the stale-row Gaussian synthesis performs; after the #1868
/// frozen-row-bundle fix the skip path performs none, so the accumulator holds
/// flat across `n`.
#[inline]
pub(crate) fn record_nfree_skip_row_touches(elems: usize) {
    NFREE_SKIP_ROW_ELEMENT_TOUCHES.fetch_add(elems as u64, Ordering::Relaxed);
}

/// Read the process-global n-free κ-trial skip-path row-touch accumulator.
/// Exposed so the spatial length-scale driver can snapshot deltas across the
/// κ-optimisation phase and thread them into the reported timing.
pub fn nfree_skip_row_element_touches() -> u64 {
    NFREE_SKIP_ROW_ELEMENT_TOUCHES.load(Ordering::Relaxed)
}

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
        | LinkFunction::LogLog
        | LinkFunction::Cauchit
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
                    LinkFunction::LogLog => -(-prevalence.ln()).ln(),
                    LinkFunction::Cauchit => (std::f64::consts::PI * (prevalence - 0.5)).tan(),
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
    // #1868: the full-assembly path is legitimately O(n) (this is the one-off
    // final fit, not a per-callback n-free skip); wrap its freshly-realised row
    // arrays in the shared `ArcArray1` representation (`.into_shared()` moves the
    // owned buffer into an `Arc`, O(1)). `finalmu`/`solvemu` share one handle.
    let final_eta_arr = working_summary.state.eta.as_ref().clone();
    let finalmu_shared = finalmu.clone().into_shared();
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
        finalweights: finalweights.clone().into_shared(),
        final_offset: offset.to_owned().into_shared(),
        final_eta: final_eta_arr.into_shared(),
        finalmu: finalmu_shared.clone(),
        solveweights: scoreweights.clone().into_shared(),
        solveworking_response: finalz.clone().into_shared(),
        solvemu: finalmu_shared,
        solve_dmu_deta: final_dmu_deta.clone().into_shared(),
        solve_d2mu_deta2: final_d2mu_deta2.clone().into_shared(),
        solve_d3mu_deta3: final_d3mu_deta3.clone().into_shared(),
        solve_c_array: final_c.clone().into_shared(),
        solve_d_array: final_d.clone().into_shared(),
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
        used_device: false,
        cache_compacted: false,
        min_penalized_deviance: working_summary.min_penalized_deviance,
    }
}

pub(super) fn detect_logit_instability(
    link: LinkFunction,
    response: &ResponseFamily,
    has_penalty: bool,
    firth_active: bool,
    summary: &WorkingModelPirlsResult,
    finalmu: &Array1<f64>,
    finalweights: &Array1<f64>,
    y: ArrayView1<'_, f64>,
) -> bool {
    // Perfect / quasi-perfect separation is a *Bernoulli/Binomial* pathology.
    // Every heuristic below is binary-response–specific: saturation toward
    // μ ∈ {0, 1}, the `yᵢ > 0.5` order-separation split, and working-weight
    // collapse only carry meaning when each `yᵢ` is a 0/1 outcome (or a
    // proportion of Bernoulli trials). The Beta family also fits through the
    // logit link, but its response is *continuous* on (0, 1): a perfectly
    // healthy monotone mean (μ increasing in a covariate ⇒ rows with y > 0.5
    // sit at higher η than rows with y ≤ 0.5) trivially satisfies the
    // `order_separated` test, so gating this detector on the logit link alone
    // misclassifies well-behaved Beta fits as separated and forces a spurious
    // inner-solve retreat at every smoothing-parameter seed (issue #499).
    // Gate strictly on the Binomial response so only binary GLMs are screened.
    if !matches!(response, ResponseFamily::Binomial) || link != LinkFunction::Logit || firth_active
    {
        return false;
    }

    // Separation-detection policy thresholds. Each is a heuristic cut-off, not
    // a math identity: they decide when a binary-logit fit has drifted into the
    // perfect/quasi-perfect separation regime and the inner solve must retreat.
    //
    // `ORDER_SEPARATION_ETA_GAP`: a strictly positive η-gap between the lowest
    //   η among y=1 rows and the highest among y=0 rows means the two classes
    //   are linearly separable on the linear predictor.
    // `EXTREME_ETA`: |η| this large drives μ to within machine-ε of {0,1}.
    // `SATURATION_FRACTION` / `SEVERE_SATURATION_FRACTION`: share of fitted μ
    //   pinned to the {0,1} boundary that flags (severe) saturation.
    // `DEGENERATE_DEVIANCE_PER_SAMPLE` / `EXTREME_DEGENERATE_DEVIANCE_PER_SAMPLE`:
    //   near-zero per-sample deviance means the model fits the data perfectly.
    // `EXTREME_BETA_NORM`: coefficient norm blow-up characteristic of the MLE
    //   escaping to infinity under separation.
    // `WEIGHT_COLLAPSE_FRACTION`: share of working weights collapsed to ~0.
    const ORDER_SEPARATION_ETA_GAP: f64 = 1e-3;
    const EXTREME_ETA: f64 = 30.0;
    const SATURATION_FRACTION: f64 = 0.98;
    const SEVERE_SATURATION_FRACTION: f64 = 0.995;
    const DEGENERATE_DEVIANCE_PER_SAMPLE: f64 = 1e-3;
    const EXTREME_DEGENERATE_DEVIANCE_PER_SAMPLE: f64 = 1e-6;
    const EXTREME_BETA_NORM: f64 = 1e4;
    const WEIGHT_COLLAPSE_FRACTION: f64 = 0.98;

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
    let order_separated =
        has_pos && has_neg && (min_eta_pos - max_eta_neg) > ORDER_SEPARATION_ETA_GAP;

    let classic_signals = max_abs_eta > EXTREME_ETA
        || sat_fraction > SATURATION_FRACTION
        || dev_per_sample < DEGENERATE_DEVIANCE_PER_SAMPLE
        || beta_norm > EXTREME_BETA_NORM;

    if !has_penalty {
        return classic_signals || order_separated;
    }

    let severe_saturation = sat_fraction > SEVERE_SATURATION_FRACTION && max_abs_eta > EXTREME_ETA;
    let weights_collapsed = weight_collapse_fraction > WEIGHT_COLLAPSE_FRACTION;
    let dev_extremely_small = dev_per_sample < EXTREME_DEGENERATE_DEVIANCE_PER_SAMPLE;

    order_separated || severe_saturation || weights_collapsed || dev_extremely_small
}

/// Stack λ-weighted penalty roots from canonical penalties into a single
/// `total_rank × p` matrix for PIRLS. Each block-local root is embedded
/// into the full column space on-the-fly.
pub(super) fn stack_lambdaweighted_penalty_root_canonical(
    penalties: &[gam_terms::construction::CanonicalPenalty],
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
    penalties: &[gam_terms::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> ReparamResult {
    // Map the engine penalty back into identity (original) coordinates. The
    // engine returns `s_transformed = Qsᵀ S Qs` (and `e_transformed = E Qs`)
    // with `S = S_λ + shrinkage·P_range` already folded in (so it matches the
    // reported `log_det`/`det1`). With the sparse-native `qs = I` we need that
    // SAME penalty expressed in original coordinates: `S_orig = Qs S_transformed
    // Qsᵀ`. Rebuilding `S_orig` from the bare lambda-weighted canonical sum
    // would DROP the shrinkage ridge and desync the inner penalized Hessian from
    // the penalty log-determinant the REML criterion uses for this fit — the
    // cross-backend λ-selection divergence (#1266 class). Round-tripping the
    // engine penalty through `Qs` keeps the inner solve, EDF, and REML logdet on
    // one penalty.
    let qs = &base.qs;
    let s_orig = if qs.nrows() == p && qs.ncols() == base.s_transformed.nrows() {
        // S_orig = Qs · S_transformed · Qsᵀ
        let qs_s = fast_ab(qs, &base.s_transformed);
        qs_s.dot(&qs.t())
    } else {
        // Degenerate fallback (engine produced no transform): use the bare
        // lambda-weighted sum. Shrinkage is zero in this branch by construction.
        let mut s_original = Array2::<f64>::zeros((p, p));
        for (k, cp) in penalties.iter().enumerate() {
            let lambda_k = lambdas.get(k).copied().unwrap_or(0.0);
            if lambda_k != 0.0 {
                cp.accumulate_weighted(&mut s_original, lambda_k);
            }
        }
        s_original
    };
    // E_orig = E_transformed · Qsᵀ  (so that E_origᵀ E_orig = S_orig and the EDF
    // augmented system matches the inner Hessian).
    let e_orig = if qs.nrows() == p && base.e_transformed.ncols() == qs.ncols() {
        base.e_transformed.dot(&qs.t())
    } else {
        stack_lambdaweighted_penalty_root_canonical(penalties, lambdas, p)
    };
    let u_original = if base.u_truncated.nrows() == p {
        fast_ab(&base.qs, &base.u_truncated)
    } else {
        Array2::<f64>::eye(p)
    };
    // In the sparse-native path, qs = I, so the penalties are already in the
    // right coordinate frame. We keep them as-is in canonical_transformed.
    let canonical_transformed: Vec<gam_terms::construction::CanonicalPenalty> = penalties.to_vec();
    ReparamResult {
        penalty_shrinkage_ridge: base.penalty_shrinkage_ridge,
        s_transformed: s_orig,
        log_det: base.log_det,
        det1: base.det1,
        qs: Array2::<f64>::eye(p),
        canonical_transformed,
        e_transformed: e_orig,
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

    const KRONECKER_STRUCTURAL_ZERO_TOL: f64 = 1e-12;
    let mut multi_idx = vec![0usize; d];
    let mut flat = 0usize;
    loop {
        let mut sigma = 0.0;
        let mut structural_sigma = 0.0;
        for k in 0..d {
            let marginal_eigenvalue = kron_result.marginal_eigenvalues[k][multi_idx[k]];
            structural_sigma += marginal_eigenvalue;
            sigma += lambdas[k] * marginal_eigenvalue;
        }
        let joint_null = structural_sigma <= KRONECKER_STRUCTURAL_ZERO_TOL;
        if kron_result.has_double_penalty && lambdas.len() > d && joint_null {
            sigma += lambdas[d];
        }
        if structural_sigma > KRONECKER_STRUCTURAL_ZERO_TOL {
            sigma += kron_result.penalty_shrinkage_ridge;
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
    penalties: &[gam_terms::construction::CanonicalPenalty],
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
    penalties: &[gam_terms::construction::CanonicalPenalty],
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
    /// Frozen-weight first-Fisher-step data-fit Gram `XᵀWX` for a GLM
    /// design-moving ψ-trial (#1111 / #1033 mechanism (c)), in *original*
    /// (conditioned `x_fit`) coordinates. When set, the iterative GLM P-IRLS
    /// serves its FIRST Fisher-scoring iteration's `XᵀWX` from this matrix
    /// instead of streaming the O(N·p²) weighted cross-product; every later
    /// iteration restreams the true moving `W`, so the converged β̂ is
    /// unchanged. Mutually distinct from `gaussian_fixed_cache` (which is the
    /// Gaussian-identity converged-objective short-circuit); this is the GLM
    /// first-step lane and never short-circuits the iteration count.
    pub glm_first_step_gram: Option<&'a Array2<f64>>,
}

// GaussianFixedCache is defined in pls_solver.
pub use super::pls_solver::GaussianFixedCache;

pub struct PenaltyConfig<'a> {
    /// Block-local canonical penalties with precomputed roots and spectral data.
    /// This is the single canonical penalty representation — no full-width
    /// `rank × p` roots are stored. When the reparameterization engine needs
    /// full-width roots, they are derived on-the-fly from these block-local roots.
    pub canonical_penalties: &'a [gam_terms::construction::CanonicalPenalty],
    pub balanced_penalty_root: Option<&'a Array2<f64>>,
    pub reparam_invariant: Option<&'a gam_terms::construction::ReparamInvariant>,
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
    pub kronecker_factored: Option<&'a gam_terms::basis::KroneckerFactoredBasis>,
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
    fit_model_for_fixed_rho_with_adaptive_kkt(
        rho,
        problem,
        penalty,
        config,
        warm_start_beta,
        None,
        false,
    )
}

/// `refine_dispersion_at_converged_eta`: when `true`, after the inner P-IRLS
/// solve converges, re-estimate the family's estimated dispersion nuisance — the
/// Gamma shape ν = 1/φ or the Beta precision φ — at the *converged* linear
/// predictor and iterate the (β, dispersion) pair to its joint fixed point at the
/// current λ (see the in-body comments at each refresh loop). This is ON only for
/// the single final, reported fit at the REML-selected λ (#678 for Gamma, #769
/// for Beta). It is deliberately OFF for every REML cost / sigma-point evaluation:
/// re-profiling the dispersion against each trial λ's converged residuals would
/// couple the scale to the smoothing parameter (a flat over-smoothed μ inflates
/// the deviance ⇒ a smaller effective precision ⇒ a smaller `deviance/(2φ)` REML
/// term), perversely rewarding over-smoothing and biasing λ selection. mgcv
/// likewise estimates the scale at the converged fit, not inside the λ search.
///
/// The Gamma and Beta cases differ in what the re-solve buys. For Gamma the shape
/// is a pure nuisance — β̂ is essentially scale-free — so the re-solve only keeps
/// the reported dispersion and SEs self-consistent. For Beta the precision φ
/// enters the *mean* score through the digamma terms
/// `μ*ᵢ = ψ(μᵢφ) − ψ((1−μᵢ)φ)`, so a φ measured at the cold null predictor
/// (μ ≈ 0.5) attenuates every slope toward zero; here the fixed point is
/// load-bearing — it is what recovers the correct mean coefficients (the betareg
/// alternating mean-fit ↔ φ-estimate scheme).
pub(crate) fn fit_model_for_fixed_rho_with_adaptive_kkt<'a, X: Into<DesignMatrix> + Clone>(
    rho: LogSmoothingParamsView<'_>,
    problem: PirlsProblem<'a, X>,
    penalty: PenaltyConfig<'_>,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
    adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
    refine_dispersion_at_converged_eta: bool,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let PirlsProblem {
        x,
        offset,
        y,
        priorweights,
        covariate_se,
        gaussian_fixed_cache,
        glm_first_step_gram,
    } = problem;
    let quadctx = crate::quadrature::QuadratureContext::new();
    // gam#1379 — finite-ceiling λ = exp(ρ). When the outer REML / spatial-κ
    // optimizer drives a redundant penalty direction's log-λ past ~709 (it does
    // so deterministically on 1-D `matern(x)` / `bs="gp"` data whose kernel
    // already controls the smoothness an operator block also penalizes, so REML
    // wants λ → ∞), `exp(ρ)` overflows to `+∞`. A literal `+∞` λ then poisons
    // every downstream consumer that forms `λ · S`: the range-penalty block
    // assembled as `Σ λ_k S_k` hits `∞ · 0 = NaN` and the eigensolve aborts, and
    // the final fit-result validation rejects the non-finite stored λ outright.
    // `exp(709.78) ≈ 1.8e308` is already the largest finite f64; capping log-λ at
    // a value whose `exp` stays finite pins the over-penalized direction exactly
    // as hard as `+∞` would for every finite-arithmetic consumer (the penalized
    // block is numerically a hard constraint at λ this large) while keeping
    // `λ · 0 = 0`. Ordinary finite λ are untouched, so non-degenerate fits and
    // their recorded λ̂ are bit-identical. `ln(1e300) ≈ 690.78` keeps this in lock
    // step with the post-exp λ ceiling (`1e300`) used by the reparam range-block
    // assembly and the stored fit result, so a fully-smoothed direction carries
    // the SAME finite λ everywhere it is consumed.
    const LOG_LAMBDA_CEILING: f64 = 690.0;
    let lambdas = rho.mapv(|r| {
        if r.is_nan() {
            r
        } else {
            r.min(LOG_LAMBDA_CEILING).exp()
        }
    });
    let lambdas_slice = lambdas.as_slice_memory_order().ok_or_else(|| {
        EstimationError::InvalidInput("non-contiguous lambda storage".to_string())
    })?;

    let likelihood = &config.likelihood;
    let link_function = config.link_function();

    use gam_terms::construction::{
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
        // The marginal eigensystems and reparameterized marginals depend only on
        // the fixed marginal designs/penalties, not on λ = exp(ρ). Memoize them
        // once per fit so each outer REML iterate reuses the eigendecomposition
        // instead of recomputing `eigh()` + `B_k·U_k` every call; only the cheap
        // λ-grid logdet/derivative sweep is redone here. Bit-identical to the
        // unmemoized engine.
        let invariant = kron.invariant_structure()?;
        let kron_result =
            gam_terms::construction::kronecker_reparameterization_engine_with_invariant(
                invariant.as_ref(),
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

    // Run the eigendecomposition engine for the dense-transformed path. The
    // sparse-native path also needs it, but only to obtain a penalty that is
    // *consistent with the REML penalty log-determinant it reports* — see the
    // sparse-native `reparam` below. The dense path keeps `qs ≠ I`; the
    // sparse-native path discards `qs` (identity coords) and reuses only the
    // shrinkage-folded `s_transformed`/`e_transformed`.
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
    // Sparse-native reparam result, in identity (original) coordinates with the
    // penalty shrinkage floor folded in. This MUST drive the inner penalized
    // solve too: when `penalty_shrinkage_floor` is active (default `Some(1e-6)`)
    // the dense engine adds `shrinkage·P_range` to every penalized range
    // direction of `S_λ` and rebuilds `s_transformed = EᵀE` from the floored
    // roots, so `base.log_det` (the REML penalty pseudo-logdet) is the
    // determinant of `S_λ + shrinkage·P_range`, NOT of the bare `S_λ`. Building
    // the inner Hessian from an UN-shrunk `S_λ` (the previous behaviour, via the
    // `cheap_s_lambda` row-sum) while reporting the shrunk `log_det` made the
    // sparse-native REML surface internally inconsistent — the penalty-logdet
    // term and the inner H / EDF / β̂ lived on different penalties — which biased
    // λ-selection relative to the dense and Kronecker backends for the SAME
    // model (the #1266 cross-backend divergence class). Reusing the engine's
    // shrinkage-folded penalty here makes all three backends solve the same
    // penalized objective.
    let sparse_native_reparam = if use_sparse_native && penalty.kronecker_factored.is_none() {
        let base = stable_reparameterization_engine_canonical(
            penalty.canonical_penalties,
            lambdas_slice,
            EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
            penalty.reparam_invariant,
            penalty.penalty_shrinkage_floor,
        )?;
        Some(build_sparse_native_reparam_result(
            base,
            penalty.canonical_penalties,
            lambdas_slice,
            penalty.p,
        ))
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
        // Sparse-native inner penalty in original (identity) coordinates. Use
        // the shrinkage-folded `s_transformed`/`e_transformed` from
        // `sparse_native_reparam` so the inner penalized Hessian
        // `H = XᵀWX + S` matches the penalty whose log-determinant the REML
        // criterion reports for this fit (`base.log_det`). Falling back to the
        // bare lambda-weighted sum here (the prior behaviour) omitted the
        // `penalty_shrinkage_floor` ridge and desynced the inner solve from the
        // REML logdet, biasing λ-selection vs the dense/Kronecker backends.
        let sparse_reparam = sparse_native_reparam
            .as_ref()
            .expect("sparse_native_reparam should be present for sparse-native path");
        PirlsPenalty::Dense {
            s_transformed: sparse_reparam.s_transformed.clone(),
            e_transformed: sparse_reparam.e_transformed.clone(),
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
            // Sparse-native path: reuse the engine result already computed for
            // `penalty_active` (with the shrinkage floor folded in and mapped to
            // identity coordinates). This is both correct — the REML
            // log-determinant now matches the penalty the inner solve used — and
            // cheaper, since the eigendecomposition is no longer run twice.
            Ok(sparse_native_reparam
                .as_ref()
                .expect("sparse_native_reparam should be present for sparse-native path")
                .clone())
        } else {
            Ok(dense_reparam_result
                .as_ref()
                .expect("dense reparam result should be present outside Kronecker path")
                .clone())
        }
    };

    // Stage 3.3-GI: GPU exact PLS dispatch — see pirls_host_dispatch::try_gaussian_pls_gpu.
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

    if matches!(link_function, LinkFunction::Identity) && linear_constraints.is_none() {
        // Gaussian-Identity zero-iteration exact solve. The unconstrained
        // penalized least-squares system is linear, so for an identity link a
        // single solve is the exact minimizer and no PIRLS iteration is needed.
        //
        // This shortcut is only valid in the *unconstrained* convex program.
        // When shape/box/linear inequality constraints are present (e.g. a
        // `shape=monotone_increasing` smooth, whose cumulative-sum box-reparam
        // bounds `γ_j ≥ 0` are folded into `linear_constraints` above), the
        // minimizer is the solution of an inequality-constrained QP, not the
        // plain normal-equations solve. Taking this branch then returns the
        // unconstrained β, which generically violates the constraints and is
        // rejected by the REML startup KKT gate (`enforce_constraint_kkt`),
        // aborting the whole fit. Gating on `linear_constraints.is_none()`
        // routes every constrained Identity fit to the iterative loop below,
        // which builds a feasible initial point and solves the exact QP via
        // the active-set solver — mirroring the gate already enforced on the
        // GPU Gaussian-PLS path in `try_gaussian_pls_gpu`.
        //
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

        // eta = offset + X Qs beta (composed, no materialization) unless a
        // design-moving ψ tensor cache explicitly says the surface rows are a
        // stale reference. In that lane the Gaussian objective and gradient are
        // fully determined by (G, r, y'Wy), so applying `x_original` would both
        // reintroduce per-trial row work and evaluate the wrong ψ.
        let qbeta = transform_active
            .as_ref()
            .map(|transform| transform.apply(beta_transformed.as_ref()))
            .unwrap_or_else(|| beta_transformed.as_ref().clone());
        let stale_row_cache = cache_for_solve.filter(|cache| cache.row_prediction_is_stale);

        // #1868: all length-`n` row arrays of the zero-iteration synthesis,
        // collected in one place so the skip path can SHARE them O(1) from the
        // once-built frozen bundle (zero row touches) while the exact path builds
        // them owned and moves them into the shared `ArcArray1` representation
        // via `.into_shared()` (O(1), no element copy).
        struct ZeroIterRows {
            final_offset: ArcArray1<f64>,
            final_eta: ArcArray1<f64>,
            finalmu: ArcArray1<f64>,
            finalz: ArcArray1<f64>,
            finalweights: ArcArray1<f64>,
            solve_dmu_deta: ArcArray1<f64>,
            solve_d2mu_deta2: ArcArray1<f64>,
            solve_d3mu_deta3: ArcArray1<f64>,
            solve_c_array: ArcArray1<f64>,
            solve_d_array: ArcArray1<f64>,
            /// Working-state η. Empty on the skip path (the stale rows are never
            /// read on the n-free κ criterion path, so it is not materialised —
            /// keeping the callback O(1)); the freshly-realised η on the exact
            /// path.
            working_eta: LinearPredictor,
            gradient_data: Array1<f64>,
            deviance: f64,
            log_likelihood: f64,
            max_abs_eta: f64,
        }

        let rows = if let Some(cache) = stale_row_cache {
            // #1868 FAST PATH: the criterion, gradient and inner solve are served
            // entirely from k-space Gram sufficient statistics; the length-`n`
            // row arrays are trial-invariant placeholders (η≡μ≡offset, z≡y,
            // w≡priorweights, constant Gaussian working-weight derivatives). When
            // the producer attached the once-built frozen bundle we clone its
            // `ArcArray1` handles (O(1), zero element touches) instead of
            // re-materialising ~16·n elements per κ callback — the #1868 fix.
            let mut grad_orig = cache.xtwx_orig.dot(&qbeta);
            grad_orig -= &cache.xtwy_orig;
            let gradient_data = transform_active
                .as_ref()
                .map(|transform| transform.apply_transpose(&grad_orig))
                .unwrap_or(grad_orig);
            let weighted_rss = (cache.centered_weighted_y_sq
                - 2.0 * qbeta.dot(&cache.xtwy_orig)
                + qbeta.dot(&cache.xtwx_orig.dot(&qbeta)))
            .max(0.0);
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            let deviance = if phi.is_finite() && phi > 0.0 {
                weighted_rss / phi
            } else {
                f64::NAN
            };

            if let Some(bundle) = cache.frozen_rows.as_ref() {
                // Zero length-`n` touches: every row array is an O(1) Arc clone
                // of the shared frozen bundle (η≡μ≡offset via `bundle.eta`).
                ZeroIterRows {
                    final_offset: bundle.eta.clone(),
                    final_eta: bundle.eta.clone(),
                    finalmu: bundle.eta.clone(),
                    finalz: bundle.z.clone(),
                    finalweights: bundle.weights.clone(),
                    solve_dmu_deta: bundle.solve_dmu_deta.clone(),
                    solve_d2mu_deta2: bundle.solve_d2mu_deta2.clone(),
                    solve_d3mu_deta3: bundle.solve_d3mu_deta3.clone(),
                    solve_c_array: bundle.solve_c_array.clone(),
                    solve_d_array: bundle.solve_d_array.clone(),
                    working_eta: LinearPredictor::new(Array1::zeros(0)),
                    gradient_data,
                    deviance,
                    log_likelihood: bundle.log_likelihood,
                    max_abs_eta: bundle.max_abs_eta,
                }
            } else {
                // No bundle attached (producer could not build it): fall back to
                // the correct-but-O(n) re-materialisation so the fit is never
                // wrong. Counted so the deterministic gate still sees this work.
                let n_rows = offset.len();
                record_nfree_skip_row_touches(11 * n_rows);
                let final_eta = offset.to_owned();
                let finalmu = final_eta.clone();
                let priorweights_owned = priorweights.to_owned();
                let (c, d, dmu_deta, d2mu_deta2, d3mu_deta3) =
                    computeworkingweight_derivatives_from_eta(
                        &config.likelihood,
                        &config.link_kind,
                        &final_eta,
                        priorweights_owned.view(),
                    )?;
                let log_likelihood = calculate_loglikelihood(y, &finalmu, likelihood, priorweights);
                let max_abs_eta = inf_norm(finalmu.iter().copied());
                ZeroIterRows {
                    final_offset: offset.to_owned().into_shared(),
                    final_eta: final_eta.into_shared(),
                    finalmu: finalmu.into_shared(),
                    finalz: y.to_owned().into_shared(),
                    finalweights: priorweights_owned.into_shared(),
                    solve_dmu_deta: dmu_deta.into_shared(),
                    solve_d2mu_deta2: d2mu_deta2.into_shared(),
                    solve_d3mu_deta3: d3mu_deta3.into_shared(),
                    solve_c_array: c.into_shared(),
                    solve_d_array: d.into_shared(),
                    working_eta: LinearPredictor::new(Array1::zeros(0)),
                    gradient_data,
                    deviance,
                    log_likelihood,
                    max_abs_eta,
                }
            }
        } else {
            // EXACT path: rows are freshly realised from the (non-stale) design.
            // Legitimately O(n) — this is the one-off final assembly / a
            // non-tensor trial, not a per-callback n-free skip.
            let priorweights_owned = priorweights.to_owned();
            let mut eta = offset.to_owned();
            eta += &x_original.apply(&qbeta);
            let final_eta = eta.clone();
            let finalmu = eta;

            let mut weighted_residual = finalmu.clone();
            weighted_residual -= &y;
            weighted_residual *= &priorweights_owned;
            // gradient = Qs^T X^T (w * residual) (composed)
            let xt_wr = x_original.apply_transpose(&weighted_residual);
            let gradient_data = transform_active
                .as_ref()
                .map(|transform| transform.apply_transpose(&xt_wr))
                .unwrap_or(xt_wr);
            let deviance = calculate_deviance(y, &finalmu, likelihood, priorweights);
            let log_likelihood = calculate_loglikelihood(y, &finalmu, likelihood, priorweights);
            let max_abs_eta = inf_norm(finalmu.iter().copied());
            let (c, d, dmu_deta, d2mu_deta2, d3mu_deta3) =
                computeworkingweight_derivatives_from_eta(
                    &config.likelihood,
                    &config.link_kind,
                    &final_eta,
                    priorweights_owned.view(),
                )?;
            ZeroIterRows {
                final_offset: offset.to_owned().into_shared(),
                working_eta: LinearPredictor::new(finalmu.clone()),
                final_eta: final_eta.into_shared(),
                finalmu: finalmu.into_shared(),
                finalz: y.to_owned().into_shared(),
                finalweights: priorweights_owned.into_shared(),
                solve_dmu_deta: dmu_deta.into_shared(),
                solve_d2mu_deta2: d2mu_deta2.into_shared(),
                solve_d3mu_deta3: d3mu_deta3.into_shared(),
                solve_c_array: c.into_shared(),
                solve_d_array: d.into_shared(),
                gradient_data,
                deviance,
                log_likelihood,
                max_abs_eta,
            }
        };
        let ZeroIterRows {
            final_offset,
            final_eta,
            finalmu,
            finalz,
            finalweights,
            solve_dmu_deta,
            solve_d2mu_deta2,
            solve_d3mu_deta3,
            solve_c_array,
            solve_d_array,
            working_eta,
            gradient_data,
            deviance,
            log_likelihood,
            max_abs_eta,
        } = rows;
        let score_norm = array1_l2_norm(&gradient_data);
        let s_beta = penalty_active.shifted_gradient(beta_transformed.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        let mut gradient = gradient_data;
        gradient += &s_beta;
        let mut penalty_term = penalty_active.shifted_quadratic(beta_transformed.as_ref());
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
        let working_state = WorkingState {
            eta: working_eta,
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

        // #1868: `solve_*`/`final_*` row arrays now come from the row synthesis
        // above (shared O(1) from the frozen bundle on the skip path); the exact
        // per-callback `computeworkingweight_derivatives_from_eta` re-computation
        // that used to run here is folded into that synthesis.
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
            finalweights: finalweights.clone(),
            final_offset,
            final_eta,
            finalmu: finalmu.clone(),
            solveweights: finalweights,
            solveworking_response: finalz,
            solvemu: finalmu,
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
            used_device: false,
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
        // Inner Firth/Jeffreys activation must agree with the caller-requested
        // mode. The REML *outer* analytic derivative assembly only carries the
        // Jeffreys score/curvature term when `firth_bias_reduction` is set
        // (`reml_robust_jeffreys_link` returns `None` otherwise), so arming the
        // inner penalty unconditionally would converge the inner mode to the
        // Firth-penalized stationary point while the outer H/u/IFT stayed
        // non-Firth — the two would then disagree by exactly the Jeffreys
        // contribution (broken τ-τ Hessian-vs-FD and stationarity-cancellation
        // identities, #825). Gate on `firth_bias_reduction` so inner and outer
        // are the same objective.
        config.firth_bias_reduction
            && matches!(config.likelihood.spec.response, ResponseFamily::Binomial)
            && inverse_link_has_fisher_weight_jet(&config.link_kind),
        transform_active.clone(),
        quadctx,
        // #1111 / #1033 mechanism (c): frozen-W first-Fisher-step XᵀWX in the
        // original (conditioned x_fit) frame, served n-free on the first inner
        // iteration. Suppressed under Firth bias reduction, which shifts the
        // working response per iteration (the installer also gates Firth off).
        if config.firth_bias_reduction {
            None
        } else {
            glm_first_step_gram.cloned()
        },
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
        // Worst per-row *scaled* (geometric) slack of the current seed against the
        // constraint cone. Negative ⇒ the seed violates a row; ~0 ⇒ the seed sits
        // ON the boundary (for a homogeneous convex/concave second-difference
        // cone, `β = 0` — the unconstrained Gaussian seed — sits on EVERY row's
        // boundary, i.e. the cone vertex). Either way the seed must be pushed
        // strictly into the interior before P-IRLS starts.
        let mut min_scaled_slack = f64::INFINITY;
        for i in 0..constraints.a.nrows() {
            let norm = constraints.a.row(i).dot(&constraints.a.row(i)).sqrt();
            let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            let slack = (constraints.a.row(i).dot(&initial_beta) - constraints.b[i]) * inv;
            min_scaled_slack = min_scaled_slack.min(slack);
        }
        // Push the seed to the nearest STRICTLY-INTERIOR feasible point whenever
        // any row is tight or violated. A seed on the cone boundary (most acutely
        // the vertex `β = 0`) hands the inner active-set QP an all-rows-active
        // working set, where it stalls on a degenerate, non-stationary face — so
        // the fit silently diverges (or aborts in release) between a cold and a
        // warm warm-start cache (#873). A strictly-interior seed makes the QP's
        // initial active set empty; it then adds only the genuinely binding rows
        // and converges to the certified constrained optimum regardless of cache
        // state. The projection keeps the data-driven curvature of `initial_beta`
        // and falls back to the min-norm feasible point only if it cannot certify
        // a strictly-interior solution.
        //
        // The min-norm fallback (`feasible_point_for_linear_constraints`) is only
        // used for a NON-homogeneous cone (`b ≠ 0`), where it returns a genuine
        // interior-of-the-offset-polyhedron point. For a HOMOGENEOUS shape cone
        // (`b ≈ 0` — the convex/concave second-difference rows) that function
        // returns the minimum-norm feasible point `β = 0`, which is the cone
        // *vertex*: the exact all-rows-tight degenerate seed #873 is about. Taking
        // it would silently reintroduce the #873 pathology whenever the strict
        // projection rarely fails to certify. So for a homogeneous cone we skip the
        // vertex fallback entirely and prefer the data-driven `initial_beta`: it
        // violates at most *some* rows (a lower-dimensional, non-degenerate face the
        // inner active-set QP can recover from), strictly better than the vertex
        // where *every* row is simultaneously tight.
        let cone_is_homogeneous = constraints.b.iter().all(|v| v.abs() <= 1e-14);
        if min_scaled_slack < active_set::interior_seed_margin() {
            let projected =
                active_set::project_point_strictly_into_feasible_cone(&initial_beta, constraints)
                    .or_else(|| {
                        if cone_is_homogeneous {
                            None
                        } else {
                            active_set::feasible_point_for_linear_constraints(
                                constraints,
                                initial_beta.len(),
                            )
                        }
                    });
            projected.unwrap_or(initial_beta)
        } else {
            initial_beta
        }
    } else {
        initial_beta
    };
    // Inner P-IRLS Firth activation. The inner penalized objective must match
    // the objective the REML outer derivatives are assembled against: the outer
    // path carries the Jeffreys/Firth score+curvature only when the caller set
    // `firth_bias_reduction` (`reml_robust_jeffreys_link` is `None` otherwise),
    // so the inner Firth term is armed iff the caller requested it AND the link
    // exposes a Fisher-weight jet (#825). Forcing it on unconditionally desynced
    // the Firth-penalized inner mode from the non-Firth outer assembly.
    let firth_active = config.firth_bias_reduction
        && matches!(config.likelihood.spec.response, ResponseFamily::Binomial)
        && inverse_link_has_fisher_weight_jet(&config.link_kind);
    let base_max_step_halving = if firth_active { 60 } else { 30 };
    let options = WorkingModelPirlsOptions {
        // The Firth-penalized P-IRLS converges at the same iteration count as
        // the unpenalized fit — the Jeffreys term is a smooth, bounded addition
        // to a Newton system that is already well conditioned (the additional
        // per-iteration LM step-halving budget above absorbs the early-iteration
        // curvature change). Bumping the outer-iteration cap to mask a
        // mis-conditioned step would only hide non-convergence, so the cap stays
        // the caller's `max_iterations` and trips as a hard error if exceeded.
        max_iterations: config.max_iterations,
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

    // Stage 3.3 GPU PIRLS-loop dispatch — see pirls_host_dispatch::try_pirls_loop_gpu.
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

    // ── Gamma dispersion: re-estimate the shape at the *converged* η (#678) ──
    //
    // The inner LM solve estimates the Gamma shape ν = 1/φ **once** from the
    // warm-start η and freezes it for the rest of the solve (see the
    // `gamma_shape_locked` doc on `GamWorkingModel`): holding ν fixed keeps the
    // product φ·λ — and hence the penalized argmin β̂ — a stationary LM target,
    // so the gain ratio compares one objective. That lock is correct *within* a
    // solve, but it pins ν to whatever η the solve started from. When the fit
    // cold-starts (the final dedicated fit at the converged ρ passes
    // `warm_start_beta = None`, and seed screening starts from a default guess),
    // that warm-start η has not yet captured the mean structure; the leftover
    // spread of μ inflates the Gamma deviance term `mean[y/μ − ln(y/μ) − 1]` and
    // biases ν **down** (φ up) by >2× whenever μ varies appreciably. The mean
    // surface still converges (β̂ is essentially scale-free here), but the frozen
    // ν that survives into `UnifiedFitResult::dispersion_phi()` — and from there
    // into every coefficient SE `Vb = H⁻¹·φ̂`, prediction interval, and
    // observation-noise interval — is the early, mean-spread-contaminated value.
    //
    // Fix: after the solve converges, re-estimate ν at the converged η. If it
    // moved, re-solve β (warm-started, ν held fixed at the refreshed value) and
    // repeat, driving the pair (β, ν) to their joint fixed point at the current
    // λ. At convergence the reported dispersion is the Gamma ML estimate at the
    // converged mean (mgcv's post-hoc Pearson/deviance scale), and the final
    // working state — `finalweights`, the penalized Hessian, the deviance, μ —
    // is rebuilt with that same ν, so `Vb = H⁻¹·φ̂` stays internally consistent.
    // Warm-started solves (every REML cost eval) already sit near the converged
    // η, so the first refresh check confirms ν and exits without a re-solve; the
    // added cost there is a single O(n) shape evaluation.
    if refine_dispersion_at_converged_eta
        && working_model.likelihood.scale.gamma_shape_is_estimated()
    {
        // A few passes suffice: the converged-η shape map is a strong
        // contraction (β̂ barely moves once the mean is captured), so cold
        // starts settle in 1–2 re-solves and warm starts in zero.
        const MAX_SHAPE_REFRESH: usize = 5;
        // Relative shape tolerance below which a re-solve cannot move any
        // reported quantity meaningfully (far under statistical resolution).
        const SHAPE_REFRESH_REL_TOL: f64 = 1e-4;
        for refresh_iter in 0..MAX_SHAPE_REFRESH {
            let refreshed_shape = super::estimate_gamma_shape_from_eta(
                y,
                working_summary.state.eta.as_ref(),
                priorweights,
            );
            let prior_shape = working_model.likelihood.gamma_shape().unwrap_or(1.0);
            let rel_change =
                (refreshed_shape - prior_shape).abs() / prior_shape.max(f64::MIN_POSITIVE);
            // Install the refreshed shape and hold it fixed for any re-solve so
            // the LM objective stays stationary (the lock is *re-armed*, not
            // released — the seed-from-warm-start branch in `update_with_curvature`
            // must not overwrite this deliberately chosen value). Because this
            // assignment evaluated the shape at the *current* converged η and no
            // re-solve follows it on the exit paths below, the reported shape
            // always equals `estimate_gamma_shape_from_eta(final_eta)` — the
            // self-consistency invariant the in-module Gamma unit test checks.
            working_model.likelihood = working_model
                .likelihood
                .clone()
                .with_gamma_shape(refreshed_shape);
            working_model.gamma_shape_locked = true;
            if rel_change <= SHAPE_REFRESH_REL_TOL {
                // Converged: the working-state buffers (weights, Hessian,
                // deviance) already reflect a shape within tolerance of
                // `refreshed_shape`, because the only way to reach here without
                // a re-solve is that the prior solve's shape already matched the
                // converged-η estimate. Nothing left to rebuild.
                break;
            }
            if refresh_iter + 1 == MAX_SHAPE_REFRESH {
                // Final allowed pass and the shape is still drifting (a
                // pathological non-contraction). Do NOT re-solve: re-solving
                // would advance `final_eta` past the η the just-installed shape
                // was evaluated at, breaking the stored-shape == estimate(final_eta)
                // invariant. Stopping here keeps the reported shape exactly the
                // ML estimate at the reported η; the residual weight/φ drift is
                // bounded by the last `rel_change` and never worse than the
                // pre-fix frozen-warm-start value.
                break;
            }
            // The shape moved: re-solve β at the corrected shape, warm-started
            // at the converged β, so the final working state is rebuilt with the
            // refreshed ν.
            working_summary = runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                &mut iteration_logger,
            )?;
        }
    }

    // ── Tweedie dispersion φ: re-estimate at the *converged* η (#771) ─────────
    //
    // Identical in spirit to the Gamma-shape refresh above: the inner LM solve
    // estimates φ **once** from the warm-start η and freezes it (the
    // `tweedie_phi_locked` lock), keeping the product φ·λ — and hence β̂ — a
    // stationary LM target. φ enters only the working weight `prior·μ^{2−p}/φ`
    // and not the working response, so (like the Gamma shape, and unlike the
    // Beta precision which couples through the digamma mean score) the mean
    // surface is essentially scale-free and β̂ barely moves when φ is corrected.
    // But the frozen warm-start φ is the value that survives into
    // `FitInference::dispersion` and the covariance `Vb = H⁻¹` (whose √φ scaling
    // lives in the weight); at a cold-started η ≈ 0 the Pearson residuals carry
    // the *marginal* spread of y, biasing the estimate. Re-estimating at the
    // converged η — re-solving β only if φ moved materially — drives (β, φ) to
    // their joint fixed point, so the reported φ is the converged-mean Pearson
    // estimate and the final weights/Hessian/SE are internally consistent with
    // it. Held OFF inside the REML λ search (the flag), φ is refreshed only at
    // the reported fit, so it cannot couple to the smoothing parameter.
    if refine_dispersion_at_converged_eta
        && working_model.likelihood.scale.tweedie_phi_is_estimated()
    {
        if let ResponseFamily::Tweedie { p } = working_model.likelihood.spec.response {
            // The converged-η Pearson map is a strong contraction (β̂ scale-free
            // here), so cold starts settle in 1–2 re-solves and warm starts in
            // zero.
            const MAX_PHI_REFRESH: usize = 5;
            // Relative φ tolerance below which a re-solve cannot move any reported
            // quantity meaningfully (far under statistical resolution).
            const PHI_REFRESH_REL_TOL: f64 = 1e-4;
            for refresh_iter in 0..MAX_PHI_REFRESH {
                let refreshed_phi = super::estimate_tweedie_phi_from_eta(
                    y,
                    working_summary.state.eta.as_ref(),
                    priorweights,
                    p,
                );
                let prior_phi = working_model.likelihood.fixed_phi().unwrap_or(1.0);
                let rel_change =
                    (refreshed_phi - prior_phi).abs() / prior_phi.max(f64::MIN_POSITIVE);
                // Install the refreshed φ (the scale metadata the working weight
                // reads via `fixed_phi()`) and re-arm the lock so a following
                // re-solve does not overwrite this converged-η value. Because the
                // exit paths below evaluate φ at the *current* η with no following
                // re-solve, the reported φ always equals
                // `estimate_tweedie_phi_from_eta(final_eta)`.
                working_model.likelihood = working_model
                    .likelihood
                    .clone()
                    .with_tweedie_phi(refreshed_phi);
                working_model.tweedie_phi_locked = true;
                if rel_change <= PHI_REFRESH_REL_TOL {
                    // Converged: the working state already reflects a φ within
                    // tolerance of `refreshed_phi`. Nothing left to rebuild.
                    break;
                }
                if refresh_iter + 1 == MAX_PHI_REFRESH {
                    // Final allowed pass and φ is still drifting. Do NOT re-solve:
                    // re-solving would advance η past the point φ was evaluated at,
                    // breaking the stored-φ == estimate(final_eta) invariant.
                    break;
                }
                // φ moved materially: re-solve β at the corrected φ, warm-started
                // at the converged β, so the final working state is rebuilt with
                // the refreshed φ.
                working_summary = runworking_model_pirls(
                    &mut working_model,
                    working_summary.beta.clone(),
                    &options,
                    &mut iteration_logger,
                )?;
            }
        }
    }

    // ── Beta precision φ: re-estimate at the *converged* η and drive (β, φ) to
    //    their joint fixed point (#769) ──────────────────────────────────────
    //
    // Like the Gamma shape above, the inner LM solve estimates φ **once** from
    // the warm-start η and freezes it for the rest of the solve (the
    // `beta_phi_locked` doc on `GamWorkingModel`): holding φ fixed keeps the
    // penalized argmin β̂ a stationary LM target so the gain ratio compares one
    // objective. But that lock pins φ to whatever η the solve started from, and
    // for the final dedicated fit at the converged ρ the warm-start is the cold
    // default guess (η ≈ 0, μ ≈ 0.5 everywhere). At the null predictor the
    // Pearson residuals `(y−μ)²/(μ(1−μ))` capture the full *marginal* spread of
    // y rather than its *conditional* spread, so the moment estimator
    // `1+φ = Σw / Σ w·s` returns a precision far too small (≈3 when the truth is
    // ≈20 here).
    //
    // Crucially — and unlike the Gamma shape — φ does **not** factor out of the
    // Beta mean score. With the logit link the score for β is
    //     ∂ℓ/∂β = φ · Σᵢ xᵢ (y*ᵢ − μ*ᵢ),   y*ᵢ = logit(yᵢ),
    //     μ*ᵢ = ψ(μᵢφ) − ψ((1−μᵢ)φ),
    // so the root β̂ depends on φ through the digamma terms. A φ that is too
    // small shrinks every fitted coefficient toward zero. So this refresh is not
    // cosmetic (as it is for Gamma): the re-solve is what *recovers the mean*.
    //
    // Fix: after the cold solve converges, re-estimate φ at the converged η,
    // re-solve β at the corrected φ (warm-started), and repeat. This is the
    // betareg alternating mean-fit ↔ φ-estimate scheme; the moment estimator is
    // a strong contraction once the mean has any structure, so the pair settles
    // in a handful of passes. Held OFF inside the REML λ search (see the flag
    // doc), φ is refreshed only here at the reported fit, so it cannot couple to
    // the smoothing parameter and reward over-smoothing. As with Gamma, every
    // exit path installs φ evaluated at the *current* η with no following
    // re-solve, so the reported φ (which flows into `EstimatedBetaPhi`, the
    // embedded `Beta { phi }`, `dispersion`, and every SE) always equals
    // `estimate_beta_phi_from_eta(final_eta)`.
    if refine_dispersion_at_converged_eta && working_model.likelihood.scale.beta_phi_is_estimated()
    {
        // The mean moves between passes (φ feeds back through the digamma
        // score), so allow a few more passes than the scale-free Gamma case;
        // the contraction is fast and warm-started re-solves are cheap.
        const MAX_PHI_REFRESH: usize = 30;
        // Relative φ tolerance below which a re-solve cannot move β̂ — and hence
        // any reported quantity — by a statistically meaningful amount.
        const PHI_REFRESH_REL_TOL: f64 = 1e-4;
        for refresh_iter in 0..MAX_PHI_REFRESH {
            let refreshed_phi = super::estimate_beta_phi_from_eta(
                y,
                working_summary.state.eta.as_ref(),
                priorweights,
            );
            let prior_phi = working_model.likelihood.fixed_phi().unwrap_or(1.0);
            let rel_change = (refreshed_phi - prior_phi).abs() / prior_phi.max(f64::MIN_POSITIVE);
            // Install the refreshed φ (updates BOTH the `Beta { phi }` family
            // variant every weight/deviance expression reads and the
            // `EstimatedBetaPhi` scale metadata) and re-arm the lock so a
            // following re-solve's `update_with_curvature` does not overwrite
            // this deliberately chosen value with a fresh cold estimate.
            working_model.likelihood = working_model
                .likelihood
                .clone()
                .with_beta_phi(refreshed_phi);
            working_model.beta_phi_locked = true;
            if rel_change <= PHI_REFRESH_REL_TOL {
                // Converged: the just-installed φ matches (to tolerance) the φ
                // the current working state was solved at, so β̂, the weights,
                // the Hessian and the deviance are already self-consistent with
                // the reported φ. Nothing left to rebuild.
                break;
            }
            if refresh_iter + 1 == MAX_PHI_REFRESH {
                // Final allowed pass and φ is still drifting. Do NOT re-solve:
                // re-solving would advance η past the point the just-installed φ
                // was evaluated at, breaking the stored-φ == estimate(final_eta)
                // invariant. Stop here so the reported φ is exactly the moment
                // estimate at the reported η.
                break;
            }
            // φ moved materially: re-solve β at the corrected φ, warm-started at
            // the converged β, so the mean is refit under the better precision
            // and the final working state is rebuilt consistently.
            working_summary = runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                &mut iteration_logger,
            )?;
        }
    }

    // ── Negative-Binomial overdispersion θ: re-estimate at the *converged* η and
    //    drive (β, θ) to their joint fixed point (#802) ───────────────────────
    //
    // Identical in spirit to the Beta-precision refresh above. The inner LM solve
    // estimates θ **once** from the warm-start η and freezes it (the
    // `negbin_theta_locked` lock), keeping the penalized argmin β̂ a stationary LM
    // target. But that lock pins θ to whatever η the solve started from, and for
    // the final dedicated fit at the converged ρ the warm-start is the cold
    // default guess (η ≈ 0). At the null predictor the Pearson residuals carry
    // the *marginal* spread of y rather than its *conditional* spread, biasing
    // the moment seed — and the frozen θ is what survives into the working weight
    // `W = μθ/(θ+μ)`, the covariance `Vb = H⁻¹` (whose overdispersion scaling
    // lives in that weight, not a post-hoc multiply), and every reported SE /
    // interval / `generate` draw.
    //
    // Like the Beta precision — and unlike the scale-free Gamma shape / Tweedie φ
    // — θ enters the NB2 working *response*, not only the weight, so re-solving β
    // under the corrected θ is not cosmetic: it recovers the mean under the right
    // variance function. Re-estimating at the converged η, re-solving β
    // (warm-started), and repeating drives (β, θ) to their joint maximum-
    // likelihood fixed point. Held OFF inside the REML λ search (the flag), θ is
    // refreshed only here at the reported fit, so it cannot couple to the
    // smoothing parameter. Every exit path installs θ evaluated at the *current*
    // η with no following re-solve, so the reported θ (which flows into the
    // embedded `NegativeBinomial { theta }`, the `EstimatedNegBinTheta` scale
    // metadata, the predictive-interval variance, and every SE) always equals
    // `estimate_negbin_theta_from_eta(final_eta)`.
    if refine_dispersion_at_converged_eta
        && working_model.likelihood.scale.negbin_theta_is_estimated()
    {
        // θ feeds back through the working response, so allow a few more passes
        // than the scale-free Gamma case; the alternation is a strong contraction
        // and warm-started re-solves are cheap.
        const MAX_THETA_REFRESH: usize = 30;
        // Relative θ tolerance below which a re-solve cannot move β̂ — and hence
        // any reported quantity — by a statistically meaningful amount.
        const THETA_REFRESH_REL_TOL: f64 = 1e-4;
        for refresh_iter in 0..MAX_THETA_REFRESH {
            let refreshed_theta = super::estimate_negbin_theta_from_eta(
                y,
                working_summary.state.eta.as_ref(),
                priorweights,
            );
            let prior_theta = working_model.likelihood.negbin_theta().unwrap_or(1.0);
            let rel_change =
                (refreshed_theta - prior_theta).abs() / prior_theta.max(f64::MIN_POSITIVE);
            // Install the refreshed θ (updates BOTH the `NegativeBinomial { theta }`
            // family variant every weight/deviance expression reads and the
            // `EstimatedNegBinTheta` scale metadata) and re-arm the lock so a
            // following re-solve's `update_with_curvature` does not overwrite this
            // deliberately chosen value with a fresh cold estimate.
            working_model.likelihood = working_model
                .likelihood
                .clone()
                .with_negbin_theta(refreshed_theta);
            working_model.negbin_theta_locked = true;
            if rel_change <= THETA_REFRESH_REL_TOL {
                // Converged: the just-installed θ matches (to tolerance) the θ the
                // current working state was solved at, so β̂, the weights, the
                // Hessian and the deviance are already self-consistent with the
                // reported θ. Nothing left to rebuild.
                break;
            }
            if refresh_iter + 1 == MAX_THETA_REFRESH {
                // Final allowed pass and θ is still drifting. Do NOT re-solve:
                // re-solving would advance η past the point the just-installed θ
                // was evaluated at, breaking the stored-θ == estimate(final_eta)
                // invariant. Stop here so the reported θ is exactly the ML
                // estimate at the reported η.
                break;
            }
            // θ moved materially: re-solve β at the corrected θ, warm-started at
            // the converged β, so the mean is refit under the better variance
            // function and the final working state is rebuilt consistently.
            working_summary = runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                &mut iteration_logger,
            )?;
        }
    }

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
        // Scale-equivariant deviance plateau band (issue #1127). The
        // `last_deviance_change` compared below and the deviance both scale as
        // `O(a²)` under a response rescaling `y → a·y` (the penalized normal
        // equations are linear in `y`, so `β → a·β` and the RSS-deviance
        // scales by `a²`). Keying the plateau band to the deviance's own
        // magnitude `+ |penalty|` makes the ratio `Δdev / dev_scale`
        // scale-invariant. The previous `.max(1.0)` absolute floor broke this:
        // for a micro-unit response (`a = 1e-6`) the deviance is `O(1e-12)`, so
        // the floor pinned the band at `1.0` — ~1e9× too loose — and this
        // max-iteration rescue declared `progress_stopped` at an over-smoothed
        // iterate, propagating an inflated `λ̂` to the outer REML loop. For a
        // well-scaled (`a ≳ 1`) or up-scaled (`a = 1e6`) objective the floor was
        // already a no-op, so those directions are byte-identical. A perfect
        // interpolating fit gives a `0` band, so the relative `Δdev` test cannot
        // fire spuriously and the scale-invariant `near_stationary_kkt`
        // certificate then governs acceptance.
        let dev_scale = summary.state.deviance.abs() + summary.state.penalty_term.abs();
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
        &final_likelihood.spec.response,
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
    /// ([`crate::arrow_schur::ArrowSchurSystem`]). When `None`
    /// (the default), the existing β-only path is used unchanged.
    ///
    /// See [`ArrowSchurInnerConfig`] for the closure contract.
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
pub(crate) fn make_reparam_operator(
    x_original: &DesignMatrix,
    qs_arc: &Arc<Array2<f64>>,
    use_sparse_native: bool,
) -> DesignMatrix {
    if use_sparse_native {
        x_original.clone()
    } else {
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(
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
    Some(
        LinearInequalityConstraints::new(a, b)
            .expect("transformed lower-bound constraint shape invariant"),
    )
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
    Some(
        LinearInequalityConstraints::new(a, b)
            .expect("transformed lower-bound constraint shape invariant"),
    )
}

pub(super) fn build_transformed_linear_constraints(
    qs: &Array2<f64>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    let lc = linear_constraints?;
    if lc.a.ncols() != qs.nrows() {
        return None;
    }
    Some(
        LinearInequalityConstraints::new(lc.a.dot(qs), lc.b.clone())
            .expect("transformed linear constraint shape invariant"),
    )
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
    // Below this column count a dense factorization beats the sparse path even
    // at high sparsity, so skip the sparsity scan entirely for narrow designs.
    const DENSE_PREFERRED_MAX_COLS: usize = 32;
    // Sparse storage + sparse Cholesky only pays off below this density (nnz as
    // a fraction of all entries); denser matrices stay dense.
    const SPARSE_DENSITY_LIMIT: f64 = 0.20;

    let nrows = x.nrows();
    let ncols = x.ncols();
    if nrows == 0 || ncols == 0 {
        return None;
    }
    // Narrow matrices are faster in dense form; avoid any sparsity scan overhead.
    if ncols <= DENSE_PREFERRED_MAX_COLS {
        return None;
    }

    const ZERO_EPS: f64 = 1e-12;
    let total = nrows.saturating_mul(ncols);
    if total == 0 {
        return None;
    }
    // If a matrix exceeds this nnz count it is too dense for sparse path; bail early.
    let sparse_nnz_limit = ((total as f64) * SPARSE_DENSITY_LIMIT).floor() as usize;
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

#[cfg(test)]
mod tests {
    use super::{PirlsPenalty, build_diagonal_penalty_from_kronecker};
    use gam_terms::construction::KroneckerReparamResult;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn kronecker_diagonal_double_penalty_hits_only_joint_null_space() {
        let kron_result = KroneckerReparamResult {
            reparameterized_marginals: std::sync::Arc::new(Vec::new()),
            marginal_eigenvalues: std::sync::Arc::new(vec![array![0.0, 2.0], array![0.0, 3.0]]),
            marginal_qs: std::sync::Arc::new(Vec::new()),
            log_det: 0.0,
            det1: Array1::zeros(3),
            det2: Array2::zeros((3, 3)),
            penalty_shrinkage_ridge: 0.5,
            has_double_penalty: true,
            marginal_dims: vec![2usize, 2usize],
        };
        let penalty = build_diagonal_penalty_from_kronecker(&kron_result, &[5.0, 7.0, 11.0]);

        let PirlsPenalty::Diagonal {
            diag,
            positive_indices,
            ..
        } = penalty
        else {
            panic!("expected diagonal Kronecker PIRLS penalty");
        };
        let expected = [11.0, 21.5, 10.5, 31.5];
        for (idx, expected_diag) in expected.iter().copied().enumerate() {
            assert!(
                (diag[idx] - expected_diag).abs() <= 1e-12,
                "diagonal {idx} got {}, expected {expected_diag}",
                diag[idx]
            );
        }
        assert_eq!(positive_indices, vec![0, 1, 2, 3]);
    }
}
