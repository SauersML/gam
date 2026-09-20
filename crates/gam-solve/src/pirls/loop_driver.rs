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
    GaussianFrozenRows,
    HessianCurvatureKind,
    // penalty types
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
    penalized_gradient_natural_scale,
    // compute functions
    calculate_deviance_from_eta,
    // edf helpers
    calculate_edf_with_penalty,
    calculate_edfwithworkspace_with_penalty,
    compute_constraint_kkt_diagnostics,
    computeworkingweight_derivatives_from_eta,
    inf_norm,
    pirls_data_log_kernel_from_eta,
    runworking_model_pirls,
    should_use_sparse_native_pirls,
    solve_penalized_least_squares_implicit,
    standard_inverse_link_jet,
    update_glmvectors,
};
use super::{GamModelFinalState, WorkingLikelihood, project_coefficients_to_lower_bounds};
use crate::active_set;
use crate::estimate::EstimationError;
use crate::gpu::pirls_host_dispatch::{try_gaussian_pls_gpu, try_pirls_loop_gpu};
use faer::sparse::{SparseColMat, Triplet};
use gam_linalg::matrix::{DesignMatrix, LinearOperator, ReparamOperator, SymmetricMatrix};
use gam_math::probability::standard_normal_quantile;
use gam_problem::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LinearPredictor, LinkFunction,
    LogSmoothingParamsView, MixtureLinkState, ResolvedLikelihoodScale, ResponseFamily,
    SasLinkState, StandardLink,
};
use gam_terms::construction::ReparamResult;
use ndarray::{ArcArray1, Array1, Array2, ArrayView1, ArrayView2, s};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Converged-η dispersion refreshes (Tweedie Pearson φ, Gaussian / inverse
/// Gaussian φ MLE) allowed after the reported solve. The φ map is a strong
/// contraction, so cold starts settle in 1–2 re-solves and warm starts in zero.
const MAX_PHI_REFRESH: usize = 5;
/// Relative φ change below which a re-solve cannot move any reported quantity
/// meaningfully (far under statistical resolution).
const PHI_REFRESH_REL_TOL: f64 = 1e-4;

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

/// Contract two vectors with a compensated accumulator and an exact product
/// split, so the result carries ~1 ulp of its own magnitude rather than ~n ulp.
///
/// This exists for one contraction: `qb^T (X^T W z)` in the Gaussian
/// zero-iteration synthesis below, where the #1033 n-free kappa-trial path
/// recovers the deviance as `z^T W z - 2 qb^T b + qb^T G qb`. The design is
/// never realized at the trial psi on that path, so the row-wise
/// `sum w (y - mu)^2` is unavailable and this cancellation is the only route to
/// the deviance. It is a large one by construction: measured on the #2624
/// fixture, `z^T W z = 3.13466938668704074e2` against a converged
/// `D_p = 4.0e-5` -- 7.1 orders. The profiled-Gaussian REML criterion then
/// multiplies the RELATIVE error of `D_p` by `(n - M_p)/2`, because its whole
/// `D_p` dependence is the single term `((n-M_p)/2) * ln(2 pi D_p/(n-M_p))`, so
/// at n = 600 an error in `D_p` reaches the outer surface 300x magnified.
/// Resolving this contraction to 1 ulp is therefore cheap insurance on a
/// quantity the outer surface is unusually sensitive to, and it costs less than
/// the spelling it replaced (one fewer matvec).
///
/// WHAT IT IS NOT. It was landed claiming to be the #2624 fix. **That claim was
/// measured and is false**, and the measurements are recorded here so the claim
/// is not re-made:
///
/// * Printing BOTH spellings on the SAME calls of the spatial fast path, the
///   compensation moves `D_p` by 1.8e-14 to 1.1e-13, while the call-to-call
///   variation of `D_p` at essentially one theta is 5.1e-12. So it perturbs the
///   value by ~1% of the noise it was supposed to remove; the dominant carrier
///   is upstream of this contraction (both spellings share `z^T W z` and the
///   tensor-served `gram_at(psi)` / `rhs_at(psi)`, so a common-mode error
///   cancels out of their difference and is invisible in it).
/// * Against the exact row deviance on the live-row lane, it improves the gap
///   by 2.32x at one point, 1.006x at another of the same depth, and 1.00x
///   elsewhere.
/// * On the non-spatial Python witness (`audit_outer_value_agreement`), a wheel
///   built from the landing commit reproduces `value-only`, `analytic-sample`
///   and `disagreement = 1.805e-5` BIT-IDENTICALLY to the pre-commit run.
///
/// The 8/13 -> 10/13 change in certified #2624 arms that accompanied the
/// landing is therefore NOT attributable to a quieter criterion. A perturbation
/// of this size reshuffles the outer trajectory, and on a fixture whose residual
/// failure mode is "the multistart certifies the wrong basin" that moves arms in
/// both directions -- which is exactly what was observed, `length_scale` 1.2,
/// 1.0 and 0.95 gaining and 0.7 regressing.
///
/// Neumaier compensation on the running sum, plus `mul_add` to recover the
/// exact product error. Falls back to the ordinary contraction on a length
/// mismatch so a shape bug surfaces where shapes are checked, not here.
fn compensated_dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    if a.len() != b.len() {
        return a.dot(b);
    }
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let product = x * y;
        // `x*y - product` exactly, when an FMA is available.
        compensation += f64::mul_add(x, y, -product);
        let next = sum + product;
        compensation += if sum.abs() >= product.abs() {
            (sum - next) + product
        } else {
            (product - next) + sum
        };
        sum = next;
    }
    sum + compensation
}

pub(crate) fn exact_lambdas_from_rho(rho: LogSmoothingParamsView<'_>) -> Array1<f64> {
    rho.exact_exp()
}

pub(super) fn default_beta_guess_external(
    p: usize,
    response: &ResponseFamily,
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
                // The Jeffreys-smoothed prevalence `(Σwy + ½)/(Σw + 1)` lies strictly
                // inside (0, 1) for any response in [0, 1] (until `Σw` nears 2⁵²), so
                // the link transforms below need no clamp (#2469).
                let prevalence = (weighted_sum + 0.5) / (totalweight + 1.0);
                beta[intercept_col] = match link_function {
                    LinkFunction::Logit => (prevalence / (1.0 - prevalence)).ln(),
                    LinkFunction::Probit => {
                        standard_normal_quantile(prevalence).unwrap_or_else(|err| {
                            // `prevalence` lies inside (0, 1); this fallback is
                            // only for defensive robustness under non-finite upstream inputs.
                            log::trace!(
                                "[PIRLS init] probit intercept seed: Φ⁻¹({prevalence:.6}) \
                                 failed ({err}); using the logit transform instead"
                            );
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
                        standard_normal_quantile(prevalence).unwrap_or_else(|err| {
                            log::trace!(
                                "[PIRLS init] intercept seed: Φ⁻¹({prevalence:.6}) failed \
                                 ({err}); using the logit transform instead"
                            );
                            (prevalence / (1.0 - prevalence)).ln()
                        })
                    }),
                    LinkFunction::BetaLogistic => solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or_else(|| {
                        standard_normal_quantile(prevalence).unwrap_or_else(|err| {
                            log::trace!(
                                "[PIRLS init] intercept seed: Φ⁻¹({prevalence:.6}) failed \
                                 ({err}); using the logit transform instead"
                            );
                            (prevalence / (1.0 - prevalence)).ln()
                        })
                    }),
                    // Outer arm guard already filtered out Log/Identity; fall
                    // back to the canonical logit transform for defensive safety
                    // if these are ever reached unexpectedly.
                    LinkFunction::Log
                    | LinkFunction::Identity
                    | LinkFunction::Sqrt
                    | LinkFunction::Inverse
                    | LinkFunction::InverseSquared => (prevalence / (1.0 - prevalence)).ln(),
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
        LinkFunction::Log if matches!(response, ResponseFamily::Binomial) => {
            // Relative-risk regression: the intercept-only root of the
            // Bernoulli score under `μ = exp(η)` is `η = ln p̂`. The
            // Jeffreys-smoothed prevalence `(Σwy + ½)/(Σw + 1)` lies strictly
            // inside (0, 1), so the seed lies strictly inside the feasible set
            // `η < 0` even when every response is one.
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                beta[intercept_col] = ((weighted_sum + 0.5) / (totalweight + 1.0)).ln();
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
                let mean_y = weighted_sum / totalweight;
                // A zero weighted mean (every response zero) has no finite
                // log: the intercept keeps its zero seed and the solve judges
                // the degenerate data itself, instead of starting at the log
                // of an invented floor (#2469).
                if mean_y > 0.0 {
                    beta[intercept_col] = mean_y.ln();
                }
            }
        }
        LinkFunction::Sqrt => {
            // The intercept-only root of every variance function's score under
            // `μ = η²` is `μ = ȳ` (weighted), i.e. `η = √ȳ`, inside the link's
            // branch `η > 0`. A non-positive mean has no such root; the
            // intercept keeps its zero seed and the solve reports the domain
            // violation itself.
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                let mean_y = weighted_sum / totalweight;
                if mean_y > 0.0 {
                    beta[intercept_col] = mean_y.sqrt();
                }
            }
        }
        LinkFunction::Inverse | LinkFunction::InverseSquared => {
            // The intercept-only root of every power-variance score under
            // `μ = η^(−a)` is `μ = ȳ` (weighted), i.e. `η = ȳ^(−1/a)`: `1/ȳ` for
            // the inverse link and `1/ȳ²` for the inverse-squared link. That
            // seed lies inside the link's domain `η > 0` on every row. A
            // non-positive mean has no such root; the intercept keeps its zero
            // seed and the solve reports the domain violation itself.
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                let mean_y = weighted_sum / totalweight;
                if mean_y > 0.0 {
                    beta[intercept_col] = match link_function {
                        LinkFunction::Inverse => mean_y.recip(),
                        _ => (mean_y * mean_y).recip(),
                    };
                }
            }
        }
    }
    beta
}

/// The Fisher working weights `w·(dμ/dη)²/V(μ)` at the cold P-IRLS start: the
/// coefficient guess [`default_beta_guess_external`] that an inner solve with
/// no warm start begins from, pushed through the same working update that
/// solve's first iterate makes. It plays the part of the working weight at
/// `mustart` in mgcv's `initial.sp`: it needs no solve, so it exists wherever
/// an inner solve could refuse, and it follows the response's units exactly as
/// the fitted working weight does, because the start mean is the weighted mean
/// of `y`. The start is the unconstrained guess; shape constraints and coefficient
/// bounds only move it inside their cone during the solve.
pub(crate) fn start_working_weights(
    x: &DesignMatrix,
    y: ArrayView1<'_, f64>,
    priorweights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    config: &PirlsConfig,
) -> Result<Array1<f64>, EstimationError> {
    let beta = default_beta_guess_external(
        x.ncols(),
        &config.likelihood.spec.response,
        config.link_function(),
        y,
        priorweights,
        config.link_kind.mixture_state(),
        config.link_kind.sas_state(),
    );
    let eta = x.matrixvectormultiply(&beta) + &offset;
    let n = eta.len();
    let mut mu = Array1::<f64>::zeros(n);
    let mut weights = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    match &config.link_kind {
        InverseLink::Standard(_) => config.likelihood.irls_update(
            y,
            &eta,
            priorweights,
            &mut mu,
            &mut weights,
            &mut z,
            None,
            None,
        )?,
        link => update_glmvectors(
            y,
            &eta,
            link,
            priorweights,
            &mut mu,
            &mut weights,
            &mut z,
            None,
        )?,
    }
    Ok(weights)
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
) -> Result<PirlsResult, EstimationError> {
    // #1868: the full-assembly path is legitimately O(n) (this is the one-off
    // final fit, not a per-callback n-free skip); wrap its freshly-realised row
    // arrays in the shared `ArcArray1` representation (`.into_shared()` moves the
    // owned buffer into an `Arc`, O(1)). `finalmu`/`solvemu` share one handle.
    let final_eta_arr = working_summary.state.eta.as_ref().clone();
    let finalmu_shared = finalmu.clone().into_shared();
    Ok(PirlsResult {
        likelihood,
        beta_transformed: working_summary.beta.clone(),
        penalized_hessian_transformed,
        stabilizedhessian_transformed,
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
        solve_c_nontrivial: final_c.iter().any(|&value| value != 0.0),
        solve_d_array: final_d.clone().into_shared(),
        derivatives_unsupported: false,
        status,
        iteration: working_summary.iterations,
        max_abs_eta: working_summary.max_abs_eta,
        lastgradient_norm: working_summary.lastgradient_norm,
        gradient_natural_scale: working_summary.state.gradient_natural_scale,
        penalized_gradient_transformed: working_summary.state.gradient.clone(),
        last_deviance_change: working_summary.last_deviance_change,
        last_step_halving: working_summary.last_step_halving,
        hessian_curvature: working_summary.state.hessian_curvature,
        exported_laplace_curvature: working_summary.exported_laplace_curvature.clone(),
        final_lm_lambda: working_summary.final_lm_lambda,
        final_accept_rho: working_summary.final_accept_rho,
        constraint_kkt: working_summary.constraint_kkt.clone(),
        final_kkt_tolerance: working_summary.final_kkt_tolerance,
        linear_constraints_transformed,
        reparam_result,
        x_transformed,
        coordinate_frame,
        used_device: false,
        cache_compacted: false,
        min_penalized_deviance: working_summary.min_penalized_deviance,
    })
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
    pub reparam_invariant: Option<&'a gam_terms::construction::ReparamInvariant>,
    pub p: usize,
    pub coefficient_lower_bounds: Option<&'a Array1<f64>>,
    pub linear_constraints_original: Option<&'a LinearInequalityConstraints>,
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
        None,
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
    // Shared invariant row carrier for a Gaussian value-only evaluation.
    //
    // `Some` requests sufficient-statistic-only result synthesis: beta,
    // deviance, gradient, and curvature remain exact, while observation-scale
    // fields share these placeholders instead of recomputing `X beta`.
    // Full gradients and accepted fits always pass `None`.
    cost_only_gaussian_rows: Option<&Arc<GaussianFrozenRows>>,
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
    let lambdas = exact_lambdas_from_rho(rho);
    let lambdas_slice = lambdas.as_slice_memory_order().ok_or_else(|| {
        EstimationError::InvalidInput("non-contiguous lambda storage".to_string())
    })?;

    let likelihood = &config.likelihood;
    // Resolve family and scalar ownership once at the fit boundary. This makes
    // malformed family/metadata pairs fail before either the CPU or GPU path
    // can interpret an absent scalar as a unit value.
    let resolved_likelihood_scale = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let link_function = config.link_function();

    use gam_terms::construction::{
        EngineDims, stable_reparameterization_engine_canonical,
        stable_reparameterization_original_frame,
    };

    // Build a cheap weighted penalty sum for the sparse-native decision
    // WITHOUT running the expensive eigendecomposition engine.
    // The full reparameterization is deferred until we know which path we need.
    let mut cheap_s_lambda = Array2::<f64>::zeros((penalty.p, penalty.p));
    for (k, cp) in penalty.canonical_penalties.iter().enumerate() {
        let lam = lambdas_slice.get(k).copied().unwrap_or(0.0);
        if lam != 0.0 {
            cp.accumulate_weighted(&mut cheap_s_lambda, lam);
        }
    }

    let x_original: DesignMatrix = x.into();
    // Auto-detect sparse structure in dense designs so the sparse-native path
    // can engage for structurally sparse models that happen to be stored dense.
    //
    // A Gaussian value-only probe already owns the exact dense coefficient
    // statistics consumed by its solve. Scanning all design rows here to
    // rediscover a sparse representation cannot change that solve and would
    // make every rho candidate O(n) before the sufficient-statistic lane even
    // begins (#2435).
    let x_original = if cost_only_gaussian_rows.is_some() {
        x_original
    } else {
        let auto_sparse = x_original
            .as_dense()
            .and_then(|dense| sparse_from_denseview(dense.view()));
        auto_sparse.unwrap_or(x_original)
    };
    // A value-only Gaussian probe is already represented completely by its
    // coefficient-space sufficient statistics and shared frozen row carrier.
    // It exits through the exact zero-iteration branch below, so constructing
    // the general workspace with n rows would allocate five length-n scratch
    // vectors that no consumer reads (#2435). Full gradients, final fits, and
    // every iterative family retain the ordinary observation workspace.
    let mut workspace = if cost_only_gaussian_rows.is_some() {
        PirlsWorkspace::coefficient_only(x_original.ncols())
    } else {
        PirlsWorkspace::new(x_original.nrows(), x_original.ncols())
    };
    // A value-only Gaussian probe on a sparse design whose cache carries the
    // sparse `XᵀWX` takes the same solve path as the full evaluations of that
    // fit: the sparse solve reads only coefficient-space statistics, and the
    // REML geometry it feeds is the sparse exact one only in these coordinates.
    let cost_only_without_sparse_gram = cost_only_gaussian_rows.is_some()
        && !(x_original.as_sparse().is_some()
            && gaussian_fixed_cache.is_some_and(|cache| cache.xtwx_sparse_orig.is_some()));
    let solver_decision = if cost_only_without_sparse_gram {
        SparsePirlsDecision {
            path: PirlsLinearSolvePath::DenseTransformed,
            reason: "gaussian_sufficient_statistics",
            p: x_original.ncols(),
            nnz_x: None,
            nnz_xtwx_symbolic: None,
            nnz_s_lambda: 0,
            nnz_h_est: None,
            density_h_est: None,
        }
    } else {
        should_use_sparse_native_pirls(
            &mut workspace,
            &x_original,
            &cheap_s_lambda,
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
    // declared `s_transformed`/`e_transformed`.
    let dense_reparam_result = if !use_sparse_native {
        Some(stable_reparameterization_engine_canonical(
            penalty.canonical_penalties,
            lambdas_slice,
            EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
            penalty.reparam_invariant,
        )?)
    } else {
        None
    };
    // Sparse-native reparameterization in identity (original) coordinates:
    // the same declared penalty, log-determinant and traces as the engine,
    // computed block by block so a random effect with thousands of levels
    // never meets a p×p factorization or frame change.
    let sparse_native_reparam = if use_sparse_native {
        Some(stable_reparameterization_original_frame(
            penalty.canonical_penalties,
            lambdas_slice,
            EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
            penalty.reparam_invariant,
        )?)
    } else {
        None
    };
    let qs_arc = dense_reparam_result
        .as_ref()
        .map(|reparam_result| Arc::new(reparam_result.qs.clone()));
    let transform_active = if use_sparse_native {
        None
    } else {
        Some(WorkingReparamTransform::Dense(Arc::clone(
            qs_arc
                .as_ref()
                .expect("dense Qs should exist for the transformed path"),
        )))
    };
    let mut penalty_active = if use_sparse_native {
        // Sparse-native inner penalty in original (identity) coordinates. Use
        // the reparameterized declared root and Gram so `H = XᵀWX + S` matches
        // the penalty whose log-determinant REML reports.
        let sparse_reparam = sparse_native_reparam
            .as_ref()
            .expect("sparse_native_reparam should be present for sparse-native path");
        PirlsPenalty::Dense {
            s_transformed: sparse_reparam.s_transformed.clone(),
            e_transformed: sparse_reparam.e_transformed.clone(),
            linear_shift: Array1::zeros(penalty.p),
            constant_shift: 0.0,
        }
    } else {
        let dense = dense_reparam_result
            .as_ref()
            .expect("dense reparam result should be present on the dense path");
        PirlsPenalty::Dense {
            s_transformed: dense.s_transformed.clone(),
            e_transformed: dense.e_transformed.clone(),
            linear_shift: Array1::zeros(penalty.p),
            constant_shift: 0.0,
        }
    };
    let (shift_original, shift_constant) =
        canonical_prior_shift(penalty.canonical_penalties, lambdas_slice, penalty.p);
    let shift_active = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(&shift_original))
        .unwrap_or(shift_original);
    attach_penalty_shift(&mut penalty_active, shift_active, shift_constant);
    // Build transformed constraints now that dense_reparam_result is available.
    let linear_constraints = if let Some(reparam) = dense_reparam_result.as_ref() {
        let tb = build_transformed_lower_bound_constraints(
            &reparam.qs,
            penalty.coefficient_lower_bounds,
        )?;
        let tl =
            build_transformed_linear_constraints(&reparam.qs, penalty.linear_constraints_original)?;
        merge_linear_constraints(tb, tl)?
    } else {
        // Sparse-native without dense reparam: constraints stay in original
        // coordinates (identity Qs).  Use an identity matrix of appropriate size.
        let p = penalty.p;
        let qs_identity = Array2::<f64>::eye(p);
        let tb = build_transformed_lower_bound_constraints(
            &qs_identity,
            penalty.coefficient_lower_bounds,
        )?;
        let tl =
            build_transformed_linear_constraints(&qs_identity, penalty.linear_constraints_original)?;
        merge_linear_constraints(tb, tl)?
    };

    let coordinate_frame = if use_sparse_native {
        PirlsCoordinateFrame::OriginalSparseNative
    } else {
        PirlsCoordinateFrame::TransformedQs
    };
    let materialize_final_reparam_result = || -> Result<ReparamResult, EstimationError> {
        if use_sparse_native {
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
                .expect("dense reparam result should be present on the dense path")
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
        // gh#2544: this dispatch pre-empts the Gaussian-Identity zero-iteration
        // branch below on the same admission predicate, so the #1868 frozen-row
        // bundle has to reach it too — otherwise the optimisation is live on the
        // branch that never runs and dead on the one that does.
        cost_only_gaussian_rows,
    ) {
        return result;
    }

    if matches!(link_function, LinkFunction::Identity)
        && likelihood.spec.is_gaussian_identity()
        && linear_constraints.is_none()
    {
        // Gaussian-Identity zero-iteration exact solve. The unconstrained
        // penalized least-squares system is linear, so for a Gaussian identity
        // model a single solve is the exact minimizer and no PIRLS iteration is
        // needed. Other identity-link families (Student-t) have a non-quadratic
        // likelihood and take the iterative loop below.
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
            cache_for_solve,
        )?;

        let beta_transformed = pls_result.beta;
        let penalized_hessian = pls_result.penalized_hessian;
        let edf = pls_result.edf;

        // eta = offset + X Qs beta (composed, no materialization) unless a
        // design-moving ψ tensor cache explicitly says the surface rows are a
        // stale reference. In that lane the Gaussian objective and gradient are
        // fully determined by (G, r, y'Wy), so applying `x_original` would both
        // reintroduce per-trial row work and evaluate the wrong ψ.
        let qbeta = transform_active
            .as_ref()
            .map(|transform| transform.apply(beta_transformed.as_ref()))
            .unwrap_or_else(|| beta_transformed.as_ref().clone());
        let sufficient_only_row_cache = cache_for_solve.filter(|cache| {
            cache.row_prediction_is_stale || cost_only_gaussian_rows.is_some()
        });

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
            /// The data score's operands `XᵀWη` and `XᵀWz` in the active basis,
            /// whose difference is `gradient_data`: the natural gradient scale
            /// is built from them (#3339).
            score_operands: [Array1<f64>; 2],
            deviance: f64,
            log_likelihood: f64,
            max_abs_eta: f64,
        }

        // Original-basis coefficient vectors to the active basis the gradient
        // is reported in.
        let to_active_basis = |v: Array1<f64>| {
            transform_active
                .as_ref()
                .map(|transform| transform.apply_transpose(&v))
                .unwrap_or(v)
        };
        let rows = if let Some(cache) = sufficient_only_row_cache {
            // #1868 FAST PATH: the criterion, gradient and inner solve are served
            // entirely from k-space Gram sufficient statistics; the length-`n`
            // row arrays are trial-invariant placeholders (η≡μ≡offset, z≡y,
            // w≡priorweights, constant Gaussian working-weight derivatives). When
            // the producer attached the once-built frozen bundle we clone its
            // `ArcArray1` handles (O(1), zero element touches) instead of
            // re-materialising ~16·n elements per κ callback — the #1868 fix.
            let gram_qbeta = cache.xtwx_orig.dot(&qbeta);
            let mut grad_orig = gram_qbeta.clone();
            grad_orig -= &cache.xtwy_orig;
            let score_operands = [
                to_active_basis(gram_qbeta),
                to_active_basis(cache.xtwy_orig.clone()),
            ];
            // #2624: `z^T W z - 2 qb^T b + qb^T G qb` regrouped as
            // `(z^T W z - qb^T b) + qb^T (G qb - b)`. The two are the same
            // number in exact arithmetic; they are not the same computation.
            // `G qb - b` is formed elementwise above and equals `-S beta` at the
            // inner mode, so the second contraction is over SMALL entries and
            // its absolute error is negligible -- which leaves exactly one
            // contraction at the `z^T W z` magnitude for `compensated_dot` to
            // resolve. It also drops a matvec, so it is cheaper than the
            // spelling it replaces. See `compensated_dot` for the measured size
            // of what this buys, which is much smaller than the claim it was
            // landed under.
            let residual_inner = qbeta.dot(&grad_orig);
            let gradient_data = to_active_basis(grad_orig);
            let weighted_rss = (cache.centered_weighted_y_sq
                - compensated_dot(&qbeta, &cache.xtwy_orig)
                + residual_inner)
                .max(0.0);
            match resolved_likelihood_scale {
                ResolvedLikelihoodScale::ProfiledGaussian
                | ResolvedLikelihoodScale::FixedGaussian { .. } => {}
                other => {
                    return Err(EstimationError::InvalidInput(format!(
                        "Gaussian identity cache received non-Gaussian resolved scale {other:?}"
                    )));
                }
            }
            // Conventional Gaussian deviance is raw weighted RSS for both a
            // profiled and a fixed dispersion. Scale enters the fixed
            // likelihood kernel and working curvature, never this reporting
            // statistic.
            let deviance = weighted_rss;

            if let Some(bundle) = cost_only_gaussian_rows.or(cache.frozen_rows.as_ref()) {
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
                    score_operands,
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
                        y,
                        &final_eta,
                        priorweights_owned.view(),
                    )?;
                let log_likelihood = pirls_data_log_kernel_from_eta(
                    y,
                    &final_eta,
                    likelihood,
                    &config.link_kind,
                    priorweights,
                    deviance,
                )?;
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
                    score_operands,
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
            let gradient_data = to_active_basis(xt_wr);
            let score_operands = [
                to_active_basis(x_original.apply_transpose(&(&finalmu * &priorweights_owned))),
                to_active_basis(x_original.apply_transpose(&(&y * &priorweights_owned))),
            ];
            let deviance = calculate_deviance_from_eta(
                y,
                &final_eta,
                likelihood,
                &config.link_kind,
                priorweights,
            )?;
            let log_likelihood = pirls_data_log_kernel_from_eta(
                y,
                &final_eta,
                likelihood,
                &config.link_kind,
                priorweights,
                deviance,
            )?;
            let max_abs_eta = inf_norm(finalmu.iter().copied());
            let (c, d, dmu_deta, d2mu_deta2, d3mu_deta3) =
                computeworkingweight_derivatives_from_eta(
                    &config.likelihood,
                    &config.link_kind,
                    y,
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
                score_operands,
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
            score_operands: [xt_w_eta, xt_w_z],
            deviance,
            log_likelihood,
            max_abs_eta,
        } = rows;
        let s_beta = penalty_active.shifted_gradient(beta_transformed.as_ref());
        let gradient_natural_scale = penalized_gradient_natural_scale(&xt_w_eta, &xt_w_z, &s_beta);
        let mut gradient = gradient_data;
        gradient += &s_beta;
        let penalty_term = penalty_active.shifted_quadratic(beta_transformed.as_ref());
        // `solve_penalized_least_squares_implicit` assembles `H = XᵀWX + S_λ`
        // with no stabilization ridge on both of its branches (#2901 V22), so
        // `penalized_hessian` is the exact matrix the outer criterion reads, and
        // `penalty_term` and the gradient carry no ridge term either.
        let stabilizedhessian = penalized_hessian.clone();

        let gradient_norm = array1_l2_norm(&gradient);
        let working_state = WorkingState {
            eta: working_eta,
            gradient: gradient.clone(),
            hessian: penalized_hessian.clone(),

            log_likelihood,
            deviance,
            deviance_magnitude: deviance.abs(),
            penalty_term,
            firth: FirthDiagnostics::Inactive,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale,
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
                compute_constraint_kkt_diagnostics(
                    beta_transformed.as_ref(),
                    &gradient,
                    gradient_natural_scale,
                    lin,
                )
            }),
            min_penalized_deviance: if zero_iter_penalized.is_finite() {
                zero_iter_penalized
            } else {
                f64::INFINITY
            },
            // Zero-iteration synthesis: the closed form is exact and no
            // certificate was evaluated, so there is no tolerance that decided
            // anything here.
            final_kkt_tolerance: None,
            // Zero-iteration synthesis: no LM damping was exercised, so there is
            // no hint to hand on; every warm-start consumer reads 0 as none (#2469).
            final_lm_lambda: 0.0,
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
            solve_c_nontrivial: false,
            solve_d_array,
            derivatives_unsupported: false,
            status: PirlsStatus::Converged,
            iteration: 1,
            max_abs_eta,
            lastgradient_norm: gradient_norm,
            gradient_natural_scale,
            penalized_gradient_transformed: gradient.clone(),
            last_deviance_change: 0.0,
            last_step_halving: 0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            exported_laplace_curvature: working_summary.exported_laplace_curvature.clone(),
            final_lm_lambda: working_summary.final_lm_lambda,
            final_accept_rho: working_summary.final_accept_rho,
            constraint_kkt: working_summary.constraint_kkt.clone(),
        final_kkt_tolerance: working_summary.final_kkt_tolerance,
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
            && config.link_kind.has_fisher_weight_jet(),
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
                &config.likelihood.spec.response,
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
        // (the convex/concave second-difference rows, where every row is active at
        // `β = 0` under the solver's own activity tolerance
        // `|b_i| ≤ ACTIVE_SET_PRIMAL_FEASIBILITY_TOL·‖a_i‖`, in the geometric units
        // the active face uses, not an absolute band on raw `b`; #2469) that function
        // returns the minimum-norm feasible point `β = 0`, which is the cone
        // *vertex*: the exact all-rows-tight degenerate seed #873 is about. Taking
        // it would silently reintroduce the #873 pathology whenever the strict
        // projection rarely fails to certify. So for a homogeneous cone we skip the
        // vertex fallback entirely and prefer the data-driven `initial_beta`: it
        // violates at most *some* rows (a lower-dimensional, non-degenerate face the
        // inner active-set QP can recover from), strictly better than the vertex
        // where *every* row is simultaneously tight.
        let cone_is_homogeneous = (0..constraints.b.len()).all(|row| {
            let norm = constraints.a.row(row).dot(&constraints.a.row(row)).sqrt();
            constraints.b[row].abs() <= crate::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL * norm
        });
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
        && config.link_kind.has_fisher_weight_jet();
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
        // together collapsed step halving to 3 under a low outer-imposed
        // iteration cap, turning recoverable
        // damping into spurious failures.
        max_step_halving: base_max_step_halving,
        firth_bias_reduction: firth_active,
        coefficient_lower_bounds: None,
        linear_constraints: linear_constraints.clone(),
        initial_lm_lambda: config.initial_lm_lambda,
    };

    let mut iteration_logger = |info: &WorkingModelIterationInfo| {
        log::trace!(
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
        Some(&mut iteration_logger),
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
    // `warm_start_beta = None`, and the first outer eval starts from a default guess),
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
    // λ. At convergence the reported dispersion is the Gamma shape that makes
    // the Laplace marginal likelihood stationary at the converged mean (the
    // profile score less the edf/(2ν) charge of −½ log|H|, #4075), and the final
    // working state — `finalweights`, the penalized Hessian, the deviance, μ —
    // is rebuilt with that same ν, so `Vb = H⁻¹·φ̂` stays internally consistent.
    // Warm-started solves (every REML cost eval) already sit near the converged
    // η, so the first refresh check confirms ν and exits without a re-solve; the
    // added cost there is a single O(n) shape evaluation.
    let gamma_scale = working_model
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if refine_dispersion_at_converged_eta
        && matches!(
            gamma_scale,
            ResolvedLikelihoodScale::Gamma {
                estimated: true,
                ..
            }
        )
    {
        // A few passes suffice: the converged-η shape map is a strong
        // contraction (β̂ barely moves once the mean is captured), so cold
        // starts settle in 1–2 re-solves and warm starts in zero.
        const MAX_SHAPE_REFRESH: usize = 5;
        // Relative shape tolerance below which a re-solve cannot move any
        // reported quantity meaningfully (far under statistical resolution).
        const SHAPE_REFRESH_REL_TOL: f64 = 1e-4;
        for refresh_iter in 0..MAX_SHAPE_REFRESH {
            // The shape is the stationary point of the Laplace marginal
            // likelihood at fixed λ, whose −½ log|H| term charges the edf of
            // the mean model (#4075); the edf is read off the current Hessian.
            let mean_model_edf =
                calculate_edf_with_penalty(&working_summary.state.hessian, &penalty_active)?;
            let refreshed_shape = super::estimate_gamma_shape_from_eta(
                &working_model.likelihood.spec.link,
                y,
                working_summary.state.eta.as_ref(),
                priorweights,
                mean_model_edf,
            )?;
            let prior_shape = working_model
                .likelihood
                .resolved_gamma_shape()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let rel_change = (refreshed_shape - prior_shape).abs() / prior_shape;
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
                // non-contracting alternation). The working state — β̂, weights,
                // Hessian, EDF — was solved at the PREVIOUS shape, which differs
                // from the just-installed one by more than the tolerance; the
                // shape rescales the penalized objective `k·D + βᵀSβ`, so β̂ is
                // not stationary at the reported shape. That is not a joint
                // (β, shape) fixed point and may not be reported as a fit
                // (#3544), exactly as the Gaussian φ refresh below refuses.
                crate::bail_invalid_estim!(
                    "Gamma shape did not reach its converged-η fixed point within \
                     {MAX_SHAPE_REFRESH} re-solves (relative change {rel_change:e} > \
                     tolerance {SHAPE_REFRESH_REL_TOL:e})"
                );
            }
            // The shape moved: re-solve β at the corrected shape, warm-started
            // at the converged β, so the final working state is rebuilt with the
            // refreshed ν.
            working_summary = runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                Some(&mut iteration_logger),
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
    let tweedie_scale = working_model
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if refine_dispersion_at_converged_eta
        && matches!(
            tweedie_scale,
            ResolvedLikelihoodScale::Tweedie {
                estimated: true,
                ..
            }
        )
    {
        if let ResponseFamily::Tweedie { p } = working_model.likelihood.spec.response {
            // The converged-η Pearson map is a strong contraction (β̂ scale-free
            // here), so cold starts settle in 1–2 re-solves and warm starts in
            // zero.
            for refresh_iter in 0..MAX_PHI_REFRESH {
                // Pearson moment on the residual degrees of freedom n₊ − edf
                // (#4075), with the edf of the current converged mean model.
                let mean_model_edf =
                    calculate_edf_with_penalty(&working_summary.state.hessian, &penalty_active)?;
                let refreshed_phi = super::estimate_tweedie_phi_from_eta(
                    y,
                    working_summary.state.eta.as_ref(),
                    priorweights,
                    p,
                    mean_model_edf,
                )?;
                let prior_phi = working_model
                    .likelihood
                    .resolved_tweedie_phi()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                let rel_change = (refreshed_phi - prior_phi).abs() / prior_phi;
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
                    // Final allowed pass and φ is still drifting: the working
                    // state was solved at a φ that differs from the installed one
                    // by more than the tolerance (φ rescales the effective
                    // penalty, so β̂ is not stationary at the reported φ). Not a
                    // joint (β, φ) fixed point, so not a fit (#3544).
                    crate::bail_invalid_estim!(
                        "Tweedie dispersion φ did not reach its converged-η fixed point \
                         within {MAX_PHI_REFRESH} re-solves (relative change {rel_change:e} > \
                         tolerance {PHI_REFRESH_REL_TOL:e})"
                    );
                }
                // φ moved materially: re-solve β at the corrected φ, warm-started
                // at the converged β, so the final working state is rebuilt with
                // the refreshed φ.
                working_summary = runworking_model_pirls(
                    &mut working_model,
                    working_summary.beta.clone(),
                    &options,
                    Some(&mut iteration_logger),
                )?;
            }
        }
    }

    // ── Gaussian (non-identity link) / inverse Gaussian dispersion φ ─────────
    //
    // The same converged-η refresh as the Tweedie φ above, with the scale that
    // makes the Laplace marginal likelihood stationary at fixed λ,
    // `φ̂ = Σ wᵢ dᵢ / (n₊ − edf)` (#4075), in place of the Pearson moment. As in
    // every converged-η refresh, a φ still moving on the last allowed pass is a
    // failed fit, not a reported one: the reported φ must be the estimate at the
    // reported η.
    if refine_dispersion_at_converged_eta
        && matches!(
            working_model
                .likelihood
                .resolved_scale()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
            ResolvedLikelihoodScale::Dispersion {
                estimated: true,
                ..
            }
        )
    {
        let mut converged = false;
        for _ in 0..MAX_PHI_REFRESH {
            let mean_model_edf =
                calculate_edf_with_penalty(&working_summary.state.hessian, &penalty_active)?;
            let refreshed_phi = super::estimate_dispersion_phi_from_eta(
                &working_model.likelihood.spec.response,
                &working_model.likelihood.spec.link,
                y,
                working_summary.state.eta.as_ref(),
                priorweights,
                mean_model_edf,
            )?;
            let prior_phi = working_model
                .likelihood
                .resolved_dispersion_phi()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let rel_change = (refreshed_phi - prior_phi).abs() / prior_phi;
            working_model.likelihood = working_model
                .likelihood
                .clone()
                .with_dispersion_phi(refreshed_phi);
            working_model.dispersion_phi_locked = true;
            if rel_change <= PHI_REFRESH_REL_TOL {
                converged = true;
                break;
            }
            working_summary = runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                Some(&mut iteration_logger),
            )?;
        }
        if !converged {
            crate::bail_invalid_estim!(
                "dispersion φ did not reach its converged-η fixed point within {MAX_PHI_REFRESH} \
                 re-solves (relative tolerance {PHI_REFRESH_REL_TOL:e})"
            );
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
    let beta_scale = working_model
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if refine_dispersion_at_converged_eta
        && matches!(
            beta_scale,
            ResolvedLikelihoodScale::BetaPrecision {
                estimated: true,
                ..
            }
        )
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
            )?;
            let prior_phi = working_model
                .likelihood
                .resolved_beta_precision()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let rel_change = (refreshed_phi - prior_phi).abs() / prior_phi;
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
                // Final allowed pass and φ is still drifting: the mean was solved
                // at a precision that differs from the installed one by more than
                // the tolerance, and φ feeds back through the digamma mean score,
                // so β̂ is not stationary at the reported φ. Not a joint (β, φ)
                // fixed point, so not a fit (#3544).
                crate::bail_invalid_estim!(
                    "Beta precision φ did not reach its converged-η fixed point within \
                     {MAX_PHI_REFRESH} re-solves (relative change {rel_change:e} > \
                     tolerance {PHI_REFRESH_REL_TOL:e})"
                );
            }
            // φ moved materially: re-solve β at the corrected φ, warm-started at
            // the converged β, so the mean is refit under the better precision
            // and the final working state is rebuilt consistently.
            //
            // Every pass of the alternation must land AT A MINIMUM of the mean
            // objective. The moment estimate reads the Pearson residuals of
            // whatever η the re-solve returned; continuing from a mean that
            // exhausted its step search or its iteration budget feeds the next
            // pass a precision that no minimum produced. Measured on a
            // noise-free logistic response (`beta_regression_fits_clean_
            // monotone_separation_prone` before its fixture carried
            // dispersion): φ ran 1.1 → 6.4 → 50 → 1.6e3 → 1.4e6 → 1.0e12, the
            // re-solve at 1.0e12 ended `LM step search exhausted`, the loop still
            // refreshed φ from that η to 5.1e23, and the fit died inside the next
            // P-IRLS with `did not converge within 300 iterations, gradient
            // 9.5e12` — a message about the wrong object. A fixed λ rescales
            // nothing when φ grows, so the effective penalty λ/φ vanishes, the
            // mean interpolates, the residuals collapse and φ has no finite
            // fixed point: that is the fact to report, at the pass where it
            // became measurable.
            //
            // `StalledAtValidMinimum` continues the alternation: it is the
            // objective-resolution exit of a mean that IS at its minimum (the
            // deviance is ∝ φ, so at a large finite φ the strict certificate is
            // routinely out of the objective's floating-point reach), and the
            // same fixture with dispersion reached φ ≈ 1.5e5 through exactly
            // such passes before its next pass converged. Whether the FINAL
            // mean is certified is fit assembly's own gate, unchanged here.
            let deviance_at_prior = working_summary.state.deviance;
            let refused = |inner_status: String| {
                EstimationError::BetaPrecisionRefinementDidNotConverge {
                    passes: refresh_iter + 1,
                    prior_phi,
                    refreshed_phi,
                    deviance: deviance_at_prior,
                    inner_status,
                }
            };
            working_summary = match runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                Some(&mut iteration_logger),
            ) {
                Ok(summary) => summary,
                Err(inner) => return Err(refused(inner.to_string())),
            };
            if !matches!(
                working_summary.status,
                PirlsStatus::Converged | PirlsStatus::StalledAtValidMinimum
            ) {
                return Err(refused(working_summary.status.label().to_string()));
            }
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
    let negbin_scale = working_model
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if refine_dispersion_at_converged_eta
        && matches!(
            negbin_scale,
            ResolvedLikelihoodScale::NegativeBinomial {
                estimated: true,
                ..
            }
        )
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
            )?;
            let prior_theta = working_model
                .likelihood
                .resolved_negbin_theta()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let rel_change = (refreshed_theta - prior_theta).abs() / prior_theta;
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
                // Final allowed pass and θ is still drifting: the mean was solved
                // under a variance function whose θ differs from the installed one
                // by more than the tolerance, and θ enters the NB2 working
                // response, so β̂ is not stationary at the reported θ. Not a joint
                // (β, θ) fixed point, so not a fit (#3544).
                crate::bail_invalid_estim!(
                    "negative-binomial θ did not reach its converged-η fixed point within \
                     {MAX_THETA_REFRESH} re-solves (relative change {rel_change:e} > \
                     tolerance {THETA_REFRESH_REL_TOL:e})"
                );
            }
            // θ moved materially: re-solve β at the corrected θ, warm-started at
            // the converged β, so the mean is refit under the better variance
            // function and the final working state is rebuilt consistently.
            working_summary = runworking_model_pirls(
                &mut working_model,
                working_summary.beta.clone(),
                &options,
                Some(&mut iteration_logger),
            )?;
        }
    }

    // Candidate screens and a rejected post-loop polish are speculative
    // mutations of the model's row-space scratch. Reinstall the certified
    // coefficient state's arrays only when that scratch no longer carries its
    // exact coefficient identity; the common accepted-state path is an O(p)
    // bit comparison and performs no extra curvature work.
    working_model.refresh_working_arrays_for_state(
        &working_summary.beta,
        &working_summary.state,
        "finalization",
    )?;

    // Extract workspace before consuming working_model so we can reuse
    // the pre-allocated buffers in calculate_edfwithworkspace_with_penalty.
    // into_final_state() drops the workspace field anyway (it uses `..` in
    // its destructure); we replace it with a zero-sized stub to satisfy the
    // borrow checker, then keep the real workspace alive for the EDF call.
    let mut saved_workspace = std::mem::replace(
        &mut working_model.workspace,
        PirlsWorkspace::new(0, 0),
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
    // Keep P-IRLS's exact Hessian so outer LAML derivatives stay consistent:
    // H_eff = X'W_H X + S_λ, with no stabilization ridge (#2901 V22).
    let penalized_hessian_transformed = working_summary.state.hessian.clone();
    let stabilizedhessian_transformed = penalized_hessian_transformed.clone();
    // Use the workspace-backed variant for the dense path to reuse the
    // `final_aug_matrix` allocation; the sparse path still allocates
    // internally because no pre-computed factor is available at this site.
    let edf = if let Some(dense_h) = penalized_hessian_transformed.as_dense() {
        calculate_edfwithworkspace_with_penalty(dense_h, &penalty_active, &mut saved_workspace)?
    } else {
        calculate_edf_with_penalty(&penalized_hessian_transformed, &penalty_active)?
    };

    // An exhausted iteration budget stays an exhausted budget. The loop's own
    // post-loop soft acceptance (`pirls_soft_acceptance`) has already decided
    // whether this state is a near-stationary plateau. There is no second rescue
    // here that relabels a max-iteration stop as `StalledAtValidMinimum` because
    // the step collapsed to a floor, which certifies nothing about stationarity
    // (SPEC rule 21, #2902).
    let status = working_summary.status;

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
    )?;

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

// A constraint whose shape does not match the coefficient vector is a caller
// error. It is refused, never dropped: dropping it would silently turn a
// constrained fit into an unconstrained one.
pub(super) fn build_transformed_lower_bound_constraints(
    qs: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Result<Option<LinearInequalityConstraints>, EstimationError> {
    let Some(lb) = coefficient_lower_bounds else {
        return Ok(None);
    };
    if lb.len() != qs.nrows() {
        crate::bail_invalid_estim!(
            "coefficient lower bounds have length {} but the model has {} coefficients",
            lb.len(),
            qs.nrows()
        );
    }
    let activerows: Vec<usize> = (0..lb.len()).filter(|&i| lb[i].is_finite()).collect();
    if activerows.is_empty() {
        return Ok(None);
    }
    let mut a = Array2::<f64>::zeros((activerows.len(), qs.ncols()));
    let mut b = Array1::<f64>::zeros(activerows.len());
    for (r, &idx) in activerows.iter().enumerate() {
        a.row_mut(r).assign(&qs.row(idx));
        b[r] = lb[idx];
    }
    LinearInequalityConstraints::new(a, b)
        .map(Some)
        .map_err(EstimationError::InvalidInput)
}

pub(super) fn build_transformed_linear_constraints(
    qs: &Array2<f64>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Result<Option<LinearInequalityConstraints>, EstimationError> {
    let Some(lc) = linear_constraints else {
        return Ok(None);
    };
    if lc.a.ncols() != qs.nrows() {
        crate::bail_invalid_estim!(
            "linear constraint matrix has {} columns but the model has {} coefficients",
            lc.a.ncols(),
            qs.nrows()
        );
    }
    LinearInequalityConstraints::new(lc.a.dot(qs), lc.b.clone())
        .map(Some)
        .map_err(EstimationError::InvalidInput)
}

pub(super) fn merge_linear_constraints(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Result<Option<LinearInequalityConstraints>, EstimationError> {
    Ok(match (first, second) {
        (None, None) => None,
        (Some(c), None) | (None, Some(c)) => Some(c),
        (Some(c1), Some(c2)) => {
            if c1.a.ncols() != c2.a.ncols() {
                crate::bail_invalid_estim!(
                    "cannot merge constraint blocks with {} and {} columns",
                    c1.a.ncols(),
                    c2.a.ncols()
                );
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
    })
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

    // Structural sparsity is exact zeros: dropping a small nonzero would change
    // the matrix the solve sees, not store it more cheaply.
    let total = nrows.saturating_mul(ncols);
    if total == 0 {
        return None;
    }
    // If a matrix exceeds this nnz count it is too dense for sparse path; bail early.
    let sparse_nnz_limit = ((total as f64) * SPARSE_DENSITY_LIMIT).floor() as usize;
    let mut nnz = 0usize;
    for &val in x.iter() {
        if val != 0.0 {
            nnz += 1;
            if nnz > sparse_nnz_limit {
                return None;
            }
        }
    }
    let mut triplets = Vec::with_capacity(nnz);
    for (row_idx, row) in x.outer_iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            if val != 0.0 {
                triplets.push(Triplet::new(row_idx, col_idx, val));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .ok()
        .map(DesignMatrix::from)
}
