//! Outer warm-start carriers, blockwise-fit assembly-from-parts, EDF
//! helpers, and the warm-start result types, split out of
//! `outer_objective.rs` by concern (#1145). Re-exported via `custom_family`.

use super::*;
use gam_problem::{ensure_finite_scalar_estimation, validate_all_finite_estimation};

pub(crate) fn screened_outer_warm_start<'a>(
    warm_start: Option<&'a ConstrainedWarmStart>,
    rho: &Array1<f64>,
) -> Option<&'a ConstrainedWarmStart> {
    warm_start.filter(|seed| seed.rho.len() == rho.len())
}

pub(crate) fn cached_inner_mode_from_result(result: &BlockwiseInnerResult) -> CachedInnerMode {
    CachedInnerMode {
        solved_inner_tol: result.solved_inner_tol,
        log_likelihood: result.log_likelihood,
        penalty_value: result.penalty_value,
        cycles: result.cycles,
        converged: result.converged,
        block_logdet_h: result.block_logdet_h,
        block_logdet_s: result.block_logdet_s,
        joint_workspace: result.joint_workspace.clone(),
        kkt_residual: result.kkt_residual.clone(),
        active_constraints: result.active_constraints.clone(),
        terminal_working_sets: result.terminal_working_sets.clone(),
        terminal_likelihood_score: result.terminal_likelihood_score.clone(),
        rho_mode_responses: None,
        objective_state: result.objective_state.clone(),
    }
}

/// Preserve the blockwise solver's complete terminal verdict at an outer
/// trial-point boundary.
///
/// Every custom-family outer route has the exact inner result in hand when it
/// learns that derivatives cannot be exposed. Constructing the refusal here
/// gives that verdict one schema and one owner; storing a sentence in
/// `CustomOuterState` and later stripping wrapper prefixes from it loses the
/// terminal decision variables #2658 exists to surface.
pub(crate) fn inner_solve_not_converged_error(
    inner: &BlockwiseInnerResult,
    options: &BlockwiseFitOptions,
    rho_dim: usize,
    psi_dim: usize,
) -> CustomFamilyError {
    CustomFamilyError::InnerSolveNotConverged {
        cycles: inner.cycles,
        terminal: inner.terminal_convergence_state.clone(),
        kkt_residual: inner
            .kkt_residual
            .as_ref()
            .map(ProjectedKktResidual::inf_norm),
        kkt_tol: inner
            .kkt_residual
            .as_ref()
            .and_then(ProjectedKktResidual::residual_tol),
        theta_dim: rho_dim + psi_dim,
        rho_dim,
        psi_dim,
        // The budget the solve actually ran against: the configured cap after
        // any seed-screening cap, exactly as the inner loop derives it.
        cycle_budget: Some(capped_inner_max_cycles(options, options.inner_max_cycles)),
        // Recorded by the exact joint route from its KKT refusal report; the
        // terminal `kkt_residual` is `None` off a converged iterate by design.
        carrying_block: inner.terminal_carrying_block.clone(),
    }
}

pub(crate) fn constrained_warm_start_from_inner(
    rho: &Array1<f64>,
    inner: &BlockwiseInnerResult,
) -> ConstrainedWarmStart {
    ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
        cached_inner: Some(cached_inner_mode_from_result(inner)),
    }
}

pub(crate) fn constrained_warm_start_from_cached_beta(
    rho_dim: usize,
    specs: &[ParameterBlockSpec],
    beta: &Array1<f64>,
) -> Result<ConstrainedWarmStart, EstimationError> {
    let expected = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if beta.len() != expected {
        crate::bail_invalid_estim!(
            "cached inner beta has length {}, but custom-family blocks require length {}",
            beta.len(),
            expected
        );
    }
    gam_problem::bail_if_cached_beta_non_finite(beta)?;

    let mut offset = 0usize;
    let mut block_beta = Vec::with_capacity(specs.len());
    for spec in specs {
        let end = offset + spec.design.ncols();
        block_beta.push(beta.slice(s![offset..end]).to_owned());
        offset = end;
    }

    Ok(ConstrainedWarmStart {
        rho: Array1::zeros(rho_dim),
        block_beta,
        active_sets: vec![None; specs.len()],
        cached_inner: None,
    })
}

pub(crate) fn inner_penalized_objective(
    inner: &BlockwiseInnerResult,
    include_logdet_h: bool,
    include_logdet_s: bool,
    context: &str,
) -> Result<f64, CustomFamilyError> {
    let reml_term = if include_logdet_h {
        0.5 * inner
            .block_logdet_h
            .ok_or_else(|| CustomFamilyError::trial_point(format!("{context}: certified Hessian logdet is unavailable")))?
    } else {
        0.0
    } - if include_logdet_s {
        0.5 * inner
            .block_logdet_s
            .ok_or_else(|| CustomFamilyError::trial_point(format!("{context}: certified penalty logdet is unavailable")))?
    } else {
        0.0
    };
    checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        reml_term,
        context,
    )
}

pub(crate) fn nonconverged_outer_efs_result(
    inner: &BlockwiseInnerResult,
    rho: &Array1<f64>,
    theta_dim: usize,
    context: &str,
) -> Result<(gam_problem::EfsEval, ConstrainedWarmStart, bool), CustomFamilyError> {
    Ok((
        gam_problem::EfsEval {
            // A non-converged coefficient iterate is not a Laplace mode, so no
            // determinant exists. This finite scalar is diagnostic only; the
            // returned `false` makes the outer optimizer reject the sample.
            cost: checked_penalizedobjective(
                inner.log_likelihood,
                inner.penalty_value,
                0.0,
                context,
            )?,
            steps: vec![0.0; theta_dim],
            beta: None,
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale: None,
            consecutive_restored_incumbents: None,
        },
        constrained_warm_start_from_inner(rho, inner),
        false,
    ))
}

pub(crate) fn warm_start_without_cached_inner_for_psi_derivatives(
    warm_start: Option<&ConstrainedWarmStart>,
    has_psi_derivatives: bool,
) -> Option<ConstrainedWarmStart> {
    if !has_psi_derivatives {
        return None;
    }
    warm_start.cloned().map(|mut warm| {
        warm.cached_inner = None;
        warm
    })
}

/// Helper struct mirroring the old `BlockwiseFitResultParts`.
pub struct BlockwiseFitResultParts {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    /// The classical family deviance at the mode when the family declares one
    /// (`CustomFamily::classical_deviance`); `None` publishes
    /// `−2·log_likelihood`, the convention for a family whose saturated
    /// log-likelihood is zero or undefined (#2786).
    pub deviance: Option<f64>,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub covariance_conditional: Option<Array2<f64>>,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    /// `None` = no gradient measured at termination (cache-hit, gradient-free,
    /// or trivial early-exit); `Some(g)` = measured norm. `outer_converged`
    /// is the authoritative convergence signal.
    pub outer_gradient_norm: Option<f64>,
    /// First-order optimality certificate from the outer smoothing solve
    /// (#934). `None` is valid only when no outer iteration ran (for example,
    /// a fixed-λ fit); an outer run that cannot produce a certificate is
    /// non-converged and is rejected before result assembly.
    pub criterion_certificate: Option<gam_solve::rho_optimizer::OuterCriterionCertificate>,
    pub inner_cycles: usize,
    pub outer_converged: bool,
    pub geometry: Option<FitGeometry>,
    /// Effective degrees of freedom computed by the caller in the *reduced*
    /// (canonical) coefficient space, where the penalized Hessian is full rank,
    /// as `(edf_total, edf_by_penalty, block_edf)`. The trace edf is invariant
    /// under the canonical reparameterization, so computing it in the reduced
    /// space and reporting it on the raw fit is exact — and it avoids the
    /// `tr((H_raw + εI)⁻¹ S_raw)` blow-up that a rank-deficient raw-lifted
    /// Hessian (zero rows/cols on canonicalization-dropped directions) would
    /// otherwise inject. `None` when the caller has no reduced geometry, in
    /// which case `blockwise_fit_from_parts` falls back to computing edf from
    /// whatever geometry it was handed.
    /// Tuple layout: `(edf_total, edf_by_penalty, block_edf, penalty_trace)`,
    /// where `penalty_trace[k] = λ_k·tr(H⁻¹S_k)` feeds the per-term EDF
    /// decomposition `|coeff_range| − Σ tr_k` (issue #1219).
    pub precomputed_edf:
        Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<gam_solve::estimate::EdfRankBound>)>,
    /// Selected per-component log-smoothing parameters of the full-width JOINT
    /// penalty at ρ* (gam#1587/#561). Surfaced on `FitArtifacts.joint_log_lambdas`
    /// so a joint-penalized family (the multinomial centered metric) can recover
    /// its converged smoothing — the per-block `lambdas` are empty for it. `None`
    /// for every per-block-only family.
    pub joint_log_lambdas: Option<Array1<f64>>,
    /// First-order ρ-uncertainty smoothing correction `C` (raw/lifted frame)
    /// with its typed provenance (#2346): `V_c = V_cond + C` is published as
    /// `beta_covariance_corrected`; `None` = typed absence (no outer ρ
    /// curvature retained, or the interior V_ρ is not honestly finite).
    pub smoothing_corrected: Option<(
        Array2<f64>,
        gam_solve::model_types::SmoothingCorrectionMethod,
    )>,
    /// Why no correction was minted on a fit that selected ρ (#2677).
    pub smoothing_correction_absence: Option<gam_solve::model_types::SmoothingCorrectionAbsence>,
}

pub(crate) fn validate_parameter_block_state_finiteness(
    label: &str,
    state: &ParameterBlockState,
) -> Result<(), CustomFamilyError> {
    validate_all_finite_estimation(&format!("{label}.beta"), state.beta.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(&format!("{label}.eta"), state.eta.iter().copied())
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn validate_lambda_pair_consistency(
    log_lambdas: &Array1<f64>,
    lambdas: &Array1<f64>,
    label: &str,
) -> Result<(), CustomFamilyError> {
    if log_lambdas.len() != lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label} length mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            ),
        });
    }
    for (idx, (&log_lambda, &lambda)) in log_lambdas.iter().zip(lambdas.iter()).enumerate() {
        let expected = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| CustomFamilyError::DimensionMismatch { reason: format!("{label} log coordinate {idx}: {error}") })?;
        if lambda.to_bits() != expected.to_bits() {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "{label}[{idx}] inconsistent with exp(log_lambda): got {lambda}, expected {expected}",
            ) });
        }
    }
    Ok(())
}

/// Effective degrees of freedom for a converged blockwise custom-family fit,
/// computed from the joint penalized Hessian `H = X'W_HX + S(λ)` and the
/// per-penalty matrices `S_k` exactly as the standard GAM path and mgcv do:
///
/// ```text
/// edf_total   = p − Σ_k λ_k · tr(H⁻¹ S_k)
/// edf_penalty = (rank_k − λ_k · tr(H⁻¹ S_k))   clamped to [0, rank_k]
/// ```
///
/// `S_k` here is the *unscaled* penalty (its `λ_k` factor is applied here), and
/// each `S_k.to_dense()` is already embedded in the joint `p × p` coefficient
/// layout (the Blockwise / Kronecker variants place their local block at the
/// correct column range), so the trace solve runs in the full joint space and
/// no per-block offset bookkeeping is required.
///
/// The custom-family path (CTN transformation-normal, Dirichlet, …) builds its
/// fit through `blockwise_fit_from_parts` and previously left `inference` at
/// `None`, so `edf_total` was unavailable for every custom family even though
/// the converged geometry already carries the penalized Hessian. This mirrors
/// the survival-path repair (`survival_transformation_edf`, #565) for the
/// blockwise engine: the same trace formula, evaluated against the exact
/// fitted penalized Hessian.
///
/// `edf_penalty` is returned aligned 1:1 with the flattened `lambdas`
/// (one entry per penalty across all blocks), matching the
/// `FitInference::edf_by_block` ↔ `lambdas` length invariant. Per-penalty EDFs
/// are not additive when penalty ranges overlap: each starts from its own
/// `rank(S_k)`. The per-parameter-block aggregate is therefore computed
/// independently as `p_block - Σ_k λ_k tr(H⁻¹S_k)`; an unpenalized block
/// contributes its full column count.
pub(crate) fn custom_family_blockwise_edf(
    penalized_hessian: &Array2<f64>,
    specs: &[ParameterBlockSpec],
    lambdas: &ndarray::ArrayView1<'_, f64>,
) -> Result<
    (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<gam_solve::estimate::EdfRankBound>),
    CustomFamilyError,
> {
    use gam_solve::estimate::reml::reml_outer_engine::penalty_matrix_root;

    let p = penalized_hessian.nrows();
    let total_cols: usize = specs.iter().map(|s| s.design.ncols()).sum();
    if penalized_hessian.ncols() != p || total_cols != p {
        return Err(CustomFamilyError::trial_point(format!(
            "custom-family edf: penalized Hessian {}x{} inconsistent with total block width {}",
            penalized_hessian.nrows(),
            penalized_hessian.ncols(),
            total_cols
        )));
    }
    let expected_rho: usize = specs.iter().map(|s| s.penalties.len()).sum();
    if lambdas.len() != expected_rho {
        return Err(CustomFamilyError::trial_point(format!(
            "custom-family edf: lambdas length {} does not match total penalty count {}",
            lambdas.len(),
            expected_rho
        )));
    }

    let h_sym = SymmetricMatrix::Dense(penalized_hessian.clone());
    // EDF and covariance are properties of the fitted Hessian, not of a
    // nearby matrix selected because it factors. Refuse invalid curvature
    // rather than silently reporting inference for a ridge-perturbed estimand.
    let factor = h_sym.factorize().map_err(|error| {
        format!("custom-family edf: exact penalized-Hessian factorization failed: {error}")
    })?;

    // Per-penalty traces, the rounding band of the solve behind each, each
    // penalty's rank-bound certificate and their block ranks, handed to the shared
    // accounting below (#2470, #2901). A custom family's penalized Hessian is the
    // observed information of an arbitrary likelihood, so `H ⪰ λ_k S_k` is
    // certified per penalty from the inertia of `H − λ_k S_k` shifted by its rounding
    // band. The tilted double well of
    // #2366 traces 2.945 against a rank of 1 at a certified mode and publishes that
    // trace unclamped. This route previously floored `edf_total` at 0, which permits
    // an effective dimension below the joint penalty null space.
    let solve = |values: &mut [f64]| -> Result<(), String> {
        let solved = factor.solve(&ndarray::Array1::from(values.to_vec()))?;
        for (slot, value) in values.iter_mut().zip(solved.iter()) {
            *slot = *value;
        }
        Ok(())
    };
    let inverse_one_norm = gam_linalg::condition::estimate_inverse_one_norm(p, solve, solve)
        .map_err(|error| format!("custom-family edf: inverse-norm estimate failed: {error}"))?;
    let mut raw_traces = vec![0.0_f64; expected_rho];
    let mut trace_bands = vec![0.0_f64; expected_rho];
    let mut rank_bounds = Vec::with_capacity(expected_rho);
    let mut penalty_ranks = vec![0_usize; expected_rho];
    // `Σ_k S_k` in the joint layout, whose rank gives the null-space floor.
    // Unscaled on purpose: the floor is a structural property of the penalty
    // geometry, exactly as the canonical path reads it off the stacked
    // penalty root rather than off `λ`.
    let mut joint_penalty = Array2::<f64>::zeros((p, p));
    let mut block_spans: Vec<(usize, usize, usize)> = Vec::with_capacity(specs.len());
    let mut penalty_offset = 0usize;
    let mut block_col_start = 0usize;
    for spec in specs.iter() {
        let block_cols = spec.design.ncols();
        block_spans.push((penalty_offset, spec.penalties.len(), block_cols));
        for (local_k, penalty) in spec.penalties.iter().enumerate() {
            let global_k = penalty_offset + local_k;
            let lambda = lambdas[global_k];
            // Embed S_k into the full p×p joint layout. `PenaltyMatrix::to_dense`
            // returns the *local* block matrix for the `Dense` variant but the
            // already-embedded full-width matrix for `Blockwise`/`Kronecker`, so
            // dispatch on the materialized dimension: a local (block_cols-wide)
            // penalty is placed at this block's column range, a full-width
            // penalty is used as-is (mirrors `survival_transformation_edf`'s
            // explicit block placement).
            let s_local = penalty.to_dense();
            // Use the same realized penalty root as the REML assembly. Its row
            // count is the exact rank of the penalty coordinate that contributes
            // `rank(S_k)·rho_k` to the criterion. Reconstructing rank from the
            // containing block width overstates every component of a
            // multi-penalty block; consulting `nullspace_dims` here is also
            // incorrect after canonical pullback, which intentionally clears
            // stale pre-transform nullities.
            let root = penalty_matrix_root(&s_local).map_err(|error| {
                format!("custom-family edf: penalty {global_k} rank factorization failed: {error}")
            })?;
            let penalty_rank = root.nrows();
            let mut s_full = Array2::<f64>::zeros((p, p));
            // The root's rows are its modes (`S_k = RᵀR`); they become the columns
            // of the right-hand side in the joint layout.
            let mut root_columns = Array2::<f64>::zeros((p, penalty_rank));
            if s_local.nrows() == p && s_local.ncols() == p {
                s_full.assign(&s_local);
                root_columns.assign(&root.t());
            } else if s_local.nrows() == block_cols && s_local.ncols() == block_cols {
                let r = block_col_start..block_col_start + block_cols;
                s_full.slice_mut(ndarray::s![r.clone(), r.clone()]).assign(&s_local);
                root_columns.slice_mut(ndarray::s![r, ..]).assign(&root.t());
            } else {
                return Err(CustomFamilyError::trial_point(format!(
                    "custom-family edf: penalty {global_k} materialized to {}x{}, expected {p}x{p} or {block_cols}x{block_cols}",
                    s_local.nrows(),
                    s_local.ncols()
                )));
            }
            // λ_k tr(H⁻¹S_k) = λ_k Σ_c r_cᵀ H⁻¹ r_c over the root columns, priced
            // against the Hessian the solve represents.
            if lambda > 0.0 {
                let solution = factor.solvemulti(&root_columns).map_err(|e| {
                    format!("custom-family edf trace solve failed for penalty {global_k}: {e}")
                })?;
                let (trace, band) = gam_linalg::roundoff::solved_penalty_trace(
                    lambda,
                    root_columns.view(),
                    solution.view(),
                    penalized_hessian.view(),
                    inverse_one_norm,
                )
                .map_err(|error| {
                    format!("custom-family edf: penalty {global_k} trace band failed: {error}")
                })?;
                raw_traces[global_k] = trace;
                trace_bands[global_k] = band;
            }
            // `λ_k S_k` on the block it penalizes: a full-width penalty at the origin, a
            // local one at this block's columns.
            let block_start = if s_local.nrows() == p { 0 } else { block_col_start };
            rank_bounds.push(
                gam_solve::estimate::numerical_rank_bound(
                    penalized_hessian.view(),
                    (&s_local * lambda.max(0.0)).view(),
                    block_start,
                    gam_runtime::resource::MemoryGovernor::global(),
                )
                .map_err(|error| CustomFamilyError::NumericalFailure {
                    reason: format!(
                        "custom-family edf: penalty {global_k} rank certificate failed: {error}"
                    ),
                })?,
            );
            joint_penalty += &s_full;
            penalty_ranks[global_k] = penalty_rank;
        }
        penalty_offset += spec.penalties.len();
        block_col_start += block_cols;
    }

    let joint_penalty_rank = penalty_matrix_root(&joint_penalty)
        .map_err(|error| CustomFamilyError::trial_point(format!("custom-family edf: joint penalty rank failed: {error}")))?
        .nrows();
    let bundle = gam_solve::estimate::penalized_edf_bundle_within_bands(
        &raw_traces,
        &trace_bands,
        &rank_bounds,
        &penalty_ranks,
        p,
        (p - joint_penalty_rank.min(p)) as f64,
    )
    .map_err(|error| CustomFamilyError::NumericalFailure {
        reason: format!("custom-family edf: {error}"),
    })?;
    let edf_by_penalty = bundle.edf_by_block;
    let penalty_trace = bundle.penalty_block_trace;
    let rank_bound = bundle.rank_bound;
    // A block's edf is its column count minus the trace its penalties spend, so
    // multiple penalties on one block compose. It is built from the ADMITTED
    // traces above, not the raw products, so the block figure and the per-penalty
    // figures cannot disagree about how much each penalty absorbed. It is clamped
    // to the block's column count only when every penalty on the block is certified
    // (#2901).
    let block_edf: Vec<f64> = block_spans
        .iter()
        .map(|&(start, count, block_cols)| {
            let spent: f64 = penalty_trace[start..start + count].iter().sum();
            let raw = block_cols as f64 - spent;
            if rank_bound[start..start + count]
                .iter()
                .all(gam_solve::estimate::EdfRankBound::is_certified)
            {
                raw.clamp(0.0, block_cols as f64)
            } else {
                raw
            }
        })
        .collect();
    let edf_total = bundle.edf_total;
    if !edf_total.is_finite()
        || edf_by_penalty.iter().any(|v| !v.is_finite())
        || block_edf.iter().any(|v| !v.is_finite())
        || penalty_trace.iter().any(|v| !v.is_finite())
    {
        return Err(CustomFamilyError::trial_point("custom-family edf: non-finite effective degrees of freedom".to_string()));
    }
    Ok((edf_total, edf_by_penalty, block_edf, penalty_trace, rank_bound))
}

/// Compute reduced-space effective degrees of freedom for a converged fit,
/// to be carried through `BlockwiseFitResultParts::precomputed_edf`.
///
/// The reduced (canonical) geometry's penalized Hessian is full rank and its
/// `reduced_specs` carry the pulled-back penalties `T_iᵀ S_k T_i`, so the trace
/// edf is computed exactly here (no rank-deficiency ridge bias). Because the
/// trace edf is invariant under the canonical reparameterization, the resulting
/// `edf_total` / per-penalty / per-block values are the same as they would be
/// in the raw basis and are reported directly on the lifted raw fit. Returns
/// `None` when no reduced geometry is available, so the caller can leave
/// `precomputed_edf` unset (and the raw-geometry fallback applies).
pub(crate) fn reduced_blockwise_edf(
    reduced_geometry: Option<&FitGeometry>,
    canonical: &gam_identifiability::canonical::CanonicalSpecs,
    lambdas: &Array1<f64>,
) -> Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<gam_solve::estimate::EdfRankBound>)> {
    let geom = reduced_geometry?;
    match custom_family_blockwise_edf(
        geom.penalized_hessian.as_array(),
        &canonical.reduced_specs,
        &lambdas.view(),
    ) {
        Ok(triple) => Some(triple),
        Err(err) => {
            log::debug!(
                "[custom-family inference] reduced-space effective degrees of freedom unavailable: {err}"
            );
            None
        }
    }
}

fn require_converged_outer_for_assembly(outer_converged: bool) -> Result<(), CustomFamilyError> {
    if outer_converged {
        return Ok(());
    }
    Err(CustomFamilyError::Optimization {
        context: "blockwise_fit_from_parts",
        reason: "refusing to assemble a fit from a non-converged outer optimization; \
                 the solver must return its checkpoint as nonconvergence evidence instead"
            .to_string(),
    })
}

/// Assemble the first-order corrected covariance `V_c = V_cond + C` (#2346).
///
/// Its diagonal goes through `gam_problem::se_from_covariance`, the gate the
/// published standard errors are derived under (gam#2955), rather than a local
/// `max(0, ·)` clamp. `V_c` is a *sum*, not a factorization, so a large negative
/// correction on a weakly identified coefficient can drive a diagonal
/// materially negative. A clamp publishes that coefficient with `SE = 0`, i.e.
/// infinite precision and a Wald `p ≈ 0`; snapping a negative diagonal to zero
/// is legitimate only inside the dimension-scaled backward-error bound, which is
/// exactly the judgement `se_from_covariance` owns. Refusing here names the
/// custom-family lane in the error, before the fit is minted.
fn corrected_covariance(
    smoothing_corrected: Option<&(
        Array2<f64>,
        gam_solve::model_types::SmoothingCorrectionMethod,
    )>,
    covariance_conditional: Option<&Array2<f64>>,
) -> Result<
    (
        Option<Array2<f64>>,
        Option<gam_solve::model_types::SmoothingCorrectionMethod>,
        Option<Array2<f64>>,
    ),
    CustomFamilyError,
> {
    let (Some((correction, method)), Some(v_cond)) = (smoothing_corrected, covariance_conditional)
    else {
        return Ok((None, None, None));
    };
    if correction.dim() != v_cond.dim() {
        return Ok((None, None, None));
    }
    let corrected = v_cond + correction;
    gam_problem::se_from_covariance(&corrected).map_err(|reason| {
        CustomFamilyError::NumericalFailure {
            reason: format!(
                "corrected covariance V_c = V_cond + C has an invalid diagonal: {reason}"
            ),
        }
    })?;
    Ok((Some(correction.clone()), Some(*method), Some(corrected)))
}

#[cfg(test)]
mod corrected_covariance_tests {
    use super::*;
    use gam_solve::model_types::SmoothingCorrectionMethod;

    fn first_order_method() -> SmoothingCorrectionMethod {
        SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
            active_rank: 1,
            rho_dimension: 1,
        }
    }

    #[test]
    fn corrected_standard_errors_are_the_covariance_diagonal_roots() {
        let v_cond = Array2::from_diag(&Array1::from_vec(vec![4.0, 9.0]));
        let correction = Array2::from_diag(&Array1::from_vec(vec![5.0, 7.0]));
        let (_, _, corrected) = corrected_covariance(
            Some(&(correction, first_order_method())),
            Some(&v_cond),
        )
        .expect("a positive-definite corrected covariance must be accepted");
        let corrected = corrected.expect("corrected covariance is published");
        assert_eq!(corrected[[0, 0]], 9.0);
        assert_eq!(corrected[[1, 1]], 16.0);
        // The published standard errors are derived from this one matrix (gam#2955).
        let se = gam_problem::se_from_covariance(&corrected)
            .expect("the corrected diagonal yields standard errors");
        assert_eq!(se[0], 3.0);
        assert_eq!(se[1], 4.0);
    }

    #[test]
    fn a_materially_negative_corrected_diagonal_is_refused_not_clamped_to_zero() {
        // `V_c = V_cond + C` with a correction that overwhelms the conditional
        // variance of coefficient 1. The clamp this seam replaced published
        // `SE = 0` here — an infinitely precise coefficient whose Wald p-value
        // is 0 — so the guard is that assembly now fails instead.
        let v_cond = Array2::from_diag(&Array1::from_vec(vec![4.0, 1.0]));
        let correction = Array2::from_diag(&Array1::from_vec(vec![0.0, -3.0]));
        let error = corrected_covariance(
            Some(&(correction, first_order_method())),
            Some(&v_cond),
        )
        .expect_err("a materially negative corrected diagonal must be refused");
        assert!(matches!(
            error,
            CustomFamilyError::NumericalFailure { reason }
                if reason.contains("V_c = V_cond + C has an invalid diagonal")
        ));
    }

    #[test]
    fn a_dimension_mismatched_correction_publishes_no_corrected_pair() {
        let v_cond = Array2::from_diag(&Array1::from_vec(vec![4.0, 9.0]));
        let correction = Array2::from_diag(&Array1::from_vec(vec![1.0]));
        let (correction_out, method, corrected) = corrected_covariance(
            Some(&(correction, first_order_method())),
            Some(&v_cond),
        )
        .expect("a mismatched correction is a typed absence, not a failure");
        assert!(correction_out.is_none());
        assert!(method.is_none());
        assert!(corrected.is_none());
    }
}

#[cfg(test)]
mod assembly_convergence_tests {
    use super::*;

    fn parts_with_outer_evidence(
        outer_iterations: usize,
        outer_converged: bool,
        criterion_certificate: Option<gam_solve::rho_optimizer::OuterCriterionCertificate>,
    ) -> BlockwiseFitResultParts {
        BlockwiseFitResultParts {
            block_states: Vec::new(),
            log_likelihood: 0.0,
            deviance: None,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            covariance_conditional: None,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations,
            outer_gradient_norm: None,
            criterion_certificate,
            inner_cycles: 0,
            outer_converged,
            geometry: None,
            precomputed_edf: None,
            joint_log_lambdas: None,
            smoothing_corrected: None,
            smoothing_correction_absence: None,
        }
    }

    #[test]
    fn nonconverged_outer_state_cannot_reach_fit_assembly() {
        let error = blockwise_fit_from_parts(parts_with_outer_evidence(1, false, None), &[])
            .expect_err("nonconverged outer state must be rejected");
        assert!(matches!(
            error,
            CustomFamilyError::Optimization { context, reason }
                if context == "blockwise_fit_from_parts"
                    && reason.contains("non-converged outer optimization")
        ));
    }

    #[test]
    fn outer_run_without_certificate_cannot_reach_fit_assembly() {
        let error = blockwise_fit_from_parts(parts_with_outer_evidence(1, true, None), &[])
            .expect_err("an outer run without a certificate must be rejected");
        assert!(matches!(
            error,
            CustomFamilyError::Optimization { context, reason }
                if context == "blockwise_fit_from_parts"
                    && reason.contains("without its analytic convergence certificate")
        ));
    }

    #[test]
    fn failed_outer_certificate_cannot_reach_fit_assembly() {
        let certificate = gam_solve::rho_optimizer::OuterCriterionCertificate {
            stationarity:
                gam_solve::rho_optimizer::OuterStationarityCertificate::AnalyticGradient {
                    grad_norm: 1.0,
                    projected_grad_norm: 1.0,
                    bound: 0.1,
                    rung: gam_solve::rho_optimizer::CertifiedRung {
                        label: "solver-band".to_string(),
                        derived_standard: false,
                    },
                },
            curvature: gam_solve::rho_optimizer::CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            newton_polish: None,
            curvature_floor: None,
        };
        let error =
            blockwise_fit_from_parts(parts_with_outer_evidence(1, true, Some(certificate)), &[])
                .expect_err("a failed analytic certificate must be rejected");
        assert!(matches!(
            error,
            CustomFamilyError::Optimization { context, reason }
                if context == "blockwise_fit_from_parts"
                    && reason.contains("analytic outer certificate failed")
        ));
    }
}

/// Build a `UnifiedFitResult` from blockwise-specific fields.
pub fn blockwise_fit_from_parts(
    parts: BlockwiseFitResultParts,
    specs: &[ParameterBlockSpec],
) -> Result<gam_solve::model_types::UnifiedFitResult, CustomFamilyError> {
    let BlockwiseFitResultParts {
        block_states,
        log_likelihood,
        deviance,
        log_lambdas,
        lambdas,
        covariance_conditional,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_gradient_norm,
        criterion_certificate,
        inner_cycles,
        outer_converged,
        geometry,
        precomputed_edf,
        joint_log_lambdas,
        smoothing_corrected,
        smoothing_correction_absence,
    } = parts;

    // SPEC 20: a fit object only ever comes from a converged optimization.
    // Assembling a `UnifiedFitResult` from a non-converged outer state would
    // mint a degraded fit (previously surfaced as `StalledAtValidMinimum`);
    // non-convergence must instead be raised as a typed error at the solver,
    // with a checkpoint, before ever reaching this assembler. This defensive
    // gate prevents direct assembly callers from bypassing that contract.
    require_converged_outer_for_assembly(outer_converged)?;
    match criterion_certificate.as_ref() {
        Some(certificate) => {
            if !certificate.certifies() {
                return Err(CustomFamilyError::Optimization {
                    context: "blockwise_fit_from_parts",
                    reason: format!(
                        "refusing to assemble a fit whose analytic outer certificate failed: {}; \
                         no fit was assembled",
                        certificate.summary()
                    ),
                });
            }
        }
        None => {
            if outer_iterations > 0 {
                return Err(CustomFamilyError::Optimization {
                    context: "blockwise_fit_from_parts",
                    reason: "refusing to assemble a fit after an outer optimization without its \
                             analytic convergence certificate"
                        .to_string(),
                });
            }
        }
    }
    if block_states.is_empty() {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "blockwise fit requires at least one block state".to_string(),
        });
    }
    ensure_finite_scalar_estimation("blockwise_fit.log_likelihood", log_likelihood)
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation("blockwise_fit.log_lambdas", log_lambdas.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation("blockwise_fit.lambdas", lambdas.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_lambda_pair_consistency(&log_lambdas, &lambdas, "blockwise_fit.lambdas")?;
    ensure_finite_scalar_estimation("blockwise_fit.penalized_objective", penalized_objective)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("blockwise_fit.stable_penalty_term", stable_penalty_term)
        .map_err(|e| e.to_string())?;
    if let Some(g) = outer_gradient_norm {
        ensure_finite_scalar_estimation("blockwise_fit.outer_gradient_norm", g)
            .map_err(|e| e.to_string())?;
    }

    if block_states.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "blockwise_fit.block_states length ({}) does not match specs length ({})",
                block_states.len(),
                specs.len()
            ),
        });
    }
    // `design`, unlike `solver_design()`, has one row per original
    // experimental unit. Survival and multi-output blocks may expand their
    // solver design, but that expansion is not a larger training sample.
    let n = specs[0].design.nrows();
    if n == 0 {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "blockwise_fit requires at least one original training row".to_string(),
        });
    }
    for (idx, spec) in specs.iter().enumerate().skip(1) {
        if spec.design.nrows() != n {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit spec {idx} has {} original training rows, expected {n}",
                    spec.design.nrows()
                ),
            });
        }
    }
    let total_p = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    for (idx, state) in block_states.iter().enumerate() {
        validate_parameter_block_state_finiteness(
            &format!("blockwise_fit.block_states[{idx}]"),
            state,
        )?;
        let expected_rows = specs[idx].solver_design().nrows();
        if state.eta.len() != expected_rows {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.block_states[{idx}] eta length mismatch: got {}, expected {} (solver design rows)",
                state.eta.len(),
                expected_rows
            ) });
        }
    }

    if let Some(cov) = covariance_conditional.as_ref() {
        validate_all_finite_estimation("blockwise_fit.covariance_conditional", cov.iter().copied())
            .map_err(|e| e.to_string())?;
        let (rows, cols) = cov.dim();
        if rows != total_p || cols != total_p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit.covariance_conditional must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            });
        }
    }

    let geom = geometry.as_ref().ok_or_else(|| CustomFamilyError::InvalidInput {
        context: "blockwise_fit_from_parts",
        reason: "a converged custom-family fit must retain coefficient gauge and penalized Hessian geometry"
            .to_string(),
    })?;
    {
        geom.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        let mut raw_block_starts = Vec::with_capacity(block_states.len() + 1);
        raw_block_starts.push(0usize);
        for state in &block_states {
            raw_block_starts.push(
                raw_block_starts
                    .last()
                    .copied()
                    .expect("raw block starts contain the initial zero")
                    .checked_add(state.beta.len())
                    .ok_or_else(|| CustomFamilyError::DimensionMismatch {
                        reason: "blockwise_fit raw coefficient partition overflows usize"
                            .to_string(),
                    })?,
            );
        }
        if geom.coefficient_gauge.block_starts_raw != raw_block_starts {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit.geometry raw gauge partition {:?} does not match fitted block partition {:?}",
                    geom.coefficient_gauge.block_starts_raw, raw_block_starts
                ),
            });
        }
        let active_dimension = geom.coefficient_gauge.reduced_total();
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != active_dimension || cols != active_dimension {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit.geometry active penalized_hessian must be {active_dimension}x{active_dimension}, got {rows}x{cols}"
                ),
            });
        }
        if !geom.coefficient_gauge.is_identity() && precomputed_edf.is_none() {
            return Err(CustomFamilyError::InvalidInput {
                context: "blockwise_fit_from_parts",
                reason: "non-identity active geometry requires its reduced-coordinate EDF; raw penalties cannot be paired with an active-coordinate Hessian"
                    .to_string(),
            });
        }
        if let Some(working) = geom.working.as_ref()
            && working.weights.len() != n
        {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.geometry single-diagonal working row count mismatch: got {}, expected {n}",
                working.weights.len(),
            ) });
        }
    }

    // Build unified blocks from the blockwise states.
    use gam_solve::model_types::{FittedBlock, FittedLinkState, UnifiedFitResultParts};
    let expected_rho: usize = specs.iter().map(|s| s.penalties.len()).sum();
    if lambdas.len() != expected_rho {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "blockwise_fit.lambdas length ({}) does not match sum of per-block penalty counts ({})",
            lambdas.len(),
            expected_rho
        ) });
    }
    // Effective degrees of freedom and the inference block. The converged
    // coefficient geometry always carries the joint penalized Hessian; compute the
    // mgcv trace edf `p − Σ_k λ_k·tr(H⁻¹ S_k)` here so every custom-family fit
    // (CTN transformation-normal, Dirichlet, …) reports `edf_total` /
    // per-block `edf` like the standard GAM path, instead of leaving inference
    // unpopulated. Optional row evidence is not part of this calculation.
    let (edf_total, edf_by_penalty, block_edf, penalty_trace, rank_bound): (
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<gam_solve::estimate::EdfRankBound>,
    ) =
        match precomputed_edf {
            // Reduced-space edf supplied by the caller (the principled path:
            // the trace is computed where the Hessian is full rank, then
            // reported on the raw fit — exact because the trace edf is
            // reparameterization-invariant).
            Some((edf_total, edf_by_penalty, block_edf, penalty_trace, rank_bound)) => {
                (edf_total, edf_by_penalty, block_edf, penalty_trace, rank_bound)
            }
            // Compute from coefficient precision when the caller did not already
            // supply the basis-invariant reduced-space traces.
            None => {
                let (edf_total, edf_by_penalty, block_edf, penalty_trace, rank_bound) =
                    custom_family_blockwise_edf(
                        geom.penalized_hessian.as_array(),
                        specs,
                        &lambdas.view(),
                    )
                    .map_err(|reason| CustomFamilyError::Optimization {
                        context: "blockwise_fit_from_parts coefficient-geometry EDF",
                        reason: format!(
                            "{reason}; refusing to assemble a fit without EDF/inference"
                        ),
                    })?;
                (edf_total, edf_by_penalty, block_edf, penalty_trace, rank_bound)
            }
        };

    let mut lambda_offset = 0usize;
    let blocks: Vec<FittedBlock> = block_states
        .iter()
        .enumerate()
        .map(|(i, bs)| {
            let role = custom_family_block_role(&specs[i].name, i, block_states.len());
            let k = specs[i].penalties.len();
            let block_lambdas = lambdas
                .slice(s![lambda_offset..lambda_offset + k])
                .to_owned();
            lambda_offset += k;
            FittedBlock {
                beta: bs.beta.clone(),
                role,
                edf: block_edf.get(i).copied().unwrap_or(0.0),
                lambdas: block_lambdas,
            }
        })
        .collect();
    // A family that declares a classical deviance publishes it — the unscaled
    // `2·Σ w·d(y, μ̂)` every standard fit reports. The others publish
    // `−2·log_likelihood`, which is that deviance exactly when the saturated
    // log-likelihood is zero (categorical responses) and the family's own
    // convention when no finite saturated point exists (#2786).
    let deviance = deviance.unwrap_or(-2.0 * log_likelihood);

    // Assemble the inference block from the converged geometry. CTN and other
    // custom families estimate their own likelihood scale, so the penalized
    // Hessian is reported unscaled (dispersion = 1) — the EDF trace is
    // dispersion-free, and downstream covariance scaling pairs `H` with the
    // family's own dispersion where needed.
    // #2346: publish the first-order corrected covariance when the outer ρ
    // curvature supplied one — `V_c = V_cond + C`, with the correction matrix
    // and its typed method provenance carried exactly like the standard lane.
    let (smoothing_correction, smoothing_correction_method, corrected_cov) =
        corrected_covariance(
            smoothing_corrected.as_ref(),
            covariance_conditional.as_ref(),
        )?;
    // The published standard errors derive from the top-level covariance, the
    // one store (gam#2955), so `display_coefficient_uncertainty()` (#2296) sees
    // this lane's pairs whenever the matrices are published. The conditional
    // diagonal goes through the same gate the corrected pair goes through above,
    // so a refusal names this lane before the fit is minted (gam-2929).
    covariance_conditional
        .as_ref()
        .map(gam_problem::se_from_covariance)
        .transpose()
        .map_err(|reason| CustomFamilyError::NumericalFailure {
            reason: format!(
                "conditional covariance V_cond has an invalid diagonal: {reason}"
            ),
        })?;
    let inference = Some(gam_solve::model_types::FitInference {
        edf_by_block: edf_by_penalty,
        penalty_block_trace: penalty_trace,
        edf_rank_bound: rank_bound,
        edf_total,
        // This custom-family lane only ever computes the first-order IFT
        // correction (never a cubature upgrade — see `smoothing_corrected`'s
        // doc comment above), so its retained "first-order" pair is exactly
        // its primary pair; mirror it so the #946 exact corrected-EDF/AIC
        // channel keeps reading a populated value from this lane too.
        smoothing_correction_first_order: smoothing_correction.clone(),
        smoothing_correction_method_first_order: smoothing_correction_method,
        smoothing_correction,
        smoothing_correction_method,
        smoothing_correction_absence,
        smoothing_correction_fallback: None,
        penalized_hessian: geom.penalized_hessian.clone(),
        reparam_qs: None,
        dispersion: gam_solve::model_types::Dispersion::UNIT,
        factorized_standard_errors: None,
        beta_covariance_frequentist: None,
        coefficient_influence: None,
        weighted_gram: None,
        identified_subspace: None,
    });

    gam_solve::model_types::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        training_sample_size: n,
        log_lambdas: log_lambdas.clone(),
        lambdas: lambdas.clone(),
        likelihood_family: None,
        likelihood_scale: gam_problem::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: gam_problem::LogLikelihoodNormalization::UserProvided,
        log_likelihood,
        deviance,
        reml_score: Some(penalized_objective),
        stable_penalty_term,
        penalized_objective: Some(penalized_objective),
        used_device: false,
        outer_iterations,
        outer_converged,
        outer_gradient_norm,
        standard_deviation: 1.0,
        covariance_conditional,
        // The result validation requires the inference-level corrected matrix to
        // be mirrored at the top level (bitwise-equal), exactly as the standard
        // lane publishes it.
        covariance_corrected: corrected_cov,
        inference,
        fitted_link: FittedLinkState::Standard(None),
        geometry,
        block_states,
        // `outer_converged == true` is guaranteed by the SPEC-20 gate at the
        // top of this assembler, so the assembled status is always honest.
        pirls_status: gam_solve::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: gam_solve::model_types::FitArtifacts {
            pirls: None,
            criterion_certificate,
            joint_log_lambdas,
            ..Default::default()
        },
        inner_cycles,
    })
    .map_err(|error| CustomFamilyError::Optimization {
        context: "blockwise_fit_from_parts result validation",
        reason: error.to_string(),
    })
}

pub(crate) fn checked_penalizedobjective(
    log_likelihood: f64,
    penalty_value: f64,
    reml_term: f64,
    context: &str,
) -> Result<f64, CustomFamilyError> {
    let objective = -log_likelihood + penalty_value + reml_term;
    if objective.is_finite() {
        Ok(objective)
    } else {
        Err(CustomFamilyError::NumericalFailure {
            reason: format!(
                "{context}: non-finite penalized objective \
             (log_likelihood={log_likelihood}, penalty_value={penalty_value}, \
             reml_term={reml_term}, objective={objective})"
            ),
        })
    }
}

#[derive(Clone)]
pub struct CustomFamilyWarmStart {
    pub(crate) inner: ConstrainedWarmStart,
}

impl CustomFamilyWarmStart {
    pub fn compatible_with_rho(&self, rho: &Array1<f64>) -> bool {
        screened_outer_warm_start(Some(&self.inner), rho).is_some()
    }

    /// Borrow the converged per-block coefficient vector for `block_idx`.
    /// Callers that need to evaluate the block's fitted linear predictor
    /// `X·β` (rather than inspect raw coefficient magnitudes) read β through
    /// this view.
    pub fn block_beta_view(&self, block_idx: usize) -> Option<ArrayView1<'_, f64>> {
        self.inner.block_beta.get(block_idx).map(|beta| beta.view())
    }

    /// Build a warm-start payload from a flat cached β and the per-block
    /// coefficient widths. The returned warm-start carries a zero `rho`
    /// (the outer cache will overwrite it on the next eval) and empty
    /// active sets; only the per-block β slices feed the next inner
    /// PIRLS / Newton solve. Used by the spatial-joint outer cache to
    /// seed the family-owned warm-start slot on cache hits so the inner
    /// solve opens at the prior converged iterate instead of cold β.
    pub fn from_cached_beta(
        block_col_counts: &[usize],
        beta: &Array1<f64>,
    ) -> Result<Self, EstimationError> {
        let expected: usize = block_col_counts.iter().copied().sum();
        if beta.len() != expected {
            crate::bail_invalid_estim!(
                "cached inner beta has length {}, but spatial-joint blocks require length {}",
                beta.len(),
                expected
            );
        }
        gam_problem::bail_if_cached_beta_non_finite(beta)?;
        let mut offset = 0usize;
        let mut block_beta = Vec::with_capacity(block_col_counts.len());
        for &width in block_col_counts {
            let end = offset + width;
            block_beta.push(beta.slice(s![offset..end]).to_owned());
            offset = end;
        }
        Ok(CustomFamilyWarmStart {
            inner: ConstrainedWarmStart {
                rho: Array1::zeros(0),
                block_beta,
                active_sets: vec![None; block_col_counts.len()],
                cached_inner: None,
            },
        })
    }
}

pub(crate) struct CustomOuterState {
    /// Inner mode of the incumbent outer iterate, the seed of every search
    /// evaluation. Only an accepted iterate replaces it (#2668). When every trial
    /// wrote its own mode here, the next trial started from wherever the last one
    /// converged. The objective then depended on search history, and two
    /// converged inner modes at one ρ alternated under the line search without
    /// end: on row 30 of #2668, probes 1e-4 apart priced 348.108 and 351.557.
    pub(crate) warm_cache: Option<ConstrainedWarmStart>,
    pub(crate) reset_warm_cache: Option<ConstrainedWarmStart>,
    /// Exact derivative-bearing coefficient mode installed by the most recent
    /// analytic outer evaluation.
    ///
    /// This is deliberately an ownership slot rather than a warm-start cache:
    /// `CustomFamilyOwnedMode` is non-`Clone`, so fit assembly can consume the
    /// one inner result that produced the certified objective and derivatives
    /// without re-entering the (potentially nonconvex) coefficient solver.
    pub(crate) terminal_mode: Option<CustomFamilyTerminalMode>,
    /// Typed refusal from the last failed or infeasible objective evaluation.
    ///
    /// This is diagnostic evidence only; it never authorizes a fit. Keeping the
    /// source whole lets terminal reporting render it once and lets tests or
    /// downstream classifiers inspect exact inner convergence fields without
    /// parsing text.
    pub(crate) last_error: Option<CustomFamilyError>,
    /// The most recent uncertified inner solve any evaluation of this search
    /// raised, recorded by [`Self::record_refusal`].
    ///
    /// Unlike [`Self::last_error`], no later evaluation, reset or reseed clears
    /// it. When the search ends uncertified it becomes
    /// `OuterSmoothingFailed::search_inner_refusal`, which the fit boundary names
    /// even when finite trials ran after it (#2943).
    pub(crate) last_inner_refusal: Option<CustomFamilyError>,
    pub(crate) outer_derivative_pilot: Option<OuterDerivativePilotSchedule>,
    /// #2349 — one-shot "re-evaluate COLD" pulse shared with the outer
    /// cost-stall guard (via `OuterProblem::with_stuck_stall_cold_reeval_signal`).
    /// The guard raises it when it grants a STUCK-stall escape; the outer-eval
    /// closures observe it, drop the warm cache for that evaluation, and latch
    /// [`Self::force_cold_latched`] so the remainder of the outer run stays on
    /// the trajectory-independent COLD surface.
    pub(crate) force_cold_signal: Arc<AtomicBool>,
    /// Latched once the cold-reeval pulse has fired: every subsequent outer
    /// evaluation re-solves the inner problem cold, so the profiled objective
    /// ARC sees is a consistent function of ρ (no warm-start hysteresis) and the
    /// optimizer can descend past the near-separating stall. Survives
    /// [`Self::reset`] on purpose — the terminal certificate must be measured on
    /// the same cold surface the descent used (see `reset`).
    pub(crate) force_cold_latched: bool,
    /// Accepted outer steps, advanced by the optimizer's accept observer through
    /// the channel `OuterProblem::with_stuck_stall_cold_reeval_signal` wires.
    pub(crate) accepted_step_signal: Arc<AtomicUsize>,
    /// How many of those steps [`Self::adopt_accepted_steps`] has folded in.
    pub(crate) accepted_steps_adopted: usize,
    /// Whether `warm_cache` already holds an evaluated incumbent's mode rather
    /// than the caller's seed.
    pub(crate) incumbent_established: bool,
    /// Mode of the latest converged first-order evaluation since the last
    /// accepted step, held until the optimizer reports that step accepted.
    pub(crate) pending_first_order_mode: Option<ConstrainedWarmStart>,
    /// Certified mode of the current walk's latest accepted iterate: the walk's
    /// starting iterate, then each iterate the optimizer accepts (#2627).
    pub(crate) walk_iterate: Option<ConstrainedWarmStart>,
    /// Certified modes of the walks a reset has ended, keyed by the bits of the θ
    /// each walk last accepted (#2627).
    ///
    /// An evaluation at a θ a walk accepted is solved from that iterate's own
    /// certified mode, the rule `ExactCoefficientModeBranch` applies to the
    /// exact-joint drivers (gam#2765). The terminal certification resets before
    /// each of its installations at the winner's θ, and a reset leaves the
    /// caller's seed. Without this, every one of those installations re-solved
    /// from that seed the mode the walk had already certified at θ: six identical
    /// cold 26-cycle solves, about 61 s, on the event-history prior-centred fit
    /// (job 1212656). Solved from the certified mode, each is a same-ρ reuse, and
    /// finalize and certify still start from one state (#2334). A filed mode is
    /// a start, not a value: the inner solve reuses it only when its own same-ρ
    /// check accepts it under the evaluation's current contract, and otherwise
    /// seeds β from it. One mode per walk, not per iterate: inside a walk the
    /// incumbent's own θ is served by `warm_cache`.
    pub(crate) walk_endpoints: Vec<(Vec<u64>, ConstrainedWarmStart)>,
    /// The converged mode of the latest value probe, filed under the bits of its θ and
    /// the identity of the seed it was solved from (#979).
    ///
    /// A line search prices a trial θ by value, then asks for the gradient at the same θ,
    /// and both lanes start from the same seed. The second lane's inner solve therefore
    /// re-derives this mode: on the n=2000 BMS flex fit, 12 value/gradient pairs at 12 θ,
    /// every pair with matching cycle counts (job 1244570). The mode is served only at
    /// bitwise that θ, and only while [`Self::seed_for`] still returns a seed of the same
    /// identity. So it seeds no other θ (#2668), and a seed that moved never inherits it.
    /// Like a walk endpoint it is a start, not a value: the inner solve reuses it only
    /// when its own same-ρ check accepts it.
    pub(crate) value_probe: Option<ValueProbeMode>,
    /// Kept rank of the criterion the most recent successful evaluation priced (#2765),
    /// published to the outer search through `OuterObjective::criterion_rank`.
    pub(crate) last_criterion_rank: Option<usize>,
}

fn theta_bits(theta: &Array1<f64>) -> Vec<u64> {
    theta.iter().map(|value| value.to_bits()).collect()
}

/// What an inner solve's result depends on in its seed: the bits of the seed's θ, block
/// coefficients and active sets, and whose objective a carried cached mode was solved
/// for. Two solves at one θ from seeds of one identity are one deterministic computation.
#[derive(PartialEq)]
pub(crate) struct SeedIdentity {
    theta: Vec<u64>,
    block_beta: Vec<Vec<u64>>,
    active_sets: Vec<Option<Vec<usize>>>,
    cached_objective: Option<crate::assembly::InnerObjectiveState>,
}

impl SeedIdentity {
    pub(crate) fn of(seed: Option<&ConstrainedWarmStart>) -> Option<Self> {
        seed.map(|seed| Self {
            theta: theta_bits(&seed.rho),
            block_beta: seed.block_beta.iter().map(theta_bits).collect(),
            active_sets: seed.active_sets.clone(),
            cached_objective: seed
                .cached_inner
                .as_ref()
                .map(|cached| cached.objective_state.clone()),
        })
    }
}

/// A converged value probe's mode with the θ it priced and the seed it was solved from.
pub(crate) struct ValueProbeMode {
    theta: Vec<u64>,
    seed: Option<SeedIdentity>,
    mode: ConstrainedWarmStart,
}

impl CustomOuterState {
    pub(crate) fn new_with_cold_signal(
        warm_start: Option<ConstrainedWarmStart>,
        force_cold_signal: Arc<AtomicBool>,
        accepted_step_signal: Arc<AtomicUsize>,
    ) -> Self {
        let accepted_steps_adopted = accepted_step_signal.load(Ordering::Relaxed);
        Self {
            warm_cache: warm_start.clone(),
            reset_warm_cache: warm_start,
            terminal_mode: None,
            last_error: None,
            last_inner_refusal: None,
            outer_derivative_pilot: None,
            force_cold_signal,
            force_cold_latched: false,
            accepted_step_signal,
            accepted_steps_adopted,
            incumbent_established: false,
            pending_first_order_mode: None,
            walk_iterate: None,
            walk_endpoints: Vec::new(),
            value_probe: None,
            last_criterion_rank: None,
        }
    }

    /// The seed of one outer evaluation at `theta`: the certified mode a walk
    /// accepted at `theta` when there is one, otherwise the incumbent's (#2627,
    /// #2668).
    pub(crate) fn seed_for(&self, theta: &Array1<f64>) -> Option<&ConstrainedWarmStart> {
        let key = theta_bits(theta);
        let endpoint = self
            .walk_endpoints
            .iter()
            .find(|(bits, _)| *bits == key)
            .map(|(_, mode)| mode);
        screened_outer_warm_start(endpoint.or(self.warm_cache.as_ref()), theta)
    }

    /// The start of one outer evaluation at `theta`: the latest value probe's mode when
    /// it priced bitwise this θ from a seed of the identity [`Self::seed_for`] returns now,
    /// otherwise that seed (#979).
    pub(crate) fn warm_start_for(&self, theta: &Array1<f64>) -> Option<&ConstrainedWarmStart> {
        let seed = self.seed_for(theta);
        match &self.value_probe {
            Some(probe) if probe.theta == theta_bits(theta) && probe.seed == SeedIdentity::of(seed) => {
                Some(&probe.mode)
            }
            _ => seed,
        }
    }

    /// File a converged value probe's mode at `theta`, solved from a seed of identity
    /// `seed` (#979). It replaces the previous probe's.
    pub(crate) fn record_value_probe(
        &mut self,
        theta: &Array1<f64>,
        seed: Option<SeedIdentity>,
        mode: ConstrainedWarmStart,
    ) {
        self.value_probe = Some(ValueProbeMode {
            theta: theta_bits(theta),
            seed,
            mode,
        });
    }

    /// Observe the shared cold-reeval pulse (consuming it) and the sticky
    /// latch: returns `true` when this and every following outer evaluation
    /// must re-solve the inner problem COLD (#2349). The pulse is raised by the
    /// outer cost-stall guard on a STUCK-stall escape; once seen it latches for
    /// the remainder of the run so ARC descends a consistent surface.
    pub(crate) fn take_force_cold(&mut self) -> bool {
        if self.force_cold_signal.swap(false, Ordering::Relaxed) {
            self.force_cold_latched = true;
        }
        self.force_cold_latched
    }

    /// Fold in every outer step the optimizer accepted since the previous
    /// evaluation (#2668). Called at the head of each search evaluation and by
    /// [`Self::reset`].
    ///
    /// The accept observer fires once the accepted iterate's first-order
    /// evaluation has run, so the pending mode is that iterate's, and it becomes
    /// the seed. A move the observer never reports leaves the older incumbent
    /// seeding, which still does not depend on any rejected trial.
    pub(crate) fn adopt_accepted_steps(&mut self) {
        let reported = self.accepted_step_signal.load(Ordering::Relaxed);
        if reported == self.accepted_steps_adopted {
            return;
        }
        self.accepted_steps_adopted = reported;
        if let Some(mode) = self.pending_first_order_mode.take() {
            self.accept_iterate(mode);
        }
    }

    fn accept_iterate(&mut self, mode: ConstrainedWarmStart) {
        self.walk_iterate = Some(mode.clone());
        self.warm_cache = Some(mode);
    }

    /// Record the inner mode of a converged first-order evaluation (#2668).
    ///
    /// The first one after a reset is the search's starting iterate and seeds at
    /// once. Every later one waits until [`Self::adopt_accepted_steps`] sees its
    /// step accepted, and a trial the optimizer rejects is replaced by the next
    /// trial's mode without ever seeding. Value probes record nothing.
    pub(crate) fn record_first_order_mode(&mut self, mode: ConstrainedWarmStart) {
        if self.incumbent_established {
            self.pending_first_order_mode = Some(mode);
        } else {
            self.accept_iterate(mode);
            self.incumbent_established = true;
        }
    }

    /// Record the typed refusal of one objective evaluation. An uncertified
    /// inner solve is also kept as the search's whole-search record, which
    /// neither a later evaluation nor a reset clears (#2943).
    pub(crate) fn record_refusal(&mut self, refusal: CustomFamilyError) {
        if matches!(refusal, CustomFamilyError::InnerSolveNotConverged { .. }) {
            self.last_inner_refusal = Some(refusal.clone());
        }
        self.last_error = Some(refusal);
    }

    pub(crate) fn with_outer_derivative_pilot(
        mut self,
        schedule: Option<OuterDerivativePilotSchedule>,
    ) -> Self {
        self.outer_derivative_pilot = schedule;
        self
    }

    pub(crate) fn begin_exact_polish(&mut self) -> bool {
        let transitioned = self
            .outer_derivative_pilot
            .as_ref()
            .is_some_and(OuterDerivativePilotSchedule::enter_exact_phase);
        if transitioned {
            // The sampled pilot's final converged inner mode is the exact
            // stage's warm baseline. `run_outer_uncertified` resets before its
            // seed loop, so promote the live cache into the reset slot first.
            self.reset_warm_cache = self.warm_cache.clone();
            self.terminal_mode = None;
            self.last_error = None;
            // The pilot's certified modes were solved on the sampled measure, so
            // none of them is the exact stage's certified mode at its θ (#2627).
            self.walk_endpoints.clear();
            self.walk_iterate = None;
            self.pending_first_order_mode = None;
        }
        transitioned
    }

    pub(crate) fn reset(&mut self) {
        // A reset ends the walk. Its last accepted iterate is usually still
        // pending, because the optimizer reports the final step accepted and then
        // stops, so no evaluation folds it in (#2627). Fold it in and file the
        // walk's certified mode under its θ before the caller's seed replaces the
        // incumbent.
        self.adopt_accepted_steps();
        if let Some(mode) = self.walk_iterate.take() {
            let key = theta_bits(&mode.rho);
            self.walk_endpoints.retain(|(bits, _)| *bits != key);
            self.walk_endpoints.push((key, mode));
        }
        self.warm_cache = self.reset_warm_cache.clone();
        self.terminal_mode = None;
        // The reset seed is the caller's, not an evaluated incumbent, and steps
        // accepted before the reset belong to the previous search (#2668).
        self.incumbent_established = false;
        self.pending_first_order_mode = None;
        self.accepted_steps_adopted = self.accepted_step_signal.load(Ordering::Relaxed);
        // #2349: the cold-reeval latch deliberately SURVIVES reset. `reset` runs
        // between screened seeds/retries AND immediately before terminal
        // certification (run.rs finalize/certify); clearing it there would
        // re-evaluate `result.rho` WARM — reintroducing the very warm-start
        // hysteresis whose descent found that point on the COLD surface, and the
        // certificate would disagree with the trajectory that produced it. Once
        // any evaluation has revealed the near-separating stall, the whole fit
        // (all remaining seeds and the terminal certificate) stays on the
        // consistent cold surface.
    }

    /// Begin one derivative-bearing outer evaluation transaction.
    ///
    /// Clearing happens before any fallible work. Only
    /// [`Self::install_terminal_mode`] commits a replacement, so every error or
    /// infeasible return leaves the state empty rather than exposing an older
    /// coefficient basin as terminal evidence.
    pub(crate) fn begin_terminal_evaluation(&mut self) {
        self.terminal_mode = None;
    }

    pub(crate) fn install_terminal_mode(
        &mut self,
        theta: &Array1<f64>,
        objective: f64,
        gradient: &Array1<f64>,
        mode: CustomFamilyOwnedMode,
    ) {
        self.terminal_mode = Some(CustomFamilyTerminalMode {
            theta: theta.clone(),
            objective,
            gradient: gradient.clone(),
            mode,
        });
    }

    pub(crate) fn seed_cached_beta(
        &mut self,
        rho_dim: usize,
        specs: &[ParameterBlockSpec],
        beta: &Array1<f64>,
    ) -> Result<gam_solve::rho_optimizer::SeedOutcome, EstimationError> {
        // A seed β whose length disagrees with this fit's per-block
        // coefficient widths is NOT an error: the outer warm-start cache
        // looks up a *row-relaxed* prefix (`cache_seed_key`), so two folds
        // of the same model share an ρ-dim and transfer ρ, but their
        // realized basis ranks — hence the flattened inner β length — are
        // row-population dependent and legitimately differ across folds
        // (the LOSO p=37-vs-p=85 case). Cross-length β transfer is the job
        // of the gauge-projected `FitArtifact` channel, which re-expresses
        // the parent's raw β into this fold's reduced subspace. Here we
        // simply decline the incompatible β and let the (already-installed)
        // ρ seed stand — a ρ-only resume, never a full cold start.
        let expected = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        if beta.len() != expected {
            return Ok(gam_solve::rho_optimizer::SeedOutcome::Incompatible);
        }
        let warm_start = constrained_warm_start_from_cached_beta(rho_dim, specs, beta)?;
        self.reset_warm_cache = Some(warm_start.clone());
        self.warm_cache = Some(warm_start);
        // A caller's seed β is not an evaluated incumbent (#2668).
        self.incumbent_established = false;
        self.pending_first_order_mode = None;
        self.last_error = None;
        Ok(gam_solve::rho_optimizer::SeedOutcome::Installed)
    }
}

/// Sealed terminal payload owned by the custom-family outer objective.
///
/// `theta`, `objective`, and `gradient` are retained bit-for-bit beside the
/// coefficient mode so the optimizer's certified result can be bound to the
/// exact evaluator state before fit assembly consumes it.
pub(crate) struct CustomFamilyTerminalMode {
    pub(crate) theta: Array1<f64>,
    pub(crate) objective: f64,
    pub(crate) gradient: Array1<f64>,
    pub(crate) mode: CustomFamilyOwnedMode,
}

pub struct CustomFamilyJointHyperResult {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub outer_hessian: gam_problem::HessianValue,
    /// Exact non-rho coordinates used to realize this evaluation.
    pub hyper_values: Array1<f64>,
    pub warm_start: CustomFamilyWarmStart,
    /// `false` when the inner blockwise/Newton solve hit its divergence
    /// early-exit or its max-cycle cap. Envelope-theorem outer gradients
    /// and analytic outer Hessians are valid only at a stationary β̂ —
    /// callers that consume `gradient`/`outer_hessian` MUST gate on this
    /// flag and treat non-converged evaluations as inexact (e.g. let ARC
    /// back off the trust region) rather than feeding pathological
    /// derivatives into the outer optimizer.
    pub inner_converged: bool,
}

/// Opaque ownership token for the exact coefficient mode that produced one
/// profiled outer-objective value.
///
/// A warm start is only a seed and cannot prove mode identity for a nonconvex
/// coefficient problem.  This carrier instead owns the complete converged
/// inner result, the exact profiled-objective bits, and the smoothing prefix
/// used to produce them.  Spatial optimization retains this token alongside
/// its terminal evaluation and fit assembly consumes it without another inner
/// solve.
pub struct CustomFamilyOwnedMode {
    pub(crate) objective: f64,
    pub(crate) rho: Array1<f64>,
    pub(crate) hyper_values: Array1<f64>,
    pub(crate) inner: BlockwiseInnerResult,
}

/// Analytic joint-hyper result together with its exact owned coefficient mode.
pub struct CustomFamilyJointHyperOwnedResult {
    pub result: CustomFamilyJointHyperResult,
    pub mode: CustomFamilyOwnedMode,
}

pub struct CustomFamilyJointHyperEfsResult {
    pub efs_eval: gam_problem::EfsEval,
    pub warm_start: CustomFamilyWarmStart,
    pub hyper_values: Array1<f64>,
    /// See [`CustomFamilyJointHyperResult::inner_converged`]. EFS gradients
    /// also assume a stationary inner solve.
    pub inner_converged: bool,
}

/// EFS joint-hyper result together with its exact owned coefficient mode.
pub struct CustomFamilyJointHyperEfsOwnedResult {
    pub result: CustomFamilyJointHyperEfsResult,
    pub mode: CustomFamilyOwnedMode,
}

pub(crate) struct OuterObjectiveEvalResult {
    pub(crate) objective: f64,
    pub(crate) criterion_components: [f64; 4],
    pub(crate) gradient: Array1<f64>,
    pub(crate) outer_hessian: gam_problem::HessianValue,
    pub(crate) warm_start: ConstrainedWarmStart,
    pub(crate) inner_converged: bool,
    pub(crate) hyper_values: Array1<f64>,
    pub(crate) ext_mode_response_cols: Option<Array2<f64>>,
    /// Kept rank of the pseudo-log-determinant this evaluation priced; `None` when the
    /// criterion is not projected. Two evaluations whose kept ranks differ price two
    /// different criteria (#2765).
    pub(crate) criterion_rank: Option<usize>,
    /// The exact coefficient mode used to assemble this objective payload.
    ///
    /// Keeping the owned result here lets an atomic multi-start evaluation
    /// reuse the already-certified mode for derivative assembly. A warm start
    /// is only a seed/cache carrier and must never stand in for this identity:
    /// re-entering the inner solver can select a different nonconvex basin.
    pub(crate) inner: BlockwiseInnerResult,
}

/// Publish an evaluation's criterion decomposition and its selected coefficient
/// mode to an in-flight outer-seed probe. A no-op outside a probe evaluation.
///
/// Every custom-family outer evaluator that owns a converged inner mode calls
/// this, so a probe lent at a seed sees the mode the evaluation priced whichever
/// route (ρ-only or joint-hyper) the fit took.
pub(crate) fn publish_outer_selected_evaluation(result: &OuterObjectiveEvalResult) {
    // An ordinary fit pays a thread-local read here, not a coefficient copy per
    // outer evaluation (#2460).
    if !gam_solve::estimate::outer_eval_capture::outer_seed_capture_armed() {
        return;
    }
    gam_solve::estimate::outer_eval_capture::record_outer_criterion_components(
        result.objective,
        result.criterion_components,
    );
    let selected_beta = Array1::from_iter(
        result
            .inner
            .block_states
            .iter()
            .flat_map(|state| state.beta.iter().copied()),
    );
    gam_solve::estimate::outer_eval_capture::record_outer_selected_mode(
        selected_beta,
        result.ext_mode_response_cols.clone(),
    );
}

pub(crate) fn outer_eval_result_into_joint_hyper_owned_result(
    result: OuterObjectiveEvalResult,
) -> CustomFamilyJointHyperOwnedResult {
    publish_outer_selected_evaluation(&result);
    let OuterObjectiveEvalResult {
        objective,
        gradient,
        outer_hessian,
        warm_start,
        inner_converged,
        hyper_values,
        inner,
        ..
    } = result;
    let rho = warm_start.rho.clone();
    CustomFamilyJointHyperOwnedResult {
        result: CustomFamilyJointHyperResult {
            objective,
            gradient,
            outer_hessian,
            hyper_values: hyper_values.clone(),
            warm_start: CustomFamilyWarmStart { inner: warm_start },
            inner_converged,
        },
        mode: CustomFamilyOwnedMode {
            objective,
            rho,
            hyper_values,
            inner,
        },
    }
}

pub(crate) struct OwnedDenseHessianOperator {
    pub(crate) matrix: Array2<f64>,
}

#[cfg(test)]
mod test_support {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use super::{ConstrainedWarmStart, CustomOuterState};

    impl CustomOuterState {
        /// Test-only constructor: fresh private cold-reeval (#2349) and
        /// accepted-step (#2668) signals, for tests that exercise neither.
        pub(crate) fn new(warm_start: Option<ConstrainedWarmStart>) -> Self {
            Self::new_with_cold_signal(
                warm_start,
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicUsize::new(0)),
            )
        }
    }
}

#[cfg(test)]
mod edf_trace_admission_2901_tests {
    use super::*;
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::array;

    /// #2901: `H ⪰ λS` bounds each penalty's trace by its rank, so a Hessian that
    /// is certified numerically on a custom family, whose Hessian is observed
    /// information. `H = 5I` against `λS = 4I` certifies and publishes `8/5`. `H = I`
    /// against the same penalty has no certified rank bound: its raw trace 8 publishes
    /// unclamped, where the old clamp published the rank 2. A non-positive-definite
    /// `H = −I` is not certified either, so its trace −8 below its band publishes raw,
    /// as the indefinite ambient precision of a cone-constrained mode does (#2635).
    #[test]
    fn the_custom_family_edf_certifies_numerically_and_publishes_an_uncertified_negative_trace_2901(
    ) {
        let specs = vec![ParameterBlockSpec {
            name: "rank_two_block".to_string(),
            design: DesignMatrix::from(Array2::<f64>::zeros((2, 2))),
            offset: Array1::zeros(2),
            penalties: vec![PenaltyMatrix::Dense(array![[4.0, 0.0], [0.0, 4.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }];
        let lambdas = array![1.0];
        let (edf_total, _, _, penalty_trace, rank_bound) = custom_family_blockwise_edf(
            &Array2::from_diag(&array![5.0, 5.0]),
            &specs,
            &lambdas.view(),
        )
        .expect("a dominated trace publishes");
        approx::assert_relative_eq!(penalty_trace[0], 1.6, epsilon = 1e-12);
        approx::assert_relative_eq!(edf_total, 0.4, epsilon = 1e-12);
        assert!(
            matches!(
                rank_bound[0],
                gam_solve::estimate::EdfRankBound::Certified(
                    gam_solve::estimate::EdfRankCertificate::Numerical { .. }
                )
            ),
            "{rank_bound:?}"
        );

        let (unbounded_total, edf_by_penalty, block_edf, unbounded_trace, unbounded_bound) =
            custom_family_blockwise_edf(&Array2::eye(2), &specs, &lambdas.view())
                .expect("an indefinite data curvature publishes its raw trace");
        assert!(
            matches!(
                unbounded_bound[0],
                gam_solve::estimate::EdfRankBound::Uncertified { smallest_pivot, band }
                    if smallest_pivot < -band
            ),
            "{unbounded_bound:?}"
        );
        approx::assert_relative_eq!(unbounded_trace[0], 8.0, epsilon = 1e-12);
        approx::assert_relative_eq!(edf_by_penalty[0], -6.0, epsilon = 1e-12);
        approx::assert_relative_eq!(block_edf[0], -6.0, epsilon = 1e-12);
        approx::assert_relative_eq!(unbounded_total, -6.0, epsilon = 1e-12);

        let (negative_total, negative_by_penalty, negative_block, negative_trace, negative_bound) =
            custom_family_blockwise_edf(&(-Array2::<f64>::eye(2)), &specs, &lambdas.view())
                .expect("an uncertified negative trace publishes raw");
        assert!(
            matches!(
                negative_bound[0],
                gam_solve::estimate::EdfRankBound::Uncertified { smallest_pivot, band }
                    if smallest_pivot < -band
            ),
            "{negative_bound:?}"
        );
        approx::assert_relative_eq!(negative_trace[0], -8.0, epsilon = 1e-12);
        approx::assert_relative_eq!(negative_by_penalty[0], 10.0, epsilon = 1e-12);
        approx::assert_relative_eq!(negative_block[0], 10.0, epsilon = 1e-12);
        approx::assert_relative_eq!(negative_total, 10.0, epsilon = 1e-12);
    }
}
