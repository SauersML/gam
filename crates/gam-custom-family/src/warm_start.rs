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

pub(crate) fn warm_start_matches_block_log_lambdas(
    seed: &ConstrainedWarmStart,
    block_log_lambdas: &[Array1<f64>],
) -> bool {
    let expected = block_log_lambdas
        .iter()
        .map(|values| values.len())
        .sum::<usize>();
    if seed.rho.len() != expected {
        return false;
    }
    let mut offset = 0usize;
    for block in block_log_lambdas {
        let end = offset + block.len();
        if seed.rho.slice(s![offset..end]) != block.view() {
            return false;
        }
        offset = end;
    }
    true
}

pub(crate) fn cached_inner_mode_from_result(result: &BlockwiseInnerResult) -> CachedInnerMode {
    CachedInnerMode {
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
) -> Result<f64, String> {
    let reml_term = if include_logdet_h {
        0.5 * inner.block_logdet_h.ok_or_else(|| {
            format!("{context}: certified Hessian logdet is unavailable")
        })?
    } else {
        0.0
    } - if include_logdet_s {
        0.5 * inner.block_logdet_s.ok_or_else(|| {
            format!("{context}: certified penalty logdet is unavailable")
        })?
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
) -> Result<(gam_problem::EfsEval, ConstrainedWarmStart, bool), String> {
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
            logdet_enclosure_gap: None,
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
    pub precomputed_edf: Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>)>,
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
}

pub(crate) fn validate_parameter_block_state_finiteness(
    label: &str,
    state: &ParameterBlockState,
) -> Result<(), String> {
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
) -> Result<(), String> {
    if log_lambdas.len() != lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label} length mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            ),
        }
        .into());
    }
    for (idx, (&log_lambda, &lambda)) in log_lambdas.iter().zip(lambdas.iter()).enumerate() {
        let expected = gam_problem::checked_exp_log_strength(log_lambda)
            .map_err(|error| format!("{label} log coordinate {idx}: {error}"))?;
        if lambda.to_bits() != expected.to_bits() {
            return Err(format!(
                "{label}[{idx}] inconsistent with exp(log_lambda): got {lambda}, expected {expected}",
            ));
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
) -> Result<(f64, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    use gam_solve::estimate::reml::reml_outer_engine::penalty_matrix_root;

    let p = penalized_hessian.nrows();
    let total_cols: usize = specs.iter().map(|s| s.design.ncols()).sum();
    if penalized_hessian.ncols() != p || total_cols != p {
        return Err(format!(
            "custom-family edf: penalized Hessian {}x{} inconsistent with total block width {}",
            penalized_hessian.nrows(),
            penalized_hessian.ncols(),
            total_cols
        ));
    }
    let expected_rho: usize = specs.iter().map(|s| s.penalties.len()).sum();
    if lambdas.len() != expected_rho {
        return Err(format!(
            "custom-family edf: lambdas length {} does not match total penalty count {}",
            lambdas.len(),
            expected_rho
        ));
    }

    let h_sym = SymmetricMatrix::Dense(penalized_hessian.clone());
    // EDF and covariance are properties of the fitted Hessian, not of a
    // nearby matrix selected because it factors. Refuse invalid curvature
    // rather than silently reporting inference for a ridge-perturbed estimand.
    let factor = h_sym.factorize().map_err(|error| {
        format!("custom-family edf: exact penalized-Hessian factorization failed: {error}")
    })?;

    let mut edf_by_penalty = vec![0.0_f64; expected_rho];
    // Raw per-penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk), aligned with edf_by_penalty
    // (issue #1219), so per-term EDF assembles as |coeff_range| − Σ tr_kk.
    let mut penalty_trace = vec![0.0_f64; expected_rho];
    let mut block_edf = Vec::with_capacity(specs.len());
    let mut total_trace = 0.0_f64;
    let mut penalty_offset = 0usize;
    let mut block_col_start = 0usize;
    for spec in specs.iter() {
        let block_cols = spec.design.ncols();
        let mut block_edf_acc = block_cols as f64;
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
            let penalty_rank = penalty_matrix_root(&s_local)
                .map_err(|error| {
                    format!(
                        "custom-family edf: penalty {global_k} rank factorization failed: {error}"
                    )
                })?
                .nrows();
            let mut s_full = Array2::<f64>::zeros((p, p));
            if s_local.nrows() == p && s_local.ncols() == p {
                s_full.assign(&s_local);
            } else if s_local.nrows() == block_cols && s_local.ncols() == block_cols {
                let r = block_col_start..block_col_start + block_cols;
                s_full.slice_mut(ndarray::s![r.clone(), r]).assign(&s_local);
            } else {
                return Err(format!(
                    "custom-family edf: penalty {global_k} materialized to {}x{}, expected {p}x{p} or {block_cols}x{block_cols}",
                    s_local.nrows(),
                    s_local.ncols()
                ));
            }
            // tr(H⁻¹ S_k) via H Z = S_k, summing the diagonal of Z.
            let z = factor.solvemulti(&s_full).map_err(|e| {
                format!("custom-family edf trace solve failed for penalty {global_k}: {e}")
            })?;
            let mut trace = 0.0_f64;
            for d in 0..p {
                trace += z[[d, d]];
            }
            let lam_trace = if lambda > 0.0 { lambda * trace } else { 0.0 };
            total_trace += lam_trace;
            penalty_trace[global_k] = lam_trace;
            let penalty_rank = penalty_rank as f64;
            let edf_k = (penalty_rank - lam_trace).clamp(0.0, penalty_rank);
            edf_by_penalty[global_k] = edf_k;
            // The block's edf is the column count minus the total trace this
            // block's penalties spend (so multiple penalties on one block
            // compose), clamped to the block's column count.
            block_edf_acc -= lam_trace;
        }
        block_edf.push(block_edf_acc.clamp(0.0, block_cols as f64));
        penalty_offset += spec.penalties.len();
        block_col_start += block_cols;
    }

    let edf_total = (p as f64 - total_trace).clamp(0.0, p as f64);
    if !edf_total.is_finite()
        || edf_by_penalty.iter().any(|v| !v.is_finite())
        || block_edf.iter().any(|v| !v.is_finite())
        || penalty_trace.iter().any(|v| !v.is_finite())
    {
        return Err("custom-family edf: non-finite effective degrees of freedom".to_string());
    }
    Ok((edf_total, edf_by_penalty, block_edf, penalty_trace))
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
) -> Option<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let geom = reduced_geometry?;
    match custom_family_blockwise_edf(
        geom.penalized_hessian.as_array(),
        &canonical.reduced_specs,
        &lambdas.view(),
    ) {
        Ok(triple) => Some(triple),
        Err(err) => {
            log::warn!(
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
                },
            hessian_psd: Some(true),
            lambdas_railed: Vec::new(),
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
    } = parts;

    // SPEC 20: a fit object only ever comes from a converged optimization.
    // Assembling a `UnifiedFitResult` from a non-converged outer state would
    // mint a degraded fit (previously surfaced as `StalledAtValidMinimum`);
    // non-convergence must instead be raised as a typed error at the solver,
    // with a checkpoint, before ever reaching this assembler. This defensive
    // gate prevents direct assembly callers from bypassing that contract.
    require_converged_outer_for_assembly(outer_converged)?;
    match criterion_certificate.as_ref() {
        Some(certificate) if !certificate.certifies() => {
            return Err(CustomFamilyError::Optimization {
                context: "blockwise_fit_from_parts",
                reason: format!(
                    "refusing to assemble a fit whose analytic outer certificate failed: {}; \
                     no fit was assembled",
                    certificate.summary()
                ),
            }
            .into());
        }
        None if outer_iterations > 0 => {
            return Err(CustomFamilyError::Optimization {
                context: "blockwise_fit_from_parts",
                reason: "refusing to assemble a fit after an outer optimization without its \
                         analytic convergence certificate"
                    .to_string(),
            }
            .into());
        }
        _ => {}
    }
    if block_states.is_empty() {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "blockwise fit requires at least one block state".to_string(),
        }
        .into());
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
        }
        .into());
    }
    let n = specs[0].design.nrows();
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
            ) }.into());
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
            }
            .into());
        }
    }

    let geom = geometry.as_ref().ok_or_else(|| CustomFamilyError::InvalidInput {
        context: "blockwise_fit_from_parts",
        reason: "a converged custom-family fit must retain coefficient gauge and penalized Hessian geometry"
            .to_string(),
    })?;
    {
        geom.validate_numeric_finiteness().map_err(|e| e.to_string())?;
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
            }
            .into());
        }
        let active_dimension = geom.coefficient_gauge.reduced_total();
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != active_dimension || cols != active_dimension {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit.geometry active penalized_hessian must be {active_dimension}x{active_dimension}, got {rows}x{cols}"
                ),
            }
            .into());
        }
        if !geom.coefficient_gauge.is_identity() && precomputed_edf.is_none() {
            return Err(CustomFamilyError::InvalidInput {
                context: "blockwise_fit_from_parts",
                reason: "non-identity active geometry requires its reduced-coordinate EDF; raw penalties cannot be paired with an active-coordinate Hessian"
                    .to_string(),
            }
            .into());
        }
        if let Some(working) = geom.working.as_ref()
            && working.weights.len() != n
        {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.geometry single-diagonal working row count mismatch: got {}, expected {n}",
                working.weights.len(),
            ) }.into());
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
        ) }.into());
    }
    // Effective degrees of freedom and the inference block. The converged
    // coefficient geometry always carries the joint penalized Hessian; compute the
    // mgcv trace edf `p − Σ_k λ_k·tr(H⁻¹ S_k)` here so every custom-family fit
    // (CTN transformation-normal, Dirichlet, …) reports `edf_total` /
    // per-block `edf` like the standard GAM path, instead of leaving inference
    // unpopulated. Optional row evidence is not part of this calculation.
    let (edf_total, edf_by_penalty, block_edf, penalty_trace): (
        f64,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) = match precomputed_edf {
        // Reduced-space edf supplied by the caller (the principled path:
        // the trace is computed where the Hessian is full rank, then
        // reported on the raw fit — exact because the trace edf is
        // reparameterization-invariant).
        Some((edf_total, edf_by_penalty, block_edf, penalty_trace)) => {
            (edf_total, edf_by_penalty, block_edf, penalty_trace)
        }
        // Compute from coefficient precision when the caller did not already
        // supply the basis-invariant reduced-space traces.
        None => {
            let (edf_total, edf_by_penalty, block_edf, penalty_trace) =
                custom_family_blockwise_edf(
                    geom.penalized_hessian.as_array(),
                    specs,
                    &lambdas.view(),
                )
                .map_err(|reason| CustomFamilyError::Optimization {
                    context: "blockwise_fit_from_parts coefficient-geometry EDF",
                    reason: format!("{reason}; refusing to assemble a fit without EDF/inference"),
                })?;
            (edf_total, edf_by_penalty, block_edf, penalty_trace)
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
    let deviance = -2.0 * log_likelihood;

    // Assemble the inference block from the converged geometry. CTN and other
    // custom families estimate their own likelihood scale, so the penalized
    // Hessian is reported unscaled (dispersion = 1) — the EDF trace is
    // dispersion-free, and downstream covariance scaling pairs `H` with the
    // family's own dispersion where needed.
    // #2346: publish the first-order corrected covariance when the outer ρ
    // curvature supplied one — `V_c = V_cond + C`, with the correction matrix
    // and its typed method provenance carried exactly like the standard lane.
    let (smoothing_correction, smoothing_correction_method, corrected_cov, corrected_se) =
        match (&smoothing_corrected, &covariance_conditional) {
            (Some((correction, method)), Some(v_cond))
                if correction.dim() == v_cond.dim() =>
            {
                let corrected = v_cond + correction;
                let se = corrected.diag().mapv(|v| v.max(0.0).sqrt());
                (
                    Some(correction.clone()),
                    Some(method.clone()),
                    Some(corrected),
                    Some(se),
                )
            }
            _ => (None, None, None, None),
        };
    let inference = Some(gam_solve::model_types::FitInference {
        edf_by_block: edf_by_penalty,
        penalty_block_trace: penalty_trace,
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
        penalized_hessian: geom.penalized_hessian.clone(),
        reparam_qs: None,
        dispersion: gam_solve::model_types::Dispersion::UNIT,
        beta_covariance: None,
        beta_standard_errors: None,
        beta_covariance_corrected: corrected_cov.clone(),
        beta_standard_errors_corrected: corrected_se,
        beta_covariance_frequentist: None,
        coefficient_influence: None,
        weighted_gram: None,
        bias_correction_beta: None,
        bias_correction_jacobian: None,
    });

    gam_solve::model_types::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        log_lambdas: log_lambdas.clone(),
        lambdas: lambdas.clone(),
        likelihood_family: None,
        likelihood_scale: gam_problem::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: gam_problem::LogLikelihoodNormalization::UserProvided,
        log_likelihood,
        deviance,
        reml_score: penalized_objective,
        stable_penalty_term,
        penalized_objective,
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
) -> Result<f64, String> {
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
        }
        .into())
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
    pub(crate) last_error: Option<String>,
    pub(crate) initial_gradient_norm: Option<f64>,
    /// The warm-trajectory-adaptive inner cycle cap shared with the outer-eval
    /// closures, plus the full cold budget it starts from. `reset()` restores
    /// the full budget: a reset objective evaluates a fresh seed — possibly a
    /// distant reseed (a snapped rail face, a saddle escape) whose cold inner
    /// solve needs the whole budget — and the adaptation re-tightens on its
    /// own as warm evaluations resume. Leaving the previous trajectory's
    /// tightened cap in place starved the #2349 face-reseed retry at 4 cycles.
    pub(crate) inner_cap: Option<Arc<AtomicUsize>>,
    pub(crate) inner_cap_full: usize,
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
}

impl CustomOuterState {
    pub(crate) fn new_with_cold_signal(
        warm_start: Option<ConstrainedWarmStart>,
        force_cold_signal: Arc<AtomicBool>,
    ) -> Self {
        Self {
            warm_cache: warm_start.clone(),
            reset_warm_cache: warm_start,
            terminal_mode: None,
            last_error: None,
            initial_gradient_norm: None,
            inner_cap: None,
            inner_cap_full: 0,
            outer_derivative_pilot: None,
            force_cold_signal,
            force_cold_latched: false,
        }
    }

    /// Share the warm-adaptive inner cycle cap so [`Self::reset`] can restore
    /// its full cold budget between seeds/retries.
    pub(crate) fn with_inner_cap(mut self, cap: Arc<AtomicUsize>, full: usize) -> Self {
        self.inner_cap = Some(cap);
        self.inner_cap_full = full;
        self
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
            self.initial_gradient_norm = None;
            self.last_error = None;
        }
        transitioned
    }

    pub(crate) fn reset(&mut self) {
        self.warm_cache = self.reset_warm_cache.clone();
        self.terminal_mode = None;
        if let Some(cap) = &self.inner_cap {
            cap.store(self.inner_cap_full, Ordering::Relaxed);
        }
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
    pub(crate) gradient: Array1<f64>,
    pub(crate) outer_hessian: gam_problem::HessianValue,
    pub(crate) warm_start: ConstrainedWarmStart,
    pub(crate) inner_converged: bool,
    pub(crate) hyper_values: Array1<f64>,
    /// The exact coefficient mode used to assemble this objective payload.
    ///
    /// Keeping the owned result here lets an atomic multi-start evaluation
    /// reuse the already-certified mode for derivative assembly. A warm start
    /// is only a seed/cache carrier and must never stand in for this identity:
    /// re-entering the inner solver can select a different nonconvex basin.
    pub(crate) inner: BlockwiseInnerResult,
}

pub(crate) fn outer_eval_result_into_joint_hyper_owned_result(
    result: OuterObjectiveEvalResult,
) -> CustomFamilyJointHyperOwnedResult {
    let OuterObjectiveEvalResult {
        objective,
        gradient,
        outer_hessian,
        warm_start,
        inner_converged,
        hyper_values,
        inner,
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
    use std::sync::atomic::AtomicBool;

    use super::{ConstrainedWarmStart, CustomOuterState};

    impl CustomOuterState {
        /// Test-only constructor: a fresh private cold-reeval signal (#2349),
        /// for tests that never exercise the stuck-stall pulse.
        pub(crate) fn new(warm_start: Option<ConstrainedWarmStart>) -> Self {
            Self::new_with_cold_signal(warm_start, Arc::new(AtomicBool::new(false)))
        }
    }
}
