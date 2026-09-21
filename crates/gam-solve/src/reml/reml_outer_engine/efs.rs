use super::*;

/// Fill the penalty-like (ρ and extended-τ) entries of the EFS step vector with
/// the universal-form multiplicative update `Δ = log(1 − 2·g_full/q_eff)`.
///
/// This is the single source of the penalty-like EFS arithmetic shared by
/// [`compute_efs_update`] and [`compute_hybrid_efs_update`]; both produce
/// byte-identical ρ/τ steps because they route through here. ψ (design-moving)
/// ext coordinates are left at `0.0` — the caller that needs them
/// ([`compute_hybrid_efs_update`]) overwrites those with the preconditioned
/// gradient step. `q_eff` for ρ carries the gamma-precision-rate prior
/// adjustment; τ uses the bare penalty-quadratic scale.
///
/// The per-coordinate arithmetic reads only that coordinate's own gradient
/// entry, penalty root, and λ, so it fans across rayon once the block is large
/// enough (from a non-rayon caller); the write-back is deterministic, so the
/// result is independent of the fan-out decision.
fn efs_penalty_like_steps(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    gradient: &[f64],
) -> Result<Vec<f64>, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    assert_eq!(
        gradient.len(),
        total,
        "efs_penalty_like_steps: gradient length {} != n_rho({k}) + n_ext({ext_dim})",
        gradient.len(),
    );
    let mut steps = vec![0.0; total];

    let (profiled_scale, dp_cgrad) = efs_profiling(solution)?;
    let lambdas = gam_problem::checked_exp_log_strengths(rho.iter().copied())
        .map_err(|error| format!("EFS rho: {error}"))?;
    let penalty_quad_atom = crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
        &lambdas,
        &solution.penalty_coords,
        &solution.beta,
    )
    .map_err(|error| format!("EFS penalty-quadratic layout: {error}"))?;

    // ── ρ coordinates: universal-form EFS with the gamma-precision-rate q_eff.
    let rho_step = |idx: usize| -> (usize, Option<f64>) {
        let lambda = lambdas[idx];
        let a_i = penalty_quad_atom.rho_frozen_d1(idx);
        let q_eff = efs_q_eff_with_gamma_rate(
            efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
            lambda,
            &solution.rho_prior,
            idx,
        );
        (idx, efs_log_step_from_grad(q_eff, gradient[idx]))
    };
    let rho_candidates: Vec<(usize, Option<f64>)> =
        if k >= HYBRID_EFS_SCALAR_PAR_THRESHOLD && gam_runtime::parallel::at_top_level() {
            gam_runtime::parallel::fan_out(|| (0..k).into_par_iter().map(rho_step).collect())
        } else {
            (0..k).map(rho_step).collect()
        };
    for (idx, candidate) in rho_candidates {
        if let Some(step) = candidate {
            steps[idx] = step;
        }
    }

    // ── Extended penalty-like (τ) coordinates: same Wood–Fasiolo update, bare
    // q_eff. ψ (design-moving) ext coords are skipped (left at 0.0): EFS has no
    // convergence guarantee there.
    let tau_local: Vec<usize> = solution
        .ext_coords
        .iter()
        .enumerate()
        .filter_map(|(ext_idx, coord)| coord.is_penalty_like.then_some(ext_idx))
        .collect();
    let tau_step = |ext_idx: usize| -> (usize, Option<f64>) {
        let coord = &solution.ext_coords[ext_idx];
        let g_idx = k + ext_idx;
        let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
        (g_idx, efs_log_step_from_grad(q_eff, gradient[g_idx]))
    };
    let tau_candidates: Vec<(usize, Option<f64>)> = if tau_local.len()
        >= HYBRID_EFS_SCALAR_PAR_THRESHOLD
        && gam_runtime::parallel::at_top_level()
    {
        gam_runtime::parallel::fan_out(|| tau_local.into_par_iter().map(tau_step).collect())
    } else {
        tau_local.into_iter().map(tau_step).collect()
    };
    for (g_idx, candidate) in tau_candidates {
        if let Some(step) = candidate {
            steps[g_idx] = step;
        }
    }

    Ok(steps)
}

/// Extended Fellner–Schall update for ρ and penalty-like (τ) hyperparameters.
///
/// Universal-form multiplicative log-λ update driven by the *full* outer
/// gradient `g_full = ∂V_total/∂θ_i`:
///
/// ```text
///   Δρ_i = log( 1 − 2 · g_full[i] / q_eff_i ).
/// ```
///
/// `q_eff_i = 2 · penalty_term_i` is the penalty-quadratic contribution
/// that `outer_gradient_entry` already pairs with the rest of the
/// gradient — i.e. `2·a_i` for `Fixed` dispersion, `2·dp_cgrad·a_i / φ̂`
/// for `ProfiledGaussian`. Since `g_full = (q_eff + t − d)/2 + g_extra`
/// covers both the base REML/LAML stationarity (`g_extra = 0`,
/// recovering the canonical `log((d − t)/q_eff)`) and any out-of-band
/// augmentations — Tierney–Kadane corrections, smoothing-parameter
/// priors, Firth bias-reduction, monotonicity barriers, SAS log-δ ridge
/// — the step automatically targets the right *augmented* stationarity
/// without any per-augmentation post-correction.
///
/// At any stationary point of `V_total`, `g_full = 0`, so `Δρ = 0`.
/// In the over-correction regime (`2·g_full ≥ q_eff`) the multiplicative
/// form has no root and the helper `efs_log_step_from_grad` returns the
/// multiplicative model's Newton step `−2·g_full/q_eff`; the outer cost
/// line-search sizes it and the canonical formula resumes once the
/// iterate re-enters the stable regime. In the pathological regime
/// (`q_eff ≤ 0`, e.g. when the inner solver placed `β̂` exactly on
/// `null(S)`) the step is zero and the iteration relies on the outer
/// fallback.
///
/// ## EFS does not generalize to ψ coordinates
///
/// EFS needs `A_k = ∂S/∂ρ_k ⪰ 0` and a parameter-independent nullspace.
/// For ψ (design-moving) coordinates, `B_{ψ_j}` contains design-motion
/// and likelihood-curvature terms with potentially mixed inertia. The
/// scalar counterexample (response.md Section 2) shows that no update
/// rule based only on `{a, tr(H⁻¹B), tr(H⁻¹BH⁻¹B)}` can be a universal
/// descent direction for V on a ψ. ψ coordinates use the preconditioned
/// gradient step in [`compute_hybrid_efs_update`] instead.
///
/// ## Hessian-drift corrections
///
/// `g_full` is the same gradient `reml_laml_evaluate` produces in
/// `EvalMode::ValueAndGradient`, which already includes the third-
/// derivative `C[v_k]` IFT correction for non-Gaussian families. The
/// EFS step inherits this correction automatically through `g_full`.
/// Gaussian/quadratic likelihoods have beta-independent observed Hessians,
/// so `C[v_k] = 0` and the classical trace fixed point is exact. For
/// non-Gaussian likelihoods, the pure MacKay/Tipping explicit-trace update
/// is exact only after the logdet Hessian-drift correction is included in
/// the outer gradient.
///
/// # Arguments
/// - `solution`: Converged inner state (β̂, H, penalties, HessianFactorization).
/// - `rho`: Current log-smoothing parameters.
/// - `gradient`: Full outer gradient `∂V_total/∂θ`, length
///   `n_rho + n_ext`. The caller must run
///   [`EvalMode::ValueAndGradient`] when
///   evaluating the cost so this slice is available.
///
/// # Returns
/// A vector of additive steps for all coordinates: first the ρ block,
/// then the ext block (in the same order as `solution.ext_coords`).
/// Apply as `θ_i^new = θ_i + step[i]`. Steps for ψ coordinates
/// (`is_penalty_like == false`) are always 0; the hybrid update handles
/// them.
///
/// Steps are returned whole: the outer fixed-point bridge clips them to the
/// outer domain and sizes them by its cost line search (#2902).
pub fn compute_efs_update(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    gradient: &[f64],
) -> Result<Vec<f64>, String> {
    // Pure penalty-like (ρ + τ) EFS: the universal-form multiplicative update
    // `Δ = log(1 − 2·g_full/q_eff)`, identical to the canonical `log((d−t)/q_eff)`
    // when no out-of-band cost terms exist (TK, prior, Firth, barrier, SAS
    // ridge) and shifted by exactly the residual gradient when they do. Any ψ
    // (design-moving) ext coords are left at 0.0; only [`compute_hybrid_efs_update`]
    // fills those. This is the same helper the hybrid path uses for its ρ/τ
    // block, so the two steppers agree bit-for-bit off the ψ coordinates.
    efs_penalty_like_steps(solution, rho, gradient)
}

/// Minimum number of scalar ρ/τ EFS candidates before `compute_hybrid_efs_update`
/// fans out with rayon.  Smaller blocks are common (1-4 smoothing parameters),
/// where task scheduling costs dominate the independent arithmetic.
pub(crate) const HYBRID_EFS_SCALAR_PAR_THRESHOLD: usize = 8;

/// Minimum number of independent ψ-ψ Gram entries before exact trace assembly
/// fans out with rayon.  This is expressed in upper-triangle pair count rather
/// than `n_psi` so 5 ψ coordinates (15 pairs) stay serial while moderate
/// anisotropic/design-moving blocks parallelize.
pub(crate) const HYBRID_EFS_GRAM_PAIR_PAR_THRESHOLD: usize = 24;

/// Minimum number of ψ drifts before materialization/projection is done in
/// parallel during exact Gram assembly.
pub(crate) const HYBRID_EFS_PSI_DRIFT_PAR_THRESHOLD: usize = 8;

/// Result of the hybrid EFS update, containing both the step vector and
/// metadata needed for backtracking on the ψ block.
pub struct HybridEfsResult {
    /// Combined step vector (EFS for ρ/τ, preconditioned gradient for ψ).
    pub steps: Vec<f64>,
    /// Indices of ψ (design-moving) coordinates in the full θ vector.
    /// Empty if no ψ coordinates are present.
    pub psi_indices: Vec<usize>,
    /// Raw REML/LAML gradient restricted to ψ coordinates.
    /// Length matches `psi_indices.len()`.
    pub psi_gradient: Vec<f64>,
}

/// Hybrid EFS + preconditioned gradient update.
///
/// Computes a combined step for all hyperparameters:
/// - **ρ (penalty-like) coordinates**: standard EFS multiplicative fixed-point
///   update, identical to [`compute_efs_update`].
/// - **ψ (design-moving) coordinates**: safeguarded preconditioned gradient step
///   using the trace Gram matrix as preconditioner:
///
///   ```text
///   Δψ = -G⁺ g_ψ
///   ```
///
///   where:
///   - `g_ψ` is the REML/LAML gradient restricted to the ψ block
///   - `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)` is the trace Gram matrix for ψ-ψ pairs
///   - `G⁺` is the Moore-Penrose pseudoinverse, truncated at the eigensolver's
///     rounding band `n·ε·λ_max`
///
/// ## Why this works (reference: response.md Section 2)
///
/// The trace Gram matrix G is the same object that EFS uses as its scalar
/// denominator for penalty-like coordinates. For ψ coordinates, G still
/// captures the local curvature structure `tr(H⁻¹ B_d H⁻¹ B_e)` — it is
/// the natural metric on the ψ-subspace induced by the penalized likelihood.
/// However, unlike the EFS case, we cannot derive a monotone fixed-point
/// iteration from G alone because B_ψ may have mixed inertia (the Frobenius
/// norm `tr(H⁻¹BH⁻¹B)` is always positive but does not bound the true
/// curvature).
///
/// The preconditioned gradient `Δψ = -G⁺ g_ψ` is the cheap replacement
/// recommended by the math team: it uses the same trace Gram matrix, stays
/// at O(1) H⁻¹ solves per iteration (same as pure EFS), and avoids
/// pretending that the Gram denominator is the true scalar curvature.
/// Compare with full BFGS which requires O(dim(θ)) gradient evaluations
/// (each involving a full inner solve) per outer step.
///
/// ## Step-size safeguarding
///
/// 1. Compute G for the ψ-ψ block from H⁻¹ B_d products (already available).
/// 2. Pseudoinverse: G⁺ via eigendecomposition, dropping the eigenvalues inside
///    the eigensolver's rounding band `n·ε·λ_max`, which it cannot tell from zero.
/// 3. Step: `Δψ = -G⁺ g_ψ`, taken whole.
/// 4. Backtracking (handled by caller): the outer fixed-point bridge clips
///    the *whole* combined step to the outer domain and wraps it in a cost
///    line search, halving the step length over the full vector. If no resolvable
///    contraction is accepted it surfaces the first-order fallback request.
///
/// # Arguments
/// - `solution`: Converged inner state (β̂, H, penalties, HessianFactorization).
/// - `rho`: Current log-smoothing parameters.
/// - `gradient`: Full REML/LAML gradient ∂V/∂θ (length = n_rho + n_ext).
///   Must be provided; the hybrid needs the gradient for ψ coordinates.
///
/// # Returns
/// A [`HybridEfsResult`] containing the combined step vector and metadata
/// for backtracking.
pub fn compute_hybrid_efs_update(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    gradient: &[f64],
) -> Result<HybridEfsResult, String> {
    let k = rho.len();
    let hop = &*solution.hessian_op;
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    assert_eq!(
        gradient.len(),
        total,
        "compute_hybrid_efs_update: gradient length {} != n_rho({k}) + n_ext({ext_dim})",
        gradient.len(),
    );

    // ── ρ + extended-τ penalty-like coordinates ──
    //
    // Route the whole penalty-like block through the shared
    // `efs_penalty_like_steps` helper (the same one `compute_efs_update` uses),
    // so the hybrid path's ρ/τ steps are bit-for-bit identical to the pure-EFS
    // path. ψ (design-moving) ext coords come back as 0.0 and are overwritten
    // with the preconditioned gradient step below.
    let mut steps = efs_penalty_like_steps(solution, rho, gradient)?;

    // Classify ψ (design-moving) ext coordinates for the joint preconditioned
    // gradient step. τ (penalty-like) coords are already filled above; here we
    // only collect the ψ indices, preserving their order for the returned
    // metadata.
    let mut psi_local_indices: Vec<usize> = Vec::new(); // index within ext_coords
    let mut psi_global_indices: Vec<usize> = Vec::new(); // index in full θ vector
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        if !coord.is_penalty_like {
            psi_local_indices.push(ext_idx);
            psi_global_indices.push(k + ext_idx);
        }
    }

    // Collect the ψ-block gradient for the caller (for backtracking).
    let psi_gradient: Vec<f64> = psi_global_indices.iter().map(|&gi| gradient[gi]).collect();

    // ── ψ coordinates: preconditioned gradient step ──
    //
    // The preconditioned gradient step for ψ (design-moving) coordinates:
    //
    //   Δψ = -G⁺ g_ψ
    //
    // where G_{de} = tr(H⁻¹ B_d H⁻¹ B_e) is the trace Gram matrix and
    // g_ψ is the REML/LAML gradient restricted to the ψ block.
    //
    // This is the practical replacement for EFS on ψ coordinates recommended
    // by the math team (response.md Section 2). It uses the same trace Gram
    // matrix that EFS computes, stays cheap (O(1) H⁻¹ solves), and avoids
    // the invalid assumption that the Gram norm bounds the true curvature.
    let n_psi = psi_local_indices.len();
    if n_psi > 0 {
        if n_psi == 1 {
            let li = psi_local_indices[0];
            let drift = &solution.ext_coords[li].drift;
            let op = hyper_coord_drift_operator_arc(drift, hop.dim());
            let dense = op.is_none().then(|| drift.materialize());
            let gram = if let Some(dense_hop) = hop.as_dense_spectral() {
                let projected = if let Some(op) = op.as_ref() {
                    dense_hop.projected_operator(&dense_hop.w_factor, op.as_ref())
                } else {
                    dense_hop
                        .projected_matrix(dense.as_ref().expect("dense drift should be cached"))
                };
                dense_hop.trace_projected_cross(&projected, &projected)
            } else {
                trace_hinv_cached_drift_cross(
                    hop,
                    dense.as_ref(),
                    op.as_deref(),
                    dense.as_ref(),
                    op.as_deref(),
                )
            };
            // `G = tr(H⁻¹BH⁻¹B)` is a squared norm: any positive value is a metric
            // for the step, which is line-searched like the rest.
            if gram > 0.0 {
                let global_idx = psi_global_indices[0];
                steps[global_idx] = -psi_gradient[0] / gram;
            }
            return Ok(HybridEfsResult {
                steps,
                psi_indices: psi_global_indices,
                psi_gradient,
            });
        }

        // Step 1: Build the trace Gram matrix
        //   G_{de} = tr(H⁻¹ B_d H⁻¹ B_e).
        //
        // Every backend prices it with exact pairwise cross traces.
        let gram = {
            let mut gram = ndarray::Array2::<f64>::zeros((n_psi, n_psi));
            let parallel_psi_drifts = n_psi >= HYBRID_EFS_PSI_DRIFT_PAR_THRESHOLD
                && gam_runtime::parallel::at_top_level();
            let drift_ops: Vec<Option<Arc<dyn HyperOperator>>> = if parallel_psi_drifts {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                gam_runtime::parallel::fan_out(|| {
                    (0..n_psi)
                        .into_par_iter()
                        .map(|idx| {
                            let drift = &solution.ext_coords[psi_local_indices[idx]].drift;
                            hyper_coord_drift_operator_arc(drift, hop.dim())
                        })
                        .collect()
                })
            } else {
                psi_local_indices
                    .iter()
                    .map(|&li| {
                        let drift = &solution.ext_coords[li].drift;
                        hyper_coord_drift_operator_arc(drift, hop.dim())
                    })
                    .collect()
            };
            let dense_drifts: Vec<Option<Array2<f64>>> = if parallel_psi_drifts {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                gam_runtime::parallel::fan_out(|| {
                    (0..n_psi)
                        .into_par_iter()
                        .map(|idx| {
                            let drift = &solution.ext_coords[psi_local_indices[idx]].drift;
                            drift_ops[idx].is_none().then(|| drift.materialize())
                        })
                        .collect()
                })
            } else {
                psi_local_indices
                    .iter()
                    .enumerate()
                    .map(|(idx, &li)| {
                        let drift = &solution.ext_coords[li].drift;
                        drift_ops[idx].is_none().then(|| drift.materialize())
                    })
                    .collect()
            };
            let pair_count = n_psi * (n_psi + 1) / 2;
            let parallel_gram_pairs = pair_count >= HYBRID_EFS_GRAM_PAIR_PAR_THRESHOLD
                && gam_runtime::parallel::at_top_level();
            if let Some(dense_hop) = hop.as_dense_spectral() {
                // Batch the operator-backed drifts so the chunked X·F sweep
                // is shared across all matching axes (compute_xf runs once,
                // kernel scalars are batched).
                let mut projected_drifts: Vec<Option<Array2<f64>>> =
                    (0..n_psi).map(|_| None).collect();
                let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
                for idx in 0..n_psi {
                    if let Some(op) = drift_ops[idx].as_ref() {
                        op_terms.push((idx, 1.0, op.as_ref()));
                    } else {
                        projected_drifts[idx] = Some(
                            dense_hop.projected_matrix(
                                dense_drifts[idx]
                                    .as_ref()
                                    .expect("dense drift should be cached"),
                            ),
                        );
                    }
                }
                if !op_terms.is_empty() {
                    let batched = projected_operator_terms_batched(
                        n_psi,
                        &op_terms,
                        &dense_hop.w_factor,
                        &dense_hop.projected_factor_cache,
                    );
                    for (idx, _, _) in &op_terms {
                        projected_drifts[*idx] = Some(batched[*idx].clone());
                    }
                }
                let projected_drifts: Vec<Array2<f64>> = projected_drifts
                    .into_iter()
                    .map(|m| m.expect("projected drift filled"))
                    .collect();
                if parallel_gram_pairs {
                    use rayon::iter::{IntoParallelIterator, ParallelIterator};
                    let pair_count = n_psi * (n_psi + 1) / 2;
                    let pair_values: Vec<(usize, usize, f64)> =
                        gam_runtime::parallel::fan_out(|| {
                            (0..pair_count)
                                .into_par_iter()
                                .map(|pair_idx| {
                                    let (d, e) = upper_triangle_pair_from_index(pair_idx, n_psi);
                                    let val = dense_hop.trace_projected_cross(
                                        &projected_drifts[d],
                                        &projected_drifts[e],
                                    );
                                    (d, e, val)
                                })
                                .collect()
                        });
                    for (d, e, val) in pair_values {
                        gram[[d, e]] = val;
                        gram[[e, d]] = val;
                    }
                } else {
                    for d in 0..n_psi {
                        for e in d..n_psi {
                            let val = dense_hop
                                .trace_projected_cross(&projected_drifts[d], &projected_drifts[e]);
                            gram[[d, e]] = val;
                            gram[[e, d]] = val;
                        }
                    }
                }
            } else if parallel_gram_pairs {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                let pair_count = n_psi * (n_psi + 1) / 2;
                let pair_values: Vec<(usize, usize, f64)> = gam_runtime::parallel::fan_out(|| {
                    (0..pair_count)
                        .into_par_iter()
                        .map(|pair_idx| {
                            let (d, e) = upper_triangle_pair_from_index(pair_idx, n_psi);
                            let val = trace_hinv_cached_drift_cross(
                                hop,
                                dense_drifts[d].as_ref(),
                                drift_ops[d].as_deref(),
                                dense_drifts[e].as_ref(),
                                drift_ops[e].as_deref(),
                            );
                            (d, e, val)
                        })
                        .collect()
                });
                for (d, e, val) in pair_values {
                    gram[[d, e]] = val;
                    gram[[e, d]] = val;
                }
            } else {
                for d in 0..n_psi {
                    for e in d..n_psi {
                        let val = trace_hinv_cached_drift_cross(
                            hop,
                            dense_drifts[d].as_ref(),
                            drift_ops[d].as_deref(),
                            dense_drifts[e].as_ref(),
                            drift_ops[e].as_deref(),
                        );
                        gram[[d, e]] = val;
                        gram[[e, d]] = val;
                    }
                }
            }
            gram
        };

        // Step 2: Pseudoinverse G⁺ via eigendecomposition.
        //
        // For small n_psi (typically 2-10 anisotropic axes), this is cheap.
        // Eigenvalues inside the eigensolver's rounding band are dropped: the
        // decomposition cannot tell them from zero (see response.md Section 4
        // for the submanifold view of the step).
        let delta_psi = pseudoinverse_times_vec(&gram, &psi_gradient)?;

        // Step 3: Δψ = -G⁺ g_ψ, taken whole: the outer bridge's line search
        // sizes it together with the ρ/τ block. The negative sign is because
        // we are descending on V(θ) (minimizing).
        for (psi_idx, &global_idx) in psi_global_indices.iter().enumerate() {
            steps[global_idx] = -delta_psi[psi_idx];
        }
    }

    Ok(HybridEfsResult {
        steps,
        psi_indices: psi_global_indices,
        psi_gradient,
    })
}

/// Compute G⁺ v where G⁺ is the pseudoinverse of the positive semi-definite
/// Gram matrix G.
///
/// Uses the eigendecomposition truncated at the eigensolver's rounding band: an
/// eigenvalue at or below `n·ε·λ_max` (negative ones included, which a Gram
/// matrix only produces by rounding) is indistinguishable from zero and
/// contributes nothing. For small matrices (typical n_psi = 2-10), the O(n³)
/// cost is negligible.
pub(crate) fn pseudoinverse_times_vec(
    gram: &ndarray::Array2<f64>,
    v: &[f64],
) -> Result<ndarray::Array1<f64>, String> {
    let n = gram.nrows();
    assert_eq!(n, v.len(), "pseudoinverse_times_vec dimension mismatch");
    if n == 0 {
        return Ok(ndarray::Array1::zeros(0));
    }

    // Special case: scalar (1x1). The entry is its own eigenvalue, and the band
    // `ε·g` never reaches a positive `g`.
    if n == 1 {
        let g = gram[[0, 0]];
        if g <= 0.0 {
            return Ok(ndarray::Array1::zeros(1));
        }
        return Ok(ndarray::Array1::from_vec(vec![v[0] / g]));
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen(gram)?;

    let max_eval = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = n as f64 * f64::EPSILON * max_eval;

    // G⁺ v = Q diag(1/λ_i for λ_i > cutoff, else 0) Q^T v
    let qt_v: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|row| eigenvectors[[row, i]] * v[row]).sum())
        .collect();

    let mut result = ndarray::Array1::zeros(n);
    for i in 0..n {
        if eigenvalues[i] > cutoff {
            let scale = qt_v[i] / eigenvalues[i];
            for row in 0..n {
                result[row] += scale * eigenvectors[[row, i]];
            }
        }
    }
    Ok(result)
}

/// Symmetric eigendecomposition via classical Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored
/// column-wise. Suitable for small matrices (n ≤ 20). For n_psi = 2-10
/// (typical anisotropic axis counts), this converges in 2-5 sweeps.
///
/// This is a self-contained implementation to avoid external dependencies.
/// For larger matrices, use faer's `SelfAdjointEigendecomposition`.
/// Self-adjoint eigendecomposition of `a` through the workspace's one owner
/// (`gam_linalg::faer_ndarray::FaerEigh`), reading the lower triangle.
///
/// This replaced a private Jacobi sweep with its own `MAX_SWEEPS = 100` and
/// `TOL = 1e-15` that returned an unconverged spectrum silently when its
/// sweeps ran out (#2470, #2469): one owner, and a refusal instead of a
/// fall-through.
pub(crate) fn symmetric_eigen(
    a: &ndarray::Array2<f64>,
) -> Result<(Vec<f64>, ndarray::Array2<f64>), String> {
    let (values, vectors) =
        <ndarray::Array2<f64> as gam_linalg::faer_ndarray::FaerEigh>::eigh(a, faer::Side::Lower)
            .map_err(|error| format!("EFS symmetric eigendecomposition: {error}"))?;
    Ok((values.to_vec(), vectors))
}
