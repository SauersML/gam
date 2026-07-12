use super::*;

/// Maximum absolute step size in log-λ for the EFS update (prevents
/// overshooting). Each iteration changes `λ` by at most `exp(EFS_MAX_STEP)`.
pub(crate) const EFS_MAX_STEP: f64 = 5.0;

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
/// form is undefined and the helper [`efs_log_step_from_grad`] returns
/// `−EFS_MAX_STEP`; the outer cost line-search trims it and the
/// canonical formula resumes once the iterate re-enters the stable
/// regime. In the pathological regime (`q_eff ≤ 0`, e.g. when the
/// inner solver placed `β̂` exactly on `null(S)`) the step is zero and
/// the iteration relies on the outer fallback.
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
/// Steps are clamped to `[-EFS_MAX_STEP, EFS_MAX_STEP]` so a single
/// iteration cannot move λ by more than `exp(EFS_MAX_STEP)`.
pub fn compute_efs_update(solution: &InnerSolution<'_>, rho: &[f64], gradient: &[f64]) -> Vec<f64> {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    assert_eq!(
        gradient.len(),
        total,
        "compute_efs_update: gradient length {} != n_rho({k}) + n_ext({ext_dim})",
        gradient.len(),
    );
    let mut steps = vec![0.0; total];

    let (profiled_scale, dp_cgrad) = efs_profiling(solution);
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let penalty_quad_atom = crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
        &lambdas,
        &solution.penalty_coords,
        &solution.beta,
    )
    .expect("EFS penalty-quadratic atom must match InnerSolution penalty layout");

    // Universal-form EFS: `Δρ_i = log(1 − 2·g_full[i]/q_eff_i)`. This is
    // identical to the canonical `log((d−t)/q_eff)` when no out-of-band
    // cost terms exist (TK, prior, Firth, barrier, SAS ridge), and shifts
    // the multiplicative target by exactly the residual gradient when
    // they do. We get the augmented stationarity for free, in exchange
    // for one `EvalMode::ValueAndGradient` evaluation per outer
    // iteration.
    for idx in 0..k {
        let lambda = lambdas[idx];
        let a_i = penalty_quad_atom.rho_frozen_d1(idx);
        let q_eff = efs_q_eff_with_gamma_rate(
            efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
            lambda,
            &solution.rho_prior,
            idx,
        );
        if let Some(step) = efs_log_step_from_grad(q_eff, gradient[idx]) {
            steps[idx] = step;
        }
    }

    // ψ coords (`!is_penalty_like`) are skipped: EFS has no convergence
    // guarantee there. The hybrid update supplies a preconditioned
    // gradient step for them.
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        if !coord.is_penalty_like {
            continue;
        }
        let g_idx = k + ext_idx;
        let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
        if let Some(step) = efs_log_step_from_grad(q_eff, gradient[g_idx]) {
            steps[g_idx] = step;
        }
    }

    steps
}

/// Regularization threshold for pseudoinverse of the trace Gram matrix.
///
/// Eigenvalues below `PSI_GRAM_PINV_TOL * max_eigenvalue` are treated as
/// zero when computing the pseudoinverse G⁺. This prevents amplification
/// of noise in near-singular directions of the ψ-ψ Gram matrix.
pub(crate) const PSI_GRAM_PINV_TOL: f64 = 1e-8;

/// Initial step-size damping factor for the preconditioned gradient on ψ.
///
/// The raw step `Δψ_raw = -G⁺ g_ψ` is scaled by α ∈ (0, 1] before
/// applying. This conservative initial value prevents overshooting in
/// early iterations when the quadratic model may be inaccurate.
pub(crate) const PSI_INITIAL_ALPHA: f64 = 1.0;

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
///   Δψ = -α G⁺ g_ψ
///   ```
///
///   where:
///   - `g_ψ` is the REML/LAML gradient restricted to the ψ block
///   - `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)` is the trace Gram matrix for ψ-ψ pairs
///   - `G⁺` is the Moore-Penrose pseudoinverse (truncated at `PSI_GRAM_PINV_TOL`)
///   - `α ∈ (0, 1]` is the damping factor
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
/// 2. Pseudoinverse: G⁺ via eigendecomposition, truncating eigenvalues below
///    `PSI_GRAM_PINV_TOL * max_eigenvalue` to avoid noise amplification in
///    near-singular directions.
/// 3. Raw step: `Δψ_raw = -G⁺ g_ψ`.
/// 4. Damping: `Δψ = α × Δψ_raw` with initial `α = PSI_INITIAL_ALPHA`.
/// 5. Capping: `||Δψ||_∞ ≤ EFS_MAX_STEP` (same cap as ρ coordinates).
/// 6. Backtracking (handled by caller): the outer fixed-point bridge wraps
///    the *whole* combined step in a cost line search, halving α over the
///    full vector. If full-vector backtracking exhausts, it retries with
///    the ψ block zeroed (ρ/τ-only fallback) before surfacing the
///    first-order fallback marker.
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
) -> HybridEfsResult {
    let k = rho.len();
    let hop = &*solution.hessian_op;
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut steps = vec![0.0; total];

    let (profiled_scale, dp_cgrad) = efs_profiling(solution);
    assert_eq!(
        gradient.len(),
        total,
        "compute_hybrid_efs_update: gradient length {} != n_rho({k}) + n_ext({ext_dim})",
        gradient.len(),
    );

    // ── ρ coordinates: universal-form EFS (see compute_efs_update) ──
    //
    // The per-coordinate candidate construction is independent: each candidate
    // reads only the converged β̂, the coordinate root, ρᵢ, and gᵢ.  Build
    // candidates in parallel once the block is large enough, then keep the
    // actual update write-back serial so fallback/backtracking decisions still
    // see a deterministic step vector.
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let penalty_quad_atom = crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
        &lambdas,
        &solution.penalty_coords,
        &solution.beta,
    )
    .expect("hybrid EFS penalty-quadratic atom must match InnerSolution penalty layout");
    let rho_candidates: Vec<(usize, Option<f64>)> =
        if k >= HYBRID_EFS_SCALAR_PAR_THRESHOLD && rayon::current_thread_index().is_none() {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..k)
                .into_par_iter()
                .map(|idx| {
                    let lambda = lambdas[idx];
                    let a_i = penalty_quad_atom.rho_frozen_d1(idx);
                    let q_eff = efs_q_eff_with_gamma_rate(
                        efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
                        lambda,
                        &solution.rho_prior,
                        idx,
                    );
                    (idx, efs_log_step_from_grad(q_eff, gradient[idx]))
                })
                .collect()
        } else {
            (0..k)
                .map(|idx| {
                    let lambda = lambdas[idx];
                    let a_i = penalty_quad_atom.rho_frozen_d1(idx);
                    let q_eff = efs_q_eff_with_gamma_rate(
                        efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
                        lambda,
                        &solution.rho_prior,
                        idx,
                    );
                    (idx, efs_log_step_from_grad(q_eff, gradient[idx]))
                })
                .collect()
        };
    for (idx, candidate) in rho_candidates {
        if let Some(step) = candidate {
            steps[idx] = step;
        }
    }

    // ── Extended penalty-like (τ) coordinates: universal-form EFS ──
    // ── ψ (design-moving) coordinates: collect for preconditioned gradient ──
    //
    // τ coords go through the same Wood–Fasiolo update as ρ. ψ coords are
    // collected and processed jointly via the ψ-ψ trace Gram matrix below.
    let mut psi_local_indices: Vec<usize> = Vec::new(); // index within ext_coords
    let mut psi_global_indices: Vec<usize> = Vec::new(); // index in full θ vector
    let mut tau_local_indices: Vec<usize> = Vec::new(); // penalty-like ext coords

    // Classify ext coordinates serially.  This preserves ψ ordering for the
    // returned metadata and keeps the penalty-like-vs-design-moving decision
    // out of the parallel update fill.
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let g_idx = k + ext_idx;
        if coord.is_penalty_like {
            tau_local_indices.push(ext_idx);
        } else {
            // ψ coordinate: collect for joint preconditioned gradient.
            psi_local_indices.push(ext_idx);
            psi_global_indices.push(g_idx);
        }
    }

    let tau_candidates: Vec<(usize, Option<f64>)> = if tau_local_indices.len()
        >= HYBRID_EFS_SCALAR_PAR_THRESHOLD
        && rayon::current_thread_index().is_none()
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        tau_local_indices
            .to_vec()
            .into_par_iter()
            .map(|ext_idx| {
                let coord = &solution.ext_coords[ext_idx];
                let g_idx = k + ext_idx;
                let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
                (g_idx, efs_log_step_from_grad(q_eff, gradient[g_idx]))
            })
            .collect()
    } else {
        tau_local_indices
            .iter()
            .map(|&ext_idx| {
                let coord = &solution.ext_coords[ext_idx];
                let g_idx = k + ext_idx;
                let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
                (g_idx, efs_log_step_from_grad(q_eff, gradient[g_idx]))
            })
            .collect()
    };
    for (g_idx, candidate) in tau_candidates {
        if let Some(step) = candidate {
            steps[g_idx] = step;
        }
    }

    // Collect the ψ-block gradient for the caller (for backtracking).
    let psi_gradient: Vec<f64> = psi_global_indices.iter().map(|&gi| gradient[gi]).collect();

    // ── ψ coordinates: preconditioned gradient step ──
    //
    // The preconditioned gradient step for ψ (design-moving) coordinates:
    //
    //   Δψ = -α G⁺ g_ψ
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
            if gram.abs() >= PSI_GRAM_PINV_TOL.max(1e-30) {
                let global_idx = psi_global_indices[0];
                let raw_step = -PSI_INITIAL_ALPHA * psi_gradient[0] / gram;
                steps[global_idx] = raw_step.clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
            }
            return HybridEfsResult {
                steps,
                psi_indices: psi_global_indices,
                psi_gradient,
            };
        }

        let total_p = hop.dim();
        let any_psi_operator = psi_local_indices.iter().any(|&li| {
            let drift = &solution.ext_coords[li].drift;
            drift.uses_operator_fast_path()
        });
        let use_stochastic_psi_gram = any_psi_operator
            && total_p > STOCHASTIC_TRACE_DIM_THRESHOLD
            && hop.prefers_stochastic_trace_estimation();

        // Step 1: Build the trace Gram matrix
        //   G_{de} = tr(H⁻¹ B_d H⁻¹ B_e).
        //
        // Large matrix-free/operator-backed problems batch this through the
        // shared stochastic second-order trace estimator. Smaller or fully
        // dense problems use exact pairwise cross traces.
        let gram = if use_stochastic_psi_gram {
            let mut dense_mats = Vec::new();
            let mut coord_has_operator = Vec::with_capacity(n_psi);
            let mut operator_arcs: Vec<Arc<dyn HyperOperator>> = Vec::new();

            for &li in &psi_local_indices {
                let coord = &solution.ext_coords[li];
                if let Some(op) = hyper_coord_drift_operator_arc(&coord.drift, hop.dim()) {
                    coord_has_operator.push(true);
                    operator_arcs.push(op);
                } else {
                    coord_has_operator.push(false);
                    dense_mats.push(coord.drift.materialize());
                }
            }

            let generic_ops: Vec<&dyn HyperOperator> =
                operator_arcs.iter().map(|op| op.as_ref()).collect();
            let impl_ops: Vec<&ImplicitHyperOperator> = generic_ops
                .iter()
                .filter_map(|&op| as_implicit(op))
                .collect();

            stochastic_trace_hinv_crosses(
                hop,
                &dense_mats,
                &coord_has_operator,
                &generic_ops,
                &impl_ops,
            )
        } else {
            let mut gram = ndarray::Array2::<f64>::zeros((n_psi, n_psi));
            let parallel_psi_drifts = n_psi >= HYBRID_EFS_PSI_DRIFT_PAR_THRESHOLD
                && rayon::current_thread_index().is_none();
            let drift_ops: Vec<Option<Arc<dyn HyperOperator>>> = if parallel_psi_drifts {
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                (0..n_psi)
                    .into_par_iter()
                    .map(|idx| {
                        let drift = &solution.ext_coords[psi_local_indices[idx]].drift;
                        hyper_coord_drift_operator_arc(drift, hop.dim())
                    })
                    .collect()
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
                (0..n_psi)
                    .into_par_iter()
                    .map(|idx| {
                        let drift = &solution.ext_coords[psi_local_indices[idx]].drift;
                        drift_ops[idx].is_none().then(|| drift.materialize())
                    })
                    .collect()
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
                && rayon::current_thread_index().is_none();
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
                    let pair_values: Vec<(usize, usize, f64)> = (0..pair_count)
                        .into_par_iter()
                        .map(|pair_idx| {
                            let (d, e) = upper_triangle_pair_from_index(pair_idx, n_psi);
                            let val = dense_hop
                                .trace_projected_cross(&projected_drifts[d], &projected_drifts[e]);
                            (d, e, val)
                        })
                        .collect();
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
                let pair_values: Vec<(usize, usize, f64)> = (0..pair_count)
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
                    .collect();
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
        // We truncate eigenvalues below PSI_GRAM_PINV_TOL * λ_max to form
        // the pseudoinverse, avoiding noise amplification in near-singular
        // directions. This is the standard approach for constrained
        // optimization on submanifolds (see response.md Section 4).
        let delta_psi = pseudoinverse_times_vec(&gram, &psi_gradient, PSI_GRAM_PINV_TOL);

        // Step 3: Apply damping and capping.
        //
        // Δψ = -α × G⁺ g_ψ, capped to ||Δψ||_∞ ≤ EFS_MAX_STEP.
        // The negative sign is because we are descending on V(θ) (minimizing).
        let alpha = PSI_INITIAL_ALPHA;
        for (psi_idx, &global_idx) in psi_global_indices.iter().enumerate() {
            let raw_step = -alpha * delta_psi[psi_idx];
            steps[global_idx] = raw_step.clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
        }
    }

    HybridEfsResult {
        steps,
        psi_indices: psi_global_indices,
        psi_gradient,
    }
}

/// Compute G⁺ v where G⁺ is the pseudoinverse of symmetric matrix G.
///
/// Uses eigendecomposition with truncation: eigenvalues below
/// `tol * max_eigenvalue` are treated as zero. For small matrices
/// (typical n_psi = 2-10), the O(n³) cost is negligible.
pub(crate) fn pseudoinverse_times_vec(
    gram: &ndarray::Array2<f64>,
    v: &[f64],
    tol: f64,
) -> ndarray::Array1<f64> {
    let n = gram.nrows();
    assert_eq!(n, v.len(), "pseudoinverse_times_vec dimension mismatch");
    if n == 0 {
        return ndarray::Array1::zeros(0);
    }

    // Special case: scalar (1x1).
    if n == 1 {
        let g = gram[[0, 0]];
        if g.abs() < tol.max(1e-30) {
            return ndarray::Array1::zeros(1);
        }
        return ndarray::Array1::from_vec(vec![v[0] / g]);
    }

    // Eigendecomposition of symmetric G via the faer crate would be ideal,
    // but to keep this self-contained we use a simple symmetric
    // eigendecomposition via Jacobi rotations for small matrices, or
    // fall back to diagonal-only pseudoinverse for safety.
    //
    // For production quality, this should use faer's `SelfAdjointEigendecomposition`.
    // Here we implement a robust fallback that works for typical n_psi = 2-10.

    // Attempt: use ndarray's built-in symmetric eigendecomposition if available,
    // otherwise fall back to a diagonal approximation.
    //
    // Robust implementation: compute G = Q Λ Q^T via iterative Jacobi.
    // For n ≤ 10 this converges in a handful of sweeps.
    let (eigenvalues, eigenvectors) = symmetric_eigen(gram);

    let max_eval = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = tol * max_eval;

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
    result
}

/// Symmetric eigendecomposition via classical Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored
/// column-wise. Suitable for small matrices (n ≤ 20). For n_psi = 2-10
/// (typical anisotropic axis counts), this converges in 2-5 sweeps.
///
/// This is a self-contained implementation to avoid external dependencies.
/// For larger matrices, use faer's `SelfAdjointEigendecomposition`.
pub(crate) fn symmetric_eigen(a: &ndarray::Array2<f64>) -> (Vec<f64>, ndarray::Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "symmetric_eigen requires square matrix");

    let mut work = a.clone();
    let mut v = ndarray::Array2::<f64>::eye(n);

    // Jacobi iteration: sweep through all off-diagonal pairs, zeroing them.
    // The off-diagonal Frobenius norm converges quadratically, so a near-machine
    // `tol` is reached in a handful of sweeps for the small matrices this serves;
    // `MAX_SWEEPS` is a generous safety cap that the convergence test hits first.
    const MAX_SWEEPS: usize = 100;
    const TOL: f64 = 1e-15;
    // Skip a pair whose off-diagonal magnitude is already two orders below `TOL`
    // (rotating it would only add round-off), and skip a rotation whose `τ`
    // magnitude is so large the pair is numerically diagonal already.
    const PAIR_SKIP_TOL: f64 = TOL * 0.01;
    const TAU_DIAGONAL_THRESHOLD: f64 = 1e15;

    let mut sweep = 0;
    while sweep < MAX_SWEEPS {
        // Check convergence: sum of squares of off-diagonal elements.
        let mut off_diag_sq = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += work[[i, j]] * work[[i, j]];
            }
        }
        if off_diag_sq < TOL * TOL {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[[p, q]];
                if apq.abs() < PAIR_SKIP_TOL {
                    continue;
                }

                let app = work[[p, p]];
                let aqq = work[[q, q]];
                let tau = (aqq - app) / (2.0 * apq);

                // Stable computation of t = sign(τ) / (|τ| + sqrt(1 + τ²))
                let t = if tau.abs() > TAU_DIAGONAL_THRESHOLD {
                    // Nearly diagonal: skip.
                    continue;
                } else {
                    let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply Jacobi rotation to work matrix.
                work[[p, p]] = app - t * apq;
                work[[q, q]] = aqq + t * apq;
                work[[p, q]] = 0.0;
                work[[q, p]] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let wrp = work[[r, p]];
                    let wrq = work[[r, q]];
                    work[[r, p]] = c * wrp - s * wrq;
                    work[[p, r]] = work[[r, p]];
                    work[[r, q]] = s * wrp + c * wrq;
                    work[[q, r]] = work[[r, q]];
                }

                // Accumulate eigenvectors.
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = c * vrp - s * vrq;
                    v[[r, q]] = s * vrp + c * vrq;
                }
            }
        }
        sweep += 1;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| work[[i, i]]).collect();
    (eigenvalues, v)
}
