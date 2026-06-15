//! Dense `K × K` assembled outer Hessian ∂²V/∂ρₖ∂ρₗ.
//!
//! Materialises the full coordinate Hessian using the precomputed
//! [`HessianOperator`] for all linear algebra. Selected when the model's
//! `(n, p, K)` shape keeps dense `p × p` drift storage and pairwise row
//! assembly cheaper than the matrix-free operator path (see `routing`).

use super::*;
use crate::solver::estimate::smooth_floor_dp;

/// Compute the outer Hessian ∂²V/∂ρₖ∂ρₗ.
///
/// Uses the precomputed HessianOperator for all linear algebra.
pub(crate) fn compute_outer_hessian(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    lambdas: &[f64],
    hop: &dyn HessianOperator,
    effective_deriv: &dyn HessianDerivativeProvider,
    workspace: Option<&RemlDerivativeWorkspace<'_>>,
) -> Result<Array2<f64>, String> {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut hess = Array2::zeros((total, total));
    let upper_active_rho = active_upper_rho_mask(rho);
    let curvature_lambdas_storage: Option<Vec<f64>> = if workspace.is_some() {
        None
    } else {
        Some(
            lambdas
                .iter()
                .copied()
                .map(|lambda| rho_curvature_lambda(solution, lambda))
                .collect(),
        )
    };
    let curvature_lambdas: &[f64] = match workspace {
        Some(ws) => ws.curvature_lambdas,
        None => curvature_lambdas_storage
            .as_deref()
            .expect("curvature_lambdas_storage populated when workspace is None"),
    };

    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

    let det2 = solution.penalty_logdet.second.as_ref().ok_or_else(|| {
        "Outer Hessian requested but penalty second derivatives not provided".to_string()
    })?;

    // ── Profiled Gaussian precomputation ──
    let (profiled_phi, profiled_nu, profiled_dp_cgrad, profiled_dp_cgrad2, is_profiled) =
        match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => {
                let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
                let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
                let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
                let phi_hat = dp_c / nu;
                (phi_hat, nu, dp_cgrad, dp_cgrad2, true)
            }
            _ => (1.0, 1.0, 1.0, 0.0, false),
        };

    // ── ρ precomputation ──

    let penalty_a_k_betas_storage: Option<Vec<Array1<f64>>> = if workspace.is_some() {
        None
    } else {
        Some(
            (0..k)
                .map(|idx| {
                    penalty_a_k_beta(&solution.penalty_coords[idx], &solution.beta, lambdas[idx])
                })
                .collect(),
        )
    };
    let curvature_a_k_betas_storage: Option<Vec<Array1<f64>>> = if workspace.is_some() {
        None
    } else {
        Some(
            (0..k)
                .map(|idx| {
                    penalty_a_k_beta(
                        &solution.penalty_coords[idx],
                        &solution.beta,
                        curvature_lambdas[idx],
                    )
                })
                .collect(),
        )
    };
    let penalty_a_k_betas: &[Array1<f64>] = match workspace {
        Some(ws) => ws.rho_penalty_a_k_betas,
        None => penalty_a_k_betas_storage.as_deref().expect("storage set"),
    };
    let curvature_a_k_betas: &[Array1<f64>] = match workspace {
        Some(ws) => ws.rho_curvature_a_k_betas,
        None => curvature_a_k_betas_storage.as_deref().expect("storage set"),
    };

    // The ONE mode-response kernel selection for this Hessian evaluation
    // (#931 pass 2): built at most once — lazily, so the production caller
    // (which supplies every mode response through the workspace) pays
    // nothing — and shared by the ρ- and ext-coordinate fallbacks below,
    // which pre-port each rebuilt the constrained Schur factorization and
    // re-stated the selection rule independently. See
    // `ThetaModeResponseKernel` for the selection math (lifted `K_T` under
    // active inequality constraints, full `H⁻¹` otherwise — the gradient
    // site in `reml_laml_evaluate` makes the same call, so the Hessian-path
    // mode responses cannot desync from the gradient's).
    let mode_kernel_cell: std::cell::OnceCell<ThetaModeResponseKernel<'_>> =
        std::cell::OnceCell::new();
    let mode_kernel = || {
        mode_kernel_cell.get_or_init(|| {
            ThetaModeResponseKernel::select(
                solution.penalty_subspace_trace.as_deref(),
                solution.active_constraints.as_deref(),
                hop,
            )
        })
    };

    let v_ks_storage: Option<Vec<Array1<f64>>> = match workspace.and_then(|ws| ws.rho_v_ks) {
        Some(_) => None,
        None => {
            // IFT mode responses `v = K · a_k` via the shared kernel; the
            // per-vector solve shape of the pre-port assembly is preserved
            // (`respond_one`), and box-masked coordinates short-circuit to
            // exact zeros without a solve, exactly as before.
            let kernel = mode_kernel();
            Some(
                curvature_a_k_betas
                    .iter()
                    .enumerate()
                    .map(|(idx, a_k_beta)| {
                        if upper_active_rho[idx] {
                            Array1::<f64>::zeros(hop.dim())
                        } else {
                            kernel.respond_one(a_k_beta)
                        }
                    })
                    .collect(),
            )
        }
    };
    let v_ks: &[Array1<f64>] = match workspace.and_then(|ws| ws.rho_v_ks) {
        Some(vs) => vs,
        None => v_ks_storage.as_deref().expect("storage set"),
    };

    // Precompute a_k = ½ λₖ(β̂-μₖ)'Sₖ(β̂-μₖ) for profiled Gaussian correction.
    let rho_a_vals: Vec<f64> = (0..k)
        .map(|idx| {
            0.5 * penalty_a_k_quadratic(&solution.penalty_coords[idx], &solution.beta, lambdas[idx])
        })
        .collect();

    // Build pure Aₖ = λₖ Rₖᵀ Rₖ and Ḣₖ = Aₖ + correction for all k.
    //
    // Both quantities are consumed downstream:
    //   - Ḣₖ (first derivative of H) drives the cross-trace Y_k = H⁻¹ Ḣₖ
    //   - Aₖ (penalty derivative only) drives the Ḧ_{kl} base and the second
    //     implicit derivative β_{kl} = H⁻¹(Ḣₗ vₖ + Aₖ vₗ − δₖₗ Aₖ β̂)
    //
    // But Aₖ ≡ Ḣₖ unless a correction was applied to that coordinate, so
    // `a_k_matrices[idx]` stores the pure Aₖ only in that case; otherwise it is
    // `None` and the diagonal base term reads `h_k_matrices[idx]` directly. This
    // drops the per-coordinate Aₖ clone (issue #922) in the common no-correction
    // (Gaussian) path.
    let mut a_k_matrices: Vec<Option<Array2<f64>>> = Vec::with_capacity(k);
    let mut h_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k);
    for idx in 0..k {
        let mut a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);

        let correction: Option<Array2<f64>> = match workspace {
            Some(ws) => match ws.coord_corrections[idx].as_ref() {
                Some(DriftDerivResult::Dense(matrix)) => Some(matrix.clone()),
                Some(DriftDerivResult::Operator(_)) => {
                    if effective_deriv.has_corrections() {
                        effective_deriv.hessian_derivative_correction(&v_ks[idx])?
                    } else {
                        None
                    }
                }
                None => None,
            },
            None => {
                if effective_deriv.has_corrections() {
                    effective_deriv.hessian_derivative_correction(&v_ks[idx])?
                } else {
                    None
                }
            }
        };
        if let Some(corr) = correction {
            a_k_matrices.push(Some(a_k.clone()));
            a_k += &corr;
        } else {
            a_k_matrices.push(None);
        }
        h_k_matrices.push(a_k);
    }

    // ── Adjoint trick precomputation ──
    //
    // For scalar GLMs with C[u] = Xᵀ diag(c ⊙ Xu) X:
    //   h^G          = diag(X G_ε(H) Xᵀ)
    //   z_c          = H⁻¹ Xᵀ (c ⊙ h^G)
    //   tr(G_ε C[u]) = uᵀ Xᵀ (c ⊙ h^G) = uᵀ (Hu_old) · z_c
    //
    // h^G also plugs into the fourth-derivative trace
    //   tr(G_ε Xᵀ diag(w) X) = Σᵢ wᵢ h^G[i],
    // collapsing per-pair O(np²) → O(n) work.
    let glm_ingredients = effective_deriv.scalar_glm_ingredients();
    let leverage = if incl_logdet_h {
        glm_ingredients
            .as_ref()
            .map(|ing| hop.xt_logdet_kernel_x_diagonal(ing.x))
    } else {
        None
    };
    let adjoint_z_c = if incl_logdet_h {
        match (glm_ingredients.as_ref(), leverage.as_ref()) {
            (Some(ing), Some(h_g)) => Some(compute_adjoint_z_c(
                ing,
                hop,
                h_g,
                solution.penalty_subspace_trace.as_deref(),
            )?),
            _ => None,
        }
    } else {
        None
    };

    // ── ext precomputation ──

    // Check if any ext coordinate uses implicit operators and if the problem
    // is large enough to warrant stochastic cross-traces instead of
    // materializing p x p Hessian drift matrices.
    let any_ext_implicit = solution.ext_coords.iter().any(|c| {
        c.drift
            .operator_ref()
            .is_some_and(|op| c.drift.uses_operator_fast_path() && op.is_implicit())
    });
    let total_p = hop.dim();
    // Stochastic cross-traces are only used when:
    // (1) implicit operators are present
    // (2) problem is large (p > 500)
    // (3) dense operator (eigendecomposition-based)
    // (4) logdet_h is included
    // (5) no third-derivative corrections (Gaussian family)
    //
    // Condition (5) ensures correctness: the stochastic estimator uses
    // B_d (the implicit operator) which equals Ḣ_d only when C[v_d] = 0.
    // For non-Gaussian families, Ḣ_d = B_d + C[v_d] and the correction
    // is a dense p x p matrix, so we fall back to dense materialization.
    let use_stochastic_cross_traces = any_ext_implicit
        && can_use_stochastic_logdet_hinv_kernel(hop, total_p, incl_logdet_h)
        && !effective_deriv.has_corrections()
        && solution.penalty_subspace_trace.is_none();

    // Precompute ext solve responses and total Hessian drifts. All ext
    // coordinates use canonical fixed-β stationarity derivatives, so
    // β_i = -K g_i and the correction provider is called with +v_i.
    //
    // INVARIANT: gradient and Hessian must use identical mode responses (same kernel K).
    // Reuse satisfies this automatically; the standalone fallback must replicate it.
    let ext_v_storage: Option<Vec<Array1<f64>>> = match workspace.and_then(|ws| ws.ext_v_is) {
        Some(vs) => {
            if vs.len() != ext_dim {
                return Err(RemlError::DimensionMismatch {
                    reason: format!(
                        "outer Hessian ext mode-response count mismatch: got {}, expected {}",
                        vs.len(),
                        ext_dim
                    ),
                }
                .into());
            }
            None
        }
        None => {
            // IFT mode responses for ext (ψ/τ) coordinates
            // (`β̂_ψ = −K · g_ψ`) through the SAME shared kernel as the ρ
            // fallback above and the gradient site — one selection per
            // evaluation, so the Hessian path structurally cannot take a
            // different solve than the gradient path in any regime.
            let kernel = mode_kernel();
            Some(
                solution
                    .ext_coords
                    .iter()
                    .map(|coord| kernel.respond_one(&coord.g))
                    .collect(),
            )
        }
    };
    let ext_v: &[Array1<f64>] = match workspace.and_then(|ws| ws.ext_v_is) {
        Some(vs) => vs,
        None => ext_v_storage.as_deref().expect("ext_v_storage populated"),
    };
    let mut ext_h_drifts: Vec<DriftDerivResult> = Vec::with_capacity(ext_dim);

    for (coord, v_i) in solution.ext_coords.iter().zip(ext_v.iter()) {
        let correction = if effective_deriv.has_corrections() {
            effective_deriv.hessian_derivative_correction_result(v_i)?
        } else {
            None
        };
        let h_i = hyper_coord_total_drift_result(&coord.drift, correction.as_ref(), hop.dim());

        ext_h_drifts.push(h_i);
    }

    let fourth_trace_matrix =
        if incl_logdet_h && solution.penalty_subspace_trace.is_none() && adjoint_z_c.is_some() {
            match (glm_ingredients.as_ref(), leverage.as_ref()) {
                (Some(ing), Some(h_g)) if ing.d_array.is_some() => {
                    let modes = v_ks.iter().chain(ext_v.iter()).collect::<Vec<_>>();
                    compute_fourth_derivative_trace_matrix(ing, &modes, h_g)?
                }
                _ => None,
            }
        } else {
            None
        };

    // ── Stochastic second-order cross-trace precomputation ──
    //
    // When implicit operators are present and the problem is large, compute
    // the full (total x total) cross-trace matrix
    //   cross[d,e] = tr(H^{-1} Hd H^{-1} He)
    // stochastically. This path is only enabled on backends where the
    // logdet-Hessian cross term is exactly -tr(H^{-1} Hd H^{-1} He).
    //
    // Estimator:
    //   u = H^{-1} z,  q_e = A_e z,  r_e = H^{-1} q_e,  estimate = u^T A_d r_e
    //
    // This avoids materializing the (p x p) Hessian drift matrices for
    // implicit operators, and uses the correct tr(H^{-1} A_d H^{-1} A_e)
    // formula rather than the WRONG tr(A_d H^{-2} A_e).
    //
    // NOTE: The sign convention here gives +tr(H^{-1} Hd H^{-1} He).
    // The outer Hessian uses -tr(H^{-1} Hj H^{-1} Hi) = -(this value).
    let stochastic_cross_traces: Option<Array2<f64>> = if use_stochastic_cross_traces {
        let total_coords = k + ext_dim;
        // Borrow the rho-coordinate Ḣₖ drifts and the dense ext drifts directly
        // — the estimator only ever reads them, so the previous per-coordinate
        // clones (issue #922) were pure copies of `h_k_matrices` / `ext_h_drifts`.
        let mut dense_mats: Vec<&Array2<f64>> = Vec::new();
        let mut coord_has_operator: Vec<bool> = Vec::with_capacity(total_coords);
        let mut operator_arcs: Vec<Arc<dyn HyperOperator>> = Vec::new();

        // rho coordinates: always dense.
        for h_k in h_k_matrices.iter().take(k) {
            dense_mats.push(h_k);
            coord_has_operator.push(false);
        }

        // ext coordinates: dense or operator-backed, including any
        // non-Gaussian third-derivative correction already composed into
        // `ext_h_drifts`.
        for drift in &ext_h_drifts {
            match drift {
                DriftDerivResult::Dense(matrix) => {
                    dense_mats.push(matrix);
                    coord_has_operator.push(false);
                }
                DriftDerivResult::Operator(operator) => {
                    operator_arcs.push(Arc::clone(operator));
                    coord_has_operator.push(true);
                }
            }
        }

        let generic_ops: Vec<&dyn HyperOperator> =
            operator_arcs.iter().map(|op| op.as_ref()).collect();
        let impl_ops: Vec<&ImplicitHyperOperator> = generic_ops
            .iter()
            .filter_map(|op| op.as_implicit())
            .collect();

        Some(stochastic_trace_hinv_crosses_with_floor(
            hop,
            &dense_mats,
            &coord_has_operator,
            &generic_ops,
            &impl_ops,
            Some(Arc::clone(&solution.stochastic_trace_state)),
        ))
    } else {
        None
    };

    // When the rank-deficient LAML fix replaces the full-space logdet
    // kernel with the projected `U_S · H_proj⁻¹ · U_Sᵀ`, the cross-trace
    // `−tr(K Ḣ_j K Ḣ_i)` must also use the projected kernel for the same
    // reason the first-order trace does (the IFT correction `D_β H[v]`
    // spills onto `null(S)` for non-Gaussian families).  Collect the
    // reduced drifts `R_d = U_Sᵀ Ḣ_d U_S` once and reuse them for every
    // pair; per-pair cost is then O(r²) instead of O(p²) per cross.
    let subspace = solution.penalty_subspace_trace.as_deref();
    let reduced_h_drifts: Option<Vec<Array2<f64>>> = subspace.map(|kernel| {
        let mut drifts = h_k_matrices
            .iter()
            .cloned()
            .map(DriftDerivResult::Dense)
            .collect::<Vec<_>>();
        drifts.extend(ext_h_drifts.iter().cloned());
        penalty_subspace_reduce_drifts_batched(kernel, &drifts)
    });
    let exact_logdet_cross_traces = if incl_logdet_h && stochastic_cross_traces.is_none() {
        if let (Some(kernel), Some(reduced)) = (subspace, reduced_h_drifts.as_ref()) {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let n = reduced.len();
            // Each `(i, j)` upper-triangular pair is an independent cross
            // trace `−tr(K · A_i · K · A_j)` over the projected kernel
            // `K = U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ`; the kernel and `reduced` slice
            // are both read-only borrows so the K(K+1)/2 pairs dispatch in
            // parallel, then we stitch the symmetric `n × n` Array2
            // sequentially.
            let pair_count = n * (n + 1) / 2;
            let pair_values: Vec<(usize, usize, f64)> = (0..pair_count)
                .into_par_iter()
                .map(|pair_idx| {
                    let (i, j) = upper_triangle_pair_from_index(pair_idx, n);
                    let value =
                        -kernel.trace_projected_logdet_cross_reduced(&reduced[i], &reduced[j]);
                    (i, j, value)
                })
                .collect();
            let mut out = Array2::<f64>::zeros((n, n));
            for (i, j, value) in pair_values {
                out[[i, j]] = value;
                if i != j {
                    out[[j, i]] = value;
                }
            }
            Some(out)
        } else if let Some(dense_hop) = hop.as_exact_dense_spectral() {
            Some(trace_logdet_hessian_crosses_dense_spectral_drifts(
                dense_hop,
                &h_k_matrices,
                &ext_h_drifts,
            ))
        } else {
            let total_coords = k + ext_dim;
            let mut out = Array2::<f64>::zeros((total_coords, total_coords));
            for ii in 0..total_coords {
                for jj in ii..total_coords {
                    let value = match (ii < k, jj < k) {
                        (true, true) => {
                            hop.trace_logdet_hessian_cross(&h_k_matrices[ii], &h_k_matrices[jj])
                        }
                        (true, false) => trace_logdet_hessian_cross_dense_drift(
                            hop,
                            &h_k_matrices[ii],
                            &ext_h_drifts[jj - k],
                        ),
                        (false, true) => trace_logdet_hessian_cross_dense_drift(
                            hop,
                            &h_k_matrices[jj],
                            &ext_h_drifts[ii - k],
                        ),
                        (false, false) => ext_h_drifts[ii - k]
                            .trace_logdet_hessian_cross(&ext_h_drifts[jj - k], hop),
                    };
                    out[[ii, jj]] = value;
                    if ii != jj {
                        out[[jj, ii]] = value;
                    }
                }
            }
            Some(out)
        }
    } else {
        None
    };

    // ── ρ-ρ block ── (uses shared helpers for all trace computations)
    //
    // The K(K+1)/2 upper-triangular pairs are independent (each writes to a
    // disjoint hess[[kk, ll]]/hess[[ll, kk]] cell pair) and the dominant
    // per-pair cost — `compute_ift_correction_trace` → `kernel.trace_operator`
    // → `op.mul_mat(U_S)` materialising a CompositeHyperOperator over a
    // (p × r) factor — scales like O(family_callback × r × n). At large-scale
    // shape (n ≈ 2·10⁵, r ≈ 24, K = 8) the sequential walk dominates the
    // outer-Hessian wall-clock by 1–2 orders of magnitude, so we dispatch
    // the pair list through rayon and stitch the symmetric Array2 sequentially.
    //
    // `effective_deriv` is `&dyn HessianDerivativeProvider` whose trait bound
    // includes `Send + Sync`, and `hop`, `subspace`, `glm_ingredients`,
    // `adjoint_z_c`, `leverage`, and `fourth_trace_matrix` are all read-only
    // shared state, so the closure body is genuinely thread-safe. The default
    // `HyperOperator::mul_mat` already short-circuits to sequential when
    // `rayon::current_thread_index().is_some()`, preventing nested-rayon
    // oversubscription inside the family callbacks invoked by
    // `compute_ift_correction_trace`.
    let rho_pair_count = k * (k + 1) / 2;
    let rho_pair_start = std::time::Instant::now();
    log::debug!(
        "[compute_outer_hessian rho-rho] starting {} pair(s), k={}",
        rho_pair_count,
        k,
    );

    let build_rho_pair_rhs = |kk: usize, ll: usize| {
        let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
        rhs += &solution.penalty_coords[kk].scaled_matvec(&v_ks[ll], curvature_lambdas[kk]);
        if kk == ll {
            rhs -= &curvature_a_k_betas[kk];
        }
        rhs
    };

    let batched_rho_pair_corrections: Option<Vec<f64>> = if incl_logdet_h
        && subspace.is_some()
        && effective_deriv.has_corrections()
        && effective_deriv.has_batched_hessian_second_derivative_corrections()
    {
        let mut rhs_matrix = Array2::<f64>::zeros((hop.dim(), rho_pair_count));
        for pair_idx in 0..rho_pair_count {
            let (kk, ll) = upper_triangle_pair_from_index(pair_idx, k);
            let rhs = build_rho_pair_rhs(kk, ll);
            rhs_matrix.column_mut(pair_idx).assign(&rhs);
        }
        // WS1a fallback: projected kernel for rank-deficient LAML path.
        let solved = if let Some(kernel) = subspace {
            let mut projected = Array2::<f64>::zeros((hop.dim(), rho_pair_count));
            for pair_idx in 0..rho_pair_count {
                let rhs = rhs_matrix.column(pair_idx).to_owned();
                projected
                    .column_mut(pair_idx)
                    .assign(&kernel.apply_pseudo_inverse(&rhs));
            }
            projected
        } else {
            hop.solve_multi(&rhs_matrix)
        };
        let triples: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = (0..rho_pair_count)
            .map(|pair_idx| {
                let (kk, ll) = upper_triangle_pair_from_index(pair_idx, k);
                (
                    v_ks[kk].clone(),
                    v_ks[ll].clone(),
                    solved.column(pair_idx).to_owned(),
                )
            })
            .collect();
        let corrections = effective_deriv.hessian_second_derivative_corrections_result(&triples)?;
        let mut correction_values = vec![0.0_f64; corrections.len()];
        if let Some(kernel) = subspace {
            let mut present_indices = Vec::new();
            let mut present_drifts = Vec::new();
            for (idx, correction) in corrections.into_iter().enumerate() {
                if let Some(drift) = correction {
                    present_indices.push(idx);
                    present_drifts.push(drift);
                }
            }
            let traced = penalty_subspace_trace_drifts_batched(kernel, &present_drifts);
            for (idx, value) in present_indices.into_iter().zip(traced) {
                correction_values[idx] = value;
            }
        }
        Some(correction_values)
    } else {
        None
    };

    let rho_pair_values: Vec<(usize, usize, f64)> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (0..rho_pair_count)
            .into_par_iter()
            .map(|pair_idx| -> Result<(usize, usize, f64), String> {
                let (kk, ll) = upper_triangle_pair_from_index(pair_idx, k);
                let pair_a = if kk == ll { rho_a_vals[kk] } else { 0.0 };

                let cross_trace = if let Some(ref exact) = exact_logdet_cross_traces {
                    exact[[kk, ll]]
                } else if let Some(ref sct) = stochastic_cross_traces {
                    -sct[[kk, ll]]
                } else {
                    hop.trace_logdet_hessian_cross(&h_k_matrices[kk], &h_k_matrices[ll])
                };

                // Second Hessian drift trace via shared helpers.
                //
                // RHS = Ḣ_l v_k + B_k v_l − δ_{kl} g_k
                // base = δ_{kl} tr(K A_k)
                // correction = compute_ift_correction_trace(RHS, v_k, v_l)
                //
                // `K` is the full-space `G_ε(H)` unless the rank-deficient LAML
                // fix is active, in which case every trace routes through the
                // projected kernel so the outer Hessian matches the projected
                // `½ log|U_Sᵀ H U_S|_+` cost.
                let base = if kk == ll {
                    // Pure Aₖ for the diagonal base term: the stored override when
                    // a correction made Aₖ ≠ Ḣₖ, else Ḣₖ itself (they coincide).
                    let a_kk = a_k_matrices[kk].as_ref().unwrap_or(&h_k_matrices[kk]);
                    if let Some(kernel) = subspace {
                        kernel.trace_projected_logdet(a_kk)
                    } else if solution.penalty_coords[kk].is_block_local() {
                        let (block, start, end) =
                            solution.penalty_coords[kk].scaled_block_local(1.0);
                        hop.trace_logdet_block_local(&block, curvature_lambdas[kk], start, end)
                    } else {
                        hop.trace_logdet_gradient(a_kk)
                    }
                } else {
                    0.0
                };

                let correction = if let Some(corrections) = batched_rho_pair_corrections.as_ref() {
                    corrections[pair_idx]
                } else {
                    let rhs = build_rho_pair_rhs(kk, ll);
                    compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &v_ks[kk],
                        &v_ks[ll],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                        leverage.as_ref(),
                        fourth_trace_matrix.as_ref().map(|trace| trace[[kk, ll]]),
                        subspace,
                    )?
                };

                let h_kl_trace = base + correction;

                let h_val = outer_hessian_entry(
                    rho_a_vals[kk],
                    rho_a_vals[ll],
                    penalty_a_k_betas[ll].dot(&v_ks[kk]),
                    pair_a,
                    cross_trace,
                    h_kl_trace,
                    det2[[kk, ll]],
                    profiled_phi,
                    profiled_nu,
                    profiled_dp_cgrad,
                    profiled_dp_cgrad2,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                Ok((kk, ll, h_val))
            })
            .collect::<Result<Vec<_>, String>>()?
    };

    for (kk, ll, h_val) in rho_pair_values {
        hess[[kk, ll]] = h_val;
        if kk != ll {
            hess[[ll, kk]] = h_val;
        }
    }

    log::debug!(
        "[compute_outer_hessian rho-rho] {} pair(s) done in {:.3}s",
        rho_pair_count,
        rho_pair_start.elapsed().as_secs_f64(),
    );

    // ── ρ-ext cross block ── (uses shared helpers for all trace computations)

    if let Some(ref rho_ext_fn) = solution.rho_ext_pair_fn {
        for rho_idx in 0..k {
            for ext_idx in 0..ext_dim {
                let pair = rho_ext_fn(rho_idx, ext_idx);
                let a_ext = solution.ext_coords[ext_idx].a;

                let (cross_trace, h2_trace) = if incl_logdet_h {
                    let cross_trace = if let Some(ref exact) = exact_logdet_cross_traces {
                        exact[[rho_idx, k + ext_idx]]
                    } else if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[rho_idx, k + ext_idx]]
                    } else {
                        trace_logdet_hessian_cross_dense_drift(
                            hop,
                            &h_k_matrices[rho_idx],
                            &ext_h_drifts[ext_idx],
                        )
                    };

                    // `coord.g` stores g_i = F_{βi} and v_i = H⁻¹g_i, so the
                    // actual mode derivative is β_i = -v_i for both ρ and ext.
                    // Differentiating stationarity gives:
                    //   H β_{rho,ext}
                    //     = -g_{rho,ext} - H_rho β_ext - Ḣ_ext β_rho
                    //     = -g_{rho,ext} + H_rho v_ext + Ḣ_ext v_rho.
                    let mut rhs = -&pair.g;
                    rhs += &solution.penalty_coords[rho_idx]
                        .scaled_matvec(&ext_v[ext_idx], curvature_lambdas[rho_idx]);
                    let beta_rho = v_ks[rho_idx].mapv(|value| -value);
                    rhs += &ext_h_drifts[ext_idx].apply(&v_ks[rho_idx]);

                    let base = compute_base_h2_trace(
                        hop,
                        &pair.b_mat,
                        pair.b_operator.as_deref(),
                        subspace,
                    );

                    let beta_ext = ext_v[ext_idx].mapv(|value| -value);
                    let m_terms = compute_drift_deriv_traces(
                        hop,
                        false, // ρ drift is β-independent
                        solution.ext_coords[ext_idx].b_depends_on_beta,
                        None,
                        Some(ext_idx),
                        &beta_rho,
                        &beta_ext,
                        solution.fixed_drift_deriv.as_ref(),
                        subspace,
                    );

                    let correction = compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &v_ks[rho_idx],
                        &ext_v[ext_idx],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                        leverage.as_ref(),
                        fourth_trace_matrix
                            .as_ref()
                            .map(|trace| trace[[rho_idx, k + ext_idx]]),
                        subspace,
                    )?;

                    (cross_trace, base + m_terms + correction)
                } else {
                    (0.0, 0.0)
                };

                let h_val = outer_hessian_entry(
                    rho_a_vals[rho_idx],
                    a_ext,
                    penalty_a_k_betas[rho_idx].dot(&ext_v[ext_idx]),
                    pair.a,
                    cross_trace,
                    h2_trace,
                    pair.ld_s,
                    profiled_phi,
                    profiled_nu,
                    profiled_dp_cgrad,
                    profiled_dp_cgrad2,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                hess[[rho_idx, k + ext_idx]] = h_val;
                hess[[k + ext_idx, rho_idx]] = h_val;
            }
        }
    }

    // ── ext-ext block ── (uses shared helpers for all trace computations)

    if let Some(ref ext_pair_fn) = solution.ext_coord_pair_fn {
        for ii in 0..ext_dim {
            for jj in ii..ext_dim {
                let pair = ext_pair_fn(ii, jj);
                let coord_i = &solution.ext_coords[ii];
                let coord_j = &solution.ext_coords[jj];

                let (cross_trace, h2_trace) = if incl_logdet_h {
                    let cross_trace = if let Some(ref exact) = exact_logdet_cross_traces {
                        exact[[k + ii, k + jj]]
                    } else if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[k + ii, k + jj]]
                    } else {
                        ext_h_drifts[ii].trace_logdet_hessian_cross(&ext_h_drifts[jj], hop)
                    };

                    // `coord.g` is g_i = F_{βi} and v_i = H⁻¹g_i, hence
                    // β_i = -v_i. Differentiating stationarity gives:
                    //   H β_{ij}
                    //     = -g_{ij} - H_i β_j - Ḣ_j β_i
                    //     = -g_{ij} + H_i v_j + Ḣ_j v_i.
                    let mut rhs = -&pair.g;
                    coord_i
                        .drift
                        .scaled_add_apply(ext_v[jj].view(), 1.0, &mut rhs);
                    rhs += &ext_h_drifts[jj].apply(&ext_v[ii]);

                    let base = compute_base_h2_trace(
                        hop,
                        &pair.b_mat,
                        pair.b_operator.as_deref(),
                        subspace,
                    );

                    let beta_i = ext_v[ii].mapv(|value| -value);
                    let beta_j = ext_v[jj].mapv(|value| -value);
                    let m_terms = compute_drift_deriv_traces(
                        hop,
                        coord_i.b_depends_on_beta,
                        coord_j.b_depends_on_beta,
                        Some(ii),
                        Some(jj),
                        &beta_i,
                        &beta_j,
                        solution.fixed_drift_deriv.as_ref(),
                        subspace,
                    );

                    let correction = compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &ext_v[ii],
                        &ext_v[jj],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                        leverage.as_ref(),
                        fourth_trace_matrix
                            .as_ref()
                            .map(|trace| trace[[k + ii, k + jj]]),
                        subspace,
                    )?;

                    let h2 = base + m_terms + correction;
                    let g_dot_v = coord_i.g.dot(&ext_v[jj]);
                    let pair_g_finite = pair.g.iter().all(|v| v.is_finite());
                    let b_mat_finite = pair.b_mat.iter().all(|v| v.is_finite());
                    let ext_vi_finite = ext_v[ii].iter().all(|v| v.is_finite());
                    let ext_vj_finite = ext_v[jj].iter().all(|v| v.is_finite());
                    let any_non_finite = !cross_trace.is_finite()
                        || !base.is_finite()
                        || !m_terms.is_finite()
                        || !correction.is_finite()
                        || !h2.is_finite()
                        || !pair.a.is_finite()
                        || !pair.ld_s.is_finite()
                        || !g_dot_v.is_finite()
                        || !pair_g_finite
                        || !b_mat_finite;
                    if any_non_finite {
                        // Probe a single bad b_mat entry so we can tell whether
                        // the NaN is structural (whole matrix bad) or localized
                        // to a particular row/col.
                        let mut first_bad_b_mat = None;
                        if !b_mat_finite {
                            'outer: for r in 0..pair.b_mat.nrows() {
                                for c in 0..pair.b_mat.ncols() {
                                    if !pair.b_mat[[r, c]].is_finite() {
                                        first_bad_b_mat = Some((r, c, pair.b_mat[[r, c]]));
                                        break 'outer;
                                    }
                                }
                            }
                        }
                        let mut first_bad_pair_g = None;
                        if !pair_g_finite {
                            for (idx, value) in pair.g.iter().enumerate() {
                                if !value.is_finite() {
                                    first_bad_pair_g = Some((idx, *value));
                                    break;
                                }
                            }
                        }
                        log::warn!(
                            "[OUTER ext-ext non-finite] ({},{}): cross_trace={} base={} m_terms={} correction={} pair.a={} pair.ld_s={} g.dot(v_jj)={} pair_g_finite={} first_bad_pair_g={:?} b_mat_finite={} first_bad_b_mat={:?} b_operator_present={} b_mat_dim={}x{} ext_v[ii]_finite={} ext_v[jj]_finite={} coord_i.b_depends_on_beta={} coord_j.b_depends_on_beta={}",
                            ii,
                            jj,
                            cross_trace,
                            base,
                            m_terms,
                            correction,
                            pair.a,
                            pair.ld_s,
                            g_dot_v,
                            pair_g_finite,
                            first_bad_pair_g,
                            b_mat_finite,
                            first_bad_b_mat,
                            pair.b_operator.is_some(),
                            pair.b_mat.nrows(),
                            pair.b_mat.ncols(),
                            ext_vi_finite,
                            ext_vj_finite,
                            coord_i.b_depends_on_beta,
                            coord_j.b_depends_on_beta,
                        );
                    }
                    (cross_trace, h2)
                } else {
                    (0.0, 0.0)
                };

                let h_val = outer_hessian_entry(
                    coord_i.a,
                    coord_j.a,
                    coord_i.g.dot(&ext_v[jj]),
                    pair.a,
                    cross_trace,
                    h2_trace,
                    pair.ld_s,
                    profiled_phi,
                    profiled_nu,
                    profiled_dp_cgrad,
                    profiled_dp_cgrad2,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                hess[[k + ii, k + jj]] = h_val;
                if ii != jj {
                    hess[[k + jj, k + ii]] = h_val;
                }
            }
        }
    }

    if hess.iter().any(|v| !v.is_finite()) {
        // NaN bisection: report which intermediate inputs were already
        // non-finite before the entry-builder summed them. This pinpoints the
        // original source (penalty drift, drift correction, cross-trace, ...)
        // instead of just flagging the final outer-Hessian entry.
        let report_finite = |name: &str, value: f64, ii: usize, jj: usize| {
            if !value.is_finite() {
                log::warn!(
                    "[OUTER non-finite] {} at ({}, {}) = {}",
                    name,
                    ii,
                    jj,
                    value,
                );
            }
        };
        for kk in 0..k {
            report_finite("rho_a_vals[kk]", rho_a_vals[kk], kk, kk);
            for entry in penalty_a_k_betas[kk].iter() {
                if !entry.is_finite() {
                    log::warn!(
                        "[OUTER non-finite] penalty_a_k_betas[{}] has non-finite",
                        kk
                    );
                    break;
                }
            }
            for entry in v_ks[kk].iter() {
                if !entry.is_finite() {
                    log::warn!("[OUTER non-finite] v_ks[{}] has non-finite", kk);
                    break;
                }
            }
        }
        if let Some(ref exact) = exact_logdet_cross_traces {
            for ii in 0..exact.nrows() {
                for jj in 0..exact.ncols() {
                    report_finite("exact_logdet_cross_traces", exact[[ii, jj]], ii, jj);
                }
            }
        }
        if let Some(ref sct) = stochastic_cross_traces {
            for ii in 0..sct.nrows() {
                for jj in 0..sct.ncols() {
                    report_finite("stochastic_cross_traces", sct[[ii, jj]], ii, jj);
                }
            }
        }
        if let Some(ref h_g) = leverage {
            for entry in h_g.iter() {
                if !entry.is_finite() {
                    log::warn!("[OUTER non-finite] leverage h^G has non-finite entries");
                    break;
                }
            }
        }
        if let Some(ref z_c) = adjoint_z_c {
            for entry in z_c.iter() {
                if !entry.is_finite() {
                    log::warn!("[OUTER non-finite] adjoint_z_c has non-finite entries");
                    break;
                }
            }
        }
        for ii in 0..total {
            for jj in 0..total {
                report_finite("hess", hess[[ii, jj]], ii, jj);
            }
        }
        return Err(
            "Outer Hessian contains non-finite entries; exact higher-order derivatives are invalid"
                .to_string(),
        );
    }

    Ok(hess)
}
