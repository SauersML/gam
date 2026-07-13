//! Derivative-trace computers shared by the dense and matrix-free outer-Hessian
//! paths: the scalar-GLM adjoint shortcut, fourth-derivative traces, the IFT
//! second-derivative correction, β-dependent drift-derivative traces, and the
//! base / cross logdet-Hessian traces (dense, dense-spectral, and stochastic).

use super::*;

//   C[u]            = Xᵀ diag(c ⊙ Xu) X
//   h^G             = diag(X G_ε(H) Xᵀ)
//   v               = Xᵀ (c ⊙ h^G)
//   z_c             = H⁻¹ v
//   tr(G_ε C[u])    = uᵀ Xᵀ (c ⊙ h^G) = uᵀ v
pub(crate) fn compute_adjoint_z_c(
    ing: &ScalarGlmIngredients<'_>,
    hop: &dyn HessianFactorization,
    leverage: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let mut weighted = Array1::<f64>::zeros(ing.c_array.len());
    Zip::from(&mut weighted)
        .and(ing.c_array)
        .and(leverage)
        .for_each(|w, &c, &h| *w = c * h);
    // Matrix-free Xᵀ · weighted via DesignMatrix transpose-apply, so
    // operator-backed (Lazy) designs at large scale never densify.
    let v = ing.x.transpose_vector_multiply(&weighted);
    // Adjoint shortcut for tr(Kernel · C[u]) where C[u] = Xᵀ diag(c ⊙ Xu) X
    // and u = β̈ is the SECOND mode response. Expanding the trace,
    //
    //     tr(Kernel · C[u]) = Σ_r c_r (Xu)_r (X Kernel Xᵀ)_rr
    //                       = uᵀ Xᵀ (c ⊙ h^G),   h^G = diag(X Kernel Xᵀ),
    //
    // and the mode response is `u = β̈ = H⁻¹ rhs` — solved with the FULL inner
    // inverse `hop.solve` in EVERY regime (it is an IFT stationarity derivative
    // living in full β-space; see `compute_ift_correction_trace`'s slow path and
    // `penalty_coordinate.rs`). Substituting,
    //
    //     tr(Kernel · C[u]) = rhsᵀ H⁻¹ Xᵀ(c ⊙ h^G) = rhsᵀ · z_c,
    //     z_c = H⁻¹ · Xᵀ(c ⊙ h^G).
    //
    // The PROJECTION enters ONLY through the leverage `h^G`, never through the
    // solve:
    //   * Full-Hessian regime: Kernel = G_ε(H), h^G = diag(X G_ε(H) Xᵀ).
    //   * Projected-subspace regime (rank-deficient LAML fix): Kernel = K, the
    //     caller passes h^{G,proj} = diag(X K Xᵀ); the solve stays `H⁻¹`.
    //
    // A previous version routed the projected regime through
    // `K · Xᵀ(c⊙h^{G,proj})` (the projected pseudo-inverse) instead of
    // `H⁻¹ · …`. That made `scalar_correction_trace` compute
    // `rhsᵀ K Xᵀ(c⊙h^{G,proj})` while the dense path's
    // `compute_ift_correction_trace` materialisation correctly traces
    // `tr(K · C[H⁻¹ rhs]) = rhsᵀ H⁻¹ Xᵀ(c⊙h^{G,proj})`, so the operator-form
    // Hessian disagreed with `compute_outer_hessian` on every non-trivial
    // subspace direction (the
    // `projected_operator_hessian_matches_dense_subspace_trace` /
    // `outer_hessian_operator_matvec_matches_dense_subspace_with_null_alpha`
    // regressions). The solve is `H⁻¹` regardless of regime; only the leverage
    // is projected.
    Ok(hop.solve(&v))
}

/// Compute the fourth-derivative trace: tr(G_ε(H) Xᵀ diag(d ⊙ (Xvₖ)(Xvₗ)) X).
///
/// Identity: tr(G_ε Xᵀ diag(w) X) = Σᵢ wᵢ · h^G[i].
/// Returns `None` if there are no fourth-derivative (d) terms.
pub(crate) fn compute_fourth_derivative_trace(
    ing: &ScalarGlmIngredients<'_>,
    v_k: &Array1<f64>,
    v_l: &Array1<f64>,
    leverage: &Array1<f64>,
) -> Result<Option<f64>, String> {
    let Some(d_array) = ing.d_array else {
        return Ok(None);
    };
    // Matrix-free X·v via DesignMatrix matvec; operator-backed (Lazy)
    // designs at large scale stream through their chunked kernels
    // instead of materializing the full (n×p) block.
    let x_vk = ing.x.matrixvectormultiply(v_k);
    let x_vl = ing.x.matrixvectormultiply(v_l);

    let mut acc = 0.0;
    Zip::from(d_array)
        .and(&x_vk)
        .and(&x_vl)
        .and(leverage)
        .for_each(|&d, &xvk, &xvl, &h| acc += d * xvk * xvl * h);
    Ok(Some(acc))
}

/// Compute every fourth-derivative trace for a coordinate set in one pass.
///
/// For scalar GLM Hessian corrections,
///
/// ```text
///   Q_ij = tr(G X' diag(d * (Xv_i) * (Xv_j)) X)
///        = Σ_r d_r h^G_r (Xv_i)_r (Xv_j)_r.
/// ```
///
/// This is a weighted Gram matrix of the row-space mode matrix `XV`.  Computing
/// it once replaces the per-pair `Xv_i` / `Xv_j` matvecs in
/// `compute_fourth_derivative_trace`, reducing the exact outer Hessian from
/// `O(T²)` design matvecs to `O(T)` design matvecs plus one `T×T` Gram.
pub(crate) fn compute_fourth_derivative_trace_matrix(
    ing: &ScalarGlmIngredients<'_>,
    modes: &[&Array1<f64>],
    leverage: &Array1<f64>,
) -> Result<Option<Array2<f64>>, String> {
    let Some(d_array) = ing.d_array else {
        return Ok(None);
    };
    let n = ing.c_array.len();
    let t = modes.len();
    if t == 0 {
        return Ok(Some(Array2::zeros((0, 0))));
    }
    if d_array.len() != n || leverage.len() != n {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "fourth-derivative trace shape mismatch: c={}, d={}, leverage={}",
                n,
                d_array.len(),
                leverage.len()
            ),
        }
        .into());
    }

    let mut x_modes = Array2::<f64>::zeros((n, t));
    for (j, mode) in modes.iter().enumerate() {
        let x_v = ing.x.matrixvectormultiply(mode);
        if x_v.len() != n {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "fourth-derivative trace Xv length mismatch for mode {j}: got {}, expected {n}",
                    x_v.len()
                ),
            }
            .into());
        }
        x_modes.column_mut(j).assign(&x_v);
    }

    let mut weighted = x_modes.clone();
    Zip::from(weighted.rows_mut())
        .and(d_array)
        .and(leverage)
        .for_each(|mut row, &d, &h| {
            let scale = d * h;
            row.mapv_inplace(|value| value * scale);
        });
    Ok(Some(gam_linalg::faer_ndarray::fast_atb(
        &x_modes, &weighted,
    )))
}

/// Compute the IFT second-derivative correction contribution to h2_trace.
///
/// This is the SINGLE implementation of the formula:
///
/// ```text
///   correction = tr(G_ε C[u_ij]) + tr(G_ε Q[v_i, v_j])
/// ```
///
/// where u_ij is the second implicit derivative RHS (already solved or
/// consumed via the adjoint shortcut), and v_i, v_j are the first-order
/// mode responses (positive convention: v = H⁻¹(g)).
///
/// When the adjoint z_c is available, uses the O(p) shortcut:
///   C_trace = rhs · z_c,  Q_trace = compute_fourth_derivative_trace(v_i, v_j)
///
/// Otherwise falls back to the O(p²) direct path:
///   u = H⁻¹(rhs),  correction = hessian_second_derivative_correction(v_i, v_j, u)
pub(crate) fn compute_ift_correction_trace(
    hop: &dyn HessianFactorization,
    rhs: &Array1<f64>,
    v_i: &Array1<f64>,
    v_j: &Array1<f64>,
    effective_deriv: &dyn HessianDerivativeProvider,
    adjoint_z_c: Option<&Array1<f64>>,
    glm_ingredients: Option<&ScalarGlmIngredients<'_>>,
    leverage: Option<&Array1<f64>>,
    precomputed_fourth_trace: Option<f64>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> Result<f64, String> {
    if !effective_deriv.has_corrections() {
        return Ok(0.0);
    }
    // The adjoint shortcut `tr(G_ε C[u]) = uᵀ z_c` is only valid for the
    // full-space kernel.  When the projected kernel is required, fall back
    // to materialising the correction and tracing through the subspace.
    if let (Some(z_c), None) = (adjoint_z_c, subspace) {
        let c_trace = rhs.dot(z_c);
        let d_trace = if let Some(trace) = precomputed_fourth_trace {
            trace
        } else {
            match (glm_ingredients, leverage) {
                (Some(ing), Some(h_g)) => {
                    compute_fourth_derivative_trace(ing, v_i, v_j, h_g)?.unwrap_or(0.0)
                }
                _ => 0.0,
            }
        };
        Ok(c_trace + d_trace)
    } else {
        // The second mode response `u = β̈ = H⁻¹ · rhs` is an IFT stationarity
        // derivative living in the FULL β-space, so it is solved with the full
        // inner inverse `hop.solve` even on the rank-deficient LAML path — the
        // same `ThetaModeResponseKernel` principle the FIRST mode response and
        // `dense.rs` follow (`penalty_coordinate.rs` doc: the IFT solve is full
        // H⁻¹ even when the logdet uses the projected trace). The
        // penalty-subspace projection acts ONLY on the trace contraction below
        // (`trace_projected_logdet`/`trace_operator`), never on the β̈ solve;
        // routing β̈ through the projected pseudo-inverse drops the `null(S₊)`
        // curvature component the projector discards.
        let u = hop.solve(rhs);
        if let Some(correction) =
            effective_deriv.hessian_second_derivative_correction_result(v_i, v_j, &u)?
        {
            if let Some(kernel) = subspace {
                // correction's DriftDerivResult materialises to a dense
                // matrix for the projected trace.
                match correction {
                    DriftDerivResult::Dense(matrix) => Ok(kernel.trace_projected_logdet(&matrix)),
                    DriftDerivResult::Operator(op) => Ok(kernel.trace_operator(op.as_ref())),
                }
            } else {
                Ok(correction.trace_logdet(hop))
            }
        } else {
            Ok(0.0)
        }
    }
}

/// Compute the β-dependent drift derivative traces: M_i[β_j] + M_j[β_i].
///
/// When a coordinate's fixed-β Hessian drift B depends on β, the second
/// Hessian drift Ḧ_{ij} includes additional terms D_β B_i[β_j] and
/// D_β B_j[β_i].  This function computes their traces through G_ε.
///
/// For ρ coordinates, B_k = A_k (penalty derivative) is β-independent, so
/// `b_depends_on_beta = false` and this returns 0.
pub(crate) fn compute_drift_deriv_traces(
    hop: &dyn HessianFactorization,
    b_i_depends: bool,
    b_j_depends: bool,
    ext_i: Option<usize>,
    ext_j: Option<usize>,
    beta_i: &Array1<f64>,
    beta_j: &Array1<f64>,
    fixed_drift_deriv: Option<&SharedFixedDriftDerivFn>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> Result<f64, String> {
    let trace_via = |result: DriftDerivResult| -> f64 {
        if let Some(kernel) = subspace {
            match result {
                DriftDerivResult::Dense(matrix) => kernel.trace_projected_logdet(&matrix),
                DriftDerivResult::Operator(op) => kernel.trace_operator(op.as_ref()),
            }
        } else {
            match result {
                DriftDerivResult::Dense(matrix) => hop.trace_logdet_gradient(&matrix),
                DriftDerivResult::Operator(op) => hop.trace_logdet_operator(op.as_ref()),
            }
        }
    };
    let mut trace = 0.0;
    // M_i[β_j] = D_β B_i[β_j]
    if b_i_depends && let (Some(ei), Some(drift_fn)) = (ext_i, fixed_drift_deriv) {
        if let Some(result) = drift_fn(ei, beta_j)? {
            trace += trace_via(result);
        }
    }
    // M_j[β_i] = D_β B_j[β_i]
    if b_j_depends && let (Some(ej), Some(drift_fn)) = (ext_j, fixed_drift_deriv) {
        if let Some(result) = drift_fn(ej, beta_i)? {
            trace += trace_via(result);
        }
    }
    Ok(trace)
}

/// Compute the base trace of the fixed-β second Hessian drift: tr(G_ε ∂²H/∂θ_i∂θ_j|_β).
///
/// Uses the operator-backed path when available, otherwise falls back to
/// dense matrix trace.  Returns 0 when neither is provided (e.g., ρ-ρ
/// off-diagonal where the fixed-β second drift is zero).
pub(crate) fn compute_base_h2_trace(
    hop: &dyn HessianFactorization,
    b_mat: &Array2<f64>,
    b_operator: Option<&dyn HyperOperator>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> f64 {
    if let Some(kernel) = subspace {
        if let Some(op) = b_operator {
            kernel.trace_operator(op)
        } else if b_mat.nrows() > 0 {
            kernel.trace_projected_logdet(b_mat)
        } else {
            0.0
        }
    } else if let Some(op) = b_operator {
        hop.trace_logdet_operator(op)
    } else if b_mat.nrows() > 0 {
        hop.trace_logdet_gradient(b_mat)
    } else {
        0.0
    }
}

pub(crate) fn compute_base_h2_traces(
    hop: &dyn HessianFactorization,
    pairs: &[&HyperCoordPair],
    subspace: Option<&PenaltySubspaceTrace>,
    trace_state: Option<Arc<Mutex<StochasticTraceState>>>,
) -> Vec<f64> {
    if pairs.is_empty() {
        return Vec::new();
    }
    if let Some(kernel) = subspace {
        let factor = penalty_subspace_trace_factor(kernel);
        let cache = ProjectedFactorCache::default();
        let mut out = vec![0.0_f64; pairs.len()];
        let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
        for (idx, pair) in pairs.iter().enumerate() {
            if let Some(op) = pair.b_operator.as_deref() {
                collect_projected_trace_terms(idx, 1.0, op, &factor, &mut out, &mut op_terms);
            } else if pair.b_mat.nrows() > 0 {
                out[idx] = dense_trace_projected_factor(&pair.b_mat, &factor);
            }
        }
        if !op_terms.is_empty() {
            let batched =
                trace_projected_operator_terms_batched(pairs.len(), &op_terms, &factor, &cache);
            for (idx, val) in batched.into_iter().enumerate() {
                out[idx] += val;
            }
        }
        return out;
    }
    // Dense-spectral batched path: collect every operator-backed pair into a
    // single chunked sweep so the implicit design (compute_xf + per-axis
    // kernel scalars) is traversed once instead of `pairs.len()` times. At
    // large scale this turns the 16+ per-call `trace_logdet_operator`
    // hot spots into a single batched evaluation.
    if subspace.is_none()
        && let Some(ds) = hop.as_exact_dense_spectral()
    {
        let mut out = vec![0.0_f64; pairs.len()];
        let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
        for (idx, pair) in pairs.iter().enumerate() {
            if let Some(op) = pair.b_operator.as_deref() {
                op_terms.push((idx, 1.0, op));
            } else if pair.b_mat.nrows() > 0 {
                out[idx] = hop.trace_logdet_gradient(&pair.b_mat);
            }
        }
        if !op_terms.is_empty() {
            let batched = trace_projected_operator_terms_batched(
                pairs.len(),
                &op_terms,
                &ds.g_factor,
                &ds.projected_factor_cache,
            );
            for (idx, val) in batched.into_iter().enumerate() {
                out[idx] += val;
            }
        }
        return out;
    }
    if subspace.is_none()
        && hop.prefers_stochastic_trace_estimation()
        && hop.logdet_traces_match_hinv_kernel()
    {
        let mut out = vec![0.0; pairs.len()];
        let mut dense_refs: Vec<&Array2<f64>> = Vec::new();
        let mut dense_slots = Vec::new();
        let mut op_refs: Vec<&dyn HyperOperator> = Vec::new();
        let mut op_slots = Vec::new();
        for (idx, pair) in pairs.iter().enumerate() {
            if let Some(op) = pair.b_operator.as_deref() {
                op_slots.push(idx);
                op_refs.push(op);
            } else if pair.b_mat.nrows() > 0 {
                dense_slots.push(idx);
                dense_refs.push(&pair.b_mat);
            }
        }
        if !dense_refs.is_empty() || !op_refs.is_empty() {
            let estimator = match trace_state {
                Some(state) => StochasticTraceEstimator::with_shared_trace_state(
                    StochasticTraceConfig::default(),
                    state,
                ),
                None => StochasticTraceEstimator::with_defaults(),
            };
            let values = estimator.estimate_traces_with_operators(hop, &dense_refs, &op_refs);
            for (local, &slot) in dense_slots.iter().enumerate() {
                out[slot] = values[local];
            }
            let offset = dense_refs.len();
            for (local, &slot) in op_slots.iter().enumerate() {
                out[slot] = values[offset + local];
            }
        }
        return out;
    }

    pairs
        .iter()
        .map(|pair| compute_base_h2_trace(hop, &pair.b_mat, pair.b_operator.as_deref(), subspace))
        .collect()
}

pub(crate) fn trace_logdet_hessian_cross_dense_drift(
    hop: &dyn HessianFactorization,
    dense: &Array2<f64>,
    drift: &DriftDerivResult,
) -> f64 {
    match drift {
        DriftDerivResult::Dense(matrix) => hop.trace_logdet_hessian_cross(dense, matrix),
        DriftDerivResult::Operator(operator) => {
            hop.trace_logdet_hessian_cross_matrix_operator(dense, operator.as_ref())
        }
    }
}

pub(crate) fn trace_logdet_hessian_crosses_dense_spectral_drifts(
    dense_hop: &DenseSpectralOperator,
    dense_drifts: &[Array2<f64>],
    ext_drifts: &[DriftDerivResult],
) -> Array2<f64> {
    let total = dense_drifts.len() + ext_drifts.len();
    let mut rotated = Vec::with_capacity(total);
    for matrix in dense_drifts {
        rotated.push(dense_hop.rotate_to_eigenbasis(matrix));
    }

    // Batch the projected_operator calls for implicit operator drifts so the
    // chunked design sweep (kernel scalars + GEMMs) is traversed once and
    // shared across all matching axes.
    let mut ext_rotated: Vec<Option<Array2<f64>>> = (0..ext_drifts.len()).map(|_| None).collect();
    let mut op_terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
    for (i, drift) in ext_drifts.iter().enumerate() {
        match drift {
            DriftDerivResult::Dense(matrix) => {
                ext_rotated[i] = Some(dense_hop.rotate_to_eigenbasis(matrix));
            }
            DriftDerivResult::Operator(operator) => {
                op_terms.push((i, 1.0, operator.as_ref()));
            }
        }
    }
    if !op_terms.is_empty() {
        let batched = projected_operator_terms_batched(
            ext_drifts.len(),
            &op_terms,
            &dense_hop.eigenvectors,
            &dense_hop.projected_factor_cache,
        );
        for (i, _, _) in &op_terms {
            ext_rotated[*i] = Some(batched[*i].clone());
        }
    }
    for r in ext_rotated {
        rotated.push(r.expect("every ext drift contributes a rotation"));
    }

    let mut out = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in i..total {
            let value = dense_hop.trace_logdet_hessian_cross_rotated(&rotated[i], &rotated[j]);
            out[[i, j]] = value;
            if i != j {
                out[[j, i]] = value;
            }
        }
    }
    out
}

#[inline]
pub(crate) fn can_use_stochastic_logdet_hinv_kernel(
    hop: &dyn HessianFactorization,
    total_p: usize,
    incl_logdet_h: bool,
) -> bool {
    total_p > STOCHASTIC_TRACE_DIM_THRESHOLD
        && hop.prefers_stochastic_trace_estimation()
        && hop.logdet_traces_match_hinv_kernel()
        && incl_logdet_h
}
