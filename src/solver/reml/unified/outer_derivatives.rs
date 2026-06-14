use super::*;
use crate::solver::estimate::smooth_floor_dp;

pub(crate) const HESSIAN_UNAVAILABLE_PREFIX: &str = "outer Hessian unavailable:";

/// Minimum coefficient dimension at which the matrix-free operator path is
/// selected unconditionally — once `p` is this large the dense `p × p`
/// assembly itself dominates and operator HVPs win regardless of `n` or `K`.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD: usize = 512;

/// Sample-count threshold for the (`n`, `p`) crossover branch: when `n` is
/// large enough that per-row work dominates, the operator path wins even
/// at moderate `p`.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD: usize = 50_000;

/// Coefficient dimension paired with [`MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD`]
/// in the (`n`, `p`) crossover branch.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N: usize = 32;

/// `n · p` linear-work cutoff: per-eval `O(K · n · p²)` dense assembly
/// dominates once `n · p` crosses this threshold even when both `n` and `p`
/// are individually below the per-axis thresholds.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD: usize = 4_000_000;

/// Smoothing-parameter count above which the operator path wins regardless
/// of `n` and `p`: the per-outer-eval Hessian-assembly cost is
/// `O(K · n · p²)`, so `K` itself drives the crossover.
pub(crate) const MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD: usize = 32;

/// Row-pair work cutoff for callback-backed outer Hessians.
///
/// Callback kernels expose exact row-local first/second Hessian drifts. Dense
/// `K x K` assembly can still be expensive at tiny coefficient dimension
/// because the dominant work is not `p x p` algebra; it is repeated row-kernel
/// contractions over the upper-triangular coordinate pairs.
pub(crate) const CALLBACK_OUTER_HESSIAN_ROW_PAIR_WORK_THRESHOLD: usize = 25_000_000;

/// Coefficient-dimension threshold above which a stochastic (Hutch++) trace
/// kernel is preferred over the exact dense trace for the logdet-H⁻¹ and ψ-Gram
/// paths. Below this the exact dense O(p³) work is cheap enough that the
/// estimator's variance is not worth trading for; above it the stochastic
/// estimator's O(p²·m) cost wins.
pub(crate) const STOCHASTIC_TRACE_DIM_THRESHOLD: usize = 500;

/// Elapsed-time (ms) above which a sparse-Cholesky trace path emits a timing
/// diagnostic. Purely observational — surfaces slow per-eval trace solves to the
/// bench runner without affecting the fit.
pub(crate) const REML_TRACE_SLOW_LOG_MS: f64 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OuterHessianRoutePlan {
    pub(crate) use_operator: bool,
    pub(crate) reason: &'static str,
    pub(crate) scale_prefers_operator: bool,
    pub(crate) dense_workspace_bytes: usize,
}

impl OuterHessianRoutePlan {
    pub(crate) fn choice(self) -> &'static str {
        if self.use_operator {
            "operator"
        } else {
            "dense"
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OuterHessianScaleDecision {
    prefers_operator: bool,
    pub(crate) reason: &'static str,
}

pub(crate) fn saturating_f64_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<f64>())
}

pub(crate) fn outer_hessian_dense_workspace_bytes(p: usize, k: usize) -> usize {
    // Dense assembly keeps first-order drifts for each coordinate and uses at
    // least one transient second-order drift while filling the K x K Hessian.
    // Charge a small safety multiple so the route never depends on fitting a
    // single p x p matrix while the actual dense path holds several.
    let drift_count = k.saturating_mul(2).saturating_add(3).max(1);
    saturating_f64_matrix_bytes(p, p).saturating_mul(drift_count)
}

pub(crate) fn outer_hessian_dense_workspace_budget_bytes() -> usize {
    crate::resource::ResourcePolicy::default_library().max_single_materialization_bytes
}

pub(crate) fn dense_outer_hessian_workspace_fits(p: usize, k: usize) -> bool {
    outer_hessian_dense_workspace_bytes(p, k) <= outer_hessian_dense_workspace_budget_bytes()
}

pub(crate) fn generic_outer_hessian_scale_decision(
    n: usize,
    p: usize,
    k: usize,
) -> OuterHessianScaleDecision {
    if !dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "dense_memory_budget",
        };
    }
    if k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_k",
        };
    }
    if p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_p",
        };
    }
    if n >= MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD
        && p >= MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N
    {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_n_moderate_p",
        };
    }
    if n.saturating_mul(p) >= MATRIX_FREE_OUTER_HESSIAN_NP_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_linear_work",
        };
    }
    OuterHessianScaleDecision {
        prefers_operator: false,
        reason: "below_crossover",
    }
}

pub(crate) fn callback_outer_hessian_scale_decision(
    n: usize,
    p: usize,
    k: usize,
) -> OuterHessianScaleDecision {
    if !dense_outer_hessian_workspace_fits(p, k) {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "dense_memory_budget",
        };
    }
    if k >= MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_k",
        };
    }
    if p >= MATRIX_FREE_OUTER_HESSIAN_DIM_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "large_p",
        };
    }
    if n.saturating_mul(k).saturating_mul(k) >= CALLBACK_OUTER_HESSIAN_ROW_PAIR_WORK_THRESHOLD {
        return OuterHessianScaleDecision {
            prefers_operator: true,
            reason: "callback_row_pair_work",
        };
    }
    OuterHessianScaleDecision {
        prefers_operator: false,
        reason: "below_crossover",
    }
}

pub(crate) fn outer_hessian_route_plan(
    n: usize,
    p: usize,
    k: usize,
    kernel_available: bool,
    callback_kernel: bool,
    subspace_trace: bool,
) -> OuterHessianRoutePlan {
    let dense_workspace_bytes = outer_hessian_dense_workspace_bytes(p, k);
    if !kernel_available {
        return OuterHessianRoutePlan {
            use_operator: false,
            reason: "kernel_absent",
            scale_prefers_operator: false,
            dense_workspace_bytes,
        };
    }

    let scale = if callback_kernel {
        callback_outer_hessian_scale_decision(n, p, k)
    } else {
        generic_outer_hessian_scale_decision(n, p, k)
    };
    let reason = if subspace_trace && scale.prefers_operator {
        "subspace_projected_operator"
    } else {
        scale.reason
    };
    OuterHessianRoutePlan {
        use_operator: scale.prefers_operator,
        reason,
        scale_prefers_operator: scale.prefers_operator,
        dense_workspace_bytes,
    }
}

/// Predicate for selecting the matrix-free Hv-operator outer-Hessian
/// representation over the dense `K × K` assembly.  Cost selects
/// representation, never capability — the operator path delivers the same
/// math as the dense path while avoiding large dense `p × p` drift storage
/// and pairwise row assembly when the model says those dominate.
pub(crate) fn prefer_outer_hessian_operator(n: usize, p: usize, k: usize) -> bool {
    generic_outer_hessian_scale_decision(n, p, k).prefers_operator
}

pub(crate) fn is_hessian_unavailable(error: &str) -> bool {
    error.starts_with(HESSIAN_UNAVAILABLE_PREFIX)
}

//   C[u]            = Xᵀ diag(c ⊙ Xu) X
//   h^G             = diag(X G_ε(H) Xᵀ)
//   v               = Xᵀ (c ⊙ h^G)
//   z_c             = H⁻¹ v
//   tr(G_ε C[u])    = uᵀ Xᵀ (c ⊙ h^G) = uᵀ v
pub(crate) fn compute_adjoint_z_c(
    ing: &ScalarGlmIngredients<'_>,
    hop: &dyn HessianOperator,
    leverage: &Array1<f64>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> Result<Array1<f64>, String> {
    let mut weighted = Array1::<f64>::zeros(ing.c_array.len());
    Zip::from(&mut weighted)
        .and(ing.c_array)
        .and(leverage)
        .for_each(|w, &c, &h| *w = c * h);
    // Matrix-free Xᵀ · weighted via DesignMatrix transpose-apply, so
    // operator-backed (Lazy) designs at large scale never densify.
    let v = ing.x.transpose_vector_multiply(&weighted);
    // Adjoint identity: tr(Kernel · C[u]) = uᵀ · Xᵀ(c ⊙ h^G), where the
    // kernel and the leverage `h^G` must come from the SAME operator.
    //
    //   * Full-Hessian regime: Kernel = G_ε(H), leverage = diag(X G_ε(H) Xᵀ),
    //     mode response u = H⁻¹ rhs, and the cheap shortcut reads
    //     `rhs · z_c` with z_c = H⁻¹ · X'(c⊙h^G).
    //   * Projected-subspace regime (rank-deficient LAML fix): Kernel = K,
    //     leverage = h^{G,proj} = diag(X K Xᵀ), mode response u = K · rhs
    //     (see `compute_ift_correction_trace`'s slow path and the standalone
    //     coord-v computation above). For the same shortcut to hold the
    //     adjoint must satisfy `rhs · z_c = (K rhs)ᵀ X'(c⊙h^{G,proj})
    //                                     = rhsᵀ K · X'(c⊙h^{G,proj})`,
    //     i.e. z_c^{proj} = K · X'(c ⊙ h^{G,proj}). Routing through
    //     `hop.solve` here produces H⁻¹ · X'(c ⊙ h^{G,proj}) instead and
    //     `scalar_correction_trace` then computes
    //     `rhsᵀ H⁻¹ X'(c⊙h^{G,proj})` while the dense path's
    //     `compute_ift_correction_trace` computes
    //     `rhsᵀ K · X'(c⊙h^{G,proj})`, so the operator-form Hessian
    //     disagrees with `compute_outer_hessian`'s materialisation on every
    //     non-trivial subspace direction (the
    //     `projected_operator_hessian_matches_dense_subspace_trace`
    //     regression).
    match subspace {
        Some(kernel) => Ok(kernel.apply_pseudo_inverse(&v)),
        None => Ok(hop.solve(&v)),
    }
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
    Ok(Some(crate::faer_ndarray::fast_atb(&x_modes, &weighted)))
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
    hop: &dyn HessianOperator,
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
        // WS1a fallback: projected kernel for rank-deficient LAML path.
        let u = if let Some(kernel) = subspace {
            kernel.apply_pseudo_inverse(rhs)
        } else {
            hop.solve(rhs)
        };
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
    hop: &dyn HessianOperator,
    b_i_depends: bool,
    b_j_depends: bool,
    ext_i: Option<usize>,
    ext_j: Option<usize>,
    beta_i: &Array1<f64>,
    beta_j: &Array1<f64>,
    fixed_drift_deriv: Option<&FixedDriftDerivFn>,
    subspace: Option<&PenaltySubspaceTrace>,
) -> f64 {
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
    if b_i_depends
        && let (Some(ei), Some(drift_fn)) = (ext_i, fixed_drift_deriv)
        && let Some(result) = drift_fn(ei, beta_j)
    {
        trace += trace_via(result);
    }
    // M_j[β_i] = D_β B_j[β_i]
    if b_j_depends
        && let (Some(ej), Some(drift_fn)) = (ext_j, fixed_drift_deriv)
        && let Some(result) = drift_fn(ej, beta_i)
    {
        trace += trace_via(result);
    }
    trace
}

/// Compute the base trace of the fixed-β second Hessian drift: tr(G_ε ∂²H/∂θ_i∂θ_j|_β).
///
/// Uses the operator-backed path when available, otherwise falls back to
/// dense matrix trace.  Returns 0 when neither is provided (e.g., ρ-ρ
/// off-diagonal where the fixed-β second drift is zero).
pub(crate) fn compute_base_h2_trace(
    hop: &dyn HessianOperator,
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
    hop: &dyn HessianOperator,
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
    hop: &dyn HessianOperator,
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
    hop: &dyn HessianOperator,
    total_p: usize,
    incl_logdet_h: bool,
) -> bool {
    total_p > STOCHASTIC_TRACE_DIM_THRESHOLD
        && hop.prefers_stochastic_trace_estimation()
        && hop.logdet_traces_match_hinv_kernel()
        && incl_logdet_h
}

/// Shared precomputed REML derivative intermediates threaded from the
/// gradient pass into the dense Hessian assembler so the per-coordinate
/// `penalty_a_k_beta` / `hop.solve` / drift-correction work is not repeated.
pub(crate) struct RemlDerivativeWorkspace<'a> {
    pub curvature_lambdas: &'a [f64],
    pub rho_penalty_a_k_betas: &'a [Array1<f64>],
    pub rho_curvature_a_k_betas: &'a [Array1<f64>],
    pub rho_v_ks: Option<&'a [Array1<f64>]>,
    pub ext_v_is: Option<&'a [Array1<f64>]>,
    pub coord_corrections: &'a [Option<DriftDerivResult>],
}

pub(crate) struct KktRhoCorrections {
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Option<Array2<f64>>,
}

pub(crate) fn solve_kkt_residual_kernel(
    hop: &dyn HessianOperator,
    subspace: Option<&PenaltySubspaceTrace>,
    rhs: &Array1<f64>,
) -> Array1<f64> {
    if let Some(kernel) = subspace {
        let projected = crate::faer_ndarray::fast_atv(&kernel.u_s, rhs);
        let solved_projected = kernel.h_proj_inverse.dot(&projected);
        crate::faer_ndarray::fast_av(&kernel.u_s, &solved_projected)
    } else {
        hop.solve(rhs)
    }
}

pub(crate) fn active_upper_rho_mask(rho: &[f64]) -> Vec<bool> {
    let latest_theta = super::super::runtime::latest_outer_theta_for_ift();
    let matching_outer_theta = latest_theta.as_ref().is_some_and(|theta| {
        theta.len() >= rho.len()
            && theta
                .iter()
                .take(rho.len())
                .zip(rho.iter())
                .all(|(&recorded, &current)| recorded.to_bits() == current.to_bits())
    });
    let upper_bounds = matching_outer_theta
        .then(super::super::runtime::latest_outer_rho_upper_bounds_for_ift)
        .flatten();
    rho.iter()
        .enumerate()
        .map(|(idx, &value)| {
            let upper = upper_bounds
                .as_ref()
                .and_then(|bounds| bounds.get(idx))
                .copied()
                .unwrap_or(crate::solver::estimate::RHO_BOUND);
            upper.is_finite() && value >= upper - 1.0e-8
        })
        .collect()
}

/// Derivatives of the same Newton/IFT residual correction used by the cost:
///
///   C(ρ) = -½ r(ρ)^T K(ρ) r(ρ),   K = H^{-1}
///
/// for fixed-dispersion LAML. At fixed β̂, `r_i = A_i β̂` and
/// `H_i = A_i`, so
///
///   C_i  = -a_i^T q + ½ q^T A_i q,
///   q    = K r,
///   q_j  = K(a_j - A_j q),
///   C_ij = -δ_ij a_i^T q - a_i^T q_j
///          + q_j^T A_i q + ½δ_ij q^T A_i q.
///
/// The dense outer Hessian already contains the exact-KKT profile term
/// `-a_i^T K a_j`. That term is valid only when `r = 0`; the residual
/// correction is therefore added as `a_i^T K a_j + C_ij`. This guarantees
/// the additive block vanishes at exact KKT and is exact for the Gaussian
/// quadratic reproduction.
pub(crate) fn compute_kkt_residual_rho_corrections(
    solution: &InnerSolution<'_>,
    hop: &dyn HessianOperator,
    lambdas: &[f64],
    penalty_a_k_betas: &[Array1<f64>],
    residual: &Array1<f64>,
    include_hessian: bool,
    upper_active_rho: &[bool],
) -> Result<KktRhoCorrections, String> {
    let k = penalty_a_k_betas.len();
    if k == 0 {
        return Ok(KktRhoCorrections {
            gradient: Array1::zeros(0),
            hessian: include_hessian.then(|| Array2::zeros((0, 0))),
        });
    }
    if lambdas.len() != k || solution.penalty_coords.len() != k {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT rho correction dimension mismatch: lambdas={} coords={} rhs={}",
                lambdas.len(),
                solution.penalty_coords.len(),
                k
            ),
        }
        .into());
    }
    if upper_active_rho.len() != k {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT rho correction active-bound mask mismatch: mask={} rhs={}",
                upper_active_rho.len(),
                k
            ),
        }
        .into());
    }
    if residual.len() != hop.dim() {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "KKT residual dimension mismatch: residual={} Hessian dim={}",
                residual.len(),
                hop.dim()
            ),
        }
        .into());
    }

    let subspace = solution.penalty_subspace_trace.as_deref();
    let q = solve_kkt_residual_kernel(hop, subspace, residual);
    let mut a_i_qs = Vec::with_capacity(k);
    let mut a_i_dot_q = Vec::with_capacity(k);
    let mut q_a_i_q = Vec::with_capacity(k);

    for idx in 0..k {
        if upper_active_rho[idx] {
            a_i_dot_q.push(0.0);
            q_a_i_q.push(0.0);
            a_i_qs.push(Array1::<f64>::zeros(hop.dim()));
            continue;
        }
        let a_i_q = solution.penalty_coords[idx].scaled_matvec(&q, lambdas[idx]);
        let linear = penalty_a_k_betas[idx].dot(&q);
        let quadratic = q.dot(&a_i_q);
        if !linear.is_finite() || !quadratic.is_finite() {
            return Err(RemlError::NonFiniteValue {
                reason: format!(
                    "KKT rho correction produced non-finite gradient ingredients at coord {idx}: \
                     linear={linear} quadratic={quadratic}"
                ),
            }
            .into());
        }
        a_i_dot_q.push(linear);
        q_a_i_q.push(quadratic);
        a_i_qs.push(a_i_q);
    }

    let mut gradient = Array1::<f64>::zeros(k);
    for idx in 0..k {
        if !upper_active_rho[idx] {
            gradient[idx] = -a_i_dot_q[idx] + 0.5 * q_a_i_q[idx];
        }
    }

    let hessian = if include_hessian {
        let mut a_solutions = Vec::with_capacity(k);
        let mut q_derivs = Vec::with_capacity(k);
        for idx in 0..k {
            if upper_active_rho[idx] {
                a_solutions.push(Array1::<f64>::zeros(hop.dim()));
                q_derivs.push(Array1::<f64>::zeros(hop.dim()));
                continue;
            }
            a_solutions.push(solve_kkt_residual_kernel(
                hop,
                subspace,
                &penalty_a_k_betas[idx],
            ));
            let mut rhs = penalty_a_k_betas[idx].clone();
            rhs -= &a_i_qs[idx];
            q_derivs.push(solve_kkt_residual_kernel(hop, subspace, &rhs));
        }

        let entry = |i: usize, j: usize| -> f64 {
            if upper_active_rho[i] || upper_active_rho[j] {
                return 0.0;
            }
            let delta = if i == j { 1.0 } else { 0.0 };
            let cancel_exact_kkt_profile_term = penalty_a_k_betas[i].dot(&a_solutions[j]);
            cancel_exact_kkt_profile_term
                - delta * a_i_dot_q[i]
                - penalty_a_k_betas[i].dot(&q_derivs[j])
                + q_derivs[j].dot(&a_i_qs[i])
                + 0.5 * delta * q_a_i_q[i]
        };

        let mut h = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in i..k {
                let raw = if i == j {
                    entry(i, j)
                } else {
                    0.5 * (entry(i, j) + entry(j, i))
                };
                if !raw.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "KKT rho correction produced non-finite Hessian entry ({i}, {j}): {raw}"
                        ),
                    }
                    .into());
                }
                h[[i, j]] = raw;
                if i != j {
                    h[[j, i]] = raw;
                }
            }
        }
        Some(h)
    } else {
        None
    };

    Ok(KktRhoCorrections { gradient, hessian })
}

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
                let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw);
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

pub(crate) struct StoredFirstDrift {
    pub(crate) dense: Option<Array2<f64>>,
    dense_rotated: Option<Array2<f64>>,
    pub(crate) operators: Vec<Arc<dyn HyperOperator>>,
}

impl StoredFirstDrift {
    pub(crate) fn from_parts(
        dense: Option<Array2<f64>>,
        dense_rotated: Option<Array2<f64>>,
        operators: Vec<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense,
            dense_rotated,
            operators,
        }
    }

    pub(crate) fn scaled_add_apply(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        out: &mut Array1<f64>,
    ) {
        assert_eq!(v.len(), out.len());
        if scale == 0.0 {
            return;
        }
        if let Some(matrix) = self.dense.as_ref() {
            dense_matvec_scaled_add_into(matrix, v, scale, out.view_mut());
        }
        if !self.operators.is_empty() {
            for op in &self.operators {
                op.scaled_add_mul_vec(v, scale, out.view_mut());
            }
        }
    }

    pub(crate) fn apply_dot(&self, v: ArrayView1<'_, f64>, test: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), test.len());
        let mut total = 0.0;
        if let Some(matrix) = self.dense.as_ref() {
            total += dense_bilinear(matrix, v, test);
        }
        for op in &self.operators {
            total += op.bilinear_view(v, test);
        }
        total
    }
}

pub(crate) struct BorrowedStoredDriftOperator<'a> {
    pub(crate) drift: &'a StoredFirstDrift,
    pub(crate) dim_hint: usize,
}

impl HyperOperator for BorrowedStoredDriftOperator<'_> {
    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.fill(0.0);
        if let Some(matrix) = self.drift.dense.as_ref() {
            dense_matvec_into(matrix, v, out.view_mut());
        }
        for op in &self.drift.operators {
            op.scaled_add_mul_vec(v, 1.0, out.view_mut());
        }
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let mut out = out;
        if let Some(matrix) = self.drift.dense.as_ref() {
            dense_matvec_scaled_add_into(matrix, v, scale, out.view_mut());
        }
        for op in &self.drift.operators {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.drift.apply_dot(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        self.drift.apply_dot(v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .drift
            .dense
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.dim_hint, self.dim_hint)));
        for op in &self.drift.operators {
            out += &op.to_dense();
        }
        out
    }

    fn is_implicit(&self) -> bool {
        !self.drift.operators.is_empty()
    }
}

/// Linear combination of `HyperOperator` factors with explicit scalar
/// weights. Used to bundle a coord's per-mode drift operators (or any other
/// per-term linear combination) into a single matrix-free operator that
/// implements the same `HyperOperator` trait, so callers downstream do not
/// need to handle a vector of (weight, op) pairs themselves.
pub struct WeightedHyperOperator {
    pub(crate) terms: Vec<(f64, Arc<dyn HyperOperator>)>,
    pub(crate) dim_hint: usize,
}

impl HyperOperator for WeightedHyperOperator {
    fn as_weighted(&self) -> Option<&WeightedHyperOperator> {
        Some(self)
    }

    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        let mut nonzero_terms = self.terms.iter().filter(|(weight, _)| *weight != 0.0);
        if let Some((weight, op)) = nonzero_terms.next()
            && nonzero_terms.next().is_none()
        {
            op.mul_vec_into(v, out.view_mut());
            if *weight != 1.0 {
                out.mapv_inplace(|value| *weight * value);
            }
            return;
        }

        out.fill(0.0);
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                op.scaled_add_mul_vec(v, *weight, out.view_mut());
            }
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let mut nonzero_terms = self.terms.iter().filter(|(weight, _)| *weight != 0.0);
        if let Some((weight, op)) = nonzero_terms.next()
            && nonzero_terms.next().is_none()
        {
            op.mul_basis_columns_into(start, out.view_mut());
            if *weight != 1.0 {
                out.mapv_inplace(|value| *weight * value);
            }
            return;
        }

        out.fill(0.0);
        let mut work = Array2::<f64>::zeros((out.nrows(), out.ncols()));
        for (weight, op) in &self.terms {
            if *weight == 0.0 {
                continue;
            }
            op.mul_basis_columns_into(start, work.view_mut());
            out.scaled_add(*weight, &work);
        }
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        for (weight, op) in &self.terms {
            let combined = scale * *weight;
            if combined != 0.0 {
                op.scaled_add_mul_vec(v, combined, out.view_mut());
            }
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear(v, u))
            .sum()
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear_view(v, u))
            .sum()
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.trace_projected_factor(factor))
            .sum()
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.trace_projected_factor_cached(factor, cache))
            .sum()
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                projected.scaled_add(*weight, &op.projected_matrix(factor));
            }
        }
        projected
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                projected.scaled_add(*weight, &op.projected_matrix_cached(factor, cache));
            }
        }
        projected
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim_hint, self.dim_hint));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                out.scaled_add(*weight, &op.to_dense());
            }
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.terms.iter().any(|(_, op)| op.is_implicit())
    }
}

/// Per-matvec contraction of the ψψ-block second-order hook (#740), with each
/// `D²_ψ H_L` drift already traced through the logdet kernel into `base_h2`.
/// Indexed by ψ output row `i = idx - k_rho`.
pub(crate) struct PsiContractedContrib {
    objective: Array1<f64>,
    score: Array2<f64>,
    pub(crate) ld_s: Array1<f64>,
    base_h2: Vec<f64>,
}

pub(crate) struct OuterHessianCoord {
    pub(crate) a: f64,
    pub(crate) g: Array1<f64>,
    v: Array1<f64>,
    total_drift: StoredFirstDrift,
    base_drift: StoredFirstDrift,
    ext_index: Option<usize>,
    b_depends_on_beta: bool,
}

impl OuterHessianCoord {
    pub(crate) fn is_ext(&self) -> bool {
        self.ext_index.is_some()
    }
}

pub(crate) struct UnifiedOuterHessianOperator {
    pub(crate) hop: Arc<dyn HessianOperator>,
    coords: Vec<OuterHessianCoord>,
    pair_a: Array2<f64>,
    pair_ld_s: Array2<f64>,
    g_dot_v: Array2<f64>,
    pair_g: Vec<Vec<Option<Array1<f64>>>>,
    base_h2: Array2<f64>,
    m_pair_trace: Array2<f64>,
    /// Precomputed pair-wise logdet-Hessian cross traces.
    /// `cross_trace[i, j] = tr(G_ε(H) Ḣ_i Ḣ_j)` decomposed across the
    /// dense and operator components of each coord's `total_drift`.
    /// Populated only when `incl_logdet_h`.  matvec recovers the alpha-combo
    /// trace as `cross_trace.row(idx).dot(alpha)`, replacing the per-HVP
    /// recomputation that previously rebuilt these traces every time the
    /// K×K outer Hessian was materialized via K matvecs.
    cross_trace: Option<Array2<f64>>,
    profiled_phi: f64,
    profiled_nu: f64,
    profiled_dp_cgrad: f64,
    profiled_dp_cgrad2: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
    pub(crate) kernel: OuterHessianDerivativeKernel,
    pub(crate) subspace: Option<Arc<PenaltySubspaceTrace>>,
    adjoint_z_c: Option<Array1<f64>>,
    leverage: Option<Array1<f64>>,
    fourth_trace: Option<Array2<f64>>,
    callback_second_modes: Option<Vec<Array1<f64>>>,
    /// Number of ρ (penalty) coordinates; coords `k_rho..` are the ψ rows. Used
    /// only when `contracted_psi` is present, to map an output index to its ψ
    /// output row.
    k_rho: usize,
    /// Direction-contracted ψψ second-order hook (#740). When `Some`, the build
    /// SKIPPED the `K²` per-pair `ext_coord_pair_fn` ψψ assembly (the `pair_a` /
    /// `pair_ld_s` / `base_h2` ψψ-block entries and the ψψ `pair_g` are left
    /// zero / `None`), and `matvec`/`apply_into` apply this once per call to add
    /// the ψ-row ψψ contributions in a single family row pass. The ρρ and ρψ
    /// blocks remain in the precomputed tables (cheap), so this changes only the
    /// representation of the ψψ block, not the math.
    contracted_psi: Option<ContractedPsiSecondOrderFn>,
}

impl UnifiedOuterHessianOperator {
    /// Exact implicit-function-theorem mode response of the inner coefficient
    /// solution along a θ-direction `α = (α_ρ, α_ψ)` (#740 primitive).
    ///
    /// At the inner optimum `g(β̂(θ), θ) = ∇_β F = 0`, differentiating gives
    /// `β̇_j = −H⁻¹ ∂g/∂θ_j` for each coordinate j; the response along a θ
    /// direction is the linear combination `β̇(α) = Σ_j α_j β̇_j`. Each
    /// per-coordinate `coord.v = H⁻¹ a_j` is precomputed exactly via the shared
    /// inner mode inverse `hop.solve_multi` (see `build_outer_hessian_operator`),
    /// so this combination is the EXACT directional `β̇(α)` with no
    /// finite-difference or low-rank approximation — the same object the profiled
    /// θ-HVP needs as `β̇ = −H⁻¹ ∂g/∂θ·v`. Extended (ψ) coordinates carry the
    /// opposite drift sign, matching the `b_depends_on_beta` convention.
    ///
    /// This is the reusable matrix-free primitive an O(K)-build θ-HVP matvec is
    /// organized around (one IFT solve per applied direction instead of the K²
    /// coordinate-pair assembly). The current `matvec` still consumes the
    /// precomputed pair traces; this primitive is the exact directional β̇ those
    /// pair traces are a (K²-amortized) re-expression of, exposed so the
    /// pair-assembly precompute can be replaced incrementally without changing
    /// the IFT mathematics.
    pub(crate) fn theta_direction_mode_response(&self, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for (j, coord) in self.coords.iter().enumerate() {
            if alpha[j] == 0.0 {
                continue;
            }
            if coord.is_ext() {
                out.scaled_add(-alpha[j], &coord.v);
            } else {
                out.scaled_add(alpha[j], &coord.v);
            }
        }
        out
    }

    pub(crate) fn pair_rhs_dot(&self, row: usize, col: usize, test: ArrayView1<'_, f64>) -> f64 {
        let row_coord = &self.coords[row];
        let col_coord = &self.coords[col];
        let pair_g_dot = self.pair_g[row][col]
            .as_ref()
            .map(|pair_g| pair_g.dot(&test))
            .unwrap_or(0.0);

        col_coord.total_drift.apply_dot(row_coord.v.view(), test)
            + row_coord.base_drift.apply_dot(col_coord.v.view(), test)
            - pair_g_dot
    }

    pub(crate) fn scaled_add_pair_rhs(
        &self,
        row: usize,
        col: usize,
        scale: f64,
        out: &mut Array1<f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let row_coord = &self.coords[row];
        let col_coord = &self.coords[col];
        col_coord
            .total_drift
            .scaled_add_apply(row_coord.v.view(), scale, out);
        row_coord
            .base_drift
            .scaled_add_apply(col_coord.v.view(), scale, out);
        if let Some(pair_g) = self.pair_g[row][col].as_ref() {
            out.scaled_add(-scale, pair_g);
        }
    }

    pub(crate) fn pair_rhs_combo(&self, idx: usize, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for j in 0..alpha.len() {
            if alpha[j] != 0.0 {
                self.scaled_add_pair_rhs(idx, j, alpha[j], &mut out);
            }
        }
        out
    }

    pub(crate) fn scalar_correction_trace(
        &self,
        idx: usize,
        alpha: &Array1<f64>,
        v_i: &Array1<f64>,
        m_alpha: &Array1<f64>,
        psi_score_alpha: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let OuterHessianDerivativeKernel::ScalarGlm {
            c_array,
            d_array,
            x,
        } = &self.kernel
        else {
            return Err(RemlError::InvalidKernelMode {
                reason: "scalar correction requested for non-scalar kernel".to_string(),
            }
            .into());
        };

        // Cheap adjoint shortcut: works for both full-Hessian and projected
        // (subspace) regimes because §10 populates `leverage`/`adjoint_z_c`
        // with the projected `h^{G,proj}` and `K · v` under subspace, and
        // the identity tr(Kernel · C[u]) = uᵀ Xᵀ(c ⊙ h^G) carries through.
        let z_c = self.adjoint_z_c.as_ref().ok_or_else(|| {
            "missing adjoint trace cache for scalar outer Hessian operator".to_string()
        })?;
        let ingredients = ScalarGlmIngredients {
            c_array,
            d_array: d_array.as_ref(),
            x,
        };
        let h_g = self.leverage.as_ref().ok_or_else(|| {
            "missing leverage cache for scalar outer Hessian operator".to_string()
        })?;
        let mut c_trace = 0.0;
        for (j, &alpha_j) in alpha.iter().enumerate() {
            if alpha_j == 0.0 {
                continue;
            }
            c_trace += alpha_j * self.pair_rhs_dot(idx, j, z_c.view());
        }
        // #740: `pair_rhs_dot` reads `pair_g[idx][j]` for the `−g_{ij}·z_c`
        // adjoint term, but the build SKIPPED the ψψ `pair_g` when the contracted
        // hook is installed. `pair_rhs_dot` therefore drops the `−Σ_j α_j
        // g_{ψ_i ψ_j}` ψψ contribution; the hook supplies it as `score.row(i)`,
        // so add the missing `−score·z_c` here (mirrors the `−score` rhs
        // injection on the Callback path in `outer_hessian_index_entry`).
        if let Some(score_alpha) = psi_score_alpha {
            c_trace -= score_alpha.dot(z_c);
        }
        let d_trace = if let Some(trace) = self.fourth_trace.as_ref() {
            let mut combo = 0.0;
            for (j, &alpha_j) in alpha.iter().enumerate() {
                if alpha_j != 0.0 {
                    combo += alpha_j * trace[[idx, j]];
                }
            }
            combo
        } else {
            compute_fourth_derivative_trace(&ingredients, v_i, m_alpha, h_g)?.unwrap_or(0.0)
        };
        Ok(c_trace + d_trace)
    }

    pub(crate) fn callback_correction_trace(
        &self,
        rhs: &Array1<f64>,
        second_v: &Array1<f64>,
        neg_m_alpha: &Array1<f64>,
    ) -> Result<f64, String> {
        let OuterHessianDerivativeKernel::Callback { first, second } = &self.kernel else {
            return Err(RemlError::InvalidKernelMode {
                reason: "callback correction requested for non-callback kernel".to_string(),
            }
            .into());
        };
        let u = match self.subspace.as_deref() {
            Some(subspace) => subspace.apply_pseudo_inverse(rhs),
            None => self.hop.solve(rhs),
        };
        let Some(term1) = first(&u)? else {
            return Ok(0.0);
        };
        let Some(term2) = second(neg_m_alpha, second_v)? else {
            return Ok(0.0);
        };
        let combined = CompositeHyperOperator {
            dense: None,
            operators: vec![term1.into_operator(), term2.into_operator()],
            dim_hint: self.hop.dim(),
        };
        if let Some(subspace) = self.subspace.as_deref() {
            Ok(subspace.trace_operator(&combined))
        } else {
            Ok(self.hop.trace_logdet_operator(&combined))
        }
    }

    /// Per-call contraction of the ψψ-block second-order hook (#740).
    ///
    /// Calls `contracted_psi` once with the ψ slice of `alpha` and pre-traces
    /// each per-output-row `D²_ψ H_L[ψ_i, ψ(α)]` drift through the logdet kernel
    /// so `outer_hessian_index_entry` reads scalars. Returns `None` when no hook
    /// is installed (the ψψ block then lives entirely in the precomputed
    /// tables). The `score` rows are carried through unchanged for injection
    /// into the callback-correction rhs (they replace the ψψ `pair_g` the build
    /// skipped). Indexed by ψ output row `i = idx - k_rho`.
    pub(crate) fn psi_contracted_contrib(
        &self,
        alpha: &Array1<f64>,
    ) -> Result<Option<PsiContractedContrib>, String> {
        let Some(hook) = self.contracted_psi.as_ref() else {
            return Ok(None);
        };
        let alpha_psi: Vec<f64> = alpha.iter().skip(self.k_rho).copied().collect();
        let Some(contracted) = hook(&alpha_psi)? else {
            // The hook declined this direction (e.g. a σ-aux axis carried
            // weight): this operator must not have been built with a skipped
            // ψψ assembly, so a decline here is a contract violation.
            return Err(RemlError::InvalidKernelMode {
                reason: "contracted ψψ hook declined a direction after the outer-Hessian \
                         build skipped per-pair ψψ assembly; the build-time and apply-time \
                         hook availability disagree"
                    .to_string(),
            }
            .into());
        };
        let base_h2: Vec<f64> = contracted
            .hessian
            .iter()
            .map(|drift| match (self.subspace.as_deref(), drift) {
                (Some(kernel), DriftDerivResult::Dense(m)) => kernel.trace_projected_logdet(m),
                (Some(kernel), DriftDerivResult::Operator(op)) => {
                    kernel.trace_operator(op.as_ref())
                }
                (None, DriftDerivResult::Dense(m)) => self.hop.trace_logdet_gradient(m),
                (None, DriftDerivResult::Operator(op)) => {
                    self.hop.trace_logdet_operator(op.as_ref())
                }
            })
            .collect();
        Ok(Some(PsiContractedContrib {
            objective: contracted.objective,
            score: contracted.score,
            ld_s: contracted.ld_s,
            base_h2,
        }))
    }

    /// Per-coordinate outer-Hessian-row × `alpha` contraction shared by the
    /// `matvec` and zero-alloc `apply_into` paths. `a_alpha`,
    /// `correction_m_alpha`, and `callback_neg_m_alpha` are the
    /// alpha-dependent quantities precomputed once per call by the caller.
    /// `psi_contrib` carries the per-call ψψ-block hook contraction (#740);
    /// `None` keeps the ψψ block in the precomputed tables.
    pub(crate) fn outer_hessian_index_entry(
        &self,
        idx: usize,
        alpha: &Array1<f64>,
        a_alpha: f64,
        correction_m_alpha: &Array1<f64>,
        callback_neg_m_alpha: Option<&Array1<f64>>,
        psi_contrib: Option<&PsiContractedContrib>,
    ) -> Result<f64, String> {
        let coord = &self.coords[idx];
        // ψ output row index into the hook contraction (when this idx is a ψ
        // coordinate and the hook is active); `None` for ρ rows or no hook.
        let psi_row = psi_contrib
            .and_then(|contrib| (idx >= self.k_rho).then(|| (contrib, idx - self.k_rho)));
        let mut pair_a = self.pair_a.row(idx).dot(alpha);
        let mut pair_ld_s = self.pair_ld_s.row(idx).dot(alpha);
        let g_dot_v_alpha = self.g_dot_v.row(idx).dot(alpha);
        let mut base_h2 = self.base_h2.row(idx).dot(alpha);
        let m_terms = self.m_pair_trace.row(idx).dot(alpha);
        if let Some((contrib, i)) = psi_row {
            // The build skipped the ψψ-block entries of these tables; add the
            // hook's α-contraction of the ψψ block (likelihood + ψψ penalty).
            pair_a += contrib.objective[i];
            pair_ld_s += contrib.ld_s[i];
            base_h2 += contrib.base_h2[i];
        }

        let cross_trace = match self.cross_trace.as_ref() {
            Some(ct) => ct.row(idx).dot(alpha),
            None => 0.0,
        };

        let correction = if self.incl_logdet_h {
            match &self.kernel {
                OuterHessianDerivativeKernel::Gaussian => 0.0,
                OuterHessianDerivativeKernel::ScalarGlm { .. } => {
                    // For ψ rows with the contracted hook, supply the
                    // `−Σ_j α_j g_{ψ_i ψ_j}` (= score.row(i)) that the skipped ψψ
                    // `pair_g` no longer provides to the scalar adjoint trace.
                    let psi_score = psi_row.map(|(contrib, i)| contrib.score.row(i).to_owned());
                    self.scalar_correction_trace(
                        idx,
                        alpha,
                        &coord.v,
                        correction_m_alpha,
                        psi_score.as_ref(),
                    )?
                }
                OuterHessianDerivativeKernel::Callback { .. } => {
                    let second_v = &self
                        .callback_second_modes
                        .as_ref()
                        .expect("callback second modes")[idx];
                    let mut rhs = self.pair_rhs_combo(idx, alpha);
                    // The build skipped the ψψ `pair_g`; the callback-correction
                    // second mode-response rhs needs `−Σ_j α_j g_{ψ_i ψ_j}`,
                    // which the hook supplies as `score.row(i)`. Inject it so the
                    // rhs matches the dense path's `pair_rhs_combo` exactly.
                    if let Some((contrib, i)) = psi_row {
                        rhs.scaled_add(-1.0, &contrib.score.row(i));
                    }
                    self.callback_correction_trace(
                        &rhs,
                        second_v,
                        callback_neg_m_alpha.expect("callback negated mode"),
                    )?
                }
            }
        } else {
            0.0
        };

        Ok(outer_hessian_entry(
            coord.a,
            a_alpha,
            g_dot_v_alpha,
            pair_a,
            cross_trace,
            base_h2 + m_terms + correction,
            pair_ld_s,
            self.profiled_phi,
            self.profiled_nu,
            self.profiled_dp_cgrad,
            self.profiled_dp_cgrad2,
            self.is_profiled,
            self.incl_logdet_h,
            self.incl_logdet_s,
        ))
    }
}

impl crate::solver::outer_strategy::OuterHessianOperator for UnifiedOuterHessianOperator {
    fn dim(&self) -> usize {
        self.coords.len()
    }

    fn matvec(&self, alpha: &Array1<f64>) -> Result<Array1<f64>, String> {
        if alpha.len() != self.coords.len() {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian alpha length mismatch: got {}, expected {}",
                    alpha.len(),
                    self.coords.len()
                ),
            }
            .into());
        }
        let mut a_alpha = 0.0;
        for (idx, coord) in self.coords.iter().enumerate() {
            if alpha[idx] != 0.0 {
                a_alpha += alpha[idx] * coord.a;
            }
        }
        let correction_m_alpha = self.theta_direction_mode_response(alpha);
        let callback_neg_m_alpha =
            matches!(self.kernel, OuterHessianDerivativeKernel::Callback { .. })
                .then(|| -&correction_m_alpha);
        // #740: one ψψ-block hook contraction per matvec (one family row pass),
        // shared read-only across the parallel per-row entries below.
        let psi_contrib = self.psi_contracted_contrib(alpha)?;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let values: Result<Vec<f64>, String> = (0..self.coords.len())
            .into_par_iter()
            .map(|idx| {
                self.outer_hessian_index_entry(
                    idx,
                    alpha,
                    a_alpha,
                    &correction_m_alpha,
                    callback_neg_m_alpha.as_ref(),
                    psi_contrib.as_ref(),
                )
            })
            .collect();

        Ok(Array1::from_vec(values?))
    }

    /// Zero-alloc override for the inner-CG hot path.
    ///
    /// Eliminates the transient `Vec<f64>` + `Array1::from_vec` allocation that
    /// `matvec` incurs on every CG step.  The parallel computation is identical;
    /// results are written directly into the caller-supplied `out` buffer via
    /// `par_iter_mut().enumerate()` rather than collected into a fresh `Vec`.
    fn apply_into(
        &self,
        alpha: &ndarray::Array1<f64>,
        out: &mut ndarray::Array1<f64>,
    ) -> Result<(), String> {
        if alpha.len() != self.coords.len() {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian alpha length mismatch: got {}, expected {}",
                    alpha.len(),
                    self.coords.len()
                ),
            }
            .into());
        }
        if out.len() != self.coords.len() {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian apply_into output length mismatch: got {}, expected {}",
                    out.len(),
                    self.coords.len()
                ),
            }
            .into());
        }
        let mut a_alpha = 0.0;
        for (idx, coord) in self.coords.iter().enumerate() {
            if alpha[idx] != 0.0 {
                a_alpha += alpha[idx] * coord.a;
            }
        }
        let correction_m_alpha = self.theta_direction_mode_response(alpha);
        let callback_neg_m_alpha =
            matches!(self.kernel, OuterHessianDerivativeKernel::Callback { .. })
                .then(|| -&correction_m_alpha);
        // #740: one ψψ-block hook contraction per matvec (see `matvec`).
        let psi_contrib = self.psi_contracted_contrib(alpha)?;
        let slice = out
            .as_slice_mut()
            .ok_or_else(|| "outer Hessian apply_into: non-contiguous output buffer".to_string())?;
        slice
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(idx, cell)| {
                *cell = self.outer_hessian_index_entry(
                    idx,
                    alpha,
                    a_alpha,
                    &correction_m_alpha,
                    callback_neg_m_alpha.as_ref(),
                    psi_contrib.as_ref(),
                )?;
                Ok(())
            })
    }
}

pub(crate) fn build_outer_hessian_operator(
    solution: &InnerSolution<'_>,
    lambdas: &[f64],
    effective_deriv: &dyn HessianDerivativeProvider,
    kernel: OuterHessianDerivativeKernel,
    precomputed_coord_vs: Option<&[Array1<f64>]>,
    precomputed_coord_corrections: Option<&[Option<DriftDerivResult>]>,
) -> Result<UnifiedOuterHessianOperator, String> {
    let hop = Arc::clone(&solution.hessian_op);
    let k = lambdas.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();

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

    let (profiled_phi, profiled_nu, profiled_dp_cgrad, profiled_dp_cgrad2, is_profiled) =
        match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => {
                let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
                let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw);
                let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
                let phi_hat = dp_c / nu;
                (phi_hat, nu, dp_cgrad, dp_cgrad2, true)
            }
            _ => (1.0, 1.0, 1.0, 0.0, false),
        };

    let rho_penalty_a_k_betas: Vec<Array1<f64>> = (0..k)
        .into_par_iter()
        .map(|idx| penalty_a_k_beta(&solution.penalty_coords[idx], &solution.beta, lambdas[idx]))
        .collect();
    let rho_curvature_a_k_betas: Vec<Array1<f64>> = (0..k)
        .into_par_iter()
        .map(|idx| {
            penalty_a_k_beta(
                &solution.penalty_coords[idx],
                &solution.beta,
                curvature_lambdas[idx],
            )
        })
        .collect();
    // Mode responses are fixed-β stationarity derivatives. The main
    // evaluator passes precomputed responses so gradient and Hessian share
    // the same solve kernel; when none are provided this standalone path
    // routes through the SAME `ThetaModeResponseKernel::select` decision as
    // the gradient site and the dense Hessian (#931 pass 2) — pre-port it
    // hand-copied the selection rule and a comment warned it to "mirror the
    // dense evaluator's selection exactly, otherwise the operator-form
    // Hessian and dense materialization disagree on every entry whose row
    // or column lives outside `range(U_S)`" (the
    // `projected_operator_hessian_matches_dense_subspace_trace` regression).
    // That mirroring obligation is now structural: one constructor, no copy
    // to drift.
    let subspace = solution.penalty_subspace_trace.as_deref();
    let coord_vs_storage;
    let coord_vs: &[Array1<f64>] = if let Some(precomputed) = precomputed_coord_vs {
        if precomputed.len() != total {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian precomputed mode-response count mismatch: got {}, expected {}",
                    precomputed.len(),
                    total
                ),
            }
            .into());
        }
        precomputed
    } else {
        let owned = if total == 0 {
            Vec::new()
        } else {
            let mode_kernel = ThetaModeResponseKernel::select(
                subspace,
                solution.active_constraints.as_deref(),
                &*hop,
            );
            let mut rhs_stack = Array2::<f64>::zeros((hop.dim(), total));
            for idx in 0..k {
                rhs_stack
                    .column_mut(idx)
                    .assign(&rho_curvature_a_k_betas[idx]);
            }
            for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
                rhs_stack.column_mut(k + ext_idx).assign(&coord.g);
            }
            let solved_stack = mode_kernel.respond_stack(&rhs_stack);
            (0..total)
                .map(|idx| solved_stack.column(idx).to_owned())
                .collect::<Vec<_>>()
        };
        coord_vs_storage = owned;
        &coord_vs_storage
    };
    for (coord_idx, response) in coord_vs.iter().enumerate() {
        if let Some((entry_idx, value)) = response
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(RemlError::NonFiniteValue {
                reason: format!(
                    "outer Hessian mode response contains non-finite entry: \
                     coord={coord_idx} entry={entry_idx} value={value}"
                ),
            }
            .into());
        }
    }

    let coord_corrections_storage;
    let coord_corrections: &[Option<DriftDerivResult>] = if let Some(precomputed) =
        precomputed_coord_corrections
    {
        if precomputed.len() != total {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian precomputed correction count mismatch: got {}, expected {}",
                    precomputed.len(),
                    total
                ),
            }
            .into());
        }
        precomputed
    } else if effective_deriv.has_corrections() {
        if effective_deriv.has_batched_hessian_derivative_corrections() {
            log::info!(
                "[STAGE] outer_hessian coord_corrections mode=batched k={} ext_dim={} n={} dim={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim()
            );
            coord_corrections_storage =
                effective_deriv.hessian_derivative_corrections_result(coord_vs)?;
        } else {
            coord_corrections_storage = coord_vs
                .par_iter()
                .map(|v_i| effective_deriv.hessian_derivative_correction_result(v_i))
                .collect::<Result<Vec<_>, _>>()?;
        }
        &coord_corrections_storage
    } else {
        coord_corrections_storage = (0..total).map(|_| None).collect::<Vec<_>>();
        &coord_corrections_storage
    };

    let mut coords = Vec::with_capacity(total);
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let curvature_a_k_beta = rho_curvature_a_k_betas[idx].clone();
        let v_k = coord_vs[idx].clone();
        let correction = coord_corrections[idx].as_ref();
        let mut total_dense = None;
        let mut total_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], correction) {
            DriftDerivResult::Dense(matrix) => total_dense = Some(matrix),
            DriftDerivResult::Operator(op) => total_operators.push(op),
        }
        let mut base_dense = None;
        let mut base_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], None) {
            DriftDerivResult::Dense(matrix) => base_dense = Some(matrix),
            DriftDerivResult::Operator(op) => base_operators.push(op),
        }
        let dense_rotated = match (hop.as_dense_spectral(), total_dense.as_ref()) {
            (Some(dense_hop), Some(matrix)) => Some(dense_hop.rotate_to_eigenbasis(matrix)),
            _ => None,
        };
        let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambdas[idx]);
        coords.push(OuterHessianCoord {
            a: a_i,
            g: curvature_a_k_beta,
            v: v_k,
            total_drift: StoredFirstDrift::from_parts(total_dense, dense_rotated, total_operators),
            base_drift: StoredFirstDrift::from_parts(base_dense, None, base_operators),
            ext_index: None,
            b_depends_on_beta: false,
        });
    }

    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let coord_idx = k + ext_idx;
        let v_i = coord_vs[coord_idx].clone();
        let correction = coord_corrections[coord_idx].as_ref();
        let (total_dense, total_operators) =
            hyper_coord_total_drift_parts(&coord.drift, correction);
        let (base_dense, base_operators) = hyper_coord_total_drift_parts(&coord.drift, None);
        let dense_rotated = match (hop.as_dense_spectral(), total_dense.as_ref()) {
            (Some(dense_hop), Some(matrix)) => Some(dense_hop.rotate_to_eigenbasis(matrix)),
            _ => None,
        };
        coords.push(OuterHessianCoord {
            a: coord.a,
            g: coord.g.clone(),
            v: v_i,
            total_drift: StoredFirstDrift::from_parts(total_dense, dense_rotated, total_operators),
            base_drift: StoredFirstDrift::from_parts(base_dense, None, base_operators),
            ext_index: Some(ext_idx),
            b_depends_on_beta: coord.b_depends_on_beta,
        });
    }

    let mut pair_a = Array2::<f64>::zeros((total, total));
    let mut pair_ld_s = Array2::<f64>::zeros((total, total));
    let mut g_dot_v = Array2::<f64>::zeros((total, total));
    let mut pair_g = vec![vec![None; total]; total];
    let mut base_h2 = Array2::<f64>::zeros((total, total));
    let mut m_pair_trace = Array2::<f64>::zeros((total, total));

    for ii in 0..total {
        for jj in ii..total {
            let value = match (coords[ii].ext_index, coords[jj].ext_index) {
                (None, None) => {
                    let rho_j = jj;
                    rho_penalty_a_k_betas[rho_j].dot(&coords[ii].v)
                }
                (None, Some(_)) => {
                    let rho_i = ii;
                    rho_penalty_a_k_betas[rho_i].dot(&coords[jj].v)
                }
                (Some(_), None) => {
                    let rho_j = jj;
                    rho_penalty_a_k_betas[rho_j].dot(&coords[ii].v)
                }
                (Some(_), Some(_)) => coords[ii].g.dot(&coords[jj].v),
            };
            g_dot_v[[ii, jj]] = value;
            g_dot_v[[jj, ii]] = value;
        }
    }

    for ii in 0..k {
        for jj in ii..k {
            pair_ld_s[[ii, jj]] = det2[[ii, jj]];
            if ii != jj {
                pair_ld_s[[jj, ii]] = det2[[ii, jj]];
            }
        }
    }

    for idx in 0..k {
        pair_a[[idx, idx]] = coords[idx].a;
        pair_g[idx][idx] = Some(coords[idx].g.clone());
        let base = if let Some(kernel) = subspace {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            kernel.trace_projected_logdet(&a_k)
        } else if solution.penalty_coords[idx].is_block_local() {
            let (block, start, end) = solution.penalty_coords[idx].scaled_block_local(1.0);
            hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_logdet_gradient(&a_k)
        };
        base_h2[[idx, idx]] = base;
    }

    if let Some(rho_ext_fn) = solution.rho_ext_pair_fn.as_ref() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pair_count = k * ext_dim;
        let entries: Vec<(usize, usize, HyperCoordPair)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let rho_idx = pair_idx / ext_dim;
                let ext_idx = pair_idx % ext_dim;
                let pair = rho_ext_fn(rho_idx, ext_idx);
                (rho_idx, ext_idx, pair)
            })
            .collect();
        // Batch all second-drift traces so `--scale-dimensions` pays one
        // shared Hutchinson solve stream for the whole rho-ext block instead
        // of one estimator per pair.  Projected subspace traces skip the
        // stochastic shortcut inside `compute_base_h2_traces`.
        let pair_refs: Vec<&HyperCoordPair> = entries.iter().map(|(_, _, pair)| pair).collect();
        let bases = compute_base_h2_traces(
            hop.as_ref(),
            &pair_refs,
            subspace,
            Some(Arc::clone(&solution.stochastic_trace_state)),
        );
        for ((rho_idx, ext_idx, pair), base) in entries.into_iter().zip(bases.into_iter()) {
            let row = rho_idx;
            let col = k + ext_idx;
            pair_a[[row, col]] = pair.a;
            pair_a[[col, row]] = pair.a;
            pair_ld_s[[row, col]] = pair.ld_s;
            pair_ld_s[[col, row]] = pair.ld_s;
            pair_g[row][col] = Some(pair.g.clone());
            pair_g[col][row] = Some(pair.g);
            base_h2[[row, col]] = base;
            base_h2[[col, row]] = base;
        }
    }

    // #740: when the direction-contracted ψψ hook is installed, the ψψ-block
    // entries of pair_a / pair_ld_s / base_h2 and the ψψ pair_g are supplied
    // per-matvec by the hook in a single family row pass — so SKIP this `K²`
    // per-pair assembly entirely (each `ext_pair_fn(ii,jj)` is an O(n) family
    // row fold and `compute_base_h2_traces` then traces each at O(n·r)). The ρρ
    // and ρψ blocks above stay in the tables (cheap, no family row pass). The
    // ψψ entries left zero here are exactly the ones the hook adds in
    // `outer_hessian_index_entry`.
    if let (Some(ext_pair_fn), None) = (
        solution.ext_coord_pair_fn.as_ref(),
        solution.contracted_psi_second_order.as_ref(),
    ) {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pair_count = ext_dim * (ext_dim + 1) / 2;
        let entries: Vec<(usize, usize, HyperCoordPair)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let (ii, jj) = upper_triangle_pair_from_index(pair_idx, ext_dim);
                let pair = ext_pair_fn(ii, jj);
                (ii, jj, pair)
            })
            .collect();
        let pair_refs: Vec<&HyperCoordPair> = entries.iter().map(|(_, _, pair)| pair).collect();
        let bases = compute_base_h2_traces(
            hop.as_ref(),
            &pair_refs,
            subspace,
            Some(Arc::clone(&solution.stochastic_trace_state)),
        );
        for ((ii, jj, pair), base) in entries.into_iter().zip(bases.into_iter()) {
            let row = k + ii;
            let col = k + jj;
            pair_a[[row, col]] = pair.a;
            pair_a[[col, row]] = pair.a;
            pair_ld_s[[row, col]] = pair.ld_s;
            pair_ld_s[[col, row]] = pair.ld_s;
            let g_pair = pair.g.clone();
            pair_g[row][col] = Some(g_pair.clone());
            pair_g[col][row] = Some(g_pair);
            base_h2[[row, col]] = base;
            base_h2[[col, row]] = base;
        }
    }

    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pair_count = total * (total + 1) / 2;
        let pair_drifts: Vec<((usize, usize), Vec<DriftDerivResult>)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let (ii, jj) = upper_triangle_pair_from_index(pair_idx, total);
                let beta_i = coords[ii].v.mapv(|value| -value);
                let beta_j = coords[jj].v.mapv(|value| -value);
                let mut drifts = Vec::new();
                if let Some(drift_fn) = solution.fixed_drift_deriv.as_ref() {
                    if coords[ii].b_depends_on_beta
                        && let Some(ext_i) = coords[ii].ext_index
                        && let Some(result) = drift_fn(ext_i, &beta_j)
                    {
                        drifts.push(result);
                    }
                    if coords[jj].b_depends_on_beta
                        && let Some(ext_j) = coords[jj].ext_index
                        && let Some(result) = drift_fn(ext_j, &beta_i)
                    {
                        drifts.push(result);
                    }
                }
                ((ii, jj), drifts)
            })
            .collect();

        let mut term_pairs = Vec::new();
        let mut term_drifts = Vec::new();
        for ((ii, jj), drifts) in pair_drifts {
            for drift in drifts {
                term_pairs.push((ii, jj));
                term_drifts.push(drift);
            }
        }

        if !term_drifts.is_empty() {
            let term_traces = if let Some(kernel) = subspace {
                penalty_subspace_trace_drifts_batched(kernel, &term_drifts)
            } else if let Some(ds) = hop.as_exact_dense_spectral() {
                dense_spectral_trace_logdet_drifts_batched(ds, &term_drifts)
            } else {
                term_drifts
                    .iter()
                    .map(|drift| drift.trace_logdet(hop.as_ref()))
                    .collect()
            };
            for ((ii, jj), trace) in term_pairs.into_iter().zip(term_traces.into_iter()) {
                m_pair_trace[[ii, jj]] += trace;
                if ii != jj {
                    m_pair_trace[[jj, ii]] += trace;
                }
            }
        }
    }

    // Precompute pair-wise logdet-Hessian cross traces:
    //   cross_trace[i, j] = tr(G_ε(H) Ḣ_i Ḣ_j)
    // Each coord's total Hessian drift Ḣ decomposes into a dense block plus
    // operator terms; the bilinear form expands across all four
    // dense-dense / dense-op / op-dense / op-op cross combinations.  By
    // bilinearity of `tr(G_ε(H) · · )` in the second factor, the full
    // alpha-combo cross trace recovered in matvec via
    //   cross_trace.row(i).dot(alpha)
    // matches the previous on-the-fly recomputation that built `alpha_dense`,
    // `alpha_dense_rotated`, and `alpha_op` at every HVP.
    let cross_trace: Option<Array2<f64>> = if incl_logdet_h {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let dense_hop_opt = hop.as_dense_spectral();
        if let Some(kernel) = subspace {
            let drift_parts = coords
                .iter()
                .map(|coord| {
                    let dense = coord.total_drift.dense.clone();
                    let op = if coord.total_drift.operators.is_empty() {
                        None
                    } else {
                        Some(Arc::new(CompositeHyperOperator {
                            dim_hint: hop.dim(),
                            dense: None,
                            operators: coord.total_drift.operators.clone(),
                        }) as Arc<dyn HyperOperator>)
                    };
                    match (dense, op) {
                        (Some(matrix), Some(operator)) => {
                            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                                dim_hint: hop.dim(),
                                dense: Some(matrix),
                                operators: vec![operator],
                            }))
                        }
                        (Some(matrix), None) => DriftDerivResult::Dense(matrix),
                        (None, Some(operator)) => DriftDerivResult::Operator(operator),
                        (None, None) => {
                            DriftDerivResult::Dense(Array2::zeros((hop.dim(), hop.dim())))
                        }
                    }
                })
                .collect::<Vec<_>>();
            let reduced = penalty_subspace_reduce_drifts_batched(kernel, &drift_parts);
            let pair_count = total * (total + 1) / 2;
            let pair_values: Vec<((usize, usize), f64)> = (0..pair_count)
                .into_par_iter()
                .map(|pair_idx| {
                    let (ii, jj) = upper_triangle_pair_from_index(pair_idx, total);
                    let value =
                        -kernel.trace_projected_logdet_cross_reduced(&reduced[ii], &reduced[jj]);
                    ((ii, jj), value)
                })
                .collect();
            let mut ct = Array2::<f64>::zeros((total, total));
            for ((ii, jj), value) in pair_values {
                if !value.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "outer Hessian operator projected cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ),
                    }
                    .into());
                }
                ct[[ii, jj]] = value;
                if ii != jj {
                    ct[[jj, ii]] = value;
                }
            }
            Some(ct)
        } else if hop.prefers_stochastic_trace_estimation() && hop.logdet_traces_match_hinv_kernel()
        {
            // Matrix-free backends expose the SPD logdet kernel
            //   ∂² log|H|[A_i,A_j] = -tr(H⁻¹ A_i H⁻¹ A_j).
            //
            // Estimate the whole coordinate matrix in one Hutchinson batch
            // rather than launching one two-coordinate estimator per upper
            // triangle entry.  For `--scale-dimensions` with 16 ψ axes this
            // replaces 136 independent solve batches with one 16-coordinate
            // batch sharing the same probes and Krylov solves.
            let bundled: Vec<BorrowedStoredDriftOperator<'_>> = coords
                .iter()
                .map(|coord| BorrowedStoredDriftOperator {
                    drift: &coord.total_drift,
                    dim_hint: hop.dim(),
                })
                .collect();
            let op_refs: Vec<&dyn HyperOperator> =
                bundled.iter().map(|op| op as &dyn HyperOperator).collect();
            let estimator = StochasticTraceEstimator::for_outer_hessian_with_trace_state(
                hop.dim(),
                total,
                Arc::clone(&solution.stochastic_trace_state),
            );
            let no_dense: [&Array2<f64>; 0] = [];
            let mut ct = estimator.estimate_second_order_traces_with_operators(
                hop.as_ref(),
                &no_dense,
                &op_refs,
            );
            ct.mapv_inplace(|value| -value);
            Some(ct)
        } else if let Some(dense_hop) = dense_hop_opt {
            // Exact smooth-logdet Hessian kernel for operator-backed drifts.
            //
            // The second derivative of
            //     log |r_epsilon(H(theta))|
            // is not, in general,
            //     -tr(H_epsilon^{-1} H_i H_epsilon^{-1} H_j).
            // That identity only holds for the unregularized SPD logdet.
            // DenseSpectralOperator uses the divided-difference kernel of
            // log r_epsilon(sigma), so every dense/operator component must be
            // rotated into the eigenbasis and contracted with that same
            // kernel.  The dense Hessian assembly path already does this;
            // the matrix-free outer-Hv path must match it exactly.
            let mut rotated: Vec<Array2<f64>> =
                coords
                    .iter()
                    .map(|coord| {
                        coord.total_drift.dense_rotated.clone().unwrap_or_else(|| {
                            Array2::<f64>::zeros((dense_hop.n_dim, dense_hop.n_dim))
                        })
                    })
                    .collect();
            let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
            for (idx, coord) in coords.iter().enumerate() {
                for op in &coord.total_drift.operators {
                    collect_projected_matrix_terms(
                        idx,
                        1.0,
                        op.as_ref(),
                        &dense_hop.eigenvectors,
                        &mut rotated,
                        &mut terms,
                    );
                }
            }
            let projected_ops = project_hyper_operators_batched(
                total,
                &terms,
                &dense_hop.eigenvectors,
                &dense_hop.projected_factor_cache,
            );
            for (dst, projected) in rotated.iter_mut().zip(projected_ops.iter()) {
                *dst += projected;
            }

            let mut ct = Array2::<f64>::zeros((total, total));
            for ii in 0..total {
                for jj in ii..total {
                    let value =
                        dense_hop.trace_logdet_hessian_cross_rotated(&rotated[ii], &rotated[jj]);
                    if !value.is_finite() {
                        return Err(RemlError::NonFiniteValue {
                            reason: format!(
                                "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                            ),
                        }
                        .into());
                    }
                    ct[[ii, jj]] = value;
                    if ii != jj {
                        ct[[jj, ii]] = value;
                    }
                }
            }
            Some(ct)
        } else {
            // Enumerate the upper triangle (`ii ≤ jj`) so each `(ii, jj)` is an
            // independent unit of work — every entry of `cross_trace` is computed
            // from `coords[ii]` / `coords[jj]` only, with no shared mutable
            // state, so we can dispatch the K(K+1)/2 pair traces in parallel.
            let pair_count = total * (total + 1) / 2;
            let pair_values: Vec<((usize, usize), f64)> = (0..pair_count)
                .into_par_iter()
                .map(|pair_idx| {
                    let (ii, jj) = upper_triangle_pair_from_index(pair_idx, total);
                    let left = &coords[ii].total_drift;
                    let right = &coords[jj].total_drift;
                    let mut value = 0.0;
                    if let (Some(left_dense), Some(right_dense)) =
                        (left.dense.as_ref(), right.dense.as_ref())
                    {
                        if let (Some(dense_hop), Some(left_rot), Some(right_rot)) = (
                            dense_hop_opt,
                            left.dense_rotated.as_ref(),
                            right.dense_rotated.as_ref(),
                        ) {
                            value +=
                                dense_hop.trace_logdet_hessian_cross_rotated(left_rot, right_rot);
                        } else {
                            value += hop.trace_logdet_hessian_cross(left_dense, right_dense);
                        }
                    }
                    if let Some(left_dense) = left.dense.as_ref() {
                        for op in &right.operators {
                            value -= hop.trace_hinv_matrix_operator_cross(left_dense, op.as_ref());
                        }
                    }
                    if let Some(right_dense) = right.dense.as_ref() {
                        for op in &left.operators {
                            value -= hop.trace_hinv_matrix_operator_cross(right_dense, op.as_ref());
                        }
                    }
                    if !left.operators.is_empty() && !right.operators.is_empty() {
                        // Bundle each side's per-mode operators into a single
                        // weight-1 linear combination so the cross trace expands
                        // as `tr(H⁻¹ Â B̂) = Σ_a Σ_b tr(H⁻¹ A_a B_b)` with one
                        // call into the cross-trace kernel instead of the full
                        // O(|left.ops|·|right.ops|) sweep. Mathematically
                        // equivalent (bilinearity of `tr(H⁻¹ · ·)`).
                        let left_bundle = WeightedHyperOperator {
                            terms: left
                                .operators
                                .iter()
                                .map(|op| (1.0, Arc::clone(op)))
                                .collect(),
                            dim_hint: hop.dim(),
                        };
                        let right_bundle = WeightedHyperOperator {
                            terms: right
                                .operators
                                .iter()
                                .map(|op| (1.0, Arc::clone(op)))
                                .collect(),
                            dim_hint: hop.dim(),
                        };
                        value -= hop.trace_hinv_operator_cross(&left_bundle, &right_bundle);
                    }
                    ((ii, jj), value)
                })
                .collect();
            let mut ct = Array2::<f64>::zeros((total, total));
            for ((ii, jj), value) in pair_values {
                if !value.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ),
                    }
                    .into());
                }
                ct[[ii, jj]] = value;
                if ii != jj {
                    ct[[jj, ii]] = value;
                }
            }
            Some(ct)
        }
    } else {
        None
    };

    // Leverage and the scalar-GLM adjoint-z_c cache support both the
    // full-Hessian and projected-subspace paths.  Under subspace,
    //   h^{G,proj}_i = Xᵢᵀ · K · Xᵢ                  (K = U_S H_proj⁻¹ U_Sᵀ)
    //   u            = K · rhs                       (mirrors
    //                                                  `compute_ift_correction_trace`
    //                                                  slow path and the
    //                                                  per-coord v computation
    //                                                  above)
    //   z_c^{proj}   = K · Xᵀ(c ⊙ h^{G,proj})
    // and the adjoint identity
    //   tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj})
    //               = (K rhs)ᵀ · Xᵀ(c ⊙ h^{G,proj})
    //               = rhsᵀ · K · Xᵀ(c ⊙ h^{G,proj})
    //               = rhsᵀ · z_c^{proj}
    // lets `scalar_correction_trace` take the cheap branch
    // `rhs · z_c^{proj}` instead of materialising the second-derivative
    // correction.  Both the leverage AND the adjoint vector must swap to
    // their projected forms — gating only one (the historical bug) made the
    // operator path's `scalar_correction_trace` compute
    // `rhsᵀ H⁻¹ Xᵀ(c⊙h^{G,proj})` while `compute_ift_correction_trace`
    // computes `rhsᵀ K · X'(c⊙h^{G,proj})`, so the operator-form Hessian
    // disagreed with the dense path materialisation in the regression
    // `projected_operator_hessian_matches_dense_subspace_trace`.
    let leverage = if incl_logdet_h {
        match &kernel {
            OuterHessianDerivativeKernel::Gaussian => None,
            OuterHessianDerivativeKernel::ScalarGlm { x, .. } => match subspace {
                Some(s) => Some(s.xt_projected_kernel_x_diagonal(x)),
                None => Some(hop.xt_logdet_kernel_x_diagonal(x)),
            },
            OuterHessianDerivativeKernel::Callback { .. } => None,
        }
    } else {
        None
    };
    let adjoint_z_c = if incl_logdet_h {
        match (&kernel, leverage.as_ref()) {
            (
                OuterHessianDerivativeKernel::ScalarGlm {
                    c_array,
                    d_array,
                    x,
                },
                Some(h_g),
            ) => Some(compute_adjoint_z_c(
                &ScalarGlmIngredients {
                    c_array,
                    d_array: d_array.as_ref(),
                    x,
                },
                hop.as_ref(),
                h_g,
                subspace,
            )?),
            _ => None,
        }
    } else {
        None
    };

    let callback_second_modes = matches!(kernel, OuterHessianDerivativeKernel::Callback { .. })
        .then(|| {
            coords
                .iter()
                .map(|coord| {
                    if coord.is_ext() {
                        coord.v.clone()
                    } else {
                        -&coord.v
                    }
                })
                .collect::<Vec<_>>()
        });
    let fourth_trace = if incl_logdet_h && adjoint_z_c.is_some() {
        match (&kernel, leverage.as_ref()) {
            (
                OuterHessianDerivativeKernel::ScalarGlm {
                    c_array,
                    d_array: Some(d_array),
                    x,
                },
                Some(h_g),
            ) => {
                let modes = coords.iter().map(|coord| &coord.v).collect::<Vec<_>>();
                compute_fourth_derivative_trace_matrix(
                    &ScalarGlmIngredients {
                        c_array,
                        d_array: Some(d_array),
                        x,
                    },
                    &modes,
                    h_g,
                )?
            }
            _ => None,
        }
    } else {
        None
    };

    Ok(UnifiedOuterHessianOperator {
        hop,
        coords,
        pair_a,
        pair_ld_s,
        g_dot_v,
        pair_g,
        base_h2,
        m_pair_trace,
        cross_trace,
        profiled_phi,
        profiled_nu,
        profiled_dp_cgrad,
        profiled_dp_cgrad2,
        is_profiled,
        incl_logdet_h,
        incl_logdet_s,
        kernel,
        subspace: solution.penalty_subspace_trace.clone(),
        adjoint_z_c,
        leverage,
        fourth_trace,
        callback_second_modes,
        k_rho: k,
        contracted_psi: solution.contracted_psi_second_order.clone(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Extended Fellner–Schall (EFS) update for all hyperparameters
