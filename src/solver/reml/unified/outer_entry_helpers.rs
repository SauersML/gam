use super::*;

/// Evaluation mode for the unified evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Compute cost only (e.g., for line search).
    ValueOnly,
    /// Compute cost and gradient (the common case).
    ValueAndGradient,
    /// Compute cost, gradient, and outer Hessian.
    ValueGradientHessian,
}

/// Result of the unified REML/LAML evaluation.
#[derive(Debug)]
pub struct RemlLamlResult {
    /// The REML/LAML objective value (to be minimized).
    pub cost: f64,
    /// Newton-decrement energy `½ rᵀH⁻¹r` of the converged inner KKT
    /// residual at this `ρ`, where `r = ∇_β L(β̂, ρ)` and `H` is the inner
    /// Hessian. Bounds the inner sub-optimality `|V(β̂) − V(β*)| ≤
    /// ½ rᵀH⁻¹r` to first order, and is consumed by:
    ///
    /// * the HyperGradientBudget controller, which uses it as the
    ///   inner-channel energy proxy `E_inner` when estimating
    ///   `s_inner` and re-allocating per-channel tolerances; and
    /// * the trust-energy gate in the outer strategy, which shrinks the
    ///   trust radius when this energy exceeds
    ///   `TRUST_ENERGY_FACTOR × |predicted_decrease|`.
    ///
    /// `None` when the inner solve did not compute an energy estimate
    /// (e.g., projected-pseudo-inverse paths that lack a full-H solve).
    pub ift_residual_energy: Option<f64>,
    /// One-Newton-step inner polish vector `w = H⁻¹ r`, populated only
    /// when the evaluator solves against the full inner Hessian `H` (not
    /// the projected pseudo-inverse used on rank-deficient paths).
    ///
    /// Applied by the runtime as a *free* refinement of the warm-start β
    /// at the next outer iteration: `β_warm ← β̂ + w` short-circuits one
    /// PIRLS step, exploiting the Hessian factorization already paid for
    /// during the cost-side IFT correction. `None` whenever the polish
    /// step was not produced (projected-pseudo-inverse path, value-only
    /// evaluation, etc.).
    pub inner_polish_step: Option<Array1<f64>>,
    /// Gradient ∂V/∂ρ (present if mode ≥ ValueAndGradient).
    pub gradient: Option<Array1<f64>>,
    /// Outer Hessian ∂²V/∂ρ² (present if mode = ValueGradientHessian).
    pub hessian: crate::solver::outer_strategy::HessianResult,
    /// Rho-coordinate mode responses, one `K · g_j` vector per column, when
    /// they were already built for derivative corrections. Consumed by the
    /// runtime IFT mode-response cache for joint-IFT warm starts.
    pub rho_mode_response_cols: Option<Array2<f64>>,
    /// Extended-coordinate mode responses, one `K · g_j` vector per column,
    /// when extended derivative coordinates required them.
    pub ext_mode_response_cols: Option<Array2<f64>>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

// Canonical definitions live in estimate.rs; re-use them here.
use crate::solver::estimate::smooth_floor_dp;

/// Ridge floor for denominator safety.
pub(crate) const DENOM_RIDGE: f64 = 1e-8;

pub(crate) fn penalty_a_k_beta(
    coord: &PenaltyCoordinate,
    beta: &Array1<f64>,
    lambda: f64,
) -> Array1<f64> {
    coord.apply_shifted_penalty(beta, lambda)
}

pub(crate) fn penalty_a_k_quadratic(
    coord: &PenaltyCoordinate,
    beta: &Array1<f64>,
    lambda: f64,
) -> f64 {
    coord.shifted_quadratic(beta, lambda)
}

/// Apply the curvature-conditioning scale `s = rho_curvature_scale` to a
/// raw ρ-coordinate `λ_k = exp(ρ_k)`.
///
/// Returns `s · λ_k`, which is the per-coordinate drift coefficient
/// `∂H_op/∂ρ_k = s · λ_k · S_k` under the convention documented on
/// [`InnerSolution::rho_curvature_scale`].  The matching
/// `hessian_logdet_correction = −p · log(s)` (additive in ρ, derivative
/// zero) cancels the `p · log(s)` term in `log|H_op|` so that the cost
/// the evaluator reports and the trace `tr(K · s·λ_k·S_k)` (with
/// `K = H_op⁻¹ = (1/s) · H_orig⁻¹`) both correspond to the SAME unscaled
/// `log|H_orig|` and its analytic derivative `tr(H_orig⁻¹ · λ_k S_k)`.
///
/// If you change this scaling, you MUST also update the corresponding
/// `hessian_logdet_correction` in every caller that sets
/// `rho_curvature_scale ≠ 1`, or the cost and gradient will disagree by
/// a factor `s` — see issue #200 for the failure mode.
#[inline]
pub(crate) fn rho_curvature_lambda(solution: &InnerSolution<'_>, lambda: f64) -> f64 {
    solution.rho_curvature_scale * lambda
}

pub(crate) fn penalty_coord_to_operator(
    coord: PenaltyCoordinate,
    scale: f64,
) -> Arc<dyn HyperOperator> {
    struct OwnedPenaltyHyperOperator {
        pub(crate) coord: PenaltyCoordinate,
        pub(crate) scale: f64,
    }

    impl HyperOperator for OwnedPenaltyHyperOperator {
        fn dim(&self) -> usize {
            self.coord.dim()
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

        fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
            self.coord.apply_penalty_view_into(v, self.scale, out);
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
            self.coord
                .scaled_add_penalty_view(v, scale * self.scale, out);
        }

        fn to_dense(&self) -> Array2<f64> {
            self.coord.scaled_dense_matrix(self.scale)
        }

        fn is_implicit(&self) -> bool {
            false
        }
    }

    Arc::new(OwnedPenaltyHyperOperator { coord, scale })
}

pub(crate) fn penalty_total_drift_result(
    coord: &PenaltyCoordinate,
    scale: f64,
    correction: Option<&DriftDerivResult>,
) -> DriftDerivResult {
    match correction {
        Some(DriftDerivResult::Dense(corr)) => {
            if coord.uses_operator_fast_path() {
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: Some(corr.clone()),
                    operators: vec![penalty_coord_to_operator(coord.clone(), scale)],
                    dim_hint: coord.dim(),
                }))
            } else {
                let mut dense = coord.scaled_dense_matrix(scale);
                dense += corr;
                DriftDerivResult::Dense(dense)
            }
        }
        Some(DriftDerivResult::Operator(corr_op)) => {
            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                dense: if coord.uses_operator_fast_path() {
                    None
                } else {
                    Some(coord.scaled_dense_matrix(scale))
                },
                operators: {
                    let mut ops = vec![Arc::clone(corr_op)];
                    if coord.uses_operator_fast_path() {
                        ops.push(penalty_coord_to_operator(coord.clone(), scale));
                    }
                    ops
                },
                dim_hint: coord.dim(),
            }))
        }
        None => {
            if coord.uses_operator_fast_path() {
                DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                    dense: None,
                    operators: vec![penalty_coord_to_operator(coord.clone(), scale)],
                    dim_hint: coord.dim(),
                }))
            } else {
                DriftDerivResult::Dense(coord.scaled_dense_matrix(scale))
            }
        }
    }
}

pub(crate) fn hyper_coord_drift_operators(drift: &HyperCoordDrift) -> Vec<Arc<dyn HyperOperator>> {
    let mut operators: Vec<Arc<dyn HyperOperator>> = Vec::new();
    if let Some(block_local) = drift.block_local.as_ref() {
        operators.push(Arc::new(block_local.clone()));
    }
    if let Some(operator) = drift.operator.as_ref() {
        operators.push(Arc::clone(operator));
    }
    operators
}

pub(crate) fn hyper_coord_drift_operator_arc(
    drift: &HyperCoordDrift,
    dim_hint: usize,
) -> Option<Arc<dyn HyperOperator>> {
    let mut operators = hyper_coord_drift_operators(drift);
    if operators.is_empty() {
        return None;
    }

    if drift.dense.is_none() && operators.len() == 1 {
        return Some(operators.pop().expect("single operator drift"));
    }

    Some(Arc::new(CompositeHyperOperator {
        dense: drift.dense.clone(),
        operators,
        dim_hint,
    }))
}

pub(crate) fn drift_parts_into_result(
    dense: Option<Array2<f64>>,
    mut operators: Vec<Arc<dyn HyperOperator>>,
    dim_hint: usize,
) -> DriftDerivResult {
    if operators.is_empty() {
        DriftDerivResult::Dense(dense.unwrap_or_else(|| Array2::<f64>::zeros((dim_hint, dim_hint))))
    } else if dense.is_none() && operators.len() == 1 {
        DriftDerivResult::Operator(operators.pop().expect("single operator drift"))
    } else {
        DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
            dense,
            operators,
            dim_hint,
        }))
    }
}

pub(crate) fn hyper_coord_total_drift_parts(
    drift: &HyperCoordDrift,
    correction: Option<&DriftDerivResult>,
) -> (Option<Array2<f64>>, Vec<Arc<dyn HyperOperator>>) {
    let mut dense = drift.dense.clone();
    let mut operators = hyper_coord_drift_operators(drift);
    if let Some(correction) = correction {
        match correction {
            DriftDerivResult::Dense(matrix) => {
                if let Some(existing) = dense.as_mut() {
                    *existing += matrix;
                } else {
                    dense = Some(matrix.clone());
                }
            }
            DriftDerivResult::Operator(operator) => operators.push(Arc::clone(operator)),
        }
    }
    (dense, operators)
}

pub(crate) fn hyper_coord_total_drift_result(
    drift: &HyperCoordDrift,
    correction: Option<&DriftDerivResult>,
    dim_hint: usize,
) -> DriftDerivResult {
    let (dense, operators) = hyper_coord_total_drift_parts(drift, correction);
    drift_parts_into_result(dense, operators, dim_hint)
}

// ─── EFS multiplicative-update helpers ───────────────────────────────────
//
// The Wood–Fasiolo Extended Fellner–Schall update is multiplicative in the
// smoothing parameter. Writing it in log coordinates `ρ = log λ`,
//
//   Δρ = log( target / q_eff )
//      = log( ( d − t ) / q_eff )
//
// where:
//   • q_eff is the penalty-quadratic contribution to the *gradient*,
//     scaled exactly the way `outer_gradient_entry` scales it. For Fixed
//     dispersion, q_eff = β̂ᵀ B β̂ = 2 a_i. For ProfiledGaussian, it picks
//     up the smooth-floor factor `dp_cgrad / φ̂` so EFS and the gradient
//     share the same stationarity equation.
//   • d = ∂ log|S_λ|₊/∂ρ_i = tr(S_λ⁺ B_i). For ρ-coords this is
//     `solution.penalty_logdet.first[idx]`; for τ-coords it is
//     `coord.ld_s`.
//   • t = tr(K · B_i) where K is the *cost's* logdet kernel — `G_ε(H)` in
//     ordinary SPD/smooth-spectral mode, or the projected
//     `U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ` under the rank-deficient LAML fix.
//
// The previous implementation used `Δρ = (2a − tr(H⁻¹B)) / tr(H⁻¹BH⁻¹B)`,
// which (a) silently dropped the `tr(S_λ⁺ B)` term, (b) used a different
// kernel from the gradient, and (c) used the Frobenius/Gram trace as a
// curvature proxy instead of the canonical EFS denominator. As a concrete
// counterexample, the scalar Gaussian/Laplace model with z = 2, λ = 1/3 is
// at the exact REML optimum (gradient = 0) but the old formula returned
// step `+8` (clamped to `+5`) — see the unit test in this module.
//
// Exactness depends on the likelihood curvature. For Gaussian/quadratic
// likelihoods, `H_obs` is beta-independent, so `C[v_k] = 0` and the
// classical explicit trace fixed point with `Ḣ_k = λ_k S_k` is exact. For
// non-Gaussian families (Cox/survival/binomial), `H_obs` depends on beta;
// the exact logdet gradient uses the total Hessian drift
// `Ḣ_k = λ_k S_k + C[v_k]`. A pure MacKay/Tipping/Wood-Fasiolo explicit
// trace update that uses only `λ_k S_k` is therefore an approximation.
//
// This code path does not use that pure explicit-trace surrogate. EFS is
// expressed in terms of the full outer gradient from `reml_laml_evaluate`;
// that gradient builds `rho_corrections`, threads them through
// `penalty_total_drift_result`, and traces the corrected `Ḣ_k`.

/// `q_eff = 2 · penalty_term` matching `outer_gradient_entry`.
#[inline]
pub(crate) fn efs_q_eff(a_i: f64, dispersion: &DispersionHandling, dp_cgrad: f64, phi: f64) -> f64 {
    match dispersion {
        DispersionHandling::ProfiledGaussian => 2.0 * dp_cgrad * a_i / phi,
        DispersionHandling::Fixed { .. } => 2.0 * a_i,
    }
}

pub(crate) fn gamma_precision_rate_for_rho(
    prior: &crate::types::RhoPrior,
    idx: usize,
) -> Option<f64> {
    match prior {
        crate::types::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
        crate::types::RhoPrior::Independent(priors) => {
            priors.get(idx).and_then(|prior| match prior {
                crate::types::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
                _ => None,
            })
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn efs_q_eff_with_gamma_rate(
    base_q_eff: f64,
    lambda: f64,
    prior: &crate::types::RhoPrior,
    idx: usize,
) -> f64 {
    match gamma_precision_rate_for_rho(prior, idx) {
        Some(rate) if rate.is_finite() && rate > 0.0 => base_q_eff + 2.0 * rate * lambda,
        _ => base_q_eff,
    }
}

/// EFS step expressed in terms of the *full* outer gradient
/// `g_full = ∂V_total/∂ρ_i` and the penalty-quadratic curvature scale
/// `q_eff`:
///
/// ```text
///   Δρ = log(1 − 2·g_full / q_eff).
/// ```
///
/// This is the universal-form Wood–Fasiolo update: when the cost is base
/// REML/LAML, the canonical `g_base = (q_eff + t − d)/2` gives
/// `1 − 2·g_base/q_eff = (d − t)/q_eff` (the classical pseudoinverse-and-
/// trace form); when out-of-band terms — Tierney–Kadane corrections,
/// smoothing-parameter priors, Firth bias-reduction, monotonicity
/// barriers, the SAS log-δ ridge — enter `g_full = g_base + g_extra`,
/// the multiplicative target shifts by exactly the right amount,
/// `1 − 2·g_full/q_eff = (d − t − 2·g_extra)/q_eff`. No per-augmentation
/// post-correction is needed in `compute_efs_update` /
/// `compute_hybrid_efs_update`. The line search in the outer
/// fixed-point bridge handles the only thing this formula can't —
/// non-PSD penalty derivatives that flip the descent direction.
///
/// Three regimes:
/// - **Stable (`q_eff > 0`, `2·g_full < q_eff`)**: clamp to `±EFS_MAX_STEP`.
/// - **Over-correction (`q_eff > 0`, `2·g_full ≥ q_eff`)**: emit
///   `−EFS_MAX_STEP`; line search trims and the canonical form resumes
///   on the next iteration.
/// - **Pathological (`q_eff ≤ 0` or non-finite)**: returns `None` so the
///   caller leaves the step at zero for that coordinate.
#[inline]
pub(crate) fn efs_log_step_from_grad(q_eff: f64, g_full: f64) -> Option<f64> {
    if !q_eff.is_finite() || q_eff <= 0.0 || !g_full.is_finite() {
        return None;
    }
    let ratio = 1.0 - 2.0 * g_full / q_eff;
    if ratio > 0.0 {
        Some(ratio.ln().clamp(-EFS_MAX_STEP, EFS_MAX_STEP))
    } else {
        Some(-EFS_MAX_STEP)
    }
}

/// EFS profiling factors (`profiled_scale`, `dp_cgrad`) matched to the
/// gradient assembly. For Fixed dispersion both are unused; we return
/// `(phi, 0.0)` so that `efs_q_eff` simply uses `2·a_i`.
#[inline]
pub(crate) fn efs_profiling(solution: &InnerSolution<'_>) -> (f64, f64) {
    match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad, _) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            (dp_c / denom, dp_cgrad)
        }
        DispersionHandling::Fixed { phi, .. } => (*phi, 0.0),
    }
}

pub(crate) fn trace_hinv_cached_drift_cross(
    hop: &dyn HessianOperator,
    left_dense: Option<&Array2<f64>>,
    left_op: Option<&dyn HyperOperator>,
    right_dense: Option<&Array2<f64>>,
    right_op: Option<&dyn HyperOperator>,
) -> f64 {
    match (left_op, right_op) {
        (Some(left), Some(right)) => hop.trace_hinv_operator_cross(left, right),
        (Some(left), None) => hop.trace_hinv_matrix_operator_cross(
            right_dense.expect("right dense drift should be cached"),
            left,
        ),
        (None, Some(right)) => hop.trace_hinv_matrix_operator_cross(
            left_dense.expect("left dense drift should be cached"),
            right,
        ),
        (None, None) => hop.trace_hinv_product_cross(
            left_dense.expect("left dense drift should be cached"),
            right_dense.expect("right dense drift should be cached"),
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared outer-derivative formulas
// ═══════════════════════════════════════════════════════════════════════════
//
// These helpers implement the analytic identities ONCE so that all
// coordinate types (ρ, τ, ψ) and all pair types (ρ-ρ, ρ-ext, ext-ext)
// go through the same formula. Any chain-rule or transformed-parameter
// fix automatically applies to every code path.

/// Compute one entry of the outer gradient.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂V/∂θ_i = a_i_scaled + ½ tr(G_ε Ḣ_i) − ½ ∂_i log|S|₊
/// ```
///
/// where:
/// - `a_i` is the fixed-β cost derivative (0.5 × β̂ᵀAₖβ̂ for ρ, coord.a for ext)
/// - `trace_logdet_i` is tr(G_ε(H) Ḣ_i) (logdet gradient operator applied to
///   the total Hessian drift including IFT correction)
/// - `ld_s_i` is ∂_i log|S|₊ (penalty pseudo-logdet derivative)
///
/// The dispersion handling scales the penalty term:
/// - Profiled Gaussian: dp_cgrad × a_i / φ̂
/// - Fixed dispersion: a_i
#[inline]
pub(crate) fn outer_gradient_entry(
    a_i: f64,
    trace_logdet_i: f64,
    ld_s_i: f64,
    dispersion: &DispersionHandling,
    dp_cgrad: f64,
    profiled_scale: f64,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let penalty_term = match dispersion {
        DispersionHandling::ProfiledGaussian => dp_cgrad * a_i / profiled_scale,
        DispersionHandling::Fixed { .. } => a_i,
    };
    let trace_term = if incl_logdet_h {
        0.5 * trace_logdet_i
    } else {
        0.0
    };
    let det_term = if incl_logdet_s { 0.5 * ld_s_i } else { 0.0 };
    penalty_term + trace_term - det_term
}

/// Compute one entry of the outer Hessian.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂²V/∂θ_i∂θ_j = Q_ij + L_ij + P_ij
/// ```
///
/// where:
/// - Q_ij = pair_a − g_i·v_j  (penalty quadratic second derivative, with
///   profiled Gaussian chain-rule terms from the smooth deviance floor)
/// - L_ij = ½ (cross_trace + h2_trace) (logdet Hessian)
/// - P_ij = −½ pair_ld_s  (penalty logdet second derivative)
///
/// The `cross_trace` is the exact logdet spectral cross term. For ordinary
/// SPD backends this is `−tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i)`; for smooth spectral logdet
/// regularization it is the divided-difference contraction of
/// `log r_ε(σ)`. The `h2_trace` is tr(G_ε Ḧ_ij) from the second Hessian
/// drift including IFT and fourth-derivative corrections.
#[inline]
pub(crate) fn outer_hessian_entry(
    a_i: f64,
    a_j: f64,
    g_i_dot_v_j: f64,
    pair_a: f64,
    cross_trace: f64,
    h2_trace: f64,
    pair_ld_s: f64,
    profiled_phi: f64,
    profiled_nu: f64,
    profiled_dp_cgrad: f64,
    profiled_dp_cgrad2: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let q_raw = pair_a - g_i_dot_v_j;
    let q = if is_profiled {
        profiled_dp_cgrad * q_raw / profiled_phi
            + 2.0
                * (profiled_dp_cgrad2 * profiled_nu * profiled_phi
                    - profiled_dp_cgrad * profiled_dp_cgrad)
                * a_i
                * a_j
                / (profiled_nu * profiled_phi * profiled_phi)
    } else {
        q_raw
    };
    let l = if incl_logdet_h {
        0.5 * (cross_trace + h2_trace)
    } else {
        0.0
    };
    let p = if incl_logdet_s { -0.5 * pair_ld_s } else { 0.0 };
    q + l + p
}

// ═══════════════════════════════════════════════════════════════════════════
//  Constraint-tangent-space projection
// ═══════════════════════════════════════════════════════════════════════════
//
// When the inner solver converges at a constrained-stationary point with a
// non-empty active inequality-constraint set `A_act β = b_act` (k_act rows),
// the Laplace approximation lives on the tangent manifold `T = β̂ + null(A_act)`.
// With orthonormal basis `Z ∈ ℝ^{p × m}` for null(A_act) (m = p − k_act), the
// principled outer LAML objective is
//
//   V_T(ρ) = -ℓ(β̂) + ½ β̂ᵀ S(λ) β̂ + ½ log|ZᵀHZ| − ½ log|Zᵀ S(λ) Z|_+ + …
//
// (β̂-quadratic terms stay in p-space; β̂ doesn't change under projection.)
// The gradient is the envelope-theorem derivative at fixed β̂:
//
//   ∂_ρ_k V_T = ½ λ_k β̂ᵀ S_k β̂ + ½ tr((ZᵀHZ)⁻¹ Zᵀ(λ_k S_k) Z)
//             − ½ λ_k tr((ZᵀSZ)⁺ ZᵀS_kZ)
//
// Refs: Wood 2011; Wood–Pya–Säfken 2016 §3; Marra–Wood 2012 §2.
//
// The implementation strategy: wrap the inner Hessian operator in a
// tangent-projected adapter that transforms its trace/solve/logdet APIs
// from p-space to tangent space, recompute `PenaltyLogdetDerivs` for
// `ZᵀS(λ)Z`, then recurse into the regular `reml_laml_evaluate` with
// `active_constraints = None`. This routes the entire downstream pipeline
// (gradient, Hessian, IFT corrections) through the projected operator
// without duplicating cost/gradient formulas.

/// Orthonormal basis `Z ∈ ℝ^{p × m}` for `null(A_act)` via eigendecomposition
/// of `A_actᵀ A_act` (PSD; `null(A_actᵀ A_act) = null(A_act)`). Returns
/// `None` when the active set is empty or the tangent space is empty
/// (k_act ≥ p).
pub(crate) fn compute_active_constraint_tangent_basis(a_act: &Array2<f64>) -> Option<Array2<f64>> {
    let k_act = a_act.nrows();
    let p = a_act.ncols();
    if k_act == 0 {
        return None;
    }
    // `null(A_act) = null(A_actᵀ A_act)`; eigendecompose the symmetric PSD
    // `p × p` matrix and pull the eigenvectors with σ ≤ threshold as the
    // null basis. This gives `m = p − rank(A_act)`, the correct tangent
    // dimension regardless of whether `A_act` has linearly dependent rows.
    let ata = a_act.t().dot(a_act);
    let (evals, evecs) = ata.eigh(faer::Side::Lower).ok()?;
    let evals_slice = evals.as_slice()?;
    let threshold = positive_eigenvalue_threshold(evals_slice);
    let null_count = evals_slice.iter().filter(|&&s| s <= threshold).count();
    if null_count == 0 || null_count == p {
        // `null_count == p` means A_act has no effective constraint (every
        // row is in the noise floor). Returning `None` skips the projection
        // and lets the full p-space evaluator run unmodified.
        return None;
    }
    Some(evecs.slice(ndarray::s![.., 0..null_count]).to_owned())
}

/// Dense `p × p` materialization of a penalty coordinate via canonical
/// basis vectors. `S_k e_j` is the `j`-th column of `S_k`; assembled into
/// a p × p matrix. Cost O(p² · matvec).
pub(crate) fn materialize_penalty_coord_dense(coord: &PenaltyCoordinate, p: usize) -> Array2<f64> {
    // Each `PenaltyCoordinate` variant already has a structure-aware
    // materializer (`scaled_dense_matrix(1.0)`):
    //   - `DenseRoot` / `DenseRootCentered` → `Rᵀ R` via faer matmul
    //     (BLAS3, parallel).
    //   - `BlockRoot` / `BlockRootCentered` → block-local `Rᵀ R` embedded
    //     into a `total_dim × total_dim` matrix.
    //   - `KroneckerMarginal` → diagonal write (no matmul needed).
    // Routing through it replaces the previous serial p-fold matvec loop
    // with the variant-appropriate O(p²) (or O(p) for Kronecker) path.
    let out = coord.scaled_dense_matrix(1.0);
    assert_eq!(out.nrows(), p, "penalty coord dim mismatch");
    assert_eq!(out.ncols(), p, "penalty coord dim mismatch");
    out
}

/// Reconstruct the *raw* Hessian `H = V · diag(σ) · Vᵀ` (pre-regularization)
/// from a `DenseSpectralOperator`. The operator stores
/// `r_ε(σ) = ½(σ + √(σ² + 4ε²))`; invert via `σ = r − ε²/r` so the tangent
/// projection `ZᵀHZ` sees the un-regularized data. The `from_symmetric`
/// call applied to that projection then performs a *single* tangent-space
/// regularization, matching `log|ZᵀHZ|` with one consistent `r_ε` instead
/// of double-regularizing (`r_ε(ZᵀV·r_ε(σ)·VᵀZ)`).
///
/// Per the math review (codex), projecting an already-regularized H_reg
/// and re-regularizing in tangent space is not exactly `log|ZᵀHZ|`; it is
/// a modified smoothed objective. Inverting `r_ε` first restores the
/// principled single-regularization identity.
pub(crate) fn assemble_h_raw_dense(op: &DenseSpectralOperator) -> Array2<f64> {
    let p = op.n_dim;
    // `ε = √ε_mach · p`. Same `spectral_epsilon` formula as the operator's
    // own construction; depends only on dim.
    let epsilon = f64::EPSILON.sqrt() * (p as f64).max(1.0);
    let eps_sq = epsilon * epsilon;
    if p == 0 {
        return Array2::<f64>::zeros((0, 0));
    }
    // Express `H = V · diag(σ_raw) · Vᵀ` as two BLAS3 matmuls (faer's
    // `fast_ab` / `fast_atb` are already parallelized internally),
    // replacing the previous triple-nested O(p³) loop.
    //
    //   sigma_j = r_j − ε²/r_j  for active, nonzero `r`; else 0.
    //   VS = V · diag(sigma)    (scale columns of V by sigma)
    //   H  = VS · Vᵀ            (= fast_abt(VS, V))
    let mut vs = op.eigenvectors.clone();
    for j in 0..p {
        let sigma = if op.active_mask[j] {
            let r = op.reg_eigenvalues[j];
            if r == 0.0 { 0.0 } else { r - eps_sq / r }
        } else {
            0.0
        };
        if sigma != 1.0 {
            let mut col = vs.column_mut(j);
            if sigma == 0.0 {
                col.fill(0.0);
            } else {
                col.mapv_inplace(|v| v * sigma);
            }
        }
    }
    // H = VS · Vᵀ without materializing Vᵀ.
    crate::faer_ndarray::fast_abt(&vs, &op.eigenvectors)
}

/// Tangent-projected `HessianOperator` adapter. Wraps an `m × m`
/// `H_T = ZᵀHZ` operator and exposes the `p × p` interface needed by the
/// existing evaluator pipeline. All p-space inputs are projected via `Z`
/// before being passed to the tangent operator; outputs are lifted back
/// via `Z`. By construction this is the constraint-aware pseudo-inverse
/// `H⁺_T = Z (ZᵀHZ)⁻¹ Zᵀ`, which is bounded independent of σ_min(H)
/// when σ_min(ZᵀHZ) is bounded.
pub(crate) struct TangentProjectedHessianOperator {
    /// Orthonormal basis for null(A_act), `p × m`.
    pub(crate) z: Array2<f64>,
    /// `H_T = ZᵀHZ`, re-eigendecomposed with its own `r_ε` regularization.
    pub(crate) h_t_op: DenseSpectralOperator,
}

impl HessianOperator for TangentProjectedHessianOperator {
    fn active_rank(&self) -> usize {
        self.h_t_op.active_rank()
    }

    fn dim(&self) -> usize {
        self.z.nrows()
    }
    fn logdet(&self) -> f64 {
        self.h_t_op.logdet()
    }
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let r_t = self.z.t().dot(rhs);
        let q_t = self.h_t_op.solve(&r_t);
        self.z.dot(&q_t)
    }
    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let r_t = self.z.t().dot(rhs);
        let q_t = self.h_t_op.solve_multi(&r_t);
        self.z.dot(&q_t)
    }
    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(Z H_T⁻¹ Zᵀ · A) = tr(H_T⁻¹ · ZᵀAZ) (cyclic permutation).
        let zaz = self.z.t().dot(a).dot(&self.z);
        self.h_t_op.trace_hinv_product(&zaz)
    }
    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) · A) where H is the wrapped tangent operator.
        // d log|ZᵀHZ|/dt = tr((ZᵀHZ)⁻¹ · Zᵀ Ḣ Z) → use H_T's logdet kernel
        // applied to ZᵀḢZ.
        let zaz = self.z.t().dot(a).dot(&self.z);
        self.h_t_op.trace_logdet_gradient(&zaz)
    }
    fn is_dense(&self) -> bool {
        self.h_t_op.is_dense()
    }
    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        self.h_t_op.logdet_traces_match_hinv_kernel()
    }
    // Deliberately keep `as_dense_spectral` and `as_exact_dense_spectral`
    // at default `None`: their consumers expect a p-space spectral basis,
    // whereas the wrapped operator lives in m-dimensional tangent space.
    // Surfacing the tangent operator there would silently let downstream
    // code mix p- and m-dim eigenvectors.
}

/// Build `PenaltyLogdetDerivs` for `log|ZᵀS(λ)Z|_+`, its first
/// derivatives, and its second derivatives. The identities are the same
/// as in p-space, applied to the projected penalty:
///   value      = log|M(λ)|_+,                M(λ) = ZᵀS(λ)Z = Σ_k λ_k Zᵀ S_k Z
///   ∂_k value  = λ_k · tr(M⁺ · Zᵀ S_k Z)
///   ∂²_kl      = δ_{kl} ∂_k value − λ_k λ_l · tr(M⁺ · Zᵀ S_l Z · M⁺ · Zᵀ S_k Z)
pub(crate) fn tangent_penalty_logdet(
    z: &Array2<f64>,
    penalty_coords: &[PenaltyCoordinate],
    lambdas: &[f64],
    p: usize,
) -> Result<PenaltyLogdetDerivs, String> {
    let m = z.ncols();
    let k = lambdas.len();
    let zsz: Vec<Array2<f64>> = penalty_coords
        .iter()
        .map(|c| {
            let s_k_full = materialize_penalty_coord_dense(c, p);
            z.t().dot(&s_k_full).dot(z)
        })
        .collect();
    let mut s_t = Array2::<f64>::zeros((m, m));
    for k_idx in 0..k {
        s_t.scaled_add(lambdas[k_idx], &zsz[k_idx]);
    }
    let (evals, evecs) = s_t
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("tangent S eigendecomposition failed: {e}"))?;
    let evals_slice = evals.as_slice().ok_or_else(|| {
        "tangent S eigendecomposition returned non-contiguous eigenvalues".to_string()
    })?;
    let threshold = positive_eigenvalue_threshold(evals_slice);
    let value = exact_pseudo_logdet(evals_slice, threshold);
    // Build M⁺ = Σ_{σ_j > τ} u_j u_jᵀ / σ_j once for first AND second derivatives.
    let mut s_t_plus = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        if evals[j] > threshold {
            let inv = 1.0 / evals[j];
            for r in 0..m {
                let factor = evecs[[r, j]] * inv;
                for c in 0..m {
                    s_t_plus[[r, c]] += factor * evecs[[c, j]];
                }
            }
        }
    }
    let mut first = Array1::<f64>::zeros(k);
    for k_idx in 0..k {
        first[k_idx] = lambdas[k_idx] * trace_matrix_product(&s_t_plus, &zsz[k_idx]);
    }
    let mut second = Array2::<f64>::zeros((k, k));
    // δ_{kl} ∂_k value contribution (from ∂_ρ_l λ_k = λ_k δ_{kl}).
    for k_idx in 0..k {
        second[[k_idx, k_idx]] += first[k_idx];
    }
    // − λ_k λ_l · tr(M⁺ · Zᵀ S_l Z · M⁺ · Zᵀ S_k Z).
    let s_plus_zsz: Vec<Array2<f64>> = zsz.iter().map(|m_k| s_t_plus.dot(m_k)).collect();
    for k_idx in 0..k {
        for l_idx in 0..=k_idx {
            let cross = trace_matrix_product(&s_plus_zsz[k_idx], &s_plus_zsz[l_idx]);
            let entry = -lambdas[k_idx] * lambdas[l_idx] * cross;
            second[[k_idx, l_idx]] += entry;
            if l_idx != k_idx {
                second[[l_idx, k_idx]] += entry;
            }
        }
    }
    Ok(PenaltyLogdetDerivs {
        value,
        first,
        second: Some(second),
    })
}

/// Borrowing adapter that lets a tangent-projected `InnerSolution` reuse
/// the original `HessianDerivativeProvider` without taking ownership.
/// The provider returns p-space drift matrices (`D_β H[v]`) which the
/// tangent-wrapped `HessianOperator` correctly projects via `ZᵀMZ` in
/// its `trace_logdet_gradient` / `trace_hinv_product` methods. So no
/// per-method projection is needed here — pure delegation suffices.
pub(crate) struct BorrowedDerivProvider<'a>(&'a dyn HessianDerivativeProvider);

impl<'a> HessianDerivativeProvider for BorrowedDerivProvider<'a> {
    fn hessian_derivative_correction(
        &self,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.hessian_derivative_correction(v)
    }
    fn hessian_derivative_correction_result(
        &self,
        v: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.0.hessian_derivative_correction_result(v)
    }
    fn hessian_derivative_corrections_result(
        &self,
        vs: &[Array1<f64>],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.0.hessian_derivative_corrections_result(vs)
    }
    fn has_batched_hessian_derivative_corrections(&self) -> bool {
        self.0.has_batched_hessian_derivative_corrections()
    }
    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.hessian_second_derivative_correction(v_k, v_l, u_kl)
    }
    fn hessian_second_derivative_correction_result(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        self.0
            .hessian_second_derivative_correction_result(v_k, v_l, u_kl)
    }
    fn hessian_second_derivative_corrections_result(
        &self,
        triples: &[(Array1<f64>, Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<DriftDerivResult>>, String> {
        self.0.hessian_second_derivative_corrections_result(triples)
    }
    fn has_batched_hessian_second_derivative_corrections(&self) -> bool {
        self.0.has_batched_hessian_second_derivative_corrections()
    }
    fn has_corrections(&self) -> bool {
        self.0.has_corrections()
    }
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        self.0.outer_hessian_derivative_kernel()
    }
    fn family_outer_hessian_operator(
        &self,
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        self.0.family_outer_hessian_operator()
    }
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        self.0.scalar_glm_ingredients()
    }
}

/// If the inner solution carries a non-empty active inequality-constraint
/// set, build a tangent-projected solution and dispatch the outer
/// derivative computation to it. Returns `Ok(None)` when no projection is
/// required (no active constraints); `Ok(Some(result))` when projection
/// succeeded and the recursive evaluate returned a value; `Err` only if
/// projection failed (e.g., dense backend required but not available).
pub(crate) fn try_tangent_projected_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<Option<RemlLamlResult>, String> {
    let block = match solution.active_constraints.as_ref() {
        Some(b) if b.a.nrows() > 0 => b,
        _ => return Ok(None),
    };
    let p = solution.beta.len();
    if block.a.ncols() != p {
        return Err(format!(
            "active_constraints.a has {} columns but β is {}-dim",
            block.a.ncols(),
            p
        ));
    }
    // Principled pass-through / projection of optional `InnerSolution`
    // features.  The cost-side scalars (`tk_correction`, barrier cost at β̂)
    // are not in β-space and require no projection.  `tk_gradient` is a
    // ρ-vector and also passes through unchanged.  `hessian_logdet_correction`
    // encodes a p-space uniform rescale `−p·log α`; under projection the
    // equivalent correction is `−m·log α`, recovered by the scalar factor
    // `m/p`.  `barrier_config` propagates to the projected solution so the
    // barrier-derivative wrapper still augments dH/dρ; the tangent operator
    // applies `ZᵀMZ` correctly in its trace methods.
    //
    // `firth` and `ext_coords` carry p-space objects (Jeffreys `½ log|J|`,
    // ext-coord g/drift).  Both are projected here under the same
    // tangent-projected LAML setup as the rest of this routine
    // (`½ log|J| → ½ log|ZᵀJZ|`, `g → Zᵀ g`, `drift M → Zᵀ M Z`).  The
    // only remaining unsupported case is the `ValueGradientHessian`
    // mode with non-empty `ext_coord_pair_fn` / `rho_ext_pair_fn` —
    // those callbacks return objects in p-space whose projected form
    // requires composing with `Z` per-call; for `ValueAndGradient` the
    // per-coord `g`/`drift` is sufficient.
    let z = match compute_active_constraint_tangent_basis(&block.a) {
        Some(z) => z,
        None => {
            // Constraint matrix spans the full p-space — tangent manifold is
            // {β̂}. There is no degree of freedom left to optimise over and
            // the outer LAML cost is the constant `-ℓ(β̂) + ½ β̂ᵀSβ̂`. We
            // can return a degenerate result, but it's simpler and clearer
            // to surface the situation.
            return Err(format!(
                "active constraint matrix has rank {} on {}-dim space; \
                 tangent manifold is a single point ({{β̂}}), no outer \
                 derivative is defined",
                block.a.nrows(),
                p
            ));
        }
    };
    let h_full = solution
        .hessian_op
        .assemble_h_dense_for_tangent_projection()?;
    let h_t = z.t().dot(&h_full).dot(&z);
    let h_t_op = DenseSpectralOperator::from_symmetric(&h_t)
        .map_err(|e| format!("tangent H eigendecomposition failed: {e}"))?;
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let projected_logdet = tangent_penalty_logdet(&z, &solution.penalty_coords, &lambdas, p)?;
    // Project the KKT residual to lift the IFT correction into tangent
    // coordinates: with `q = H⁺_T r = Z (ZᵀHZ)⁻¹ Zᵀ r`, the formulas
    // `-½ rᵀ q` and `-aᵀ_k q + ½ qᵀ A_k q` are the same as the p-space
    // formulas (Zᵀ cancels through the operator wrapper). We pass r in
    // p-space; the wrapper does the projection internally.
    let projected_kkt = solution.kkt_residual.clone();
    let m_tangent = z.ncols();
    let wrapper = TangentProjectedHessianOperator {
        z: z.clone(),
        h_t_op,
    };
    // Rank-aware projection of a uniform-rescale correction:
    //   the p-space correction encodes `−p·log α` so that
    //   `log|H| = log|H'| − p·log α`. Under tangent projection the
    //   correction becomes `−m·log α = (m/p) · (−p·log α)`, i.e. the
    //   same scalar scaled by the rank ratio m/p.
    let projected_hlogdet_correction = if p == 0 {
        0.0
    } else {
        solution.hessian_logdet_correction * (m_tangent as f64 / p as f64)
    };
    // Construct the projected InnerSolution. The fields that must be
    // overridden are: hessian_op (now tangent-wrapped), penalty_logdet
    // (now in tangent space), hessian_logdet_correction (rank-ratio
    // rescaled to tangent space), penalty_subspace_trace (None; direct
    // tangent-H path replaces the kernel route), active_constraints
    // (None; prevents recursion). `tk_*` (ρ-/scalar-space) and
    // `barrier_config` (cost evaluated at β̂ in p-space; barrier-derivative
    // wrapper produces p-space drift that the tangent operator projects
    // via ZᵀMZ in its trace methods) pass through unchanged.
    // Active-constraint tangent projection for the Firth/Jeffreys term.
    // Replace the operator's full-space `½ log|J|` with the projected
    // `½ log|ZᵀJZ|`. The same underlying `FirthDenseOperator` is retained
    // so any downstream β-gradient consumer still sees a consistent
    // operator — only the scalar contribution to the outer LAML cost is
    // overridden. This projection-aware Firth is exact under the same
    // tangent-projected LAML setup as the rest of
    // `try_tangent_projected_evaluate` (mode = ValueAndGradient or below).
    let projected_firth = solution
        .firth
        .as_ref()
        .map(|term| match term.operator_arc() {
            Some(op_arc) => {
                let projected_value = op_arc.jeffreys_logdet_projected(z.view());
                ExactJeffreysTerm::with_projected_value(op_arc, projected_value)
            }
            // Tier-B value-only carrier: the scalar Φ(β̂) is already final (the
            // coupled joint path owns its own constraint handling upstream), so
            // the term passes through unchanged.
            None => term.clone(),
        });
    // Active-constraint tangent projection for ext coords. The tangent
    // hessian wrapper accepts p-space `g` and p-space drift `M` and
    // applies the `Zᵀ · Z` projection internally inside its `solve` /
    // `trace_logdet_gradient` / `trace_hinv_product` methods, so
    // pass-through is mathematically equivalent to projecting `g → Zᵀg`
    // and `M → ZᵀMZ` here (the wrapper composes the projections with
    // the inner H_T operator). This is the same pattern
    // `BorrowedDerivProvider` uses for the deriv-provider corrections.
    //
    // The pair callbacks (`ext_coord_pair_fn`, `rho_ext_pair_fn`) return
    // `HyperCoordPair` objects with p-space `b_mat` / `b_operator`.
    // `ValueAndGradient` mode does not contract those pair objects (they
    // only enter outer-Hessian assembly), so they are dropped (set to
    // `None`) in the projected inner solution — gradient evaluations are
    // unaffected. `ValueGradientHessian` mode would actually consume the
    // pair callbacks; the tangent hessian wrapper cannot re-project
    // their p-space second-drift outputs a posteriori, so refuse that
    // combination upfront when callbacks are present.
    if mode == EvalMode::ValueGradientHessian
        && !solution.ext_coords.is_empty()
        && (solution.ext_coord_pair_fn.is_some() || solution.rho_ext_pair_fn.is_some())
    {
        return Err(
            "active constraints + ext_coords + mode=ValueGradientHessian not yet supported; \
             fall back to ValueAndGradient. The ext-coord pair callbacks return p-space \
             second-drift objects that the tangent hessian wrapper does not re-project."
                .to_string(),
        );
    }
    let projected = InnerSolution {
        log_likelihood: solution.log_likelihood,
        penalty_quadratic: solution.penalty_quadratic,
        hessian_op: Arc::new(wrapper),
        beta: solution.beta.clone(),
        penalty_coords: solution.penalty_coords.clone(),
        penalty_logdet: projected_logdet,
        deriv_provider: Box::new(BorrowedDerivProvider(solution.deriv_provider.as_ref())),
        tk_correction: solution.tk_correction,
        tk_gradient: solution.tk_gradient.clone(),
        // Same operator, projection-aware scalar contribution.
        firth: projected_firth,
        hessian_logdet_correction: projected_hlogdet_correction,
        // Direct tangent-H path; the projected-kernel route is unused here.
        penalty_subspace_trace: None,
        rho_curvature_scale: solution.rho_curvature_scale,
        rho_prior: solution.rho_prior.clone(),
        n_observations: solution.n_observations,
        nullspace_dim: solution.nullspace_dim,
        gaussian_weight_log_sum_half: solution.gaussian_weight_log_sum_half,
        dp_floor_scale: solution.dp_floor_scale,
        dispersion: solution.dispersion.clone(),
        // ext_coord g/drift pass-through: projection is applied by the
        // tangent hessian wrapper's trace and solve methods.
        ext_coords: solution.ext_coords.clone(),
        ext_coord_pair_fn: None,
        rho_ext_pair_fn: None,
        // Second-order pair callbacks are dropped on the projected path (same
        // reason as the ext-coord/rho pair fns: the tangent hessian wrapper
        // cannot re-project their p-space second-drift outputs).
        contracted_psi_second_order: None,
        fixed_drift_deriv: None,
        barrier_config: solution.barrier_config.clone(),
        kkt_residual: projected_kkt,
        // Prevents recursion via `try_tangent_projected_evaluate`.
        active_constraints: None,
        stochastic_trace_state: solution.stochastic_trace_state.clone(),
    };
    let result = reml_laml_evaluate(&projected, rho, mode, prior_cost_gradient)?;
    Ok(Some(result))
}
