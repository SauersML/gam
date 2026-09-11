use super::*;

/// Result of the unified REML/LAML evaluation.
#[derive(Debug)]
pub struct RemlLamlResult {
    /// The REML/LAML objective value (to be minimized).
    pub cost: f64,
    /// Additive scalar decomposition of `cost`, retained through outer
    /// correction atoms so structured finite-difference audits can compare
    /// each analytic gradient atom with the derivative of the scalar it owns.
    pub criterion_components: RemlCriterionComponents,
    /// Newton-decrement energy `½ rᵀH⁻¹r` of the converged inner KKT
    /// residual at this `ρ`, where `r = ∇_β L(β̂, ρ)` and `H` is the inner
    /// Hessian. Bounds the inner sub-optimality `|V(β̂) − V(β*)| ≤
    /// ½ rᵀH⁻¹r` to first order, and is consumed by the trust-energy gate in
    /// the outer strategy, which shrinks the trust radius when this energy
    /// exceeds `TRUST_ENERGY_FACTOR × |predicted_decrease|`.
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
    pub hessian: gam_problem::HessianValue,
    /// Rho-coordinate mode responses, one `K · g_j` vector per column, when
    /// they were already built for derivative corrections. Consumed by the
    /// runtime IFT mode-response cache for joint-IFT warm starts.
    pub rho_mode_response_cols: Option<Array2<f64>>,
    /// Extended-coordinate mode responses, one `K · g_j` vector per column,
    /// when extended derivative coordinates required them.
    pub ext_mode_response_cols: Option<Array2<f64>>,
}

/// Four additive scalar atoms of the unified criterion.
///
/// `fixed_beta` owns every scalar other than the two determinant terms and the
/// accepted-inner-mode correction. This includes configured priors, barriers,
/// Firth, Tierney–Kadane, and sampled-block corrections. The four values always
/// sum to [`RemlLamlResult::cost`].
#[derive(Clone, Copy, Debug)]
pub struct RemlCriterionComponents {
    pub fixed_beta: f64,
    pub logdet_h: f64,
    pub logdet_s: f64,
    pub kkt: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

// Canonical definitions live in estimate.rs; re-use them here.
use crate::estimate::smooth_floor_dp;

/// Residual degrees of freedom `ν = n − M_p` of the profiled-Gaussian scale.
///
/// `n` is the positive-weight observation count and `M_p = p − rank(S_λ)` the
/// number of UNPENALIZED coefficient directions, so `ν` is a difference of two
/// integer counts: it is either `≥ 1` or `≤ 0`, never in between. Both
/// consumers — the profiled scale `φ̂ = D_p/ν` and the `(ν/2)·log(2πφ̂)`
/// REML term — are undefined at `ν ≤ 0`: a design whose unpenalized directions
/// already exhaust the observations carries no residual information from which
/// to estimate a scale. This refuses there, which is what the rest of the
/// codebase does with exactly this condition
/// (`estimate/optimizer.rs`, `gaussian_reml.rs` × 3,
/// `fit_orchestration/drivers/design_construction.rs`).
///
/// It replaces a `.max(1e-8)` clamp (#2669). Because `ν` is integer-valued that
/// clamp could never interpolate: it was exactly `if ν ≤ 0 { 1e-8 }`, and what
/// it produced there was `φ̂ = D_p/1e-8`, which collapses the data-fit term
/// `D_p/(2φ̂) = ν/2` to `5e-9` INDEPENDENTLY of the response and leaves the
/// outer optimizer selecting λ against a bare `½(log|H| − log|S|)` determinant
/// ratio. Fabricating a finite criterion for a structurally invalid fit is
/// worse than refusing it (SPEC: a fit object must only ever come from a
/// converged optimization).
///
/// Takes the two scalars rather than the whole `InnerSolution` so the refusal
/// is directly exercisable — see `profiled_gaussian_residual_dof_tests`.
pub(crate) fn profiled_gaussian_residual_dof(
    n_observations: usize,
    nullspace_dim: f64,
) -> Result<f64, String> {
    let dof = n_observations as f64 - nullspace_dim;
    if dof > 0.0 {
        Ok(dof)
    } else {
        Err(format!(
            "profiled Gaussian residual degrees of freedom must be positive; got              n({n_observations}) − M_p({nullspace_dim}) = {dof}. Every unpenalized              coefficient direction consumes one observation, so this design leaves              nothing to estimate the Gaussian scale from: penalize the offending              directions or drop them."
        ))
    }
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

        fn as_any(&self) -> &(dyn std::any::Any + 'static) {
            self
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
    prior: &gam_problem::RhoPrior,
    idx: usize,
) -> Option<f64> {
    match prior {
        gam_problem::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
        gam_problem::RhoPrior::Independent(priors) => {
            priors.get(idx).and_then(|prior| match prior {
                gam_problem::RhoPrior::GammaPrecision { rate, .. } => Some(*rate),
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
    prior: &gam_problem::RhoPrior,
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
pub(crate) fn efs_profiling(solution: &InnerSolution<'_>) -> Result<(f64, f64), String> {
    match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad, _) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
            let denom =
                profiled_gaussian_residual_dof(solution.n_observations, solution.nullspace_dim)?;
            Ok((dp_c / denom, dp_cgrad))
        }
        DispersionHandling::Fixed { phi, .. } => Ok((*phi, 0.0)),
    }
}

pub(crate) fn trace_hinv_cached_drift_cross(
    hop: &dyn HessianFactorization,
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

/// Authoritative coefficient geometry of a non-empty active constraint face.
pub enum ActiveConstraintTangentGeometry {
    /// The active rows span coefficient space, so the mode is fully pinned.
    FullyPinned,
    /// Orthonormal basis `Z` for the non-empty tangent `null(A_act)`.
    Tangent(Array2<f64>),
}

/// Row-normalize a non-empty active constraint block, returning the normalized
/// rows alongside the norms that were divided out.
///
/// Constraint feasibility, working-face membership, and every active-set KKT
/// gate are defined in scaled slack `(a·beta-b)/‖a‖`; rank, tangent and affine
/// solutions must be invariant to multiplying an inequality by an arbitrary
/// positive constant as well. Factoring raw Khatri–Rao rows lets large-norm
/// rows numerically erase independent small-norm rows, producing a direction
/// that is tangent only in an unscaled least-squares aggregate and leaves the
/// solver's actual face. A zero or non-finite row is left alone with a unit
/// norm so the caller's own row validity check — not this scaling — is what
/// reports it.
fn normalize_active_face_rows(a_act: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut normalized = a_act.clone();
    let mut row_norms = Array1::<f64>::ones(a_act.nrows());
    for (index, mut row) in normalized.rows_mut().into_iter().enumerate() {
        let norm = row.dot(&row).sqrt();
        if norm.is_finite() && norm > 0.0 {
            row /= norm;
            row_norms[index] = norm;
        }
    }
    (normalized, row_norms)
}

/// Numerical row rank and tangent geometry of an already-normalized active
/// block, read off a factorization the caller already paid for.
///
/// Splitting this out is what lets the tangent, the rank and the affine
/// particular solution of one face be answered from ONE factorization instead
/// of from three that can disagree (gam#2600).
fn tangent_from_row_factorization(
    normalized: &Array2<f64>,
    singular: &Array1<f64>,
    vt: &Array2<f64>,
) -> Result<(usize, ActiveConstraintTangentGeometry), String> {
    let p = normalized.ncols();
    let smax = singular.iter().fold(0.0_f64, |largest, &s| largest.max(s));
    let rank_threshold = 100.0 * f64::EPSILON * (normalized.nrows().max(p) as f64) * smax;
    let rank = singular.iter().filter(|&&s| s > rank_threshold).count();
    if rank == 0 {
        return Err("non-empty active constraint block has zero numerical row rank".to_string());
    }
    if rank == p {
        return Ok((rank, ActiveConstraintTangentGeometry::FullyPinned));
    }

    let null_count = p - rank;
    let mut orthonormal_basis: Vec<Array1<f64>> = (0..rank).map(|i| vt.row(i).to_owned()).collect();
    let mut z = Array2::<f64>::zeros((p, null_count));
    let mut collected = 0usize;
    let independence_floor = 100.0 * f64::EPSILON * p as f64;
    for axis in 0..p {
        if collected == null_count {
            break;
        }
        let mut candidate = Array1::<f64>::zeros(p);
        candidate[axis] = 1.0;
        // A second pass is deliberate: modified Gram–Schmidt reorthogonalization
        // keeps the constructed complement accurate for ill-conditioned faces.
        for _ in 0..2 {
            for q in &orthonormal_basis {
                let projection = q.dot(&candidate);
                candidate.scaled_add(-projection, q);
            }
        }
        let norm = candidate.dot(&candidate).sqrt();
        if norm > independence_floor {
            candidate /= norm;
            z.column_mut(collected).assign(&candidate);
            orthonormal_basis.push(candidate);
            collected += 1;
        }
    }
    if collected != null_count {
        return Err(format!(
            "active constraint tangent complement construction produced {collected} of {null_count} columns"
        ));
    }

    // This routine is the shared authority for both optimization and LAML
    // projection. Refuse geometry that cannot preserve the solver's working-
    // face contract instead of silently returning a numerically false tangent.
    for row in normalized.rows() {
        let row_norm = row.dot(&row).sqrt();
        if row_norm == 0.0 {
            continue;
        }
        for tangent in z.columns() {
            let relative_leakage = row.dot(&tangent).abs() / row_norm;
            if relative_leakage > crate::active_set::ACTIVE_SET_WORKING_FACE_TOL {
                return Err(format!(
                    "active constraint tangent leaks through its working face: relative leakage {relative_leakage:.3e}"
                ));
            }
        }
    }
    Ok((rank, ActiveConstraintTangentGeometry::Tangent(z)))
}

/// Compute the coefficient geometry of a non-empty active constraint face.
///
/// This is shared by the terminal inner determinant and the outer evaluator so
/// the value cannot classify curvature in one coefficient space while its
/// derivatives use another. A non-empty row block with zero numerical rank is
/// invalid active-set evidence, not an unconstrained fallback.
pub fn active_constraint_tangent_geometry(
    a_act: &Array2<f64>,
) -> Result<ActiveConstraintTangentGeometry, String> {
    if a_act.nrows() == 0 {
        return Err("active constraint tangent geometry requires at least one row".to_string());
    }
    let p = a_act.ncols();
    if p == 0 {
        return Ok(ActiveConstraintTangentGeometry::FullyPinned);
    }
    let (normalized, _row_norms) = normalize_active_face_rows(a_act);

    // Factor the rectangular normalized row block directly. Forming `A_actᵀ A_act`
    // squares its condition number and can erase independent active rows near
    // machine precision. That produces a tangent which leaks in a constraint-
    // normal direction: the next accepted point then falls off the working
    // face and discards the entire warm active set. The thin SVD gives the row
    // rank without normal equations; complete its right-singular row basis to
    // an orthonormal null basis using twice-reorthogonalized coordinate axes.
    //
    // Left singular vectors are NOT requested here: this entry point answers
    // only the tangent question, and the callers that also need the affine
    // particular solution go through `active_constraint_face_geometry`, which
    // pays for `U` once and shares this exact rank rule.
    let (_u, singular, vt) = normalized
        .svd(false, true)
        .map_err(|error| format!("active constraint tangent SVD failed: {error}"))?;
    let vt = vt.ok_or_else(|| "active constraint tangent SVD omitted Vᵀ".to_string())?;
    tangent_from_row_factorization(&normalized, &singular, &vt).map(|(_rank, geometry)| geometry)
}

/// Minimum-norm particular solution of an active face's affine system,
/// together with the residual the numerically null directions leave behind.
pub struct ParticularFaceSolution {
    /// The step `δ` of least Euclidean norm over the numerically identified
    /// row space, satisfying `A δ = rhs` up to `residual_inf`.
    pub delta: Array1<f64>,
    /// `‖A δ − rhs‖∞`, in scaled-slack units — the same units feasibility,
    /// working-face membership and every active-set gate are stated in.
    pub residual_inf: f64,
    /// Largest residual a *consistent* face can leave at this scale. Anything
    /// above it means the equalities cannot be met simultaneously: the face is
    /// wrong, not the arithmetic.
    pub residual_tolerance: f64,
}

/// Rank-revealing affine geometry of a non-empty active constraint face.
///
/// A reduced-face solve needs three things from one face: its numerical row
/// rank, an orthonormal basis of `null(A)`, and a particular solution of the
/// affine system `A δ = rhs`. Deciding those from three different
/// factorizations — which is what the physical reduced face used to do, taking
/// the rank from a Gram–Schmidt scan of `A`, the tangent from an SVD of `A`,
/// and the particular solution from an SVD of the trust-whitened `A D^{-1/2}`
/// — lets one face be simultaneously full rank and singular. That is exactly
/// how gam#2600 refused: `A` was accepted as 39 independent rows while the
/// whitened block reported `σ_min = 4.8e-18` against a `1.1e-13` floor, and
/// the trust metric's dynamic range, not the geometry, decided it.
///
/// This type carries ONE factorization of the row-normalized block and answers
/// all three from it. The trust metric is not involved: it selects *which*
/// solution of an underdetermined face is smallest, which is a separate
/// projection the caller applies afterwards, and it must not be allowed to
/// decide whether the face has a solution at all.
pub struct ActiveConstraintFaceGeometry {
    rank: usize,
    tangent: ActiveConstraintTangentGeometry,
    normalized: Array2<f64>,
    row_norms: Array1<f64>,
    left: Array2<f64>,
    singular: Array1<f64>,
    right_transposed: Array2<f64>,
}

impl ActiveConstraintFaceGeometry {
    /// Numerical row rank of the row-normalized face.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Largest singular value of the row-normalized face.
    pub fn largest_singular_value(&self) -> f64 {
        self.singular.iter().fold(0.0_f64, |largest, &s| largest.max(s))
    }

    /// Smallest singular value retained by the rank decision. `0.0` for an
    /// empty rank, which `active_constraint_face_geometry` already refuses.
    pub fn smallest_retained_singular_value(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.singular[self.rank - 1]
        }
    }

    /// Tangent geometry of the face, decided by the same rank rule.
    pub fn tangent(&self) -> &ActiveConstraintTangentGeometry {
        &self.tangent
    }

    /// Consume the geometry for its tangent basis.
    pub fn into_tangent(self) -> ActiveConstraintTangentGeometry {
        self.tangent
    }

    /// Minimum-Euclidean-norm solution of `A δ = rhs` over the numerically
    /// identified row space.
    ///
    /// `rhs` is stated against the ORIGINAL (unnormalized) rows and is scaled
    /// here by the same row norms the factorization used, so a face and its
    /// positively rescaled twin produce the identical `δ`.
    ///
    /// Truncating the directions below the rank floor is what makes this
    /// total: those directions cannot be resolved by any step of bounded norm,
    /// so the honest answer is the solution that ignores them plus the
    /// residual they leave — reported, not swallowed.
    pub fn minimum_norm_particular(
        &self,
        rhs: &Array1<f64>,
    ) -> Result<ParticularFaceSolution, String> {
        let rows = self.normalized.nrows();
        let cols = self.normalized.ncols();
        if rhs.len() != rows {
            return Err(format!(
                "active constraint face affine system needs one right-hand side per face row \
                 (rows={rows}, rhs={})",
                rhs.len()
            ));
        }
        let scaled_rhs: Array1<f64> = rhs
            .iter()
            .zip(self.row_norms.iter())
            .map(|(value, norm)| value / norm)
            .collect();
        if scaled_rhs.iter().any(|value| !value.is_finite()) {
            return Err(
                "active constraint face right-hand side is non-finite in scaled-slack units"
                    .to_string(),
            );
        }
        let mut delta = Array1::<f64>::zeros(cols);
        for mode in 0..self.rank {
            let coefficient = self.left.column(mode).dot(&scaled_rhs) / self.singular[mode];
            delta.scaled_add(coefficient, &self.right_transposed.row(mode));
        }
        if delta.iter().any(|value| !value.is_finite()) {
            return Err("active constraint face particular solution is non-finite".to_string());
        }
        let residual = self.normalized.dot(&delta) - &scaled_rhs;
        let residual_inf = residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let delta_norm = delta.dot(&delta).sqrt();
        let rhs_norm = scaled_rhs.dot(&scaled_rhs).sqrt();
        // Backward-error bound for a truncated-SVD least-squares solve. The
        // rows are unit-normalized, so `‖A‖₂ ≤ √rows`, and a CONSISTENT face
        // leaves at most `O(eps)·(‖A‖‖δ‖ + ‖rhs‖)`. Same `100·eps·max(k,p)`
        // factor the rank floor above uses, and — like it — scale-covariant
        // with no absolute floor, so a face stated in tiny units is not
        // declared inconsistent for being small.
        let residual_tolerance = 100.0
            * f64::EPSILON
            * (rows.max(cols).max(1) as f64)
            * ((rows as f64).sqrt() * delta_norm + rhs_norm);
        Ok(ParticularFaceSolution {
            delta,
            residual_inf,
            residual_tolerance,
        })
    }
}

/// Factor a non-empty active constraint face once, for every affine question
/// a reduced-face solve asks of it. See [`ActiveConstraintFaceGeometry`].
pub fn active_constraint_face_geometry(
    a_act: &Array2<f64>,
) -> Result<ActiveConstraintFaceGeometry, String> {
    if a_act.nrows() == 0 {
        return Err("active constraint face geometry requires at least one row".to_string());
    }
    let p = a_act.ncols();
    if p == 0 {
        return Err("active constraint face geometry requires at least one coefficient".to_string());
    }
    let (normalized, row_norms) = normalize_active_face_rows(a_act);
    let (u, singular, vt) = normalized
        .svd(true, true)
        .map_err(|error| format!("active constraint face SVD failed: {error}"))?;
    let left = u.ok_or_else(|| "active constraint face SVD omitted U".to_string())?;
    let right_transposed = vt.ok_or_else(|| "active constraint face SVD omitted Vᵀ".to_string())?;
    let (rank, tangent) = tangent_from_row_factorization(&normalized, &singular, &right_transposed)?;
    Ok(ActiveConstraintFaceGeometry {
        rank,
        tangent,
        normalized,
        row_norms,
        left,
        singular,
        right_transposed,
    })
}

#[cfg(test)]
mod active_constraint_tangent_geometry_tests {
    use super::*;

    #[test]
    fn direct_rectangular_factorization_preserves_ill_conditioned_row_rank() {
        // The two independent rows have condition number O(1e8), so forming
        // AᵀA pushes their squared condition to machine precision. The true
        // tangent is exactly the third coordinate and must remain one-dimensional.
        let a = ndarray::array![[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-7, 0.0]];
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("direct SVD geometry")
        else {
            panic!("ill-conditioned rank-two face must have a tangent");
        };
        assert_eq!(z.dim(), (3, 1));
        assert!((z.column(0).dot(&z.column(0)) - 1.0).abs() < 1e-12);
        for row in a.rows() {
            assert!(row.dot(&z.column(0)).abs() / row.dot(&row).sqrt() < 1e-12);
        }
    }

    #[test]
    fn dependent_rows_produce_the_full_exact_null_space() {
        let a = ndarray::array![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("rank-one geometry")
        else {
            panic!("rank-one face in three dimensions must have a tangent");
        };
        assert_eq!(z.dim(), (3, 2));
        let gram = z.t().dot(&z);
        for i in 0..2 {
            for j in 0..2 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((gram[[i, j]] - target).abs() < 1e-12);
            }
        }
        assert!(a.dot(&z).iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn tangent_geometry_is_invariant_to_constraint_row_scaling() {
        let a = ndarray::array![[1e12, 1e12, 0.0], [1e-12, 1e-12 + 1e-19, 0.0]];
        let ActiveConstraintTangentGeometry::Tangent(z) =
            active_constraint_tangent_geometry(&a).expect("row-scaled geometry")
        else {
            panic!("two independent normalized rows must leave one tangent dimension");
        };
        assert_eq!(z.dim(), (3, 1));
        for row in a.rows() {
            assert!(row.dot(&z.column(0)).abs() / row.dot(&row).sqrt() < 1e-12);
        }
    }

    #[test]
    fn face_geometry_and_tangent_geometry_cannot_disagree_about_rank_2600() {
        // Two rows whose normalized independence is O(1e-8) and a third
        // coordinate nobody touches. Both entry points must report the same
        // rank and the same tangent dimension, because they now read the same
        // rule off the same factorization.
        let a = ndarray::array![[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-8, 0.0]];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        assert_eq!(geometry.rank(), 2);
        let ActiveConstraintTangentGeometry::Tangent(direct) =
            active_constraint_tangent_geometry(&a).expect("tangent geometry")
        else {
            panic!("rank-two face in three dimensions must have a tangent");
        };
        let ActiveConstraintTangentGeometry::Tangent(shared) = geometry.into_tangent() else {
            panic!("face geometry must agree that the face has a tangent");
        };
        assert_eq!(direct.dim(), shared.dim());
        // Both bases are built by the same twice-reorthogonalized completion
        // from the same rank, so they span the same line; assert the span
        // rather than the bits, since only one of the two SVD calls also asks
        // for `U` and the factorization is free to differ in the last digits.
        for row in a.rows() {
            for column in shared.columns() {
                assert!(row.dot(&column).abs() / row.dot(&row).sqrt() < 1e-12);
            }
        }
    }

    #[test]
    fn minimum_norm_particular_solves_a_consistent_face_and_reports_its_residual_2600() {
        // Rank-one face in three dimensions: `x = 2` with a redundant restated
        // copy at ten times the scale. The minimum-norm solution is the axis
        // point, and it must not depend on the redundancy or the row scaling.
        let a = ndarray::array![[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let rhs = ndarray::array![2.0, 20.0];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        assert_eq!(geometry.rank(), 1);
        let particular = geometry
            .minimum_norm_particular(&rhs)
            .expect("consistent rank-one face");
        assert!((particular.delta[0] - 2.0).abs() < 1e-12);
        assert!(particular.delta[1].abs() < 1e-12);
        assert!(particular.delta[2].abs() < 1e-12);
        assert!(
            particular.residual_inf <= particular.residual_tolerance,
            "a consistent face must not be reported inconsistent \
             (residual={:.6e}, tolerance={:.6e})",
            particular.residual_inf,
            particular.residual_tolerance
        );
    }

    #[test]
    fn minimum_norm_particular_names_an_inconsistent_face_instead_of_solving_it_2600() {
        // The same two parallel rows now demand contradictory offsets. No step
        // satisfies both, and that has to surface as a residual above the
        // backward-error bound rather than as a plausible-looking `delta`.
        let a = ndarray::array![[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let rhs = ndarray::array![2.0, 30.0];
        let geometry = active_constraint_face_geometry(&a).expect("face geometry");
        let particular = geometry
            .minimum_norm_particular(&rhs)
            .expect("least-squares answer still exists");
        assert!(
            particular.residual_inf > particular.residual_tolerance,
            "a contradictory face must exceed its own consistency bound \
             (residual={:.6e}, tolerance={:.6e})",
            particular.residual_inf,
            particular.residual_tolerance
        );
    }

    #[test]
    fn minimum_norm_particular_is_invariant_to_positive_row_rescaling_2600() {
        let a = ndarray::array![[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]];
        let rhs = ndarray::array![3.0, 5.0];
        let scale = ndarray::array![[1e9, 1e9, 0.0], [0.0, 1e-9, 1e-9]];
        let scaled_rhs = ndarray::array![3.0e9, 5.0e-9];
        let plain = active_constraint_face_geometry(&a)
            .expect("plain geometry")
            .minimum_norm_particular(&rhs)
            .expect("plain particular");
        let rescaled = active_constraint_face_geometry(&scale)
            .expect("rescaled geometry")
            .minimum_norm_particular(&scaled_rhs)
            .expect("rescaled particular");
        for (left, right) in plain.delta.iter().zip(rescaled.delta.iter()) {
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(1.0),
                "row rescaling moved the particular solution: {left:.17e} vs {right:.17e}"
            );
        }
    }
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
    gam_linalg::faer_ndarray::fast_abt(&vs, &op.eigenvectors)
}

/// Tangent-projected `HessianFactorization` adapter. Wraps an `m × m`
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

impl HessianFactorization for TangentProjectedHessianOperator {
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
    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        // Matrix-free tangent projection of an operator-backed Hessian drift.
        //
        // The `HessianFactorization` trait default densifies `op` (`op.to_dense()`,
        // p forward HVPs + a p×p transient) and then evaluates
        // `trace_logdet_gradient`, which internally forms `Zᵀ Bdense Z`. For a
        // spectral tangent operator `logdet_traces_match_hinv_kernel()` is
        // false, so that default never reaches the Hutch++ fast path and
        // unconditionally hits the warn-and-materialize branch — the dominant
        // source of `trace_logdet_operator: materializing implicit
        // HyperOperator` spam (and O(p²) work per outer eval per penalty) on
        // every shape-constrained REML fit.
        //
        // `Zᵀ B Z` is exactly `op.projected_matrix(Z) = Zᵀ (B·Z)`, where `B·Z`
        // is `op.mul_mat(Z)` — the operator's own matrix-free action, only
        // m ≤ p HVPs and no dense p×p B. The two are algebraically identical
        // (both have entry `z_iᵀ B z_j`), so this override changes only the
        // arithmetic path, never the value: `tr(G_ε(H_T) · ZᵀBZ)` via the
        // wrapped spectral logdet kernel.
        let zbz = op.projected_matrix(&self.z);
        self.h_t_op.trace_logdet_gradient(&zbz)
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

/// Borrowing adapter that lets the constrained-response `InnerSolution` reuse
/// the original `HessianDerivativeProvider` without taking ownership. The
/// provider's drift matrices stay in the full coefficient space because the
/// LAML value and trace kernel stay there; only the mode-response vectors fed
/// into those drifts are tangent-restricted.
pub(crate) struct BorrowedDerivProvider<'a>(&'a dyn HessianDerivativeProvider);

impl<'a> HessianDerivativeProvider for BorrowedDerivProvider<'a> {
    fn mode_response_rhs_correction(&self) -> Option<ModeResponseRhsCorrectionFn> {
        self.0.mode_response_rhs_correction()
    }
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
    fn family_outer_hessian_operator(&self) -> Option<Arc<dyn gam_problem::HessianOperator>> {
        self.0.family_outer_hessian_operator()
    }
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        self.0.scalar_glm_ingredients()
    }
}

/// A zero-dimensional inverse on the tangent of a fully pinned mode.
///
/// This operator is installed only as InnerSolution::mode_response_op; the
/// full-space Hessian remains the sole owner of the LAML value and traces.
/// Every response of a mode with null(A_act) = {0} is exactly zero.
struct FullyPinnedModeResponse {
    dimension: usize,
}

impl HessianFactorization for FullyPinnedModeResponse {
    fn logdet(&self) -> f64 {
        0.0
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), (self.dimension, self.dimension));
        0.0
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        assert_eq!(rhs.len(), self.dimension);
        Array1::zeros(self.dimension)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        assert_eq!(rhs.nrows(), self.dimension);
        Array2::zeros(rhs.raw_dim())
    }

    fn dim(&self) -> usize {
        self.dimension
    }
    fn active_rank(&self) -> usize {
        0
    }
}

/// If the inner solution carries a non-empty active inequality-constraint
/// set, keep the LAML criterion on the fitted model's full coefficient space
/// and restrict only the implicit response of its constrained mode.
///
/// The constraint polytope is part of the model, but the rows a numerical QP
/// happens to list as active are not a new statistical model and cannot change
/// the dimension of its Laplace integral. In particular, listing a row that is
/// tight with zero multiplier changes neither the mode nor the likelihood, so
/// it must not change 1/2 log|H(beta_hat)| - 1/2 log|S(rho)|+.
///
/// Active geometry enters through the derivative of the constrained mode. With
/// Z an orthonormal basis of null(A_act),
///
/// d beta_hat / d theta = -Z (Z' M_true Z)^-1 Z' d g / d theta,
///
/// where M_true is the inner stationarity system (which may deliberately
/// differ from the log-determinant operator; #2612). The borrowed solution
/// therefore retains every full-space value/trace object and installs only this
/// tangent-restricted mode-response operator. Clearing active_constraints on
/// it prevents recursion; the constraint's first-order effect is already
/// represented by the installed operator.
///
/// Returns Ok(None) when no active constraints are present and Ok(Some(result))
/// after evaluating the full-space criterion with the constrained response. A
/// backend that cannot materialize the true response curvature returns a named
/// error rather than silently differentiating through the log-determinant
/// curvature.
pub(crate) fn try_tangent_projected_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<Option<RemlLamlResult>, String> {
    let block = match solution.active_constraints.as_ref() {
        Some(block) if block.a.nrows() > 0 => block,
        _ => return Ok(None),
    };
    let p = solution.beta.len();
    if block.a.ncols() != p {
        return Err(format!(
            "active_constraints.a has {} columns but beta is {}-dim",
            block.a.ncols(),
            p
        ));
    }

    let constrained_mode_response: Arc<dyn HessianFactorization> =
        match active_constraint_tangent_geometry(&block.a)? {
            ActiveConstraintTangentGeometry::FullyPinned => {
                Arc::new(FullyPinnedModeResponse { dimension: p })
            }
            ActiveConstraintTangentGeometry::Tangent(z) => {
                // Differentiate the stationarity system the inner solve
                // actually used, not the operator that owns the Laplace
                // log-determinant. The two differ under a Jeffreys completion
                // (#2612).
                let response_full = solution
                    .mode_response_operator()
                    .assemble_h_dense_for_tangent_projection()
                    .map_err(|error| {
                        format!(
                            "active-constraint mode response needs a dense stationarity \
                             curvature: {error}"
                        )
                    })?;
                // #979: locate the smallest eigenvalue the criterion's
                // pseudo-log-determinant keeps relative to this face. A kept
                // direction normal to the face prices curvature the constrained
                // mode never explores; one inside the tangent disagrees with the
                // certified tangent curvature there.
                if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                    let inverse = &kernel.h_proj_inverse;
                    let rank = inverse.nrows();
                    let diagonal = (0..rank)
                        .all(|i| (0..rank).all(|j| i == j || inverse[[i, j]] == 0.0));
                    let shaped = inverse.ncols() == rank
                        && kernel.u_s.ncols() == rank
                        && kernel.u_s.nrows() == z.nrows();
                    let smallest = (0..rank).max_by(|&left, &right| {
                        inverse[[left, left]].total_cmp(&inverse[[right, right]])
                    });
                    if let (true, true, Some(column)) = (diagonal, shaped, smallest) {
                        let direction = kernel.u_s.column(column);
                        let tangent_part = z.t().dot(&direction);
                        // The value prices the log-determinant operator; the inner
                        // certificate prices the stationarity curvature. Report both
                        // spectra on the full space and on the face.
                        let smallest_eigenvalue = |matrix: &Array2<f64>| {
                            DenseSpectralOperator::from_symmetric(matrix).ok().map(|operator| {
                                operator.raw_spectrum().iter().copied().fold(f64::INFINITY, f64::min)
                            })
                        };
                        let value_full =
                            solution.hessian_op.assemble_h_dense_for_tangent_projection().ok();
                        let value_min = value_full.as_ref().and_then(|matrix| smallest_eigenvalue(matrix));
                        let value_tangent_min = value_full
                            .as_ref()
                            .and_then(|matrix| smallest_eigenvalue(&z.t().dot(matrix).dot(&z)));
                        let true_min = smallest_eigenvalue(&response_full);
                        let true_tangent_min = smallest_eigenvalue(&z.t().dot(&response_full).dot(&z));
                        log::info!(
                            "[979-FACE-LOGDET] kept_rank={rank}/{} tangent_dim={} \
                             sigma_min_kept={:.6e} normal_fraction={:.3e} \
                             value_min={value_min:?} value_tangent_min={value_tangent_min:?} \
                             true_min={true_min:?} true_tangent_min={true_tangent_min:?}",
                            z.nrows(),
                            z.ncols(),
                            1.0 / inverse[[column, column]],
                            direction.dot(&direction) - tangent_part.dot(&tangent_part),
                        );
                    }
                }
                let response_tangent = z.t().dot(&response_full).dot(&z);
                let response_tangent_op =
                    DenseSpectralOperator::from_symmetric(&response_tangent).map_err(|error| {
                        format!(
                            "constrained mode-response eigendecomposition failed: {error}"
                        )
                    })?;
                Arc::new(TangentProjectedHessianOperator {
                    z,
                    h_t_op: response_tangent_op,
                })
            }
        };

    let constrained = InnerSolution {
        log_likelihood: solution.log_likelihood,
        penalty_quadratic: solution.penalty_quadratic,
        // Value and trace geometry stay on the fitted model's full coefficient
        // space. Only the constrained mode response is tangent-restricted.
        hessian_op: Arc::clone(&solution.hessian_op),
        mode_response_op: Some(constrained_mode_response),
        beta: solution.beta.clone(),
        penalty_coords: solution.penalty_coords.clone(),
        penalty_logdet: solution.penalty_logdet.clone(),
        deriv_provider: Box::new(BorrowedDerivProvider(solution.deriv_provider.as_ref())),
        firth: solution.firth.clone(),
        hessian_logdet_correction: solution.hessian_logdet_correction,
        penalty_subspace_trace: solution.penalty_subspace_trace.clone(),
        rho_curvature_scale: solution.rho_curvature_scale,
        rho_prior: solution.rho_prior.clone(),
        n_observations: solution.n_observations,
        nullspace_dim: solution.nullspace_dim,
        gaussian_weight_log_sum_half: solution.gaussian_weight_log_sum_half,
        dp_floor_scale: solution.dp_floor_scale,
        dispersion: solution.dispersion.clone(),
        ext_coords: solution.ext_coords.clone(),
        ext_coord_pair_fn: solution.ext_coord_pair_fn.clone(),
        rho_ext_pair_fn: solution.rho_ext_pair_fn.clone(),
        contracted_psi_second_order: solution.contracted_psi_second_order.clone(),
        fixed_drift_deriv: solution.fixed_drift_deriv.clone(),
        barrier_config: solution.barrier_config.clone(),
        kkt_residual: solution.kkt_residual.clone(),
        // Prevent recursive constrained-response installation. The operator
        // above already carries the active geometry.
        active_constraints: None,
        stochastic_trace_state: solution.stochastic_trace_state.clone(),
    };
    reml_laml_evaluate(&constrained, rho, mode, prior_cost_gradient).map(Some)
}

#[cfg(test)]
mod profiled_gaussian_residual_dof_tests {
    use super::profiled_gaussian_residual_dof;

    /// The positive control: the refusal this replaced `.max(DENOM_RIDGE)`
    /// with must actually fire, at the step of one in an integer where the old
    /// clamp used to take over.
    #[test]
    fn refuses_at_and_below_zero_residual_dof_and_accepts_one() {
        assert_eq!(
            profiled_gaussian_residual_dof(8, 7.0).expect("nu = 1 is a fittable model"),
            1.0
        );
        for nullspace_dim in [8.0, 9.0, 23.0] {
            let refusal = profiled_gaussian_residual_dof(8, nullspace_dim)
                .expect_err("nu <= 0 has no profiled scale and must refuse");
            assert!(
                refusal.contains("residual degrees of freedom must be positive"),
                "refusal must name the condition, got: {refusal}"
            );
        }
    }

    /// The old clamp's whole justification was "denominator safety", and the
    /// value it delivered at `nu <= 0` was `1e-8`, eight orders of magnitude
    /// below the `nu = 1` it neighbours across a step of one in an integer.
    /// Nothing may return a positive number in that regime again.
    #[test]
    fn no_fabricated_positive_denominator_below_one() {
        assert!(profiled_gaussian_residual_dof(0, 0.0).is_err());
        assert!(profiled_gaussian_residual_dof(3, 3.0).is_err());
        assert_eq!(
            profiled_gaussian_residual_dof(3, 2.0).expect("nu = 1"),
            1.0
        );
    }
}
