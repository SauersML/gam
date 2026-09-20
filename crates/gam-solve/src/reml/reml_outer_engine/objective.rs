use super::*;
use crate::estimate::reml::atoms::CriterionAtom;
use crate::estimate::smooth_floor_dp;

/// `tr(G_ε(H) · λ_k S_k)` from the coordinate's penalty ROOT, when it has one.
///
/// `S_k = R_kᵀR_k`, so this is `‖√λ_k R_k · G_block‖_F²`. The equivalent
/// contraction against the SQUARED block is `O(ε·κ(H))` on a trace the theory
/// bounds by `rank(S_k)` — see
/// [`HessianFactorization::trace_logdet_block_root`] (#2644).
///
/// An operator priced from the Hessian's own root reads coordinate `penalty`'s
/// trace off that root's left singular vectors instead (#2959 D2); see
/// [`DenseSpectralOperator::root_penalty_mode_terms`].
///
/// `None` only when `lambda` is negative or not finite.
fn penalty_logdet_trace_from_root_opt(
    hop: &dyn HessianFactorization,
    penalty: usize,
    coord: &gam_problem::PenaltyCoordinate,
    lambda: f64,
) -> Option<f64> {
    if let Some(terms) = hop
        .as_exact_dense_spectral()
        .and_then(|ds| ds.root_penalty_mode_terms(penalty, lambda))
    {
        return Some(terms.sum());
    }
    let (root, start, end) = coord.scaled_block_root(lambda)?;
    Some(hop.trace_logdet_block_root(root.view(), start, end))
}

/// [`penalty_logdet_trace_from_root_opt`] with the squared-block fallback for a
/// scale that admits no real root.
fn penalty_logdet_trace_from_root(
    hop: &dyn HessianFactorization,
    penalty: usize,
    coord: &gam_problem::PenaltyCoordinate,
    lambda: f64,
) -> f64 {
    penalty_logdet_trace_from_root_opt(hop, penalty, coord, lambda).unwrap_or_else(|| {
        let (block, start, end) = coord.scaled_block_local(1.0);
        hop.trace_logdet_block_local(&block, lambda, start, end)
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  The single evaluator
// ═══════════════════════════════════════════════════════════════════════════

/// Unified REML/LAML evaluation.
///
/// This is the SINGLE implementation of the outer objective. It handles:
/// - Gaussian REML with profiled scale
/// - Non-Gaussian LAML with fixed dispersion
/// - Any backend (dense spectral, sparse Cholesky, block-coupled)
/// - Any family (Gaussian, GLM, GAMLSS, survival, link wiggles)
///
/// Cost and gradient share intermediates by construction — they are computed
/// in the same function scope, using the same `HessianFactorization`, the same
/// penalty derivatives, and the same coefficients. Drift between cost and
/// gradient is structurally impossible because there is no second function.
///
/// # Observed information requirement (see response.md Section 3)
///
/// The Laplace approximation to the marginal likelihood integral
///   int exp(-F(beta)) dbeta  ~  exp(-F(beta_hat)) * (2pi)^{p/2} / sqrt(|H_obs|)
/// requires H_obs = nabla^2 F(beta_hat), the **observed** (actual) Hessian at
/// the mode --- NOT the expected Fisher information. Replacing H_obs with
/// E[H] changes the quadratic approximation itself, yielding a PQL-type
/// surrogate rather than the true Laplace/LAML criterion.
///
/// For this evaluator, the `solution.hessian_op` MUST encode log|H_obs| and
/// provide traces tr(H_obs^{-1} A_k) using the observed Hessian. Callers
/// (runtime.rs, joint.rs) are responsible for constructing H from the
/// observed-information weights W_obs = W_Fisher - (y-mu)*B at the mode.
///
/// The **mixed strategy** is valid and deliberately used here:
/// - The inner P-IRLS solver may use Fisher scoring (expected information)
///   as its iteration matrix --- any convergent algorithm finds the same mode.
/// - The outer REML criterion uses the observed Hessian at that mode.
/// This is correct because the inner algorithm is just a solver; only the
/// outer log|H| and trace terms define the Laplace approximation.
///
/// For canonical links (for example logit-Binomial and log-Poisson), observed
/// equals expected, so no correction is needed. For non-canonical links
/// (including probit, cloglog, SAS, mixture/flexible, and Gamma-log), the observed weight includes a
/// residual-dependent correction:
///   W_obs = W_Fisher - (y - mu) * B,
///   B = (h'' V - h'^2 V') / (phi V^2)
/// and the c/d arrays (dW/deta, d^2W/deta^2) similarly include observed
/// corrections. These are computed by `compute_observed_hessian_curvature_arrays`
/// in pirls.rs and flow through `PirlsResult` into the `InnerSolution`.
///
/// # Arguments
/// - `solution`: The converged inner state (beta_hat, H_obs, penalties, corrections).
/// - `rho`: Log smoothing parameters (rho_k = log lambda_k).
/// - `mode`: What to compute (value only, value+gradient, or all three).
/// - `prior_cost_gradient`: Optional soft prior on rho (value, gradient, optional Hessian).
pub(crate) fn reml_laml_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<RemlLamlResult, RemlLamlError> {
    // Validate the complete raw entry vector before tangent recursion or any
    // objective work. This makes every downstream exponential dominated by a
    // deterministic, smallest-coordinate refusal rather than optimizer bounds.
    let lambdas = gam_problem::checked_exp_log_strengths(rho.iter().copied())
        .map_err(|error| format!("REML/LAML rho: {error}"))?;
    // Constraint-tangent-space dispatch. When the inner converged at a
    // constrained-stationary point with a non-empty active inequality set,
    // the principled LAML outer objective lives on `null(A_act)`. Build a
    // tangent-projected `InnerSolution` (wrapped operator + recomputed
    // penalty logdet) and recurse with `active_constraints = None`. See
    // the header comment on `try_tangent_projected_evaluate`.
    if let Some(result) =
        try_tangent_projected_evaluate(solution, rho, mode, prior_cost_gradient.clone())?
    {
        return Ok(result);
    }
    let cost_phase_start = std::time::Instant::now();
    // Enforce the `rho_curvature_scale` contract documented on
    // `InnerSolution::rho_curvature_scale`.  A non-positive or non-finite
    // scale silently corrupts BOTH the cost (through `hessian_logdet_correction`
    // chosen to match `−p·log(s)`) and the gradient trace (through
    // `curvature_lambdas = s · λ`); refuse to evaluate rather than emit a
    // garbage outer-derivative pair.  See issue #200.
    if !solution.rho_curvature_scale.is_finite() || solution.rho_curvature_scale <= 0.0 {
        return Err(RemlError::NonFiniteValue {
            reason: format!(
                "rho_curvature_scale must be strictly positive and finite (got {}); the \
                 unified evaluator scales the gradient drift by this factor and relies on \
                 the caller having scaled `hessian_op` by the same factor with a matching \
                 `hessian_logdet_correction = −p·log(scale)` — see issue #200",
                solution.rho_curvature_scale,
            ),
        }
        .into());
    }
    let k = rho.len();
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();
    let hop = &*solution.hessian_op;
    let upper_active_rho = active_upper_rho_mask(rho);

    // ─── Shared intermediates (computed once, used by both cost and gradient) ───

    // `hop.logdet()` is the operator's own determinant; the two corrections on
    // top of it are DIFFERENT objects with different transformation laws.
    // `hessian_logdet_correction` un-scales a uniform curvature rescale
    // (`−p·log s`) and belongs to the solution. The pseudo-logdet route's
    // correction belongs to the KERNEL that differentiates it, and is read from
    // there, so a lane that drops the kernel cannot keep the value it corrected
    // (#2765).
    let log_det_h = hop.logdet()
        + solution.hessian_logdet_correction
        + solution
            .penalty_subspace_trace
            .as_ref()
            .map_or(0.0, |kernel| kernel.logdet_correction);
    let log_det_s = solution.penalty_logdet.value;
    // The penalty-quadratic term `½ β̂ᵀSβ̂` enters the cost ONLY through this
    // atom (#931): its `value()` carries the production stable-basis scalar
    // `½ · stable_penalty_term`, the SAME `value()` the full gradient-bearing
    // `PenaltyQuadAtom` (built downstream for `rho_frozen_d1`) exposes. The cost
    // can no longer read a raw `solution.penalty_quadratic` that the gradient
    // atom does not own — value and ρ-derivative are projections of one atom.
    let penalty_quad_value_atom = crate::estimate::reml::atoms::PenaltyQuadAtom::stable_value_only(
        0.5 * solution.penalty_quadratic,
    );
    let penalty_quad_value = penalty_quad_value_atom.value();
    let (cost, profiled_scale, dp_cgrad, _dp_cgrad2) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            // Gaussian REML with profiled scale:
            //   V(ρ) = D_p/(2φ̂) + ½ log|H| − ½ log|S|₊
            //          + ((n−M_p)/2) log(2πφ̂) − ½ Σᵢ log(wᵢ)
            // where D_p = deviance + penalty, φ̂ = D_p/(n−M_p).
            //
            // The final `− ½ Σ log(wᵢ)` is the per-observation Gaussian
            // normalization constant that the log-likelihood deliberately
            // drops (`Var(yᵢ) = φ/wᵢ` ⇒ ½ Σ log(2π φ/wᵢ) =
            // (n/2) log(2πφ) − ½ Σ log wᵢ). It is constant in ρ — it does NOT
            // move the argmin — but WITHOUT it the cost VALUE is not invariant
            // to a global prior-weight rescale `w → c·w`: the invariance-
            // preserving smoothing `λ → c·λ` keeps the cost shape fixed yet
            // inflates its value by `(n/2) log c`, and that inflation breaks
            // the exact weight-scale invariance of λ̂ / EDF / fit (issue #877).
            // Adding `−½ Σ log(wᵢ)` (which contributes `−(n/2) log c` under the
            // rescale) cancels the inflation, so the ProfiledGaussian cost
            // VALUE — not just its argmin — is exactly invariant, matching mgcv
            // (the profiled σ̂² absorbs the c factor).
            // `dp_raw = deviance + penalty = −2ℓ + βᵀSβ`. The atom carries the
            // ½-scaled penalty energy, so the deviance-scale (un-halved) penalty
            // is `2 · penalty_quad_value`.
            let dp_raw = -2.0 * solution.log_likelihood + 2.0 * penalty_quad_value;
            let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
            let denom =
                profiled_gaussian_residual_dof(solution.n_observations, solution.nullspace_dim)?;
            let phi = dp_c / denom;

            let cost = dp_c / (2.0 * phi)
                + 0.5 * (log_det_h - log_det_s)
                + (denom / 2.0) * (2.0 * std::f64::consts::PI * phi).ln()
                - solution.gaussian_weight_log_sum_half;

            // #2454: the `fixed_beta` gradient channel is
            // `dp_cgrad · (½ λ_k q_k) / φ`, so the audit needs the two
            // penalty-energy spellings and the three scalars that connect
            // them to it.
            if crate::estimate::outer_eval_capture::rho_outer_audit_enabled() {
                let block_sum: f64 = lambdas
                    .iter()
                    .zip(solution.penalty_coords.iter())
                    .map(|(&lambda, coord)| lambda * coord.shifted_quadratic(&solution.beta, 1.0))
                    .sum();
                crate::estimate::outer_eval_capture::record_rho_penalty_energy(
                    crate::estimate::outer_eval_capture::PenaltyEnergyAudit {
                        stable: solution.penalty_quadratic,
                        block_sum,
                        dp_raw,
                        dp_floored: dp_c,
                        dp_cgrad,
                        phi,
                    },
                );
            }

            (cost, phi, dp_cgrad, dp_cgrad2)
        }
        DispersionHandling::Fixed {
            phi,
            include_logdet_h,
            include_logdet_s,
        } => {
            // Fixed-dispersion Laplace / maximum penalized likelihood:
            //   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂
            //         + [½ log|H| + frozen-curvature TK − Firth]  if include_logdet_h
            //         − [½ log|S|₊]               if include_logdet_s
            //
            // The additive Gaussian normalization constant 0.5 * M * log(2πφ)
            // is intentionally omitted here. It does not affect outer
            // derivatives, and the custom-family exact paths already define
            // their scalar objective without it. Keeping the fixed-dispersion
            // evaluator aligned with those exact paths avoids objective drift
            // between the unified and direct custom-family implementations.
            //
            // Pair-subtract `log|H| − log|S|_+` before scaling by 0.5 and
            // summing with the rest, mirroring the profiled-Gaussian cost
            // expression above.  The pair `(log|H|, log|S|_+)` has nearly-
            // identical ρ-motion at a rank-deficient optimum (the analytic
            // gradient is their difference, which is tiny), so subtracting
            // them FIRST preserves the leading-order cancellation in f64
            // precision; adding them to `cost` independently would bury
            // the difference below ~ULP(cost) ≈ f64::EPSILON * cost.
            let logdet_pair_h = if *include_logdet_h { log_det_h } else { 0.0 };
            let logdet_pair_s = if *include_logdet_s { log_det_s } else { 0.0 };
            let cost_logdet_diff = 0.5 * (logdet_pair_h - logdet_pair_s);
            let mut cost = cost_logdet_diff + (-solution.log_likelihood) + penalty_quad_value;
            if *include_logdet_h {
                // Firth `−½ log|J|`: VALUE here, ρ-DERIVATIVE folded into the
                // per-coordinate LAML trace (`a_i`) below — paired by
                // construction through the Jeffreys operator. Tierney-Kadane
                // is applied by `RemlState::apply_theta_correction_atom_to_result`
                // after the unified evaluator returns, so this core evaluator
                // no longer carries a loose TK value/gradient pair.
                cost -= solution
                    .firth
                    .as_ref()
                    .map_or(0.0, ExactJeffreysTerm::value);
            }
            // #2454/#2644: record the SAME two penalty-energy spellings here.
            //
            // The audit — and therefore the structural half of the #2454 gate
            // ("the criterion's penalty energy and the gradient's block sum are
            // ONE quantity") — used to be emitted only from the
            // profiled-Gaussian arm above. Every non-Gaussian family lands on
            // THIS arm, so the whole binomial/Poisson/Gamma/survival half of
            // the criterion was outside the instrument, even though the
            // exposure is identical and link-independent: `cost` reads the
            // stable-basis `penalty_quad_value` while the gradient's
            // `fixed_beta` channel is a per-block projection of `Σ_k λ_k q_k`
            // rebuilt from the outer penalty coordinates. If those two spellings
            // are not the same number, the ρ-derivative multiplies the
            // difference by `λ_k`.
            //
            // `dp_cgrad = 1.0` and `phi = 1.0` are not placeholders: the
            // documented reconstruction of the channel is `dp_cgrad · a_k / phi`,
            // and on fixed dispersion that channel is exactly `a_k` (there is no
            // smooth deviance floor and no profiled scale in it — the `phi` of
            // `DispersionHandling::Fixed` scales the LIKELIHOOD, not this term).
            // Recording 1.0/1.0 keeps one reconstruction formula true on both
            // arms rather than making the audit's meaning branch-dependent.
            // `dp_raw`/`dp_floored` carry the deviance-scale penalized deviance
            // for context; no criterion term on this arm reads them.
            if crate::estimate::outer_eval_capture::rho_outer_audit_enabled() {
                let block_sum: f64 = lambdas
                    .iter()
                    .zip(solution.penalty_coords.iter())
                    .map(|(&lambda, coord)| lambda * coord.shifted_quadratic(&solution.beta, 1.0))
                    .sum();
                let dp_raw = -2.0 * solution.log_likelihood + 2.0 * penalty_quad_value;
                crate::estimate::outer_eval_capture::record_rho_penalty_energy(
                    crate::estimate::outer_eval_capture::PenaltyEnergyAudit {
                        stable: solution.penalty_quadratic,
                        block_sum,
                        dp_raw,
                        dp_floored: dp_raw,
                        dp_cgrad: 1.0,
                        phi: 1.0,
                    },
                );
            }
            (cost, *phi, 0.0, 0.0)
        }
    };

    // Add prior.
    let mut cost = match &prior_cost_gradient {
        Some((pc, _, _)) => cost + pc,
        None => cost,
    };

    // Add log-barrier cost for monotonicity-constrained coefficients.
    // `barrier_cost` returns `+∞` on infeasible β by contract (continuous
    // extension of `−τ Σ log Δ` past the boundary). Propagating that `+∞`
    // through `cost` lets outer line-search reject the infeasible step by
    // ordinary scalar comparison — no `Err` channel required.
    if let Some(ref barrier_cfg) = solution.barrier_config {
        cost += barrier_cfg.barrier_cost(&solution.beta);
    }

    // ─── Implicit-function-theorem cost correction ───
    //
    // Define β*(ρ) as the exact inner optimum (g_β(β*, ρ) = 0).  The outer
    // objective we *want* is V(β*(ρ), ρ); what the envelope formula above
    // computes is V(β̂, ρ) at the inner-returned β̂ ≠ β*.  First-order
    // implicit-function theorem gives
    //   β* − β̂ ≈ −H⁻¹ r,   r := ∇_β L_pen(β̂) = S(λ)β̂ − ∇ℓ(β̂)
    //   V(β*) ≈ V(β̂) + (∂V/∂β)ᵀ(β* − β̂) = V(β̂) + rᵀ · (−H⁻¹ r)
    //         = V(β̂) − ½ rᵀ H⁻¹ r            (using ∂V/∂β = r at β̂, second-
    //                                          order expansion of V symmetric
    //                                          in (β* − β̂)).
    //
    // The cost correction strictly vanishes when the inner reached exact
    // KKT (r = 0).  When the inner exits with a certified small residual it
    // absorbs the leading error; the gradient and Hessian corrections below are
    // the exact first and second ρ derivatives of this same scalar Newton
    // correction under fixed-dispersion LAML. If a projected-Hessian kernel
    // would need to drop a residual component larger than the inner KKT
    // tolerance band, the evaluator rejects the state as a contract violation
    // instead of manufacturing a range-only certificate.
    //
    // Filter: callers populate `kkt_residual` only on convergent inner paths.
    // The [`ProjectedKktResidual`] newtype lifts the projection invariant into
    // the type system (callers cannot construct one without going through the
    // active-set-aware projection helper), so the only thing left to validate
    // here is the length match against the Hessian operator. `None` means the
    // caller is presenting an exact-KKT mode and the envelope identities are
    // already valid.
    let kkt_residual_vec: Option<std::borrow::Cow<'_, Array1<f64>>> =
        match solution.kkt_residual.as_ref() {
            Some(residual) => {
                let r = residual.as_array();
                if r.len() != hop.dim() {
                    return Err(RemlError::DimensionMismatch {
                        reason: format!(
                            "projected KKT residual length mismatch: got {}, expected {}",
                            r.len(),
                            hop.dim()
                        ),
                    }
                    .into());
                }
                if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                    let reduced = residual
                        .projected_into_reduced_range(kernel)
                        .map_err(|reason| RemlError::ContractViolation { reason })?;
                    Some(std::borrow::Cow::Owned(reduced.as_array().clone()))
                } else {
                    Some(std::borrow::Cow::Borrowed(r))
                }
            }
            None => None,
        };
    let kkt_residual_correction_active = kkt_residual_vec.is_some()
        && matches!(solution.dispersion, DispersionHandling::Fixed { .. });
    // One-shot structured log of the IFT gate. Debug-level so it doesn't
    // spam normal runs but is immediately greppable when debugging an
    // envelope-gradient consistency failure (search for `[ift-gate]`).
    log::trace!(
        "[ift-gate] kkt_residual.is_some()={} kkt_residual.subspace={:?} dispersion={} correction_active={} subspace_trace.is_some()={} hop.dim()={} k={}",
        solution.kkt_residual.is_some(),
        solution
            .kkt_residual
            .as_ref()
            .map(ProjectedKktResidual::subspace),
        match &solution.dispersion {
            DispersionHandling::Fixed { .. } => "Fixed",
            DispersionHandling::ProfiledGaussian => "ProfiledGaussian",
        },
        kkt_residual_correction_active,
        solution.penalty_subspace_trace.is_some(),
        hop.dim(),
        k,
    );
    // #931 pass 2 / #2612: the ONE mode-response kernel of this evaluation. The
    // mode responses `v_k`, the cost-side IFT correction below and that
    // correction's θ-derivatives (`compute_kkt_residual_theta_corrections`) are
    // all one-step Newton displacements of the INNER stationarity system, so
    // every one of them reads this object, built on that system's own operator
    // `mode_response_operator()`.
    let mode_kernel = ThetaModeResponseKernel::select(
        solution.penalty_subspace_trace.as_deref(),
        solution.active_constraints.as_deref(),
        solution.mode_response_operator(),
    );
    // gam#2765 / gam#979: the normalizer is the Gaussian integral of a quadratic model about the
    // mode, so a mode whose softest direction leaves the Laplace series without a leading term is
    // at a fold, however well it is solved. One verdict at one point refuses the value and every
    // derivative alike, through the error channel, before anything is built on the mode.
    //
    // The verdict reads the softest curvature against its rounding band alone, so an evaluation
    // prices no `t₃`: that is one directional drift of the log-determinant operator, a full row
    // pass, for a record no consumer of the evaluation reads (#979: +20.5 s on the n=2000 BMS flex
    // smoke fit, job 1267301).
    if let Some(span) = mode_kernel.inverted_span() {
        let fold = grade_inner_mode_fold(&span, solution.rho_curvature_scale, None).map_err(
            |reason| RemlError::ContractViolation {
                reason: format!("inner-mode fold verdict (gam#2765): {reason}"),
            },
        )?;
        log::debug!("[inner-mode fold] {fold}");
        if !fold.is_valid() {
            log::debug!("[inner-mode fold] refusing this trial point: {fold}");
            return Err(RemlLamlError::InnerModeFold(fold));
        }
    }
    // #2954: the factor `log|H_β|` is read from, for the certificate's band on
    // the criterion's value. Where a kernel replaces the operator's determinant,
    // the operator's bound is not the criterion's (`determinant_forward_error`).
    if crate::estimate::outer_eval_capture::certificate_parts_capture_enabled()
        && let Some(logdet_forward_error) = match solution.penalty_subspace_trace.as_ref() {
            Some(kernel) => kernel.determinant_forward_error(hop.logdet_forward_error()),
            None => hop.logdet_forward_error(),
        }
    {
        crate::estimate::outer_eval_capture::record_certificate_inner_factor(
            crate::estimate::outer_eval_capture::InnerFactorCondition {
                logdet_forward_error,
            },
        );
    }
    let mut ift_residual_energy: Option<f64> = None;
    let mut inner_polish_step: Option<Array1<f64>> = None;
    if let Some(r) = kkt_residual_vec
        .as_ref()
        .filter(|_| kkt_residual_correction_active)
        .map(|r| r.as_ref())
    {
        // Cost-side IFT correction `−½ rᵀ w`, where `w` is the returned mode's
        // pending one-step displacement `β̂ − β*`. It is shared between the
        // inner-objective residual energy `−½ rᵀ w` and the moving-Hessian
        // log-det response below.
        //
        // When the rank-deficient LAML fix is active (`penalty_subspace_trace =
        // Some`), `r` has already been reduced into the kernel's identified range
        // (`projected_into_reduced_range` above), so no component on the cost's
        // gauge directions reaches the inverse — the large-scale survival
        // marginal-slope noise amplification that range reduction exists to stop.
        //
        // The displacement itself belongs to the stationarity system the inner
        // solve iterated, not to the object the log-det is priced on. Before
        // #2695 the projected lane applied `penalty_subspace_trace`'s own
        // pseudo-inverse, built on `hessian_op` = H + S_λ + H_Φ, while the mode
        // responses and the `full_h` lane solved on `mode_response_operator()`,
        // which carries the Jeffreys completion when one exists. On the #2904
        // SAS location-scale FD pin (Jeffreys-armed, projected lane) that direction
        // was (−8.1e-8, −6.0e-7, −9.2e-7) against a measured β̂ − β* of (1.18e-7,
        // 8.2e-9, −3.05e-8), so the moving-Hessian term was 10.26× the log-det
        // displacement it corrects, and the ρ0/ρ1 central differences read
        // 0.7806/−0.2109 against the analytic 0.7520/−0.1551 (pool jobs 1112782,
        // 1114313).
        let w = mode_kernel.respond_one(r);
        let cost_correction = -0.5_f64 * r.view().dot(&w);
        let (polish_step_for_warm_start, branch) = if solution.penalty_subspace_trace.is_some() {
            (None, "projected")
        } else {
            (Some(w.clone()), "full_h")
        };
        // MOVING-HESSIAN LOG-DET RESPONSE (gam#1395). The outer criterion
        // `V(β) = −ℓ(β) + ½βᵀSβ + ½log|H(β)+S| − ½log|S|` has the β-gradient
        // `∇_βV = r + g_ld`, where `r = ∇_β(−ℓ+½βᵀSβ)` is the inner KKT residual
        // and `g_ld_j = ½ tr((H+S)⁻¹ ∂_{β_j}(H+S))` is the β-gradient of the
        // Laplace log-det term. When the family's joint Hessian depends on β
        // (`hessian_derivative_correction` non-`None`), the second-order
        // Taylor expansion of `V` about the inner-returned β̂ at the exact mode
        // `β* = β̂ − H⁻¹r` is
        //   V(β*) − V(β̂) ≈ −½ rᵀH⁻¹r − g_ldᵀH⁻¹r,
        // so the bare `−½ rᵀH⁻¹r` energy omits `−g_ldᵀ w` (w = H⁻¹r). Using
        // `g_ldᵀ w = ½ tr((H+S)⁻¹ D_βH[w])` (linearity of the directional
        // derivative), the missing term is `−½ tr((H+S)⁻¹ D_βH[w])`, evaluated
        // through the SAME inverse/pseudo-inverse the residual energy used so the
        // value stays consistent. For a β-independent Hessian (Gaussian,
        // `hessian_derivative_correction → None`) the term is exactly zero and
        // the released path is byte-unchanged; it only activates on a β-dependent
        // curvature evaluated at a non-stationary β̂ (loose inner solve), exactly
        // the regime gam#1395's `requires_joint_stationarity` fixture pins.
        //
        // SIGN: `hessian_derivative_correction(v)` returns the moving-Hessian
        // part of `∂H/∂ρ_k` at the mode response `v_k = H⁻¹(A_kβ̂)`, i.e.
        // `D_βH[∂β̂/∂ρ_k] = D_βH[−v_k] = −D_βH[v_k]`. Called with `w` it returns
        // `−D_βH[w]`, so `D_βH[w] = −correction` and the missing term
        // `−½ tr(H⁻¹ D_βH[w]) = +½ tr(H⁻¹ · correction)`.
        let moving_hessian_logdet_response =
            match solution.deriv_provider.hessian_derivative_correction(&w) {
                Ok(Some(d_beta_h)) => {
                    let trace = match solution.penalty_subspace_trace.as_ref() {
                        Some(kernel) => kernel.trace_projected_logdet(&d_beta_h),
                        None => hop.trace_hinv_product(&d_beta_h),
                    };
                    0.5_f64 * trace
                }
                Ok(None) => 0.0,
                Err(reason) => {
                    return Err(RemlError::ContractViolation {
                        reason: format!(
                            "moving-Hessian IFT log-det response (gam#1395) failed: {reason}"
                        ),
                    }
                    .into());
                }
            };
        let cost_correction = cost_correction + moving_hessian_logdet_response;
        inner_polish_step = polish_step_for_warm_start;
        let residual_energy = -cost_correction;
        log::debug!(
            "[IFT-ENERGY] residual_energy={:.3e} cost_correction={:.3e} branch={}",
            residual_energy,
            cost_correction,
            branch,
        );
        if cost_correction.is_finite() {
            ift_residual_energy = Some(residual_energy);
            cost += cost_correction;
        }
    }
    // #2954: the certificate's band charges the error `V` carries because the
    // inner mode stops at a residual, `E_r = ½·rᵀH_β⁻¹r` in `V`'s own units,
    // wherever an armed evaluation can form it: the correction's own energy
    // where the correction ran, otherwise the residual of the correction's own
    // construction that the assembly handed over for the band alone, priced
    // through the same mode-response kernel and never added to the cost. The
    // profiled-Gaussian criterion reads the penalized deviance `D_p` through
    // `((n−M_p)/2)·log D_p`, so an excess `δD_p = rᵀH⁻¹r` in `D_p` is
    // `E_r·dD_p'/φ̂` in `V`.
    if crate::estimate::outer_eval_capture::certificate_parts_capture_enabled() {
        let handed = crate::estimate::outer_eval_capture::take_certificate_band_residual();
        let source = handed.as_ref().map_or(
            crate::estimate::outer_eval_capture::InnerResidualSource::InnerGradient,
            |(_, source)| *source,
        );
        let energy = ift_residual_energy.or_else(|| {
            let (residual, _) = handed?;
            if residual.as_array().len() != hop.dim() {
                return None;
            }
            let reduced = match solution.penalty_subspace_trace.as_ref() {
                Some(kernel) => residual
                    .projected_into_reduced_range(kernel)
                    .ok()?
                    .as_array()
                    .clone(),
                None => residual.as_array().clone(),
            };
            let half_energy = 0.5 * reduced.dot(&mode_kernel.respond_one(&reduced));
            Some(match &solution.dispersion {
                DispersionHandling::ProfiledGaussian => half_energy * dp_cgrad / profiled_scale,
                DispersionHandling::Fixed { .. } => half_energy,
            })
        });
        if let Some(energy) = energy.filter(|energy| energy.is_finite()) {
            crate::estimate::outer_eval_capture::record_certificate_inner_residual(
                crate::estimate::outer_eval_capture::InnerResidualCharge { energy, source },
            );
        }
    }

    // Extract logdet flags once (same for all coordinates) — needed here for
    // the guarded TK correction, and reused for the gradient/Hessian below.
    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };
    // gam#2765: the constrained Laplace normalizer. `½ log|M|` above integrates the quadratic
    // model over the whole coefficient space; a constrained mode's Laplace integral runs over the
    // feasible cone, which adds `C = −½gᵀM⁻¹g − ln P(u ≥ 0)` (see `ConeNormalizer`). `C` names no
    // active set, so the criterion stays continuous where the face changes. It reads the same
    // precision the log-determinant prices, in unscaled units (`M⁻¹ = s·M_op⁻¹`).
    let cone_scale = solution.rho_curvature_scale;
    let cone_solve = |rhs: &Array1<f64>| -> Array1<f64> {
        let solved = match solution.penalty_subspace_trace.as_ref() {
            Some(kernel) => kernel.apply_pseudo_inverse(rhs),
            None => hop.solve(rhs),
        };
        solved * cone_scale
    };
    let cone_normalizer = match solution.cone_normalizer.as_ref() {
        // A profiled scale moves the posterior precision `H/φ̂` with ρ, which `C` does not price.
        Some(_) if matches!(solution.dispersion, DispersionHandling::ProfiledGaussian) => {
            return Err(RemlError::ContractViolation {
                reason: "the constrained Laplace normalizer is priced at fixed dispersion; a \
                         profiled-Gaussian solution cannot carry one (gam#2765)"
                    .to_string(),
            }
            .into());
        }
        Some(input) if incl_logdet_h => {
            let normalizer = crate::constrained_posterior::ConeNormalizer::evaluate(
                &input.rows,
                &input.bounds,
                &solution.beta,
                &input.gradient,
                &cone_solve,
            )
            .map_err(RemlLamlError::ConeNormalizer)?;
            log::debug!(
                "[2765-CONE] value={:.9e} log_mass={:.9e} retained_rows={} ep_sweeps={} ep_fraction={:e}",
                normalizer.value(),
                normalizer.log_mass(),
                normalizer.retained_rows(),
                normalizer.sweeps(),
                normalizer.ep_step_fraction(),
            );
            cost += normalizer.value();
            Some((input, normalizer))
        }
        _ => None,
    };
    let logdet_h_component = if incl_logdet_h { 0.5 * log_det_h } else { 0.0 };
    let logdet_s_component = if incl_logdet_s { -0.5 * log_det_s } else { 0.0 };
    let kkt_component = ift_residual_energy.map_or(0.0, |energy| -energy);
    let criterion_components = RemlCriterionComponents {
        fixed_beta: cost - logdet_h_component - logdet_s_component - kkt_component,
        logdet_h: logdet_h_component,
        logdet_s: logdet_s_component,
        kkt: kkt_component,
    };

    if !cost.is_finite() {
        return Err(RemlError::NonFiniteValue {
            reason: format!(
                "REML/LAML cost is non-finite ({cost}); check inner solver convergence"
            ),
        }
        .into());
    }

    if mode == EvalMode::ValueOnly {
        return Ok(RemlLamlResult {
            cost,
            criterion_components,
            ift_residual_energy,
            inner_polish_step,
            gradient: None,
            hessian: gam_problem::HessianValue::Unavailable,
            rho_mode_response_cols: None,
            ext_mode_response_cols: None,
        });
    }

    log::debug!(
        "[STAGE] reml_laml cost_only_done k={} ext_dim={} dim={} elapsed={:.3}s",
        k,
        solution.ext_coords.len(),
        hop.dim(),
        cost_phase_start.elapsed().as_secs_f64(),
    );

    // ─── Gradient (uses SAME hop, SAME intermediates) ───

    // When a barrier is active, wrap the inner derivative provider so that
    // dH/dρ and d²H/dρ² include barrier-Hessian correction terms.
    let barrier_deriv_holder: Option<BarrierDerivativeProvider<'_>> = if let Some(ref barrier_cfg) =
        solution.barrier_config
    {
        match BarrierDerivativeProvider::new(&*solution.deriv_provider, barrier_cfg, &solution.beta)
        {
            Ok(bdp) => Some(bdp),
            Err(e) => {
                log::debug!("BarrierDerivativeProvider skipped (infeasible): {e}");
                None
            }
        }
    } else {
        None
    };
    let effective_deriv: &dyn HessianDerivativeProvider = match barrier_deriv_holder {
        Some(ref bdp) => bdp,
        None => &*solution.deriv_provider,
    };

    // `incl_logdet_h` / `incl_logdet_s` were extracted once above (before the
    // value-only early return) and are reused here for the gradient/Hessian.

    let ext_dim = solution.ext_coords.len();
    let mut grad = Array1::zeros(k + ext_dim);
    // Coordinate-local fixed-β penalty terms, mode responses, and family
    // derivative corrections are independent within a single outer evaluation.
    // Keep the dependency-ordered BFGS/line-search loops serial, but use rayon
    // here so each accepted outer iterate evaluates its objective derivatives
    // by farming out the per-coordinate Hessian/gradient work.
    // The full gradient-bearing penalty atom. It carries the SAME stable-basis
    // value the cost above consumed, so `value()` and `rho_frozen_d1` are
    // projections of one object (and any fold over this atom reports the
    // numerically-sound stable energy, not the original-basis sum).
    let penalty_quad_atom = crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
        &lambdas,
        &solution.penalty_coords,
        &solution.beta,
    )?
    .with_stable_value(0.5 * solution.penalty_quadratic);
    let curvature_penalty_quad_atom =
        crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
            &curvature_lambdas,
            &solution.penalty_coords,
            &solution.beta,
        )?;
    let rho_penalty_a_k_betas: Vec<Array1<f64>> = penalty_quad_atom.block_penalty_scores().to_vec();
    let rho_curvature_a_k_betas: Vec<Array1<f64>> =
        curvature_penalty_quad_atom.block_penalty_scores().to_vec();
    let need_family_corrections = effective_deriv.has_corrections();
    // The constrained normalizer's gradient reads every coordinate's mode response (gam#2765).
    let need_rho_mode_responses = need_family_corrections
        || mode == EvalMode::ValueGradientHessian
        || cone_normalizer.is_some();
    // Stack the K curvature-penalty RHS whenever a later stage will need
    // rho mode responses (family logdet corrections or outer Hessian
    // assembly), plus all ext-coordinate gradient RHS, into one
    // (dim, total_cols) solve. The dense-spectral backend turns that into
    // one `Uᵀ·R`, one per-eigendirection scale, and one `U·projected` — a
    // BLAS-3 pass with much better cache locality than independent BLAS-2
    // single-RHS solves, and it lets large-`dim` fits hit the batched solve
    // route in `gpu/policy.rs`.
    let dim = hop.dim();
    let ext_dim_local = solution.ext_coords.len();
    let total_cols = if need_rho_mode_responses {
        k + ext_dim_local
    } else {
        ext_dim_local
    };
    let (rho_v_ks, ext_v_is): (Option<Vec<Array1<f64>>>, Vec<Array1<f64>>) = if total_cols == 0 {
        (
            if need_rho_mode_responses {
                Some(Vec::new())
            } else {
                None
            },
            Vec::new(),
        )
    } else {
        // Per-coordinate `v = H⁻¹ · a` mode responses for the IFT chain
        // rule (β̂_ψ = −H⁻¹ · g_ψ). Choice of solve:
        //
        //   * Active inequality constraints recorded → use the lifted
        //     kernel `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. The inner
        //     SCOP solver clamps β̂(ψ) onto the manifold
        //     `T = range(S₊) ∩ ker(A_act)`, so its true IFT derivative
        //     lives in T and the lifted kernel gives the minimum-norm
        //     solution there. The full `hop.solve_multi` amplifies any
        //     component of `a` outside `range(H_free)` by
        //     `1/σ_min(H_active_normal)` — which on large-scale
        //     survival marginal-slope (commit d6b17a7f) is ~10¹² and
        //     trips the envelope-consistency check downstream; the
        //     lifted kernel drops that null-space contribution by
        //     construction and stays bounded.
        //
        //   * Otherwise (no active constraints) → use the full
        //     `hop.solve_multi`. The inner solver converges β̂ ∈ R^p in
        //     the unconstrained full space, so the IFT derivative is
        //     `β̂_ψ = −H⁻¹ · g_ψ` with the FULL Hessian, even when the
        //     LAML cost surface itself uses the projected logdet
        //     `½ log|U_Sᵀ H U_S|`. Projecting `v` through bare K_S
        //     = U_S·(U_Sᵀ H U_S)⁻¹·U_Sᵀ here would discard the
        //     `null(S₊)` component of dβ̂/dψ, which is non-zero for any
        //     family whose `X` has columns living in `null(S₊)` (e.g.,
        //     an intercept under a wiggle smoothing penalty under
        //     Probit / Logit / cloglog, where the working weight
        //     `W(η(β̂))` changes with ψ and pushes the intercept along
        //     with β̂); on near-separable data the projected
        //     `(U_Sᵀ H U_S)⁻¹` then over-amplifies that component and
        //     blows the analytic ψ-gradient to O(1e6) vs the FD ≈ −1.
        //     The penalty-subspace projection on the TRACE side (the
        //     outer `tr[K · …]` contraction with K_S) is unchanged —
        //     that is what the projected LAML cost identity demands —
        //     only the IFT *direction* `v` is the full solve, which the
        //     kernel `duchon_probit_per_row_dnu_dpsi_fd_vs_analytic`
        //     pins via the FD reference `c · dη/dψ_total = c · (η_+ − η_−)/2h`.
        //
        // The choice itself is no longer made here: it is the ONE
        // `ThetaModeResponseKernel::select` decision (#931 pass 2), shared
        // verbatim by `compute_outer_hessian` and
        // `build_outer_hessian_operator`. Box-masked ρ coordinates keep
        // their RHS column zero, which both kernel arms map to exact zeros.
        // #2612: the mode response is differentiated through the INNER
        // stationarity system, which is not always the object the logdet is
        // priced on. `mode_response_operator()` is that system's operator and
        // equals `hop` on every lane that has not installed a distinct one;
        // `mode_kernel` is built on it once, above the cost correction.
        let mut rhs_stack = Array2::<f64>::zeros((dim, total_cols));
        let mut col_idx = 0;
        if need_rho_mode_responses {
            // Every rho coordinate gets its mode response, including one sitting
            // on a box bound. `v_k = dbeta/drho_k` is a property of the inner map
            // `beta(rho)`, not of the outer feasible set: zeroing it there made
            // the criterion's own derivative depend on which box the caller
            // happened to pass (#2615).
            for a_k_beta in rho_curvature_a_k_betas.iter() {
                rhs_stack.column_mut(col_idx).assign(a_k_beta);
                col_idx += 1;
            }
        }
        for coord in solution.ext_coords.iter() {
            rhs_stack.column_mut(col_idx).assign(&coord.g);
            col_idx += 1;
        }
        assert_eq!(col_idx, total_cols);
        let solved_stack = mode_kernel.respond_stack(&rhs_stack);
        let rho_v_ks = if need_rho_mode_responses {
            Some((0..k).map(|i| solved_stack.column(i).to_owned()).collect())
        } else {
            None
        };
        let ext_offset = if need_rho_mode_responses { k } else { 0 };
        let ext_v_is: Vec<Array1<f64>> = (0..ext_dim_local)
            .map(|i| solved_stack.column(ext_offset + i).to_owned())
            .collect();
        (rho_v_ks, ext_v_is)
    };
    let coord_corrections: Vec<Option<DriftDerivResult>> = if effective_deriv.has_corrections() {
        let rho_vs = rho_v_ks
            .as_ref()
            .expect("rho mode responses required for Hessian corrections");
        let mut correction_vs: Vec<Array1<f64>> = Vec::with_capacity(k + ext_dim);
        correction_vs.extend(rho_vs.iter().cloned());
        correction_vs.extend(ext_v_is.iter().cloned());
        let correction_work = solution
            .n_observations
            .saturating_mul(hop.dim())
            .saturating_mul((k + ext_dim).max(1));
        // Preferred path: families whose `D_beta H[u_k]` operators share
        // row-local state across all smoothing coordinates (e.g. the BMS exact
        // joint-Newton workspace) expose a batched hook that fuses the whole
        // per-row scan over the k+ext_dim directions into a SINGLE n-row pass
        // (amortizing the per-row cached cell-moment / third-tensor work that
        // would otherwise be recomputed once per direction) and parallelizes
        // that pass across rows internally. Routing through it turns the former
        // `serial(inner-parallel)` k× n-passes — which left the machine idle
        // (`active_threads=0`) while each thin single-direction crossproduct
        // failed to fill the pool — into one wide, fully-occupied pass.
        if effective_deriv.has_batched_hessian_derivative_corrections() {
            log::debug!(
                "[STAGE] reml_laml coord_corrections mode=batched(row-parallel) k={} ext_dim={} n={} dim={} work={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim(),
                correction_work
            );
            // Named heartbeat scope so the active-scope line attributes the
            // coord_corrections wall time (the biobank's dominant REML stage).
            let coord_corr_scope = gam_runtime::process_monitor::track_scope(format!(
                "reml_laml coord_corrections batched k={k} ext_dim={ext_dim} n={} dim={}",
                solution.n_observations,
                hop.dim()
            ));
            let coord_corrections_result =
                effective_deriv.hessian_derivative_corrections_result(&correction_vs);
            drop(coord_corr_scope);
            coord_corrections_result?
        } else {
            // Fallback for providers without a fused hook: each
            // `hessian_derivative_correction_result` is an `Xᵀ·diag(c⊙Xvₖ)·X`
            // crossproduct that, when large, already saturates every core via
            // faer's global parallelism (`streaming_blas_xt_diag_x` →
            // `get_global_parallelism`). Wrapping the outer per-coordinate map
            // in `par_iter` for large work would nest a rayon fan-out around
            // already-parallel BLAS-3 kernels and oversubscribe the pool. The
            // small-work branch goes parallel only because each correction is
            // too thin to fill the pool on its own, so the fan-out is free.
            let parallel_corrections = correction_work <= 64_000_000;
            if parallel_corrections {
                correction_vs
                    .par_iter()
                    .map(|v_k| effective_deriv.hessian_derivative_correction_result(v_k))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                log::debug!(
                    "[STAGE] reml_laml coord_corrections mode=serial(inner-parallel) k={} ext_dim={} n={} dim={} work={}",
                    k,
                    ext_dim,
                    solution.n_observations,
                    hop.dim(),
                    correction_work
                );
                correction_vs
                    .iter()
                    .map(|v_k| effective_deriv.hessian_derivative_correction_result(v_k))
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
    } else {
        (0..(k + ext_dim)).map(|_| None).collect()
    };
    if coord_corrections.len() != k + ext_dim {
        return Err(RemlError::DimensionMismatch {
            reason: format!(
                "REML/LAML derivative correction count mismatch: got {}, expected {}",
                coord_corrections.len(),
                k + ext_dim
            ),
        }
        .into());
    }
    let rho_corrections = &coord_corrections[..k];
    let ext_corrections = &coord_corrections[k..];

    // Stash the per-coordinate mode-response columns so downstream callers
    // (notably the cached outer-Hessian path) can reuse `v_k = H⁻¹ a_k` and
    // `v_i = H⁻¹ g_i` without re-solving.  Each column is one coordinate's
    // mode response; rho columns are present only when `rho_v_ks` was built.
    let rho_mode_response_cols: Option<Array2<f64>> = rho_v_ks.as_ref().map(|cols| {
        let p = hop.dim();
        let mut out = Array2::<f64>::zeros((p, cols.len()));
        for (idx, v) in cols.iter().enumerate() {
            out.column_mut(idx).assign(v);
        }
        out
    });
    let ext_mode_response_cols: Option<Array2<f64>> = if ext_v_is.is_empty() {
        None
    } else {
        let p = hop.dim();
        let mut out = Array2::<f64>::zeros((p, ext_v_is.len()));
        for (idx, v) in ext_v_is.iter().enumerate() {
            out.column_mut(idx).assign(v);
        }
        Some(out)
    };

    let build_trace_drifts = || {
        let mut drifts = Vec::with_capacity(k + ext_dim);
        for idx in 0..k {
            drifts.push(penalty_total_drift_result(
                &solution.penalty_coords[idx],
                curvature_lambdas[idx],
                rho_corrections[idx].as_ref(),
            ));
        }
        for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
            drifts.push(hyper_coord_total_drift_result(
                &coord.drift,
                ext_corrections[ext_idx].as_ref(),
                hop.dim(),
            ));
        }
        drifts
    };

    let projected_trace_values: Option<Vec<f64>> =
        if incl_logdet_h {
            solution
                .penalty_subspace_trace
                .as_ref()
                .map(|kernel| penalty_subspace_trace_drifts_batched(kernel, &build_trace_drifts()))
        } else {
            None
        };

    let exact_dense_trace_values: Option<Vec<f64>> =
        if incl_logdet_h && projected_trace_values.is_none() {
            hop.as_exact_dense_spectral()
                .map(|ds| dense_spectral_trace_logdet_drifts_batched(ds, &build_trace_drifts()))
        } else {
            None
        };

    // Exact trace batching for operator-backed ρ corrections.  The hot large-scale
    // GAMLSS path has many Duchon smoothing coordinates whose correction
    // operators share the same design and spectral factor.  Evaluating them one
    // by one repeats the same `X·F` projection and row-kernel setup for every
    // coordinate.  The trace is linear, so split `tr(K·(A_i + C_i))` into the
    // cheap penalty part plus a batched exact vector of `tr(K·C_i)` values.
    // This preserves the full first-order gradient (no iteration caps or
    // stochastic approximation) while collapsing the per-coordinate trace pass.
    let rho_operator_correction_traces: Option<Vec<Option<f64>>> = if incl_logdet_h
        && solution.penalty_subspace_trace.is_none()
    {
        let pairs: Vec<(usize, Arc<dyn HyperOperator>)> = rho_corrections
            .iter()
            .enumerate()
            .filter_map(|(idx, correction)| match correction {
                Some(DriftDerivResult::Operator(op)) => Some((idx, Arc::clone(op))),
                _ => None,
            })
            .collect();
        if pairs.len() >= 2 {
            hop.as_exact_dense_spectral().map(|ds| {
                let ops: Vec<Arc<dyn HyperOperator>> =
                    pairs.iter().map(|(_, op)| Arc::clone(op)).collect();
                let values = dense_spectral_trace_logdet_operators_batched(ds, &ops);
                let mut traces = vec![None; k];
                for ((idx, _), value) in pairs.into_iter().zip(values) {
                    traces[idx] = Some(value);
                }
                traces
            })
        } else {
            None
        }
    } else {
        None
    };

    // Cancellation-free fused logdet gradient for singleton penalty blocks (a2).
    //
    // On the exact-dense-spectral path the ρ_k-gradient of `½·log|H|` subtracts
    // the det derivative `first[k] = ∂_{ρ_k} log|S(λ)|₊` from `½·tr(G_ε(H)·Ḣ_k)`.
    // When coordinate k is the SOLE penalty of its span, `first[k]` is the exact
    // integer `rank(S_k)`, and at the over-smoothing rail (H ≈ λ_k S_k) the trace
    // `tr(G_ε(H)·λ_k S_k) → rank(S_k)`, so `trace − first` catastrophically
    // cancels — the surviving O(1/λ_k) gradient is then decided by the last bits
    // of a rank-sized sum and drifts with the host's summation order. For those
    // coordinates we fuse the subtraction eigenpair-by-eigenpair so the result is
    // host-arithmetic-independent (see
    // `DenseSpectralOperator::fused_logdet_gradient_minus_rank_full_block`).
    //
    // Three fused forms cover the exact-dense path (#2331), all cancellation-free
    // reassociations of `tr(G_ε λ_k S_k) − det1[k]`:
    //   • INTEGER det derivative (proportional singleton `log|λ_k S_k|₊ =
    //     rank·ρ_k + const`): fuse `−rank` against the block-coordinate identity
    //     (`fused_logdet_gradient_minus_rank_full_block`, square full rank) or the
    //     range projector `P_{S_k}` (`..._minus_rank_deficient_block`, rank-def).
    //   • FRACTIONAL det derivative (`det1[k] = λ_k·tr(S_λ⁺ S_k)`, the joint
    //     normalizer `log|Σ_l λ_l S_l|₊` — overlapping / coalesced / full-span
    //     stabilization ridge): fuse the per-direction weights `w_jk = λ_k·u_jᵀ
    //     S_λ⁺ S_k u_j` from the joint whitening `W_S`
    //     (`fused_logdet_gradient_weighted_block`), trusted only when `Σ_j w_jk`
    //     reproduces the cost's `det1[k]`.
    // A masked numerical null space (`active_rank() < dim()`, i.e. `HardPseudo`
    // masking `σ_j ≤ ε`) needs no special handling here (#2354): the trace side
    // reads `g_factor`, which is zeroed on the masked eigenpairs, so
    // `Σ_j scale·s_term_j` is exactly the active-subspace trace the naive path
    // also forms (`trace_logdet_block_local` reads the same `g_factor`); the
    // `−det1[k]` distribution sums its per-eigenpair share over the COMPLETE
    // (UNMASKED) eigenbasis, so the completeness identity `Σ_j share_j = det1[k]`
    // (`Σ_j Σ_{i∈blk} u_j[i]² = width = rank` full-block; `Σ_j ‖Qᵀ u_j^blk‖² =
    // rank` deficient; `Σ_j w_jk = λ_k·tr(S_λ⁺ S_k)` weighted) still holds — rows
    // of the full orthogonal `U` are unit-norm regardless of the mask. Hence the
    // fused value equals `trace_active − det1[k]` exactly and cancellation-free:
    // the active pairs `scale·s_term_j − share_j` stay O(1/λ_k) as in the
    // full-rank rail derivation, and each masked pair contributes only the
    // non-negative lump `0 − share_j` (no large-minus-large).
    let fused_logdet_minus_rank: Vec<Option<f64>> = if incl_logdet_h
        && incl_logdet_s
        && projected_trace_values.is_none()
    {
        match hop.as_exact_dense_spectral() {
            Some(ds) => {
                // Joint penalty whitening `W_S` (`W_S W_Sᵀ = S_λ⁺`,
                // `S_λ = Σ_l λ_l S_l`) reconstructed from the penalty coordinates
                // — needed ONLY when some coordinate carries a FRACTIONAL det
                // derivative `det1[k] = λ_k·tr(S_λ⁺ S_k)` (the joint-normalizer
                // case: overlapping / coalesced / full-span-ridge penalties). The
                // integer-rank singletons keep the cheaper block-indicator /
                // range-projector fusions and never pay this extra
                // eigendecomposition. `from_assembled` matches the tolerance of the
                // penalty-logdet cost path (both eigendecompose the same `S_λ`), so
                // the reconstructed weight sum reproduces `det1[k]` — the runtime
                // gate below trusts the fused value only when it does.
                let any_fractional = (0..k).any(|idx| {
                    let rank = solution.penalty_coords[idx].rank();
                    (solution.penalty_logdet.first[idx] - rank as f64).abs()
                        > 1e-9 * (1.0 + rank as f64)
                });
                let joint_whitening: Option<Array2<f64>> = if any_fractional {
                    let p = ds.dim();
                    let mut s_lambda = Array2::<f64>::zeros((p, p));
                    for l in 0..k {
                        let (block, start, end) =
                            solution.penalty_coords[l].scaled_block_local(curvature_lambdas[l]);
                        let mut sub = s_lambda.slice_mut(ndarray::s![start..end, start..end]);
                        sub += &block;
                    }
                    super::super::penalty_logdet::PenaltyPseudologdet::from_assembled(
                        s_lambda, None,
                    )
                    .ok()
                    .map(|pld| pld.w_factor)
                } else {
                    None
                };
                (0..k)
                    .map(|idx| {
                        let coord = &solution.penalty_coords[idx];
                        let rank = coord.rank();
                        let (s_block, start, end) = coord.scaled_block_local(1.0);
                        let det1_k = solution.penalty_logdet.first[idx];
                        // The family curvature correction C[v_k] has no paired det
                        // term; add its logdet trace back so the fused value equals
                        // `tr(G_ε·(λ_k S_k + C)) − det1[k]`.
                        // Reuses the batched `tr(G_ε·C_k)` when it was formed
                        // above rather than tracing each correction a second time.
                        let correction_trace = rho_operator_correction_traces
                            .as_ref()
                            .and_then(|traces| traces[idx])
                            .or_else(|| rho_corrections[idx].as_ref().map(|c| c.trace_logdet(hop)))
                            .unwrap_or(0.0);
                        // Integer det derivative ⇒ PROPORTIONAL SINGLETON block
                        // (`log|λ_k S_k|₊ = rank·ρ_k + const`), det term is exactly
                        // `−rank`: fuse against the block-coordinate identity
                        // (square full rank) or the range projector `P_{S_k}`
                        // (rank-deficient). Both are value-identical to
                        // `trace − first[idx]` and cancellation-free at the rail.
                        let det_is_integer_rank =
                            (det1_k - rank as f64).abs() <= 1e-9 * (1.0 + rank as f64);
                        if det_is_integer_rank {
                            let is_square_full_rank = end - start == rank;
                            let fused = if is_square_full_rank {
                                ds.fused_logdet_gradient_minus_rank_full_block(
                                    idx,
                                    &s_block,
                                    start,
                                    end,
                                    curvature_lambdas[idx],
                                )
                            } else {
                                let (range_root, root_start, root_end) =
                                    coord.block_local_root()?;
                                // The root chart's span must be the span this
                                // coordinate is being evaluated over, or the fused
                                // gradient below reads the wrong block. Checked
                                // unconditionally: a `debug_assert` states an
                                // invariant that then vanishes from every shipped
                                // build, which is where a mismatch would actually
                                // do its damage. Two usize comparisons on a path
                                // that follows a Cholesky are free.
                                assert_eq!(
                                    (root_start, root_end),
                                    (start, end),
                                    "block-local root chart span ({root_start}, {root_end}) \
                                     must match the coordinate's evaluated span ({start}, {end})"
                                );
                                ds.fused_logdet_gradient_minus_rank_from_root_chart(
                                    idx,
                                    &s_block,
                                    range_root,
                                    start,
                                    end,
                                    curvature_lambdas[idx],
                                )
                            };
                            return Some(fused + correction_trace);
                        }
                        // FRACTIONAL det derivative (joint normalizer,
                        // `log|Σ_l λ_l S_l|₊`): weighted fusion over the joint range
                        // chart. Trusted only when its own weight sum reproduces the
                        // cost's `det1[k]` — the runtime self-consistency gate that
                        // keeps this off any lane whose `det1` is not this exact
                        // joint quantity (e.g. a not-yet-cutover per-block seam).
                        let ws = joint_whitening.as_ref()?;
                        let (fused, weight_sum) = ds.fused_logdet_gradient_weighted_block(
                            idx,
                            &s_block,
                            start,
                            end,
                            curvature_lambdas[idx],
                            ws,
                        );
                        if (weight_sum - det1_k).abs() > 1e-7 * (1.0 + det1_k.abs()) {
                            return None;
                        }
                        Some(fused + correction_trace)
                    })
                    .collect()
            }
            None => vec![None; k],
        }
    } else {
        vec![None; k]
    };

    // ── Gradient: one shared formula for ALL coordinate types ──
    //
    // Both ρ and ext coordinates are processed through outer_gradient_entry()
    // so that the three-term formula (penalty + trace − det) is written once.

    // #2454: the ρ-block audit needs each entry split into the SAME additive
    // parts the criterion value carries, not just their sum. Computed inside
    // the parallel map and returned alongside the entry (rayon workers cannot
    // see the requesting thread's thread-local window).
    // `tr(K · Ḣ)` under whichever kernel the cost's log-determinant pairs with.
    // ONE definition, shared by the ρ block's audit split below and by the ψ/ext
    // block further down: the two must contract the same kernel or their
    // `logdet_h` parts would be measured against different surfaces.
    let trace_logdet_drift =
        |drift: &DriftDerivResult| match (&solution.penalty_subspace_trace, drift) {
            (Some(kernel), DriftDerivResult::Dense(matrix)) => {
                kernel.trace_projected_logdet(matrix)
            }
            (Some(kernel), DriftDerivResult::Operator(op)) => kernel.trace_operator(op.as_ref()),
            (None, DriftDerivResult::Dense(matrix)) => hop.trace_logdet_h_k(matrix, None),
            (None, DriftDerivResult::Operator(op)) => hop.trace_logdet_operator(op.as_ref()),
        };
    // The drift split is audit-only. The parts themselves are also published to
    // an armed certificate capture (#2954). Both flags are read HERE, on the
    // calling thread: the map below runs on pool threads, where a thread-local
    // reads disarmed.
    let capture_drift_split = crate::estimate::outer_eval_capture::rho_outer_audit_enabled();
    let capture_rho_parts = capture_drift_split
        || crate::estimate::outer_eval_capture::certificate_parts_capture_enabled();
    type RhoGradEntry = (usize, f64, f64, f64, f64, f64, f64, f64, f64);
    let rho_grad_entries: Vec<RhoGradEntry> = (0..k)
        .into_par_iter()
        .map(|idx| {
            // Every coordinate's entry is the TRUE partial derivative of the
            // criterion, including one sitting on a box bound. The KKT
            // projection is applied ONCE, to the assembled gradient, below
            // (#2615) — see the note there for why it cannot be a distance
            // test taken here.

            // Cost derivative for the shifted penalty:
            // a_i = ½ λₖ (β̂ - μₖ)' Sₖ (β̂ - μₖ).
            //
            // The β-gradient derivative is λₖSₖ(β̂-μₖ); dotting it with β̂
            // would drop the μₖ'λₖSₖμₖ half of the chain rule.
            let a_i = penalty_quad_atom.rho_frozen_d1(idx);

            let coord = &solution.penalty_coords[idx];

            // Trace term: tr(K · Ḣₖ) where Ḣₖ = Aₖ + C[vₖ].
            //
            // Kernel choice mirrors the ψ/τ block: full-space `G_ε(H)` when the
            // cost uses the smooth-floored `log|H|`, or the intrinsic spectral
            // kernel `K = H_pen⁺` when the rank-deficient LAML fix is active
            // (#901) — the exact derivative of the cost's `log|H_pen|₊` for the
            // TOTAL drift, including the third-derivative correction
            // `C[vₖ] = X'·diag(c ⊙ X vₖ)·X` that leaks onto `null(S)` for
            // non-Gaussian families. The two kernels disagree whenever
            // `hessian_logdet_correction ≠ 0` (they treat sub-threshold
            // eigendirections differently), so the pairing with the cost
            // identity is what keeps analytic and FD gradients on one surface.
            // Fused (a2) path: `fused` already equals `tr(G_ε·Ḣ_k) − rank`, so
            // the `−rank` det term is folded in and `ld_s_i` must be passed as
            // 0 to `outer_gradient_entry` (subtracting it again would
            // double-count). Every non-fused coordinate keeps the exact
            // `trace_logdet_i` / `first[idx]` det pairing unchanged.
            let (trace_logdet_i, ld_s_i) = if let Some(fused) = fused_logdet_minus_rank[idx] {
                (fused, 0.0)
            } else {
                let trace = if !incl_logdet_h {
                    0.0
                } else if let Some(ref projected_traces) = projected_trace_values {
                    projected_traces[idx]
                } else if let Some(ref exact_traces) = exact_dense_trace_values {
                    exact_traces[idx]
                } else if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                    let drift = penalty_total_drift_result(
                        coord,
                        curvature_lambdas[idx],
                        rho_corrections[idx].as_ref(),
                    );
                    match drift {
                        DriftDerivResult::Dense(matrix) => kernel.trace_projected_logdet(&matrix),
                        DriftDerivResult::Operator(op) => kernel.trace_operator(op.as_ref()),
                    }
                } else if let Some(correction_trace) = rho_operator_correction_traces
                    .as_ref()
                    .and_then(|traces| traces[idx])
                {
                    // The PURE-PENALTY half of the drift trace, `tr(G_ε·λ_kS_k)`,
                    // is taken from the coordinate's ROOT (#2644): the squared
                    // block carries `S_k`'s roundoff linearly through a metric
                    // scaled by `σ(H)^{-1}`, i.e. `O(ε·κ(H))` on a trace bounded
                    // by `rank(S_k)`. The moving-curvature half `C[v_k]` is not
                    // PSD and has no root, so it keeps its own path and is added
                    // here unchanged.
                    penalty_logdet_trace_from_root(hop, idx, coord, curvature_lambdas[idx])
                        + correction_trace
                } else if rho_corrections[idx].is_none()
                    && let Some(trace) =
                        penalty_logdet_trace_from_root_opt(hop, idx, coord, curvature_lambdas[idx])
                {
                    // No moving-curvature correction, so the whole drift IS the
                    // penalty and the root form prices all of it (#2644).
                    trace
                } else if coord.is_block_local() && rho_corrections[idx].is_none() {
                    // Reached only when the scale admits no real root to price from.
                    let (block, start, end) = coord.scaled_block_local(1.0);
                    hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
                } else {
                    penalty_total_drift_result(
                        coord,
                        curvature_lambdas[idx],
                        rho_corrections[idx].as_ref(),
                    )
                    .trace_logdet(hop)
                };
                (trace, solution.penalty_logdet.first[idx])
            };
            let value = outer_gradient_entry(
                a_i,
                trace_logdet_i,
                ld_s_i,
                &solution.dispersion,
                dp_cgrad,
                profiled_scale,
                incl_logdet_h,
                incl_logdet_s,
            );
            // Per-coordinate breakdown of the outer-gradient entry. Was a
            // floor-level eprintln during the LAML cost-trajectory
            // investigation; demoted to trace! so RUST_LOG=trace can still
            // recover it without 91-line-per-iter stderr noise on default
            // runs.
            log::trace!(
                "[RHO-GRAD] idx={} value={:+.6e} a_i={:+.6e} trace_logdet={:+.6e} ld_s={:+.6e} fused={} incl_h={} incl_s={}",
                idx, value, a_i, trace_logdet_i, ld_s_i, fused_logdet_minus_rank[idx].is_some(), incl_logdet_h, incl_logdet_s
            );
            // Split into the criterion-value components. `fixed_beta` is the
            // dispersion-scaled penalty quadratic derivative; `logdet_h` /
            // `logdet_s` are the two determinant channels. The fused (a2) route
            // returns `tr(G_ε·Ḣ_k) − rank` as one number, so it is attributed
            // whole to `logdet_h` with `logdet_s = 0` — exactly how the entry
            // itself is assembled.
            let (part_fixed_beta, part_logdet_h, part_logdet_s) = if capture_rho_parts {
                (
                    outer_gradient_entry(
                        a_i,
                        0.0,
                        0.0,
                        &solution.dispersion,
                        dp_cgrad,
                        profiled_scale,
                        false,
                        false,
                    ),
                    if incl_logdet_h { 0.5 * trace_logdet_i } else { 0.0 },
                    if incl_logdet_s { -0.5 * ld_s_i } else { 0.0 },
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            // The same FROZEN / MODE-RESPONSE split the ψ block publishes
            // (#2765). `Ḣ_k = λ_k S_k + D_β H[v_k]`, and only the second half
            // reads the coefficient mode response — so when a `logdet_h`
            // disagreement has to be attributed, the split is what says which
            // half owns it. The frozen half is additionally a SIGN CHECK that
            // needs no oracle: `tr(K · λ_k S_k)` with `K` and `S_k` both PSD
            // cannot be negative.
            let (part_frozen_logdet_h, part_mode_response_logdet_h) =
                if capture_drift_split && incl_logdet_h {
                    let frozen = penalty_total_drift_result(coord, curvature_lambdas[idx], None);
                    let mode_response = rho_corrections[idx]
                        .as_ref()
                        .map_or(0.0, |drift| trace_logdet_drift(drift));
                    (0.5 * trace_logdet_drift(&frozen), 0.5 * mode_response)
                } else {
                    (0.0, 0.0)
                };
            let block_quadratic = if capture_rho_parts && idx < penalty_quad_atom.lambdas.len() {
                penalty_quad_atom.block_quadratics[idx]
            } else {
                0.0
            };
            (
                idx,
                value,
                lambdas[idx],
                block_quadratic,
                part_fixed_beta,
                part_logdet_h,
                part_frozen_logdet_h,
                part_mode_response_logdet_h,
                part_logdet_s,
            )
        })
        .collect();
    let mut rho_audit_parts: Vec<crate::estimate::outer_eval_capture::RhoGradientParts> =
        Vec::new();
    for (
        idx,
        value,
        lambda,
        block_quadratic,
        fixed_beta,
        ld_h,
        frozen_ld_h,
        mode_response_ld_h,
        ld_s,
    ) in rho_grad_entries
    {
        grad[idx] = value;
        if capture_rho_parts {
            rho_audit_parts.push(crate::estimate::outer_eval_capture::RhoGradientParts {
                index: idx,
                lambda,
                block_quadratic,
                rank: solution.penalty_coords[idx].rank(),
                dim: solution.penalty_coords[idx].dim(),
                fixed_beta,
                logdet_h: ld_h,
                frozen_logdet_h: frozen_ld_h,
                mode_response_logdet_h: mode_response_ld_h,
                logdet_s: ld_s,
                total: value,
            });
        }
    }

    // ─── Implicit-function-theorem gradient correction ───
    //
    // The envelope formula above is the total derivative dV/dρ_k *only* when
    // β̂ satisfies the inner KKT condition ∇_β L_pen(β̂) = 0.  When the inner
    // exits via the noise-floor certificate with `r = ∇_β L_pen(β̂) ≠ 0`,
    // the corrected scalar objective is the one-step Newton profile
    //
    //   Ṽ(ρ) = V(β̂, ρ) − ½ rᵀ H⁻¹ r.
    //
    // Holding β̂ fixed while differentiating the correction gives, with
    // q = H⁻¹r, A_k = λ_k S_k, and a_k = A_k β̂:
    //
    //   ∂_k r = a_k,        ∂_k H = A_k
    //   ∂_k q = H⁻¹(a_k − A_k q)
    //   ∂_k(-½ rᵀq) = −a_kᵀq + ½ qᵀA_kq.
    //
    // The leading `−a_kᵀq` term is the familiar `−rᵀv_k` correction; the
    // `+½ qᵀA_kq` term is second-order in the KKT residual but is required if
    // the analytic gradient is to be the derivative of the corrected scalar
    // objective. The Hessian builder receives the corresponding second
    // derivative from `compute_kkt_residual_theta_corrections` so ARC sees one
    // coherent objective model instead of an envelope Hessian for a corrected
    // value/gradient pair.
    //
    // The correction strictly vanishes when r = 0.  When the inner exit
    // accepts ‖r‖ > 0 on a coordinate whose H block is poorly conditioned
    // (e.g., the failing large-scale survival marginal-slope case where ‖H⁻¹‖
    // is ~10¹² on the one unpinned λ), the dropped term inflates by
    // ‖H⁻¹‖·‖r‖ and the envelope reports a gradient component orders of
    // magnitude past anything the function can actually produce — TR
    // rejects every step and collapses to its floor.  This term recovers
    // the legitimate descent direction.
    //
    // Use `rho_penalty_a_k_betas` (with `lambdas`), NOT `rho_v_ks` (whose
    // computation may use `curvature_lambdas = rho_curvature_scale · lambdas`):
    // the residual correction is in the actual S(λ) basis, and the curvature
    // scale only applies to the H-dependent trace terms.
    // KKT-residual correction over the FULL θ = (ρ ‖ ψ) coordinate set.
    //
    // The corrected scalar objective is `Ṽ(θ) = V(β̂,θ) − ½ rᵀ H⁻¹ r`. Its
    // gradient AND Hessian were previously corrected on the ρ block only (the
    // ψ/ext gradient was patched inline, but the cross-ρψ and ψψ Hessian blocks
    // were dropped — silently biasing the LAML curvature, hence smoothing
    // selection and SEs, on any near-singular fit that exits with ‖r‖>0). The
    // generalized `compute_kkt_residual_theta_corrections` emits all blocks from
    // ONE factorization with one algebra: ρ coordinates feed `r_i = λ_iS_iβ̂`,
    // `A_i[v] = λ_iS_i v`; ψ/ext coordinates feed `r_i = coord.g`, `A_i[v] =
    // B_i v` (the frozen ψ Hessian drift). It vanishes identically at exact KKT.
    let ext_frozen_drifts: Vec<DriftDerivResult> = (0..ext_dim)
        .map(|ext_idx| {
            hyper_coord_total_drift_result(&solution.ext_coords[ext_idx].drift, None, hop.dim())
        })
        .collect();
    let kkt_theta_corrections = if let Some(r) = kkt_residual_vec
        .as_ref()
        .filter(|_| kkt_residual_correction_active && (k + ext_dim) > 0)
        .map(|r| r.as_ref())
    {
        // Per-coordinate score derivatives r_i in packed θ = (ρ ‖ ψ) order.
        let mut score_derivs: Vec<Array1<f64>> = Vec::with_capacity(k + ext_dim);
        score_derivs.extend(rho_penalty_a_k_betas.iter().cloned());
        score_derivs.extend((0..ext_dim).map(|ext_idx| solution.ext_coords[ext_idx].g.clone()));
        // Total drift dH/dtheta_i[v] for the IFT envelope correction. This MUST be
        // the SAME total drift the log|H| trace uses: the penalty term
        // curvature_lambda_i * S_i PLUS the family curvature term D_beta H[v_i]
        // (rho_corrections[i]). Pairing the value correction -1/2 r^T H^-1 r with a
        // gradient correction that differentiates a DIFFERENT H makes them disagree
        // whenever the inner residual r is not exactly zero -- the survival
        // large-lambda desync, where the event-time Hessian scales as 1/s^2 in the
        // penalized coefficient so the dropped D_beta H term is large. For
        // Gaussian/canonical families rho_corrections[i] is None, so this reduces to
        // the penalty term and matches the prior behaviour exactly.
        let rho_total_drifts: Vec<DriftDerivResult> = (0..k)
            .map(|idx| {
                penalty_total_drift_result(
                    &solution.penalty_coords[idx],
                    curvature_lambdas[idx],
                    rho_corrections[idx].as_ref(),
                )
            })
            .collect();
        let drift_apply = |idx: usize, v: &Array1<f64>| -> Array1<f64> {
            if idx < k {
                rho_total_drifts[idx].apply(v)
            } else {
                ext_frozen_drifts[idx - k].apply(v)
            }
        };
        // No coordinate is frozen while the correction is FORMED. Which
        // components the feasible set removes is decided once, on the assembled
        // gradient, by the KKT projection below (#2615); masking them here made
        // the correction and the envelope block disagree about what was being
        // differentiated.
        let active = vec![false; k + ext_dim];
        // The KKT-correction self-derivative term `δ_ij·C_i` is non-zero only
        // for coordinates whose r_i and A_i scale multiplicatively with their
        // own coordinate. ρ coordinates do (`λ_i = exp(ρ_i)` ⇒ `∂_ρᵢrᵢ = rᵢ`,
        // `∂_ρᵢAᵢ = Aᵢ`); ψ/ext coordinates feed a FROZEN drift (`∂A = 0`) and
        // an affine score, so their second self-derivative is zero here.
        let mut exponential_self_coupling = vec![true; k];
        exponential_self_coupling.extend(std::iter::repeat_n(false, ext_dim));
        Some(compute_kkt_residual_theta_corrections(
            &mode_kernel,
            &score_derivs,
            drift_apply,
            r,
            mode == EvalMode::ValueGradientHessian,
            &active,
            &exponential_self_coupling,
        )?)
    } else {
        None
    };
    if let Some(corrections) = kkt_theta_corrections.as_ref() {
        grad += &corrections.gradient;
    }

    // #2454: publish the ρ-block audit AFTER the KKT/IFT fold, so `total` is
    // the gradient entry the caller actually consumes and
    // `total − (fixed_beta + logdet_h + logdet_s)` is the `kkt` part.
    if capture_rho_parts {
        for part in rho_audit_parts.iter_mut() {
            part.total = grad[part.index];
        }
        crate::estimate::outer_eval_capture::record_certificate_parts(&rho_audit_parts);
        crate::estimate::outer_eval_capture::record_rho_gradient_parts(rho_audit_parts);
    }

    // Extended hyperparameter gradient (ψ/τ coordinates).
    //
    // Uses the SAME outer_gradient_entry() formula as ρ coordinates above.
    //
    // All extended coordinates store canonical fixed-β stationarity
    // derivatives g_i = F_{βi}. IFT gives β_i = -H^{-1}g_i, exactly like
    // the ρ block.
    let ext_grad_entries: Result<Vec<(usize, f64)>, String> = (0..ext_dim)
        .into_par_iter()
        .map(|ext_idx| {
            let coord = &solution.ext_coords[ext_idx];
            let ext_coord_start = std::time::Instant::now();
            let grad_idx = k + ext_idx;

            // Trace term: tr(K · Ḣ_i) where Ḣ_i = B_i + D_β H[−v_i].
            //
            // Kernel choice pairs with the cost:
            //   * Default cost `½ log|H|` (or `Σ log r_ε(σ_j)` under Smooth spectral
            //     regularization) → K = G_ε(H), computed full-space.
            //   * Rank-deficient LAML fix (`hessian_logdet_correction ≠ 0`, #901)
            //     uses cost `½ log|H_pen|₊` over `range(H_pen)`, which pairs with
            //     the intrinsic spectral kernel K = H_pen⁺.
            //
            // `tr(H_pen⁺ · Ḣ)` is the exact pseudo-logdet derivative for the
            // TOTAL drift on a constant-rank stratum: the ψ basis drift `B_i`
            // (whose `range(Sλ(ψ))` rotates with ψ — first-order eigenvector
            // motion cancels, so no `dU/dψ` term exists for the intrinsic
            // object) AND the non-Gaussian IFT correction `D_β H[−v_i]`,
            // which has support on `null(S)` whenever `X` contains an
            // all-ones intercept column. The historical range(S_+)-projected
            // kernel dropped both the penalty-null Schur curvature (ρ sign
            // flips) and the moving-subspace ψ term (~1e5 FD blow-ups).
            // The standard dense assemblies (`build_dense_assembly`,
            // `build_dense_original_assembly`) install no `penalty_subspace_trace`:
            // their operator is exact on H's identified subspace (#2901 V22), so
            // its own `G_ε(H)` with `ε = 0` on the kept eigenpairs IS `H⁺`, and
            // they drop into the `None` arm below.
            let trace_logdet_i = if !incl_logdet_h {
                0.0
            } else if let Some(ref projected_traces) = projected_trace_values {
                projected_traces[k + ext_idx]
            } else if let Some(ref exact_traces) = exact_dense_trace_values {
                exact_traces[k + ext_idx]
            } else {
                let correction = ext_corrections[ext_idx].as_ref();
                let drift = hyper_coord_total_drift_result(&coord.drift, correction, hop.dim());
                match (&solution.penalty_subspace_trace, &drift) {
                    (Some(kernel), DriftDerivResult::Dense(matrix)) => {
                        kernel.trace_projected_logdet(matrix)
                    }
                    (Some(kernel), DriftDerivResult::Operator(op)) => {
                        kernel.trace_operator(op.as_ref())
                    }
                    (None, DriftDerivResult::Dense(matrix)) => hop.trace_logdet_h_k(matrix, None),
                    (None, DriftDerivResult::Operator(op)) => {
                        hop.trace_logdet_operator(op.as_ref())
                    }
                }
            };

            let value = outer_gradient_entry(
                coord.a,
                trace_logdet_i,
                coord.ld_s,
                &solution.dispersion,
                dp_cgrad,
                profiled_scale,
                incl_logdet_h,
                incl_logdet_s,
            );
            log::trace!(
                "[EXT-GRAD] ext_idx={} value={:+.6e} coord.a={:+.6e} trace_logdet={:+.6e} ld_s={:+.6e} incl_h={} incl_s={}",
                ext_idx, value, coord.a, trace_logdet_i, coord.ld_s, incl_logdet_h, incl_logdet_s
            );
            log::debug!(
                "[STAGE] reml_laml ext_coord_trace ext_idx={} elapsed={:.3}s",
                ext_idx,
                ext_coord_start.elapsed().as_secs_f64(),
            );
            Ok((grad_idx, value))
        })
        .collect();
    for (idx, value) in ext_grad_entries? {
        // ACCUMULATE, do not overwrite: the unified `kkt_theta_corrections`
        // block above already folded the ψ/ext KKT-residual correction
        // `−coord.gᵀH⁻¹r + ½(H⁻¹r)ᵀB(H⁻¹r)` into `grad[k + ext_idx]`. A plain
        // `grad[idx] = value` would discard it (the ρ block survives only
        // because its main entries were assigned BEFORE the fold). Discarding
        // it is exactly what left the SAS/mixture link-parameter gradient at
        // the raw capped-β̂ main term `coord.a(β̂)` — collapsing the ε gradient
        // from the true stationary ≈−9 to ≈−0.02 and stalling recovery (#1876).
        // `grad[k + ext_idx]` holds 0 when no correction is active (the fold is
        // skipped), so `+=` is byte-identical to the old assignment there.
        grad[idx] += value;
    }

    // (The ψ/ext gradient KKT-residual correction is folded into `grad` by the
    // unified `kkt_theta_corrections.gradient` block above, and the `+=`
    // accumulation of the ext main entries here PRESERVES it rather than
    // overwriting — see the full-θ correction block before this loop.)

    // Add prior gradient (ρ-only).
    if let Some((_, ref pg, _)) = prior_cost_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += pg;
        }
    }

    // gam#2765: the constrained normalizer's derivative along every outer coordinate, through
    // the one sensitivity state the rest of this gradient reads. The mode response is
    // `β̂̇ = −v` (the evaluator's `v = K a` responds to `−∇F`), the precision moves by the same
    // total drift `Ḣ_j` the log-determinant traces, and on a face the KKT gradient moves by
    // `ġ = M_true β̂̇ + ∂_θ∇F`. Operator-side objects carry the curvature scale `s`, so each is
    // divided by it to reach the unscaled units `C` is priced in.
    // Each coordinate's motion and first-order data, reused by the outer Hessian.
    let mut cone_coordinates: Vec<(
        crate::constrained_posterior::ConeCoordinateMotion,
        crate::constrained_posterior::ConeFirstOrder,
    )> = Vec::new();
    if let Some((input, normalizer)) = cone_normalizer.as_ref() {
        let drifts = build_trace_drifts();
        let y = normalizer.solved_gradient();
        let basis = normalizer.covariance_basis();
        let rho_vs = rho_v_ks
            .as_ref()
            .expect("the constrained normalizer requests every rho mode response");
        // Where `cone_solve` is the kernel's pseudo-inverse its derivative carries the kernel's
        // kept–dropped rotation, read off the drift on the dropped basis (gam#2952). The kernel
        // prices operator units, so the rotation takes the curvature scale like `cone_solve`.
        let pseudo_inverse_kernel = solution.penalty_subspace_trace.as_deref();
        let generator = normalizer.covariance_generator();
        for coordinate in 0..(k + ext_dim) {
            let (response, fixed_beta_rate) = if coordinate < k {
                (&rho_vs[coordinate], &rho_curvature_a_k_betas[coordinate])
            } else {
                (&ext_v_is[coordinate - k], &solution.ext_coords[coordinate - k].g)
            };
            let mode_response = -response;
            let gradient_rate = match &input.gradient_motion {
                ConeGradientMotion::Stationary => Array1::zeros(mode_response.len()),
                ConeGradientMotion::OnFace(stationarity) => {
                    (stationarity.dot(&mode_response) + fixed_beta_rate) / cone_scale
                }
                ConeGradientMotion::Pinned => fixed_beta_rate / cone_scale,
            };
            let drift = &drifts[coordinate];
            let precision_rate_on_y = drift.apply(y) / cone_scale;
            let mut precision_rate_on_basis = Array2::<f64>::zeros(basis.raw_dim());
            for column in 0..basis.ncols() {
                precision_rate_on_basis
                    .column_mut(column)
                    .assign(&(drift.apply(&basis.column(column).to_owned()) / cone_scale));
            }
            let (inverse_rotation_on_gradient, inverse_rotation_on_generator) =
                match pseudo_inverse_kernel {
                    Some(kernel) => {
                        let mut rate_on_dropped = Array2::<f64>::zeros(kernel.dropped_basis.raw_dim());
                        for column in 0..kernel.dropped_basis.ncols() {
                            rate_on_dropped
                                .column_mut(column)
                                .assign(&drift.apply(&kernel.dropped_basis.column(column).to_owned()));
                        }
                        let rotation = kernel.pseudo_inverse_rotation(&rate_on_dropped)?;
                        (
                            rotation.apply(&input.gradient) * cone_scale,
                            rotation.apply_columns(generator) * cone_scale,
                        )
                    }
                    None => (Array1::zeros(y.len()), Array2::zeros(generator.raw_dim())),
                };
            let motion = crate::constrained_posterior::ConeCoordinateMotion {
                mode_response,
                gradient_rate,
                precision_rate_on_y,
                precision_rate_on_basis,
                inverse_rotation_on_gradient,
                inverse_rotation_on_generator,
            };
            let first = normalizer.first_order(&motion, &cone_solve);
            grad[coordinate] += first.derivative;
            cone_coordinates.push((motion, first));
        }
    }

    // KKT projection onto the model's canonical upper face (#197, corrected by
    // #2615).
    //
    // `active_upper_rho_mask` answers a PRIMAL question — "does this coordinate
    // sit on its upper bound?" — and #197 used that answer alone to return
    // exactly 0 for the entry. Sitting on the bound is not activity. At an upper
    // bound the feasible directions are DECREASING rho, so an entry with
    // `dV/drho_k > 0` is feasible descent the search must be allowed to take;
    // only a NEGATIVE entry (whose descent step `-g` leaves the box) is the
    // infeasible bound multiplier. Dropping both made the box a one-way trap:
    // any coordinate that ever touched its upper bound reported zero derivative
    // forever, so no optimizer could leave it, and a seed projected onto the box
    // certified as stationary at iteration 0 with `|g| = 0`. Measured on the
    // penguins multinomial, whose effective-df-floor walls sit ABOVE the seed:
    // all 24 coordinates pinned, `raw_g = 0.0` at each, "→ stationary", and the
    // shipped smoothing parameters equal the walls to the last bit — i.e. the
    // floor constant, not REML, was selecting lambda (#2615).
    //
    // This is the same rule `project_gradient_vector` applies in the optimizer,
    // stated here so the criterion's own reported gradient and the optimizer's
    // active-set verdict cannot disagree.
    for idx in 0..k {
        if upper_active_rho[idx] && grad[idx] < 0.0 {
            log::trace!(
                "[RHO-GRAD] idx={idx} entry {:+.6e} is the infeasible upper-bound multiplier; \
                 projected to 0",
                grad[idx],
            );
            grad[idx] = 0.0;
        }
    }

    if let Some((idx, value)) = grad.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(RemlError::NonFiniteValue {
            reason: format!("REML/LAML gradient contains non-finite entry at index {idx}: {value}"),
        }
        .into());
    }

    // Run the envelope-gradient sanity check *before* the outer-Hessian
    // assembly. The check is intentionally applied after all analytic
    // correction terms have been folded into `grad`; if the final derivative
    // still predicts a sqrt(eps)-step cost change > 4*|cost|, it is not a
    // valid local derivative of this cost surface. The Hessian computed from
    // the same ill-conditioned inner state would also be untrustworthy and
    // would just be discarded by the outer optimizer, and for large-scale
    // custom families the assembly can take 20+ minutes per evaluation.
    // Decide once here, then reuse the verdict for both the gradient and
    // Hessian outputs.
    let cost_scale = cost.abs().max(1.0);
    let resolve_step = f64::EPSILON.sqrt();
    let envelope_inconsistent = grad
        .iter()
        .enumerate()
        .map(|(i, g)| (i, g.abs()))
        .reduce(|a, b| if a.1 >= b.1 { a } else { b })
        .and_then(|(max_idx, max_abs)| {
            let predicted_change = max_abs * resolve_step;
            if max_abs.is_finite() && predicted_change > 4.0 * cost_scale {
                Some((max_idx, max_abs, predicted_change))
            } else {
                None
            }
        });
    // Principled rule: `envelope_inconsistent` is evaluated on the
    // *post-correction* gradient (the `kkt_theta_corrections.gradient`
    // additive block above has already been folded into `grad`). If the
    // predicted √ε-step cost change still exceeds 4·|cost| after that
    // fold, the gradient is invalid as a descent direction. A previous
    // exception kept the gradient when `kkt_residual_correction_active`
    // was true, but that was self-contradictory: the tripwire fires on
    // the same gradient the correction was supposed to have repaired,
    // and when `‖r_proj‖∞ ≈ 0` (the cert-exit contract) the correction
    // `-aᵀ_k q + ½ qᵀA_k q` with `q = H⁻¹·r ≡ 0` is identically zero
    // (see `compute_kkt_residual_theta_corrections`). Active-constraint
    // cases are handled before reaching here by the tangent-space
    // dispatch at the top of this function (`try_tangent_projected_evaluate`,
    // refs Wood 2011 §4; Wood–Pya–Säfken 2016 §3; Marra–Wood 2012 §2).
    let envelope_suppresses_outputs = envelope_inconsistent.is_some();
    if envelope_inconsistent.is_some()
        && matches!(solution.dispersion, DispersionHandling::Fixed { .. })
        && solution.kkt_residual.is_none()
    {
        return Err(RemlError::ContractViolation {
            reason: "REML/LAML fixed-dispersion derivative contract violated: envelope gradient \
                     is inconsistent but no projected KKT residual was supplied. A convergent \
                     custom-family inner path must populate BlockwiseInnerResult::kkt_residual \
                     using the active-set-aware projected residual before requesting analytic \
                     outer derivatives"
                .to_string(),
        }
        .into());
    }
    // (Active-constraint tangent-space dispatch lives at the very top of
    // this function; by the time we reach this point we are already on
    // the post-projection recursion or the unconstrained path.)

    // Outer Hessian (if requested).
    // gam#2765: the constrained normalizer's exact outer Hessian, added to whichever route
    // assembles the rest of it.
    let cone_hessian: Option<Array2<f64>> = match cone_normalizer.as_ref() {
        Some((input, normalizer))
            if mode == EvalMode::ValueGradientHessian && !envelope_suppresses_outputs =>
        {
            let rho_vs = rho_v_ks
                .as_ref()
                .expect("the constrained normalizer requests every rho mode response");
            let mode_responses: Vec<&Array1<f64>> = rho_vs.iter().chain(ext_v_is.iter()).collect();
            let assembly_start = std::time::Instant::now();
            let normalizer_hessian = cone_normalizer_outer_hessian(
                solution,
                input,
                normalizer,
                &cone_coordinates,
                &mode_responses,
                &build_trace_drifts(),
                &curvature_lambdas,
                &rho_curvature_a_k_betas,
                effective_deriv,
                &mode_kernel,
                cone_scale,
            )?;
            log::debug!(
                "[OUTER hessian-elapsed] constrained normalizer k={} ext={} elapsed={:.3}s",
                k,
                ext_dim,
                assembly_start.elapsed().as_secs_f64()
            );
            Some(normalizer_hessian)
        }
        _ => None,
    };
    let hessian = if mode == EvalMode::ValueGradientHessian && !envelope_suppresses_outputs {
        // First, allow the family to short-circuit with its own exact outer
        // Hv operator.  Default `None` keeps the fall-through identical to
        // the historical kernel-based assembly path; CTN/survival/GAMLSS
        // families that implement a directional θθ HVP will return Some(op)
        // here and skip the kernel-based dispatch entirely.
        if let Some(family_op) = effective_deriv.family_outer_hessian_operator() {
            // Family's own exact Hv operator. Emit the same routing markers
            // as the kernel-based path so the bench runner's outer_h
            // aggregation captures this route too — without these the
            // family-op count silently disappears from the verdict, and
            // CTN/survival/GAMLSS fits look like they never built an outer
            // Hessian at all. The "family_op" reason is distinguishable
            // from the kernel-based reasons so the analyzer can tell which
            // representation a particular fit actually used.
            let n_obs = effective_deriv
                .scalar_glm_ingredients()
                .map(|ing| ing.x.nrows())
                .unwrap_or(solution.n_observations);
            let p_dim = hop.dim();
            let k_outer = k + solution.ext_coords.len();
            log::debug!(
                "[OUTER hessian-route] choice=operator reason=family_op \
                 n={n_obs} p={p_dim} k={k_outer} \
                 callback_kernel=false subspace_trace={subspace} \
                 scale_prefers_operator=irrelevant",
                subspace = solution.penalty_subspace_trace.is_some(),
            );
            if family_op.dim() != k_outer {
                return Err(RemlError::DimensionMismatch {
                    reason: format!(
                        "family outer Hessian operator dimension mismatch: got {}, expected {}",
                        family_op.dim(),
                        k_outer
                    ),
                }
                .into());
            }
            let assembly_start = std::time::Instant::now();
            let mut hessian = gam_problem::HessianValue::Operator(family_op);
            // Full-θ correction: the matrix spans (ρ ‖ ψ) = the operator's whole
            // dimension, so this folds the cross-ρψ and ψψ blocks too, not just ρρ.
            if let Some(kkt_hessian) = kkt_theta_corrections
                .as_ref()
                .and_then(|corrections| corrections.hessian.as_ref())
            {
                crate::objective_base::add_rho_block_dense_to_hessian(&mut hessian, kkt_hessian)?;
            }
            if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                crate::objective_base::add_rho_block_dense_to_hessian(&mut hessian, ph)?;
            }
            if let Some(ref normalizer_hessian) = cone_hessian {
                crate::objective_base::add_rho_block_dense_to_hessian(&mut hessian, normalizer_hessian)?;
            }
            log::debug!(
                "[OUTER hessian-elapsed] choice=operator reason=family_op \
                 n={n_obs} p={p_dim} k={k_outer} elapsed={:.3}s",
                assembly_start.elapsed().as_secs_f64(),
            );
            return Ok(RemlLamlResult {
                cost,
                criterion_components,
                ift_residual_energy,
                inner_polish_step,
                gradient: Some(grad),
                hessian,
                rho_mode_response_cols,
                ext_mode_response_cols,
            });
        }
        let hessian_kernel = effective_deriv.outer_hessian_derivative_kernel();
        // Cost selects representation (operator vs dense), not capability.
        // Both representations build the same drifts and pair traces and the
        // operator pays its products on top, so the dense assembly is taken
        // whenever its workspace fits the materialization cap
        // (`outer_hessian_route_plan`).
        //
        // The matrix-free operator path supports both full-space and projected
        // logdet kernels.  When a `penalty_subspace_trace` is installed, the
        // operator traces first/second Hessian drifts through
        // `U_S (U_Sᵀ H U_S)⁻¹ U_Sᵀ`, matching the dense analytic path without
        // forcing p×p assembly solely for rank-deficient penalties.
        let n_obs = effective_deriv
            .scalar_glm_ingredients()
            .map(|ing| ing.x.nrows())
            .unwrap_or(solution.n_observations);
        let p_dim = hop.dim();
        let k_outer = k + solution.ext_coords.len();
        let callback_operator_kernel = matches!(
            hessian_kernel,
            Some(OuterHessianDerivativeKernel::Callback { .. })
        );
        let has_subspace_trace = solution.penalty_subspace_trace.is_some();
        let route_plan = outer_hessian_route_plan(
            p_dim,
            k_outer,
            hessian_kernel.is_some(),
            has_subspace_trace,
        );
        // #740: when the direction-contracted ψψ hook is installed, the operator
        // route is strictly cheaper than dense at every scale — the dense
        // `compute_outer_hessian` path would re-run the `K²` per-pair ψψ
        // assembly the hook exists to avoid, whereas the operator applies the
        // hook once per matvec. So force the operator representation whenever the
        // hook is present (it still requires a kernel to build the operator).
        let use_operator = route_plan.use_operator
            || (solution.contracted_psi_second_order.is_some() && hessian_kernel.is_some());
        let route_choice = route_plan.choice();
        let route_reason = route_plan.reason;
        log::debug!(
            "[OUTER hessian-route] choice={route_choice} reason={route_reason} \
             n={n_obs} p={p_dim} k={k_outer} \
             callback_kernel={callback_operator_kernel} subspace_trace={has_subspace_trace} \
             scale_prefers_operator={} dense_workspace_bytes={}",
            route_plan.scale_prefers_operator,
            route_plan.dense_workspace_bytes,
        );
        let assembly_start = std::time::Instant::now();
        let result = if use_operator {
            let coord_vs_for_hessian = rho_v_ks.as_ref().map(|rho_vs| {
                let mut all = Vec::with_capacity(k + ext_dim);
                all.extend(rho_vs.iter().cloned());
                all.extend(ext_v_is.iter().cloned());
                all
            });
            match build_outer_hessian_operator(
                solution,
                &lambdas,
                effective_deriv,
                hessian_kernel.expect("checked is_some above"),
                coord_vs_for_hessian.as_deref(),
                Some(&coord_corrections),
            ) {
                Ok(op) => {
                    let mut hessian = gam_problem::HessianValue::Operator(Arc::new(op));
                    // Full-θ correction (ρρ + cross-ρψ + ψψ); the matrix is the
                    // operator's whole dimension.
                    if let Some(kkt_hessian) = kkt_theta_corrections
                        .as_ref()
                        .and_then(|corrections| corrections.hessian.as_ref())
                    {
                        crate::objective_base::add_rho_block_dense_to_hessian(
                            &mut hessian,
                            kkt_hessian,
                        )?;
                    }
                    if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                        crate::objective_base::add_rho_block_dense_to_hessian(&mut hessian, ph)?;
                    }
                    if let Some(ref normalizer_hessian) = cone_hessian {
                        crate::objective_base::add_rho_block_dense_to_hessian(
                            &mut hessian,
                            normalizer_hessian,
                        )?;
                    }
                    hessian
                }
                Err(err) => return Err(err.into()),
            }
        } else {
            let reml_workspace = RemlDerivativeWorkspace {
                curvature_lambdas: &curvature_lambdas,
                rho_penalty_a_k_betas: &rho_penalty_a_k_betas,
                rho_curvature_a_k_betas: &rho_curvature_a_k_betas,
                rho_v_ks: rho_v_ks.as_deref(),
                ext_v_is: Some(ext_v_is.as_slice()),
                coord_corrections: &coord_corrections,
            };
            match compute_outer_hessian(
                solution,
                rho,
                &lambdas,
                hop,
                effective_deriv,
                Some(&reml_workspace),
            ) {
                Ok(mut h) => {
                    // KKT-residual Hessian correction. The correction adds back
                    // `r_iᵀKr_j + C_ij`, valid only where the base outer Hessian
                    // already carries the exact-KKT profile term `−r_iᵀKr_j`.
                    // The dense path computes that profile term for the cross-ρψ
                    // and ψψ blocks ONLY when the family supplies the pair
                    // callbacks (`rho_ext_pair_fn` / `ext_coord_pair_fn`); absent
                    // them those blocks are left at zero, so applying the full-θ
                    // correction there would inject an uncancelled profile term.
                    // Apply the full k_outer × k_outer correction when the cross/
                    // ψψ blocks exist (or there are no ext coords); otherwise add
                    // only the ρρ sub-block, exactly as before.
                    if let Some(kkt_hessian) = kkt_theta_corrections
                        .as_ref()
                        .and_then(|corrections| corrections.hessian.as_ref())
                    {
                        let dense_has_ext_blocks = ext_dim == 0
                            || (solution.rho_ext_pair_fn.is_some()
                                && solution.ext_coord_pair_fn.is_some());
                        if dense_has_ext_blocks {
                            h += kkt_hessian;
                        } else {
                            let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                            sl += &kkt_hessian.slice(ndarray::s![..k, ..k]);
                        }
                    }
                    // gam#2765: the constrained normalizer's Hessian, on the blocks this route
                    // assembles.
                    if let Some(ref normalizer_hessian) = cone_hessian {
                        let dense_has_ext_blocks = ext_dim == 0
                            || (solution.rho_ext_pair_fn.is_some()
                                && solution.ext_coord_pair_fn.is_some());
                        if dense_has_ext_blocks {
                            h += normalizer_hessian;
                        } else {
                            let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                            sl += &normalizer_hessian.slice(ndarray::s![..k, ..k]);
                        }
                    }
                    // Add prior Hessian (second derivatives of the soft prior on ρ, ρ-only).
                    if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                        let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                        sl += ph;
                    }
                    gam_problem::HessianValue::Dense(h)
                }
                Err(err) => return Err(err.into()),
            }
        };
        log::debug!(
            "[OUTER hessian-elapsed] choice={route_choice} reason={route_reason} \
             n={n_obs} p={p_dim} k={k_outer} elapsed={:.3}s",
            assembly_start.elapsed().as_secs_f64(),
        );
        result
    } else {
        gam_problem::HessianValue::Unavailable
    };

    // Envelope-gradient sanity tripwire — last line of defense.
    //
    // The post-IFT-correction gradient is what `envelope_inconsistent` is
    // computed on (the `kkt_theta_corrections.gradient` block was folded in
    // earlier in this function). If the predicted √ε-step cost change
    // still exceeds 4·|cost| after that fold, the gradient is invalid as
    // a descent direction. Suppress so the outer optimizer rejects the
    // seed (analytic gradient unavailable). Threshold ratio > 4 keeps healthy
    // near-stationary gradients (|g|∞ ≈ √ε·|cost|, ratio ≈ 1) from
    // tripping.
    let gradient_out = match envelope_inconsistent {
        Some((max_idx, max_abs, predicted_change)) => {
            // Self-diagnosing warning. The three gates that control whether
            // the IFT correction can run all map to observable booleans, so
            // the next failing run pinpoints the cause without another
            // debugging round.
            let kkt_some = solution.kkt_residual.is_some();
            let dispersion_label = match &solution.dispersion {
                DispersionHandling::Fixed { .. } => "Fixed",
                DispersionHandling::ProfiledGaussian => "ProfiledGaussian",
            };
            let kernel_present = solution.penalty_subspace_trace.is_some();
            log::debug!(
                "[reml_laml envelope-gradient consistency] |g|∞ = {:.3e} at coord {} predicts \
                 |Δcost| ≈ {:.3e} along a √ε step while |cost| = {:.3e} (ratio {:.2e}). \
                 Envelope formula contaminated by inner KKT residual on ill-conditioned H block; \
                 marking analytic gradient unavailable so outer optimizer does not chase a \
                 mathematically impossible descent direction. Outer-Hessian assembly skipped on \
                 this evaluation to avoid spending wall-clock on a result the optimizer would \
                 discard. \
                 IFT-gate diagnostics: kkt_residual.is_some()={} (must be true; this is the \
                 projected-KKT residual the inner solver hands over), dispersion={} (must be \
                 `Fixed` for the LAML IFT identity to hold), penalty_subspace_trace.is_some()={} \
                 (when true the residual is reduced into the kernel's identified range before \
                 the mode-response solve; when false no range reduction precedes the solve, \
                 which is numerically unreliable on near-singular H). \
                 If kkt_residual.is_some()=false under fixed dispersion, the convergent inner \
                 path forgot to populate `BlockwiseInnerResult::kkt_residual` (call \
                 `exact_newton_joint_kkt_residual_for_ift` on return) \
                 and this evaluation is a contract error. If kkt_residual.is_some()=true and \
                 the warning still fires, the projected-residual correction was insufficient: \
                 the post-correction gradient is still inconsistent and must not be handed to \
                 the outer optimizer.",
                max_abs,
                max_idx,
                predicted_change,
                cost_scale,
                predicted_change / cost_scale,
                kkt_some,
                dispersion_label,
                kernel_present,
            );
            None
        }
        None => Some(grad),
    };

    Ok(RemlLamlResult {
        cost,
        criterion_components,
        ift_residual_energy,
        inner_polish_step,
        gradient: gradient_out,
        hessian,
        rho_mode_response_cols,
        ext_mode_response_cols,
    })
}

/// The constrained Laplace normalizer's exact outer Hessian (gam#2765).
///
/// Each pair `(i, j)` moves the state the normalizer reads at second order: the mode by
/// `β̈_ij = K·rhs_ij` (the same second-order stationarity right-hand side the log-determinant's
/// Hessian solves, with `β̂̇ = −v`), the KKT gradient by `g̈_ij = M_true β̈_ij − rhs_ij` (zero at an
/// unconstrained mode, `−rhs_ij` when every direction is pinned), and the precision by its second
/// total drift `M̈_ij = ∂²M|_β + D_β(∂M)[β̂̇] + D²_βM[β̂̇_i, β̂̇_j] + D_βM[β̈_ij]` — the three pieces the
/// log-determinant's pair trace sums, here applied to `y = M⁻¹g` and the columns of `R = M⁻¹Aᵀ`.
/// Cross and `ψψ` pairs are assembled only where the family supplies their fixed-β pair objects,
/// exactly the pairs the dense Hessian assembles. Operator-side objects carry the curvature scale.
fn cone_normalizer_outer_hessian(
    solution: &InnerSolution<'_>,
    input: &ConeNormalizerInput,
    normalizer: &crate::constrained_posterior::ConeNormalizer,
    coordinates: &[(
        crate::constrained_posterior::ConeCoordinateMotion,
        crate::constrained_posterior::ConeFirstOrder,
    )],
    mode_responses: &[&Array1<f64>],
    drifts: &[DriftDerivResult],
    curvature_lambdas: &[f64],
    curvature_a_k_betas: &[Array1<f64>],
    effective_deriv: &dyn HessianDerivativeProvider,
    mode_kernel: &ThetaModeResponseKernel<'_>,
    scale: f64,
) -> Result<Array2<f64>, RemlLamlError> {
    let k = curvature_lambdas.len();
    let total = mode_responses.len();
    let y = normalizer.solved_gradient();
    let basis = normalizer.covariance_basis();
    let mode_rhs_correction = effective_deriv.mode_response_rhs_correction();
    // The family's fixed-β pair objects, fetched across the pool as the dense Hessian fetches its
    // own; the solution memoizes them, so the log-determinant's Hessian reads these same objects.
    let pair_indices: Vec<(usize, usize, usize, usize)> = (0..total)
        .flat_map(|i| (i..total).map(move |j| (i, j)))
        .filter_map(|(i, j)| {
            let ej = j.checked_sub(k)?;
            Some((i, j, i.checked_sub(k).unwrap_or(i), ej))
        })
        .collect();
    let fetched: Vec<Option<gam_problem::HyperCoordPair>> = {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        let (rho_ext_pair_fn, ext_pair_fn) =
            (solution.rho_ext_pair_fn.as_ref(), solution.ext_coord_pair_fn.as_ref());
        pair_indices
            .par_iter()
            .map(|&(i, _, first, second)| {
                let pair_fn = if i < k { rho_ext_pair_fn } else { ext_pair_fn };
                pair_fn
                    .map(|pair_fn| gam_problem::with_nested_parallel(|| pair_fn(first, second)))
                    .transpose()
            })
            .collect::<Result<_, String>>()?
    };
    let mut pairs: std::collections::HashMap<(usize, usize), Option<gam_problem::HyperCoordPair>> =
        pair_indices
            .iter()
            .zip(fetched)
            .map(|(&(i, j, _, _), pair)| ((i, j), pair))
            .collect();
    // The second-order stationarity right-hand side `M β̈_ij = rhs_ij` and its response, pair by
    // pair. A pair the family supplies no object for is one the dense Hessian leaves unassembled.
    struct PairState {
        i: usize,
        j: usize,
        pair: Option<gam_problem::HyperCoordPair>,
        rhs: Array1<f64>,
        second_response: Array1<f64>,
    }
    let mut states: Vec<PairState> = Vec::new();
    for i in 0..total {
        for j in i..total {
            let v_i = mode_responses[i];
            let v_j = mode_responses[j];
            let ext_i = i.checked_sub(k);
            let ext_j = j.checked_sub(k);
            // With i ≤ j an ext first index implies an ext second index.
            let pair = match ext_j {
                None => None,
                Some(_) => match pairs.remove(&(i, j)).flatten() {
                    Some(pair) => Some(pair),
                    None => continue,
                },
            };
            let mut rhs = drifts[j].apply(v_i);
            match ext_i {
                None => rhs += &solution.penalty_coords[i].scaled_matvec(v_j, curvature_lambdas[i]),
                Some(ei) => solution.ext_coords[ei].drift.scaled_add_apply(v_j.view(), 1.0, &mut rhs),
            }
            match pair.as_ref() {
                Some(pair) => rhs -= &pair.g,
                None if i == j => rhs -= &curvature_a_k_betas[i],
                None => {}
            }
            if let Some(correction) = &mode_rhs_correction {
                rhs += &correction(ext_i, ext_j, v_i, v_j)?;
            }
            let second_response = mode_kernel.respond_one(&rhs);
            states.push(PairState { i, j, pair, rhs, second_response });
        }
    }
    // `D²_βM[β̂̇_i, β̂̇_j] + D_βM[β̈_ij]` for every pair in one call where the family fuses the row
    // walk across pairs, otherwise across the pool.
    let corrections: Vec<Option<DriftDerivResult>> = if effective_deriv.has_corrections() {
        let triples: Vec<(Array1<f64>, Array1<f64>, Array1<f64>)> = states
            .iter()
            .map(|state| {
                (
                    mode_responses[state.i].clone(),
                    mode_responses[state.j].clone(),
                    state.second_response.clone(),
                )
            })
            .collect();
        if effective_deriv.has_batched_hessian_second_derivative_corrections() {
            effective_deriv.hessian_second_derivative_corrections_result(&triples)?
        } else {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
            triples
                .par_iter()
                .map(|(v_k, v_l, u_kl)| {
                    gam_problem::with_nested_parallel(|| {
                        effective_deriv.hessian_second_derivative_correction_result(v_k, v_l, u_kl)
                    })
                })
                .collect::<Result<_, String>>()?
        }
    } else {
        states.iter().map(|_| None).collect()
    };
    // `D_β(∂M/∂ψ_e)[β̂̇_c]` depends on the ext coordinate and the moving one, not on the pair: each
    // is formed once.
    let drift_keys: Vec<(usize, usize)> = {
        let mut keys: Vec<(usize, usize)> = states
            .iter()
            .flat_map(|state| {
                let moving = |ext: Option<usize>, coordinate: usize| {
                    ext.filter(|&e| solution.ext_coords[e].b_depends_on_beta).map(|e| (e, coordinate))
                };
                [moving(state.i.checked_sub(k), state.j), moving(state.j.checked_sub(k), state.i)]
            })
            .flatten()
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    };
    let moving_drift_values: Vec<Option<DriftDerivResult>> = match solution.fixed_drift_deriv.as_ref() {
        Some(drift_fn) => {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
            drift_keys
                .par_iter()
                .map(|&(ext, coordinate)| {
                    gam_problem::with_nested_parallel(|| {
                        drift_fn(ext, &mode_responses[coordinate].mapv(|value| -value))
                    })
                })
                .collect::<Result<_, String>>()?
        }
        None => drift_keys.iter().map(|_| None).collect(),
    };
    let moving_drift_at: std::collections::HashMap<(usize, usize), &DriftDerivResult> = drift_keys
        .iter()
        .zip(moving_drift_values.iter())
        .filter_map(|(&key, value)| value.as_ref().map(|drift| (key, drift)))
        .collect();
    // Where `M⁻¹` is the kernel's kept-spectrum pseudo-inverse, each pair also carries its second
    // rotation (`PenaltySubspaceTrace::pseudo_inverse_second_rotation`), read off the drifts on the
    // dropped basis in operator units and scaled like the solves (gam#2952).
    let pseudo_inverse_kernel = solution.penalty_subspace_trace.as_deref();
    let probes = pseudo_inverse_kernel.map(|_| {
        let generator = normalizer.covariance_generator();
        let mut probes = Array2::<f64>::zeros((y.len(), 1 + generator.ncols()));
        probes.column_mut(0).assign(&input.gradient);
        probes.slice_mut(ndarray::s![.., 1..]).assign(generator);
        probes
    });
    let dropped_rates: Vec<Array2<f64>> = match pseudo_inverse_kernel {
        Some(kernel) => drifts
            .iter()
            .map(|drift| {
                let mut rate = Array2::<f64>::zeros(kernel.dropped_basis.raw_dim());
                for column in 0..kernel.dropped_basis.ncols() {
                    rate.column_mut(column)
                        .assign(&drift.apply(&kernel.dropped_basis.column(column).to_owned()));
                }
                rate
            })
            .collect(),
        None => Vec::new(),
    };
    let mut hessian = Array2::<f64>::zeros((total, total));
    for (state, correction) in states.iter().zip(corrections.iter()) {
        let (i, j) = (state.i, state.j);
        // M̈_ij applied to x, in the operator's scaled units.
        let fixed_beta_second_drift = |x: &Array1<f64>| -> Array1<f64> {
            match state.pair.as_ref() {
                Some(pair) => match pair.b_operator.as_ref() {
                    Some(operator) => operator.mul_vec(x),
                    None => pair.b_mat.dot(x),
                },
                None if i == j => solution.penalty_coords[i].scaled_matvec(x, curvature_lambdas[i]),
                None => Array1::zeros(x.len()),
            }
        };
        let mut moving_drifts: Vec<&DriftDerivResult> = Vec::new();
        if let Some(ei) = i.checked_sub(k)
            && let Some(drift) = moving_drift_at.get(&(ei, j))
        {
            moving_drifts.push(drift);
        }
        if let Some(ej) = j.checked_sub(k)
            && let Some(drift) = moving_drift_at.get(&(ej, i))
        {
            moving_drifts.push(drift);
        }
        if let Some(drift) = correction.as_ref() {
            moving_drifts.push(drift);
        }
        let second_drift = |x: &Array1<f64>| -> Array1<f64> {
            let mut out = fixed_beta_second_drift(x);
            for drift in &moving_drifts {
                out += &drift.apply(x);
            }
            out / scale
        };
        let gradient_rate = match &input.gradient_motion {
            ConeGradientMotion::Stationary => Array1::zeros(state.rhs.len()),
            ConeGradientMotion::OnFace(stationarity) => {
                (stationarity.dot(&state.second_response) - &state.rhs) / scale
            }
            ConeGradientMotion::Pinned => -&state.rhs / scale,
        };
        let mut precision_rate_on_basis = Array2::<f64>::zeros(basis.raw_dim());
        for column in 0..basis.ncols() {
            precision_rate_on_basis
                .column_mut(column)
                .assign(&second_drift(&basis.column(column).to_owned()));
        }
        let (inverse_rotation_on_gradient, inverse_rotation_on_generator) =
            match (pseudo_inverse_kernel, probes.as_ref()) {
                (Some(kernel), Some(probes)) => {
                    let mut second_on_dropped = Array2::<f64>::zeros(kernel.dropped_basis.raw_dim());
                    for column in 0..kernel.dropped_basis.ncols() {
                        second_on_dropped.column_mut(column).assign(
                            &(second_drift(&kernel.dropped_basis.column(column).to_owned()) * scale),
                        );
                    }
                    let turned = kernel.pseudo_inverse_second_rotation(
                        &|v: &Array1<f64>| drifts[i].apply(v),
                        &|v: &Array1<f64>| drifts[j].apply(v),
                        &dropped_rates[i],
                        &dropped_rates[j],
                        &second_on_dropped,
                        probes,
                    )? * scale;
                    (
                        turned.column(0).to_owned(),
                        turned.slice(ndarray::s![.., 1..]).to_owned(),
                    )
                }
                _ => (Array1::zeros(y.len()), Array2::zeros(normalizer.covariance_generator().raw_dim())),
            };
        let pair_motion = crate::constrained_posterior::ConePairMotion {
            mode_response: state.second_response.clone(),
            gradient_rate,
            precision_rate_on_y: second_drift(y),
            precision_rate_on_basis,
            inverse_rotation_on_gradient,
            inverse_rotation_on_generator,
        };
        let value = normalizer
            .second_order(
                &coordinates[i].0,
                &coordinates[i].1,
                &coordinates[j].0,
                &coordinates[j].1,
                &pair_motion,
            )
            .map_err(RemlLamlError::ConeNormalizer)?;
        hessian[[i, j]] = value;
        hessian[[j, i]] = value;
    }
    Ok(hessian)
}
