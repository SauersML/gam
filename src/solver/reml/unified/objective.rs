use super::*;
use crate::solver::estimate::smooth_floor_dp;

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
/// in the same function scope, using the same `HessianOperator`, the same
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
pub fn reml_laml_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<RemlLamlResult, String> {
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
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();
    let hop = &*solution.hessian_op;
    let upper_active_rho = active_upper_rho_mask(rho);

    // ─── Shared intermediates (computed once, used by both cost and gradient) ───

    let log_det_h = hop.logdet() + solution.hessian_logdet_correction;
    let log_det_s = solution.penalty_logdet.value;
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
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            let phi = dp_c / denom;

            let cost = dp_c / (2.0 * phi)
                + 0.5 * (log_det_h - log_det_s)
                + (denom / 2.0) * (2.0 * std::f64::consts::PI * phi).ln()
                - solution.gaussian_weight_log_sum_half;

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
            let mut cost =
                cost_logdet_diff + (-solution.log_likelihood) + 0.5 * solution.penalty_quadratic;
            if *include_logdet_h {
                // Firth `−½ log|J|`: VALUE here, ρ-DERIVATIVE folded into the
                // per-coordinate LAML trace (`a_i`) below — already paired by
                // construction, so it is NOT routed through `GuardedCorrection`.
                // The Tierney–Kadane correction, by contrast, is a genuinely
                // loose `(tk_correction, tk_gradient)` pair; it flows through a
                // single `GuardedCorrection` object (built below) whose one
                // `include` guard gates BOTH its value and its ρ-gradient, so
                // the two can never desync.
                cost -= solution
                    .firth
                    .as_ref()
                    .map_or(0.0, ExactJeffreysTerm::value);
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
    log::debug!(
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
    let mut ift_residual_energy: Option<f64> = None;
    let mut inner_polish_step: Option<Array1<f64>> = None;
    if let Some(r) = kkt_residual_vec
        .as_ref()
        .filter(|_| kkt_residual_correction_active)
        .map(|r| r.as_ref())
    {
        // Cost-side IFT correction `−½ rᵀ H⁻¹ r`. When the rank-deficient
        // LAML fix is active (`penalty_subspace_trace = Some`), the
        // mathematically correct inverse here is the Moore-Penrose
        // pseudo-inverse on the kernel's identified subspace (`range(H_pen)`
        // in the intrinsic #901 form; historically `range(S_+)`) — not the
        // regularized full `H⁻¹`. The full-H solve at near-singular boundary
        // states (the large-scale survival marginal-slope pathology) amplifies
        // floating-point noise in `r` along sub-threshold directions by
        // `1/σ_min(H) ≈ 10¹²`, which then propagates into a 10¹³-magnitude
        // gradient component and traps the outer optimizer at max-iter.
        // The spectral pseudo-inverse zeroes any component of `r` on
        // directions excluded from the cost's pseudo-logdet before the
        // inverse is applied — the same threshold for value and correction —
        // recovering the honest correction.
        let (cost_correction, branch) =
            if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                (-0.5_f64 * kernel.bilinear_pseudo_inverse(r, r), "projected")
            } else {
                let mut rhs = Array2::<f64>::zeros((hop.dim(), 1));
                rhs.column_mut(0).assign(r);
                let w_mat = hop.solve_multi(&rhs);
                let w: Array1<f64> = w_mat.column(0).to_owned();
                let cost_correction = -0.5_f64 * r.view().dot(&w);
                inner_polish_step = Some(w);
                (cost_correction, "full_h")
            };
        let residual_energy = -cost_correction;
        log::info!(
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

    // Build the Tierney–Kadane frozen-curvature correction as ONE object that
    // owns the single `include` guard. Its VALUE must land on the cost before
    // the `EvalMode::ValueOnly` early return below, and its ρ-GRADIENT lands on
    // `grad` further down — but BOTH read `include` from this same object, so
    // the two contributions cannot desync (the structural cure for the
    // objective↔gradient drift class behind #752/#748/#808). Historically the
    // value was added inside the dispersion match and the gradient added at the
    // gradient site, each under an independently hand-written
    // `if include_logdet_h { … }` guard — exactly the divergence shape this
    // type makes impossible.
    let tk_correction = GuardedCorrection::new(
        solution.tk_correction,
        solution.tk_gradient.clone(),
        incl_logdet_h,
    );
    tk_correction.apply_value(&mut cost);

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
            ift_residual_energy,
            inner_polish_step,
            gradient: None,
            hessian: crate::solver::outer_strategy::HessianResult::Unavailable,
            rho_mode_response_cols: None,
            ext_mode_response_cols: None,
        });
    }

    log::info!(
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
                log::warn!("BarrierDerivativeProvider skipped (infeasible): {e}");
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
    let need_family_corrections = effective_deriv.has_corrections();
    let need_rho_mode_responses = need_family_corrections || mode == EvalMode::ValueGradientHessian;
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
        let mode_kernel = ThetaModeResponseKernel::select(
            solution.penalty_subspace_trace.as_deref(),
            solution.active_constraints.as_deref(),
            hop,
        );
        let mut rhs_stack = Array2::<f64>::zeros((dim, total_cols));
        let mut col_idx = 0;
        if need_rho_mode_responses {
            for (idx, a_k_beta) in rho_curvature_a_k_betas.iter().enumerate() {
                if !upper_active_rho[idx] {
                    rhs_stack.column_mut(col_idx).assign(a_k_beta);
                }
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
            log::info!(
                "[STAGE] reml_laml coord_corrections mode=batched(row-parallel) k={} ext_dim={} n={} dim={} work={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim(),
                correction_work
            );
            // Named heartbeat scope so the active-scope line attributes the
            // coord_corrections wall time (the biobank's dominant REML stage).
            let coord_corr_scope = crate::process_monitor::track_scope(format!(
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
                log::info!(
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

    // --- Stochastic trace estimation decision ---
    //
    // Hutchinson traces based on H^{-1} are only valid for logdet-gradient
    // terms on backends where the logdet kernel is exactly H^{-1}.
    // Smooth spectral regularization uses G_eps(H) instead, so those backends
    // must stay on the exact trace path.  The rank-deficient LAML fix also
    // replaces the kernel with the projected `U_S · (U_Sᵀ H U_S)⁻¹ · U_Sᵀ`,
    // which the Hutchinson path cannot produce — stay exact when it is active.
    let total_p = hop.dim();
    let use_stochastic_traces = can_use_stochastic_logdet_hinv_kernel(hop, total_p, incl_logdet_h)
        && solution.penalty_subspace_trace.is_none();

    // When using stochastic traces, pre-collect all H_k drifts (both rho and
    // ext coordinates) and batch them through a single StochasticTraceEstimator.
    // This amortizes the H^{-1} solve cost: ONE solve per probe, shared across
    // all k + ext_dim coordinates. The collector must inspect the fully
    // assembled drift (base coordinate plus SCOP/family correction) before
    // deciding dense vs operator; checking only the base coordinate misses
    // matrix-free derivative corrections and silently densifies them.
    let stochastic_trace_values: Option<Vec<f64>> = if use_stochastic_traces {
        let mut dense_matrices: Vec<Array2<f64>> = Vec::with_capacity(k + ext_dim);
        let mut operators: Vec<Arc<dyn HyperOperator>> = Vec::new();
        let mut coord_has_operator = Vec::with_capacity(k + ext_dim);

        // rho-coordinates: H_k = A_k + correction(v_k)
        for idx in 0..k {
            match penalty_total_drift_result(
                &solution.penalty_coords[idx],
                curvature_lambdas[idx],
                rho_corrections[idx].as_ref(),
            ) {
                DriftDerivResult::Dense(matrix) => {
                    dense_matrices.push(matrix);
                    coord_has_operator.push(false);
                }
                DriftDerivResult::Operator(op) => {
                    operators.push(op);
                    coord_has_operator.push(true);
                }
            }
        }

        // ext-coordinates: H_i = B_i + D_beta H[-v_i].
        for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
            let correction = ext_corrections[ext_idx].as_ref();
            match hyper_coord_total_drift_result(&coord.drift, correction, hop.dim()) {
                DriftDerivResult::Dense(matrix) => {
                    dense_matrices.push(matrix);
                    coord_has_operator.push(false);
                }
                DriftDerivResult::Operator(op) => {
                    operators.push(op);
                    coord_has_operator.push(true);
                }
            }
        }

        let dense_refs: Vec<&Array2<f64>> = dense_matrices.iter().collect();
        let generic_ops: Vec<&dyn HyperOperator> = operators.iter().map(|op| op.as_ref()).collect();
        let implicit_ops: Vec<&ImplicitHyperOperator> =
            operators.iter().filter_map(|op| op.as_implicit()).collect();

        // ── Block 2.5: GPU-adaptive Hutchinson bypass.
        //
        // When the gate fires we replace the CPU stochastic estimator with
        // the device-resident path in `gpu::reml_trace::evidence_traces_adaptive`,
        // which factors `H` once on device (potrf), then for an adaptive
        // K-schedule (16→32→64→128) solves `H W = Z` in one batched potrs
        // and reduces `q_{j,k} = z_k^T H_j w_k` on device. CRN is preserved
        // because the SplitMix probe RNG hashes `(seed, k_index, i)`
        // statelessly. The gate requires:
        //   * `operators.is_empty()`  — every drift is a dense `p × p`
        //     matrix, so each `H_j` can be packed as
        //     `DerivativeHessian::Dense(...)` without re-walking penalty
        //     operators on device.
        //   * `gpu::reml_trace::should_bypass_cpu_with_gpu_adaptive(...)`
        //     — `p ≥ 512`, plain SPD logdet kernel, dense backend, projected
        //     penalty subspace inactive.
        // Both must hold; otherwise we fall through to the existing CPU
        // path. The dispatch entry inside `evidence_traces_adaptive`
        // itself falls back to the SplitMix CPU reference when the CUDA
        // runtime is absent — so non-GPU hosts keep the existing
        // estimator behaviour even with the gate set.
        let gpu_bypass_raw_traces: Option<Vec<f64>> = if operators.is_empty()
            && crate::gpu::reml_trace::should_bypass_cpu_with_gpu_adaptive(
                total_p,
                hop.as_exact_dense_spectral().is_some(),
                hop.logdet_traces_match_hinv_kernel() && hop.is_dense(),
                hop.prefers_stochastic_trace_estimation(),
                solution.penalty_subspace_trace.is_some(),
            ) {
            use crate::gpu::reml_trace::{
                DerivativeHessian, HUTCHINSON_ADAPTIVE_REL_TOL, HUTCHINSON_ADAPTIVE_TAU_REL,
                ProbeSeed, evidence_traces_adaptive,
            };
            // Build the dense SPD H from the operator's spectral
            // decomposition (same path the LAML/tangent-projection code
            // uses, so the matrix is bit-identical to what other backends
            // see). The unwrap is safe: the gate verified
            // `hop.as_exact_dense_spectral().is_some()`.
            let dense_op = hop
                .as_exact_dense_spectral()
                .expect("gate guarantees as_exact_dense_spectral().is_some()");
            let h_dense = assemble_h_raw_dense(dense_op);
            let derivatives: Vec<DerivativeHessian<'_>> = dense_refs
                .iter()
                .map(|m| DerivativeHessian::Dense(m.view()))
                .collect();
            // Seed the device RNG. We use a single deterministic seed
            // across REML iterations: the SplitMix probe RNG is stateless
            // (`(seed, k_index, i) → ±1`), so the K=16, 32, 64, 128
            // adaptive schedule within one call already enjoys CRN, and
            // the cross-iteration bias from probe reuse is bounded by the
            // adaptive SE check (a fixed-seed Hutchinson is still
            // unbiased; only its variance across REML trajectories is
            // correlated). Matching `StochasticTraceConfig::default().seed`
            // keeps the GPU probes bit-identical to the CPU reference for
            // parity testing (Block 2.6 test "same-probes CPU vs GPU").
            let probe_seed = ProbeSeed::default();
            match evidence_traces_adaptive(
                h_dense.view(),
                derivatives,
                None,
                probe_seed,
                HUTCHINSON_ADAPTIVE_REL_TOL,
                HUTCHINSON_ADAPTIVE_TAU_REL,
            ) {
                Ok(evidence) => Some(evidence.traces.to_vec()),
                Err(_) => None,
            }
        } else {
            None
        };

        let raw_traces = if let Some(gpu_traces) = gpu_bypass_raw_traces {
            gpu_traces
        } else if generic_ops.is_empty() {
            stochastic_trace_hinv_products_with_floor(
                hop,
                StochasticTraceTargets::Dense(&dense_refs),
                Some(Arc::clone(&solution.stochastic_trace_state)),
            )
        } else if generic_ops.len() == implicit_ops.len() {
            stochastic_trace_hinv_products_with_floor(
                hop,
                StochasticTraceTargets::Structural {
                    dense_matrices: &dense_refs,
                    implicit_ops: &implicit_ops,
                },
                Some(Arc::clone(&solution.stochastic_trace_state)),
            )
        } else {
            stochastic_trace_hinv_products_with_floor(
                hop,
                StochasticTraceTargets::Mixed {
                    dense_matrices: &dense_refs,
                    operators: &generic_ops,
                },
                Some(Arc::clone(&solution.stochastic_trace_state)),
            )
        };

        let mut result = Vec::with_capacity(k + ext_dim);
        let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
        let mut dense_cursor = 0usize;
        let mut operator_cursor = n_dense_total;
        for &has_operator in &coord_has_operator {
            if has_operator {
                result.push(raw_traces[operator_cursor]);
                operator_cursor += 1;
            } else {
                result.push(raw_traces[dense_cursor]);
                dense_cursor += 1;
            }
        }
        Some(result)
    } else {
        None
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
        if incl_logdet_h && stochastic_trace_values.is_none() {
            solution
                .penalty_subspace_trace
                .as_ref()
                .map(|kernel| penalty_subspace_trace_drifts_batched(kernel, &build_trace_drifts()))
        } else {
            None
        };

    let exact_dense_trace_values: Option<Vec<f64>> =
        if incl_logdet_h && stochastic_trace_values.is_none() && projected_trace_values.is_none() {
            hop.as_exact_dense_spectral()
                .map(|ds| dense_spectral_trace_logdet_drifts_batched(ds, &build_trace_drifts()))
        } else {
            None
        };

    // ── Gradient: one shared formula for ALL coordinate types ──
    //
    // Both ρ and ext coordinates are processed through outer_gradient_entry()
    // so that the three-term formula (penalty + trace − det) is written once.

    // Exact trace batching for operator-backed ρ corrections.  The hot large-scale
    // GAMLSS path has many Duchon smoothing coordinates whose correction
    // operators share the same design and spectral factor.  Evaluating them one
    // by one repeats the same `X·F` projection and row-kernel setup for every
    // coordinate.  The trace is linear, so split `tr(K·(A_i + C_i))` into the
    // cheap penalty part plus a batched exact vector of `tr(K·C_i)` values.
    // This preserves the full first-order gradient (no iteration caps or
    // stochastic approximation) while collapsing the per-coordinate trace pass.
    let rho_operator_correction_traces: Option<Vec<Option<f64>>> = if incl_logdet_h
        && stochastic_trace_values.is_none()
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

    let rho_grad_entries: Vec<(usize, f64)> = (0..k)
        .into_par_iter()
        .map(|idx| {
            // Active upper-bound projection (Option A in #197):
            //
            // When ρ_k has been pinned at its upper bound by the outer
            // bound-constrained solver, the IFT mode response `v_k` is
            // already zeroed above and the IFT residual correction in
            // `compute_kkt_residual_rho_corrections` is masked to 0 for
            // this coord. The envelope main block must use the SAME
            // gradient-projection convention — otherwise the trace term
            // `½·tr(K·λ_k S_k)` and the penalty quadratic `½·λ_k β'S_kβ`
            // produce a phantom non-zero gradient component along a
            // frozen axis, mixing constrained (v_k = 0) and unconstrained
            // (λ_k held active) terms in the same scalar — which matches
            // the analytic derivative of neither cost. Returning exactly
            // 0 here aligns the envelope, the IFT correction, and the
            // box-constrained optimizer's gradient projection (#197).
            if upper_active_rho[idx] {
                log::trace!(
                    "[RHO-GRAD] idx={} value=0.0 (upper-bound projection, see #197)",
                    idx
                );
                return (idx, 0.0);
            }

            let coord = &solution.penalty_coords[idx];

            // Cost derivative for the shifted penalty:
            // a_i = ½ λₖ (β̂ - μₖ)' Sₖ (β̂ - μₖ).
            //
            // The β-gradient derivative is λₖSₖ(β̂-μₖ); dotting it with β̂
            // would drop the μₖ'λₖSₖμₖ half of the chain rule.
            let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambdas[idx]);

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
            let trace_logdet_i = if !incl_logdet_h {
                0.0
            } else if let Some(ref stoch_traces) = stochastic_trace_values {
                stoch_traces[idx]
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
                let penalty_trace = if coord.is_block_local() {
                    let (block, start, end) = coord.scaled_block_local(1.0);
                    hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
                } else {
                    penalty_total_drift_result(coord, curvature_lambdas[idx], None)
                        .trace_logdet(hop)
                };
                penalty_trace + correction_trace
            } else if coord.is_block_local() && rho_corrections[idx].is_none() {
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
            let value = outer_gradient_entry(
                a_i,
                trace_logdet_i,
                solution.penalty_logdet.first[idx],
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
                "[RHO-GRAD] idx={} value={:+.6e} a_i={:+.6e} trace_logdet={:+.6e} ld_s_first={:+.6e} incl_h={} incl_s={}",
                idx, value, a_i, trace_logdet_i, solution.penalty_logdet.first[idx], incl_logdet_h, incl_logdet_s
            );
            (idx, value)
        })
        .collect();
    for (idx, value) in rho_grad_entries {
        grad[idx] = value;
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
    // derivative from `compute_kkt_residual_rho_corrections` so ARC sees one
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
    let kkt_rho_corrections = if let Some(r) = kkt_residual_vec
        .as_ref()
        .filter(|_| kkt_residual_correction_active && k > 0)
        .map(|r| r.as_ref())
    {
        Some(compute_kkt_residual_rho_corrections(
            solution,
            hop,
            &lambdas,
            &rho_penalty_a_k_betas,
            r,
            mode == EvalMode::ValueGradientHessian,
            &upper_active_rho,
        )?)
    } else {
        None
    };
    if let Some(corrections) = kkt_rho_corrections.as_ref() {
        let mut sl = grad.slice_mut(ndarray::s![..k]);
        sl += &corrections.gradient;
    }

    // Extended hyperparameter gradient (ψ/τ coordinates).
    //
    // Uses the SAME outer_gradient_entry() formula as ρ coordinates above.
    //
    // All extended coordinates store canonical fixed-β stationarity
    // derivatives g_i = F_{βi}. IFT gives β_i = -H^{-1}g_i, exactly like
    // the ρ block.
    // Per-call sink for the EIG-DECOMP diagnostic stash. Test builds populate
    // this on the rayon worker for `ext_idx == 0`; after the par_iter completes
    // the calling thread copies it into its own thread-local via
    // `debug_stash::store_terms`. Building the stash inside the worker and
    // handing it back through a per-call sink eliminates the cross-thread
    // overwrites that a process-global sink suffered under concurrent tests.
    let ext_stash_sink: std::sync::Arc<std::sync::Mutex<Option<debug_stash::TermStash>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    let ext_stash_sink_for_closure = ext_stash_sink.clone();
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
            // For canonical Gaussian (Identity link) the assembly skips
            // installing `penalty_subspace_trace` at all — `c ≡ 0` forces
            // `D_β H ≡ 0`, the classical Gaussian REML cost identity reads
            // smooth-floored `log|H|`, and `G_ε(H)` is the kernel matching
            // that cost surface within FD precision — see the `c_nontrivial`
            // gate in `build_dense_assembly` / `build_dense_original_assembly`.
            // Drops into the `None` arm below in that branch.
            // Diagnostic stash for the iso-κ Duchon FD investigation. Filled
            // only for the first extended coordinate (`ext_idx == 0`), only
            // when the log|H| contribution is included, and only while a test
            // holds a `debug_stash::CaptureGuard`. The capture is far from
            // free — it re-derives the drift, runs three extra spectral
            // traces (one of them the unprojected full-space trace) and a
            // second cubic IFT-correction pass — which at large scale
            // multiplied the dominant `ext_coord_trace` stage several-fold
            // per outer eval. Production gradient evals therefore skip it
            // entirely; the consuming FD tests opt in via the guard.
            let mut diag_stash: Option<debug_stash::TermStash> =
                if incl_logdet_h && ext_idx == 0 && debug_stash::capture_requested() {
                    Some(debug_stash::TermStash::default())
                } else {
                    None
                };

            let trace_logdet_i = if !incl_logdet_h {
                0.0
            } else if let Some(ref stoch_traces) = stochastic_trace_values {
                stoch_traces[k + ext_idx]
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

            // For the iso-κ Duchon FD investigation, capture the production
            // trace plus the unprojected full-space counterpart so the
            // `_pins_trace` test can verify that the projection kernel
            // changes the trace by an order of magnitude relative to
            // `G_ε(H)`. The drift for `ext_idx == 0` is recomputed here
            // regardless of which precomputed-trace branch fed
            // `trace_logdet_i`; this is cheap (a single coord) and lets the
            // diagnostic survive whether the gradient took the
            // `projected_trace_values` batched path or the per-coord
            // `else` fallback.
            if let Some(stash) = diag_stash.as_mut() {
                stash.projection_active = Some(solution.penalty_subspace_trace.is_some());
                stash.production_tr = Some(trace_logdet_i);
                // HVP ψ-gradient attribution (#740): expose the cost-derivative
                // `a` and penalty-logdet `ld_s` pieces of this coordinate's
                // outer gradient so a per-component FD of the outer value can be
                // matched against each analytic term independently.
                stash.coord_a = Some(coord.a);
                stash.coord_ld_s = Some(coord.ld_s);
                // Outer VALUE components, so a per-component FD of the objective
                // (β̂ re-solved at each ψ) can be matched against each analytic
                // gradient piece: FD(log_det_h) ↔ production_tr (probe b),
                // FD(cost − ½log_det_h + ½log_det_s) ↔ coord_a (probe a),
                // FD(log_det_s) ↔ coord_ld_s.
                stash.coord_log_det_h = Some(log_det_h);
                stash.coord_log_det_s = Some(log_det_s);
                stash.coord_cost = Some(cost);
                let correction = ext_corrections[ext_idx].as_ref();
                let drift = hyper_coord_total_drift_result(&coord.drift, correction, hop.dim());
                let unprojected = match &drift {
                    DriftDerivResult::Dense(matrix) => hop.trace_logdet_h_k(matrix, None),
                    DriftDerivResult::Operator(op) => hop.trace_logdet_operator(op.as_ref()),
                };
                stash.unprojected_tr = Some(unprojected);

                // #901-layer-2 split: the same kernel K traced against the
                // FROZEN drift B_i (no cubic correction) and against the
                // cubic correction D_βH[−v_i] alone, so the FD reference can
                // attribute the desync to one component.
                let frozen_only = hyper_coord_total_drift_result(&coord.drift, None, hop.dim());
                let trace_with_kernel = |d: &DriftDerivResult| -> f64 {
                    match (&solution.penalty_subspace_trace, d) {
                        (Some(kernel), DriftDerivResult::Dense(m)) => {
                            kernel.trace_projected_logdet(m)
                        }
                        (Some(kernel), DriftDerivResult::Operator(op)) => {
                            kernel.trace_operator(op.as_ref())
                        }
                        (None, DriftDerivResult::Dense(m)) => hop.trace_logdet_h_k(m, None),
                        (None, DriftDerivResult::Operator(op)) => {
                            hop.trace_logdet_operator(op.as_ref())
                        }
                    }
                };
                let frozen_tr = trace_with_kernel(&frozen_only);
                stash.frozen_tr = Some(frozen_tr);
                stash.correction_tr = Some(trace_logdet_i - frozen_tr);

                // #901-layer-2 candidate fix: recompute the cubic correction
                // with the IFT direction taken from the SAME pseudo-inverse the
                // cost uses (`v = H_pen⁺·coord.g`) instead of the full
                // `hop.solve`. The intrinsic kernel keeps the null(S₊)
                // curvature the old range(S₊) projection dropped AND floors the
                // truly-unidentified `ker(H_pen)` directions the full solve
                // over-amplifies. If this matches the FD cubic (≈ fd_total −
                // frozen_tr), the desync is the IFT direction using a different
                // inverse than the criterion.
                if let Some(kernel) = solution.penalty_subspace_trace.as_ref() {
                    let v_proj = kernel.apply_pseudo_inverse(&coord.g);
                    if let Ok(corr_proj) =
                        effective_deriv.hessian_derivative_correction_result(&v_proj)
                    {
                        let total_proj = hyper_coord_total_drift_result(
                            &coord.drift,
                            corr_proj.as_ref(),
                            hop.dim(),
                        );
                        let proj_total_tr = trace_with_kernel(&total_proj);
                        stash.correction_tr_proj = Some(proj_total_tr - frozen_tr);
                    }
                }
            }

            // Per-row diagnostic stash: captures term4's `c · X_τβ̂` diagonal
            // plus `X · v_ψ` per row, where v_ψ = ext_v_is[ext_idx] = hop⁻¹·coord.g.
            // The diagonal entering the correction sandwich is `c · X · v_ψ`;
            // multiplying by the test's known c recovers that diagonal.
            if let Some(mut stash) = diag_stash.take() {
                if let Some(op_arc) = coord.drift.operator.as_ref()
                    && let Some(sd) = op_arc.as_sparse_directional()
                {
                    stash.c_x_tau_beta_diag = sd.c_x_tau_beta.clone();
                    let v_i = &ext_v_is[ext_idx];
                    stash.c_x_v_psi_diag = Some(sd.x_design.matrixvectormultiply(v_i));
                }
                // Hand the stash back through the per-call sink. The
                // calling thread copies it into its own thread-local
                // after the par_iter completes, so concurrent tests
                // each end up reading their OWN ext_idx==0 capture
                // rather than racing on a shared global slot.
                *ext_stash_sink_for_closure
                    .lock()
                    .expect("EIG-DECOMP stash sink mutex poisoned") = Some(stash);
            }

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
            log::info!(
                "[STAGE] reml_laml ext_coord_trace ext_idx={} elapsed={:.3}s",
                ext_idx,
                ext_coord_start.elapsed().as_secs_f64(),
            );
            Ok((grad_idx, value))
        })
        .collect();
    for (idx, value) in ext_grad_entries? {
        grad[idx] = value;
    }

    // KKT-residual correction for the ψ (ext) gradient — the exact analogue of
    // the ρ correction applied above (`kkt_rho_corrections`), which was missing
    // for the extended coordinates.
    //
    // When the inner solve exits at β̂ with a nonzero KKT residual
    // `r = ∇_β L_pen(β̂)`, the cost carries the one-step Newton profile
    // correction `−½ rᵀ H⁻¹ r` (applied above), so the CORRECTED scalar
    // objective is `Ṽ(θ) = V(β̂,θ) − ½ rᵀ H⁻¹ r`. Differentiating the
    // correction w.r.t. a ψ coordinate (holding β̂ fixed, the same convention
    // the ρ block uses) with `q = H⁻¹ r`:
    //
    //   ∂_ψ r = ∂_ψ(∇_β L_pen)|_β = coord.g   (= score_psi + S_ψ β̂)
    //   ∂_ψ H = Ḣ_ψ|_β            = the FROZEN ψ Hessian drift B_i
    //   ∂_ψ(−½ rᵀ H⁻¹ r) = −(∂_ψ r)ᵀ q + ½ qᵀ (∂_ψ H) q
    //                     = −coord.gᵀ q + ½ qᵀ B_i q.
    //
    // This is `compute_kkt_residual_rho_corrections`'s `C_i = −a_iᵀq + ½ qᵀA_iq`
    // with the ρ ingredients `(a_i = A_iβ̂, A_i = λ_iS_i)` replaced by the ψ
    // ingredients `(coord.g, B_i)`. It vanishes identically at exact KKT
    // (`r = 0 ⇒ q = 0`); when the logslope block is near-singular and the inner
    // exit accepts `‖r‖ > 0`, the dropped `−coord.gᵀq` term is amplified by
    // `‖H⁻¹‖·‖r‖` and is exactly the large component the envelope ψ-gradient
    // was missing relative to the centered FD of the corrected objective. The
    // FROZEN drift (no IFT correction) is used — the IFT/β-response of the
    // logdet trace is handled separately in the trace term; the residual
    // correction differentiates `−½rᵀH⁻¹r` at fixed β̂ exactly as the ρ block
    // does.
    if ext_dim > 0
        && let Some(r) = kkt_residual_vec
            .as_ref()
            .filter(|_| kkt_residual_correction_active)
            .map(|r| r.as_ref())
    {
        let subspace = solution.penalty_subspace_trace.as_deref();
        let q = solve_kkt_residual_kernel(hop, subspace, r);
        for ext_idx in 0..ext_dim {
            let coord = &solution.ext_coords[ext_idx];
            // FROZEN ψ Hessian drift B_i (no IFT correction); `apply` matvecs the
            // dense or operator form against `q`.
            let frozen_drift = hyper_coord_total_drift_result(&coord.drift, None, hop.dim());
            let b_i_q = frozen_drift.apply(&q);
            let linear = coord.g.dot(&q);
            let quadratic = q.dot(&b_i_q);
            if !linear.is_finite() || !quadratic.is_finite() {
                return Err(RemlError::NonFiniteValue {
                    reason: format!(
                        "KKT ext correction produced non-finite ingredients at ext coord \
                         {ext_idx}: linear={linear} quadratic={quadratic}"
                    ),
                }
                .into());
            }
            grad[k + ext_idx] += -linear + 0.5 * quadratic;
        }
    }

    // Drain the per-call EIG-DECOMP sink into the calling thread's
    // thread-local stash. The rayon worker that produced the stash may
    // live on a different thread, but `store_terms` writes to the
    // current thread's TLS — which is also the thread the test will
    // call `take_terms()` from, because everything between
    // `evaluate_joint_reml_outer_eval_at_theta` and `reml_laml_evaluate`
    // is synchronous and stays on the test thread.
    if let Some(stash) = ext_stash_sink
        .lock()
        .expect("EIG-DECOMP stash sink mutex poisoned")
        .take()
    {
        debug_stash::store_terms(stash);
    }

    // Apply the ρ-GRADIENT half of the guarded Tierney–Kadane correction built
    // above (its VALUE half was applied to `cost` before the value-only early
    // return). Both halves read the SAME `include` guard from the SAME
    // `tk_correction` object, so the value and derivative cannot desync — a
    // `MaxPenalizedLikelihood`-style assembly (`include_logdet_h = false`)
    // carrying a nonzero `tk_correction` omits BOTH, never one without the
    // other. (The runtime `apply_tk_to_result` path is independently paired —
    // it always applies value+gradient+hessian together.)
    tk_correction.apply_gradient(&mut grad);

    // Add prior gradient (ρ-only).
    if let Some((_, ref pg, _)) = prior_cost_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += pg;
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
    // *post-correction* gradient (the `kkt_rho_corrections.gradient`
    // additive block above has already been folded into `grad`). If the
    // predicted √ε-step cost change still exceeds 4·|cost| after that
    // fold, the gradient is invalid as a descent direction. A previous
    // exception kept the gradient when `kkt_residual_correction_active`
    // was true, but that was self-contradictory: the tripwire fires on
    // the same gradient the correction was supposed to have repaired,
    // and when `‖r_proj‖∞ ≈ 0` (the cert-exit contract) the correction
    // `-aᵀ_k q + ½ qᵀA_k q` with `q = H⁻¹·r ≡ 0` is identically zero
    // (see `compute_kkt_residual_rho_corrections`). Active-constraint
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
            log::info!(
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
            let mut hessian = crate::solver::outer_strategy::HessianResult::Operator(family_op);
            if let Some(kkt_hessian) = kkt_rho_corrections
                .as_ref()
                .and_then(|corrections| corrections.hessian.as_ref())
            {
                hessian.add_rho_block_dense(kkt_hessian)?;
            }
            if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                hessian.add_rho_block_dense(ph)?;
            }
            log::info!(
                "[OUTER hessian-elapsed] choice=operator reason=family_op \
                 n={n_obs} p={p_dim} k={k_outer} elapsed={:.3}s",
                assembly_start.elapsed().as_secs_f64(),
            );
            return Ok(RemlLamlResult {
                cost,
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
        // The (n, p, K) scale rule routes large-scale problems through the
        // matrix-free Hv operator path even when the per-axis thresholds
        // (`p >= 512` or `K >= 32`) alone do not fire.  At Matern large-scale
        // scale (n=320 000, p=101, K=6) the dense path's per-outer-eval
        // O(K·n·p²) assembly is ≈ 2·10¹⁰ FLOPs and dominates wall-clock; the
        // operator path absorbs it via O(n·p) HVPs.
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
            n_obs,
            p_dim,
            k_outer,
            hessian_kernel.is_some(),
            callback_operator_kernel,
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
        log::info!(
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
                    let mut hessian =
                        crate::solver::outer_strategy::HessianResult::Operator(Arc::new(op));
                    if let Some(kkt_hessian) = kkt_rho_corrections
                        .as_ref()
                        .and_then(|corrections| corrections.hessian.as_ref())
                    {
                        hessian.add_rho_block_dense(kkt_hessian)?;
                    }
                    if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                        hessian.add_rho_block_dense(ph)?;
                    }
                    hessian
                }
                Err(err) if is_hessian_unavailable(&err) => {
                    log::warn!("{err}");
                    crate::solver::outer_strategy::HessianResult::Unavailable
                }
                Err(err) => return Err(err),
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
                    if let Some(kkt_hessian) = kkt_rho_corrections
                        .as_ref()
                        .and_then(|corrections| corrections.hessian.as_ref())
                    {
                        let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                        sl += kkt_hessian;
                    }
                    // Add prior Hessian (second derivatives of the soft prior on ρ, ρ-only).
                    if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                        let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                        sl += ph;
                    }
                    crate::solver::outer_strategy::HessianResult::Analytic(h)
                }
                Err(err) if is_hessian_unavailable(&err) => {
                    log::warn!("{err}");
                    crate::solver::outer_strategy::HessianResult::Unavailable
                }
                Err(err) => return Err(err),
            }
        };
        log::info!(
            "[OUTER hessian-elapsed] choice={route_choice} reason={route_reason} \
             n={n_obs} p={p_dim} k={k_outer} elapsed={:.3}s",
            assembly_start.elapsed().as_secs_f64(),
        );
        result
    } else {
        crate::solver::outer_strategy::HessianResult::Unavailable
    };

    // Envelope-gradient sanity tripwire — last line of defense.
    //
    // The post-IFT-correction gradient is what `envelope_inconsistent` is
    // computed on (the `kkt_rho_corrections.gradient` block was folded in
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
            log::warn!(
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
                 (when true the cost IFT uses bilinear_pseudo_inverse on range(S₊); when false \
                 the full H⁻¹·r solve is the only path and is numerically unreliable on \
                 near-singular H). \
                 If kkt_residual.is_some()=false under fixed dispersion, the convergent inner \
                 path forgot to populate `BlockwiseInnerResult::kkt_residual` (call \
                 `exact_newton_joint_kkt_residual_for_ift(..., Some(active_sets))` on return) \
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
        ift_residual_energy,
        inner_polish_step,
        gradient: gradient_out,
        hessian,
        rho_mode_response_cols,
        ext_mode_response_cols,
    })
}
