
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
            let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw);
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
        // Serial-when-large is deliberate, NOT a throughput bug: each
        // `hessian_derivative_correction_result` is an `Xᵀ·diag(c⊙Xvₖ)·X`
        // crossproduct that already saturates every core via faer's global
        // parallelism (`streaming_blas_xt_diag_x` → `get_global_parallelism`).
        // Wrapping the outer per-coordinate map in `par_iter` for large work
        // would nest a rayon fan-out around already-parallel BLAS-3 kernels and
        // oversubscribe the thread pool, regressing wall-clock. The small-work
        // branch goes parallel only because each correction is too thin to fill
        // the pool on its own, so the outer fan-out is free there.
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


const HESSIAN_UNAVAILABLE_PREFIX: &str = "outer Hessian unavailable:";


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
const STOCHASTIC_TRACE_DIM_THRESHOLD: usize = 500;


/// Elapsed-time (ms) above which a sparse-Cholesky trace path emits a timing
/// diagnostic. Purely observational — surfaces slow per-eval trace solves to the
/// bench runner without affecting the fit.
const REML_TRACE_SLOW_LOG_MS: f64 = 100.0;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OuterHessianRoutePlan {
    pub(crate) use_operator: bool,
    reason: &'static str,
    scale_prefers_operator: bool,
    dense_workspace_bytes: usize,
}


impl OuterHessianRoutePlan {
    fn choice(self) -> &'static str {
        if self.use_operator {
            "operator"
        } else {
            "dense"
        }
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OuterHessianScaleDecision {
    prefers_operator: bool,
    reason: &'static str,
}


fn saturating_f64_matrix_bytes(rows: usize, cols: usize) -> usize {
    rows.saturating_mul(cols)
        .saturating_mul(std::mem::size_of::<f64>())
}


fn outer_hessian_dense_workspace_bytes(p: usize, k: usize) -> usize {
    // Dense assembly keeps first-order drifts for each coordinate and uses at
    // least one transient second-order drift while filling the K x K Hessian.
    // Charge a small safety multiple so the route never depends on fitting a
    // single p x p matrix while the actual dense path holds several.
    let drift_count = k.saturating_mul(2).saturating_add(3).max(1);
    saturating_f64_matrix_bytes(p, p).saturating_mul(drift_count)
}


fn outer_hessian_dense_workspace_budget_bytes() -> usize {
    crate::resource::ResourcePolicy::default_library().max_single_materialization_bytes
}


fn dense_outer_hessian_workspace_fits(p: usize, k: usize) -> bool {
    outer_hessian_dense_workspace_bytes(p, k) <= outer_hessian_dense_workspace_budget_bytes()
}


fn generic_outer_hessian_scale_decision(n: usize, p: usize, k: usize) -> OuterHessianScaleDecision {
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


fn callback_outer_hessian_scale_decision(
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


fn is_hessian_unavailable(error: &str) -> bool {
    error.starts_with(HESSIAN_UNAVAILABLE_PREFIX)
}


//   C[u]            = Xᵀ diag(c ⊙ Xu) X
//   h^G             = diag(X G_ε(H) Xᵀ)
//   v               = Xᵀ (c ⊙ h^G)
//   z_c             = H⁻¹ v
//   tr(G_ε C[u])    = uᵀ Xᵀ (c ⊙ h^G) = uᵀ v
fn compute_adjoint_z_c(
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
fn compute_fourth_derivative_trace(
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
fn compute_fourth_derivative_trace_matrix(
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
fn compute_ift_correction_trace(
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
fn compute_drift_deriv_traces(
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
fn compute_base_h2_trace(
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


fn compute_base_h2_traces(
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


fn trace_logdet_hessian_cross_dense_drift(
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


fn trace_logdet_hessian_crosses_dense_spectral_drifts(
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
fn can_use_stochastic_logdet_hinv_kernel(
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


struct KktRhoCorrections {
    gradient: Array1<f64>,
    hessian: Option<Array2<f64>>,
}


fn solve_kkt_residual_kernel(
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


fn active_upper_rho_mask(rho: &[f64]) -> Vec<bool> {
    let latest_theta = super::runtime::latest_outer_theta_for_ift();
    let matching_outer_theta = latest_theta.as_ref().is_some_and(|theta| {
        theta.len() >= rho.len()
            && theta
                .iter()
                .take(rho.len())
                .zip(rho.iter())
                .all(|(&recorded, &current)| recorded.to_bits() == current.to_bits())
    });
    let upper_bounds = matching_outer_theta
        .then(super::runtime::latest_outer_rho_upper_bounds_for_ift)
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
fn compute_kkt_residual_rho_corrections(
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
fn compute_outer_hessian(
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


struct StoredFirstDrift {
    dense: Option<Array2<f64>>,
    dense_rotated: Option<Array2<f64>>,
    operators: Vec<Arc<dyn HyperOperator>>,
}


impl StoredFirstDrift {
    fn from_parts(
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

    fn scaled_add_apply(&self, v: ArrayView1<'_, f64>, scale: f64, out: &mut Array1<f64>) {
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

    fn apply_dot(&self, v: ArrayView1<'_, f64>, test: ArrayView1<'_, f64>) -> f64 {
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


struct BorrowedStoredDriftOperator<'a> {
    drift: &'a StoredFirstDrift,
    dim_hint: usize,
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

    fn scaled_add_mul_vec(&self, v: ArrayView1<'_, f64>, scale: f64, out: ArrayViewMut1<'_, f64>) {
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
struct PsiContractedContrib {
    objective: Array1<f64>,
    score: Array2<f64>,
    ld_s: Array1<f64>,
    base_h2: Vec<f64>,
}


struct OuterHessianCoord {
    a: f64,
    g: Array1<f64>,
    v: Array1<f64>,
    total_drift: StoredFirstDrift,
    base_drift: StoredFirstDrift,
    ext_index: Option<usize>,
    b_depends_on_beta: bool,
}


impl OuterHessianCoord {
    fn is_ext(&self) -> bool {
        self.ext_index.is_some()
    }
}


struct UnifiedOuterHessianOperator {
    hop: Arc<dyn HessianOperator>,
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
    kernel: OuterHessianDerivativeKernel,
    subspace: Option<Arc<PenaltySubspaceTrace>>,
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

    fn pair_rhs_dot(&self, row: usize, col: usize, test: ArrayView1<'_, f64>) -> f64 {
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

    fn scaled_add_pair_rhs(&self, row: usize, col: usize, scale: f64, out: &mut Array1<f64>) {
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

    fn pair_rhs_combo(&self, idx: usize, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for j in 0..alpha.len() {
            if alpha[j] != 0.0 {
                self.scaled_add_pair_rhs(idx, j, alpha[j], &mut out);
            }
        }
        out
    }

    fn scalar_correction_trace(
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

    fn callback_correction_trace(
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
    fn psi_contracted_contrib(
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
    fn outer_hessian_index_entry(
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


fn build_outer_hessian_operator(
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
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum absolute step size in log-λ for the EFS update (prevents
/// overshooting). Each iteration changes `λ` by at most `exp(EFS_MAX_STEP)`.
const EFS_MAX_STEP: f64 = 5.0;


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
/// - `solution`: Converged inner state (β̂, H, penalties, HessianOperator).
/// - `rho`: Current log-smoothing parameters.
/// - `gradient`: Full outer gradient `∂V_total/∂θ`, length
///   `n_rho + n_ext`. The caller must run
///   [`super::reml::unified::EvalMode::ValueAndGradient`] when
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

    // Universal-form EFS: `Δρ_i = log(1 − 2·g_full[i]/q_eff_i)`. This is
    // identical to the canonical `log((d−t)/q_eff)` when no out-of-band
    // cost terms exist (TK, prior, Firth, barrier, SAS ridge), and shifts
    // the multiplicative target by exactly the residual gradient when
    // they do. We get the augmented stationarity for free, in exchange
    // for one `EvalMode::ValueAndGradient` evaluation per outer
    // iteration.
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let lambda = rho[idx].exp();
        let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
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


/// Diagnostics for the bam-style single-loop EFS route.
///
/// `gradient_residual` is the normalized difference between the full
/// REML/LAML gradient available at this evaluation and the gradient implied
/// by the EFS multiplicative surrogate after step clipping/pathological
/// denominators. `inner_residual` is the PIRLS relative gradient residual
/// from the deliberately partial inner sweep. The guard uses
/// `bias_proxy = max(gradient_residual, inner_residual)` as a drift detector;
/// it does not feed back into the EFS step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EfsSingleLoopDiagnostics {
    pub(crate) bias_proxy: f64,
    pub(crate) gradient_residual: f64,
    pub(crate) inner_residual: f64,
    pub(crate) gradient_norm: f64,
    pub(crate) step_inf_norm: f64,
}


/// Estimate how far the single-loop EFS surrogate has drifted from the
/// current outer-gradient geometry.
pub(crate) fn efs_single_loop_diagnostics(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    gradient: &[f64],
    steps: &[f64],
    inner_residual: f64,
) -> EfsSingleLoopDiagnostics {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    assert_eq!(gradient.len(), total);
    assert_eq!(steps.len(), total);

    let (profiled_scale, dp_cgrad) = efs_profiling(solution);
    let mut grad_sq = 0.0_f64;
    let mut residual_sq = 0.0_f64;
    let mut step_inf_norm = 0.0_f64;

    for idx in 0..k {
        let g = gradient[idx];
        grad_sq += g * g;
        step_inf_norm = step_inf_norm.max(steps[idx].abs());
        let coord = &solution.penalty_coords[idx];
        let lambda = rho[idx].exp();
        let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
        let q_eff = efs_q_eff_with_gamma_rate(
            efs_q_eff(a_i, &solution.dispersion, dp_cgrad, profiled_scale),
            lambda,
            &solution.rho_prior,
            idx,
        );
        if q_eff.is_finite() && q_eff > 0.0 && steps[idx].is_finite() {
            let g_efs = 0.5 * q_eff * (1.0 - steps[idx].exp());
            let d = g - g_efs;
            residual_sq += d * d;
        } else {
            residual_sq += g * g;
        }
    }

    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let idx = k + ext_idx;
        let g = gradient[idx];
        grad_sq += g * g;
        step_inf_norm = step_inf_norm.max(steps[idx].abs());
        if !coord.is_penalty_like {
            continue;
        }
        let q_eff = efs_q_eff(coord.a, &solution.dispersion, dp_cgrad, profiled_scale);
        if q_eff.is_finite() && q_eff > 0.0 && steps[idx].is_finite() {
            let g_efs = 0.5 * q_eff * (1.0 - steps[idx].exp());
            let d = g - g_efs;
            residual_sq += d * d;
        } else {
            residual_sq += g * g;
        }
    }

    let gradient_norm = grad_sq.sqrt();
    let gradient_residual = residual_sq.sqrt() / (1.0 + gradient_norm);
    let inner_residual = if inner_residual.is_finite() && inner_residual >= 0.0 {
        inner_residual
    } else {
        f64::INFINITY
    };
    let bias_proxy = gradient_residual.max(inner_residual);

    EfsSingleLoopDiagnostics {
        bias_proxy,
        gradient_residual,
        inner_residual,
        gradient_norm,
        step_inf_norm,
    }
}


/// Regularization threshold for pseudoinverse of the trace Gram matrix.
///
/// Eigenvalues below `PSI_GRAM_PINV_TOL * max_eigenvalue` are treated as
/// zero when computing the pseudoinverse G⁺. This prevents amplification
/// of noise in near-singular directions of the ψ-ψ Gram matrix.
const PSI_GRAM_PINV_TOL: f64 = 1e-8;


/// Initial step-size damping factor for the preconditioned gradient on ψ.
///
/// The raw step `Δψ_raw = -G⁺ g_ψ` is scaled by α ∈ (0, 1] before
/// applying. This conservative initial value prevents overshooting in
/// early iterations when the quadratic model may be inaccurate.
const PSI_INITIAL_ALPHA: f64 = 1.0;


/// Minimum number of scalar ρ/τ EFS candidates before `compute_hybrid_efs_update`
/// fans out with rayon.  Smaller blocks are common (1-4 smoothing parameters),
/// where task scheduling costs dominate the independent arithmetic.
const HYBRID_EFS_SCALAR_PAR_THRESHOLD: usize = 8;


/// Minimum number of independent ψ-ψ Gram entries before exact trace assembly
/// fans out with rayon.  This is expressed in upper-triangle pair count rather
/// than `n_psi` so 5 ψ coordinates (15 pairs) stay serial while moderate
/// anisotropic/design-moving blocks parallelize.
const HYBRID_EFS_GRAM_PAIR_PAR_THRESHOLD: usize = 24;


/// Minimum number of ψ drifts before materialization/projection is done in
/// parallel during exact Gram assembly.
const HYBRID_EFS_PSI_DRIFT_PAR_THRESHOLD: usize = 8;


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
/// - `solution`: Converged inner state (β̂, H, penalties, HessianOperator).
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
    let rho_candidates: Vec<(usize, Option<f64>)> =
        if k >= HYBRID_EFS_SCALAR_PAR_THRESHOLD && rayon::current_thread_index().is_none() {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..k)
                .into_par_iter()
                .map(|idx| {
                    let coord = &solution.penalty_coords[idx];
                    let lambda = rho[idx].exp();
                    let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
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
                    let coord = &solution.penalty_coords[idx];
                    let lambda = rho[idx].exp();
                    let a_i = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambda);
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
                .filter_map(|op| op.as_implicit())
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
fn pseudoinverse_times_vec(
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
fn symmetric_eigen(a: &ndarray::Array2<f64>) -> (Vec<f64>, ndarray::Array2<f64>) {
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


// ═══════════════════════════════════════════════════════════════════════════
//  Corrected coefficient covariance (smoothing-parameter uncertainty)
// ═══════════════════════════════════════════════════════════════════════════

/// Diagnostic returned when the (free-subspace) outer Hessian is indefinite.
///
/// An indefinite outer Hessian at the reported optimum means one of:
///  - the outer optimization has not converged to a stationary point,
///  - the reported point is a saddle, not a minimum,
///  - the active-bound set on θ is wrong (the unconstrained Hessian is
///    being inspected on directions that the constraint set actually pins),
///  - the objective being inspected is a surrogate / smoothed version of
///    the true REML/LAML criterion.
///
/// The previous implementation silently clamped the negative eigenvalues to
/// zero, which under-reports the uncertainty in those directions (it pretends
/// the directions don't exist). That is not "conservative" — it is wrong.
/// We refuse to fabricate a covariance and instead return this diagnostic.
#[derive(Debug, Clone)]
pub struct OuterHessianIndefinite {
    /// Most-negative eigenvalue of the projected (free-subspace) outer Hessian.
    pub min_eigenvalue: f64,
    /// Indices of θ-coordinates that were detected as active on a bound.
    pub active_constraints: Vec<usize>,
    /// θ at the reported optimum (if available; empty otherwise).
    pub theta: Vec<f64>,
    /// L2 norm of the outer gradient at θ (NaN if unavailable).
    pub gradient_norm: f64,
    /// Frobenius norm of the outer Hessian.
    pub hessian_norm: f64,
    /// Suggested next action for the caller / user.
    pub suggested_action: &'static str,
}


/// Errors that can arise while building the corrected covariance.
#[derive(Debug, Clone)]
pub enum CorrectedCovarianceError {
    /// Argument shapes do not agree at a named corrected-covariance stage.
    ShapeMismatch {
        context: &'static str,
        reason: String,
    },
    /// The eigendecomposition failed numerically at a named matrix stage.
    EigendecompositionFailed {
        context: &'static str,
        reason: String,
    },
    /// The projected outer Hessian is indefinite — see diagnostic.
    Indefinite(OuterHessianIndefinite),
}


impl core::fmt::Display for CorrectedCovarianceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch { context, reason } => {
                write!(f, "shape mismatch in {context}: {reason}")
            }
            Self::EigendecompositionFailed { context, reason } => {
                write!(f, "eigendecomposition failed in {context}: {reason}")
            }
            Self::Indefinite(d) => write!(
                f,
                "outer Hessian indefinite on free subspace (min eigenvalue = {:.3e}, \
                 ||H||_F = {:.3e}, ||g||_2 = {:.3e}, active = {:?}, theta = {:?}); {}",
                d.min_eigenvalue,
                d.hessian_norm,
                d.gradient_norm,
                d.active_constraints,
                d.theta,
                d.suggested_action,
            ),
        }
    }
}


impl std::error::Error for CorrectedCovarianceError {}


/// Result describing the corrected covariance plus structural diagnostics.
#[derive(Debug, Clone)]
pub struct CorrectedCovariance {
    /// The p×p corrected covariance V*_α.
    pub matrix: Array2<f64>,
    /// θ-indices that were treated as active on a bound and excluded from V_θ.
    pub active_constraints: Vec<usize>,
    /// θ-indices in the free subspace whose curvature was so close to zero
    /// that they were treated as structurally rank-deficient (pseudoinverse).
    pub rank_deficient_directions: Vec<usize>,
}


impl CorrectedCovariance {
    fn has_structural_diagnostics(&self) -> bool {
        !self.active_constraints.is_empty() || !self.rank_deficient_directions.is_empty()
    }
}


/// Suggested action text returned with `OuterHessianIndefinite`.
const INDEFINITE_SUGGESTED_ACTION: &str = "refit with a tighter outer tolerance, verify the inspected objective is the true \
     REML/LAML cost rather than a surrogate, and audit recent active-set transitions";


/// Detect θ-coordinates that are sitting on the [-RHO_BOUND, RHO_BOUND] bound.
///
/// We use the same `tolerance = 1e-8` as the rest of the outer code path so the
/// active-set view here agrees with the optimizer's view at the reported optimum.
fn detect_active_theta_bounds(theta: Option<&[f64]>, q: usize) -> Vec<usize> {
    let Some(theta) = theta else {
        return Vec::new();
    };
    if theta.len() != q {
        return Vec::new();
    }
    let bound = crate::solver::estimate::RHO_BOUND;
    // Same active-bound tolerance the outer optimizer uses, so this active-set
    // view agrees with the optimizer's at the reported optimum.
    const ACTIVE_THETA_BOUND_TOL: f64 = 1e-8;
    theta
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v.abs() >= bound - ACTIVE_THETA_BOUND_TOL).then_some(i))
        .collect()
}


/// Decide which θ-coordinates are bounded (ρ-style) vs unbounded (ψ-style).
///
/// We treat the LAST `ext_len` coordinates as ψ (unbounded extended
/// hyperparameters) and the FIRST `rho_len` as ρ (bounded by ±RHO_BOUND).
/// This matches the layout used everywhere else in this file: J_α has ρ
/// columns first, then ext columns.
fn active_bound_indices_for_theta(
    theta: Option<&[f64]>,
    rho_len: usize,
    ext_len: usize,
) -> Vec<usize> {
    let q = rho_len + ext_len;
    let mut active = detect_active_theta_bounds(theta, q);
    // Drop ψ-coordinates: they are unbounded by construction.
    active.retain(|&i| i < rho_len);
    active
}


/// Inertia gate + projected-inverse on the free subspace of θ.
///
/// Returns `(V_θ_full, rank_deficient_free_indices)` where `V_θ_full` is q×q
/// with rows/columns of active coordinates set to zero. If the projected
/// Hessian is indefinite beyond tolerance, returns the diagnostic instead.
fn projected_inverse_with_inertia_gate(
    outer_hessian: &Array2<f64>,
    active: &[usize],
    theta_for_diag: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<(Array2<f64>, Vec<usize>), CorrectedCovarianceError> {
    let q = outer_hessian.nrows();
    let mut is_active = vec![false; q];
    for &i in active {
        if i < q {
            is_active[i] = true;
        }
    }
    let free: Vec<usize> = (0..q).filter(|i| !is_active[*i]).collect();
    let qf = free.len();

    let h_norm = outer_hessian.iter().map(|v| v * v).sum::<f64>().sqrt();

    let mut v_theta_full = Array2::<f64>::zeros((q, q));
    if qf == 0 {
        return Ok((v_theta_full, Vec::new()));
    }

    let mut h_ff = Array2::<f64>::zeros((qf, qf));
    for (a, &ia) in free.iter().enumerate() {
        for (b, &ib) in free.iter().enumerate() {
            h_ff[[a, b]] = outer_hessian[[ia, ib]];
        }
    }

    let (evals, evecs) = h_ff.eigh(faer::Side::Lower).map_err(|e| {
        CorrectedCovarianceError::EigendecompositionFailed {
            context: "projected outer Hessian",
            reason: e.to_string(),
        }
    })?;

    let eps = f64::EPSILON;
    let neg_tol = 8.0 * eps * (q.max(1) as f64) * h_norm.max(1.0);
    let min_eig = evals.iter().copied().fold(f64::INFINITY, f64::min);
    if min_eig < -neg_tol {
        let diagnostic = OuterHessianIndefinite {
            min_eigenvalue: min_eig,
            active_constraints: active.to_vec(),
            theta: theta_for_diag.map(|t| t.to_vec()).unwrap_or_default(),
            gradient_norm,
            hessian_norm: h_norm,
            suggested_action: INDEFINITE_SUGGESTED_ACTION,
        };
        return Err(CorrectedCovarianceError::Indefinite(diagnostic));
    }

    let pos_tol = 8.0 * eps * (q.max(1) as f64) * h_norm.max(1.0);
    let mut v_theta_ff = Array2::<f64>::zeros((qf, qf));
    let mut rank_deficient_free: Vec<usize> = Vec::new();
    for j in 0..qf {
        let sigma = evals[j];
        if sigma.abs() <= pos_tol {
            rank_deficient_free.push(j);
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let u = evecs.column(j);
        for a in 0..qf {
            let ua = inv_sigma * u[a];
            for b in a..qf {
                let val = ua * u[b];
                v_theta_ff[[a, b]] += val;
                if a != b {
                    v_theta_ff[[b, a]] += val;
                }
            }
        }
    }

    for (a, &ia) in free.iter().enumerate() {
        for (b, &ib) in free.iter().enumerate() {
            v_theta_full[[ia, ib]] = v_theta_ff[[a, b]];
        }
    }

    let rank_deficient_directions: Vec<usize> =
        rank_deficient_free.into_iter().map(|j| free[j]).collect();

    Ok((v_theta_full, rank_deficient_directions))
}


/// Corrected covariance of the coefficient vector, accounting for uncertainty
/// in the smoothing/hyperparameters θ = (ρ, ψ).
///
/// The standard conditional covariance H^{-1} ignores uncertainty in θ.
/// The corrected covariance adds the propagation term:
///
/// ```text
///   V*_α = H^{-1} + J_α V_θ J_α^T
/// ```
///
/// where:
/// - H^{-1} is obtained via `hop.solve` on identity columns,
/// - J_α = [-v_1, …, -v_k, -ext_v_1, …, -ext_v_m] is the p×q matrix of
///   negated mode responses (implicit-function sensitivities ∂β̂/∂θ),
/// - V_θ is the inverse of the outer Hessian RESTRICTED to the free subspace
///   of θ (coordinates that are not pinned to a bound) and inertia-gated.
///
/// # Active-bound handling and inertia gate
///
/// If `theta_at_optimum` is supplied, ρ-coordinates sitting on ±`RHO_BOUND`
/// are treated as active and excluded from V_θ. The remaining free Hessian
/// block H_FF is eigen-decomposed:
///   - if min(σ) < -8·ε·q·‖H‖_F → return [`CorrectedCovarianceError::Indefinite`]
///     with a diagnostic (the previous behavior of clamping negatives to zero
///     under-reports uncertainty and is therefore refused);
///   - if |σ| ≤ 8·ε·q·‖H‖_F → that direction is treated as structurally
///     rank-deficient (Moore-Penrose drop) and listed in
///     `rank_deficient_directions` for the caller to surface;
///   - otherwise H_FF is inverted exactly via the spectral expansion.
pub fn compute_corrected_covariance(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, CorrectedCovarianceError> {
    compute_corrected_covariance_with_constraints(v_ks, ext_v, outer_hessian, hop, None, f64::NAN)
        .map(|cov| {
            if cov.has_structural_diagnostics() {
                log::debug!(
                    "corrected covariance diagnostics: active_constraints={:?} rank_deficient_directions={:?}",
                    cov.active_constraints,
                    cov.rank_deficient_directions
                );
            }
            cov.matrix
        })
}


/// Constraint- and inertia-aware version of [`compute_corrected_covariance`].
///
/// Prefer this entry point when θ at the optimum and the outer-gradient norm
/// are available — it auto-derives the active-bound set on ρ and emits the
/// rank-deficient diagnostic alongside the matrix.
pub fn compute_corrected_covariance_with_constraints(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
    theta_at_optimum: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<CorrectedCovariance, CorrectedCovarianceError> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    if q == 0 {
        let eye = Array2::eye(p);
        return Ok(CorrectedCovariance {
            matrix: hop.solve_multi(&eye),
            active_constraints: Vec::new(),
            rank_deficient_directions: Vec::new(),
        });
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(CorrectedCovarianceError::ShapeMismatch {
            context: "compute_corrected_covariance outer Hessian",
            reason: format!(
                "dimension ({}, {}) does not match total hyperparameter count q = {} (rho: {}, ext: {})",
                outer_hessian.nrows(),
                outer_hessian.ncols(),
                q,
                v_ks.len(),
                ext_v.len(),
            ),
        });
    }

    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    let active = active_bound_indices_for_theta(theta_at_optimum, v_ks.len(), ext_v.len());

    let (v_theta, rank_deficient_directions) = projected_inverse_with_inertia_gate(
        outer_hessian,
        &active,
        theta_at_optimum,
        gradient_norm,
    )?;

    let j_v_theta = j_alpha.dot(&v_theta);
    let correction = j_v_theta.dot(&j_alpha.t());

    let eye = Array2::eye(p);
    let mut h_inv = hop.solve_multi(&eye);
    h_inv += &correction;

    enforce_symmetry_inplace(&mut h_inv);

    Ok(CorrectedCovariance {
        matrix: h_inv,
        active_constraints: active,
        rank_deficient_directions,
    })
}


/// Compute only the diagonal of the corrected covariance V*_alpha.
///
/// This is much cheaper than the full p x p matrix: O(p q) instead of O(p^2 q).
///
/// ```text
///   diag(V*_alpha) = diag(H^{-1}) + row_norms(J_alpha L_theta)^2
/// ```
///
/// where L_theta is the Cholesky-like square root of V_theta. When V_theta
/// is obtained via positive-projected eigendecomposition, L_theta = U sqrt(D+)
/// where D+ contains the positive-part eigenvalues.
///
/// # Arguments
/// - `v_ks`: mode responses for rho coordinates
/// - `ext_v`: mode responses for extended (psi) coordinates
/// - `outer_hessian`: the q x q outer Hessian
/// - `hop`: the HessianOperator providing H^{-1}
///
/// # Returns
/// A p-vector of corrected marginal variances.
pub fn compute_corrected_covariance_diagonal(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array1<f64>, CorrectedCovarianceError> {
    compute_corrected_covariance_diagonal_with_constraints(
        v_ks,
        ext_v,
        outer_hessian,
        hop,
        None,
        f64::NAN,
    )
    .map(|d| {
        if d.has_structural_diagnostics() {
            log::debug!(
                "corrected covariance diagonal diagnostics: active_constraints={:?} rank_deficient_directions={:?}",
                d.active_constraints,
                d.rank_deficient_directions
            );
        }
        d.diagonal
    })
}


/// Diagonal of the corrected covariance plus active-set / rank-deficiency
/// diagnostics. See [`compute_corrected_covariance_with_constraints`] for the
/// full version (the inertia gate logic is identical).
#[derive(Debug, Clone)]
pub struct CorrectedCovarianceDiagonal {
    pub diagonal: Array1<f64>,
    pub active_constraints: Vec<usize>,
    pub rank_deficient_directions: Vec<usize>,
}


impl CorrectedCovarianceDiagonal {
    fn has_structural_diagnostics(&self) -> bool {
        !self.active_constraints.is_empty() || !self.rank_deficient_directions.is_empty()
    }
}


pub fn compute_corrected_covariance_diagonal_with_constraints(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
    theta_at_optimum: Option<&[f64]>,
    gradient_norm: f64,
) -> Result<CorrectedCovarianceDiagonal, CorrectedCovarianceError> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    let mut diag = Array1::zeros(p);
    for i in 0..p {
        let mut e_i = Array1::zeros(p);
        e_i[i] = 1.0;
        let h_inv_ei = hop.solve(&e_i);
        diag[i] = h_inv_ei[i];
    }

    if q == 0 {
        return Ok(CorrectedCovarianceDiagonal {
            diagonal: diag,
            active_constraints: Vec::new(),
            rank_deficient_directions: Vec::new(),
        });
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(CorrectedCovarianceError::ShapeMismatch {
            context: "compute_corrected_covariance_diagonal outer Hessian",
            reason: format!(
                "dimension ({}, {}) does not match q = {}",
                outer_hessian.nrows(),
                outer_hessian.ncols(),
                q,
            ),
        });
    }

    let active = active_bound_indices_for_theta(theta_at_optimum, v_ks.len(), ext_v.len());
    let (v_theta_full, rank_deficient_directions) = projected_inverse_with_inertia_gate(
        outer_hessian,
        &active,
        theta_at_optimum,
        gradient_norm,
    )?;

    // Symmetric square root of V_θ via eigendecomposition (PSD by construction).
    let (sym_evals, sym_evecs) = v_theta_full.eigh(faer::Side::Lower).map_err(|e| {
        CorrectedCovarianceError::EigendecompositionFailed {
            context: "corrected covariance hyperparameter covariance",
            reason: e.to_string(),
        }
    })?;
    let mut v_theta_sqrt = Array2::<f64>::zeros((q, q));
    for j in 0..q {
        let s = sym_evals[j];
        if s <= 0.0 {
            continue;
        }
        let scale = s.sqrt();
        for row in 0..q {
            v_theta_sqrt[[row, j]] = sym_evecs[[row, j]] * scale;
        }
    }

    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    let m = j_alpha.dot(&v_theta_sqrt);
    for i in 0..p {
        let mut row_norm_sq = 0.0;
        for j in 0..m.ncols() {
            row_norm_sq += m[[i, j]] * m[[i, j]];
        }
        diag[i] += row_norm_sq;
    }

    Ok(CorrectedCovarianceDiagonal {
        diagonal: diag,
        active_constraints: active,
        rank_deficient_directions,
    })
}


/// Enforce exact symmetry on a square matrix by averaging off-diagonal pairs.
fn enforce_symmetry_inplace(m: &mut Array2<f64>) {
    let n = m.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[[i, j]] + m[[j, i]]);
            m[[i, j]] = avg;
            m[[j, i]] = avg;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Smooth spectral regularization
// ═══════════════════════════════════════════════════════════════════════════
//
// For indefinite or near-singular Hessians, hard eigenvalue clamping
// `max(σ, ε)` is non-smooth and creates inconsistency between log|H| and
// H⁻¹ at the threshold boundary. We use instead the C∞ regularizer:
//
//   r_ε(σ) = ½(σ + √(σ² + 4ε²))
//
// Properties:
//   - C∞ and strictly positive for all σ ∈ ℝ
//   - r_ε(σ) → σ  as σ → +∞  (transparent for well-conditioned eigenvalues)
//   - r_ε(σ) → ε  as σ → 0   (smooth floor)
//   - r_ε(σ) → ε²/|σ| as σ → -∞  (damps negative eigenvalues)
//
// Its derivative is:
//
//   r'_ε(σ) = ½(1 + σ/√(σ² + 4ε²))
//
// Using the SAME r_ε for both log-determinant and inverse ensures the
// gradient is the exact derivative of a single scalar objective — no
// inconsistency from mixing different regularizations.

/// Smooth spectral regularizer: `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.
///
/// Returns a strictly positive value for any real `sigma`. For large positive
/// `sigma` this is approximately `sigma`; near zero it smoothly floors at `epsilon`.
#[inline]
pub(crate) fn spectral_regularize(sigma: f64, epsilon: f64) -> f64 {
    let disc = sigma.hypot(2.0 * epsilon);
    if sigma >= 0.0 {
        0.5 * sigma + 0.5 * disc
    } else {
        // Avoid catastrophic cancellation in 0.5 * (σ + disc) when σ is
        // large and negative: r_ε(σ) = 2 ε² / (disc - σ).
        (2.0 * epsilon * epsilon) / (disc - sigma)
    }
}


/// Compute the spectral regularization scale for a set of eigenvalues.
///
/// `ε = √(machine_eps) · p` where `p` is the matrix dimension — a
/// ρ-INDEPENDENT numerical-stability floor on the smooth regularization
/// `r_ε(σ)`.  The previous formulation `ε = √(machine_eps) · max(|σ_max|, 1)`
/// coupled ε to the largest eigenvalue of H(ρ), which made ε a function of
/// ρ whenever the Hessian spectrum moved with ρ.  That coupling leaked a
/// spurious ∂log|H|_reg/∂ρ contribution through the near-zero eigenvalues:
/// for σ_j ≪ ε we have `log r_ε(σ_j) ≈ log ε`, so `d log r_ε(σ_j)/dρ`
/// picks up `(1/ε) · dε/dρ` whenever max|σ_j| moved.  That created a
/// first-order derivative mismatch in outer REML gradients (up to ~1.5% of
/// the dominant `d log|H|/dρ` term on problems with one near-singular
/// direction, e.g. multi-block GAMLSS wiggle models where the intercept/wiggle
/// direction is effectively in the null space of the likelihood curvature).
///
/// The analytic gradient formula `tr(G_ε(H) · dH/dρ_k)` assumes ε is
/// fixed; removing the ρ-coupling restores that assumption.  Scaling ε
/// by the matrix dimension `p` (a ρ-independent integer, set by the
/// problem geometry) gives numerical stability for larger systems without
/// reintroducing ρ leakage.  The absolute floor stays below any physically
/// meaningful eigenvalue (for p ≤ 10⁶, ε ≤ 1.5e-2; well-conditioned
/// problems have min σ ≫ ε and are unaffected).
#[inline]
pub(crate) fn spectral_epsilon(eigenvalues: &[f64]) -> f64 {
    f64::EPSILON.sqrt() * (eigenvalues.len() as f64).max(1.0)
}


/// How the penalized Hessian's log-determinant and its derivatives treat the
/// spectrum below the stability floor `ε = spectral_epsilon(·)`.
///
/// Two conventions, both mathematically internally consistent:
///
/// ## `Smooth` (default — appropriate for almost all GLM/GAM families)
///
/// Eigenvalues above the structural positive-eigenvalue threshold — the same
/// ~100·p·ε_mach·‖H‖ cutoff that `fixed_subspace_penalty_rank_and_logdet`
/// applies to `log|S|_+` — contribute to `log|H|` via the smooth regularizer
/// `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.  Gradients use `φ'(σ) = 1/√(σ² + 4ε²)`
/// so that `d log|H|_reg/dρ = Σ φ'(σ_j) · u_j^T (dH/dρ) u_j` is the EXACT
/// derivative of the scalar objective `Σ log r_ε(σ_j)` over the active set.
/// For a well-conditioned H the threshold sits far below every genuine
/// eigenvalue and every pair is active, so behaviour matches the previous
/// unfiltered soft-floor formulation.  In the rank-deficient regime where
/// `rank(X) + rank(S) < p` (e.g. small-n high-dim Duchon), H has eigenvalues
/// inside the numerical noise band; those directions are also null in S, so
/// excluding them from BOTH `log|H|` and `log|S|_+` keeps the LAML ratio
/// well-defined on the identified subspace rather than driving
/// `½ log|H| − ½ log|S|_+` to −∞.
///
/// ## `HardPseudo` (opt-in for structurally rank-deficient families)
///
/// When the model is known to carry a numerical null-space direction that
/// is not informative — e.g. multi-block GAMLSS wiggle models where the
/// threshold + constant wiggle-intercept are collinear — the smooth floor
/// still contributes to `log|H|_reg` through that direction, and its
/// first-order `dσ/dρ = u^T (dH/dρ) u` estimate is unreliable because the
/// eigenvector u for a near-zero σ is a random linear combination of
/// whatever the numerical eigensolver selected inside the null space.
///
/// Under `HardPseudo`, eigenvalues satisfying `σ_j ≤ ε` are EXCLUDED from
/// `log|H|`, `tr(G_ε · A)`, `tr(H⁻¹ · ·)`, and every cross-trace.  This is
/// the exact pseudo-logdeterminant on the active eigenspace:
///
///   log|H|₊  = Σ_{σ_j > ε} log σ_j
///   d/dρ_k   = Σ_{σ_j > ε} (1/σ_j) · u_j^T (dH/dρ_k) u_j
///
/// with the smooth floor `r_ε(σ)` retained in place of `log σ` / `1/σ` so
/// there is no discontinuity as an eigenvalue crosses ε.  The key property
/// is that null-space directions drop out of both the cost and the
/// gradient in a matched way; first-order perturbation theory applies only to
/// directions that actually have curvature to perturb.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PseudoLogdetMode {
    #[default]
    Smooth,
    HardPseudo,
}


// ═══════════════════════════════════════════════════════════════════════════
//  Dense spectral HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Dense spectral Hessian operator using eigendecomposition.
///
/// Computes logdet, trace, and solve from a single eigendecomposition,
/// guaranteeing spectral consistency. Indefinite or near-singular eigenvalues
/// are handled via smooth spectral regularization `r_ε(σ)` rather than hard
/// clamping, ensuring that logdet and inverse use the same smooth mapping.
pub struct DenseSpectralOperator {
    /// Regularized eigenvalues: `r_ε(σ_i)` for each raw eigenvalue `σ_i`.
    reg_eigenvalues: Vec<f64>,
    /// Per-eigenvalue mask: `true` if the eigenpair participates in all
    /// traces, solves, and logdet contributions.  Under
    /// [`PseudoLogdetMode::Smooth`] every entry is `true`.  Under
    /// [`PseudoLogdetMode::HardPseudo`] entries with `σ_j ≤ ε` are `false`,
    /// so the numerical null space is excluded consistently from
    /// `log|H|_+`, its gradient, its cross-traces, AND `H⁻¹` solves
    /// (`H⁺` on the active subspace).
    active_mask: Vec<bool>,
    /// Eigenvectors of H (columns).
    eigenvectors: Array2<f64>,
    /// Precomputed: W = U diag(1/√r_ε(σ)) for efficient traces.
    /// trace(H⁻¹ A) = Σ (AW ⊙ W)
    w_factor: Array2<f64>,
    /// Precomputed kernel K_ab = 1 / (r_a r_b) for exact H⁻¹ cross traces in
    /// the eigenbasis.
    hinv_cross_kernel: Array2<f64>,
    /// Precomputed: G = U diag(1/√(√(σ² + 4ε²))) for logdet gradient traces.
    /// trace(G_ε(H) A) = Σ (AG ⊙ G) where G_ε uses φ'(σ) = 1/√(σ² + 4ε²).
    g_factor: Array2<f64>,
    /// Precomputed divided-difference kernel Γ for exact logdet Hessian cross traces
    /// in the eigenbasis.
    logdet_hessian_kernel: Array2<f64>,
    /// Precomputed log-determinant: Σ ln(r_ε(σ_i)).
    cached_logdet: f64,
    projected_factor_cache: ProjectedFactorCache,
    /// Full dimension.
    n_dim: usize,
}


impl DenseSpectralOperator {
    pub fn reg_eigenvalue(&self, k: usize) -> f64 {
        self.reg_eigenvalues[k]
    }
    pub fn eigenvector_entry(&self, i: usize, k: usize) -> f64 {
        self.eigenvectors[[i, k]]
    }
    /// Whether eigenpair `k` is active in the operator's logdet, traces,
    /// and solves. Under `PseudoLogdetMode::Smooth` this is always `true`;
    /// under `HardPseudo` it is `false` when `σ_k ≤ ε`.
    pub fn eigenpair_active(&self, k: usize) -> bool {
        self.active_mask[k]
    }

    /// Create from a symmetric matrix (may be indefinite or singular).
    ///
    /// The eigendecomposition is computed once. Eigenvalues are smoothly
    /// regularized via `r_ε(σ)`. All subsequent operations (logdet, trace,
    /// solve) use the regularized spectrum, ensuring mathematical consistency.
    pub fn from_symmetric(h: &Array2<f64>) -> Result<Self, String> {
        Self::from_symmetric_with_mode(h, PseudoLogdetMode::Smooth)
    }

    /// Variant of [`from_symmetric`](Self::from_symmetric) that selects the
    /// log-determinant convention.
    ///
    /// See [`PseudoLogdetMode`] for the derivation and the exact set of
    /// kernels that differ between the two modes.  At a high level:
    /// `Smooth` keeps every eigenpair in play with a soft floor, whereas
    /// `HardPseudo` masks out `σ_j ≤ ε` consistently across logdet,
    /// gradient traces, cross-traces, and the H⁻¹ kernels.
    pub fn from_symmetric_with_mode(
        h: &Array2<f64>,
        mode: PseudoLogdetMode,
    ) -> Result<Self, String> {
        use faer::Side;

        let n = h.nrows();
        if n != h.ncols() {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "HessianOperator: expected square matrix, got {}×{}",
                    n,
                    h.ncols()
                ),
            }
            .into());
        }

        let (eigenvalues, eigenvectors) = h
            .eigh(Side::Lower)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        let epsilon = spectral_epsilon(eigenvalues.as_slice().unwrap());

        // `active[j]` selects which eigenpairs participate in every trace
        // and in the cached logdet.
        //
        // `Smooth` is the regularized full-spectrum mode: every eigenpair stays
        // active and singular directions are handled only through
        // `r_ε(σ)`. This is the documented default semantics used by the
        // unified REML/LAML objective.
        //
        // `HardPseudo` is the identified-subspace mode: eigenpairs with
        // `σ_j ≤ ε` are excluded consistently from logdet, traces, and solves.
        // Families that need exact pseudo-determinant behaviour opt into this
        // mode explicitly through `pseudo_logdet_mode()`.
        let active: Vec<bool> = match mode {
            PseudoLogdetMode::Smooth => vec![true; n],
            PseudoLogdetMode::HardPseudo => eigenvalues.iter().map(|&s| s > epsilon).collect(),
        };

        // Apply smooth regularization to all eigenvalues (even inactive ones:
        // `reg_eigenvalues[j]` is still consulted by `trace_hinv_product`
        // when using `w_factor[:, j]`, but we zero-out `w_factor[:, j]` for
        // inactive eigenpairs so those entries never enter any sum).
        let reg_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .map(|&sigma| spectral_regularize(sigma, epsilon))
            .collect();

        // Build W factor for traces: W[:, j] = u_j / sqrt(r_ε(σ_j)) on
        // active eigenpairs, 0 otherwise.
        let mut w_factor = Array2::zeros((n, n));
        for j in 0..n {
            if !active[j] {
                continue;
            }
            let scale = 1.0 / reg_eigenvalues[j].sqrt();
            for row in 0..n {
                w_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        let mut hinv_cross_kernel = Array2::zeros((n, n));
        for a in 0..n {
            if !active[a] {
                continue;
            }
            let inv_ra = 1.0 / reg_eigenvalues[a];
            for b in 0..n {
                if !active[b] {
                    continue;
                }
                hinv_cross_kernel[[a, b]] = inv_ra / reg_eigenvalues[b];
            }
        }

        // Build G factor for logdet gradient traces: G[:, j] = u_j / sqrt(√(σ_j² + 4ε²))
        // φ'(σ) = 1/√(σ² + 4ε²), so we need 1/√(φ'(σ)) = (σ² + 4ε²)^{1/4}
        // Actually: tr(G_ε A) = Σ_j φ'(σ_j) u_jᵀ A u_j = Σ (AG ⊙ G)
        // where G[:, j] = u_j · √(φ'(σ_j)) = u_j / (σ_j² + 4ε²)^{1/4}
        let four_eps_sq = 4.0 * epsilon * epsilon;
        let mut g_factor = Array2::zeros((n, n));
        for j in 0..n {
            if !active[j] {
                continue;
            }
            let sigma = eigenvalues[j];
            let phi_prime = 1.0 / (sigma * sigma + four_eps_sq).sqrt();
            let scale = phi_prime.sqrt();
            for row in 0..n {
                g_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        let mut logdet_hessian_kernel = Array2::zeros((n, n));
        let sqrt_disc: Vec<f64> = eigenvalues
            .iter()
            .map(|&s| (s * s + four_eps_sq).sqrt())
            .collect();
        for a in 0..n {
            if !active[a] {
                continue;
            }
            let sigma_a = eigenvalues[a];
            let sqrt_a = sqrt_disc[a];
            for b in 0..n {
                if !active[b] {
                    continue;
                }
                logdet_hessian_kernel[[a, b]] = if a == b {
                    -sigma_a / (sqrt_a * sqrt_a * sqrt_a)
                } else {
                    let sigma_b = eigenvalues[b];
                    let sqrt_b = sqrt_disc[b];
                    -(sigma_a + sigma_b) / (sqrt_a * sqrt_b * (sqrt_a + sqrt_b))
                };
            }
        }

        // Precompute logdet: Σ_{active} ln(r_ε(σ_i)).
        let cached_logdet: f64 = reg_eigenvalues
            .iter()
            .zip(active.iter())
            .filter_map(|(&v, &act)| if act { Some(v.ln()) } else { None })
            .sum();

        Ok(Self {
            reg_eigenvalues,
            active_mask: active,
            eigenvectors,
            w_factor,
            hinv_cross_kernel,
            g_factor,
            logdet_hessian_kernel,
            cached_logdet,
            projected_factor_cache: ProjectedFactorCache::default(),
            n_dim: n,
        })
    }

    #[inline]
    fn rotate_to_eigenbasis(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = crate::faer_ndarray::fast_atb(&self.eigenvectors, matrix);
        crate::faer_ndarray::fast_ab(&left, &self.eigenvectors)
    }

    /// Factor `F` satisfying `trace(G_epsilon(H) A) = trace(F^T A F)`.
    ///
    /// Structured row-local operators use this to contract the logdet-gradient
    /// trace directly in row space without forming `A F` in coefficient space.
    pub fn logdet_gradient_factor(&self) -> &Array2<f64> {
        &self.g_factor
    }

    #[inline]
    fn trace_hinv_product_cross_rotated(&self, a_rot: &Array2<f64>, b_rot: &Array2<f64>) -> f64 {
        let mut result = 0.0;
        for ((kernel_row, a_row), b_col) in self
            .hinv_cross_kernel
            .rows()
            .into_iter()
            .zip(a_rot.rows().into_iter())
            .zip(b_rot.columns().into_iter())
        {
            for ((kernel, a_value), b_value) in kernel_row
                .iter()
                .copied()
                .zip(a_row.iter().copied())
                .zip(b_col.iter().copied())
            {
                result += kernel * a_value * b_value;
            }
        }
        result
    }

    #[inline]
    fn trace_hinv_product_cross_dense(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let a_rot = self.rotate_to_eigenbasis(a);
        if std::ptr::eq(a, b) {
            return self.trace_hinv_product_cross_rotated(&a_rot, &a_rot);
        }
        let b_rot = self.rotate_to_eigenbasis(b);
        self.trace_hinv_product_cross_rotated(&a_rot, &b_rot)
    }

    #[inline]
    fn projected_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let left = crate::faer_ndarray::fast_atb(&self.w_factor, matrix);
        crate::faer_ndarray::fast_ab(&left, &self.w_factor)
    }

    #[inline]
    fn projected_operator(&self, factor: &Array2<f64>, op: &dyn HyperOperator) -> Array2<f64> {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result = op.projected_matrix_cached(factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::projected_operator dim={} rank={} implicit={}",
                self.n_dim,
                factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.projected_matrix_cached(factor, &self.projected_factor_cache)
        }
    }

    #[inline]
    fn trace_projected_cross(&self, left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        let mut result = 0.0;
        for (left_row, right_col) in left.rows().into_iter().zip(right.columns().into_iter()) {
            for (left_value, right_value) in left_row.iter().copied().zip(right_col.iter().copied())
            {
                result += left_value * right_value;
            }
        }
        result
    }

    #[inline]
    fn trace_logdet_hessian_cross_rotated(
        &self,
        h_i_rot: &Array2<f64>,
        h_j_rot: &Array2<f64>,
    ) -> f64 {
        let mut result = 0.0;
        for ((kernel_row, h_i_row), h_j_col) in self
            .logdet_hessian_kernel
            .rows()
            .into_iter()
            .zip(h_i_rot.rows().into_iter())
            .zip(h_j_rot.columns().into_iter())
        {
            for ((kernel, h_i_value), h_j_value) in kernel_row
                .iter()
                .copied()
                .zip(h_i_row.iter().copied())
                .zip(h_j_col.iter().copied())
            {
                result += kernel * h_i_value * h_j_value;
            }
        }
        result
    }
}


/// Coalesce repeated identical `[STAGE]` log lines from `DenseSpectralOperator`
/// methods. First occurrence of a (method, dims, implicit-flags) signature
/// logs immediately; identical consecutive repeats are silenced and accrue
/// into a counter, emitting heartbeat summaries at doubling cadence
/// (2, 4, 8, 16, …) and a final summary when the signature changes.
fn dense_spectral_stage_log(signature: &str, elapsed_s: f64) {
    use std::sync::Mutex;
    struct Repeat {
        signature: String,
        count: u64,
        total: f64,
        min: f64,
        max: f64,
        next_heartbeat: u64,
    }
    static REPEAT: Mutex<Option<Repeat>> = Mutex::new(None);

    let mut guard = match REPEAT.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some(state) = guard.as_mut() {
        if state.signature == signature {
            state.count += 1;
            state.total += elapsed_s;
            if elapsed_s < state.min {
                state.min = elapsed_s;
            }
            if elapsed_s > state.max {
                state.max = elapsed_s;
            }
            if state.count >= state.next_heartbeat {
                log::info!(
                    "[STAGE] {} (×{} so far, total={:.3}s min={:.3}s max={:.3}s avg={:.3}s)",
                    state.signature,
                    state.count,
                    state.total,
                    state.min,
                    state.max,
                    state.total / state.count as f64,
                );
                state.next_heartbeat = state.next_heartbeat.saturating_mul(2);
            }
            return;
        }
        // Signature changed — flush a final summary for the previous one
        // when it ran more than once (the first occurrence already logged
        // its own line, so a count of 1 needs no follow-up).
        if state.count > 1 {
            log::info!(
                "[STAGE] {} final ×{} total={:.3}s min={:.3}s max={:.3}s avg={:.3}s",
                state.signature,
                state.count,
                state.total,
                state.min,
                state.max,
                state.total / state.count as f64,
            );
        }
    }

    log::info!("[STAGE] {} elapsed={:.3}s", signature, elapsed_s);
    *guard = Some(Repeat {
        signature: signature.to_string(),
        count: 1,
        total: elapsed_s,
        min: elapsed_s,
        max: elapsed_s,
        next_heartbeat: 2,
    });
}


impl HessianOperator for DenseSpectralOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn as_exact_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }

    fn assemble_h_dense_for_tangent_projection(&self) -> Result<Array2<f64>, String> {
        Ok(assemble_h_raw_dense(self))
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(H_reg⁻¹ A) = Σ_j (1/r_ε(σ_j)) uⱼᵀAuⱼ
        // Computed as Σ (AW ⊙ W) where W = U diag(1/√r_ε(σ)).
        let aw = a.dot(&self.w_factor);
        aw.iter()
            .zip(self.w_factor.iter())
            .map(|(&a, &w)| a * w)
            .sum()
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // H_reg⁻¹ v = Σ_j (1/r_ε(σ_j)) (uⱼᵀv) uⱼ.  Inactive eigenpairs
        // (σ_j ≤ ε under `HardPseudo`) are skipped so the returned vector
        // lives entirely in the active subspace — otherwise v_k picks up a
        // huge spurious component along the numerical null space direction
        // (coefficient ~ 1/r_ε(σ_j) for σ_j ≈ 0) that is not part of the
        // IFT mode response `dβ̂/dρ` and would leak into the REML gradient.
        let mut result = Array1::zeros(self.n_dim);
        for j in 0..self.n_dim {
            if !self.active_mask[j] {
                continue;
            }
            let u = self.eigenvectors.column(j);
            let coeff = u.dot(rhs) / self.reg_eigenvalues[j];
            for row in 0..self.n_dim {
                result[row] += coeff * u[row];
            }
        }
        result
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut projected = self.eigenvectors.t().dot(rhs);
        for j in 0..self.n_dim {
            if self.active_mask[j] {
                let scale = 1.0 / self.reg_eigenvalues[j];
                projected.row_mut(j).mapv_inplace(|value| value * scale);
            } else {
                // Zero out inactive eigendirections so `H⁺` acts on the
                // active subspace only (mirroring the single-vector `solve`).
                projected.row_mut(j).fill(0.0);
            }
        }
        self.eigenvectors.dot(&projected)
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        self.trace_hinv_product_cross_dense(a, b)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result =
                op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::trace_hinv_operator dim={} rank={} implicit={}",
                self.n_dim,
                self.w_factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.trace_projected_factor_cached(&self.w_factor, &self.projected_factor_cache)
        }
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let left = self.w_factor.t().dot(matrix).dot(&self.w_factor);
        let right = self.projected_operator(&self.w_factor, op);
        self.trace_projected_cross(&left, &right)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let left_proj = self.projected_operator(&self.w_factor, left);
            let result = if std::ptr::addr_eq(left, right) {
                self.trace_projected_cross(&left_proj, &left_proj)
            } else {
                let right_proj = self.projected_operator(&self.w_factor, right);
                self.trace_projected_cross(&left_proj, &right_proj)
            };
            let signature = format!(
                "DenseSpectralOperator::trace_hinv_operator_cross dim={} rank={} left_implicit={} right_implicit={}",
                self.n_dim,
                self.w_factor.ncols(),
                left.is_implicit(),
                right.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            let left_proj = self.projected_operator(&self.w_factor, left);
            if std::ptr::addr_eq(left, right) {
                self.trace_projected_cross(&left_proj, &left_proj)
            } else {
                let right_proj = self.projected_operator(&self.w_factor, right);
                self.trace_projected_cross(&left_proj, &right_proj)
            }
        }
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) A) = Σ_j φ'(σ_j) uⱼᵀAuⱼ
        // where φ'(σ) = 1/√(σ² + 4ε²).
        // Computed as Σ (AG ⊙ G) where G = U diag(√φ'(σ)).
        let ag = a.dot(&self.g_factor);
        ag.iter()
            .zip(self.g_factor.iter())
            .map(|(&a, &g)| a * g)
            .sum()
    }

    fn xt_logdet_kernel_x_diagonal(&self, x: &DesignMatrix) -> Array1<f64> {
        // h^G_i = ‖(X G)_{i,:}‖² where G_ε = G Gᵀ and G = self.g_factor.
        // The dominant cost at large scale is the (n × p)·(p × rank) matmul
        // — for matern60 with n=320K, p=101 that's ~3.3 GFLOPs and the
        // ndarray default `.dot()` runs single-threaded (no BLAS feature
        // enabled in this crate's build), so we route through faer's parallel
        // SIMD GEMM. For operator-backed (Lazy) designs we additionally
        // stream by row chunk so we never materialize the full (n×p) block
        // at large scale.
        let n = x.nrows();
        let p = x.ncols();
        let rank = self.g_factor.ncols();
        let mut h = Array1::<f64>::zeros(n);
        if n == 0 || p == 0 || rank == 0 {
            return h;
        }
        // Issue #922: offload this n-dependent pass to the device pool when a
        // GPU was probed and n·p² clears the dispatch floor. The result is the
        // same f64 arithmetic (X·G then row-wise ‖·‖²), just relocated across
        // every device via `scatter_batched`; any failure falls through to the
        // faer CPU stream below so the REML criterion is byte-for-byte
        // unchanged on machines without a GPU.
        if let Some(gpu) =
            crate::gpu::linalg::try_fast_spectral_leverage_diagonal(x, self.g_factor.view())
        {
            return gpu;
        }
        let chunk_rows = byte_balanced_row_chunk(p + rank, n);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let rows = x.try_row_chunk(start..end).unwrap_or_else(|err| {
                // SAFETY: `try_row_chunk` only fails on operator implementation
                // bugs — the `start..end` range is constructed from
                // `0..n = 0..x.nrows()` with `end = (start+block).min(n)`,
                // so it is always a valid sub-range of `x`. A failure here
                // means the operator violated its row-chunk contract.
                // SAFETY: row range built from 0..x.nrows(); failure means operator broke its contract.
                reml_contract_panic(format!(
                    "xt_logdet_kernel_x_diagonal: row chunk failed: {err}"
                ))
            });
            let xg = crate::faer_ndarray::fast_ab(&rows, &self.g_factor);
            for (local, row) in xg.outer_iter().enumerate() {
                h[start + local] = row.iter().map(|v| v * v).sum();
            }
            start = end;
        }
        h
    }

    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(G_ε A) = Σ (A·G ⊙ G) for block-local A.
        // Only needs G[start..end, :] — O(block² × rank) instead of O(p² × rank).
        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let ag = block.dot(&g_block);
        scale
            * ag.iter()
                .zip(g_block.iter())
                .map(|(&a, &g)| a * g)
                .sum::<f64>()
    }

    fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(H_reg⁻¹ A) = Σ (A·W ⊙ W) for block-local A.
        let w_block = self.w_factor.slice(ndarray::s![start..end, ..]);
        let aw = block.dot(&w_block);
        scale
            * aw.iter()
                .zip(w_block.iter())
                .map(|(&a, &w)| a * w)
                .sum::<f64>()
    }

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(H⁻¹ A H⁻¹ A) where A = scale · embed(block, start, end) and
        // `block` is the symmetric (b × b) local matrix.
        //
        // H⁻¹ = W W^T, so the symmetric block is
        //   H⁻¹_block = W_block · W_block^T,   W_block = W[start..end, :].
        // For block-local A, only the [start..end, start..end] sub-block of
        //   H⁻¹ A H⁻¹ A
        // contributes nonzero diagonal entries:
        //   tr(H⁻¹ A H⁻¹ A) = scale² · tr( (H⁻¹_block · B)² )
        //                    = scale² · tr( (W_block^T B W_block)² )
        // (cyclic on the rank-sized symmetric M = W_block^T B W_block, then
        // tr(M²) = ||M||_F² because B is symmetric so M is symmetric).
        let w_block = self.w_factor.slice(ndarray::s![start..end, ..]);
        let bw = block.dot(&w_block); // (b × rank)
        let m = w_block.t().dot(&bw); // (rank × rank), symmetric for symmetric block
        let scale_sq = scale * scale;
        scale_sq * m.iter().map(|&v| v * v).sum::<f64>()
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if log::log_enabled!(log::Level::Info) {
            let start = std::time::Instant::now();
            let result =
                op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache);
            let signature = format!(
                "DenseSpectralOperator::trace_logdet_operator dim={} rank={} implicit={}",
                self.n_dim,
                self.g_factor.ncols(),
                op.is_implicit(),
            );
            dense_spectral_stage_log(&signature, start.elapsed().as_secs_f64());
            result
        } else {
            op.trace_projected_factor_cached(&self.g_factor, &self.projected_factor_cache)
        }
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        if std::ptr::eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.rotate_to_eigenbasis(h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_cross_matrix_operator(
        &self,
        h_i: &Array2<f64>,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.rotate_to_eigenbasis(h_i);
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_cross_operator(
        &self,
        h_i: &dyn HyperOperator,
        h_j: &dyn HyperOperator,
    ) -> f64 {
        let hp_i = self.projected_operator(&self.eigenvectors, h_i);
        if std::ptr::addr_eq(h_i, h_j) {
            return self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_i);
        }
        let hp_j = self.projected_operator(&self.eigenvectors, h_j);
        self.trace_logdet_hessian_cross_rotated(&hp_i, &hp_j)
    }

    fn trace_logdet_hessian_crosses(&self, matrices: &[&Array2<f64>]) -> Array2<f64> {
        let n = matrices.len();
        let rotated = matrices
            .iter()
            .map(|matrix| self.rotate_to_eigenbasis(matrix))
            .collect::<Vec<_>>();
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let value = self.trace_logdet_hessian_cross_rotated(&rotated[i], &rotated[j]);
                out[[i, j]] = value;
                out[[j, i]] = value;
            }
        }
        out
    }

    fn active_rank(&self) -> usize {
        self.active_mask.iter().filter(|&&active| active).count()
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
        false
    }

    fn logdet_traces_match_hinv_kernel(&self) -> bool {
        false
    }

    fn as_dense_spectral(&self) -> Option<&DenseSpectralOperator> {
        Some(self)
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Cholesky HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse Cholesky Hessian operator.
///
/// Wraps an existing `SparseExactFactor` and provides logdet, trace, and solve
/// from the same Cholesky factorization.
pub struct SparseCholeskyOperator {
    /// The sparse Cholesky factorization.
    factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
    /// Takahashi selected inverse (precomputed H^{-1} entries on the filled pattern of L).
    /// When available, trace computations use direct lookups instead of column solves.
    takahashi: Option<std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>>,
    /// Precomputed log-determinant from the Cholesky diagonal.
    cached_logdet: f64,
    /// Dimension of H.
    n_dim: usize,
}
