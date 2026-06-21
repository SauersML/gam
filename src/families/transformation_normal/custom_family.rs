use super::*;

// ---------------------------------------------------------------------------
// CustomFamily implementation
// ---------------------------------------------------------------------------

impl CustomFamily for TransformationNormalFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        crate::families::block_layout::block_count::validate_block_count::<
            TransformationNormalError,
        >("TransformationNormalFamily", 1, block_states.len())?;
        let evaluate_start = std::time::Instant::now();
        let beta = &block_states[0].beta;
        let row_q_start = std::time::Instant::now();
        let row_quantities = self.row_quantities(beta)?;
        log::info!(
            "[STAGE] CTN row_quantities (h, h', 1/h', powers) n={} elapsed={:.3}s",
            row_quantities.h.len(),
            row_q_start.elapsed().as_secs_f64(),
        );
        let h = row_quantities.h.as_ref();
        let n = h.len();

        let log_likelihood = row_quantities.log_likelihood;
        // SCOP gradient and exact negative Hessian. Response column 0 is the
        // linear location component b(x); response columns >=1 are squared
        // γ_k(x)^2 shape components.
        let grad_start = std::time::Instant::now();
        let (grad, hessian) = self.scop_gradient_and_negative_hessian(beta, &row_quantities)?;
        log::info!(
            "[STAGE] CTN gradient terms n={} p={} elapsed={:.3}s",
            n,
            grad.len(),
            grad_start.elapsed().as_secs_f64(),
        );

        let hess_start = std::time::Instant::now();
        let p_dim = hessian.nrows() as u64;
        let n_u64 = n as u64;
        log::info!(
            "[STAGE] CTN hessian terms (SCOP exact dense) n={} p={} flops~{} elapsed={:.3}s",
            n,
            p_dim,
            n_u64.saturating_mul(p_dim).saturating_mul(p_dim),
            hess_start.elapsed().as_secs_f64(),
        );
        log::info!(
            "[STAGE] CTN evaluate end n={} p={} elapsed={:.3}s",
            n,
            p_dim,
            evaluate_start.elapsed().as_secs_f64(),
        );

        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: grad,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        crate::families::block_layout::block_count::validate_block_count::<
            TransformationNormalError,
        >("TransformationNormalFamily", 1, block_states.len())?;
        // The line search uses NEG_INFINITY as the barrier-violation signal,
        // so we can't propagate the row_quantities Err here. Translate any
        // h' validation failure back into the NEG_INFINITY rejection contract.
        let row_quantities = match self.row_quantities(&block_states[0].beta) {
            Ok(rq) => rq,
            Err(_) => return Ok(f64::NEG_INFINITY),
        };
        Ok(row_quantities.log_likelihood)
    }

    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        // When an outer-score subsample is installed, route through a
        // mask-aware family clone whose `effective_weights()` returns the
        // HT-weighted per-row weights. Because every term inside
        // `build_transformation_row_derived` is linear in `wᵢ`, the row-LL
        // accumulator yields `Σᵢ (mᵢ · wᵢ) · row_ll_i` — the unbiased
        // Horvitz-Thompson estimator of the full-data LL.
        match self.maybe_with_outer_subsample_from_options(options) {
            Ok(Some(masked)) => masked.log_likelihood_only(block_states),
            Ok(None) => self.log_likelihood_only(block_states),
            Err(e) => Err(e.into()),
        }
    }

    /// Log-likelihood + flat joint gradient without building the dense Hessian.
    ///
    /// The default trait implementation returns `None`, so the joint-Newton
    /// inner solver falls back to `evaluate()` to obtain the gradient — and
    /// that side-effects a full `Θ(n p²)` `weighted_gram` Hessian build at
    /// every inner iteration. CTN's gradient is structurally
    ///
    ///   `∇ℓ = -X_val^T (w·h) + X_deriv^T (w/h')`,
    ///
    /// which is two `transpose_mul`s through the existing Khatri-Rao operators
    /// and one `Θ(n)` row reduction — `Θ(n p)` total. At large scale that is
    /// ~10⁷ FLOPs per call versus ~3·10¹⁰ for the full `evaluate`, so wiring
    /// this override is the gating condition for routing CTN's inner solve
    /// through the matrix-free joint-Newton path without paying the dense H
    /// tax on every gradient refresh.
    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        crate::families::block_layout::block_count::validate_block_count::<
            TransformationNormalError,
        >("TransformationNormalFamily", 1, block_states.len())?;
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let log_likelihood = row_quantities.log_likelihood;
        let gradient = self.scop_gradient(beta, &row_quantities)?;
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // The Hessian depends on β through 1/h'² where h' = X_deriv · β.
        true
    }

    fn joint_jeffreys_term_required(&self) -> bool {
        // CTN models a continuous response through a monotone transformation
        // `h(Y|x) ~ N(0,1)`; there is no separation/under-identification
        // regime to bound. The Fisher information is `O(n)` on every
        // identified direction at every working point, so the conditioning
        // gate inside `joint_jeffreys_term` smooth-steps the contribution to
        // zero as soon as `λ_min ≥ 16`. The construction up to that gate is
        // not free though: each evaluation runs `p` SCOP directional
        // derivatives of the joint Hessian, called three times per inner
        // cycle (head-KKT gradient, joint Newton step RHS, post-step KKT
        // residual) and once per outer evaluation. At large scale —
        // `bench/large_scale` `rust_margslope_aniso_duchon16d_*` with
        // `p=144`, `n=20000` — that single source dominates each inner
        // cycle (~230 s/cycle observed in CI; ~5 700 cycles × 5.7 min
        // ⇒ multi-hour hang) and exhausts the 40-minute CI budget before
        // the inner solve converges. Disabling the term here keeps the
        // un-augmented inner Newton path (still consistent with the
        // outer LAML logdet, which also drops the `H_Φ` contribution
        // through the same family gate).
        false
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Khatri–Rao tensor design: the coefficient block is X = R ⊙ C with
        // rows length p_resp · p_cov. Two regimes:
        //
        // * **Dense regime** (small enough that the unified evaluator builds
        //   `weighted_gram` directly): per-evaluation cost is the dense
        //   `n · (p_resp · p_cov)²` Khatri–Rao gram build.
        //
        // * **Matrix-free regime** (large enough that
        //   `use_joint_matrix_free_path` returns true and the evaluator
        //   factors `H v` through `forward_mul` / `transpose_mul` on the
        //   Khatri–Rao operands): per-`Hv` matvec cost is just
        //   `n · (p_resp + p_cov)` flops — see `ctn_matrix_free_workspace`.
        //   This is only an inner coefficient-space cost estimate; outer
        //   θθ Hessian availability is declared separately.
        let n_usize = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols() as u64;
        let p_cov = self.covariate_design.ncols() as u64;
        let expected_p_total = p_resp.saturating_mul(p_cov);
        // Block-spec preview is optional. Callers without an assembled
        // ParameterBlockSpec — cost estimators, planners, the
        // BlockwiseFitOptions screen, every code path that asks "how
        // expensive would the Hessian be on *this* family?" — pass `&[]`.
        // The Khatri–Rao layout is fully determined by `p_resp · p_cov`
        // from the family state, so fall back to `expected_p_total` for
        // the empty-specs preview rather than returning the `u64::MAX`
        // unreachable sentinel that would dominate every cost comparison.
        // When specs IS supplied we still enforce the structural
        // expectation `spec.design.ncols() == p_resp · p_cov`; a mismatch
        // is the only condition that legitimately surfaces the sentinel.
        let p_total = match specs {
            [] => expected_p_total,
            [spec] if spec.design.ncols() as u64 == expected_p_total => spec.design.ncols() as u64,
            _ => return u64::MAX,
        };
        let n = n_usize as u64;
        // Shared operator-aware gate (see `coefficient_cost`): matrix-free Hv
        // streams the Khatri–Rao operands at `n · (p_resp + p_cov)`; the dense
        // fallback is the `n · p_total²` Khatri–Rao gram build. The dense count
        // is supplied inline rather than via `joint_coupled_coefficient_hessian_cost`
        // because the empty-specs preview must still report `n · p_total²` from
        // the family-derived `p_total`, not the `n · 0²` an empty `specs` sum yields.
        crate::families::coefficient_cost::operator_aware_hessian_cost(
            p_total,
            n,
            n.saturating_mul(p_resp.saturating_add(p_cov)),
            n.saturating_mul(p_total.saturating_mul(p_total)),
        )
    }

    fn coefficient_gradient_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // One row-quantity pass plus two transpose products. The SCOP derivative
        // is structurally positive, so coefficient line searches no longer run a
        // full derivative-grid fraction-to-boundary scan on every attempt.
        self.coefficient_hessian_cost(specs) / 2
    }

    fn outer_derivative_policy(
        &self,
        specs: &[crate::families::custom_family::ParameterBlockSpec],
        psi_dim: usize,
        options: &crate::families::custom_family::BlockwiseFitOptions,
    ) -> crate::families::custom_family::OuterDerivativePolicy {
        // The generic default model in `CustomFamily::outer_derivative_policy`
        // uses `coefficient_hessian_cost × (rho_dim + psi_dim)`, which
        // overstates CTN's actual per-eval Hessian work because the SCOP
        // joint-Hessian path is row-streaming through the Khatri-Rao jet
        // (its `O(n · p)` matrix-free HVP, not `O(n · p²)` dense build).
        // Use a CTN-specific shape:
        //
        // * gradient ≈ `n · (rho_dim + psi_dim) · p_total`
        //   (one directional jet sweep per outer coordinate, row-streamed)
        // * Hessian  ≈ min(dense build, matrix-free HVP loop)
        //   * dense  ≈ `n · (rho_dim + psi_dim) · p_total^2`
        //   * mfree  ≈ `n · (rho_dim + psi_dim) · p_total · rho_dim`
        let capability = self.exact_outer_derivative_order(specs, options);
        let n = specs.first().map_or(0u128, |s| s.design.nrows() as u128);
        let p_total: u128 = specs
            .iter()
            .map(|s| s.design.ncols() as u128)
            .fold(0u128, |acc, x| acc.saturating_add(x));
        let rho_dim: u128 = specs
            .iter()
            .map(|s| s.penalties.len() as u128)
            .fold(0u128, |acc, x| acc.saturating_add(x));
        let k = rho_dim.saturating_add(psi_dim as u128).max(1);
        let p_eff = p_total.max(1);
        // Gradient work: one row sweep per outer coordinate.
        let work_grad = n.saturating_mul(k).saturating_mul(p_eff);
        // Hessian work: pick whichever access shape would dominate. The
        // amortization gate in `should_build_dense` (P2.2) picks the
        // cheaper path at execution time; the policy budget mirrors that
        // by taking the min so that genuinely Hessian-prohibitive
        // problems still downgrade through the budget ceiling.
        let dense_hess = work_grad.saturating_mul(p_eff);
        let mfree_hess = work_grad.saturating_mul(rho_dim.max(1));
        let work_hess = dense_hess.min(mfree_hess);
        crate::families::custom_family::OuterDerivativePolicy {
            capability,
            predicted_hessian_work: work_hess,
            predicted_gradient_work: work_grad,
            // CTN's outer-score reductions are mathematically per-row
            // sums whose contributions are linear in `wᵢ` at every assembly
            // site (gradient, joint Hessian dense / matvec / diagonal, ψ,
            // ψ-ψ, log-likelihood). The `_with_options` overrides install a
            // mask-aware family clone whose `effective_weights()` returns
            // `wᵢ · mᵢ` (HT-weighted), yielding an unbiased estimator
            // `E[score_subsample] = score_full`. The persistent
            // dense-Hessian cache is keyed on the mask hash so subsampled
            // and full-data builds at the same β do not alias.
            subsample_capable: true,
        }
    }

    fn outer_seed_config(&self, n_params: usize) -> crate::seeding::SeedConfig {
        crate::seeding::SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: if n_params <= 8 { 1 } else { 2 },
            seed_budget: 1,
            screen_max_inner_iterations: 2,
            risk_profile: crate::seeding::SeedRiskProfile::Gaussian,
            num_auxiliary_trailing: 0,
            over_smoothing_probe_rho: None,
        }
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        crate::families::block_layout::block_count::validate_block_count::<
            TransformationNormalError,
        >("TransformationNormalFamily", 1, block_states.len())?;
        if delta.len() != block_states[0].beta.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "CTN line-search step length {} != beta length {}",
                    delta.len(),
                    block_states[0].beta.len()
                ),
            }
            .into());
        }
        // SCOP encodes monotonicity as
        //   h'(y, x) = epsilon + sum_k M_k(y) * gamma_k(x)^2.
        // With nonnegative M-spline derivative basis rows, every finite beta is
        // interior-feasible. A derivative-grid fraction-to-boundary scan is pure
        // overhead and was the dominant CTN large-scale line-search cost.
        Ok(None)
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_index != 0 {
            return Ok(None);
        }
        // The CTN tensor design is intentionally factored. Strict monotonicity
        // is encoded structurally as `h' = ε + Σ M_r γ_r²`, so there are no
        // dense active-set constraints to expose here.
        Ok(None)
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let dd = self.scop_hessian_directional_derivative(beta, d_beta, &row_quantities)?;
        Ok(Some(dd))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Single block: joint Hessian = block Hessian.
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let (_, hessian) = self.scop_gradient_and_negative_hessian(beta, &row_quantities)?;
        Ok(Some(hessian))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_hessian_directional_derivative(block_states, 0, d_beta_flat)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let d2 = self.scop_hessian_second_directional_derivative(
            beta,
            d_beta_u_flat,
            d_beta_v_flat,
            &row_quantities,
        )?;
        Ok(Some(d2))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if psi_derivs.is_empty() || psi_index >= psi_derivs[0].len() {
            return Ok(None);
        }
        let psi_first_start = std::time::Instant::now();
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let row = self.row_quantities(beta)?;
        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let op_arc = Arc::clone(
            deriv
                .implicit_operator
                .as_ref()
                .expect("validated CTN psi derivative operator disappeared"),
        );
        let terms = self.scop_psi_terms(beta, &row, op, op_arc, axis)?;

        log::info!(
            "[STAGE] CTN psi first-order terms axis={} psi_index={} elapsed={:.3}s",
            deriv.implicit_axis,
            psi_index,
            psi_first_start.elapsed().as_secs_f64(),
        );

        Ok(Some(terms))
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if psi_derivs.is_empty() || psi_i >= psi_derivs[0].len() || psi_j >= psi_derivs[0].len() {
            return Ok(None);
        }
        let psi_pair_start = std::time::Instant::now();
        let deriv_i = &psi_derivs[0][psi_i];
        let deriv_j = &psi_derivs[0][psi_j];
        let beta = &block_states[0].beta;
        let row = self.row_quantities(beta)?;
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi terms beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }

        let op = deriv_i
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis_i = deriv_i.implicit_axis;
        let axis_j = deriv_j.implicit_axis;

        let (objective_psi_psi, score_psi_psi, _) = self
            .scop_psi_psi_value_score_hvp_from_operator(
                beta,
                op,
                axis_i,
                axis_j,
                row.gamma.view(),
                row.h.view(),
                row.h_prime.view(),
                row.endpoint_q.as_slice(),
                None,
            )?;
        let hessian_psi_psi_operator: Box<dyn HyperOperator> =
            Box::new(TransformationNormalPsiPsiHessianOperator::new(
                Arc::new(self.clone()),
                beta.clone(),
                Arc::clone(
                    deriv_i
                        .implicit_operator
                        .as_ref()
                        .expect("validated CTN psi derivative has an implicit operator"),
                ),
                axis_i,
                axis_j,
                Arc::clone(&row.gamma),
                Arc::clone(&row.h),
                Arc::clone(&row.h_prime),
                Arc::clone(&row.endpoint_q),
            ));

        // Result-validation gate. A trial point can still make the SCOP row
        // terms non-finite through an invalid h' or an exploding ψ second
        // derivative in the covariate basis. Surface that as an infeasible
        // exact-Newton evaluation instead of passing NaNs into the unified
        // outer evaluator.
        if !objective_psi_psi.is_finite() || !score_psi_psi.iter().all(|v| v.is_finite()) {
            return Err(TransformationNormalError::NonFinite {
                reason: format!(
                    "TransformationNormalFamily exact ψ-ψ second-order terms produced \
                 non-finite values at psi_i={psi_i}, psi_j={psi_j}: \
                 obj_finite={}, score_all_finite={}. \
                 The outer evaluator should retreat from this trial point.",
                    objective_psi_psi.is_finite(),
                    score_psi_psi.iter().all(|v| v.is_finite()),
                ),
            }
            .into());
        }

        log::info!(
            "[STAGE] CTN psi-psi pair (psi_i={}, psi_j={}, axes={},{}) elapsed={:.3}s",
            psi_i,
            psi_j,
            deriv_i.implicit_axis,
            deriv_j.implicit_axis,
            psi_pair_start.elapsed().as_secs_f64(),
        );

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        if psi_derivs.is_empty() || psi_index >= psi_derivs[0].len() {
            return Ok(None);
        }
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let row = self.row_quantities(beta)?;
        let hess =
            self.scop_psi_hessian_directional_derivative(beta, d_beta_flat, &row, op, axis)?;
        Ok(Some(hess))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        crate::families::block_layout::block_count::validate_block_count::<
            TransformationNormalError,
        >("TransformationNormalFamily", 1, block_states.len())?;
        if !self.inner_coefficient_hessian_hvp_available(specs) {
            return Err(TransformationNormalError::InvalidInput {
                reason: "TransformationNormalFamily joint Hessian workspace received incompatible block specs"
                    .to_string(),
            }
            .into());
        }
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        // Expected HVP reuse this workspace will service before its
        // `(β, row_quantities)` key advances. The outer-eval trace path
        // performs ~`2·rho_dim` HVPs plus one diagonal call against the
        let workspace = TransformationNormalJointHessianWorkspace::new(
            Arc::new(self.clone()),
            beta.clone(),
            row_quantities.clone(),
        )?;
        Ok(Some(
            Arc::new(workspace) as Arc<dyn ExactNewtonJointHessianWorkspace>
        ))
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.inner_coefficient_hessian_hvp_available(specs) {
            return Err(TransformationNormalError::InvalidInput {
                reason: "TransformationNormalFamily joint psi workspace received incompatible block specs"
                    .to_string(),
            }
            .into());
        }
        Ok(Some(Arc::new(TransformationNormalPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        ))))
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        // Route through a mask-aware family clone when an outer-score
        // subsample is active. The cloned family's `effective_weights()`
        // returns `wᵢ · mᵢ` (`mᵢ = 1/πᵢ` on sampled rows, `0` elsewhere),
        // and every CTN assembly site reads weights through that accessor.
        // Each per-row contribution is linear in `wᵢ`, so the workspace's
        // gradient / dense Hessian / matrix-free HVP / diagonal are exact
        // Horvitz-Thompson estimators of the full-data quantities.
        match self.maybe_with_outer_subsample_from_options(options)? {
            Some(masked) => masked.exact_newton_joint_hessian_workspace(block_states, specs),
            None => self.exact_newton_joint_hessian_workspace(block_states, specs),
        }
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.inner_coefficient_hessian_hvp_available(specs) {
            return Err(TransformationNormalError::InvalidInput {
                reason: "TransformationNormalFamily joint psi workspace received incompatible block specs"
                    .to_string(),
            }
            .into());
        }
        // Route through a mask-aware family clone when an outer-score
        // subsample is active. Every CTN ψ assembly site — including the
        // workspace's `compute_all_axes` (per-row reduction near line ~13916)
        // and `compute_pair_cache` (per-row reduction near line ~14263) —
        // reads its row weight via `self.family.effective_weights()`, which
        // on the cloned family returns `wᵢ · mᵢ`. Because each per-row
        // contribution is linear in `wᵢ`, the workspace's per-axis ψ and
        // per-axis-pair ψ-ψ outputs are exact Horvitz-Thompson estimators
        // of the full-data quantities. The persistent dense-Hessian cache
        // and `row_quantity_cache` on the cloned family are fresh, so
        // subsampled builds cannot alias a later full-data probe at the
        // same β.
        let family = match self.maybe_with_outer_subsample_from_options(options)? {
            Some(masked) => masked,
            None => self.clone(),
        };
        Ok(Some(Arc::new(TransformationNormalPsiWorkspace::new(
            family,
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        ))))
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        // CTN's per-axis [`scop_psi_terms`] kernel walks all `n` rows serially
        // and is invoked once per ψ axis. Opting in here amortizes the per-row
        // state load across axes and parallelizes the row walk via the
        // workspace's [`compute_all_axes`] kernel — the dominant outer
        // gradient-evaluation cost at large scale.
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // CTN's SCOP coefficient-space joint Hessian is supplied as a
        // row-streaming matrix-free Hv operator.
        matches!(specs, [spec] if spec.design.ncols()
            == self.response_val_basis.ncols().saturating_mul(self.covariate_design.ncols()))
    }

    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.inner_coefficient_hessian_hvp_available(specs)
    }

    fn outer_hyper_hessian_dense_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Dense materialization remains mathematically available through the
        // outer-HVP operator, but SCOP's primary production path is the
        // matrix-free θθ operator above.
        self.inner_coefficient_hessian_hvp_available(specs)
    }
}
