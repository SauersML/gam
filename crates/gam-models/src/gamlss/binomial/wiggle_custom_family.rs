// Real concern-organized submodule of the gamlss family stack.
// Cross-module items are re-exported flat through the parent (`gamlss.rs`),
// so `use super::*;` makes the sibling-concern symbols this module references
// resolve through the parent namespace.
use super::*;

impl CustomFamily for BinomialLocationScaleWiggleFamily {
    // NO full-span Firth/Jeffreys for THIS family (gam#932). The trait default is
    // OFF (gam#1395 flat-prior exact-Newton objective); plain binomial families
    // opt back in for their separation regime, but the location-scale-WIGGLE
    // variant must NOT, because it carries an EXACT structural gauge null: the
    // threshold β_t and the wiggle-intercept `β_wᵀB(q₀)` both shift `q = q₀ + Bᵀβ_w`
    // additively (documented on `pseudo_logdet_mode` below), so the reduced Fisher
    // information is singular at `σ_min ≈ ridge_floor ≈ 1e-10`. The always-on
    // full-span Firth term floor-inverts that gauge direction into a
    // `1/floor ≈ 1e9` curvature "wall" that couples into the identified threshold
    // block and freezes the constrained joint Newton (the binomial-wiggle+matérn
    // refit stall: a feasible 9e-2 step gutted to ~7e-6 while the threshold
    // carried |g|≈8). The conditioning gate cannot save it — the gate reads the
    // gauge null as ill-conditioning and ARMS the term, mistaking a structural
    // gauge for near-separation. There is no clean fix at the Firth-spectrum
    // layer: the wiggle's monotone modes form a sub-ε continuum, so any hard
    // eigenvalue cutoff flips the kept-set cycle-to-cycle and steps Φ
    // discontinuously (re-collapsing the trust region), while a smooth/floor
    // variant only trades the 1e9 wall for a smaller-but-still-stalling one.
    //
    // The bias-correction term is also UNNECESSARY here: the base penalized
    // Hessian is well-conditioned (the smoothing penalty breaks the gauge,
    // cond≈7e2) and the monotone wiggle is already bounded by its β≥0 inequality
    // constraint, so genuine separation is regularized by the penalty + cone, not
    // by Firth. Disabling the term lets the well-conditioned Newton converge.
    fn joint_jeffreys_term_required(&self) -> bool {
        false
    }

    /// The Binomial location-scale-wiggle joint Hessian depends on β because
    /// it involves the nonlinear link function evaluated at the combined
    /// predictor, which changes with all three coefficient blocks.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: matrix-free workspace applies joint Hv at
        // O(n · (p_t + p_ℓ + p_w)); only fall back to the dense build cost when
        // `use_joint_matrix_free_path` declines the operator path.
        crate::location_scale_engine::location_scale_coefficient_hessian_cost(
            self.y.len() as u64,
            specs,
        )
    }

    /// The wiggle family carries a structural null-space direction: the
    /// threshold β_t and the overall wiggle-intercept combination
    /// `β_w^⊤ B(q₀)` both shift q = q₀ + B^⊤ β_w additively, which makes the
    /// penalized joint Hessian H = H_L + S near-singular along that
    /// direction (σ_min ≈ ridge_floor ≈ 1e-10).  Under the default `Smooth`
    /// regularization this null direction contributes a first-order
    /// component to `d log|H|/dρ` via `φ'(σ_min) · dσ_min/dρ` that cannot
    /// be matched by the analytic `u^⊤ (dH/dρ) u` formula — the
    /// eigenvector `u` for a near-zero σ is numerically arbitrary inside
    /// the null space, so first-order perturbation theory breaks down.
    /// `HardPseudo` excludes σ ≤ ε from BOTH log|H| and its gradient
    /// consistently, so the null direction drops out of the analytic geometry.
    fn pseudo_logdet_mode(&self) -> crate::custom_family::PseudoLogdetMode {
        crate::custom_family::PseudoLogdetMode::HardPseudo
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(None);
        }
        Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()))
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx != Self::BLOCK_WIGGLE {
            return Ok(beta);
        }
        let beta = project_monotone_wiggle_beta_nonnegative(beta);
        validate_monotone_wiggle_beta_nonnegative(
            &beta,
            "BinomialLocationScaleWiggleFamily post-update",
        )?;
        Ok(beta)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;

        let core = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let wiggle_design = self.wiggle_design(core.q0.view())?;
        let dq_dq0 =
            self.wiggle_dq_dq0(core.q0.view(), block_states[Self::BLOCK_WIGGLE].beta.view())?;
        let threshold_design = self.threshold_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing threshold design"
                .to_string()
        })?;
        let log_sigma_design = self.log_sigma_design.as_ref().ok_or_else(|| {
            "BinomialLocationScaleWiggleFamily exact-newton path is missing log-sigma design"
                .to_string()
        })?;

        // Per-block gradients from the eta-space score.
        //
        //   q = q0 + w(q0), a = dq/dq0
        //   score_q = -m1   (m1 = dF/dq, F = -ℓ)
        //   grad_eta_t[i]  = score_q * a * q0_t
        //   grad_eta_ls[i] = score_q * a * q0_ls
        //   grad_q[i]      = score_q          (wiggle basis acts on q)
        let mut grad_eta_t = Array1::<f64>::zeros(n);
        let mut grad_eta_ls = Array1::<f64>::zeros(n);
        let mut grad_q = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q_i = core.q0[i] + etaw[i];
            let (m1, _, _) = binomial_neglog_q_derivatives_dispatch(
                self.y[i],
                self.weights[i],
                q_i,
                core.mu[i],
                core.dmu_dq[i],
                core.d2mu_dq2[i],
                core.d3mu_dq3[i],
                &self.link_kind,
            );
            let score_q = -m1;
            let q0d = nonwiggle_q_derivs(eta_t[i], core.sigma[i]);
            grad_eta_t[i] = score_q * dq_dq0[i] * q0d.q_t;
            grad_eta_ls[i] = score_q * dq_dq0[i] * q0d.q_ls;
            grad_q[i] = score_q;
        }
        let grad_t = threshold_design.transpose_vector_multiply(&grad_eta_t);
        let grad_ls = log_sigma_design.transpose_vector_multiply(&grad_eta_ls);
        let grad_w = fast_atv(&wiggle_design, &grad_q);

        // Per-block diagonal Hessians without ever materializing the full p×p
        // joint matrix. The shared row-pieces struct exposes block diagonals
        // directly, so the cross blocks (h_tl, h_tw, h_lw) are not formed.
        let (x_t, x_ls) = self
            .exact_joint_dense_block_designs(None)?
            .ok_or("BinomialLocationScaleWiggleFamily: joint block designs unavailable")?;
        let pieces = self.wiggle_order2_rows(block_states)?;
        let (h_tt, h_ll, h_ww) = pieces.assemble_block_diagonals(&x_t, &x_ls)?;
        Ok(FamilyEvaluation {
            log_likelihood: core.log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(h_tt),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_ls,
                    hessian: SymmetricMatrix::Dense(h_ll),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_w,
                    hessian: SymmetricMatrix::Dense(h_ww),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let (_n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;
        binomial_location_scale_ll_only(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )
    }

    /// Outer-only log-likelihood with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `Some`, only the sampled rows
    /// contribute; each row's per-row log-likelihood term is multiplied by
    /// `WeightedOuterRow.weight`, the Horvitz–Thompson inverse-inclusion
    /// factor 1/π_i (uniform or stratified sampling both supported), so the
    /// partial sum is an unbiased estimator of the full-data log-likelihood.
    /// When `None`, this returns the full-data `log_likelihood_only`. Inner
    /// PIRLS line searches never install the subsample option, so they
    /// continue to score the exact full-data log-likelihood.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        let (_n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;
        let link_kind = &self.link_kind;
        let rows = &subsample.rows;
        let ll = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            rows.len(),
            |range| -> Result<f64, String> {
                let mut acc = 0.0_f64;
                for k in range {
                    let row = &rows[k];
                    let i = row.index;
                    let wi = self.weights[i];
                    if wi == 0.0 {
                        continue;
                    }
                    let SigmaJet1 { sigma, .. } = exp_sigma_jet1_scalar(eta_ls[i]);
                    let q0 = binomial_location_scale_q0(eta_t[i], sigma);
                    let q = q0 + etaw[i];
                    let mu = if matches!(link_kind, InverseLink::Standard(StandardLink::Probit)) {
                        0.5
                    } else {
                        let jet = inverse_link_jet_for_inverse_link(link_kind, q).map_err(|e| {
                            format!("location-scale inverse-link evaluation failed: {e}")
                        })?;
                        jet.mu
                    };
                    let term =
                        binomial_location_scale_log_likelihood(self.y[i], wi, q, link_kind, mu)?;
                    acc += row.weight * term;
                }
                Ok(acc)
            },
            |a, b| Ok(a + b),
        )?;
        Ok(ll.unwrap_or(0.0))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        validate_block_count::<GamlssError>(
            "BinomialLocationScaleWiggleFamily",
            3,
            block_states.len(),
        )?;
        let (x_t, x_ls) = self.dense_block_designs()?;
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            None,
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let total = pt + pls + pw;

        let (range_start, range_end) = match block_idx {
            Self::BLOCK_T => (0usize, pt),
            Self::BLOCK_LOG_SIGMA => (pt, pt + pls),
            Self::BLOCK_WIGGLE => (pt + pls, total),
            _ => return Ok(None),
        };
        if d_beta.len() != (range_end - range_start) {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} d_beta length mismatch: got {}, expected {}",
                    d_beta.len(),
                    range_end - range_start
                ),
            }
            .into());
        }

        // Block-local exact Newton directional derivative is extracted from the
        // full joint directional Hessian.
        //
        // For the 3-block wiggle model with beta=(beta_t,beta_ls,betaw),
        // define the full negative-loglik Hessian H(beta) in flattened block
        // coordinates. For a direction that moves only one block,
        //
        //   u = [u_t, 0,   0]   or
        //   u = [0,   u_ls,0]   or
        //   u = [0,   0,   uw],
        //
        // the exact blockwise directional Hessian required by the trait is just
        // the corresponding principal block of D H[u]:
        //
        //   D H_block[u_block]
        //   = (D H_joint[u])_{block,block}.
        //
        // This avoids maintaining a second, partially duplicated derivation for
        // the block-local case and keeps the exact-newton block callback aligned
        // with the already-validated joint formulas.
        let mut d_beta_flat = Array1::<f64>::zeros(total);
        match block_idx {
            Self::BLOCK_T => {
                d_beta_flat.slice_mut(s![0..pt]).assign(d_beta);
            }
            Self::BLOCK_LOG_SIGMA => {
                d_beta_flat.slice_mut(s![pt..pt + pls]).assign(d_beta);
            }
            Self::BLOCK_WIGGLE => {
                d_beta_flat.slice_mut(s![pt + pls..]).assign(d_beta);
            }
            _ => {}
        }
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative(block_states, &d_beta_flat)?
            .ok_or_else(|| "missing exact wiggle joint dH".to_string())?;
        let out = d_joint
            .slice(s![range_start..range_end, range_start..range_end])
            .to_owned();
        Ok(Some(out))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact joint Hessian for the 3-block binomial location-scale wiggle family.
        //
        // Model:
        //   q0 = -eta_t / sigma(eta_ls),
        //   q  = q0 + betaw^T B(q0),
        //   mu = Phi(q),
        //   F  = -sum_i ell_i(mu_i).
        //
        // The typed probe lowering evaluates the canonical row expression once
        // per observation and extracts its structured B/B' channels.  Dense
        // and matrix-free paths consume those identical generated rows.
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pieces = self.wiggle_order2_rows(block_states)?;
        Ok(Some(pieces.assemble_dense(&x_t, &x_ls)?))
    }

    fn joint_jeffreys_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_wiggle_information_with_specs(block_states, specs)
    }

    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_wiggle_information_directional_with_specs(block_states, specs, d_beta_flat)
    }

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.expected_wiggle_information_second_directional_with_specs(
            block_states,
            specs,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn joint_jeffreys_information_matches_observed_hessian(&self) -> bool {
        // Expected Fisher information override (gam#1020): observed-Hessian
        // conditioning pre-checks must not skip the expected-information gate.
        false
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Exact directional derivative dH[u] for the same 3-block model.
        //
        // Direction:
        //   u = (u_t, u_l, uw),
        //   d_eta_t = X_t u_t, d_eta_l = X_l u_l.
        //
        // Canonical objective identity for scalar-q composition:
        //   dH_ab[u] =
        //      m3 * dq * q_a q_b
        //    + m2 * (dq_a q_b + q_a dq_b + dq q_ab)
        //    + m1 * dq_ab
        // where (m1,m2,m3) are derivatives of F wrt q.
        //
        // Log-likelihood derivative relation used in code:
        //   s = d ell/dq, c = d² ell/dq², t = d³ ell/dq³
        //   m1 = -s, m2 = -c, m3 = -t.
        //
        // Required analytic chain terms:
        //
        // 1) Wiggle scalars:
        //   m  = 1 + betaw^T B'(q0)
        //   g2 = betaw^T B''(q0)
        //   g3 = betaw^T B'''(q0)
        //
        // 2) Directional wiggle scalars:
        //   dm  = (B'·uw)  + g2*dq0
        //   dg2 = (B''·uw) + g3*dq0
        //
        // 3) Directional q pieces:
        //   dq   = m*dq0 + B·uw
        //   dq_t = dm*q0_t + m*dq0_t
        //   dq_l = dm*q0_l + m*dq0_l
        //
        // 4) Directional second q pieces:
        //   dq_tt = dg2*q0_t*q0_t + g2*(2*q0_t*dq0_t)
        //   dq_tl = dg2*q0_t*q0_l + g2*(dq0_t*q0_l + q0_t*dq0_l)
        //           + dm*q0_tl + m*dq0_tl
        //   dq_ll = dg2*q0_l*q0_l + g2*(2*q0_l*dq0_l)
        //           + dm*q0_ll + m*dq0_ll
        //
        // 5) Mixed w-block directional terms:
        //   qw   = B,         dqw   = B' dq0
        //   q_tw  = q0_t B',   dq_tw  = dq0_t B' + dq0 q0_t B''
        //   q_lw  = q0_l B',   dq_lw  = dq0_l B' + dq0 q0_l B''
        //   qww  = 0,         dqww  = 0
        //
        // Implementation below follows these formulas exactly block-by-block.
        let (n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;

        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_flat, "wiggle joint d_beta")?;
        let d_eta_t = fast_av(&x_t, &u_t);
        let d_eta_ls = fast_av(&x_ls, &u_ls);

        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        if d0.ncols() != betaw0.len() || dd0.ncols() != betaw0.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "wiggle derivative/beta mismatch in exact joint dH: B'={} B''={} betaw={}",
                    d0.ncols(),
                    dd0.ncols(),
                    betaw0.len()
                ),
            }
            .into());
        }
        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let (sigma, ..) = exp_sigma_derivs_up_to_third(eta_ls.view());

        let BinomialWiggleDhRowCoeffs {
            coeff_tt,
            coeff_tl,
            coeff_ll,
            coeff_tw_b,
            coeff_tw_d,
            coeff_tw_dd,
            coeff_lw_b,
            coeff_lw_d,
            coeff_lw_dd,
            coeffww_bb,
            coeffww_db,
        } = self.binomial_wiggle_dh_row_coeffs(
            n,
            &BinomialWiggleDhRowInputs {
                core0: &core0,
                eta_t,
                etaw,
                sigma: &sigma,
                m: &m,
                g2: &g2,
                g3: &g3,
                b0: &b0,
                d0: &d0,
                dd0: &dd0,
                uw: &uw,
                d_eta_t: &d_eta_t,
                d_eta_ls: &d_eta_ls,
            },
        );
        let d_h_tt = xt_diag_x_dense(&x_t, &coeff_tt)?;
        let d_h_tl = xt_diag_y_dense(&x_t, &coeff_tl, &x_ls)?;
        let d_h_ll = xt_diag_x_dense(&x_ls, &coeff_ll)?;
        let d_h_tw = xt_diag_y_dense(&x_t, &coeff_tw_b, &b0)?
            + &xt_diag_y_dense(&x_t, &coeff_tw_d, &d0)?
            + &xt_diag_y_dense(&x_t, &coeff_tw_dd, &dd0)?;
        let d_h_lw = xt_diag_y_dense(&x_ls, &coeff_lw_b, &b0)?
            + &xt_diag_y_dense(&x_ls, &coeff_lw_d, &d0)?
            + &xt_diag_y_dense(&x_ls, &coeff_lw_dd, &dd0)?;
        let mut d_hww = xt_diag_x_dense(&b0, &coeffww_bb)?;
        d_hww += &xt_diag_y_dense(&d0, &coeffww_db, &b0)?;
        d_hww += &xt_diag_y_dense(&b0, &coeffww_db, &d0)?;

        let mut d_h = Array2::<f64>::zeros((total, total));
        d_h.slice_mut(s![0..pt, 0..pt]).assign(&d_h_tt);
        d_h.slice_mut(s![0..pt, pt..pt + pls]).assign(&d_h_tl);
        d_h.slice_mut(s![pt..pt + pls, pt..pt + pls])
            .assign(&d_h_ll);
        d_h.slice_mut(s![0..pt, pt + pls..total]).assign(&d_h_tw);
        d_h.slice_mut(s![pt..pt + pls, pt + pls..total])
            .assign(&d_h_lw);
        d_h.slice_mut(s![pt + pls..total, pt + pls..total])
            .assign(&d_hww);
        mirror_upper_to_lower(&mut d_h);
        Ok(Some(d_h))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let (n, eta_t, eta_ls, etaw) = self.validated_block_etas(block_states)?;

        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(None)? else {
            return Ok(None);
        };
        let pt = x_t.ncols();
        let pls = x_ls.ncols();
        let betaw0 = block_states[Self::BLOCK_WIGGLE].beta.clone();
        let core0 = binomial_location_scale_core(
            &self.y,
            &self.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &self.link_kind,
        )?;
        let b0 = self.wiggle_design(core0.q0.view())?;
        let d0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::first_derivative())?;
        let dd0 =
            self.wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())?;
        let d3_basis = self.wiggle_d3basis_constrained(core0.q0.view())?;
        let d3q = self.wiggle_d3q_dq03(core0.q0.view(), betaw0.view())?;
        let d4q = self.wiggle_d4q_dq04(core0.q0.view(), betaw0.view())?;
        let pw = b0.ncols();
        let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
        let total = beta_layout.total();
        if d0.ncols() != betaw0.len()
            || dd0.ncols() != betaw0.len()
            || d3_basis.ncols() != betaw0.len()
        {
            return Err(GamlssError::DimensionMismatch { reason: format!(
                "wiggle derivative/beta mismatch in exact joint d2H: B'={} B''={} B'''={} betaw={}",
                d0.ncols(),
                dd0.ncols(),
                d3_basis.ncols(),
                betaw0.len()
            ) }.into());
        }

        let (u_t, u_ls, uw) = beta_layout.split_three(d_beta_u_flat, "wiggle joint d_beta_u")?;
        let (v_t, v_ls, vw) = beta_layout.split_three(d_betav_flat, "wiggle joint d_betav")?;
        let d_eta_t_u = fast_av(&x_t, &u_t);
        let d_eta_ls_u = fast_av(&x_ls, &u_ls);
        let d_eta_tv = fast_av(&x_t, &v_t);
        let d_eta_lsv = fast_av(&x_ls, &v_ls);

        let m = d0.dot(&betaw0) + 1.0;
        let g2 = dd0.dot(&betaw0);
        let g3 = d3q;
        let g4 = d4q;
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_array(eta_ls.view());

        let mut d2_h: Array2<f64> = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = Array2::<f64>::zeros((total, total));
                for i in range {
                    let mut row_h = Array2::<f64>::zeros((total, total));
                    // Per-row scalar objective derivatives for F_i(q).
                    let q_i = core0.q0[i] + etaw[i];
                    let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                        self.y[i],
                        self.weights[i],
                        q_i,
                        core0.mu[i],
                        core0.dmu_dq[i],
                        core0.d2mu_dq2[i],
                        core0.d3mu_dq3[i],
                        &self.link_kind,
                    );
                    let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                        self.y[i],
                        self.weights[i],
                        q_i,
                        core0.mu[i],
                        core0.dmu_dq[i],
                        core0.d2mu_dq2[i],
                        core0.d3mu_dq3[i],
                        &self.link_kind,
                    )?;

                    // Non-wiggle q0(eta_t, eta_ls) derivatives and sigma-ratio helpers.
                    let q0 = nonwiggle_q_derivs(eta_t[i], sigma[i]);
                    let s_safe = sigma[i];
                    let s2 = s_safe * s_safe;
                    let s3 = s2 * s_safe;
                    let s4 = s3 * s_safe;
                    let s5 = s4 * s_safe;
                    let q0_tl_ls_ls =
                        d3s[i] / s2 - 6.0 * ds[i] * d2s[i] / s3 + 6.0 * ds[i] * ds[i] * ds[i] / s4;
                    let q0_tl_ls_ls_ls =
                        d4s[i] / s2 - 8.0 * ds[i] * d3s[i] / s3 - 6.0 * d2s[i] * d2s[i] / s3
                            + 36.0 * ds[i] * ds[i] * d2s[i] / s4
                            - 24.0 * ds[i] * ds[i] * ds[i] * ds[i] / s5;
                    let q0_ll_ls_ls = eta_t[i] * q0_tl_ls_ls_ls;

                    let u_t_i = d_eta_t_u[i];
                    let u_ls_i = d_eta_ls_u[i];
                    let v_t_i = d_eta_tv[i];
                    let v_ls_i = d_eta_lsv[i];

                    // Directional z=q0 primitives for u and v.
                    let dq0_u = q0.q_t * u_t_i + q0.q_ls * u_ls_i;
                    let dq0v = q0.q_t * v_t_i + q0.q_ls * v_ls_i;
                    let d2q0_uv =
                        q0.q_tl * (u_t_i * v_ls_i + v_t_i * u_ls_i) + q0.q_ll * u_ls_i * v_ls_i;

                    let dq0_t_u = q0.q_tl * u_ls_i;
                    let dq0_tv = q0.q_tl * v_ls_i;
                    let dq0_ls_u = q0.q_tl * u_t_i + q0.q_ll * u_ls_i;
                    let dq0_lsv = q0.q_tl * v_t_i + q0.q_ll * v_ls_i;
                    let dq0_tl_u = q0.q_tl_ls * u_ls_i;
                    let dq0_tlv = q0.q_tl_ls * v_ls_i;
                    let dq0_ll_u = q0.q_tl_ls * u_t_i + q0.q_ll_ls * u_ls_i;
                    let dq0_llv = q0.q_tl_ls * v_t_i + q0.q_ll_ls * v_ls_i;

                    let d2q0_t_uv = q0.q_tl_ls * u_ls_i * v_ls_i;
                    let d2q0_ls_uv = q0.q_tl_ls * (u_ls_i * v_t_i + v_ls_i * u_t_i)
                        + q0.q_ll_ls * u_ls_i * v_ls_i;
                    let d2q0_tl_uv = q0_tl_ls_ls * u_ls_i * v_ls_i;
                    let d2q0_ll_uv = q0_tl_ls_ls * (u_t_i * v_ls_i + v_t_i * u_ls_i)
                        + q0_ll_ls_ls * u_ls_i * v_ls_i;

                    let br = b0.row(i);
                    let dr = d0.row(i);
                    let ddr = dd0.row(i);
                    let d3r = d3_basis.row(i);
                    let b_u = br.dot(&uw);
                    let bv = br.dot(&vw);
                    let b1_u = dr.dot(&uw);
                    let b1v = dr.dot(&vw);
                    let b2_u = ddr.dot(&uw);
                    let b2v = ddr.dot(&vw);
                    let b3_u = d3r.dot(&uw);
                    let b3v = d3r.dot(&vw);

                    // Wiggle scalar chain terms:
                    //   m = 1 + g1,     g2 = betaw^T B''(q0),
                    //   dm[u]   = B'·uw + g2*dq0[u],
                    //   d2m[u,v]= g3*dq0[u]dq0[v] + g2*d2q0[u,v] + (B''·vw)dq0[u] + (B''·uw)dq0[v],
                    //   dg2[u]  = B''·uw + g3*dq0[u],
                    //   d2g2[u,v]=g4*dq0[u]dq0[v] + g3*d2q0[u,v] + (B'''·vw)dq0[u] + (B'''·uw)dq0[v].
                    let dm_u = b1_u + g2[i] * dq0_u;
                    let dmv = b1v + g2[i] * dq0v;
                    let d2m_uv = g3[i] * dq0_u * dq0v + g2[i] * d2q0_uv + b2v * dq0_u + b2_u * dq0v;
                    let dg2_u = b2_u + g3[i] * dq0_u;
                    let dg2v = b2v + g3[i] * dq0v;
                    let d2g2_uv =
                        g4[i] * dq0_u * dq0v + g3[i] * d2q0_uv + b3v * dq0_u + b3_u * dq0v;

                    // First/second directional terms for total q.
                    let dq_u = m[i] * dq0_u + b_u;
                    let dqv = m[i] * dq0v + bv;
                    // Simplify exact formula for q = q0 + betaw^T B(q0):
                    //   D²q[u,v] = m*d²q0 + g2*dq0[u]dq0[v] + (B'·uw)dq0[v] + (B'·vw)dq0[u].
                    let d2q_uv = m[i] * d2q0_uv + g2[i] * dq0_u * dq0v + b1_u * dq0v + b1v * dq0_u;

                    // q partials by block and their first/second directional derivatives.
                    let q_t = m[i] * q0.q_t;
                    let q_ls = m[i] * q0.q_ls;
                    let q_tt = g2[i] * q0.q_t * q0.q_t;
                    let q_tl = g2[i] * q0.q_t * q0.q_ls + m[i] * q0.q_tl;
                    let q_ll = g2[i] * q0.q_ls * q0.q_ls + m[i] * q0.q_ll;

                    let dq_t_u = dm_u * q0.q_t + m[i] * dq0_t_u;
                    let dq_tv = dmv * q0.q_t + m[i] * dq0_tv;
                    let dq_ls_u = dm_u * q0.q_ls + m[i] * dq0_ls_u;
                    let dq_lsv = dmv * q0.q_ls + m[i] * dq0_lsv;

                    let d2q_t_uv =
                        d2m_uv * q0.q_t + dm_u * dq0_tv + dmv * dq0_t_u + m[i] * d2q0_t_uv;
                    let d2q_ls_uv =
                        d2m_uv * q0.q_ls + dm_u * dq0_lsv + dmv * dq0_ls_u + m[i] * d2q0_ls_uv;

                    let dq_tt_u = dg2_u * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_t_u);
                    let dq_ttv = dg2v * q0.q_t * q0.q_t + g2[i] * (2.0 * q0.q_t * dq0_tv);
                    let d2q_tt_uv = d2g2_uv * q0.q_t * q0.q_t
                        + dg2_u * (2.0 * q0.q_t * dq0_tv)
                        + dg2v * (2.0 * q0.q_t * dq0_t_u)
                        + g2[i] * (2.0 * dq0_t_u * dq0_tv + 2.0 * q0.q_t * d2q0_t_uv);

                    let dq_tl_u = dg2_u * q0.q_t * q0.q_ls
                        + g2[i] * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                        + dm_u * q0.q_tl
                        + m[i] * dq0_tl_u;
                    let dq_tlv = dg2v * q0.q_t * q0.q_ls
                        + g2[i] * (dq0_tv * q0.q_ls + q0.q_t * dq0_lsv)
                        + dmv * q0.q_tl
                        + m[i] * dq0_tlv;
                    let d2q_tl_uv = d2g2_uv * q0.q_t * q0.q_ls
                        + dg2_u * (dq0_tv * q0.q_ls + q0.q_t * dq0_lsv)
                        + dg2v * (dq0_t_u * q0.q_ls + q0.q_t * dq0_ls_u)
                        + g2[i]
                            * (d2q0_t_uv * q0.q_ls
                                + dq0_t_u * dq0_lsv
                                + dq0_tv * dq0_ls_u
                                + q0.q_t * d2q0_ls_uv)
                        + d2m_uv * q0.q_tl
                        + dm_u * dq0_tlv
                        + dmv * dq0_tl_u
                        + m[i] * d2q0_tl_uv;

                    let dq_ll_u = dg2_u * q0.q_ls * q0.q_ls
                        + g2[i] * (2.0 * q0.q_ls * dq0_ls_u)
                        + dm_u * q0.q_ll
                        + m[i] * dq0_ll_u;
                    let dq_llv = dg2v * q0.q_ls * q0.q_ls
                        + g2[i] * (2.0 * q0.q_ls * dq0_lsv)
                        + dmv * q0.q_ll
                        + m[i] * dq0_llv;
                    let d2q_ll_uv = d2g2_uv * q0.q_ls * q0.q_ls
                        + dg2_u * (2.0 * q0.q_ls * dq0_lsv)
                        + dg2v * (2.0 * q0.q_ls * dq0_ls_u)
                        + g2[i] * (2.0 * dq0_ls_u * dq0_lsv + 2.0 * q0.q_ls * d2q0_ls_uv)
                        + d2m_uv * q0.q_ll
                        + dm_u * dq0_llv
                        + dmv * dq0_ll_u
                        + m[i] * d2q0_ll_uv;

                    // Exact second directional coefficients for the scalar block weights.
                    let coeff_tt = second_directionalhessian_coeff_fromobjective_q_terms(
                        m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_t, q_tt, dq_t_u, dq_tv, dq_t_u,
                        dq_tv, d2q_t_uv, d2q_t_uv, dq_tt_u, dq_ttv, d2q_tt_uv,
                    );
                    let coeff_tl = second_directionalhessian_coeff_fromobjective_q_terms(
                        m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, q_ls, q_tl, dq_t_u, dq_tv, dq_ls_u,
                        dq_lsv, d2q_t_uv, d2q_ls_uv, dq_tl_u, dq_tlv, d2q_tl_uv,
                    );
                    let coeff_ll = second_directionalhessian_coeff_fromobjective_q_terms(
                        m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, q_ls, q_ll, dq_ls_u, dq_lsv,
                        dq_ls_u, dq_lsv, d2q_ls_uv, d2q_ls_uv, dq_ll_u, dq_llv, d2q_ll_uv,
                    );

                    let xtr = x_t.row(i);
                    let xlsr = x_ls.row(i);
                    for a_idx in 0..pt {
                        for b_idx in a_idx..pt {
                            row_h[[a_idx, b_idx]] += coeff_tt * xtr[a_idx] * xtr[b_idx];
                        }
                    }
                    for a_idx in 0..pt {
                        for b_idx in 0..pls {
                            row_h[[a_idx, pt + b_idx]] += coeff_tl * xtr[a_idx] * xlsr[b_idx];
                        }
                    }
                    for a_idx in 0..pls {
                        for b_idx in a_idx..pls {
                            row_h[[pt + a_idx, pt + b_idx]] += coeff_ll * xlsr[a_idx] * xlsr[b_idx];
                        }
                    }

                    for j in 0..pw {
                        let qw = br[j];
                        let dqw_u = dr[j] * dq0_u;
                        let dqwv = dr[j] * dq0v;
                        let d2qw_uv = ddr[j] * dq0_u * dq0v + dr[j] * d2q0_uv;
                        let q_tw = dr[j] * q0.q_t;
                        let q_lw = dr[j] * q0.q_ls;
                        let dq_tw_u = ddr[j] * dq0_u * q0.q_t + dr[j] * dq0_t_u;
                        let dq_twv = ddr[j] * dq0v * q0.q_t + dr[j] * dq0_tv;
                        let d2q_tw_uv = d3r[j] * dq0_u * dq0v * q0.q_t
                            + ddr[j] * (d2q0_uv * q0.q_t + dq0_u * dq0_tv + dq0v * dq0_t_u)
                            + dr[j] * d2q0_t_uv;
                        let dq_lw_u = ddr[j] * dq0_u * q0.q_ls + dr[j] * dq0_ls_u;
                        let dq_lwv = ddr[j] * dq0v * q0.q_ls + dr[j] * dq0_lsv;
                        let d2q_lw_uv = d3r[j] * dq0_u * dq0v * q0.q_ls
                            + ddr[j] * (d2q0_uv * q0.q_ls + dq0_u * dq0_lsv + dq0v * dq0_ls_u)
                            + dr[j] * d2q0_ls_uv;

                        let coeff_tw = second_directionalhessian_coeff_fromobjective_q_terms(
                            m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, qw, q_tw, dq_t_u, dq_tv, dqw_u,
                            dqwv, d2q_t_uv, d2qw_uv, dq_tw_u, dq_twv, d2q_tw_uv,
                        );
                        let coeff_lw = second_directionalhessian_coeff_fromobjective_q_terms(
                            m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_ls, qw, q_lw, dq_ls_u, dq_lsv,
                            dqw_u, dqwv, d2q_ls_uv, d2qw_uv, dq_lw_u, dq_lwv, d2q_lw_uv,
                        );

                        for a_idx in 0..pt {
                            row_h[[a_idx, pt + pls + j]] += coeff_tw * xtr[a_idx];
                        }
                        for a_idx in 0..pls {
                            row_h[[pt + a_idx, pt + pls + j]] += coeff_lw * xlsr[a_idx];
                        }
                    }

                    for j in 0..pw {
                        let qwj = br[j];
                        let dqwj_u = dr[j] * dq0_u;
                        let dqwjv = dr[j] * dq0v;
                        let d2qwj_uv = ddr[j] * dq0_u * dq0v + dr[j] * d2q0_uv;
                        for k in j..pw {
                            let qwk = br[k];
                            let dqwk_u = dr[k] * dq0_u;
                            let dqwkv = dr[k] * dq0v;
                            let d2qwk_uv = ddr[k] * dq0_u * dq0v + dr[k] * d2q0_uv;
                            let coeffww = second_directionalhessian_coeff_fromobjective_q_terms(
                                m1, m2, m3, m4, dq_u, dqv, d2q_uv, qwj, qwk, 0.0, dqwj_u, dqwjv,
                                dqwk_u, dqwkv, d2qwj_uv, d2qwk_uv, 0.0, 0.0, 0.0,
                            );
                            row_h[[pt + pls + j, pt + pls + k]] += coeffww;
                        }
                    }

                    acc += &row_h;
                }
                Ok(acc)
            },
            |mut a, b| {
                a += &b;
                Ok(a)
            },
        )?
        .unwrap_or_else(|| Array2::<f64>::zeros((total, total)));

        mirror_upper_to_lower(&mut d2_h);
        Ok(Some(d2_h))
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessian(block_states)
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some(shadow) = self.shadow_with_exact_joint_designs(specs)? else {
            return Ok(None);
        };
        shadow.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiTerms>, String> {
        // These three joint psi hooks are the wiggle family's exact
        // likelihood-side contribution to the unified full [rho, psi] outer
        // Hessian:
        //
        //   exact_newton_joint_psi_terms(...)                    -> D_a, D_{beta a}, D_{beta beta a}
        //   exact_newton_joint_psisecond_order_terms(...)       -> D_ab, D_{beta ab}, D_{beta beta ab}
        //   exact_newton_joint_psihessian_directional_derivative(...) -> T_a[u]
        //
        // Generic exact-joint code in custom_family.rs adds all realized
        // penalty motion S_a / S_ab and combines these likelihood-only objects
        // with the joint mode solves beta_i, beta_ij and the total Hessian
        // drifts dot H_i, ddot H_ij. Keeping this contract explicit is what
        // makes the wiggle family's full [rho, psi] Hessian real rather than a
        // gradient-only or block-local surrogate.
        self.exact_newton_joint_psi_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<gam_problem::ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.exact_newton_joint_psisecond_order_terms_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_i,
            psi_j,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_psihessian_directional_derivative_for_specs(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if !self.exact_joint_supported() {
            return Ok(None);
        }
        Ok(Some(Arc::new(
            BinomialLocationScaleWiggleExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs,
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &crate::custom_family::ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if spec.name != "wiggle" {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        if block_states.len() < 2 {
            return Err(GamlssError::UnsupportedConfiguration {
                reason: "wiggle geometry requires threshold and log-sigma blocks".to_string(),
            }
            .into());
        }
        let eta_t = &block_states[Self::BLOCK_T].eta;
        let eta_ls = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        if eta_t.len() != self.y.len() || eta_ls.len() != self.y.len() {
            return Err(GamlssError::DimensionMismatch {
                reason: "wiggle geometry input size mismatch".to_string(),
            }
            .into());
        }
        let mut q0 = Array1::<f64>::zeros(eta_t.len());
        for i in 0..q0.len() {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            q0[i] = binomial_location_scale_q0(eta_t[i], sigma);
        }
        let x = self.wiggle_design(q0.view())?;
        if x.ncols() != spec.design.ncols() {
            return Err(GamlssError::DimensionMismatch {
                reason: format!(
                    "dynamic wiggle design col mismatch: got {}, expected {}",
                    x.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let nrows = x.nrows();
        Ok((
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x)),
            Array1::zeros(nrows),
        ))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let workspace = BinomialLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t.into_owned(),
            x_ls.into_owned(),
        )?;
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-aware joint-Hessian workspace with optional row subsample.
    ///
    /// When `options.outer_score_subsample` is `None`, this is byte-identical
    /// to `exact_newton_joint_hessian_workspace`. When `Some`, the precomputed
    /// per-row coefficient arrays in `pieces` (`coeff_tt`, `coeff_tl`,
    /// `coeff_ll`, `coeff_tw_b`, `coeff_tw_d`, `coeff_lw_b`, `coeff_lw_d`,
    /// `coeffww`) — which every downstream assembly (`hessian_dense`,
    /// `hessian_matvec`, `hessian_diagonal`) consumes row-linearly via
    /// `Xᵀ diag(W) Y` — are replaced by a Horvitz–Thompson mask: each sampled
    /// row's coefficient is multiplied by `WeightedOuterRow.weight` (the
    /// inverse-inclusion factor 1/π_i; uniform or stratified sampling both
    /// supported), and non-sampled rows are zeroed. The resulting joint
    /// Hessian is an unbiased estimator of the full-data joint Hessian.
    /// Inner PIRLS never installs the option, so the inner solve continues
    /// to consume the exact full-data Hessian.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let Some((x_t, x_ls)) = self.exact_joint_dense_block_designs(Some(specs))? else {
            return Ok(None);
        };
        let mut workspace = BinomialLocationScaleWiggleHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            x_t.into_owned(),
            x_ls.into_owned(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    /// Outer-derivative policy: declare HT-subsample capability.
    ///
    /// BinomialLocationScaleWiggleFamily overrides
    /// `log_likelihood_only_with_options` and
    /// `exact_newton_joint_hessian_workspace_with_options` to consume
    /// `options.outer_score_subsample` with per-row Horvitz–Thompson weights
    /// (each sampled row's contribution is multiplied by
    /// `WeightedOuterRow.weight = 1/π_i`; non-sampled rows are zeroed),
    /// yielding unbiased estimators of the full-data log-likelihood and
    /// joint Hessian. The ψ-workspace path is not yet subsample-aware: it
    /// builds the exact full-data ψ Hessian blocks, which are trivially
    /// unbiased; so the outer-score components are a sum of HT-unbiased and
    /// exact-unbiased pieces and the total remains an unbiased estimator of
    /// the full-data outer score. Inner-PIRLS and final-covariance paths
    /// never install the option, so they continue to consume the exact
    /// full-data quantities.
    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        // Same gating as the workspace impl: matrix-free path is available
        // when both threshold and log-σ block designs are present (the
        // wiggle block is folded into the per-row pieces inside
        // `BinomialLocationScaleWiggleHessianWorkspace`). This advertises
        // β-space representation support only.
        self.exact_joint_supported()
            && matches!(
                self.exact_joint_dense_block_designs(Some(specs)),
                Ok(Some(_))
            )
    }
}
