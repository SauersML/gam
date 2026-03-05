use super::*;
use crate::types::{InverseLink, SasLinkState};

impl<'a> RemlState<'a> {
    pub(super) fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
        // Keep expensive diagnostics out of the hot path unless they can
        // be surfaced. This has zero effect on optimization math.
        (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
            && (eval_idx == 1 || eval_idx % 200 == 0)
    }

    pub(super) fn log_gam_cost(
        &self,
        rho: &Array1<f64>,
        lambdas: &[f64],
        laml: f64,
        stab_cond: f64,
        raw_cond: f64,
        edf: f64,
        trace_h_inv_s_lambda: f64,
    ) {
        const GAM_REPEAT_EMIT: u64 = 50;
        const GAM_MIN_EMIT_GAP: u64 = 200;
        let rho_q = quantize_vec(rho.as_slice().unwrap_or_default(), 5e-3, 1e-6);
        let smooth_q = quantize_vec(lambdas, 5e-3, 1e-6);
        let stab_q = quantize_value(stab_cond, 5e-3, 1e-6);
        let raw_q = quantize_value(raw_cond, 5e-3, 1e-6);
        let key = CostKey::new(&rho_q, &smooth_q, stab_q, raw_q);

        let mut last_opt = self.arena.cost_last.write().unwrap();
        let mut repeat = self.arena.cost_repeat.write().unwrap();
        let mut last_emit = self.arena.cost_last_emit.write().unwrap();
        let eval_idx = *self.arena.cost_eval_count.read().unwrap();

        if let Some(last) = last_opt.as_mut() {
            if last.key.approx_eq(&key) {
                last.update(laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
                *repeat += 1;
                if *repeat >= GAM_REPEAT_EMIT
                    && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP
                {
                    println!("[GAM COST] {}", last.format_summary());
                    *repeat = 0;
                    *last_emit = eval_idx;
                }
                return;
            }

            let emit_prev =
                last.count > 1 && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP;
            if emit_prev {
                println!("[GAM COST] {}", last.format_summary());
                *last_emit = eval_idx;
            }
        }

        let new_agg = CostAgg::new(key, laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
        if eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP {
            println!("[GAM COST] {}", new_agg.format_summary());
            *last_emit = eval_idx;
        }
        *last_opt = Some(new_agg);
        *repeat = 0;
    }

    fn invalidate_link_dependent_state(&self) {
        self.cache_manager.clear_eval_and_factor_caches();
        self.cache_manager.pirls_cache.write().unwrap().clear();
        self.warm_start_beta.write().unwrap().take();
    }

    pub(crate) fn set_link_states(
        &mut self,
        mixture_link_state: Option<crate::types::MixtureLinkState>,
        sas_link_state: Option<SasLinkState>,
    ) {
        let changed = self.runtime_mixture_link_state != mixture_link_state
            || self.runtime_sas_link_state != sas_link_state;
        if !changed {
            return;
        }
        self.runtime_mixture_link_state = mixture_link_state;
        self.runtime_sas_link_state = sas_link_state;
        self.invalidate_link_dependent_state();
    }

    /// Compute soft prior cost without needing workspace
    pub(super) fn compute_soft_prior_cost(&self, rho: &Array1<f64>) -> f64 {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return 0.0;
        }

        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        let mut cost = 0.0;
        for &ri in rho.iter() {
            let scaled = sharp * ri * inv_bound;
            cost += scaled.cosh().ln();
        }

        cost * RHO_SOFT_PRIOR_WEIGHT
    }

    /// Compute soft prior gradient without workspace mutation.
    pub(super) fn compute_soft_prior_grad(&self, rho: &Array1<f64>) -> Array1<f64> {
        let len = rho.len();
        let mut grad = Array1::<f64>::zeros(len);
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return grad;
        }
        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        for (g, &ri) in grad.iter_mut().zip(rho.iter()) {
            let scaled = sharp * ri * inv_bound;
            *g = sharp * inv_bound * scaled.tanh() * RHO_SOFT_PRIOR_WEIGHT;
        }
        grad
    }

    /// Add the exact Hessian of the soft rho prior in place.
    ///
    /// Prior definition per coordinate:
    ///   C_i(rho_i) = w * log(cosh(a * rho_i)),
    ///   a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND,
    ///   w = RHO_SOFT_PRIOR_WEIGHT.
    ///
    /// Then:
    ///   dC_i/drho_i   = w * a * tanh(a * rho_i),
    ///   d²C_i/drho_i² = w * a² * sech²(a * rho_i)
    ///                = w * a² * (1 - tanh²(a * rho_i)).
    ///
    /// The prior is separable across coordinates, so off-diagonals are zero.
    pub(super) fn add_soft_prior_hessian_in_place(
        &self,
        rho: &Array1<f64>,
        hess: &mut Array2<f64>,
    ) {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return;
        }
        let a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND;
        let prefactor = RHO_SOFT_PRIOR_WEIGHT * a * a;
        for i in 0..len {
            let t = (a * rho[i]).tanh();
            hess[[i, i]] += prefactor * (1.0 - t * t);
        }
    }

    /// Returns the effective Hessian and the ridge value used (if any).
    /// Uses the same Hessian matrix in both cost and gradient calculations.
    ///
    /// PIRLS folds any stabilization ridge directly into the penalized objective:
    ///   l_p(β; ρ) = l(β) - 0.5 * βᵀ (S_λ + ridge I) β.
    /// Therefore the curvature used in LAML is
    ///   H_eff = X'WX + S_λ + ridge I,
    /// and adding another ridge here places the Laplace expansion on a different surface.
    pub(super) fn effective_hessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        let base = pr.stabilized_hessian_transformed.clone();

        if base.cholesky(Side::Lower).is_ok() {
            return Ok((base, pr.ridge_passport));
        }

        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }

    pub(crate) fn new_with_offset<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        s_list: Vec<Array2<f64>>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        Self::new_with_offset_shared(
            y,
            x,
            weights,
            offset,
            Arc::new(s_list),
            p,
            config,
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
        )
    }

    pub(crate) fn new_with_offset_shared<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        s_list: Arc<Vec<Array2<f64>>>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        // Pre-compute penalty square roots once
        let rs_list = compute_penalty_square_roots(&s_list)?;
        let x = x.into();

        let expected_len = s_list.len();
        let nullspace_dims = match nullspace_dims {
            Some(dims) => {
                if dims.len() != expected_len {
                    return Err(EstimationError::InvalidInput(format!(
                        "nullspace_dims length {} does not match penalties {}",
                        dims.len(),
                        expected_len
                    )));
                }
                dims
            }
            None => vec![0; expected_len],
        };

        let penalty_count = rs_list.len();
        let workspace = RemlWorkspace::new(penalty_count);

        let balanced_penalty_root = create_balanced_penalty_root(&s_list, p)?;
        let reparam_invariant = precompute_reparam_invariant(&rs_list, p)?;
        let sparse_penalty_blocks = build_sparse_penalty_blocks(&s_list, &rs_list)?.map(Arc::new);

        Ok(Self {
            y,
            x,
            weights,
            offset: offset.to_owned(),
            s_full_list: s_list,
            rs_list,
            balanced_penalty_root,
            reparam_invariant,
            sparse_penalty_blocks,
            p,
            config,
            runtime_mixture_link_state: config.link_kind.mixture_state().cloned(),
            runtime_sas_link_state: config.link_kind.sas_state().copied(),
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(workspace),
            warm_start_beta: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
        })
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure consistency in caching.
    pub(super) fn rho_key_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
        EvalCacheManager::sanitized_rho_key(rho)
    }

    pub(super) fn prepare_eval_bundle_with_key(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let decision = self.select_reml_geometry(rho);
        match decision.geometry {
            RemlGeometry::SparseExactSpd => {
                match self.prepare_sparse_eval_bundle_with_key(rho, key.clone()) {
                    Ok(bundle) => {
                        log::info!(
                            "[reml-geometry] sparse_exact_spd reason={} p={} nnz_x={} nnz_h_est={} density_h_est={}",
                            decision.reason,
                            decision.p,
                            decision.nnz_x,
                            decision
                                .nnz_h_upper_est
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "na".to_string()),
                            decision
                                .density_h_upper_est
                                .map(|v| format!("{v:.4}"))
                                .unwrap_or_else(|| "na".to_string()),
                        );
                        Ok(bundle)
                    }
                    Err(err) => {
                        log::warn!(
                            "[reml-geometry] sparse_exact_spd failed ({}); falling back to dense spectral",
                            err
                        );
                        self.prepare_dense_eval_bundle_with_key(rho, key)
                    }
                }
            }
            RemlGeometry::DenseSpectral => self.prepare_dense_eval_bundle_with_key(rho, key),
        }
    }

    pub(super) fn obtain_eval_bundle(
        &self,
        rho: &Array1<f64>,
    ) -> Result<EvalShared, EstimationError> {
        let key = self.rho_key_sanitized(rho);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundle_with_key(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    pub(crate) fn objective_inner_hessian(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        Ok(bundle.h_total.as_ref().clone())
    }

    pub(crate) fn pirls_result_and_hpos_for_rho(
        &self,
        rho: &Array1<f64>,
    ) -> Result<(Arc<crate::pirls::PirlsResult>, Arc<Array2<f64>>), EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        Ok((bundle.pirls_result.clone(), bundle.h_pos_factor_w.clone()))
    }

    pub(super) fn edf_from_h_and_e(
        &self,
        e_transformed: &Array2<f64>, // rank x p_eff
        lambdas: ArrayView1<'_, f64>,
        h_eff: &Array2<f64>, // p_eff x p_eff
    ) -> Result<f64, EstimationError> {
        // Why caching by ρ is sound:
        // The effective degrees of freedom (EDF) calculation is one of only two places where
        // we ask for a Faer factorization through `get_faer_factor`.  The cache inside that
        // helper uses only the vector of log smoothing parameters (ρ) as the key.  At first
        // glance that can look risky—two different Hessians with the same ρ might appear to be
        // conflated.  The surrounding call graph prevents that situation:
        //   • Identity / Gaussian models call `edf_from_h_and_rk` with the stabilized Hessian
        //     `pirls_result.stabilized_hessian_transformed`.
        //   • Non-Gaussian (logit / LAML) models call it with the effective / ridged Hessian
        //     returned by `effective_hessian(pr)`.
        // Within a given `RemlState` we never switch between those two flavours—the state is
        // constructed for a single link function, so the cost/gradient pathways stay aligned.
        // Because of that design, a given ρ vector corresponds to exactly one Hessian type in
        // practice, and the cache cannot hand back a factorization of an unintended matrix.

        // Prefer an un-ridged factorization when the stabilized Hessian is already PD.
        // Only fall back to the RidgePlanner path if direct factorization fails.
        let rho_like = lambdas.mapv(|lam| lam.ln());
        let factor = {
            let h_view = FaerArrayView::new(h_eff);
            if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                Arc::new(FaerFactor::Llt(f))
            } else if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                Arc::new(FaerFactor::Ldlt(f))
            } else {
                self.get_faer_factor(&rho_like, h_eff)
            }
        };

        // Use the single λ-weighted penalty root E for S_λ = Eᵀ E to compute
        // trace(H⁻¹ S_λ) = ⟨H⁻¹ Eᵀ, Eᵀ⟩_F directly.
        let e_t = e_transformed.t().to_owned(); // (p_eff × rank_total)
        let e_view = FaerArrayView::new(&e_t);
        let x = factor.solve(e_view.as_ref());
        let trace_h_inv_s_lambda = faer_frob_inner(x.as_ref(), e_view.as_ref());

        // Calculate EDF as p - trace, clamped to the penalty nullspace dimension
        let p = h_eff.ncols() as f64;
        let rank_s = e_transformed.nrows() as f64;
        let mp = (p - rank_s).max(0.0);
        let edf = (p - trace_h_inv_s_lambda).clamp(mp, p);

        Ok(edf)
    }

    pub(super) fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
        let lin = pr.linear_constraints_transformed.as_ref()?;
        let active_tol = 1e-8;
        let beta_t = pr.beta_transformed.as_ref();
        let mut active_rows: Vec<Array1<f64>> = Vec::new();
        for i in 0..lin.a.nrows() {
            let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
            if slack <= active_tol {
                active_rows.push(lin.a.row(i).to_owned());
            }
        }
        if active_rows.is_empty() {
            return None;
        }

        let p_t = lin.a.ncols();
        let mut a_t = Array2::<f64>::zeros((p_t, active_rows.len()));
        for (j, row) in active_rows.iter().enumerate() {
            for k in 0..p_t {
                a_t[[k, j]] = row[k];
            }
        }

        let q_row = Self::orthonormalize_columns(&a_t, 1e-10); // basis for active row-space^T
        let rank = q_row.ncols();
        if rank == 0 {
            return None;
        }
        if rank >= p_t {
            return Some(Array2::<f64>::zeros((p_t, 0)));
        }

        // Build orthonormal basis for null(A_active) as complement of row-space.
        let mut z = Array2::<f64>::zeros((p_t, p_t - rank));
        let mut kept = 0usize;
        for j in 0..p_t {
            let mut v = Array1::<f64>::zeros(p_t);
            v[j] = 1.0;
            for t in 0..rank {
                let qt = q_row.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            for t in 0..kept {
                let zt = z.column(t);
                let proj = zt.dot(&v);
                v -= &zt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-10 {
                z.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
                if kept == p_t - rank {
                    break;
                }
            }
        }
        Some(z.slice(ndarray::s![.., 0..kept]).to_owned())
    }

    pub(super) fn enforce_constraint_kkt(&self, pr: &PirlsResult) -> Result<(), EstimationError> {
        let Some(kkt) = pr.constraint_kkt.as_ref() else {
            return Ok(());
        };
        let tol_primal = 1e-7;
        let tol_dual = 1e-7;
        let tol_comp = 1e-7;
        let tol_stat = 5e-6;
        if kkt.primal_feasibility > tol_primal
            || kkt.dual_feasibility > tol_dual
            || kkt.complementarity > tol_comp
            || kkt.stationarity > tol_stat
        {
            let mut worst_row_msg = String::new();
            if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                let mut worst = 0.0_f64;
                let mut worst_row = 0usize;
                for i in 0..lin.a.nrows() {
                    let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                    let viol = (-slack).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worst_row = i;
                    }
                }
                if worst > 0.0 {
                    worst_row_msg =
                        format!("; worst_row={} worst_violation={:.3e}", worst_row, worst);
                }
            }
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "KKT residuals exceed tolerance: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}; active={}/{}{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                kkt.n_active,
                kkt.n_constraints,
                worst_row_msg
            )));
        }
        Ok(())
    }

    pub(super) fn project_with_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let zt_m = z.t().dot(matrix);
        zt_m.dot(z)
    }

    pub(super) fn positive_part_factor_w(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Build W such that M_+^dagger = W W^T on the retained positive subspace.
        // This is the objective-consistent generalized inverse used for both
        // pseudo-logdet derivatives and IFT sensitivities in exact joint blocks.
        let (evals, evecs) = matrix
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let p = matrix.nrows();
        let max_ev = evals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let tol = (p.max(1) as f64) * f64::EPSILON * max_ev;
        let keep: Vec<usize> = evals
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v > tol { Some(i) } else { None })
            .collect();
        if keep.is_empty() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        let mut w = Array2::<f64>::zeros((p, keep.len()));
        for (col_idx, &eig_idx) in keep.iter().enumerate() {
            let scale = 1.0 / evals[eig_idx].sqrt();
            let u_col = evecs.column(eig_idx);
            let mut w_col = w.column_mut(col_idx);
            Zip::from(&mut w_col)
                .and(&u_col)
                .for_each(|w_elem, &u_elem| *w_elem = u_elem * scale);
        }
        Ok(w)
    }

    pub(super) fn fixed_subspace_penalty_rank_and_logdet(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(usize, f64), EstimationError> {
        let structural_rank = e_transformed.nrows().min(e_transformed.ncols());
        if structural_rank == 0 {
            return Ok((0, 0.0));
        }

        // Keep objective rank fixed to the structural penalty rank to avoid
        // rho-dependent rank flips from tiny eigenvalue jitter.
        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..s_lambda.nrows() {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, _) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let max_ev = order
            .first()
            .map(|&idx| evals[idx].abs())
            .unwrap_or(1.0)
            .max(1.0);
        let floor = (1e-12 * max_ev).max(1e-12);
        let log_det = order
            .iter()
            .take(structural_rank)
            .map(|&idx| evals[idx].max(floor).ln())
            .sum();
        Ok((structural_rank, log_det))
    }

    pub(super) fn fixed_subspace_penalty_trace(
        &self,
        e_transformed: &Array2<f64>,
        s_direction: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<f64, EstimationError> {
        // Use the exact same structural-rank convention as log|S|_+.
        let (structural_rank, _) =
            self.fixed_subspace_penalty_rank_and_logdet(e_transformed, ridge_passport)?;
        let p_dim = e_transformed.ncols();
        if structural_rank == 0 || p_dim == 0 {
            return Ok(0.0);
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, evecs) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let max_ev = order
            .first()
            .map(|&idx| evals[idx].abs())
            .unwrap_or(1.0)
            .max(1.0);
        let floor = (1e-12 * max_ev).max(1e-12);
        // Direct fixed-subspace contraction:
        //   tr(S^+ S_tau) = tr(D_+^{-1} U_+^T S_tau U_+),
        // where (U_+, D_+) are the kept structural positive modes.
        let mut trace = 0.0;
        for &idx in order.iter().take(structural_rank) {
            let ev = evals[idx].max(floor);
            let u = evecs.column(idx).to_owned();
            let spsi_u = s_direction.dot(&u);
            trace += u.dot(&spsi_u) / ev;
        }
        Ok(trace)
    }

    pub(super) fn update_warm_start_from(&self, pr: &PirlsResult) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match pr.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                let beta_original = match pr.coordinate_frame {
                    pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                        pr.beta_transformed.as_ref().clone()
                    }
                    pirls::PirlsCoordinateFrame::TransformedQs => {
                        pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
                    }
                };
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta_original));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    pub(crate) fn set_warm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match beta_original {
            Some(beta) if beta.len() == self.p => {
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta.to_owned()));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    /// Clear warm-start state. Used in tests to ensure consistent starting points
    /// when comparing different gradient computation paths.
    #[cfg(test)]
    pub fn clear_warm_start(&self) {
        self.warm_start_beta.write().unwrap().take();
        self.cache_manager.invalidate_eval_bundle();
    }

    /// Returns the per-penalty square-root matrices in the transformed coefficient basis
    /// without any λ weighting. Each returned R_k satisfies S_k = R_kᵀ R_k in that basis.
    /// Using these avoids accidental double counting of λ when forming derivatives.
    ///
    /// # Arguments
    /// * `pr` - The PIRLS result with the transformation matrix Qs
    ///
    /// # Returns
    pub(super) fn factorize_faer(&self, h: &Array2<f64>) -> FaerFactor {
        let mut planner = RidgePlanner::new(h);
        loop {
            let ridge = planner.ridge();
            if ridge > 0.0 {
                let regularized = add_ridge(h, ridge);
                let view = FaerArrayView::new(&regularized);
                if let Ok(f) = FaerLlt::new(view.as_ref(), Side::Lower) {
                    return FaerFactor::Llt(f);
                }
                if let Ok(f) = FaerLdlt::new(view.as_ref(), Side::Lower) {
                    return FaerFactor::Ldlt(f);
                }
                if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                    let f = FaerLblt::new(view.as_ref(), Side::Lower);
                    return FaerFactor::Lblt(f);
                }
            } else {
                let h_view = FaerArrayView::new(h);
                if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                    return FaerFactor::Llt(f);
                }
                if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                    return FaerFactor::Ldlt(f);
                }
            }
            planner.bump_with_matrix(h);
        }
    }

    pub(super) fn get_faer_factor(&self, rho: &Array1<f64>, h: &Array2<f64>) -> Arc<FaerFactor> {
        // Cache strategy: ρ alone is the key.
        // The cache deliberately ignores which Hessian matrix we are factoring.  Today this is
        // sound because every caller obeys a single rule:
        //   • Identity/Gaussian REML cost & gradient only ever request factors of the
        //     stabilized Hessian.
        //   • Non-Gaussian (logit/LAML) cost and gradient request factors of the effective/ridged Hessian.
        // Consequently each ρ corresponds to exactly one matrix within the lifetime of a
        // `RemlState`, so returning the cached factorization is correct.
        // This design is still brittle: adding a new code path that calls `get_faer_factor`
        // with a different H for the same ρ would silently reuse the wrong factor.  If such a
        // path ever appears, extend the key (for example by tagging the Hessian variant) or
        // split the cache.  The current key maximizes cache
        // hits across repeated EDF/gradient evaluations for the same smoothing parameters.
        let key_opt = self.rho_key_sanitized(rho);
        if let Some(key) = &key_opt
            && let Some(f) = self
                .cache_manager
                .faer_factor_cache
                .read()
                .unwrap()
                .get(key)
        {
            return Arc::clone(f);
        }
        let fact = Arc::new(self.factorize_faer(h));

        if let Some(key) = key_opt {
            let mut cache = self.cache_manager.faer_factor_cache.write().unwrap();
            if cache.len() > 64 {
                cache.clear();
            }
            cache.insert(key, Arc::clone(&fact));
        }
        fact
    }

    pub(super) const MIN_DMU_DETA: f64 = 1e-6;

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    pub(crate) fn weights(&self) -> ArrayView1<'a, f64> {
        self.weights
    }

    pub(super) fn sparse_penalty_logdet_runtime(
        &self,
        rho: &Array1<f64>,
        blocks: &[SparsePenaltyBlock],
    ) -> (f64, Array1<f64>) {
        let mut logdet = 0.0_f64;
        let mut det1 = Array1::<f64>::zeros(rho.len());
        for block in blocks {
            let rank = block.positive_eigenvalues.len() as f64;
            if block.term_index < det1.len() {
                det1[block.term_index] = rank;
            }
            logdet += rank * rho[block.term_index];
            for &eig in block.positive_eigenvalues.iter() {
                logdet += eig.ln();
            }
        }
        (logdet, det1)
    }

    pub(super) fn prepare_dense_eval_bundle_with_key(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        let (h_eff, ridge_passport) = self.effective_hessian(pirls_result.as_ref())?;

        const EIG_REL_THRESHOLD: f64 = 1e-10;
        const EIG_ABS_FLOOR: f64 = 1e-14;

        let dim = h_eff.nrows();
        let mut h_total = h_eff.clone();
        let mut firth_dense_operator: Option<Arc<FirthDenseOperator>> = None;
        if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let x_dense = pirls_result.x_transformed.to_dense_arc();
            let firth_op = Arc::new(Self::build_firth_dense_operator(
                x_dense.as_ref(),
                &pirls_result.final_eta,
            )?);
            // Firth-adjusted inner Jacobian for implicit differentiation:
            //   H_total = Xᵀ W X + S - H_φ,
            //   H_φ     = ∇²_β Φ
            //          = 0.5 [ Xᵀ diag(w'' ⊙ h) X - Bᵀ P B ].
            // This keeps B_k/B_{kl} solves on the same objective surface as
            // the Firth-augmented stationarity system.
            //
            // Conceptually Φ is the pseudodeterminant term 0.5 log|XᵀWX|_+.
            // The h_phi block below is therefore the curvature of that
            // identifiable-subspace Jeffreys penalty, represented in the
            // current transformed basis.
            let mut weighted_xtdx = Array2::<f64>::zeros(firth_op.x_dense.raw_dim());
            let diag_term = Self::xt_diag_x_dense_into(
                &firth_op.x_dense,
                &(&firth_op.w2 * &firth_op.h_diag),
                &mut weighted_xtdx,
            );
            let bpb = fast_ab(&firth_op.b_base.t().to_owned(), &firth_op.p_b_base);
            let mut h_phi = 0.5 * (diag_term - bpb);
            // Numerical symmetry guard.
            for i in 0..h_phi.nrows() {
                for j in 0..i {
                    let avg = 0.5 * (h_phi[[i, j]] + h_phi[[j, i]]);
                    h_phi[[i, j]] = avg;
                    h_phi[[j, i]] = avg;
                }
            }
            // Keep tiny numerical noise from making the solve surface less stable.
            if h_phi.iter().all(|v| v.is_finite()) {
                h_total -= &h_phi;
            }
            firth_dense_operator = Some(firth_op);
        }
        let (eigvals, eigvecs) = h_total
            .eigh(Side::Lower)
            .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
        let max_eig = eigvals.iter().copied().fold(0.0_f64, f64::max);
        let eig_threshold = if self.config.link_function() == LinkFunction::Identity {
            (max_eig * EIG_REL_THRESHOLD).max(EIG_ABS_FLOOR)
        } else {
            EIG_ABS_FLOOR
        };
        let h_total_log_det: f64 = eigvals
            .iter()
            .filter(|&&v| v > eig_threshold)
            .map(|&v| v.ln())
            .sum();
        if !h_total_log_det.is_finite() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }

        let valid_indices: Vec<usize> = eigvals
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v > eig_threshold { Some(i) } else { None })
            .collect();
        let min_kept = valid_indices
            .iter()
            .map(|&i| eigvals[i])
            .fold(f64::INFINITY, f64::min);
        let max_dropped = eigvals
            .iter()
            .copied()
            .filter(|&v| v <= eig_threshold)
            .fold(0.0_f64, f64::max);
        let active_subspace_rel_gap =
            if min_kept.is_finite() && max_dropped.is_finite() && min_kept > 0.0 {
                let gap = (min_kept - max_dropped).max(0.0);
                Some(gap / min_kept.max(1e-16))
            } else {
                None
            };
        let active_subspace_unstable =
            active_subspace_rel_gap.is_some_and(|rel_gap| rel_gap < 1e-6);
        // Active-subspace stability diagnostic for pseudo-logdet derivatives.
        // Exact second-order identities for log|H|_+ assume a fixed positive
        // eigenspace. When retained and dropped eigenvalues crowd the threshold,
        // the objective is near a non-smooth boundary and Hessian updates can be
        // numerically fragile.
        if log::log_enabled!(log::Level::Warn) {
            if active_subspace_unstable {
                let rel_gap = active_subspace_rel_gap.unwrap_or(f64::NAN);
                log::warn!(
                    "[REML] H_+ active-subspace is near instability: min_kept={:.3e}, max_dropped={:.3e}, threshold={:.3e}, rel_gap={:.3e}",
                    min_kept,
                    max_dropped,
                    eig_threshold,
                    rel_gap
                );
            }
        }

        let valid_count = valid_indices.len();
        let mut w = Array2::<f64>::zeros((dim, valid_count));

        for (w_col_idx, &eig_idx) in valid_indices.iter().enumerate() {
            let val = eigvals[eig_idx];
            let scale = 1.0 / val.sqrt();
            let u_col = eigvecs.column(eig_idx);
            let mut w_col = w.column_mut(w_col_idx);
            Zip::from(&mut w_col)
                .and(&u_col)
                .for_each(|w_elem, &u_elem| *w_elem = u_elem * scale);
        }

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::DenseSpectral,
            h_eff: Arc::new(h_eff),
            h_total: Arc::new(h_total),
            h_pos_factor_w: Arc::new(w),
            h_total_log_det,
            active_subspace_rel_gap,
            active_subspace_unstable,
            sparse_exact: None,
            firth_dense_operator,
        })
    }

    pub(super) fn prepare_sparse_eval_bundle_with_key(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        if !matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::OriginalSparseNative
        ) {
            return Err(EstimationError::InvalidInput(
                "sparse exact geometry requires sparse-native PIRLS coordinates".to_string(),
            ));
        }
        let ridge_passport = pirls_result.ridge_passport;
        let x_sparse = match &self.x {
            DesignMatrix::Sparse(s) => s,
            DesignMatrix::Dense(_) => {
                return Err(EstimationError::InvalidInput(
                    "sparse exact geometry requires sparse original design".to_string(),
                ));
            }
        };
        let penalty_blocks = self
            .sparse_penalty_blocks
            .as_ref()
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "sparse exact geometry requires block-separable penalties".to_string(),
                )
            })?
            .clone();

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, s_k) in self.s_full_list.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                s_lambda.scaled_add(lambdas[k], s_k);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &pirls_result.solve_weights,
            &s_lambda,
            ridge_passport.delta,
        )?;
        let (logdet_s_pos, det1_values) =
            self.sparse_penalty_logdet_runtime(rho, penalty_blocks.as_ref());
        let firth_dense_operator = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let x_dense = self.x().to_dense_arc();
            Some(Arc::new(Self::build_firth_dense_operator(
                x_dense.as_ref(),
                &pirls_result.final_eta,
            )?))
        } else {
            None
        };

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::SparseExactSpd,
            h_eff: Arc::new(Array2::zeros((0, 0))),
            h_total: Arc::new(Array2::zeros((0, 0))),
            h_pos_factor_w: Arc::new(Array2::zeros((0, 0))),
            h_total_log_det: 0.0,
            active_subspace_rel_gap: None,
            active_subspace_unstable: false,
            sparse_exact: Some(Arc::new(SparseExactEvalData {
                factor: Arc::new(sparse_system.factor),
                penalty_blocks,
                logdet_h: sparse_system.logdet_h,
                logdet_s_pos,
                det1_values: Arc::new(det1_values),
                trace_workspace: Arc::new(Mutex::new(SparseTraceWorkspace::default())),
            })),
            firth_dense_operator,
        })
    }

    /// Runs the inner P-IRLS loop, caching the result.
    pub(super) fn execute_pirls_if_needed(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Arc<PirlsResult>, EstimationError> {
        let use_cache = self
            .cache_manager
            .pirls_cache_enabled
            .load(Ordering::Relaxed);
        // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
        let key_opt = self.rho_key_sanitized(rho);
        if use_cache
            && let Some(key) = &key_opt
            && let Some(cached) = self.cache_manager.pirls_cache.write().unwrap().get(key)
        {
            // Do not overwrite the current warm start from cache hits.
            // Line search / multi-eval outer loops revisit older rho keys and
            // replacing a recent nearby beta with an older cached mode can
            // materially slow subsequent PIRLS convergence.
            return Ok(cached);
        }

        // Run P-IRLS with original matrices to perform fresh reparameterization
        // The returned result will include the transformation matrix qs
        let pirls_result = {
            let warm_start_holder = self.warm_start_beta.read().unwrap();
            let warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                warm_start_holder.as_ref()
            } else {
                None
            };
            let mut pirls_config = self.config.as_pirls_config();
            pirls_config.link_kind = if let Some(state) = self.runtime_mixture_link_state.clone() {
                InverseLink::Mixture(state)
            } else if let Some(state) = self.runtime_sas_link_state {
                if matches!(self.config.link_function(), LinkFunction::BetaLogistic) {
                    InverseLink::BetaLogistic(state)
                } else {
                    InverseLink::Sas(state)
                }
            } else {
                InverseLink::Standard(self.config.link_function())
            };
            pirls::fit_model_for_fixed_rho_matrix(
                LogSmoothingParamsView::new(rho.view()),
                &self.x,
                self.offset.view(),
                self.y,
                self.weights,
                &self.rs_list,
                Some(&self.balanced_penalty_root),
                Some(&self.reparam_invariant),
                self.p,
                &pirls_config,
                warm_start_ref,
                self.coefficient_lower_bounds.as_ref(),
                self.linear_constraints.as_ref(),
                None, // No SE for base model
            )
        };

        if let Err(e) = &pirls_result {
            println!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
            // Keep the previous successful warm start even when a trial point
            // fails. Outer line search commonly probes unstable candidates and
            // then returns to nearby feasible rho values where the prior warm
            // beta remains the best initializer.
        }

        let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
        let pirls_result = Arc::new(pirls_result);
        self.enforce_constraint_kkt(pirls_result.as_ref())?;

        // Check the status returned by the P-IRLS routine.
        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                self.update_warm_start_from(pirls_result.as_ref());
                // This is a successful fit. Cache only if key is valid (not NaN).
                if use_cache && let Some(key) = key_opt {
                    self.cache_manager
                        .pirls_cache
                        .write()
                        .unwrap()
                        .insert(key, Arc::clone(&pirls_result));
                }
                Ok(pirls_result)
            }
            pirls::PirlsStatus::Unstable => {
                // The fit was unstable. This is where we throw our specific, user-friendly error.
                // Pass the diagnostic info into the error
                Err(EstimationError::PerfectSeparationDetected {
                    iteration: pirls_result.iteration,
                    max_abs_eta: pirls_result.max_abs_eta,
                })
            }
            pirls::PirlsStatus::MaxIterationsReached => {
                // Envelope-theorem outer gradients require the inner solve to be near
                // stationarity; a loose max-iter acceptance threshold (e.g. 1.0) causes
                // persistent KKT/envelope diagnostic failures and inaccurate hyper-gradients.
                let acceptable_kkt = if self.runtime_sas_link_state.is_some() {
                    // SAS-link outer sweeps can stall at boundary-heavy rho configurations
                    // before satisfying strict default KKT tolerances; allow a wider
                    // near-stationary band to avoid false hard failures in those regimes.
                    (self.config.convergence_tolerance * 50.0).max(1e-2)
                } else {
                    (self.config.convergence_tolerance * 10.0).max(1e-4)
                };
                if pirls_result.last_gradient_norm > acceptable_kkt {
                    // The fit timed out and gradient is still too large for reliable outer
                    // derivatives, so fail fast and let the caller backtrack.
                    log::error!(
                        "P-IRLS failed convergence check: gradient norm {:.3e} > {:.3e} (iter {})",
                        pirls_result.last_gradient_norm,
                        acceptable_kkt,
                        pirls_result.iteration
                    );
                    Err(EstimationError::PirlsDidNotConverge {
                        max_iterations: pirls_result.iteration,
                        last_change: pirls_result.last_gradient_norm,
                    })
                } else {
                    // Near-stationary despite max iterations; treat as usable.
                    log::warn!(
                        "P-IRLS reached max iterations but gradient norm {:.3e} <= {:.3e}; accepting near-stationary fit.",
                        pirls_result.last_gradient_norm,
                        acceptable_kkt
                    );
                    self.update_warm_start_from(pirls_result.as_ref());
                    Ok(pirls_result)
                }
            }
        }
    }
}
impl<'a> RemlState<'a> {
    /// Compute the objective function for BFGS optimization.
    ///
    /// FULL OBJECTIVE REFERENCE
    /// ------------------------
    /// This function returns the scalar outer cost minimized over ρ.
    ///
    /// Non-Gaussian branch (negative LAML form used by optimizer):
    ///   V_LAML(ρ) =
    ///      -ℓ(β̂(ρ))
    ///      + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// where:
    ///   S(ρ) = Σ_k exp(ρ_k) S_k + δI
    ///   H(ρ) = -∇²ℓ(β̂(ρ)) + S(ρ)
    ///
    /// Gaussian identity-link REML branch:
    ///   V_REML(ρ, φ) =
    ///      D_p(ρ)/(2φ)
    ///      + (n_r/2) log φ
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// with profiled φ:
    ///   φ̂(ρ) = D_p(ρ)/n_r
    ///   V_REML,prof(ρ) =
    ///      (n_r/2) log D_p(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const.
    ///
    /// Consistency rule enforced throughout:
    ///   The same stabilized matrices/factorizations are used for
    ///   objective and gradient/Hessian terms. Mixing different H/S variants
    ///   causes deterministic gradient mismatch and unstable outer optimization.
    ///
    /// Determinant conventions:
    ///   - log|H| may use positive-part/stabilized spectrum conventions when needed.
    ///   - log|S|_+ follows fixed-rank pseudo-determinant conventions in the
    ///     transformed penalty basis, optionally including ridge policy.
    /// These conventions are mirrored in gradient code via corresponding trace terms.
    pub fn compute_cost(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
        let cost_call_idx = {
            let mut calls = self.arena.cost_eval_count.write().unwrap();
            *calls += 1;
            *calls
        };
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                // Inner linear algebra says "too singular" — treat as barrier.
                log::warn!(
                    "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                );
                // Diagnostics: which rho are at bounds
                let at_lower: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if v <= -RHO_BOUND + 1e-8 {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let at_upper: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                    .collect();
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    eprintln!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower, at_upper
                    );
                }
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                // Other errors still bubble up
                // Provide bounds diagnostics here too
                let at_lower: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if v <= -RHO_BOUND + 1e-8 {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let at_upper: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                    .collect();
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    eprintln!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower, at_upper
                    );
                }
                return Err(e);
            }
        };
        if Self::geometry_backend_kind(&bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_cost_sparse_exact(p, &bundle);
        }
        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_used = bundle.ridge_passport.delta;

        let lambdas = p.mapv(f64::exp);
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut e_eval = pirls_result.reparam_result.e_transformed.clone();
        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
            e_eval = pirls_result.reparam_result.e_transformed.dot(z);
        }
        let h_eff = &h_eff_eval;

        // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
        if !p.is_empty() {
            let k_lambda = p.len();
            let k_r = pirls_result.reparam_result.rs_transformed.len();
            let k_d = pirls_result.reparam_result.det1.len();
            if !(k_lambda == k_r && k_r == k_d) {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                    k_lambda, k_r, k_d
                )));
            }
            if self.nullspace_dims.len() != k_lambda {
                return Err(EstimationError::LayoutError(format!(
                    "Nullspace dimension mismatch: expected {} entries, got {}",
                    k_lambda,
                    self.nullspace_dims.len()
                )));
            }
        }

        // Don't barrier on non-PD; we'll stabilize and continue like mgcv
        // Only check eigenvalues if we needed to add a ridge
        const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
        let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
        if ridge_used > 0.0
            && let Ok((eigs, _)) = pirls_result.penalized_hessian_transformed.eigh(Side::Lower)
            && let Some(min_eig) = eigs.iter().cloned().reduce(f64::min)
        {
            if should_emit_h_min_eig_diag(min_eig) {
                eprintln!(
                    "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                    min_eig, ridge_used
                );
            }

            if min_eig <= 0.0 {
                log::warn!(
                    "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                    ridge_used
                );
            }

            if want_hot_diag
                && (!min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE)
            {
                let condition_number =
                    calculate_condition_number(&pirls_result.penalized_hessian_transformed)
                        .ok()
                        .unwrap_or(f64::INFINITY);

                log::warn!(
                    "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                    condition_number
                );
            }
        }
        // Use stable penalty calculation - no need to reconstruct matrices
        // The penalty term is already calculated stably in the P-IRLS loop

        match self.config.link_function() {
            LinkFunction::Identity => {
                let ridge_passport = pirls_result.ridge_passport;
                // From Wood (2017), Chapter 6, Eq. 6.24:
                // V_r(λ) = D_p/(2φ) + (r/2φ) + ½log|X'X/φ + S_λ/φ| - ½log|S_λ/φ|_+
                // where D_p = ||y - Xβ̂||² + β̂'S_λβ̂ is the PENALIZED deviance
                //
                // With profiled dispersion φ̂ = D_p/(n-M_p), this becomes:
                //   V_REML(ρ) =
                //     D_p/(2φ̂)
                //   + 0.5 log|H|
                //   - 0.5 log|S|_+
                //   + ((n-M_p)/2) log(2πφ̂),
                // where H = XᵀW0X + S(ρ), S(ρ)=Σ_k exp(ρ_k) S_k + δI.
                //
                // Because Gaussian identity has c=d=0, there is no third/fourth derivative
                // correction in H_k: ∂H/∂ρ_k = S_k^ρ exactly.

                // Check condition number with improved thresholds per Wood (2011)
                const MAX_CONDITION_NUMBER: f64 = 1e12;
                if want_hot_diag {
                    let cond = pirls_result
                        .penalized_hessian_transformed
                        .eigh(Side::Lower)
                        .ok()
                        .map(|(evals, _)| {
                            let max_ev = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            let min_ev = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            if min_ev <= 1e-12 {
                                f64::INFINITY
                            } else {
                                max_ev / min_ev
                            }
                        })
                        .unwrap_or(f64::NAN);
                    *self.arena.gaussian_cond_snapshot.write().unwrap() = cond;
                }
                let condition_number = *self.arena.gaussian_cond_snapshot.read().unwrap();
                if condition_number.is_finite() {
                    if condition_number > MAX_CONDITION_NUMBER {
                        log::warn!(
                            "Penalized Hessian very ill-conditioned (cond={:.2e}); proceeding despite poor conditioning.",
                            condition_number
                        );
                    } else if condition_number > 1e8 {
                        log::warn!(
                            "Penalized Hessian is ill-conditioned but proceeding: condition number = {condition_number:.2e}"
                        );
                    }
                }

                // STRATEGIC DESIGN DECISION: Use unweighted sample count for mgcv parity
                // In standard WLS theory, one might use sum(weights) as effective sample size.
                // However, mgcv deliberately uses the unweighted count 'n.true' in gam.fit3.
                let n = self.y.len() as f64;
                // Number of coefficients (transformed basis)

                // Calculate PENALIZED deviance D_p = ||y - Xβ̂||² + β̂'S_λβ̂
                let rss = pirls_result.deviance; // Unpenalized ||y - μ||²
                // Use stable penalty term calculated in P-IRLS
                let penalty = pirls_result.stable_penalty_term;

                let dp = rss + penalty;

                // Calculate EDF = p - tr((X'X + S_λ)⁻¹S_λ)
                // Work directly in the transformed basis for efficiency and numerical stability
                // This avoids transforming matrices back to the original basis unnecessarily
                // Penalty roots are available in reparam_result if needed

                // Nullspace dimension M_p is constant with respect to ρ.  Use it to profile φ
                // following the standard REML identity φ = D_p / (n - M_p).
                let (penalty_rank, log_det_s_plus) =
                    self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;
                let p_eff_dim = h_eff.ncols();
                let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;

                // EDF diagnostics are expensive; compute only when diagnostics are enabled.
                if want_hot_diag {
                    let edf = self.edf_from_h_and_e(&e_eval, lambdas.view(), h_eff)?;
                    log::debug!("[Diag] EDF total={:.3}", edf);
                    if n - edf < 1.0 {
                        log::warn!("Effective DoF exceeds samples; model may be overfit.");
                    }
                }

                let denom = (n - mp).max(LAML_RIDGE);
                let (dp_c, _) = smooth_floor_dp(dp);
                if dp < DP_FLOOR {
                    log::warn!(
                        "Penalized deviance {:.3e} fell below DP_FLOOR; clamping to maintain REML stability.",
                        dp
                    );
                }
                let phi = dp_c / denom;

                // log |H| = log |X'X + S_λ + ridge I| using the single effective
                // Hessian shared with the gradient. Ridge is already baked into h_eff.
                //
                // This is the same stabilized H used in compute_gradient;
                // otherwise the chain-rule pieces and determinant pieces are taken on
                // different objective surfaces and the optimizer sees inconsistent derivatives.
                let h_for_det = h_eff.clone();

                let chol = h_for_det.cholesky(Side::Lower).map_err(|_| {
                    let min_eig = h_eff
                        .clone()
                        .eigh(Side::Lower)
                        .ok()
                        .and_then(|(eigs, _)| eigs.iter().cloned().reduce(f64::min))
                        .unwrap_or(f64::NAN);
                    EstimationError::HessianNotPositiveDefinite {
                        min_eigenvalue: min_eig,
                    }
                })?;
                let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();

                // log |S_λ + ridge I|_+ (pseudo-determinant) to match the
                // stabilized penalty used by PIRLS.
                //
                // Fixed-rank rule: unpenalized/null directions do not contribute to the
                // pseudo-logdet. This keeps the objective continuous in ρ when S is singular
                // (or near-singular before ridge augmentation).
                // Standard REML expression from Wood (2017), 6.5.1
                // V = (n/2)log(2πσ²) + D_p/(2σ²) + ½log|H| - ½log|S_λ|_+ + (M_p-1)/2 log(2πσ²)
                // Simplifying: V = D_p/(2φ) + ½log|H| - ½log|S_λ|_+ + ((n-M_p)/2) log(2πφ)
                let reml = dp_c / (2.0 * phi)
                    + 0.5 * (log_det_h - log_det_s_plus)
                    + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                let prior_cost = self.compute_soft_prior_cost(p);

                Ok(reml + prior_cost)
            }
            _ => {
                // For non-Gaussian GLMs, use the LAML approximation
                // Note: Deviance = -2 * log-likelihood + C. So -0.5 * Deviance = log-likelihood - C/2.
                // Use stable penalty term calculated in P-IRLS
                let mut penalised_ll =
                    -0.5 * pirls_result.deviance - 0.5 * pirls_result.stable_penalty_term;

                let ridge_passport = pirls_result.ridge_passport;
                // Include Firth log-det term in LAML for consistency with inner PIRLS
                if self.config.firth_bias_reduction
                    && matches!(self.config.link_function(), LinkFunction::Logit)
                {
                    if let Some(firth_log_det) = pirls_result.firth_log_det {
                        penalised_ll += firth_log_det; // Jeffreys prior contribution
                    }
                }

                // Use the stabilized log|Sλ|_+ from the reparameterization (consistent with gradient)
                let (_penalty_rank, log_det_s) =
                    self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;

                // Log-determinant of the effective Hessian.
                // HESSIAN PASSPORT: Use the pre-computed h_total and its factorization
                // from the bundle to ensure exact consistency with gradient computation.
                // For Firth: h_total = h_eff - h_phi (computed in prepare_eval_bundle)
                // For non-Firth: h_total = h_eff
                //
                // LAML objective:
                //   V_LAML(ρ) =
                //      -ℓ(β̂) + 0.5 β̂ᵀSβ̂
                //    - 0.5 log|S|_+
                //    + 0.5 log|H|
                //    + const.
                //
                // For non-Gaussian families, H depends on ρ both directly through S and
                // indirectly through β̂(ρ), which induces the dH/dρ_k third-derivative term in
                // the exact gradient path (documented in compute_gradient).
                let log_det_h = if free_basis_opt.is_some() {
                    if h_total_eval.nrows() == 0 {
                        0.0
                    } else {
                        let (evals, _) = h_total_eval
                            .eigh(Side::Lower)
                            .map_err(EstimationError::EigendecompositionFailed)?;
                        let floor = 1e-10;
                        evals.iter().filter(|&&v| v > floor).map(|&v| v.ln()).sum()
                    }
                } else {
                    bundle.h_total_log_det
                };

                // Mp is null space dimension (number of unpenalized coefficients)
                // For logit, scale parameter is typically fixed at 1.0, but include for completeness
                let phi = 1.0; // Logit family typically has dispersion parameter = 1

                // Compute null space dimension using the TRANSFORMED, STABLE basis
                // Use the rank of the lambda-weighted transformed penalty root (e_transformed)
                // to determine M_p with the transformed penalty basis.
                let (penalty_rank, _) =
                    self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;
                let p_eff_dim = h_eff.ncols();
                let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;

                let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_h
                    + (mp / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                // Diagnostics below are expensive and not needed for objective value.
                let (edf, trace_h_inv_s_lambda, stab_cond) = if want_hot_diag {
                    let p_eff = h_eff.ncols() as f64;
                    let edf = self.edf_from_h_and_e(&e_eval, lambdas.view(), h_eff)?;
                    let trace_h_inv_s_lambda = (p_eff - edf).max(0.0);
                    let stab_cond = pirls_result
                        .penalized_hessian_transformed
                        .eigh(Side::Lower)
                        .ok()
                        .map(|(evals, _)| {
                            let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            max / min.max(1e-12)
                        })
                        .unwrap_or(f64::NAN);
                    (edf, trace_h_inv_s_lambda, stab_cond)
                } else {
                    (f64::NAN, f64::NAN, f64::NAN)
                };

                // Raw-condition diagnostics are rate-limited in this loop.
                // We only refresh occasionally, and keep the last snapshot otherwise.
                let raw_cond = if matches!(self.x(), DesignMatrix::Dense(_)) && want_hot_diag {
                    let x_orig_arc = self.x().to_dense_arc();
                    let x_orig = x_orig_arc.as_ref();
                    let w_orig = self.weights();
                    let sqrt_w = w_orig.mapv(|w| w.max(0.0).sqrt());
                    let wx = x_orig * &sqrt_w.insert_axis(Axis(1));
                    let mut h_raw = fast_ata(&wx);
                    for (k, &lambda) in lambdas.iter().enumerate() {
                        let s_k = &self.s_full_list[k];
                        if lambda != 0.0 {
                            h_raw.scaled_add(lambda, s_k);
                        }
                    }
                    let raw = h_raw
                        .eigh(Side::Lower)
                        .ok()
                        .map(|(evals, _)| {
                            let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            max / min.max(1e-12)
                        })
                        .unwrap_or(f64::NAN);
                    *self.arena.raw_cond_snapshot.write().unwrap() = raw;
                    raw
                } else {
                    *self.arena.raw_cond_snapshot.read().unwrap()
                };
                if want_hot_diag {
                    self.log_gam_cost(
                        &p,
                        lambdas.as_slice().unwrap_or(&[]),
                        laml,
                        stab_cond,
                        raw_cond,
                        edf,
                        trace_h_inv_s_lambda,
                    );
                }

                let prior_cost = self.compute_soft_prior_cost(p);

                Ok(-laml + prior_cost)
            }
        }
    }

    ///
    /// -------------------------------------------------------------------------
    /// Exact non-Laplace evidence identities (reference comments; not runtime path)
    /// -------------------------------------------------------------------------
    /// We optimize a Laplace-style outer objective for scalability, but the exact
    /// marginal likelihood for non-Gaussian models can be written analytically as:
    ///
    ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
    ///
    /// Universal exact gradient identity (when differentiation under the integral
    /// is justified and L(ρ) < ∞):
    ///
    ///   ∂_{ρ_k} log L(ρ)
    ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ βᵀ S_k β ].
    ///
    /// Laplace bridge to implemented terms:
    /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
    ///     E[βᵀ S_k β] ≈ β̂ᵀ S_k β̂ + tr(H^{-1} S_k),
    ///   giving the familiar quadratic + trace structure.
    /// - In this code those appear as:
    ///     0.5 * β̂ᵀ S_k^ρ β̂,
    ///     -0.5 * tr(S^+ S_k^ρ),
    ///     +0.5 * tr(H^{-1} H_k).
    ///
    /// Why this does NOT collapse to only tr(H^{-1}S_k):
    /// - The exact identity differentiates the true integral measure.
    /// - LAML differentiates a moving approximation:
    ///     V_LAML(ρ) = -ℓ(β̂(ρ)) + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///                 + 0.5 log|H(ρ)| - 0.5 log|S(ρ)|_+.
    /// - Here both center β̂(ρ) and curvature H(ρ) move with ρ.
    /// - For non-Gaussian families, H_k includes the third-derivative tensor path
    ///   through β̂(ρ), i.e. H_k != S_k^ρ. These are the explicit dH/dρ_k terms
    ///   retained below to differentiate the Laplace objective exactly.
    ///
    /// For Bernoulli-logit, an exact Pólya-Gamma augmentation gives:
    ///
    ///   L(ρ) = 2^{-n} (2π)^{p/2}
    ///          E_{ω_i ~ PG(1,0)} [ |Q(ω,ρ)|^{-1/2} exp(0.5 bᵀ Q^{-1} b) ],
    ///   Q(ω,ρ)=S(ρ)+XᵀΩX, b=Xᵀ(y-1/2).
    ///
    /// and
    ///
    ///   ∂_{ρ_k} log L
    ///   = -0.5 * exp(ρ_k) *
    ///     E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ],  μ=Q^{-1}b.
    /// Equivalently, since β|ω,y,ρ ~ N(μ,Q^{-1}):
    ///   E[βᵀS_kβ | ω,y,ρ] = tr(S_k Q^{-1}) + μᵀS_kμ.
    ///
    /// yielding exact (but high-dimensional) contour integrals / series after
    /// analytically integrating β.
    ///
    /// Practical note:
    /// - These are exact equalities but generally not polynomial-time tractable
    ///   for arbitrary dense (X, n, p).
    /// - This code therefore uses deterministic Laplace/implicit-differentiation
    ///   machinery for the main optimizer path, with exact tensor terms where
    ///   feasible (H_k, H_{kℓ}, c/d arrays), and scalable trace backends.
    ///
    /// FULL OUTER-DERIVATIVE REFERENCE (exact system, sign convention used here)
    /// -------------------------------------------------------------------------
    /// This optimizer minimizes an outer cost V(ρ).
    ///
    /// Common definitions:
    ///   λ_k = exp(ρ_k)
    ///   S(ρ) = Σ_k λ_k S_k + δI
    ///   A_k = ∂S/∂ρ_k = λ_k S_k
    ///   A_{kℓ} = ∂²S/(∂ρ_k∂ρ_ℓ) = δ_{kℓ} A_k
    ///
    /// Inner mode (β̂):
    ///   ∇_β ℓ(β̂) - S(ρ) β̂ = 0
    ///
    /// Curvature:
    ///   H(ρ) = -∇²_β ℓ(β̂(ρ)) + S(ρ)
    ///
    ///   w_i = -∂²ℓ_i/∂η_i²
    ///   d_i = -∂³ℓ_i/∂η_i³
    ///   e_i = -∂⁴ℓ_i/∂η_i⁴
    ///
    /// Then:
    ///   H_k = A_k + Xᵀ diag(d ⊙ u_k) X,     u_k := X B_k
    ///   H_{kℓ} = δ_{kℓ}A_k + Xᵀ diag(e ⊙ u_k ⊙ u_ℓ + d ⊙ u_{kℓ}) X
    ///
    /// with implicit derivatives:
    ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
    ///   H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ}A_k β̂ + A_k B_ℓ)
    ///
    /// Non-Gaussian negative LAML cost:
    ///   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀSβ̂ + 0.5 log|H| - 0.5 log|S|_+
    ///
    /// Exact gradient:
    ///   g_k = 0.5 β̂ᵀA_kβ̂ + 0.5 tr(H^{-1}H_k) - 0.5 ∂_k log|S|_+
    ///
    /// Exact Hessian decomposition:
    ///   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
    ///
    ///   Q_{kℓ} = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
    ///
    ///   L_{kℓ} = 0.5 [ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
    ///
    ///   P_{kℓ} = -0.5 ∂²_{kℓ} log|S|_+
    ///
    /// Here, this function computes the exact gradient terms (including dH/dρ_k via d_i).
    /// The full exact Hessian is not assembled in this loop because it requires B_{kℓ}
    /// solves and fourth-derivative terms for every (k,ℓ) pair.
    ///
    /// Gaussian REML note:
    ///   In identity-link Gaussian, d=e=0 so H_k=A_k and H_{kℓ}=δ_{kℓ}A_k.
    ///   With profiled φ, use either:
    ///   - explicit profiled objective derivatives, or
    ///   - Schur complement in (ρ, log φ):
    ///       H_prof = H_{ρρ} - H_{ρα} H_{αα}^{-1} H_{αρ}.
    ///
    /// Pseudo-determinant note:
    ///   The code uses fixed-rank/stabilized conventions for log|S|_+ to keep objective
    ///   derivatives smooth and consistent with the transformed penalty basis used by PIRLS.
    ///
    /// This is the core of the outer optimization loop and provides the search direction for the BFGS algorithm.
    /// The calculation differs significantly between the Gaussian (REML) and non-Gaussian (LAML) cases.
    ///
    /// # Mathematical Basis (Gaussian/REML Case)
    ///
    /// For Gaussian models (Identity link), we minimize the negative REML log-likelihood, which serves as our cost function.
    /// From Wood (2011, JRSSB, Eq. 4), the cost function to minimize is:
    ///
    ///   Cost(ρ) = -l_r(ρ) = D_p / (2φ) + (1/2)log|XᵀWX + S(ρ)| - (1/2)log|S(ρ)|_+
    ///
    /// where D_p is the penalized deviance, H = XᵀWX + S(ρ) is the penalized Hessian, S(ρ) is the total
    /// penalty matrix, and |S(ρ)|_+ is the pseudo-determinant.
    ///
    /// The gradient ∇Cost(ρ) is computed term-by-term. A key simplification for the Gaussian case is the
    /// **envelope theorem**: at the P-IRLS optimum for β̂, the derivative of the cost function with respect to β̂ is zero.
    /// This means we only need the *partial* derivatives with respect to ρ, and the complex indirect derivatives
    /// involving ∂β̂/∂ρ can be ignored.
    ///
    /// # Mathematical Basis (Non-Gaussian/LAML Case)
    ///
    /// For non-Gaussian models, the envelope theorem does not apply because the weight matrix W depends on β̂.
    /// The gradient requires calculating the full derivative, including the indirect term (∂V/∂β̂)ᵀ(∂β̂/∂ρ).
    /// This leads to a different final formula involving derivatives of the weight matrix, as detailed in
    /// Wood (2011, Appendix D).
    ///
    /// This method handles two distinct statistical criteria for marginal likelihood optimization:
    ///
    /// - For Gaussian models (Identity link), this calculates the exact REML gradient
    ///   (Restricted Maximum Likelihood).
    /// - For non-Gaussian GLMs, this calculates the LAML gradient (Laplace Approximate
    ///   Marginal Likelihood) as derived in Wood (2011, Appendix C & D).
    ///
    /// # Mathematical Theory
    ///
    /// The gradient calculation requires careful application of the chain rule and envelope theorem
    /// due to the nested optimization structure of GAMs:
    ///
    /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
    ///   for a fixed set of smoothing parameters ρ.
    /// - The outer loop (BFGS) finds smoothing parameters ρ that maximize the marginal likelihood.
    ///
    /// Since β̂ is an implicit function of ρ, the total derivative is:
    ///
    ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
    ///
    /// By the envelope theorem, (∂V_R/∂β̂) = 0 at the optimum β̂, so the first term vanishes.
    ///
    /// # Key Distinction Between REML and LAML Gradients
    ///
    /// - Gaussian (REML): by the envelope theorem the indirect β̂ terms vanish. The deviance
    ///   contribution reduces to the penalty-only derivative, yielding the familiar
    ///   (β̂ᵀS_kβ̂)/σ² piece in the gradient.
    /// - Non-Gaussian (LAML): there is no cancellation of the penalty derivative within the
    ///   deviance component. The derivative of the penalized deviance contains both
    ///   d(D)/dρ_k and d(βᵀSβ)/dρ_k. Our implementation follows mgcv’s gdi1: we add the penalty
    ///   derivative to the deviance derivative before applying the 1/2 factor.
    // Stage: Start with the chain rule for any λₖ,
    //     dV/dλₖ = ∂V/∂λₖ  (holding β̂ fixed)  +  (∂V/∂β̂)ᵀ · (∂β̂/∂λₖ).
    //     The first summand is called the direct part, the second the indirect part.
    //
    // Stage: Note the two outer criteria—Gaussian likelihood maximizes REML, while non-Gaussian likelihood
    //     maximizes a Laplace approximation to the marginal likelihood (LAML). These objectives respond differently to β̂.
    //
    //     2.1  Gaussian case, REML.
    //          The REML construction integrates the fixed effects out of the likelihood.  At the optimum
    //          the partial derivative ∂V/∂β̂ is exactly zero.  The indirect part therefore vanishes.
    //          What remains is the direct derivative of the penalty and determinant terms.  The penalty
    //          contribution is found by differentiating −½ β̂ᵀ S_λ β̂ / σ² with respect to λₖ; this yields
    //          −½ β̂ᵀ Sₖ β̂ / σ².  No opposing term exists, so the quantity stays in the REML gradient.
    //          The code path selected by LinkFunction::Identity therefore computes
    //          beta_term = β̂ᵀ Sₖ β̂ and places it inside
    //          gradient[k] = 0.5 * λₖ * (beta_term / σ² − trace_term).
    //
    //     2.2  Non-Gaussian case, LAML.
    //          The Laplace objective contains −½ log |H_p| with H_p = Xᵀ W(β̂) X + S_λ.  Because W
    //          depends on β̂, the total derivative includes dW/dλₖ via β̂.  Differentiating the
    //          optimality condition for β̂ gives
    //          ∂β̂/∂λₖ = −λₖ H_p⁻¹ Sₖ β̂.  The penalized log-likelihood L(β̂, λ) still obeys the
    //          envelope theorem, so dL/dλₖ = −½ β̂ᵀ Sₖ β̂ (no implicit term).
    //          The resulting cost gradient combines four pieces:
    //            +½ λₖ β̂ᵀ Sₖ β̂
    //            +½ λₖ tr(H_p⁻¹ Sₖ)
    //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
    //            −½ λₖ tr(S_λ⁺ Sₖ)
    //
    // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
    //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
    //     direct quadratic pieces are exact negatives, which is what the algebra requires.
    pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.arena
            .last_gradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        // Get the converged P-IRLS result for the current rho (`p`)
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(err @ EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(err);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };
        let analytic = self.compute_gradient_with_bundle(p, &bundle)?;
        Ok(analytic)
    }
}
