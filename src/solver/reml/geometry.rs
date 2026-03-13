use super::*;
use crate::linalg::sparse_exact::SparseOperatorBlockSupport;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SparseExactLikelihoodStructure {
    GaussianProfiledScale,
    SinglePredictorDiagonalCurvature,
}

impl<'a> RemlState<'a> {
    fn sparse_exact_likelihood_structure(&self) -> SparseExactLikelihoodStructure {
        match self.config.link_function() {
            // Important scope note:
            // `RemlState` is the single-predictor GLM/GAM REML engine.
            // In the currently supported family set, `Identity` reaches this
            // code only for Gaussian-with-profiled-scale fits. Multi-block
            // GAMLSS and survival models are routed through the custom-family
            // engine instead, where joint block curvature is handled explicitly.
            LinkFunction::Identity => SparseExactLikelihoodStructure::GaussianProfiledScale,
            LinkFunction::Logit
            | LinkFunction::Probit
            | LinkFunction::CLogLog
            | LinkFunction::Sas
            | LinkFunction::BetaLogistic => {
                SparseExactLikelihoodStructure::SinglePredictorDiagonalCurvature
            }
        }
    }

    fn operator_block_is_local(
        direction_block: &Array2<f64>,
        p_start: usize,
        p_end: usize,
        tol: f64,
    ) -> bool {
        let p = direction_block.nrows();
        let cols = direction_block.ncols();
        for col in 0..cols {
            for row in 0..p_start {
                if direction_block[[row, col]].abs() > tol {
                    return false;
                }
            }
            for row in p_end..p {
                if direction_block[[row, col]].abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn trace_from_selected_block(
        h_block: &Array2<f64>,
        direction_block: &Array2<f64>,
        p_start: usize,
    ) -> f64 {
        let block_dim = h_block.ncols();
        let mut trace = 0.0_f64;
        for col in 0..block_dim {
            for i in 0..block_dim {
                trace += h_block[[col, i]] * direction_block[[p_start + i, col]];
            }
        }
        trace
    }

    fn trace_hinv_operator_sparse_dispatch<F>(
        &self,
        sparse: &SparseExactEvalData,
        p: usize,
        support: Option<SparseOperatorBlockSupport>,
        apply_direction: F,
    ) -> Result<f64, EstimationError>
    where
        F: FnMut(&Array2<f64>) -> Result<Array2<f64>, EstimationError>,
    {
        if let Some(sup) = support {
            let mut workspace = sparse.traceworkspace.lock().unwrap();
            self.trace_hinv_operator_sparse_exact(
                &sparse.factor,
                p,
                Some(&mut workspace),
                Some(sup),
                apply_direction,
            )
        } else {
            self.trace_hinv_operator_sparse_exact(&sparse.factor, p, None, None, apply_direction)
        }
    }

    pub(super) fn sparse_exact_beta_original(&self, pirls_result: &PirlsResult) -> Array1<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                pirls_result.beta_transformed.as_ref().clone()
            }
            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                .reparam_result
                .qs
                .dot(pirls_result.beta_transformed.as_ref()),
        }
    }

    pub(super) fn trace_hinv_operator_sparse_exact<F>(
        &self,
        factor: &SparseExactFactor,
        p: usize,
        mut workspace: Option<&mut SparseTraceWorkspace>,
        support: Option<SparseOperatorBlockSupport>,
        mut apply_direction: F,
    ) -> Result<f64, EstimationError>
    where
        F: FnMut(&Array2<f64>) -> Result<Array2<f64>, EstimationError>,
    {
        if p == 0 {
            return Ok(0.0);
        }
        if let (Some(ws), Some(block)) = (workspace.as_deref_mut(), support) {
            if block.strict && block.p_end > block.p_start && block.p_end <= p {
                let block_dim = block.p_end - block.p_start;
                let mut basis_block = Array2::<f64>::zeros((p, block_dim));
                for local_col in 0..block_dim {
                    basis_block[[block.p_start + local_col, local_col]] = 1.0;
                }
                let direction_block = apply_direction(&basis_block)?;
                if direction_block.nrows() != p || direction_block.ncols() != block_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "trace_hinv_operator_sparse_exact apply returned {}x{} for basis block {}x{}",
                        direction_block.nrows(),
                        direction_block.ncols(),
                        p,
                        block_dim
                    )));
                }

                const SUPPORT_TOL: f64 = 1e-12;
                if Self::operator_block_is_local(
                    &direction_block,
                    block.p_start,
                    block.p_end,
                    SUPPORT_TOL,
                ) {
                    let h_block = ws.selected_block_inverse(factor, block.p_start, block.p_end)?;
                    return Ok(Self::trace_from_selected_block(
                        h_block,
                        &direction_block,
                        block.p_start,
                    ));
                }
            }
        }
        let chunk = 32usize;
        let mut trace = 0.0_f64;
        let mut basis_buf = Array2::<f64>::zeros((p, chunk));
        // Lazily allocated for the final (potentially smaller) chunk; the initial
        // value is never read — it is always overwritten inside the loop.
        #[allow(unused_assignments)]
        let mut small_buf = Array2::<f64>::zeros((0, 0));
        let mut start = 0usize;
        while start < p {
            let end = (start + chunk).min(p);
            let block_cols = end - start;
            // Reuse pre-allocated buffer for full-size chunks; allocate only
            // for the (at most one) final chunk that is smaller.
            let basis_block_ref = if block_cols == chunk {
                basis_buf.fill(0.0);
                for local_col in 0..block_cols {
                    basis_buf[[start + local_col, local_col]] = 1.0;
                }
                &basis_buf
            } else {
                small_buf = Array2::<f64>::zeros((p, block_cols));
                for local_col in 0..block_cols {
                    small_buf[[start + local_col, local_col]] = 1.0;
                }
                &small_buf
            };
            let direction_block = apply_direction(basis_block_ref)?;
            if direction_block.nrows() != p || direction_block.ncols() != block_cols {
                return Err(EstimationError::InvalidInput(format!(
                    "trace_hinv_operator_sparse_exact apply returned {}x{} for basis block {}x{}",
                    direction_block.nrows(),
                    direction_block.ncols(),
                    p,
                    block_cols
                )));
            }
            let solved = solve_sparse_spdmulti(factor, &direction_block)?;
            for local_col in 0..block_cols {
                let global_col = start + local_col;
                trace += solved[[global_col, local_col]];
            }
            start = end;
        }
        Ok(trace)
    }

    pub(super) fn sparse_exactweighted_cross_trace_xtau(
        &self,
        factor: &SparseExactFactor,
        x_tau: &HyperDesignDerivative,
        weights_diag: &Array1<f64>,
    ) -> Result<f64, EstimationError> {
        let n = x_tau.nrows();
        let p = x_tau.ncols();
        if n == 0 || p == 0 {
            return Ok(0.0);
        }
        if weights_diag.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "weighted_cross_trace_xtau weights length mismatch: weights={}, n={}",
                weights_diag.len(),
                n
            )));
        }
        let mut total = 0.0_f64;
        let batch = 32usize;

        let x_dense_opt = self.x().as_dense();
        let x_csr_opt = self.x().as_sparse().and_then(|x| x.to_csr_arc());

        let mut start = 0usize;
        while start < n {
            let end = (start + batch).min(n);
            let rhs = x_tau.transpose_row_block(start..end);
            let solved = solve_sparse_spdmulti(factor, &rhs)?;
            for local_col in 0..(end - start) {
                let row = start + local_col;
                let z_col = solved.column(local_col);
                let row_dot = if let Some(x_dense) = x_dense_opt {
                    x_dense.row(row).dot(&z_col)
                } else if let Some(csr) = x_csr_opt.as_ref() {
                    let symbolic = csr.symbolic();
                    let row_ptr = symbolic.row_ptr();
                    let col_idx = symbolic.col_idx();
                    let values = csr.val();
                    let mut acc = 0.0_f64;
                    let r0 = row_ptr[row];
                    let r1 = row_ptr[row + 1];
                    for idx in r0..r1 {
                        let col = col_idx[idx];
                        acc += values[idx] * z_col[col];
                    }
                    acc
                } else {
                    0.0
                };
                total += weights_diag[row] * row_dot;
            }
            start = end;
        }
        Ok(total)
    }

    pub(super) fn compute_directional_hypergradient_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();
        let p = self.p;
        let n = self.y.len();
        if hyper_dir.x_tau_original.nrows() != n || hyper_dir.x_tau_original.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "X_tau shape mismatch for sparse directional gradient: expected {}x{}, got {}x{}",
                n,
                p,
                hyper_dir.x_tau_original.nrows(),
                hyper_dir.x_tau_original.ncols()
            )));
        }
        Self::validate_penalty_component_shapes(hyper_dir.penalty_first_components(), p, "S_tau")?;
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);

        let beta = self.sparse_exact_beta_original(pirls_result);
        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);

        let x_tau_beta = hyper_dir.x_tau_original.dot(&beta);
        let s_tau_total = hyper_dir.penalty_total_at(rho, p)?;
        let weighted_x_tau_beta = &pirls_result.solveweights * &x_tau_beta;
        let mut g_psi = hyper_dir.x_tau_original.t_dot(&u)
            - self.x().transpose_vector_multiply(&weighted_x_tau_beta)
            - s_tau_total.dot(&beta);

        let mut fit_firth_partial = 0.0_f64;
        let mut firth_op_opt: Option<FirthDenseOperator> = None;
        let mut hphi_tau_kernel_opt: Option<FirthTauPartialKernel> = None;
        if firth_logit_active {
            let op = if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                cached.as_ref().clone()
            } else {
                let x_dense_arc = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML directional hyper-gradient requires dense design",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Self::build_firth_dense_operator(x_dense_arc.as_ref(), &pirls_result.final_eta)?
            };
            let need_tau_kernel = hyper_dir.x_tau_original.any_nonzero();
            let tau_bundle = Self::firth_exact_tau_kernel_compact(
                &op,
                &hyper_dir.x_tau_original,
                &beta,
                need_tau_kernel,
            );
            g_psi -= &tau_bundle.gphi_tau;
            fit_firth_partial = tau_bundle.phi_tau_partial;
            if need_tau_kernel {
                hphi_tau_kernel_opt = Some(tau_bundle.tau_kernel.expect(
                    "exact sparse assembly requires tau kernel when design drift is active",
                ));
            }
            firth_op_opt = Some(op);
        }

        let beta_tau = solve_sparse_spd(&sparse.factor, &g_psi)?;
        let eta_tau = &x_tau_beta + &self.x().matrixvectormultiply(&beta_tau);

        let fit_block =
            -u.dot(&x_tau_beta) + 0.5 * beta.dot(&s_tau_total.dot(&beta)) + fit_firth_partial;

        let trace_s_tau = self.trace_hinv_operator_sparse_dispatch(
            sparse,
            p,
            None,
            |basis_block: &Array2<f64>| Ok(s_tau_total.dot(basis_block)),
        )?;
        let cross = self.sparse_exactweighted_cross_trace_xtau(
            &sparse.factor,
            &hyper_dir.x_tau_original,
            &pirls_result.solveweights,
        )?;
        let mut trace_h = trace_s_tau + 2.0 * cross;

        match self.sparse_exact_likelihood_structure() {
            SparseExactLikelihoodStructure::GaussianProfiledScale => {
                let e = &pirls_result.reparam_result.e_transformed;
                let (penalty_rank, _) =
                    self.fixed_subspace_penalty_rank_and_logdet(e, pirls_result.ridge_passport)?;
                let mp = (p.saturating_sub(penalty_rank)) as f64;
                let dp = pirls_result.deviance + pirls_result.stable_penalty_term;
                let (dp_c, _) = smooth_floor_dp(dp);
                let phi = dp_c / ((n as f64 - mp).max(LAML_RIDGE));
                if !phi.is_finite() || phi <= 0.0 {
                    return Err(EstimationError::InvalidInput(
                        "invalid profiled Gaussian dispersion in sparse directional hyper-gradient"
                            .to_string(),
                    ));
                }
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &pirls_result.reparam_result.e_transformed,
                    &s_tau_total,
                    pirls_result.ridge_passport,
                )?;
                let dp_tau = 2.0 * fit_block;
                Ok(dp_tau / (2.0 * phi) + 0.5 * trace_h - 0.5 * pseudo_det_trace)
            }
            SparseExactLikelihoodStructure::SinglePredictorDiagonalCurvature => {
                let runtime_link = self.runtime_inverse_link();
                let w_tau = crate::pirls::directionalworking_curvature_from_etawith_state(
                    &runtime_link,
                    &pirls_result.final_eta,
                    self.weights,
                    &pirls_result.solveweights,
                    &eta_tau,
                )?;
                let leverages = leverages_from_factor(&sparse.factor, self.x())?;
                match w_tau {
                    DirectionalWorkingCurvature::Diagonal(diag) => {
                        if diag.len() != leverages.len() {
                            return Err(EstimationError::InvalidInput(format!(
                                "W_tau/leverages length mismatch in sparse directional gradient: w_tau={}, leverages={}",
                                diag.len(),
                                leverages.len()
                            )));
                        }
                        trace_h += leverages.dot(&diag);
                    }
                }
                if let Some(op) = firth_op_opt.as_ref() {
                    // Fully matrix-free sparse trace for Firth curvature drift:
                    //   tr(H^{-1}(H_{phi,tau}|beta + D(Hphi)[beta_tau])).
                    // We avoid dense p×p directional matrix materialization by
                    // applying both operators to identity blocks on demand.
                    let firth_dir = Self::firth_direction(op, &beta_tau);
                    let p_dim = self.p;
                    let tau_kernel = hphi_tau_kernel_opt.clone();
                    let tau_x = &hyper_dir.x_tau_original;
                    let tr_firth = self.trace_hinv_operator_sparse_dispatch(
                        sparse,
                        p_dim,
                        None,
                        |basis_block: &Array2<f64>| {
                            Ok(Self::firth_hphi_trace_apply_combined_compact(
                                op,
                                &firth_dir,
                                tau_x,
                                tau_kernel.as_ref(),
                                basis_block,
                            ))
                        },
                    )?;
                    trace_h -= tr_firth;
                }
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &pirls_result.reparam_result.e_transformed,
                    &s_tau_total,
                    pirls_result.ridge_passport,
                )?;
                Ok(fit_block + 0.5 * trace_h - 0.5 * pseudo_det_trace)
            }
        }
    }

    pub(crate) fn lastridge_used(&self) -> Option<f64> {
        self.cache_manager
            .current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .map(|bundle| bundle.ridge_passport.delta)
    }
}

impl<'a> RemlState<'a> {
    pub(super) fn geometry_backend_kind(bundle: &EvalShared) -> GeometryBackendKind {
        bundle.backend_kind()
    }

    pub(super) fn selecthessian_strategy_policy(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> HessianStrategyDecision {
        if self.usesobjective_consistentfdgradient(rho) {
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::DiagnosticNumeric,
                reason: "objective_consistent_numericgradient",
            };
        }
        if bundle.active_subspace_unstable {
            // Hard-truncated logdet identities are only branch-local in rho.
            // Once the retained H_+ eigenspace is near a threshold crossing,
            // we stop calling the spectral Hessian "exact" and fall back to
            // the safer analytic policy.
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::AnalyticFallback,
                reason: "active_subspace_unstable",
            };
        }
        HessianStrategyDecision {
            strategy: HessianEvalStrategyKind::SpectralExact,
            reason: "exact_preferred",
        }
    }

    pub(super) fn compute_lamlhessian_by_strategy(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        decision: HessianStrategyDecision,
    ) -> Result<Array2<f64>, EstimationError> {
        match decision.strategy {
            HessianEvalStrategyKind::SpectralExact => self.compute_lamlhessian_exact(rho),
            HessianEvalStrategyKind::AnalyticFallback => {
                self.compute_lamlhessian_analytic_fallback(rho, Some(bundle))
            }
            HessianEvalStrategyKind::DiagnosticNumeric => {
                log::warn!(
                    "Using diagnostic numeric Hessian strategy routing (reason={}); dispatching to deterministic analytic fallback.",
                    decision.reason
                );
                self.compute_lamlhessian_analytic_fallback(rho, Some(bundle))
            }
        }
    }

    pub(super) fn select_reml_geometry(&self, rho: &Array1<f64>) -> SparseRemlDecision {
        let p = self.p;
        let has_dense_constraints =
            self.linear_constraints.is_some() || self.coefficient_lower_bounds.is_some();
        let x_sparse = self.x.as_sparse();
        let nnz_x = x_sparse.map(|s| s.val().len()).unwrap_or(0);
        let dense_backend =
            |reason: &'static str,
             nnz_h_upper_est: Option<usize>,
             density_h_upper_est: Option<f64>| SparseRemlDecision {
                geometry: RemlGeometry::DenseSpectral,
                reason,
                p,
                nnz_x,
                nnz_h_upper_est,
                density_h_upper_est,
            };

        // This selector only chooses geometry for the single-predictor
        // `RemlState` engine. Distributional GAMLSS and survival families do
        // not route here; they use the custom-family exact-Newton engine.
        if self.config.firth_bias_reduction
            && !matches!(self.config.link_function(), LinkFunction::Logit)
        {
            return dense_backend("firth_non_logit", None, None);
        }
        if p < 256 {
            return dense_backend("p_below_threshold", None, None);
        }
        if has_dense_constraints {
            return dense_backend("constraints_present", None, None);
        }
        let Some(x_sparse) = x_sparse else {
            return dense_backend("design_not_sparse", None, None);
        };
        let Some(blocks) = self.sparse_penalty_blocks.as_ref() else {
            return dense_backend("penalty_blocks_not_separable", None, None);
        };

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, s_k) in self.s_full_list.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                s_lambda.scaled_add(lambdas[k], s_k);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        match workspace.sparse_penalized_system_stats(x_sparse, &s_lambda) {
            Ok(stats) if stats.density_upper < 0.10 && !blocks.is_empty() => SparseRemlDecision {
                geometry: RemlGeometry::SparseExactSpd,
                reason: "sparse_exact_spd",
                p,
                nnz_x,
                nnz_h_upper_est: Some(stats.nnz_h_upper),
                density_h_upper_est: Some(stats.density_upper),
            },
            Ok(stats) => dense_backend(
                "penalized_hessian_too_dense",
                Some(stats.nnz_h_upper),
                Some(stats.density_upper),
            ),
            Err(_) => dense_backend("sparse_stats_failed", None, None),
        }
    }
}
