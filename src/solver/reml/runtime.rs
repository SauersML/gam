use std::sync::Arc;
use super::*;
use crate::faer_ndarray::FaerLblt;
use crate::linalg::utils::{
    StableSolver, boundary_hit_indices, symmetric_spectrum_condition_number,
};
use crate::pirls::PirlsWorkspace;
use crate::types::{InverseLink, LinkFunction, SasLinkState};

impl<'a> RemlState<'a> {
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

    pub(crate) fn last_ridge_used(&self) -> Option<f64> {
        self.cache_manager
            .current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .map(|bundle| bundle.ridge_passport.delta)
    }

    pub(crate) fn third_derivative_projection_from_design(
        &self,
        design: &DesignMatrix,
        d_vec: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if let Some(x_sparse) = design.as_sparse() {
            let mut out = Array1::<f64>::zeros(x_sparse.ncols());
            let (symbolic, values) = x_sparse.as_ref().parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            for col in 0..x_sparse.ncols() {
                let mut acc = 0.0;
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    let xij = values[ptr];
                    acc += d_vec[row_idx[ptr]] * xij * xij * xij;
                }
                out[col] = acc;
            }
            Ok(out)
        } else {
            let x_dense = design.as_dense().ok_or_else(|| {
                EstimationError::InvalidInput("design matrix should be dense or sparse".to_string())
            })?;
            let mut out = Array1::<f64>::zeros(x_dense.ncols());
            for j in 0..x_dense.ncols() {
                let mut acc = 0.0;
                for i in 0..x_dense.nrows() {
                    let xij = x_dense[[i, j]];
                    acc += d_vec[i] * xij * xij * xij;
                }
                out[j] = acc;
            }
            Ok(out)
        }
    }

    /// Compute the Tierney-Kadane correction to the Laplace approximation.
    ///
    /// The TK correction value is the third-order Bartlett refinement:
    ///
    /// ```text
    /// TK = -(1/6) Σ_j (H⁻¹)_jj³ · d3_j
    /// ```
    ///
    /// where H = H_pen is the penalized Hessian and d3_j = Σ_i c_i · X_ij³
    /// is the third-derivative projection of the negative log-likelihood.
    ///
    /// When `gradient_info` is `Some((penalty_roots, lambdas))`, also computes
    /// the TK gradient using a basis-invariant trace formula:
    ///
    /// ```text
    /// ∂V_TK/∂ρ_k = -½ tr(H_obs⁻¹ A_k^obs)
    /// ```
    ///
    /// where H_obs = X^T W X (observation Hessian without penalty) and
    /// A_k^obs = X^T diag(c ⊙ dη_k) X with dη_k = -X v_k the mode
    /// sensitivity direction (v_k = H_pen⁻¹(λ_k S_k β̂)).
    ///
    /// The main REML gradient already includes ½ tr(H_pen⁻¹ A_k^pen), so
    /// this TK gradient adds only the observation-Hessian part that the
    /// main gradient does not cover.
    ///
    /// Computing H_obs⁻¹ exactly requires a separate factorization of the
    /// (potentially rank-deficient) observation Hessian. We use the
    /// approximation H_obs⁻¹ ≈ H_pen⁻¹, which is valid when the penalty
    /// is small relative to the data information (well-identified regime).
    /// This gives:
    ///
    /// ```text
    /// ∂V_TK/∂ρ_k ≈ -½ tr(H_pen⁻¹ A_k^obs)
    ///             = ½ Σ_i c_i · (X v_k)_i · hat_ii
    /// ```
    ///
    /// where hat_ii = (X H_pen⁻¹ X^T)_{ii} are the leverages.
    ///
    /// NOTE: This is an approximation. The exact formula requires
    /// factorizing H_obs = X^T W X separately. For well-conditioned
    /// problems with moderate penalization the error is O(||S||/||H_obs||).
    fn tierney_kadane_laml_correction(
        &self,
        pirls_result: &PirlsResult,
        h_eff_eval: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        gradient_info: Option<(&[Array2<f64>], &[f64])>,
        beta: &Array1<f64>,
    ) -> Result<(f64, Option<Array1<f64>>), EstimationError> {
        let mut d_vec = pirls_result.solve_c_array.clone();
        if d_vec.is_empty() {
            return Ok((
                0.0,
                gradient_info.map(|(roots, _)| Array1::zeros(roots.len())),
            ));
        }
        for val in &mut d_vec {
            if !val.is_finite() {
                *val = 0.0;
            }
        }

        let p_eff = h_eff_eval.ncols();
        if p_eff == 0 {
            return Ok((
                0.0,
                gradient_info.map(|(roots, _)| Array1::zeros(roots.len())),
            ));
        }

        // Compute the third-derivative projection for the TK value.
        let third_deriv = if let Some(z) = free_basis_opt {
            let mut out = Array1::<f64>::zeros(z.ncols());
            for j in 0..z.ncols() {
                let xz_col = pirls_result
                    .x_transformed
                    .matrixvectormultiply(&z.column(j).to_owned());
                out[j] = d_vec
                    .iter()
                    .zip(xz_col.iter())
                    .map(|(&d, &x)| d * x * x * x)
                    .sum();
            }
            out
        } else {
            self.third_derivative_projection_from_design(&pirls_result.x_transformed, &d_vec)?
        };

        let num_penalties = gradient_info.map_or(0, |(roots, _)| roots.len());
        let compute_gradient = gradient_info.is_some();

        // The effective design matrix: X_eff = X Z when projected, else X.
        // Needed for computing hat leverages and mode response projections.
        let x_eff_dense = if compute_gradient {
            if let Some(z) = free_basis_opt {
                Some(Arc::new(pirls_result.x_transformed.to_dense().dot(z)))
            } else {
                Some(pirls_result.x_transformed.to_dense_arc())
            }
        } else {
            None
        };

        let n_obs = d_vec.len();
        let mut h_inv_diag = Array1::<f64>::zeros(p_eff);
        // hat_leverages[i] = (X_eff H_pen⁻¹ X_eff^T)_{ii}, accumulated
        // column-by-column during the H⁻¹ diagonal computation.
        let mut hat_leverages = if compute_gradient {
            Array1::<f64>::zeros(n_obs)
        } else {
            Array1::<f64>::zeros(0)
        };

        if let Ok(chol) = h_eff_eval.cholesky(Side::Lower) {
            for j in 0..p_eff {
                let mut e_j = Array1::<f64>::zeros(p_eff);
                e_j[j] = 1.0;
                let f_j = chol.solvevec(&e_j);
                h_inv_diag[j] = f_j[j];

                // Accumulate hat leverage: hat[i] += X_eff[i,j] * (X_eff f_j)[i]
                if compute_gradient {
                    let x_dense = x_eff_dense.as_ref().unwrap();
                    let x_fj: Array1<f64> = x_dense.dot(&f_j);
                    for i in 0..n_obs {
                        hat_leverages[i] += x_dense[[i, j]] * x_fj[i];
                    }
                }
            }

            // Compute the trace-based TK gradient using the Cholesky factor.
            if compute_gradient {
                let (roots, lambdas) = gradient_info.unwrap();
                let x_dense = x_eff_dense.as_ref().unwrap();
                let mut tk_grad = Array1::<f64>::zeros(num_penalties);

                for k in 0..num_penalties {
                    // Mode sensitivity: v_k = H_pen⁻¹(λ_k S_k β̂)
                    let r_beta = roots[k].dot(beta);
                    let s_k_beta = fast_atv(&roots[k], &r_beta);
                    let a_k_beta = &s_k_beta * lambdas[k];
                    let v_k = chol.solvevec(&a_k_beta);

                    // X v_k (n-vector): the predicted change in η per unit ρ_k
                    let x_v_k: Array1<f64> = x_dense.dot(&v_k);

                    // TK gradient[k] = ½ Σ_i c_i · (X v_k)_i · hat_ii
                    // (derived from -½ tr(H_obs⁻¹ A_k^obs) with H_obs⁻¹ ≈ H_pen⁻¹)
                    let mut acc = 0.0;
                    for i in 0..n_obs {
                        acc += d_vec[i] * x_v_k[i] * hat_leverages[i];
                    }
                    tk_grad[k] = 0.5 * acc;
                }

                // Zero out non-finite entries for robustness.
                for v in tk_grad.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }

                let correction = -h_inv_diag
                    .iter()
                    .zip(third_deriv.iter())
                    .map(|(&hjj, &d3)| hjj * hjj * hjj * d3)
                    .sum::<f64>()
                    / 6.0;
                let correction = if correction.is_finite() {
                    correction
                } else {
                    0.0
                };
                return Ok((correction, Some(tk_grad)));
            }
        } else {
            let (evals, evecs) = h_eff_eval
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let floor = 1e-12;

            for j in 0..p_eff {
                let mut acc = 0.0;
                for m in 0..evals.len() {
                    let ev = evals[m];
                    if ev > floor {
                        let u = evecs[[j, m]];
                        acc += (u * u) / ev;
                    }
                }
                h_inv_diag[j] = acc;

                // Accumulate hat leverage from eigendecomposition path.
                if compute_gradient {
                    // Reconstruct f_j = H⁻¹ e_j for hat leverage accumulation.
                    let mut f_j = Array1::<f64>::zeros(p_eff);
                    for m in 0..evals.len() {
                        let ev = evals[m];
                        if ev > floor {
                            let u_jm = evecs[[j, m]];
                            let scale = u_jm / ev;
                            for ii in 0..p_eff {
                                f_j[ii] += scale * evecs[[ii, m]];
                            }
                        }
                    }
                    let x_dense = x_eff_dense.as_ref().unwrap();
                    let x_fj: Array1<f64> = x_dense.dot(&f_j);
                    for i in 0..n_obs {
                        hat_leverages[i] += x_dense[[i, j]] * x_fj[i];
                    }
                }
            }

            // Compute the trace-based TK gradient using the eigendecomposition.
            if compute_gradient {
                let (roots, lambdas) = gradient_info.unwrap();
                let x_dense = x_eff_dense.as_ref().unwrap();
                let mut tk_grad = Array1::<f64>::zeros(num_penalties);

                // Reconstruct H_pen⁻¹ for solving v_k.
                let mut h_inv = Array2::<f64>::zeros((p_eff, p_eff));
                for m in 0..evals.len() {
                    let ev = evals[m];
                    if ev > floor {
                        let col = evecs.column(m);
                        for i in 0..p_eff {
                            for j in 0..p_eff {
                                h_inv[[i, j]] += col[i] * col[j] / ev;
                            }
                        }
                    }
                }

                for k in 0..num_penalties {
                    let r_beta = roots[k].dot(beta);
                    let s_k_beta = fast_atv(&roots[k], &r_beta);
                    let a_k_beta = &s_k_beta * lambdas[k];
                    let v_k = h_inv.dot(&a_k_beta);

                    let x_v_k: Array1<f64> = x_dense.dot(&v_k);

                    let mut acc = 0.0;
                    for i in 0..n_obs {
                        acc += d_vec[i] * x_v_k[i] * hat_leverages[i];
                    }
                    tk_grad[k] = 0.5 * acc;
                }

                for v in tk_grad.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }

                let correction = -h_inv_diag
                    .iter()
                    .zip(third_deriv.iter())
                    .map(|(&hjj, &d3)| hjj * hjj * hjj * d3)
                    .sum::<f64>()
                    / 6.0;
                let correction = if correction.is_finite() {
                    correction
                } else {
                    0.0
                };
                return Ok((correction, Some(tk_grad)));
            }
        }

        // Value-only path (no gradient requested).
        let correction = -h_inv_diag
            .iter()
            .zip(third_deriv.iter())
            .map(|(&hjj, &d3)| hjj * hjj * hjj * d3)
            .sum::<f64>()
            / 6.0;

        let correction = if correction.is_finite() {
            correction
        } else {
            0.0
        };
        Ok((correction, None))
    }

    pub(super) fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
        // Keep expensive diagnostics out of the hot path unless they can
        // be surfaced. This has zero effect on optimization math.
        (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
            && (eval_idx == 1 || eval_idx % 200 == 0)
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

    /// Returns the eta-derivative carriers (c = dW/deta, d = d^2W/deta^2) for
    /// the exact Hessian surface that PIRLS accepted at the mode.
    ///
    /// For canonical links these coincide with the Fisher arrays; for
    /// non-canonical links they carry the clamped **observed-information**
    /// curvature used on the PIRLS left-hand side, including the residual-
    /// dependent corrections:
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// These flow into the outer REML gradient's C[v] correction term and
    /// the outer Hessian's Q[v_k, v_l] correction term. Using the observed
    /// versions is required for the exact Laplace approximation; Fisher
    /// versions would yield a PQL-type surrogate. See response.md Section 3.
    fn hessian_cd_arrays(
        &self,
        pirls_result: &PirlsResult,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        Ok((
            pirls_result.solve_c_array.clone(),
            pirls_result.solve_d_array.clone(),
        ))
    }

    /// Compute soft prior cost without needing workspace
    pub(super) fn compute_soft_priorcost(&self, rho: &Array1<f64>) -> f64 {
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
    pub(super) fn compute_soft_priorgrad(&self, rho: &Array1<f64>) -> Array1<f64> {
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
    ///   a = rho_soft_prior_sharpness / rho_bound,
    ///   w = rho_soft_prior_weight.
    ///
    /// Then:
    ///   dC_i/drho_i   = w * a * tanh(a * rho_i),
    ///   d²C_i/drho_i² = w * a² * sech²(a * rho_i)
    ///                = w * a² * (1 - tanh²(a * rho_i)).
    ///
    /// The prior is separable across coordinates, so off-diagonals are zero.
    pub(super) fn add_soft_priorhessian_in_place(&self, rho: &Array1<f64>, hess: &mut Array2<f64>) {
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

    /// Compute the soft prior Hessian as a standalone matrix.
    ///
    /// This is the diagonal matrix of second derivatives of the soft prior penalty.
    /// Returns `None` when the prior contributes zero curvature (empty ρ or zero weight).
    pub(super) fn compute_soft_priorhess(&self, rho: &Array1<f64>) -> Option<Array2<f64>> {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return None;
        }
        let mut hess = Array2::<f64>::zeros((len, len));
        self.add_soft_priorhessian_in_place(rho, &mut hess);
        if hess.iter().any(|&v| v != 0.0) {
            Some(hess)
        } else {
            None
        }
    }

    /// Returns the effective Hessian and the ridge value used (if any).
    /// Uses the same Hessian matrix in both cost and gradient calculations.
    ///
    /// PIRLS folds any stabilization ridge directly into the penalized objective:
    ///   l_p(beta; rho) = l(beta) - 0.5 * beta^T (S_lambda + ridge I) beta.
    /// Therefore the curvature used in LAML is
    ///   H_eff = X' W_H X + S_lambda + ridge I,
    /// and adding another ridge here places the Laplace expansion on a different surface.
    ///
    /// IMPORTANT: W_H here is the **observed-information** weight for non-canonical
    /// links (or Fisher weight for canonical links, where the two coincide). The
    /// `stabilizedhessian_transformed` from PIRLS was assembled with the Hessian-side
    /// weights (`lasthessian_weights`), which are observed when PIRLS accepted that
    /// curvature. This ensures log|H_eff| in the outer REML uses the correct Laplace
    /// Hessian. See response.md Section 3 for the mathematical justification.
    ///
    pub(super) fn effectivehessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        let base = pr.stabilizedhessian_transformed.clone();

        if base.cholesky(Side::Lower).is_ok() {
            return Ok((base, pr.ridge_passport));
        }

        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }

    pub(crate) fn newwith_offset<X>(
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
        Self::newwith_offset_shared(
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

    pub(crate) fn newwith_offset_shared<X>(
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
            penalty_shrinkage_floor: None,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(),
            warm_start_beta: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
        })
    }

    /// Sets the shrinkage floor for penalized block eigenvalues.
    /// The ridge magnitude will be `epsilon * max_balanced_eigenvalue` (rho-independent).
    /// This prevents barely-penalized directions from causing pathological
    /// non-Gaussianity in the posterior. Typical value: `Some(1e-6)`.
    pub(crate) fn set_penalty_shrinkage_floor(&mut self, floor: Option<f64>) {
        self.penalty_shrinkage_floor = floor;
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure consistency in caching.
    pub(super) fn rhokey_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
        EvalCacheManager::sanitized_rhokey(rho)
    }

    pub(super) fn prepare_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let decision = self.select_reml_geometry(rho);
        match decision.geometry {
            RemlGeometry::SparseExactSpd => {
                match self.prepare_sparse_eval_bundlewithkey(rho, key.clone()) {
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
                        self.prepare_dense_eval_bundlewithkey(rho, key)
                    }
                }
            }
            RemlGeometry::DenseSpectral => self.prepare_dense_eval_bundlewithkey(rho, key),
        }
    }

    pub(crate) fn obtain_eval_bundle(
        &self,
        rho: &Array1<f64>,
    ) -> Result<EvalShared, EstimationError> {
        let key = self.rhokey_sanitized(rho);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundlewithkey(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    pub(crate) fn objective_innerhessian(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        Ok(bundle.h_total.as_ref().clone())
    }


    pub(super) fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
        let lin = pr.linear_constraints_transformed.as_ref()?;
        let active_tol = 1e-8;
        let beta_t = pr.beta_transformed.as_ref();
        let mut activerows: Vec<Array1<f64>> = Vec::new();
        for i in 0..lin.a.nrows() {
            let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
            if slack <= active_tol {
                activerows.push(lin.a.row(i).to_owned());
            }
        }
        if activerows.is_empty() {
            return None;
        }

        let p_t = lin.a.ncols();
        let mut a_t = Array2::<f64>::zeros((p_t, activerows.len()));
        for (j, row) in activerows.iter().enumerate() {
            for k in 0..p_t {
                a_t[[k, j]] = row[k];
            }
        }

        let qrow = Self::orthonormalize_columns(&a_t, 1e-10); // basis for active row-space^T
        let rank = qrow.ncols();
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
                let qt = qrow.column(t);
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

    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds (β_j ≥ b_i).
    ///
    /// Delegates to `BarrierConfig::from_constraints` and logs a diagnostic
    /// barrier-curvature check at a test point near the bounds.
    fn barrier_config_from_constraints(
        constraints: &crate::pirls::LinearInequalityConstraints,
    ) -> Option<super::unified::BarrierConfig> {
        let config = super::unified::BarrierConfig::from_constraints(Some(constraints))?;
        // Diagnostic: check curvature significance at a test point near bounds.
        {
            let max_idx = config
                .constrained_indices
                .iter()
                .max()
                .copied()
                .unwrap_or(0);
            let mut beta_test = Array1::<f64>::zeros(max_idx + 1);
            for (&idx, &lb) in config
                .constrained_indices
                .iter()
                .zip(config.lower_bounds.iter())
            {
                beta_test[idx] = lb + 0.01; // slightly above constraint
            }
            let significant = config.barrier_curvature_is_significant(&beta_test, 1.0, 0.05);
            log::trace!(
                "[barrier] curvature significant={significant} (tau={:.2e}, n_constrained={})",
                config.tau,
                config.constrained_indices.len(),
            );
        }
        Some(config)
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
            let mut worstrow_msg = String::new();
            if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                let mut worst = 0.0_f64;
                let mut worstrow = 0usize;
                for i in 0..lin.a.nrows() {
                    let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                    let viol = (-slack).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worstrow = i;
                    }
                }
                if worst > 0.0 {
                    worstrow_msg = format!("; worstrow={} worstviolation={:.3e}", worstrow, worst);
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
                worstrow_msg
            )));
        }
        Ok(())
    }

    pub(super) fn projectwith_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let zt_m = z.t().dot(matrix);
        zt_m.dot(z)
    }

    /// Compute the structural penalty rank and smooth pseudo-logdet L_δ(S).
    ///
    /// Uses the smooth δ-regularized pseudo-logdet:
    ///   L_δ(S) = log det(S + δI) − m₀ log δ
    /// where m₀ is the total penalty nullity from `self.nullspace_dims`.
    ///
    /// This replaces the hard ε-threshold truncation, making the objective C∞
    /// in outer parameters θ and eliminating artificial kinks when eigenvalues
    /// of S(ρ) cross the threshold.
    ///
    /// Reference: response.md Section 7.
    pub(super) fn fixed_subspace_penalty_rank_and_logdet(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(usize, f64), EstimationError> {
        let structural_rank = e_transformed.nrows().min(e_transformed.ncols());
        if structural_rank == 0 {
            return Ok((0, 0.0));
        }

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

        // Total penalty nullity m₀ from the known structural nullspace dimensions.
        // This is the number of eigenvalues that are structurally zero (e.g.,
        // polynomial nullspace for Duchon/TPS, differencing order for B-splines).
        let total_nullity: usize = self.nullspace_dims.iter().copied().max().unwrap_or(0);
        // Effective nullity: p_dim − structural_rank gives the count of truncated
        // directions from the reparameterization. Use the larger of structural
        // nullity and the known family nullity.
        let p_dim = evals.len();
        let reparam_nullity = p_dim.saturating_sub(structural_rank);
        let m0 = total_nullity.max(reparam_nullity);

        // Choose δ proportional to machine epsilon × spectral scale.
        let delta = super::unified::smooth_logdet_delta(evals.as_slice().unwrap());

        // L_δ(S) = log det(S + δI) − m₀ log δ
        let log_det = super::unified::smooth_pseudo_logdet(evals.as_slice().unwrap(), m0, delta);

        Ok((structural_rank, log_det))
    }

    /// Compute tr((S + δI)⁻¹ S_direction) — the first derivative of L_δ(S)
    /// in the direction S_direction.
    ///
    /// Uses the smooth δ-regularized inverse (S + δI)⁻¹ instead of the
    /// hard-truncated pseudoinverse S₊⁻¹. Because (S + δI) is always full rank,
    /// all eigenvectors participate and no eigenspace partitioning is needed.
    ///
    /// Reference: response.md Section 7.
    pub(super) fn fixed_subspace_penalty_trace(
        &self,
        e_transformed: &Array2<f64>,
        s_direction: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<f64, EstimationError> {
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

        // δ-regularized trace: tr((S + δI)⁻¹ S_direction).
        //
        // Every eigenvalue is shifted by δ, making (S + δI) full rank.
        // No eigenspace partitioning, no nullspace/positive split.
        let delta = super::unified::smooth_logdet_delta(evals.as_slice().unwrap());

        let mut trace = 0.0;
        for idx in 0..p_dim {
            let ev = evals[idx];
            let u = evecs.column(idx).to_owned();
            let spsi_u = s_direction.dot(&u);
            trace += u.dot(&spsi_u) / (ev + delta);
        }
        Ok(trace)
    }

    pub(super) fn updatewarm_start_from(&self, pr: &PirlsResult) {
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

    pub(crate) fn setwarm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
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
    pub fn clearwarm_start(&self) {
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
        let stable_solver = StableSolver::new("reml hessian");
        loop {
            let ridge = planner.ridge();
            let regularized = addridge(h, ridge);
            if let Ok(factor) = stable_solver.factorize(&regularized) {
                return match factor {
                    crate::faer_ndarray::FaerSymmetricFactor::Llt(inner) => FaerFactor::Llt(inner),
                    crate::faer_ndarray::FaerSymmetricFactor::Ldlt(inner) => {
                        FaerFactor::Ldlt(inner)
                    }
                    crate::faer_ndarray::FaerSymmetricFactor::Lblt(inner) => {
                        FaerFactor::Lblt(inner)
                    }
                };
            }
            if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                let view = FaerArrayView::new(&regularized);
                let f = FaerLblt::new(view.as_ref(), Side::Lower);
                return FaerFactor::Lblt(f);
            }
            planner.bumpwith_matrix(h);
        }
    }

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    /// Compute the smooth δ-regularized pseudo-logdet for the sparse penalty path.
    ///
    /// For each sparse penalty block k with eigenvalues {σ_1, ..., σ_r, 0, ..., 0},
    /// the smooth logdet of λ_k S_k is:
    ///
    ///   L_δ(λ_k S_k) = Σ_i log(λ_k σ_i + δ) − m₀ log δ
    ///
    /// where m₀ = p_block − rank (the known nullity).
    ///
    /// The first derivative is:
    ///   ∂/∂ρ_k L_δ = Σ_i λ_k σ_i / (λ_k σ_i + δ)
    ///
    /// As δ → 0, this recovers the exact formula: rank * rho_k + Σ log(σ_i).
    ///
    /// Reference: response.md Section 7.
    pub(super) fn sparse_penalty_logdet_runtime(
        &self,
        rho: &Array1<f64>,
        blocks: &[SparsePenaltyBlock],
    ) -> (f64, Array1<f64>) {
        let mut logdet = 0.0_f64;
        let mut det1 = Array1::<f64>::zeros(rho.len());
        for block in blocks {
            let lambda_k = rho[block.term_index].exp();

            // Choose δ based on the spectral scale of λ_k S_k.
            let max_ev = block
                .positive_eigenvalues
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
                * lambda_k;
            let delta = super::unified::SMOOTH_LOGDET_DELTA_SCALE * max_ev.max(1.0);

            // L_δ(λ_k S_k) = Σ_i log(λ_k σ_i + δ) + m₀ log(δ) − m₀ log δ
            //               = Σ_i log(λ_k σ_i + δ) [null eigenvalues cancel]
            //
            // For the positive eigenvalues: log(λ_k σ_i + δ)
            // For the null eigenvalues: log(0 + δ) = log(δ), and the −m₀ log δ cancels.
            for &eig in block.positive_eigenvalues.iter() {
                logdet += (lambda_k * eig + delta).ln();
            }
            // Null eigenvalues contribute log(δ) each, cancelled by −m₀ log δ.
            // Net contribution from null eigenvalues: 0.

            // First derivative: ∂/∂ρ_k L_δ = Σ_i λ_k σ_i / (λ_k σ_i + δ).
            // (Since ∂/∂ρ_k (λ_k σ_i) = λ_k σ_i.)
            if block.term_index < det1.len() {
                let mut d1 = 0.0_f64;
                for &eig in block.positive_eigenvalues.iter() {
                    let shifted = lambda_k * eig + delta;
                    d1 += lambda_k * eig / shifted;
                }
                // Null eigenvalues contribute λ_k * 0 / δ = 0.
                det1[block.term_index] = d1;
            }
        }
        (logdet, det1)
    }

    pub(super) fn prepare_dense_eval_bundlewithkey(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        let (h_eff, ridge_passport) = self.effectivehessian(pirls_result.as_ref())?;

        const EIG_REL_THRESHOLD: f64 = 1e-10;
        const EIG_ABS_FLOOR: f64 = 1e-14;

        let dim = h_eff.nrows();
        let mut h_total = h_eff.clone();
        let mut firth_dense_operator: Option<Arc<FirthDenseOperator>> = None;
        if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let x_dense = pirls_result
                .x_transformed
                .try_to_dense_arc(
                    "dense REML eval bundle requires dense transformed design for Firth operator",
                )
                .map_err(EstimationError::InvalidInput)?;
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
            // The hphi block below is therefore the curvature of that
            // identifiable-subspace Jeffreys penalty, represented in the
            // current transformed basis.
            let mut weighted_xtdx = Array2::<f64>::zeros((0, 0));
            let diag_term = Self::xt_diag_x_dense_into(
                &firth_op.x_dense,
                &(&firth_op.w2 * &firth_op.h_diag),
                &mut weighted_xtdx,
            );
            let bpb = fast_ab(&firth_op.b_base.t().to_owned(), &firth_op.p_b_base);
            let mut hphi = 0.5 * (diag_term - bpb);
            // Numerical symmetry guard.
            for i in 0..hphi.nrows() {
                for j in 0..i {
                    let avg = 0.5 * (hphi[[i, j]] + hphi[[j, i]]);
                    hphi[[i, j]] = avg;
                    hphi[[j, i]] = avg;
                }
            }
            // Keep tiny numerical noise from making the solve surface less stable.
            if hphi.iter().all(|v| v.is_finite()) {
                h_total -= &hphi;
            }
            firth_dense_operator = Some(firth_op);
        }

        // Add log-barrier Hessian diagonal for monotonicity-constrained coefficients.
        // This augments the penalized Hessian before the spectral decomposition so
        // that logdet, trace, and solve operations all reflect the barrier curvature.
        if let Some(ref lin) = pirls_result.linear_constraints_transformed {
            if let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin) {
                let beta_t = pirls_result.beta_transformed.as_ref();
                if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut h_total, beta_t) {
                    log::warn!("Barrier Hessian diagonal skipped: {e}");
                }
            }
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
        // Active-subspace stability diagnostic for hard-truncated logdet
        // derivatives. The branch-local formulas used elsewhere assume a fixed
        // retained eigenspace; once retained and dropped eigenvalues crowd the
        // threshold, even first-order exactness is no longer justified at the
        // crossing and second-order updates are especially fragile.
        if log::log_enabled!(log::Level::Warn) {
            if active_subspace_unstable {
                let rel_gap = active_subspace_rel_gap.unwrap_or(f64::NAN);
                log::warn!(
                    "[REML] H_+ active-subspace is near a hard-threshold crossing: min_kept={:.3e}, max_dropped={:.3e}, threshold={:.3e}, rel_gap={:.3e}; exact truncated-logdet derivatives are branch-local only",
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
            active_subspace_rel_gap,
            active_subspace_unstable,
            sparse_exact: None,
            firth_dense_operator,
            firth_dense_operator_original: None,
        })
    }

    pub(super) fn prepare_sparse_eval_bundlewithkey(
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
        let x_sparse = self.x().as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse exact geometry requires sparse original design".to_string(),
            )
        })?;
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
        // Add log-barrier Hessian diagonal for monotonicity-constrained
        // coefficients (sparse path uses original coordinates).
        if let Some(ref lin) = self.linear_constraints {
            if let Some(barrier_cfg) = Self::barrier_config_from_constraints(lin) {
                let beta_orig = self.sparse_exact_beta_original(pirls_result.as_ref());
                if let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut s_lambda, &beta_orig)
                {
                    log::warn!("Sparse barrier Hessian diagonal skipped: {e}");
                }
            }
        }

        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &pirls_result.finalweights,
            &s_lambda,
            ridge_passport.delta,
        )?;
        let (logdet_s_pos, det1_values) =
            self.sparse_penalty_logdet_runtime(rho, penalty_blocks.as_ref());
        let firth_dense_operator_original = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let x_dense = self
                .x()
                .try_to_dense_arc(
                    "sparse exact REML runtime requires dense design for Firth operator",
                )
                .map_err(EstimationError::InvalidInput)?;
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
            active_subspace_rel_gap: None,
            active_subspace_unstable: false,
            sparse_exact: Some(Arc::new(SparseExactEvalData {
                factor: Arc::new(sparse_system.factor),
                logdet_h: sparse_system.logdet_h,
                logdet_s_pos,
                det1_values: Arc::new(det1_values),
            })),
            firth_dense_operator: None,
            firth_dense_operator_original,
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
        let key_opt = self.rhokey_sanitized(rho);
        if use_cache
            && let Some(key) = &key_opt
            && let Some(cached) = self.cache_manager.pirls_cache.write().unwrap().get(key)
        {
            // Do not overwrite the current warm start from cache hits.
            // Line search / multi-eval outer loops revisit older rho keys and
            // replacing a recent nearby beta with an older cached mode can
            // materially slow subsequent PIRLS convergence.
            if cached.cache_compacted {
                let mut pirls_config = self.config.as_pirls_config();
                pirls_config.link_kind =
                    if let Some(state) = self.runtime_mixture_link_state.clone() {
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
                return Ok(Arc::new(cached.rehydrate_after_reml_cache(
                    self.x(),
                    self.y,
                    self.weights,
                    self.offset.view(),
                    &pirls_config.link_kind,
                )?));
            }
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
            let problem = pirls::PirlsProblem {
                x: &self.x,
                offset: self.offset.view(),
                y: self.y,
                priorweights: self.weights,
                covariate_se: None,
            };
            let penalty = pirls::PenaltyConfig {
                rs_original: &self.rs_list,
                balanced_penalty_root: Some(&self.balanced_penalty_root),
                reparam_invariant: Some(&self.reparam_invariant),
                p: self.p,
                coefficient_lower_bounds: self.coefficient_lower_bounds.as_ref(),
                linear_constraints_original: self.linear_constraints.as_ref(),
                penalty_shrinkage_floor: self.penalty_shrinkage_floor,
            };
            pirls::fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                problem,
                penalty,
                &pirls_config,
                warm_start_ref,
            )
        };

        if let Err(e) = &pirls_result {
            log::warn!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
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
                self.updatewarm_start_from(pirls_result.as_ref());
                // This is a successful fit. Cache only if key is valid (not NaN).
                if use_cache && let Some(key) = key_opt {
                    self.cache_manager
                        .pirls_cache
                        .write()
                        .unwrap()
                        .insert(key, Arc::new(pirls_result.compact_for_reml_cache()));
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
                if pirls_result.lastgradient_norm > acceptable_kkt {
                    // The fit timed out and gradient is still too large for reliable outer
                    // derivatives, so fail fast and let the caller backtrack.
                    log::error!(
                        "P-IRLS failed convergence check: gradient norm {:.3e} > {:.3e} (iter {})",
                        pirls_result.lastgradient_norm,
                        acceptable_kkt,
                        pirls_result.iteration
                    );
                    Err(EstimationError::PirlsDidNotConverge {
                        max_iterations: pirls_result.iteration,
                        last_change: pirls_result.lastgradient_norm,
                    })
                } else {
                    // Near-stationary despite max iterations; treat as usable.
                    log::warn!(
                        "P-IRLS reached max iterations but gradient norm {:.3e} <= {:.3e}; accepting near-stationary fit.",
                        pirls_result.lastgradient_norm,
                        acceptable_kkt
                    );
                    self.updatewarm_start_from(pirls_result.as_ref());
                    Ok(pirls_result)
                }
            }
        }
    }
}

impl<'a> RemlState<'a> {
    /// Compute the objective function for BFGS optimization.
    ///
    /// Full objective reference.
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
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                // Other errors still bubble up
                // Provide bounds diagnostics here too
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Err(e);
            }
        };
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let result =
                self.evaluate_unified_sparse(p, &bundle, super::unified::EvalMode::ValueOnly)?;
            return Ok(result.cost);
        }
        {
            // Validation and diagnostics (before delegating to unified evaluator).
            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;

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

            const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
            let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
            if ridge_used > 0.0
                && want_hot_diag
                && let Ok((eigs, _)) = pirls_result.penalized_hessian_transformed.eigh(Side::Lower)
                && let Some(min_eig) = eigs.iter().cloned().reduce(f64::min)
            {
                if should_emit_h_min_eig_diag(min_eig) {
                    log::debug!(
                        "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                        min_eig,
                        ridge_used
                    );
                }
                if min_eig <= 0.0 {
                    log::warn!(
                        "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                        ridge_used
                    );
                }
                if !min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE {
                    let condition_number = symmetric_spectrum_condition_number(
                        &pirls_result.penalized_hessian_transformed,
                    );
                    log::warn!(
                        "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                        condition_number
                    );
                }
            }
        }
        // Delegate to the unified evaluator for the actual formula computation.
        // This ensures cost and gradient share the exact same formula.
        let result = self.evaluate_unified(p, &bundle, super::unified::EvalMode::ValueOnly)?;
        Ok(result.cost)
    }

    ///
    /// Exact non-Laplace evidence identities (reference comments; not runtime path)
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
    /// Full outer-derivative reference (exact system, sign convention used here).
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
    /// Holds the family-dependent components of the REML/LAML evaluation
    /// that are shared across dense path variants (`evaluate_unified`,
    /// `evaluate_unified_with_ext_from_bundle`, `evaluate_unified_for_efs`).
    /// Extracting these into a struct eliminates ~100 lines of duplication
    /// across the three dense-path methods.

    /// Build the derivative provider, dispersion handling, TK correction,
    /// Firth log-det, log-likelihood, Firth operator, and barrier config
    /// that are common to all dense evaluation paths.
    ///
    /// # Arguments
    /// - `pirls_result`: the inner-loop solution
    /// - `bundle`: the evaluation bundle (carries h_eff, firth operator, etc.)
    /// - `mode`: evaluation mode; controls whether TK gradient is computed
    /// - `rho`: log-smoothing parameters (needed for TK gradient lambdas)
    /// - `free_basis_opt`: optional constraint-projection basis
    /// - `penalty_roots`: penalty matrix square-roots (possibly projected)
    /// - `beta`: coefficient vector in the operator's basis
    /// - `include_firth_derivs`: whether to wrap the derivative provider with
    ///   Firth-aware derivatives (false for the EFS path which does not need
    ///   Firth corrections in trace-based updates)
    fn build_dense_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        rho: &Array1<f64>,
        free_basis_opt: &Option<Array2<f64>>,
        penalty_roots: &[Array2<f64>],
        beta: &Array1<f64>,
        include_firth_derivs: bool,
    ) -> Result<
        (
            Box<dyn super::unified::HessianDerivativeProvider>,
            super::unified::DispersionHandling,
            f64,                                               // tk_correction
            Option<Array1<f64>>,                               // tk_gradient
            f64,                                               // firth_logdet
            f64,                                               // log_likelihood
            Option<std::sync::Arc<super::FirthDenseOperator>>, // firth_op
            Option<super::unified::BarrierConfig>,
        ),
        EstimationError,
    > {
        use super::unified::{
            DispersionHandling, GaussianDerivatives, SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;

        // Derivative provider.
        let firth_active_for_derivs = include_firth_derivs
            && self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
            && free_basis_opt.is_none();
        let deriv_provider: Box<dyn super::unified::HessianDerivativeProvider> =
            if is_gaussian_identity {
                Box::new(GaussianDerivatives)
            } else {
                // Differentiate the same Hessian-side curvature arrays that the
                // inner PIRLS solve accepted at the mode.
                let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
                let x_transformed = if let Some(z) = free_basis_opt.as_ref() {
                    // Project the design: X_proj = X Z
                    let x_dense = pirls_result.x_transformed.to_dense();
                    crate::linalg::matrix::DesignMatrix::Dense(Arc::new(x_dense.dot(z)))
                } else {
                    pirls_result.x_transformed.clone()
                };
                let base = SinglePredictorGlmDerivatives {
                    c_array,
                    d_array: Some(d_array),
                    x_transformed,
                };
                if firth_active_for_derivs {
                    if let Some(firth_op) = bundle.firth_dense_operator.clone() {
                        Box::new(super::unified::FirthAwareGlmDerivatives { base, firth_op })
                    } else {
                        Box::new(base)
                    }
                } else {
                    Box::new(base)
                }
            };

        // Dispersion handling.
        let dispersion = if is_gaussian_identity {
            DispersionHandling::ProfiledGaussian
        } else {
            DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            }
        };

        // TK correction and gradient (non-Gaussian only).
        let (tk_correction, tk_gradient) = if !is_gaussian_identity {
            let h_eff_eval = if let Some(z) = free_basis_opt.as_ref() {
                Self::projectwith_basis(bundle.h_eff.as_ref(), z)
            } else {
                bundle.h_eff.as_ref().clone()
            };
            let grad_info = if mode != super::unified::EvalMode::ValueOnly {
                let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
                Some((penalty_roots, lambdas))
            } else {
                None
            };
            let grad_ref = grad_info
                .as_ref()
                .map(|(roots, lams)| (*roots as &[Array2<f64>], lams.as_slice()));
            self.tierney_kadane_laml_correction(
                pirls_result,
                &h_eff_eval,
                free_basis_opt.as_ref(),
                grad_ref,
                beta,
            )?
        } else {
            (0.0, None)
        };

        // Firth log-det.
        let firth_logdet = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            pirls_result.firth_log_det().unwrap_or(0.0)
        } else {
            0.0
        };

        // Log-likelihood: for both Gaussian and GLM, deviance = -2 * log_likelihood.
        let log_likelihood = -0.5 * pirls_result.deviance;

        // Firth operator: passed through to the unified evaluator which computes
        // the Firth gradient and Hessian internally. Only active when Firth is
        // enabled AND no constraint projection (the FirthDenseOperator is built
        // in the unprojected reparameterized basis).
        let firth_op = if firth_logdet != 0.0 && free_basis_opt.is_none() {
            bundle.firth_dense_operator.clone()
        } else {
            None
        };

        // Construct barrier config for monotonicity constraints when no
        // active-set projection is in effect (barrier indices are in the
        // full transformed-coefficient space).
        let barrier_config = if free_basis_opt.is_none() {
            pirls_result
                .linear_constraints_transformed
                .as_ref()
                .and_then(Self::barrier_config_from_constraints)
        } else {
            None
        };

        Ok((
            deriv_provider,
            dispersion,
            tk_correction,
            tk_gradient,
            firth_logdet,
            log_likelihood,
            firth_op,
            barrier_config,
        ))
    }

    /// Build the dispersion handling, derivative provider, Firth log-det,
    /// log-likelihood, Firth operator, and barrier config that are common
    /// to both sparse evaluation paths (`evaluate_unified_sparse` and
    /// `evaluate_unified_sparse_for_efs`).
    ///
    /// TK correction is NOT included here because the sparse paths compute
    /// it via sparse Cholesky solves with completely different logic.
    fn build_sparse_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
    ) -> Result<
        (
            super::unified::DispersionHandling,
            Box<dyn super::unified::HessianDerivativeProvider>,
            f64,                                               // firth_logdet
            f64,                                               // log_likelihood
            Option<std::sync::Arc<super::FirthDenseOperator>>, // firth_op
            Option<super::unified::BarrierConfig>,
        ),
        EstimationError,
    > {
        use super::unified::{
            DispersionHandling, FirthAwareGlmDerivatives, GaussianDerivatives,
            SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;

        // Firth log-det and operator. Sparse exact still uses the same dense
        // Fisher-information reduction as the dense backend; only the H^{-1}
        // applications move to the sparse Cholesky operator.
        let firth_logdet = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            pirls_result.firth_log_det().unwrap_or(0.0)
        } else {
            0.0
        };
        let firth_op = if firth_logdet != 0.0 {
            if let Some(cached) = bundle.firth_dense_operator_original.clone() {
                Some(cached)
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(std::sync::Arc::new(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                )?))
            }
        } else {
            None
        };

        // Dispersion and derivative provider depend on family.
        let (dispersion, deriv_provider): (_, Box<dyn super::unified::HessianDerivativeProvider>) =
            if is_gaussian_identity {
                (
                    DispersionHandling::ProfiledGaussian,
                    Box::new(GaussianDerivatives),
                )
            } else {
                // Differentiate the same Hessian-side curvature arrays that the
                // inner PIRLS solve accepted at the mode.
                let (c_array, d_array) = self.hessian_cd_arrays(pirls_result)?;
                (
                    DispersionHandling::Fixed {
                        phi: 1.0,
                        include_logdet_h: true,
                        include_logdet_s: true,
                    },
                    {
                        let base = SinglePredictorGlmDerivatives {
                            c_array,
                            d_array: Some(d_array),
                            x_transformed: self.x().clone(),
                        };
                        // Match the dense exact path: when Firth-logit is
                        // active, the directional log|H_total| drift includes
                        // -D(H_phi)[B_k]. The sparse backend differs only in how
                        // B_k = H^{-1} rhs is solved.
                        if let Some(firth_op) = firth_op.clone() {
                            Box::new(FirthAwareGlmDerivatives { base, firth_op })
                        } else {
                            Box::new(base)
                        }
                    },
                )
            };

        let log_likelihood = -0.5 * pirls_result.deviance;

        // Construct barrier config for monotonicity constraints.
        // The sparse path operates in the original (non-reparameterized)
        // coordinate system, so use the original linear constraints.
        let barrier_config = self
            .linear_constraints
            .as_ref()
            .and_then(Self::barrier_config_from_constraints);

        Ok((
            dispersion,
            deriv_provider,
            firth_logdet,
            log_likelihood,
            firth_op,
            barrier_config,
        ))
    }

    /// Evaluate the REML/LAML objective (and optionally gradient) through the
    /// unified evaluator, using a pre-obtained eval bundle.
    ///
    /// This is the single bridge from the existing PIRLS infrastructure to the
    /// unified `reml_laml_evaluate` function. Both `compute_cost` and
    /// `compute_gradient` ultimately delegate here, ensuring that cost and
    /// gradient share the exact same formula.
    pub fn evaluate_unified(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        use super::unified::{DenseSpectralOperator, PenaltyLogdetDerivs};

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        // Constraint projection: if active constraints reduce the effective
        // dimension, project the Hessian and penalty roots into the free subspace.
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        // Build HessianOperator from the (possibly projected) penalized Hessian.
        let hessian_op = Box::new(
            DenseSpectralOperator::from_symmetric(&h_for_operator).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "DenseSpectralOperator from PIRLS Hessian: {e}"
                ))
            })?,
        );

        // Penalty logdet derivatives from the reparameterization result.
        let (penalty_rank, log_det_s) =
            self.fixed_subspace_penalty_rank_and_logdet(&e_for_logdet, ridge_passport)?;

        // Penalty roots (possibly projected).
        let penalty_roots: Vec<Array2<f64>> = if let Some(z) = free_basis_opt.as_ref() {
            pirls_result
                .reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect()
        } else {
            pirls_result.reparam_result.rs_transformed.clone()
        };

        // Penalty logdet second derivatives (for outer Hessian).
        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) = self.structural_penalty_logdet_derivatives(
                &penalty_roots,
                &lambdas,
                penalty_rank,
                ridge_passport.penalty_logdet_ridge(),
            )?;
            Some(det2)
        } else {
            None
        };
        let penalty_logdet = PenaltyLogdetDerivs {
            value: log_det_s,
            first: pirls_result.reparam_result.det1.clone(),
            second: det2,
        };

        // Beta in the operator's basis.
        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        // Nullspace dimension.
        let p_eff_dim = h_for_operator.ncols();
        let nullspace_dim = p_eff_dim.saturating_sub(penalty_rank) as f64;

        // Build the family-dependent derivative context (deriv provider,
        // dispersion, TK, Firth, barrier) via the shared helper.
        let (
            deriv_provider,
            dispersion,
            tk_correction,
            tk_gradient,
            firth_logdet,
            log_likelihood,
            firth_op,
            barrier_config,
        ) = self.build_dense_derivative_context(
            pirls_result,
            bundle,
            mode,
            rho,
            &free_basis_opt,
            &penalty_roots,
            &beta,
            true, // include Firth derivatives in deriv_provider
        )?;

        self.evaluate_unified_tail(
            rho,
            mode,
            hessian_op,
            beta,
            penalty_roots,
            penalty_logdet,
            deriv_provider,
            tk_correction,
            tk_gradient,
            firth_logdet,
            firth_op,
            nullspace_dim,
            dispersion,
            log_likelihood,
            pirls_result.stable_penalty_term,
            barrier_config,
        )
    }

    /// Sparse-exact bridge: builds a `SparseCholeskyOperator` from the pre-computed
    /// sparse Cholesky factor and delegates to the unified evaluator.
    pub fn evaluate_unified_sparse(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        use super::unified::{
            HessianOperator, PenaltyLogdetDerivs, SparseCholeskyOperator, penalty_matrix_root,
        };

        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();

        // Beta in original coordinates (sparse native).
        let beta = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta.len();

        // Build HessianOperator from the sparse Cholesky factor.
        let hessian_op = Box::new(SparseCholeskyOperator::new(
            sparse.factor.clone(),
            sparse.logdet_h,
            p_dim,
        ));

        log::trace!(
            "SparseCholeskyOperator: dim={}, active_rank={}",
            hessian_op.dim(),
            hessian_op.active_rank()
        );

        // Penalty roots from s_full_list via eigendecomposition.
        let penalty_roots: Vec<Array2<f64>> = self
            .s_full_list
            .iter()
            .enumerate()
            .map(|(idx, s_k)| {
                penalty_matrix_root(s_k).map_err(|e| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "penalty root for S_{idx} failed: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Nullspace dimension.
        let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;

        // Penalty logdet second derivatives (for outer Hessian).
        let structural_rank = (p_dim as f64 - mp) as usize;
        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) = self.structural_penalty_logdet_derivatives(
                &penalty_roots,
                &lambdas,
                structural_rank,
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            Some(det2)
        } else {
            None
        };

        // Penalty log-determinant derivatives from sparse data.
        let penalty_logdet = PenaltyLogdetDerivs {
            value: sparse.logdet_s_pos,
            first: sparse.det1_values.as_ref().clone(),
            second: det2,
        };

        // Build the family-dependent context (dispersion, deriv provider,
        // Firth, barrier) via the shared sparse helper.
        let (dispersion, deriv_provider, firth_logdet, log_likelihood, firth_op, barrier_config) =
            self.build_sparse_derivative_context(pirls_result, bundle)?;

        // TK correction and gradient (non-Gaussian only): use sparse solves.
        //
        // The TK value uses the Bartlett third-order correction.
        // The TK gradient uses the basis-invariant trace formula:
        //   ∂V_TK/∂ρ_k ≈ -½ tr(H_pen⁻¹ A_k^obs)
        //              = ½ Σ_i c_i · (X v_k)_i · hat_ii
        // with H_obs⁻¹ ≈ H_pen⁻¹ (see tierney_kadane_laml_correction docs).
        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;
        let (tk_correction, tk_gradient) = if !is_gaussian_identity {
            let mut d_vec = pirls_result.solve_c_array.clone();
            for val in &mut d_vec {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }
            let third_deriv = self.third_derivative_projection_from_design(self.x(), &d_vec)?;

            let compute_grad = mode != super::unified::EvalMode::ValueOnly;
            let n_obs = d_vec.len();

            let mut h_inv_diag = Array1::<f64>::zeros(p_dim);
            let hat_leverages = if compute_grad {
                crate::linalg::sparse_exact::leverages_from_factor(&sparse.factor, self.x())?
            } else {
                Array1::<f64>::zeros(0)
            };

            for j in 0..p_dim {
                let mut e_j = Array1::<f64>::zeros(p_dim);
                e_j[j] = 1.0;
                if let Ok(f_j) = crate::linalg::sparse_exact::solve_sparse_spd(&sparse.factor, &e_j)
                {
                    h_inv_diag[j] = f_j[j];
                }
            }

            let correction = -h_inv_diag
                .iter()
                .zip(third_deriv.iter())
                .map(|(&hjj, &d3)| hjj * hjj * hjj * d3)
                .sum::<f64>()
                / 6.0;

            let correction = if correction.is_finite() {
                correction
            } else {
                0.0
            };

            let gradient = if compute_grad {
                let k = penalty_roots.len();
                let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
                let mut tk_grad = Array1::<f64>::zeros(k);

                for kidx in 0..k {
                    // Mode sensitivity: v_k = H_pen⁻¹(λ_k S_k β̂)
                    let r_beta = penalty_roots[kidx].dot(&beta);
                    let s_k_beta = fast_atv(&penalty_roots[kidx], &r_beta);
                    let a_k_beta = &s_k_beta * lambdas[kidx];
                    let v_k = match crate::linalg::sparse_exact::solve_sparse_spd(
                        &sparse.factor,
                        &a_k_beta,
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let x_v_k: Array1<f64> = self.x().matrixvectormultiply(&v_k);

                    // TK gradient[k] = ½ Σ_i c_i · (X v_k)_i · hat_ii
                    let mut acc = 0.0;
                    for i in 0..n_obs {
                        acc += d_vec[i] * x_v_k[i] * hat_leverages[i];
                    }
                    tk_grad[kidx] = 0.5 * acc;
                }

                for v in tk_grad.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }
                Some(tk_grad)
            } else {
                None
            };

            (correction, gradient)
        } else {
            (0.0, None)
        };

        self.evaluate_unified_tail(
            rho,
            mode,
            hessian_op,
            beta,
            penalty_roots,
            penalty_logdet,
            deriv_provider,
            tk_correction,
            tk_gradient,
            firth_logdet,
            firth_op,
            mp,
            dispersion,
            log_likelihood,
            pirls_result.stable_penalty_term,
            barrier_config,
        )
    }

    /// Builds the InnerSolution from a sparse eval bundle and returns EFS steps.
    /// Mirrors `evaluate_unified_sparse` but delegates to `evaluate_unified_tail_efs`
    /// instead of `evaluate_unified_tail`, since EFS only needs ValueOnly cost
    /// plus the trace-based fixed-point update.
    fn evaluate_unified_sparse_for_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        use super::unified::{PenaltyLogdetDerivs, SparseCholeskyOperator, penalty_matrix_root};

        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();

        // Beta in original coordinates (sparse native).
        let beta = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta.len();

        // Build HessianOperator from the sparse Cholesky factor.
        let hessian_op = Box::new(SparseCholeskyOperator::new(
            sparse.factor.clone(),
            sparse.logdet_h,
            p_dim,
        ));

        // Penalty roots from s_full_list via eigendecomposition.
        let penalty_roots: Vec<Array2<f64>> = self
            .s_full_list
            .iter()
            .enumerate()
            .map(|(idx, s_k)| {
                penalty_matrix_root(s_k).map_err(|e| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "penalty root for S_{idx} failed: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Nullspace dimension.
        let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;

        // EFS only needs ValueOnly — no second derivatives of penalty logdet.
        let penalty_logdet = PenaltyLogdetDerivs {
            value: sparse.logdet_s_pos,
            first: sparse.det1_values.as_ref().clone(),
            second: None,
        };

        // Build the family-dependent context via the shared sparse helper.
        let (dispersion, deriv_provider, firth_logdet, log_likelihood, firth_op, barrier_config) =
            self.build_sparse_derivative_context(pirls_result, bundle)?;

        // TK correction (non-Gaussian only). EFS is ValueOnly so we only need
        // the scalar correction, not the gradient.
        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;
        let tk_correction = if !is_gaussian_identity {
            let mut d_vec = pirls_result.solve_c_array.clone();
            for val in &mut d_vec {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }
            let third_deriv = self.third_derivative_projection_from_design(self.x(), &d_vec)?;

            let mut h_inv_diag = Array1::<f64>::zeros(p_dim);
            for j in 0..p_dim {
                let mut e_j = Array1::<f64>::zeros(p_dim);
                e_j[j] = 1.0;
                if let Ok(f_j) = crate::linalg::sparse_exact::solve_sparse_spd(&sparse.factor, &e_j)
                {
                    h_inv_diag[j] = f_j[j];
                }
            }

            let correction = -h_inv_diag
                .iter()
                .zip(third_deriv.iter())
                .map(|(&hjj, &d3)| hjj * hjj * hjj * d3)
                .sum::<f64>()
                / 6.0;

            if correction.is_finite() {
                correction
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.evaluate_unified_tail_efs(
            rho,
            hessian_op,
            beta,
            penalty_roots,
            penalty_logdet,
            deriv_provider,
            tk_correction,
            firth_logdet,
            firth_op,
            mp,
            dispersion,
            log_likelihood,
            pirls_result.stable_penalty_term,
            barrier_config,
        )
    }

    /// Shared tail for `evaluate_unified` and `evaluate_unified_sparse`:
    /// assembles the `InnerSolution`, computes prior, and calls `reml_laml_evaluate`.
    fn evaluate_unified_tail(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
        hessian_op: Box<dyn super::unified::HessianOperator>,
        beta: Array1<f64>,
        penalty_roots: Vec<Array2<f64>>,
        penalty_logdet: super::unified::PenaltyLogdetDerivs,
        deriv_provider: Box<dyn super::unified::HessianDerivativeProvider>,
        tk_correction: f64,
        tk_gradient: Option<Array1<f64>>,
        firth_logdet: f64,
        firth_op: Option<std::sync::Arc<super::FirthDenseOperator>>,
        nullspace_dim: f64,
        dispersion: super::unified::DispersionHandling,
        log_likelihood: f64,
        penalty_quadratic: f64,
        barrier_config: Option<super::unified::BarrierConfig>,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        use super::unified::{InnerSolutionBuilder, reml_laml_evaluate};

        let n_observations = self.y.len();
        let _ = nullspace_dim;
        let inner_solution = InnerSolutionBuilder::new(
            log_likelihood,
            penalty_quadratic,
            beta,
            n_observations,
            hessian_op,
            penalty_roots,
            penalty_logdet,
            dispersion,
        )
        .deriv_provider(deriv_provider)
        .tk(tk_correction, tk_gradient)
        .firth(firth_logdet, firth_op)
        .nullspace_dim_override(nullspace_dim)
        .barrier_config(barrier_config)
        .build();

        let prior = if mode == super::unified::EvalMode::ValueOnly {
            let pc = self.compute_soft_priorcost(rho);
            if pc.abs() > 0.0 {
                Some((pc, Array1::zeros(rho.len()), None))
            } else {
                None
            }
        } else {
            let pc = self.compute_soft_priorcost(rho);
            let pg = self.compute_soft_priorgrad(rho);
            let ph = if mode == super::unified::EvalMode::ValueGradientHessian {
                self.compute_soft_priorhess(rho)
            } else {
                None
            };
            if pc.abs() > 0.0 || pg.iter().any(|&v| v != 0.0) {
                Some((pc, pg, ph))
            } else {
                None
            }
        };

        reml_laml_evaluate(&inner_solution, rho.as_slice().unwrap(), mode, prior)
            .map_err(|e| EstimationError::InvalidInput(e))
    }

    /// Evaluate the unified REML/LAML objective with anisotropic ψ ext_coords
    /// injected into the `InnerSolution`.
    ///
    /// This is the primary bridge for joint [ρ, ψ] optimization of anisotropic
    /// spatial smooths. It:
    /// 1. Obtains the PIRLS bundle at the current ρ (with the design already
    ///    rebuilt for the current ψ).
    /// 2. Builds `HyperCoord` ext_coords from the supplied `DirectionalHyperParam`
    ///    list via the dense/transformed or sparse/native τ builders, depending
    ///    on the active backend.
    /// 3. Builds the full `InnerSolution` (mirroring `evaluate_unified`), injects
    ///    the ext_coords, and calls `reml_laml_evaluate`.
    ///
    /// The caller is responsible for rebuilding the `RemlState` with the design
    /// X(ψ) and penalties S(ψ) corresponding to the current ψ values before
    /// calling this method. The unified evaluator then preserves the backend:
    /// dense spectral paths stay dense, and sparse exact paths stay sparse.
    pub fn evaluate_unified_with_psi_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;

        let (ext_coords, ext_pair_fn, rho_ext_pair_fn) = if !hyper_dirs.is_empty() {
            if mode == super::unified::EvalMode::ValueGradientHessian {
                let (coords, epf, repf) =
                    self.build_tau_unified_objects_from_bundle(rho, &bundle, hyper_dirs)?;
                (coords, Some(epf), Some(repf))
            } else if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                (
                    self.build_tau_hyper_coords_sparse_exact(rho, &bundle, hyper_dirs)?,
                    None,
                    None,
                )
            } else {
                (
                    self.build_tau_hyper_coords(rho, &bundle, hyper_dirs)?,
                    None,
                    None,
                )
            }
        } else {
            (Vec::new(), None, None)
        };

        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse_with_ext_from_bundle(
                rho,
                &bundle,
                mode,
                ext_coords,
                ext_pair_fn,
                rho_ext_pair_fn,
            )
        } else {
            self.evaluate_unified_with_ext_from_bundle(
                rho,
                &bundle,
                mode,
                ext_coords,
                ext_pair_fn,
                rho_ext_pair_fn,
            )
        }
    }

    /// Assemble the full `InnerSolution` from a pre-obtained PIRLS bundle
    /// (mirroring `evaluate_unified`) and inject ext_coords, then call
    /// `reml_laml_evaluate`.
    ///
    /// This duplicates the assembly logic of `evaluate_unified` so that
    /// ext_coords can be threaded into the builder. The alternative — making
    /// `evaluate_unified` generic over an optional ext_coords — would touch
    /// every existing call site, so this targeted method is preferred.
    fn evaluate_unified_with_ext_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        ext_coords: Vec<super::unified::HyperCoord>,
        ext_coord_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
        rho_ext_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        use super::unified::{
            DenseSpectralOperator, HessianOperator, InnerSolutionBuilder, PenaltyLogdetDerivs,
            reml_laml_evaluate,
        };

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        // Constraint projection.
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        // HessianOperator.
        let hessian_op: Box<dyn HessianOperator> = Box::new(
            DenseSpectralOperator::from_symmetric(&h_for_operator).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "DenseSpectralOperator from PIRLS Hessian (psi-ext): {e}"
                ))
            })?,
        );

        // Penalty logdet.
        let (penalty_rank, log_det_s) =
            self.fixed_subspace_penalty_rank_and_logdet(&e_for_logdet, ridge_passport)?;
        let penalty_roots: Vec<Array2<f64>> = if let Some(z) = free_basis_opt.as_ref() {
            pirls_result
                .reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect()
        } else {
            pirls_result.reparam_result.rs_transformed.clone()
        };
        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) = self.structural_penalty_logdet_derivatives(
                &penalty_roots,
                &lambdas,
                penalty_rank,
                ridge_passport.penalty_logdet_ridge(),
            )?;
            Some(det2)
        } else {
            None
        };
        let penalty_logdet = PenaltyLogdetDerivs {
            value: log_det_s,
            first: pirls_result.reparam_result.det1.clone(),
            second: det2,
        };

        // Beta.
        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let p_eff_dim = h_for_operator.ncols();
        let nullspace_dim = p_eff_dim.saturating_sub(penalty_rank) as f64;

        // Build the family-dependent derivative context via the shared helper.
        let (
            deriv_provider,
            dispersion,
            tk_correction,
            tk_gradient,
            firth_logdet,
            log_likelihood,
            firth_op,
            barrier_config,
        ) = self.build_dense_derivative_context(
            pirls_result,
            bundle,
            mode,
            rho,
            &free_basis_opt,
            &penalty_roots,
            &beta,
            true, // include Firth derivatives in deriv_provider
        )?;

        // Build InnerSolution with ext_coords injected.
        let n_observations = self.y.len();
        let mut builder = InnerSolutionBuilder::new(
            log_likelihood,
            pirls_result.stable_penalty_term,
            beta,
            n_observations,
            hessian_op,
            penalty_roots,
            penalty_logdet,
            dispersion,
        )
        .deriv_provider(deriv_provider)
        .tk(tk_correction, tk_gradient)
        .firth(firth_logdet, firth_op)
        .nullspace_dim_override(nullspace_dim)
        .barrier_config(barrier_config)
        .ext_coords(ext_coords);

        if let Some(f) = ext_coord_pair_fn {
            builder = builder.ext_coord_pair_fn(f);
        }
        if let Some(f) = rho_ext_pair_fn {
            builder = builder.rho_ext_pair_fn(f);
        }

        let inner_solution = builder.build();

        // Prior (ρ-only; ψ coordinates are unconstrained soft-prior-free).
        let prior = if mode == super::unified::EvalMode::ValueOnly {
            let pc = self.compute_soft_priorcost(rho);
            if pc.abs() > 0.0 {
                Some((pc, Array1::zeros(rho.len()), None))
            } else {
                None
            }
        } else {
            let pc = self.compute_soft_priorcost(rho);
            let pg = self.compute_soft_priorgrad(rho);
            let ph = if mode == super::unified::EvalMode::ValueGradientHessian {
                self.compute_soft_priorhess(rho)
            } else {
                None
            };
            if pc.abs() > 0.0 || pg.iter().any(|&v| v != 0.0) {
                Some((pc, pg, ph))
            } else {
                None
            }
        };

        reml_laml_evaluate(&inner_solution, rho.as_slice().unwrap(), mode, prior)
            .map_err(|e| EstimationError::InvalidInput(e))
    }

    /// Sparse-exact bridge for unified evaluation with extended τ/ψ coordinates.
    ///
    /// This mirrors `evaluate_unified_sparse` but injects ext-coordinates into
    /// the `InnerSolution` so the unified evaluator can compute joint [ρ, τ/ψ]
    /// value/gradient/Hessian while still using the sparse Cholesky operator.
    fn evaluate_unified_sparse_with_ext_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::unified::EvalMode,
        ext_coords: Vec<super::unified::HyperCoord>,
        ext_coord_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
        rho_ext_pair_fn: Option<
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        >,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        use super::unified::{
            HessianOperator, InnerSolutionBuilder, PenaltyLogdetDerivs, SparseCholeskyOperator,
            penalty_matrix_root, reml_laml_evaluate,
        };

        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();

        let beta = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta.len();
        let hessian_op: Box<dyn HessianOperator> = Box::new(SparseCholeskyOperator::new(
            sparse.factor.clone(),
            sparse.logdet_h,
            p_dim,
        ));

        let penalty_roots: Vec<Array2<f64>> = self
            .s_full_list
            .iter()
            .enumerate()
            .map(|(idx, s_k)| {
                penalty_matrix_root(s_k).map_err(|e| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "penalty root for S_{idx} failed: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;
        let structural_rank = (p_dim as f64 - mp) as usize;
        let det2 = if mode == super::unified::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) = self.structural_penalty_logdet_derivatives(
                &penalty_roots,
                &lambdas,
                structural_rank,
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            Some(det2)
        } else {
            None
        };
        let penalty_logdet = PenaltyLogdetDerivs {
            value: sparse.logdet_s_pos,
            first: sparse.det1_values.as_ref().clone(),
            second: det2,
        };

        let (dispersion, deriv_provider, firth_logdet, log_likelihood, firth_op, barrier_config) =
            self.build_sparse_derivative_context(pirls_result, bundle)?;

        let is_gaussian_identity = self.config.link_function() == LinkFunction::Identity;
        let (tk_correction, tk_gradient) = if !is_gaussian_identity {
            let mut d_vec = pirls_result.solve_c_array.clone();
            for val in &mut d_vec {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }
            let third_deriv = self.third_derivative_projection_from_design(self.x(), &d_vec)?;
            let compute_grad = mode != super::unified::EvalMode::ValueOnly;
            let n_obs = d_vec.len();

            let mut h_inv_diag = Array1::<f64>::zeros(p_dim);
            let hat_leverages = if compute_grad {
                crate::linalg::sparse_exact::leverages_from_factor(&sparse.factor, self.x())?
            } else {
                Array1::<f64>::zeros(0)
            };

            for j in 0..p_dim {
                let mut e_j = Array1::<f64>::zeros(p_dim);
                e_j[j] = 1.0;
                if let Ok(f_j) = crate::linalg::sparse_exact::solve_sparse_spd(&sparse.factor, &e_j)
                {
                    h_inv_diag[j] = f_j[j];
                }
            }

            let correction = -h_inv_diag
                .iter()
                .zip(third_deriv.iter())
                .map(|(&hjj, &d3)| hjj * hjj * hjj * d3)
                .sum::<f64>()
                / 6.0;
            let correction = if correction.is_finite() {
                correction
            } else {
                0.0
            };

            let gradient = if compute_grad {
                let k = penalty_roots.len();
                let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
                let mut tk_grad = Array1::<f64>::zeros(k);
                for kidx in 0..k {
                    let r_beta = penalty_roots[kidx].dot(&beta);
                    let s_k_beta = fast_atv(&penalty_roots[kidx], &r_beta);
                    let a_k_beta = &s_k_beta * lambdas[kidx];
                    let v_k = match crate::linalg::sparse_exact::solve_sparse_spd(
                        &sparse.factor,
                        &a_k_beta,
                    ) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let x_v_k: Array1<f64> = self.x().matrixvectormultiply(&v_k);
                    let mut acc = 0.0;
                    for i in 0..n_obs {
                        acc += d_vec[i] * x_v_k[i] * hat_leverages[i];
                    }
                    tk_grad[kidx] = 0.5 * acc;
                }
                for v in tk_grad.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }
                Some(tk_grad)
            } else {
                None
            };

            (correction, gradient)
        } else {
            (0.0, None)
        };

        let n_observations = self.y.len();
        let mut builder = InnerSolutionBuilder::new(
            log_likelihood,
            pirls_result.stable_penalty_term,
            beta,
            n_observations,
            hessian_op,
            penalty_roots,
            penalty_logdet,
            dispersion,
        )
        .deriv_provider(deriv_provider)
        .tk(tk_correction, tk_gradient)
        .firth(firth_logdet, firth_op)
        .nullspace_dim_override(mp)
        .barrier_config(barrier_config)
        .ext_coords(ext_coords);

        if let Some(f) = ext_coord_pair_fn {
            builder = builder.ext_coord_pair_fn(f);
        }
        if let Some(f) = rho_ext_pair_fn {
            builder = builder.rho_ext_pair_fn(f);
        }

        let inner_solution = builder.build();

        let prior = if mode == super::unified::EvalMode::ValueOnly {
            let pc = self.compute_soft_priorcost(rho);
            if pc.abs() > 0.0 {
                Some((pc, Array1::zeros(rho.len()), None))
            } else {
                None
            }
        } else {
            let pc = self.compute_soft_priorcost(rho);
            let pg = self.compute_soft_priorgrad(rho);
            let ph = if mode == super::unified::EvalMode::ValueGradientHessian {
                self.compute_soft_priorhess(rho)
            } else {
                None
            };
            if pc.abs() > 0.0 || pg.iter().any(|&v| v != 0.0) {
                Some((pc, pg, ph))
            } else {
                None
            }
        };

        reml_laml_evaluate(&inner_solution, rho.as_slice().unwrap(), mode, prior)
            .map_err(|e| EstimationError::InvalidInput(e))
    }

    /// Shared tail for EFS: builds InnerSolution, computes cost (ValueOnly),
    /// and returns EFS step vector via `compute_efs_update`.
    ///
    /// When ψ (design-moving) coordinates are present in the inner solution's
    /// `ext_coords`, this automatically switches to the hybrid path:
    /// - Evaluates cost AND gradient (ValueAndGradient) instead of cost only
    /// - Uses `compute_hybrid_efs_update` which combines EFS for ρ/τ coords
    ///   with preconditioned gradient steps for ψ coords
    /// - Returns the ψ-block gradient and indices in the `EfsEval` for
    ///   backtracking by the outer iteration
    fn evaluate_unified_tail_efs(
        &self,
        rho: &Array1<f64>,
        hessian_op: Box<dyn super::unified::HessianOperator>,
        beta: Array1<f64>,
        penalty_roots: Vec<Array2<f64>>,
        penalty_logdet: super::unified::PenaltyLogdetDerivs,
        deriv_provider: Box<dyn super::unified::HessianDerivativeProvider>,
        tk_correction: f64,
        firth_logdet: f64,
        firth_op: Option<std::sync::Arc<super::FirthDenseOperator>>,
        nullspace_dim: f64,
        dispersion: super::unified::DispersionHandling,
        log_likelihood: f64,
        penalty_quadratic: f64,
        barrier_config: Option<super::unified::BarrierConfig>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        use super::unified::{
            InnerSolutionBuilder, compute_efs_update, compute_hybrid_efs_update, reml_laml_evaluate,
        };

        let n_observations = self.y.len();
        let beta_for_barrier = beta.clone();
        let inner_solution = InnerSolutionBuilder::new(
            log_likelihood,
            penalty_quadratic,
            beta,
            n_observations,
            hessian_op,
            penalty_roots,
            penalty_logdet,
            dispersion,
        )
        .deriv_provider(deriv_provider)
        .tk(tk_correction, None)
        .firth(firth_logdet, firth_op)
        .nullspace_dim_override(nullspace_dim)
        .barrier_config(barrier_config)
        .build();

        // Check whether any ψ (design-moving) coordinates are present.
        // If so, we need the hybrid path which requires the gradient.
        let has_psi = inner_solution.ext_coords.iter().any(|c| !c.is_penalty_like);

        // For pure EFS: cost only (no gradient needed).
        // For hybrid EFS: cost + gradient (gradient needed for ψ block).
        let eval_mode = if has_psi {
            super::unified::EvalMode::ValueAndGradient
        } else {
            super::unified::EvalMode::ValueOnly
        };

        let prior = {
            let pc = self.compute_soft_priorcost(rho);
            if pc.abs() > 0.0 {
                // For hybrid mode, we need the prior gradient too.
                let prior_grad = if has_psi {
                    self.compute_soft_priorgrad(rho)
                } else {
                    Array1::zeros(rho.len())
                };
                Some((pc, prior_grad, None))
            } else {
                None
            }
        };
        let cost_result =
            reml_laml_evaluate(&inner_solution, rho.as_slice().unwrap(), eval_mode, prior)
                .map_err(|e| EstimationError::InvalidInput(e))?;

        if has_psi {
            // Hybrid path: EFS for ρ/τ + preconditioned gradient for ψ.
            let gradient = cost_result.gradient.as_ref().expect(
                "hybrid EFS requires gradient (ValueAndGradient mode should have computed it)",
            );
            let hybrid = compute_hybrid_efs_update(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
            );

            let psi_gradient = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(ndarray::Array1::from_vec(hybrid.psi_gradient))
            };
            let psi_indices = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(hybrid.psi_indices)
            };

            Ok(crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps: hybrid.steps,
                beta: Some(beta_for_barrier),
                psi_gradient,
                psi_indices,
            })
        } else {
            // Pure EFS path: no ψ coordinates.
            let steps = compute_efs_update(&inner_solution, rho.as_slice().unwrap());

            Ok(crate::solver::outer_strategy::EfsEval {
                cost: cost_result.cost,
                steps,
                beta: Some(beta_for_barrier),
                psi_gradient: None,
                psi_indices: None,
            })
        }
    }

    /// Compute EFS (Extended Fellner-Schall) evaluation: runs the inner solve
    /// at the given rho, computes the REML/LAML cost, and returns the EFS
    /// step vector from `compute_efs_update`.
    ///
    /// This is the entry point called by the EFS branch of `run_outer` via
    /// the `OuterObjective::eval_efs` method.
    pub fn compute_efs_steps(
        &self,
        p: &Array1<f64>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(EstimationError::RemlOptimizationFailed(
                    "inner solve ill-conditioned during EFS evaluation".to_string(),
                ));
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };

        // Dispatch to the appropriate backend.
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            return self.evaluate_unified_sparse_for_efs(p, &bundle);
        }

        // Reuse the same setup as evaluate_unified but call the EFS tail.
        self.evaluate_unified_for_efs(p, &bundle)
    }

    /// Builds the InnerSolution from an eval bundle and returns EFS steps.
    /// Mirrors `evaluate_unified` but delegates to `evaluate_unified_tail_efs`.
    fn evaluate_unified_for_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        use super::unified::{DenseSpectralOperator, PenaltyLogdetDerivs};

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        // Constraint projection (same as evaluate_unified).
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                pirls_result.reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                pirls_result.reparam_result.e_transformed.clone(),
            )
        };

        let hessian_op = Box::new(
            DenseSpectralOperator::from_symmetric(&h_for_operator).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "DenseSpectralOperator from PIRLS Hessian (EFS): {e}"
                ))
            })?,
        );

        let (penalty_rank, log_det_s) =
            self.fixed_subspace_penalty_rank_and_logdet(&e_for_logdet, ridge_passport)?;

        let penalty_roots: Vec<Array2<f64>> = if let Some(z) = free_basis_opt.as_ref() {
            pirls_result
                .reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect()
        } else {
            pirls_result.reparam_result.rs_transformed.clone()
        };

        // EFS only needs ValueOnly — no second derivatives of penalty logdet.
        let penalty_logdet = PenaltyLogdetDerivs {
            value: log_det_s,
            first: pirls_result.reparam_result.det1.clone(),
            second: None,
        };

        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t()
                .dot(&pirls_result.beta_transformed.as_ref().to_owned())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let p_eff_dim = h_for_operator.ncols();
        let nullspace_dim = p_eff_dim.saturating_sub(penalty_rank) as f64;

        // Build derivative context via the shared helper. EFS uses ValueOnly
        // mode (no TK gradient) and skips Firth derivative wrapping.
        let (
            deriv_provider,
            dispersion,
            tk_correction,
            _,
            firth_logdet,
            log_likelihood,
            firth_op,
            barrier_config,
        ) = self.build_dense_derivative_context(
            pirls_result,
            bundle,
            super::unified::EvalMode::ValueOnly,
            rho,
            &free_basis_opt,
            &penalty_roots,
            &beta,
            false, // EFS: no Firth derivatives in deriv_provider
        )?;

        self.evaluate_unified_tail_efs(
            rho,
            hessian_op,
            beta,
            penalty_roots,
            penalty_logdet,
            deriv_provider,
            tk_correction,
            firth_logdet,
            firth_op,
            nullspace_dim,
            dispersion,
            log_likelihood,
            pirls_result.stable_penalty_term,
            barrier_config,
        )
    }

    pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.arena
            .lastgradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
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
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let result = self.evaluate_unified_sparse(
                p,
                &bundle,
                super::unified::EvalMode::ValueAndGradient,
            )?;
            return Ok(result.gradient.unwrap());
        }
        let result =
            self.evaluate_unified(p, &bundle, super::unified::EvalMode::ValueAndGradient)?;
        Ok(result.gradient.unwrap())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified REML evaluation with link-parameter ext_coords
    // ═══════════════════════════════════════════════════════════════════════════

    /// Evaluate the unified REML/LAML objective with SAS or mixture link
    /// parameters injected as ext_coords into the `InnerSolution`.
    ///
    /// This replaces the manual gradient computation in the outer optimizer's
    /// eval_fn closure. The link parameter derivatives (du/dθ, dW/dθ, dℓ/dθ)
    /// are wrapped as `HyperCoord` objects and passed through the same unified
    /// evaluator that handles ρ coordinates, giving a single Newton/BFGS step
    /// over `[ρ, θ_link]` jointly.
    ///
    /// # Parameters
    ///
    /// - `rho`: log-smoothing parameters (penalty coordinates only, length k)
    /// - `mode`: evaluation mode (cost-only, cost+gradient, cost+gradient+Hessian)
    ///
    /// # Returns
    ///
    /// `RemlLamlResult` whose gradient (when requested) has length `k + aux_dim`,
    /// with the link parameters appended after the ρ coordinates. The caller is
    /// responsible for:
    /// - Applying SAS epsilon reparameterization chain rule (`grad[k] *= d_eps/d_raw`)
    /// - Adding SAS ridge/barrier gradient contributions to `grad[k+1]`
    /// - Adding SAS ridge/barrier cost contributions to `result.cost`
    pub fn evaluate_unified_with_link_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::unified::EvalMode,
    ) -> Result<super::unified::RemlLamlResult, EstimationError> {
        // Firth bias reduction modifies the score equations in ways that the
        // link ext_coord construction does not yet account for. Reject early.
        if self.config.firth_bias_reduction {
            return Err(EstimationError::InvalidInput(
                "link-parameter ext_coord optimization is incompatible with \
                 Firth-adjusted outer gradients"
                    .to_string(),
            ));
        }

        let bundle = self.obtain_eval_bundle(rho)?;

        // Build link ext_coords from the current runtime link state.
        let ext_coords = if let Some(sas_state) = &self.runtime_sas_link_state {
            let is_beta_logistic = matches!(
                self.config.link_function(),
                crate::types::LinkFunction::BetaLogistic
            );
            self.build_sas_link_ext_coords(&bundle, sas_state, is_beta_logistic)?
        } else if let Some(mix_state) = &self.runtime_mixture_link_state {
            self.build_mixture_link_ext_coords(&bundle, mix_state)?
        } else {
            Vec::new()
        };

        if ext_coords.is_empty() {
            // No link parameters to optimize — fall back to standard path.
            return self.evaluate_unified(rho, &bundle, mode);
        }

        // Delegate to the existing ext_coords injection path.
        // No pair callbacks for link parameters: the outer Hessian for link
        // coords is left to BFGS (HessianResult::Unavailable). This matches
        // the current behavior where link parameters have gradient-only.
        self.evaluate_unified_with_ext_from_bundle(rho, &bundle, mode, ext_coords, None, None)
    }
}
