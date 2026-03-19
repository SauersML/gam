use super::*;
use crate::matrix::DenseRightProductView;

// ─── Binomial auxiliary terms for link-parameter ext_coord construction ───
//
// Mirrors the `BinomialAuxTerms` in estimate.rs but is local to the REML
// module so that hyper.rs can compute per-observation likelihood derivatives
// without depending on the estimate module's private helpers.

pub(crate) const LINK_BINOMIAL_AUX_MU_EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
pub(crate) struct LinkBinomialAux {
    /// dℓ_i/dμ_i = w_i (y_i/μ_i − (1−y_i)/(1−μ_i))
    pub(crate) a1: f64,
    /// d²ℓ_i/dμ_i² = w_i (−y_i/μ_i² − (1−y_i)/(1−μ_i)²)
    pub(crate) a2: f64,
    /// μ(1−μ) — binomial variance function
    pub(crate) variance: f64,
    /// dVar/dμ = 1−2μ
    pub(crate) variancemu_scale: f64,
}

#[inline]
pub(crate) fn link_binomial_aux(yi: f64, wi: f64, mu: f64) -> LinkBinomialAux {
    let mu = if mu.is_finite() {
        mu.clamp(LINK_BINOMIAL_AUX_MU_EPS, 1.0 - LINK_BINOMIAL_AUX_MU_EPS)
    } else {
        0.5
    };
    let one_minusmu = 1.0 - mu;
    let a1 = wi * (yi / mu - (1.0 - yi) / one_minusmu);
    let a2 = wi * (-(yi / (mu * mu)) - (1.0 - yi) / (one_minusmu * one_minusmu));
    LinkBinomialAux {
        a1,
        a2,
        variance: mu * one_minusmu,
        variancemu_scale: 1.0 - 2.0 * mu,
    }
}

#[derive(Clone)]
enum TauTauDesignTerm {
    Dense(Array2<f64>),
    Implicit(HyperDesignDerivative),
}

impl<'a> RemlState<'a> {
    fn get_pairwisesecond_penalty_components(
        hyper_dirs: &[DirectionalHyperParam],
        i: usize,
        j: usize,
    ) -> Vec<PenaltyDerivativeComponent> {
        if let Some(components) = hyper_dirs
            .get(i)
            .and_then(|dir| dir.penaltysecond_components_for(j))
        {
            return components.to_vec();
        }
        hyper_dirs
            .get(j)
            .and_then(|dir| dir.penaltysecond_components_for(i))
            .map(|components| components.to_vec())
            .unwrap_or_default()
    }

    pub(crate) fn validate_penalty_component_shapes(
        components: &[PenaltyDerivativeComponent],
        p: usize,
        label: &str,
    ) -> Result<(), EstimationError> {
        for component in components {
            if component.matrix.nrows() != p || component.matrix.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "{} shape mismatch for penalty {}: expected {}x{}, got {}x{}",
                    label,
                    component.penalty_index,
                    p,
                    p,
                    component.matrix.nrows(),
                    component.matrix.ncols()
                )));
            }
        }
        Ok(())
    }

    fn transform_penalty_components(
        components: &[PenaltyDerivativeComponent],
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Vec<PenaltyDerivativeComponent> {
        components
            .iter()
            .map(|component| PenaltyDerivativeComponent {
                penalty_index: component.penalty_index,
                matrix: HyperPenaltyDerivative::from(
                    component
                        .matrix
                        .transformed(qs, free_basis_opt)
                        .expect("valid transformed hyper penalty component"),
                ),
            })
            .collect()
    }

    fn weighted_cross_from_design_derivative(
        deriv: &HyperDesignDerivative,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        x_dense: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let p = x_dense.ncols();
        let n = x_dense.nrows();
        if weights.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "weighted_cross_from_design_derivative weight length mismatch: got {}, expected {}",
                weights.len(),
                n
            )));
        }

        let mut out = Array2::<f64>::zeros((p, p));
        for col in 0..p {
            let rhs = weights * &x_dense.column(col).to_owned();
            let result = deriv.transformed_transpose_mul(qs, free_basis_opt, &rhs)?;
            if result.len() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "weighted_cross_from_design_derivative column size mismatch: got {}, expected {}",
                    result.len(),
                    p
                )));
            }
            out.column_mut(col).assign(&result);
        }
        Ok(out)
    }

    pub(crate) fn compute_joint_hypercostgradienthessian(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
        let rho = theta.slice(s![..rho_dim]).to_owned();

        if !hyper_dirs.is_empty() {
            // Unified evaluator path: produces the joint [ρ, ψ] gradient and
            // Hessian from a single InnerSolution + reml_laml_evaluate call.
            let result = self.evaluate_unified_with_psi_ext(
                &rho,
                super::unified::EvalMode::ValueGradientHessian,
                hyper_dirs,
            )?;
            let cost = result.cost;
            let grad = result
                .gradient
                .unwrap_or_else(|| Array1::zeros(theta.len()));
            let hess = result
                .hessian
                .unwrap_or_else(|| Array2::zeros((theta.len(), theta.len())));
            log::trace!(
                "[joint-hyper] unified evaluator: cost={:.6e}, grad_norm={:.4e}, hess_dim={}x{}",
                cost,
                grad.iter().map(|g| g * g).sum::<f64>().sqrt(),
                hess.nrows(),
                hess.ncols(),
            );
            Ok((cost, grad, hess))
        } else {
            // Rho-only path: no directional hyperparameters, use the standard
            // rho-only evaluators and let inner_strategy pick the Hessian path.
            let cost = self.compute_cost(&rho)?;
            let grad = self.compute_gradient(&rho)?;
            let hess = self.compute_lamlhessian_consistent(&rho)?;
            Ok((cost, grad, hess))
        }
    }

    pub(crate) fn build_tau_unified_objects_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Vec<super::unified::HyperCoord>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        // Guard: non-sparse tau coordinate construction requires dense design.
        // Skip for large models that would blow memory.
        let n_x = self.x().nrows();
        let p_x = self.x().ncols();
        const HYPER_MAX_DENSE_WORK: usize = 50_000_000;
        if n_x.saturating_mul(p_x) > HYPER_MAX_DENSE_WORK
            && bundle.backend_kind() != GeometryBackendKind::SparseExactSpd
        {
            log::warn!(
                "skipping tau hyper-coordinate construction (n={n_x}, p={p_x}): \
                 dense design materialization too large; falling back to rho-only REML"
            );
            let identity_pair: Box<
                dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync,
            > = Box::new(|_, _| super::unified::HyperCoordPair::zero());
            let identity_pair2: Box<
                dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync,
            > = Box::new(|_, _| super::unified::HyperCoordPair::zero());
            return Ok((Vec::new(), identity_pair, identity_pair2));
        }
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let ext_coords = self.build_tau_hyper_coords_sparse_exact(rho, bundle, hyper_dirs)?;
            let (ext_pair_fn, rho_ext_pair_fn) =
                self.build_tau_pair_callbacks_sparse_exact(rho, bundle, hyper_dirs)?;
            Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
        } else if matches!(
            bundle.pirls_result.coordinate_frame,
            crate::pirls::PirlsCoordinateFrame::TransformedQs
        ) && self
            .active_constraint_free_basis(bundle.pirls_result.as_ref())
            .is_none()
        {
            let ext_coords = self.build_tau_hyper_coords_original_basis(rho, bundle, hyper_dirs)?;
            let (ext_pair_fn, rho_ext_pair_fn) =
                self.build_tau_pair_callbacks_original_basis(rho, bundle, hyper_dirs)?;
            Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
        } else {
            let ext_coords = self.build_tau_hyper_coords(rho, bundle, hyper_dirs)?;
            let (ext_pair_fn, rho_ext_pair_fn) =
                self.build_tau_pair_callbacks(rho, bundle, hyper_dirs)?;
            Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified HyperCoord builders for τ (directional) hyperparameters
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build [`HyperCoord`] objects for τ (directional / design-moving)
    /// hyperparameters.
    ///
    /// This produces generic `(a_j, g_j, B_j, ld_s_j)` tuples that the
    /// unified evaluator in `unified.rs` can consume for first-order
    /// directional hyperparameter derivatives.
    ///
    /// # Field derivation
    ///
    /// For each τ direction j, holding β fixed at β̂:
    ///
    /// | Field   | Formula |
    /// | `a`     | `−u^T X_{τ_j} β̂ + 0.5 β̂^T S_{τ_j} β̂ [+ Φ_{τ_j}\|_β]` |
    /// | `g`     | `X_{τ_j}^T u − X^T diag(w) X_{τ_j} β̂ − S_{τ_j} β̂ [− (g_φ)_{τ_j}]` |
    /// | `B`     | `X_{τ_j}^T W X + X^T W X_{τ_j} + X^T diag(c ⊙ X_{τ_j} β̂) X + S_{τ_j} [− Firth drifts]` |
    /// | `ld_s`  | `tr(S⁺ S_{τ_j})` |
    ///
    /// For Gaussian identity link, `c = 0` and W is constant, so the
    /// third-derivative correction in B vanishes and `b_depends_on_beta = false`.
    pub(crate) fn build_tau_hyper_coords(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        let psi_dim = hyper_dirs.len();

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        // --- Transformed design, beta, penalty basis ---
        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let x_dense_arc = pirls_result
            .x_transformed
            .try_to_dense_arc("build_tau_hyper_coords requires dense transformed design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense = x_dense_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_arc.as_ref());

        let e_eval;
        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            e_eval = reparam_result.e_transformed.dot(z);
        } else {
            e_eval = reparam_result.e_transformed.clone();
        }
        let p_dim = beta_eval.len();
        if p_dim == 0 {
            return Ok((0..psi_dim)
                .map(|j| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: false,
                    is_penalty_like: hyper_dirs[j].is_penalty_like,
                })
                .collect::<Vec<_>>());
        }

        // Working residual u = w ⊙ (z − η̂).
        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = &pirls_result.finalweights;
        let c_array = &pirls_result.solve_c_array;

        // Whether third-derivative corrections are needed (non-Gaussian).
        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);
        let b_depends_on_beta = !is_gaussian_identity;

        // Firth operator (Firth-logit only).
        // Match the active coefficient basis exactly: when constraints project
        // to β = Z β_free, the Jeffreys objects for τ/ψ derivatives must use
        // the projected design X_eff = X Z rather than the cached full-basis
        // operator from the unconstrained transformed space.
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_logit_active {
            if free_basis_opt.is_none() {
                if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                    Some(cached.as_ref().clone())
                } else {
                    Some(Self::build_firth_dense_operator(
                        x_dense,
                        &pirls_result.final_eta,
                        self.weights,
                    )?)
                }
            } else {
                Some(Self::build_firth_dense_operator(
                    x_dense,
                    &pirls_result.final_eta,
                    self.weights,
                )?)
            }
        } else {
            None
        };

        // --- Implicit operator activation ---
        // Check whether any tau direction has an implicit design derivative and
        // the problem is large enough to benefit from implicit B_j operators.
        // When active, we build ImplicitHyperOperator instances that compute
        // B_j · v on the fly, avoiding materialization of the dense (p × p) B_j.
        let any_has_implicit = hyper_dirs.iter().any(|d| d.has_implicit_operator());
        let n_obs = x_dense.nrows();
        // Determine the number of anisotropic axes from the first implicit operator.
        let implicit_n_axes = if any_has_implicit {
            hyper_dirs
                .iter()
                .find_map(|d| d.implicit_first_axis_info())
                .map(|(op, _)| op.n_axes())
                .unwrap_or(0)
        } else {
            0
        };
        let use_implicit = any_has_implicit
            && crate::terms::basis::should_use_implicit_operators(n_obs, p_dim, implicit_n_axes);

        // Shared Arcs for the implicit operator path (only allocated when needed).
        let x_dense_shared: Option<std::sync::Arc<Array2<f64>>> = if use_implicit {
            Some(std::sync::Arc::new(x_dense.to_owned()))
        } else {
            None
        };
        let w_diag_shared: Option<std::sync::Arc<Array1<f64>>> = if use_implicit {
            Some(std::sync::Arc::new(w_diag.clone()))
        } else {
            None
        };

        let mut coords = Vec::with_capacity(psi_dim);

        for j in 0..psi_dim {
            // --- X_{τ_j} and S_{τ_j} in transformed coordinates ---
            let x_tau_j =
                hyper_dirs[j].transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?;
            let penalty_components_j = Self::transform_penalty_components(
                hyper_dirs[j].penalty_first_components(),
                &reparam_result.qs,
                free_basis_opt.as_ref(),
            );
            let s_tau_j = penalty_components_j.iter().try_fold(
                Array2::<f64>::zeros((p_dim, p_dim)),
                |mut acc, component| {
                    if component.penalty_index >= rho.len() {
                        return Err(EstimationError::InvalidInput(format!(
                            "penalty_index {} out of bounds for rho dimension {}",
                            component.penalty_index,
                            rho.len()
                        )));
                    }
                    component
                        .matrix
                        .scaled_add_to(&mut acc, rho[component.penalty_index].exp())?;
                    Ok(acc)
                },
            )?;

            // --- a_j: fixed-β cost derivative (envelope term) ---
            // a_j = −u^T (X_{τ_j} β̂) + 0.5 β̂^T S_{τ_j} β̂  [+ Φ_{τ_j}|_β for Firth]
            let x_tau_beta_j = x_tau_j.dot(&beta_eval);
            let mut a_j = -u.dot(&x_tau_beta_j) + 0.5 * beta_eval.dot(&s_tau_j.dot(&beta_eval));
            // Firth partial: Φ_{τ_j}|_β = 0.5 tr(I_r^{-1} I_{r,τ_j}).
            let mut firth_tau_kernel_j: Option<FirthTauPartialKernel> = None;
            if let Some(op) = firth_op.as_ref() {
                let need_kernel = x_tau_j.iter().any(|v| *v != 0.0);
                let tau_bundle =
                    Self::firth_exact_tau_kernel(op, &x_tau_j, &beta_eval, need_kernel);
                a_j += tau_bundle.phi_tau_partial;
                if need_kernel {
                    firth_tau_kernel_j = Some(tau_bundle.tau_kernel.expect(
                        "firth_exact_tau_kernel should return kernel when need_kernel=true",
                    ));
                }
            }

            // --- g_j: fixed-β score (the implicit-function RHS) ---
            // g_j = X_{τ_j}^T u − X^T diag(w)(X_{τ_j} β̂) − S_{τ_j} β̂  [− (g_φ)_{τ_j}]
            let weighted_x_tau_beta_j = w_diag * &x_tau_beta_j;
            let mut g_j = x_tau_j.t().dot(&u)
                - x_dense.t().dot(&weighted_x_tau_beta_j)
                - s_tau_j.dot(&beta_eval);
            if let Some(op) = firth_op.as_ref() {
                let tau_bundle = Self::firth_exact_tau_kernel(op, &x_tau_j, &beta_eval, false);
                g_j -= &tau_bundle.gphi_tau;
            }

            // --- B_j: fixed-β Hessian drift ---
            // B_j = X_{τ_j}^T W X + X^T W X_{τ_j} + S_{τ_j}
            //     [+ X^T diag(c ⊙ X_{τ_j} β̂) X]  (non-Gaussian only)
            //     [− Firth Hessian drifts]

            // --- b_operator: implicit operator for B_j · v (when activated) ---
            // When the problem is large enough and this coordinate has an implicit
            // design derivative, we build an ImplicitHyperOperator that computes
            // B_j · v without materializing the full (p × p) matrix.
            //
            // Note: the ImplicitHyperOperator covers the three dominant terms:
            //   (∂X/∂ψ_d)^T (W · (X · v)) + X^T (W · ((∂X/∂ψ_d) · v)) + S_{ψ_d} · v
            // Third-derivative corrections (non-Gaussian) and Firth drifts are NOT
            // included in the implicit operator — they are small relative to the
            // dominant terms and would require additional storage. For problems large
            // enough to trigger implicit operators, these corrections contribute
            // negligibly to the stochastic trace estimate.
            let b_operator: Option<Box<dyn super::unified::HyperOperator>> = if use_implicit {
                if let Some((implicit_deriv, axis)) = hyper_dirs[j].implicit_first_axis_info() {
                    Some(Box::new(super::unified::ImplicitHyperOperator {
                        implicit_deriv,
                        axis,
                        x_dense: x_dense_shared.clone().unwrap(),
                        w_diag: w_diag_shared.clone().unwrap(),
                        s_psi: s_tau_j.clone(),
                        p: p_dim,
                    }))
                } else {
                    None
                }
            } else {
                None
            };

            // Materialize the dense B_j matrix only when we are not using the
            // operator-backed fast path. The HyperCoord drift wrapper can carry
            // either representation cleanly, without a dummy payload contract.
            let dense_b = if b_operator.is_some() {
                None
            } else {
                let mut b_j = Self::weighted_cross(&x_tau_j, x_dense, w_diag);
                b_j += &Self::weighted_cross(x_dense, &x_tau_j, w_diag);
                b_j += &s_tau_j;

                if !is_gaussian_identity {
                    // Third-derivative correction: X^T diag(c ⊙ X_{τ_j} β̂) X.
                    let c_x_tau_beta = c_array * &x_tau_beta_j;
                    let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
                    b_j +=
                        &Self::xt_diag_x_dense_into(x_dense, &c_x_tau_beta, &mut weighted_scratch);
                }

                // Firth Hessian drifts: −(H_φ)_{τ_j}|_β.
                // The D(H_φ)[β_{τ_j}] part is NOT included here because it
                // depends on β_{τ_j} (the IFT solve result), which the unified
                // evaluator computes itself. Only the fixed-β partial goes in B_j.
                if let Some(op) = firth_op.as_ref() {
                    if let Some(kernel) = firth_tau_kernel_j.as_ref() {
                        let eye = Array2::<f64>::eye(p_dim);
                        let hphi_tau_partial =
                            Self::firth_hphi_tau_partial_apply(op, &x_tau_j, kernel, &eye);
                        b_j -= &hphi_tau_partial;
                    }
                }

                Some(b_j)
            };

            // --- ld_s_j: penalty pseudo-logdet derivative ---
            // ld_s_j = tr(S⁺ S_{τ_j}).
            // Uses the exact pseudoinverse (via fixed_subspace_penalty_trace).
            let ld_s_j =
                self.fixed_subspace_penalty_trace(&e_eval, &s_tau_j, pirls_result.ridge_passport)?;

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_parts(dense_b, b_operator),
                ld_s: ld_s_j,
                b_depends_on_beta,
                is_penalty_like: hyper_dirs[j].is_penalty_like,
            });
        }

        Ok(coords)
    }

    /// Sparse-exact τ builder in the original sparse/native coefficient basis.
    ///
    /// Unlike the dense/transformed path, this builder keeps `β`, `X`, `X_τ`,
    /// and `S_τ` in the original sparse-native coordinates used by the sparse
    /// Cholesky factor. The fixed-β Hessian drift is attached as an operator,
    /// not a dense matrix, so the unified evaluator can compute
    /// `tr(H^{-1} B_τ)` exactly without falling back to dense spectral
    /// materialization.
    pub(crate) fn build_tau_hyper_coords_original_basis(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        let psi_dim = hyper_dirs.len();

        let pirls_result = bundle.pirls_result.as_ref();
        let beta_eval = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta_eval.len();
        let n_obs = self.y.len();
        if p_dim == 0 {
            return Ok((0..psi_dim)
                .map(|j| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: false,
                    is_penalty_like: hyper_dirs[j].is_penalty_like,
                })
                .collect());
        }

        for (j, dir) in hyper_dirs.iter().enumerate() {
            if dir.x_tau_original.nrows() != n_obs || dir.x_tau_original.ncols() != p_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "X_tau shape mismatch for sparse exact tau coord {}: expected {}x{}, got {}x{}",
                    j,
                    n_obs,
                    p_dim,
                    dir.x_tau_original.nrows(),
                    dir.x_tau_original.ncols()
                )));
            }
            Self::validate_penalty_component_shapes(
                dir.penalty_first_components(),
                p_dim,
                "S_tau",
            )?;
        }

        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = std::sync::Arc::new(pirls_result.finalweights.clone());
        let x_design = self.x().clone();

        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle
                .firth_dense_operator_original
                .as_ref()
                .or(bundle.firth_dense_operator.as_ref())
            {
                Some(cached.as_ref().clone())
            } else {
                let x_dense_arc = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact tau coords require dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(Self::build_firth_dense_operator(
                    x_dense_arc.as_ref(),
                    &pirls_result.final_eta,
                    self.weights,
                )?)
            }
        } else {
            None
        };
        let pld = super::penalty_logdet::PenaltyPseudologdet::from_penalties(
            &self.canonical_penalties,
            &rho.mapv(f64::exp).to_vec(),
            bundle.ridge_passport.penalty_logdet_ridge(),
            p_dim,
        )
        .map_err(EstimationError::InvalidInput)?;

        let mut coords = Vec::with_capacity(psi_dim);
        for dir in hyper_dirs {
            let s_tau_j = dir.penalty_total_at(rho, p_dim)?;
            let x_tau_beta_j = dir.x_tau_original.forward_mul_original(&beta_eval)?;
            let weighted_x_tau_beta_j = &*w_diag * &x_tau_beta_j;

            let mut a_j = -u.dot(&x_tau_beta_j) + 0.5 * beta_eval.dot(&s_tau_j.dot(&beta_eval));
            let mut g_j = dir.x_tau_original.transpose_mul_original(&u)?
                - self.x().transpose_vector_multiply(&weighted_x_tau_beta_j)
                - s_tau_j.dot(&beta_eval);

            let mut firth_hphi_tau_partial = None;
            if let Some(op) = firth_op.as_ref() {
                let x_tau_dense = dir.x_tau_dense();
                let need_kernel = dir.x_tau_original.any_nonzero();
                let tau_bundle =
                    Self::firth_exact_tau_kernel(op, &x_tau_dense, &beta_eval, need_kernel);
                g_j -= &tau_bundle.gphi_tau;
                a_j += tau_bundle.phi_tau_partial;
                if let Some(kernel) = tau_bundle.tau_kernel.as_ref() {
                    let eye = Array2::<f64>::eye(p_dim);
                    firth_hphi_tau_partial = Some(Self::firth_hphi_tau_partial_apply(
                        op,
                        &x_tau_dense,
                        kernel,
                        &eye,
                    ));
                }
            }

            let c_x_tau_beta = if is_gaussian_identity {
                None
            } else {
                Some(&pirls_result.solve_c_array * &x_tau_beta_j)
            };

            let ld_s_j = pld.tau_gradient_component(&s_tau_j);

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_operator(Box::new(
                    super::unified::SparseDirectionalHyperOperator {
                        x_tau: dir.x_tau_original.clone(),
                        x_design: x_design.clone(),
                        w_diag: w_diag.clone(),
                        s_tau: s_tau_j,
                        c_x_tau_beta,
                        firth_hphi_tau_partial,
                        p: p_dim,
                    },
                )),
                ld_s: ld_s_j,
                b_depends_on_beta: !is_gaussian_identity,
                is_penalty_like: dir.is_penalty_like,
            });
        }

        Ok(coords)
    }

    pub(crate) fn build_tau_hyper_coords_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        if bundle.sparse_exact.is_none() {
            return Err(EstimationError::InvalidInput(
                "missing sparse exact evaluation payload".to_string(),
            ));
        }
        self.build_tau_hyper_coords_original_basis(rho, bundle, hyper_dirs)
    }

    /// Sparse-exact τ×τ and ρ×τ pair callbacks in original coordinates.
    ///
    /// This path prioritizes correctness and consistent basis alignment with the
    /// sparse Cholesky operator. It densifies the design only for second-order
    /// pair assembly, which is acceptable because pair callbacks are used only
    /// when the caller requests the full outer Hessian.
    pub(crate) fn build_tau_pair_callbacks_original_basis(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        let pirls_result = bundle.pirls_result.as_ref();
        let beta_eval = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta_eval.len();
        let psi_dim = hyper_dirs.len();
        let k_count = rho.len();
        let lambdas = rho.mapv(f64::exp);

        if p_dim == 0 {
            let tau_tau_pair_fn = move |_: usize, _: usize| super::unified::HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(p_dim),
                b_mat: Array2::zeros((p_dim, p_dim)),
                ld_s: 0.0,
            };
            let rho_tau_pair_fn = move |_: usize, _: usize| super::unified::HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(p_dim),
                b_mat: Array2::zeros((p_dim, p_dim)),
                ld_s: 0.0,
            };
            return Ok((Box::new(tau_tau_pair_fn), Box::new(rho_tau_pair_fn)));
        }

        let s_tau_list: Vec<Array2<f64>> = hyper_dirs
            .iter()
            .map(|dir| dir.penalty_total_at(rho, p_dim))
            .collect::<Result<Vec<_>, _>>()?;

        let mut s_tau_tau: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let second_components =
                    Self::get_pairwisesecond_penalty_components(hyper_dirs, i, j);
                if second_components.is_empty() {
                    continue;
                }
                Self::validate_penalty_component_shapes(&second_components, p_dim, "S_tau_tau")?;
                let total = second_components.iter().try_fold(
                    Array2::<f64>::zeros((p_dim, p_dim)),
                    |mut acc, component| {
                        if component.penalty_index >= rho.len() {
                            return Err(EstimationError::InvalidInput(format!(
                                "penalty_index {} out of bounds for rho dimension {}",
                                component.penalty_index,
                                rho.len()
                            )));
                        }
                        component
                            .matrix
                            .scaled_add_to(&mut acc, rho[component.penalty_index].exp())?;
                        Ok(acc)
                    },
                )?;
                s_tau_tau[i][j] = Some(total);
            }
        }

        // Use block-factored penalty logdet when penalties are disjoint.
        let pld = super::penalty_logdet::PenaltyPseudologdet::from_penalties(
            &self.canonical_penalties,
            &lambdas.as_slice().unwrap_or(&[]),
            0.0,
            p_dim,
        )
        .map_err(EstimationError::InvalidInput)?;

        let x_dense = self
            .x()
            .try_to_dense_arc("sparse exact tau pair callbacks require dense design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense: Array2<f64> = x_dense.as_ref().clone();
        let x_tau_list: Vec<Array2<f64>> = hyper_dirs.iter().map(|dir| dir.x_tau_dense()).collect();
        let x_tau_beta_list: Vec<Array1<f64>> = x_tau_list
            .iter()
            .map(|x_tau| x_tau.dot(&beta_eval))
            .collect();

        let mut x_tau_tau: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in i..psi_dim {
                let xij = hyper_dirs[i]
                    .x_tau_tau_entry_at(j)
                    .or_else(|| hyper_dirs[j].x_tau_tau_entry_at(i))
                    .map(|entry| entry.materialize());
                if xij.is_some() {
                    x_tau_tau[j][i] = xij.clone();
                }
                x_tau_tau[i][j] = xij;
            }
        }

        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = pirls_result.finalweights.clone();
        let c_array = pirls_result.solve_c_array.clone();
        let d_array = pirls_result.solve_d_array.clone();
        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);

        let penalty_components_per_dir: Vec<Vec<PenaltyDerivativeComponent>> = hyper_dirs
            .iter()
            .map(|dir| dir.penalty_first_components().to_vec())
            .collect();
        let mut a_k_tau_j_mats: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; k_count]; psi_dim];
        let mut ds_k_dtau_j_mats: Vec<Vec<Option<Array2<f64>>>> =
            vec![vec![None; k_count]; psi_dim];
        for j in 0..psi_dim {
            for component in &penalty_components_per_dir[j] {
                let k = component.penalty_index;
                if k < k_count {
                    a_k_tau_j_mats[j][k] = Some(component.matrix.scaled_materialize(lambdas[k]));
                    ds_k_dtau_j_mats[j][k] = Some(component.matrix.scaled_materialize(1.0));
                }
            }
        }

        let s_tau_tau = std::sync::Arc::new(s_tau_tau);
        let pld = std::sync::Arc::new(pld);
        let beta_eval = std::sync::Arc::new(beta_eval);
        let x_dense = std::sync::Arc::new(x_dense);
        let x_tau_list = std::sync::Arc::new(x_tau_list);
        let x_tau_beta_list = std::sync::Arc::new(x_tau_beta_list);
        let x_tau_tau = std::sync::Arc::new(x_tau_tau);
        let u = std::sync::Arc::new(u);
        let w_diag = std::sync::Arc::new(w_diag);
        let c_array = std::sync::Arc::new(c_array);
        let d_array = std::sync::Arc::new(d_array);
        let a_k_tau_j_mats = std::sync::Arc::new(a_k_tau_j_mats);
        let ds_k_dtau_j_mats = std::sync::Arc::new(ds_k_dtau_j_mats);
        let s_tau_list = std::sync::Arc::new(s_tau_list);

        let s_tau_tau_tt = std::sync::Arc::clone(&s_tau_tau);
        let pld_tt = std::sync::Arc::clone(&pld);
        let s_tau_list_tt = std::sync::Arc::clone(&s_tau_list);
        let beta_tt = std::sync::Arc::clone(&beta_eval);
        let x_dense_tt = std::sync::Arc::clone(&x_dense);
        let x_tau_list_tt = std::sync::Arc::clone(&x_tau_list);
        let x_tau_beta_tt = std::sync::Arc::clone(&x_tau_beta_list);
        let x_tau_tau_tt = std::sync::Arc::clone(&x_tau_tau);
        let u_tt = std::sync::Arc::clone(&u);
        let w_tt = std::sync::Arc::clone(&w_diag);
        let c_tt = std::sync::Arc::clone(&c_array);
        let d_tt = std::sync::Arc::clone(&d_array);
        let p_dim_tt = p_dim;
        let is_gaussian_tt = is_gaussian_identity;

        let tau_tau_pair_fn = move |i: usize, j: usize| -> super::unified::HyperCoordPair {
            let ld_s_ij = pld_tt.tau_hessian_component(
                &s_tau_list_tt[i],
                &s_tau_list_tt[j],
                s_tau_tau_tt[i][j].as_ref().map(|m| m as &Array2<f64>),
            );

            let x_tau_i_beta = &x_tau_beta_tt[i];
            let x_tau_j_beta = &x_tau_beta_tt[j];
            let w_x_tau_j_beta = w_tt.as_ref() * x_tau_j_beta;
            let a_ij_likelihood = x_tau_i_beta.dot(&w_x_tau_j_beta);
            let x_tau_tau_beta = x_tau_tau_tt[i][j]
                .as_ref()
                .map(|xij| xij.dot(beta_tt.as_ref()))
                .unwrap_or_else(|| Array1::<f64>::zeros(u_tt.len()));
            let a_ij_design2 = if x_tau_tau_tt[i][j].is_some() {
                -u_tt.dot(&x_tau_tau_beta)
            } else {
                0.0
            };
            let a_ij_penalty = 0.5
                * s_tau_tau_tt[i][j]
                    .as_ref()
                    .map(|s_ij| beta_tt.dot(&s_ij.dot(beta_tt.as_ref())))
                    .unwrap_or(0.0);
            let a_ij = a_ij_likelihood + a_ij_design2 + a_ij_penalty;

            let x_tau_i = &x_tau_list_tt[i];
            let x_tau_j = &x_tau_list_tt[j];
            let term1 = x_tau_tau_tt[i][j]
                .as_ref()
                .map(|xij| xij.t().dot(u_tt.as_ref()))
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_tt));
            let term2 = x_tau_j.t().dot(&(w_tt.as_ref() * x_tau_i_beta));
            let term3 = x_tau_i.t().dot(&w_x_tau_j_beta);
            let c_x_tau_i_beta = c_tt.as_ref() * x_tau_i_beta;
            let term4 = x_dense_tt.t().dot(&(&c_x_tau_i_beta * x_tau_j_beta));
            let term5 = if x_tau_tau_tt[i][j].is_some() {
                x_dense_tt.t().dot(&(w_tt.as_ref() * &x_tau_tau_beta))
            } else {
                Array1::<f64>::zeros(p_dim_tt)
            };
            let term6 = s_tau_tau_tt[i][j]
                .as_ref()
                .map(|s_ij| s_ij.dot(beta_tt.as_ref()))
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_tt));
            let g_ij = term1 - &term2 - &term3 - &term4 - &term5 - &term6;

            let mut b_ij = Array2::<f64>::zeros((p_dim_tt, p_dim_tt));
            if let Some(xij) = x_tau_tau_tt[i][j].as_ref() {
                let cross = Self::weighted_cross(xij, x_dense_tt.as_ref(), w_tt.as_ref());
                b_ij += &cross;
                b_ij += &cross.t().to_owned();
            }
            b_ij += &Self::weighted_cross(x_tau_i, x_tau_j, w_tt.as_ref());
            b_ij += &Self::weighted_cross(x_tau_j, x_tau_i, w_tt.as_ref());

            if !is_gaussian_tt {
                b_ij += &Self::weighted_cross(x_tau_j, x_dense_tt.as_ref(), &c_x_tau_i_beta);
                b_ij += &Self::weighted_cross(x_dense_tt.as_ref(), x_tau_j, &c_x_tau_i_beta);

                let c_x_tau_j_beta = c_tt.as_ref() * x_tau_j_beta;
                b_ij += &Self::weighted_cross(x_tau_i, x_dense_tt.as_ref(), &c_x_tau_j_beta);
                b_ij += &Self::weighted_cross(x_dense_tt.as_ref(), x_tau_i, &c_x_tau_j_beta);

                let d_cross = d_tt.as_ref() * &(x_tau_i_beta * x_tau_j_beta);
                let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
                b_ij += &Self::xt_diag_x_dense_into(
                    x_dense_tt.as_ref(),
                    &d_cross,
                    &mut weighted_scratch,
                );

                if x_tau_tau_tt[i][j].is_some() {
                    let c_xij_beta = c_tt.as_ref() * &x_tau_tau_beta;
                    b_ij += &Self::xt_diag_x_dense_into(
                        x_dense_tt.as_ref(),
                        &c_xij_beta,
                        &mut weighted_scratch,
                    );
                }
            }

            if let Some(s_ij) = s_tau_tau_tt[i][j].as_ref() {
                b_ij += s_ij;
            }

            super::unified::HyperCoordPair {
                a: a_ij,
                g: g_ij,
                b_mat: b_ij,
                ld_s: ld_s_ij,
            }
        };

        let pld_rt = std::sync::Arc::clone(&pld);
        let canonical_penalties_rt = std::sync::Arc::new(self.canonical_penalties.as_ref().clone());
        let s_tau_list_rt = std::sync::Arc::clone(&s_tau_list);
        let ds_k_dtau_j_rt = std::sync::Arc::clone(&ds_k_dtau_j_mats);
        let lambdas_rt = lambdas.clone();
        let a_k_tau_j_rt = std::sync::Arc::clone(&a_k_tau_j_mats);
        let beta_rt = std::sync::Arc::clone(&beta_eval);
        let p_dim_rt = p_dim;

        let rho_tau_pair_fn = move |k: usize, j: usize| -> super::unified::HyperCoordPair {
            let ld_s_kj = if k < canonical_penalties_rt.len() {
                pld_rt.rho_tau_hessian_component_block_local(
                    &canonical_penalties_rt[k],
                    lambdas_rt[k],
                    &s_tau_list_rt[j],
                    ds_k_dtau_j_rt[j][k].as_ref().map(|m| m as &Array2<f64>),
                )
            } else {
                0.0
            };

            let a_kj = 0.5
                * a_k_tau_j_rt[j][k]
                    .as_ref()
                    .map(|a_kt| beta_rt.dot(&a_kt.dot(beta_rt.as_ref())))
                    .unwrap_or(0.0);
            let g_kj = a_k_tau_j_rt[j][k]
                .as_ref()
                .map(|a_kt| -(a_kt.dot(beta_rt.as_ref())))
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_rt));
            let b_kj = a_k_tau_j_rt[j][k]
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Array2::<f64>::zeros((p_dim_rt, p_dim_rt)));

            super::unified::HyperCoordPair {
                a: a_kj,
                g: g_kj,
                b_mat: b_kj,
                ld_s: ld_s_kj,
            }
        };

        Ok((Box::new(tau_tau_pair_fn), Box::new(rho_tau_pair_fn)))
    }

    pub(crate) fn build_tau_pair_callbacks_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        self.build_tau_pair_callbacks_original_basis(rho, bundle, hyper_dirs)
    }

    /// Build pair callbacks for τ×τ and ρ×τ second-order [`HyperCoordPair`]
    /// entries.
    ///
    /// The returned closures produce `HyperCoordPair` objects on demand for:
    /// - `tau_tau_pair_fn(i, j)` : τ_i × τ_j entries
    /// - `rho_tau_pair_fn(k, j)` : ρ_k × τ_j entries
    ///
    /// These produce the second-order pair data used by the unified
    /// evaluator for τ-τ and ρ-τ Hessian blocks.
    ///
    /// # Field derivation (τ_i × τ_j pair)
    ///
    /// | Field   | Formula |
    /// | `a`     | `β̂^T S_{τ_i} β_{τ_j} + 0.5 β̂^T S_{τ_i τ_j} β̂` |
    /// | `g`     | second score involving X_{τ_i}, X_{τ_j}, X_{τ_i τ_j}, S_{τ_i τ_j}` |
    /// | `B`     | cross-design + cross-curvature + second-design + S_{τ_i τ_j}` |
    /// | `ld_s`  | `tr(S⁺ S_{τ_i τ_j}) − tr(S⁺ S_{τ_i} S⁺ S_{τ_j}) + 2 tr(Σ₊⁻² L_i L_jᵀ)` |
    ///
    /// # Notes
    ///
    /// The closures capture transformed design matrices, penalty objects, and
    /// the PIRLS state. They are `Send + Sync` so that the unified evaluator
    /// can call them from parallel contexts.
    ///
    /// All `a`, `g`, `B` fields are fixed-beta objects: they depend only on
    /// the current beta-hat and hyperparameters, not on IFT-derived mode
    /// responses. The IFT solves (beta_i, beta_j, beta_ij) are handled
    /// entirely by the unified evaluator after receiving these pair objects.
    pub(crate) fn build_tau_pair_callbacks(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
        }
        let p_dim = beta_eval.len();
        let psi_dim = hyper_dirs.len();
        let k_count = rho.len();
        let lambdas = rho.mapv(f64::exp);

        // Pre-transform first-order penalty components for each τ direction.
        let s_tau_list: Vec<Array2<f64>> = hyper_dirs
            .iter()
            .map(|dir| {
                let components = Self::transform_penalty_components(
                    dir.penalty_first_components(),
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                );
                components
                    .iter()
                    .try_fold(Array2::<f64>::zeros((p_dim, p_dim)), |mut acc, c| {
                        if c.penalty_index >= rho.len() {
                            return Err(EstimationError::InvalidInput(format!(
                                "penalty_index {} out of bounds for rho dimension {} in build_tau_pair_callbacks (first-order)",
                                c.penalty_index,
                                rho.len()
                            )));
                        }
                        c.matrix
                            .scaled_add_to(&mut acc, rho[c.penalty_index].exp())?;
                        Ok(acc)
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Pre-compute second-order penalty matrices S_{τ_i τ_j} for all pairs.
        let mut s_tau_tau: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let second_components =
                    Self::get_pairwisesecond_penalty_components(hyper_dirs, i, j);
                if second_components.is_empty() {
                    continue;
                }
                let transformed = Self::transform_penalty_components(
                    &second_components,
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                );
                let total =
                    transformed
                        .iter()
                        .try_fold(Array2::<f64>::zeros((p_dim, p_dim)), |mut acc, c| {
                            if c.penalty_index >= rho.len() {
                                return Err(EstimationError::InvalidInput(format!(
                                    "penalty_index {} out of bounds for rho dimension {} in build_tau_pair_callbacks (second-order)",
                                    c.penalty_index,
                                    rho.len()
                                )));
                            }
                            c.matrix
                                .scaled_add_to(&mut acc, rho[c.penalty_index].exp())?;
                            Ok(acc)
                        })?;
                s_tau_tau[i][j] = Some(total);
            }
        }

        // ── Build PenaltyPseudologdet for ld_s pair computations ──
        //
        // Canonical penalty pseudo-logdeterminant: eigendecomposes S once and
        // provides tau_hessian_component / rho_tau_hessian_component methods
        // for all derivative queries on log|S|₊.
        let ct_eval: Vec<crate::construction::CanonicalPenalty> =
            if let Some(z) = free_basis_opt.as_ref() {
                reparam_result
                    .canonical_transformed
                    .iter()
                    .map(|cp| {
                        let projected_root = cp.root.dot(z);
                        crate::construction::CanonicalPenalty::from_dense_root(
                            projected_root,
                            z.ncols(),
                        )
                    })
                    .collect()
            } else {
                reparam_result.canonical_transformed.clone()
            };
        let mut s_eval = Array2::<f64>::zeros((p_dim, p_dim));
        for (k, cp) in ct_eval.iter().enumerate() {
            cp.accumulate_weighted(&mut s_eval, lambdas[k]);
        }
        let pld = super::penalty_logdet::PenaltyPseudologdet::from_assembled(s_eval)
            .map_err(EstimationError::InvalidInput)?;

        // Unscaled penalty component matrices S_k = R_k^T R_k for ρ-τ pairs.
        let s_k_unscaled: Vec<Array2<f64>> = ct_eval
            .iter()
            .map(|cp| {
                let r = cp.full_width_root();
                r.t().dot(&r)
            })
            .collect();

        // Pre-compute transformed design matrices X_{τ_j} for each τ direction.
        let x_dense_arc = pirls_result
            .x_transformed
            .try_to_dense_arc("build_tau_pair_callbacks requires dense transformed design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense: Array2<f64> = x_dense_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_arc.as_ref())
            .clone();

        let x_tau_list: Vec<Array2<f64>> = hyper_dirs
            .iter()
            .map(|dir| {
                dir.transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())
                    .expect("valid transformed X_tau in build_tau_pair_callbacks")
            })
            .collect();

        // Pre-compute X_{τ_i β̂} for each τ direction.
        let x_tau_beta_list: Vec<Array1<f64>> = x_tau_list
            .iter()
            .map(|x_tau| x_tau.dot(&beta_eval))
            .collect();

        let qs_eval = std::sync::Arc::new(reparam_result.qs.clone());
        let free_basis_eval: std::sync::Arc<Option<Array2<f64>>> =
            std::sync::Arc::new(free_basis_opt.clone());

        // Pre-compute second-order design terms for all pairs.
        // Dense derivatives are transformed eagerly. Implicit derivatives keep
        // the original operator-backed representation and are evaluated through
        // transformed forward/transpose products inside the pair callback.
        let mut x_tau_tau: Vec<Vec<Option<TauTauDesignTerm>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in i..psi_dim {
                let xij =
                    hyper_dirs[i]
                        .x_tau_tau_entry_at(j)
                        .or_else(|| hyper_dirs[j].x_tau_tau_entry_at(i))
                        .map(|entry| {
                            if entry.uses_implicit_storage() {
                                TauTauDesignTerm::Implicit(entry)
                            } else {
                                TauTauDesignTerm::Dense(
                            entry
                                .transformed(&reparam_result.qs, free_basis_opt.as_ref())
                                .expect("valid transformed X_tau_tau in build_tau_pair_callbacks"),
                        )
                            }
                        });
                if xij.is_some() {
                    x_tau_tau[j][i] = xij.clone();
                }
                x_tau_tau[i][j] = xij;
            }
        }

        // Working residual u = w ⊙ (z − η̂).
        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let w_diag = pirls_result.finalweights.clone();
        let c_array = pirls_result.solve_c_array.clone();
        let d_array = pirls_result.solve_d_array.clone();
        let is_gaussian_identity = matches!(self.config.link_function(), LinkFunction::Identity);

        // Pre-compute per-penalty-index components of S_{τ_j} for ρ-τ cross
        // derivative A_{k,τ_j} = λ_k (dS_k/dτ_j).
        // For each τ direction j and penalty index k, store the transformed
        // penalty derivative matrix (unscaled by λ_k).
        let penalty_components_per_dir: Vec<Vec<PenaltyDerivativeComponent>> = hyper_dirs
            .iter()
            .map(|dir| {
                Self::transform_penalty_components(
                    dir.penalty_first_components(),
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                )
            })
            .collect();

        // Build A_{k,τ_j} = λ_k * (component of S_{τ_j} at penalty k).
        // Stored as a_k_tau_j[j][k]: Option<Array2<f64>>.
        let mut a_k_tau_j_mats: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; k_count]; psi_dim];
        let mut ds_k_dtau_j_mats: Vec<Vec<Option<Array2<f64>>>> =
            vec![vec![None; k_count]; psi_dim];
        for j in 0..psi_dim {
            for component in &penalty_components_per_dir[j] {
                let k = component.penalty_index;
                if k < k_count {
                    a_k_tau_j_mats[j][k] = Some(component.matrix.scaled_materialize(lambdas[k]));
                    ds_k_dtau_j_mats[j][k] = Some(component.matrix.scaled_materialize(1.0));
                }
            }
        }

        // Capture into Arc for shared ownership in closures.
        let s_tau_tau = Arc::new(s_tau_tau);
        let pld = Arc::new(pld);
        let s_k_unscaled = Arc::new(s_k_unscaled);
        let s_tau_list = Arc::new(s_tau_list);
        let beta_eval = Arc::new(beta_eval);
        let x_dense = Arc::new(x_dense);
        let x_tau_list = Arc::new(x_tau_list);
        let x_tau_beta_list = Arc::new(x_tau_beta_list);
        let x_tau_tau = Arc::new(x_tau_tau);
        let qs_eval_tt = Arc::clone(&qs_eval);
        let free_basis_tt = Arc::clone(&free_basis_eval);
        let u = Arc::new(u);
        let w_diag = Arc::new(w_diag);
        let c_array = Arc::new(c_array);
        let d_array = Arc::new(d_array);
        let a_k_tau_j_mats = Arc::new(a_k_tau_j_mats);
        let ds_k_dtau_j_mats = Arc::new(ds_k_dtau_j_mats);

        // ─── τ×τ pair callback ───────────────────────────────────────────
        let s_tau_tau_tt = Arc::clone(&s_tau_tau);
        let pld_tt = Arc::clone(&pld);
        let s_tau_list_tt = Arc::clone(&s_tau_list);
        let beta_tt = Arc::clone(&beta_eval);
        let x_dense_tt = Arc::clone(&x_dense);
        let x_tau_list_tt = Arc::clone(&x_tau_list);
        let x_tau_beta_tt = Arc::clone(&x_tau_beta_list);
        let x_tau_tau_tt = Arc::clone(&x_tau_tau);
        let u_tt = Arc::clone(&u);
        let w_tt = Arc::clone(&w_diag);
        let c_tt = Arc::clone(&c_array);
        let d_tt = Arc::clone(&d_array);
        let p_dim_tt = p_dim;
        let is_gaussian_tt = is_gaussian_identity;

        let tau_tau_pair_fn = move |i: usize, j: usize| -> super::unified::HyperCoordPair {
            let ld_s_ij = pld_tt.tau_hessian_component(
                &s_tau_list_tt[i],
                &s_tau_list_tt[j],
                s_tau_tau_tt[i][j].as_ref().map(|m| m as &Array2<f64>),
            );

            // a_ij — fixed-β second-order cost derivative.
            //
            // ∂²F/∂τ_i ∂τ_j|_β where F = -ℓ + 0.5 β^T S β.
            //
            // Likelihood part: (X_{τ_i} β̂)^T W (X_{τ_j} β̂)
            //   This arises from -ℓ''(η)[η_{τ_i}, η_{τ_j}] at fixed β,
            //   where ℓ''(η) = -W (working weights).
            //
            // Second-design part: -u^T(X_{τ_i τ_j} β̂)
            //   From -ℓ'(η)^T η_{τ_i τ_j} where ℓ'(η) = u.
            //
            // Penalty part: 0.5 β̂^T S_{τ_i τ_j} β̂.
            let x_tau_i_beta = &x_tau_beta_tt[i];
            let x_tau_j_beta = &x_tau_beta_tt[j];
            let w_x_tau_j_beta = w_tt.as_ref() * x_tau_j_beta;
            let a_ij_likelihood = x_tau_i_beta.dot(&w_x_tau_j_beta);
            let x_tau_tau_beta = match x_tau_tau_tt[i][j].as_ref() {
                Some(TauTauDesignTerm::Dense(xij)) => xij.dot(beta_tt.as_ref()),
                Some(TauTauDesignTerm::Implicit(deriv)) => deriv
                    .transformed_forward_mul(
                        qs_eval_tt.as_ref(),
                        free_basis_tt.as_ref().as_ref(),
                        beta_tt.as_ref(),
                    )
                    .expect("valid transformed implicit X_tau_tau beta product"),
                None => Array1::<f64>::zeros(u_tt.len()),
            };
            let a_ij_design2 = if x_tau_tau_tt[i][j].is_some() {
                -u_tt.dot(&x_tau_tau_beta)
            } else {
                0.0
            };
            let a_ij_penalty = 0.5
                * s_tau_tau_tt[i][j]
                    .as_ref()
                    .map(|s_ij| beta_tt.dot(&s_ij.dot(beta_tt.as_ref())))
                    .unwrap_or(0.0);
            let a_ij = a_ij_likelihood + a_ij_design2 + a_ij_penalty;

            // g_ij — fixed-β second-order score vector.
            //
            // ∂²/∂τ_i ∂τ_j [X^T u − S β]|_β (the score ∇_β ℓ_p).
            //
            // Derivation by differentiating g_j = X_{τ_j}^T u − X^T W(X_{τ_j} β̂)
            //   − S_{τ_j} β̂ w.r.t. τ_i at fixed β:
            //
            //   term1: X_{τ_i τ_j}^T u  (second design on score)
            //   term2: -X_{τ_j}^T diag(w)(X_{τ_i} β̂)  (u drift from η_{τ_i})
            //   term3: -X_{τ_i}^T W(X_{τ_j} β̂)  (design drift from X_{τ_i})
            //   term4: -X^T diag(c ⊙ X_{τ_i} β̂)(X_{τ_j} β̂)  (weight drift)
            //   term5: -X^T W(X_{τ_i τ_j} β̂)  (second design on Hessian-score)
            //   term6: -S_{τ_i τ_j} β̂  (second penalty)
            let x_tau_i = &x_tau_list_tt[i];
            let x_tau_j = &x_tau_list_tt[j];

            // term1: X_{τ_i τ_j}^T u
            let term1 = match x_tau_tau_tt[i][j].as_ref() {
                Some(TauTauDesignTerm::Dense(xij)) => xij.t().dot(u_tt.as_ref()),
                Some(TauTauDesignTerm::Implicit(deriv)) => deriv
                    .transformed_transpose_mul(
                        qs_eval_tt.as_ref(),
                        free_basis_tt.as_ref().as_ref(),
                        u_tt.as_ref(),
                    )
                    .expect("valid transformed implicit X_tau_tau^T u product"),
                None => Array1::<f64>::zeros(p_dim_tt),
            };

            // term2: -X_{τ_j}^T diag(w)(X_{τ_i} β̂)
            let term2 = x_tau_j.t().dot(&(w_tt.as_ref() * x_tau_i_beta));

            // term3: -X_{τ_i}^T W(X_{τ_j} β̂)
            let term3 = x_tau_i.t().dot(&w_x_tau_j_beta);

            // term4: -X^T diag(c ⊙ X_{τ_i} β̂)(X_{τ_j} β̂)
            let c_x_tau_i_beta = c_tt.as_ref() * x_tau_i_beta;
            let term4 = x_dense_tt.t().dot(&(&c_x_tau_i_beta * x_tau_j_beta));

            // term5: -X^T W(X_{τ_i τ_j} β̂)
            let term5 = if x_tau_tau_tt[i][j].is_some() {
                x_dense_tt.t().dot(&(w_tt.as_ref() * &x_tau_tau_beta))
            } else {
                Array1::<f64>::zeros(p_dim_tt)
            };

            // term6: -S_{τ_i τ_j} β̂
            let term6 = s_tau_tau_tt[i][j]
                .as_ref()
                .map(|s_ij| s_ij.dot(beta_tt.as_ref()))
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_tt));

            let g_ij = term1 - &term2 - &term3 - &term4 - &term5 - &term6;

            // B_ij — fixed-β second-order Hessian drift.
            //
            // ∂²H/∂τ_i ∂τ_j|_β where H = X^T W X + S.
            //
            // Components (all at fixed β, with η_{τ_i}|_β = X_{τ_i} β̂):
            //
            //   (a) X_{τ_i τ_j}^T W X + X^T W X_{τ_i τ_j}   [second-design]
            //   (b) X_{τ_i}^T W X_{τ_j} + X_{τ_j}^T W X_{τ_i}  [cross-design]
            //   (c) X_{τ_j}^T diag(c ⊙ X_{τ_i} β̂) X
            //       + X^T diag(c ⊙ X_{τ_i} β̂) X_{τ_j}  [weight drift × design_j]
            //   (d) X_{τ_i}^T diag(c ⊙ X_{τ_j} β̂) X
            //       + X^T diag(c ⊙ X_{τ_j} β̂) X_{τ_i}  [weight drift × design_i]
            //   (e) X^T diag(d ⊙ (X_{τ_i} β̂) ⊙ (X_{τ_j} β̂)) X  [curvature]
            //   (f) X^T diag(c ⊙ X_{τ_i τ_j} β̂) X  [second-design weight]
            //   (g) S_{τ_i τ_j}  [penalty]
            let mut b_ij = Array2::<f64>::zeros((p_dim_tt, p_dim_tt));

            // (a) second-design terms
            if let Some(xij) = x_tau_tau_tt[i][j].as_ref() {
                let cross = match xij {
                    TauTauDesignTerm::Dense(xij) => {
                        Self::weighted_cross(xij, x_dense_tt.as_ref(), w_tt.as_ref())
                    }
                    TauTauDesignTerm::Implicit(deriv) => {
                        Self::weighted_cross_from_design_derivative(
                            deriv,
                            qs_eval_tt.as_ref(),
                            free_basis_tt.as_ref().as_ref(),
                            x_dense_tt.as_ref(),
                            w_tt.as_ref(),
                        )
                        .expect("valid transformed implicit weighted cross")
                    }
                };
                b_ij += &cross;
                b_ij += &cross.t().to_owned();
            }

            // (b) cross-design terms
            b_ij += &Self::weighted_cross(x_tau_i, x_tau_j, w_tt.as_ref());
            b_ij += &Self::weighted_cross(x_tau_j, x_tau_i, w_tt.as_ref());

            if !is_gaussian_tt {
                // (c) weight drift × design_j
                b_ij += &Self::weighted_cross(x_tau_j, x_dense_tt.as_ref(), &c_x_tau_i_beta);
                b_ij += &Self::weighted_cross(x_dense_tt.as_ref(), x_tau_j, &c_x_tau_i_beta);

                // (d) weight drift × design_i
                let c_x_tau_j_beta = c_tt.as_ref() * x_tau_j_beta;
                b_ij += &Self::weighted_cross(x_tau_i, x_dense_tt.as_ref(), &c_x_tau_j_beta);
                b_ij += &Self::weighted_cross(x_dense_tt.as_ref(), x_tau_i, &c_x_tau_j_beta);

                // (e) curvature: X^T diag(d ⊙ (X_{τ_i} β̂) ⊙ (X_{τ_j} β̂)) X
                let d_cross = d_tt.as_ref() * &(x_tau_i_beta * x_tau_j_beta);
                let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
                b_ij += &Self::xt_diag_x_dense_into(
                    x_dense_tt.as_ref(),
                    &d_cross,
                    &mut weighted_scratch,
                );

                // (f) second-design weight: X^T diag(c ⊙ X_{τ_i τ_j} β̂) X
                if x_tau_tau_tt[i][j].is_some() {
                    let c_xij_beta = c_tt.as_ref() * &x_tau_tau_beta;
                    b_ij += &Self::xt_diag_x_dense_into(
                        x_dense_tt.as_ref(),
                        &c_xij_beta,
                        &mut weighted_scratch,
                    );
                }
            }

            // (g) penalty
            if let Some(s_ij) = s_tau_tau_tt[i][j].as_ref() {
                b_ij += s_ij;
            }

            super::unified::HyperCoordPair {
                a: a_ij,
                g: g_ij,
                b_mat: b_ij,
                ld_s: ld_s_ij,
            }
        };

        // ─── ρ×τ pair callback ───────────────────────────────────────────
        let pld_rt = Arc::clone(&pld);
        let s_k_unscaled_rt = Arc::clone(&s_k_unscaled);
        let s_tau_list_rt = Arc::clone(&s_tau_list);
        let ds_k_dtau_j_rt = Arc::clone(&ds_k_dtau_j_mats);
        let lambdas_rt = lambdas.clone();
        let a_k_tau_j_rt = Arc::clone(&a_k_tau_j_mats);
        let beta_rt = Arc::clone(&beta_eval);
        let p_dim_rt = p_dim;

        let rho_tau_pair_fn = move |k: usize, j: usize| -> super::unified::HyperCoordPair {
            // ld_s_{k,τ_j}: second derivative of log|S|₊ w.r.t. ρ_k, τ_j.
            //
            // Uses PenaltyPseudologdet::rho_tau_hessian_component for the
            // canonical formula:
            //   ∂²_{ρ_k, τ_j} L = λ_k [tr(S⁺ dS_k/dτ_j) − tr(S⁺ S_k S⁺ S_{τ_j})]
            let ld_s_kj = if k < s_k_unscaled_rt.len() {
                pld_rt.rho_tau_hessian_component(
                    &s_k_unscaled_rt[k],
                    lambdas_rt[k],
                    &s_tau_list_rt[j],
                    ds_k_dtau_j_rt[j][k].as_ref().map(|m| m as &Array2<f64>),
                )
            } else {
                0.0
            };

            // a_kj — fixed-β mixed cost derivative.
            //
            // ∂²F/∂ρ_k ∂τ_j|_β = 0.5 β̂^T A_{k,τ_j} β̂.
            //
            // The ∂F/∂ρ_k|_β = 0.5 β̂^T A_k β̂ term, when differentiated
            // w.r.t. τ_j at fixed β, gives 0.5 β̂^T A_{k,τ_j} β̂ from the
            // penalty change. The likelihood doesn't directly depend on ρ_k
            // at fixed β, so there is no likelihood cross term.
            let a_kj = 0.5
                * a_k_tau_j_rt[j][k]
                    .as_ref()
                    .map(|a_kt| beta_rt.dot(&a_kt.dot(beta_rt.as_ref())))
                    .unwrap_or(0.0);

            // g_kj — fixed-β mixed score vector.
            //
            // ∂²/∂ρ_k ∂τ_j [∇_β(ℓ_p)]|_β.
            //
            // The first-order ρ score is g_k = -A_k β̂. Differentiating w.r.t.
            // τ_j at fixed β gives -A_{k,τ_j} β̂ (since A_k doesn't depend on
            // τ, but A_{k,τ_j} is the cross derivative of the penalty).
            let g_kj = a_k_tau_j_rt[j][k]
                .as_ref()
                .map(|a_kt| -(a_kt.dot(beta_rt.as_ref())))
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_rt));

            // B_kj — fixed-β mixed Hessian drift.
            //
            // ∂²H/∂ρ_k ∂τ_j|_β.
            //
            // The ρ_k Hessian drift is B_k = A_k (β-independent for the
            // penalty term). For non-Gaussian families, B_k also includes
            // X^T diag(c ⊙ X B_k^{IFT}) X, but that term depends on the
            // IFT solve B_k^{IFT} = H^{-1}(-A_k β̂) and is therefore NOT
            // a fixed-β object — it is handled by the unified evaluator
            // via the drift derivative callback.
            //
            // The fixed-β part is simply A_{k,τ_j} = λ_k (dS_k/dτ_j),
            // since A_k itself does not depend on τ (the penalty matrices
            // R_k^T R_k are τ-independent in the canonical decomposition).
            let b_kj = a_k_tau_j_rt[j][k]
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Array2::<f64>::zeros((p_dim_rt, p_dim_rt)));

            super::unified::HyperCoordPair {
                a: a_kj,
                g: g_kj,
                b_mat: b_kj,
                ld_s: ld_s_kj,
            }
        };

        Ok((Box::new(tau_tau_pair_fn), Box::new(rho_tau_pair_fn)))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified HyperCoord builders for link parameters (SAS ε/log δ, mixture ρ)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build [`HyperCoord`] objects for SAS link parameters (ε, log δ).
    ///
    /// For each SAS parameter θ_j (j=0: epsilon, j=1: log_delta), the
    /// fixed-β derivative objects are:
    ///
    /// | Field   | Formula |
    /// | `a`     | `−∂ℓ/∂θ_j\|_{η fixed}` (direct likelihood derivative) |
    /// | `g`     | `−X^T (∂u/∂θ_j)\|_{η fixed}` (score sensitivity) |
    /// | `B`     | `X^T diag(∂W_explicit/∂θ_j) X` (working-weight drift) |
    /// | `ld_s`  | `0` (penalties don't depend on link parameters) |
    ///
    /// The working-weight derivative `∂W/∂θ_j` uses **observed information**
    /// weights, not Fisher. For a non-canonical link:
    ///   W_obs = W_F − (y−μ)·B,  B = (h''V − h'²V') / V²
    /// so the θ-derivative includes the residual correction:
    ///   ∂W_obs/∂θ = ∂W_F/∂θ + (dμ/dθ)·B − (y−μ)·∂B/∂θ
    ///
    /// This is exact for REML/LAML evaluation with learnable link
    /// parameters. The IFT-mediated `c ⊙ X dβ/dθ` part is handled
    /// separately by the unified evaluator's third-derivative correction.
    ///
    /// SAS epsilon reparameterization (tanh bounding) is NOT applied here;
    /// the caller should apply the chain rule `grad[ε_raw] *= d_eps/d_raw`
    /// after the unified evaluator returns.
    pub(crate) fn build_sas_link_ext_coords(
        &self,
        bundle: &EvalShared,
        sas_state: &crate::types::SasLinkState,
        is_beta_logistic: bool,
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        use crate::mixture_link::{
            beta_logistic_inverse_link_jetwith_param_partials,
            sas_inverse_link_jetwith_param_partials,
        };

        let pirls_result = bundle.pirls_result.as_ref();
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        // Guard: SAS link ext coords require dense design materialization.
        let n_x = pirls_result.x_transformed.nrows();
        let p_x = pirls_result.x_transformed.ncols();
        const LINK_EXT_MAX_DENSE_WORK: usize = 50_000_000;
        if n_x.saturating_mul(p_x) > LINK_EXT_MAX_DENSE_WORK {
            log::warn!(
                "skipping SAS link ext coordinate construction (n={n_x}, p={p_x}): \
                 dense design materialization too large"
            );
            return Ok(Vec::new());
        }

        // Transformed design matrix (dense required for link-param B construction).
        let x_dense_arc = pirls_result
            .x_transformed
            .try_to_dense_arc("build_sas_link_ext_coords requires dense transformed design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense = x_dense_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_arc.as_ref());

        let p_dim = x_dense.ncols();
        let nobs = pirls_result.final_eta.len();
        let aux_dim = 2usize; // epsilon, log_delta

        if p_dim == 0 {
            return Ok((0..aux_dim)
                .map(|_| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: true,
                    is_penalty_like: false,
                })
                .collect());
        }

        // Per-observation link jet with parameter partials.
        let mut direct_ll = [0.0_f64; 2];
        let mut du_by_j = [Array1::<f64>::zeros(nobs), Array1::<f64>::zeros(nobs)];
        let mut dw_explicit_by_j = [Array1::<f64>::zeros(nobs), Array1::<f64>::zeros(nobs)];

        for i in 0..nobs {
            let eta_i = pirls_result.final_eta[i].clamp(-30.0, 30.0);
            let jets = if is_beta_logistic {
                beta_logistic_inverse_link_jetwith_param_partials(
                    eta_i,
                    sas_state.log_delta,
                    sas_state.epsilon,
                )
            } else {
                sas_inverse_link_jetwith_param_partials(
                    eta_i,
                    sas_state.epsilon,
                    sas_state.log_delta,
                )
            };
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let yi = self.y[i];
            let wi = self.weights[i].max(0.0);
            let aux = link_binomial_aux(yi, wi, mu);

            for j in 0..aux_dim {
                let dj = if j == 0 {
                    jets.djet_depsilon
                } else {
                    jets.djet_dlog_delta
                };
                let dmu = dj.mu;
                let dd1 = dj.d1;
                let dd2 = dj.d2;
                direct_ll[j] += aux.a1 * dmu;
                du_by_j[j][i] = aux.a2 * dmu * d1 + aux.a1 * dd1;
                let variance = aux.variance;
                let variance_param = aux.variancemu_scale * dmu;
                let numerator = d1 * d1;
                let numerator_param = 2.0 * d1 * dd1;
                // Fisher weight derivative: ∂W_F/∂θ = ∂(h'²/V)/∂θ
                let dw_fisher = wi * (numerator_param * variance - numerator * variance_param)
                    / (variance * variance);

                // Observed correction: W_obs = W_F − (y−μ)·B where
                //   B = (h''·V − h'²·V') / V²
                // so ∂W_obs/∂θ = ∂W_F/∂θ + (dμ/dθ)·B − (y−μ)·∂B/∂θ
                //
                // For binomial: V = μ(1−μ), V' = 1−2μ, V'' = −2.
                let h2 = jets.jet.d2;
                let resid = yi - mu;
                let v_prime = aux.variancemu_scale; // 1 − 2μ
                let v_dprime = -2.0_f64; // V''(μ) for binomial

                let b_num = h2 * variance - numerator * v_prime;
                let var_sq = variance * variance;
                let b_val = b_num / var_sq;

                // ∂B/∂θ via quotient rule on b_num / V²:
                //   dV/dθ  = V'·dμ/dθ
                //   dV'/dθ = V''·dμ/dθ
                //   d(b_num)/dθ = dd2·V + h''·dV − 2·h'·dd1·V' − h'²·dV'
                let d_var = v_prime * dmu; // dV/dθ
                let d_vprime = v_dprime * dmu; // dV'/dθ
                let db_num =
                    dd2 * variance + h2 * d_var - 2.0 * d1 * dd1 * v_prime - numerator * d_vprime;
                // Quotient rule: (db_num·V² − b_num·2V·dV) / V⁴
                //              = (db_num − 2·B·dV) / V²
                let db_val = (db_num - 2.0 * b_val * d_var * variance) / var_sq;

                // OBSERVED weight derivative for the outer REML (see response.md Section 3):
                //   dW_obs/dtheta = dW_Fisher/dtheta + (dmu/dtheta)*B - (y-mu)*dB/dtheta
                // This is the exact Laplace derivative, not the PQL surrogate.
                dw_explicit_by_j[j][i] = dw_fisher + wi * (dmu * b_val - resid * db_val);
            }
        }

        // Build HyperCoord for each link parameter.
        let mut coords = Vec::with_capacity(aux_dim);
        let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
        for j in 0..aux_dim {
            // a_j = dF/dθ_j|_{β fixed} = -dℓ/dθ_j|_{η fixed}.
            let a_j = -direct_ll[j];

            // g_j = -X^T (du/dθ_j) — the score sensitivity at fixed β.
            let g_j = {
                let xt_du = x_dense.t().dot(&du_by_j[j]);
                -xt_du
            };

            // B_j = X^T diag(dw_obs_j) X — the fixed-β observed Hessian drift.
            // Uses observed-information weight derivatives (not Fisher) for exact
            // REML/LAML with non-canonical links (SAS, beta-logistic).
            let b_j =
                Self::xt_diag_x_dense_into(x_dense, &dw_explicit_by_j[j], &mut weighted_scratch);

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_dense(b_j),
                ld_s: 0.0,
                // Link parameters affect working weights through the link,
                // and the working weights depend on β through η = Xβ,
                // so B_j depends on β.
                b_depends_on_beta: true,
                // Link parameters are design-moving (not penalty-like):
                // they change W through the link function, not through
                // penalty matrix derivatives. Not eligible for EFS.
                is_penalty_like: false,
            });
        }

        Ok(coords)
    }

    /// Build [`HyperCoord`] objects for mixture (blended inverse-link) logit
    /// parameters.
    ///
    /// Similar structure to SAS coords but with K−1 free logits instead of
    /// (ε, log δ). Each logit ρ_j controls the softmax weight of the j-th
    /// component link function.
    ///
    /// Uses **observed information** weight derivatives (not Fisher) for
    /// exact REML/LAML with non-canonical links. See the doc comment on
    /// `build_sas_link_ext_coords` for the mathematical derivation.
    pub(crate) fn build_mixture_link_ext_coords(
        &self,
        bundle: &EvalShared,
        mix_state: &crate::types::MixtureLinkState,
    ) -> Result<Vec<super::unified::HyperCoord>, EstimationError> {
        use crate::mixture_link::mixture_inverse_link_jetwith_rho_partials_into;

        let pirls_result = bundle.pirls_result.as_ref();
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        // Guard: mixture link ext coords require dense design materialization.
        let n_x = pirls_result.x_transformed.nrows();
        let p_x = pirls_result.x_transformed.ncols();
        const LINK_EXT_MAX_DENSE_WORK: usize = 50_000_000;
        if n_x.saturating_mul(p_x) > LINK_EXT_MAX_DENSE_WORK {
            log::warn!(
                "skipping mixture link ext coordinate construction (n={n_x}, p={p_x}): \
                 dense design materialization too large"
            );
            return Ok(Vec::new());
        }

        let x_dense_arc = pirls_result
            .x_transformed
            .try_to_dense_arc("build_mixture_link_ext_coords requires dense transformed design")
            .map_err(EstimationError::InvalidInput)?;
        let x_dense_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense = x_dense_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_arc.as_ref());

        let p_dim = x_dense.ncols();
        let nobs = pirls_result.final_eta.len();
        let aux_dim = mix_state.rho.len(); // K-1 free logits

        if aux_dim == 0 {
            return Ok(Vec::new());
        }

        if p_dim == 0 {
            return Ok((0..aux_dim)
                .map(|_| super::unified::HyperCoord {
                    a: 0.0,
                    g: Array1::zeros(0),
                    drift: super::unified::HyperCoordDrift::none(),
                    ld_s: 0.0,
                    b_depends_on_beta: true,
                    is_penalty_like: false,
                })
                .collect());
        }

        let mut direct_ll = vec![0.0_f64; aux_dim];
        let mut du_by_j: Vec<Array1<f64>> =
            (0..aux_dim).map(|_| Array1::<f64>::zeros(nobs)).collect();
        let mut dw_explicit_by_j: Vec<Array1<f64>> =
            (0..aux_dim).map(|_| Array1::<f64>::zeros(nobs)).collect();
        let mut mix_partials = vec![
            crate::mixture_link::InverseLinkJet {
                mu: 0.0,
                d1: 0.0,
                d2: 0.0,
                d3: 0.0,
            };
            aux_dim
        ];

        for i in 0..nobs {
            let jet = mixture_inverse_link_jetwith_rho_partials_into(
                mix_state,
                pirls_result.final_eta[i],
                &mut mix_partials,
            );
            let mu = jet.mu;
            let d1 = jet.d1;
            let yi = self.y[i];
            let wi = self.weights[i].max(0.0);
            let aux = link_binomial_aux(yi, wi, mu);

            for j in 0..aux_dim {
                let dj = mix_partials[j];
                let dmu = dj.mu;
                let dd1 = dj.d1;
                let dd2 = dj.d2;
                direct_ll[j] += aux.a1 * dmu;
                du_by_j[j][i] = aux.a2 * dmu * d1 + aux.a1 * dd1;
                let variance = aux.variance;
                let variance_param = aux.variancemu_scale * dmu;
                let numerator = d1 * d1;
                let numerator_param = 2.0 * d1 * dd1;
                // Fisher weight derivative: ∂W_F/∂θ = ∂(h'²/V)/∂θ
                let dw_fisher = wi * (numerator_param * variance - numerator * variance_param)
                    / (variance * variance);

                // Observed correction: W_obs = W_F − (y−μ)·B where
                //   B = (h''·V − h'²·V') / V²
                // so ∂W_obs/∂θ = ∂W_F/∂θ + (dμ/dθ)·B − (y−μ)·∂B/∂θ
                //
                // For binomial: V = μ(1−μ), V' = 1−2μ, V'' = −2.
                let h2 = jet.d2;
                let resid = yi - mu;
                let v_prime = aux.variancemu_scale; // 1 − 2μ
                let v_dprime = -2.0_f64; // V''(μ) for binomial

                let b_num = h2 * variance - numerator * v_prime;
                let var_sq = variance * variance;
                let b_val = b_num / var_sq;

                // ∂B/∂θ via quotient rule on b_num / V²:
                //   dV/dθ  = V'·dμ/dθ
                //   dV'/dθ = V''·dμ/dθ
                //   d(b_num)/dθ = dd2·V + h''·dV − 2·h'·dd1·V' − h'²·dV'
                let d_var = v_prime * dmu; // dV/dθ
                let d_vprime = v_dprime * dmu; // dV'/dθ
                let db_num =
                    dd2 * variance + h2 * d_var - 2.0 * d1 * dd1 * v_prime - numerator * d_vprime;
                let db_val = (db_num - 2.0 * b_val * d_var * variance) / var_sq;

                // OBSERVED weight derivative for the outer REML (see response.md Section 3):
                //   dW_obs/dtheta = dW_Fisher/dtheta + (dmu/dtheta)*B - (y-mu)*dB/dtheta
                // This is the exact Laplace derivative, not the PQL surrogate.
                dw_explicit_by_j[j][i] = dw_fisher + wi * (dmu * b_val - resid * db_val);
            }
        }

        let mut coords = Vec::with_capacity(aux_dim);
        let mut weighted_scratch = Array2::<f64>::zeros((0, 0));
        for j in 0..aux_dim {
            let a_j = -direct_ll[j];
            let g_j = {
                let xt_du = x_dense.t().dot(&du_by_j[j]);
                -xt_du
            };
            // B_j = X^T diag(dw_obs_j) X — the fixed-β observed Hessian drift.
            // Uses observed-information weight derivatives (not Fisher) for exact
            // REML/LAML with non-canonical links (mixture/blended).
            let b_j =
                Self::xt_diag_x_dense_into(x_dense, &dw_explicit_by_j[j], &mut weighted_scratch);

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                drift: super::unified::HyperCoordDrift::from_dense(b_j),
                ld_s: 0.0,
                b_depends_on_beta: true,
                is_penalty_like: false,
            });
        }

        Ok(coords)
    }
}
