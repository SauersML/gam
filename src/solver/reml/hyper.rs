use super::*;
use crate::matrix::DenseRightProductView;

// ─── Binomial auxiliary terms for link-parameter ext_coord construction ───
//
// Mirrors the `BinomialAuxTerms` in estimate.rs but is local to the REML
// module so that hyper.rs can compute per-observation likelihood derivatives
// without depending on the estimate module's private helpers.

const LINK_BINOMIAL_AUX_MU_EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
struct LinkBinomialAux {
    /// dℓ_i/dμ_i = w_i (y_i/μ_i − (1−y_i)/(1−μ_i))
    a1: f64,
    /// d²ℓ_i/dμ_i² = w_i (−y_i/μ_i² − (1−y_i)/(1−μ_i)²)
    a2: f64,
    /// μ(1−μ) — binomial variance function
    variance: f64,
    /// dVar/dμ = 1−2μ
    variancemu_scale: f64,
}

#[inline]
fn link_binomial_aux(yi: f64, wi: f64, mu: f64) -> LinkBinomialAux {
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

    fn add_penalty_components_to_state(
        s_mod: &mut [Array2<f64>],
        components: &[PenaltyDerivativeComponent],
        amp: f64,
    ) -> Result<(), EstimationError> {
        for component in components {
            if component.penalty_index >= s_mod.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "penalty_index {} out of bounds for {} penalties",
                    component.penalty_index,
                    s_mod.len()
                )));
            }
            component
                .matrix
                .scaled_add_to(&mut s_mod[component.penalty_index], amp)?;
        }
        Ok(())
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

    fn get_pairwisesecond_derivative(
        hyper_dirs: &[DirectionalHyperParam],
        i: usize,
        j: usize,
        x_term: bool,
    ) -> Option<Array2<f64>> {
        if !x_term {
            return None;
        }
        hyper_dirs
            .get(i)
            .and_then(|d| d.x_tau_tau_nth_dense(j))
            .or_else(|| hyper_dirs.get(j).and_then(|d| d.x_tau_tau_nth_dense(i)))
    }

    pub(super) fn build_joint_perturbed_state(
        &self,
        psi: &Array1<f64>,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<RemlState<'a>, EstimationError> {
        if psi.len() != hyper_dirs.len() {
            return Err(EstimationError::InvalidInput(format!(
                "psi/hyper_dirs mismatch: psi={}, hyper_dirs={}",
                psi.len(),
                hyper_dirs.len()
            )));
        }
        let has_design_drift = hyper_dirs.iter().enumerate().any(|(j, dir)| {
            if psi[j] == 0.0 {
                return false;
            }
            if dir.x_tau_original.any_nonzero() {
                return true;
            }
            if let Some(x2) = dir.x_tau_tau_original.as_ref() {
                return x2.iter().flatten().any(HyperDesignDerivative::any_nonzero);
            }
            false
        });
        let mut x_mod_dense = if has_design_drift {
            Some(
                self.x()
                    .try_to_dense_arc("perturbed REML state requires dense design for design drift")
                    .map_err(EstimationError::InvalidInput)?
                    .as_ref()
                    .clone(),
            )
        } else {
            None
        };
        let mut s_mod = self.s_full_list.as_ref().clone();
        for (j, dir) in hyper_dirs.iter().enumerate() {
            let amp = psi[j];
            if amp == 0.0 {
                continue;
            }
            if let Some(x_mod) = x_mod_dense.as_ref()
                && (dir.x_tau_original.nrows() != x_mod.nrows()
                    || dir.x_tau_original.ncols() != x_mod.ncols())
            {
                return Err(EstimationError::InvalidInput(format!(
                    "joint perturbation X_tau shape mismatch: expected {}x{}, got {}x{}",
                    x_mod.nrows(),
                    x_mod.ncols(),
                    dir.x_tau_original.nrows(),
                    dir.x_tau_original.ncols()
                )));
            }
            Self::validate_penalty_component_shapes(
                dir.penalty_first_components(),
                self.p,
                "joint perturbation S_tau",
            )?;
            if let Some(x_mod) = x_mod_dense.as_mut() {
                dir.x_tau_original.scaled_add_to(x_mod, amp)?;
            }
            Self::add_penalty_components_to_state(&mut s_mod, dir.penalty_first_components(), amp)?;
        }
        // Optional nonlinear tau parameterization:
        //   X(psi) += 0.5 * sum_{i,j} psi_i psi_j X_{ij},
        //   S(psi) += 0.5 * sum_{i,j} psi_i psi_j S_{ij}.
        // Pairwise derivatives can be provided from either direction i->j or j->i.
        for i in 0..hyper_dirs.len() {
            if psi[i] == 0.0 {
                continue;
            }
            for j in 0..hyper_dirs.len() {
                if psi[j] == 0.0 {
                    continue;
                }
                let amp = 0.5 * psi[i] * psi[j];
                if amp == 0.0 {
                    continue;
                }
                if let Some(x_ij) = Self::get_pairwisesecond_derivative(hyper_dirs, i, j, true)
                    && let Some(x_mod) = x_mod_dense.as_mut()
                {
                    x_mod.scaled_add(amp, &x_ij);
                }
                let second_components =
                    Self::get_pairwisesecond_penalty_components(hyper_dirs, i, j);
                Self::add_penalty_components_to_state(&mut s_mod, &second_components, amp)?;
            }
        }
        let x_mod = x_mod_dense
            .map(DesignMatrix::from)
            .unwrap_or_else(|| self.x().clone());
        let warm_start_beta = self
            .warm_start_beta
            .read()
            .unwrap()
            .as_ref()
            .map(|beta| beta.as_ref().clone());
        let mut pert_state = RemlState::newwith_offset(
            self.y,
            x_mod,
            self.weights,
            self.offset.view(),
            s_mod,
            self.p,
            self.config,
            Some(self.nullspace_dims.clone()),
            self.coefficient_lower_bounds.clone(),
            self.linear_constraints.clone(),
        )?;
        pert_state.set_link_states(
            self.runtime_mixture_link_state.clone(),
            self.runtime_sas_link_state,
        );
        if let Some(beta) = warm_start_beta.as_ref() {
            pert_state.setwarm_start_original_beta(Some(beta.view()));
        }
        Ok(pert_state)
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
            // Rho-only path: no directional hyperparameters, use standard
            // rho-only evaluators directly.
            let cost = self.compute_cost(&rho)?;
            let grad = self.compute_gradient(&rho)?;
            let hess = self.compute_lamlhessian_exact(&rho)?;
            Ok((cost, grad, hess))
        }
    }

    /// Build the extended-coordinate objects for τ (directional) hyperparameters
    /// using the unified evaluator representation. This assembles both the
    /// first-order `HyperCoord` objects and the second-order pair callbacks,
    /// suitable for passing to `InnerSolutionBuilder::ext_coords()` and
    /// related pair-callback setters.
    ///
    /// Returns `(coords, ext_pair_fn, rho_ext_pair_fn)`.
    pub(crate) fn build_tau_unified_objects(
        &self,
        rho: &Array1<f64>,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<
        (
            Vec<super::unified::HyperCoord>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
            Box<dyn Fn(usize, usize) -> super::unified::HyperCoordPair + Send + Sync>,
        ),
        EstimationError,
    > {
        let bundle = self.obtain_eval_bundle(rho)?;
        let ext_coords = self.build_tau_hyper_coords(rho, &bundle, hyper_dirs)?;
        let (ext_pair_fn, rho_ext_pair_fn) =
            self.build_tau_pair_callbacks(rho, &bundle, hyper_dirs)?;
        Ok((ext_coords, ext_pair_fn, rho_ext_pair_fn))
    }

    pub(super) fn compute_directional_hypergradientwith_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        // Exact directional hyper-gradient for one moving hyperparameter τ with
        // both X(τ) and S(τ) varying, under the same piecewise-smooth
        // fixed-active-subspace conventions used for rho-derivatives.
        //
        // Objective-level identity (inner minimization convention):
        //   V_τ
        //   = ∂_τ L*(β̂,τ)
        //     + 0.5 tr(H_+^dagger H_τ)
        //     - 0.5 tr(S_+^dagger S_τ),
        // where L* = -ℓ + 0.5 β'Sβ - Φ and H = X'W_HX + S - H_φ on the Firth path.
        //
        // The envelope theorem removes explicit β̂_τ terms from ∂_τ L*(β̂,τ);
        // β̂_τ appears only through total curvature drift H_τ.
        //
        // This routine evaluates all currently implemented exact pieces:
        //   - fit block: -u'X_τβ̂ + 0.5 β̂'S_τβ̂ + Φ_τ|β (Firth),
        //   - log|H|_+ trace on positive spectral subspace when available,
        //   - log|S|_+ structural penalty trace.
        // Dense Firth branch includes design-moving terms in g_τ and fit_part,
        // and evaluates H_τ on the full chain:
        //   H_τ = H_std,τ - (H_φ)_τ|β - D(H_φ)[β_τ],
        // where (H_φ)_τ|β is assembled from reduced-space exact kernels.
        if rho.len() != self.s_full_list.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho dimension mismatch for psi gradient: expected {}, got {}",
                self.s_full_list.len(),
                rho.len()
            )));
        }
        if Self::geometry_backend_kind(bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_directional_hypergradient_sparse_exact(rho, bundle, hyper_dir);
        }
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_eff_eval, h_total_eval, beta_eval, e_eval);
        let x_dense_arc;
        let x_dense_owned;

        let x_psi_t = hyper_dir.transformed_x_tau(&reparam_result.qs, free_basis_opt.as_ref())?;

        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::projectwith_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::projectwith_basis(bundle.h_total.as_ref(), z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            let constrained_dense = pirls_result
                .x_transformed
                .try_to_dense_arc(
                    "directional hyper-gradient with active constraints requires dense transformed design",
                )
                .map_err(EstimationError::InvalidInput)?;
            x_dense_owned = Some(
                DenseRightProductView::new(constrained_dense.as_ref())
                    .with_factor(z)
                    .materialize(),
            );
            x_dense_arc = None;
            e_eval = reparam_result.e_transformed.dot(z);
        } else {
            h_eff_eval = bundle.h_eff.as_ref().clone();
            h_total_eval = bundle.h_total.as_ref().clone();
            beta_eval = pirls_result.beta_transformed.as_ref().clone();
            e_eval = reparam_result.e_transformed.clone();
            x_dense_arc = Some(
                pirls_result
                    .x_transformed
                    .try_to_dense_arc(
                        "exact directional hyper-gradient requires dense transformed design",
                    )
                    .map_err(EstimationError::InvalidInput)?,
            );
            x_dense_owned = None;
        }
        let x_dense = x_dense_owned.as_ref().unwrap_or_else(|| {
            x_dense_arc
                .as_ref()
                .expect("dense design arc missing")
                .as_ref()
        });
        let transformed_penalty_components = Self::transform_penalty_components(
            hyper_dir.penalty_first_components(),
            &reparam_result.qs,
            free_basis_opt.as_ref(),
        );
        let s_psi_t = transformed_penalty_components.iter().try_fold(
            Array2::<f64>::zeros((beta_eval.len(), beta_eval.len())),
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

        if x_psi_t.nrows() != self.y.len() || x_psi_t.ncols() != beta_eval.len() {
            return Err(EstimationError::InvalidInput(format!(
                "X_psi shape mismatch: expected {}x{}, got {}x{}",
                self.y.len(),
                beta_eval.len(),
                x_psi_t.nrows(),
                x_psi_t.ncols()
            )));
        }
        if s_psi_t.nrows() != beta_eval.len() || s_psi_t.ncols() != beta_eval.len() {
            return Err(EstimationError::InvalidInput(format!(
                "S_psi shape mismatch: expected {}x{}, got {}x{}",
                beta_eval.len(),
                beta_eval.len(),
                s_psi_t.nrows(),
                s_psi_t.ncols()
            )));
        }
        let s_psi_total_t = s_psi_t.clone();
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    x_dense,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };
        let h_solve_eval = if firth_logit_active {
            &h_total_eval
        } else {
            &h_eff_eval
        };
        // Objective-consistent generalized inverse for implicit solves:
        // when the retained positive eigenspace factor W is available in the
        // current coordinates (no free-basis projection), use
        //   H_+^dagger v = W (W' v)
        // for beta_tau and all stacked linear solves in this routine.
        //
        // This matches the same pseudo-logdet active subspace used in
        // 0.5*tr(H_+^dagger H_tau), avoiding inverse-mismatch between trace and IFT.
        let prefer_h_pos = !bundle.active_subspace_unstable;
        let h_posw_for_solve = if prefer_h_pos {
            if free_basis_opt.is_none() && bundle.h_pos_factorw.nrows() == h_solve_eval.nrows() {
                Some(bundle.h_pos_factorw.as_ref().clone())
            } else {
                None
            }
        } else {
            log::warn!(
                "Directional hyper-gradient using stabilized direct fallback for implicit solves (active subspace unstable)."
            );
            None
        };
        let h_posw_for_solve_t = h_posw_for_solve.as_ref().map(|w| w.t().to_owned());
        enum DirectionalSolveFactor {
            Cached(Arc<FaerFactor>),
            Local(FaerFactor),
        }
        impl DirectionalSolveFactor {
            fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
                match self {
                    DirectionalSolveFactor::Cached(f) => f.solve_in_place(rhs),
                    DirectionalSolveFactor::Local(f) => f.solve_in_place(rhs),
                }
            }
        }
        let h_solve_factor = if h_posw_for_solve.is_some() {
            None
        } else if free_basis_opt.is_none() && !firth_logit_active {
            Some(DirectionalSolveFactor::Cached(
                self.get_faer_factor(rho, h_solve_eval),
            ))
        } else {
            Some(DirectionalSolveFactor::Local(
                self.factorize_faer(h_solve_eval),
            ))
        };
        let solve_hvec = |rhs: &Array1<f64>| -> Array1<f64> {
            if rhs.is_empty() {
                return rhs.clone();
            }
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                // Minimum-norm active-subspace solve:
                //   beta_tau = H_+^dagger rhs = W (W' rhs),  H_+^dagger = W W'.
                let tmp = w_t.dot(rhs);
                return w.dot(&tmp);
            }
            let mut rhs_mat = Array2::<f64>::zeros((rhs.len(), 1));
            rhs_mat.column_mut(0).assign(rhs);
            let mut rhsview = array2_to_matmut(&mut rhs_mat);
            if let Some(f) = h_solve_factor.as_ref() {
                f.solve_in_place(rhsview.as_mut());
            }
            rhs_mat.column(0).to_owned()
        };
        let solve_h_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            if rhs.ncols() == 0 {
                return rhs.clone();
            }
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut outview = array2_to_matmut(&mut out);
            if let Some(f) = h_solve_factor.as_ref() {
                f.solve_in_place(outview.as_mut());
            }
            out
        };
        // Preferred exact trace evaluator for the log|H|_+ term:
        //   0.5 * tr(H_+^dagger A) = 0.5 * tr(W^T A W), H_+^dagger = W W^T.
        // Fall back to solve-surface contraction only when projected/active
        // coordinates are not aligned with the cached spectral basis.
        let h_posw = if prefer_h_pos {
            if free_basis_opt.is_none() && bundle.h_pos_factorw.nrows() == h_solve_eval.nrows() {
                Some(bundle.h_pos_factorw.as_ref().clone())
            } else {
                None
            }
        } else {
            None
        };
        let h_posw_t = h_posw.as_ref().map(|w| w.t().to_owned());
        let half_trace_h_pos = |a: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_posw.as_ref(), h_posw_t.as_ref()) {
                // Derivation:
                //   d/dτ [0.5 log|H|_+] = 0.5 tr(H_+^dagger H_τ),
                //   H_+^dagger = W W^T on the retained positive eigenspace.
                // Therefore:
                //   0.5 tr(H_+^dagger A) = 0.5 tr(W^T A W).
                let wt_a = fast_ab(w_t, a);
                let g = fast_ab(&wt_a, w);
                0.5 * g.diag().sum()
            } else {
                let h_solved = solve_h_mat(a);
                0.5 * h_solved.diag().sum()
            }
        };

        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        // `u` is the current predictor-space score contribution g pulled through
        // the PIRLS linearization, so X_τ^T g is represented by x_psi_t^T u.
        let xpsi_beta = x_psi_t.dot(&beta_eval);
        let weighted_xpsi_beta = &pirls_result.finalweights * &xpsi_beta;
        // Differentiate the stationarity equation
        //   0 = X^T g - S β̂
        // along the solution path β̂(τ):
        //
        //   0 = X_τ^T g + X^T(dg/dτ) - S_τ β̂ - S B,
        //   dg/dτ = -W η̇,
        //   η̇ = X_τ β̂ + X B.
        //
        // Rearranging gives the implicit solve
        //   H_solve B = X_τ^T g - X^T W(X_τ β̂) - S_τ β̂ - (gphi)_tau|beta,
        // where:
        //   - non-Firth: H_solve = X^T W X + S,
        //   - Firth logit: H_solve = H_total = X^T W X + S - Hphi.
        // This matches the stationarity surface used by the fitted objective.
        //
        // With Firth enabled this is:
        //   g_τ = (g_std)_τ - (gphi)_τ|_{beta fixed},
        // where (gphi)_τ comes from reduced Fisher pseudodeterminant calculus:
        //   Phi = 0.5 log|I_r|_+, I_r = X_r^T W X_r.
        // This RHS is exactly `g_psi`, and `solve_hvec(g_psi)` computes B
        // without ever forming H^{-1} explicitly.
        let mut g_psi = x_psi_t.t().dot(&u)
            - x_dense.t().dot(&weighted_xpsi_beta)
            - s_psi_total_t.dot(&beta_eval);
        let mut fit_firth_partial = 0.0_f64;
        let mut tau_kernel_opt: Option<FirthTauPartialKernel> = None;
        if let Some(op) = firth_op.as_ref() {
            // Exact Firth design-moving partial score correction (beta held fixed):
            //   g_tau = (g_std)_tau - (gphi)_tau,
            // where (gphi)_tau and Phi_tau|beta are computed from reduced Fisher
            // pseudodeterminant calculus under fixed active subspace.
            //
            // Implemented formulas:
            //   (gphi)_tau
            //   = 0.5 X_tau' (w' ⊙ h)
            //     + 0.5 X'((w'' ⊙ (X_tau β)) ⊙ h + w' ⊙ h_tau|beta),
            //   Phi_tau|beta = 0.5 tr(I_r^{-1} I_{r,tau}),
            // with I_r = X_r' W X_r and h_i = x_{r,i}' I_r^{-1} x_{r,i}.
            let need_tau_kernel = x_psi_t.iter().any(|v| *v != 0.0);
            let tau_bundle =
                Self::firth_exact_tau_kernel(op, &x_psi_t, &beta_eval, need_tau_kernel);
            g_psi -= &tau_bundle.gphi_tau;
            fit_firth_partial = tau_bundle.phi_tau_partial;
            if need_tau_kernel {
                tau_kernel_opt = Some(tau_bundle.tau_kernel.expect(
                    "exact directional assembly requires tau kernel when design drift is active",
                ));
            }
        }
        let beta_psi = solve_hvec(&g_psi);
        // Total predictor drift:
        //   η̇ = d/dτ(X β̂) = X_τ β̂ + X B.
        let eta_psi = &xpsi_beta + &x_dense.dot(&beta_psi);

        // Penalized-fit block:
        //   d/dτ[-ℓ(β̂,τ) + 0.5 β̂^T S β̂]
        //     = -ℓ_β^T B - ℓ_τ + β̂^T S B + 0.5 β̂^T S_τ β̂.
        //
        // Since stationarity gives ℓ_β = S β̂, the B terms cancel exactly:
        //   -ℓ_β^T B + β̂^T S B = 0,
        // leaving only the explicit τ-dependence
        //   -ℓ_τ + 0.5 β̂^T S_τ β̂ = -g^T X_τ β̂ + 0.5 β̂^T S_τ β̂.
        // Envelope fit block:
        //   d/dtau J(beta_hat,tau) = partial_tau J | beta_hat,
        // i.e. no explicit B term.
        // For Firth-logit:
        //   Phi_tau|beta = 0.5 tr(I_r^+ I_{r,tau}),
        // added via `fit_firth_partial`.
        let fit_block = -u.dot(&xpsi_beta)
            + 0.5 * beta_eval.dot(&s_psi_total_t.dot(&beta_eval))
            + fit_firth_partial;
        // Literal mapping to V_τ decomposition used in return values:
        //   V_τ = fit_block + trace_term + pseudo_det_term,
        // where
        //   fit_block      = ∂_τ[-ℓ + 0.5β^T S β - Φ]|_{β=β̂}  (envelope form),
        //   trace_term     = 0.5 tr(H_+^† H_τ^tot),
        //   pseudo_det_term= -0.5 tr(S_+^† S_τ).
        //
        // Non-Gaussian branch computes H_τ^tot explicitly via X_τ, W_τ, S_τ and
        // Firth drifts. Gaussian identity branch uses profiled dispersion algebra:
        //   D_{p,τ} = 2*fit_block and profiled_fit_term = D_{p,τ}/(2φ̂).

        let w = &pirls_result.finalweights;
        let mut weighted_xtdx = Array2::<f64>::zeros((0, 0));
        let mut h_psi = Self::weighted_cross(&x_psi_t, &x_dense, w);
        h_psi += &Self::weighted_cross(&x_dense, &x_psi_t, w);

        match self.config.link_function() {
            LinkFunction::Identity => {
                // Profiled Gaussian REML:
                //   V = D_p/(2φ̂) + 0.5 log|H| - 0.5 log|S|_+ + ((n-M_p)/2) log(2πφ̂),
                //   φ̂ = D_p/(n-M_p).
                //
                // Exact profiled derivative (holding M_p locally fixed):
                //   dV/dτ = D_{p,τ}/(2φ̂)
                //         + 0.5 tr(H^{-1} H_τ)
                //         - 0.5 tr(S^+ S_τ),
                // with
                //   H_τ = X_τ^T W X + X^T W X_τ + S_τ
                // (W_τ = 0 for Gaussian identity) and
                //   D_{p,τ} = 2 * fit_block
                // by stationarity cancellation in the fit block.
                h_psi += &s_psi_total_t;

                let n = self.y.len() as f64;
                let (penalty_rank, _) = self
                    .fixed_subspace_penalty_rank_and_logdet(&e_eval, pirls_result.ridge_passport)?;
                let p_eff_dim = h_eff_eval.ncols();
                let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;
                let dp = pirls_result.deviance + pirls_result.stable_penalty_term;
                let denom = (n - mp).max(LAML_RIDGE);
                let (dp_c, _) = smooth_floor_dp(dp);
                let phi = dp_c / denom;
                if !phi.is_finite() || phi <= 0.0 {
                    return Err(EstimationError::InvalidInput(
                        "invalid profiled Gaussian dispersion in directional hyper-gradient"
                            .to_string(),
                    ));
                }

                let trace_term = half_trace_h_pos(&h_psi);
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &e_eval,
                    &s_psi_total_t,
                    pirls_result.ridge_passport,
                )?;
                let pseudo_det_term = -0.5 * pseudo_det_trace;
                let dp_tau = 2.0 * fit_block;
                let profiled_fit_term = dp_tau / (2.0 * phi);
                Ok(profiled_fit_term + trace_term + pseudo_det_term)
            }
            _ => {
                let w_direction = crate::pirls::directionalworking_curvature_from_c_array(
                    &pirls_result.solve_c_array,
                    &pirls_result.finalweights,
                    &eta_psi,
                );
                match &w_direction {
                    // Directional curvature derivative:
                    //   W_τ = T[η̇].
                    //
                    // The family-dispatched PIRLS callback returns the operator form of
                    // this derivative. Built-in links are diagonal in observation space,
                    // so the operator is represented by the per-row vector `w_direction`.
                    DirectionalWorkingCurvature::Diagonal(w_direction) => {
                        h_psi +=
                            &Self::xt_diag_x_dense_into(&x_dense, w_direction, &mut weighted_xtdx);
                    }
                }
                // Exact total curvature derivative:
                //   H_τ = d/dτ(X^T W X + S)
                //       = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ.
                h_psi += &s_psi_total_t;
                if let Some(op) = firth_op.as_ref() {
                    // Firth-adjusted curvature surface:
                    //   H_total = H_std - Hphi.
                    // Exact total Firth contribution in this branch:
                    //   H_{phi,tau} = (Hphi)_tau|beta + D(Hphi)[beta_tau].
                    // Therefore:
                    //   H_total,tau = H_std,tau - H_{phi,tau}
                    //               = H_std,tau
                    //                 - (Hphi)_tau|beta
                    //                 - D(Hphi)[beta_tau].
                    //
                    // We apply both terms explicitly.
                    if x_psi_t.iter().any(|v| *v != 0.0) {
                        let kernel = tau_kernel_opt.as_ref().expect(
                            "exact directional assembly requires cached tau kernel for Hphi,tau",
                        );
                        let eye = Array2::<f64>::eye(beta_eval.len());
                        let hphi_tau_partial =
                            Self::firth_hphi_tau_partial_apply(op, &x_psi_t, kernel, &eye);
                        h_psi -= &hphi_tau_partial;
                    }
                    let firth_dir = Self::firth_direction(op, &beta_psi);
                    h_psi -= &Self::firth_hphi_direction(op, &firth_dir);
                }

                let trace_term = half_trace_h_pos(&h_psi);
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &e_eval,
                    &s_psi_total_t,
                    pirls_result.ridge_passport,
                )?;
                let pseudo_det_term = -0.5 * pseudo_det_trace;
                Ok(fit_block + trace_term + pseudo_det_term)
            }
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
    /// | `ld_s`  | `tr((S + δI)⁻¹ S_{τ_j})` |
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
        if psi_dim == 0 {
            return Ok(Vec::new());
        }

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
                    b_mat: Array2::zeros((0, 0)),
                    b_operator: None,
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
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    x_dense,
                    &pirls_result.final_eta,
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

            // Materialize the dense B_j matrix. When b_operator is active, we use
            // a zero-sized placeholder since the unified evaluator checks
            // b_operator.is_some() before accessing b_mat.
            let b_j = if b_operator.is_some() {
                Array2::<f64>::zeros((0, 0))
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

                b_j
            };

            // --- ld_s_j: smooth penalty pseudo-logdet derivative ---
            // ld_s_j = tr((S + δI)⁻¹ S_{τ_j}).
            // Uses the δ-regularized inverse (via fixed_subspace_penalty_trace).
            let ld_s_j =
                self.fixed_subspace_penalty_trace(&e_eval, &s_tau_j, pirls_result.ridge_passport)?;

            coords.push(super::unified::HyperCoord {
                a: a_j,
                g: g_j,
                b_mat: b_j,
                b_operator,
                ld_s: ld_s_j,
                b_depends_on_beta,
                is_penalty_like: hyper_dirs[j].is_penalty_like,
            });
        }

        Ok(coords)
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
    /// | `ld_s`  | `tr((S+δI)⁻¹ S_{τ_j} (S+δI)⁻¹ S_{τ_i}) − tr((S+δI)⁻¹ S_{τ_i τ_j})` |
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
                    .fold(Array2::<f64>::zeros((p_dim, p_dim)), |mut acc, c| {
                        c.matrix
                            .scaled_add_to(&mut acc, rho[c.penalty_index].exp())
                            .expect("valid penalty component in build_tau_pair_callbacks");
                        acc
                    })
            })
            .collect();

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
                        .fold(Array2::<f64>::zeros((p_dim, p_dim)), |mut acc, c| {
                            c.matrix
                                .scaled_add_to(&mut acc, rho[c.penalty_index].exp())
                                .expect("valid second penalty component");
                            acc
                        });
                s_tau_tau[i][j] = Some(total);
            }
        }

        // ── Build (S + δI)⁻¹ for ld_s pair computations ──
        //
        // Smooth δ-regularization (response.md Section 7): instead of computing
        // the pseudoinverse S⁺ with hard eigenvalue thresholding (which creates
        // non-smooth derivatives and requires explicit leakage/moving-nullspace
        // corrections), we use the full-rank regularized inverse (S + δI)⁻¹.
        //
        // This is C∞ in all outer parameters and automatically captures what
        // previously required:
        //   - Eigenspace partitioning (U₊/U₀ split)
        //   - Leakage matrices L_j = U₊ᵀ S_{τ_j} U₀
        //   - Moving-nullspace correction tr(Σ⁺² L_i L_jᵀ)
        //
        // All of these are now subsumed by the single (S + δI)⁻¹ computation.
        let rs_eval: Vec<Array2<f64>> = if let Some(z) = free_basis_opt.as_ref() {
            reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect()
        } else {
            reparam_result.rs_transformed.clone()
        };
        let s_eval = rs_eval
            .iter()
            .enumerate()
            .fold(Array2::<f64>::zeros((p_dim, p_dim)), |acc, (k, r)| {
                acc + r.t().dot(r).mapv(|v| lambdas[k] * v)
            });
        let (s_eigs, svecs) = s_eval
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Choose δ proportional to machine epsilon × spectral scale.
        let delta = super::unified::smooth_logdet_delta(s_eigs.as_slice().unwrap());

        // Build (S + δI)⁻¹ via the eigendecomposition.
        // Every eigenvalue is shifted by δ, so all are positive — no partitioning needed.
        let mut s_reg_inv = Array2::<f64>::zeros((p_dim, p_dim));
        for idx in 0..p_dim {
            let ev = s_eigs[idx];
            let inv_ev = 1.0 / (ev + delta);
            let ucol = svecs.column(idx).to_owned();
            let outer = ucol
                .view()
                .insert_axis(Axis(1))
                .dot(&ucol.view().insert_axis(Axis(0)));
            s_reg_inv += &outer.mapv(|v| v * inv_ev);
        }

        // Pre-compute (S + δI)⁻¹ S_{τ_j} products for the trace terms.
        let sdag_s_tau: Vec<Array2<f64>> = s_tau_list.iter().map(|s| s_reg_inv.dot(s)).collect();

        // Pre-compute A_k = λ_k R_k^T R_k for ρ-τ pairs.
        let a_k_mats: Vec<Array2<f64>> = rs_eval
            .iter()
            .enumerate()
            .map(|(k, r)| r.t().dot(r).mapv(|v| lambdas[k] * v))
            .collect();

        // Precompute (S + δI)⁻¹ A_k for ρ-τ pairs.
        let sdag_a_k: Vec<Array2<f64>> = a_k_mats.iter().map(|a| s_reg_inv.dot(a)).collect();

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

        // Pre-compute second-order design matrices X_{τ_i τ_j} for all pairs.
        let mut x_tau_tau: Vec<Vec<Option<Array2<f64>>>> = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in i..psi_dim {
                let xij = hyper_dirs[i]
                    .transformed_x_tau_tau_at(j, &reparam_result.qs, free_basis_opt.as_ref())
                    .ok()
                    .flatten()
                    .or_else(|| {
                        hyper_dirs[j]
                            .transformed_x_tau_tau_at(
                                i,
                                &reparam_result.qs,
                                free_basis_opt.as_ref(),
                            )
                            .ok()
                            .flatten()
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
        for j in 0..psi_dim {
            for component in &penalty_components_per_dir[j] {
                let k = component.penalty_index;
                if k < k_count {
                    let mat = component.matrix.scaled_materialize(lambdas[k]);
                    a_k_tau_j_mats[j][k] = Some(mat);
                }
            }
        }

        // Capture into Arc for shared ownership in closures.
        let s_tau_tau = Arc::new(s_tau_tau);
        let s_reg_inv = Arc::new(s_reg_inv);
        let sdag_s_tau = Arc::new(sdag_s_tau);
        let sdag_a_k = Arc::new(sdag_a_k);
        let beta_eval = Arc::new(beta_eval);
        let x_dense = Arc::new(x_dense);
        let x_tau_list = Arc::new(x_tau_list);
        let x_tau_beta_list = Arc::new(x_tau_beta_list);
        let x_tau_tau = Arc::new(x_tau_tau);
        let u = Arc::new(u);
        let w_diag = Arc::new(w_diag);
        let c_array = Arc::new(c_array);
        let d_array = Arc::new(d_array);
        let a_k_tau_j_mats = Arc::new(a_k_tau_j_mats);

        // ─── τ×τ pair callback ───────────────────────────────────────────
        let s_tau_tau_tt = Arc::clone(&s_tau_tau);
        let s_reg_inv_tt = Arc::clone(&s_reg_inv);
        let sdag_s_tau_tt = Arc::clone(&sdag_s_tau);
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
            // ld_s_{ij}: second derivative of L_δ(S) = log det(S + δI) − m₀ log δ
            // w.r.t. τ_i, τ_j.
            //
            // Using the smooth δ-regularized inverse (S + δI)⁻¹ instead of the
            // pseudoinverse S⁺:
            //
            //   ∂²_ij L_δ = tr((S+δI)⁻¹ S_j (S+δI)⁻¹ S_i) − tr((S+δI)⁻¹ S_ij)
            //
            // No leakage/moving-nullspace correction is needed because (S + δI)
            // is always full rank. The leakage correction that was previously
            // required (tr(Σ⁺² L_i L_jᵀ) terms) is automatically captured by
            // the full-rank inverse.
            //
            // Reference: response.md Section 7.
            let ld_s_quad = Self::trace_product(&sdag_s_tau_tt[j], &sdag_s_tau_tt[i]);
            let ld_s_linear = s_tau_tau_tt[i][j]
                .as_ref()
                .map(|s_ij| Self::trace_product(&s_reg_inv_tt, s_ij))
                .unwrap_or(0.0);
            let ld_s_ij = ld_s_quad - ld_s_linear;

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
            let a_ij_design2 = x_tau_tau_tt[i][j]
                .as_ref()
                .map(|xij| -u_tt.dot(&xij.dot(beta_tt.as_ref())))
                .unwrap_or(0.0);
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
            let term1 = x_tau_tau_tt[i][j]
                .as_ref()
                .map(|xij| xij.t().dot(u_tt.as_ref()))
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_tt));

            // term2: -X_{τ_j}^T diag(w)(X_{τ_i} β̂)
            let term2 = x_tau_j.t().dot(&(w_tt.as_ref() * x_tau_i_beta));

            // term3: -X_{τ_i}^T W(X_{τ_j} β̂)
            let term3 = x_tau_i.t().dot(&w_x_tau_j_beta);

            // term4: -X^T diag(c ⊙ X_{τ_i} β̂)(X_{τ_j} β̂)
            let c_x_tau_i_beta = c_tt.as_ref() * x_tau_i_beta;
            let term4 = x_dense_tt.t().dot(&(&c_x_tau_i_beta * x_tau_j_beta));

            // term5: -X^T W(X_{τ_i τ_j} β̂)
            let term5 = x_tau_tau_tt[i][j]
                .as_ref()
                .map(|xij| {
                    x_dense_tt
                        .t()
                        .dot(&(w_tt.as_ref() * &xij.dot(beta_tt.as_ref())))
                })
                .unwrap_or_else(|| Array1::<f64>::zeros(p_dim_tt));

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
                b_ij += &Self::weighted_cross(xij, x_dense_tt.as_ref(), w_tt.as_ref());
                b_ij += &Self::weighted_cross(x_dense_tt.as_ref(), xij, w_tt.as_ref());
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
                if let Some(xij) = x_tau_tau_tt[i][j].as_ref() {
                    let c_xij_beta = c_tt.as_ref() * &xij.dot(beta_tt.as_ref());
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
        let s_reg_inv_rt = Arc::clone(&s_reg_inv);
        let sdag_s_tau_rt = Arc::clone(&sdag_s_tau);
        let sdag_a_k_rt = Arc::clone(&sdag_a_k);
        let a_k_tau_j_rt = Arc::clone(&a_k_tau_j_mats);
        let beta_rt = Arc::clone(&beta_eval);
        let p_dim_rt = p_dim;

        let rho_tau_pair_fn = move |k: usize, j: usize| -> super::unified::HyperCoordPair {
            // ld_s_{k,τ_j}: second derivative of L_δ(S) w.r.t. ρ_k, τ_j.
            //
            // Using smooth δ-regularized inverse (S + δI)⁻¹:
            //   ∂²_{k,τ_j} L_δ = tr((S+δI)⁻¹ A_{k,τ_j}) − tr((S+δI)⁻¹ S_{τ_j} (S+δI)⁻¹ A_k)
            //
            // A_{k,τ_j} = λ_k * (dS_k/dτ_j) — the cross derivative.
            //
            // No leakage correction needed: (S + δI) is full rank.
            // Reference: response.md Section 7.
            //
            // Quadratic trace term:
            let ld_s_quad = Self::trace_product(&sdag_s_tau_rt[j], &sdag_a_k_rt[k]);
            // Linear trace term: tr((S + δI)⁻¹ A_{k,τ_j}).
            let ld_s_linear = a_k_tau_j_rt[j][k]
                .as_ref()
                .map(|a_kt| Self::trace_product(&s_reg_inv_rt, a_kt))
                .unwrap_or(0.0);
            let ld_s_kj = ld_s_linear - ld_s_quad;

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
                    b_mat: Array2::zeros((0, 0)),
                    b_operator: None,
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
                b_mat: b_j,
                b_operator: None,
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
                    b_mat: Array2::zeros((0, 0)),
                    b_operator: None,
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
                b_mat: b_j,
                b_operator: None,
                ld_s: 0.0,
                b_depends_on_beta: true,
                is_penalty_like: false,
            });
        }

        Ok(coords)
    }
}
