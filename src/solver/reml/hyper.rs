use super::*;

struct JointHyperThetaBlocks<'a> {
    theta: &'a Array1<f64>,
    rho_dim: usize,
}

impl<'a> JointHyperThetaBlocks<'a> {
    fn new(theta: &'a Array1<f64>, rho_dim: usize) -> Self {
        Self { theta, rho_dim }
    }

    fn rho_owned(&self) -> Array1<f64> {
        self.theta.slice(s![..self.rho_dim]).to_owned()
    }

    fn psi_owned(&self) -> Array1<f64> {
        self.theta.slice(s![self.rho_dim..]).to_owned()
    }

    fn assemblegradient(
        &self,
        rhograd: &Array1<f64>,
        psigrad: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if rhograd.len() != self.rho_dim
            || psigrad.len() != self.theta.len().saturating_sub(self.rho_dim)
        {
            return Err(EstimationError::InvalidInput(format!(
                "joint hyper gradient split mismatch: rhograd={}, psigrad={}, rho_dim={}, theta_dim={}",
                rhograd.len(),
                psigrad.len(),
                self.rho_dim,
                self.theta.len()
            )));
        }
        let mut out = Array1::<f64>::zeros(self.theta.len());
        out.slice_mut(s![..self.rho_dim]).assign(rhograd);
        out.slice_mut(s![self.rho_dim..]).assign(psigrad);
        Ok(out)
    }
}

impl<'a> RemlState<'a> {
    fn sum_penalty_components(
        p: usize,
        base: &[PenaltyDerivativeComponent],
        extra: Option<&[PenaltyDerivativeComponent]>,
        scale: f64,
    ) -> Result<Vec<(usize, Array2<f64>)>, EstimationError> {
        let mut out: Vec<(usize, Array2<f64>)> = base
            .iter()
            .map(|component| (component.penalty_index, component.matrix.clone()))
            .collect();
        if let Some(extra_components) = extra {
            for component in extra_components {
                if component.matrix.nrows() != p || component.matrix.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "penalty derivative component shape mismatch: expected {}x{}, got {}x{}",
                        p,
                        p,
                        component.matrix.nrows(),
                        component.matrix.ncols()
                    )));
                }
                if let Some((_, existing)) = out
                    .iter_mut()
                    .find(|(penalty_index, _)| *penalty_index == component.penalty_index)
                {
                    existing.scaled_add(scale, &component.matrix);
                } else {
                    out.push((
                        component.penalty_index,
                        component.matrix.mapv(|v| scale * v),
                    ));
                }
            }
        }
        Ok(out)
    }

    fn relinearized_penaltysecond_components(
        hyper_dirs: &[DirectionalHyperParam],
        j: usize,
        psi_dim: usize,
        p: usize,
    ) -> Result<Option<Vec<Vec<(usize, Array2<f64>)>>>, EstimationError> {
        let mut out = vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
        for target in 0..psi_dim {
            let extra = hyper_dirs[j]
                .penaltysecond_components_for(target)
                .or_else(|| {
                    hyper_dirs
                        .get(target)
                        .and_then(|dir| dir.penaltysecond_components_for(j))
                });
            out[target] = Self::sum_penalty_components(p, &[], extra, 1.0)?;
        }
        Ok(Some(out))
    }

    fn relinearized_hyper_dirs(
        hyper_dirs: &[DirectionalHyperParam],
        psi: &Array1<f64>,
        p: usize,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let psi_dim = hyper_dirs.len();
        let mut out = Vec::with_capacity(psi_dim);
        for j in 0..psi_dim {
            let mut x_j = hyper_dirs[j].x_tau_original.clone();
            let mut s_j = hyper_dirs[j]
                .penalty_first_components()
                .iter()
                .map(|component| (component.penalty_index, component.matrix.clone()))
                .collect::<Vec<_>>();
            for i in 0..psi_dim {
                let amp = psi[i];
                if amp == 0.0 {
                    continue;
                }
                if let Some(x_ji) = Self::get_pairwisesecond_derivative(hyper_dirs, j, i, true) {
                    x_j.scaled_add(amp, x_ji);
                }
                let extra = hyper_dirs[j].penaltysecond_components_for(i).or_else(|| {
                    hyper_dirs
                        .get(i)
                        .and_then(|dir| dir.penaltysecond_components_for(j))
                });
                s_j = Self::sum_penalty_components(p, &[], Some(extra.unwrap_or(&[])), amp)?
                    .into_iter()
                    .fold(s_j, |mut acc, (penalty_index, matrix)| {
                        if let Some((_, existing)) = acc
                            .iter_mut()
                            .find(|(existing_index, _)| *existing_index == penalty_index)
                        {
                            existing.scaled_add(1.0, &matrix);
                        } else {
                            acc.push((penalty_index, matrix));
                        }
                        acc
                    });
            }
            out.push(DirectionalHyperParam::new(
                x_j,
                s_j,
                hyper_dirs[j].x_tau_tau_original.clone(),
                Self::relinearized_penaltysecond_components(hyper_dirs, j, psi_dim, p)?,
            )?);
        }
        Ok(out)
    }

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
            s_mod[component.penalty_index].scaled_add(amp, &component.matrix);
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
            .map(|component| {
                let mut transformed = qs.t().dot(&component.matrix).dot(qs);
                if let Some(z) = free_basis_opt {
                    transformed = z.t().dot(&transformed).dot(z);
                }
                PenaltyDerivativeComponent {
                    penalty_index: component.penalty_index,
                    matrix: transformed,
                }
            })
            .collect()
    }

    fn get_pairwisesecond_derivative<'b>(
        hyper_dirs: &'b [DirectionalHyperParam],
        i: usize,
        j: usize,
        x_term: bool,
    ) -> Option<&'b Array2<f64>> {
        let from_i = if x_term {
            hyper_dirs
                .get(i)
                .and_then(|d| d.x_tau_tau_original.as_ref())
                .and_then(|v| v.get(j))
        } else {
            None
        };
        if from_i.is_some() {
            return from_i;
        }
        if x_term {
            hyper_dirs
                .get(j)
                .and_then(|d| d.x_tau_tau_original.as_ref())
                .and_then(|v| v.get(i))
        } else {
            None
        }
    }

    pub(super) fn validate_joint_hyper_inputs(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<usize, EstimationError> {
        if rho_dim > theta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            )));
        }
        let psi_dim = theta.len() - rho_dim;
        if hyper_dirs.len() != psi_dim {
            return Err(EstimationError::InvalidInput(format!(
                "joint hyper-gradient derivative count mismatch: psi_dim={}, hyper_dirs={}",
                psi_dim,
                hyper_dirs.len(),
            )));
        }
        for (j, dir) in hyper_dirs.iter().enumerate() {
            for component in dir.penalty_first_components() {
                if component.penalty_index >= self.s_full_list.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "penalty_index {} out of bounds for {} penalties (dir {})",
                        component.penalty_index,
                        self.s_full_list.len(),
                        j
                    )));
                }
            }
            Self::validate_penalty_component_shapes(
                dir.penalty_first_components(),
                self.p,
                &format!("S_tau[{j}]"),
            )?;
            if let Some(x2) = dir.x_tau_tau_original.as_ref() {
                if x2.len() != psi_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "X_tau_tau[{j}] length mismatch: expected {}, got {}",
                        psi_dim,
                        x2.len()
                    )));
                }
                for (i, x_ij) in x2.iter().enumerate() {
                    if x_ij.nrows() != self.y.len() || x_ij.ncols() != self.p {
                        return Err(EstimationError::InvalidInput(format!(
                            "X_tau_tau[{j}][{i}] shape mismatch: expected {}x{}, got {}x{}",
                            self.y.len(),
                            self.p,
                            x_ij.nrows(),
                            x_ij.ncols()
                        )));
                    }
                }
            }
            if let Some(s2) = dir.penaltysecond_components.as_ref() {
                if s2.len() != psi_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "S_tau_tau[{j}] length mismatch: expected {}, got {}",
                        psi_dim,
                        s2.len()
                    )));
                }
                for (i, components) in s2.iter().enumerate() {
                    for component in components {
                        if component.penalty_index >= self.s_full_list.len() {
                            return Err(EstimationError::InvalidInput(format!(
                                "penalty_index {} out of bounds for {} penalties (S_tau_tau[{j}][{i}])",
                                component.penalty_index,
                                self.s_full_list.len(),
                            )));
                        }
                    }
                    Self::validate_penalty_component_shapes(
                        components,
                        self.p,
                        &format!("S_tau_tau[{j}][{i}]"),
                    )?;
                }
            }
        }
        Ok(psi_dim)
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
            if dir.x_tau_original.iter().any(|v| *v != 0.0) {
                return true;
            }
            if let Some(x2) = dir.x_tau_tau_original.as_ref() {
                return x2.iter().any(|m| m.iter().any(|v| *v != 0.0));
            }
            false
        });
        let mut x_mod_dense = if has_design_drift {
            Some(self.x().to_dense_arc().as_ref().clone())
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
                && dir.x_tau_original.raw_dim() != x_mod.raw_dim()
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
                x_mod.scaled_add(amp, &dir.x_tau_original);
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
                    x_mod.scaled_add(amp, x_ij);
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

    pub(super) fn computemulti_psigradientwith_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array1<f64>, EstimationError> {
        let mut psigrad = Array1::<f64>::zeros(hyper_dirs.len());
        for (i, hyper_dir) in hyper_dirs.iter().enumerate() {
            psigrad[i] =
                self.compute_directional_hypergradientwith_bundle(rho, bundle, hyper_dir)?;
        }
        Ok(psigrad)
    }

    pub(crate) fn compute_joint_hypercostgradient(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        let psi_dim = self.validate_joint_hyper_inputs(theta, rho_dim, hyper_dirs)?;
        let blocks = JointHyperThetaBlocks::new(theta, rho_dim);
        let rho = blocks.rho_owned();
        let psi = blocks.psi_owned();
        let eff_dirs = Self::relinearized_hyper_dirs(hyper_dirs, &psi, self.p)?;
        let (cost, rhograd, psigrad) = if psi_dim > 0 {
            let pert_state = self.build_joint_perturbed_state(&psi, hyper_dirs)?;
            let bundle = pert_state.obtain_eval_bundle(&rho)?;
            (
                pert_state.compute_cost(&rho)?,
                pert_state.compute_gradient_with_bundle(&rho, &bundle)?,
                pert_state.computemulti_psigradientwith_bundle(&rho, &bundle, &eff_dirs)?,
            )
        } else {
            let bundle = self.obtain_eval_bundle(&rho)?;
            (
                self.compute_cost(&rho)?,
                self.compute_gradient_with_bundle(&rho, &bundle)?,
                Array1::<f64>::zeros(0),
            )
        };
        let out = blocks.assemblegradient(&rhograd, &psigrad)?;
        Ok((cost, out))
    }

    pub(super) fn compute_mixed_rho_tau_block(
        &self,
        pert_state: &RemlState<'a>,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array2<f64>, EstimationError> {
        let k_count = rho.len();
        let psi_dim = hyper_dirs.len();
        let mut mixed = Array2::<f64>::zeros((k_count, psi_dim));
        if k_count == 0 || psi_dim == 0 {
            return Ok(mixed);
        }
        // Analytic mixed block assembly for each directional hyperparameter tau_j:
        //
        // Objective (profiled, fixed-active-subspace regime):
        //   V(ρ,ψ)
        //   = L*(β̂(ρ,ψ),ρ,ψ)
        //     + 0.5 log|H(β̂,ρ,ψ)|_+
        //     - 0.5 log|S(ρ,ψ)|_+.
        //
        // Mixed directional identity (u=ρ_k, v=τ_j):
        //   V_uv
        //   = (L*_{uv} - g_u^T H_+^† gv)
        //     + 0.5[ tr(H_+^† H_uv^tot) - tr(H_+^† H_u^tot H_+^† Hv^tot) ]
        //     - 0.5[ tr(S_+^† S_uv) - tr(S_+^† S_u S_+^† Sv) ].
        //
        // In this block:
        //   - u=ρ_k has X_u=0, S_u=A_k=λ_k S_k.
        //   - v=τ_j may move both X and S: Xv=X_{τ_j}, Sv=S_{τ_j}.
        //   - Firth/logit enters through gv and Hv^tot via
        //       (g_φ)v = Φ_{βv},
        //       (H_φ)v|β = Φ_{ββv},
        //       D(H_φ)[βv] = Φ_{βββ}[βv].
        //
        // Implemented equivalent decomposition:
        //   V_{k,tau}
        //   = beta^T A_k beta_tau + 0.5 beta^T A_{k,tau} beta
        //     + 0.5[ tr(H^{-1} H_{k,tau}) - tr(H^{-1} H_tau H^{-1} H_k) ]
        //     - 0.5[ tr(S^+ A_{k,tau}) - tr(S^+ S_tau S^+ A_k) ].
        //
        // with coupled IFT solves
        //   H beta_tau  = g_tau,
        //   H B_k       = -A_k beta,
        //   H B_{k,tau} = -(H_tau B_k + A_k beta_tau + A_{k,tau} beta).
        //
        // This removes the previous FD-in-(tau amplitude) mixed block path.
        for j in 0..psi_dim {
            let col = pert_state.compute_mixed_rho_tau_column_analyticwith_bundle(
                rho,
                bundle,
                &hyper_dirs[j],
            )?;
            mixed.column_mut(j).assign(&col);
        }
        Ok(mixed)
    }

    pub(super) fn compute_mixed_rho_tau_column_analyticwith_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<Array1<f64>, EstimationError> {
        let k_count = rho.len();
        let mut out = Array1::<f64>::zeros(k_count);
        if k_count == 0 {
            return Ok(out);
        }
        // Literal derivation map for one mixed column V_{ρ_k,τ}
        // Let τ denote the current directional hyperparameter (hyper_dir).
        //
        // Profiled objective:
        //   V = L*(β̂,θ) + 0.5 log|H|_+ - 0.5 log|S|_+.
        //
        // Mixed block identity:
        //   V_{k,τ}
        //   = Q_{k,τ} + L_{k,τ} + P_{k,τ},
        // where
        //   Q_{k,τ} = β^T A_k β_τ + 0.5 β^T A_{k,τ} β
        //   L_{k,τ} = 0.5[ tr(H_+^† H_{k,τ}) - tr(H_+^† H_τ H_+^† H_k) ]
        //   P_{k,τ} = -0.5[ tr(S_+^† A_{k,τ}) - tr(S_+^† S_τ S_+^† A_k) ].
        //
        // Penalty derivatives in this implementation:
        //   A_k     = dS/dρ_k      = λ_k R_k^T R_k
        //   A_{k,τ} = d²S/(dρ_k dτ)= λ_k S_τ.
        //
        // Coupled IFT solves:
        //   H B_k       = -A_k β
        //   H β_τ       = g_τ
        //   H B_{k,τ}   = -(H_τ B_k + A_k β_τ + A_{k,τ} β).
        //
        // Firth/logit corrections are objective-consistent:
        //   g_τ      <- g_τ      - (g_φ)_τ|β
        //   H_τ      <- H_τ      - (H_φ)_τ|β - D(H_φ)[β_τ]
        //   H_{k,τ}  <- H_{k,τ}  - D(H_φ)[B_{k,τ}] - D²(H_φ)[B_k,β_τ].
        //
        // Variable mapping below:
        //   B_k        -> b_k_mat.column(k)
        //   β_τ        -> beta_tau
        //   B_{k,τ}    -> b_k_tau
        //   H_k        -> h_k[k]
        //   H_τ        -> h_tau
        //   H_{k,τ}    -> h_k_tau (plus explicit Firth trace corrections)
        //   Q_{k,τ}    -> q_mixed
        //   L_{k,τ}    -> 0.5*(t_linear - t_quad)
        //   P_{k,τ}    -> p_mixed
        //
        // n-th order continuation:
        //   For additional hyper directions α_1,...,α_m, continue by repeatedly
        //   differentiating G(β̂,θ)=0 and solving
        //      H β̂_{I} = -R_I
        //   for each multi-index I=(α_1,...,α_m), where R_I is assembled from
        //   partition sums of mixed derivatives of G with lower-order β̂ terms.
        //   Then apply the same partition calculus to f, log|H|_+, and log|S|_+.

        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);

        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut rs_eval = reparam_result.rs_transformed.clone();
        let mut x_eval = pirls_result
            .x_transformed
            .to_dense_arc()
            .as_ref()
            .to_owned();
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut x_tau_t = hyper_dir.x_tau_original.dot(&reparam_result.qs);

        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
            x_eval = x_eval.dot(z);
            h_eff_eval = Self::projectwith_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::projectwith_basis(bundle.h_total.as_ref(), z);
            x_tau_t = x_tau_t.dot(z);
        }
        let p_dim = beta_eval.len();
        if p_dim == 0 {
            return Ok(out);
        }
        let penalty_first_components_t = Self::transform_penalty_components(
            hyper_dir.penalty_first_components(),
            &reparam_result.qs,
            free_basis_opt.as_ref(),
        );
        let s_tau_total_t = penalty_first_components_t.iter().try_fold(
            Array2::<f64>::zeros((p_dim, p_dim)),
            |mut acc, component| {
                if component.penalty_index >= rho.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "penalty_index {} out of bounds for rho dimension {}",
                        component.penalty_index,
                        rho.len()
                    )));
                }
                acc.scaled_add(rho[component.penalty_index].exp(), &component.matrix);
                Ok(acc)
            },
        )?;
        if x_eval.ncols() != p_dim || x_tau_t.ncols() != p_dim {
            return Err(EstimationError::InvalidInput(format!(
                "mixed rho-tau analytic shape mismatch: X={}x{}, X_tau={}x{}, p={}",
                x_eval.nrows(),
                x_eval.ncols(),
                x_tau_t.nrows(),
                x_tau_t.ncols(),
                p_dim
            )));
        }

        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let h_solve_eval = if firth_logit_active {
            &h_total_eval
        } else {
            &h_eff_eval
        };
        // Smart solve backend:
        // 1) Prefer objective-consistent positive-subspace solve H_+^dagger = W W^T.
        // 2) Fall back to stabilized direct factor solve only when active-subspace diagnostics
        //    are unstable or H_+ factor construction fails in current coordinates.
        // Near a hard threshold crossing we deliberately stop reusing the
        // positive-part spectral inverse for IFT solves: that inverse matches
        // the branch-local truncated-logdet surface, but the branch itself is
        // no longer stable enough to treat as exact.
        let prefer_h_pos = !bundle.active_subspace_unstable;
        let h_posw_for_solve = if prefer_h_pos {
            if free_basis_opt.is_none() && bundle.h_pos_factorw.nrows() == p_dim {
                Some(bundle.h_pos_factorw.as_ref().clone())
            } else {
                match Self::positive_part_factorw(h_solve_eval) {
                    Ok(w) => Some(w),
                    Err(err) => {
                        log::warn!(
                            "Mixed rho-tau analytic solve using stabilized direct fallback (failed H_+ projector build: {}).",
                            err
                        );
                        None
                    }
                }
            }
        } else {
            log::warn!(
                "Mixed rho-tau analytic solve using stabilized direct fallback (active subspace unstable)."
            );
            None
        };
        let h_posw_for_solve_t = h_posw_for_solve.as_ref().map(|w| w.t().to_owned());
        let h_factor = if h_posw_for_solve.is_none() {
            Some(self.factorize_faer(h_solve_eval))
        } else {
            None
        };
        let solve_hvec = |rhs: &Array1<f64>| -> Array1<f64> {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let tmp = w_t.dot(rhs);
                return w.dot(&tmp);
            }
            let mut rhs_mat = Array2::<f64>::zeros((rhs.len(), 1));
            rhs_mat.column_mut(0).assign(rhs);
            let mut rhsview = array2_to_matmut(&mut rhs_mat);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(rhsview.as_mut());
            }
            rhs_mat.column(0).to_owned()
        };
        let solve_h_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut outview = array2_to_matmut(&mut out);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(outview.as_mut());
            }
            out
        };
        let trace_hdag = |a: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let wt_a = fast_ab(w_t, a);
                let g = fast_ab(&wt_a, w);
                g.diag().sum()
            } else {
                let solved = solve_h_mat(a);
                solved.diag().sum()
            }
        };
        // Matrix-free trace contraction for D²(Hphi)[u,v]:
        //   tr(H_+^dagger D²Hphi[u,v]) = tr(W^T (D²Hphi[u,v] W))
        // when the positive-subspace solve projector W is available.
        // Falls back to dense assembly only when W is unavailable.
        let trace_hdag_firthsecond =
            |op: &FirthDenseOperator, u_dir: &FirthDirection, v_dir: &FirthDirection| -> f64 {
                Self::trace_hdag_operator_apply(
                    p_dim,
                    h_posw_for_solve.as_ref(),
                    h_posw_for_solve_t.as_ref(),
                    |a| solve_h_mat(a),
                    |basis| Self::firth_hphisecond_direction_apply(op, u_dir, v_dir, basis),
                )
            };
        let trace_hdag_firth_first = |op: &FirthDenseOperator, u_dir: &FirthDirection| -> f64 {
            Self::trace_hdag_operator_apply(
                p_dim,
                h_posw_for_solve.as_ref(),
                h_posw_for_solve_t.as_ref(),
                |a| solve_h_mat(a),
                |basis| Self::firth_hphi_direction_apply(op, u_dir, basis),
            )
        };
        let trace_hdag_b_hdag_c = |b: &Array2<f64>, c_mat: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let gb = fast_ab(&fast_ab(w_t, b), w);
                let gc = fast_ab(&fast_ab(w_t, c_mat), w);
                Self::trace_product(&gb, &gc)
            } else {
                let left = solve_h_mat(b);
                let right = solve_h_mat(c_mat);
                Self::trace_product(&left, &right)
            }
        };

        let lambdas = rho.mapv(f64::exp);
        let mut a_k_mats = Vec::<Array2<f64>>::with_capacity(k_count);
        let mut a_k_beta = Vec::<Array1<f64>>::with_capacity(k_count);
        let mut rhs_bk = Array2::<f64>::zeros((p_dim, k_count));
        for k in 0..k_count {
            let r_k = &rs_eval[k];
            let s_k = r_k.t().dot(r_k);
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            let a_kb = a_k.dot(&beta_eval);
            rhs_bk.column_mut(k).assign(&a_kb.mapv(|v| -v));
            a_k_mats.push(a_k);
            a_k_beta.push(a_kb);
        }
        let b_k_mat = solve_h_mat(&rhs_bk);
        let u_k_mat = x_eval.dot(&b_k_mat);

        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let x_tau_beta = x_tau_t.dot(&beta_eval);
        let weighted_x_tau_beta = &pirls_result.solveweights * &x_tau_beta;
        let mut g_tau = x_tau_t.t().dot(&u)
            - x_eval.t().dot(&weighted_x_tau_beta)
            - s_tau_total_t.dot(&beta_eval);

        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    &x_eval,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };
        let mut tau_kernel_opt: Option<FirthTauPartialKernel> = None;
        if let Some(op) = firth_op.as_ref() {
            let need_tau_kernel = x_tau_t.iter().any(|v| *v != 0.0);
            let tau_bundle =
                Self::firth_exact_tau_kernel(op, &x_tau_t, &beta_eval, need_tau_kernel);
            g_tau -= &tau_bundle.gphi_tau;
            if need_tau_kernel {
                tau_kernel_opt = Some(tau_bundle.tau_kernel.expect(
                    "exact directional assembly requires tau kernel when design drift is active",
                ));
            }
        }
        let beta_tau = solve_hvec(&g_tau);
        let eta_tau = &x_tau_beta + &x_eval.dot(&beta_tau);

        let runtime_link = self.runtime_inverse_link();
        let w_tau = match crate::pirls::directionalworking_curvature_from_etawith_state(
            &runtime_link,
            &pirls_result.final_eta,
            self.weights,
            &pirls_result.solveweights,
            &eta_tau,
        )? {
            DirectionalWorkingCurvature::Diagonal(diag) => diag,
        };

        let c = &pirls_result.solve_c_array;
        let d = &pirls_result.solve_d_array;
        let c_tau = d * &eta_tau;
        let mut weighted = Array2::<f64>::zeros(x_eval.raw_dim());
        let wx = Self::row_scale(&x_eval, &pirls_result.solveweights);
        let wx_tau = Self::row_scale(&x_tau_t, &pirls_result.solveweights);
        let mut h_tau = x_tau_t.t().dot(&wx)
            + x_eval.t().dot(&wx_tau)
            + Self::xt_diag_x_dense_into(&x_eval, &w_tau, &mut weighted)
            + &s_tau_total_t;
        if let Some(op) = firth_op.as_ref() {
            if x_tau_t.iter().any(|v| *v != 0.0) {
                let kernel = tau_kernel_opt
                    .as_ref()
                    .expect("exact directional assembly requires cached tau kernel for Hphi,tau");
                let eye = Array2::<f64>::eye(p_dim);
                let hphi_tau_partial =
                    Self::firth_hphi_tau_partial_apply(op, &x_tau_t, kernel, &eye);
                h_tau -= &hphi_tau_partial;
            }
            let dir_tau = Self::firth_direction(op, &beta_tau);
            h_tau -= &Self::firth_hphi_direction(op, &dir_tau);
        }

        let mut h_k = Vec::<Array2<f64>>::with_capacity(k_count);
        let firth_dirs = firth_op.as_ref().map(|op| {
            (0..k_count)
                .map(|k| Self::firth_direction(op, &b_k_mat.column(k).to_owned()))
                .collect::<Vec<_>>()
        });
        for k in 0..k_count {
            let mut hk = a_k_mats[k].clone();
            let diag_like_k = c * &u_k_mat.column(k).to_owned();
            hk += &Self::xt_diag_x_dense_into(&x_eval, &diag_like_k, &mut weighted);
            if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                hk -= &Self::firth_hphi_direction(op, &dirs[k]);
            }
            h_k.push(hk);
        }

        // Structural penalty pseudoinverse for mixed -0.5 log|S|_+ term.
        let s_eval = rs_eval
            .iter()
            .enumerate()
            .fold(Array2::<f64>::zeros((p_dim, p_dim)), |acc, (k, r)| {
                acc + r.t().dot(r).mapv(|v| lambdas[k] * v)
            });
        let (s_eigs, svecs) = s_eval
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let s_max = s_eigs
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            .max(1.0);
        let s_tol = (p_dim.max(1) as f64) * f64::EPSILON * s_max;
        let mut s_dag = Array2::<f64>::zeros((p_dim, p_dim));
        for i in 0..p_dim {
            let ev = s_eigs[i];
            if ev > s_tol {
                let ucol = svecs.column(i).to_owned();
                let scale = 1.0 / ev;
                let outer = ucol
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&ucol.view().insert_axis(Axis(0)));
                s_dag += &outer.mapv(|v| scale * v);
            }
        }

        for k in 0..k_count {
            let b_k = b_k_mat.column(k).to_owned();
            let a_k = &a_k_mats[k];
            let a_k_tau = penalty_first_components_t
                .iter()
                .find(|component| component.penalty_index == k)
                .map(|component| component.matrix.mapv(|v| lambdas[k] * v))
                .unwrap_or_else(|| Array2::<f64>::zeros((p_dim, p_dim)));
            // B_{k,τ} solve:
            //   H B_{k,τ} = -(H_τ B_k + A_k β_τ + A_{k,τ} β).
            let rhs_bktau = -(&h_tau.dot(&b_k) + &a_k.dot(&beta_tau) + &a_k_tau.dot(&beta_eval));
            let b_k_tau = solve_hvec(&rhs_bktau);

            let u_k = u_k_mat.column(k).to_owned();
            let u_k_tau = x_eval.dot(&b_k_tau);
            let x_tau_bk = x_tau_t.dot(&b_k);
            let diag_q = c * &u_k;
            // d/dτ [c ⊙ u_k] = c_τ ⊙ u_k + c ⊙ (X_τ B_k) + c ⊙ (X B_{k,τ}).
            let diag_q_tau = &(&c_tau * &u_k) + &(c * &x_tau_bk) + &(c * &u_k_tau);
            let mut h_k_tau = a_k_tau.clone();
            h_k_tau += &x_tau_t.t().dot(&Self::row_scale(&x_eval, &diag_q));
            h_k_tau += &x_eval.t().dot(&Self::row_scale(&x_tau_t, &diag_q));
            h_k_tau += &Self::xt_diag_x_dense_into(&x_eval, &diag_q_tau, &mut weighted);
            let mut d1_trace_correction = 0.0_f64;
            let mut d2_trace_correction = 0.0_f64;
            if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                let dir_ktau = Self::firth_direction(op, &b_k_tau);
                let dir_tau = Self::firth_direction(op, &beta_tau);
                d1_trace_correction = trace_hdag_firth_first(op, &dir_ktau);
                d2_trace_correction = trace_hdag_firthsecond(op, &dirs[k], &dir_tau);
            }

            // Q_{k,τ} from profiled fit block.
            let q_mixed =
                beta_eval.dot(&a_k.dot(&beta_tau)) + 0.5 * beta_eval.dot(&a_k_tau.dot(&beta_eval));
            // L_{k,τ} first part: tr(H_+^† H_{k,τ}) with explicit Firth removals.
            let t_linear = trace_hdag(&h_k_tau) - d1_trace_correction - d2_trace_correction;
            // L_{k,τ} second part: tr(H_+^† H_τ H_+^† H_k).
            let t_quad = trace_hdag_b_hdag_c(&h_tau, &h_k[k]);
            // P_{k,τ} exact pseudodet block.
            let p_mixed = -0.5
                * (Self::trace_product(&s_dag, &a_k_tau)
                    - Self::trace_product(&s_dag.dot(&s_tau_total_t).dot(&s_dag), a_k));
            out[k] = q_mixed + 0.5 * (t_linear - t_quad) + p_mixed;
        }

        Ok(out)
    }

    pub(super) fn compute_tau_tau_block(
        &self,
        pert_state: &RemlState<'a>,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array2<f64>, EstimationError> {
        let psi_dim = hyper_dirs.len();
        let mut h_tt = Array2::<f64>::zeros((psi_dim, psi_dim));
        if psi_dim == 0 {
            return Ok(h_tt);
        }
        // τ-τ block for moving-design/moving-penalty hyperparameters:
        //
        // For u=τ_i, v=τ_j with stationarity g(β̂,θ)=0:
        //   V_uv
        //   = (L*_{uv} - g_u^T H_+^† gv)
        //     + 0.5[ tr(H_+^† H_uv^tot) - tr(H_+^† H_u^tot H_+^† Hv^tot) ]
        //     - 0.5[ tr(S_+^† S_uv) - tr(S_+^† S_u S_+^† Sv) ].
        //
        // Key IFT solves used below:
        //   H β_u = -g_u,   H βv = -gv,
        //   H β_uv = -(g_uv + H_u βv + Hv β_u + D_βH[β_u,βv]).
        //
        // Here H = X'WX + S - H_φ (Firth path), so H_u/H_uv include:
        //   - moving X terms (X_u, X_uv),
        //   - moving W terms through η_u = X_u β + X β_u and η_uv,
        //   - moving penalty terms (S_u, S_uv),
        //   - Firth drifts: -(H_φ)_u|β, -(H_φ)_uv|β, -D(H_φ)[β_u], -D(H_φ)[β_uv].
        //
        // All traces use the same positive-subspace generalized inverse as the
        // pseudo-logdet terms (H_+^† and S_+^†) to keep objective consistency.
        //
        // n-th order continuation:
        //   This τ-τ assembly is the m=2 instance of the general recursion
        //   for D^mV[τ_1,...,τ_m]. For m>2, use:
        //   - implicit partition recursion for β̂_{...},
        //   - total-derivative partition expansion for H(β̂(θ),θ),
        //   - partition logdet formula for D^m log|H|_+ and D^m log|S|_+.
        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;
        let free_basis_opt = pert_state.active_constraint_free_basis(pirls_result);

        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut x_eval = pirls_result
            .x_transformed
            .to_dense_arc()
            .as_ref()
            .to_owned();
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut x_tau_t = Vec::<Array2<f64>>::with_capacity(psi_dim);
        for dir in hyper_dirs {
            x_tau_t.push(dir.x_tau_original.dot(&reparam_result.qs));
        }
        let mut x_tau_tau_t = vec![vec![None; psi_dim]; psi_dim];
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                if let Some(x_ij) = Self::get_pairwisesecond_derivative(hyper_dirs, i, j, true) {
                    x_tau_tau_t[i][j] = Some(x_ij.dot(&reparam_result.qs));
                }
            }
        }
        if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            x_eval = x_eval.dot(z);
            h_eff_eval = Self::projectwith_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::projectwith_basis(bundle.h_total.as_ref(), z);
            for j in 0..psi_dim {
                x_tau_t[j] = x_tau_t[j].dot(z);
            }
            for i in 0..psi_dim {
                for j in 0..psi_dim {
                    if let Some(x_ij) = x_tau_tau_t[i][j].as_mut() {
                        *x_ij = x_ij.dot(z);
                    }
                }
            }
        }

        let p_dim = beta_eval.len();
        if p_dim == 0 {
            return Ok(h_tt);
        }
        let firth_logit_active = pert_state.config.firth_bias_reduction
            && matches!(pert_state.config.link_function(), LinkFunction::Logit);
        let h_solve_eval = if firth_logit_active {
            &h_total_eval
        } else {
            &h_eff_eval
        };
        // Single-inverse doctrine for analytic tau-tau:
        // use the same positive-subspace generalized inverse used by log|H|_+.
        // This keeps all IFT solves and trace contractions on one objective-consistent surface.
        // Same policy as above: use H_+^dagger only while the retained
        // eigenspace is stably separated from the dropped spectrum.
        let prefer_h_pos = !bundle.active_subspace_unstable;
        let h_posw_for_solve = if prefer_h_pos {
            if free_basis_opt.is_none() && bundle.h_pos_factorw.nrows() == p_dim {
                Some(bundle.h_pos_factorw.as_ref().clone())
            } else {
                match Self::positive_part_factorw(h_solve_eval) {
                    Ok(w) => Some(w),
                    Err(err) => {
                        log::warn!(
                            "Tau-tau analytic solve using stabilized direct fallback (failed H_+ projector build: {}).",
                            err
                        );
                        None
                    }
                }
            }
        } else {
            log::warn!(
                "Tau-tau analytic solve using stabilized direct fallback (active subspace unstable)."
            );
            None
        };
        let h_posw_for_solve_t = h_posw_for_solve.as_ref().map(|w| w.t().to_owned());
        let h_factor = if h_posw_for_solve.is_none() {
            Some(pert_state.factorize_faer(h_solve_eval))
        } else {
            None
        };
        let solve_hvec = |rhs: &Array1<f64>| -> Array1<f64> {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let tmp = w_t.dot(rhs);
                return w.dot(&tmp);
            }
            let mut rhs_mat = Array2::<f64>::zeros((rhs.len(), 1));
            rhs_mat.column_mut(0).assign(rhs);
            let mut rhsview = array2_to_matmut(&mut rhs_mat);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(rhsview.as_mut());
            }
            rhs_mat.column(0).to_owned()
        };
        let solve_h_mat = |rhs: &Array2<f64>| -> Array2<f64> {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut outview = array2_to_matmut(&mut out);
            if let Some(f) = h_factor.as_ref() {
                f.solve_in_place(outview.as_mut());
            }
            out
        };
        let trace_hdag = |a: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let wt_a = fast_ab(w_t, a);
                let g = fast_ab(&wt_a, w);
                g.diag().sum()
            } else {
                let mut solved = a.clone();
                let mut solvedview = array2_to_matmut(&mut solved);
                if let Some(f) = h_factor.as_ref() {
                    f.solve_in_place(solvedview.as_mut());
                }
                solved.diag().sum()
            }
        };
        // Matrix-free D²(Hphi) trace contraction, identical to mixed block:
        // use tr(W^T (D²Hphi[u,v] W)) on the active positive solve subspace.
        let trace_hdag_firthsecond =
            |op: &FirthDenseOperator, u_dir: &FirthDirection, v_dir: &FirthDirection| -> f64 {
                Self::trace_hdag_operator_apply(
                    p_dim,
                    h_posw_for_solve.as_ref(),
                    h_posw_for_solve_t.as_ref(),
                    |a| solve_h_mat(a),
                    |basis| Self::firth_hphisecond_direction_apply(op, u_dir, v_dir, basis),
                )
            };
        let trace_hdag_firth_first = |op: &FirthDenseOperator, u_dir: &FirthDirection| -> f64 {
            Self::trace_hdag_operator_apply(
                p_dim,
                h_posw_for_solve.as_ref(),
                h_posw_for_solve_t.as_ref(),
                |a| solve_h_mat(a),
                |basis| Self::firth_hphi_direction_apply(op, u_dir, basis),
            )
        };
        let trace_hdag_firth_tau_partial =
            |op: &FirthDenseOperator, x_tau: &Array2<f64>, kernel: &FirthTauPartialKernel| -> f64 {
                Self::trace_hdag_operator_apply(
                    p_dim,
                    h_posw_for_solve.as_ref(),
                    h_posw_for_solve_t.as_ref(),
                    |a| solve_h_mat(a),
                    |basis| Self::firth_hphi_tau_partial_apply(op, x_tau, kernel, basis),
                )
            };
        let trace_hdag_b_hdag_c = |b: &Array2<f64>, c_mat: &Array2<f64>| -> f64 {
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let gb = fast_ab(&fast_ab(w_t, b), w);
                let gc = fast_ab(&fast_ab(w_t, c_mat), w);
                Self::trace_product(&gb, &gc)
            } else {
                let mut left = b.clone();
                let mut right = c_mat.clone();
                let mut leftview = array2_to_matmut(&mut left);
                let mut rightview = array2_to_matmut(&mut right);
                if let Some(f) = h_factor.as_ref() {
                    f.solve_in_place(leftview.as_mut());
                    f.solve_in_place(rightview.as_mut());
                }
                Self::trace_product(&left, &right)
            }
        };

        // Build first directional solves for all tau directions.
        let u = &pirls_result.solveweights
            * &(&pirls_result.solveworking_response - &pirls_result.final_eta);
        let c = &pirls_result.solve_c_array;
        let d = &pirls_result.solve_d_array;
        let w_diag = &pirls_result.solveweights;
        let lambdas = rho.mapv(f64::exp);
        let transformed_first_components: Vec<Vec<PenaltyDerivativeComponent>> = hyper_dirs
            .iter()
            .map(|dir| {
                Self::transform_penalty_components(
                    dir.penalty_first_components(),
                    &reparam_result.qs,
                    free_basis_opt.as_ref(),
                )
            })
            .collect();
        let a_tau: Vec<Array2<f64>> = transformed_first_components
            .iter()
            .map(|components| {
                components.iter().fold(
                    Array2::<f64>::zeros((p_dim, p_dim)),
                    |mut acc, component| {
                        acc.scaled_add(rho[component.penalty_index].exp(), &component.matrix);
                        acc
                    },
                )
            })
            .collect();
        let mut a_tau_tau = vec![vec![None; psi_dim]; psi_dim];
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
                let total = transformed.iter().fold(
                    Array2::<f64>::zeros((p_dim, p_dim)),
                    |mut acc, component| {
                        acc.scaled_add(rho[component.penalty_index].exp(), &component.matrix);
                        acc
                    },
                );
                a_tau_tau[i][j] = Some(total);
            }
        }
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    &x_eval,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };

        let mut beta_tau = vec![Array1::<f64>::zeros(p_dim); psi_dim];
        let mut eta_tau = vec![Array1::<f64>::zeros(x_eval.nrows()); psi_dim];
        let mut h_tau = vec![Array2::<f64>::zeros((p_dim, p_dim)); psi_dim];
        let mut firth_tau_dirs: Option<Vec<FirthDirection>> = firth_op
            .as_ref()
            .map(|_| Vec::<FirthDirection>::with_capacity(psi_dim));
        let mut firth_tau_kernels: Option<Vec<FirthTauPartialKernel>> = firth_op
            .as_ref()
            .map(|_| Vec::<FirthTauPartialKernel>::with_capacity(psi_dim));
        let mut x_tau_beta = vec![Array1::<f64>::zeros(x_eval.nrows()); psi_dim];
        let mut weighted = Array2::<f64>::zeros(x_eval.raw_dim());
        for j in 0..psi_dim {
            x_tau_beta[j] = x_tau_t[j].dot(&beta_eval);
            let weighted_x_tau_beta = w_diag * &x_tau_beta[j];
            let mut g_tau = x_tau_t[j].t().dot(&u)
                - x_eval.t().dot(&weighted_x_tau_beta)
                - a_tau[j].dot(&beta_eval);
            if let Some(op) = firth_op.as_ref() {
                let tau_bundle = Self::firth_exact_tau_kernel(op, &x_tau_t[j], &beta_eval, true);
                g_tau -= &tau_bundle.gphi_tau;
                if let Some(v) = firth_tau_kernels.as_mut() {
                    let kernel = tau_bundle.tau_kernel.expect(
                        "firth_exact_tau_kernel(include_hphi_tau_kernel=true) should return kernel",
                    );
                    v.push(kernel);
                }
            }
            beta_tau[j] = solve_hvec(&g_tau);
            eta_tau[j] = &x_tau_beta[j] + &x_eval.dot(&beta_tau[j]);

            let runtime_link = pert_state.runtime_inverse_link();
            let w_tau_j = match crate::pirls::directionalworking_curvature_from_etawith_state(
                &runtime_link,
                &pirls_result.final_eta,
                pert_state.weights,
                &pirls_result.solveweights,
                &eta_tau[j],
            )? {
                DirectionalWorkingCurvature::Diagonal(diag) => diag,
            };
            let mut h_j = x_tau_t[j].t().dot(&Self::row_scale(&x_eval, w_diag));
            h_j += &x_eval.t().dot(&Self::row_scale(&x_tau_t[j], w_diag));
            h_j += &Self::xt_diag_x_dense_into(&x_eval, &w_tau_j, &mut weighted);
            h_j += &a_tau[j];
            if let Some(op) = firth_op.as_ref() {
                if x_tau_t[j].iter().any(|v| *v != 0.0) {
                    let k = firth_tau_kernels
                        .as_ref()
                        .and_then(|v| v.get(j))
                        .expect("firth tau kernel should be cached")
                        .clone();
                    h_j -= &Self::firth_hphi_tau_partial_apply(
                        op,
                        &x_tau_t[j],
                        &k,
                        &Array2::<f64>::eye(p_dim),
                    );
                }
                let dir = Self::firth_direction(op, &beta_tau[j]);
                h_j -= &Self::firth_hphi_direction(op, &dir);
                if let Some(v) = firth_tau_dirs.as_mut() {
                    v.push(dir);
                }
            }
            h_tau[j] = h_j;
        }

        // Structural S pseudo-inverse for +0.5 tr(S^+ A_j S^+ A_i) block.
        let mut s_eval = Array2::<f64>::zeros((p_dim, p_dim));
        for (k, r_k) in reparam_result.rs_transformed.iter().enumerate() {
            let r_proj = if let Some(z) = free_basis_opt.as_ref() {
                r_k.dot(z)
            } else {
                r_k.clone()
            };
            s_eval += &r_proj.t().dot(&r_proj).mapv(|v| lambdas[k] * v);
        }
        let (s_eigs, svecs) = s_eval
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let s_max = s_eigs
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            .max(1.0);
        let s_tol = (p_dim.max(1) as f64) * f64::EPSILON * s_max;
        let mut s_dag = Array2::<f64>::zeros((p_dim, p_dim));
        for i in 0..p_dim {
            let ev = s_eigs[i];
            if ev > s_tol {
                let ucol = svecs.column(i).to_owned();
                let outer = ucol
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&ucol.view().insert_axis(Axis(0)));
                s_dag += &outer.mapv(|v| v / ev);
            }
        }
        let sdag_a: Vec<Array2<f64>> = a_tau.iter().map(|a| s_dag.dot(a)).collect();

        for i in 0..psi_dim {
            for j in i..psi_dim {
                // Coupled IFT second derivative:
                //   H * beta_{tau_i,tau_j} = -(H_{tau_j} beta_{tau_i} + g_{tau_i,tau_j})
                // with
                //   g_{tau_i,tau_j} = d/dtau_j [ g_{tau_i} ].
                //
                // Expanded score-side terms in this implementation:
                //   g_tau = X_tau^T u - X^T W(X_tau beta) - S_tau beta - (gphi)_tau|beta,
                //   g_taui,tauj = d/dtau_j[g_taui] at fixed-beta partial stage,
                // then totalized through beta_tau_j via the linear solve above.
                //
                // Term map:
                //   term1 = X_{τ_i}^T u_{τ_j}
                //   term2a+term2b+term2c = d/dτ_j [ X^T W(X_{τ_i}β) ]
                //   a_tau[i].dot(beta_tau[j]) = S_{τ_i} β_{τ_j}.
                let u_tau_j = -(w_diag * &eta_tau[j]);
                let term1 = x_tau_t[i].t().dot(&u_tau_j);
                let x_tau_tau_ij = x_tau_tau_t[i][j].as_ref();
                let term1_xij = x_tau_tau_ij
                    .map(|xij| xij.t().dot(&u))
                    .unwrap_or_else(|| Array1::<f64>::zeros(p_dim));
                let term2a = x_tau_t[j].t().dot(&(w_diag * &x_tau_beta[i]));
                let term2b = x_eval.t().dot(&((&(c * &eta_tau[j])) * &x_tau_beta[i]));
                let term2c = x_eval.t().dot(&(w_diag * &x_tau_t[i].dot(&beta_tau[j])));
                let xij_beta = x_tau_tau_ij
                    .map(|xij| xij.dot(&beta_eval))
                    .unwrap_or_else(|| Array1::<f64>::zeros(x_eval.nrows()));
                let term2d = x_eval.t().dot(&(w_diag * &xij_beta));
                let mut g_ij = term1 + term1_xij
                    - term2a
                    - term2b
                    - term2c
                    - term2d
                    - a_tau[i].dot(&beta_tau[j])
                    - a_tau_tau[i][j]
                        .as_ref()
                        .map(|a| a.dot(&beta_eval))
                        .unwrap_or_else(|| Array1::<f64>::zeros(p_dim));
                if let (Some(op), Some(kernels)) = (firth_op.as_ref(), firth_tau_kernels.as_ref()) {
                    let mut btj = Array2::<f64>::zeros((p_dim, 1));
                    btj.column_mut(0).assign(&beta_tau[j]);
                    let gphi_ij =
                        Self::firth_hphi_tau_partial_apply(op, &x_tau_t[i], &kernels[i], &btj)
                            .column(0)
                            .to_owned();
                    g_ij -= &gphi_ij;
                    if let Some(xij) = x_tau_tau_ij {
                        let xij_bundle = Self::firth_exact_tau_kernel(op, xij, &beta_eval, true);
                        g_ij -= &xij_bundle.gphi_tau;
                    }
                }
                let rhs_ij = -(&h_tau[j].dot(&beta_tau[i]) + &g_ij);
                let beta_ij = solve_hvec(&rhs_ij);
                let mut eta_ij = x_tau_t[i].dot(&beta_tau[j])
                    + x_tau_t[j].dot(&beta_tau[i])
                    + x_eval.dot(&beta_ij);
                eta_ij += &xij_beta;

                let diag_i = c * &eta_tau[i];
                let diag_j = c * &eta_tau[j];
                let c_tau_j = d * &eta_tau[j];
                let diag_ij = &(&c_tau_j * &eta_tau[i]) + &(c * &eta_ij);

                let mut h_ij = Array2::<f64>::zeros((p_dim, p_dim));
                if let Some(xij) = x_tau_tau_ij {
                    h_ij += &xij.t().dot(&Self::row_scale(&x_eval, w_diag));
                    h_ij += &x_eval.t().dot(&Self::row_scale(xij, w_diag));
                }
                h_ij += &x_tau_t[i].t().dot(&Self::row_scale(&x_eval, &diag_j));
                h_ij += &x_eval.t().dot(&Self::row_scale(&x_tau_t[i], &diag_j));
                h_ij += &x_tau_t[i]
                    .t()
                    .dot(&Self::row_scale(&x_tau_t[j], &pirls_result.solveweights));
                h_ij += &x_tau_t[j]
                    .t()
                    .dot(&Self::row_scale(&x_tau_t[i], &pirls_result.solveweights));
                h_ij += &x_tau_t[j].t().dot(&Self::row_scale(&x_eval, &diag_i));
                h_ij += &x_eval.t().dot(&Self::row_scale(&x_tau_t[j], &diag_i));
                h_ij += &Self::xt_diag_x_dense_into(&x_eval, &diag_ij, &mut weighted);
                if let Some(aij) = a_tau_tau[i][j].as_ref() {
                    h_ij += aij;
                }

                let mut d1_trace_correction = 0.0_f64;
                let mut d2_trace_correction = 0.0_f64;
                let mut dtau_trace_correction = 0.0_f64;
                if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_tau_dirs.as_ref()) {
                    let dir_ij = Self::firth_direction(op, &beta_ij);
                    d1_trace_correction = trace_hdag_firth_first(op, &dir_ij);
                    d2_trace_correction = trace_hdag_firthsecond(op, &dirs[i], &dirs[j]);
                    if let Some(xij) = x_tau_tau_ij {
                        let xij_bundle = Self::firth_exact_tau_kernel(op, xij, &beta_eval, true);
                        if let Some(kernel) = xij_bundle.tau_kernel.as_ref() {
                            dtau_trace_correction += trace_hdag_firth_tau_partial(op, xij, kernel);
                        }
                    }
                    if x_tau_t[i].iter().any(|v| *v != 0.0) {
                        let k_i_jtau =
                            Self::firth_hphi_tau_partial_prepare(op, &x_tau_t[i], &beta_tau[j]);
                        dtau_trace_correction +=
                            trace_hdag_firth_tau_partial(op, &x_tau_t[i], &k_i_jtau);
                    }
                    if x_tau_t[j].iter().any(|v| *v != 0.0) {
                        let k_j_itau =
                            Self::firth_hphi_tau_partial_prepare(op, &x_tau_t[j], &beta_tau[i]);
                        dtau_trace_correction +=
                            trace_hdag_firth_tau_partial(op, &x_tau_t[j], &k_j_itau);
                    }
                }

                let q = beta_eval.dot(&a_tau[i].dot(&beta_tau[j]))
                    + 0.5
                        * a_tau_tau[i][j]
                            .as_ref()
                            .map(|aij| beta_eval.dot(&aij.dot(&beta_eval)))
                            .unwrap_or(0.0);
                // l = 0.5[ tr(H_+^† H_{ij}) - tr(H_+^† H_j H_+^† H_i) ]
                // with explicit subtractive Firth trace corrections for:
                //   D(Hphi)[beta_ij], D²(Hphi)[beta_i,beta_j],
                //   and mixed tau-partial terms.
                let l = 0.5
                    * (trace_hdag(&h_ij)
                        - d1_trace_correction
                        - d2_trace_correction
                        - dtau_trace_correction
                        - trace_hdag_b_hdag_c(&h_tau[j], &h_tau[i]));
                // P_{ij} = -0.5[ tr(S^+ S_{ij}) - tr(S^+ S_j S^+ S_i) ].
                let p_term = 0.5 * Self::trace_product(&sdag_a[j], &sdag_a[i])
                    - 0.5
                        * a_tau_tau[i][j]
                            .as_ref()
                            .map(|aij| Self::trace_product(&s_dag, aij))
                            .unwrap_or(0.0);
                let val = q + l + p_term;
                h_tt[[i, j]] = val;
                h_tt[[j, i]] = val;
            }
        }
        for i in 0..psi_dim {
            for j in 0..i {
                let avg = 0.5 * (h_tt[[i, j]] + h_tt[[j, i]]);
                h_tt[[i, j]] = avg;
                h_tt[[j, i]] = avg;
            }
        }
        Ok(h_tt)
    }

    pub(crate) fn compute_joint_hyperhessian(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<Array2<f64>, EstimationError> {
        let psi_dim = self.validate_joint_hyper_inputs(theta, rho_dim, hyper_dirs)?;
        let blocks = JointHyperThetaBlocks::new(theta, rho_dim);
        let rho = blocks.rho_owned();
        let psi = blocks.psi_owned();
        let eff_dirs = Self::relinearized_hyper_dirs(hyper_dirs, &psi, self.p)?;
        let (h_rr, h_rtau, h_tt) = if psi_dim > 0 {
            let pert_state = self.build_joint_perturbed_state(&psi, hyper_dirs)?;
            let bundle = pert_state.obtain_eval_bundle(&rho)?;
            (
                // Joint block assembly is kept fully analytic/objective-consistent.
                // Do not route through FD fallback policy here.
                pert_state.compute_lamlhessian_exact(&rho)?,
                self.compute_mixed_rho_tau_block(&pert_state, &rho, &bundle, &eff_dirs)?,
                self.compute_tau_tau_block(&pert_state, &rho, &bundle, &eff_dirs)?,
            )
        } else {
            (
                self.compute_lamlhessian_exact(&rho)?,
                Array2::<f64>::zeros((rho_dim, 0)),
                Array2::<f64>::zeros((0, 0)),
            )
        };
        if h_rr.nrows() != rho_dim || h_rr.ncols() != rho_dim {
            return Err(EstimationError::InvalidInput(format!(
                "rho Hessian shape mismatch in joint assembly: expected {}x{}, got {}x{}",
                rho_dim,
                rho_dim,
                h_rr.nrows(),
                h_rr.ncols()
            )));
        }
        let mut h = Array2::<f64>::zeros((rho_dim + psi_dim, rho_dim + psi_dim));
        h.slice_mut(s![..rho_dim, ..rho_dim]).assign(&h_rr);
        if psi_dim > 0 {
            h.slice_mut(s![..rho_dim, rho_dim..]).assign(&h_rtau);
            h.slice_mut(s![rho_dim.., ..rho_dim])
                .assign(&h_rtau.t().to_owned());
            h.slice_mut(s![rho_dim.., rho_dim..]).assign(&h_tt);
        }
        Ok(h)
    }

    pub(crate) fn compute_joint_hypercostgradienthessian(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
        let (cost, grad) = self.compute_joint_hypercostgradient(theta, rho_dim, hyper_dirs)?;
        let hess = self.compute_joint_hyperhessian(theta, rho_dim, hyper_dirs)?;
        Ok((cost, grad, hess))
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
        // where L* = -ℓ + 0.5 β'Sβ - Φ and H = X'WX + S - H_φ on the Firth path.
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
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut x_transformed_eval = pirls_result.x_transformed.clone();
        let mut e_eval = reparam_result.e_transformed.clone();

        let mut x_psi_t = hyper_dir.x_tau_original.dot(&reparam_result.qs);

        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::projectwith_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::projectwith_basis(bundle.h_total.as_ref(), z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            let x_dense_arc = pirls_result.x_transformed.to_dense_arc();
            x_transformed_eval = DesignMatrix::Dense(x_dense_arc.as_ref().dot(z));
            e_eval = reparam_result.e_transformed.dot(z);
            x_psi_t = x_psi_t.dot(z);
        }
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
                acc.scaled_add(rho[component.penalty_index].exp(), &component.matrix);
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
        let x_dense_arc = x_transformed_eval.to_dense_arc();
        let x_dense = x_dense_arc.as_ref();
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
        let weighted_xpsi_beta = &pirls_result.solveweights * &xpsi_beta;
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

        let w = &pirls_result.solveweights;
        let wx = Self::row_scale(&x_dense, w);
        let wx_psi = Self::row_scale(&x_psi_t, w);

        let mut h_psi = x_psi_t.t().dot(&wx);
        h_psi += &x_dense.t().dot(&wx_psi);

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
                let runtime_link = self.runtime_inverse_link();
                let w_direction = crate::pirls::directionalworking_curvature_from_etawith_state(
                    &runtime_link,
                    &pirls_result.final_eta,
                    self.weights,
                    &pirls_result.solveweights,
                    &eta_psi,
                )?;
                let mut xwpsi = x_dense.clone();
                match &w_direction {
                    // Directional curvature derivative:
                    //   W_τ = T[η̇].
                    //
                    // The family-dispatched PIRLS callback returns the operator form of
                    // this derivative. Built-in links are diagonal in observation space,
                    // so the operator is represented by the per-row vector `w_direction`.
                    DirectionalWorkingCurvature::Diagonal(w_direction) => {
                        xwpsi = Self::row_scale(&xwpsi, w_direction);
                    }
                }
                // Exact total curvature derivative:
                //   H_τ = d/dτ(X^T W X + S)
                //       = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ.
                h_psi += &x_dense.t().dot(&xwpsi);
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
}
