// Analytic differential of the hard Marchenko--Pastur rank charge used by the
// quasi-Laplace criterion. This file is included from `construction.rs` and
// therefore shares that module's imports and private access.

pub(crate) struct HardRankChargeDerivative {
    pub(crate) direct_rho: Array1<f64>,
    pub(crate) theta: SaeArrowVector,
}

struct HardRankChargeAtomDifferential {
    gram: Array2<f64>,
    occupancy: f64,
}

impl SaeManifoldTerm {
    fn rank_charge_assignment_derivative(
        &self,
        row: usize,
        wrt_atom: usize,
        atom: usize,
        assignments: &[f64],
    ) -> f64 {
        if self.assignment.logit_is_fixed(wrt_atom) {
            return 0.0;
        }
        match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                let a_atom = assignments[atom];
                let a_wrt = assignments[wrt_atom];
                a_atom * ((if atom == wrt_atom { 1.0 } else { 0.0 }) - a_wrt) / temperature
            }
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } if atom == wrt_atom => {
                let a = assignments[atom];
                a * (1.0 - a) / temperature
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } if atom == wrt_atom && self.assignment.logits[[row, atom]] > threshold => {
                let a = assignments[atom];
                a * (1.0 - a) / temperature
            }
            AssignmentMode::OrderedBetaBernoulli { .. }
            | AssignmentMode::ThresholdGate { .. }
            | AssignmentMode::TopK { .. } => 0.0,
        }
    }

    /// Differential of
    /// `C = Σ_k ½ rank_hard,k · basis_edf,k · log(max(N_eff,k, 1))` on one
    /// fixed MP-rank branch.
    ///
    /// The hard MP count is integer-valued and therefore locally constant away
    /// from an eigenvalue/noise-edge crossing. The smooth pieces are
    /// `basis_edf = tr(G(G+λS)⁻¹)` and `N_eff = Σ_i a_i²`, with
    /// `G = Σ_i a_i² φ_i φ_iᵀ`. Their exact differential supplies both the
    /// direct `log λ_smooth` channel and the implicit `(logit, t)` response.
    /// Decoder coefficients affect only the discrete hard-rank branch, so the
    /// within-branch beta differential is exactly zero.
    pub(crate) fn hard_rank_charge_derivative(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<HardRankChargeDerivative, String> {
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion = self.reconstruction_dispersion(loss, cache, rho, Some(residual.view()))?;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams);
        let n_eff = self.per_atom_effective_sample_size();
        let lambda = rho.lambda_smooth_vec();
        let p = self.output_dim() as f64;
        let mut atom_differentials = Vec::with_capacity(self.k_atoms());
        let mut direct_rho = Array1::<f64>::zeros(rho.to_flat().len());

        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let gram = &grams[atom_idx];
            let m = atom.basis_size();
            let n_atom = n_eff[atom_idx];
            let spectrum = super::wbic_audit::recon_spectrum(
                gram,
                &atom.decoder_coefficients,
                n_atom,
                p,
                dispersion,
                lambda[atom_idx],
                Some(&atom.smooth_penalty),
            )?;
            let rank = spectrum.rank_hard();
            if !(rank > 0.0) {
                return Err(format!(
                    "hard_rank_charge_derivative: atom {atom_idx} is on the rank-zero \
                     Laplace-invalid branch"
                ));
            }
            let log_n = n_atom.max(1.0).ln();
            if m == 0 || log_n == 0.0 {
                atom_differentials.push(HardRankChargeAtomDifferential {
                    gram: Array2::<f64>::zeros((m, m)),
                    occupancy: 0.0,
                });
                continue;
            }

            let mut penalized_gram = gram.clone();
            for row in 0..m {
                for col in 0..m {
                    penalized_gram[[row, col]] +=
                        lambda[atom_idx] * atom.smooth_penalty[[row, col]];
                }
            }
            let factor = penalized_gram.cholesky(Side::Lower).map_err(|error| {
                format!(
                    "hard_rank_charge_derivative: atom {atom_idx} penalized Gram \
                     factorization failed: {error}"
                )
            })?;
            let inverse = factor.solve_mat(&Array2::<f64>::eye(m));
            let edf_matrix = factor.solve_mat(gram);
            let raw_edf = (0..m).map(|i| edf_matrix[[i, i]]).sum::<f64>();
            let edf_is_interior = raw_edf > 0.0 && raw_edf < m as f64;
            let edf = raw_edf.clamp(0.0, m as f64);
            let mut gram_differential = Array2::<f64>::zeros((m, m));
            let mut log_lambda_differential = 0.0_f64;
            if edf_is_interior {
                // d tr((G+λS)⁻¹G) / dG = A⁻¹ − A⁻¹GA⁻¹.
                // Writing this identity directly keeps the derivative paired to
                // the exact matrix used by the value, with no hidden diagonal
                // regularizer whose differential would otherwise be omitted.
                let inverse_gram_inverse = inverse.dot(gram).dot(&inverse);
                gram_differential = (&inverse - &inverse_gram_inverse)
                    * (0.5 * rank * log_n);
                let inv_g_inv_s = inverse.dot(gram).dot(&inverse).dot(&atom.smooth_penalty);
                let edf_log_lambda =
                    -lambda[atom_idx] * (0..m).map(|i| inv_g_inv_s[[i, i]]).sum::<f64>();
                let raw_log_lambda = rho.log_lambda_smooth[atom_idx];
                if SaeManifoldRho::clamped_log_strength(raw_log_lambda) == raw_log_lambda {
                    log_lambda_differential = 0.5 * rank * log_n * edf_log_lambda;
                }
            }
            direct_rho[rho.smooth_flat_index(atom_idx)] += log_lambda_differential;
            let occupancy_differential = if n_atom > 1.0 {
                0.5 * rank * edf / n_atom
            } else {
                0.0
            };
            atom_differentials.push(HardRankChargeAtomDifferential {
                gram: gram_differential,
                occupancy: occupancy_differential,
            });
        }

        let mut theta_t = Array1::<f64>::zeros(cache.delta_t_len());
        let theta_beta = Array1::<f64>::zeros(cache.k);
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        for row in 0..self.n_obs() {
            self.assignment.try_assignments_row_into(
                row,
                assignments
                    .as_slice_mut()
                    .expect("rank-charge assignment scratch is contiguous"),
            )?;
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let base = cache.row_offsets[row];
            for (slot, var) in vars.into_iter().enumerate() {
                theta_t[base + slot] = match var {
                    SaeLocalRowVar::Coord { atom, axis } => {
                        let a = assignments[atom];
                        if a == 0.0 {
                            0.0
                        } else {
                            let phi = self.atoms[atom].basis_values.row(row);
                            let dphi = self.atoms[atom].basis_jacobian.slice(s![row, .., axis]);
                            2.0 * a * a * dphi.dot(&atom_differentials[atom].gram.dot(&phi))
                        }
                    }
                    SaeLocalRowVar::Logit { atom: wrt_atom } => {
                        let mut derivative = 0.0_f64;
                        for atom in 0..self.k_atoms() {
                            let da = self.rank_charge_assignment_derivative(
                                row,
                                wrt_atom,
                                atom,
                                assignments
                                    .as_slice()
                                    .expect("rank-charge assignment scratch is contiguous"),
                            );
                            if da == 0.0 {
                                continue;
                            }
                            let a = assignments[atom];
                            let phi = self.atoms[atom].basis_values.row(row);
                            let gram_quadratic = phi.dot(&atom_differentials[atom].gram.dot(&phi));
                            derivative += 2.0
                                * a
                                * da
                                * (gram_quadratic + atom_differentials[atom].occupancy);
                        }
                        derivative
                    }
                };
            }
        }

        Ok(HardRankChargeDerivative {
            direct_rho,
            theta: SaeArrowVector {
                t: theta_t,
                beta: theta_beta,
            },
        })
    }
}
