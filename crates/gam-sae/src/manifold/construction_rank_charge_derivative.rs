// Analytic differential of the production chargeable-rank penalty used by the
// quasi-Laplace criterion. This file is included from `construction.rs` and
// therefore shares that module's imports and private access.

pub(crate) struct ProductionRankChargeDerivative {
    pub(crate) direct_rho: Array1<f64>,
    pub(crate) theta: SaeArrowVector,
}

struct ProductionRankChargeAtomDifferential {
    gram: Array2<f64>,
    occupancy: f64,
}

impl SaeManifoldTerm {
    fn rank_charge_assignment_derivative(
        &self,
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
            AssignmentMode::ThresholdGate { temperature, .. } if atom == wrt_atom => {
                let a = assignments[atom];
                a * (1.0 - a) / temperature
            }
            AssignmentMode::OrderedBetaBernoulli { .. }
            | AssignmentMode::ThresholdGate { .. }
            | AssignmentMode::TopK { .. } => 0.0,
        }
    }

    /// Differential of
    /// `C = Σ_k ½ rank_chargeable,k · basis_edf,k · log(max(N_eff,k, 1))`
    /// on one fixed production-rank branch.
    ///
    /// The chargeable rank is integer-valued and therefore locally constant away
    /// from an MP-edge crossing or the vanished/alive threshold. The smooth pieces are
    /// `basis_edf = tr(G(G+λS)⁻¹)` and `N_eff = Σ_i a_i²`, with
    /// `G = Σ_i a_i² φ_i φ_iᵀ`. Their exact differential supplies both the
    /// direct `log λ_smooth` channel and the implicit `(logit, t)` response.
    /// Decoder coefficients affect only the discrete production-rank branch, so the
    /// within-branch beta differential is exactly zero.
    pub(crate) fn production_rank_charge_derivative(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<ProductionRankChargeDerivative, String> {
        self.assignment.validate_rho_domain(rho)?;
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion = self.reconstruction_dispersion(loss, cache, rho, Some(residual.view()))?;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams)?;
        let n_eff = self.per_atom_effective_sample_size();
        let lambda = rho.lambda_smooth_vec()?;
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
                Some(atom.smooth_penalty()),
            )?;
            // #2258 — the CHARGEABLE rank, not the raw hard MP count: the
            // value path promotes a below-reconstruction-rank-edge-but-alive atom to
            // rank 1, and the derivative must take the SAME branch (the
            // promoted rank is locally constant, so the differential's form
            // is unchanged). Only a genuinely VANISHED decoder — the
            // Laplace-invalid regime the veto prices +∞ — remains an error
            // here, matching the value side's categorical veto.
            let rank = spectrum.production_chargeable_rank() as f64;
            if !(rank > 0.0) {
                return Err(format!(
                    "production_rank_charge_derivative: atom {atom_idx} is on the rank-zero \
                     Laplace-invalid branch (vanished decoder)"
                ));
            }
            let log_n = n_atom.max(1.0).ln();
            if m == 0 || log_n == 0.0 {
                atom_differentials.push(ProductionRankChargeAtomDifferential {
                    gram: Array2::<f64>::zeros((m, m)),
                    occupancy: 0.0,
                });
                continue;
            }

            let mut penalized_gram = gram.clone();
            for row in 0..m {
                for col in 0..m {
                    penalized_gram[[row, col]] +=
                        lambda[atom_idx] * atom.smooth_penalty()[[row, col]];
                }
            }
            let factor = penalized_gram.cholesky(Side::Lower).map_err(|error| {
                format!(
                    "production_rank_charge_derivative: atom {atom_idx} penalized Gram \
                     factorization failed: {error}"
                )
            })?;
            let inverse = factor.solve_mat(&Array2::<f64>::eye(m));
            let edf_matrix = factor.solve_mat(gram);
            let raw_edf = (0..m).map(|i| edf_matrix[[i, i]]).sum::<f64>();
            let edf = super::construction::certified_basis_edf(
                raw_edf,
                m,
                "production_rank_charge_derivative",
            )?;
            let edf_is_interior = edf > 0.0 && edf < m as f64;
            let mut gram_differential = Array2::<f64>::zeros((m, m));
            let mut log_lambda_differential = 0.0_f64;
            if edf_is_interior {
                // d tr((G+λS)⁻¹G) / dG = A⁻¹ − A⁻¹GA⁻¹.
                // Writing this identity directly keeps the derivative paired to
                // the exact matrix used by the value, with no hidden diagonal
                // regularizer whose differential would otherwise be omitted.
                let inverse_gram_inverse = inverse.dot(gram).dot(&inverse);
                gram_differential = (&inverse - &inverse_gram_inverse) * (0.5 * rank * log_n);
                let inv_g_inv_s = inverse
                    .dot(gram)
                    .dot(&inverse)
                    .dot(atom.smooth_penalty());
                let edf_log_lambda =
                    -lambda[atom_idx] * (0..m).map(|i| inv_g_inv_s[[i, i]]).sum::<f64>();
                log_lambda_differential = 0.5 * rank * log_n * edf_log_lambda;
            }
            direct_rho[rho.smooth_flat_index(atom_idx)] += log_lambda_differential;
            let occupancy_differential = if n_atom > 1.0 {
                0.5 * rank * edf / n_atom
            } else {
                0.0
            };
            atom_differentials.push(ProductionRankChargeAtomDifferential {
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

        Ok(ProductionRankChargeDerivative {
            direct_rho,
            theta: SaeArrowVector {
                t: theta_t,
                beta: theta_beta,
            },
        })
    }

    /// PATH C channel — exact fixed-stratum second derivative of the rank-charge
    /// `direct_rho` channel (the `log λ_smooth` derivative of
    /// `C = Σ_k ½ rank_k · basis_edf_k · log N_eff,k`).
    ///
    /// At a frozen inner state the chargeable rank and `N_eff` are locally
    /// constant, so with the per-atom `A = G + λ S` (`G` = decoder Gram, `S` =
    /// reference smooth penalty), `basis_edf = tr(A⁻¹G)` and the gradient's
    /// direct channel is `direct_rho_k = ½ rank · log N · d(edf)/d log λ` with
    /// `d(edf)/d log λ = −λ tr(A⁻¹G A⁻¹S)`. Differentiating once more (using
    /// `dA⁻¹/dλ = −A⁻¹ S A⁻¹`) gives
    /// `d²(edf)/d(log λ)² = d(edf)/d log λ + 2 λ² tr((A⁻¹S)² A⁻¹G)`, hence the
    /// diagonal smooth entry
    /// `H_kk = direct_rho_k + rank · log N · λ² · tr(A⁻¹G (A⁻¹S)²)`.
    /// Atoms are independent (diagonal block); a non-interior-EDF atom's charge
    /// is on a locally constant branch, so both its gradient and this Hessian
    /// entry are zero — matching the value/gradient branch exactly. Mirrors the
    /// setup of [`Self::production_rank_charge_derivative`] so the two
    /// differentiate one object.
    pub(crate) fn rank_charge_direct_rho_hessian(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Array2<f64>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n_params = rho.to_flat().len();
        let mut hessian = Array2::<f64>::zeros((n_params, n_params));
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion =
            self.reconstruction_dispersion(loss, cache, rho, Some(residual.view()))?;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams)?;
        let n_eff = self.per_atom_effective_sample_size();
        let lambda = rho.lambda_smooth_vec()?;
        let p = self.output_dim() as f64;
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
                Some(atom.smooth_penalty()),
            )?;
            let rank = spectrum.production_chargeable_rank() as f64;
            if !(rank > 0.0) {
                return Err(format!(
                    "rank_charge_direct_rho_hessian: atom {atom_idx} is on the rank-zero \
                     Laplace-invalid branch (vanished decoder)"
                ));
            }
            let log_n = n_atom.max(1.0).ln();
            if m == 0 || log_n == 0.0 {
                continue;
            }
            let lam = lambda[atom_idx];
            let mut penalized_gram = gram.clone();
            for r in 0..m {
                for c in 0..m {
                    penalized_gram[[r, c]] += lam * atom.smooth_penalty()[[r, c]];
                }
            }
            let factor = penalized_gram.cholesky(Side::Lower).map_err(|error| {
                format!(
                    "rank_charge_direct_rho_hessian: atom {atom_idx} penalized Gram \
                     factorization failed: {error}"
                )
            })?;
            let m_g = factor.solve_mat(gram); // A⁻¹ G
            let raw_edf = (0..m).map(|i| m_g[[i, i]]).sum::<f64>();
            let edf = super::construction::certified_basis_edf(
                raw_edf,
                m,
                "rank_charge_direct_rho_hessian",
            )?;
            if !(edf > 0.0 && edf < m as f64) {
                // Non-interior EDF ⇒ locally constant charge ⇒ zero curvature.
                continue;
            }
            let m_s = factor.solve_mat(atom.smooth_penalty()); // A⁻¹ S
            let mg_ms = m_g.dot(&m_s); // A⁻¹G A⁻¹S
            let t1 = (0..m).map(|i| mg_ms[[i, i]]).sum::<f64>();
            let mg_ms2 = m_g.dot(&m_s.dot(&m_s)); // A⁻¹G (A⁻¹S)²
            let t_extra = (0..m).map(|i| mg_ms2[[i, i]]).sum::<f64>();
            let direct = 0.5 * rank * log_n * (-lam * t1);
            let h_kk = direct + rank * log_n * lam * lam * t_extra;
            let idx = rho.smooth_flat_index(atom_idx);
            hessian[[idx, idx]] += h_kk;
        }
        Ok(hessian)
    }
}
