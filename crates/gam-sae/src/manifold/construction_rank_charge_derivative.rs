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
        if self.assignment.logits_are_fixed() {
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
    /// The chargeable rank is integer-valued, so `C` is piecewise smooth: constant
    /// in the rank away from an MP-edge crossing or the vanished/alive threshold,
    /// and discontinuous at them. This is the within-branch differential on the
    /// branch `wbic_audit::rank_charge_stratum` classifies at this state, the same
    /// producer the value calls; `RankChargeStratum::nearest_mp_boundary` names the
    /// direction closest to the edge and the jump its crossing adds. The generic
    /// outer contract (`OuterEval`) carries a value and this gradient but no branch
    /// identity, so a trial point across a boundary shows its jump only through the
    /// value. The smooth pieces are
    /// `basis_edf = tr(G(G+λS)⁻¹)` and `N_eff = Σ_i a_i²`, with
    /// `G = Σ_i a_i² φ_i φ_iᵀ`. Their exact differential supplies both the
    /// direct `log λ_smooth` channel and the implicit `(logit, t)` response.
    /// Decoder coefficients affect only the discrete production-rank branch, so the
    /// within-branch beta differential is exactly zero.
    ///
    /// #2267 — `geometry` is the spectral block the dense evaluation priced `½log|A|` on, so
    /// the dispersion's fitted-response divergence reads it instead of decomposing `A` a
    /// second time, and where the value already priced the dispersion on it (#2933 F39) the
    /// dispersion is read as priced. `None` is the streaming route, whose value routes the
    /// divergence by admission; the dense gradient refuses to assemble without its
    /// evaluation's block.
    pub(crate) fn production_rank_charge_derivative(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        geometry: Option<&DenseExactAGeometry>,
    ) -> Result<ProductionRankChargeDerivative, String> {
        self.assignment.validate_rho_domain(rho)?;
        // #2933 F39 — the value priced this state's dispersion and left it on the geometry
        // it handed here, so the fitted-response divergence is not formed a second time.
        let dispersion = match geometry.and_then(|geometry| geometry.rank_charge_dispersion) {
            Some(dispersion) => dispersion,
            None => {
                let residual = self.reconstruction_residual(target, rho)?;
                self.reconstruction_dispersion_with_geometry(
                    loss,
                    cache,
                    rho,
                    residual.view(),
                    geometry.map(|geometry| HeldResponseGeometry::FixedFrame(&geometry.block)),
                )?
            }
        }
        .raw_output_noise_variance;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams)?;
        let n_eff = self.per_atom_effective_sample_size();
        let lambda = rho.lambda_smooth_vec()?;
        let p = self.output_dim() as f64;
        let mut atom_differentials = Vec::with_capacity(self.k_atoms());
        let mut direct_rho = Array1::<f64>::zeros(rho.flat_coordinates().len());

        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let gram = &grams[atom_idx];
            let m = atom.basis_size();
            let n_atom = n_eff[atom_idx];
            let stratum = super::wbic_audit::rank_charge_stratum(
                gram,
                atom.decoder_coefficients(),
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
            let rank = stratum.production_chargeable_rank() as f64;
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
            // #2935 — a curvature-parameterised penalty moves `basis_edf` through
            // `S(κ)`: `∂tr((G+λS)⁻¹G)/∂κ = −λ·tr((G+λS)⁻¹G(G+λS)⁻¹ ∂S/∂κ)`.
            let kappa_penalty_derivative = rho
                .kappa_flat_index(atom_idx)
                .zip(atom.smooth_penalty_kappa_derivative()?);
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
                if let Some((kappa_index, ds)) = kappa_penalty_derivative {
                    let inv_g_inv_ds = inverse_gram_inverse.dot(ds);
                    let edf_kappa =
                        -lambda[atom_idx] * (0..m).map(|i| inv_g_inv_ds[[i, i]]).sum::<f64>();
                    direct_rho[kappa_index] += 0.5 * rank * log_n * edf_kappa;
                }
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

    /// Rank-charge audit of every atom at one evaluated state (#2933 F31).
    ///
    /// Each [`AtomRankChargeAudit`] reads the same residual, dispersion, decoder
    /// Grams and occupancies that `production_rank_charge_derivative` and
    /// the criterion price, and classifies through the same stratum producer, so
    /// its stratum is the priced branch. Beside it the audit reports the
    /// conditional noise-only law of the reconstruction energies and the exact
    /// tempered posterior of the decoder block at `β = 1/ln N_eff,k`, with the
    /// conditional score `b = Φᵀdiag(a)·(target − other atoms) = G·B − Φᵀdiag(a)·r`
    /// for the residual `r = fitted − target`.
    ///
    /// That decoder likelihood is the data fit only when rows are unweighted, the
    /// row metric does not whiten, frames are inactive and no behavior block
    /// augments the output. Other configurations are refused, and an atom with
    /// `N_eff ≤ 1` has no inverse temperature `1/ln N_eff` and is refused too.
    pub fn rank_charge_audit(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<AtomRankChargeAudit>, String> {
        self.assignment.validate_rho_domain(rho)?;
        if self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood())
        {
            return Err("SaeManifoldTerm::rank_charge_audit: a whitening row metric makes the \
                        decoder likelihood non-isotropic; the conditional tempered posterior \
                        assumes the identity output metric"
                .to_string());
        }
        if self
            .row_loss_weights
            .as_deref()
            .is_some_and(|weights| weights.iter().any(|&weight| weight != 1.0))
        {
            return Err("SaeManifoldTerm::rank_charge_audit: row loss weights reweight the \
                        decoder likelihood; the conditional tempered posterior assumes unit \
                        row weights"
                .to_string());
        }
        if self.frames_active() {
            return Err("SaeManifoldTerm::rank_charge_audit: an active Grassmann frame constrains \
                        the decoder; the conditional tempered posterior integrates the full \
                        decoder block"
                .to_string());
        }
        if self.behavior.is_some() {
            return Err("SaeManifoldTerm::rank_charge_audit: a behavior block augments the output; \
                        the conditional tempered posterior assumes the reconstruction target \
                        alone"
                .to_string());
        }
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion = self
            .reconstruction_dispersion(loss, cache, rho, residual.view())?
            .raw_output_noise_variance;
        let mut grams = self.empty_decoder_gram_accumulator();
        self.accumulate_decoder_gram(&mut grams)?;
        let n_eff = self.per_atom_effective_sample_size();
        let lambda = rho.lambda_smooth_vec()?;
        let assignments = self.assignment.assignments();
        let p = self.output_dim();
        let mut audits = Vec::with_capacity(self.k_atoms());
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let coordinate = &self.assignment.coords[atom_idx];
            let gram = &grams[atom_idx];
            let decoder = atom.decoder_coefficients();
            let m = atom.basis_size();
            let occupancy = n_eff[atom_idx];
            let stratum = super::wbic_audit::rank_charge_stratum(
                gram,
                decoder,
                occupancy,
                p as f64,
                dispersion,
                lambda[atom_idx],
                Some(atom.smooth_penalty()),
            )?;
            let noise_null = conditional_noise_null(
                gram,
                occupancy,
                p,
                OutputNoiseSpectrum::isotropic(dispersion, p),
                lambda[atom_idx],
                Some(atom.smooth_penalty()),
            )?;
            let mp_false_rank_probability_bound =
                noise_null.false_rank_probability_bound(stratum.mp_reconstruction_rank_edge())?;
            let log_occupancy = occupancy.ln();
            if !(log_occupancy.is_finite() && log_occupancy > 0.0) {
                return Err(format!(
                    "SaeManifoldTerm::rank_charge_audit: atom {atom_idx} has N_eff={occupancy}; \
                     the inverse temperature 1/ln N_eff needs N_eff > 1"
                ));
            }
            let inverse_temperature = log_occupancy.recip();
            let mut score = gram.dot(decoder);
            for row in 0..self.n_obs() {
                let gate = assignments[[row, atom_idx]];
                if gate == 0.0 {
                    continue;
                }
                for basis_col in 0..m {
                    let weight = gate * atom.basis_values[[row, basis_col]];
                    if weight == 0.0 {
                        continue;
                    }
                    for out_col in 0..p {
                        score[[basis_col, out_col]] -= weight * residual[[row, out_col]];
                    }
                }
            }
            let tempered_posterior = conditional_decoder_tempered_posterior(
                gram,
                &score,
                dispersion,
                lambda[atom_idx],
                Some(atom.smooth_penalty()),
                inverse_temperature,
            )?;
            audits.push(AtomRankChargeAudit {
                atom: atom_idx,
                basis_dim: m,
                output_dim: p,
                storage_dim: m * p,
                intrinsic_dim: coordinate.manifold().intrinsic_dim(coordinate.latent_dim()),
                dispersion,
                lambda_smooth: lambda[atom_idx],
                inverse_temperature,
                production_minus_tempered: stratum.production_charge()
                    - tempered_posterior.total(),
                nearest_mp_boundary: stratum.nearest_mp_boundary(),
                stratum,
                noise_null,
                mp_false_rank_probability_bound,
                tempered_posterior,
            });
        }
        Ok(audits)
    }
}
