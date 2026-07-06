//! Reconstruction-dispersion and shape-uncertainty methods, split out of the
//! tail of `construction.rs` to keep that tracked file under the #780 10k-line
//! gate. Holds the contiguous trailing `impl SaeManifoldTerm` block:
//! `reconstruction_dispersion` (the Gaussian dispersion `φ̂` estimator),
//! `assemble_shape_uncertainty`, `complete_born_atom_shape_bands`, and
//! `shape_uncertainty_without_decoder_covariance`. All are reached bare by
//! callers through `use super::*`, so their visibility is unchanged.

use super::*;

/// Per-row competing-basin summary for the #2133 SURE/Stein hard-selection
/// correction. `weight` is the Laplace posterior basin weight, `edf` is the
/// basin-local ARD-shrunk coordinate EDF, and `prediction` is the decoded row
/// at that basin mode.
#[derive(Clone, Debug)]
pub struct BasinSureSummary {
    pub weight: f64,
    pub edf: f64,
    pub prediction: Vec<f64>,
}

impl SaeManifoldTerm {
    /// Install the already-computed SURE/Stein RSS-deflation degrees of freedom
    /// for penalized hard basin selection. This is deliberately an additive
    /// deflation count, not a replacement for the local ARD trace: a
    /// single-dominant-basin row contributes zero here and is priced exactly by
    /// the historical single-basin formula.
    pub fn set_coordinate_selection_sure_dof(
        &mut self,
        dof: f64,
    ) -> Result<(), String> {
        if !dof.is_finite() || dof < 0.0 {
            return Err(format!(
                "set_coordinate_selection_sure_dof: expected finite non-negative dof, got {dof}"
            ));
        }
        self.coordinate_selection_sure_dof = dof;
        Ok(())
    }

    /// SURE/Stein RSS-deflation EDF of a penalized hard basin-selecting MAP row,
    /// averaged under the per-basin Laplace evidence.
    ///
    /// The returned value is an **extra** selection EDF beyond the current
    /// selected-basin local EDF. It satisfies the two analytic anchors required
    /// by #2133: if one basin dominates, the extra cost is zero; if multiple
    /// basins make identical predictions, the extra cost is zero. The scale is
    /// the posterior disagreement of the live basin predictions, deflated by the
    /// posterior certainty of the hard selector; unlike the mixture-mean EDF it
    /// never charges the whole between-basin variance to a discontinuous
    /// estimator that returns one mode rather than the posterior mean.
    pub fn hard_selection_sure_extra_dof(
        basins: &[BasinSureSummary],
        phi: f64,
    ) -> Result<f64, String> {
        if basins.len() <= 1 {
            return Ok(0.0);
        }
        if !phi.is_finite() || phi <= 0.0 {
            return Err(format!(
                "hard_selection_sure_extra_dof: φ must be finite positive, got {phi}"
            ));
        }
        let p = basins[0].prediction.len();
        if p == 0 {
            return Err("hard_selection_sure_extra_dof: empty prediction".into());
        }
        let mut weight_sum = 0.0_f64;
        let mut max_weight = 0.0_f64;
        let mut mean = vec![0.0_f64; p];
        let mut local_edf = 0.0_f64;
        for (idx, basin) in basins.iter().enumerate() {
            if !basin.weight.is_finite() || basin.weight < 0.0 {
                return Err(format!(
                    "hard_selection_sure_extra_dof: basin {idx} has invalid weight {}",
                    basin.weight
                ));
            }
            if !basin.edf.is_finite() || basin.edf < 0.0 {
                return Err(format!(
                    "hard_selection_sure_extra_dof: basin {idx} has invalid edf {}",
                    basin.edf
                ));
            }
            if basin.prediction.len() != p {
                return Err(format!(
                    "hard_selection_sure_extra_dof: basin {idx} prediction length {} != {p}",
                    basin.prediction.len()
                ));
            }
            weight_sum += basin.weight;
            max_weight = max_weight.max(basin.weight);
            local_edf += basin.weight * basin.edf;
            for (m, &value) in mean.iter_mut().zip(basin.prediction.iter()) {
                if !value.is_finite() {
                    return Err(format!(
                        "hard_selection_sure_extra_dof: basin {idx} has non-finite prediction"
                    ));
                }
                *m += basin.weight * value;
            }
        }
        if weight_sum <= 0.0 {
            return Err("hard_selection_sure_extra_dof: zero total basin weight".into());
        }
        for m in &mut mean {
            *m /= weight_sum;
        }
        let mut between = 0.0_f64;
        for basin in basins {
            let w = basin.weight / weight_sum;
            let sq: f64 = basin
                .prediction
                .iter()
                .zip(mean.iter())
                .map(|(&x, &m)| {
                    let d = x - m;
                    d * d
                })
                .sum();
            between += w * sq;
        }
        let certainty = (max_weight / weight_sum).clamp(0.0, 1.0);
        let extra = (between / phi) * certainty * (1.0 - certainty);
        Ok(extra.min(local_edf.max(0.0)))
    }

    /// Gaussian reconstruction dispersion `φ̂`, the scale that turns the
    /// unscaled inverse-Hessian β-block `S_β⁻¹` into a posterior covariance
    /// `Cov(β) = φ̂·S_β⁻¹` — the same `Vb = φ·H⁻¹` convention the main GAM
    /// inference path uses.
    ///
    /// `RSS = Σ_{i,c} (z_{ic} − ẑ_{ic})² = 2·data_fit` (the loss stores the
    /// half-sum `½Σr²`). The residual degrees of freedom subtract the effective
    /// parameter count from the `N·p` scalar observations:
    ///   * decoder β: `beta_dim − tr(λ_smooth · S_β⁻¹ · ⊕_k S_k⊗I_p)`, the
    ///     smoothness effective-dof already assembled for the Fellner-Schall
    ///     step (penalty-shrunk directions do not cost a full parameter);
    ///   * latent coordinates: enabled ARD axes use the exact ARD-shrunk trace
    ///     `Σ_k Σ_j (n_active_k − α_{kj}·tr_{kj}(H⁻¹))`; atoms with disabled
    ///     native ARD charge the full active coordinate count because those
    ///     latent variables are estimated without an ARD precision.
    ///
    /// The coordinate term is the **exact** ARD-shrunk effective dof of the
    /// latent block: along axis `(k,j)` the MacKay/Fellner-Schall edf is
    /// `n_active_k − α_{kj}·tr_{kj}(H⁻¹)`, the well-determined-direction count
    /// after the ARD prior `α_{kj}` shrinks each coordinate. `tr_{kj}(H⁻¹)` is
    /// the same posterior-variance trace [`Self::ard_inverse_traces`] assembles
    /// for the EFS ARD step (reused here, not recomputed), so the dispersion is
    /// consistent with the precision update `α_new = n/(‖t‖²+tr(H⁻¹))`. The
    /// per-axis scalar count `n_active_k` must match the support the trace sums
    /// over: `n` for the dense full-support layout, or the number of rows where
    /// atom `k` is active for the compact active-set layout (inactive
    /// prior-dominated coordinates contribute 0 to both the trace and the
    /// count, hence 0 edf). The residual dof is floored at 1 so `φ̂` stays
    /// finite and positive.
    pub(crate) fn reconstruction_dispersion(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
    ) -> Result<f64, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let n_scalar = (n * p) as f64;
        let rss = 2.0 * loss.data_fit;
        let smooth_edf: f64 = self
            .decoder_smoothness_effective_dof_per_atom(cache, &rho.lambda_smooth_vec())
            .map_err(|e| format!("reconstruction_dispersion: smooth edf: {e}"))?
            .iter()
            .sum();
        // #972 / #977 T1: the raw decoder-parameter count is `beta_dim` on the
        // full-`B` path, but when frames are active the estimated decoder freedom
        // is the factored border `Σ M_k·r_k` PLUS the `Σ r_k·(p−r_k)` Grassmann
        // frame degrees profiled out (both are genuinely estimated), which the
        // smoothness shrinkage `smooth_edf` (taken over the factored border) then
        // discounts. On the full-`B` path `factored_border_dim == beta_dim` and
        // `grassmann_evidence_dimension == 0`, so this is exactly `beta_dim`.
        let raw_decoder_dof = if self.frames_active() {
            (self.factored_border_dim() + self.grassmann_evidence_dimension()) as f64
        } else {
            self.beta_dim() as f64
        };
        let beta_edf = (raw_decoder_dof - smooth_edf).max(0.0);
        // Exact ARD-shrunk latent-coordinate edf, reusing the EFS trace cache.
        let traces = self
            .ard_inverse_traces(cache)
            .map_err(|e| format!("reconstruction_dispersion: ARD traces: {e}"))?;
        if rho.log_ard.len() != self.atoms.len() {
            return Err(format!(
                "reconstruction_dispersion: ρ has {} ARD atoms but term has {}",
                rho.log_ard.len(),
                self.atoms.len()
            ));
        }
        let mut coord_edf = 0.0_f64;
        for (k, atom) in self.atoms.iter().enumerate() {
            let d_k = atom.latent_dim;
            if traces[k].len() != d_k {
                return Err(format!(
                    "reconstruction_dispersion: trace shape mismatch at atom {k} \
                     (traces={}, d_k={d_k})",
                    traces[k].len()
                ));
            }
            let ard_len = rho.log_ard[k].len();
            if ard_len != 0 && ard_len != d_k {
                return Err(format!(
                    "reconstruction_dispersion: ARD shape mismatch at atom {k} \
                     (log_ard={ard_len}, d_k={d_k})"
                ));
            }
            // Scalar count matched to the trace support (see fn doc).
            let n_active_k = match self.last_row_layout {
                Some(ref layout) => layout
                    .active_atoms
                    .iter()
                    .filter(|active| active.contains(&k))
                    .count() as f64,
                None => n as f64,
            };
            if ard_len == 0 {
                coord_edf += n_active_k * d_k as f64;
                continue;
            }
            for j in 0..d_k {
                let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[k][j]);
                // edf_kj ∈ [0, n_active_k]; clamp against numerical drift.
                let edf_kj = (n_active_k - alpha * traces[k][j]).clamp(0.0, n_active_k);
                coord_edf += edf_kj;
            }
        }
        coord_edf += self.coordinate_selection_sure_dof;
        let coord_edf = coord_edf.clamp(0.0, n_scalar - beta_edf);
        let resid_dof = (n_scalar - beta_edf - coord_edf).max(1.0);
        let phi = rss / resid_dof;
        if !phi.is_finite() || phi < 0.0 {
            return Err(format!(
                "reconstruction_dispersion: non-finite/negative φ̂={phi} \
                 (RSS={rss}, resid_dof={resid_dof}, beta_edf={beta_edf}, coord_edf={coord_edf})"
            ));
        }
        Ok(phi.max(f64::MIN_POSITIVE))
    }

    /// Posterior covariance and ambient shape band for every atom — the
    /// user-facing uncertainty of the fitted manifold shapes.
    ///
    /// For atom `k` with decoder-block range `r_k` (see
    /// [`Self::beta_block_offsets`]), `Cov(β_k) = φ·S_β⁻¹[r_k, r_k]` is the
    /// φ-scaled posterior covariance of its decoder coefficients with the
    /// latent coordinates marginalized out. The ambient point at a coordinate
    /// `t` is `m_k(t) = Φ_k(t)·B_k`, *linear* in `β_k`, so its per-channel
    /// posterior variance is the closed form
    /// `Var_c(t) = Σ_{b1,b2} Φ_k(t)[b1] Φ_k(t)[b2] · Cov(β_k)[(b1,c),(b2,c)]`
    /// — no sampling. The band is evaluated at up to [`SHAPE_BAND_MAX_POINTS`]
    /// evenly-strided of the atom's own on-atom coordinates, reusing the basis
    /// values already stored on the atom, so it reports uncertainty exactly
    /// where the data lives and needs no basis-kind-specific grid.
    ///
    /// A near-degenerate atom has a near-singular Schur block, so `Cov(β_k)` —
    /// and the band — fans out automatically: the band width is a
    /// per-coordinate visual of how well each atom is identified.
    pub fn assemble_shape_uncertainty(
        &self,
        cache: &ArrowFactorCache,
        dispersion: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let p = self.output_dim();
        // #972 / #977 T1: the cache β block is the FACTORED border when frames
        // are active, so each atom's Schur inverse block is the `(M_k·r_k)`
        // coordinate covariance `Cov(vec C_k)`. We LIFT it to the full
        // `(M_k·p)` decoder covariance `Cov(vec B_k) = (I_{M_k} ⊗ U_k) Cov(vec
        // C_k)(I_{M_k} ⊗ U_k)ᵀ` (since `B_k = C_k U_kᵀ`) so the downstream band
        // code — which reads the `b·p + c` flat layout — is unchanged. On the
        // full-`B` path the block is already `(M_k·p)` and the lift is skipped.
        let frames_active = self.frames_active();
        let frame_projection = FrameProjection::new(self);
        let block_ranges = if frames_active {
            (0..self.k_atoms())
                .map(|k| frame_projection.atom_border_range(k))
                .collect::<Vec<_>>()
        } else {
            self.beta_block_offsets().to_vec()
        };
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let cov_block = cache
                .schur_inverse_block(block_ranges[k].clone())
                .map_err(|e| format!("assemble_shape_uncertainty: atom {k}: {e}"))?;
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            // Evenly-strided evaluation rows bound the band cost.
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let mut band_sd = Array2::<f64>::zeros((g, p));
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }

            let framed = frames_active && atom.decoder_frame.is_some();
            let dense_entries = (m * p).saturating_mul(m * p);
            let cov = if framed && dense_entries > SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES {
                // LLM-scale ambient `p`: the dense `(M_k·p)²` lift would be
                // gigabytes per atom and exists only to export the full
                // covariance. Compute the band variance EXACTLY from the
                // factored frame covariance instead: with `B_k = C_k·U_kᵀ`,
                //   Var_c(t) = (φ ⊗ u_c)ᵀ Cov(vec C_k) (φ ⊗ u_c)
                // which is the r×r quadratic form `u_cᵀ Y u_c` with
                //   Y = Σ_{b1,b2} φ[b1] φ[b2] Cov(C)[(b1,·),(b2,·)].
                let mut cov_c = cov_block;
                cov_c.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    let basis = atom.basis_values.row(row);
                    for c in 0..p {
                        let var = frame_projection.output_variance(k, cov_c.view(), basis, c);
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                None
            } else {
                // Lift the factored `(M_k·r_k)` coordinate covariance to the
                // full `(M_k·p)` decoder covariance through this atom's frame;
                // identity (a plain scaled copy) on the un-framed full-`B` path.
                let mut cov = if framed {
                    frame_projection.lift_block(k, cov_block.view())
                } else {
                    cov_block
                };
                cov.mapv_inplace(|v| v * dispersion);
                for (gi, &row) in eval_rows.iter().enumerate() {
                    // Var_c = Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]; the flat
                    // decoder index is basis·p + channel (row-major (M_k, p)).
                    for c in 0..p {
                        let var = frame_projection.full_output_variance(
                            k,
                            cov.view(),
                            atom.basis_values.row(row),
                            c,
                        );
                        band_sd[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
                Some(cov)
            };
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: cov,
                band_coords,
                band_mean,
                band_sd,
                band_sd_robust: None,
            });
        }
        Ok(SaeShapeUncertainty { dispersion, atoms })
    }

    /// #977 — complete the per-atom shape band for any atom the pre-search
    /// Schur factor could not cover (a structure-search-BORN atom, whose index
    /// is ≥ the seed `K` the Schur cache was assembled at), from that atom's OWN
    /// fitted penalized inner Hessian.
    ///
    /// The Schur path ([`Self::assemble_shape_uncertainty`]) reads the joint
    /// inverse-Hessian β-block per atom, but that factor is assembled ONCE before
    /// the structure search runs, so it is indexed by the SEED dictionary. A born
    /// atom therefore has no Schur block and would otherwise be reported with NO
    /// uncertainty band — a silent gap. This method closes it: every atom carries
    /// a band, none is reported without one.
    ///
    /// The principled per-atom band is the Laplace posterior of the atom's inner
    /// reconstruction smooth, which [`Self::set_atom_inner_fits`] already fits at
    /// the settled state for EVERY atom (born included). With the Gaussian-identity
    /// inner smooth, each output channel `c`'s decoder posterior is
    /// `Cov(β_{k,c}) = φ · H_k⁻¹`, where `H_k = Φ_kᵀ W_k Φ_k + S̃_k` is the atom's
    /// fitted penalized inner Hessian (`AtomInnerFit::penalized_hessian`). The
    /// ambient point `m_k(t) = Φ_k(t)·B_k` is linear in `B_k`, so its per-channel
    /// posterior variance is the closed form
    ///   `Var_c(t) = φ · Φ_k(t)ᵀ H_k⁻¹ Φ_k(t)`,
    /// which is the SAME for every channel `c` (the inner Hessian is shared across
    /// channels; the decoder differs only in the mean). The band is evaluated at
    /// the same evenly-strided on-atom coordinate subset the Schur path uses, so a
    /// born atom's band is reported exactly where its data lives.
    ///
    /// This is a strict completion: an atom whose band the Schur path already
    /// filled (a finite `band_sd`) is left untouched; only atoms with a missing
    /// entry (index past the assembled set) or an all-NaN band are filled. An
    /// all-NaN band arises either as the no-decoder-covariance fallback OR when
    /// the caller deliberately invalidated a stale PRE-search band via
    /// [`SaeShapeUncertainty::invalidate_bands_for_recompute`] after a structure
    /// move re-converged the dictionary (#1230); in both cases the band is
    /// recomputed here against the FINAL model. When a band is (re)filled the
    /// whole slot — `band_coords`, `band_mean`, AND `band_sd` — is rebuilt from
    /// the current fitted atom, so an atom whose coordinates / decoded mean / row
    /// count shifted under a structure-search refit gets a fully consistent band
    /// (never a stale-coordinate or shape-mismatched one). An atom whose inner fit
    /// is degenerate (`None` — no active rows / non-SPD inner Hessian) is left
    /// with its NaN band, faithfully reporting "unidentified" rather than
    /// fabricating a number. Requires [`Self::set_atom_inner_fits`] to have run;
    /// without it the completion is a no-op (the band stays as the Schur path left
    /// it).
    pub fn complete_born_atom_shape_bands(
        &self,
        unc: &mut SaeShapeUncertainty,
    ) -> Result<(), String> {
        let inner_fits = match &self.atom_inner_fits {
            Some(fits) => fits,
            // No inner fits harvested: nothing to complete from. Leave the bands
            // as the Schur path produced them.
            None => return Ok(()),
        };
        let p = self.output_dim();
        let dispersion = unc.dispersion;
        // Grow the per-atom band list to the post-search atom count so a born
        // atom (index past the Schur-assembled set) has a slot. New slots start
        // as NaN bands and are filled below from the inner fit.
        while unc.atoms.len() < self.k_atoms() {
            let k = unc.atoms.len();
            let atom = &self.atoms[k];
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let band_sd = Array2::<f64>::from_elem((g, p), f64::NAN);
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }
            unc.atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords,
                band_mean,
                band_sd,
                band_sd_robust: None,
            });
        }

        for (k, atom) in self.atoms.iter().enumerate() {
            let band = &mut unc.atoms[k];
            // Only complete a MISSING band: an atom the Schur path already filled
            // (a finite sd anywhere) keeps its joint-Hessian band untouched.
            let already_filled = band.band_sd.iter().any(|v| v.is_finite());
            if already_filled {
                continue;
            }
            let inner = match inner_fits.get(k).and_then(|f| f.as_ref()) {
                Some(f) => f,
                // Degenerate atom (no active rows / non-SPD inner Hessian): leave
                // the NaN band — honestly "unidentified", never a fabricated band.
                None => continue,
            };
            let m = atom.basis_size();
            if inner.penalized_hessian.dim() != (m, m) {
                return Err(format!(
                    "complete_born_atom_shape_bands: atom {k} inner Hessian {:?} != ({m}, {m})",
                    inner.penalized_hessian.dim()
                ));
            }
            // Factor the atom's own penalized inner Hessian H_k = ΦᵀWΦ + S̃_k. It
            // was checked SPD when the inner fit was built; re-factor here to solve
            // H_k⁻¹ Φ(t). A factorization failure (numerical drift since the inner
            // fit) leaves the NaN band rather than a fabricated number.
            let chol = match inner.penalized_hessian.cholesky(Side::Lower) {
                Ok(c) => c,
                Err(_) => continue,
            };
            // Evenly-strided on-atom rows, matched to the band the Schur path uses.
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            // Rebuild the ENTIRE band slot (coords / mean / sd) from the CURRENT
            // fitted atom rather than only overwriting `band_sd`. #1230 — a seed
            // atom whose pre-search band was invalidated for recompute (because
            // structure search re-converged the dictionary) may have changed its
            // coordinates, decoded mean, AND on-atom row count, so reusing the old
            // `band_coords` / `band_mean` (or indexing the old-shaped `band_sd`)
            // would mismatch the final model. A born atom whose slot was just
            // pushed with the right shape is rebuilt identically — same result.
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let mut band_sd = Array2::<f64>::from_elem((g, p), f64::NAN);
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
                // Φ_k(t) at this on-atom row.
                let phi_t = atom.basis_values.row(row).to_owned();
                // H_k⁻¹ Φ(t), then the quadratic form Φ(t)ᵀ H_k⁻¹ Φ(t).
                let solved = chol.solvevec(&phi_t);
                let quad = phi_t.dot(&solved).max(0.0);
                // Var_c(t) = φ · Φ(t)ᵀ H_k⁻¹ Φ(t) — identical across channels (the
                // inner Hessian is shared; the decoder differs only in the mean).
                let sd = (dispersion * quad).sqrt();
                for c in 0..p {
                    band_sd[[gi, c]] = sd;
                }
            }
            band.band_coords = band_coords;
            band.band_mean = band_mean;
            band.band_sd = band_sd;
        }
        Ok(())
    }

    pub(crate) fn shape_uncertainty_without_decoder_covariance(
        &self,
        dispersion: f64,
    ) -> SaeShapeUncertainty {
        let p = self.output_dim();
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let n_rows = atom.n_obs();
            let d = atom.latent_dim;
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let coords_mat = self.assignment.coords[k].as_matrix();
            let mut band_coords = Array2::<f64>::zeros((g, d));
            let mut band_mean = Array2::<f64>::zeros((g, p));
            let band_sd = Array2::<f64>::from_elem((g, p), f64::NAN);
            let mut decoded = vec![0.0_f64; p];
            for (gi, &row) in eval_rows.iter().enumerate() {
                for axis in 0..d {
                    band_coords[[gi, axis]] = coords_mat[[row, axis]];
                }
                atom.fill_decoded_row(row, &mut decoded);
                for c in 0..p {
                    band_mean[[gi, c]] = decoded[c];
                }
            }
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords,
                band_mean,
                band_sd,
                band_sd_robust: None,
            });
        }
        SaeShapeUncertainty { dispersion, atoms }
    }

    /// Sandwich (Godambe / robust) companion to
    /// [`Self::assemble_shape_uncertainty`]: computes the model-based bands
    /// exactly as before AND fills each atom's `band_sd_robust` with the
    /// misspecification-robust band, so BOTH are reported side by side
    /// (the [`RobustCovarianceMode`] contract).
    ///
    /// The model-based band uses `Cov(β) = φ̂ H⁻¹` (the inverse expected-Fisher
    /// covariance, valid only under a correctly specified reconstruction
    /// likelihood). The robust band replaces the within-channel covariance block
    /// `A_c⁻¹` with the Godambe sandwich `A_c⁻¹ J_cc A_c⁻¹`, where the "meat"
    /// `J_cc = (1/φ̂²) Σ_i r_{ic}² g_{ik} g_{ik}ᵀ` is the empirical outer product
    /// of the per-observation Gaussian scores and `g_{ik} = a_{ik} Φ_k(t_i)` is
    /// atom `k`'s gate-scaled effective design row (see [`super::sandwich`] for
    /// the full derivation). The two dispersion factors cancel, so the robust
    /// band is scale-free and lets the data's own per-observation residual
    /// energy — heteroskedastic tokens, LayerNorm structure, cross-channel
    /// template correlation — set the width. When the residuals actually satisfy
    /// the working likelihood the sandwich collapses back onto the model-based
    /// band, so any divergence is an honest, local misspecification diagnostic.
    ///
    /// `band_sd_robust` stays `None` for any atom whose dense within-channel
    /// bread was not materialized (the huge-`p` factored path where
    /// `decoder_covariance` is `None`) — an honest gap, never a fabricated band.
    pub fn assemble_shape_uncertainty_robust(
        &self,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        dispersion: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "assemble_shape_uncertainty_robust: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "assemble_shape_uncertainty_robust: dispersion must be finite and positive, \
                 got {dispersion}"
            ));
        }
        // Model-based bands + per-atom φ-scaled decoder covariance (the bread).
        let mut unc = self.assemble_shape_uncertainty(cache, dispersion)?;
        // Reconstruction residuals r_{ic} = z_{ic} − ẑ_{ic} (score sign z − ẑ).
        let fitted = self.try_fitted_with_rho(Some(rho), false)?;
        let mut residuals = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for c in 0..p {
                residuals[[row, c]] = target[[row, c]] - fitted[[row, c]];
            }
        }
        // Per-row gate mass a_{ik} for every atom, so g_{ik} = a_{ik} Φ_k(t_i).
        let mut gate = Array2::<f64>::zeros((n, self.k_atoms()));
        for row in 0..n {
            let a = self.assignment.try_assignments_row_for_rho(row, rho)?;
            for k in 0..self.k_atoms() {
                gate[[row, k]] = a[k];
            }
        }
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            // Robust band only where the dense within-channel bread exists.
            let cov = match unc.atoms[k].decoder_covariance.as_ref() {
                Some(cov) => cov.clone(),
                None => continue,
            };
            if cov.dim() != (m * p, m * p) {
                // Frame-lifted or full-B block must be (M·p)²; anything else is a
                // layout we cannot slice per channel — leave the honest None.
                continue;
            }
            // Effective (gate-scaled) design g_{ik} = a_{ik} Φ_k(t_i), (n × M).
            let mut design = Array2::<f64>::zeros((n, m));
            for row in 0..n {
                let a_k = gate[[row, k]];
                if a_k == 0.0 {
                    continue;
                }
                let basis = atom.basis_values.row(row);
                for b in 0..m {
                    design[[row, b]] = a_k * basis[b];
                }
            }
            // Within-channel score meat J_cc, one (M × M) per output channel.
            let meat = super::sandwich::gaussian_within_channel_meat(
                design.view(),
                residuals.view(),
                dispersion,
            )?;
            // Evaluation rows: reuse the exact strided grid the model-based band
            // used, so band_sd and band_sd_robust are aligned point-for-point.
            let n_rows = atom.n_obs();
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let mut band_sd_robust = Array2::<f64>::zeros((g, p));
            // Per-channel within-channel bread block A_c⁻¹ = cov[(b1,c),(b2,c)].
            let mut ok = true;
            for c in 0..p {
                let mut bread_c = Array2::<f64>::zeros((m, m));
                for b1 in 0..m {
                    for b2 in 0..m {
                        bread_c[[b1, b2]] = cov[[b1 * p + c, b2 * p + c]];
                    }
                }
                for (gi, &row) in eval_rows.iter().enumerate() {
                    let phi_t = atom.basis_values.row(row);
                    match super::sandwich::robust_channel_band_variance(
                        bread_c.view(),
                        meat[c].view(),
                        phi_t,
                    ) {
                        Ok(var) => band_sd_robust[[gi, c]] = var.sqrt(),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    break;
                }
            }
            if ok {
                unc.atoms[k].band_sd_robust = Some(band_sd_robust);
            }
        }
        Ok(unc)
    }

    /// Composite-likelihood model-selection charge for the decoder block,
    /// reported under BOTH the model-based and the CLIC (sandwich) accounting.
    ///
    /// The model-based effective dof `tr(F A⁻¹)` (the smoother/hat trace assumed
    /// by a well-specified-likelihood charge) is compared against the
    /// composite-likelihood `tr(J A⁻¹)`, which replaces the expected information
    /// `F` by the empirical score meat `J = Σ_i s_i s_iᵀ` (see [`super::sandwich`]).
    /// Both are summed over atoms and output channels from the same fitted
    /// residuals, so they sit on identical footing and coincide exactly when the
    /// reconstruction residuals satisfy the working Gaussian likelihood. A
    /// selection routine that prices a structure move should use `clic_dof` when
    /// it cannot assume iid Gaussian residuals, and can read
    /// [`CompositeLikelihoodCharge::misspecification_ratio`] to see the gap.
    ///
    /// Only atoms whose dense within-channel bread is materialized contribute
    /// (the huge-`p` factored path is skipped, mirroring the robust band); the
    /// returned charge is therefore over the decoder blocks that carry an
    /// exportable covariance.
    pub fn composite_likelihood_charge(
        &self,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        dispersion: f64,
    ) -> Result<CompositeLikelihoodCharge, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "composite_likelihood_charge: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "composite_likelihood_charge: dispersion must be finite and positive, \
                 got {dispersion}"
            ));
        }
        // Model-based bread (φ-scaled per-atom decoder covariance).
        let unc = self.assemble_shape_uncertainty(cache, dispersion)?;
        let fitted = self.try_fitted_with_rho(Some(rho), false)?;
        let mut residuals = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for c in 0..p {
                residuals[[row, c]] = target[[row, c]] - fitted[[row, c]];
            }
        }
        let mut gate = Array2::<f64>::zeros((n, self.k_atoms()));
        for row in 0..n {
            let a = self.assignment.try_assignments_row_for_rho(row, rho)?;
            for k in 0..self.k_atoms() {
                gate[[row, k]] = a[k];
            }
        }
        let mut model_based_dof = 0.0_f64;
        let mut clic_dof = 0.0_f64;
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let cov = match unc.atoms[k].decoder_covariance.as_ref() {
                Some(cov) if cov.dim() == (m * p, m * p) => cov,
                _ => continue,
            };
            let mut design = Array2::<f64>::zeros((n, m));
            for row in 0..n {
                let a_k = gate[[row, k]];
                if a_k == 0.0 {
                    continue;
                }
                let basis = atom.basis_values.row(row);
                for b in 0..m {
                    design[[row, b]] = a_k * basis[b];
                }
            }
            let meat = super::sandwich::gaussian_within_channel_meat(
                design.view(),
                residuals.view(),
                dispersion,
            )?;
            let expected =
                super::sandwich::gaussian_within_channel_expected_meat(design.view(), dispersion)?;
            for c in 0..p {
                let mut bread_c = Array2::<f64>::zeros((m, m));
                for b1 in 0..m {
                    for b2 in 0..m {
                        bread_c[[b1, b2]] = cov[[b1 * p + c, b2 * p + c]];
                    }
                }
                model_based_dof +=
                    super::sandwich::clic_effective_dof(bread_c.view(), expected.view())?;
                clic_dof += super::sandwich::clic_effective_dof(bread_c.view(), meat[c].view())?;
            }
        }
        Ok(CompositeLikelihoodCharge {
            model_based_dof,
            clic_dof,
        })
    }
}
