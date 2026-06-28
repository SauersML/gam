use super::*;

/// Cap on the number of coordinates at which a per-atom shape band is
/// materialized. The full per-atom decoder covariance is exact and exposed
/// regardless; this only bounds the cost of the convenience band, which is
/// evaluated at an evenly-strided subset of the atom's own on-atom coordinates.
pub const SHAPE_BAND_MAX_POINTS: usize = 512;

/// Entry budget for materializing one atom's dense `(M_k·p)²` decoder
/// covariance in the fit payload. Above it (LLM-scale ambient `p`) the band
/// quantities are computed exactly from the factored frame covariance and the
/// dense export is omitted (`decoder_covariance: None`) — the python reader
/// treats it as optional. 2^24 f64 entries = 128 MiB per atom.
pub const SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES: usize = 1 << 24;

/// Posterior uncertainty of one fitted atom's manifold shape.
///
/// Produced by [`SaeManifoldTerm::assemble_shape_uncertainty`]. The covariance
/// is the φ-scaled β-block of the joint inverse Hessian (coordinates
/// marginalized out); the band is its closed-form push-forward through the
/// linear basis→ambient map `m_k(t) = Φ_k(t)·B_k`.
#[derive(Debug, Clone)]
pub struct SaeAtomShapeUncertainty {
    /// φ-scaled posterior covariance of this atom's decoder coefficients,
    /// `Cov(β_k) = φ·S_β⁻¹[block_k]`, shape `(M_k·p, M_k·p)` in the decoder's
    /// row-major `(basis, channel)` flat layout (flat index `b·p + c`).
    ///
    /// `None` when materializing it would exceed
    /// [`SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES`] (LLM-scale ambient `p`: at
    /// `(M=8, p=2048)` the dense block is 2 GiB *per atom*, at
    /// `(M=16, p=5120)` ~50 GiB). The band quantities below are still exact
    /// in that case — they are computed directly from the factored
    /// `(M_k·r_k)²` frame covariance without ever lifting it.
    pub decoder_covariance: Option<Array2<f64>>,
    /// Coordinates at which the band is evaluated, shape `(G, d_k)`.
    pub band_coords: Array2<f64>,
    /// Fitted ambient point `m_k(t) = Φ_k(t)·B_k` at each band coordinate,
    /// shape `(G, p)`.
    pub band_mean: Array2<f64>,
    /// Posterior standard deviation of each ambient channel at each band
    /// coordinate, `sqrt(Var_c(t))` with
    /// `Var_c(t) = Σ_{b1,b2} Φ[b1] Φ[b2] Cov(β_k)[(b1,c),(b2,c)]`, shape
    /// `(G, p)`.
    pub band_sd: Array2<f64>,
}

/// Posterior shape uncertainty for a whole SAE-manifold fit: one band per atom
/// plus the shared Gaussian reconstruction dispersion `φ̂` used to scale every
/// covariance. See [`SaeManifoldTerm::assemble_shape_uncertainty`].
#[derive(Debug, Clone)]
pub struct SaeShapeUncertainty {
    /// Gaussian reconstruction scale `φ̂ = RSS / residual-dof`.
    pub dispersion: f64,
    /// One entry per atom, in atom order.
    pub atoms: Vec<SaeAtomShapeUncertainty>,
}

impl SaeShapeUncertainty {
    /// #1230 — invalidate every PRE-search seed band so it is recomputed from the
    /// FINAL post-structure-search model state.
    ///
    /// The production fit assembles these bands from the joint Hessian at the ρ
    /// the OUTER optimizer settled on, BEFORE evidence-guarded structure search.
    /// When a structure move lands (a certified birth/fission/fusion, or a demoted
    /// death), the warm refit re-converges the WHOLE dictionary at a new ρ, so the
    /// seed atoms' decoders / coordinates / inner curvature change and their
    /// pre-search joint-Hessian bands no longer describe the returned model. This
    /// resets each existing band's posterior `band_sd` to `NaN` and drops the
    /// stale dense `decoder_covariance`, so the subsequent
    /// [`SaeManifoldTerm::complete_born_atom_shape_bands`] pass — which fills every
    /// `NaN` band from each atom's OWN final penalized inner Hessian
    /// `H_k = Φ_kᵀ W_k Φ_k + S̃_k` harvested at the settled post-search state —
    /// recomputes the band for EVERY atom (seed and born) against the final model,
    /// not just the born atoms. A genuinely-degenerate atom keeps an honest `NaN`
    /// band rather than a stale fabricated one. `band_coords` / `band_mean` are
    /// re-derived directly from the final fitted atom by the completion pass, so
    /// they stay consistent too.
    ///
    /// No-op when the structure did not change (every atom keeps its exact
    /// joint-Hessian band, which is still valid and strictly higher quality than
    /// the per-atom Laplace approximation).
    pub fn invalidate_bands_for_recompute(&mut self) {
        for atom in &mut self.atoms {
            atom.decoder_covariance = None;
            atom.band_sd.fill(f64::NAN);
        }
    }
}

/// Posterior shape-uncertainty assembly for the fitted manifold term.
///
/// These methods produce and complete the per-atom [`SaeAtomShapeUncertainty`]
/// bands defined above; they live here with the types they populate rather
/// than in the term-construction module.
impl SaeManifoldTerm {
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
            });
        }
        SaeShapeUncertainty { dispersion, atoms }
    }
}
