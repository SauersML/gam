//! Reconstruction-dispersion and shape-uncertainty methods, split out of the
//! tail of `construction.rs` to keep that tracked file under the #780 10k-line
//! gate. Holds the contiguous trailing `impl SaeManifoldTerm` block:
//! `reconstruction_dispersion` (the Gaussian dispersion `φ̂` estimator),
//! `assemble_shape_uncertainty`, `complete_born_atom_shape_bands`, and
//! `shape_uncertainty_without_decoder_covariance`. All are reached bare by
//! callers through `use super::*`, so their visibility is unchanged.

use super::*;

fn persisted_atom_basis_values(
    kind: &SaeAtomBasisKind,
    coords: ArrayView2<'_, f64>,
    decoder_width: usize,
    latent_dim: usize,
    atom_idx: usize,
) -> Result<Array2<f64>, String> {
    match kind {
        SaeAtomBasisKind::Periodic => {
            if decoder_width == 0 || decoder_width % 2 == 0 {
                return Err(format!(
                    "reconstruct_persisted_atom_set: periodic atom {atom_idx} decoder width \
                     must be odd and positive; got {decoder_width}"
                ));
            }
            if coords.ncols() == 0 {
                return Err(format!(
                    "reconstruct_persisted_atom_set: periodic atom {atom_idx} needs at least \
                     one coordinate column"
                ));
            }
            if latent_dim != 1 {
                return Err(format!(
                    "reconstruct_persisted_atom_set: periodic atom {atom_idx} expects \
                     latent_dim=1, got {latent_dim}"
                ));
            }
            let evaluator = PeriodicHarmonicEvaluator::new(decoder_width)?;
            let (phi, _jet) = evaluator.evaluate(coords.slice(s![.., 0..1]))?;
            Ok(phi)
        }
        SaeAtomBasisKind::Sphere => {
            if decoder_width != 7 {
                return Err(format!(
                    "reconstruct_persisted_atom_set: sphere atom {atom_idx} decoder width \
                     must be 7, got {decoder_width}"
                ));
            }
            if latent_dim != 2 {
                return Err(format!(
                    "reconstruct_persisted_atom_set: sphere atom {atom_idx} expects \
                     latent_dim=2, got {latent_dim}"
                ));
            }
            let (phi, _jet) = SphereChartEvaluator.evaluate(coords)?;
            Ok(phi)
        }
        other => Err(format!(
            "reconstruct_persisted_atom_set: atom {atom_idx} basis {other:?} is not a \
             centers-free analytic persisted basis; rebuild a SaeManifoldTerm for this topology"
        )),
    }
}

/// Reconstruct a persisted SAE-manifold atom set from frozen coordinates,
/// assignment masses, and decoder blocks.
///
/// This is the stateless counterpart to [`SaeManifoldTerm::try_fitted`]: Python
/// artifacts that intentionally dropped the full term still carry enough
/// persisted atom state to materialize `Σ_k a_ik · Φ_k(t_ik)B_k`. Keeping the
/// basis evaluation, GEMM, and weighted atom sum here prevents the Python facade
/// from becoming a second decoder implementation.
pub fn reconstruct_persisted_atom_set(
    basis_kinds: &[SaeAtomBasisKind],
    atom_dims: &[usize],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    assignments: ArrayView2<'_, f64>,
    p_out: usize,
) -> Result<Array2<f64>, String> {
    let k_atoms = basis_kinds.len();
    if atom_dims.len() != k_atoms || decoder_blocks.len() != k_atoms || coords.len() != k_atoms {
        return Err(format!(
            "reconstruct_persisted_atom_set: metadata lengths must all equal K={k_atoms} \
             (atom_dims={}, decoder_blocks={}, coords={})",
            atom_dims.len(),
            decoder_blocks.len(),
            coords.len()
        ));
    }
    let n_rows = assignments.nrows();
    if assignments.ncols() != k_atoms {
        return Err(format!(
            "reconstruct_persisted_atom_set: assignments {:?} must have K={k_atoms} columns",
            assignments.dim()
        ));
    }
    if p_out == 0 {
        return Err("reconstruct_persisted_atom_set: p_out must be positive".to_string());
    }
    let mut out = Array2::<f64>::zeros((n_rows, p_out));
    for atom_idx in 0..k_atoms {
        let decoder = decoder_blocks[atom_idx];
        let (basis_width, decoder_p) = decoder.dim();
        if decoder_p != p_out {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} decoder output width \
                 {decoder_p} != p_out {p_out}"
            ));
        }
        let atom_coords = coords[atom_idx];
        if atom_coords.nrows() != n_rows {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} coords rows {} != {n_rows}",
                atom_coords.nrows()
            ));
        }
        let phi = persisted_atom_basis_values(
            &basis_kinds[atom_idx],
            atom_coords,
            basis_width,
            atom_dims[atom_idx],
            atom_idx,
        )?;
        if phi.dim() != (n_rows, basis_width) {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} basis {:?} != ({n_rows}, {basis_width})",
                phi.dim()
            ));
        }
        let decoded = phi.dot(&decoder);
        for row in 0..n_rows {
            let gate = assignments[[row, atom_idx]];
            if gate == 0.0 {
                continue;
            }
            for col in 0..p_out {
                out[[row, col]] += gate * decoded[[row, col]];
            }
        }
    }
    Ok(out)
}

impl SaeManifoldTerm {
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
    /// `residual` is the per-row reconstruction residual `f(θ̂) − y` (n×p) at the
    /// same state that produced `cache`. When supplied it engages the #2133 SURE
    /// within-basin second-order deflation correction
    /// ([`Self::coordinate_sure_deflation_correction`]) — the exact-Newton
    /// completion of the Gauss-Newton `coord_edf`, which removes the
    /// incidental-parameters under-dispersion of the per-row coordinate MAP.
    /// `None` reproduces the historical Gauss-Newton dispersion exactly (used by
    /// callers with no residual in hand — the correction is then simply absent).
    pub(crate) fn reconstruction_dispersion(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        residual: Option<ArrayView2<'_, f64>>,
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
        // #2133 — restore the second-order residual-curvature term the
        // Gauss-Newton `coord_edf` above drops, turning the per-row GN divergence
        // into the exact within-basin SURE divergence of the coordinate MAP. Pure
        // additive readout; only engaged when the caller supplies the residual.
        if let Some(residual) = residual {
            coord_edf = (coord_edf + self.coordinate_sure_deflation_correction(residual, rho)?)
                .clamp(0.0, n_scalar);
        }
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

    /// Recompute the JOINT inverse-Hessian shape bands at the CURRENT (final)
    /// term + ρ state — the same joint covariance
    /// [`Self::assemble_shape_uncertainty`] forms, but rebuilt AFTER a
    /// structure-changing or finalization move invalidated the pre-search Schur
    /// factor.
    ///
    /// [`super::SaeManifoldOuterObjective::decoder_shape_uncertainty`] reads the
    /// joint factor off the outer objective BEFORE `into_fitted` consumes it, so
    /// the bands it returns describe the PRE-search dictionary at the settled ρ.
    /// When evidence-guarded structure search grows / re-converges the whole
    /// dictionary (a certified birth / fission / fusion or a demoted death), or a
    /// finalization fallback swaps the settled basin / canonicalizes charts, that
    /// factor no longer describes the returned model. This rebuilds the undamped
    /// Direct joint-Hessian factor from THIS (final) term at `rho` — the exact
    /// factor the REML criterion forms at the inner optimum — and reads the
    /// per-atom covariance and bands off its Schur factor, scaling by the
    /// reconstruction dispersion `φ̂`. The result is the DOCUMENTED joint
    /// covariance: it carries the cross-atom covariance and the decoder-coordinate
    /// Schur couplings, and its per-channel band varies across output channels —
    /// unlike the per-atom inner-Hessian marginal
    /// [`Self::complete_born_atom_shape_bands`] falls back to. Every atom is
    /// covered, seed AND structure-search-born, because the factor is assembled at
    /// the final dictionary's `k_atoms()`.
    ///
    /// The term is already at its optimum, so the inner re-solve converges
    /// immediately. Mirrors `decoder_shape_uncertainty`'s admission fallback: when
    /// the streaming plan cannot admit the dense Direct factor (LLM-scale fits
    /// with no dense Schur), it returns
    /// [`Self::shape_uncertainty_without_decoder_covariance`] — honest NaN bands,
    /// never a fabricated number. Call before [`Self::into_fitted`] has run is not
    /// required; it takes the fitted `term`/`rho` directly.
    pub fn recompute_joint_shape_uncertainty(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        inner_max_iter: usize,
        learning_rate: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
    ) -> Result<SaeShapeUncertainty, String> {
        let plan = self.streaming_plan().admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !plan.direct_logdet_admitted() {
            // No dense Direct Schur factor at this scale: the joint covariance
            // cannot be materialized. Report the honest without-covariance bands
            // (NaN sd) rather than a per-atom stand-in dressed up as joint.
            let loss = self.loss(target, rho)?;
            let n_scalar = (self.n_obs().saturating_mul(self.output_dim())).max(1) as f64;
            let dispersion = (2.0 * loss.data_fit / n_scalar).max(f64::MIN_POSITIVE);
            return Ok(self.shape_uncertainty_without_decoder_covariance(dispersion));
        }
        let (_cost, loss, cache) = self.reml_criterion_with_cache(
            target,
            rho,
            registry,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        let residual = self.reconstruction_residual(target, rho)?;
        let dispersion =
            self.reconstruction_dispersion(&loss, &cache, rho, Some(residual.view()))?;
        self.assemble_shape_uncertainty(&cache, dispersion)
    }

    /// #977 — complete the per-atom shape band for any atom the joint Schur
    /// factor could not cover (a structure-search-BORN atom whose index is ≥ the
    /// seed `K` a pre-search cache was assembled at, or an atom whose joint block
    /// came back non-finite), from that atom's OWN fitted penalized inner Hessian.
    ///
    /// NOTE: the band this fills is a per-atom MARGINAL, NOT the joint covariance.
    /// It is `Var_c(t) = φ · Φ_k(t)ᵀ H_k⁻¹ Φ_k(t)` from the atom's own inner
    /// Hessian `H_k = Φ_kᵀ W_k Φ_k + S̃_k`, so it DROPS the cross-atom covariance
    /// and the decoder-coordinate Schur couplings the joint factor carries, and is
    /// identical across output channels (the inner Hessian is shared across
    /// channels; the decoder differs only in the mean). The production fit
    /// recomputes the JOINT bands via [`Self::recompute_joint_shape_uncertainty`]
    /// after a structure / finalization change, so this completion runs only as a
    /// backstop for atoms the joint factor genuinely left unidentified (all-NaN).
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

#[cfg(test)]
mod persisted_reconstruct_tests {
    use super::*;

    // Exercises `reconstruct_persisted_atom_set` (and, transitively,
    // `persisted_atom_basis_values`): a stateless K=1 periodic-atom round trip that
    // must equal `a_i · (Φ(t_i) · B)` computed directly from the same evaluator.
    #[test]
    fn reconstruct_persisted_periodic_atom_matches_direct_decode() {
        let n_rows = 4usize;
        let p_out = 2usize;
        let width = 3usize; // odd decoder width required for periodic
        let coords = Array2::from_shape_vec((n_rows, 1), vec![0.1, 0.7, 1.9, 2.8]).unwrap();
        let decoder =
            Array2::from_shape_vec((width, p_out), vec![0.5, -0.2, 0.3, 0.9, -0.4, 0.1]).unwrap();
        let assignments =
            Array2::from_shape_vec((n_rows, 1), vec![1.0, 0.5, 0.8, 0.2]).unwrap();

        let out = reconstruct_persisted_atom_set(
            &[SaeAtomBasisKind::Periodic],
            &[1usize],
            &[decoder.view()],
            &[coords.view()],
            assignments.view(),
            p_out,
        )
        .expect("reconstruct persisted periodic atom");
        assert_eq!(out.dim(), (n_rows, p_out));

        let evaluator = PeriodicHarmonicEvaluator::new(width).unwrap();
        let (phi, _jet) = evaluator.evaluate(coords.view()).unwrap();
        let decoded = phi.dot(&decoder);
        for i in 0..n_rows {
            for j in 0..p_out {
                let expected = assignments[[i, 0]] * decoded[[i, j]];
                assert!(
                    (out[[i, j]] - expected).abs() < 1.0e-9,
                    "row {i} col {j}: got {} expected {expected}",
                    out[[i, j]]
                );
            }
        }
    }
}
