//! Reconstruction-dispersion and shape-uncertainty methods, split out of the
//! tail of `construction.rs` to keep that tracked file under the #780 10k-line
//! gate. Holds the contiguous trailing `impl SaeManifoldTerm` block:
//! `reconstruction_dispersion` (the Gaussian dispersion `φ̂` estimator),
//! `assemble_shape_uncertainty`, `recompute_joint_shape_uncertainty`, and the
//! explicit streaming-unavailable shape report. All are reached bare by
//! callers through `use super::*`, so their visibility is unchanged.

use super::*;

/// Certify the ARD identity
/// `edf = n_active - alpha * tr(H^-1)` against its exact `[0, n_active]`
/// interval.  A tiny excursion at the forward-error scale of the accumulated
/// trace is snapped to the boundary; a material excursion is a failed trace
/// certificate, not an EDF that may be silently projected into another model.
pub(super) fn certified_ard_axis_edf(
    n_active: f64,
    alpha: f64,
    inverse_trace: f64,
    atom: usize,
    axis: usize,
) -> Result<f64, String> {
    if !(n_active.is_finite() && n_active >= 0.0) {
        return Err(format!(
            "reconstruction_dispersion: ARD active count at atom {atom}, axis {axis} \
             must be finite and non-negative; got {n_active}"
        ));
    }
    if !(alpha.is_finite() && alpha > 0.0 && inverse_trace.is_finite()) {
        return Err(format!(
            "reconstruction_dispersion: ARD precision/trace at atom {atom}, axis {axis} \
             must be finite with positive precision; got alpha={alpha}, trace={inverse_trace}"
        ));
    }
    let shrinkage = alpha * inverse_trace;
    let raw = n_active - shrinkage;
    if !shrinkage.is_finite() || !raw.is_finite() {
        return Err(format!(
            "reconstruction_dispersion: ARD EDF arithmetic is unrepresentable at atom \
             {atom}, axis {axis} (n_active={n_active}, alpha={alpha}, trace={inverse_trace})"
        ));
    }
    let tolerance = 64.0
        * n_active.max(1.0)
        * f64::EPSILON
        * (n_active.abs() + shrinkage.abs()).max(f64::MIN_POSITIVE);
    if raw < -tolerance || raw > n_active + tolerance {
        return Err(format!(
            "reconstruction_dispersion: ARD EDF at atom {atom}, axis {axis} is \
             {raw:.6e}, outside certified [0, {n_active}] (roundoff tolerance \
             {tolerance:.6e}; alpha={alpha:.6e}, trace={inverse_trace:.6e})"
        ));
    }
    Ok(raw.clamp(0.0, n_active))
}

#[cfg(test)]
mod ard_edf_certificate_tests {
    use super::certified_ard_axis_edf;

    #[test]
    fn snaps_only_trace_roundoff_at_the_ard_edf_faces() {
        let n = 8.0;
        let tiny = 8.0 * f64::EPSILON;
        assert_eq!(certified_ard_axis_edf(n, 1.0, -tiny, 0, 0).unwrap(), n);
        assert_eq!(certified_ard_axis_edf(n, 1.0, n + tiny, 0, 0).unwrap(), 0.0);
    }

    #[test]
    fn refuses_material_or_nonfinite_ard_edf_excursions() {
        for trace in [-1.0e-8, 8.0 + 1.0e-8, f64::NAN, f64::INFINITY] {
            assert!(certified_ard_axis_edf(8.0, 1.0, trace, 2, 3).is_err());
        }
    }
}

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

/// Stateless on-manifold STEER of a persisted atom set (gam#2234): the ambient
/// steering DELTA `a_{ik}·(Φ_k(t_i ⊕ δ) − Φ_k(t_i))·B_k` for the single atom
/// `steer_atom`, one row per input row, shape `(n_rows, p_out)`. The caller adds
/// it to the ambient activation `x`.
///
/// This is the stateless counterpart of [`SaeManifoldTerm::steer_rows`] for the
/// Python facade's persisted-artifact path (E1), single-sourced against the SAME
/// [`persisted_atom_basis_values`] evaluator as [`reconstruct_persisted_atom_set`]
/// so the facade never becomes a second decoder. The group action `⊕` is the
/// atom's own [`LatentManifold::retract`] (Circle phase add modulo period,
/// Euclidean translate, product blockwise), derived from the persisted basis kind
/// — never re-implemented modular arithmetic. The persisted decoder folds the
/// atom magnitude in its coefficients (as in `reconstruct_persisted_atom_set`).
/// Gates are read from the persisted `assignments` and left untouched by the steer.
pub fn steer_persisted_atom_set(
    basis_kinds: &[SaeAtomBasisKind],
    atom_dims: &[usize],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    assignments: ArrayView2<'_, f64>,
    p_out: usize,
    steer_atom: usize,
    delta: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let k_atoms = basis_kinds.len();
    if atom_dims.len() != k_atoms || decoder_blocks.len() != k_atoms || coords.len() != k_atoms {
        return Err(format!(
            "steer_persisted_atom_set: metadata lengths must all equal K={k_atoms} \
             (atom_dims={}, decoder_blocks={}, coords={})",
            atom_dims.len(),
            decoder_blocks.len(),
            coords.len()
        ));
    }
    if steer_atom >= k_atoms {
        return Err(format!(
            "steer_persisted_atom_set: steer_atom {steer_atom} out of range (K={k_atoms})"
        ));
    }
    let n_rows = assignments.nrows();
    if assignments.ncols() != k_atoms {
        return Err(format!(
            "steer_persisted_atom_set: assignments {:?} must have K={k_atoms} columns",
            assignments.dim()
        ));
    }
    if p_out == 0 {
        return Err("steer_persisted_atom_set: p_out must be positive".to_string());
    }
    let d = atom_dims[steer_atom];
    if delta.len() != d {
        return Err(format!(
            "steer_persisted_atom_set: delta length {} != atom {steer_atom} latent_dim {d}",
            delta.len()
        ));
    }
    let decoder = decoder_blocks[steer_atom];
    let (basis_width, decoder_p) = decoder.dim();
    if decoder_p != p_out {
        return Err(format!(
            "steer_persisted_atom_set: atom {steer_atom} decoder output width {decoder_p} != \
             p_out {p_out}"
        ));
    }
    let atom_coords = coords[steer_atom];
    if atom_coords.nrows() != n_rows {
        return Err(format!(
            "steer_persisted_atom_set: atom {steer_atom} coords rows {} != {n_rows}",
            atom_coords.nrows()
        ));
    }
    if atom_coords.ncols() != d {
        return Err(format!(
            "steer_persisted_atom_set: atom {steer_atom} coords cols {} != latent_dim {d}",
            atom_coords.ncols()
        ));
    }
    // The group action `t ⊕ δ` via the atom's own manifold retraction.
    let manifold = basis_kinds[steer_atom].latent_manifold(d);
    let mut steered = Array2::<f64>::zeros((n_rows, d));
    for row in 0..n_rows {
        let moved = manifold.retract(atom_coords.row(row), delta);
        for a in 0..d {
            steered[[row, a]] = moved[a];
        }
    }
    let phi_base = persisted_atom_basis_values(
        &basis_kinds[steer_atom],
        atom_coords,
        basis_width,
        d,
        steer_atom,
    )?;
    let phi_steer = persisted_atom_basis_values(
        &basis_kinds[steer_atom],
        steered.view(),
        basis_width,
        d,
        steer_atom,
    )?;
    let base_dec = phi_base.dot(&decoder);
    let steer_dec = phi_steer.dot(&decoder);
    let mut out = Array2::<f64>::zeros((n_rows, p_out));
    for row in 0..n_rows {
        let gate = assignments[[row, steer_atom]];
        if gate == 0.0 {
            continue;
        }
        for col in 0..p_out {
            out[[row, col]] = gate * (steer_dec[[row, col]] - base_dec[[row, col]]);
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
        self.assignment.validate_rho_domain(rho)?;
        let n = self.n_obs();
        let p = self.output_dim();
        // Design-honesty weights are normalized to mean one, so they redistribute
        // residual mass without changing the scalar observation count.
        let n_scalar = (n * p) as f64;
        // FRAME CONSISTENCY (#2228/#2258 tier-0 root cause): under an active
        // WHITENING row metric the likelihood's `loss.data_fit` is the
        // WHITENED residual energy — ≈ n·p BY CONSTRUCTION (whitening
        // normalizes residuals to unit scale) — so a φ̂ built from it prices
        // the noise floor at ~n·p/resid_dof ≈ 2 REGARDLESS of the actual fit
        // quality. Every consumer of this dispersion lives in the RAW output
        // frame: the rank-charge MP edge compares against the unwhitened
        // reconstruction Gram (measured veto: R=2.16 vs top signal 1.01 on a
        // fitted EV=0.998 circle → rank_eff=0 → categorical +∞ → 'infeasible
        // at the requested rho' for every structured pass), and the shape
        // bands are φ-scaled output-frame covariances. Price φ from the RAW
        // residual whenever the caller supplied it and the metric whitens.
        let metric_whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let rss = if metric_whitens && residual.is_some() {
            residual
                .as_ref()
                .map(|res| res.iter().map(|value| value * value).sum::<f64>())
                .unwrap_or(2.0 * loss.data_fit)
        } else {
            2.0 * loss.data_fit
        };
        let smooth_edf: f64 = self
            .decoder_smoothness_effective_dof_per_atom(cache, &rho.lambda_smooth_vec()?)
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
        let ard_precisions = rho.ard_precisions()?;
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
                let alpha = ard_precisions[k][j];
                let edf_kj = certified_ard_axis_edf(n_active_k, alpha, traces[k][j], k, j)?;
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
            // #2133 — the basin-SELECTION (search) deflation dof: the boundary
            // Stein term the within-basin correction above omits. The per-row charge
            // depends on σ̂ = √φ̂, so seed it with the within-basin-corrected but
            // search-UNcorrected φ̂ and take ONE monotone fixed-point pass (the charge
            // is decreasing in σ̂ through the margin z, so one pass contracts). It is
            // identically 0 for single-basin / hard-frozen / genuinely-soft rows, so
            // w=None + non-selecting fits are bit-for-bit today's φ̂.
            let phi_seed = rss / (n_scalar - beta_edf - coord_edf).max(1.0);
            let df_search = self.basin_selection_deflation_correction(residual, phi_seed)?;
            coord_edf = (coord_edf + df_search).clamp(0.0, n_scalar);
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
                band_coords: Some(band_coords),
                band_mean: Some(band_mean),
                band_sd: Some(band_sd),
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
    /// factor the penalized quasi-Laplace criterion forms at the inner optimum — and reads the
    /// per-atom covariance and bands off its Schur factor, scaling by the
    /// reconstruction dispersion `φ̂`. The result is the DOCUMENTED joint
    /// covariance: it carries the cross-atom covariance and the decoder-coordinate
    /// Schur couplings, and its per-channel band varies across output channels.
    /// Every atom is covered because the factor is assembled at the final
    /// dictionary's `k_atoms()`.
    ///
    /// The term is already at its optimum, so the inner re-solve converges
    /// immediately. When the streaming plan cannot expose the exact Direct
    /// factor, the returned atom entries carry explicit `None` bands. Call before
    /// [`Self::into_fitted`] has run is not required; it takes the fitted
    /// `term`/`rho` directly.
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
            // No exact Direct Schur factor at this scale: report explicit
            // unavailability rather than substituting a different covariance.
            let loss = self.loss(target, rho)?;
            let n_scalar = (self.n_obs().saturating_mul(self.output_dim())).max(1) as f64;
            let dispersion = (2.0 * loss.data_fit / n_scalar).max(f64::MIN_POSITIVE);
            return Ok(self.unavailable_shape_uncertainty(dispersion));
        }
        let (_cost, loss, cache) = self.penalized_quasi_laplace_criterion_with_cache(
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

    /// Explicitly unavailable joint shape uncertainty for a streaming fit whose
    /// execution plan cannot expose the exact joint Schur factor.
    pub(crate) fn unavailable_shape_uncertainty(&self, dispersion: f64) -> SaeShapeUncertainty {
        let atoms = self
            .atoms
            .iter()
            .map(|_| SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords: None,
                band_mean: None,
                band_sd: None,
                band_sd_robust: None,
            })
            .collect();
        SaeShapeUncertainty { dispersion, atoms }
    }

    /// Joint empirical-score sandwich in the cache's complete decoder-border
    /// coordinates, plus `tr(J A^-1)` for the complete fitted state.
    ///
    /// Each observation contributes one score vector over its row-local routing
    /// and coordinate variables and the shared decoder border.  Solving the
    /// exact stationarity Jacobian against that complete score before extracting
    /// the beta influence is what retains cross-atom and nuisance effects:
    /// `(A^-1 J A^-1)_beta,beta = sum_i u_i,beta u_i,beta'`,
    /// `u_i = A^-1 s_i`.  The score is built from the same compact support,
    /// physical decoder Jacobian, row weights, and row metric as the fitted
    /// objective. Its single vector per row also retains every cross-output
    /// residual product in the outer product.
    fn joint_score_sandwich(
        &self,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        dispersion: f64,
    ) -> Result<(Array2<f64>, f64), String> {
        self.assignment.validate_rho_domain(rho)?;
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "joint_score_sandwich: target {:?} != ({n}, {p})",
                target.dim()
            ));
        }
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "joint_score_sandwich: dispersion must be finite and positive, got {dispersion}"
            ));
        }
        let total_t = cache.delta_t_len();
        let beta_dim = cache.k;
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let b_solver = self.outer_gradient_arrow_solver(cache, &rho.lambda_smooth_vec()?)?;
        let fitted_full = self.try_fitted_with_rho(Some(rho), false)?;
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let row_weights = self.row_loss_weights.as_deref();
        let mut beta_cov = Array2::<f64>::zeros((beta_dim, beta_dim));
        let mut joint_clic_dof = 0.0_f64;
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut decoded = vec![0.0_f64; p];
        let mut residual = Array1::<f64>::zeros(p);
        let mut rhs_t = Array1::<f64>::zeros(total_t);
        let mut rhs_beta = Array1::<f64>::zeros(beta_dim);

        for row in 0..n {
            self.assignment.try_assignments_row_into(
                row,
                assignments.as_slice_mut().expect("contiguous assignments"),
            )?;
            for out_col in 0..p {
                residual[out_col] = target[[row, out_col]] - fitted_full[[row, out_col]];
            }
            // `try_fitted_with_rho` is the full dictionary reconstruction.  The
            // compact objective treats dropped atoms as identically zero, so add
            // those contributions back to the residual to recover
            // target - sum_{k in A_i} a_ik g_ik.
            if let Some(layout) = self.last_row_layout.as_ref() {
                let active = &layout.active_atoms[row];
                for atom_idx in 0..self.k_atoms() {
                    if active.binary_search(&atom_idx).is_ok() {
                        continue;
                    }
                    self.atoms[atom_idx].fill_decoded_row(row, &mut decoded);
                    let a_k = assignments[atom_idx];
                    for out_col in 0..p {
                        residual[out_col] += a_k * decoded[out_col];
                    }
                }
            }
            let metric_residual = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, residual.view()),
                _ => residual.to_vec(),
            };
            let sqrt_weight = row_weights.map_or(1.0, |weights| weights[row].sqrt());
            let error_metric: Vec<f64> = metric_residual
                .into_iter()
                .map(|value| sqrt_weight * value)
                .collect();
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let jets =
                self.row_jets_for_logdet(row, vars, assignments.view(), &second_jets, &border)?;
            rhs_t.fill(0.0);
            rhs_beta.fill(0.0);
            let base = cache.row_offsets[row];
            for local in 0..jets.vars.len() {
                rhs_t[base + local] = sae_dot(jets.first(local), &error_metric);
            }
            for (beta_pos, channel) in border.iter().enumerate() {
                rhs_beta[channel.index] += sae_dot(jets.beta(beta_pos), &error_metric);
            }
            let rhs = SaeArrowVector {
                t: rhs_t.clone(),
                beta: rhs_beta.clone(),
            };
            let influence = self.solve_exact_stationarity(rho, target, cache, &b_solver, &rhs)?;
            for a in 0..beta_dim {
                let ua = influence.beta[a];
                for b in 0..beta_dim {
                    beta_cov[[a, b]] += ua * influence.beta[b];
                }
            }
            joint_clic_dof +=
                (rhs_t.dot(&influence.t) + rhs_beta.dot(&influence.beta)) / dispersion;
        }
        if beta_cov.iter().any(|value| !value.is_finite())
            || !(joint_clic_dof.is_finite() && joint_clic_dof >= 0.0)
        {
            return Err("joint_score_sandwich: non-finite or negative joint result".to_string());
        }
        Ok((beta_cov, joint_clic_dof))
    }

    /// Sandwich (Godambe / robust) companion to
    /// [`Self::assemble_shape_uncertainty`]: computes the model-based bands
    /// exactly as before AND fills each atom's `band_sd_robust` with the
    /// misspecification-robust band, so both are reported side by side.
    ///
    /// The robust block is extracted only after forming the complete joint
    /// sandwich.  Thus a reported atom band retains cross-atom, routing,
    /// coordinate, frame, and cross-output effects; factored decoders are pushed
    /// forward directly without materializing an ambient covariance.
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
        // Model-based bands remain reported alongside the robust result.
        let mut unc = self.assemble_shape_uncertainty(cache, dispersion)?;
        let (joint_beta_cov, _joint_clic_dof) =
            self.joint_score_sandwich(cache, rho, target, dispersion)?;
        let frames_active = self.frames_active();
        let frame_projection = FrameProjection::new(self);
        let block_ranges = if frames_active {
            (0..self.k_atoms())
                .map(|k| frame_projection.atom_border_range(k))
                .collect::<Vec<_>>()
        } else {
            self.beta_block_offsets().to_vec()
        };
        for (k, atom) in self.atoms.iter().enumerate() {
            let m = atom.basis_size();
            let range = block_ranges[k].clone();
            let block_dim = range.end - range.start;
            let mut raw_block = Array2::<f64>::zeros((block_dim, block_dim));
            for i in 0..block_dim {
                for j in 0..block_dim {
                    raw_block[[i, j]] = joint_beta_cov[[range.start + i, range.start + j]];
                }
            }
            let n_rows = atom.n_obs();
            let stride = n_rows.div_ceil(SHAPE_BAND_MAX_POINTS).max(1);
            let eval_rows: Vec<usize> = (0..n_rows).step_by(stride).collect();
            let g = eval_rows.len();
            let mut band_sd_robust = Array2::<f64>::zeros((g, p));
            let framed = frames_active && atom.decoder_frame.is_some();
            if framed {
                for (gi, &row) in eval_rows.iter().enumerate() {
                    let basis = atom.basis_values.row(row);
                    for c in 0..p {
                        let var = frame_projection.output_variance(k, raw_block.view(), basis, c);
                        band_sd_robust[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
            } else {
                if raw_block.dim() != (m * p, m * p) {
                    return Err(format!(
                        "assemble_shape_uncertainty_robust: atom {k} joint block {:?} != ({},{})",
                        raw_block.dim(),
                        m * p,
                        m * p
                    ));
                }
                for (gi, &row) in eval_rows.iter().enumerate() {
                    for c in 0..p {
                        let var = frame_projection.full_output_variance(
                            k,
                            raw_block.view(),
                            atom.basis_values.row(row),
                            c,
                        );
                        band_sd_robust[[gi, c]] = var.max(0.0).sqrt();
                    }
                }
            }
            unc.atoms[k].band_sd_robust = Some(band_sd_robust);
        }
        Ok(unc)
    }

    /// Composite-likelihood model-selection charge
    /// `tr(J A^-1)` for the complete joint fitted state.  Scores, sensitivity,
    /// compact support, metric, weights, amplitudes, and nuisance couplings are
    /// identical to [`Self::joint_score_sandwich`].
    pub fn composite_likelihood_charge(
        &self,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        dispersion: f64,
    ) -> Result<CompositeLikelihoodCharge, String> {
        let (_joint_beta_cov, joint_clic_dof) =
            self.joint_score_sandwich(cache, rho, target, dispersion)?;
        Ok(CompositeLikelihoodCharge { joint_clic_dof })
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
        let assignments = Array2::from_shape_vec((n_rows, 1), vec![1.0, 0.5, 0.8, 0.2]).unwrap();

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
