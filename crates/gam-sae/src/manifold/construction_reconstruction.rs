//! Reconstruction-dispersion and shape-uncertainty methods, split out of the
//! tail of `construction.rs` to keep that tracked file under the #780 10k-line
//! gate. Holds the contiguous trailing `impl SaeManifoldTerm` block:
//! `reconstruction_dispersion` (the Gaussian dispersion `φ̂` estimator),
//! `assemble_shape_uncertainty`, `recompute_joint_shape_uncertainty`, and the
//! explicit streaming-unavailable shape report. All are reached bare by
//! callers through `use super::*`, so their visibility is unchanged.

use super::*;

/// Reconstruct a persisted SAE-manifold atom set from frozen coordinates,
/// assignment masses, and decoder blocks.
///
/// This is the stateless counterpart to [`SaeManifoldTerm::try_fitted`]: Python
/// artifacts that intentionally dropped the full term still carry enough
/// persisted atom state to materialize `Σ_k a_ik · Φ_k(t_ik)B_k`. Keeping the
/// basis evaluation, GEMM, and weighted atom sum here prevents the Python facade
/// from becoming a second decoder implementation.
pub fn reconstruct_persisted_atom_set(
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    assignments: ArrayView2<'_, f64>,
    p_out: usize,
) -> Result<Array2<f64>, String> {
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms || coords.len() != k_atoms {
        return Err(format!(
            "reconstruct_persisted_atom_set: decoder and coordinate counts must equal \
             geometry-plan count K={k_atoms} (decoder_blocks={}, coords={})",
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
        let plan = &geometry_plans[atom_idx];
        let basis_width = plan.basis_size()?;
        let decoder = decoder_blocks[atom_idx];
        if decoder.dim() != (basis_width, p_out) {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} decoder shape {:?} must \
                 equal plan-derived ({basis_width}, {p_out})",
                decoder.dim()
            ));
        }
        let atom_coords = coords[atom_idx];
        if atom_coords.nrows() != n_rows {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} coords rows {} != {n_rows}",
                atom_coords.nrows()
            ));
        }
        if atom_coords.ncols() != plan.latent_dim() {
            return Err(format!(
                "reconstruct_persisted_atom_set: atom {atom_idx} coordinate width {} must \
                 equal plan latent_dim {}",
                atom_coords.ncols(),
                plan.latent_dim()
            ));
        }
        let (phi, _) = plan.build_evaluator()?.evaluate(atom_coords)?;
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
    /// Residual energies in the two frames of [`SaeReconstructionDispersion`]:
    /// `(raw_rss, likelihood_rss, likelihood_frame)`.
    ///
    /// `likelihood_rss = 2·data_fit` is the energy the likelihood sums, whitened
    /// when the metric whitens. Under a whitening metric the raw energy is not
    /// recoverable from the loss, so the raw residual is then required. A row with
    /// zero design weight is excluded from estimation and carries no energy in
    /// either frame.
    fn reconstruction_residual_energies(
        &self,
        loss: &SaeManifoldLoss,
        residual: Option<ArrayView2<'_, f64>>,
    ) -> Result<(f64, f64, SaeLikelihoodFrame), String> {
        if let Some(residual) = residual.as_ref() {
            if residual.dim() != (self.n_obs(), self.output_dim()) {
                return Err(format!(
                    "reconstruction residual shape {:?} does not match ({}, {})",
                    residual.dim(),
                    self.n_obs(),
                    self.output_dim(),
                ));
            }
            if residual.iter().any(|value| !value.is_finite()) {
                return Err("reconstruction residual must be finite".to_string());
            }
        }
        let likelihood_rss = 2.0 * loss.data_fit;
        let (raw_rss, frame) = match self
            .row_metric
            .as_ref()
            .filter(|metric| metric.whitens_likelihood())
        {
            Some(metric) => {
                let residual = residual.as_ref().ok_or_else(|| {
                    "the raw output noise variance under a whitening row metric needs the raw \
                     reconstruction residual"
                        .to_string()
                })?;
                let weights = self.row_loss_weights.as_deref();
                let raw_rss = residual
                    .outer_iter()
                    .enumerate()
                    .filter(|(row, _)| weights.is_none_or(|weights| weights[*row] > 0.0))
                    .map(|(_, values)| values.iter().map(|value| value * value).sum::<f64>())
                    .sum::<f64>();
                (
                    raw_rss,
                    SaeLikelihoodFrame::Whitened {
                        metric_rank: metric.metric_rank(),
                    },
                )
            }
            None => (likelihood_rss, SaeLikelihoodFrame::RawOutput),
        };
        for (label, rss) in [("raw", raw_rss), ("likelihood", likelihood_rss)] {
            if !(rss.is_finite() && rss >= 0.0) {
                return Err(format!(
                    "{label} reconstruction residual sum of squares must be finite and \
                     non-negative; got {rss}"
                ));
            }
        }
        Ok((raw_rss, likelihood_rss, frame))
    }

    /// Both noise scales with no effective-dof correction, for execution plans
    /// that expose no Schur factor to price the EDF. An exact fit reports zero.
    pub(crate) fn unfactored_reconstruction_dispersion(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<SaeReconstructionDispersion, String> {
        let loss = self.loss(target, rho)?;
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let residual = if whitens {
            Some(self.reconstruction_residual(target, rho)?)
        } else {
            None
        };
        let (raw_rss, likelihood_rss, likelihood_frame) =
            self.reconstruction_residual_energies(&loss, residual.as_ref().map(|r| r.view()))?;
        let (likelihood_scalars, raw_scalars) = self.fitted_response_scalar_counts()?;
        Ok(SaeReconstructionDispersion {
            raw_output_noise_variance: raw_rss / raw_scalars.max(1.0),
            likelihood_dispersion: likelihood_rss / likelihood_scalars.max(1.0),
            likelihood_frame,
            selection_conditioning: SaeSelectionConditioning::ConditionalOnFittedRouting,
        })
    }

    /// The two Gaussian reconstruction noise scales of
    /// [`SaeReconstructionDispersion`], each the root of its frame's scale equation.
    ///
    /// * `likelihood_dispersion` is the dispersion of the likelihood the fit
    ///   minimizes, in its own frame. It turns the metric-weighted inverse
    ///   information into a posterior covariance, the `Vb = φ·H⁻¹` convention the
    ///   main GAM inference path uses. Under a whitening metric the information
    ///   already carries the metric's variance scale, so this multiplier is
    ///   dimensionless and counts the whitened observations.
    /// * `raw_output_noise_variance` is the raw output-frame noise per scalar,
    ///   whatever the metric. Under a whitening metric `data_fit` is ≈ n·p by
    ///   construction, so the Marchenko–Pastur rank edge, which compares against
    ///   the unwhitened decoder Gram, must read this one (#2228/#2258: the whitened
    ///   value vetoed a fitted EV = 0.998 circle as rank zero). On the isotropic
    ///   frame the two coincide exactly.
    ///
    /// `residual` is the per-row reconstruction residual `f(θ̂) − y` (n×p) at the
    /// same state that produced `cache`.
    ///
    /// # The scale equation (#2933 F40)
    ///
    /// The noise model is the likelihood's own: `yᵢ = fᵢ(θ) + εᵢ` with
    /// `Cov(εᵢ) = φ·(wᵢMᵢ)⁻¹` on the likelihood frame, and `Cov(εᵢ) = φ·I` on the raw
    /// output frame. Linearize the fit about its converged inner state,
    /// `f̂ ≈ f̂(μ) + R·(y − μ)`, with `R = ∂f̂/∂y` taken through the exact observed
    /// information of the logits, coordinates and decoder border together
    /// ([`Self::fitted_response_divergence`], #2933 F36). Then, in the frame's norm
    /// over its `N` scalars of positive weight,
    ///
    /// ```text
    ///   E‖y − f̂‖² = ‖(I − R)μ‖² + φ·ν,   ν = ‖I − R‖²_F = N − 2 tr R + ‖R‖²_F,
    /// ```
    ///
    /// and each scale is `φ̂ = RSS/ν`. It is unbiased when the fitted response
    /// reproduces the mean, `(I − R)μ = 0`, for example a mean in the unpenalized
    /// space of every prior. Otherwise the bias `‖(I − R)μ‖²/ν` is non-negative,
    /// so the scale errs toward wider bands and a higher rank edge. The historical
    /// `RSS/(N − tr R)` is the special case `R² = R` of a projection. For a
    /// shrinking smoother it is biased low: with `R = ½I` and `μ = 0` its
    /// expectation is `½φ`. No floor is put on `ν`. A response that reproduces
    /// every observation has `ν = 0`, leaves no residual to estimate a scale from,
    /// and is refused.
    ///
    /// Learned decoder frames are estimated, so wherever every framed decoder has
    /// its frame's rank the divergence and both residual dofs integrate them
    /// ([`SaeFrameConditioning::MarginalOverLearnedFrames`], #2933 F39), densely or
    /// by output-space probes as the host's memory admits. Where a frame is rank
    /// deficient, or the host cannot hold the unframed evidence factor either route
    /// reads, the response holds the frames at their fitted orientation, and each
    /// frame's `r·(p − r)` unpenalized tangent dimensions are charged as fully
    /// determined response directions: `tr R` and `‖R‖²_F` each gain that count and
    /// `ν` loses it. When the frame block carries Gauss–Newton curvature and no
    /// prior, that count bounds the frame-coupled divergence from above, which makes
    /// that scale conservative.
    ///
    /// # Selection is conditioned on, not charged
    ///
    /// Both scales are conditional on the fitted routing. They hold the selected
    /// TopK support or the basin the inner solve converged to fixed, as frozen
    /// routing already does, and add no search degrees of freedom for the
    /// selection itself (#2933 F37). An observed-margin boundary charge used to be
    /// added here, but no single-draw statistic estimates that boundary term. For
    /// fixed candidates `±a` and `y ~ N(μ, σ²)`, the selection `ŷ = a·sign(y)` has
    /// `df_search = (2a/σ)·φ(μ/σ)`, which is a function of the unknown mean. Any
    /// statistic has `E T(y) = (T ∗ N(0, σ²))(μ)`, and `φ(·/σ)` is itself an
    /// `N(0, σ²)` kernel, so an unbiased `T` would have to be a point mass. The
    /// removed plug-in `(2a/σ)·φ(y/σ)` has expectation `(2a/σ)·φ(μ/(√2σ))/√2`. That
    /// is `1/√2` of the truth at the boundary and more than the truth once
    /// `|μ| > σ·√(2 ln 2)`, and no rescaling corrects a bias that changes sign. The
    /// production selection is not an argmin over two fixed candidates either.
    /// Supports come from seeded routing logits and basins are local minima of the
    /// inner solve, so neither has decision boundaries in closed form. A softmax,
    /// ordered Beta--Bernoulli or threshold gate is a smooth map of its logits, so a
    /// saturated assignment is not a discontinuity and gets no boundary term.
    ///
    /// The omission does not make the scale conservative. With `Cov(y) = φ·I` in
    /// the frame's norm, `E‖y − f̂‖² = E‖f̂ − μ‖² + N·φ − 2φ·df` for
    /// `df = Σⱼ Cov(f̂ⱼ, yⱼ)/φ`, and a routing chosen on the same data adds its
    /// search degrees of freedom to `df`. At a fixed risk `E‖f̂ − μ‖²` each one
    /// lowers the expected RSS by `2φ`, so `φ̂` is biased low by what it omits, the
    /// opposite sign to the non-negative bias above. Neither bias is bounded here.
    ///
    /// # The root is explicit (#2933 F38)
    ///
    /// `ν` and the frame charge are read from the data, the fitted state and `ρ`.
    /// The inner objective carries no `φ`, so under `Vb = φ·H⁻¹` the prior precision
    /// scales with `φ` together with the likelihood, and neither `R` nor `ν` moves
    /// with `φ`. Each frame's scale equation `φ·ν = RSS` is therefore explicit. Its
    /// root is returned in one evaluation, with no seed, fixed-point pass or
    /// contraction argument.
    pub(crate) fn reconstruction_dispersion(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        residual: ArrayView2<'_, f64>,
    ) -> Result<SaeReconstructionDispersion, String> {
        self.reconstruction_dispersion_with_geometry(loss, cache, rho, residual, None)
    }

    /// [`Self::reconstruction_dispersion`] with the fitted-response divergence read
    /// off a stationarity operator the caller already holds, so a shape report pays
    /// one dense eigendecomposition for the divergence and the covariance together
    /// (#2933 F33), on the fixed-frame route and on the frame-marginal one. `None`
    /// routes the divergence by admission.
    pub(crate) fn reconstruction_dispersion_with_geometry(
        &self,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        rho: &SaeManifoldRho,
        residual: ArrayView2<'_, f64>,
        geometry: Option<super::construction::HeldResponseGeometry<'_>>,
    ) -> Result<SaeReconstructionDispersion, String> {
        self.assignment.validate_rho_domain(rho)?;
        // FRAME CONSISTENCY: the raw energy prices the output-frame noise the MP
        // edge compares against; the likelihood energy prices the covariance
        // multiplier of the metric-weighted Hessian. See the fn doc.
        let (raw_rss, likelihood_rss, likelihood_frame) =
            self.reconstruction_residual_energies(loss, Some(residual))?;
        let fitted = self.try_fitted_for_rho(rho)?;
        if fitted.dim() != residual.dim() {
            return Err(format!(
                "reconstruction_dispersion: fitted {:?} != residual {:?}",
                fitted.dim(),
                residual.dim()
            ));
        }
        let target = &fitted - &residual;
        let response = match geometry {
            Some(geometry) => {
                self.fitted_response_divergence_from_geometry(geometry, rho, target.view(), cache)
            }
            None => self.fitted_response_divergence(target.view(), rho, cache),
        }
        .map_err(|refusal| format!("reconstruction_dispersion: {refusal}"))?;
        match response.estimator {
            FittedResponseDivergenceEstimator::ExactSpectral => log::debug!(
                "[SAE-DISPERSION] exact spectral fitted-response divergence {:.6e}, residual dof \
                 {:.6e} likelihood / {:.6e} raw",
                response.divergence,
                response.likelihood_residual_dof,
                response.raw_residual_dof
            ),
            FittedResponseDivergenceEstimator::Hutchinson { likelihood, raw } => log::debug!(
                "[SAE-DISPERSION] Hutchinson fitted-response divergence {:.6e} (standard error \
                 {:.3e}); residual dof {:.6e} (standard error {:.3e}) likelihood from {} probes, \
                 {:.6e} (standard error {:.3e}) raw from {} probes",
                likelihood.divergence,
                likelihood.divergence_standard_error,
                likelihood.residual_dof,
                likelihood.residual_dof_standard_error,
                likelihood.probes,
                raw.map_or(likelihood.residual_dof, |raw| raw.residual_dof),
                raw.map_or(likelihood.residual_dof_standard_error, |raw| {
                    raw.residual_dof_standard_error
                }),
                raw.map_or(likelihood.probes, |raw| raw.probes)
            ),
        }
        let frame_dimension = match response.frame_conditioning {
            SaeFrameConditioning::ConditionalOnFittedFrames(_) => {
                self.grassmann_evidence_dimension() as f64
            }
            SaeFrameConditioning::NoLearnedFrames
            | SaeFrameConditioning::MarginalOverLearnedFrames => 0.0,
        };
        let scale = |label: &str, rss: f64, residual_dof: f64| -> Result<f64, String> {
            if !(residual_dof.is_finite() && residual_dof > 0.0) {
                return Err(format!(
                    "reconstruction_dispersion: the {label} has no residual degrees of freedom: \
                     ‖I − R‖²_F = {residual_dof:.6e} at divergence {:.6e} and profiled frame \
                     dimension {frame_dimension}; the fitted response reproduces the \
                     observations, so no noise scale is estimable",
                    response.divergence
                ));
            }
            let phi = rss / residual_dof;
            if !phi.is_finite() || phi < 0.0 {
                return Err(format!(
                    "reconstruction_dispersion: non-finite/negative {label} {phi} (RSS={rss}, \
                     residual dof={residual_dof})"
                ));
            }
            Ok(phi.max(f64::MIN_POSITIVE))
        };
        Ok(SaeReconstructionDispersion {
            raw_output_noise_variance: scale(
                "raw output noise variance",
                raw_rss,
                response.raw_residual_dof - frame_dimension,
            )?,
            likelihood_dispersion: scale(
                "likelihood dispersion",
                likelihood_rss,
                response.likelihood_residual_dof - frame_dimension,
            )?,
            likelihood_frame,
            selection_conditioning: SaeSelectionConditioning::ConditionalOnFittedRouting,
        })
    }

    /// The border range of each atom's decoder block in the joint cache layout:
    /// the factored `(M_k·r_k)` range when frames are active, the full
    /// `(M_k·p)` range otherwise. One owner, because the selected inverse blocks
    /// of `A` and the bands that consume them must name the same coordinates.
    pub(crate) fn shape_covariance_border_ranges(&self) -> Vec<std::ops::Range<usize>> {
        if self.frames_active() {
            let frame_projection = FrameProjection::new(self);
            (0..self.k_atoms())
                .map(|k| frame_projection.atom_border_range(k))
                .collect()
        } else {
            self.beta_block_offsets().to_vec()
        }
    }

    /// Posterior covariance and ambient shape band for every atom — the
    /// user-facing uncertainty of the fitted manifold shapes.
    ///
    /// For atom `k` with decoder-block range `r_k` (see
    /// [`Self::shape_covariance_border_ranges`]),
    /// `Cov(β_k) = φ·[A⁺]_ββ[r_k, r_k]` is the φ-scaled posterior covariance of
    /// its decoder coefficients with the latent coordinates and the other atoms
    /// marginalized out through the exact observed information `A` (#2933 F33;
    /// see [`SaeObservedInformationCovariance`]). An
    /// [`SaeShapeInformation::Unavailable`] input yields explicit `None` bands
    /// carrying its reason. The ambient point at a coordinate
    /// `t` is `m_k(t) = Φ_k(t)·B_k`, *linear* in `β_k`, so its per-channel
    /// posterior variance is the closed form
    /// `Var_c(t) = Σ_{b1,b2} Φ_k(t)[b1] Φ_k(t)[b2] · Cov(β_k)[(b1,c),(b2,c)]`
    /// — no sampling. The band is evaluated at up to [`SHAPE_BAND_MAX_POINTS`]
    /// evenly-strided of the atom's own on-atom coordinates, reusing the basis
    /// values already stored on the atom, so it reports uncertainty exactly
    /// where the data lives and needs no basis-kind-specific grid.
    ///
    /// A near-degenerate atom has small retained curvature in `A`, so
    /// `Cov(β_k)` — and the band — fans out automatically: the band width is a
    /// per-coordinate visual of how well each atom is identified.
    ///
    /// `φ` is [`SaeReconstructionDispersion::posterior_covariance_scale`], the
    /// likelihood-frame dispersion. `A` is assembled through the same row
    /// metric as the likelihood, so under a whitening metric it already carries
    /// the metric's variance scale and the multiplier is dimensionless: scaling
    /// every observation unit by `c` (with `Σ → c²Σ`) scales `Cov(β)` by `c²`.
    ///
    /// The information's [`SaeFrameConditioning`] names its layout. Integrated
    /// over the learned frames (or with no frames) each block is already the
    /// `(M_k·p)` decoder covariance; held at the fitted frames it is the factored
    /// `(M_k·r_k)` coordinate covariance, lifted through `U_k` (#2933 F35).
    pub fn assemble_shape_uncertainty(
        &self,
        information: &SaeShapeInformation,
        dispersion: SaeReconstructionDispersion,
    ) -> Result<SaeShapeUncertainty, String> {
        let covariance = match information {
            SaeShapeInformation::ObservedInformation(covariance) => covariance,
            SaeShapeInformation::Unavailable(reason) => {
                return Ok(self.unavailable_shape_uncertainty(dispersion, reason.clone()));
            }
        };
        let p = self.output_dim();
        let covariance_scale = dispersion.posterior_covariance_scale();
        // #972 / #977 T1: held at the fitted frames, each atom's selected inverse
        // block is the FACTORED `(M_k·r_k)` coordinate covariance `Cov(vec C_k)`.
        // We LIFT it to the `(M_k·p)` decoder covariance `Cov(vec B_k) =
        // (I_{M_k} ⊗ U_k) Cov(vec C_k)(I_{M_k} ⊗ U_k)ᵀ` (since `B_k = C_k U_kᵀ`) so
        // the band code reads the `b·p + c` flat layout; that lift is conditional
        // on `U_k` (#2933 F35). Integrated over the frames, or with no frames, the
        // block is already `(M_k·p)` and the lift is skipped.
        let factored_layout = matches!(
            covariance.frame_conditioning,
            SaeFrameConditioning::ConditionalOnFittedFrames(_)
        );
        let frame_projection = FrameProjection::new(self);
        let block_ranges = if factored_layout {
            self.shape_covariance_border_ranges()
        } else {
            self.beta_block_offsets().to_vec()
        };
        if covariance.blocks.len() != self.k_atoms() {
            return Err(format!(
                "assemble_shape_uncertainty: the information carries {} atom blocks for K={}",
                covariance.blocks.len(),
                self.k_atoms()
            ));
        }
        // A framed atom's dense `(M_k·p)²` lift exists only to export the full decoder
        // covariance: its band variances are exact from the factored covariance either
        // way. The lifts are exported when every framed atom's covariance together fits
        // the memory governor's single-materialization cap, and they are charged on the
        // ledger while they are built. A refusal is an error (#2900).
        let export_framed_lift = factored_layout && self.framed_decoder_covariance_admitted();
        let lift_charge = match self.framed_decoder_covariance_bytes() {
            Some(bytes) if export_framed_lift => Some(
                gam_runtime::resource::MemoryGovernor::global()
                    .try_reserve(bytes, "SaeManifoldTerm framed decoder covariance export")
                    .map_err(|error| {
                        format!(
                            "assemble_shape_uncertainty: refusing the {bytes}-byte framed decoder \
                             covariance export: {error}"
                        )
                    })?,
            ),
            _ => None,
        };
        let mut atoms = Vec::with_capacity(self.k_atoms());
        for (k, atom) in self.atoms.iter().enumerate() {
            let width = block_ranges[k].len();
            if covariance.blocks[k].dim() != (width, width) {
                return Err(format!(
                    "assemble_shape_uncertainty: atom {k} information block {:?} does not \
                     match its border range of width {width}",
                    covariance.blocks[k].dim()
                ));
            }
            let cov_block = covariance.blocks[k].clone();
            let n_rows = atom.n_obs();
            let d = atom.latent_dim();
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

            let framed = factored_layout && atom.decoder_frame.is_some();
            let cov = if framed && !export_framed_lift {
                // The dense `(M_k·p)²` lifts do not fit the governor's cap, and
                // they exist only to export the full covariance. Compute the band
                // variance EXACTLY from the factored frame covariance instead:
                // with `B_k = C_k·U_kᵀ`,
                //   Var_c(t) = (φ ⊗ u_c)ᵀ Cov(vec C_k) (φ ⊗ u_c)
                // which is the r×r quadratic form `u_cᵀ Y u_c` with
                //   Y = Σ_{b1,b2} φ[b1] φ[b2] Cov(C)[(b1,·),(b2,·)].
                let mut cov_c = cov_block;
                cov_c.mapv_inplace(|v| v * covariance_scale);
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
                cov.mapv_inplace(|v| v * covariance_scale);
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
            // Row-sandwich band from this atom's block of `[A⁺]_ββ J [A⁺]_ββ`,
            // pushed forward in its own border layout (through `U_k` on a framed
            // atom) with no dispersion multiplier.
            let robust_block = covariance
                .robust_blocks
                .get(k)
                .filter(|block| block.dim() == (width, width))
                .ok_or_else(|| {
                    format!(
                        "assemble_shape_uncertainty: atom {k} has no robust block matching its \
                         border range of width {width}"
                    )
                })?;
            let mut band_sd_robust = Array2::<f64>::zeros((g, p));
            for (gi, &row) in eval_rows.iter().enumerate() {
                let basis = atom.basis_values.row(row);
                for c in 0..p {
                    let var = if framed {
                        frame_projection.output_variance(k, robust_block.view(), basis, c)
                    } else {
                        frame_projection.full_output_variance(k, robust_block.view(), basis, c)
                    };
                    band_sd_robust[[gi, c]] = var.max(0.0).sqrt();
                }
            }
            atoms.push(SaeAtomShapeUncertainty {
                decoder_covariance: cov,
                band_coords: Some(band_coords),
                band_mean: Some(band_mean),
                band_sd: Some(band_sd),
                band_sd_robust: Ok(band_sd_robust),
            });
        }
        drop(lift_charge);
        Ok(SaeShapeUncertainty {
            dispersion,
            operator: SaeShapeCovarianceOperator::ObservedInformation {
                identified_rank: covariance.identified_rank,
                ambient_dim: covariance.ambient_dim,
                frame_conditioning: covariance.frame_conditioning,
            },
            atoms,
            selection_conditioning: SaeSelectionConditioning::ConditionalOnFittedRouting,
        })
    }

    /// Recompute the JOINT observed-information shape bands at the CURRENT
    /// (final) term + ρ state — the same joint covariance
    /// [`Self::assemble_shape_uncertainty`] forms, but rebuilt AFTER a
    /// structure-changing or finalization move invalidated the pre-search state.
    ///
    /// [`super::SaeManifoldOuterObjective::decoder_shape_uncertainty`] reads the
    /// information off the outer objective BEFORE `into_fitted` consumes it, so
    /// the bands it returns describe the PRE-search dictionary at the settled ρ.
    /// When evidence-guarded structure search grows / re-converges the whole
    /// dictionary (a certified birth / fission / fusion or a demoted death), or a
    /// finalization fallback swaps the settled basin / canonicalizes charts, that
    /// information no longer describes the returned model. This re-converges THIS
    /// (final) term at `rho` through the penalized quasi-Laplace criterion,
    /// re-forms the exact observed information `A` at that inner optimum
    /// ([`Self::exact_observed_information_shape_covariance`]), and reads the
    /// per-atom covariance and bands off its selected inverse, scaling by the
    /// reconstruction dispersion `φ̂`. The result carries the cross-atom
    /// covariance and the decoder-coordinate couplings, and its per-channel band
    /// varies across output channels. Every atom is covered because `A` is
    /// assembled at the final dictionary's `k_atoms()`.
    ///
    /// The term is already at its optimum, so the inner re-solve converges
    /// immediately. When the streaming plan cannot materialize `A`, or `A` has
    /// resolved negative curvature at the state, the returned atom entries carry
    /// explicit `None` bands and [`SaeShapeUncertainty::operator`] names why.
    /// Call before `Self::into_fitted` has run is not required; it takes the fitted
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
        let plan = self.streaming_plan()?.admitted_or_error(
            self.n_obs(),
            self.output_dim(),
            self.k_atoms(),
        )?;
        if !plan.direct_logdet_admitted() {
            // No dense exact observed information at this scale: report explicit
            // unavailability rather than substituting a different covariance.
            let dispersion = self.unfactored_reconstruction_dispersion(target, rho)?;
            return Ok(self.unavailable_shape_uncertainty(
                dispersion,
                SaeShapeCovarianceUnavailable::NoDenseObservedInformation,
            ));
        }
        let (_cost, loss, cache) = self
            .penalized_quasi_laplace_criterion_with_cache(
                target,
                rho,
                registry,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .map_err(|error| error.to_string())?;
        let residual = self.reconstruction_residual(target, rho)?;
        // One decision of which operator the report inverts, which feeds both the
        // dispersion's divergence and the covariance (#2933 F33).
        let route = self.shape_information_route(rho, target, registry, &cache)?;
        let dispersion = self.reconstruction_dispersion_with_geometry(
            &loss,
            &cache,
            rho,
            residual.view(),
            Some(route.held_response_geometry()),
        )?;
        let information = self.shape_information(&route, rho, target, &cache)?;
        self.assemble_shape_uncertainty(&information, dispersion)
    }

    /// Bytes of every framed atom's dense `(M_k·p)²` decoder covariance together,
    /// or `None` when the count overflows.
    pub(crate) fn framed_decoder_covariance_bytes(&self) -> Option<usize> {
        let p = self.output_dim();
        self.atoms
            .iter()
            .filter(|atom| atom.decoder_frame.is_some())
            .try_fold(0_usize, |total, atom| {
                let width = atom.basis_size().checked_mul(p)?;
                total.checked_add(gam_runtime::resource::dense_f64_bytes(width, width)?)
            })
    }

    /// Whether every framed atom's dense `(M_k·p)²` decoder covariance can be held
    /// at once: their total against the memory governor's stationary
    /// single-materialization cap (#2900).
    pub(crate) fn framed_decoder_covariance_admitted(&self) -> bool {
        self.framed_decoder_covariance_bytes().is_some_and(|bytes| {
            bytes
                <= gam_runtime::resource::MemoryGovernor::global()
                    .single_materialization_cap_bytes()
        })
    }

    /// Explicitly unavailable joint shape uncertainty, carrying why no covariance
    /// exists: an execution plan that cannot materialize the dense observed
    /// information, or a state whose observed information has resolved negative
    /// curvature.
    pub(crate) fn unavailable_shape_uncertainty(
        &self,
        dispersion: SaeReconstructionDispersion,
        reason: SaeShapeCovarianceUnavailable,
    ) -> SaeShapeUncertainty {
        let atoms = self
            .atoms
            .iter()
            .map(|_| SaeAtomShapeUncertainty {
                decoder_covariance: None,
                band_coords: None,
                band_mean: None,
                band_sd: None,
                band_sd_robust: Err(reason.clone()),
            })
            .collect();
        SaeShapeUncertainty {
            dispersion,
            operator: SaeShapeCovarianceOperator::Unavailable(reason),
            atoms,
            selection_conditioning: SaeSelectionConditioning::ConditionalOnFittedRouting,
        }
    }

}

#[cfg(test)]
mod persisted_reconstruct_tests {
    use super::*;

    // Exercises `reconstruct_persisted_atom_set`: a stateless K=1 periodic-atom
    // round trip that must equal `a_i · (Φ(t_i) · B)` computed directly from the
    // exact evaluator declared by the persisted geometry plan.
    #[test]
    fn reconstruct_persisted_periodic_atom_matches_direct_decode() {
        let n_rows = 4usize;
        let p_out = 2usize;
        let width = 3usize; // odd decoder width required for periodic
        let coords = Array2::from_shape_vec((n_rows, 1), vec![0.1, 0.7, 1.9, 2.8]).unwrap();
        let decoder =
            Array2::from_shape_vec((width, p_out), vec![0.5, -0.2, 0.3, 0.9, -0.4, 0.1]).unwrap();
        let assignments = Array2::from_shape_vec((n_rows, 1), vec![1.0, 0.5, 0.8, 0.2]).unwrap();

        let plan = SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Periodic,
            1,
            SaeBasisResolution::PeriodicHarmonics { order: 1 },
            SaeReferenceMetricPlan::UnitCircle,
        )
        .unwrap();
        let out = reconstruct_persisted_atom_set(
            &[plan],
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

#[cfg(test)]
mod framed_decoder_covariance_export_2900_tests {
    use super::*;
    use ndarray::{Array3, array};

    /// #2900 — a framed atom's dense `(M_k·p)²` decoder covariance used to be
    /// exported only up to 2^24 entries, an entry-count literal outside the memory
    /// governor. The export is admitted on the governor's cap now. One atom with
    /// `M = 2` and `p = 2049` has 16,793,604 entries, just past the old window. Held
    /// at its fitted frame `U`, its factored covariance `Cov(vec C)` must export the
    /// lift `φ·(I ⊗ U)·Cov(vec C)·(I ⊗ U)ᵀ`, and the band read from the export must
    /// equal the factored closed form `φ·u_c²·φ(t)ᵀ Cov(vec C) φ(t)`.
    #[test]
    fn framed_decoder_covariance_is_exported_past_the_old_entry_window_2900() {
        let (n, m, p) = (4, 2, 2049);
        assert!((m * p) * (m * p) > 1 << 24, "the fixture must sit past the old window");
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| 0.25 * i as f64);
        let basis = Array2::from_shape_fn((n, m), |(i, b)| if b == 0 { 1.0 } else { coords[[i, 0]] });
        let jet = Array3::from_shape_fn((n, m, 1), |(_, b, _)| if b == 0 { 0.0 } else { 1.0 });
        let raw_frame = Array2::from_shape_fn((p, 1), |(c, _)| (c % 7) as f64 - 3.0);
        let frame_norm = raw_frame.mapv(|v| v * v).sum().sqrt();
        let unit_frame = raw_frame.mapv(|v| v / frame_norm);
        let decoder = Array2::from_shape_fn((m, p), |(b, c)| [1.5, -0.5][b] * unit_frame[[c, 0]]);
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "framed_line_2900",
            SaeAtomBasisKind::Linear,
            1,
            basis.clone(),
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .expect("atom shapes agree");
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .expect("assignment shapes agree");
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
        term.atoms[0].decoder_frame = Some(GrassmannFrame::from_oriented(unit_frame, array![1.0]));
        let u = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("framed atom")
            .frame()
            .to_owned();
        assert_eq!(
            term.framed_decoder_covariance_bytes(),
            Some(8 * (m * p) * (m * p)),
            "the export is priced as one dense (M·p)² f64 block"
        );
        assert!(term.framed_decoder_covariance_admitted());

        let factored = array![[0.3, 0.1], [0.1, 0.2]];
        let information = SaeShapeInformation::ObservedInformation(SaeObservedInformationCovariance {
            blocks: vec![factored.clone()],
            robust_blocks: vec![factored.clone()],
            identified_rank: n + m,
            ambient_dim: n + m,
            frame_conditioning: SaeFrameConditioning::ConditionalOnFittedFrames(
                SaeFrameMarginalUnavailable::UnframedObservedInformationNotAdmitted,
            ),
        });
        let scale = 0.25;
        let dispersion = SaeReconstructionDispersion {
            raw_output_noise_variance: scale,
            likelihood_dispersion: scale,
            likelihood_frame: SaeLikelihoodFrame::RawOutput,
            selection_conditioning: SaeSelectionConditioning::ConditionalOnFittedRouting,
        };
        let shape = term
            .assemble_shape_uncertainty(&information, dispersion)
            .expect("shape uncertainty held at the fitted frame");
        assert_eq!(
            shape.operator.as_str(),
            "observed_information_conditional_on_fitted_frames",
            "the result must name the covariance it holds"
        );
        assert_eq!(
            shape.operator.frame_conditioning_reason(),
            Some("unframed_observed_information_not_admitted"),
            "the result must name why the frame is held fixed"
        );
        let cov = shape.atoms[0]
            .decoder_covariance
            .as_ref()
            .expect("the lift must be exported past the old 2^24-entry window");
        assert_eq!(cov.dim(), (m * p, m * p));
        let largest = factored.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())) * scale;
        for b1 in 0..m {
            for b2 in 0..m {
                for c1 in 0..p {
                    for c2 in 0..p {
                        let expected = scale * u[[c1, 0]] * factored[[b1, b2]] * u[[c2, 0]];
                        let got = cov[[b1 * p + c1, b2 * p + c2]];
                        assert!(
                            (got - expected).abs() <= 1.0e-12 * largest,
                            "Cov[({b1},{c1}),({b2},{c2})] = {got:.6e}, lift {expected:.6e}"
                        );
                    }
                }
            }
        }
        let band = shape.atoms[0].band_sd.as_ref().expect("model-based band");
        for row in 0..n {
            let phi = basis.row(row);
            let quadratic = phi.dot(&factored.dot(&phi));
            for c in 0..p {
                let expected = scale * u[[c, 0]] * u[[c, 0]] * quadratic;
                let got = band[[row, c]] * band[[row, c]];
                assert!(
                    (got - expected).abs() <= 1.0e-10 * expected.abs() + 1.0e-15,
                    "row {row}, channel {c}: band variance {got:.6e}, factored {expected:.6e}"
                );
            }
        }
    }
}
