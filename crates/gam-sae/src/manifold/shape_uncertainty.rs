use super::*;

/// Cap on the number of coordinates at which a per-atom shape band is
/// materialized. The full per-atom decoder covariance is exact and exposed
/// regardless; this only bounds the cost of the convenience band, which is
/// evaluated at an evenly-strided subset of the atom's own on-atom coordinates.
pub const SHAPE_BAND_MAX_POINTS: usize = 512;

/// Posterior uncertainty of one fitted atom's manifold shape.
///
/// In the primary path — [`SaeManifoldTerm::assemble_shape_uncertainty`], and
/// after a structure/finalization change its final-state twin
/// [`SaeManifoldTerm::recompute_joint_shape_uncertainty`] — the covariance is
/// the φ-scaled β-block of the pseudo-inverse of the JOINT exact observed
/// information `A` (coordinates and the other atoms marginalized out), on the
/// identified space the exact-A criterion prices; see
/// [`SaeObservedInformationCovariance`]. The band is its closed-form
/// push-forward through the linear basis→ambient map `m_k(t) = Φ_k(t)·B_k`.
/// This is what the production fit returns.
///
/// When no such covariance exists — a streaming plan that cannot materialize
/// `A`, or a state where `A` has resolved negative curvature — every band field
/// is `None`, the robust band carries the reason, and
/// [`SaeShapeUncertainty::operator`] names it. Neither a per-atom marginal nor
/// the majorizer's inverse is substituted.
#[derive(Debug, Clone)]
pub struct SaeAtomShapeUncertainty {
    /// φ-scaled posterior covariance of this atom's decoder coefficients,
    /// `Cov(β_k) = φ·[A⁺]_ββ[block_k]`, shape `(M_k·p, M_k·p)` in the decoder's
    /// row-major `(basis, channel)` flat layout (flat index `b·p + c`).
    ///
    /// `None` on a framed atom held at its fitted frame when every framed atom's
    /// dense `(M_k·p)²` covariance together exceeds the memory governor's
    /// single-materialization cap (LLM-scale ambient `p`: at `(M=8, p=2048)` the
    /// dense block is 2 GiB *per atom*). The band quantities below are then
    /// computed from the factored `(M_k·r_k)²` frame covariance without lifting
    /// it, which holds the frame at its fitted value (see
    /// [`SaeFrameConditioning`]).
    pub decoder_covariance: Option<Array2<f64>>,
    /// Coordinates at which the band is evaluated, shape `(G, d_k)`.
    pub band_coords: Option<Array2<f64>>,
    /// Fitted ambient point `m_k(t) = Φ_k(t)·B_k` at each band coordinate,
    /// shape `(G, p)`.
    pub band_mean: Option<Array2<f64>>,
    /// Posterior standard deviation of each ambient channel at each band
    /// coordinate, `sqrt(Var_c(t))` with
    /// `Var_c(t) = Σ_{b1,b2} Φ[b1] Φ[b2] Cov(β_k)[(b1,c),(b2,c)]`, shape
    /// `(G, p)`.
    ///
    /// This is the MODEL-BASED band (`Cov = φ̂ A⁺`), correct only when the
    /// working reconstruction likelihood is correctly specified. See
    /// [`Self::band_sd_robust`] for the row-sandwich sampling companion.
    pub band_sd: Option<Array2<f64>>,
    /// Row-sandwich (HC0) sampling standard deviation of each ambient channel at
    /// each band coordinate, shape `(G, p)`: the push-forward of
    /// `Cov_rob(β) = [A⁺ J A⁺]_ββ` through the same basis map as
    /// [`Self::band_sd`].
    ///
    /// * Estimating equation: the joint stationarity of the fit over the per-row
    ///   coordinates and gate logits and the decoder border. The bread is the
    ///   pseudo-inverse `A⁺` of its exact observed information, the operator the
    ///   model-based band inverts.
    /// * Meat: `J = Σ_i (g_i − ḡ)(g_i − ḡ)ᵀ` over the rows' estimating-function
    ///   summands, every one centred about its row mean. The totals are not
    ///   sampling noise: at a penalized root `Σ_i s_i = −∇_β P ≠ 0`, and the mass
    ///   summands total `M_k`. An uncentred outer product would fold those totals
    ///   into the variance. When each row's coordinates and logits depend on that
    ///   row alone, its row-local data score is balanced at the root by its own
    ///   prior gradient and only the border survives:
    ///   `Cov_rob(β) = [A⁺]_ββ J_ββ [A⁺]_ββ` with
    ///   `J_ββ = Σ_i (s_i − s̄)(s_i − s̄)ᵀ` and `s_i = ∂/∂β ½ w_i r_iᵀ M_i r_i`, the
    ///   profile sandwich. It is formed over the whole border because `J_ββ`
    ///   couples atoms and output channels.
    /// * Aggregate masses: the ordered Beta--Bernoulli gate prior couples rows
    ///   through each atom's mass `M_k = Σ_i w_i z_ik`, whose cross-row curvature
    ///   `Σ_k c_k u_k u_kᵀ` is part of `A`. The masses enter as estimating
    ///   equations `M_k − Σ_i w_i z_ik = 0`. Eliminating them recovers `A`, row
    ///   `i`'s effective score becomes
    ///   `[A⁺]_ββ (s_i − s̄) − Σ_k [A⁺(c_k u_k)]_β (q̄_k − w_i z_ik)`, and the meat
    ///   carries the masses' sampling variability.
    /// * Target: the frequentist sampling covariance of the penalized decoder
    ///   estimate about its own limit, not a posterior covariance. Score and bread
    ///   come from one loss, so a known or estimated noise scale cancels and no
    ///   dispersion multiplies it.
    /// * Randomness included: independent sampling of rows. The smoothing and ARD
    ///   hyperparameters, the frozen collapse gates, the row metric and the
    ///   support/structure selection are held at their fitted values. The learned
    ///   decoder frames are integrated or held fixed exactly as the model-based
    ///   band's [`SaeFrameConditioning`] records. Dependence between rows of one
    ///   sequence is not modelled: the fit carries no row cluster structure.
    ///
    /// Reported ALONGSIDE the model-based band, never in place of it. `Err`
    /// carries the reason no bread exists.
    pub band_sd_robust: Result<Array2<f64>, SaeShapeCovarianceUnavailable>,
}

/// The frame in which the reconstruction likelihood measures its residuals,
/// which fixes the units of [`SaeReconstructionDispersion::likelihood_dispersion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeLikelihoodFrame {
    /// Isotropic data fit `½‖r_n‖²` (no metric, or a gauge-only metric): the
    /// dispersion is in squared output units over `n·p` scalar observations.
    RawOutput,
    /// Whitened data fit `½ r_nᵀ M_n r_n` with `M_n = U_n U_nᵀ` of rank
    /// `metric_rank`: the dispersion is dimensionless over `n·metric_rank`
    /// whitened scalar observations.
    Whitened { metric_rank: usize },
}

/// The two reconstruction noise scales of a fit, which answer different
/// questions and need not share units or values.
///
/// With an inverse-noise metric `M_n = Σ_n⁻¹` the joint Hessian
/// `H = JᵀMJ + P` already carries `Σ`'s variance scale: for `M = σ⁻²I` and a
/// linear decoder `H⁻¹ = σ²(XᵀX + σ²λS)⁻¹`. Multiplying that inverse by a raw
/// residual variance `≈ σ²` produces `σ⁴(…)⁻¹`, so the covariance multiplier is
/// the dimensionless metric-frame dispersion, never the raw one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaeReconstructionDispersion {
    /// Raw output-frame noise variance per scalar observation, `RSS/ν` with
    /// `ν = ‖I − R‖²_F` over the raw frame's scalars of positive weight and
    /// `R = ∂f̂/∂y` the fitted response (#2933 F40, see
    /// `SaeManifoldTerm::reconstruction_dispersion`), in squared output units
    /// whatever metric the likelihood uses. Consumers that compare against
    /// unwhitened output quantities read this: the Marchenko–Pastur rank edge
    /// (against the raw decoder Gram), the incoherence SNR and the per-atom inner
    /// fits.
    pub raw_output_noise_variance: f64,
    /// Dispersion `φ̂` of the working likelihood `exp(−½ Σ w_n r_nᵀ M_n r_n / φ)`,
    /// `RSS/ν` in the frame named by [`Self::likelihood_frame`], with `RSS` and
    /// `ν = ‖I − R‖²_F` in that frame's norm. Equal to
    /// [`Self::raw_output_noise_variance`] when that frame is
    /// [`SaeLikelihoodFrame::RawOutput`].
    pub likelihood_dispersion: f64,
    /// Units and scalar-observation count of [`Self::likelihood_dispersion`].
    pub likelihood_frame: SaeLikelihoodFrame,
    /// The routing both scales hold fixed. `φ̂` is conditional on the fitted
    /// routing and omits its search degrees of freedom, so it is biased low
    /// (#2933 F37; see [`SaeSelectionConditioning`]).
    pub selection_conditioning: SaeSelectionConditioning,
}

impl SaeReconstructionDispersion {
    /// Multiplier turning the metric-weighted inverse joint Hessian `H⁻¹` into
    /// the posterior covariance, `Cov = φ̂·H⁻¹`.
    ///
    /// The inner objective is `½ Σ w_n r_nᵀ M_n r_n + P(θ; λ)` with no `φ`, so
    /// each fitted penalty strength is the product `λ = φ·λ′` of a prior
    /// precision `λ′` and the likelihood dispersion. The posterior
    /// `exp(−½ rᵀMr/φ − P(θ; λ′))` then has Hessian `H/φ` and covariance exactly
    /// `φ·H⁻¹`: the prior scales with the dispersion. Holding `λ′ = λ` fixed
    /// instead would give `(JᵀMJ/φ + P″)⁻¹`, which is not a multiple of `H⁻¹`.
    pub fn posterior_covariance_scale(&self) -> f64 {
        self.likelihood_dispersion
    }
}

/// The routing a noise scale or a shape covariance holds fixed (#2933 F37).
///
/// The TopK support, a frozen routing, or the basin the inner solve converged to
/// is chosen from the same data the fit is scored on. Holding that choice fixed
/// omits its search degrees of freedom. For fixed candidates `±a` and
/// `y ~ N(μ, σ²)`, the selection `a·sign(y)` has `(2a/σ)·φ(μ/σ)` of them, a function
/// of the unknown mean. A statistic `T` of one draw has `E T(y) = (T ∗ N(0, σ²))(μ)`,
/// so an unbiased `T` would have to be a point mass, and a resampling estimate that
/// reruns routing and the joint fit would make every criterion evaluation
/// stochastic. The selection is therefore conditioned on, not estimated. By
/// `E‖y − f̂‖² = E‖f̂ − μ‖² + N·φ − 2φ·df`, each omitted search degree of freedom
/// lowers the expected RSS by `2φ` at fixed risk, so the routing-conditional `φ̂`
/// is biased low.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeSelectionConditioning {
    /// The fitted routing is held fixed. Its search degrees of freedom are omitted,
    /// not estimated, and `φ̂` is biased low by them.
    ConditionalOnFittedRouting,
}

/// Whether a shape covariance integrates over the learned Grassmann decoder
/// frames or holds them at their fitted values (#2933 F35).
///
/// A framed decoder `B_k = C_k U_kᵀ` moves along `δB_k = δC_k U_kᵀ + C_k δU_kᵀ`.
/// The lift `(I ⊗ U_k)·Cov(vec C_k)·(I ⊗ U_k)ᵀ` keeps the first term only: with
/// `U = e₁` in two outputs, `C = 3` and angular variance `0.01` it reports zero
/// variance along `e₂`, where the frame rotation contributes `C²·0.01 = 0.09`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeFrameConditioning {
    /// No atom carries a learned frame, so there is nothing to integrate.
    NoLearnedFrames,
    /// The Laplace covariance on the product of the fixed-rank decoder manifolds.
    /// The observed information is written in the identified tangent coordinates
    /// `(vec δC_k, vec W_k)` with `δU_k = U_k⊥ W_k`, including the bilinear cross
    /// curvature `⟨∇_B L, δC_k W_kᵀ U_k⊥ᵀ⟩`, and its pseudo-inverse is pushed
    /// forward to `vec B_k`. The `GL(r_k)` factorization gauge never enters: the
    /// tangent coordinates name physical decoder motions only.
    MarginalOverLearnedFrames,
    /// Every frame is held at its fitted `U_k`: orientation uncertainty is omitted.
    ConditionalOnFittedFrames(SaeFrameMarginalUnavailable),
}

/// Why a framed fit reports a covariance conditional on its fitted frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeFrameMarginalUnavailable {
    /// The execution plan does not admit the dense observed information of the
    /// unframed decoder `(t, vec B)`, from which the tangent operator is formed.
    UnframedObservedInformationNotAdmitted,
    /// The dense `(M_k·p)²` decoder covariances of every framed atom together
    /// exceed the memory governor's single-materialization cap.
    DecoderCovariancesExceedMaterializationCap,
    /// Atom `atom`'s decoder has numerical rank below its frame rank. The
    /// fixed-rank manifold is singular there and has no tangent space of
    /// dimension `r_k(M_k + p − r_k)`.
    FrameCoordinatesRankDeficient { atom: usize },
}

/// Selected inverse blocks of the exact observed information on its identified
/// space, one per atom (#2933 F33).
///
/// `A = ∇²_θθ L` is the Hessian of the penalized inner objective in the joint
/// `(t, β)` cache layout: Gauss–Newton curvature, the residual curvature
/// `Σ_{i,c} (WMr)_{ic} ∇²f_{ic}`, and the exact prior curvature. It is not the
/// positive majorizer `B` the inner Newton solve factors, whose Schur inverse
/// this replaced: with `B = 2` and a residual-curvature correction of `−1`,
/// `B⁻¹ = 0.5` while the observed-information variance is `A⁻¹ = 1`, and no
/// accuracy in solving `B` turns one into the other.
///
/// Block `k` is `[A⁺]_ββ[r_k, r_k]` over atom `k`'s border range `r_k`: every
/// latent coordinate and every other atom is marginalized through the whole
/// joint operator. `A⁺` is the covariant pseudo-inverse over the generalized eigenpairs
/// `(μᵢ, wᵢ)` of the pencil `(A, Φ)` that the exact-A value path retains
/// (`μᵢ > rank_floor(i)`, #2933 F07), so the covariance lives on the same identified
/// space the criterion prices. The blocks are unscaled;
/// [`SaeReconstructionDispersion::posterior_covariance_scale`] multiplies them
/// at assembly.
#[derive(Debug, Clone)]
pub struct SaeObservedInformationCovariance {
    /// `[A⁺]_ββ[r_k, r_k]` per atom. Under
    /// [`SaeFrameConditioning::ConditionalOnFittedFrames`] it is in the cache's
    /// factored `(M_k·r_k)` layout; otherwise it is the full `(M_k·p)` decoder
    /// block.
    pub(crate) blocks: Vec<Array2<f64>>,
    /// The row-sandwich covariance per atom in the same layout: the border block
    /// of the joint sandwich over the per-row data scores and, under the ordered
    /// Beta--Bernoulli prior, the aggregate-mass estimating equations (see
    /// [`SaeAtomShapeUncertainty::band_sd_robust`]). Never scaled by a dispersion.
    pub(crate) robust_blocks: Vec<Array2<f64>>,
    /// Number of retained directions of `A`.
    pub(crate) identified_rank: usize,
    /// Dimension of the joint `(t, β)` operator.
    pub(crate) ambient_dim: usize,
    /// Whether the blocks integrate the learned frames, and so which layout they
    /// are in.
    pub(crate) frame_conditioning: SaeFrameConditioning,
}

/// The information a shape covariance is assembled from, or why none exists.
#[derive(Debug, Clone)]
pub enum SaeShapeInformation {
    /// Selected inverse blocks of the exact observed information.
    ObservedInformation(SaeObservedInformationCovariance),
    /// No covariance exists at this state or on this execution plan.
    Unavailable(SaeShapeCovarianceUnavailable),
}

/// Why a fit reports no shape covariance.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeShapeCovarianceUnavailable {
    /// The admitted execution plan does not materialize the dense joint
    /// observed information, so no selected inverse of `A` is formed.
    NoDenseObservedInformation,
    /// `A` has resolved negative curvature: the state is not a mode of the
    /// penalized objective, so `A⁺` is not a Laplace covariance. The value path
    /// may still price such a basin through the majorizer's concave clamp, but
    /// that clamped operator has no probabilistic contract as a covariance.
    IndefiniteObservedInformation {
        negative_directions: usize,
        most_negative_curvature: f64,
    },
}

/// The operator a reported shape covariance inverts.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeShapeCovarianceOperator {
    /// `φ̂·[A⁺]_ββ` on the identified space of the exact observed information.
    ObservedInformation {
        identified_rank: usize,
        ambient_dim: usize,
        frame_conditioning: SaeFrameConditioning,
    },
    /// No covariance was produced; every band field is `None`.
    Unavailable(SaeShapeCovarianceUnavailable),
}

impl SaeShapeCovarianceOperator {
    /// Wire name of the operator, owned here so bindings marshal rather than map.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ObservedInformation {
                frame_conditioning: SaeFrameConditioning::NoLearnedFrames,
                ..
            } => "observed_information",
            Self::ObservedInformation {
                frame_conditioning: SaeFrameConditioning::MarginalOverLearnedFrames,
                ..
            } => "observed_information_marginal_over_learned_frames",
            Self::ObservedInformation {
                frame_conditioning: SaeFrameConditioning::ConditionalOnFittedFrames(_),
                ..
            } => "observed_information_conditional_on_fitted_frames",
            Self::Unavailable(SaeShapeCovarianceUnavailable::NoDenseObservedInformation) => {
                "unavailable_no_dense_observed_information"
            }
            Self::Unavailable(
                SaeShapeCovarianceUnavailable::IndefiniteObservedInformation { .. },
            ) => "unavailable_indefinite_observed_information",
        }
    }

    /// Wire name of why every learned frame is held at its fitted value, or `None`
    /// when the covariance integrates the frames, no atom carries one, or no
    /// covariance was produced. Admission to the integrated covariance depends on
    /// the host's memory, so a result says which covariance it holds and why.
    pub fn frame_conditioning_reason(&self) -> Option<&'static str> {
        match self {
            Self::ObservedInformation {
                frame_conditioning: SaeFrameConditioning::ConditionalOnFittedFrames(reason),
                ..
            } => Some(reason.as_str()),
            Self::ObservedInformation { .. } | Self::Unavailable(_) => None,
        }
    }
}

impl SaeFrameMarginalUnavailable {
    /// Wire name of the reason, owned here so bindings marshal rather than map.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UnframedObservedInformationNotAdmitted => {
                "unframed_observed_information_not_admitted"
            }
            Self::DecoderCovariancesExceedMaterializationCap => {
                "decoder_covariances_exceed_materialization_cap"
            }
            Self::FrameCoordinatesRankDeficient { .. } => "frame_coordinates_rank_deficient",
        }
    }
}

/// Posterior shape uncertainty for a whole SAE-manifold fit: one band per atom
/// plus the reconstruction noise scales, of which
/// [`SaeReconstructionDispersion::posterior_covariance_scale`] scales every
/// covariance. See [`SaeManifoldTerm::assemble_shape_uncertainty`].
#[derive(Debug, Clone)]
pub struct SaeShapeUncertainty {
    /// Raw and likelihood-frame reconstruction noise scales.
    pub dispersion: SaeReconstructionDispersion,
    /// The operator every atom's covariance inverts, or why none was produced.
    pub operator: SaeShapeCovarianceOperator,
    /// One entry per atom, in atom order.
    pub atoms: Vec<SaeAtomShapeUncertainty>,
    /// The routing the report holds fixed. `A` is the information at the selected
    /// support and basin, and [`Self::dispersion`] is routing-conditional, so the
    /// bands omit selection uncertainty and `φ̂` is biased low (#2933 F37).
    pub selection_conditioning: SaeSelectionConditioning,
}

/// The row-sandwich meat of the joint estimating equations, with the carriers of
/// the aggregate-mass equations it includes; see
/// [`SaeAtomShapeUncertainty::band_sd_robust`].
#[derive(Debug, Clone)]
pub(crate) struct RowSandwichMeat {
    /// `Σ_i (g_i − ḡ)(g_i − ḡ)ᵀ` over `g_i = (s_i, −q_i)`: the border data scores
    /// followed by one summand per aggregate mass, each centred about its row mean.
    pub(crate) meat: Array2<f64>,
    /// `(c_k, [(t index, u_ik)])` per aggregate mass, so the observed
    /// information's cross-row block is `Σ_k c_k u_k u_kᵀ`. Empty when no prior
    /// couples rows.
    pub(crate) mass_carriers: Vec<(f64, Vec<(usize, f64)>)>,
}

impl SaeManifoldTerm {
    /// The row-sandwich meat at `target`. Row `i`'s border score is the gradient
    /// of its weighted data fit `½ w_i r_iᵀ M_i r_i` with
    /// `r_i = Σ_k a_ik m_k(t_ik) − z_i`:
    /// `s_i[(k, b, c)] = w_i a_ik Φ_k(t_ik)[b] (M_i r_i)[c]`, projected through the
    /// atom frames when they are active. It is the per-row summand of the β
    /// data-fit gradient the arrow assembly accumulates. Under the ordered
    /// Beta--Bernoulli prior each aggregate mass `M_k = Σ_i q_ik`, `q_ik = w_i z_ik`,
    /// adds the summand `−q_ik` of its equation `M_k − Σ_i q_ik`. Every summand is
    /// centred about its row mean ([`CentredOuterProduct`]).
    pub(crate) fn row_sandwich_meat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        target: ArrayView2<'_, f64>,
    ) -> Result<RowSandwichMeat, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.dim() != (n, p) {
            return Err(format!(
                "row_sandwich_meat: target must be ({n}, {p}); got {:?}",
                target.dim()
            ));
        }
        let mass_carriers = self.ordered_mass_hessian_carriers(rho, cache)?;
        let mass_dim = mass_carriers.len();
        let mass_rows = ordered_beta_bernoulli_weighted_active_mass_rows(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?
        .unwrap_or_else(|| Array1::<f64>::zeros(0));
        if mass_rows.len() != n * mass_dim {
            return Err(format!(
                "row_sandwich_meat: {} mass summands for {n} rows and {mass_dim} aggregate masses",
                mass_rows.len()
            ));
        }
        let frames_active = self.frames_active();
        let frame_projection = FrameProjection::new(self);
        let border_dim = if frames_active {
            frame_projection.border_dim()
        } else {
            frame_projection.beta_dim()
        };
        let width = border_dim + mass_dim;
        let whitening_metric = self
            .row_metric
            .as_ref()
            .filter(|metric| metric.whitens_likelihood());
        // Rows fold in blocks of `width` scores, so the transient score block
        // never outgrows the meat itself.
        let block_rows = width.max(1);
        let mut meat = CentredOuterProduct::new(width);
        let mut assignments = vec![0.0_f64; self.k_atoms()];
        let mut decoded = vec![0.0_f64; p];
        let mut residual = Array1::<f64>::zeros(p);
        let mut score = Array1::<f64>::zeros(frame_projection.beta_dim());
        let mut block_start = 0usize;
        while block_start < n {
            let block_end = (block_start + block_rows).min(n);
            let mut scores = Array2::<f64>::zeros((block_end - block_start, width));
            for row in block_start..block_end {
                self.assignment
                    .try_assignments_row_into(row, &mut assignments)?;
                for c in 0..p {
                    residual[c] = -target[[row, c]];
                }
                for (atom_idx, atom) in self.atoms.iter().enumerate() {
                    atom.fill_decoded_row(row, &mut decoded);
                    for c in 0..p {
                        residual[c] += assignments[atom_idx] * decoded[c];
                    }
                }
                let metric_residual = match whitening_metric {
                    Some(metric) => metric.apply_metric_row(row, residual.view()),
                    None => residual.to_vec(),
                };
                let row_weight = self.row_loss_weights.as_ref().map_or(1.0, |w| w[row]);
                score.fill(0.0);
                for (atom_idx, atom) in self.atoms.iter().enumerate() {
                    let load = row_weight * assignments[atom_idx];
                    if load == 0.0 {
                        continue;
                    }
                    for basis_col in 0..atom.basis_size() {
                        let basis_load = load * atom.basis_values[[row, basis_col]];
                        let base = frame_projection.beta_offsets[atom_idx] + basis_col * p;
                        for c in 0..p {
                            score[base + c] = basis_load * metric_residual[c];
                        }
                    }
                }
                let mut out = scores.row_mut(row - block_start);
                if frames_active {
                    out.slice_mut(s![..border_dim])
                        .assign(&frame_projection.project_border_vec(score.view()));
                } else {
                    out.slice_mut(s![..border_dim]).assign(&score);
                }
                for k in 0..mass_dim {
                    out[border_dim + k] = -mass_rows[row * mass_dim + k];
                }
            }
            meat.add_rows(&scores);
            block_start = block_end;
        }
        Ok(RowSandwichMeat {
            meat: meat.finish(),
            mass_carriers,
        })
    }
}

/// Streaming centred outer product `Σ_i (g_i − ḡ)(g_i − ḡ)ᵀ` of row summands,
/// accumulated as `Σ_i g_i g_iᵀ − (Σ_i g_i)(Σ_i g_i)ᵀ / n`, so no more than one
/// block of summands is held. It is invariant under adding one constant vector
/// to every summand.
pub(crate) struct CentredOuterProduct {
    outer: Array2<f64>,
    total: Array1<f64>,
    rows: usize,
}

impl CentredOuterProduct {
    pub(crate) fn new(width: usize) -> Self {
        Self {
            outer: Array2::<f64>::zeros((width, width)),
            total: Array1::<f64>::zeros(width),
            rows: 0,
        }
    }

    /// Folds in one block of summands, one per row.
    pub(crate) fn add_rows(&mut self, block: &Array2<f64>) {
        self.outer += &fast_atb(block, block);
        for summand in block.rows() {
            self.total += &summand;
        }
        self.rows += block.nrows();
    }

    pub(crate) fn finish(self) -> Array2<f64> {
        let Self {
            mut outer,
            total,
            rows,
        } = self;
        if rows > 0 {
            let n = rows as f64;
            for r in 0..total.len() {
                for c in 0..total.len() {
                    outer[[r, c]] -= total[r] * total[c] / n;
                }
            }
        }
        symmetrized(outer)
    }
}

/// Per-atom blocks of `[A⁺]_ββ = W_β W_βᵀ` and of the row sandwich over the joint
/// estimating equations. `border_factor` is `W_β = V_β[:, retained]·Λ^{−½}`, and
/// column `k` of `mass_projections` is `z_k = Λ^{−½}V_retᵀ(c_k u_k)`, so the
/// response of `β` to the `k`-th aggregate-mass equation is `−W_β z_k`. `meat` is
/// `Σ_i g̃_i g̃_iᵀ` over `(border, masses)`. The whole border inverse is formed
/// once, because the meat couples atoms and channels.
pub(crate) fn observed_information_border_blocks(
    border_factor: ArrayView2<'_, f64>,
    mass_projections: ArrayView2<'_, f64>,
    meat: ArrayView2<'_, f64>,
    ranges: &[std::ops::Range<usize>],
) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), String> {
    let border_dim = border_factor.nrows();
    let mass_dim = mass_projections.ncols();
    let width = border_dim + mass_dim;
    if mass_projections.nrows() != border_factor.ncols() || meat.dim() != (width, width) {
        return Err(format!(
            "exact observed-information shape covariance: border factor {:?}, mass projections \
             {:?} and meat {:?} disagree",
            border_factor.dim(),
            mass_projections.dim(),
            meat.dim()
        ));
    }
    let border_inverse = symmetrized(fast_ab(&border_factor, &border_factor.t()));
    let mut response = Array2::<f64>::zeros((border_dim, width));
    response
        .slice_mut(s![.., ..border_dim])
        .assign(&border_inverse);
    if mass_dim > 0 {
        let mass_response = fast_ab(&border_factor, &mass_projections);
        response
            .slice_mut(s![.., border_dim..])
            .assign(&mass_response.mapv(|value| -value));
    }
    let sandwich = symmetrized(fast_ab(&fast_ab(&response, &meat), &response.t()));
    let mut blocks = Vec::with_capacity(ranges.len());
    let mut robust_blocks = Vec::with_capacity(ranges.len());
    for (atom, range) in ranges.iter().enumerate() {
        if range.end > border_dim {
            return Err(format!(
                "exact observed-information shape covariance: atom {atom} border range \
                 {range:?} exceeds the border dimension {border_dim}"
            ));
        }
        let block = border_inverse
            .slice(s![range.clone(), range.clone()])
            .to_owned();
        let robust = sandwich.slice(s![range.clone(), range.clone()]).to_owned();
        if !block.iter().chain(robust.iter()).all(|value| value.is_finite()) {
            return Err(format!(
                "exact observed-information shape covariance: atom {atom} block is non-finite"
            ));
        }
        blocks.push(block);
        robust_blocks.push(robust);
    }
    Ok((blocks, robust_blocks))
}

/// Average a matrix with its transpose, clearing product rounding asymmetry.
fn symmetrized(mut matrix: Array2<f64>) -> Array2<f64> {
    let width = matrix.nrows();
    for row in 0..width {
        for col in (row + 1)..width {
            let average = 0.5 * (matrix[[row, col]] + matrix[[col, row]]);
            matrix[[row, col]] = average;
            matrix[[col, row]] = average;
        }
    }
    matrix
}

#[cfg(test)]
mod robust_shape_band_tests {
    use super::*;
    use ndarray::array;

    /// Gauss–Jordan inverse with partial pivoting: an oracle that shares no code
    /// with the eigendecomposition the production bread reads.
    fn dense_inverse(matrix: &Array2<f64>) -> Array2<f64> {
        let n = matrix.nrows();
        let mut work = matrix.clone();
        let mut inverse = Array2::<f64>::eye(n);
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&i, &j| work[[i, col]].abs().total_cmp(&work[[j, col]].abs()))
                .expect("non-empty pivot column");
            for c in 0..n {
                work.swap([col, c], [pivot, c]);
                inverse.swap([col, c], [pivot, c]);
            }
            let diag = work[[col, col]];
            assert!(diag.abs() > 1.0e-12, "oracle matrix must be invertible");
            for c in 0..n {
                work[[col, c]] /= diag;
                inverse[[col, c]] /= diag;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = work[[r, col]];
                for c in 0..n {
                    let work_pivot = work[[col, c]];
                    let inverse_pivot = inverse[[col, c]];
                    work[[r, c]] -= factor * work_pivot;
                    inverse[[r, c]] -= factor * inverse_pivot;
                }
            }
        }
        inverse
    }

    /// `V Λ^{−½}` restricted to the border rows `offset..`, over every direction
    /// of a positive definite test operator.
    fn border_factor(operator: &Array2<f64>, offset: usize) -> Array2<f64> {
        let (eigenvalues, eigenvectors) = operator
            .eigh(Side::Lower)
            .expect("symmetric eigendecomposition");
        assert!(
            eigenvalues.iter().all(|&lambda| lambda > 0.0),
            "the test operator must be positive definite"
        );
        Array2::from_shape_fn((operator.nrows() - offset, eigenvalues.len()), |(row, col)| {
            eigenvectors[[offset + row, col]] / eigenvalues[col].sqrt()
        })
    }

    /// Linear heteroskedastic regression: the observed information is `XᵀX`, the
    /// meat `Σ_i x_i x_iᵀ e_i²` of the OLS scores, and the sandwich must be White's
    /// HC0 `(XᵀX)⁻¹ Xᵀ diag(e²) X (XᵀX)⁻¹`, evaluated here through the 2×2
    /// adjugate. The residuals sit at the extreme and central design points, so
    /// HC0 and the model-based `σ̂²(XᵀX)⁻¹` part materially.
    #[test]
    fn row_sandwich_recovers_hc0_heteroskedastic_regression() {
        let x = [-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [-2.0_f64, -1.0, 1.0, 1.0, 5.0, 7.0, 10.0];
        let n = x.len();
        let gram = [
            [n as f64, x.iter().sum::<f64>()],
            [x.iter().sum::<f64>(), x.iter().map(|v| v * v).sum::<f64>()],
        ];
        let det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
        let gram_inverse = [
            [gram[1][1] / det, -gram[0][1] / det],
            [-gram[1][0] / det, gram[0][0] / det],
        ];
        let xty = [
            y.iter().sum::<f64>(),
            x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>(),
        ];
        let beta = [
            gram_inverse[0][0] * xty[0] + gram_inverse[0][1] * xty[1],
            gram_inverse[1][0] * xty[0] + gram_inverse[1][1] * xty[1],
        ];
        let residuals: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| yi - beta[0] - beta[1] * xi)
            .collect();
        let mut meat = Array2::<f64>::zeros((2, 2));
        for (&xi, &ei) in x.iter().zip(residuals.iter()) {
            let design_row = [1.0, xi];
            for a in 0..2 {
                for b in 0..2 {
                    meat[[a, b]] += design_row[a] * design_row[b] * ei * ei;
                }
            }
        }
        let mut hc0 = [[0.0_f64; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        hc0[a][b] += gram_inverse[a][c] * meat[[c, d]] * gram_inverse[d][b];
                    }
                }
            }
        }

        let information = array![[gram[0][0], gram[0][1]], [gram[1][0], gram[1][1]]];
        let (blocks, robust) = observed_information_border_blocks(
            border_factor(&information, 0).view(),
            Array2::<f64>::zeros((2, 0)).view(),
            meat.view(),
            &[0..2],
        )
        .expect("border blocks");
        for a in 0..2 {
            for b in 0..2 {
                assert!(
                    (blocks[0][[a, b]] - gram_inverse[a][b]).abs()
                        <= 1.0e-11 * gram_inverse[a][b].abs().max(1.0),
                    "bread [{a},{b}] = {} vs (XᵀX)⁻¹ {}",
                    blocks[0][[a, b]],
                    gram_inverse[a][b]
                );
                assert!(
                    (robust[0][[a, b]] - hc0[a][b]).abs() <= 1.0e-11 * hc0[a][b].abs().max(1.0),
                    "sandwich [{a},{b}] = {} vs HC0 {}",
                    robust[0][[a, b]],
                    hc0[a][b]
                );
            }
        }
        let sigma2 = residuals.iter().map(|e| e * e).sum::<f64>() / (n - 2) as f64;
        let model_slope_variance = sigma2 * gram_inverse[1][1];
        assert!(
            (hc0[1][1] / model_slope_variance - 1.0).abs() > 0.25,
            "fixture must separate HC0 slope variance {} from the model-based {model_slope_variance}",
            hc0[1][1]
        );
    }

    /// Latent rows coupled to a border of two atoms (one basis function, two
    /// output channels each) and a meat correlated across atoms and channels. The
    /// robust blocks must be those of the JOINT sandwich `A⁻¹ J A⁻ᵀ`, formed from
    /// the materialized bordered `A` and its dense inverse. Inverting the isolated
    /// border `H_ββ`, keeping only each atom's own bread block, or inverting one
    /// channel's Schur block `A_c⁻¹ J_cc A_c⁻¹` must each miss it.
    #[test]
    fn row_sandwich_is_the_joint_marginal_under_latent_atom_and_channel_coupling() {
        let (n, d, k) = (3usize, 2usize, 4usize);
        let htt = [
            array![[3.0_f64, 0.4], [0.4, 2.5]],
            array![[2.8_f64, -0.3], [-0.3, 3.2]],
            array![[3.5_f64, 0.2], [0.2, 2.2]],
        ];
        let htbeta = [
            array![[1.2_f64, 0.5, -0.4, 0.9], [0.3, -1.1, 0.8, 0.2]],
            array![[-0.6_f64, 0.9, 1.0, -0.5], [1.1, 0.4, -0.3, 1.3]],
            array![[0.8_f64, -0.7, 0.6, 1.0], [-0.9, 1.2, 0.5, -0.4]],
        ];
        let hbb = array![
            [6.0_f64, 1.5, 0.8, 0.3],
            [1.5, 5.5, 0.4, 0.9],
            [0.8, 0.4, 6.5, 1.2],
            [0.3, 0.9, 1.2, 5.8]
        ];
        let border = n * d;
        let dim = border + k;
        let mut joint = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            joint
                .slice_mut(s![base..base + d, base..base + d])
                .assign(&htt[i]);
            joint
                .slice_mut(s![base..base + d, border..])
                .assign(&htbeta[i]);
            joint
                .slice_mut(s![border.., base..base + d])
                .assign(&htbeta[i].t());
        }
        joint.slice_mut(s![border.., border..]).assign(&hbb);
        // Border scores in the per-atom `(atom, channel)` layout; every score
        // loads both atoms and both channels.
        let scores = array![
            [1.0_f64, 0.8, -0.5, 0.3],
            [-0.4, 1.2, 0.9, -0.7],
            [0.6, -0.2, 1.1, 0.9],
            [0.9, 1.0, 0.2, -1.2]
        ];
        let meat = scores.t().dot(&scores);
        let ranges = [0..2, 2..4];
        let (blocks, robust) = observed_information_border_blocks(
            border_factor(&joint, border).view(),
            Array2::<f64>::zeros((dim, 0)).view(),
            meat.view(),
            &ranges,
        )
        .expect("border blocks");

        let joint_inverse = dense_inverse(&joint);
        let border_inverse = joint_inverse.slice(s![border.., border..]).to_owned();
        let mut joint_meat = Array2::<f64>::zeros((dim, dim));
        joint_meat.slice_mut(s![border.., border..]).assign(&meat);
        let expected = joint_inverse
            .dot(&joint_meat)
            .dot(&joint_inverse)
            .slice(s![border.., border..])
            .to_owned();
        let scale = expected.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 0.0, "the joint sandwich must not be vacuous");
        for (atom, range) in ranges.iter().enumerate() {
            for (local_r, r) in range.clone().enumerate() {
                for (local_c, c) in range.clone().enumerate() {
                    assert!(
                        (blocks[atom][[local_r, local_c]] - border_inverse[[r, c]]).abs()
                            <= 1.0e-10 * scale.max(1.0),
                        "atom {atom} bread [{r},{c}] = {} vs joint oracle {}",
                        blocks[atom][[local_r, local_c]],
                        border_inverse[[r, c]]
                    );
                    assert!(
                        (robust[atom][[local_r, local_c]] - expected[[r, c]]).abs()
                            <= 1.0e-10 * scale,
                        "atom {atom} sandwich [{r},{c}] = {} vs joint oracle {}",
                        robust[atom][[local_r, local_c]],
                        expected[[r, c]]
                    );
                }
            }
        }

        let hbb_inverse = dense_inverse(&hbb);
        let isolated = hbb_inverse.dot(&meat).dot(&hbb_inverse);
        let mut per_atom_bread = Array2::<f64>::zeros((k, k));
        for range in &ranges {
            per_atom_bread
                .slice_mut(s![range.clone(), range.clone()])
                .assign(&border_inverse.slice(s![range.clone(), range.clone()]));
        }
        let per_atom = per_atom_bread.dot(&meat).dot(&per_atom_bread);
        let schur = dense_inverse(&border_inverse);
        let mut isolated_departure = 0.0_f64;
        let mut per_atom_departure = 0.0_f64;
        let mut within_channel_departure = 0.0_f64;
        for channel in 0..2 {
            let idx = [channel, 2 + channel];
            let a_c = Array2::from_shape_fn((2, 2), |(i, j)| schur[[idx[i], idx[j]]]);
            let j_cc = Array2::from_shape_fn((2, 2), |(i, j)| meat[[idx[i], idx[j]]]);
            let a_c_inverse = dense_inverse(&a_c);
            let within = a_c_inverse.dot(&j_cc).dot(&a_c_inverse);
            for (local, &global) in idx.iter().enumerate() {
                let joint_variance = expected[[global, global]];
                within_channel_departure = within_channel_departure
                    .max((within[[local, local]] / joint_variance - 1.0).abs());
                isolated_departure = isolated_departure
                    .max((isolated[[global, global]] / joint_variance - 1.0).abs());
                per_atom_departure = per_atom_departure
                    .max((per_atom[[global, global]] / joint_variance - 1.0).abs());
            }
        }
        assert!(
            isolated_departure > 0.1,
            "an isolated-border sandwich must miss the latent coupling (departure \
             {isolated_departure:.3e})"
        );
        assert!(
            per_atom_departure > 0.05,
            "a per-atom bread must miss the cross-atom coupling (departure \
             {per_atom_departure:.3e})"
        );
        assert!(
            within_channel_departure > 0.05,
            "a within-channel sandwich must miss the cross-channel coupling (departure \
             {within_channel_departure:.3e})"
        );
    }

    /// A cross-row rank-one block `c u uᵀ` in `A`, the shape the ordered
    /// Beta--Bernoulli prior's aggregate mass installs. The robust block must be
    /// that of the augmented estimating equations `(G_t, G_β, M − Σ_i q_i)`, formed
    /// from their dense non-symmetric Jacobian `Ã` and its Gauss–Jordan inverse.
    /// Eliminating the mass equation must recover `A`, and dropping the masses'
    /// meat must miss the robust block.
    #[test]
    fn row_sandwich_includes_the_aggregate_mass_equations() {
        let (n, d, k) = (3usize, 2usize, 2usize);
        let border = n * d;
        let dim = border + k;
        let htt = [
            array![[3.0_f64, 0.4], [0.4, 2.5]],
            array![[2.8_f64, -0.3], [-0.3, 3.2]],
            array![[3.5_f64, 0.2], [0.2, 2.2]],
        ];
        let htbeta = [
            array![[1.1_f64, -0.4], [0.3, 0.9]],
            array![[-0.7_f64, 0.8], [1.0, -0.2]],
            array![[0.6_f64, 0.5], [-0.8, 1.1]],
        ];
        let mut local = Array2::<f64>::zeros((dim, dim));
        for i in 0..n {
            let base = i * d;
            local
                .slice_mut(s![base..base + d, base..base + d])
                .assign(&htt[i]);
            local
                .slice_mut(s![base..base + d, border..])
                .assign(&htbeta[i]);
            local
                .slice_mut(s![border.., base..base + d])
                .assign(&htbeta[i].t());
        }
        local
            .slice_mut(s![border.., border..])
            .assign(&array![[5.0_f64, 1.2], [1.2, 4.5]]);
        // The mass derivative lives on each row's logit slot; the coefficient is
        // negative like the integrated prior's `d²L/dM²`.
        let coefficient = -0.6_f64;
        let carrier = [(0usize, 0.8_f64), (2, -0.5), (4, 0.9)];
        let mut u = Array1::<f64>::zeros(dim);
        for &(index, value) in &carrier {
            u[index] = value;
        }
        let joint = Array2::from_shape_fn((dim, dim), |(r, c)| {
            local[[r, c]] + coefficient * u[r] * u[c]
        });
        // Border scores then the centred mass summand, for four rows.
        let scores = array![
            [1.0_f64, 0.8, 0.3],
            [-0.4, 1.2, -0.5],
            [0.6, -0.2, 0.9],
            [0.9, 1.0, -0.7]
        ];
        let meat = scores.t().dot(&scores);

        let (eigenvalues, eigenvectors) = joint
            .eigh(Side::Lower)
            .expect("symmetric eigendecomposition");
        assert!(
            eigenvalues.iter().all(|&lambda| lambda > 0.0),
            "the test operator must be positive definite"
        );
        let rank = eigenvalues.len();
        let border_factor = Array2::from_shape_fn((k, rank), |(row, col)| {
            eigenvectors[[border + row, col]] / eigenvalues[col].sqrt()
        });
        let mass_projections = Array2::from_shape_fn((rank, 1), |(col, _)| {
            carrier
                .iter()
                .map(|&(index, value)| coefficient * value * eigenvectors[[index, col]])
                .sum::<f64>()
                / eigenvalues[col].sqrt()
        });
        let (blocks, robust) = observed_information_border_blocks(
            border_factor.view(),
            mass_projections.view(),
            meat.view(),
            &[0..k],
        )
        .expect("border blocks");

        // Jacobian over `(t, β, M)`: the mass enters the t rows through `c u`, and
        // its own equation `M − Σ_i q_i` has t-derivative `−u`.
        let mut augmented = Array2::<f64>::zeros((dim + 1, dim + 1));
        augmented.slice_mut(s![..dim, ..dim]).assign(&local);
        for &(index, value) in &carrier {
            augmented[[index, dim]] = coefficient * value;
            augmented[[dim, index]] = -value;
        }
        augmented[[dim, dim]] = 1.0;
        let augmented_inverse = dense_inverse(&augmented);
        let joint_inverse = dense_inverse(&joint);
        for r in 0..k {
            for c in 0..k {
                let reference = joint_inverse[[border + r, border + c]];
                assert!(
                    (augmented_inverse[[border + r, border + c]] - reference).abs()
                        <= 1.0e-10 * reference.abs().max(1.0),
                    "eliminating the mass equation must recover A at [{r},{c}]"
                );
                assert!(
                    (blocks[0][[r, c]] - reference).abs() <= 1.0e-10 * reference.abs().max(1.0),
                    "bread [{r},{c}] = {} vs A⁻¹ {reference}",
                    blocks[0][[r, c]]
                );
            }
        }
        let mut augmented_meat = Array2::<f64>::zeros((dim + 1, dim + 1));
        augmented_meat
            .slice_mut(s![border.., border..])
            .assign(&meat);
        let expected = augmented_inverse
            .dot(&augmented_meat)
            .dot(&augmented_inverse.t())
            .slice(s![border..dim, border..dim])
            .to_owned();
        let scale = expected.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 0.0, "the augmented sandwich must not be vacuous");
        for r in 0..k {
            for c in 0..k {
                assert!(
                    (robust[0][[r, c]] - expected[[r, c]]).abs() <= 1.0e-10 * scale,
                    "sandwich [{r},{c}] = {} vs augmented oracle {}",
                    robust[0][[r, c]],
                    expected[[r, c]]
                );
            }
        }
        let border_inverse = joint_inverse.slice(s![border.., border..]).to_owned();
        let border_meat = meat.slice(s![..k, ..k]).to_owned();
        let without_masses = border_inverse.dot(&border_meat).dot(&border_inverse);
        let departure = (0..k)
            .map(|i| (without_masses[[i, i]] / expected[[i, i]] - 1.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            departure > 0.05,
            "a border-only meat must miss the aggregate-mass variability (departure \
             {departure:.3e})"
        );
    }

    /// The centred meat must equal the explicit Gram of the mean-removed summands
    /// however the rows are folded, and must not change when one constant vector
    /// is added to every summand, the shape a penalty gradient's share takes.
    /// The uncentred outer product must fold that constant in.
    #[test]
    fn centred_meat_is_invariant_under_a_constant_added_to_every_summand() {
        let summands = array![
            [1.0_f64, 0.8, 0.3],
            [-0.4, 1.2, -0.5],
            [0.6, -0.2, 0.9],
            [0.9, 1.0, -0.7],
            [-1.1, 0.3, 0.4]
        ];
        let offset = array![2.5_f64, -1.75, 0.6];
        let shifted = &summands + &offset;
        let mut folded = CentredOuterProduct::new(3);
        folded.add_rows(&summands.slice(s![..2, ..]).to_owned());
        folded.add_rows(&summands.slice(s![2.., ..]).to_owned());
        let centred = folded.finish();
        let mut whole = CentredOuterProduct::new(3);
        whole.add_rows(&shifted);
        let shifted_centred = whole.finish();
        let mean = summands
            .mean_axis(ndarray::Axis(0))
            .expect("the fixture has rows");
        let deviations = &summands - &mean;
        let explicit = deviations.t().dot(&deviations);
        let scale = explicit.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (centred[[r, c]] - explicit[[r, c]]).abs() <= 1.0e-12 * scale,
                    "folded meat [{r},{c}] = {} vs explicit centred Gram {}",
                    centred[[r, c]],
                    explicit[[r, c]]
                );
                assert!(
                    (shifted_centred[[r, c]] - explicit[[r, c]]).abs() <= 1.0e-12 * scale,
                    "shifted meat [{r},{c}] = {} vs explicit centred Gram {}",
                    shifted_centred[[r, c]],
                    explicit[[r, c]]
                );
            }
        }
        let uncentred = shifted.t().dot(&shifted);
        let departure = (0..3)
            .map(|i| (uncentred[[i, i]] / explicit[[i, i]] - 1.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            departure > 1.0,
            "the uncentred outer product must fold the constant in (departure {departure:.3e})"
        );
    }

    fn converged_tiny_state(
        install: Option<Box<dyn FnOnce(&mut SaeManifoldTerm)>>,
    ) -> (
        SaeManifoldTerm,
        Array2<f64>,
        SaeManifoldRho,
        ArrowFactorCache,
        SaeReconstructionDispersion,
    ) {
        let (mut term, target, mut rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -1.0;
            }
        }
        if let Some(install) = install {
            install(&mut term);
        }
        converged_state(term, target, rho, 40)
    }

    fn converged_state(
        mut term: SaeManifoldTerm,
        target: Array2<f64>,
        rho: SaeManifoldRho,
        inner_max_iter: usize,
    ) -> (
        SaeManifoldTerm,
        Array2<f64>,
        SaeManifoldRho,
        ArrowFactorCache,
        SaeReconstructionDispersion,
    ) {
        let (_cost, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                inner_max_iter,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("converged joint cache");
        let residual = term
            .reconstruction_residual(target.view(), &rho)
            .expect("reconstruction residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("dispersion");
        (term, target, rho, cache, dispersion)
    }

    /// The oracle's robust bands with the aggregate-mass estimating equations and
    /// with the border data scores alone. They coincide when no prior couples rows.
    struct OracleBands {
        joint: Vec<Array2<f64>>,
        border_only: Vec<Array2<f64>>,
    }

    /// Independent oracle for the robust band. Per-row decoder scores come from
    /// central differences of each row's own weighted data fit through the public
    /// forward map (the row fit is quadratic in the decoder, so the difference is
    /// exact up to rounding). The bread is the pseudo-inverse of the materialized
    /// observed information over its `identified_rank` largest directions, which
    /// must be separated from the rest by a clear spectral gap. The push-forward is
    /// the explicit basis sum.
    ///
    /// Under the ordered Beta--Bernoulli prior the masses `M_k = Σ_i q_ik` enter
    /// as estimating equations. `q_ik = w_i a_ik` is read off the forward gates and
    /// `u_ik = ∂q_ik/∂ℓ_ik` is their central difference. The information's
    /// cross-row block must be `Σ_k c_k u_k u_kᵀ` over the logit slots, and `c_k` is
    /// its least-squares fit. The response of `β` to mass `k` is `−[A⁺ c_k u_k]_β`.
    fn finite_difference_robust_bands(
        term: &SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        identified_rank: usize,
    ) -> OracleBands {
        let n = term.n_obs();
        let p = term.output_dim();
        let beta_dim = term.beta_dim();
        assert!(
            !term.frames_active() && cache.k == beta_dim,
            "the oracle reads the full decoder layout"
        );
        let row_fits = |state: &SaeManifoldTerm| -> Vec<f64> {
            let fitted = state.try_fitted_for_rho(rho).expect("fitted reconstruction");
            (0..n)
                .map(|row| {
                    let r = &fitted.row(row) - &target.row(row);
                    let weight = state.row_loss_weights().map_or(1.0, |w| w[row]);
                    let quadratic = match state.row_metric.as_ref() {
                        Some(metric) if metric.whitens_likelihood() => {
                            metric.quad_form(row, r.view())
                        }
                        _ => r.dot(&r),
                    };
                    0.5 * weight * quadratic
                })
                .collect()
        };
        let offsets = term.beta_offsets();
        let step = 1.0e-5;
        let mut scores = Array2::<f64>::zeros((n, beta_dim));
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            for basis_col in 0..atom.basis_size() {
                for c in 0..p {
                    let mut plus = term.clone();
                    plus.atoms[atom_idx].decoder_coefficients_mut()[[basis_col, c]] += step;
                    let mut minus = term.clone();
                    minus.atoms[atom_idx].decoder_coefficients_mut()[[basis_col, c]] -= step;
                    let (fits_plus, fits_minus) = (row_fits(&plus), row_fits(&minus));
                    for row in 0..n {
                        scores[[row, offsets[atom_idx] + basis_col * p + c]] =
                            (fits_plus[row] - fits_minus[row]) / (2.0 * step);
                    }
                }
            }
        }
        let total_t = cache.delta_t_len();
        let (information, _gap_border) = term
            .materialize_exact_hessian_dense_with_gap_border(rho, target, cache)
            .expect("materialized observed information");
        // #2933 F07 — the bread is classified in the evidence factor's pencil `(A, Φ)`, whose
        // retained pseudo-inverse `Σ wᵢwᵢᵀ/μᵢ` production reports; an ordinary eigenbasis of
        // `A` names a different, coordinate-dependent pseudo-inverse once a direction drops.
        let oracle = crate::manifold::tests::PencilOracle::new(&information, cache);
        let (eigenvalues, eigenvectors) = (&oracle.values, &oracle.vectors);
        let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
        order.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));
        let (retained, dropped) = order.split_at(identified_rank);
        let smallest_retained = retained
            .iter()
            .map(|&i| eigenvalues[i])
            .fold(f64::INFINITY, f64::min);
        let largest_dropped = dropped
            .iter()
            .map(|&i| eigenvalues[i].abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "ROBUST_BAND_ORACLE identified_rank={identified_rank} smallest_retained=\
             {smallest_retained:.6e} largest_dropped={largest_dropped:.6e}"
        );
        assert!(
            smallest_retained > 1.0e3 * largest_dropped,
            "the oracle needs a spectral gap at the identified rank {identified_rank}: smallest \
             retained {smallest_retained:.3e}, largest dropped {largest_dropped:.3e}"
        );

        let k_atoms = term.k_atoms();
        let mass_dim = if matches!(
            term.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        ) {
            k_atoms
        } else {
            0
        };
        let mut mass_rows = Array2::<f64>::zeros((n, mass_dim));
        let mut carriers = Array2::<f64>::zeros((total_t, mass_dim));
        let mut slot_rows = vec![0usize; total_t];
        for row in 0..n {
            for slot in cache.row_offsets[row]..cache.row_offsets[row + 1] {
                slot_rows[slot] = row;
            }
        }
        if mass_dim > 0 {
            let gate_step = 1.0e-6;
            let mut probe = term.clone();
            for row in 0..n {
                let weight = term.row_loss_weights().map_or(1.0, |w| w[row]);
                let gates = term
                    .assignment
                    .try_assignments_row(row)
                    .expect("forward gates");
                let layout = term
                    .row_vars_for_cache_row(row, cache)
                    .expect("cache row layout");
                for (local, variable) in layout.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *variable else {
                        continue;
                    };
                    mass_rows[[row, atom]] = weight * gates[atom];
                    let logit = probe.assignment.logits[[row, atom]];
                    probe.assignment.logits[[row, atom]] = logit + gate_step;
                    let plus = probe
                        .assignment
                        .try_assignments_row(row)
                        .expect("forward gates")[atom];
                    probe.assignment.logits[[row, atom]] = logit - gate_step;
                    let minus = probe
                        .assignment
                        .try_assignments_row(row)
                        .expect("forward gates")[atom];
                    probe.assignment.logits[[row, atom]] = logit;
                    carriers[[cache.row_offsets[row] + local, atom]] =
                        weight * (plus - minus) / (2.0 * gate_step);
                }
            }
        }
        let mut coefficients = vec![0.0_f64; mass_dim];
        for (atom, coefficient) in coefficients.iter_mut().enumerate() {
            let (mut numerator, mut denominator) = (0.0_f64, 0.0_f64);
            for a in 0..total_t {
                for b in 0..total_t {
                    if slot_rows[a] != slot_rows[b] {
                        let product = carriers[[a, atom]] * carriers[[b, atom]];
                        numerator += information[[a, b]] * product;
                        denominator += product * product;
                    }
                }
            }
            assert!(
                denominator > 0.0,
                "atom {atom}: the mass derivative must be live"
            );
            *coefficient = numerator / denominator;
        }
        // The masses must explain every cross-row entry, and no other cross-row
        // coupling may exist, or the oracle's estimating equations are incomplete.
        let mut cross_row_signal = 0.0_f64;
        let mut cross_row_misfit = 0.0_f64;
        for a in 0..total_t {
            for b in 0..total_t {
                if slot_rows[a] != slot_rows[b] {
                    let model = (0..mass_dim)
                        .map(|atom| coefficients[atom] * carriers[[a, atom]] * carriers[[b, atom]])
                        .sum::<f64>();
                    cross_row_signal = cross_row_signal.max(information[[a, b]].abs());
                    cross_row_misfit = cross_row_misfit.max((information[[a, b]] - model).abs());
                }
            }
        }
        eprintln!(
            "ROBUST_BAND_ORACLE masses={mass_dim} coefficients={coefficients:?} \
             cross_row_signal={cross_row_signal:.6e} cross_row_misfit={cross_row_misfit:.6e}"
        );
        assert!(
            cross_row_misfit <= 1.0e-6 * cross_row_signal,
            "the cross-row information {cross_row_signal:.3e} is not the mass carriers' \
             (misfit {cross_row_misfit:.3e})"
        );

        let mut border_inverse = Array2::<f64>::zeros((beta_dim, beta_dim));
        let mut mass_response = Array2::<f64>::zeros((beta_dim, mass_dim));
        for &index in retained {
            let projections: Vec<f64> = (0..mass_dim)
                .map(|atom| {
                    coefficients[atom]
                        * (0..total_t)
                            .map(|slot| eigenvectors[[slot, index]] * carriers[[slot, atom]])
                            .sum::<f64>()
                })
                .collect();
            for r in 0..beta_dim {
                let left = eigenvectors[[total_t + r, index]] / eigenvalues[index];
                for c in 0..beta_dim {
                    border_inverse[[r, c]] += left * eigenvectors[[total_t + c, index]];
                }
                for (atom, &projection) in projections.iter().enumerate() {
                    mass_response[[r, atom]] -= left * projection;
                }
            }
        }
        // Row summands `(s_i, −q_i)`, each centred about its row mean.
        let width = beta_dim + mass_dim;
        let mut estimating = Array2::<f64>::zeros((n, width));
        estimating.slice_mut(s![.., ..beta_dim]).assign(&scores);
        for atom in 0..mass_dim {
            for row in 0..n {
                estimating[[row, beta_dim + atom]] = -mass_rows[[row, atom]];
            }
        }
        let means = estimating
            .mean_axis(ndarray::Axis(0))
            .expect("the oracle has rows");
        let deviations = &estimating - &means;
        let joint_meat = deviations.t().dot(&deviations);
        let border_meat = joint_meat.slice(s![..beta_dim, ..beta_dim]).to_owned();
        let border_only = border_inverse.dot(&border_meat).dot(&border_inverse);
        let mut response = Array2::<f64>::zeros((beta_dim, width));
        response
            .slice_mut(s![.., ..beta_dim])
            .assign(&border_inverse);
        response
            .slice_mut(s![.., beta_dim..])
            .assign(&mass_response);
        let joint = response.dot(&joint_meat).dot(&response.t());
        let frobenius = |matrix: ndarray::ArrayView2<'_, f64>| {
            matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
        };
        eprintln!(
            "ROBUST_BAND_MEAT border_meat_frob={:.6e} mass_meat_frob={:.6e} \
             cross_meat_frob={:.6e} border_sandwich_frob={:.6e} mass_share_frob={:.6e}",
            frobenius(joint_meat.slice(s![..beta_dim, ..beta_dim])),
            frobenius(joint_meat.slice(s![beta_dim.., beta_dim..])),
            frobenius(joint_meat.slice(s![..beta_dim, beta_dim..])),
            frobenius(border_only.view()),
            frobenius((&joint - &border_only).view())
        );
        let push_forward = |covariance: &Array2<f64>| -> Vec<Array2<f64>> {
            term.atoms
                .iter()
                .enumerate()
                .map(|(atom_idx, atom)| {
                    let m = atom.basis_size();
                    let mut band = Array2::<f64>::zeros((n, p));
                    for row in 0..n {
                        for c in 0..p {
                            let mut variance = 0.0;
                            for b1 in 0..m {
                                for b2 in 0..m {
                                    variance += atom.basis_values[[row, b1]]
                                        * atom.basis_values[[row, b2]]
                                        * covariance[[
                                            offsets[atom_idx] + b1 * p + c,
                                            offsets[atom_idx] + b2 * p + c,
                                        ]];
                                }
                            }
                            band[[row, c]] = variance.max(0.0).sqrt();
                        }
                    }
                    band
                })
                .collect()
        };
        OracleBands {
            joint: push_forward(&joint),
            border_only: push_forward(&border_only),
        }
    }

    /// The agreement budget for a band entry `sd`: the relative precision the
    /// oracle's rounding supports, plus an absolute floor for a vanishing entry.
    fn band_agreement_budget(sd: f64) -> f64 {
        1.0e-6 * sd + 1.0e-12
    }

    /// How far apart the robust band's two candidates sit, in agreement budgets.
    struct RobustBandSeparation {
        /// `max |border-only − oracle| / budget(oracle)`: how badly a band that
        /// dropped the mass summands would miss the oracle.
        omission_ratio: f64,
        /// `max |production − border-only| / budget(border-only)`.
        production_departure_ratio: f64,
    }

    /// Asserts the user-facing robust band against the oracle's joint sandwich,
    /// within one agreement budget per entry, and reports its separation from the
    /// border-only sandwich.
    fn assert_robust_band_matches_oracle(
        term: &SaeManifoldTerm,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        dispersion: SaeReconstructionDispersion,
    ) -> RobustBandSeparation {
        let geometry = term
            .materialize_exact_stationarity_geometry(rho, target, cache)
            .expect("exact stationarity geometry at the converged state");
        let information = term
            .exact_observed_information_shape_covariance(&geometry, rho, target, cache)
            .expect("exact observed information at the converged state");
        let uncertainty = term
            .assemble_shape_uncertainty(&information, dispersion)
            .expect("joint shape bands");
        let identified_rank = match uncertainty.operator {
            SaeShapeCovarianceOperator::ObservedInformation {
                identified_rank, ..
            } => identified_rank,
            ref other => panic!("the converged fixture must yield observed information: {other:?}"),
        };
        let oracle = finite_difference_robust_bands(term, target, rho, cache, identified_rank);
        assert_eq!(uncertainty.atoms.len(), oracle.joint.len());
        let mut largest_error_ratio = 0.0_f64;
        let mut largest_departure_from_model = 0.0_f64;
        let mut omission_ratio = 0.0_f64;
        let mut production_departure_ratio = 0.0_f64;
        for (k, atom) in uncertainty.atoms.iter().enumerate() {
            let robust = atom
                .band_sd_robust
                .as_ref()
                .expect("observed information must yield the robust band");
            let model = atom.band_sd.as_ref().expect("joint model-based band");
            let expected = &oracle.joint[k];
            assert_eq!(robust.dim(), expected.dim(), "atom {k} robust band shape");
            assert!(
                expected.iter().any(|&v| v > 1.0e-8),
                "atom {k}: the oracle band must not be vacuous"
            );
            for (((&got, &want), &model_sd), &border_sd) in robust
                .iter()
                .zip(expected.iter())
                .zip(model.iter())
                .zip(oracle.border_only[k].iter())
            {
                largest_error_ratio =
                    largest_error_ratio.max((got - want).abs() / band_agreement_budget(want));
                omission_ratio =
                    omission_ratio.max((border_sd - want).abs() / band_agreement_budget(want));
                production_departure_ratio = production_departure_ratio
                    .max((got - border_sd).abs() / band_agreement_budget(border_sd));
                if model_sd > 0.0 {
                    largest_departure_from_model =
                        largest_departure_from_model.max((got / model_sd - 1.0).abs());
                }
            }
        }
        eprintln!(
            "ROBUST_BAND_ORACLE error_ratio={largest_error_ratio:.3e} departure_from_model=\
             {largest_departure_from_model:.3e} omission_ratio={omission_ratio:.3e} \
             production_departure_ratio={production_departure_ratio:.3e}"
        );
        assert!(
            largest_error_ratio <= 1.0,
            "robust band misses the finite-difference sandwich by {largest_error_ratio:.3e} \
             agreement budgets 1e-6·sd + 1e-12"
        );
        assert!(
            largest_departure_from_model > 1.0e-2,
            "the row sandwich must depart from the model-based band on this fixture (largest \
             relative departure {largest_departure_from_model:.3e})"
        );
        RobustBandSeparation {
            omission_ratio,
            production_departure_ratio,
        }
    }

    #[test]
    fn robust_shape_band_matches_finite_difference_row_sandwich() {
        let (term, target, rho, cache, dispersion) = converged_tiny_state(None);
        assert_robust_band_matches_oracle(&term, target.view(), &rho, &cache, dispersion);
    }

    /// Unequal row weights and a whitening metric with off-diagonal precision:
    /// the score contracts `w_i M_i r_i`, which couples output channels.
    #[test]
    fn weighted_whitened_robust_shape_band_matches_finite_difference_row_sandwich() {
        let (term, target, rho, cache, dispersion) = converged_tiny_state(Some(Box::new(|term: &mut SaeManifoldTerm| {
            let n = term.n_obs();
            let p = term.output_dim();
            term.set_row_loss_weights((0..n).map(|row| 0.5 + 0.5 * (row % 4) as f64).collect())
                .expect("row weights");
            let factors = Array2::from_shape_fn((n, p * p), |(row, flat)| {
                let (output, probe) = (flat / p, flat % p);
                let scale = 0.8 + 0.15 * (row % 3) as f64;
                if output == probe {
                    scale
                } else if probe == (output + 1) % p {
                    0.35 * scale
                } else {
                    0.0
                }
            });
            term.set_row_metric(
                gam_problem::RowMetric::whitened_structured(Arc::new(factors), p, p)
                    .expect("whitening metric"),
            )
            .expect("metric matches the term");
        })));
        assert_robust_band_matches_oracle(&term, target.view(), &rho, &cache, dispersion);
    }

    /// The ordered Beta--Bernoulli gate prior couples rows through each atom's
    /// mass `M_k = Σ_i w_i z_ik`, so the user-facing robust band must carry the
    /// mass estimating equations.
    ///
    /// The agreement assertion accepts a band within one budget of the oracle. A
    /// production band that dropped the mass summands would sit on the border-only
    /// sandwich and miss the oracle by `omission_ratio` budgets, so the fixture must
    /// make that at least `k = 10`: the omission then fails the agreement tenfold
    /// instead of hiding inside it. The production band must also stand `k`
    /// budgets away from the border-only sandwich. The decoder and target are
    /// scaled by three, raising every row's logit data curvature ninefold, which is
    /// what gives the masses that resolvable share.
    fn assert_ordered_beta_bernoulli_robust_band_carries_the_masses(weighted: bool) {
        const SEPARATION: f64 = 10.0;
        let (mut term, mut target, mut rho) =
            crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, 0.0);
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -1.0;
            }
        }
        let scale = 3.0_f64;
        let p = term.output_dim();
        target.mapv_inplace(|v| v * scale);
        for atom in term.atoms.iter_mut() {
            for b in 0..atom.basis_size() {
                for c in 0..p {
                    atom.decoder_coefficients_mut()[[b, c]] *= scale;
                }
            }
        }
        if weighted {
            term.set_row_loss_weights(
                (0..term.n_obs())
                    .map(|row| 0.5 + 0.25 * (row % 4) as f64)
                    .collect(),
            )
            .expect("positive design weights");
        }
        let (term, target, rho, cache, dispersion) = converged_state(term, target, rho, 200);
        let separation =
            assert_robust_band_matches_oracle(&term, target.view(), &rho, &cache, dispersion);
        assert!(
            separation.omission_ratio >= SEPARATION,
            "weighted={weighted}: dropping the mass summands must miss the oracle by at least \
             {SEPARATION} agreement budgets (omission ratio {:.3e})",
            separation.omission_ratio
        );
        assert!(
            separation.production_departure_ratio >= SEPARATION,
            "weighted={weighted}: the production band must carry the mass summands, standing at \
             least {SEPARATION} budgets from the border-only sandwich (ratio {:.3e})",
            separation.production_departure_ratio
        );
    }

    #[test]
    fn ordered_beta_bernoulli_robust_shape_band_carries_the_aggregate_mass_equations() {
        assert_ordered_beta_bernoulli_robust_band_carries_the_masses(false);
    }

    /// Unequal design weights enter both `M_k = Σ_i w_i z_ik` and the data scores.
    #[test]
    fn weighted_ordered_beta_bernoulli_robust_shape_band_carries_the_aggregate_mass_equations() {
        assert_ordered_beta_bernoulli_robust_band_carries_the_masses(true);
    }

    #[test]
    fn unavailable_observed_information_names_the_robust_band_reason() {
        let (term, _target, _rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        let dispersion = SaeReconstructionDispersion {
            raw_output_noise_variance: 0.25,
            likelihood_dispersion: 0.25,
            likelihood_frame: SaeLikelihoodFrame::RawOutput,
            selection_conditioning: SaeSelectionConditioning::ConditionalOnFittedRouting,
        };
        for reason in [
            SaeShapeCovarianceUnavailable::NoDenseObservedInformation,
            SaeShapeCovarianceUnavailable::IndefiniteObservedInformation {
                negative_directions: 2,
                most_negative_curvature: -0.5,
            },
        ] {
            let uncertainty = term
                .assemble_shape_uncertainty(
                    &SaeShapeInformation::Unavailable(reason.clone()),
                    dispersion,
                )
                .expect("explicit unavailable shape uncertainty");
            assert_eq!(
                uncertainty.operator,
                SaeShapeCovarianceOperator::Unavailable(reason.clone())
            );
            assert_eq!(uncertainty.atoms.len(), term.k_atoms());
            for atom in &uncertainty.atoms {
                assert!(atom.band_sd.is_none());
                assert_eq!(atom.band_sd_robust, Err(reason.clone()));
            }
        }
    }
}
