// [#2933 F36] The joint fitted-response divergence `tr(∂f̂/∂y)` of the SAE
// reconstruction. It reads the materialized exact stationarity eigensystem owned
// by `construction_exact_hessian.rs`, so it is `include!`d beside that file into
// `construction.rs` and shares its module scope.

/// Why [`SaeManifoldTerm::fitted_response_divergence`] produced no divergence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FittedResponseDivergenceRefusal {
    /// An atom's basis exposes no second jet, so the residual curvature of the
    /// observed information is unknown. Unknown is not zero: no divergence is
    /// formed from the Gauss--Newton block alone.
    SecondJetsUnavailable { reason: String },
    /// The exact stationarity operator, the data curvature, or a solve failed at
    /// this state.
    Numerical { reason: String },
}

impl std::fmt::Display for FittedResponseDivergenceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SecondJetsUnavailable { reason } => write!(
                f,
                "fitted-response divergence unavailable: the observed information needs \
                 every atom's second jet ({reason})"
            ),
            Self::Numerical { reason } => write!(f, "fitted-response divergence failed: {reason}"),
        }
    }
}

/// Which estimator produced a [`FittedResponseDivergence`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FittedResponseDivergenceEstimator {
    /// `Σ_retained vᵢᵀ G vᵢ / λᵢ` on the materialized exact stationarity
    /// eigensystem: the trace itself, to the arithmetic of one symmetric
    /// eigendecomposition.
    ExactSpectral,
    /// Rademacher output-space probes where the dense eigensystem is not
    /// admitted, one Krylov `A⁺` solve per probe ([`FittedResponseProbeEstimate`]).
    /// `raw` is the raw frame's own estimate under a whitening metric, and `None`
    /// where the raw frame is the likelihood frame. The divergence is the
    /// likelihood frame's.
    Hutchinson {
        likelihood: FittedResponseProbeEstimate,
        raw: Option<FittedResponseProbeEstimate>,
    },
}

/// One frame's Rademacher estimate of its residual dof `ν = ‖I − R‖²_F` and of
/// `tr R`, from output-space probes `z` with `E zzᵀ = I`. Each probe reads both
/// `‖z − Rz‖²` and `zᵀRz` off one solve.
///
/// The probe count is derived from the consumer of `ν`. Under the noise model of
/// [`SaeManifoldTerm::reconstruction_dispersion`], `RSS/φ` has variance `2ν`, so
/// `φ̂ = RSS/ν` carries a relative sampling error `√(2/ν)` however `ν` is
/// computed. The probes stop at the least count whose upper confidence bound on
/// the variance of `‖z − Rz‖²`, over the count, is within `2ν̂`
/// ([`hutchinson_residual_dof_resolved`]): there the Monte Carlo relative error of
/// `ν̂` is within that sampling error. With `C = (I − R)ᵀ(I − R)`,
/// `Var(zᵀCz) = 2(‖C‖²_F − Σᵢ Cᵢᵢ²) ≤ 2‖I − R‖²₂·ν`, so the count settles near
/// `‖I − R‖²₂`. That norm is not known before probing: bounding it needs the
/// smallest retained pencil eigenvalue, which no matrix-free route certifies.
/// Each standard error is a sample standard deviation over `√probes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FittedResponseProbeEstimate {
    pub(crate) probes: usize,
    pub(crate) residual_dof: f64,
    pub(crate) residual_dof_standard_error: f64,
    pub(crate) divergence: f64,
    pub(crate) divergence_standard_error: f64,
}

/// The within-basin Stein degrees of freedom of the fitted reconstruction.
///
/// Only the smooth response inside one basin is differentiated. The selected TopK
/// support, frozen routing, and the basin the inner solve converged to are held
/// fixed, so the selection (search) degrees of freedom of a support swap or a basin
/// switch are omitted, not estimated (#2933 F37).
///
/// A frame's residual degrees of freedom are `‖I − R‖²_F` for its response
/// `R = ∂f̂/∂y`. If the noise covariance is `φ` times the inverse of the frame's
/// weight, the residual of the linearized fit has
/// `E‖y − f̂‖² = ‖(I − R)μ‖² + φ·‖I − R‖²_F`, and
/// `‖I − R‖²_F = N − 2 tr R + ‖R‖²_F`. That equals `N − tr R` only when `R` is a
/// projection. For the ridge smoother `R = ½I` it is `N/4`, not `N/2` (#2933 F40).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FittedResponseDivergence {
    pub(crate) divergence: f64,
    /// `‖I − Ω^{½} R Ω^{−½}‖²_F` over the likelihood-frame scalars of the rows
    /// with positive weight, `Ω = ⊕ᵢ wᵢMᵢ`.
    pub(crate) likelihood_residual_dof: f64,
    /// `‖I − R‖²_F` over the raw output scalars of the rows with positive weight.
    /// It equals the likelihood value unless the metric whitens.
    pub(crate) raw_residual_dof: f64,
    pub(crate) estimator: FittedResponseDivergenceEstimator,
    /// Whether the response moves the learned decoder frames with the data, or
    /// holds them at their fitted orientation (#2933 F39).
    pub(crate) frame_conditioning: SaeFrameConditioning,
}

/// The output inner product a dense data curvature is assembled with, over the
/// `√w`-weighted row jets `J̃ᵢ = √wᵢ Jᵢ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponseOutputGram {
    /// `Mᵢ`: `Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ = JᵀΩJ`, the likelihood frame's curvature `G`.
    Likelihood,
    /// `wᵢMᵢ²`: `Σᵢ J̃ᵢᵀ wᵢMᵢ² J̃ᵢ = JᵀΩ²J`.
    SquaredLikelihood,
    /// `I/wᵢ` on rows with `wᵢ > 0`: `Σ_{wᵢ>0} JᵢᵀJᵢ`.
    UnweightedRaw,
}

/// The frame a residual degree-of-freedom count is taken in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FittedResponseFrame {
    /// The likelihood scalars, with noise covariance `φ·Ω⁻¹`.
    Likelihood,
    /// The raw output scalars, with noise covariance `φ·I`.
    Raw,
}

/// Base seed of the fitted-response Rademacher probes. Any fixed stream gives
/// unbiased probes. One fixed stream keeps every estimate bit-reproducible across
/// runs, hosts and outer iterations. Each frame draws on its own stream.
const FITTED_RESPONSE_HUTCHINSON_SEED: u64 = 0x5AED_A3D0_1ACE_9C01;

/// A stationarity operator the caller already holds, which the fitted-response
/// divergence reads instead of forming its own (#2933 F33).
#[derive(Clone, Copy)]
pub(crate) enum HeldResponseGeometry<'a> {
    /// The fixed-frame exact stationarity eigensystem.
    FixedFrame(&'a ExactHessianSpectralBlock),
    /// A shape report's frame-integrated information (#2933 F35).
    FrameMarginal(&'a FrameMarginalInformation),
}

/// How the frame-integrated fitted response is computed at a state's size.
enum FrameIntegratedRoute {
    /// The joint `(t, ξ)` pencil's dense eigensystem
    /// ([`SaeManifoldTerm::frame_marginal_information`]).
    Dense,
    /// Output-space probes through the lifted evidence factor
    /// ([`FrameIntegratedResponseOperator`]).
    MatrixFree,
    /// The unframed evidence factor the tangent operator is built on is not
    /// admitted, so no route integrates the frames on this host.
    Unavailable,
}

/// What a response probe reads off a term at one evidence factor: every atom's
/// second jet, the border channels of that factor's layout, and the sphere
/// blocks whose tangent projector the contraction and the read-out pass through.
struct FittedResponseProbeRows {
    second_jets: Vec<Array4<f64>>,
    border: Vec<SaeBorderChannel>,
    sphere_tangents: Vec<SphereTangentBlock>,
}

/// The frame-integrated fitted response applied through the lifted evidence
/// factor, where the dense operator of
/// [`SaeManifoldTerm::frame_marginal_information`] is not admitted (#2933 F39).
///
/// With `vec B = T·ξ`, the joint `(t, ξ)` layout's operators are the unframed ones
/// pulled back: `A_ξ = diag(I, Tᵀ)·A·diag(I, T) + E` and
/// `Φ_ξ = diag(I, Tᵀ)·Φ·diag(I, T)`. A probe's contraction and read-out pass
/// through `diag(I, T)`, so `R = J·diag(I, T)·A_ξ⁺·diag(I, T)ᵀ·JᵀΩ`, the response the
/// dense route decomposes. `A` and `Φ` are applied through the unframed evidence
/// factor, as the dense route's probes and metric apply them, so nothing
/// `(t + ξ)²` is formed.
pub(crate) struct FrameIntegratedResponseOperator {
    rho: SaeManifoldRho,
    unframed: SaeManifoldTerm,
    cache: ArrowFactorCache,
    tangent: LearnedFrameTangentMap,
    prepared: PreparedDecoderPriorBetaCurvature,
    residual: PreparedResidualCurvatureRows,
    rows: FittedResponseProbeRows,
}

impl FrameIntegratedResponseOperator {
    /// `(‖z − Rz‖², zᵀRz)` for one output-space probe `z` of `frame`, given per row
    /// and empty on a row of zero weight
    /// ([`SaeManifoldTerm::fitted_response_probe`]).
    pub(crate) fn probe(
        &self,
        frame: FittedResponseFrame,
        z: &[Vec<f64>],
    ) -> Result<(f64, f64), String> {
        self.unframed
            .fitted_response_probe(&self.cache, &self.rows, frame, z, &|rhs| self.solve(rhs))
    }

    /// `A_ξ⁺` on a contraction in the unframed `(t, vec B)` layout, returned in that
    /// layout: pull the border back through `Tᵀ`, solve on the pencil
    /// `(A_ξ, Φ_ξ)`, and lift the border through `T`.
    fn solve(&self, rhs: &SaeArrowVector) -> Result<SaeArrowVector, String> {
        let lift = &self.tangent.lift;
        let lifted = |vector: &SaeArrowVector| SaeArrowVector {
            t: vector.t.clone(),
            beta: lift.dot(&vector.beta),
        };
        let pulled_back = |image: SaeArrowVector| SaeArrowVector {
            beta: lift.t().dot(&image.beta),
            t: image.t,
        };
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let image = self.unframed.apply_exact_hessian_prepared(
                &self.rho,
                &self.cache,
                &lifted(vector),
                &self.prepared,
                &self.residual,
            )?;
            let mut pulled = pulled_back(image);
            pulled.beta += &self.tangent.cross_curvature.dot(&vector.beta);
            Ok(pulled)
        };
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let direction = lifted(vector);
            let image = crate::manifold::arrow_solver::apply_cached_arrow_hessian(
                &self.cache,
                direction.t.view(),
                direction.beta.view(),
            )?;
            Ok(pulled_back(image))
        };
        let apply_b_raw = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let direction = lifted(vector);
            let image = apply_raw_cached_arrow_hessian(
                &self.cache,
                direction.t.view(),
                direction.beta.view(),
            )?;
            Ok(pulled_back(image))
        };
        // An image sums `ξ` terms in the lift, is applied in the unframed `t + k_u`
        // coordinates, sums `k_u` more in the pull-back, and on `A_ξ` adds `E·ξ`, so an
        // entry accumulates at most `t + k_u` plus both lifts' inner dimensions plus one.
        let (unframed_border, tangent_dim) = lift.dim();
        let operator_terms =
            self.cache.delta_t_len() + 2 * unframed_border + tangent_dim + 1;
        let solved = solve_exact_stationarity_krylov_with_rounding(
            &pulled_back(rhs.clone()),
            &apply_a,
            &apply_b,
            &apply_b_raw,
            operator_terms,
        )?;
        Ok(lifted(&solved))
    }
}

/// Group border channels by output vector. A channel's reconstruction jet is a
/// scalar times the output vector of its `(atom, output channel)` pair, which
/// every basis column of that atom shares, so the border data curvature needs
/// one inner product per pair of classes rather than per pair of channels.
fn border_output_classes(border: &[SaeBorderChannel]) -> (Vec<usize>, Vec<Vec<f64>>) {
    let mut class_of = Vec::with_capacity(border.len());
    let mut outputs: Vec<Vec<f64>> = Vec::new();
    let mut index: std::collections::HashMap<(usize, Vec<u64>), usize> =
        std::collections::HashMap::new();
    for channel in border {
        let key = (
            channel.atom,
            channel.output.iter().map(|value| value.to_bits()).collect::<Vec<u64>>(),
        );
        let class = *index.entry(key).or_insert_with(|| {
            outputs.push(channel.output.clone());
            outputs.len() - 1
        });
        class_of.push(class);
    }
    (class_of, outputs)
}

/// The mean and the unbiased sample variance of at least two values.
fn sample_mean_and_variance(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let spread = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>();
    (mean, spread / (count - 1.0))
}

/// One-sided level of the fitted-response probe count's variance bound
/// ([`hutchinson_residual_dof_resolved`]): the rule stops early, with the
/// residual dof's Monte Carlo variance still above the dispersion's sampling
/// variance, at most this often for normal probe values. A declared level, not a
/// derived one.
const FITTED_RESPONSE_PROBE_VARIANCE_BOUND_LEVEL: f64 = 0.05;

/// Whether the probe values `residuals`, each `‖z − Rz‖²`, resolve their mean
/// `ν̂` to the sampling error the dispersion already carries
/// ([`FittedResponseProbeEstimate`]).
///
/// The sample variance `σ̂²` of `s` normal values has `(s − 1)σ̂²/σ² ~ χ²_{s−1}`,
/// so `σ²_U = (s − 1)σ̂²/χ²_{s−1}(α)`, with `χ²_{s−1}(α)` the lower `α` quantile,
/// bounds the probe variance from above at level `1 − α`. The rule is
/// `σ²_U/s ≤ 2ν̂`. A small sample under-reads the variance of the term it probes,
/// and the bound's `(s − 1)/χ²_{s−1}(α)` inflation, about 254 at `s = 2` for
/// `α = 0.05`, is what keeps two values that happen to sit close from stopping
/// the probes. One value has no variance degrees of freedom and bounds nothing,
/// so the least count follows from the quantile's domain.
pub(crate) fn hutchinson_residual_dof_resolved(residuals: &[f64]) -> Result<bool, String> {
    use statrs::distribution::ContinuousCDF;
    let degrees_of_freedom = residuals.len().saturating_sub(1);
    if degrees_of_freedom == 0 {
        return Ok(false);
    }
    let (residual_dof, variance) = sample_mean_and_variance(residuals);
    let quantile = statrs::distribution::ChiSquared::new(degrees_of_freedom as f64)
        .map_err(|error| {
            format!("Hutchinson probe count: χ² with {degrees_of_freedom} degrees of freedom: {error}")
        })?
        .inverse_cdf(FITTED_RESPONSE_PROBE_VARIANCE_BOUND_LEVEL);
    if !(quantile.is_finite() && quantile > 0.0) {
        return Err(format!(
            "Hutchinson probe count: the lower {FITTED_RESPONSE_PROBE_VARIANCE_BOUND_LEVEL} quantile \
             of χ² with {degrees_of_freedom} degrees of freedom is {quantile}"
        ));
    }
    let variance_bound = degrees_of_freedom as f64 * variance / quantile;
    Ok(variance_bound / residuals.len() as f64 <= 2.0 * residual_dof)
}

impl SaeManifoldTerm {
    /// Joint fitted-response divergence of the reconstruction at the converged
    /// inner state (#2933 F36).
    ///
    /// With row weights `wᵢ` and output metrics `Mᵢ`, the inner objective is
    /// `L(θ; y) = ½ Σᵢ wᵢ (fᵢ(θ) − yᵢ)ᵀ Mᵢ (fᵢ(θ) − yᵢ) + P(θ)` over every
    /// estimated coordinate `θ = (gate logits, chart coordinates, decoder border)`.
    /// Its stationarity `g = Σᵢ wᵢ Jᵢᵀ Mᵢ (fᵢ − yᵢ) + ∇P = 0` has `∂g/∂θ = A`, the
    /// exact observed information (residual curvature and the actual prior
    /// curvature included), and `∂g/∂yᵢ = −wᵢ Jᵢᵀ Mᵢ`. The implicit-function theorem
    /// gives `∂θ̂/∂yᵢ = A⁺ wᵢ Jᵢᵀ Mᵢ`, hence
    ///
    /// ```text
    ///   tr(∂f̂/∂y) = Σᵢ tr(Jᵢ A⁺ Jᵢᵀ wᵢ Mᵢ) = tr(A⁺ G),   G = Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ,  J̃ᵢ = √wᵢ Jᵢ.
    /// ```
    ///
    /// `A⁺` is the symmetric pseudoinverse with the null band of
    /// [`ExactHessianSpectralBlock::rank_floor`], the same classification the IFT
    /// stationarity solve owns: a direction the objective cannot resolve carries
    /// no identified response, and resolved negative modes are inverted rather
    /// than dropped. The trace couples every block through `A⁺`. Mixed coordinate
    /// curvature, cross-atom gate coupling, free logits, the decoder border, atoms
    /// without ARD, and a periodic prior on the concave side of its axis all enter
    /// through the one operator, never as a sum of separately inverted diagonal
    /// entries.
    ///
    /// A learned decoder frame is estimated from the data, so wherever every framed
    /// decoder has its frame's rank, and the fixed-rank manifold has a tangent
    /// space, the response integrates the frames
    /// ([`Self::frame_integrated_fitted_response_divergence`], #2933 F39). The
    /// host's memory picks how: the dense joint eigensystem or output-space probes.
    /// Both read the unframed evidence factor, so a host that cannot hold it holds
    /// the frames at their fitted orientation, tagged
    /// `UnframedObservedInformationNotAdmitted`. A rank-deficient frame has no
    /// tangent space, so there too the frames are held fixed and the divergence is
    /// conditional on them, tagged with that reason. The TopK support and the
    /// converged basin are always held fixed, so the trace carries no selection
    /// degrees of freedom (#2933 F37).
    pub(crate) fn fitted_response_divergence(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let numerical = |reason: String| FittedResponseDivergenceRefusal::Numerical { reason };
        self.admit_fitted_response_divergence(target)?;
        let frame_conditioning = self.fitted_response_frame_conditioning().map_err(numerical)?;
        if frame_conditioning == SaeFrameConditioning::MarginalOverLearnedFrames {
            return self.frame_integrated_fitted_response_divergence(rho, target, cache, None);
        }
        self.fixed_frame_fitted_response_divergence(rho, target, cache, frame_conditioning)
    }

    /// The divergence over the coordinates of `cache` itself, with no learned frame
    /// integrated: the exact spectral trace where the dense eigensystem is
    /// admitted, output-space probes through the cached evidence factor otherwise.
    fn fixed_frame_fitted_response_divergence(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        frame_conditioning: SaeFrameConditioning,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let numerical = |reason: String| FittedResponseDivergenceRefusal::Numerical { reason };
        let dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
        if sae_exact_stationarity_admitted(dim, self.host_available_bytes) {
            let geometry = self
                .materialize_exact_stationarity_geometry(rho, target, cache)
                .map_err(numerical)?;
            return self.fitted_response_divergence_exact_spectral(
                &geometry,
                cache,
                frame_conditioning,
            );
        }
        let rows = self.fitted_response_probe_rows(cache).map_err(numerical)?;
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        let residual = self
            .prepare_residual_curvature_rows(target, cache)
            .map_err(numerical)?;
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            self.apply_exact_hessian_prepared(rho, cache, vector, &prepared, &residual)
        };
        let apply_b = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            crate::manifold::arrow_solver::apply_cached_arrow_hessian(
                cache,
                vector.t.view(),
                vector.beta.view(),
            )
        };
        let apply_b_raw = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            apply_raw_cached_arrow_hessian(cache, vector.t.view(), vector.beta.view())
        };
        let solve = |rhs: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            solve_exact_stationarity_krylov(rhs, &apply_a, &apply_b, &apply_b_raw)
        };
        let (likelihood, raw) = self
            .hutchinson_fitted_response(&|frame, z| {
                self.fitted_response_probe(cache, &rows, frame, z, &solve)
            })
            .map_err(numerical)?;
        Ok(Self::hutchinson_fitted_response_divergence(
            likelihood,
            raw,
            frame_conditioning,
        ))
    }

    /// [`Self::fitted_response_divergence`] off a stationarity operator the caller
    /// already holds, so one shape report or criterion evaluation decomposes it
    /// once for the divergence and its other consumers (#2933 F33). A fixed-frame
    /// eigensystem is read with the exact spectral trace, whatever the admission
    /// would have routed, because it is already paid for, and resolved negative
    /// modes are inverted exactly as on the admission-routed path. Where the
    /// learned frames are integrated, a fixed-frame geometry describes a different
    /// response, so the frame-integrated one is formed instead, unless the caller
    /// holds it.
    pub(crate) fn fitted_response_divergence_from_geometry(
        &self,
        geometry: HeldResponseGeometry<'_>,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let numerical = |reason: String| FittedResponseDivergenceRefusal::Numerical { reason };
        self.admit_fitted_response_divergence(target)?;
        let frame_conditioning = self.fitted_response_frame_conditioning().map_err(numerical)?;
        match (frame_conditioning, geometry) {
            (
                SaeFrameConditioning::MarginalOverLearnedFrames,
                HeldResponseGeometry::FrameMarginal(information),
            ) => self.fitted_response_divergence_from_frame_marginal(information),
            (
                SaeFrameConditioning::MarginalOverLearnedFrames,
                HeldResponseGeometry::FixedFrame(geometry),
            ) => self.frame_integrated_fitted_response_divergence(rho, target, cache, Some(geometry)),
            (frame_conditioning, HeldResponseGeometry::FixedFrame(geometry)) => {
                self.fitted_response_divergence_exact_spectral(geometry, cache, frame_conditioning)
            }
            (frame_conditioning, HeldResponseGeometry::FrameMarginal(_)) => Err(numerical(format!(
                "a frame-integrated information was held for a state whose fitted response is \
                 {frame_conditioning:?}"
            ))),
        }
    }

    /// Which frame conditioning the fitted response takes at this state: none
    /// without learned frames, integrated wherever every framed decoder has its
    /// frame's rank, and conditional on the fitted frames where one does not,
    /// because the fixed-rank manifold has no tangent space there. The host's
    /// memory does not enter here: [`Self::frame_integrated_route`] reads it to pick
    /// the route.
    fn fitted_response_frame_conditioning(&self) -> Result<SaeFrameConditioning, String> {
        if !self.frames_active() {
            return Ok(SaeFrameConditioning::NoLearnedFrames);
        }
        for (atom_idx, atom) in self.atoms.iter().enumerate() {
            let Some(frame) = atom.decoder_frame.as_ref() else {
                continue;
            };
            if atom.decoder_numerical_rank()? < frame.rank() {
                return Ok(SaeFrameConditioning::ConditionalOnFittedFrames(
                    SaeFrameMarginalUnavailable::FrameCoordinatesRankDeficient { atom: atom_idx },
                ));
            }
        }
        Ok(SaeFrameConditioning::MarginalOverLearnedFrames)
    }

    /// Which route computes the frame-integrated response at this state's size,
    /// with `total_t` the coordinate block of the fitted cache, which dropping the
    /// frames leaves unchanged.
    ///
    /// Both routes read the unframed evidence factor, whose reduced Schur is dense
    /// in the unframed border `k_u = Σ M_k·p`. The dense route also materializes
    /// `A` and `G` at `t + k_u` and decomposes the joint pencil at `t + ξ`, and
    /// `ξ ≤ k_u` because each frame's rank is at most its basis size, so it is
    /// admitted by [`sae_exact_stationarity_admitted`] at `t + k_u`, the predicate
    /// the fixed-frame route asks at its own dimensions. The matrix-free route keeps
    /// only the factor, the lift `T` (`k_u × ξ`) and `E` (`ξ × ξ`), each within the
    /// factor's `k_u²`.
    fn frame_integrated_route(&self, total_t: usize) -> Result<FrameIntegratedRoute, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_basis: usize = self.atoms.iter().map(|atom| atom.basis_size()).sum();
        let d_max = self
            .atoms
            .iter()
            .map(SaeManifoldAtom::latent_dim)
            .max()
            .unwrap_or(0);
        let unframed_border = self.beta_dim();
        let Ok(plan) = sae_streaming_plan_for_shape_with_available(
            n,
            total_basis,
            k_atoms,
            d_max,
            unframed_border,
            self.gpu_policy,
            self.host_available_bytes,
        )?
        .admitted_or_error(n, self.output_dim(), k_atoms) else {
            return Ok(FrameIntegratedRoute::Unavailable);
        };
        if !plan.direct_admitted {
            return Ok(FrameIntegratedRoute::Unavailable);
        }
        let dense_dim = sae_exact_stationarity_dim(total_t, unframed_border);
        Ok(
            if plan.direct_logdet_admitted()
                && sae_exact_stationarity_admitted(dense_dim, self.host_available_bytes)
            {
                FrameIntegratedRoute::Dense
            } else {
                FrameIntegratedRoute::MatrixFree
            },
        )
    }

    /// The fitted-response divergence integrated over every learned frame (#2933
    /// F39), on the route [`Self::frame_integrated_route`] admits.
    ///
    /// A framed decoder moves along `vec B = T·ξ`, `ξ = (vec δC, vec W)`, and its
    /// stationarity in `(t, ξ)` has Jacobian
    /// `A_ξ = [[A_tt, A_tB·T], [Tᵀ·A_Bt, Tᵀ·A_BB·T + E]]`
    /// ([`Self::frame_marginal_information`]). The data reach that stationarity
    /// through `∂F_ξ/∂y = −diag(I, T)ᵀ·JᵀΩ` at `ξ = 0`, with no `E`, so
    /// `R = J·diag(I, T)·A_ξ⁺·diag(I, T)ᵀ·JᵀΩ`. The frame orientations then carry
    /// their actual response instead of a count of fully determined directions.
    ///
    /// Where neither route is admitted, the frames are held at their fitted
    /// orientation on the fitted `cache`, tagged
    /// `UnframedObservedInformationNotAdmitted`: `fixed_frame` is the caller's
    /// eigensystem of that cache when it holds one.
    fn frame_integrated_fitted_response_divergence(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        fixed_frame: Option<&ExactHessianSpectralBlock>,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let numerical = |reason: String| FittedResponseDivergenceRefusal::Numerical { reason };
        match self.frame_integrated_route(cache.delta_t_len()).map_err(numerical)? {
            FrameIntegratedRoute::Dense => {
                let information = self
                    .frame_marginal_information(rho, target, None)
                    .map_err(numerical)?;
                self.fitted_response_divergence_from_frame_marginal(&information)
            }
            FrameIntegratedRoute::MatrixFree => {
                let operator = self
                    .frame_integrated_response_operator(rho, target, None)
                    .map_err(numerical)?;
                let (likelihood, raw) = self
                    .hutchinson_fitted_response(&|frame, z| operator.probe(frame, z))
                    .map_err(numerical)?;
                Ok(Self::hutchinson_fitted_response_divergence(
                    likelihood,
                    raw,
                    SaeFrameConditioning::MarginalOverLearnedFrames,
                ))
            }
            FrameIntegratedRoute::Unavailable => {
                let frame_conditioning = SaeFrameConditioning::ConditionalOnFittedFrames(
                    SaeFrameMarginalUnavailable::UnframedObservedInformationNotAdmitted,
                );
                match fixed_frame {
                    Some(geometry) => self.fitted_response_divergence_exact_spectral(
                        geometry,
                        cache,
                        frame_conditioning,
                    ),
                    None => self.fixed_frame_fitted_response_divergence(
                        rho,
                        target,
                        cache,
                        frame_conditioning,
                    ),
                }
            }
        }
    }

    /// The frame-integrated divergence off the dense joint information: every trace
    /// of [`Self::spectral_fitted_response_traces`] reads the lifted curvature
    /// `G_ξ = diag(I, T)ᵀ·G·diag(I, T)` of the unframed `(t, vec B)` data curvature,
    /// over the retained directions of `A_ξ`.
    fn fitted_response_divergence_from_frame_marginal(
        &self,
        information: &FrameMarginalInformation,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let numerical = |reason: String| FittedResponseDivergenceRefusal::Numerical { reason };
        let total_t = information.total_t;
        let lift = &information.tangent.lift;
        let (beta_dim, xi_dim) = lift.dim();
        let lifted_curvature = |gram: ResponseOutputGram| -> Result<Array2<f64>, String> {
            let curvature = information
                .unframed
                .materialize_data_gauss_newton_dense(&information.cache, gram)?;
            if curvature.dim() != (total_t + beta_dim, total_t + beta_dim) {
                return Err(format!(
                    "frame-integrated divergence: data curvature {:?} does not match t dimension \
                     {total_t} plus beta dimension {beta_dim}",
                    curvature.dim()
                ));
            }
            let coupling = curvature.slice(ndarray::s![..total_t, total_t..]).dot(lift);
            let border = information
                .tangent
                .congruence(&curvature.slice(ndarray::s![total_t.., total_t..]).to_owned());
            let mut lifted = Array2::<f64>::zeros((total_t + xi_dim, total_t + xi_dim));
            lifted
                .slice_mut(ndarray::s![..total_t, ..total_t])
                .assign(&curvature.slice(ndarray::s![..total_t, ..total_t]));
            lifted
                .slice_mut(ndarray::s![..total_t, total_t..])
                .assign(&coupling);
            lifted
                .slice_mut(ndarray::s![total_t.., ..total_t])
                .assign(&coupling.t());
            lifted
                .slice_mut(ndarray::s![total_t.., total_t..])
                .assign(&border);
            Ok(lifted)
        };
        let (divergence, likelihood_residual_dof, raw_residual_dof) = information
            .unframed
            .spectral_fitted_response_traces(&information.joint, lifted_curvature)
            .map_err(numerical)?;
        Ok(FittedResponseDivergence {
            divergence,
            likelihood_residual_dof,
            raw_residual_dof,
            estimator: FittedResponseDivergenceEstimator::ExactSpectral,
            frame_conditioning: SaeFrameConditioning::MarginalOverLearnedFrames,
        })
    }

    /// What every divergence estimator needs: a target of the fitted shape, and
    /// every atom's second jet, without which the residual curvature is unknown.
    fn admit_fitted_response_divergence(
        &self,
        target: ArrayView2<'_, f64>,
    ) -> Result<(), FittedResponseDivergenceRefusal> {
        if target.dim() != (self.n_obs(), self.output_dim()) {
            return Err(FittedResponseDivergenceRefusal::Numerical {
                reason: format!(
                    "target {:?} != ({}, {})",
                    target.dim(),
                    self.n_obs(),
                    self.output_dim()
                ),
            });
        }
        if let Err(reason) = self.atom_second_jets() {
            return Err(FittedResponseDivergenceRefusal::SecondJetsUnavailable { reason });
        }
        Ok(())
    }

    /// The exact spectral [`FittedResponseDivergence`] off `geometry`.
    fn fitted_response_divergence_exact_spectral(
        &self,
        geometry: &ExactHessianSpectralBlock,
        cache: &ArrowFactorCache,
        frame_conditioning: SaeFrameConditioning,
    ) -> Result<FittedResponseDivergence, FittedResponseDivergenceRefusal> {
        let (divergence, likelihood_residual_dof, raw_residual_dof) = self
            .spectral_fitted_response_traces(geometry, |gram| {
                self.materialize_data_gauss_newton_dense(cache, gram)
            })
            .map_err(|reason| FittedResponseDivergenceRefusal::Numerical { reason })?;
        Ok(FittedResponseDivergence {
            divergence,
            likelihood_residual_dof,
            raw_residual_dof,
            estimator: FittedResponseDivergenceEstimator::ExactSpectral,
            frame_conditioning,
        })
    }

    /// `tr(A⁺G)` and each frame's residual dof off a materialized stationarity
    /// eigensystem, with `curvature` materializing each output Gram's data
    /// curvature in the eigensystem's coordinates.
    ///
    /// With the retained generalized eigenpairs `(μᵢ, wᵢ)` of the pencil `(A, Φ)`, so that
    /// `A⁺ = Σ_retained wᵢwᵢᵀ/μᵢ` (#2933 F07), and `W_x = WᵀG_xW`,
    /// `tr(A⁺G) = Σᵢ Wᵢᵢ/μᵢ` and `tr(A⁺G_aA⁺G_b) = Σᵢⱼ (W_a)ᵢⱼ(W_b)ⱼᵢ/(μᵢμⱼ)`. The
    /// likelihood frame's `‖Ω^{½}RΩ^{−½}‖²_F` is `tr(A⁺GA⁺G)`. Under a whitening metric
    /// the raw frame's `‖R‖²_F` for `R = JA⁺JᵀΩ` is `tr(A⁺G_aA⁺G_b)` with
    /// `G_a = JᵀΩ²J` and `G_b = JᵀJ`.
    fn spectral_fitted_response_traces(
        &self,
        geometry: &ExactHessianSpectralBlock,
        curvature: impl Fn(ResponseOutputGram) -> Result<Array2<f64>, String>,
    ) -> Result<(f64, f64, f64), String> {
        let dim = geometry.eigenvalues.len();
        if geometry.eigenvectors.dim() != (dim, dim) {
            return Err(format!(
                "exact spectral divergence: eigenvectors {:?} do not match the \
                 {dim}-dimensional spectrum",
                geometry.eigenvectors.dim()
            ));
        }
        // #2234 — `A⁺ = B·diag(w)·Bᵀ` over the resolved directions, from the block's one owner, so an
        // orbit-stiffened block hands out the exact-`A` response rather than `A_s⁺`.
        let (basis, inverse) = geometry.retained_pseudo_inverse_factors();
        let project = |gram: ResponseOutputGram| -> Result<Array2<f64>, String> {
            let curvature = curvature(gram)?;
            if curvature.dim() != (dim, dim) {
                return Err(format!(
                    "exact spectral divergence: data curvature {:?} does not match the \
                     {dim}-dimensional spectrum",
                    curvature.dim()
                ));
            }
            Ok(basis.t().dot(&curvature.dot(&basis)))
        };
        let paired_trace = |left: &Array2<f64>, right: &Array2<f64>| -> f64 {
            let mut total = 0.0_f64;
            for i in 0..inverse.len() {
                for j in 0..inverse.len() {
                    total += left[[i, j]] * right[[j, i]] * inverse[i] * inverse[j];
                }
            }
            total
        };
        let likelihood = project(ResponseOutputGram::Likelihood)?;
        let divergence: f64 = (0..inverse.len())
            .map(|i| likelihood[[i, i]] * inverse[i])
            .sum();
        let (likelihood_scalars, raw_scalars) = self.fitted_response_scalar_counts()?;
        let likelihood_residual_dof =
            likelihood_scalars - 2.0 * divergence + paired_trace(&likelihood, &likelihood);
        let raw_residual_dof = if self.data_curvature_metric()?.is_some() {
            let squared = project(ResponseOutputGram::SquaredLikelihood)?;
            let unweighted = project(ResponseOutputGram::UnweightedRaw)?;
            raw_scalars - 2.0 * divergence + paired_trace(&squared, &unweighted)
        } else {
            likelihood_residual_dof
        };
        for (label, value) in [
            ("divergence", divergence),
            ("likelihood residual dof", likelihood_residual_dof),
            ("raw residual dof", raw_residual_dof),
        ] {
            if !value.is_finite() {
                return Err(format!("exact spectral {label} is non-finite: {value}"));
            }
        }
        Ok((divergence, likelihood_residual_dof, raw_residual_dof))
    }

    /// Scalar observations of the likelihood frame and of the raw frame on the
    /// rows with positive weight. A zero-weight row is excluded from estimation,
    /// so it carries no scalar in either frame.
    pub(crate) fn fitted_response_scalar_counts(&self) -> Result<(f64, f64), String> {
        let live_rows = match self.row_loss_weights.as_deref() {
            Some(weights) => weights.iter().filter(|&&weight| weight > 0.0).count(),
            None => self.n_obs(),
        };
        let channels = match self.data_curvature_metric()? {
            Some(metric) => metric.metric_rank(),
            None => self.output_dim(),
        };
        Ok((
            (live_rows * channels) as f64,
            (live_rows * self.output_dim()) as f64,
        ))
    }

    /// The unframed model at this state, assembled and factored for evidence: the
    /// term with every frame dropped, its assembled system, and that system's
    /// undamped evidence factor. Both frame-integrated routes read it (#2933
    /// F35/F39).
    ///
    /// The decoder `B_k` is authoritative on a framed atom, so dropping every frame
    /// leaves the state unchanged. `Clone` resets the collapse-prevention gates, and
    /// an unframed assembly with no gates re-derives them from the state. The
    /// criterion priced the gates this term holds, declared by the outer objective
    /// or frozen at the criterion's entry, so the clone declares exactly those.
    /// Otherwise the separation barrier's routing coactivations `q_jk` and effective
    /// sample sizes, the repulsion gate and the amplitude turn-on radius of the
    /// frame-integrated operator would be re-derived at this state, and it would not
    /// be the operator the criterion and the fixed-frame covariance read.
    fn unframed_evidence_factorization(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<(SaeManifoldTerm, ArrowSchurSystem, ArrowFactorCache), String> {
        let mut unframed = self.clone();
        for atom in unframed.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        unframed.declare_collapse_prevention_gates(&self.collapse_prevention_gates());
        let mut system = unframed.assemble_arrow_schur(target, rho, registry)?;
        Self::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
        let (_delta_t, _delta_beta, cache) = solve_arrow_newton_step_with_options(
            &system,
            0.0,
            0.0,
            &unframed.evidence_factor_options(),
        )
        .map_err(|err| format!("frame-integrated operator: unframed evidence factor: {err}"))?;
        Ok((unframed, system, cache))
    }

    /// The frame-integrated fitted response through the lifted evidence factor at
    /// this state ([`FrameIntegratedResponseOperator`]).
    pub(crate) fn frame_integrated_response_operator(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        registry: Option<&AnalyticPenaltyRegistry>,
    ) -> Result<FrameIntegratedResponseOperator, String> {
        let (unframed, system, cache) =
            self.unframed_evidence_factorization(rho, target, registry)?;
        let tangent = LearnedFrameTangentMap::new(self, system.gb.view())?;
        // The assembled system holds the dense `k_u × k_u` border Hessian, which the
        // factor has already reduced; the probes read only the factor.
        drop(system);
        let prepared = unframed.prepare_decoder_prior_beta_curvature(1.0);
        let residual = unframed.prepare_residual_curvature_rows(target, &cache)?;
        let rows = unframed.fitted_response_probe_rows(&cache)?;
        Ok(FrameIntegratedResponseOperator {
            rho: rho.clone(),
            unframed,
            cache,
            tangent,
            prepared,
            residual,
            rows,
        })
    }

    /// What [`Self::fitted_response_probe`] reads off this term at `cache`.
    fn fitted_response_probe_rows(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<FittedResponseProbeRows, String> {
        Ok(FittedResponseProbeRows {
            second_jets: self.atom_second_jets()?,
            border: self.border_channels_for_cache(cache)?,
            sphere_tangents: self.sphere_tangent_blocks(&cache.row_dims)?,
        })
    }

    /// Entries of one row's output-space probe on `frame`: the whitening metric's
    /// rank on the likelihood frame, the output dimension otherwise.
    fn fitted_response_probe_width(&self, frame: FittedResponseFrame) -> Result<usize, String> {
        Ok(match (frame, self.data_curvature_metric()?) {
            (FittedResponseFrame::Likelihood, Some(metric)) => metric.metric_rank(),
            _ => self.output_dim(),
        })
    }

    /// `(‖z − Rz‖², zᵀRz)` for one output-space probe `z` of `frame` against the
    /// evidence factor `cache`, with `z` given per row and empty on a row of zero
    /// weight, which carries no scalar. `z` is contracted into the stationarity's
    /// right-hand side, `solve` applies `A⁺` in `cache`'s `(t, β)` layout, and the
    /// response is read back out.
    ///
    /// For `z` with `E zzᵀ = I`, `E‖(I − R)z‖² = ‖I − R‖²_F` and `E zᵀRz = tr R`. Each
    /// `‖(I − R)z‖²` is non-negative, whereas `N − 2 tr R + ‖R‖²_F` from separate
    /// trace estimates can cancel to either sign near interpolation. On the
    /// likelihood frame `Ω^{½} = √wᵢ Uᵢᵀ` with `Mᵢ = UᵢUᵢᵀ` (`√wᵢ I` unwhitened), so a
    /// probe contracts `J̃ᵢᵀUᵢzᵢ`, solves `A⁺`, and reads `UᵢᵀJ̃ᵢu`. On the raw frame
    /// `R = JA⁺JᵀΩ`, so it contracts `J̃ᵢᵀ√wᵢMᵢzᵢ` and reads `J̃ᵢu/√wᵢ`, and its `zᵀRz`
    /// estimates the same trace, because that `R` is similar to the likelihood
    /// frame's. #2933 F36 — `R = J·P·A⁺·P·JᵀΩ` on sphere blocks: the contraction and
    /// the solve's read-out both pass through the tangent projector.
    fn fitted_response_probe(
        &self,
        cache: &ArrowFactorCache,
        rows: &FittedResponseProbeRows,
        frame: FittedResponseFrame,
        z: &[Vec<f64>],
        solve: &dyn Fn(&SaeArrowVector) -> Result<SaeArrowVector, String>,
    ) -> Result<(f64, f64), String> {
        let n = self.n_obs();
        let p = self.output_dim();
        let total_t = cache.delta_t_len();
        let metric = self.data_curvature_metric()?;
        let weights = self.row_loss_weights.as_deref();
        let width = self.fitted_response_probe_width(frame)?;
        if z.len() != n {
            return Err(format!(
                "response probe: {} probe rows for {n} observations",
                z.len()
            ));
        }
        let mut rhs = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        let mut window: std::collections::VecDeque<SaeRowJets> = std::collections::VecDeque::new();
        let mut next = 0usize;
        for row in 0..n {
            if window.is_empty() {
                next = self.refill_jet_window(
                    next,
                    cache,
                    &rows.second_jets,
                    &rows.border,
                    &mut window,
                )?;
            }
            let jets = window
                .pop_front()
                .ok_or_else(|| format!("response probe: the jet refill built no row {row}"))?;
            let weight = weights.map_or(1.0, |weights| weights[row]);
            let z_row = &z[row];
            let expected = if weight > 0.0 { width } else { 0 };
            if z_row.len() != expected {
                return Err(format!(
                    "response probe: row {row} of weight {weight} carries {} probe entries, not \
                     {expected}",
                    z_row.len()
                ));
            }
            if z_row.is_empty() {
                continue;
            }
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            if jets.vars.len() != q {
                return Err(format!(
                    "response probe: row {row} jets have {} variables, cache row has {q}",
                    jets.vars.len()
                ));
            }
            let output: Vec<f64> = match (frame, metric) {
                (FittedResponseFrame::Likelihood, Some(metric)) => (0..p)
                    .map(|out| {
                        z_row
                            .iter()
                            .enumerate()
                            .map(|(column, &value)| metric.factor_entry(row, out, column) * value)
                            .sum()
                    })
                    .collect(),
                (FittedResponseFrame::Raw, Some(metric)) => metric
                    .apply_metric_row(row, ndarray::aview1(z_row))
                    .into_iter()
                    .map(|value| weight.sqrt() * value)
                    .collect(),
                (_, None) => z_row.clone(),
            };
            for a in 0..q {
                rhs.t[base + a] += sae_dot(jets.first(a), &output);
            }
            for (position, channel) in rows.border.iter().enumerate() {
                rhs.beta[channel.index] += sae_dot(jets.beta(position), &output);
            }
        }
        project_sphere_tangent_slots(&rows.sphere_tangents, &cache.row_offsets, &mut rhs.t.view_mut());
        let mut solved = solve(&rhs)?;
        if solved.t.len() != total_t || solved.beta.len() != cache.k {
            return Err(format!(
                "response probe: the solve returned (t={}, beta={}) for (t={total_t}, beta={})",
                solved.t.len(),
                solved.beta.len(),
                cache.k
            ));
        }
        project_sphere_tangent_slots(
            &rows.sphere_tangents,
            &cache.row_offsets,
            &mut solved.t.view_mut(),
        );
        let mut window: std::collections::VecDeque<SaeRowJets> = std::collections::VecDeque::new();
        let mut next = 0usize;
        let mut response = vec![0.0_f64; p];
        let mut residual = 0.0_f64;
        let mut divergence = 0.0_f64;
        for row in 0..n {
            if window.is_empty() {
                next = self.refill_jet_window(
                    next,
                    cache,
                    &rows.second_jets,
                    &rows.border,
                    &mut window,
                )?;
            }
            let jets = window
                .pop_front()
                .ok_or_else(|| format!("response probe: the jet refill built no row {row}"))?;
            let z_row = &z[row];
            if z_row.is_empty() {
                continue;
            }
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            response.fill(0.0);
            for a in 0..q {
                let coefficient = solved.t[base + a];
                for (slot, &value) in response.iter_mut().zip(jets.first(a)) {
                    *slot += coefficient * value;
                }
            }
            for (position, channel) in rows.border.iter().enumerate() {
                let coefficient = solved.beta[channel.index];
                for (slot, &value) in response.iter_mut().zip(jets.beta(position)) {
                    *slot += coefficient * value;
                }
            }
            let weight = weights.map_or(1.0, |weights| weights[row]);
            let image: Vec<f64> = match (frame, metric) {
                (FittedResponseFrame::Likelihood, Some(metric)) => {
                    metric.whiten_residual_row(row, ndarray::aview1(&response))
                }
                (FittedResponseFrame::Raw, Some(_)) => {
                    response.iter().map(|value| value / weight.sqrt()).collect()
                }
                (_, None) => response.clone(),
            };
            if image.len() != z_row.len() {
                return Err(format!(
                    "response probe: row {row} reads {} response entries for {} probe entries",
                    image.len(),
                    z_row.len()
                ));
            }
            for (probe_value, image_value) in z_row.iter().zip(&image) {
                residual += (probe_value - image_value).powi(2);
                divergence += probe_value * image_value;
            }
        }
        Ok((residual, divergence))
    }

    /// Both frames' [`FittedResponseProbeEstimate`]s through `probe`, one
    /// [`Self::fitted_response_probe`] against some solve: the likelihood frame's,
    /// and the raw frame's own where the metric whitens.
    fn hutchinson_fitted_response(
        &self,
        probe: &dyn Fn(FittedResponseFrame, &[Vec<f64>]) -> Result<(f64, f64), String>,
    ) -> Result<(FittedResponseProbeEstimate, Option<FittedResponseProbeEstimate>), String> {
        let likelihood = self.hutchinson_fitted_response_frame(
            FittedResponseFrame::Likelihood,
            FITTED_RESPONSE_HUTCHINSON_SEED.wrapping_add(1 << 32),
            probe,
        )?;
        let raw = match self.data_curvature_metric()? {
            Some(_) => Some(self.hutchinson_fitted_response_frame(
                FittedResponseFrame::Raw,
                FITTED_RESPONSE_HUTCHINSON_SEED.wrapping_add(2 << 32),
                probe,
            )?),
            None => None,
        };
        Ok((likelihood, raw))
    }

    /// One frame's [`FittedResponseProbeEstimate`]: Rademacher probes on the stream
    /// `seed`, drawn until the residual dof's Monte Carlo variance is within the
    /// sampling variance the dispersion already carries.
    fn hutchinson_fitted_response_frame(
        &self,
        frame: FittedResponseFrame,
        seed: u64,
        probe: &dyn Fn(FittedResponseFrame, &[Vec<f64>]) -> Result<(f64, f64), String>,
    ) -> Result<FittedResponseProbeEstimate, String> {
        let mut residuals: Vec<f64> = Vec::new();
        let mut divergences: Vec<f64> = Vec::new();
        loop {
            let z = self.rademacher_response_probe(frame, seed.wrapping_add(residuals.len() as u64))?;
            let (residual, divergence) = probe(frame, &z)?;
            if !(residual.is_finite() && divergence.is_finite()) {
                return Err(format!(
                    "Hutchinson probe {} of the {frame:?} frame is non-finite: ‖z − Rz‖² = \
                     {residual}, zᵀRz = {divergence}",
                    residuals.len()
                ));
            }
            residuals.push(residual);
            divergences.push(divergence);
            if hutchinson_residual_dof_resolved(&residuals)? {
                let probes = residuals.len();
                let (residual_dof, residual_variance) = sample_mean_and_variance(&residuals);
                let (divergence, divergence_variance) = sample_mean_and_variance(&divergences);
                return Ok(FittedResponseProbeEstimate {
                    probes,
                    residual_dof,
                    residual_dof_standard_error: (residual_variance / probes as f64).sqrt(),
                    divergence,
                    divergence_standard_error: (divergence_variance / probes as f64).sqrt(),
                });
            }
        }
    }

    /// One Rademacher output-space probe of `frame` on the stream `seed`, per row
    /// and empty on a row of zero weight.
    fn rademacher_response_probe(
        &self,
        frame: FittedResponseFrame,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>, String> {
        let width = self.fitted_response_probe_width(frame)?;
        let weights = self.row_loss_weights.as_deref();
        let mut state = seed;
        let mut bits = 0u64;
        let mut remaining = 0u32;
        let mut draw = || {
            if remaining == 0 {
                bits = gam_linalg::utils::splitmix64(&mut state);
                remaining = 64;
            }
            let value = if bits & 1 == 1 { 1.0 } else { -1.0 };
            bits >>= 1;
            remaining -= 1;
            value
        };
        Ok((0..self.n_obs())
            .map(|row| {
                if weights.map_or(1.0, |weights| weights[row]) > 0.0 {
                    (0..width).map(|_| draw()).collect()
                } else {
                    Vec::new()
                }
            })
            .collect())
    }

    /// A [`FittedResponseDivergence`] off both frames' Rademacher estimates.
    fn hutchinson_fitted_response_divergence(
        likelihood: FittedResponseProbeEstimate,
        raw: Option<FittedResponseProbeEstimate>,
        frame_conditioning: SaeFrameConditioning,
    ) -> FittedResponseDivergence {
        FittedResponseDivergence {
            divergence: likelihood.divergence,
            likelihood_residual_dof: likelihood.residual_dof,
            raw_residual_dof: raw.map_or(likelihood.residual_dof, |raw| raw.residual_dof),
            estimator: FittedResponseDivergenceEstimator::Hutchinson { likelihood, raw },
            frame_conditioning,
        }
    }

    /// The whitening metric the data likelihood is assembled through, or `None`
    /// when the metric is the identity.
    fn data_curvature_metric(&self) -> Result<Option<&gam_problem::RowMetric>, String> {
        if !self.whiten_logdet_row_jets() {
            return Ok(None);
        }
        self.row_metric
            .as_ref()
            .map(Some)
            .ok_or_else(|| "data curvature: whitening metric absent".to_string())
    }

    /// The data Gauss--Newton curvature `G = Σᵢ J̃ᵢᵀ Mᵢ J̃ᵢ` as a dense matrix in the
    /// joint `(t, β)` cache layout. No prior, penalty, or residual curvature
    /// enters: those belong to `A`, and `G` is what the target perturbation reaches
    /// the stationarity through.
    ///
    /// A border channel's jet is `sᵢ_b u_b`, a scalar (`√wᵢ·a_k·Φ_b(tᵢ)`) times its
    /// output class vector, so the border block accumulates
    /// `Σᵢ sᵢ sᵢᵀ ∘ Ωᵢ` with `Ωᵢ[c, c'] = u_cᵀ Mᵢ u_c'`. The scalar is read back off
    /// the emitted jet (`⟨jet, u⟩/⟨u, u⟩`), so the row program stays the only
    /// definition of the channel.
    ///
    /// `output_gram` names the per-row output inner product (see
    /// [`ResponseOutputGram`]): the likelihood metric for `G` itself, or the two
    /// raw-frame Grams a whitened residual dof pairs.
    fn materialize_data_gauss_newton_dense(
        &self,
        cache: &ArrowFactorCache,
        output_gram: ResponseOutputGram,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let dim = sae_exact_stationarity_dim(total_t, cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let metric = self.data_curvature_metric()?;
        let weights = self.row_loss_weights.as_deref();
        let (class_of, class_outputs) = border_output_classes(&border);
        let n_classes = class_outputs.len();
        let class_norm_sq: Vec<f64> = class_outputs.iter().map(|u| sae_dot(u, u)).collect();
        let gram = |left: &[Vec<f64>], right: &[Vec<f64>]| -> Vec<f64> {
            let mut omega = vec![0.0_f64; n_classes * n_classes];
            for (c, u) in left.iter().enumerate() {
                for (d, v) in right.iter().enumerate() {
                    omega[c * n_classes + d] = sae_dot(u, v);
                }
            }
            omega
        };
        let unwhitened_omega = (metric.is_none() && output_gram == ResponseOutputGram::Likelihood)
            .then(|| gram(&class_outputs, &class_outputs));
        let mut g = Array2::<f64>::zeros((dim, dim));
        let mut window: std::collections::VecDeque<SaeRowJets> = std::collections::VecDeque::new();
        let mut next = 0usize;
        let mut scalars = vec![0.0_f64; border.len()];
        let mut live: Vec<usize> = Vec::with_capacity(border.len());
        for row in 0..n {
            if window.is_empty() {
                next = self.refill_jet_window(next, cache, &second_jets, &border, &mut window)?;
            }
            let jets = window.pop_front().ok_or_else(|| {
                format!("data curvature materialization: the jet refill built no row {row}")
            })?;
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            if jets.vars.len() != q {
                return Err(format!(
                    "data curvature materialization: row {row} jets have {} variables, cache \
                     row has {q}",
                    jets.vars.len()
                ));
            }
            let weight = weights.map_or(1.0, |weights| weights[row]);
            let metric_apply = |values: &[f64]| -> Vec<f64> {
                match metric {
                    Some(metric) => metric.apply_metric_row(row, ndarray::aview1(values)),
                    None => values.to_vec(),
                }
            };
            let apply = |values: &[f64]| -> Vec<f64> {
                match output_gram {
                    ResponseOutputGram::Likelihood => metric_apply(values),
                    ResponseOutputGram::SquaredLikelihood => metric_apply(&metric_apply(values))
                        .into_iter()
                        .map(|value| weight * value)
                        .collect(),
                    ResponseOutputGram::UnweightedRaw if weight > 0.0 => {
                        values.iter().map(|value| value / weight).collect()
                    }
                    ResponseOutputGram::UnweightedRaw => vec![0.0; values.len()],
                }
            };
            let metric_first: Vec<Vec<f64>> = (0..q).map(|a| apply(jets.first(a))).collect();
            for a in 0..q {
                for b in 0..q {
                    g[[base + a, base + b]] += sae_dot(jets.first(a), &metric_first[b]);
                }
            }
            live.clear();
            for (position, &class) in class_of.iter().enumerate() {
                scalars[position] = if class_norm_sq[class] > 0.0 {
                    sae_dot(jets.beta(position), &class_outputs[class]) / class_norm_sq[class]
                } else {
                    0.0
                };
                if scalars[position] != 0.0 {
                    live.push(position);
                }
            }
            if live.is_empty() {
                continue;
            }
            let row_omega;
            let omega: &[f64] = match unwhitened_omega.as_ref() {
                Some(omega) => omega.as_slice(),
                None => {
                    let metric_outputs: Vec<Vec<f64>> =
                        class_outputs.iter().map(|u| apply(u)).collect();
                    row_omega = gram(&class_outputs, &metric_outputs);
                    row_omega.as_slice()
                }
            };
            for a in 0..q {
                let first_dot_class: Vec<f64> = class_outputs
                    .iter()
                    .map(|u| sae_dot(&metric_first[a], u))
                    .collect();
                for &position in &live {
                    let value = scalars[position] * first_dot_class[class_of[position]];
                    let border_slot = total_t + border[position].index;
                    g[[base + a, border_slot]] += value;
                    g[[border_slot, base + a]] += value;
                }
            }
            for &left in &live {
                let left_slot = total_t + border[left].index;
                let left_class = class_of[left];
                for &right in &live {
                    g[[left_slot, total_t + border[right].index]] += scalars[left]
                        * scalars[right]
                        * omega[left_class * n_classes + class_of[right]];
                }
            }
        }
        // #2933 F36 — `P·G·P` on every sphere block, as the response probe projects its
        // contraction and read-out: project each column's coordinate slots, then each row's.
        let sphere_tangents = self.sphere_tangent_blocks(&cache.row_dims)?;
        if !sphere_tangents.is_empty() {
            for mut column in g.axis_iter_mut(ndarray::Axis(1)) {
                project_sphere_tangent_slots(&sphere_tangents, &cache.row_offsets, &mut column);
            }
            for mut g_row in g.axis_iter_mut(ndarray::Axis(0)) {
                project_sphere_tangent_slots(&sphere_tangents, &cache.row_offsets, &mut g_row);
            }
        }
        Ok(g)
    }
}
