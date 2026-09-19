//! Stage E: compile the best retained response of a known block into a small executable function (#2946).
//!
//! # Draws
//!
//! Under the declared law the retained coordinates `s = Qᵀ Z` are `N(0, I_k)`, so training and held-out points are drawn
//! from that law directly, which is data generation and not a finite difference, and each is read at `z = Q s`. The
//! draws come from the repository's one owner: the SplitMix64 stream `gam_linalg::utils::splitmix64` seeded by the
//! caller, transformed by inversion in `gam_math::probability::standard_normal_from_uniform_bits`. The stream continues
//! from the training draws into the held-out draws, and the seed is recorded; a re-derivation on another platform holds
//! only to its libm rounding, so a consumer reads the recorded compile rather than re-deriving the draws.
//!
//! # Fit
//!
//! The response `F̄_P` is whitened by the output metric, `ỹ = Lᵀ y` with `M = L Lᵀ`, so `‖y − ŷ‖_M = ‖ỹ − ŷ̃‖`, and
//! centered. Directions whose training variance lies inside the rounding band of the response Gram (its eigensolver's
//! backward error plus the Gram's formation error) carry no response the draws can resolve, so the response is rotated
//! onto its principal directions and only the resolvable ones are fitted; the others stay at the training mean.
//!
//! Every fitted direction is fitted at once, on one Duchon design in the `k` retained coordinates, by the one owner of
//! exact multi-response, multi-penalty Gaussian REML (`gam_solve::gaussian_reml_multi_penalty`): one reduction of the
//! design for all directions, one smoothing vector and one dispersion. In the whitened output a shared dispersion is
//! exactly the metric likelihood `y ~ N(g, σ² M⁻¹)` of the vector output, and one smoothing vector gives the vector
//! function one smoothness. The coefficients are the posterior mean at the REML smoothing vector (for a Gaussian mean the
//! posterior mode is the posterior mean), and only a converged, certified optimum yields `g`: any refusal is typed and no
//! partial `g` exists. With a shared smoothing vector and dispersion the fit is equivariant under rotations of the
//! output, so the principal rotation only drops unresolvable directions.
//!
//! The penalties are the default Duchon set: the curvature Gram, the null-space trend ridge, and the mass and tension
//! operators, each with its own REML strength, so the prior on `g` shrinks toward the null (SPEC 12, 14). The ridge
//! penalizes the slopes of the curvature Gram's polynomial null space, and every penalty annihilates the constant, so
//! the joint null space the fit declares is the unpenalized intercept (SPEC 12: an intercept carries no penalty).
//!
//! `F̄_P` is exact, so the residual is pure basis-approximation error. Only the intercept is unpenalized and the
//! response is centered, so every fitted direction carries penalized energy: a response the design reproduces, such as
//! a linear `F̄_P`, is not an interpolation without a finite optimum. The penalties it loads rail at the lower edge of
//! the owner's resolvability domain, where each is unpenalized to the criterion gradient's resolution, and the fit
//! reports every penalty's placement. At that edge a penalized direction is shrunk by at most `√ε` of itself, so such a
//! response is reproduced to rounding. Any refusal of the owner, the #2723 refusal of a profile with no finite
//! dispersion among them, is returned typed.
//!
//! The fitted basis is frozen (centers, radial chart, identifiability transform), so the compiled function is replayed
//! exactly at any new point.
//!
//! # Error split
//!
//! R2 gives `E‖F − g(PZ)‖²_M = E(P) + A` with `A = E‖F̄_P − g‖²_M`, because `F − F̄_P` is orthogonal to every function
//! of `PZ`. `E(P)` is the operator's exact value. `A` is estimated by the mean of `‖F̄_P − g‖²_M` over draws
//! independent of the training draws: conditional on `g` the mean is unbiased, and its standard error is the sample
//! deviation over `√n`.
//!
//! # Dominance call
//!
//! The frame acts only on the first term and the function only on the second. Over the frames containing `P` the first
//! term ranges over `[0, E(P)]`; over the functions of `PZ` the second has infimum `0`, at `g = F̄_P`. So `E(P)` bounds
//! what any frame move can remove and `A` bounds what any enrichment of `g` can remove, and the call names the
//! component that dominates: add a direction when `E(P) > A`, enrich `g` otherwise. It compares upper bounds on the
//! achievable reductions, not the gains of the next step. `A` is known through `Â` with standard error `se`, so the
//! call compares `E(P)` with `Â`, the posterior mean of `A` under a flat prior on its central-limit likelihood, with no
//! threshold, and reports `Φ(−|E(P) − Â|/se)`, the probability that the true order is the other one. A tie enriches
//! `g`: an enrichment leaves the reader frame unchanged.
//!
//! The next steps' measured gains are reported beside the call. The frame step is cheap and always measured: the
//! frame is extended by the leading direction of the horizontal gradient `∇_Q V`, and `E(P) − E(P ⊕ q*)` is exact. The
//! enrichment step refits `g` at the next resolution, so [`enrichment_step`] runs it only when asked, and reports
//! `Â − Â_enriched` with its standard error.
//!
//! # Price
//!
//! The ledger is a report of counts; it is never summed into a selection score, since a count is not a code length
//! (#2933 F20). A frame of rank `r` in `ℝⁿ` has the Stiefel count `r·n − r(r + 1)/2`; when the function read through it
//! is invariant under rotations of its coordinates, `(Q, g) ~ (QR, g∘Rᵀ)` removes `dim O(r) = r(r − 1)/2` and leaves
//! the Grassmann count `r(n − r)` ([`frame_parameter_count`]). The reader frame's gauge follows from the declared smooth
//! class: an isotropic, non-periodic Duchon smooth is rotation invariant. The writer frame of the `r` fitted output
//! directions always has the gauge, because every direction's coefficients live on one shared design, so
//! `(V_r, B) ~ (V_r R, B R)`. Beside the frames the ledger counts the output mean (`m`), the function coefficients
//! (`p·r`, with the one effective-degrees-of-freedom figure of the representation beside them), the connections of `g`
//! (`k·r` edges from retained coordinates to output directions) and the residual. A residual is either left
//! unexplained, and priced as the error `E(P) + Â`, or executed. Only the original block executes it, so an executed
//! residual is charged the block's every parameter, `h(d + 1 + m) + m` with the output bias: the untouched block is
//! never a free residual mechanism.

use super::subspace::{FrameGradient, KnownBlock, ResponseError};
use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::{accumulation_growth, symmetric_spectrum_rounding_band};
use gam_linalg::utils::splitmix64;
use gam_math::probability::{normal_cdf, standard_normal_from_uniform_bits};
use gam_problem::EstimationError;
use gam_solve::gaussian_reml_multi_penalty::{GaussianRemlMultiPenaltyProblem, GaussianRemlMultiPenaltyRhoPlacement};
use gam_terms::basis::{
    BasisBuildResult, BasisMetadata, CenterStrategy, DuchonBasisSpec, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis, default_num_centers,
    duchon_cubic_default, refined_num_centers, starting_num_centers,
};
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::fmt;

/// A refusal to compile a retained response.
#[derive(Debug)]
pub enum CompileError {
    Response(ResponseError),
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    /// The experiment cannot support the compile: no retained coordinate, fewer than two held-out draws, or a basis at
    /// least as wide as the training draws, which leaves REML no residual degrees of freedom.
    InvalidDesign {
        reason: String,
    },
    /// The Duchon basis could not be built, or its fit-time state cannot be replayed at new points.
    Basis {
        context: &'static str,
        message: String,
    },
    /// A standard-normal draw was refused by its owner.
    Draw {
        message: String,
    },
    /// The multi-response REML fit refused or did not certify convergence; no partial `g` exists.
    Fit(EstimationError),
    NonFinite {
        context: &'static str,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Response(error) => write!(f, "retained-response operator: {error}"),
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected}, got {got}"),
            Self::InvalidDesign { reason } => write!(f, "compile design refused: {reason}"),
            Self::Basis { context, message } => write!(f, "{context}: {message}"),
            Self::Draw { message } => write!(f, "standard-normal draw refused: {message}"),
            Self::Fit(error) => write!(f, "multi-response REML fit of the compiled response failed: {error}"),
            Self::NonFinite { context } => write!(f, "{context} holds a non-finite entry"),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Response(error) => Some(error),
            Self::Fit(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ResponseError> for CompileError {
    fn from(error: ResponseError) -> Self {
        Self::Response(error)
    }
}

/// The declared experiment of one compile: how many draws train the smooth, how many estimate its error, and the
/// Duchon resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileDesign {
    pub training_draws: usize,
    pub holdout_draws: usize,
    pub centers: usize,
}

impl CompileDesign {
    /// The pilot resolution: gam-terms' starting center count for `training_draws` rows in `retained_dim` coordinates,
    /// over the cubic Duchon default's affine null space (`retained_dim + 1` columns).
    pub fn pilot(training_draws: usize, holdout_draws: usize, retained_dim: usize) -> Self {
        Self {
            training_draws,
            holdout_draws,
            centers: starting_num_centers(training_draws, retained_dim, retained_dim.saturating_add(1)),
        }
    }

    /// The same experiment at gam-terms' next evidence-backed resolution, or `None` once the production ceiling
    /// `default_num_centers` is reached.
    pub fn enriched(&self, retained_dim: usize) -> Option<Self> {
        let ceiling = default_num_centers(self.training_draws, retained_dim);
        let centers = refined_num_centers(self.centers).min(ceiling);
        (centers > self.centers).then_some(Self {
            centers,
            ..*self
        })
    }
}

/// How the compiled function represents `F̄_P`.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionRepresentation {
    /// No output direction has training variance above its rounding band: `g` is the training mean.
    Constant,
    /// The REML posterior mean at the shared smoothing vector.
    Reml {
        /// One strength per default Duchon penalty, in the basis builder's penalty order.
        smoothing_parameters: Vec<f64>,
        /// Per penalty, interior or railed at an edge of the owner's resolvability domain, where it is switched off
        /// (upper) or unpenalized (lower) to the criterion gradient's resolution: a structural result, not a wall.
        placement: Vec<GaussianRemlMultiPenaltyRhoPlacement>,
        /// The one effective degrees of freedom of the shared smoother, `tr(K⁻¹XᵀX)`.
        smoother_edf: f64,
    },
}

impl FunctionRepresentation {
    /// The one effective-degrees-of-freedom figure of the representation: `0` for the mean, the REML figure otherwise.
    pub fn effective_degrees_of_freedom(&self) -> f64 {
        match self {
            Self::Constant => 0.0,
            Self::Reml { smoother_edf, .. } => *smoother_edf,
        }
    }
}

/// R2's two terms, reported separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorSplit {
    /// `E(P) = E‖F − F̄_P‖²_M`, the operator's exact discarded-input error.
    pub discarded_error: f64,
    /// `Â`, the held-out mean of `‖F̄_P − g‖²_M`.
    pub function_error: f64,
    /// The standard error of `Â`.
    pub function_error_standard_error: f64,
    pub holdout_draws: usize,
}

impl ErrorSplit {
    /// `E(P) + Â`, the estimate of `E‖F − g(PZ)‖²_M`.
    pub fn total(&self) -> f64 {
        self.discarded_error + self.function_error
    }

    /// Which component dominates; see the module documentation for what the call does and does not compare.
    pub fn dominance_call(&self) -> DominanceCall {
        let action = if self.discarded_error > self.function_error {
            CompileAction::AddDirection
        } else {
            CompileAction::EnrichFunction
        };
        let gap = (self.discarded_error - self.function_error).abs();
        let reversal_probability = if self.function_error_standard_error > 0.0 {
            normal_cdf(-gap / self.function_error_standard_error)
        } else if gap > 0.0 {
            0.0
        } else {
            0.5
        };
        DominanceCall {
            action,
            reversal_probability,
        }
    }
}

/// The move whose error component dominates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileAction {
    /// `E(P)` is larger: extend the reader frame.
    AddDirection,
    /// `Â` is at least as large: enrich the compiled function.
    EnrichFunction,
}

/// A dominance call between the split's components: upper bounds on the achievable reductions, not next-step gains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DominanceCall {
    pub action: CompileAction,
    /// `Φ(−|E(P) − Â|/se)`: the probability, under the central-limit law of `Â`, that `A` lies on the other side of
    /// `E(P)`.
    pub reversal_probability: f64,
}

/// The measured gain of extending the frame by one direction along the frame gradient.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStepGain {
    /// `q*`, the unit leading direction of the horizontal gradient `∇_Q V`, orthogonal to the frame.
    pub direction: Array1<f64>,
    /// `E(P ⊕ q*)`, exact.
    pub discarded_error_after: f64,
    /// `E(P) − E(P ⊕ q*)`.
    pub gain: f64,
}

/// The measured gain of refitting `g` at the next resolution.
#[derive(Debug, Clone)]
pub struct EnrichmentStep {
    /// The compile at the enriched design, on the same frame.
    pub enriched: CompiledResponse,
    /// `Â − Â_enriched`.
    pub gain: f64,
    /// The two estimates come from independent held-out draws, so the standard errors add in quadrature.
    pub gain_standard_error: f64,
}

/// The residual `F − g(PZ)`, priced both ways.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualPrice {
    /// Left unexplained, the residual is the compiled mechanism's error `E(P) + Â`.
    pub unexplained_error: f64,
    /// The standard error of `unexplained_error`, all of it from `Â`.
    pub unexplained_standard_error: f64,
    /// Executed, the residual needs the original block, charged at its every parameter `h(d + 1 + m) + m`: readers,
    /// biases, writers and the output bias.
    pub executed_parameters: usize,
}

/// The counts of a compiled mechanism: a report, never summed into a selection score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MechanismPrice {
    /// The reader frame's count under the declared smooth class's gauge.
    pub reader_frame: usize,
    /// `r(m − r)`, the Grassmann count of the fitted output directions in the whitened output.
    pub writer_frame: usize,
    /// `m`, the output mean.
    pub output_mean: usize,
    /// `p·r`, one Duchon coefficient vector per fitted output direction.
    pub function_coefficients: usize,
    /// [`FunctionRepresentation::effective_degrees_of_freedom`], the one figure the representation carries.
    pub function_edf: f64,
    /// `k·r`, the edges from retained coordinates to fitted output directions.
    pub connections: usize,
    pub residual: ResidualPrice,
}

/// The parameter count of a rank-`rank` frame in `ambient` dimensions: the Stiefel count `rank·ambient − rank(rank+1)/2`,
/// less `dim O(rank) = rank(rank − 1)/2` when the function read through the frame is rotation invariant, which leaves the
/// Grassmann count `rank(ambient − rank)`.
pub fn frame_parameter_count(rank: usize, ambient: usize, rotation_gauge: bool) -> usize {
    let stiefel = rank * ambient - rank * (rank + 1) / 2;
    if rotation_gauge {
        stiefel - rank * rank.saturating_sub(1) / 2
    } else {
        stiefel
    }
}

/// A compiled retained response `g`: an executable function of the retained coordinates, its error split, the
/// dominance call and measured frame step, and its price.
#[derive(Debug, Clone)]
pub struct CompiledResponse {
    /// `Q`, `d × k`.
    frame: Array2<f64>,
    /// The fit-time Duchon basis, frozen for replay.
    basis: DuchonBasisSpec,
    /// `p × r`, one column per fitted output direction.
    coefficients: Array2<f64>,
    /// The fit-time active Duchon penalties, `p × p` each, in the basis builder's order.
    penalties: Vec<Array2<f64>>,
    /// `p × M_p`, the declared joint null space `∩_k ker S_k`.
    null_space: Array2<f64>,
    /// `L⁻ᵀ V_r`, `m × r`: the fitted output directions in the original output coordinates.
    writer: Array2<f64>,
    /// The training mean of `F̄_P`, length `m`.
    output_mean: Array1<f64>,
    design: CompileDesign,
    /// The seed of the draw stream the compile trained and held out on.
    seed: u64,
    representation: FunctionRepresentation,
    split: ErrorSplit,
    frame_step: Option<FrameStepGain>,
    price: MechanismPrice,
}

impl CompiledResponse {
    /// `Q`, the reader frame the compiled function reads through.
    pub fn frame(&self) -> ArrayView2<'_, f64> {
        self.frame.view()
    }

    pub fn retained_dim(&self) -> usize {
        self.frame.ncols()
    }

    pub fn output_dim(&self) -> usize {
        self.output_mean.len()
    }

    /// `r`, the number of output directions the training draws resolved.
    pub fn fitted_directions(&self) -> usize {
        self.coefficients.ncols()
    }

    /// The Duchon coefficients, `p × r`, one column per fitted output direction in the whitened principal frame.
    pub fn coefficients(&self) -> ArrayView2<'_, f64> {
        self.coefficients.view()
    }

    /// The fit-time active Duchon penalties on the coefficients of [`Self::basis_rows`], `p × p` each, in the basis
    /// builder's order (curvature, null-space trend ridge, mass, tension), each with its own REML strength.
    pub fn penalties(&self) -> &[Array2<f64>] {
        &self.penalties
    }

    /// `p × M_p`, an orthonormal basis of the joint null space `∩_k ker S_k` the fit declared to the REML owner: the
    /// unpenalized intercept.
    pub fn null_space(&self) -> ArrayView2<'_, f64> {
        self.null_space.view()
    }

    /// `L⁻ᵀ V_r`, `m × r`, which maps the fitted directions back to the original output coordinates.
    pub fn writer(&self) -> ArrayView2<'_, f64> {
        self.writer.view()
    }

    pub fn design(&self) -> CompileDesign {
        self.design
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn representation(&self) -> FunctionRepresentation {
        self.representation.clone()
    }

    pub fn split(&self) -> ErrorSplit {
        self.split
    }

    pub fn dominance_call(&self) -> DominanceCall {
        self.split.dominance_call()
    }

    /// The measured frame step, or `None` when the frame already spans the input or the frame gradient vanishes.
    pub fn frame_step(&self) -> Option<&FrameStepGain> {
        self.frame_step.as_ref()
    }

    pub fn price(&self) -> MechanismPrice {
        self.price
    }

    /// `Φ(s)`, the frozen Duchon design at each row `s` of `coordinates` (`n × k`), as an `n × p` matrix: the rows the
    /// compiled function is evaluated on.
    pub fn basis_rows(&self, coordinates: ArrayView2<'_, f64>) -> Result<Array2<f64>, CompileError> {
        require_length("compiled-response coordinate columns", self.retained_dim(), coordinates.ncols())?;
        require_finite("compiled-response coordinates", coordinates.iter())?;
        let rows = replayed_design(&self.basis, coordinates)?;
        require_length("replayed Duchon design columns", self.coefficients.nrows(), rows.ncols())?;
        Ok(rows)
    }

    /// `g(s)` at each row `s` of `coordinates` (`n × k`), as an `n × m` matrix.
    pub fn evaluate_coordinates(&self, coordinates: ArrayView2<'_, f64>) -> Result<Array2<f64>, CompileError> {
        let rows = self.basis_rows(coordinates)?;
        let mut values = Array2::<f64>::zeros((coordinates.nrows(), self.output_dim()));
        values.rows_mut().into_iter().for_each(|mut row| row.assign(&self.output_mean));
        if self.fitted_directions() > 0 {
            let components = fast_ab(&rows, &self.coefficients);
            values += &fast_abt(&components, &self.writer);
        }
        Ok(values)
    }

    /// `g(Qᵀ z)` at each row `z` of `points` (`n × d`), as an `n × m` matrix.
    pub fn evaluate_input(&self, points: ArrayView2<'_, f64>) -> Result<Array2<f64>, CompileError> {
        require_length("compiled-response input columns", self.frame.nrows(), points.ncols())?;
        self.evaluate_coordinates(fast_ab(&points, &self.frame).view())
    }
}

/// Compile the best retained response of `block` through `frame` (`d × k`, orthonormal columns) under `design`, drawing
/// from the stream seeded by `seed`.
pub fn compile_retained_response(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    design: CompileDesign,
    seed: u64,
) -> Result<CompiledResponse, CompileError> {
    // The gradient pass validates the frame against the block before anything is drawn, and yields `E(P)` with it.
    let gradient = block.explained_variance_gradient(frame)?;
    let discarded_error = gradient.discarded_error.value;
    let input_dim = block.input_dim();
    let output_dim = block.output_dim();
    let retained_dim = frame.ncols();
    if retained_dim == 0 {
        return Err(CompileError::InvalidDesign {
            reason: "a compiled response needs at least one retained coordinate".to_string(),
        });
    }
    if design.holdout_draws < 2 {
        return Err(CompileError::InvalidDesign {
            reason: format!(
                "the function-error standard error needs at least two held-out draws, got {}",
                design.holdout_draws
            ),
        });
    }
    if design.centers >= design.training_draws {
        return Err(CompileError::InvalidDesign {
            reason: format!(
                "{} Duchon centers for {} training draws leave REML no residual degrees of freedom",
                design.centers, design.training_draws
            ),
        });
    }
    let frame_step = next_frame_step(block, frame, &gradient)?;
    let metric_factor = block
        .metric()
        .cholesky(Side::Lower)
        .map_err(|error| ResponseError::MetricNotPositiveDefinite {
            reason: error.to_string(),
        })?;
    let lower = metric_factor.lower_triangular();

    let mut state = seed;
    let training = standard_normal_draws(&mut state, design.training_draws, retained_dim)?;
    let training_response = block.retained_response(frame, fast_abt(&training, &frame).view())?;
    let output_mean = training_response
        .mean_axis(Axis(0))
        .ok_or(CompileError::InvalidDesign {
            reason: "no training draws".to_string(),
        })?;
    let whitened = fast_ab(&(&training_response - &output_mean), &lower);
    let entry_rounding = centered_whitened_rounding(&training_response, &output_mean, &lower);
    let (directions, components) = principal_components(whitened, entry_rounding)?;
    let fitted_directions = directions.ncols();

    let (nullspace_order, power) = duchon_cubic_default(retained_dim);
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::EqualMass {
            num_centers: design.centers,
        },
        periodic: None,
        length_scale: None,
        power,
        nullspace_order,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: None,
    };
    let built = build_duchon_basis(training.view(), &spec).map_err(|error| CompileError::Basis {
        context: "Duchon basis of the retained coordinates",
        message: error.to_string(),
    })?;
    let basis = replay_spec(&built.metadata)?;
    let training_design = dense_design(&built)?;
    let width = training_design.ncols();
    if width >= design.training_draws {
        return Err(CompileError::InvalidDesign {
            reason: format!(
                "the realized Duchon design has {width} columns for {} training draws",
                design.training_draws
            ),
        });
    }
    let penalties: Vec<Array2<f64>> = built.active_penalties.iter().map(|penalty| penalty.matrix.clone()).collect();
    let null_space = declared_joint_null_space(&built)?;
    let (coefficients, representation) = if fitted_directions == 0 {
        (Array2::<f64>::zeros((width, 0)), FunctionRepresentation::Constant)
    } else {
        let problem = GaussianRemlMultiPenaltyProblem::new(
            training_design.view(),
            components.view(),
            &penalties,
            null_space.ncols(),
        )
        .map_err(CompileError::Fit)?;
        let fit = problem.fit(None).map_err(CompileError::Fit)?;
        require_length("fitted Duchon coefficient rows", width, fit.coefficients.nrows())?;
        require_length("fitted output directions", fitted_directions, fit.coefficients.ncols())?;
        (
            fit.coefficients,
            FunctionRepresentation::Reml {
                smoothing_parameters: fit.evaluation.lambdas.to_vec(),
                placement: fit.rho_placement,
                smoother_edf: fit.evaluation.edf,
            },
        )
    };
    require_finite("fitted Duchon coefficients", coefficients.iter())?;
    // `L⁻ᵀ V = (L Lᵀ)⁻¹ L V`, from the factor already in hand.
    let writer = metric_factor.solve_mat(&fast_ab(&lower, &directions));
    let reader_frame = frame_parameter_count(retained_dim, input_dim, rotation_invariant(&basis));
    let function_edf = representation.effective_degrees_of_freedom();

    let mut compiled = CompiledResponse {
        frame: frame.to_owned(),
        basis,
        coefficients,
        penalties,
        null_space,
        writer,
        output_mean,
        design,
        seed,
        representation,
        split: ErrorSplit {
            discarded_error,
            function_error: 0.0,
            function_error_standard_error: 0.0,
            holdout_draws: design.holdout_draws,
        },
        frame_step,
        price: MechanismPrice {
            reader_frame,
            writer_frame: frame_parameter_count(fitted_directions, output_dim, true),
            output_mean: output_dim,
            function_coefficients: width * fitted_directions,
            function_edf,
            connections: retained_dim * fitted_directions,
            residual: ResidualPrice {
                unexplained_error: 0.0,
                unexplained_standard_error: 0.0,
                // Readers `h·d`, biases `h`, writers `m·h` and the output bias `m`.
                executed_parameters: block.width() * (input_dim + 1 + output_dim) + output_dim,
            },
        },
    };

    let holdout = standard_normal_draws(&mut state, design.holdout_draws, retained_dim)?;
    let holdout_response = block.retained_response(frame, fast_abt(&holdout, &frame).view())?;
    let residual = &holdout_response - &compiled.evaluate_coordinates(holdout.view())?;
    let whitened_residual = fast_ab(&residual, &lower);
    let errors: Vec<f64> = whitened_residual.rows().into_iter().map(|row| row.dot(&row)).collect();
    let count = errors.len() as f64;
    let function_error = errors.iter().sum::<f64>() / count;
    let deviation = errors.iter().map(|error| (error - function_error).powi(2)).sum::<f64>() / (count - 1.0);
    let function_error_standard_error = (deviation / count).sqrt();
    if !(function_error.is_finite() && function_error_standard_error.is_finite()) {
        return Err(CompileError::NonFinite {
            context: "held-out function error",
        });
    }
    compiled.split.function_error = function_error;
    compiled.split.function_error_standard_error = function_error_standard_error;
    compiled.price.residual.unexplained_error = compiled.split.total();
    compiled.price.residual.unexplained_standard_error = function_error_standard_error;
    Ok(compiled)
}

/// Recompile `compiled` at the next resolution on the same frame, drawing from the stream seeded by `seed`, and measure
/// `Â − Â_enriched`, or `None` once the design is at gam-terms' production ceiling. `block` must be the block
/// `compiled` was compiled from.
pub fn enrichment_step(
    block: &KnownBlock,
    compiled: &CompiledResponse,
    seed: u64,
) -> Result<Option<EnrichmentStep>, CompileError> {
    let Some(design) = compiled.design.enriched(compiled.retained_dim()) else {
        return Ok(None);
    };
    let enriched = compile_retained_response(block, compiled.frame(), design, seed)?;
    let before = compiled.split;
    let after = enriched.split;
    Ok(Some(EnrichmentStep {
        gain: before.function_error - after.function_error,
        gain_standard_error: before
            .function_error_standard_error
            .hypot(after.function_error_standard_error),
        enriched,
    }))
}

/// Extend the frame by `q*`, the leading direction of the horizontal gradient `G = ∇_Q V` (`d × k`), and measure the
/// exact `E(P ⊕ q*)`. `q* = G v/‖G v‖` with `v` the leading eigenvector of `GᵀG` (`k × k`), so no `d × d` matrix is
/// formed.
fn next_frame_step(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    gradient: &FrameGradient,
) -> Result<Option<FrameStepGain>, CompileError> {
    let (input_dim, retained_dim) = frame.dim();
    if retained_dim >= input_dim {
        return Ok(None);
    }
    let tangent = &gradient.horizontal_gradient;
    let gram = fast_atb(tangent, tangent);
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(|error| CompileError::Basis {
        context: "leading direction of the frame gradient",
        message: error.to_string(),
    })?;
    let Some(leading) = (0..eigenvalues.len()).max_by(|&left, &right| eigenvalues[left].total_cmp(&eigenvalues[right]))
    else {
        return Ok(None);
    };
    if !(eigenvalues[leading] > 0.0) {
        return Ok(None);
    }
    let mut direction = tangent.dot(&eigenvectors.column(leading));
    // The tangent is horizontal, so `direction ⊥ span Q` up to rounding; one projection restores it before extending.
    let in_frame = frame.t().dot(&direction);
    direction -= &frame.dot(&in_frame);
    let norm = direction.dot(&direction).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Ok(None);
    }
    direction /= norm;
    let mut extended = Array2::<f64>::zeros((input_dim, retained_dim + 1));
    extended.slice_mut(s![.., ..retained_dim]).assign(&frame);
    extended.column_mut(retained_dim).assign(&direction);
    let discarded_error_after = block.discarded_error(extended.view())?.value;
    Ok(Some(FrameStepGain {
        direction,
        discarded_error_after,
        gain: gradient.discarded_error.value - discarded_error_after,
    }))
}

/// Whether the declared smooth class is invariant under rotations of the retained coordinates: an isotropic,
/// non-periodic Duchon smooth is.
fn rotation_invariant(basis: &DuchonBasisSpec) -> bool {
    basis.aniso_log_scales.is_none() && basis.periodic.is_none()
}

/// `∩_k ker S_k` of the realized penalty set as an orthonormal coefficient-space basis, declared from the penalties' own
/// structure. The curvature Gram's null space is the structural frame `Z` it declares, the polynomial block (#2445), not
/// its numerical nullity: the builder conditions the frame's slopes with a `√ε`-relative ridge inside the curvature
/// matrix, and whether a rank test counts those slopes moves with the center set (ten centers in two dimensions leave a
/// numerical nullity of 1 against a frame of 3). The null-space ridge acts inside the frame and penalizes its slopes, not
/// its constant; mass `Σ(f − f̄)²` and tension `Σ‖∇f‖²` annihilate constants too. So the joint null space is the part of
/// the frame the ridge leaves, of dimension `dim(Z) − rank(S_ridge)` (the constant): `Z` times the eigenvectors of
/// `ZᵀS_ridge Z` outside its `rank(S_ridge)` largest eigenvalues, so the count comes from the ridge's own rank and no
/// threshold is read here. With no realized ridge it is the whole frame. The compile builds without an identifiability
/// transform, so `Z` is the polynomial block's coordinate columns and the basis is orthonormal. The declaration assumes
/// that mass and tension add no rank inside that space beyond what the ridge already penalizes, which holds for the affine
/// null space every `duchon_cubic_default` order realizes. The REML owner checks the declared dimension against its rank
/// predicate and refuses any disagreement, so a refusal there names a wrong declaration, not a solver defect.
fn declared_joint_null_space(built: &BasisBuildResult) -> Result<Array2<f64>, CompileError> {
    let penalty_of = |source: PenaltySource| built.active_penalties.iter().find(|penalty| penalty.info.source == source);
    let curvature = penalty_of(PenaltySource::Primary).ok_or(CompileError::Basis {
        context: "Duchon penalty set",
        message: "the basis realized no curvature penalty".to_string(),
    })?;
    let frame = curvature.info.structural_null_frame.as_ref().ok_or(CompileError::Basis {
        context: "Duchon penalty set",
        message: "the curvature penalty declares no structural null frame".to_string(),
    })?;
    let Some(ridge) = penalty_of(PenaltySource::DoublePenaltyNullspace) else {
        return Ok(frame.clone());
    };
    let frame_dim = frame.ncols();
    let nullity = frame_dim.checked_sub(ridge.info.effective_rank).ok_or(CompileError::Basis {
        context: "Duchon penalty set",
        message: format!(
            "the null-space ridge has rank {} but the curvature penalty's structural null frame has dimension {}",
            ridge.info.effective_rank, frame_dim
        ),
    })?;
    let restricted = fast_atb(frame, &fast_ab(&ridge.matrix, frame));
    let (eigenvalues, eigenvectors) = restricted.eigh(Side::Lower).map_err(|error| CompileError::Basis {
        context: "null-space ridge inside the curvature penalty's structural null frame",
        message: error.to_string(),
    })?;
    let mut order: Vec<usize> = (0..frame_dim).collect();
    order.sort_by(|&left, &right| eigenvalues[left].total_cmp(&eigenvalues[right]));
    let mut kept = Array2::<f64>::zeros((frame_dim, nullity));
    for (column, &index) in order.iter().take(nullity).enumerate() {
        kept.column_mut(column).assign(&eigenvectors.column(index));
    }
    Ok(fast_ab(frame, &kept))
}

/// The rounding band of a Gram `AᵀA` formed from `a` (`n × c`), in eigenvalue units: the backward error of its symmetric
/// eigensolver plus the formation error `‖E‖₂ ≤ Σ_jk γ_n Σ_i |a_ij a_ik| = γ_n Σ_i (Σ_j |a_ij|)²`, since every entry is an
/// inner product of length `n`.
fn gram_eigenvalue_band(a: &Array2<f64>, eigenvalues: &Array1<f64>) -> f64 {
    let row_absolute_sums: f64 = a
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>().powi(2))
        .sum();
    symmetric_spectrum_rounding_band(&eigenvalues.to_vec()) + accumulation_growth(a.nrows()) * row_absolute_sums
}

/// `‖δỸ‖_F`, a bound on the rounding the formation of `Ỹ = (Y − 1μᵀ) L` leaves in its entries. The mean is an
/// accumulation of `n` terms and a division, and centering one subtraction, so
/// `|δc_ij| ≤ γ_{n+2} (|y_ij| + |μ_j| + Σ_i|y_ij|/n)`. Whitening accumulates `m` products per entry, so
/// `|δỹ_ik| ≤ Σ_j (|δc_ij| + γ_m |c_ij|) |L_jk|`. A direction the exact response does not carry has computed singular
/// value at most `‖δỸ‖₂ ≤ ‖δỸ‖_F` (Weyl), so its Gram eigenvalue lies at or below `‖δỸ‖_F²`.
fn centered_whitened_rounding(response: &Array2<f64>, mean: &Array1<f64>, lower: &Array2<f64>) -> f64 {
    let (rows, output_dim) = response.dim();
    let centering_growth = accumulation_growth(rows + 2);
    let whitening_growth = accumulation_growth(output_dim);
    let column_absolute_means: Array1<f64> = response
        .columns()
        .into_iter()
        .map(|column| column.iter().map(|value| value.abs()).sum::<f64>() / rows as f64)
        .collect();
    let mut entry_bounds = Array2::<f64>::zeros((rows, output_dim));
    for ((row, column), bound) in entry_bounds.indexed_iter_mut() {
        let value = response[[row, column]];
        let centered = value - mean[column];
        *bound = centering_growth * (value.abs() + mean[column].abs() + column_absolute_means[column])
            + whitening_growth * centered.abs();
    }
    let absolute_lower = lower.mapv(f64::abs);
    let whitened_bounds = fast_ab(&entry_bounds, &absolute_lower);
    whitened_bounds.iter().map(|bound| bound * bound).sum::<f64>().sqrt()
}

/// The principal directions of the whitened, centered training response `Ỹ` (`n × m`) whose Gram eigenvalue lies above
/// its rounding band, largest first, with the response's components along them: `(V_r, Ỹ V_r)`. The band is the Gram's
/// own formation and eigensolver band plus `entry_rounding²`, the energy [`centered_whitened_rounding`] bounds.
fn principal_components(
    whitened: Array2<f64>,
    entry_rounding: f64,
) -> Result<(Array2<f64>, Array2<f64>), CompileError> {
    require_finite("whitened training response", whitened.iter())?;
    let output_dim = whitened.ncols();
    let gram = fast_atb(&whitened, &whitened);
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(|error| CompileError::Basis {
        context: "principal directions of the training response",
        message: error.to_string(),
    })?;
    let band = gram_eigenvalue_band(&whitened, &eigenvalues) + entry_rounding * entry_rounding;
    let mut order: Vec<usize> = (0..eigenvalues.len())
        .filter(|&index| eigenvalues[index] > band)
        .collect();
    order.sort_by(|&left, &right| eigenvalues[right].total_cmp(&eigenvalues[left]));
    let mut directions = Array2::<f64>::zeros((output_dim, order.len()));
    for (column, &index) in order.iter().enumerate() {
        directions.column_mut(column).assign(&eigenvectors.column(index));
    }
    let components = fast_ab(&whitened, &directions);
    Ok((directions, components))
}

/// The fit-time Duchon state as a spec that rebuilds the same design at any points: the selected centers, the frozen
/// data-metric radial chart and the identifiability transform, with no operator penalties (a replay needs the design
/// only).
fn replay_spec(metadata: &BasisMetadata) -> Result<DuchonBasisSpec, CompileError> {
    let BasisMetadata::Duchon {
        centers,
        power,
        nullspace_order,
        identifiability_transform,
        radial_reparam,
        spectral_basis,
        ..
    } = metadata
    else {
        return Err(CompileError::Basis {
            context: "Duchon replay",
            message: "the Duchon builder returned non-Duchon metadata".to_string(),
        });
    };
    if spectral_basis.is_some() {
        return Err(CompileError::Basis {
            context: "Duchon replay",
            message: "a spectral Duchon basis was realized for an equal-mass center request".to_string(),
        });
    }
    // Without a frozen chart the builder derives one from the rows it is handed, so a replay at new points would
    // realize a different design.
    let Some(radial_reparam) = radial_reparam else {
        return Err(CompileError::Basis {
            context: "Duchon replay",
            message: "the fit carries no frozen radial chart, so the design cannot be replayed at new points".to_string(),
        });
    };
    Ok(DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        periodic: None,
        length_scale: None,
        power: *power,
        nullspace_order: *nullspace_order,
        identifiability: match identifiability_transform {
            Some(transform) => SpatialIdentifiability::FrozenTransform {
                transform: transform.clone(),
            },
            None => SpatialIdentifiability::None,
        },
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::all_disabled(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: Some(radial_reparam.clone()),
    })
}

fn replayed_design(basis: &DuchonBasisSpec, coordinates: ArrayView2<'_, f64>) -> Result<Array2<f64>, CompileError> {
    let built = build_duchon_basis(coordinates, basis).map_err(|error| CompileError::Basis {
        context: "Duchon replay",
        message: error.to_string(),
    })?;
    dense_design(&built)
}

fn dense_design(built: &BasisBuildResult) -> Result<Array2<f64>, CompileError> {
    match built.design.as_dense_ref() {
        Some(matrix) => Ok(matrix.clone()),
        None => built
            .design
            .try_to_dense_by_chunks("compiled retained response design")
            .map_err(|error| CompileError::Basis {
                context: "Duchon design materialization",
                message: error.to_string(),
            }),
    }
}

/// `rows × columns` standard normal draws from the repository's one owner, continuing the stream `state`.
fn standard_normal_draws(state: &mut u64, rows: usize, columns: usize) -> Result<Array2<f64>, CompileError> {
    let mut draws = Array2::<f64>::zeros((rows, columns));
    for slot in draws.iter_mut() {
        *slot = standard_normal_from_uniform_bits(splitmix64(state)).map_err(|message| CompileError::Draw { message })?;
    }
    Ok(draws)
}

fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), CompileError> {
    if expected == got {
        Ok(())
    } else {
        Err(CompileError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

fn require_finite<'a>(
    context: &'static str,
    mut values: impl Iterator<Item = &'a f64>,
) -> Result<(), CompileError> {
    if values.all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(CompileError::NonFinite { context })
    }
}

#[cfg(test)]
#[path = "compile_tests.rs"]
mod compile_tests;
