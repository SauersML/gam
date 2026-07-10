//! Typed, Python-independent optimization of latent coordinates.
//!
//! This module owns the outer manifold optimization contract: request
//! validation, deterministic restarts, trust-region execution, resumable
//! checkpoints, and the stationarity certificate required before a caller may
//! construct a fitted model. Concrete latent likelihoods implement
//! [`LatentCoordinateObjective`]; no FFI type participates in the contract.

use std::error::Error as StdError;
use std::fmt;

use gam_geometry::{
    GeometryError, GeometryResult, ManifoldSpec, RiemannianManifold, RiemannianObjective,
    RiemannianTrustRegion,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The per-observation geometry of the latent coordinates.
///
/// `Sphere` means `S^(latent_dimension - 1)` embedded in
/// `R^latent_dimension`. The full optimization manifold is the product over
/// observations. String aliases deliberately do not belong in this core API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentCoordinateManifold {
    Euclidean,
    Circle,
    Sphere,
    Torus,
}

/// Decoder-domain topology for one ambient latent coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentCoordinateAxisDomain {
    Open,
    Periodic { period: f64 },
}

impl LatentCoordinateManifold {
    /// Domain descriptor consumed by latent basis construction.
    ///
    /// A sphere has no independently periodic ambient axis: its geometry is
    /// carried by the unit-norm manifold constraint and retraction.
    pub fn axis_domains(
        self,
        latent_dimension: usize,
    ) -> Result<Vec<LatentCoordinateAxisDomain>, LatentCoordinateRequestError> {
        validate_manifold_dimension(self, latent_dimension)?;
        let domain = match self {
            Self::Circle | Self::Torus => LatentCoordinateAxisDomain::Periodic {
                period: std::f64::consts::TAU,
            },
            Self::Euclidean | Self::Sphere => LatentCoordinateAxisDomain::Open,
        };
        Ok(vec![domain; latent_dimension])
    }

    fn build(
        self,
        n_observations: usize,
        latent_dimension: usize,
    ) -> GeometryResult<Box<dyn RiemannianManifold>> {
        let per_observation = match self {
            Self::Euclidean => {
                return ManifoldSpec::Euclidean(n_observations * latent_dimension).build();
            }
            Self::Circle => ManifoldSpec::Circle,
            Self::Sphere => ManifoldSpec::Sphere {
                intrinsic_dim: latent_dimension - 1,
            },
            Self::Torus => ManifoldSpec::Torus {
                dim: latent_dimension,
            },
        };
        ManifoldSpec::Product(vec![per_observation; n_observations]).build()
    }
}

/// Controls a latent-coordinate optimization run.
///
/// No defaults are supplied: convergence precision, budget, radius, and
/// restart policy are part of the statistical procedure and must be selected
/// explicitly by the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentCoordinateOptimizationOptions {
    /// Trust-region iterations available to each restart.
    pub max_iterations: usize,
    /// Required relative projected-gradient tolerance, in `(0, 1]`.
    pub stationarity_tolerance: f64,
    /// Initial trust-region radius.
    pub initial_trust_radius: f64,
    /// Hard upper bound on the trust-region radius.
    pub max_trust_radius: f64,
    /// Total number of starts, including the unperturbed start.
    pub restart_count: usize,
    /// Standard deviation of tangent-space perturbations for later starts.
    pub restart_scale: f64,
    /// Seed for deterministic restart perturbations.
    pub seed: u64,
}

/// A fresh start or a continuation from a typed checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentCoordinateStart {
    Initial(Array1<f64>),
    Resume(LatentCoordinateCheckpoint),
}

/// Complete request for the generic latent-coordinate optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentCoordinateOptimizationRequest {
    pub n_observations: usize,
    pub latent_dimension: usize,
    pub manifold: LatentCoordinateManifold,
    pub start: LatentCoordinateStart,
    pub options: LatentCoordinateOptimizationOptions,
}

/// Resume state emitted only from a fully evaluated, non-stationary candidate.
///
/// The original stationarity reference is retained so continuation applies the
/// same certificate across process or wall-clock boundaries instead of silently
/// renormalizing at the checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentCoordinateCheckpoint {
    coordinates: Array1<f64>,
    n_observations: usize,
    latent_dimension: usize,
    manifold: LatentCoordinateManifold,
    stationarity_reference: f64,
    restart_index: usize,
}

impl LatentCoordinateCheckpoint {
    /// Construct binding-friendly resume state while enforcing shape,
    /// finiteness, manifold feasibility, and a valid original reference.
    pub fn new(
        coordinates: Array1<f64>,
        n_observations: usize,
        latent_dimension: usize,
        manifold: LatentCoordinateManifold,
        stationarity_reference: f64,
        restart_index: usize,
    ) -> Result<Self, LatentCoordinateCheckpointError> {
        let expected = validate_dimensions(n_observations, latent_dimension, manifold)?;
        validate_coordinates("checkpoint", coordinates.view(), expected)?;
        if !(stationarity_reference.is_finite() && stationarity_reference >= 0.0) {
            return Err(LatentCoordinateRequestError::InvalidCheckpointReference {
                value: stationarity_reference,
            }
            .into());
        }
        let geometry = manifold.build(n_observations, latent_dimension)?;
        let coordinates = canonicalize_and_validate_point(geometry.as_ref(), coordinates)?;
        Ok(Self {
            coordinates,
            n_observations,
            latent_dimension,
            manifold,
            stationarity_reference,
            restart_index,
        })
    }

    pub fn coordinates(&self) -> ArrayView1<'_, f64> {
        self.coordinates.view()
    }

    pub const fn n_observations(&self) -> usize {
        self.n_observations
    }

    pub const fn latent_dimension(&self) -> usize {
        self.latent_dimension
    }

    pub const fn manifold(&self) -> LatentCoordinateManifold {
        self.manifold
    }

    pub const fn stationarity_reference(&self) -> f64 {
        self.stationarity_reference
    }

    pub const fn restart_index(&self) -> usize {
        self.restart_index
    }

    pub fn into_coordinates(self) -> Array1<f64> {
        self.coordinates
    }
}

/// Exact evidence used to accept or reject the best restart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentCoordinateStationarityEvidence {
    pub objective_value: f64,
    pub projected_gradient_norm: f64,
    pub stationarity_reference: f64,
    pub relative_gradient: f64,
    pub tolerance: f64,
    pub coordinate_spread: f64,
    pub restart_index: usize,
    pub restart_count: usize,
    pub iteration_budget: usize,
    pub objective_evaluations: usize,
    pub hessian_vector_evaluations: usize,
}

impl LatentCoordinateStationarityEvidence {
    /// Whether this evidence satisfies the optimizer's first-order contract.
    pub fn certifies_stationarity(&self) -> bool {
        self.relative_gradient.is_finite() && self.relative_gradient <= self.tolerance
    }
}

impl fmt::Display for LatentCoordinateStationarityEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "latent-coordinate optimization did not reach stationarity at restart {} of {}: \
             relative gradient {:.6e} exceeds tolerance {:.6e} (projected gradient {:.6e}, \
             stationarity reference {:.6e}, objective {:.9e}, iteration budget {})",
            self.restart_index,
            self.restart_count,
            self.relative_gradient,
            self.tolerance,
            self.projected_gradient_norm,
            self.stationarity_reference,
            self.objective_value,
            self.iteration_budget,
        )
    }
}

/// A latent coordinate vector that passed the required stationarity test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentCoordinateOptimizationResult {
    pub coordinates: Array1<f64>,
    pub evidence: LatentCoordinateStationarityEvidence,
}

/// Value and ambient Euclidean differential returned by an objective.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentCoordinateEvaluation {
    pub objective_value: f64,
    pub euclidean_gradient: Array1<f64>,
}

/// A concrete likelihood supplies the value and analytic latent derivatives.
///
/// Objective failures remain in the associated error type all the way through
/// [`optimize_latent_coordinates`]. An implementation must not translate a
/// failed inner solve into an infinite value or a zero gradient.
pub trait LatentCoordinateObjective {
    type Error: StdError + 'static;

    fn value_and_gradient(
        &mut self,
        coordinates: ArrayView1<'_, f64>,
    ) -> Result<LatentCoordinateEvaluation, Self::Error>;

    /// Optional analytic Riemannian Hessian-vector product.
    ///
    /// Returning `None` selects the trust region's Cauchy model. This hook is
    /// intentionally analytic-only; callers must not approximate it with
    /// finite differences in production.
    fn hessian_vector_product(
        &mut self,
        _coordinates: ArrayView1<'_, f64>,
        _tangent: ArrayView1<'_, f64>,
    ) -> Result<Option<Array1<f64>>, Self::Error> {
        Ok(None)
    }
}

/// Structural request failures detected before an objective is optimized.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LatentCoordinateRequestError {
    #[error("n_observations must be positive")]
    EmptyObservations,
    #[error("latent_dimension must be positive")]
    EmptyLatentDimension,
    #[error(
        "n_observations * latent_dimension overflows usize ({n_observations} * {latent_dimension})"
    )]
    DimensionOverflow {
        n_observations: usize,
        latent_dimension: usize,
    },
    #[error("circle latent coordinates require latent_dimension == 1, got {latent_dimension}")]
    CircleDimension { latent_dimension: usize },
    #[error("sphere latent coordinates require latent_dimension >= 2, got {latent_dimension}")]
    SphereDimension { latent_dimension: usize },
    #[error(
        "{origin} coordinate length must equal n_observations * latent_dimension = {expected}, got {actual}"
    )]
    CoordinateLength {
        origin: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{origin} coordinate {index} must be finite, got {value}")]
    NonFiniteCoordinate {
        origin: &'static str,
        index: usize,
        value: f64,
    },
    #[error("stationarity_tolerance must be finite and in (0, 1], got {value}")]
    InvalidStationarityTolerance { value: f64 },
    #[error("initial_trust_radius must be finite and positive, got {value}")]
    InvalidInitialTrustRadius { value: f64 },
    #[error(
        "max_trust_radius must be finite and at least initial_trust_radius ({initial}), got {maximum}"
    )]
    InvalidMaximumTrustRadius { initial: f64, maximum: f64 },
    #[error("restart_count must be at least one")]
    EmptyRestarts,
    #[error("restart_scale must be finite and positive, got {value}")]
    InvalidRestartScale { value: f64 },
    #[error(
        "checkpoint shape ({checkpoint_observations}, {checkpoint_dimension}) does not match request shape ({request_observations}, {request_dimension})"
    )]
    CheckpointShapeMismatch {
        checkpoint_observations: usize,
        checkpoint_dimension: usize,
        request_observations: usize,
        request_dimension: usize,
    },
    #[error("checkpoint manifold {checkpoint:?} does not match request manifold {request:?}")]
    CheckpointManifoldMismatch {
        checkpoint: LatentCoordinateManifold,
        request: LatentCoordinateManifold,
    },
    #[error("checkpoint stationarity reference must be finite and non-negative, got {value}")]
    InvalidCheckpointReference { value: f64 },
}

/// Invalid numerical data returned through the objective contract.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LatentCoordinateObjectiveContractError {
    #[error("objective received a non-finite coordinate at index {index}: {value}")]
    NonFinitePoint { index: usize, value: f64 },
    #[error("objective value must be finite, got {value}")]
    NonFiniteValue { value: f64 },
    #[error("objective gradient length must be {expected}, got {actual}")]
    GradientLength { expected: usize, actual: usize },
    #[error("objective gradient component {index} must be finite, got {value}")]
    NonFiniteGradient { index: usize, value: f64 },
    #[error("objective gradient norm is not representable as a finite f64")]
    NonFiniteGradientNorm,
    #[error("Hessian-vector product length must be {expected}, got {actual}")]
    HessianVectorLength { expected: usize, actual: usize },
    #[error("Hessian-vector product component {index} must be finite, got {value}")]
    NonFiniteHessianVector { index: usize, value: f64 },
    #[error("Hessian-vector product norm is not representable as a finite f64")]
    NonFiniteHessianVectorNorm,
}

/// Fatal optimization errors, preserving the concrete objective error type.
#[derive(Debug, Error)]
pub enum LatentCoordinateOptimizationError<E: StdError + 'static> {
    #[error(transparent)]
    InvalidRequest(#[from] LatentCoordinateRequestError),
    #[error("latent-coordinate objective failed during restart {restart_index}: {source}")]
    Objective {
        restart_index: usize,
        #[source]
        source: E,
    },
    #[error("invalid latent-coordinate objective output during restart {restart_index}: {source}")]
    InvalidObjectiveOutput {
        restart_index: usize,
        #[source]
        source: LatentCoordinateObjectiveContractError,
    },
    #[error("latent-coordinate geometry failed during restart {restart_index}: {source}")]
    Geometry {
        restart_index: usize,
        #[source]
        source: GeometryError,
    },
    #[error("{evidence}")]
    NonConverged {
        evidence: LatentCoordinateStationarityEvidence,
        checkpoint: LatentCoordinateCheckpoint,
    },
}

impl<E: StdError + 'static> LatentCoordinateOptimizationError<E> {
    pub fn stationarity_evidence(&self) -> Option<&LatentCoordinateStationarityEvidence> {
        match self {
            Self::NonConverged { evidence, .. } => Some(evidence),
            _ => None,
        }
    }

    pub fn checkpoint(&self) -> Option<&LatentCoordinateCheckpoint> {
        match self {
            Self::NonConverged { checkpoint, .. } => Some(checkpoint),
            _ => None,
        }
    }
}

enum ObjectiveBridgeFailure<E> {
    Objective(E),
    Contract(LatentCoordinateObjectiveContractError),
}

struct ObjectiveBridge<'a, O: LatentCoordinateObjective + ?Sized> {
    objective: &'a mut O,
    expected_dimension: usize,
    failure: Option<ObjectiveBridgeFailure<O::Error>>,
    objective_evaluations: usize,
    hessian_vector_evaluations: usize,
}

impl<'a, O: LatentCoordinateObjective + ?Sized> ObjectiveBridge<'a, O> {
    fn new(objective: &'a mut O, expected_dimension: usize) -> Self {
        Self {
            objective,
            expected_dimension,
            failure: None,
            objective_evaluations: 0,
            hessian_vector_evaluations: 0,
        }
    }

    fn fail_contract<T>(
        &mut self,
        failure: LatentCoordinateObjectiveContractError,
    ) -> GeometryResult<T> {
        self.failure = Some(ObjectiveBridgeFailure::Contract(failure));
        Err(GeometryError::InvalidPoint(
            "latent-coordinate objective contract failed",
        ))
    }

    fn checked_value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> GeometryResult<LatentCoordinateEvaluation> {
        self.objective_evaluations += 1;
        if let Some((index, value)) = first_non_finite(point) {
            return self.fail_contract(LatentCoordinateObjectiveContractError::NonFinitePoint {
                index,
                value,
            });
        }
        let evaluation = match self.objective.value_and_gradient(point) {
            Ok(evaluation) => evaluation,
            Err(source) => {
                self.failure = Some(ObjectiveBridgeFailure::Objective(source));
                return Err(GeometryError::InvalidPoint(
                    "latent-coordinate objective evaluation failed",
                ));
            }
        };
        if !evaluation.objective_value.is_finite() {
            return self.fail_contract(LatentCoordinateObjectiveContractError::NonFiniteValue {
                value: evaluation.objective_value,
            });
        }
        if evaluation.euclidean_gradient.len() != self.expected_dimension {
            return self.fail_contract(LatentCoordinateObjectiveContractError::GradientLength {
                expected: self.expected_dimension,
                actual: evaluation.euclidean_gradient.len(),
            });
        }
        if let Some((index, value)) = first_non_finite(evaluation.euclidean_gradient.view()) {
            return self.fail_contract(LatentCoordinateObjectiveContractError::NonFiniteGradient {
                index,
                value,
            });
        }
        if !stable_euclidean_norm(evaluation.euclidean_gradient.view()).is_finite() {
            return self
                .fail_contract(LatentCoordinateObjectiveContractError::NonFiniteGradientNorm);
        }
        Ok(evaluation)
    }

    fn checked_hessian_vector_product(
        &mut self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Option<Array1<f64>>> {
        self.hessian_vector_evaluations += 1;
        let product = match self.objective.hessian_vector_product(point, tangent) {
            Ok(product) => product,
            Err(source) => {
                self.failure = Some(ObjectiveBridgeFailure::Objective(source));
                return Err(GeometryError::InvalidPoint(
                    "latent-coordinate objective Hessian-vector product failed",
                ));
            }
        };
        let Some(product) = product else {
            return Ok(None);
        };
        if product.len() != self.expected_dimension {
            return self.fail_contract(
                LatentCoordinateObjectiveContractError::HessianVectorLength {
                    expected: self.expected_dimension,
                    actual: product.len(),
                },
            );
        }
        if let Some((index, value)) = first_non_finite(product.view()) {
            return self.fail_contract(
                LatentCoordinateObjectiveContractError::NonFiniteHessianVector { index, value },
            );
        }
        if !stable_euclidean_norm(product.view()).is_finite() {
            return self
                .fail_contract(LatentCoordinateObjectiveContractError::NonFiniteHessianVectorNorm);
        }
        Ok(Some(product))
    }
}

impl<O: LatentCoordinateObjective + ?Sized> RiemannianObjective for ObjectiveBridge<'_, O> {
    fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)> {
        let evaluation = self.checked_value_gradient(point)?;
        Ok((evaluation.objective_value, evaluation.euclidean_gradient))
    }

    fn hessian_vector_product(
        &mut self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Option<Array1<f64>>> {
        self.checked_hessian_vector_product(point, tangent)
    }
}

struct Candidate {
    coordinates: Array1<f64>,
    evidence: LatentCoordinateStationarityEvidence,
}

/// Optimize a concrete analytic objective over typed latent-coordinate geometry.
///
/// The function returns a result only when the lowest-objective restart passes
/// the post-hoc projected-gradient certificate. Budget exhaustion or a
/// trust-region stall therefore yields [`LatentCoordinateOptimizationError::NonConverged`]
/// with resume state, never a partial fit-shaped success value.
pub fn optimize_latent_coordinates<O: LatentCoordinateObjective + ?Sized>(
    request: LatentCoordinateOptimizationRequest,
    objective: &mut O,
) -> Result<LatentCoordinateOptimizationResult, LatentCoordinateOptimizationError<O::Error>> {
    validate_request(&request)?;
    let LatentCoordinateOptimizationRequest {
        n_observations,
        latent_dimension,
        manifold: manifold_kind,
        start,
        options,
    } = request;
    let expected_dimension = n_observations * latent_dimension;
    let (base_coordinates, resume_reference) = match start {
        LatentCoordinateStart::Initial(coordinates) => (coordinates, None),
        LatentCoordinateStart::Resume(checkpoint) => {
            let reference = checkpoint.stationarity_reference;
            (checkpoint.coordinates, Some(reference))
        }
    };
    let manifold = manifold_kind
        .build(n_observations, latent_dimension)
        .map_err(|source| LatentCoordinateOptimizationError::Geometry {
            restart_index: 0,
            source,
        })?;
    let base_coordinates = canonicalize_and_validate_point(manifold.as_ref(), base_coordinates)
        .map_err(|source| LatentCoordinateOptimizationError::Geometry {
            restart_index: 0,
            source,
        })?;

    let mut best = optimize_one_restart(
        manifold.as_ref(),
        objective,
        base_coordinates.clone(),
        0,
        &options,
        resume_reference,
    )?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(options.seed);
    for restart_index in 1..options.restart_count {
        let noise = Array1::from_shape_fn(expected_dimension, |_| {
            let standard_normal: f64 = StandardNormal.sample(&mut rng);
            options.restart_scale * standard_normal
        });
        let tangent = manifold
            .project_tangent(base_coordinates.view(), noise.view())
            .map_err(|source| LatentCoordinateOptimizationError::Geometry {
                restart_index,
                source,
            })?;
        let restart_coordinates = manifold
            .retract(base_coordinates.view(), tangent.view())
            .map_err(|source| LatentCoordinateOptimizationError::Geometry {
                restart_index,
                source,
            })?;
        let candidate = optimize_one_restart(
            manifold.as_ref(),
            objective,
            restart_coordinates,
            restart_index,
            &options,
            resume_reference,
        )?;
        if candidate.evidence.objective_value < best.evidence.objective_value {
            best = candidate;
        }
    }

    if best.evidence.certifies_stationarity() {
        return Ok(LatentCoordinateOptimizationResult {
            coordinates: best.coordinates,
            evidence: best.evidence,
        });
    }
    let checkpoint = LatentCoordinateCheckpoint {
        coordinates: best.coordinates,
        n_observations,
        latent_dimension,
        manifold: manifold_kind,
        stationarity_reference: best.evidence.stationarity_reference,
        restart_index: best.evidence.restart_index,
    };
    Err(LatentCoordinateOptimizationError::NonConverged {
        evidence: best.evidence,
        checkpoint,
    })
}

fn optimize_one_restart<O: LatentCoordinateObjective + ?Sized>(
    manifold: &dyn RiemannianManifold,
    objective: &mut O,
    start: Array1<f64>,
    restart_index: usize,
    options: &LatentCoordinateOptimizationOptions,
    resume_reference: Option<f64>,
) -> Result<Candidate, LatentCoordinateOptimizationError<O::Error>> {
    let start = canonicalize_and_validate_point(manifold, start).map_err(|source| {
        LatentCoordinateOptimizationError::Geometry {
            restart_index,
            source,
        }
    })?;
    let mut bridge = ObjectiveBridge::new(objective, start.len());
    let start_evaluation_result = bridge.checked_value_gradient(start.view());
    let start_evaluation =
        translate_bridge_result(&mut bridge, restart_index, start_evaluation_result)?;
    let start_gradient_norm = projected_gradient_norm(
        manifold,
        start.view(),
        start_evaluation.euclidean_gradient.view(),
    )
    .map_err(|source| LatentCoordinateOptimizationError::Geometry {
        restart_index,
        source,
    })?;
    let stationarity_reference = resume_reference.unwrap_or(start_gradient_norm);
    let solver_tolerance = options.stationarity_tolerance * stationarity_reference.max(1.0)
        / start_gradient_norm.max(1.0);
    let trust_region = RiemannianTrustRegion {
        radius: options.initial_trust_radius,
        max_radius: options.max_trust_radius,
        max_iter: options.max_iterations,
        grad_tol: solver_tolerance,
    };
    let optimized_result = trust_region.minimize(manifold, &mut bridge, start.view());
    let optimized = translate_bridge_result(&mut bridge, restart_index, optimized_result)?;
    let final_evaluation_result = bridge.checked_value_gradient(optimized.view());
    let final_evaluation =
        translate_bridge_result(&mut bridge, restart_index, final_evaluation_result)?;
    let final_gradient_norm = projected_gradient_norm(
        manifold,
        optimized.view(),
        final_evaluation.euclidean_gradient.view(),
    )
    .map_err(|source| LatentCoordinateOptimizationError::Geometry {
        restart_index,
        source,
    })?;
    let relative_gradient = relative_stationarity(final_gradient_norm, stationarity_reference);
    let evidence = LatentCoordinateStationarityEvidence {
        objective_value: final_evaluation.objective_value,
        projected_gradient_norm: final_gradient_norm,
        stationarity_reference,
        relative_gradient,
        tolerance: options.stationarity_tolerance,
        coordinate_spread: coordinate_spread(optimized.view()),
        restart_index,
        restart_count: options.restart_count,
        iteration_budget: options.max_iterations,
        objective_evaluations: bridge.objective_evaluations,
        hessian_vector_evaluations: bridge.hessian_vector_evaluations,
    };
    Ok(Candidate {
        coordinates: optimized,
        evidence,
    })
}

fn translate_bridge_result<T, O: LatentCoordinateObjective + ?Sized>(
    bridge: &mut ObjectiveBridge<'_, O>,
    restart_index: usize,
    result: GeometryResult<T>,
) -> Result<T, LatentCoordinateOptimizationError<O::Error>> {
    match result {
        Ok(value) => Ok(value),
        Err(source) => match bridge.failure.take() {
            Some(ObjectiveBridgeFailure::Objective(source)) => {
                Err(LatentCoordinateOptimizationError::Objective {
                    restart_index,
                    source,
                })
            }
            Some(ObjectiveBridgeFailure::Contract(source)) => {
                Err(LatentCoordinateOptimizationError::InvalidObjectiveOutput {
                    restart_index,
                    source,
                })
            }
            None => Err(LatentCoordinateOptimizationError::Geometry {
                restart_index,
                source,
            }),
        },
    }
}

fn canonicalize_and_validate_point(
    manifold: &dyn RiemannianManifold,
    point: Array1<f64>,
) -> GeometryResult<Array1<f64>> {
    let zero = Array1::<f64>::zeros(point.len());
    let canonical = manifold.retract(point.view(), zero.view())?;
    if first_non_finite(canonical.view()).is_some() {
        return Err(GeometryError::InvalidPoint(
            "latent-coordinate retraction produced a non-finite point",
        ));
    }
    // For embedded manifolds this validates feasibility as well as dimension.
    // In particular, SphereManifold rejects non-unit rows here.
    let validation_gradient = manifold.riemannian_gradient(canonical.view(), zero.view())?;
    if first_non_finite(validation_gradient.view()).is_some() {
        return Err(GeometryError::InvalidPoint(
            "latent-coordinate manifold produced a non-finite tangent vector",
        ));
    }
    Ok(canonical)
}

fn projected_gradient_norm(
    manifold: &dyn RiemannianManifold,
    point: ArrayView1<'_, f64>,
    euclidean_gradient: ArrayView1<'_, f64>,
) -> GeometryResult<f64> {
    let gradient = manifold.riemannian_gradient(point, euclidean_gradient)?;
    let norm = stable_euclidean_norm(gradient.view());
    if !norm.is_finite() {
        return Err(GeometryError::InvalidPoint(
            "latent-coordinate Riemannian gradient norm is non-finite",
        ));
    }
    // Every manifold exposed by LatentCoordinateManifold carries the induced
    // ambient identity metric, so this Euclidean norm is exactly its metric
    // norm without allocating a dense ambient-dimension-squared tensor.
    Ok(norm)
}

fn relative_stationarity(gradient_norm: f64, stationarity_reference: f64) -> f64 {
    if !gradient_norm.is_finite() || !stationarity_reference.is_finite() {
        return f64::INFINITY;
    }
    gradient_norm / stationarity_reference.max(1.0)
}

fn stable_euclidean_norm(values: ArrayView1<'_, f64>) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn first_non_finite(values: ArrayView1<'_, f64>) -> Option<(usize, f64)> {
    values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
}

fn coordinate_spread(coordinates: ArrayView1<'_, f64>) -> f64 {
    if coordinates.is_empty() {
        return 0.0;
    }
    let mut count = 0.0_f64;
    let mut mean = 0.0_f64;
    let mut squared_deviation = 0.0_f64;
    for value in coordinates {
        count += 1.0;
        let delta = value - mean;
        mean += delta / count;
        squared_deviation += delta * (value - mean);
    }
    (squared_deviation / count).max(0.0).sqrt()
}

fn validate_request(
    request: &LatentCoordinateOptimizationRequest,
) -> Result<(), LatentCoordinateRequestError> {
    if request.n_observations == 0 {
        return Err(LatentCoordinateRequestError::EmptyObservations);
    }
    if request.latent_dimension == 0 {
        return Err(LatentCoordinateRequestError::EmptyLatentDimension);
    }
    let expected = request
        .n_observations
        .checked_mul(request.latent_dimension)
        .ok_or(LatentCoordinateRequestError::DimensionOverflow {
            n_observations: request.n_observations,
            latent_dimension: request.latent_dimension,
        })?;
    match request.manifold {
        LatentCoordinateManifold::Circle if request.latent_dimension != 1 => {
            return Err(LatentCoordinateRequestError::CircleDimension {
                latent_dimension: request.latent_dimension,
            });
        }
        LatentCoordinateManifold::Sphere if request.latent_dimension < 2 => {
            return Err(LatentCoordinateRequestError::SphereDimension {
                latent_dimension: request.latent_dimension,
            });
        }
        _ => {}
    }
    let options = &request.options;
    if !(options.stationarity_tolerance.is_finite()
        && options.stationarity_tolerance > 0.0
        && options.stationarity_tolerance <= 1.0)
    {
        return Err(LatentCoordinateRequestError::InvalidStationarityTolerance {
            value: options.stationarity_tolerance,
        });
    }
    if !(options.initial_trust_radius.is_finite() && options.initial_trust_radius > 0.0) {
        return Err(LatentCoordinateRequestError::InvalidInitialTrustRadius {
            value: options.initial_trust_radius,
        });
    }
    if !(options.max_trust_radius.is_finite()
        && options.max_trust_radius >= options.initial_trust_radius)
    {
        return Err(LatentCoordinateRequestError::InvalidMaximumTrustRadius {
            initial: options.initial_trust_radius,
            maximum: options.max_trust_radius,
        });
    }
    if options.restart_count == 0 {
        return Err(LatentCoordinateRequestError::EmptyRestarts);
    }
    if !(options.restart_scale.is_finite() && options.restart_scale > 0.0) {
        return Err(LatentCoordinateRequestError::InvalidRestartScale {
            value: options.restart_scale,
        });
    }
    match &request.start {
        LatentCoordinateStart::Initial(coordinates) => {
            validate_coordinates("initial", coordinates.view(), expected)?;
        }
        LatentCoordinateStart::Resume(checkpoint) => {
            if checkpoint.n_observations != request.n_observations
                || checkpoint.latent_dimension != request.latent_dimension
            {
                return Err(LatentCoordinateRequestError::CheckpointShapeMismatch {
                    checkpoint_observations: checkpoint.n_observations,
                    checkpoint_dimension: checkpoint.latent_dimension,
                    request_observations: request.n_observations,
                    request_dimension: request.latent_dimension,
                });
            }
            if checkpoint.manifold != request.manifold {
                return Err(LatentCoordinateRequestError::CheckpointManifoldMismatch {
                    checkpoint: checkpoint.manifold,
                    request: request.manifold,
                });
            }
            if !(checkpoint.stationarity_reference.is_finite()
                && checkpoint.stationarity_reference >= 0.0)
            {
                return Err(LatentCoordinateRequestError::InvalidCheckpointReference {
                    value: checkpoint.stationarity_reference,
                });
            }
            validate_coordinates("checkpoint", checkpoint.coordinates.view(), expected)?;
        }
    }
    Ok(())
}

fn validate_coordinates(
    origin: &'static str,
    coordinates: ArrayView1<'_, f64>,
    expected: usize,
) -> Result<(), LatentCoordinateRequestError> {
    if coordinates.len() != expected {
        return Err(LatentCoordinateRequestError::CoordinateLength {
            origin,
            expected,
            actual: coordinates.len(),
        });
    }
    if let Some((index, value)) = first_non_finite(coordinates) {
        return Err(LatentCoordinateRequestError::NonFiniteCoordinate {
            origin,
            index,
            value,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView1, array};
    use thiserror::Error;

    use super::*;

    #[derive(Debug, Error, PartialEq)]
    enum TestObjectiveError {
        #[error("inner REML solve failed")]
        InnerSolve,
    }

    struct QuadraticObjective {
        target: Array1<f64>,
    }

    impl LatentCoordinateObjective for QuadraticObjective {
        type Error = TestObjectiveError;

        fn value_and_gradient(
            &mut self,
            coordinates: ArrayView1<'_, f64>,
        ) -> Result<LatentCoordinateEvaluation, Self::Error> {
            let gradient = &coordinates - &self.target;
            let objective_value = 0.5 * gradient.dot(&gradient);
            Ok(LatentCoordinateEvaluation {
                objective_value,
                euclidean_gradient: gradient,
            })
        }

        fn hessian_vector_product(
            &mut self,
            _coordinates: ArrayView1<'_, f64>,
            tangent: ArrayView1<'_, f64>,
        ) -> Result<Option<Array1<f64>>, Self::Error> {
            Ok(Some(tangent.to_owned()))
        }
    }

    fn options(max_iterations: usize, restart_count: usize) -> LatentCoordinateOptimizationOptions {
        LatentCoordinateOptimizationOptions {
            max_iterations,
            stationarity_tolerance: 1.0e-10,
            initial_trust_radius: 10.0,
            max_trust_radius: 10.0,
            restart_count,
            restart_scale: 0.2,
            seed: 42,
        }
    }

    fn euclidean_request(
        coordinates: Array1<f64>,
        options: LatentCoordinateOptimizationOptions,
    ) -> LatentCoordinateOptimizationRequest {
        LatentCoordinateOptimizationRequest {
            n_observations: coordinates.len(),
            latent_dimension: 1,
            manifold: LatentCoordinateManifold::Euclidean,
            start: LatentCoordinateStart::Initial(coordinates),
            options,
        }
    }

    #[test]
    fn quadratic_returns_only_a_certified_result() {
        let request = euclidean_request(array![5.0, -3.0], options(2, 1));
        let mut objective = QuadraticObjective {
            target: array![1.0, 2.0],
        };
        let result = optimize_latent_coordinates(request, &mut objective).unwrap();
        assert!(result.evidence.certifies_stationarity());
        assert!(result.evidence.relative_gradient <= 1.0e-10);
        assert!(
            (&result.coordinates - &array![1.0, 2.0])
                .iter()
                .all(|difference| difference.abs() <= 1.0e-12)
        );
    }

    #[test]
    fn exhausted_run_returns_evidence_and_a_serializable_checkpoint() {
        let request = euclidean_request(array![5.0], options(0, 1));
        let mut objective = QuadraticObjective {
            target: array![0.0],
        };
        let error = optimize_latent_coordinates(request, &mut objective).unwrap_err();
        let LatentCoordinateOptimizationError::NonConverged {
            evidence,
            checkpoint,
        } = error
        else {
            panic!("expected typed non-convergence");
        };
        assert!(!evidence.certifies_stationarity());
        assert_eq!(evidence.stationarity_reference, 5.0);
        assert_eq!(checkpoint.coordinates(), array![5.0].view());
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let decoded: LatentCoordinateCheckpoint = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, checkpoint);
    }

    #[test]
    fn resume_preserves_the_original_stationarity_reference() {
        let first_request = euclidean_request(array![5.0], options(0, 1));
        let mut first_objective = QuadraticObjective {
            target: array![0.0],
        };
        let first_error =
            optimize_latent_coordinates(first_request, &mut first_objective).unwrap_err();
        let checkpoint = first_error.checkpoint().unwrap().clone();
        let resumed_request = LatentCoordinateOptimizationRequest {
            n_observations: 1,
            latent_dimension: 1,
            manifold: LatentCoordinateManifold::Euclidean,
            start: LatentCoordinateStart::Resume(checkpoint),
            options: options(2, 1),
        };
        let mut resumed_objective = QuadraticObjective {
            target: array![0.0],
        };
        let result = optimize_latent_coordinates(resumed_request, &mut resumed_objective).unwrap();
        assert_eq!(result.evidence.stationarity_reference, 5.0);
        assert!(result.evidence.certifies_stationarity());
    }

    struct FailingObjective;

    impl LatentCoordinateObjective for FailingObjective {
        type Error = TestObjectiveError;

        fn value_and_gradient(
            &mut self,
            _coordinates: ArrayView1<'_, f64>,
        ) -> Result<LatentCoordinateEvaluation, Self::Error> {
            Err(TestObjectiveError::InnerSolve)
        }
    }

    #[test]
    fn fatal_objective_failure_is_preserved_exactly() {
        let request = euclidean_request(array![1.0], options(4, 1));
        let error = optimize_latent_coordinates(request, &mut FailingObjective).unwrap_err();
        match error {
            LatentCoordinateOptimizationError::Objective {
                restart_index,
                source,
            } => {
                assert_eq!(restart_index, 0);
                assert_eq!(source, TestObjectiveError::InnerSolve);
            }
            other => panic!("expected objective error, got {other:?}"),
        }
    }

    struct NonFiniteObjective;

    impl LatentCoordinateObjective for NonFiniteObjective {
        type Error = TestObjectiveError;

        fn value_and_gradient(
            &mut self,
            coordinates: ArrayView1<'_, f64>,
        ) -> Result<LatentCoordinateEvaluation, Self::Error> {
            Ok(LatentCoordinateEvaluation {
                objective_value: f64::INFINITY,
                euclidean_gradient: Array1::zeros(coordinates.len()),
            })
        }
    }

    #[test]
    fn non_finite_objective_is_not_false_stationarity() {
        let request = euclidean_request(array![1.0], options(4, 1));
        let error = optimize_latent_coordinates(request, &mut NonFiniteObjective).unwrap_err();
        assert!(matches!(
            error,
            LatentCoordinateOptimizationError::InvalidObjectiveOutput {
                source: LatentCoordinateObjectiveContractError::NonFiniteValue { .. },
                ..
            }
        ));
    }

    struct RecordingStationaryObjective {
        points: Vec<Array1<f64>>,
    }

    impl LatentCoordinateObjective for RecordingStationaryObjective {
        type Error = TestObjectiveError;

        fn value_and_gradient(
            &mut self,
            coordinates: ArrayView1<'_, f64>,
        ) -> Result<LatentCoordinateEvaluation, Self::Error> {
            self.points.push(coordinates.to_owned());
            Ok(LatentCoordinateEvaluation {
                objective_value: 0.0,
                euclidean_gradient: Array1::zeros(coordinates.len()),
            })
        }
    }

    #[test]
    fn sphere_restarts_remain_on_the_product_manifold() {
        let request = LatentCoordinateOptimizationRequest {
            n_observations: 2,
            latent_dimension: 3,
            manifold: LatentCoordinateManifold::Sphere,
            start: LatentCoordinateStart::Initial(array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            options: options(0, 3),
        };
        let mut objective = RecordingStationaryObjective { points: Vec::new() };
        let result = optimize_latent_coordinates(request, &mut objective).unwrap();
        assert!(result.evidence.certifies_stationarity());
        assert!(!objective.points.is_empty());
        for point in &objective.points {
            for row in point.as_slice().unwrap().chunks_exact(3) {
                let norm = row.iter().fold(0.0_f64, |acc, value| acc.hypot(*value));
                assert!((norm - 1.0).abs() <= 1.0e-10);
            }
        }
    }
}
