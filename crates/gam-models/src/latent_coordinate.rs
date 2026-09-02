//! Typed, Python-independent optimization of latent coordinates.
//!
//! This module owns the outer manifold optimization contract: request
//! validation, deterministic restarts, trust-region execution, resumable
//! checkpoints, and the stationarity certificate required before a caller may
//! construct a fitted model. Concrete latent likelihoods implement
//! [`LatentCoordinateObjective`]; no FFI type participates in the contract.

use std::error::Error as StdError;
use std::fmt;

use gam_geometry::GeometryError;
use ndarray::{Array1, ArrayView1};
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

/// Failure constructing a typed resume checkpoint.
///
/// A checkpoint carries the same structural contract as a fresh request
/// ([`LatentCoordinateRequestError`]) *and* must land on the manifold it names,
/// so it additionally surfaces the geometric feasibility failure
/// ([`GeometryError`]) raised when the stored point cannot be retracted onto the
/// manifold (e.g. a non-unit sphere row).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LatentCoordinateCheckpointError {
    #[error(transparent)]
    Request(#[from] LatentCoordinateRequestError),
    #[error(transparent)]
    Geometry(#[from] GeometryError),
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

    pub fn checkpoint(&self) -> Option<&LatentCoordinateCheckpoint> {
        match self {
            Self::NonConverged { checkpoint, .. } => Some(checkpoint),
            _ => None,
        }
    }
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
            coordinates: ArrayView1<'_, f64>,
            tangent: ArrayView1<'_, f64>,
        ) -> Result<Option<Array1<f64>>, Self::Error> {
            assert_eq!(coordinates.len(), tangent.len());
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
        // Name the variant that arrived. "expected typed non-convergence" alone
        // cannot distinguish the two ways this fails — a run that certified when
        // it should not have, versus a non-convergence reported under a variant
        // carrying neither evidence nor a checkpoint — and those have opposite
        // fixes. Rendered before the destructuring move so the else arm can
        // report it.
        let reported = format!("{error:?}");
        let LatentCoordinateOptimizationError::NonConverged {
            evidence,
            checkpoint,
        } = error
        else {
            panic!("expected typed non-convergence, got {reported}");
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
            coordinates: ArrayView1<'_, f64>,
        ) -> Result<LatentCoordinateEvaluation, Self::Error> {
            assert!(coordinates.iter().all(|value| value.is_finite()));
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
