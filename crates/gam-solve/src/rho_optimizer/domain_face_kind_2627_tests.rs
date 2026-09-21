//! #2627: box-KKT projection certifies an optimum only at a face that is a
//! constraint of the model.
//!
//! Every fixture is the same one-coordinate quadratic `½(θ − target)²` judged at a
//! face `F` of its domain, with `target` past the face, so the criterion pulls the
//! coordinate outward with gradient `F − target`. Projection removes that pull at
//! any face. What the face IS decides whether that removal is stationarity:
//! - a [`DomainFaceKind::Constraint`] face certifies;
//! - a [`DomainFaceKind::Resolution`] face certifies only while the pull is within
//!   the certificate's own stationarity band;
//! - a [`DomainFaceKind::Representability`] face, the log-strength fallback, never
//!   certifies a pull beyond that band.
//!
//! Each refusal is typed: [`EstimationError::OuterDomainFaceRefused`] names the
//! coordinate, the face, its kind and the outward gradient.

use super::*;
use ndarray::array;

/// The face every declared-face fixture is judged at.
const FACE: f64 = 2.5;

/// `½(θ − target)²` with the upper face its producer declared.
struct FacedQuadratic {
    target: f64,
    upper: Option<DomainFaces>,
}

impl FacedQuadratic {
    fn declaring(target: f64, kind: DomainFaceKind) -> Self {
        Self {
            target,
            upper: Some(DomainFaces::uniform(array![FACE], kind)),
        }
    }
}

impl OuterObjective for FacedQuadratic {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Dense,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: false,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let delta = rho[0] - self.target;
        Ok(0.5 * delta * delta)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let delta = rho[0] - self.target;
        Ok(OuterEval {
            cost: 0.5 * delta * delta,
            gradient: array![delta],
            hessian: HessianValue::Dense(array![[1.0]]),
            inner_beta_hint: None,
        })
    }

    fn reset(&mut self) {}

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "face-kind test objective was offered a non-finite inner seed of length {}",
                beta.len()
            )));
        }
        Ok(SeedOutcome::NoSlot)
    }

    fn outer_domain_upper_bound(&self) -> Result<Option<DomainFaces>, EstimationError> {
        Ok(self.upper.clone())
    }
}

fn audit(
    obj: &mut FacedQuadratic,
    config: OuterConfig,
    at: f64,
    context: &str,
) -> Result<OuterResult, OuterStationaryPointRejection> {
    audit_stationary_point_in(obj, config, array![at], context)
}

fn plain_config() -> OuterConfig {
    OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .config()
}

/// The refused faces and band of a typed #2627 refusal, or the refusal that came
/// back instead, rendered.
fn refused_faces(
    rejection: OuterStationaryPointRejection,
) -> Result<(Vec<RefusedDomainFace>, f64), String> {
    match rejection.source {
        EstimationError::OuterDomainFaceRefused { faces, bound, .. } => Ok((faces, bound)),
        other => Err(format!("expected a typed OuterDomainFaceRefused refusal, got: {other}")),
    }
}

#[test]
fn an_outward_pull_at_a_constraint_face_certifies_2627() {
    let mut obj = FacedQuadratic::declaring(FACE + 1.0, DomainFaceKind::Constraint);
    let result = audit(&mut obj, plain_config(), FACE, "constraint face #2627")
        .unwrap_or_else(|rejection| {
            panic!(
                "a box-KKT point on a constraint of the model must certify; refused with: {}",
                rejection.source
            )
        });
    let certificate = result
        .criterion_certificate
        .as_ref()
        .expect("a certified point records its certificate");
    assert!(certificate.certifies(), "{}", certificate.summary());
}

#[test]
fn a_caller_box_is_a_constraint_of_the_callers_model_2627() {
    // `with_bounds` declares its faces Constraint, so a caller box still
    // certifies a box-KKT point on its face.
    let mut obj = FacedQuadratic {
        target: FACE + 1.0,
        upper: None,
    };
    let config = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_bounds(array![-FACE], array![FACE])
        .config();
    let result = audit(&mut obj, config, FACE, "caller box #2627").unwrap_or_else(|rejection| {
        panic!(
            "a box-KKT point on the caller's declared box must certify; refused with: {}",
            rejection.source
        )
    });
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|certificate| certificate.certifies())
    );
}

#[test]
fn a_resolution_face_certifies_a_pull_within_the_band_2627() {
    // A resolution face priced correctly leaves at most a pull inside the
    // gradient's resolution. 1e-12 is far inside any band the certificate forms.
    let mut obj = FacedQuadratic::declaring(FACE + 1.0e-12, DomainFaceKind::Resolution);
    let result = audit(&mut obj, plain_config(), FACE, "in-band resolution face #2627")
        .unwrap_or_else(|rejection| {
            panic!(
                "an in-band pull at a resolution face must certify; refused with: {}",
                rejection.source
            )
        });
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|certificate| certificate.certifies())
    );
}

#[test]
fn a_resolution_face_refuses_a_pull_beyond_the_band_typed_2627() {
    let mut obj = FacedQuadratic::declaring(FACE + 1.0, DomainFaceKind::Resolution);
    let rejection = audit(&mut obj, plain_config(), FACE, "out-of-band resolution face #2627")
        .expect_err("a pull beyond the band at a resolution face says the face was mispriced");
    let (faces, bound) = refused_faces(rejection).unwrap_or_else(|other| panic!("{other}"));
    assert_eq!(faces.len(), 1, "{faces:?}");
    let face = &faces[0];
    assert_eq!(face.coordinate, 0);
    assert_eq!(face.kind, DomainFaceKind::Resolution);
    assert_eq!(face.side, DomainFaceSide::Upper);
    assert_eq!(face.face, FACE);
    assert_eq!(face.outward_gradient, -1.0);
    assert!(
        face.outward_gradient.abs() > bound,
        "the refused pull {} must exceed the band {bound}",
        face.outward_gradient
    );
}

#[test]
fn the_log_strength_fallback_face_refuses_typed_2627() {
    // No caller box and no declared face: the domain is the log-strength
    // fallback, whose faces are the edge of what exp(rho) represents.
    let face_value = gam_problem::LOG_STRENGTH_MAX;
    let mut obj = FacedQuadratic {
        target: face_value + 1.0,
        upper: None,
    };
    let rejection = audit(&mut obj, plain_config(), face_value, "representability face #2627")
        .expect_err("a pull past the representable domain is not an optimum of the model");
    let (faces, _) = refused_faces(rejection).unwrap_or_else(|other| panic!("{other}"));
    assert_eq!(faces.len(), 1, "{faces:?}");
    assert_eq!(faces[0].kind, DomainFaceKind::Representability);
    assert_eq!(faces[0].side, DomainFaceSide::Upper);
    assert_eq!(faces[0].face, face_value);
}

#[test]
fn installing_declared_faces_keeps_the_kind_of_the_face_that_wins_2627() {
    // The configured fallback is Representability; a tighter declared face
    // replaces it with its own kind, and a looser one leaves the fallback in
    // place.
    let mut config = plain_config();
    let mut upper = DomainFaces::uniform(
        array![FACE, gam_problem::LOG_STRENGTH_MAX + 1.0],
        DomainFaceKind::Resolution,
    );
    upper.replace(1, gam_problem::LOG_STRENGTH_MAX + 1.0, DomainFaceKind::Constraint);
    install_objective_domain(&mut config, 2, None, Some(upper)).expect("a consistent domain");
    let (lower, upper) = outer_model_domain_faces(&config, 2).expect("installed kinds");
    assert_eq!(upper.values(), &array![FACE, gam_problem::LOG_STRENGTH_MAX]);
    assert_eq!(
        upper.kinds(),
        &[DomainFaceKind::Resolution, DomainFaceKind::Representability]
    );
    assert_eq!(
        lower.kinds(),
        &[DomainFaceKind::Representability, DomainFaceKind::Representability]
    );
}
