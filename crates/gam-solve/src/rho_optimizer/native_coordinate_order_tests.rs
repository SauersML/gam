//! #2817: a refusal raised inside a permuted outer search names coordinates in
//! the caller's native order.
//!
//! Keys out of canonical order make [`run_outer`] run the whole search with the
//! coordinates permuted. The checkpoint was mapped back (#2902), but every
//! coordinate the reason text named by index (the railed facts, the tail-snap
//! declines) was written inside the canonical run and named a canonical slot.
//!
//! Two pins, because one refusal cannot carry both halves by construction.
//! - The tail-snap declines are pinned end to end. Native 0 is flat (zero
//!   gradient, zero curvature) and natives 1..=3 are an off-lattice quartic far
//!   from its optimum under a one-iteration budget. Native 0 never moves off its
//!   seed value, so nothing rails and no active-set or wrong-rail reseed is minted.
//!   The refusal is not first-order stationary, so no saddle reseed is minted, and
//!   the tail snap declines, so it publishes no reseed either: the first refusal
//!   stands, whichever seed the cascade reports. The tail snap declines only a
//!   coordinate whose gradient exceeds the band, so it never declines native 0: a
//!   native rendering cannot print `k=0`, while a canonical one prints `k=0` for
//!   native 1 (canonical slot 0).
//! - The railed facts are pinned on the certificate rendering itself. A railed
//!   refusal cannot be held end to end: a railed coordinate with a descending
//!   interior mints the #2392 active-set reseed on every refusal, and the
//!   certify-last loop admits reseeds while the certified value strictly falls, so
//!   it polishes the interior until it certifies.

use super::*;
use ndarray::array;

/// Structural keys out of order: `perm = [1, 3, 2, 0]`, so canonical slot `c`
/// holds native coordinate `perm[c]`.
const KEYS: [u64; 4] = [40, 10, 30, 20];
const PERM: [usize; 4] = [1, 3, 2, 0];
const BOX: f64 = gam_problem::LOG_STRENGTH_MAX;
/// Native coordinate 1, whose decline a canonical rendering prints as `k=0`.
const DECLINED: usize = 1;
/// Off-lattice optima and unequal weights for natives 1..=3, as in the #2902 pin,
/// so no seed certifies and the residual gradient stays far above the band after
/// one iteration.
const OFFSET: [f64; 3] = [0.5, -0.25, 0.125];
const WEIGHT: [f64; 3] = [1.0e6, 3.0e6, 2.0e6];

fn criterion(theta: &Array1<f64>) -> f64 {
    WEIGHT[0] * (theta[1] - OFFSET[0]).powi(4)
        + WEIGHT[1] * (theta[2] - OFFSET[1]).powi(4)
        + WEIGHT[2] * (theta[3] - OFFSET[2]).powi(4)
}

fn criterion_eval(theta: &Array1<f64>) -> OuterEval {
    let d = [theta[1] - OFFSET[0], theta[2] - OFFSET[1], theta[3] - OFFSET[2]];
    OuterEval {
        cost: criterion(theta),
        // Native 0 is flat: no gradient and no curvature.
        gradient: array![
            0.0,
            4.0 * WEIGHT[0] * d[0].powi(3),
            4.0 * WEIGHT[1] * d[1].powi(3),
            4.0 * WEIGHT[2] * d[2].powi(3)
        ],
        hessian: HessianValue::Dense(array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 12.0 * WEIGHT[0] * d[0].powi(2), 0.0, 0.0],
            [0.0, 0.0, 12.0 * WEIGHT[1] * d[1].powi(2), 0.0],
            [0.0, 0.0, 0.0, 12.0 * WEIGHT[2] * d[2].powi(2)],
        ]),
        inner_beta_hint: None,
    }
}

fn permuted_run() -> Result<OuterResult, EstimationError> {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let problem = OuterProblem::new(4)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_bounds(Array1::from_elem(4, -BOX), Array1::from_elem(4, BOX))
        .with_initial_rho(array![0.0, 5.0, -3.0, 4.0])
        .with_max_iter(1)
        .with_rho_canonical_keys(Some(KEYS.to_vec()));
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(criterion(theta)),
        |_: &mut (), theta: &Array1<f64>| Ok(criterion_eval(theta)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem.run(&mut obj, "native coordinate order 2817")
}

#[test]
fn a_permuted_refusal_names_declined_coordinates_natively_2817() {
    assert_eq!(
        canonical_permutation(&KEYS),
        Some(PERM.to_vec()),
        "the keys must put native 1 at canonical slot 0"
    );
    let (reason, checkpoint, bound) = match permuted_run() {
        Err(EstimationError::RemlDidNotConverge {
            reason,
            rho_checkpoint,
            stationarity_standard,
            ..
        }) => (reason, rho_checkpoint, stationarity_standard.bound()),
        other => (format!("not a typed REML non-convergence: {other:?}"), Vec::new(), None),
    };
    assert_eq!(
        checkpoint.len(),
        4,
        "the permuted search must refuse with a native checkpoint; got: {reason}"
    );
    // Precondition, printed before it is judged: the tail snap declines native 1 only
    // when its residual exceeds the band the refusal applied. That is measured at the
    // reported point, not derived, so it is shown and enforced here.
    let declined_gradient = criterion_eval(&Array1::from_vec(checkpoint.clone())).gradient[DECLINED];
    eprintln!(
        "[#2817 native order] precondition |g{DECLINED}|={:.6e} bound={bound:?} checkpoint={checkpoint:?}",
        declined_gradient.abs()
    );
    eprintln!("[#2817 native order] {reason}");
    assert!(
        bound.is_some_and(|bound| declined_gradient.abs() > bound),
        "precondition: native {DECLINED}'s residual {:.6e} must exceed the refusal's band \
         {bound:?} for the tail snap to decline it",
        declined_gradient.abs()
    );
    // Precondition: the first refusal stands. Each reseed kind has a precondition the
    // refusal renders: an active-set or wrong-rail reseed needs a railed coordinate, a
    // saddle escape needs a first-order stationary point, and a tail snap reseeds only
    // when it does not decline.
    assert!(
        reason.contains("railed=[]")
            && reason.contains("NOT STATIONARY")
            && reason.contains("tail-snap declined"),
        "the refusal must rail nothing, be non-stationary and decline the tail snap, so no \
         reseed is minted; got: {reason}"
    );
    let native_one_decline = format!("k={DECLINED}: ρ={:.2}", checkpoint[DECLINED]);
    assert!(
        reason.contains(&native_one_decline),
        "the tail-snap decline must name native coordinate {DECLINED} at its checkpoint value \
         ({native_one_decline}); got: {reason}"
    );
    assert!(
        !reason.contains("k=0: ρ="),
        "native 0 is flat and never declined, so only a canonical rendering can print k=0 \
         (canonical slot 0 holds native 1); got: {reason}"
    );
}

// Railed facts: the certificate's own rendering.

fn railed_on_canonical_slot_three() -> OuterCriterionCertificate {
    OuterCriterionCertificate {
        stationarity: OuterStationarityCertificate::AnalyticGradient {
            grad_norm: 1.0,
            projected_grad_norm: 1.0,
            bound: 1.0e-3,
            rung: CertifiedRung {
                label: "solver-band".to_string(),
                derived_standard: false,
            },
        },
        curvature: CurvatureEvidence::Measured { psd: true },
        lambdas_railed: vec![3],
        railed_facts: vec![RailedCoordinateFact {
            index: 3,
            theta: BOX,
            lower: -BOX,
            upper: BOX,
            margin: 0.5,
            face: crate::model_types::RailFaceKind::Representability,
        }],
        newton_polish: None,
        curvature_floor: None,
    }
}

#[test]
fn a_permuted_certificate_renders_railed_coordinates_natively_2817() {
    let certificate = railed_on_canonical_slot_three();
    // Control: the canonical rendering names the slot the search held the coordinate at.
    let canonical = certificate.summary();
    assert!(
        canonical.contains("railed=[3]") && canonical.contains("#3 theta="),
        "the canonical rendering must name canonical slot 3; got: {canonical}"
    );
    let config = OuterConfig {
        native_coordinate_order: Some(PERM.to_vec()),
        ..OuterConfig::default()
    };
    let native = native_certificate_summary(&certificate, &config);
    assert!(
        native.contains("railed=[0]") && native.contains("#0 theta="),
        "canonical slot 3 holds native 0, so the refusal must name native 0; got: {native}"
    );
    assert!(
        !native.contains("railed=[3]") && !native.contains("#3 theta="),
        "the native rendering must not name canonical slot 3; got: {native}"
    );
}

#[test]
fn a_permuted_rail_test_renders_its_native_coordinate_2817() {
    let config = OuterConfig {
        native_coordinate_order: Some(PERM.to_vec()),
        ..OuterConfig::default()
    };
    let theta = array![0.0, 0.0, 0.0, BOX];
    let rail = RailTest::evaluate(&theta, 3, &config);
    assert!(rail.is_railed(), "canonical slot 3 sits on its upper face");
    let rendered = rail.to_string();
    assert!(
        rendered.starts_with("#0 theta="),
        "the rail test at canonical slot 3 must render native 0; got: {rendered}"
    );
}
