//! #2817: a refusal raised inside a permuted outer search names coordinates in
//! the caller's native order.
//!
//! Keys out of canonical order make [`run_outer`] run the whole search with the
//! coordinates permuted. The checkpoint was mapped back (#2902), but every
//! coordinate the reason text named by index (the railed facts) was written
//! inside the canonical run and named a canonical slot.
//!
//! The railed facts are pinned on the certificate rendering itself. A railed
//! refusal cannot be held end to end: a railed coordinate with a descending
//! interior mints the #2392 active-set reseed on every refusal, and the
//! certify-last loop admits reseeds while the certified value strictly falls, so
//! it polishes the interior until it certifies.

use super::*;
use ndarray::array;

/// Structural keys out of order: `perm = [1, 3, 2, 0]`, so canonical slot `c`
/// holds native coordinate `perm[c]`.
const KEYS: [u64; 4] = [40, 10, 30, 20];
const PERM: [usize; 4] = [1, 3, 2, 0];
const BOX: f64 = gam_problem::LOG_STRENGTH_MAX;

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
    assert_eq!(
        canonical_permutation(&KEYS),
        Some(PERM.to_vec()),
        "the keys must put native 0 at canonical slot 3"
    );
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
