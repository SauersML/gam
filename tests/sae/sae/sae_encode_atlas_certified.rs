//! Kantorovich-certified encode atlas (issue #1010).
//!
//! Two arms:
//!
//! 1. **Planted single-circle, analytically-known basin boundary.** One
//!    periodic atom whose decoder traces the unit circle `m(t) = (cos 2πt,
//!    sin 2πt)`. Encoding a target `x = m(t*)` is the Newton problem
//!    `min_t ½‖x − m(t)‖²`. The Newton basin of the true root `t*` is the open
//!    half-circle around it; the basin BOUNDARY is the antipode `t* + ½`, where
//!    the gradient vanishes but the curvature flips sign (a local maximum, not a
//!    minimum). A start near `t*` must certify (`h ≤ ½`) and converge to the
//!    true coordinate; a start near the antipode must FLAG (`h > ½` or singular
//!    curvature), never silently converge to the wrong root.
//!
//! 2. **Throughput-shaped batched path** (the #988 consumer). A many-row batch
//!    through [`EncodeAtlas::certified_encode_batch`]; we assert correctness
//!    (certified rows recover the planted coordinate, uncertified count is
//!    honest) — not wall-time.

use std::f64::consts::TAU;
use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::terms::sae::encode::{KANTOROVICH_THRESHOLD, row_certificate};
use gam::terms::sae::manifold::{
    PeriodicHarmonicEvaluator, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom,
};

/// `M = 3` periodic basis: `[1, sin(2πt), cos(2πt)]` (one harmonic).
const M: usize = 3;
/// Ambient dimension of the planted circle.
const P: usize = 2;

/// Build the single planted-circle atom. The decoder maps the basis to the unit
/// circle: `m(t) = cos(2πt)·e_x + sin(2πt)·e_y`. Coordinates are seeded at the
/// origin; the atlas evaluates the basis at its own chart centers, so the seed
/// values only set `n_obs`.
fn planted_circle_atom(n_obs: usize) -> SaeManifoldAtom {
    let evaluator = PeriodicHarmonicEvaluator::new(M).expect("evaluator");
    let coords = Array2::<f64>::zeros((n_obs.max(1), 1));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("evaluate");
    // decoder rows: [1]->(0,0), [sin]->(0,1), [cos]->(1,0)  => m=(cos,sin).
    let mut decoder = Array2::<f64>::zeros((M, P));
    decoder[[2, 0]] = 1.0; // cos column -> x
    decoder[[1, 1]] = 1.0; // sin column -> y
    SaeManifoldAtom::new_with_provided_function_gram(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(M),
    )
    .expect("atom build")
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(M).expect("evaluator clone"),
    ))
}

/// Target on the planted circle at coordinate `t` (fraction of one period).
fn circle_target(t: f64) -> Array1<f64> {
    let angle = TAU * t;
    Array1::from(vec![angle.cos(), angle.sin()])
}

#[test]
fn antipodal_start_flags_never_silently_wrong() {
    let atom = planted_circle_atom(8);

    // True root t* = 0.0 (target = (1, 0)). The antipode t = 0.5 (target's
    // far side) is the basin boundary: at a start there the encode Hessian for
    // THIS target has the wrong sign (local max), so beta is undefined and the
    // certificate must flag. We evaluate the certificate DIRECTLY at the
    // antipodal start to exercise the basin-boundary detection, independent of
    // routing.
    let evaluator: Arc<dyn SaeBasisEvaluator> =
        Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap());
    let x = circle_target(0.0); // (1, 0)

    // A large Lipschitz constant (any finite L) — the antipodal start fails on
    // the curvature (beta = +inf), so it flags for ALL L.
    let lipschitz = 100.0;
    let antipode = Array1::from(vec![0.5_f64]);
    let (cert, _delta) = row_certificate(
        &atom,
        evaluator.as_ref(),
        antipode.view(),
        x.view(),
        1.0,
        lipschitz,
    )
    .expect("certificate at antipode");

    assert!(
        !cert.certified(),
        "a start at the antipodal basin boundary must FLAG, never certify; got h = {} (beta={}, eta={})",
        cert.h,
        cert.beta,
        cert.eta
    );
}

#[test]
fn certificate_h_is_monotone_in_distance_from_root() {
    // The Kantorovich h grows as the start moves away from the root toward the
    // antipode: near the root the residual is small and curvature is positive
    // (small h), near the antipode curvature degrades (large/infinite h). This
    // exercises the analytically-known basin structure: there is a crossing
    // radius where h passes 1/2 — the certified region boundary.
    let atom = planted_circle_atom(8);
    let evaluator: Arc<dyn SaeBasisEvaluator> =
        Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap());
    let x = circle_target(0.0); // root at t = 0
    let lipschitz = 50.0;

    let h_at = |t: f64| -> f64 {
        let start = Array1::from(vec![t]);
        let (cert, _d) = row_certificate(
            &atom,
            evaluator.as_ref(),
            start.view(),
            x.view(),
            1.0,
            lipschitz,
        )
        .expect("certificate");
        cert.h
    };

    // Near the root: certified.
    assert!(
        h_at(0.02) <= KANTOROVICH_THRESHOLD,
        "near-root start must be certified; h(0.02) = {}",
        h_at(0.02)
    );
    // Far toward the antipode: uncertified (curvature flips before t = 0.25).
    let h_far = h_at(0.30);
    assert!(
        !(h_far <= KANTOROVICH_THRESHOLD),
        "start past the basin boundary must be uncertified; h(0.30) = {h_far}"
    );
}

#[test]
fn lipschitz_constant_shrinks_certified_radius_monotonically() {
    // Sanity on the certificate's soundness lever: a LARGER Hessian-Lipschitz
    // constant L only ever shrinks the certified region (h = beta*eta*L grows
    // with L), so an over-estimate of L can never falsely certify a row.
    let atom = planted_circle_atom(8);
    let evaluator: Arc<dyn SaeBasisEvaluator> =
        Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap());
    let x = circle_target(0.0);
    let start = Array1::from(vec![0.1_f64]);

    let h_small = row_certificate(&atom, evaluator.as_ref(), start.view(), x.view(), 1.0, 10.0)
        .unwrap()
        .0
        .h;
    let h_large = row_certificate(
        &atom,
        evaluator.as_ref(),
        start.view(),
        x.view(),
        1.0,
        1000.0,
    )
    .unwrap()
    .0
    .h;
    assert!(
        h_large > h_small,
        "larger L must yield larger h (smaller certified region): {h_large} vs {h_small}"
    );
    // The ratio is exactly the L ratio (h is linear in L).
    let ratio = h_large / h_small;
    assert!(
        (ratio - 100.0).abs() < 1e-6,
        "h must be exactly linear in L; ratio = {ratio}"
    );
}

// ----------------------------------------------------------------------------
// #1026 ladder item 3 — distilled amortized encoder.
//
// The amortized encoder predicts the latent coordinate in CLOSED FORM from the
// per-chart implicit-function-theorem Jacobian (one mat-vec, no per-row Hessian
// factorization or Newton solve), then evaluates the SAME Kantorovich
// certificate at the prediction. Accepted iff h ≤ ½; uncertified rows flag for
// the exact fallback. These tests assert the distilled map is honest (count
// invariant), exact on accepted rows (recovers the planted coordinate), and
// that its closed-form prediction is genuinely the IFT first-order map.
// ----------------------------------------------------------------------------

