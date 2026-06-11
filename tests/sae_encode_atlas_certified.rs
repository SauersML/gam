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

use gam::terms::sae_candidate_index::{IndexConfig, RandomProjectionFrameSketch, SaeCandidateIndex};
use gam::terms::sae_encode_atlas::{
    row_certificate, AtlasConfig, EncodeAtlas, KANTOROVICH_THRESHOLD,
};
use gam::terms::sae_manifold::{
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
    SaeManifoldAtom::new(
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
fn certified_start_near_root_converges_to_planted_coordinate() {
    let atom = planted_circle_atom(8);
    // Amplitude bound 1 (the circle is unit-radius); target norm bound 1.
    let atlas = EncodeAtlas::build(
        std::slice::from_ref(&atom),
        &[1.0],
        1.0,
        AtlasConfig {
            grid_resolution: 32,
            ridge: 1.0e-9,
            newton_steps: 2,
        },
    )
    .expect("atlas build");

    // True coordinate t* = 0.2; its target is on the circle.
    let t_star = 0.2_f64;
    let x = circle_target(t_star);

    let (t_hat, cert) = atlas
        .certified_encode_row(&atom, 0, x.view(), 1.0)
        .expect("encode row");

    assert!(
        cert.certified(),
        "a start routed near the true root must certify; got h = {} (beta={}, eta={}, L={})",
        cert.h,
        cert.beta,
        cert.eta,
        cert.lipschitz
    );
    assert!(
        cert.h <= KANTOROVICH_THRESHOLD,
        "certificate h must be <= 1/2; got {}",
        cert.h
    );

    // The recovered coordinate reconstructs the target: m(t_hat) ~= x. Compare
    // reconstructions (the coordinate itself is only unique modulo the period).
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let recon = {
        let c = Array2::from_shape_vec((1, 1), vec![t_hat[0]]).unwrap();
        let (phi, _j) = evaluator.evaluate(c.view()).unwrap();
        Array1::from(vec![phi[[0, 2]], phi[[0, 1]]]) // (cos, sin)
    };
    let err = (&recon - &x).dot(&(&recon - &x)).sqrt();
    assert!(
        err < 1e-3,
        "certified encode must reconstruct the target: ‖m(t_hat) − x‖ = {err}"
    );
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
        1.0e-12,
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
            1.0e-12,
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
fn batched_certified_encode_reports_honest_uncertified_count() {
    // Throughput-shaped batch (the #988 consumer): many rows, the certified
    // ones recover the planted coordinate, and the uncertified count is the
    // exact number flagged — never silently approximated.
    let atom = planted_circle_atom(16);
    let atlas = EncodeAtlas::build(
        std::slice::from_ref(&atom),
        &[1.0],
        1.0,
        AtlasConfig {
            grid_resolution: 48,
            ridge: 1.0e-9,
            newton_steps: 2,
        },
    )
    .expect("atlas build");

    // 256 rows planted around the circle, all on-manifold => all routable to a
    // near-root chart and certifiable.
    let n = 256usize;
    let mut targets = Array2::<f64>::zeros((n, P));
    for row in 0..n {
        let t = row as f64 / n as f64;
        let x = circle_target(t);
        targets.row_mut(row).assign(&x);
    }
    let amplitudes = Array1::<f64>::ones(n);

    let result = atlas
        .certified_encode_batch(&atom, 0, targets.view(), amplitudes.view())
        .expect("batch encode");

    // Honesty invariant: the reported count equals the actual flagged rows.
    let actual_flagged = result.certified.iter().filter(|c| !**c).count();
    assert_eq!(
        result.encode_uncertified_count, actual_flagged,
        "encode_uncertified_count must equal the number of flagged rows"
    );

    // On-manifold rows with a fine grid certify in the large majority: assert a
    // strong certified fraction (the throughput target is exact-or-flagged, and
    // here the data is exactly on the planted circle).
    let certified_count = n - result.encode_uncertified_count;
    assert!(
        certified_count * 100 >= n * 80,
        "at least 80% of on-manifold rows must certify; got {certified_count}/{n}"
    );

    // Every CERTIFIED row must reconstruct its target (exactness of the fast
    // path): check the worst certified-row reconstruction error.
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let mut worst = 0.0_f64;
    for row in 0..n {
        if !result.certified[row] {
            continue;
        }
        let t_hat = result.coords[[row, 0]];
        let c = Array2::from_shape_vec((1, 1), vec![t_hat]).unwrap();
        let (phi, _j) = evaluator.evaluate(c.view()).unwrap();
        let recon = Array1::from(vec![phi[[0, 2]], phi[[0, 1]]]);
        let err = (&recon - &targets.row(row).to_owned())
            .dot(&(&recon - &targets.row(row).to_owned()))
            .sqrt();
        worst = worst.max(err);
    }
    assert!(
        worst < 5e-2,
        "certified rows must reconstruct their targets; worst error = {worst}"
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

    let h_small = row_certificate(
        &atom,
        evaluator.as_ref(),
        start.view(),
        x.view(),
        1.0,
        10.0,
        1.0e-12,
    )
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
        1.0e-12,
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

#[test]
fn lsh_routed_encode_matches_direct_atom_encode() {
    // The production routing path: the SaeCandidateIndex (#985/#994) selects the
    // atom per row, then the atlas encodes against that atom's certified charts.
    // With a single-atom dictionary every row routes to atom 0, so the routed
    // result must equal the direct per-atom batch encode — verifying the
    // composition (LSH atom-selection ∘ atlas chart-routing ∘ certificate).
    let atom = planted_circle_atom(8);
    let atlas = EncodeAtlas::build(
        std::slice::from_ref(&atom),
        &[1.0],
        1.0,
        AtlasConfig {
            grid_resolution: 32,
            ridge: 1.0e-9,
            newton_steps: 2,
        },
    )
    .expect("atlas build");

    // Build the LSH index over the single atom's column-space frame. The sketch
    // decoder block is p×r = 2×2 (the planted circle spans R²); routing returns
    // atom 0 for every direction.
    let frame = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let sketch = RandomProjectionFrameSketch::from_decoder_blocks(&[frame], 8, 7)
        .expect("sketch build");
    let index = SaeCandidateIndex::build(&sketch, IndexConfig::auto(8, 1, 7)).expect("index build");

    let n = 64usize;
    let mut targets = Array2::<f64>::zeros((n, P));
    for row in 0..n {
        let t = row as f64 / n as f64;
        targets.row_mut(row).assign(&circle_target(t));
    }
    let amplitudes = Array1::<f64>::ones(n);

    let direct = atlas
        .certified_encode_batch(&atom, 0, targets.view(), amplitudes.view())
        .expect("direct batch");
    let routed = atlas
        .certified_encode_with_index(
            std::slice::from_ref(&atom),
            &index,
            &sketch,
            targets.view(),
            amplitudes.view(),
            1,
        )
        .expect("routed batch");

    assert_eq!(
        routed.certified, direct.certified,
        "single-atom LSH routing must match the direct per-atom encode certificates"
    );
    assert_eq!(
        routed.encode_uncertified_count, direct.encode_uncertified_count,
        "routed uncertified count must match the direct path"
    );
    let coord_diff = (&routed.coords - &direct.coords)
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(
        coord_diff < 1e-12,
        "routed coordinates must equal the direct encode; max abs diff = {coord_diff}"
    );
}
