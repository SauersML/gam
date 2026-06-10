//! Object 4 — the Certificate (`residual_gauge`) end-to-end tests.
//!
//! These tests assert the certificate names EXACTLY the gauge group a fitted
//! SAE-manifold model is identified up to. Replicate fits (planted, fixed seeds)
//! must agree up to exactly that named group: both
//!
//!   * under-claiming — reporting a residual freedom the data/isometry pin
//!     actually removes, and
//!   * over-claiming — omitting a real residual freedom
//!
//! must FAIL the test. Two fixtures are constructed by hand so the truth is
//! analytic: one with a genuine residual rotation freedom (isometry pin
//! inactive) and one where the isometry pin removes that same rotation.

use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::sae_identifiability::{
    AtomTopology, FittedAtom, FittedSaeManifold, GeneratorFamily, residual_gauge,
};
use ndarray::Array2;
use std::sync::Arc;

/// A single 2-latent-dim Euclidean-patch atom with output_dim = 1 and decoder
/// frame `[[1, 0]]` (one output channel, two latent axes). Its only `Isom(M_k)`
/// generator is the `so(2)` rotation of the two frame columns: as a tangent
/// direction on the flattened frame `[f00, f01]` it is
/// `ξ_rot = [-f01, f00] = [0, 1]`. Keeping output_dim = 1 makes the frame row
/// dimension match the metric's `p_out`, so the certificate's Jacobian/metric
/// view and its frame generators live in one coherent param space (param_dim = 2)
/// with no spurious global output-frame rotation (needs ≥ 2 output axes).
fn identity_patch_atom() -> FittedAtom {
    let mut frame = Array2::<f64>::zeros((1, 2));
    frame[[0, 0]] = 1.0;
    FittedAtom {
        name: "patch".to_string(),
        topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
        frame,
        // No ARD prior here — the rotation's pinned/unpinned status is decided
        // purely by the data + isometry penalty, which is what we want to test.
        ard_variances: None,
    }
}

/// Euclidean metric over `n` rows, output dim `p`. Whitening is the identity, so
/// the data span is the bare Jacobian span.
fn euclidean_metric(n: usize, p: usize) -> RowMetric {
    RowMetric::euclidean(n, p).expect("euclidean metric")
}

/// Build the data Jacobian rows so the data pins the frame direction orthogonal
/// to the `so(2)` rotation but is invariant to the rotation `ξ_rot = [0, 1]`.
///
/// We give the data curvature along `[1, 0]` (the f00 axis), which has no
/// component along `ξ_rot = [0, 1]`, so the data leaves the rotation free while
/// pinning the orthogonal frame coordinate.
fn data_jacobian_rows_invariant_to_rotation() -> Vec<Vec<f64>> {
    // p = 1 output channel, param_dim = 2. Each "row" is a 1×2 Jacobian
    // (p * param_dim = 2 entries).
    vec![vec![1.0, 0.0]]
}

/// Fixture A: genuine residual rotation freedom. One identity-frame patch atom,
/// data invariant to the `so(2)` rotation, and the isometry pin INACTIVE
/// (`isometry_penalty_root` has zero rows). The rotation must come back as an
/// unpinned residual gauge freedom, and the report must escalate to
/// diffeomorphism-unpinned.
fn fixture_a_free_rotation() -> FittedSaeManifold {
    let atom = identity_patch_atom();
    let param_dim = atom.frame.len(); // 4
    let p = 1usize;
    FittedSaeManifold {
        atoms: vec![atom],
        jacobian_rows: data_jacobian_rows_invariant_to_rotation(),
        isometry_penalty_root: Array2::<f64>::zeros((0, param_dim)),
        metric: euclidean_metric(3, p),
    }
}

/// Fixture B: same atom and data, but the isometry pin is ACTIVE and gives the
/// rotation direction curvature. The single isometry-penalty root row IS the
/// rotation tangent `ξ_rot = [0,1,-1,0]`, so `range(H_isometry)` contains it and
/// the rotation is pinned. The report must NOT escalate to
/// diffeomorphism-unpinned and must report zero residual gauge generators.
fn fixture_b_pinned_rotation() -> FittedSaeManifold {
    let atom = identity_patch_atom();
    let param_dim = atom.frame.len(); // 4
    let p = 1usize;
    let mut root = Array2::<f64>::zeros((1, param_dim));
    root[[0, 1]] = 1.0; // ξ_rot = [0, 1]
    FittedSaeManifold {
        atoms: vec![atom],
        jacobian_rows: data_jacobian_rows_invariant_to_rotation(),
        isometry_penalty_root: root,
        metric: euclidean_metric(3, p),
    }
}

#[test]
fn fixture_a_certifies_a_genuine_residual_rotation_freedom() {
    let report = residual_gauge(&fixture_a_free_rotation()).expect("certificate");

    // Metric provenance is read straight off the RowMetric — Euclidean here.
    assert_eq!(report.metric_provenance, MetricProvenance::Euclidean);

    // The isometry pin is inactive → diffeomorphism-unpinned escalation.
    assert!(
        report.diffeomorphism_unpinned,
        "isometry pin inactive must escalate to diffeomorphism-unpinned"
    );

    // The single Isom(M_k) so(2) rotation generator must be UNPINNED.
    let rot = report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("an Isom(M_k) generator must be enumerated");
    assert!(
        rot.unpinned,
        "the so(2) rotation must be a residual freedom when the data is \
         rotation-invariant and the isometry pin is inactive; got pinned. {}",
        report.summary
    );
    assert!(
        rot.generator_norm > 0.0,
        "rotation generator must be non-trivial"
    );

    // Exactly one residual freedom (the rotation): OVER-claiming guard — if the
    // certificate invented extra freedoms this count would exceed 1.
    assert_eq!(
        report.residual_gauge_dim, 1,
        "exactly one residual gauge freedom expected; over/under-claim. {}",
        report.summary
    );
}

#[test]
fn fixture_b_isometry_pin_removes_the_rotation_freedom() {
    let report = residual_gauge(&fixture_b_pinned_rotation()).expect("certificate");

    assert_eq!(report.metric_provenance, MetricProvenance::Euclidean);

    // Isometry pin active → no diffeomorphism escalation.
    assert!(
        !report.diffeomorphism_unpinned,
        "an active isometry pin must NOT escalate to diffeomorphism-unpinned"
    );

    // The so(2) rotation must now be PINNED.
    let rot = report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("an Isom(M_k) generator must be enumerated");
    assert!(
        !rot.unpinned,
        "the isometry penalty must pin the so(2) rotation (UNDER-claiming guard \
         fails if it is still reported free). {}",
        report.summary
    );

    // No residual gauge freedoms survive.
    assert_eq!(
        report.residual_gauge_dim, 0,
        "isometry-pinned model must have zero residual gauge freedoms. {}",
        report.summary
    );
}

#[test]
fn replicate_fits_agree_up_to_exactly_the_named_group() {
    // Two independent constructions of the SAME planted model (fixed by hand,
    // deterministic) must produce the SAME certified group signature. Replicate
    // agreement is signature equality.
    let rep1 = residual_gauge(&fixture_a_free_rotation()).expect("certificate 1");
    let rep2 = residual_gauge(&fixture_a_free_rotation()).expect("certificate 2");
    assert_eq!(
        rep1.group_signature(),
        rep2.group_signature(),
        "replicate fits of the same planted model must be identified up to the \
         exact same group"
    );

    // And the two DIFFERENT fixtures (free vs pinned) must NOT share a
    // signature: the certificate distinguishes the larger gauge group from the
    // smaller one. If under-/over-claiming collapsed them this would fail.
    let rep_pinned = residual_gauge(&fixture_b_pinned_rotation()).expect("certificate pinned");
    assert_ne!(
        rep1.group_signature(),
        rep_pinned.group_signature(),
        "the free-rotation fixture and the isometry-pinned fixture must NOT be \
         certified up to the same group"
    );
}

#[test]
fn under_claim_is_caught_data_pins_rotation_directly() {
    // A variant of fixture A where the DATA itself pins the rotation (one
    // Jacobian row IS the rotation tangent). Even with the isometry pin inactive
    // the rotation is no longer free: the certificate must report it pinned. A
    // certificate that claimed the rotation free (under-claiming the pinning)
    // would fail here.
    let atom = identity_patch_atom();
    let param_dim = atom.frame.len();
    let mut rows = data_jacobian_rows_invariant_to_rotation();
    rows.push(vec![0.0, 1.0]); // data row along ξ_rot = [0,1] pins it
    let model = FittedSaeManifold {
        atoms: vec![atom],
        jacobian_rows: rows,
        isometry_penalty_root: Array2::<f64>::zeros((0, param_dim)),
        metric: euclidean_metric(4, 1),
    };
    let report = residual_gauge(&model).expect("certificate");
    let rot = report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("Isom generator");
    assert!(
        !rot.unpinned,
        "data that spans the rotation tangent must pin it even with no isometry \
         penalty. {}",
        report.summary
    );
    assert_eq!(report.residual_gauge_dim, 0);
}

#[test]
fn sym_f_triviality_checked_only_under_output_fisher() {
    // Build a two-atom, topology-identical model under an OutputFisher metric.
    // The Sym(F) atom-permutation generator must be PINNED (output-Fisher
    // separates the atoms behaviorally), so sym_f_trivial_under_output_fisher
    // must be Some(true). Under Euclidean the check does not apply (None).

    // Two topology-identical patch atoms with DISTINCT frames, so the
    // atom-exchange swap direction is non-trivial (a genuine Sym(F) generator,
    // not a structurally-zero one); param_dim = 8.
    let make = |name: &str, frame: Array2<f64>| FittedAtom {
        name: name.to_string(),
        topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
        frame,
        ard_variances: None,
    };
    let frame_a = Array2::<f64>::eye(2);
    let mut frame_b = Array2::<f64>::zeros((2, 2));
    frame_b[[0, 0]] = 2.0;
    frame_b[[1, 1]] = 3.0;
    let param_dim = 8usize;
    let p = 2usize; // output dim
    let rank = 1usize;

    // Build a Jacobian + isometry root that pin every direction including the
    // atom-exchange swap, so no spurious permutation freedom appears.
    // Isometry root = identity on all 8 params: pins everything.
    let isom_root = Array2::<f64>::eye(param_dim);

    // One Jacobian row of full p*param_dim length (zeros is fine; the isometry
    // root alone pins the parameter space — the data only adds to the span).
    let jac_rows = vec![vec![0.0_f64; p * param_dim]];

    // OutputFisher metric: supply per-row factors U_n ∈ ℝ^{p × rank}; here a
    // single row, factors = all-ones (a valid PSD low-rank factor: W_n = U_n U_nᵀ
    // is rank-1 PSD).
    let u = Arc::new(Array2::<f64>::ones((1, p * rank)));
    let metric_of =
        RowMetric::output_fisher(Arc::clone(&u), p, rank).expect("output-fisher metric");

    let model = FittedSaeManifold {
        atoms: vec![make("a", frame_a.clone()), make("b", frame_b.clone())],
        jacobian_rows: jac_rows,
        isometry_penalty_root: isom_root,
        metric: metric_of,
    };
    let report = residual_gauge(&model).expect("certificate");
    // The atom-exchange generator must be non-trivial (distinct frames).
    let perm = report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::AtomPermutation)
        .expect("an atom-permutation generator must be enumerated");
    assert!(
        perm.generator_norm > 0.0,
        "distinct frames must give a non-trivial swap generator"
    );
    assert!(
        matches!(
            report.metric_provenance,
            MetricProvenance::OutputFisher { .. }
        ),
        "metric provenance must be OutputFisher, got {:?}",
        report.metric_provenance
    );
    assert_eq!(
        report.sym_f_trivial_under_output_fisher,
        Some(true),
        "Sym(F) must be trivially pinned under OutputFisher; a surviving \
         permutation is a certificate violation. {}",
        report.summary
    );

    // Same structure under Euclidean: the check does not apply.
    let model_euc = FittedSaeManifold {
        atoms: vec![make("a", frame_a), make("b", frame_b)],
        jacobian_rows: vec![vec![0.0_f64; param_dim]],
        isometry_penalty_root: Array2::<f64>::eye(param_dim),
        metric: euclidean_metric(1, 1),
    };
    let report_euc = residual_gauge(&model_euc).expect("certificate euc");
    assert_eq!(report_euc.sym_f_trivial_under_output_fisher, None);
}
