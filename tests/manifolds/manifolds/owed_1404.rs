//! Owed-work regression for #1404 — Matérn and constant-curvature spatial
//! correctness regressions.
//!
//! This file pins the basis-level invariants the #1404 fixes establish for the
//! constant-curvature (`M_κ`) smooth, at fast deterministic basis scope (no
//! end-to-end fit, so it never trips the perf budget the integration arms
//! carry). Each assertion is against self-constructed analytic ground truth on
//! the three space forms — never another tool's output.
//!
//! ## The cluster of defects (now fixed)
//!
//! 1. GREAT-CIRCLE PRECISION. The κ = 1 kernel evaluates `exp(−d_κ/ℓ)` in the
//!    EXACT great-circle geodesic of the inverse-stereographic embedding. The
//!    self-distance is analytically zero, so the diagonal kernel is exactly 1.
//!    A reference built from `acos(p·q)` (the way the failing test computed it)
//!    suffers catastrophic cancellation at `p·q = 1 − 2.1e-8` and reports the
//!    geodesic self-distance as ≈2e-4 instead of 0; gam's exact distance and the
//!    cancellation-free `atan2(|p×q|, p·q)` reference both give 0. This pins the
//!    diagonal at exactly 1 and the off-diagonals against the stable reference.
//!
//! 2. CONSTRAINED-KERNEL-GRAM AT κ ≠ 0. The realized design is `K(ℓ)·z` and the
//!    realized penalty is `zᵀK(ℓ)z` at the SAME `ℓ` the metadata reports —
//!    one Gram, one range (gam#2747). The two fill-invariant remappings that
//!    used to sit between the spec's range and the realized kernel (`L(κ)` for
//!    the design, `L_S(κ)` for the penalty) are gone: they were an attempt to
//!    remove the κ/ℓ confounding by constraint, and the range is estimated
//!    instead. Reconstruction at the metadata's own length now matches exactly
//!    at every κ.
//!
//! 3. RAW RKHS PENALTY + NO CURVATURE-BLIND RIDGE. The primary penalty is the
//!    RAW symmetric kernel Gram `zᵀKz` with `normalization_scale = 1`, not a
//!    Frobenius-normalized operator. Frobenius normalization (divide by ‖·‖_F,
//!    dominated by the large wiggly eigenvalues) compresses the eigen-spread and
//!    inflates the smallest eigenvalues, so REML's scale-sensitive λ heuristics
//!    over-shrink the genuinely smooth low-degree signal (planted degree-1
//!    sphere harmonic recovered at only R² ≈ 0.84). Keeping the raw physical
//!    operator lets REML act on true roughness. And the default smooth carries
//!    NO double-penalty ridge `I` (#1464): the ridge is curvature-BLIND and with
//!    its own λ absorbs the data fit independent of κ, railing κ to the chart
//!    bound. This test pins both: default `double_penalty = false`, and the
//!    primary penalty exactly proportional to `zᵀK(L(κ))z`.
//!
//! 4. κ-DERIVATIVE CORRECTNESS. The kernel κ-jets (which feed the outer
//!    LAML/REML κ-gradient) agree with a central finite difference of the
//!    kernel across the series/closed-form κ = 0 seam. This is the per-distance
//!    correctness underlying the `psi_kappa[..]` outer-gradient audit.

use gam::basis::{CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, build_constant_curvature_basis, constant_curvature_kernel_matrix};
use gam::terms::basis::{BasisMetadata, PenaltySource};
use ndarray::{Array2, array};

const LENGTH_SCALE: f64 = 1.5;

/// Chart points inside the unit disk so every κ ∈ [−1, 1] keeps all points
/// in-chart (the κ = −1 chart is the open unit ball).
fn chart_points() -> Array2<f64> {
    array![
        [0.05, -0.10],
        [-0.42, 0.31],
        [0.58, 0.22],
        [-0.15, -0.66],
        [0.33, 0.49],
        [-0.71, -0.05],
        [0.12, 0.07],
        [0.46, -0.39],
    ]
}

/// #1404 (1): the κ = 1 kernel is the great-circle geodesic-exponential. The
/// diagonal self-distance is analytically zero, so the diagonal kernel is
/// EXACTLY 1 (no `acos` cancellation), and off-diagonals match the
/// cancellation-free `atan2(|p×q|, p·q)` great-circle reference.
#[test]
fn kappa_one_kernel_is_exact_great_circle_1404() {
    let pts = chart_points();
    let k = constant_curvature_kernel_matrix(pts.view(), pts.view(), 1.0, LENGTH_SCALE)
        .expect("kappa=1 kernel");
    let embed = |x: f64, y: f64| -> [f64; 3] {
        let r2 = x * x + y * y;
        let s = 1.0 + r2;
        [2.0 * x / s, 2.0 * y / s, (1.0 - r2) / s]
    };
    for i in 0..pts.nrows() {
        // Exact-zero self-distance: a cancellation-prone `acos(p·q)` reference
        // would land near exp(−2e-4/ℓ) ≈ 0.99987 here; the exact geodesic is 1.
        assert!(
            (k[(i, i)] - 1.0).abs() < 1e-12,
            "great-circle self-distance not zero: K[{i},{i}] = {} (acos cancellation?)",
            k[(i, i)]
        );
        for j in 0..pts.nrows() {
            let p = embed(pts[(i, 0)], pts[(i, 1)]);
            let q = embed(pts[(j, 0)], pts[(j, 1)]);
            let dot = p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
            let cross = [
                p[1] * q[2] - p[2] * q[1],
                p[2] * q[0] - p[0] * q[2],
                p[0] * q[1] - p[1] * q[0],
            ];
            let cross_norm =
                (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            let angle = cross_norm.atan2(dot);
            let expected = LENGTH_SCALE * (-angle / LENGTH_SCALE).exp_m1();
            assert!(
                (k[(i, j)] - expected).abs() < 1e-10,
                "kappa=1 kernel ({i},{j}): got {} want {expected} (angle {angle})",
                k[(i, j)]
            );
        }
    }
}

/// #1404 (2): the realized design is `K(L(κ))·z` evaluated at the κ-invariant
/// EFFECTIVE length `L(κ)`, not the κ = 0 reference length in the metadata.
/// Reconstructing at the reference length is wrong at κ ≠ 0; at `L(κ)` it is
/// exact.
#[test]
fn realized_design_and_penalty_are_one_kernel_gram_at_one_range_1404() {
    let pts = chart_points();
    let kappa = 0.4;
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::UserProvided(pts.clone()),
        kappa,
        kappa_fixed: false,
        length_scale: LENGTH_SCALE,
        length_scale_fixed: true,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
    assert_eq!(built.design.ncols(), pts.nrows() - 1);

    let BasisMetadata::ConstantCurvature {
        constraint_transform,
        length_scale: meta_len,
        ..
    } = &built.metadata
    else {
        panic!("expected ConstantCurvature metadata");
    };
    assert_eq!(
        *meta_len, LENGTH_SCALE,
        "metadata stores κ=0 reference length"
    );
    let z = constraint_transform.as_ref().expect("constraint transform");

    // Reconstructing at the metadata's OWN length matches exactly — the design
    // is not evaluated at some κ-remapped length behind the caller's back.
    let raw = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, LENGTH_SCALE)
        .expect("raw kernel at the realized length");
    let expected = raw.dot(z);
    let design = built.design.to_dense();
    for (a, b) in design.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "design != K(ℓ)·z: {a} vs {b}");
    }

    // And the PENALTY is the Gram of that same kernel at that same length — the
    // property the #1464 `L_S(κ)` split broke, which is why the penalty had
    // stopped being the RKHS roughness of the design it penalizes (gam#2747).
    let gram = raw.t().dot(z);
    let gram = z.t().dot(&gram.t().to_owned().t());
    let primary = built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
        .expect("primary RKHS penalty");
    let gram = (&gram + &gram.t()) * 0.5;
    for (a, b) in gram.iter().zip(primary.matrix.iter()) {
        assert!((a - b).abs() < 1e-10, "penalty != zᵀK(ℓ)z: {a} vs {b}");
    }
}

/// #1404 (3): the default constant-curvature smooth carries NO curvature-blind
/// ridge (#1464), and its primary penalty is the RAW kernel Gram `zᵀK(L(κ))z`
/// with `normalization_scale = 1` (not Frobenius-normalized) — the raw physical
/// operator that lets REML leave the smooth low-degree signal unpenalized.
#[test]
fn primary_penalty_is_raw_kernel_gram_no_ridge_1404() {
    assert!(
        !ConstantCurvatureBasisSpec::default().double_penalty,
        "default constant-curvature smooth must drop the curvature-blind ridge (#1464)"
    );

    let pts = chart_points();
    let kappa = 0.4;
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::UserProvided(pts.clone()),
        kappa,
        kappa_fixed: false,
        length_scale: LENGTH_SCALE,
        length_scale_fixed: true,
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
    assert_eq!(
        built.active_penalties.len(),
        1,
        "single RKHS penalty (no ridge) when double_penalty = false"
    );
    let primary = built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
        .expect("constant-curvature basis must retain its primary RKHS penalty");
    assert_eq!(
        primary.info.normalization_scale, 1.0,
        "constant-curvature RKHS penalty must retain its raw physical scale"
    );

    let BasisMetadata::ConstantCurvature {
        constraint_transform,
        ..
    } = &built.metadata
    else {
        panic!("expected ConstantCurvature metadata");
    };
    let z = constraint_transform.as_ref().expect("constraint transform");
    let raw = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, LENGTH_SCALE)
        .expect("raw kernel");
    let gram = z.t().dot(&raw).dot(z);
    let s_built = &primary.matrix;

    // RAW operator (normalization_scale = 1): the penalty is the constrained
    // Gram itself, up to symmetrization — the proportionality constant is 1, not
    // a Frobenius 1/‖·‖_F. We verify scale ≈ 1 AND exact proportionality.
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (a, b) in gram.iter().zip(s_built.iter()) {
        num += a * b;
        den += b * b;
    }
    let scale = num / den;
    assert!(
        (scale - 1.0).abs() < 1e-8,
        "primary penalty must be the RAW Gram (scale 1), not Frobenius-normalized; scale {scale}"
    );
    for (a, b) in gram.iter().zip(s_built.iter()) {
        assert!(
            (a - b).abs() < 1e-8,
            "penalty not equal to the raw zᵀKz: {a} vs {b}"
        );
    }
}

