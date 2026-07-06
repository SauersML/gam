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
//! 2. CONSTRAINED-KERNEL-GRAM AT κ ≠ 0. The realized design is `K(L(κ))·z`
//!    where `L(κ)` is the κ-invariant EFFECTIVE length (the #944/#1059
//!    fill-invariance fix), NOT the κ = 0 reference length stored in metadata.
//!    Reconstructing the design with the reference length (the way the failing
//!    test did) is wrong at κ ≠ 0 (it reported `-0.2420` vs `-0.2317`). The fix
//!    exposes `constant_curvature_effective_length`; reconstruction at `L(κ)`
//!    matches exactly.
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

use gam::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, constant_curvature_effective_length,
    constant_curvature_kernel_kappa_jets, constant_curvature_kernel_matrix,
};
use gam::terms::basis::BasisMetadata;
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
            let expected = (-angle / LENGTH_SCALE).exp();
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
fn realized_design_uses_effective_length_kernel_gram_1404() {
    let pts = chart_points();
    let kappa = 0.4;
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::UserProvided(pts.clone()),
        kappa,
        kappa_fixed: false,
        length_scale: LENGTH_SCALE,
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

    let ell_eff = constant_curvature_effective_length(pts.view(), pts.view(), LENGTH_SCALE, kappa)
        .expect("effective length");
    assert!(
        (ell_eff - LENGTH_SCALE).abs() > 1e-6,
        "at κ=0.4 the effective length must differ from the reference (got {ell_eff})"
    );

    // Reconstructing at the EFFECTIVE length matches exactly.
    let raw_eff = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, ell_eff)
        .expect("raw kernel at L(κ)");
    let expected = raw_eff.dot(z);
    let design = built.design.to_dense();
    for (a, b) in design.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "design != K(L(κ))·z: {a} vs {b}");
    }

    // Reconstructing at the κ=0 REFERENCE length does NOT match (the bug): this
    // pins that the build genuinely uses L(κ), not the stored reference length.
    let raw_ref = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, LENGTH_SCALE)
        .expect("raw kernel at reference length");
    let wrong = raw_ref.dot(z);
    let max_gap = design
        .iter()
        .zip(wrong.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_gap > 1e-4,
        "reference-length reconstruction must visibly disagree (the #1404 bug); gap {max_gap}"
    );
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
        double_penalty: false,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
    assert_eq!(
        built.penalties.len(),
        1,
        "single RKHS penalty (no ridge) when double_penalty = false"
    );

    let BasisMetadata::ConstantCurvature {
        constraint_transform,
        ..
    } = &built.metadata
    else {
        panic!("expected ConstantCurvature metadata");
    };
    let z = constraint_transform.as_ref().expect("constraint transform");
    let ell_eff = constant_curvature_effective_length(pts.view(), pts.view(), LENGTH_SCALE, kappa)
        .expect("effective length");
    let raw = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, ell_eff)
        .expect("raw kernel");
    let gram = z.t().dot(&raw).dot(z);
    let s_built = &built.penalties[0];

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

/// #1404 (4): the kernel κ-jets (which feed the outer LAML/REML κ-gradient)
/// agree with a central finite difference of the kernel across the
/// series/closed-form κ = 0 seam — the per-distance correctness underlying the
/// `psi_kappa[..]` outer-gradient audit.
#[test]
fn kernel_kappa_jets_match_central_fd_1404() {
    let pts = chart_points();
    let centers = pts.slice(ndarray::s![..4, ..]).to_owned();
    for &kappa in &[-0.7_f64, -1e-6, 0.0, 1e-6, 0.9] {
        let (k0, dk, dkk) =
            constant_curvature_kernel_kappa_jets(pts.view(), centers.view(), kappa, LENGTH_SCALE)
                .expect("jets");
        let k_at = |kk: f64| {
            constant_curvature_kernel_matrix(pts.view(), centers.view(), kk, LENGTH_SCALE)
                .expect("kernel")
        };
        for (a, b) in k0.iter().zip(k_at(kappa).iter()) {
            assert!(
                (a - b).abs() < 1e-13,
                "jet value channel desynced from the plain kernel at kappa={kappa}"
            );
        }
        let h = 1e-5;
        let kp = k_at(kappa + h);
        let km = k_at(kappa - h);
        for i in 0..pts.nrows() {
            for j in 0..centers.nrows() {
                let fd1 = (kp[(i, j)] - km[(i, j)]) / (2.0 * h);
                assert!(
                    (dk[(i, j)] - fd1).abs() < 1e-6 + 1e-5 * fd1.abs(),
                    "dK/dκ mismatch at kappa={kappa} ({i},{j}): jet {} fd {fd1}",
                    dk[(i, j)]
                );
                let fd2 = (kp[(i, j)] - 2.0 * k0[(i, j)] + km[(i, j)]) / (h * h);
                assert!(
                    (dkk[(i, j)] - fd2).abs() < 1e-3 + 1e-3 * fd2.abs(),
                    "d²K/dκ² mismatch at kappa={kappa} ({i},{j}): jet {} fd {fd2}",
                    dkk[(i, j)]
                );
            }
        }
    }
}
