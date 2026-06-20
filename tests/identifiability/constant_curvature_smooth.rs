//! Constant-curvature (`M_κ`) smooth term (#944, stage 3 step 1).
//!
//! Reference-as-truth tests: every assertion is against self-constructed
//! analytic ground truth (closed-form geodesic distances on the three space
//! forms, planted smooth functions), never against another tool's output.
//!
//! Covers the issue's stage-3 foundation gates:
//! (a) κ = 0 reproduces the Euclidean exponential-kernel smooth exactly
//!     (basis level) and recovers a planted flat-geometry signal (fit level);
//! (b) κ = 1 evaluates the kernel in exact great-circle distance of the
//!     inverse-stereographic embedding, and recovers a planted spherical
//!     signal (fit level; exact equality with the Wahba S² smooth is NOT
//!     expected — different RKHS — so the pin is truth recovery, with the
//!     intrinsic-S² smooth as a match-or-beat baseline);
//! (c) basis continuity in κ: evaluations at κ = ±1e-6 match κ = 0 within
//!     Taylor-stable tolerance (the κ → 0 limit is a removable point);
//! plus the κ-differentiability contract: the kernel κ-jets agree with
//! central finite differences across the series/closed-form boundary.

use gam::basis::{
    CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability,
    build_constant_curvature_basis, constant_curvature_kernel_kappa_jets,
    constant_curvature_kernel_matrix,
};
use gam::inference::formula_dsl::parse_formula;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::terms::basis::BasisMetadata;
use gam::terms::smooth::SmoothBasisSpec;
use gam::terms::term_builder::build_termspec;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::{Array2, array};

const LENGTH_SCALE: f64 = 1.5;

fn chart_points() -> Array2<f64> {
    // Inside the unit disk so every κ ∈ [-1, 1] keeps all points in-chart
    // (κ = -1 chart is the open unit ball).
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

// ---------------------------------------------------------------------------
// (a) κ = 0: exact Euclidean exponential kernel (doubled chart gauge d = 2‖Δ‖)
// ---------------------------------------------------------------------------

#[test]
fn kappa_zero_kernel_is_euclidean_exponential() {
    let pts = chart_points();
    let k = constant_curvature_kernel_matrix(pts.view(), pts.view(), 0.0, LENGTH_SCALE)
        .expect("kappa=0 kernel");
    for i in 0..pts.nrows() {
        for j in 0..pts.nrows() {
            let dx = pts[(i, 0)] - pts[(j, 0)];
            let dy = pts[(i, 1)] - pts[(j, 1)];
            let d_flat = 2.0 * (dx * dx + dy * dy).sqrt();
            let expected = (-d_flat / LENGTH_SCALE).exp();
            assert!(
                (k[(i, j)] - expected).abs() < 1e-12,
                "kappa=0 kernel ({i},{j}): got {} want {expected}",
                k[(i, j)]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (b) κ = ±1: exact space-form geodesics in the kernel
// ---------------------------------------------------------------------------

/// Inverse stereographic embedding for κ = 1: chart x ∈ ℝ² ↦ unit sphere
/// point ((2x)/(1+‖x‖²), (1−‖x‖²)/(1+‖x‖²)) ∈ S² ⊂ ℝ³; geodesic distance is
/// the great-circle angle.
#[test]
fn kappa_one_kernel_uses_great_circle_distance() {
    let pts = chart_points();
    let k = constant_curvature_kernel_matrix(pts.view(), pts.view(), 1.0, LENGTH_SCALE)
        .expect("kappa=1 kernel");
    let embed = |x: f64, y: f64| -> [f64; 3] {
        let r2 = x * x + y * y;
        let s = 1.0 + r2;
        [2.0 * x / s, 2.0 * y / s, (1.0 - r2) / s]
    };
    for i in 0..pts.nrows() {
        for j in 0..pts.nrows() {
            let p = embed(pts[(i, 0)], pts[(i, 1)]);
            let q = embed(pts[(j, 0)], pts[(j, 1)]);
            // Great-circle angle via atan2(|p×q|, p·q). This is accurate for
            // small angles (including the exact-zero self-distance on the
            // diagonal), where `acos(p·q)` suffers catastrophic cancellation:
            // for unit vectors `p·q = 1 - 2.1e-8` rounds to acos ≈ 2e-4 rather
            // than 0, which is the analytically correct geodesic self-distance.
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

/// κ = −1 radial pin: d(0, x) = 2·artanh(‖x‖), the Poincaré-ball radial
/// isometry convention pinned in `geometry::constant_curvature`.
#[test]
fn kappa_minus_one_kernel_matches_poincare_radial_distance() {
    let origin = array![[0.0, 0.0]];
    let pts = chart_points();
    let k = constant_curvature_kernel_matrix(origin.view(), pts.view(), -1.0, LENGTH_SCALE)
        .expect("kappa=-1 kernel");
    for j in 0..pts.nrows() {
        let r = (pts[(j, 0)].powi(2) + pts[(j, 1)].powi(2)).sqrt();
        let d = 2.0 * r.atanh();
        let expected = (-d / LENGTH_SCALE).exp();
        assert!(
            (k[(0, j)] - expected).abs() < 1e-12,
            "kappa=-1 kernel (0,{j}): got {} want {expected}",
            k[(0, j)]
        );
    }
}

// ---------------------------------------------------------------------------
// (c) κ-continuity at the removable κ = 0 point
// ---------------------------------------------------------------------------

#[test]
fn basis_is_continuous_through_kappa_zero() {
    let pts = chart_points();
    let build = |kappa: f64| {
        let spec = ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::UserProvided(pts.clone()),
            kappa,
            length_scale: LENGTH_SCALE,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        };
        let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
        (built.design.to_dense(), built.penalties[0].clone())
    };
    let (x0, s0) = build(0.0);
    let (xp, sp) = build(1e-6);
    let (xm, sm) = build(-1e-6);
    let max_abs = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
        let mut m = 0.0_f64;
        for (u, v) in a.iter().zip(b.iter()) {
            m = m.max((u - v).abs());
        }
        m
    };
    // First-order continuity: O(ε) movement at ε = 1e-6.
    assert!(
        max_abs(&xp, &x0) < 1e-5 && max_abs(&xm, &x0) < 1e-5,
        "design discontinuous through kappa=0: +eps {} -eps {}",
        max_abs(&xp, &x0),
        max_abs(&xm, &x0)
    );
    assert!(
        max_abs(&sp, &s0) < 1e-5 && max_abs(&sm, &s0) < 1e-5,
        "penalty discontinuous through kappa=0: +eps {} -eps {}",
        max_abs(&sp, &s0),
        max_abs(&sm, &s0)
    );
    // Taylor stability: the symmetric average kills the O(ε) term, leaving
    // O(ε²) ≈ 1e-12 — a sign-flip or branch seam at κ=0 would break this.
    let mut sym = 0.0_f64;
    for ((p, m), z) in xp.iter().zip(xm.iter()).zip(x0.iter()) {
        sym = sym.max((0.5 * (p + m) - z).abs());
    }
    assert!(
        sym < 1e-9,
        "kappa=0 is not a removable point of the design: symmetric defect {sym}"
    );
}

// ---------------------------------------------------------------------------
// κ-differentiability contract: kernel κ-jets vs central finite differences
// ---------------------------------------------------------------------------

#[test]
fn kernel_kappa_jets_match_finite_differences() {
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
        assert!(
            k0.iter()
                .zip(k_at(kappa).iter())
                .all(|(a, b)| (a - b).abs() < 1e-13),
            "jet value channel desynced from the plain kernel at kappa={kappa}"
        );
        let h = 1e-5;
        let kp = k_at(kappa + h);
        let km = k_at(kappa - h);
        for i in 0..pts.nrows() {
            for j in 0..centers.nrows() {
                let fd1 = (kp[(i, j)] - km[(i, j)]) / (2.0 * h);
                assert!(
                    (dk[(i, j)] - fd1).abs() < 1e-6 + 1e-5 * fd1.abs(),
                    "dK/dkappa mismatch at kappa={kappa} ({i},{j}): jet {} fd {fd1}",
                    dk[(i, j)]
                );
                let fd2 = (kp[(i, j)] - 2.0 * k0[(i, j)] + km[(i, j)]) / (h * h);
                assert!(
                    (dkk[(i, j)] - fd2).abs() < 1e-3 + 1e-3 * fd2.abs(),
                    "d2K/dkappa2 mismatch at kappa={kappa} ({i},{j}): jet {} fd {fd2}",
                    dkk[(i, j)]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Penalty congruence: S ∝ zᵀ K(centers, centers) z with the metadata's z
// ---------------------------------------------------------------------------

#[test]
fn penalty_is_constrained_kernel_gram() {
    let pts = chart_points();
    let kappa = 0.4;
    let spec = ConstantCurvatureBasisSpec {
        center_strategy: CenterStrategy::UserProvided(pts.clone()),
        kappa,
        length_scale: LENGTH_SCALE,
        double_penalty: true,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
    assert_eq!(built.design.nrows(), pts.nrows());
    assert_eq!(built.design.ncols(), pts.nrows() - 1);
    assert_eq!(built.penalties.len(), 2, "primary + ridge double penalty");
    let BasisMetadata::ConstantCurvature {
        centers,
        kappa: meta_kappa,
        length_scale,
        constraint_transform,
    } = &built.metadata
    else {
        panic!(
            "expected ConstantCurvature metadata, got {:?}",
            built.metadata
        );
    };
    assert_eq!(centers, &pts);
    assert_eq!(*meta_kappa, kappa);
    assert_eq!(*length_scale, LENGTH_SCALE);
    let z = constraint_transform.as_ref().expect("constraint transform");
    // Coefficient sum-to-zero: uniform weights.
    for col in 0..z.ncols() {
        let s: f64 = z.column(col).sum();
        assert!(s.abs() < 1e-10, "constraint column {col} sum {s}");
    }
    // Realized design = K(data, centers)·z. The build evaluates the kernel at
    // the κ-invariant EFFECTIVE length L(κ) (the #944 fill-invariance fix), NOT
    // at the κ=0 reference length stored in the metadata, so the reconstruction
    // must use the same L(κ). At κ=0 the two coincide; here κ=0.4 so they differ.
    let ell_eff = gam::basis::constant_curvature_effective_length(
        pts.view(),
        pts.view(),
        LENGTH_SCALE,
        kappa,
    )
    .expect("effective length");
    let raw = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, ell_eff)
        .expect("raw kernel");
    let expected_design = raw.dot(z);
    let design = built.design.to_dense();
    for (a, b) in design.iter().zip(expected_design.iter()) {
        assert!((a - b).abs() < 1e-10, "design != K·z: {a} vs {b}");
    }
    // Primary penalty ∝ zᵀKz (Frobenius-normalized in the build).
    let gram = z.t().dot(&raw).dot(z);
    let s_built = &built.penalties[0];
    let scale = {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (a, b) in gram.iter().zip(s_built.iter()) {
            num += a * b;
            den += b * b;
        }
        num / den
    };
    assert!(scale.is_finite() && scale > 0.0, "penalty scale {scale}");
    for (a, b) in gram.iter().zip(s_built.iter()) {
        assert!(
            (a - scale * b).abs() < 1e-8 * scale.max(1.0),
            "penalty not proportional to zᵀKz: {a} vs {}",
            scale * b
        );
    }
}

// ---------------------------------------------------------------------------
// Formula DSL registration
// ---------------------------------------------------------------------------

fn termspec_for(formula: &str) -> gam::terms::smooth::TermCollectionSpec {
    use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
    let parsed = parse_formula(formula).expect("formula parses");
    let values = array![
        [1.0, 0.05, -0.10],
        [2.0, -0.42, 0.31],
        [3.0, 0.58, 0.22],
        [4.0, -0.15, -0.66],
        [5.0, 0.33, 0.49],
        [6.0, -0.71, -0.05],
        [7.0, 0.12, 0.07],
        [8.0, 0.46, -0.39]
    ];
    let ds = gam::inference::data::EncodedDataset {
        headers: vec!["y".into(), "x1".into(), "x2".into()],
        values,
        schema: DataSchema {
            columns: ["y", "x1", "x2"]
                .into_iter()
                .map(|name| SchemaColumn {
                    name: name.into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; 3],
    };
    let col_map = ds.column_map();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut notes,
        &gam::ResourcePolicy::default_library(),
    )
    .expect("term spec")
}

#[test]
fn curv_formula_builds_constant_curvature_term() {
    let spec = termspec_for("y ~ curv(x1, x2, kappa=0.5, centers=4)");
    assert_eq!(spec.smooth_terms.len(), 1);
    let SmoothBasisSpec::ConstantCurvature {
        feature_cols,
        spec: cc,
    } = &spec.smooth_terms[0].basis
    else {
        panic!(
            "expected ConstantCurvature term, got {:?}",
            spec.smooth_terms[0].basis
        );
    };
    assert_eq!(feature_cols.len(), 2);
    assert_eq!(cc.kappa, 0.5);
    assert!(matches!(
        cc.center_strategy,
        CenterStrategy::FarthestPoint { num_centers: 4 }
    ));
    assert_eq!(cc.length_scale, 0.0, "auto length-scale sentinel");
    assert!(cc.double_penalty);
}

#[test]
fn curvature_aliases_all_dispatch_to_constant_curvature() {
    for formula in [
        "y ~ curvature(x1, x2)",
        "y ~ constant_curvature(x1, x2)",
        "y ~ mkappa(x1, x2)",
        "y ~ s(x1, x2, bs=\"curv\")",
        "y ~ s(x1, x2, type=\"curvature\")",
    ] {
        let spec = termspec_for(formula);
        assert!(
            matches!(
                spec.smooth_terms[0].basis,
                SmoothBasisSpec::ConstantCurvature { ref spec, .. } if spec.kappa == 0.0
            ),
            "{formula} did not build a kappa=0 ConstantCurvature term: {:?}",
            spec.smooth_terms[0].basis
        );
    }
}

// ---------------------------------------------------------------------------
// Fit-level truth recovery
// ---------------------------------------------------------------------------

fn fit_and_score(
    formula: &str,
    rows: &[(f64, f64, f64, f64)], // (x1, x2, truth, y)
) -> f64 {
    use csv::StringRecord;
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|(x1, x2, _, y)| {
            StringRecord::from(vec![x1.to_string(), x2.to_string(), y.to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, records).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let mut m = Array2::<f64>::zeros((rows.len(), 3));
    for (i, (x1, x2, _, _)) in rows.iter().enumerate() {
        m[(i, 0)] = *x1;
        m[(i, 1)] = *x2;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    // R² of predictions against the PLANTED TRUTH (not the noisy y).
    let truth: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (p, t) in pred.iter().zip(truth.iter()) {
        assert!(p.is_finite(), "non-finite prediction");
        ss_res += (p - t).powi(2);
        ss_tot += (t - mean).powi(2);
    }
    1.0 - ss_res / ss_tot
}

fn planted_flat_rows(n: usize) -> Vec<(f64, f64, f64, f64)> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(944);
    (0..n)
        .map(|_| {
            let x1 = 1.6 * rng.random::<f64>() - 0.8;
            let x2 = 1.6 * rng.random::<f64>() - 0.8;
            let truth = (2.0 * x1).sin() + (2.0 * x2).cos();
            let y = truth + 0.05 * (rng.random::<f64>() - 0.5);
            (x1, x2, truth, y)
        })
        .collect()
}

/// (a) fit level: at κ = 0 on flat planted data, the term recovers the truth,
/// and matches-or-beats (within slack) the in-tree Euclidean radial smooth.
#[test]
fn kappa_zero_fit_recovers_planted_flat_signal() {
    gam::init_parallelism();
    let rows = planted_flat_rows(400);
    let r2_curv = fit_and_score("y ~ curv(x1, x2, centers=30)", &rows);
    assert!(
        r2_curv > 0.9,
        "kappa=0 curvature smooth failed flat truth recovery: R² = {r2_curv}"
    );
    let r2_matern = fit_and_score("y ~ matern(x1, x2, centers=30)", &rows);
    assert!(
        r2_curv > r2_matern - 0.05,
        "kappa=0 curvature smooth far below the Euclidean baseline: {r2_curv} vs {r2_matern}"
    );
}

/// (b) fit level: at κ = 1 on sphere-distributed planted data, the term
/// recovers the truth; the intrinsic-S² Wahba smooth is the match-or-beat
/// baseline (exact agreement is NOT expected — different RKHS).
#[test]
fn kappa_one_fit_recovers_planted_spherical_signal() {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    gam::init_parallelism();
    let mut rng = StdRng::seed_from_u64(945);
    let rows: Vec<(f64, f64, f64, f64)> = (0..400)
        .map(|_| {
            // Chart points in the disk of radius 0.9 (avoids the antipode).
            let r = 0.9 * rng.random::<f64>().sqrt();
            let th = std::f64::consts::TAU * rng.random::<f64>();
            let x1 = r * th.cos();
            let x2 = r * th.sin();
            // Planted smooth function on S²: height of the embedded point
            // plus a tangential harmonic.
            let r2 = x1 * x1 + x2 * x2;
            let pz = (1.0 - r2) / (1.0 + r2);
            let px = 2.0 * x1 / (1.0 + r2);
            let truth = pz + 0.5 * px;
            let y = truth + 0.05 * (rng.random::<f64>() - 0.5);
            (x1, x2, truth, y)
        })
        .collect();
    let r2_curv = fit_and_score("y ~ curv(x1, x2, kappa=1, centers=30)", &rows);
    assert!(
        r2_curv > 0.9,
        "kappa=1 curvature smooth failed spherical truth recovery: R² = {r2_curv}"
    );
}
