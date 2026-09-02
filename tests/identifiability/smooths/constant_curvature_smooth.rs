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

use gam::basis::{CenterStrategy, ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, build_constant_curvature_basis, constant_curvature_kernel_matrix};
use gam::inference::formula_dsl::parse_formula;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::terms::basis::{BasisMetadata, PenaltySource};
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
            let expected = LENGTH_SCALE * (-d_flat / LENGTH_SCALE).exp_m1();
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
            let expected = LENGTH_SCALE * (-angle / LENGTH_SCALE).exp_m1();
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
        let expected = LENGTH_SCALE * (-d / LENGTH_SCALE).exp_m1();
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
            kappa_fixed: false,
            length_scale: LENGTH_SCALE,
            length_scale_fixed: true,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        };
        let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
        let penalty = built
            .active_penalties
            .iter()
            .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
            .expect("constant-curvature basis must retain its primary RKHS penalty")
            .matrix
            .clone();
        (built.design.to_dense(), penalty)
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
        kappa_fixed: false,
        length_scale: LENGTH_SCALE,
        length_scale_fixed: true,
        double_penalty: true,
        identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
    };
    let built = build_constant_curvature_basis(pts.view(), &spec).expect("build");
    assert_eq!(built.design.nrows(), pts.nrows());
    assert_eq!(built.design.ncols(), pts.nrows() - 1);
    assert_eq!(
        built.active_penalties.len(),
        2,
        "primary + ridge double penalty"
    );
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
    // Realized design = K(data, centers)·z at the spec's OWN range. Design and
    // penalty are two blocks of ONE Gram at ONE ℓ (gam#2747): the fill-invariant
    // `L(κ)` / `L_S(κ)` remapping is gone, so the reconstruction uses the length
    // the metadata reports and matches at every κ, not only at κ = 0.
    let raw = constant_curvature_kernel_matrix(pts.view(), pts.view(), kappa, *length_scale)
        .expect("raw kernel");
    let expected_design = raw.dot(z);
    let design = built.design.to_dense();
    for (a, b) in design.iter().zip(expected_design.iter()) {
        assert!((a - b).abs() < 1e-10, "design != K·z: {a} vs {b}");
    }
    // Primary penalty ∝ zᵀKz (Frobenius-normalized in the build).
    let gram = z.t().dot(&raw).dot(z);
    let s_built = &built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
        .expect("constant-curvature basis must retain its primary RKHS penalty")
        .matrix;
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
    // curv() defaults to NO double-penalty ridge (#1464: the curvature-blind ridge
    // `I` absorbs the data fit independently of κ and railed the fitted curvature
    // to a chart bound; the RKHS Gram penalty is already full-rank PD). An explicit
    // `double_penalty=` is still honoured.
    assert!(!cc.double_penalty);
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
    // What geometry did the fit actually realize? The range is the smooth's
    // second outer coordinate (gam#2747) and this file's whole subject is what
    // limits the recovery, so the realized κ and ℓ belong in the record rather
    // than being inferred from the formula.
    for term in fit.resolvedspec.smooth_terms.iter() {
        if let gam::smooth::SmoothBasisSpec::ConstantCurvature { spec, .. } = &term.basis {
            eprintln!(
                "[2687-realized] {formula}  ->  kappa={} (pinned={})  length_scale={} (pinned={})",
                spec.kappa, spec.kappa_fixed, spec.length_scale, spec.length_scale_fixed
            );
        }
    }
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
    gam_runtime::test_support::install_diagnostic_logger();
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
///
/// **This test is NOT a κ-search-box victim** (gam#2687 listed it as one). The
/// `kappa=1` here is a PIN: `constant_curvature_kappa_is_fixed` filters the term
/// out of the κ optimization, so `constant_curvature_kappa_bounds` is never
/// consulted and the `0.5/0.81 = 0.617` cap quoted on that issue does not act on
/// this fit. What fails is the R² bar, at 0.8937 against 0.9.
///
/// `measure_2687_kappa_one_spherical_recovery_against_capacity_and_baselines`
/// above records what actually limits it: the basis is capacity-starved at 30
/// centers (0.9384 at 60), the auto length scale is ~6× broader than the
/// sweep's optimum (0.9428 at ℓ = 0.3 with the same 30 centers), and `matern` at
/// the same budget reaches 0.99988 — 900× less residual — on data generated on
/// the very sphere this basis models. The baseline named in the line above is
/// still not the one this assertion makes.
///
/// The number moved from 0.8937 to 0.8910 under gam#2747, and the 0.003 is the
/// whole of the effect: the design is now evaluated at the range the spec
/// states, where it used to be silently re-evaluated at the fill-invariant
/// `L(κ)`, which at κ = 1 happened to be shorter and therefore slightly closer
/// to this fixture's R²-optimal 0.3. Honouring the stated range is the point;
/// the remedy the measurement names — more centers, or a range chosen for this
/// data — is unchanged and is not what this assertion tests.
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

/// gam#2687 MEASUREMENT: what limits `kappa_one_fit_recovers_planted_spherical_
/// signal`, recorded so the next reader does not have to re-derive it.
///
/// #2687 listed that test as a victim of the κ search box, at "the production
/// cap `0.5/0.81 = 0.617` against a planted κ⋆ = 1.0, 62% outside the interval
/// the estimator is boxed to". **That attribution is wrong**, and this
/// measurement is why. The fixture PINS `kappa=1`, so
/// `constant_curvature_kappa_is_fixed` filters the term out of the κ
/// optimization entirely and `constant_curvature_kappa_bounds` is never
/// consulted; the geometry really is built at κ = 1. What fails is a bare
/// R² bar, by 0.6%.
///
/// The measured picture (400 rows on a radius-0.9 disk, planted `pz + 0.5·px`):
///
/// | fit | R² |
/// |---|---|
/// | `curv(kappa=1, centers=15)` | 0.7875 |
/// | `curv(kappa=1, centers=30)` — **the fixture** | 0.8937 |
/// | `curv(kappa=1, centers=60)` | 0.9402 |
/// | `curv(kappa=1, centers=120)` | 0.9550 |
/// | `curv(kappa=1, centers=30, length_scale=0.3)` | 0.9523 |
/// | `curv(kappa=0, centers=30)` | 0.8744 |
/// | `curv(kappa=2, centers=30)` | 0.9348 |
/// | **`matern(centers=30)`** | **0.99988** |
/// | **`thinplate(...)`** | **0.99987** |
///
/// Three things the fixture's 0.6% miss was hiding:
///
/// 1. **The Euclidean baselines nail it.** `matern` at the SAME center budget
///    leaves 885× less residual than `curv` does at the CORRECT curvature, on
///    data generated on the very sphere `curv` models. The fixture's own doc
///    names the intrinsic-S² Wahba smooth as its "match-or-beat baseline" and
///    then asserts a bare `> 0.9` instead — so the comparison that would have
///    shown this was documented and never computed. Its κ = 0 sibling
///    (`kappa_zero_fit_recovers_planted_flat_signal`) does compute one.
/// 2. **The auto length scale is ~6× too broad.** `realized_constant_curvature_
///    length_scale` is the median pairwise CHART distance among centers, doubled
///    — ≈ 1.8 here — and the sweep peaks at ℓ ≈ 0.3, which recovers 0.9523 at the
///    fixture's own 30 centers. The median heuristic is sized to the cloud
///    DIAMETER while the exponential kernel `exp(−d/ℓ)` wants something nearer
///    the center SPACING.
/// 3. **κ = 2 fits data generated at κ = 1 better than κ = 1 does** (0.9348 vs
///    0.8937), which is the same monotone-in-κ preference the coverage fixture
///    rails on, showing up in pure fit quality at a PINNED κ.
///
/// None of the three is the κ box.
///
/// ## The obvious repair, A/B'd and REJECTED
///
/// Point 2 suggests replacing the median PAIRWISE distance (a cloud-DIAMETER
/// scale, which makes every basis function nearly constant across the data) with
/// the median NEAREST-NEIGHBOUR distance (a centre-SPACING scale) — the textbook
/// RBF range rule. Measured over the whole constant-curvature identifiability
/// suite, that arm **fixes `kappa_one_fit_recovers_planted_spherical_signal` and
/// breaks its κ = 0 sibling**: `kappa_zero_fit_recovers_planted_flat_signal`
/// drops to R² = 0.9491 against its `matern − 0.05` baseline bar of 0.94994,
/// missing by 0.0008. One in, one out, both marginal — so the heuristic is not
/// the leading term and the change is not landed. Recorded here so the next
/// reader does not re-run it.
///
/// What the κ = 0 arm also shows, incidentally, is that `curv` sits at ~0.95
/// against `matern`'s 0.99994 on FLAT data too, and that fixture's 0.05 slack is
/// sized to accommodate exactly that. The gap is the basis, on both branches,
/// not the curvature.
///
/// This test asserts only that the measurement ran, so it records the numbers
/// without adding a bar nobody derived.
#[test]
fn measure_2687_kappa_one_spherical_recovery_against_capacity_and_baselines() {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    gam::init_parallelism();
    let mut rng = StdRng::seed_from_u64(945);
    let rows: Vec<(f64, f64, f64, f64)> = (0..400)
        .map(|_| {
            let r = 0.9 * rng.random::<f64>().sqrt();
            let th = std::f64::consts::TAU * rng.random::<f64>();
            let x1 = r * th.cos();
            let x2 = r * th.sin();
            let r2 = x1 * x1 + x2 * x2;
            let pz = (1.0 - r2) / (1.0 + r2);
            let px = 2.0 * x1 / (1.0 + r2);
            let truth = pz + 0.5 * px;
            let y = truth + 0.05 * (rng.random::<f64>() - 0.5);
            (x1, x2, truth, y)
        })
        .collect();
    let mut scored = Vec::new();
    for f in [
        "y ~ curv(x1, x2, kappa=1, centers=15)",
        "y ~ curv(x1, x2, kappa=1, centers=30)",
        "y ~ curv(x1, x2, kappa=1, centers=60)",
        "y ~ curv(x1, x2, kappa=1, centers=30, length_scale=0.3)",
        "y ~ curv(x1, x2, kappa=0, centers=30)",
        "y ~ curv(x1, x2, kappa=2, centers=30)",
        "y ~ matern(x1, x2, centers=30)",
        "y ~ thinplate(x1, x2)",
    ] {
        let r2 = fit_and_score(f, &rows);
        eprintln!("[2687-measure] {f:<52} R2 = {r2:.6}");
        assert!(r2.is_finite(), "{f}: R² must be finite to be a measurement");
        scored.push((f, r2));
    }
    // The one relation worth pinning, because it is the finding and it is not a
    // tuning choice: the Euclidean baseline beats the constant-curvature smooth
    // at the CORRECT curvature and the SAME center budget, by a wide margin. If
    // that ever stops being true, the constant-curvature basis has been repaired
    // and this measurement should be re-read rather than trusted.
    let curv30 = scored[1].1;
    let matern30 = scored[6].1;
    assert!(
        matern30 > curv30,
        "the finding this records: matern at 30 centers ({matern30:.6}) must be \
         compared against curv at 30 centers ({curv30:.6}); if curv now wins, the \
         basis changed and the surrounding docs are stale"
    );
}

// ---------------------------------------------------------------------------
// gam#2152: a user-PINNED `kappa=` is a FIXED sectional curvature that the fit
// must honour verbatim — never re-derive via the #944/#1464 κ estimation.
// ---------------------------------------------------------------------------

/// A small smooth surface on the ball `‖(x1, x2)‖² < 1/3`, strictly inside BOTH
/// the κ = +K and κ = −K stereographic charts for K ≤ 3, so a pinned fit at
/// either sign is well posed rather than chart-rejected.
fn small_ball_rows(n: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x1 = 0.6 * rng.random::<f64>() - 0.3;
            let x2 = 0.6 * rng.random::<f64>() - 0.3;
            let truth = (4.0 * x1).sin() * (4.0 * x2).cos();
            let y = truth + 0.05 * (rng.random::<f64>() - 0.5);
            (x1, x2, truth, y)
        })
        .collect()
}

/// Read the FROZEN sectional curvature and its pin flag out of a fitted
/// `curv(...)` spec (the value/geometry the design was actually built and kept
/// at, post-fit).
fn fitted_kappa_and_pin(formula: &str, rows: &[(f64, f64, f64, f64)]) -> (f64, bool) {
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
    // What geometry did the fit actually realize? The range is the smooth's
    // second outer coordinate (gam#2747) and this file's whole subject is what
    // limits the recovery, so the realized κ and ℓ belong in the record rather
    // than being inferred from the formula.
    for term in fit.resolvedspec.smooth_terms.iter() {
        if let gam::smooth::SmoothBasisSpec::ConstantCurvature { spec, .. } = &term.basis {
            eprintln!(
                "[2687-realized] {formula}  ->  kappa={} (pinned={})  length_scale={} (pinned={})",
                spec.kappa, spec.kappa_fixed, spec.length_scale, spec.length_scale_fixed
            );
        }
    }
    let SmoothBasisSpec::ConstantCurvature { spec, .. } = &fit.resolvedspec.smooth_terms[0].basis
    else {
        panic!("expected a ConstantCurvature term after fit");
    };
    (spec.kappa, spec.kappa_fixed)
}

/// The core #2152 contract at the ENGINE level: the fit keeps the design and
/// penalty at the user's pinned κ. A spherical `kappa=+3` freezes at +3 and a
/// hyperbolic `kappa=-3` freezes at −3 — the fit never re-derives them to the
/// same estimated κ̂ (the byte-identical-fits symptom). This is a different
/// angle than the prediction-equality Python repro: it inspects the realized
/// geometry directly, so it catches a regression even if predictions happened to
/// coincide for unrelated reasons.
#[test]
fn pinned_kappa_is_kept_verbatim_by_the_fit() {
    gam::init_parallelism();
    let rows = small_ball_rows(600, 2152);

    let (k_pos, pinned_pos) = fitted_kappa_and_pin("y ~ curv(x1, x2, kappa=3, centers=20)", &rows);
    assert!(
        pinned_pos,
        "explicit kappa=3 must mark the term kappa_fixed"
    );
    assert!(
        (k_pos - 3.0).abs() < 1e-9,
        "pinned kappa=+3 was re-derived: fit kept κ = {k_pos} (want +3)"
    );

    let (k_neg, pinned_neg) = fitted_kappa_and_pin("y ~ curv(x1, x2, kappa=-3, centers=20)", &rows);
    assert!(
        pinned_neg,
        "explicit kappa=-3 must mark the term kappa_fixed"
    );
    assert!(
        (k_neg + 3.0).abs() < 1e-9,
        "pinned kappa=-3 was re-derived: fit kept κ = {k_neg} (want -3)"
    );

    // A pinned flat kappa=0 must also be kept exactly (the window collapses onto
    // 0), not left free to drift to a chart bound.
    let (k_flat, pinned_flat) =
        fitted_kappa_and_pin("y ~ curv(x1, x2, kappa=0, centers=20)", &rows);
    assert!(
        pinned_flat,
        "explicit kappa=0 must mark the term kappa_fixed"
    );
    assert!(
        k_flat.abs() < 1e-9,
        "pinned kappa=0 drifted off flat: fit kept κ = {k_flat} (want 0)"
    );

    // The spherical and hyperbolic pinned fits landed at genuinely different κ.
    assert!(
        (k_pos - k_neg).abs() > 1.0,
        "pinned +3 and -3 collapsed to the same κ ({k_pos} vs {k_neg})"
    );
}

/// Guard against OVER-correction: omitting `kappa=` must still leave κ FREE for
/// the #944/#1464 outer estimation (kappa_fixed = false). The pin path must not
/// swallow the estimation path.
#[test]
fn omitted_kappa_stays_free_for_estimation() {
    gam::init_parallelism();
    let rows = small_ball_rows(400, 7);
    let (_k, pinned) = fitted_kappa_and_pin("y ~ curv(x1, x2, centers=20)", &rows);
    assert!(
        !pinned,
        "omitted kappa= must leave the term free to estimate κ (kappa_fixed=false)"
    );
}

/// The issue's independent tell: because a pinned κ genuinely builds the design
/// at that κ, an out-of-chart pinned κ must now be REJECTED loudly by
/// `validate_chart_points` (`1 + κ‖x‖² > 0`) — under the old
/// silently-re-derive-κ behaviour such a fit was wrongly ACCEPTED (the realized
/// design used a different, in-chart estimated κ). Here the data reaches
/// ‖x‖² ≈ 0.18, and κ = −50 needs ‖x‖² < 1/50 = 0.02, so every far row is out of
/// chart.
#[test]
fn pinned_out_of_chart_kappa_is_rejected() {
    gam::init_parallelism();
    let rows = small_ball_rows(300, 99);
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
    let result = fit_from_formula("y ~ curv(x1, x2, kappa=-50, centers=20)", &data, &cfg);
    assert!(
        result.is_err(),
        "a pinned out-of-chart kappa=-50 must be rejected (design genuinely built \
         at κ=-50, so validate_chart_points must fire); got Ok — κ was silently re-derived"
    );
}
