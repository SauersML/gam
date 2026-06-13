//! #944 end-to-end: curvature as an estimand — κ̂ + profile CI + κ=0 LR test
//! from a real constant-curvature (`M_κ`) fit.
//!
//! This is the inferential payoff of the whole #944 program: not "we chose
//! hyperbolic space" but "κ̂ = … (95% CI …), flat rejected at p = …". Data are
//! GENERATED on a known `ConstantCurvature` geometry (self-constructed truth —
//! never another tool's output); a `curv(x1, x2)` smooth is fitted with κ as an
//! outer ψ-coordinate; and `curvature_inference_forspec` reports κ̂, its
//! profile-likelihood CI, and the interior-point flatness test built on the
//! REAL profiled REML criterion `V_p(κ)`.
//!
//! Three arms, one fit each (bounded for CI cost):
//!   * spherical truth  (κ⋆ = +2)  ⇒ κ̂ > 0, flatness rejected, verdict ≠ Hyperbolic
//!   * flat truth       (κ⋆ =  0)  ⇒ flatness NOT rejected, CI straddles 0 (verdict Flat)
//!   * hyperbolic truth (κ⋆ = −2)  ⇒ κ̂ < 0, flatness rejected, verdict ≠ Spherical
//!
//! The assertions are truth-recovery + correct-size, not tight coverage (which
//! needs many replicates); sign-recovery and the flatness direction are the
//! issue's headline claims and are the stable single-dataset statements.

use gam::estimate::FitOptions;
use gam::geometry::constant_curvature::ConstantCurvature;
use gam::geometry::curvature_estimand::CurvatureVerdict;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::{
    CurvatureInference, SpatialLengthScaleOptimizationOptions, curvature_inference_forspec,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use gam::terms::term_builder::build_termspec;
use gam::types::LikelihoodSpec;
use ndarray::{Array1, Array2};

// --- deterministic RNG (splitmix64 → unit / gaussian), no external deps ------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Build a `TermCollectionSpec` for a `curv(...)` formula (FarthestPoint
/// centers, auto length scale, κ seeded at 0 so the optimizer has to move it).
fn termspec_for(formula: &str) -> gam::smooth::TermCollectionSpec {
    let parsed = parse_formula(formula).expect("formula parses");
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let ds = EncodedDataset {
        headers: headers.clone(),
        values: Array2::<f64>::zeros((1, 3)),
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.clone(),
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

/// `n` chart points uniformly in a disk of radius `radius`, with a Gaussian
/// response that is a smooth function of the M_κ geodesic distance to the
/// origin — a signal the constant-curvature kernel can represent, and whose
/// shape genuinely depends on κ⋆ (so curvature is identified).
fn dataset_on_m_kappa(
    n: usize,
    kappa_star: f64,
    radius: f64,
    noise_sd: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let mut st = seed;
    let manifold = ConstantCurvature::new(2, kappa_star);
    let reference = ndarray::array![0.0_f64, 0.0_f64];
    let mut feats = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (x1, x2) = loop {
            let a = 2.0 * next_unit(&mut st) - 1.0;
            let b = 2.0 * next_unit(&mut st) - 1.0;
            if a * a + b * b <= 1.0 {
                break (a * radius, b * radius);
            }
        };
        let pt = ndarray::array![x1, x2];
        let d = manifold
            .distance(pt.view(), reference.view())
            .expect("in-chart geodesic distance");
        // Smooth planted signal of the geodesic distance + noise.
        let mu = 2.0 * (-d).exp() - 1.0;
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        y[i] = mu + noise_sd * next_gauss(&mut st);
    }
    (feats, y)
}

/// Fit `curv(x1, x2)` on data built into a 3-column `[y, x1, x2]` frame with κ
/// optimized as an outer ψ-coordinate, then return the full curvature report.
fn fit_and_infer(feats: &Array2<f64>, y: &Array1<f64>) -> CurvatureInference {
    let n = y.len();
    // The design driver consumes feature columns by index; the spec built for
    // "y ~ curv(x1, x2)" references columns x1, x2 which in the encoded frame
    // are indices 1, 2 — but the spatial fit takes the FEATURE matrix directly,
    // so we hand it the [x1, x2] columns and a spec whose feature_cols are 0, 1.
    // build_termspec resolves x1->1, x2->2 against the 3-col schema; to keep the
    // column indices consistent we pass the full 3-col frame's feature view.
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 0)] = y[i];
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let spec = termspec_for("y ~ curv(x1, x2, centers=10)");

    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 24,
        rel_tol: 1e-5,
        pilot_subsample_threshold: 0,
        ..SpatialLengthScaleOptimizationOptions::default()
    };

    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        frame.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodSpec::gaussian_identity(),
        &options,
        &kappa_options,
    )
    .expect("constant-curvature fit with κ optimization");

    curvature_inference_forspec(
        frame.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &fitted.resolvedspec,
        0,
        LikelihoodSpec::gaussian_identity(),
        &options,
        0.95,
    )
    .expect("curvature inference")
}

#[test]
fn spherical_truth_recovers_positive_kappa_and_rejects_flat() {
    let (feats, y) = dataset_on_m_kappa(600, 2.0, 0.5, 0.05, 0x5151_0001);
    let inf = fit_and_infer(&feats, &y);
    eprintln!(
        "[spherical] κ̂={:.4} CI=[{:.4}, {:.4}] verdict={:?} flat_p={:.4} lr={:.4}",
        inf.kappa_hat, inf.ci.ci_lo, inf.ci.ci_hi, inf.ci.verdict, inf.flatness.p_value, inf.flatness.lr_stat
    );
    assert!(
        inf.kappa_hat > 0.0,
        "spherical truth κ⋆=+2 should give κ̂ > 0, got {}",
        inf.kappa_hat
    );
    assert_ne!(
        inf.ci.verdict,
        CurvatureVerdict::Hyperbolic,
        "spherical truth must not be called hyperbolic"
    );
    assert!(
        inf.flatness.p_value < 0.05,
        "spherical truth should reject flatness (p<0.05), got p={}",
        inf.flatness.p_value
    );
}

#[test]
fn flat_truth_does_not_reject_flatness() {
    let (feats, y) = dataset_on_m_kappa(600, 0.0, 0.5, 0.05, 0x5151_0002);
    let inf = fit_and_infer(&feats, &y);
    eprintln!(
        "[flat] κ̂={:.4} CI=[{:.4}, {:.4}] verdict={:?} flat_p={:.4} lr={:.4}",
        inf.kappa_hat, inf.ci.ci_lo, inf.ci.ci_hi, inf.ci.verdict, inf.flatness.p_value, inf.flatness.lr_stat
    );
    // Correct size: flat data must NOT be spuriously called curved.
    assert!(
        inf.flatness.p_value > 0.05,
        "flat truth κ⋆=0 should NOT reject flatness, got p={}",
        inf.flatness.p_value
    );
    assert_eq!(
        inf.ci.verdict,
        CurvatureVerdict::Flat,
        "flat truth CI must straddle 0 (verdict Flat); CI=[{}, {}]",
        inf.ci.ci_lo,
        inf.ci.ci_hi
    );
}

#[test]
fn hyperbolic_truth_recovers_negative_kappa_and_rejects_flat() {
    let (feats, y) = dataset_on_m_kappa(600, -2.0, 0.5, 0.05, 0x5151_0003);
    let inf = fit_and_infer(&feats, &y);
    eprintln!(
        "[hyperbolic] κ̂={:.4} CI=[{:.4}, {:.4}] verdict={:?} flat_p={:.4} lr={:.4}",
        inf.kappa_hat, inf.ci.ci_lo, inf.ci.ci_hi, inf.ci.verdict, inf.flatness.p_value, inf.flatness.lr_stat
    );
    assert!(
        inf.kappa_hat < 0.0,
        "hyperbolic truth κ⋆=−2 should give κ̂ < 0, got {}",
        inf.kappa_hat
    );
    assert_ne!(
        inf.ci.verdict,
        CurvatureVerdict::Spherical,
        "hyperbolic truth must not be called spherical"
    );
    assert!(
        inf.flatness.p_value < 0.05,
        "hyperbolic truth should reject flatness (p<0.05), got p={}",
        inf.flatness.p_value
    );
}
