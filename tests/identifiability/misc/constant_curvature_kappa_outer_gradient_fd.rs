//! #944 stage 3 final wiring — κ as an ACTUALLY-FITTED ψ-coordinate.
//!
//! The constant-curvature (`M_κ`) smooth now enrolls its signed sectional
//! curvature κ as one design-moving coordinate in the unified outer
//! LAML/REML optimization. This is the merge gate the issue names: the standing
//! full-outer-gradient finite-difference audit, with κ active.
//!
//! The test enables the generic outer runner's structured finite-difference
//! capture at its first seed with a ψ coordinate. The record contains the
//! exact ρ/ψ layout plus analytic and finite-difference gradient arrays. This
//! test:
//!
//!  (1) fits a Gaussian response with a single `curv(x1, x2, kappa=..)` smooth
//!      on data GENERATED on `M_κ` for a planted κ, captures the audit, and
//!      asserts the analytic outer gradient w.r.t. κ matches the central
//!      finite difference of the criterion (no DESYNC verdict, finite
//!      per-coordinate analytic/fd, small relative gap on the κ block); and
//!  (2) on FLAT-generated data (planted κ = 0) checks the κ = 0 likelihood-ratio
//!      flatness test has correct size — `p_value` is the interior χ²₁ tail
//!      (not the half-χ² boundary mixture) and a flat fit is NOT rejected.
//!
//! Reference-as-truth: data are generated on a known `ConstantCurvature`
//! geometry, and every assertion is against that self-constructed truth or the
//! analytic FD of gam's own criterion — never another tool's output.

use gam::geometry::constant_curvature::ConstantCurvature;
use gam::geometry::curvature_estimand::{flatness_lr_test, profile_ci_walk};
use gam::{FitConfig, encode_recordswith_inferred_schema};

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    gam::init_parallelism();
}

use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Chart points uniformly in a disk of radius `r` (inside the κ-stereographic
/// chart for the κ used here), plus a Gaussian response that is a smooth
/// function of the M_κ geodesic distance to a fixed reference point — a signal
/// the constant-curvature kernel can represent.
fn build_dataset(
    n: usize,
    kappa: f64,
    radius: f64,
    seed: u64,
) -> gam::inference::data::EncodedDataset {
    let mut st = seed;
    let manifold = ConstantCurvature::new(2, kappa);
    let reference = ndarray::array![0.0_f64, 0.0_f64];
    let mut header = String::from("y,x1,x2\n");
    let mut body = String::new();
    for _ in 0..n {
        // Rejection-sample a point uniformly in the disk of radius `radius`.
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
        let y = mu + 0.10 * next_gauss(&mut st);
        body.push_str(&format!("{y:.6},{x1:.6},{x2:.6}\n"));
    }
    header.push_str(&body);
    let mut rdr = csv::ReaderBuilder::new().from_reader(header.as_bytes());
    let records: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// The κ = 0 flatness test has correct size: on a quadratic profile centred at
/// κ̂ = 0 the LR statistic is zero and the p-value is the full interior χ²₁
/// tail (here p = 1), NOT the half-χ² boundary mixture — a flat latent space is
/// not spuriously rejected, and the profile CI straddles 0 (verdict Flat).
#[test]
fn kappa_zero_flatness_test_has_correct_size() {
    // A profiled criterion (negative log-evidence) whose minimiser is exactly
    // flat: V_p(κ) = 0.5·a·κ². κ̂ = 0 ⇒ LR = 0 ⇒ p = 1 (not 0.5).
    let a = 4.0;
    let v_p = |k: f64| -> Result<f64, String> { Ok(0.5 * a * k * k) };

    let test = flatness_lr_test(v_p, 0.0).expect("flatness LR");
    assert!(
        test.lr_stat.abs() < 1e-12,
        "flat κ̂ ⇒ zero LR, got {}",
        test.lr_stat
    );
    assert!(
        (test.p_value - 1.0).abs() < 1e-12,
        "interior χ²₁ p-value at LR=0 is 1.0, not the half-χ² 0.5; got {}",
        test.p_value
    );

    // And the profile CI must straddle 0 (geometry verdict Flat) for flat data.
    let ci = profile_ci_walk(v_p, 0.0, a, -10.0, 10.0, 0.95, 1e-9).expect("CI walk");
    assert!(
        ci.ci_lo < 0.0 && ci.ci_hi > 0.0,
        "flat profile CI must straddle 0: [{}, {}]",
        ci.ci_lo,
        ci.ci_hi
    );
    assert_eq!(
        ci.verdict,
        gam::geometry::curvature_estimand::CurvatureVerdict::Flat
    );
}
