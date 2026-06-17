//! #944 stage-4 validation sims — the deferred quantitative half of "curvature
//! as an estimand": across REPLICATES of data generated on a known `M_κ`, the
//! profile-likelihood machinery must (1) RECOVER the planted curvature with low
//! bias, (2) COVER the true κ⋆ with its 95% profile CI at ≈ the nominal rate,
//! and (3) hold SIZE on the interior κ=0 flatness test (flat data is not
//! spuriously rejected) while having POWER (curved data is rejected). The
//! single-dataset e2e test (`constant_curvature_kappa_inference_e2e`) asserts
//! sign-recovery and flatness DIRECTION; this test adds the replicate-level
//! calibration the issue charter names ("recovery of κ̂, CI coverage, size of
//! the κ=0 test") — the claims that make "κ̂ = … (95% CI …)" a statistically
//! honest sentence rather than a point estimate.
//!
//! Reference-as-truth: every dataset is generated on a known `ConstantCurvature`
//! geometry and every assertion is against that self-constructed truth or the
//! exact χ² calibration of gam's own profiled REML criterion — never another
//! tool's output. Bars are sized to the small replicate count `R` so they catch
//! a genuinely miscalibrated estimator/CI/test without flaking on binomial noise
//! (kept CI-cheap: small n, few centers, a handful of replicates).

use gam::estimate::FitOptions;
use gam::geometry::constant_curvature::ConstantCurvature;
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

/// Build a `TermCollectionSpec` for a `curv(...)` formula. Mirrors the e2e
/// inference test's builder: a 3-column `[y, x1, x2]` continuous schema so the
/// `curv(x1, x2)` term resolves and is not rejected as a constant-column smooth.
fn termspec_for(formula: &str, frame: &Array2<f64>) -> gam::smooth::TermCollectionSpec {
    let parsed = parse_formula(formula).expect("formula parses");
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let ds = EncodedDataset {
        headers: headers.clone(),
        values: frame.clone(),
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
/// response that is a smooth function of the `M_κ` geodesic distance to the
/// origin — a κ⋆-dependent signal the constant-curvature kernel can represent,
/// so curvature is identified.
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
        let mu = 2.0 * (-d).exp() - 1.0;
        feats[(i, 0)] = x1;
        feats[(i, 1)] = x2;
        y[i] = mu + noise_sd * next_gauss(&mut st);
    }
    (feats, y)
}

/// Fit `curv(x1, x2)` with κ optimized as an outer ψ-coordinate, then run the
/// full curvature inference (κ̂ + profile CI + κ=0 LR test) off the REAL
/// profiled REML criterion. CI-cheap: small `centers`, capped outer iters.
fn fit_and_infer(feats: &Array2<f64>, y: &Array1<f64>) -> CurvatureInference {
    let n = y.len();
    let mut frame = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        frame[(i, 0)] = y[i];
        frame[(i, 1)] = feats[(i, 0)];
        frame[(i, 2)] = feats[(i, 1)];
    }
    let spec = termspec_for("y ~ curv(x1, x2, centers=6)", &frame);

    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let options = FitOptions::default();
    let kappa_options = SpatialLengthScaleOptimizationOptions {
        max_outer_iter: 8,
        rel_tol: 1e-4,
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

/// Number of replicate datasets per arm. Small (CI cost) but enough to expose a
/// badly-biased estimator or a grossly mis-covering CI; bars are binomial-aware.
const R: usize = 5;

/// CI COVERAGE + κ̂ RECOVERY on CURVED truth. Across `R` independent M_κ
/// datasets at a planted spherical κ⋆, the 95% profile CI must cover κ⋆ at close
/// to the nominal rate and κ̂ must recover κ⋆ with low bias and correct sign.
#[test]
#[ignore = "MSI-only #944 statistical coverage simulation; runs repeated profiled κ fits"]
fn profile_ci_covers_planted_curvature_across_replicates() {
    gam::init_parallelism();
    let kappa_star = 1.5_f64;
    let mut covered = 0usize;
    let mut sign_correct = 0usize;
    let mut sum_khat = 0.0_f64;
    let mut khats = Vec::with_capacity(R);
    for r in 0..R {
        let seed = 0x5EED_0944_0000_0000 ^ ((r as u64) << 8);
        let (feats, y) = dataset_on_m_kappa(120, kappa_star, 0.6, 0.10, seed);
        let inf = fit_and_infer(&feats, &y);
        let covers = inf.ci.ci_lo <= kappa_star && kappa_star <= inf.ci.ci_hi;
        if covers {
            covered += 1;
        }
        if inf.kappa_hat > 0.0 {
            sign_correct += 1;
        }
        sum_khat += inf.kappa_hat;
        khats.push(inf.kappa_hat);
        eprintln!(
            "[cov κ⋆=+{kappa_star}] r={r} κ̂={:+.3} CI=[{:+.3},{:+.3}] covers={covers}",
            inf.kappa_hat, inf.ci.ci_lo, inf.ci.ci_hi
        );
    }
    let mean_khat = sum_khat / R as f64;
    eprintln!(
        "[cov κ⋆=+{kappa_star}] covered {covered}/{R}  sign_correct {sign_correct}/{R}  \
         mean κ̂={mean_khat:+.3}  κ̂={khats:?}"
    );

    // (1) COVERAGE: a 95% profile CI must cover the truth in a large majority of
    // replicates. At nominal 0.95 over R=5 the expected miss count is ~0.25; a
    // CI that systematically excludes the truth (wrong width or off-center) drops
    // far below. Requiring ≥ 4/5 covers tolerates the small-sample binomial slack
    // (P(Bin(5,0.95) ≤ 3) ≈ 0.001, so a correctly-calibrated CI essentially never
    // fails) while still failing a grossly mis-covering interval.
    assert!(
        covered >= 4,
        "profile CI covered the planted κ⋆=+{kappa_star} in only {covered}/{R} replicates \
         (expected ~5 at nominal 95%)"
    );
    // (2) SIGN RECOVERY: spherical truth ⇒ κ̂ > 0 in a strong majority.
    assert!(
        sign_correct >= 4,
        "κ̂ sign recovered (>0) in only {sign_correct}/{R} replicates for κ⋆=+{kappa_star}"
    );
    // (3) LOW BIAS: the mean estimate tracks the truth within a tolerance honest
    // about the noisy Gaussian signal at n=120 — not railed to a chart bound, not
    // collapsed toward 0.
    assert!(
        (mean_khat - kappa_star).abs() < 1.0,
        "mean κ̂={mean_khat:+.3} too far from planted κ⋆=+{kappa_star} (bias bar 1.0)"
    );
}

/// SIZE of the interior κ=0 flatness test on FLAT truth. Across `R` flat
/// datasets the LR test must NOT spuriously reject (a badly-sized test would
/// reject most), and the profile CI must cover κ=0 (verdict Flat) in a large
/// majority — the controlled-size "is my latent space flat?" claim.
#[test]
#[ignore = "MSI-only #944 statistical size simulation; runs repeated profiled κ fits"]
fn flatness_test_holds_size_across_flat_replicates() {
    gam::init_parallelism();
    let alpha = 0.05_f64;
    let mut rejections = 0usize;
    let mut ci_covers_zero = 0usize;
    let mut pvals = Vec::with_capacity(R);
    for r in 0..R {
        let seed = 0x71A7_0944_0000_0000 ^ ((r as u64) << 8);
        let (feats, y) = dataset_on_m_kappa(120, 0.0, 0.6, 0.10, seed);
        let inf = fit_and_infer(&feats, &y);
        if inf.flatness.p_value < alpha {
            rejections += 1;
        }
        if inf.ci.ci_lo <= 0.0 && 0.0 <= inf.ci.ci_hi {
            ci_covers_zero += 1;
        }
        pvals.push(inf.flatness.p_value);
        eprintln!(
            "[size κ⋆=0] r={r} κ̂={:+.3} p={:.4} CI=[{:+.3},{:+.3}]",
            inf.kappa_hat, inf.flatness.p_value, inf.ci.ci_lo, inf.ci.ci_hi
        );
    }
    eprintln!(
        "[size κ⋆=0] rejected {rejections}/{R} at α={alpha}  CI⊇0 in {ci_covers_zero}/{R}  \
         p-values={pvals:?}"
    );

    // SIZE CONTROL: a level-α interior χ²₁ test on truly flat data rejects ~α of
    // the time. At α=0.05 over R=5 the expected rejection count is ~0.25; a test
    // that over-rejects (wrong reference, e.g. a phantom curvature from the basis,
    // or a mis-scaled LR) would reject many. Allow ≤ 2/5 to absorb the small-R
    // binomial tail (P(Bin(5,0.05) ≥ 3) ≈ 0.001) while still failing a test that
    // rejects flat data routinely.
    assert!(
        rejections <= 2,
        "κ=0 flatness test rejected truly-flat data in {rejections}/{R} replicates at α={alpha} \
         (size-inflated): p-values {pvals:?}"
    );
    // The profile CI must straddle 0 (verdict Flat) for flat data in a strong
    // majority — the CI-side mirror of the size claim.
    assert!(
        ci_covers_zero >= 4,
        "profile CI failed to cover κ=0 on flat data in {}/{R} replicates",
        R - ci_covers_zero
    );
}
