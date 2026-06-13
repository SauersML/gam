//! Regression: a multi-dimensional measure-jet (`mjs`) fit must CONVERGE even
//! when the geometry-seeded anisotropy candidate fails to beat the isotropic
//! baseline.
//!
//! The exact-joint spatial-κ optimizer fits an isotropic baseline (`best`),
//! then tries the anisotropy-seeded joint `[ρ, ψ]` candidate. When that
//! candidate worsens the certified profiled REML score, the optimizer
//! correctly keeps the frozen baseline geometry and refits it under the full
//! inference option set as a β/inference harvester. The harvest re-derives a
//! `reml_score` that drifts a few REML units below the certified baseline
//! score because it runs the full-inference path + adaptive spatial overlay
//! rather than the superseded baseline path that produced `best`. The
//! downstream gate `require_successful_spatial_optimization_result` compares
//! the returned `fit_score` against `fit_score(&best.fit)`; before the fix the
//! drift spuriously read as "spatial kappa optimization made REML score worse"
//! and aborted the WHOLE fit with `RemlOptimizationFailed`.
//!
//! The fix stamps the certified baseline score onto the harvested fit (exactly
//! as the optimized branch already stamps its certified `joint_final_value`),
//! so the returned score is consistent with the gate decision that selected
//! the geometry. This test drives the real 3-D helix that triggered the
//! regression through `fit_from_formula` and asserts (a) the fit returns
//! without error and (b) the helix signal is recovered (training R² well above
//! the noise floor).

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 2500;
const D: usize = 3;

/// One helix arc embedded in R³ with coordinate noise, plus a smooth response
/// `y = sin(4t)` on the arc-length parameter `t`. The arc bends through all
/// three axes (cos/sin in the plane, a linear third axis), so a geometry-blind
/// isotropic kernel already fits it well — the anisotropy candidate has little
/// to gain and tends to worsen the certified score, which is exactly the path
/// the regression lived on.
fn helix_dataset(seed: u64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let t_dist = Uniform::new(0.0_f64, 1.0_f64).expect("uniform");
    let coord_noise = Normal::new(0.0_f64, 0.05_f64).expect("normal");
    let y_noise = Normal::new(0.0_f64, 0.05_f64).expect("normal");

    let headers: Vec<String> = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "v".to_string(),
    ];

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    let mut truth: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        let t = t_dist.sample(&mut rng);
        let a = (6.0 * t).cos() * 0.6 + coord_noise.sample(&mut rng);
        let b = (6.0 * t).sin() * 0.6 + coord_noise.sample(&mut rng);
        let c = t * 1.6 - 0.8 + coord_noise.sample(&mut rng);
        let signal = (4.0 * t).sin();
        let v = signal + y_noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            c.to_string(),
            v.to_string(),
        ]));
        truth.push(signal);
    }

    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode helix dataset");
    (data, truth)
}

/// Training-set coefficient of determination of the fitted mean vs the response.
fn training_r2(fit: &gam::StandardFitResult, data: &gam::data::EncodedDataset) -> f64 {
    // Replay the frozen design on the training values and read η = Xβ (identity
    // link ⇒ fitted mean) the same way the web-quality suite does. The encoded
    // `values` matrix already carries the same column layout the frozen spec
    // was built against, so it can be replayed directly.
    let design = build_term_collection_design(data.values.view(), &fit.resolvedspec)
        .expect("replay frozen design on training rows");
    let fitted = design.design.apply(&fit.fit.beta);

    let cmap = data.column_map();
    let vcol = cmap["v"];
    let y = data.values.column(vcol);
    let ybar = y.sum() / y.len() as f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (yi, fi) in y.iter().zip(fitted.iter()) {
        ss_res += (yi - fi) * (yi - fi);
        ss_tot += (yi - ybar) * (yi - ybar);
    }
    1.0 - ss_res / ss_tot
}

#[test]
fn measure_jet_3d_converges_when_aniso_loses_to_isotropic() {
    let (data, _truth) = helix_dataset(0);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // Before the fix this returned `Err(RemlOptimizationFailed("spatial kappa
    // optimization made REML score worse (... -> ...)"))`.
    let result = fit_from_formula("v ~ mjs(a,b,c,centers=32)", &data, &cfg)
        .expect("3-D measure-jet fit must converge, not abort on a spurious score-worse gate");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit result");
    };

    let r2 = training_r2(&fit, &data);
    assert!(
        r2 > 0.7,
        "the helix signal must be recovered (training R² = {r2:.4}, expected > 0.7)"
    );
}
