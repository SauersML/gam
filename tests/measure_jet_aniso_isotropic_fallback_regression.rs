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
//! the geometry. These tests drive the real helix that triggered the
//! regression through `fit_from_formula` at both 3-D and 5-D and assert (a) the
//! fit returns without error and (b) the helix signal is recovered (training R²
//! well above the noise floor).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 2500;

/// A helix arc embedded in `R^d` (`d ≥ 3`) with coordinate noise, plus a smooth
/// response `y = sin(4t)` on the arc-length parameter `t`. The first three axes
/// carry the planar circle + linear rise; any extra axes carry lower-frequency
/// trig of `t` so the arc genuinely bends through every dimension. A
/// geometry-blind isotropic kernel already fits the arc well — the anisotropy
/// candidate has little to gain and tends to worsen the certified score, which
/// is exactly the path the regression lived on.
fn helix_dataset(d: usize, seed: u64) -> gam::data::EncodedDataset {
    assert!(d >= 3, "helix embedding needs at least 3 axes");
    let mut rng = StdRng::seed_from_u64(seed);
    let t_dist = Uniform::new(0.0_f64, 1.0_f64).expect("uniform");
    let coord_noise = Normal::new(0.0_f64, 0.05_f64).expect("normal");
    let y_noise = Normal::new(0.0_f64, 0.05_f64).expect("normal");

    let mut headers: Vec<String> = (0..d).map(|k| format!("x{k}")).collect();
    headers.push("v".to_string());

    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for _ in 0..N {
        let t = t_dist.sample(&mut rng);
        let mut fields: Vec<String> = Vec::with_capacity(d + 1);
        for k in 0..d {
            let coord = match k {
                0 => (6.0 * t).cos() * 0.6,
                1 => (6.0 * t).sin() * 0.6,
                2 => t * 1.6 - 0.8,
                // Extra axes: distinct lower-frequency trig of t so the arc
                // bends through them too (keeps the embedding genuinely d-D).
                _ => {
                    let freq = 2.0 + 0.5 * (k as f64);
                    (freq * t).sin() * 0.4
                }
            };
            fields.push((coord + coord_noise.sample(&mut rng)).to_string());
        }
        let signal = (4.0 * t).sin();
        fields.push((signal + y_noise.sample(&mut rng)).to_string());
        rows.push(StringRecord::from(fields));
    }

    encode_recordswith_inferred_schema(headers, rows).expect("encode helix dataset")
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

/// Fit a measure-jet smooth over the first `d` ambient columns and assert the
/// helix signal is recovered without the spatial-κ gate spuriously aborting.
fn assert_mjs_converges_and_recovers(d: usize, seed: u64) {
    let data = helix_dataset(d, seed);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let axes: Vec<String> = (0..d).map(|k| format!("x{k}")).collect();
    let formula = format!("v ~ mjs({}, centers=32)", axes.join(","));

    // Before the fix this returned `Err(RemlOptimizationFailed("spatial kappa
    // optimization made REML score worse (... -> ...)"))`.
    let result = fit_from_formula(&formula, &data, &cfg).unwrap_or_else(|e| {
        panic!("{d}-D measure-jet fit must converge, not abort on a spurious score-worse gate: {e}")
    });
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard fit result");
    };

    let r2 = training_r2(&fit, &data);
    assert!(
        r2 > 0.7,
        "the {d}-D helix signal must be recovered (training R² = {r2:.4}, expected > 0.7)"
    );
}

#[test]
fn measure_jet_3d_converges_when_aniso_loses_to_isotropic() {
    assert_mjs_converges_and_recovers(3, 0);
}

#[test]
fn measure_jet_5d_converges_when_aniso_loses_to_isotropic() {
    assert_mjs_converges_and_recovers(5, 1);
}
