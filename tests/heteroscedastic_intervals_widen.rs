//! FAILING TEST (potentially) — ticket: for genuinely heteroscedastic data
//! with σ(x) growing across the domain, the response-scale uncertainty
//! should track σ(x) — at minimum the SE at the high-noise side must be
//! noticeably larger than the SE at the low-noise side.
//!
//! Standard Gaussian-identity fits assume constant scale, so this test
//! measures the *posterior* uncertainty on the conditional mean — that
//! should still adapt somewhat to local data density / leverage. If the
//! conditional-mean SE is flat (no variation across x) on data that visibly
//! demands variable scale, the uncertainty quantification is mis-calibrated
//! and the user should be steered toward a location-scale model.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn smooth_fit_se_widens_in_high_noise_region() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(53);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let n = 300usize;
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // True mean = sin(πx); noise σ(x) = 0.03 + 0.45·x  (15× wider at x=1 vs x=0)
    let y: Vec<f64> = x
        .iter()
        .map(|t| {
            let sigma = 0.03 + 0.45 * t;
            let noise = Normal::new(0.0, sigma).expect("normal");
            (std::f64::consts::PI * t).sin() + noise.sample(&mut rng)
        })
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!("expected standard fit") };

    // Probe SE at low-noise (x≈0.1) and high-noise (x≈0.9) regions.
    let probe = [0.10_f64, 0.90_f64];
    let mut new_data = Array2::<f64>::zeros((2, 2));
    for (i, &t) in probe.iter().enumerate() {
        new_data[[i, 0]] = t;
        new_data[[i, 1]] = 0.0;
    }
    let test_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("conditional covariance must be present");

    // x_i^T Σ x_i
    let mut x_dense = Array2::<f64>::zeros((2, fit.fit.beta.len()));
    for i in 0..2 {
        let mut e = Array1::<f64>::zeros(2);
        e[i] = 1.0;
        let row = test_design.design.apply_transpose(&e);
        for j in 0..row.len() {
            x_dense[[i, j]] = row[j];
        }
    }
    let se_low = {
        let xi = x_dense.row(0).to_owned();
        let cxi: Array1<f64> = cov.dot(&xi);
        xi.iter().zip(cxi.iter()).map(|(a, b)| a * b).sum::<f64>().max(0.0).sqrt()
    };
    let se_high = {
        let xi = x_dense.row(1).to_owned();
        let cxi: Array1<f64> = cov.dot(&xi);
        xi.iter().zip(cxi.iter()).map(|(a, b)| a * b).sum::<f64>().max(0.0).sqrt()
    };
    eprintln!("[hetero-se] SE@x=0.1 = {se_low:.4}  SE@x=0.9 = {se_high:.4}  ratio = {:.2}", se_high / se_low.max(1e-12));
    // Truth σ ratio is (0.03 + 0.45)/(0.03 + 0.045) = 6.4×. Conditional-mean SE
    // can't see σ(x) directly (Gaussian-identity assumes constant σ̂), but it
    // should still adapt via local leverage. Require at least 1.5×.
    assert!(
        se_high >= 1.5 * se_low,
        "Gaussian-identity conditional-mean SE is essentially flat: SE@hi/SE@lo = {:.2} (expected ≥ 1.5 on data with σ(x) ratio 6.4×) — either steer to a location-scale model or document this as a known calibration limit",
        se_high / se_low.max(1e-12),
    );
}
