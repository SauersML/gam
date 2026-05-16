//! Conditional-mean SE for a Gaussian-identity smooth must rise where the
//! training data is sparser. The SE comes from `σ̂² · (XᵀX + S)⁻¹` so it
//! reflects local Fisher information (column-density of the design), not the
//! noise process itself.
//!
//! Setup: train x ∈ [0,1] but with deliberate density variation — 240 samples
//! drawn from a triangular distribution peaked at x=0 (~6× higher density
//! near 0 than near 1) and uniform low noise. Probe SE at x=0.05
//! (dense, low SE expected) and x=0.95 (sparse, high SE expected). Assert
//! the sparse-region SE is at least 1.3× the dense-region SE. If the SE is
//! ~flat across density, the conditional-covariance pipeline is mis-wired
//! and credible intervals at sparse-data regions will be over-confident.

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
fn smooth_fit_se_widens_in_sparse_density_region() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(53);
    // Triangular density on [0, 1] peaked at 0: sample via inverse CDF of
    // 2(1 - x).  Density ratio at x=0.05 vs x=0.95 is (1-0.05)/(1-0.95)=19×.
    let u01 = Uniform::new(0.0, 1.0).expect("uniform");
    let n = 240usize;
    let noise = Normal::new(0.0, 0.08).expect("normal");
    let mut x: Vec<f64> = (0..n)
        .map(|_| 1.0 - (1.0 - u01.sample(&mut rng) as f64).sqrt())
        .collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
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
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    // Probe SE at dense (x=0.05, near peak of density) and sparse (x=0.95).
    let probe = [0.05_f64, 0.95_f64];
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
        xi.iter()
            .zip(cxi.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .max(0.0)
            .sqrt()
    };
    let se_high = {
        let xi = x_dense.row(1).to_owned();
        let cxi: Array1<f64> = cov.dot(&xi);
        xi.iter()
            .zip(cxi.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .max(0.0)
            .sqrt()
    };
    let ratio = se_high / se_low.max(1e-12);
    eprintln!(
        "[density-se] SE@x=0.05 (dense) = {se_low:.4}  SE@x=0.95 (sparse) = {se_high:.4}  ratio = {ratio:.2}",
    );
    // Density ratio is ~19×; SE ∝ 1/√density would give √19 ≈ 4.4×. Allow
    // generous slack for smoothing + finite p: require ≥ 1.3×. If SE is
    // essentially flat across density, the conditional covariance is
    // mis-using the design and credible intervals are over-confident
    // exactly where the user least wants them to be.
    assert!(
        ratio >= 1.3,
        "Conditional-mean SE doesn't adapt to local design density: \
         SE@x=0.95 / SE@x=0.05 = {ratio:.2} (expected ≥ 1.3 with 19× density ratio). \
         Credible intervals in sparse regions will be over-confident.",
    );
}
