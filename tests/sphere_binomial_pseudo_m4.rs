//! Binomial(logit) + sphere with the high-order Wahba pseudo-spline
//! kernel. The pseudo m=4 case used to collapse the smooth to ~0 for
//! Gaussian REML (Wahba m=4 kernel values are tiny — `K(p,p) ≈ 3e-4`).
//! The REML scale-invariance fix (cycle 45/46) cleared the absolute
//! floor on the penalty-rank tolerance, so the non-Gaussian PIRLS-inner
//! REML loop should also be cured. Verify here.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

fn truth_prob(lat: f64, lon: f64) -> f64 {
    let lat_r = lat.to_radians();
    let lon_r = lon.to_radians();
    let logit = -0.5 + 1.5 * lat_r.sin() + 0.8 * lon_r.cos();
    1.0 / (1.0 + (-logit).exp())
}

fn make_binom_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let p = truth_prob(lat, lon);
        let y = if u01.sample(&mut rng) < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str) -> (f64, f64, f64) {
    let data = make_binom_dataset(800, 17);
    let cfg = FitConfig {
        family: Some("binomial(logit)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut pts = Vec::new();
    for i in 0..10 {
        let lat = -60.0 + 120.0 * (i as f64) / 9.0;
        for j in 0..20 {
            let lon = -160.0 + 320.0 * (j as f64) / 19.0;
            pts.push((lat, lon));
        }
    }
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    let mut var = 0.0_f64;
    for p in &pred {
        var += (p - mean).powi(2);
    }
    let std = (var / pred.len() as f64).sqrt();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[binom-sphere] `{formula}` mean={mean:.3} std={std:.3} range=[{mn:.3},{mx:.3}]");
    (mean, std, mx - mn)
}

#[test]
fn binomial_logit_sphere_pseudo_m4_does_not_collapse() {
    init_parallelism();
    // The original collapse scenario: pseudo-spline m=4 with small kernel
    // values produced a Gram that REML's rank-tolerance falsely truncated.
    // After the fix, the smooth should retain its signal: pred std across
    // a 10×20 grid > 0.1 (truth has eta peak-to-peak ≈ 4).
    let (_, std, range) = fit_predict("y ~ sphere(lat, lon, k=30, m=4, kernel=pseudo)");
    assert!(
        std > 0.1,
        "binomial pseudo m=4 collapsed: pred std={std:.3} (range={range:.3}). \
         The smooth contribution is essentially zero — REML still has a \
         scale-sensitivity bug for the Bernoulli inner loop.",
    );
    assert!(range > 0.5, "pred range {range:.3} too small for truth eta-range ~4");
}

#[test]
fn binomial_logit_sphere_sobolev_m4_does_not_collapse() {
    init_parallelism();
    let (_, std, range) = fit_predict("y ~ sphere(lat, lon, k=30, m=4, kernel=sobolev)");
    assert!(std > 0.1, "binomial sobolev m=4 collapsed: pred std={std:.3}");
    assert!(range > 0.5, "pred range {range:.3} too small");
}

#[test]
fn binomial_logit_sphere_both_kernels_agree_under_reml() {
    // If REML is truly scale-invariant, both kernels should produce
    // similar logit predictions (different λ, but identical smoother).
    init_parallelism();
    let (mean_sob, std_sob, _) =
        fit_predict("y ~ sphere(lat, lon, k=30, m=4, kernel=sobolev)");
    let (mean_pse, std_pse, _) =
        fit_predict("y ~ sphere(lat, lon, k=30, m=4, kernel=pseudo)");
    // Means should match to a couple decimals; stds within ~30% of each
    // other (allow some divergence because Bernoulli REML adds PIRLS
    // inner-loop nonlinearity on top).
    assert!(
        (mean_sob - mean_pse).abs() < 0.2,
        "binomial m=4 fit means diverge: sob={mean_sob:.3} pse={mean_pse:.3}",
    );
    let rel = (std_sob - std_pse).abs() / std_sob.max(std_pse).max(1e-6);
    assert!(
        rel < 0.5,
        "binomial m=4 fit stds diverge: sob={std_sob:.3} pse={std_pse:.3} rel={rel:.3}",
    );
}
