//! #4024: `knot_placement=quantile` must certify the designs #3078 found it
//! refused.
//!
//! PR #3078 made quantile the default placement for measurement and saw five
//! fits that certify under uniform knots end in a `DominatedCertifiedPlateau`
//! refusal: the convex and concave shape-constrained parabolas, a global smooth
//! plus a sum-to-zero factor smooth, and the logistic `s(x)` of the binomial
//! point-invariance test (the fifth, `stiefel(k=1)`, is a response-geometry fit
//! of the same `s(x)`). Each fixture here is the Rust image of one of those
//! designs with the opt-in placement spelled out, and each must mint a fit:
//! a refusal on a design whose uniform-knot fit certifies is an outer-search
//! defect, never an acceptable outcome of a supported option.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Distribution, Normal, Uniform};

fn dataset(headers: &[&str], columns: &[Vec<String>]) -> EncodedDataset {
    let headers: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
    let n = columns[0].len();
    let rows = (0..n)
        .map(|i| StringRecord::from(columns.iter().map(|c| c[i].clone()).collect::<Vec<_>>()))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit(formula: &str, ds: &EncodedDataset, family: &str) -> StandardFitResult {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(formula, ds, &cfg) {
        Ok(FitResult::Standard(s)) => s,
        Ok(_) => panic!("{formula}: expected a standard fit"),
        Err(e) => panic!("#4024: {formula} must certify under quantile knots: {e}"),
    }
}

fn to_strings(v: &[f64]) -> Vec<String> {
    v.iter().map(f64::to_string).collect()
}

/// R² of the fitted linear predictor against the truth at the training rows.
fn r2_vs_truth(fitted: &StandardFitResult, truth: &[f64]) -> f64 {
    let eta = fitted.design.design.matrixvectormultiply(&fitted.fit.beta);
    let n = truth.len() as f64;
    let resid: Vec<f64> = truth.iter().zip(eta.iter()).map(|(t, e)| t - e).collect();
    let var = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / n;
        v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / n
    };
    1.0 - var(&resid) / var(truth)
}

/// The #1380 parabola: `y = ±(x − 0.5)² + 0.05 ε`, `n = 400`, four seeds.
fn shaped_parabola(shape: &str, sign: f64) {
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(4024 + seed);
        let ux = Uniform::new(0.0_f64, 1.0).unwrap();
        let noise = Normal::new(0.0, 0.05).unwrap();
        let mut x: Vec<f64> = (0..400).map(|_| ux.sample(&mut rng)).collect();
        x.sort_by(f64::total_cmp);
        let truth: Vec<f64> = x.iter().map(|&v| sign * (v - 0.5).powi(2)).collect();
        let y: Vec<f64> = truth.iter().map(|t| t + noise.sample(&mut rng)).collect();
        let ds = dataset(&["x", "y"], &[to_strings(&x), to_strings(&y)]);
        let formula = format!("y ~ s(x, shape=\"{shape}\", knot_placement=quantile)");
        let fitted = fit(&formula, &ds, "gaussian");
        let r2 = r2_vs_truth(&fitted, &truth);
        eprintln!("#4024 {shape} seed {seed}: r2 vs truth {r2:.4}");
        // The unconstrained smooth recovers this truth at R² ≈ 0.99; the
        // #1380 collapse to the linear corner scores ≈ 0.
        assert!(r2 > 0.7, "#4024 {shape} seed {seed}: r2 vs truth {r2:.4}");
    }
}

#[test]
fn convex_quantile_smooth_certifies_4024() {
    shaped_parabola("convex", 1.0);
}

#[test]
fn concave_quantile_smooth_certifies_4024() {
    shaped_parabola("concave", -1.0);
}

#[test]
fn global_plus_sz_quantile_smooth_certifies_4024() {
    // `y = 2x + bump_g(x) + 0.2 ε` over three levels, n = 600.
    let mut rng = StdRng::seed_from_u64(40240);
    let ux = Uniform::new(0.0_f64, 1.0).unwrap();
    let ug = Uniform::new(0usize, 3).unwrap();
    let noise = Normal::new(0.0, 0.2).unwrap();
    let n = 600;
    let (mut x, mut g, mut y, mut truth) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let level = ug.sample(&mut rng);
        let bump = match level {
            0 => 0.5,
            1 => -0.5,
            _ => 0.0,
        } * (2.0 * std::f64::consts::PI * xi).sin();
        let t = 2.0 * xi + bump;
        x.push(xi);
        g.push(["A", "B", "C"][level].to_string());
        truth.push(t);
        y.push(t + noise.sample(&mut rng));
    }
    let ds = dataset(&["x", "g", "y"], &[to_strings(&x), g, to_strings(&y)]);
    let fitted = fit(
        "y ~ s(x, knot_placement=quantile) + s(g, x, bs=sz, knot_placement=quantile)",
        &ds,
        "gaussian",
    );
    let r2 = r2_vs_truth(&fitted, &truth);
    eprintln!("#4024 s(x) + sz: r2 vs truth {r2:.4}");
    assert!(
        fitted.fit.beta.iter().all(|b| b.is_finite()),
        "#4024 s(x) + sz: coefficients must be finite"
    );
}

/// The production embedding installs the #784 block quadrature corrector at
/// process start; install it so the binomial fit runs the criterion the
/// Python fit runs.
fn install_production_correctors() {
    drop(gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
        gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
    )));
    drop(gam_problem::rho_posterior::set_rho_posterior_escalator(Box::new(
        gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator,
    )));
}

#[test]
fn binomial_quantile_smooth_certifies_4024() {
    // The point-invariance fixture: logit p = −0.5 + 2x, x ~ U(0, 1), n = 2000.
    install_production_correctors();
    let mut rng = StdRng::seed_from_u64(40241);
    let ux = Uniform::new(0.0_f64, 1.0).unwrap();
    let n = 2000;
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let p = 1.0 / (1.0 + (-(-0.5 + 2.0 * xi)).exp());
            f64::from(u8::from(Bernoulli::new(p).unwrap().sample(&mut rng)))
        })
        .collect();
    let ds = dataset(&["x", "y"], &[to_strings(&x), to_strings(&y)]);
    let fitted = fit("y ~ s(x, knot_placement=quantile)", &ds, "binomial");
    let edf = fitted.fit.edf_total().unwrap_or(f64::NAN);
    eprintln!("#4024 binomial: edf {edf:.3}");
    assert!(fitted.fit.beta.iter().all(|b| b.is_finite()));
}
