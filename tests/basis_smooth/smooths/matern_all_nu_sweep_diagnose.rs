//! Run matern fit with all supported nu values (1/2, 3/2, 5/2, 7/2, 9/2)
//! on a moderate-frequency smooth truth + noise. Each ν must recover the
//! truth to within its own Bayesian band: the across-the-function coverage
//! statistic of the fit's `Vb` against the known truth must not exceed its
//! derived bound (see [`across_function_coverage`]).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_math::probability::chi_square_quantile;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset(n: usize, sigma: f64, seed: u64) -> (Vec<f64>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let truth: Vec<f64> = x
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + 0.3 * t)
        .collect();
    let y_noisy: Vec<f64> = truth.iter().map(|&v| v + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (x, data)
}

/// Family-wise significance level of this test's coverage gates, split
/// Bonferroni-style over the cases the test sweeps.
const FAMILY_ALPHA: f64 = 0.01;

/// Across-the-function coverage statistic of the fit's Bayesian band
/// (Nychka 1988; Marra & Wood 2012), with its derived upper bound.
///
/// `x_probe` is the probe design, `cov` the fit's conditional Bayesian
/// coefficient covariance `Vb` (`fit.beta_covariance()`) and `err` the
/// errors `f̂(x_i) − f(x_i)` against the known truth. With `C = X_p Vb X_pᵀ`,
/// `s_i = √C_ii` and correlation `R = D⁻¹CD⁻¹`, the statistic is
/// `Q = (1/P) Σ e_i²/s_i²`. Under the model behind the band, `e ~ N(0, C)`, so
/// `E[Q] = 1` and `Var[Q] = 2·tr(R²)/P²`. Smoothing bias adds to `e` while
/// over-smoothing shrinks `s`, so a fit that loses signal drives `Q ≫ 1`.
/// The returned bound is the `1 − alpha` quantile of the moment-matched
/// scaled `χ²` (Satterthwaite): `Q ≈ g·χ²_h` with `g = tr(R²)/P²`, `h = 1/g`.
fn across_function_coverage(
    x_probe: &Array2<f64>,
    cov: &Array2<f64>,
    err: &[f64],
    alpha: f64,
) -> Result<(f64, f64), String> {
    let c = x_probe.dot(cov).dot(&x_probe.t());
    let p = err.len();
    let s: Vec<f64> = (0..p).map(|i| c[[i, i]].sqrt()).collect();
    if let Some(i) = s.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
        return Err(format!("posterior SE at probe {i} is {}", s[i]));
    }
    let q = err
        .iter()
        .zip(s.iter())
        .map(|(e, si)| (e / si).powi(2))
        .sum::<f64>()
        / p as f64;
    let mut tr_r2 = 0.0;
    for i in 0..p {
        for j in 0..p {
            let r = c[[i, j]] / (s[i] * s[j]);
            tr_r2 += r * r;
        }
    }
    let g = tr_r2 / (p * p) as f64;
    let bound = g * chi_square_quantile(1.0 - alpha, 1.0 / g);
    Ok((q, bound))
}

/// Fit one ν and return `(rmse, Q, bound)` on the probe grid.
fn try_fit(nu: &str, alpha: f64) -> (String, Result<(f64, f64, f64), String>) {
    let (_, data) = make_dataset(300, 0.05, 41);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ matern(x, nu={nu})");
    let result = match fit_from_formula(&formula, &data, &cfg) {
        Ok(r) => r,
        Err(e) => return (formula, Err(format!("fit: {e}"))),
    };
    let FitResult::Standard(fit) = result else {
        return (formula, Err("non-standard".into()));
    };
    let xg: Vec<f64> = (0..200).map(|i| 0.005 + 0.99 * i as f64 / 199.0).collect();
    let truth_g: Vec<f64> = xg
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + 0.3 * t)
        .collect();
    let mut m = Array2::<f64>::zeros((xg.len(), 2));
    for (i, &v) in xg.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = match build_term_collection_design(m.view(), &fit.resolvedspec) {
        Ok(d) => d,
        Err(e) => return (formula, Err(format!("design: {e:?}"))),
    };
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return (formula, Err("non-finite predictions".into()));
    }
    let err: Vec<f64> = pred.iter().zip(truth_g.iter()).map(|(p, t)| p - t).collect();
    let rmse = (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64).sqrt();
    let Some(cov) = fit.fit.beta_covariance() else {
        return (formula, Err("fit carries no Bayesian covariance Vb".into()));
    };
    let x_probe = design.design.to_dense();
    let (q, bound) = match across_function_coverage(&x_probe, cov, &err, alpha) {
        Ok(v) => v,
        Err(e) => return (formula, Err(e)),
    };
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[matern-nu] `{formula}` rmse={rmse:.4} Q={q:.3} (E=1, bound={bound:.3}) \
         pred_range=[{mn:.3}, {mx:.3}]"
    );
    (formula, Ok((rmse, q, bound)))
}

#[test]
fn matern_all_nu_values_fit_reasonably() {
    init_parallelism();
    let nus = ["1/2", "3/2", "5/2", "7/2", "9/2"];
    let alpha = FAMILY_ALPHA / nus.len() as f64;
    let mut failures = Vec::new();
    for nu in nus {
        let (formula, r) = try_fit(nu, alpha);
        match r {
            Ok((rmse, q, bound)) => {
                if q > bound {
                    failures.push(format!(
                        "`{formula}` coverage statistic Q={q:.3} > bound {bound:.3} \
                         (alpha={alpha:.4}); rmse={rmse:.4}"
                    ));
                }
            }
            Err(e) => failures.push(format!("`{formula}`: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "matern nu sweep failures:\n  - {}",
        failures.join("\n  - "),
    );
}
