//! Periodic 1D B-spline with non-default degree. Default is cubic
//! (degree=3); verify lower (linear=1, quadratic=2) and higher
//! (quintic=5) all fit, and that each recovers the truth to within its own
//! Bayesian band: the across-the-function coverage statistic of the fit's
//! `Vb` against the known truth must not exceed its derived bound (see
//! [`across_function_coverage`]). Every degree here has k = degree + 10 ≥ 11
//! basis functions on 200 distinct points, so every case is well-posed and a
//! refusal is a failure.

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

const TAU: f64 = std::f64::consts::TAU;

fn make_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| theta.cos() + 0.3 * (2.0 * theta).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
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

/// Fit one degree and return `(rmse, Q, bound)` on the probe grid.
fn try_fit(degree: usize, alpha: f64) -> Result<(f64, f64, f64), String> {
    let data = make_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!(
        "y ~ s(t, periodic=true, period=6.283185307179586, k={}, degree={degree})",
        degree + 10,
    );
    let result = fit_from_formula(&formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    let probes: Vec<f64> = (0..50).map(|i| TAU * (i as f64) / 49.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err("non-finite".to_string());
    }
    let truth: Vec<f64> = probes
        .iter()
        .map(|t| t.cos() + 0.3 * (2.0 * t).sin())
        .collect();
    let err: Vec<f64> = pred.iter().zip(truth.iter()).map(|(p, t)| p - t).collect();
    let rmse = (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64).sqrt();
    let cov = fit
        .fit
        .beta_covariance()
        .ok_or_else(|| "fit carries no Bayesian covariance Vb".to_string())?;
    let x_probe = design.design.to_dense();
    let (q, bound) = across_function_coverage(&x_probe, cov, &err, alpha)?;
    eprintln!("[per-deg{degree}] rmse={rmse:.4} Q={q:.3} (E=1, bound={bound:.3})");
    Ok((rmse, q, bound))
}

#[test]
fn periodic_1d_degree_sweep() {
    init_parallelism();
    let degrees = [1usize, 2, 3, 4, 5];
    let alpha = FAMILY_ALPHA / degrees.len() as f64;
    let mut failures = Vec::new();
    for degree in degrees {
        match try_fit(degree, alpha) {
            Ok((rmse, q, bound)) => {
                if q > bound {
                    failures.push(format!(
                        "degree={degree}: coverage statistic Q={q:.3} > bound {bound:.3} \
                         (alpha={alpha:.4}); rmse={rmse:.4}"
                    ));
                }
            }
            Err(e) => failures.push(format!("degree={degree}: well-posed fit refused: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "periodic degree sweep failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn periodic_1d_degree_0_rejected_cleanly() {
    init_parallelism();
    let data = make_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let r = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586, k=12, degree=0)",
        &data,
        &cfg,
    );
    let err = match r {
        Ok(_) => panic!("periodic B-spline degree=0 must be rejected"),
        Err(e) => e,
    };
    // The refusal must name the offending option and value ("degree=0 requests
    // a piecewise-constant spline ..."). A bare `k` disjunct matched almost
    // any error text, so an unrelated failure passed.
    let msg = err.to_string();
    assert!(
        msg.contains("degree=0"),
        "degree=0 rejection must name degree=0: {err}",
    );
    eprintln!("[per-deg0] rejected: {err}");
}
