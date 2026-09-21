//! Sphere fit with each supported `penalty_order` (m=1..4).
//! Each sweep case must recover the truth to within its own Bayesian band:
//! the across-the-function coverage statistic of the fit's `Vb` against the
//! known truth must not exceed its derived bound (see
//! [`across_function_coverage`]).

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

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.6 * lat.to_radians().sin()
            + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
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

/// Fit `formula` and return `(rmse, min, max, Q, bound)` on the probe grid,
/// with `bound` at level `alpha`.
fn run(formula: &str, alpha: f64) -> Result<(f64, f64, f64, f64, f64), String> {
    let data = make_dataset(400);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..15 {
            let lon = -175.0 + 350.0 * (j as f64) / 14.0;
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
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err("non-finite: range".to_string());
    }
    let truth: Vec<f64> = pts
        .iter()
        .map(|(lat, lon)| {
            0.5 + 0.6 * lat.to_radians().sin()
                + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
        })
        .collect();
    let err: Vec<f64> = pred.iter().zip(truth.iter()).map(|(p, t)| p - t).collect();
    let rmse = (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64).sqrt();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cov = fit
        .fit
        .beta_covariance()
        .ok_or_else(|| "fit carries no Bayesian covariance Vb".to_string())?;
    let x_probe = design.design.to_dense();
    let (q, bound) = across_function_coverage(&x_probe, cov, &err, alpha)?;
    eprintln!(
        "[m-sweep] `{formula}` rmse={rmse:.4} Q={q:.3} (E=1, bound={bound:.3}) \
         range=[{mn:.3}, {mx:.3}]"
    );
    Ok((rmse, mn, mx, q, bound))
}

#[test]
fn sphere_wahba_penalty_order_sweep_low_orders() {
    init_parallelism();
    // Wahba m=1, 2, 3 use closed-form kernels that fit a smooth truth
    // to RMSE ≲ 0.02 with σ=0.05 noise. m=4 (rarely used in practice)
    // has a numerical conditioning issue in the current implementation
    // where REML chooses an extremely large λ and the smooth contribution
    // collapses to zero — see the dedicated documented test below.
    //
    // m=1 carries an explicit `lmax=`. The untruncated Sobolev `K_1` is
    // log-singular at coincidence, so it has no Gram diagonal and the basis
    // builder refuses the family (#2475); `lmax=` is the shipped way to state
    // the spectral resolution a finite m=1 diagonal implies.
    let orders = [1usize, 2, 3];
    let alpha = FAMILY_ALPHA / orders.len() as f64;
    let mut failures = Vec::new();
    for m in orders {
        let formula = if m == 1 {
            "y ~ sphere(lat, lon, k=30, penalty_order=1, lmax=200)".to_string()
        } else {
            format!("y ~ sphere(lat, lon, k=30, penalty_order={m})")
        };
        match run(&formula, alpha) {
            Ok((rmse, mn, mx, q, bound)) => {
                if q > bound {
                    failures.push(format!(
                        "m={m}: coverage statistic Q={q:.3} > bound {bound:.3} \
                         (alpha={alpha:.4}); rmse={rmse:.4} range=[{mn:.3}, {mx:.3}]"
                    ));
                }
            }
            Err(e) => failures.push(format!("m={m}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "wahba m sweep failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn sphere_wahba_m4_must_fit_smooth_truth() {
    // BUG TICKET: Wahba m=4 collapses the fit to a near-constant on a
    // smooth low-degree truth. The closed-form q4 polynomial in
    // `wahba_sphere_kernel_from_cos` agrees with its SIMD sibling and
    // the Gram is PSD, so the issue is not a SIMD/scalar mismatch — it
    // looks like a constant-offset / normalization error in the m=4
    // kernel form that pushes REML to a degenerate λ.
    //
    // Observed at HEAD: rmse=0.43, predictions collapse to [0.502, 0.502]
    // (the response mean) — i.e. the smooth contribution is ~0 while the
    // truth peak-to-peak is ~1.4.
    //
    // This test asserts the FIXED quality target. It will fail until
    // someone derives the correct m=4 kernel constants. Don't silence
    // it — that's the whole point of failing here.
    init_parallelism();
    let (rmse, mn, mx, ..) = run("y ~ sphere(lat, lon, k=30, penalty_order=4)", FAMILY_ALPHA)
        .expect("wahba m=4 fit must succeed");
    // The other Wahba orders (m=1, 2, 3) all hit rmse ≤ 0.018 on the
    // same data. Require m=4 to be in the same ballpark — generous 5×
    // budget = 0.10.
    assert!(
        rmse <= 0.10,
        "Wahba m=4 collapsed: rmse={rmse:.4} (budget 0.10), range=[{mn:.3}, {mx:.3}]. \
         m=1,2,3 all fit at rmse ≤ 0.018 — m=4 must reach the same quality. \
         Likely cause: constant-offset / normalization in the closed-form q4 \
         polynomial in wahba_sphere_kernel_from_cos (basis.rs:13685).",
    );
}

#[test]
fn sphere_harmonic_penalty_order_sweep() {
    init_parallelism();
    let orders = [1usize, 2, 3, 4];
    let alpha = FAMILY_ALPHA / orders.len() as f64;
    let mut failures = Vec::new();
    for m in orders {
        let formula = format!("y ~ sphere(lat, lon, method=harmonic, max_degree=4, penalty_order={m})");
        match run(&formula, alpha) {
            Ok((rmse, mn, mx, q, bound)) => {
                if q > bound {
                    failures.push(format!(
                        "m={m}: coverage statistic Q={q:.3} > bound {bound:.3} \
                         (alpha={alpha:.4}); rmse={rmse:.4} range=[{mn:.3}, {mx:.3}]"
                    ));
                }
            }
            Err(e) => failures.push(format!("m={m}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "harmonic m sweep failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn sphere_invalid_penalty_order_rejected_cleanly() {
    init_parallelism();
    let data = make_dataset(100);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for bad_m in [0usize, 5, 10, 99] {
        let formula = format!("y ~ sphere(lat, lon, k=20, penalty_order={bad_m})");
        let r = fit_from_formula(&formula, &data, &cfg);
        match r {
            Ok(_) => panic!("m={bad_m} must be rejected (valid range is 1..=4)"),
            Err(e) => {
                // The refusal must name the option and the offending value:
                // "... penalty_order must be one of 1, 2, 3, 4; got {bad_m}".
                // Any weaker disjunct (e.g. a bare letter) is satisfied by
                // an unrelated error and cannot fail.
                let msg = e.to_string();
                assert!(
                    msg.contains("penalty_order") && msg.contains(&format!("got {bad_m}")),
                    "m={bad_m} reject must name penalty_order and the value {bad_m}; got: {e}",
                );
                eprintln!("[m-sweep] m={bad_m}: clean error: {e}");
            }
        }
    }
}
