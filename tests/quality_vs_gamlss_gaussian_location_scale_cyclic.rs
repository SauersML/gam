//! End-to-end quality: gam's Gaussian location-scale fit with a *cyclic*
//! (periodic) smooth in BOTH the mean (mu) and the log-scale (sigma) block must
//! match `gamlss` — the mature, standard GAMLSS-style distributional-regression
//! implementation — on the same periodic, heteroscedastic data.
//!
//! Why gamlss, and why this configuration
//! --------------------------------------
//! `gamlss::gamlss(family = NO())` is the reference for distributional
//! regression: it fits separate additive predictors for mu and sigma by
//! penalized likelihood, exactly the quantity gam's Gaussian location-scale
//! path targets. Routing the smooths through mgcv's cyclic cubic basis
//! (`ga(~ s(x, bs = "cc"))`) makes gamlss enforce the SAME wrap-around
//! constraint gam's `s(x, bs='cc')` enforces: the fitted function and its
//! derivatives are continuous across the period boundary. Comparing the two
//! constrained smooth *shapes* (not just in-sample predictions) is what reveals
//! bugs in how gam propagates the cyclic-basis null space and the blockwise
//! penalty into both blocks when mu and sigma are fitted jointly.
//!
//! Data: a fixed-seed periodic, heteroscedastic signal,
//!   x = seq(0, 2*pi, length = 150),  y ~ N(sin(x), (0.15 + 0.1*cos(x))^2),
//! with x circular (0 and 2*pi identified). Both engines receive the IDENTICAL
//! (x, y) and the SAME explicit period [0, 2*pi], so the only thing under test
//! is the fitted cyclic mu- and sigma-shapes.
//!
//! Note on the formula: this is the faithful location-scale + cyclic-both
//! configuration the metric targets. We deliberately do NOT add a `linkwiggle`
//! mean-warp here — gamlss `NO()` has no inverse-link warp, so adding one would
//! make the test measure a quantity the reference cannot express and would
//! corrupt the very comparison the spec asks for.
//!
//! Bound: cyclic smooths add the wrap-around derivative-continuity constraint,
//! so close agreement with gamlss validates that gam's `cc` basis and blockwise
//! penalty correctly carry that structure into BOTH blocks. We assert
//!   relative_l2(mu_gam, mu_gamlss)              < 0.02   (mean on the cycle)
//!   relative_l2(log sigma_gam, log sigma_gamlss) < 0.04   (log-scale on cycle)
//! evaluated on 50 equally-spaced points in [0, 2*pi). The sigma bound is looser
//! because the log-scale block is identified one likelihood-derivative removed
//! from the data (it sees residual second moments, not the response directly),
//! so REML/penalty-selection differences between the two engines legitimately
//! perturb log sigma more than mu — yet 0.04 still pins the *shape* tightly
//! (a real divergence in the cyclic sigma smooth would blow well past it).

use csv::StringRecord;
use gam::families::sigma_link::logb_sigma_from_eta_scalar;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// True mean function on the circle.
fn true_mu(x: f64) -> f64 {
    x.sin()
}

/// True standard deviation function on the circle (strictly positive on [0,2π]).
fn true_sigma(x: f64) -> f64 {
    0.15 + 0.1 * x.cos()
}

/// Deterministic standard-normal draws via Box–Muller from a tiny LCG, so the
/// data handed to gam and to gamlss is bit-identical and reproducible without
/// pulling an RNG-crate dependency that could drift between versions. Seed 123.
fn standard_normals(n: usize, seed: u64) -> Vec<f64> {
    // 64-bit LCG (Numerical Recipes constants).
    let mut state = seed;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 53 bits -> (0,1)
        let bits = state >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out.truncate(n);
    out
}

#[test]
fn gam_cyclic_location_scale_matches_gamlss() {
    init_parallelism();

    // ---- build the synthetic circular, heteroscedastic dataset ------------
    // x = seq(0, 2*pi, length = 150); y ~ N(sin x, (0.15 + 0.1 cos x)^2), seed=123.
    let n = 150usize;
    let period = 2.0 * PI;
    let xs: Vec<f64> = (0..n)
        .map(|i| period * (i as f64) / ((n - 1) as f64))
        .collect();
    let z = standard_normals(n, 123);
    let ys: Vec<f64> = xs
        .iter()
        .zip(z.iter())
        .map(|(&x, &zi)| true_mu(x) + true_sigma(x) * zi)
        .collect();

    // Encode identically for gam (the same numbers go to R below).
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| StringRecord::from(vec![format!("{y:.17e}"), format!("{x:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit with gam: Gaussian location-scale, cyclic smooth in BOTH blocks
    // Pin the period explicitly to [0, 2*pi] so gam's cyclic boundary matches
    // the `knots = list(x = c(0, 2*pi))` we hand mgcv inside gamlss below.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(
            "1 + s(x, bs='cc', period_start=0, period_end=6.283185307179586)".to_string(),
        ),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bs='cc', period_start=0, period_end=6.283185307179586)",
        &ds,
        &cfg,
    )
    .expect("gam cyclic location-scale fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit for a Gaussian noise_formula model");
    };

    // Mean (Location) and log-sigma (Scale) coefficient blocks.
    let beta_mu = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location-scale fit carries a Location (mu) block")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("location-scale fit carries a Scale (log-sigma) block")
        .beta
        .clone();
    // A smooth sigma must materialize a multi-column basis (intercept + cc),
    // otherwise the cyclic structure never reached the scale block.
    assert!(
        beta_noise.len() >= 2,
        "cyclic noise_formula must materialize a multi-coefficient scale basis, got {}",
        beta_noise.len()
    );

    // ---- 50 equally-spaced evaluation points in [0, 2*pi) -----------------
    let m = 50usize;
    let grid_x: Vec<f64> = (0..m).map(|i| period * (i as f64) / (m as f64)).collect();

    // Rebuild the mean and noise designs from the FROZEN resolved specs at the
    // evaluation grid (identity mean link => eta_mu = X_mu·beta_mu; gam's logb
    // sigma link => sigma = 0.01 + exp(X_noise·beta_noise), response_scale = 1
    // on the library fit path). This is the same plug-in path gam's predictor
    // takes; we reconstruct it from the resolved specs so the comparison is on
    // the smooth SHAPE off the training points, not in-sample fitted values.
    let mut eval_grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &gx) in grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let mean_design = build_term_collection_design(eval_grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at eval grid");
    let noise_design = build_term_collection_design(eval_grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at eval grid");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mu.len(),
        "mean design columns ({}) must match mu coefficient count ({})",
        mean_design.design.ncols(),
        beta_mu.len()
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_noise.len(),
        "noise design columns ({}) must match log-sigma coefficient count ({})",
        noise_design.design.ncols(),
        beta_noise.len()
    );

    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_mu).to_vec();
    let eta_noise: Array1<f64> = noise_design.design.apply(&beta_noise);
    let gam_log_sigma: Vec<f64> = eta_noise
        .iter()
        .map(|&e| logb_sigma_from_eta_scalar(e).ln())
        .collect();

    // ---- fit the SAME model with gamlss (the mature reference) ------------
    // family = NO() (normal, identity mu, log sigma); mu and sigma each get a
    // cyclic cubic smooth via mgcv's `ga(~ s(x, bs="cc"))`, with the cyclic
    // knot endpoints pinned to [0, 2*pi] to match gam's explicit period.
    let r = run_r(
        &[Column::new("x", &xs), Column::new("y", &ys)],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        suppressPackageStartupMessages(library(gamlss.add))
        kn <- list(x = c(0, 2*pi))
        m <- gamlss(
            y ~ ga(~ s(x, bs = "cc"), control = list(knots = kn)),
            sigma.formula = ~ ga(~ s(x, bs = "cc"), control = list(knots = kn)),
            family = NO(),
            control = gamlss.control(n.cyc = 200, trace = FALSE)
        )
        xg <- seq(0, 2*pi, length.out = 51)[1:50]
        nd <- data.frame(x = xg)
        mu <- as.numeric(predict(m, what = "mu", newdata = nd, type = "response", data = df))
        ls <- as.numeric(predict(m, what = "sigma", newdata = nd, type = "link", data = df))
        emit("mu", mu)
        emit("log_sigma", ls)
        "#,
    );
    let gamlss_mu = r.vector("mu");
    // gamlss NO() uses a log link for sigma, so the "link"-scale sigma predictor
    // is exactly log(sigma): directly comparable to gam's log-sigma curve.
    let gamlss_log_sigma = r.vector("log_sigma");
    assert_eq!(gamlss_mu.len(), m, "gamlss mu length mismatch");
    assert_eq!(
        gamlss_log_sigma.len(),
        m,
        "gamlss log-sigma length mismatch"
    );

    // ---- compare the constrained cyclic shapes ----------------------------
    let mu_rel = relative_l2(&gam_mu, gamlss_mu);
    let log_sigma_rel = relative_l2(&gam_log_sigma, gamlss_log_sigma);

    eprintln!(
        "cyclic location-scale vs gamlss: n={n} m={m} \
         mu_rel_l2={mu_rel:.4} log_sigma_rel_l2={log_sigma_rel:.4} \
         beta_mu={} beta_sigma={}",
        beta_mu.len(),
        beta_noise.len()
    );

    // Both engines fit a cyclic cubic smooth in each block by penalized
    // likelihood on identical data with the identical period; the constrained
    // mu-shape must essentially coincide.
    assert!(
        mu_rel < 0.02,
        "cyclic mu smooth diverges from gamlss: rel_l2={mu_rel:.4} (bound 0.02)"
    );
    // The log-scale block is identified one derivative removed from the data, so
    // its bound is the looser 0.04 justified above — still tight enough to catch
    // any real failure to carry the cyclic constraint into the sigma block.
    assert!(
        log_sigma_rel < 0.04,
        "cyclic log-sigma smooth diverges from gamlss: rel_l2={log_sigma_rel:.4} (bound 0.04)"
    );
}
