//! Hold-out CRPS head-to-head: gam's Gaussian location-scale fit vs the
//! `gamlss::gamlss(family = NO())` reference — the de-facto standard for
//! distributional (mean + log-sigma) regression in R.
//!
//! Every other gamlss-comparison test in this suite compares the *recovered
//! smooth shapes* (mu(x), log-sigma(x)) on the training support. This one does
//! the thing a practitioner actually cares about: it splits the data into a
//! train set (140 rows) and a held-out test set (60 rows), fits BOTH engines on
//! the identical 140 training rows, predicts each engine's full predictive
//! distribution (mu, sigma) on the identical 60 test rows, and scores those
//! predictive distributions with the *same* proper scoring rule — the
//! Continuous Ranked Probability Score (CRPS) for a Gaussian predictive law.
//!
//! Both engines are scored against the same held-out y with the same closed-form
//! Gaussian CRPS:
//!     CRPS(N(mu, sigma), y) = sigma * [ w*(2*Phi(w) - 1) + 2*phi(w) - 1/sqrt(pi) ]
//! with w = (y - mu)/sigma. To remove any chance of a formula mismatch biasing
//! the comparison, gamlss's CRPS is computed in R with the `scoringrules`
//! package (`crps_norm`) and gam's CRPS is computed in Python with
//! `properscoring.crps_gaussian` — two independent third-party implementations
//! of the exact same metric. Comparing element-wise:
//!   * Pearson r between the two engines' per-observation CRPS vectors must be
//!     >= 0.98 — both engines must agree on WHICH held-out points are hard
//!     (high CRPS) vs easy (low CRPS); this is a rank-ordering / calibration
//!     agreement check that is insensitive to a uniform scale offset.
//!   * gam's mean CRPS must be <= gamlss's mean CRPS * 1.05 — gam must match or
//!     beat the standard. The 5% slack is principled, not a weakening: a
//!     well-regularized GAM can trade a little hold-out bias for lower variance
//!     and integrate to a slightly different (often lower) CRPS; we only forbid
//!     gam being materially WORSE than the gamlss reference.
//!   * The held-out log-sigma RMSE (rel_l2) between engines must be < 0.15 — the
//!     scale is a second-moment quantity estimated from squared residuals and is
//!     genuinely harder than the mean, hence a looser bound than the mean's.
//!
//! gam-side API pinned by reading the source (mirrors
//! tests/quality_vs_gamlss_gaussian_location_scale.rs):
//!   * `fit_from_formula(..., FitConfig{ noise_formula: Some(...), .. })` routes
//!     to `FitResult::GaussianLocationScale`; this in-Rust path does NOT rescale
//!     y, so mu/sigma are already in raw response units.
//!   * sigma = LOGB_SIGMA_FLOOR + exp(eta_scale), LOGB_SIGMA_FLOOR = 0.01
//!     (`families::sigma_link`); location block = BlockRole::Location, log-sigma
//!     block = BlockRole::Scale.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python, run_r};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use ndarray::Array2;

/// gam's location-scale noise link floor: sigma = 0.01 + exp(eta_scale).
/// Mirrors `families::sigma_link::LOGB_SIGMA_FLOOR` (and mgcv `gaulss(b=0.01)`).
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn gam_gaussian_location_scale_crps_matches_gamlss() {
    init_parallelism();

    // ---- synthetic bivariate heteroscedastic Gaussian (seed=9999) ----------
    // n=200; x, z ~ Uniform(0,1) independent; y ~ N(sin(2*pi*x) + 0.5*z, s^2)
    // with s = 0.12 + 0.18*|x - 0.5|. A deterministic seeded LCG draws the
    // uniforms and the Box-Muller normals so the EXACT same (x, z, y) rows are
    // reproducible in pure Rust and handed verbatim (via CSV) to both engines.
    let n = 200usize;
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut state: u64 = 9999;
    let mut next_unit = || -> f64 {
        // Numerical Recipes LCG; high bits give a uniform in [0,1).
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    let zc: Vec<f64> = (0..n).map(|_| next_unit()).collect();

    // Box-Muller standard normals from the same continuing LCG stream.
    let mut eps: Vec<f64> = Vec::with_capacity(n);
    while eps.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        eps.push(r * (two_pi * u2).cos());
        if eps.len() < n {
            eps.push(r * (two_pi * u2).sin());
        }
    }

    let mu_true = |xi: f64, zi: f64| (two_pi * xi).sin() + 0.5 * zi;
    let sigma_true = |xi: f64| 0.12 + 0.18 * (xi - 0.5).abs();
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x[i], zc[i]) + sigma_true(x[i]) * eps[i])
        .collect();

    // ---- train / test split: first 140 = train, last 60 = test ------------
    // The split is on row index of the seeded stream, identical for both
    // engines (both receive the same per-row train flag below).
    let n_train = 140usize;
    let n_test = n - n_train; // 60
    let x_train = &x[..n_train];
    let z_train = &zc[..n_train];
    let y_train = &y[..n_train];
    let x_test = &x[n_train..];
    let z_test = &zc[n_train..];
    let y_test = &y[n_train..];

    // ---- fit gam on the 140 TRAINING rows ----------------------------------
    // mu ~ s(x, k=7) + s(z, k=5); log-sigma ~ 1 + s(x, k=7).
    let headers: Vec<String> = vec!["x".to_string(), "z".to_string(), "y".to_string()];
    let train_rows: Vec<csv::StringRecord> = (0..n_train)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x_train[i]),
                format!("{:.17e}", z_train[i]),
                format!("{:.17e}", y_train[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, train_rows)
        .expect("encode training data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, k=7)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=7) + s(z, k=5)", &ds, &cfg)
        .expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    // ---- predict gam's (mu, sigma) on the 60 held-out TEST rows ------------
    // Rebuild the frozen mean / log-sigma designs at the test (x, z) and apply
    // each block's coefficients. mu = X_mean*beta_location;
    // sigma = LOGB_SIGMA_FLOOR + exp(X_scale*beta_scale).
    let mut test_grid = Array2::<f64>::zeros((n_test, ncols));
    for i in 0..n_test {
        test_grid[[i, x_idx]] = x_test[i];
        test_grid[[i, z_idx]] = z_test[i];
    }
    let mean_design = build_term_collection_design(test_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at test points");
    let scale_design = build_term_collection_design(test_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at test points");

    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_location).to_vec();
    let gam_eta_sigma: Vec<f64> = scale_design.design.apply(&beta_scale).to_vec();
    let gam_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| LOGB_SIGMA_FLOOR + e.exp())
        .collect();
    let gam_log_sigma: Vec<f64> = gam_sigma.iter().map(|&s| s.ln()).collect();
    assert_eq!(gam_mu.len(), n_test);
    assert_eq!(gam_sigma.len(), n_test);

    // ---- fit gamlss on the SAME 140 train rows, predict on the SAME 60 test
    // rows, and score with scoringrules::crps_norm in R. The full (x, z, y) and
    // a per-row `train` flag are sent so gamlss subsets to exactly the rows gam
    // trained on; the model is `y ~ pb(x) + pb(z)`, sigma.formula = ~ pb(x).
    let train_flag: Vec<f64> = (0..n).map(|i| if i < n_train { 1.0 } else { 0.0 }).collect();

    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &zc),
            Column::new("y", &y),
            Column::new("train", &train_flag),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        suppressPackageStartupMessages(library(scoringrules))
        tr <- df[df$train > 0.5, ]
        te <- df[df$train < 0.5, ]
        m <- gamlss(y ~ pb(x) + pb(z), sigma.formula = ~ pb(x), family = NO(),
                    data = tr, control = gamlss.control(trace = FALSE))
        nd <- data.frame(x = te$x, z = te$z)
        mu <- as.numeric(predict(m, what = "mu", newdata = nd, type = "response"))
        sigma <- as.numeric(predict(m, what = "sigma", newdata = nd, type = "response"))
        # scoringrules::crps_norm: per-observation CRPS of N(mu, sigma) at y.
        crps <- as.numeric(crps_norm(te$y, mean = mu, sd = sigma))
        emit("mu", mu)
        emit("sigma", sigma)
        emit("crps", crps)
        "#,
    );
    let gamlss_mu = r.vector("mu");
    let gamlss_sigma = r.vector("sigma");
    let gamlss_crps = r.vector("crps");
    assert_eq!(gamlss_mu.len(), n_test, "gamlss mu test length mismatch");
    assert_eq!(gamlss_sigma.len(), n_test, "gamlss sigma test length mismatch");
    assert_eq!(gamlss_crps.len(), n_test, "gamlss crps test length mismatch");
    let gamlss_log_sigma: Vec<f64> = gamlss_sigma.iter().map(|&s| s.ln()).collect();

    // ---- score gam's predictive distribution with the SAME proper scoring
    // rule, via an independent third-party implementation
    // (properscoring.crps_gaussian) on the identical held-out y. gam's predicted
    // (mu, sigma) are passed in; this isolates the scoring library from gam's
    // internals and from R's scoringrules so any CRPS-formula bias cancels.
    let py = run_python(
        &[
            Column::new("y", y_test),
            Column::new("mu", &gam_mu),
            Column::new("sigma", &gam_sigma),
        ],
        r#"
        import properscoring as ps
        y = np.asarray(df["y"], dtype=float)
        mu = np.asarray(df["mu"], dtype=float)
        sigma = np.asarray(df["sigma"], dtype=float)
        crps = ps.crps_gaussian(y, mu=mu, sig=sigma)
        emit("crps", crps)
        "#,
    );
    let gam_crps = py.vector("crps");
    assert_eq!(gam_crps.len(), n_test, "gam crps test length mismatch");

    // ---- compare the two engines element-wise on the held-out set ----------
    let crps_corr = pearson(gam_crps, gamlss_crps);
    let gam_mean_crps: f64 = gam_crps.iter().sum::<f64>() / n_test as f64;
    let gamlss_mean_crps: f64 = gamlss_crps.iter().sum::<f64>() / n_test as f64;
    let crps_ratio = gam_mean_crps / gamlss_mean_crps;
    let rel_log_sigma = relative_l2(&gam_log_sigma, &gamlss_log_sigma);

    eprintln!(
        "gaussian loc-scale CRPS vs gamlss NO(): n_train={n_train} n_test={n_test} \
         gam_mean_crps={gam_mean_crps:.5} gamlss_mean_crps={gamlss_mean_crps:.5} \
         crps_ratio={crps_ratio:.4} crps_pearson={crps_corr:.5} \
         rel_l2(log sigma)={rel_log_sigma:.5}"
    );

    // Both engines fit the same Gaussian location-scale likelihood on the same
    // 140 rows and are scored by the same CRPS on the same 60 held-out rows, so
    // they must agree on which observations are hard vs easy: a per-observation
    // CRPS Pearson r >= 0.98 is a tight calibration-agreement bound (it is the
    // signal that both engines recovered essentially the same predictive law).
    assert!(
        crps_corr >= 0.98,
        "per-observation hold-out CRPS disagrees with gamlss: pearson={crps_corr:.5}"
    );
    // gam must MATCH OR BEAT the gamlss standard on integrated hold-out CRPS;
    // the 5% slack only forbids gam being materially worse, it does not let gam
    // off the hook for a real regression.
    assert!(
        crps_ratio <= 1.05,
        "gam hold-out mean CRPS materially worse than gamlss: ratio={crps_ratio:.4} \
         (gam={gam_mean_crps:.5}, gamlss={gamlss_mean_crps:.5})"
    );
    // The scale (log-sigma) is a second-moment quantity estimated from squared
    // residuals, genuinely harder than the mean, hence the looser 0.15 bound on
    // the held-out log-sigma agreement.
    assert!(
        rel_log_sigma < 0.15,
        "held-out log-sigma diverges from gamlss: rel_l2(log sigma)={rel_log_sigma:.5}"
    );
}
