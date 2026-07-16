//! End-to-end quality: gam's 1-D Duchon spline `duchon(x, k=...)` across MORE
//! noise/smoothness/family regimes than the single sine test, each judged by
//! OBJECTIVE TRUTH RECOVERY and held to match-or-beat mgcv on that same metric.
//!
//! Every test here mirrors the philosophy of `quality_vs_mgcv_duchon_smooth.rs`:
//! the pass/fail claim is RMSE of gam's fit against the KNOWN generating
//! function (truth recovery), and we additionally assert gam's truth-recovery
//! RMSE ≤ mgcv's × 1.10 (match-or-beat the mature baseline on accuracy vs the
//! truth — NOT closeness of the two fitted curves). The gam-vs-mgcv relative-L2
//! is printed for context only, never asserted.
//!
//! THE OBJECT UNDER TEST is the DEFAULT `duchon(x, k=...)`: gam's redesigned
//! non-periodic Euclidean structural amplitude/slope/curvature smoother on the
//! cubic (`r³`) polyharmonic basis (affine null space, power s=(d−1)/2=0 in 1D).
//! BASELINE COMPARATOR throughout: `mgcv::gam(y ~ s(x, bs="ds", k=..,
//! m=c(2,0)), method="REML")` — the canonical 1-D cubic-spline-equivalent
//! Duchon smoother (2nd-derivative penalty, degree-1 null space), the mature
//! analogue of gam's default.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

/// Fit gam's default 1-D Duchon at the given `k`, fit mgcv's `bs="ds"`,
/// `m=c(2,0)` analogue on byte-identical data, and assert (1) gam recovers the
/// KNOWN `truth` below `abs_bar` and (2) gam's truth-recovery RMSE match-or-beats
/// mgcv's within 10%. `truth` maps x → the noise-free signal; `sigma` is the
/// Gaussian noise SD used to corrupt it.
fn assert_gaussian_regime(
    label: &str,
    n: usize,
    k: usize,
    sigma: f64,
    seed: u64,
    truth: &dyn Fn(f64) -> f64,
    abs_bar: f64,
) {
    init_parallelism();

    // ---- synthetic data: x on a uniform grid in [0,1], y = truth + N(0,σ) ---
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit with gam: y ~ duchon(x, k), REML (default cubic smoother) ------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ duchon(x, k={k})");
    let result = fit_from_formula(&formula, &ds, &cfg).expect("gam duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian Duchon smooth");
    };

    // ---- dense interior test grid in [0,1] (avoid extrapolation edges) ------
    let m = 201usize;
    let x_test: Vec<f64> = (0..m)
        .map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0))
        .collect();
    let y_truth: Vec<f64> = x_test.iter().map(|&t| truth(t)).collect();

    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv bs="ds", m=c(2,0) (mature reference) --
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));

    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        &format!(
            r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(y ~ s(x, bs = "ds", k = {k}, m = c(2, 0)), data = train, method = "REML")
        emit("fitted", as.numeric(predict(m, newdata = grid)))
        "#
        ),
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), m, "mgcv prediction length mismatch");

    // ---- OBJECTIVE METRIC: truth recovery on the interior grid ------------
    let gam_truth_rmse = rmse(&gam_fitted, &y_truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &y_truth);
    let rel_gam_vs_mgcv = relative_l2(&gam_fitted, mgcv_fitted);

    eprintln!(
        "duchon-regime[{label}]: n={n} k={k} sigma={sigma} \
         gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4} \
         (context: rel_l2(gam,mgcv)={rel_gam_vs_mgcv:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            &format!("quality_vs_mgcv_duchon_regimes::{label}"),
            "truth_rmse",
            gam_truth_rmse,
            "mgcv",
            mgcv_truth_rmse,
        )
        .line()
    );

    // (1) ABSOLUTE recovery bar tied to the regime: gam must reconstruct the
    // signal below abs_bar (chosen per regime from the signal/noise structure),
    // catching a blown-up or non-denoising fit.
    assert!(
        gam_truth_rmse < abs_bar,
        "[{label}] gam Duchon failed to recover the truth: \
         RMSE-vs-truth={gam_truth_rmse:.4} (bar {abs_bar})"
    );

    // (2) MATCH-OR-BEAT mgcv ON ACCURACY VS THE TRUTH (within 10%).
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "[{label}] gam recovers the truth worse than mgcv: \
         gam RMSE-vs-truth={gam_truth_rmse:.4} > 1.10 * mgcv={mgcv_truth_rmse:.4}"
    );
}

/// REGIME A — heavy denoising. A SMOOTH low-frequency truth (a single broad
/// sinusoid, half a period over [0,1]) buried in relatively HIGH noise (σ=0.20).
/// A good smoother should strip the noise and reveal the gentle wave; here the
/// achievable error is dominated by noise, so the recovery bar sits well below
/// the noise SD (the smoother must average the noise down, not track it).
#[test]
fn gam_duchon_1d_heavy_noise_smooth_truth_matches_mgcv() {
    // truth: 0.8 * sin(pi*x) — one smooth hump, range [0, 0.8], well-resolved by k=15.
    let truth = |t: f64| 0.8 * (std::f64::consts::PI * t).sin();
    // σ=0.20 noise on n=300; a denoising smoother should reach RMSE well under
    // the 0.20 noise SD. 0.08 is a principled "clearly averaging the noise down"
    // bar; the real accuracy gate is match-or-beat-mgcv.
    assert_gaussian_regime("heavy_noise_smooth", 300, 15, 0.20, 4242, &truth, 0.08);
}

/// REGIME B — a richer (but still smooth) truth at a LOWER noise and a LARGER k.
/// A two-component wave `sin(2π x) + 0.4 sin(6π x)` (range ≈ [-1.3, 1.3]) at
/// σ=0.07, n=300, fit with k=30. This exercises a different smoothness/k regime
/// than the existing single-frequency test and the heavy-noise regime above.
#[test]
fn gam_duchon_1d_two_component_wave_larger_k_matches_mgcv() {
    let truth = |t: f64| {
        let tp = std::f64::consts::PI * t;
        (2.0 * tp).sin() + 0.4 * (6.0 * tp).sin()
    };
    // With k=30 the basis resolves a 3-period component comfortably; at σ=0.07
    // the recovery should reach a small fraction of the ≈1.3 signal amplitude.
    assert_gaussian_regime("two_component_k30", 300, 30, 0.07, 99, &truth, 0.12);
}

/// REGIME C — NON-GAUSSIAN: Poisson counts from a known smooth log-mean curve,
/// fit with gam's Duchon under the Poisson/log family and benchmarked against
/// mgcv's `s(x, bs="ds")` Poisson fit. The objective metric is recovery of the
/// TRUE MEAN on the RESPONSE scale (the count mean exp(eta_true)), the scale a
/// practitioner cares about; both gam and mgcv are scored by RMSE of their
/// predicted mean against the true mean, and gam must match-or-beat mgcv.
#[test]
fn gam_duchon_1d_poisson_log_mean_recovery_matches_mgcv() {
    init_parallelism();

    // ---- known smooth log-mean: eta_true(x) = 1.0 + sin(2π x); the response-
    // scale mean is mu_true(x) = exp(eta_true), ranging over ~[1.0, 7.4]. counts
    // ~ Poisson(mu_true). Fixed seed => gam and mgcv see identical draws.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(777);
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let eta_true = |t: f64| 1.0 + (2.0 * std::f64::consts::PI * t).sin();
    let count: Vec<f64> = x
        .iter()
        .map(|&t| {
            let lambda = eta_true(t).exp().max(1e-12);
            Poisson::new(lambda)
                .expect("valid Poisson rate")
                .sample(&mut rng)
        })
        .collect();

    let headers = ["x", "count"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(count.iter())
        .map(|(a, c)| csv::StringRecord::from(vec![a.to_string(), c.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit with gam: count ~ duchon(x, k=20), Poisson / log link, REML ----
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("count ~ duchon(x, k=20)", &ds, &cfg).expect("gam poisson duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Poisson Duchon smooth");
    };

    // ---- dense interior test grid; truth on the RESPONSE (mean) scale -------
    let m = 201usize;
    let x_test: Vec<f64> = (0..m)
        .map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0))
        .collect();
    let mu_truth: Vec<f64> = x_test.iter().map(|&t| eta_true(t).exp()).collect();

    // gam: design*beta is the log-mean (eta); exp() gives the predicted count mean.
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Poisson Duchon design at test grid");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model with mgcv bs="ds", Poisson/log (mature reference)
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut count_all = count.clone();
    count_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));

    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("count", &count_all),
            Column::new("is_train", &is_train),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(count ~ s(x, bs = "ds", k = 20, m = c(2, 0)), data = train,
                 family = poisson(link = "log"), method = "REML")
        emit("mu", as.numeric(predict(m, newdata = grid, type = "response")))
        "#,
    );
    let mgcv_mu = r.vector("mu");
    assert_eq!(mgcv_mu.len(), m, "mgcv prediction length mismatch");

    // ---- OBJECTIVE METRIC: response-scale mean recovery on the interior grid
    let gam_mu_rmse = rmse(&gam_mu, &mu_truth);
    let mgcv_mu_rmse = rmse(mgcv_mu, &mu_truth);
    let rel_gam_vs_mgcv = relative_l2(&gam_mu, mgcv_mu);

    // RMS of the true mean: the error a zero predictor would incur; the
    // mean-of-truth predictor would do somewhat better, so the recovery bar
    // sits well below this "do-nothing" reference (~4.0 here).
    let rms_mu = (mu_truth.iter().map(|v| v * v).sum::<f64>() / mu_truth.len() as f64).sqrt();

    eprintln!(
        "duchon-poisson-mean-recovery-1d: n={n} grid={m} \
         gam_mu_rmse={gam_mu_rmse:.4} mgcv_mu_rmse={mgcv_mu_rmse:.4} \
         rms_mu={rms_mu:.4} (context: rel_l2(gam,mgcv)={rel_gam_vs_mgcv:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_duchon_regimes::mu",
            "mu_rmse",
            gam_mu_rmse,
            "mgcv",
            mgcv_mu_rmse,
        )
        .line()
    );

    // (1) ABSOLUTE recovery bar: gam must recover the smooth count-mean curve.
    // mu_true ranges ~[1.0, 7.4] (signal RMS ≈ 4.0). A correct Poisson/log
    // Duchon fit reconstructs it to a small fraction of that; 1.0 is a
    // principled "clearly recovering, not collapsing to the flat mean" bar that
    // still catches a broken link/penalty interaction.
    assert!(
        gam_mu_rmse < 1.0,
        "gam Poisson Duchon failed to recover the count-mean curve: \
         RMSE-vs-truth={gam_mu_rmse:.4} (signal RMS≈{rms_mu:.4}, bar 1.0)"
    );

    // (2) MATCH-OR-BEAT mgcv ON ACCURACY VS THE TRUTH (within 10%).
    assert!(
        gam_mu_rmse <= mgcv_mu_rmse * 1.10,
        "gam recovers the Poisson mean worse than mgcv: \
         gam RMSE-vs-truth={gam_mu_rmse:.4} > 1.10 * mgcv={mgcv_mu_rmse:.4}"
    );
}
