//! End-to-end quality: gam's 1-D Duchon spline `duchon(x, k=...)` across MORE
//! noise/smoothness/family regimes than the single sine test, each judged by
//! OBJECTIVE TRUTH RECOVERY and held to match-or-beat mgcv on that same metric.
//!
//! The pass/fail claim is RMSE of gam's fit against the KNOWN generating function
//! (truth recovery); we additionally assert gam's truth-recovery RMSE ≤ mgcv's ×
//! 1.10 (match-or-beat the mature baseline on accuracy vs the truth — NOT
//! closeness of the two fitted curves).
//!
//! #2395 K-seed averaging: the former single fixed noise seed put the gam-vs-mgcv
//! margin on a knife-edge whose sign flipped across seeds (pure sampling noise).
//! Each regime now regenerates its data under K noise seeds (the deterministic
//! x-grid and the truth are unchanged; only the noise draw varies), fits gam and
//! mgcv on the SAME K datasets (identical per-seed response columns shipped into
//! the R body), and averages the truth-recovery RMSE before asserting. Averaging a
//! lower-variance metric against the same bars is strictly harder than the former
//! single seed, never a weakening.
//!
//! THE OBJECT UNDER TEST is the DEFAULT `duchon(x, k=...)`: gam's redesigned
//! non-periodic Euclidean structural amplitude/slope/curvature smoother on the
//! cubic (`r³`) polyharmonic basis (affine null space, power s=(d−1)/2=0 in 1D).
//! BASELINE COMPARATOR throughout: `mgcv::gam(y ~ s(x, bs="ds", k=..,
//! m=c(2,0)), method="REML")` — the canonical 1-D cubic-spline-equivalent
//! Duchon smoother, the mature analogue of gam's default.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

/// #2395: K noise seeds per regime. The n=300 Duchon/`ds` fits are sub-second, so
/// 2*K=20 fits stay far inside the fast envelope while cutting the truth-recovery
/// metric's standard error ~sqrt(K)=3.2x — the sampling noise that flipped the
/// gam-vs-mgcv sign across single seeds.
const K_SEEDS: usize = 10;

/// Dense interior test grid length (truth-recovery grid, shared by every seed).
const GRID_M: usize = 201;

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Fit gam's default 1-D Duchon at the given `k`, fit mgcv's `bs="ds"`,
/// `m=c(2,0)` analogue on byte-identical data, over K noise seeds, and assert on
/// the AVERAGED truth-recovery RMSE that (1) gam recovers the KNOWN `truth` below
/// `abs_bar` and (2) gam match-or-beats mgcv within 10%. `truth` maps x → the
/// noise-free signal; `sigma` is the Gaussian noise SD; `seed_base` seeds the K
/// noise draws (`seed_base + s`).
fn assert_gaussian_regime(
    label: &str,
    n: usize,
    k: usize,
    sigma: f64,
    seed_base: u64,
    truth: &dyn Fn(f64) -> f64,
    abs_bar: f64,
) {
    init_parallelism();

    // Deterministic x-grid and truth grid — identical across every noise seed.
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let x_test: Vec<f64> = (0..GRID_M)
        .map(|i| 0.005 + 0.99 * i as f64 / (GRID_M as f64 - 1.0))
        .collect();
    let y_truth: Vec<f64> = x_test.iter().map(|&t| truth(t)).collect();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ duchon(x, k={k})");

    let mut gam_rmses = Vec::with_capacity(K_SEEDS);
    // Per-seed response columns for the mgcv arm: train y (n) followed by GRID_M
    // zeros so every column length matches x_all = x ++ x_test.
    let mut y_cols: Vec<Vec<f64>> = Vec::with_capacity(K_SEEDS);
    let mut y_names: Vec<String> = Vec::with_capacity(K_SEEDS);

    for s in 0..K_SEEDS {
        let mut rng = StdRng::seed_from_u64(seed_base + s as u64);
        let noise = Normal::new(0.0, sigma).expect("normal");
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
        let ds =
            encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic dataset");
        let x_idx = ds.column_map()["x"];

        let result = fit_from_formula(&formula, &ds, &cfg).expect("gam duchon fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a gaussian Duchon smooth");
        };
        let mut grid = Array2::<f64>::zeros((GRID_M, ds.headers.len()));
        for (i, &t) in x_test.iter().enumerate() {
            grid[[i, x_idx]] = t;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild Duchon design at test grid");
        let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse(&gam_fitted, &y_truth));

        let mut ycol = y.clone();
        ycol.extend(std::iter::repeat_n(0.0, GRID_M));
        y_cols.push(ycol);
        y_names.push(format!("y{s}"));
    }

    // mgcv on the SAME K datasets: x_all = train x ++ grid x; is_train mask; the
    // truth on the grid rows; one response column per seed. R loops the seeds.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, GRID_M));
    let mut truth_col = vec![0.0; n];
    truth_col.extend_from_slice(&y_truth);

    let mut columns: Vec<Column> = vec![
        Column::new("x", &x_all),
        Column::new("is_train", &is_train),
        Column::new("truth", &truth_col),
    ];
    for (name, data) in y_names.iter().zip(y_cols.iter()) {
        columns.push(Column::new(name, data));
    }
    let r = run_r(
        &columns,
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            K <- {K_SEEDS}
            rmses <- numeric(K)
            train_mask <- df$is_train > 0.5
            grid_mask  <- df$is_train < 0.5
            truth_grid <- df$truth[grid_mask]
            for (s in 0:(K - 1)) {{
              ys <- df[[paste0("y", s)]]
              train <- data.frame(x = df$x[train_mask], y = ys[train_mask])
              grid  <- data.frame(x = df$x[grid_mask])
              m <- gam(y ~ s(x, bs = "ds", k = {k}, m = c(2, 0)), data = train, method = "REML")
              pred <- as.numeric(predict(m, newdata = grid))
              rmses[s + 1] <- sqrt(mean((pred - truth_grid)^2))
            }}
            emit("mgcv_rmses", rmses)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    assert_eq!(mgcv_rmses.len(), K_SEEDS, "mgcv per-seed rmse count mismatch");

    let gam_rmse_avg = mean(&gam_rmses);
    let mgcv_rmse_avg = mean(mgcv_rmses);

    eprintln!(
        "duchon-regime[{label}] #2395 K={K_SEEDS}-seed avg: n={n} k={k} sigma={sigma} \
         gam_truth_rmse_avg={gam_rmse_avg:.4} mgcv_truth_rmse_avg={mgcv_rmse_avg:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            &format!("quality_vs_mgcv_duchon_regimes::{label}"),
            "truth_rmse",
            gam_rmse_avg,
            "mgcv",
            mgcv_rmse_avg,
        )
        .line()
    );

    // (1) ABSOLUTE recovery bar tied to the regime (averaged): gam must
    // reconstruct the signal below abs_bar, catching a blown-up or non-denoising
    // fit.
    assert!(
        gam_rmse_avg < abs_bar,
        "[{label}] gam Duchon failed to recover the truth: \
         averaged RMSE-vs-truth={gam_rmse_avg:.4} (bar {abs_bar})"
    );

    // (2) MATCH-OR-BEAT mgcv ON ACCURACY VS THE TRUTH (within 10%, averaged).
    assert!(
        gam_rmse_avg <= mgcv_rmse_avg * 1.10,
        "[{label}] gam recovers the truth worse than mgcv: \
         averaged gam RMSE-vs-truth={gam_rmse_avg:.4} > 1.10 * mgcv={mgcv_rmse_avg:.4}"
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
/// mgcv's `s(x, bs="ds")` Poisson fit, over K count-draw seeds (#2395). The
/// objective metric is recovery of the TRUE MEAN on the RESPONSE scale (the count
/// mean exp(eta_true)); both gam and mgcv are scored by AVERAGED RMSE of their
/// predicted mean against the true mean, and gam must match-or-beat mgcv.
#[test]
fn gam_duchon_1d_poisson_log_mean_recovery_matches_mgcv() {
    init_parallelism();

    // ---- known smooth log-mean: eta_true(x) = 1.0 + sin(2π x); the response-
    // scale mean is mu_true(x) = exp(eta_true), ranging over ~[1.0, 7.4]. counts
    // ~ Poisson(mu_true). Each seed draws a fresh count vector; gam and mgcv see
    // identical draws per seed.
    let n = 300usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let eta_true = |t: f64| 1.0 + (2.0 * std::f64::consts::PI * t).sin();
    let x_test: Vec<f64> = (0..GRID_M)
        .map(|i| 0.005 + 0.99 * i as f64 / (GRID_M as f64 - 1.0))
        .collect();
    let mu_truth: Vec<f64> = x_test.iter().map(|&t| eta_true(t).exp()).collect();

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };

    let mut gam_rmses = Vec::with_capacity(K_SEEDS);
    let mut count_cols: Vec<Vec<f64>> = Vec::with_capacity(K_SEEDS);
    let mut count_names: Vec<String> = Vec::with_capacity(K_SEEDS);

    for s in 0..K_SEEDS {
        let mut rng = StdRng::seed_from_u64(777 + s as u64);
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
        let ds =
            encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
        let x_idx = ds.column_map()["x"];

        let result = fit_from_formula("count ~ duchon(x, k=20)", &ds, &cfg)
            .expect("gam poisson duchon fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a Poisson Duchon smooth");
        };
        let mut grid = Array2::<f64>::zeros((GRID_M, ds.headers.len()));
        for (i, &t) in x_test.iter().enumerate() {
            grid[[i, x_idx]] = t;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild Poisson Duchon design at test grid");
        let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();
        gam_rmses.push(rmse(&gam_mu, &mu_truth));

        let mut ccol = count.clone();
        ccol.extend(std::iter::repeat_n(0.0, GRID_M));
        count_cols.push(ccol);
        count_names.push(format!("count{s}"));
    }

    // mgcv Poisson/log on the SAME K count draws; R loops, scoring response-scale
    // mean recovery against the true mean on the grid.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, GRID_M));
    let mut mu_col = vec![0.0; n];
    mu_col.extend_from_slice(&mu_truth);

    let mut columns: Vec<Column> = vec![
        Column::new("x", &x_all),
        Column::new("is_train", &is_train),
        Column::new("mu_true", &mu_col),
    ];
    for (name, data) in count_names.iter().zip(count_cols.iter()) {
        columns.push(Column::new(name, data));
    }
    let r = run_r(
        &columns,
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            K <- {K_SEEDS}
            rmses <- numeric(K)
            train_mask <- df$is_train > 0.5
            grid_mask  <- df$is_train < 0.5
            mu_grid <- df$mu_true[grid_mask]
            for (s in 0:(K - 1)) {{
              cs <- df[[paste0("count", s)]]
              train <- data.frame(x = df$x[train_mask], count = cs[train_mask])
              grid  <- data.frame(x = df$x[grid_mask])
              m <- gam(count ~ s(x, bs = "ds", k = 20, m = c(2, 0)), data = train,
                       family = poisson(link = "log"), method = "REML")
              pred <- as.numeric(predict(m, newdata = grid, type = "response"))
              rmses[s + 1] <- sqrt(mean((pred - mu_grid)^2))
            }}
            emit("mgcv_rmses", rmses)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    assert_eq!(mgcv_rmses.len(), K_SEEDS, "mgcv per-seed rmse count mismatch");

    let gam_mu_rmse_avg = mean(&gam_rmses);
    let mgcv_mu_rmse_avg = mean(mgcv_rmses);
    let rms_mu = (mu_truth.iter().map(|v| v * v).sum::<f64>() / mu_truth.len() as f64).sqrt();

    eprintln!(
        "duchon-poisson-mean-recovery-1d #2395 K={K_SEEDS}-seed avg: n={n} grid={GRID_M} \
         gam_mu_rmse_avg={gam_mu_rmse_avg:.4} mgcv_mu_rmse_avg={mgcv_mu_rmse_avg:.4} rms_mu={rms_mu:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_duchon_regimes::mu",
            "mu_rmse",
            gam_mu_rmse_avg,
            "mgcv",
            mgcv_mu_rmse_avg,
        )
        .line()
    );

    // (1) ABSOLUTE recovery bar (averaged): gam must recover the smooth count-mean
    // curve. mu_true ranges ~[1.0, 7.4] (signal RMS ≈ 4.0); 1.0 is a principled
    // "clearly recovering, not collapsing to the flat mean" bar.
    assert!(
        gam_mu_rmse_avg < 1.0,
        "gam Poisson Duchon failed to recover the count-mean curve: \
         averaged RMSE-vs-truth={gam_mu_rmse_avg:.4} (signal RMS≈{rms_mu:.4}, bar 1.0)"
    );

    // (2) MATCH-OR-BEAT mgcv ON ACCURACY VS THE TRUTH (within 10%, averaged).
    assert!(
        gam_mu_rmse_avg <= mgcv_mu_rmse_avg * 1.10,
        "gam recovers the Poisson mean worse than mgcv: \
         averaged gam RMSE-vs-truth={gam_mu_rmse_avg:.4} > 1.10 * mgcv={mgcv_mu_rmse_avg:.4}"
    );
}
