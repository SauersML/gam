//! End-to-end quality: gam's 95% confidence-interval *coverage* must be
//! uniform across the covariate range, benchmarked head-to-head against
//! **mgcv** — the mature, standard penalized-GAM implementation — using its
//! `predict(..., se.fit = TRUE)` standard-error path.
//!
//! Why mgcv is the right comparator here: mgcv's `predict.gam(se.fit=TRUE)`
//! returns the Bayesian posterior standard error of the linear predictor
//! (Wood 2006, §4.8 / §6.10), built from the same conditional covariance
//! `Vb = H^{-1} * phi` that gam exposes through `beta_covariance()`. Wald
//! intervals `eta_hat ± z * se` from that covariance are the canonical
//! frequentist-Bayesian smooth interval. A *single* fit cannot reveal whether
//! those intervals have the right coverage — only a Monte-Carlo sweep can —
//! so this test fits 50 independent replicates of a known Gaussian smooth and
//! measures the *empirical* coverage of the true function, binned along the
//! covariate range, for BOTH engines.
//!
//! The intrinsic correctness property (independent of mgcv) is coverage
//! *uniformity*: a correctly conditioned smoother whose SEs honour the local
//! information landscape attains close to the nominal 95% in every region of
//! the covariate, not just on average. Coverage that collapses at the
//! data-sparse tails (boundary basis instability, mis-scaled shrinkage) or
//! that is wildly conservative in the interior is a real bug. We therefore
//! assert a principled band on gam's per-bin coverage and report mgcv's bands
//! alongside as the mature reference.
//!
//! Data (fed IDENTICALLY to both engines): n=300, x~U[0,1],
//! true eta = 2*sin(2*pi*x), y ~ N(eta, sd=0.15), per-replicate seeds derived
//! deterministically from base seed 99. Model: y ~ s(x, k=12), Gaussian
//! identity, REML. Boundary correction is disabled in gam's
//! `PredictUncertaintyOptions` to isolate the base conditional-covariance
//! interval (the exact analogue of mgcv's `se.fit`), so the comparison is
//! apples-to-apples.

use gam::estimate::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::types::LikelihoodSpec;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

const N_TRAIN: usize = 300;
const N_REPLICATES: usize = 50;
const N_GRID: usize = 60; // evaluation grid resolution over [0, 1]
const N_BINS: usize = 5;
const NOISE_SD: f64 = 0.15;
const CONFIDENCE_LEVEL: f64 = 0.95;
const BASE_SEED: u64 = 99;

#[inline]
fn true_eta(x: f64) -> f64 {
    2.0 * (2.0 * PI * x).sin()
}

#[test]
fn gam_confidence_interval_coverage_is_uniform_vs_mgcv_se_fit() {
    init_parallelism();

    // Shared evaluation grid over [0, 1]. Both engines score the SAME true
    // function on the SAME grid, binned identically into N_BINS equal-width
    // bins, so coverage is compared region-by-region with no alignment slack.
    let grid: Vec<f64> = (0..N_GRID)
        .map(|i| i as f64 / (N_GRID as f64 - 1.0))
        .collect();
    let truth_grid: Vec<f64> = grid.iter().map(|&x| true_eta(x)).collect();
    let bin_of = |x: f64| -> usize {
        // x in [0,1]; map to [0, N_BINS-1].
        let b = (x * N_BINS as f64).floor() as isize;
        b.clamp(0, N_BINS as isize - 1) as usize
    };

    // gam per-bin coverage accumulators (hits and total over all replicates).
    let mut gam_hits = [0usize; N_BINS];
    let mut gam_total = [0usize; N_BINS];

    // Stacked long-format training data for a SINGLE mgcv call: (rep, x, y).
    // Generating identical (x, y) in Rust and shipping it to R guarantees both
    // engines fit byte-identical data.
    let mut rep_col: Vec<f64> = Vec::with_capacity(N_TRAIN * N_REPLICATES);
    let mut x_col: Vec<f64> = Vec::with_capacity(N_TRAIN * N_REPLICATES);
    let mut y_col: Vec<f64> = Vec::with_capacity(N_TRAIN * N_REPLICATES);

    let z = standard_normal_quantile(0.5 + 0.5 * CONFIDENCE_LEVEL);

    for rep in 0..N_REPLICATES {
        // Deterministic per-replicate seed derived from the base seed.
        let mut rng = StdRng::seed_from_u64(BASE_SEED.wrapping_mul(1_000_003).wrapping_add(rep as u64));
        let unif = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
        let noise = Normal::new(0.0, NOISE_SD).expect("normal noise");

        let mut x = Vec::with_capacity(N_TRAIN);
        let mut y = Vec::with_capacity(N_TRAIN);
        for _ in 0..N_TRAIN {
            let xi = unif.sample(&mut rng);
            let yi = true_eta(xi) + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
        }

        // Stash for the (single) mgcv reference call.
        for i in 0..N_TRAIN {
            rep_col.push(rep as f64);
            x_col.push(x[i]);
            y_col.push(y[i]);
        }

        // ---- fit with gam: y ~ s(x, k=12), Gaussian identity, REML --------
        let headers = ["x", "y"].into_iter().map(String::from).collect();
        let rows: Vec<StringRecord> = (0..N_TRAIN)
            .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
            .collect();
        let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode replicate");
        let col = ds.column_map();
        let x_idx = col["x"];

        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ s(x, k=12)", &ds, &cfg).expect("gam gaussian smooth fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for y ~ s(x, k=12)");
        };

        // Conditional posterior covariance Vb = H^{-1}*phi — the exact analogue
        // of what mgcv's predict(se.fit=TRUE) reports for the linear predictor.
        let cov: Array2<f64> = fit
            .fit
            .beta_covariance()
            .expect("gam must expose a conditional coefficient covariance for a Gaussian smooth")
            .clone();

        // Rebuild the frozen design at the shared eval grid.
        let mut grid_mat = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
        for (i, &xg) in grid.iter().enumerate() {
            grid_mat[[i, x_idx]] = xg;
        }
        let eval_design = build_term_collection_design(grid_mat.view(), &fit.resolvedspec)
            .expect("rebuild s(x) design at eval grid");
        let offset = Array1::<f64>::zeros(N_GRID);

        // Base conditional-covariance Wald interval: boundary correction OFF so
        // this is the unadorned eta_hat ± z*se path, matching mgcv's se.fit.
        let pred = predict_gamwith_uncertainty(
            &eval_design.design,
            fit.fit.beta.view(),
            offset.view(),
            LikelihoodSpec::gaussian_identity(),
            &cov,
            &PredictUncertaintyOptions {
                confidence_level: CONFIDENCE_LEVEL,
                covariance_mode: InferenceCovarianceMode::Conditional,
                mean_interval_method: MeanIntervalMethod::TransformEta,
                includeobservation_interval: false,
                apply_bias_correction: false,
                edgeworth_one_sided: false,
                boundary_correction: false,
                ood_inflation: false,
                multi_point_joint: false,
                ..PredictUncertaintyOptions::default()
            },
        )
        .expect("gam conditional-covariance interval prediction");

        // Score: is the TRUE eta inside [eta_lower, eta_upper] at each grid point?
        assert_eq!(pred.eta_lower.len(), N_GRID, "gam eta_lower length mismatch");
        for g in 0..N_GRID {
            let lo = pred.eta_lower[g];
            let hi = pred.eta_upper[g];
            let b = bin_of(grid[g]);
            gam_total[b] += 1;
            if lo <= truth_grid[g] && truth_grid[g] <= hi {
                gam_hits[b] += 1;
            }
        }
    }

    // ---- fit ALL 50 replicates with mgcv (the mature reference) -----------
    // One R process: split the stacked frame by `rep`, fit y ~ s(x, k=12) by
    // REML, predict on the shared grid with se.fit=TRUE, build the same 95%
    // Wald interval, and accumulate per-bin coverage of the identical truth.
    use gam::test_support::reference::{Column, run_r};
    let r = run_r(
        &[
            Column::new("rep", &rep_col),
            Column::new("xcol", &x_col),
            Column::new("ycol", &y_col),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            NG    <- {ng}
            NB    <- {nb}
            NR    <- {nr}
            zmult <- {zmult}
            xg    <- seq(0, 1, length.out = NG)
            truth <- 2 * sin(2 * pi * xg)
            binid <- pmin(NB - 1L, pmax(0L, floor(xg * NB)))  # 0-based bin per grid pt
            hits  <- rep(0, NB)
            tot   <- rep(0, NB)
            for (rr in 0:(NR - 1)) {{
              sub <- df[df$rep == rr, ]
              m   <- gam(ycol ~ s(xcol, k = 12), data = sub, method = "REML")
              pr  <- predict(m, newdata = data.frame(xcol = xg), se.fit = TRUE)
              lo  <- pr$fit - zmult * pr$se.fit
              hi  <- pr$fit + zmult * pr$se.fit
              inside <- as.integer(lo <= truth & truth <= hi)
              for (g in 1:NG) {{
                b <- binid[g] + 1L
                tot[b]  <- tot[b] + 1
                hits[b] <- hits[b] + inside[g]
              }}
            }}
            emit("mgcv_cov", hits / tot)
            "#,
            ng = N_GRID,
            nb = N_BINS,
            nr = N_REPLICATES,
            zmult = z,
        ),
    );
    let mgcv_cov = r.vector("mgcv_cov");
    assert_eq!(mgcv_cov.len(), N_BINS, "mgcv must report one coverage per bin");

    // ---- gam per-bin coverage ---------------------------------------------
    let gam_cov: Vec<f64> = (0..N_BINS)
        .map(|b| {
            assert!(gam_total[b] > 0, "bin {b} had no grid points");
            gam_hits[b] as f64 / gam_total[b] as f64
        })
        .collect();

    let gam_min = gam_cov.iter().cloned().fold(f64::INFINITY, f64::min);
    let gam_max = gam_cov.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gam_mean = gam_cov.iter().sum::<f64>() / N_BINS as f64;
    let gam_range = gam_max - gam_min;

    let mgcv_min = mgcv_cov.iter().cloned().fold(f64::INFINITY, f64::min);
    let mgcv_max = mgcv_cov.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mgcv_mean = mgcv_cov.iter().sum::<f64>() / N_BINS as f64;

    eprintln!(
        "95% CI coverage sweep (n={N_TRAIN}, reps={N_REPLICATES}, grid={N_GRID}, bins={N_BINS})"
    );
    eprintln!("  gam  per-bin coverage : {gam_cov:?}");
    eprintln!("  mgcv per-bin coverage : {mgcv_cov:?}");
    eprintln!(
        "  gam  (min, mean, max) = ({gam_min:.3}, {gam_mean:.3}, {gam_max:.3})  range={gam_range:.3}"
    );
    eprintln!(
        "  mgcv (min, mean, max) = ({mgcv_min:.3}, {mgcv_mean:.3}, {mgcv_max:.3})"
    );

    // ---- principled, un-weakened bounds (spec) ----------------------------
    // For a correctly conditioned Gaussian smooth, nominal-95% Wald intervals
    // from Vb should attain ~95% empirical coverage everywhere. The spec band
    // is: every bin in [0.90, 1.00] and the spread across bins < 0.08. With
    // 50 reps * (N_GRID/N_BINS = 12) grid points per bin = 600 trials/bin, the
    // binomial standard error at p=0.95 is ~0.009, so a true-95% smoother lands
    // in [0.90, 1.00] with overwhelming probability; a bin below 0.90, or a
    // tail-vs-interior spread above 0.08, signals coverage that is genuinely
    // non-uniform across the covariate (the failure this test exists to catch).
    for (b, &c) in gam_cov.iter().enumerate() {
        assert!(
            (0.90..=1.0).contains(&c),
            "gam bin {b} coverage {c:.3} fell outside [0.90, 1.00]; \
             intervals do not respect the local information landscape \
             (per-bin coverage: {gam_cov:?})"
        );
    }
    assert!(
        gam_range < 0.08,
        "gam coverage is non-uniform across the covariate: spread {gam_range:.3} >= 0.08 \
         (per-bin coverage: {gam_cov:?}); SEs fail to track basis shrinkage away from \
         data-rich regions"
    );
}

/// Standard-normal quantile (inverse CDF) via the Acklam rational
/// approximation — used to build the z-multiplier for the 95% Wald interval on
/// BOTH sides so gam and mgcv use the identical critical value.
fn standard_normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "quantile probability must be in (0,1)");
    // Coefficients (Acklam, 2003).
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
