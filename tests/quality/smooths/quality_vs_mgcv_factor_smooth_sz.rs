//! End-to-end quality: gam's sum-to-zero factor smooth `s(group, x, bs="sz")`.
//!
//! OBJECTIVE METRIC ASSERTED: this test generates data from a KNOWN truth —
//! per-group curves f_g(x) = sin(2*pi*x) * z_g with the amplitudes z_g centered
//! so sum_g z_g = 0 (hence sum_g f_g(x) = 0 pointwise) — and asserts that gam
//! RECOVERS that truth and ENFORCES the defining structural constraint:
//!
//!   1. TRUTH RECOVERY (primary, category 1): the RMSE of gam's estimated
//!      per-group deviation curves against the true f_g(x), evaluated at every
//!      training row, is no larger than the observation noise sigma. Because the
//!      truth is mean-zero (no intercept), gam's deviation = fitted - intercept
//!      is compared directly to f_g(x_i). A smoother that recovered noise rather
//!      than the signal would blow past this bar.
//!   2. MATCH-OR-BEAT ACCURACY (category 1 addendum): mgcv — the mature, standard
//!      GAM implementation — is fit on the IDENTICAL data and its own per-row
//!      smooth-term estimate is scored against the SAME truth. gam's truth-
//!      recovery RMSE must be no worse than mgcv's by more than 10%. mgcv is a
//!      BASELINE TO MATCH-OR-BEAT on accuracy, not an output to reproduce.
//!   3. STRUCTURE / CONSTRAINT (category 4): the across-group sum of gam's fitted
//!      per-level smooth deviations is ~0 at every x — the defining identifiability
//!      property of the `sz` basis — measured after removing the shared intercept.
//!
//! The primary claim is that gam recovers the generating function; matching mgcv
//! is only a secondary accuracy guardrail. We still compute and print gam-vs-mgcv
//! relative L2 for context, but it is NOT a pass/fail criterion.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_GROUPS: usize = 6;
const PER_GROUP: usize = 60; // n = 360
const SEED: u64 = 77;

#[test]
fn gam_factor_smooth_sz_matches_mgcv() {
    init_parallelism();

    // ---- synthetic data ----------------------------------------------------
    // x ~ U(0,1); g in {0..5}; f_g(x) = sin(2*pi*x) * z_g with z_g ~ N(0,0.8)
    // CENTERED across groups so sum_g z_g = 0 => sum_g f_g(x) = 0 pointwise (the
    // truth itself satisfies the sz identifiability constraint). y = f_g(x)+eps.
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let znorm = Normal::new(0.0, 0.8).expect("amplitude normal");
    let eps = Normal::new(0.0, 0.2).expect("noise normal");

    // Draw per-group amplitudes and center them so the truth sums to zero.
    let mut z: Vec<f64> = (0..N_GROUPS).map(|_| znorm.sample(&mut rng)).collect();
    let zbar: f64 = z.iter().sum::<f64>() / N_GROUPS as f64;
    for zi in z.iter_mut() {
        *zi -= zbar;
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let n = N_GROUPS * PER_GROUP;
    let mut group_code: Vec<f64> = Vec::with_capacity(n); // 0..5 (mgcv factor levels)
    let mut group_str: Vec<String> = Vec::with_capacity(n); // "g0".."g5" for gam categorical
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let mut truth: Vec<f64> = Vec::with_capacity(n); // f_g(x_i), the mean-zero signal
    // Row order encounters group 0 first, then 1, ... so gam's level codes
    // (assigned in encounter order) coincide with mgcv's numeric labels 0..5.
    for g in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi = ux.sample(&mut rng);
            let fi = (two_pi * xi).sin() * z[g];
            let yi = fi + eps.sample(&mut rng);
            group_code.push(g as f64);
            group_str.push(format!("g{g}"));
            x.push(xi);
            y.push(yi);
            truth.push(fi);
        }
    }
    // Observation noise standard deviation that generated y from the truth.
    const NOISE_SD: f64 = 0.2;
    let signal_sd = {
        let m = y.iter().sum::<f64>() / n as f64;
        (y.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
    };

    // ---- fit with gam: y ~ s(group, x, bs="sz"), REML ----------------------
    // The group column is string-valued => inferred Categorical, which the
    // sz path requires; the numeric companion x carries the smooth.
    let headers: Vec<String> = vec!["group".into(), "x".into(), "y".into()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                group_str[i].clone(),
                x[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode sz dataset");
    let col = ds.column_map();
    let group_idx = col["group"];
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(group, x, bs=\"sz\")", &ds, &cfg).expect("gam sz fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the sz factor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
    let n_cols = ds.headers.len();
    eprintln!(
        "[#1074-sz] edf_total={:.3} edf_by_block={:?} log_lambdas={:?} reml={:.4} converged={} iters={}",
        gam_edf,
        fit.fit
            .edf_by_block()
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
        fit.fit
            .log_lambdas
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
        fit.fit.reml_score,
        fit.fit.outer_converged,
        fit.fit.outer_iterations,
    );

    // gam fitted values at the training rows (identity link => design*beta).
    let mut train_grid = Array2::<f64>::zeros((n, n_cols));
    for i in 0..n {
        train_grid[[i, group_idx]] = group_code[i];
        train_grid[[i, x_idx]] = x[i];
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild gam design at training rows");
    let intercept_idx = train_design.intercept_range.start;
    let intercept = fit.fit.beta[intercept_idx];
    let gam_fitted: Vec<f64> = train_design.design.apply(&fit.fit.beta).to_vec();

    // ---- constraint check on a shared x-grid (gam side) --------------------
    // For each group, evaluate the fitted mean on a common x-grid, subtract the
    // shared intercept to isolate the per-level smooth deviation, then sum the
    // deviations across the 6 groups at every grid point. The sz constraint
    // forces that across-group sum to be ~0 everywhere.
    let grid_m = 40usize;
    let xg: Vec<f64> = (0..grid_m)
        .map(|i| 0.01 + 0.98 * i as f64 / (grid_m as f64 - 1.0))
        .collect();
    let mut sum_dev = vec![0.0f64; grid_m]; // sum over groups of (fit_g - intercept)
    for g in 0..N_GROUPS {
        let mut gg = Array2::<f64>::zeros((grid_m, n_cols));
        for (i, &xv) in xg.iter().enumerate() {
            gg[[i, group_idx]] = g as f64;
            gg[[i, x_idx]] = xv;
        }
        let gd = build_term_collection_design(gg.view(), &fit.resolvedspec)
            .expect("rebuild gam design on grid");
        let fit_g: Vec<f64> = gd.design.apply(&fit.fit.beta).to_vec();
        for i in 0..grid_m {
            sum_dev[i] += fit_g[i] - intercept;
        }
    }
    let constraint_max = sum_dev.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let constraint_rms = (sum_dev.iter().map(|v| v * v).sum::<f64>() / grid_m as f64).sqrt();

    // ---- fit the SAME model with mgcv (the mature reference) ---------------
    // mgcv requires the grouping variable to be a factor for bs="sz"; build it
    // from the integer codes. predict(type="terms") returns the per-row smooth
    // contribution (intercept excluded), exactly the deviations the constraint
    // governs.
    let r = run_r(
        &[
            Column::new("g", &group_code),
            Column::new("x", &x),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        df$gf <- factor(df$g)
        m <- gam(y ~ s(gf, x, bs = "sz"), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        # Per-row smooth-term estimate (intercept excluded): mgcv's estimate of
        # the mean-zero per-group deviation, directly comparable to the truth.
        tt_train <- predict(m, type = "terms")
        emit("term", as.numeric(rowSums(as.matrix(tt_train))))
        # Constraint on a shared x-grid: sum the per-level smooth deviations
        # across groups at each grid point; sz forces this to ~0.
        levs <- levels(df$gf)
        xg <- seq(0.01, 0.99, length.out = 40)
        sum_dev <- numeric(length(xg))
        for (lv in levs) {
          nd <- data.frame(gf = factor(lv, levels = levs), x = xg)
          tt <- predict(m, newdata = nd, type = "terms")
          # the single smooth term column carries the per-level deviation
          sum_dev <- sum_dev + as.numeric(tt[, 1])
        }
        emit("constraint_max", max(abs(sum_dev)))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_term = r.vector("term");
    let mgcv_constraint_max = r.scalar("constraint_max");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");
    assert_eq!(mgcv_term.len(), n, "mgcv term length mismatch");

    // ---- truth-recovery RMSE (the objective metric) ------------------------
    // gam's estimate of the mean-zero per-group deviation at each training row
    // is fitted - intercept; compare directly to the generating signal f_g(x_i).
    let gam_term: Vec<f64> = gam_fitted.iter().map(|v| v - intercept).collect();
    let gam_rmse = rmse(&gam_term, &truth);
    // mgcv's intercept-excluded smooth term scored against the SAME truth.
    let mgcv_rmse = rmse(mgcv_term, &truth);

    // Context only (NOT a pass/fail criterion): how closely the two engines'
    // fitted values track each other.
    let rel = relative_l2(&gam_fitted, mgcv_fitted);

    eprintln!(
        "sz factor smooth: n={n} groups={N_GROUPS} signal_sd={signal_sd:.4} noise_sd={NOISE_SD:.4} \
         gam_rmse_vs_truth={gam_rmse:.5} mgcv_rmse_vs_truth={mgcv_rmse:.5} \
         gam_constraint_max={constraint_max:.5} gam_constraint_rms={constraint_rms:.5} \
         mgcv_constraint_max={mgcv_constraint_max:.5} rel_l2={rel:.4} gam_edf={gam_edf:.3}"
    );

    // (1) TRUTH RECOVERY (PRIMARY). gam must recover the generating per-group
    // curves to within the observation noise: the row-wise RMSE of the estimated
    // deviation against the true f_g(x) is no larger than the noise sd that
    // corrupted y. A smoother that fit the noise instead of the signal — or that
    // failed to separate the per-group amplitudes — would exceed this bar.
    assert!(
        gam_rmse <= NOISE_SD,
        "gam sz does not recover the truth: rmse_vs_truth={gam_rmse:.5} > noise_sd={NOISE_SD:.4}"
    );

    // (2) MATCH-OR-BEAT mgcv ON ACCURACY. gam's truth-recovery error must be no
    // worse than the mature reference's by more than 10%. mgcv is a baseline to
    // match-or-beat on accuracy, not an output to reproduce.
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam sz less accurate than mgcv on truth recovery: gam_rmse={gam_rmse:.5} \
         > 1.10 * mgcv_rmse={:.5}",
        mgcv_rmse * 1.10
    );

    // (3) STRUCTURE / CONSTRAINT. The sz basis is *defined* by
    // sum_g deviation_g(x) = 0; gam builds the last level as the exact negative
    // sum of the rest, so the residual is pure floating-point / design-rebuild
    // noise. A bound of 0.01 * signal_sd is far above round-off yet small enough
    // that a basis that failed to enforce the constraint (deviations the size of
    // the signal) would trip it immediately. This is an objective property of
    // gam's own fit, asserted directly.
    let constraint_bound = 0.01 * signal_sd;
    assert!(
        constraint_max < constraint_bound,
        "gam sz violates sum-to-zero across groups: max|sum_g dev|={constraint_max:.5} \
         >= {constraint_bound:.5} (=0.01*signal_sd)"
    );
}
