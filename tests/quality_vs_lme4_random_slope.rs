//! End-to-end quality: gam's factor-smooth random-slope structure must RECOVER
//! the known per-group linear trends of a synthetic random-slope DGP, and do so
//! at least as accurately as `lme4::lmer` (the mature mixed-model reference,
//! demoted here to a baseline-to-match-or-beat).
//!
//! OBJECTIVE METRIC (the pass criterion): TRUTH RECOVERY. The data is generated
//! from a known function y = 2 + (3 + β_g)·x + ε with known per-group slopes
//! 3 + β_g and known conditional means E[y | g, x] = 2 + (3 + β_g)·x. We fit
//! gam's `fs(x, g)` and assert, against that GROUND TRUTH:
//!   1. RMSE of gam's per-(group, x) predicted means vs the true conditional
//!      means is below a noise-scaled bar (well under the residual σ = 0.3),
//!   2. RMSE of gam's recovered per-group SLOPES vs the true slopes is a small
//!      fraction of the between-group slope spread (SD = 1.5).
//! Both bars are absolute statements about how well gam recovers the signal —
//! they make no reference to lme4's output.
//!
//! lme4 REMAINS, as a BASELINE TO MATCH-OR-BEAT on those same truth-referenced
//! errors: we additionally require gam's prediction RMSE-to-truth and slope
//! RMSE-to-truth to be no worse than 1.10× lme4's. (We still COMPUTE and print
//! the gam-vs-lmer agreement for context, but it is NOT a pass criterion —
//! tracking a peer tool's noisy fit would prove nothing on its own.)
//!
//! `fs(x, g)` is mgcv's factor-smooth: one smooth of x per level of g, all
//! sharing a single smoothing parameter (a random-effect variance over the
//! per-group curves) — the GAM analogue of lmer's correlated random intercept +
//! slope `(x | g)`, with the shared penalty doing the partial pooling.
//!
//! Truth: y = 2 + 3·x + β_g·x + ε, β_g ~ N(0, 1.5²) (so the per-group slope is
//! 3 + β_g), ε ~ N(0, 0.3), n = 240 over 6 groups of 40. With n_per_group = 40
//! the slope is well-identified per group, so a genuine recovery failure (gam
//! over-shrinking the slope variance, or collapsing the groups) drives these
//! RMSEs up and fails the test — a real bug, not a tolerance issue.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_GROUPS: usize = 6;
const N_PER_GROUP: usize = 40;
const SLOPE_SD: f64 = 1.5;
const NOISE_SD: f64 = 0.3;
const FIXED_INTERCEPT: f64 = 2.0;
const FIXED_SLOPE: f64 = 3.0;
const SEED: u64 = 55;

/// Evaluation x-values at which we compare the two engines' group curves.
const X_EVAL: [f64; 4] = [0.2, 0.4, 0.6, 0.8];

#[test]
fn gam_factor_smooth_random_slope_matches_lme4() {
    init_parallelism();

    // ---- synthesize identical data for both engines ----------------------
    // n = 240 across 6 groups of 40. Each group g has its own slope
    // 3 + β_g with β_g ~ N(0, 1.5²); the response is y = 2 + (3 + β_g)·x + ε,
    // ε ~ N(0, 0.3). Group blocks are emitted in order g = 0..5 so the
    // categorical inference maps the level codes consistently for both engines.
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let slope_dist = Normal::new(0.0, SLOPE_SD).expect("slope normal");
    let noise = Normal::new(0.0, NOISE_SD).expect("noise normal");

    // Draw all per-group slope deviations first so the data-generating order is
    // deterministic and independent of how many points each group has.
    let beta_g: Vec<f64> = (0..N_GROUPS).map(|_| slope_dist.sample(&mut rng)).collect();
    let true_slope: Vec<f64> = beta_g.iter().map(|&b| FIXED_SLOPE + b).collect();

    let n = N_GROUPS * N_PER_GROUP;
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g_code = Vec::<f64>::with_capacity(n); // 0.0 .. 5.0, handed to R numeric
    let mut y = Vec::<f64>::with_capacity(n);
    let mut group_of_row = Vec::<usize>::with_capacity(n);

    for gi in 0..N_GROUPS {
        for _ in 0..N_PER_GROUP {
            let xi = ux.sample(&mut rng);
            let yi = FIXED_INTERCEPT + true_slope[gi] * xi + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            g_code.push(gi as f64);
            group_of_row.push(gi);
        }
    }

    // ---- fit with gam: y ~ fs(x, g), REML --------------------------------
    // `fs(x, g)` is mgcv's factor-smooth: one smooth of x per level of g, all
    // sharing a single smoothing parameter (a random-effect variance over the
    // per-group curves). The marginal smooth of x is the population trend; the
    // shared penalty does the partial pooling, exactly like lmer's (x | g).
    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                x[i].to_string(),
                // Emit g as a NON-numeric label ("g0".."g5"). gam's schema
                // inference classifies any column whose every value parses as
                // f64 as Continuous — an integer-valued string like "3" would
                // infer numeric and `fs(x, g)` would reject it ("requires one
                // categorical factor variable"). A "g" prefix forces the
                // Categorical kind the factor-smooth needs.
                format!("g{}", g_code[i] as i64),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode random-slope dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ fs(x, g)", &ds, &cfg).expect("gam fs fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian factor-smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Build the prediction grid: every (group, x_eval) pair. The factor column
    // carries its encoded level index so each row is evaluated against ITS OWN
    // group's smooth block; identity link => design*beta = predicted mean.
    let grid_len = N_GROUPS * X_EVAL.len();
    let mut grid = Array2::<f64>::zeros((grid_len, ds.headers.len()));
    let mut row = 0usize;
    for gi in 0..N_GROUPS {
        for &xe in X_EVAL.iter() {
            grid[[row, x_idx]] = xe;
            grid[[row, g_idx]] = gi as f64;
            row += 1;
        }
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at evaluation grid");
    let gam_pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_pred.len(), grid_len, "gam prediction length mismatch");

    // gam per-group slope: the X_EVAL block for group gi is contiguous; fit the
    // straight line through its 4 predicted points by least squares (slope =
    // cov(x, yhat) / var(x)). The fs() marginal is a penalized cubic B-spline,
    // but the wiggliness penalty shrinks each group's curve toward its linear
    // null space (the true DGP is linear), so the LS slope through 4 evenly
    // spaced points is a faithful summary of the recovered per-group trend.
    let gam_slope = group_slopes(&gam_pred);

    // ---- fit the SAME model with lme4 (the mature reference) -------------
    // g arrives as a numeric 0..5 column; rebuild it as the matching factor so
    // lmer's (x | g) gives a correlated random intercept + slope per level.
    // Conditional expectation E[y | g, x] = (β0 + b0_g) + (β1 + b1_g)·x is read
    // off coef(m)$g, which already adds the fixed effects to the BLUPs — the
    // exact analogue of gam's per-group fitted curve.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("g", &g_code),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(lme4))
        df$g <- factor(as.integer(round(df$g)))
        m <- lmer(y ~ 1 + x + (x | g), data = df,
                  control = lmerControl(check.conv.singular = "ignore"))
        cf <- coef(m)$g               # per-group (Intercept) and x columns
        a  <- cf[, "(Intercept)"]     # group-specific intercepts (β0 + b0_g)
        b  <- cf[, "x"]               # group-specific slopes      (β1 + b1_g)
        lv <- sort(as.integer(levels(df$g)))
        xe <- c(0.2, 0.4, 0.6, 0.8)
        preds <- numeric(0)
        for (gi in lv) {
          ai <- a[as.character(gi)]
          bi <- b[as.character(gi)]
          preds <- c(preds, ai + bi * xe)
        }
        slopes <- b[as.character(lv)]
        emit("pred", as.numeric(preds))
        emit("slope", as.numeric(slopes))
        "#,
    );
    let lmer_pred = r.vector("pred");
    let lmer_slope = r.vector("slope");
    assert_eq!(
        lmer_pred.len(),
        grid_len,
        "lmer prediction grid length mismatch"
    );
    assert_eq!(lmer_slope.len(), N_GROUPS, "lmer slope count mismatch");

    // ---- GROUND TRUTH at the same grid / slopes --------------------------
    // The DGP is exactly linear in x per group, so the true conditional mean at
    // (group gi, x) is FIXED_INTERCEPT + true_slope[gi]·x, and the true slope of
    // group gi is true_slope[gi]. These are the quantities the test asserts
    // recovery of — no reference tool involved.
    let mut truth_pred = Vec::<f64>::with_capacity(grid_len);
    for gi in 0..N_GROUPS {
        for &xe in X_EVAL.iter() {
            truth_pred.push(FIXED_INTERCEPT + true_slope[gi] * xe);
        }
    }

    // ---- truth-referenced errors: gam (primary) and lmer (baseline) ------
    let gam_pred_rmse = rmse(&gam_pred, &truth_pred);
    let gam_slope_rmse = rmse(&gam_slope, &true_slope);
    let lmer_pred_rmse = rmse(lmer_pred, &truth_pred);
    let lmer_slope_rmse = rmse(lmer_slope, &true_slope);

    // ---- gam-vs-lmer AGREEMENT: printed for context only, NOT asserted ---
    let pred_corr = pearson(&gam_pred, lmer_pred);
    let slope_corr = pearson(&gam_slope, lmer_slope);

    eprintln!(
        "fs(x,g) random-slope recovery: n={n} groups={N_GROUPS} per_group={N_PER_GROUP} gam_edf={gam_edf:.3}\n  \
         RMSE-to-truth  pred:  gam={gam_pred_rmse:.5} lmer={lmer_pred_rmse:.5} (over {grid_len} group×x points, noise σ={NOISE_SD})\n  \
         RMSE-to-truth  slope: gam={gam_slope_rmse:.5} lmer={lmer_slope_rmse:.5} (slope spread SD={SLOPE_SD})\n  \
         [context only] gam-vs-lmer pearson: pred={pred_corr:.5} slope={slope_corr:.5}\n  \
         true_slopes={true_slope:?}\n  \
         gam_slopes={gam_slope:?}\n  \
         lmer_slopes={lmer_slope:?}"
    );

    // (1) PRIMARY — gam recovers the true conditional means. The data carries
    // residual noise σ = 0.3; with 40 points/group the partially-pooled mean
    // curve is estimated far better than a single observation, so the average
    // error of the fitted means at the evaluation grid must sit well below σ.
    // The bar 0.5·σ = 0.15 is an absolute accuracy statement (it never mentions
    // lme4); breaching it means gam mis-estimated the per-group trends.
    let pred_bar = 0.5 * NOISE_SD;
    assert!(
        gam_pred_rmse <= pred_bar,
        "gam predicted means missed the truth: RMSE-to-truth={gam_pred_rmse:.5} > bar {pred_bar:.5}"
    );

    // (2) PRIMARY — gam recovers the per-group SLOPES, the random-slope quantity
    // itself. The between-group slope spread is SD = 1.5; recovering the slopes
    // to within RMSE 0.30 (one fifth of that spread) means the slope
    // heterogeneity is genuinely captured, not shrunk to the population mean.
    // Absolute bar, independent of any reference tool.
    let slope_bar = 0.2 * SLOPE_SD;
    assert!(
        gam_slope_rmse <= slope_bar,
        "gam slopes missed the truth: RMSE-to-truth={gam_slope_rmse:.5} > bar {slope_bar:.5}"
    );

    // (3) MATCH-OR-BEAT — on the SAME truth-referenced error, gam must be no
    // worse than 1.10× the mature mixed-model reference. lme4 is the accuracy
    // baseline here, never the target: we beat-or-match its recovery error, we
    // do not reproduce its fit.
    assert!(
        gam_pred_rmse <= lmer_pred_rmse * 1.10,
        "gam predicted-mean recovery worse than lme4: gam={gam_pred_rmse:.5} > 1.10·lmer={:.5}",
        lmer_pred_rmse * 1.10
    );
    assert!(
        gam_slope_rmse <= lmer_slope_rmse * 1.10,
        "gam slope recovery worse than lme4: gam={gam_slope_rmse:.5} > 1.10·lmer={:.5}",
        lmer_slope_rmse * 1.10
    );
}

/// Least-squares slope of each group's contiguous `X_EVAL` block of predictions
/// against `X_EVAL`. Returns one slope per group, in group order.
fn group_slopes(preds: &[f64]) -> Vec<f64> {
    let k = X_EVAL.len();
    let xmean = X_EVAL.iter().sum::<f64>() / k as f64;
    let sxx: f64 = X_EVAL.iter().map(|&xe| (xe - xmean) * (xe - xmean)).sum();
    let mut out = Vec::with_capacity(N_GROUPS);
    for gi in 0..N_GROUPS {
        let block = &preds[gi * k..(gi + 1) * k];
        let ymean = block.iter().sum::<f64>() / k as f64;
        let sxy: f64 = X_EVAL
            .iter()
            .zip(block)
            .map(|(&xe, &yh)| (xe - xmean) * (yh - ymean))
            .sum();
        out.push(sxy / sxx);
    }
    out
}
