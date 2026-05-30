//! End-to-end quality: gam's factor-smooth random-slope structure must match
//! `lme4::lmer` — the mature, standard mixed-model implementation — at recovering
//! per-group linear trends when the slope varies by group.
//!
//! This benchmarks gam's `fs(x, g)` (mgcv factor-smooth "fs": one smooth per
//! group sharing a SINGLE marginal smoothing parameter, i.e. the smooths are
//! treated as a random effect with a common variance component) against
//! `lme4::lmer(y ~ 1 + x + (x | g))` (a correlated random intercept + random
//! slope per group). Both are the GAM/mixed-model analogue of the SAME
//! statistical object: slope heterogeneity across groups estimated under a
//! shared variance component, with partial pooling shrinking each group's
//! curve toward the population trend. We give the two engines byte-identical
//! data and assert:
//!   1. the per-(group, x) predicted means agree across a 6-group × 4-point
//!      grid (gam fitted values vs lmer conditional expectations E[y | g, x]),
//!   2. the recovered per-group SLOPES β̂_g agree (the random-slope quantity
//!      that is the whole point of the structure).
//!
//! Truth: y = 2 + 3·x + β_g·x + ε, β_g ~ N(0, 1.5²) (so the per-group slope is
//! 3 + β_g), ε ~ N(0, 0.3), n = 240 over 6 groups of 40. With n_per_group = 40
//! the slope is well-identified per group and both engines should recover the
//! same partially-pooled trends. A genuine divergence (e.g. gam over-shrinking
//! the slope variance, or collapsing the groups) is a real bug, not a tolerance
//! issue, so the bounds are tight and grounded in the math: r > 0.98 on the
//! group×point prediction grid and r > 0.95 on the per-group slopes.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, run_r};
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

    // ---- compare ----------------------------------------------------------
    let pred_corr = pearson(&gam_pred, lmer_pred);
    let slope_corr = pearson(&gam_slope, lmer_slope);

    // Also correlate each engine's recovered slopes against the TRUE per-group
    // slopes — a sanity check that both engines actually recovered the signal,
    // not merely that they agree with each other.
    let gam_truth_corr = pearson(&gam_slope, &true_slope);
    let lmer_truth_corr = pearson(lmer_slope, &true_slope);

    eprintln!(
        "fs(x,g) vs lmer(x|g): n={n} groups={N_GROUPS} per_group={N_PER_GROUP} gam_edf={gam_edf:.3}\n  \
         pred grid pearson={pred_corr:.5} (over {grid_len} group×x points)\n  \
         slope pearson(gam,lmer)={slope_corr:.5}\n  \
         slope-vs-truth: gam={gam_truth_corr:.4} lmer={lmer_truth_corr:.4}\n  \
         true_slopes={true_slope:?}\n  \
         gam_slopes={gam_slope:?}\n  \
         lmer_slopes={lmer_slope:?}"
    );

    // (1) Prediction-grid agreement. Both engines fit a correlated random
    // intercept+slope per group under a shared variance component, so their
    // partially-pooled conditional means E[y | g, x] must track closely across
    // the full 6×4 grid. r > 0.98 is tight given the wide between-group slope
    // spread (SD = 1.5 → predictions at x=0.8 differ by ~several units across
    // groups), yet leaves margin for the basis/parameterization differences
    // between a penalized factor-smooth and an explicit Gaussian random effect.
    assert!(
        pred_corr > 0.98,
        "group×x predicted means diverge from lme4: pearson={pred_corr:.5}"
    );

    // (2) Per-group slope agreement — the random-slope quantity itself. Both
    // engines shrink the per-group slopes toward the population slope by the
    // same partial-pooling logic, so the recovered β̂_g must line up. r > 0.95
    // tolerates differing shrinkage factors while catching a real failure to
    // recover slope heterogeneity (which would push this toward 0).
    assert!(
        slope_corr > 0.95,
        "per-group slope estimates diverge from lme4: pearson={slope_corr:.5}"
    );

    // (3) Signal sanity — neither engine may have merely agreed on noise. With
    // SD = 1.5 slope spread vs σ = 0.3 residual over 40 pts/group the slopes are
    // strongly identified, so both must correlate highly with the truth.
    assert!(
        gam_truth_corr > 0.95,
        "gam failed to recover the true per-group slopes: pearson={gam_truth_corr:.4}"
    );
    assert!(
        lmer_truth_corr > 0.95,
        "lme4 failed to recover the true per-group slopes: pearson={lmer_truth_corr:.4}"
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
