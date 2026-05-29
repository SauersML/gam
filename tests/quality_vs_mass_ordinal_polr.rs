//! End-to-end quality: gam's proportional-odds (cumulative-logit) ordinal
//! regression must agree with `MASS::polr` — the canonical R reference for
//! proportional-odds logistic regression — and is cross-checked against
//! `ordinal::clm` (cumulative logit). Both are *the* standard tools for
//! ordinal regression under the proportional-odds (shared-slope) assumption.
//!
//! gam has no bespoke "ordinal" family, but the proportional-odds model
//!
//!     P(Y <= j | x) = logit^{-1}( alpha_j + g(x) + beta * x2 ),   j = 1..J-1
//!
//! is *exactly* a binomial-logit GAM on the equivalent stacked dataset: for
//! every observation we emit J-1 binary rows z_j = 1{Y <= j}, share the
//! covariate effects across the cumulant index, and let level-specific fixed
//! intercepts (threshold dummies thr2, thr3 added as plain linear terms, with
//! j=1 the baseline) realize the J-1 thresholds alpha_j. This is the textbook
//! "binary expansion" identity for proportional odds (McCullagh 1980); the
//! likelihood gam maximizes on the stacked frame is bit-for-bit the polr/clm
//! likelihood. We fit gam's smooth-covariate version
//!     y ~ s(x, bs='cc') + x2
//! and the *same* proportional-odds likelihood in R.
//!
//! Because `MASS::polr` / `ordinal::clm` are strictly *linear* predictors, the
//! reference represents the periodic x-effect with the matching low-order
//! cyclic harmonics sin(x)+cos(x) (the linear analogue of a cyclic-cubic
//! smooth on a single oscillation), plus the shared linear x2. Identical data
//! (a fixed-seed synthetic ordinal sample) is handed to both engines.
//!
//! We assert agreement on the quantities that actually matter for an ordinal
//! model:
//!   1. fitted cumulative probabilities P(Y <= j) per level: max-abs-diff,
//!   2. per-level class probabilities P(Y = j): relative-L2,
//!   3. the shared linear x2 effect (slope on the logit scale): relative diff,
//! all over a common x-grid spanning the data. A genuine divergence here is a
//! real bug in gam's binomial/cyclic machinery, not a tolerance artifact.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_r};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

/// Latent periodic x-effect: one clean oscillation over the data window
/// [-3, 3] with period exactly 6, so g(-3) = g(3) = 0. This honors the seam
/// continuity that a cyclic-cubic (`bs='cc'`) smooth imposes (f(min) = f(max)),
/// making the cyclic basis the genuinely correct model — not an approximation
/// fighting a boundary discontinuity. Its linear-model analogue (used by polr)
/// is the first cyclic harmonic pair sin(pi x/3) + cos(pi x/3).
fn g_of_x(x: f64) -> f64 {
    0.9 * (std::f64::consts::PI * x / 3.0).sin()
}

/// Logistic CDF (inverse logit).
fn inv_logit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[test]
fn gam_proportional_odds_matches_mass_polr() {
    init_parallelism();

    // ---- synthesize a fixed-seed ordinal sample (J = 4 levels) -------------
    // latent eta = g(x) + beta * x2; assign level by fixed thresholds on eta.
    // Identical raw (x, x2, y) handed to gam and to R.
    let n = 250usize;
    let beta_x2_true = 0.7;
    // Proportional-odds thresholds (on the cumulative-logit scale): the data
    // generator places eta against these cutpoints to pick a level in 1..=4.
    let cut = [-0.85_f64, 0.0, 0.85];

    let mut rng = StdRng::seed_from_u64(20240529);
    let ux = Uniform::new(-3.0_f64, 3.0).expect("uniform x");
    let ux2 = Uniform::new(-1.0_f64, 1.0).expect("uniform x2");
    // Logistic noise so the generative model is exactly proportional-odds
    // logistic (latent-variable threshold formulation).
    let uunit = Uniform::new(1e-9_f64, 1.0 - 1e-9).expect("uniform unit");

    let mut x = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let x2i = ux2.sample(&mut rng);
        // logistic latent: eta = g(x) + beta*x2 + logistic_noise
        let u = uunit.sample(&mut rng);
        let noise = (u / (1.0 - u)).ln(); // standard logistic quantile
        let eta = g_of_x(xi) + beta_x2_true * x2i + noise;
        // Level = number of thresholds the latent exceeds, in 1..=4.
        let mut level = 1.0;
        for &c in &cut {
            if eta > c {
                level += 1.0;
            }
        }
        x.push(xi);
        x2.push(x2i);
        y.push(level);
    }

    // ---- fit gam: proportional-odds via the equivalent stacked binomial ----
    // For each obs i and cumulant j in {1,2,3}: row (z = 1{y_i <= j}, x_i,
    // x2_i, thr2 = 1{j>=2}, thr3 = 1{j>=3}). Shared s(x,bs='cc') + x2; the
    // threshold dummies are plain linear fixed effects giving alpha_2, alpha_3
    // relative to the j=1 baseline intercept. This IS proportional odds.
    let cumulants = [1.0_f64, 2.0, 3.0];
    let n_stack = n * cumulants.len();
    let mut sx = Vec::with_capacity(n_stack);
    let mut sx2 = Vec::with_capacity(n_stack);
    let mut sthr2 = Vec::with_capacity(n_stack);
    let mut sthr3 = Vec::with_capacity(n_stack);
    let mut sz = Vec::with_capacity(n_stack);
    for i in 0..n {
        for &j in &cumulants {
            sx.push(x[i]);
            sx2.push(x2[i]);
            sthr2.push(if j >= 2.0 { 1.0 } else { 0.0 });
            sthr3.push(if j >= 3.0 { 1.0 } else { 0.0 });
            sz.push(if y[i] <= j { 1.0 } else { 0.0 });
        }
    }

    let headers = ["z", "x", "x2", "thr2", "thr3"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows = (0..n_stack)
        .map(|r| {
            csv::StringRecord::from(vec![
                sz[r].to_string(),
                sx[r].to_string(),
                sx2[r].to_string(),
                sthr2[r].to_string(),
                sthr3[r].to_string(),
            ])
        })
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode stacked dataset");
    let colmap = ds.column_map();
    let xi_col = colmap["x"];
    let x2i_col = colmap["x2"];
    let thr2_col = colmap["thr2"];
    let thr3_col = colmap["thr3"];
    let n_headers = ds.headers.len();

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("z ~ s(x, bs='cc') + x2 + thr2 + thr3", &ds, &cfg)
        .expect("gam proportional-odds (stacked binomial) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard binomial GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluation grid spanning the x range; x2 fixed at 0 so the cumulative
    // probabilities trace the threshold + smooth structure cleanly. We build a
    // design row per (grid-x, cumulant) and read the logit-scale eta.
    let n_grid = 40usize;
    let grid_x: Vec<f64> = (0..n_grid)
        .map(|k| -3.0 + 6.0 * (k as f64) / ((n_grid - 1) as f64))
        .collect();

    // Helper: build the gam design at given (x, x2, thr2, thr3) rows and return
    // the logit-scale linear predictor eta = design * beta.
    let gam_eta = |xs: &[f64], x2s: &[f64], thr2s: &[f64], thr3s: &[f64]| -> Vec<f64> {
        let m = xs.len();
        let mut grid = Array2::<f64>::zeros((m, n_headers));
        for r in 0..m {
            grid[[r, xi_col]] = xs[r];
            grid[[r, x2i_col]] = x2s[r];
            grid[[r, thr2_col]] = thr2s[r];
            grid[[r, thr3_col]] = thr3s[r];
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild gam design at grid");
        design.design.apply(&fit.fit.beta).to_vec()
    };

    // Cumulative probabilities P(Y <= j | x, x2=0) for j = 1,2,3 over the grid.
    let mut gam_cum = vec![Vec::with_capacity(n_grid); cumulants.len()];
    for (jdx, &j) in cumulants.iter().enumerate() {
        let thr2 = if j >= 2.0 { 1.0 } else { 0.0 };
        let thr3 = if j >= 3.0 { 1.0 } else { 0.0 };
        let xs = grid_x.clone();
        let x2s = vec![0.0; n_grid];
        let thr2s = vec![thr2; n_grid];
        let thr3s = vec![thr3; n_grid];
        let eta = gam_eta(&xs, &x2s, &thr2s, &thr3s);
        for &e in &eta {
            gam_cum[jdx].push(inv_logit(e));
        }
    }
    // Per-level class probabilities P(Y = j): differences of the cumulants,
    // with P(Y<=0)=0 and P(Y<=4)=1 implicit.
    let mut gam_class = vec![Vec::with_capacity(n_grid); 4];
    for g in 0..n_grid {
        let c1 = gam_cum[0][g];
        let c2 = gam_cum[1][g];
        let c3 = gam_cum[2][g];
        gam_class[0].push(c1);
        gam_class[1].push(c2 - c1);
        gam_class[2].push(c3 - c2);
        gam_class[3].push(1.0 - c3);
    }

    // gam shared x2 slope on the logit scale: finite difference of eta wrt x2
    // at a representative x grid (proportional odds => slope is x-independent;
    // average over the grid for a stable read). Robust to coefficient ordering.
    let eta_lo = gam_eta(&grid_x, &vec![0.0; n_grid], &vec![0.0; n_grid], &vec![0.0; n_grid]);
    let eta_hi = gam_eta(&grid_x, &vec![1.0; n_grid], &vec![0.0; n_grid], &vec![0.0; n_grid]);
    let gam_x2_slope: f64 =
        eta_hi.iter().zip(&eta_lo).map(|(h, l)| h - l).sum::<f64>() / (n_grid as f64);

    // ---- fit the SAME proportional-odds likelihood in R (MASS::polr) -------
    // Reference predictor: cyclic harmonics sin(x)+cos(x) (linear analogue of
    // the cyclic-cubic smooth on one oscillation) + linear x2. polr returns
    // cumulative probabilities directly; clm cross-checks the coefficients.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(MASS))
        suppressPackageStartupMessages(library(ordinal))
        df$yf <- factor(round(df$y), levels = c(1,2,3,4), ordered = TRUE)
        df$sx <- sin(pi * df$x / 3)
        df$cx <- cos(pi * df$x / 3)
        m  <- polr(yf ~ sx + cx + x2, data = df, method = "logistic", Hess = TRUE)
        mc <- clm(yf ~ sx + cx + x2, data = df, link = "logit")

        # Shared x2 slope on the cumulative-logit scale (proportional odds).
        emit("polr_x2", as.numeric(coef(m)["x2"]))
        emit("clm_x2",  as.numeric(coef(mc)["x2"]))

        # Cumulative probabilities P(Y<=j | x, x2=0) on the SAME 40-pt x grid.
        gx <- seq(-3, 3, length.out = 40)
        nd <- data.frame(sx = sin(pi * gx / 3), cx = cos(pi * gx / 3), x2 = rep(0, length(gx)))
        pc <- predict(m, newdata = nd, type = "probs")  # n x 4 class probs
        cum1 <- pc[,1]
        cum2 <- pc[,1] + pc[,2]
        cum3 <- pc[,1] + pc[,2] + pc[,3]
        emit("cum1", as.numeric(cum1))
        emit("cum2", as.numeric(cum2))
        emit("cum3", as.numeric(cum3))
        emit("class1", as.numeric(pc[,1]))
        emit("class2", as.numeric(pc[,2]))
        emit("class3", as.numeric(pc[,3]))
        emit("class4", as.numeric(pc[,4]))
        "#,
    );

    let polr_x2 = r.scalar("polr_x2");
    let clm_x2 = r.scalar("clm_x2");
    let ref_cum = [r.vector("cum1"), r.vector("cum2"), r.vector("cum3")];
    let ref_class = [
        r.vector("class1"),
        r.vector("class2"),
        r.vector("class3"),
        r.vector("class4"),
    ];
    for v in ref_cum.iter() {
        assert_eq!(v.len(), n_grid, "polr cumulative grid length mismatch");
    }

    // ---- compare -----------------------------------------------------------
    // Cumulative probabilities: max-abs-diff per level (bound 0.02).
    let cum_mads: Vec<f64> = (0..cumulants.len())
        .map(|j| max_abs_diff(&gam_cum[j], ref_cum[j]))
        .collect();
    let cum_mad = cum_mads.iter().cloned().fold(0.0, f64::max);

    // Class probabilities: relative-L2 per level, take the worst (bound 0.025).
    let class_rels: Vec<f64> = (0..4)
        .map(|j| relative_l2(&gam_class[j], ref_class[j]))
        .collect();
    let class_rel = class_rels.iter().cloned().fold(0.0, f64::max);

    // Shared linear x2 slope: relative diff vs polr (bound 0.03), with clm as a
    // sanity cross-check that the two R references themselves agree.
    //
    // Sign convention: gam fits z = 1{Y<=j} with a "+predictor" cumulative
    // logit, so its slope is d/dx2 logit P(Y<=j) = +beta_cum. MASS::polr (and
    // ordinal::clm) parameterize logit P(Y<=j) = zeta_j - x*beta, reporting the
    // *latent-scale* coefficient = -beta_cum. So gam's cumulative slope must
    // match the NEGATED polr/clm coefficient. (Fitted probabilities are
    // invariant to this and need no sign handling.)
    let polr_cum_slope = -polr_x2;
    let clm_cum_slope = -clm_x2;
    let x2_rel = (gam_x2_slope - polr_cum_slope).abs() / polr_cum_slope.abs().max(1e-8);
    let ref_internal_rel = (polr_cum_slope - clm_cum_slope).abs() / polr_cum_slope.abs().max(1e-8);

    eprintln!(
        "ordinal proportional-odds: n={n} J=4 gam_edf={gam_edf:.3} \
         cum_mad={cum_mad:.4} (per-level {cum_mads:?}) \
         class_rel_l2={class_rel:.4} (per-level {class_rels:?}) \
         gam_x2_cum={gam_x2_slope:.4} polr_x2_cum={polr_cum_slope:.4} clm_x2_cum={clm_cum_slope:.4} \
         x2_rel={x2_rel:.4} polr_vs_clm_rel={ref_internal_rel:.5}"
    );

    // The two R references implement the identical model and must agree to
    // optimizer tolerance; if they don't, the comparison itself is unsound.
    assert!(
        ref_internal_rel < 0.01,
        "MASS::polr and ordinal::clm disagree on the x2 slope: \
         polr={polr_cum_slope:.5} clm={clm_cum_slope:.5} (rel={ref_internal_rel:.5})"
    );

    // gam (stacked binomial logit) and polr maximize the identical
    // proportional-odds likelihood on identical data, with sin/cos vs a
    // single-oscillation cyclic-cubic basis for g(x). The probability surfaces
    // must coincide to within sampling/basis slack; the spec bounds (0.02 on
    // cumulative probabilities, 0.025 relative on class probabilities) are
    // tight enough that any real divergence in gam's binomial/cyclic path
    // trips them.
    assert!(
        cum_mad < 0.02,
        "fitted cumulative probabilities diverge from MASS::polr: \
         max-abs-diff={cum_mad:.4} (per-level {cum_mads:?})"
    );
    assert!(
        class_rel < 0.025,
        "fitted class probabilities diverge from MASS::polr: \
         max relative-L2={class_rel:.4} (per-level {class_rels:?})"
    );
    // The proportional-odds slope is the shared, identifiable linear effect;
    // both engines must recover the same x2 coefficient (rel < 0.03).
    assert!(
        x2_rel < 0.03,
        "shared x2 slope disagrees with MASS::polr: \
         gam={gam_x2_slope:.4} polr(cum)={polr_cum_slope:.4} (rel={x2_rel:.4})"
    );
}
