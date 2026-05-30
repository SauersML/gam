//! End-to-end quality: gam's ordinal (continuation/stopping-ratio) regression
//! must agree with `VGAM::vglm(family = sratio(parallel = TRUE))` — the mature,
//! standard R reference for sequential/continuation-ratio ordinal models — and
//! is cross-checked against the same stopping-ratio likelihood. VGAM is *the*
//! canonical tool for the whole vector-GLM ordinal family (cumulative,
//! continuation, stopping, adjacent-category).
//!
//! gam has no bespoke "ordinal" family, but the **stopping-ratio** ordinal model
//! is, *exactly*, a binomial-logit GAM on a stacked dataset — and this is an
//! algebraic identity, not an approximation. For an observation with level Y=y
//! over J levels, the multinomial probability factorizes by the chain rule into
//! a product of conditional Bernoulli terms,
//!
//!     P(Y = y) = [∏_{j<y} (1 - q_j)] · q_y,   q_j = logit^{-1}(θ_j + g(x) + β·x2),
//!
//! where q_j = P(Y = j | Y ≥ j) is the stopping probability at cutpoint j (and
//! the last level needs no factor). Each factor is an independent Bernoulli for
//! the binary response z_j = 1{Y = j} evaluated only on the rows that *reached*
//! cutpoint j (Y ≥ j). Summing those independent Bernoulli log-likelihoods is
//! therefore *bit-for-bit* the multinomial log-likelihood — the Fienberg/Tutz
//! continuation-ratio factorization. (This independence identity is the property
//! that the *cumulative*-logit / proportional-odds model lacks, which is why we
//! model the stopping-ratio here, not cumulative logits.)
//!
//! So for every observation we emit, for each cutpoint j it reached, a binary
//! row z_j = 1{Y = j}; we share the covariate effects g(x) + β·x2 across cut-
//! points and let cutpoint-specific intercepts (threshold dummies thr2, thr3,
//! with j=1 the baseline) realize θ_2, θ_3. gam fits this stacked frame with
//!     z ~ s(x, bs='cc') + x2 + thr2 + thr3
//! and the SAME stopping-ratio likelihood is fit in R by VGAM. The data are
//! synthesized from an exact stopping-ratio generative model so both engines are
//! correctly specified.
//!
//! Because `VGAM::vglm` uses a *linear* predictor, the reference represents the
//! periodic x-effect with the matching low-order cyclic harmonics sin(x)+cos(x)
//! (the linear analogue of a cyclic-cubic smooth on a single oscillation), plus
//! the shared linear x2. Identical data (a fixed-seed synthetic ordinal sample)
//! is handed to both engines.
//!
//! We assert agreement on the quantities that actually matter for an ordinal
//! model, all over a common x-grid spanning the data:
//!   1. fitted per-level class probabilities P(Y = j): relative-L2,
//!   2. fitted cumulative probabilities P(Y <= j): max-abs-diff,
//!   3. the shared linear x2 effect (logit-scale slope): relative diff.
//! A genuine divergence here is a real bug in gam's binomial/cyclic machinery,
//! not a tolerance artifact.

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
/// fighting a boundary discontinuity. Its linear-model analogue (used by VGAM)
/// is the first cyclic harmonic pair sin(pi x/3) + cos(pi x/3).
fn g_of_x(x: f64) -> f64 {
    0.9 * (std::f64::consts::PI * x / 3.0).sin()
}

/// Logistic CDF (inverse logit).
fn inv_logit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[test]
fn gam_continuation_ratio_matches_vgam_sratio() {
    init_parallelism();

    // ---- synthesize a fixed-seed ordinal sample (J = 4 levels) -------------
    // EXACT stopping-ratio generative model: at each cutpoint j the observation
    // "stops" (Y = j) with prob q_j = logit^{-1}(theta_j + g(x) + beta*x2),
    // otherwise it advances; reaching the last level if it never stops. This is
    // precisely the model gam's stacked binomial and VGAM::sratio both fit, so
    // both are correctly specified. Identical raw (x, x2, y) handed to both.
    let n = 250usize;
    let beta_x2_true = 0.7;
    // Stopping-ratio cutpoint intercepts theta_j on the conditional-logit scale.
    let theta = [-0.85_f64, 0.0, 0.85];

    let mut rng = StdRng::seed_from_u64(20240529);
    let ux = Uniform::new(-3.0_f64, 3.0).expect("uniform x");
    let ux2 = Uniform::new(-1.0_f64, 1.0).expect("uniform x2");
    // Uniform draws compared against the conditional stopping probability q_j.
    let uunit = Uniform::new(0.0_f64, 1.0).expect("uniform unit");

    let mut x = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let x2i = ux2.sample(&mut rng);
        let shared = g_of_x(xi) + beta_x2_true * x2i;
        // Sequentially decide where to stop. Level in 1..=4.
        let mut level = 4.0;
        for (j, &th) in theta.iter().enumerate() {
            let q_j = inv_logit(th + shared); // P(Y = j+1 | Y >= j+1)
            if uunit.sample(&mut rng) < q_j {
                level = (j + 1) as f64;
                break;
            }
        }
        x.push(xi);
        x2.push(x2i);
        y.push(level);
    }

    // ---- fit gam: stopping-ratio via the equivalent stacked binomial -------
    // For each obs i, emit one binary row for every cutpoint j in {1,2,3} that
    // the obs REACHED (y_i >= j): response z = 1{y_i == j} ("stop here"), with
    // covariates x_i, x2_i and cutpoint dummies thr2 = 1{j>=2}, thr3 = 1{j>=3}.
    // Shared s(x,bs='cc') + x2; the threshold dummies are plain linear fixed
    // effects giving theta_2, theta_3 relative to the j=1 baseline intercept.
    // The summed Bernoulli log-likelihood over these rows IS the stopping-ratio
    // multinomial log-likelihood (chain-rule factorization).
    let cutpoints = [1.0_f64, 2.0, 3.0];
    let mut sx = Vec::new();
    let mut sx2 = Vec::new();
    let mut sthr2 = Vec::new();
    let mut sthr3 = Vec::new();
    let mut sz = Vec::new();
    for i in 0..n {
        for &j in &cutpoints {
            // Only rows that reached cutpoint j (Y >= j) enter the likelihood.
            if y[i] < j {
                continue;
            }
            sx.push(x[i]);
            sx2.push(x2[i]);
            sthr2.push(if j >= 2.0 { 1.0 } else { 0.0 });
            sthr3.push(if j >= 3.0 { 1.0 } else { 0.0 });
            sz.push(if (y[i] - j).abs() < 0.5 { 1.0 } else { 0.0 });
        }
    }
    let n_stack = sz.len();
    assert!(n_stack > n, "stacked frame should have >n conditional rows");

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
        .expect("gam stopping-ratio (stacked binomial) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard binomial GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluation grid spanning the x range; x2 fixed at 0 so the probabilities
    // trace the cutpoint + smooth structure cleanly. We build a design row per
    // (grid-x, cutpoint) and read the logit-scale eta = q_j on the conditional
    // (stopping) scale.
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

    // Conditional stopping probabilities q_j = P(Y = j | Y >= j | x, x2=0) for
    // j = 1,2,3 over the grid, then assemble unconditional class probabilities
    // by the chain rule.
    let mut gam_q = vec![Vec::with_capacity(n_grid); cutpoints.len()];
    for (jdx, &j) in cutpoints.iter().enumerate() {
        let thr2 = if j >= 2.0 { 1.0 } else { 0.0 };
        let thr3 = if j >= 3.0 { 1.0 } else { 0.0 };
        let xs = grid_x.clone();
        let x2s = vec![0.0; n_grid];
        let thr2s = vec![thr2; n_grid];
        let thr3s = vec![thr3; n_grid];
        let eta = gam_eta(&xs, &x2s, &thr2s, &thr3s);
        for &e in &eta {
            gam_q[jdx].push(inv_logit(e));
        }
    }
    // Per-level class probabilities P(Y = j) via the stopping-ratio chain rule:
    //   P(1) = q1, P(2) = (1-q1)q2, P(3) = (1-q1)(1-q2)q3, P(4) = ∏(1-q_j).
    let mut gam_class = vec![Vec::with_capacity(n_grid); 4];
    for g in 0..n_grid {
        let q1 = gam_q[0][g];
        let q2 = gam_q[1][g];
        let q3 = gam_q[2][g];
        gam_class[0].push(q1);
        gam_class[1].push((1.0 - q1) * q2);
        gam_class[2].push((1.0 - q1) * (1.0 - q2) * q3);
        gam_class[3].push((1.0 - q1) * (1.0 - q2) * (1.0 - q3));
    }
    // Cumulative probabilities P(Y <= j) for j = 1,2,3 (running sums).
    let mut gam_cum = vec![Vec::with_capacity(n_grid); cutpoints.len()];
    for g in 0..n_grid {
        let c1 = gam_class[0][g];
        let c2 = c1 + gam_class[1][g];
        let c3 = c2 + gam_class[2][g];
        gam_cum[0].push(c1);
        gam_cum[1].push(c2);
        gam_cum[2].push(c3);
    }

    // gam shared x2 slope on the conditional-logit scale: finite difference of
    // eta wrt x2 across the grid (parallel/shared slope => x-independent;
    // average for a stable read). Robust to coefficient ordering.
    let eta_lo = gam_eta(&grid_x, &vec![0.0; n_grid], &vec![0.0; n_grid], &vec![0.0; n_grid]);
    let eta_hi = gam_eta(&grid_x, &vec![1.0; n_grid], &vec![0.0; n_grid], &vec![0.0; n_grid]);
    let gam_x2_slope: f64 =
        eta_hi.iter().zip(&eta_lo).map(|(h, l)| h - l).sum::<f64>() / (n_grid as f64);

    // ---- fit the SAME stopping-ratio likelihood in R (VGAM::vglm) ----------
    // Reference predictor: cyclic harmonics sin(x)+cos(x) (linear analogue of
    // the cyclic-cubic smooth on one oscillation) + linear x2, with PARALLEL
    // (shared) slopes across cutpoints — exactly the model gam fits. sratio()
    // models logit P(Y = j | Y >= j), so its reported slope is on the same
    // conditional-logit scale and with the same sign as gam's stacked binomial.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("x2", &x2),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(VGAM))
        df$yf <- factor(round(df$y), levels = c(1,2,3,4))
        df$sx <- sin(pi * df$x / 3)
        df$cx <- cos(pi * df$x / 3)
        # Stopping-ratio with parallel (shared) covariate slopes across cutpoints.
        m <- vglm(yf ~ sx + cx + x2,
                  family = sratio(link = "logitlink", parallel = TRUE),
                  data = df)

        # Shared x2 slope on the conditional (stopping) logit scale. With
        # parallel=TRUE the x2 coefficient is a single shared value.
        cf <- coef(m)
        emit("vgam_x2", as.numeric(cf["x2"]))

        # Fitted class probabilities P(Y = j) on the SAME 40-pt x grid, x2 = 0.
        gx <- seq(-3, 3, length.out = 40)
        nd <- data.frame(sx = sin(pi * gx / 3), cx = cos(pi * gx / 3), x2 = rep(0, length(gx)))
        pc <- predict(m, newdata = nd, type = "response")  # n x 4 class probs
        emit("class1", as.numeric(pc[,1]))
        emit("class2", as.numeric(pc[,2]))
        emit("class3", as.numeric(pc[,3]))
        emit("class4", as.numeric(pc[,4]))
        emit("cum1", as.numeric(pc[,1]))
        emit("cum2", as.numeric(pc[,1] + pc[,2]))
        emit("cum3", as.numeric(pc[,1] + pc[,2] + pc[,3]))
        "#,
    );

    let vgam_x2 = r.scalar("vgam_x2");
    let ref_cum = [r.vector("cum1"), r.vector("cum2"), r.vector("cum3")];
    let ref_class = [
        r.vector("class1"),
        r.vector("class2"),
        r.vector("class3"),
        r.vector("class4"),
    ];
    for v in ref_cum.iter() {
        assert_eq!(v.len(), n_grid, "VGAM cumulative grid length mismatch");
    }
    for v in ref_class.iter() {
        assert_eq!(v.len(), n_grid, "VGAM class grid length mismatch");
    }

    // ---- compare -----------------------------------------------------------
    // Class probabilities: relative-L2 per level, take the worst.
    let class_rels: Vec<f64> = (0..4)
        .map(|j| relative_l2(&gam_class[j], ref_class[j]))
        .collect();
    let class_rel = class_rels.iter().cloned().fold(0.0, f64::max);

    // Cumulative probabilities: max-abs-diff per level.
    let cum_mads: Vec<f64> = (0..cutpoints.len())
        .map(|j| max_abs_diff(&gam_cum[j], ref_cum[j]))
        .collect();
    let cum_mad = cum_mads.iter().cloned().fold(0.0, f64::max);

    // Shared linear x2 slope: relative diff vs VGAM. Both engines fit the same
    // conditional-logit stopping model with the same sign convention (logit of
    // the per-cutpoint stop probability increases with +beta*x2), so the slopes
    // match directly with NO sign flip.
    let x2_rel = (gam_x2_slope - vgam_x2).abs() / vgam_x2.abs().max(1e-8);

    eprintln!(
        "ordinal stopping-ratio: n={n} n_stack={n_stack} J=4 gam_edf={gam_edf:.3} \
         class_rel_l2={class_rel:.4} (per-level {class_rels:?}) \
         cum_mad={cum_mad:.4} (per-level {cum_mads:?}) \
         gam_x2={gam_x2_slope:.4} vgam_x2={vgam_x2:.4} x2_rel={x2_rel:.4}"
    );

    // gam (stacked binomial logit) and VGAM::sratio maximize the IDENTICAL
    // stopping-ratio multinomial likelihood on identical data (the stacked
    // Bernoulli sum is bit-for-bit that likelihood). The only modeling
    // difference is gam's penalized cyclic-cubic basis for g(x) vs the reference's
    // two fixed harmonics; on data generated from one smooth oscillation these
    // span essentially the same function, so the probability surfaces agree to
    // basis/sampling slack. 0.02 max-abs on cumulative probabilities and 0.03
    // relative-L2 on class probabilities are tight enough that any real
    // divergence in gam's binomial/cyclic path trips them, yet leave margin for
    // the spline-vs-harmonic and REML-smoothing-vs-unpenalized gap.
    assert!(
        class_rel < 0.03,
        "fitted class probabilities diverge from VGAM::sratio: \
         max relative-L2={class_rel:.4} (per-level {class_rels:?})"
    );
    assert!(
        cum_mad < 0.02,
        "fitted cumulative probabilities diverge from VGAM::sratio: \
         max-abs-diff={cum_mad:.4} (per-level {cum_mads:?})"
    );
    // The shared (parallel) slope is the identifiable linear effect; both
    // engines must recover the same x2 coefficient. The smooth's penalty does
    // not touch the parametric x2 column, so this is the tightest comparison:
    // rel < 0.06 covers finite-sample estimation noise at n=250 while still
    // catching a genuine scale/sign error.
    assert!(
        x2_rel < 0.06,
        "shared x2 slope disagrees with VGAM::sratio: \
         gam={gam_x2_slope:.4} vgam={vgam_x2:.4} (rel={x2_rel:.4})"
    );
}
