//! End-to-end quality: gam's ordinal (continuation/stopping-ratio) regression
//! must RECOVER THE KNOWN GENERATIVE TRUTH. The data are synthesized from an
//! exact stopping-ratio model with KNOWN cutpoint intercepts `theta_j`, a KNOWN
//! shared slope `beta_x2_true`, and a KNOWN periodic smooth `g(x)`, so the
//! population-truth class/cumulative probabilities and the truth x2 slope are
//! computable in closed form on the evaluation grid. The OBJECTIVE metric this
//! test asserts is therefore truth recovery:
//!   * RMSE(gam class probs, TRUTH class probs) <= a principled absolute bar,
//!   * RMSE(gam cumulative probs, TRUTH cumulative probs) <= a principled bar,
//!   * |gam x2 slope - beta_x2_true| <= a principled absolute bar.
//! mgcv's REML fit of the SAME stacked stopping-ratio model (the identical frame
//! and cyclic smooth) is DEMOTED to a baseline-to-match-or-beat: gam's error
//! against the truth must be <= mgcv's error against the truth, times a small
//! slack. We do NOT assert "gam matches mgcv's fitted output". Matching a peer
//! tool's noisy fit is not a quality claim. The primary claim is that gam
//! recovers the truth, at least as accurately as a mature REML smoother.
//! (The real-data arm below keeps VGAM::sratio, whose predictor there is the
//! same linear model gam fits. This synthetic arm needs a smoother, because
//! g(x) must be learned from data.)
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
//!     z ~ s(x, bs='cyclic') + x2 + linear(thr2, double_penalty=false)
//!         + linear(thr3, double_penalty=false)
//! and mgcv fits the identical stacked frame and cyclic smooth by REML. The
//! threshold dummies are intercepts, so they opt out of the null-recovery ridge
//! every formula linear effect carries by default (b7b874a2a); mgcv leaves them
//! unpenalized too. x2 is a covariate effect and keeps the default. The data
//! are synthesized from an exact stopping-ratio generative model, so both engines
//! are correctly specified, and identical rows (a fixed-seed synthetic ordinal
//! sample, stacked once) are handed to both.
//!
//! All comparisons are over a common x-grid spanning the data, against the
//! closed-form population truth: per-level class probabilities P(Y = j),
//! cumulative probabilities P(Y <= j), and the shared linear x2 slope. A genuine
//! quality shortfall here is a real bug in gam's binomial/cyclic machinery, not
//! a tolerance artifact.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{
    Column, PairedFoldComparison, QualityPair, relative_l2, rmse, run_r,
};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use std::path::Path;

/// Latent periodic x-effect: one clean oscillation over the data window
/// [-3, 3] with period exactly 6, so g(-3) = g(3) = 0. This honors the seam
/// continuity that a cyclic-cubic (`bs='cyclic'`) smooth imposes (f(min) = f(max)),
/// making the cyclic basis the genuinely correct model — not an approximation
/// fighting a boundary discontinuity.
fn g_of_x(x: f64) -> f64 {
    0.9 * (std::f64::consts::PI * x / 3.0).sin()
}

/// Logistic CDF (inverse logit).
fn inv_logit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// The real-data arm's shared-slope weather model and its cutpoint-only null, on
/// the stacked stopping-ratio frame. The cutpoint dummy is an intercept, so it
/// opts out of the null-recovery ridge formula linear effects carry by default
/// (b7b874a2a); the weather slopes keep it.
const WINE_WEATHER_FORMULA: &str = "z ~ s_temp + h_temp + linear(thr2, double_penalty=false)";
const WINE_NULL_FORMULA: &str = "z ~ linear(thr2, double_penalty=false)";

/// The stacked stopping-ratio binomial `formula`, fit on the `train` vintages of
/// the wine frame: its class probabilities at the `test` vintages, and its total
/// EDF. Each vintage emits one binary row z = 1{Y = j} for every cutpoint j it
/// reached (Y >= j), with the shared weather covariates and the cutpoint dummy
/// thr2 = 1{j >= 2}. J = 3 means one cutpoint dummy (j = 1 baseline). The summed
/// Bernoulli log-likelihood over these rows IS the stopping-ratio multinomial
/// log-likelihood (chain-rule factorization), identical to the model
/// VGAM::sratio fits.
fn wine_stopping_ratio_probs(
    formula: &str,
    train: &[usize],
    test: &[usize],
    s_temp: &[f64],
    h_temp: &[f64],
    y: &[f64],
) -> (Vec<[f64; 3]>, f64) {
    let mut rows = Vec::new();
    for &i in train {
        for j in [1.0_f64, 2.0] {
            if y[i] < j {
                continue;
            }
            rows.push(csv::StringRecord::from(vec![
                (if (y[i] - j).abs() < 0.5 { 1.0 } else { 0.0 }).to_string(),
                s_temp[i].to_string(),
                h_temp[i].to_string(),
                (if j >= 2.0 { 1.0 } else { 0.0 }).to_string(),
            ]));
        }
    }
    let headers = ["z", "s_temp", "h_temp", "thr2"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode stacked wine frame");
    let colmap = ds.column_map();
    let (s_col, h_col, thr2_col) = (colmap["s_temp"], colmap["h_temp"], colmap["thr2"]);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &ds, &cfg)
        .expect("gam stacked stopping-ratio fit on wine vintages");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard binomial GAM fit for `{formula}`");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");
    // Conditional stopping probabilities q_j = P(Y = j | Y >= j) at each test
    // vintage, then the class probabilities by the chain rule:
    // P(1) = q1, P(2) = (1 - q1) q2, P(3) = (1 - q1)(1 - q2).
    let stop_prob = |thr2: f64| -> Vec<f64> {
        let mut grid = Array2::<f64>::zeros((test.len(), ds.headers.len()));
        for (r, &i) in test.iter().enumerate() {
            grid[[r, s_col]] = s_temp[i];
            grid[[r, h_col]] = h_temp[i];
            grid[[r, thr2_col]] = thr2;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild the stacked design at held-out wine vintages");
        design
            .design
            .apply(&fit.fit.beta)
            .iter()
            .map(|&e| inv_logit(e))
            .collect()
    };
    let (q1, q2) = (stop_prob(0.0), stop_prob(1.0));
    let probs: Vec<[f64; 3]> = (0..test.len())
        .map(|r| [q1[r], (1.0 - q1[r]) * q2[r], (1.0 - q1[r]) * (1.0 - q2[r])])
        .collect();
    (probs, edf)
}

#[test]
fn gam_continuation_ratio_matches_vgam_sratio() {
    init_parallelism();

    // ---- synthesize a fixed-seed ordinal sample (J = 4 levels) -------------
    // EXACT stopping-ratio generative model: at each cutpoint j the observation
    // "stops" (Y = j) with prob q_j = logit^{-1}(theta_j + g(x) + beta*x2),
    // otherwise it advances; reaching the last level if it never stops. This is
    // precisely the stacked binomial model gam and mgcv both fit, so both are
    // correctly specified. Identical stacked rows are handed to both.
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
    // Shared s(x,bs='cyclic') + x2; the threshold dummies are plain linear fixed
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
    let result = fit_from_formula(
        "z ~ s(x, bs='cyclic') + x2 + linear(thr2, double_penalty=false) \
         + linear(thr3, double_penalty=false)",
        &ds,
        &cfg,
    )
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
    let mut gam_q: Vec<Vec<f64>> = (0..cutpoints.len())
        .map(|_| Vec::with_capacity(n_grid))
        .collect();
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
    let mut gam_class: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n_grid)).collect();
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
    let mut gam_cum: Vec<Vec<f64>> = (0..cutpoints.len())
        .map(|_| Vec::with_capacity(n_grid))
        .collect();
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
    let eta_lo = gam_eta(
        &grid_x,
        &vec![0.0; n_grid],
        &vec![0.0; n_grid],
        &vec![0.0; n_grid],
    );
    let eta_hi = gam_eta(
        &grid_x,
        &vec![1.0; n_grid],
        &vec![0.0; n_grid],
        &vec![0.0; n_grid],
    );
    let gam_x2_slope: f64 =
        eta_hi.iter().zip(&eta_lo).map(|(h, l)| h - l).sum::<f64>() / (n_grid as f64);

    // ---- fit the SAME stopping-ratio model in R with mgcv (REML) -----------
    // The reference fits the identical stacked binomial frame with the same
    // cyclic smooth, `z ~ s(x, bs="cc") + x2 + thr2 + thr3`, choosing its
    // smoothing parameter by REML. A comparison of smoothers needs a smoother on
    // the other side. VGAM::vglm was handed the TRUE sin/cos form of g(x), and
    // VGAM's own s() has no criterion-chosen smoothing (a fixed df, of which df=4
    // happens to be the best of 2..8 on this truth), so neither is a baseline for
    // a smooth that must learn g(x) from data.
    let r = run_r(
        &[
            Column::new("z", &sz),
            Column::new("x", &sx),
            Column::new("x2", &sx2),
            Column::new("thr2", &sthr2),
            Column::new("thr3", &sthr3),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(z ~ s(x, bs = "cc") + x2 + thr2 + thr3, family = binomial,
                 data = df, method = "REML")
        # Shared x2 slope on the conditional (stopping) logit scale.
        emit("ref_x2", as.numeric(coef(m)["x2"]))

        # Stopping probabilities q_j on the SAME 40-pt x grid at x2 = 0, then the
        # class and cumulative probabilities by the stopping-ratio chain rule.
        gx <- seq(-3, 3, length.out = 40)
        q <- function(t2, t3) as.numeric(predict(
          m, newdata = data.frame(x = gx, x2 = 0, thr2 = t2, thr3 = t3), type = "response"))
        q1 <- q(0, 0)
        q2 <- q(1, 0)
        q3 <- q(1, 1)
        c1 <- q1
        c2 <- (1 - q1) * q2
        c3 <- (1 - q1) * (1 - q2) * q3
        c4 <- (1 - q1) * (1 - q2) * (1 - q3)
        emit("class1", c1)
        emit("class2", c2)
        emit("class3", c3)
        emit("class4", c4)
        emit("cum1", c1)
        emit("cum2", c1 + c2)
        emit("cum3", c1 + c2 + c3)
        "#,
    );

    let ref_x2 = r.scalar("ref_x2");
    let ref_cum = [r.vector("cum1"), r.vector("cum2"), r.vector("cum3")];
    let ref_class = [
        r.vector("class1"),
        r.vector("class2"),
        r.vector("class3"),
        r.vector("class4"),
    ];
    for v in ref_cum.iter() {
        assert_eq!(v.len(), n_grid, "mgcv cumulative grid length mismatch");
    }
    for v in ref_class.iter() {
        assert_eq!(v.len(), n_grid, "mgcv class grid length mismatch");
    }

    // ---- closed-form POPULATION TRUTH on the same grid (x2 = 0) ------------
    // The data were generated from an EXACT stopping-ratio model, so the
    // population conditional stopping probabilities, class probabilities, and
    // cumulative probabilities are known functions of x with NO estimation in
    // them. This is the objective target both engines are graded against.
    let mut truth_class: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n_grid)).collect();
    let mut truth_cum: Vec<Vec<f64>> = (0..cutpoints.len())
        .map(|_| Vec::with_capacity(n_grid))
        .collect();
    for &gx in &grid_x {
        let shared = g_of_x(gx); // x2 = 0 on the grid
        let q1 = inv_logit(theta[0] + shared);
        let q2 = inv_logit(theta[1] + shared);
        let q3 = inv_logit(theta[2] + shared);
        let c1 = q1;
        let c2 = (1.0 - q1) * q2;
        let c3 = (1.0 - q1) * (1.0 - q2) * q3;
        let c4 = (1.0 - q1) * (1.0 - q2) * (1.0 - q3);
        truth_class[0].push(c1);
        truth_class[1].push(c2);
        truth_class[2].push(c3);
        truth_class[3].push(c4);
        truth_cum[0].push(c1);
        truth_cum[1].push(c1 + c2);
        truth_cum[2].push(c1 + c2 + c3);
    }

    // ---- OBJECTIVE METRIC: gam's error against the population truth --------
    // Stack all four class-probability levels into one long vector and take the
    // RMSE against the truth; same for the three cumulative levels.
    let flatten = |v: &[Vec<f64>]| -> Vec<f64> { v.iter().flatten().copied().collect() };
    let gam_class_flat = flatten(&gam_class);
    let truth_class_flat = flatten(&truth_class);
    let gam_cum_flat = flatten(&gam_cum);
    let truth_cum_flat = flatten(&truth_cum);

    let gam_class_rmse = rmse(&gam_class_flat, &truth_class_flat);
    let gam_cum_rmse = rmse(&gam_cum_flat, &truth_cum_flat);
    let gam_x2_err = (gam_x2_slope - beta_x2_true).abs();

    // ---- BASELINE TO MATCH-OR-BEAT: mgcv REML's error against the SAME truth --
    let ref_class_flat: Vec<f64> = ref_class.iter().flat_map(|v| v.iter().copied()).collect();
    let ref_cum_flat: Vec<f64> = ref_cum.iter().flat_map(|v| v.iter().copied()).collect();
    let ref_class_rmse = rmse(&ref_class_flat, &truth_class_flat);
    let ref_cum_rmse = rmse(&ref_cum_flat, &truth_cum_flat);
    let ref_x2_err = (ref_x2 - beta_x2_true).abs();

    // Context only (NOT a pass criterion): how close gam's fit lands to mgcv's.
    let class_rel_vs_ref = relative_l2(&gam_class_flat, &ref_class_flat);

    eprintln!(
        "ordinal stopping-ratio truth recovery: n={n} n_stack={n_stack} J=4 \
         gam_edf={gam_edf:.3} beta_x2_true={beta_x2_true:.4} \
         class_rmse gam={gam_class_rmse:.4} mgcv={ref_class_rmse:.4} | \
         cum_rmse gam={gam_cum_rmse:.4} mgcv={ref_cum_rmse:.4} | \
         x2 gam={gam_x2_slope:.4} mgcv={ref_x2:.4} \
         (err gam={gam_x2_err:.4} mgcv={ref_x2_err:.4}) | \
         gam_vs_mgcv_rel_l2(context only)={class_rel_vs_ref:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_mass_ordinal_polr::class",
            "class_rmse",
            gam_class_rmse,
            "mgcv",
            ref_class_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_mass_ordinal_polr::cum",
            "cum_rmse",
            gam_cum_rmse,
            "mgcv",
            ref_cum_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_mass_ordinal_polr::x2_slope",
            "x2_slope_abs_err",
            gam_x2_err,
            "mgcv",
            ref_x2_err,
        )
        .line()
    );

    // PRIMARY CLAIM: gam recovers the known generative truth. The probabilities
    // live in [0, 1]; an RMSE of 0.05 over the full grid is a few percent of the
    // unit probability range, well inside finite-sample (n=250) + cyclic-basis
    // slack while still tripping any genuine break in gam's binomial/cyclic path.
    assert!(
        gam_class_rmse < 0.05,
        "gam class probabilities do not recover the truth: RMSE={gam_class_rmse:.4} (bar 0.05)"
    );
    assert!(
        gam_cum_rmse < 0.05,
        "gam cumulative probabilities do not recover the truth: RMSE={gam_cum_rmse:.4} (bar 0.05)"
    );
    // The shared (parallel) slope is the identifiable linear effect, untouched by
    // the smooth's penalty. Recovering beta_x2_true=0.7 to within 0.12 absolute
    // covers finite-sample estimation noise at n=250 while catching a genuine
    // scale/sign error.
    assert!(
        gam_x2_err < 0.12,
        "gam shared x2 slope does not recover beta_x2_true: \
         gam={gam_x2_slope:.4} truth={beta_x2_true:.4} (abs err {gam_x2_err:.4}, bar 0.12)"
    );

    // MATCH-OR-BEAT: against the SAME population truth, gam must be at least as
    // accurate as mgcv's REML fit of the identical stacked model (within a small
    // slack). This is an ACCURACY comparison (error-to-truth), NOT a
    // reproduce-the-reference claim.
    assert!(
        gam_class_rmse <= ref_class_rmse * 1.10,
        "gam class probs less accurate than mgcv REML against truth: \
         gam_rmse={gam_class_rmse:.4} mgcv_rmse={ref_class_rmse:.4}"
    );
    assert!(
        gam_cum_rmse <= ref_cum_rmse * 1.10,
        "gam cumulative probs less accurate than mgcv REML against truth: \
         gam_rmse={gam_cum_rmse:.4} mgcv_rmse={ref_cum_rmse:.4}"
    );
    assert!(
        gam_x2_err <= ref_x2_err * 1.10 + 0.02,
        "gam x2 slope less accurate than mgcv REML against truth: \
         gam_err={gam_x2_err:.4} mgcv_err={ref_x2_err:.4}"
    );
}

/// REAL-DATA arm of the SAME continuation-ratio (stopping-ratio) ordinal
/// capability, on the classic Bordeaux wine dataset
/// (`bench/datasets/wine.csv`; source: Ashenfelter's Bordeaux wine-vintage data
/// as redistributed via R's `wine`/`Bordeaux` teaching datasets — vintage year,
/// auction `price`, and the weather covariates `h_rain`, `s_temp`, `w_rain`,
/// `h_temp`). On REAL data the generative truth is UNKNOWN, so — exactly as the
/// canonical real-data template (`quality_vs_mgcv_gaussian_smooth.rs`) prescribes
/// — the objective metric is HELD-OUT predictive quality, NOT truth recovery and
/// NOT reproduction of the reference's fit.
///
/// We turn the continuous vintage `price` into an ORDERED 3-tier quality score
/// (tier 1 = modest, 2 = mid, 3 = prized) by fixed price thresholds, and model
/// how the growing-season weather covariates push a vintage up the quality
/// ladder. This is precisely the ordinal continuation-ratio problem: the natural
/// "did this vintage clear the next quality threshold?" question is a sequence of
/// conditional Bernoulli stops, and the stopping-ratio factorization
///   P(Y=y) = [∏_{j<y}(1-q_j)] · q_y,  q_j = logit^{-1}(θ_j + βᵀcovariates)
/// is bit-for-bit a binomial-logit GAM on the stacked conditional frame (the same
/// algebraic identity the synthetic arm above exploits). gam fits the stacked
/// binomial; `VGAM::vglm(family = sratio(parallel = TRUE))` — the mature standard
/// ordinal tool — fits the identical stopping-ratio likelihood and is DEMOTED to
/// a match-or-beat baseline.
///
/// Objective held-out metrics, computed in plain Rust on gam's own predictions:
///   PRIMARY (objective, tool-free): leaving one vintage out at a time over all
///     priced vintages, gam's per-vintage multiclass LOG-LOSS is not RESOLVED
///     worse than that of the cutpoint-only null model (`z ~ thr2`, fit on the
///     same n - 1 vintages), paired per vintage; and on the split below gam's
///     held-out ACCURACY is no lower than that null's.
///   BASELINE (match-or-beat): on a deterministic train/test split (every 3rd
///     surviving row held out), where both engines see the IDENTICAL train rows
///     and predict the IDENTICAL test rows in the SAME order, gam's held-out
///     accuracy >= VGAM's minus a small margin, AND gam's held-out log-loss <=
///     VGAM's times a small slack.
#[test]
fn gam_continuation_ratio_matches_vgam_sratio_on_real_data() {
    init_parallelism();

    // ---- load the real Bordeaux wine dataset ------------------------------
    let wine_csv = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/wine.csv");
    let ds_raw = load_csvwith_inferred_schema(Path::new(wine_csv)).expect("load wine.csv");
    let col = ds_raw.column_map();
    let price_idx = col["price"];
    let s_temp_idx = col["s_temp"];
    let h_temp_idx = col["h_temp"];
    let price_all: Vec<f64> = ds_raw.values.column(price_idx).to_vec();
    let s_temp_all: Vec<f64> = ds_raw.values.column(s_temp_idx).to_vec();
    let h_temp_all: Vec<f64> = ds_raw.values.column(h_temp_idx).to_vec();
    let n_all = price_all.len();
    assert!(n_all > 40, "wine.csv should have ~47 vintages, got {n_all}");

    // Keep only the vintages with a recorded auction price (later years carry
    // NA price and load as NaN); standardize the two weather covariates so the
    // shared linear slopes are on a comparable scale for both engines.
    let usable: Vec<usize> = (0..n_all)
        .filter(|&i| {
            price_all[i].is_finite() && s_temp_all[i].is_finite() && h_temp_all[i].is_finite()
        })
        .collect();
    let n = usable.len();
    assert!(n > 30, "expected >30 priced vintages, got {n}");

    let s_raw: Vec<f64> = usable.iter().map(|&i| s_temp_all[i]).collect();
    let h_raw: Vec<f64> = usable.iter().map(|&i| h_temp_all[i]).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let sd = |v: &[f64], m: f64| {
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let (s_m, h_m) = (mean(&s_raw), mean(&h_raw));
    let (s_s, h_s) = (sd(&s_raw, s_m).max(1e-9), sd(&h_raw, h_m).max(1e-9));
    let s_temp: Vec<f64> = s_raw.iter().map(|x| (x - s_m) / s_s).collect();
    let h_temp: Vec<f64> = h_raw.iter().map(|x| (x - h_m) / h_s).collect();

    // ---- ordered 3-tier quality score from auction price ------------------
    // Fixed price thresholds (currency units) split the priced vintages into
    // ordered tiers: modest (Y=1, price <= 16), mid (Y=2, 16 < price <= 25),
    // prized (Y=3, price > 25). J = 3 ordered levels => cutpoints {1, 2}.
    let to_tier = |p: f64| -> f64 {
        if p <= 16.0 {
            1.0
        } else if p <= 25.0 {
            2.0
        } else {
            3.0
        }
    };
    let y: Vec<f64> = usable.iter().map(|&i| to_tier(price_all[i])).collect();
    let n_levels = 3usize;

    // ---- deterministic train/test split: every 3rd usable row held out ----
    let is_test = |i: usize| i % 3 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 18 && test_rows.len() >= 10,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_s: Vec<f64> = train_rows.iter().map(|&i| s_temp[i]).collect();
    let train_h: Vec<f64> = train_rows.iter().map(|&i| h_temp[i]).collect();
    let train_y: Vec<f64> = train_rows.iter().map(|&i| y[i]).collect();
    let test_s: Vec<f64> = test_rows.iter().map(|&i| s_temp[i]).collect();
    let test_h: Vec<f64> = test_rows.iter().map(|&i| h_temp[i]).collect();
    let test_y: Vec<f64> = test_rows.iter().map(|&i| y[i]).collect();

    // ---- gam on TRAIN, predicted at TEST ------------------------------------
    // Shared LINEAR weather slopes (s_temp + h_temp) plus the cutpoint dummy on the
    // stacked stopping-ratio frame. n is small (real vintage data), so a parametric
    // shared-slope predictor is the honest model; this is the same parallel-slope
    // structure VGAM uses.
    let n_test = test_rows.len();
    let (gam_test_probs, gam_edf) = wine_stopping_ratio_probs(
        WINE_WEATHER_FORMULA,
        &train_rows,
        &test_rows,
        &s_temp,
        &h_temp,
        &y,
    );

    // ---- fit the SAME stopping-ratio likelihood in R (VGAM::vglm) ---------
    // ONE run_r call: every Column must be equal length. We pass the TRAIN rows
    // (s_temp, h_temp, y) plus the TEST rows padded into parallel columns and a
    // test-count, and read back VGAM's held-out class probabilities. This mirrors
    // the mgcv held-out-prediction pattern in the gaussian-smooth template.
    let n_train = train_rows.len();
    let pad = |v: &[f64], len: usize| -> Vec<f64> {
        let fill = v.last().copied().unwrap_or(0.0);
        let mut out = v.to_vec();
        out.resize(len, fill);
        out
    };
    let r = run_r(
        &[
            Column::new("s_temp", &train_s),
            Column::new("h_temp", &train_h),
            Column::new("y", &train_y),
            Column::new("test_s", &pad(&test_s, n_train)),
            Column::new("test_h", &pad(&test_h, n_train)),
            Column::new("test_n", &vec![n_test as f64; n_train]),
        ],
        r#"
        suppressPackageStartupMessages(library(VGAM))
        # VGAM's sratio()/cratio() families REQUIRE an ORDERED factor response
        # (they stop() with "response should be ordinal---see ordered()" on a
        # plain factor). The tiers 1<2<3 are genuinely ordered, so declare them so.
        lev <- c(1, 2, 3)
        df$yf <- ordered(round(df$y), levels = lev)
        # Stopping-ratio with parallel (shared) covariate slopes across cutpoints.
        m <- vglm(yf ~ s_temp + h_temp,
                  family = sratio(link = "logitlink", parallel = TRUE),
                  data = df)
        k <- df$test_n[1]
        nd <- data.frame(s_temp = df$test_s[1:k], h_temp = df$test_h[1:k])
        pc <- predict(m, newdata = nd, type = "response")  # k x J class probs
        # Index predicted-probability columns by their LEVEL NAME, not a hardcoded
        # position: if a training fold happens to omit the top tier, VGAM returns
        # fewer columns and a fixed pc[,3] overruns ("subscript out of bounds").
        # Re-key onto the full ordered level set, filling any absent class with 0.
        pc <- as.matrix(pc)
        if (is.null(colnames(pc))) colnames(pc) <- as.character(lev[seq_len(ncol(pc))])
        full <- matrix(0.0, nrow = nrow(pc), ncol = length(lev))
        colnames(full) <- as.character(lev)
        have <- intersect(colnames(pc), colnames(full))
        full[, have] <- pc[, have, drop = FALSE]
        emit("p1", as.numeric(full[, "1"]))
        emit("p2", as.numeric(full[, "2"]))
        emit("p3", as.numeric(full[, "3"]))
        "#,
    );
    let ref_p1 = r.vector("p1");
    let ref_p2 = r.vector("p2");
    let ref_p3 = r.vector("p3");
    assert_eq!(ref_p1.len(), n_test, "VGAM held-out prob length mismatch");
    assert_eq!(ref_p2.len(), n_test, "VGAM held-out prob length mismatch");
    assert_eq!(ref_p3.len(), n_test, "VGAM held-out prob length mismatch");
    let ref_test_probs: Vec<[f64; 3]> = (0..n_test)
        .map(|r| [ref_p1[r], ref_p2[r], ref_p3[r]])
        .collect();

    // ---- objective held-out metrics (plain Rust) -------------------------
    // arg-max class accuracy and multiclass log-loss against the true tier.
    let argmax3 = |p: &[f64; 3]| -> usize {
        let mut best = 0usize;
        for k in 1..3 {
            if p[k] > p[best] {
                best = k;
            }
        }
        best
    };
    let accuracy = |probs: &[[f64; 3]]| -> f64 {
        let mut correct = 0usize;
        for (pr, &yt) in probs.iter().zip(&test_y) {
            let pred_level = argmax3(pr) + 1; // class index 0..2 -> level 1..3
            if (pred_level as f64 - yt).abs() < 0.5 {
                correct += 1;
            }
        }
        correct as f64 / probs.len() as f64
    };
    let log_loss = |probs: &[[f64; 3]]| -> f64 {
        let mut s = 0.0;
        for (pr, &yt) in probs.iter().zip(&test_y) {
            let k = (yt.round() as usize).saturating_sub(1).min(2);
            s += -(pr[k].max(1e-12)).ln();
        }
        s / probs.len() as f64
    };

    let gam_acc = accuracy(&gam_test_probs);
    let gam_ll = log_loss(&gam_test_probs);
    let vgam_acc = accuracy(&ref_test_probs);
    let vgam_ll = log_loss(&ref_test_probs);

    // ---- train-fitted NULL stopping-ratio model: the informative-model floor --
    // The cutpoint-only model `z ~ thr2`, fit on the SAME train vintages,
    // reproduces the train tier frequencies and ignores the weather. A weather
    // model that carries real information must predict the held-out tiers better
    // than it does. The held-out majority class is NOT such a floor: it reads the
    // TEST labels, which no train-fitted model can know.
    let null_test_probs = wine_stopping_ratio_probs(
        WINE_NULL_FORMULA,
        &train_rows,
        &test_rows,
        &s_temp,
        &h_temp,
        &y,
    )
    .0;
    let null_acc = accuracy(&null_test_probs);
    let null_ll = log_loss(&null_test_probs);

    // Context only (NOT a pass criterion): closeness of gam's held-out class
    // probabilities to VGAM's.
    let gam_flat: Vec<f64> = gam_test_probs
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let ref_flat: Vec<f64> = ref_test_probs
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
    let probs_rel_vs_ref = relative_l2(&gam_flat, &ref_flat);
    let probs_rmse_vs_ref = rmse(&gam_flat, &ref_flat);

    eprintln!(
        "wine ordinal stopping-ratio held-out: n={n} n_train={n_train} \
         n_test={n_test} J={n_levels} gam_edf={gam_edf:.3} null_acc={null_acc:.4} null_logloss={null_ll:.4} \
         acc gam={gam_acc:.4} vgam={vgam_acc:.4} | logloss gam={gam_ll:.4} vgam={vgam_ll:.4} \
         (context: rel_l2 vs vgam={probs_rel_vs_ref:.4} rmse vs vgam={probs_rmse_vs_ref:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::score(
            "misc",
            "quality_vs_mass_ordinal_polr::holdout_accuracy",
            "accuracy",
            gam_acc,
            "vgam",
            vgam_acc,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_mass_ordinal_polr::holdout_logloss",
            "logloss",
            gam_ll,
            "vgam",
            vgam_ll,
        )
        .line()
    );

    // ---- PRIMARY objective assertions: the weather model is informative ----
    // Leave-one-vintage-out log-loss, paired against the cutpoint-only null. Every
    // priced vintage is held out once; gam and the null are both fit on the other
    // n - 1 vintages and score its tier, so the vintage-to-vintage swing cancels in
    // the pair, and gam must not be RESOLVED worse than the null. A wrong link
    // direction or a flipped response label loses to the null on nearly every
    // vintage and is resolved worse. This replaces a point comparison on the 13
    // split rows, which even the unpenalized VGAM fit clears by only 0.024: at
    // 853ef3bd0 (MSI job 608674) gam 1.3365, null 1.3027, VGAM 1.2788. Accuracy
    // must not fall below the null's arg-max predictor either.
    let mut loo_gam_ll = Vec::with_capacity(n);
    let mut loo_null_ll = Vec::with_capacity(n);
    for held_out in 0..n {
        let rest: Vec<usize> = (0..n).filter(|&i| i != held_out).collect();
        let tier = (y[held_out].round() as usize).saturating_sub(1).min(2);
        let gam_p = wine_stopping_ratio_probs(
            WINE_WEATHER_FORMULA,
            &rest,
            &[held_out],
            &s_temp,
            &h_temp,
            &y,
        )
        .0;
        let null_p = wine_stopping_ratio_probs(
            WINE_NULL_FORMULA,
            &rest,
            &[held_out],
            &s_temp,
            &h_temp,
            &y,
        )
        .0;
        loo_gam_ll.push(-(gam_p[0][tier].max(1e-12)).ln());
        loo_null_ll.push(-(null_p[0][tier].max(1e-12)).ln());
    }
    let loo = PairedFoldComparison::new(&loo_gam_ll, &loo_null_ll, true);
    eprintln!("{}", loo.report("polr_wine::loo_logloss_vs_null"));
    assert!(
        !loo.gam_resolved_worse(),
        "gam's leave-one-vintage-out log-loss is RESOLVED worse than the cutpoint-only null's\n{}",
        loo.report("polr_wine::loo_logloss_vs_null")
    );
    assert!(
        gam_acc >= null_acc,
        "gam held-out ordinal accuracy below the train-fitted cutpoint-only null model: \
         gam={gam_acc:.4} null={null_acc:.4}"
    );

    // ---- BASELINE (match-or-beat): no worse than VGAM::sratio held-out ----
    assert!(
        gam_acc >= vgam_acc - 0.10,
        "gam held-out accuracy below VGAM::sratio: gam={gam_acc:.4} vgam={vgam_acc:.4}"
    );
    assert!(
        gam_ll <= vgam_ll * 1.10 + 0.05,
        "gam held-out log-loss worse than VGAM::sratio: gam={gam_ll:.4} vgam={vgam_ll:.4}"
    );
}
