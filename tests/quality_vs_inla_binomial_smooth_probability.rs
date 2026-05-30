//! End-to-end OBJECTIVE quality: gam's REML/Laplace penalized smooth under the
//! **binomial** family (logit link) must RECOVER THE TRUE latent probability
//! function that generated the data — on the original (0,1) response scale —
//! and do so at least as accurately as **R-INLA**, the mature, standard
//! integrated-nested-Laplace-approximation engine, which is used here only as a
//! baseline-to-match-or-beat (never as the thing gam is required to reproduce).
//!
//! Objective metric asserted (primary, truth recovery):
//!   RMSE(gam_prob, p_true) over the training covariate, where
//!   `p_true(age) = logit^{-1}( 1.4*sin((age-30)/13) - 0.9 )` is the KNOWN
//!   smooth latent probability that the synthetic Bernoulli response was drawn
//!   from. We require this error to be a small fraction of the probability
//!   signal's own range (the spread of `p_true`), i.e. gam reconstructs the
//!   generating curve rather than chasing the Bernoulli noise. This is a pure
//!   correctness claim about gam, independent of any reference tool.
//!
//! Match-or-beat baseline (accuracy, secondary): we additionally fit the SAME
//! data with R-INLA (canonical binomial penalized smooth, `f(age, model="rw2",
//! scale.model=TRUE)`) and require gam's RMSE-to-truth to be no worse than
//! INLA's by more than a 10% margin. INLA is the incumbent; gam may not be
//! meaningfully less accurate at recovering the truth. We do NOT assert gam
//! reproduces INLA's fitted output — only that gam is at least as close to the
//! GROUND TRUTH.
//!
//! Calibration (uncertainty, secondary, against truth): gam's delta-method
//! posterior SD on the probability scale must give honest coverage. We check
//! that the +/- 2 SD band around gam's fitted probability covers `p_true` for
//! close to the nominal fraction of training points (a one-sided lower bar on
//! empirical coverage), so the reported uncertainty is not anti-conservatively
//! narrow. This is coverage of the TRUE curve, not agreement with INLA's SD.
//!
//! Why this is the right rework. The previous version asserted
//! `rel_l2(gam_prob, inla_prob)` and `|gam_SD - inla_SD|` were small — i.e. that
//! gam reproduces INLA's (itself approximate, noisy) fit. Matching a peer tool
//! proves nothing about quality: both could overfit the Bernoulli draws in the
//! same direction. Because the data has a KNOWN generating probability, the
//! honest quality question is "does gam recover p_true?", which we now assert
//! directly, with INLA demoted to an accuracy baseline on that same truth.
//!
//! Data. The Haberman breast-cancer study (`bench/datasets/haberman.csv`,
//! n = 306). The smooth covariate is patient **age** at operation; the binary
//! response is the synthetic censoring indicator drawn from `p_true(age)` plus a
//! fixed-seed Bernoulli draw. The *same* {age, y} pair is handed to BOTH engines
//! (gam reads them from the encoded dataset; INLA reads the identical columns),
//! so there is zero data-encoding skew and both are scored against the identical
//! `p_true`.
//!
//! Both engines fit `y ~ s(age)` (gam: thin-plate `s(x, bs='tp')`; INLA:
//! `f(age, model="rw2", scale.model=TRUE)`, the canonical INLA penalized smooth),
//! binomial/logit.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use std::path::Path;

const HABERMAN_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/haberman.csv");

fn invlogit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Known latent probability that generated the synthetic response.
fn p_true(age: f64) -> f64 {
    invlogit(1.4 * ((age - 30.0) / 13.0).sin() - 0.9)
}

/// Patient ages (first column of haberman.csv, which has NO header row).
fn haberman_ages() -> Vec<f64> {
    let text = std::fs::read_to_string(Path::new(HABERMAN_CSV)).expect("read haberman.csv");
    let mut ages = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let first = line.split(',').next().expect("haberman row has a column");
        let age: f64 = first.parse().expect("haberman age column is numeric");
        ages.push(age);
    }
    assert_eq!(ages.len(), 306, "haberman.csv should carry 306 rows");
    ages
}

#[test]
fn gam_binomial_smooth_recovers_true_probability() {
    init_parallelism();

    // ---- identical data for both engines ----------------------------------
    // Covariate: real patient age. Response: synthetic binary censoring
    // indicator drawn from the KNOWN smooth latent probability p_true(age),
    //   eta_true(age) = 1.4*sin((age-30)/13) - 0.9,  p_true = logit^{-1}(eta),
    // then y ~ Bernoulli(p_true) with a fixed seed. The exact {age, y} pair
    // below is what BOTH gam and INLA receive, and p_true is the ground truth
    // both are scored against.
    let ages = haberman_ages();
    let n = ages.len();
    let mut rng = StdRng::seed_from_u64(20260529);
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let mut y = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for &age in &ages {
        let p = p_true(age);
        truth.push(p);
        let draw = if u01.sample(&mut rng) < p { 1.0 } else { 0.0 };
        y.push(draw);
    }
    // Sanity: the response must be a genuine two-class signal, not degenerate.
    let n_pos: usize = y.iter().filter(|&&v| v > 0.5).count();
    assert!(
        n_pos > 30 && n_pos < n - 30,
        "synthetic binary response is degenerate: {n_pos}/{n} positive"
    );
    // The latent probability must carry real signal: a meaningful spread is what
    // makes "recover the curve" a non-trivial claim and sets the accuracy scale.
    let p_min = truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = p_max - p_min;
    assert!(
        signal_range > 0.3,
        "latent probability signal too flat to test recovery: range={signal_range:.3}"
    );

    // ---- fit with gam: y ~ s(age, bs='tp'), binomial/logit, REML ----------
    let headers: Vec<String> = ["age", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(csv::StringRecord::from(vec![
            ages[i].to_string(),
            y[i].to_string(),
        ]));
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode binomial dataset");
    let col = ds.column_map();
    let age_idx = col["age"];

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(age)", &ds, &cfg).expect("gam binomial smooth fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial/logit s(age)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam linear predictor (logit scale) and its posterior covariance at the
    // training points. Rebuild the frozen design X at the observed ages; with a
    // logit link, X*beta IS eta and the link is applied separately.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, age_idx]] = ages[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild s(age) design at training points");
    let xmat = design.design.to_dense();
    assert_eq!(xmat.nrows(), n, "design row count mismatch");
    let p_dim = xmat.ncols();

    let gam_eta = design.design.apply(&fit.fit.beta);
    assert_eq!(gam_eta.len(), n, "gam eta length mismatch");

    // Vp: smoothing-uncertainty-corrected Bayesian covariance of beta — gam's
    // Laplace posterior covariance of the latent field. We propagate it through
    // the design to get a probability-scale posterior SD for the coverage check.
    let vp = fit
        .fit
        .covariance_corrected
        .as_ref()
        .expect("gam reports the corrected (Vp) Bayesian covariance");
    assert_eq!(
        vp.nrows(),
        p_dim,
        "Vp dimension must match the design column count"
    );

    // Probability-scale fitted values and posterior SD via the delta method:
    //   var(eta_i) = x_i^T Vp x_i = sum_jk X[i,j] Vp[j,k] X[i,k]
    //   p_i        = logit^{-1}(eta_i)
    //   sd_p_i     = p_i (1 - p_i) * sqrt(var(eta_i))   (d p / d eta = p(1-p))
    let mut gam_prob = Vec::with_capacity(n);
    let mut gam_prob_sd = Vec::with_capacity(n);
    for i in 0..n {
        let xi = xmat.row(i);
        let mut var_eta = 0.0;
        for j in 0..p_dim {
            let xij = xi[j];
            if xij == 0.0 {
                continue;
            }
            let mut acc = 0.0;
            for k in 0..p_dim {
                acc += vp[[j, k]] * xi[k];
            }
            var_eta += xij * acc;
        }
        let sd_eta = var_eta.max(0.0).sqrt();
        let p = invlogit(gam_eta[i]);
        gam_prob.push(p);
        gam_prob_sd.push(p * (1.0 - p) * sd_eta);
    }

    // ---- PRIMARY objective metric: gam recovers the TRUE probability ------
    let gam_rmse_truth = rmse(&gam_prob, &truth);

    // ---- fit the SAME data with R-INLA (baseline to match-or-beat) --------
    // INLA's canonical penalized smooth for a 1-D covariate is a second-order
    // random walk f(x, model="rw2", scale.model=TRUE). We score INLA's fitted
    // probability against the SAME ground-truth p_true; INLA is the incumbent
    // accuracy bar, not a target gam must reproduce.
    let r = run_r(
        &[Column::new("age", &ages), Column::new("y", &y)],
        r#"
        suppressPackageStartupMessages(library(INLA))
        # rw2 requires an integer location index; group identical ages so the
        # random walk is defined on the sorted unique-age grid, then map each
        # observation back to its grid node. This is the standard INLA recipe
        # for a smooth effect of a continuous covariate.
        df$ageidx <- as.integer(factor(df$age, levels = sort(unique(df$age))))
        m <- inla(
            y ~ -1 + f(ageidx, model = "rw2", scale.model = TRUE,
                       constr = TRUE) + 1,
            family = "binomial",
            data = df,
            Ntrials = rep(1, nrow(df)),
            control.predictor = list(compute = TRUE, link = 1),
            control.compute = list(config = TRUE)
        )
        fv <- m$summary.fitted.values
        emit("prob", as.numeric(fv$mean[seq_len(nrow(df))]))
        "#,
    );
    let inla_prob = r.vector("prob");
    assert_eq!(
        inla_prob.len(),
        n,
        "INLA fitted-probability length mismatch"
    );
    let inla_rmse_truth = rmse(inla_prob, &truth);

    // Empirical coverage of p_true by gam's +/- 2 SD probability band.
    let mut covered = 0usize;
    for i in 0..n {
        let lo = gam_prob[i] - 2.0 * gam_prob_sd[i];
        let hi = gam_prob[i] + 2.0 * gam_prob_sd[i];
        if truth[i] >= lo && truth[i] <= hi {
            covered += 1;
        }
    }
    let coverage = covered as f64 / n as f64;

    // Context only (NOT a pass criterion): how close the two fits happen to be.
    let rel_prob_vs_inla = relative_l2(&gam_prob, inla_prob);

    eprintln!(
        "haberman s(age) binomial/logit  n={n}  pos={n_pos}  gam_edf={gam_edf:.3}  \
         signal_range={signal_range:.3}\n  \
         RMSE_to_truth: gam={gam_rmse_truth:.4}  inla={inla_rmse_truth:.4}  \
         (gam/inla={:.3})\n  \
         +/-2SD coverage of p_true (gam)={coverage:.3}  \
         [context only] rel_l2(gam,inla)={rel_prob_vs_inla:.4}",
        gam_rmse_truth / inla_rmse_truth.max(1e-12)
    );

    // ---- principled, un-weakened objective bounds -------------------------
    // (1) TRUTH RECOVERY (primary). gam's fitted probability must reconstruct
    // the generating curve to a small fraction of the probability signal's own
    // range. With ~300 Bernoulli points the irreducible per-point sampling
    // wobble is O(sqrt(p(1-p)/local_n)); a well-smoothed fit averages it out, so
    // RMSE against the smooth truth should sit far below the signal spread. 25%
    // of the signal range is a generous-but-real bar: a fit that ignored age and
    // predicted the grand mean would land near the signal's own RMS spread
    // (~30-40% of range), so 25% genuinely demands the curve be recovered, while
    // staying loose enough to absorb honest Bernoulli noise.
    let truth_bar = 0.25 * signal_range;
    assert!(
        gam_rmse_truth < truth_bar,
        "gam failed to recover the true probability curve: \
         RMSE_to_truth={gam_rmse_truth:.4} (bound {truth_bar:.4} = 0.25*signal_range)"
    );

    // (2) MATCH-OR-BEAT INLA on accuracy. gam may not be meaningfully less
    // accurate than the incumbent at recovering the same ground truth.
    assert!(
        gam_rmse_truth <= inla_rmse_truth * 1.10,
        "gam less accurate than INLA at recovering the truth: \
         gam_RMSE={gam_rmse_truth:.4} > 1.10*inla_RMSE={:.4}",
        inla_rmse_truth * 1.10
    );

    // (3) CALIBRATION against truth. gam's reported probability-scale
    // uncertainty must not be anti-conservatively narrow: a +/- 2 SD band
    // (nominal ~95% for a Gaussian latent posterior) should cover the TRUE curve
    // for a large majority of points. We require >= 0.80 empirical coverage — a
    // one-sided floor that catches a posterior SD collapsed too tight to be
    // honest, without penalizing the legitimate slack of a smooth fit at a true
    // curve (the band is around the fit, which itself tracks the truth).
    assert!(
        coverage >= 0.80,
        "gam's +/-2SD probability band under-covers the true curve: \
         coverage={coverage:.3} (floor 0.80)"
    );
}
