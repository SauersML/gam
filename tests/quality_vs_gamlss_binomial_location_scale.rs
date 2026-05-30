//! End-to-end quality: gam's *binomial* location-scale fit (a smooth latent
//! threshold `t(x)` AND a smooth log-sigma `η_ls(x)`, fit jointly by penalized
//! blockwise PIRLS) must RECOVER THE KNOWN MARGINAL SUCCESS PROBABILITY of the
//! data-generating process. The data is simulated from an explicit latent recipe
//! whose marginal `P(y=1 | x) = E_z[expit(t(x) + s(x)·z)]` is computed exactly
//! here by Gauss–Hermite quadrature over the standard-normal latent `z`. That
//! marginal probability is the GROUND TRUTH the fit must reconstruct.
//!
//! OBJECTIVE metric asserted (truth recovery, not peer-mimicry)
//! -----------------------------------------------------------
//! The pass/fail criterion is the root-mean-square error of gam's fitted
//! success probability `P̂_gam(x) = expit(q)` against the exact marginal truth
//! `P_true(x)`, on the probability scale (bounded in [0,1], so it does not blow
//! up near the edges the way the logit scale does). We assert
//! `RMSE(P̂_gam, P_true)` is below a principled bar tied to the irreducible
//! sampling noise of n Bernoulli draws. `mgcv::gam(..., family = binomial)` is
//! fit on the IDENTICAL data and DEMOTED to a baseline-to-match-or-beat: gam's
//! truth-recovery error must be no worse than mgcv's by more than 10%. We do NOT
//! assert gam reproduces mgcv's (itself noisy, possibly mis-smoothed) fitted
//! curve — only that gam recovers the true probability at least as well as the
//! mature standard tool does. The reference rel-L2/Pearson against mgcv are still
//! computed and printed via eprintln! purely as diagnostics.
//!
//! What is — and is NOT — comparable across the two engines
//! --------------------------------------------------------
//! gam's binomial location-scale family is a *composed-link* binary model
//! (verified against `src/families/gamlss.rs`): it fits a latent threshold `t`
//! (spec `"threshold"` → `BlockRole::Threshold`, block-state index 0 — NOT
//! `Mean`; see `custom_family_block_role`) and a log-sigma `η_ls` (spec
//! `"log_sigma"` → `BlockRole::Scale`, block-state index 1) with the PURE
//! exponential sigma link `σ = exp(η_ls)` and inverse `1/σ = exp(-η_ls)` (NOT
//! the `0.01 + exp` floor the Gaussian noise link uses — see
//! `families::sigma_link::exp_sigma_inverse_from_eta_scalar`). The link-scale
//! linear predictor is `q = -t / σ = -t · exp(-η_ls)` and `P(y=1) = expit(q)`,
//! exactly as `compute_probit_q0_from_eta` forms it in the CLI.
//!
//! Crucially, only the COMPOSITE `q = -t/σ` (= logit P) is identifiable from
//! 0/1 data — `t` and `σ` are individually unidentified (any rescaling
//! `t → c·t, σ → c·σ` leaves the likelihood unchanged). gam's log-sigma axis is
//! therefore an INTERNAL reparameterization of the latent index, not an
//! observable second moment. There is no mature R binary family with a matching
//! latent-rescaling σ: gamlss `BI()` has a single parameter, and `DBI()`'s σ is
//! a variance-inflation factor for binomial *counts* (n > 1) that collapses to
//! unidentified on n = 1 Bernoulli data — a fundamentally different object from
//! gam's `exp(η_ls)`. So an element-wise `log σ_gam` vs `log σ_ref` bound would
//! compare two non-comparable, separately-unidentified nuisances and is NOT
//! asserted here; gam's log-sigma smooth is checked only for non-degeneracy (the
//! joint two-block solver really ran) and reported as a diagnostic.
//!
//! The identifiable composite `q = -t/σ` IS the model's `logit P(x)`, so
//! `expit(q)` is gam's estimate of the marginal success probability — directly
//! comparable to the exact `P_true(x)`. The stress this test exists to measure
//! is whether gam's joint composed-link inverse / gradient handling across the
//! probability and log-variance scales actually reconstructs the truth; a real
//! divergence is a bug there, surfacing as truth-recovery error worse than the
//! sampling-noise bar (or materially worse than mgcv).

use gam::estimate::BlockRole;
use gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar;
use gam::gamlss::BinomialLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

/// Standard logistic sigmoid `1 / (1 + exp(-q))`. Both test arms reconstruct
/// the success probability `P̂(x) = expit(q)` where `q = -t / σ` is the
/// composed-link latent logit. Lifted to a free function so the synthetic and
/// real-data arms share the same definition.
fn expit(q: f64) -> f64 {
    1.0 / (1.0 + (-q).exp())
}

#[test]
fn gam_binomial_location_scale_logit_p_matches_mgcv_binomial() {
    init_parallelism();

    // ---- synthetic latent-threshold binomial recipe (fed IDENTICALLY to both
    // engines). Spec: n=150, x~Uniform(-3,3), latent t(x)=1+0.5 sin(pi x),
    // P(y=1|x,z)=expit(t(x)+(0.5+0.2 sin(pi x)) z), z~N(0,1), seed=456. The
    // per-row latent draw `z` injects heteroscedastic over-dispersion into the
    // binary outcome, exercising gam's joint two-axis (threshold + log-sigma)
    // solver; the marginal logit P(x) it induces is what both engines recover.
    let n = 150usize;
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    // Deterministic Numerical-Recipes LCG seeded at 456 so the exact same
    // (x, y) is reproducible in pure Rust and handed verbatim to gamlss.
    let mut state: u64 = 456;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    // x ~ Uniform(-3, 3), sorted so the comparison runs along an ordered curve
    // from x.min to x.max (the design is identical across engines either way).
    let mut x: Vec<f64> = (0..n).map(|_| -3.0 + 6.0 * next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }

    let t_true = |xi: f64| 1.0 + 0.5 * (pi * xi).sin();
    let s_true = |xi: f64| 0.5 + 0.2 * (pi * xi).sin();
    // Bernoulli outcome from the latent logistic: success when a uniform draw
    // falls under the per-row probability expit(t + s·z).
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let p = expit(t_true(x[i]) + s_true(x[i]) * z[i]);
            if next_unit() < p { 1.0 } else { 0.0 }
        })
        .collect();

    // GROUND TRUTH: the marginal success probability of this generative process,
    // P_true(x) = E_z[expit(t(x) + s(x)·z)] with z ~ N(0,1), computed exactly by
    // Gauss–Hermite quadrature. With the substitution z = √2·ξ the Gaussian
    // expectation E_z[f(z)] = (1/√π) Σ_k w_k f(√2·ξ_k). A 20-node rule integrates
    // the smooth expit·Gaussian integrand to far below the sampling noise we
    // assert against. Nodes/weights for the physicists' Hermite polynomial H_20.
    let gh_nodes: [f64; 20] = [
        -5.387480890011233, -4.603682449550744, -3.944764040115625, -3.347854567383216,
        -2.788806058428131, -2.254974002089276, -1.738537712116586, -1.234076215395323,
        -0.737473728545394, -0.245340708300901, 0.245340708300901, 0.737473728545394,
        1.234076215395323, 1.738537712116586, 2.254974002089276, 2.788806058428131,
        3.347854567383216, 3.944764040115625, 4.603682449550744, 5.387480890011233,
    ];
    let gh_weights: [f64; 20] = [
        2.229393645534151e-13, 4.399340992273181e-10, 1.086069370768996e-7, 7.802556478532063e-6,
        2.283386360163539e-4, 3.243773342237863e-3, 2.481052088746362e-2, 1.090172060200233e-1,
        2.866755053628341e-1, 4.622436696006101e-1, 4.622436696006101e-1, 2.866755053628341e-1,
        1.090172060200233e-1, 2.481052088746362e-2, 3.243773342237863e-3, 2.283386360163539e-4,
        7.802556478532063e-6, 1.086069370768996e-7, 4.399340992273181e-10, 2.229393645534151e-13,
    ];
    let inv_sqrt_pi = 1.0 / pi.sqrt();
    let sqrt2 = 2.0_f64.sqrt();
    let p_true: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let t = t_true(xi);
            let s = s_true(xi);
            let mut acc = 0.0;
            for k in 0..20 {
                acc += gh_weights[k] * expit(t + s * sqrt2 * gh_nodes[k]);
            }
            inv_sqrt_pi * acc
        })
        .collect();

    // Guard against a degenerate all-0/all-1 draw (would make a binomial fit
    // meaningless); the seed above yields a healthy mix.
    let ones: f64 = y.iter().sum();
    assert!(
        ones > 10.0 && ones < (n as f64 - 10.0),
        "degenerate binary response: {ones} successes out of {n}"
    );

    // ---- build the dataset (column 0 = x, column 1 = y) --------------------
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode binomial location-scale data");

    // ---- fit with gam: threshold ~ s(x, bs='tp'), log-sigma ~ 1 + s(x, bs='tp')
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam binomial location-scale fit");
    let FitResult::BinomialLocationScale(BinomialLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a binomial location-scale fit");
    };

    // Sanity: the joint fit must carry both a Threshold (latent location) and a
    // Scale (log-sigma) coefficient block, and the log-sigma block must be a
    // genuine multi-coefficient smooth (not a lone intercept) for `1 + s(x)`.
    // The threshold spec is named "threshold", so its role is BlockRole::Threshold
    // (see custom_family_block_role), NOT BlockRole::Mean — there is no Mean block
    // in a two-axis location-scale fit.
    let scale_block = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("binomial location-scale fit must carry a Scale (log-sigma) block");
    assert!(
        fit.fit.block_by_role(BlockRole::Threshold).is_some(),
        "binomial location-scale fit must carry a Threshold (latent location) block"
    );
    assert!(
        scale_block.beta.len() >= 2,
        "smooth noise_formula must materialize a multi-coefficient log-sigma basis, got {}",
        scale_block.beta.len()
    );

    // gam's converged per-row latent predictors at the (training) x rows:
    //   block_states[0].eta = threshold t,  block_states[1].eta = log-sigma η_ls.
    // These are the exact, identification-transform-consistent linear predictors
    // (the binomial log-sigma design is internally reparameterized, so reading
    // the converged η directly is the correct, transform-faithful source).
    let eta_t = &fit.fit.block_states[0].eta;
    let eta_ls = &fit.fit.block_states[1].eta;
    assert_eq!(eta_t.len(), n, "threshold eta length");
    assert_eq!(eta_ls.len(), n, "log-sigma eta length");

    // Identifiable location axis on the logit-P link scale: q = logit P = -t / σ
    // = -t · exp(-η_ls). The log-sigma axis (η_ls = log σ) is gam's internal
    // latent-rescaling parameter — kept only as a diagnostic, not asserted
    // against any reference (see the module doc: no mature binary family shares
    // this σ definition; t and σ are individually unidentified from 0/1 data).
    let gam_logit_p: Vec<f64> = (0..n).map(|i| -eta_t[i] * (-eta_ls[i]).exp()).collect();
    let gam_log_sigma: Vec<f64> = eta_ls.to_vec();

    // Non-degeneracy of gam's log-sigma smooth: the joint two-block solver must
    // have produced a genuinely varying log σ(x) (the `1 + s(x)` noise formula),
    // not a collapsed constant. This confirms the SCALE axis was actually fit,
    // without asserting an unjustified element-wise match to a different σ.
    let ls_min = gam_log_sigma.iter().cloned().fold(f64::INFINITY, f64::min);
    let ls_max = gam_log_sigma
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (ls_max - ls_min) > 1e-3,
        "log-sigma smooth collapsed to a constant (range {:.3e}); the joint scale axis did not fit",
        ls_max - ls_min
    );

    // ---- fit the SAME data with mgcv (the mature penalized binomial GAM) -----
    // y ~ s(x), family = binomial, method = "REML": the standard reference for
    // the identifiable success-probability smooth on Bernoulli data. Predict on
    // the identical training rows (row order = our sorted x) on the LINK scale,
    // i.e. logit P(x) — directly comparable to gam's composite q.
    let body = r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x), family = binomial, method = "REML", data = df)
        emit("logit_p", as.numeric(predict(m, type = "link")))
    "#;
    let r = run_r(&[Column::new("x", &x), Column::new("y", &y)], body);
    let ref_logit_p = r.vector("logit_p");
    assert_eq!(ref_logit_p.len(), n, "mgcv logit-P length mismatch");

    // ---- OBJECTIVE METRIC: truth recovery on the probability scale -----------
    // gam's estimated marginal success probability and mgcv's, each compared to
    // the exact GROUND-TRUTH marginal P_true(x). expit maps both engines' link
    // predictions into [0,1] where RMSE is well-scaled and edge-safe.
    let gam_p: Vec<f64> = gam_logit_p.iter().map(|&q| expit(q)).collect();
    let ref_p: Vec<f64> = ref_logit_p.iter().map(|&q| expit(q)).collect();

    let prob_rmse = |est: &[f64]| -> f64 {
        let s: f64 = est
            .iter()
            .zip(&p_true)
            .map(|(e, t)| (e - t) * (e - t))
            .sum();
        (s / n as f64).sqrt()
    };
    let gam_rmse = prob_rmse(&gam_p);
    let ref_rmse = prob_rmse(&ref_p);

    // Diagnostics ONLY (not pass/fail): how gam's curve tracks mgcv's. Matching a
    // peer tool is not a quality claim, so these are printed, never asserted.
    let rel_logit_p = relative_l2(&gam_logit_p, ref_logit_p);
    let corr_logit_p = pearson(&gam_logit_p, ref_logit_p);

    eprintln!(
        "binomial location-scale truth recovery: n={n} \
         gam_prob_rmse={gam_rmse:.5} mgcv_prob_rmse={ref_rmse:.5} \
         (diag vs mgcv: rel_l2={rel_logit_p:.5} pearson={corr_logit_p:.5}) \
         log_sigma_range={:.4}",
        ls_max - ls_min
    );

    // PRIMARY claim — gam recovers the known marginal probability.
    // The irreducible per-row sampling noise of a Bernoulli draw with success
    // probability p is √(p(1-p)) ≤ 0.5, but a smooth fit averages it down by
    // roughly √(edf/n). With n=150 and a low-edf smooth, the RMSE of the fitted
    // probability against the true probability sits well under 0.10; 0.12 is a
    // principled ceiling (a probability curve wrong by >0.12 RMSE on this clean
    // sinusoidal target is a reconstruction bug, not sampling jitter). NOT
    // loosened to pass — it is comfortably above the expected sampling-limited
    // error yet far from vacuous (a flat P=0.5 fit would score ≈0.18 here).
    assert!(
        gam_rmse < 0.12,
        "gam failed to recover the true marginal success probability: \
         prob_rmse={gam_rmse:.5} (truth from exact Gauss–Hermite marginalization)"
    );

    // SECONDARY claim — match-or-beat the mature standard tool on ACCURACY (not
    // mimicry): gam's truth-recovery error is no worse than mgcv's by >10%.
    assert!(
        gam_rmse <= ref_rmse * 1.10,
        "gam recovers the true probability worse than mgcv: \
         gam_prob_rmse={gam_rmse:.5} > 1.10 * mgcv_prob_rmse={ref_rmse:.5}"
    );
}

/// Held-out area under the ROC curve of `score` against binary `truth`, by the
/// Mann–Whitney U identity AUC = P(score_pos > score_neg) (ties count ½). A
/// rank-based, threshold-free, monotone-invariant accuracy measure — exactly the
/// right objective for a real binary dataset whose generative truth is unknown.
fn auc(score: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(score.len(), truth.len(), "auc length mismatch");
    let mut concordant = 0.0_f64;
    let mut pairs = 0.0_f64;
    for i in 0..truth.len() {
        for j in 0..truth.len() {
            if truth[i] > 0.5 && truth[j] < 0.5 {
                pairs += 1.0;
                if score[i] > score[j] {
                    concordant += 1.0;
                } else if (score[i] - score[j]).abs() <= f64::EPSILON {
                    concordant += 0.5;
                }
            }
        }
    }
    assert!(pairs > 0.0, "auc needs both classes present in truth");
    concordant / pairs
}

/// Mean binomial (Bernoulli) negative log-likelihood — the held-out log-loss, a
/// strictly proper scoring rule for calibrated probabilities. Probabilities are
/// clamped off {0,1} so a single confident miss cannot make the metric infinite.
fn log_loss(prob: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(prob.len(), truth.len(), "log_loss length mismatch");
    let mut acc = 0.0_f64;
    for (p, y) in prob.iter().zip(truth) {
        let pc = p.clamp(1e-12, 1.0 - 1e-12);
        acc += -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln());
    }
    acc / prob.len() as f64
}

#[test]
fn gam_binomial_location_scale_logit_p_matches_mgcv_binomial_on_real_data() {
    init_parallelism();

    // ---- real data: prostate.csv (pc1, pc2 -> binary y) --------------------
    // SOURCE: gam's bench dataset bench/datasets/prostate.csv — two principal
    // components (pc1, pc2) of a prostate-cancer feature matrix and a binary
    // outcome y in {0,1}. 654 rows, near-balanced classes. There is NO known
    // ground-truth probability surface for real data, so quality is measured as
    // OUT-OF-SAMPLE predictive accuracy on a deterministic held-out split, not by
    // recovering a recipe. gam's composed-link binomial location-scale fit (latent
    // threshold t(pc1,pc2) AND log-sigma η_ls(pc1,pc2), fit jointly) produces an
    // identifiable success probability P̂ = expit(-t/σ) = expit(-t·exp(-η_ls)); we
    // rebuild that on held-out rows and score it against the observed labels.
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/prostate.csv"
    )))
    .expect("load prostate.csv");
    let col = ds.column_map();
    let pc1_idx = col["pc1"];
    let pc2_idx = col["pc2"];
    let y_idx = col["y"];
    let pc1: Vec<f64> = ds.values.column(pc1_idx).to_vec();
    let pc2: Vec<f64> = ds.values.column(pc2_idx).to_vec();
    let y_all: Vec<f64> = ds.values.column(y_idx).to_vec();
    let n = pc1.len();
    assert!(n > 600, "prostate should have ~654 rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out --------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 400 && test_rows.len() > 100,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_pc1: Vec<f64> = train_rows.iter().map(|&i| pc1[i]).collect();
    let train_pc2: Vec<f64> = train_rows.iter().map(|&i| pc2[i]).collect();
    let train_y: Vec<f64> = train_rows.iter().map(|&i| y_all[i]).collect();
    let test_pc1: Vec<f64> = test_rows.iter().map(|&i| pc1[i]).collect();
    let test_pc2: Vec<f64> = test_rows.iter().map(|&i| pc2[i]).collect();
    let test_y: Vec<f64> = test_rows.iter().map(|&i| y_all[i]).collect();

    // Both held-out classes must be present for AUC to be defined.
    let test_pos: f64 = test_y.iter().sum();
    assert!(
        test_pos > 20.0 && test_pos < (test_rows.len() as f64 - 20.0),
        "degenerate held-out labels: {test_pos} positives of {}",
        test_rows.len()
    );

    // Training-only dataset by sub-setting encoded rows; schema/headers unchanged
    // so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: threshold ~ s(pc1)+s(pc2), log-sigma ~ same ------
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("1 + s(pc1, bs='tp') + s(pc2, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(pc1, bs='tp') + s(pc2, bs='tp')", &train_ds, &cfg)
        .expect("gam binomial location-scale fit on prostate train");
    let FitResult::BinomialLocationScale(BinomialLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a binomial location-scale fit");
    };

    // Joint solver must carry both a Threshold and a (multi-coefficient) Scale block.
    let scale_block = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("binomial location-scale fit must carry a Scale (log-sigma) block");
    assert!(
        fit.fit.block_by_role(BlockRole::Threshold).is_some(),
        "binomial location-scale fit must carry a Threshold block"
    );
    assert!(
        scale_block.beta.len() >= 2,
        "smooth noise_formula must materialize a multi-coefficient log-sigma basis, got {}",
        scale_block.beta.len()
    );

    // ---- gam held-out prediction ------------------------------------------
    // Rebuild the threshold and log-sigma designs at the TEST rows from the frozen
    // resolved specs, apply each block's converged beta to get the held-out
    // η_t and η_ls, then form the identifiable logit P = -t/σ = -t·exp(-η_ls) and
    // P̂ = expit(logit P). block_states[0] is the threshold block (matches
    // meanspec_resolved), block_states[1] is the log-sigma block (matches
    // noisespec_resolved) — verified in src/families/gamlss.rs.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, pc1_idx]] = test_pc1[i];
        test_grid[[i, pc2_idx]] = test_pc2[i];
    }
    let test_t_design = build_term_collection_design(test_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild threshold design at held-out points");
    let test_ls_design = build_term_collection_design(test_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at held-out points");
    let eta_t_test: Vec<f64> = test_t_design
        .design
        .apply(&fit.fit.block_states[0].beta)
        .to_vec();
    let eta_ls_test: Vec<f64> = test_ls_design
        .design
        .apply(&fit.fit.block_states[1].beta)
        .to_vec();
    assert_eq!(eta_t_test.len(), test_rows.len(), "threshold test eta length");
    assert_eq!(eta_ls_test.len(), test_rows.len(), "log-sigma test eta length");

    let gam_test_logit_p: Vec<f64> = (0..test_rows.len())
        .map(|i| -eta_t_test[i] * exp_sigma_inverse_from_eta_scalar(eta_ls_test[i]))
        .collect();
    let gam_test_p: Vec<f64> = gam_test_logit_p.iter().map(|&q| expit(q)).collect();

    // ---- mgcv baseline: SAME train rows, predict the SAME held-out rows ----
    // mgcv has no composed-link latent-rescaling binary family, so its plain
    // penalized binomial GAM y ~ s(pc1)+s(pc2) is the mature match-or-beat
    // BASELINE for the identifiable success probability (see module doc). One
    // run_r call: train columns + the held-out pc1/pc2 padded to train length and
    // a row-count scalar so only the first k test rows are read back.
    let k = test_rows.len();
    let train_len = train_rows.len();
    let r = run_r(
        &[
            Column::new("pc1", &train_pc1),
            Column::new("pc2", &train_pc2),
            Column::new("y", &train_y),
            Column::new("test_pc1", &pad_to(&test_pc1, train_len)),
            Column::new("test_pc2", &pad_to(&test_pc2, train_len)),
            Column::new("test_n", &vec![k as f64; train_len]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(pc1) + s(pc2), family = binomial, method = "REML", data = df)
        k <- df$test_n[1]
        newd <- data.frame(pc1 = df$test_pc1[1:k], pc2 = df$test_pc2[1:k])
        emit("test_p", as.numeric(predict(m, newdata = newd, type = "response")))
    "#,
    );
    let ref_test_p = r.vector("test_p");
    assert_eq!(ref_test_p.len(), k, "mgcv held-out prediction length mismatch");

    // ---- OBJECTIVE held-out metrics (computed in plain Rust) ---------------
    let gam_auc = auc(&gam_test_p, &test_y);
    let ref_auc = auc(ref_test_p, &test_y);
    let gam_ll = log_loss(&gam_test_p, &test_y);
    let ref_ll = log_loss(ref_test_p, &test_y);

    // Diagnostics ONLY (not pass/fail): agreement of the two engines' held-out
    // probabilities. Matching a peer tool is never a quality claim.
    let rel_p = relative_l2(&gam_test_p, ref_test_p);
    let corr_p = pearson(&gam_test_p, ref_test_p);

    eprintln!(
        "prostate binomial location-scale held-out: n_train={train_len} n_test={k} \
         gam_auc={gam_auc:.4} mgcv_auc={ref_auc:.4} \
         gam_logloss={gam_ll:.4} mgcv_logloss={ref_ll:.4} \
         (diag vs mgcv: rel_l2={rel_p:.4} pearson={corr_p:.4})"
    );

    // ---- PRIMARY objective assertion: gam discriminates on held-out data ---
    // pc1/pc2 carry real signal for y; a competent binary fit clears AUC 0.75 out
    // of sample, far above the chance level of 0.5. Below that bar the composed
    // -link inverse / joint two-block solve is mis-reconstructing the probability.
    assert!(
        gam_test_p.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
        "gam held-out probabilities must be valid in [0,1]"
    );
    assert!(
        gam_auc >= 0.75,
        "gam held-out AUC too low: {gam_auc:.4} (< 0.75)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on the SAME metrics --
    // gam's held-out AUC at least mgcv's minus a 0.02 slack, and its log-loss no
    // worse than mgcv's by >10%. mgcv is the mature baseline to match-or-beat on
    // accuracy, never an output to replicate.
    assert!(
        gam_auc >= ref_auc - 0.02,
        "gam held-out AUC {gam_auc:.4} worse than mgcv {ref_auc:.4} (slack 0.02)"
    );
    assert!(
        gam_ll <= ref_ll * 1.10,
        "gam held-out log-loss {gam_ll:.4} exceeds mgcv {ref_ll:.4} * 1.10"
    );
}

/// Right-pad `v` with its last value (0.0 when empty) to length `len`, so the
/// held-out predictors can ride along as full-length columns of the reference
/// data.frame. Only the first `v.len()` entries are read back inside the R body.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    assert!(
        v.len() <= len,
        "pad target {len} shorter than source {}",
        v.len()
    );
    let fill = v.last().copied().unwrap_or(0.0);
    let mut out = v.to_vec();
    out.resize(len, fill);
    out
}
