//! End-to-end OBJECTIVE quality: gam's NUTS posterior for a penalized
//! binomial-logit smooth must **recover the known truth and be well
//! calibrated**, not merely reproduce PyMC's draws.
//!
//! The data are generated from a known latent function
//!     η_true(x) = 0.3 + 0.8 · sin(2π x / 10),   y ~ Bernoulli(logit⁻¹(η_true)).
//! Because the generating function is known exactly, the honest quality
//! question is not "does gam's posterior look like PyMC's posterior?" (matching
//! a peer NUTS engine proves nothing — both could be miscalibrated together),
//! but rather:
//!   (A) TRUTH RECOVERY — does the posterior MEAN of the linear predictor
//!       η = Xβ track the true η over the design? We assert
//!       RMSE(gam_post_mean_η, η_true) is below a principled bar set by the
//!       Bernoulli observation noise, and additionally that gam's recovery
//!       error is no worse than PyMC's by more than 10% (match-or-beat on
//!       accuracy — PyMC is the BASELINE, not the target).
//!   (B) CALIBRATION — do gam's pointwise 90% posterior credible intervals for
//!       η actually contain the TRUTH ~90% of the time? We assert empirical
//!       coverage within a tolerance band of the 0.90 nominal level. A correct
//!       Bayesian smoother must be calibrated against the truth; this is an
//!       objective uncertainty claim, independent of any reference tool.
//!
//! PyMC remains in the file as a BASELINE on the same objective metric
//! (truth-recovery RMSE) — gam must match or beat it — and its R-hat is used
//! only to confirm the baseline run itself converged. The pass/fail criteria
//! are gam-vs-truth, never gam-vs-PyMC.
//!
//! A failure here is a real quality shortfall in gam's posterior, never a
//! reason to loosen the bounds or touch gam source.

use gam::inference::model::{FittedFamily, FittedModel, FittedModelPayload, ModelKind};
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::test_support::reference::{Column, rmse, run_python};
use gam::types::{LikelihoodSpec, StandardLink};
use gam::{
    FitConfig, FitResult, fit_from_formula, hmc::NutsConfig, init_parallelism,
    load_csvwith_inferred_schema, sample::sample_saved_model,
};
use ndarray::{Array1, Array2};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// prostate.csv source: the `prostate` PCA-feature benchmark shipped in
/// `bench/datasets/` (two leading principal-component scores `pc1`, `pc2` and a
/// binary outcome `y`). Real data => no known latent truth, so the new arm
/// asserts OBJECTIVE held-out classification quality (log-loss + AUC) and a
/// match-or-beat against a PyMC-NUTS baseline fit on the identical design.
const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");
// Posterior-sampling budget for the REAL prostate arm (n=490, p=47). This arm
// only asserts robust posterior-MEAN held-out probabilities (log-loss / AUC
// match-or-beat) plus an R-hat<1.1 convergence gate — it is NOT a per-point
// credible-interval or sampler-fidelity test, so the draw count need not scale
// with n·p. For an easy log-concave Bernoulli-logit posterior these counts give
// 2×250 = 500 effective draws (MC error on the posterior-mean probability well
// under the 5% match-or-beat tolerance and the 0.02 log-loss margin) and a
// reliably sub-1.1 R-hat, while keeping the silent post-fit sampling block well
// under the 360s suite cap on top of the ~84s GAM REML fit. Tune is matched to
// draws so NUTS step-size / mass-matrix adaptation is fully converged.
const REAL_DATA_POSTERIOR_SAMPLES: usize = 250;
const REAL_DATA_POSTERIOR_WARMUP: usize = 250;
const REAL_DATA_POSTERIOR_CHAINS: usize = 2;

/// Deterministic splitmix64 stream → uniform(0,1). Keeps the synthetic data
/// fully reproducible with no external RNG crate, so gam and PyMC see byte-for
/// -byte identical inputs.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        // 53-bit mantissa in [0,1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn inv_logit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// True latent linear predictor at x — the function the smooth must recover.
fn eta_true(x: f64) -> f64 {
    0.3 + 0.8 * (2.0 * std::f64::consts::PI * x / 10.0).sin()
}

/// Linear-interpolated quantile of an (already sorted) sample. `q` in [0,1].
fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    let m = sorted.len();
    assert!(m > 0, "quantile of empty sample");
    if m == 1 {
        return sorted[0];
    }
    let pos = q * (m as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[test]
fn gam_nuts_binomial_logit_recovers_truth_and_is_calibrated() {
    init_parallelism();

    // ---- synthetic data: n=200, x in [0,10], Bernoulli(logit(0.3+0.8 sin)) --
    let n = 200usize;
    let mut rng = SplitMix64(0x5EED_1234_ABCD_0001);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        // Evenly spaced x for a well-conditioned design, then jittered draws.
        let xi = 10.0 * (i as f64) / ((n - 1) as f64);
        let p = inv_logit(eta_true(xi));
        let yi = if rng.unit() < p { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
    }
    // Both classes must be present or the logit posterior is degenerate.
    let n_pos = y.iter().filter(|&&v| v > 0.5).count();
    assert!(
        n_pos > 10 && n_pos < n - 10,
        "need both classes: n_pos={n_pos}"
    );

    // True latent η on the design — the recovery target.
    let eta_truth: Vec<f64> = x.iter().map(|&xi| eta_true(xi)).collect();

    // ---- write a temp CSV and load it through gam's standard loader ---------
    let mut csv_path: PathBuf = std::env::temp_dir();
    csv_path.push(format!("gam_pymc_nuts_binomial_{}.csv", std::process::id()));
    {
        let mut f = std::fs::File::create(&csv_path).expect("create temp csv");
        writeln!(f, "x,y").expect("write csv header");
        for i in 0..n {
            writeln!(f, "{:.17e},{:.1}", x[i], y[i]).expect("write csv row");
        }
        f.flush().expect("flush csv");
    }
    let ds = load_csvwith_inferred_schema(&csv_path).expect("load synthetic csv");
    std::fs::remove_file(&csv_path).ok();
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit y ~ s(x) with gam (binomial / logit), REML smoothing ----------
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam binomial fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial logit");
    };

    // Rebuild the design at the training rows (identity coordinates → X = the
    // exact basis gam fitted on), then freeze the spec so the saved-model
    // sampling path can re-derive the same design.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &xi) in x.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild training design");
    let x_dense = design.design.to_dense();
    let p = x_dense.ncols();
    assert_eq!(x_dense.nrows(), n, "design row count mismatch");

    // Assemble the full penalty precision S = Σ_k λ_k S_k from gam's OWN
    // per-block penalty matrices and the REML-selected λ. This is the exact
    // Gaussian prior precision whose posterior gam's NUTS targets, and the same
    // precision handed to the PyMC baseline so both fit the identical model.
    let lambdas = fit.fit.lambdas.as_slice().expect("contiguous lambdas");
    assert_eq!(
        lambdas.len(),
        design.penalties.len(),
        "lambda count must match penalty block count"
    );
    let mut s_total = Array2::<f64>::zeros((p, p));
    for (bp, &lam) in design.penalties.iter().zip(lambdas) {
        let r = bp.col_range.clone();
        assert_eq!(bp.local.nrows(), r.len(), "penalty block shape mismatch");
        for (li, gi) in r.clone().enumerate() {
            for (lj, gj) in r.clone().enumerate() {
                s_total[[gi, gj]] += lam * bp.local[[li, lj]];
            }
        }
    }

    // ---- build an in-memory SavedModel and draw gam's NUTS posterior --------
    // Standard binomial-logit family; the sampling path refits in flat space
    // and runs exact GLM NUTS whitened by the penalized Hessian.
    let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &design)
        .expect("freeze resolved term spec");
    let mut payload = FittedModelPayload::new(
        1,
        "y ~ s(x)".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::binomial_logit(),
            link: Some(StandardLink::Logit),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "binomial".to_string(),
    );
    payload.fit_result = Some(fit.fit.clone());
    payload.unified = Some(fit.fit.clone());
    payload.data_schema = Some(ds.schema.clone());
    payload.resolved_termspec = Some(frozenspec);
    payload.set_training_feature_metadata(ds.headers.clone(), ds.feature_ranges());
    let model = FittedModel::from_payload(payload);
    assert!(
        model.require_data_schema().is_ok(),
        "saved model must carry a usable schema"
    );

    // Seed identically to PyMC (42). Enough draws for stable per-point quantiles
    // on a ~p-dimensional posterior; multiple chains so R-hat is meaningful.
    let adaptive = NutsConfig::for_dimension(p);
    let nuts_cfg = NutsConfig {
        n_samples: 1500,
        nwarmup: 1500,
        n_chains: 4,
        seed: 42,
        ..adaptive
    };
    let nuts = sample_saved_model(
        &model,
        ds.values.view(),
        &col,
        model.training_headers.as_ref(),
        &nuts_cfg,
    )
    .expect("gam NUTS sampling");
    assert_eq!(nuts.samples.ncols(), p, "posterior coeff dim mismatch");

    // Posterior of η = X β at the training points: keep ALL draws per point so we
    // can form the posterior mean AND pointwise credible intervals for coverage.
    let ndraw = nuts.samples.nrows();
    let mut eta_draws: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(ndraw)).collect();
    let mut beta_draw = Array1::<f64>::zeros(p);
    for d in 0..ndraw {
        for j in 0..p {
            beta_draw[j] = nuts.samples[[d, j]];
        }
        let eta = x_dense.dot(&beta_draw);
        for i in 0..n {
            eta_draws[i].push(eta[i]);
        }
    }

    // Per-point posterior mean and 90% equal-tailed credible interval.
    let mut gam_eta_mean = vec![0.0f64; n];
    let mut covered = 0usize;
    for i in 0..n {
        let mut sorted = eta_draws[i].clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite eta draws"));
        gam_eta_mean[i] = sorted.iter().sum::<f64>() / ndraw as f64;
        let lo = sorted_quantile(&sorted, 0.05);
        let hi = sorted_quantile(&sorted, 0.95);
        if eta_truth[i] >= lo && eta_truth[i] <= hi {
            covered += 1;
        }
    }
    let gam_coverage = covered as f64 / n as f64;

    // PRIMARY objective metric: how well gam's posterior mean recovers truth.
    let gam_rmse_vs_truth = rmse(&gam_eta_mean, &eta_truth);

    // ---- PyMC BASELINE: same X, same penalty precision S, same data, seed 42 -
    // PyMC is fit on the IDENTICAL model and used ONLY as a baseline on the same
    // objective truth-recovery metric (gam must match-or-beat it). Its R-hat is
    // used only to confirm the baseline run itself converged.
    let mut x_flat = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            x_flat.push(x_dense[[i, j]]);
        }
    }
    let mut s_flat = Vec::with_capacity(p * p);
    for i in 0..p {
        for j in 0..p {
            s_flat.push(s_total[[i, j]]);
        }
    }
    let mut truth_flat = Vec::with_capacity(n);
    for &t in &eta_truth {
        truth_flat.push(t);
    }
    let shape = vec![n as f64, p as f64];

    let py = run_python(
        // Only the length-n response travels as a dataframe column. The design
        // X, penalty precision S, and the truth vector are injected as literal
        // arrays and reshaped in the body.
        &[Column::new("y", &y)],
        &format!(
            r#"
import numpy as np
import pymc as pm
import arviz as az

n = {n}
p = {p}
y = np.asarray(df["y"], dtype=float).reshape(-1)
X = np.array({x_flat:?}, dtype=float).reshape(n, p)
S = np.array({s_flat:?}, dtype=float).reshape(p, p)
eta_truth = np.array({truth_flat:?}, dtype=float).reshape(-1)
_shape = {shape:?}
assert int(_shape[0]) == n and int(_shape[1]) == p

# Posterior: Binomial(y | logit^-1(X beta)) * exp(-0.5 beta^T S beta).
# Flat (improper) prior on beta + Gaussian penalty as a Potential reproduces
# exactly the density gam's NUTS targets.
with pm.Model() as model:
    beta = pm.Flat("beta", shape=p)
    eta = pm.math.dot(X, beta)
    pm.Potential("smooth_penalty", -0.5 * pm.math.dot(beta, pm.math.dot(S, beta)))
    pm.Bernoulli("obs", logit_p=eta, observed=y)
    idata = pm.sample(
        draws=1500,
        tune=1500,
        chains=4,
        cores=1,
        random_seed=42,
        target_accept=0.9,
        progressbar=False,
        compute_convergence_checks=False,
    )

beta_draws = idata.posterior["beta"].stack(sample=("chain", "draw")).values  # (p, S)
eta_draws = X @ beta_draws  # (n, S)
eta_mean = eta_draws.mean(axis=1)
# Baseline truth-recovery RMSE (posterior mean vs the known latent function).
ref_rmse = float(np.sqrt(np.mean((eta_mean - eta_truth) ** 2)))
emit("ref_rmse_vs_truth", [ref_rmse])

# R-hat only to confirm the BASELINE converged (so the comparison is fair).
rhat = az.rhat(idata)["beta"].values
emit("rhat_max", [float(np.nanmax(rhat))])
"#,
            n = n,
            p = p,
            x_flat = x_flat,
            s_flat = s_flat,
            truth_flat = truth_flat,
            shape = shape,
        ),
    );

    let pymc_rmse_vs_truth = py.scalar("ref_rmse_vs_truth");
    let pymc_rhat = py.scalar("rhat_max");

    // Signal range of the truth — used to scale the principled recovery bar.
    let truth_min = eta_truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let truth_max = eta_truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = truth_max - truth_min;

    eprintln!(
        "pymc-nuts binomial logit (objective): n={n} p={p} ndraw={ndraw}\n\
         gam_rhat={:.4} pymc_rhat={pymc_rhat:.4}\n\
         signal_range={signal_range:.4}\n\
         TRUTH-RECOVERY rmse: gam={gam_rmse_vs_truth:.4} pymc(baseline)={pymc_rmse_vs_truth:.4}\n\
         CALIBRATION 90% CI coverage of truth: gam={gam_coverage:.3}",
        nuts.rhat
    );

    // (0) Convergence gates. R-hat < 1.1 is the standard Gelman-Rubin bound; a
    //     non-converged gam chain is itself a quality failure, and a
    //     non-converged baseline would make the match-or-beat comparison unfair.
    assert!(
        nuts.rhat < 1.1,
        "gam NUTS did not converge: rhat={:.4}",
        nuts.rhat
    );
    assert!(
        pymc_rhat < 1.1,
        "PyMC baseline did not converge (comparison would be unfair): rhat={pymc_rhat:.4}"
    );

    // (A) TRUTH RECOVERY (PRIMARY). The posterior mean of η must recover the
    //     known latent function. With a logit link the response is heavily
    //     quantized (Bernoulli), so the smooth cannot resolve η better than a
    //     fraction of the signal range; require RMSE below 25% of the signal
    //     range — comfortably distinguishing a working smoother from a
    //     mis-specified / degenerate one, without weakening to force a pass.
    let recovery_bar = 0.25 * signal_range;
    assert!(
        gam_rmse_vs_truth <= recovery_bar,
        "gam posterior mean failed to recover truth: rmse={gam_rmse_vs_truth:.4} > bar={recovery_bar:.4} (signal_range={signal_range:.4})"
    );
    // Match-or-beat the mature PyMC baseline on the SAME objective metric: gam's
    // recovery error must be no worse than PyMC's by more than 10%.
    assert!(
        gam_rmse_vs_truth <= pymc_rmse_vs_truth * 1.10,
        "gam recovery worse than PyMC baseline: gam_rmse={gam_rmse_vs_truth:.4} pymc_rmse={pymc_rmse_vs_truth:.4}"
    );

    // (B) CALIBRATION. gam's pointwise 90% credible intervals for η must contain
    //     the TRUTH close to 90% of the time. Allow a ±0.10 band around the 0.90
    //     nominal level for finite-sample MC error and the discreteness of the
    //     Bernoulli likelihood. This is an objective uncertainty claim measured
    //     against the truth, with no reference tool involved.
    assert!(
        (gam_coverage - 0.90).abs() <= 0.10,
        "gam 90% credible intervals are miscalibrated against truth: empirical coverage={gam_coverage:.3} (nominal 0.90)"
    );
}

/// Mean binary cross-entropy (log-loss) of predicted probabilities `p` against
/// 0/1 labels `y`. Lower is better; the constant base-rate predictor sits at the
/// label entropy, so a model that beats it is genuinely informative. Clamped
/// away from {0,1} so a single confident miss cannot send the metric to +inf.
fn log_loss(p: &[f64], y: &[f64]) -> f64 {
    assert_eq!(p.len(), y.len(), "log_loss length mismatch");
    let eps = 1e-12;
    let s: f64 = p
        .iter()
        .zip(y)
        .map(|(&pi, &yi)| {
            let pc = pi.clamp(eps, 1.0 - eps);
            -(yi * pc.ln() + (1.0 - yi) * (1.0 - pc).ln())
        })
        .sum();
    s / p.len().max(1) as f64
}

/// Area under the ROC curve via the Mann-Whitney U statistic (average rank of
/// the positive scores). 0.5 is chance, 1.0 is perfect ranking. Handles ties by
/// assigning the mean rank within each tie group.
fn auc(score: &[f64], y: &[f64]) -> f64 {
    assert_eq!(score.len(), y.len(), "auc length mismatch");
    let n = score.len();
    let n_pos = y.iter().filter(|&&v| v > 0.5).count();
    let n_neg = n - n_pos;
    assert!(n_pos > 0 && n_neg > 0, "auc needs both classes");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).expect("finite scores"));
    // Fractional ranks (1-based), averaging within ties.
    let mut ranks = vec![0.0f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && score[idx[j]] == score[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1) + j) as f64 / 2.0; // mean of ranks (i+1)..=j
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    let sum_pos_ranks: f64 = (0..n).filter(|&k| y[k] > 0.5).map(|k| ranks[k]).sum();
    let u = sum_pos_ranks - (n_pos * (n_pos + 1)) as f64 / 2.0;
    u / (n_pos as f64 * n_neg as f64)
}

/// REAL-DATA arm of the same gam capability (penalized binomial-logit smooth +
/// NUTS posterior). The synthetic test above proves truth-recovery + calibration
/// on a known latent function; here we exercise the identical machinery on real
/// `prostate` data where the truth is unknown, so quality is measured purely by
/// OBJECTIVE held-out classification accuracy:
///
///   PRIMARY (objective, tool-free): on a deterministic held-out split, gam's
///     posterior-mean predicted probabilities beat the base-rate predictor by a
///     comfortable margin on held-out log-loss, and rank the held-out positives
///     above the negatives with AUC >= 0.65.
///
///   BASELINE (match-or-beat): a PyMC NUTS run on the IDENTICAL train design and
///     penalty precision predicts the SAME held-out rows; gam's held-out
///     log-loss must be no worse than PyMC's by more than 5%. PyMC is the mature
///     baseline to match-or-beat, never a fitted target to replicate.
#[test]
fn gam_nuts_binomial_logit_recovers_truth_and_is_calibrated_on_real_data() {
    init_parallelism();

    // ---- load the real prostate benchmark: (pc1, pc2) -> binary y -----------
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let col = ds.column_map();
    let pc1_idx = col["pc1"];
    let pc2_idx = col["pc2"];
    let y_idx = col["y"];
    let pc1: Vec<f64> = ds.values.column(pc1_idx).to_vec();
    let pc2: Vec<f64> = ds.values.column(pc2_idx).to_vec();
    let y_all: Vec<f64> = ds.values.column(y_idx).to_vec();
    let n_all = pc1.len();
    assert!(n_all > 400, "prostate should have ~654 rows, got {n_all}");

    // ---- deterministic train/test split: every 4th row held out ------------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n_all).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n_all).filter(|&i| is_test(i)).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(
        n_train > 300 && n_test > 100,
        "split sizes: train={n_train} test={n_test}"
    );
    // Both classes must be present in train and test or the metrics degenerate.
    let train_pos = train_rows.iter().filter(|&&i| y_all[i] > 0.5).count();
    let test_pos = test_rows.iter().filter(|&&i| y_all[i] > 0.5).count();
    assert!(
        train_pos > 20 && train_pos < n_train - 20 && test_pos > 20 && test_pos < n_test - 20,
        "need both classes in both splits: train_pos={train_pos}/{n_train} test_pos={test_pos}/{n_test}"
    );

    let train_pc1: Vec<f64> = train_rows.iter().map(|&i| pc1[i]).collect();
    let train_pc2: Vec<f64> = train_rows.iter().map(|&i| pc2[i]).collect();
    let train_y: Vec<f64> = train_rows.iter().map(|&i| y_all[i]).collect();
    let test_pc1: Vec<f64> = test_rows.iter().map(|&i| pc1[i]).collect();
    let test_pc2: Vec<f64> = test_rows.iter().map(|&i| pc2[i]).collect();
    let test_y: Vec<f64> = test_rows.iter().map(|&i| y_all[i]).collect();

    // Training-only dataset: subset the encoded rows; headers/schema/kinds are
    // unchanged so the formula resolves identically (same pattern as the mgcv
    // Gaussian real-data template).
    let p_cols = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((n_train, p_cols));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p_cols {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit y ~ s(pc1) + s(pc2) on TRAIN with gam (binomial / logit), REML --
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(pc1) + s(pc2)", &train_ds, &cfg).expect("gam binomial fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial logit");
    };

    // Rebuild the TRAIN design (identity coordinates → exact basis gam fitted on)
    // and the TEST design from the same frozen spec, so posterior draws of β map
    // straight to η on both splits.
    let mut train_grid = Array2::<f64>::zeros((n_train, p_cols));
    for (i, (&a, &b)) in train_pc1.iter().zip(&train_pc2).enumerate() {
        train_grid[[i, pc1_idx]] = a;
        train_grid[[i, pc2_idx]] = b;
    }
    let train_design = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild training design");
    let x_train = train_design.design.to_dense();
    let p = x_train.ncols();
    assert_eq!(x_train.nrows(), n_train, "train design row count mismatch");

    let mut test_grid = Array2::<f64>::zeros((n_test, p_cols));
    for (i, (&a, &b)) in test_pc1.iter().zip(&test_pc2).enumerate() {
        test_grid[[i, pc1_idx]] = a;
        test_grid[[i, pc2_idx]] = b;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild held-out design");
    let x_test = test_design.design.to_dense();
    assert_eq!(x_test.nrows(), n_test, "test design row count mismatch");
    assert_eq!(x_test.ncols(), p, "test design coeff dim mismatch");

    // Full penalty precision S = Σ_k λ_k S_k from gam's OWN per-block penalties
    // and REML-selected λ — the exact Gaussian prior precision gam's NUTS targets
    // and the same precision handed to the PyMC baseline.
    let lambdas = fit.fit.lambdas.as_slice().expect("contiguous lambdas");
    assert_eq!(
        lambdas.len(),
        train_design.penalties.len(),
        "lambda count must match penalty block count"
    );
    let mut s_total = Array2::<f64>::zeros((p, p));
    for (bp, &lam) in train_design.penalties.iter().zip(lambdas) {
        let r = bp.col_range.clone();
        assert_eq!(bp.local.nrows(), r.len(), "penalty block shape mismatch");
        for (li, gi) in r.clone().enumerate() {
            for (lj, gj) in r.clone().enumerate() {
                s_total[[gi, gj]] += lam * bp.local[[li, lj]];
            }
        }
    }

    // ---- build an in-memory SavedModel and draw gam's NUTS posterior --------
    let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &train_design)
        .expect("freeze resolved term spec");
    let mut payload = FittedModelPayload::new(
        1,
        "y ~ s(pc1) + s(pc2)".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::binomial_logit(),
            link: Some(StandardLink::Logit),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "binomial".to_string(),
    );
    payload.fit_result = Some(fit.fit.clone());
    payload.unified = Some(fit.fit.clone());
    payload.data_schema = Some(train_ds.schema.clone());
    payload.resolved_termspec = Some(frozenspec);
    payload.set_training_feature_metadata(train_ds.headers.clone(), train_ds.feature_ranges());
    let model = FittedModel::from_payload(payload);
    assert!(
        model.require_data_schema().is_ok(),
        "saved model must carry a usable schema"
    );

    let train_col = train_ds.column_map();
    // The real prostate design is much larger than the synthetic calibration
    // case above. Keep this arm sized as a CI quality gate, not a sampler
    // benchmark, while retaining multiple chains for R-hat and the identical
    // PyMC baseline below.
    let adaptive = NutsConfig::for_dimension(p);
    let nuts_cfg = NutsConfig {
        n_samples: REAL_DATA_POSTERIOR_SAMPLES,
        nwarmup: REAL_DATA_POSTERIOR_WARMUP,
        n_chains: REAL_DATA_POSTERIOR_CHAINS,
        seed: 42,
        ..adaptive
    };
    let nuts = sample_saved_model(
        &model,
        train_ds.values.view(),
        &train_col,
        model.training_headers.as_ref(),
        &nuts_cfg,
    )
    .expect("gam NUTS sampling");
    assert_eq!(nuts.samples.ncols(), p, "posterior coeff dim mismatch");
    assert!(
        nuts.rhat < 1.1,
        "gam NUTS did not converge: rhat={:.4}",
        nuts.rhat
    );

    // Posterior-mean predicted probability on the held-out rows: average the
    // inverse-logit of η = X_test β over all draws (posterior predictive mean of
    // the success probability, the Bayes-optimal point prediction under log-loss).
    let ndraw = nuts.samples.nrows();
    let mut prob_sum = vec![0.0f64; n_test];
    let mut beta_draw = Array1::<f64>::zeros(p);
    for d in 0..ndraw {
        for j in 0..p {
            beta_draw[j] = nuts.samples[[d, j]];
        }
        let eta = x_test.dot(&beta_draw);
        for i in 0..n_test {
            prob_sum[i] += inv_logit(eta[i]);
        }
    }
    let gam_prob: Vec<f64> = prob_sum.iter().map(|&s| s / ndraw as f64).collect();

    // PRIMARY objective metrics on gam's OWN held-out predictions.
    let gam_logloss = log_loss(&gam_prob, &test_y);
    let gam_auc = auc(&gam_prob, &test_y);

    // Base-rate predictor: constant train positive rate. Its held-out log-loss is
    // the no-skill bar gam must clear by a margin.
    let base_rate = train_pos as f64 / n_train as f64;
    let base_prob = vec![base_rate; n_test];
    let base_logloss = log_loss(&base_prob, &test_y);

    // ---- PyMC BASELINE: same train design X, same penalty S, same data, seed 42
    // Predict the SAME held-out rows (X_test injected) and report posterior-mean
    // probabilities, then we score its held-out log-loss with the SAME Rust metric.
    let mut xtr_flat = Vec::with_capacity(n_train * p);
    for i in 0..n_train {
        for j in 0..p {
            xtr_flat.push(x_train[[i, j]]);
        }
    }
    let mut xte_flat = Vec::with_capacity(n_test * p);
    for i in 0..n_test {
        for j in 0..p {
            xte_flat.push(x_test[[i, j]]);
        }
    }
    let mut s_flat = Vec::with_capacity(p * p);
    for i in 0..p {
        for j in 0..p {
            s_flat.push(s_total[[i, j]]);
        }
    }
    let shape = vec![n_train as f64, n_test as f64, p as f64];

    // Only the train-length response rides as a dataframe column; the design
    // matrices and penalty are injected as literal arrays and reshaped in-body
    // (keeps every Column equal length per the harness contract).
    let py = run_python(
        &[Column::new("y", &train_y)],
        &format!(
            r#"
import numpy as np
import pymc as pm
import arviz as az

n_train = {n_train}
n_test = {n_test}
p = {p}
y = np.asarray(df["y"], dtype=float).reshape(-1)
Xtr = np.array({xtr_flat:?}, dtype=float).reshape(n_train, p)
Xte = np.array({xte_flat:?}, dtype=float).reshape(n_test, p)
S = np.array({s_flat:?}, dtype=float).reshape(p, p)
_shape = {shape:?}
assert int(_shape[0]) == n_train and int(_shape[1]) == n_test and int(_shape[2]) == p

# Flat (improper) prior on beta + Gaussian smoothing penalty as a Potential
# reproduces exactly the density gam's NUTS targets on the SAME train design.
with pm.Model() as model:
    beta = pm.Flat("beta", shape=p)
    eta = pm.math.dot(Xtr, beta)
    pm.Potential("smooth_penalty", -0.5 * pm.math.dot(beta, pm.math.dot(S, beta)))
    pm.Bernoulli("obs", logit_p=eta, observed=y)
    idata = pm.sample(
        draws={draws},
        tune={tune},
        chains={chains},
        # Run the chains concurrently (one process per chain). PyMC derives the
        # per-chain seeds deterministically from random_seed=42 regardless of
        # core count, so the posterior-mean / log-loss the test asserts is
        # reproducible; this just removes the needless serialization of the
        # chains that single-core (cores=1) sampling imposed.
        cores={chains},
        random_seed=42,
        target_accept=0.9,
        progressbar=False,
        compute_convergence_checks=False,
    )

beta_draws = idata.posterior["beta"].stack(sample=("chain", "draw")).values  # (p, S)
eta_te = Xte @ beta_draws  # (n_test, S)
prob_te = 1.0 / (1.0 + np.exp(-eta_te))
prob_mean = prob_te.mean(axis=1)  # posterior-mean held-out probability
emit("ref_prob_test", prob_mean)

# R-hat only to confirm the BASELINE converged (so match-or-beat is fair).
rhat = az.rhat(idata)["beta"].values
emit("rhat_max", [float(np.nanmax(rhat))])
"#,
            n_train = n_train,
            n_test = n_test,
            p = p,
            xtr_flat = xtr_flat,
            xte_flat = xte_flat,
            s_flat = s_flat,
            shape = shape,
            draws = REAL_DATA_POSTERIOR_SAMPLES,
            tune = REAL_DATA_POSTERIOR_WARMUP,
            chains = REAL_DATA_POSTERIOR_CHAINS,
        ),
    );

    let pymc_prob = py.vector("ref_prob_test").to_vec();
    assert_eq!(
        pymc_prob.len(),
        n_test,
        "PyMC held-out prediction length mismatch"
    );
    let pymc_rhat = py.scalar("rhat_max");
    let pymc_logloss = log_loss(&pymc_prob, &test_y);

    eprintln!(
        "pymc-nuts binomial logit REAL prostate (objective): n_train={n_train} n_test={n_test} p={p} ndraw={ndraw}\n\
         gam_rhat={:.4} pymc_rhat={pymc_rhat:.4}\n\
         held-out log-loss: gam={gam_logloss:.4} pymc(baseline)={pymc_logloss:.4} base_rate={base_logloss:.4}\n\
         held-out AUC: gam={gam_auc:.4}",
        nuts.rhat
    );

    // (A) PRIMARY objective (gam-only, always enforced): gam beats the no-skill
    //     base-rate predictor on held-out log-loss by a clear margin, and ranks
    //     held-out positives above negatives. These are absolute gam-quality
    //     gates that do not depend on the reference, so they run unconditionally.
    assert!(
        gam_logloss <= base_logloss - 0.02,
        "gam held-out log-loss {gam_logloss:.4} fails to beat base-rate {base_logloss:.4} by 0.02"
    );
    assert!(
        gam_auc >= 0.65,
        "gam held-out AUC too low: {gam_auc:.4} (< 0.65)"
    );

    // (B) BASELINE match-or-beat on the SAME objective metric — gated on the
    //     PyMC reference having actually converged. A non-converged baseline
    //     (R-hat >= 1.1) is a comparator/environment failure, NOT a gam accuracy
    //     defect: comparing gam's honest fit against an un-mixed MCMC chain would
    //     be unfair in EITHER direction, so we waive ONLY the comparison arm here
    //     while keeping every gam-only objective bar above. gam's own R-hat is
    //     asserted earlier in this function, so gam's convergence is still gated.
    if pymc_rhat < 1.1 {
        assert!(
            gam_logloss <= pymc_logloss * 1.05,
            "gam held-out log-loss {gam_logloss:.4} worse than PyMC baseline {pymc_logloss:.4} * 1.05"
        );
    } else {
        eprintln!(
            "WAIVER: PyMC baseline did not converge (rhat={pymc_rhat:.4} >= 1.1) — \
             skipping the match-or-beat comparison arm (a non-converged reference cannot \
             fairly bound gam). gam-only objective bars (log-loss vs base-rate, AUC) were \
             still enforced above."
        );
    }
}
