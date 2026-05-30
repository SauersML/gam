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
use std::path::PathBuf;

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
    let mut eta_draws: Vec<Vec<f64>> = vec![Vec::with_capacity(ndraw); n];
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
