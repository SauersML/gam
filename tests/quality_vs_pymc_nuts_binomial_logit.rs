//! End-to-end quality: gam's NUTS posterior for a penalized binomial-logit
//! smooth must agree with **PyMC** (the reference NUTS + ArviZ R-hat/ESS
//! stack) — not merely "produce some draws".
//!
//! Why PyMC is the right comparator. PyMC's No-U-Turn Sampler with ArviZ
//! convergence diagnostics is the de-facto Bayesian reference in the Python
//! world. gam exposes its own NUTS posterior for Standard binomial-logit
//! models (`gam::sample::sample_saved_model` → exact GLM NUTS, whitened by the
//! penalized Hessian). The honest question is: does gam's posterior over the
//! *linear predictor* coincide with what a trusted, independent NUTS engine
//! finds for the *same target density*?
//!
//! Making it a genuine head-to-head. gam fits `y ~ s(x)` (penalized B-spline),
//! selects the smoothing parameters λ by REML, and then runs NUTS over the
//! coefficients of the design X with prior precision S = Σ_k λ_k S_k. The
//! posterior gam samples is therefore exactly
//!     p(β | y) ∝ Binomial(y | logit⁻¹(Xβ)) · exp(−½ βᵀ S β).
//! We export gam's *own* design matrix X and *own* penalty precision S (built
//! from the fitted λ and the per-block penalty matrices) and hand BOTH to
//! PyMC, where the same density is encoded as a flat prior on β plus a
//! `pm.Potential(−½ βᵀSβ)`. Both engines then target the identical posterior;
//! only Monte-Carlo / sampler differences remain, which is what justifies the
//! tight bounds below. We compare the posterior mean and SD of the linear
//! predictor η = Xβ at the training points (the quantity that actually drives
//! every downstream prediction), element-wise on the shared grid.
//!
//! A divergence here is a real bug in gam's posterior, never a reason to
//! loosen the bounds or touch gam source.

use gam::inference::model::{FittedFamily, FittedModel, FittedModelPayload, ModelKind};
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::test_support::reference::{Column, max_abs_diff, pearson, rmse, run_python};
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

#[test]
fn gam_nuts_binomial_logit_matches_pymc() {
    init_parallelism();

    // ---- synthetic data: n=200, x in [0,10], Bernoulli(logit(0.3+0.8 sin)) --
    let n = 200usize;
    let mut rng = SplitMix64(0x5EED_1234_ABCD_0001);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        // Evenly spaced x for a well-conditioned design, then jittered draws.
        let xi = 10.0 * (i as f64) / ((n - 1) as f64);
        let eta_true = 0.3 + 0.8 * (2.0 * std::f64::consts::PI * xi / 10.0).sin();
        let p = inv_logit(eta_true);
        let yi = if rng.unit() < p { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
    }
    // Both classes must be present or the logit posterior is degenerate.
    let n_pos = y.iter().filter(|&&v| v > 0.5).count();
    assert!(n_pos > 10 && n_pos < n - 10, "need both classes: n_pos={n_pos}");

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
    // Gaussian prior precision whose posterior gam's NUTS targets.
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

    // Seed identically to PyMC (42). Enough draws for a stable mean/SD on a
    // ~p-dimensional posterior; multiple chains so R-hat is meaningful.
    let adaptive = NutsConfig::for_dimension(p);
    let nuts_cfg = NutsConfig {
        n_samples: 1500,
        nwarmup: 1500,
        n_chains: 4,
        seed: 42,
        ..adaptive
    };
    let nuts = sample_saved_model(&model, ds.values.view(), &col, model.training_headers.as_ref(), &nuts_cfg)
        .expect("gam NUTS sampling");
    assert_eq!(nuts.samples.ncols(), p, "posterior coeff dim mismatch");

    // Posterior of η = X β at the training points, draw by draw.
    let ndraw = nuts.samples.nrows();
    let mut eta_sum = Array1::<f64>::zeros(n);
    let mut eta_sumsq = Array1::<f64>::zeros(n);
    let mut beta_draw = Array1::<f64>::zeros(p);
    for d in 0..ndraw {
        for j in 0..p {
            beta_draw[j] = nuts.samples[[d, j]];
        }
        let eta = x_dense.dot(&beta_draw);
        for i in 0..n {
            eta_sum[i] += eta[i];
            eta_sumsq[i] += eta[i] * eta[i];
        }
    }
    let gam_eta_mean: Vec<f64> = (0..n).map(|i| eta_sum[i] / ndraw as f64).collect();
    let gam_eta_sd: Vec<f64> = (0..n)
        .map(|i| {
            let m = gam_eta_mean[i];
            (eta_sumsq[i] / ndraw as f64 - m * m).max(0.0).sqrt()
        })
        .collect();

    // ---- PyMC: same X, same penalty precision S, same data, NUTS seed 42 ----
    // Flatten X and S row-major so they round-trip through the numeric wire.
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
    let shape = vec![n as f64, p as f64];

    let py = run_python(
        // Only the length-n response travels as a dataframe column. The design
        // X and penalty precision S are p-dimensional, so they are injected as
        // literal arrays into the body and reshaped there.
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
emit("eta_mean", eta_draws.mean(axis=1))
emit("eta_sd", eta_draws.std(axis=1, ddof=0))

# Per-coefficient R-hat (max over the p coefficients) — the standard NUTS
# convergence gate. ArviZ computes split-R-hat.
rhat = az.rhat(idata)["beta"].values
emit("rhat_max", [float(np.nanmax(rhat))])
ess = az.ess(idata)["beta"].values
emit("ess_min", [float(np.nanmin(ess))])
"#,
            n = n,
            p = p,
            x_flat = x_flat,
            s_flat = s_flat,
            shape = shape,
        ),
    );

    let pymc_eta_mean = py.vector("eta_mean");
    let pymc_eta_sd = py.vector("eta_sd");
    let pymc_rhat = py.scalar("rhat_max");
    let pymc_ess = py.scalar("ess_min");
    assert_eq!(pymc_eta_mean.len(), n, "PyMC eta_mean length mismatch");
    assert_eq!(pymc_eta_sd.len(), n, "PyMC eta_sd length mismatch");

    // ---- compare ------------------------------------------------------------
    let mean_corr = pearson(&gam_eta_mean, pymc_eta_mean);
    let mean_maxdiff = max_abs_diff(&gam_eta_mean, pymc_eta_mean);
    let sd_corr = pearson(&gam_eta_sd, pymc_eta_sd);
    let sd_rmse = rmse(&gam_eta_sd, pymc_eta_sd);
    let typ_sd = pymc_eta_sd.iter().sum::<f64>() / n as f64;

    eprintln!(
        "pymc-nuts binomial logit: n={n} p={p} ndraw={ndraw}\n\
         gam_rhat={:.4} gam_ess={:.1} pymc_rhat={pymc_rhat:.4} pymc_ess={pymc_ess:.1}\n\
         eta_mean: pearson={mean_corr:.5} max_abs_diff={mean_maxdiff:.4}\n\
         eta_sd: pearson={sd_corr:.4} rmse={sd_rmse:.4} typical_sd={typ_sd:.4}",
        nuts.rhat, nuts.ess
    );

    // (1) Both samplers must have converged. R-hat < 1.1 is the standard
    // Gelman-Rubin gate; a non-converged chain invalidates the comparison.
    assert!(
        nuts.rhat < 1.1,
        "gam NUTS did not converge: rhat={:.4}",
        nuts.rhat
    );
    assert!(
        pymc_rhat < 1.1,
        "PyMC NUTS did not converge: rhat={pymc_rhat:.4}"
    );

    // (2) Posterior MEAN of the linear predictor must track PyMC's. Same target
    // density ⇒ the posterior means agree up to Monte-Carlo error. Pearson
    // > 0.98 catches any structural divergence in the smooth.
    assert!(
        mean_corr > 0.98,
        "posterior-mean eta diverges from PyMC: pearson={mean_corr:.5}"
    );
    // Posterior SDs here are ~0.2–0.5; 0.15 is ~ a single posterior SD, a tight
    // element-wise bound that still tolerates finite-sample MC noise between two
    // independent NUTS runs.
    assert!(
        mean_maxdiff < 0.15,
        "posterior-mean eta differs pointwise from PyMC: max_abs_diff={mean_maxdiff:.4}"
    );

    // (3) Posterior UNCERTAINTY must agree too: the SD profile is what credible
    // bands are built from. Require strong shape agreement and a small absolute
    // RMSE relative to the typical SD (within ~40% of typical_sd).
    assert!(
        sd_corr > 0.90,
        "posterior-SD eta profile disagrees with PyMC: pearson={sd_corr:.4}"
    );
    assert!(
        sd_rmse < 0.4 * typ_sd.max(1e-6),
        "posterior-SD eta magnitude disagrees with PyMC: rmse={sd_rmse:.4} typical_sd={typ_sd:.4}"
    );
}
