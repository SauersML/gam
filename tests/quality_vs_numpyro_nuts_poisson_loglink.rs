//! End-to-end quality: gam's NUTS posterior for a penalized **Poisson
//! log-link** count regression must match NumPyro — the modern reference NUTS
//! sampler (pure JAX, with ArviZ R-hat / ESS diagnostics) — on real data.
//!
//! Benchmarked tool: **NumPyro** (`numpyro.infer.MCMC` + `NUTS`) with **ArviZ**
//! convergence diagnostics. NumPyro is the best-in-class open reference for
//! gradient-based Bayesian posterior sampling: its NUTS is the same No-U-Turn
//! algorithm gam implements, and its ArviZ split-R-hat / bulk-ESS are the
//! community-standard convergence metrics. If gam's exponential-family
//! log-likelihood **gradient and Hessian** for the Poisson family were wrong,
//! the NUTS leapfrog trajectories would explore a different posterior and the
//! posterior-mean linear predictor `eta = X beta` would diverge from NumPyro's.
//!
//! Identical model fed to both engines. We:
//!   1. aggregate the ICU survival dataset into Poisson **count** observations
//!      — death count per (age, length-of-stay) cell — a textbook count
//!      regression with ~100-200 cells;
//!   2. fit `count ~ s(age) + s(los)` with gam (Poisson/log), which selects the
//!      smoothing parameters lambda by REML;
//!   3. extract gam's **exact** penalized design `X` and total penalty matrix
//!      `S_lambda = sum_k lambda_k S_k` (the Gaussian-prior precision on the
//!      coefficients), and hand the *same* `X`, `S_lambda`, and counts `y` to
//!      NumPyro, which samples the identical posterior
//!         beta ~ N(0, S_lambda^+),   y_i ~ Poisson(exp((X beta)_i))
//!      with the **same** seed, chains=2, warmup=1000, samples=1000;
//!   4. compare the posterior-mean linear predictor `eta = X beta` element-wise
//!      across all cells, and assert NumPyro reports clean convergence.
//!
//! Both samplers target the *same* un-normalized log-density, so their
//! posterior-mean eta must coincide up to Monte-Carlo error; a real divergence
//! is a real bug in gam's Poisson gradient/Hessian or its NUTS whitening.

use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
};
use gam::smooth::{freeze_term_collection_from_design, weighted_blockwise_penalty_sum};
use gam::test_support::reference::{Column, max_abs_diff, pearson, run_python};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
};
use std::io::Write;
use std::path::Path;

const ICU_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/icu_survival_death.csv"
);

/// Number of equal-width bins per covariate axis. 14 x 12 = 168 candidate
/// cells; after dropping empty cells we land in the spec's ~100-200 range.
const AGE_BINS: usize = 14;
const LOS_BINS: usize = 12;

#[test]
fn gam_nuts_poisson_loglink_matches_numpyro() {
    init_parallelism();

    // ---- load the ICU survival dataset (per-patient rows) -----------------
    // columns: time (length-of-stay, days), event (death 0/1), age (years), ...
    let ds = load_csvwith_inferred_schema(Path::new(ICU_CSV)).expect("load icu_survival_death.csv");
    let col = ds.column_map();
    let age_raw: Vec<f64> = ds.values.column(col["age"]).to_vec();
    let los_raw: Vec<f64> = ds.values.column(col["time"]).to_vec();
    let event: Vec<f64> = ds.values.column(col["event"]).to_vec();
    let n_rows = age_raw.len();
    assert!(n_rows > 1000, "icu dataset should be large, got {n_rows}");

    // ---- aggregate into Poisson count cells -------------------------------
    // Bin (age, los) into a fixed grid; the death count in each cell is a
    // Poisson observation whose covariates are the cell-center age / los. This
    // turns 20k Bernoulli rows into ~100-200 count observations — the standard
    // exposure-free Poisson count model. log(los+1) compresses the very long
    // length-of-stay tail so cells are populated across the axis.
    let los_t: Vec<f64> = los_raw.iter().map(|&v| (v + 1.0).ln()).collect();
    let (age_lo, age_hi) = minmax(&age_raw);
    let (los_lo, los_hi) = minmax(&los_t);
    let age_w = (age_hi - age_lo) / AGE_BINS as f64;
    let los_w = (los_hi - los_lo) / LOS_BINS as f64;
    let bin = |v: f64, lo: f64, w: f64, nb: usize| -> usize {
        (((v - lo) / w).floor() as isize).clamp(0, nb as isize - 1) as usize
    };

    let n_cells = AGE_BINS * LOS_BINS;
    let mut cell_count = vec![0.0_f64; n_cells];
    let mut cell_exposure = vec![0.0_f64; n_cells];
    for i in 0..n_rows {
        let ai = bin(age_raw[i], age_lo, age_w, AGE_BINS);
        let li = bin(los_t[i], los_lo, los_w, LOS_BINS);
        let c = ai * LOS_BINS + li;
        cell_count[c] += event[i];
        cell_exposure[c] += 1.0;
    }

    // Keep only populated cells; covariates are the geometric cell centers.
    let mut agg_age = Vec::new();
    let mut agg_los = Vec::new();
    let mut agg_count = Vec::new();
    for ai in 0..AGE_BINS {
        for li in 0..LOS_BINS {
            let c = ai * LOS_BINS + li;
            if cell_exposure[c] > 0.0 {
                agg_age.push(age_lo + (ai as f64 + 0.5) * age_w);
                agg_los.push(los_lo + (li as f64 + 0.5) * los_w);
                agg_count.push(cell_count[c]);
            }
        }
    }
    let n = agg_count.len();
    assert!(
        (80..=260).contains(&n),
        "aggregated count cells should be ~100-200, got {n}"
    );
    let total_deaths: f64 = agg_count.iter().sum();
    assert!(
        total_deaths > 100.0,
        "aggregated cells must carry real signal, got {total_deaths} deaths"
    );

    // ---- write the aggregated count table and load it as a gam dataset ----
    // Round-trip through the inferred-schema CSV loader (the canonical path) so
    // gam sees exactly the same numbers we hand to NumPyro.
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_numpyro_poisson_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create aggregated csv");
        writeln!(f, "count,age,los").expect("write header");
        for i in 0..n {
            writeln!(
                f,
                "{:.0},{:.17e},{:.17e}",
                agg_count[i], agg_age[i], agg_los[i]
            )
            .expect("write row");
        }
    }
    let agg_ds = load_csvwith_inferred_schema(&tmp).expect("load aggregated csv");
    std::fs::remove_file(&tmp).ok();

    // ---- fit count ~ s(age) + s(los) with gam (Poisson / log link) --------
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("count ~ s(age) + s(los)", &agg_ds, &cfg).expect("gam poisson fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson count regression");
    };

    // gam's exact penalized design and total penalty (the prior precision the
    // sampler whitens against): X is the training design, S_lambda is the
    // REML-selected blockwise penalty sum. These are the *identical* objects
    // gam's own NUTS path rebuilds and samples over.
    let x_dense = fit.design.design.to_dense();
    let p = x_dense.ncols();
    assert_eq!(x_dense.nrows(), n, "design rows must match cell count");
    let lambdas = fit
        .fit
        .lambdas
        .as_slice()
        .expect("contiguous lambda vector");
    let penalty = weighted_blockwise_penalty_sum(&fit.design.penalties, lambdas, p);
    assert_eq!(penalty.shape(), [p, p], "penalty must be p x p");

    // ---- run gam's NUTS over the saved model ------------------------------
    // Build an in-memory standard Poisson/log model and sample it through the
    // production `sample_saved_model` dispatch (the same entry point the CLI
    // `gam sample` and the Python `Model.sample` use). Same seed / chains /
    // warmup / draws we give NumPyro below.
    let frozen = freeze_term_collection_from_design(&fit.resolvedspec, &fit.design)
        .expect("freeze resolved term-collection spec");
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "count ~ s(age) + s(los)".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            ),
            link: Some(StandardLink::Log),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "poisson".to_string(),
    );
    payload.resolved_termspec = Some(frozen);
    payload.set_training_feature_metadata(
        agg_ds.headers.clone(),
        agg_ds.feature_ranges(),
    );
    let model = FittedModel::from_payload(payload);

    let nuts_cfg = gam::hmc::NutsConfig {
        n_samples: 1000,
        nwarmup: 1000,
        n_chains: 2,
        target_accept: 0.9,
        seed: 42,
    };
    let agg_col_map = agg_ds.column_map();
    let training_headers = agg_ds.headers.clone();
    let nuts = gam::sample::sample_saved_model(
        &model,
        agg_ds.values.view(),
        &agg_col_map,
        Some(&training_headers),
        &nuts_cfg,
    )
    .expect("gam NUTS Poisson posterior sampling");
    assert_eq!(
        nuts.samples.ncols(),
        p,
        "gam NUTS coefficient dimension must match the design"
    );

    // gam posterior-mean linear predictor eta_i = E[ x_i^T beta ] per cell.
    let gam_eta: Vec<f64> = (0..n)
        .map(|i| {
            let row = x_dense.row(i).to_owned();
            nuts.posterior_mean_of(|b| b.dot(&row))
        })
        .collect();

    // ---- sample the SAME posterior with NumPyro (the mature reference) -----
    // The reference harness requires every data column to have the *same*
    // length, so we pad each payload (counts, row-major X (n*p), row-major
    // S (p*p), and the [n, p] dims header) to a common length L and have the
    // Python body slice each back to its true prefix and reshape. NumPyro then
    // samples the identical penalized Poisson model and ArviZ reports
    // per-coefficient R-hat and bulk-ESS.
    let mut x_flat = vec![0.0_f64; n * p];
    for i in 0..n {
        for j in 0..p {
            x_flat[i * p + j] = x_dense[[i, j]];
        }
    }
    let s_flat: Vec<f64> = penalty.iter().copied().collect(); // row-major p*p
    let dims: Vec<f64> = vec![n as f64, p as f64];

    let len = n.max(n * p).max(p * p).max(dims.len());
    let count_col = pad_to(&agg_count, len);
    let x_col = pad_to(&x_flat, len);
    let s_col = pad_to(&s_flat, len);
    let dims_col = pad_to(&dims, len);

    let r = run_python(
        &[
            Column::new("count", &count_col),
            Column::new("xflat", &x_col),
            Column::new("sflat", &s_col),
            Column::new("dims", &dims_col),
        ],
        r#"
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

n = int(round(float(df["dims"][0])))
p = int(round(float(df["dims"][1])))
y = np.asarray(df["count"], dtype=float)[:n]
X = np.asarray(df["xflat"], dtype=float)[: n * p].reshape(n, p)
S = np.asarray(df["sflat"], dtype=float)[: p * p].reshape(p, p)

# S_lambda is the Gaussian-prior precision on beta. It is rank-deficient (the
# smooth null space is unpenalized), so we draw beta ~ N(0, S^+) by working in
# the eigenbasis: penalized directions get sd = 1/sqrt(eig); the null space
# (eig ~ 0) gets a broad-but-proper sd so the improper flat prior is
# numerically realized exactly as gam's whitening treats it.
S = 0.5 * (S + S.T)
evals, evecs = np.linalg.eigh(S)
tol = max(1e-8, 1e-10 * float(evals.max()))
sd = np.where(evals > tol, 1.0 / np.sqrt(np.maximum(evals, tol)), 1.0e3)
evecs_j = jnp.asarray(evecs)
sd_j = jnp.asarray(sd)
X_j = jnp.asarray(X)
y_j = jnp.asarray(y)

def model():
    # z ~ N(0, I) in the eigenbasis -> beta = V diag(sd) z ~ N(0, S^+).
    z = numpyro.sample("z", dist.Normal(jnp.zeros(p), jnp.ones(p)))
    beta = evecs_j @ (sd_j * z)
    numpyro.deterministic("beta", beta)
    eta = X_j @ beta
    numpyro.sample("y", dist.Poisson(jnp.exp(eta)), obs=y_j)

# Identical sampler budget to gam: seed=42, chains=2, warmup=1000, draws=1000.
numpyro.set_host_device_count(2)
kernel = NUTS(model, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=2,
            chain_method="sequential", progress_bar=False)
mcmc.run(jax.random.PRNGKey(42))

post = mcmc.get_samples(group_by_chain=True)        # dict: name -> (chains, draws, ...)
beta_draws = np.asarray(post["beta"])               # (2, 1000, p)
flat_beta = beta_draws.reshape(-1, p)               # (2000, p)
post_mean_beta = flat_beta.mean(axis=0)             # (p,)
eta_mean = X @ post_mean_beta                        # (n,) posterior-mean eta

# Per-coefficient diagnostics on beta via ArviZ.
idata = az.from_dict(posterior={"beta": beta_draws})
summ = az.summary(idata, var_names=["beta"], round_to=None)
rhat = np.asarray(summ["r_hat"], dtype=float)
ess = np.asarray(summ["ess_bulk"], dtype=float)

emit("eta_mean", eta_mean)
emit("max_rhat", [float(np.nanmax(rhat))])
emit("min_ess", [float(np.nanmin(ess))])
emit("p", [float(p)])
"#,
    );

    let np_eta = r.vector("eta_mean");
    let max_rhat = r.scalar("max_rhat");
    let min_ess = r.scalar("min_ess");
    assert_eq!(np_eta.len(), n, "numpyro eta length mismatch");
    assert_eq!(
        r.scalar("p").round() as usize,
        p,
        "numpyro / gam coefficient dimension disagree"
    );

    // ---- compare ----------------------------------------------------------
    let corr = pearson(&gam_eta, np_eta);
    let mad = max_abs_diff(&gam_eta, np_eta);

    eprintln!(
        "icu poisson count ~ s(age)+s(los): n={n} p={p} total_deaths={total_deaths:.0} \
         gam_edf={:.2} numpyro_max_rhat={max_rhat:.4} numpyro_min_ess={min_ess:.1} \
         pearson(eta)={corr:.5} max_abs_diff(eta)={mad:.4}",
        fit.fit.edf_total().expect("gam total edf"),
    );

    // (1) NumPyro must itself converge — otherwise its posterior means are not
    //     a trustworthy reference. Strict split-R-hat < 1.05 (convergence) and
    //     bulk-ESS > 200 per coefficient (reasonable mixing for 2000 draws).
    assert!(
        max_rhat < 1.05,
        "numpyro NUTS did not converge: max split-R-hat={max_rhat:.4} (need < 1.05)"
    );
    assert!(
        min_ess > 200.0,
        "numpyro NUTS mixed poorly: min bulk-ESS={min_ess:.1} (need > 200)"
    );

    // (2) gam's posterior-mean linear predictor must track NumPyro's. Both
    //     target the identical penalized Poisson log-density, so the only
    //     admissible disagreement is Monte-Carlo noise. The Poisson response
    //     variance ~lambda makes per-cell eta posterior sds ~0.3-0.4, so a
    //     0.12 absolute tolerance is a few MC standard errors of the
    //     posterior-mean estimator — tight enough that any systematic error in
    //     gam's Poisson gradient/Hessian or NUTS whitening would trip it.
    assert!(
        corr > 0.97,
        "gam vs numpyro posterior-mean eta correlation too low: pearson={corr:.5} (need > 0.97)"
    );
    assert!(
        mad < 0.12,
        "gam vs numpyro posterior-mean eta diverge: max_abs_diff={mad:.4} (need < 0.12)"
    );
}

fn minmax(v: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in v {
        if x.is_finite() {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    // Nudge the upper edge so the max value lands inside the last bin.
    (lo, hi + (hi - lo).abs().max(1.0) * 1e-9)
}

/// Zero-pad a flat buffer to exactly `len` entries so every reference-data
/// column shares one length (the harness requires equal-length columns). The
/// Python body slices each column back to the exact prefix it needs.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    assert!(v.len() <= len, "pad_to target shorter than input");
    let mut out = v.to_vec();
    out.resize(len, 0.0);
    out
}
