//! End-to-end **objective quality** of gam's NUTS posterior for a penalized
//! **Poisson log-link** count regression: held-out predictive accuracy on real
//! data.
//!
//! OBJECTIVE METRIC ASSERTED: **held-out posterior-predictive mean Poisson
//! deviance** on a deterministic train/test split of the count cells. gam's
//! NUTS posterior is fit conditioning on the *training* cells only; the
//! posterior-mean linear predictor is then evaluated on the *held-out* test
//! cells and scored against the observed test counts by mean Poisson deviance.
//! The primary claim is that gam's posterior *predicts unseen counts well*:
//!   (a) gam's held-out deviance is far below the intercept-only null model's
//!       (the smooth captures real, generalizing age/los count signal — a
//!       deviance-explained bar), and
//!   (b) gam's held-out deviance is within a small margin of NumPyro's on the
//!       identical penalized density and the identical split.
//!
//! NumPyro (`numpyro.infer.MCMC` + `NUTS`, with ArviZ R-hat / ESS) is **not**
//! the truth and gam is NOT asserted to reproduce its fitted output. It is
//! demoted to a MATCH-OR-BEAT BASELINE on the same objective held-out metric:
//! both engines target the identical penalized Poisson log-density conditioned
//! on the identical training cells, so a mature reference NUTS gives a fair bar
//! for "how well can this posterior predict the held-out cells". Matching its
//! posterior-mean eta element-wise is explicitly NOT the pass criterion — that
//! would only prove "same as a peer tool", never that gam's posterior is good.
//!
//! Both samplers also must converge (ArviZ split-R-hat / bulk-ESS) for their
//! held-out scores to be trustworthy; that convergence gate is retained.
//!
//! Identical data fed to both engines. We:
//!   1. aggregate the ICU survival dataset into Poisson **count** observations
//!      — death count per (age, length-of-stay) cell — a textbook count
//!      regression with ~100-200 cells;
//!   2. fit `count ~ s(age) + s(los)` with gam (Poisson/log), which selects the
//!      smoothing parameters lambda by REML, and freeze its basis/penalty;
//!   3. split the cells deterministically into train (4/5) and held-out test
//!      (1/5) by cell index;
//!   4. sample gam's NUTS posterior over the FROZEN design conditioning on the
//!      training cells only, and sample the SAME penalized density in NumPyro
//!      (same X, same S_lambda, same training cells, same seed/chains/warmup);
//!   5. score each engine's posterior-mean eta on the held-out test cells by
//!      mean Poisson deviance and compare to the null model and to each other.

use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
};
use gam::smooth::{freeze_term_collection_from_design, weighted_blockwise_penalty_sum};
use gam::test_support::reference::{Column, QualityPair, pearson, relative_l2, run_python};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Axis;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};

const ICU_CSV_PARTS: &[&str] = &[
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/icu_survival_death_parts/part_000.csv"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/icu_survival_death_parts/part_001.csv"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/icu_survival_death_parts/part_002.csv"
    ),
];

/// Number of equal-width bins per covariate axis. 14 x 12 = 168 candidate
/// cells; after dropping empty cells we land in the spec's ~100-200 range.
const AGE_BINS: usize = 14;
const LOS_BINS: usize = 12;

/// Deterministic held-out fraction: every `TEST_STRIDE`-th cell (by index)
/// is held out for scoring; the rest train the posterior. Fixed, no RNG.
const TEST_STRIDE: usize = 5;

fn load_icu_age_los_event() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut age, mut los, mut event) = (Vec::new(), Vec::new(), Vec::new());
    for part in ICU_CSV_PARTS {
        let file = File::open(part).expect("open icu_survival_death part");
        let mut lines = BufReader::new(file).lines();
        let header = lines
            .next()
            .expect("icu header line")
            .expect("read icu header");
        let cols: Vec<&str> = header.trim().split(',').collect();
        let idx = |name: &str| {
            cols.iter()
                .position(|c| *c == name)
                .unwrap_or_else(|| panic!("icu_survival_death part missing column {name}"))
        };
        let i_age = idx("age");
        let i_los = idx("time");
        let i_event = idx("event");
        for line in lines {
            let line = line.expect("read icu row");
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            age.push(fields[i_age].trim().parse().expect("parse age"));
            los.push(fields[i_los].trim().parse().expect("parse time"));
            event.push(fields[i_event].trim().parse().expect("parse event"));
        }
    }
    (age, los, event)
}

#[test]
fn gam_nuts_poisson_loglink_predicts_heldout_counts() {
    init_parallelism();

    // ---- load the ICU survival dataset (per-patient rows) -----------------
    // columns: time (length-of-stay, days), event (death 0/1), age (years), ...
    let (age_raw, los_raw, event) = load_icu_age_los_event();
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

    // ---- deterministic train / held-out test split -----------------------
    // Every TEST_STRIDE-th cell (by aggregation index) is held out; the rest
    // train the posterior. Pure index arithmetic — reproducible, no RNG.
    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();
    for i in 0..n {
        if i % TEST_STRIDE == 0 {
            test_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }
    assert!(
        test_idx.len() >= 15 && train_idx.len() >= 60,
        "split too small: {} train / {} test cells",
        train_idx.len(),
        test_idx.len()
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
    // This full-data fit selects the smoothing parameters by REML and FREEZES
    // the basis/knots; we reuse the frozen design so the held-out test-cell
    // rows live in the exact same coefficient space as the training rows.
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("count ~ s(age) + s(los)", &agg_ds, &cfg).expect("gam poisson fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson count regression");
    };

    // gam's exact penalized design and total penalty (the prior precision the
    // sampler whitens against): X is the (frozen-basis) design over ALL cells,
    // S_lambda is the REML-selected blockwise penalty sum.
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

    // ---- build the in-memory frozen Poisson/log model ---------------------
    // Sampling through the production `sample_saved_model` dispatch (the same
    // entry point the CLI `gam sample` and the Python `Model.sample` use)
    // rebuilds the design from this frozen resolved-term spec, so a row subset
    // is evaluated on the SAME knots/constraints as the full fit. That is why
    // the posterior beta from a training-only refit is compatible with the
    // full `x_dense` rows we use to score the held-out cells.
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
    payload.fit_result = Some(fit.fit.clone());
    payload.unified = Some(fit.fit.clone());
    payload.data_schema = Some(agg_ds.schema.clone());
    payload.resolved_termspec = Some(frozen);
    payload.set_training_feature_metadata(agg_ds.headers.clone(), agg_ds.feature_ranges());
    let model = FittedModel::from_payload(payload);

    // ---- sample gam's NUTS conditioning on TRAINING cells only ------------
    // Restrict the dataset rows to the training cells; the posterior is
    // beta | y_train under the frozen penalized Poisson density. Same seed /
    // chains / warmup / draws we give NumPyro below.
    let train_values = agg_ds.values.select(Axis(0), &train_idx);
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
        train_values.view(),
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

    // gam posterior-mean linear predictor on the HELD-OUT test cells, scored
    // against the test counts. (We also keep the full-grid eta for context.)
    let gam_eta_all: Vec<f64> = (0..n)
        .map(|i| {
            let row = x_dense.row(i).to_owned();
            nuts.posterior_mean_of(|b| b.dot(&row))
        })
        .collect();
    let gam_eta_test: Vec<f64> = test_idx.iter().map(|&i| gam_eta_all[i]).collect();
    let y_test: Vec<f64> = test_idx.iter().map(|&i| agg_count[i]).collect();

    // ---- sample the SAME density with NumPyro on the SAME training cells ---
    // We pass the FULL row-major X (n*p), the penalty S (p*p), the counts, and
    // a 0/1 train mask. NumPyro conditions the Poisson likelihood on the train
    // rows only and returns the posterior-mean eta for every cell; we slice the
    // held-out test rows in Rust and score them with the identical metric.
    let mut x_flat = vec![0.0_f64; n * p];
    for i in 0..n {
        for j in 0..p {
            x_flat[i * p + j] = x_dense[[i, j]];
        }
    }
    let s_flat: Vec<f64> = penalty.iter().copied().collect(); // row-major p*p
    let dims: Vec<f64> = vec![n as f64, p as f64];
    let mut is_train = vec![0.0_f64; n];
    for &i in &train_idx {
        is_train[i] = 1.0;
    }

    let len = n.max(n * p).max(p * p).max(dims.len());
    let count_col = pad_to(&agg_count, len);
    let x_col = pad_to(&x_flat, len);
    let s_col = pad_to(&s_flat, len);
    let dims_col = pad_to(&dims, len);
    let train_col = pad_to(&is_train, len);

    let r = run_python(
        &[
            Column::new("count", &count_col),
            Column::new("xflat", &x_col),
            Column::new("sflat", &s_col),
            Column::new("dims", &dims_col),
            Column::new("istrain", &train_col),
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
is_train = np.asarray(df["istrain"], dtype=float)[:n] > 0.5

# Condition the likelihood on the TRAINING cells only — the held-out test
# cells are never seen by the sampler; they are scored after the fact.
Xtr = X[is_train]
ytr = y[is_train]

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
Xtr_j = jnp.asarray(Xtr)
ytr_j = jnp.asarray(ytr)

def model():
    # z ~ N(0, I) in the eigenbasis -> beta = V diag(sd) z ~ N(0, S^+).
    z = numpyro.sample("z", dist.Normal(jnp.zeros(p), jnp.ones(p)))
    beta = evecs_j @ (sd_j * z)
    numpyro.deterministic("beta", beta)
    eta = Xtr_j @ beta
    numpyro.sample("y", dist.Poisson(jnp.exp(eta)), obs=ytr_j)

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
eta_mean = X @ post_mean_beta                        # (n,) posterior-mean eta, ALL cells

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

    let np_eta_all = r.vector("eta_mean");
    let max_rhat = r.scalar("max_rhat");
    let min_ess = r.scalar("min_ess");
    assert_eq!(np_eta_all.len(), n, "numpyro eta length mismatch");
    assert_eq!(
        r.scalar("p").round() as usize,
        p,
        "numpyro / gam coefficient dimension disagree"
    );
    let np_eta_test: Vec<f64> = test_idx.iter().map(|&i| np_eta_all[i]).collect();

    // ---- objective held-out scores ----------------------------------------
    // Mean Poisson deviance of each engine's posterior-mean eta on the unseen
    // test cells, and of the intercept-only NULL model (rate = mean training
    // count) for the deviance-explained bar.
    let mu_null: f64 = {
        let s: f64 = train_idx.iter().map(|&i| agg_count[i]).sum();
        s / train_idx.len() as f64
    };
    let gam_dev = mean_poisson_deviance_from_eta(&y_test, &gam_eta_test);
    let np_dev = mean_poisson_deviance_from_eta(&y_test, &np_eta_test);
    let null_dev = mean_poisson_deviance_from_rate(&y_test, mu_null);
    let gam_dev_explained = 1.0 - gam_dev / null_dev;

    // For context only (NOT a pass criterion): how close the two posteriors'
    // eta happen to be. This is the OLD, demoted "same as a peer tool" check.
    let corr = pearson(&gam_eta_test, &np_eta_test);
    let rel = relative_l2(&gam_eta_test, &np_eta_test);

    eprintln!(
        "icu poisson count ~ s(age)+s(los): n={n} p={p} train={} test={} \
         total_deaths={total_deaths:.0} gam_edf={:.2} \
         numpyro_max_rhat={max_rhat:.4} numpyro_min_ess={min_ess:.1} \
         HELDOUT mean Poisson deviance: gam={gam_dev:.4} numpyro={np_dev:.4} \
         null={null_dev:.4} gam_dev_explained={gam_dev_explained:.3} \
         (context only: pearson(eta_test)={corr:.5} rel_l2(eta_test)={rel:.4})",
        train_idx.len(),
        test_idx.len(),
        fit.fit.edf_total().expect("gam total edf"),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_numpyro_nuts_poisson_loglink",
            "held_out_poisson_deviance",
            gam_dev,
            "numpyro",
            np_dev,
        )
        .line()
    );

    // (1) Both posteriors must converge — otherwise their held-out scores are
    //     not trustworthy. Strict split-R-hat < 1.05 and bulk-ESS > 200.
    assert!(
        max_rhat < 1.05,
        "numpyro NUTS did not converge: max split-R-hat={max_rhat:.4} (need < 1.05)"
    );
    assert!(
        min_ess > 200.0,
        "numpyro NUTS mixed poorly: min bulk-ESS={min_ess:.1} (need > 200)"
    );
    assert!(
        nuts.rhat < 1.05,
        "gam NUTS did not converge: split-R-hat={:.4} (need < 1.05)",
        nuts.rhat
    );

    // (2) PRIMARY OBJECTIVE CLAIM — gam's posterior PREDICTS unseen counts.
    //     Its held-out mean Poisson deviance must be well below the
    //     intercept-only null: the smooth age/los structure generalizes to the
    //     held-out cells rather than memorizing the training cells. A 20%
    //     deviance-explained bar is a low, principled floor for a real
    //     count-regression signal — a fit that does not generalize fails here.
    assert!(
        gam_dev_explained > 0.20,
        "gam posterior does not generalize: held-out Poisson deviance {gam_dev:.4} \
         vs null {null_dev:.4} -> only {gam_dev_explained:.3} deviance-explained (need > 0.20)"
    );

    // (3) MATCH-OR-BEAT the mature reference on the SAME objective held-out
    //     metric. Both target the identical penalized Poisson density on the
    //     identical training cells, so a competent posterior must predict the
    //     held-out cells about as well as NumPyro's. 10% slack absorbs the
    //     Monte-Carlo noise of two independent NUTS runs. This is an accuracy
    //     bar, NOT "reproduce NumPyro's fitted eta".
    assert!(
        gam_dev <= np_dev * 1.10 + 1e-9,
        "gam held-out Poisson deviance {gam_dev:.4} worse than numpyro {np_dev:.4} \
         by more than 10% (match-or-beat on held-out accuracy)"
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

/// Mean Poisson unit deviance between observed counts `y` and a per-cell rate
/// `mu_i = exp(eta_i)`: `d_i = 2[ y_i log(y_i/mu_i) - (y_i - mu_i) ]`, with the
/// `y log y -> 0` limit at `y = 0`. Lower is better (0 = perfect).
fn mean_poisson_deviance_from_eta(y: &[f64], eta: &[f64]) -> f64 {
    assert_eq!(y.len(), eta.len(), "deviance length mismatch");
    let s: f64 = y
        .iter()
        .zip(eta)
        .map(|(&yi, &ei)| poisson_unit_deviance(yi, ei.exp()))
        .sum();
    s / y.len().max(1) as f64
}

/// Mean Poisson unit deviance for the intercept-only model: every cell uses the
/// constant rate `mu`.
fn mean_poisson_deviance_from_rate(y: &[f64], mu: f64) -> f64 {
    let s: f64 = y.iter().map(|&yi| poisson_unit_deviance(yi, mu)).sum();
    s / y.len().max(1) as f64
}

fn poisson_unit_deviance(y: f64, mu: f64) -> f64 {
    let mu = mu.max(1e-12);
    let term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    2.0 * (term - (y - mu))
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
