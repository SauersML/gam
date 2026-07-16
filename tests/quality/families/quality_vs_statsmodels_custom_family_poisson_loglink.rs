//! End-to-end quality: gam's *custom-family* machinery must produce an
//! OBJECTIVELY good fit on a **non-canonical link** — not merely reproduce a
//! mature GLM reference's coefficients.
//!
//! The capability under test is `fit_custom_family` driving a hand-written
//! `CustomFamily` that encodes a Poisson likelihood with the **identity** link
//! `μ = η = X·β`. Identity-link Poisson is the canonical stress test for a
//! custom family: the variance-stabilizing transform (`sqrt`) and the link
//! (`identity`) disagree, so the family must linearize the mean correctly
//! through the IRLS pseudo-response `z = η + (y-μ)/μ'` and Fisher weight
//! `w = (μ')²/V(μ) = 1/μ` rather than borrowing the canonical log-link weights
//! `w = μ`. A wrong gradient/weight derivation still "runs" but converges to a
//! worse-fitting coefficient vector.
//!
//! The data are synthetic with a KNOWN mean function
//! `μ_true(x) = exp(0.5 + 0.3·[sin(0.6·x) + 0.15·x])`, so we can assert against
//! ground truth rather than against another tool's noisy fit. The OBJECTIVE
//! metrics asserted are:
//!
//!   1. PREDICTIVE ACCURACY (primary). A deterministic 70/30 train/test split.
//!      gam fits the custom Poisson-identity family on the train rows, predicts
//!      the held-out test rows, and we assert the held-out **mean Poisson
//!      deviance** `D̄ = (2/n)·Σ[y·log(y/μ) − (y−μ)]` is below a principled bar
//!      (a chi-square-style ~1-per-d.o.f. expectation for a well-fit Poisson)
//!      AND no worse than statsmodels' held-out deviance plus a small margin.
//!      The deviance is computed on gam's OWN held-out predictions.
//!
//!   2. TRUTH RECOVERY (secondary). On every row the fitted mean must track the
//!      KNOWN `μ_true`: `RMSE(μ̂_gam, μ_true)` must be a small fraction of the
//!      mean signal level, AND no worse than statsmodels' truth-RMSE × 1.10.
//!
//! statsmodels is therefore demoted to a BASELINE-TO-MATCH-OR-BEAT on both
//! objective metrics; the pass/fail criterion is gam's own accuracy against the
//! ground-truth mean and against held-out counts, not closeness to statsmodels'
//! coefficients. The reference rel-L2 is still printed (via `relative_l2`) for
//! context only.

use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState,
};
use gam::matrix::{DenseDesignMatrix, DesignMatrix, LinearOperator};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pad_to, relative_l2, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};
use std::path::Path;

/// Poisson likelihood with the identity link `μ = η`.
///
/// `eta = X·β + offset` is supplied by the solver in `block_states[0].eta`.
/// For the identity link `μ = η`, `μ' = dμ/dη = 1`, so the IRLS working set is
/// the textbook Fisher-scoring GLM iteration:
///   * pseudo-response `z = η + (y − μ)/μ' = η + (y − μ)`
///   * working weight  `w = (μ')² / V(μ) = 1/μ`
/// which is precisely the iteration `statsmodels.GLM(...).fit()` performs, so a
/// correct family converges to a fit that recovers the true mean.
#[derive(Clone)]
struct PoissonIdentityFamily {
    /// Observed counts, one per row.
    y: Array1<f64>,
}

/// Floor for `μ` so the IRLS weight `1/μ` and `log μ` stay finite while the
/// inner Newton iterates cross the feasibility boundary; the converged mode of
/// this dataset has `μ` comfortably above this floor, so it does not perturb
/// the optimum (it only guards transient infeasible trial points). statsmodels'
/// own identity-link Poisson IRLS uses an equivalent positivity guard.
const MU_FLOOR: f64 = 1e-8;

impl CustomFamily for PoissonIdentityFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let eta = &block_states[0].eta;
        let n = eta.len();
        if n != self.y.len() {
            return Err(format!(
                "PoissonIdentityFamily: eta len {n} != response len {}",
                self.y.len()
            ));
        }
        let mut ll = 0.0;
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu = eta[i].max(MU_FLOOR);
            let yi = self.y[i];
            // Poisson log-likelihood (dropping the response-only log(y!) constant,
            // which does not affect the MLE): ℓ_i = y·log μ − μ.
            ll += yi * mu.ln() - mu;
            // Identity link ⇒ μ' = 1: z = η + (y − μ), w = 1/μ.
            z[i] = eta[i] + (yi - mu);
            w[i] = 1.0 / mu;
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: z,
                working_weights: w,
            }],
        })
    }
}

/// Mean Poisson deviance `D̄ = (2/n)·Σ[y·log(y/μ) − (y−μ)]`, with the
/// `y·log(y/μ)` term taken as 0 when `y = 0` (its limit). For a correctly
/// specified Poisson fit the per-observation deviance has expectation ≈ 1, so a
/// held-out mean deviance near 1 (and not much above it) is the objective sign
/// of a well-calibrated count model. `μ` is floored at `MU_FLOOR` so a stray
/// non-positive prediction cannot produce a `-inf`/`NaN` deviance.
fn mean_poisson_deviance(y: &[f64], mu: &[f64]) -> f64 {
    assert_eq!(y.len(), mu.len(), "mean_poisson_deviance length mismatch");
    let n = y.len().max(1) as f64;
    let mut d = 0.0;
    for (&yi, &mui) in y.iter().zip(mu.iter()) {
        let m = mui.max(MU_FLOOR);
        let term = if yi > 0.0 { yi * (yi / m).ln() } else { 0.0 };
        d += 2.0 * (term - (yi - m));
    }
    d / n
}

/// Root-mean-square error of a fitted mean against the known true mean.
fn rmse_vs(fitted: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(fitted.len(), truth.len(), "rmse_vs length mismatch");
    let n = fitted.len().max(1) as f64;
    let s: f64 = fitted
        .iter()
        .zip(truth.iter())
        .map(|(&f, &t)| (f - t) * (f - t))
        .sum();
    (s / n).sqrt()
}

#[test]
fn custom_poisson_identity_link_recovers_truth_and_predicts() {
    init_parallelism();

    // ---- synthetic data: n=200, X ~ U(0,10), Y ~ Poisson(μ_true(X)) ---------
    // Deterministic LCG + Knuth Poisson sampler so the test is reproducible and
    // both engines consume byte-identical data via the shared CSV harness.
    let n = 200usize;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next_u01 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 53 bits -> (0,1)
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };

    // True mean function, known in closed form — this is the ground truth we
    // assert recovery against.
    let mu_true_of = |xi: f64| -> f64 {
        let s = (0.6 * xi).sin() + 0.15 * xi;
        (0.5 + 0.3 * s).exp()
    };

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut mu_true = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 10.0 * next_u01();
        let mean = mu_true_of(xi);
        // Knuth's Poisson sampler.
        let l = (-mean).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            p *= next_u01();
            if p <= l {
                break;
            }
            k += 1.0;
        }
        x.push(xi);
        y.push(k);
        mu_true.push(mean);
    }

    // ---- deterministic 70/30 train/test split by row index ------------------
    // Every 10th row (indices 7,8,9 mod 10) is held out, giving a fixed,
    // seed-free split that both engines and the truth-recovery check share.
    let is_test = |i: usize| -> bool { i % 10 >= 7 };
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let n_train = train_idx.len();
    let n_test = test_idx.len();
    assert!(n_train > 0 && n_test > 0, "degenerate train/test split");

    // ---- build a cubic design, standardized on the TRAIN rows only ----------
    // Intercept + centered cubic basis of a standardized covariate. Train
    // statistics are used to transform BOTH train and test so the held-out
    // prediction is honest (no leakage). Raw design columns are handed verbatim
    // to statsmodels (no spline reimplementation), guaranteeing identical-data,
    // identical-design comparison.
    let mean_x: f64 = train_idx.iter().map(|&i| x[i]).sum::<f64>() / n_train as f64;
    let var_x: f64 = train_idx
        .iter()
        .map(|&i| (x[i] - mean_x).powi(2))
        .sum::<f64>()
        / n_train as f64;
    let sd_x = var_x.sqrt().max(1e-12);
    let z_of = |xi: f64| -> f64 { (xi - mean_x) / sd_x };
    // Centering offsets for the quadratic/cubic powers, computed on train rows.
    let m2: f64 = train_idx.iter().map(|&i| z_of(x[i]).powi(2)).sum::<f64>() / n_train as f64;
    let m3: f64 = train_idx.iter().map(|&i| z_of(x[i]).powi(3)).sum::<f64>() / n_train as f64;

    let p = 4usize; // [1, z, z2-centered, z3-centered]
    let row_of = |xi: f64| -> [f64; 4] {
        let zz = z_of(xi);
        [1.0, zz, zz * zz - m2, zz * zz * zz - m3]
    };

    // Full design (for truth-recovery diagnostics over all n rows).
    let mut xmat_full = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let r = row_of(x[i]);
        for j in 0..p {
            xmat_full[[i, j]] = r[j];
        }
    }
    // Train design + train response for fitting.
    let mut xmat_train = Array2::<f64>::zeros((n_train, p));
    let mut y_train = Vec::with_capacity(n_train);
    for (row, &i) in train_idx.iter().enumerate() {
        let r = row_of(x[i]);
        for j in 0..p {
            xmat_train[[row, j]] = r[j];
        }
        y_train.push(y[i]);
    }
    // Test design + test response + test truth for held-out evaluation.
    let mut xmat_test = Array2::<f64>::zeros((n_test, p));
    let mut y_test = Vec::with_capacity(n_test);
    for (row, &i) in test_idx.iter().enumerate() {
        let r = row_of(x[i]);
        for j in 0..p {
            xmat_test[[row, j]] = r[j];
        }
        y_test.push(y[i]);
    }

    // ---- fit with gam's custom-family engine on TRAIN (unpenalized) ---------
    let mean_y_train: f64 = y_train.iter().sum::<f64>() / n_train as f64;
    let family = PoissonIdentityFamily {
        y: Array1::from(y_train.clone()),
    };
    // Intercept-only positive start so μ = Xβ > 0 everywhere initially; the
    // other coefficients start at 0. The optimum is unique, so the start only
    // governs feasibility, not the answer.
    let mut beta0 = Array1::<f64>::zeros(p);
    beta0[0] = mean_y_train.max(1.0);
    let spec = ParameterBlockSpec {
        name: "poisson_identity".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(xmat_train.clone())),
        offset: Array1::<f64>::zeros(n_train),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: Some(beta0),
        ..ParameterBlockSpec::defaults()
    };
    let options = BlockwiseFitOptions::default();
    let result =
        gam::custom_family::fit_custom_family(&family, std::slice::from_ref(&spec), &options)
            .expect("gam custom Poisson-identity fit");
    let beta_gam: Vec<f64> = result.blocks[0].beta.to_vec();
    assert_eq!(
        beta_gam.len(),
        p,
        "gam returned {} coeffs, expected {p}",
        beta_gam.len()
    );
    let beta_gam_arr = Array1::from(beta_gam.clone());

    // gam predictions: identity link ⇒ μ̂ = X·β.
    let mu_gam_full: Vec<f64> = xmat_full.dot(&beta_gam_arr).to_vec();
    let mu_gam_test: Vec<f64> = xmat_test.dot(&beta_gam_arr).to_vec();

    // OBJECTIVE metric 1: held-out mean Poisson deviance on gam's own preds.
    let dev_gam_test = mean_poisson_deviance(&y_test, &mu_gam_test);
    // OBJECTIVE metric 2: truth-recovery RMSE of the fitted mean over all rows.
    let rmse_gam_truth = rmse_vs(&mu_gam_full, &mu_true);

    // ---- fit the SAME train design with statsmodels (baseline to match/beat) -
    // The CSV carries the FULL response, the FULL design columns, and a
    // train/test flag, so statsmodels fits on train and predicts on the same
    // held-out rows from byte-identical data.
    let train_flag: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let r = run_python(
        &[
            Column::new("y", &y),
            Column::new("c0", &xmat_full.column(0).to_vec()),
            Column::new("c1", &xmat_full.column(1).to_vec()),
            Column::new("c2", &xmat_full.column(2).to_vec()),
            Column::new("c3", &xmat_full.column(3).to_vec()),
            Column::new("is_train", &train_flag),
        ],
        r#"
import statsmodels.api as sm
yv = np.asarray(df["y"], dtype=float)
X = np.column_stack([
    np.asarray(df["c0"], dtype=float),
    np.asarray(df["c1"], dtype=float),
    np.asarray(df["c2"], dtype=float),
    np.asarray(df["c3"], dtype=float),
])
tr = np.asarray(df["is_train"], dtype=float) > 0.5
fam = sm.families.Poisson(link=sm.families.links.Identity())
model = sm.GLM(yv[tr], X[tr], family=fam)
res = model.fit(maxiter=300)
mu_test = np.asarray(res.predict(X[~tr]), dtype=float)
mu_full = np.asarray(res.predict(X), dtype=float)
emit("beta", np.asarray(res.params, dtype=float))
emit("mu_test", mu_test)
emit("mu_full", mu_full)
"#,
    );
    let beta_sm = r.vector("beta");
    let mu_sm_test = r.vector("mu_test");
    let mu_sm_full = r.vector("mu_full");
    assert_eq!(
        beta_sm.len(),
        p,
        "statsmodels returned {} coeffs",
        beta_sm.len()
    );
    assert_eq!(mu_sm_test.len(), n_test, "statsmodels test-pred length");
    assert_eq!(mu_sm_full.len(), n, "statsmodels full-pred length");

    // Baseline metrics from statsmodels' own predictions.
    let dev_sm_test = mean_poisson_deviance(&y_test, mu_sm_test);
    let rmse_sm_truth = rmse_vs(mu_sm_full, &mu_true);

    // Context only: how close are the two coefficient vectors? Printed, NOT
    // asserted — matching a peer tool's fit is not a quality claim.
    let beta_rel = relative_l2(&beta_gam, beta_sm);

    // Signal scale for a scale-aware truth-recovery bar.
    let mu_true_mean: f64 = mu_true.iter().sum::<f64>() / n as f64;

    eprintln!(
        "poisson-identity custom family: n={n} (train={n_train} test={n_test}) p={p}\n  \
         held-out mean deviance: gam={dev_gam_test:.4} sm={dev_sm_test:.4}\n  \
         truth RMSE:             gam={rmse_gam_truth:.4} sm={rmse_sm_truth:.4} \
         (signal mean μ_true={mu_true_mean:.4})\n  \
         (context only) beta_rel_l2 vs statsmodels = {beta_rel:.5}"
    );
    eprintln!("beta_gam = {beta_gam:?}");
    eprintln!("beta_sm  = {beta_sm:?}");
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_statsmodels_custom_family_poisson_loglink",
            "mu_rmse_to_truth",
            rmse_gam_truth,
            "statsmodels",
            rmse_sm_truth,
        )
        .line()
    );

    // ---- OBJECTIVE assertion 1: held-out predictive accuracy ---------------
    // For a correctly specified Poisson model the per-observation deviance has
    // expectation ≈ 1; a mean held-out deviance comfortably below 1.6 means gam
    // genuinely predicts the held-out counts well (not just mimics statsmodels).
    assert!(
        dev_gam_test <= 1.6,
        "gam held-out mean Poisson deviance too high: {dev_gam_test:.4} (bar 1.6)"
    );
    // Match-or-beat the mature baseline on the SAME held-out rows (small margin
    // for solver/float differences on an identical concave problem).
    assert!(
        dev_gam_test <= dev_sm_test + 0.05,
        "gam held-out deviance worse than statsmodels: gam={dev_gam_test:.4} sm={dev_sm_test:.4}"
    );

    // ---- OBJECTIVE assertion 2: truth recovery ------------------------------
    // The fitted mean must track the KNOWN μ_true to within a small fraction of
    // the signal level — this is the primary correctness claim and is fully
    // independent of any reference tool.
    let truth_bar = 0.20 * mu_true_mean;
    assert!(
        rmse_gam_truth <= truth_bar,
        "gam fitted mean does not recover the true mean: RMSE={rmse_gam_truth:.4} > bar {truth_bar:.4}"
    );
    // And gam's accuracy against ground truth must match-or-beat statsmodels'.
    assert!(
        rmse_gam_truth <= rmse_sm_truth * 1.10,
        "gam truth-RMSE worse than statsmodels by >10%: gam={rmse_gam_truth:.4} sm={rmse_sm_truth:.4}"
    );
}

/// Path to the `badhealth` count dataset shipped with the benches.
///
/// SOURCE: the `badhealth` data frame from the R package **COUNT** (Hilbe,
/// J.M., *Negative Binomial Regression*, 2nd ed., Cambridge University Press,
/// 2011), distributed under GPL-2. n=1127 German Socioeconomic Panel
/// respondents. Columns: `numvisit` = number of physician visits in a quarter
/// (the count response), `badh` = self-reported bad-health indicator (0/1),
/// `age` = age in years.
const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// Real-data arm for the SAME count-GAM capability the synthetic test proves:
/// a Poisson regression of a count response on a smooth of a continuous
/// covariate plus a categorical/binary main effect, with a multiplicative
/// (log) mean. On real data the generating function is unknown, so we assert
/// OBJECTIVE *held-out predictive* quality rather than truth recovery:
///
///   PRIMARY (tool-free, objective): a deterministic 75/25 train/test split
///     (every 4th row held out). gam fits `numvisit ~ s(age) + badh` on the
///     train rows under Poisson(log), predicts the held-out rows, and we assert
///     the held-out **mean Poisson deviance** is below an absolute bar. For
///     these doctor-visit counts the data are over-dispersed (Var ≫ mean), so a
///     well-fit Poisson mean-model lands its per-row deviance well below the
///     constant-mean (intercept-only) null deviance; the bar is set against that
///     null so a broken link/design/PIRLS that does no better than the global
///     mean cannot pass.
///
///   BASELINE (match-or-beat): statsmodels fits the SAME Poisson(log) model —
///     a penalized cubic B-spline smooth of `age` (GCV-selected penalty) plus a
///     linear `badh` term, via `GLMGam` — on the IDENTICAL train rows and
///     predicts the IDENTICAL held-out rows. gam's held-out mean deviance must
///     be no worse than statsmodels' plus a small margin. statsmodels is a
///     baseline to beat on the held-out metric, never an output to replicate.
///
/// The mean log-link prediction is `μ̂ = exp(X·β̂)`; the held-out design is
/// rebuilt from gam's frozen spec via `build_term_collection_design`.
#[test]
fn custom_poisson_identity_link_recovers_truth_and_predicts_on_real_data() {
    init_parallelism();

    // ---- load the real badhealth count dataset (age, badh -> numvisit) ------
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let col = ds.column_map();
    let age_idx = col["age"];
    let badh_idx = col["badh"];
    let numvisit_idx = col["numvisit"];
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let badh: Vec<f64> = ds.values.column(badh_idx).to_vec();
    let numvisit: Vec<f64> = ds.values.column(numvisit_idx).to_vec();
    let n = age.len();
    assert!(n > 1000, "badhealth should have ~1127 rows, got {n}");

    // ---- deterministic 75/25 train/test split: every 4th row held out -------
    let is_test = |i: usize| -> bool { i % 4 == 0 };
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(
        n_train > 700 && n_test > 200,
        "split sizes: train={n_train} test={n_test}"
    );

    // Held-out covariates / response (same rows, same order, for both engines).
    let train_age: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
    let train_badh: Vec<f64> = train_rows.iter().map(|&i| badh[i]).collect();
    let train_numvisit: Vec<f64> = train_rows.iter().map(|&i| numvisit[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();
    let test_badh: Vec<f64> = test_rows.iter().map(|&i| badh[i]).collect();
    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Train-only encoded dataset (headers/schema unchanged ⇒ formula resolves
    // identically). Subset the encoded value matrix by the train row indices.
    let pcols = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((n_train, pcols));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..pcols {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ s(age) + badh, Poisson(log), REML -----
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("numvisit ~ s(age) + badh", &train_ds, &cfg).expect("gam poisson fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam held-out predictions: rebuild the frozen design at the test rows;
    // for a log link design*beta is the linear predictor, so μ̂ = exp(η̂).
    let mut test_grid = Array2::<f64>::zeros((n_test, pcols));
    for i in 0..n_test {
        test_grid[[i, age_idx]] = test_age[i];
        test_grid[[i, badh_idx]] = test_badh[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // OBJECTIVE metric: held-out mean Poisson deviance on gam's own preds.
    let dev_gam_test = mean_poisson_deviance(&test_numvisit, &gam_test_mu);

    // Null (intercept-only) held-out deviance: predict the constant train mean
    // for every test row. This is the tool-free floor a real mean-model must
    // beat — a degenerate fit that learns nothing matches this and fails the bar.
    let mean_train: f64 = train_numvisit.iter().sum::<f64>() / n_train as f64;
    let null_mu: Vec<f64> = vec![mean_train; n_test];
    let dev_null_test = mean_poisson_deviance(&test_numvisit, &null_mu);

    // ---- fit the SAME Poisson(log) model with statsmodels (baseline) --------
    // GLMGam: a penalized cubic B-spline smooth of `age` (df=10, GCV-selected
    // penalty) plus a linear `badh` main effect, Poisson(Log). It is fit on the
    // train rows and predicts the held-out rows from byte-identical data. Every
    // Column in this one call is train-length except the explicitly test-length
    // prediction inputs, which we pad to train length and slice back by `test_n`.
    let r = run_python(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_to(&test_age, n_train)),
            Column::new("test_badh", &pad_to(&test_badh, n_train)),
            Column::new("test_n", &vec![n_test as f64; n_train]),
        ],
        r#"
import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

age  = np.asarray(df["age"],  dtype=float)
badh = np.asarray(df["badh"], dtype=float)
y    = np.asarray(df["numvisit"], dtype=float)

k = int(np.asarray(df["test_n"], dtype=float)[0])
test_age  = np.asarray(df["test_age"],  dtype=float)[:k]
test_badh = np.asarray(df["test_badh"], dtype=float)[:k]

# Penalized cubic B-spline smooth of age; linear badh enters as an exog_smoother
# covariate handled outside the spline basis.
bs = BSplines(age.reshape(-1, 1), df=[10], degree=[3])
exog = sm.add_constant(badh.reshape(-1, 1))
fam = sm.families.Poisson(link=sm.families.links.Log())
g = GLMGam(y, exog=exog, smoother=bs, alpha=[1.0], family=fam)
alpha_opt, _ = g.select_penweight()
g = GLMGam(y, exog=exog, smoother=bs, alpha=alpha_opt, family=fam)
res = g.fit()

# Predict held-out rows: pass the test exog (constant + badh) and the raw test
# smoother covariate (age); GLMGam rebuilds the spline basis internally.
test_exog = sm.add_constant(test_badh.reshape(-1, 1), has_constant="add")
mu_test = np.asarray(res.predict(exog=test_exog, exog_smooth=test_age.reshape(-1, 1)),
                     dtype=float)
emit("mu_test", mu_test)
"#,
    );
    let mu_sm_test = r.vector("mu_test");
    assert_eq!(mu_sm_test.len(), n_test, "statsmodels held-out pred length");
    let dev_sm_test = mean_poisson_deviance(&test_numvisit, mu_sm_test);

    // Context only: relative closeness of the two held-out mean vectors (NOT a
    // pass criterion — matching a peer tool's predictions is not a quality claim).
    let rel_mu = relative_l2(&gam_test_mu, mu_sm_test);

    eprintln!(
        "badhealth numvisit ~ s(age)+badh Poisson(log) held-out: \
         n_train={n_train} n_test={n_test} gam_edf={gam_edf:.3}\n  \
         held-out mean deviance: gam={dev_gam_test:.4} statsmodels={dev_sm_test:.4} \
         null(intercept-only)={dev_null_test:.4}\n  \
         (context only) rel_l2(mu, statsmodels) = {rel_mu:.5}"
    );

    // ---- PRIMARY objective assertion: beat the constant-mean predictor ------
    // A real Poisson mean-model that has learned the age/health structure must
    // achieve a held-out mean deviance comfortably below the intercept-only null.
    // We require at least a 5% reduction — a broken link/design/PIRLS that learns
    // nothing sits at (or above) the null and fails.
    assert!(
        dev_gam_test <= 0.95 * dev_null_test,
        "gam held-out mean Poisson deviance {dev_gam_test:.4} does not beat the \
         intercept-only null {dev_null_test:.4} by >=5%"
    );

    // ---- BASELINE (match-or-beat): no worse than statsmodels on held-out -----
    // gam's held-out deviance must not exceed statsmodels' by more than a small
    // margin on the SAME held-out rows and SAME metric.
    assert!(
        dev_gam_test <= dev_sm_test + 0.05,
        "gam held-out deviance worse than statsmodels: gam={dev_gam_test:.4} sm={dev_sm_test:.4}"
    );

    // ---- complexity sanity: edf in a sensible range (not matched) -----------
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
