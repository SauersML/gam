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
use gam::init_parallelism;
use gam::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::test_support::reference::{Column, relative_l2, run_python};
use ndarray::{Array1, Array2};

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
    let var_x: f64 =
        train_idx.iter().map(|&i| (x[i] - mean_x).powi(2)).sum::<f64>() / n_train as f64;
    let sd_x = var_x.sqrt().max(1e-12);
    let z_of = |xi: f64| -> f64 { (xi - mean_x) / sd_x };
    // Centering offsets for the quadratic/cubic powers, computed on train rows.
    let m2: f64 = train_idx
        .iter()
        .map(|&i| z_of(x[i]).powi(2))
        .sum::<f64>()
        / n_train as f64;
    let m3: f64 = train_idx
        .iter()
        .map(|&i| z_of(x[i]).powi(3))
        .sum::<f64>()
        / n_train as f64;

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
    let mu_true_test: Vec<f64> = test_idx.iter().map(|&i| mu_true[i]).collect();

    // OBJECTIVE metric 1: held-out mean Poisson deviance on gam's own preds.
    let dev_gam_test = mean_poisson_deviance(&y_test, &mu_gam_test);
    // OBJECTIVE metric 2: truth-recovery RMSE of the fitted mean over all rows.
    let rmse_gam_truth = rmse_vs(&mu_gam_full, &mu_true);

    // ---- fit the SAME train design with statsmodels (baseline to match/beat) -
    // The CSV carries the FULL response, the FULL design columns, and a
    // train/test flag, so statsmodels fits on train and predicts on the same
    // held-out rows from byte-identical data.
    let train_flag: Vec<f64> = (0..n)
        .map(|i| if is_test(i) { 0.0 } else { 1.0 })
        .collect();
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
