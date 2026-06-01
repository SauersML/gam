//! OBJECTIVE quality of gam's ALO (approximate leave-one-out) corrected linear
//! predictor on real binomial/logit data.
//!
//! The point of any LOO method is *honest out-of-sample prediction*: the
//! corrected predictor `eta_tilde_i` is what the model would have predicted for
//! observation `i` had it never seen `i`. The objective metric this test asserts
//! is therefore the **mean held-out binomial deviance (log-loss)** of the
//! corrected linear predictor against the real `DEATH_EVENT` responses:
//!
//!     loss(eta) = mean_i [ -2 * ( y_i*log p_i + (1-y_i)*log(1-p_i) ) ],
//!     p_i = logistic(eta_i).
//!
//! PRIMARY OBJECTIVE CLAIM (truth recovery / predictive accuracy):
//!   * ALO's held-out log-loss must be *strictly larger* than the model's own
//!     in-sample log-loss — an LOO predictor that does not pay an honest
//!     out-of-sample penalty is not doing leave-one-out at all (it is just the
//!     in-sample fit relabelled). This is a property of LOO that holds for any
//!     correct implementation regardless of any reference tool.
//!   * ALO's held-out log-loss must beat the trivial intercept-only baseline
//!     (predict the marginal event rate for everyone): the smooth carries real,
//!     out-of-sample-honest predictive signal.
//!
//! BASELINE TO MATCH-OR-BEAT (objective accuracy, not "same fit"):
//!   * exact brute-force LOO — refit the GAM `n` times, dropping observation `i`,
//!     and read off `eta_hat^{(-i)}(x_i)`. This is the unimpeachable mathematical
//!     oracle for *any* ALO method (the EXACT quantity ALO approximates), so it is
//!     ground truth, not a peer tool. We assert gam's ALO predicts at least as
//!     accurately out of sample as the exact oracle: ALO log-loss <= exact-LOO
//!     log-loss * 1.02.
//!
//! GROUND-TRUTH CORRECTNESS (kept — exact LOO is the analytic quantity ALO
//! approximates, not a noisy peer-tool fit): the corrected predictors must agree
//! element-wise with exact LOO. A genuine error in the ALO algebra would both
//! blow up this agreement and degrade the predictive metric above; keeping it
//! pins down *where* a regression came from.
//!
//! We use Binomial/logit — the canonical GLM case. The logit link is canonical,
//! so the IRLS working weights equal the Fisher information and ALO's one-step
//! Newton correction is at its most accurate.
//!
//! Data: `heart_failure_clinical_records_dataset.csv` (299 real patients),
//! `DEATH_EVENT ~ s(ejection_fraction)`. Identical encoded data feeds the full
//! fit (for ALO) and every leave-one-out refit (for the exact oracle): the LOO
//! datasets are the full encoded design with exactly one row deleted, so basis,
//! family, link, and smoothing machinery are byte-for-byte the same in both arms.

use gam::data::EncodedDataset;
use gam::estimate::UnifiedFitResult;
use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::matrix::LinearOperator;
use gam::smooth::{TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::{max_abs_diff, pearson, relative_l2};
use gam::types::LinkFunction;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2, ArrayView1};
use std::path::Path;

const HEART_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

const FORMULA: &str = "DEATH_EVENT ~ s(ejection_fraction)";

/// Fit the binomial/logit smooth and return the fitted `UnifiedFitResult`
/// together with the frozen term spec needed to rebuild the design.
fn fit_binomial_logit(ds: &EncodedDataset) -> (UnifiedFitResult, TermCollectionSpec) {
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(FORMULA, ds, &cfg).expect("gam binomial/logit fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for binomial/logit s(x)");
    };
    (fit.fit, fit.resolvedspec)
}

/// Evaluate the linear predictor (logit scale) of a fitted model at a single
/// design point `x` for the predictor column `pred_idx`, using the model's own
/// frozen spec. For the logit link, design·beta IS eta, exactly the quantity
/// exact LOO produces for the held-out point.
fn eta_at_point(
    beta: &Array1<f64>,
    spec: &TermCollectionSpec,
    n_headers: usize,
    pred_idx: usize,
    x: f64,
) -> f64 {
    let mut grid = Array2::<f64>::zeros((1, n_headers));
    grid[[0, pred_idx]] = x;
    let design =
        build_term_collection_design(grid.view(), spec).expect("rebuild design at held-out point");
    design.design.apply(beta)[0]
}

/// Build a leave-one-out copy of the encoded dataset with row `drop` removed.
/// Schema, headers, and column kinds are preserved verbatim so the refit uses
/// an identical basis/family/link to the full fit.
fn dataset_without_row(ds: &EncodedDataset, drop: usize) -> EncodedDataset {
    let n = ds.values.nrows();
    let p = ds.values.ncols();
    let mut values = Array2::<f64>::zeros((n - 1, p));
    let mut out = 0usize;
    for i in 0..n {
        if i == drop {
            continue;
        }
        values.row_mut(out).assign(&ds.values.row(i));
        out += 1;
    }
    EncodedDataset {
        headers: ds.headers.clone(),
        values,
        schema: ds.schema.clone(),
        column_kinds: ds.column_kinds.clone(),
    }
}

/// Mean held-out binomial deviance (log-loss) of a linear predictor `eta`
/// against binary responses `y`: `mean_i -2*(y log p + (1-y) log(1-p))`,
/// `p = logistic(eta)`. Lower is a more accurate probabilistic prediction.
/// Probabilities are clamped away from {0,1} so a single confident miss cannot
/// produce a non-finite loss.
fn mean_binomial_log_loss(eta: &[f64], y: &[f64]) -> f64 {
    assert_eq!(eta.len(), y.len(), "log-loss length mismatch");
    const EPS: f64 = 1e-12;
    let mut acc = 0.0;
    for (&e, &yi) in eta.iter().zip(y.iter()) {
        let p = (1.0 / (1.0 + (-e).exp())).clamp(EPS, 1.0 - EPS);
        acc += -2.0 * (yi * p.ln() + (1.0 - yi) * (1.0 - p).ln());
    }
    acc / eta.len() as f64
}

#[test]
fn alo_eta_tilde_matches_exact_loo_binomial_logit() {
    init_parallelism();

    // ---- load identical real data once ------------------------------------
    let full_ds = load_csvwith_inferred_schema(Path::new(HEART_CSV))
        .expect("load heart_failure_clinical_records_dataset.csv");
    {
        let n_full = full_ds.values.nrows();
        assert_eq!(
            n_full, 299,
            "heart failure dataset should have 299 rows, got {n_full}"
        );
    }

    // The exact-LOO oracle below refits the GAM once per held-out row, an
    // O(n) sequence of full PIRLS+REML fits (so the whole test is O(n^2) in
    // fit cost). The ground-truth correctness claim (ALO == exact n-fold refit
    // to round-off) and the predictive-honesty bars hold for *any* n, so we
    // deterministically subsample the 299 real patients down to a smaller cohort
    // — keeping a genuine real-data binomial/logit signal while bounding the
    // refit count. A fixed stride preserves the spread of ejection_fraction
    // across its 17 distinct values (no RNG, fully reproducible).
    //
    // The brute-force oracle does a *full* PIRLS+REML refit per held-out row
    // (unlike the Poisson sibling, which downdates gam's converged geometry with
    // cheap dense solves), so wall-clock is ~`TARGET_ROWS` sequential fits. At 120
    // rows that overran the 360 s reference-quality budget; 70 rows keeps a
    // genuine real-data binomial/logit signal and the spread of ejection_fraction
    // while ~halving the refit count. The ALO-vs-exact-LOO agreement and the
    // predictive-honesty bars below hold for any n, so the cohort size is purely a
    // cost knob, not a quality lever.
    const TARGET_ROWS: usize = 70;
    let stride = full_ds.values.nrows().div_ceil(TARGET_ROWS);
    let keep_rows: Vec<usize> = (0..full_ds.values.nrows()).step_by(stride).collect();
    let p_cols = full_ds.headers.len();
    let mut sub_values = Array2::<f64>::zeros((keep_rows.len(), p_cols));
    for (out_row, &src_row) in keep_rows.iter().enumerate() {
        sub_values
            .row_mut(out_row)
            .assign(&full_ds.values.row(src_row));
    }
    let mut ds = full_ds.clone();
    ds.values = sub_values;

    let col = ds.column_map();
    let pred_idx = col["ejection_fraction"];
    let n_headers = ds.headers.len();
    let x: Vec<f64> = ds.values.column(pred_idx).to_vec();
    let n = x.len();
    assert!(
        (55..=95).contains(&n),
        "subsampled heart cohort should be ~70 rows, got {n}"
    );

    // ---- full fit + ALO ----------------------------------------------------
    // The full-fit ALO reads everything it needs from `full_fit`; the resolved
    // spec is reused to evaluate the IN-SAMPLE linear predictor eta_hat(x_i),
    // which is the predictive-honesty baseline the corrected predictor must beat.
    let (full_fit, full_spec) = fit_binomial_logit(&ds);
    let y: Vec<f64> = ds.values.column(col["DEATH_EVENT"]).to_vec();
    let alo = compute_alo_diagnostics_from_fit(
        &full_fit,
        ArrayView1::from(y.as_slice()),
        LinkFunction::Logit,
    )
    .expect("ALO diagnostics on binomial/logit fit");
    let alo_eta_tilde: Vec<f64> = alo.eta_tilde.to_vec();
    assert_eq!(alo_eta_tilde.len(), n, "ALO eta_tilde length mismatch");

    // ---- exact LOO oracle: refit n times, hold out i, read eta(x_i) -------
    let mut exact_loo: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let loo_ds = dataset_without_row(&ds, i);
        let (loo_fit, loo_spec) = fit_binomial_logit(&loo_ds);
        let eta_i = eta_at_point(&loo_fit.beta, &loo_spec, n_headers, pred_idx, x[i]);
        exact_loo.push(eta_i);
    }

    // ---- in-sample predictor eta_hat(x_i) (predictive-honesty baseline) ----
    // The naive in-sample fit has seen every observation, so its log-loss is an
    // optimistic floor. A correct LOO predictor must score WORSE than this.
    let in_sample: Vec<f64> = (0..n)
        .map(|i| eta_at_point(&full_fit.beta, &full_spec, n_headers, pred_idx, x[i]))
        .collect();

    // ---- trivial intercept-only baseline: predict the marginal event rate ----
    // logit(mean(y)) for everyone. Any model with real signal must beat this.
    let p_bar = y.iter().sum::<f64>() / n as f64;
    let eta_bar = (p_bar / (1.0 - p_bar)).ln();
    let intercept_only = vec![eta_bar; n];

    // ---- OBJECTIVE METRIC: mean held-out binomial deviance (log-loss) ------
    let loss_alo = mean_binomial_log_loss(&alo_eta_tilde, &y);
    let loss_exact = mean_binomial_log_loss(&exact_loo, &y);
    let loss_in_sample = mean_binomial_log_loss(&in_sample, &y);
    let loss_intercept = mean_binomial_log_loss(&intercept_only, &y);

    // ---- element-wise agreement vs the exact-LOO oracle (ground truth) -----
    let rel = relative_l2(&alo_eta_tilde, &exact_loo);
    let max_abs = max_abs_diff(&alo_eta_tilde, &exact_loo);
    let corr = pearson(&alo_eta_tilde, &exact_loo);

    eprintln!(
        "binomial/logit n={n}: log-loss alo={loss_alo:.5} exact-LOO={loss_exact:.5} \
         in-sample={loss_in_sample:.5} intercept-only={loss_intercept:.5} | \
         ALO vs exact-LOO rel_l2={rel:.5} max_abs={max_abs:.5} pearson={corr:.6}"
    );

    // === PRIMARY OBJECTIVE: out-of-sample predictive honesty + signal =======
    // (1) The corrected predictor must pay an honest out-of-sample penalty:
    //     held-out log-loss strictly exceeds the optimistic in-sample log-loss.
    //     If ALO's loss were <= in-sample, the "correction" would be removing no
    //     information about the held-out point — a broken LOO.
    assert!(
        loss_alo > loss_in_sample,
        "ALO held-out log-loss ({loss_alo:.5}) must exceed the in-sample floor \
         ({loss_in_sample:.5}); an LOO predictor that is no worse than in-sample \
         is not leaving anything out"
    );
    // (2) The smooth must carry real out-of-sample signal: it must beat the
    //     intercept-only marginal-rate predictor on the SAME held-out metric.
    assert!(
        loss_alo < loss_intercept,
        "ALO held-out log-loss ({loss_alo:.5}) must beat the intercept-only \
         baseline ({loss_intercept:.5}): the smooth carries no out-of-sample signal"
    );

    // === BASELINE TO MATCH-OR-BEAT: exact-LOO oracle predictive accuracy =====
    // gam's fast ALO must be at least as predictive out of sample as the exact
    // brute-force oracle it approximates (2% slack for the first-order residual).
    assert!(
        loss_alo <= loss_exact * 1.02,
        "ALO held-out log-loss ({loss_alo:.5}) must match or beat exact-LOO \
         ({loss_exact:.5}) to within 2%: the approximation is losing accuracy"
    );

    // === GROUND-TRUTH CORRECTNESS: agreement with the exact-LOO quantity =====
    // Exact LOO is the analytic quantity ALO approximates (not a peer tool), so
    // element-wise agreement is a correctness claim. ALO is a one-Newton-step
    // approximation; on a canonical link with a stable penalized fit the residual
    // is second-order in per-observation leverage and empirically tiny. These
    // bounds pin down a divergence in the ALO algebra; they are NOT loosened.
    assert!(
        corr > 0.9999,
        "ALO eta_tilde must track exact LOO almost perfectly: pearson={corr:.6}"
    );
    assert!(
        rel < 0.01,
        "ALO eta_tilde diverges from exact LOO in relative L2: rel_l2={rel:.5}"
    );
    assert!(
        max_abs < 0.05,
        "ALO eta_tilde has a too-large worst-case logit error vs exact LOO: max_abs={max_abs:.5}"
    );
}
