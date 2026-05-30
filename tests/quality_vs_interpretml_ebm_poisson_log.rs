//! End-to-end quality: gam's penalized PoissonGAM must reach predictive parity
//! with **InterpretML's `ExplainableBoostingRegressor`** (the ML-world glassbox
//! GAM) on the exact same count data, under the exact same objective.
//!
//! InterpretML's EBM is the machine-learning lineage's answer to the GAM: a
//! purely additive model `eta = f1(x1) + f2(x2) + ...` learned by cyclic
//! gradient boosting on shallow trees, with an explicit Poisson-deviance
//! objective (`objective="poisson_deviance"`, log link). It is the best-in-class
//! ML comparator here precisely because it shares gam's *additive* structure and
//! gam's *objective* (Poisson deviance) while using a completely different
//! estimation strategy (boosted bins vs REML penalized PIRLS). If the two agree,
//! it is because both recovered the same smooth count surface — strong, method-
//! independent evidence that gam's log-link inversion and smoothing are correct.
//!
//! Setup (identical bytes fed to both engines):
//!   * synthetic n=250, x1,x2 ~ U[0,10], seed 20260530,
//!     true_eta = 0.5 + 0.2*sin(pi*x1/5) + 0.15*cos(pi*x2/5),
//!     count ~ Poisson(exp(true_eta)).
//!   * a fixed 70/30 train/test split (indices computed identically Rust-side and
//!     handed to both engines as a `fold` column: 1 = test, 0 = train).
//!   * gam fits `count ~ s(x1, k=6) + s(x2, k=6)` (Poisson, log link, REML) on the
//!     TRAIN rows, then predicts mu on the TEST rows.
//!   * EBM fits on the TRAIN rows, predicts mu on the TEST rows.
//!
//! Assertions (Poisson is the second-most-common applied GLM family; the budgets
//! absorb boosting-vs-REML slack but still bite):
//!   1. Poisson deviance on the TEST fold agrees: deviance(y,mu) =
//!      2*sum(y*log(y/mu) - (y-mu)) is the *same* objective both engines minimize,
//!      so rel_l2 of the per-row deviance contributions must be < 0.10.
//!   2. Fitted means on the exp scale agree: rel_l2(mu_gam, mu_ebm) < 0.05 over the
//!      test fold — this is the log-link inversion fidelity test.
//!   3. Ranked mean prediction Pearson > 0.97: the two additive surfaces must
//!      order the test points near-identically.
//! A genuine divergence is a real bug in gam's Poisson/PIRLS path, never a reason
//! to weaken these bounds.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};

const N: usize = 250;
const SEED: u64 = 20260530;

fn truth_eta(x1: f64, x2: f64) -> f64 {
    let pi = std::f64::consts::PI;
    0.5 + 0.2 * (pi * x1 / 5.0).sin() + 0.15 * (pi * x2 / 5.0).cos()
}

/// Per-row Poisson deviance contribution: 2*(y*log(y/mu) - (y - mu)), with the
/// y=0 limit y*log(y/mu) -> 0. This is the exact quantity both engines minimize.
fn poisson_dev_unit(y: f64, mu: f64) -> f64 {
    let term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    2.0 * (term - (y - mu))
}

#[test]
fn gam_poisson_log_matches_interpretml_ebm() {
    init_parallelism();

    // ---- synthetic count data + fixed train/test fold (identical to both) ----
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform 0..10");
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let pois = Poisson::new(truth_eta(a, b).exp()).expect("poisson mean > 0");
        let count: f64 = pois.sample(&mut rng);
        x1.push(a);
        x2.push(b);
        y.push(count);
    }

    // Deterministic 70/30 split by row index: every 10th-and-beyond-7 row is
    // test. fold[i] == 1.0 means TEST, 0.0 means TRAIN — handed verbatim to EBM.
    let fold: Vec<f64> = (0..N).map(|i| if i % 10 >= 7 { 1.0 } else { 0.0 }).collect();
    let train_idx: Vec<usize> = (0..N).filter(|&i| fold[i] == 0.0).collect();
    let test_idx: Vec<usize> = (0..N).filter(|&i| fold[i] == 1.0).collect();
    assert!(
        train_idx.len() > 100 && test_idx.len() > 50,
        "split sanity: {} train / {} test",
        train_idx.len(),
        test_idx.len()
    );

    // ---- fit with gam on TRAIN rows: count ~ s(x1,k=6)+s(x2,k=6), Poisson -----
    let headers = ["x1", "x2", "count"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = train_idx
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode train dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("count ~ s(x1, k=6) + s(x2, k=6)", &ds, &cfg)
        .expect("gam poisson fit on train fold");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild the frozen design at the TEST covariates; for a log link
    // design*beta is eta_hat and exp(eta_hat) is the fitted mean mu_hat.
    let n_test = test_idx.len();
    let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (row, &i) in test_idx.iter().enumerate() {
        grid[[row, x1_idx]] = x1[i];
        grid[[row, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at test points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();
    assert_eq!(gam_mu.len(), n_test, "gam test-mu length mismatch");

    // ---- fit the SAME data with InterpretML EBM (Poisson deviance) -----------
    // ExplainableBoostingRegressor with the Poisson-deviance objective is the ML
    // analog of a PoissonGAM: additive shape functions of x1 and x2, log link.
    // We pass the fold column so the Python side splits on the EXACT same rows.
    let r = run_python(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("count", &y),
            Column::new("fold", &fold),
        ],
        r#"
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

x1   = np.asarray(df["x1"],    dtype=float)
x2   = np.asarray(df["x2"],    dtype=float)
y    = np.asarray(df["count"], dtype=float)
fold = np.asarray(df["fold"],  dtype=float)

train = fold == 0.0
test  = fold == 1.0
X = np.column_stack([x1, x2])

# Poisson-deviance objective => log link, additive shape functions only
# (interactions=0 keeps it a pure additive GAM, matching s(x1)+s(x2)).
ebm = ExplainableBoostingRegressor(
    objective="poisson_deviance",
    interactions=0,
    random_state=20260530,
)
ebm.fit(X[train], y[train])
mu = np.asarray(ebm.predict(X[test]), dtype=float)
# EBM's regression predictions are on the response (mean) scale already; guard
# against any non-positive mean before it reaches a Poisson deviance.
mu = np.clip(mu, 1e-8, None)
emit("mu", mu)
"#,
    );
    let ebm_mu = r.vector("mu");
    assert_eq!(ebm_mu.len(), n_test, "EBM test-mu length mismatch");

    // ---- compare on the TEST fold --------------------------------------------
    let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();

    // (1) Poisson deviance agreement: per-row deviance contributions under each
    //     engine's fitted mean. Same objective, so these must nearly coincide.
    let gam_dev: Vec<f64> = y_test
        .iter()
        .zip(&gam_mu)
        .map(|(&yi, &mu)| poisson_dev_unit(yi, mu))
        .collect();
    let ebm_dev: Vec<f64> = y_test
        .iter()
        .zip(ebm_mu)
        .map(|(&yi, &mu)| poisson_dev_unit(yi, mu))
        .collect();
    let rel_dev = relative_l2(&gam_dev, &ebm_dev);

    // (2) exp-scale fitted-mean agreement: the log-link inversion fidelity test.
    let rel_mu = relative_l2(&gam_mu, ebm_mu);

    // (3) ranked mean-prediction correlation (Spearman = Pearson on ranks): the
    //     two additive surfaces must order the test points near-identically.
    let gam_rank = ranks(&gam_mu);
    let ebm_rank = ranks(ebm_mu);
    let corr_rank = pearson(&gam_rank, &ebm_rank);

    let gam_test_dev: f64 = gam_dev.iter().sum();
    let ebm_test_dev: f64 = ebm_dev.iter().sum();
    eprintln!(
        "EBM-vs-gam poisson: n_train={} n_test={} gam_edf={gam_edf:.3} \
         gam_test_dev={gam_test_dev:.3} ebm_test_dev={ebm_test_dev:.3} \
         rel_l2(dev)={rel_dev:.4} rel_l2(mu)={rel_mu:.4} pearson(rank_mu)={corr_rank:.5}",
        train_idx.len(),
        n_test
    );

    // Poisson deviance is the literal objective both engines minimize; on a held-
    // out fold of a smooth low-frequency truth the per-row deviance contributions
    // must agree to within 10% relative L2 (boosting-vs-REML slack).
    assert!(
        rel_dev < 0.10,
        "test-fold Poisson deviance diverges between gam and EBM: rel_l2={rel_dev:.4}"
    );
    // The fitted means on the exp scale are the same quantity recovered two ways;
    // 5% relative L2 is a tight log-link-inversion fidelity budget.
    assert!(
        rel_mu < 0.05,
        "test-fold fitted means disagree on the exp scale: rel_l2={rel_mu:.4}"
    );
    // Both additive surfaces resolve the same monotone-within-period signal, so
    // their rankings of the test points must be >0.97 correlated.
    assert!(
        corr_rank > 0.97,
        "ranked mean predictions disagree between gam and EBM: pearson={corr_rank:.5}"
    );
}

/// Fractional ranks (average ties), so Pearson on these is Spearman correlation.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("no NaN in predictions"));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && v[order[j]] == v[order[i]] {
            j += 1;
        }
        // average rank (1-based) over the tie block [i, j)
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &idx in &order[i..j] {
            out[idx] = avg;
        }
        i = j;
    }
    out
}
