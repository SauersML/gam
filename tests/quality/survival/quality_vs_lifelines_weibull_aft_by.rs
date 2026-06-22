//! End-to-end quality: gam's parametric **Weibull AFT** survival fit with a
//! **by-factor smooth** covariate effect must recover the KNOWN data-generating
//! survival surface — the objective ground truth this test asserts.
//!
//! ## Objective metric (the pass/fail claim)
//!
//! The data are simulated from an explicit AFT model with known per-group
//! Weibull baselines and known per-group acceleration slopes, so the true
//! survival function `S_true(t | x, g)` is computable in closed form (see
//! `true_survival` below). The PRIMARY assertion is that gam's predicted
//! survival surface recovers that ground truth:
//!   `RMSE( S_gam(t|x,g), S_true(t|x,g) ) <= 0.05`  over a (group × x × t) grid.
//! Survival probabilities live in [0,1], so a root-mean-square error of 0.05 is
//! a tight, signal-appropriate bar for a ~100-event-per-stratum, partially
//! censored AFT fit (it is a few percent of the [0,1] range and well under the
//! Monte-Carlo / finite-sample spread of the surface at this n).
//!
//! `lifelines.WeibullAFTFitter` (the mature, standard parametric-AFT reference)
//! is fit on the SAME data per stratum and kept ONLY as a baseline-to-beat on
//! that same truth-recovery metric: gam's RMSE-to-truth must be no worse than
//! 1.10× lifelines' RMSE-to-truth. We never assert "gam reproduces lifelines'
//! output" — matching another fit proves nothing about correctness; recovering
//! the generating function does. lifelines is held to the identical objective
//! yardstick (error vs the true surface), not used as the target itself.
//!
//! As a structural sanity check we also assert gam's survival surface is a valid
//! survival function on the grid: every `S` lies in [0,1] and is non-increasing
//! in `t` for each (group, x) — a property the AFT factorization must satisfy
//! regardless of any reference tool.
//!
//! ## What this benchmarks
//!
//! gam fits a single shared Weibull baseline cumulative hazard
//!   `H0(t) = (t / scale)^shape`,   `log H0(t) = shape·(log t − log scale)`
//! and adds a covariate log-cumulative-hazard term built from
//! `x + s(x, by=group)`. The by-factor smooth `s(x, by=group)` gives each
//! group its OWN covariate (acceleration) curve, so the survival function
//! differs by stratum through a stratum-specific multiplicative shift of the
//! shared baseline hazard:
//!   `S(t | x, g) = exp( -exp( log H0(t) + f_g(x) ) )`.
//! The objective metric above validates that this factorization recovers the
//! true two-stratum AFT surface.
//!
//! `group` is fed to gam as a categorical label ("A"/"B"): a numeric "0"/"1"
//! column infers as Binary and turns `s(x, by=group)` into a single continuous
//! varying-coefficient smooth (basis × value, which zeroes group A at value 0),
//! NOT the per-level by-FACTOR expansion this test is meant to validate.
//!
//! ## Data (fixed seed, n=200, 100 per group)
//!
//! Group A baseline ~ Weibull(scale=0.8, shape=1.1); Group B baseline ~
//! Weibull(scale=1.5, shape=1.1) — a COMMON shape (the shared-baseline `log t`
//! slope gam's single baseline can represent) with distinct scales and distinct,
//! opposite-sign AFT acceleration slopes of x (the per-group signal the by-factor
//! smooth must capture as a proportional log-hazard shift). Both with x ~ N(0,1).
//! Right-censoring via an independent exponential keeps ~20-35% censored —
//! realistic, and identical rows go to both engines.

use csv::StringRecord;
use gam::families::survival::construction::evaluate_survival_baseline;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::{Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Normal, Weibull};
use std::path::Path;

const N_PER_GROUP: usize = 100;
const SEED: u64 = 20260529;

// Group baselines (rand_distr::Weibull::new(scale, shape): CDF 1 - exp(-(t/scale)^shape)).
//
// The two strata share a COMMON Weibull SHAPE and differ only in scale and in the
// AFT acceleration slope of `x`. This is deliberate and load-bearing: the gam
// model this test fits is `s(x, by=group)` over a SINGLE shared baseline
// cumulative hazard `H0(t) = (t/scale)^shape` (one shape coefficient — the slope
// of `log H0` in `log t`). A by-factor smooth can express a *proportional*
// per-group log-hazard shift (a different scale and a different x-acceleration per
// group), but it CANNOT bend the shared baseline's `log t` slope per group, so a
// per-group shape would make gam's shared-baseline model structurally
// mis-specified — it could never recover the truth, and the comparison against
// lifelines (which fits a fully independent Weibull, including its own shape, per
// stratum) would be apples-to-oranges. With a shared shape the gam model is
// correctly specified for the DGP, so the truth-recovery and match-or-beat claims
// are a fair, like-for-like test of the by-factor acceleration signal. Distinct
// scales (0.8 vs 1.5) and opposite-sign acceleration slopes (BETA_A vs BETA_B)
// keep the per-group by-factor signal the smooth must recover.
const SCALE_A: f64 = 0.8;
const SHAPE_A: f64 = 1.1;
const SCALE_B: f64 = 1.5;
const SHAPE_B: f64 = 1.1;
// Group-specific AFT acceleration of x: log(T) gets `BETA_g * x` added, so the
// x slope differs by group — the signal the by-factor smooth must recover.
const BETA_A: f64 = 0.35;
const BETA_B: f64 = -0.25;

/// Closed-form ground-truth survival under the data-generating AFT model.
///
/// The DGP is `T = T0 · exp(beta_g · x)` with `T0 ~ Weibull(scale_g, shape_g)`,
/// whose survival is `P(T0 > s) = exp(-(s/scale_g)^shape_g)`. Hence
///   `S(t | x, g) = P(T0 > t·exp(-beta_g·x))
///               = exp( -( t·exp(-beta_g·x) / scale_g )^shape_g )`.
/// This is the objective truth gam (and lifelines) are measured against.
fn true_survival(t: f64, x: f64, group_a: bool) -> f64 {
    let (scale, shape, beta) = if group_a {
        (SCALE_A, SHAPE_A, BETA_A)
    } else {
        (SCALE_B, SHAPE_B, BETA_B)
    };
    let s = t * (-beta * x).exp();
    (-((s / scale).powf(shape))).exp()
}

#[test]
fn gam_weibull_aft_by_factor_recovers_true_survival() {
    init_parallelism();

    // ---- synthesize identical data for both engines ----------------------
    // group A rows first, then group B, so categorical inference maps A->0,
    // B->1 (first-appearance level order). Both engines see the same rows.
    let mut rng = StdRng::seed_from_u64(SEED);
    let weib_a = Weibull::new(SCALE_A, SHAPE_A).expect("weibull A");
    let weib_b = Weibull::new(SCALE_B, SHAPE_B).expect("weibull B");
    let xdist = Normal::new(0.0, 1.0).expect("normal x");
    // Censoring exponentials tuned per group so each stratum keeps a healthy
    // fraction of observed events (mean censoring time > median survival time).
    let cens_a = Exp::new(0.25_f64).expect("exp censor A");
    let cens_b = Exp::new(0.20_f64).expect("exp censor B");

    let n = 2 * N_PER_GROUP;
    let mut time = Vec::<f64>::with_capacity(n);
    let mut event = Vec::<f64>::with_capacity(n);
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g_code = Vec::<f64>::with_capacity(n); // 0.0=A, 1.0=B
    let mut is_a = Vec::<bool>::with_capacity(n);

    for group_a in [true, false] {
        let (weib, beta, cens) = if group_a {
            (&weib_a, BETA_A, &cens_a)
        } else {
            (&weib_b, BETA_B, &cens_b)
        };
        for _ in 0..N_PER_GROUP {
            let xi = xdist.sample(&mut rng);
            // AFT acceleration: multiply the baseline time by exp(beta_g * x).
            let t0 = weib.sample(&mut rng);
            let ti = t0 * (beta * xi).exp();
            let ci = cens.sample(&mut rng) + 1e-3;
            let observed = ti.min(ci);
            let ev = if ti <= ci { 1.0 } else { 0.0 };
            time.push(observed);
            event.push(ev);
            x.push(xi);
            g_code.push(if group_a { 0.0 } else { 1.0 });
            is_a.push(group_a);
        }
    }

    // ---- fit with gam: Weibull AFT + by-factor smooth on x ----------------
    // survival_likelihood="weibull" selects the parametric Weibull baseline
    // (linear log-cumulative-hazard time basis whose two coefficients recover
    // scale/shape). The covariate side `x + s(x, by=group)` gives each group
    // its own acceleration curve. The `survmodel(...)` term states the intent
    // in-formula; the likelihood mode is driven by the config field.
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "x".to_string(),
        "group".to_string(),
    ];
    // `group` MUST be fed to gam as a categorical label ("A"/"B"), not the
    // numeric code: schema inference treats "0"/"1" as a Binary numeric column,
    // which makes `s(x, by=group)` a single continuous varying-coefficient
    // smooth (basis * value, zeroing out group A at value 0). Only a Categorical
    // by-variable triggers the per-level by-FACTOR expansion gam advertises here
    // (one smooth per level + an unpenalized treatment-coded factor main effect).
    // Group A rows come first, so first-appearance level order gives A->0, B->1,
    // matching the numeric `g_code` used by the prediction grid and the Python
    // comparator (which filters on the numeric `group` column it receives).
    let group_label = |group_a: bool| if group_a { "A" } else { "B" };
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                time[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
                group_label(is_a[i]).to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode survival dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["group"];

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, event) ~ x + s(x, by=group) + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &ds,
        &cfg,
    )
    .expect("gam Weibull-AFT by-factor fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for survival_likelihood=weibull");
    };

    // gam's shared baseline Weibull (scale, shape), recovered from the linear
    // time-basis coefficients — printed for context only.
    let gam_scale = fit
        .baseline_cfg
        .scale
        .expect("gam recovers a Weibull baseline scale");
    let gam_shape = fit
        .baseline_cfg
        .shape
        .expect("gam recovers a Weibull baseline shape");

    // Covariate coefficient slice: beta = [time(2 cols), covariate...]; the
    // covariate block begins at `time_base_ncols`.
    let cov_start = fit.time_base_ncols;
    let beta = &fit.fit.beta;
    assert!(
        beta.len() > cov_start,
        "expected covariate coefficients after the {cov_start} time columns, got beta.len()={}",
        beta.len()
    );

    // Survival prediction grid. For each group evaluate S(t | x, g) at three
    // representative covariate values and a shared time grid, using gam's OWN
    // forward map: log H(t|x,g) = log H0(t) + covariate_design(x,g)·cov_beta,
    // S = exp(-exp(log H)). log H0 comes from gam's recovered baseline via
    // `evaluate_survival_baseline`, so the reconstruction is self-consistent
    // with whatever gam fit (no hand-rederived offsets).
    let x_eval = [-1.0_f64, 0.0, 1.0];
    let t_grid: Vec<f64> = (1..=10).map(|k| 0.25 * k as f64).collect();

    // Build the covariate design once at every (group, x_eval) prediction row.
    let pred_groups = [0.0_f64, 1.0];
    let n_pred_rows = pred_groups.len() * x_eval.len();
    let mut grid = Array2::<f64>::zeros((n_pred_rows, ds.headers.len()));
    let mut row = 0usize;
    for &gc in &pred_groups {
        for &xv in &x_eval {
            grid[[row, x_idx]] = xv;
            grid[[row, g_idx]] = gc;
            row += 1;
        }
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at prediction rows");
    let dense = design.design.to_dense();
    assert_eq!(
        dense.ncols(),
        beta.len() - cov_start,
        "covariate design width must match the covariate coefficient slice"
    );
    let cov_beta = beta.slice(s![cov_start..]).to_owned();

    // gam predicted survival surface and the matched closed-form TRUTH surface,
    // both laid out group-major / x-major / t-minor on the identical grid.
    let mut gam_surv: Vec<f64> = Vec::with_capacity(n_pred_rows * t_grid.len());
    let mut true_surv: Vec<f64> = Vec::with_capacity(n_pred_rows * t_grid.len());
    let mut idx = 0usize;
    for &gc in &pred_groups {
        let group_a = gc == 0.0;
        for &xv in &x_eval {
            let cov_eta: f64 = dense.row(idx).dot(&cov_beta);
            for &t in &t_grid {
                let (log_h0, _) = evaluate_survival_baseline(t, &fit.baseline_cfg)
                    .expect("evaluate gam baseline log-cumulative-hazard");
                let s = (-(log_h0 + cov_eta).exp()).exp();
                gam_surv.push(s);
                true_surv.push(true_survival(t, xv, group_a));
            }
            idx += 1;
        }
    }

    // ---- STRUCTURAL CHECK: gam's surface is a valid survival function ------
    // Every S in [0,1] and non-increasing in t within each (group, x) block.
    let nt = t_grid.len();
    for r in 0..n_pred_rows {
        let block = &gam_surv[r * nt..(r + 1) * nt];
        for (k, &s) in block.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&s),
                "gam survival out of [0,1] at row {r}, t-index {k}: S={s}"
            );
            if k > 0 {
                assert!(
                    s <= block[k - 1] + 1e-9,
                    "gam survival not non-increasing at row {r}, t-index {k}: {} -> {s}",
                    block[k - 1]
                );
            }
        }
    }

    // ---- fit the SAME data per stratum with lifelines (BASELINE to beat) ---
    // For each group fit an independent WeibullAFTFitter on `time ~ x` and emit
    // that stratum's predicted survival surface on the identical (x_eval × t_grid)
    // grid. lifelines is held to the SAME truth-recovery yardstick as gam; it is
    // never the target, only a peer whose error-vs-truth gam must match or beat.
    let py = run_python(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("x", &x),
            Column::new("group", &g_code),
        ],
        r#"
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

x_eval = [-1.0, 0.0, 1.0]
t_grid = [0.25 * k for k in range(1, 11)]

frame = pd.DataFrame({
    "time": np.asarray(df["time"], dtype=float),
    "event": np.asarray(df["event"], dtype=float),
    "x": np.asarray(df["x"], dtype=float),
    "group": np.asarray(df["group"], dtype=float),
})

surv_rows = []
for gc in (0.0, 1.0):
    sub = frame[frame["group"] == gc][["time", "event", "x"]].reset_index(drop=True)
    aft = WeibullAFTFitter()
    aft.fit(sub, duration_col="time", event_col="event")
    # Predicted survival at each x_eval over the shared time grid.
    pred_df = pd.DataFrame({"x": x_eval})
    sf = aft.predict_survival_function(pred_df, times=t_grid)
    # sf: rows=times (t_grid order), cols=rows of pred_df (x_eval order).
    for j in range(len(x_eval)):
        col = sf.iloc[:, j].to_numpy()
        surv_rows.extend(float(v) for v in col)

emit("surv", surv_rows)
"#,
    );

    let life_surv = py.vector("surv");
    assert_eq!(
        life_surv.len(),
        gam_surv.len(),
        "lifelines survival surface length mismatch: gam={} lifelines={}",
        gam_surv.len(),
        life_surv.len()
    );

    let cens_a_frac = 1.0
        - is_a
            .iter()
            .zip(&event)
            .filter(|(a, _)| **a)
            .map(|(_, e)| *e)
            .sum::<f64>()
            / N_PER_GROUP as f64;
    let cens_b_frac = 1.0
        - is_a
            .iter()
            .zip(&event)
            .filter(|(a, _)| !**a)
            .map(|(_, e)| *e)
            .sum::<f64>()
            / N_PER_GROUP as f64;

    // Objective truth-recovery errors for both engines, on the same grid/truth.
    let gam_rmse_truth = rmse(&gam_surv, &true_surv);
    let life_rmse_truth = rmse(life_surv, &true_surv);
    // Reference closeness, printed for context only (NOT a pass criterion).
    let rel_gam_vs_life = relative_l2(&gam_surv, life_surv);

    eprintln!(
        "weibull-AFT by-factor: n={n} censoring A={cens_a_frac:.2} B={cens_b_frac:.2}\n  \
         gam baseline: scale={gam_scale:.4} shape={gam_shape:.4}\n  \
         RMSE(S) vs TRUTH: gam={gam_rmse_truth:.4} lifelines={life_rmse_truth:.4}\n  \
         (context) rel_l2(gam, lifelines)={rel_gam_vs_life:.4}"
    );

    // (1) PRIMARY: gam recovers the true generating survival surface. 0.05 RMSE
    // on S in [0,1] is a tight, signal-appropriate bar for a ~100-event,
    // partially-censored AFT fit; a broken by-factor factorization (collapsed
    // strata, wrong acceleration sign) would blow far past it.
    assert!(
        gam_rmse_truth <= 0.05,
        "gam fails to recover the true Weibull-AFT survival surface: RMSE(S vs truth)={gam_rmse_truth:.4} > 0.05"
    );

    // (2) MATCH-OR-BEAT the mature reference on ACCURACY (not on agreement):
    // gam's error vs the true surface must be no worse than 1.10× lifelines'
    // error vs that same true surface. This is an apples-to-apples objective
    // comparison — both engines measured against ground truth.
    assert!(
        gam_rmse_truth <= life_rmse_truth * 1.10,
        "gam less accurate than lifelines vs truth: gam RMSE={gam_rmse_truth:.4}, lifelines RMSE={life_rmse_truth:.4} (1.10x={:.4})",
        life_rmse_truth * 1.10
    );
}

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Dataset SOURCE: the Veterans' Administration lung-cancer randomized trial
// (`veteran` in the R `survival` package; Kalbfleisch & Prentice, "The
// Statistical Analysis of Failure Time Data"). 137 patients, columns
// `time` (days), `status` (1=death, 0=censored — 9 censored), the numeric
// covariate `karno` (Karnofsky performance score, the dominant prognostic
// signal) and the four-level factor `celltype`
// (squamous / smallcell / adeno / large), shipped at
// `bench/datasets/veteran_lung.csv`.
//
// This arm exercises the SAME gam capability as the synthetic test above —
// a parametric **Weibull AFT** survival fit with a **by-FACTOR smooth**
// covariate effect, `s(karno, by=celltype)`, giving each cell type its own
// karnofsky→risk curve over a shared Weibull baseline cumulative hazard.
//
// Because this is real data the data-generating survival surface is UNKNOWN,
// so RMSE-to-truth is not computable. The objective, tool-free quality metric
// is therefore the **held-out concordance index** (Harrell's C): a fixed,
// deterministic train/test split, fit gam on train, score the held-out
// patients by gam's OWN covariate log-cumulative-hazard risk, and assert how
// well that risk ranking agrees with the observed (time, event) ordering.
// Higher covariate log-cumulative-hazard ⇒ higher hazard ⇒ shorter survival,
// so a well-fit model gives a high C-index (0.5 = random, 1.0 = perfect).
//   PRIMARY (objective): held-out C-index >= 0.62 — well above the 0.5 random
//     baseline for a ~30-patient held-out set; a broken by-factor fit (wrong
//     karnofsky sign, collapsed cell-type strata) would not clear it.
//   BASELINE (match-or-beat): lifelines.WeibullAFTFitter, the mature standard
//     parametric-AFT reference, is fit on the SAME train rows and scored on
//     the SAME held-out patients (by predicted expected survival time, which
//     it turns into its own C-index via lifelines.utils.concordance_index);
//     gam's held-out C must be no worse than lifelines' C minus a 0.03 margin.
//     lifelines is a yardstick to match-or-beat on the identical held-out
//     metric, never a fitted output to reproduce.

/// Harrell's concordance index for survival risk scores. `risk[i]` is monotone
/// INCREASING in hazard (higher risk ⇒ shorter expected survival). A pair
/// `(i, j)` is comparable when the earlier observed time belongs to an event;
/// it is concordant when that earlier-failing subject also carries the higher
/// risk. Risk ties on a comparable pair count as half-concordant. Returns the
/// fraction of comparable pairs that are concordant (0.5 = random ordering).
fn concordance_index(risk: &[f64], time: &[f64], event: &[f64]) -> f64 {
    assert_eq!(
        risk.len(),
        time.len(),
        "concordance: risk/time length mismatch"
    );
    assert_eq!(
        time.len(),
        event.len(),
        "concordance: time/event length mismatch"
    );
    let n = risk.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Identify the earlier-failing member of the pair; the pair is only
            // comparable when that earlier observed time is an actual event.
            let (early, late) = if time[i] < time[j] {
                (i, j)
            } else if time[j] < time[i] {
                (j, i)
            } else {
                // Equal observed times carry no usable ordering information.
                continue;
            };
            if event[early] != 1.0 {
                continue;
            }
            comparable += 1.0;
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if risk[early] == risk[late] {
                concordant += 0.5;
            }
        }
    }
    assert!(
        comparable > 0.0,
        "no comparable survival pairs — degenerate held-out set"
    );
    concordant / comparable
}

#[test]
fn gam_weibull_aft_by_factor_recovers_true_survival_on_real_data() {
    init_parallelism();

    // ---- load the real Veterans' lung-cancer trial ------------------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/veteran_lung.csv"
    )))
    .expect("load veteran_lung.csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let celltype_idx = col["celltype"];

    let time: Vec<f64> = ds.values.column(time_idx).to_vec();
    let status: Vec<f64> = ds.values.column(status_idx).to_vec();
    let karno: Vec<f64> = ds.values.column(karno_idx).to_vec();
    // `celltype` is a string column, so schema inference encodes it as a single
    // Categorical code column (level codes by first appearance:
    // squamous=0, smallcell=1, adeno=2, large=3). That categorical kind is what
    // makes `s(karno, by=celltype)` expand into the per-level by-FACTOR smooth
    // this test validates (one karnofsky curve per cell type + a treatment-coded
    // factor main effect), exactly as in the synthetic arm.
    let celltype: Vec<f64> = ds.values.column(celltype_idx).to_vec();
    let n = time.len();
    assert!(n > 120, "veteran_lung should have ~137 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out ----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 90 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically
    // (the categorical level table lives in the schema, not the row values).
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: Weibull AFT + by-factor smooth on karno --------
    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, status) ~ karno + s(karno, by=celltype) + survmodel(spec=\"transformation\", distribution=\"weibull\")",
        &train_ds,
        &cfg,
    )
    .expect("gam Weibull-AFT by-factor fit on veteran_lung train rows");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for survival_likelihood=weibull");
    };

    // ---- score the held-out patients by gam's OWN covariate risk ----------
    // The covariate block adds `cov_eta = design(karno, celltype)·cov_beta` to
    // the shared log-cumulative-hazard, so `cov_eta` IS the per-patient log-risk
    // (monotone increasing in hazard). Rebuild the covariate design at the
    // held-out rows from the frozen spec — no baseline evaluation is needed for
    // a ranking metric, since every patient shares the same baseline H0(t).
    let cov_start = fit.time_base_ncols;
    let beta = &fit.fit.beta;
    assert!(
        beta.len() > cov_start,
        "expected covariate coefficients after the {cov_start} time columns, got beta.len()={}",
        beta.len()
    );
    let cov_beta = beta.slice(s![cov_start..]).to_owned();

    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (out_row, &src_row) in test_rows.iter().enumerate() {
        for c in 0..p {
            test_grid[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at held-out rows");
    let test_dense = test_design.design.to_dense();
    assert_eq!(
        test_dense.ncols(),
        cov_beta.len(),
        "held-out covariate design width must match the covariate coefficient slice"
    );
    let gam_risk: Vec<f64> = (0..test_rows.len())
        .map(|r| test_dense.row(r).dot(&cov_beta))
        .collect();

    let test_time: Vec<f64> = test_rows.iter().map(|&i| time[i]).collect();
    let test_status: Vec<f64> = test_rows.iter().map(|&i| status[i]).collect();
    let gam_c = concordance_index(&gam_risk, &test_time, &test_status);

    // ---- fit the SAME model on TRAIN with lifelines, score the SAME TEST ---
    // One run_python call, all columns the SAME length (full n): a per-row
    // `is_train` mask separates the fit rows from the held-out rows so we never
    // mix train-length and test-length columns. lifelines fits on the masked
    // train rows and scores the held-out rows by predicted EXPECTED survival
    // time, then forms its own held-out C-index (lifelines orients C so that
    // higher predicted survival ⇒ lower risk). Held to the identical metric.
    let is_train: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let py = run_python(
        &[
            Column::new("time", &time),
            Column::new("status", &status),
            Column::new("karno", &karno),
            Column::new("celltype", &celltype),
            Column::new("is_train", &is_train),
        ],
        r#"
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index

frame = pd.DataFrame({
    "time": np.asarray(df["time"], dtype=float),
    "status": np.asarray(df["status"], dtype=float),
    "karno": np.asarray(df["karno"], dtype=float),
    "celltype": np.asarray(df["celltype"], dtype=float).astype(int),
    "is_train": np.asarray(df["is_train"], dtype=float),
})

# Treatment-coded indicators for the 4-level cell type plus karno and its
# per-celltype interaction: the parametric-AFT analogue of karno + s(karno,
# by=celltype). Drop level 0 (squamous) as the reference to keep the design
# full rank, matching gam's treatment-coded by-factor expansion.
for lev in (1, 2, 3):
    frame[f"ct{lev}"] = (frame["celltype"] == lev).astype(float)
    frame[f"karno_ct{lev}"] = frame["karno"] * frame[f"ct{lev}"]

feat = ["karno", "ct1", "ct2", "ct3", "karno_ct1", "karno_ct2", "karno_ct3"]
train = frame[frame["is_train"] == 1.0].reset_index(drop=True)
test = frame[frame["is_train"] == 0.0].reset_index(drop=True)

aft = WeibullAFTFitter(penalizer=0.01)
aft.fit(train[feat + ["time", "status"]], duration_col="time", event_col="status")

# Higher predicted expected survival time => lower risk; concordance_index
# expects predicted scores that increase with survival, so pass expectations.
pred_exp = aft.predict_expectation(test[feat]).to_numpy().reshape(-1)
c = concordance_index(test["time"].to_numpy(), pred_exp, test["status"].to_numpy())
emit("cindex", [float(c)])
"#,
    );
    let life_c = py.scalar("cindex");

    let cens_frac = 1.0 - status.iter().sum::<f64>() / n as f64;
    eprintln!(
        "veteran_lung weibull-AFT by-factor held-out: n={n} n_train={} n_test={} \
         censoring={cens_frac:.2}\n  \
         held-out C-index: gam={gam_c:.4} lifelines={life_c:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam ranks held-out risk well --------
    // C-index >= 0.62 is well above the 0.5 random-ranking baseline for a small
    // held-out set; a broken by-factor fit (wrong karnofsky sign, collapsed
    // cell-type strata) would not clear it.
    assert!(
        gam_c >= 0.62,
        "gam's held-out concordance too low: {gam_c:.4} (< 0.62)"
    );

    // ---- BASELINE (match-or-beat): no worse than lifelines on held-out C ---
    assert!(
        gam_c >= life_c - 0.03,
        "gam less concordant than lifelines on held-out data: gam C={gam_c:.4}, lifelines C={life_c:.4} (margin 0.03)"
    );
}
