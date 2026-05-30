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
//! Group A baseline ~ Weibull(scale=0.8, shape=1.2); Group B baseline ~
//! Weibull(scale=1.5, shape=0.9); both with x ~ N(0,1) and a group-specific
//! AFT acceleration of x (the slope of x on log-time differs by group, which
//! the by-factor smooth must capture). Right-censoring via an independent
//! exponential keeps ~20-35% censored — realistic, and identical rows go to
//! both engines.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::survival_construction::evaluate_survival_baseline;
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Normal, Weibull};

const N_PER_GROUP: usize = 100;
const SEED: u64 = 20260529;

// Group baselines (rand_distr::Weibull::new(scale, shape): CDF 1 - exp(-(t/scale)^shape)).
const SCALE_A: f64 = 0.8;
const SHAPE_A: f64 = 1.2;
const SCALE_B: f64 = 1.5;
const SHAPE_B: f64 = 0.9;
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
