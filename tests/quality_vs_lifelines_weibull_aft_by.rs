//! End-to-end quality: gam's parametric **Weibull AFT** survival fit with a
//! **by-factor smooth** covariate effect must reproduce — stratum by stratum —
//! what `lifelines.WeibullAFTFitter` (the mature, standard parametric-AFT
//! reference in Python) recovers when fit separately on each stratum.
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
//! This is exactly the gam factorization that the test must validate.
//!
//! lifelines has no native by-factor mechanism, so we use the standard
//! reference workaround: fit an independent `WeibullAFTFitter` on each
//! stratum's rows and read back that stratum's recovered Weibull `rho_`
//! (shape), baseline `lambda_` (scale at the covariate reference), and the
//! per-stratum predicted survival surface. Comparing gam's single by-factored
//! fit against the two independent lifelines stratum fits is the canonical way
//! to check that gam's by-factor factorization is correct. For the single
//! shared baseline *shape* (which is not a per-stratum quantity) the matched
//! reference is instead a POOLED `WeibullAFTFitter` (x + group on all rows),
//! which estimates one `rho_` under the same shared-shape assumption gam makes.
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
//!
//! ## Bounds (each justified at its assertion)
//!
//!   * Per-stratum *scale* within 0.10 absolute: the effective Weibull scale at
//!     each group's covariate reference is set by the shared baseline shifted by
//!     that group's AFT term, which the by-factor smooth fits per stratum.
//!   * Shared *shape* within 0.08 of lifelines' POOLED shape: gam shares one
//!     baseline shape, so the honest matched target is the single pooled-fit
//!     `rho_`, and 0.08 is on the order of the shape MLE's asymptotic SE at n=200.
//!   * `relative_l2` on `S(t | x, g)` over a (group × x × t) grid ≤ 0.025 — the
//!     load-bearing quantity; slightly looser than the single-smooth case
//!     because the by-factor block carries an extra identifiability null space.
//!
//! Both engines target the same Weibull-AFT likelihood, so close agreement is
//! the correct expectation; a genuine divergence (e.g. a broken by-factor
//! factorization, or a shared-shape model that cannot track two strata) makes
//! the test fail honestly, which is the intended measurement.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::survival_construction::evaluate_survival_baseline;
use gam::test_support::reference::{Column, relative_l2, run_python};
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

#[test]
fn gam_weibull_aft_by_factor_matches_lifelines_per_stratum() {
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
    // time-basis coefficients.
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

    // gam predicted survival surface, laid out group-major / x-major / t-minor.
    let mut gam_surv: Vec<f64> = Vec::with_capacity(n_pred_rows * t_grid.len());
    for r in 0..n_pred_rows {
        let cov_eta: f64 = dense.row(r).dot(&cov_beta);
        for &t in &t_grid {
            let (log_h0, _) = evaluate_survival_baseline(t, &fit.baseline_cfg)
                .expect("evaluate gam baseline log-cumulative-hazard");
            let s = (-(log_h0 + cov_eta).exp()).exp();
            gam_surv.push(s);
        }
    }

    // gam's effective per-stratum Weibull scale at the covariate reference x=0.
    // log H(t | x=0, g) = shape·(log t − log scale) + c_g, with
    // c_g = covariate_eta at (x=0, group g). Setting this equal to
    // shape·(log t − log scale_eff_g) gives log scale_eff_g = log scale − c_g/shape.
    let cov_eta_ref = |group_code: f64| -> f64 {
        let mut one = Array2::<f64>::zeros((1, ds.headers.len()));
        one[[0, x_idx]] = 0.0;
        one[[0, g_idx]] = group_code;
        let d = build_term_collection_design(one.view(), &fit.resolvedspec)
            .expect("rebuild covariate design at reference row");
        d.design.to_dense().row(0).dot(&cov_beta)
    };
    let gam_scale_eff_a = gam_scale * (-cov_eta_ref(0.0) / gam_shape).exp();
    let gam_scale_eff_b = gam_scale * (-cov_eta_ref(1.0) / gam_shape).exp();

    // ---- fit the SAME data per stratum with lifelines (mature reference) ---
    // For each group fit an independent WeibullAFTFitter on `time ~ x` and emit
    // that stratum's shape (rho_), baseline scale (exp of the lambda_ intercept,
    // i.e. the Weibull scale at x=0), and the predicted survival surface on the
    // identical (x_eval × t_grid) grid.
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
    # rho_ is the Weibull shape; the lambda_ Intercept is log(scale at x=0).
    rho = float(aft.params_.loc[("rho_", "Intercept")])
    log_lambda0 = float(aft.params_.loc[("lambda_", "Intercept")])
    shape = np.exp(rho)
    scale0 = np.exp(log_lambda0)
    tag = "a" if gc == 0.0 else "b"
    emit("shape_" + tag, [shape])
    emit("scale_" + tag, [scale0])
    # Predicted survival at each x_eval over the shared time grid.
    pred_df = pd.DataFrame({"x": x_eval})
    sf = aft.predict_survival_function(pred_df, times=t_grid)
    # sf: rows=times (t_grid order), cols=rows of pred_df (x_eval order).
    for j in range(len(x_eval)):
        col = sf.iloc[:, j].to_numpy()
        surv_rows.extend(float(v) for v in col)

# Pooled WeibullAFT on ALL rows with x + group covariates. This shares ONE
# Weibull shape (rho_) across strata under the same shared-shape assumption gam
# makes, so its rho_ is the honest matched target for gam's single baseline
# shape (the arithmetic mean of two independent stratum shapes is only a proxy).
pooled = WeibullAFTFitter()
pooled.fit(frame, duration_col="time", event_col="event")
emit("shape_pooled", [float(np.exp(pooled.params_.loc[("rho_", "Intercept")]))])

emit("surv", surv_rows)
"#,
    );

    let life_shape_a = py.scalar("shape_a");
    let life_shape_b = py.scalar("shape_b");
    let life_shape_pooled = py.scalar("shape_pooled");
    let life_scale_a = py.scalar("scale_a");
    let life_scale_b = py.scalar("scale_b");
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

    let rel_surv = relative_l2(&gam_surv, life_surv);

    eprintln!(
        "weibull-AFT by-factor: n={n} censoring A={cens_a_frac:.2} B={cens_b_frac:.2}\n  \
         gam baseline: scale={gam_scale:.4} shape={gam_shape:.4}\n  \
         gam eff scale A={gam_scale_eff_a:.4} B={gam_scale_eff_b:.4}\n  \
         lifelines shape A={life_shape_a:.4} B={life_shape_b:.4} pooled={life_shape_pooled:.4}\n  \
         lifelines scale A={life_scale_a:.4} B={life_scale_b:.4}\n  \
         S(t|x,g) rel_l2={rel_surv:.4}"
    );

    // (1) Per-stratum scale. gam's by-factor smooth shifts the shared baseline
    // hazard per group; the effective Weibull scale at each group's covariate
    // reference must land near the independently-fit lifelines stratum scale.
    // 0.10 absolute is the spec bound; it tolerates the finite-sample AFT
    // estimation error of a ~100-row, partially-censored stratum.
    let scale_err_a = (gam_scale_eff_a - life_scale_a).abs();
    let scale_err_b = (gam_scale_eff_b - life_scale_b).abs();
    assert!(
        scale_err_a < 0.10,
        "group A effective scale diverges from lifelines: gam={gam_scale_eff_a:.4} lifelines={life_scale_a:.4} (|Δ|={scale_err_a:.4})"
    );
    assert!(
        scale_err_b < 0.10,
        "group B effective scale diverges from lifelines: gam={gam_scale_eff_b:.4} lifelines={life_scale_b:.4} (|Δ|={scale_err_b:.4})"
    );

    // (2) Shared shape. gam uses ONE Weibull baseline shape across strata. The
    // honest matched target is lifelines' POOLED WeibullAFT fit (x + group on all
    // rows), which estimates a single rho_ under the same shared-shape assumption
    // — not the arithmetic mean of two independent stratum shapes, which is only a
    // proxy and need not equal the likelihood-pooled MLE. 0.08 absolute: with
    // n=200 the Weibull shape MLE has an asymptotic SE on the order of
    // shape/sqrt(events) ~ 1.0/sqrt(~150) ~ 0.08, so two correctly-specified
    // pooled fits should agree to well within one such SE.
    let shape_err = (gam_shape - life_shape_pooled).abs();
    assert!(
        shape_err < 0.08,
        "gam shared shape diverges from lifelines pooled shape: gam={gam_shape:.4} lifelines_pooled={life_shape_pooled:.4} (|Δ|={shape_err:.4})"
    );

    // (3) Survival surface — the load-bearing quantity. Both engines target the
    // Weibull-AFT survival function; gam's by-factored S(t|x,g) must track the
    // per-stratum lifelines surface across (group × x × t). 0.025 relative L2 is
    // the spec bound: slightly looser than the single-smooth case because the
    // by-factor block adds an extra identifiability null space, yet still tight
    // enough that a broken factorization (collapsed strata, wrong acceleration)
    // would blow past it.
    assert!(
        rel_surv < 0.025,
        "by-factor Weibull-AFT survival surface diverges from lifelines: rel_l2={rel_surv:.4}"
    );
}
