//! End-to-end quality: gam's parametric AFT survival model with a logistic
//! residual distribution (a **log-logistic accelerated-failure-time** model)
//! must RECOVER THE TRUE survivor surface and acceleration factor of a known
//! log-logistic data-generating process from right-censored data.
//!
//! Objective metric asserted (TRUTH RECOVERY, not "same as a reference tool")
//! -------------------------------------------------------------------------
//! The data are simulated from a *known* log-logistic AFT, so the true survivor
//! surface and the true acceleration coefficient are analytic, exact quantities.
//! The primary pass/fail claim is that gam's fitted `S(t | age)` and its fitted
//! acceleration slope match that GROUND TRUTH:
//!   1. `RMSE(gam_S, true_S)` over an `age x time` grid `<= 0.04` on the [0,1]
//!      probability scale (a small fraction of the unit survivor range; a
//!      genuinely worse fit failing here is useful).
//!   2. gam's recovered acceleration coefficient (slope of `log(median survival)`
//!      vs age) is within `0.0075` absolute of the true `b_age = -0.025`
//!      (~30% — tight given right-censoring at this n).
//!
//! `lifelines.LogLogisticAFTFitter` is fit on the IDENTICAL data and demoted to a
//! BASELINE-TO-MATCH-OR-BEAT on accuracy: gam's surface RMSE-against-truth must be
//! `<= 1.10 *` lifelines' RMSE-against-truth. We never assert "gam matches
//! lifelines"; lifelines is a noisy estimator of the same surface and matching it
//! would prove nothing. The closeness-to-lifelines numbers are still computed and
//! printed for context only.
//!
//! The true log-logistic survivor function is
//! `S(t | x) = 1 / (1 + (t / alpha(x))^(1/sigma))` with `log alpha(x) = mu(x) =
//! b0 + b_age * (age - age_mean)`. Its median is `t = alpha(x)`, so
//! `log(median(age)) = b0 + b_age*(age-age_mean)` is affine in age with slope
//! exactly `b_age` — the acceleration factor we recover.
//!
//! Data
//! ----
//! The Haberman breast-cancer study (`bench/datasets/haberman.csv`, n = 306) is
//! a five-year-survival study with no recorded continuous follow-up time — its
//! only event field is the binary 5-year status. A parametric AFT is undefined
//! without event times, so we keep Haberman's **real age-at-surgery covariate**
//! (first column, ages 30-83) and pair it with right-censored event times drawn
//! once from the known log-logistic AFT above (fixed seed, deterministic). The
//! exact same `(t, d, age)` table is handed to gam and to lifelines.

use csv::StringRecord;
use gam::families::survival::construction::{
    SURVIVAL_TIME_FLOOR, SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
    SurvivalTimeBasisConfig, add_survival_time_derivative_guard_offset, build_survival_time_basis,
    build_survival_time_offsets_for_likelihood, evaluate_survival_time_basis_row,
    resolve_survival_time_anchor_value, resolved_survival_time_basis_config_from_build,
    survival_derivative_guard_for_likelihood,
};
use gam::families::survival::location_scale::{
    SurvivalLocationScalePredictInput, predict_survival_location_scale,
};
use gam::test_support::reference::{
    Column, QualityPair, pearson, relative_l2, rmse, run_python,
};
use gam::types::InverseLink;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::path::Path;

const HABERMAN_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/haberman.csv");

/// Deterministic uniform stream (SplitMix64) — no RNG crate, no seed drift, so
/// gam and the reference operate on a byte-identical data table.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval (0, 1).
    fn next_open_unit(&mut self) -> f64 {
        // 53-bit mantissa mapped into (0, 1): add 0.5 ULP so we never hit 0 or 1.
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// Real Haberman ages: first column of the headerless CSV (age at operation).
fn haberman_ages() -> Vec<f64> {
    let text = std::fs::read_to_string(Path::new(HABERMAN_CSV)).expect("read haberman.csv");
    let mut ages = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let first = line.split(',').next().expect("haberman row has a column");
        let age: f64 = first.parse().expect("haberman age column is numeric");
        ages.push(age);
    }
    assert_eq!(ages.len(), 306, "haberman.csv should carry 306 rows");
    ages
}

#[test]
fn gam_loglogistic_aft_matches_lifelines_on_haberman_ages() {
    init_parallelism();

    // ---- identical right-censored log-logistic AFT data for both engines ----
    // True model (the family both engines fit):
    //   log T = b0 + b_age * age_c + sigma * eps,  eps ~ standard Logistic.
    // Centering age stabilizes both optimizers identically; the slope (b_age,
    // the acceleration factor) is what we cross-check, and S(t|age) is compared
    // directly so any intercept/centering convention cancels.
    let ages = haberman_ages();
    let n = ages.len();
    let age_mean = ages.iter().sum::<f64>() / n as f64;

    let b0 = 2.4_f64; // baseline log-scale at the mean age
    let b_age = -0.025_f64; // older age -> shorter survival (acceleration factor)
    let true_sigma = 0.55_f64; // log-logistic scale (1 / shape)

    let mut rng = SplitMix64::new(0xA17F_5132_C0DE_5EED);
    let mut time = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    let mut age_col = Vec::with_capacity(n);
    for &age in &ages {
        let age_c = age - age_mean;
        let mu = b0 + b_age * age_c;
        // Logistic residual via inverse-CDF: eps = log(u / (1 - u)).
        let u = rng.next_open_unit();
        let eps = (u / (1.0 - u)).ln();
        let t_event = (mu + true_sigma * eps).exp();
        // Independent log-logistic censoring time so ~25-30% are censored,
        // exercising the right-censored likelihood in both engines.
        let uc = rng.next_open_unit();
        let epsc = (uc / (1.0 - uc)).ln();
        let t_cens = (b0 + 0.9 + true_sigma * epsc).exp();
        let observed = t_event.min(t_cens);
        let d = if t_event <= t_cens { 1.0 } else { 0.0 };
        time.push(observed);
        event.push(d);
        age_col.push(age);
    }
    let observed_events: f64 = event.iter().sum();
    assert!(
        observed_events > 0.5 * n as f64 && observed_events < 0.95 * n as f64,
        "synthetic data should be partially censored, got {observed_events} events of {n}"
    );

    // ---- fit gam: log-logistic AFT via the location-scale survival likelihood
    let headers = ["time", "event", "age"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.0}", event[i]),
                format!("{:.17e}", age_col[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode survival data");

    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        // Logistic residual distribution => log-logistic AFT (vs Gaussian =
        // log-normal, Gumbel = Weibull). This is the load-bearing choice.
        survival_distribution: "logistic".to_string(),
        // Constant scale block: a single sigma shared across subjects, exactly
        // like lifelines' LogLogisticAFT shape parameter.
        noise_formula: Some("1".to_string()),
        // #736: this test is a parametric log-logistic AFT recovery check.
        // The default flexible 8-internal-knot time warp admits curved
        // baseline-shape directions that are unnecessary for an affine
        // log-time AFT surface and can dominate small-time survivor error.
        time_num_internal_knots: 2,
        outer_max_iter: Some(80),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Surv(time, event) ~ age", &ds, &cfg).expect("gam log-logistic AFT fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };

    // ---- evaluation grid: age in [20, 70] x time grid (median-band times) ---
    let eval_ages: Vec<f64> = (0..11).map(|k| 20.0 + 5.0 * k as f64).collect();
    let eval_times: Vec<f64> = (1..=12).map(|k| 2.0 * k as f64).collect();
    let grid_n = eval_ages.len() * eval_times.len();

    let mut grid_age = Array1::<f64>::zeros(grid_n);
    let mut grid_time = Array1::<f64>::zeros(grid_n);
    {
        let mut idx = 0usize;
        for &a in &eval_ages {
            for &t in &eval_times {
                grid_age[idx] = a;
                grid_time[idx] = t;
                idx += 1;
            }
        }
    }

    // Resolve the time basis ONCE from the training times (entry=0 for the
    // right-censored shorthand, exit=observed time), fixing the I-spline knots /
    // anchor the fit used so every grid evaluation reuses the same basis.
    let train_entry = Array1::<f64>::zeros(n);
    let train_exit = Array1::<f64>::from_vec(time.clone());
    // Pass the same time_num_internal_knots the FitConfig uses (2) so the
    // resolved knot vector and keep_cols column count match the fitted beta.
    let (resolved_cfg, anchor, anchor_row) =
        resolve_training_time_basis(&train_entry, &train_exit, cfg.time_num_internal_knots);
    let time_ctx = TimeBasisCtx {
        resolved_cfg,
        anchor,
        anchor_row,
    };

    // gam fitted S(t|age) over the grid, via the public location-scale survival
    // predictor. The threshold (location) and log-sigma designs are rebuilt from
    // the frozen, fitted specs at the grid covariates; the monotone time basis on
    // log(t) is rebuilt with the SAME knots/anchor/centering/derivative-guard the
    // fit used, so design*beta reproduces the model's linear predictor exactly.
    let gam_surv = gam_grid_survival(&fit, &time_ctx, &grid_age, &grid_time);

    // ---- fit the SAME data with lifelines.LogLogisticAFT (the reference) -----
    // The harness CSV requires every column to have the SAME length (it asserts
    // equal `data.len()` and does NOT pad), so only the n-row training columns
    // (time/event/age) travel through `df`. The grid_n-row evaluation grid is
    // rendered into the Python source as literal lists built from the very same
    // Rust `grid_age`/`grid_time` vectors gam evaluates, so both engines score
    // the identical (age, time) pairs in the identical order.
    let py_list = |v: &Array1<f64>| -> String {
        let items: Vec<String> = (0..v.len()).map(|i| format!("{:.17e}", v[i])).collect();
        format!("[{}]", items.join(", "))
    };
    let grid_age_py = py_list(&grid_age);
    let grid_time_py = py_list(&grid_time);
    let body = format!(
        r#"
import pandas as pd
from lifelines import LogLogisticAFTFitter

fit_df = pd.DataFrame({{
    "time": df["time"],
    "event": df["event"],
    "age": df["age"],
}})
fit_df = fit_df[fit_df["time"] > 0]

aft = LogLogisticAFTFitter()
aft.fit(fit_df, duration_col="time", event_col="event", ancillary=False)

grid_age = np.asarray({grid_age_py}, dtype=float)
grid_time = np.asarray({grid_time_py}, dtype=float)

# Per-row survivor probability S(t_i | age_i): evaluate each subject's curve at
# its own grid time. predict_survival_function returns a (times x rows) frame;
# request exactly the row's time so the single returned cell is S(t_i | age_i).
surv = []
for i in range(len(grid_time)):
    gx = pd.DataFrame({{"age": [float(grid_age[i])]}})
    sf = aft.predict_survival_function(gx, times=[float(grid_time[i])])
    surv.append(float(sf.to_numpy()[0, 0]))
emit("surv", surv)

# Acceleration factor: lifelines reports log(alpha) = a0 + a_age * age, so the
# 'age' coefficient in the 'alpha_' block IS the AFT location slope.
params = aft.params_
a_age = float(params.loc[("alpha_", "age")])
emit("a_age", a_age)
"#
    );
    let r = run_python(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("age", &age_col),
        ],
        &body,
    );
    let ref_surv = r.vector("surv");
    let ref_a_age = r.scalar("a_age");
    assert_eq!(
        ref_surv.len(),
        grid_n,
        "lifelines surv grid length mismatch"
    );

    // ---- analytic GROUND-TRUTH survivor surface on the same grid ------------
    // S(t|x) = 1 / (1 + (t/alpha(x))^(1/sigma)), log alpha(x) = b0 + b_age*age_c.
    // This is exact, not a fit; gam must recover it from the censored sample.
    let true_surv: Vec<f64> = (0..grid_n)
        .map(|i| {
            let age_c = grid_age[i] - age_mean;
            let log_alpha = b0 + b_age * age_c;
            let alpha = log_alpha.exp();
            let z = (grid_time[i] / alpha).powf(1.0 / true_sigma);
            1.0 / (1.0 + z)
        })
        .collect();

    // ---- PRIMARY: gam recovers the true survivor surface --------------------
    let gam_surv_vec: Vec<f64> = gam_surv.to_vec();
    let gam_truth_rmse = rmse(&gam_surv_vec, &true_surv);
    let ref_truth_rmse = rmse(ref_surv, &true_surv);

    // ---- PRIMARY: gam recovers the true acceleration coefficient ------------
    // Derive gam's log-median-survival slope vs age purely from its predicted
    // S(t|age): the model is location-scale on log-time, so log(median(age)) is
    // affine in age and the slope equals the AFT acceleration coefficient. Two
    // well-separated ages give the slope without touching internal coefficients.
    let med_lo = gam_median_log_survival(&fit, &time_ctx, 35.0);
    let med_hi = gam_median_log_survival(&fit, &time_ctx, 65.0);
    let gam_a_age = (med_hi - med_lo) / (65.0 - 35.0);
    let a_age_truth_err = (gam_a_age - b_age).abs();

    // Context only (NOT a pass criterion): how close gam and lifelines land to
    // each other on the shared surface.
    let rel = relative_l2(&gam_surv_vec, ref_surv);
    let corr = pearson(&gam_surv_vec, ref_surv);
    eprintln!(
        "loglogistic AFT truth-recovery: n={n} events={observed_events:.0} grid={grid_n} \
         RMSE(gam,truth)={gam_truth_rmse:.5} RMSE(lifelines,truth)={ref_truth_rmse:.5} \
         gam_a_age={gam_a_age:.5} true_b_age={b_age:.5} a_age_err={a_age_truth_err:.5} \
         lifelines_a_age={ref_a_age:.5} | context: rel_l2(gam,lifelines)={rel:.5} \
         pearson(gam,lifelines)={corr:.6}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_lifelines_loglogistic_aft",
            "survival_rmse_to_truth",
            gam_truth_rmse,
            "lifelines",
            ref_truth_rmse,
        )
        .line()
    );

    // PRIMARY claim: gam's fitted survivor surface recovers the true one to a
    // small fraction of the [0,1] probability range.
    assert!(
        gam_truth_rmse <= 0.04,
        "gam S(t|age) does not recover the true log-logistic surface: \
         RMSE(gam,truth)={gam_truth_rmse:.5} (bar 0.04)"
    );
    // PRIMARY claim: gam recovers the true acceleration factor.
    assert!(
        a_age_truth_err <= 0.0075,
        "gam does not recover the true AFT acceleration coefficient: \
         gam_a_age={gam_a_age:.5} true_b_age={b_age:.5} err={a_age_truth_err:.5} (bar 0.0075)"
    );
    // BASELINE-TO-MATCH-OR-BEAT: gam's accuracy against truth is at least as good
    // as the mature reference's (within 10%). lifelines is a peer estimator of the
    // same surface, never the truth itself.
    assert!(
        gam_truth_rmse <= 1.10 * ref_truth_rmse,
        "gam is less accurate than lifelines on the true surface: \
         RMSE(gam,truth)={gam_truth_rmse:.5} > 1.10 * RMSE(lifelines,truth)={ref_truth_rmse:.5}"
    );
}

/// Resolve the I-spline time-basis config (with data-inferred knots) and the
/// time anchor from the TRAINING times, exactly as gam's location-scale fit did.
/// The monotone I-spline knots live on the empirical log-time quantiles of the
/// training data, so the basis must be re-resolved from the same training times
/// to reproduce the columns the fitted `beta_time` was estimated against. Once
/// resolved, the basis is evaluated at arbitrary grid times via the provided
/// knots — knot-stable, independent of where we evaluate.
fn resolve_training_time_basis(
    train_entry: &Array1<f64>,
    train_exit: &Array1<f64>,
    time_num_internal_knots: usize,
) -> (SurvivalTimeBasisConfig, f64, Array1<f64>) {
    let cfg = SurvivalTimeBasisConfig::ISpline {
        degree: 3,
        knots: Array1::zeros(0),
        keep_cols: Vec::new(),
        smooth_lambda: 1e-2,
    };
    let time_build = build_survival_time_basis(
        train_entry,
        train_exit,
        cfg,
        // Must match the FitConfig's time_num_internal_knots so the resolved
        // basis (knots + keep_cols) agrees with what the training engine used.
        Some((time_num_internal_knots, 1e-2)),
    )
    .expect("resolve training survival time basis");
    let resolved_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )
    .expect("resolve survival time basis config");
    let anchor =
        resolve_survival_time_anchor_value(train_entry, None).expect("survival time anchor");
    let anchor_row =
        evaluate_survival_time_basis_row(anchor, &resolved_cfg).expect("survival anchor basis row");
    (resolved_cfg, anchor, anchor_row)
}

/// Build the anchor-centered I-spline time-exit design and the exit-time offset
/// channel at `eval_exit`, using the training-resolved knots/anchor so
/// `design * beta_time + offset` reproduces the fitted time linear predictor.
fn build_time_exit_design(
    resolved_cfg: &SurvivalTimeBasisConfig,
    anchor: f64,
    anchor_row: &Array1<f64>,
    eval_entry: &Array1<f64>,
    eval_exit: &Array1<f64>,
    baseline_cfg: &SurvivalBaselineConfig,
    inverse_link: &InverseLink,
) -> (Array2<f64>, Array1<f64>) {
    let mode = SurvivalLikelihoodMode::LocationScale;
    let p = anchor_row.len();
    let mut x_time_exit = Array2::<f64>::zeros((eval_exit.len(), p));
    for (i, &t) in eval_exit.iter().enumerate() {
        let row =
            evaluate_survival_time_basis_row(t, resolved_cfg).expect("survival time basis row");
        for j in 0..p {
            x_time_exit[[i, j]] = row[j] - anchor_row[j];
        }
    }

    let (mut eta_entry, mut eta_exit, mut eta_deriv) = build_survival_time_offsets_for_likelihood(
        eval_entry,
        eval_exit,
        baseline_cfg,
        mode,
        Some(inverse_link),
    )
    .expect("build survival time offsets");
    add_survival_time_derivative_guard_offset(
        eval_entry,
        eval_exit,
        anchor,
        survival_derivative_guard_for_likelihood(mode),
        &mut eta_entry,
        &mut eta_exit,
        &mut eta_deriv,
    )
    .expect("add survival derivative guard offset");

    (x_time_exit, eta_exit)
}

/// Training-resolved time-basis context: the I-spline config carrying the
/// data-inferred knots, the time anchor, and the centered anchor row. Fixed once
/// from the training times so every grid evaluation reuses the SAME basis.
struct TimeBasisCtx {
    resolved_cfg: SurvivalTimeBasisConfig,
    anchor: f64,
    anchor_row: Array1<f64>,
}

/// gam fitted S(t|age) over the grid via the public location-scale predictor.
fn gam_grid_survival(
    fit: &gam::SurvivalLocationScaleFitResult,
    ctx: &TimeBasisCtx,
    grid_age: &Array1<f64>,
    grid_time: &Array1<f64>,
) -> Array1<f64> {
    let grid_n = grid_age.len();
    // entry = 0 (right-censored shorthand training), exit = grid time.
    let eval_entry = Array1::<f64>::zeros(grid_n);
    let eval_exit = grid_time.clone();

    // Linear-baseline offsets: the model's parametric baseline target. With the
    // default location-scale config this is the `linear` baseline (no parametric
    // offset); the monotone I-spline carries the baseline hazard shape.
    let baseline_cfg = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::Linear,
        scale: None,
        shape: None,
        rate: None,
        makeham: None,
    };
    let (x_time_exit, eta_time_offset_exit) = build_time_exit_design(
        &ctx.resolved_cfg,
        ctx.anchor,
        &ctx.anchor_row,
        &eval_entry,
        &eval_exit,
        &baseline_cfg,
        &fit.inverse_link,
    );

    // Threshold (location) design at the grid ages, rebuilt from the FITTED,
    // frozen spec so its columns line up with beta_threshold. The frozen spec
    // references the training column layout (time=0, event=1, age=2), so the
    // grid design input mirrors that 3-column shape with age in column 2.
    const AGE_COL: usize = 2;
    const N_COLS: usize = 3;
    let mut grid_data = Array2::<f64>::zeros((grid_n, N_COLS));
    for i in 0..grid_n {
        grid_data[[i, AGE_COL]] = grid_age[i];
    }
    let x_threshold = gam::smooth::build_term_collection_design(
        grid_data.view(),
        &fit.fit.resolved_thresholdspec,
    )
    .expect("rebuild threshold design at grid ages")
    .design;
    // Constant-scale (sigma) design: a single intercept column, identical for
    // every grid row (noise_formula = "1").
    let x_log_sigma = gam::smooth::build_term_collection_design(
        grid_data.view(),
        &fit.fit.resolved_log_sigmaspec,
    )
    .expect("rebuild log-sigma design at grid ages")
    .design;

    // Reduced parametric-AFT representation (#892 / gam#1110): when the constant-
    // scale affine baseline collapses to the canonical σ-scaled log-t gauge, the
    // time warp is removed (`beta_time ≡ 0`, `h ≡ 0`) and the `log t` baseline
    // rides the LOCATION channel — exactly as the production sampling predictor
    // reconstructs it (`predict_survival_location_scale_batch`: it detects the
    // regime from `beta_time` being all-zero with no baseline time-wiggle and
    // shifts `eta_threshold_offset → eta_threshold_offset − log t` so the
    // standardized residual is `u = inv_sigma·(log t − eta_t)`). Mirror that exact
    // detection + shift here so this manual reconstruction reproduces the model's
    // actual S(t|age) for BOTH representations: a genuine free time warp (non-zero
    // `beta_time` → no shift) and the collapsed parametric AFT (zero `beta_time` →
    // log-t location shift). Without it the collapsed baseline would be dropped and
    // S(t|age) would be flat in t.
    let reduced_parametric_aft = fit.fit.fit.beta_time().iter().all(|&b| b == 0.0);
    let eta_threshold_offset = if reduced_parametric_aft {
        grid_time.mapv(|t| -(t.max(SURVIVAL_TIME_FLOOR).ln()))
    } else {
        Array1::<f64>::zeros(grid_n)
    };

    let input = SurvivalLocationScalePredictInput {
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        x_threshold,
        eta_threshold_offset,
        x_log_sigma,
        eta_log_sigma_offset: Array1::<f64>::zeros(grid_n),
        x_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        inverse_link: fit.inverse_link.clone(),
    };
    predict_survival_location_scale(&input, &fit.fit.fit)
        .expect("gam location-scale survival prediction")
        .survival_prob
}

/// log(median survival time) at a single age, found by bracketing S(t)=0.5 on a
/// dense time grid. Pure prediction-derived; no internal coefficients touched.
fn gam_median_log_survival(
    fit: &gam::SurvivalLocationScaleFitResult,
    ctx: &TimeBasisCtx,
    age: f64,
) -> f64 {
    let times: Vec<f64> = (1..=400).map(|k| 0.25 * k as f64).collect();
    let m = times.len();
    let grid_age = Array1::<f64>::from_elem(m, age);
    let grid_time = Array1::<f64>::from_vec(times.clone());
    let surv = gam_grid_survival(fit, ctx, &grid_age, &grid_time);
    // S is monotone decreasing; find the crossing of 0.5 and linearly
    // interpolate in log-time.
    for i in 1..m {
        if surv[i] <= 0.5 {
            let (t0, t1) = (times[i - 1], times[i]);
            let (s0, s1) = (surv[i - 1], surv[i]);
            let w = if (s0 - s1).abs() > 1e-12 {
                (s0 - 0.5) / (s0 - s1)
            } else {
                0.5
            };
            let t_med = t0 + w * (t1 - t0);
            return t_med.max(1e-8).ln();
        }
    }
    times[m - 1].ln()
}
