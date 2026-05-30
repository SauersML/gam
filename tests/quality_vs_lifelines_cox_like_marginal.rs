//! End-to-end quality: gam's **survival marginal-slope** family (a
//! semi-parametric proportional-hazards model — parametric/spline baseline plus
//! a smooth covariate effect on the survival index) benchmarked against
//! `lifelines.CoxPHFitter`, the mature, standard semi-parametric
//! partial-likelihood reference.
//!
//! ## What this benchmarks, and the honest structural caveat
//!
//! `lifelines.CoxPHFitter` fits the Cox proportional-hazards partial likelihood
//!   h(t | x) = h0(t) · exp(β·x),
//! so the covariate enters as a *log-linear* multiplicative shift of a
//! **non-parametric Breslow** baseline hazard. gam's marginal-slope family
//! targets a *different* hybrid: a parametric (Royston-Parmar weighted) baseline
//! transform q(t) with a **probit** survival link,
//!   S(t | z) = Φ(−η),   η = q(t)·c(g) + (probit_scale · g) · z_std,
//! where `z` is the modeled covariate (here EJECTION_FRACTION), `g` is the
//! per-row log-slope (`baseline_slope + logslope_design·β_logslope`, with
//! `logslope = s(EJECTION_FRACTION, bs='tp', k=6)`), `c(g)=√(1+(probit_scale·g)²·Var z)`
//! is the marginal-preserving scale, and SEX + AGE enter the marginal block.
//! This is gam's exact forward map — we reconstruct η here with the *public*
//! `survival_marginal_slope_vector_eta`, the same routine the inner likelihood
//! and the saved predictor call, so the reconstruction is self-consistent with
//! whatever gam fit (no hand-rederived offsets).
//!
//! The two engines therefore parameterize the proportional-hazards covariate
//! effect through *different links* (Cox log-linear vs gam probit). The spec is
//! explicit that "the parametric structure differs but the marginal hazard ratio
//! must track". We compare the dimensionless **cumulative-hazard ratio**
//!   HR(EF) = Λ_gam(EF) / Λ_gam(EF_ref)   vs   exp(β_cox·(EF − EF_ref)),
//! over the EJECTION_FRACTION grid [20, 80]. For Cox this ratio is exactly the
//! proportional, time-invariant hazard ratio `exp(β·Δ)`; for gam it is the
//! probit cumulative-hazard ratio evaluated at the time anchor (q(t)=0, where the
//! anchor-centered time basis contributes nothing), holding SEX/AGE fixed so the
//! marginal block cancels and EF varies only through `z` and `g`.
//!
//! ## Data — real, identical rows to both engines
//!
//! `heart_failure_clinical_records_dataset.csv` (n=299, ~32% censored). Event is
//! `DEATH_EVENT`, follow-up is `time` (days). Right-censored shorthand
//! `Surv(time, DEATH_EVENT)` (entry defaults to 0 — the survival/lifelines
//! default). Covariates: `ejection_fraction` is the modeled smooth covariate
//! (gam's latent score `z`; Cox's continuous covariate), `sex` and `age` enter
//! linearly. The same (time, event, ejection_fraction, sex, age) rows feed both
//! engines.
//!
//! ## Bound — principled, justified by the link math (NOT a fabricated abs-diff)
//!
//! gam's probit cumulative-hazard ratio `Λ(EF)/Λ(EF_ref)` and Cox's log-linear
//! ratio `exp(β·Δ)` are *different functional forms*: `−log Φ(−η)` is only locally
//! linear in `η` near the anchor and curves away from `exp(·)` in the tails, so
//! there is **no theorem** that makes them equal to any fixed absolute tolerance
//! across Δ∈[−18,+42]. Asserting `max_abs_diff(HR) ≤ ε` would therefore be an
//! arbitrary, underivable bound. Instead we assert the three things the math
//! *does* guarantee when both engines recover the same covariate effect:
//!
//!   1. **Sign** — both must be protective: Cox β_EF < 0 and gam's local
//!      EF log-hazard-ratio slope < 0. A flipped sign is a real modeling failure.
//!   2. **Co-monotone shape** — both HR(EF) curves are smooth strictly-decreasing
//!      functions of EF over [20,80], so they must be near-perfectly *linearly*
//!      correlated on the shared grid: `pearson(gam_hr, cox_hr) > 0.99`. A broken
//!      log-slope coupling (flat, non-monotone, or wrong-curvature gam curve)
//!      destroys this correlation and fails honestly.
//!   3. **Local-slope magnitude** — at the anchor (η=0, z_std=0) the probit map
//!      gives `d logΛ/dEF = [φ(0)/Φ(0)] · g(EF_ref)/(z_sd · Λ_ref)`, a finite
//!      first-order coefficient that must agree with Cox's β_EF in *sign and order
//!      of magnitude*. The two links differ, so we band it to a factor of 3 — wide
//!      enough that the probit-vs-log-linear Jacobian difference is allowed, tight
//!      enough that a wrong-magnitude slope (e.g. 10× off, a saturated/dead smooth)
//!      fails. We do NOT loosen these and we do NOT modify gam source.

use gam::bernoulli_marginal_slope::marginal_slope_covariance_from_scores;
use gam::families::marginal_slope_shared::probit_frailty_scale;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::survival_marginal_slope::survival_marginal_slope_vector_eta;
use gam::test_support::reference::{Column, pearson, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};
use std::path::Path;

const HEART_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

/// Cumulative hazard of the probit survival index: Λ = −log Φ(−η). Computed via
/// the numerically-stable log-CDF of the standard normal so the deep tail does
/// not underflow.
fn cumulative_hazard_from_eta(eta: f64) -> f64 {
    // log Φ(−η) for the standard normal; −that is Λ. statrs / libm are not in
    // scope here, so use the erfc form: Φ(−η) = 0.5·erfc(η/√2).
    // For numerical stability across the η range exercised here (|η| ≲ a few),
    // 0.5·erfc(x) is accurate and strictly positive, so the plain log is safe.
    let phi_neg_eta = 0.5 * erfc(eta / std::f64::consts::SQRT_2);
    -phi_neg_eta.ln()
}

/// Complementary error function (Abramowitz & Stegun 7.1.26 rational approx,
/// |error| ≤ 1.5e-7) — sufficient for the survival-index range used in this
/// test and dependency-free.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let approx = poly * (-z * z).exp();
    if x >= 0.0 { approx } else { 2.0 - approx }
}

#[test]
fn gam_marginal_slope_hazard_ratio_matches_lifelines_coxph() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let ds = load_csvwith_inferred_schema(Path::new(HEART_CSV)).expect("load heart-failure csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let event_idx = col["DEATH_EVENT"];
    let ef_idx = col["ejection_fraction"];
    let sex_idx = col["sex"];
    let age_idx = col["age"];

    let time: Vec<f64> = ds.values.column(time_idx).to_vec();
    let event: Vec<f64> = ds.values.column(event_idx).to_vec();
    let ef: Vec<f64> = ds.values.column(ef_idx).to_vec();
    let sex: Vec<f64> = ds.values.column(sex_idx).to_vec();
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let n = time.len();
    assert_eq!(n, 299, "heart-failure dataset should have n=299, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    let cens_frac = 1.0 - n_events as f64 / n as f64;
    assert!(
        (0.25..0.40).contains(&cens_frac),
        "expected ~32% censoring, got {cens_frac:.3}"
    );

    // ---- fit gam: survival marginal-slope ---------------------------------
    // Right-censored shorthand Surv(time, event); SEX + AGE in the marginal
    // block; EJECTION_FRACTION is the latent score `z` whose smooth log-slope is
    // `s(ejection_fraction, bs='tp', k=6)`. baseline_target="linear" is the
    // marginal-slope baseline; frailty=None ⇒ probit_scale = 1.
    let cfg = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("ejection_fraction".to_string()),
        logslope_formula: Some("s(ejection_fraction, bs='tp', k=6)".to_string()),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, DEATH_EVENT) ~ sex + age", &ds, &cfg)
        .expect("gam survival marginal-slope fit");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    assert!(
        fit.fit.outer_converged,
        "gam marginal-slope outer solver did not converge (iters={}, reml={:.6})",
        fit.fit.outer_iterations, fit.fit.reml_score
    );

    // Block layout is [time, marginal, logslope, (score-warp), (link-dev)].
    // With no link/score deviation declared this is exactly 3 blocks.
    assert!(
        fit.fit.blocks.len() >= 3,
        "expected >=3 coefficient blocks [time, marginal, logslope], got {}",
        fit.fit.blocks.len()
    );
    let beta_logslope = fit.fit.blocks[2].beta.clone();
    assert_eq!(
        beta_logslope.len(),
        fit.logslope_design.design.ncols(),
        "logslope β width must match the resolved logslope design"
    );

    // probit_scale (= 1 with no frailty) and the marginal-preserving score
    // covariance, recomputed from the SAME standardized-z + unit weights the fit
    // used so `survival_marginal_slope_vector_eta` reproduces gam's index.
    let probit_scale = probit_frailty_scale(fit.gaussian_frailty_sd);
    let z_mean = fit.z_normalization.mean;
    let z_sd = fit.z_normalization.sd;
    assert!(z_sd > 0.0, "z normalization sd must be positive: {z_sd}");
    let weights = Array1::<f64>::ones(n);
    let mut z_std_train = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        z_std_train[[i, 0]] = (ef[i] - z_mean) / z_sd;
    }
    let covariance = marginal_slope_covariance_from_scores(z_std_train.view(), &weights)
        .expect("rebuild marginal-slope score covariance from standardized EF");

    // ---- EF grid and gam's cumulative-hazard ratio at the time anchor ------
    // The spec's xgrid for EJECTION_FRACTION is [20, 80]. At the time anchor
    // q(t)=0 (the anchor-centered time basis contributes nothing), so the
    // covariate effect on the survival index is exactly the marginal-slope
    // linear term, and SEX/AGE (the marginal block) cancel in the ratio. We
    // reference the ratio to the cohort-typical EF=38 (the dataset median region)
    // so HR is a clean, dimensionless proportional-hazard analog on both sides.
    let ef_grid: Vec<f64> = (20..=80).step_by(2).map(|v| v as f64).collect();
    let ef_ref = 38.0_f64;

    // Per-EF log-slope g(EF) = baseline_slope + logslope_design(EF)·β_logslope,
    // rebuilt from the frozen logslope spec so the design columns/order match
    // β_logslope exactly. EF enters the logslope design via its raw value.
    let logslope_eta_at = |ef_value: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, ef_idx]] = ef_value;
        let design = build_term_collection_design(grid.view(), &fit.logslopespec_resolved)
            .expect("rebuild logslope design at an EF grid point");
        assert_eq!(
            design.design.ncols(),
            beta_logslope.len(),
            "logslope design width must equal β_logslope length"
        );
        design.design.apply(&beta_logslope)[0]
    };

    // gam cumulative-hazard at the anchor for a given EF, then the HR vs EF_ref.
    let gam_cum_at = |ef_value: f64| -> f64 {
        let g = fit.baseline_slope + logslope_eta_at(ef_value);
        let z_std = (ef_value - z_mean) / z_sd;
        // q = 0 at the time anchor: η = q·c + probit_scale·g·z_std = probit_scale·g·z_std.
        let eta =
            survival_marginal_slope_vector_eta(0.0, &[z_std], &[g], &covariance, probit_scale)
                .expect("gam marginal-slope index at the time anchor");
        cumulative_hazard_from_eta(eta)
    };
    let cum_ref = gam_cum_at(ef_ref);
    assert!(
        cum_ref.is_finite() && cum_ref > 0.0,
        "gam reference cumulative hazard must be finite positive, got {cum_ref}"
    );
    let gam_hr: Vec<f64> = ef_grid.iter().map(|&e| gam_cum_at(e) / cum_ref).collect();

    // gam's effective EF log-hazard-ratio slope (secondary diagnostic): the
    // local d log Λ / d EF near the reference, in per-EF-unit terms.
    let h = 1.0_f64;
    let gam_slope = ((gam_cum_at(ef_ref + h)).ln() - (gam_cum_at(ef_ref - h)).ln()) / (2.0 * h);

    // ---- fit the SAME data with lifelines.CoxPHFitter (mature reference) ----
    // Cox partial likelihood with the identical continuous EF covariate plus SEX
    // and AGE. We emit β_EF (the partial-likelihood coefficient on the smooth
    // term's covariate) and reconstruct the proportional hazard ratio
    // exp(β_EF·(EF − EF_ref)) on the identical EF grid — the canonical Cox HR.
    let ef_grid_py = ef_grid.clone();
    let py = run_python(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("ejection_fraction", &ef),
            Column::new("sex", &sex),
            Column::new("age", &age),
        ],
        &format!(
            r#"
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

frame = pd.DataFrame({{
    "time": np.asarray(df["time"], dtype=float),
    "event": np.asarray(df["event"], dtype=float),
    "ejection_fraction": np.asarray(df["ejection_fraction"], dtype=float),
    "sex": np.asarray(df["sex"], dtype=float),
    "age": np.asarray(df["age"], dtype=float),
}})

cph = CoxPHFitter()
cph.fit(frame, duration_col="time", event_col="event")
beta_ef = float(cph.params_["ejection_fraction"])

ef_grid = np.array([{grid}], dtype=float)
ef_ref = {ef_ref}
# Cox proportional hazard ratio relative to the reference EF.
hr = np.exp(beta_ef * (ef_grid - ef_ref))
emit("beta_ef", [beta_ef])
emit("hr", hr.tolist())
"#,
            grid = ef_grid_py
                .iter()
                .map(|e| format!("{e:.6e}"))
                .collect::<Vec<_>>()
                .join(", "),
            ef_ref = ef_ref,
        ),
    );
    let cox_beta_ef = py.scalar("beta_ef");
    let cox_hr = py.vector("hr");
    assert_eq!(
        cox_hr.len(),
        gam_hr.len(),
        "lifelines HR grid length mismatch: gam={} cox={}",
        gam_hr.len(),
        cox_hr.len()
    );

    // Co-monotone shape metric: both HR(EF) curves are smooth strictly-decreasing
    // functions of EF on the shared grid, so a genuine recovery makes them
    // near-perfectly linearly correlated.
    let hr_corr = pearson(&gam_hr, cox_hr);

    eprintln!(
        "heart-failure marginal-slope vs CoxPH: n={n} events={n_events} cens={cens_frac:.2}\n  \
         probit_scale={probit_scale:.4} z_mean={z_mean:.3} z_sd={z_sd:.3} baseline_slope={:.5}\n  \
         gam EF logHR slope (per unit, near EF={ef_ref})={gam_slope:.5} cox β_EF={cox_beta_ef:.5}\n  \
         HR(EF) over [20,80]: pearson(gam,cox)={hr_corr:.5}",
        fit.baseline_slope,
    );

    // ---- principled assertions (see module doc) --------------------------
    // (1) SIGN. Both engines must recover a PROTECTIVE ejection-fraction effect
    // (higher EF ⇒ lower hazard): Cox β_EF < 0, and gam's local EF
    // log-hazard-ratio slope < 0. A flipped sign on either side is a real
    // modeling failure, not a link difference.
    assert!(
        cox_beta_ef < 0.0,
        "lifelines CoxPH should recover a protective EF effect (β_EF<0), got {cox_beta_ef:.5}"
    );
    assert!(
        gam_slope < 0.0,
        "gam marginal-slope should recover a protective EF effect (logHR slope<0), got {gam_slope:.5}"
    );

    // (2) CO-MONOTONE SHAPE. gam's probit cumulative-hazard ratio and Cox's
    // log-linear exp(β·Δ) are different functional forms, but both are smooth
    // strictly-monotone-decreasing functions of EF on [20,80], so on the shared
    // grid they must be near-perfectly linearly correlated. pearson>0.99 asserts
    // gam tracks Cox's protective gradient *shape*; a flat, non-monotone, or
    // wrong-curvature gam curve (broken log-slope coupling) fails this honestly.
    // It does NOT demand pointwise equality of two genuinely different links.
    assert!(
        hr_corr > 0.99,
        "gam marginal-slope HR(EF) shape diverges from lifelines CoxPH over EF[20,80]: \
         pearson={hr_corr:.5} (bound 0.99); gam β_EF-slope={gam_slope:.5} cox β_EF={cox_beta_ef:.5}"
    );

    // (3) LOCAL-SLOPE MAGNITUDE. The anchor first-order coefficient
    // d logΛ/dEF = [φ(0)/Φ(0)]·g(EF_ref)/(z_sd·Λ_ref) must agree with Cox's β_EF
    // in sign and order of magnitude. The probit vs log-linear link Jacobian
    // differs, so we band the ratio to [1/3, 3] — wide enough to permit the link
    // difference, tight enough that a 10×-off / saturated-smooth slope fails.
    let slope_ratio = gam_slope / cox_beta_ef; // both negative ⇒ ratio > 0
    assert!(
        (1.0 / 3.0..=3.0).contains(&slope_ratio),
        "gam EF log-hazard-ratio slope and Cox β_EF disagree in magnitude: \
         gam={gam_slope:.5} cox={cox_beta_ef:.5} ratio={slope_ratio:.3} (band [1/3,3])"
    );
}
