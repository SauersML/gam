//! End-to-end quality: gam's Royston-Parmar flexible parametric survival
//! baseline (the `survival_likelihood="transformation"` family with a monotone
//! I-spline log-cumulative-hazard time basis) must agree with
//! `flexsurv::flexsurvspline` — the canonical, mature reference for
//! Royston-Parmar flexible parametric survival — on real, censored data with a
//! continuous covariate.
//!
//! Why flexsurv (not mgcv): `flexsurvspline(..., scale = "hazard")` is the
//! direct, textbook Royston-Parmar model. It writes the log cumulative hazard
//! as a restricted-cubic-spline in `log(t)` plus a proportional linear
//! covariate effect,
//!
//!     log Λ(t | x) = s(log t ; γ) + β·x ,
//!
//! which is exactly the model gam's transformation likelihood targets: a smooth
//! (spline) baseline log-cumulative-hazard plus a linear covariate shift on the
//! same scale. Both engines therefore parameterize the *same* quantity, so the
//! correct expectation is that their fitted `log Λ(t | Age)` surfaces coincide
//! up to spline-basis / knot-placement differences (gam uses a monotone I-spline
//! on `log t`; flexsurv uses a natural cubic spline on `log t`).
//!
//! Real data: the PBC `cirrhosis.csv` cohort (n≈418). Time is `N_Days`, the
//! event is death (`Status == "D"`); transplant (`CL`) and alive (`C`) are
//! right-censored — the standard single-endpoint coding used in every
//! lifelines / flexsurv PBC tutorial. The covariate is `Age` (converted from
//! days to years for conditioning). Identical `(time, event, Age)` rows feed
//! both engines.
//!
//! What we assert, grid-aligned on the quantity that matters:
//!   1. relative-L2 of `log Λ(t | Age)` over a 15-time × 5-age-quantile grid,
//!   2. Pearson correlation of the survival surface `S(t | Age)`,
//!   3. gam's penalized REML effective df is bracketed below by the unpenalized
//!      parametric part and above by flexsurv's nominal `m$npars` (a penalty can
//!      only shrink df below the MLE count — no basis-specific tolerance needed).
//!
//! gam's `log Λ(t | Age)` is reconstructed from first principles from the
//! converged fit — exactly as `gam::families::survival_predict::evaluate_rp_row`
//! assembles it — namely
//!
//!     η(t, Age) = [b(t) − b(anchor)]·β_time + c(Age)·β_cov ,
//!     log Λ = η,   S = exp(−exp(η)),
//!
//! where `b(·)` is the (anchor-centered) I-spline time-basis row built by
//! `evaluate_survival_time_basis_row`, `c(Age)` is the frozen covariate design,
//! `β = [β_time | β_cov]` is the joint coefficient vector, and the Linear
//! baseline target contributes a zero eta-offset for the transformation mode.

use csv::StringRecord;
use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

// Royston-Parmar flexible-parametric spline flexibility. gam's transformation
// time basis uses a monotone I-spline on log(t); flexsurv uses a natural cubic
// spline on log(t) with `k` internal knots. We match the interior-knot count so
// the two smooth baselines have comparable wiggliness.
const N_INTERNAL_KNOTS: usize = 3;

/// Parse `cirrhosis.csv` into numeric `(N_Days, event, Age_years)` rows, coding
/// death (`Status == "D"`) as the event and everything else (alive `C`,
/// transplant `CL`) as right-censored — the standard PBC single-endpoint coding.
fn load_cirrhosis() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(CIRRHOSIS_CSV)).expect("open cirrhosis.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("cirrhosis header line")
        .expect("read cirrhosis header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("cirrhosis.csv missing column {name}"))
    };
    let i_days = idx("N_Days");
    let i_status = idx("Status");
    let i_age = idx("Age");

    let (mut days, mut event, mut age_years) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read cirrhosis row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let t: f64 = f[i_days].parse().expect("parse N_Days");
        let a: f64 = f[i_age].parse().expect("parse Age");
        // Death is the modeled event; alive / transplant are right-censored.
        let e = if f[i_status] == "D" { 1.0 } else { 0.0 };
        days.push(t);
        event.push(e);
        age_years.push(a / 365.25);
    }
    (days, event, age_years)
}

#[test]
fn gam_rp_spline_baseline_matches_flexsurvspline_on_cirrhosis() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let (days, event, age_years) = load_cirrhosis();
    let n = days.len();
    assert!(n > 300, "cirrhosis should have ~418 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(n_events > 100, "expected >100 deaths, got {n_events}");

    // Encode the numeric survival frame for gam: N_Days (time), event (0/1),
    // Age_years (continuous covariate). Identical values go to flexsurv below.
    let headers = ["N_Days", "event", "Age_years"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", days[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", age_years[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cirrhosis frame");

    // ---- fit gam: Royston-Parmar flexible parametric baseline -------------
    // `survival_likelihood="transformation"` is gam's Royston-Parmar family:
    // it models log Λ(t|x) directly. The monotone I-spline time basis on log(t)
    // is the smooth flexible-parametric baseline; `Age_years` enters as a
    // proportional linear covariate on the log-cumulative-hazard scale — exactly
    // the flexsurvspline(scale="hazard") structure.
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Surv(N_Days, event) ~ Age_years", &ds, &cfg).expect("gam RP-spline fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };
    let gam_edf = fit
        .fit
        .edf_total()
        .expect("gam reports total edf for the RP fit");

    // Reconstruct the fitted log-cumulative-hazard exactly as
    // `survival_predict::evaluate_rp_row` does: η = [b(t) − b(anchor)]·β_time +
    // c(Age)·β_cov, with a zero eta-offset for the Linear baseline target under
    // the transformation likelihood. β = [β_time | β_cov].
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_time = beta.slice(ndarray::s![..p_time]).to_owned();
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // Resolved (knot-frozen) time-basis config + anchor row, mirroring the
    // engine's anchor-centered I-spline rows on log(t).
    let time_cfg: SurvivalTimeBasisConfig = resolved_survival_time_basis_config_from_build(
        &fit.time_basis.basisname,
        fit.time_basis.degree,
        fit.time_basis.knots.as_ref(),
        fit.time_basis.keep_cols.as_ref(),
        fit.time_basis.smooth_lambda,
    )
    .expect("resolve frozen survival time-basis config");
    let anchor_row = evaluate_survival_time_basis_row(fit.time_basis.anchor, &time_cfg)
        .expect("evaluate time-basis anchor row");
    assert_eq!(
        anchor_row.len(),
        p_time,
        "anchor row width must equal the RP time block width"
    );

    // ---- evaluation grid: 15 times × 5 age quantiles ----------------------
    // Times span the interior of the observed range on a log scale (Λ moves
    // most in log-time, the natural axis of the RP spline). Ages are sampled at
    // the 10/30/50/70/90% quantiles so the covariate effect is exercised across
    // the cohort, not only at its center.
    let mut sorted_t = days.clone();
    sorted_t.sort_by(f64::total_cmp);
    let t_lo = sorted_t[(0.05 * n as f64) as usize];
    let t_hi = sorted_t[((0.95 * n as f64) as usize).min(n - 1)];
    let n_t = 15usize;
    let times: Vec<f64> = (0..n_t)
        .map(|j| {
            let frac = j as f64 / (n_t - 1) as f64;
            (t_lo.ln() + frac * (t_hi.ln() - t_lo.ln())).exp()
        })
        .collect();

    let mut sorted_age = age_years.clone();
    sorted_age.sort_by(f64::total_cmp);
    let age_quantile = |q: f64| sorted_age[((q * n as f64) as usize).min(n - 1)];
    let ages: Vec<f64> = [0.10, 0.30, 0.50, 0.70, 0.90]
        .into_iter()
        .map(age_quantile)
        .collect();

    // Covariate design row c(Age) per age, rebuilt from the frozen spec so its
    // column order and basis match β_cov exactly (the time axis is carried by
    // the separate I-spline block, not by this covariate design).
    let age_idx = ds.column_map()["Age_years"];
    let cov_rows: Vec<Vec<f64>> = ages
        .iter()
        .map(|&age| {
            let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
            grid[[0, age_idx]] = age;
            let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
                .expect("rebuild covariate design at an age");
            assert_eq!(
                design.design.ncols(),
                beta_cov.len(),
                "covariate design width must equal β_cov length"
            );
            design.design.apply(&beta_cov).to_vec()
        })
        .collect();

    // gam log Λ(t | Age) and S(t | Age) over the (age, time) grid.
    let mut gam_log_cumhaz = Vec::with_capacity(ages.len() * times.len());
    let mut gam_survival = Vec::with_capacity(ages.len() * times.len());
    for (ai, _age) in ages.iter().enumerate() {
        let cov_contrib = cov_rows[ai][0];
        for &t in &times {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut eta = cov_contrib;
            for k in 0..p_time {
                eta += (b[k] - anchor_row[k]) * beta_time[k];
            }
            gam_log_cumhaz.push(eta);
            gam_survival.push((-eta.exp()).exp());
        }
    }

    // ---- fit the SAME model with flexsurv::flexsurvspline ------------------
    // scale="hazard" => Royston-Parmar log-cumulative-hazard spline; k interior
    // knots match gam's interior-knot count. summary(type="cumhaz") returns the
    // cumulative hazard Λ(t|Age) on the requested time grid per newdata age.
    let age_columns: Vec<f64> = ages.clone();
    let r = run_r(
        &[
            Column::new("N_Days", &days),
            Column::new("event", &event),
            Column::new("Age_years", &age_years),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            times <- c({times})
            ages  <- c({ages})
            m <- flexsurvspline(Surv(N_Days, event) ~ Age_years, data = df,
                                k = {k}, scale = "hazard")
            nd <- data.frame(Age_years = ages)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = times, ci = FALSE)
            sv <- summary(m, newdata = nd, type = "survival", t = times, ci = FALSE)
            logcum <- c()
            surv   <- c()
            for (i in seq_along(ages)) {{
              logcum <- c(logcum, log(ch[[i]]$est))
              surv   <- c(surv, sv[[i]]$est)
            }}
            emit("logcum", logcum)
            emit("surv", surv)
            # flexsurv's actual estimated-parameter count: the (k+2) spline
            # coefficients (gamma0..gamma_{{k+1}}) plus the single Age slope. This
            # is the *unpenalized* MLE df, the honest reference for "model size".
            emit("npars", m$npars)
            "#,
            times = times
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            ages = age_columns
                .iter()
                .map(|a| format!("{a:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            k = N_INTERNAL_KNOTS,
        ),
    );
    let flex_logcum = r.vector("logcum");
    let flex_surv = r.vector("surv");
    let flex_npars = r.scalar("npars");
    assert_eq!(
        flex_logcum.len(),
        gam_log_cumhaz.len(),
        "flexsurv log-cumhaz grid length mismatch"
    );
    assert_eq!(
        flex_surv.len(),
        gam_survival.len(),
        "flexsurv survival grid length mismatch"
    );

    // ---- compare on the grid that matters ---------------------------------
    let rel_logcum = relative_l2(&gam_log_cumhaz, flex_logcum);
    let corr_surv = pearson(&gam_survival, flex_surv);

    eprintln!(
        "cirrhosis RP-spline vs flexsurvspline: n={n} events={n_events} \
         gam_edf={gam_edf:.3} grid={}x{} rel_l2(logLambda)={rel_logcum:.4} \
         pearson(S)={corr_surv:.5}",
        ages.len(),
        times.len()
    );

    // Both engines fit the SAME Royston-Parmar log-cumulative-hazard model on
    // identical data; the only legitimate source of disagreement is the spline
    // family (gam: penalized monotone I-spline on log t; flexsurv: unpenalized
    // natural cubic spline on log t) and interior-knot placement. The calibrated
    // bound for this exact pairing — same gam transformation family, same
    // flexsurvspline(scale="hazard") comparator — is a 7% relative-L2 on log Λ
    // over the interior time/age grid (see the sibling ICU test
    // `quality_vs_flexsurv_piecewise_constant_vs_rp_baseline`). Tight enough that
    // any real divergence in the baseline shape or covariate slope fails, loose
    // enough for the genuine basis/knot/penalty difference the two engines have.
    assert!(
        rel_logcum <= 0.07,
        "gam's RP-spline log cumulative hazard diverges from flexsurvspline: rel_l2={rel_logcum:.4}"
    );
    // The survival surface is a smooth monotone transform of log Λ; on the same
    // fitted model the two surfaces must be essentially collinear.
    assert!(
        corr_surv >= 0.998,
        "gam's survival surface diverges from flexsurvspline: pearson={corr_surv:.5}"
    );

    // Model-size sanity. flexsurv reports its *nominal, unpenalized* parameter
    // count `m$npars` = (k+2) spline coefficients + 1 Age slope = N_INTERNAL_KNOTS+3.
    // gam fits the same structure but penalizes the smooth baseline by REML, so
    // its effective df must be (a) at least ~2 — a non-degenerate model needs the
    // Age slope plus a non-trivial (level/trend) baseline contribution — and
    // (b) no larger than the unpenalized MLE count `flex_npars`, since a penalty
    // can only shrink effective df below the nominal parameter count. This
    // brackets gam's penalized EDF between a hard non-degeneracy floor and
    // flexsurv's nominal count without inventing a basis-specific tolerance.
    assert!(
        (N_INTERNAL_KNOTS as f64 + 3.0 - flex_npars).abs() < 0.5,
        "flexsurv parameter count should be (k+2 spline + 1 Age) = {}, got {flex_npars}",
        N_INTERNAL_KNOTS + 3
    );
    assert!(
        gam_edf >= 2.0 && gam_edf <= flex_npars,
        "penalized RP baseline EDF out of the bracket [2, flex_npars={flex_npars}]: gam_edf={gam_edf:.3}"
    );
}
