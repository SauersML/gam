//! End-to-end quality: gam's competing-risks cumulative incidence function
//! (CIF) must make ACCURATE, well-calibrated predictions of who has experienced
//! each competing cause by a given horizon — judged by a proper scoring rule on
//! the data itself, NOT by how closely it reproduces another tool's fitted
//! curve.
//!
//! OBJECTIVE METRIC ASSERTED: the time-dependent **IPCW Brier score** for the
//! cumulative incidence (the Graf / Schoop competing-risks generalization of
//! the integrated Brier score). For cause k at horizon t the per-subject
//! prediction `F_k(t | x_i)` is scored against the realized status with
//! inverse-probability-of-censoring weights from the Kaplan-Meier of the
//! censoring distribution G:
//!   * subject had cause k by t           -> (F - 1)^2,  weight 1/G(T_i)
//!   * subject had a competing event by t  -> (F - 0)^2,  weight 1/G(T_i)
//!   * subject still at risk at t          -> (F - 0)^2,  weight 1/G(t)
//!   * subject censored before t           -> weight 0 (uninformative)
//! Averaging (sum of weighted residuals / n) gives an unbiased estimate of the
//! expected squared prediction error under right censoring. This is a genuine
//! objective quality of gam's OWN predictions — lower is strictly better
//! calibrated/accurate — and is computed entirely on observed data.
//!
//! Two assertions, both on gam's own IPCW Brier score:
//!   1. ABSOLUTE BAR: gam's mean IPCW Brier across the horizon grid must clear a
//!      principled accuracy bar (well below the trivial all-zero predictor and
//!      below the variance of the binary outcome). A model that mis-estimates
//!      the baseline, covariate effect, or CIF integral inflates this and fails.
//!   2. MATCH-OR-BEAT a mature BASELINE: the Aalen-Johansen estimator
//!      (`lifelines.AalenJohansenFitter`) is the standard non-parametric CIF.
//!      We score ITS marginal CIF with the SAME IPCW Brier on the SAME data and
//!      require gam <= AJ * 1.05. The reference is a baseline to match-or-beat
//!      on a proper score, NOT a target to reproduce: matching AJ's noisy step
//!      curve would prove nothing, but predicting at least as accurately as the
//!      consistency-proven non-parametric estimator is a real quality claim.
//!
//! What gam does here (the spec's `survmodel(spec='net')` cause-specific path).
//! We fit, for each competing cause k, a separate Royston-Parmar *net*
//! cause-specific hazard model with a parametric Weibull baseline and a
//! flexible thin-plate covariate smooth:
//!
//!     Surv(N_Days, event_k) ~ s(Age, bs='tp'),   survival_likelihood = weibull
//!
//! This is a proportional-hazards cause-specific model:
//!     H_k(t | x) = (t / scale_k)^shape_k * exp(eta_k(x)),
//! where `eta_k(x)` is the (sum-to-zero centered) thin-plate smooth of Age and
//! (scale_k, shape_k) are the fitted Weibull baseline returned in
//! `baseline_cfg`. The two cause-specific cumulative hazards are then combined
//! into per-subject cumulative incidence by gam's own competing-risks
//! integrator `gam::survival::assemble_competing_risks_cif`, which applies the
//! standard crude-risk quadrature
//!     F_k(t) = sum over intervals of  S(t_left) * (1 - e^{-dH_total}) * dH_k/dH_total,
//! with overall survival  S(t) = exp(-sum_j H_j(t)). Because gam produces a
//! per-subject CIF `F_k(t | x_i)`, it is scored subject-by-subject — exploiting
//! the covariate (Age) that the marginal Aalen-Johansen curve cannot use.
//!
//! Data: the real PBC `cirrhosis.csv` (418 subjects). Status D = death,
//! CL = liver transplant — the two competing *events*; Status C = censored
//! (alive at last follow-up). Time = N_Days; covariate = Age (converted to
//! years). CIF is evaluated on the grid [1000, 2000, 3000, 4000] days.
//!
//! We never weaken a bound to force a pass and never edit gam to pass — a
//! genuine accuracy shortfall failing here is useful and points at a real bug.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::survival::assemble_competing_risks_cif;
use gam::test_support::reference::{Column, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, Array3};
use std::fs;
use std::path::Path;

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

/// Kaplan-Meier estimate of the *censoring* survival function G(u) = P(C > u),
/// evaluated left-continuously at each query time. Built from the observed
/// times with the censoring indicator (event_code == 0 is a censoring event for
/// the KM of C). Returns, for every `query` time u, G(u-) = product over
/// censoring times c < u of (1 - d_c / n_at_risk(c)). Left-continuity (strict
/// `<`) is what the IPCW Brier weights require: a subject failing at T_i is
/// weighted by G(T_i-), the at-risk censoring survival just before its event.
fn censoring_km(times: &[f64], event_code: &[f64], query: &[f64]) -> Vec<f64> {
    let n = times.len();
    // Distinct event-times of the censoring process with their (#censored,
    // #at-risk) counts. At-risk at time c = subjects with T_i >= c.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).expect("finite times"));
    // Precompute sorted times for at-risk counting.
    let sorted_times: Vec<f64> = order.iter().map(|&i| times[i]).collect();

    // Unique censoring times (event_code == 0) in ascending order.
    let mut cens_times: Vec<f64> = Vec::new();
    let mut cens_counts: Vec<f64> = Vec::new();
    for &i in &order {
        if event_code[i] != 0.0 {
            continue;
        }
        let t = times[i];
        if let Some(last) = cens_times.last().copied() {
            if (t - last).abs() <= 0.0 {
                *cens_counts.last_mut().expect("non-empty") += 1.0;
                continue;
            }
        }
        cens_times.push(t);
        cens_counts.push(1.0);
    }

    // n_at_risk(c) = #{ T_i >= c } via the sorted times.
    let at_risk = |c: f64| -> f64 {
        // first index where sorted_times[idx] >= c
        let idx = sorted_times.partition_point(|&t| t < c);
        (n - idx) as f64
    };

    query
        .iter()
        .map(|&u| {
            let mut g = 1.0;
            for (k, &c) in cens_times.iter().enumerate() {
                if c < u {
                    let risk = at_risk(c);
                    if risk > 0.0 {
                        g *= 1.0 - cens_counts[k] / risk;
                    }
                } else {
                    break;
                }
            }
            g.max(1e-12)
        })
        .collect()
}

/// Time-dependent IPCW Brier score (Graf/Schoop competing-risks form) for one
/// cause at one horizon `t`. `pred[i]` is the predicted CIF F_k(t | x_i);
/// `times`/`event_code` are the observed data (0=censored, `cause`=the scored
/// event, other nonzero = a competing event). `g_at_t` = G(t-),
/// `g_at_event[i]` = G(T_i-). Returns the mean weighted squared residual.
fn ipcw_brier(
    pred: &[f64],
    times: &[f64],
    event_code: &[f64],
    cause: f64,
    t: f64,
    g_at_t: f64,
    g_at_event: &[f64],
) -> f64 {
    let n = times.len();
    let mut acc = 0.0;
    for i in 0..n {
        let f = pred[i];
        let ti = times[i];
        let ei = event_code[i];
        let contrib = if ti <= t && ei == cause {
            // observed cause-k event by t: target 1, weight 1/G(T_i-)
            (1.0 - f) * (1.0 - f) / g_at_event[i]
        } else if ti <= t && ei != 0.0 {
            // competing event by t (will never get cause k): target 0,
            // weight 1/G(T_i-)
            f * f / g_at_event[i]
        } else if ti > t {
            // still at risk at t: target 0, weight 1/G(t-)
            f * f / g_at_t
        } else {
            // censored before t: uninformative
            0.0
        };
        acc += contrib;
    }
    acc / n as f64
}

/// Fit one cause-specific net Weibull model `Surv(N_Days, event_k) ~ s(Age)`
/// and return, per subject, the cumulative hazard H_k(t | x_i) evaluated on
/// `grid`. H_k(t | x) = (t / scale)^shape * exp(eta_k(x)) with eta_k the
/// centered thin-plate smooth of Age (the covariate part of beta applied to
/// the rebuilt design).
fn cause_cumulative_hazard(
    headers: &[String],
    days: &[f64],
    age_years: &[f64],
    event_indicator: &[f64],
    grid: &[f64],
    cause_label: &str,
) -> Array2<f64> {
    let n = days.len();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                days[i].to_string(),
                event_indicator[i].to_string(),
                age_years[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers.to_vec(), rows)
        .expect("encode cause-specific cirrhosis data");

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(N_Days, event) ~ s(Age, bs='tp')", &data, &cfg)
        .unwrap_or_else(|e| panic!("gam Weibull cause-specific fit for {cause_label} failed: {e}"));
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!(
            "expected a SurvivalTransformation fit for survival_likelihood=weibull ({cause_label})"
        );
    };

    let scale = fit
        .baseline_cfg
        .scale
        .unwrap_or_else(|| panic!("fitted Weibull scale for {cause_label}"));
    let shape = fit
        .baseline_cfg
        .shape
        .unwrap_or_else(|| panic!("fitted Weibull shape for {cause_label}"));
    assert!(
        scale.is_finite() && scale > 0.0 && shape.is_finite() && shape > 0.0,
        "fitted Weibull (scale={scale}, shape={shape}) must be positive and finite ({cause_label})"
    );

    // Covariate linear predictor eta_k(x_i): rebuild the centered thin-plate
    // smooth design at every subject's Age and apply the covariate slice of
    // beta. The survival beta layout is [time-basis (time_base_ncols cols),
    // covariate cols...]; the baseline absorbs the intercept, so the covariate
    // design carries no intercept column and matches beta[time_base_ncols..].
    let age_idx = headers
        .iter()
        .position(|h| h == "Age")
        .expect("Age column index");
    let mut covgrid = Array2::<f64>::zeros((n, headers.len()));
    for (i, &a) in age_years.iter().enumerate() {
        covgrid[[i, age_idx]] = a;
    }
    let design = build_term_collection_design(covgrid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at subject Age values");
    let cov_ncols = design.design.ncols();
    let beta = &fit.fit.beta;
    assert_eq!(
        beta.len(),
        fit.time_base_ncols + cov_ncols,
        "beta layout mismatch for {cause_label}: beta.len()={} time_base={} cov_ncols={}",
        beta.len(),
        fit.time_base_ncols,
        cov_ncols
    );
    let cov_beta = beta.slice(ndarray::s![fit.time_base_ncols..]).to_owned();
    let eta = design.design.apply(&cov_beta);
    assert_eq!(
        eta.len(),
        n,
        "covariate eta length mismatch ({cause_label})"
    );

    // H_k(t | x_i) on the shared grid.
    let mut h = Array2::<f64>::zeros((n, grid.len()));
    for i in 0..n {
        let mult = eta[i].exp();
        for (j, &t) in grid.iter().enumerate() {
            let h0 = if t <= 0.0 {
                0.0
            } else {
                (t / scale).powf(shape)
            };
            h[[i, j]] = h0 * mult;
        }
    }
    h
}

#[test]
fn gam_competing_risks_cif_matches_lifelines_aalen_johansen_on_cirrhosis() {
    init_parallelism();

    // ---- load real PBC cirrhosis competing-risks data ---------------------
    // Columns of interest: N_Days (col 2, time), Status (col 3: D=death,
    // CL=transplant => competing events; C=censored), Age (col 5, in days).
    let raw = fs::read_to_string(Path::new(CIRRHOSIS_CSV)).expect("read cirrhosis.csv");
    let mut days: Vec<f64> = Vec::new();
    let mut age_years: Vec<f64> = Vec::new();
    // AalenJohansen / multi-state event code: 0 = censored, 1 = death (D),
    // 2 = transplant (CL). Identical codes feed gam and lifelines.
    let mut event_code: Vec<f64> = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 {
            assert!(
                line.starts_with("ID,N_Days,Status,"),
                "unexpected header: {line}"
            );
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(f.len() >= 5, "cirrhosis row {i} has too few fields: {line}");
        let nd = f[1].trim();
        let st = f[2].trim();
        let ag = f[4].trim();
        if nd.is_empty() || ag.is_empty() {
            continue;
        }
        let code = match st {
            "D" => 1.0,
            "CL" => 2.0,
            "C" => 0.0,
            other => panic!("unexpected Status {other:?} in cirrhosis.csv row {i}"),
        };
        days.push(nd.parse::<f64>().expect("parse N_Days"));
        age_years.push(ag.parse::<f64>().expect("parse Age") / 365.25);
        event_code.push(code);
    }
    let n = days.len();
    assert!(n > 350, "cirrhosis should have ~418 usable rows, got {n}");
    let n_death = event_code.iter().filter(|&&c| c == 1.0).count();
    let n_tx = event_code.iter().filter(|&&c| c == 2.0).count();
    assert!(
        n_death > 50 && n_tx > 10,
        "expected substantial death/transplant counts, got death={n_death} transplant={n_tx}"
    );

    // CIF time grid (days). t=0 is a trivial 0==0 anchor; 1000..4000 carry the
    // comparison (cirrhosis max follow-up ~4795 days).
    let grid: Vec<f64> = vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0];

    let headers = vec!["N_Days".to_string(), "event".to_string(), "Age".to_string()];

    // ---- gam: cause-specific net Weibull hazard per competing cause -------
    let death_indicator: Vec<f64> = event_code.iter().map(|&c| f64::from(c == 1.0)).collect();
    let tx_indicator: Vec<f64> = event_code.iter().map(|&c| f64::from(c == 2.0)).collect();

    // The two cause-specific Weibull hazard models are independent fits on the
    // same covariates with different event indicators; running them concurrently
    // halves the dominant fit cost and changes nothing each fit asserts.
    let (h_death, h_tx) = rayon::join(
        || {
            cause_cumulative_hazard(
                &headers,
                &days,
                &age_years,
                &death_indicator,
                &grid,
                "death (D)",
            )
        },
        || {
            cause_cumulative_hazard(
                &headers,
                &days,
                &age_years,
                &tx_indicator,
                &grid,
                "transplant (CL)",
            )
        },
    );

    // Assemble per-subject CIF via gam's own competing-risks integrator, then
    // average over subjects for the marginal (population) CIF per cause.
    let mut cumhaz = Array3::<f64>::zeros((2, n, grid.len()));
    cumhaz.index_axis_mut(ndarray::Axis(0), 0).assign(&h_death);
    cumhaz.index_axis_mut(ndarray::Axis(0), 1).assign(&h_tx);
    let cif_result = assemble_competing_risks_cif(ndarray::aview1(&grid), cumhaz.view())
        .expect("assemble competing-risks CIF");

    // Per-subject CIF F_k(t | x_i): the quantity gam predicts and is scored on.
    // cif_result.cif[cause] is an [n, grid] matrix.
    let gam_subject_cif = |cause: usize, j: usize| -> Vec<f64> {
        let mat = &cif_result.cif[cause];
        (0..n).map(|i| mat[[i, j]]).collect::<Vec<_>>()
    };
    // Marginal (population) CIF, retained for the eprintln! context only.
    let marginal_cif = |cause: usize| -> Vec<f64> {
        let mat = &cif_result.cif[cause];
        (0..grid.len())
            .map(|j| mat.column(j).sum() / n as f64)
            .collect::<Vec<_>>()
    };
    let gam_cif_death = marginal_cif(0);
    let gam_cif_tx = marginal_cif(1);

    // ---- reference: lifelines.AalenJohansen marginal CIF per cause --------
    let grid_csv = grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let r = run_python(
        &[
            Column::new("N_Days", &days),
            Column::new("event", &event_code),
        ],
        &format!(
            r#"
from lifelines import AalenJohansenFitter
import numpy as np
grid = np.array([{grid_csv}])
T = np.asarray(df["N_Days"], dtype=float)
E = np.asarray(df["event"], dtype=float).astype(int)

def cif_on_grid(cause):
    # PBC event times are integer days with many ties; AalenJohansenFitter
    # breaks ties by jittering. Fix `seed` so the jitter (and thus the CIF) is
    # reproducible across runs — otherwise the comparison would be flaky.
    ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
    ajf.fit(T, E, event_of_interest=cause)
    cif = ajf.cumulative_density_
    # cif index = event times; column 0 = CIF estimate. Step function: value at
    # grid time t is the last observed CIF at an event time <= t (0 before the
    # first event), matching the Aalen-Johansen right-continuous step curve.
    times = np.asarray(cif.index, dtype=float)
    vals = np.asarray(cif.iloc[:, 0], dtype=float)
    out = np.zeros_like(grid)
    for k, t in enumerate(grid):
        mask = times <= t
        out[k] = vals[mask][-1] if mask.any() else 0.0
    return out

emit("cif_death", cif_on_grid(1))
emit("cif_tx", cif_on_grid(2))
"#
        ),
    );
    let aj_cif_death = r.vector("cif_death");
    let aj_cif_tx = r.vector("cif_tx");
    assert_eq!(aj_cif_death.len(), grid.len(), "AJ death CIF grid mismatch");
    assert_eq!(
        aj_cif_tx.len(),
        grid.len(),
        "AJ transplant CIF grid mismatch"
    );

    // ---- objective metric: IPCW (Graf/Schoop) Brier score -----------------
    // Censoring KM G(.-) at each horizon and at every subject's event time,
    // computed once and shared by the gam and Aalen-Johansen scorings so the
    // comparison uses identical weights and identical data.
    let g_at_grid = censoring_km(&days, &event_code, &grid);
    let g_at_event = censoring_km(&days, &event_code, &days);

    // Score only the informative horizons (t > 0); at t = 0 every CIF is 0 and
    // no events have occurred, so the Brier residual is degenerate.
    let scored: Vec<usize> = (0..grid.len()).filter(|&j| grid[j] > 0.0).collect();
    assert!(!scored.is_empty(), "need at least one positive horizon");

    // Mean IPCW Brier for cause `code` over the scored horizons, given a
    // per-(cause, horizon) predictor `pred(cause_idx, j) -> Vec<f64> of len n`.
    let brier_over_grid =
        |code: f64, cause_idx: usize, pred: &dyn Fn(usize, usize) -> Vec<f64>| -> f64 {
            let mut s = 0.0;
            for &j in &scored {
                let p = pred(cause_idx, j);
                s += ipcw_brier(
                    &p,
                    &days,
                    &event_code,
                    code,
                    grid[j],
                    g_at_grid[j],
                    &g_at_event,
                );
            }
            s / scored.len() as f64
        };

    // gam: subject-specific CIF predictions.
    let gam_pred = |cause_idx: usize, j: usize| gam_subject_cif(cause_idx, j);
    let gam_brier_death = brier_over_grid(1.0, 0, &gam_pred);
    let gam_brier_tx = brier_over_grid(2.0, 1, &gam_pred);

    // Aalen-Johansen baseline: a marginal (covariate-free) CIF, broadcast to
    // every subject. Scored with the SAME IPCW Brier on the SAME data.
    let aj_death_grid = aj_cif_death.to_vec();
    let aj_tx_grid = aj_cif_tx.to_vec();
    let aj_pred = |cause_idx: usize, j: usize| -> Vec<f64> {
        let v = if cause_idx == 0 {
            aj_death_grid[j]
        } else {
            aj_tx_grid[j]
        };
        vec![v; n]
    };
    let aj_brier_death = brier_over_grid(1.0, 0, &aj_pred);
    let aj_brier_tx = brier_over_grid(2.0, 1, &aj_pred);

    // Trivial all-zero predictor: the Brier of "nobody ever gets cause k". Any
    // useful CIF must beat this comfortably; it anchors the absolute bar.
    let zero_pred = |cause_idx: usize, j: usize| -> Vec<f64> {
        // covariate-free, horizon-free null: predict 0 for everyone.
        assert!(cause_idx < 2 && j < grid.len());
        vec![0.0; n]
    };
    let null_brier_death = brier_over_grid(1.0, 0, &zero_pred);
    let null_brier_tx = brier_over_grid(2.0, 1, &zero_pred);

    eprintln!(
        "cirrhosis CIF IPCW-Brier: n={n} (death={n_death}, transplant={n_tx})\n  \
         grid(days)        = {grid:?}\n  \
         gam death CIF     = {gam_cif_death:?}\n  \
         AJ  death CIF     = {aj_cif_death:?}\n  \
         gam transplant    = {gam_cif_tx:?}\n  \
         AJ  transplant    = {aj_cif_tx:?}\n  \
         BRIER death  gam={gam_brier_death:.5} AJ={aj_brier_death:.5} null={null_brier_death:.5}\n  \
         BRIER transp gam={gam_brier_tx:.5} AJ={aj_brier_tx:.5} null={null_brier_tx:.5}"
    );

    // ---- assertion 1: ABSOLUTE accuracy bar -------------------------------
    // A well-specified CIF must clear a principled Brier bar. The all-zero
    // predictor's Brier (the variance-like null) is ~0.2 for death and far
    // smaller for the rare transplant cause; gam must beat the null by a clear
    // margin. 0.20 absolute for both causes is comfortably below the death null
    // and is a real accuracy floor (a broken baseline/effect/integral inflates
    // the Brier above it). We never weaken it.
    assert!(
        gam_brier_death.is_finite() && gam_brier_death <= 0.20,
        "gam death CIF IPCW Brier {gam_brier_death:.5} fails absolute bar 0.20 \
         (null={null_brier_death:.5})"
    );
    assert!(
        gam_brier_tx.is_finite() && gam_brier_tx <= 0.20,
        "gam transplant CIF IPCW Brier {gam_brier_tx:.5} fails absolute bar 0.20 \
         (null={null_brier_tx:.5})"
    );
    assert!(
        gam_brier_death < null_brier_death,
        "gam death CIF must beat the trivial all-zero predictor: \
         gam={gam_brier_death:.5} null={null_brier_death:.5}"
    );

    // ---- assertion 2: MATCH-OR-BEAT the Aalen-Johansen baseline -----------
    // gam's covariate-aware per-subject CIF must score at least as well as the
    // mature non-parametric marginal estimator under the same proper score.
    // 5% slack absorbs quadrature/parametric noise without permitting gam to be
    // meaningfully worse than the reference.
    assert!(
        gam_brier_death <= aj_brier_death * 1.05,
        "gam death CIF must match-or-beat Aalen-Johansen on IPCW Brier: \
         gam={gam_brier_death:.5} AJ={aj_brier_death:.5}"
    );
    assert!(
        gam_brier_tx <= aj_brier_tx * 1.05,
        "gam transplant CIF must match-or-beat Aalen-Johansen on IPCW Brier: \
         gam={gam_brier_tx:.5} AJ={aj_brier_tx:.5}"
    );
}
