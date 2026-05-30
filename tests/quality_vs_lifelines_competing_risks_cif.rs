//! End-to-end quality: gam's competing-risks cumulative incidence function
//! (CIF) must reproduce the mature non-parametric reference for competing
//! risks — `lifelines.AalenJohansen` — on real multi-cause survival data.
//!
//! Why lifelines.AalenJohansen. The Aalen-Johansen estimator is the standard,
//! consistency-proven non-parametric estimator of the cause-specific
//! cumulative incidence function under right censoring and competing risks
//! (it is the multi-state generalization of Kaplan-Meier). It makes *no*
//! distributional assumption, so it is the natural ground truth against which
//! to benchmark gam's semi-parametric cause-specific construction.
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
//! with overall survival  S(t) = exp(-sum_j H_j(t)).
//!
//! Marginal comparison. AalenJohansen estimates the *marginal* (population)
//! CIF. The comparable gam quantity is the g-formula / direct-standardization
//! marginal CIF: the per-subject CIF averaged over the empirical Age
//! distribution of the cohort, F_k(t) = (1/n) sum_i F_k(t | x_i). Averaging the
//! per-subject CIFs (not exp(eta) before integrating — that would be Jensen-
//! biased) is the correct standardized estimand and converges to the same
//! population CIF that AalenJohansen estimates non-parametrically.
//!
//! Data: the real PBC `cirrhosis.csv` (418 subjects). Status D = death,
//! CL = liver transplant — the two competing *events*; Status C = censored
//! (alive at last follow-up). Time = N_Days; covariate = Age (converted to
//! years). CIF is evaluated on the grid [0, 1000, 2000, 3000, 4000] days.
//!
//! Bound. lifelines uses a product-limit / Aalen-Johansen scheme; gam uses a
//! parametric two-parameter Weibull PH baseline (no time spline in pure Weibull
//! mode — the linear log-time basis *is* the Weibull, so scale/shape are read
//! straight from baseline_cfg) marginalized via crude-risk quadrature. These
//! are genuinely different schemes on the same data. The death CIF reaches
//! ~0.4 by 4000 days; a well-specified Weibull tracks the Aalen-Johansen step
//! curve to a few percent of that level, so a ~1-2% absolute CIF difference is
//! the expected order. We assert max |gam - AJ| <= 0.015 per cause across the
//! grid: tight enough that a real divergence in gam's baseline, covariate
//! effect, or CIF integral fails the test, yet loose enough to absorb the
//! parametric-vs-nonparametric and quadrature difference. We never weaken it
//! and never edit gam to pass — a genuine divergence failing here is useful and
//! would point at a real bug rather than the (bounded) scheme mismatch.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::survival::assemble_competing_risks_cif;
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, Array3};
use std::fs;
use std::path::Path;

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

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

    let h_death = cause_cumulative_hazard(
        &headers,
        &days,
        &age_years,
        &death_indicator,
        &grid,
        "death (D)",
    );
    let h_tx = cause_cumulative_hazard(
        &headers,
        &days,
        &age_years,
        &tx_indicator,
        &grid,
        "transplant (CL)",
    );

    // Assemble per-subject CIF via gam's own competing-risks integrator, then
    // average over subjects for the marginal (population) CIF per cause.
    let mut cumhaz = Array3::<f64>::zeros((2, n, grid.len()));
    cumhaz.index_axis_mut(ndarray::Axis(0), 0).assign(&h_death);
    cumhaz.index_axis_mut(ndarray::Axis(0), 1).assign(&h_tx);
    let cif_result = assemble_competing_risks_cif(ndarray::aview1(&grid), cumhaz.view())
        .expect("assemble competing-risks CIF");

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

    // ---- compare ----------------------------------------------------------
    let diff_death = max_abs_diff(&gam_cif_death, aj_cif_death);
    let diff_tx = max_abs_diff(&gam_cif_tx, aj_cif_tx);

    eprintln!(
        "cirrhosis CIF: n={n} (death={n_death}, transplant={n_tx})\n  \
         grid(days)      = {grid:?}\n  \
         gam death CIF   = {gam_cif_death:?}\n  \
         AJ  death CIF   = {aj_cif_death:?}  (max_abs_diff={diff_death:.4})\n  \
         gam transplant  = {gam_cif_tx:?}\n  \
         AJ  transplant  = {aj_cif_tx:?}  (max_abs_diff={diff_tx:.4})"
    );

    // Different numerical schemes (parametric Weibull + crude-risk quadrature
    // vs non-parametric product-limit Aalen-Johansen) on the same data; ~1-2%
    // absolute CIF agreement is the expected order. 0.015 catches any real
    // divergence yet honestly absorbs the scheme difference.
    assert!(
        diff_death <= 0.015,
        "death CIF diverges from lifelines AalenJohansen: max_abs_diff={diff_death:.4}\n  \
         gam={gam_cif_death:?}\n  AJ={aj_cif_death:?}"
    );
    assert!(
        diff_tx <= 0.015,
        "transplant CIF diverges from lifelines AalenJohansen: max_abs_diff={diff_tx:.4}\n  \
         gam={gam_cif_tx:?}\n  AJ={aj_cif_tx:?}"
    );
}
