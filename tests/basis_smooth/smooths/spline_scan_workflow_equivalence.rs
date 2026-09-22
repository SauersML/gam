//! Workflow gate for the exact O(n) cubic-smoothing-spline scan fast path.
//!
//! 1. Equivalence gate: a `y ~ s(x, double_penalty=false)` Gaussian fit routed
//!    through the public formula entry (`fit_spline_scan_from_formula`) must
//!    reproduce, at the scan-selected λ, the EXACT dense posterior of the same
//!    intrinsic order-2 prior — self-constructed dense truth (#904 style, same
//!    construction pattern as tests/spline_scan_exact_oracle.rs): fitted
//!    values, SEs, and the exact EDF identity `tr(S) = Σ w_t·Var_t/σ²` agree
//!    to 1e-6 relative.
//! 2. e2e truth recovery: the scan-routed fit's predictions recover a known
//!    smooth truth at least as well as the dense reduced-rank path on the same
//!    formula (match-or-beat), plus an absolute truth-recovery bound.
//! 3. Detection negatives: ineligible shapes fall through (`Ok(None)`).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    fit_model, fit_spline_scan_from_formula, init_parallelism, materialize,
};
use ndarray::Array2;

/// Dense reference fit on a scan-eligible formula, bypassing the `fit_from_formula`
/// auto-route (#1030) so the dense reduced-rank path is available as a
/// match-or-beat baseline. `fit_from_formula` now structurally routes a single
/// 1-D Gaussian single-penalty smooth through the exact O(n) scan and returns
/// `FitResult::SplineScan`; to compare the scan against the dense estimator we
/// materialize and call `fit_model` (which carries no fast path) directly.
fn dense_reference_fit(
    formula: &str,
    data: &gam::data::EncodedDataset,
    cfg: &FitConfig,
) -> StandardFitResult {
    let mat = materialize(formula, data, cfg).expect("materialize dense reference");
    match fit_model(mat.request).expect("dense fit") {
        FitResult::Standard(r) => r,
        _ => panic!("dense reference path must return a standard fit"),
    }
}

/// Dense in-test Gaussian elimination solve A·X = B (partial pivoting).
fn dense_solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend_from_slice(&b[i]);
            row
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap();
        aug.swap(col, piv);
        let p = aug[col][col];
        assert!(p.abs() > 1e-300, "dense oracle: singular pivot");
        for i in 0..n {
            if i == col {
                continue;
            }
            let f = aug[i][col] / p;
            if f == 0.0 {
                continue;
            }
            for k in col..n + m {
                aug[i][k] -= f * aug[col][k];
            }
        }
    }
    (0..n)
        .map(|i| (0..m).map(|j| aug[i][n + j] / aug[i][i]).collect())
        .collect()
}

/// Dense exact posterior (mean + pointwise variance at unit σ²) of the SAME
/// intrinsic order-2 Markov prior the scan integrates: joint precision over
/// states (f_t, f'_t) at the sorted distinct abscissae, improper (zero) prior
/// on the first state = the diffuse `{1, x}` null space.
fn dense_truth(x: &[f64], y: &[f64], w: &[f64], log_lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let m = x.len();
    let q = (-log_lambda).exp();
    let dim = 2 * m;
    let mut prior = vec![vec![0.0_f64; dim]; dim];
    for t in 0..m - 1 {
        let d = x[t + 1] - x[t];
        let (d2, d3) = (d * d, d * d * d);
        let (a, b, c) = (q * d3 / 3.0, q * d2 / 2.0, q * d);
        let det = a * c - b * b;
        let qi = [[c / det, -b / det], [-b / det, a / det]];
        let trows = [[-1.0, -d, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]];
        for r1 in 0..2 {
            for r2 in 0..2 {
                let coef = qi[r1][r2];
                if coef == 0.0 {
                    continue;
                }
                for c1 in 0..4 {
                    if trows[r1][c1] == 0.0 {
                        continue;
                    }
                    for c2 in 0..4 {
                        if trows[r2][c2] == 0.0 {
                            continue;
                        }
                        prior[2 * t + c1][2 * t + c2] += trows[r1][c1] * coef * trows[r2][c2];
                    }
                }
            }
        }
    }
    let mut lambda = prior;
    let mut rhs = vec![vec![0.0_f64]; dim];
    for t in 0..m {
        lambda[2 * t][2 * t] += w[t];
        rhs[2 * t][0] = w[t] * y[t];
    }
    let mean_full = dense_solve(&lambda, &rhs);
    let eye: Vec<Vec<f64>> = (0..dim)
        .map(|i| (0..dim).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    let cov = dense_solve(&lambda, &eye);
    let mean: Vec<f64> = (0..m).map(|t| mean_full[2 * t][0]).collect();
    let var: Vec<f64> = (0..m).map(|t| cov[2 * t][2 * t]).collect();
    (mean, var)
}

/// Deterministic truth + quasi-noise training data (no RNG): irregular
/// abscissae, smooth truth `sin(6x) + 0.5x²`, golden-ratio rotation noise.
fn truth_fn(x: f64) -> f64 {
    (6.0 * x).sin() + 0.5 * x * x
}

fn training_xy(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        // Mildly irregular spacing.
        let xi = u + 0.35 * (std::f64::consts::PI * u).sin() / (n as f64);
        let noise = ((i as f64 * 0.618_033_988_749_894_9).fract() - 0.5) * 0.3;
        x.push(xi);
        y.push(truth_fn(xi) + noise);
    }
    (x, y)
}

fn encode_xy(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn gaussian_config() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

const SCAN_FORMULA: &str = "y ~ s(x, double_penalty=false)";
/// Order-1 (random-walk / linear) smoothing spline: degree 2m−1 = 1,
/// penalty_order m = 1 — the form `spline_scan_fast_path` routes to the m = 1
/// exact O(n) scan (#1034 item 2).
const SCAN_FORMULA_M1: &str = "y ~ s(x, degree=1, penalty_order=1, double_penalty=false)";
/// Order-3 (quintic) smoothing spline: degree 2m−1 = 5, penalty_order m = 3 —
/// the form `spline_scan_fast_path` routes to the m = 3 exact O(n) scan (#1044),
/// whose two partially-diffuse leading nodes are recovered by the exact diffuse
/// leading-block smoother.
const SCAN_FORMULA_M3: &str = "y ~ s(x, degree=5, penalty_order=3, double_penalty=false)";

#[test]
fn scan_routed_workflow_fit_matches_dense_oracle_at_selected_lambda() {
    init_parallelism();
    let (x, y) = training_xy(140);
    let data = encode_xy(&x, &y);
    let fit = fit_spline_scan_from_formula(SCAN_FORMULA, &data, &gaussian_config())
        .expect("scan-routed fit")
        .expect("detection must fire for a single 1-D single-penalty Gaussian smooth");

    // Dense oracle at the scan-selected λ over the scan's own pooled knots
    // (unweighted training data ⇒ unit weights per distinct abscissa; the
    // training x here are strictly increasing, so knots == x).
    assert_eq!(fit.knots.len(), x.len(), "no ties expected in this design");
    let w = vec![1.0_f64; fit.knots.len()];
    let knot_y: Vec<f64> = {
        // Map responses to the sorted-knot order.
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
        idx.iter().map(|&i| y[i]).collect()
    };
    let (oracle_mean, oracle_var_unit) = dense_truth(&fit.knots, &knot_y, &w, fit.log_lambda());

    for t in 0..fit.knots.len() {
        let dm = (fit.mean[t] - oracle_mean[t]).abs();
        assert!(
            dm <= 1e-6 * oracle_mean[t].abs().max(1e-3),
            "fitted value mismatch at knot {t}: scan={} dense={}",
            fit.mean[t],
            oracle_mean[t]
        );
        let se_scan = fit.var[t].sqrt();
        let se_dense = (oracle_var_unit[t] * fit.sigma2).sqrt();
        assert!(
            (se_scan - se_dense).abs() <= 1e-6 * se_dense.max(1e-12),
            "SE mismatch at knot {t}: scan={se_scan} dense={se_dense}"
        );
    }

    // Exact EDF identity cross-check: tr(S) = Σ_t w_t · Var_t/σ², which at
    // unit-σ² scale is Σ_t w_t · C̃_tt — computable independently from the
    // dense oracle's posterior covariance diagonal.
    let dense_edf: f64 = w
        .iter()
        .zip(oracle_var_unit.iter())
        .map(|(wt, vt)| wt * vt)
        .sum();
    let scan_edf = fit.edf();
    assert!(
        (scan_edf - dense_edf).abs() <= 1e-6 * dense_edf.max(1e-12),
        "EDF identity mismatch: scan tr(S)={scan_edf} dense tr(S)={dense_edf}"
    );
    assert!(
        scan_edf > 2.0 && scan_edf < x.len() as f64,
        "EDF {scan_edf} must exceed the unpenalized null-space dimension 2 and stay below n"
    );
}

/// Observation-interval oracle (#1047): the scan-routed predict arm emits the
/// response-scale predictive (observation) band `mean ± z·√(Var(f) + σ²)`. This
/// reconstructs the exact band the FFI predict arm builds in
/// `crates/gam-pyffi/src/lib.rs` (`predict_columns`, scan branch) from the same
/// public `SplineScanFit` surface it reads — `se = √fit.var`, then
/// `obs_se = √(se² + fit.sigma2)` — and asserts it equals an INDEPENDENT dense
/// predictive band built from the self-constructed dense posterior of the same
/// order-2 prior: `obs_se_dense = √(σ²·C̃_tt + σ²)`. The two agree to tight tol,
/// which pins the FFI scan observation interval to the dense path's
/// `observation_interval` it must match. It also confirms the band is the
/// credible band STRICTLY inflated by σ² (a dropped/degenerate σ² == 0 would
/// collapse the two and fail).
#[test]
fn scan_routed_observation_interval_matches_dense_predictive_band() {
    init_parallelism();
    let (x, y) = training_xy(140);
    let data = encode_xy(&x, &y);
    let fit = fit_spline_scan_from_formula(SCAN_FORMULA, &data, &gaussian_config())
        .expect("scan-routed fit")
        .expect("detection must fire for a single 1-D single-penalty Gaussian smooth");

    assert_eq!(fit.knots.len(), x.len(), "no ties expected in this design");
    let w = vec![1.0_f64; fit.knots.len()];
    let knot_y: Vec<f64> = {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
        idx.iter().map(|&i| y[i]).collect()
    };
    let (_oracle_mean, oracle_var_unit) = dense_truth(&fit.knots, &knot_y, &w, fit.log_lambda());

    // Same z the FFI scan arm uses (two-sided 95%).
    let level = 0.95_f64;
    let z = gam::probability::standard_normal_quantile(0.5 + level * 0.5).expect("normal quantile");
    assert!(fit.sigma2 > 0.0, "profiled σ² must be strictly positive");

    for t in 0..fit.knots.len() {
        // FFI scan arm reconstruction: confidence SE then σ²-inflated predictive SE.
        let se = fit.var[t].max(0.0).sqrt();
        let obs_se = (se * se + fit.sigma2).max(0.0).sqrt();
        let obs_half = z * obs_se;
        let mean_half = z * se;

        // Independent dense predictive half-width from the self-constructed
        // posterior: Var(y*) = σ²·C̃_tt + σ².
        let dense_var_f = oracle_var_unit[t] * fit.sigma2;
        let dense_obs_half = z * (dense_var_f + fit.sigma2).sqrt();

        assert!(
            (obs_half - dense_obs_half).abs() <= 1e-6 * dense_obs_half.max(1e-12),
            "observation half-width mismatch at knot {t}: scan={obs_half} dense={dense_obs_half}"
        );
        // Predictive band strictly wider than the credible band by exactly the
        // σ² inflation — rejects a dropped or degenerate σ².
        assert!(
            obs_half > mean_half,
            "observation band not strictly wider than credible band at knot {t}"
        );
        let expected_inflation = z * ((se * se + fit.sigma2).sqrt() - se);
        assert!(
            ((obs_half - mean_half) - expected_inflation).abs()
                <= 1e-9 * expected_inflation.max(1e-12),
            "σ² inflation mismatch at knot {t}"
        );
    }
}

#[test]
fn scan_routed_fit_recovers_truth_at_least_as_well_as_dense_path() {
    init_parallelism();
    let (x, y) = training_xy(160);
    let data = encode_xy(&x, &y);
    let cfg = gaussian_config();

    // Scan-routed fast path.
    let scan = fit_spline_scan_from_formula(SCAN_FORMULA, &data, &cfg)
        .expect("scan-routed fit")
        .expect("detection must fire");

    // The library entry now structurally auto-routes this scan-eligible shape
    // (#1030): `fit_from_formula` returns the exact O(n) scan posterior.
    match fit_from_formula(SCAN_FORMULA, &data, &cfg).expect("auto-routed fit") {
        FitResult::SplineScan(_) => {}
        _ => panic!("fit_from_formula must auto-route a single 1-D Gaussian single-penalty smooth"),
    }

    // Dense reference path on the identical model formula (bypasses the
    // auto-route so the reduced-rank dense estimator is the match-or-beat
    // baseline).
    let dense = dense_reference_fit(SCAN_FORMULA, &data, &cfg);

    // Interior prediction grid (truth-recovery, #904 style: the assertion is
    // against the self-constructed truth; the mature/dense path is only the
    // match-or-beat baseline).
    let grid: Vec<f64> = (0..200).map(|i| 0.02 + 0.96 * i as f64 / 199.0).collect();
    let mut dense_design = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &t) in grid.iter().enumerate() {
        dense_design[[i, 0]] = t;
    }
    let design = build_term_collection_design(dense_design.view(), &dense.resolvedspec)
        .expect("dense predict design rebuild");
    let dense_pred = design.design.apply(&dense.fit.beta).to_vec();

    let mut scan_sse = 0.0;
    let mut dense_sse = 0.0;
    for (i, &t) in grid.iter().enumerate() {
        let truth = truth_fn(t);
        let (scan_mean, scan_var) = scan.predict(t).expect("scan predict");
        assert!(
            scan_mean.is_finite() && scan_var.is_finite() && scan_var > 0.0,
            "scan prediction must be finite with positive variance at x={t}"
        );
        scan_sse += (scan_mean - truth) * (scan_mean - truth);
        dense_sse += (dense_pred[i] - truth) * (dense_pred[i] - truth);
    }
    let scan_mse = scan_sse / grid.len() as f64;
    let dense_mse = dense_sse / grid.len() as f64;

    // Absolute truth-recovery: well under the noise variance (~0.0075).
    assert!(
        scan_mse < 0.01,
        "scan-routed fit fails absolute truth recovery: MSE={scan_mse}"
    );
    // Match-or-beat the dense reduced-rank path on the same formula.
    assert!(
        scan_mse <= 1.10 * dense_mse + 1e-12,
        "scan-routed fit worse than dense path: scan MSE={scan_mse}, dense MSE={dense_mse}"
    );
}

/// #1030, #2627: the scan recovers the truth at the largest size this lane can
/// run, and reports a sane EDF there.
///
/// # What this test used to assert, and why it no longer does
///
/// It fitted `n = 1e5` and `n = 1e6` and gated `t_big <= 30 * t_small` as a
/// sub-quadratic check, under the name `..._scales_linearly_...`. Three things
/// were wrong with that and #2627 measured all of them.
///
/// **The fit is not linear.** Its cost is `evaluations x O(n)`, and the left
/// factor is not flat: the certified log-lambda search spends
/// `411.7 + 5.67*sqrt(n)` criterion evaluations, measured across two disjoint
/// size ranges agreeing on the coefficient to 5%. Per evaluation the cost is
/// cleanly linear, exponent `0.994`. The fit is therefore `n^1.5`. The
/// attribution names the mechanism: the growth is the dominated-cell census of
/// the subdivision search, not its enclosure.
///
/// **The bar admitted the defect.** On that law the ratio from `1e5` to `1e6` is
/// `27.6` against a bar of `30`, so the test passed with 8% to spare while the
/// cost was superlinear, and its comment attributed the slack to a logarithmic
/// search overhead that is really the square-root census.
///
/// **The sizes could not run.** One fit at `n = 1e5` is about 4.5 hours and at
/// `n = 1e6` about 123 hours, from a law that reproduces the three measured
/// wall-clocks to within 2%.
///
/// The deterministic content — how the evaluation count grows — is now asserted
/// directly against the search's own subdivision budget in
/// `certified_search_evaluation_count_stays_inside_its_subdivision_budget_2627`
/// (`gam-solve`, about 285 s). A deterministic assertion on the mechanism beats
/// a wall-clock ratio in a shared binary, which measures the machine as much as
/// the code. Nothing here times anything.
///
/// # What it asserts now
///
/// Truth recovery and the EDF, at `n = 4_000`: the largest size that stays well
/// inside the per-test cap, at about 230 s measured. The recovery bar is
/// match-or-beat against the dense reduced-rank path on the same formula, the
/// shape `scan_routed_fit_recovers_truth_at_least_as_well_as_dense_path` already
/// uses, which needs no absolute constant. The retired `mse < 1e-3` was chosen
/// for `n = 1e6` and does not transfer: the variance floor `sigma^2*edf/n`, with
/// the fixture's injected `sigma^2 = 0.0075` and an EDF near 50, is `9.4e-5` at
/// `n = 4_000` against `3.8e-7` at `n = 1e6`, so an absolute bar would have to
/// move with `n` while a match-or-beat does not.
///
/// This is not biobank scale, and it does not claim to be. That the scan cannot
/// be exercised at biobank scale in any lane until the evaluation count is fixed
/// is the finding, recorded on #2627, not something for a bar to paper over.
#[test]
fn spline_scan_recovers_truth_and_a_sane_edf_at_scale() {
    init_parallelism();
    const N: usize = 4_000;
    let cfg = gaussian_config();
    let (x, y) = training_xy(N);
    let data = encode_xy(&x, &y);

    let scan = fit_spline_scan_from_formula(SCAN_FORMULA, &data, &cfg)
        .expect("scan-routed fit")
        .expect("detection must fire for a single 1-D single-penalty Gaussian smooth");

    // The dense reduced-rank estimator on the identical formula, as the
    // match-or-beat baseline. It bypasses the auto-route so the comparison is
    // between two estimators of the same model, not the scan against itself.
    let dense = dense_reference_fit(SCAN_FORMULA, &data, &cfg);

    let grid: Vec<f64> = (0..200).map(|i| 0.02 + 0.96 * i as f64 / 199.0).collect();
    let mut dense_design = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &t) in grid.iter().enumerate() {
        dense_design[[i, 0]] = t;
    }
    let design = build_term_collection_design(dense_design.view(), &dense.resolvedspec)
        .expect("dense predict design rebuild");
    let dense_pred = design.design.apply(&dense.fit.beta).to_vec();

    let mut scan_sse = 0.0;
    let mut dense_sse = 0.0;
    for (i, &t) in grid.iter().enumerate() {
        let truth = truth_fn(t);
        let (mean, var) = scan.predict(t).expect("scan predict at scale");
        assert!(
            mean.is_finite() && var.is_finite() && var > 0.0,
            "scan prediction must be finite with positive variance at x={t}"
        );
        scan_sse += (mean - truth) * (mean - truth);
        dense_sse += (dense_pred[i] - truth) * (dense_pred[i] - truth);
    }
    let scan_mse = scan_sse / grid.len() as f64;
    let dense_mse = dense_sse / grid.len() as f64;
    let edf = scan.edf();
    eprintln!(
        "[spline-scan scale] n={N} scan MSE={scan_mse:.6e} dense MSE={dense_mse:.6e} edf={edf:.3}"
    );

    // Match-or-beat the dense reduced-rank path on the same formula, at a size
    // where both estimators are solvable.
    assert!(
        scan_mse <= 1.10 * dense_mse + 1e-12,
        "scan-routed fit worse than dense path at n={N}: scan MSE={scan_mse}, dense MSE={dense_mse}"
    );

    // A real smooth was fit: above the unpenalized null-space dimension and far
    // below the row count, so neither the linear trend nor an interpolant.
    assert!(
        edf > 2.0 && edf < N as f64 / 4.0,
        "scan EDF {edf} outside the sane band (2, {}) at n={N}",
        N as f64 / 4.0
    );
}

/// #1034 item 2 end-to-end: the order-1 (random-walk/linear) smoothing spline
/// routes through the formula → detection → exact O(n) scan → predict pipeline.
/// Detection must fire, the fit must carry `order == 1`, and the scan must
/// recover the known smooth truth (#904 self-constructed truth, PRIMARY).
///
/// NO match-or-beat against the dense path here, deliberately: the dense
/// formula fit is a DIFFERENT estimator — a reduced-rank (~40-knot) degree-1
/// B-spline with the UNWEIGHTED first-difference penalty, vs the scan's exact
/// full-knot spacing-weighted random-walk spline (`λ∫f′²`). For a smooth
/// truth the rank reduction acts as extra regularization and can legitimately
/// win on MSE; #904 reserves match-or-beat for references computing the SAME
/// estimand. Exactness of the scan's own estimand is gated by the dense
/// random-walk joint-precision oracle in the unit tests. A wide catastrophe
/// guard (≤ 5× dense MSE) remains to catch silent λ-selection collapse
/// (#1023 class) without asserting cross-estimator dominance.
#[test]
fn order_one_scan_routes_and_recovers_truth_vs_dense() {
    init_parallelism();
    let (x, y) = training_xy(180);
    let data = encode_xy(&x, &y);
    let cfg = gaussian_config();

    let scan = fit_spline_scan_from_formula(SCAN_FORMULA_M1, &data, &cfg)
        .expect("scan-routed m=1 fit")
        .expect("detection must fire for a single 1-D order-1 single-penalty Gaussian smooth");
    assert_eq!(scan.order, 1, "m=1 formula must select the order-1 scan");

    // The library entry must also auto-route the m=1 scan shape (#1030).
    match fit_from_formula(SCAN_FORMULA_M1, &data, &cfg).expect("auto-routed m=1 fit") {
        FitResult::SplineScan(s) => assert_eq!(s.order, 1, "auto-route must select the m=1 scan"),
        _ => panic!("fit_from_formula must auto-route the m=1 scan shape"),
    }

    // Dense reduced-rank reference on the identical formula (bypasses the
    // auto-route, see `dense_reference_fit`).
    let dense = dense_reference_fit(SCAN_FORMULA_M1, &data, &cfg);
    let grid: Vec<f64> = (0..200).map(|i| 0.02 + 0.96 * i as f64 / 199.0).collect();
    let mut dense_design = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &t) in grid.iter().enumerate() {
        dense_design[[i, 0]] = t;
    }
    let design = build_term_collection_design(dense_design.view(), &dense.resolvedspec)
        .expect("dense m=1 predict design rebuild");
    let dense_pred = design.design.apply(&dense.fit.beta).to_vec();

    let mut scan_sse = 0.0;
    let mut dense_sse = 0.0;
    for (i, &t) in grid.iter().enumerate() {
        let truth = truth_fn(t);
        let (scan_mean, scan_var) = scan.predict(t).expect("m=1 scan predict");
        assert!(
            scan_mean.is_finite() && scan_var.is_finite() && scan_var > 0.0,
            "m=1 scan prediction must be finite with positive variance at x={t}"
        );
        scan_sse += (scan_mean - truth) * (scan_mean - truth);
        dense_sse += (dense_pred[i] - truth) * (dense_pred[i] - truth);
    }
    let scan_mse = scan_sse / grid.len() as f64;
    let dense_mse = dense_sse / grid.len() as f64;
    // PRIMARY — the order-1 smoother is piecewise-linear, so it tracks the
    // curved truth more coarsely than the cubic arm, but must still resolve
    // it well under the noise variance.
    assert!(
        scan_mse < 0.02,
        "m=1 scan fails absolute truth recovery: MSE={scan_mse}"
    );
    // Catastrophe guard only (see header): different estimators, so no
    // dominance claim — but a silent λ-selection collapse would blow far
    // past 5× the reduced-rank fit.
    assert!(
        scan_mse <= 5.0 * dense_mse + 1e-12,
        "m=1 scan catastrophically worse than the reduced-rank fit: scan MSE={scan_mse}, dense MSE={dense_mse}"
    );
}

/// #1044 end-to-end: the order-3 (quintic) smoothing spline routes through the
/// formula → detection → exact O(n) scan → predict pipeline. Detection must
/// fire, the fit must carry `order == 3`, and the scan must recover the known
/// smooth truth (#904 self-constructed truth, PRIMARY).
///
/// As with the m=1 arm, NO match-or-beat against the dense path: the dense
/// formula fit is a DIFFERENT estimator (a reduced-rank degree-5 B-spline with
/// a discrete third-difference penalty) vs the scan's exact full-knot
/// spacing-weighted quintic spline (`λ∫(f‴)²`). Exactness of the scan's own
/// estimand is gated by the dense order-3 joint-precision oracle in
/// tests/spline_scan_exact_oracle.rs. A wide catastrophe guard (≤ 5× dense MSE)
/// catches a silent λ-selection or leading-node collapse without asserting
/// cross-estimator dominance.
#[test]
fn order_three_scan_routes_and_recovers_truth_vs_dense() {
    init_parallelism();
    let (x, y) = training_xy(200);
    let data = encode_xy(&x, &y);
    let cfg = gaussian_config();

    let scan = fit_spline_scan_from_formula(SCAN_FORMULA_M3, &data, &cfg)
        .expect("scan-routed m=3 fit")
        .expect("detection must fire for a single 1-D order-3 single-penalty Gaussian smooth");
    assert_eq!(scan.order, 3, "m=3 formula must select the order-3 scan");

    // The library entry must also auto-route the m=3 scan shape (#1030).
    match fit_from_formula(SCAN_FORMULA_M3, &data, &cfg).expect("auto-routed m=3 fit") {
        FitResult::SplineScan(s) => assert_eq!(s.order, 3, "auto-route must select the m=3 scan"),
        _ => panic!("fit_from_formula must auto-route the m=3 scan shape"),
    }

    // Dense reduced-rank reference on the identical formula (bypasses the
    // auto-route, see `dense_reference_fit`).
    let dense = dense_reference_fit(SCAN_FORMULA_M3, &data, &cfg);
    let grid: Vec<f64> = (0..200).map(|i| 0.02 + 0.96 * i as f64 / 199.0).collect();
    let mut dense_design = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &t) in grid.iter().enumerate() {
        dense_design[[i, 0]] = t;
    }
    let design = build_term_collection_design(dense_design.view(), &dense.resolvedspec)
        .expect("dense m=3 predict design rebuild");
    let dense_pred = design.design.apply(&dense.fit.beta).to_vec();

    let mut scan_sse = 0.0;
    let mut dense_sse = 0.0;
    for (i, &t) in grid.iter().enumerate() {
        let truth = truth_fn(t);
        let (scan_mean, scan_var) = scan.predict(t).expect("m=3 scan predict");
        assert!(
            scan_mean.is_finite() && scan_var.is_finite() && scan_var > 0.0,
            "m=3 scan prediction must be finite with positive variance at x={t}"
        );
        scan_sse += (scan_mean - truth) * (scan_mean - truth);
        dense_sse += (dense_pred[i] - truth) * (dense_pred[i] - truth);
    }
    let scan_mse = scan_sse / grid.len() as f64;
    let dense_mse = dense_sse / grid.len() as f64;
    // PRIMARY — the quintic smoother is C⁴ and tracks the curved truth at least
    // as tightly as the cubic; it must resolve it well under the noise variance.
    assert!(
        scan_mse < 0.01,
        "m=3 scan fails absolute truth recovery: MSE={scan_mse}"
    );
    // Catastrophe guard only (different estimators).
    assert!(
        scan_mse <= 5.0 * dense_mse + 1e-12,
        "m=3 scan catastrophically worse than the reduced-rank fit: scan MSE={scan_mse}, dense MSE={dense_mse}"
    );
}

#[test]
fn detection_falls_through_for_ineligible_shapes() {
    init_parallelism();
    let (x, y) = training_xy(60);
    let data = encode_xy(&x, &y);
    let cfg = gaussian_config();

    // Default s(x) carries the double (null-space shrinkage) penalty. That
    // second penalty — the Marra & Wood (2011) null-space ridge that lets REML
    // shrink the unpenalized polynomial null space toward EDF→0 (mgcv
    // `select = TRUE`) — is NOT representable by the scan, whose null space is an
    // improper diffuse prior it can never shrink. Routing a double-penalty fit
    // through the scan silently drops that penalty and inflates EDF (#1266), so
    // it must fall through to the dense two-rho path that owns both penalties.
    let default_s = fit_spline_scan_from_formula("y ~ s(x)", &data, &cfg).expect("materialize");
    assert!(
        default_s.is_none(),
        "double-penalty default s(x) must fall through to the dense two-rho path (#1266)"
    );

    // The single-penalty smooth, by contrast, IS the exact scan's shape: it has
    // only the bending penalty and a genuinely unpenalized polynomial null space,
    // which the scan integrates exactly in O(n). It stays on the scan.
    let single_penalty =
        fit_spline_scan_from_formula("y ~ s(x, double_penalty=false)", &data, &cfg)
            .expect("materialize");
    assert!(
        single_penalty.is_some(),
        "single-penalty s(x) must route through the exact scan"
    );

    // Extra parametric term ⇒ not the single-smooth problem.
    let with_linear =
        fit_spline_scan_from_formula("y ~ x + s(x, double_penalty=false)", &data, &cfg)
            .expect("materialize");
    assert!(
        with_linear.is_none(),
        "smooth + linear term must fall through to the dense path"
    );

    // Shape-constrained smooth ⇒ not an unconstrained Gaussian posterior.
    let shaped = fit_spline_scan_from_formula(
        "y ~ s(x, double_penalty=false, shape=increasing)",
        &data,
        &cfg,
    )
    .expect("materialize");
    assert!(
        shaped.is_none(),
        "shape-constrained smooth must fall through to the dense path"
    );
}
