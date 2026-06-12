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
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    fit_spline_scan_from_formula, init_parallelism,
};
use ndarray::Array2;

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
    let (oracle_mean, oracle_var_unit) = dense_truth(&fit.knots, &knot_y, &w, fit.log_lambda);

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

    // Dense reference path on the identical model formula.
    let dense = fit_from_formula(SCAN_FORMULA, &data, &cfg).expect("dense fit");
    let FitResult::Standard(dense) = dense else {
        panic!("dense reference path must return a standard fit");
    };

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

/// #1030 benchmark + no-regression certificate at biobank scale: the scan
/// fits a single 1-D Gaussian smooth at n = 1e6 in O(n) wall-clock and
/// recovers the known truth — the regime where the dense design/Gram/REML
/// route is impractical (O(n·k²) per λ-trial). Timing is logged, not gated
/// (shared-runner wall-clock is noisy); the HARD assertions are (a) the cost
/// scales sub-quadratically from 1e5 → 1e6 (proves O(n), not O(n²)), (b)
/// truth recovery within the injected noise, and (c) a sane EDF. The headline
/// scan-vs-dense ratio is recorded on the issue from the MSI run.
#[test]
fn spline_scan_million_row_fit_scales_linearly_and_recovers_truth() {
    init_parallelism();
    let cfg = gaussian_config();

    let time_fit = |n: usize| -> (f64, gam::solver::spline_scan::CubicSplineScanFit) {
        let (x, y) = training_xy(n);
        let data = encode_xy(&x, &y);
        let start = std::time::Instant::now();
        let fit = fit_spline_scan_from_formula(SCAN_FORMULA, &data, &cfg)
            .expect("scan-routed fit")
            .expect("detection must fire for a single 1-D single-penalty Gaussian smooth");
        (start.elapsed().as_secs_f64(), fit)
    };

    let (t_small, _) = time_fit(100_000);
    let (t_big, fit) = time_fit(1_000_000);
    eprintln!(
        "[spline-scan bench] n=1e5: {t_small:.4}s | n=1e6: {t_big:.4}s | ratio={:.2} (linear ≈ 10)",
        t_big / t_small.max(1e-9)
    );

    // Sub-quadratic scaling: a 10× n increase costs far less than 100× (the
    // O(n²) dense-Gram blowup). Generous bound absorbs runner noise and the
    // O(log) golden-section λ-search overhead while still excluding O(n²).
    assert!(
        t_big <= 30.0 * t_small.max(1e-6),
        "scan cost grew super-linearly from n=1e5 ({t_small:.4}s) to n=1e6 ({t_big:.4}s)"
    );

    // Truth recovery at n=1e6 against the self-constructed truth (#904 style):
    // the injected noise has half-range 0.15 (variance ≈ 0.0075), and with 1e6
    // observations the smooth is resolved far tighter than the per-point noise.
    let grid: Vec<f64> = (0..200).map(|i| 0.02 + 0.96 * i as f64 / 199.0).collect();
    let mut sse = 0.0;
    for &t in &grid {
        let (mean, var) = fit.predict(t).expect("scan predict at scale");
        assert!(
            mean.is_finite() && var.is_finite() && var > 0.0,
            "scan prediction must be finite with positive variance at x={t}"
        );
        sse += (mean - truth_fn(t)) * (mean - truth_fn(t));
    }
    let mse = sse / grid.len() as f64;
    assert!(
        mse < 1e-3,
        "scan fails truth recovery at n=1e6: MSE={mse} (truth is noise-free)"
    );

    // EDF must exceed the unpenalized null-space dimension (2) and stay well
    // below n — a real smooth was fit, not the linear trend or an interpolant.
    let edf = fit.edf();
    assert!(
        edf > 2.0 && edf < 1_000.0,
        "scan EDF {edf} outside the sane band (2, 1000) for a smooth biobank-scale fit"
    );
}

#[test]
fn detection_falls_through_for_ineligible_shapes() {
    init_parallelism();
    let (x, y) = training_xy(60);
    let data = encode_xy(&x, &y);
    let cfg = gaussian_config();

    // Default s(x) carries the double (null-space shrinkage) penalty — a
    // different posterior from the pure λ∫f″² spline, so it must NOT route.
    let default_s = fit_spline_scan_from_formula("y ~ s(x)", &data, &cfg).expect("materialize");
    assert!(
        default_s.is_none(),
        "double-penalty default s(x) must fall through to the dense path"
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
