//! [#1470-DIAG] Empirical probe — NOT a gate. Surfaces the final selected
//! smoothing parameters, per-block EDF, and per-penalty influence trace for
//! `y ~ s(x) + s(z) + ti(x, z)` on an exact tensor grid vs the SAME grid plus a
//! 1e-5 coordinate jitter, so the off-grid collapse is visible as a direct
//! GRID-vs-JITTER diff. The symptom (#1470) is that the s(x)/s(z) marginal
//! lambdas rail (1e3..1e13) and total EDF collapses off-grid. Run with
//! `--nocapture`. Ban-compliant: no env::var, no `{:?}` debug formatting.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

fn truth(x: f64, z: f64) -> f64 {
    (2.0 * PI * x).sin() + (2.0 * PI * z).cos() + (2.0 * PI * x).sin() * (2.0 * PI * z).sin()
}

/// Fit the model on `(x, z)`, then build and print a plain-text report of the
/// selected lambdas / EDF / traces (every value formatted explicitly — no
/// debug formatting). Returns `(rmse_to_truth, edf_total)` for the caller.
fn probe(label: &str, x: &[f64], z: &[f64]) -> (f64, f64) {
    let n = x.len();
    let y: Vec<f64> = (0..n).map(|i| truth(x[i], z[i])).collect();

    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| StringRecord::from(vec![x[r].to_string(), z[r].to_string(), y[r].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode diag-1470 dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x) + s(z) + ti(x, z)", &ds, &cfg).expect("diag-1470 fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x)+s(z)+ti(x,z)");
    };

    // Fitted values at the training points (identity link => design*beta).
    let mut din = Array2::<f64>::zeros((n, ds.headers.len()));
    for r in 0..n {
        din[[r, x_idx]] = x[r];
        din[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(din.view(), &fit.resolvedspec)
        .expect("rebuild diag-1470 design at training points");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let rmse_truth = rmse(&fitted, &y);

    let inf = fit.fit.inference.as_ref().expect("inference present");
    let edf_total = inf.edf_total;
    let edf_by_block = inf.edf_by_block.clone();
    let block_trace = inf.penalty_block_trace.clone();
    let lambdas: Vec<f64> = fit.fit.lambdas.to_vec();
    let log_lambdas: Vec<f64> = fit.fit.log_lambdas.to_vec();

    // Per-penalty inventory, labelled by coefficient range so a reader can map
    // each lambda to s(x) / s(z) / ti (penalties are in penalty order, aligned
    // 1:1 with lambdas / penalty_block_trace).
    let mut report = String::new();
    report.push_str("\n==================== #1470 LAMBDA PROBE [");
    report.push_str(label);
    report.push_str("] ====================\n");
    report.push_str(&format!(
        "n={n}  rmse_to_truth={rmse_truth:.6}  edf_total={edf_total:.4}  num_penalties={}\n",
        fit.design.penalties.len()
    ));
    for (i, bp) in fit.design.penalties.iter().enumerate() {
        let (r, c) = bp.local.dim();
        let lam = lambdas.get(i).copied().unwrap_or(f64::NAN);
        let rho = log_lambdas.get(i).copied().unwrap_or(f64::NAN);
        let tr = block_trace.get(i).copied().unwrap_or(f64::NAN);
        let edf_b = edf_by_block.get(i).copied().unwrap_or(f64::NAN);
        // col_range printed as explicit start..end (no debug formatting).
        report.push_str(&format!(
            "  penalty[{i}] cols=[{}..{}] dim={r}x{c}  lambda={lam:.6e}  rho={rho:.4}  trace={tr:.4}  edf_block={edf_b:.4}\n",
            bp.col_range.start, bp.col_range.end
        ));
    }
    // Flat lambda list, each value explicit (helps eyeball which margins rail).
    report.push_str("  lambdas:");
    for lam in &lambdas {
        report.push_str(&format!(" {lam:.6e}"));
    }
    report.push('\n');
    report.push_str("  edf_by_block:");
    for e in &edf_by_block {
        report.push_str(&format!(" {e:.4}"));
    }
    report.push('\n');

    eprintln!("{}", report);
    (rmse_truth, edf_total)
}

#[test]
fn diag_1470_grid_vs_jitter_lambda_probe() {
    init_parallelism();

    // ---- exact m x m tensor grid (interior points avoid duplicate knots) ----
    let m = 25usize;
    let mut gx: Vec<f64> = Vec::with_capacity(m * m);
    let mut gz: Vec<f64> = Vec::with_capacity(m * m);
    for i in 0..m {
        for j in 0..m {
            gx.push((i as f64 + 0.5) / m as f64);
            gz.push((j as f64 + 0.5) / m as f64);
        }
    }

    // ---- same grid + deterministic 1e-5 jitter (off-grid trigger) ----------
    const JITTER: f64 = 1e-5;
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut signed_unit = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut w = state;
        w = (w ^ (w >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        w = (w ^ (w >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        w ^= w >> 31;
        2.0 * ((w >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    let jx: Vec<f64> = gx.iter().map(|&v| v + JITTER * signed_unit()).collect();
    let jz: Vec<f64> = gz.iter().map(|&v| v + JITTER * signed_unit()).collect();

    let (rmse_grid, edf_grid) = probe("GRID", &gx, &gz);
    let (rmse_jitter, edf_jitter) = probe("JITTER", &jx, &jz);

    // Plain summary line (explicit formatting only).
    eprintln!(
        "[1470-DIAG] SUMMARY  grid: rmse={:.6} edf={:.4}   jitter: rmse={:.6} edf={:.4}",
        rmse_grid, edf_grid, rmse_jitter, edf_jitter
    );

    // This is a PROBE, not a gate: keep it green so it always prints. The only
    // assertion is that both fits actually produced finite numbers to compare.
    assert!(
        rmse_grid.is_finite() && rmse_jitter.is_finite(),
        "diag-1470 probe failed to produce finite RMSE (grid {rmse_grid}, jitter {rmse_jitter})"
    );
}
