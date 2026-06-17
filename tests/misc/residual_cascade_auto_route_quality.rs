//! End-to-end QUALITY gate for the O(n log n) multiresolution residual-cascade
//! auto-route (#1032).
//!
//! The cascade is the compute-first primitive for scattered low-d Gaussian
//! Duchon/Matérn smooths at huge `n`, where the dense radial kernel saturates
//! its center cap and can no longer densify with `n`. Its detection seam
//! (`residual_cascade_fast_path`) and its dispatch inside `fit_from_formula`
//! are already covered structurally by `residual_cascade_workflow_detection.rs`
//! (the NEGATIVES: every ineligible shape and the below-cliff duchon fall
//! through), and the cascade ESTIMATOR itself is certified bit-for-bit against
//! an in-test dense penalized oracle in `residual_cascade_certification.rs`.
//!
//! What those two suites do NOT assert is the thing this file owns: that the
//! cascade estimator actually delivers the quality of the dense estimator at a
//! tractable end-to-end size. The cliff-scale auto-route shape is covered
//! structurally in `residual_cascade_workflow_detection.rs`; this file does not
//! allocate a half-million-row formula dataset just to prove dispatch.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::residual_cascade::fit_residual_cascade;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Deterministic low-discrepancy scattered sample on the unit cube with a
/// smooth, noise-perturbed truth, as RAW columns: `dim` coordinate axes, the
/// response, and unit weights. `dim ∈ {2, 3}` selects the cascade's domain.
/// This is the single source of the data; [`sample`] just stringifies it into
/// an encoded dataset, so the dense-formula and direct-cascade arms see the
/// EXACT same observations.
fn coords_y(dim: usize, n: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let golden = 0.618_033_988_749_894_9_f64;
    let sqrt2 = std::f64::consts::SQRT_2.fract();
    let sqrt3 = std::f64::consts::SQRT_2.fract() * 0.5 + 0.371_f64;
    let mut axes = vec![Vec::with_capacity(n); dim];
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let coords = [
            ((i + 1) as f64 * golden).fract(),
            ((i + 1) as f64 * sqrt2).fract(),
            ((i + 7) as f64 * sqrt3).fract(),
        ];
        for (a, axis) in axes.iter_mut().enumerate() {
            axis.push(coords[a]);
        }
        let noise = (((i + 3) as f64 * golden).fract() - 0.5) * 0.1;
        y.push(truth(&coords[..dim]) + noise);
        w.push(1.0);
    }
    (axes, y, w)
}

/// The same scattered sample as [`coords_y`], stringified into an encoded
/// dataset for the formula entry points.
fn sample(dim: usize, n: usize) -> gam::data::EncodedDataset {
    let (axes, y, _w) = coords_y(dim, n);
    let mut headers: Vec<String> = (0..dim).map(|a| format!("x{}", a + 1)).collect();
    headers.push("y".to_string());
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let mut fields: Vec<String> = axes.iter().map(|axis| axis[i].to_string()).collect();
            fields.push(y[i].to_string());
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

/// Smooth, noise-free truth. Matches the certification suite's construction so
/// the recovery bound is meaningful: a separable sinusoid, lifted by a mild
/// linear modulation on the third axis in 3-D.
fn truth(p: &[f64]) -> f64 {
    let base =
        (2.0 * std::f64::consts::PI * p[0]).sin() * (2.0 * std::f64::consts::PI * p[1]).sin();
    match p.len() {
        2 => base,
        3 => base * (0.6 + 0.8 * p[2]),
        _ => unreachable!("truth: dim must be 2 or 3"),
    }
}

/// A deterministic grid of interior probe points (avoiding the boundary where
/// every scattered radial smooth is least resolved) for truth-recovery RMSE.
fn probe_grid(dim: usize, per_axis: usize) -> Vec<Vec<f64>> {
    let axis: Vec<f64> = (0..per_axis)
        .map(|i| 0.15 + 0.70 * i as f64 / (per_axis - 1) as f64)
        .collect();
    let mut grid = Vec::new();
    match dim {
        2 => {
            for &u in &axis {
                for &v in &axis {
                    grid.push(vec![u, v]);
                }
            }
        }
        3 => {
            for &u in &axis {
                for &v in &axis {
                    for &w in &axis {
                        grid.push(vec![u, v, w]);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    grid
}

fn gaussian_config() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// Truth-recovery root-mean-square error of the dense `fit_from_formula` duchon
/// fit, evaluated at `probes` by rebuilding the term design and applying the
/// fitted coefficients (the standard dense-prediction path).
fn dense_duchon_rmse(formula: &str, data: &gam::data::EncodedDataset, probes: &[Vec<f64>]) -> f64 {
    let cfg = gaussian_config();
    let result = fit_from_formula(formula, data, &cfg).expect("dense duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("below-cliff duchon must return a dense standard fit, not the cascade");
    };
    let dim = probes[0].len();
    let mut m = Array2::<f64>::zeros((probes.len(), dim));
    for (i, p) in probes.iter().enumerate() {
        for (a, &v) in p.iter().enumerate() {
            m[[i, a]] = v;
        }
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
    let yhat = design.design.apply(&fit.fit.beta).to_vec();
    rmse(&yhat, probes)
}

fn rmse(yhat: &[f64], probes: &[Vec<f64>]) -> f64 {
    let sse: f64 = yhat
        .iter()
        .zip(probes.iter())
        .map(|(&m, p)| {
            let e = m - truth(p);
            e * e
        })
        .sum();
    (sse / probes.len() as f64).sqrt()
}

/// Arm 1 — match-or-beat the dense estimator on truth recovery.
///
/// At a tractable `n` (where the dense reduced-rank radial path is still cheap)
/// the cascade ESTIMATOR — invoked directly via [`fit_residual_cascade`], the
/// same estimator the auto-route dispatches past the cliff, since the
/// formula-level entry refuses to materialize the cascade below the cliff —
/// must recover the same known smooth truth at least as well as the dense
/// `fit_from_formula` duchon fit on the SAME data. The cascade is a different
/// posterior, so this is the no-regression bar that earns it the right to stand
/// in past the cliff.
#[test]
fn cascade_matches_or_beats_dense_duchon_on_truth_recovery() {
    init_parallelism();
    // Kept modest: the dense `fit_from_formula` duchon REML over its ~K-center
    // Gram is the cost here (the cascade itself is cheap), and the comparison
    // only needs enough data to resolve the truth — the certification suite
    // already recovers it at n = 2500.
    let n = 2_000;
    let data = sample(2, n);
    let formula = "y ~ duchon(x1, x2)";
    let probes = probe_grid(2, 10);

    // Run the cascade estimator directly on the identical raw data at the native
    // 2-D Sobolev order s = (d+3)/2 = 2.5 (the order the fast path clamps a
    // default duchon into; same window the certification suite exercises). Unit
    // isotropic metric matches the auto-route's `ResidualCascadeInputs`.
    let (axes, y, w) = coords_y(2, n);
    let xs: Vec<&[f64]> = axes.iter().map(|a| a.as_slice()).collect();
    let cascade = fit_residual_cascade(&xs, &y, &w, &[1.0, 1.0], 2.5)
        .expect("cascade must fit an eligible scattered 2-D sample");
    let cascade_yhat: Vec<f64> = probes
        .iter()
        .map(|p| cascade.predict(p).expect("cascade predict").0)
        .collect();
    let cascade_rmse = rmse(&cascade_yhat, &probes);

    let dense_rmse = dense_duchon_rmse(formula, &data, &probes);

    eprintln!(
        "[cascade auto-route quality] n={n} cascade_rmse={cascade_rmse:.5} dense_rmse={dense_rmse:.5}"
    );

    // Absolute truth-recovery bound: the smooth is resolved far inside the
    // per-point noise (half-range 0.05) given thousands of observations.
    assert!(
        cascade_rmse < 0.05,
        "cascade fails absolute truth recovery: RMSE={cascade_rmse} (truth is noise-free)"
    );
    // Match-or-beat: a small multiplicative slack absorbs the different finite
    // bases (multilevel Wendland frame vs reduced-rank radial kernel) without
    // letting a genuine quality regression pass.
    assert!(
        cascade_rmse <= 1.10 * dense_rmse,
        "cascade truth recovery regressed vs dense duchon: \
         cascade RMSE={cascade_rmse} > 1.10 × dense RMSE={dense_rmse}"
    );
}
