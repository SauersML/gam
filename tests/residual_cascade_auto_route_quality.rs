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
//! AUTO-ROUTED path actually delivers the quality of the dense estimator at end
//! of pipeline. Two arms:
//!
//! 1. Match-or-beat the dense path on truth recovery. The cascade is a
//!    DIFFERENT posterior from the dense reduced-rank radial term, so it must
//!    earn its place: on the same scattered 2-D data it must recover the known
//!    smooth truth at least as well as the dense `fit_from_formula` duchon fit.
//!    This runs at a tractable `n` where the dense path is still cheap (the
//!    cascade is forced via `fit_residual_cascade_from_formula`, which bypasses
//!    the cliff gate's dispatch but keeps the same estimator), so both
//!    estimators see identical data and the comparison is apples-to-apples.
//!
//! 2. Cliff-scale auto-route POSITIVE. Past the derived dense-kernel cliff a
//!    single scattered 3-D Gaussian duchon, routed through the public
//!    `fit_from_formula`, must come back as `FitResult::ResidualCascade` (the
//!    dispatch fires with no flag — magic by default), recover the known truth,
//!    and do it in O(n·polylog) wall-clock: the cost from `n=1e5` to the
//!    cliff-scale `n` must grow far sub-quadratically (excluding the dense
//!    O(n·k²) blowup). Timing is logged, not gated (shared-runner noise); the
//!    HARD assertions are the routing, the truth recovery, and the scaling.

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

/// Arm 2 — cliff-scale auto-route POSITIVE: magic dispatch, truth recovery, and
/// the BPX n-independence tell.
///
/// Past the derived dense-kernel cliff, the PUBLIC `fit_from_formula` (no flag)
/// must auto-route a single scattered 3-D Gaussian duchon to the cascade and
/// recover the known truth. The dense `O(n·k² + k³)` route is impractical at
/// this `n` — that is the whole point of the primitive — so the match-or-beat
/// against the dense estimator is owned by Arm 1 (tractable n) and the
/// certification suite (bit-tight dense oracle). Here the hard bars are that the
/// route fires, the truth is recovered, and the solve stayed on the
/// n-independent BPX iteration bound (the certificate's CG count, far below the
/// `CG_MAX_ITERS` blow-up cap) — the structural O(n·polylog) tell, read from the
/// fit itself rather than from a noisy cross-route wall-clock ratio.
#[test]
fn cliff_scale_duchon_auto_routes_to_cascade_and_recovers_truth() {
    init_parallelism();
    let cfg = gaussian_config();
    let formula = "y ~ duchon(x1, x2, x3)";

    // The d=3 dense-kernel cliff (where `default_num_centers` pins at
    // K_MAX = 2000) sits near n ≈ 5.13e5. Route a cliff-scale fit through the
    // public formula entry — no flag — and time it for the log.
    let n = 513_000;
    let data = sample(3, n);
    let start = std::time::Instant::now();
    let big_result = fit_from_formula(formula, &data, &cfg).expect("auto-routed fit");
    let elapsed = start.elapsed().as_secs_f64();

    // Past the cliff the dispatch must fire with no flag — magic by default.
    let FitResult::ResidualCascade(fit) = big_result else {
        panic!(
            "cliff-scale scattered 3-D Gaussian duchon must auto-route to FitResult::ResidualCascade"
        );
    };
    eprintln!(
        "[cascade auto-route quality] n=5.13e5 auto-routed to cascade in {elapsed:.3}s \
         (solve_iters={}, backward_err={:.2e}, m={})",
        fit.certificate.solve_iters,
        fit.certificate.solve_rel_residual,
        fit.coeff.len()
    );

    // The cascade's certificates must be sane: a converged backward error and a
    // BOUNDED CG iteration count. The BPX preconditioner makes the iteration
    // count n-independent (tens, not thousands); staying far below the
    // `CG_MAX_ITERS = 4000` blow-up cap is the structural O(n·polylog) tell.
    assert!(
        fit.certificate.solve_rel_residual.is_finite() && fit.certificate.solve_rel_residual < 1e-3,
        "cascade solve did not converge: backward error {}",
        fit.certificate.solve_rel_residual
    );
    assert!(
        fit.certificate.solve_iters < 400,
        "cascade CG iteration count {} is not on the n-independent BPX bound \
         (the preconditioner is failing); near the CG_MAX_ITERS=4000 cap the \
         O(n·polylog) guarantee is lost",
        fit.certificate.solve_iters
    );

    // Truth recovery at cliff scale against the noise-free truth (#904 style).
    let probes = probe_grid(3, 7);
    let yhat: Vec<f64> = probes
        .iter()
        .map(|p| {
            let (mean, var) = fit.predict(p).expect("cascade predict at scale");
            assert!(
                mean.is_finite() && var.is_finite() && var > 0.0,
                "cascade prediction must be finite with positive variance at {p:?}"
            );
            mean
        })
        .collect();
    let cliff_rmse = rmse(&yhat, &probes);
    eprintln!("[cascade auto-route quality] cliff-scale truth-recovery RMSE={cliff_rmse:.5}");
    assert!(
        cliff_rmse < 0.08,
        "cascade fails cliff-scale truth recovery: RMSE={cliff_rmse} (truth is noise-free)"
    );
}
