//! Regression (different angle on #1126): the formula/FFI fit path for a 1-D
//! measure-jet smooth `s(x, bs="mjs")` must not merely *avoid aborting* when the
//! joint spatial-κ optimizer fails to converge at the tight outer tolerance — it
//! must fall back to a geometry that is *genuinely usable*, recovering the
//! underlying signal.
//!
//! The committed bug-hunt test
//! (`bug_hunt_measure_jet_formula_fit_aborts_at_tight_outer_tol`) asserts the fit
//! returns a finite result with a sane effective dof. That guards the abort, but
//! a regression could in principle satisfy it with a degenerate baseline (wrong
//! λ, a near-flat line). This test closes that gap from the prediction side:
//! after the κ optimizer hits its iteration cap at the formula path's
//! `tol = 1e-10`, the graceful fallback to the frozen baseline geometry
//! (`fit_frozen_baseline_geometry`) must still reconstruct `sin(2πx)` to a small
//! RMSE — far below the ~0.71 a flat fit would incur.
//!
//! It uses a *different* deterministic dataset (a phase-shifted sine on a
//! jittered grid, distinct from the bug-hunt repro's regular grid) so the
//! coverage is genuinely independent and not tied to one input layout — the
//! original report notes the Python path failed on ~60% of random datasets while
//! the CLI fit every one.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N: usize = 220;

/// SplitMix64 finalizer mapped to [0, 1): deterministic, RNG-free per-index
/// pseudo-noise so the dataset is bit-reproducible across machines.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// True (noise-free) signal: a phase-shifted single-cycle sine on [0, 1].
fn signal(x: f64) -> f64 {
    (std::f64::consts::TAU * x + 0.7).sin()
}

/// `y = signal(x) + 0.08·noise` on a *jittered* grid in [0, 1] — deliberately
/// not the regular grid the bug-hunt repro uses, so this is an independent input
/// geometry. `x` is sorted to keep it a clean 1-D scatter.
fn jittered_sine_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut xs: Vec<f64> = (0..N)
        .map(|i| {
            let base = i as f64 / (N as f64 - 1.0);
            // jitter within the local grid spacing, deterministic per index
            let jitter = (hashed_unit(i as u64 ^ 0xABCD) - 0.5) / (N as f64 - 1.0);
            (base + jitter).clamp(0.0, 1.0)
        })
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rows = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let noise = 2.0 * hashed_unit((i as u64).wrapping_mul(2_654_435_761)) - 1.0;
            let y = signal(x) + 0.08 * noise;
            StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode jittered sine dataset")
}

#[test]
fn measure_jet_formula_fit_recovers_signal_after_kappa_nonconvergence() {
    init_parallelism();
    let data = jittered_sine_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    // The formula path runs the joint spatial-κ optimizer at the tightened
    // outer tolerance (tol = 1e-10, the #893 replication-invariance tolerance).
    // Before #1126 a non-converged κ run aborted this fit; it must now degrade
    // to the frozen baseline geometry and return a usable model.
    let result = fit_from_formula("y ~ s(x, bs=\"mjs\")", &data, &config)
        .expect("measure-jet formula fit must succeed via graceful κ fallback (#1126)");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };

    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "fitted coefficients must be finite"
    );

    // Reconstruct the fitted curve on a held-out fine grid and compare to the
    // *noise-free* truth. A genuine smooth recovers the sine; a degenerate
    // (flat / mis-penalized) fallback would sit near the mean with RMSE ~0.71.
    let grid: Vec<f64> = (0..400).map(|i| 0.002 + 0.996 * i as f64 / 399.0).collect();
    let mut m = Array2::<f64>::zeros((grid.len(), 2));
    for (i, &t) in grid.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild measure-jet design on the held-out grid");
    let preds = design.design.apply(&fit.fit.beta).to_vec();

    let mut sse = 0.0;
    for (&t, &p) in grid.iter().zip(preds.iter()) {
        let e = p - signal(t);
        sse += e * e;
    }
    let rmse = (sse / grid.len() as f64).sqrt();

    // Budget 0.15: an order of magnitude below the ~0.71 of a flat fit and the
    // ~1.0 signal amplitude, but generous enough that the un-optimized baseline
    // κ (the fallback geometry, not the REML-tuned optimum) clears it.
    assert!(
        rmse < 0.15,
        "measure-jet fallback fit must recover the sine signal; RMSE={rmse:.4} \
         (a flat/degenerate fit would be ~0.71)"
    );
}
