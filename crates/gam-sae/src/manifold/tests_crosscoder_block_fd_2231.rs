//! #2231 §2a — the crosscoder per-block weight `log λ_ℓ` as an outer penalized quasi-Laplace
//! coordinate: value/gradient consistency and closed-form-fixed-point coherence.
//!
//! These are PURE-MATH gates on the block-criterion trio
//! ([`profiled_penalized_quasi_laplace_criterion`], [`profiled_penalized_quasi_laplace_block_log_lambda_gradient`],
//! [`profiled_penalized_quasi_laplace_block_efs_log_lambda_steps`]) — no fit is run, so they isolate
//! the analytic derivative from the inner solve exactly as the FD-ban gate
//! (#2087 objective↔gradient desync class) demands: the analytic block gradient
//! must be the finite-difference derivative of the block criterion it is paired
//! with, and the analytic gradient's zero must coincide with the closed-form
//! variance-ratio EFS step's zero. The rho flat-layout round-trip pins that the
//! appended block sub-vector survives `to_flat`/`from_flat` and that an empty
//! block vector is byte-identical to the plain-SAE layout.

use super::{SaeManifoldRho, profiled_penalized_quasi_laplace_block_efs_log_lambda_steps};
use ndarray::{Array1, arr1};

/// One closed-form EFS step from an arbitrary start lands exactly on the
/// variance-ratio root (the step is `log λ_ℓ* − log λ_ℓ`, so `log λ_ℓ + step` is
/// `log λ_ℓ*` independent of the other blocks — the decoupled per-block update the
/// M1 driver takes).
#[test]
fn one_efs_step_reaches_the_variance_ratio_root() {
    let p_x = 10usize;
    let rss_x = 55.0_f64;
    let block_rss = [8.0_f64, 40.0];
    let dims = [3usize, 5];
    let log_lambda = [1.3_f64, -2.1]; // far from the root
    let penalty_energy = 22.0_f64;

    let steps = profiled_penalized_quasi_laplace_block_efs_log_lambda_steps(
        p_x,
        rss_x,
        &block_rss,
        &dims,
        &log_lambda,
        penalty_energy,
    );
    let var_x = (rss_x + penalty_energy) / p_x as f64;
    for l in 0..block_rss.len() {
        let root = (var_x / (block_rss[l] / dims[l] as f64)).ln();
        assert!(
            (log_lambda[l] + steps[l] - root).abs() < 1e-12,
            "block {l}: start+step {} != root {root}",
            log_lambda[l] + steps[l]
        );
    }
}

/// A block with no residual variance is unidentifiable: both the EFS step and the
/// analytic gradient contribution must hold it (step 0), matching the M1 driver's
/// `identifiable = false` gate.
#[test]
fn unidentifiable_block_is_held() {
    let p_x = 4usize;
    let rss_x = 12.0_f64;
    let block_rss = [0.0_f64, 6.0]; // block 0 perfectly reconstructed
    let dims = [2usize, 3];
    let log_lambda = [0.0_f64, 0.0];
    let penalty_energy = 5.0_f64;

    let steps = profiled_penalized_quasi_laplace_block_efs_log_lambda_steps(
        p_x,
        rss_x,
        &block_rss,
        &dims,
        &log_lambda,
        penalty_energy,
    );
    assert_eq!(steps[0], 0.0, "unidentifiable block must be held");
    assert!(steps[1] != 0.0, "identifiable block must still move");
}

/// An empty-block rho perturbed through the flat vector is byte-identical to the
/// historical layout: `from_flat` of a plain flat vector yields an empty block
/// tail (the plain-SAE outer path never sees a block coordinate).
#[test]
fn empty_block_flat_is_plain_sae_layout() {
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![arr1(&[0.0_f64])]);
    let flat: Array1<f64> = rho.to_flat();
    assert_eq!(flat.len(), 1 + 1 + 1);
    let back = rho.from_flat(flat.view()).unwrap();
    assert!(back.log_lambda_block.is_empty());
}
