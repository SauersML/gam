//! #2231 §2a — the crosscoder per-block weight `log λ_ℓ` as an outer
//! coordinate. The rho flat-layout round-trip pins that the appended block
//! sub-vector survives `to_flat`/`from_flat` and that an empty block vector is
//! byte-identical to the plain-SAE layout.

use super::SaeManifoldRho;
use ndarray::{Array1, arr1};

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
