//! #1556 regression: `SaeManifoldRho::log_lambda_smooth` is a genuinely
//! per-atom vector (length K). Kept in its own module so the catch-all
//! `tests.rs` stays under the #780 10k-line gate.

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;

/// #1556: `log_lambda_smooth` is a genuinely per-atom vector (length K), so
/// distinct atoms can carry distinct decoder-smoothness strengths, and the
/// `to_flat` → `from_flat` round-trip must preserve every per-atom entry. The
/// flat outer-coordinate vector must carry K smoothness coordinates (not 1).
#[test]
pub(crate) fn sae_manifold_lambda_smooth_is_per_atom_and_roundtrips_1556() {
    // K = 3 dictionary (per-atom ARD blocks fix the atom count), with three
    // DISTINCT per-atom smoothness strengths.
    let log_ard = vec![array![0.1], array![-0.3, 0.4], array![0.7]];
    let per_atom_smooth = vec![-2.0, 0.5, 1.25];
    let rho = SaeManifoldRho::with_per_atom_smooth(0.3, per_atom_smooth.clone(), log_ard);
    let k = rho.log_lambda_smooth.len();
    assert_eq!(k, 3, "K must equal the number of atoms");

    // Distinct atoms carry distinct λ_smooth values (the whole point of #1556).
    assert_ne!(
        rho.log_lambda_smooth[0], rho.log_lambda_smooth[1],
        "atoms 0 and 1 must be able to hold distinct λ_smooth"
    );
    assert_ne!(rho.log_lambda_smooth[1], rho.log_lambda_smooth[2]);

    // The flat vector reflects K smoothness coordinates: layout is
    // [sparse, <K smooth>, <ARD = 1 + 2 + 1 = 4>], so length = 1 + 3 + 4 = 8.
    let flat = rho.to_flat();
    let ard_len: usize = rho.log_ard.iter().map(|a| a.len()).sum();
    assert_eq!(ard_len, 4);
    assert_eq!(
        flat.len(),
        1 + k + ard_len,
        "flat vector must carry K (={k}) smoothness coordinates, not 1"
    );
    // The smoothness block sits at indices 1..1+K, in atom order.
    for (atom, &expected) in per_atom_smooth.iter().enumerate() {
        assert_abs_diff_eq!(flat[1 + atom], expected, epsilon = 0.0);
    }

    // Round-trip: from_flat reconstructs every per-atom smoothness entry exactly.
    let restored = rho.from_flat(flat.view()).unwrap();
    assert_eq!(restored.log_lambda_smooth.len(), k);
    for atom in 0..k {
        assert_abs_diff_eq!(
            restored.log_lambda_smooth[atom],
            rho.log_lambda_smooth[atom],
            epsilon = 0.0
        );
    }
    assert_abs_diff_eq!(
        restored.log_lambda_sparse,
        rho.log_lambda_sparse,
        epsilon = 0.0
    );
}
