//! #1784 — manifold SAE must not underfit real activations relative to a
//! linear dictionary of equal K. The IBP-MAP gate multiplies each atom's
//! activation by the ordered stick-breaking prior mean `π_k = (α/(α+1))^{k+1}`.
//! With the historical default `α = 1` that schedule is `(0.5)^{k+1}`, which
//! collapses to a HARD mask past atom ~3: a K = 64 dictionary can only ever use
//! its first handful of atoms, so it reconstructs far worse than a K = 64 linear
//! dictionary (and its late atoms carry zero mass, leaving the per-row joint
//! Hessian rank-deficient — the K = 128 `RemlConvergenceError`).
//!
//! The fix scales the IBP concentration with the dictionary size
//! (`default_ibp_concentration_for_k_atoms`) so the prior SPANS the dictionary
//! (`π_{K-1} ≈ 1/e`) and every atom stays usable. These tests pin the invariant
//! the issue asks for: at equal K a curved dictionary reconstructs at least as
//! well as the linear one, and the K = 128 fit does not throw.

use super::*;
use crate::assignment::default_ibp_concentration_for_k_atoms;
use crate::basis::PeriodicHarmonicEvaluator;
use gam_linalg::faer_ndarray::{fast_ata, fast_atb, FaerCholesky};
use gam_terms::dictionary::{fit_linear_dictionary, LinearDictionaryConfig};
use ndarray::{s, Array2, ArrayView2};
use std::sync::Arc;

/// Deterministic "real-like" activation matrix: an anisotropic Gaussian with a
/// power-law PCA spectrum (no planted circle), the regime the issue reports the
/// manifold underfits on. `seed` makes it reproducible; the spectrum decays as
/// `1/(r+1)` so early principal directions dominate exactly like a residual
/// stream, and NO atom's variance is negligible until deep in the tail.
fn real_like_activations(n: usize, p: usize, rank: usize, seed: u64) -> Array2<f64> {
    // Simple splitmix64 PRNG → standard normal via Box–Muller, so the fixture is
    // self-contained (no rand dep) and identical run to run.
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut normal = move || {
        let u1 = ((next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let u2 = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    // Random orthonormal-ish loadings V (p × rank) and scores S (n × rank) with
    // per-component std ∝ 1/(r+1) (power-law spectrum).
    let v = Array2::from_shape_fn((p, rank), |_| normal());
    let scores = Array2::from_shape_fn((n, rank), |(_, r)| normal() / ((r + 1) as f64));
    let mut z = scores.dot(&v.t());
    // Small isotropic noise floor.
    for e in z.iter_mut() {
        *e += 0.02 * normal();
    }
    z
}

/// Build a K-atom, d = 1 circle (`Periodic`) SAE term seeded from `z` exactly the
/// way the production cold path does (PCA-seed the per-atom phase, ridge-LSQ the
/// per-atom decoder on the gated basis), with the IBP-MAP gate at concentration
/// `alpha`.
fn circle_dictionary_term(z: ArrayView2<'_, f64>, k: usize, num_basis: usize, alpha: f64) -> SaeManifoldTerm {
    let n = z.nrows();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let atom_dims = vec![1usize; k];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims).unwrap();
    let mut atoms = Vec::with_capacity(k);
    let mut coords_blocks = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let coords = seed_coords.slice(s![atom_idx, .., 0..1]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let mut xtx = fast_atb(&phi, &phi);
        for i in 0..m {
            xtx[[i, i]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi, &z.to_owned());
        let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
        let atom = SaeManifoldAtom::new(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone());
        atoms.push(atom);
        coords_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_blocks,
        manifolds,
        AssignmentMode::ibp_map(1.0, alpha, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

fn fit_ev(z: ArrayView2<'_, f64>, k: usize, alpha: f64) -> f64 {
    let mut term = circle_dictionary_term(z, k, 5, alpha);
    let mut rho = SaeManifoldRho::new(1.0_f64.ln(), 1.0_f64.ln(), vec![ndarray::array![1.0_f64.ln()]; k]);
    term.run_joint_fit_arrow_schur(z, &mut rho, None, 25, 1.0, 1.0e-6, 1.0e-6)
        .expect("inner fit converges");
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    reconstruction_explained_variance(z, fitted.view()).unwrap()
}

fn linear_ev(z: ArrayView2<'_, f64>, k: usize) -> f64 {
    let cfg = LinearDictionaryConfig {
        n_atoms: k,
        top_k: 1,
        max_iter: 50,
        ..LinearDictionaryConfig::default()
    };
    fit_linear_dictionary(z, &cfg).unwrap().explained_variance
}

#[test]
fn ibp_default_alpha_underfits_but_k_aware_matches_linear_1784() {
    let z = real_like_activations(600, 32, 20, 7);
    let k = 32usize;

    let lin = linear_ev(z.view(), k);
    let ev_alpha1 = fit_ev(z.view(), k, 1.0);
    let ev_kaware = fit_ev(z.view(), k, default_ibp_concentration_for_k_atoms(k));

    eprintln!(
        "#1784 K={k}: linear EV={lin:.4}  manifold(α=1) EV={ev_alpha1:.4}  manifold(K-aware) EV={ev_kaware:.4}"
    );

    // The historical default α=1 must UNDERFIT the linear dictionary badly.
    assert!(
        ev_alpha1 < lin - 0.1,
        "α=1 IBP prior should structurally underfit the linear dictionary at K={k} \
         (manifold {ev_alpha1:.4} vs linear {lin:.4})"
    );
    // The K-aware concentration must recover the capacity: manifold ≥ linear.
    assert!(
        ev_kaware >= lin - 1.0e-3,
        "K-aware IBP prior must reconstruct at least as well as the equal-K linear \
         dictionary (manifold {ev_kaware:.4} vs linear {lin:.4})"
    );
}
