//! #1784 — manifold SAE must not underfit real activations relative to a
//! linear dictionary of equal K. The IBP-MAP gate multiplies each atom's
//! activation by the ordered stick-breaking prior mean `π_k = (α/(α+1))^{k+1}`.
//! With the historical default `α = 1` that schedule is `(0.5)^{k+1}`, which
//! collapses to a near-hard mask past atom ~3: a K-atom dictionary can only ever
//! use its first handful of atoms, so it reconstructs far worse than a K-atom
//! linear dictionary — and its late atoms carry zero mass, leaving the per-row
//! joint Hessian rank-deficient (the K = 128 `RemlConvergenceError`).
//!
//! The fix scales the IBP concentration with the dictionary size
//! (`default_ibp_concentration_for_k_atoms`) so the prior SPANS the dictionary
//! (`π_{K-1} ≈ 1/e`) and every atom stays usable. These tests pin the invariant
//! the issue asks for: at equal K a curved dictionary reconstructs at least as
//! well as the linear one, and at K = 128 the prior no longer masks the tail
//! (the rank-deficiency that throws).
//!
//! Kept deliberately tiny (few rows / atoms / inner iterations, and a pure
//! arithmetic check for the K = 128 arm) so the module runs in seconds and in a
//! few MB under the RAM-tight shared build gate.

use super::*;
use crate::assignment::{default_ibp_concentration_for_k_atoms, ordered_geometric_shrinkage_prior};
use crate::basis::PeriodicHarmonicEvaluator;
use gam_linalg::faer_ndarray::{fast_atb, FaerCholesky};
use gam_terms::dictionary::{fit_linear_dictionary, LinearDictionaryConfig};
use ndarray::{s, Array2, ArrayView2};
use std::sync::Arc;

/// Deterministic "real-like" activation matrix: an anisotropic Gaussian with a
/// power-law PCA spectrum (no planted circle), the regime the issue reports the
/// manifold underfits on. Per-component std decays as `1/(r+1)` so early
/// principal directions dominate like a residual stream and every atom up to
/// `rank` carries real (non-negligible) variance.
fn real_like_activations(n: usize, p: usize, rank: usize, seed: u64) -> Array2<f64> {
    // splitmix64 PRNG → standard normal via Box–Muller; self-contained (no rand
    // dep) and identical run to run.
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
    let v = Array2::from_shape_fn((p, rank), |_| normal());
    let scores = Array2::from_shape_fn((n, rank), |(_, r)| normal() / ((r + 1) as f64));
    let mut z = scores.dot(&v.t());
    for e in z.iter_mut() {
        *e += 0.02 * normal();
    }
    z
}

/// Build a K-atom, d = 1 circle (`Periodic`) SAE term seeded from `z` the way the
/// production cold path does (PCA-seed the per-atom phase, ridge-LSQ the per-atom
/// decoder on the gated basis), with the IBP-MAP gate at concentration `alpha`.
fn circle_dictionary_term(
    z: ArrayView2<'_, f64>,
    k: usize,
    num_basis: usize,
    alpha: f64,
) -> SaeManifoldTerm {
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

/// In-sample reconstruction EV of a fixed-ρ inner joint fit of a K-atom circle
/// dictionary at IBP concentration `alpha`.
fn fit_ev(
    z: ArrayView2<'_, f64>,
    k: usize,
    alpha: f64,
    num_basis: usize,
    max_iter: usize,
) -> Result<f64, String> {
    let mut term = circle_dictionary_term(z, k, num_basis, alpha);
    let mut rho = SaeManifoldRho::new(
        1.0_f64.ln(),
        1.0_f64.ln(),
        vec![ndarray::array![1.0_f64.ln()]; k],
    );
    term.run_joint_fit_arrow_schur(z, &mut rho, None, max_iter, 1.0, 1.0e-6, 1.0e-6)?;
    let fitted = term.try_fitted_for_rho(&rho)?;
    reconstruction_explained_variance(z, fitted.view()).ok_or_else(|| {
        "reconstruction_explained_variance undefined (shape mismatch or degenerate total variance)"
            .to_string()
    })
}

fn linear_ev(z: ArrayView2<'_, f64>, k: usize) -> f64 {
    let cfg = LinearDictionaryConfig {
        n_atoms: k,
        top_k: 1,
        max_iter: 30,
        ..LinearDictionaryConfig::default()
    };
    fit_linear_dictionary(z, &cfg).unwrap().explained_variance
}

/// At equal K the historical default `α = 1` structurally underfits an equal-K
/// linear dictionary (the geometric-by-index prior masks late atoms), while the
/// K-aware concentration recovers the capacity and matches-or-beats linear.
/// Tiny (N=64, K=8) so it runs in seconds / a few MB.
#[test]
fn ibp_default_alpha_underfits_but_k_aware_matches_linear_1784() {
    let z = real_like_activations(64, 10, 6, 7);
    let k = 8usize;
    let num_basis = 3usize;
    let max_iter = 8usize;

    let lin = linear_ev(z.view(), k);
    let ev_alpha1 = fit_ev(z.view(), k, 1.0, num_basis, max_iter).expect("alpha=1 fit runs");
    let ev_kaware = fit_ev(
        z.view(),
        k,
        default_ibp_concentration_for_k_atoms(k),
        num_basis,
        max_iter,
    )
    .expect("K-aware fit runs");

    eprintln!(
        "#1784 K={k}: linear EV={lin:.4}  manifold(alpha=1) EV={ev_alpha1:.4}  manifold(K-aware) EV={ev_kaware:.4}"
    );

    // Margins are calibrated to the deterministic effect at THIS (RAM-safe) scale.
    // At the original K=32 / N=600 scale the α=1 underfit and K-aware recovery each
    // cleared ~0.05 EV; shrinking to K=8 / N=64 / num_basis=3 (issue #1784, to keep
    // the K=128 sibling test off the OOM path) preserves the qualitative ordering
    // — α=1 (≈0.862) < linear (≈0.893) < K-aware (≈0.908) — but with smaller gaps
    // (underfit ≈0.031, recovery ≈0.046). The thresholds below sit at roughly half
    // the observed gap so the strict ordering is enforced with ~2× headroom without
    // pinning to fragile exact values. The fit is deterministic here (no RNG; the
    // parallel fold is bit-invariant per #1557), so the headroom guards only
    // toolchain drift, not run-to-run noise.

    // The historical default α=1 must UNDERFIT the linear dictionary.
    assert!(
        ev_alpha1 + 0.015 < lin,
        "alpha=1 IBP prior should structurally underfit the equal-K linear dictionary \
         (manifold {ev_alpha1:.4} vs linear {lin:.4}) at K={k}"
    );
    // The K-aware concentration must recover the capacity: manifold ≥ linear (a
    // curved atom is a strict generalization of a linear one, so at equal K it
    // must reconstruct at least as well). Small numerical slack only.
    assert!(
        ev_kaware + 0.02 >= lin,
        "K-aware IBP prior must reconstruct at least as well as the equal-K linear \
         dictionary (manifold {ev_kaware:.4} vs linear {lin:.4}) at K={k}"
    );
    // And the fix must strictly beat the broken default.
    assert!(
        ev_kaware > ev_alpha1 + 0.02,
        "K-aware concentration must recover capacity the alpha=1 mask threw away \
         (K-aware {ev_kaware:.4} vs alpha=1 {ev_alpha1:.4})"
    );
}

/// The K = 128 `RemlConvergenceError` root cause, pinned WITHOUT running a fit
/// (which at K = 128 would allocate GBs on this RAM-tight box). The outer REML
/// throws because the geometric-by-index prior zeroes the atom tail — a masked
/// atom carries no gate mass, so the per-row joint Hessian is rank-deficient and
/// the criterion refuses to rank an off-optimum Laplace value. The fix is exactly
/// that the K-aware concentration keeps EVERY one of the 128 atoms' gate priors
/// alive (`π_{127} ≈ 1/e`), so no atom is structurally masked and the joint solve
/// stays well-posed. Pure arithmetic: instant, no allocation beyond a length-128
/// vector.
#[test]
fn ibp_k_aware_prior_keeps_all_128_atoms_alive_1784() {
    let k = 128usize;

    // Historical default α = 1: the tail is masked to ~0 (2.9e-39 for atom 127),
    // the rank-deficiency that makes the K = 128 fit throw RemlConvergenceError.
    let prior_alpha1 = ordered_geometric_shrinkage_prior(k, 1.0);
    assert!(
        prior_alpha1[k - 1] < 1.0e-30,
        "alpha=1 must mask the last atom (pi_127={:e}) — the dead-atom rank deficiency \
         behind the K=128 throw",
        prior_alpha1[k - 1]
    );

    // K-aware concentration: the prior SPANS the dictionary, so the last atom keeps
    // prior mass ≈ 1/e and every atom stays usable / the joint solve is well-posed.
    let alpha = default_ibp_concentration_for_k_atoms(k);
    let prior = ordered_geometric_shrinkage_prior(k, alpha);
    assert!(
        prior[k - 1] > 0.3,
        "K-aware prior must keep the last of {k} atoms alive (pi_127={:.4}, alpha={alpha:.2})",
        prior[k - 1]
    );
    // Monotone, but the whole-dictionary span is bounded by ≈ e — no structural
    // mask (α=1 spans 2.9e39 head-to-tail; the fix spans < 3).
    assert!(
        prior[0] / prior[k - 1] < 3.0,
        "K-aware prior head/tail span must be <= ~e (got {:.3})",
        prior[0] / prior[k - 1]
    );
}
