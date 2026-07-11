//! #1026 — the discriminating EV-vs-K measurement + the hybrid-dominance claim,
//! as a fast regression test over the crate's PUBLIC API.
//!
//! Two things the issue asks for, pinned here at RAM-safe scale (tens of rows,
//! K <= 8, single-digit inner iterations — runs in a few seconds / a few MB):
//!
//!  1. THE FRONTIER (wanted regardless). The collapsed-linear large-K lane
//!     ([`fit_sparse_dictionary`]) is the "large linear SAE" this issue is about.
//!     Its reconstruction EV must CLIMB with K (more atoms = more captured
//!     variance) and must USE its whole budget — every atom load-bearing at a K
//!     the data can support. The `revived == 0` convergence gate (#1026, this
//!     PR, `sparse_dict::update`) is what makes the tail load-bearing before the
//!     trainer is allowed to stop; this test pins the observable it guarantees.
//!
//!  2. THE DOMINANCE CLAIM (high-confidence half of the issue). A curved atom
//!     STRICTLY GENERALIZES a linear one (Euclidean d=1 with a linear basis is a
//!     single decoder direction), so on genuinely curved activation geometry it
//!     must match-or-beat — here, decisively beat — a linear dictionary at the
//!     SAME K and the SAME active budget. Pinned at K=1 (one curved atom vs one
//!     linear atom on a planted circle), the regime the issue documents (hue
//!     circles / position waves) and the byte-for-byte-stable single-atom path.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, fast_atb};
use gam_sae::assignment::{
    AssignmentMode, SaeAssignment, default_ordered_beta_bernoulli_concentration_for_k_atoms,
};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{
    SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, sae_pca_seed_initial_coords,
};
use gam_sae::sparse_dict::{SparseDictConfig, SparseDictFit, fit_sparse_dictionary};
use gam_terms::dictionary::{LinearDictionaryConfig, fit_linear_dictionary};
use gam_terms::latent::LatentManifold;
use ndarray::{Array2, ArrayView2, array, s};
use std::sync::Arc;

/// Deterministic standard-normal stream (splitmix64 + Box–Muller); no `rand`
/// dependency, identical run to run and across thread counts.
fn normal_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    move || {
        let u1 = ((next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let u2 = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Reconstruction explained variance `1 − RSS/TSS` (column-centred TSS), the same
/// definition the two lanes report internally — recomputed here so the curved arm
/// (whose internal EV helper is crate-private) is scored on identical footing.
fn explained_variance(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let n = target.nrows();
    let p = target.ncols();
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            means[c] += target[[i, c]];
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for i in 0..n {
        for c in 0..p {
            let r = target[[i, c]] - fitted[[i, c]];
            rss += r * r;
            let t = target[[i, c]] - means[c];
            tss += t * t;
        }
    }
    if tss <= 1.0e-24 {
        if rss <= 1.0e-24 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

/// Anisotropic Gaussian with a power-law PCA spectrum (per-component std ~
/// `1/(r+1)`): a residual-stream-like matrix whose leading `rank` directions each
/// carry real, non-negligible variance, so EV can genuinely keep climbing until
/// K covers the spectrum.
fn anisotropic_activations(n: usize, p: usize, rank: usize, seed: u64) -> Array2<f64> {
    let mut normal = normal_stream(seed);
    let v = Array2::from_shape_fn((p, rank), |_| normal());
    let scores = Array2::from_shape_fn((n, rank), |(_, r)| normal() / ((r + 1) as f64));
    let mut z = scores.dot(&v.t());
    for e in z.iter_mut() {
        *e += 0.01 * normal();
    }
    z
}

/// Number of atoms that fire (non-zero code) for at least one row.
fn alive_atom_count(fit: &SparseDictFit, k: usize) -> usize {
    let mut alive = vec![false; k];
    for i in 0..fit.indices.nrows() {
        for j in 0..fit.active {
            if fit.codes[[i, j]] != 0.0 {
                alive[fit.indices[[i, j]] as usize] = true;
            }
        }
    }
    alive.iter().filter(|&&a| a).count()
}

/// (1) EV-vs-K frontier for the large-K linear lane: reconstruction EV must climb
/// with K and, at a K the data supports, every atom must be load-bearing.
#[test]
fn linear_lane_ev_climbs_with_k_and_uses_full_budget_1026() {
    let rank = 8usize;
    let z64 = anisotropic_activations(120, 12, rank, 7);
    let z32 = z64.mapv(|v| v as f32);

    let ks = [1usize, 2, 4, 8];
    let mut curve = Vec::with_capacity(ks.len());
    for &k in &ks {
        let cfg = SparseDictConfig {
            n_atoms: k,
            active: 1,
            minibatch: 64,
            max_epochs: 40,
            score_tile: 16,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-8,
            score_mode: gam_gpu::GpuPolicy::Off,
        };
        let fit = fit_sparse_dictionary(z32.view(), &cfg).expect("linear lane fit runs");
        println!(
            "#1026 linear-lane K={k}: EV={:.4}  alive={}/{k}  epochs={}",
            fit.explained_variance,
            alive_atom_count(&fit, k),
            fit.epochs
        );
        curve.push(fit);
    }

    // Monotone capacity: adding atoms cannot lose EV (tiny f32 slack only).
    for w in curve.windows(2) {
        assert!(
            w[1].explained_variance + 5.0e-3 >= w[0].explained_variance,
            "EV must be non-decreasing in K (got {:.4} -> {:.4})",
            w[0].explained_variance,
            w[1].explained_variance
        );
    }
    // Material climb: the K=8 dictionary must capture substantially more of the
    // rank-8 spectrum than a single atom.
    let ev_first = curve.first().unwrap().explained_variance;
    let ev_last = curve.last().unwrap().explained_variance;
    assert!(
        ev_last - ev_first > 0.15,
        "EV must climb materially over K=1..8 on rank-8 data (K=1 {ev_first:.4} -> K=8 {ev_last:.4})"
    );
    // Full-budget utilization at K = rank: the `revived == 0` convergence gate
    // must leave EVERY atom load-bearing, not stop with a still-dead tail.
    let last = curve.last().unwrap();
    let alive = alive_atom_count(last, *ks.last().unwrap());
    assert_eq!(
        alive,
        *ks.last().unwrap(),
        "at K=8 on rank-8 data every atom must be load-bearing (alive={alive}/8, EV={:.4})",
        last.explained_variance
    );
}

/// A planted circle embedded in a 2-plane of R^p (`r·(cosθ·u + sinθ·v)` for
/// orthonormal `u,v`), plus tiny isotropic noise. Its point cloud has two equal-
/// variance directions, so the best rank-1 (single linear atom) reconstruction
/// caps near EV = 0.5, while one curved (periodic) atom can trace the whole ring.
fn planted_circle(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut normal = normal_stream(seed);
    let mut u = vec![0.0f64; p];
    let mut v = vec![0.0f64; p];
    for c in 0..p {
        u[c] = normal();
        v[c] = normal();
    }
    // Gram–Schmidt to orthonormal (u, v).
    let un: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in u.iter_mut() {
        *x /= un;
    }
    let uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    for c in 0..p {
        v[c] -= uv * u[c];
    }
    let vn: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v.iter_mut() {
        *x /= vn;
    }
    Array2::from_shape_fn((n, p), |(i, c)| {
        let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
        theta.cos() * u[c] + theta.sin() * v[c] + 0.01 * normal()
    })
}

/// Single-curved-atom reconstruction EV of a planted circle: PCA-seed a d=1
/// periodic atom, ridge-LSQ its decoder, then run the inner joint fit — the
/// production cold path at K=1 (the byte-for-byte-stable single-atom regime).
fn curved_single_atom_ev(z: ArrayView2<'_, f64>, num_basis: usize) -> f64 {
    let n = z.nrows();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(num_basis).unwrap());
    let basis_kinds = [SaeAtomBasisKind::Periodic];
    let atom_dims = [1usize];
    let seed_coords = sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims).unwrap();
    let coords = seed_coords.slice(s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let mut xtx = fast_atb(&phi, &phi);
    for i in 0..m {
        xtx[[i, i]] += 1.0e-8;
    }
    let xtz = fast_atb(&phi, &z.to_owned());
    let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ordered_beta_bernoulli(
            1.0,
            default_ordered_beta_bernoulli_concentration_for_k_atoms(1),
            false,
        ),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let mut rho = SaeManifoldRho::new(
        1.0e-3_f64.ln(),
        1.0e-3_f64.ln(),
        vec![array![1.0e-3_f64.ln()]],
    );
    term.run_joint_fit_arrow_schur(z, &mut rho, None, 12, 1.0, 1.0e-6, 1.0e-6)
        .expect("curved K=1 inner fit runs");
    let fitted = term.try_fitted().expect("curved fitted");
    explained_variance(z, fitted.view())
}

/// Single-linear-atom (one decoder direction, matched active budget) EV.
fn linear_single_atom_ev(z: ArrayView2<'_, f64>) -> f64 {
    let cfg = LinearDictionaryConfig {
        n_atoms: 1,
        top_k: 1,
        max_iter: 30,
        ..LinearDictionaryConfig::default()
    };
    fit_linear_dictionary(z, &cfg).unwrap().explained_variance
}

/// (2) Hybrid-dominance claim at matched K=1: on a planted circle one curved atom
/// must match-or-beat — here decisively beat — one linear atom.
#[test]
fn curved_atom_beats_linear_atom_on_circle_at_matched_k_1026() {
    let z = planted_circle(80, 6, 11);
    let ev_lin = linear_single_atom_ev(z.view());
    let ev_curved = curved_single_atom_ev(z.view(), 4);
    println!("#1026 K=1 circle: linear EV={ev_lin:.4}  curved EV={ev_curved:.4}");

    // A single linear direction cannot exceed the rank-1 cap on a full ring
    // (~0.5): this guards that the comparison really is on curved geometry.
    assert!(
        ev_lin < 0.7,
        "single linear atom must be capped well below full reconstruction on a circle (got {ev_lin:.4})"
    );
    // Match-or-beat (the strict-generalization contract) ...
    assert!(
        ev_curved + 1.0e-3 >= ev_lin,
        "curved atom must match-or-beat the linear atom at matched K (curved {ev_curved:.4} vs linear {ev_lin:.4})"
    );
    // ... and decisively beat on genuinely curved structure.
    assert!(
        ev_curved > ev_lin + 0.2,
        "one curved atom must decisively out-reconstruct one linear atom on a circle (curved {ev_curved:.4} vs linear {ev_lin:.4})"
    );
}
