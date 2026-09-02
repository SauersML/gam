//! Massive-K SAE encode — K-scaling throughput curve (issue #988).
//!
//! ## Why this test exists
//!
//! `encode_full_path_throughput` measures the certified encode against a SINGLE
//! atom. The user's target is the massive-K manifold SAE (K up to 32,000), where
//! the encode's dominant cost is per-row ATOM ROUTING over the whole dictionary.
//! The naive router (`route_exact`'s universal-bound certificate never fires for a
//! realistic dictionary, so it falls back to an O(K) full scan per row) makes the
//! whole encode O(N·K) — it blows up at K=32k.
//!
//! The production SPEED path (`amortized_encode_with_index_fast` /
//! `amortized_reconstruct_with_index_fast`) instead routes each row via the
//! sublinear LSH gather (`SaeCandidateIndex::propose`, which touches only
//! `~num_tables·bucket_occupancy = O(log K)` atoms and scores `~budget = 8·log2(K)`
//! candidates), so the routing — and therefore the whole fast encode→decode — is
//! sublinear in K. The per-atom encode is K-independent (each routed row only
//! touches its own atom's chart atlas).
//!
//! ## What it asserts
//!
//! It builds K circle atoms embedded into distinct subspaces of `R^p`, an LSH
//! index + sketch over them, and the certified [`EncodeAtlas`], then TIMES the
//! fast index-routed encode over a fixed batch of `N` rows at `K = 1024, 8192,
//! 32000` and reports rows/sec. The contract: throughput does NOT collapse
//! linearly with K — the K=32000 rate stays a large fraction of the K=1024 rate
//! (sublinear), which an O(N·K) router could never do (its rate would fall ~31×).

use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};

use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::candidate_index::{IndexConfig, RandomProjectionFrameSketch, SaeCandidateIndex};
use gam_sae::encode::{AtlasConfig, EncodeAtlas};
use gam_sae::manifold::{SaeAtomBasisKind, SaeManifoldAtom};

/// Deterministic orthonormal pair `(u_k, v_k)` in `R^p` for atom `k`: two
/// distinct directions so atom `k`'s decoded circle spans its OWN 2-plane, making
/// the atoms distinguishable to the router. Pseudo-random but reproducible.
fn atom_plane(k: usize, p: usize) -> (Array1<f64>, Array1<f64>) {
    let kf = k as f64;
    let mut u = Array1::from_shape_fn(p, |j| {
        ((0.7 * kf + 1.3 * j as f64 + 0.1).sin()) + 0.15 * ((kf * 0.031 + j as f64 * 0.017).cos())
    });
    let un = u.dot(&u).sqrt().max(1e-12);
    u.mapv_inplace(|x| x / un);
    let mut v = Array1::from_shape_fn(p, |j| {
        ((0.29 * kf + 0.91 * j as f64 + 0.4).cos()) + 0.11 * ((kf * 0.047 + j as f64 * 0.023).sin())
    });
    let proj = v.dot(&u);
    v = &v - &(&u * proj);
    let vn = v.dot(&v).sqrt().max(1e-12);
    v.mapv_inplace(|x| x / vn);
    (u, v)
}

/// `N` on-manifold target rows, each planted on a (round-robin) atom's circle at a
/// random phase with amplitude in `[0.8, 1.2]`.
fn planted_targets(atoms: &[SaeManifoldAtom], n: usize, p: usize) -> (Array2<f64>, Array1<f64>) {
    let k = atoms.len();
    let mut targets = Array2::<f64>::zeros((n, p));
    let mut amps = Array1::<f64>::ones(n);
    let evaluator = atoms[0].basis_evaluator.as_ref().unwrap().clone();
    for row in 0..n {
        let atom_k = (row * 2654435761) % k; // scattered across the dictionary
        let t = ((row as f64 * 0.6180339887) % 1.0).abs();
        let z = 0.8 + 0.4 * ((row as f64 * 0.123).sin() * 0.5 + 0.5);
        amps[row] = z;
        let coord = Array2::from_shape_fn((1, 1), |_| t);
        let (phi, _) = evaluator.evaluate(coord.view()).unwrap();
        let decoded = phi.dot(atoms[atom_k].decoder_coefficients()); // (1 × p), amp-1
        for c in 0..p {
            targets[[row, c]] = z * decoded[[0, c]];
        }
    }
    (targets, amps)
}

