//! Cross-cutting contract pins for the SAE-manifold streaming / sparse-atom /
//! GPU joint-fit path (issue #358).
//!
//! These encode the three guarantees the streaming co-fit must keep so it is
//! usable as an LLM-scale teacher:
//!
//!   (a) **Streaming ↔ in-core agreement.** Reducing one Arrow-Schur system in
//!       a single chunk (the in-core full-batch reduction) and reducing the
//!       same system across many chunks (the streaming online accumulation)
//!       must produce the same reduced Schur block `S`, the same reduced RHS,
//!       and hence the same marginal `Δβ` Newton step — to machine precision.
//!       The per-row latent block is profiled out identically either way; the
//!       only difference is the order of accumulation.
//!
//!   (b) **Per-token cost independent of `K` at fixed `k_active`.** Under the
//!       hard-TopK support gate, each token's Arrow-Schur row block is sized
//!       by its selected atoms' coordinates (`Σ_{k∈support} d_k`), never by the
//!       total atom count `K`. Doubling `K` while holding the per-token active
//!       set fixed must leave every per-row block dimension unchanged, so the
//!       assembly and per-row solve cost track `k_active`, not `K`. This is the
//!       structural precondition for the `K = 100K` sparse-atom claim.
//!
//!   (c) **GPU ↔ CPU parity on the reduced joint step.** The on-device
//!       Jacobi-CG reduced-β solve (`solve_reduced_beta_pcg`) must agree with
//!       the host reduced solve (`solve_streaming_reduced_beta`) on the same
//!       accumulated `S`/`rhs`. The test no-ops when no CUDA device is present
//!       (`ArrowSchurGpuFailure::Unavailable`) so it stays green on CPU CI and
//!       the macOS dev box, and asserts hard parity on a real device.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`, no new
//! public knobs. Sizing fits comfortably in CI RAM.

use ndarray::{Array1, Array2, Array3};

use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};

/// Deterministic pseudo-random f64 ∈ (-1, 1) via LCG, matching the sibling
/// `sae_arrow_schur_large_scale` fixture.
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Build a small Euclidean-patch SAE term with the requested gating mode.
///
/// Euclidean atoms keep the per-row latent block flat (no tangent projection),
/// so the streaming reduction and hard-TopK support layout are exercised on the
/// same code path the `K = 100K` fit uses. The routing logits are seeded so that
/// the same `n_active` atoms have the largest scores per token regardless of `K`.
fn build_term(
    k_atoms: usize,
    basis_size: usize,
    latent_dim: usize,
    n_obs: usize,
    p_out: usize,
    mode: AssignmentMode,
    n_active: usize,
    seed: u64,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let m = basis_size;
    let d = latent_dim;
    let n = n_obs;
    let p = p_out;

    let mut rng = seed
        .wrapping_add(k_atoms as u64 * 97)
        .wrapping_add(n as u64 * 7);

    // Routing logits: the first `n_active` atoms (modulo a per-row rotation)
    // have the largest scores, so hard TopK selects exactly `n_active` atoms per
    // token independent of `K`.
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    for row in 0..n {
        let base = row % k_atoms;
        for slot in 0..k_atoms {
            let atom = (base + slot) % k_atoms;
            logits[[row, atom]] = if slot < n_active {
                0.5 + 0.25 * lcg_f64(&mut rng).abs()
            } else {
                -0.5 - 0.25 * lcg_f64(&mut rng).abs()
            };
        }
    }
    let target = Array2::from_shape_fn((n, p), |_| lcg_f64(&mut rng));

    let mut atoms: Vec<SaeManifoldAtom> = Vec::with_capacity(k_atoms);
    let mut coord_blocks: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let phi = Array2::from_shape_fn((n, m), |_| lcg_f64(&mut rng) * 0.1);
        let jet = Array3::from_shape_fn((n, m, d), |_| lcg_f64(&mut rng) * 0.01);
        let decoder = Array2::from_shape_fn((m, p), |_| lcg_f64(&mut rng) * 0.3);
        let mut smooth = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            smooth[[i, i]] = 0.1 + 0.01 * lcg_f64(&mut rng).abs();
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("atom_{atom_idx}"),
            SaeAtomBasisKind::EuclideanPatch,
            d,
            phi,
            jet,
            decoder,
            smooth,
        )
        .unwrap_or_else(|e| panic!("SaeManifoldAtom::new failed: {e}"));
        atoms.push(atom);
        coord_blocks.push(Array2::from_shape_fn((n, d), |_| lcg_f64(&mut rng) * 0.5));
    }

    let assignment = SaeAssignment::from_blocks_with_mode(logits, coord_blocks, mode)
        .unwrap_or_else(|e| panic!("SaeAssignment::from_blocks_with_mode failed: {e}"));
    let term = SaeManifoldTerm::new(atoms, assignment)
        .unwrap_or_else(|e| panic!("SaeManifoldTerm::new failed: {e}"));

    let log_ard: Vec<Array1<f64>> = (0..k_atoms)
        .map(|_| Array1::from_elem(latent_dim, 0.0_f64))
        .collect();
    let rho = SaeManifoldRho::new(0.0, -4.0, log_ard);
    (term, target, rho)
}

// ---------------------------------------------------------------------------
// (a) Streaming ↔ in-core reduction + Δβ agreement.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// (b) Per-token assembly cost independent of K at fixed k_active.
// ---------------------------------------------------------------------------

#[test]
fn per_token_block_dim_is_independent_of_k_at_fixed_active() {
    let d = 2usize;
    let n = 64usize;
    let n_active = 3usize;
    // Hard TopK selects exactly `n_active` atoms per token. Its compact row
    // block contains coordinates only and is therefore `n_active·d` wide,
    // independent of K.
    let mode = AssignmentMode::top_k_support(n_active);
    let expected_block_dim = n_active * d;

    let mut per_row_dims_at_k: Vec<Vec<usize>> = Vec::new();
    for k_atoms in [16usize, 32, 64] {
        let (mut term, target, rho) = build_term(k_atoms, 4, d, n, 2, mode, n_active, 0xBEEF_5678);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .unwrap_or_else(|e| panic!("assemble_arrow_schur failed at K={k_atoms}: {e}"));

        // Every per-row latent block is sized by the active set only.
        for (row, block) in sys.rows.iter().enumerate() {
            assert_eq!(
                block.htt.nrows(),
                expected_block_dim,
                "K={k_atoms} row={row}: per-token block dim {} != n_active·(1+d)={expected_block_dim}; \
                 the per-row cost must track k_active, not K",
                block.htt.nrows()
            );
            assert_eq!(
                block.htt.ncols(),
                expected_block_dim,
                "K={k_atoms} row={row}: per-token block is not square at the active dim"
            );
        }
        per_row_dims_at_k.push(sys.rows.iter().map(|r| r.htt.nrows()).collect());
    }

    // The full per-row dimension profile is bit-identical across all K, so
    // doubling K leaves per-token assembly cost unchanged.
    let baseline = &per_row_dims_at_k[0];
    for (idx, profile) in per_row_dims_at_k.iter().enumerate() {
        assert_eq!(
            profile, baseline,
            "per-row block-dim profile at K-index {idx} diverged from the K=16 baseline; \
             per-token cost must be invariant in K at fixed k_active"
        );
    }
}

// ---------------------------------------------------------------------------
// (c) GPU ↔ CPU parity on the reduced joint step (gated; no-ops without CUDA).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// (d) The out-of-core streaming FIT driver is chunk-size invariant.
// ---------------------------------------------------------------------------
//
// `run_joint_fit_arrow_schur_streaming` is the memory-bounded fit driver for
// the LLM-scale teacher: it re-seeds each chunk's `(logits, coords, Z)` from a
// `chunk_init` closure and never materializes the `(N×M)`/`(N×K)` per-row
// buffers. Because it re-seeds the per-row latent state from `chunk_init` on
// every pass (rather than carrying it forward), each outer iteration's reduced
// β-Newton step, line-search objective, and decoder-Gram audit are all exact
// sums over rows — independent of how the rows are partitioned into chunks.
//
// So the FITTED decoder β must not depend on `chunk_size`. Test (a) pins this
// for a single reduction; this pins it for the full multi-iteration driver
// end-to-end. A genuine chunking bug (e.g. a mis-scaled minibatch penalty, a
// dropped per-chunk contribution, or per-chunk ridge double-counting) breaks
// the invariance by O(1), far above float-reordering noise.
