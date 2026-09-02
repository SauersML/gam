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

use ndarray::{Array1, Array2, Array3, s};

use gam::solver::arrow_schur::{ArrowSchurSystem, ArrowSolveOptions, StreamingArrowSchur};
use gam::solver::gpu_kernels::arrow_schur::ArrowSchurGpuFailure;
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

#[test]
fn streaming_reduction_matches_in_core_single_chunk() {
    // K = 6, N = 240 so multiple chunk splittings are non-trivial.
    let (mut term, target, rho) = build_term(
        6,
        4,
        1,
        240,
        2,
        AssignmentMode::softmax(1.0),
        6,
        0xC0FFEE_1234,
    );
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));

    let options = ArrowSolveOptions::automatic(sys.k);
    let ridge_t = 1e-4;
    let ridge_beta = 1e-4;

    // In-core: one chunk covering all rows.
    let (s_full, rhs_full) = reduce_in_chunks(&sys, sys.rows.len(), ridge_t, ridge_beta, &options);
    let delta_full = solve_streaming_reduced_beta(&s_full, &rhs_full, &options)
        .unwrap_or_else(|e| panic!("in-core reduced solve failed: {e}"));

    // Streaming: split into several chunk sizes; every split must reproduce the
    // single-chunk reduction and the same Δβ to machine precision.
    for chunk_size in [16usize, 37, 60, 113] {
        let (s_stream, rhs_stream) =
            reduce_in_chunks(&sys, chunk_size, ridge_t, ridge_beta, &options);
        let mut max_s = 0.0_f64;
        let mut max_rhs = 0.0_f64;
        for i in 0..sys.k {
            max_rhs = max_rhs.max((rhs_stream[i] - rhs_full[i]).abs());
            for j in 0..sys.k {
                max_s = max_s.max((s_stream[[i, j]] - s_full[[i, j]]).abs());
            }
        }
        assert!(
            max_s < 1e-9,
            "chunk_size={chunk_size}: reduced Schur block deviates from in-core: max|ΔS|={max_s:.3e}"
        );
        assert!(
            max_rhs < 1e-9,
            "chunk_size={chunk_size}: reduced RHS deviates from in-core: max|Δrhs|={max_rhs:.3e}"
        );

        let delta_stream = solve_streaming_reduced_beta(&s_stream, &rhs_stream, &options)
            .unwrap_or_else(|e| panic!("streaming reduced solve failed: {e}"));
        let mut max_delta = 0.0_f64;
        for i in 0..sys.k {
            max_delta = max_delta.max((delta_stream[i] - delta_full[i]).abs());
        }
        assert!(
            max_delta < 1e-8,
            "chunk_size={chunk_size}: streaming Δβ deviates from in-core Δβ: max|Δ|={max_delta:.3e}"
        );
    }
}

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

#[test]
fn gpu_reduced_beta_solve_matches_cpu_when_available() {
    // Assemble a system and form the reduced `(S, rhs)` on the host; the GPU and
    // CPU reduced solves must agree on the same inputs.
    //
    // The reduced-β GPU PCG is dispatched by its CG-amortised work
    // `2·k²·max_iterations` against the 1e8 GEMM FLOP floor (#1017 re-key). With
    // `max_iterations = 200` (default), the device path is admitted only when
    // `k ≳ 500`. `beta_dim = Σ_atoms basis_size·p_out = k_atoms·4·2`, so 64 atoms
    // give k = 512 ⇒ `2·512²·200 ≈ 1.05e8 ≥ 1e8` and the device PCG actually runs
    // on a GPU host. The earlier k=64 fixture (8 atoms) was far below the floor:
    // `solve_reduced_beta_pcg` always returned Unavailable on a real GPU and the
    // parity assertion NEVER ran on hardware — a vacuous skip-pass (device-PCG
    // class, eee12f6b2).
    let (mut term, target, rho) = build_term(
        64,
        4,
        1,
        200,
        2,
        AssignmentMode::softmax(1.0),
        8,
        0x5EED_9012,
    );
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap_or_else(|e| panic!("assemble_arrow_schur failed: {e}"));
    let options = ArrowSolveOptions::automatic(sys.k);
    let (s_acc, rhs_acc) = reduce_in_chunks(&sys, sys.rows.len(), 1e-4, 1e-4, &options);

    // Symmetrize `S` the way the streaming solver does before factoring, so the
    // device PCG and the host solve see the identical operator.
    let k = sys.k;
    let mut s_sym = s_acc.clone();
    for i in 0..k {
        for j in (i + 1)..k {
            let avg = 0.5 * (s_sym[[i, j]] + s_sym[[j, i]]);
            s_sym[[i, j]] = avg;
            s_sym[[j, i]] = avg;
        }
    }

    let gpu = match solve_reduced_beta_pcg(
        &s_sym,
        &rhs_acc,
        options.trust_region.max_iterations,
        options.trust_region.steihaug_relative_tolerance,
    ) {
        Ok(delta) => delta,
        Err(ArrowSchurGpuFailure::Unavailable) => {
            // The fixture (k=512, iters=200) clears the 1e8 device dispatch floor,
            // so with a CUDA runtime present the reduced-β GPU PCG must run. An
            // Unavailable here means the device declined a workload it was sized to
            // run — a real fault, not a no-CUDA skip. Fail loud unless this is a
            // genuinely CPU-only host (device-PCG skip-pass class, eee12f6b2).
            assert!(
                gam::gpu::device_runtime::GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
                    .unwrap_or_else(|error| {
                        panic!("GPU probe fault in streaming Arrow-Schur test: {error}")
                    })
                    .is_none(),
                "[sae_streaming/gpu_parity] reduced-β GPU PCG returned Unavailable on a \
                 floor-clearing fixture (k=512, iters=200) with a CUDA runtime present — \
                 the device path declined a workload it must run"
            );
            eprintln!(
                "[sae_streaming/gpu_parity] no CUDA device — skipping GPU↔CPU \
                 reduced-β parity (CPU path is exercised by (a))"
            );
            return;
        }
        Err(other) => panic!("[sae_streaming/gpu_parity] GPU reduced solve failed: {other:?}"),
    };

    let cpu = solve_streaming_reduced_beta(&s_sym, &rhs_acc, &options)
        .unwrap_or_else(|e| panic!("CPU reduced solve failed: {e}"));

    assert_eq!(gpu.len(), cpu.len(), "GPU/CPU Δβ length mismatch");
    let mut max_abs = 0.0_f64;
    for i in 0..k {
        max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
    }
    assert!(
        max_abs < 1e-7,
        "GPU reduced-β solve deviates from CPU on the same S/rhs: max|Δ|={max_abs:.3e}"
    );
}

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
