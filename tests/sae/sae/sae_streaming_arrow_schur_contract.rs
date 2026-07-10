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
//!       JumpReLU structural gate, each token's Arrow-Schur row block is sized
//!       by its *active* atom set (`|active| + Σ_{k∈active} d_k`), never by the
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

use gam::solver::arrow_schur::{
    ArrowSchurSystem, ArrowSolveOptions, StreamingArrowSchur, solve_streaming_reduced_beta,
};
use gam::solver::gpu_kernels::arrow_schur::{ArrowSchurGpuFailure, solve_reduced_beta_pcg};
use gam::terms::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
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
/// so the streaming reduction and the JumpReLU active-set layout are exercised
/// on the same code path the `K = 100K` fit uses. The gating logits are seeded
/// so that, under JumpReLU with `threshold = 0`, exactly `n_active` atoms clear
/// the gate per token regardless of `K`.
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

    // Gating logits: the first `n_active` atoms (modulo a per-row rotation) are
    // driven positive, the rest negative, so a `threshold = 0` JumpReLU gate
    // admits exactly `n_active` atoms per token independent of `K`.
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
        let atom = SaeManifoldAtom::new(
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

/// Reduce an Arrow-Schur system into `(S, rhs)` by streaming the rows in
/// `chunk_size`-row chunks, mirroring `SaeManifoldTerm::run_joint_fit_arrow_schur_streaming`'s
/// per-chunk accumulation: seed `S` with `H_ββ` once (no per-chunk ridge),
/// accumulate the per-row reduction across all chunks, then fold the global
/// β-ridge in exactly once.
fn reduce_in_chunks(
    sys: &ArrowSchurSystem,
    chunk_size: usize,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
) -> (Array2<f64>, Array1<f64>) {
    let k = sys.k;
    let n = sys.rows.len();
    let mut s_acc = Array2::<f64>::zeros((k, k));
    let mut rhs_acc = Array1::<f64>::zeros(k);
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk_size).min(n);
        let mut streaming = StreamingArrowSchur::from_system(sys, (end - start).max(1));
        streaming
            .reset_accumulator(0.0)
            .unwrap_or_else(|e| panic!("reset_accumulator failed: {e}"));
        streaming
            .accumulate_chunk(start, end, ridge_t, options.mode)
            .unwrap_or_else(|e| panic!("accumulate_chunk failed: {e}"));
        let (contrib_s, contrib_rhs) = streaming.take_accumulators();
        // `contrib_s` is `H_ββ − Σ_{i∈chunk} reduction`; subtracting the
        // shared `H_ββ` once per chunk and adding it back a single time below
        // reconstructs `H_ββ − Σ_all reduction` exactly. We accumulate the
        // per-chunk reductions by tracking the delta from the seeded `H_ββ`.
        for i in 0..k {
            rhs_acc[i] += contrib_rhs[i];
            for j in 0..k {
                s_acc[[i, j]] += contrib_s[[i, j]];
            }
        }
        start = end;
    }
    // Each chunk re-seeded `S` with the full `H_ββ`; collapse the redundant
    // seedings so a single `H_ββ` remains, then add the global β ridge once.
    let n_chunks = n.div_ceil(chunk_size).max(1);
    let hbb = sys.effective_penalty_op().to_dense();
    for i in 0..k {
        for j in 0..k {
            s_acc[[i, j]] -= (n_chunks as f64 - 1.0) * hbb[[i, j]];
        }
        s_acc[[i, i]] += ridge_beta;
    }
    (s_acc, rhs_acc)
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
    // JumpReLU with threshold 0: exactly `n_active` atoms clear the gate per
    // token, so each token's compact row block is `n_active·(1 + d)` wide,
    // independent of K.
    let mode = AssignmentMode::threshold_gate(1.0, 0.0);
    let expected_block_dim = n_active * (1 + d);

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
                gam::gpu::device_runtime::GpuRuntime::global().is_none(),
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
#[test]
fn streaming_full_fit_is_chunk_size_invariant() {
    let (k, m, d, n, p) = (4usize, 4usize, 1usize, 36usize, 2usize);

    // Deterministic per-row seed (logits, coords) + targets, generated once and
    // sliced identically for every chunking. The driver re-seeds from this each
    // pass, so it fully determines the fit independent of any resident state.
    let mut rng = 0xA11CE_u64;
    let full_logits = Array2::<f64>::from_shape_fn((n, k), |_| 0.4 * lcg_f64(&mut rng));
    let full_coords: Vec<Array2<f64>> = (0..k)
        .map(|_| Array2::<f64>::from_shape_fn((n, d), |_| 0.3 * lcg_f64(&mut rng)))
        .collect();
    let full_target = Array2::<f64>::from_shape_fn((n, p), |_| lcg_f64(&mut rng));

    let fit_with_chunk = |chunk_size: usize| -> Array1<f64> {
        // Rebuild an identical term (deterministic seed) for each fit, so only
        // the chunking differs between runs.
        let (mut term, _t, mut rho) =
            build_term(k, m, d, n, p, AssignmentMode::softmax(1.0), k, 0x5EED_99);
        let logits = full_logits.clone();
        let coords = full_coords.clone();
        let z_full = full_target.clone();
        let seeder = move |start: usize, end: usize| {
            let lg = logits.slice(s![start..end, ..]).to_owned();
            let cd: Vec<Array2<f64>> = coords
                .iter()
                .map(|c| c.slice(s![start..end, ..]).to_owned())
                .collect();
            let z = z_full.slice(s![start..end, ..]).to_owned();
            Ok::<_, String>((lg, cd, z))
        };
        term.run_joint_fit_arrow_schur_streaming(
            n, chunk_size, &mut rho, None, 2, 1.0, 1e-4, 1e-4, seeder,
        )
        .unwrap_or_else(|e| panic!("streaming fit (chunk_size={chunk_size}) failed: {e}"));
        term.flatten_beta()
    };

    let beta_one_chunk = fit_with_chunk(n);
    assert!(
        beta_one_chunk.iter().all(|v| v.is_finite()),
        "single-chunk streaming fit produced non-finite decoder β"
    );
    for chunk_size in [7usize, 13, 25] {
        let beta_chunked = fit_with_chunk(chunk_size);
        let mut max_dev = 0.0_f64;
        for (a, b) in beta_one_chunk.iter().zip(beta_chunked.iter()) {
            max_dev = max_dev.max((a - b).abs());
        }
        assert!(
            max_dev < 1e-6,
            "streaming fit decoder β depends on chunk_size={chunk_size}: max|Δβ|={max_dev:.3e}"
        );
    }
}
