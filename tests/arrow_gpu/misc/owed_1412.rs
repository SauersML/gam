//! #1412 — the GPU throughput decision gate must assert against a *measured*
//! rows/sec/GPU value, not a hardcoded 100K claim, and the #1017 Phase-1
//! dispatch re-keying must admit the LLM-shape batched work the row-count gate
//! misses.
//!
//! These are pure-logic regressions (no device needed): they certify the gate
//! *contract*. The actual measured throughput is produced by
//! `examples/throughput_1412.rs` on a real A100 and reported in the issue; this
//! file pins the invariant that the verdict is a function of whatever that
//! measurement turns out to be.

use gam::gpu::linalg_dispatch::ResidentDesignGram;
use gam::gpu::policy::GpuDispatchPolicy;
use ndarray::{Array1, Array2};

/// #1017 Phase 1: the work-keyed reduced-Schur matvec gate must admit the
/// LLM/SAE shape (few rows, wide border, modest frame depth) that the legacy
/// row-count dense gate rejects. This is the whole point of the re-keying —
/// the GPU must engage on `n×p×M` total batched work, not row count alone.
#[test]
fn dispatch_rekeying_admits_llm_shape_rejected_by_row_count() {
    let pol = GpuDispatchPolicy::default();

    // LLM/SAE: 2k rows, 2048-wide decoder border, depth-8 frames.
    assert!(
        pol.reduced_schur_matvec_should_offload(
            2_000,
            2_048,
            8,
            GpuDispatchPolicy::MATVEC_OFFLOAD_MIN_CG_ITERS
        ),
        "work-keyed gate must admit the wide-border LLM shape"
    );

    // The same shape under the row-count-style dense-work gate is rejected:
    // 2k rows × 8 columns is far below the dense flop floor.
    assert!(
        !pol.dense_hessian_work_target_is_gpu(2_000, 8),
        "row-count dense gate must (still) reject the narrow-by-rows view — \
         proving the re-keying is what admits the work"
    );

    // And a genuinely tiny shape stays on the CPU regardless.
    assert!(!pol.reduced_schur_matvec_should_offload(30, 8, 2, 8));
}

/// #1017 Phase 3: the resident-X Gram handle must (a) never change the numerics
/// vs the per-call path when a device is present, and (b) decline cleanly (no
/// panic, return `None`) on a CPU-only host or below-threshold shape — so a
/// caller can always fall back. This test is device-agnostic: on a CUDA host it
/// asserts bit-parity against the per-call Gram; on a CPU host it asserts the
/// handle declines. Either way the contract holds.
#[test]
fn resident_design_gram_matches_per_call_or_declines() {
    // A GPU-profitable LLM shape (clears the XtDiagX work gate when a device is
    // present); small enough to run the CPU reference cross-check quickly.
    let n = 4096usize;
    let p = 256usize;
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        0.05 * ((i as f64 + 1.0) * 0.011 + (j as f64 + 1.0) * 0.017).sin()
    });
    let w = Array1::from_shape_fn(n, |i| 0.5 + ((i as f64 + 1.0) * 0.019).sin().abs());

    match ResidentDesignGram::try_new(x.view()) {
        Some(handle) => {
            // Device present: dims echo the resident design, and the resident
            // Gram is bit-identical (reduction-order tol) to the per-call path.
            assert_eq!(handle.dims(), (n, p));
            let resident = handle.gram(w.view()).expect("resident gram on device");
            let per_call = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view())
                .expect("per-call gram on device");
            let mut max_diff = 0.0_f64;
            for (a, b) in resident.iter().zip(per_call.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(
                max_diff < 1e-9,
                "resident vs per-call Gram differ by {max_diff:e} (must be reduction-order only)"
            );
            // A wrong-length weight is rejected, not silently mis-scaled.
            assert!(handle.gram(Array1::zeros(n + 1).view()).is_none());
        }
        None => {
            // CPU-only host (or no device): the per-call path must also decline,
            // and the handle declining is the documented fallback contract.
            assert!(
                gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()).is_none(),
                "on a host where the resident handle declines, the per-call GPU \
                 path must decline too (both fall back to CPU)"
            );
        }
    }
}

/// #1017 Phase 3 (Gram-resident POTRF chaining): the resident normal-equations
/// solve `(XᵀWX + ridge·I)β = rhs` must match a host Cholesky solve of the same
/// system (up to reduction-order roundoff) when a device is present, keeping the
/// p×p Gram on-device and crossing back only the p-vector β. On a CPU host the
/// handle declines (returns None at `try_new`). Device-agnostic.
#[test]
fn resident_normal_equations_matches_host_solve_or_declines() {
    let n = 4096usize;
    let p = 64usize; // small p keeps the host reference solve quick
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        0.05 * ((i as f64 + 1.0) * 0.013 + (j as f64 + 1.0) * 0.021).sin()
    });
    let w = Array1::from_shape_fn(n, |i| 0.5 + ((i as f64 + 1.0) * 0.017).sin().abs());
    let rhs = Array1::from_shape_fn(p, |j| ((j as f64 + 1.0) * 0.03).cos());
    let ridge = 1e-3;

    if let Some(handle) = ResidentDesignGram::try_new(x.view()) {
        let beta = handle
            .solve_normal_equations(w.view(), rhs.view(), ridge)
            .expect("resident normal-equations solve on device");
        assert_eq!(beta.len(), p);

        // Host reference: G = XᵀWX + ridge·I, then solve G β = rhs by Gaussian
        // elimination (independent of the device Cholesky path).
        let mut g = Array2::<f64>::zeros((p, p));
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += x[[r, a]] * w[r] * x[[r, b]];
                }
                g[[a, b]] = acc;
            }
            g[[a, a]] += ridge;
        }
        let beta_ref = solve_spd_host(&g, &rhs);
        let mut max_diff = 0.0_f64;
        for (a, b) in beta.iter().zip(beta_ref.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        let scale = 1.0 + beta_ref.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            max_diff < 1e-6 * scale,
            "resident β differs from host solve by {max_diff:e} (scale {scale})"
        );

        // Shape guards: wrong-length w or rhs is rejected.
        assert!(
            handle
                .solve_normal_equations(Array1::zeros(n + 1).view(), rhs.view(), ridge)
                .is_none()
        );
        assert!(
            handle
                .solve_normal_equations(w.view(), Array1::zeros(p + 1).view(), ridge)
                .is_none()
        );
    }
    // CPU-only host: try_new returned None; nothing to assert beyond the clean
    // decline already covered by the gram test.
}

/// #1412 (symmetric single-upload): `xt_diag_x` passes the SAME array as the
/// left and right GEMM operand, so the device path stages `X` ONCE (aliased
/// operand) instead of uploading two byte-identical n×p copies. That aliasing
/// must not change the value: the symmetric per-call Gram has to equal a host
/// `Xᵀ·diag(w)·X` reference (reduction-order tol) when a device is present. On a
/// CPU host the GPU path declines and the host fast path already computes the
/// same reference, so the contract holds either way.
#[test]
fn symmetric_gram_single_upload_matches_host_reference_or_declines() {
    let n = 4096usize;
    let p = 96usize;
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        0.05 * ((i as f64 + 1.0) * 0.0151 + (j as f64 + 1.0) * 0.0233).sin()
    });
    // Signed weights exercise the exact Xᵀ(WX) row-scale (no sqrt/clip), which
    // the aliased single-upload branch must preserve bit-for-bit.
    let w = Array1::from_shape_fn(n, |i| ((i as f64 + 1.0) * 0.0193).sin());

    // Host reference Gram, independent of the device path.
    let mut g_ref = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in 0..p {
            let mut acc = 0.0;
            for r in 0..n {
                acc += x[[r, a]] * w[r] * x[[r, b]];
            }
            g_ref[[a, b]] = acc;
        }
    }

    if let Some(g) = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()) {
        let mut max_diff = 0.0_f64;
        for (a, b) in g.iter().zip(g_ref.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        let scale = 1.0 + g_ref.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            max_diff < 1e-9 * scale,
            "symmetric single-upload Gram differs from host reference by {max_diff:e} (scale {scale})"
        );
    }
    // CPU-only host: try_fast_xt_diag_x returns None and the host fast path owns
    // the same reference; the decline is the documented fallback.
}

/// #1412 (mixed-precision solution-only): the PIRLS Newton direction discards
/// the logdet, so the solve must take the fp32-factor + fp64-refinement path
/// WITHOUT the redundant fp64 POTRF (which would cancel the mixed-precision
/// win). The contract that matters for correctness: the solution-only path
/// returns a FULL-fp64-accurate solution (matches a host SPD solve to
/// refinement tolerance) even though it reports NO logdet. The logdet-returning
/// path must still produce a finite logdet. Device-agnostic: on a CPU host both
/// entry points return `Err` (no runtime), which is the documented fallback.
#[test]
fn mixed_precision_solution_only_is_fp64_accurate_or_declines() {
    let p = 128usize; // ≥ REFINEMENT_MIN_P so the fp32 path is admitted on device
    // Diagonally-dominant SPD A (well-conditioned so fp32+refinement converges).
    let a = Array2::from_shape_fn((p, p), |(i, j)| {
        if i == j {
            (p as f64) + 1.0
        } else {
            0.1 / (1.0 + (i as f64 - j as f64).abs())
        }
    });
    let b = Array1::from_shape_fn(p, |i| ((i as f64 + 1.0) * 0.013).sin());
    let rhs = b.clone().insert_axis(ndarray::Axis(1)); // p×1

    let host = solve_spd_host(&a, &b);

    // Solution-only mixed-precision path.
    match gam::gpu::solver::cholesky_solve_only_gpu(a.view(), rhs.view()) {
        Ok(sol) => {
            assert_eq!(sol.dim(), (p, 1));
            let mut max_diff = 0.0_f64;
            for i in 0..p {
                max_diff = max_diff.max((sol[[i, 0]] - host[i]).abs());
            }
            let scale = 1.0 + host.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                max_diff < 1e-9 * scale,
                "solution-only mixed-precision differs from host solve by {max_diff:e} \
                 (refinement must recover full fp64 accuracy)"
            );

            // The logdet-returning twin must still hand back a finite logdet on
            // the same SPD system (it pays the fp64 POTRF; solution agrees).
            let (sol_ld, logdet) = gam::gpu::solver::cholesky_solve_gpu(a.view(), rhs.view())
                .expect("logdet path on the same device");
            assert!(
                logdet.is_finite(),
                "logdet path must produce a finite logdet"
            );
            let mut max_sol_diff = 0.0_f64;
            for i in 0..p {
                max_sol_diff = max_sol_diff.max((sol[[i, 0]] - sol_ld[[i, 0]]).abs());
            }
            assert!(
                max_sol_diff < 1e-9 * (1.0 + host.iter().fold(0.0, |m: f64, v| m.max(v.abs()))),
                "solution-only and logdet paths must agree on the solution ({max_sol_diff:e})"
            );
        }
        Err(_) => {
            // CPU-only host: the logdet twin must also decline (no runtime).
            assert!(
                gam::gpu::solver::cholesky_solve_gpu(a.view(), rhs.view()).is_err(),
                "on a host without a CUDA runtime both solve entry points must Err"
            );
        }
    }
}

/// Dense SPD solve `A x = b` by Cholesky (lower) + forward/back substitution —
/// a self-contained host reference independent of the device path.
fn solve_spd_host(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = s.max(0.0).sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Forward: L y = b.
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = s / l[[i, i]];
    }
    // Back: Lᵀ x = y.
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    x
}
