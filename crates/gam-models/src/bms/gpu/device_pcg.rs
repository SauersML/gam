// ────────────────────────────────────────────────────────────────────────
// Block 9 Phase 5 — device-resident PCG against the BMS-FLEX row-Hessian
// operator.
//
// The inner Newton solve in `BernoulliMarginalSlope` (matrix-free path,
// large-scale shape n=195k, p=44, r=20) currently reaches the GPU as a
// per-CG-iteration call to `launch_bms_flex_row_hvp` returning a host
// `Vec<f64>`. With ~6400 inner CG iterations per outer iteration that round-
// trip cost dominates: each iter pays one `stream.synchronize()` plus one
// DtoH download. At p=44 the download itself is 352 bytes — trivial in
// bandwidth, painful in latency.
//
// Phase 5 keeps every PCG vector on the device and runs the outer loop with
// only a single small scalar download per iteration (the squared residual
// norm for the convergence check). The Hv kernel becomes `into_device`
// (Block 9 addition to `bms_flex_row.rs`), and the axpy / dot / diagonal-
// preconditioner / scale-and-add steps run as tiny NVRTC kernels on the
// same default stream so the sequence is implicitly ordered without sync.
// ────────────────────────────────────────────────────────────────────────

/// Inputs to [`run_pcg_against_row_hessian_device`]. The right-hand-side
/// `b` is supplied as a host slice (it is the only host-resident vector
/// that needs to enter the loop — the iterate, residual, search direction,
/// and Hv output all live on the device).
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgInput<'a> {
    /// Per-fit row-Hessian + design storage. The PCG operator is
    /// `v ↦ launch_bms_flex_row_hvp_into_device(storage, ...)`.
    pub storage: &'a crate::bms::gpu::row::DeviceResidentRowHess,
    /// Right-hand-side `b`, length `storage.block.p_total`. Uploaded once.
    pub b: &'a [f64],
    /// Convergence tolerance on relative residual `‖r‖₂ / ‖b‖₂`.
    pub rel_tol: f64,
    /// Hard cap on iterations (the inner loop also bails on stagnation).
    pub max_iters: usize,
    /// Floor on `|diag(H)[i]|` used by the Jacobi preconditioner. Set to
    /// `1e-12` for the matrix-free row-Hessian path; the row-primary
    /// Hessian's diagonal is positive-definite by construction.
    pub precond_diag_floor: f64,
}

/// Output of [`run_pcg_against_row_hessian_device`].
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgOutput {
    /// Solution `x` such that `H · x ≈ b`, length `storage.block.p_total`.
    pub x: Vec<f64>,
    /// Number of PCG iterations consumed (final iter does not count if it
    /// converged immediately after the dot reduction).
    pub iterations: usize,
    /// Final achieved relative residual `‖r‖₂ / ‖b‖₂`.
    pub final_rel_residual: f64,
}

/// Block 9 Phase 5 — V100 parity for `run_pcg_against_row_hessian_device`.
///
/// Builds a small `(n=64, r=20, p=44)` BMS-FLEX row-Hessian fixture, computes
/// the dense joint Hessian via the same CPU oracle the HVP parity test uses,
/// solves `H · x = b` on the host via dense LU as ground truth, and asserts
/// the device-resident PCG iterate matches to a tight tolerance.
#[cfg(all(test, target_os = "linux"))]
mod pcg_device_parity_tests {
    use super::*;
    use crate::bms::gpu::row::{BmsFlexBlockLayout, BmsFlexPrimaryLayout, DeviceResidentRowHess};
    use ndarray::Array2;

    #[test]
    fn pcg_device_matches_dense_oracle_at_n64_r20_p44() {
        let n = 64_usize;
        let p_m = 14_usize;
        let p_g = 12_usize;
        let p_h_dim = 10_usize;
        let p_w_dim = 8_usize;
        let r = 2 + p_h_dim + p_w_dim;
        let p_total = p_m + p_g + p_h_dim + p_w_dim;
        let block = BmsFlexBlockLayout {
            p_m,
            p_g,
            h: Some(p_m + p_g..p_m + p_g + p_h_dim),
            w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
            p_total,
        };
        let primary = BmsFlexPrimaryLayout {
            h: Some(2..2 + p_h_dim),
            w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
            r,
        };

        // Same deterministic symmetric Hessians + designs as the HVP parity
        // gate, so any drift between Phase 4 and Phase 5 surfaces here too.
        let mut row_hessians = vec![0.0_f64; n * r * r];
        for row in 0..n {
            let base = row * r * r;
            for u in 0..r {
                for v in 0..r {
                    let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (v as f64) * 0.317;
                    let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                    row_hessians[base + u * r + v] = a;
                }
            }
            for u in 0..r {
                for v in (u + 1)..r {
                    let upper = row_hessians[base + u * r + v];
                    let lower = row_hessians[base + v * r + u];
                    let sym = 0.5 * (upper + lower);
                    row_hessians[base + u * r + v] = sym;
                    row_hessians[base + v * r + u] = sym;
                }
                // Boost the diagonal heavily so each H_i is positive
                // definite — guarantees the joint pulled-back Hessian is
                // SPD, which PCG requires.
                row_hessians[base + u * r + u] += 4.0 * (r as f64);
            }
        }
        let mut marginal = vec![0.0_f64; n * p_m];
        for row in 0..n {
            for j in 0..p_m {
                // Orthonormal DCT-II columns make the aggregate pullback
                // full-rank by construction. The former phase-shifted
                // sinusoids were nearly collinear, so row-wise SPD did not
                // imply a numerically SPD joint fixture.
                let scale = if j == 0 {
                    (n as f64).sqrt().recip()
                } else {
                    (2.0 / n as f64).sqrt()
                };
                marginal[row * p_m + j] =
                    scale * (std::f64::consts::PI * (row as f64 + 0.5) * j as f64 / n as f64).cos();
            }
        }
        let mut slope = vec![0.0_f64; n * p_g];
        for row in 0..n {
            for j in 0..p_g {
                let scale = if j == 0 {
                    (n as f64).sqrt().recip()
                } else {
                    (2.0 / n as f64).sqrt()
                };
                slope[row * p_g + j] =
                    scale * (std::f64::consts::PI * (row as f64 + 0.5) * j as f64 / n as f64).cos();
            }
        }

        // Pick a non-trivial RHS.
        let b: Vec<f64> = (0..p_total)
            .map(|i| {
                let seed = (i as f64) * 0.157 + 0.6;
                seed.sin() * 0.55 + (seed * 0.4).cos() * 0.35
            })
            .collect();

        let h_dense =
            cpu_dense_joint_hessian(&row_hessians, &marginal, &slope, &block, &primary, n);
        let x_oracle = cpu_pcg_oracle(&h_dense, &b, 1e-12);

        // #2422 EVERY HOST: the fixture's SPD certificate. The oracle must
        // actually solve `H x = b`, otherwise a device-free run returns having
        // verified nothing and a CUDA run grades the device against an
        // unchecked reference.
        {
            let x = ndarray::Array1::from_vec(x_oracle.clone());
            let residual = h_dense.dot(&x) - ndarray::Array1::from_vec(b.clone());
            let r_inf = residual.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let b_inf = b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            assert!(
                r_inf <= 1e-8 * b_inf,
                "CPU PCG oracle does not solve the joint system: ‖Hx − b‖∞ = {r_inf:.3e} \
                 (‖b‖∞ = {b_inf:.3e})"
            );
        }

        // Keep the SPD fixture certificate CPU-reachable. CUDA availability
        // controls only the device-parity half of this test.
        let runtime = match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(runtime)) => runtime,
            Ok(None) => {
                eprintln!("[pcg_device parity] host SPD oracle passed; no CUDA device");
                return;
            }
            Err(error) => panic!("[pcg_device parity] CUDA probe failed: {error}"),
        };

        // Grab the same CUDA context + default stream that the bms_flex_row
        // kernels will use when `run_pcg_against_row_hessian_device` probes
        // its own backend. Going through the public runtime APIs keeps the
        // test independent of any private kernel-backend symbols.
        // Past the lossless Auto-resolution gate above: a context-creation or
        // HtoD-upload failure here is a real device fault on a CUDA host, not a
        // no-CUDA skip — fail loud (device-PCG skip-pass class, eee12f6b2). The old
        // arms returned, so a context/upload fault on a GPU host passed silently.
        let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.selected_device().ordinal)
            .expect("[pcg_device parity] cuda_context_for must succeed on a CUDA host");
        let stream = ctx.default_stream();
        let d_h = stream
            .clone_htod(&row_hessians)
            .expect("[pcg_device parity] upload h must succeed on a CUDA host");
        let d_m = stream
            .clone_htod(&marginal)
            .expect("[pcg_device parity] upload marginal must succeed on a CUDA host");
        let d_g = stream
            .clone_htod(&slope)
            .expect("[pcg_device parity] upload slope must succeed on a CUDA host");
        let storage = DeviceResidentRowHess {
            neglog: stream
                .alloc_zeros::<f64>(n)
                .expect("[pcg_device parity] alloc neglog"),
            grad: stream
                .alloc_zeros::<f64>(n * r)
                .expect("[pcg_device parity] alloc grad"),
            hess: d_h,
            marginal_design: d_m,
            slope_design: d_g,
            n,
            r,
            block,
            primary,

            bytes: ((n + n * r + n * r * r + n * p_m + n * p_g)
                * std::mem::size_of::<f64>()) as u64,
        };

        let out = run_pcg_against_row_hessian_device(DeviceResidentPcgInput {
            storage: &storage,
            b: &b,
            rel_tol: 1e-10,
            max_iters: 4 * p_total,
            precond_diag_floor: 1e-12,
        })
        .expect("device-resident PCG must succeed on SPD fixture");

        assert_eq!(out.x.len(), p_total);
        let mut max_abs = 0.0_f64;
        for i in 0..p_total {
            let diff = (out.x[i] - x_oracle[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        // Each iteration introduces O(1) ULPs of round-off in the dot/
        // axpy ladder; with ~88 iters max at p=44 we expect ‖Δx‖∞ comfortably
        // below 1e-7. Anything larger means a code bug, not float noise.
        assert!(
            max_abs <= 1e-7,
            "pcg_device parity ‖Δx‖∞={max_abs:.3e} > 1e-7 after {} iters \
             (final rel residual={:.3e})",
            out.iterations,
            out.final_rel_residual
        );
        eprintln!(
            "[pcg_device parity] n={n} p={p_total} r={r}: iters={} rel_res={:.3e} ‖Δx‖∞={:.3e}",
            out.iterations, out.final_rel_residual, max_abs
        );
    }
}
