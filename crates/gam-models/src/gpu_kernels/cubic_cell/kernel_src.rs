//! NVRTC-compilable CUDA C++ source for the de-nested cubic-cell derivative
//! moment kernel, emitted as Rust string constants.
//!
//! The actual nvcc/NVRTC invocation lives elsewhere; this module is pure
//! string construction and is only compiled on Linux+CUDA where
//! `CubicCellGpuBackend::module_for_degree` consumes the emitted source.
//! One specialization is emitted per `max_degree`; consumers use
//! `_d9`, `_d15`, `_d21` for the published high-water marks.

#![cfg(target_os = "linux")]

use std::fmt::Write as _;

use crate::cubic_cell_kernel::{GL_NODES_FOR_GPU_KERNEL, GL_WEIGHTS_FOR_GPU_KERNEL};

/// Emit the full NVRTC source for one `max_degree` specialization. The kernel
/// symbol is `cubic_deriv_moments_d{max_degree}`. Output is deterministic:
/// byte-equal across calls with the same argument.
pub(crate) fn build_cubic_deriv_moments_kernel_source(max_degree: usize) -> String {
    let mut src = String::with_capacity(64 * 1024);

    src.push_str(HEADER);
    writeln!(src, "#define MAX_DEGREE {}", max_degree).expect("writes to String are infallible");
    src.push_str("#define MOMENT_STRIDE (MAX_DEGREE + 1)\n");
    src.push_str("#define GL_N 384\n");
    src.push_str("#define LANES_PER_WARP 32\n");
    src.push_str("#define NODES_PER_LANE 12\n\n");

    src.push_str("__constant__ double GL_NODES[GL_N] = {\n");
    emit_table(&mut src, GL_NODES_FOR_GPU_KERNEL);
    src.push_str("};\n\n");

    src.push_str("__constant__ double GL_WEIGHTS[GL_N] = {\n");
    emit_table(&mut src, GL_WEIGHTS_FOR_GPU_KERNEL);
    src.push_str("};\n\n");

    src.push_str(DEVICE_HELPERS);

    writeln!(
        src,
        "extern \"C\" __global__ void cubic_deriv_moments_d{degree}(",
        degree = max_degree
    )
    .expect("writes to String are infallible");
    src.push_str(KERNEL_BODY);

    src
}

fn emit_table(dst: &mut String, table: &[f64; 384]) {
    for value in table.iter() {
        writeln!(dst, "    {value:.17e},").expect("writes to String are infallible");
    }
}

const HEADER: &str = r#"// AUTO-GENERATED CUDA C++ source for the de-nested cubic-cell derivative
// moment kernel. Do not edit by hand; see src/gpu/cubic_cell/kernel_src.rs.
//
// One warp processes one cell. For non-affine finite cells each lane folds
// 12 of the 384 GL nodes (stride 32) and the warp reduces via __shfl_xor_sync
// butterflies. For affine and affine-tail cells lane 0 runs the closed-form
// q'-recurrence and broadcasts via __shfl_sync.

// We deliberately do NOT `#include <stdint.h>`. NVRTC compiles with no usable
// C library: even when invoked with `--include-path=/usr/include` (the
// arch-aware compile path) the system <stdint.h> transitively pulls in
// <gnu/stubs.h> → <gnu/stubs-32.h>, which is absent on boxes without 32-bit dev
// libs, aborting the JIT with "cannot open source file gnu/stubs-32.h". The
// kernel only needs two fixed-width integer types, and the CUDA device ABI
// fixes their widths (`unsigned char` = 8, `unsigned int` = 32), so we typedef
// them inline — self-contained, no host headers, robust across boxes. This
// mirrors the inline `CUDART_INF` below (same "NVRTC has no headers" reason).
typedef unsigned char  uint8_t;
typedef unsigned int   uint32_t;

// CUDART_INF normally comes from the CUDA math-constants header, but that
// header is not on NVRTC's default search path, so pulling it in aborts the JIT
// compile with "could not open source file". Define the one symbol we use
// inline instead, matching the CUDA header's own value. __longlong_as_double is
// an always-available NVRTC builtin (no header).
#define CUDART_INF (__longlong_as_double(0x7ff0000000000000ULL))

#define STATUS_OK              0
#define STATUS_INVALID         1
#define STATUS_NONAFFINE_INF   2
#define STATUS_NONFINITE_COEF  3
#define STATUS_NONFINITE_Q     4

#define BRANCH_AFFINE          0
#define BRANCH_NONAFFINE_FIN   1
#define BRANCH_AFFINE_TAIL     2

"#;

const DEVICE_HELPERS: &str = r#"
__device__ __forceinline__ double warp_allreduce_sum(double v) {
    // Butterfly all-reduce across 32 lanes.
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ double safe_eta(double z, double c0, double c1, double c2, double c3) {
    return ((c3 * z + c2) * z + c1) * z + c0;
}

__device__ __forceinline__ double q_of_z(double z, double c0, double c1, double c2, double c3) {
    double eta = safe_eta(z, c0, c1, c2, c3);
    return 0.5 * (z * z + eta * eta);
}

__device__ __forceinline__ bool is_finite_d(double v) {
    return isfinite(v);
}

// Normal CDF expressed through erfc for numerical stability on tails.
__device__ __forceinline__ double phi_cdf(double x) {
    if (isinf(x)) {
        return x > 0.0 ? 1.0 : 0.0;
    }
    return 0.5 * erfc(-x * 0.70710678118654752440);
}

"#;

const KERNEL_BODY: &str = r#"    const double* __restrict__ cell_left,
    const double* __restrict__ cell_right,
    const double* __restrict__ c0_arr,
    const double* __restrict__ c1_arr,
    const double* __restrict__ c2_arr,
    const double* __restrict__ c3_arr,
    const uint8_t* __restrict__ branch_code,
    double* __restrict__ moment_output,
    uint8_t* __restrict__ status,
    uint32_t n_cells)
{
    // One warp per cell. Block layout: blockDim.x = 32 * warps_per_block.
    const unsigned warp_id_in_block = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warps_per_block = blockDim.x >> 5;
    const unsigned cell_id = blockIdx.x * warps_per_block + warp_id_in_block;
    if (cell_id >= n_cells) {
        return;
    }

    const double L = cell_left[cell_id];
    const double R = cell_right[cell_id];
    const double c0 = c0_arr[cell_id];
    const double c1 = c1_arr[cell_id];
    const double c2 = c2_arr[cell_id];
    const double c3 = c3_arr[cell_id];
    const uint8_t branch = branch_code[cell_id];

    const unsigned out_base = cell_id * (unsigned)MOMENT_STRIDE;

    // Lane 0 validates and broadcasts a status code; on non-OK we zero and exit.
    uint8_t local_status = STATUS_OK;
    if (lane == 0) {
        if (!is_finite_d(c0) || !is_finite_d(c1) || !is_finite_d(c2) || !is_finite_d(c3)) {
            local_status = STATUS_NONFINITE_COEF;
        } else if (branch == BRANCH_NONAFFINE_FIN) {
            if (!(R > L) || !is_finite_d(L) || !is_finite_d(R)) {
                local_status = STATUS_INVALID;
            }
        } else if (branch == BRANCH_AFFINE_TAIL) {
            // Host classifier vets c2/c3 against NORMALIZED_CELL_BRANCH_TOL
            // (1e-10 by default); tails with material curvature never reach
            // this kernel. We treat sub-tol c2/c3 as exact zero below so the
            // q'-recurrence stays the only branch that runs and the device
            // result matches `affine_anchor_moment_vector` byte-for-byte.
            if (!(R > L)) {
                local_status = STATUS_INVALID;
            }
        } else if (branch == BRANCH_AFFINE) {
            if (!(R > L)) {
                local_status = STATUS_INVALID;
            }
        } else {
            local_status = STATUS_INVALID;
        }
    }
    unsigned status_bcast = __shfl_sync(0xffffffff, (unsigned)local_status, 0);
    if (status_bcast != STATUS_OK) {
        if (lane == 0) {
            status[cell_id] = (uint8_t)status_bcast;
        }
        for (int k = (int)lane; k < (int)MOMENT_STRIDE; k += 32) {
            moment_output[out_base + (unsigned)k] = 0.0;
        }
        return;
    }

    if (branch == BRANCH_NONAFFINE_FIN) {
        // Map GL nodes from [-1, 1] to [L, R]: z = mid + half * t.
        const double half = 0.5 * (R - L);
        const double mid  = 0.5 * (R + L);

        // Per-lane partial moments.
        double partial[MOMENT_STRIDE];
        #pragma unroll
        for (int k = 0; k < (int)MOMENT_STRIDE; ++k) {
            partial[k] = 0.0;
        }

        bool nonfinite_q = false;
        #pragma unroll 1
        for (int j = 0; j < NODES_PER_LANE; ++j) {
            int idx = (int)lane + j * LANES_PER_WARP;
            // idx in [0, 384).
            double t = GL_NODES[idx];
            double w = GL_WEIGHTS[idx];
            double z = mid + half * t;
            double q = q_of_z(z, c0, c1, c2, c3);
            if (!is_finite_d(q)) {
                nonfinite_q = true;
                break;
            }
            double f = exp(-q) * w * half;
            double zk = 1.0;
            #pragma unroll
            for (int k = 0; k < (int)MOMENT_STRIDE; ++k) {
                partial[k] += f * zk;
                zk *= z;
            }
        }

        // Reduce nonfinite flag across the warp via integer OR.
        unsigned bad = __any_sync(0xffffffff, nonfinite_q ? 1 : 0);
        if (bad) {
            if (lane == 0) {
                status[cell_id] = STATUS_NONFINITE_Q;
            }
            for (int k = (int)lane; k < (int)MOMENT_STRIDE; k += 32) {
                moment_output[out_base + (unsigned)k] = 0.0;
            }
            return;
        }

        // All-reduce each moment across the warp; lane k writes M_k for k < MOMENT_STRIDE.
        #pragma unroll
        for (int k = 0; k < (int)MOMENT_STRIDE; ++k) {
            double m = warp_allreduce_sum(partial[k]);
            if ((int)lane == k) {
                moment_output[out_base + (unsigned)k] = m;
            }
        }
        if (lane == 0) {
            status[cell_id] = STATUS_OK;
        }
        return;
    }

    // Affine / Affine-tail: lane 0 runs the closed-form q'-recurrence.
    if (lane == 0) {
        const double alpha = c0;
        const double beta  = c1;
        const double d0 = alpha * beta;
        const double d1 = 1.0 + beta * beta;
        // q(z) = 0.5*(1+beta^2)*(z - mu)^2 + alpha^2 / (2*(1+beta^2))
        const double mu = -d0 / d1;
        const double s  = sqrt(d1);
        const double SQRT_TWO_PI = 2.50662827463100050241;
        const double prefactor = exp(-(alpha * alpha) / (2.0 * d1)) / s * SQRT_TWO_PI;

        // Endpoint boundary terms B_n = R^n * exp(-q(R)) - L^n * exp(-q(L)).
        // Infinite endpoints contribute zero.
        double qL = 0.0, qR = 0.0;
        double expL = 0.0, expR = 0.0;
        bool L_finite = isfinite(L);
        bool R_finite = isfinite(R);
        // Affine path uses c2=c3=0 exactly so the q values agree with the
        // CPU `affine_anchor_moment_vector` reference even when the host
        // classifier let sub-tolerance c2/c3 through.
        if (L_finite) {
            qL = q_of_z(L, c0, c1, 0.0, 0.0);
            if (!isfinite(qL)) {
                status[cell_id] = STATUS_NONFINITE_Q;
                for (int k = 0; k < (int)MOMENT_STRIDE; ++k) {
                    moment_output[out_base + (unsigned)k] = 0.0;
                }
                goto affine_broadcast_zero;
            }
            expL = exp(-qL);
        }
        if (R_finite) {
            qR = q_of_z(R, c0, c1, 0.0, 0.0);
            if (!isfinite(qR)) {
                status[cell_id] = STATUS_NONFINITE_Q;
                for (int k = 0; k < (int)MOMENT_STRIDE; ++k) {
                    moment_output[out_base + (unsigned)k] = 0.0;
                }
                goto affine_broadcast_zero;
            }
            expR = exp(-qR);
        }

        // M_0 via the normal-CDF closed form. s*(z - mu) is the standardized argument.
        double argL = L_finite ? s * (L - mu) : -CUDART_INF;
        double argR = R_finite ? s * (R - mu) :  CUDART_INF;
        // M_0 = exp(-alpha^2 / (2*(1+beta^2))) / sqrt(1+beta^2) * sqrt(2*pi)
        //       * [Phi(s*(R-mu)) - Phi(s*(L-mu))].
        double m0 = prefactor * (phi_cdf(argR) - phi_cdf(argL));

        double moms[MOMENT_STRIDE];
        moms[0] = m0;

        // M_{n+1} = (n * M_{n-1} - d0 * M_n - B_n) / d1, with M_{-1} treated as 0.
        // B_n = R^n * expR - L^n * expL; the infinite-endpoint contributions are
        // suppressed by setting expR / expL to zero above.
        double Rn = 1.0; // R^0
        double Ln = 1.0; // L^0
        for (int n = 0; n < (int)MOMENT_STRIDE - 1; ++n) {
            double M_nm1 = (n == 0) ? 0.0 : moms[n - 1];
            double M_n   = moms[n];
            double Bn_R = R_finite ? Rn * expR : 0.0;
            double Bn_L = L_finite ? Ln * expL : 0.0;
            double Bn = Bn_R - Bn_L;
            double next = (((double)n) * M_nm1 - d0 * M_n - Bn) / d1;
            moms[n + 1] = next;
            if (R_finite) Rn *= R;
            if (L_finite) Ln *= L;
        }

        for (int k = 0; k < (int)MOMENT_STRIDE; ++k) {
            moment_output[out_base + (unsigned)k] = moms[k];
        }
        status[cell_id] = STATUS_OK;

affine_broadcast_zero:
        ;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_includes_gl_table_and_kernel_decl_d9() {
        let src = build_cubic_deriv_moments_kernel_source(9);
        assert!(src.contains("__constant__ double GL_NODES"));
        assert!(src.contains("__constant__ double GL_WEIGHTS"));
        assert!(src.contains("cubic_deriv_moments_d9("));
        assert!(src.contains("MAX_DEGREE 9"));
        assert!(src.contains("__shfl_xor_sync"));
        // NVRTC is invoked with no `-I`, so the source must not pull in a CUDA
        // header it cannot find; `CUDART_INF` is supplied inline instead.
        assert!(
            !src.contains("math_constants.h"),
            "NVRTC cannot resolve <math_constants.h> without an include path"
        );
        assert!(src.contains("#define CUDART_INF"));
        assert!(src.contains("CUDART_INF"));
    }

    #[test]
    fn kernel_source_includes_gl_table_and_kernel_decl_d15() {
        let src = build_cubic_deriv_moments_kernel_source(15);
        assert!(src.contains("__constant__ double GL_NODES"));
        assert!(src.contains("cubic_deriv_moments_d15("));
        assert!(src.contains("MAX_DEGREE 15"));
        assert!(src.contains("__shfl_xor_sync"));
    }

    #[test]
    fn kernel_source_includes_gl_table_and_kernel_decl_d21() {
        let src = build_cubic_deriv_moments_kernel_source(21);
        assert!(src.contains("__constant__ double GL_NODES"));
        assert!(src.contains("cubic_deriv_moments_d21("));
        assert!(src.contains("MAX_DEGREE 21"));
        assert!(src.contains("__shfl_xor_sync"));
    }

    #[test]
    fn kernel_source_is_deterministic() {
        let a = build_cubic_deriv_moments_kernel_source(15);
        let b = build_cubic_deriv_moments_kernel_source(15);
        assert_eq!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn kernel_source_contains_full_384_node_table() {
        let src = build_cubic_deriv_moments_kernel_source(9);
        let pos = src.matches("e+").count() + src.matches("e-").count();
        assert!(
            pos >= 700,
            "expected at least 700 scientific-notation literals, found {pos}"
        );
    }
}
