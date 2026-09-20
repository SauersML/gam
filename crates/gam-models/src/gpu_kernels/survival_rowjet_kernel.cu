// NVRTC device source for the survival marginal-slope rigid per-row V/G/H jet.
// This is the order-2 CUDA lowering of the canonical five-feature rigid row
// program plus its mechanical scalar/shared pullback. The generated schedule
// differentiates the sole likelihood SSA graph symbolically, computes only
// nonzero channels, and uses full f64 arithmetic without fast-math.

// NVRTC does not include <math.h>/<cmath>, so define the constants it omits.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
#endif
#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000LL))
#endif

// The probability leaves are `gam_gpu::numerics_device::PROBIT_NUMERICS_CU`,
// which `survival_rowjet_source` prepends: the shared device copy of the CPU
// `normal_logcdf_derivatives` contract (exact-square erfcx, log1p right tail,
// and the Laplace continued fraction for the deep-left-tail curvature).

// Order-2 unary stacks: [f, f', f''].
__device__ __forceinline__ void neglog_phi_stack(
        double margin, double weight, double out[3]) {
    if (weight == 0.0 || margin == INFINITY) {
        out[0] = out[1] = out[2] = 0.0;
        return;
    }
    if (margin == -INFINITY) {
        out[0] = INFINITY;
        out[1] = -INFINITY;
        out[2] = weight;
        return;
    }
    if (isnan(margin)) {
        out[0] = out[1] = out[2] = NAN;
        return;
    }
    double log_cdf;
    double mills;
    double curvature;
    log_ndtr_mills_curvature(margin, &log_cdf, &mills, &curvature);
    out[0] = -weight * log_cdf;
    out[1] = -weight * mills;
    out[2] = weight * curvature;
}

// No floor on the radicand, as on the host (`unary_derivatives_sqrt`,
// `unary_derivatives_inverse_sqrt`): a corrupted argument surfaces as a
// non-finite channel instead of a fabricated finite derivative.
__device__ __forceinline__ void d_sqrt(double x, double out[3]) {
    double root = sqrt(x);
    out[0] = root;
    out[1] = 0.5 / root;
    out[2] = -0.25 / (x * root);
}

__device__ __forceinline__ void d_inverse_sqrt(double x, double out[3]) {
    // `1.0 / sqrt(x)` and NOT the `rsqrt` intrinsic: the host leaf
    // (`unary_derivatives_inverse_sqrt`) is an IEEE divide of an IEEE square
    // root, and the device/host parity test compares these bit for bit.
    double reciprocal_root = 1.0 / sqrt(x);
    out[0] = reciprocal_root;
    out[1] = -0.5 * reciprocal_root / x;
    out[2] = 0.75 * reciprocal_root / (x * x);
}

__device__ __forceinline__ void d_log(double x, double out[3]) {
    out[0] = log(x);
    out[1] = 1.0 / x;
    out[2] = -1.0 / (x * x);
}

__device__ __forceinline__ void d_lognormpdf(double x, double out[3]) {
    double constant = 0.5 * log(2.0 * M_PI);
    out[0] = -0.5 * x * x - constant;
    out[1] = -x;
    out[2] = -1.0;
}

struct RowIn {
    double wi;
    // Weight of the entry survival factor: zero for a row entering at the time
    // origin, where S(0) = 1 (gnomon#2336).
    double wi_entry;
    double di;
    double z_sum;
    double covariance_ones;
    double probit_scale;
    // The row program's activity constant for the slope-rate terms of eta'_1;
    // this kernel evaluates the time-constant slope frame only.
    double follow_up_varying;
};

// __GAM_ROW_PROGRAM_CUDA_VGH__

extern "C" __global__ void __launch_bounds__(128, 1) survival_rowjet_vgh(
        int n,
        const double* __restrict__ q0,
        const double* __restrict__ q1,
        const double* __restrict__ qd1,
        const double* __restrict__ g,
        const double* __restrict__ wi,
        const double* __restrict__ wi_entry,
        const double* __restrict__ di,
        const double* __restrict__ z_sum,
        const double* __restrict__ cov_ones,
        double probit_scale,
        double* __restrict__ out_value,
        double* __restrict__ out_gradient,
        double* __restrict__ out_hessian) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    RowIn in;
    in.wi = wi[row];
    in.wi_entry = wi_entry[row];
    in.di = di[row];
    in.z_sum = z_sum[row];
    in.covariance_ones = cov_ones[row];
    in.probit_scale = probit_scale;
    in.follow_up_varying = 0.0;
    rigid_feature_program_pullback4(
        q0[row],
        q1[row],
        qd1[row],
        g[row],
        in,
        &out_value[row],
        &out_gradient[(size_t)row * 4],
        &out_hessian[(size_t)row * 16]);
}
