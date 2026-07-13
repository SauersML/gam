// NVRTC device source for the survival marginal-slope rigid per-row V/G/H jet.
// This is the order-2 CUDA lowering of the canonical five-feature rigid row
// program plus its mechanical scalar/shared pullback. The generated schedule
// differentiates the sole likelihood SSA graph symbolically, computes only
// nonzero channels, and uses full f64 arithmetic without fast-math.

// NVRTC does not include <math.h>/<cmath>, so define the constants it omits.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
#endif
#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000LL))
#endif

// Full-precision probability leaves matching the CPU primitive contract.
__device__ __forceinline__ double erfcx_nn(double x) {
    if (!isfinite(x)) return x > 0.0 ? 0.0 : INFINITY;
    if (x <= 0.0) return 1.0;
    if (x < 26.0) return exp(fmin(x * x, 700.0)) * erfc(x);
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    double poly = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2
        - 1.875 * inv2 * inv2 * inv2
        + 6.5625 * inv2 * inv2 * inv2 * inv2;
    return inv * poly / sqrt(M_PI);
}

__device__ __forceinline__ double normal_pdf(double x) {
    const double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * exp(-0.5 * x * x);
}

__device__ __forceinline__ double normal_cdf(double x) {
    return 0.5 * erfc(-x / M_SQRT2);
}

__device__ __forceinline__ void sp_logcdf_mills(
        double x, double* log_cdf, double* mills) {
    if (x == INFINITY) {
        *log_cdf = 0.0;
        *mills = 0.0;
        return;
    }
    if (x == -INFINITY) {
        *log_cdf = -INFINITY;
        *mills = INFINITY;
        return;
    }
    if (isnan(x)) {
        *log_cdf = NAN;
        *mills = NAN;
        return;
    }
    if (x < 0.0) {
        double u = -x / M_SQRT2;
        double scaled = fmax(erfcx_nn(u), 1e-300);
        *log_cdf = -u * u + log(0.5 * scaled);
        *mills = sqrt(2.0 / M_PI) / scaled;
    } else {
        double cdf = fmin(fmax(normal_cdf(x), 1e-300), 1.0);
        *log_cdf = log(cdf);
        *mills = normal_pdf(x) / cdf;
    }
}

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
    sp_logcdf_mills(margin, &log_cdf, &mills);
    double k1 = -mills;
    double k2 = mills * (margin + mills);
    out[0] = -weight * log_cdf;
    out[1] = weight * k1;
    out[2] = weight * k2;
}

__device__ __forceinline__ void d_sqrt(double x, double out[3]) {
    double admitted = fmax(x, 1e-300);
    double root = sqrt(admitted);
    out[0] = root;
    out[1] = 0.5 / root;
    out[2] = -0.25 / (admitted * root);
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
    double di;
    double z_sum;
    double covariance_ones;
    double probit_scale;
};

// __GAM_ROW_PROGRAM_CUDA_VGH__

extern "C" __global__ void __launch_bounds__(128, 1) survival_rowjet_vgh(
        int n,
        const double* __restrict__ q0,
        const double* __restrict__ q1,
        const double* __restrict__ qd1,
        const double* __restrict__ g,
        const double* __restrict__ wi,
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
    in.di = di[row];
    in.z_sum = z_sum[row];
    in.covariance_ones = cov_ones[row];
    in.probit_scale = probit_scale;
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
