//! Metal Shading Language sources for the Apple-GPU backend.
//!
//! One library, compiled once per process at probe time with the safe math
//! mode (no reassociation, no fast-math substitutions) and precise math
//! functions. `#pragma clang fp contract(off)` additionally forbids fusing a
//! separate `a*b + c` into an fma, so every fused operation in these sources
//! is an explicit `fma(...)` call and every other `+`/`*` is one rounded
//! binary32 operation: the operation counts the bands in
//! `crate::precision_bounds` rest on are exactly the operations written here.

/// Names of the kernel functions the probe builds pipelines for.
pub(crate) const KERNELS: &[&str] = &[GEMM_DF64, SELF_TEST_DF64];

pub(crate) const GEMM_DF64: &str = "gam_gemm_df64";
pub(crate) const SELF_TEST_DF64: &str = "gam_self_test_df64";

/// Output tile of the df64 GEMM: a `16 × 16` threadgroup computes a `32 × 32`
/// block, two by two outputs per thread.
pub(crate) const GEMM_DF64_TILE: usize = 32;
pub(crate) const GEMM_DF64_THREADS_PER_SIDE: usize = 16;

pub(crate) const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#pragma clang fp contract(off)

// ---------------------------------------------------------------------------
// Double-float (df64) primitives. A df64 is float2(hi, lo) with hi + lo the
// value and |lo| <= ulp(hi)/2. Algorithms and bounds: Joldes, Muller, Popescu,
// ACM TOMS 44(2) 2017, corrected by Muller and Rideau, ACM TOMS 48(1) 2022.
// ---------------------------------------------------------------------------

// Knuth's TwoSum: s + e == a + b exactly (no branch, any ordering).
inline float2 df_two_sum(float a, float b) {
    float s  = a + b;
    float bp = s - a;
    float ap = s - bp;
    float db = b - bp;
    float da = a - ap;
    return float2(s, da + db);
}

// Dekker's Fast2Sum, exact when |a| >= |b| (or a == 0).
inline float2 df_fast_two_sum(float a, float b) {
    float s = a + b;
    float z = s - a;
    return float2(s, b - z);
}

// Accurate DWPlusDW (JMP Algorithm 6): relative error <= 3u^2/(1-4u).
inline float2 df_add(float2 x, float2 y) {
    float2 s = df_two_sum(x.x, y.x);
    float2 t = df_two_sum(x.y, y.y);
    float  c = s.y + t.x;
    float2 v = df_fast_two_sum(s.x, c);
    float  w = t.y + v.y;
    return df_fast_two_sum(v.x, w);
}

// DWTimesDW3 (JMP Algorithm 12, fma): relative error <= 5u^2.
inline float2 df_mul(float2 x, float2 y) {
    float ch  = x.x * y.x;
    float cl1 = fma(x.x, y.x, -ch);
    float tl0 = x.y * y.y;
    float tl1 = fma(x.x, y.y, tl0);
    float cl2 = fma(x.y, y.x, tl1);
    float cl3 = cl1 + cl2;
    return df_fast_two_sum(ch, cl3);
}

// Row-major C[b] = A[b] (m x k) * op(B[b]), op(B) = B (k x n, ldb = row
// stride) or, with trans_b, the transpose of a stored n x k matrix (ldb = its
// row stride). A zero batch stride broadcasts.
struct GemmParams {
    uint m; uint n; uint k;
    uint lda; uint ldb; uint ldc;
    uint accumulate; uint trans_b;
    ulong stride_a; ulong stride_b; ulong stride_c;
};

// ---------------------------------------------------------------------------
// Batched df64 GEMM: each thread accumulates a 2 x 2 block of dot products in
// double-float, one df_mul and one df_add per term, in order l = 0..k-1.
// ---------------------------------------------------------------------------

constant constexpr uint DF_TILE = 32;
constant constexpr uint DF_BK = 16;

kernel void gam_gemm_df64(
    device const float2* A       [[buffer(0)]],
    device const float2* B       [[buffer(1)]],
    device float2*       C       [[buffer(2)]],
    constant GemmParams& p       [[buffer(3)]],
    uint3 tg                     [[threadgroup_position_in_grid]],
    uint3 lt                     [[thread_position_in_threadgroup]])
{
    threadgroup float2 As[DF_TILE * DF_BK];
    threadgroup float2 Bs[DF_BK * DF_TILE];

    A += (ulong)tg.z * p.stride_a;
    B += (ulong)tg.z * p.stride_b;
    C += (ulong)tg.z * p.stride_c;
    const uint row0 = tg.y * DF_TILE;
    const uint col0 = tg.x * DF_TILE;
    const uint tid = lt.y * 16 + lt.x;

    float2 acc00 = float2(0.0f), acc01 = float2(0.0f);
    float2 acc10 = float2(0.0f), acc11 = float2(0.0f);

    for (uint k0 = 0; k0 < p.k; k0 += DF_BK) {
        for (uint e = tid; e < DF_TILE * DF_BK; e += 256) {
            uint r = e / DF_BK, c = e % DF_BK;
            uint gr = row0 + r, gc = k0 + c;
            As[e] = (gr < p.m && gc < p.k) ? A[(ulong)gr * p.lda + gc] : float2(0.0f);
        }
        for (uint e = tid; e < DF_BK * DF_TILE; e += 256) {
            uint r = e / DF_TILE, c = e % DF_TILE;
            uint gr = k0 + r, gc = col0 + c;
            ulong at = p.trans_b == 0u ? (ulong)gr * p.ldb + gc : (ulong)gc * p.ldb + gr;
            Bs[e] = (gr < p.k && gc < p.n) ? B[at] : float2(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint kmax = min(DF_BK, p.k - k0);
        for (uint l = 0; l < kmax; ++l) {
            float2 a0 = As[(2 * lt.y) * DF_BK + l];
            float2 a1 = As[(2 * lt.y + 1) * DF_BK + l];
            float2 b0 = Bs[l * DF_TILE + 2 * lt.x];
            float2 b1 = Bs[l * DF_TILE + 2 * lt.x + 1];
            acc00 = df_add(acc00, df_mul(a0, b0));
            acc01 = df_add(acc01, df_mul(a0, b1));
            acc10 = df_add(acc10, df_mul(a1, b0));
            acc11 = df_add(acc11, df_mul(a1, b1));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint r = row0 + 2 * lt.y;
    const uint c = col0 + 2 * lt.x;
    if (r < p.m && c < p.n)         C[(ulong)r * p.ldc + c] = acc00;
    if (r < p.m && c + 1 < p.n)     C[(ulong)r * p.ldc + c + 1] = acc01;
    if (r + 1 < p.m && c < p.n)     C[(ulong)(r + 1) * p.ldc + c] = acc10;
    if (r + 1 < p.m && c + 1 < p.n) C[(ulong)(r + 1) * p.ldc + c + 1] = acc11;
}

// ---------------------------------------------------------------------------
// Probe-time self-test of the df64 primitives: for each (a, b) pair, write
// TwoSum and the fma residual of the product. The host checks both are exact,
// which fails if the compiler reassociated or the device fused/ rounded
// differently from IEEE binary32.
// ---------------------------------------------------------------------------

kernel void gam_self_test_df64(
    device const float2* ab      [[buffer(0)]],
    device float4*       out     [[buffer(1)]],
    constant uint&       count   [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float a = ab[gid].x, b = ab[gid].y;
    float2 s = df_two_sum(a, b);
    float  p = a * b;
    float  e = fma(a, b, -p);
    out[gid] = float4(s.x, s.y, p, e);
}
"#;
