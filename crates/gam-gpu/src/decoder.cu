// The decoder layer's fused kernels (`tensor::cuda::decoder`): an RMS norm with its gain, the
// queries' and keys' per-head norms and rotation, and the MLP's activations, each forward and
// backward (attention itself is `attention.cu`). Inputs and cotangents are f32; a product's operand
// is written as bfloat16 (rounded to nearest, ties to even) where the next product reads it.
#define BLOCK 256
typedef unsigned long long u64;

__device__ __forceinline__ unsigned short to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    if (x != x) return (unsigned short)((bits >> 16) | 0x40u);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float from_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The sum of every thread's `v` in a block of BLOCK threads.
__device__ float block_sum(float v, float* shared) {
    v = warp_sum(v);
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = threadIdx.x < BLOCK / 32 ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) shared[0] = v;
    __syncthreads();
    v = shared[0];
    __syncthreads();
    return v;
}

// Row r of x (rows × d): k = 1/√(mean x² + ε), y = x k g. Writes k, and y in f32 (y32) and/or
// bfloat16 (y16) where those are not null. One block per row.
extern "C" __global__ void rms_gain(unsigned int rows, unsigned int d, float epsilon, const float* x, const float* gain, float* y32, unsigned short* y16, float* rstd) {
    __shared__ float shared[32];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (u64)r * d;
    float s = 0.0f;
    for (unsigned int c = threadIdx.x; c < d; c += BLOCK) s += xr[c] * xr[c];
    s = block_sum(s, shared);
    float k = rsqrtf(s / (float)d + epsilon);
    if (threadIdx.x == 0) rstd[r] = k;
    for (unsigned int c = threadIdx.x; c < d; c += BLOCK) {
        float y = xr[c] * k * gain[c];
        if (y32) y32[(u64)r * d + c] = y;
        if (y16) y16[(u64)r * d + c] = to_bf16(y);
    }
}

// The cotangent of `rms_gain`'s input added into gx: k gy g − (k³/d) x Σ gy g x.
extern "C" __global__ void rms_gain_backward(unsigned int rows, unsigned int d, const float* x, const float* gain, const float* rstd, const float* gy, float* gx) {
    __shared__ float shared[32];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (u64)r * d;
    const float* gr = gy + (u64)r * d;
    float dot = 0.0f;
    for (unsigned int c = threadIdx.x; c < d; c += BLOCK) dot += gr[c] * gain[c] * xr[c];
    dot = block_sum(dot, shared);
    float k = rstd[r];
    float coefficient = k * k * k * dot / (float)d;
    for (unsigned int c = threadIdx.x; c < d; c += BLOCK) gx[(u64)r * d + c] += k * gr[c] * gain[c] - coefficient * xr[c];
}

// The pair of coordinate `i` of a head's rotary plane `plane` (rotate-half: i and i + planes;
// interleaved: 2 plane and 2 plane + 1).
__device__ __forceinline__ void plane_pair(unsigned int plane, unsigned int planes, int half_split, unsigned int* a, unsigned int* b) {
    if (half_split) { *a = plane; *b = plane + planes; } else { *a = 2u * plane; *b = 2u * plane + 1u; }
}

// Projections p (rows × (q + 2 kv) heads of width w: the queries' heads, the keys', the values')
// into y (bfloat16, the same layout): each query and key head RMS-normed with its gain row (gains,
// (q + kv) × w; `normed` zero leaves them as they are, `rstd` (rows × (q + kv)) receives each
// norm's scale) and turned by its row's angles (cos, sin: rows × planes; the first 2 planes
// coordinates rotate); each value head copied. One warp per (row, head), the warps striding.
extern "C" __global__ void heads_rope(unsigned int rows, unsigned int q, unsigned int kv, unsigned int w, unsigned int planes, int half_split, int normed, float epsilon,
    const float* p, const float* gains, const float* cosines, const float* sines, unsigned short* y, float* rstd) {
    unsigned int heads = q + 2u * kv;
    unsigned int lane = threadIdx.x & 31u;
    for (u64 item = (u64)blockIdx.x * (BLOCK / 32) + (threadIdx.x >> 5); item < (u64)rows * heads; item += (u64)gridDim.x * (BLOCK / 32)) {
    unsigned int r = (unsigned int)(item / heads), h = (unsigned int)(item % heads);
    u64 base = ((u64)r * heads + h) * w;
    const float* x = p + base;
    unsigned short* out = y + base;
    if (h >= q + kv) {
        for (unsigned int i = lane; i < w; i += 32u) out[i] = to_bf16(x[i]);
        continue;
    }
    float k = 1.0f;
    if (normed) {
        float s = 0.0f;
        for (unsigned int i = lane; i < w; i += 32u) s += x[i] * x[i];
        s = warp_sum(s);
        k = rsqrtf(s / (float)w + epsilon);
        if (lane == 0) rstd[(u64)r * (q + kv) + h] = k;
    }
    const float* g = gains + (u64)h * w;
    const float* c = cosines + (u64)r * planes;
    const float* sn = sines + (u64)r * planes;
    for (unsigned int plane = lane; plane < planes; plane += 32u) {
        unsigned int a, b;
        plane_pair(plane, planes, half_split, &a, &b);
        float za = normed ? x[a] * k * g[a] : x[a], zb = normed ? x[b] * k * g[b] : x[b];
        out[a] = to_bf16(c[plane] * za - sn[plane] * zb);
        out[b] = to_bf16(sn[plane] * za + c[plane] * zb);
    }
    for (unsigned int i = 2u * planes + lane; i < w; i += 32u) out[i] = to_bf16(normed ? x[i] * k * g[i] : x[i]);
    }
}

// The cotangent of `heads_rope`'s input p from that of its output, gy (both f32, rows × heads·w).
extern "C" __global__ void heads_rope_backward(unsigned int rows, unsigned int q, unsigned int kv, unsigned int w, unsigned int planes, int half_split, int normed,
    const float* p, const float* gains, const float* cosines, const float* sines, const float* rstd, const float* gy, float* gp) {
    unsigned int heads = q + 2u * kv;
    unsigned int lane = threadIdx.x & 31u;
    for (u64 item = (u64)blockIdx.x * (BLOCK / 32) + (threadIdx.x >> 5); item < (u64)rows * heads; item += (u64)gridDim.x * (BLOCK / 32)) {
    unsigned int r = (unsigned int)(item / heads), h = (unsigned int)(item % heads);
    u64 base = ((u64)r * heads + h) * w;
    const float* x = p + base;
    const float* g = gy + base;
    float* out = gp + base;
    if (h >= q + kv) {
        for (unsigned int i = lane; i < w; i += 32u) out[i] = g[i];
        continue;
    }
    // The rotation's transpose turns the cotangent back: gz = Rᵀ g; the unrotated coordinates pass.
    const float* c = cosines + (u64)r * planes;
    const float* sn = sines + (u64)r * planes;
    for (unsigned int plane = lane; plane < planes; plane += 32u) {
        unsigned int a, b;
        plane_pair(plane, planes, half_split, &a, &b);
        float ga = g[a], gb = g[b];
        out[a] = c[plane] * ga + sn[plane] * gb;
        out[b] = -sn[plane] * ga + c[plane] * gb;
    }
    for (unsigned int i = 2u * planes + lane; i < w; i += 32u) out[i] = g[i];
    if (!normed) continue;
    __syncwarp();
    // The norm's: gx = k gz γ − (k³/w) x Σ gz γ x, in place.
    const float* gamma = gains + (u64)h * w;
    float k = rstd[(u64)r * (q + kv) + h];
    float dot = 0.0f;
    for (unsigned int i = lane; i < w; i += 32u) dot += out[i] * gamma[i] * x[i];
    dot = warp_sum(dot);
    float coefficient = k * k * k * dot / (float)w;
    __syncwarp();
    for (unsigned int i = lane; i < w; i += 32u) out[i] = k * out[i] * gamma[i] - coefficient * x[i];
    }
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// The gated MLP's activations from its two input products h (rows × 2m: the gates' m columns,
// then the inputs'): a = silu(gate) · input, as bfloat16 (rows × m).
extern "C" __global__ void swiglu(u64 rows, unsigned int m, const float* h, unsigned short* a) {
    u64 n = rows * m;
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) {
        u64 r = i / m, j = i % m;
        float g = h[r * 2 * m + j], u = h[r * 2 * m + m + j];
        a[i] = to_bf16(g * sigmoid(g) * u);
    }
}

// `swiglu`'s input cotangent (rows × 2m) from its output's (ga, rows × m).
extern "C" __global__ void swiglu_backward(u64 rows, unsigned int m, const float* h, const float* ga, float* gh) {
    u64 n = rows * m;
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) {
        u64 r = i / m, j = i % m;
        float g = h[r * 2 * m + j], u = h[r * 2 * m + m + j], s = sigmoid(g), d = ga[i];
        gh[r * 2 * m + j] = d * u * s * (1.0f + g * (1.0f - s));
        gh[r * 2 * m + m + j] = d * g * s;
    }
}

// GELU in its tanh form of h plus a bias row (none when null), as bfloat16 (rows × m).
extern "C" __global__ void gelu_tanh(u64 rows, unsigned int m, const float* h, const float* bias, unsigned short* a) {
    const float c = 0.7978845608028654f, k = 0.044715f;
    u64 n = rows * m;
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) {
        float x = h[i] + (bias ? bias[i % m] : 0.0f);
        a[i] = to_bf16(0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x))));
    }
}

extern "C" __global__ void gelu_tanh_backward(u64 rows, unsigned int m, const float* h, const float* bias, const float* ga, float* gh) {
    const float c = 0.7978845608028654f, k = 0.044715f;
    u64 n = rows * m;
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) {
        float x = h[i] + (bias ? bias[i % m] : 0.0f);
        float t = tanhf(c * (x + k * x * x * x));
        gh[i] = ga[i] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x));
    }
}
