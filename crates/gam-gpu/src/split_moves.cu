// The operand terms of the split products (`Arithmetic::Tf32x3`, `Arithmetic::Bf16x3`) and the row
// moves of `Device::gather_ranges` and `Device::scatter_ranges` (`tensor::cuda`'s `split_moves` module).
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

// An f32 operand of `Arithmetic::Bf16x3` as two bfloat16 terms, hi = bf16(x) and lo = bf16(x − hi).
extern "C" __global__ void bf16_split(u64 n, const float* x, unsigned short* hi, unsigned short* lo) {
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) {
        unsigned short h = to_bf16(x[i]);
        hi[i] = h;
        lo[i] = to_bf16(x[i] - from_bf16(h));
    }
}

// An f32 operand of `Arithmetic::Tf32x3` as two f32 terms: big, x rounded to TF32 (10 stored
// significand bits, to nearest, ties to even), exact in TF32 however the tensor cores read it; and
// small = x − big, exact in f32, which the tensor cores read to within 2⁻¹⁰ of itself.
extern "C" __global__ void tf32_split(u64 n, const float* x, float* big, float* small) {
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) {
        float v = x[i];
        unsigned int bits = __float_as_uint(v);
        float b = (v != v || isinf(v)) ? v : __uint_as_float((bits + 0x0fffu + ((bits >> 13) & 1u)) & 0xffffe000u);
        big[i] = b;
        small[i] = (v != v || isinf(v)) ? 0.0f : v - b;
    }
}

#define ROW_RANGES 320
// The moves of one launch: rows from[i]..from[i] + length[i] of x go to rows to[i].. of y.
struct RowRanges {
    unsigned int count;
    unsigned int from[ROW_RANGES];
    unsigned int to[ROW_RANGES];
    unsigned int length[ROW_RANGES];
};

// Rows of x copied into y (cols columns each), move blockIdx.y of `ranges`, the blocks striding
// over its entries.
extern "C" __global__ void copy_ranges(const RowRanges ranges, unsigned int cols, const float* x, float* y) {
    unsigned int m = blockIdx.y;
    if (m >= ranges.count) return;
    u64 n = (u64)ranges.length[m] * cols;
    const float* source = x + (u64)ranges.from[m] * cols;
    float* target = y + (u64)ranges.to[m] * cols;
    for (u64 i = (u64)blockIdx.x * BLOCK + threadIdx.x; i < n; i += (u64)gridDim.x * BLOCK) target[i] = source[i];
}
