// Causal attention of an attention layer's heads (`Device::causal_attention`), forward and reverse,
// tiled so that no sequence's scores leave the chip: each block of threads holds one tile of query
// rows (forward) or of key rows (reverse) in shared memory and sweeps the other side's tiles past
// it, with the row maxima and sums of the softmax carried across tiles (the online softmax of
// Milakov and Gimelshein, 2018, as tiled by FlashAttention, Dao et al., 2022, and Dao, 2023). The
// products run on the tensor cores (mma.sync m16n8k16, bfloat16 operands, f32 accumulation; sm_80
// and later).
//
// The heads `y` are bfloat16, row-major, one row per position: the query heads, then the key heads,
// then the value heads, each HEAD_W columns. Query head h reads key-value head h / (hq / hk). The
// rows hold sequences given as (start row, length) pairs; position i of a sequence attends to
// positions 0..=i of the same sequence. The source is compiled once per head width with HEAD_W
// defined; heads are padded with zeros to HEAD_D columns (64 or 128) inside the kernels.
//
// Forward: out = softmax(scale q kᵀ) v per query head (bfloat16, rows × hq·HEAD_W), and each row's
// log partition lse = log Σⱼ exp(scale qᵢ·kⱼ) per query head (f32, rows × hq). The weights are
// rounded to bfloat16 before the values' product reads them, and the row sums add the rounded
// weights, so each output row is a convex combination of the value rows.
// Reverse: with Dᵢ = Σ dOᵢ·Oᵢ, P = exp(scale q kᵀ − lse) and dS = P ∘ (dO vᵀ − D):
//   dV = Pᵀ dO, dK = scale dSᵀ Q, dQ = scale dS K, each key-value head's summed over its query heads,
// in two passes that each own what they write: one over blocks of key rows (dK, dV), one over
// blocks of query rows (dQ), each recomputing the weights from lse. Neither adds into memory another
// block writes.
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;

#define HEAD_D (HEAD_W <= 64 ? 64 : 128)
// 16-byte chunks per row of a head tile, and of a score tile (64 columns).
#define CHUNKS (HEAD_D / 8)

// The sequences of one launch, passed by value (a kernel's parameters hold 4 KB). The tiles' sizes
// and MAX_SEQUENCES are defined by the caller (`tensor::cuda`), which launches with them: a block of
// query rows has ROWS_WARPS warps of 16 rows and takes FORWARD_KEYS keys at a time in the forward,
// ROWS_KEYS in the queries' reverse; a block of key rows has KEYS_WARPS warps of 16 keys and takes
// KEYS_ROWS query rows at a time (the keys' and values' reverse).
struct Sequences {
    u32 count;
    u32 start[MAX_SEQUENCES];
    u32 length[MAX_SEQUENCES];
};

#define NEG_INF (-__int_as_float(0x7f800000))
#define POS_INF (__int_as_float(0x7f800000))

__device__ __forceinline__ unsigned short to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    if (x != x) return (unsigned short)((bits >> 16) | 0x40u);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float from_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// Two f32 values as a bfloat16 pair (rounded to nearest, ties to even), `lo` in the low half.
__device__ __forceinline__ u32 pack_bf16(float lo, float hi) {
    u32 d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ float bf16_lo(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(u32 v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ u32 shared_address(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

// The byte offset of element (r, c) in a shared tile of `chunks` 16-byte chunks per row, each row's
// chunks permuted by r mod 8 (an exclusive or), so that the eight rows one ldmatrix reads at a
// column fall in different banks.
__device__ __forceinline__ u32 swizzle(u32 r, u32 c, u32 chunks) {
    return (r * chunks + ((c >> 3) ^ (r & 7u))) * 16u + (c & 7u) * 2u;
}

__device__ __forceinline__ void ldsm(u32 address, u32 (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

__device__ __forceinline__ void ldsm_t(u32 address, u32 (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

// c += a b on one 16 × 8 × 16 tile: a (16 × 16, row-major fragment), b (16 × 8, column-major).
__device__ __forceinline__ void mma(float (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(u32 dst, const void* src, u32 bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// Rows [0, R) of a head tile from `src` (its row 0, column 0; `stride` elements between rows) into
// the shared tile at `tile`: rows from `valid` on and columns from HEAD_W on are zeros.
template <int R, int THREADS>
__device__ __forceinline__ void load_head_tile(u32 tile, const u16* src, u64 stride, u32 valid) {
#pragma unroll
    for (u32 i = threadIdx.x; i < R * CHUNKS; i += THREADS) {
        u32 r = i / CHUNKS, c = (i % CHUNKS) * 8u;
        u32 dst = tile + swizzle(r, c, CHUNKS);
#if HEAD_W % 8 == 0
        bool inside = r < valid && c < HEAD_W;
        cp_async16(dst, inside ? (const void*)(src + (u64)r * stride + c) : (const void*)src, inside ? 16u : 0u);
#else
        u32 v[4];
#pragma unroll
        for (u32 k = 0; k < 4; ++k) {
            u32 c0 = c + 2u * k;
            float a = r < valid && c0 < HEAD_W ? from_bf16(src[(u64)r * stride + c0]) : 0.0f;
            float b = r < valid && c0 + 1u < HEAD_W ? from_bf16(src[(u64)r * stride + c0 + 1u]) : 0.0f;
            v[k] = pack_bf16(a, b);
        }
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(dst), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
#endif
    }
}

// acc[MT][NT] += A B for a warp: A is (16 MT) × (16 KS) from the shared tile `a` (`ach` chunks per
// row) at (am0, ak0), stored as A when !AT and as Aᵀ when AT; B is (16 KS) × (8 NT) from `b` at
// (bk0, bn0), stored as Bᵀ (rows n) when !BT and as B (rows k) when BT. NT is even.
template <int MT, int NT, int KS, bool AT, bool BT>
__device__ __forceinline__ void warp_mma(float (&acc)[MT][NT][4], u32 a, u32 ach, u32 am0, u32 ak0, u32 b, u32 bch, u32 bk0, u32 bn0, u32 lane) {
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
        u32 af[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            u32 m0 = am0 + mt * 16, k0 = ak0 + ks * 16;
            if (!AT) {
                ldsm(a + swizzle(m0 + (lane & 15u), k0 + ((lane >> 4) << 3), ach), af[mt]);
            } else {
                ldsm_t(a + swizzle(k0 + (lane & 7u) + ((lane >> 4) << 3), m0 + (((lane >> 3) & 1u) << 3), ach), af[mt]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt += 2) {
            u32 n0 = bn0 + nt * 8, k0 = bk0 + ks * 16;
            u32 bf[4];
            if (!BT) {
                ldsm(b + swizzle(n0 + (lane & 7u) + ((lane >> 4) << 3), k0 + (((lane >> 3) & 1u) << 3), bch), bf);
            } else {
                ldsm_t(b + swizzle(k0 + (lane & 7u) + (((lane >> 3) & 1u) << 3), n0 + ((lane >> 4) << 3), bch), bf);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                mma(acc[mt][nt], af[mt], bf[0], bf[1]);
                mma(acc[mt][nt + 1], af[mt], bf[2], bf[3]);
            }
        }
    }
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ---- Forward ----
// A block holds ROWS_TILE query rows of one query head of one sequence (16 per warp, their query
// fragments in registers) and sweeps the key and value tiles of FORWARD_KEYS rows up to its last row,
// loading the value tile while the scores' products run and the next key tile while the values'
// run.
#define ROWS_THREADS (ROWS_WARPS * 32)
#define ROWS_TILE (ROWS_WARPS * 16)

// Blocks: x + z gridDim.x = sequence × hq + query head (the pairs in chunks of gridDim.x, whose keys
// and values stay in L2 while the chunk runs); y = the query tile, counted from the sequence's last
// (the longest sweep first). Dynamic shared memory: (ROWS_TILE + 2 FORWARD_KEYS) HEAD_D bfloat16.
extern "C" __global__ void __launch_bounds__(ROWS_THREADS) attention_forward(const Sequences sequences, u32 hq, u32 hk, float scale_log2, const u16* __restrict__ y, u16* __restrict__ out, float* __restrict__ lse) {
    extern __shared__ __align__(128) unsigned char smem[];
    const u32 pair = blockIdx.z * gridDim.x + blockIdx.x;
    if (pair >= sequences.count * hq) return;
    const u32 head = pair % hq, sequence = pair / hq;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 tiles = (length + ROWS_TILE - 1) / ROWS_TILE;
    if (blockIdx.y >= tiles) return;
    const u32 q0 = (tiles - 1 - blockIdx.y) * ROWS_TILE;
    const u32 stride = (hq + 2 * hk) * HEAD_W, kv = head / (hq / hk);
    const u16* yq = y + (u64)(start + q0) * stride + head * HEAD_W;
    const u16* yk = y + (u64)start * stride + (hq + kv) * HEAD_W;
    const u16* yv = y + (u64)start * stride + (hq + hk + kv) * HEAD_W;
    const u32 sq = shared_address(smem), sk = sq + ROWS_TILE * HEAD_D * 2, sv = sk + FORWARD_KEYS * HEAD_D * 2;
    const u32 keys = min(q0 + ROWS_TILE, length), key_tiles = (keys + FORWARD_KEYS - 1) / FORWARD_KEYS;
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;

    load_head_tile<ROWS_TILE, ROWS_THREADS>(sq, yq, stride, min((u32)ROWS_TILE, length - q0));
    load_head_tile<FORWARD_KEYS, ROWS_THREADS>(sk, yk, stride, min((u32)FORWARD_KEYS, keys));
    cp_commit();

    u32 qf[HEAD_D / 16][4];
    float o[HEAD_D / 8][4];
#pragma unroll
    for (int n = 0; n < HEAD_D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};
    // This thread's rows (relative to the sequence) and columns within a score tile.
    const u32 row0 = q0 + warp * 16 + (lane >> 2), col0 = 2 * (lane & 3u);

    for (u32 kt = 0; kt < key_tiles; ++kt) {
        const u32 k0 = kt * FORWARD_KEYS;
        cp_wait_all();
        __syncthreads();
        if (kt == 0) {
#pragma unroll
            for (int ks = 0; ks < HEAD_D / 16; ++ks) ldsm(sq + swizzle(warp * 16 + (lane & 15u), ks * 16 + ((lane >> 4) << 3), CHUNKS), qf[ks]);
        }
        load_head_tile<FORWARD_KEYS, ROWS_THREADS>(sv, yv + (u64)k0 * stride, stride, min((u32)FORWARD_KEYS, keys - k0));
        cp_commit();
        float s[FORWARD_KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < FORWARD_KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < HEAD_D / 16; ++ks) {
#pragma unroll
            for (int nt = 0; nt < FORWARD_KEYS / 8; nt += 2) {
                u32 bf[4];
                ldsm(sk + swizzle(nt * 8 + (lane & 7u) + ((lane >> 4) << 3), ks * 16 + (((lane >> 3) & 1u) << 3), CHUNKS), bf);
                mma(s[nt], qf[ks], bf[0], bf[1]);
                mma(s[nt + 1], qf[ks], bf[2], bf[3]);
            }
        }
        cp_wait_all();
        __syncthreads();
        if (kt + 1 < key_tiles) {
            load_head_tile<FORWARD_KEYS, ROWS_THREADS>(sk, yk + (u64)(k0 + FORWARD_KEYS) * stride, stride, min((u32)FORWARD_KEYS, keys - k0 - FORWARD_KEYS));
            cp_commit();
        }
        // Scores in base-2 units; keys after a row's position, or past the sequence, weigh nothing.
        const bool masked = k0 + FORWARD_KEYS > q0 + warp * 16 + 1 || k0 + FORWARD_KEYS > length;
#pragma unroll
        for (int n = 0; n < FORWARD_KEYS / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[n][e] * scale_log2;
                if (masked) {
                    u32 key = k0 + n * 8 + col0 + (e & 1), row = row0 + 8 * (e >> 1);
                    if (key > row || key >= length) v = NEG_INF;
                }
                s[n][e] = v;
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float mx = m[i];
#pragma unroll
            for (int n = 0; n < FORWARD_KEYS / 8; ++n) mx = fmaxf(mx, fmaxf(s[n][2 * i], s[n][2 * i + 1]));
            mx = quad_max(mx);
            const float base = mx == NEG_INF ? 0.0f : mx;
            const float alpha = exp2_approx(m[i] - base);
            m[i] = mx;
            l[i] *= alpha;
#pragma unroll
            for (int n = 0; n < HEAD_D / 8; ++n) {
                o[n][2 * i] *= alpha;
                o[n][2 * i + 1] *= alpha;
            }
#pragma unroll
            for (int n = 0; n < FORWARD_KEYS / 8; ++n) {
                s[n][2 * i] = exp2_approx(s[n][2 * i] - base);
                s[n][2 * i + 1] = exp2_approx(s[n][2 * i + 1] - base);
            }
        }
        // The weights as the values' product's operand (bfloat16); the row sums add the same.
#pragma unroll
        for (int kk = 0; kk < FORWARD_KEYS / 16; ++kk) {
            u32 a[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]), pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            l[0] += bf16_lo(a[0]) + bf16_hi(a[0]) + bf16_lo(a[2]) + bf16_hi(a[2]);
            l[1] += bf16_lo(a[1]) + bf16_hi(a[1]) + bf16_lo(a[3]) + bf16_hi(a[3]);
#pragma unroll
            for (int nd = 0; nd < HEAD_D / 8; nd += 2) {
                u32 bf[4];
                ldsm_t(sv + swizzle(kk * 16 + (lane & 7u) + (((lane >> 3) & 1u) << 3), nd * 8 + ((lane >> 4) << 3), CHUNKS), bf);
                mma(o[nd], a, bf[0], bf[1]);
                mma(o[nd + 1], a, bf[2], bf[3]);
            }
        }
    }
    const u32 reads = hq * HEAD_W;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float total = quad_sum(l[i]);
        const u32 row = row0 + 8 * i;
        if (row >= length) continue;
        const float inverse = 1.0f / total;
        u16* dst = out + (u64)(start + row) * reads + head * HEAD_W;
#pragma unroll
        for (int n = 0; n < HEAD_D / 8; ++n) {
            const u32 c = n * 8 + col0;
#if HEAD_W % 8 == 0
            if (c < HEAD_W) *(u32*)(dst + c) = pack_bf16(o[n][2 * i] * inverse, o[n][2 * i + 1] * inverse);
#else
            if (c < HEAD_W) dst[c] = to_bf16(o[n][2 * i] * inverse);
            if (c + 1 < HEAD_W) dst[c + 1] = to_bf16(o[n][2 * i + 1] * inverse);
#endif
        }
        if ((lane & 3u) == 0) lse[(u64)(start + row) * hq + head] = (m[i] + __log2f(total)) * 0.6931471805599453f;
    }
}

// ---- Reverse ----
// D = Σ dO·O per row and query head (rows × hq, f32), and dO as bfloat16 (the products' operand).
// One warp per (row, head).
extern "C" __global__ void attention_backward_sums(u32 rows, u32 hq, const u16* __restrict__ out, const float* __restrict__ ga, u16* __restrict__ ga16, float* __restrict__ dsum) {
    const u32 lane = threadIdx.x & 31u;
    for (u64 item = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5; item < (u64)rows * hq; item += ((u64)gridDim.x * blockDim.x) >> 5) {
        const u64 base = item * HEAD_W;
        float s = 0.0f;
        for (u32 t = lane; t < HEAD_W; t += 32u) {
            const float g = ga[base + t];
            s += g * from_bf16(out[base + t]);
            ga16[base + t] = to_bf16(g);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) dsum[item] = s;
    }
}

// ---- Reverse: the keys' and values' cotangents ----
// A block holds KEYS_TILE key rows of one key-value head of one sequence (16 per warp) and, for
// each query head of the group, sweeps the query tiles of KEYS_ROWS rows from its own diagonal on,
// the next tile loading while this one's products run. Each warp forms its keys' transposed scores
// Sᵀ = K Qᵀ, Pᵀ, dPᵀ = V dOᵀ and dSᵀ in registers, where they already are the operands of
// dV += Pᵀ dO and dK += dSᵀ Q; dK and dV stay in registers until the end.
#define KEYS_THREADS (KEYS_WARPS * 32)
#define KEYS_TILE (KEYS_WARPS * 16)

// Blocks: x + z gridDim.x = sequence × hk + key-value head (in chunks, as the forward's); y = the key
// tile (the first, the longest sweep, first). Dynamic shared memory: (2 KEYS_TILE + 4 KEYS_ROWS)
// HEAD_D bfloat16.
extern "C" __global__ void __launch_bounds__(KEYS_THREADS) attention_backward_keys(const Sequences sequences, u32 hq, u32 hk, float scale, const u16* __restrict__ y, const float* __restrict__ lse, const u16* __restrict__ ga16, const float* __restrict__ dsum, float* __restrict__ gy) {
    extern __shared__ __align__(128) unsigned char smem[];
    const u32 pair = blockIdx.z * gridDim.x + blockIdx.x;
    if (pair >= sequences.count * hk) return;
    const u32 kv = pair % hk, sequence = pair / hk;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 k0 = blockIdx.y * KEYS_TILE;
    if (k0 >= length) return;
    const u32 stride = (hq + 2 * hk) * HEAD_W, reads = hq * HEAD_W, group = hq / hk;
    const float scale_log2 = scale * 1.4426950408889634f;
    const u32 row_bytes = HEAD_D * 2;
    const u32 sk = shared_address(smem), sv = sk + KEYS_TILE * row_bytes, sq = sv + KEYS_TILE * row_bytes, sdo = sq + 2 * KEYS_ROWS * row_bytes;
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const u32 first = k0 / KEYS_ROWS, per_head = (length + KEYS_ROWS - 1) / KEYS_ROWS - first, steps = group * per_head;

    load_head_tile<KEYS_TILE, KEYS_THREADS>(sk, y + (u64)(start + k0) * stride + (hq + kv) * HEAD_W, stride, min((u32)KEYS_TILE, length - k0));
    load_head_tile<KEYS_TILE, KEYS_THREADS>(sv, y + (u64)(start + k0) * stride + (hq + hk + kv) * HEAD_W, stride, min((u32)KEYS_TILE, length - k0));
    // Step t reads query head kv·group + t / per_head, rows from (first + t mod per_head) KEYS_ROWS.
    {
        const u32 h = kv * group, q0 = first * KEYS_ROWS;
        load_head_tile<KEYS_ROWS, KEYS_THREADS>(sq, y + (u64)(start + q0) * stride + h * HEAD_W, stride, min((u32)KEYS_ROWS, length - q0));
        load_head_tile<KEYS_ROWS, KEYS_THREADS>(sdo, ga16 + (u64)(start + q0) * reads + h * HEAD_W, reads, min((u32)KEYS_ROWS, length - q0));
    }
    cp_commit();

    float dk[HEAD_D / 8][4], dv[HEAD_D / 8][4];
#pragma unroll
    for (int n = 0; n < HEAD_D / 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) dk[n][e] = dv[n][e] = 0.0f;
    // This warp's first key and this thread's keys (rows of Sᵀ) and rows (its columns).
    const u32 warp_key = k0 + warp * 16, key_a = warp_key + (lane >> 2), col0 = 2 * (lane & 3u);

    for (u32 t = 0; t < steps; ++t) {
        const u32 h = kv * group + t / per_head, q0 = (first + t % per_head) * KEYS_ROWS, stage = t & 1u;
        cp_wait_all();
        __syncthreads();
        if (t + 1 < steps) {
            const u32 nh = kv * group + (t + 1) / per_head, nq0 = (first + (t + 1) % per_head) * KEYS_ROWS;
            load_head_tile<KEYS_ROWS, KEYS_THREADS>(sq + (stage ^ 1u) * KEYS_ROWS * row_bytes, y + (u64)(start + nq0) * stride + nh * HEAD_W, stride, min((u32)KEYS_ROWS, length - nq0));
            load_head_tile<KEYS_ROWS, KEYS_THREADS>(sdo + (stage ^ 1u) * KEYS_ROWS * row_bytes, ga16 + (u64)(start + nq0) * reads + nh * HEAD_W, reads, min((u32)KEYS_ROWS, length - nq0));
            cp_commit();
        }
        // A warp whose keys all come after every row of the tile (or past the sequence) adds nothing.
        if (warp_key >= length || warp_key > q0 + KEYS_ROWS - 1) continue;
        const u32 sqs = sq + stage * KEYS_ROWS * row_bytes, sdos = sdo + stage * KEYS_ROWS * row_bytes;
        // The log partitions (base 2) and D of this thread's rows; rows past the sequence weigh nothing.
        float lse2[KEYS_ROWS / 8][2], d[KEYS_ROWS / 8][2];
#pragma unroll
        for (int n = 0; n < KEYS_ROWS / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const u32 r = q0 + n * 8 + col0 + j;
                const bool inside = r < length;
                lse2[n][j] = inside ? __ldg(lse + (u64)(start + r) * hq + h) * 1.4426950408889634f : POS_INF;
                d[n][j] = inside ? __ldg(dsum + (u64)(start + r) * hq + h) : 0.0f;
            }
        }
        float st[1][KEYS_ROWS / 8][4], dpt[1][KEYS_ROWS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS_ROWS / 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) st[0][n][e] = dpt[0][n][e] = 0.0f;
        warp_mma<1, KEYS_ROWS / 8, HEAD_D / 16, false, false>(st, sk, CHUNKS, warp * 16, 0, sqs, CHUNKS, 0, 0, lane);
        warp_mma<1, KEYS_ROWS / 8, HEAD_D / 16, false, false>(dpt, sv, CHUNKS, warp * 16, 0, sdos, CHUNKS, 0, 0, lane);
        // Keys after a row weigh nothing for it (keys past the sequence have no cotangent written).
        const bool masked = warp_key + 15 > q0;
#pragma unroll
        for (int n = 0; n < KEYS_ROWS / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float p = exp2_approx(st[0][n][e] * scale_log2 - lse2[n][e & 1]);
                if (masked && key_a + 8 * (e >> 1) > q0 + n * 8 + col0 + (e & 1)) p = 0.0f;
                st[0][n][e] = p;
                dpt[0][n][e] = p * (dpt[0][n][e] - d[n][e & 1]);
            }
        }
        // dV += Pᵀ dO and dK += dSᵀ Q, the rows of the tile contracted.
#pragma unroll
        for (int kk = 0; kk < KEYS_ROWS / 16; ++kk) {
            const u32 ap[4] = {pack_bf16(st[0][2 * kk][0], st[0][2 * kk][1]), pack_bf16(st[0][2 * kk][2], st[0][2 * kk][3]), pack_bf16(st[0][2 * kk + 1][0], st[0][2 * kk + 1][1]), pack_bf16(st[0][2 * kk + 1][2], st[0][2 * kk + 1][3])};
            const u32 as[4] = {pack_bf16(dpt[0][2 * kk][0], dpt[0][2 * kk][1]), pack_bf16(dpt[0][2 * kk][2], dpt[0][2 * kk][3]), pack_bf16(dpt[0][2 * kk + 1][0], dpt[0][2 * kk + 1][1]), pack_bf16(dpt[0][2 * kk + 1][2], dpt[0][2 * kk + 1][3])};
            const u32 row = kk * 16 + (lane & 7u) + (((lane >> 3) & 1u) << 3);
#pragma unroll
            for (int nd = 0; nd < HEAD_D / 8; nd += 2) {
                const u32 offset = swizzle(row, nd * 8 + ((lane >> 4) << 3), CHUNKS);
                u32 b[4];
                ldsm_t(sdos + offset, b);
                mma(dv[nd], ap, b[0], b[1]);
                mma(dv[nd + 1], ap, b[2], b[3]);
                ldsm_t(sqs + offset, b);
                mma(dk[nd], as, b[0], b[1]);
                mma(dk[nd + 1], as, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const u32 key = key_a + 8 * i;
        if (key >= length) continue;
        float* row = gy + (u64)(start + key) * stride;
#pragma unroll
        for (int n = 0; n < HEAD_D / 8; ++n) {
            const u32 c = n * 8 + col0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (c + j < HEAD_W) {
                    row[(hq + kv) * HEAD_W + c + j] = scale * dk[n][2 * i + j];
                    row[(hq + hk + kv) * HEAD_W + c + j] = dv[n][2 * i + j];
                }
            }
        }
    }
}

// ---- Reverse: the queries' cotangents ----
// A block holds ROWS_TILE query rows of one query head of one sequence (16 per warp, their query
// fragments in registers) and sweeps the key and value tiles up to its last row, the next pair
// loading while this one's products run. P, dP and dS stay in registers, where dS already is the
// operand of dQ += dS K; dQ stays in registers until the end.
// Blocks as the forward's. Dynamic shared memory: (2 ROWS_TILE + 4 ROWS_KEYS) HEAD_D bfloat16.
extern "C" __global__ void __launch_bounds__(ROWS_THREADS) attention_backward_queries(const Sequences sequences, u32 hq, u32 hk, float scale, const u16* __restrict__ y, const float* __restrict__ lse, const u16* __restrict__ ga16, const float* __restrict__ dsum, float* __restrict__ gy) {
    extern __shared__ __align__(128) unsigned char smem[];
    const u32 pair = blockIdx.z * gridDim.x + blockIdx.x;
    if (pair >= sequences.count * hq) return;
    const u32 head = pair % hq, sequence = pair / hq;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 tiles = (length + ROWS_TILE - 1) / ROWS_TILE;
    if (blockIdx.y >= tiles) return;
    const u32 q0 = (tiles - 1 - blockIdx.y) * ROWS_TILE;
    const u32 stride = (hq + 2 * hk) * HEAD_W, reads = hq * HEAD_W, kv = head / (hq / hk);
    const float scale_log2 = scale * 1.4426950408889634f;
    const u16* yk = y + (u64)start * stride + (hq + kv) * HEAD_W;
    const u16* yv = y + (u64)start * stride + (hq + hk + kv) * HEAD_W;
    const u32 row_bytes = HEAD_D * 2;
    const u32 sq = shared_address(smem), sdo = sq + ROWS_TILE * row_bytes, sk = sdo + ROWS_TILE * row_bytes, sv = sk + 2 * ROWS_KEYS * row_bytes;
    const u32 keys = min(q0 + ROWS_TILE, length), key_tiles = (keys + ROWS_KEYS - 1) / ROWS_KEYS;
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const u32 valid = min((u32)ROWS_TILE, length - q0);

    load_head_tile<ROWS_TILE, ROWS_THREADS>(sq, y + (u64)(start + q0) * stride + head * HEAD_W, stride, valid);
    load_head_tile<ROWS_TILE, ROWS_THREADS>(sdo, ga16 + (u64)(start + q0) * reads + head * HEAD_W, reads, valid);
    load_head_tile<ROWS_KEYS, ROWS_THREADS>(sk, yk, stride, min((u32)ROWS_KEYS, keys));
    load_head_tile<ROWS_KEYS, ROWS_THREADS>(sv, yv, stride, min((u32)ROWS_KEYS, keys));
    cp_commit();

    const u32 row0 = q0 + warp * 16 + (lane >> 2), col0 = 2 * (lane & 3u);
    float lse2[2], d[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const u32 r = row0 + 8 * i;
        const bool inside = r < length;
        lse2[i] = inside ? lse[(u64)(start + r) * hq + head] * 1.4426950408889634f : POS_INF;
        d[i] = inside ? dsum[(u64)(start + r) * hq + head] : 0.0f;
    }
    u32 qf[HEAD_D / 16][4];
    float dq[HEAD_D / 8][4];
#pragma unroll
    for (int n = 0; n < HEAD_D / 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.0f;

    for (u32 kt = 0; kt < key_tiles; ++kt) {
        const u32 k0 = kt * ROWS_KEYS, stage = kt & 1u;
        const u32 sks = sk + stage * ROWS_KEYS * row_bytes, svs = sv + stage * ROWS_KEYS * row_bytes;
        cp_wait_all();
        __syncthreads();
        if (kt == 0) {
#pragma unroll
            for (int ks = 0; ks < HEAD_D / 16; ++ks) ldsm(sq + swizzle(warp * 16 + (lane & 15u), ks * 16 + ((lane >> 4) << 3), CHUNKS), qf[ks]);
        }
        if (kt + 1 < key_tiles) {
            const u32 next = k0 + ROWS_KEYS;
            load_head_tile<ROWS_KEYS, ROWS_THREADS>(sk + (stage ^ 1u) * ROWS_KEYS * row_bytes, yk + (u64)next * stride, stride, min((u32)ROWS_KEYS, keys - next));
            load_head_tile<ROWS_KEYS, ROWS_THREADS>(sv + (stage ^ 1u) * ROWS_KEYS * row_bytes, yv + (u64)next * stride, stride, min((u32)ROWS_KEYS, keys - next));
            cp_commit();
        }
        float s[ROWS_KEYS / 8][4], dp[ROWS_KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < ROWS_KEYS / 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[n][e] = dp[n][e] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < HEAD_D / 16; ++ks) {
            u32 dof[4];
            ldsm(sdo + swizzle(warp * 16 + (lane & 15u), ks * 16 + ((lane >> 4) << 3), CHUNKS), dof);
#pragma unroll
            for (int nt = 0; nt < ROWS_KEYS / 8; nt += 2) {
                const u32 offset = swizzle(nt * 8 + (lane & 7u) + ((lane >> 4) << 3), ks * 16 + (((lane >> 3) & 1u) << 3), CHUNKS);
                u32 b[4];
                ldsm(sks + offset, b);
                mma(s[nt], qf[ks], b[0], b[1]);
                mma(s[nt + 1], qf[ks], b[2], b[3]);
                ldsm(svs + offset, b);
                mma(dp[nt], dof, b[0], b[1]);
                mma(dp[nt + 1], dof, b[2], b[3]);
            }
        }
        // dS = P ∘ (dP − D); keys after a row weigh nothing for it (keys past the sequence are zeros).
        const bool masked = k0 + ROWS_KEYS > q0 + warp * 16 + 1;
#pragma unroll
        for (int n = 0; n < ROWS_KEYS / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float p = exp2_approx(s[n][e] * scale_log2 - lse2[e >> 1]);
                if (masked && k0 + n * 8 + col0 + (e & 1) > row0 + 8 * (e >> 1)) p = 0.0f;
                s[n][e] = p * (dp[n][e] - d[e >> 1]);
            }
        }
        // dQ += dS K, the keys of the tile contracted.
#pragma unroll
        for (int kk = 0; kk < ROWS_KEYS / 16; ++kk) {
            const u32 a[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]), pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int nd = 0; nd < HEAD_D / 8; nd += 2) {
                u32 b[4];
                ldsm_t(sks + swizzle(kk * 16 + (lane & 7u) + (((lane >> 3) & 1u) << 3), nd * 8 + ((lane >> 4) << 3), CHUNKS), b);
                mma(dq[nd], a, b[0], b[1]);
                mma(dq[nd + 1], a, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const u32 row = row0 + 8 * i;
        if (row >= length) continue;
        float* dst = gy + (u64)(start + row) * stride + head * HEAD_W;
#pragma unroll
        for (int n = 0; n < HEAD_D / 8; ++n) {
            const u32 c = n * 8 + col0;
            if (c < HEAD_W) dst[c] = scale * dq[n][2 * i];
            if (c + 1 < HEAD_W) dst[c + 1] = scale * dq[n][2 * i + 1];
        }
    }
}

// ---- Hopper (sm_90a): warpgroup products ----
// On Hopper the products run as wgmma: a warpgroup (4 warps) multiplies 64 rows at a time, its
// operands read from shared memory through descriptors (or the left one from registers), so the
// shared tiles take the layout those read: HEAD_D / 64 blocks of 64 columns, each row of a block
// 128 bytes, its 16-byte chunks permuted by the row mod 8 (the 128-byte swizzle; blocks 1024-byte
// aligned). Accumulators and register operands keep mma.sync's per-warp fragments (warp w of a
// warpgroup holds its rows 16w..16w + 15).
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

__device__ __forceinline__ u32 hopper_offset(u32 r, u32 c, u32 rows) {
    return (c >> 6) * rows * 128u + r * 128u + ((((c & 63u) >> 3) ^ (r & 7u)) << 4) + (c & 7u) * 2u;
}

// Rows [0, R) of a head tile from `src` into the shared tile at `tile` in the warpgroup layout.
template <int R, int THREADS>
__device__ __forceinline__ void load_hopper_tile(u32 tile, const u16* src, u64 stride, u32 valid) {
#pragma unroll
    for (u32 i = threadIdx.x; i < R * CHUNKS; i += THREADS) {
        u32 r = i / CHUNKS, c = (i % CHUNKS) * 8u;
        u32 dst = tile + hopper_offset(r, c, R);
#if HEAD_W % 8 == 0
        bool inside = r < valid && c < HEAD_W;
        cp_async16(dst, inside ? (const void*)(src + (u64)r * stride + c) : (const void*)src, inside ? 16u : 0u);
#else
        u32 v[4];
#pragma unroll
        for (u32 k = 0; k < 4; ++k) {
            u32 c0 = c + 2u * k;
            float a = r < valid && c0 < HEAD_W ? from_bf16(src[(u64)r * stride + c0]) : 0.0f;
            float b = r < valid && c0 + 1u < HEAD_W ? from_bf16(src[(u64)r * stride + c0 + 1u]) : 0.0f;
            v[k] = pack_bf16(a, b);
        }
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(dst), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
#endif
    }
}

// A shared-memory matrix descriptor with the 128-byte swizzle: start address, leading and stride
// byte offsets.
__device__ __forceinline__ u64 descriptor(u32 address, u32 leading, u32 stride) {
    return (u64)((address & 0x3FFFFu) >> 4) | ((u64)((leading & 0x3FFFFu) >> 4) << 16) | ((u64)((stride & 0x3FFFFu) >> 4) << 32) | (1ull << 62);
}

// An operand stored with its 16 columns of step s contiguous (K-major), from row `row0` of a tile
// of `rows` rows: 8-row groups 1024 bytes apart.
__device__ __forceinline__ u64 k_major(u32 tile, u32 rows, u32 row0, u32 s) {
    return descriptor(tile + (s >> 2) * rows * 128u + row0 * 128u + (s & 3u) * 32u, 16u, 1024u);
}

// The right operand of step s read with its N columns contiguous (MN-major): rows 16 s.. of a tile
// of `rows` rows, every column; 64-column blocks rows · 128 bytes apart, 8-row groups 1024.
__device__ __forceinline__ u64 mn_major(u32 tile, u32 rows, u32 s) {
    return descriptor(tile + s * 16u * 128u, rows * 128u, 1024u);
}

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
// Writes by threads (cp.async, st.shared) made visible to the warpgroup products' reads.
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// The accumulators' values pinned at this point of the program (the products write them
// asynchronously, so no read may move above the wait, nor a write below the next issue).
template <int N>
__device__ __forceinline__ void pin(float (&d)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
// d (+)= A B on a warpgroup: A 64 × 16 and B 16 × 64 from shared memory (descriptors, both stored
// with K contiguous); `accumulate` zero overwrites d.
__device__ __forceinline__ void wgmma_64_ss(float (&d)[32], u64 a, u64 b, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a), "l"(b), "r"(accumulate));
}

// d (+)= A B on a warpgroup: A 64 × 16 and B 16 × 128 from shared memory (descriptors, both stored
// with K contiguous); `accumulate` zero overwrites d.
__device__ __forceinline__ void wgmma_128_ss(float (&d)[64], u64 a, u64 b, int accumulate) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a), "l"(b), "r"(accumulate));
}

// d += A B on a warpgroup: A 64 × 16 from registers (each warp's 16 rows as mma.sync's A fragment),
// B 16 × 64 from shared memory stored with N contiguous (transposed on the way in).
__device__ __forceinline__ void wgmma_64_rs(float (&d)[32], const u32 (&a)[4], u64 b) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(1));
}

// d += A B on a warpgroup: A 64 × 16 from registers (each warp's 16 rows as mma.sync's A fragment),
// B 16 × 128 from shared memory stored with N contiguous (transposed on the way in).
__device__ __forceinline__ void wgmma_128_rs(float (&d)[64], const u32 (&a)[4], u64 b) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(1));
}


template <int N> struct Wg;
template <> struct Wg<64> {
    static __device__ __forceinline__ void ss(float (&d)[32], u64 a, u64 b, int accumulate) { wgmma_64_ss(d, a, b, accumulate); }
    static __device__ __forceinline__ void rs(float (&d)[32], const u32 (&a)[4], u64 b) { wgmma_64_rs(d, a, b); }
};
template <> struct Wg<128> {
    static __device__ __forceinline__ void ss(float (&d)[64], u64 a, u64 b, int accumulate) { wgmma_128_ss(d, a, b, accumulate); }
    static __device__ __forceinline__ void rs(float (&d)[64], const u32 (&a)[4], u64 b) { wgmma_128_rs(d, a, b); }
};

#define HOPPER_THREADS (HOPPER_GROUPS * 128)
#define HOPPER_ROWS (HOPPER_GROUPS * 64)

// The forward on Hopper: a block holds HOPPER_ROWS query rows (64 per warpgroup) and sweeps key and
// value tiles of HOPPER_KEYS rows, double-buffered; each warpgroup forms its scores S = Q Kᵀ with
// Q and K read from shared memory, the online softmax in registers, and O += P V with P from
// registers and V read transposed from shared memory.
// Blocks as attention_forward's. Dynamic shared memory: (HOPPER_ROWS + 4 HOPPER_KEYS) HEAD_D bfloat16
// and 1024 bytes of alignment.
extern "C" __global__ void __launch_bounds__(HOPPER_THREADS) attention_forward_sm90(const Sequences sequences, u32 hq, u32 hk, float scale_log2, const u16* __restrict__ y, u16* __restrict__ out, float* __restrict__ lse) {
    extern __shared__ __align__(1024) unsigned char smem_hopper[];
    const u32 pair = blockIdx.z * gridDim.x + blockIdx.x;
    if (pair >= sequences.count * hq) return;
    const u32 head = pair % hq, sequence = pair / hq;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 tiles = (length + HOPPER_ROWS - 1) / HOPPER_ROWS;
    if (blockIdx.y >= tiles) return;
    const u32 q0 = (tiles - 1 - blockIdx.y) * HOPPER_ROWS;
    const u32 stride = (hq + 2 * hk) * HEAD_W, kv = head / (hq / hk);
    const u16* yk = y + (u64)start * stride + (hq + kv) * HEAD_W;
    const u16* yv = y + (u64)start * stride + (hq + hk + kv) * HEAD_W;
    const u32 sq = (shared_address(smem_hopper) + 1023u) & ~1023u, tile_k = HOPPER_KEYS * HEAD_D * 2;
    const u32 sk = sq + HOPPER_ROWS * HEAD_D * 2, sv = sk + 2 * tile_k;
    const u32 keys = min(q0 + HOPPER_ROWS, length), key_tiles = (keys + HOPPER_KEYS - 1) / HOPPER_KEYS;
    const u32 group = threadIdx.x >> 7, warp = (threadIdx.x >> 5) & 3u, lane = threadIdx.x & 31u;

    load_hopper_tile<HOPPER_ROWS, HOPPER_THREADS>(sq, y + (u64)(start + q0) * stride + head * HEAD_W, stride, min((u32)HOPPER_ROWS, length - q0));
    load_hopper_tile<HOPPER_KEYS, HOPPER_THREADS>(sk, yk, stride, min((u32)HOPPER_KEYS, keys));
    load_hopper_tile<HOPPER_KEYS, HOPPER_THREADS>(sv, yv, stride, min((u32)HOPPER_KEYS, keys));
    cp_commit();

    float o[HEAD_D / 2];
#pragma unroll
    for (int i = 0; i < HEAD_D / 2; ++i) o[i] = 0.0f;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};
    // This warpgroup's first row, this thread's rows and columns within a score tile.
    const u32 group_row = q0 + group * 64, row0 = group_row + warp * 16 + (lane >> 2), col0 = 2 * (lane & 3u);

    for (u32 kt = 0; kt < key_tiles; ++kt) {
        const u32 k0 = kt * HOPPER_KEYS, stage = kt & 1u;
        const u32 sks = sk + stage * tile_k, svs = sv + stage * tile_k;
        cp_wait_all();
        fence_async_shared();
        __syncthreads();
        if (kt + 1 < key_tiles) {
            const u32 next = k0 + HOPPER_KEYS;
            load_hopper_tile<HOPPER_KEYS, HOPPER_THREADS>(sk + (stage ^ 1u) * tile_k, yk + (u64)next * stride, stride, min((u32)HOPPER_KEYS, keys - next));
            load_hopper_tile<HOPPER_KEYS, HOPPER_THREADS>(sv + (stage ^ 1u) * tile_k, yv + (u64)next * stride, stride, min((u32)HOPPER_KEYS, keys - next));
            cp_commit();
        }
        // A warpgroup whose rows all come before the tile's keys has nothing to add.
        if (k0 > group_row + 63) continue;
        float s[HOPPER_KEYS / 2];
        wg_fence();
#pragma unroll
        for (int ks = 0; ks < HEAD_D / 16; ++ks) Wg<HOPPER_KEYS>::ss(s, k_major(sq, HOPPER_ROWS, group * 64, ks), k_major(sks, HOPPER_KEYS, 0, ks), ks);
        wg_commit();
        wg_wait();
        pin(s);
        // Scores in base-2 units; keys after a row's position, or past the sequence, weigh nothing.
        const bool masked = k0 + HOPPER_KEYS > group_row + warp * 16 + 1 || k0 + HOPPER_KEYS > length;
#pragma unroll
        for (int n = 0; n < HOPPER_KEYS / 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[4 * n + e] * scale_log2;
                if (masked) {
                    u32 key = k0 + n * 8 + col0 + (e & 1), row = row0 + 8 * (e >> 1);
                    if (key > row || key >= length) v = NEG_INF;
                }
                s[4 * n + e] = v;
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float mx = m[i];
#pragma unroll
            for (int n = 0; n < HOPPER_KEYS / 8; ++n) mx = fmaxf(mx, fmaxf(s[4 * n + 2 * i], s[4 * n + 2 * i + 1]));
            mx = quad_max(mx);
            const float base = mx == NEG_INF ? 0.0f : mx;
            const float alpha = exp2_approx(m[i] - base);
            m[i] = mx;
            l[i] *= alpha;
#pragma unroll
            for (int n = 0; n < HEAD_D / 8; ++n) {
                o[4 * n + 2 * i] *= alpha;
                o[4 * n + 2 * i + 1] *= alpha;
            }
#pragma unroll
            for (int n = 0; n < HOPPER_KEYS / 8; ++n) {
                s[4 * n + 2 * i] = exp2_approx(s[4 * n + 2 * i] - base);
                s[4 * n + 2 * i + 1] = exp2_approx(s[4 * n + 2 * i + 1] - base);
            }
        }
        // The weights as the values' product's left operand (bfloat16); the row sums add the same.
        u32 p[HOPPER_KEYS / 16][4];
#pragma unroll
        for (int kk = 0; kk < HOPPER_KEYS / 16; ++kk) {
            const int a = 8 * kk, b = 8 * kk + 4;
            p[kk][0] = pack_bf16(s[a], s[a + 1]);
            p[kk][1] = pack_bf16(s[a + 2], s[a + 3]);
            p[kk][2] = pack_bf16(s[b], s[b + 1]);
            p[kk][3] = pack_bf16(s[b + 2], s[b + 3]);
            l[0] += bf16_lo(p[kk][0]) + bf16_hi(p[kk][0]) + bf16_lo(p[kk][2]) + bf16_hi(p[kk][2]);
            l[1] += bf16_lo(p[kk][1]) + bf16_hi(p[kk][1]) + bf16_lo(p[kk][3]) + bf16_hi(p[kk][3]);
        }
        pin(o);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < HOPPER_KEYS / 16; ++kk) Wg<HEAD_D>::rs(o, p[kk], mn_major(svs, HOPPER_KEYS, kk));
        wg_commit();
        wg_wait();
        pin(o);
    }
    const u32 reads = hq * HEAD_W;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float total = quad_sum(l[i]);
        const u32 row = row0 + 8 * i;
        if (row >= length) continue;
        const float inverse = 1.0f / total;
        u16* dst = out + (u64)(start + row) * reads + head * HEAD_W;
#pragma unroll
        for (int n = 0; n < HEAD_D / 8; ++n) {
            const u32 c = n * 8 + col0;
#if HEAD_W % 8 == 0
            if (c < HEAD_W) *(u32*)(dst + c) = pack_bf16(o[4 * n + 2 * i] * inverse, o[4 * n + 2 * i + 1] * inverse);
#else
            if (c < HEAD_W) dst[c] = to_bf16(o[4 * n + 2 * i] * inverse);
            if (c + 1 < HEAD_W) dst[c + 1] = to_bf16(o[4 * n + 2 * i + 1] * inverse);
#endif
        }
        if ((lane & 3u) == 0) lse[(u64)(start + row) * hq + head] = (m[i] + __log2f(total)) * 0.6931471805599453f;
    }
}

#endif
