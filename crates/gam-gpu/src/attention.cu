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
//   dV = Pᵀ dO, dK = scale dSᵀ Q, dQ = scale dS K, each key-value head's summed over its query heads.
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;

#define HEAD_D (HEAD_W <= 64 ? 64 : 128)
// 16-byte chunks per row of a head tile, and of a score tile (64 columns).
#define CHUNKS (HEAD_D / 8)
#define SCORE_CHUNKS 8

// The sequences of one launch, passed by value (a kernel's parameters hold 4 KB). The tiles'
// sizes (FWD_WARPS, FWD_KEYS, BWD_WARPS, BWD_ROWS, BWD_KEYS) and MAX_SEQUENCES are defined by the
// caller (`tensor::cuda`), which launches with them.
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
// A block holds FWD_ROWS query rows of one query head of one sequence (16 per warp, their query
// fragments in registers) and sweeps the key and value tiles of FWD_KEYS rows up to its last row,
// loading the value tile while the scores' products run and the next key tile while the values'
// run.
#define FWD_THREADS (FWD_WARPS * 32)
#define FWD_ROWS (FWD_WARPS * 16)

// Blocks: x = sequence × hq + query head; y = the query tile, counted from the sequence's last (the
// longest sweep first). Dynamic shared memory: (FWD_ROWS + 2 FWD_KEYS) HEAD_D bfloat16.
extern "C" __global__ void __launch_bounds__(FWD_THREADS) attention_forward(const Sequences sequences, u32 hq, u32 hk, float scale_log2, const u16* __restrict__ y, u16* __restrict__ out, float* __restrict__ lse) {
    extern __shared__ __align__(128) unsigned char smem[];
    const u32 head = blockIdx.x % hq, sequence = blockIdx.x / hq;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 tiles = (length + FWD_ROWS - 1) / FWD_ROWS;
    if (blockIdx.y >= tiles) return;
    const u32 q0 = (tiles - 1 - blockIdx.y) * FWD_ROWS;
    const u32 stride = (hq + 2 * hk) * HEAD_W, kv = head / (hq / hk);
    const u16* yq = y + (u64)(start + q0) * stride + head * HEAD_W;
    const u16* yk = y + (u64)start * stride + (hq + kv) * HEAD_W;
    const u16* yv = y + (u64)start * stride + (hq + hk + kv) * HEAD_W;
    const u32 sq = shared_address(smem), sk = sq + FWD_ROWS * HEAD_D * 2, sv = sk + FWD_KEYS * HEAD_D * 2;
    const u32 keys = min(q0 + FWD_ROWS, length), key_tiles = (keys + FWD_KEYS - 1) / FWD_KEYS;
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;

    load_head_tile<FWD_ROWS, FWD_THREADS>(sq, yq, stride, min((u32)FWD_ROWS, length - q0));
    load_head_tile<FWD_KEYS, FWD_THREADS>(sk, yk, stride, min((u32)FWD_KEYS, keys));
    cp_commit();

    u32 qf[HEAD_D / 16][4];
    float o[HEAD_D / 8][4];
#pragma unroll
    for (int n = 0; n < HEAD_D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};
    // This thread's rows (relative to the sequence) and columns within a score tile.
    const u32 row0 = q0 + warp * 16 + (lane >> 2), col0 = 2 * (lane & 3u);

    for (u32 kt = 0; kt < key_tiles; ++kt) {
        const u32 k0 = kt * FWD_KEYS;
        cp_wait_all();
        __syncthreads();
        if (kt == 0) {
#pragma unroll
            for (int ks = 0; ks < HEAD_D / 16; ++ks) ldsm(sq + swizzle(warp * 16 + (lane & 15u), ks * 16 + ((lane >> 4) << 3), CHUNKS), qf[ks]);
        }
        load_head_tile<FWD_KEYS, FWD_THREADS>(sv, yv + (u64)k0 * stride, stride, min((u32)FWD_KEYS, keys - k0));
        cp_commit();
        float s[FWD_KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < FWD_KEYS / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < HEAD_D / 16; ++ks) {
#pragma unroll
            for (int nt = 0; nt < FWD_KEYS / 8; nt += 2) {
                u32 bf[4];
                ldsm(sk + swizzle(nt * 8 + (lane & 7u) + ((lane >> 4) << 3), ks * 16 + (((lane >> 3) & 1u) << 3), CHUNKS), bf);
                mma(s[nt], qf[ks], bf[0], bf[1]);
                mma(s[nt + 1], qf[ks], bf[2], bf[3]);
            }
        }
        cp_wait_all();
        __syncthreads();
        if (kt + 1 < key_tiles) {
            load_head_tile<FWD_KEYS, FWD_THREADS>(sk, yk + (u64)(k0 + FWD_KEYS) * stride, stride, min((u32)FWD_KEYS, keys - k0 - FWD_KEYS));
            cp_commit();
        }
        // Scores in base-2 units; keys after a row's position, or past the sequence, weigh nothing.
        const bool masked = k0 + FWD_KEYS > q0 + warp * 16 + 1 || k0 + FWD_KEYS > length;
#pragma unroll
        for (int n = 0; n < FWD_KEYS / 8; ++n) {
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
            for (int n = 0; n < FWD_KEYS / 8; ++n) mx = fmaxf(mx, fmaxf(s[n][2 * i], s[n][2 * i + 1]));
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
            for (int n = 0; n < FWD_KEYS / 8; ++n) {
                s[n][2 * i] = exp2_approx(s[n][2 * i] - base);
                s[n][2 * i + 1] = exp2_approx(s[n][2 * i + 1] - base);
            }
        }
        // The weights as the values' product's operand (bfloat16); the row sums add the same.
#pragma unroll
        for (int kk = 0; kk < FWD_KEYS / 16; ++kk) {
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
extern "C" __global__ void attention_backward_rows(u32 rows, u32 hq, const u16* __restrict__ out, const float* __restrict__ ga, u16* __restrict__ ga16, float* __restrict__ dsum) {
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

// A block holds BWD_KEYS key rows of one key-value head of one sequence (keys and values in shared
// memory, their cotangents in registers) and sweeps, for each query head of the group, the query
// tiles from its own diagonal on: the next query tile loads while this one's products run, the
// next tile's dO while the queries' cotangent is formed. The queries' cotangents are added into
// gy atomically; the keys' and values' are written once at the end. gy holds zeros before.
#if BWD_WARPS != 8 || BWD_ROWS != 64 || BWD_KEYS != 64
#error "the reverse's warp tilings take 8 warps over 64 query rows and 64 keys"
#endif
#define BWD_THREADS (BWD_WARPS * 32)

// Blocks: x = sequence × hk + key-value head; y = the key tile (the first, the longest sweep, first).
// Dynamic shared memory: (2 BWD_KEYS + 3 BWD_ROWS) HEAD_D + 2 BWD_ROWS BWD_KEYS bfloat16.
extern "C" __global__ void __launch_bounds__(BWD_THREADS) attention_backward(const Sequences sequences, u32 hq, u32 hk, float scale, const u16* __restrict__ y, const float* __restrict__ lse, const u16* __restrict__ ga16, const float* __restrict__ dsum, float* __restrict__ gy) {
    extern __shared__ __align__(128) unsigned char smem[];
    const u32 kv = blockIdx.x % hk, sequence = blockIdx.x / hk;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 k0 = blockIdx.y * BWD_KEYS;
    if (k0 >= length) return;
    const u32 stride = (hq + 2 * hk) * HEAD_W, reads = hq * HEAD_W, group = hq / hk;
    const float scale_log2 = scale * 1.4426950408889634f;
    const u32 tile = HEAD_D * 2;
    const u32 sk = shared_address(smem), sv = sk + BWD_KEYS * tile, sq = sv + BWD_KEYS * tile;
    const u32 sdo = sq + 2 * BWD_ROWS * tile, sp = sdo + BWD_ROWS * tile, sds = sp + BWD_ROWS * BWD_KEYS * 2;
    const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const u32 kvalid = min((u32)BWD_KEYS, length - k0);
    const u32 first = k0 / BWD_ROWS, last = (length - 1) / BWD_ROWS, per_head = last - first + 1;

    load_head_tile<BWD_KEYS, BWD_THREADS>(sk, y + (u64)(start + k0) * stride + (hq + kv) * HEAD_W, stride, kvalid);
    load_head_tile<BWD_KEYS, BWD_THREADS>(sv, y + (u64)(start + k0) * stride + (hq + hk + kv) * HEAD_W, stride, kvalid);
    {
        const u32 h = kv * group, q0 = first * BWD_ROWS;
        load_head_tile<BWD_ROWS, BWD_THREADS>(sq, y + (u64)(start + q0) * stride + h * HEAD_W, stride, min((u32)BWD_ROWS, length - q0));
        load_head_tile<BWD_ROWS, BWD_THREADS>(sdo, ga16 + (u64)(start + q0) * reads + h * HEAD_W, reads, min((u32)BWD_ROWS, length - q0));
    }
    cp_commit();

    // dK and dV: warps 2 (keys, 32 each) × 4 (head columns, HEAD_D / 4 each).
    const u32 kw = warp & 1u, dw = warp >> 1;
    float dk[2][HEAD_D / 32][4], dv[2][HEAD_D / 32][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < HEAD_D / 32; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) dk[a][b][e] = dv[a][b][e] = 0.0f;
    // Scores: warps 4 (query rows, 16 each) × 2 (keys, 32 each).
    const u32 sw_r = warp & 3u, sw_c = warp >> 2;
    const u32 col0 = 2 * (lane & 3u);

    u32 buffer = 0;
    for (u32 step = 0; step < group * per_head; ++step) {
        const u32 h = kv * group + step / per_head, q0 = (first + step % per_head) * BWD_ROWS;
        const u32 sqb = sq + buffer * BWD_ROWS * tile;
        cp_wait_all();
        __syncthreads();
        const bool more = step + 1 < group * per_head;
        const u32 next_h = kv * group + (step + 1) / per_head, next_q0 = (first + (step + 1) % per_head) * BWD_ROWS;
        if (more) {
            load_head_tile<BWD_ROWS, BWD_THREADS>(sq + (buffer ^ 1u) * BWD_ROWS * tile, y + (u64)(start + next_q0) * stride + next_h * HEAD_W, stride, min((u32)BWD_ROWS, length - next_q0));
            cp_commit();
        }
        // This thread's two query rows of the score tile, their log partitions and D.
        const u32 r_a = q0 + sw_r * 16 + (lane >> 2);
        float lse2[2], d[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const u32 r = r_a + 8 * i;
            const bool inside = r < length;
            lse2[i] = inside ? lse[(u64)(start + r) * hq + h] * 1.4426950408889634f : POS_INF;
            d[i] = inside ? dsum[(u64)(start + r) * hq + h] : 0.0f;
        }
        float s[1][4][4], dp[1][4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[0][n][e] = dp[0][n][e] = 0.0f;
        warp_mma<1, 4, HEAD_D / 16, false, false>(s, sqb, CHUNKS, sw_r * 16, 0, sk, CHUNKS, 0, sw_c * 32, lane);
        warp_mma<1, 4, HEAD_D / 16, false, false>(dp, sdo, CHUNKS, sw_r * 16, 0, sv, CHUNKS, 0, sw_c * 32, lane);
        const bool masked = k0 + BWD_KEYS > q0 + sw_r * 16 + 1 || q0 + BWD_ROWS > length || k0 + BWD_KEYS > length;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float p[2], ds[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int e = 2 * i + j;
                    p[j] = exp2_approx(s[0][n][e] * scale_log2 - lse2[i]);
                    if (masked) {
                        const u32 key = k0 + sw_c * 32 + n * 8 + col0 + j, row = r_a + 8 * i;
                        if (key > row || key >= length || row >= length) p[j] = 0.0f;
                    }
                    ds[j] = p[j] * (dp[0][n][e] - d[i]);
                }
                const u32 offset = swizzle(sw_r * 16 + (lane >> 2) + 8 * i, sw_c * 32 + n * 8 + col0, SCORE_CHUNKS);
                asm volatile("st.shared.u32 [%0], %1;" ::"r"(sp + offset), "r"(pack_bf16(p[0], p[1])) : "memory");
                asm volatile("st.shared.u32 [%0], %1;" ::"r"(sds + offset), "r"(pack_bf16(ds[0], ds[1])) : "memory");
            }
        }
        __syncthreads();
        // dV += Pᵀ dO and dK += dSᵀ Q (P and dS stored as [query][key]: Aᵀ).
        warp_mma<2, HEAD_D / 32, BWD_ROWS / 16, true, true>(dv, sp, SCORE_CHUNKS, kw * 32, 0, sdo, CHUNKS, 0, dw * (HEAD_D / 4), lane);
        warp_mma<2, HEAD_D / 32, BWD_ROWS / 16, true, true>(dk, sds, SCORE_CHUNKS, kw * 32, 0, sqb, CHUNKS, 0, dw * (HEAD_D / 4), lane);
        __syncthreads();
        if (more) {
            load_head_tile<BWD_ROWS, BWD_THREADS>(sdo, ga16 + (u64)(start + next_q0) * reads + next_h * HEAD_W, reads, min((u32)BWD_ROWS, length - next_q0));
            cp_commit();
        }
        // dQ = scale dS K: warps 4 (query rows, 16 each) × 2 (head columns, HEAD_D / 2 each).
        {
            float dq[1][HEAD_D / 16][4];
#pragma unroll
            for (int n = 0; n < HEAD_D / 16; ++n) dq[0][n][0] = dq[0][n][1] = dq[0][n][2] = dq[0][n][3] = 0.0f;
            warp_mma<1, HEAD_D / 16, BWD_KEYS / 16, false, true>(dq, sds, SCORE_CHUNKS, sw_r * 16, 0, sk, CHUNKS, 0, sw_c * (HEAD_D / 2), lane);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const u32 row = r_a + 8 * i;
                if (row >= length) continue;
                float* dst = gy + (u64)(start + row) * stride + h * HEAD_W;
#pragma unroll
                for (int n = 0; n < HEAD_D / 16; ++n) {
                    const u32 c = sw_c * (HEAD_D / 2) + n * 8 + col0;
                    if (c < HEAD_W) atomicAdd(dst + c, scale * dq[0][n][2 * i]);
                    if (c + 1 < HEAD_W) atomicAdd(dst + c + 1, scale * dq[0][n][2 * i + 1]);
                }
            }
        }
        buffer ^= 1u;
    }
    // The keys' and values' cotangents.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const u32 key = kw * 32 + mt * 16 + (lane >> 2) + 8 * i;
            if (key >= kvalid) continue;
            float* row = gy + (u64)(start + k0 + key) * stride;
#pragma unroll
            for (int n = 0; n < HEAD_D / 32; ++n) {
                const u32 c = dw * (HEAD_D / 4) + n * 8 + col0;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if (c + j < HEAD_W) {
                        row[(hq + kv) * HEAD_W + c + j] = scale * dk[mt][n][2 * i + j];
                        row[(hq + hk + kv) * HEAD_W + c + j] = dv[mt][n][2 * i + j];
                    }
                }
            }
        }
    }
}
