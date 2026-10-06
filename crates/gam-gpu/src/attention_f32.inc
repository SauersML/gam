// Causal attention in f32 (`Device::causal_attention`), forward and reverse, as kernel bodies written
// once for both GPU backends: the CUDA module (`tensor::cuda`, NVRTC) and the Apple GPU's library
// (`tensor::apple`, Metal Shading Language) prepend their definitions of the macros below and wrap
// each body in a kernel. Every product is formed in f32 with fused multiply-adds on the
// multiprocessors' f32 units (no tensor cores): f32 is the precision the decoder's parity needs (each
// row's softmax normalizes away no rounding of the scores), and on the RTX 4090 and the L40 the f32
// units do f32 work faster than bfloat16 tensor-core products split into the three terms f32 would
// need (82.6 against 165.2 / 6 TFLOP/s on the 4090).
//
// The heads `y` are f32, row-major, one row per position: the query heads, then the key heads, then
// the value heads, each `w` columns (w ≤ D, D = 64 or 128 a template parameter; columns from w on
// are zeros in the shared tiles). Query head h reads key-value head h / (hq / hk). The rows hold
// sequences given as (start row, length) pairs; position i of a sequence attends to positions 0..=i
// of the same sequence. A block of threads holds one tile of rows of one sequence and head in shared
// memory and sweeps the other side's tiles past it, so no T × T matrix reaches device memory.
//
// Forward: out = softmax(scale q kᵀ) v per query head (rows × hq·w), and each row's log partition
// lse = log Σⱼ exp(scale qᵢ·kⱼ) per query head (rows × hq), the row maxima and sums carried across
// key tiles (the online softmax). Reverse, with D = Σ dO·O per row and head, P = exp(scale q kᵀ − lse)
// and dS = P ∘ (dO vᵀ − D): dV = Pᵀ dO, dK = scale dSᵀ Q, dQ = scale dS K, a key-value head's summed
// over its query heads; one pass over blocks of key rows (dK, dV) and one over blocks of query rows
// (dQ), each owning what it writes.
//
// Macros: DEVICE (a function the kernels call), UNROLL (a loop of a fixed count, unrolled), GLOBAL, SHARED, CONSTANT (address spaces), BARRIER(), SHUFFLE_XOR(v, lane mask), EXP2,
// LOG2, FMA, FMAX, NEG_INF, POS_INF, f4 (four floats) with F4(a, b, c, d), LOAD4(p), STORE4(p, v),
// COPY4(shared dst, global src) (may complete later), COPY_COMMIT() (closes a group of copies),
// COPY_WAIT() (every copy issued has landed), COPY_WAIT_ONE() (every group but the newest has),
// u32, u64, and MAX_SEQUENCES.

struct Sequences {
    u32 count;
    u32 start[MAX_SEQUENCES];
    u32 length[MAX_SEQUENCES];
};

// Row stride of a shared head tile (floats): four columns of padding put the rows that a quarter-warp
// reads at one column in different banks.
#define TILE_STRIDE(D) ((D) + 4)

// Rows [0, rows) of a head tile from `src` (its row 0, column 0; `stride` floats between rows) into
// the shared tile `tile`; rows from `valid` on and columns from `w` on are zeros. Threads `t` of
// `threads` share the copies.
template <int D>
DEVICE void load_tile(SHARED float* tile, GLOBAL const float* src, u64 stride, u32 rows, u32 valid, u32 w, u32 t, u32 threads) {
    for (u32 i = t; i < rows * (D / 4); i += threads) {
        u32 r = i / (D / 4), c = (i % (D / 4)) * 4u;
        SHARED float* dst = tile + r * TILE_STRIDE(D) + c;
        if (r < valid && c + 4u <= w && (w & 3u) == 0u) {
            COPY4(dst, src + (u64)r * stride + c);
        } else {
            float v[4];
            for (u32 k = 0; k < 4u; ++k) v[k] = r < valid && c + k < w ? src[(u64)r * stride + c + k] : 0.0f;
            STORE4(dst, F4(v[0], v[1], v[2], v[3]));
        }
    }
}

// Rows [0, rows) and columns [0, C) of a matrix at `src` (`stride` floats between rows) into the
// shared tile `dst` (`dst_stride` floats between rows); rows from `valid` on and columns from `w` on
// are zeros. Threads `t` of `threads` share the copies.
template <int C>
DEVICE void load_block(SHARED float* dst, u32 dst_stride, GLOBAL const float* src, u64 stride, u32 rows, u32 valid, u32 w, u32 t, u32 threads) {
    for (u32 i = t; i < rows * (C / 4); i += threads) {
        u32 r = i / (C / 4), c = (i % (C / 4)) * 4u;
        SHARED float* d = dst + r * dst_stride + c;
        if (r < valid && c + 4u <= w && (w & 3u) == 0u) {
            COPY4(d, src + (u64)r * stride + c);
        } else {
            float v[4];
            for (u32 k = 0; k < 4u; ++k) v[k] = r < valid && c + k < w ? src[(u64)r * stride + c + k] : 0.0f;
            STORE4(d, F4(v[0], v[1], v[2], v[3]));
        }
    }
}

// The maximum and the sum over the `group` consecutive lanes holding a row (a power of two ≤ 32).
DEVICE float lane_max(float v, u32 group) {
    for (u32 o = 1u; o < group; o <<= 1) v = FMAX(v, SHUFFLE_XOR(v, o));
    return v;
}
DEVICE float lane_sum(float v, u32 group) {
    for (u32 o = 1u; o < group; o <<= 1) v += SHUFFLE_XOR(v, o);
    return v;
}

// ---- Forward ----
// A block of BR query rows (of one query head of one sequence) sweeps the key tiles of BC rows up to
// its last row. Its NT compute threads form RG = BR / TR row groups of CG = NT / RG lanes: thread
// (g, c) holds rows g + RG i (i < TR), the scores of keys c + CG j (j < BC / CG) and the output
// columns 4c + 4 CG m.. (m < D / (4 CG)) of its rows, so a row's statistics live in its CG lanes.
// With TR and BC / CG near 8 every shared load of four floats feeds about sixteen fused multiply-adds,
// which is what keeps the f32 units busy. The queries stay in shared memory; each key tile streams
// through two buffers in chunks of DC columns (the scores accumulate over them), then the value tile
// in chunks of DC rows, each chunk loading while the one before is multiplied. Shared memory:
// BR (D + 4) + 2 max(BC (DC + 4), DC (D + 4)) + BR (BC + 4) floats.
#define FORWARD_CHUNK(D, BC, DC) ((BC) * ((DC) + 4) > (DC) * ((D) + 4) ? (BC) * ((DC) + 4) : (DC) * ((D) + 4))
#define FORWARD_FLOATS(D, BR, BC, DC) ((BR) * ((D) + 4) + 2 * FORWARD_CHUNK(D, BC, DC) + (BR) * ((BC) + 4))

template <int D, int BR, int BC, int NT, int TR, int DC>
DEVICE void forward_body(CONSTANT Sequences& sequences, u32 hq, u32 hk, u32 w, float scale_log2, GLOBAL const float* y, GLOBAL float* out, GLOBAL float* lse,
                         u32 pair, u32 tile, u32 t, u32 threads, SHARED float* smem) {
    constexpr u32 RG = BR / TR, CG = NT / RG, KP = BC / CG, DP = D / CG, NK = D / DC, NV = BC / DC;
    constexpr u32 CHUNK = FORWARD_CHUNK(D, BC, DC);
    if (pair >= sequences.count * hq) return;
    const u32 head = pair % hq, sequence = pair / hq;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 tiles = (length + BR - 1u) / BR;
    if (tile >= tiles) return;
    const u32 q0 = (tiles - 1u - tile) * BR;
    const u32 stride = (hq + 2u * hk) * w, kv = head / (hq / hk);
    GLOBAL const float* yk = y + (u64)start * stride + (hq + kv) * w;
    GLOBAL const float* yv = y + (u64)start * stride + (hq + hk + kv) * w;
    SHARED float* sq = smem;
    SHARED float* sbuf = sq + BR * TILE_STRIDE(D);
    SHARED float* sp = sbuf + 2u * CHUNK;
    const u32 keys = min(q0 + BR, length), key_tiles = (keys + BC - 1u) / BC;
    const bool computes = t < NT;
    const u32 g = t / CG, c = t % CG;

    // Chunk u of key tile kt: u < NK the keys' columns [u DC, (u + 1) DC), else the values' rows
    // [(u − NK) DC, ...); every chunk goes to buffer (its number in the block's sequence) mod 2.
    load_tile<D>(sq, y + (u64)(start + q0) * stride + head * w, stride, BR, min((u32)BR, length - q0), w, t, threads);
    load_block<DC>(sbuf, DC + 4u, yk, stride, BC, min((u32)BC, keys), w, t, threads);
    COPY_COMMIT();

    float m[TR], l[TR], o[TR][DP], s[TR][KP];
    UNROLL for (u32 i = 0; i < TR; ++i) {
        m[i] = NEG_INF;
        l[i] = 0.0f;
        UNROLL for (u32 d = 0; d < DP; ++d) o[i][d] = 0.0f;
    }
    u32 chunk = 0;
    for (u32 kt = 0; kt < key_tiles; ++kt) {
        const u32 k0 = kt * BC, valid_keys = min((u32)BC, keys - k0);
        UNROLL for (u32 i = 0; i < TR; ++i)
            UNROLL for (u32 j = 0; j < KP; ++j) s[i][j] = 0.0f;
        for (u32 u = 0; u < NK + NV; ++u, ++chunk) {
            COPY_WAIT();
            BARRIER();
            // The next chunk: this tile's, or the next tile's first.
            SHARED float* next = sbuf + ((chunk + 1u) & 1u) * CHUNK;
            if (u + 1u < NK) {
                load_block<DC>(next, DC + 4u, yk + (u64)k0 * stride + (u + 1u) * DC, stride, BC, valid_keys, w > (u + 1u) * DC ? w - (u + 1u) * DC : 0u, t, threads);
            } else if (u + 1u < NK + NV) {
                const u32 r0 = (u + 1u - NK) * DC;
                load_block<D>(next, D + 4u, yv + (u64)(k0 + r0) * stride, stride, DC, valid_keys > r0 ? min((u32)DC, valid_keys - r0) : 0u, w, t, threads);
            } else if (kt + 1u < key_tiles) {
                load_block<DC>(next, DC + 4u, yk + (u64)(k0 + BC) * stride, stride, BC, min((u32)BC, keys - k0 - BC), w, t, threads);
            }
            COPY_COMMIT();
            SHARED float* cur = sbuf + (chunk & 1u) * CHUNK;
            if (u < NK) {
                if (computes) {
                    for (u32 d = 0; d < DC; d += 4u) {
                        f4 qv[TR], kvv[KP];
                        UNROLL for (u32 i = 0; i < TR; ++i) qv[i] = LOAD4(sq + (g + RG * i) * TILE_STRIDE(D) + u * DC + d);
                        UNROLL for (u32 j = 0; j < KP; ++j) kvv[j] = LOAD4(cur + (c + CG * j) * (DC + 4u) + d);
                        UNROLL for (u32 i = 0; i < TR; ++i)
                            UNROLL for (u32 j = 0; j < KP; ++j) s[i][j] = FMA(qv[i].w, kvv[j].w, FMA(qv[i].z, kvv[j].z, FMA(qv[i].y, kvv[j].y, FMA(qv[i].x, kvv[j].x, s[i][j]))));
                    }
                }
                if (u + 1u == NK && computes) {
                    // The tile's weights: scores in base-2 units; keys after a row's position, or past
                    // the sequence, weigh nothing.
                    UNROLL for (u32 i = 0; i < TR; ++i) {
                        const u32 row = q0 + g + RG * i;
                        float mx = m[i];
                        UNROLL for (u32 j = 0; j < KP; ++j) {
                            const u32 key = k0 + c + CG * j;
                            s[i][j] = key > row || key >= length ? NEG_INF : s[i][j] * scale_log2;
                            mx = FMAX(mx, s[i][j]);
                        }
                        mx = lane_max(mx, CG);
                        const float base = mx == NEG_INF ? 0.0f : mx;
                        const float alpha = EXP2(m[i] - base);
                        m[i] = mx;
                        l[i] *= alpha;
                        UNROLL for (u32 d = 0; d < DP; ++d) o[i][d] *= alpha;
                        UNROLL for (u32 j = 0; j < KP; ++j) {
                            const float p = EXP2(s[i][j] - base);
                            l[i] += p;
                            sp[(g + RG * i) * (BC + 4u) + c + CG * j] = p;
                        }
                    }
                }
            } else if (computes && (u - NK) * DC < valid_keys) {
                // The values' rows of this chunk against the weights' columns.
                const u32 r0 = (u - NK) * DC;
                for (u32 k = 0; k < DC; k += 4u) {
                    f4 pv[TR];
                    UNROLL for (u32 i = 0; i < TR; ++i) pv[i] = LOAD4(sp + (g + RG * i) * (BC + 4u) + r0 + k);
                    UNROLL for (u32 kk = 0; kk < 4u; ++kk) {
                        UNROLL for (u32 n = 0; n < DP / 4u; ++n) {
                            const f4 vv = LOAD4(cur + (k + kk) * TILE_STRIDE(D) + 4u * c + 4u * CG * n);
                            UNROLL for (u32 i = 0; i < TR; ++i) {
                                const float p = kk == 0u ? pv[i].x : kk == 1u ? pv[i].y : kk == 2u ? pv[i].z : pv[i].w;
                                o[i][4u * n] = FMA(p, vv.x, o[i][4u * n]);
                                o[i][4u * n + 1u] = FMA(p, vv.y, o[i][4u * n + 1u]);
                                o[i][4u * n + 2u] = FMA(p, vv.z, o[i][4u * n + 2u]);
                                o[i][4u * n + 3u] = FMA(p, vv.w, o[i][4u * n + 3u]);
                            }
                        }
                    }
                }
            }
        }
    }
    if (!computes) return;
    const u32 reads = hq * w;
    UNROLL for (u32 i = 0; i < TR; ++i) {
        const float total = lane_sum(l[i], CG);
        const u32 row = q0 + g + RG * i;
        if (row >= length) continue;
        const float inverse = 1.0f / total;
        GLOBAL float* dst = out + (u64)(start + row) * reads + head * w;
        UNROLL for (u32 n = 0; n < DP / 4u; ++n)
            UNROLL for (u32 e = 0; e < 4u; ++e) {
                const u32 col = 4u * c + 4u * CG * n + e;
                if (col < w) dst[col] = o[i][4u * n + e] * inverse;
            }
        if (c == 0u) lse[(u64)(start + row) * hq + head] = (m[i] + LOG2(total)) * 0.6931471805599453f;
    }
}

// ---- Reverse ----
// D = Σ dO·O per row and query head (rows × hq), one simdgroup or warp of 32 threads per (row, head):
// thread t of `threads` works on items t / 32, t / 32 + threads / 32, ...
DEVICE void sums_body(u32 rows, u32 hq, u32 w, GLOBAL const float* out, GLOBAL const float* ga, GLOBAL float* dsum, u32 t, u32 threads) {
    const u32 lane = t & 31u;
    for (u64 item = t >> 5; item < (u64)rows * hq; item += threads >> 5) {
        const u64 base = item * w;
        float s = 0.0f;
        for (u32 c = lane; c < w; c += 32u) s = FMA(ga[base + c], out[base + c], s);
        s = lane_sum(s, 32u);
        if (lane == 0u) dsum[item] = s;
    }
}

// The keys' and values' cotangents: a block of BK key rows (of one key-value head of one sequence)
// sweeps, for each query head of the group, the query tiles of BQ rows from its own diagonal on. Its
// NT compute threads form KG = BK / TK key groups of CG = NT / KG lanes: thread (g, c) holds keys
// g + KG i (i < TK), their transposed scores Sᵀ = K Qᵀ and dPᵀ = V dOᵀ at rows c + CG j (j < BQ / CG),
// and dK, dV at columns 4c + 4 CG m.. (m < D / (4 CG)). Pᵀ and dSᵀ go through shared memory to the
// threads that contract them with dO and Q. The next tile's queries load into a second buffer while
// this tile's products run, its cotangents while dK forms. Shared memory: (2 BK + 3 BQ) (D + 4) +
// 2 BK (BQ + 4) + 2 BQ floats.
template <int D, int BK, int BQ, int NT, int TK>
DEVICE void keys_body(CONSTANT Sequences& sequences, u32 hq, u32 hk, u32 w, float scale, GLOBAL const float* y, GLOBAL const float* lse, GLOBAL const float* ga,
                      GLOBAL const float* dsum, GLOBAL float* gy, u32 pair, u32 tile, u32 t, u32 threads, SHARED float* smem) {
    constexpr u32 KG = BK / TK, CG = NT / KG, QP = BQ / CG, DP = D / CG;
    if (pair >= sequences.count * hk) return;
    const u32 kv = pair % hk, sequence = pair / hk;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 k0 = tile * BK;
    if (k0 >= length) return;
    const u32 stride = (hq + 2u * hk) * w, reads = hq * w, group = hq / hk;
    const float scale_log2 = scale * 1.4426950408889634f;
    SHARED float* sk = smem;
    SHARED float* sv = sk + BK * TILE_STRIDE(D);
    SHARED float* sq = sv + BK * TILE_STRIDE(D);
    SHARED float* sdo = sq + 2u * BQ * TILE_STRIDE(D);
    SHARED float* sp = sdo + BQ * TILE_STRIDE(D);
    SHARED float* sds = sp + BK * (BQ + 4u);
    SHARED float* srow = sds + BK * (BQ + 4u);
    const u32 first = k0 / BQ, per_head = (length + BQ - 1u) / BQ - first, steps = group * per_head;
    const bool computes = t < NT;
    const u32 g = t / CG, c = t % CG;

    // Step s reads query head kv·group + s / per_head, rows from (first + s mod per_head) BQ.
    load_tile<D>(sk, y + (u64)(start + k0) * stride + (hq + kv) * w, stride, BK, min((u32)BK, length - k0), w, t, threads);
    load_tile<D>(sv, y + (u64)(start + k0) * stride + (hq + hk + kv) * w, stride, BK, min((u32)BK, length - k0), w, t, threads);
    {
        const u32 q0 = first * BQ;
        load_tile<D>(sq, y + (u64)(start + q0) * stride + kv * group * w, stride, BQ, min((u32)BQ, length - q0), w, t, threads);
        COPY_COMMIT();
        load_tile<D>(sdo, ga + (u64)(start + q0) * reads + kv * group * w, reads, BQ, min((u32)BQ, length - q0), w, t, threads);
        COPY_COMMIT();
    }

    float dk[TK][DP], dv[TK][DP];
    UNROLL for (u32 i = 0; i < TK; ++i)
        UNROLL for (u32 d = 0; d < DP; ++d) dk[i][d] = dv[i][d] = 0.0f;
    for (u32 step = 0; step < steps; ++step) {
        const u32 h = kv * group + step / per_head, q0 = (first + step % per_head) * BQ;
        const u32 valid = min((u32)BQ, length - q0);
        const bool more = step + 1u < steps;
        SHARED float* sqs = sq + (step & 1u) * BQ * TILE_STRIDE(D);
        // This tile's queries have landed (its cotangents, the newest copies, may not have); the
        // previous step's contractions are done with the buffers about to be replaced.
        COPY_WAIT_ONE();
        BARRIER();
        if (more) {
            const u32 nh = kv * group + (step + 1u) / per_head, nq0 = (first + (step + 1u) % per_head) * BQ;
            load_tile<D>(sq + ((step + 1u) & 1u) * BQ * TILE_STRIDE(D), y + (u64)(start + nq0) * stride + nh * w, stride, BQ, min((u32)BQ, length - nq0), w, t, threads);
            COPY_COMMIT();
        }
        // Rows past the sequence weigh nothing (an infinite log partition).
        for (u32 r = t; r < BQ; r += threads) {
            const bool inside = r < valid;
            srow[r] = inside ? lse[(u64)(start + q0 + r) * hq + h] * 1.4426950408889634f : POS_INF;
            srow[BQ + r] = inside ? dsum[(u64)(start + q0 + r) * hq + h] : 0.0f;
        }
        float st[TK][QP], dpt[TK][QP];
        if (computes) {
            UNROLL for (u32 i = 0; i < TK; ++i)
                UNROLL for (u32 j = 0; j < QP; ++j) st[i][j] = dpt[i][j] = 0.0f;
            for (u32 d = 0; d < D; d += 4u) {
                f4 kr[TK], qr[QP];
                UNROLL for (u32 i = 0; i < TK; ++i) kr[i] = LOAD4(sk + (g + KG * i) * TILE_STRIDE(D) + d);
                UNROLL for (u32 j = 0; j < QP; ++j) qr[j] = LOAD4(sqs + (c + CG * j) * TILE_STRIDE(D) + d);
                UNROLL for (u32 i = 0; i < TK; ++i)
                    UNROLL for (u32 j = 0; j < QP; ++j) st[i][j] = FMA(kr[i].w, qr[j].w, FMA(kr[i].z, qr[j].z, FMA(kr[i].y, qr[j].y, FMA(kr[i].x, qr[j].x, st[i][j]))));
            }
        }
        // This tile's cotangents (the next tile's queries, if issued, are the newest copies).
        if (more) {
            COPY_WAIT_ONE();
        } else {
            COPY_WAIT();
        }
        BARRIER();
        if (computes) {
            for (u32 d = 0; d < D; d += 4u) {
                f4 vr[TK], dor[QP];
                UNROLL for (u32 i = 0; i < TK; ++i) vr[i] = LOAD4(sv + (g + KG * i) * TILE_STRIDE(D) + d);
                UNROLL for (u32 j = 0; j < QP; ++j) dor[j] = LOAD4(sdo + (c + CG * j) * TILE_STRIDE(D) + d);
                UNROLL for (u32 i = 0; i < TK; ++i)
                    UNROLL for (u32 j = 0; j < QP; ++j) dpt[i][j] = FMA(vr[i].w, dor[j].w, FMA(vr[i].z, dor[j].z, FMA(vr[i].y, dor[j].y, FMA(vr[i].x, dor[j].x, dpt[i][j]))));
            }
            // Pᵀ and dSᵀ; a key after a row weighs nothing for it.
            UNROLL for (u32 i = 0; i < TK; ++i)
                UNROLL for (u32 j = 0; j < QP; ++j) {
                    const u32 key = k0 + g + KG * i, r = c + CG * j;
                    const float p = key > q0 + r ? 0.0f : EXP2(FMA(st[i][j], scale_log2, -srow[r]));
                    sp[(g + KG * i) * (BQ + 4u) + r] = p;
                    sds[(g + KG * i) * (BQ + 4u) + r] = p * (dpt[i][j] - srow[BQ + r]);
                }
        }
        BARRIER();
        if (computes) {
            // dV += Pᵀ dO, the tile's rows contracted.
            for (u32 r = 0; r < BQ; r += 4u) {
                f4 pr[TK];
                UNROLL for (u32 i = 0; i < TK; ++i) pr[i] = LOAD4(sp + (g + KG * i) * (BQ + 4u) + r);
                UNROLL for (u32 rr = 0; rr < 4u; ++rr) {
                    UNROLL for (u32 n = 0; n < DP / 4u; ++n) {
                        const f4 dov = LOAD4(sdo + (r + rr) * TILE_STRIDE(D) + 4u * c + 4u * CG * n);
                        UNROLL for (u32 i = 0; i < TK; ++i) {
                            const float p = rr == 0u ? pr[i].x : rr == 1u ? pr[i].y : rr == 2u ? pr[i].z : pr[i].w;
                            dv[i][4u * n] = FMA(p, dov.x, dv[i][4u * n]);
                            dv[i][4u * n + 1u] = FMA(p, dov.y, dv[i][4u * n + 1u]);
                            dv[i][4u * n + 2u] = FMA(p, dov.z, dv[i][4u * n + 2u]);
                            dv[i][4u * n + 3u] = FMA(p, dov.w, dv[i][4u * n + 3u]);
                        }
                    }
                }
            }
        }
        // Every thread is done with this tile's cotangents: the next tile's load into their buffer.
        BARRIER();
        if (more) {
            const u32 nh = kv * group + (step + 1u) / per_head, nq0 = (first + (step + 1u) % per_head) * BQ;
            load_tile<D>(sdo, ga + (u64)(start + nq0) * reads + nh * w, reads, BQ, min((u32)BQ, length - nq0), w, t, threads);
            COPY_COMMIT();
        }
        if (computes) {
            // dK += dSᵀ Q, the tile's rows contracted.
            for (u32 r = 0; r < BQ; r += 4u) {
                f4 sr[TK];
                UNROLL for (u32 i = 0; i < TK; ++i) sr[i] = LOAD4(sds + (g + KG * i) * (BQ + 4u) + r);
                UNROLL for (u32 rr = 0; rr < 4u; ++rr) {
                    UNROLL for (u32 n = 0; n < DP / 4u; ++n) {
                        const f4 qv = LOAD4(sqs + (r + rr) * TILE_STRIDE(D) + 4u * c + 4u * CG * n);
                        UNROLL for (u32 i = 0; i < TK; ++i) {
                            const float s = rr == 0u ? sr[i].x : rr == 1u ? sr[i].y : rr == 2u ? sr[i].z : sr[i].w;
                            dk[i][4u * n] = FMA(s, qv.x, dk[i][4u * n]);
                            dk[i][4u * n + 1u] = FMA(s, qv.y, dk[i][4u * n + 1u]);
                            dk[i][4u * n + 2u] = FMA(s, qv.z, dk[i][4u * n + 2u]);
                            dk[i][4u * n + 3u] = FMA(s, qv.w, dk[i][4u * n + 3u]);
                        }
                    }
                }
            }
        }
    }
    if (!computes) return;
    UNROLL for (u32 i = 0; i < TK; ++i) {
        const u32 key = k0 + g + KG * i;
        if (key >= length) continue;
        GLOBAL float* row = gy + (u64)(start + key) * stride;
        UNROLL for (u32 n = 0; n < DP / 4u; ++n)
            UNROLL for (u32 e = 0; e < 4u; ++e) {
                const u32 col = 4u * c + 4u * CG * n + e;
                if (col < w) {
                    row[(hq + kv) * w + col] = scale * dk[i][4u * n + e];
                    row[(hq + hk + kv) * w + col] = dv[i][4u * n + e];
                }
            }
    }
}

// The queries' cotangents: a block of BR query rows (of one query head of one sequence) sweeps the
// key tiles up to its last row, as the forward's block does: thread (g, c) holds rows g + RG i, their
// scores and dP = dO Vᵀ at keys c + CG j, and dQ at columns 4c + 4 CG m..; dS goes through shared
// memory to the threads that contract it with K. Key tiles are double-buffered (the next loads
// while this one's products run), the value tile loads while the scores form. Shared memory:
// (2 BR + 3 BC) (D + 4) + BR (BC + 4) floats.
template <int D, int BR, int BC, int NT, int TR>
DEVICE void queries_body(CONSTANT Sequences& sequences, u32 hq, u32 hk, u32 w, float scale, GLOBAL const float* y, GLOBAL const float* lse, GLOBAL const float* ga,
                         GLOBAL const float* dsum, GLOBAL float* gy, u32 pair, u32 tile, u32 t, u32 threads, SHARED float* smem) {
    constexpr u32 RG = BR / TR, CG = NT / RG, KP = BC / CG, DP = D / CG;
    if (pair >= sequences.count * hq) return;
    const u32 head = pair % hq, sequence = pair / hq;
    const u32 start = sequences.start[sequence], length = sequences.length[sequence];
    const u32 tiles = (length + BR - 1u) / BR;
    if (tile >= tiles) return;
    const u32 q0 = (tiles - 1u - tile) * BR;
    const u32 stride = (hq + 2u * hk) * w, reads = hq * w, kv = head / (hq / hk);
    const float scale_log2 = scale * 1.4426950408889634f;
    GLOBAL const float* yk = y + (u64)start * stride + (hq + kv) * w;
    GLOBAL const float* yv = y + (u64)start * stride + (hq + hk + kv) * w;
    SHARED float* sq = smem;
    SHARED float* sdo = sq + BR * TILE_STRIDE(D);
    SHARED float* sk = sdo + BR * TILE_STRIDE(D);
    SHARED float* sv = sk + 2u * BC * TILE_STRIDE(D);
    SHARED float* sds = sv + BC * TILE_STRIDE(D);
    const u32 keys = min(q0 + BR, length), key_tiles = (keys + BC - 1u) / BC;
    const u32 valid = min((u32)BR, length - q0);
    const bool computes = t < NT;
    const u32 g = t / CG, c = t % CG;

    load_tile<D>(sq, y + (u64)(start + q0) * stride + head * w, stride, BR, valid, w, t, threads);
    load_tile<D>(sdo, ga + (u64)(start + q0) * reads + head * w, reads, BR, valid, w, t, threads);
    load_tile<D>(sk, yk, stride, BC, min((u32)BC, keys), w, t, threads);
    COPY_COMMIT();
    load_tile<D>(sv, yv, stride, BC, min((u32)BC, keys), w, t, threads);
    COPY_COMMIT();
    float lse2[TR], dr[TR], dq[TR][DP];
    UNROLL for (u32 i = 0; i < TR; ++i) {
        const u32 row = q0 + g + RG * i;
        const bool inside = computes && row < length;
        lse2[i] = inside ? lse[(u64)(start + row) * hq + head] * 1.4426950408889634f : POS_INF;
        dr[i] = inside ? dsum[(u64)(start + row) * hq + head] : 0.0f;
        UNROLL for (u32 d = 0; d < DP; ++d) dq[i][d] = 0.0f;
    }
    for (u32 kt = 0; kt < key_tiles; ++kt) {
        const u32 k0 = kt * BC;
        const bool more = kt + 1u < key_tiles;
        SHARED float* sks = sk + (kt & 1u) * BC * TILE_STRIDE(D);
        // This tile's keys have landed (its values, the newest copies, may not have); the previous
        // tile's products are done with the buffers about to be replaced.
        COPY_WAIT_ONE();
        BARRIER();
        if (more) {
            load_tile<D>(sk + ((kt + 1u) & 1u) * BC * TILE_STRIDE(D), yk + (u64)(k0 + BC) * stride, stride, BC, min((u32)BC, keys - k0 - BC), w, t, threads);
            COPY_COMMIT();
        }
        float s[TR][KP], dp[TR][KP];
        if (computes) {
            UNROLL for (u32 i = 0; i < TR; ++i)
                UNROLL for (u32 j = 0; j < KP; ++j) s[i][j] = dp[i][j] = 0.0f;
            for (u32 d = 0; d < D; d += 4u) {
                f4 qr[TR], kr[KP];
                UNROLL for (u32 i = 0; i < TR; ++i) qr[i] = LOAD4(sq + (g + RG * i) * TILE_STRIDE(D) + d);
                UNROLL for (u32 j = 0; j < KP; ++j) kr[j] = LOAD4(sks + (c + CG * j) * TILE_STRIDE(D) + d);
                UNROLL for (u32 i = 0; i < TR; ++i)
                    UNROLL for (u32 j = 0; j < KP; ++j) s[i][j] = FMA(qr[i].w, kr[j].w, FMA(qr[i].z, kr[j].z, FMA(qr[i].y, kr[j].y, FMA(qr[i].x, kr[j].x, s[i][j]))));
            }
        }
        // This tile's values (the next tile's keys, if issued, are the newest copies).
        if (more) {
            COPY_WAIT_ONE();
        } else {
            COPY_WAIT();
        }
        BARRIER();
        if (computes) {
            for (u32 d = 0; d < D; d += 4u) {
                f4 dor[TR], vr[KP];
                UNROLL for (u32 i = 0; i < TR; ++i) dor[i] = LOAD4(sdo + (g + RG * i) * TILE_STRIDE(D) + d);
                UNROLL for (u32 j = 0; j < KP; ++j) vr[j] = LOAD4(sv + (c + CG * j) * TILE_STRIDE(D) + d);
                UNROLL for (u32 i = 0; i < TR; ++i)
                    UNROLL for (u32 j = 0; j < KP; ++j) dp[i][j] = FMA(dor[i].w, vr[j].w, FMA(dor[i].z, vr[j].z, FMA(dor[i].y, vr[j].y, FMA(dor[i].x, vr[j].x, dp[i][j]))));
            }
            // dS = P ∘ (dP − D); a key after a row weighs nothing for it.
            UNROLL for (u32 i = 0; i < TR; ++i)
                UNROLL for (u32 j = 0; j < KP; ++j) {
                    const u32 key = k0 + c + CG * j, row = q0 + g + RG * i;
                    const float p = key > row ? 0.0f : EXP2(FMA(s[i][j], scale_log2, -lse2[i]));
                    sds[(g + RG * i) * (BC + 4u) + c + CG * j] = p * (dp[i][j] - dr[i]);
                }
        }
        // Every thread is done with this tile's values: the next tile's load into their buffer.
        BARRIER();
        if (more) {
            load_tile<D>(sv, yv + (u64)(k0 + BC) * stride, stride, BC, min((u32)BC, keys - k0 - BC), w, t, threads);
            COPY_COMMIT();
        }
        if (computes) {
            // dQ += dS K, the tile's keys contracted.
            for (u32 k = 0; k < BC; k += 4u) {
                f4 sr[TR];
                UNROLL for (u32 i = 0; i < TR; ++i) sr[i] = LOAD4(sds + (g + RG * i) * (BC + 4u) + k);
                UNROLL for (u32 kk = 0; kk < 4u; ++kk) {
                    UNROLL for (u32 n = 0; n < DP / 4u; ++n) {
                        const f4 kv4 = LOAD4(sks + (k + kk) * TILE_STRIDE(D) + 4u * c + 4u * CG * n);
                        UNROLL for (u32 i = 0; i < TR; ++i) {
                            const float s = kk == 0u ? sr[i].x : kk == 1u ? sr[i].y : kk == 2u ? sr[i].z : sr[i].w;
                            dq[i][4u * n] = FMA(s, kv4.x, dq[i][4u * n]);
                            dq[i][4u * n + 1u] = FMA(s, kv4.y, dq[i][4u * n + 1u]);
                            dq[i][4u * n + 2u] = FMA(s, kv4.z, dq[i][4u * n + 2u]);
                            dq[i][4u * n + 3u] = FMA(s, kv4.w, dq[i][4u * n + 3u]);
                        }
                    }
                }
            }
        }
    }
    if (!computes) return;
    UNROLL for (u32 i = 0; i < TR; ++i) {
        const u32 row = q0 + g + RG * i;
        if (row >= length) continue;
        GLOBAL float* dst = gy + (u64)(start + row) * stride + head * w;
        UNROLL for (u32 n = 0; n < DP / 4u; ++n)
            UNROLL for (u32 e = 0; e < 4u; ++e) {
                const u32 col = 4u * c + 4u * CG * n + e;
                if (col < w) dst[col] = scale * dq[i][4u * n + e];
            }
    }
}
