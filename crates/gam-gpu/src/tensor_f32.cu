// The f32-storage twins of tensor.rs's CUDA kernels (`Storage::F32`, its module note). Values are
// float and every map runs in float (expf, tanhf, normcdff), while a reduction whose result leaves
// its row (a log partition, a KL, an entropy, a norm) sums in double: a per-element FP64 add costs
// nothing next to the row's memory traffic, and the float exps keep the FP64 units out of the inner
// loops. Every kernel keeps its float64 twin's name and parameter list, scalars included (passed as
// double, narrowed once here), so one launch serves both storages; per-row outputs stay double.
// Reductions run on warp shuffles. Rows are one 256-thread block each, as for float64.
#define BLOCK 256
#define WARPS (BLOCK / 32)
#define FULL 0xffffffffu
typedef unsigned long long u64;
#define NEG_INF __int_as_float(0xff800000)
#define POS_INF __int_as_float(0x7f800000)
#define QNAN_D __longlong_as_double(0x7ff8000000000000LL)
#define FINITE(x) (fabsf(x) <= 3.40282347e38f)

#define GRID_STRIDE(i, n) for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (u64)gridDim.x * blockDim.x)

// Butterfly reductions: every lane ends with the bitwise-same total.
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ __forceinline__ double warp_sum_d(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

// The block's total in every thread (`shared` holds WARPS values and is free again on return).
__device__ float block_sum(float v, float* shared) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = warp_sum(lane < WARPS ? shared[lane] : 0.0f);
    __syncthreads();
    return v;
}

__device__ double block_sum_d(double v, double* shared) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum_d(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = warp_sum_d(lane < WARPS ? shared[lane] : 0.0);
    __syncthreads();
    return v;
}

__device__ float block_max(float v, float* shared) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = warp_max(lane < WARPS ? shared[lane] : NEG_INF);
    __syncthreads();
    return v;
}

// A running log partition: the largest value `m` seen and `s = Σ exp(value − m)`, rescaled when the
// largest grows (one exp per value).
__device__ __forceinline__ void lse_push(float x, float& m, double& s) {
    if (x > m) {
        s = s * (double)expf(m - x) + 1.0;
        m = x;
    } else if (m > NEG_INF) {
        s += (double)expf(x - m);
    }
}

// Two running log partitions merged (symmetric in its arguments, so a butterfly agrees bitwise).
__device__ __forceinline__ void lse_merge(float& m, double& s, float m2, double s2) {
    float top = fmaxf(m, m2);
    if (top == NEG_INF) return;
    s = (m == NEG_INF ? 0.0 : s * (double)expf(m - top)) + (m2 == NEG_INF ? 0.0 : s2 * (double)expf(m2 - top));
    m = top;
}

__device__ void block_lse(float& m, double& s, float* sm, double* ss) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) lse_merge(m, s, __shfl_xor_sync(FULL, m, o), __shfl_xor_sync(FULL, s, o));
    if (lane == 0) { sm[warp] = m; ss[warp] = s; }
    __syncthreads();
    m = lane < WARPS ? sm[lane] : NEG_INF;
    s = lane < WARPS ? ss[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) lse_merge(m, s, __shfl_xor_sync(FULL, m, o), __shfl_xor_sync(FULL, s, o));
    __syncthreads();
}

// The row's log partition `max + log Σ exp(z − max)` in double, its max in `m`, its sum in `s`.
__device__ double row_lse(const float* z, unsigned int cols, float* sm, double* ss, float* m, double* s) {
    float mm = NEG_INF;
    double total = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) lse_push(z[c], mm, total);
    block_lse(mm, total, sm, ss);
    *m = mm;
    *s = total;
    return (double)mm + log(total);
}

// The first class whose cumulative probability passes `pick` (the last if none does): each thread
// sums a contiguous chunk of the row in double, then one thread walks the chunk sums and the chosen
// chunk. Every thread returns the label.
__device__ unsigned int pick_label(const float* q, unsigned int cols, double pick, double* chunks, unsigned int* label) {
    unsigned int t = threadIdx.x;
    unsigned int chunk = (cols + BLOCK - 1) / BLOCK;
    unsigned int lo = min(t * chunk, cols), hi = min(lo + chunk, cols);
    double part = 0.0;
    for (unsigned int c = lo; c < hi; c++) part += (double)q[c];
    chunks[t] = part;
    __syncthreads();
    if (t == 0) {
        unsigned int chosen = cols - 1;
        double left = pick;
        for (unsigned int k = 0; k < BLOCK; k++) {
            if (left < chunks[k]) {
                unsigned int start = k * chunk, end = min(start + chunk, cols);
                chosen = end - 1;
                for (unsigned int c = start; c < end; c++) {
                    if (left < (double)q[c]) { chosen = c; break; }
                    left -= (double)q[c];
                }
                break;
            }
            left -= chunks[k];
        }
        *label = chosen;
    }
    __syncthreads();
    unsigned int chosen = *label;
    __syncthreads();
    return chosen;
}

extern "C" __global__ void axpy(u64 n, double alpha, const float* x, float* y) {
    float a = (float)alpha;
    GRID_STRIDE(i, n) y[i] += a * x[i];
}

extern "C" __global__ void columns_of(u64 n, u64 source_cols, u64 width, u64 start, const float* source, float* out) {
    GRID_STRIDE(i, n) out[i] = source[(i / width) * source_cols + start + i % width];
}

extern "C" __global__ void set_columns(u64 n, u64 output_cols, u64 input_cols, u64 start, const float* input, float* output) {
    GRID_STRIDE(i, n) output[(i / input_cols) * output_cols + start + i % input_cols] = input[i];
}

extern "C" __global__ void hadamard(u64 n, const float* a, const float* b, float* out, int accumulate) {
    GRID_STRIDE(i, n) out[i] = accumulate ? out[i] + a[i] * b[i] : a[i] * b[i];
}

extern "C" __global__ void add_row(u64 n, unsigned int cols, double alpha, const float* row, float* x) {
    float a = (float)alpha;
    GRID_STRIDE(i, n) x[i] += a * row[i % cols];
}

extern "C" __global__ void scale_columns(u64 n, unsigned int cols, const float* x, const float* d, float* out, int accumulate) {
    GRID_STRIDE(i, n) {
        float term = x[i] * d[i % cols];
        out[i] = accumulate ? out[i] + term : term;
    }
}

extern "C" __global__ void gather_rows(u64 n, unsigned int cols, const float* table, const unsigned int* ids, float* out) {
    GRID_STRIDE(i, n) out[i] = table[(u64)ids[i / cols] * cols + i % cols];
}

// `x[i] ← factor[row] x[i]`: per-row rescaling (a swept log partition's accumulator).
extern "C" __global__ void scale_rows(u64 n, unsigned int cols, const float* factor, float* x) {
    GRID_STRIDE(i, n) x[i] *= factor[i / cols];
}

// Round to nearest even bfloat16 (the high 16 bits of the float); NaN stays a quiet NaN.
extern "C" __global__ void to_bf16(u64 n, const float* x, unsigned short* y) {
    GRID_STRIDE(i, n) {
        unsigned int bits = __float_as_uint(x[i]);
        y[i] = (bits & 0x7fffffffu) > 0x7f800000u ? (unsigned short)((bits >> 16) | 0x40u)
                                                  : (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    }
}

__device__ float law_value(unsigned int code, float t, float c) {
    switch (code) {
        case 0: return t > 0.0f ? t : 0.0f;
        case 1: return t;
        case 2: return 0.0f;
        case 3: return t / (1.0f + expf(-t));
        case 4: return t * normcdff(t);
        default: {
            float inner = c * (t + 0.044715f * t * t * t);
            return 0.5f * t * (1.0f + tanhf(inner));
        }
    }
}

__device__ float law_slope(unsigned int code, float t, float c) {
    switch (code) {
        case 0: return t > 0.0f ? 1.0f : 0.0f;
        case 1: return 1.0f;
        case 2: return 0.0f;
        case 3: {
            float sigma = 1.0f / (1.0f + expf(-t));
            return sigma * (1.0f + t * (1.0f - sigma));
        }
        case 4: return normcdff(t) + t * (expf(-0.5f * t * t) * 0.398942280401432678f);
        default: {
            float inner = c * (t + 0.044715f * t * t * t);
            float th = tanhf(inner);
            return 0.5f * (1.0f + th) + 0.5f * t * (1.0f - th * th) * c * (1.0f + 3.0f * 0.044715f * t * t);
        }
    }
}

extern "C" __global__ void laws(u64 n, unsigned int cols, const float* x, const float* g, int slopes, const unsigned int* codes, double c, float* out) {
    float cf = (float)c;
    GRID_STRIDE(i, n) {
        unsigned int code = codes[i % cols];
        out[i] = slopes ? g[i] * law_slope(code, x[i], cf) : law_value(code, x[i], cf);
    }
}

// mode 0: value; 1: cotangent given g; 2: tangent along g.
extern "C" __global__ void rms(unsigned int rows, unsigned int cols, int mode, double epsilon, const float* x, const float* g, float* out) {
    __shared__ float shared[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (u64)r * cols;
    float* o = out + (u64)r * cols;
    float squares = 0.0f, inner = 0.0f;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        squares += xr[c] * xr[c];
        if (mode != 0) inner += xr[c] * g[(u64)r * cols + c];
    }
    float n = (float)cols;
    float mean = block_sum(squares, shared) / n;
    float scale = 1.0f / sqrtf(mean + (float)epsilon);
    if (mode == 0) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) o[c] = xr[c] * scale;
        return;
    }
    float dot = block_sum(inner, shared);
    const float* gr = g + (u64)r * cols;
    if (mode == 1) {
        float k = scale * scale * scale / n * dot;
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) o[c] = scale * gr[c] - k * xr[c];
    } else {
        float ds = -0.5f * scale * scale * scale * (2.0f * dot / n);
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) o[c] = gr[c] * scale + xr[c] * ds;
    }
}

extern "C" __global__ void rotate_planes(unsigned int rows, unsigned int cols, unsigned int planes, int half_split, double sign,
                                         const float* x, const float* cosines, const float* sines, float* out) {
    float sg = (float)sign;
    GRID_STRIDE(i, (u64)rows * planes) {
        u64 r = i / planes;
        unsigned int p = (unsigned int)(i % planes);
        unsigned int a = half_split ? p : 2 * p;
        unsigned int b = half_split ? p + planes : 2 * p + 1;
        float c = cosines[i];
        float s = sg * sines[i];
        float xa = x[r * cols + a], xb = x[r * cols + b];
        out[r * cols + a] = c * xa - s * xb;
        out[r * cols + b] = s * xa + c * xb;
    }
}

extern "C" __global__ void softmax_rows(unsigned int rows, unsigned int width, int causal, unsigned int start, unsigned int period, float* s) {
    __shared__ float shared[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    float* row = s + (u64)r * width;
    unsigned int valid = causal ? start + r % period + 1 : width;
    float m = NEG_INF;
    for (unsigned int c = threadIdx.x; c < valid; c += BLOCK) m = fmaxf(m, row[c]);
    m = block_max(m, shared);
    float total = 0.0f;
    for (unsigned int c = threadIdx.x; c < width; c += BLOCK) {
        if (c < valid) {
            float e = expf(row[c] - m);
            row[c] = e;
            total += e;
        } else {
            row[c] = 0.0f;
        }
    }
    float inverse = 1.0f / block_sum(total, shared);
    for (unsigned int c = threadIdx.x; c < valid; c += BLOCK) row[c] *= inverse;
}

extern "C" __global__ void softmax_backward(unsigned int rows, unsigned int cols, const float* alpha, const float* d, float* out) {
    __shared__ float shared[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const float* a = alpha + (u64)r * cols;
    const float* dr = d + (u64)r * cols;
    float partial = 0.0f;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += a[c] * dr[c];
    float mean = block_sum(partial, shared);
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) out[(u64)r * cols + c] = a[c] * (dr[c] - mean);
}

// Per row its probabilities in place and (log partition, Σ p log p) in double.
extern "C" __global__ void softmax_stats_rows(unsigned int rows, unsigned int cols, float* logits,
                                             const unsigned int* scored, int use_scored, double* out) {
    __shared__ float sm[WARPS];
    __shared__ double ss[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    float* z = logits + (u64)r * cols;
    if (use_scored && !scored[r]) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = 0.0f;
        if (threadIdx.x == 0) { out[(u64)r * 2] = 0.0; out[(u64)r * 2 + 1] = 0.0; }
        return;
    }
    int bad = 0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) bad |= !FINITE(z[c]);
    if (__syncthreads_or(bad)) {
        if (threadIdx.x == 0) { out[(u64)r * 2] = QNAN_D; out[(u64)r * 2 + 1] = QNAN_D; }
        return;
    }
    float m;
    double sum;
    double log_partition = row_lse(z, cols, sm, ss, &m, &sum);
    double log_sum = log_partition - (double)m;
    float inverse = (float)(1.0 / sum);
    double entropy = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        float shifted = z[c] - m;
        float probability = expf(shifted) * inverse;
        if (probability > 0.0f) entropy += (double)probability * ((double)shifted - log_sum);
        z[c] = probability;
    }
    double total = block_sum_d(entropy, ss);
    if (threadIdx.x == 0) { out[(u64)r * 2] = log_partition; out[(u64)r * 2 + 1] = total; }
}

// KL(softmax(t) ‖ softmax(z)) per row as Σ p (d − D), d = t − z the logit gap and D = log Z_t −
// log Z_z in double: exact when the gap is a constant, whatever the logits' size.
extern "C" __global__ void kl_rows(unsigned int rows, unsigned int cols, const float* target, float* logits,
                                   const unsigned int* scored, int use_scored, int gradient, double* kl) {
    __shared__ float sm[WARPS];
    __shared__ double ss[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const float* t = target + (u64)r * cols;
    float* z = logits + (u64)r * cols;
    if (use_scored && scored[r] == 0) {
        if (gradient) for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = 0.0f;
        if (threadIdx.x == 0) kl[r] = 0.0;
        return;
    }
    float mt = NEG_INF, mz = NEG_INF;
    double st = 0.0, sz = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        lse_push(t[c], mt, st);
        lse_push(z[c], mz, sz);
    }
    block_lse(mt, st, sm, ss);
    block_lse(mz, sz, sm, ss);
    double gap = ((double)mt + log(st)) - ((double)mz + log(sz));
    float it = (float)(1.0 / st), iz = (float)(1.0 / sz);
    double acc = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        float tc = t[c], zc = z[c];
        float p = expf(tc - mt) * it;
        if (p > 0.0f) acc += (double)p * ((double)(tc - zc) - gap);
        if (gradient) z[c] = expf(zc - mz) * iz - p;
    }
    double total = block_sum_d(acc, ss);
    if (threadIdx.x == 0) kl[r] = total;
}

extern "C" __global__ void sampled_cotangent(unsigned int rows, unsigned int cols, float* logits, const float* uniforms,
                                             const unsigned int* scored, int use_scored) {
    __shared__ float sm[WARPS];
    __shared__ double ss[WARPS];
    __shared__ double chunks[BLOCK];
    __shared__ unsigned int label;
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    float* z = logits + (u64)r * cols;
    if (use_scored && scored[r] == 0) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = 0.0f;
        return;
    }
    float m;
    double sum;
    row_lse(z, cols, sm, ss, &m, &sum);
    float inverse = (float)(1.0 / sum);
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = expf(z[c] - m) * inverse;
    __syncthreads();
    unsigned int chosen = pick_label(z, cols, (double)uniforms[r], chunks, &label);
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] -= (c == chosen) ? 1.0f : 0.0f;
}

extern "C" __global__ void block_products(u64 n, unsigned int cols, unsigned int blocks,
    const float* left, const float* right, const unsigned int* offsets, float* out) {
    GRID_STRIDE(i, n) {
        u64 row = i / blocks;
        unsigned int block = (unsigned int)(i % blocks);
        float sum = 0.0f;
        for (unsigned int c = offsets[block]; c < offsets[block + 1]; c++)
            sum += left[row * cols + c] * right[row * cols + c];
        out[i] = sum;
    }
}

extern "C" __global__ void sampled_head_cotangent(unsigned int rows, unsigned int classes, unsigned int width,
    const float* probabilities, const float* mean, const float* head, int transposed,
    const float* uniforms, const unsigned int* scored, int use_scored, float* out) {
    __shared__ double chunks[BLOCK];
    __shared__ unsigned int label;
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    if (use_scored && scored[r] == 0) {
        for (unsigned int h = threadIdx.x; h < width; h += BLOCK) out[(u64)r * width + h] = 0.0f;
        return;
    }
    unsigned int chosen = pick_label(probabilities + (u64)r * classes, classes, (double)uniforms[r], chunks, &label);
    for (unsigned int h = threadIdx.x; h < width; h += BLOCK) {
        u64 index = transposed ? (u64)h * classes + chosen : (u64)chosen * width + h;
        out[(u64)r * width + h] = mean[(u64)r * width + h] - head[index];
    }
}

extern "C" __global__ void adam(u64 n, double rate, double beta1, double beta2, double epsilon, double c1, double c2,
    const float* g, float* m, float* v, float* w) {
    float b1 = (float)beta1, b2 = (float)beta2, o1 = (float)(1.0 - beta1), o2 = (float)(1.0 - beta2);
    float step = (float)rate, eps = (float)epsilon, k1 = (float)(1.0 / c1), k2 = (float)(1.0 / c2);
    GRID_STRIDE(i, n) {
        float gi = g[i];
        float mi = b1 * m[i] + o1 * gi;
        float vi = b2 * v[i] + o2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        w[i] -= step * (mi * k1) / (sqrtf(vi * k2) + eps);
    }
}

// Whether (ka, ia) ranks before (kb, ib): larger key first, ties by column.
__device__ bool ranks_before(float ka, unsigned int ia, float kb, unsigned int ib) {
    return ka > kb || (ka == kb && ia < ib);
}

__device__ float ranking_key(float size, float bits) {
    if (bits > 0.0f) return size / bits;
    return size > 0.0f ? POS_INF : 0.0f;
}

// The float64 kernel's `select_sets` on float values (its ranking and best-prefix scan).
extern "C" __global__ void select_sets(unsigned int rows, unsigned int cols, unsigned int width,
    const float* a, const float* q, const float* bits, const float* left, const float* weight,
    float* keys, unsigned int* order, float* sizes, float* mask) {
    __shared__ float shared[WARPS];
    float* key = keys + (u64)blockIdx.x * width;
    unsigned int* idx = order + (u64)blockIdx.x * width;
    float* sr = sizes + (u64)blockIdx.x * width;
    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        float* m = mask + (u64)r * cols;
        float partial = 0.0f, off = 0.0f, on_bits = 0.0f;
        for (unsigned int c = threadIdx.x; c < width; c += BLOCK) {
            if (c < cols) {
                sr[c] = fabsf(a[(u64)r * cols + c]) * q[(u64)r * cols + c];
                partial += sr[c];
                if (m[c] == 0.0f) off += sr[c]; else on_bits += bits[c];
                key[c] = ranking_key(sr[c], bits[c]);
            } else {
                key[c] = NEG_INF;
            }
            idx[c] = c;
        }
        float all = block_sum(partial, shared);
        float held_off = block_sum(off, shared);
        float held_listed = block_sum(on_bits, shared);
        for (unsigned int k = 2; k <= width; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = threadIdx.x; i < width; i += BLOCK) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool first = (i & k) == 0;
                        bool swap = first ? ranks_before(key[l], idx[l], key[i], idx[i]) : ranks_before(key[i], idx[i], key[l], idx[l]);
                        if (swap) {
                            float tk = key[i]; key[i] = key[l]; key[l] = tk;
                            unsigned int ti = idx[i]; idx[i] = idx[l]; idx[l] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
        if (threadIdx.x == 0) {
            float w = weight[r], lr = left[r];
            float listed = 0.0f, bound = lr + all;
            float best_code = w * bound * bound;
            unsigned int best = 0;
            for (unsigned int k = 0; k < cols; k++) {
                unsigned int c = idx[k];
                listed += bits[c];
                bound -= sr[c];
                float code = listed + w * bound * bound;
                if (code < best_code) { best = k + 1; best_code = code; }
            }
            float held = lr + held_off;
            if (best_code < held_listed + w * held * held) {
                for (unsigned int c = 0; c < cols; c++) m[c] = 0.0f;
                for (unsigned int k = 0; k < best; k++) m[idx[k]] = 1.0f;
            }
            bound = lr;
            for (unsigned int c = 0; c < cols; c++) if (m[c] == 0.0f) bound += sr[c];
            for (unsigned int sweep = 0; sweep < cols; sweep++) {
                int flipped = 0;
                for (unsigned int c = 0; c < cols; c++) {
                    int on = m[c] == 1.0f;
                    float next = on ? bound + sr[c] : fmaxf(bound - sr[c], 0.0f);
                    float delta = (on ? -bits[c] : bits[c]) + w * (next * next - bound * bound);
                    if (delta < 0.0f) { m[c] = on ? 0.0f : 1.0f; bound = next; flipped = 1; }
                }
                if (!flipped) break;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void divide_sums(u64 n, unsigned int cols, const float* r, const float* c, double floor, float* x) {
    float f = (float)floor;
    GRID_STRIDE(i, n) {
        float sum = r[i / cols] + c[i % cols];
        x[i] = sum > f ? x[i] / sum : 0.0f;
    }
}

extern "C" __global__ void box_charge(unsigned int rows, unsigned int cols, const float* z, const float* mask,
    const float* q, double* norms, float* cot, float* coefficient) {
    __shared__ double shared[WARPS];
    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        u64 base = (u64)r * cols;
        double partial = 0.0;
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += (double)((1.0f - mask[base + c]) * fabsf(z[base + c]) * q[base + c]);
        double total = block_sum_d(partial, shared);
        if (threadIdx.x == 0) norms[r] = total;
        float n = (float)total;
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
            float off = 1.0f - mask[base + c], zc = z[base + c], qc = q[base + c];
            if (off != 0.0f && zc != 0.0f) cot[base + c] += n * off * (zc > 0.0f ? 1.0f : -1.0f) * qc;
            coefficient[base + c] = qc > 0.0f ? n * off * fabsf(zc) / qc : 0.0f;
        }
    }
}

extern "C" __global__ void argmax_rows(unsigned int rows, unsigned int cols, const float* x, double* out) {
    __shared__ float values[BLOCK];
    __shared__ unsigned int columns[BLOCK];
    unsigned int r = blockIdx.x, t = threadIdx.x;
    if (r >= rows) return;
    const float* z = x + (u64)r * cols;
    float best = NEG_INF;
    unsigned int at = cols;
    for (unsigned int c = t; c < cols; c += BLOCK) if (z[c] > best) { best = z[c]; at = c; }
    values[t] = best;
    columns[t] = at;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s && (values[t + s] > values[t] || (values[t + s] == values[t] && columns[t + s] < columns[t]))) {
            values[t] = values[t + s];
            columns[t] = columns[t + s];
        }
        __syncthreads();
    }
    if (t == 0) out[r] = columns[0] == cols ? 0.0 : (double)columns[0];
}

extern "C" __global__ void fill_entries(u64 n, const unsigned int* at, double value, float* x) {
    float v = (float)value;
    GRID_STRIDE(i, n) x[at[i]] = v;
}

extern "C" __global__ void softmax_quadratic(unsigned int rows, unsigned int cols, const float* logits, const float* tangent, double* out) {
    __shared__ float sm[WARPS];
    __shared__ double ss[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const float* z = logits + (u64)r * cols;
    const float* t = tangent + (u64)r * cols;
    float m;
    double sum;
    row_lse(z, cols, sm, ss, &m, &sum);
    float inverse = (float)(1.0 / sum);
    double partial = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += (double)(expf(z[c] - m) * inverse * t[c]);
    float mean = (float)block_sum_d(partial, ss);
    partial = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        float q = expf(z[c] - m) * inverse;
        float centred = t[c] - mean;
        partial += (double)(q * centred * centred);
    }
    double total = block_sum_d(partial, ss);
    if (threadIdx.x == 0) out[r] = total;
}

// One chunk of a swept log partition (`Device::head_log_partition`): row r's logits for the chunk's
// `cols` classes merge into its running largest `m[r]` and sum `s[r]`, `factor[r]` gets exp(old
// largest − new) (the rescaling of what the row accumulated before), and when `expected` the
// logits become exp(logit − new largest) in place, ready to weight the chunk's head rows.
extern "C" __global__ void head_chunk(unsigned int rows, unsigned int cols, float* logits, float* m, double* s, float* factor, int expected) {
    __shared__ float shared[WARPS];
    __shared__ double sd[WARPS];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    float* z = logits + (u64)r * cols;
    float top = NEG_INF;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) top = fmaxf(top, z[c]);
    top = block_max(top, shared);
    float old = m[r];
    float next = fmaxf(old, top);
    double part = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        float e = next == NEG_INF ? 0.0f : expf(z[c] - next);
        part += (double)e;
        if (expected) z[c] = e;
    }
    part = block_sum_d(part, sd);
    if (threadIdx.x == 0) {
        float f = old == NEG_INF ? 0.0f : expf(old - next);
        s[r] = s[r] * (double)f + part;
        m[r] = next;
        factor[r] = f;
    }
}

// A swept log partition's results: log Z_r = m_r + log s_r into `out`, and the accumulated expected
// head row divided by s_r; unscored rows get zero of both.
extern "C" __global__ void head_finish(unsigned int rows, unsigned int width, const float* m, const double* s,
    const unsigned int* scored, int use_scored, int expected, float* mean, double* out) {
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    int on = !use_scored || scored[r] != 0;
    if (threadIdx.x == 0) out[r] = on ? (double)m[r] + log(s[r]) : 0.0;
    if (!expected) return;
    float inverse = on ? (float)(1.0 / s[r]) : 0.0f;
    for (unsigned int h = threadIdx.x; h < width; h += BLOCK) {
        u64 i = (u64)r * width + h;
        mean[i] = on ? mean[i] * inverse : 0.0f;
    }
}
