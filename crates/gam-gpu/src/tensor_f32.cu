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

// A running log partition: the largest value `m` seen and `s = Σ exp(value − m)`, one float exp per
// value; the rare rescalings as the largest grows (about log n per thread, and the merges) take a
// double exp, so `s` carries only its terms' own float errors.
__device__ __forceinline__ void lse_push(float x, float& m, double& s) {
    if (x > m) {
        s = (m == NEG_INF ? 0.0 : s * exp((double)m - (double)x)) + 1.0;
        m = x;
    } else if (m > NEG_INF) {
        s += (double)expf(x - m);
    }
}

// Two running log partitions merged (symmetric in its arguments, so a butterfly agrees bitwise).
__device__ __forceinline__ void lse_merge(float& m, double& s, float m2, double s2) {
    float top = fmaxf(m, m2);
    if (top == NEG_INF) return;
    s = (m == NEG_INF ? 0.0 : s * exp((double)m - (double)top)) + (m2 == NEG_INF ? 0.0 : s2 * exp((double)m2 - (double)top));
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

extern "C" __global__ void fill(u64 n, double value, float* x) {
    float v = (float)value;
    GRID_STRIDE(i, n) x[i] = v;
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

// Heads between a row-major `rows × cols` tensor's column blocks (`heads` of `width` from `start`)
// and head-major order (row `(b·heads + h)·length + l` for row `b·length + l`): `merge` = 0 copies
// row-major to head-major, 1 back. The first `2·planes` columns of each head turn by the tables
// (`rows × planes`, pairing as `rotate_planes`, `sign` −1 backwards) on the way.
extern "C" __global__ void heads_permute(unsigned int rows, unsigned int cols, unsigned int start, unsigned int heads, unsigned int width,
                                         unsigned int length, unsigned int planes, int half_split, double sign, int merge,
                                         const float* x, const float* cosines, const float* sines, float* out) {
    GRID_STRIDE(i, (u64)rows * heads * width) {
        unsigned int j = (unsigned int)(i % width);
        u64 t = i / width;
        unsigned int l = (unsigned int)(t % length);
        t /= length;
        unsigned int h = (unsigned int)(t % heads);
        u64 r = (t / heads) * length + l;
        u64 wide = r * cols + start + (u64)h * width, narrow = i - j;
        const float* src = x + (merge ? narrow : wide);
        float v = src[j];
        if (j < 2 * planes) {
            unsigned int p = half_split ? (j < planes ? j : j - planes) : j / 2;
            int first = half_split ? j < planes : (j % 2) == 0;
            unsigned int partner = half_split ? (first ? j + planes : j - planes) : (first ? j + 1 : j - 1);
            float c = cosines[r * planes + p], s = (float)sign * sines[r * planes + p];
            float o = src[partner];
            v = first ? c * v - s * o : s * o + c * v;
        }
        out[(merge ? wide : narrow) + j] = v;
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

// A uniform on [0, 1) for chunk `k` of row `r` of a drawn sweep whose row uniform is `u`: output
// k + 1 of SplitMix64 (Steele, Lea and Flood 2014) seeded with the row and the uniform's bits.
__device__ double chunk_uniform(float u, unsigned int r, unsigned int k) {
    u64 z = (((u64)r << 32) | (u64)__float_as_uint(u)) + 0x9E3779B97F4A7C15ull * ((u64)k + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

// One chunk of a swept log partition (`Device::head_log_partition`): row r's logits for the chunk's
// `cols` classes merge into its running largest `m[r]` and sum `s[r]`, `factor[r]` gets exp(old
// largest − new) (the rescaling of what the row accumulated before), and when `expected` the
// logits become exp(logit − new largest) in place, ready to weight the chunk's head rows. With
// `draw` (`Device::head_log_partition_drawn`), chunk `chunk` of `chunks` (its first class `start`)
// also draws row r's candidate class, each with probability proportional to its exponential: by
// the inverse distribution function over the classes in the order the threads hold them (thread
// t's classes t, t + BLOCK, ..., then thread t + 1's), with `chunk_uniform`; the candidate goes to
// `candidates[r * chunks + chunk]` and the chunk's log mass `next + log Σ e` to `masses` there.
extern "C" __global__ void head_chunk(unsigned int rows, unsigned int cols, float* logits, float* m, double* s, float* factor, int expected,
    const float* uniforms, unsigned int start, unsigned int chunk, unsigned int chunks, unsigned int* candidates, double* masses, int draw) {
    __shared__ float shared[WARPS];
    __shared__ double sd[WARPS];
    __shared__ double parts[BLOCK];
    __shared__ unsigned int owner;
    __shared__ double rest;
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
    if (draw) parts[threadIdx.x] = part;
    part = block_sum_d(part, sd);
    if (threadIdx.x == 0) {
        double f = old == NEG_INF ? 0.0 : exp((double)old - (double)next);
        s[r] = s[r] * f + part;
        m[r] = next;
        factor[r] = (float)f;
    }
    if (!draw) return;
    if (threadIdx.x < 32) {
        // Warp 0 finds the thread whose classes hold the pick: each lane sums BLOCK / 32 threads'
        // masses, the lanes' sums are scanned, and the lane holding the pick walks its threads.
        // Past the end by rounding, the pick goes to the last thread with mass.
        const unsigned int per = BLOCK / 32;
        unsigned int lane = threadIdx.x;
        double own = 0.0;
        for (unsigned int i = 0; i < per; i++) own += parts[lane * per + i];
        double scan = own;
        for (unsigned int o = 1; o < 32; o <<= 1) {
            double v = __shfl_up_sync(FULL, scan, o);
            if (lane >= o) scan += v;
        }
        double pick = chunk_uniform(uniforms[r], r, chunk) * part;
        unsigned int hit = __ballot_sync(FULL, pick < scan), with = __ballot_sync(FULL, own > 0.0);
        unsigned int at = hit ? __ffs(hit) - 1 : (with ? 31 - __clz(with) : 0);
        double before = __shfl_sync(FULL, scan - own, at);
        if (lane == at) {
            double left = pick - before;
            unsigned int k = lane * per;
            for (unsigned int t = lane * per; t < (lane + 1) * per; t++) {
                if (parts[t] > 0.0) k = t;
                if (left < parts[t]) { k = t; break; }
                left -= parts[t];
            }
            owner = k;
            rest = left;
            masses[(u64)r * chunks + chunk] = part > 0.0 ? (double)next + log(part) : (double)NEG_INF;
        }
    }
    __syncthreads();
    if (threadIdx.x == owner) {
        double left = rest;
        unsigned int chosen = owner < cols ? owner : 0;
        for (unsigned int c = owner; c < cols; c += BLOCK) {
            double e = expected ? (double)z[c] : (double)(next == NEG_INF ? 0.0f : expf(z[c] - next));
            if (e > 0.0) chosen = c;
            if (left < e) break;
            left -= e;
        }
        candidates[(u64)r * chunks + chunk] = start + chosen;
    }
}

// A drawn sweep's labels (`Device::head_log_partition_drawn`), after `head_finish`: per row the
// chunk by the inverse distribution function over the chunks' masses exp(masses − largest) with
// the row's uniform, its candidate the label y, and out_r = mean_r − head[y] (the expected head
// row less the label's); an unscored row gets zero.
extern "C" __global__ void head_draw(unsigned int rows, unsigned int chunks, unsigned int classes, unsigned int width,
    const float* uniforms, const unsigned int* candidates, const double* masses, const unsigned int* scored, int use_scored,
    const float* mean, const float* head, int transposed, float* out) {
    __shared__ unsigned int label;
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    float* o = out + (u64)r * width;
    if (use_scored && scored[r] == 0) {
        for (unsigned int h = threadIdx.x; h < width; h += BLOCK) o[h] = 0.0f;
        return;
    }
    if (threadIdx.x < 32) {
        // Warp 0 picks the chunk: each lane holds a run of consecutive chunks, the lanes' sums of
        // exp(mass − largest) are scanned, and the lane holding the pick walks its run (past the
        // end by rounding, the last chunk with mass).
        unsigned int lane = threadIdx.x, per = (chunks + 31) / 32;
        unsigned int lo = min(lane * per, chunks), hi = min(lo + per, chunks);
        const double* w = masses + (u64)r * chunks;
        double top = (double)NEG_INF;
        for (unsigned int k = lo; k < hi; k++) top = fmax(top, w[k]);
        for (int o = 16; o > 0; o >>= 1) top = fmax(top, __shfl_xor_sync(FULL, top, o));
        double own = 0.0;
        if (top != (double)NEG_INF)
            for (unsigned int k = lo; k < hi; k++) own += exp(w[k] - top);
        double scan = own;
        for (unsigned int o = 1; o < 32; o <<= 1) {
            double v = __shfl_up_sync(FULL, scan, o);
            if (lane >= o) scan += v;
        }
        double pick = (double)uniforms[r] * __shfl_sync(FULL, scan, 31);
        unsigned int hit = __ballot_sync(FULL, pick < scan), with = __ballot_sync(FULL, own > 0.0);
        unsigned int at = hit ? __ffs(hit) - 1 : (with ? 31 - __clz(with) : 0);
        double before = __shfl_sync(FULL, scan - own, at);
        if (lane == at) {
            double left = pick - before;
            unsigned int chosen = lo < chunks ? lo : 0;
            for (unsigned int k = lo; k < hi; k++) {
                double p = exp(w[k] - top);
                if (p > 0.0) chosen = k;
                if (left < p) break;
                left -= p;
            }
            label = candidates[(u64)r * chunks + chosen];
        }
    }
    __syncthreads();
    unsigned int y = label;
    for (unsigned int h = threadIdx.x; h < width; h += BLOCK)
        o[h] = mean[(u64)r * width + h] - head[transposed ? (u64)h * classes + y : (u64)y * width + h];
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

// The Ozaki split of a Gram matrix's operand (Device::gram_split). Per column j of the rows × cols
// row-major x, the exponent e_j with every |x_kj| < 2^e_j (0 for a zero column).
extern "C" __global__ void split_exponents(unsigned int rows, unsigned int cols, const float* x, int* e) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float m = 0.0f;
    for (unsigned int k = 0; k < rows; ++k) m = fmaxf(m, fabsf(x[(u64)k * cols + j]));
    int p = 0;
    if (m > 0.0f) frexpf(m, &p);
    e[j] = p;
}

// The int8 slices a_s of x: x_kj = 2^(e_j - 7) Σ_{s < slices} a_s,kj 2^(-7 s) + r_kj with
// |r_kj| < 2^(e_j - 7 slices), each |a| ≤ 127 (truncation of a scaled value below 128 in magnitude;
// every step exact in float). Slice s is column-major (rows padded with zeros to `stride`) at
// out + s · cols · stride.
extern "C" __global__ void split_slices(unsigned int rows, unsigned int cols, unsigned int stride, unsigned int slices, const float* x, const int* e, signed char* out) {
    GRID_STRIDE(i, (u64)stride * cols) {
        unsigned int j = (unsigned int)(i / stride), k = (unsigned int)(i % stride);
        float v = k < rows ? ldexpf(x[(u64)k * cols + j], 7 - e[j]) : 0.0f;
        for (unsigned int s = 0; s < slices; ++s) {
            float a = truncf(v);
            out[(u64)s * cols * stride + i] = (signed char)a;
            v = (v - a) * 128.0f;
        }
    }
}

// g (n × n row-major float64) += 2^(e_i + e_j - 15 - 7 shift) (c_ij + c_ji), c the column-major
// n × n int32 sum of one shift's slice products (twice each a_sᵀ a_t with s < t, once a_sᵀ a_s),
// whose symmetric part is the shift's whole sum Σ_{s+t=shift} a_sᵀ a_t; the halving is exact. 32 × 32
// tiles through shared memory, so both c_ij and c_ji are read along their columns.
extern "C" __global__ void split_combine(unsigned int n, int shift, const int* c, const int* e, double* g) {
    __shared__ int tile[32][33];
    unsigned int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
    for (unsigned int y = threadIdx.y; y < 32; y += blockDim.y) {
        unsigned int r = r0 + threadIdx.x, col = c0 + y;
        tile[y][threadIdx.x] = (r < n && col < n) ? c[r + (u64)col * n] : 0;
    }
    __syncthreads();
    for (unsigned int y = threadIdx.y; y < 32; y += blockDim.y) {
        unsigned int r = r0 + y, col = c0 + threadIdx.x;
        if (r < n && col < n) {
            double v = (double)tile[threadIdx.x][y] + (double)c[col + (u64)r * n];
            g[(u64)r * n + col] += ldexp(v, e[r] + e[col] - 15 - 7 * shift);
        }
    }
}
