// NVRTC device source for the SAE FUSED reconstruction + on-device sae_dot
// contraction (#932 → A100 100x cutover). One thread-block per row.
//
// Provenance / single source: this is a byte-faithful port of the FUSED
// scalar program in `scratchpad/sae_fused_bench.rs` (`fused_resid` / `fused_gram`),
// which is in turn the contraction of the production per-row jet channels
//   first[a][c]  = ∂ẑ_c/∂p_a        (consumer: construction.rs:8210, Gram)
//   second[a][b][c] = ∂²ẑ_c/∂p_a∂p_b (consumer: construction.rs:9239, residual)
// against the per-row error_metric / themselves. The decisive design choice
// (proto README_survival_jet_932.md, "Key engineering finding"): the full
// out_dim-wide tensors are NEVER needed off-device — every consumer contracts
// the p-axis (`sae_dot`, arrow_solver.rs:349) into O(q²) scalars. Shipping the
// full p-wide tensor back is transfer-bound (~17x, 424 B/row genus). Reducing
// on-device and returning only the tiny q×q slabs recovers the kernel-bound
// 100x+ (162x end-to-end / 504x kernel-only measured on the survival sibling).
//
// FUSION (the win): the p-axis is contracted BEFORE the q² blow-up.
//   Residual: push error_metric into the decoder ONCE  ê_{k,b}=Σ_c dec_{k,b,c}·e_c
//             (A·NB·p — the ONLY p-loop), then R[a][b] = chain-rule coeffs · ē.
//             Work:  q²·p  →  A·NB·p + q²·A.   p DROPS out of the q² loop.
//
// Variable / primary layout (mirrors SaeLocalRowVar + sae_fused_bench.rs):
//   var <  A   → Logit{atom = var}
//   var >= A   → Coord{atom = (var-A)/L, axis = (var-A)%L}
//   Q = A*(1+L) primaries per row.
//
// Compiled by the host via `cudarc::nvrtc::compile_ptx` with the four shape
// macros prepended (#define AA / LL / NB_ / PP), exactly like the sibling
// `sae_rowjet.rs::softmax_kernel_source`. Full f64, no fast-math.

// ---- Shape macros. The host prepends concrete values; the #ifndef defaults
// let this file NVRTC-/syntax-compile standalone (mirrors survival_rowjet's
// `#define K 4`). AA=atoms, LL=latent dim/atom, NB_=basis fns/atom, PP=out_dim.
#ifndef AA
#define AA 8
#endif
#ifndef LL
#define LL 2
#endif
#ifndef NB_
#define NB_ 6
#endif
#ifndef PP
#define PP 16
#endif

// NVRTC does NOT predefine INFINITY (it is a <math.h> macro, not a CUDA
// builtin); the softmax max-reduction seeds with -INFINITY, so without this the
// module fails to compile and the path silently falls back to CPU (same genus
// as the M_PI / INFINITY fixes in survival_rowjet_kernel.cu and sae_rowjet.rs).
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
#endif

#define QQ (AA * (1 + LL))
// LL may be 0 (pure-softmax rows, Q==A): guard the compile-time divisor so the
// coord index expressions are well-formed constants even though the coord
// branches are never taken at runtime when LL==0.
#define LDIV ((LL) > 0 ? (LL) : 1)
// Zero-length shared arrays are ill-formed; clamp coord-channel sizes to >=1.
#define ALSZ ((AA * LL) > 0 ? (AA * LL) : 1)
#define AL2SZ ((AA * LL * LL) > 0 ? (AA * LL * LL) : 1)

__device__ __forceinline__ int is_logit(int v) { return v < AA; }
__device__ __forceinline__ int coord_atom(int v) { return (v - AA) / LDIV; }
__device__ __forceinline__ int coord_axis(int v) { return (v - AA) % LDIV; }

// ─────────────────────────────────────────────────────────────────────────
// FUSED RESIDUAL kernel (the jet-bound consumer, biggest win).
//
//   out[row][a][b] = Σ_c error_metric_row[c] · second_row[a][b][c]
//
// returning only the Q×Q row-major slab per row (NOT the p-wide second tensor).
// This is exactly the `r_ab = sae_dot(&error_metric, &jets.second[a][b])` slab
// the arrow residual-curvature consumer (construction.rs:9239) reads, the SAME
// `row_hessian_slabs` layout the resident workspace uploads.
//
// Inputs are SoA, flat, row-major. phi/dphi/d2phi/decoder are RESIDENT across
// inner-Newton / REML iterations; only `error_metric` (and `logits`, when the
// gate moves) is refreshed per step.
//   logits   [n * AA]
//   phi      [n * AA * NB_]
//   dphi     [n * AA * NB_ * LL]
//   d2phi    [n * AA * NB_ * LL * LL]   (symmetrised, as the CPU stores it)
//   decoder  [n * AA * NB_ * PP]
//   error    [n * PP]                   (per-row error_metric = M_n r_n)
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(128, 1) sae_fused_residual(
        int n,
        double inv_tau,
        const double* __restrict__ logits,
        const double* __restrict__ phi,
        const double* __restrict__ dphi,
        const double* __restrict__ d2phi,
        const double* __restrict__ decoder,
        const double* __restrict__ error,
        double* __restrict__ out)            // [n * QQ * QQ]
{
    const int row = blockIdx.x;
    if (row >= n) return;
    const int t = threadIdx.x;
    const int nt = blockDim.x;

    // Per-row input base offsets.
    const double* L   = logits  + (size_t)row * AA;
    const double* PHI = phi     + (size_t)row * AA * NB_;
    const double* DPH = dphi    + (size_t)row * AA * NB_ * LL;
    const double* D2P = d2phi   + (size_t)row * AA * NB_ * LL * LL;
    const double* DEC = decoder + (size_t)row * AA * NB_ * PP;
    const double* ERR = error   + (size_t)row * PP;
    double* OUT       = out     + (size_t)row * QQ * QQ;

    // Shared scratch. d2z is computed INLINE in the (logit,logit) assembly
    // (closed form from z) so the A³ tensor is never materialised — keeps
    // shared small. dz (A²) is cheap and reused, so it is stored.
    __shared__ double z[AA];          // softmax assignment ζ_k
    __shared__ double dz[AA * AA];    // ∂ζ_k/∂ℓ_j  = dz[j*AA + k]
    __shared__ double ehat[AA * NB_]; // ê_{k,b} = Σ_c dec_{k,b,c}·e_c
    __shared__ double ed[AA];         // Σ_b φ_b·ê
    __shared__ double eD1[ALSZ];      // Σ_b ∂φ_b·ê        (coord first-deriv leg)
    __shared__ double ed2[AL2SZ];     // Σ_b ∂²φ_b·ê       (coord second-deriv leg)

    // ── Phase A: ê_{k,b} = Σ_c decoder_{k,b,c}·error_c  (the ONLY p-loop).
    // One thread per (k,b); the c-sum is sequential => bit-faithful to the CPU
    // `fused_resid` summation order.
    for (int kb = t; kb < AA * NB_; kb += nt) {
        const double* drow = DEC + (size_t)kb * PP;
        double acc = 0.0;
        for (int c = 0; c < PP; ++c) acc += drow[c] * ERR[c];
        ehat[kb] = acc;
    }

    // ── Phase B (concurrent with A): softmax ζ and its Jacobian dz.
    // Thread 0 does the reduction (A is small; keeps the max/denominator order
    // identical to the CPU and avoids a cross-thread reorder).
    if (t == 0) {
        double mx = -INFINITY;
        for (int j = 0; j < AA; ++j) mx = fmax(mx, L[j]);
        double shift = mx * inv_tau;
        double denom = 0.0;
        for (int j = 0; j < AA; ++j) {
            double e = exp(L[j] * inv_tau - shift);
            z[j] = e;          // store unnormalised exp temporarily
            denom += e;
        }
        double inv_denom = 1.0 / denom;
        for (int j = 0; j < AA; ++j) z[j] *= inv_denom;
        // dz[j*AA + k] = ζ_k (δ_{kj} − ζ_j) · inv_tau
        for (int j = 0; j < AA; ++j)
            for (int k = 0; k < AA; ++k) {
                double ind = (k == j) ? 1.0 : 0.0;
                dz[j * AA + k] = z[k] * (ind - z[j]) * inv_tau;
            }
    }
    __syncthreads();

    // ── Phase C: ē channels from ê (no p-axis). One thread per atom k.
    for (int k = t; k < AA; k += nt) {
        double s = 0.0;
        for (int b = 0; b < NB_; ++b) s += PHI[k * NB_ + b] * ehat[k * NB_ + b];
        ed[k] = s;
        for (int ax = 0; ax < LL; ++ax) {
            double e = 0.0;
            for (int b = 0; b < NB_; ++b)
                e += DPH[(k * NB_ + b) * LL + ax] * ehat[k * NB_ + b];
            eD1[k * LL + ax] = e;
        }
        for (int xa = 0; xa < LL; ++xa)
            for (int xb = 0; xb < LL; ++xb) {
                double e = 0.0;
                for (int b = 0; b < NB_; ++b)
                    e += D2P[((k * NB_ + b) * LL + xa) * LL + xb] * ehat[k * NB_ + b];
                ed2[(k * LL + xa) * LL + xb] = e;
            }
    }
    __syncthreads();

    // ── Phase D: assemble R[va][vb] (q²·A, no p). One thread per (va,vb).
    const double it2 = inv_tau * inv_tau;
    for (int idx = t; idx < QQ * QQ; idx += nt) {
        const int va = idx / QQ;
        const int vb = idx % QQ;
        double val;
        if (is_logit(va) && is_logit(vb)) {
            // R = Σ_k d2z(va,vb,k)·ed[k], d2z computed inline (no A³ tensor):
            //   d2z = ζ_k[(δ_{k,vb}−ζ_vb)(δ_{k,va}−ζ_va) − ζ_va(δ_{va,vb}−ζ_vb)]·it²
            const int j = va, ll = vb;
            const double ijl = (j == ll) ? 1.0 : 0.0;
            double s = 0.0;
            for (int k = 0; k < AA; ++k) {
                double ikl = (k == ll) ? 1.0 : 0.0;
                double ikj = (k == j) ? 1.0 : 0.0;
                double d2 = z[k] * ((ikl - z[ll]) * (ikj - z[j]) - z[j] * (ijl - z[ll])) * it2;
                s += d2 * ed[k];
            }
            val = s;
        } else if (is_logit(va) && !is_logit(vb)) {
            int k = coord_atom(vb), ax = coord_axis(vb);
            val = dz[va * AA + k] * eD1[k * LL + ax];
        } else if (!is_logit(va) && is_logit(vb)) {
            int k = coord_atom(va), ax = coord_axis(va);
            val = dz[vb * AA + k] * eD1[k * LL + ax];
        } else {
            int ka = coord_atom(va), xa = coord_axis(va);
            int kb = coord_atom(vb), xb = coord_axis(vb);
            // cross-atom coord×coord = 0 (block-diagonal in the atom index).
            val = (ka == kb) ? z[ka] * ed2[(ka * LL + xa) * LL + xb] : 0.0;
        }
        OUT[idx] = val;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// FUSED GRAM kernel (the α-trace / Gauss-Newton data curvature consumer,
// construction.rs:8210).  out[row][a][b] = Σ_c first[a][c]·first[b][c].
//
// What the Gram path NEEDS beyond the residual kernel (port of `fused_gram`):
//   1. Build decoded[a][p] = Σ_b φ_b·dec  and  d1[a*l][p] = Σ_b ∂φ_b·dec
//      (out_dim-wide channel sources — these are the only p-wide buffers).
//   2. Decoded-Grams (q²·p/2, the only remaining p-loops):
//        Gdd[k][l]   = Σ_c decoded_k·decoded_l
//        GdD[k][m]   = Σ_c decoded_k·d1_m
//        GDD[m][n]   = Σ_c d1_m·d1_n
//   3. W[va][l] = Σ_k dz[va][k]·Gdd[k][l]   (A³ precompute, avoids A⁴)
//   4. Assemble H[va][vb] as a quadratic form in the softmax-derivative coeffs.
//
// SHARED-MEMORY NOTE (the reason this is a separate kernel, not validated to
// the same depth here): decoded (A·p) and d1 (A·L·p) are p-wide. For the worst
// admitted shape (A=16, L=3, p=64) d1 alone is 16·3·64·8 B = 24 KB and GDD is
// (A·L)²·8 = 18 KB — together with decoded/Gdd/GdD this exceeds the 48 KB
// default shared budget. Production must place decoded/d1 in a per-block GLOBAL
// scratch slab (the proto's resident-arena pattern, row_hessian_ops.rs) rather
// than __shared__, or tile the p-axis. The arithmetic below is the in-shared
// form for the common small-p shape; it is written for review, NOT yet wired.
// The residual kernel above is the one taken to completion.
extern "C" __global__ void __launch_bounds__(128, 1) sae_fused_gram(
        int n,
        double inv_tau,
        const double* __restrict__ logits,
        const double* __restrict__ phi,
        const double* __restrict__ dphi,
        const double* __restrict__ decoder,
        double* __restrict__ decoded_scratch,  // [n * AA * PP]      global scratch
        double* __restrict__ d1_scratch,       // [n * AA * LL * PP] global scratch
        double* __restrict__ out)              // [n * QQ * QQ]
{
    const int row = blockIdx.x;
    if (row >= n) return;
    const int t = threadIdx.x;
    const int nt = blockDim.x;

    const double* Lg  = logits  + (size_t)row * AA;
    const double* PHI = phi     + (size_t)row * AA * NB_;
    const double* DPH = dphi    + (size_t)row * AA * NB_ * LL;
    const double* DEC = decoder + (size_t)row * AA * NB_ * PP;
    double* DECODED   = decoded_scratch + (size_t)row * AA * PP;
    double* D1        = d1_scratch + (size_t)row * AA * LL * PP;
    double* OUT       = out + (size_t)row * QQ * QQ;

    __shared__ double z[AA];
    __shared__ double dz[AA * AA];
    __shared__ double gdd[AA * AA];
    __shared__ double gdD[AA * ALSZ];
    __shared__ double gDD[ALSZ * ALSZ];
    __shared__ double wll[AA * AA];

    // softmax ζ + dz (thread 0; same reduction order as CPU)
    if (t == 0) {
        double mx = -INFINITY;
        for (int j = 0; j < AA; ++j) mx = fmax(mx, Lg[j]);
        double shift = mx * inv_tau;
        double denom = 0.0;
        for (int j = 0; j < AA; ++j) { double e = exp(Lg[j] * inv_tau - shift); z[j] = e; denom += e; }
        double inv_denom = 1.0 / denom;
        for (int j = 0; j < AA; ++j) z[j] *= inv_denom;
        for (int j = 0; j < AA; ++j)
            for (int k = 0; k < AA; ++k) {
                double ind = (k == j) ? 1.0 : 0.0;
                dz[j * AA + k] = z[k] * (ind - z[j]) * inv_tau;
            }
    }

    // decoded[k][c] = Σ_b φ·dec ; d1[(k*L+ax)][c] = Σ_b ∂φ·dec   (one thread per (k,c))
    for (int kc = t; kc < AA * PP; kc += nt) {
        const int k = kc / PP, c = kc % PP;
        double dsum = 0.0;
        for (int b = 0; b < NB_; ++b) dsum += PHI[k * NB_ + b] * DEC[(k * NB_ + b) * PP + c];
        DECODED[k * PP + c] = dsum;
        for (int ax = 0; ax < LL; ++ax) {
            double s = 0.0;
            for (int b = 0; b < NB_; ++b)
                s += DPH[(k * NB_ + b) * LL + ax] * DEC[(k * NB_ + b) * PP + c];
            D1[(k * LL + ax) * PP + c] = s;
        }
    }
    __syncthreads();

    // Grams. (one thread per pair; sequential c-sum preserves CPU order)
    for (int kl = t; kl < AA * AA; kl += nt) {
        const int k = kl / AA, l = kl % AA;
        if (l < k) continue;                      // symmetric: fill upper, mirror
        double s = 0.0;
        for (int c = 0; c < PP; ++c) s += DECODED[k * PP + c] * DECODED[l * PP + c];
        gdd[k * AA + l] = s; gdd[l * AA + k] = s;
    }
    const int AL = AA * LL;
    const int ALd = (AL > 0) ? AL : 1;   // guarded divisor (AL==0 ⇒ loops empty)
    for (int km = t; km < AA * AL; km += nt) {
        const int k = km / ALd, m = km % ALd;
        double s = 0.0;
        for (int c = 0; c < PP; ++c) s += DECODED[k * PP + c] * D1[m * PP + c];
        gdD[k * AL + m] = s;
    }
    for (int mn = t; mn < AL * AL; mn += nt) {
        const int m = mn / ALd, nn = mn % ALd;
        if (nn < m) continue;
        double s = 0.0;
        for (int c = 0; c < PP; ++c) s += D1[m * PP + c] * D1[nn * PP + c];
        gDD[m * AL + nn] = s; gDD[nn * AL + m] = s;
    }
    __syncthreads();

    // W[va][l] = Σ_k dz[va][k]·Gdd[k][l]  (A³ → logit-logit H is then A²·A)
    for (int vl = t; vl < AA * AA; vl += nt) {
        const int va = vl / AA, l = vl % AA;
        double s = 0.0;
        for (int k = 0; k < AA; ++k) s += dz[va * AA + k] * gdd[k * AA + l];
        wll[va * AA + l] = s;
    }
    __syncthreads();

    // assemble H[va][vb] (no p)
    for (int idx = t; idx < QQ * QQ; idx += nt) {
        const int va = idx / QQ, vb = idx % QQ;
        double val;
        if (is_logit(va) && is_logit(vb)) {
            double s = 0.0;
            for (int l = 0; l < AA; ++l) s += wll[va * AA + l] * dz[vb * AA + l];
            val = s;
        } else if (is_logit(va) && !is_logit(vb)) {
            int kb = coord_atom(vb), bx = coord_axis(vb), m = kb * LL + bx;
            double s = 0.0;
            for (int k = 0; k < AA; ++k) s += dz[va * AA + k] * gdD[k * AL + m];
            val = z[kb] * s;
        } else if (!is_logit(va) && is_logit(vb)) {
            int ka = coord_atom(va), ax = coord_axis(va), m = ka * LL + ax;
            double s = 0.0;
            for (int k = 0; k < AA; ++k) s += dz[vb * AA + k] * gdD[k * AL + m];
            val = z[ka] * s;
        } else {
            int ka = coord_atom(va), ax = coord_axis(va);
            int kb = coord_atom(vb), bx = coord_axis(vb);
            val = z[ka] * z[kb] * gDD[(ka * LL + ax) * AL + (kb * LL + bx)];
        }
        OUT[idx] = val;
    }
}
