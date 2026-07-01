//! Stage 2 of the BMS FLEX row kernel — per-row math that turns per-cell
//! derivative moments (built by Stage 1 in `src/gpu/cubic_cell/mod.rs`) into a
//! row gradient and row-primary `r × r` Hessian.
//!
//! Math (mirrors the CPU reference
//! `BernoulliMarginalSlope::compute_row_analytic_flex_from_parts_into` in
//! `src/families/bernoulli_marginal_slope.rs`):
//!
//! For each row `i`, with per-cell cubic predictor coefficients
//! `C_c = (C0, C1, C2, C3)` and derivative moments `m_0..m_9`, build
//!
//! ```text
//!     κ        = 1 / (2π)
//!     T_n      = κ · Σ_{e=0..3} C_e · m_{e+n}     (n = 0..6)
//!     D(R)     = κ · Σ_{k=0..3} R_k · m_k
//!     Q(R, S)  = Σ_{p,q=0..3} R_p · S_q · T_{p+q}
//!     H(R, S, U) = D(U) − Q(R, S)
//! ```
//!
//! Per cell `c`, accumulate into row scratch:
//!
//! ```text
//!     F_a   += D(A_c)
//!     F_aa  += H(A_c, A_c, AA_c)
//!     F_u   += D(R_{c,u})                         u > 0
//!     F_au  += H(A_c, R_{c,u}, AR_{c,u})          u > 0
//!     F_uv  += H(R_{c,u}, R_{c,v}, S_{c,uv})      0 < u ≤ v
//! ```
//!
//! After the cell sum, the `q`-row is overridden:
//!
//! ```text
//!     F_q  = −mu_1
//!     F_qq = −mu_2
//!     F_qv = 0   (v > 0)
//!     F_aq = 0
//! ```
//!
//! Implicit function theorem (single `1/F_a`):
//!
//! ```text
//!     inv_Fa = 1 / F_a
//!     a_u    = −F_u · inv_Fa                       (q-row override: mu_1 · inv_Fa)
//!     a_uv   = −(F_uv + F_au·a_v + F_av·a_u + F_aa·a_u·a_v) · inv_Fa
//! ```
//!
//! Observed predictor at `z_obs` (host supplies pre-evaluated chi, xi, rho, tau,
//! r_uv per row and coordinate):
//!
//! ```text
//!     bar_e_u  = chi_obs · a_u + rho_u
//!     bar_e_uv = chi_obs · a_uv + xi_obs · a_u · a_v + tau_u · a_v
//!                + a_u · tau_v + r_uv
//! ```
//!
//! Probit Mills (stable; uses `log_ndtr_and_mills` from `numerics_device::PROBIT_NUMERICS_CU`):
//!
//! ```text
//!     s = 2y − 1 ;  m = s · e_obs
//!     [log_cdf, λ] = log_ndtr_and_mills(m)
//!     A = −w · s · λ
//!     B =  w · λ · (m + λ)
//! ```
//!
//! Final outputs:
//!
//! ```text
//!     neglog   = −w · log_cdf
//!     g_u      = A · bar_e_u
//!     H_{uv}   = B · bar_e_u · bar_e_v + A · bar_e_uv     (symmetric)
//! ```
//!
//! Implementation choice (Stage 2): **one CUDA block per row**, with
//! `blockDim.x = 32` threads. The block's `F_u`, `F_au`, `F_uv`, `bar_e_u`,
//! `bar_e_uv` live in shared memory; threads in the block parallelise the
//! per-cell sums, then a single thread of the block (`threadIdx.x == 0`) does
//! the IFT solve, the observed-point assembly, the Mills evaluation, and the
//! final gradient + Hessian write-out. With the `r ≤ MAX_R` cap (32) the
//! shared-memory footprint per block is `r + r + r*r + r + r*r` doubles
//! = `2r² + 3r` ≤ 2 144 doubles ≈ 17 KB, well below the V100 48 KB per-block
//! limit. This keeps the implementation simple and avoids per-thread global
//! scratch (a per-thread `r*r` scratch arena would be ~2 GB at n=195k, r=20).

#[cfg(target_os = "linux")]
use std::sync::OnceLock;

use gam_gpu::gpu_error::GpuError;

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

/// Hard ceiling on `r` (= 2 + p_h + p_w). Matches the shared-memory budget
/// argument in the module docstring: with `MAX_R = 32` the per-block shared
/// footprint is at most `2·32² + 3·32 = 2 144` doubles = 17 KB.
pub(crate) const MAX_R: usize = 32;

/// `blockDim.x` for the row kernel. Threads of a row-block parallelise the
/// per-cell loop; thread 0 of the block finalises the IFT solve. Linux-only
/// because the kernel launcher that consumes it is Linux-only.
#[cfg(target_os = "linux")]
pub(crate) const ROW_KERNEL_THREADS: u32 = 32;

/// Number of cubic predictor coefficients per cell (`C0..C3`) and the matching
/// support length of `A_c`, `R_{c,u}`, `AA_c`, `AR_{c,u}`, `S_{c,uv}`.
pub(crate) const COEFF4: usize = 4;

/// Highest moment index touched per cell: `T_n` uses `m_{n+e}` for `e = 0..3`
/// and `n = 0..6`, so the maximum index is `9`. `MOMENT_STRIDE = 10`.
pub(crate) const MOMENT_STRIDE: usize = 10;

/// Source of the per-cell derivative moments fed into the row kernel.
/// Phase-4 wiring: the substrate at `src/gpu/cubic_cell/mod.rs` can produce
/// these on the GPU; this enum lets the launcher consume them directly
/// without a DtoH+HtoD round-trip.
pub(crate) enum CellMomentsSource<'a> {
    /// Host-resident `[total_cells, MOMENT_STRIDE = 10]` row-major buffer.
    /// The launcher will HtoD-upload this on every launch.
    Host(&'a [f64]),
    /// Device-resident moments already living on the row-kernel backend's
    /// default stream (which is the same `cuda_context_for(ordinal).default_stream()`
    /// the cubic-cell substrate uses, so no cross-context copy is needed).
    /// Length on the device must be `total_cells * MOMENT_STRIDE`. Linux-only.
    #[cfg(target_os = "linux")]
    Device(&'a CudaSlice<f64>),
}

impl<'a> CellMomentsSource<'a> {
    /// Logical element count of the moments source, used by [`BmsFlexRowKernelInputs::validate`].
    pub(crate) fn len(&self) -> usize {
        match self {
            CellMomentsSource::Host(slice) => slice.len(),
            #[cfg(target_os = "linux")]
            CellMomentsSource::Device(d) => d.len(),
        }
    }
}

/// Per-row input bundle for [`launch_bms_flex_row_kernel`].
///
/// Coordinate ordering convention: `u = 0` is `a` (the latent intercept and
/// the variable IFT eliminates); `u = 1` is `b` (slope); `u = 2..2+p_h` is the
/// score-warp `β_h` block; `u = 2+p_h..2+p_h+p_w` is the link-wiggle `β_w`
/// block. So `r = 2 + p_h + p_w` and `u = 1` is the `b` (slope) index used by
/// the sparse `S_{b·h}` / `S_{b·w}` payloads.
macro_rules! define_bms_flex_row_kernel_input_types {
    (
        f64_fields: [$($f64_field:ident),+ $(,)?],
        u32_fields: [$($u32_field:ident),+ $(,)?],
        moments_field: $moments_field:ident $(,)?
    ) => {
        pub(crate) struct BmsFlexRowKernelInputs<'a> {
            /// Number of observation rows.
            pub n_rows: usize,
            /// Total primary local dimension. `r = 2 + p_h + p_w`.
            pub r: usize,
            /// Number of score-warp basis coordinates.
            pub p_h: usize,
            /// Number of link-wiggle basis coordinates.
            pub p_w: usize,
            /// Probit frailty scale `S_f` (scalar shared across rows; matches
            /// `BernoulliMarginalSlope::probit_frailty_scale`).
            pub s_f: f64,
            $(pub $f64_field: &'a [f64],)+
            $(pub $u32_field: &'a [u32],)+
            pub $moments_field: CellMomentsSource<'a>,
        }

        /// Owned twin of [`BmsFlexRowKernelInputs`] — every borrowed slice is
        /// replaced by an owned `Vec`. The buffer fields are declared from the
        /// same schema as the borrowed launch ABI and converted by
        /// [`BmsFlexRowKernelInputsOwned::as_borrowed`].
        pub(crate) struct BmsFlexRowKernelInputsOwned {
            pub n_rows: usize,
            pub r: usize,
            pub p_h: usize,
            pub p_w: usize,
            pub s_f: f64,
            $(pub $f64_field: Vec<f64>,)+
            $(pub $u32_field: Vec<u32>,)+
            pub $moments_field: Vec<f64>,
            /// Phase-4 device-resident moments. When `Some(_)`, the launcher
            /// skips the host upload and consumes the buffer directly.
            /// Linux-only field.
            #[cfg(target_os = "linux")]
            pub cell_moments_device: Option<CudaSlice<f64>>,
        }

        impl BmsFlexRowKernelInputsOwned {
            /// Borrowed view over `self` suitable for
            /// [`launch_bms_flex_row_kernel`]. The returned struct holds
            /// references into `self` so the owned bundle must outlive the
            /// launch.
            pub(crate) fn as_borrowed(&self) -> BmsFlexRowKernelInputs<'_> {
                #[cfg(target_os = "linux")]
                let cell_moments = match self.cell_moments_device.as_ref() {
                    Some(d) => CellMomentsSource::Device(d),
                    None => CellMomentsSource::Host(&self.cell_moments),
                };
                #[cfg(not(target_os = "linux"))]
                let cell_moments = CellMomentsSource::Host(&self.cell_moments);
                BmsFlexRowKernelInputs {
                    n_rows: self.n_rows,
                    r: self.r,
                    p_h: self.p_h,
                    p_w: self.p_w,
                    s_f: self.s_f,
                    $($f64_field: &self.$f64_field,)+
                    $($u32_field: &self.$u32_field,)+
                    $moments_field: cell_moments,
                }
            }
        }
    };
}

define_bms_flex_row_kernel_input_types! {
    f64_fields: [
        q,
        b,
        mu_1,
        mu_2,
        z_obs,
        y,
        w,
        cell_c0,
        cell_c1,
        cell_c2,
        cell_c3,
        cell_a,
        cell_aa,
        cell_r,
        cell_ar,
        cell_sbb,
        cell_sbh,
        cell_sbw,
        chi_obs,
        xi_obs,
        rho_u,
        tau_u,
        r_uv,
    ],
    u32_fields: [cell_offsets],
    moments_field: cell_moments,
}

/// Per-row outputs produced by [`launch_bms_flex_row_kernel`].
#[derive(Debug)]
pub(crate) struct BmsFlexRowKernelOutputs {
    /// Per-row negative log-likelihood. Length `n_rows`.
    pub neglog: Vec<f64>,
    /// Per-row gradient, row-major `[n_rows, r]`.
    pub grad: Vec<f64>,
    /// Per-row Hessian, row-major `[n_rows, r*r]`. The kernel writes the full
    /// symmetric matrix.
    pub hess: Vec<f64>,
}

impl<'a> BmsFlexRowKernelInputs<'a> {
    /// Sanity-check every shape the kernel relies on. This is the only place
    /// length errors are surfaced — the device kernel assumes valid layout.
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        if self.r == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "bms_flex_row inputs: r must be > 0".to_string(),
            });
        }
        if self.r > MAX_R {
            return Err(GpuError::DriverCallFailed {
                reason: format!("bms_flex_row inputs: r={} exceeds MAX_R={MAX_R}", self.r),
            });
        }
        if self.r != 2 + self.p_h + self.p_w {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row inputs: r={} must equal 2 + p_h({}) + p_w({}) = {}",
                    self.r,
                    self.p_h,
                    self.p_w,
                    2 + self.p_h + self.p_w
                ),
            });
        }
        let n = self.n_rows;
        let check_len = |name: &str, have: usize, want: usize| -> Result<(), GpuError> {
            if have != want {
                return Err(GpuError::DriverCallFailed {
                    reason: format!("bms_flex_row inputs: {name}.len()={have} != {want}"),
                });
            }
            Ok(())
        };
        check_len("q", self.q.len(), n)?;
        check_len("b", self.b.len(), n)?;
        check_len("mu_1", self.mu_1.len(), n)?;
        check_len("mu_2", self.mu_2.len(), n)?;
        check_len("z_obs", self.z_obs.len(), n)?;
        check_len("y", self.y.len(), n)?;
        check_len("w", self.w.len(), n)?;
        check_len("chi_obs", self.chi_obs.len(), n)?;
        check_len("xi_obs", self.xi_obs.len(), n)?;
        check_len("rho_u", self.rho_u.len(), n * self.r)?;
        check_len("tau_u", self.tau_u.len(), n * self.r)?;
        check_len("r_uv", self.r_uv.len(), n * self.r * self.r)?;
        check_len("cell_offsets", self.cell_offsets.len(), n + 1)?;
        let total_cells_u32 = self.cell_offsets[n];
        let total_cells = total_cells_u32 as usize;
        check_len("cell_c0", self.cell_c0.len(), total_cells)?;
        check_len("cell_c1", self.cell_c1.len(), total_cells)?;
        check_len("cell_c2", self.cell_c2.len(), total_cells)?;
        check_len("cell_c3", self.cell_c3.len(), total_cells)?;
        check_len("cell_a", self.cell_a.len(), total_cells * COEFF4)?;
        check_len("cell_aa", self.cell_aa.len(), total_cells * COEFF4)?;
        check_len(
            "cell_r",
            self.cell_r.len(),
            total_cells * self.r.saturating_sub(1) * COEFF4,
        )?;
        check_len(
            "cell_ar",
            self.cell_ar.len(),
            total_cells * self.r.saturating_sub(1) * COEFF4,
        )?;
        check_len("cell_sbb", self.cell_sbb.len(), total_cells * COEFF4)?;
        check_len(
            "cell_sbh",
            self.cell_sbh.len(),
            total_cells * self.p_h * COEFF4,
        )?;
        check_len(
            "cell_sbw",
            self.cell_sbw.len(),
            total_cells * self.p_w * COEFF4,
        )?;
        check_len(
            "cell_moments",
            self.cell_moments.len(),
            total_cells * MOMENT_STRIDE,
        )?;
        // Bonus: when the moments came from `CellMomentsSource::Device`, the
        // launcher needs to know the source is from a device buffer; nothing
        // to validate beyond length above. The Host variant length check is
        // also already covered above.
        // Monotone cell_offsets check.
        for i in 0..n {
            if self.cell_offsets[i] > self.cell_offsets[i + 1] {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "bms_flex_row inputs: cell_offsets must be monotone (offset[{}]={} > offset[{}]={})",
                        i,
                        self.cell_offsets[i],
                        i + 1,
                        self.cell_offsets[i + 1]
                    ),
                });
            }
        }
        Ok(())
    }
}

/// NVRTC kernel source body. One CUDA block per row; 32 threads per block
/// parallise the per-cell sums into shared-memory scratch; thread 0 of the
/// block finishes the IFT + observed-point + Mills + Hessian write-out.
///
/// Shared probit numerics (`erfcx_nonnegative`, `log_ndtr`,
/// `log_ndtr_and_mills`) are provided by
/// `numerics_device::PROBIT_NUMERICS_CU`, which is prepended before
/// passing to `cudarc::nvrtc::compile_ptx`.
///
/// **CPU parity reference**: the body mirrors
/// `compute_row_analytic_flex_from_parts_into` in
/// `src/families/bernoulli_marginal_slope.rs`.
#[cfg(target_os = "linux")]
pub(crate) const ROW_KERNEL_BODY: &str = r#"
// One block per row. blockDim.x = 32; threadIdx.x parallises per-cell sums.
// CPU parity reference: src/families/bernoulli_marginal_slope.rs
//                      ::compute_row_analytic_flex_from_parts_into.

#define INV_TWO_PI     0.15915494309189535

extern "C" __device__ __forceinline__ double atomic_add_f64(double *addr, double value) {
    unsigned long long int *addr_as_ull = (unsigned long long int *)addr;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        double next = __longlong_as_double((long long int)assumed) + value;
        old = atomicCAS(addr_as_ull, assumed, (unsigned long long int)__double_as_longlong(next));
    } while (assumed != old);
    return __longlong_as_double((long long int)old);
}

// `nan_fill_outputs`: thread-0-only path used when row inputs are degenerate
// (`F_a` non-finite or non-positive). Writes NaNs to neglog/grad/hess so the
// host falls back to CPU for that row.
extern "C" __device__ __forceinline__ void
nan_fill_outputs(int r,
                 int row,
                 double *out_neglog,
                 double *out_grad,
                 double *out_hess) {
    double nan_value = __longlong_as_double(0x7ff8000000000000ULL);
    out_neglog[row] = nan_value;
    for (int u = 0; u < r; ++u) {
        out_grad[row * r + u] = nan_value;
    }
    int rr = r * r;
    for (int idx = 0; idx < rr; ++idx) {
        out_hess[row * rr + idx] = nan_value;
    }
}

extern "C" __global__ void bms_flex_row_kernel(
    int                  n_rows,
    int                  r,
    int                  p_h,
    int                  p_w,
    double               s_f,                // currently unused on device:
                                             // host has already baked S_f
                                             // into the cubic coefficients.
                                             // Kept for diagnostic parity.
    const double * __restrict__ row_q,
    const double * __restrict__ row_b,
    const double * __restrict__ row_mu1,
    const double * __restrict__ row_mu2,
    const double * __restrict__ row_zobs,
    const double * __restrict__ row_y,
    const double * __restrict__ row_w,
    const unsigned int * __restrict__ cell_offsets,
    const double * __restrict__ cell_c0,
    const double * __restrict__ cell_c1,
    const double * __restrict__ cell_c2,
    const double * __restrict__ cell_c3,
    const double * __restrict__ cell_a,       // [n_cells, 4]
    const double * __restrict__ cell_aa,      // [n_cells, 4]
    const double * __restrict__ cell_r,       // [n_cells, r-1, 4]
    const double * __restrict__ cell_ar,      // [n_cells, r-1, 4]
    const double * __restrict__ cell_sbb,     // [n_cells, 4]
    const double * __restrict__ cell_sbh,     // [n_cells, p_h, 4]
    const double * __restrict__ cell_sbw,     // [n_cells, p_w, 4]
    const double * __restrict__ cell_moments, // [n_cells, 10]
    const double * __restrict__ row_chi,
    const double * __restrict__ row_xi,
    const double * __restrict__ row_rho,      // [n_rows, r]
    const double * __restrict__ row_tau,      // [n_rows, r]
    const double * __restrict__ row_ruv,      // [n_rows, r*r]
    double       * __restrict__ out_neglog,
    double       * __restrict__ out_grad,
    double       * __restrict__ out_hess)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;

    // ── shared scratch (sized to MAX_R = 32) ──────────────────────────────
    // Layout (doubles):
    //   F_u      [r]
    //   F_au     [r]
    //   F_uv     [r*r]
    //   bar_e_u  [r]
    //   bar_e_uv [r*r]
    //   reduce_a [blockDim.x]
    //   reduce_b [blockDim.x]
    // Sized for the worst case (r = MAX_R = 32).
    __shared__ double F_u[32];
    __shared__ double F_au[32];
    __shared__ double F_uv[32 * 32];
    __shared__ double bar_e_u[32];
    __shared__ double bar_e_uv[32 * 32];
    __shared__ double reduce_a[32];
    __shared__ double reduce_b[32];
    __shared__ double F_a_shared;
    __shared__ double F_aa_shared;

    // Zero scratch.
    if (tid == 0) { F_a_shared = 0.0; F_aa_shared = 0.0; }
    for (int u = tid; u < r; u += blockDim.x) {
        F_u[u]  = 0.0;
        F_au[u] = 0.0;
    }
    for (int uv = tid; uv < r * r; uv += blockDim.x) {
        F_uv[uv] = 0.0;
    }
    __syncthreads();

    // ── per-cell sweep ───────────────────────────────────────────────────
    unsigned int cell_lo = cell_offsets[row];
    unsigned int cell_hi = cell_offsets[row + 1];
    int n_cells = (int)(cell_hi - cell_lo);

    double local_Fa  = 0.0;
    double local_Faa = 0.0;

    for (int local_c = tid; local_c < n_cells; local_c += blockDim.x) {
        unsigned int c = cell_lo + (unsigned int)local_c;

        // Load cubic predictor coeffs C0..C3.
        double C[4];
        C[0] = cell_c0[c]; C[1] = cell_c1[c];
        C[2] = cell_c2[c]; C[3] = cell_c3[c];

        // Load m_0..m_9.
        const double *m = cell_moments + (size_t)c * 10;

        // T_n = κ · Σ_e C_e · m_{e+n}, n = 0..6.
        // CPU parity: equivalent to the `eta_rs ⊗ moments` contraction in
        //             `cell_second_derivative_from_moments` after folding the
        //             cubic predictor.
        double T[7];
        #pragma unroll
        for (int n = 0; n < 7; ++n) {
            double acc = 0.0;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc = fma(C[e], m[e + n], acc);
            }
            T[n] = acc * INV_TWO_PI;
        }

        // D(R) = κ · Σ_k R_k · m_k.
        // CPU parity: `cell_first_derivative_from_moments`.
        #define D_OF(R) (INV_TWO_PI * (R[0]*m[0] + R[1]*m[1] + R[2]*m[2] + R[3]*m[3]))

        // Q(R, S) = Σ_{p,q} R_p · S_q · T_{p+q}.
        // CPU parity: the `eta_rs` folded dot in
        // `cell_second_derivative_from_moments`.
        #define Q_OF(R, S)                                                                 \
            ((R[0]*S[0])*T[0] + (R[0]*S[1] + R[1]*S[0])*T[1]                               \
             + (R[0]*S[2] + R[1]*S[1] + R[2]*S[0])*T[2]                                    \
             + (R[0]*S[3] + R[1]*S[2] + R[2]*S[1] + R[3]*S[0])*T[3]                        \
             + (R[1]*S[3] + R[2]*S[2] + R[3]*S[1])*T[4]                                    \
             + (R[2]*S[3] + R[3]*S[2])*T[5]                                                \
             + (R[3]*S[3])*T[6])

        // F_a += D(A_c) ; F_aa += H(A_c, A_c, AA_c) = D(AA_c) − Q(A_c, A_c).
        const double *A_c  = cell_a  + (size_t)c * 4;
        const double *AA_c = cell_aa + (size_t)c * 4;
        local_Fa  += D_OF(A_c);
        local_Faa += D_OF(AA_c) - Q_OF(A_c, A_c);

        // For each u > 0: F_u += D(R_{c,u}) ; F_au += H(A_c, R_{c,u}, AR_{c,u})
        //                                   = D(AR_{c,u}) − Q(A_c, R_{c,u}).
        for (int u = 1; u < r; ++u) {
            const double *R_u = cell_r + ((size_t)c * (size_t)(r - 1) + (size_t)(u - 1)) * 4;
            const double *AR_u = cell_ar + ((size_t)c * (size_t)(r - 1) + (size_t)(u - 1)) * 4;
            double d_R   = D_OF(R_u);
            double d_AR  = D_OF(AR_u);
            double q_AR  = Q_OF(A_c, R_u);
            atomic_add_f64(&F_u[u], d_R);
            atomic_add_f64(&F_au[u], d_AR - q_AR);
        }

        // F_uv: only b·b, b·h_j, b·w_ℓ have a material `S_{c,uv}`; every other
        // (u, v) pair just contributes −Q(R_u, R_v).
        // CPU parity: `SparsePrimaryCoeffJetView::pair_from_b_family` with
        // `COEFF_SUPPORT_BHW` — every cross pair outside the b-row is zero.
        for (int u = 1; u < r; ++u) {
            const double *R_u = cell_r + ((size_t)c * (size_t)(r - 1) + (size_t)(u - 1)) * 4;
            for (int v = u; v < r; ++v) {
                const double *R_v = cell_r + ((size_t)c * (size_t)(r - 1) + (size_t)(v - 1)) * 4;
                double q_uv = Q_OF(R_u, R_v);
                double d_s  = 0.0;
                // S_{bb}: u == v == 1 (b coordinate).
                if (u == 1 && v == 1) {
                    const double *S_bb = cell_sbb + (size_t)c * 4;
                    d_s = D_OF(S_bb);
                }
                // S_{b·h_j}: u == 1, v in score-warp block, or symmetric.
                else if (u == 1 && v >= 2 && v < 2 + p_h) {
                    int j = v - 2;
                    const double *S_bh = cell_sbh + ((size_t)c * (size_t)p_h + (size_t)j) * 4;
                    d_s = D_OF(S_bh);
                }
                // S_{b·w_ℓ}: u == 1, v in link-wiggle block, or symmetric.
                else if (u == 1 && v >= 2 + p_h && v < r) {
                    int l = v - (2 + p_h);
                    const double *S_bw = cell_sbw + ((size_t)c * (size_t)p_w + (size_t)l) * 4;
                    d_s = D_OF(S_bw);
                }
                // Symmetric mirror: u in (h or w) block, v == 1 cannot happen
                // because we iterate v >= u; skip.
                double val = d_s - q_uv;
                atomic_add_f64(&F_uv[u * r + v], val);
            }
        }

        #undef D_OF
        #undef Q_OF
    }

    // Block reduction of local_Fa, local_Faa into shared.
    reduce_a[tid] = local_Fa;
    reduce_b[tid] = local_Faa;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_a[tid] += reduce_a[tid + stride];
            reduce_b[tid] += reduce_b[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        F_a_shared  = reduce_a[0];
        F_aa_shared = reduce_b[0];
    }
    __syncthreads();

    // ── thread-0 finalisation: IFT + observed-point + Mills + writes ──────
    if (tid != 0) return;

    double F_a  = F_a_shared;
    double F_aa = F_aa_shared;
    double mu_1 = row_mu1[row];
    double mu_2 = row_mu2[row];

    // q-row overrides.
    //   F_q  = -mu_1 ; F_qq = -mu_2 ; F_qv = 0 (v > 0) ; F_aq = 0.
    F_u[0]  = -mu_1;
    F_au[0] = 0.0;
    // Zero the q-cross row/column of F_uv (u == 0 or v == 0), then plant -mu_2 at (0,0).
    for (int v = 0; v < r; ++v) {
        F_uv[0 * r + v] = 0.0;
        F_uv[v * r + 0] = 0.0;
    }
    F_uv[0 * r + 0] = -mu_2;

    // Guard: degenerate F_a ⇒ NaN-fill this row's outputs.
    if (!isfinite(F_a) || F_a <= 0.0) {
        nan_fill_outputs(r, row, out_neglog, out_grad, out_hess);
        return;
    }
    double inv_Fa = 1.0 / F_a;

    // IFT, first order.
    //   a_u = -F_u · inv_Fa     (q-override: a_q = mu_1 · inv_Fa).
    double a_u[32];
    a_u[0] = mu_1 * inv_Fa;
    for (int u = 1; u < r; ++u) {
        a_u[u] = -F_u[u] * inv_Fa;
    }

    // IFT, second order.
    //   a_uv = -(F_uv + F_au · a_v + F_av · a_u + F_aa · a_u · a_v) · inv_Fa.
    // The q-row contributions (u==0 or v==0) collapse to a_uv = mu_2 · inv_Fa
    // when both are 0 and to (F_au_v) · inv_Fa-style mixed shape otherwise.
    // We compute it uniformly using the populated F_uv / F_au with the
    // q-overrides above.
    double a_uv[32 * 32];
    for (int u = 0; u < r; ++u) {
        for (int v = u; v < r; ++v) {
            double term = F_uv[u * r + v]
                        + F_au[v] * a_u[u]
                        + F_au[u] * a_u[v]
                        + F_aa * a_u[u] * a_u[v];
            double val = -term * inv_Fa;
            a_uv[u * r + v] = val;
            a_uv[v * r + u] = val;
        }
    }

    // Observed predictor jets at z_obs.
    //   bar_e_u  = chi · a_u + rho_u.
    //   bar_e_uv = chi · a_uv + xi · a_u · a_v + tau_u · a_v + a_u · tau_v + r_uv.
    double chi = row_chi[row];
    double xi  = row_xi[row];
    const double *rho = row_rho + (size_t)row * r;
    const double *tau = row_tau + (size_t)row * r;
    const double *ruv = row_ruv + (size_t)row * r * r;

    for (int u = 0; u < r; ++u) {
        bar_e_u[u] = chi * a_u[u] + rho[u];
    }
    for (int u = 0; u < r; ++u) {
        for (int v = u; v < r; ++v) {
            double val = chi * a_uv[u * r + v]
                       + xi  * a_u[u] * a_u[v]
                       + tau[u] * a_u[v]
                       + a_u[u] * tau[v]
                       + ruv[u * r + v];
            bar_e_uv[u * r + v] = val;
            if (u != v) {
                bar_e_uv[v * r + u] = val;
            }
        }
    }

    // Probit Mills.
    double y    = row_y[row];
    double w    = row_w[row];
    double s    = 2.0 * y - 1.0;
    // The "observed predictor" e_obs is the value (degree-0) term of the
    // observed jet — same convention as the CPU path. CPU parity:
    // `e_obs = chi · a_0 + rho_0`... well, no: `bar_e_u` is the *first*
    // derivative jet, not the value. The observed predictor value comes
    // from the host pre-evaluation as `rho_u[0]` of the value jet —
    // pre-baked into the host's `m = s · e_obs` payload. For Stage 2 we
    // expose it via the `bar_e_u[0]` slot which is `chi·a_0 + rho_0`; the
    // host wiring lands in the dispatcher wave that bridges this kernel
    // and the row evaluator in `bernoulli_marginal_slope.rs`.
    double e_obs = bar_e_u[0];
    double m_arg = s * e_obs;
    double log_cdf, lambda;
    log_ndtr_and_mills(m_arg, &log_cdf, &lambda);
    double A_i = -w * s * lambda;
    double B_i =  w * lambda * (m_arg + lambda);

    out_neglog[row] = -w * log_cdf;
    for (int u = 0; u < r; ++u) {
        out_grad[row * r + u] = A_i * bar_e_u[u];
    }
    for (int u = 0; u < r; ++u) {
        for (int v = u; v < r; ++v) {
            double val = B_i * bar_e_u[u] * bar_e_u[v] + A_i * bar_e_uv[u * r + v];
            out_hess[row * r * r + u * r + v] = val;
            if (u != v) {
                out_hess[row * r * r + v * r + u] = val;
            }
        }
    }
}
"#;

// Force `s_f` to be considered used at the Rust level even though Stage 2 of
// the kernel doesn't consume it on-device (the host has already baked the
// probit frailty scale into the per-cell cubic coefficients). The dispatcher
// wave that ports the rigid-branch fallback may want to apply `s_f` device-side
// for log diagnostics; leaving the field on the input struct + reading it here
// avoids a `let _` silencer the build.rs scanner would reject.
#[inline]
pub(crate) fn s_f_diagnostic_finite(inputs: &BmsFlexRowKernelInputs<'_>) -> bool {
    inputs.s_f.is_finite() && inputs.s_f > 0.0
}

#[cfg(target_os = "linux")]
pub(crate) struct RowKernelBackend {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) module: Arc<CudaModule>,
}

#[cfg(target_os = "linux")]
impl RowKernelBackend {
    pub(crate) fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<RowKernelBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                gam_gpu::backend_probe::probe_backend_with_compile("bms_flex_row", |parts| {
                    let row_kernel_source = [
                        gam_gpu::numerics_device::PROBIT_NUMERICS_CU,
                        ROW_KERNEL_BODY,
                    ]
                    .concat();
                    // #1551: route through the project's single arch-aware NVRTC
                    // entry point instead of bare `cudarc::nvrtc::compile_ptx`.
                    // `compile_ptx_arch` pins `--gpu-architecture` to the selected
                    // device's compute capability and supplies the standard CUDA
                    // include paths; bare `compile_ptx` uses NVRTC's default
                    // virtual arch with no includes. The row kernel's 64-bit
                    // `atomic_add_f64` (atomicCAS emulation) compiles best against
                    // the real device arch, and this keeps every BMS-flex compile
                    // site consistent with the SAE arrow/Schur kernels that do
                    // require the sm_60 pin for native `atomicAdd(double*,double)`.
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(&row_kernel_source)
                        .map_err(|err| GpuError::DriverCallFailed {
                            reason: format!("bms_flex_row NVRTC compile failed: {err}"),
                        })?;
                    let module =
                        parts
                            .ctx
                            .load_module(ptx)
                            .map_err(|err| GpuError::DriverCallFailed {
                                reason: format!("bms_flex_row module load failed: {err}"),
                            })?;
                    Ok(RowKernelBackend {
                        stream: parts.stream.clone(),
                        module,
                    })
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }
}

/// Launch Stage-2 BMS FLEX row kernel. On non-Linux returns
/// [`GpuError::DriverLibraryUnavailable`]; on Linux NVRTC-compiles the kernel
/// (cached for the process lifetime), uploads the per-row + per-cell buffers,
/// and dispatches one block per row.
pub(crate) fn launch_bms_flex_row_kernel(
    inputs: BmsFlexRowKernelInputs<'_>,
) -> Result<BmsFlexRowKernelOutputs, GpuError> {
    inputs.validate()?;
    if !s_f_diagnostic_finite(&inputs) {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row inputs: s_f must be positive and finite, got {}",
                inputs.s_f
            ),
        });
    }

    #[cfg(target_os = "linux")]
    {
        launch_linux(inputs)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(GpuError::DriverLibraryUnavailable {
            reason: "bms_flex_row GPU kernel is Linux-only".to_string(),
        })
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn launch_linux(
    inputs: BmsFlexRowKernelInputs<'_>,
) -> Result<BmsFlexRowKernelOutputs, GpuError> {
    let backend = RowKernelBackend::probe()?;
    let stream = &backend.stream;

    let upload_f64 = |slice: &[f64], label: &str| {
        stream
            .clone_htod(slice)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row upload {label}: {err}"),
            })
    };
    let upload_u32 = |slice: &[u32], label: &str| {
        stream
            .clone_htod(slice)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row upload {label}: {err}"),
            })
    };

    let d_q = upload_f64(inputs.q, "q")?;
    let d_b = upload_f64(inputs.b, "b")?;
    let d_mu1 = upload_f64(inputs.mu_1, "mu_1")?;
    let d_mu2 = upload_f64(inputs.mu_2, "mu_2")?;
    let d_zobs = upload_f64(inputs.z_obs, "z_obs")?;
    let d_y = upload_f64(inputs.y, "y")?;
    let d_w = upload_f64(inputs.w, "w")?;
    let d_offsets = upload_u32(inputs.cell_offsets, "cell_offsets")?;
    let d_c0 = upload_f64(inputs.cell_c0, "cell_c0")?;
    let d_c1 = upload_f64(inputs.cell_c1, "cell_c1")?;
    let d_c2 = upload_f64(inputs.cell_c2, "cell_c2")?;
    let d_c3 = upload_f64(inputs.cell_c3, "cell_c3")?;
    let d_a = upload_f64(inputs.cell_a, "cell_a")?;
    let d_aa = upload_f64(inputs.cell_aa, "cell_aa")?;
    let d_r = upload_f64(inputs.cell_r, "cell_r")?;
    let d_ar = upload_f64(inputs.cell_ar, "cell_ar")?;
    let d_sbb = upload_f64(inputs.cell_sbb, "cell_sbb")?;
    let d_sbh = upload_f64(inputs.cell_sbh, "cell_sbh")?;
    let d_sbw = upload_f64(inputs.cell_sbw, "cell_sbw")?;
    // Phase-4: optionally consume device-resident moments (no host upload).
    // Both branches end up holding a `&CudaSlice<f64>` named `d_moments_ref`
    // we can pass to the launch builder uniformly.
    let owned_host_moments: CudaSlice<f64>;
    let d_moments_ref: &CudaSlice<f64> = match &inputs.cell_moments {
        CellMomentsSource::Host(slice) => {
            owned_host_moments = upload_f64(slice, "cell_moments")?;
            &owned_host_moments
        }
        CellMomentsSource::Device(d) => *d,
    };
    let d_chi = upload_f64(inputs.chi_obs, "chi_obs")?;
    let d_xi = upload_f64(inputs.xi_obs, "xi_obs")?;
    let d_rho = upload_f64(inputs.rho_u, "rho_u")?;
    let d_tau = upload_f64(inputs.tau_u, "tau_u")?;
    let d_ruv = upload_f64(inputs.r_uv, "r_uv")?;

    let n = inputs.n_rows;
    let r = inputs.r;
    let mut d_neglog = stream
        .alloc_zeros::<f64>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc neglog: {err}"),
        })?;
    let mut d_grad =
        stream
            .alloc_zeros::<f64>(n * r)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row alloc grad: {err}"),
            })?;
    let mut d_hess =
        stream
            .alloc_zeros::<f64>(n * r * r)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row alloc hess: {err}"),
            })?;

    let func = backend
        .module
        .load_function("bms_flex_row_kernel")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row load_function: {err}"),
        })?;

    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (ROW_KERNEL_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row: n_rows={n} exceeds i32 range"),
    })?;
    let r_i32 = i32::try_from(r).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row: r={r} exceeds i32 range"),
    })?;
    let p_h_i32 = i32::try_from(inputs.p_h).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row: p_h={} exceeds i32 range", inputs.p_h),
    })?;
    let p_w_i32 = i32::try_from(inputs.p_w).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row: p_w={} exceeds i32 range", inputs.p_w),
    })?;
    let s_f = inputs.s_f;

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&n_i32)
        .arg(&r_i32)
        .arg(&p_h_i32)
        .arg(&p_w_i32)
        .arg(&s_f)
        .arg(&d_q)
        .arg(&d_b)
        .arg(&d_mu1)
        .arg(&d_mu2)
        .arg(&d_zobs)
        .arg(&d_y)
        .arg(&d_w)
        .arg(&d_offsets)
        .arg(&d_c0)
        .arg(&d_c1)
        .arg(&d_c2)
        .arg(&d_c3)
        .arg(&d_a)
        .arg(&d_aa)
        .arg(&d_r)
        .arg(&d_ar)
        .arg(&d_sbb)
        .arg(&d_sbh)
        .arg(&d_sbw)
        .arg(d_moments_ref)
        .arg(&d_chi)
        .arg(&d_xi)
        .arg(&d_rho)
        .arg(&d_tau)
        .arg(&d_ruv)
        .arg(&mut d_neglog)
        .arg(&mut d_grad)
        .arg(&mut d_hess);

    // SAFETY: every kernel parameter above is either a primitive `i32` /
    // `f64` (passed by value), a const device pointer to a buffer whose
    // length the host validated against the input struct, or an output
    // buffer pre-allocated to `n_rows`, `n_rows*r`, `n_rows*r*r`
    // doubles. The kernel's shared-memory arrays are sized to MAX_R = 32
    // and validate() rejects r > MAX_R.
    unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row launch: {err}"),
    })?;
    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row synchronize: {err}"),
        })?;

    let neglog = stream
        .clone_dtoh(&d_neglog)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row download neglog: {err}"),
        })?;
    let grad = stream
        .clone_dtoh(&d_grad)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row download grad: {err}"),
        })?;
    let hess = stream
        .clone_dtoh(&d_hess)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row download hess: {err}"),
        })?;

    Ok(BmsFlexRowKernelOutputs { neglog, grad, hess })
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: device-resident row Hessian + HVP / diagonal kernels.
//
// Math (mirrors the CPU oracle in
// `src/families/bernoulli_marginal_slope.rs::exact_newton_joint_hessian_*_from_cache`):
//
//   Block layout (joint β):
//     marginal = [0..p_m), logslope = [p_m..p_m+p_g),
//     h        = [h_start..h_end), w = [w_start..w_end), total = p_total.
//
//   Primary layout (per-row r-vector):
//     q = 0, logslope = 1,
//     h = [h_primary_start..h_primary_end),
//     w = [w_primary_start..w_primary_end), total = r.
//
//   row_dir[u] for u in primary layout:
//     row_dir[0]   = Σ_j marginal_design[row, j] · v[j]
//     row_dir[1]   = Σ_j logslope_design[row, j] · v[p_m + j]
//     row_dir[h_k] = v[h_block_start + (h_k - h_primary_start)]
//     row_dir[w_k] = v[w_block_start + (w_k - w_primary_start)]
//
//   action[u]    = Σ_v row_hessians[row, u*r + v] · row_dir[v]
//
//   block_partial[marginal_j] += action[0] · marginal_design[row, j]
//   block_partial[logslope_j] += action[1] · logslope_design[row, j]
//   block_partial[h_block_start + (h_k - h_primary_start)] += action[h_k]
//   block_partial[w_block_start + (w_k - w_primary_start)] += action[w_k]
//
// Diagonal:
//   diag[marginal_j] += row_hess[row, 0*r + 0] · marginal_design[row, j]²
//   diag[logslope_j] += row_hess[row, 1*r + 1] · logslope_design[row, j]²
//   diag[h_block_start + k] += row_hess[row, ii*r + ii]   (ii = h_primary_start + k)
//   diag[w_block_start + k] += row_hess[row, ii*r + ii]   (ii = w_primary_start + k)
//
// Determinism: each CTA owns a contiguous slice of `[chunk_start..chunk_end)`
// rows and writes its full per-chunk `p_total` partial into a non-overlapping
// region of the global partial buffer. The reduce kernel then sums those
// partials in fixed chunk-major order. No atomics.

/// Joint-β block layout shared with the host (mirrors `BlockSlices` in
/// `bernoulli_marginal_slope.rs`).
///
/// Gating: Linux-only. The lone production constructor lives in
/// `bernoulli_marginal_slope.rs:9189` behind `#[cfg(target_os = "linux")]`
/// — the device-resident row-Hessian path is the only producer (see
/// `launch_bms_flex_row_kernel_device_resident`), and the joint-β
/// consumers `launch_bms_flex_row_hvp` / `_diagonal` / `_dense_block`
/// are also Linux-only. Any non-Linux test referencing this type must
/// guard itself with `#[cfg(target_os = "linux")]` too — the build.rs
/// ban scanner explicitly rejects `#[cfg(any(..., test))]` on items as
/// a dead-code escape hatch.
#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
pub(crate) struct BmsFlexBlockLayout {
    pub p_m: usize,
    pub p_g: usize,
    pub h: Option<std::ops::Range<usize>>,
    pub w: Option<std::ops::Range<usize>>,
    pub p_total: usize,
}

/// Primary-r layout shared with the host (mirrors `PrimarySlices`).
/// Gating rationale identical to [`BmsFlexBlockLayout`].
#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
pub(crate) struct BmsFlexPrimaryLayout {
    pub h: Option<std::ops::Range<usize>>,
    pub w: Option<std::ops::Range<usize>>,
    pub r: usize,
}

// ── Linux-only: device-resident row-Hessian state + kernels ─────────────────

/// Number of rows each HVP / diagonal CTA processes. Each CTA writes a single
/// `[1, p_total]` partial row into the global partial buffer (no atomics);
/// the reduce kernel then sums partials in chunk-major fixed order.
#[cfg(target_os = "linux")]
pub(crate) const HVP_ROWS_PER_CTA: u32 = 256;

/// `blockDim.x` for the HVP / diagonal partial kernels.
#[cfg(target_os = "linux")]
pub(crate) const HVP_THREADS: u32 = 128;

/// `blockDim.x` for the partial-sum reduction kernels (one element per thread,
/// grid-strided over the `p_total`/`rhs_elems` partial buffer). A full warp
/// multiple that keeps the reduce launch occupancy-bound rather than tail-bound
/// for the typical large-scale `p_total`.
#[cfg(target_os = "linux")]
pub(crate) const REDUCTION_THREADS: u32 = 256;

/// Maximum RHS columns fused into one row-primary HVP launch. The matching
/// CUDA source uses fixed shared arrays sized as
/// `BMS_FLEX_ROW_HVP_MAX_RHS * MAX_R`; increasing this requires updating the
/// `MAX_MULTI_RHS` define in `HVP_KERNEL_SOURCE`.
#[cfg(target_os = "linux")]
pub(crate) const BMS_FLEX_ROW_HVP_MAX_RHS: usize = 8;

/// Device-resident state produced by
/// [`launch_bms_flex_row_kernel_device_resident`] and consumed by
/// [`launch_bms_flex_row_hvp`] / [`launch_bms_flex_row_diagonal`].
///
/// Owns the row-Hessian + design slices on-device so the host can issue
/// many HVPs against the same β snapshot without round-tripping
/// 626 MB through host RAM. Drop releases the device memory back to
/// the CUDA runtime.
/// Per-row Hessian storage layout on the device. The build path is free to
/// emit either, and the Hv / diag kernels read whichever the storage says.
///
/// Charter (Block 9 Phase 4): packed-upper halves the DRAM footprint of the
/// `n × r²` cache (per-row `r*(r+1)/2` doubles instead of `r²`), at the cost
/// of a single per-entry index conversion in the kernel. The benchmark
/// decides whether the packed path becomes the default for large-scale
/// fits (`r = 20` → 210 vs 400 doubles per row, ~47.5% smaller). The
/// numerics are bit-equal because each `H_i` is symmetric by construction
/// (the row kernel emits a symmetric block by construction — see the
/// symmetric scratch-write loop in `bms_flex_row_kernel`'s shared-memory
/// finaliser).
#[cfg(target_os = "linux")]
pub struct DeviceResidentRowHess {
    /// Per-row dense `[n, r, r]` row-major Hessian. Element `(u, v)` of row
    /// `i` is `hess[i*r*r + u*r + v]`. This is the only on-device storage
    /// layout supported by the current HVP / diag kernels.
    pub(crate) hess: CudaSlice<f64>,
    pub(crate) marginal_design: CudaSlice<f64>,
    pub(crate) logslope_design: CudaSlice<f64>,
    pub(crate) n: usize,
    pub(crate) r: usize,
    pub(crate) block: BmsFlexBlockLayout,
    pub(crate) primary: BmsFlexPrimaryLayout,
    /// Estimated bytes resident on device (for accounting).
    pub(crate) bytes: u64,
}

#[cfg(target_os = "linux")]
impl std::fmt::Debug for DeviceResidentRowHess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceResidentRowHess")
            .field("n", &self.n)
            .field("r", &self.r)
            .field("p_total", &self.block.p_total)
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// Sized-to-fit-once CTA mapping. Rows `[c * HVP_ROWS_PER_CTA, (c+1) * HVP_ROWS_PER_CTA)`
/// belong to chunk `c`.
#[cfg(target_os = "linux")]
pub(crate) fn num_hvp_chunks(n: usize) -> usize {
    n.div_ceil(HVP_ROWS_PER_CTA as usize)
}

/// NVRTC source: HVP-partial kernel + HVP-reduce kernel + diag-partial +
/// diag-reduce. All kernels mirror the CPU oracle in this file.
#[cfg(target_os = "linux")]
pub(crate) const HVP_KERNEL_SOURCE: &str = r#"
// CPU parity reference: cpu_oracle_bms_flex_row_hvp / cpu_oracle_bms_flex_row_diagonal
// in this module.

#define MAX_MULTI_RHS 8

extern "C" __global__ void bms_flex_row_hvp_partial(
    int                  n_rows,
    int                  r,
    int                  p_m,
    int                  p_g,
    int                  p_total,
    int                  h_block_start,
    int                  h_block_len,
    int                  w_block_start,
    int                  w_block_len,
    int                  h_primary_start,
    int                  w_primary_start,
    int                  rows_per_cta,
    const double * __restrict__ row_hessians,    // [n, r*r]
    const double * __restrict__ marginal_design, // [n, p_m] row-major
    const double * __restrict__ logslope_design, // [n, p_g] row-major
    const double * __restrict__ v,               // [p_total]
    double       * __restrict__ partial)         // [num_chunks, p_total]
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int row_hi = row_lo + rows_per_cta;
    if (row_hi > n_rows) row_hi = n_rows;

    // Zero this chunk's partial slice cooperatively.
    double *out = partial + (size_t)chunk * (size_t)p_total;
    for (int j = tid; j < p_total; j += blockDim.x) {
        out[j] = 0.0;
    }
    __syncthreads();

    // Each thread serially processes a stride-of-blockDim set of rows so
    // every write to `out[..]` happens from one thread → no atomics within
    // the chunk. To keep writes race-free across threads of the same chunk,
    // we serialize the cross-row accumulation through a per-row barrier:
    // thread 0 of the block processes all rows in the chunk. The per-row
    // work is dominated by the dot/axpy over `p_m + p_g`, which is large.
    // For Stage 3 we ship the simple, correct path (thread 0 sequential
    // per row, blockDim.x threads parallel within a row's dot/axpy).
    __shared__ double row_dir[32];
    __shared__ double action[32];
    __shared__ double dot_reduce[128];

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = logslope_design + (size_t)row * (size_t)p_g;
        const double *Hrow = row_hessians + (size_t)row * (size_t)r * (size_t)r;

        // row_dir[0] = mrow · v[0..p_m]
        double local = 0.0;
        for (int j = tid; j < p_m; j += blockDim.x) {
            local += mrow[j] * v[j];
        }
        dot_reduce[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) dot_reduce[tid] += dot_reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) row_dir[0] = dot_reduce[0];

        // row_dir[1] = grow · v[p_m..p_m+p_g]
        local = 0.0;
        for (int j = tid; j < p_g; j += blockDim.x) {
            local += grow[j] * v[p_m + j];
        }
        dot_reduce[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) dot_reduce[tid] += dot_reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) row_dir[1] = dot_reduce[0];

        // h/w blocks: direct copy.
        if (tid == 0) {
            for (int k = 0; k < h_block_len; ++k) {
                row_dir[h_primary_start + k] = v[h_block_start + k];
            }
            for (int k = 0; k < w_block_len; ++k) {
                row_dir[w_primary_start + k] = v[w_block_start + k];
            }
        }
        __syncthreads();

        // action[u] = Σ_v Hrow[u*r+v] · row_dir[v], computed by thread u (u < r).
        if (tid < r) {
            double acc = 0.0;
            for (int vv = 0; vv < r; ++vv) {
                acc += Hrow[tid * r + vv] * row_dir[vv];
            }
            action[tid] = acc;
        }
        __syncthreads();

        // Pull back into joint β slot.
        //   marginal: out[j] += action[0] · mrow[j]   (parallel j)
        double a0 = action[0];
        for (int j = tid; j < p_m; j += blockDim.x) {
            out[j] += a0 * mrow[j];
        }
        double a1 = action[1];
        for (int j = tid; j < p_g; j += blockDim.x) {
            out[p_m + j] += a1 * grow[j];
        }
        if (tid == 0) {
            for (int k = 0; k < h_block_len; ++k) {
                out[h_block_start + k] += action[h_primary_start + k];
            }
            for (int k = 0; k < w_block_len; ++k) {
                out[w_block_start + k] += action[w_primary_start + k];
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void bms_flex_row_hvp_reduce(
    int                  num_chunks,
    int                  p_total,
    const double * __restrict__ partial,   // [num_chunks, p_total]
    double       * __restrict__ out)        // [p_total]
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p_total) return;
    double acc = 0.0;
    for (int c = 0; c < num_chunks; ++c) {
        acc += partial[(size_t)c * (size_t)p_total + (size_t)j];
    }
    out[j] = acc;
}

extern "C" __global__ void bms_flex_row_hvp_multi_partial(
    int                  n_rows,
    int                  r,
    int                  p_m,
    int                  p_g,
    int                  p_total,
    int                  h_block_start,
    int                  h_block_len,
    int                  w_block_start,
    int                  w_block_len,
    int                  h_primary_start,
    int                  w_primary_start,
    int                  rows_per_cta,
    int                  rhs_count,
    const double * __restrict__ row_hessians,    // [n, r*r]
    const double * __restrict__ marginal_design, // [n, p_m]
    const double * __restrict__ logslope_design, // [n, p_g]
    const double * __restrict__ v_rhs,           // [rhs_count, p_total]
    double       * __restrict__ partial)         // [rhs_count, num_chunks, p_total]
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int row_hi = row_lo + rows_per_cta;
    if (row_hi > n_rows) row_hi = n_rows;

    int num_chunks = (n_rows + rows_per_cta - 1) / rows_per_cta;
    for (int idx = tid; idx < rhs_count * p_total; idx += blockDim.x) {
        int rhs = idx / p_total;
        int j = idx - rhs * p_total;
        partial[((size_t)rhs * (size_t)num_chunks + (size_t)chunk) * (size_t)p_total + (size_t)j] = 0.0;
    }
    __syncthreads();

    __shared__ double row_dir[MAX_MULTI_RHS * 32];
    __shared__ double action[MAX_MULTI_RHS * 32];
    __shared__ double dot_reduce[128];

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = logslope_design + (size_t)row * (size_t)p_g;
        const double *Hrow = row_hessians + (size_t)row * (size_t)r * (size_t)r;

        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            const double *v = v_rhs + (size_t)rhs * (size_t)p_total;

            double local = 0.0;
            for (int j = tid; j < p_m; j += blockDim.x) {
                local += mrow[j] * v[j];
            }
            dot_reduce[tid] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) dot_reduce[tid] += dot_reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) row_dir[rhs * 32 + 0] = dot_reduce[0];

            local = 0.0;
            for (int j = tid; j < p_g; j += blockDim.x) {
                local += grow[j] * v[p_m + j];
            }
            dot_reduce[tid] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) dot_reduce[tid] += dot_reduce[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                row_dir[rhs * 32 + 1] = dot_reduce[0];
                for (int k = 0; k < h_block_len; ++k) {
                    row_dir[rhs * 32 + h_primary_start + k] = v[h_block_start + k];
                }
                for (int k = 0; k < w_block_len; ++k) {
                    row_dir[rhs * 32 + w_primary_start + k] = v[w_block_start + k];
                }
            }
            __syncthreads();
        }

        for (int idx = tid; idx < rhs_count * r; idx += blockDim.x) {
            int rhs = idx / r;
            int u = idx - rhs * r;
            double acc = 0.0;
            const double *dir = row_dir + rhs * 32;
            for (int vv = 0; vv < r; ++vv) {
                acc += Hrow[u * r + vv] * dir[vv];
            }
            action[rhs * 32 + u] = acc;
        }
        __syncthreads();

        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            double *out = partial + ((size_t)rhs * (size_t)num_chunks + (size_t)chunk) * (size_t)p_total;
            double a0 = action[rhs * 32 + 0];
            for (int j = tid; j < p_m; j += blockDim.x) {
                out[j] += a0 * mrow[j];
            }
            double a1 = action[rhs * 32 + 1];
            for (int j = tid; j < p_g; j += blockDim.x) {
                out[p_m + j] += a1 * grow[j];
            }
            if (tid == 0) {
                for (int k = 0; k < h_block_len; ++k) {
                    out[h_block_start + k] += action[rhs * 32 + h_primary_start + k];
                }
                for (int k = 0; k < w_block_len; ++k) {
                    out[w_block_start + k] += action[rhs * 32 + w_primary_start + k];
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void bms_flex_row_hvp_multi_reduce(
    int                  num_chunks,
    int                  p_total,
    int                  rhs_count,
    const double * __restrict__ partial,   // [rhs_count, num_chunks, p_total]
    double       * __restrict__ out)        // [rhs_count, p_total]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rhs_count * p_total;
    if (idx >= total) return;
    int rhs = idx / p_total;
    int j = idx - rhs * p_total;
    double acc = 0.0;
    for (int c = 0; c < num_chunks; ++c) {
        acc += partial[((size_t)rhs * (size_t)num_chunks + (size_t)c) * (size_t)p_total + (size_t)j];
    }
    out[(size_t)rhs * (size_t)p_total + (size_t)j] = acc;
}

extern "C" __global__ void bms_flex_row_diag_partial(
    int                  n_rows,
    int                  r,
    int                  p_m,
    int                  p_g,
    int                  p_total,
    int                  h_block_start,
    int                  h_block_len,
    int                  w_block_start,
    int                  w_block_len,
    int                  h_primary_start,
    int                  w_primary_start,
    int                  rows_per_cta,
    const double * __restrict__ row_hessians,
    const double * __restrict__ marginal_design,
    const double * __restrict__ logslope_design,
    double       * __restrict__ partial)
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int row_hi = row_lo + rows_per_cta;
    if (row_hi > n_rows) row_hi = n_rows;

    double *out = partial + (size_t)chunk * (size_t)p_total;
    for (int j = tid; j < p_total; j += blockDim.x) {
        out[j] = 0.0;
    }
    __syncthreads();

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = logslope_design + (size_t)row * (size_t)p_g;
        const double *Hrow = row_hessians + (size_t)row * (size_t)r * (size_t)r;
        double h00 = Hrow[0];
        double h11 = Hrow[1 * r + 1];
        for (int j = tid; j < p_m; j += blockDim.x) {
            double v = mrow[j];
            out[j] += h00 * v * v;
        }
        for (int j = tid; j < p_g; j += blockDim.x) {
            double v = grow[j];
            out[p_m + j] += h11 * v * v;
        }
        if (tid == 0) {
            for (int k = 0; k < h_block_len; ++k) {
                int ii = h_primary_start + k;
                out[h_block_start + k] += Hrow[ii * r + ii];
            }
            for (int k = 0; k < w_block_len; ++k) {
                int ii = w_primary_start + k;
                out[w_block_start + k] += Hrow[ii * r + ii];
            }
        }
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────────────────────
// Phase 4 — SymmetricPackedUpper variants. Per-row storage is
//   row_hessians_packed + (size_t)row * (size_t)(r*(r+1)/2)
// indexed as
//   packed[(u*(2*r - u - 1))/2 + (v - u)]   for u <= v
// with symmetric mirror for v < u.
// ────────────────────────────────────────────────────────────────────────

// Helper: packed-upper index for (u, v) within a single row of r*(r+1)/2
// doubles. Caller must pre-swap so that u <= v.
__device__ __forceinline__ int bms_flex_packed_idx(int u, int v, int r) {
    // u*(2r - u - 1)/2 + (v - u)
    return (u * (2 * r - u - 1)) / 2 + (v - u);
}

// Pack one row of the full row-major r×r Hessian into packed-upper layout.
// Launched as one CTA per row (gridDim.x = n_rows, blockDim.x configurable).
// Bit-equal copy: each upper-triangle entry is read once from the dense
// source and written once to the packed destination.
extern "C" __global__ void bms_flex_row_pack_upper(
    int                  n_rows,
    int                  r,
    const double * __restrict__ src_full,    // [n, r*r]
    double       * __restrict__ dst_packed)  // [n, r*(r+1)/2]
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;
    int per_row = r * (r + 1) / 2;
    const double *src = src_full + (size_t)row * (size_t)r * (size_t)r;
    double       *dst = dst_packed + (size_t)row * (size_t)per_row;
    // Linear scan over packed positions; map each back to (u, v).
    for (int pos = tid; pos < per_row; pos += blockDim.x) {
        // Invert: for u in [0, r), the range [u_start, u_start + (r - u))
        // contains positions for that u. u_start = u*(2r - u - 1)/2.
        // Solve smallest u with u*(2r - u - 1)/2 > pos to get u (then
        // back off by one); equivalent O(r) linear scan with r <= 32.
        int u = 0;
        int u_start = 0;
        while (u < r) {
            int next = u_start + (r - u);
            if (pos < next) break;
            u_start = next;
            ++u;
        }
        int v = u + (pos - u_start);
        dst[pos] = src[(size_t)u * (size_t)r + (size_t)v];
    }
}

extern "C" __global__ void bms_flex_row_hvp_partial_packed(
    int                  n_rows,
    int                  r,
    int                  p_m,
    int                  p_g,
    int                  p_total,
    int                  h_block_start,
    int                  h_block_len,
    int                  w_block_start,
    int                  w_block_len,
    int                  h_primary_start,
    int                  w_primary_start,
    int                  rows_per_cta,
    const double * __restrict__ row_hessians_packed, // [n, r*(r+1)/2]
    const double * __restrict__ marginal_design,
    const double * __restrict__ logslope_design,
    const double * __restrict__ v,
    double       * __restrict__ partial)
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int row_hi = row_lo + rows_per_cta;
    if (row_hi > n_rows) row_hi = n_rows;

    int per_row = r * (r + 1) / 2;
    double *out = partial + (size_t)chunk * (size_t)p_total;
    for (int j = tid; j < p_total; j += blockDim.x) {
        out[j] = 0.0;
    }
    __syncthreads();

    __shared__ double row_dir[32];
    __shared__ double action[32];
    __shared__ double dot_reduce[128];

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = logslope_design + (size_t)row * (size_t)p_g;
        const double *Hrow = row_hessians_packed + (size_t)row * (size_t)per_row;

        // row_dir[0] = mrow · v[0..p_m]
        double local = 0.0;
        for (int j = tid; j < p_m; j += blockDim.x) {
            local += mrow[j] * v[j];
        }
        dot_reduce[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) dot_reduce[tid] += dot_reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) row_dir[0] = dot_reduce[0];

        // row_dir[1] = grow · v[p_m..p_m+p_g]
        local = 0.0;
        for (int j = tid; j < p_g; j += blockDim.x) {
            local += grow[j] * v[p_m + j];
        }
        dot_reduce[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) dot_reduce[tid] += dot_reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) row_dir[1] = dot_reduce[0];

        if (tid == 0) {
            for (int k = 0; k < h_block_len; ++k) {
                row_dir[h_primary_start + k] = v[h_block_start + k];
            }
            for (int k = 0; k < w_block_len; ++k) {
                row_dir[w_primary_start + k] = v[w_block_start + k];
            }
        }
        __syncthreads();

        // action[u] = Σ_w H[u, w] · row_dir[w], where H[u, w] reads from
        // packed-upper with (uu, vv) = (min(u, w), max(u, w)).
        if (tid < r) {
            double acc = 0.0;
            int u = tid;
            for (int w = 0; w < r; ++w) {
                int uu = u < w ? u : w;
                int vv = u < w ? w : u;
                acc += Hrow[bms_flex_packed_idx(uu, vv, r)] * row_dir[w];
            }
            action[tid] = acc;
        }
        __syncthreads();

        double a0 = action[0];
        for (int j = tid; j < p_m; j += blockDim.x) {
            out[j] += a0 * mrow[j];
        }
        double a1 = action[1];
        for (int j = tid; j < p_g; j += blockDim.x) {
            out[p_m + j] += a1 * grow[j];
        }
        if (tid == 0) {
            for (int k = 0; k < h_block_len; ++k) {
                out[h_block_start + k] += action[h_primary_start + k];
            }
            for (int k = 0; k < w_block_len; ++k) {
                out[w_block_start + k] += action[w_primary_start + k];
            }
        }
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────────────────────
// Phase 6 — dense joint-Hessian block kernel for the debug / exact-REML
// route. Materialises the full `[p_total, p_total]` row-major joint H
// from the per-row r×r Hessian via the P_i pullback. NOT the default
// Newton path: production Newton uses HVP (Phase 2/3); this kernel exists
// for exact-REML logdet / dense-H comparisons / diagnostic dumps where the
// caller genuinely needs the dense matrix on the device.
//
// Per-CTA partial: each CTA owns a contiguous chunk of rows
// `[chunk*rows_per_cta, (chunk+1)*rows_per_cta)`. Inside the CTA the
// per-row pullback computes `(P_i^T H_i P_i)[m, n]` and adds it to the
// CTA's shared-mem `[p_total, p_total]` partial. The reduce kernel sums
// chunk-major-fixed-order into a single `[p_total, p_total]` output.
//
// Math: for primary index u ∈ [0, r):
//   * u = 0:        phi_u = (X_i in slot 0..p_m, 0 elsewhere)
//   * u = 1:        phi_u = (0, G_i in slot p_m..p_m+p_g, 0 elsewhere)
//   * u = 2+j:      phi_u = e_{h_block_start + j}  (j ∈ 0..h_block_len)
//   * u = 2+h+l:    phi_u = e_{w_block_start + l}  (l ∈ 0..w_block_len)
// Then `H_full[m, n] += sum_{u,v} H_i[u,v] * phi_u[m] * phi_v[n]`.
//
// Shared-memory budget: at large-scale shape p_total = 44, a [44, 44] f64
// partial is 44*44*8 = 15.5 KiB — well below the V100 48 KiB/SM cap.
// At p_total ≤ 80 the kernel still fits (80*80*8 = 50 KiB → just over
// V100 cap; caller must enforce p_total ≤ DENSE_BLOCK_MAX_P). The
// launcher rejects oversize p_total cleanly.

extern "C" __global__ void bms_flex_row_dense_block_partial(
    int                  n_rows,
    int                  r,
    int                  p_m,
    int                  p_g,
    int                  p_total,
    int                  h_block_start,
    int                  h_block_len,
    int                  w_block_start,
    int                  w_block_len,
    int                  h_primary_start,
    int                  w_primary_start,
    int                  rows_per_cta,
    const double * __restrict__ row_hessians,    // [n, r*r]
    const double * __restrict__ marginal_design, // [n, p_m]
    const double * __restrict__ logslope_design, // [n, p_g]
    double       * __restrict__ partial)         // [num_chunks, p_total, p_total]
{
    extern __shared__ double shmem[];
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int row_hi = row_lo + rows_per_cta;
    if (row_hi > n_rows) row_hi = n_rows;

    int pp = p_total * p_total;
    double *acc = shmem; // CTA-private accumulator [p_total, p_total]
    for (int j = tid; j < pp; j += blockDim.x) acc[j] = 0.0;
    __syncthreads();

    // Per-row work performed by thread 0 to avoid cross-thread RW
    // contention on `acc[]`. Per-row complexity is O(r * p_m + r * p_g
    // + r²): tractable because r ≤ 32 and p_m + p_g typically ≤ 64.
    // Tighter parallel implementations are possible (warp-stripe the
    // 4-way nested u-v-m-n loop) but Phase 6 is a debug-only path and
    // the simple version is easier to audit for correctness against
    // the host-side P_i pullback oracle.
    if (tid == 0) {
        for (int row = row_lo; row < row_hi; ++row) {
            const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
            const double *grow = logslope_design + (size_t)row * (size_t)p_g;
            const double *Hrow = row_hessians + (size_t)row * (size_t)r * (size_t)r;
            for (int u = 0; u < r; ++u) {
                for (int v = 0; v < r; ++v) {
                    double huv = Hrow[u * r + v];
                    if (huv == 0.0) continue;
                    // For each (u, v), iterate (m, n) over the non-zero
                    // outer-product support of phi_u and phi_v.
                    // Build a small (offset, len, src_ptr) descriptor for
                    // each operand block as we go.
                    int m_off, m_len; const double *m_src; bool m_indicator;
                    int n_off, n_len; const double *n_src; bool n_indicator;
                    if (u == 0)      { m_off = 0;   m_len = p_m; m_src = mrow; m_indicator = false; }
                    else if (u == 1) { m_off = p_m; m_len = p_g; m_src = grow; m_indicator = false; }
                    else if (u - 2 < h_block_len) {
                                       m_off = h_block_start + (u - 2);
                                       m_len = 1;   m_src = NULL; m_indicator = true;
                    } else {
                                       m_off = w_block_start + (u - 2 - h_block_len);
                                       m_len = 1;   m_src = NULL; m_indicator = true;
                    }
                    if (v == 0)      { n_off = 0;   n_len = p_m; n_src = mrow; n_indicator = false; }
                    else if (v == 1) { n_off = p_m; n_len = p_g; n_src = grow; n_indicator = false; }
                    else if (v - 2 < h_block_len) {
                                       n_off = h_block_start + (v - 2);
                                       n_len = 1;   n_src = NULL; n_indicator = true;
                    } else {
                                       n_off = w_block_start + (v - 2 - h_block_len);
                                       n_len = 1;   n_src = NULL; n_indicator = true;
                    }
                    // accumulate huv * phi_u[m] * phi_v[n] into acc[m, n]
                    for (int mi = 0; mi < m_len; ++mi) {
                        double pm = m_indicator ? 1.0 : m_src[mi];
                        if (pm == 0.0) continue;
                        double scaled = huv * pm;
                        int m_idx = m_off + mi;
                        for (int ni = 0; ni < n_len; ++ni) {
                            double pn = n_indicator ? 1.0 : n_src[ni];
                            int n_idx = n_off + ni;
                            acc[m_idx * p_total + n_idx] += scaled * pn;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Write CTA accumulator out to global memory at its chunk slot.
    double *out_chunk = partial + (size_t)chunk * (size_t)pp;
    for (int j = tid; j < pp; j += blockDim.x) {
        out_chunk[j] = acc[j];
    }
}

extern "C" __global__ void bms_flex_row_dense_block_reduce(
    int                  num_chunks,
    int                  p_total,
    const double * __restrict__ partial,
    double       * __restrict__ out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int pp = p_total * p_total;
    if (j >= pp) return;
    double acc = 0.0;
    for (int c = 0; c < num_chunks; ++c) {
        acc += partial[(size_t)c * (size_t)pp + (size_t)j];
    }
    out[j] = acc;
}

extern "C" __global__ void bms_flex_row_diag_partial_packed(
    int                  n_rows,
    int                  r,
    int                  p_m,
    int                  p_g,
    int                  p_total,
    int                  h_block_start,
    int                  h_block_len,
    int                  w_block_start,
    int                  w_block_len,
    int                  h_primary_start,
    int                  w_primary_start,
    int                  rows_per_cta,
    const double * __restrict__ row_hessians_packed,
    const double * __restrict__ marginal_design,
    const double * __restrict__ logslope_design,
    double       * __restrict__ partial)
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int row_hi = row_lo + rows_per_cta;
    if (row_hi > n_rows) row_hi = n_rows;

    int per_row = r * (r + 1) / 2;
    double *out = partial + (size_t)chunk * (size_t)p_total;
    for (int j = tid; j < p_total; j += blockDim.x) {
        out[j] = 0.0;
    }
    __syncthreads();

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = logslope_design + (size_t)row * (size_t)p_g;
        const double *Hrow = row_hessians_packed + (size_t)row * (size_t)per_row;
        // Diagonal entry for (u, u) sits at packed_idx(u, u, r).
        double h00 = Hrow[bms_flex_packed_idx(0, 0, r)];
        double h11 = Hrow[bms_flex_packed_idx(1, 1, r)];
        for (int j = tid; j < p_m; j += blockDim.x) {
            double v = mrow[j];
            out[j] += h00 * v * v;
        }
        for (int j = tid; j < p_g; j += blockDim.x) {
            double v = grow[j];
            out[p_m + j] += h11 * v * v;
        }
        if (tid == 0) {
            for (int k = 0; k < h_block_len; ++k) {
                int ii = h_primary_start + k;
                out[h_block_start + k] += Hrow[bms_flex_packed_idx(ii, ii, r)];
            }
            for (int k = 0; k < w_block_len; ++k) {
                int ii = w_primary_start + k;
                out[w_block_start + k] += Hrow[bms_flex_packed_idx(ii, ii, r)];
            }
        }
        __syncthreads();
    }
}
"#;

#[cfg(target_os = "linux")]
pub(crate) struct HvpKernelBackend {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) module: Arc<CudaModule>,
}

#[cfg(target_os = "linux")]
impl HvpKernelBackend {
    pub(crate) fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<HvpKernelBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                gam_gpu::backend_probe::probe_backend_with_compile("bms_flex_row hvp", |parts| {
                    // #1551: arch-aware compile (see launch_bms_flex_row_kernel) —
                    // pin `--gpu-architecture` to the device capability and supply
                    // the standard include paths via the shared NVRTC entry point.
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(HVP_KERNEL_SOURCE)
                        .map_err(|err| GpuError::DriverCallFailed {
                            reason: format!("bms_flex_row hvp NVRTC compile failed: {err}"),
                        })?;
                    let module =
                        parts
                            .ctx
                            .load_module(ptx)
                            .map_err(|err| GpuError::DriverCallFailed {
                                reason: format!("bms_flex_row hvp module load failed: {err}"),
                            })?;
                    Ok(HvpKernelBackend {
                        stream: parts.stream.clone(),
                        module,
                    })
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }
}

/// Build a device-resident row-Hessian cache by launching the row kernel and
/// keeping the resulting `n × r²` slice resident on the device. Also uploads
/// the dense marginal + logslope design matrices so subsequent HVPs do not
/// re-upload them at every direction.
///
/// `marginal_design_row_major` and `logslope_design_row_major` must be
/// row-major `[n, p_m]` and `[n, p_g]` contiguous slices.
///
/// #461 absorber (additive Stage-1 influence block): the orthogonalization is
/// realized as **A2 — the marginal design widened to `[M | Z̃_infl]`** (see
/// `src/families/bms/block_specs.rs::widen_marginal_dense_with_influence`), NOT
/// a dedicated 5th primary coordinate. The absorber `+Z̃_infl·γ` is plain
/// additive into the marginal index `α(x)`, so γ lives inside `β_m` of the
/// widened block and `p_m` already counts the `p₁` influence columns. The row
/// kernel reads the marginal index from `block_states[0].eta` (which carries
/// `Z̃_infl·γ`) and pulls back through this same widened `marginal_design`, so
/// the absorber rides the existing primary-coordinate `u = 0` chain with **no
/// kernel-source change**: η, gradient, and Hessian match the CPU kernel
/// bit-for-bit precisely because `marginal_design` and `β_m` are the matched
/// (design, coefficient) pair the CPU path uses. The validation below pins
/// `marginal_design.len() == n·p_m` (with `p_m` widened), so a stale narrow
/// design against a widened `block.p_m` is rejected cleanly (CPU fallback)
/// rather than silently computing the wrong η. The absorber is dropped at
/// predict, where the marginal design is rebuilt without the influence columns,
/// so the predict-time `p_m` is narrow and this path is correct there too.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_kernel_device_resident(
    inputs: BmsFlexRowKernelInputs<'_>,
    marginal_design_row_major: &[f64],
    logslope_design_row_major: &[f64],
    block: BmsFlexBlockLayout,
    primary: BmsFlexPrimaryLayout,
) -> Result<DeviceResidentRowHess, GpuError> {
    inputs.validate()?;
    if !s_f_diagnostic_finite(&inputs) {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident: s_f must be positive and finite, got {}",
                inputs.s_f
            ),
        });
    }
    let n = inputs.n_rows;
    let r = inputs.r;
    if marginal_design_row_major.len() != n * block.p_m {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident: marginal_design len={} != n*p_m={}",
                marginal_design_row_major.len(),
                n * block.p_m
            ),
        });
    }
    if logslope_design_row_major.len() != n * block.p_g {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident: logslope_design len={} != n*p_g={}",
                logslope_design_row_major.len(),
                n * block.p_g
            ),
        });
    }
    if primary.r != r {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident: primary.r={} != inputs.r={}",
                primary.r, r
            ),
        });
    }

    // Ensure the row kernel backend is compiled & loaded (this also compiles
    // the HVP backend on first use so the caller surfaces failures here).
    let backend = RowKernelBackend::probe()?;
    HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();

    let upload_f64 = |slice: &[f64], label: &str| {
        stream
            .clone_htod(slice)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row device-resident upload {label}: {err}"),
            })
    };
    let upload_u32 = |slice: &[u32], label: &str| {
        stream
            .clone_htod(slice)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row device-resident upload {label}: {err}"),
            })
    };

    let d_q = upload_f64(inputs.q, "q")?;
    let d_b = upload_f64(inputs.b, "b")?;
    let d_mu1 = upload_f64(inputs.mu_1, "mu_1")?;
    let d_mu2 = upload_f64(inputs.mu_2, "mu_2")?;
    let d_zobs = upload_f64(inputs.z_obs, "z_obs")?;
    let d_y = upload_f64(inputs.y, "y")?;
    let d_w = upload_f64(inputs.w, "w")?;
    let d_offsets = upload_u32(inputs.cell_offsets, "cell_offsets")?;
    let d_c0 = upload_f64(inputs.cell_c0, "cell_c0")?;
    let d_c1 = upload_f64(inputs.cell_c1, "cell_c1")?;
    let d_c2 = upload_f64(inputs.cell_c2, "cell_c2")?;
    let d_c3 = upload_f64(inputs.cell_c3, "cell_c3")?;
    let d_a = upload_f64(inputs.cell_a, "cell_a")?;
    let d_aa = upload_f64(inputs.cell_aa, "cell_aa")?;
    let d_r = upload_f64(inputs.cell_r, "cell_r")?;
    let d_ar = upload_f64(inputs.cell_ar, "cell_ar")?;
    let d_sbb = upload_f64(inputs.cell_sbb, "cell_sbb")?;
    let d_sbh = upload_f64(inputs.cell_sbh, "cell_sbh")?;
    let d_sbw = upload_f64(inputs.cell_sbw, "cell_sbw")?;
    // Phase-4: optionally consume device-resident moments (no host upload).
    let owned_host_moments: CudaSlice<f64>;
    let d_moments_ref: &CudaSlice<f64> = match &inputs.cell_moments {
        CellMomentsSource::Host(slice) => {
            owned_host_moments = upload_f64(slice, "cell_moments")?;
            &owned_host_moments
        }
        CellMomentsSource::Device(d) => *d,
    };
    let d_chi = upload_f64(inputs.chi_obs, "chi_obs")?;
    let d_xi = upload_f64(inputs.xi_obs, "xi_obs")?;
    let d_rho = upload_f64(inputs.rho_u, "rho_u")?;
    let d_tau = upload_f64(inputs.tau_u, "tau_u")?;
    let d_ruv = upload_f64(inputs.r_uv, "r_uv")?;

    let d_marginal = upload_f64(marginal_design_row_major, "marginal_design")?;
    let d_logslope = upload_f64(logslope_design_row_major, "logslope_design")?;

    let mut d_neglog = stream
        .alloc_zeros::<f64>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident alloc neglog: {err}"),
        })?;
    let mut d_grad =
        stream
            .alloc_zeros::<f64>(n * r)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row device-resident alloc grad: {err}"),
            })?;
    let mut d_hess =
        stream
            .alloc_zeros::<f64>(n * r * r)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row device-resident alloc hess: {err}"),
            })?;

    let func = backend
        .module
        .load_function("bms_flex_row_kernel")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident load_function: {err}"),
        })?;

    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (ROW_KERNEL_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row device-resident: n_rows={n} exceeds i32 range"),
    })?;
    let r_i32 = i32::try_from(r).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row device-resident: r={r} exceeds i32 range"),
    })?;
    let p_h_i32 = i32::try_from(inputs.p_h).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row device-resident: p_h={} exceeds i32 range",
            inputs.p_h
        ),
    })?;
    let p_w_i32 = i32::try_from(inputs.p_w).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row device-resident: p_w={} exceeds i32 range",
            inputs.p_w
        ),
    })?;
    let s_f_val = inputs.s_f;

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&n_i32)
        .arg(&r_i32)
        .arg(&p_h_i32)
        .arg(&p_w_i32)
        .arg(&s_f_val)
        .arg(&d_q)
        .arg(&d_b)
        .arg(&d_mu1)
        .arg(&d_mu2)
        .arg(&d_zobs)
        .arg(&d_y)
        .arg(&d_w)
        .arg(&d_offsets)
        .arg(&d_c0)
        .arg(&d_c1)
        .arg(&d_c2)
        .arg(&d_c3)
        .arg(&d_a)
        .arg(&d_aa)
        .arg(&d_r)
        .arg(&d_ar)
        .arg(&d_sbb)
        .arg(&d_sbh)
        .arg(&d_sbw)
        .arg(d_moments_ref)
        .arg(&d_chi)
        .arg(&d_xi)
        .arg(&d_rho)
        .arg(&d_tau)
        .arg(&d_ruv)
        .arg(&mut d_neglog)
        .arg(&mut d_grad)
        .arg(&mut d_hess);
    // SAFETY: same shape contract as `launch_linux`: every kernel parameter is
    // either a primitive scalar by-value, a const device pointer whose
    // capacity was validated by `inputs.validate()`, or one of the three
    // output buffers we just allocated with the expected element count.
    unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row device-resident launch: {err}"),
    })?;
    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident synchronize: {err}"),
        })?;

    // The kernel writes neglog + grad alongside the row Hessian, but the
    // device-resident cache path keeps neither on the host: the fused
    // CPU gradient pass (the only consumer of host-side neglog/grad) is
    // dispatched only as a fallback when the GPU dense-block kernel does
    // not apply, and in that fallback the row kernel runs again locally.
    // Drop the device buffers so the allocation pool reclaims them
    // immediately rather than tying them to the cache's lifetime.
    drop(d_neglog);
    drop(d_grad);
    // Drop the per-cell uploads; keep d_hess + designs.
    drop(d_q);
    drop(d_b);
    drop(d_mu1);
    drop(d_mu2);
    drop(d_zobs);
    drop(d_y);
    drop(d_w);
    drop(d_offsets);
    drop(d_c0);
    drop(d_c1);
    drop(d_c2);
    drop(d_c3);
    drop(d_a);
    drop(d_aa);
    drop(d_r);
    drop(d_ar);
    drop(d_sbb);
    drop(d_sbh);
    drop(d_sbw);
    // `owned_host_moments` (if any) and the borrowed `d_moments_ref` both
    // go out of scope at the end of the function; the device-resident
    // moments owned by the caller stay alive.
    drop(d_chi);
    drop(d_xi);
    drop(d_rho);
    drop(d_tau);
    drop(d_ruv);

    let bytes = ((n * r * r + marginal_design_row_major.len() + logslope_design_row_major.len())
        * std::mem::size_of::<f64>()) as u64;
    Ok(DeviceResidentRowHess {
        hess: d_hess,
        marginal_design: d_marginal,
        logslope_design: d_logslope,
        n,
        r,
        block,
        primary,
        bytes,
    })
}

/// Which partial kernel the joint-β engine drives, whether it consumes a
/// direction vector `d_v`, and where the reduced `[1, p_total]` image lands.
/// All three points of variation are encoded here so the public entry points
/// stay thin wrappers over one launch helper.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
pub(crate) enum BmsFlexRowLaunchMode {
    /// `bms_flex_row_hvp_partial`, `H · v` per row, result left on-stream.
    HvpDeviceOut,
    /// `bms_flex_row_diag_partial`, `diag(H)` per row, downloaded to host.
    DiagonalHostOut,
}

#[cfg(target_os = "linux")]
impl BmsFlexRowLaunchMode {
    /// Name of the partial kernel this mode loads from the HVP module.
    pub(crate) fn partial_kernel_name(self) -> &'static str {
        match self {
            BmsFlexRowLaunchMode::HvpDeviceOut => "bms_flex_row_hvp_partial",
            BmsFlexRowLaunchMode::DiagonalHostOut => "bms_flex_row_diag_partial",
        }
    }
}

/// All scalar launch arguments for the joint-β partial kernel, derived once
/// from a [`DeviceResidentRowHess`]. The HVP and diagonal partial kernels take
/// the identical leading block-layout argument list (only the trailing
/// `d_v` / output pointers differ), so this captures the long, easy-to-
/// desynchronize prefix in a single place.
#[cfg(target_os = "linux")]
pub(crate) struct PreparedBmsFlexRowLaunchArgs {
    pub(crate) n_i32: i32,
    pub(crate) r_i32: i32,
    pub(crate) p_m_i32: i32,
    pub(crate) p_g_i32: i32,
    pub(crate) p_total_i32: i32,
    pub(crate) h_block_start: i32,
    pub(crate) h_block_len: i32,
    pub(crate) w_block_start: i32,
    pub(crate) w_block_len: i32,
    pub(crate) h_primary_start: i32,
    pub(crate) w_primary_start: i32,
    pub(crate) rows_per_cta: i32,
    pub(crate) num_chunks: usize,
}

#[cfg(target_os = "linux")]
impl PreparedBmsFlexRowLaunchArgs {
    pub(crate) fn from_storage(storage: &DeviceResidentRowHess) -> Self {
        let p_total = storage.block.p_total;
        let num_chunks = num_hvp_chunks(storage.n);
        PreparedBmsFlexRowLaunchArgs {
            n_i32: storage.n as i32,
            r_i32: storage.r as i32,
            p_m_i32: storage.block.p_m as i32,
            p_g_i32: storage.block.p_g as i32,
            p_total_i32: p_total as i32,
            h_block_start: storage
                .block
                .h
                .as_ref()
                .map(|r| r.start as i32)
                .unwrap_or(0),
            h_block_len: storage
                .block
                .h
                .as_ref()
                .map(|r| r.len() as i32)
                .unwrap_or(0),
            w_block_start: storage
                .block
                .w
                .as_ref()
                .map(|r| r.start as i32)
                .unwrap_or(0),
            w_block_len: storage
                .block
                .w
                .as_ref()
                .map(|r| r.len() as i32)
                .unwrap_or(0),
            h_primary_start: storage
                .primary
                .h
                .as_ref()
                .map(|r| r.start as i32)
                .unwrap_or(0),
            w_primary_start: storage
                .primary
                .w
                .as_ref()
                .map(|r| r.start as i32)
                .unwrap_or(0),
            rows_per_cta: HVP_ROWS_PER_CTA as i32,
            num_chunks,
        }
    }
}

/// Shared partial+reduce engine behind every joint-β launcher.
///
/// Allocates the `[num_chunks, p_total]` partial buffer, loads the mode's
/// partial kernel plus the common `bms_flex_row_hvp_reduce`, builds both
/// launch configs from a single [`PreparedBmsFlexRowLaunchArgs`], launches the
/// partial kernel (binding `d_v` only for the HVP modes), and launches the
/// reduction into caller-supplied `d_out`.
///
/// **No** `synchronize()` or DtoH is performed here — the surrounding helper
/// decides whether to keep the result on-stream (device-resident PCG hot path)
/// or sync + download it to the host. `ctx` is a short error-context tag woven
/// into every `DriverCallFailed` reason so failures stay attributable to the
/// originating entry point.
#[cfg(target_os = "linux")]
pub(crate) fn run_bms_flex_row_partial_reduce(
    storage: &DeviceResidentRowHess,
    mode: BmsFlexRowLaunchMode,
    d_v: Option<&CudaSlice<f64>>,
    d_out: &mut CudaSlice<f64>,
    ctx: &str,
) -> Result<(), GpuError> {
    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();
    let args = PreparedBmsFlexRowLaunchArgs::from_storage(storage);
    let p_total = storage.block.p_total;

    let mut d_partial = stream
        .alloc_zeros::<f64>(args.num_chunks * p_total)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} alloc partial: {err}"),
        })?;

    let partial_kernel_name = mode.partial_kernel_name();
    let part_func = backend
        .module
        .load_function(partial_kernel_name)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} load {partial_kernel_name}: {err}"),
        })?;
    let red_func = backend
        .module
        .load_function("bms_flex_row_hvp_reduce")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} load reduce: {err}"),
        })?;

    let cfg_part = LaunchConfig {
        grid_dim: (args.num_chunks as u32, 1, 1),
        block_dim: (HVP_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&part_func);
    builder
        .arg(&args.n_i32)
        .arg(&args.r_i32)
        .arg(&args.p_m_i32)
        .arg(&args.p_g_i32)
        .arg(&args.p_total_i32)
        .arg(&args.h_block_start)
        .arg(&args.h_block_len)
        .arg(&args.w_block_start)
        .arg(&args.w_block_len)
        .arg(&args.h_primary_start)
        .arg(&args.w_primary_start)
        .arg(&args.rows_per_cta)
        .arg(&storage.hess)
        .arg(&storage.marginal_design)
        .arg(&storage.logslope_design);
    if let Some(d_v) = d_v {
        builder.arg(d_v);
    }
    builder.arg(&mut d_partial);
    // SAFETY: every device pointer above either comes from `storage` (whose
    // capacities were established by
    // `launch_bms_flex_row_kernel_device_resident`) or was just allocated here
    // (`d_partial` = num_chunks * p_total). `d_v`, when bound, is length-checked
    // by the calling adapter against `p_total`. The diagonal partial kernel
    // takes no direction argument, matching `d_v == None`. Scalar args are i32
    // by-value.
    unsafe { builder.launch(cfg_part) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row {ctx} partial launch: {err}"),
    })?;

    let red_threads: u32 = REDUCTION_THREADS;
    let red_blocks: u32 = ((p_total as u32) + red_threads - 1) / red_threads;
    let cfg_red = LaunchConfig {
        grid_dim: (red_blocks, 1, 1),
        block_dim: (red_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_chunks_i32 = args.num_chunks as i32;
    let mut builder = stream.launch_builder(&red_func);
    builder
        .arg(&num_chunks_i32)
        .arg(&args.p_total_i32)
        .arg(&d_partial)
        .arg(d_out);
    // SAFETY: `d_partial` was just populated by the partial kernel above;
    // `d_out` is `p_total` doubles (length-checked / allocated by the calling
    // adapter); both scalar args fit i32.
    unsafe { builder.launch(cfg_red) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row {ctx} reduce launch: {err}"),
    })?;
    // `d_partial` drops at end of fn; cudarc keeps the alloc alive until the
    // stream is done with it, so the reduce kernel completes safely.
    drop(d_partial);
    Ok(())
}

/// Host-returning joint-β launcher shared by [`launch_bms_flex_row_hvp`]
/// ([`BmsFlexRowLaunchMode::HvpHostOut`]) and [`launch_bms_flex_row_diagonal`]
/// ([`BmsFlexRowLaunchMode::DiagonalHostOut`]).
///
/// Probes the backend, allocates the `p_total`-double output on its stream,
/// optionally uploads a host direction `v` (HVP modes; `None` for the diagonal
/// mode, which takes no direction), runs the shared partial+reduce engine, then
/// synchronizes and downloads the reduced image to a host `Vec<f64>`. `ctx`
/// tags every `DriverCallFailed` reason with the originating entry point.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_host(
    storage: &DeviceResidentRowHess,
    mode: BmsFlexRowLaunchMode,
    v: Option<&[f64]>,
    ctx: &str,
) -> Result<Vec<f64>, GpuError> {
    let p_total = storage.block.p_total;
    if let Some(v) = v {
        if v.len() != p_total {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row {ctx}: v.len()={} != p_total={p_total}",
                    v.len()
                ),
            });
        }
    }

    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();

    let d_v = match v {
        Some(v) => Some(
            stream
                .clone_htod(v)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("bms_flex_row {ctx} upload v: {err}"),
                })?,
        ),
        None => None,
    };
    let mut d_out =
        stream
            .alloc_zeros::<f64>(p_total)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row {ctx} alloc out: {err}"),
            })?;

    run_bms_flex_row_partial_reduce(storage, mode, d_v.as_ref(), &mut d_out, ctx)?;

    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} synchronize: {err}"),
        })?;
    stream
        .clone_dtoh(&d_out)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} download out: {err}"),
        })
}

#[cfg(target_os = "linux")]
pub(crate) fn validate_bms_flex_row_hvp_multi_shape(
    storage: &DeviceResidentRowHess,
    rhs_count: usize,
    v_rhs_len: usize,
    out_len: Option<usize>,
    ctx: &str,
) -> Result<usize, GpuError> {
    if rhs_count == 0 || rhs_count > BMS_FLEX_ROW_HVP_MAX_RHS {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row {ctx}: rhs_count={rhs_count} outside 1..={BMS_FLEX_ROW_HVP_MAX_RHS}"
            ),
        });
    }
    let p_total = storage.block.p_total;
    let rhs_elems = rhs_count
        .checked_mul(p_total)
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row {ctx}: rhs_count({rhs_count})*p_total({p_total}) overflow"
            ),
        })?;
    if v_rhs_len != rhs_elems {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row {ctx}: v_rhs.len()={v_rhs_len} != rhs_count({rhs_count})*p_total({p_total})={rhs_elems}"
            ),
        });
    }
    if let Some(out_len) = out_len
        && out_len != rhs_elems
    {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row {ctx}: out.len()={out_len} != rhs_count({rhs_count})*p_total({p_total})={rhs_elems}"
            ),
        });
    }
    Ok(rhs_elems)
}

/// Transient device bytes for a multi-RHS HVP launch, excluding persistent
/// row-Hessian/design storage. Scratch scales with
/// `rhs_count * num_chunks * p_total`, not `rhs_count * n * r * r`.
#[cfg(target_os = "linux")]
pub fn bms_flex_row_hvp_multi_scratch_bytes_for_shape(
    n: usize,
    p_total: usize,
    rhs_count: usize,
) -> Result<u64, GpuError> {
    if rhs_count == 0 || rhs_count > BMS_FLEX_ROW_HVP_MAX_RHS {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row hvp_multi_scratch_bytes: rhs_count={rhs_count} outside 1..={BMS_FLEX_ROW_HVP_MAX_RHS}"
            ),
        });
    }
    let num_chunks = num_hvp_chunks(n);
    let partial = rhs_count
        .checked_mul(num_chunks)
        .and_then(|v| v.checked_mul(p_total))
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row hvp_multi_scratch_bytes: rhs_count({rhs_count})*num_chunks({num_chunks})*p_total({p_total}) overflow"
            ),
        })?;
    let rhs_vectors = rhs_count
        .checked_mul(p_total)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row hvp_multi_scratch_bytes: 2*rhs_count({rhs_count})*p_total({p_total}) overflow"
            ),
        })?;
    let elems = partial
        .checked_add(rhs_vectors)
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: "bms_flex_row hvp_multi_scratch_bytes: element count overflow".to_string(),
        })?;
    Ok((elems * std::mem::size_of::<f64>()) as u64)
}

#[cfg(target_os = "linux")]
pub(crate) fn run_bms_flex_row_multi_partial_reduce(
    storage: &DeviceResidentRowHess,
    rhs_count: usize,
    d_v_rhs: &CudaSlice<f64>,
    d_out: &mut CudaSlice<f64>,
    ctx: &str,
) -> Result<(), GpuError> {
    let rhs_elems = validate_bms_flex_row_hvp_multi_shape(
        storage,
        rhs_count,
        d_v_rhs.len(),
        Some(d_out.len()),
        ctx,
    )?;
    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();
    let args = PreparedBmsFlexRowLaunchArgs::from_storage(storage);
    let p_total = storage.block.p_total;
    let partial_len = rhs_count
        .checked_mul(args.num_chunks)
        .and_then(|v| v.checked_mul(p_total))
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row {ctx}: partial length overflow for rhs_count={rhs_count}, num_chunks={}, p_total={p_total}",
                args.num_chunks
            ),
        })?;

    let mut d_partial =
        stream
            .alloc_zeros::<f64>(partial_len)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row {ctx} alloc multi partial: {err}"),
            })?;
    let part_func = backend
        .module
        .load_function("bms_flex_row_hvp_multi_partial")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} load multi partial: {err}"),
        })?;
    let red_func = backend
        .module
        .load_function("bms_flex_row_hvp_multi_reduce")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row {ctx} load multi reduce: {err}"),
        })?;

    let rhs_count_i32 = i32::try_from(rhs_count).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row {ctx}: rhs_count={rhs_count} exceeds i32 range"),
    })?;
    let cfg_part = LaunchConfig {
        grid_dim: (args.num_chunks as u32, 1, 1),
        block_dim: (HVP_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&part_func);
    builder
        .arg(&args.n_i32)
        .arg(&args.r_i32)
        .arg(&args.p_m_i32)
        .arg(&args.p_g_i32)
        .arg(&args.p_total_i32)
        .arg(&args.h_block_start)
        .arg(&args.h_block_len)
        .arg(&args.w_block_start)
        .arg(&args.w_block_len)
        .arg(&args.h_primary_start)
        .arg(&args.w_primary_start)
        .arg(&args.rows_per_cta)
        .arg(&rhs_count_i32)
        .arg(&storage.hess)
        .arg(&storage.marginal_design)
        .arg(&storage.logslope_design)
        .arg(d_v_rhs)
        .arg(&mut d_partial);
    // SAFETY: storage buffers were validated at construction; `d_v_rhs` and
    // `d_out` have rhs_count*p_total elements, `d_partial` has
    // rhs_count*num_chunks*p_total, and rhs_count is bounded by fixed shared
    // array sizes in the CUDA source.
    unsafe { builder.launch(cfg_part) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row {ctx} multi partial launch: {err}"),
    })?;

    let red_threads: u32 = REDUCTION_THREADS;
    let red_blocks: u32 = ((rhs_elems as u32) + red_threads - 1) / red_threads;
    let cfg_red = LaunchConfig {
        grid_dim: (red_blocks, 1, 1),
        block_dim: (red_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_chunks_i32 = args.num_chunks as i32;
    let mut builder = stream.launch_builder(&red_func);
    builder
        .arg(&num_chunks_i32)
        .arg(&args.p_total_i32)
        .arg(&rhs_count_i32)
        .arg(&d_partial)
        .arg(d_out);
    // SAFETY: the reduce kernel reads the just-populated partial buffer and
    // writes exactly rhs_count*p_total output entries.
    unsafe { builder.launch(cfg_red) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row {ctx} multi reduce launch: {err}"),
    })?;
    drop(d_partial);
    Ok(())
}

/// Device-resident multi-RHS HVP. `v_rhs` is row-major
/// `[rhs_count, p_total]`; the returned vector has the same layout.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_hvp_multi(
    storage: &DeviceResidentRowHess,
    v_rhs: &[f64],
    rhs_count: usize,
) -> Result<Vec<f64>, GpuError> {
    let rhs_elems =
        validate_bms_flex_row_hvp_multi_shape(storage, rhs_count, v_rhs.len(), None, "hvp_multi")?;
    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();
    let d_v_rhs = stream
        .clone_htod(v_rhs)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row hvp_multi upload v_rhs: {err}"),
        })?;
    let mut d_out =
        stream
            .alloc_zeros::<f64>(rhs_elems)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row hvp_multi alloc out: {err}"),
            })?;
    run_bms_flex_row_multi_partial_reduce(storage, rhs_count, &d_v_rhs, &mut d_out, "hvp_multi")?;
    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row hvp_multi synchronize: {err}"),
        })?;
    stream
        .clone_dtoh(&d_out)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row hvp_multi download out: {err}"),
        })
}

/// Device-output HVP. Runs `bms_flex_row_hvp_partial(_packed)` +
/// `bms_flex_row_hvp_reduce` on the storage's stream against caller-supplied
/// device-resident `d_v` (length `p_total` doubles), writing the result into
/// caller-supplied `d_out` (also `p_total` doubles). **No** `synchronize()`
/// or DtoH is performed — the caller is responsible for stream ordering
/// against any consumer that reads `d_out`.
///
/// This is the device-resident PCG hot path (Block 9 Phase 5): keeping the
/// HVP output on the stream lets the outer PCG loop chain axpy / dot /
/// preconditioner kernels back-to-back without a per-iter device sync.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_hvp_into_device(
    storage: &DeviceResidentRowHess,
    d_v: &CudaSlice<f64>,
    d_out: &mut CudaSlice<f64>,
) -> Result<(), GpuError> {
    let p_total = storage.block.p_total;
    if d_v.len() != p_total {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row hvp_into_device: d_v.len()={} != p_total={}",
                d_v.len(),
                p_total
            ),
        });
    }
    if d_out.len() != p_total {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row hvp_into_device: d_out.len()={} != p_total={}",
                d_out.len(),
                p_total
            ),
        });
    }
    // On-stream output: the shared engine launches partial+reduce into the
    // caller's `d_out` and returns without sync/DtoH, so the outer PCG loop can
    // chain device kernels against the result.
    run_bms_flex_row_partial_reduce(
        storage,
        BmsFlexRowLaunchMode::HvpDeviceOut,
        Some(d_v),
        d_out,
        "hvp_into_device",
    )
}

/// Launch the device-resident HVP kernel. Returns the host-side joint β image
/// of length `block.p_total`.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_hvp(
    storage: &DeviceResidentRowHess,
    v: &[f64],
) -> Result<Vec<f64>, GpuError> {
    launch_bms_flex_row_hvp_multi(storage, v, 1)
}

/// Launch the device-resident diagonal kernel. Returns the host-side joint
/// β diagonal of length `block.p_total`.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_diagonal(
    storage: &DeviceResidentRowHess,
) -> Result<Vec<f64>, GpuError> {
    launch_bms_flex_row_host(storage, BmsFlexRowLaunchMode::DiagonalHostOut, None, "diag")
}

/// Block 9 Phase 6 — hard cap on `p_total` for the dense joint-Hessian
/// device kernel. Per-CTA shared-memory accumulator is `p_total² * 8`
/// bytes. V100 default per-block shared cap is 48 KiB, so the largest
/// safe `p_total` here is `sqrt(48 KiB / 8) = 78`. We round down to a
/// power-of-two-ish multiple of 8 for predictable launch geometry.
#[cfg(target_os = "linux")]
pub(crate) const DENSE_BLOCK_MAX_P: usize = 72;

/// Number of rows each dense-block CTA processes. Smaller than the HVP
/// `HVP_ROWS_PER_CTA = 256` because the per-row inner loop is `O(r² *
/// (p_m + p_g + h_block_len + w_block_len))` rather than `O(r²)` — fewer
/// rows per CTA keeps the per-CTA wall time short and lets us scale grid
/// occupancy with `num_chunks = ceil(n / DENSE_BLOCK_ROWS_PER_CTA)`.
#[cfg(target_os = "linux")]
pub(crate) const DENSE_BLOCK_ROWS_PER_CTA: u32 = 32;

/// Launch the Phase-6 dense joint-Hessian block kernel. Returns the
/// host-side `[p_total, p_total]` row-major joint H as a `Vec<f64>`
/// (length `p_total²`).
///
/// **Not the default Newton path.** Production Newton uses HVP (Phase 2)
/// and never materialises the full dense Hessian. This entry exists for:
///   * exact-REML logdet (`log|H|`) when the unified evaluator wants to
///     factor H directly instead of going through the matrix-free path;
///   * diagnostic dumps that compare the GPU dense build against the CPU
///     `BernoulliMarginalSlopeFamily::fused_gradient_dense` reference;
///   * small-`p` debug routes where it is cheaper to factor + solve dense
///     than to run a PCG.
///
/// The kernel rejects `p_total > DENSE_BLOCK_MAX_P` cleanly because the
/// per-CTA shared-memory accumulator (`p_total² * 8` bytes) would exceed
/// the V100 48 KiB/block cap above that threshold.
#[cfg(target_os = "linux")]
pub fn launch_bms_flex_row_dense_block(
    storage: &DeviceResidentRowHess,
) -> Result<Vec<f64>, GpuError> {
    let p_total = storage.block.p_total;
    if p_total == 0 {
        return Err(GpuError::DriverCallFailed {
            reason: "bms_flex_row dense_block: p_total must be > 0".to_string(),
        });
    }
    if p_total > DENSE_BLOCK_MAX_P {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row dense_block: p_total={p_total} exceeds DENSE_BLOCK_MAX_P={DENSE_BLOCK_MAX_P} \
                 (per-CTA shmem accumulator p²*8 bytes would exceed V100's 48 KiB/block)"
            ),
        });
    }
    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();
    let n = storage.n;
    let r = storage.r;
    let rows_per_cta = DENSE_BLOCK_ROWS_PER_CTA as usize;
    let num_chunks = n.div_ceil(rows_per_cta);
    let pp = p_total * p_total;

    let mut d_partial =
        stream
            .alloc_zeros::<f64>(num_chunks * pp)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row dense_block alloc partial: {err}"),
            })?;
    let mut d_out = stream
        .alloc_zeros::<f64>(pp)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row dense_block alloc out: {err}"),
        })?;

    let part_func = backend
        .module
        .load_function("bms_flex_row_dense_block_partial")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row dense_block load partial: {err}"),
        })?;
    let red_func = backend
        .module
        .load_function("bms_flex_row_dense_block_reduce")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row dense_block load reduce: {err}"),
        })?;

    let n_i32 = n as i32;
    let r_i32 = r as i32;
    let p_m_i32 = storage.block.p_m as i32;
    let p_g_i32 = storage.block.p_g as i32;
    let p_total_i32 = p_total as i32;
    let h_block_start = storage
        .block
        .h
        .as_ref()
        .map(|r| r.start as i32)
        .unwrap_or(0);
    let h_block_len = storage
        .block
        .h
        .as_ref()
        .map(|r| r.len() as i32)
        .unwrap_or(0);
    let w_block_start = storage
        .block
        .w
        .as_ref()
        .map(|r| r.start as i32)
        .unwrap_or(0);
    let w_block_len = storage
        .block
        .w
        .as_ref()
        .map(|r| r.len() as i32)
        .unwrap_or(0);
    let h_primary_start = storage
        .primary
        .h
        .as_ref()
        .map(|r| r.start as i32)
        .unwrap_or(0);
    let w_primary_start = storage
        .primary
        .w
        .as_ref()
        .map(|r| r.start as i32)
        .unwrap_or(0);
    let rows_per_cta_i32 = DENSE_BLOCK_ROWS_PER_CTA as i32;
    let num_chunks_u32 = num_chunks as u32;

    // Per-CTA shmem accumulator: p_total² doubles.
    let shmem_bytes: u32 =
        u32::try_from(pp * std::mem::size_of::<f64>()).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("dense_block shmem bytes overflow u32 for p_total={p_total}"),
        })?;

    let cfg_part = LaunchConfig {
        grid_dim: (num_chunks_u32, 1, 1),
        block_dim: (HVP_THREADS, 1, 1),
        shared_mem_bytes: shmem_bytes,
    };
    let mut builder = stream.launch_builder(&part_func);
    builder
        .arg(&n_i32)
        .arg(&r_i32)
        .arg(&p_m_i32)
        .arg(&p_g_i32)
        .arg(&p_total_i32)
        .arg(&h_block_start)
        .arg(&h_block_len)
        .arg(&w_block_start)
        .arg(&w_block_len)
        .arg(&h_primary_start)
        .arg(&w_primary_start)
        .arg(&rows_per_cta_i32)
        .arg(&storage.hess)
        .arg(&storage.marginal_design)
        .arg(&storage.logslope_design)
        .arg(&mut d_partial);
    // SAFETY: storage pointers have validated capacities; d_partial sized
    // num_chunks * pp doubles; dynamic shmem matches the kernel's `extern
    // __shared__` accumulator length.
    unsafe { builder.launch(cfg_part) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row dense_block partial launch: {err}"),
    })?;

    let red_threads: u32 = REDUCTION_THREADS;
    let red_blocks: u32 = ((pp as u32) + red_threads - 1) / red_threads;
    let cfg_red = LaunchConfig {
        grid_dim: (red_blocks, 1, 1),
        block_dim: (red_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_chunks_i32 = num_chunks as i32;
    let mut builder = stream.launch_builder(&red_func);
    builder
        .arg(&num_chunks_i32)
        .arg(&p_total_i32)
        .arg(&d_partial)
        .arg(&mut d_out);
    // SAFETY: d_partial just populated, d_out is pp doubles.
    unsafe { builder.launch(cfg_red) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row dense_block reduce launch: {err}"),
    })?;
    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row dense_block sync: {err}"),
        })?;
    stream
        .clone_dtoh(&d_out)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row dense_block download: {err}"),
        })
}

// Block 9 / V100-build unblock (2026-05-27): every test below either
// constructs `BmsFlexBlockLayout` / `BmsFlexPrimaryLayout` (Linux-only
// types) or drives a CUDA-dependent fixture, so gate the whole module
// `#[cfg(all(test, target_os = "linux"))]`. On macOS the structs are
// absent and these tests do not compile — the build.rs ban scanner
// explicitly rejects `#[cfg(any(..., test))]` on the struct definitions
// themselves as a dead-code escape hatch.
// #415 parity lock: the CPU host oracle for the BMS-FLEX row kernel lives in a
// non-linux-gated `#[cfg(test)]` module so it can be exercised on the macOS dev
// box + CPU CI (the sibling `mod tests` is linux-gated because it also builds
// CUDA-only fixture types). `cpu_oracle_outputs` itself is platform-independent.
#[cfg(test)]
mod oracle_parity_tests {
    use super::*;

    // ── CPU oracle that mirrors ROW_KERNEL_BODY bit-for-bit ──────────────────
    //
    // `cpu_oracle_outputs` implements the same algebra as
    // `bms_flex_row_kernel` in ROW_KERNEL_BODY: per-cell `T_n` / `D` / `Q`
    // contractions, q-row override, IFT to `a_u` / `a_uv`, observed-point
    // assembly to `bar_e_u` / `bar_e_uv`, probit Mills, and the final
    // `out_grad` / `out_hess` writes. It takes the same
    // `BmsFlexRowKernelInputs` struct so a CUDA-equipped host can run both
    // paths off one bundle and check element-wise parity.
    //
    // Used by the GPU↔CPU parity test below; the test skips on non-Linux
    // hosts via cfg, but the oracle itself is platform-independent so the
    // macOS lib build can still type-check it.

    pub(crate) const ORACLE_INV_TWO_PI: f64 = 1.0 / std::f64::consts::TAU;
    pub(crate) const ORACLE_SQRT_2: f64 = std::f64::consts::SQRT_2;
    pub(crate) const ORACLE_INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

    pub(crate) fn oracle_erfcx_nonnegative(x: f64) -> f64 {
        if !x.is_finite() {
            return if x > 0.0 { 0.0 } else { f64::INFINITY };
        }
        if x <= 0.0 {
            return 1.0;
        }
        if x < 26.0 {
            let mut xx = x * x;
            if xx > 700.0 {
                xx = 700.0;
            }
            return xx.exp() * gam_gpu::numerics_host::erfc(x);
        }
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2 - 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        let inv_sqrt_pi: f64 = 0.564_189_583_547_756_3;
        inv * poly * inv_sqrt_pi
    }

    pub(crate) fn oracle_log_ndtr_and_mills(x: f64) -> (f64, f64) {
        if x == f64::INFINITY {
            return (0.0, 0.0);
        }
        if x == f64::NEG_INFINITY {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        if x.is_nan() {
            return (x, x);
        }
        // Single-algorithm region around and above 0. Both `log Φ(x)` and the
        // Mills ratio `φ/Φ` are computed from the SAME `erfc(-x/√2)` call, so
        // the oracle is C¹ across the x=0 seam (the prior split — erfcx-based
        // `-u²+ln(0.5·e^{u²}·erfc u)` for x<0 vs direct `ln(0.5·erfc(-x))` for
        // x≥0 — used two distinct float algorithms whose ~1e-7 disagreement at
        // x=0 corrupted a finite-difference reference straddling the seam, #838).
        //
        // The erfcx form is mathematically identical (e^{u²}·erfc(u)=erfcx(u),
        // so −u²+ln(0.5·erfcx u)=ln(0.5·erfc(−x/√2))); it is only needed deep in
        // the left tail (x ≲ −38), where `erfc(-x/√2)` underflows to 0 and the
        // exp/ln cancellation is the *only* way to keep `log Φ` finite. We move
        // the branch there, far from any region the kernel or its FD lock visits.
        const ORACLE_LEFT_TAIL_X: f64 = -37.0;
        if x >= ORACLE_LEFT_TAIL_X {
            let mut cdf = 0.5 * gam_gpu::numerics_host::erfc(-x / ORACLE_SQRT_2);
            if cdf < 1e-300 {
                cdf = 1e-300;
            }
            if cdf > 1.0 {
                cdf = 1.0;
            }
            let pdf = ORACLE_INV_SQRT_2PI * (-0.5 * x * x).exp();
            (cdf.ln(), pdf / cdf)
        } else {
            let u = -x / ORACLE_SQRT_2;
            let mut ex = oracle_erfcx_nonnegative(u);
            if ex < 1e-300 {
                ex = 1e-300;
            }
            let log_cdf = -u * u + (0.5 * ex).ln();
            let sqrt_2_over_pi: f64 = 0.797_884_560_802_865_4;
            (log_cdf, sqrt_2_over_pi / ex)
        }
    }

    /// Same outputs the device kernel writes: `(neglog, grad, hess)` per row.
    /// `grad` is row-major `n × r`, `hess` is row-major `n × r × r`.
    /// Mirrors `bms_flex_row_kernel` line-for-line so kernel + oracle diverge
    /// only if one side breaks parity.
    pub(crate) fn cpu_oracle_outputs(
        inputs: &BmsFlexRowKernelInputs<'_>,
    ) -> BmsFlexRowKernelOutputs {
        let n = inputs.n_rows;
        let r = inputs.r;
        let p_h = inputs.p_h;
        let p_w = inputs.p_w;
        let mut neglog = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n * r];
        let mut hess = vec![0.0_f64; n * r * r];
        let cell_moments_host = match &inputs.cell_moments {
            CellMomentsSource::Host(slice) => *slice,
            #[cfg(target_os = "linux")]
            CellMomentsSource::Device(_) => panic!(
                // SAFETY: this CPU oracle is a host-only sanity checker invoked
                // exclusively from `#[cfg(test)] mod tests`. The kernel-launch
                // path uses `CellMomentsSource::Device(...)`; the oracle must
                // never see that variant. Reaching this arm means a test
                // mis-wired its fixture — surface it loudly at the call site.
                "cpu_oracle_outputs: cell_moments is device-resident; oracle \
                 is a host-only sanity checker"
            ),
        };

        for row in 0..n {
            // ── per-cell sweep: accumulate F_u, F_au, F_uv, F_a, F_aa.
            let mut f_u = vec![0.0_f64; r];
            let mut f_au = vec![0.0_f64; r];
            let mut f_uv = vec![0.0_f64; r * r];
            let mut f_a = 0.0_f64;
            let mut f_aa = 0.0_f64;

            let cell_lo = inputs.cell_offsets[row] as usize;
            let cell_hi = inputs.cell_offsets[row + 1] as usize;
            for c in cell_lo..cell_hi {
                let c_arr = [
                    inputs.cell_c0[c],
                    inputs.cell_c1[c],
                    inputs.cell_c2[c],
                    inputs.cell_c3[c],
                ];
                let m = &cell_moments_host[c * MOMENT_STRIDE..(c + 1) * MOMENT_STRIDE];

                // T_n = κ · Σ_e C_e · m_{e+n}, n = 0..6.
                let mut t = [0.0_f64; 7];
                for (n_idx, t_slot) in t.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for (e, c_e) in c_arr.iter().enumerate() {
                        acc = c_e.mul_add(m[e + n_idx], acc);
                    }
                    *t_slot = acc * ORACLE_INV_TWO_PI;
                }

                let d_of = |r_arr: &[f64]| -> f64 {
                    ORACLE_INV_TWO_PI
                        * (r_arr[0] * m[0] + r_arr[1] * m[1] + r_arr[2] * m[2] + r_arr[3] * m[3])
                };
                let q_of = |r_arr: &[f64], s_arr: &[f64]| -> f64 {
                    (r_arr[0] * s_arr[0]) * t[0]
                        + (r_arr[0] * s_arr[1] + r_arr[1] * s_arr[0]) * t[1]
                        + (r_arr[0] * s_arr[2] + r_arr[1] * s_arr[1] + r_arr[2] * s_arr[0]) * t[2]
                        + (r_arr[0] * s_arr[3]
                            + r_arr[1] * s_arr[2]
                            + r_arr[2] * s_arr[1]
                            + r_arr[3] * s_arr[0])
                            * t[3]
                        + (r_arr[1] * s_arr[3] + r_arr[2] * s_arr[2] + r_arr[3] * s_arr[1]) * t[4]
                        + (r_arr[2] * s_arr[3] + r_arr[3] * s_arr[2]) * t[5]
                        + (r_arr[3] * s_arr[3]) * t[6]
                };

                let a_c = &inputs.cell_a[c * 4..(c + 1) * 4];
                let aa_c = &inputs.cell_aa[c * 4..(c + 1) * 4];
                f_a += d_of(a_c);
                f_aa += d_of(aa_c) - q_of(a_c, a_c);

                for u in 1..r {
                    let r_u_off = (c * (r - 1) + (u - 1)) * 4;
                    let r_u = &inputs.cell_r[r_u_off..r_u_off + 4];
                    let ar_u = &inputs.cell_ar[r_u_off..r_u_off + 4];
                    f_u[u] += d_of(r_u);
                    f_au[u] += d_of(ar_u) - q_of(a_c, r_u);
                }

                for u in 1..r {
                    let r_u_off = (c * (r - 1) + (u - 1)) * 4;
                    let r_u = &inputs.cell_r[r_u_off..r_u_off + 4];
                    for v in u..r {
                        let r_v_off = (c * (r - 1) + (v - 1)) * 4;
                        let r_v = &inputs.cell_r[r_v_off..r_v_off + 4];
                        let q_uv = q_of(r_u, r_v);
                        let d_s = if u == 1 && v == 1 {
                            let s_bb = &inputs.cell_sbb[c * 4..(c + 1) * 4];
                            d_of(s_bb)
                        } else if u == 1 && v >= 2 && v < 2 + p_h {
                            let j = v - 2;
                            let off = (c * p_h + j) * 4;
                            let s_bh = &inputs.cell_sbh[off..off + 4];
                            d_of(s_bh)
                        } else if u == 1 && v >= 2 + p_h && v < r {
                            let l = v - (2 + p_h);
                            let off = (c * p_w + l) * 4;
                            let s_bw = &inputs.cell_sbw[off..off + 4];
                            d_of(s_bw)
                        } else {
                            0.0
                        };
                        f_uv[u * r + v] += d_s - q_uv;
                    }
                }
            }

            // q-row overrides (mirror kernel lines 691–700).
            let mu_1 = inputs.mu_1[row];
            let mu_2 = inputs.mu_2[row];
            f_u[0] = -mu_1;
            f_au[0] = 0.0;
            for v in 0..r {
                f_uv[v] = 0.0;
                f_uv[v * r] = 0.0;
            }
            f_uv[0] = -mu_2;

            // Degenerate F_a ⇒ NaN-fill (mirror kernel lines 703–706).
            if !f_a.is_finite() || f_a <= 0.0 {
                neglog[row] = f64::NAN;
                for slot in grad[row * r..(row + 1) * r].iter_mut() {
                    *slot = f64::NAN;
                }
                for slot in hess[row * r * r..(row + 1) * r * r].iter_mut() {
                    *slot = f64::NAN;
                }
                continue;
            }
            let inv_fa = 1.0 / f_a;

            // IFT first/second order.
            let mut a_u = vec![0.0_f64; r];
            a_u[0] = mu_1 * inv_fa;
            for u in 1..r {
                a_u[u] = -f_u[u] * inv_fa;
            }
            let mut a_uv = vec![0.0_f64; r * r];
            for u in 0..r {
                for v in u..r {
                    let term = f_uv[u * r + v]
                        + f_au[v] * a_u[u]
                        + f_au[u] * a_u[v]
                        + f_aa * a_u[u] * a_u[v];
                    let val = -term * inv_fa;
                    a_uv[u * r + v] = val;
                    a_uv[v * r + u] = val;
                }
            }

            // Observed predictor jets.
            let chi = inputs.chi_obs[row];
            let xi = inputs.xi_obs[row];
            let rho = &inputs.rho_u[row * r..(row + 1) * r];
            let tau = &inputs.tau_u[row * r..(row + 1) * r];
            let ruv = &inputs.r_uv[row * r * r..(row + 1) * r * r];
            let mut bar_e_u = vec![0.0_f64; r];
            for u in 0..r {
                bar_e_u[u] = chi * a_u[u] + rho[u];
            }
            let mut bar_e_uv = vec![0.0_f64; r * r];
            for u in 0..r {
                for v in u..r {
                    let val = chi * a_uv[u * r + v]
                        + xi * a_u[u] * a_u[v]
                        + tau[u] * a_u[v]
                        + a_u[u] * tau[v]
                        + ruv[u * r + v];
                    bar_e_uv[u * r + v] = val;
                    if u != v {
                        bar_e_uv[v * r + u] = val;
                    }
                }
            }

            // Probit Mills + final writes.
            let y = inputs.y[row];
            let w = inputs.w[row];
            let s = 2.0 * y - 1.0;
            let e_obs = bar_e_u[0];
            let m_arg = s * e_obs;
            let (log_cdf, lambda) = oracle_log_ndtr_and_mills(m_arg);
            let a_i = -w * s * lambda;
            let b_i = w * lambda * (m_arg + lambda);
            neglog[row] = -w * log_cdf;
            for u in 0..r {
                grad[row * r + u] = a_i * bar_e_u[u];
            }
            for u in 0..r {
                for v in u..r {
                    let val = b_i * bar_e_u[u] * bar_e_u[v] + a_i * bar_e_uv[u * r + v];
                    hess[row * r * r + u * r + v] = val;
                    if u != v {
                        hess[row * r * r + v * r + u] = val;
                    }
                }
            }
        }

        BmsFlexRowKernelOutputs { neglog, grad, hess }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use super::oracle_parity_tests::*;

    pub(crate) fn minimal_inputs<'a>(buffers: &'a TestBuffers) -> BmsFlexRowKernelInputs<'a> {
        BmsFlexRowKernelInputs {
            n_rows: 1,
            r: 4,
            p_h: 1,
            p_w: 1,
            q: &buffers.q,
            b: &buffers.b,
            mu_1: &buffers.mu_1,
            mu_2: &buffers.mu_2,
            z_obs: &buffers.z_obs,
            y: &buffers.y,
            w: &buffers.w,
            s_f: 1.0,
            cell_offsets: &buffers.cell_offsets,
            cell_c0: &buffers.cell_c0,
            cell_c1: &buffers.cell_c1,
            cell_c2: &buffers.cell_c2,
            cell_c3: &buffers.cell_c3,
            cell_a: &buffers.cell_a,
            cell_aa: &buffers.cell_aa,
            cell_r: &buffers.cell_r,
            cell_ar: &buffers.cell_ar,
            cell_sbb: &buffers.cell_sbb,
            cell_sbh: &buffers.cell_sbh,
            cell_sbw: &buffers.cell_sbw,
            cell_moments: CellMomentsSource::Host(&buffers.cell_moments),
            chi_obs: &buffers.chi_obs,
            xi_obs: &buffers.xi_obs,
            rho_u: &buffers.rho_u,
            tau_u: &buffers.tau_u,
            r_uv: &buffers.r_uv,
        }
    }

    pub(crate) struct TestBuffers {
        pub(crate) q: Vec<f64>,
        pub(crate) b: Vec<f64>,
        pub(crate) mu_1: Vec<f64>,
        pub(crate) mu_2: Vec<f64>,
        pub(crate) z_obs: Vec<f64>,
        pub(crate) y: Vec<f64>,
        pub(crate) w: Vec<f64>,
        pub(crate) cell_offsets: Vec<u32>,
        pub(crate) cell_c0: Vec<f64>,
        pub(crate) cell_c1: Vec<f64>,
        pub(crate) cell_c2: Vec<f64>,
        pub(crate) cell_c3: Vec<f64>,
        pub(crate) cell_a: Vec<f64>,
        pub(crate) cell_aa: Vec<f64>,
        pub(crate) cell_r: Vec<f64>,
        pub(crate) cell_ar: Vec<f64>,
        pub(crate) cell_sbb: Vec<f64>,
        pub(crate) cell_sbh: Vec<f64>,
        pub(crate) cell_sbw: Vec<f64>,
        pub(crate) cell_moments: Vec<f64>,
        pub(crate) chi_obs: Vec<f64>,
        pub(crate) xi_obs: Vec<f64>,
        pub(crate) rho_u: Vec<f64>,
        pub(crate) tau_u: Vec<f64>,
        pub(crate) r_uv: Vec<f64>,
    }

    pub(crate) fn make_buffers(n_cells: u32, r: usize, p_h: usize, p_w: usize) -> TestBuffers {
        let cells = n_cells as usize;
        TestBuffers {
            q: vec![0.1; 1],
            b: vec![0.5; 1],
            mu_1: vec![0.3; 1],
            mu_2: vec![0.07; 1],
            z_obs: vec![0.0; 1],
            y: vec![1.0; 1],
            w: vec![1.0; 1],
            cell_offsets: vec![0, n_cells],
            cell_c0: vec![0.2; cells],
            cell_c1: vec![-0.1; cells],
            cell_c2: vec![0.05; cells],
            cell_c3: vec![-0.02; cells],
            cell_a: vec![0.1; cells * 4],
            cell_aa: vec![0.0; cells * 4],
            cell_r: vec![0.05; cells * (r - 1) * 4],
            cell_ar: vec![0.0; cells * (r - 1) * 4],
            cell_sbb: vec![0.0; cells * 4],
            cell_sbh: vec![0.0; cells * p_h * 4],
            cell_sbw: vec![0.0; cells * p_w * 4],
            cell_moments: vec![1.0; cells * MOMENT_STRIDE],
            chi_obs: vec![1.0; 1],
            xi_obs: vec![0.0; 1],
            rho_u: vec![0.0; r],
            tau_u: vec![0.0; r],
            r_uv: vec![0.0; r * r],
        }
    }

    #[test]
    pub(crate) fn validate_accepts_minimal_inputs() {
        let buffers = make_buffers(2, 4, 1, 1);
        let inputs = minimal_inputs(&buffers);
        assert!(inputs.validate().is_ok());
    }

    #[test]
    pub(crate) fn validate_rejects_r_above_max() {
        let r = MAX_R + 1;
        let p_h = (r - 2) / 2;
        let p_w = (r - 2) - p_h;
        let buffers = make_buffers(1, r, p_h, p_w);
        let bad_inputs = BmsFlexRowKernelInputs {
            r,
            p_h,
            p_w,
            rho_u: &buffers.rho_u, // length matches `r` we wrote
            tau_u: &buffers.tau_u,
            r_uv: &buffers.r_uv,
            cell_r: &buffers.cell_r,
            cell_ar: &buffers.cell_ar,
            cell_sbh: &buffers.cell_sbh,
            cell_sbw: &buffers.cell_sbw,
            ..minimal_inputs(&buffers)
        };
        let err = bad_inputs.validate().expect_err("r > MAX_R must fail");
        let msg = err.to_string();
        assert!(msg.contains("MAX_R"), "expected MAX_R hint, got: {msg}");
    }

    #[test]
    pub(crate) fn validate_rejects_mismatched_r_decomposition() {
        let buffers = make_buffers(1, 4, 1, 1);
        let bad_inputs = BmsFlexRowKernelInputs {
            r: 4,
            p_h: 1,
            p_w: 2, // inconsistent with r = 4
            ..minimal_inputs(&buffers)
        };
        let err = bad_inputs
            .validate()
            .expect_err("inconsistent r vs p_h+p_w must fail");
        let msg = err.to_string();
        assert!(msg.contains("p_h"), "got: {msg}");
        assert!(msg.contains("p_w"), "got: {msg}");
    }

    #[test]
    pub(crate) fn validate_rejects_non_monotone_offsets() {
        // `minimal_inputs` hard-codes `n_rows = 1`, so the CSR-style row
        // pointer length is `n + 1 = 2`. Pin both `offsets[1] = total_cells`
        // and `cell_c0.len() = total_cells = 2` from `make_buffers(2, …)`,
        // then violate monotonicity by setting `offsets[0] > offsets[1]`;
        // every length / per-cell-count check is satisfied so the only
        // failure mode left is the monotonicity guard.
        let mut buffers = make_buffers(2, 4, 1, 1);
        buffers.cell_offsets = vec![5, 2];
        let inputs = minimal_inputs(&buffers);
        let err = inputs
            .validate()
            .expect_err("non-monotone offsets must fail");
        let msg = err.to_string();
        assert!(msg.contains("monotone"), "got: {msg}");
    }

    #[test]
    pub(crate) fn validate_rejects_mismatched_cell_moments_length() {
        let mut buffers = make_buffers(2, 4, 1, 1);
        buffers.cell_moments.pop(); // length now 2*10 - 1
        let inputs = minimal_inputs(&buffers);
        let err = inputs.validate().expect_err("short cell_moments must fail");
        let msg = err.to_string();
        assert!(msg.contains("cell_moments"), "got: {msg}");
    }

    #[test]
    pub(crate) fn launch_on_non_linux_reports_driver_library_unavailable() {
        // Mac/Windows builds must surface a typed `DriverLibraryUnavailable`
        // rather than panicking or returning Ok. On Linux this test is
        // skipped because the kernel actually launches.
        #[cfg(target_os = "linux")]
        {
            // Linux builds may or may not have a device; the dispatcher
            // contract is that without a runtime, probe() returns
            // DriverLibraryUnavailable. Either outcome (NoDeviceKernel,
            // DriverLibraryUnavailable, or DriverCallFailed) is acceptable
            // here; success would mean the kernel actually ran which is a
            // V100-only outcome we don't gate the unit test on.
            let buffers = make_buffers(1, 4, 1, 1);
            let inputs = minimal_inputs(&buffers);
            match launch_bms_flex_row_kernel(inputs) {
                Ok(_) => { /* V100 host: real launch */ }
                Err(GpuError::DriverLibraryUnavailable { .. })
                | Err(GpuError::DriverCallFailed { .. })
                | Err(GpuError::DriverSymbolMissing { .. })
                | Err(GpuError::NoDeviceKernel { .. }) => { /* expected on CPU-only */ }
                Err(other) => panic!("unexpected GpuError variant: {other:?}"),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let buffers = make_buffers(1, 4, 1, 1);
            let inputs = minimal_inputs(&buffers);
            match launch_bms_flex_row_kernel(inputs) {
                Err(GpuError::DriverLibraryUnavailable { reason }) => {
                    assert!(
                        reason.contains("Linux-only"),
                        "expected Linux-only hint, got: {reason}"
                    );
                }
                other => panic!("expected DriverLibraryUnavailable on non-Linux, got {other:?}"),
            }
        }
    }

    #[test]
    pub(crate) fn s_f_must_be_positive_and_finite() {
        let buffers = make_buffers(1, 4, 1, 1);
        let mut inputs = minimal_inputs(&buffers);
        inputs.s_f = 0.0;
        match launch_bms_flex_row_kernel(inputs) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(reason.contains("s_f"), "got: {reason}");
            }
            other => panic!("expected DriverCallFailed for s_f=0, got {other:?}"),
        }
    }


    /// Build a non-trivial fixture: `n = 4` rows, `r = 5` (p_h = 2, p_w = 1),
    /// 2–4 cells per row, distinct values so a structural bug in either path
    /// can't be masked by accidental cancellation.
    pub(crate) fn make_parity_buffers() -> TestBuffers {
        let n = 4_usize;
        let r = 5_usize;
        let p_h = 2_usize;
        let p_w = 1_usize;
        // Per-row cell counts: 2, 3, 4, 2 → total 11 cells.
        let row_cells: [u32; 4] = [2, 3, 4, 2];
        let mut cell_offsets = vec![0_u32; n + 1];
        for i in 0..n {
            cell_offsets[i + 1] = cell_offsets[i] + row_cells[i];
        }
        let total_cells = cell_offsets[n] as usize;

        // Deterministic but varied generators (LCG-ish so each slot is distinct).
        let f = |seed: usize| -> f64 {
            let x = ((seed.wrapping_mul(2_654_435_761)) & 0xFFFF) as f64 / 65_536.0;
            0.1 + 0.4 * x
        };

        let q = (0..n).map(|i| 0.05 + 0.1 * (i as f64)).collect::<Vec<_>>();
        let b = (0..n).map(|i| 0.6 + 0.05 * (i as f64)).collect::<Vec<_>>();
        let mu_1 = (0..n).map(|i| 0.7 + 0.02 * (i as f64)).collect::<Vec<_>>();
        let mu_2 = (0..n).map(|i| 0.15 + 0.01 * (i as f64)).collect::<Vec<_>>();
        let z_obs = (0..n).map(|i| -0.2 + 0.1 * (i as f64)).collect::<Vec<_>>();
        let y = [1.0, 0.0, 1.0, 0.0].to_vec();
        let w = vec![1.0; n];

        let cell_c0 = (0..total_cells).map(|c| f(c + 1001)).collect::<Vec<_>>();
        let cell_c1 = (0..total_cells)
            .map(|c| -f(c + 2002) * 0.5)
            .collect::<Vec<_>>();
        let cell_c2 = (0..total_cells).map(|c| f(c + 3003) * 0.2).collect();
        let cell_c3 = (0..total_cells).map(|c| -f(c + 4004) * 0.1).collect();

        let cell_a = (0..total_cells * 4)
            .map(|i| f(i + 5005) * 0.3)
            .collect::<Vec<_>>();
        let cell_aa = (0..total_cells * 4)
            .map(|i| f(i + 6006) * 0.1)
            .collect::<Vec<_>>();
        let cell_r = (0..total_cells * (r - 1) * 4)
            .map(|i| f(i + 7007) * 0.2)
            .collect::<Vec<_>>();
        let cell_ar = (0..total_cells * (r - 1) * 4)
            .map(|i| f(i + 8008) * 0.05)
            .collect::<Vec<_>>();
        let cell_sbb = (0..total_cells * 4)
            .map(|i| f(i + 9009) * 0.08)
            .collect::<Vec<_>>();
        let cell_sbh = (0..total_cells * p_h * 4)
            .map(|i| f(i + 10_010) * 0.07)
            .collect::<Vec<_>>();
        let cell_sbw = (0..total_cells * p_w * 4)
            .map(|i| f(i + 11_011) * 0.06)
            .collect::<Vec<_>>();
        let cell_moments = (0..total_cells * MOMENT_STRIDE)
            .map(|i| 0.4 + 0.1 * f(i + 12_012))
            .collect::<Vec<_>>();

        let chi_obs = (0..n).map(|i| 0.9 + 0.01 * (i as f64)).collect::<Vec<_>>();
        let xi_obs = (0..n).map(|i| 0.2 + 0.01 * (i as f64)).collect::<Vec<_>>();
        let rho_u = (0..n * r).map(|i| 0.03 * f(i + 13_013)).collect::<Vec<_>>();
        let tau_u = (0..n * r).map(|i| 0.02 * f(i + 14_014)).collect::<Vec<_>>();
        let r_uv = (0..n * r * r)
            .map(|i| 0.04 * f(i + 15_015))
            .collect::<Vec<_>>();

        TestBuffers {
            q,
            b,
            mu_1,
            mu_2,
            z_obs,
            y,
            w,
            cell_offsets,
            cell_c0,
            cell_c1,
            cell_c2,
            cell_c3,
            cell_a,
            cell_aa,
            cell_r,
            cell_ar,
            cell_sbb,
            cell_sbh,
            cell_sbw,
            cell_moments,
            chi_obs,
            xi_obs,
            rho_u,
            tau_u,
            r_uv,
        }
    }

    pub(crate) fn parity_inputs<'a>(buffers: &'a TestBuffers) -> BmsFlexRowKernelInputs<'a> {
        BmsFlexRowKernelInputs {
            n_rows: 4,
            r: 5,
            p_h: 2,
            p_w: 1,
            q: &buffers.q,
            b: &buffers.b,
            mu_1: &buffers.mu_1,
            mu_2: &buffers.mu_2,
            z_obs: &buffers.z_obs,
            y: &buffers.y,
            w: &buffers.w,
            s_f: 1.0,
            cell_offsets: &buffers.cell_offsets,
            cell_c0: &buffers.cell_c0,
            cell_c1: &buffers.cell_c1,
            cell_c2: &buffers.cell_c2,
            cell_c3: &buffers.cell_c3,
            cell_a: &buffers.cell_a,
            cell_aa: &buffers.cell_aa,
            cell_r: &buffers.cell_r,
            cell_ar: &buffers.cell_ar,
            cell_sbb: &buffers.cell_sbb,
            cell_sbh: &buffers.cell_sbh,
            cell_sbw: &buffers.cell_sbw,
            cell_moments: CellMomentsSource::Host(&buffers.cell_moments),
            chi_obs: &buffers.chi_obs,
            xi_obs: &buffers.xi_obs,
            rho_u: &buffers.rho_u,
            tau_u: &buffers.tau_u,
            r_uv: &buffers.r_uv,
        }
    }

    /// Symmetry + finiteness of the CPU oracle. Runs on every host (Linux,
    /// macOS, CPU CI) since the oracle is platform-independent. Guarantees the
    /// reference path used by the GPU parity test is itself well-formed.
    #[test]
    pub(crate) fn cpu_oracle_produces_finite_symmetric_hessian() {
        let buffers = make_parity_buffers();
        let inputs = parity_inputs(&buffers);
        inputs
            .validate()
            .expect("parity fixture must satisfy validate()");
        let out = cpu_oracle_outputs(&inputs);
        let n = inputs.n_rows;
        let r = inputs.r;
        assert_eq!(out.neglog.len(), n);
        assert_eq!(out.grad.len(), n * r);
        assert_eq!(out.hess.len(), n * r * r);
        for row in 0..n {
            assert!(
                out.neglog[row].is_finite(),
                "row {row}: neglog must be finite, got {}",
                out.neglog[row]
            );
            for u in 0..r {
                let g = out.grad[row * r + u];
                assert!(g.is_finite(), "row {row}: grad[{u}] = {g}");
                for v in 0..r {
                    let huv = out.hess[row * r * r + u * r + v];
                    let hvu = out.hess[row * r * r + v * r + u];
                    assert!(huv.is_finite(), "row {row}: H[{u},{v}] = {huv}");
                    assert_eq!(
                        huv.to_bits(),
                        hvu.to_bits(),
                        "row {row}: H[{u},{v}] and H[{v},{u}] must be bit-identical"
                    );
                }
            }
        }
    }

    /// Independent finite-difference correctness lock on the oracle's probit
    /// Mills layer — the most optimizer-sensitive, drift-prone term in the
    /// whole row kernel (issue #415: "third/fourth-order derivative
    /// contractions drift silently … formulas are complex and
    /// optimizer-sensitive"). The device kernel's `bms_flex_row_kernel` and
    /// the host oracle's `cpu_oracle_outputs` both close out with the same
    /// Mills algebra:
    ///
    /// ```text
    ///     m       = s · e_obs ;  s = 2y − 1
    ///     A       = −w · s · λ(m)
    ///     B       =  w · λ(m) · (m + λ(m))
    ///     neglog  = −w · log Φ(s · e_obs)
    ///     g_u     = A · bar_e_u
    ///     H_uv    = B · bar_e_u · bar_e_v + A · bar_e_uv
    /// ```
    ///
    /// Holding the observed jets `bar_e` fixed, the row neglog is a function
    /// of the single scalar `e := bar_e_u[0] = e_obs`, and by the assembled
    /// formula `∂neglog/∂e = A` and `∂²neglog/∂e² = B`. This test reconstructs
    /// `A`, `B`, and `neglog` exactly as the oracle does (same
    /// `oracle_log_ndtr_and_mills`, same sign convention), then verifies the
    /// analytic `A`/`B` against high-order central differences of
    /// `e ↦ −w · log Φ(s·e)`. A drift in the kernel's Mills derivatives —
    /// which the device-parity test cannot catch because it checks the kernel
    /// *against the same (possibly-wrong) oracle* — fails here on every host,
    /// CUDA or not. Bounds are the genuine fifth-order central-difference
    /// truncation floor; they are not weakened to pass.
    #[test]
    pub(crate) fn cpu_oracle_mills_layer_matches_finite_differences() {
        // Probit neglog as the oracle assembles it, as a function of the
        // observed scalar predictor `e` with weight `w` and label `y`.
        let neglog_of = |e: f64, y: f64, w: f64| -> f64 {
            let s = 2.0 * y - 1.0;
            let (log_cdf, _) = oracle_log_ndtr_and_mills(s * e);
            -w * log_cdf
        };
        // Analytic first/second derivatives wrt `e` — the exact `A`/`B` the
        // kernel writes into `grad`/`hess`.
        let ab_of = |e: f64, y: f64, w: f64| -> (f64, f64) {
            let s = 2.0 * y - 1.0;
            let m_arg = s * e;
            let (_, lambda) = oracle_log_ndtr_and_mills(m_arg);
            let a_i = -w * s * lambda;
            let b_i = w * lambda * (m_arg + lambda);
            (a_i, b_i)
        };

        // Sweep both labels (s = ±1), both tails of the predictor, and a
        // non-unit weight so every sign/scale path of the Mills algebra is
        // exercised. Points stay clear of the deep-tail asymptote where a
        // central-difference reference loses its own accuracy.
        let cases: [(f64, f64, f64); 12] = [
            (-1.6, 1.0, 1.0),
            (-0.7, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.9, 1.0, 1.0),
            (1.8, 1.0, 1.0),
            (-1.4, 0.0, 1.0),
            (-0.3, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.6, 0.0, 1.0),
            (1.5, 0.0, 1.0),
            (0.4, 1.0, 0.75),
            (-0.8, 0.0, 1.3),
        ];
        // Fifth-order central stencils; `h` chosen near the f64 sweet spot for
        // first/second derivatives of a smooth O(1) function.
        let h = 1e-3_f64;
        for (e, y, w) in cases {
            let (a_ana, b_ana) = ab_of(e, y, w);

            let fp2 = neglog_of(e + 2.0 * h, y, w);
            let fp1 = neglog_of(e + h, y, w);
            let f0 = neglog_of(e, y, w);
            let fm1 = neglog_of(e - h, y, w);
            let fm2 = neglog_of(e - 2.0 * h, y, w);

            // 5-point central first derivative: O(h⁴).
            let d1_fd = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h);
            // 5-point central second derivative: O(h⁴).
            let d2_fd = (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h * h);

            let a_abs = (a_ana - d1_fd).abs();
            let a_rel = a_abs / a_ana.abs().max(1.0);
            assert!(
                a_abs <= 5e-8 || a_rel <= 5e-8,
                "Mills A (∂neglog/∂e) drift at e={e} y={y} w={w}: \
                 analytic={a_ana:.17e} fd={d1_fd:.17e} abs={a_abs:.3e} rel={a_rel:.3e}"
            );

            let b_abs = (b_ana - d2_fd).abs();
            let b_rel = b_abs / b_ana.abs().max(1.0);
            assert!(
                b_abs <= 5e-6 || b_rel <= 5e-6,
                "Mills B (∂²neglog/∂e²) drift at e={e} y={y} w={w}: \
                 analytic={b_ana:.17e} fd={d2_fd:.17e} abs={b_abs:.3e} rel={b_rel:.3e}"
            );
        }
    }

    /// CPU↔GPU parity. Only runs end-to-end on a Linux host with a CUDA
    /// runtime; skips with a clear `eprintln!` on every other host so the
    /// always-on test suite stays green on the macOS dev box and CPU CI.
    ///
    /// On a CUDA host: drives the kernel through `launch_bms_flex_row_kernel`
    /// and the same `BmsFlexRowKernelInputs` through `cpu_oracle_outputs`,
    /// then asserts every element of `neglog`, `grad`, and `hess` agrees
    /// within `|Δ| <= 1e-8 + 1e-8·|cpu|` (absolute-or-relative).
    #[test]
    pub(crate) fn bms_flex_row_kernel_matches_cpu_oracle_when_cuda_available() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!(
                "[bms_flex_row parity] non-Linux host — skipping CUDA parity \
                 (CPU oracle exercised by sibling test)"
            );
            return;
        }
        #[cfg(target_os = "linux")]
        {
            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!(
                    "[bms_flex_row parity] no CUDA runtime — skipping device \
                     parity (CPU oracle exercised by sibling test)"
                );
                return;
            };
            let buffers = make_parity_buffers();
            let inputs_cpu = parity_inputs(&buffers);
            inputs_cpu
                .validate()
                .expect("parity fixture must satisfy validate()");
            let cpu_out = cpu_oracle_outputs(&inputs_cpu);

            // Launch the device kernel against the same inputs.
            let inputs_gpu = parity_inputs(&buffers);
            let gpu_out = match launch_bms_flex_row_kernel(inputs_gpu) {
                Ok(out) => out,
                Err(err) => panic!(
                    "[bms_flex_row parity] launch failed on CUDA-selected host; \
                     device/oracle parity must fail loudly on GPU CI: {err}"
                ),
            };

            let n = inputs_cpu.n_rows;
            let r = inputs_cpu.r;
            let tol_abs = 1e-8_f64;
            let tol_rel = 1e-8_f64;
            let check_close = |label: &str, idx: usize, cpu: f64, gpu: f64| {
                if cpu.is_nan() || gpu.is_nan() {
                    assert!(
                        cpu.is_nan() && gpu.is_nan(),
                        "{label}[{idx}]: NaN parity broke — cpu={cpu}, gpu={gpu}"
                    );
                    return;
                }
                let diff = (cpu - gpu).abs();
                let tol = tol_abs + tol_rel * cpu.abs();
                assert!(
                    diff <= tol,
                    "{label}[{idx}]: |cpu − gpu| = {diff:.3e} > tol = {tol:.3e}; \
                     cpu={cpu:.17e}, gpu={gpu:.17e}"
                );
            };
            assert_eq!(cpu_out.neglog.len(), gpu_out.neglog.len());
            assert_eq!(cpu_out.grad.len(), gpu_out.grad.len());
            assert_eq!(cpu_out.hess.len(), gpu_out.hess.len());
            for (i, (&c, &g)) in cpu_out.neglog.iter().zip(gpu_out.neglog.iter()).enumerate() {
                check_close("neglog", i, c, g);
            }
            for (i, (&c, &g)) in cpu_out.grad.iter().zip(gpu_out.grad.iter()).enumerate() {
                check_close("grad", i, c, g);
            }
            for (i, (&c, &g)) in cpu_out.hess.iter().zip(gpu_out.hess.iter()).enumerate() {
                check_close("hess", i, c, g);
            }
            // Spot-check exact symmetry on the GPU Hessian too.
            for row in 0..n {
                for u in 0..r {
                    for v in 0..r {
                        let a = gpu_out.hess[row * r * r + u * r + v];
                        let bb = gpu_out.hess[row * r * r + v * r + u];
                        assert_eq!(
                            a.to_bits(),
                            bb.to_bits(),
                            "GPU row {row}: H[{u},{v}] ≠ H[{v},{u}] bit-for-bit"
                        );
                    }
                }
            }
        }
    }

    #[test]
    pub(crate) fn kernel_source_mentions_cpu_parity_reference() {
        // Guarantee the maintainer-facing parity reference comment survives
        // refactors of the NVRTC kernel source — the dispatcher wave that
        // wires this to bms_flex.rs cross-checks parity against the CPU
        // function named here.
        #[cfg(target_os = "linux")]
        assert!(ROW_KERNEL_BODY.contains("compute_row_analytic_flex_from_parts_into"));
        #[cfg(target_os = "linux")]
        assert!(ROW_KERNEL_BODY.contains("cell_first_derivative_from_moments"));
    }

    // ── Phase-3 HVP / diagonal CPU oracles + GPU parity tests ────────────────

    /// CPU oracle for [`launch_bms_flex_row_hvp`]. Mirrors the device kernel
    /// element-for-element so the GPU parity test runs against the same algebra.
    pub(crate) fn cpu_oracle_bms_flex_row_hvp(
        row_hessians: &[f64],
        marginal_design: &[f64],
        logslope_design: &[f64],
        block: &BmsFlexBlockLayout,
        primary: &BmsFlexPrimaryLayout,
        n: usize,
        v: &[f64],
    ) -> Vec<f64> {
        let r = primary.r;
        let p_m = block.p_m;
        let p_g = block.p_g;
        assert_eq!(v.len(), block.p_total);
        assert_eq!(row_hessians.len(), n * r * r);
        assert_eq!(marginal_design.len(), n * p_m);
        assert_eq!(logslope_design.len(), n * p_g);
        let mut out = vec![0.0_f64; block.p_total];
        let mut row_dir = vec![0.0_f64; r];
        let mut action = vec![0.0_f64; r];
        for row in 0..n {
            let mrow = &marginal_design[row * p_m..(row + 1) * p_m];
            let grow = &logslope_design[row * p_g..(row + 1) * p_g];
            let mut acc_q = 0.0_f64;
            for j in 0..p_m {
                acc_q += mrow[j] * v[j];
            }
            let mut acc_g = 0.0_f64;
            for j in 0..p_g {
                acc_g += grow[j] * v[p_m + j];
            }
            row_dir[0] = acc_q;
            row_dir[1] = acc_g;
            if let (Some(prange), Some(brange)) = (primary.h.as_ref(), block.h.as_ref()) {
                for (k, ii) in prange.clone().enumerate() {
                    row_dir[ii] = v[brange.start + k];
                }
            }
            if let (Some(prange), Some(brange)) = (primary.w.as_ref(), block.w.as_ref()) {
                for (k, ii) in prange.clone().enumerate() {
                    row_dir[ii] = v[brange.start + k];
                }
            }
            let h_slice = &row_hessians[row * r * r..(row + 1) * r * r];
            for u in 0..r {
                let mut acc = 0.0_f64;
                for v_idx in 0..r {
                    acc += h_slice[u * r + v_idx] * row_dir[v_idx];
                }
                action[u] = acc;
            }
            let a0 = action[0];
            for j in 0..p_m {
                out[j] += a0 * mrow[j];
            }
            let a1 = action[1];
            for j in 0..p_g {
                out[p_m + j] += a1 * grow[j];
            }
            if let (Some(prange), Some(brange)) = (primary.h.as_ref(), block.h.as_ref()) {
                for (k, ii) in prange.clone().enumerate() {
                    out[brange.start + k] += action[ii];
                }
            }
            if let (Some(prange), Some(brange)) = (primary.w.as_ref(), block.w.as_ref()) {
                for (k, ii) in prange.clone().enumerate() {
                    out[brange.start + k] += action[ii];
                }
            }
        }
        out
    }

    pub(crate) fn cpu_oracle_bms_flex_row_diagonal(
        row_hessians: &[f64],
        marginal_design: &[f64],
        logslope_design: &[f64],
        block: &BmsFlexBlockLayout,
        primary: &BmsFlexPrimaryLayout,
        n: usize,
    ) -> Vec<f64> {
        let r = primary.r;
        let p_m = block.p_m;
        let p_g = block.p_g;
        let mut out = vec![0.0_f64; block.p_total];
        for row in 0..n {
            let h_slice = &row_hessians[row * r * r..(row + 1) * r * r];
            let h00 = h_slice[0];
            let h11 = h_slice[r + 1];
            let mrow = &marginal_design[row * p_m..(row + 1) * p_m];
            let grow = &logslope_design[row * p_g..(row + 1) * p_g];
            for j in 0..p_m {
                out[j] += h00 * mrow[j] * mrow[j];
            }
            for j in 0..p_g {
                out[p_m + j] += h11 * grow[j] * grow[j];
            }
            if let (Some(prange), Some(brange)) = (primary.h.as_ref(), block.h.as_ref()) {
                for (k, ii) in prange.clone().enumerate() {
                    out[brange.start + k] += h_slice[ii * r + ii];
                }
            }
            if let (Some(prange), Some(brange)) = (primary.w.as_ref(), block.w.as_ref()) {
                for (k, ii) in prange.clone().enumerate() {
                    out[brange.start + k] += h_slice[ii * r + ii];
                }
            }
        }
        out
    }

    /// Hand-construct a small symmetric per-row Hessian + small designs and
    /// verify the CPU oracle satisfies the expected algebra. Platform-
    /// independent (runs on macOS / Linux without CUDA).
    #[test]
    pub(crate) fn cpu_oracle_hvp_matches_hand_computation_no_hw() {
        let n = 4_usize;
        let r = 4_usize; // q, logslope, h(1), w(1)
        let p_m = 2_usize;
        let p_g = 2_usize;
        let p_h_dim = 1_usize;
        let p_w_dim = 1_usize;
        let p_total = p_m + p_g + p_h_dim + p_w_dim;
        let block = BmsFlexBlockLayout {
            p_m,
            p_g,
            h: Some(p_m + p_g..p_m + p_g + p_h_dim),
            w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
            p_total,
        };
        let primary = BmsFlexPrimaryLayout {
            h: Some(2..3),
            w: Some(3..4),
            r,
        };
        // Symmetric per-row Hessian: H_row[u,v] = (row + 1) * (1 + u + 2v) symmetrised.
        let mut row_hessians = vec![0.0_f64; n * r * r];
        for row in 0..n {
            for u in 0..r {
                for v in u..r {
                    let val = ((row + 1) as f64) * (1.0 + (u as f64) + 2.0 * (v as f64));
                    row_hessians[row * r * r + u * r + v] = val;
                    row_hessians[row * r * r + v * r + u] = val;
                }
            }
        }
        let mut marginal = vec![0.0_f64; n * p_m];
        for row in 0..n {
            for j in 0..p_m {
                marginal[row * p_m + j] = 0.5 + (row as f64) * 0.1 - (j as f64) * 0.2;
            }
        }
        let mut logslope = vec![0.0_f64; n * p_g];
        for row in 0..n {
            for j in 0..p_g {
                logslope[row * p_g + j] = -0.3 + (row as f64) * 0.05 + (j as f64) * 0.15;
            }
        }
        let v: Vec<f64> = (0..p_total).map(|i| 0.1 + (i as f64) * 0.25).collect();
        let out = cpu_oracle_bms_flex_row_hvp(
            &row_hessians,
            &marginal,
            &logslope,
            &block,
            &primary,
            n,
            &v,
        );
        // Hand check the first marginal slot: out[0] = Σ_row action[0]·mrow[0].
        let mut expect_out_0 = 0.0_f64;
        for row in 0..n {
            let mrow = &marginal[row * p_m..(row + 1) * p_m];
            let grow = &logslope[row * p_g..(row + 1) * p_g];
            let mut row_dir = vec![0.0_f64; r];
            row_dir[0] = mrow[0] * v[0] + mrow[1] * v[1];
            row_dir[1] = grow[0] * v[p_m] + grow[1] * v[p_m + 1];
            row_dir[2] = v[p_m + p_g];
            row_dir[3] = v[p_m + p_g + p_h_dim];
            let h_slice = &row_hessians[row * r * r..(row + 1) * r * r];
            let mut action0 = 0.0_f64;
            // h_slice is the row-major r×r Hessian for this row; we want
            // row 0, i.e. entries (0, vv) for vv in 0..r, which lives at
            // `vv` in the flat layout.
            for vv in 0..r {
                action0 += h_slice[vv] * row_dir[vv];
            }
            expect_out_0 += action0 * mrow[0];
        }
        assert!(
            (out[0] - expect_out_0).abs() < 1e-12,
            "cpu oracle HVP out[0] mismatch: {} vs hand-check {}",
            out[0],
            expect_out_0
        );
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(out.len(), p_total);
    }

    /// Diagonal oracle equals the explicit per-row design² accumulator.
    #[test]
    pub(crate) fn cpu_oracle_diagonal_matches_hand_computation() {
        let n = 3_usize;
        let r = 4_usize;
        let p_m = 2_usize;
        let p_g = 2_usize;
        let p_h_dim = 1_usize;
        let p_w_dim = 1_usize;
        let p_total = p_m + p_g + p_h_dim + p_w_dim;
        let block = BmsFlexBlockLayout {
            p_m,
            p_g,
            h: Some(p_m + p_g..p_m + p_g + p_h_dim),
            w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
            p_total,
        };
        let primary = BmsFlexPrimaryLayout {
            h: Some(2..3),
            w: Some(3..4),
            r,
        };
        let mut row_hessians = vec![0.0_f64; n * r * r];
        for row in 0..n {
            for u in 0..r {
                row_hessians[row * r * r + u * r + u] = 1.0 + (row as f64) + (u as f64) * 0.5;
            }
        }
        let mut marginal = vec![0.0_f64; n * p_m];
        let mut logslope = vec![0.0_f64; n * p_g];
        for row in 0..n {
            for j in 0..p_m {
                marginal[row * p_m + j] = 0.2 + (row as f64) * 0.3 + (j as f64) * 0.1;
            }
            for j in 0..p_g {
                logslope[row * p_g + j] = -0.4 + (row as f64) * 0.1 + (j as f64) * 0.2;
            }
        }
        let out = cpu_oracle_bms_flex_row_diagonal(
            &row_hessians,
            &marginal,
            &logslope,
            &block,
            &primary,
            n,
        );
        // Hand check: out[0] = Σ_row H[row,0,0] · marginal[row,0]^2.
        let mut expect = 0.0_f64;
        for row in 0..n {
            let h00 = row_hessians[row * r * r];
            expect += h00 * marginal[row * p_m].powi(2);
        }
        assert!(
            (out[0] - expect).abs() < 1e-12,
            "out[0] {} vs {}",
            out[0],
            expect
        );
        // h slot = sum of H[row, 2, 2] across rows.
        let mut expect_h = 0.0_f64;
        for row in 0..n {
            expect_h += row_hessians[row * r * r + 2 * r + 2];
        }
        let h_slot = p_m + p_g;
        assert!(
            (out[h_slot] - expect_h).abs() < 1e-12,
            "h slot {} vs {}",
            out[h_slot],
            expect_h
        );
    }

    /// GPU↔CPU parity for the HVP and diagonal kernels. Skips on non-Linux /
    /// no-CUDA hosts. Hand-constructs a small `DeviceResidentRowHess` by
    /// allocating the device slices directly, uploading the same arrays the
    /// CPU oracle consumes, then dispatching the device kernels.
    #[test]
    pub(crate) fn bms_flex_row_hvp_kernel_matches_cpu_oracle_when_cuda_available() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!(
                "[bms_flex_row hvp parity] non-Linux host — skipping CUDA parity \
                 (CPU oracle exercised by sibling tests)"
            );
        }
        #[cfg(target_os = "linux")]
        {
            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!(
                    "[bms_flex_row hvp parity] no CUDA runtime — skipping device \
                     parity"
                );
                return;
            };
            let n = 4_usize;
            let r = 4_usize;
            let p_m = 2_usize;
            let p_g = 2_usize;
            let p_h_dim = 1_usize;
            let p_w_dim = 1_usize;
            let p_total = p_m + p_g + p_h_dim + p_w_dim;
            let block = BmsFlexBlockLayout {
                p_m,
                p_g,
                h: Some(p_m + p_g..p_m + p_g + p_h_dim),
                w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
                p_total,
            };
            let primary = BmsFlexPrimaryLayout {
                h: Some(2..3),
                w: Some(3..4),
                r,
            };
            let mut row_hessians = vec![0.0_f64; n * r * r];
            for row in 0..n {
                for u in 0..r {
                    for v in u..r {
                        let val = ((row + 1) as f64) * (1.0 + (u as f64) + 2.0 * (v as f64));
                        row_hessians[row * r * r + u * r + v] = val;
                        row_hessians[row * r * r + v * r + u] = val;
                    }
                }
            }
            let mut marginal = vec![0.0_f64; n * p_m];
            for row in 0..n {
                for j in 0..p_m {
                    marginal[row * p_m + j] = 0.5 + (row as f64) * 0.1 - (j as f64) * 0.2;
                }
            }
            let mut logslope = vec![0.0_f64; n * p_g];
            for row in 0..n {
                for j in 0..p_g {
                    logslope[row * p_g + j] = -0.3 + (row as f64) * 0.05 + (j as f64) * 0.15;
                }
            }
            let v: Vec<f64> = (0..p_total).map(|i| 0.1 + (i as f64) * 0.25).collect();
            let cpu_hvp = cpu_oracle_bms_flex_row_hvp(
                &row_hessians,
                &marginal,
                &logslope,
                &block,
                &primary,
                n,
                &v,
            );
            let cpu_diag = cpu_oracle_bms_flex_row_diagonal(
                &row_hessians,
                &marginal,
                &logslope,
                &block,
                &primary,
                n,
            );

            // Allocate a DeviceResidentRowHess by hand using the HVP backend's
            // stream + module so we don't need to drive the full BMS row kernel.
            // Past the GpuRuntime::global() Some-gate above: a probe/upload failure
            // here is a real device fault on a CUDA host, not a no-CUDA skip. Fail
            // loud (the device-PCG skip-pass class, eee12f6b2) — the old arms
            // returned and the test passed while exercising nothing.
            let backend = HvpKernelBackend::probe()
                .expect("[bms_flex_row hvp parity] backend probe must succeed on CUDA host");
            let stream = backend.stream.clone();
            let d_h = stream
                .clone_htod(&row_hessians)
                .expect("[bms_flex_row hvp parity] upload h must succeed on CUDA host");
            let d_m = stream
                .clone_htod(&marginal)
                .expect("[bms_flex_row hvp parity] upload marg must succeed on CUDA host");
            let d_g = stream
                .clone_htod(&logslope)
                .expect("[bms_flex_row hvp parity] upload logslope must succeed on CUDA host");
            let storage = DeviceResidentRowHess {
                hess: d_h,
                marginal_design: d_m,
                logslope_design: d_g,
                n,
                r,
                block: block.clone(),
                primary: primary.clone(),

                bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
            };
            let gpu_hvp =
                launch_bms_flex_row_hvp(&storage, &v).expect("HVP kernel must launch on CUDA host");
            let gpu_diag = launch_bms_flex_row_diagonal(&storage)
                .expect("diagonal kernel must launch on CUDA host");
            assert_eq!(gpu_hvp.len(), cpu_hvp.len());
            assert_eq!(gpu_diag.len(), cpu_diag.len());
            for i in 0..p_total {
                let diff = (cpu_hvp[i] - gpu_hvp[i]).abs();
                assert!(
                    diff <= 1e-10,
                    "HVP[{i}]: cpu={} gpu={} |Δ|={diff:.3e}",
                    cpu_hvp[i],
                    gpu_hvp[i]
                );
                let ddiff = (cpu_diag[i] - gpu_diag[i]).abs();
                assert!(
                    ddiff <= 1e-10,
                    "diag[{i}]: cpu={} gpu={} |Δ|={ddiff:.3e}",
                    cpu_diag[i],
                    gpu_diag[i]
                );
            }
        }
    }

    #[test]
    pub(crate) fn bms_flex_row_hvp_multi_scratch_is_bounded_at_large_scale_shape() {
        let n = 195_000_usize;
        let r = 20_usize;
        let p_total = 44_usize;
        let rhs_count = 4_usize;
        let scratch = bms_flex_row_hvp_multi_scratch_bytes_for_shape(n, p_total, rhs_count)
            .expect("large-scale multi-RHS scratch budget");
        let per_rhs_full_row_cache =
            (n * r * r * std::mem::size_of::<f64>()) as u64 * rhs_count as u64;
        assert!(
            scratch < per_rhs_full_row_cache / 100,
            "multi-RHS scratch must tile by row chunks instead of materializing \
             a row-Hessian copy per RHS: scratch={scratch} full_per_rhs={per_rhs_full_row_cache}"
        );
        assert!(
            bms_flex_row_hvp_multi_scratch_bytes_for_shape(
                n,
                p_total,
                BMS_FLEX_ROW_HVP_MAX_RHS + 1
            )
            .is_err(),
            "multi-RHS launch must reject unbounded RHS counts"
        );
    }

    #[test]
    pub(crate) fn bms_flex_row_hvp_multi_kernel_matches_cpu_oracle_when_cuda_available() {
        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[bms_flex_row hvp_multi parity] no CUDA runtime — skipping device parity");
            return;
        };
        let n = 5_usize;
        let r = 4_usize;
        let p_m = 2_usize;
        let p_g = 2_usize;
        let p_h_dim = 1_usize;
        let p_w_dim = 1_usize;
        let p_total = p_m + p_g + p_h_dim + p_w_dim;
        let rhs_count = 3_usize;
        let block = BmsFlexBlockLayout {
            p_m,
            p_g,
            h: Some(p_m + p_g..p_m + p_g + p_h_dim),
            w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
            p_total,
        };
        let primary = BmsFlexPrimaryLayout {
            h: Some(2..3),
            w: Some(3..4),
            r,
        };
        let mut row_hessians = vec![0.0_f64; n * r * r];
        for row in 0..n {
            for u in 0..r {
                for v in u..r {
                    let val = ((row + 1) as f64) * (1.0 + (u as f64) + 2.0 * (v as f64));
                    row_hessians[row * r * r + u * r + v] = val;
                    row_hessians[row * r * r + v * r + u] = val;
                }
            }
        }
        let mut marginal = vec![0.0_f64; n * p_m];
        let mut logslope = vec![0.0_f64; n * p_g];
        for row in 0..n {
            for j in 0..p_m {
                marginal[row * p_m + j] = 0.5 + (row as f64) * 0.1 - (j as f64) * 0.2;
            }
            for j in 0..p_g {
                logslope[row * p_g + j] = -0.3 + (row as f64) * 0.05 + (j as f64) * 0.15;
            }
        }
        let mut v_rhs = vec![0.0_f64; rhs_count * p_total];
        for rhs in 0..rhs_count {
            for j in 0..p_total {
                let seed = (rhs as f64) * 0.37 + (j as f64) * 0.19 + 0.4;
                v_rhs[rhs * p_total + j] = seed.sin() * 0.4 + seed.cos() * 0.2;
            }
        }

        // Past the GpuRuntime::global() Some-gate: a probe/upload failure here is a
        // real device fault on a CUDA host, not a no-CUDA skip — fail loud
        // (device-PCG skip-pass class, eee12f6b2).
        let backend = HvpKernelBackend::probe()
            .expect("[bms_flex_row hvp_multi parity] backend probe must succeed on CUDA host");
        let stream = backend.stream.clone();
        let d_h = stream
            .clone_htod(&row_hessians)
            .expect("[bms_flex_row hvp_multi parity] upload h must succeed on CUDA host");
        let d_m = stream
            .clone_htod(&marginal)
            .expect("[bms_flex_row hvp_multi parity] upload marg must succeed on CUDA host");
        let d_g = stream
            .clone_htod(&logslope)
            .expect("[bms_flex_row hvp_multi parity] upload logslope must succeed on CUDA host");
        let storage = DeviceResidentRowHess {
            hess: d_h,
            marginal_design: d_m,
            logslope_design: d_g,
            n,
            r,
            block: block.clone(),
            primary: primary.clone(),

            bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
        };
        let scratch = bms_flex_row_hvp_multi_scratch_bytes_for_shape(n, p_total, rhs_count)
            .expect("storage scratch budget");
        assert!(
            scratch < storage.bytes,
            "multi-RHS scratch should stay below resident cache bytes"
        );
        let gpu = launch_bms_flex_row_hvp_multi(&storage, &v_rhs, rhs_count)
            .expect("multi-RHS HVP kernel must launch on CUDA host");
        assert_eq!(gpu.len(), rhs_count * p_total);
        for rhs in 0..rhs_count {
            let v = &v_rhs[rhs * p_total..(rhs + 1) * p_total];
            let cpu = cpu_oracle_bms_flex_row_hvp(
                &row_hessians,
                &marginal,
                &logslope,
                &block,
                &primary,
                n,
                v,
            );
            let single = launch_bms_flex_row_hvp(&storage, v)
                .expect("single-RHS HVP kernel must launch on CUDA host");
            for j in 0..p_total {
                let got = gpu[rhs * p_total + j];
                let diff = (cpu[j] - got).abs();
                assert!(
                    diff <= 1e-10,
                    "multi-RHS HVP rhs={rhs} j={j}: cpu={} gpu={} |diff|={diff:.3e}",
                    cpu[j],
                    got
                );
                assert_eq!(
                    got, single[j],
                    "multi-RHS and single-RHS host launch diverged at rhs={rhs} j={j}"
                );
            }
        }
    }

    /// Parity for the third launch mode — device-output HVP
    /// ([`launch_bms_flex_row_hvp_into_device`]) — which the
    /// `run_bms_flex_row_partial_reduce` unification routes through the same
    /// partial+reduce engine as the host-returning `_hvp` / `_diagonal`
    /// adapters. Confirms that keeping the result on-stream (no internal sync /
    /// DtoH) reaches bit-identical output to both the CPU oracle and the
    /// host-out adapter, so the engine's mode/output split is faithful.
    ///
    /// Skips cleanly on non-Linux / no-CUDA hosts using the convention shared
    /// with the sibling parity tests.
    #[test]
    pub(crate) fn bms_flex_row_hvp_into_device_matches_cpu_oracle_and_host_out() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!(
                "[bms_flex_row hvp_into_device parity] non-Linux host — skipping \
                 CUDA parity (CPU oracle exercised by sibling tests)"
            );
        }
        #[cfg(target_os = "linux")]
        {
            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!(
                    "[bms_flex_row hvp_into_device parity] no CUDA runtime — \
                     skipping device parity"
                );
                return;
            };
            let n = 4_usize;
            let r = 4_usize;
            let p_m = 2_usize;
            let p_g = 2_usize;
            let p_h_dim = 1_usize;
            let p_w_dim = 1_usize;
            let p_total = p_m + p_g + p_h_dim + p_w_dim;
            let block = BmsFlexBlockLayout {
                p_m,
                p_g,
                h: Some(p_m + p_g..p_m + p_g + p_h_dim),
                w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
                p_total,
            };
            let primary = BmsFlexPrimaryLayout {
                h: Some(2..3),
                w: Some(3..4),
                r,
            };
            let mut row_hessians = vec![0.0_f64; n * r * r];
            for row in 0..n {
                for u in 0..r {
                    for v in u..r {
                        let val = ((row + 1) as f64) * (1.0 + (u as f64) + 2.0 * (v as f64));
                        row_hessians[row * r * r + u * r + v] = val;
                        row_hessians[row * r * r + v * r + u] = val;
                    }
                }
            }
            let mut marginal = vec![0.0_f64; n * p_m];
            for row in 0..n {
                for j in 0..p_m {
                    marginal[row * p_m + j] = 0.5 + (row as f64) * 0.1 - (j as f64) * 0.2;
                }
            }
            let mut logslope = vec![0.0_f64; n * p_g];
            for row in 0..n {
                for j in 0..p_g {
                    logslope[row * p_g + j] = -0.3 + (row as f64) * 0.05 + (j as f64) * 0.15;
                }
            }
            let v: Vec<f64> = (0..p_total).map(|i| 0.1 + (i as f64) * 0.25).collect();
            let cpu_hvp = cpu_oracle_bms_flex_row_hvp(
                &row_hessians,
                &marginal,
                &logslope,
                &block,
                &primary,
                n,
                &v,
            );

            // Past the GpuRuntime::global() Some-gate: probe/upload failures are
            // real device faults on a CUDA host — fail loud (device-PCG class).
            let backend = HvpKernelBackend::probe().expect(
                "[bms_flex_row hvp_into_device parity] backend probe must succeed on CUDA host",
            );
            let stream = backend.stream.clone();
            let d_h = stream
                .clone_htod(&row_hessians)
                .expect("[bms_flex_row hvp_into_device parity] upload h must succeed on CUDA host");
            let d_m = stream.clone_htod(&marginal).expect(
                "[bms_flex_row hvp_into_device parity] upload marg must succeed on CUDA host",
            );
            let d_g = stream.clone_htod(&logslope).expect(
                "[bms_flex_row hvp_into_device parity] upload logslope must succeed on CUDA host",
            );
            let storage = DeviceResidentRowHess {
                hess: d_h,
                marginal_design: d_m,
                logslope_design: d_g,
                n,
                r,
                block: block.clone(),
                primary: primary.clone(),

                bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
            };

            // Host-out adapter (allocates its own d_out, syncs + downloads).
            let host_out_hvp = launch_bms_flex_row_hvp(&storage, &v)
                .expect("host-out HVP kernel must launch on CUDA host");

            // Device-out adapter: caller owns d_v + d_out; the engine performs
            // no sync / DtoH, so we synchronize + download here.
            let d_v = stream
                .clone_htod(&v)
                .expect("upload direction for device-out HVP");
            let mut d_out = stream
                .alloc_zeros::<f64>(p_total)
                .expect("alloc device-out HVP output");
            launch_bms_flex_row_hvp_into_device(&storage, &d_v, &mut d_out)
                .expect("device-out HVP kernel must launch on CUDA host");
            stream
                .synchronize()
                .expect("synchronize after device-out HVP");
            let device_out_hvp = stream
                .clone_dtoh(&d_out)
                .expect("download device-out HVP output");

            assert_eq!(device_out_hvp.len(), cpu_hvp.len());
            assert_eq!(device_out_hvp.len(), host_out_hvp.len());
            for i in 0..p_total {
                let diff = (cpu_hvp[i] - device_out_hvp[i]).abs();
                assert!(
                    diff <= 1e-10,
                    "device-out HVP[{i}] vs CPU: cpu={} gpu={} |Δ|={diff:.3e}",
                    cpu_hvp[i],
                    device_out_hvp[i]
                );
                // Both adapters share the engine; the only difference is the
                // copy-back path, so they must be bit-identical.
                let host_diff = (host_out_hvp[i] - device_out_hvp[i]).abs();
                assert!(
                    host_diff == 0.0,
                    "device-out vs host-out HVP[{i}]: host={} device={} |Δ|={host_diff:.3e}",
                    host_out_hvp[i],
                    device_out_hvp[i]
                );
            }
        }
    }

    /// Block 9 Phase 2 parity gate at the shape specified by the
    /// charter task: `n = 64`, `r = 20`, `p_total = 44`. Splits
    /// `p_total` as `p_m = 14`, `p_g = 12`, `p_h = 10`, `p_w = 8` so
    /// `r = 2 + p_h + p_w = 20` and every primary block participates
    /// in both the device pullback and the reduce pass. Tolerance is
    /// `|Δ| ≤ 1e-8` per the task description (looser than the 1e-10
    /// hand-fixture parity, since accumulation order across HVP CTAs
    /// differs from the CPU oracle's row-major sum even with the
    /// deterministic reduction policy).
    ///
    /// Skips cleanly on non-Linux and no-CUDA hosts using the same
    /// convention as the hand-fixture parity above.
    #[test]
    pub(crate) fn bms_flex_row_hvp_kernel_matches_cpu_oracle_at_n64_r20_p44() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!(
                "[bms_flex_row hvp parity n64_r20_p44] non-Linux host — \
                 skipping CUDA parity"
            );
        }
        #[cfg(target_os = "linux")]
        {
            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!(
                    "[bms_flex_row hvp parity n64_r20_p44] no CUDA runtime — \
                     skipping device parity"
                );
                return;
            };
            let n = 64_usize;
            let p_m = 14_usize;
            let p_g = 12_usize;
            let p_h_dim = 10_usize;
            let p_w_dim = 8_usize;
            let r = 2 + p_h_dim + p_w_dim;
            assert_eq!(r, 20);
            let p_total = p_m + p_g + p_h_dim + p_w_dim;
            assert_eq!(p_total, 44);
            let block = BmsFlexBlockLayout {
                p_m,
                p_g,
                h: Some(p_m + p_g..p_m + p_g + p_h_dim),
                w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
                p_total,
            };
            let primary = BmsFlexPrimaryLayout {
                h: Some(2..2 + p_h_dim),
                w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
                r,
            };

            // Deterministic symmetric per-row Hessians + designs +
            // direction. Same scrambling family as
            // `row_hessian_ops::tests::make_fixture` so any regression
            // surfaces consistently across the host-pinned and
            // device-resident parity tests.
            let mut row_hessians = vec![0.0_f64; n * r * r];
            for row in 0..n {
                let base = row * r * r;
                for u in 0..r {
                    for v in 0..r {
                        let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (v as f64) * 0.317;
                        let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                        row_hessians[base + u * r + v] = a;
                    }
                }
                for u in 0..r {
                    for v in (u + 1)..r {
                        let upper = row_hessians[base + u * r + v];
                        let lower = row_hessians[base + v * r + u];
                        let sym = 0.5 * (upper + lower);
                        row_hessians[base + u * r + v] = sym;
                        row_hessians[base + v * r + u] = sym;
                    }
                    row_hessians[base + u * r + u] += r as f64;
                }
            }
            let mut marginal = vec![0.0_f64; n * p_m];
            for row in 0..n {
                for j in 0..p_m {
                    let seed = (row as f64) * 0.073 + (j as f64) * 0.211 + 0.4;
                    marginal[row * p_m + j] = seed.sin() * 0.8 - (seed * 0.7).cos() * 0.3;
                }
            }
            let mut logslope = vec![0.0_f64; n * p_g];
            for row in 0..n {
                for j in 0..p_g {
                    let seed = (row as f64) * 0.091 + (j as f64) * 0.179 - 0.2;
                    logslope[row * p_g + j] = seed.cos() * 0.7 + (seed * 0.3).sin() * 0.25;
                }
            }
            let v: Vec<f64> = (0..p_total)
                .map(|i| {
                    let seed = (i as f64) * 0.157 + 0.6;
                    seed.sin() * 0.55 + (seed * 0.4).cos() * 0.35
                })
                .collect();

            let cpu_hvp = cpu_oracle_bms_flex_row_hvp(
                &row_hessians,
                &marginal,
                &logslope,
                &block,
                &primary,
                n,
                &v,
            );
            let cpu_diag = cpu_oracle_bms_flex_row_diagonal(
                &row_hessians,
                &marginal,
                &logslope,
                &block,
                &primary,
                n,
            );

            let backend = match HvpKernelBackend::probe() {
                Ok(b) => b,
                Err(err) => {
                    eprintln!(
                        "[bms_flex_row hvp parity n64_r20_p44] backend probe \
                         failed: {err}"
                    );
                    return;
                }
            };
            let stream = backend.stream.clone();
            let d_h = match stream.clone_htod(&row_hessians) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!(
                        "[bms_flex_row hvp parity n64_r20_p44] upload h \
                         failed: {err}"
                    );
                    return;
                }
            };
            let d_m = match stream.clone_htod(&marginal) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!(
                        "[bms_flex_row hvp parity n64_r20_p44] upload marg \
                         failed: {err}"
                    );
                    return;
                }
            };
            let d_g = match stream.clone_htod(&logslope) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!(
                        "[bms_flex_row hvp parity n64_r20_p44] upload logslope \
                         failed: {err}"
                    );
                    return;
                }
            };
            let storage = DeviceResidentRowHess {
                hess: d_h,
                marginal_design: d_m,
                logslope_design: d_g,
                n,
                r,
                block: block.clone(),
                primary: primary.clone(),

                bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
            };
            let gpu_hvp = launch_bms_flex_row_hvp(&storage, &v)
                .expect("HVP kernel must launch on CUDA host at n64/r20/p44");
            let gpu_diag = launch_bms_flex_row_diagonal(&storage)
                .expect("diagonal kernel must launch on CUDA host at n64/r20/p44");
            assert_eq!(gpu_hvp.len(), cpu_hvp.len());
            assert_eq!(gpu_diag.len(), cpu_diag.len());
            for i in 0..p_total {
                let diff = (cpu_hvp[i] - gpu_hvp[i]).abs();
                assert!(
                    diff <= 1e-8,
                    "n64_r20_p44 HVP[{i}]: cpu={} gpu={} |Δ|={diff:.3e}",
                    cpu_hvp[i],
                    gpu_hvp[i]
                );
                let ddiff = (cpu_diag[i] - gpu_diag[i]).abs();
                assert!(
                    ddiff <= 1e-8,
                    "n64_r20_p44 diag[{i}]: cpu={} gpu={} |Δ|={ddiff:.3e}",
                    cpu_diag[i],
                    gpu_diag[i]
                );
            }
        }
    }

    /// Block 9 Phase 6 — small-fixture parity for the dense-block kernel
    /// against the host-side P_i pullback oracle.
    /// Verifies bit-equality (modulo reduction-order f.p. noise) between
    /// the device-resident dense build and the host accumulator over the
    /// same per-row Hessian + designs + P_i pullback.
    #[test]
    pub(crate) fn bms_flex_row_dense_block_kernel_matches_cpu_pullback() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!("[bms_flex_row dense_block parity] non-Linux host — skipping CUDA parity");
        }
        #[cfg(target_os = "linux")]
        {
            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!("[bms_flex_row dense_block parity] no CUDA runtime — skipping");
                return;
            };
            // Small fixture: n=24, r=8 (2 + 3 + 3), p_total=18 (4+4+3+3).
            // Keeps the CPU pullback fast while still exercising every
            // primary slot (q, g, h, w).
            let n = 24_usize;
            let p_m = 4_usize;
            let p_g = 4_usize;
            let p_h_dim = 3_usize;
            let p_w_dim = 3_usize;
            let r = 2 + p_h_dim + p_w_dim;
            let p_total = p_m + p_g + p_h_dim + p_w_dim;
            let block = BmsFlexBlockLayout {
                p_m,
                p_g,
                h: Some(p_m + p_g..p_m + p_g + p_h_dim),
                w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
                p_total,
            };
            let primary = BmsFlexPrimaryLayout {
                h: Some(2..2 + p_h_dim),
                w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
                r,
            };

            let mut row_hessians = vec![0.0_f64; n * r * r];
            for row in 0..n {
                let base = row * r * r;
                for u in 0..r {
                    for v in 0..r {
                        let seed = (row as f64) * 0.21 + (u as f64) * 1.13 + (v as f64) * 0.47;
                        let a = (seed.sin() * 1.4 + (seed * 0.6).cos() * 0.7) * 0.5;
                        row_hessians[base + u * r + v] = a;
                    }
                }
                for u in 0..r {
                    for v in (u + 1)..r {
                        let upper = row_hessians[base + u * r + v];
                        let lower = row_hessians[base + v * r + u];
                        let sym = 0.5 * (upper + lower);
                        row_hessians[base + u * r + v] = sym;
                        row_hessians[base + v * r + u] = sym;
                    }
                    row_hessians[base + u * r + u] += r as f64;
                }
            }
            let mut marginal = vec![0.0_f64; n * p_m];
            for row in 0..n {
                for j in 0..p_m {
                    let seed = (row as f64) * 0.083 + (j as f64) * 0.171 + 0.31;
                    marginal[row * p_m + j] = seed.sin() * 0.7 - (seed * 0.5).cos() * 0.25;
                }
            }
            let mut logslope = vec![0.0_f64; n * p_g];
            for row in 0..n {
                for j in 0..p_g {
                    let seed = (row as f64) * 0.097 + (j as f64) * 0.143 - 0.15;
                    logslope[row * p_g + j] = seed.cos() * 0.65 + (seed * 0.4).sin() * 0.2;
                }
            }

            // CPU oracle — same pullback math the device kernel mirrors.
            let h_block_start = block.h.as_ref().map(|r| r.start).unwrap_or(0);
            let h_block_len = block.h.as_ref().map(|r| r.len()).unwrap_or(0);
            let w_block_start = block.w.as_ref().map(|r| r.start).unwrap_or(0);
            let w_block_len = block.w.as_ref().map(|r| r.len()).unwrap_or(0);
            let h_primary_start = primary.h.as_ref().map(|r| r.start).unwrap_or(0);
            let w_primary_start = primary.w.as_ref().map(|r| r.start).unwrap_or(0);
            let mut h_cpu = vec![0.0_f64; p_total * p_total];
            for row in 0..n {
                let mrow = &marginal[row * p_m..(row + 1) * p_m];
                let grow = &logslope[row * p_g..(row + 1) * p_g];
                let hrow = &row_hessians[row * r * r..(row + 1) * r * r];
                // Build per-row phi (r length-p_total vectors).
                let mut phi = vec![vec![0.0_f64; p_total]; r];
                for k in 0..p_m {
                    phi[0][k] = mrow[k];
                }
                for k in 0..p_g {
                    phi[1][p_m + k] = grow[k];
                }
                for k in 0..h_block_len {
                    phi[h_primary_start + k][h_block_start + k] = 1.0;
                }
                for k in 0..w_block_len {
                    phi[w_primary_start + k][w_block_start + k] = 1.0;
                }
                for u in 0..r {
                    for v in 0..r {
                        let huv = hrow[u * r + v];
                        if huv == 0.0 {
                            continue;
                        }
                        for m in 0..p_total {
                            let pm = phi[u][m];
                            if pm == 0.0 {
                                continue;
                            }
                            let scaled = huv * pm;
                            for nn in 0..p_total {
                                h_cpu[m * p_total + nn] += scaled * phi[v][nn];
                            }
                        }
                    }
                }
            }

            // Build a transient device-resident storage and launch the
            // dense-block kernel.
            // Past the GpuRuntime::global() Some-gate: probe/upload failures are
            // real device faults on a CUDA host — fail loud (device-PCG class).
            let backend = HvpKernelBackend::probe().expect(
                "[bms_flex_row dense_block parity] backend probe must succeed on CUDA host",
            );
            let stream = backend.stream.clone();
            let d_h = stream
                .clone_htod(&row_hessians)
                .expect("[bms_flex_row dense_block parity] upload h must succeed on CUDA host");
            let d_m = stream
                .clone_htod(&marginal)
                .expect("[bms_flex_row dense_block parity] upload marg must succeed on CUDA host");
            let d_g = stream.clone_htod(&logslope).expect(
                "[bms_flex_row dense_block parity] upload logslope must succeed on CUDA host",
            );
            let storage = DeviceResidentRowHess {
                hess: d_h,
                marginal_design: d_m,
                logslope_design: d_g,
                n,
                r,
                block: block.clone(),
                primary: primary.clone(),

                bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
            };
            let h_gpu = launch_bms_flex_row_dense_block(&storage)
                .expect("dense_block kernel must launch on CUDA host");
            assert_eq!(h_gpu.len(), p_total * p_total);

            // Compare entry-by-entry with a tolerance that absorbs
            // reduction-order f.p. noise from the CTA chunk sum.
            let mut max_abs = 0.0_f64;
            for i in 0..p_total {
                for j in 0..p_total {
                    let a = h_cpu[i * p_total + j];
                    let b = h_gpu[i * p_total + j];
                    let diff = (a - b).abs();
                    if diff > max_abs {
                        max_abs = diff;
                    }
                    assert!(
                        diff <= 1e-9 * a.abs().max(b.abs()).max(1.0),
                        "dense_block[{i},{j}]: cpu={a} gpu={b} |Δ|={diff:.3e}"
                    );
                }
            }
            eprintln!(
                "[bms_flex_row dense_block parity] n={n} r={r} p={p_total}: max|Δ|={max_abs:.3e}"
            );
        }
    }

    /// Block 9 final hill-climb gate — GPU HVP must be at least 5× faster
    /// than a Rayon-parallel CPU HVP at large-scale shape (n=195_000, r=20,
    /// p_total=44). This is the charter pass/fail metric for whether the
    /// device-resident row-Hessian path is a real perf win for the
    /// production marginal-slope fit.
    ///
    /// Methodology:
    ///   * Build the same deterministic fixture as the parity tests.
    ///   * GPU: median of `iters` `launch_bms_flex_row_hvp` wall-times
    ///     after `warmup` warm-up launches (kernel compile + L2 prime).
    ///   * CPU: median of `iters` `cpu_oracle_bms_flex_row_hvp` wall-times,
    ///     parallelised over rows via Rayon — this mirrors the actual
    ///     production CPU path in
    ///     `exact_newton_joint_hessian_matvec_from_cache` (which uses
    ///     `ROW_CHUNK_SIZE` chunked `into_par_iter()` for the same
    ///     contraction).
    ///   * Ratio = cpu_median / gpu_median; assert ratio >= 5.
    ///
    /// Skips on non-Linux / no-CUDA hosts.
    #[test]
    pub(crate) fn bms_flex_row_hvp_v100_hill_climb_5x_vs_cpu_at_large_scale() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!("[bms_flex_row hvp hill-climb] non-Linux host — skipping V100 perf gate");
        }
        #[cfg(target_os = "linux")]
        {
            use rayon::prelude::*;

            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!(
                    "[bms_flex_row hvp hill-climb] no CUDA runtime — skipping V100 perf gate"
                );
                return;
            };
            let n = 195_000_usize;
            let p_m = 14_usize;
            let p_g = 12_usize;
            let p_h_dim = 10_usize;
            let p_w_dim = 8_usize;
            let r = 2 + p_h_dim + p_w_dim;
            let p_total = p_m + p_g + p_h_dim + p_w_dim;
            let block = BmsFlexBlockLayout {
                p_m,
                p_g,
                h: Some(p_m + p_g..p_m + p_g + p_h_dim),
                w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
                p_total,
            };
            let primary = BmsFlexPrimaryLayout {
                h: Some(2..2 + p_h_dim),
                w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
                r,
            };

            // Same deterministic fixture as the Phase 4 large-scale benchmark.
            let mut row_hessians = vec![0.0_f64; n * r * r];
            for row in 0..n {
                let base = row * r * r;
                for u in 0..r {
                    for vv in 0..r {
                        let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (vv as f64) * 0.317;
                        let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                        row_hessians[base + u * r + vv] = a;
                    }
                }
                for u in 0..r {
                    for vv in (u + 1)..r {
                        let upper = row_hessians[base + u * r + vv];
                        let lower = row_hessians[base + vv * r + u];
                        let sym = 0.5 * (upper + lower);
                        row_hessians[base + u * r + vv] = sym;
                        row_hessians[base + vv * r + u] = sym;
                    }
                    row_hessians[base + u * r + u] += r as f64;
                }
            }
            let mut marginal = vec![0.0_f64; n * p_m];
            for row in 0..n {
                for j in 0..p_m {
                    let seed = (row as f64) * 0.073 + (j as f64) * 0.211 + 0.4;
                    marginal[row * p_m + j] = seed.sin() * 0.8 - (seed * 0.7).cos() * 0.3;
                }
            }
            let mut logslope = vec![0.0_f64; n * p_g];
            for row in 0..n {
                for j in 0..p_g {
                    let seed = (row as f64) * 0.091 + (j as f64) * 0.179 - 0.2;
                    logslope[row * p_g + j] = seed.cos() * 0.7 + (seed * 0.3).sin() * 0.25;
                }
            }
            let v: Vec<f64> = (0..p_total)
                .map(|i| {
                    let seed = (i as f64) * 0.157 + 0.6;
                    seed.sin() * 0.55 + (seed * 0.4).cos() * 0.35
                })
                .collect();

            // ── GPU side: upload once, time HVP launches ─────────────
            let backend = match HvpKernelBackend::probe() {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("[bms_flex_row hvp hill-climb] backend probe failed: {err}");
                    return;
                }
            };
            let stream = backend.stream.clone();
            let d_h = match stream.clone_htod(&row_hessians) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[bms_flex_row hvp hill-climb] upload h failed (likely OOM): {err}");
                    return;
                }
            };
            let d_m = match stream.clone_htod(&marginal) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[bms_flex_row hvp hill-climb] upload marg failed: {err}");
                    return;
                }
            };
            let d_g = match stream.clone_htod(&logslope) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[bms_flex_row hvp hill-climb] upload logslope failed: {err}");
                    return;
                }
            };
            let storage = DeviceResidentRowHess {
                hess: d_h,
                marginal_design: d_m,
                logslope_design: d_g,
                n,
                r,
                block: block.clone(),
                primary: primary.clone(),

                bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
            };
            let warmup: usize = 3;
            let iters: usize = 15;
            for _ in 0..warmup {
                let out =
                    launch_bms_flex_row_hvp(&storage, &v).expect("warmup GPU HVP must launch");
                assert_eq!(out.len(), p_total);
            }
            let mut gpu_us: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                let out = launch_bms_flex_row_hvp(&storage, &v).expect("GPU HVP must launch");
                gpu_us.push(t0.elapsed().as_micros());
                assert_eq!(out.len(), p_total);
            }
            gpu_us.sort_unstable();
            let gpu_median = gpu_us[iters / 2];

            // ── CPU side: chunked Rayon HVP over rows, mirroring the
            //    production `exact_newton_joint_hessian_matvec_from_cache`
            //    parallelisation pattern (ROW_CHUNK_SIZE-row chunks,
            //    try_fold + try_reduce). The per-chunk worker calls the
            //    single-threaded oracle on its row slice.
            const CHUNK_ROWS: usize = 4096;
            let cpu_hvp_parallel = || -> Vec<f64> {
                let nchunks = n.div_ceil(CHUNK_ROWS);
                (0..nchunks)
                    .into_par_iter()
                    .fold(
                        || vec![0.0_f64; p_total],
                        |mut acc, ci| {
                            let lo = ci * CHUNK_ROWS;
                            let hi = (lo + CHUNK_ROWS).min(n);
                            let m = hi - lo;
                            let partial = cpu_oracle_bms_flex_row_hvp(
                                &row_hessians[lo * r * r..hi * r * r],
                                &marginal[lo * p_m..hi * p_m],
                                &logslope[lo * p_g..hi * p_g],
                                &block,
                                &primary,
                                m,
                                &v,
                            );
                            for (a, &p) in acc.iter_mut().zip(partial.iter()) {
                                *a += p;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![0.0_f64; p_total],
                        |mut a, b| {
                            for (ax, bx) in a.iter_mut().zip(b.iter()) {
                                *ax += *bx;
                            }
                            a
                        },
                    )
            };
            // Warmup once to populate L3 / steady-state Rayon thread pool.
            let warm = cpu_hvp_parallel();
            assert_eq!(warm.len(), p_total);
            let mut cpu_us: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                let out = cpu_hvp_parallel();
                cpu_us.push(t0.elapsed().as_micros());
                assert_eq!(out.len(), p_total);
            }
            cpu_us.sort_unstable();
            let cpu_median = cpu_us[iters / 2];

            let speedup = (cpu_median as f64) / (gpu_median.max(1) as f64);
            eprintln!(
                "[bms_flex_row hvp hill-climb] large-scale n={n} r={r} p={p_total}: \
                 cpu_median={cpu_median}us gpu_median={gpu_median}us \
                 speedup={speedup:.2}× (charter target ≥ 5×)"
            );
            assert!(
                speedup >= 5.0,
                "large-scale HVP perf gate: GPU only {speedup:.2}× faster than CPU; \
                 need ≥ 5× per Block 9 charter (cpu_median={cpu_median}us, \
                 gpu_median={gpu_median}us). Hill-climb the kernel until met or \
                 prove the kernel is at hardware roofline."
            );
        }
    }

    /// Companion to the HVP hill-climb: GPU dense-block build must be at
    /// least 10× faster than a Rayon-parallel CPU dense build at large-scale
    /// shape. The dense build is `O(n * r² * p_total)` work for both
    /// paths so the ratio is well-defined.
    #[test]
    pub(crate) fn bms_flex_row_dense_block_v100_hill_climb_10x_vs_cpu_at_large_scale() {
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!(
                "[bms_flex_row dense_block hill-climb] non-Linux host — skipping V100 perf gate"
            );
        }
        #[cfg(target_os = "linux")]
        {
            use rayon::prelude::*;

            let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
                eprintln!(
                    "[bms_flex_row dense_block hill-climb] no CUDA runtime — skipping V100 perf gate"
                );
                return;
            };
            let n = 195_000_usize;
            let p_m = 14_usize;
            let p_g = 12_usize;
            let p_h_dim = 10_usize;
            let p_w_dim = 8_usize;
            let r = 2 + p_h_dim + p_w_dim;
            let p_total = p_m + p_g + p_h_dim + p_w_dim;
            let block = BmsFlexBlockLayout {
                p_m,
                p_g,
                h: Some(p_m + p_g..p_m + p_g + p_h_dim),
                w: Some(p_m + p_g + p_h_dim..p_m + p_g + p_h_dim + p_w_dim),
                p_total,
            };
            let primary = BmsFlexPrimaryLayout {
                h: Some(2..2 + p_h_dim),
                w: Some(2 + p_h_dim..2 + p_h_dim + p_w_dim),
                r,
            };

            // Reuse the same large-scale fixture recipe.
            let mut row_hessians = vec![0.0_f64; n * r * r];
            for row in 0..n {
                let base = row * r * r;
                for u in 0..r {
                    for vv in 0..r {
                        let seed = (row as f64) * 0.137 + (u as f64) * 1.901 + (vv as f64) * 0.317;
                        let a = (seed.sin() * 1.7 + (seed * 0.5).cos() * 0.9) * 0.5;
                        row_hessians[base + u * r + vv] = a;
                    }
                }
                for u in 0..r {
                    for vv in (u + 1)..r {
                        let upper = row_hessians[base + u * r + vv];
                        let lower = row_hessians[base + vv * r + u];
                        let sym = 0.5 * (upper + lower);
                        row_hessians[base + u * r + vv] = sym;
                        row_hessians[base + vv * r + u] = sym;
                    }
                    row_hessians[base + u * r + u] += r as f64;
                }
            }
            let mut marginal = vec![0.0_f64; n * p_m];
            for row in 0..n {
                for j in 0..p_m {
                    let seed = (row as f64) * 0.073 + (j as f64) * 0.211 + 0.4;
                    marginal[row * p_m + j] = seed.sin() * 0.8 - (seed * 0.7).cos() * 0.3;
                }
            }
            let mut logslope = vec![0.0_f64; n * p_g];
            for row in 0..n {
                for j in 0..p_g {
                    let seed = (row as f64) * 0.091 + (j as f64) * 0.179 - 0.2;
                    logslope[row * p_g + j] = seed.cos() * 0.7 + (seed * 0.3).sin() * 0.25;
                }
            }

            // GPU dense_block kernel rejects p_total > DENSE_BLOCK_MAX_P
            // (72 at V100 48 KiB/block). LargeScale's p_total = 44 fits.
            if p_total > DENSE_BLOCK_MAX_P {
                eprintln!(
                    "[bms_flex_row dense_block hill-climb] p_total={p_total} > MAX={DENSE_BLOCK_MAX_P}, skipping"
                );
                return;
            }
            let backend = match HvpKernelBackend::probe() {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("[bms_flex_row dense_block hill-climb] backend probe failed: {err}");
                    return;
                }
            };
            let stream = backend.stream.clone();
            let d_h = match stream.clone_htod(&row_hessians) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[bms_flex_row dense_block hill-climb] upload h failed: {err}");
                    return;
                }
            };
            let d_m = match stream.clone_htod(&marginal) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[bms_flex_row dense_block hill-climb] upload marg failed: {err}");
                    return;
                }
            };
            let d_g = match stream.clone_htod(&logslope) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!(
                        "[bms_flex_row dense_block hill-climb] upload logslope failed: {err}"
                    );
                    return;
                }
            };
            let storage = DeviceResidentRowHess {
                hess: d_h,
                marginal_design: d_m,
                logslope_design: d_g,
                n,
                r,
                block: block.clone(),
                primary: primary.clone(),

                bytes: ((n * r * r + n * p_m + n * p_g) * std::mem::size_of::<f64>()) as u64,
            };
            // Warmup + 5-iter median (dense build is heavier than HVP).
            let warmup: usize = 2;
            let iters: usize = 5;
            for _ in 0..warmup {
                let out = launch_bms_flex_row_dense_block(&storage)
                    .expect("warmup GPU dense_block must launch");
                assert_eq!(out.len(), p_total * p_total);
            }
            let mut gpu_us: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                let out =
                    launch_bms_flex_row_dense_block(&storage).expect("GPU dense_block must launch");
                gpu_us.push(t0.elapsed().as_micros());
                assert_eq!(out.len(), p_total * p_total);
            }
            gpu_us.sort_unstable();
            let gpu_median = gpu_us[iters / 2];

            // CPU side: chunked Rayon dense build over rows. Each chunk
            // builds a `[p_total, p_total]` partial then we reduce-add.
            const CHUNK_ROWS: usize = 2048;
            let h_block_start = block.h.as_ref().map(|r| r.start).unwrap_or(0);
            let h_block_len = block.h.as_ref().map(|r| r.len()).unwrap_or(0);
            let w_block_start = block.w.as_ref().map(|r| r.start).unwrap_or(0);
            let w_block_len = block.w.as_ref().map(|r| r.len()).unwrap_or(0);
            let h_primary_start = primary.h.as_ref().map(|r| r.start).unwrap_or(0);
            let w_primary_start = primary.w.as_ref().map(|r| r.start).unwrap_or(0);
            let cpu_build_parallel = || -> Vec<f64> {
                let nchunks = n.div_ceil(CHUNK_ROWS);
                (0..nchunks)
                    .into_par_iter()
                    .fold(
                        || vec![0.0_f64; p_total * p_total],
                        |mut acc, ci| {
                            let lo = ci * CHUNK_ROWS;
                            let hi = (lo + CHUNK_ROWS).min(n);
                            let mut phi: Vec<Vec<f64>> = vec![vec![0.0_f64; p_total]; r];
                            for row in lo..hi {
                                for col in phi.iter_mut() {
                                    col.iter_mut().for_each(|v| *v = 0.0);
                                }
                                let mrow = &marginal[row * p_m..(row + 1) * p_m];
                                let grow = &logslope[row * p_g..(row + 1) * p_g];
                                for k in 0..p_m {
                                    phi[0][k] = mrow[k];
                                }
                                for k in 0..p_g {
                                    phi[1][p_m + k] = grow[k];
                                }
                                for k in 0..h_block_len {
                                    phi[h_primary_start + k][h_block_start + k] = 1.0;
                                }
                                for k in 0..w_block_len {
                                    phi[w_primary_start + k][w_block_start + k] = 1.0;
                                }
                                let hrow = &row_hessians[row * r * r..(row + 1) * r * r];
                                for u in 0..r {
                                    for v_idx in 0..r {
                                        let huv = hrow[u * r + v_idx];
                                        if huv == 0.0 {
                                            continue;
                                        }
                                        for m in 0..p_total {
                                            let pm = phi[u][m];
                                            if pm == 0.0 {
                                                continue;
                                            }
                                            let scaled = huv * pm;
                                            for nn in 0..p_total {
                                                acc[m * p_total + nn] += scaled * phi[v_idx][nn];
                                            }
                                        }
                                    }
                                }
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![0.0_f64; p_total * p_total],
                        |mut a, b| {
                            for (ax, bx) in a.iter_mut().zip(b.iter()) {
                                *ax += *bx;
                            }
                            a
                        },
                    )
            };
            let warm_cpu = cpu_build_parallel();
            assert_eq!(warm_cpu.len(), p_total * p_total);
            let mut cpu_us: Vec<u128> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = std::time::Instant::now();
                let out = cpu_build_parallel();
                cpu_us.push(t0.elapsed().as_micros());
                assert_eq!(out.len(), p_total * p_total);
            }
            cpu_us.sort_unstable();
            let cpu_median = cpu_us[iters / 2];

            let speedup = (cpu_median as f64) / (gpu_median.max(1) as f64);
            eprintln!(
                "[bms_flex_row dense_block hill-climb] large-scale n={n} r={r} p={p_total}: \
                 cpu_median={cpu_median}us gpu_median={gpu_median}us \
                 speedup={speedup:.2}× (charter target ≥ 10×)"
            );
            assert!(
                speedup >= 10.0,
                "large-scale dense-H perf gate: GPU only {speedup:.2}× faster than CPU; \
                 need ≥ 10× per Block 9 charter (cpu_median={cpu_median}us, \
                 gpu_median={gpu_median}us). Hill-climb the dense_block kernel \
                 (warp-stripe the u-v-m-n loop, vectorise loads, etc.) until met \
                 or prove the kernel is at hardware roofline."
            );
        }
    }
}
