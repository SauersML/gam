//! Stage 2 of the BMS FLEX row kernel — per-row math that turns per-cell
//! derivative moments (built by Stage 1 in `src/gpu/cubic_cell/mod.rs`) into a
//! row gradient and row-primary `r × r` Hessian.
//!
//! Math (mirrors the CPU reference
//! `BernoulliMarginalSlope::lower_bms_flex_row_order2_from_parts` in
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
//! Probit Mills (stable; uses the shared curvature primitive from
//! `numerics_device::PROBIT_NUMERICS_CU`):
//!
//! ```text
//!     s = 2y − 1 ;  m = s · e_obs
//!     [log_cdf, λ, C] = log_ndtr_mills_curvature(m),  C = −d²logΦ(m)/dm²
//!     A = −w · s · λ
//!     B =  w · C
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
//! `blockDim.x = 32` threads. The already-required output buffers are the
//! width-general scratch authority: `out_grad[row]` evolves `F_u → a_u → g`,
//! and `out_hess[row]` evolves `F_uv → a_uv → H`. One additional checked
//! `[n, r]` buffer holds `F_au`. Only the fixed 32-thread scalar reduction is
//! shared, so primary width has no shared-memory or thread-stack ceiling.

#[cfg(target_os = "linux")]
use std::sync::OnceLock;

use gam_gpu::gpu_error::GpuError;

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

#[cfg(target_os = "linux")]
use super::super::flex_row_program::{
    BmsFlexCalibrationOrder2Phase, BmsFlexRowOrder2FinalizerPhase, BmsFlexRowProgram,
};

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
        e_obs,
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

fn checked_shape_len(context: &str, dimensions: &[usize]) -> Result<usize, GpuError> {
    dimensions
        .iter()
        .copied()
        .try_fold(1_usize, |product, dimension| {
            product
                .checked_mul(dimension)
                .ok_or_else(|| GpuError::DriverCallFailed {
                    reason: format!(
                        "bms_flex_row {context}: shape product overflow for dimensions {dimensions:?}"
                    ),
                })
        })
}

impl<'a> BmsFlexRowKernelInputs<'a> {
    /// Sanity-check every shape the kernel relies on. This is the only place
    /// length errors are surfaced — the device kernel assumes valid layout.
    pub(crate) fn validate(&self) -> Result<(), GpuError> {
        if self.n_rows == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "bms_flex_row inputs: n_rows must be > 0".to_string(),
            });
        }
        if self.r == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "bms_flex_row inputs: r must be > 0".to_string(),
            });
        }
        let decomposed_r = 2_usize
            .checked_add(self.p_h)
            .and_then(|value| value.checked_add(self.p_w))
            .ok_or_else(|| GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row inputs: primary decomposition overflow for p_h={} p_w={}",
                    self.p_h, self.p_w
                ),
            })?;
        if self.r != decomposed_r {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row inputs: r={} must equal 2 + p_h({}) + p_w({}) = {}",
                    self.r, self.p_h, self.p_w, decomposed_r
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
        check_len("e_obs", self.e_obs.len(), n)?;
        check_len("chi_obs", self.chi_obs.len(), n)?;
        check_len("xi_obs", self.xi_obs.len(), n)?;
        let nr = checked_shape_len("validate [n,r]", &[n, self.r])?;
        let nrr = checked_shape_len("validate [n,r,r]", &[n, self.r, self.r])?;
        check_len("rho_u", self.rho_u.len(), nr)?;
        check_len("tau_u", self.tau_u.len(), nr)?;
        check_len("r_uv", self.r_uv.len(), nrr)?;
        let offsets_len = n.checked_add(1).ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row inputs: n_rows={n} cannot form n+1 offsets"),
        })?;
        check_len("cell_offsets", self.cell_offsets.len(), offsets_len)?;
        let total_cells_u32 = self.cell_offsets[n];
        let total_cells = total_cells_u32 as usize;
        check_len("cell_c0", self.cell_c0.len(), total_cells)?;
        check_len("cell_c1", self.cell_c1.len(), total_cells)?;
        check_len("cell_c2", self.cell_c2.len(), total_cells)?;
        check_len("cell_c3", self.cell_c3.len(), total_cells)?;
        let cells_coeff4 = checked_shape_len("validate cell coeff4", &[total_cells, COEFF4])?;
        check_len("cell_a", self.cell_a.len(), cells_coeff4)?;
        check_len("cell_aa", self.cell_aa.len(), cells_coeff4)?;
        check_len(
            "cell_r",
            self.cell_r.len(),
            checked_shape_len(
                "validate cell_r",
                &[total_cells, self.r.saturating_sub(1), COEFF4],
            )?,
        )?;
        check_len(
            "cell_ar",
            self.cell_ar.len(),
            checked_shape_len(
                "validate cell_ar",
                &[total_cells, self.r.saturating_sub(1), COEFF4],
            )?,
        )?;
        check_len("cell_sbb", self.cell_sbb.len(), cells_coeff4)?;
        check_len(
            "cell_sbh",
            self.cell_sbh.len(),
            checked_shape_len("validate cell_sbh", &[total_cells, self.p_h, COEFF4])?,
        )?;
        check_len(
            "cell_sbw",
            self.cell_sbw.len(),
            checked_shape_len("validate cell_sbw", &[total_cells, self.p_w, COEFF4])?,
        )?;
        check_len(
            "cell_moments",
            self.cell_moments.len(),
            checked_shape_len("validate cell_moments", &[total_cells, MOMENT_STRIDE])?,
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

/// Non-semantic CUDA scaffolding for the generated row kernel. One CUDA block
/// per row; the generated launch width parallelises the per-cell sums into
/// shared-memory scratch. The calibration and finalizer markers are replaced
/// by interpreting [`BmsFlexRowProgram`]'s typed node streams.
///
/// Shared probit numerics (`erfcx_nonnegative`, `log_ndtr`,
/// `log_ndtr_and_mills`) are provided by
/// `numerics_device::PROBIT_NUMERICS_CU`, which is prepended before
/// passing to `cudarc::nvrtc::compile_ptx`.
///
#[cfg(target_os = "linux")]
const CUDA_ROW_KERNEL_TEMPLATE: &str = r#"
// One block per row. threadIdx.x parallelises per-cell sums.
// Semantic calibration/finalization visits are generated from BmsFlexRowProgram.

#define INV_TWO_PI     0.15915494309189535
#define BMS_FLEX_ROW_THREADS /*__BMS_FLEX_ROW_THREADS__*/

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
// (`F_a` non-finite or non-positive). The status channel makes the host reject
// the entire selected-GPU execution before any output can enter a cache.
extern "C" __device__ __forceinline__ void
nan_fill_outputs(int r,
                 int row,
                 double *out_neglog,
                 double *out_grad,
                 double *out_hess,
                 unsigned int *out_status) {
    double nan_value = __longlong_as_double(0x7ff8000000000000ULL);
    out_status[row] = 1U;
    out_neglog[row] = nan_value;
    size_t row_r = (size_t)row * (size_t)r;
    for (int u = 0; u < r; ++u) {
        out_grad[row_r + (size_t)u] = nan_value;
    }
    size_t rr = (size_t)r * (size_t)r;
    size_t row_rr = (size_t)row * rr;
    for (size_t idx = 0; idx < rr; ++idx) {
        out_hess[row_rr + idx] = nan_value;
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
    const double * __restrict__ row_e_obs,    // [n_rows] observed predictor VALUE
    double       * __restrict__ row_f_au,      // [n_rows, r] general-width scratch
    double       * __restrict__ out_neglog,
    double       * __restrict__ out_grad,
    double       * __restrict__ out_hess,
    unsigned int * __restrict__ out_status)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;

    // Width-general row scratch. Reuse the final output allocations in-place:
    // F_u → a_u → gradient and F_uv → a_uv → Hessian. Only F_au needs one
    // additional checked [n,r] device allocation.
    size_t row_r_base = (size_t)row * (size_t)r;
    size_t rr = (size_t)r * (size_t)r;
    size_t row_rr_base = (size_t)row * rr;
    double *F_u = out_grad + row_r_base;
    double *F_au = row_f_au + row_r_base;
    double *F_uv = out_hess + row_rr_base;
    __shared__ double reduce_a[BMS_FLEX_ROW_THREADS];
    __shared__ double reduce_b[BMS_FLEX_ROW_THREADS];
    __shared__ double F_a_shared;
    __shared__ double F_aa_shared;

    // Zero scratch.
    if (tid == 0) { F_a_shared = 0.0; F_aa_shared = 0.0; }
    for (int u = tid; u < r; u += blockDim.x) {
        F_u[u]  = 0.0;
        F_au[u] = 0.0;
    }
    for (size_t uv = (size_t)tid; uv < rr; uv += (size_t)blockDim.x) {
        F_uv[uv] = 0.0;
    }
    __syncthreads();

    // ── per-cell sweep ───────────────────────────────────────────────────
    unsigned int cell_lo = cell_offsets[row];
    unsigned int cell_hi = cell_offsets[row + 1];
    unsigned int n_cells = cell_hi - cell_lo;

    double local_Fa  = 0.0;
    double local_Faa = 0.0;

    for (unsigned int local_c = (unsigned int)tid;
         local_c < n_cells;
         local_c += (unsigned int)blockDim.x) {
        unsigned int c = cell_lo + local_c;

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
        // The argument is parenthesized because callers pass pointer
        // ARITHMETIC (`D_OF(base + offset)`): without it the expansion binds
        // as `base + offset[0]`, which NVRTC rejects ("pointer-to-object
        // type" on the integer term) — the calibration-phase emitter was the
        // first caller to hit this.
        #define D_OF(R) (INV_TWO_PI * ((R)[0]*m[0] + (R)[1]*m[1] + (R)[2]*m[2] + (R)[3]*m[3]))

        // Q(R, S) = Σ_{p,q} R_p · S_q · T_{p+q}.
        // CPU parity: the `eta_rs` folded dot in
        // `cell_second_derivative_from_moments`.
        #define Q_OF(R, S)                                                                 \
            (((R)[0]*(S)[0])*T[0] + ((R)[0]*(S)[1] + (R)[1]*(S)[0])*T[1]                   \
             + ((R)[0]*(S)[2] + (R)[1]*(S)[1] + (R)[2]*(S)[0])*T[2]                        \
             + ((R)[0]*(S)[3] + (R)[1]*(S)[2] + (R)[2]*(S)[1] + (R)[3]*(S)[0])*T[3]        \
             + ((R)[1]*(S)[3] + (R)[2]*(S)[2] + (R)[3]*(S)[1])*T[4]                        \
             + ((R)[2]*(S)[3] + (R)[3]*(S)[2])*T[5]                                        \
             + ((R)[3]*(S)[3])*T[6])

        // The typed calibration schedule below consumes these primitive
        // coefficient views through D_OF/Q_OF.
        const double *A_c  = cell_a  + (size_t)c * 4;
        const double *AA_c = cell_aa + (size_t)c * 4;
        /*__BMS_FLEX_CALIBRATION_ORDER2__*/

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
        F_uv[(size_t)v] = 0.0;
        F_uv[(size_t)v * (size_t)r] = 0.0;
    }
    F_uv[0] = -mu_2;

    // Guard: degenerate F_a ⇒ NaN-fill this row's outputs.
    if (!isfinite(F_a) || F_a <= 0.0) {
        nan_fill_outputs(r, row, out_neglog, out_grad, out_hess, out_status);
        return;
    }
    double inv_Fa = 1.0 / F_a;

    // Storage consumed by the generated dependency-ordered finalizer. Both
    // aliases overwrite their no-longer-needed calibration predecessors.
    double *a_u = F_u;
    double *a_uv = F_uv;
    double chi = row_chi[row];
    double xi  = row_xi[row];
    const double *rho = row_rho + (size_t)row * r;
    const double *tau = row_tau + (size_t)row * r;
    const double *ruv = row_ruv + row_rr_base;

    // Probit Mills.
    double y    = row_y[row];
    double w    = row_w[row];
    double s    = 2.0 * y - 1.0;
    // The "observed predictor" e_obs is the VALUE (degree-0 term) of the
    // observed jet η(a(θ), θ; z_obs) — NOT `bar_e_u[0]`, which is the u=0
    // FIRST-derivative jet (`chi·a_0 + rho_0 = dη_obs/dq`). The host packs
    // the observed value directly in `row_e_obs[row]` (see
    // `pack_bms_flex_row_kernel_inputs`, `eta_val = eval_coeff4_at(obs.coeff,
    // z_obs)`), matching the CPU family `lower_bms_flex_row_order2_from_parts`
    // which forms `signed_margin = s_y · eta_val`. #415 parity lock.
    double e_obs = row_e_obs[row];
    double m_arg = s * e_obs;
    double log_cdf, lambda, probit_curvature;
    log_ndtr_mills_curvature(m_arg, &log_cdf, &lambda, &probit_curvature);
    double A_i = -w * s * lambda;
    double B_i =  w * probit_curvature;

    out_neglog[row] = -w * log_cdf;
    /*__BMS_FLEX_ORDER2_FINALIZER__*/
    if (!isfinite(out_neglog[row])) {
        out_status[row] = 2U;
    }
    for (int u = 0; u < r; ++u) {
        if (!isfinite(out_grad[row_r_base + (size_t)u])) {
            out_status[row] = 2U;
        }
    }
    for (size_t uv = 0; uv < rr; ++uv) {
        if (!isfinite(out_hess[row_rr_base + uv])) {
            out_status[row] = 2U;
        }
    }
}
"#;

#[cfg(target_os = "linux")]
fn build_generated_row_kernel_source() -> String {
    const CALIBRATION_MARKER: &str = "        /*__BMS_FLEX_CALIBRATION_ORDER2__*/";
    const FINALIZER_MARKER: &str = "    /*__BMS_FLEX_ORDER2_FINALIZER__*/";

    let (prefix, remainder) = CUDA_ROW_KERNEL_TEMPLATE
        .split_once(CALIBRATION_MARKER)
        .expect("CUDA row template must contain the calibration marker");
    let (between, suffix) = remainder
        .split_once(FINALIZER_MARKER)
        .expect("CUDA row template must contain the finalizer marker");
    let mut source = String::with_capacity(CUDA_ROW_KERNEL_TEMPLATE.len() + 16_000);
    source.push_str(prefix);

    BmsFlexRowProgram::try_for_each_calibration_order2_phase(
        true,
        |phase| -> Result<(), std::convert::Infallible> {
            source.push_str(&format!(
                "        // canonical calibration phase: {phase:?}\n"
            ));
            match phase {
                BmsFlexCalibrationOrder2Phase::InterceptFirst => {
                    source.push_str("        local_Fa += D_OF(A_c);\n");
                }
                BmsFlexCalibrationOrder2Phase::InterceptSecond => {
                    source.push_str("        local_Faa += D_OF(AA_c) - Q_OF(A_c, A_c);\n");
                }
                BmsFlexCalibrationOrder2Phase::PrimaryFirstAndInterceptSecond => {
                    source.push_str(
                        r#"        for (int u = 1; u < r; ++u) {
            const double *R_u = cell_r + ((size_t)c * (size_t)(r - 1) + (size_t)(u - 1)) * 4;
            const double *AR_u = cell_ar + ((size_t)c * (size_t)(r - 1) + (size_t)(u - 1)) * 4;
            atomic_add_f64(&F_u[u], D_OF(R_u));
            atomic_add_f64(&F_au[u], D_OF(AR_u) - Q_OF(A_c, R_u));
        }
"#,
                    );
                }
                BmsFlexCalibrationOrder2Phase::PrimaryPairSecond => {
                    source.push_str(
                        r#"        for (int u = 1; u < r; ++u) {
            const double *R_u = cell_r + ((size_t)c * (size_t)(r - 1) + (size_t)(u - 1)) * 4;
            for (int v = u; v < r; ++v) {
                const double *R_v = cell_r + ((size_t)c * (size_t)(r - 1) + (size_t)(v - 1)) * 4;
                double explicit_second = 0.0;
                if (u == 1 && v == 1) {
                    explicit_second = D_OF(cell_sbb + (size_t)c * 4);
                } else if (u == 1 && v < 2 + p_h) {
                    int j = v - 2;
                    explicit_second = D_OF(cell_sbh + ((size_t)c * (size_t)p_h + (size_t)j) * 4);
                } else if (u == 1) {
                    int l = v - (2 + p_h);
                    explicit_second = D_OF(cell_sbw + ((size_t)c * (size_t)p_w + (size_t)l) * 4);
                }
                atomic_add_f64(&F_uv[(size_t)u * (size_t)r + (size_t)v], explicit_second - Q_OF(R_u, R_v));
            }
        }
"#,
                    );
                }
            }
            Ok(())
        },
    )
    .expect("the infallible calibration phase emitter cannot fail");

    source.push_str(between);
    BmsFlexRowProgram::try_for_each_order2_finalizer_phase(
        true,
        |phase| -> Result<(), std::convert::Infallible> {
            source.push_str(&format!("    // canonical finalizer phase: {phase:?}\n"));
            match phase {
                BmsFlexRowOrder2FinalizerPhase::ImplicitFirst => {
                    source.push_str(
                        r#"    for (int u = 0; u < r; ++u) {
        a_u[u] = -F_u[u] * inv_Fa;
    }
"#,
                    );
                }
                BmsFlexRowOrder2FinalizerPhase::ImplicitFirstComplete => {
                    source.push_str("    // Canonical implicit-first stage complete.\n");
                }
                BmsFlexRowOrder2FinalizerPhase::ImplicitSecond => {
                    source.push_str(
                        r#"    for (int u = 0; u < r; ++u) {
        for (int v = u; v < r; ++v) {
            size_t uv = (size_t)u * (size_t)r + (size_t)v;
            size_t vu = (size_t)v * (size_t)r + (size_t)u;
            double term = F_uv[uv]
                        + F_au[v] * a_u[u]
                        + F_au[u] * a_u[v]
                        + F_aa * a_u[u] * a_u[v];
            double value = -term * inv_Fa;
            a_uv[uv] = value;
            a_uv[vu] = value;
        }
    }
"#,
                    );
                }
                BmsFlexRowOrder2FinalizerPhase::ObservedFirst => {
                    source.push_str("    // Observed first derivatives are derived on demand.\n");
                }
                BmsFlexRowOrder2FinalizerPhase::ObservedScoreSensitivity => {
                    source.push_str(
                        "    // Score sensitivity has no Stage-2 device output channel.\n",
                    );
                }
                BmsFlexRowOrder2FinalizerPhase::ObservedSecond => {
                    source.push_str(
                        r#"    for (int u = 0; u < r; ++u) {
        for (int v = u; v < r; ++v) {
            size_t uv = (size_t)u * (size_t)r + (size_t)v;
            size_t vu = (size_t)v * (size_t)r + (size_t)u;
            double bar_e_u = chi * a_u[u] + rho[u];
            double bar_e_v = chi * a_u[v] + rho[v];
            double observed_second = chi * a_uv[uv]
                                   + xi * a_u[u] * a_u[v]
                                   + tau[u] * a_u[v]
                                   + a_u[u] * tau[v]
                                   + ruv[uv];
            double hessian_value =
                B_i * bar_e_u * bar_e_v + A_i * observed_second;
            out_hess[row_rr_base + uv] = hessian_value;
            out_hess[row_rr_base + vu] = hessian_value;
        }
    }
"#,
                    );
                }
                BmsFlexRowOrder2FinalizerPhase::NegLogFirst => {
                    source.push_str(
                        r#"    for (int u = 0; u < r; ++u) {
        double bar_e_u = chi * a_u[u] + rho[u];
        out_grad[row_r_base + (size_t)u] = A_i * bar_e_u;
    }
"#,
                    );
                }
            }
            Ok(())
        },
    )
    .expect("the infallible finalizer phase emitter cannot fail");
    source.push_str(suffix);
    source.replace(
        "/*__BMS_FLEX_ROW_THREADS__*/",
        &ROW_KERNEL_THREADS.to_string(),
    )
}

#[cfg(target_os = "linux")]
pub(crate) fn generated_row_kernel_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(build_generated_row_kernel_source)
}

// Force `s_f` to be considered used at the Rust level even though Stage 2 of
// the kernel doesn't consume it on-device (the host has already baked the
// probit frailty scale into the per-cell cubic coefficients). The dispatcher
// validates that the host-baked cubic coefficients came from a finite,
// positive frailty scale; reading it here also avoids a `let _` silencer.
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
                        generated_row_kernel_source(),
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
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(&row_kernel_source).map_err(
                        |err| GpuError::DriverCallFailed {
                            reason: format!("bms_flex_row NVRTC compile failed: {err}"),
                        },
                    )?;
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
    let d_e_obs = upload_f64(inputs.e_obs, "e_obs")?;

    let n = inputs.n_rows;
    let r = inputs.r;
    let nr = checked_shape_len("launch [n,r]", &[n, r])?;
    let nrr = checked_shape_len("launch [n,r,r]", &[n, r, r])?;
    let mut d_neglog = stream
        .alloc_zeros::<f64>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc neglog: {err}"),
        })?;
    let mut d_grad = stream
        .alloc_zeros::<f64>(nr)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc grad: {err}"),
        })?;
    let mut d_hess = stream
        .alloc_zeros::<f64>(nrr)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc hess: {err}"),
        })?;
    let mut d_f_au = stream
        .alloc_zeros::<f64>(nr)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc F_au scratch: {err}"),
        })?;
    let mut d_status = stream
        .alloc_zeros::<u32>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc status: {err}"),
        })?;

    let func = backend
        .module
        .load_function("bms_flex_row_kernel")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row load_function: {err}"),
        })?;

    let n_u32 = u32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row: n_rows={n} exceeds CUDA grid range"),
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n_u32, 1, 1),
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
        .arg(&d_e_obs)
        .arg(&mut d_f_au)
        .arg(&mut d_neglog)
        .arg(&mut d_grad)
        .arg(&mut d_hess)
        .arg(&mut d_status);

    // SAFETY: every kernel parameter above is either a primitive `i32` /
    // `f64` (passed by value), a const device pointer to a buffer whose
    // length the host validated against the input struct, or an output
    // buffers pre-allocated to checked `n_rows`, `n_rows*r`, and
    // `n_rows*r*r` lengths. Primary-width scratch aliases those outputs plus
    // the checked `d_f_au` allocation; no fixed-width device array exists.
    unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row launch: {err}"),
    })?;
    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row synchronize: {err}"),
        })?;

    let status = stream
        .clone_dtoh(&d_status)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row download status: {err}"),
        })?;
    if let Some((row, code)) = status
        .iter()
        .copied()
        .enumerate()
        .find(|(_, code)| *code != 0)
    {
        return Err(GpuError::DriverCallFailed {
            reason: format!("bms_flex_row rejected non-finite row {row} with status {code}"),
        });
    }

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
//     marginal = [0..p_m), slope = [p_m..p_m+p_g),
//     h        = [h_start..h_end), w = [w_start..w_end), total = p_total.
//
//   Primary layout (per-row r-vector):
//     q = 0, slope = 1,
//     h = [h_primary_start..h_primary_end),
//     w = [w_primary_start..w_primary_end), total = r.
//
//   row_dir[u] for u in primary layout:
//     row_dir[0]   = Σ_j marginal_design[row, j] · v[j]
//     row_dir[1]   = Σ_j slope_design[row, j] · v[p_m + j]
//     row_dir[h_k] = v[h_block_start + (h_k - h_primary_start)]
//     row_dir[w_k] = v[w_block_start + (w_k - w_primary_start)]
//
//   action[u]    = Σ_v row_hessians[row, u*r + v] · row_dir[v]
//
//   block_partial[marginal_j] += action[0] · marginal_design[row, j]
//   block_partial[slope_j] += action[1] · slope_design[row, j]
//   block_partial[h_block_start + (h_k - h_primary_start)] += action[h_k]
//   block_partial[w_block_start + (w_k - w_primary_start)] += action[w_k]
//
// Diagonal:
//   diag[marginal_j] += row_hess[row, 0*r + 0] · marginal_design[row, j]²
//   diag[slope_j] += row_hess[row, 1*r + 1] · slope_design[row, j]²
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

/// Maximum RHS columns fused into one row-primary HVP launch. This is an
/// internal batching width, not a matrix-width limit: wider dense matrices are
/// materialised in consecutive batches. The CUDA source has four scalar
/// shared arrays of this length; primary directions are derived on demand.
#[cfg(target_os = "linux")]
pub(crate) const BMS_FLEX_ROW_HVP_MAX_RHS: usize = 8;

/// Device-resident state produced by
/// `launch_bms_flex_row_kernel_device_resident` and consumed by
/// `launch_bms_flex_row_hvp` / `launch_bms_flex_row_diagonal`.
///
/// Owns the canonical row value, gradient, Hessian, and design slices on-device
/// so every downstream value/score/Hessian consumer shares one row evaluation
/// without round-tripping the large cache through host RAM. Drop releases the
/// device memory back to the CUDA runtime.
#[cfg(target_os = "linux")]
pub struct DeviceResidentRowHess {
    /// Per-row negative log likelihood emitted by the same canonical row
    /// program that emits `grad` and `hess`.
    pub(crate) neglog: CudaSlice<f64>,
    /// Per-row objective gradient `[n, r]`, row-major. Joint-score consumers
    /// negate and pull this buffer back through the two designs/direct blocks.
    pub(crate) grad: CudaSlice<f64>,
    /// Per-row dense `[n, r, r]` row-major Hessian. Element `(u, v)` of row
    /// `i` is `hess[i*r*r + u*r + v]`. This is the only on-device storage
    /// layout supported by the current HVP / diag kernels.
    pub(crate) hess: CudaSlice<f64>,
    pub(crate) marginal_design: CudaSlice<f64>,
    pub(crate) slope_design: CudaSlice<f64>,
    pub(crate) n: usize,
    pub(crate) r: usize,
    pub(crate) block: BmsFlexBlockLayout,
    pub(crate) primary: BmsFlexPrimaryLayout,
    /// Estimated bytes resident on device (for accounting).
    pub(crate) bytes: u64,
}

/// Host image of the deterministic device reduction over the canonical row
/// value/gradient buffers. The gradient uses log-likelihood/score sign.
#[cfg(target_os = "linux")]
pub(crate) struct BmsFlexDeviceJointGradient {
    pub(crate) log_likelihood: f64,
    pub(crate) gradient: Vec<f64>,
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

/// NVRTC source: deterministic joint-gradient, HVP, diagonal, and dense
/// partial+reduce kernels. All kernels mirror CPU oracles in this file.
#[cfg(target_os = "linux")]
pub(crate) const HVP_KERNEL_SOURCE: &str = r#"
// CPU parity reference: cpu_oracle_bms_flex_row_hvp / cpu_oracle_bms_flex_row_diagonal
// in this module.

#define MAX_MULTI_RHS 8

__device__ __forceinline__ double bms_flex_primary_direction(
    int primary_idx,
    int h_block_start,
    int h_block_len,
    int w_block_start,
    int w_block_len,
    int h_primary_start,
    int w_primary_start,
    double direction_q,
    double direction_g,
    const double * __restrict__ v)
{
    if (primary_idx == 0) return direction_q;
    if (primary_idx == 1) return direction_g;
    if (primary_idx >= h_primary_start && primary_idx < h_primary_start + h_block_len) {
        return v[h_block_start + primary_idx - h_primary_start];
    }
    if (primary_idx >= w_primary_start && primary_idx < w_primary_start + w_block_len) {
        return v[w_block_start + primary_idx - w_primary_start];
    }
    return 0.0;
}

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
    const double * __restrict__ slope_design, // [n, p_g] row-major
    const double * __restrict__ v,               // [p_total]
    double       * __restrict__ partial)         // [num_chunks, p_total]
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int remaining_rows = n_rows - row_lo;
    int row_hi = row_lo + (remaining_rows < rows_per_cta ? remaining_rows : rows_per_cta);

    // Zero this chunk's partial slice cooperatively.
    double *out = partial + (size_t)chunk * (size_t)p_total;
    for (int j = tid; j < p_total; j += blockDim.x) {
        out[j] = 0.0;
    }
    __syncthreads();

    // Width-general scratch: only the two design directions/actions are
    // shared. Every h/w direction is read directly from v, and the thread
    // owning primary coordinate u accumulates that coordinate's action.
    __shared__ double direction_q;
    __shared__ double direction_g;
    __shared__ double action_q;
    __shared__ double action_g;
    __shared__ double dot_reduce[128];

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = slope_design + (size_t)row * (size_t)p_g;
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
        if (tid == 0) direction_q = dot_reduce[0];

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
        if (tid == 0) direction_g = dot_reduce[0];
        __syncthreads();

        for (int u = tid; u < r; u += blockDim.x) {
            double acc = 0.0;
            for (int vv = 0; vv < r; ++vv) {
                double row_direction = bms_flex_primary_direction(
                    vv,
                    h_block_start, h_block_len,
                    w_block_start, w_block_len,
                    h_primary_start, w_primary_start,
                    direction_q, direction_g, v);
                acc += Hrow[(size_t)u * (size_t)r + (size_t)vv] * row_direction;
            }
            if (u == 0) {
                action_q = acc;
            } else if (u == 1) {
                action_g = acc;
            } else if (u >= h_primary_start && u < h_primary_start + h_block_len) {
                out[h_block_start + u - h_primary_start] += acc;
            } else if (u >= w_primary_start && u < w_primary_start + w_block_len) {
                out[w_block_start + u - w_primary_start] += acc;
            }
        }
        __syncthreads();

        // Pull back into joint β slot.
        double a0 = action_q;
        for (int j = tid; j < p_m; j += blockDim.x) {
            out[j] += a0 * mrow[j];
        }
        double a1 = action_g;
        for (int j = tid; j < p_g; j += blockDim.x) {
            out[p_m + j] += a1 * grow[j];
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

extern "C" __global__ void bms_flex_row_joint_gradient_partial(
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
    const double * __restrict__ row_neglog,       // [n]
    const double * __restrict__ row_grad,         // [n, r]
    const double * __restrict__ marginal_design,  // [n, p_m]
    const double * __restrict__ slope_design,  // [n, p_g]
    double       * __restrict__ partial)          // [num_chunks, 1+p_total]
{
    int chunk = blockIdx.x;
    int tid = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int remaining_rows = n_rows - row_lo;
    int row_hi = row_lo + (remaining_rows < rows_per_cta ? remaining_rows : rows_per_cta);
    int output_width = p_total + 1;
    double *out = partial + (size_t)chunk * (size_t)output_width;

    // One thread owns each output coordinate for the whole row chunk. The
    // inner row loop therefore has a fixed order and needs no atomics.
    for (int output_idx = tid; output_idx < output_width; output_idx += blockDim.x) {
        double acc = 0.0;
        if (output_idx == 0) {
            for (int row = row_lo; row < row_hi; ++row) {
                acc -= row_neglog[row];
            }
        } else {
            int beta_idx = output_idx - 1;
            if (beta_idx < p_m) {
                for (int row = row_lo; row < row_hi; ++row) {
                    acc -= row_grad[(size_t)row * (size_t)r]
                         * marginal_design[(size_t)row * (size_t)p_m + (size_t)beta_idx];
                }
            } else if (beta_idx < p_m + p_g) {
                int j = beta_idx - p_m;
                for (int row = row_lo; row < row_hi; ++row) {
                    acc -= row_grad[(size_t)row * (size_t)r + 1]
                         * slope_design[(size_t)row * (size_t)p_g + (size_t)j];
                }
            } else if (beta_idx >= h_block_start && beta_idx < h_block_start + h_block_len) {
                int primary_idx = h_primary_start + beta_idx - h_block_start;
                for (int row = row_lo; row < row_hi; ++row) {
                    acc -= row_grad[(size_t)row * (size_t)r + (size_t)primary_idx];
                }
            } else if (beta_idx >= w_block_start && beta_idx < w_block_start + w_block_len) {
                int primary_idx = w_primary_start + beta_idx - w_block_start;
                for (int row = row_lo; row < row_hi; ++row) {
                    acc -= row_grad[(size_t)row * (size_t)r + (size_t)primary_idx];
                }
            }
        }
        out[output_idx] = acc;
    }
}

extern "C" __global__ void bms_flex_row_joint_gradient_reduce(
    int                  num_chunks,
    int                  output_width,
    const double * __restrict__ partial,   // [num_chunks, output_width]
    double       * __restrict__ out)        // [output_width]
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= output_width) return;
    double acc = 0.0;
    for (int c = 0; c < num_chunks; ++c) {
        acc += partial[(size_t)c * (size_t)output_width + (size_t)j];
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
    const double * __restrict__ slope_design, // [n, p_g]
    const double * __restrict__ v_rhs,           // [rhs_count, p_total]
    double       * __restrict__ partial)         // [rhs_count, num_chunks, p_total]
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int remaining_rows = n_rows - row_lo;
    int row_hi = row_lo + (remaining_rows < rows_per_cta ? remaining_rows : rows_per_cta);

    int num_chunks = 1 + (n_rows - 1) / rows_per_cta;
    for (int idx = tid; idx < rhs_count * p_total; idx += blockDim.x) {
        int rhs = idx / p_total;
        int j = idx - rhs * p_total;
        partial[((size_t)rhs * (size_t)num_chunks + (size_t)chunk) * (size_t)p_total + (size_t)j] = 0.0;
    }
    __syncthreads();

    __shared__ double direction_q[MAX_MULTI_RHS];
    __shared__ double direction_g[MAX_MULTI_RHS];
    __shared__ double action_q[MAX_MULTI_RHS];
    __shared__ double action_g[MAX_MULTI_RHS];
    __shared__ double dot_reduce[128];

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = slope_design + (size_t)row * (size_t)p_g;
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
            if (tid == 0) direction_q[rhs] = dot_reduce[0];

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
            if (tid == 0) direction_g[rhs] = dot_reduce[0];
            __syncthreads();
        }

        size_t total_actions = (size_t)rhs_count * (size_t)r;
        for (size_t idx = (size_t)tid; idx < total_actions; idx += (size_t)blockDim.x) {
            int rhs = (int)(idx / (size_t)r);
            int u = (int)(idx - (size_t)rhs * (size_t)r);
            const double *v = v_rhs + (size_t)rhs * (size_t)p_total;
            double *out = partial + ((size_t)rhs * (size_t)num_chunks + (size_t)chunk) * (size_t)p_total;
            double acc = 0.0;
            for (int vv = 0; vv < r; ++vv) {
                double row_direction = bms_flex_primary_direction(
                    vv,
                    h_block_start, h_block_len,
                    w_block_start, w_block_len,
                    h_primary_start, w_primary_start,
                    direction_q[rhs], direction_g[rhs], v);
                acc += Hrow[(size_t)u * (size_t)r + (size_t)vv] * row_direction;
            }
            if (u == 0) {
                action_q[rhs] = acc;
            } else if (u == 1) {
                action_g[rhs] = acc;
            } else if (u >= h_primary_start && u < h_primary_start + h_block_len) {
                out[h_block_start + u - h_primary_start] += acc;
            } else if (u >= w_primary_start && u < w_primary_start + w_block_len) {
                out[w_block_start + u - w_primary_start] += acc;
            }
        }
        __syncthreads();

        for (int rhs = 0; rhs < rhs_count; ++rhs) {
            double *out = partial + ((size_t)rhs * (size_t)num_chunks + (size_t)chunk) * (size_t)p_total;
            double a0 = action_q[rhs];
            for (int j = tid; j < p_m; j += blockDim.x) {
                out[j] += a0 * mrow[j];
            }
            double a1 = action_g[rhs];
            for (int j = tid; j < p_g; j += blockDim.x) {
                out[p_m + j] += a1 * grow[j];
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
    const double * __restrict__ slope_design,
    double       * __restrict__ partial)
{
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int remaining_rows = n_rows - row_lo;
    int row_hi = row_lo + (remaining_rows < rows_per_cta ? remaining_rows : rows_per_cta);

    double *out = partial + (size_t)chunk * (size_t)p_total;
    for (int j = tid; j < p_total; j += blockDim.x) {
        out[j] = 0.0;
    }
    __syncthreads();

    for (int row = row_lo; row < row_hi; ++row) {
        const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
        const double *grow = slope_design + (size_t)row * (size_t)p_g;
        const double *Hrow = row_hessians + (size_t)row * (size_t)r * (size_t)r;
        double h00 = Hrow[0];
        double h11 = Hrow[(size_t)r + 1U];
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
                out[h_block_start + k] +=
                    Hrow[(size_t)ii * (size_t)r + (size_t)ii];
            }
            for (int k = 0; k < w_block_len; ++k) {
                int ii = w_primary_start + k;
                out[w_block_start + k] +=
                    Hrow[(size_t)ii * (size_t)r + (size_t)ii];
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
    const double * __restrict__ slope_design, // [n, p_g]
    double       * __restrict__ partial)         // [num_chunks, p_total, p_total]
{
    extern __shared__ double shmem[];
    int chunk = blockIdx.x;
    int tid   = threadIdx.x;
    int row_lo = chunk * rows_per_cta;
    int remaining_rows = n_rows - row_lo;
    int row_hi = row_lo + (remaining_rows < rows_per_cta ? remaining_rows : rows_per_cta);

    int pp = p_total * p_total;
    double *acc = shmem; // CTA-private accumulator [p_total, p_total]
    for (int j = tid; j < pp; j += blockDim.x) acc[j] = 0.0;
    __syncthreads();

    // Per-row work performed by thread 0 to avoid cross-thread RW
    // contention on `acc[]`. Per-row complexity is O(r² + p_total²); the host
    // selects this direct algorithm only for small p_total, while r remains a
    // checked runtime width with no semantic ceiling.
    // Tighter parallel implementations are possible (warp-stripe the
    // 4-way nested u-v-m-n loop) but Phase 6 is a debug-only path and
    // the simple version is easier to audit for correctness against
    // the host-side P_i pullback oracle.
    if (tid == 0) {
        for (int row = row_lo; row < row_hi; ++row) {
            const double *mrow = marginal_design + (size_t)row * (size_t)p_m;
            const double *grow = slope_design + (size_t)row * (size_t)p_g;
            const double *Hrow = row_hessians + (size_t)row * (size_t)r * (size_t)r;
            for (int u = 0; u < r; ++u) {
                for (int v = 0; v < r; ++v) {
                    double huv = Hrow[(size_t)u * (size_t)r + (size_t)v];
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
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(HVP_KERNEL_SOURCE).map_err(
                        |err| GpuError::DriverCallFailed {
                            reason: format!("bms_flex_row hvp NVRTC compile failed: {err}"),
                        },
                    )?;
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
/// the dense marginal + slope design matrices so subsequent HVPs do not
/// re-upload them at every direction.
///
/// `marginal_design_row_major` and `slope_design_row_major` must be
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
/// design against a widened `block.p_m` is rejected cleanly rather than
/// silently computing the wrong η. The absorber is dropped at
/// predict, where the marginal design is rebuilt without the influence columns,
/// so the predict-time `p_m` is narrow and this path is correct there too.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_kernel_device_resident(
    inputs: BmsFlexRowKernelInputs<'_>,
    marginal_design_row_major: &[f64],
    slope_design_row_major: &[f64],
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
    let nr = checked_shape_len("device-resident [n,r]", &[n, r])?;
    let nrr = checked_shape_len("device-resident [n,r,r]", &[n, r, r])?;
    let marginal_len = checked_shape_len("device-resident marginal design", &[n, block.p_m])?;
    let slope_len = checked_shape_len("device-resident slope design", &[n, block.p_g])?;
    if marginal_design_row_major.len() != marginal_len {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident: marginal_design len={} != n*p_m={}",
                marginal_design_row_major.len(),
                marginal_len
            ),
        });
    }
    if slope_design_row_major.len() != slope_len {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident: slope_design len={} != n*p_g={}",
                slope_design_row_major.len(),
                slope_len
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
    let d_e_obs = upload_f64(inputs.e_obs, "e_obs")?;

    let d_marginal = upload_f64(marginal_design_row_major, "marginal_design")?;
    let d_slope = upload_f64(slope_design_row_major, "slope_design")?;

    let mut d_neglog = stream
        .alloc_zeros::<f64>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident alloc neglog: {err}"),
        })?;
    let mut d_grad = stream
        .alloc_zeros::<f64>(nr)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident alloc grad: {err}"),
        })?;
    let mut d_hess = stream
        .alloc_zeros::<f64>(nrr)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident alloc hess: {err}"),
        })?;
    let mut d_f_au = stream
        .alloc_zeros::<f64>(nr)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident alloc F_au scratch: {err}"),
        })?;
    let mut d_status = stream
        .alloc_zeros::<u32>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident alloc status: {err}"),
        })?;

    let func = backend
        .module
        .load_function("bms_flex_row_kernel")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident load_function: {err}"),
        })?;

    let n_u32 = u32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row device-resident: n_rows={n} exceeds CUDA grid range"),
    })?;
    let cfg = LaunchConfig {
        grid_dim: (n_u32, 1, 1),
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
        .arg(&d_e_obs)
        .arg(&mut d_f_au)
        .arg(&mut d_neglog)
        .arg(&mut d_grad)
        .arg(&mut d_hess)
        .arg(&mut d_status);
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

    let status = stream
        .clone_dtoh(&d_status)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row device-resident download status: {err}"),
        })?;
    if let Some((row, code)) = status
        .iter()
        .copied()
        .enumerate()
        .find(|(_, code)| *code != 0)
    {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row device-resident rejected non-finite row {row} with status {code}"
            ),
        });
    }
    drop(d_status);
    drop(d_f_au);

    // Drop the per-cell uploads; keep the canonical row value, gradient,
    // Hessian, and both pullback designs in one device-resident authority.
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

    let resident_elements = n
        .checked_add(nr)
        .and_then(|value| value.checked_add(nrr))
        .and_then(|value| value.checked_add(marginal_len))
        .and_then(|value| value.checked_add(slope_len))
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: "bms_flex_row device-resident: resident element count overflow".to_string(),
        })?;
    let resident_bytes = resident_elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: "bms_flex_row device-resident: resident byte count overflow".to_string(),
        })?;
    let bytes = u64::try_from(resident_bytes).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row device-resident: resident bytes={resident_bytes} exceed u64 range"
        ),
    })?;
    Ok(DeviceResidentRowHess {
        neglog: d_neglog,
        grad: d_grad,
        hess: d_hess,
        marginal_design: d_marginal,
        slope_design: d_slope,
        n,
        r,
        block,
        primary,
        bytes,
    })
}

/// Reduce the canonical per-row value and objective gradient into the joint
/// log-likelihood and score gradient without re-running row calculus on CPU.
/// Both stages use fixed row/chunk order and no atomics.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_joint_gradient(
    storage: &DeviceResidentRowHess,
) -> Result<BmsFlexDeviceJointGradient, GpuError> {
    let p_total = storage.block.p_total;
    let output_width = p_total
        .checked_add(1)
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: "bms_flex_row joint gradient: output width overflow".to_string(),
        })?;
    if storage.n == 0 {
        return Ok(BmsFlexDeviceJointGradient {
            log_likelihood: 0.0,
            gradient: vec![0.0; p_total],
        });
    }

    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();
    let args = PreparedBmsFlexRowLaunchArgs::from_storage(storage)?;
    let partial_len = args
        .num_chunks
        .checked_mul(output_width)
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row joint gradient: partial length overflow for chunks={} width={output_width}",
                args.num_chunks
            ),
        })?;
    let mut d_partial =
        stream
            .alloc_zeros::<f64>(partial_len)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row joint gradient alloc partial: {err}"),
            })?;
    let mut d_out =
        stream
            .alloc_zeros::<f64>(output_width)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row joint gradient alloc output: {err}"),
            })?;
    let partial_func = backend
        .module
        .load_function("bms_flex_row_joint_gradient_partial")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row joint gradient load partial: {err}"),
        })?;
    let reduce_func = backend
        .module
        .load_function("bms_flex_row_joint_gradient_reduce")
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row joint gradient load reduce: {err}"),
        })?;

    let num_chunks_u32 =
        u32::try_from(args.num_chunks).map_err(|_| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row joint gradient: num_chunks={} exceeds u32 range",
                args.num_chunks
            ),
        })?;
    let cfg_partial = LaunchConfig {
        grid_dim: (num_chunks_u32, 1, 1),
        block_dim: (HVP_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&partial_func);
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
        .arg(&storage.neglog)
        .arg(&storage.grad)
        .arg(&storage.marginal_design)
        .arg(&storage.slope_design)
        .arg(&mut d_partial);
    // SAFETY: all resident buffers were allocated and shape-validated by the
    // row-kernel producer; `d_partial` has `num_chunks * (1+p_total)` entries.
    unsafe { builder.launch(cfg_partial) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row joint gradient partial launch: {err}"),
    })?;

    let output_width_i32 = i32::try_from(output_width).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row joint gradient: output_width={output_width} exceeds i32 range"
        ),
    })?;
    let num_chunks_i32 =
        i32::try_from(args.num_chunks).map_err(|_| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row joint gradient: num_chunks={} exceeds i32 range",
                args.num_chunks
            ),
        })?;
    let output_width_u32 = u32::try_from(output_width).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row joint gradient: output_width={output_width} exceeds u32 range"
        ),
    })?;
    let reduce_blocks = output_width_u32.div_ceil(REDUCTION_THREADS);
    let cfg_reduce = LaunchConfig {
        grid_dim: (reduce_blocks, 1, 1),
        block_dim: (REDUCTION_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&reduce_func);
    builder
        .arg(&num_chunks_i32)
        .arg(&output_width_i32)
        .arg(&d_partial)
        .arg(&mut d_out);
    // SAFETY: the partial launch above populated the exact partial shape and
    // `d_out` owns `output_width` entries.
    unsafe { builder.launch(cfg_reduce) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row joint gradient reduce launch: {err}"),
    })?;
    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row joint gradient synchronize: {err}"),
        })?;
    let host = stream
        .clone_dtoh(&d_out)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row joint gradient download: {err}"),
        })?;
    if let Some((index, value)) = host
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row joint gradient produced non-finite output[{index}]={value}"
            ),
        });
    }
    Ok(BmsFlexDeviceJointGradient {
        log_likelihood: host[0],
        gradient: host[1..].to_vec(),
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
    /// `bms_flex_row_diag_partial`, `diag(H)` per row, downloaded to host.
    DiagonalHostOut,
}

#[cfg(target_os = "linux")]
impl BmsFlexRowLaunchMode {
    /// Name of the partial kernel this mode loads from the HVP module.
    pub(crate) fn partial_kernel_name(self) -> &'static str {
        match self {
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
    pub(crate) num_chunks_i32: i32,
    pub(crate) num_chunks_u32: u32,
    pub(crate) p_total_u32: u32,
}

#[cfg(target_os = "linux")]
impl PreparedBmsFlexRowLaunchArgs {
    pub(crate) fn from_storage(storage: &DeviceResidentRowHess) -> Result<Self, GpuError> {
        if storage.n == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "bms_flex_row launch: n_rows must be > 0".to_string(),
            });
        }
        if storage.r < 2 {
            return Err(GpuError::DriverCallFailed {
                reason: format!("bms_flex_row launch: r={} must be >= 2", storage.r),
            });
        }
        let p_total = storage.block.p_total;
        if p_total == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "bms_flex_row launch: p_total must be > 0".to_string(),
            });
        }
        if storage.primary.r != storage.r {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row launch: primary.r={} != storage.r={}",
                    storage.primary.r, storage.r
                ),
            });
        }
        let h_block_len = storage.block.h.as_ref().map_or(0, |range| range.len());
        let w_block_len = storage.block.w.as_ref().map_or(0, |range| range.len());
        let h_primary_len = storage.primary.h.as_ref().map_or(0, |range| range.len());
        let w_primary_len = storage.primary.w.as_ref().map_or(0, |range| range.len());
        if h_block_len != h_primary_len || w_block_len != w_primary_len {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row launch: block/primary direct lengths disagree: h={h_block_len}/{h_primary_len}, w={w_block_len}/{w_primary_len}"
                ),
            });
        }
        let h_block_start = storage
            .block
            .p_m
            .checked_add(storage.block.p_g)
            .ok_or_else(|| GpuError::DriverCallFailed {
                reason: "bms_flex_row launch: p_m+p_g overflow".to_string(),
            })?;
        let w_block_start =
            h_block_start
                .checked_add(h_block_len)
                .ok_or_else(|| GpuError::DriverCallFailed {
                    reason: "bms_flex_row launch: h block end overflow".to_string(),
                })?;
        let expected_p_total =
            w_block_start
                .checked_add(w_block_len)
                .ok_or_else(|| GpuError::DriverCallFailed {
                    reason: "bms_flex_row launch: w block end overflow".to_string(),
                })?;
        let w_primary_start =
            2_usize
                .checked_add(h_primary_len)
                .ok_or_else(|| GpuError::DriverCallFailed {
                    reason: "bms_flex_row launch: h primary end overflow".to_string(),
                })?;
        let expected_r = w_primary_start.checked_add(w_primary_len).ok_or_else(|| {
            GpuError::DriverCallFailed {
                reason: "bms_flex_row launch: w primary end overflow".to_string(),
            }
        })?;
        let check_range = |name: &str,
                           range: Option<&std::ops::Range<usize>>,
                           expected_start: usize,
                           expected_len: usize|
         -> Result<(), GpuError> {
            match (range, expected_len) {
                (None, 0) => Ok(()),
                (Some(range), len)
                    if len > 0
                        && range.start == expected_start
                        && range.end == expected_start + len =>
                {
                    Ok(())
                }
                _ => Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "bms_flex_row launch: {name}={range:?} must be {expected_start}..{}",
                        expected_start + expected_len
                    ),
                }),
            }
        };
        check_range(
            "block.h",
            storage.block.h.as_ref(),
            h_block_start,
            h_block_len,
        )?;
        check_range(
            "block.w",
            storage.block.w.as_ref(),
            w_block_start,
            w_block_len,
        )?;
        check_range("primary.h", storage.primary.h.as_ref(), 2, h_primary_len)?;
        check_range(
            "primary.w",
            storage.primary.w.as_ref(),
            w_primary_start,
            w_primary_len,
        )?;
        if p_total != expected_p_total || storage.r != expected_r {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row launch: inconsistent layout p_total={p_total}/{expected_p_total}, r={}/{}",
                    storage.r, expected_r
                ),
            });
        }
        let expected_nr = checked_shape_len("launch storage [n,r]", &[storage.n, storage.r])?;
        let expected_nrr =
            checked_shape_len("launch storage [n,r,r]", &[storage.n, storage.r, storage.r])?;
        let expected_marginal = checked_shape_len(
            "launch storage marginal design",
            &[storage.n, storage.block.p_m],
        )?;
        let expected_slope = checked_shape_len(
            "launch storage slope design",
            &[storage.n, storage.block.p_g],
        )?;
        for (name, have, want) in [
            ("neglog", storage.neglog.len(), storage.n),
            ("grad", storage.grad.len(), expected_nr),
            ("hess", storage.hess.len(), expected_nrr),
            (
                "marginal_design",
                storage.marginal_design.len(),
                expected_marginal,
            ),
            (
                "slope_design",
                storage.slope_design.len(),
                expected_slope,
            ),
        ] {
            if have != want {
                return Err(GpuError::DriverCallFailed {
                    reason: format!("bms_flex_row launch: storage {name}.len()={have} != {want}"),
                });
            }
        }
        let num_chunks = num_hvp_chunks(storage.n);
        let to_i32 = |name: &str, value: usize| {
            i32::try_from(value).map_err(|_| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row launch: {name}={value} exceeds i32 range"),
            })
        };
        let to_u32 = |name: &str, value: usize| {
            u32::try_from(value).map_err(|_| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row launch: {name}={value} exceeds u32 range"),
            })
        };
        Ok(PreparedBmsFlexRowLaunchArgs {
            n_i32: to_i32("n_rows", storage.n)?,
            r_i32: to_i32("r", storage.r)?,
            p_m_i32: to_i32("p_m", storage.block.p_m)?,
            p_g_i32: to_i32("p_g", storage.block.p_g)?,
            p_total_i32: to_i32("p_total", p_total)?,
            h_block_start: storage
                .block
                .h
                .as_ref()
                .map(|range| to_i32("h_block_start", range.start))
                .transpose()?
                .unwrap_or(0),
            h_block_len: storage
                .block
                .h
                .as_ref()
                .map(|range| to_i32("h_block_len", range.len()))
                .transpose()?
                .unwrap_or(0),
            w_block_start: storage
                .block
                .w
                .as_ref()
                .map(|range| to_i32("w_block_start", range.start))
                .transpose()?
                .unwrap_or(0),
            w_block_len: storage
                .block
                .w
                .as_ref()
                .map(|range| to_i32("w_block_len", range.len()))
                .transpose()?
                .unwrap_or(0),
            h_primary_start: storage
                .primary
                .h
                .as_ref()
                .map(|range| to_i32("h_primary_start", range.start))
                .transpose()?
                .unwrap_or(0),
            w_primary_start: storage
                .primary
                .w
                .as_ref()
                .map(|range| to_i32("w_primary_start", range.start))
                .transpose()?
                .unwrap_or(0),
            rows_per_cta: i32::try_from(HVP_ROWS_PER_CTA).map_err(|_| {
                GpuError::DriverCallFailed {
                    reason: format!(
                        "bms_flex_row launch: rows_per_cta={HVP_ROWS_PER_CTA} exceeds i32 range"
                    ),
                }
            })?,
            num_chunks,
            num_chunks_i32: to_i32("num_chunks", num_chunks)?,
            num_chunks_u32: to_u32("num_chunks", num_chunks)?,
            p_total_u32: to_u32("p_total", p_total)?,
        })
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
    let args = PreparedBmsFlexRowLaunchArgs::from_storage(storage)?;
    let p_total = storage.block.p_total;

    let partial_len = checked_shape_len(
        &format!("{ctx} partial [num_chunks,p_total]"),
        &[args.num_chunks, p_total],
    )?;
    let mut d_partial =
        stream
            .alloc_zeros::<f64>(partial_len)
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
        grid_dim: (args.num_chunks_u32, 1, 1),
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
        .arg(&storage.slope_design);
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
    let red_blocks = args.p_total_u32.div_ceil(red_threads);
    let cfg_red = LaunchConfig {
        grid_dim: (red_blocks, 1, 1),
        block_dim: (red_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&red_func);
    builder
        .arg(&args.num_chunks_i32)
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

/// Host-returning diagonal adapter. HVP host output uses the multi-RHS engine,
/// so this function exposes only the one live no-direction mode.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_diagonal_host(
    storage: &DeviceResidentRowHess,
) -> Result<Vec<f64>, GpuError> {
    let p_total = storage.block.p_total;
    let backend = HvpKernelBackend::probe()?;
    let stream = backend.stream.clone();
    let mut d_out =
        stream
            .alloc_zeros::<f64>(p_total)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex_row diag alloc out: {err}"),
            })?;

    run_bms_flex_row_partial_reduce(
        storage,
        BmsFlexRowLaunchMode::DiagonalHostOut,
        None,
        &mut d_out,
        "diag",
    )?;

    stream
        .synchronize()
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row diag synchronize: {err}"),
        })?;
    stream
        .clone_dtoh(&d_out)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row diag download out: {err}"),
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
    i32::try_from(rhs_elems).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row {ctx}: rhs_count({rhs_count})*p_total({p_total})={rhs_elems} exceeds CUDA int indexing range"
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
    let bytes = elems
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: "bms_flex_row hvp_multi_scratch_bytes: byte count overflow".to_string(),
        })?;
    u64::try_from(bytes).map_err(|_| GpuError::DriverCallFailed {
        reason: format!(
            "bms_flex_row hvp_multi_scratch_bytes: byte count={bytes} exceeds u64 range"
        ),
    })
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
    let args = PreparedBmsFlexRowLaunchArgs::from_storage(storage)?;
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
        grid_dim: (args.num_chunks_u32, 1, 1),
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
        .arg(&storage.slope_design)
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
    let rhs_elems_u32 = u32::try_from(rhs_elems).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row {ctx}: rhs elements={rhs_elems} exceed u32 range"),
    })?;
    let red_blocks = rhs_elems_u32.div_ceil(red_threads);
    let cfg_red = LaunchConfig {
        grid_dim: (red_blocks, 1, 1),
        block_dim: (red_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&red_func);
    builder
        .arg(&args.num_chunks_i32)
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

/// Materialize a row-major dense matrix from batched column images `H * I`.
/// Each launcher input/output is row-major `[rhs_count, p_total]`; the output
/// vectors are columns of `H`, so this routine performs the one required
/// transpose while copying them into `[row, column]` storage.
#[cfg(target_os = "linux")]
fn materialize_dense_from_hvp_batches(
    p_total: usize,
    mut launch: impl FnMut(&[f64], usize) -> Result<Vec<f64>, GpuError>,
) -> Result<Vec<f64>, GpuError> {
    if p_total == 0 {
        return Err(GpuError::DriverCallFailed {
            reason: "bms_flex_row dense HVP materialization: p_total must be > 0".to_string(),
        });
    }
    let dense_len = p_total
        .checked_mul(p_total)
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row dense HVP materialization: p_total={p_total} square overflow"
            ),
        })?;
    let mut dense = vec![0.0_f64; dense_len];
    for column_start in (0..p_total).step_by(BMS_FLEX_ROW_HVP_MAX_RHS) {
        let rhs_count = (p_total - column_start).min(BMS_FLEX_ROW_HVP_MAX_RHS);
        let batch_len = checked_shape_len(
            "dense HVP materialization [rhs_count,p_total]",
            &[rhs_count, p_total],
        )?;
        let mut basis = vec![0.0_f64; batch_len];
        for local_column in 0..rhs_count {
            basis[local_column * p_total + column_start + local_column] = 1.0;
        }
        let images = launch(&basis, rhs_count)?;
        if images.len() != basis.len() {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex_row dense HVP materialization: batch at column {column_start} returned {} values, expected {}",
                    images.len(),
                    basis.len()
                ),
            });
        }
        for local_column in 0..rhs_count {
            let column = column_start + local_column;
            let image = &images[local_column * p_total..(local_column + 1) * p_total];
            for (row, &value) in image.iter().enumerate() {
                dense[row * p_total + column] = value;
            }
        }
    }
    Ok(dense)
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
    launch_bms_flex_row_diagonal_host(storage)
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

/// Materialize the selected device-resident joint Hessian using the fastest
/// CUDA algorithm supported by its width. The direct shared-memory kernel is
/// used through [`DENSE_BLOCK_MAX_P`]; wider matrices are formed as batched
/// `H * I` column images through the existing bounded multi-RHS HVP kernel.
/// This is an up-front device algorithm choice, not a CUDA-to-CPU fallback.
#[cfg(target_os = "linux")]
pub(crate) fn launch_bms_flex_row_dense(
    storage: &DeviceResidentRowHess,
) -> Result<Vec<f64>, GpuError> {
    let p_total = storage.block.p_total;
    if p_total <= DENSE_BLOCK_MAX_P {
        return launch_bms_flex_row_dense_block(storage);
    }
    materialize_dense_from_hvp_batches(p_total, |basis, rhs_count| {
        launch_bms_flex_row_hvp_multi(storage, basis, rhs_count)
    })
}

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
    let args = PreparedBmsFlexRowLaunchArgs::from_storage(storage)?;
    let n = storage.n;
    let rows_per_cta = DENSE_BLOCK_ROWS_PER_CTA as usize;
    let num_chunks = n.div_ceil(rows_per_cta);
    let pp = checked_shape_len("dense_block [p_total,p_total]", &[p_total, p_total])?;
    let partial_len = checked_shape_len("dense_block partial", &[num_chunks, pp])?;

    let mut d_partial =
        stream
            .alloc_zeros::<f64>(partial_len)
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

    let rows_per_cta_i32 = i32::try_from(DENSE_BLOCK_ROWS_PER_CTA).map_err(|_| {
        GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex_row dense_block: rows_per_cta={DENSE_BLOCK_ROWS_PER_CTA} exceeds i32 range"
            ),
        }
    })?;
    let num_chunks_u32 = u32::try_from(num_chunks).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row dense_block: num_chunks={num_chunks} exceeds u32 range"),
    })?;
    let num_chunks_i32 = i32::try_from(num_chunks).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row dense_block: num_chunks={num_chunks} exceeds i32 range"),
    })?;
    let pp_u32 = u32::try_from(pp).map_err(|_| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row dense_block: p_total²={pp} exceeds u32 range"),
    })?;

    // Per-CTA shmem accumulator: p_total² doubles.
    let shmem_bytes_usize =
        pp.checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| GpuError::DriverCallFailed {
                reason: format!("dense_block shmem bytes overflow for p_total={p_total}"),
            })?;
    let shmem_bytes: u32 =
        u32::try_from(shmem_bytes_usize).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("dense_block shmem bytes overflow u32 for p_total={p_total}"),
        })?;

    let cfg_part = LaunchConfig {
        grid_dim: (num_chunks_u32, 1, 1),
        block_dim: (HVP_THREADS, 1, 1),
        shared_mem_bytes: shmem_bytes,
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
        .arg(&rows_per_cta_i32)
        .arg(&storage.hess)
        .arg(&storage.marginal_design)
        .arg(&storage.slope_design)
        .arg(&mut d_partial);
    // SAFETY: storage pointers have validated capacities; d_partial sized
    // num_chunks * pp doubles; dynamic shmem matches the kernel's `extern
    // __shared__` accumulator length.
    unsafe { builder.launch(cfg_part) }.map_err(|err| GpuError::DriverCallFailed {
        reason: format!("bms_flex_row dense_block partial launch: {err}"),
    })?;

    let red_threads: u32 = REDUCTION_THREADS;
    let red_blocks = pp_u32.div_ceil(red_threads);
    let cfg_red = LaunchConfig {
        grid_dim: (red_blocks, 1, 1),
        block_dim: (red_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&red_func);
    builder
        .arg(&num_chunks_i32)
        .arg(&args.p_total_i32)
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

// Host numerical primitives and the production CPU↔generated-CUDA parity lock.
#[cfg(test)]
mod row_kernel_tests {

    // Sole consumer is the `cfg(all(test, target_os = "linux"))` device-test
    // module below; off-Linux this would be dead code under `-D warnings`.

    // #415 parity lock: one fitted StandardNormal FLEX family supplies both
    // the production CPU lowering and the generated CUDA launch.
    pub(crate) mod parity_415 {
        use crate::bms::family::*;
        use crate::bms::{DeviationBlockConfig, LatentMeasureKind, exact_kernel};
        use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
        use gam_problem::{InverseLink, ParameterBlockState, StandardLink};
        use ndarray::{Array1, Array2};
        use std::sync::{Arc, Mutex};

        /// Build a small but REAL flex BMS family in the `StandardNormal`
        /// latent-measure branch with BOTH a score-warp (`p_h > 0`) and a
        /// link-deviation (`p_w > 0`) block active, plus mixed labels y ∈ {0,1}.
        /// Ported from the `gradient_paths` flex oracle fixture so the cache is
        /// populated by the production cell-moment assembly (never hand-faked).
        pub(crate) fn make_flex_parity_family(
            n: usize,
            score_internal_knots: usize,
            link_internal_knots: usize,
        ) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
            let score_seed = Array1::linspace(-2.0, 2.0, n.max(6));
            let link_seed = Array1::linspace(-1.8, 1.8, n.max(6));
            let score_cfg = DeviationBlockConfig {
                num_internal_knots: score_internal_knots,
                ..DeviationBlockConfig::default()
            };
            let link_cfg = DeviationBlockConfig {
                num_internal_knots: link_internal_knots,
                ..DeviationBlockConfig::default()
            };
            let score_prepared =
                build_score_warp_deviation_block_from_seed(&score_seed, &score_cfg)
                    .expect("build score warp block");
            let link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
                &link_seed, &link_seed, &link_cfg,
            )
            .expect("build link deviation block");

            // Mixed labels y ∈ {0,1} so both s_y = ±1 Mills branches are exercised.
            let y: Array1<f64> =
                Array1::from_iter((0..n).map(|i| if (i * 17 + 3) % 7 >= 4 { 1.0 } else { 0.0 }));
            let weights: Array1<f64> =
                Array1::from_iter((0..n).map(|i| 0.75 + ((i * 11 + 5) % 5) as f64 * 0.05));
            let z: Array1<f64> =
                Array1::from_iter((0..n).map(|i| -1.7 + 3.4 * (i as f64 + 0.5) / n as f64));
            let marginal_x = Array2::from_shape_fn((n, 2), |(i, j)| {
                if j == 0 {
                    1.0
                } else {
                    -0.4 + 0.8 * ((i * 19 + 7) % n) as f64 / n as f64
                }
            });
            let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
                if j == 0 {
                    1.0
                } else {
                    0.3 - 0.6 * ((i * 23 + 11) % n) as f64 / n as f64
                }
            });

            let family = BernoulliMarginalSlopeFamily {
                y: Arc::new(y),
                weights: Arc::new(weights),
                z: Arc::new(z.clone()),
                latent_measure: LatentMeasureKind::StandardNormal,
                gaussian_frailty_sd: Some(0.15),
                base_link: InverseLink::Standard(StandardLink::Probit),
                marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
                slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone())),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
                policy: gam_runtime::resource::ResourcePolicy::default_library(),
                cell_moment_lru: Arc::new(exact_kernel::CellMomentLruCache::new(1024)),
                cell_moment_cache_stats: Arc::new(exact_kernel::CellMomentCacheStats::default()),
                intercept_warm_starts: None,
                auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                auto_subsample_last_rho: Arc::new(Mutex::new(None)),
            };

            let beta_m = Array1::from_vec(vec![0.12, -0.04]);
            let beta_g = Array1::from_vec(vec![0.35, 0.03]);
            let beta_h = Array1::from_iter(
                (0..score_prepared.runtime.basis_dim()).map(|idx| 0.0015 * (idx as f64 + 1.0)),
            );
            let beta_w = Array1::from_iter(
                (0..link_prepared.runtime.basis_dim()).map(|idx| -0.001 * (idx as f64 + 1.0)),
            );
            let states = vec![
                ParameterBlockState {
                    eta: marginal_x.dot(&beta_m),
                    beta: beta_m,
                },
                ParameterBlockState {
                    eta: slope_x.dot(&beta_g),
                    beta: beta_g,
                },
                ParameterBlockState {
                    beta: beta_h,
                    eta: Array1::zeros(z.len()),
                },
                ParameterBlockState {
                    beta: beta_w,
                    eta: Array1::zeros(z.len()),
                },
            ];
            (family, states)
        }

        #[test]
        fn full_flex_canonical_exact_cache_admits_material_finite_cell_curvature_2321() {
            let (family, states) = make_flex_parity_family(256, 8, 6);
            let cache = family
                .build_exact_eval_cache(&states)
                .expect("the full-FLEX host cache must preserve non-affine finite cells");

            let score_width = cache
                .primary
                .h
                .as_ref()
                .expect("the full-FLEX fixture must retain its score-warp block")
                .len();
            let deviation_width = cache
                .primary
                .w
                .as_ref()
                .expect("the full-FLEX fixture must retain its link-deviation block")
                .len();
            assert!(score_width > 0 && deviation_width > 0);
            assert_eq!(
                cache.primary.total,
                2 + score_width + deviation_width,
                "the canonical primary layout must contain exactly q, slope, score-warp, and link-deviation coordinates"
            );
            assert!(
                cache.row_cell_moments.is_some(),
                "the production full-FLEX fixture must materialize its exact row-cell cache"
            );
        }

    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    // Deliberately NOT importing `configure_global_policy`: the process-wide
    // policy is a first-writer-wins `OnceLock`, so a test that writes it decides
    // the backend every other test in this binary selects. `GpuPolicy` itself is
    // no longer named here either — the availability probe moved behind
    // `gam_gpu::test_gate::gpu_for_test`, which owns the explicit `Auto`.

    #[test]
    fn dense_hvp_batches_transpose_column_images_in_bounded_groups_932() {
        let p_total = 2 * BMS_FLEX_ROW_HVP_MAX_RHS + 3;
        let matrix = (0..p_total * p_total)
            .map(|index| {
                let row = index / p_total;
                let column = index % p_total;
                1000.0 * row as f64 + column as f64 + 0.25
            })
            .collect::<Vec<_>>();
        let mut observed_batch_sizes = Vec::new();
        let dense = materialize_dense_from_hvp_batches(p_total, |basis, rhs_count| {
            observed_batch_sizes.push(rhs_count);
            let mut images = vec![0.0_f64; rhs_count * p_total];
            for rhs in 0..rhs_count {
                for row in 0..p_total {
                    images[rhs * p_total + row] = (0..p_total)
                        .map(|column| {
                            matrix[row * p_total + column] * basis[rhs * p_total + column]
                        })
                        .sum();
                }
            }
            Ok(images)
        })
        .expect("synthetic H*I batches must materialize");
        assert_eq!(dense, matrix);
        assert_eq!(
            observed_batch_sizes,
            vec![BMS_FLEX_ROW_HVP_MAX_RHS, BMS_FLEX_ROW_HVP_MAX_RHS, 3]
        );
    }

    #[test]
    pub(crate) fn checked_shape_len_rejects_arithmetic_overflow() {
        let err = checked_shape_len("overflow test", &[usize::MAX, 2])
            .expect_err("shape multiplication must fail closed");
        assert!(err.to_string().contains("shape product overflow"));
    }

    #[test]
    pub(crate) fn generated_source_interprets_compact_canonical_phase_streams() {
        let source = generated_row_kernel_source();
        assert!(!source.contains("__BMS_FLEX_CALIBRATION_ORDER2__"));
        assert!(!source.contains("__BMS_FLEX_ORDER2_FINALIZER__"));
        assert!(!source.contains("__BMS_FLEX_ROW_THREADS__"));
        assert!(source.contains("for (int u = 1; u < r; ++u)"));
        assert!(source.contains("for (int v = u; v < r; ++v)"));
        assert!(source.contains("Canonical implicit-first stage complete"));
        assert!(source.contains("double *F_u = out_grad + row_r_base"));
        assert!(source.contains("double *F_au = row_f_au + row_r_base"));
        assert!(source.contains("double *F_uv = out_hess + row_rr_base"));
        for forbidden in [
            "MAX_R",
            "double F_u[",
            "double F_au[",
            "double F_uv[",
            "double a_u[",
            "double a_uv[",
            "double bar_e_u[",
        ] {
            assert!(
                !source.contains(forbidden),
                "generated row source restored width-bound scratch: {forbidden}"
            );
        }
        for forbidden in [
            "MAX_R",
            "double row_dir[",
            "double action[",
            "bms_flex_row_hvp_partial_packed",
            "bms_flex_row_diag_partial_packed",
            "bms_flex_row_pack_upper",
        ] {
            assert!(
                !HVP_KERNEL_SOURCE.contains(forbidden),
                "HVP source restored a dead or width-bound path: {forbidden}"
            );
        }
        assert!(HVP_KERNEL_SOURCE.contains("bms_flex_primary_direction"));
        assert!(HVP_KERNEL_SOURCE.contains("direction_q[MAX_MULTI_RHS]"));
        assert!(HVP_KERNEL_SOURCE.contains("action_g[MAX_MULTI_RHS]"));
        let mut cursor = 0usize;
        for marker in [
            "canonical calibration phase: InterceptFirst",
            "canonical calibration phase: InterceptSecond",
            "canonical calibration phase: PrimaryFirstAndInterceptSecond",
            "canonical calibration phase: PrimaryPairSecond",
            "canonical finalizer phase: ImplicitFirst",
            "canonical finalizer phase: ImplicitFirstComplete",
            "canonical finalizer phase: ImplicitSecond",
            "canonical finalizer phase: ObservedFirst",
            "canonical finalizer phase: ObservedScoreSensitivity",
            "canonical finalizer phase: ObservedSecond",
            "canonical finalizer phase: NegLogFirst",
        ] {
            let relative = source[cursor..]
                .find(marker)
                .unwrap_or_else(|| panic!("generated CUDA source omitted phase {marker}"));
            cursor += relative + marker.len();
        }
        assert!(
            source.len() < 40_000,
            "generated CUDA source unexpectedly bloated"
        );
    }

    // ── Phase-3 HVP / diagonal CPU oracles + GPU parity tests ────────────────

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

}
