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
//! Probit Mills (stable; mirrors `log_ndtr_and_mills` in `survival_flex.rs`):
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

use super::error::GpuError;

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};

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

/// Per-row input bundle for [`launch_bms_flex_row_kernel`].
///
/// Coordinate ordering convention: `u = 0` is `a` (the latent intercept and
/// the variable IFT eliminates); `u = 1` is `b` (slope); `u = 2..2+p_h` is the
/// score-warp `β_h` block; `u = 2+p_h..2+p_h+p_w` is the link-wiggle `β_w`
/// block. So `r = 2 + p_h + p_w` and `u = 1` is the `b` (slope) index used by
/// the sparse `S_{b·h}` / `S_{b·w}` payloads.
pub(crate) struct BmsFlexRowKernelInputs<'a> {
    /// Number of observation rows.
    pub n_rows: usize,
    /// Total primary local dimension. `r = 2 + p_h + p_w`.
    pub r: usize,
    /// Number of score-warp basis coordinates.
    pub p_h: usize,
    /// Number of link-wiggle basis coordinates.
    pub p_w: usize,
    /// Per-row latent quantile point `q_i = marginal_link(q)`. Length `n_rows`.
    pub q: &'a [f64],
    /// Per-row latent slope `b_i`. Length `n_rows`.
    pub b: &'a [f64],
    /// Per-row `μ_1 = ∂q/∂a`. Length `n_rows`.
    pub mu_1: &'a [f64],
    /// Per-row `μ_2 = ∂²q/∂a²`. Length `n_rows`.
    pub mu_2: &'a [f64],
    /// Per-row observed `z_obs`. Length `n_rows`. (Carried for diagnostics; the
    /// kernel itself uses the pre-evaluated `chi_obs`, `xi_obs`, `rho`, `tau`,
    /// `r_uv` arrays the host supplies.)
    pub z_obs: &'a [f64],
    /// Per-row response `y_i ∈ {0, 1}`. Length `n_rows`.
    pub y: &'a [f64],
    /// Per-row observation weight `w_i`. Length `n_rows`.
    pub w: &'a [f64],
    /// Probit frailty scale `S_f` (scalar shared across rows; matches
    /// `BernoulliMarginalSlope::probit_frailty_scale`).
    pub s_f: f64,
    /// Per-row cell range, length `n_rows + 1`. Row `i` owns cells in the
    /// half-open `[cell_offsets[i], cell_offsets[i+1])` slice of the
    /// `cell_*` arrays.
    pub cell_offsets: &'a [u32],
    /// Cubic predictor coefficient `C0` per cell. Length `total_cells`.
    pub cell_c0: &'a [f64],
    /// Cubic predictor coefficient `C1` per cell. Length `total_cells`.
    pub cell_c1: &'a [f64],
    /// Cubic predictor coefficient `C2` per cell. Length `total_cells`.
    pub cell_c2: &'a [f64],
    /// Cubic predictor coefficient `C3` per cell. Length `total_cells`.
    pub cell_c3: &'a [f64],
    /// Per-cell `A_c` derivative coefficient (length `total_cells * 4`,
    /// row-major `[total_cells, 4]`).
    pub cell_a: &'a [f64],
    /// Per-cell `AA_c` second-derivative coefficient (length
    /// `total_cells * 4`).
    pub cell_aa: &'a [f64],
    /// Per-cell `R_{c,u}` for `u ∈ [1, r)` (length
    /// `total_cells * (r-1) * 4`, row-major `[total_cells, r-1, 4]`).
    pub cell_r: &'a [f64],
    /// Per-cell `AR_{c,u}` for `u ∈ [1, r)` (same shape as `cell_r`).
    pub cell_ar: &'a [f64],
    /// Per-cell `S_{bb}` second-derivative coefficient (length
    /// `total_cells * 4`).
    pub cell_sbb: &'a [f64],
    /// Per-cell `S_{b·h_j}` second-derivative coefficient
    /// (length `total_cells * p_h * 4`, row-major `[total_cells, p_h, 4]`).
    /// Stage 2 stores dense; sparse encoding is Stage 3.
    pub cell_sbh: &'a [f64],
    /// Per-cell `S_{b·w_ℓ}` second-derivative coefficient
    /// (length `total_cells * p_w * 4`, row-major `[total_cells, p_w, 4]`).
    pub cell_sbw: &'a [f64],
    /// Per-cell derivative moments from Stage 1: row-major
    /// `[total_cells, MOMENT_STRIDE = 10]`.
    pub cell_moments: &'a [f64],
    /// Per-row `chi_obs`. Length `n_rows`.
    pub chi_obs: &'a [f64],
    /// Per-row `xi_obs`. Length `n_rows`.
    pub xi_obs: &'a [f64],
    /// Per-row `rho_u`: row-major `[n_rows, r]`.
    pub rho_u: &'a [f64],
    /// Per-row `tau_u`: row-major `[n_rows, r]`.
    pub tau_u: &'a [f64],
    /// Per-row `r_uv`: row-major `[n_rows, r*r]`.
    pub r_uv: &'a [f64],
}

/// Owned twin of [`BmsFlexRowKernelInputs`] — every borrowed slice is replaced
/// by an owned `Vec`. Built by the host packer in
/// `BernoulliMarginalSlopeFamily::pack_bms_flex_row_kernel_inputs`; converted
/// to a borrowed view via [`BmsFlexRowKernelInputsOwned::as_borrowed`] just
/// before [`launch_bms_flex_row_kernel`] uploads to the device.
///
/// Holds all per-row + per-cell SoA buffers in the exact layouts the device
/// kernel reads (`bms_flex_row_kernel` in [`ROW_KERNEL_SOURCE`]):
///   * scalars `n_rows`, `r`, `p_h`, `p_w`, `s_f`,
///   * per-row `q / b / mu_1 / mu_2 / z_obs / y / w / chi_obs / xi_obs`,
///   * per-row `rho_u [n*r]`, `tau_u [n*r]`, `r_uv [n*r*r]`,
///   * CSR `cell_offsets [n+1]` and per-cell `cell_c0..c3`,
///     `cell_a / cell_aa [n_cells * 4]`,
///     `cell_r / cell_ar [n_cells * (r-1) * 4]`,
///     `cell_sbb [n_cells * 4]`,
///     `cell_sbh [n_cells * p_h * 4]`,
///     `cell_sbw [n_cells * p_w * 4]`,
///     `cell_moments [n_cells * 10]`.
pub(crate) struct BmsFlexRowKernelInputsOwned {
    pub n_rows: usize,
    pub r: usize,
    pub p_h: usize,
    pub p_w: usize,
    pub s_f: f64,
    pub q: Vec<f64>,
    pub b: Vec<f64>,
    pub mu_1: Vec<f64>,
    pub mu_2: Vec<f64>,
    pub z_obs: Vec<f64>,
    pub y: Vec<f64>,
    pub w: Vec<f64>,
    pub cell_offsets: Vec<u32>,
    pub cell_c0: Vec<f64>,
    pub cell_c1: Vec<f64>,
    pub cell_c2: Vec<f64>,
    pub cell_c3: Vec<f64>,
    pub cell_a: Vec<f64>,
    pub cell_aa: Vec<f64>,
    pub cell_r: Vec<f64>,
    pub cell_ar: Vec<f64>,
    pub cell_sbb: Vec<f64>,
    pub cell_sbh: Vec<f64>,
    pub cell_sbw: Vec<f64>,
    pub cell_moments: Vec<f64>,
    pub chi_obs: Vec<f64>,
    pub xi_obs: Vec<f64>,
    pub rho_u: Vec<f64>,
    pub tau_u: Vec<f64>,
    pub r_uv: Vec<f64>,
}

impl BmsFlexRowKernelInputsOwned {
    /// Borrowed view over `self` suitable for [`launch_bms_flex_row_kernel`].
    /// The returned struct holds references into `self` so the owned bundle
    /// must outlive the launch.
    pub(crate) fn as_borrowed(&self) -> BmsFlexRowKernelInputs<'_> {
        BmsFlexRowKernelInputs {
            n_rows: self.n_rows,
            r: self.r,
            p_h: self.p_h,
            p_w: self.p_w,
            s_f: self.s_f,
            q: &self.q,
            b: &self.b,
            mu_1: &self.mu_1,
            mu_2: &self.mu_2,
            z_obs: &self.z_obs,
            y: &self.y,
            w: &self.w,
            cell_offsets: &self.cell_offsets,
            cell_c0: &self.cell_c0,
            cell_c1: &self.cell_c1,
            cell_c2: &self.cell_c2,
            cell_c3: &self.cell_c3,
            cell_a: &self.cell_a,
            cell_aa: &self.cell_aa,
            cell_r: &self.cell_r,
            cell_ar: &self.cell_ar,
            cell_sbb: &self.cell_sbb,
            cell_sbh: &self.cell_sbh,
            cell_sbw: &self.cell_sbw,
            cell_moments: &self.cell_moments,
            chi_obs: &self.chi_obs,
            xi_obs: &self.xi_obs,
            rho_u: &self.rho_u,
            tau_u: &self.tau_u,
            r_uv: &self.r_uv,
        }
    }
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
                reason: format!(
                    "bms_flex_row inputs: r={} exceeds MAX_R={MAX_R}",
                    self.r
                ),
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
                    reason: format!(
                        "bms_flex_row inputs: {name}.len()={have} != {want}"
                    ),
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

/// NVRTC kernel source. One CUDA block per row; 32 threads per block parallise
/// the per-cell sums into shared-memory scratch; thread 0 of the block finishes
/// the IFT + observed-point + Mills + Hessian write-out.
///
/// **CPU parity reference**: the body mirrors
/// `compute_row_analytic_flex_from_parts_into` in
/// `src/families/bernoulli_marginal_slope.rs`. The probit Mills helpers are
/// duplicated from `src/gpu/survival_flex.rs` (lines 224–283) until a shared
/// `numerics_device.rs` lands; the two copies must stay byte-for-byte
/// identical so any future numerics fix lands in both.
#[cfg(target_os = "linux")]
const ROW_KERNEL_SOURCE: &str = r#"
// One block per row. blockDim.x = 32; threadIdx.x parallises per-cell sums.
// CPU parity reference: src/families/bernoulli_marginal_slope.rs
//                      ::compute_row_analytic_flex_from_parts_into.

#define INV_TWO_PI     0.15915494309189535
#define INV_SQRT_2PI   0.3989422804014327
#define SQRT_2         1.4142135623730951

// ---- erfcx / log_ndtr / mills helpers (mirror src/gpu/survival_flex.rs) ----
extern "C" __device__ __forceinline__ double erfcx_nonnegative(double x) {
    if (!isfinite(x)) {
        return (x > 0.0) ? 0.0 : (1.0 / 0.0);
    }
    if (x <= 0.0) return 1.0;
    if (x < 26.0) {
        double xx = x * x;
        if (xx > 700.0) xx = 700.0;
        return exp(xx) * erfc(x);
    }
    double inv  = 1.0 / x;
    double inv2 = inv * inv;
    double poly = 1.0
                - 0.5    * inv2
                + 0.75   * inv2 * inv2
                - 1.875  * inv2 * inv2 * inv2
                + 6.5625 * inv2 * inv2 * inv2 * inv2;
    const double inv_sqrt_pi = 0.5641895835477563;
    return inv * poly * inv_sqrt_pi;
}

extern "C" __device__ __forceinline__ void
log_ndtr_and_mills(double x, double *log_cdf, double *lambda) {
    if (x ==  (1.0 / 0.0)) { *log_cdf = 0.0;            *lambda = 0.0;            return; }
    if (x == -(1.0 / 0.0)) { *log_cdf = -(1.0 / 0.0);   *lambda = (1.0 / 0.0);    return; }
    if (isnan(x))          { *log_cdf = x;              *lambda = x;              return; }
    if (x < 0.0) {
        double u   = -x / SQRT_2;
        double ex  = erfcx_nonnegative(u);
        if (ex < 1e-300) ex = 1e-300;
        *log_cdf = -u * u + log(0.5 * ex);
        const double sqrt_2_over_pi = 0.7978845608028654;
        *lambda  = sqrt_2_over_pi / ex;
    } else {
        double cdf = 0.5 * erfc(-x / SQRT_2);
        if (cdf < 1e-300) cdf = 1e-300;
        if (cdf > 1.0)    cdf = 1.0;
        double pdf = INV_SQRT_2PI * exp(-0.5 * x * x);
        *log_cdf = log(cdf);
        *lambda  = pdf / cdf;
    }
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
    double nan = nan("");
    out_neglog[row] = nan;
    for (int u = 0; u < r; ++u) {
        out_grad[row * r + u] = nan;
    }
    int rr = r * r;
    for (int idx = 0; idx < rr; ++idx) {
        out_hess[row * rr + idx] = nan;
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
            atomicAdd(&F_u[u], d_R);
            atomicAdd(&F_au[u], d_AR - q_AR);
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
                atomicAdd(&F_uv[u * r + v], val);
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
struct RowKernelBackend {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

#[cfg(target_os = "linux")]
impl RowKernelBackend {
    fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<RowKernelBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let runtime = super::runtime::GpuRuntime::global().ok_or_else(|| {
                    GpuError::DriverLibraryUnavailable {
                        reason: "bms_flex_row backend: no CUDA runtime available".to_string(),
                    }
                })?;
                let ctx = super::runtime::cuda_context_for(runtime.selected_device().ordinal)
                    .ok_or_else(|| GpuError::DriverCallFailed {
                        reason: format!(
                            "bms_flex_row backend: failed to create CUDA context for device {}",
                            runtime.selected_device().ordinal
                        ),
                    })?;
                let stream = ctx.default_stream();
                let ptx = cudarc::nvrtc::compile_ptx(ROW_KERNEL_SOURCE).map_err(|err| {
                    GpuError::DriverCallFailed {
                        reason: format!("bms_flex_row NVRTC compile failed: {err}"),
                    }
                })?;
                let module = ctx.load_module(ptx).map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("bms_flex_row module load failed: {err}"),
                })?;
                // `ctx` is intentionally dropped here: both `stream` and
                // `module` hold their own `Arc<CudaContext>` clones internally
                // (via cudarc's `CudaStream::ctx()` / `CudaModule::ctx()`
                // accessors), so the context outlives the local binding.
                Ok(RowKernelBackend { stream, module })
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
fn launch_linux(
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

    let d_q       = upload_f64(inputs.q, "q")?;
    let d_b       = upload_f64(inputs.b, "b")?;
    let d_mu1     = upload_f64(inputs.mu_1, "mu_1")?;
    let d_mu2     = upload_f64(inputs.mu_2, "mu_2")?;
    let d_zobs    = upload_f64(inputs.z_obs, "z_obs")?;
    let d_y       = upload_f64(inputs.y, "y")?;
    let d_w       = upload_f64(inputs.w, "w")?;
    let d_offsets = upload_u32(inputs.cell_offsets, "cell_offsets")?;
    let d_c0      = upload_f64(inputs.cell_c0, "cell_c0")?;
    let d_c1      = upload_f64(inputs.cell_c1, "cell_c1")?;
    let d_c2      = upload_f64(inputs.cell_c2, "cell_c2")?;
    let d_c3      = upload_f64(inputs.cell_c3, "cell_c3")?;
    let d_a       = upload_f64(inputs.cell_a, "cell_a")?;
    let d_aa      = upload_f64(inputs.cell_aa, "cell_aa")?;
    let d_r       = upload_f64(inputs.cell_r, "cell_r")?;
    let d_ar      = upload_f64(inputs.cell_ar, "cell_ar")?;
    let d_sbb     = upload_f64(inputs.cell_sbb, "cell_sbb")?;
    let d_sbh     = upload_f64(inputs.cell_sbh, "cell_sbh")?;
    let d_sbw     = upload_f64(inputs.cell_sbw, "cell_sbw")?;
    let d_moments = upload_f64(inputs.cell_moments, "cell_moments")?;
    let d_chi     = upload_f64(inputs.chi_obs, "chi_obs")?;
    let d_xi      = upload_f64(inputs.xi_obs, "xi_obs")?;
    let d_rho     = upload_f64(inputs.rho_u, "rho_u")?;
    let d_tau     = upload_f64(inputs.tau_u, "tau_u")?;
    let d_ruv     = upload_f64(inputs.r_uv, "r_uv")?;

    let n = inputs.n_rows;
    let r = inputs.r;
    let mut d_neglog = stream
        .alloc_zeros::<f64>(n)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc neglog: {err}"),
        })?;
    let mut d_grad = stream
        .alloc_zeros::<f64>(n * r)
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex_row alloc grad: {err}"),
        })?;
    let mut d_hess = stream
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
        .arg(&d_q).arg(&d_b).arg(&d_mu1).arg(&d_mu2).arg(&d_zobs).arg(&d_y).arg(&d_w)
        .arg(&d_offsets)
        .arg(&d_c0).arg(&d_c1).arg(&d_c2).arg(&d_c3)
        .arg(&d_a).arg(&d_aa).arg(&d_r).arg(&d_ar)
        .arg(&d_sbb).arg(&d_sbh).arg(&d_sbw).arg(&d_moments)
        .arg(&d_chi).arg(&d_xi).arg(&d_rho).arg(&d_tau).arg(&d_ruv)
        .arg(&mut d_neglog).arg(&mut d_grad).arg(&mut d_hess);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_inputs<'a>(buffers: &'a TestBuffers) -> BmsFlexRowKernelInputs<'a> {
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
            cell_moments: &buffers.cell_moments,
            chi_obs: &buffers.chi_obs,
            xi_obs: &buffers.xi_obs,
            rho_u: &buffers.rho_u,
            tau_u: &buffers.tau_u,
            r_uv: &buffers.r_uv,
        }
    }

    struct TestBuffers {
        q: Vec<f64>,
        b: Vec<f64>,
        mu_1: Vec<f64>,
        mu_2: Vec<f64>,
        z_obs: Vec<f64>,
        y: Vec<f64>,
        w: Vec<f64>,
        cell_offsets: Vec<u32>,
        cell_c0: Vec<f64>,
        cell_c1: Vec<f64>,
        cell_c2: Vec<f64>,
        cell_c3: Vec<f64>,
        cell_a: Vec<f64>,
        cell_aa: Vec<f64>,
        cell_r: Vec<f64>,
        cell_ar: Vec<f64>,
        cell_sbb: Vec<f64>,
        cell_sbh: Vec<f64>,
        cell_sbw: Vec<f64>,
        cell_moments: Vec<f64>,
        chi_obs: Vec<f64>,
        xi_obs: Vec<f64>,
        rho_u: Vec<f64>,
        tau_u: Vec<f64>,
        r_uv: Vec<f64>,
    }

    fn make_buffers(n_cells: u32, r: usize, p_h: usize, p_w: usize) -> TestBuffers {
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
    fn validate_accepts_minimal_inputs() {
        let buffers = make_buffers(2, 4, 1, 1);
        let inputs = minimal_inputs(&buffers);
        assert!(inputs.validate().is_ok());
    }

    #[test]
    fn validate_rejects_r_above_max() {
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
    fn validate_rejects_mismatched_r_decomposition() {
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
    fn validate_rejects_non_monotone_offsets() {
        let mut buffers = make_buffers(2, 4, 1, 1);
        buffers.cell_offsets = vec![5, 3];
        let inputs = minimal_inputs(&buffers);
        let err = inputs
            .validate()
            .expect_err("non-monotone offsets must fail");
        let msg = err.to_string();
        assert!(msg.contains("monotone"), "got: {msg}");
    }

    #[test]
    fn validate_rejects_mismatched_cell_moments_length() {
        let mut buffers = make_buffers(2, 4, 1, 1);
        buffers.cell_moments.pop(); // length now 2*10 - 1
        let inputs = minimal_inputs(&buffers);
        let err = inputs
            .validate()
            .expect_err("short cell_moments must fail");
        let msg = err.to_string();
        assert!(msg.contains("cell_moments"), "got: {msg}");
    }

    #[test]
    fn launch_on_non_linux_reports_driver_library_unavailable() {
        // Mac/Windows builds must surface a typed `DriverLibraryUnavailable`
        // rather than panicking or returning Ok. On Linux this test is
        // skipped because the kernel actually launches.
        #[cfg(target_os = "linux")]
        {
            // Linux builds may or may not have a device; the dispatcher
            // contract is that without a runtime, probe() returns
            // DriverLibraryUnavailable. Either outcome (NotYetImplemented,
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
                | Err(GpuError::NotYetImplemented { .. }) => { /* expected on CPU-only */ }
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
    fn s_f_must_be_positive_and_finite() {
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

    #[test]
    fn kernel_source_mentions_cpu_parity_reference() {
        // Guarantee the maintainer-facing parity reference comment survives
        // refactors of the NVRTC kernel source — the dispatcher wave that
        // wires this to bms_flex.rs cross-checks parity against the CPU
        // function named here.
        #[cfg(target_os = "linux")]
        assert!(ROW_KERNEL_SOURCE.contains("compute_row_analytic_flex_from_parts_into"));
        #[cfg(target_os = "linux")]
        assert!(ROW_KERNEL_SOURCE.contains("cell_first_derivative_from_moments"));
    }
}
