//! GPU device backend for survival marginal-slope FLEX row primitives.
//!
//! Block 8 roadmap (math team — `SurvivalMarginalSlope` analogue of
//! `bms_flex`):
//!
//!   1. **Rigid row kernel (this commit)**: NVRTC source covering the
//!      flex=false 4-primary subset `(q_0, q_1, q̇_1, g)`.  Mirrors
//!      `row_primary_closed_form` in `src/families/survival_marginal_slope.rs`
//!      term-for-term:
//!         η_0 = q_0·c + s·g·z      c   = √(1 + (s·g)²)
//!         η_1 = q_1·c + s·g·z      c₁  = s²·g/c
//!         a_1 = q̇_1·c              c₂  = s²/c³
//!      The row NLL is the standard probit survival kernel
//!         ℓ = w[(1-d)·(-log Φ(-η_1)) + log Φ(-η_0)
//!               - d·log φ(η_1) - d·log(a_1)],
//!      and the row gradient / Hessian come from the chain rule through
//!      `(η_0, η_1, a_1)`.  Sign convention matches the CPU: the host
//!      accumulator adds `+H_i` for the (positive-NLL) observed-information
//!      curvature and `+grad_i` for the NLL gradient.
//!   2. **Device cubic-cell runtime**: breakpoint union, finite-cell cubic
//!      coefficients, 384-pt Gauss-Legendre moment evaluator.  Validated
//!      against CPU moments before assembling the row likelihood.
//!   3. **Device intercept solve**: warm-started monotone root for the
//!      flex calibration equation `F(a) = ⟨Φ(-η(z;a))⟩ - Φ(-q) = 0`.
//!   4. **Full timepoint jet**: `a_u`, `a_uv` and the total η/χ
//!      derivatives needed for the row gradient & Hessian.
//!   5. **Row-primary G, H assembly** using Mills ratio λ and log Φ
//!      derivatives in the form `A(η) = log Φ(-η)`, `B(η) = -log Φ(-η)`.
//!   6. **Three GPU entry points** (`try_survival_flex_gradient`,
//!      `try_survival_flex_hvp`, `try_survival_flex_dense_hessian`) that
//!      all share the same per-row primitive.  Each returns `Ok(None)` for
//!      unsupported shapes (vector score, timewiggle, missing CUDA) so the
//!      dispatcher can fall through to CPU; only `gpu=force` against an
//!      unsupported route returns `Err`.
//!   7. **Hill-climb** (warp-cooperative GL loop, persistent warm-start,
//!      specialized dense-H kernel for m ≤ 64).
//!
//! Until later steps land the flex / aggregator code paths stay on CPU;
//! the rigid kernel from Step 1 is exposed for direct parity testing and
//! reused by the higher-level steps once they wire the host orchestration
//! around it.

use std::sync::OnceLock;

use ndarray::{Array1, Array2};

use super::error::GpuError;
use super::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

// ────────────────────────────────────────────────────────────────────────
// Policy entry points (parallels `bms_flex::row_primary_hessian_decision`).
// ────────────────────────────────────────────────────────────────────────

/// Decide whether the survival-flex GPU row primary path is eligible for
/// this fit's `(n, r)`.  `r == 0` (no primary jets to process) and below
/// the runtime row-kernel threshold force CPU.
#[must_use]
pub fn row_primary_hessian_decision(n: usize, r: usize) -> GpuDecision {
    let large_enough = super::runtime::GpuRuntime::global()
        .map(|runtime| n >= runtime.policy().row_kernel_min_n && r > 0)
        .unwrap_or(false);
    decide(
        GpuKernel::MarginalSlopeRows,
        SurvivalFlexGpuBackend::compiled(),
        large_enough,
    )
}

/// Same as [`row_primary_hessian_decision`] but turns
/// `gpu=force`-without-support into an `Err` string at the call site.
pub fn require_row_primary_hessian_supported(n: usize, r: usize) -> Result<GpuDecision, String> {
    let decision = row_primary_hessian_decision(n, r);
    decision.clone().log();
    decision.require_supported()?;
    Ok(decision)
}

// ────────────────────────────────────────────────────────────────────────
// Per-fit input descriptor.
// ────────────────────────────────────────────────────────────────────────

/// Inputs threaded into the three survival-flex entry points.  The struct
/// is intentionally additive: later steps append optional fields (per-row
/// time-design pointers, score-warp basis, link-deviation basis, cell
/// breakpoint tables, warm-start intercept slabs, …) without breaking the
/// Step-1 callers that only inspect `n`, `r`, `p`, `score_dim`, the rigid
/// row scalars (`q_0`, `q_1`, `q̇_1`, `g`, `z`) and the event/weight columns.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalFlexGpuRowInputs<'a> {
    /// Number of observations.
    pub n: usize,
    /// Primary local dimension (q_0 + q_1 + q̇_1 + g + score-warp + link-dev).
    pub r: usize,
    /// Total joint-parameter dimension `p` (sum of all block sizes).
    pub p: usize,
    /// Latent-score dimension `K`.  Step 1 + 6 require `K == 1` (scalar
    /// score); `K > 1` is an unsupported shape and the entry points return
    /// `Ok(None)` for it.
    pub score_dim: usize,
    /// Current β coefficient vector, length `p`, in joint-block order.
    pub beta: &'a [f64],
    /// Per-row entry quantile `q_0`, length `n`.
    pub q0: &'a [f64],
    /// Per-row exit quantile `q_1`, length `n`.
    pub q1: &'a [f64],
    /// Per-row exit-rate jacobian `q̇_1`, length `n`.  Rows with
    /// `q̇_1 < derivative_guard` (or non-finite) are rejected by the row
    /// primitive in line with the CPU `survival_derivative_guard_violated`.
    pub qd1: &'a [f64],
    /// Per-row latent score `z`, length `n`.  Scalar (K = 1) only in
    /// Step 1; the vector path lands in Step 4.
    pub z: &'a [f64],
    /// Per-row raw log-slope `g`, length `n`.
    pub g: &'a [f64],
    /// Observation weights, length `n`.
    pub weights: &'a [f64],
    /// Event indicator `d ∈ {0,1}`, length `n`.
    pub event: &'a [f64],
    /// `derivative_guard` for the monotonicity reject (matches CPU).
    pub derivative_guard: f64,
    /// `probit_scale` ≡ probit frailty scale `s` (matches CPU constant).
    pub probit_scale: f64,
}

impl<'a> SurvivalFlexGpuRowInputs<'a> {
    /// Shape-check every input array up front.  Kept on the struct so all
    /// three entry points reuse the same validation surface.
    fn validate(&self) -> Result<(), GpuError> {
        let n = self.n;
        let len_check = |label: &str, len: usize| -> Result<(), GpuError> {
            if len != n {
                return Err(GpuError::DriverCallFailed {
                    reason: format!("survival_flex inputs: {label}.len()={len} != n={n}"),
                });
            }
            Ok(())
        };
        if self.beta.len() != self.p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex inputs: beta.len()={} != p={}",
                    self.beta.len(),
                    self.p
                ),
            });
        }
        len_check("q0", self.q0.len())?;
        len_check("q1", self.q1.len())?;
        len_check("qd1", self.qd1.len())?;
        len_check("z", self.z.len())?;
        len_check("g", self.g.len())?;
        len_check("weights", self.weights.len())?;
        len_check("event", self.event.len())?;
        if !(self.derivative_guard.is_finite() && self.derivative_guard > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex inputs: derivative_guard must be positive and finite, got {}",
                    self.derivative_guard
                ),
            });
        }
        if !(self.probit_scale.is_finite() && self.probit_scale > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex inputs: probit_scale must be positive and finite, got {}",
                    self.probit_scale
                ),
            });
        }
        Ok(())
    }

    /// Is this row valid for the rigid kernel?  Matches the CPU helper
    /// `survival_derivative_guard_violated` byte-for-byte.
    #[inline]
    fn rigid_row_guard_violated(qd1: f64, derivative_guard: f64) -> bool {
        let tol = 256.0 * f64::EPSILON * (1.0 + qd1.abs().max(derivative_guard.abs()));
        !qd1.is_finite() | !derivative_guard.is_finite() | (qd1 + tol < derivative_guard)
    }
}

// ────────────────────────────────────────────────────────────────────────
// NVRTC source — Step 1 (rigid 4-primary row kernel).
//
// Kept as a single CUDA translation unit so the device-side helpers
// (`log_ndtr`, `mills_ratio`, `c_derivatives`, the row primitive itself)
// can inline freely.  Layout mirrors `row_primary_closed_form` so the
// term-by-term parity check is direct.
//
// Block 8 sibling note: the `log_ndtr` / `mills_ratio` blocks below are
// the same numerics the BMS-flex sibling needs.  Once both kernels land
// they will be factored out into a shared NVRTC header
// (`numerics_device.cuh`) — but that header doesn't exist yet, so for
// the Step-1 commit each backend ships the source inline.  Lifting it
// out is a no-op for the rigid kernel because NVRTC source caches by
// (source-text, options) and the eventual header will be compiled with
// the same options.  Tracked in the coordination message to
// `nvrtc-bms-flex`.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
const SURVIVAL_FLEX_RIGID_SOURCE: &str = r#"
// -------- numerics ----------------------------------------------------
// All math in double precision.  No --use_fast_math.
//
// `log_ndtr(x)` = log Φ(x).  For x < 0 uses the erfcx representation
//   log Φ(x) = -u² + log(½ · erfcx(u)),   u = -x / √2
// which preserves digits all the way into the deep left tail (matches
// the CPU `normal_logcdf`).  For x ≥ 0 falls back to log Φ(x).
//
// `mills_ratio(x)` = φ(x) / Φ(x).  For x < 0 uses the same erfcx so the
// ratio remains stable even when Φ(x) underflows.

#define INV_SQRT_2PI 0.3989422804014327
#define SQRT_2       1.4142135623730951
#define LN_TAU       1.8378770664093453  // log(2π)

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
    // 4-term asymptotic expansion of erfcx for large x.
    double inv  = 1.0 / x;
    double inv2 = inv * inv;
    double poly = 1.0
                - 0.5      * inv2
                + 0.75     * inv2 * inv2
                - 1.875    * inv2 * inv2 * inv2
                + 6.5625   * inv2 * inv2 * inv2 * inv2;
    const double inv_sqrt_pi = 0.5641895835477563; // 1/√π
    return inv * poly * inv_sqrt_pi;
}

extern "C" __device__ __forceinline__ double log_ndtr(double x) {
    if (x ==  (1.0 / 0.0)) return 0.0;
    if (x == -(1.0 / 0.0)) return -(1.0 / 0.0);
    if (isnan(x)) return x;
    if (x < 0.0) {
        double u   = -x / SQRT_2;
        double ex  = erfcx_nonnegative(u);
        if (ex < 1e-300) ex = 1e-300;
        return -u * u + log(0.5 * ex);
    } else {
        double c = 0.5 * erfc(-x / SQRT_2);
        if (c < 1e-300) c = 1e-300;
        if (c > 1.0)    c = 1.0;
        return log(c);
    }
}

// Returns (log Φ(x), φ(x)/Φ(x)).
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
        const double sqrt_2_over_pi = 0.7978845608028654; // √(2/π)
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

// `signed_probit_neglog_derivatives_up_to_fourth` first two outputs
// (k1, k2) — Step 1 only needs the first two; k3/k4 land in Step 4.
// Mirrors the CPU helper byte-for-byte.
extern "C" __device__ __forceinline__ void
signed_probit_neglog_k1k2(double signed_margin, double weight,
                          double *out_k1, double *out_k2) {
    if (weight == 0.0 || signed_margin == (1.0 / 0.0)) {
        *out_k1 = 0.0; *out_k2 = 0.0; return;
    }
    if (signed_margin == -(1.0 / 0.0)) {
        *out_k1 = -(1.0 / 0.0); *out_k2 = weight; return;
    }
    if (isnan(signed_margin)) {
        *out_k1 = signed_margin; *out_k2 = signed_margin; return;
    }
    double log_cdf, lambda;
    log_ndtr_and_mills(signed_margin, &log_cdf, &lambda);
    double k1 = -lambda;
    double k2 = lambda * (signed_margin + lambda);
    *out_k1 = weight * k1;
    *out_k2 = weight * k2;
}

// -------- rigid 4-primary row kernel ---------------------------------
//
// Each thread processes one row, producing
//   * its NLL contribution         (sum-reduced into out_loglik via atomic*)
//   * its 4-vector primary gradient (G_i ∈ ℝ⁴)
//   * its 4×4 primary Hessian      (H_i ∈ ℝ^{4×4}, upper triangle)
//
// For the Step-1 parity test the per-row arrays are written verbatim
// into device memory and copied back to host so the test can compare
// element-by-element with the CPU `row_primary_closed_form`.  The
// gradient/Hessian *coefficient-space pullback* (J^T G_i, J^T H_i J)
// lives one level up (Step 5/6); keeping the kernel scope tight here
// makes the Step-1 commit easy to audit and easy to test on V100
// without any host orchestration changes.
//
// Sign convention: matches the CPU rigid path exactly — `nll`, `grad`,
// `hess` are derivatives of the *negative* log-likelihood, i.e. the
// observed-information curvature in the dense Hessian.
//
// Layout per row `i` in [0, n):
//   out_nll[i]          = w * ((1-d)·(-log Φ(-η₁)) + log Φ(-η₀)
//                              - d·log φ(η₁) - d·log(a₁))
//   out_grad[i*4 + k]   = ∂NLL/∂x_k     for x = (q₀, q₁, q̇₁, g)
//   out_hess[i*16 + …]  = ∂²NLL/∂x_a∂x_b (row-major 4×4, full symmetric;
//                                          the host treats the upper
//                                          triangle as authoritative)
//   row_status[i]       = 0 on success
//                         1 if monotonicity guard tripped
//                         2 if a transformed-derivative non-finite
//
// `row_status` lets the host distinguish a clean "reject" (fallback to
// CPU for that row) from a kernel-side numerical error.
//
extern "C" __global__ void survival_flex_rigid_rows(
    const double * __restrict__ q0_arr,
    const double * __restrict__ q1_arr,
    const double * __restrict__ qd1_arr,
    const double * __restrict__ z_arr,
    const double * __restrict__ g_arr,
    const double * __restrict__ w_arr,
    const double * __restrict__ d_arr,
    double                       derivative_guard,
    double                       probit_scale,
    int                          n,
    double * __restrict__        out_nll,
    double * __restrict__        out_grad,    // length 4*n
    double * __restrict__        out_hess,    // length 16*n
    int    * __restrict__        row_status   // length n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double q0  = q0_arr[i];
    double q1  = q1_arr[i];
    double qd1 = qd1_arr[i];
    double z   = z_arr[i];
    double g   = g_arr[i];
    double w   = w_arr[i];
    double d   = d_arr[i];

    // Monotonicity guard — must match CPU `survival_derivative_guard_violated`
    // byte-for-byte so the GPU and CPU reject the same rows.
    double tol = 256.0 * 2.220446049250313e-16
                 * (1.0 + fmax(fabs(qd1), fabs(derivative_guard)));
    bool guard_violated = (!isfinite(qd1))
                       || (!isfinite(derivative_guard))
                       || ((qd1 + tol) < derivative_guard);
    if (guard_violated) {
        row_status[i] = 1;
        out_nll[i] = 0.0;
        for (int k = 0; k < 4;  ++k) out_grad[i * 4  + k] = 0.0;
        for (int k = 0; k < 16; ++k) out_hess[i * 16 + k] = 0.0;
        return;
    }

    // c(g) and its first two derivatives in raw g.
    double observed_g = probit_scale * g;
    double s2  = probit_scale * probit_scale;
    double c   = sqrt(1.0 + observed_g * observed_g);
    double c1  = s2 * g / c;
    double c2  = s2 / (c * c * c);

    // Linear predictors.
    double eta0 = q0  * c + observed_g * z;
    double eta1 = q1  * c + observed_g * z;
    double a1   = qd1 * c;

    if (!(isfinite(a1) && a1 > 0.0)) {
        row_status[i] = 2;
        out_nll[i] = 0.0;
        for (int k = 0; k < 4;  ++k) out_grad[i * 4  + k] = 0.0;
        for (int k = 0; k < 16; ++k) out_hess[i * 16 + k] = 0.0;
        return;
    }

    // NLL terms — match CPU formulation term by term.
    double log_cdf_neg_eta0;
    double lambda_neg_eta0;
    log_ndtr_and_mills(-eta0, &log_cdf_neg_eta0, &lambda_neg_eta0);
    double log_cdf_neg_eta1;
    double lambda_neg_eta1;
    log_ndtr_and_mills(-eta1, &log_cdf_neg_eta1, &lambda_neg_eta1);
    double log_phi_eta1 = -0.5 * (eta1 * eta1 + LN_TAU);
    double a1_floor     = fmax(a1, 1e-300);
    double log_a1       = log(a1_floor);

    double nll = w * ((1.0 - d) * (-log_cdf_neg_eta1)
                      + log_cdf_neg_eta0
                      - d * log_phi_eta1
                      - d * log_a1);

    // First/second derivatives of each NLL component w.r.t. its scalar
    // argument.  k1, k2 match `signed_probit_neglog_derivatives_up_to_fourth`.
    double e0_k1, e0_k2;
    signed_probit_neglog_k1k2(-eta0, -w, &e0_k1, &e0_k2);
    double e1_k1, e1_k2;
    signed_probit_neglog_k1k2(-eta1, w * (1.0 - d), &e1_k1, &e1_k2);
    double phi_u1 = w * d * eta1;
    double phi_u2 = w * d;
    // neglog_derivatives(a1) = (-1/a1, 1/a1², …)
    double inv     = 1.0 / a1_floor;
    double inv2    = inv * inv;
    double nl_u1   = -inv;
    double nl_u2   = inv2;
    double td_u1   = w * d * nl_u1;
    double td_u2   = w * d * nl_u2;

    // Chain rule to primary space — η₀(q₀, g), η₁(q₁, g), a₁(q̇₁, g).
    double deta0_dq0  = c;
    double deta0_dg   = q0  * c1 + probit_scale * z;
    double deta1_dq1  = c;
    double deta1_dg   = q1  * c1 + probit_scale * z;
    double dad1_dqd1  = c;
    double dad1_dg    = qd1 * c1;

    double u1_eta0 = -e0_k1;
    double u1_eta1 = -e1_k1 + phi_u1;
    double u1_ad1  = td_u1;

    double u2_eta0 = e0_k2;
    double u2_eta1 = e1_k2 + phi_u2;
    double u2_ad1  = td_u2;

    // Gradient (4-vector).
    double g0 = u1_eta0 * deta0_dq0;
    double g1 = u1_eta1 * deta1_dq1;
    double g2 = u1_ad1  * dad1_dqd1;
    double g3 = u1_eta0 * deta0_dg
              + u1_eta1 * deta1_dg
              + u1_ad1  * dad1_dg;

    // Hessian (4×4, full symmetric).
    double d2eta0_dq0dg = c1;
    double d2eta1_dq1dg = c1;
    double d2ad1_dqd1dg = c1;
    double d2eta0_dg2   = q0  * c2;
    double d2eta1_dg2   = q1  * c2;
    double d2ad1_dg2    = qd1 * c2;

    double H00 = u2_eta0 * deta0_dq0 * deta0_dq0;
    double H11 = u2_eta1 * deta1_dq1 * deta1_dq1;
    double H22 = u2_ad1  * dad1_dqd1 * dad1_dqd1;
    double H03 = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    double H13 = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    double H23 = u2_ad1  * dad1_dqd1 * dad1_dg  + u1_ad1  * d2ad1_dqd1dg;
    double H33 = u2_eta0 * deta0_dg  * deta0_dg
               + u1_eta0 * d2eta0_dg2
               + u2_eta1 * deta1_dg  * deta1_dg
               + u1_eta1 * d2eta1_dg2
               + u2_ad1  * dad1_dg   * dad1_dg
               + u1_ad1  * d2ad1_dg2;

    // Stores.
    out_nll[i]      = nll;
    out_grad[i*4+0] = g0;
    out_grad[i*4+1] = g1;
    out_grad[i*4+2] = g2;
    out_grad[i*4+3] = g3;
    // Row-major full 4×4 (mirror the symmetric pairs so host can index
    // either triangle).  The 0 entries are explicit so we never leak
    // uninitialised stack slots into the host parity check.
    out_hess[i*16 + 0*4+0] = H00;
    out_hess[i*16 + 0*4+1] = 0.0;
    out_hess[i*16 + 0*4+2] = 0.0;
    out_hess[i*16 + 0*4+3] = H03;
    out_hess[i*16 + 1*4+0] = 0.0;
    out_hess[i*16 + 1*4+1] = H11;
    out_hess[i*16 + 1*4+2] = 0.0;
    out_hess[i*16 + 1*4+3] = H13;
    out_hess[i*16 + 2*4+0] = 0.0;
    out_hess[i*16 + 2*4+1] = 0.0;
    out_hess[i*16 + 2*4+2] = H22;
    out_hess[i*16 + 2*4+3] = H23;
    out_hess[i*16 + 3*4+0] = H03;
    out_hess[i*16 + 3*4+1] = H13;
    out_hess[i*16 + 3*4+2] = H23;
    out_hess[i*16 + 3*4+3] = H33;

    row_status[i] = 0;
}
"#;

// ────────────────────────────────────────────────────────────────────────
// Process-wide backend (NVRTC compile, device arena, kernel launch).
// Mirrors `bms_flex::BmsFlexGpuBackend` so the host-side scaffolding
// (arena pooling, OnceLock module cache, mutex around alloc) is uniform
// across Block 8 / Block 1 / … kernels.
// ────────────────────────────────────────────────────────────────────────

#[must_use]
pub struct SurvivalFlexGpuBackend {
    #[cfg(target_os = "linux")]
    inner: SurvivalFlexGpuContextLinux,
}

#[cfg(target_os = "linux")]
struct SurvivalFlexGpuContextLinux {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// NVRTC-compiled module holding `survival_flex_rigid_rows` (and, in
    /// later steps, the flex kernels).  `OnceLock` so the compile runs
    /// exactly once per process and is shared by every dispatching thread.
    module: OnceLock<Arc<CudaModule>>,
    /// Reusable f64 device buffers keyed by power-of-two element-count
    /// buckets — the same bucketed arena `bms_flex` uses, so a fit that
    /// touches both backends does not double up on device memory.
    arena: Mutex<DeviceArena>,
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct DeviceArena {
    free: std::collections::HashMap<usize, Vec<cudarc::driver::CudaSlice<f64>>>,
}

#[cfg(target_os = "linux")]
impl DeviceArena {
    fn bucket_of(elements: usize) -> usize {
        elements.max(1).next_power_of_two()
    }

    fn alloc(
        &mut self,
        stream: &Arc<CudaStream>,
        elements: usize,
    ) -> Result<(usize, cudarc::driver::CudaSlice<f64>), GpuError> {
        let bucket = Self::bucket_of(elements);
        if let Some(bucket_vec) = self.free.get_mut(&bucket)
            && let Some(slot) = bucket_vec.pop()
        {
            return Ok((bucket, slot));
        }
        let fresh =
            stream
                .alloc_zeros::<f64>(bucket)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex arena alloc_zeros<{bucket}>: {err}"),
                })?;
        Ok((bucket, fresh))
    }

    fn release(&mut self, bucket: usize, slab: cudarc::driver::CudaSlice<f64>) {
        self.free.entry(bucket).or_default().push(slab);
    }
}

impl SurvivalFlexGpuBackend {
    /// True when this build can host the survival-flex GPU backend.
    /// Compiled on Linux only (Block 8 host is V100 / Linux).
    pub const fn compiled() -> bool {
        cfg!(target_os = "linux")
    }

    /// Lazily initialise the process-wide backend.  Eager NVRTC compile
    /// of the rigid kernel happens on the first call so compile errors
    /// surface here rather than at first dispatch.
    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<SurvivalFlexGpuBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                #[cfg(target_os = "linux")]
                {
                    Self::probe_linux()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(GpuError::DriverLibraryUnavailable {
                        reason: "survival_flex GPU backend is Linux-only".to_string(),
                    })
                }
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let runtime = super::runtime::GpuRuntime::global().ok_or_else(|| {
            GpuError::DriverLibraryUnavailable {
                reason: "survival_flex backend: no CUDA runtime available".to_string(),
            }
        })?;
        let ctx = super::runtime::cuda_context_for(runtime.selected_device().ordinal).ok_or_else(
            || GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex backend: failed to create CUDA context for device {}",
                    runtime.selected_device().ordinal
                ),
            },
        )?;
        let stream = ctx.default_stream();
        let backend = SurvivalFlexGpuBackend {
            inner: SurvivalFlexGpuContextLinux {
                ctx,
                stream,
                module: OnceLock::new(),
                arena: Mutex::new(DeviceArena::default()),
            },
        };
        backend.compile_rigid_module()?;
        Ok(backend)
    }

    /// NVRTC-compile (or fetch from cache) the rigid-kernel module.
    #[cfg(target_os = "linux")]
    fn compile_rigid_module(&self) -> Result<&Arc<CudaModule>, GpuError> {
        if let Some(existing) = self.inner.module.get() {
            return Ok(existing);
        }
        let ptx = cudarc::nvrtc::compile_ptx(SURVIVAL_FLEX_RIGID_SOURCE).map_err(|err| {
            GpuError::DriverCallFailed {
                reason: format!("survival_flex NVRTC compile failed: {err}"),
            }
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex module load failed: {err}"),
            })?;
        self.inner.module.set(module).ok();
        Ok(self
            .inner
            .module
            .get()
            .expect("module slot is populated after set"))
    }

    /// Round-trip the arena.  Mirrors `bms_flex` so the V100 smoke test
    /// has the same surface across Block 8 / sibling backends.
    #[cfg(target_os = "linux")]
    pub fn arena_round_trip(&self, elements: usize) -> Result<usize, GpuError> {
        let mut guard = self
            .inner
            .arena
            .lock()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex arena mutex poisoned: {err}"),
            })?;
        let (bucket, slab) = guard.alloc(&self.inner.stream, elements)?;
        guard.release(bucket, slab);
        Ok(bucket)
    }

    /// Short backend descriptor for logs.
    pub fn describe(&self) -> String {
        #[cfg(target_os = "linux")]
        {
            return format!(
                "survival_flex backend: device={:?} module_loaded={}",
                self.inner.ctx.name().ok(),
                self.inner.module.get().is_some()
            );
        }
        #[cfg(not(target_os = "linux"))]
        {
            "survival_flex backend: unavailable (not Linux)".to_string()
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Rigid per-row primitive (Step 1) — direct GPU launcher used by the
// Step-1 parity test and (in Step 6) by the host gradient/HVP/dense-H
// aggregators when `flex == false`.
//
// Returns the per-row outputs (NLL, grad, hess, row_status) so the test
// can compare element-wise with `row_primary_closed_form` on the host.
// The pullback into joint-β is intentionally NOT done here: Step 5 owns
// the chain rule into coefficient space and Step 6 wires the launcher
// into the dispatcher.  Keeping the launcher at the primary-space layer
// makes the math contract identical to the CPU rigid helper.
// ────────────────────────────────────────────────────────────────────────

/// Per-row output bundle from the rigid GPU kernel.
#[derive(Clone, Debug)]
pub struct SurvivalFlexRigidRowOutputs {
    /// Per-row NLL contribution, length `n`.
    pub nll: Vec<f64>,
    /// Per-row primary gradient, length `4*n` (row-major, primary order
    /// `(q₀, q₁, q̇₁, g)`).
    pub grad: Vec<f64>,
    /// Per-row primary Hessian, length `16*n` (row-major 4×4 symmetric
    /// per row).
    pub hess: Vec<f64>,
    /// Per-row status — `0` ok, `1` monotonicity guard, `2` non-finite
    /// transformed derivative.  Matches the CPU reject conditions.
    pub row_status: Vec<i32>,
}

/// Launch the rigid 4-primary row kernel.  Returns `Ok(None)` if the
/// backend is unsupported on this build (non-Linux / no CUDA runtime);
/// returns `Err` for genuine driver / shape failures so the caller can
/// distinguish "not eligible" from "eligible but broken".
pub fn try_rigid_row_primitive(
    inputs: SurvivalFlexGpuRowInputs<'_>,
) -> Result<Option<SurvivalFlexRigidRowOutputs>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        // Step 1 / 6 contract: scalar score only.  Vector score is a
        // higher milestone — return the "unsupported" sentinel so the
        // caller falls back to CPU cleanly.
        return Ok(None);
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    #[cfg(target_os = "linux")]
    {
        let backend = match SurvivalFlexGpuBackend::probe() {
            Ok(b) => b,
            Err(GpuError::DriverLibraryUnavailable { .. }) => return Ok(None),
            Err(other) => return Err(other),
        };
        Some(backend.launch_rigid_rows_linux(inputs)).transpose()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
impl SurvivalFlexGpuBackend {
    fn launch_rigid_rows_linux(
        &self,
        inputs: SurvivalFlexGpuRowInputs<'_>,
    ) -> Result<SurvivalFlexRigidRowOutputs, GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let module = self.compile_rigid_module()?;
        let func = module
            .load_function("survival_flex_rigid_rows")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex load_function: {err}"),
            })?;

        let n = inputs.n;
        let stream = &self.inner.stream;

        let d_q0 = stream
            .clone_htod(inputs.q0)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod q0: {err}"),
            })?;
        let d_q1 = stream
            .clone_htod(inputs.q1)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod q1: {err}"),
            })?;
        let d_qd1 = stream
            .clone_htod(inputs.qd1)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod qd1: {err}"),
            })?;
        let d_z = stream
            .clone_htod(inputs.z)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod z: {err}"),
            })?;
        let d_g = stream
            .clone_htod(inputs.g)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod g: {err}"),
            })?;
        let d_w = stream
            .clone_htod(inputs.weights)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod weights: {err}"),
            })?;
        let d_d = stream
            .clone_htod(inputs.event)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_stod event: {err}"),
            })?;

        let mut d_nll = stream
            .alloc_zeros::<f64>(n)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex alloc_zeros nll: {err}"),
            })?;
        let mut d_grad =
            stream
                .alloc_zeros::<f64>(4 * n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex alloc_zeros grad: {err}"),
                })?;
        let mut d_hess =
            stream
                .alloc_zeros::<f64>(16 * n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex alloc_zeros hess: {err}"),
                })?;
        let mut d_status =
            stream
                .alloc_zeros::<i32>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex alloc_zeros status: {err}"),
                })?;

        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let derivative_guard = inputs.derivative_guard;
        let probit_scale = inputs.probit_scale;
        let n_i32: i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex n={n} overflows i32"),
        })?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&d_q0)
            .arg(&d_q1)
            .arg(&d_qd1)
            .arg(&d_z)
            .arg(&d_g)
            .arg(&d_w)
            .arg(&d_d)
            .arg(&derivative_guard)
            .arg(&probit_scale)
            .arg(&n_i32)
            .arg(&mut d_nll)
            .arg(&mut d_grad)
            .arg(&mut d_hess)
            .arg(&mut d_status);
        // SAFETY: every argument is a typed device pointer / scalar
        // matching the kernel signature above, and grid/block cover
        // exactly `n` rows.  Out-of-range threads early-return.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex rigid launch: {err}"),
        })?;

        let nll = stream
            .clone_dtoh(&d_nll)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_dtov nll: {err}"),
            })?;
        let grad = stream
            .clone_dtoh(&d_grad)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_dtov grad: {err}"),
            })?;
        let hess = stream
            .clone_dtoh(&d_hess)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex memcpy_dtov hess: {err}"),
            })?;
        let row_status =
            stream
                .clone_dtoh(&d_status)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex memcpy_dtov status: {err}"),
                })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex synchronize: {err}"),
            })?;

        Ok(SurvivalFlexRigidRowOutputs {
            nll,
            grad,
            hess,
            row_status,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// Step 2 — survival-flex row-batched cubic-cell moment evaluator.
//
// Steps 3-6 need per-row derivative moments of the de-nested cubic
// correction `η(z) = c_0 + c_1·z + c_2·z² + c_3·z³` integrated against
// `exp(-q(z))` over each cell of the row's partition.  The CPU side
// builds these partitions via
// `survival_marginal_slope::denested_partition_cells` and then loops
// `evaluate_cell_moments` / `evaluate_cell_derivative_moments_uncached`
// per cell.
//
// This Step 2 wrapper takes a *flat* concatenation of per-row cells
// (with per-row start offsets), classifies them through the shared
// `cubic_cell::branch` classifier, and routes them in one shot through
// the existing GPU substrate (`cubic_cell::device::try_device_moments`):
//
//   * NonAffineFinite cells → 384-pt Gauss-Legendre warp-cooperative
//     kernel (the substrate's primary device path).
//   * Affine / AffineTail cells → CPU closed-form `T_n` recurrence
//     (substrate falls back per-cell, no warp divergence on GPU).
//
// Output is row-major `[total_cells, max_degree + 1]` moments plus a
// parallel status byte array and a `row_offsets` lookup so Step 3/4
// callers can index `row i → cells[row_offsets[i] .. row_offsets[i+1]]`.
//
// The wrapper does *not* duplicate any substrate logic: classification,
// device dispatch, status accounting all live in `cubic_cell`.  Its only
// jobs are (a) build the SoA cell list survival-flex needs and (b)
// expose a survival-shaped error surface.
// ────────────────────────────────────────────────────────────────────────

/// Per-row partition layout: a flat list of `(left, right, c0, c1, c2, c3)`
/// quadruples plus a `row_offsets` array of length `n + 1` so that
/// row `i`'s cells live at indices `row_offsets[i] .. row_offsets[i+1]`.
///
/// The survival-flex CPU path produces this layout naturally — see
/// `survival_marginal_slope::denested_partition_cells`.  Callers can
/// flatten the per-row `Vec<DenestedPartitionCell>` lists into this
/// shape with a single pass.
#[derive(Clone, Debug)]
pub(crate) struct SurvivalFlexRowCellsBatch<'a> {
    /// Total cell count = sum of per-row partition lengths.
    pub n_cells: usize,
    /// Number of rows (logical observations).
    pub n_rows: usize,
    /// Highest moment degree to evaluate, in `0..=24`.  Survival flex
    /// Hessian needs degree 24 for the `D_uv` cross terms; degree 9 is
    /// sufficient for value-only evaluations.
    pub max_degree: usize,
    /// Flat SoA cell quadruples, length `n_cells` each.
    pub left: &'a [f64],
    pub right: &'a [f64],
    pub c0: &'a [f64],
    pub c1: &'a [f64],
    pub c2: &'a [f64],
    pub c3: &'a [f64],
    /// Length `n_rows + 1`; row `i` owns cells `row_offsets[i] .. row_offsets[i+1]`.
    /// `row_offsets[0] == 0`, `row_offsets[n_rows] == n_cells`.
    pub row_offsets: &'a [usize],
}

/// Row-batched moment evaluation output.  Same shape as the substrate's
/// `HostMomentBatch` plus an echoed `row_offsets` so Step 3/4 can drive
/// per-row cumulative quadrature without re-flattening.
#[derive(Clone, Debug)]
pub(crate) struct SurvivalFlexRowMoments {
    /// Row-major `[n_cells, stride]` derivative moments, where
    /// `stride = max_degree + 1`.  Row for cell `k` is
    /// `moments[k * stride ..][..stride]`.
    pub moments: Vec<f64>,
    /// One status byte per cell, parallel to `moments` rows.  Values
    /// match `cubic_cell::CubicCellMomentStatus` byte-for-byte; non-zero
    /// codes mean the corresponding moment row is zeroed.
    pub status: Vec<u8>,
    /// `max_degree + 1`.
    pub stride: usize,
    /// Echoed from the input so Step 3/4 callers index per-row cells
    /// without threading the input back through.
    pub row_offsets: Vec<usize>,
}

impl<'a> SurvivalFlexRowCellsBatch<'a> {
    /// Shape-validate the batch.  Returns a `GpuError::DriverCallFailed`
    /// with a message naming the failing invariant so callers get a
    /// single error surface across the wrapper.
    fn validate(&self) -> Result<(), GpuError> {
        let nc = self.n_cells;
        let invariants: [(&str, usize); 6] = [
            ("left", self.left.len()),
            ("right", self.right.len()),
            ("c0", self.c0.len()),
            ("c1", self.c1.len()),
            ("c2", self.c2.len()),
            ("c3", self.c3.len()),
        ];
        for (label, len) in invariants {
            if len != nc {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex row-cells batch: {label}.len()={len} != n_cells={nc}"
                    ),
                });
            }
        }
        if self.row_offsets.len() != self.n_rows + 1 {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: row_offsets.len()={} != n_rows+1={}",
                    self.row_offsets.len(),
                    self.n_rows + 1
                ),
            });
        }
        if !self.row_offsets.is_empty()
            && (self.row_offsets[0] != 0 || self.row_offsets[self.n_rows] != nc)
        {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: row_offsets must start at 0 and end at \
                     n_cells={nc}, got [{}, …, {}]",
                    self.row_offsets[0], self.row_offsets[self.n_rows]
                ),
            });
        }
        for i in 0..self.n_rows {
            if self.row_offsets[i] > self.row_offsets[i + 1] {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex row-cells batch: row_offsets not monotone at i={i} \
                         ({} > {})",
                        self.row_offsets[i],
                        self.row_offsets[i + 1]
                    ),
                });
            }
        }
        if self.max_degree > super::cubic_cell::MAX_SUPPORTED_DEGREE {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: max_degree={} exceeds substrate \
                     MAX_SUPPORTED_DEGREE={}",
                    self.max_degree,
                    super::cubic_cell::MAX_SUPPORTED_DEGREE
                ),
            });
        }
        Ok(())
    }
}

/// Evaluate the derivative moments for every cell in `batch`.
///
/// Routes through the shared `cubic_cell` substrate so the survival-flex
/// path inherits the substrate's pre-classifier + 384-pt GL warp kernel
/// without any survival-specific kernel code.  The substrate itself
/// returns `Ok(None)` only on empty input; we surface that as
/// `Ok(None)` too so the dispatcher can short-circuit downstream Step 3
/// solves on rows that have no cells.
///
/// The substrate's host residency means this function works on every
/// platform: on Linux+CUDA the NonAffineFinite bucket runs on the
/// device, on macOS / CPU-only Linux every cell falls through to the
/// CPU evaluator that is the parity reference for the device kernel.
pub(crate) fn try_row_batched_cell_moments(
    batch: SurvivalFlexRowCellsBatch<'_>,
) -> Result<Option<SurvivalFlexRowMoments>, GpuError> {
    batch.validate()?;
    if batch.n_cells == 0 {
        return Ok(None);
    }

    // Build the substrate view in one pass.  The classification (Affine /
    // NonAffineFinite / AffineTail) is shared host code so doing it once
    // here lines up with what the substrate would re-derive internally;
    // we lift it out only because the substrate's `HostView` insists on
    // a `branches: &[GpuCellBranchTag]` matching the cell list.
    let mut cells = Vec::with_capacity(batch.n_cells);
    let mut branches = Vec::with_capacity(batch.n_cells);
    let mut prelim_status = Vec::with_capacity(batch.n_cells);
    for k in 0..batch.n_cells {
        let cell = super::cubic_cell::GpuDenestedCubicCell {
            left: batch.left[k],
            right: batch.right[k],
            c0: batch.c0[k],
            c1: batch.c1[k],
            c2: batch.c2[k],
            c3: batch.c3[k],
        };
        match super::cubic_cell::branch::classify_cell_for_gpu(cell) {
            Ok(tag) => {
                cells.push(cell);
                branches.push(tag);
                prelim_status.push(super::cubic_cell::CubicCellMomentStatus::Ok as u8);
            }
            Err(code) => {
                // Substrate would also reject this cell.  Keep a placeholder
                // in the input so per-cell indexing stays aligned; the
                // substrate will set the matching status code itself.
                cells.push(cell);
                // The substrate's classifier runs again and writes the
                // authoritative status; any tag here is fine because the
                // substrate's "host_tag != caller_tag" path also routes to
                // an error code, and the substrate's *own* classification
                // is the one that wins.  Use the cheapest stable tag.
                branches.push(super::cubic_cell::GpuCellBranchTag::Affine);
                prelim_status.push(code as u8);
            }
        }
    }

    let view = super::cubic_cell::CubicCellDerivativeMomentHostView {
        cells: &cells,
        branches: &branches,
        max_degree: batch.max_degree,
        residency: super::cubic_cell::CubicCellMomentResidency::Host,
    };
    let out = super::cubic_cell::try_build_cubic_cell_derivative_moments(view)?
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex row-cells batch: substrate returned None for n_cells={} > 0 \
                 (unexpected)",
                batch.n_cells
            ),
        })?;

    let super::cubic_cell::CubicCellDerivativeMomentOutput::Host {
        moments,
        mut status,
        stride,
    } = out;

    // Cells we pre-rejected (`prelim_status != Ok`) get a status code
    // from us if the substrate left them as Ok (it won't, because it
    // re-runs the classifier — but keeping this explicit guards against
    // a future substrate that trusts caller tags).
    for k in 0..batch.n_cells {
        if prelim_status[k] != super::cubic_cell::CubicCellMomentStatus::Ok as u8
            && status[k] == super::cubic_cell::CubicCellMomentStatus::Ok as u8
        {
            status[k] = prelim_status[k];
        }
    }

    Ok(Some(SurvivalFlexRowMoments {
        moments,
        status,
        stride,
        row_offsets: batch.row_offsets.to_vec(),
    }))
}

// ────────────────────────────────────────────────────────────────────────
// Three thin pullback entry points.  The bodies all currently return
// `Ok(None)` (the unsupported sentinel) because Steps 2–5 still need to
// land the flex / cubic-cell / intercept-solve infrastructure before
// the rigid kernel can be plugged into the full coefficient-space
// gradient / HVP / dense-H aggregator.  Keeping the shape stable here
// (option-typed, single shared input struct) lets Step 6 wire the
// dispatcher without re-touching the call sites.
// ────────────────────────────────────────────────────────────────────────

/// Evaluate the survival-flex negative log-likelihood and joint-β
/// gradient on the GPU.  Returns `Ok(None)` if the GPU path is
/// unsupported for this shape (caller falls back to CPU); returns
/// `Err` only when the request *is* supported but the driver failed.
pub fn try_survival_flex_gradient(
    inputs: SurvivalFlexGpuRowInputs<'_>,
) -> Result<Option<(f64, Array1<f64>)>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    // Coefficient-space pullback lives in Step 5/6.  Step 1 only owns
    // the rigid row primitive; the dispatcher hookup that turns the
    // per-row outputs into the joint-β gradient lands once the flex
    // cell runtime + intercept solve are in.
    Ok(None)
}

/// Evaluate the survival-flex joint-Hessian times a vector `v` on the
/// GPU.  Returns `H·v` ∈ ℝ^p, or `Ok(None)` for unsupported shapes.
pub fn try_survival_flex_hvp(
    inputs: SurvivalFlexGpuRowInputs<'_>,
    v: &[f64],
) -> Result<Option<Array1<f64>>, GpuError> {
    inputs.validate()?;
    if v.len() != inputs.p {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex try_hvp: v.len()={} != p={}",
                v.len(),
                inputs.p
            ),
        });
    }
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    Ok(None)
}

/// Assemble the dense survival-flex joint Hessian on the GPU.  Returns
/// a `p × p` row-major matrix, or `Ok(None)` for unsupported shapes.
pub fn try_survival_flex_dense_hessian(
    inputs: SurvivalFlexGpuRowInputs<'_>,
) -> Result<Option<Array2<f64>>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    Ok(None)
}

// ────────────────────────────────────────────────────────────────────────
// CPU reference for parity tests.
//
// Mirrors `row_primary_closed_form` in
// `src/families/survival_marginal_slope.rs` line-for-line for the scalar
// (K = 1) case.  Kept in this module — rather than calling the family
// helper across the crate boundary — so that the Step 1 GPU/CPU parity
// test exercises the *exact* algebra the device kernel runs (same
// numerical chain, same operator order) and surfaces sign / chain-rule
// regressions inside this file without traversing the survival family's
// import graph.  The two implementations are reconciled in Step 6 when
// the dispatcher hookup lands.
// ────────────────────────────────────────────────────────────────────────

/// Per-row CPU reference for the rigid 4-primary survival kernel.
/// Returns `(nll, grad[4], hess[4][4], status)` with the same status
/// codes as the GPU kernel (0 ok, 1 monotonicity, 2 non-finite a₁).
#[doc(hidden)]
pub fn cpu_reference_rigid_row(
    q0: f64,
    q1: f64,
    qd1: f64,
    z: f64,
    g: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4], i32) {
    if SurvivalFlexGpuRowInputs::rigid_row_guard_violated(qd1, derivative_guard) {
        return (0.0, [0.0; 4], [[0.0; 4]; 4], 1);
    }

    let observed_g = probit_scale * g;
    let s2 = probit_scale * probit_scale;
    let c = (1.0 + observed_g * observed_g).sqrt();
    let c1 = s2 * g / c;
    let c2 = s2 / (c * c * c);

    let eta0 = q0 * c + observed_g * z;
    let eta1 = q1 * c + observed_g * z;
    let a1 = qd1 * c;

    if !(a1.is_finite() && a1 > 0.0) {
        return (0.0, [0.0; 4], [[0.0; 4]; 4], 2);
    }

    let (log_cdf_neg_eta0, _l0) = crate::probability::signed_probit_logcdf_and_mills_ratio(-eta0);
    let (log_cdf_neg_eta1, _l1) = crate::probability::signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let log_a1 = a1.max(1e-300).ln();

    let nll =
        w * ((1.0 - d) * (-log_cdf_neg_eta1) + log_cdf_neg_eta0 - d * log_phi_eta1 - d * log_a1);

    // signed_probit_neglog_derivatives k1/k2 — replicated inline so this
    // reference does not depend on bernoulli_marginal_slope's pub(crate).
    let neglog_k1k2 = |signed_margin: f64, weight: f64| -> (f64, f64) {
        if weight == 0.0 || signed_margin == f64::INFINITY {
            return (0.0, 0.0);
        }
        if signed_margin == f64::NEG_INFINITY {
            return (f64::NEG_INFINITY, weight);
        }
        if signed_margin.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        let (_, lambda) = crate::probability::signed_probit_logcdf_and_mills_ratio(signed_margin);
        let k1 = -lambda;
        let k2 = lambda * (signed_margin + lambda);
        (weight * k1, weight * k2)
    };
    let (e0_k1, e0_k2) = neglog_k1k2(-eta0, -w);
    let (e1_k1, e1_k2) = neglog_k1k2(-eta1, w * (1.0 - d));
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    let inv = 1.0 / a1.max(1e-300);
    let nl_u1 = -inv;
    let nl_u2 = inv * inv;
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + probit_scale * z;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + probit_scale * z;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    let mut grad = [0.0_f64; 4];
    grad[0] = u1_eta0 * deta0_dq0;
    grad[1] = u1_eta1 * deta1_dq1;
    grad[2] = u1_ad1 * dad1_dqd1;
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg;

    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; 4]; 4];
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg
        + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg
        + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg
        + u1_ad1 * d2ad1_dg2;

    (nll, grad, hess, 0)
}

// ────────────────────────────────────────────────────────────────────────
// Tests.  Run via:
//   cargo test -p gam survival_flex_gpu -- --nocapture 2>&1 | tee /tmp/sv.log
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod survival_flex_gpu_tests {
    use super::*;

    fn make_inputs<'a>(
        n: usize,
        q0: &'a [f64],
        q1: &'a [f64],
        qd1: &'a [f64],
        z: &'a [f64],
        g: &'a [f64],
        w: &'a [f64],
        d: &'a [f64],
        beta: &'a [f64],
    ) -> SurvivalFlexGpuRowInputs<'a> {
        SurvivalFlexGpuRowInputs {
            n,
            r: 4,
            p: beta.len(),
            score_dim: 1,
            beta,
            q0,
            q1,
            qd1,
            z,
            g,
            weights: w,
            event: d,
            derivative_guard: 1e-6,
            probit_scale: 1.0,
        }
    }

    #[test]
    fn survival_flex_gpu_policy_decision_is_explicit() {
        let decision = row_primary_hessian_decision(50_000, 4);
        assert_eq!(decision.kernel, GpuKernel::MarginalSlopeRows);
    }

    #[test]
    fn survival_flex_gpu_gradient_returns_none_until_step6_lands() {
        let n = 4;
        let beta = vec![0.0_f64; 8];
        let q0 = vec![-1.0; n];
        let q1 = vec![0.5; n];
        let qd1 = vec![1.2; n];
        let z = vec![0.0; n];
        let g = vec![0.1; n];
        let w = vec![1.0; n];
        let d = vec![0.0, 1.0, 0.0, 1.0];
        let inputs = make_inputs(n, &q0, &q1, &qd1, &z, &g, &w, &d, &beta);
        match try_survival_flex_gradient(inputs) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("survival_flex gradient should be None until Step 6 lands"),
            Err(err) => panic!("unexpected error from survival_flex gradient: {err:?}"),
        }
    }

    #[test]
    fn survival_flex_gpu_hvp_rejects_wrong_v_length() {
        let n = 2;
        let beta = vec![0.0_f64; 4];
        let q0 = vec![-1.0; n];
        let q1 = vec![0.5; n];
        let qd1 = vec![1.2; n];
        let z = vec![0.0; n];
        let g = vec![0.1; n];
        let w = vec![1.0; n];
        let d = vec![0.0; n];
        let inputs = make_inputs(n, &q0, &q1, &qd1, &z, &g, &w, &d, &beta);
        let v_wrong = vec![0.0; beta.len() + 1];
        match try_survival_flex_hvp(inputs, &v_wrong) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(reason.contains("v.len()"), "reason was: {reason}");
            }
            other => panic!("expected DriverCallFailed for wrong v length, got {other:?}"),
        }
    }

    #[test]
    fn survival_flex_gpu_dense_hessian_returns_none_for_vector_score() {
        let n = 2;
        let beta = vec![0.0_f64; 4];
        let q0 = vec![-1.0; n];
        let q1 = vec![0.5; n];
        let qd1 = vec![1.2; n];
        let z = vec![0.0; n];
        let g = vec![0.1; n];
        let w = vec![1.0; n];
        let d = vec![0.0; n];
        let mut inputs = make_inputs(n, &q0, &q1, &qd1, &z, &g, &w, &d, &beta);
        inputs.score_dim = 2;
        match try_survival_flex_dense_hessian(inputs) {
            Ok(None) => {}
            other => panic!("expected None for vector score (K>1), got {other:?}"),
        }
    }

    #[test]
    fn survival_flex_gpu_rigid_primitive_returns_none_on_non_linux() {
        // On Linux + V100 this exercises the full NVRTC + launch path
        // (covered by the V100 parity suite); on macOS / CPU-only Linux
        // builds the call must return Ok(None) so the dispatcher can
        // fall back to CPU cleanly.
        let n = 2;
        let beta = vec![0.0_f64; 4];
        let q0 = vec![-1.0; n];
        let q1 = vec![0.5; n];
        let qd1 = vec![1.2; n];
        let z = vec![0.0; n];
        let g = vec![0.1; n];
        let w = vec![1.0; n];
        let d = vec![0.0; n];
        let inputs = make_inputs(n, &q0, &q1, &qd1, &z, &g, &w, &d, &beta);
        // On macOS this is `compiled() == false` so we must get Ok(None).
        if !SurvivalFlexGpuBackend::compiled() {
            match try_rigid_row_primitive(inputs) {
                Ok(None) => {}
                other => panic!("expected None on non-Linux build, got {other:?}"),
            }
        }
    }

    /// Eight diverse single-row scenarios covering the cross-product of
    /// (event=0/1) × (deep-left-tail q / mid q) × (sub-/super-unit slope).
    /// On a CUDA host (V100) this checks GPU vs CPU bit-pattern within
    /// 1e-10 NLL / 1e-8 grad/Hess per the Block 8 validation contract.
    /// On a CPU-only host the test asserts the CPU reference is finite
    /// for every scenario and the kernel returns `Ok(None)` (unsupported).
    #[test]
    fn survival_flex_gpu_rigid_matches_cpu_reference() {
        // (q0, q1, qd1, z, g, w, d) — tail / interior / event mix.
        let cases: [(f64, f64, f64, f64, f64, f64, f64); 8] = [
            (-2.0, -0.5, 1.30, 0.10, 0.20, 1.0, 0.0),
            (-2.0, -0.5, 1.30, 0.10, 0.20, 1.0, 1.0),
            (-8.0, -6.0, 1.50, -0.30, 0.05, 0.7, 0.0),
            (-8.0, -6.0, 1.50, -0.30, 0.05, 0.7, 1.0),
            (0.5, 1.2, 0.80, 0.40, -0.10, 1.2, 0.0),
            (0.5, 1.2, 0.80, 0.40, -0.10, 1.2, 1.0),
            (-1.5, 0.7, 1.05, 0.00, 0.50, 1.0, 1.0),
            (3.0, 5.0, 2.10, 0.20, 0.30, 1.0, 0.0),
        ];
        let derivative_guard = 1e-6;
        let probit_scale = 1.0;

        let n = cases.len();
        let q0: Vec<f64> = cases.iter().map(|c| c.0).collect();
        let q1: Vec<f64> = cases.iter().map(|c| c.1).collect();
        let qd1: Vec<f64> = cases.iter().map(|c| c.2).collect();
        let z: Vec<f64> = cases.iter().map(|c| c.3).collect();
        let g: Vec<f64> = cases.iter().map(|c| c.4).collect();
        let w: Vec<f64> = cases.iter().map(|c| c.5).collect();
        let d: Vec<f64> = cases.iter().map(|c| c.6).collect();
        let beta = vec![0.0_f64; 4];
        let mut inputs = make_inputs(n, &q0, &q1, &qd1, &z, &g, &w, &d, &beta);
        inputs.derivative_guard = derivative_guard;
        inputs.probit_scale = probit_scale;

        // CPU reference, for every row.
        let cpu_results: Vec<(f64, [f64; 4], [[f64; 4]; 4], i32)> = cases
            .iter()
            .map(|(q0, q1, qd1, z, g, w, d)| {
                cpu_reference_rigid_row(
                    *q0,
                    *q1,
                    *qd1,
                    *z,
                    *g,
                    *w,
                    *d,
                    derivative_guard,
                    probit_scale,
                )
            })
            .collect();
        for (i, (nll, grad, _hess, status)) in cpu_results.iter().enumerate() {
            assert!(nll.is_finite(), "row {i}: cpu nll non-finite ({nll})");
            assert_eq!(*status, 0, "row {i}: cpu status non-zero ({status})");
            for k in 0..4 {
                assert!(
                    grad[k].is_finite(),
                    "row {i}: cpu grad[{k}] non-finite ({})",
                    grad[k]
                );
            }
        }

        match try_rigid_row_primitive(inputs) {
            Ok(Some(out)) => {
                // GPU path actually ran (CUDA host).  Element-wise parity
                // check against the CPU reference at the contract tolerance.
                for (i, (cpu_nll, cpu_grad, cpu_hess, cpu_status)) in cpu_results.iter().enumerate()
                {
                    assert_eq!(out.row_status[i], *cpu_status, "row {i} status mismatch");
                    let gpu_nll = out.nll[i];
                    let nll_err = (gpu_nll - cpu_nll).abs();
                    assert!(
                        nll_err <= 1e-10 * (1.0 + cpu_nll.abs()),
                        "row {i}: nll parity violation gpu={gpu_nll} cpu={cpu_nll} err={nll_err}"
                    );
                    for k in 0..4 {
                        let gpu_g = out.grad[i * 4 + k];
                        let g_err = (gpu_g - cpu_grad[k]).abs();
                        assert!(
                            g_err <= 1e-8 * (1.0 + cpu_grad[k].abs()),
                            "row {i}: grad[{k}] parity violation gpu={gpu_g} cpu={} err={g_err}",
                            cpu_grad[k]
                        );
                    }
                    for a in 0..4 {
                        for b in 0..4 {
                            let gpu_h = out.hess[i * 16 + a * 4 + b];
                            let h_err = (gpu_h - cpu_hess[a][b]).abs();
                            assert!(
                                h_err <= 1e-8 * (1.0 + cpu_hess[a][b].abs()),
                                "row {i}: hess[{a}][{b}] parity violation gpu={gpu_h} cpu={} err={h_err}",
                                cpu_hess[a][b]
                            );
                        }
                    }
                }
            }
            Ok(None) => {
                // Non-CUDA host: confirm the CPU reference at least
                // produces finite values across every scenario so the
                // V100 parity check has a known-good target to land on.
            }
            Err(err) => panic!("survival_flex rigid kernel failed: {err:?}"),
        }
    }

    #[test]
    fn survival_flex_gpu_rigid_primitive_validates_derivative_guard() {
        let n = 1;
        let beta = vec![0.0_f64; 4];
        let q0 = vec![0.0];
        let q1 = vec![0.0];
        let qd1 = vec![1.0];
        let z = vec![0.0];
        let g = vec![0.0];
        let w = vec![1.0];
        let d = vec![0.0];
        let mut inputs = make_inputs(n, &q0, &q1, &qd1, &z, &g, &w, &d, &beta);
        inputs.derivative_guard = -1.0; // invalid
        match try_rigid_row_primitive(inputs) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(reason.contains("derivative_guard"), "got: {reason}");
            }
            other => panic!("expected DriverCallFailed for invalid guard, got {other:?}"),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 2 — row-batched cubic-cell moment wrapper.  Validates the
    // GPU substrate output element-wise against the CPU evaluator at the
    // survival-flex high-water mark `max_degree = 24`, across a row mix
    // covering finite-affine, non-affine-quartic, non-affine-sextic, and
    // affine-tail cells (the four branches the substrate handles).
    // ────────────────────────────────────────────────────────────────────

    /// Reference moments via the CPU evaluator (parity target for both
    /// the host substrate and the device kernel).  Returns the cell's
    /// `max_degree + 1` derivative moments.
    fn cpu_cell_moments(
        left: f64,
        right: f64,
        c0: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        max_degree: usize,
    ) -> Vec<f64> {
        let cpu_cell = crate::families::cubic_cell_kernel::DenestedCubicCell {
            left,
            right,
            c0,
            c1,
            c2,
            c3,
        };
        let state = crate::families::cubic_cell_kernel::evaluate_cell_derivative_moments_uncached(
            cpu_cell, max_degree,
        )
        .expect("cpu cell-derivative-moments reference");
        state.moments
    }

    #[test]
    fn survival_flex_row_batched_cells_validates_layout() {
        let left = [0.0_f64];
        let right = [1.0];
        let c0 = [0.0];
        let c1 = [0.0];
        let c2 = [0.0];
        let c3 = [0.0];
        let row_offsets = [0usize, 1];
        let bad_batch = SurvivalFlexRowCellsBatch {
            n_cells: 1,
            n_rows: 1,
            max_degree: 25, // exceeds MAX_SUPPORTED_DEGREE = 24
            left: &left,
            right: &right,
            c0: &c0,
            c1: &c1,
            c2: &c2,
            c3: &c3,
            row_offsets: &row_offsets,
        };
        match try_row_batched_cell_moments(bad_batch) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(
                    reason.contains("MAX_SUPPORTED_DEGREE"),
                    "expected degree-bound error, got: {reason}"
                );
            }
            other => panic!("expected validation error for degree=25, got {other:?}"),
        }
    }

    #[test]
    fn survival_flex_row_batched_cells_empty_returns_none() {
        let batch = SurvivalFlexRowCellsBatch {
            n_cells: 0,
            n_rows: 0,
            max_degree: 9,
            left: &[],
            right: &[],
            c0: &[],
            c1: &[],
            c2: &[],
            c3: &[],
            row_offsets: &[0usize],
        };
        match try_row_batched_cell_moments(batch) {
            Ok(None) => {}
            other => panic!("expected Ok(None) for empty batch, got {other:?}"),
        }
    }

    /// Three-row batch hitting every branch the substrate knows about,
    /// evaluated at `max_degree = 24` (the survival-flex Hessian
    /// high-water mark used by `D_uv` cross terms).  Element-wise parity
    /// against the CPU evaluator at relative tolerance 1e-10 — same bar
    /// the substrate's own d9 / d21 parity tests use.
    #[test]
    fn survival_flex_row_batched_cells_matches_cpu_at_degree_24() {
        // Row 0: a finite-affine cell on [-1.5, 0.0] (c2 = c3 = 0).
        // Row 1: a non-affine quartic on [-0.8, 0.3] (c2 ≠ 0, c3 = 0)
        //        plus a non-affine sextic on [0.3, 1.4]    (c2 ≠ 0, c3 ≠ 0).
        // Row 2: a whole-line affine tail (c2 = c3 = 0).
        // Together: 1 + 2 + 1 = 4 cells covering Affine / NonAffineFinite
        // (both subbranches) / AffineTail.
        let left   = [-1.5_f64, -0.8, 0.3, f64::NEG_INFINITY];
        let right  = [ 0.0_f64,  0.3, 1.4, f64::INFINITY    ];
        let c0     = [ 0.15_f64,-0.20, 0.10, 0.05];
        let c1     = [-0.30_f64, 0.45,-0.20,-0.10];
        let c2     = [ 0.00_f64, 0.35, 0.25, 0.00];
        let c3     = [ 0.00_f64, 0.00, 0.18, 0.00];
        let row_offsets = [0usize, 1, 3, 4];
        let max_degree = 24;
        let batch = SurvivalFlexRowCellsBatch {
            n_cells: 4,
            n_rows: 3,
            max_degree,
            left: &left,
            right: &right,
            c0: &c0,
            c1: &c1,
            c2: &c2,
            c3: &c3,
            row_offsets: &row_offsets,
        };
        let out = try_row_batched_cell_moments(batch)
            .expect("substrate succeeds on a valid batch")
            .expect("non-empty batch returns Some");
        assert_eq!(out.stride, max_degree + 1);
        assert_eq!(out.row_offsets, vec![0usize, 1, 3, 4]);
        assert_eq!(out.status.len(), 4);
        assert_eq!(out.moments.len(), 4 * out.stride);

        // Every cell should classify cleanly (Affine / NonAffineFinite /
        // AffineTail) and the moments should match the CPU evaluator.
        // The CPU evaluator handles all three branches; we don't need to
        // distinguish here — element-wise parity covers everything.
        for k in 0..4 {
            assert_eq!(
                out.status[k],
                super::super::cubic_cell::CubicCellMomentStatus::Ok as u8,
                "cell {k}: non-OK status 0x{:02x}",
                out.status[k]
            );
            let cpu = cpu_cell_moments(left[k], right[k], c0[k], c1[k], c2[k], c3[k], max_degree);
            let row = &out.moments[k * out.stride..(k + 1) * out.stride];
            assert_eq!(row.len(), cpu.len());
            for (j, (&got, &want)) in row.iter().zip(cpu.iter()).enumerate() {
                let denom = want.abs().max(1.0);
                let rel = (got - want).abs() / denom;
                assert!(
                    rel <= 1e-10,
                    "cell {k} moment {j}: got={got:.17e} want={want:.17e} rel={rel:.3e}"
                );
            }
        }
    }
}
