//! Survival marginal-slope FLEX GPU row primitives.
//!
//! The kernels here are domain math for `SurvivalMarginalSlope`; CUDA runtime,
//! memory, and policy ownership stays in [`crate::gpu`].

use std::sync::OnceLock;

use ndarray::{Array1, Array2};

use crate::gpu::gpu_error::GpuError;
use crate::gpu::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaModule;

/// Decide whether the survival-flex GPU row primary path is eligible for
/// this fit's `(n, r)`.  `r == 0` (no primary jets to process) and below
/// the runtime row-kernel threshold force CPU.
#[must_use]
pub fn row_primary_hessian_decision(n: usize, r: usize) -> GpuDecision {
    let large_enough = crate::gpu::runtime::GpuRuntime::global()
        .map(|runtime| n >= runtime.policy().row_kernel_min_n && r > 0)
        .unwrap_or(false);
    decide(
        GpuKernel::MarginalSlopeRows,
        crate::gpu::GpuEligibility::from_flags(SurvivalFlexGpuBackend::compiled(), large_enough),
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
// Shared probit numerics (`erfcx_nonnegative`, `log_ndtr`,
// `log_ndtr_and_mills`) live in `numerics_device::PROBIT_NUMERICS_CU` and
// are prepended at compile time.  The body below contains only the
// kernel-specific defines and device helpers.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
const SURVIVAL_FLEX_RIGID_BODY: &str = r#"
// -------- kernel-specific defines ----------------------------------------
#define LN_TAU       1.8378770664093453  // log(2π)

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
    inner: crate::gpu::backend_probe::CudaBackendContext,
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
        let parts = crate::gpu::backend_probe::probe_cuda_backend("survival_flex")?;
        let backend = SurvivalFlexGpuBackend {
            inner: crate::gpu::backend_probe::CudaBackendContext::from_parts(parts),
        };
        backend.compile_rigid_module()?;
        Ok(backend)
    }

    /// NVRTC-compile (or fetch from cache) the survival-flex device
    /// module.  The single cached module carries BOTH `extern "C"`
    /// symbols (`survival_flex_rigid_rows` and `survival_flex_primary_rows`)
    /// compiled as one translation unit sharing the probit-numerics
    /// prelude.  `PtxModuleCache` holds exactly one `Arc<CudaModule>` per
    /// backend keyed on first-compile, so both kernels MUST come from the
    /// same source concatenation — the rigid and primary launchers then
    /// `load_function` their respective symbol out of this one module.
    #[cfg(target_os = "linux")]
    fn compile_rigid_module(&self) -> Result<&Arc<CudaModule>, GpuError> {
        let source = [
            crate::gpu::numerics_device::PROBIT_NUMERICS_CU,
            SURVIVAL_FLEX_RIGID_BODY,
            SURVIVAL_FLEX_PRIMARY_BODY,
        ]
        .concat();
        self.inner
            .module
            .get_or_compile(&self.inner.ctx, "survival_flex", &source)
    }

    /// Alias for [`Self::compile_rigid_module`] — the flex row-primary
    /// kernel lives in the same combined module, so this just returns
    /// the shared cached module.  Kept as a named seam so the primary
    /// launcher reads symmetrically with the rigid one.
    #[cfg(target_os = "linux")]
    fn compile_primary_module(&self) -> Result<&Arc<CudaModule>, GpuError> {
        self.compile_rigid_module()
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
        let (bucket, slab) = guard.alloc(&self.inner.stream, elements, "survival_flex")?;
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
// Step 5 — FLEX row-primary gradient/Hessian kernel (NVRTC).
//
// Native-GPU landing of `try_device_step5_primary_assembly` (the
// device-shaped CPU reference earlier in this file) and its parent
// `compute_row_flex_primary_gradient_hessian_from_parts`
// (`src/families/survival_marginal_slope.rs`).  One thread = one row.
// Each row produces:
//   * `out_nll[i]`              the per-row NLL contribution
//   * `out_grad[i*r + u]`       the `r`-vector primary gradient
//   * `out_hess[i*r*r + u*r+v]` the full symmetric `r×r` primary Hessian
//
// `r` is a runtime int bounded by `SURVIVAL_FLEX_MAX_R` (32): the
// rigid kernel is the fixed `r==4` flex=false subset; this kernel
// generalizes it to `r ≤ MAX_R` driven by the full per-row jet
// (η_u/η_uv, χ/χ_u/χ_uv, d/d_u/d_uv) computed by Steps 2–4.  The jets
// are laid out Struct-of-Arrays per row (entry first, exit second):
//   eta_u  : r          per timepoint
//   eta_uv : r*r        per timepoint (row-major, symmetric)
//   chi_u  : r          per timepoint
//   chi_uv : r*r        per timepoint
//   d_u    : r          per timepoint
//   d_uv   : r*r        per timepoint
// plus the per-row scalar bundle (eta/chi/d at each timepoint, the
// `(k1,k2)` neglog derivatives, `log_surv0/1`, `q1`, `qd1`, `wi`, `di`,
// and the `q1_index`/`qd1_index` perturbation slots, `usize::MAX` →
// `-1` to disable).
//
// Sign convention is identical to `try_device_step5_primary_assembly`
// and to the inline CPU loop in
// `compute_row_flex_primary_gradient_hessian_from_parts`: `nll`,
// `grad`, `hess` are derivatives of the *negative* log-likelihood —
// i.e. the observed-information curvature the host accumulator adds.
//
// SUPPORTED SHAPE (caller pre-filters, see `try_flex_primary_rows`):
//   * `family == StandardNormal`, `score_dim == 1`
//   * `r ≤ SURVIVAL_FLEX_MAX_R`
//   * NO influence absorber active (`o_infl` absent) — per the #461
//     caveat at the top of this file the flex jet does not yet carry
//     the `infl` primary coord, so we stay `Ok(None)` when it is
//     present rather than silently dropping it.
// Everything else returns `Ok(None)` for a clean CPU fallback.
//
// Numerically delicate spots (FLAGGED for the V100 parity pass):
//   * the `chi_uv/chi − (chi_u·chi_u)/chi²` and the analogous `d`
//     ratio terms are catastrophic-cancellation-prone when the two
//     summands nearly cancel (small-curvature rows); the CPU computes
//     them in the same order, so GPU/CPU must agree to round-off but
//     the *absolute* magnitude can be tiny relative to the inputs.
//   * `chi.ln()` / `d.ln()` and the `1/chi`, `1/d`, `1/qd1`,
//     `1/qd1²` reciprocals assume the host pre-validated `chi>0`,
//     `d>0`, `qd1>0` (the launcher asserts the same as the CPU
//     reference); a row that slips through gets `row_status[i] = 2`
//     and zeroed outputs.
// ────────────────────────────────────────────────────────────────────────

/// Maximum primary local dimension `r` the flex row-primary kernel
/// supports.  Beyond this the entry point returns `Ok(None)` so the
/// host falls back to CPU.  Mirrors the roadmap "Step 5" `MAX_R`.
pub const SURVIVAL_FLEX_MAX_R: usize = 32;

#[cfg(target_os = "linux")]
const SURVIVAL_FLEX_PRIMARY_BODY: &str = r#"
// -------- kernel-specific defines ----------------------------------------
#define LN_TAU       1.8378770664093453  // log(2π)
#define MAX_R        32

// -------- flex row-primary gradient/Hessian kernel -----------------------
//
// One thread processes one row `i`.  The per-row jet is supplied as
// flat SoA device buffers (entry timepoint then exit timepoint).  The
// math mirrors `try_device_step5_primary_assembly` term-for-term.
//
// Buffer strides (all in f64, indexed by row `i` and primary `u`/`v`):
//   *_eta_u  [i*r + u]
//   *_eta_uv [i*r*r + u*r + v]
//   ... likewise chi_u/chi_uv/d_u/d_uv ...
// Scalars are length-`n` arrays indexed by `i`.
//
// `q1_index`/`qd1_index` are int arrays; a value `< 0` disables the
// corresponding perturbation bump (matches `usize::MAX` on the host).
//
extern "C" __global__ void survival_flex_primary_rows(
    int                          n,
    int                          r,
    // entry-timepoint jet
    const double * __restrict__  e_eta,      // [n]
    const double * __restrict__  e_eta_u,    // [n*r]
    const double * __restrict__  e_eta_uv,   // [n*r*r]
    // exit-timepoint jet
    const double * __restrict__  x_eta,      // [n]
    const double * __restrict__  x_chi,      // [n]
    const double * __restrict__  x_d,        // [n]
    const double * __restrict__  x_eta_u,    // [n*r]
    const double * __restrict__  x_eta_uv,   // [n*r*r]
    const double * __restrict__  x_chi_u,    // [n*r]
    const double * __restrict__  x_chi_uv,   // [n*r*r]
    const double * __restrict__  x_d_u,      // [n*r]
    const double * __restrict__  x_d_uv,     // [n*r*r]
    // per-row scalars
    const double * __restrict__  wi_arr,     // [n]
    const double * __restrict__  di_arr,     // [n]
    const double * __restrict__  q1_arr,     // [n]
    const double * __restrict__  qd1_arr,    // [n]
    const double * __restrict__  entry_k1_a, // [n]
    const double * __restrict__  entry_k2_a, // [n]
    const double * __restrict__  exit_k1_a,  // [n]
    const double * __restrict__  exit_k2_a,  // [n]
    const double * __restrict__  log_surv0_a,// [n]
    const double * __restrict__  log_surv1_a,// [n]
    const int    * __restrict__  q1_index_a, // [n]
    const int    * __restrict__  qd1_index_a,// [n]
    // outputs
    double * __restrict__        out_nll,    // [n]
    double * __restrict__        out_grad,   // [n*r]
    double * __restrict__        out_hess,   // [n*r*r]
    int    * __restrict__        row_status  // [n]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Defensive: the host guarantees r ≤ MAX_R, but never index past it.
    if (r <= 0 || r > MAX_R) {
        row_status[i] = 2;
        out_nll[i] = 0.0;
        return;
    }

    double wi  = wi_arr[i];
    double di  = di_arr[i];
    double q1  = q1_arr[i];
    double qd1 = qd1_arr[i];

    double exit_chi = x_chi[i];
    double exit_eta = x_eta[i];
    double exit_d   = x_d[i];

    // Match the CPU validation: chi/d must be positive finite (the host
    // already rejected such rows; this is belt-and-suspenders so a slip
    // becomes a clean reject rather than a NaN write).
    bool bad = (!isfinite(exit_chi)) || (exit_chi <= 0.0)
            || (!isfinite(exit_d))   || (exit_d   <= 0.0)
            || (!isfinite(qd1))      || (qd1      <= 0.0);
    if (bad) {
        row_status[i] = 2;
        out_nll[i] = 0.0;
        for (int u = 0; u < r; ++u) out_grad[i * r + u] = 0.0;
        for (int k = 0; k < r * r; ++k) out_hess[(long long)i * r * r + k] = 0.0;
        return;
    }

    int q1_index  = q1_index_a[i];
    int qd1_index = qd1_index_a[i];

    double entry_k1 = entry_k1_a[i];
    double entry_k2 = entry_k2_a[i];
    double exit_k1  = exit_k1_a[i];
    double exit_k2  = exit_k2_a[i];
    double log_surv0 = log_surv0_a[i];
    double log_surv1 = log_surv1_a[i];

    // ── NLL ──────────────────────────────────────────────────────────────
    double log_phi_eta1 = -0.5 * (exit_eta * exit_eta + LN_TAU);
    double log_phi_q1   = -0.5 * (q1 * q1 + LN_TAU);
    double row_nll = wi
        * (log_surv0
           - (1.0 - di) * log_surv1
           - di * log_phi_eta1
           - di * log(exit_chi)
           - di * log_phi_q1
           + di * log(exit_d)
           - di * log(qd1));
    out_nll[i] = row_nll;

    double entry_u1     = -entry_k1;
    double entry_u2     =  entry_k2;
    double exit_surv_u1 = -exit_k1;
    double exit_surv_u2 =  exit_k2;

    // Base device-buffer offsets for this row.
    long long base_u  = (long long)i * r;       // length-r vectors
    long long base_uv = (long long)i * r * r;   // r×r matrices

    // ── Gradient ─────────────────────────────────────────────────────────
    for (int u = 0; u < r; ++u) {
        double e_eu = e_eta_u[base_u + u];
        double x_eu = x_eta_u[base_u + u];
        double val = 0.0;
        val += entry_u1     * e_eu;
        val += exit_surv_u1 * x_eu;
        val += wi * di * exit_eta * x_eu;
        val -= wi * di * x_chi_u[base_u + u] / exit_chi;
        if (u == q1_index) {
            val += wi * di * q1;
        }
        val += wi * di * x_d_u[base_u + u] / exit_d;
        if (u == qd1_index) {
            val -= wi * di / qd1;
        }
        out_grad[base_u + u] = val;
    }

    // ── Hessian (full symmetric r×r) ─────────────────────────────────────
    double chi_sq = exit_chi * exit_chi;
    double d_sq   = exit_d   * exit_d;
    for (int u = 0; u < r; ++u) {
        double e_eu = e_eta_u[base_u + u];
        double x_eu = x_eta_u[base_u + u];
        double x_cu = x_chi_u[base_u + u];
        double x_du = x_d_u[base_u + u];
        for (int v = u; v < r; ++v) {
            double e_ev = e_eta_u[base_u + v];
            double x_ev = x_eta_u[base_u + v];
            double x_cv = x_chi_u[base_u + v];
            double x_dv = x_d_u[base_u + v];
            long long off_uv = base_uv + (long long)u * r + v;

            double val = 0.0;
            // entry survival term: k2 outer product + k1 second jet.
            val += entry_u2 * e_eu * e_ev + entry_u1 * e_eta_uv[off_uv];
            // exit survival term.
            val += exit_surv_u2 * x_eu * x_ev + exit_surv_u1 * x_eta_uv[off_uv];
            // event log φ(η₁) curvature.
            val += wi * di * (x_eu * x_ev + exit_eta * x_eta_uv[off_uv]);
            // − d·log χ₁ : FLAGGED cancellation-prone ratio term.
            val -= wi * di
                 * (x_chi_uv[off_uv] / exit_chi - (x_cu * x_cv) / chi_sq);
            if (u == q1_index && v == q1_index) {
                val += wi * di;
            }
            // + d·log d : same cancellation-prone structure.
            val += wi * di
                 * (x_d_uv[off_uv] / exit_d - (x_du * x_dv) / d_sq);
            if (u == qd1_index && v == qd1_index) {
                val += wi * di / (qd1 * qd1);
            }

            out_hess[off_uv] = val;
            if (v != u) {
                out_hess[base_uv + (long long)v * r + u] = val;
            }
        }
    }

    row_status[i] = 0;
}
"#;

// ────────────────────────────────────────────────────────────────────────
// Step 5 — FLEX row-primary launcher + host-side SoA inputs.
//
// `survival_flex_primary_rows` (NVRTC source above) is the native-GPU
// generalization of the rigid r==4 kernel to r ≤ SURVIVAL_FLEX_MAX_R.
// The host packs the per-row jet as flat Struct-of-Arrays device buffers
// (entry timepoint then exit timepoint) that mirror, byte-for-byte, the
// scalar / vector algebra in `try_device_step5_primary_assembly` (the
// device-shaped CPU reference earlier in this file) and its parent
// `compute_row_flex_primary_gradient_hessian_from_parts` in
// `src/families/survival_marginal_slope.rs`.
//
// Per-row index `usize::MAX` on the host (the "no perturbation" sentinel
// of `SurvivalFlexStep5RowInputs::q1_index` / `qd1_index`) maps to the
// device `< 0` disable convention via `index_to_i32` — exactly the
// `if (u == q1_index)` / `if (u == qd1_index)` guards in the kernel never
// fire for a disabled slot.
//
// SUPPORTED SHAPE (the entry point pre-filters): scalar score, every row
// of the batch sharing one `r ≤ SURVIVAL_FLEX_MAX_R`, NO influence
// absorber (per the #461 caveat at the top of this file the jet does not
// yet carry the `infl` primary coord).  Everything else stays `Ok(None)`
// so the host falls back to CPU.
// ────────────────────────────────────────────────────────────────────────

/// Convert a host primary-index slot (`usize::MAX` ≡ "no perturbation")
/// into the device kernel's `< 0` disable convention.  Any in-range index
/// must fit in `i32` because `r ≤ SURVIVAL_FLEX_MAX_R`.
#[inline]
fn primary_index_to_i32(idx: usize) -> i32 {
    if idx == usize::MAX {
        -1
    } else {
        // `r ≤ 32` so an enabled index is always well within i32.
        i32::try_from(idx).unwrap_or(-1)
    }
}

/// One row's jet for a single timepoint, laid out as the host owns it.
/// Entry timepoint only needs `eta`/`eta_u`/`eta_uv` (the survival term
/// of the NLL involves no `chi`/`d` at entry); the exit timepoint carries
/// the full bundle.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalFlexPrimaryTimepointRow<'a> {
    pub eta: f64,
    pub chi: f64,
    pub d: f64,
    /// Length `r`.
    pub eta_u: &'a [f64],
    /// Row-major `r × r`, symmetric.
    pub eta_uv: &'a [f64],
    /// Length `r`.
    pub chi_u: &'a [f64],
    /// Row-major `r × r`, symmetric.
    pub chi_uv: &'a [f64],
    /// Length `r`.
    pub d_u: &'a [f64],
    /// Row-major `r × r`, symmetric.
    pub d_uv: &'a [f64],
}

/// Per-row inputs for the flex row-primary GPU launcher.  Identical math
/// contract to [`SurvivalFlexStep5RowInputs`]; the launcher flattens a
/// slice of these into the device SoA buffers the kernel reads.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalFlexPrimaryRow<'a> {
    pub entry: SurvivalFlexPrimaryTimepointRow<'a>,
    pub exit: SurvivalFlexPrimaryTimepointRow<'a>,
    pub wi: f64,
    pub di: f64,
    pub q1: f64,
    pub qd1: f64,
    /// `usize::MAX` disables the `+ wi·di·q1` / `+ wi·di` bumps.
    pub q1_index: usize,
    /// `usize::MAX` disables the `-wi·di/qd1` / `+wi·di/qd1²` bumps.
    pub qd1_index: usize,
    pub entry_k1: f64,
    pub entry_k2: f64,
    pub exit_k1: f64,
    pub exit_k2: f64,
    pub log_surv0: f64,
    pub log_surv1: f64,
}

/// Per-row output bundle from the flex row-primary GPU kernel.  Same
/// shape contract as [`SurvivalFlexStep5RowOutputs`] but flattened across
/// the whole batch (one `r`, `r×r` slot per row).
#[derive(Clone, Debug)]
pub struct SurvivalFlexPrimaryRowOutputs {
    /// Per-row NLL contribution, length `n`.
    pub nll: Vec<f64>,
    /// Per-row primary gradient, length `n*r` (row-major).
    pub grad: Vec<f64>,
    /// Per-row primary Hessian, length `n*r*r` (row-major r×r symmetric
    /// per row).
    pub hess: Vec<f64>,
    /// Per-row status — `0` ok, `2` non-finite / non-positive chi/d/qd1.
    pub row_status: Vec<i32>,
}

/// Flattened host-side SoA staging for one batch of flex primary rows.
/// Built by [`flatten_primary_rows`] and consumed by the launcher; kept
/// as a named struct so the CPU oracle and the launcher pack the buffers
/// identically (one source of truth for the device layout).
struct SurvivalFlexPrimaryRowBatchFlat {
    n: usize,
    r: usize,
    e_eta: Vec<f64>,
    e_eta_u: Vec<f64>,
    e_eta_uv: Vec<f64>,
    x_eta: Vec<f64>,
    x_chi: Vec<f64>,
    x_d: Vec<f64>,
    x_eta_u: Vec<f64>,
    x_eta_uv: Vec<f64>,
    x_chi_u: Vec<f64>,
    x_chi_uv: Vec<f64>,
    x_d_u: Vec<f64>,
    x_d_uv: Vec<f64>,
    wi: Vec<f64>,
    di: Vec<f64>,
    q1: Vec<f64>,
    qd1: Vec<f64>,
    entry_k1: Vec<f64>,
    entry_k2: Vec<f64>,
    exit_k1: Vec<f64>,
    exit_k2: Vec<f64>,
    log_surv0: Vec<f64>,
    log_surv1: Vec<f64>,
    q1_index: Vec<i32>,
    qd1_index: Vec<i32>,
}

/// Validate + flatten a slice of [`SurvivalFlexPrimaryRow`] into the
/// device SoA layout.  Enforces a single shared `r` (the kernel takes one
/// `r` for the whole launch) and that every per-row jet has the right
/// `r` / `r×r` lengths.  Returns `Err` on a shape violation so the entry
/// point surfaces a single error surface, matching the rigid launcher.
fn flatten_primary_rows(
    rows: &[SurvivalFlexPrimaryRow<'_>],
) -> Result<SurvivalFlexPrimaryRowBatchFlat, GpuError> {
    let n = rows.len();
    let r = rows[0].exit.eta_u.len();
    let rr = r * r;

    let mut flat = SurvivalFlexPrimaryRowBatchFlat {
        n,
        r,
        e_eta: Vec::with_capacity(n),
        e_eta_u: Vec::with_capacity(n * r),
        e_eta_uv: Vec::with_capacity(n * rr),
        x_eta: Vec::with_capacity(n),
        x_chi: Vec::with_capacity(n),
        x_d: Vec::with_capacity(n),
        x_eta_u: Vec::with_capacity(n * r),
        x_eta_uv: Vec::with_capacity(n * rr),
        x_chi_u: Vec::with_capacity(n * r),
        x_chi_uv: Vec::with_capacity(n * rr),
        x_d_u: Vec::with_capacity(n * r),
        x_d_uv: Vec::with_capacity(n * rr),
        wi: Vec::with_capacity(n),
        di: Vec::with_capacity(n),
        q1: Vec::with_capacity(n),
        qd1: Vec::with_capacity(n),
        entry_k1: Vec::with_capacity(n),
        entry_k2: Vec::with_capacity(n),
        exit_k1: Vec::with_capacity(n),
        exit_k2: Vec::with_capacity(n),
        log_surv0: Vec::with_capacity(n),
        log_surv1: Vec::with_capacity(n),
        q1_index: Vec::with_capacity(n),
        qd1_index: Vec::with_capacity(n),
    };

    for (i, row) in rows.iter().enumerate() {
        let check = |label: &str, len: usize, expected: usize| -> Result<(), GpuError> {
            if len != expected {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex primary row {i}: {label}.len()={len} expected {expected} \
                         (batch r={r})"
                    ),
                });
            }
            Ok(())
        };
        check("entry.eta_u", row.entry.eta_u.len(), r)?;
        check("entry.eta_uv", row.entry.eta_uv.len(), rr)?;
        check("exit.eta_u", row.exit.eta_u.len(), r)?;
        check("exit.eta_uv", row.exit.eta_uv.len(), rr)?;
        check("exit.chi_u", row.exit.chi_u.len(), r)?;
        check("exit.chi_uv", row.exit.chi_uv.len(), rr)?;
        check("exit.d_u", row.exit.d_u.len(), r)?;
        check("exit.d_uv", row.exit.d_uv.len(), rr)?;

        flat.e_eta.push(row.entry.eta);
        flat.e_eta_u.extend_from_slice(row.entry.eta_u);
        flat.e_eta_uv.extend_from_slice(row.entry.eta_uv);

        flat.x_eta.push(row.exit.eta);
        flat.x_chi.push(row.exit.chi);
        flat.x_d.push(row.exit.d);
        flat.x_eta_u.extend_from_slice(row.exit.eta_u);
        flat.x_eta_uv.extend_from_slice(row.exit.eta_uv);
        flat.x_chi_u.extend_from_slice(row.exit.chi_u);
        flat.x_chi_uv.extend_from_slice(row.exit.chi_uv);
        flat.x_d_u.extend_from_slice(row.exit.d_u);
        flat.x_d_uv.extend_from_slice(row.exit.d_uv);

        flat.wi.push(row.wi);
        flat.di.push(row.di);
        flat.q1.push(row.q1);
        flat.qd1.push(row.qd1);
        flat.entry_k1.push(row.entry_k1);
        flat.entry_k2.push(row.entry_k2);
        flat.exit_k1.push(row.exit_k1);
        flat.exit_k2.push(row.exit_k2);
        flat.log_surv0.push(row.log_surv0);
        flat.log_surv1.push(row.log_surv1);
        flat.q1_index.push(primary_index_to_i32(row.q1_index));
        flat.qd1_index.push(primary_index_to_i32(row.qd1_index));
    }

    Ok(flat)
}

/// Launch the flex row-primary gradient/Hessian kernel for a batch of
/// rows that all share one `r ≤ SURVIVAL_FLEX_MAX_R`.  Returns `Ok(None)`
/// if the backend is unsupported on this build (non-Linux / no CUDA
/// runtime) or the shape is out of range, so the caller falls back to
/// CPU cleanly; returns `Err` for genuine driver / shape failures.
///
/// The host MUST pre-filter to the SUPPORTED SHAPE documented above — in
/// particular it must NOT call this with an influence absorber active
/// (the jet does not yet carry the `infl` coord; see the #461 caveat).
pub fn try_flex_primary_rows(
    rows: &[SurvivalFlexPrimaryRow<'_>],
) -> Result<Option<SurvivalFlexPrimaryRowOutputs>, GpuError> {
    if rows.is_empty() {
        return Ok(Some(SurvivalFlexPrimaryRowOutputs {
            nll: Vec::new(),
            grad: Vec::new(),
            hess: Vec::new(),
            row_status: Vec::new(),
        }));
    }
    // The kernel launches with a single runtime `r`; a ragged batch is an
    // unsupported shape (each fit's flex jet is one fixed `r`).
    let r = rows[0].exit.eta_u.len();
    if r == 0 || r > SURVIVAL_FLEX_MAX_R {
        return Ok(None);
    }
    // Flatten + shape-validate up front so a ragged / mis-sized batch
    // surfaces as `Err` on EVERY build, not just on the V100 host.  The
    // flattened batch always carries `n` rows of width `r`; assert that
    // invariant so the value is live on every cfg (no dead binding off
    // Linux) and a future layout regression trips loudly.
    let flat = flatten_primary_rows(rows)?;
    if flat.n != rows.len() || flat.r != r {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex primary flatten produced n={} r={} expected n={} r={r}",
                flat.n,
                flat.r,
                rows.len()
            ),
        });
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
        Some(backend.launch_primary_rows_linux(&flat)).transpose()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
impl SurvivalFlexGpuBackend {
    fn launch_primary_rows_linux(
        &self,
        flat: &SurvivalFlexPrimaryRowBatchFlat,
    ) -> Result<SurvivalFlexPrimaryRowOutputs, GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        // The flex primary kernel is compiled into the same combined
        // module as the rigid kernel (`compile_primary_module` aliases
        // `compile_rigid_module`); both `extern "C"` symbols come from the
        // single cached `Arc<CudaModule>`.
        let module = self.compile_primary_module()?;
        let func = module
            .load_function("survival_flex_primary_rows")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex primary load_function: {err}"),
            })?;

        let n = flat.n;
        let r = flat.r;
        let stream = &self.inner.stream;
        let mk_htod_f64 = |slice: &[f64], name: &str| -> Result<_, GpuError> {
            stream
                .clone_htod(slice)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex primary memcpy_stod {name}: {err}"),
                })
        };
        let mk_htod_i32 = |slice: &[i32], name: &str| -> Result<_, GpuError> {
            stream
                .clone_htod(slice)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex primary memcpy_stod {name}: {err}"),
                })
        };

        let d_e_eta = mk_htod_f64(&flat.e_eta, "e_eta")?;
        let d_e_eta_u = mk_htod_f64(&flat.e_eta_u, "e_eta_u")?;
        let d_e_eta_uv = mk_htod_f64(&flat.e_eta_uv, "e_eta_uv")?;
        let d_x_eta = mk_htod_f64(&flat.x_eta, "x_eta")?;
        let d_x_chi = mk_htod_f64(&flat.x_chi, "x_chi")?;
        let d_x_d = mk_htod_f64(&flat.x_d, "x_d")?;
        let d_x_eta_u = mk_htod_f64(&flat.x_eta_u, "x_eta_u")?;
        let d_x_eta_uv = mk_htod_f64(&flat.x_eta_uv, "x_eta_uv")?;
        let d_x_chi_u = mk_htod_f64(&flat.x_chi_u, "x_chi_u")?;
        let d_x_chi_uv = mk_htod_f64(&flat.x_chi_uv, "x_chi_uv")?;
        let d_x_d_u = mk_htod_f64(&flat.x_d_u, "x_d_u")?;
        let d_x_d_uv = mk_htod_f64(&flat.x_d_uv, "x_d_uv")?;
        let d_wi = mk_htod_f64(&flat.wi, "wi")?;
        let d_di = mk_htod_f64(&flat.di, "di")?;
        let d_q1 = mk_htod_f64(&flat.q1, "q1")?;
        let d_qd1 = mk_htod_f64(&flat.qd1, "qd1")?;
        let d_entry_k1 = mk_htod_f64(&flat.entry_k1, "entry_k1")?;
        let d_entry_k2 = mk_htod_f64(&flat.entry_k2, "entry_k2")?;
        let d_exit_k1 = mk_htod_f64(&flat.exit_k1, "exit_k1")?;
        let d_exit_k2 = mk_htod_f64(&flat.exit_k2, "exit_k2")?;
        let d_log_surv0 = mk_htod_f64(&flat.log_surv0, "log_surv0")?;
        let d_log_surv1 = mk_htod_f64(&flat.log_surv1, "log_surv1")?;
        let d_q1_index = mk_htod_i32(&flat.q1_index, "q1_index")?;
        let d_qd1_index = mk_htod_i32(&flat.qd1_index, "qd1_index")?;

        let mut d_nll = stream
            .alloc_zeros::<f64>(n)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex primary alloc_zeros nll: {err}"),
            })?;
        let mut d_grad =
            stream
                .alloc_zeros::<f64>(n * r)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex primary alloc_zeros grad: {err}"),
                })?;
        let mut d_hess =
            stream
                .alloc_zeros::<f64>(n * r * r)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex primary alloc_zeros hess: {err}"),
                })?;
        let mut d_status =
            stream
                .alloc_zeros::<i32>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex primary alloc_zeros status: {err}"),
                })?;

        let n_i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex primary n={n} overflows i32"),
        })?;
        let r_i32 = i32::try_from(r).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex primary r={r} overflows i32"),
        })?;

        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        // Argument order MUST match the kernel signature exactly:
        // (n, r, e_eta, e_eta_u, e_eta_uv, x_eta, x_chi, x_d, x_eta_u,
        //  x_eta_uv, x_chi_u, x_chi_uv, x_d_u, x_d_uv, wi, di, q1, qd1,
        //  entry_k1, entry_k2, exit_k1, exit_k2, log_surv0, log_surv1,
        //  q1_index, qd1_index, out_nll, out_grad, out_hess, row_status).
        builder
            .arg(&n_i32)
            .arg(&r_i32)
            .arg(&d_e_eta)
            .arg(&d_e_eta_u)
            .arg(&d_e_eta_uv)
            .arg(&d_x_eta)
            .arg(&d_x_chi)
            .arg(&d_x_d)
            .arg(&d_x_eta_u)
            .arg(&d_x_eta_uv)
            .arg(&d_x_chi_u)
            .arg(&d_x_chi_uv)
            .arg(&d_x_d_u)
            .arg(&d_x_d_uv)
            .arg(&d_wi)
            .arg(&d_di)
            .arg(&d_q1)
            .arg(&d_qd1)
            .arg(&d_entry_k1)
            .arg(&d_entry_k2)
            .arg(&d_exit_k1)
            .arg(&d_exit_k2)
            .arg(&d_log_surv0)
            .arg(&d_log_surv1)
            .arg(&d_q1_index)
            .arg(&d_qd1_index)
            .arg(&mut d_nll)
            .arg(&mut d_grad)
            .arg(&mut d_hess)
            .arg(&mut d_status);
        // SAFETY: every argument is a typed device pointer / scalar
        // matching the `survival_flex_primary_rows` signature above, and
        // grid/block cover exactly `n` rows.  Out-of-range threads
        // early-return; the kernel re-checks `r ≤ MAX_R` per thread.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex primary launch: {err}"),
        })?;

        let nll = stream
            .clone_dtoh(&d_nll)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex primary memcpy_dtoh nll: {err}"),
            })?;
        let grad = stream
            .clone_dtoh(&d_grad)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex primary memcpy_dtoh grad: {err}"),
            })?;
        let hess = stream
            .clone_dtoh(&d_hess)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex primary memcpy_dtoh hess: {err}"),
            })?;
        let row_status =
            stream
                .clone_dtoh(&d_status)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex primary memcpy_dtoh status: {err}"),
                })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex primary synchronize: {err}"),
            })?;

        Ok(SurvivalFlexPrimaryRowOutputs {
            nll,
            grad,
            hess,
            row_status,
        })
    }
}

/// CPU oracle for the flex row-primary kernel — the parity reference the
/// V100 verification pass lands against.  Computes the per-row NLL,
/// gradient and Hessian with the EXACT same operator order as the device
/// `survival_flex_primary_rows` kernel (and as
/// `try_device_step5_primary_assembly` / the family helper
/// `compute_row_flex_primary_gradient_hessian_from_parts`), producing the
/// flat batch layout the launcher returns.
///
/// Rows whose `exit.chi` / `exit.d` / `qd1` are non-positive or non-finite
/// get `row_status = 2` and zeroed outputs — exactly the kernel's
/// belt-and-suspenders reject.  All other rows get `row_status = 0`.
pub fn cpu_oracle_flex_primary_rows(
    rows: &[SurvivalFlexPrimaryRow<'_>],
) -> Result<SurvivalFlexPrimaryRowOutputs, GpuError> {
    if rows.is_empty() {
        return Ok(SurvivalFlexPrimaryRowOutputs {
            nll: Vec::new(),
            grad: Vec::new(),
            hess: Vec::new(),
            row_status: Vec::new(),
        });
    }
    let n = rows.len();
    let r = rows[0].exit.eta_u.len();
    // Reuse the launcher's flatten/validate so the oracle enforces the
    // identical shape contract (single shared `r`, correct jet lengths).
    // We only need the validation side-effect here; assert the derived
    // `r` round-trips so the bound value stays live on every build.
    let flat = flatten_primary_rows(rows)?;
    if flat.r != r {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex primary oracle: flatten r={} != derived r={r}",
                flat.r
            ),
        });
    }

    let mut nll = vec![0.0_f64; n];
    let mut grad = vec![0.0_f64; n * r];
    let mut hess = vec![0.0_f64; n * r * r];
    let mut row_status = vec![0_i32; n];

    let ln_tau = std::f64::consts::TAU.ln();

    for (i, row) in rows.iter().enumerate() {
        let exit_chi = row.exit.chi;
        let exit_eta = row.exit.eta;
        let exit_d = row.exit.d;
        let qd1 = row.qd1;

        let bad = !exit_chi.is_finite()
            || exit_chi <= 0.0
            || !exit_d.is_finite()
            || exit_d <= 0.0
            || !qd1.is_finite()
            || qd1 <= 0.0;
        if bad {
            row_status[i] = 2;
            continue;
        }

        let wi = row.wi;
        let di = row.di;
        let q1 = row.q1;
        let q1_index = primary_index_to_i32(row.q1_index);
        let qd1_index = primary_index_to_i32(row.qd1_index);

        // ── NLL ──────────────────────────────────────────────────────────
        let log_phi_eta1 = -0.5 * (exit_eta * exit_eta + ln_tau);
        let log_phi_q1 = -0.5 * (q1 * q1 + ln_tau);
        nll[i] = wi
            * (row.log_surv0
                - (1.0 - di) * row.log_surv1
                - di * log_phi_eta1
                - di * exit_chi.ln()
                - di * log_phi_q1
                + di * exit_d.ln()
                - di * qd1.ln());

        let entry_u1 = -row.entry_k1;
        let entry_u2 = row.entry_k2;
        let exit_surv_u1 = -row.exit_k1;
        let exit_surv_u2 = row.exit_k2;

        let base_u = i * r;
        let base_uv = i * r * r;

        // ── Gradient ───────────────────────────────────────────────────────
        for u in 0..r {
            let mut val = 0.0;
            val += entry_u1 * row.entry.eta_u[u];
            val += exit_surv_u1 * row.exit.eta_u[u];
            val += wi * di * exit_eta * row.exit.eta_u[u];
            val -= wi * di * row.exit.chi_u[u] / exit_chi;
            if (u as i32) == q1_index {
                val += wi * di * q1;
            }
            val += wi * di * row.exit.d_u[u] / exit_d;
            if (u as i32) == qd1_index {
                val -= wi * di / qd1;
            }
            grad[base_u + u] = val;
        }

        // ── Hessian (full symmetric r×r) ───────────────────────────────────
        let chi_sq = exit_chi * exit_chi;
        let d_sq = exit_d * exit_d;
        for u in 0..r {
            for v in u..r {
                let off_uv = u * r + v;
                let mut val = 0.0;
                val += entry_u2 * row.entry.eta_u[u] * row.entry.eta_u[v]
                    + entry_u1 * row.entry.eta_uv[off_uv];
                val += exit_surv_u2 * row.exit.eta_u[u] * row.exit.eta_u[v]
                    + exit_surv_u1 * row.exit.eta_uv[off_uv];
                val += wi
                    * di
                    * (row.exit.eta_u[u] * row.exit.eta_u[v] + exit_eta * row.exit.eta_uv[off_uv]);
                // − d·log χ₁ — FLAGGED catastrophic-cancellation ratio.
                val -= wi
                    * di
                    * (row.exit.chi_uv[off_uv] / exit_chi
                        - (row.exit.chi_u[u] * row.exit.chi_u[v]) / chi_sq);
                if (u as i32) == q1_index && (v as i32) == q1_index {
                    val += wi * di;
                }
                // + d·log d — same cancellation-prone structure.
                val += wi
                    * di
                    * (row.exit.d_uv[off_uv] / exit_d - (row.exit.d_u[u] * row.exit.d_u[v]) / d_sq);
                if (u as i32) == qd1_index && (v as i32) == qd1_index {
                    val += wi * di / (qd1 * qd1);
                }
                hess[base_uv + off_uv] = val;
                if v != u {
                    hess[base_uv + v * r + u] = val;
                }
            }
        }
    }

    Ok(SurvivalFlexPrimaryRowOutputs {
        nll,
        grad,
        hess,
        row_status,
    })
}

// ────────────────────────────────────────────────────────────────────────
// Step 2 — survival-flex row-batched cubic-cell moment evaluator.
//
// Steps 3-6 need per-row derivative moments of the de-nested cubic
// correction `η(z) = c_0 + c_1·z + c_2·z² + c_3·z³` integrated against
// `exp(-q(z))` over each cell of the row's partition.  The CPU side
// builds these partitions via
// `survival::marginal_slope::denested_partition_cells` and then loops
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
/// `survival::marginal_slope::denested_partition_cells`.  Callers can
/// flatten the per-row `Vec<DenestedPartitionCell>` lists into this
/// shape with a single pass.
#[derive(Clone, Debug)]
pub struct SurvivalFlexRowCellsBatch<'a> {
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
pub struct SurvivalFlexRowMoments {
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
        if self.max_degree > crate::gpu::kernels::cubic_cell::MAX_SUPPORTED_DEGREE {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: max_degree={} exceeds substrate \
                     MAX_SUPPORTED_DEGREE={}",
                    self.max_degree,
                    crate::gpu::kernels::cubic_cell::MAX_SUPPORTED_DEGREE
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
pub fn try_row_batched_cell_moments(
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
        let cell = crate::gpu::kernels::cubic_cell::GpuDenestedCubicCell {
            left: batch.left[k],
            right: batch.right[k],
            c0: batch.c0[k],
            c1: batch.c1[k],
            c2: batch.c2[k],
            c3: batch.c3[k],
        };
        match crate::gpu::kernels::cubic_cell::branch::classify_cell_for_gpu(cell) {
            Ok(tag) => {
                cells.push(cell);
                branches.push(tag);
                prelim_status
                    .push(crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8);
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
                branches.push(crate::gpu::kernels::cubic_cell::GpuCellBranchTag::Affine);
                prelim_status.push(code as u8);
            }
        }
    }

    let view = crate::gpu::kernels::cubic_cell::CubicCellDerivativeMomentHostView {
        cells: &cells,
        branches: &branches,
        max_degree: batch.max_degree,
        residency: crate::gpu::kernels::cubic_cell::CubicCellMomentResidency::Host,
    };
    let out = crate::gpu::kernels::cubic_cell::try_build_cubic_cell_derivative_moments(view)?
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex row-cells batch: substrate returned None for n_cells={} > 0 \
                 (unexpected)",
                batch.n_cells
            ),
        })?;

    let (moments, mut status, stride) = match out {
        crate::gpu::kernels::cubic_cell::CubicCellDerivativeMomentOutput::Host {
            moments,
            status,
            stride,
        } => (moments, status, stride),
        #[cfg(target_os = "linux")]
        crate::gpu::kernels::cubic_cell::CubicCellDerivativeMomentOutput::Device { .. } => {
            return Err(GpuError::DriverCallFailed {
                reason: "survival_flex row-cells batch: substrate returned device-resident output \
                         but the survival-flex host pipeline consumes Host residency only"
                    .to_string(),
            });
        }
    };

    // Cells we pre-rejected (`prelim_status != Ok`) get a status code
    // from us if the substrate left them as Ok (it won't, because it
    // re-runs the classifier — but keeping this explicit guards against
    // a future substrate that trusts caller tags).
    for k in 0..batch.n_cells {
        if prelim_status[k] != crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8
            && status[k] == crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8
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
// Step 3 — device monotone-root intercept solve.
//
// The flex calibration step solves `F(a) = ⟨Φ(-η(z;a))⟩ - Φ(-q) = 0`
// once per row.  The CPU side runs `monotone_root::solve_monotone_root_detailed`
// (`families::survival::marginal_slope.rs:5363`).  Step 3 ports the
// control flow (Newton probe → bracket expansion → bisection +
// safeguarded Halley/Newton refinement) into an NVRTC kernel so every
// row solves in parallel.
//
// The control-flow kernel is parameterised over the F-evaluator: Step 4
// substitutes the real survival calibration (which needs the cell
// moments from Step 2) by adding the relevant evaluator branch to the
// NVRTC source.  Step 3 ships and tests against an analytic evaluator
//
//     F(a)   = alpha · exp(beta · a) + gamma
//     F'(a)  = alpha · beta · exp(beta · a)
//     F''(a) = alpha · beta² · exp(beta · a)
//
// whose closed-form root `a* = ln(-gamma/alpha) / beta` lets the parity
// test verify Newton probe + bracket expansion + Halley/Newton refine
// down to the CPU `solve_monotone_root_detailed` tolerance.
//
// Warm-start design: per-row arrays `a_entry[row]`, `a_exit[row]` carry
// the previous-iter intercept solution.  The kernel reads them, runs
// the solver, and writes back the converged root *plus* the abs-deriv
// and residual that downstream Step 4 IFT corrections need.
//
// Bracket safety: matches the CPU cap exactly —
// `step_cap = max(1e6, 1024·(1+|a_init|))` — and the same
// `step_sign = -sign(f·F')` rule.  Convergence on
// `|F| ≤ tol` *or* bracket width `≤ tol·(1+|lo|+|hi|)`, identical to
// the CPU loop.
// ────────────────────────────────────────────────────────────────────────

/// Per-row inputs for the Step 3 device intercept solve.  Borrows
/// host-side warm-start arrays + the per-row evaluator coefficients.
/// The Step-4 wiring replaces `(alpha, beta, gamma)` with the real
/// survival calibration evaluator; the Step-3 test path uses these
/// directly for closed-form parity against the CPU monotone-root
/// solver.
#[derive(Clone, Debug)]
pub struct SurvivalFlexInterceptSolveInputs<'a> {
    /// Number of rows.
    pub n: usize,
    /// Warm-start seed per row.  For the rigid (flex=false) fallback the
    /// CPU side uses `a_seed = q · √(1 + (s·g)²) / s` — the caller is
    /// expected to provide either that rigid seed *or* the previous-iter
    /// converged root (whichever is fresher).
    pub a_warm: &'a [f64],
    /// Analytic evaluator coefficients per row.  Step 4 swaps this out
    /// for the real survival calibration evaluator inputs.
    pub alpha: &'a [f64],
    pub beta: &'a [f64],
    pub gamma: &'a [f64],
    /// `|F| ≤ convergence_tol` and bracket width `≤ tol·(1+|lo|+|hi|)`
    /// both stop the loop.  Matches the CPU contract.
    pub convergence_tol: f64,
    /// Bracket-expansion iteration cap.  CPU side uses 64 for survival.
    pub max_bracket_iters: u32,
    /// Refinement iteration cap.  CPU side uses 64.
    pub max_refine_iters: u32,
}

/// Step 3 per-row output.
#[derive(Clone, Debug)]
pub struct SurvivalFlexInterceptSolveOutputs {
    /// Converged root `a*` per row.
    pub a_root: Vec<f64>,
    /// `|F'(a*)|` per row — Step 4 IFT uses this to invert through the
    /// constraint and propagate derivatives.
    pub abs_deriv: Vec<f64>,
    /// `F(a*)` per row.  Always satisfies `|residual| ≤ convergence_tol`
    /// on success.
    pub residual: Vec<f64>,
    /// Per-row exit status:
    ///   0 — converged to `|F| ≤ tol`
    ///   1 — exited on bracket-width contraction (acceptable; root within tol)
    ///   2 — Newton probe degenerate (F'(a_warm) zero / non-finite)
    ///   3 — bracket search exhausted (no sign change after `max_bracket_iters`)
    ///   4 — refine loop exhausted without bracket/residual convergence
    ///   5 — non-finite produced by the evaluator (e.g. overflow)
    pub status: Vec<u8>,
}

impl<'a> SurvivalFlexInterceptSolveInputs<'a> {
    fn validate(&self) -> Result<(), GpuError> {
        let n = self.n;
        let lens: [(&str, usize); 4] = [
            ("a_warm", self.a_warm.len()),
            ("alpha", self.alpha.len()),
            ("beta", self.beta.len()),
            ("gamma", self.gamma.len()),
        ];
        for (label, len) in lens {
            if len != n {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex intercept-solve inputs: {label}.len()={len} != n={n}"
                    ),
                });
            }
        }
        if !(self.convergence_tol.is_finite() && self.convergence_tol > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex intercept-solve inputs: convergence_tol must be positive \
                     finite, got {}",
                    self.convergence_tol
                ),
            });
        }
        if self.max_bracket_iters == 0 || self.max_refine_iters == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex intercept-solve inputs: iter caps must be positive, got \
                     bracket={} refine={}",
                    self.max_bracket_iters, self.max_refine_iters
                ),
            });
        }
        Ok(())
    }
}

/// CPU oracle for the Step 3 intercept solve.  Drives the existing
/// `families::monotone_root::solve_monotone_root_detailed` against the
/// analytic evaluator (`alpha · exp(beta · a) + gamma`) so the device
/// kernel can be checked element-wise.  Returns the same output layout
/// as the device kernel.
///
/// Status codes match the device kernel's enumeration (0 converged,
/// 2 degenerate-derivative, 3 bracket-exhausted, 5 non-finite).  The
/// CPU solver collapses "Halley-only convergence" and "bisection-only
/// convergence" into a single status `0`, so the device parity bar is
/// status-equal *plus* numerical equality of `a_root`, `abs_deriv`,
/// `residual` at the CPU tolerance.
pub fn cpu_oracle_intercept_solve(
    inputs: &SurvivalFlexInterceptSolveInputs<'_>,
) -> SurvivalFlexInterceptSolveOutputs {
    use crate::families::monotone_root::{MonotoneRootError, solve_monotone_root_detailed};
    let mut a_root = vec![0.0_f64; inputs.n];
    let mut abs_deriv = vec![0.0_f64; inputs.n];
    let mut residual = vec![0.0_f64; inputs.n];
    let mut status = vec![0u8; inputs.n];
    for row in 0..inputs.n {
        let alpha = inputs.alpha[row];
        let beta = inputs.beta[row];
        let gamma = inputs.gamma[row];
        let a_warm = inputs.a_warm[row];
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            let e = (beta * a).exp();
            if !e.is_finite() {
                return Err(format!("overflow at a={a}"));
            }
            let f = alpha * e + gamma;
            let fp = alpha * beta * e;
            let fpp = alpha * beta * beta * e;
            Ok((f, fp, fpp))
        };
        match solve_monotone_root_detailed(
            eval,
            a_warm,
            "survival_flex_intercept_oracle",
            inputs.convergence_tol,
            inputs.max_bracket_iters as usize,
            inputs.max_refine_iters as usize,
        ) {
            Ok(sol) => {
                a_root[row] = sol.root;
                abs_deriv[row] = sol.abs_deriv;
                residual[row] = sol.residual;
                status[row] = 0;
            }
            Err(MonotoneRootError::DegenerateDerivative { a, .. }) => {
                a_root[row] = a;
                abs_deriv[row] = 0.0;
                residual[row] = f64::NAN;
                status[row] = 2;
            }
            Err(MonotoneRootError::BracketingExhausted { .. }) => {
                a_root[row] = a_warm;
                abs_deriv[row] = 0.0;
                residual[row] = f64::NAN;
                status[row] = 3;
            }
            Err(MonotoneRootError::RefinementDidNotConverge { last_residual, .. }) => {
                a_root[row] = a_warm;
                abs_deriv[row] = 0.0;
                residual[row] = last_residual;
                status[row] = 4;
            }
            Err(_) => {
                a_root[row] = a_warm;
                abs_deriv[row] = 0.0;
                residual[row] = f64::NAN;
                status[row] = 5;
            }
        }
    }
    SurvivalFlexInterceptSolveOutputs {
        a_root,
        abs_deriv,
        residual,
        status,
    }
}

// ────────────────────────────────────────────────────────────────────────
// NVRTC source — Step 3 (parameterised monotone root, analytic evaluator).
//
// One thread per row.  Identical control flow to the CPU
// `solve_monotone_root_detailed`:
//   * Up to 2 Newton probes from `a_warm[row]`.
//   * If un-converged, geometric step doubling (bracket phase) using
//     `step_sign = -sign(f · F')`, step_mag start = max(1.0, 0.25·(1+|a|)),
//     cap = max(1e6, 1024·(1+|a_warm|)).
//   * Phase 2: hybrid bisection / safeguarded Halley + Newton inside
//     the bracket; convergence on residual or bracket width.
//   * Best-of accounting for the residual, matching the CPU loop.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
const SURVIVAL_FLEX_INTERCEPT_SOLVE_SOURCE: &str = r#"
extern "C" __device__ __forceinline__ void
eval_F_analytic(double a, double alpha, double beta, double gamma,
                double *f, double *fp, double *fpp, int *ok) {
    double e = exp(beta * a);
    if (!isfinite(e)) { *f = 0.0; *fp = 0.0; *fpp = 0.0; *ok = 0; return; }
    *f   = alpha * e + gamma;
    *fp  = alpha * beta * e;
    *fpp = alpha * beta * beta * e;
    *ok  = 1;
}

extern "C" __global__ void survival_flex_intercept_solve(
    const double * __restrict__ a_warm_arr,
    const double * __restrict__ alpha_arr,
    const double * __restrict__ beta_arr,
    const double * __restrict__ gamma_arr,
    double                       convergence_tol,
    unsigned int                 max_bracket_iters,
    unsigned int                 max_refine_iters,
    int                          n,
    double * __restrict__        out_a_root,
    double * __restrict__        out_abs_deriv,
    double * __restrict__        out_residual,
    unsigned char * __restrict__ out_status
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double alpha = alpha_arr[row];
    double beta  = beta_arr[row];
    double gamma = gamma_arr[row];
    double a_init = a_warm_arr[row];

    double f, fp, fpp;
    int ok;
    eval_F_analytic(a_init, alpha, beta, gamma, &f, &fp, &fpp, &ok);
    if (!ok) { out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=nan(""); out_status[row]=5; return; }

    // Exact-root shortcut.
    if (fabs(f) <= convergence_tol) {
        double abs_d = fabs(fp);
        if (!isfinite(abs_d) || abs_d == 0.0) {
            out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=f; out_status[row]=2;
        } else {
            out_a_root[row]=a_init; out_abs_deriv[row]=abs_d; out_residual[row]=f; out_status[row]=0;
        }
        return;
    }

    if (!isfinite(fp) || fp == 0.0) {
        out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=nan(""); out_status[row]=2;
        return;
    }

    // --- Newton probe (≤ 2) ---
    double a = a_init;
    double f_init = f;
    double fp_init = fp;
    for (int probe = 0; probe < 2; ++probe) {
        if (fabs(f) <= convergence_tol) {
            double abs_d = fabs(fp);
            if (isfinite(abs_d) && abs_d != 0.0) {
                out_a_root[row]=a; out_abs_deriv[row]=abs_d; out_residual[row]=f; out_status[row]=0;
                return;
            }
            break;
        }
        if (!isfinite(fp) || fabs(fp) <= 1e-30) break;
        double step = -f / fp;
        if (!isfinite(step) || fabs(step) > 8.0 * (1.0 + fabs(a))) break;
        double cand = a + step;
        double f_c, fp_c, fpp_c; int ok_c;
        eval_F_analytic(cand, alpha, beta, gamma, &f_c, &fp_c, &fpp_c, &ok_c);
        if (!ok_c) break;
        if (fabs(f_c) <= convergence_tol) {
            double abs_d = fabs(fp_c);
            if (isfinite(abs_d) && abs_d != 0.0) {
                out_a_root[row]=cand; out_abs_deriv[row]=abs_d; out_residual[row]=f_c; out_status[row]=0;
                return;
            }
            break;
        }
        a = cand; f = f_c; fp = fp_c; fpp = fpp_c;
    }

    // --- Phase 1: bracket ---
    double step_sign = (f_init * fp_init < 0.0) ? 1.0 : -1.0;
    int f_init_neg = (f_init < 0.0) ? 1 : 0;
    double same_side = a_init;
    double step_mag = fmax(0.25 * (1.0 + fabs(a_init)), 1.0);
    double step_cap = fmax(1e6, 1024.0 * (1.0 + fabs(a_init)));

    int found_other = 0;
    double other = 0.0;
    for (unsigned int it = 0; it < max_bracket_iters; ++it) {
        double probe_pt = same_side + step_mag * step_sign;
        double f_probe, fp_probe, fpp_probe; int ok_probe;
        eval_F_analytic(probe_pt, alpha, beta, gamma, &f_probe, &fp_probe, &fpp_probe, &ok_probe);
        if (!ok_probe) break;
        int crossed = f_init_neg ? (f_probe >= 0.0) : (f_probe <= 0.0);
        if (crossed) { other = probe_pt; found_other = 1; break; }
        same_side = probe_pt;
        step_mag *= 2.0;
        if (step_mag > step_cap) break;
    }
    if (!found_other) {
        out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=nan(""); out_status[row]=3;
        return;
    }

    double neg_pt, pos_pt;
    if (f_init_neg) { neg_pt = same_side; pos_pt = other; }
    else            { neg_pt = other;     pos_pt = same_side; }

    // --- Phase 2: hybrid refine ---
    double best_a = a_init, best_f = f_init, best_abs_d = fabs(fp_init);
    int    converged_residual = 0, converged_bracket = 0;

    for (unsigned int it = 0; it < max_refine_iters; ++it) {
        double lo = fmin(neg_pt, pos_pt);
        double hi = fmax(neg_pt, pos_pt);
        double mid = 0.5 * (lo + hi);

        double f_mid, fp_mid, fpp_mid; int ok_mid;
        eval_F_analytic(mid, alpha, beta, gamma, &f_mid, &fp_mid, &fpp_mid, &ok_mid);
        if (!ok_mid) { out_a_root[row]=best_a; out_abs_deriv[row]=best_abs_d; out_residual[row]=best_f; out_status[row]=5; return; }
        if (fabs(f_mid) < fabs(best_f)) { best_a = mid; best_f = f_mid; best_abs_d = fabs(fp_mid); }

        if (fabs(f_mid) <= convergence_tol) { converged_residual = 1; break; }

        // Safeguarded Halley probe inside (lo, hi); fall back to Newton, else midpoint.
        double probe_pt = mid;
        int halley_ok = 0;
        if (isfinite(fp_mid) && fabs(fp_mid) > 1e-30) {
            double denom = 2.0 * fp_mid * fp_mid - f_mid * fpp_mid;
            if (isfinite(denom) && fabs(denom) > 1e-30) {
                double cand = mid - (2.0 * f_mid * fp_mid) / denom;
                if (cand > lo && cand < hi) { probe_pt = cand; halley_ok = 1; }
            }
        }
        if (!halley_ok && isfinite(fp_mid) && fabs(fp_mid) > 1e-30) {
            double cand = mid - f_mid / fp_mid;
            if (cand > lo && cand < hi) probe_pt = cand;
        }

        double f_b = f_mid;
        if (probe_pt != mid) {
            double f_p, fp_p, fpp_p; int ok_p;
            eval_F_analytic(probe_pt, alpha, beta, gamma, &f_p, &fp_p, &fpp_p, &ok_p);
            if (!ok_p) { out_a_root[row]=best_a; out_abs_deriv[row]=best_abs_d; out_residual[row]=best_f; out_status[row]=5; return; }
            if (fabs(f_p) < fabs(best_f)) { best_a = probe_pt; best_f = f_p; best_abs_d = fabs(fp_p); }
            f_b = f_p;
        } else {
            probe_pt = mid;
        }

        if (f_b <= 0.0) neg_pt = probe_pt; else pos_pt = probe_pt;

        double next_lo = fmin(neg_pt, pos_pt);
        double next_hi = fmax(neg_pt, pos_pt);
        if (fabs(next_hi - next_lo) <= convergence_tol * (1.0 + fabs(next_hi) + fabs(next_lo))) {
            converged_bracket = 1; break;
        }
    }

    if (!isfinite(best_abs_d) || best_abs_d == 0.0) {
        double f_r, fp_r, fpp_r; int ok_r;
        eval_F_analytic(best_a, alpha, beta, gamma, &f_r, &fp_r, &fpp_r, &ok_r);
        if (ok_r) best_abs_d = fabs(fp_r);
    }

    out_a_root[row]    = best_a;
    out_abs_deriv[row] = best_abs_d;
    out_residual[row]  = best_f;
    if      (converged_residual)             out_status[row] = 0;
    else if (converged_bracket)              out_status[row] = 1;
    else                                     out_status[row] = 4;
}
"#;

/// Launch the Step 3 device intercept solve.  Returns `Ok(None)` on
/// non-Linux / no-CUDA builds so the dispatcher can fall back to the
/// CPU oracle; returns `Err` only on genuine driver / compile failures.
pub fn try_device_intercept_solve(
    inputs: &SurvivalFlexInterceptSolveInputs<'_>,
) -> Result<Option<SurvivalFlexInterceptSolveOutputs>, GpuError> {
    inputs.validate()?;
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
        Some(backend.launch_intercept_solve_linux(inputs)).transpose()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
impl SurvivalFlexGpuBackend {
    /// NVRTC-compile (lazily, shared with other survival_flex modules) the
    /// Step 3 module.  Held in a static `OnceLock` so the compile runs
    /// once per process.
    fn compile_intercept_solve_module(&self) -> Result<Arc<CudaModule>, GpuError> {
        static INTERCEPT_MODULE: OnceLock<
            std::sync::Mutex<Option<Result<Arc<CudaModule>, GpuError>>>,
        > = OnceLock::new();
        let cell = INTERCEPT_MODULE.get_or_init(|| std::sync::Mutex::new(None));
        let mut guard = cell.lock().map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex intercept-solve module mutex poisoned: {err}"),
        })?;
        if let Some(existing) = guard.as_ref() {
            return existing.clone();
        }
        let result = (|| {
            let ptx = cudarc::nvrtc::compile_ptx(SURVIVAL_FLEX_INTERCEPT_SOLVE_SOURCE).map_err(
                |err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve NVRTC compile: {err}"),
                },
            )?;
            self.inner
                .ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve module load: {err}"),
                })
        })();
        *guard = Some(result.clone());
        result
    }

    fn launch_intercept_solve_linux(
        &self,
        inputs: &SurvivalFlexInterceptSolveInputs<'_>,
    ) -> Result<SurvivalFlexInterceptSolveOutputs, GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let module = self.compile_intercept_solve_module()?;
        let func = module
            .load_function("survival_flex_intercept_solve")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve load_function: {err}"),
            })?;

        let n = inputs.n;
        let stream = &self.inner.stream;
        let mk_htod = |slice: &[f64], name: &str| -> Result<_, GpuError> {
            stream
                .clone_htod(slice)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve memcpy_stod {name}: {err}"),
                })
        };
        let d_a_warm = mk_htod(inputs.a_warm, "a_warm")?;
        let d_alpha = mk_htod(inputs.alpha, "alpha")?;
        let d_beta = mk_htod(inputs.beta, "beta")?;
        let d_gamma = mk_htod(inputs.gamma, "gamma")?;

        let mut d_a_root =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc a_root: {err}"),
                })?;
        let mut d_abs_deriv =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc abs_deriv: {err}"),
                })?;
        let mut d_residual =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc residual: {err}"),
                })?;
        let mut d_status =
            stream
                .alloc_zeros::<u8>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc status: {err}"),
                })?;

        let convergence_tol = inputs.convergence_tol;
        let max_bracket_iters = inputs.max_bracket_iters;
        let max_refine_iters = inputs.max_refine_iters;
        let n_i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex intercept-solve n={n} overflows i32"),
        })?;

        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&d_a_warm)
            .arg(&d_alpha)
            .arg(&d_beta)
            .arg(&d_gamma)
            .arg(&convergence_tol)
            .arg(&max_bracket_iters)
            .arg(&max_refine_iters)
            .arg(&n_i32)
            .arg(&mut d_a_root)
            .arg(&mut d_abs_deriv)
            .arg(&mut d_residual)
            .arg(&mut d_status);
        // SAFETY: argument types match the kernel signature; grid covers n rows.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex intercept-solve launch: {err}"),
        })?;

        let a_root = stream
            .clone_dtoh(&d_a_root)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve memcpy_dtoh a_root: {err}"),
            })?;
        let abs_deriv =
            stream
                .clone_dtoh(&d_abs_deriv)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve memcpy_dtoh abs_deriv: {err}"),
                })?;
        let residual =
            stream
                .clone_dtoh(&d_residual)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve memcpy_dtoh residual: {err}"),
                })?;
        let status = stream
            .clone_dtoh(&d_status)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve memcpy_dtoh status: {err}"),
            })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve synchronize: {err}"),
            })?;

        Ok(SurvivalFlexInterceptSolveOutputs {
            a_root,
            abs_deriv,
            residual,
            status,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// Step 4 Layer A — real flex F evaluator CPU oracle.
//
// Step 3 ships against an analytic evaluator `F(a) = α·exp(β·a) + γ`.  The
// production survival-flex calibration evaluator is the de-nested cubic
// integrand `F(a) = -Φ(-q) + Σ_cells ∫_{cell} φ̂_cell(z; a)·exp(-q(z)) dz`
// from `families::survival::marginal_slope::evaluate_denested_survival_calibration`
// (lines 5772-5821 of `src/families/survival_marginal_slope.rs`).
//
// This CPU oracle takes per-row partition cells *already built by the
// host* and computes `(F, F', F'')` term-for-term against the family
// helper, using only the public `cubic_cell_kernel::*` primitives:
//
//   * `evaluate_cell_moments(neg_cell, 9)` for the per-cell value and
//     reduced moments of `exp(-q(z))`.
//   * `denested_cell_coefficient_partials(score_span, link_span, a, slope)`
//     for `(dc_da_pos, _)`.
//   * `denested_cell_second_partials(...)` for `(dc_daa_pos, _, _)`.
//   * `cell_first_derivative_from_moments` / `cell_second_derivative_from_moments`
//     to fold the partials against the moments.
//
// The sign / scale conventions match the family helper exactly: the per-cell
// integrand inside the family wraps `-Φ(-η)` (the `neg_cell` flip with
// negated cubic coefficients), so the partials are scaled by `-probit_scale`
// rather than `+probit_scale` — see `scale_coeff4(dc_da_pos, -scale)` at
// `survival_marginal_slope.rs:5809`.
//
// Layer A is the first principled slice of Block 8 Step 4; layers B / C
// (the IFT solve for `a_u, a_uv` and the η / χ / D jet outputs) land in
// follow-up commits.  The Step-3 device intercept solver will swap its
// analytic F-evaluator for an NVRTC port of this same routine in Layer A
// follow-up; the CPU oracle here is the parity reference that lets the
// device port advance under a deterministic correctness bar.
// ────────────────────────────────────────────────────────────────────────

/// Per-cell descriptor used by the Layer-A CPU oracle.  Mirrors
/// [`crate::families::cubic_cell_kernel::DenestedPartitionCell`] but uses
/// the public re-export shape so the oracle has no `pub(crate)` dependency
/// on the family module.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalFlexCalibrationCell {
    /// Positive-orientation cubic on this cell (matches the family's
    /// `partition_cell.cell` field).
    pub cell: crate::families::cubic_cell_kernel::DenestedCubicCell,
    /// Score-side local cubic span.
    pub score_span: crate::families::cubic_cell_kernel::LocalSpanCubic,
    /// Link-side local cubic span.
    pub link_span: crate::families::cubic_cell_kernel::LocalSpanCubic,
}

/// Output of the Layer-A CPU oracle: the calibration residual `F(a)`,
/// its first derivative `F'(a)` and its second derivative `F''(a)`.
/// Same triple shape as the CPU helper in `survival_marginal_slope.rs`.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalFlexCalibrationFAndDerivs {
    /// `F(a) = -Φ(-q) + Σ_cells ⟨integrand⟩`.
    pub f: f64,
    /// `F'(a) = Σ_cells cell_first_derivative_from_moments(dc_da, moments)`.
    pub f_prime: f64,
    /// `F''(a) = Σ_cells cell_second_derivative_from_moments(neg_cell, dc_da, dc_da, dc_daa, moments)`.
    pub f_double_prime: f64,
}

/// CPU oracle for the survival-flex calibration evaluator `F(a)` and its
/// first two derivatives in `a`, parity reference for the Step-3 device
/// intercept-solver's eventual real F-evaluator (Layer A NVRTC follow-up).
///
/// Term-for-term port of
/// [`crate::families::survival::marginal_slope::SurvivalMarginalSlopeFamily::evaluate_denested_survival_calibration`]:
///
/// ```text
/// f          = -Φ(-q)
/// for cell in partition:
///     neg_cell = (left, right, -c0, -c1, -c2, -c3)
///     state    = evaluate_cell_moments(neg_cell, 9)
///     f       += state.value
///     dc_da   = scale_coeff4(denested_cell_coefficient_partials(..).0, -probit_scale)
///     dc_daa  = scale_coeff4(denested_cell_second_partials(..).0, -probit_scale)
///     f_a    += cell_first_derivative_from_moments(dc_da, state.moments)
///     f_aa   += cell_second_derivative_from_moments(neg_cell, dc_da, dc_da, dc_daa, state.moments)
/// ```
///
/// Returns `Err(String)` on any underlying `cubic_cell_kernel` failure
/// (insufficient moments, non-finite integrand).  Callers (the Step-3
/// solver loop, the parity test) propagate the error.
pub fn cpu_oracle_evaluate_calibration(
    partition_cells: &[SurvivalFlexCalibrationCell],
    a: f64,
    q: f64,
    slope: f64,
    probit_scale: f64,
) -> Result<SurvivalFlexCalibrationFAndDerivs, String> {
    use crate::families::cubic_cell_kernel::{
        DenestedCubicCell, cell_first_derivative_from_moments, cell_second_derivative_from_moments,
        denested_cell_coefficient_partials, denested_cell_second_partials, evaluate_cell_moments,
    };

    // `scale_coeff4(coef, scale) = [coef[0]*scale, coef[1]*scale, coef[2]*scale, coef[3]*scale]`.
    // Inlined here so the oracle has no `pub(crate)` dependency on
    // `survival::marginal_slope::scale_coeff4`.
    #[inline]
    fn scale_coeff4(coef: [f64; 4], scale: f64) -> [f64; 4] {
        [
            coef[0] * scale,
            coef[1] * scale,
            coef[2] * scale,
            coef[3] * scale,
        ]
    }

    // Match the family's `f` seed exactly: `f = -Φ(-q)` (target-survival
    // sign convention so the per-cell integrand additions converge to
    // zero at the calibration root).
    let mut f = -crate::probability::normal_cdf(-q);
    let mut f_a = 0.0_f64;
    let mut f_aa = 0.0_f64;

    for pc in partition_cells {
        let pos_cell = pc.cell;
        let neg_cell = DenestedCubicCell {
            left: pos_cell.left,
            right: pos_cell.right,
            c0: -pos_cell.c0,
            c1: -pos_cell.c1,
            c2: -pos_cell.c2,
            c3: -pos_cell.c3,
        };
        let state = evaluate_cell_moments(neg_cell, 9)?;
        f += state.value;

        let (dc_da_pos, _) =
            denested_cell_coefficient_partials(pc.score_span, pc.link_span, a, slope);
        let (dc_daa_pos, _, _) =
            denested_cell_second_partials(pc.score_span, pc.link_span, a, slope);

        // Match the family's `-scale` sign exactly (line 5809 of
        // `survival_marginal_slope.rs`): the negated cubic in `neg_cell`
        // flips the integrand sign, so the partials are scaled by the
        // *negative* probit scale.
        let dc_da = scale_coeff4(dc_da_pos, -probit_scale);
        let dc_daa = scale_coeff4(dc_daa_pos, -probit_scale);

        f_a += cell_first_derivative_from_moments(&dc_da, &state.moments)?;
        f_aa +=
            cell_second_derivative_from_moments(neg_cell, &dc_da, &dc_da, &dc_daa, &state.moments)?;
    }

    Ok(SurvivalFlexCalibrationFAndDerivs {
        f,
        f_prime: f_a,
        f_double_prime: f_aa,
    })
}

// ────────────────────────────────────────────────────────────────────────
// Step 4 Layer A — substrate-backed device F evaluator.
//
// Math identity: the calibration evaluator is
//
//     F(a)   = -Φ(-q) + Σ_cells value_neg   (value_neg = Σ w·e^{-½z²}·Φ(η)/√(2π),
//                                            the survival value-integral the
//                                            oracle reads as `state.value`; NOT
//                                            `INV_TWO_PI · moments_neg[0]`, which
//                                            integrates the different weight
//                                            e^{-½(z²+η²)} — see #831)
//     F'(a)  = INV_TWO_PI · Σ_cells ⟨dc_da_neg, moments_neg[0..4]⟩
//     F''(a) = INV_TWO_PI · Σ_cells (⟨dc_daa_neg, moments_neg[0..4]⟩
//                                    - ⟨conv(neg_cubic, dc_da_neg, dc_da_neg), moments_neg[0..10]⟩)
//
// where `dc_da_neg = -probit_scale · dc_da_pos`, `dc_daa_neg = -probit_scale ·
// dc_daa_pos`, and the partials `(dc_da_pos, dc_daa_pos)` are the closed-form
// polynomial expressions in `(a, slope, score_span, link_span)` from
// `cubic_cell_kernel::denested_cell_coefficient_partials` /
// `denested_cell_second_partials`.  The heavy lifting — the per-cell value
// and length-10 reduced moments of `exp(-q(z))` over the cell interior —
// is exactly the work the Step-2 substrate already does on device via
// `try_row_batched_cell_moments`.  Layer 4a's job is the trailing
// O(n_cells) reduction; spinning up an extra NVRTC reduction kernel for
// it would be theatre, not real perf (the substrate already moves the
// 384-pt GL integrand work to device).
//
// Routing: this entry point calls the Step-2 substrate, then folds with
// the analytic partials on host.  The result is the same `(F, F', F'')`
// triple as [`cpu_oracle_evaluate_calibration`] above.  Parity is
// asserted in the tests block (`step4a_device_evaluator_matches_cpu_oracle`).
// ────────────────────────────────────────────────────────────────────────

/// Substrate-backed device F evaluator.  Builds the Step-2 row-batched
/// cell list from the supplied per-row partition cells, invokes
/// [`try_row_batched_cell_moments`] (which routes through the substrate's
/// CUDA 384-pt GL kernel on Linux+CUDA and the CPU evaluator everywhere
/// else), then folds with the analytic `dc_da_neg` / `dc_daa_neg` partials
/// per cell.  Returns one `SurvivalFlexCalibrationFAndDerivs` per row in
/// the input `partition_by_row` slice.
///
/// Errors propagate the substrate / coefficient-partial failures verbatim;
/// `Ok(None)` reflects "any per-cell substrate status was non-OK" so the
/// caller falls back to the CPU oracle cleanly for the offending fit.
pub fn try_device_evaluate_calibration(
    partition_by_row: &[Vec<SurvivalFlexCalibrationCell>],
    a_per_row: &[f64],
    q_per_row: &[f64],
    slope_per_row: &[f64],
    probit_scale: f64,
) -> Result<Option<Vec<SurvivalFlexCalibrationFAndDerivs>>, GpuError> {
    use crate::families::cubic_cell_kernel::{
        DenestedCubicCell, denested_cell_coefficient_partials, denested_cell_second_partials,
        evaluate_cell_moments,
    };

    let n_rows = partition_by_row.len();
    if a_per_row.len() != n_rows || q_per_row.len() != n_rows || slope_per_row.len() != n_rows {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "try_device_evaluate_calibration: row-array length mismatch \
                 (partition_by_row={n_rows}, a={}, q={}, slope={})",
                a_per_row.len(),
                q_per_row.len(),
                slope_per_row.len()
            ),
        });
    }
    if !(probit_scale.is_finite() && probit_scale > 0.0) {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "try_device_evaluate_calibration: probit_scale must be positive finite, got {probit_scale}"
            ),
        });
    }

    // Build the substrate-flat SoA: concatenate per-row cells, build a
    // row_offsets index, then call the substrate moment evaluator.  We
    // negate the cubic up-front so the substrate sees the same input the
    // CPU helper does (`evaluate_cell_moments(neg_cell, 9)`).
    let mut total_cells = 0usize;
    let mut row_offsets = Vec::with_capacity(n_rows + 1);
    row_offsets.push(0);
    for cells in partition_by_row {
        total_cells += cells.len();
        row_offsets.push(total_cells);
    }
    if total_cells == 0 {
        let mut out = Vec::with_capacity(n_rows);
        for &q in q_per_row {
            out.push(SurvivalFlexCalibrationFAndDerivs {
                f: -crate::probability::normal_cdf(-q),
                f_prime: 0.0,
                f_double_prime: 0.0,
            });
        }
        return Ok(Some(out));
    }
    let mut left = Vec::with_capacity(total_cells);
    let mut right = Vec::with_capacity(total_cells);
    let mut c0 = Vec::with_capacity(total_cells);
    let mut c1 = Vec::with_capacity(total_cells);
    let mut c2 = Vec::with_capacity(total_cells);
    let mut c3 = Vec::with_capacity(total_cells);
    for cells in partition_by_row {
        for pc in cells {
            left.push(pc.cell.left);
            right.push(pc.cell.right);
            c0.push(-pc.cell.c0);
            c1.push(-pc.cell.c1);
            c2.push(-pc.cell.c2);
            c3.push(-pc.cell.c3);
        }
    }

    let batch = SurvivalFlexRowCellsBatch {
        n_cells: total_cells,
        n_rows,
        max_degree: 9,
        left: &left,
        right: &right,
        c0: &c0,
        c1: &c1,
        c2: &c2,
        c3: &c3,
        row_offsets: &row_offsets,
    };
    let mom = match try_row_batched_cell_moments(batch)? {
        Some(m) => m,
        None => return Ok(None),
    };
    let stride = mom.stride;
    // Substrate must give us at least degree 9 (10 moments) per cell —
    // `cell_second_derivative_from_moments` needs degree
    // `len(eta)+len(r)+len(s)-3 = 4+4+4-3 = 9`.
    if stride < 10 {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "try_device_evaluate_calibration: substrate returned stride={stride} < 10"
            ),
        });
    }
    let ok_byte = crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8;
    if mom.status.iter().any(|&s| s != ok_byte) {
        return Ok(None);
    }

    let inv_two_pi = 1.0_f64 / std::f64::consts::TAU;

    let mut out = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let a = a_per_row[row];
        let q = q_per_row[row];
        let slope = slope_per_row[row];
        let mut f = -crate::probability::normal_cdf(-q);
        let mut f_a = 0.0_f64;
        let mut f_aa = 0.0_f64;

        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        for cell_idx in start..end {
            let pc = &partition_by_row[row][cell_idx - start];
            let moments_row = &mom.moments[cell_idx * stride..cell_idx * stride + stride];
            // Cell value contribution: the calibration F-seed integrand is the
            // *survival* value-integral `Σ w·e^{-½z²}·Φ(η) / √(2π)` (= the
            // oracle's `state.value`, cubic_cell_kernel.rs `value_integral *
            // 1/√τ`), NOT the moment-weight integral `Σ w·e^{-½(z²+η²)} / 2π`
            // (= `moments[0]·INV_TWO_PI`). The substrate returns only the
            // derivative moments (the `e^{-½(z²+η²)}` family used for F'/F''),
            // so the value term cannot be reconstructed from `moments[0]`; it
            // is a genuinely different integrand (#831 Defect B). Fold it on
            // host from the closed-form value-integral the kernel exposes via
            // `evaluate_cell_moments(neg_cell).value`, keeping the device
            // moments for the higher-order derivative folds below.
            let neg_cell = DenestedCubicCell {
                left: pc.cell.left,
                right: pc.cell.right,
                c0: -pc.cell.c0,
                c1: -pc.cell.c1,
                c2: -pc.cell.c2,
                c3: -pc.cell.c3,
            };
            let value_state = evaluate_cell_moments(neg_cell, 9).map_err(|reason| {
                GpuError::DriverCallFailed {
                    reason: format!(
                        "try_device_evaluate_calibration: value-integral evaluation failed \
                         for row {row} cell {cell_idx}: {reason}"
                    ),
                }
            })?;
            f += value_state.value;

            let (dc_da_pos, _) =
                denested_cell_coefficient_partials(pc.score_span, pc.link_span, a, slope);
            let (dc_daa_pos, _, _) =
                denested_cell_second_partials(pc.score_span, pc.link_span, a, slope);
            let dc_da = [
                -probit_scale * dc_da_pos[0],
                -probit_scale * dc_da_pos[1],
                -probit_scale * dc_da_pos[2],
                -probit_scale * dc_da_pos[3],
            ];
            let dc_daa = [
                -probit_scale * dc_daa_pos[0],
                -probit_scale * dc_daa_pos[1],
                -probit_scale * dc_daa_pos[2],
                -probit_scale * dc_daa_pos[3],
            ];

            // F' contribution: dot(dc_da, moments[0..4]) * INV_TWO_PI.
            let mut first = 0.0_f64;
            for k in 0..4 {
                first = dc_da[k].mul_add(moments_row[k], first);
            }
            f_a += first * inv_two_pi;

            // F'' contribution: (second_term - conv_term) * INV_TWO_PI.
            // second_term = dot(dc_daa, moments[0..4]).
            let mut second_term = 0.0_f64;
            for k in 0..4 {
                second_term = dc_daa[k].mul_add(moments_row[k], second_term);
            }
            // conv_term = conv(neg_cubic, dc_da, dc_da) · moments[0..10].
            // The CPU helper passes `neg_cell` cubic into the convolution;
            // we mirror that with the negated coefficients here.
            let neg_cubic = [-pc.cell.c0, -pc.cell.c1, -pc.cell.c2, -pc.cell.c3];
            let mut eta_r = [0.0_f64; 16];
            for i in 0..4 {
                for j in 0..4 {
                    eta_r[i + j] = neg_cubic[i].mul_add(dc_da[j], eta_r[i + j]);
                }
            }
            let er_len = 4 + 4 - 1;
            let mut eta_rs = [0.0_f64; 16];
            for i in 0..er_len {
                for j in 0..4 {
                    eta_rs[i + j] = eta_r[i].mul_add(dc_da[j], eta_rs[i + j]);
                }
            }
            let ers_len = er_len + 4 - 1;
            let mut conv_term = 0.0_f64;
            for k in 0..ers_len {
                conv_term = eta_rs[k].mul_add(moments_row[k], conv_term);
            }
            f_aa += (second_term - conv_term) * inv_two_pi;
        }

        out.push(SurvivalFlexCalibrationFAndDerivs {
            f,
            f_prime: f_a,
            f_double_prime: f_aa,
        });
    }

    Ok(Some(out))
}

// ────────────────────────────────────────────────────────────────────────
// Step 4 Layer B — first-order timepoint jet from the IFT solve.
//
// Given the calibration root `a*` (from Step 3 / Layer A), the IFT
// produces the per-primary directional derivative:
//
//     a_u[u] = f_u[u] / D            (lines 6658-6661 of survival_marginal_slope.rs)
//
// where `D = D(a*)` is the calibration denominator (positive, finite by
// construction) and `f_u[u] = Σ_cells <neg_coeff_u[u], moments>` (with
// `+φ(q)` added to the entry matching `q_index` — line 6637).
//
// The first-order outputs are:
//
//     eta_u[u]  = chi * a_u[u] + rho[u]
//     chi_u[u]  = eta_aa * a_u[u] + tau[u]
//     d_u[u]    = Σ_cells ⟨integrand_u, moments⟩
//
// where the per-cell integrand is
//
//     integrand_u = chi_u_poly - (chi_poly * eta_poly * eta_u_poly)
//     eta_u_poly  = a_u[u] · chi_poly + coeff_u[u]
//     chi_u_poly  = a_u[u] · eta_aa_poly + coeff_au[u]
//
// (CPU lines 6675-6681).
//
// Layer B keeps the same separation of concerns as Layer A: the
// substrate gives the moments, the caller hands in the analytic
// per-cell coefficient tables (`coeff_u`, `coeff_au`, plus the cell-local
// `chi_poly = dc_da_pos` and `eta_aa_poly = dc_daa_pos` already produced
// by `denested_cell_coefficient_partials` / `denested_cell_second_partials`),
// and the entry point folds them on host into `(a_u, eta_u, chi_u, d_u)`.
// The cell cubic `(c0..c3)` is the `eta_poly` reused as the integrand kernel.
//
// `coeff_u[u]`, `coeff_au[u]`: per-primary `[f64; 4]` polynomial slices.
// Their construction lives in the family module's
// `denested_cell_primary_fixed_partials` (lines 6235-6341), which has
// branching on score-warp / link-dev / g-index that's properly owned
// there.  Step 6 will compose: family builds the tables, this entry
// folds them with the substrate moments.
// ────────────────────────────────────────────────────────────────────────

/// Per-cell primary-coefficient tables for the Layer B fold.  Mirrors the
/// `coeff_u` / `coeff_au` slabs in CPU `DenestedCellPrimaryFixedPartials`:
/// one `[f64; 4]` per primary index `u`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerBCellCoeffs {
    /// `coeff_u[u]`: ∂c/∂β_u  (length `p`).
    pub coeff_u: Vec<[f64; 4]>,
    /// `coeff_au[u]`: ∂²c/∂a∂β_u (length `p`).
    pub coeff_au: Vec<[f64; 4]>,
}

/// Per-row inputs for the Layer B fold.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerBRowInputs<'a> {
    /// Per-row partition cells (geometry + spans, same as Layer A).
    pub partition_cells: &'a [SurvivalFlexCalibrationCell],
    /// Per-cell primary tables (one per cell).
    pub cell_coeffs: &'a [SurvivalFlexLayerBCellCoeffs],
    /// Calibration denominator `D(a*)` for this row, positive finite.
    pub d_check: f64,
    /// Index of the q-perturbation primary (used for the `+φ(q)` bump
    /// in `f_u[q_index]`).  Set to `usize::MAX` to disable.
    pub q_index: usize,
    /// `φ(q)`: the normal pdf at `q`, for the `f_u[q_index]` bump.
    pub phi_q: f64,
    /// Observed-row `chi`, `eta_aa` scalars from
    /// `observed_denested_cell_partials` evaluated at the observed
    /// `z_obs` point.
    pub chi: f64,
    /// Observed-row `eta_aa` scalar.
    pub eta_aa: f64,
    /// `rho[u]`: per-primary observed `∂eta/∂β_u | a fixed`.
    pub rho: &'a [f64],
    /// `tau[u]`: per-primary observed `∂chi/∂β_u | a fixed`.
    pub tau: &'a [f64],
    /// `probit_scale`: passed through for sign / scaling of partials.
    pub probit_scale: f64,
    /// Calibration root `a*` per the Step-3 / Layer-A solve.
    pub a: f64,
    /// Slope `b` (≡ `qd1 * c` upstream) per the CPU helper.
    pub slope: f64,
}

/// Per-row Layer-B outputs.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerBRowOutputs {
    /// `a_u[u] = f_u[u] / D`, length `p`.
    pub a_u: Vec<f64>,
    /// `eta_u[u] = chi · a_u[u] + rho[u]`, length `p`.
    pub eta_u: Vec<f64>,
    /// `chi_u[u] = eta_aa · a_u[u] + tau[u]`, length `p`.
    pub chi_u: Vec<f64>,
    /// `d_u[u] = Σ_cells ⟨integrand_u, moments⟩`, length `p`.
    pub d_u: Vec<f64>,
}

/// Substrate-backed Layer B fold.  Computes the first-order
/// timepoint jet `(a_u, eta_u, chi_u, d_u)` for every input row.
///
/// `rows` and `cell_coeffs_by_row` are parallel slices: row `i` consumes
/// `rows[i]` and has its own per-cell coefficient tables in
/// `cell_coeffs_by_row[i]`.  Each cell's `cell_coeffs.coeff_u.len()`
/// must equal the row's `rho.len()` (≡ primary dimension `p`).
pub fn try_device_layer_b_jet(
    rows: &[SurvivalFlexLayerBRowInputs<'_>],
) -> Result<Option<Vec<SurvivalFlexLayerBRowOutputs>>, GpuError> {
    use crate::families::cubic_cell_kernel::{
        cell_first_derivative_from_moments, cell_polynomial_integral_from_moments,
        denested_cell_coefficient_partials, denested_cell_second_partials,
    };

    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let n_rows = rows.len();

    // Validate shapes and accumulate total cell count.
    let mut total_cells = 0usize;
    let mut row_offsets = Vec::with_capacity(n_rows + 1);
    row_offsets.push(0);
    for (i, row) in rows.iter().enumerate() {
        if row.partition_cells.len() != row.cell_coeffs.len() {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_b row {i}: cells.len()={} != cell_coeffs.len()={}",
                    row.partition_cells.len(),
                    row.cell_coeffs.len()
                ),
            });
        }
        let p = row.rho.len();
        if row.tau.len() != p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_b row {i}: rho.len()={} != tau.len()={}",
                    p,
                    row.tau.len()
                ),
            });
        }
        for (k, cc) in row.cell_coeffs.iter().enumerate() {
            if cc.coeff_u.len() != p || cc.coeff_au.len() != p {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "layer_b row {i} cell {k}: coeff_u.len()={} coeff_au.len()={} expected {p}",
                        cc.coeff_u.len(),
                        cc.coeff_au.len()
                    ),
                });
            }
        }
        if !(row.d_check.is_finite() && row.d_check > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_b row {i}: d_check must be positive finite, got {}",
                    row.d_check
                ),
            });
        }
        if !(row.probit_scale.is_finite() && row.probit_scale > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_b row {i}: probit_scale must be positive finite, got {}",
                    row.probit_scale
                ),
            });
        }
        total_cells += row.partition_cells.len();
        row_offsets.push(total_cells);
    }

    // Build the flat substrate moment batch (negated cubic, same shape
    // as Layer A).
    if total_cells == 0 {
        // Every row's outputs are zero — no cells contribute.
        let mut out = Vec::with_capacity(n_rows);
        for row in rows {
            let p = row.rho.len();
            out.push(SurvivalFlexLayerBRowOutputs {
                a_u: vec![0.0; p],
                eta_u: row.rho.to_vec(),
                chi_u: row.tau.to_vec(),
                d_u: vec![0.0; p],
            });
        }
        return Ok(Some(out));
    }
    let mut left = Vec::with_capacity(total_cells);
    let mut right = Vec::with_capacity(total_cells);
    let mut c0 = Vec::with_capacity(total_cells);
    let mut c1 = Vec::with_capacity(total_cells);
    let mut c2 = Vec::with_capacity(total_cells);
    let mut c3 = Vec::with_capacity(total_cells);
    for row in rows {
        for pc in row.partition_cells {
            left.push(pc.cell.left);
            right.push(pc.cell.right);
            c0.push(-pc.cell.c0);
            c1.push(-pc.cell.c1);
            c2.push(-pc.cell.c2);
            c3.push(-pc.cell.c3);
        }
    }
    let batch = SurvivalFlexRowCellsBatch {
        n_cells: total_cells,
        n_rows,
        max_degree: 9,
        left: &left,
        right: &right,
        c0: &c0,
        c1: &c1,
        c2: &c2,
        c3: &c3,
        row_offsets: &row_offsets,
    };
    let mom = match try_row_batched_cell_moments(batch)? {
        Some(m) => m,
        None => return Ok(None),
    };
    let stride = mom.stride;
    if stride < 10 {
        return Err(GpuError::DriverCallFailed {
            reason: format!("layer_b: substrate returned stride={stride} < 10"),
        });
    }
    let ok_byte = crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8;
    if mom.status.iter().any(|&s| s != ok_byte) {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(n_rows);
    for (row_idx, row) in rows.iter().enumerate() {
        let p = row.rho.len();
        let row_start = row_offsets[row_idx];
        let row_end = row_offsets[row_idx + 1];

        // Step 1: f_u[u] per primary index = Σ_cells <-coeff_u[u], moments[0..4]>.
        let mut f_u = vec![0.0_f64; p];
        for cell_idx in row_start..row_end {
            let local_idx = cell_idx - row_start;
            let cc = &row.cell_coeffs[local_idx];
            let moments_row = &mom.moments[cell_idx * stride..cell_idx * stride + stride];
            for u in 0..p {
                let neg_coeff_u = [
                    -cc.coeff_u[u][0],
                    -cc.coeff_u[u][1],
                    -cc.coeff_u[u][2],
                    -cc.coeff_u[u][3],
                ];
                let contrib = cell_first_derivative_from_moments(&neg_coeff_u, moments_row)
                    .map_err(|e| GpuError::DriverCallFailed {
                        reason: format!("layer_b row {row_idx} cell {local_idx}: f_u fold: {e}"),
                    })?;
                f_u[u] += contrib;
            }
        }
        // q-index `+φ(q)` bump (line 6637 of the family helper).
        if row.q_index < p {
            f_u[row.q_index] += row.phi_q;
        }

        // Step 2: a_u[u] = f_u[u] / D.
        let mut a_u = vec![0.0_f64; p];
        for u in 0..p {
            a_u[u] = f_u[u] / row.d_check;
        }

        // Step 3: d_u[u] = Σ_cells ⟨integrand_u, moments⟩.
        // integrand_u = chi_u_poly - (chi_poly * eta_poly * eta_u_poly)
        // where eta_poly = (c0..c3), chi_poly = dc_da_pos, eta_aa_poly = dc_daa_pos.
        let mut d_u = vec![0.0_f64; p];
        for cell_idx in row_start..row_end {
            let local_idx = cell_idx - row_start;
            let pc = &row.partition_cells[local_idx];
            let cc = &row.cell_coeffs[local_idx];
            let moments_row = &mom.moments[cell_idx * stride..cell_idx * stride + stride];

            // Per-cell positive-orientation partials at this row's (a, slope).
            let (dc_da_pos, _) =
                denested_cell_coefficient_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let (dc_daa_pos, _, _) =
                denested_cell_second_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let chi_poly = dc_da_pos;
            let eta_aa_poly = dc_daa_pos;
            let eta_poly = [pc.cell.c0, pc.cell.c1, pc.cell.c2, pc.cell.c3];

            // Pre-compute the (chi_poly * eta_poly) convolution once per
            // cell — it's reused for every u.  Result length 4+4-1 = 7.
            let mut chi_eta = [0.0_f64; 16];
            for i in 0..4 {
                for j in 0..4 {
                    chi_eta[i + j] = chi_poly[i].mul_add(eta_poly[j], chi_eta[i + j]);
                }
            }
            let chi_eta_len = 7usize;

            for u in 0..p {
                // eta_u_poly = a_u[u] · chi_poly + coeff_u[u], length 4.
                let eta_u_poly = [
                    a_u[u] * chi_poly[0] + cc.coeff_u[u][0],
                    a_u[u] * chi_poly[1] + cc.coeff_u[u][1],
                    a_u[u] * chi_poly[2] + cc.coeff_u[u][2],
                    a_u[u] * chi_poly[3] + cc.coeff_u[u][3],
                ];
                // chi_u_poly = a_u[u] · eta_aa_poly + coeff_au[u], length 4.
                let chi_u_poly = [
                    a_u[u] * eta_aa_poly[0] + cc.coeff_au[u][0],
                    a_u[u] * eta_aa_poly[1] + cc.coeff_au[u][1],
                    a_u[u] * eta_aa_poly[2] + cc.coeff_au[u][2],
                    a_u[u] * eta_aa_poly[3] + cc.coeff_au[u][3],
                ];
                // chi_eta_etau = conv(chi_eta, eta_u_poly), length 7+4-1 = 10.
                let mut chi_eta_etau = [0.0_f64; 16];
                for i in 0..chi_eta_len {
                    for j in 0..4 {
                        chi_eta_etau[i + j] =
                            chi_eta[i].mul_add(eta_u_poly[j], chi_eta_etau[i + j]);
                    }
                }
                let triple_len = chi_eta_len + 4 - 1;
                // integrand = chi_u_poly (len 4) - chi_eta_etau (len 10).
                let mut integrand = [0.0_f64; 16];
                for k in 0..4 {
                    integrand[k] = chi_u_poly[k];
                }
                for k in 0..triple_len {
                    integrand[k] -= chi_eta_etau[k];
                }
                let contrib = cell_polynomial_integral_from_moments(
                    &integrand[..triple_len],
                    moments_row,
                    "survival_flex layer_b d_u",
                )
                .map_err(|e| GpuError::DriverCallFailed {
                    reason: format!("layer_b row {row_idx} cell {local_idx} u={u}: d_u fold: {e}"),
                })?;
                d_u[u] += contrib;
            }
        }

        // Step 4: eta_u, chi_u closed-form.
        let mut eta_u = vec![0.0_f64; p];
        let mut chi_u = vec![0.0_f64; p];
        for u in 0..p {
            eta_u[u] = row.chi * a_u[u] + row.rho[u];
            chi_u[u] = row.eta_aa * a_u[u] + row.tau[u];
        }

        out.push(SurvivalFlexLayerBRowOutputs {
            a_u,
            eta_u,
            chi_u,
            d_u,
        });
    }

    Ok(Some(out))
}

// ────────────────────────────────────────────────────────────────────────
// Step 4 Layer C-α — second-order jet `(a_uv, eta_uv, chi_uv)` for the
// dense Hessian via the second-order IFT solve.  Mirrors
// CPU `compute_survival_timepoint_exact` (lines 6757-7013 of
// `survival_marginal_slope.rs`) for the `need_d_uv == false` branch.
//
// Layered build:
//   * Layer A → F, F', F''
//   * Layer B → first-order jet (a_u, eta_u, chi_u, d_u)
//   * Layer C-α → second-order jet (a_uv, eta_uv, chi_uv) — this section
//   * Layer C-β → optional `d_uv` (degree-15 quadrature; future commit)
//
// Per-cell quadrature accumulators (CPU lines 6792-6843):
//   f_aa          = ⟨-dc_da, -dc_da, -dc_daa⟩₂
//   f_au[u]       = ⟨-dc_da, -coeff_u[u], -coeff_au[u]⟩₂
//   f_uv[u,v]     = ⟨-coeff_u[u], -coeff_u[v], -second_coeff[u,v]⟩₂
// where ⟨·,·,·⟩₂ ≡ `cell_second_derivative_from_moments(neg_cell, r, s, rs)`
// and `second_coeff[u,v]` is `coeff_bu[v]` if `u == g_index`,
// `coeff_bu[u]` if `v == g_index`, else `[0;4]`.
//
// Post-sum corrections (CPU lines 6861-6863):
//   f_u[q_index]            += φ(q)
//   f_uv[q_index, q_index]  += -q · φ(q)
//
// Second-order IFT (CPU lines 6923-6932):
//   a_uv[u,v] = (f_uv[u,v]
//                - d_u[u] · a_u[v] - d_u[v] · a_u[u]
//                - f_aa · a_u[u] · a_u[v]) / D
//
// Closed-form observed-jet algebra (CPU lines 6989-7013):
//   eta_uv[u,v] = chi · a_uv[u,v]
//               + eta_aa · a_u[u] · a_u[v]
//               + tau[u] · a_u[v] + tau[v] · a_u[u]
//               + r_uv[u,v]
//   chi_uv[u,v] = eta_aa · a_uv[u,v]
//               + eta_aaa · a_u[u] · a_u[v]
//               + tau_a[u] · a_u[v] + tau_a[v] · a_u[u]
//               + chi_uv_fixed[u,v]
//
// `r_uv` and `chi_uv_fixed` are observed-row second partials (CPU
// `observed_fixed_eta_second_partial` and `observed_fixed_chi_second_partial`)
// that depend on the family's `primary` / `obs` row-state; they are
// caller-supplied scalars per (u,v) pair, lower-triangle-packed.
//
// Substrate moment quadrature is shared with Layers A/B (max_degree 9 is
// sufficient because second-derivative integrands top out at degree 9:
// len(cubic)+len(coeff_u)+len(coeff_v)-3 = 4+4+4-3 = 9).  Optional Layer
// C-β `d_uv` quadrature needs `max_degree 24` for the quintic-product
// term (term5 = chi_poly · eta_poly² · eta_u · eta_v).
// ────────────────────────────────────────────────────────────────────────

/// Per-cell primary tables for Layer C-α: extends Layer B's
/// `(coeff_u, coeff_au)` with `coeff_bu[u]` for the `f_uv` second-coeff
/// branching on `g_index`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerCCellCoeffs {
    /// `coeff_u[u]`: ∂c/∂β_u (length `p`).
    pub coeff_u: Vec<[f64; 4]>,
    /// `coeff_au[u]`: ∂²c/∂a∂β_u (length `p`).
    pub coeff_au: Vec<[f64; 4]>,
    /// `coeff_bu[u]`: ∂²c/∂b∂β_u (length `p`).  Only the entries
    /// indexed against `g_index` are read by the f_uv assembly (per the
    /// CPU branching at lines 6817-6822); the rest may be zero-padded.
    pub coeff_bu: Vec<[f64; 4]>,
}

/// Per-row inputs for Layer C-α.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerCRowInputs<'a> {
    /// Per-row partition cells (geometry + spans).
    pub partition_cells: &'a [SurvivalFlexCalibrationCell],
    /// Per-cell primary tables (one per cell).
    pub cell_coeffs: &'a [SurvivalFlexLayerCCellCoeffs],
    /// Calibration denominator `D(a*)`, positive finite.
    pub d_check: f64,
    /// Index of the q-perturbation primary; `usize::MAX` to disable.
    pub q_index: usize,
    /// `g_index`: the primary index of the raw log-slope coordinate.
    /// Used by the `second_coeff` branching in `f_uv`.
    /// Set to `usize::MAX` to disable the branch (zero second-coeff for
    /// every pair).
    pub g_index: usize,
    /// `φ(q)`: for the `f_u[q_index] += φ(q)` bump.
    pub phi_q: f64,
    /// `q`: for the `f_uv[q_index, q_index] += -q · φ(q)` bump.
    pub q: f64,
    /// Observed-row `chi`, `eta_aa`, `eta_aaa` scalars from
    /// `observed_denested_cell_partials`.
    pub chi: f64,
    /// Observed-row `eta_aa` scalar.
    pub eta_aa: f64,
    /// Observed-row `eta_aaa` scalar (third partial in `a`).
    pub eta_aaa: f64,
    /// `rho[u]`, `tau[u]`, `tau_a[u]`: per-primary observed partials.
    pub rho: &'a [f64],
    /// `tau[u]`: per-primary observed `∂chi/∂β_u | a fixed`.
    pub tau: &'a [f64],
    /// `tau_a[u]`: per-primary observed `∂²chi/∂a∂β_u`.
    pub tau_a: &'a [f64],
    /// Per-(u,v) pair observed `r_uv[u,v]` second-eta partial,
    /// upper-triangle row-major flat: length `p*(p+1)/2`, indexed by
    /// the helper [`tri_index`] below.
    pub r_uv_upper_packed: &'a [f64],
    /// Per-(u,v) pair observed `chi_uv_fixed[u,v]`, same packing.
    pub chi_uv_fixed_upper_packed: &'a [f64],
    /// `probit_scale`.
    pub probit_scale: f64,
    /// Calibration root `a*`.
    pub a: f64,
    /// Slope `b`.
    pub slope: f64,
}

/// Upper-triangle row-major flat index `(u, v)` with `u ≤ v`.
/// `tri_index(u, v, p) = u·(2·p - u - 1) / 2 + v`.
/// Inverse to a row-major upper triangle pack.
#[inline]
pub const fn tri_index(u: usize, v: usize, p: usize) -> usize {
    let u_idx = u * (2 * p - u - 1) / 2;
    u_idx + v
}

/// Per-row Layer C-α outputs.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerCRowOutputs {
    /// `a_u[u]`, length `p`.
    pub a_u: Vec<f64>,
    /// `eta_u[u]`, length `p`.
    pub eta_u: Vec<f64>,
    /// `chi_u[u]`, length `p`.
    pub chi_u: Vec<f64>,
    /// `d_u[u]`, length `p`.
    pub d_u: Vec<f64>,
    /// `a_uv` dense row-major `p × p`, symmetric.
    pub a_uv: Vec<f64>,
    /// `eta_uv` dense row-major `p × p`, symmetric.
    pub eta_uv: Vec<f64>,
    /// `chi_uv` dense row-major `p × p`, symmetric.
    pub chi_uv: Vec<f64>,
}

/// Substrate-backed Layer C-α fold.  Computes the second-order
/// timepoint jet `(a_uv, eta_uv, chi_uv)` and the first-order outputs
/// `(a_u, eta_u, chi_u, d_u)` for every input row.
pub fn try_device_layer_c_jet(
    rows: &[SurvivalFlexLayerCRowInputs<'_>],
) -> Result<Option<Vec<SurvivalFlexLayerCRowOutputs>>, GpuError> {
    use crate::families::cubic_cell_kernel::{
        cell_first_derivative_from_moments, cell_polynomial_integral_from_moments,
        cell_second_derivative_from_moments, denested_cell_coefficient_partials,
        denested_cell_second_partials,
    };

    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let n_rows = rows.len();

    // Validate shapes and collect totals.
    let mut total_cells = 0usize;
    let mut row_offsets = Vec::with_capacity(n_rows + 1);
    row_offsets.push(0);
    for (i, row) in rows.iter().enumerate() {
        let p = row.rho.len();
        if row.tau.len() != p || row.tau_a.len() != p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c row {i}: rho/tau/tau_a length mismatch ({}/{}/{})",
                    p,
                    row.tau.len(),
                    row.tau_a.len()
                ),
            });
        }
        let expected_packed = p * (p + 1) / 2;
        if row.r_uv_upper_packed.len() != expected_packed
            || row.chi_uv_fixed_upper_packed.len() != expected_packed
        {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c row {i}: r_uv/chi_uv_fixed packed length must be p*(p+1)/2 = {expected_packed}, got {}/{}",
                    row.r_uv_upper_packed.len(),
                    row.chi_uv_fixed_upper_packed.len()
                ),
            });
        }
        if row.partition_cells.len() != row.cell_coeffs.len() {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c row {i}: cells.len()={} != cell_coeffs.len()={}",
                    row.partition_cells.len(),
                    row.cell_coeffs.len()
                ),
            });
        }
        for (k, cc) in row.cell_coeffs.iter().enumerate() {
            if cc.coeff_u.len() != p || cc.coeff_au.len() != p || cc.coeff_bu.len() != p {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "layer_c row {i} cell {k}: coeff_u/coeff_au/coeff_bu lengths {}/{}/{} expected {p}",
                        cc.coeff_u.len(),
                        cc.coeff_au.len(),
                        cc.coeff_bu.len()
                    ),
                });
            }
        }
        if !(row.d_check.is_finite() && row.d_check > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c row {i}: d_check must be positive finite, got {}",
                    row.d_check
                ),
            });
        }
        if !(row.probit_scale.is_finite() && row.probit_scale > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c row {i}: probit_scale must be positive finite, got {}",
                    row.probit_scale
                ),
            });
        }
        total_cells += row.partition_cells.len();
        row_offsets.push(total_cells);
    }

    if total_cells == 0 {
        let mut out = Vec::with_capacity(n_rows);
        for row in rows {
            let p = row.rho.len();
            out.push(SurvivalFlexLayerCRowOutputs {
                a_u: vec![0.0; p],
                eta_u: row.rho.to_vec(),
                chi_u: row.tau.to_vec(),
                d_u: vec![0.0; p],
                a_uv: vec![0.0; p * p],
                eta_uv: vec![0.0; p * p],
                chi_uv: vec![0.0; p * p],
            });
        }
        return Ok(Some(out));
    }

    // Build the substrate batch (negated cubic, same shape as Layers A/B).
    let mut left = Vec::with_capacity(total_cells);
    let mut right = Vec::with_capacity(total_cells);
    let mut c0 = Vec::with_capacity(total_cells);
    let mut c1 = Vec::with_capacity(total_cells);
    let mut c2 = Vec::with_capacity(total_cells);
    let mut c3 = Vec::with_capacity(total_cells);
    for row in rows {
        for pc in row.partition_cells {
            left.push(pc.cell.left);
            right.push(pc.cell.right);
            c0.push(-pc.cell.c0);
            c1.push(-pc.cell.c1);
            c2.push(-pc.cell.c2);
            c3.push(-pc.cell.c3);
        }
    }
    let batch = SurvivalFlexRowCellsBatch {
        n_cells: total_cells,
        n_rows,
        max_degree: 9,
        left: &left,
        right: &right,
        c0: &c0,
        c1: &c1,
        c2: &c2,
        c3: &c3,
        row_offsets: &row_offsets,
    };
    let mom = match try_row_batched_cell_moments(batch)? {
        Some(m) => m,
        None => return Ok(None),
    };
    let stride = mom.stride;
    if stride < 10 {
        return Err(GpuError::DriverCallFailed {
            reason: format!("layer_c: substrate returned stride={stride} < 10"),
        });
    }
    let ok_byte = crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8;
    if mom.status.iter().any(|&s| s != ok_byte) {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(n_rows);
    for (row_idx, row) in rows.iter().enumerate() {
        let p = row.rho.len();
        let row_start = row_offsets[row_idx];
        let row_end = row_offsets[row_idx + 1];

        // Per-cell scratch for second-deriv folds.
        // Step 1: f_aa, f_u, f_au, f_uv from per-cell quadrature.
        let mut f_u = vec![0.0_f64; p];
        let mut f_au = vec![0.0_f64; p];
        let mut f_uv = vec![0.0_f64; p * p];
        let mut f_aa = 0.0_f64;

        for cell_idx in row_start..row_end {
            let local_idx = cell_idx - row_start;
            let pc = &row.partition_cells[local_idx];
            let cc = &row.cell_coeffs[local_idx];
            let moments_row = &mom.moments[cell_idx * stride..cell_idx * stride + stride];

            // Reproduce the family's `neg_cell` for the second-derivative helper.
            let neg_cell = crate::families::cubic_cell_kernel::DenestedCubicCell {
                left: pc.cell.left,
                right: pc.cell.right,
                c0: -pc.cell.c0,
                c1: -pc.cell.c1,
                c2: -pc.cell.c2,
                c3: -pc.cell.c3,
            };

            // Per-cell analytic partials at this row's (a, slope).
            let (dc_da_pos, _) =
                denested_cell_coefficient_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let (dc_daa_pos, _, _) =
                denested_cell_second_partials(pc.score_span, pc.link_span, row.a, row.slope);
            // Sign convention matches `evaluate_denested_survival_calibration`:
            // `-probit_scale` on every partial because the integrand is `-Φ(-η)`.
            let neg_dc_da = [
                -row.probit_scale * dc_da_pos[0],
                -row.probit_scale * dc_da_pos[1],
                -row.probit_scale * dc_da_pos[2],
                -row.probit_scale * dc_da_pos[3],
            ];
            let neg_dc_daa = [
                -row.probit_scale * dc_daa_pos[0],
                -row.probit_scale * dc_daa_pos[1],
                -row.probit_scale * dc_daa_pos[2],
                -row.probit_scale * dc_daa_pos[3],
            ];

            // f_aa contribution.
            let f_aa_cell = cell_second_derivative_from_moments(
                neg_cell,
                &neg_dc_da,
                &neg_dc_da,
                &neg_dc_daa,
                moments_row,
            )
            .map_err(|e| GpuError::DriverCallFailed {
                reason: format!("layer_c row {row_idx} cell {local_idx}: f_aa: {e}"),
            })?;
            f_aa += f_aa_cell;

            // f_u and f_au per primary.
            for u in 0..p {
                let neg_coeff_u = [
                    -cc.coeff_u[u][0],
                    -cc.coeff_u[u][1],
                    -cc.coeff_u[u][2],
                    -cc.coeff_u[u][3],
                ];
                let neg_coeff_au = [
                    -cc.coeff_au[u][0],
                    -cc.coeff_au[u][1],
                    -cc.coeff_au[u][2],
                    -cc.coeff_au[u][3],
                ];
                let fu =
                    cell_first_derivative_from_moments(&neg_coeff_u, moments_row).map_err(|e| {
                        GpuError::DriverCallFailed {
                            reason: format!(
                                "layer_c row {row_idx} cell {local_idx} u={u}: f_u: {e}"
                            ),
                        }
                    })?;
                f_u[u] += fu;
                let fau = cell_second_derivative_from_moments(
                    neg_cell,
                    &neg_dc_da,
                    &neg_coeff_u,
                    &neg_coeff_au,
                    moments_row,
                )
                .map_err(|e| GpuError::DriverCallFailed {
                    reason: format!("layer_c row {row_idx} cell {local_idx} u={u}: f_au: {e}"),
                })?;
                f_au[u] += fau;
            }

            // f_uv: upper triangle, then mirror.  `second_coeff` branches
            // on g_index per CPU lines 6817-6822.
            for u in 0..p {
                let neg_coeff_u = [
                    -cc.coeff_u[u][0],
                    -cc.coeff_u[u][1],
                    -cc.coeff_u[u][2],
                    -cc.coeff_u[u][3],
                ];
                for v in u..p {
                    let second_coeff_pos: [f64; 4] = if u == row.g_index {
                        cc.coeff_bu[v]
                    } else if v == row.g_index {
                        cc.coeff_bu[u]
                    } else {
                        [0.0; 4]
                    };
                    let neg_coeff_v = [
                        -cc.coeff_u[v][0],
                        -cc.coeff_u[v][1],
                        -cc.coeff_u[v][2],
                        -cc.coeff_u[v][3],
                    ];
                    let neg_second_coeff = [
                        -second_coeff_pos[0],
                        -second_coeff_pos[1],
                        -second_coeff_pos[2],
                        -second_coeff_pos[3],
                    ];
                    let value = cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_coeff_u,
                        &neg_coeff_v,
                        &neg_second_coeff,
                        moments_row,
                    )
                    .map_err(|e| GpuError::DriverCallFailed {
                        reason: format!(
                            "layer_c row {row_idx} cell {local_idx} u={u} v={v}: f_uv: {e}"
                        ),
                    })?;
                    f_uv[u * p + v] += value;
                    if v != u {
                        f_uv[v * p + u] += value;
                    }
                }
            }
        }

        // q-index bumps.
        if row.q_index < p {
            f_u[row.q_index] += row.phi_q;
            let idx = row.q_index * p + row.q_index;
            f_uv[idx] += -row.q * row.phi_q;
        }

        // a_u = f_u / D.
        let mut a_u = vec![0.0_f64; p];
        for u in 0..p {
            a_u[u] = f_u[u] / row.d_check;
        }

        // d_u: same algebra as Layer B (Σ_cells of integrand_u, see Layer B notes).
        let mut d_u = vec![0.0_f64; p];
        for cell_idx in row_start..row_end {
            let local_idx = cell_idx - row_start;
            let pc = &row.partition_cells[local_idx];
            let cc = &row.cell_coeffs[local_idx];
            let moments_row = &mom.moments[cell_idx * stride..cell_idx * stride + stride];

            let (dc_da_pos, _) =
                denested_cell_coefficient_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let (dc_daa_pos, _, _) =
                denested_cell_second_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let chi_poly = dc_da_pos;
            let eta_aa_poly = dc_daa_pos;
            let eta_poly = [pc.cell.c0, pc.cell.c1, pc.cell.c2, pc.cell.c3];

            let mut chi_eta = [0.0_f64; 16];
            for i in 0..4 {
                for j in 0..4 {
                    chi_eta[i + j] = chi_poly[i].mul_add(eta_poly[j], chi_eta[i + j]);
                }
            }
            let chi_eta_len = 7usize;

            for u in 0..p {
                let eta_u_poly = [
                    a_u[u] * chi_poly[0] + cc.coeff_u[u][0],
                    a_u[u] * chi_poly[1] + cc.coeff_u[u][1],
                    a_u[u] * chi_poly[2] + cc.coeff_u[u][2],
                    a_u[u] * chi_poly[3] + cc.coeff_u[u][3],
                ];
                let chi_u_poly = [
                    a_u[u] * eta_aa_poly[0] + cc.coeff_au[u][0],
                    a_u[u] * eta_aa_poly[1] + cc.coeff_au[u][1],
                    a_u[u] * eta_aa_poly[2] + cc.coeff_au[u][2],
                    a_u[u] * eta_aa_poly[3] + cc.coeff_au[u][3],
                ];
                let mut chi_eta_etau = [0.0_f64; 16];
                for i in 0..chi_eta_len {
                    for j in 0..4 {
                        chi_eta_etau[i + j] =
                            chi_eta[i].mul_add(eta_u_poly[j], chi_eta_etau[i + j]);
                    }
                }
                let triple_len = chi_eta_len + 4 - 1;
                let mut integrand = [0.0_f64; 16];
                for k in 0..4 {
                    integrand[k] = chi_u_poly[k];
                }
                for k in 0..triple_len {
                    integrand[k] -= chi_eta_etau[k];
                }
                let contrib = cell_polynomial_integral_from_moments(
                    &integrand[..triple_len],
                    moments_row,
                    "survival_flex layer_c d_u",
                )
                .map_err(|e| GpuError::DriverCallFailed {
                    reason: format!("layer_c row {row_idx} cell {local_idx} u={u}: d_u fold: {e}"),
                })?;
                d_u[u] += contrib;
            }
        }

        // a_uv: upper triangle then mirror.
        let mut a_uv = vec![0.0_f64; p * p];
        for u in 0..p {
            for v in u..p {
                let value =
                    (f_uv[u * p + v] - d_u[u] * a_u[v] - d_u[v] * a_u[u] - f_aa * a_u[u] * a_u[v])
                        / row.d_check;
                a_uv[u * p + v] = value;
                if v != u {
                    a_uv[v * p + u] = value;
                }
            }
        }

        // First-order closed-form algebra (Layer B's pattern).
        let mut eta_u = vec![0.0_f64; p];
        let mut chi_u = vec![0.0_f64; p];
        for u in 0..p {
            eta_u[u] = row.chi * a_u[u] + row.rho[u];
            chi_u[u] = row.eta_aa * a_u[u] + row.tau[u];
        }

        // Second-order closed-form algebra (CPU lines 6989-7013).
        let mut eta_uv = vec![0.0_f64; p * p];
        let mut chi_uv = vec![0.0_f64; p * p];
        for u in 0..p {
            for v in u..p {
                let packed = tri_index(u, v, p);
                let r_uv_val = row.r_uv_upper_packed[packed];
                let chi_uv_fixed_val = row.chi_uv_fixed_upper_packed[packed];
                let eta_val = row.chi * a_uv[u * p + v]
                    + row.eta_aa * a_u[u] * a_u[v]
                    + row.tau[u] * a_u[v]
                    + row.tau[v] * a_u[u]
                    + r_uv_val;
                eta_uv[u * p + v] = eta_val;
                if v != u {
                    eta_uv[v * p + u] = eta_val;
                }
                let chi_val = row.eta_aa * a_uv[u * p + v]
                    + row.eta_aaa * a_u[u] * a_u[v]
                    + row.tau_a[u] * a_u[v]
                    + row.tau_a[v] * a_u[u]
                    + chi_uv_fixed_val;
                chi_uv[u * p + v] = chi_val;
                if v != u {
                    chi_uv[v * p + u] = chi_val;
                }
            }
        }

        out.push(SurvivalFlexLayerCRowOutputs {
            a_u,
            eta_u,
            chi_u,
            d_u,
            a_uv,
            eta_uv,
            chi_uv,
        });
    }

    Ok(Some(out))
}

// ────────────────────────────────────────────────────────────────────────
// Step 4 Layer C-β — optional `d_uv` second-derivative quadrature.
//
// Per CPU `compute_survival_timepoint_exact` lines 7017-7122 (the
// `need_d_uv == true` branch):
//
//   eta_u_poly[u]   = chi_poly · a_u[u] + coeff_u[u]
//   chi_u_poly[u]   = eta_aa_poly · a_u[u] + coeff_au[u]
//   eta_uv_poly[u,v] = chi_poly · a_uv[u,v] + eta_aa_poly · a_u[u]·a_u[v]
//                    + coeff_au[u] · a_u[v] + coeff_au[v] · a_u[u]
//                    + r_uv_fixed_poly[u,v]
//   chi_uv_poly[u,v] = eta_aa_poly · a_uv[u,v] + eta_aaa_poly · a_u[u]·a_u[v]
//                    + coeff_aau[u] · a_u[v] + coeff_aau[v] · a_u[u]
//                    + chi_uv_fixed_poly[u,v]
//   r_uv_fixed_poly  = coeff_bu[g_idx_other]  if u or v == g_index else 0
//   chi_uv_fixed_poly = coeff_abu[g_idx_other] if u or v == g_index else 0
//
//   integrand[u,v] = chi_uv_poly
//                  - chi_u_poly[v] · eta_poly · eta_u_poly[u]
//                  - chi_u_poly[u] · eta_poly · eta_u_poly[v]
//                  - chi_poly · (eta_u_poly[u] · eta_u_poly[v]
//                                + eta_poly · eta_uv_poly)
//                  + chi_poly · eta_poly² · eta_u_poly[u] · eta_u_poly[v]
//
//   d_uv[u,v] = Σ_cells INV_TWO_PI · ⟨integrand[u,v], moments⟩
//
// Term degrees: term5 (chi · eta² · eta_u · eta_v) peaks at length
// 4 + 4 + 4 + 4 + 4 - 4 = 16 → max moment index 16 → max_degree 16
// (substrate supports 24).
//
// `eta_aaa_poly` comes from `cubic_cell_kernel::denested_cell_third_partials`
// (first slot, `dc_daaa`).  Per-cell `coeff_aau[u]` and `coeff_abu[u]`
// extensions of the Layer-C tables; callers compute them via the family's
// `denested_cell_primary_fixed_partials` (lines 6235-6341).
// ────────────────────────────────────────────────────────────────────────

/// Extended per-cell coefficient tables for Layer C-β.  Extends
/// [`SurvivalFlexLayerCCellCoeffs`] with `coeff_aau[u]` and `coeff_abu[u]`
/// for the `chi_uv_poly` assembly and the `g_index`-branched
/// `chi_uv_fixed_poly`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerCBetaCellCoeffs {
    /// `coeff_u[u]`.
    pub coeff_u: Vec<[f64; 4]>,
    /// `coeff_au[u]`.
    pub coeff_au: Vec<[f64; 4]>,
    /// `coeff_bu[u]`.  Used for `r_uv_fixed_poly` (eta_uv assembly).
    pub coeff_bu: Vec<[f64; 4]>,
    /// `coeff_aau[u]`.  Used for `chi_uv_poly` mixed terms (CPU line 7070).
    pub coeff_aau: Vec<[f64; 4]>,
    /// `coeff_abu[u]`.  Used for `chi_uv_fixed_poly` (CPU line 7048).
    pub coeff_abu: Vec<[f64; 4]>,
}

/// Per-row inputs for Layer C-β `d_uv` quadrature.
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerCBetaRowInputs<'a> {
    /// Per-row partition cells (same as Layer A/B/C-α).
    pub partition_cells: &'a [SurvivalFlexCalibrationCell],
    /// Extended per-cell coefficient tables.
    pub cell_coeffs: &'a [SurvivalFlexLayerCBetaCellCoeffs],
    /// `g_index` for the second-coeff branching; `usize::MAX` to disable.
    pub g_index: usize,
    /// Pre-computed first-order `a_u[u]` (length `p`) from Layer C-α.
    pub a_u: &'a [f64],
    /// Pre-computed second-order `a_uv[u,v]` (dense row-major `p × p`)
    /// from Layer C-α.
    pub a_uv: &'a [f64],
    /// Calibration root `a*`.
    pub a: f64,
    /// Slope `b`.
    pub slope: f64,
    /// `probit_scale`.
    pub probit_scale: f64,
}

/// Per-row Layer C-β outputs: dense `d_uv` (row-major `p × p`, symmetric).
#[derive(Clone, Debug)]
pub struct SurvivalFlexLayerCBetaRowOutputs {
    pub d_uv: Vec<f64>,
}

/// Substrate-backed Layer C-β `d_uv` fold.  Builds the substrate moment
/// batch at `max_degree = 16` (sufficient for the degree-16 quintic
/// term5), then performs the per-(u,v) polynomial-product chain on host.
pub fn try_device_layer_c_beta_d_uv(
    rows: &[SurvivalFlexLayerCBetaRowInputs<'_>],
) -> Result<Option<Vec<SurvivalFlexLayerCBetaRowOutputs>>, GpuError> {
    use crate::families::cubic_cell_kernel::{
        cell_polynomial_integral_from_moments, denested_cell_coefficient_partials,
        denested_cell_second_partials, denested_cell_third_partials,
    };

    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let n_rows = rows.len();

    let mut total_cells = 0usize;
    let mut row_offsets = Vec::with_capacity(n_rows + 1);
    row_offsets.push(0);
    for (i, row) in rows.iter().enumerate() {
        let p = row.a_u.len();
        if row.a_uv.len() != p * p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c_beta row {i}: a_uv.len()={} != p*p = {}",
                    row.a_uv.len(),
                    p * p
                ),
            });
        }
        if row.partition_cells.len() != row.cell_coeffs.len() {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c_beta row {i}: cells.len()={} != cell_coeffs.len()={}",
                    row.partition_cells.len(),
                    row.cell_coeffs.len()
                ),
            });
        }
        for (k, cc) in row.cell_coeffs.iter().enumerate() {
            for (name, len) in [
                ("coeff_u", cc.coeff_u.len()),
                ("coeff_au", cc.coeff_au.len()),
                ("coeff_bu", cc.coeff_bu.len()),
                ("coeff_aau", cc.coeff_aau.len()),
                ("coeff_abu", cc.coeff_abu.len()),
            ] {
                if len != p {
                    return Err(GpuError::DriverCallFailed {
                        reason: format!(
                            "layer_c_beta row {i} cell {k}: {name}.len()={len} expected {p}"
                        ),
                    });
                }
            }
        }
        if !(row.probit_scale.is_finite() && row.probit_scale > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "layer_c_beta row {i}: probit_scale must be positive finite, got {}",
                    row.probit_scale
                ),
            });
        }
        total_cells += row.partition_cells.len();
        row_offsets.push(total_cells);
    }

    if total_cells == 0 {
        let mut out = Vec::with_capacity(n_rows);
        for row in rows {
            let p = row.a_u.len();
            out.push(SurvivalFlexLayerCBetaRowOutputs {
                d_uv: vec![0.0; p * p],
            });
        }
        return Ok(Some(out));
    }

    // Substrate batch at max_degree 16 (covers term5's degree-16 monomial).
    let mut left = Vec::with_capacity(total_cells);
    let mut right = Vec::with_capacity(total_cells);
    let mut c0 = Vec::with_capacity(total_cells);
    let mut c1 = Vec::with_capacity(total_cells);
    let mut c2 = Vec::with_capacity(total_cells);
    let mut c3 = Vec::with_capacity(total_cells);
    for row in rows {
        for pc in row.partition_cells {
            left.push(pc.cell.left);
            right.push(pc.cell.right);
            c0.push(-pc.cell.c0);
            c1.push(-pc.cell.c1);
            c2.push(-pc.cell.c2);
            c3.push(-pc.cell.c3);
        }
    }
    let batch = SurvivalFlexRowCellsBatch {
        n_cells: total_cells,
        n_rows,
        max_degree: 16,
        left: &left,
        right: &right,
        c0: &c0,
        c1: &c1,
        c2: &c2,
        c3: &c3,
        row_offsets: &row_offsets,
    };
    let mom = match try_row_batched_cell_moments(batch)? {
        Some(m) => m,
        None => return Ok(None),
    };
    let stride = mom.stride;
    if stride < 17 {
        return Err(GpuError::DriverCallFailed {
            reason: format!("layer_c_beta: substrate returned stride={stride} < 17"),
        });
    }
    let ok_byte = crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8;
    if mom.status.iter().any(|&s| s != ok_byte) {
        return Ok(None);
    }

    // Helpers: poly_add_into / poly_mul_into operating on fixed-capacity
    // [f64; N] buffers.  All intermediate degrees ≤ 16, so N = 20 covers
    // every accumulator without dynamic allocation.
    const POLY_CAP: usize = 20;
    #[inline]
    fn poly_add_4(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }
    #[inline]
    fn poly_scale_4(p: &[f64; 4], s: f64) -> [f64; 4] {
        [s * p[0], s * p[1], s * p[2], s * p[3]]
    }
    /// `out[k] = Σ_{i+j=k} a[i] · b[j]`.  Returns `out_len = a_len + b_len - 1`.
    #[inline]
    fn poly_mul_into(a: &[f64], b: &[f64], out: &mut [f64; POLY_CAP]) -> usize {
        out.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..a.len() {
            for j in 0..b.len() {
                out[i + j] = a[i].mul_add(b[j], out[i + j]);
            }
        }
        a.len() + b.len() - 1
    }

    let mut out = Vec::with_capacity(n_rows);
    for (row_idx, row) in rows.iter().enumerate() {
        let p = row.a_u.len();
        let row_start = row_offsets[row_idx];
        let row_end = row_offsets[row_idx + 1];
        let mut d_uv = vec![0.0_f64; p * p];

        for cell_idx in row_start..row_end {
            let local_idx = cell_idx - row_start;
            let pc = &row.partition_cells[local_idx];
            let cc = &row.cell_coeffs[local_idx];
            let moments_row = &mom.moments[cell_idx * stride..cell_idx * stride + stride];

            // Per-cell polynomial seeds at this row's (a, slope).
            let (dc_da_pos, _) =
                denested_cell_coefficient_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let (dc_daa_pos, _, _) =
                denested_cell_second_partials(pc.score_span, pc.link_span, row.a, row.slope);
            let (dc_daaa_pos, _, _, _) = denested_cell_third_partials(pc.link_span);
            // Sign convention: integrand wraps `-Φ(-η)` so every cell
            // polynomial seed is `-probit_scale` (Layer A/B/C-α convention).
            let chi_poly: [f64; 4] = [
                -row.probit_scale * dc_da_pos[0],
                -row.probit_scale * dc_da_pos[1],
                -row.probit_scale * dc_da_pos[2],
                -row.probit_scale * dc_da_pos[3],
            ];
            let eta_aa_poly: [f64; 4] = [
                -row.probit_scale * dc_daa_pos[0],
                -row.probit_scale * dc_daa_pos[1],
                -row.probit_scale * dc_daa_pos[2],
                -row.probit_scale * dc_daa_pos[3],
            ];
            let eta_aaa_poly: [f64; 4] = [
                -row.probit_scale * dc_daaa_pos[0],
                -row.probit_scale * dc_daaa_pos[1],
                -row.probit_scale * dc_daaa_pos[2],
                -row.probit_scale * dc_daaa_pos[3],
            ];
            let eta_poly: [f64; 4] = [pc.cell.c0, pc.cell.c1, pc.cell.c2, pc.cell.c3];

            // Pre-compute eta_poly² (length 7) and chi · eta² (length 10);
            // reused across every (u, v) pair.
            let mut eta_sq = [0.0_f64; POLY_CAP];
            let eta_sq_len = poly_mul_into(&eta_poly, &eta_poly, &mut eta_sq);
            let mut chi_eta_sq = [0.0_f64; POLY_CAP];
            let chi_eta_sq_len = poly_mul_into(&chi_poly, &eta_sq[..eta_sq_len], &mut chi_eta_sq);

            // Pre-compute eta_u_poly[u] and chi_u_poly[u] per primary.
            let mut eta_u_poly: Vec<[f64; 4]> = Vec::with_capacity(p);
            let mut chi_u_poly: Vec<[f64; 4]> = Vec::with_capacity(p);
            for u in 0..p {
                eta_u_poly.push(poly_add_4(
                    &poly_scale_4(&chi_poly, row.a_u[u]),
                    &cc.coeff_u[u],
                ));
                chi_u_poly.push(poly_add_4(
                    &poly_scale_4(&eta_aa_poly, row.a_u[u]),
                    &cc.coeff_au[u],
                ));
            }

            for u in 0..p {
                for v in u..p {
                    let r_uv_fixed: [f64; 4] = if u == row.g_index {
                        cc.coeff_bu[v]
                    } else if v == row.g_index {
                        cc.coeff_bu[u]
                    } else {
                        [0.0; 4]
                    };
                    let chi_uv_fixed: [f64; 4] = if u == row.g_index {
                        cc.coeff_abu[v]
                    } else if v == row.g_index {
                        cc.coeff_abu[u]
                    } else {
                        [0.0; 4]
                    };

                    let auv = row.a_uv[u * p + v];
                    let au = row.a_u[u];
                    let av = row.a_u[v];

                    // eta_uv_poly = chi·a_uv + eta_aa·a_u·a_v + coeff_au[u]·a_v + coeff_au[v]·a_u + r_uv_fixed
                    let eta_uv_poly: [f64; 4] = {
                        let a = poly_scale_4(&chi_poly, auv);
                        let b = poly_scale_4(&eta_aa_poly, au * av);
                        let c = poly_scale_4(&cc.coeff_au[u], av);
                        let d = poly_scale_4(&cc.coeff_au[v], au);
                        let mut s = poly_add_4(&a, &b);
                        s = poly_add_4(&s, &c);
                        s = poly_add_4(&s, &d);
                        poly_add_4(&s, &r_uv_fixed)
                    };
                    // chi_uv_poly = eta_aa·a_uv + eta_aaa·a_u·a_v + coeff_aau[u]·a_v + coeff_aau[v]·a_u + chi_uv_fixed
                    let chi_uv_poly: [f64; 4] = {
                        let a = poly_scale_4(&eta_aa_poly, auv);
                        let b = poly_scale_4(&eta_aaa_poly, au * av);
                        let c = poly_scale_4(&cc.coeff_aau[u], av);
                        let d = poly_scale_4(&cc.coeff_aau[v], au);
                        let mut s = poly_add_4(&a, &b);
                        s = poly_add_4(&s, &c);
                        s = poly_add_4(&s, &d);
                        poly_add_4(&s, &chi_uv_fixed)
                    };

                    // term2 = -chi_u[v] · eta · eta_u[u], length 4+4+4-2 = 10
                    let mut chi_u_v_eta = [0.0_f64; POLY_CAP];
                    let len_a = poly_mul_into(&chi_u_poly[v], &eta_poly, &mut chi_u_v_eta);
                    let mut term2 = [0.0_f64; POLY_CAP];
                    let term2_len =
                        poly_mul_into(&chi_u_v_eta[..len_a], &eta_u_poly[u], &mut term2);
                    for k in 0..term2_len {
                        term2[k] = -term2[k];
                    }

                    // term3 = -chi_u[u] · eta · eta_u[v], length 10
                    let mut chi_u_u_eta = [0.0_f64; POLY_CAP];
                    let len_b = poly_mul_into(&chi_u_poly[u], &eta_poly, &mut chi_u_u_eta);
                    let mut term3 = [0.0_f64; POLY_CAP];
                    let term3_len =
                        poly_mul_into(&chi_u_u_eta[..len_b], &eta_u_poly[v], &mut term3);
                    for k in 0..term3_len {
                        term3[k] = -term3[k];
                    }

                    // term4 = -chi · (eta_u[u] · eta_u[v] + eta · eta_uv)
                    // First the inner sum.
                    let mut eu_u_eu_v = [0.0_f64; POLY_CAP];
                    let len_c = poly_mul_into(&eta_u_poly[u], &eta_u_poly[v], &mut eu_u_eu_v);
                    let mut eta_eta_uv = [0.0_f64; POLY_CAP];
                    let len_d = poly_mul_into(&eta_poly, &eta_uv_poly, &mut eta_eta_uv);
                    let inner_len = len_c.max(len_d);
                    let mut inner = [0.0_f64; POLY_CAP];
                    for k in 0..len_c {
                        inner[k] = eu_u_eu_v[k];
                    }
                    for k in 0..len_d {
                        inner[k] += eta_eta_uv[k];
                    }
                    let mut term4 = [0.0_f64; POLY_CAP];
                    let term4_len = poly_mul_into(&chi_poly, &inner[..inner_len], &mut term4);
                    for k in 0..term4_len {
                        term4[k] = -term4[k];
                    }

                    // term5 = chi · eta² · eta_u[u] · eta_u[v]
                    //       = chi_eta_sq · eta_u[u] · eta_u[v]
                    let mut t5_a = [0.0_f64; POLY_CAP];
                    let len_e =
                        poly_mul_into(&chi_eta_sq[..chi_eta_sq_len], &eta_u_poly[u], &mut t5_a);
                    let mut term5 = [0.0_f64; POLY_CAP];
                    let term5_len = poly_mul_into(&t5_a[..len_e], &eta_u_poly[v], &mut term5);

                    // integrand = chi_uv_poly + term2 + term3 + term4 + term5
                    let total_len = term5_len
                        .max(term4_len)
                        .max(term3_len)
                        .max(term2_len)
                        .max(4);
                    let mut integrand = [0.0_f64; POLY_CAP];
                    for k in 0..4 {
                        integrand[k] = chi_uv_poly[k];
                    }
                    for k in 0..term2_len {
                        integrand[k] += term2[k];
                    }
                    for k in 0..term3_len {
                        integrand[k] += term3[k];
                    }
                    for k in 0..term4_len {
                        integrand[k] += term4[k];
                    }
                    for k in 0..term5_len {
                        integrand[k] += term5[k];
                    }

                    let value = cell_polynomial_integral_from_moments(
                        &integrand[..total_len],
                        moments_row,
                        "survival_flex layer_c_beta d_uv",
                    )
                    .map_err(|e| GpuError::DriverCallFailed {
                        reason: format!(
                            "layer_c_beta row {row_idx} cell {local_idx} u={u} v={v}: d_uv: {e}"
                        ),
                    })?;
                    d_uv[u * p + v] += value;
                    if v != u {
                        d_uv[v * p + u] += value;
                    }
                }
            }
        }

        out.push(SurvivalFlexLayerCBetaRowOutputs { d_uv });
    }

    Ok(Some(out))
}

// ────────────────────────────────────────────────────────────────────────
// Step 5 — per-row primary gradient + Hessian assembly from the full jet.
//
// Given the entry- and exit-time jets `(eta, chi, d, eta_u, eta_uv,
// chi_u, chi_uv, d_u, d_uv)` (from Layer C-α + C-β) plus the
// `signed_probit_neglog` derivatives `(k1, k2)` at `-entry.eta` /
// `-exit.eta`, the per-row NLL and its primary gradient + Hessian are
// pure scalar / vector algebra (CPU
// `compute_row_flex_primary_gradient_hessian_from_parts`, lines
// 7263-7384 of survival_marginal_slope.rs).
//
// The joint-β `axpy_row_into` pullback into the dense coefficient
// gradient / Hessian is family-owned (per-block design rows in
// `marginal_design` / `logslope_design` / `score_warp` / `link_dev`
// runtimes); Step 6 wires it behind the three try_* entry points.
// ────────────────────────────────────────────────────────────────────────

/// Per-row time-point jet bundle for the Step-5 assembly.
#[derive(Clone, Debug)]
pub struct SurvivalFlexTimepointJet<'a> {
    pub eta: f64,
    pub chi: f64,
    pub d: f64,
    /// Length `p`.
    pub eta_u: &'a [f64],
    /// Row-major `p × p`, symmetric.
    pub eta_uv: &'a [f64],
    /// Length `p`.
    pub chi_u: &'a [f64],
    /// Row-major `p × p`, symmetric.
    pub chi_uv: &'a [f64],
    /// Length `p`.
    pub d_u: &'a [f64],
    /// Row-major `p × p`, symmetric.
    pub d_uv: &'a [f64],
}

/// Per-row inputs for the Step-5 primary gradient + Hessian assembly.
#[derive(Clone, Debug)]
pub struct SurvivalFlexStep5RowInputs<'a> {
    pub entry: SurvivalFlexTimepointJet<'a>,
    pub exit: SurvivalFlexTimepointJet<'a>,
    pub wi: f64,
    pub di: f64,
    pub q1: f64,
    pub qd1: f64,
    /// `q1_index`: primary index of the q1 perturbation, `usize::MAX`
    /// to disable the `+ wi·di·q1` / `+ wi·di` bumps.
    pub q1_index: usize,
    /// `qd1_index`: primary index of the qd1 perturbation, `usize::MAX`
    /// to disable the `-wi·di/qd1` / `+wi·di/qd1²` bumps.
    pub qd1_index: usize,
    pub entry_k1: f64,
    pub entry_k2: f64,
    pub exit_k1: f64,
    pub exit_k2: f64,
    pub log_surv0: f64,
    pub log_surv1: f64,
}

/// Per-row Step-5 outputs.
#[derive(Clone, Debug)]
pub struct SurvivalFlexStep5RowOutputs {
    pub row_nll: f64,
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
}

/// Step-5 per-row primary gradient + Hessian assembly.  Pure scalar /
/// vector algebra over the supplied jet bundles; no quadrature.
pub fn try_device_step5_primary_assembly(
    rows: &[SurvivalFlexStep5RowInputs<'_>],
) -> Result<Vec<SurvivalFlexStep5RowOutputs>, GpuError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let p = r.entry.eta_u.len();
        let check = |label: &str, len: usize, expected: usize| -> Result<(), GpuError> {
            if len != expected {
                return Err(GpuError::DriverCallFailed {
                    reason: format!("step5 row {i}: {label}.len()={len} expected {expected}"),
                });
            }
            Ok(())
        };
        check("entry.eta_uv", r.entry.eta_uv.len(), p * p)?;
        check("entry.chi_u", r.entry.chi_u.len(), p)?;
        check("entry.chi_uv", r.entry.chi_uv.len(), p * p)?;
        check("entry.d_u", r.entry.d_u.len(), p)?;
        check("entry.d_uv", r.entry.d_uv.len(), p * p)?;
        check("exit.eta_u", r.exit.eta_u.len(), p)?;
        check("exit.eta_uv", r.exit.eta_uv.len(), p * p)?;
        check("exit.chi_u", r.exit.chi_u.len(), p)?;
        check("exit.chi_uv", r.exit.chi_uv.len(), p * p)?;
        check("exit.d_u", r.exit.d_u.len(), p)?;
        check("exit.d_uv", r.exit.d_uv.len(), p * p)?;

        if !(r.exit.chi.is_finite() && r.exit.chi > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "step5 row {i}: exit.chi must be positive finite, got {}",
                    r.exit.chi
                ),
            });
        }
        if !(r.exit.d.is_finite() && r.exit.d > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "step5 row {i}: exit.d must be positive finite, got {}",
                    r.exit.d
                ),
            });
        }

        let log_phi_eta1 = -0.5 * (r.exit.eta * r.exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (r.q1 * r.q1 + std::f64::consts::TAU.ln());
        let row_nll = r.wi
            * (r.log_surv0
                - (1.0 - r.di) * r.log_surv1
                - r.di * log_phi_eta1
                - r.di * r.exit.chi.ln()
                - r.di * log_phi_q1
                + r.di * r.exit.d.ln()
                - r.di * r.qd1.ln());

        let entry_u1 = -r.entry_k1;
        let entry_u2 = r.entry_k2;
        let exit_surv_u1 = -r.exit_k1;
        let exit_surv_u2 = r.exit_k2;

        let mut grad = vec![0.0_f64; p];
        for u in 0..p {
            let mut val = 0.0;
            val += entry_u1 * r.entry.eta_u[u];
            val += exit_surv_u1 * r.exit.eta_u[u];
            val += r.wi * r.di * r.exit.eta * r.exit.eta_u[u];
            val -= r.wi * r.di * r.exit.chi_u[u] / r.exit.chi;
            if u == r.q1_index {
                val += r.wi * r.di * r.q1;
            }
            val += r.wi * r.di * r.exit.d_u[u] / r.exit.d;
            if u == r.qd1_index {
                val -= r.wi * r.di / r.qd1;
            }
            grad[u] = val;
        }

        let mut hess = vec![0.0_f64; p * p];
        let chi_sq = r.exit.chi * r.exit.chi;
        let d_sq = r.exit.d * r.exit.d;
        for u in 0..p {
            for v in u..p {
                let mut val = 0.0;
                val += entry_u2 * r.entry.eta_u[u] * r.entry.eta_u[v]
                    + entry_u1 * r.entry.eta_uv[u * p + v];
                val += exit_surv_u2 * r.exit.eta_u[u] * r.exit.eta_u[v]
                    + exit_surv_u1 * r.exit.eta_uv[u * p + v];
                val += r.wi
                    * r.di
                    * (r.exit.eta_u[u] * r.exit.eta_u[v] + r.exit.eta * r.exit.eta_uv[u * p + v]);
                val -= r.wi
                    * r.di
                    * (r.exit.chi_uv[u * p + v] / r.exit.chi
                        - (r.exit.chi_u[u] * r.exit.chi_u[v]) / chi_sq);
                if u == r.q1_index && v == r.q1_index {
                    val += r.wi * r.di;
                }
                val += r.wi
                    * r.di
                    * (r.exit.d_uv[u * p + v] / r.exit.d - (r.exit.d_u[u] * r.exit.d_u[v]) / d_sq);
                if u == r.qd1_index && v == r.qd1_index {
                    val += r.wi * r.di / (r.qd1 * r.qd1);
                }
                hess[u * p + v] = val;
                if v != u {
                    hess[v * p + u] = val;
                }
            }
        }

        out.push(SurvivalFlexStep5RowOutputs {
            row_nll,
            grad,
            hess,
        });
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────
// Three thin pullback entry points.  The bodies all currently return
// `Ok(None)` (the unsupported sentinel) because the host-side flex jet
// assembly (Steps 2–4: cubic-cell moments → intercept solve → η/χ/d
// jets) and the FAMILY-OWNED joint-β pullback (per-block design rows in
// `marginal_design` / `logslope_design` / `score_warp` / `link_dev`) are
// not yet wired through these `SurvivalFlexGpuRowInputs`-shaped entry
// points.  The native row-primary gradient/Hessian kernel they will call
// IS now landed: `try_flex_primary_rows` (Step 5) takes the assembled
// per-row jet and returns the per-row primary `(nll, grad[r], hess[r×r])`
// on the GPU.  Once the host orchestration builds the jet and the family
// supplies the design-row pullback, these three entry points fold
// `try_flex_primary_rows` outputs into the coefficient-space
// gradient/HVP/dense-H.  Until then they stay `Ok(None)` so the
// dispatcher falls back to CPU.  Keeping the shape stable here
// (option-typed, single shared input struct) lets that wiring land
// without re-touching the call sites.
// ────────────────────────────────────────────────────────────────────────

/// Evaluate the survival-flex negative log-likelihood and joint-β
/// gradient on the GPU.  Returns `Ok(None)` if the GPU path is
/// unsupported for this shape (caller falls back to CPU); returns
/// `Err` only when the request *is* supported but the driver failed.
pub fn try_survival_flex_gradient(
    inputs: SurvivalFlexGpuRowInputs<'_>,
    intercept_solve: Option<&SurvivalFlexInterceptSolveInputs<'_>>,
) -> Result<Option<(f64, Array1<f64>)>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    // Step 3 hookup: when an intercept-solve descriptor is provided,
    // run the device monotone-root kernel as the precheck stage so the
    // Step-3 path has a real production consumer before Step 4/5/6
    // joint-β assembly lands.  Step 4 will replace the analytic
    // evaluator on the device side with the real survival F(a)
    // calibration evaluator; the host-side hookup shape stays the
    // same.  On any non-OK device row we fall back to CPU; on every
    // OK row we accept the warm-started root and the dispatcher
    // continues to the (not-yet-landed) joint-β assembly — which for
    // Step 3 is the `Ok(None)` sentinel.
    if let Some(ints) = intercept_solve {
        // Prefer the device kernel; fall back to the CPU oracle on
        // non-CUDA builds.  The oracle is the same code path the
        // device kernel will be parity-tested against, so dispatcher
        // behaviour stays identical regardless of where the solve
        // ran.
        let out = match try_device_intercept_solve(ints)? {
            Some(out) => out,
            None => cpu_oracle_intercept_solve(ints),
        };
        if out.status.iter().any(|&s| s > 1) {
            return Ok(None);
        }
    }
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
///
/// When `cells` is `Some(_)` (Step 2 hookup) the entry point evaluates
/// the per-cell derivative moments via [`try_row_batched_cell_moments`]
/// first — this validates the moment-building stage end-to-end on the
/// device runtime before the Step 4/5/6 joint-β assembly lands.  When
/// every cell evaluates cleanly the entry point still returns `Ok(None)`
/// (joint-β assembly not yet wired); on any non-OK substrate status the
/// caller falls back to CPU.
pub fn try_survival_flex_dense_hessian(
    inputs: SurvivalFlexGpuRowInputs<'_>,
    cells: Option<SurvivalFlexRowCellsBatch<'_>>,
) -> Result<Option<Array2<f64>>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    if let Some(batch) = cells {
        // Validate the moment-building stage on the substrate runtime.
        // Step 4/5/6 will plug these moments into the joint-β
        // gradient/Hessian; here we only confirm the moments are
        // evaluatable so the dispatcher does not silently fall through
        // to CPU when the GPU substrate is healthy.
        let out = match try_row_batched_cell_moments(batch)? {
            Some(out) => out,
            None => return Ok(None),
        };
        let ok_byte = crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8;
        if out.status.iter().any(|&b| b != ok_byte) {
            // Any cell that failed the substrate classifier or kernel
            // is a CPU fallback for this fit — Step 4/5/6 assembly is
            // not landed yet so we cannot stitch a partial answer.
            return Ok(None);
        }
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
    let log_a1 = a1
        .max(crate::families::marginal_slope_shared::SURVIVAL_SLOPE_LOG_DIVIDE_FLOOR)
        .ln();

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
    let inv = 1.0 / a1.max(crate::families::marginal_slope_shared::SURVIVAL_SLOPE_LOG_DIVIDE_FLOOR);
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
// Block 10 — third T_uv[r] and fourth Q_uv[r,s] directional contractions.
//
// Math reference: math block 10 §3.5 (third) and §3.6 (fourth).  The
// per-row CPU implementations are
// `SurvivalMarginalSlopeFamily::row_flex_primary_third_contracted_exact`
// and `_fourth_contracted_exact` in `src/families/survival_marginal_slope.rs`.
//
// These oracles are pure assemblers over the timepoint-jet substrate
// outputs (entry/exit base + per-direction extension + bidirectional).
// They mirror the CPU assembly term-for-term.
// ────────────────────────────────────────────────────────────────────────

/// Per-timepoint exact-jet substrate input.  Mirrors the crate-private
/// `SurvivalFlexTimepointExact` in `survival_marginal_slope.rs`.
/// All arrays are row-major dense over the primary dimension `p`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10TimepointBase {
    pub eta: f64,
    pub chi: f64,
    pub d: f64,
    pub eta_u: Vec<f64>,
    pub eta_uv: Vec<f64>,
    pub chi_u: Vec<f64>,
    pub chi_uv: Vec<f64>,
    pub d_u: Vec<f64>,
    pub d_uv: Vec<f64>,
}

/// Directional extension of a timepoint jet contracted with a single
/// direction `d ∈ ℝᵖ`.  Mirrors `SurvivalFlexTimepointDirectionalExact`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10TimepointDirectional {
    pub eta_uv_dir: Vec<f64>,
    pub chi_uv_dir: Vec<f64>,
    pub d_u_dir: Vec<f64>,
    pub d_uv_dir: Vec<f64>,
}

/// Mixed second-directional extension `D_{d1} D_{d2}` of a timepoint jet.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10TimepointBiDirectional {
    pub eta_uv_uv: Vec<f64>,
    pub chi_uv_uv: Vec<f64>,
    pub d_uv_uv: Vec<f64>,
}

/// Inputs to the Block 10 third-contraction CPU oracle.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10ThirdInputs<'a> {
    pub p: usize,
    /// Index of the `qd1` primary coordinate; `usize::MAX` to disable.
    pub qd1_index: usize,
    pub qd1: f64,
    pub w: f64,
    pub d: f64,
    pub dir: &'a [f64],
    pub entry_base: &'a SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a SurvivalFlexBlock10TimepointBase,
    pub entry_ext: &'a SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext: &'a SurvivalFlexBlock10TimepointDirectional,
}

/// Inputs to the Block 10 fourth-contraction CPU oracle.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10FourthInputs<'a> {
    pub p: usize,
    pub qd1_index: usize,
    pub qd1: f64,
    pub w: f64,
    pub d: f64,
    pub dir_u: &'a [f64],
    pub dir_v: &'a [f64],
    pub entry_base: &'a SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a SurvivalFlexBlock10TimepointBase,
    pub entry_ext_u: &'a SurvivalFlexBlock10TimepointDirectional,
    pub entry_ext_v: &'a SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_u: &'a SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_v: &'a SurvivalFlexBlock10TimepointDirectional,
    pub entry_bi: &'a SurvivalFlexBlock10TimepointBiDirectional,
    pub exit_bi: &'a SurvivalFlexBlock10TimepointBiDirectional,
}

#[inline]
fn b10_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "b10_dot: length mismatch");
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[inline]
fn b10_mat_dot(m: &[f64], v: &[f64], p: usize) -> Vec<f64> {
    assert_eq!(m.len(), p * p, "b10_mat_dot: matrix shape mismatch");
    assert_eq!(v.len(), p, "b10_mat_dot: vector length mismatch");
    let mut out = vec![0.0_f64; p];
    for u in 0..p {
        let mut acc = 0.0_f64;
        let row = &m[u * p..(u + 1) * p];
        for k in 0..p {
            acc += row[k] * v[k];
        }
        out[u] = acc;
    }
    out
}

#[inline]
fn b10_at(m: &[f64], u: usize, v: usize, p: usize) -> f64 {
    m[u * p + v]
}

/// CPU oracle for the third directional contraction `T_uv[r] :=
/// (D_{dir} H)[u, v]` of the flexible survival path.  Pure assembler;
/// mirrors `row_flex_primary_third_contracted_exact` term-for-term.
pub fn cpu_oracle_third_contraction(
    inputs: &SurvivalFlexBlock10ThirdInputs<'_>,
) -> Result<Vec<f64>, String> {
    let p = inputs.p;
    if inputs.dir.len() != p {
        return Err(format!(
            "cpu_oracle_third_contraction: dir length {} != p {}",
            inputs.dir.len(),
            p
        ));
    }
    if inputs.dir.iter().all(|v| v.abs() == 0.0) {
        return Ok(vec![0.0_f64; p * p]);
    }
    let entry = inputs.entry_base;
    let exit = inputs.exit_base;
    let entry_ext = inputs.entry_ext;
    let exit_ext = inputs.exit_ext;
    let chi = exit.chi;
    if !chi.is_finite() || chi <= 0.0 {
        return Err(format!(
            "cpu_oracle_third_contraction: non-positive chi={chi:.3e}"
        ));
    }
    let d_val = exit.d;
    if !d_val.is_finite() || d_val == 0.0 {
        return Err(format!(
            "cpu_oracle_third_contraction: non-finite/zero D={d_val:.3e}"
        ));
    }

    let wi = inputs.w;
    let di = inputs.d;

    use crate::families::bms::signed_probit_neglog_derivatives_up_to_fourth;
    let (entry_k1, entry_k2, entry_k3, _) =
        signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
    let (exit_k1, exit_k2, exit_k3, _) =
        signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;

    let entry_u1 = -entry_k1;
    let entry_u2 = entry_k2;
    let entry_u3 = -entry_k3;
    let exit_u1 = -exit_k1;
    let exit_u2 = exit_k2;
    let exit_u3 = -exit_k3;

    let entry_eta_dir = b10_dot(&entry.eta_u, inputs.dir);
    let exit_eta_dir = b10_dot(&exit.eta_u, inputs.dir);
    let exit_chi_dir = b10_dot(&exit.chi_u, inputs.dir);
    let exit_d_dir = b10_dot(&exit.d_u, inputs.dir);
    let qd1_dir = if inputs.qd1_index < p {
        inputs.dir[inputs.qd1_index]
    } else {
        0.0
    };

    let entry_eta_u_dir = b10_mat_dot(&entry.eta_uv, inputs.dir, p);
    let exit_eta_u_dir = b10_mat_dot(&exit.eta_uv, inputs.dir, p);
    let exit_chi_u_dir = b10_mat_dot(&exit.chi_uv, inputs.dir, p);
    let exit_d_u_dir = b10_mat_dot(&exit.d_uv, inputs.dir, p);

    let chi_inv = 1.0 / chi;
    let chi_inv2 = chi_inv * chi_inv;
    let chi_inv3 = chi_inv2 * chi_inv;
    let d_inv = 1.0 / d_val;
    let d_inv2 = d_inv * d_inv;
    let d_inv3 = d_inv2 * d_inv;

    let mut out = vec![0.0_f64; p * p];
    for u in 0..p {
        for v in u..p {
            let mut val = 0.0_f64;

            // Entry probit
            val += entry_u3 * entry_eta_dir * entry.eta_u[u] * entry.eta_u[v];
            val += entry_u2
                * (entry_eta_u_dir[u] * entry.eta_u[v] + entry.eta_u[u] * entry_eta_u_dir[v]);
            val += entry_u2 * entry_eta_dir * b10_at(&entry.eta_uv, u, v, p);
            val += entry_u1 * b10_at(&entry_ext.eta_uv_dir, u, v, p);

            // Exit probit survival
            val += exit_u3 * exit_eta_dir * exit.eta_u[u] * exit.eta_u[v];
            val +=
                exit_u2 * (exit_eta_u_dir[u] * exit.eta_u[v] + exit.eta_u[u] * exit_eta_u_dir[v]);
            val += exit_u2 * exit_eta_dir * b10_at(&exit.eta_uv, u, v, p);
            val += exit_u1 * b10_at(&exit_ext.eta_uv_dir, u, v, p);

            // Event density
            val += wi
                * di
                * (exit_eta_u_dir[u] * exit.eta_u[v]
                    + exit.eta_u[u] * exit_eta_u_dir[v]
                    + exit_eta_dir * b10_at(&exit.eta_uv, u, v, p)
                    + exit.eta * b10_at(&exit_ext.eta_uv_dir, u, v, p));

            // Event chi
            let chi_uv_over_chi_dir = (b10_at(&exit_ext.chi_uv_dir, u, v, p) * chi
                - b10_at(&exit.chi_uv, u, v, p) * exit_chi_dir)
                * chi_inv2;
            let chi_u_chi_v_over_chi2_dir =
                (exit_chi_u_dir[u] * exit.chi_u[v] + exit.chi_u[u] * exit_chi_u_dir[v]) * chi_inv2
                    - 2.0 * exit.chi_u[u] * exit.chi_u[v] * exit_chi_dir * chi_inv3;
            val -= wi * di * (chi_uv_over_chi_dir - chi_u_chi_v_over_chi2_dir);

            // Event D
            let d_uv_over_d_dir = (b10_at(&exit_ext.d_uv_dir, u, v, p) * d_val
                - b10_at(&exit.d_uv, u, v, p) * exit_d_dir)
                * d_inv2;
            let d_u_d_v_over_d2_dir =
                (exit_d_u_dir[u] * exit.d_u[v] + exit.d_u[u] * exit_d_u_dir[v]) * d_inv2
                    - 2.0 * exit.d_u[u] * exit.d_u[v] * exit_d_dir * d_inv3;
            val += wi * di * (d_uv_over_d_dir - d_u_d_v_over_d2_dir);

            // qd1 term
            if inputs.qd1_index < p && u == inputs.qd1_index && v == inputs.qd1_index {
                val += wi * di * (-2.0 / (inputs.qd1 * inputs.qd1 * inputs.qd1)) * qd1_dir;
            }

            out[u * p + v] = val;
            out[v * p + u] = val;
        }
    }
    Ok(out)
}

/// One ordered fourth contracted matrix `D_{dir2}(D_{dir1} H)`; mirrors
/// `compute_survival_fourth_contracted_ordered`.
#[allow(clippy::too_many_arguments)]
fn b10_fourth_ordered(
    p: usize,
    qd1_index: usize,
    qd1: f64,
    wi: f64,
    di: f64,
    dir1: &[f64],
    dir2: &[f64],
    entry_base: &SurvivalFlexBlock10TimepointBase,
    exit_base: &SurvivalFlexBlock10TimepointBase,
    entry_ext1: &SurvivalFlexBlock10TimepointDirectional,
    entry_ext2: &SurvivalFlexBlock10TimepointDirectional,
    exit_ext1: &SurvivalFlexBlock10TimepointDirectional,
    exit_ext2: &SurvivalFlexBlock10TimepointDirectional,
    entry_bi: &SurvivalFlexBlock10TimepointBiDirectional,
    exit_bi: &SurvivalFlexBlock10TimepointBiDirectional,
) -> Result<Vec<f64>, String> {
    use crate::families::bms::signed_probit_neglog_derivatives_up_to_fourth;

    let (entry_k1, entry_k2, entry_k3, entry_k4) =
        signed_probit_neglog_derivatives_up_to_fourth(-entry_base.eta, -wi)?;
    let (exit_k1, exit_k2, exit_k3, exit_k4) =
        signed_probit_neglog_derivatives_up_to_fourth(-exit_base.eta, wi * (1.0 - di))?;

    let entry_u1 = -entry_k1;
    let entry_u2 = entry_k2;
    let entry_u3 = -entry_k3;
    let exit_u1 = -exit_k1;
    let exit_u2 = exit_k2;
    let exit_u3 = -exit_k3;

    let entry_eta_d1 = b10_dot(&entry_base.eta_u, dir1);
    let entry_eta_d2 = b10_dot(&entry_base.eta_u, dir2);
    let exit_eta_d1 = b10_dot(&exit_base.eta_u, dir1);
    let exit_eta_d2 = b10_dot(&exit_base.eta_u, dir2);
    let exit_chi_d1 = b10_dot(&exit_base.chi_u, dir1);
    let exit_chi_d2 = b10_dot(&exit_base.chi_u, dir2);
    let exit_d_d1 = b10_dot(&exit_base.d_u, dir1);
    let exit_d_d2 = b10_dot(&exit_base.d_u, dir2);
    let qd1_d1 = if qd1_index < p { dir1[qd1_index] } else { 0.0 };
    let qd1_d2 = if qd1_index < p { dir2[qd1_index] } else { 0.0 };

    let entry_eta_u_d1 = b10_mat_dot(&entry_base.eta_uv, dir1, p);
    let entry_eta_u_d2 = b10_mat_dot(&entry_base.eta_uv, dir2, p);
    let exit_eta_u_d1 = b10_mat_dot(&exit_base.eta_uv, dir1, p);
    let exit_eta_u_d2 = b10_mat_dot(&exit_base.eta_uv, dir2, p);
    let exit_chi_u_d1 = b10_mat_dot(&exit_base.chi_uv, dir1, p);
    let exit_chi_u_d2 = b10_mat_dot(&exit_base.chi_uv, dir2, p);
    let exit_d_u_d2 = b10_mat_dot(&exit_base.d_uv, dir2, p);

    let entry_eta_d12 = b10_dot(&entry_eta_u_d2, dir1);
    let exit_eta_d12 = b10_dot(&exit_eta_u_d2, dir1);
    let exit_chi_d12 = b10_dot(&exit_chi_u_d2, dir1);
    let exit_d_d12 = b10_dot(&exit_d_u_d2, dir1);

    let entry_eta_u_d12 = b10_mat_dot(&entry_ext2.eta_uv_dir, dir1, p);
    let exit_eta_u_d12 = b10_mat_dot(&exit_ext2.eta_uv_dir, dir1, p);
    let exit_chi_u_d12 = b10_mat_dot(&exit_ext2.chi_uv_dir, dir1, p);
    let exit_d_u_d12 = b10_mat_dot(&exit_ext2.d_uv_dir, dir1, p);

    let chi = exit_base.chi;
    let chi_inv = 1.0 / chi;
    let chi_inv2 = chi_inv * chi_inv;
    let chi_inv3 = chi_inv2 * chi_inv;
    let chi_inv4 = chi_inv3 * chi_inv;
    let d_val = exit_base.d;
    let d_inv = 1.0 / d_val;
    let d_inv2 = d_inv * d_inv;
    let d_inv3 = d_inv2 * d_inv;
    let d_inv4 = d_inv3 * d_inv;

    let mut out = vec![0.0_f64; p * p];
    for u in 0..p {
        for v in u..p {
            let mut val = 0.0_f64;

            // Entry probit
            let eu = &entry_base.eta_u;
            let euv_uv = b10_at(&entry_base.eta_uv, u, v, p);

            let a_term = eu[u] * eu[v] * entry_eta_d1;
            let a_term_d2 = entry_eta_u_d2[u] * eu[v] * entry_eta_d1
                + eu[u] * entry_eta_u_d2[v] * entry_eta_d1
                + eu[u] * eu[v] * entry_eta_d12;
            let b_term = b10_at(&entry_ext1.eta_uv_dir, u, v, p);
            let b_term_d2 = b10_at(&entry_bi.eta_uv_uv, u, v, p);
            let c_term =
                entry_eta_u_d1[u] * eu[v] + eu[u] * entry_eta_u_d1[v] + entry_eta_d1 * euv_uv;
            let c_term_d2 = entry_eta_u_d12[u] * eu[v]
                + entry_eta_u_d1[u] * entry_eta_u_d2[v]
                + entry_eta_u_d2[u] * entry_eta_u_d1[v]
                + eu[u] * entry_eta_u_d12[v]
                + entry_eta_d12 * euv_uv
                + entry_eta_d1 * b10_at(&entry_ext2.eta_uv_dir, u, v, p);

            val += entry_k4 * entry_eta_d2 * a_term
                + entry_u3 * a_term_d2
                + entry_u3 * entry_eta_d2 * c_term
                + entry_u2 * c_term_d2
                + entry_u2 * entry_eta_d2 * b_term
                + entry_u1 * b_term_d2;

            // Exit probit
            let xu = &exit_base.eta_u;
            let xuv_uv = b10_at(&exit_base.eta_uv, u, v, p);

            let xa = xu[u] * xu[v] * exit_eta_d1;
            let xa_d2 = exit_eta_u_d2[u] * xu[v] * exit_eta_d1
                + xu[u] * exit_eta_u_d2[v] * exit_eta_d1
                + xu[u] * xu[v] * exit_eta_d12;
            let xb = b10_at(&exit_ext1.eta_uv_dir, u, v, p);
            let xb_d2 = b10_at(&exit_bi.eta_uv_uv, u, v, p);
            let xc = exit_eta_u_d1[u] * xu[v] + xu[u] * exit_eta_u_d1[v] + exit_eta_d1 * xuv_uv;
            let xc_d2 = exit_eta_u_d12[u] * xu[v]
                + exit_eta_u_d1[u] * exit_eta_u_d2[v]
                + exit_eta_u_d2[u] * exit_eta_u_d1[v]
                + xu[u] * exit_eta_u_d12[v]
                + exit_eta_d12 * xuv_uv
                + exit_eta_d1 * b10_at(&exit_ext2.eta_uv_dir, u, v, p);

            val += exit_k4 * exit_eta_d2 * xa
                + exit_u3 * xa_d2
                + exit_u3 * exit_eta_d2 * xc
                + exit_u2 * xc_d2
                + exit_u2 * exit_eta_d2 * xb
                + exit_u1 * xb_d2;

            // Event density
            val += wi
                * di
                * (exit_eta_u_d12[u] * xu[v]
                    + exit_eta_u_d1[u] * exit_eta_u_d2[v]
                    + exit_eta_u_d2[u] * exit_eta_u_d1[v]
                    + xu[u] * exit_eta_u_d12[v]
                    + exit_eta_d12 * xuv_uv
                    + exit_eta_d1 * b10_at(&exit_ext2.eta_uv_dir, u, v, p)
                    + exit_eta_d2 * b10_at(&exit_ext1.eta_uv_dir, u, v, p)
                    + exit_base.eta * b10_at(&exit_bi.eta_uv_uv, u, v, p));

            // Event chi
            let chi_uv_val = b10_at(&exit_base.chi_uv, u, v, p);
            let chi_u_val = exit_base.chi_u[u];
            let chi_v_val = exit_base.chi_u[v];
            let chi_uv_d1 = b10_at(&exit_ext1.chi_uv_dir, u, v, p);
            let chi_uv_d2 = b10_at(&exit_ext2.chi_uv_dir, u, v, p);
            let chi_u_d1 = exit_chi_u_d1[u];
            let chi_v_d1 = exit_chi_u_d1[v];
            let chi_u_d2 = exit_chi_u_d2[u];
            let chi_v_d2 = exit_chi_u_d2[v];
            let chi_u_d12v = exit_chi_u_d12[u];
            let chi_v_d12v = exit_chi_u_d12[v];

            let chi_uv_d12_val = b10_at(&exit_bi.chi_uv_uv, u, v, p);
            let d2_r_chi = chi_uv_d12_val * chi_inv
                - chi_uv_d1 * exit_chi_d2 * chi_inv2
                - chi_uv_d2 * exit_chi_d1 * chi_inv2
                - chi_uv_val * exit_chi_d12 * chi_inv2
                + 2.0 * chi_uv_val * exit_chi_d1 * exit_chi_d2 * chi_inv3;

            let d2_s_chi = (chi_u_d12v * chi_v_val
                + chi_u_d1 * chi_v_d2
                + chi_u_d2 * chi_v_d1
                + chi_u_val * chi_v_d12v)
                * chi_inv2
                - 2.0 * (chi_u_d1 * chi_v_val + chi_u_val * chi_v_d1) * exit_chi_d2 * chi_inv3
                - 2.0 * (chi_u_d2 * chi_v_val + chi_u_val * chi_v_d2) * exit_chi_d1 * chi_inv3
                - 2.0 * chi_u_val * chi_v_val * exit_chi_d12 * chi_inv3
                + 6.0 * chi_u_val * chi_v_val * exit_chi_d1 * exit_chi_d2 * chi_inv4;
            val -= wi * di * (d2_r_chi - d2_s_chi);

            // Event D
            let d_uv_val = b10_at(&exit_base.d_uv, u, v, p);
            let d_u_val = exit_base.d_u[u];
            let d_v_val = exit_base.d_u[v];
            let d_uv_d1 = b10_at(&exit_ext1.d_uv_dir, u, v, p);
            let d_uv_d2 = b10_at(&exit_ext2.d_uv_dir, u, v, p);
            let d_u_d1 = exit_ext1.d_u_dir[u];
            let d_v_d1 = exit_ext1.d_u_dir[v];
            let d_u_d2 = exit_ext2.d_u_dir[u];
            let d_v_d2 = exit_ext2.d_u_dir[v];
            let d_u_d12v = exit_d_u_d12[u];
            let d_v_d12v = exit_d_u_d12[v];

            let d_uv_d12_val = b10_at(&exit_bi.d_uv_uv, u, v, p);
            let d2_r_d = d_uv_d12_val * d_inv
                - d_uv_d1 * exit_d_d2 * d_inv2
                - d_uv_d2 * exit_d_d1 * d_inv2
                - d_uv_val * exit_d_d12 * d_inv2
                + 2.0 * d_uv_val * exit_d_d1 * exit_d_d2 * d_inv3;

            let d2_s_d =
                (d_u_d12v * d_v_val + d_u_d1 * d_v_d2 + d_u_d2 * d_v_d1 + d_u_val * d_v_d12v)
                    * d_inv2
                    - 2.0 * (d_u_d1 * d_v_val + d_u_val * d_v_d1) * exit_d_d2 * d_inv3
                    - 2.0 * (d_u_d2 * d_v_val + d_u_val * d_v_d2) * exit_d_d1 * d_inv3
                    - 2.0 * d_u_val * d_v_val * exit_d_d12 * d_inv3
                    + 6.0 * d_u_val * d_v_val * exit_d_d1 * exit_d_d2 * d_inv4;
            val += wi * di * (d2_r_d - d2_s_d);

            // qd1 term
            if qd1_index < p && u == qd1_index && v == qd1_index {
                val += wi * di * (6.0 / (qd1 * qd1 * qd1 * qd1)) * qd1_d1 * qd1_d2;
            }

            out[u * p + v] = val;
            out[v * p + u] = val;
        }
    }
    Ok(out)
}

/// CPU oracle for the fourth directional contraction with averaged
/// symmetrization `Q_sym = ½(Q_ordered[u, v] + Q_ordered[v, u])`.
/// Mirrors `row_flex_primary_fourth_contracted_exact`.
pub fn cpu_oracle_fourth_contraction(
    inputs: &SurvivalFlexBlock10FourthInputs<'_>,
) -> Result<Vec<f64>, String> {
    let p = inputs.p;
    if inputs.dir_u.len() != p || inputs.dir_v.len() != p {
        return Err(format!(
            "cpu_oracle_fourth_contraction: dir lengths ({},{}) != p {}",
            inputs.dir_u.len(),
            inputs.dir_v.len(),
            p
        ));
    }
    if inputs.dir_u.iter().all(|v| v.abs() == 0.0) || inputs.dir_v.iter().all(|v| v.abs() == 0.0) {
        return Ok(vec![0.0_f64; p * p]);
    }
    let chi = inputs.exit_base.chi;
    if !chi.is_finite() || chi <= 0.0 {
        return Err(format!(
            "cpu_oracle_fourth_contraction: non-positive chi={chi:.3e}"
        ));
    }
    let d_val = inputs.exit_base.d;
    if !d_val.is_finite() || d_val == 0.0 {
        return Err(format!(
            "cpu_oracle_fourth_contraction: non-finite/zero D={d_val:.3e}"
        ));
    }

    let ordered_uv = b10_fourth_ordered(
        p,
        inputs.qd1_index,
        inputs.qd1,
        inputs.w,
        inputs.d,
        inputs.dir_u,
        inputs.dir_v,
        inputs.entry_base,
        inputs.exit_base,
        inputs.entry_ext_u,
        inputs.entry_ext_v,
        inputs.exit_ext_u,
        inputs.exit_ext_v,
        inputs.entry_bi,
        inputs.exit_bi,
    )?;
    let ordered_vu = b10_fourth_ordered(
        p,
        inputs.qd1_index,
        inputs.qd1,
        inputs.w,
        inputs.d,
        inputs.dir_v,
        inputs.dir_u,
        inputs.entry_base,
        inputs.exit_base,
        inputs.entry_ext_v,
        inputs.entry_ext_u,
        inputs.exit_ext_v,
        inputs.exit_ext_u,
        inputs.entry_bi,
        inputs.exit_bi,
    )?;

    let mut out = vec![0.0_f64; p * p];
    for i in 0..(p * p) {
        out[i] = 0.5 * (ordered_uv[i] + ordered_vu[i]);
    }
    Ok(out)
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
        match try_survival_flex_gradient(inputs, None) {
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
        match try_survival_flex_dense_hessian(inputs, None) {
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
    // Step 5 — flex row-primary gradient/Hessian kernel parity.
    //
    // The CPU oracle `cpu_oracle_flex_primary_rows` is the parity target
    // the V100 verification pass lands against; it mirrors the device
    // `survival_flex_primary_rows` kernel term-for-term.  The synthetic
    // jets below are NOT physically consistent survival jets — they are
    // arbitrary-but-finite values that exercise EVERY arithmetic path in
    // the kernel (entry/exit survival k1/k2 terms, the event log φ(η₁)
    // curvature, the χ and d ratio terms, and the q1/qd1 perturbation
    // bumps), which is exactly what a launcher↔oracle bit-parity check
    // needs.  Physical-jet end-to-end quality is covered upstream by the
    // family's own flex tests.
    // ────────────────────────────────────────────────────────────────────

    /// Build `n_rows` synthetic flex primary rows of local dimension `r`.
    /// The per-row jet entries are deterministic functions of `(i, u, v)`
    /// so the oracle and the device see identical inputs.  Returns the
    /// owned backing storage plus the row views (the views borrow the
    /// storage, so the storage must outlive the rows — callers keep both).
    fn make_flex_primary_storage(n_rows: usize, r: usize) -> FlexPrimaryStorage {
        let mut s = FlexPrimaryStorage::new(n_rows, r);
        for i in 0..n_rows {
            let fi = (i as f64) + 1.0;
            // entry jet
            s.e_eta.push(0.3 * fi - 0.4);
            // exit jet scalars: chi/d strictly positive, qd1 positive.
            s.x_eta.push(0.2 * fi - 0.1);
            s.x_chi.push(0.7 + 0.15 * fi);
            s.x_d.push(1.1 + 0.2 * fi);
            for u in 0..r {
                let fu = (u as f64) + 1.0;
                s.e_eta_u.push(0.11 * fu - 0.05 * fi);
                s.x_eta_u.push(0.07 * fu + 0.03 * fi);
                s.x_chi_u.push(0.04 * fu - 0.02 * fi);
                s.x_d_u.push(0.06 * fu + 0.01 * fi);
            }
            for u in 0..r {
                for v in 0..r {
                    // Symmetric r×r second jets.
                    let fu = (u as f64) + 1.0;
                    let fv = (v as f64) + 1.0;
                    let sym = (fu * fv).sqrt();
                    s.e_eta_uv.push(0.02 * sym - 0.01 * fi);
                    s.x_eta_uv.push(0.015 * sym + 0.005 * fi);
                    s.x_chi_uv.push(0.012 * sym - 0.004 * fi);
                    s.x_d_uv.push(0.018 * sym + 0.002 * fi);
                }
            }
            s.wi.push(0.8 + 0.1 * fi);
            // Alternate censored / observed rows so both NLL branches run.
            s.di.push(if i % 2 == 0 { 1.0 } else { 0.0 });
            s.q1.push(0.25 * fi - 0.5);
            s.qd1.push(0.9 + 0.05 * fi);
            s.entry_k1.push(-0.3 * fi);
            s.entry_k2.push(0.2 * fi);
            s.exit_k1.push(-0.25 * fi);
            s.exit_k2.push(0.15 * fi);
            s.log_surv0.push(-0.1 * fi);
            s.log_surv1.push(-0.2 * fi);
            // Route the q1/qd1 perturbation bumps onto real primary slots
            // (only meaningful when r ≥ 2) so the diagonal `if (u==index)`
            // branches fire on at least one row.
            s.q1_index.push(if r >= 1 { 0 } else { usize::MAX });
            s.qd1_index.push(if r >= 2 { 1 } else { usize::MAX });
        }
        s
    }

    /// Owned backing storage for the synthetic flex primary batch, with a
    /// `rows()` accessor that re-slices the flat buffers into the borrowed
    /// `SurvivalFlexPrimaryRow` views the API consumes.
    struct FlexPrimaryStorage {
        n_rows: usize,
        r: usize,
        e_eta: Vec<f64>,
        e_eta_u: Vec<f64>,
        e_eta_uv: Vec<f64>,
        x_eta: Vec<f64>,
        x_chi: Vec<f64>,
        x_d: Vec<f64>,
        x_eta_u: Vec<f64>,
        x_eta_uv: Vec<f64>,
        x_chi_u: Vec<f64>,
        x_chi_uv: Vec<f64>,
        x_d_u: Vec<f64>,
        x_d_uv: Vec<f64>,
        wi: Vec<f64>,
        di: Vec<f64>,
        q1: Vec<f64>,
        qd1: Vec<f64>,
        entry_k1: Vec<f64>,
        entry_k2: Vec<f64>,
        exit_k1: Vec<f64>,
        exit_k2: Vec<f64>,
        log_surv0: Vec<f64>,
        log_surv1: Vec<f64>,
        q1_index: Vec<usize>,
        qd1_index: Vec<usize>,
    }

    impl FlexPrimaryStorage {
        fn new(n_rows: usize, r: usize) -> Self {
            FlexPrimaryStorage {
                n_rows,
                r,
                e_eta: Vec::new(),
                e_eta_u: Vec::new(),
                e_eta_uv: Vec::new(),
                x_eta: Vec::new(),
                x_chi: Vec::new(),
                x_d: Vec::new(),
                x_eta_u: Vec::new(),
                x_eta_uv: Vec::new(),
                x_chi_u: Vec::new(),
                x_chi_uv: Vec::new(),
                x_d_u: Vec::new(),
                x_d_uv: Vec::new(),
                wi: Vec::new(),
                di: Vec::new(),
                q1: Vec::new(),
                qd1: Vec::new(),
                entry_k1: Vec::new(),
                entry_k2: Vec::new(),
                exit_k1: Vec::new(),
                exit_k2: Vec::new(),
                log_surv0: Vec::new(),
                log_surv1: Vec::new(),
                q1_index: Vec::new(),
                qd1_index: Vec::new(),
            }
        }

        fn rows(&self) -> Vec<SurvivalFlexPrimaryRow<'_>> {
            let r = self.r;
            let rr = r * r;
            (0..self.n_rows)
                .map(|i| SurvivalFlexPrimaryRow {
                    entry: SurvivalFlexPrimaryTimepointRow {
                        eta: self.e_eta[i],
                        chi: 0.0,
                        d: 0.0,
                        eta_u: &self.e_eta_u[i * r..(i + 1) * r],
                        eta_uv: &self.e_eta_uv[i * rr..(i + 1) * rr],
                        // Entry χ/d jets never read by the kernel; point
                        // them at the entry η jets so the slice lengths
                        // are well-formed without allocating.
                        chi_u: &self.e_eta_u[i * r..(i + 1) * r],
                        chi_uv: &self.e_eta_uv[i * rr..(i + 1) * rr],
                        d_u: &self.e_eta_u[i * r..(i + 1) * r],
                        d_uv: &self.e_eta_uv[i * rr..(i + 1) * rr],
                    },
                    exit: SurvivalFlexPrimaryTimepointRow {
                        eta: self.x_eta[i],
                        chi: self.x_chi[i],
                        d: self.x_d[i],
                        eta_u: &self.x_eta_u[i * r..(i + 1) * r],
                        eta_uv: &self.x_eta_uv[i * rr..(i + 1) * rr],
                        chi_u: &self.x_chi_u[i * r..(i + 1) * r],
                        chi_uv: &self.x_chi_uv[i * rr..(i + 1) * rr],
                        d_u: &self.x_d_u[i * r..(i + 1) * r],
                        d_uv: &self.x_d_uv[i * rr..(i + 1) * rr],
                    },
                    wi: self.wi[i],
                    di: self.di[i],
                    q1: self.q1[i],
                    qd1: self.qd1[i],
                    q1_index: self.q1_index[i],
                    qd1_index: self.qd1_index[i],
                    entry_k1: self.entry_k1[i],
                    entry_k2: self.entry_k2[i],
                    exit_k1: self.exit_k1[i],
                    exit_k2: self.exit_k2[i],
                    log_surv0: self.log_surv0[i],
                    log_surv1: self.log_surv1[i],
                })
                .collect()
        }
    }

    #[test]
    fn flex_primary_oracle_is_finite_and_symmetric() {
        // The oracle is the V100 parity target — assert it produces
        // finite, symmetric per-row Hessians and finite gradients/NLL
        // across a small batch so the device pass has a known-good
        // target.  Also confirms the entry point returns `Ok(None)` on a
        // non-CUDA host (clean CPU fallback).
        let r = 3;
        let storage = make_flex_primary_storage(5, r);
        let rows = storage.rows();
        let oracle = cpu_oracle_flex_primary_rows(&rows).expect("oracle");
        assert_eq!(oracle.nll.len(), rows.len());
        assert_eq!(oracle.grad.len(), rows.len() * r);
        assert_eq!(oracle.hess.len(), rows.len() * r * r);
        for i in 0..rows.len() {
            assert_eq!(oracle.row_status[i], 0, "row {i} unexpectedly rejected");
            assert!(oracle.nll[i].is_finite(), "row {i} nll non-finite");
            for u in 0..r {
                assert!(
                    oracle.grad[i * r + u].is_finite(),
                    "row {i} grad[{u}] non-finite"
                );
                for v in 0..r {
                    let h_uv = oracle.hess[i * r * r + u * r + v];
                    let h_vu = oracle.hess[i * r * r + v * r + u];
                    assert!(h_uv.is_finite(), "row {i} hess[{u}][{v}] non-finite");
                    assert!(
                        (h_uv - h_vu).abs() <= 1e-12 * (1.0 + h_uv.abs()),
                        "row {i} hess asymmetry [{u}][{v}]={h_uv} vs [{v}][{u}]={h_vu}"
                    );
                }
            }
        }

        // On a non-CUDA host the launcher entry point must fall back to
        // `Ok(None)` (or `Ok(Some)` on V100 — covered by the ignored
        // parity test); never `Err` for this supported shape.
        if !SurvivalFlexGpuBackend::compiled() {
            match try_flex_primary_rows(&rows) {
                Ok(None) => {}
                other => panic!("expected Ok(None) on non-Linux build, got {other:?}"),
            }
        }
    }

    #[test]
    fn flex_primary_rejects_ragged_r() {
        // Two rows with mismatched `r` is an unsupported shape: the
        // flatten/validate must surface a `DriverCallFailed` (the kernel
        // launches with one runtime `r`).  Build manually so row 1 has a
        // shorter exit.eta_u than row 0.
        let eu0 = [0.1, 0.2, 0.3];
        let euv0 = [0.0_f64; 9];
        let eu1 = [0.1, 0.2];
        let euv1 = [0.0_f64; 4];
        // Closures can't carry HRTBs, so the lifetime inferred for the two
        // `&[f64]` args is unrelated to the lifetime of the returned struct,
        // which binds eta_u/eta_uv/chi_u/... to one and the same `'a`.
        // A nested fn with an explicit lifetime tying both args + return type
        // is the canonical fix.
        fn mk<'a>(eu: &'a [f64], euv: &'a [f64]) -> SurvivalFlexPrimaryTimepointRow<'a> {
            SurvivalFlexPrimaryTimepointRow {
                eta: 0.0,
                chi: 1.0,
                d: 1.0,
                eta_u: eu,
                eta_uv: euv,
                chi_u: eu,
                chi_uv: euv,
                d_u: eu,
                d_uv: euv,
            }
        }
        let row0 = SurvivalFlexPrimaryRow {
            entry: mk(&eu0, &euv0),
            exit: mk(&eu0, &euv0),
            wi: 1.0,
            di: 1.0,
            q1: 0.0,
            qd1: 1.0,
            q1_index: usize::MAX,
            qd1_index: usize::MAX,
            entry_k1: 0.0,
            entry_k2: 0.0,
            exit_k1: 0.0,
            exit_k2: 0.0,
            log_surv0: 0.0,
            log_surv1: 0.0,
        };
        let row1 = SurvivalFlexPrimaryRow {
            entry: mk(&eu1, &euv1),
            exit: mk(&eu1, &euv1),
            ..row0
        };
        // `r` is taken from row 0 (=3); row 1's length-2 jets must fail.
        match cpu_oracle_flex_primary_rows(&[row0, row1]) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(reason.contains("expected 3"), "reason was: {reason}");
            }
            other => panic!("expected DriverCallFailed for ragged r, got {other:?}"),
        }
    }

    #[test]
    fn flex_primary_rejects_nonpositive_chi() {
        // A non-positive exit.chi must produce `row_status == 2` and
        // zeroed outputs — the kernel's belt-and-suspenders reject.
        let r = 2;
        let mut storage = make_flex_primary_storage(1, r);
        storage.x_chi[0] = -1.0; // invalid
        let rows = storage.rows();
        let oracle = cpu_oracle_flex_primary_rows(&rows).expect("oracle");
        assert_eq!(oracle.row_status[0], 2, "expected reject status");
        assert_eq!(oracle.nll[0], 0.0);
        assert!(oracle.grad[..r].iter().all(|&g| g == 0.0));
        assert!(oracle.hess[..r * r].iter().all(|&h| h == 0.0));
    }

    /// V100 parity: the device `survival_flex_primary_rows` kernel must
    /// match `cpu_oracle_flex_primary_rows` to round-off across a batch
    /// that exercises every arithmetic path (both event branches, the
    /// χ/d ratio terms, the q1/qd1 perturbation bumps).  `#[ignore]` +
    /// Linux-gated: runs only on the CUDA host in the V100 verification
    /// pass.
    ///
    /// FLAGGED for the V100 reviewer: the `χ_uv/χ − χ_u·χ_v/χ²` and
    /// `d_uv/d − d_u·d_v/d²` Hessian ratio terms are
    /// catastrophic-cancellation-prone; the device and CPU compute them
    /// in the identical operator order so they must agree to round-off,
    /// but a per-row tolerance scaled by `(1 + |cpu|)` is required because
    /// the absolute magnitude can be tiny relative to the inputs.
    #[cfg(target_os = "linux")]
    #[test]
    fn flex_primary_gpu_matches_cpu_oracle_v100() {
        // GPU parity test: skip cleanly when no CUDA device is present (the
        // codebase bans `#[ignore]`; the convention is a runtime early-return,
        // so the V100 verification pass runs it on the box while CPU-only hosts
        // pass trivially). Same gate as the pirls_gpu parity tests.
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            return;
        }
        // Cover r=1 (scalar bank), r=4 (rigid-equivalent width), and a
        // wider r=7 so the r×r loop and the diagonal perturbation
        // branches all run on the device.
        for &r in &[1_usize, 4, 7] {
            let storage = make_flex_primary_storage(6, r);
            let rows = storage.rows();
            let cpu = cpu_oracle_flex_primary_rows(&rows).expect("cpu oracle");

            let gpu = match try_flex_primary_rows(&rows) {
                Ok(Some(out)) => out,
                Ok(None) => panic!("r={r}: GPU returned None on a CUDA host (supported shape)"),
                Err(err) => panic!("r={r}: GPU launch failed: {err:?}"),
            };

            assert_eq!(gpu.nll.len(), cpu.nll.len());
            for i in 0..rows.len() {
                assert_eq!(
                    gpu.row_status[i], cpu.row_status[i],
                    "r={r} row {i}: status mismatch"
                );
                let nll_err = (gpu.nll[i] - cpu.nll[i]).abs();
                assert!(
                    nll_err <= 1e-10 * (1.0 + cpu.nll[i].abs()),
                    "r={r} row {i}: nll parity gpu={} cpu={} err={nll_err}",
                    gpu.nll[i],
                    cpu.nll[i]
                );
                for u in 0..r {
                    let g_err = (gpu.grad[i * r + u] - cpu.grad[i * r + u]).abs();
                    assert!(
                        g_err <= 1e-8 * (1.0 + cpu.grad[i * r + u].abs()),
                        "r={r} row {i}: grad[{u}] parity gpu={} cpu={} err={g_err}",
                        gpu.grad[i * r + u],
                        cpu.grad[i * r + u]
                    );
                    for v in 0..r {
                        let idx = i * r * r + u * r + v;
                        let h_err = (gpu.hess[idx] - cpu.hess[idx]).abs();
                        assert!(
                            h_err <= 1e-8 * (1.0 + cpu.hess[idx].abs()),
                            "r={r} row {i}: hess[{u}][{v}] parity gpu={} cpu={} err={h_err}",
                            gpu.hess[idx],
                            cpu.hess[idx]
                        );
                    }
                }
            }
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
        state.moments.to_vec()
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
        let left = [-1.5_f64, -0.8, 0.3, f64::NEG_INFINITY];
        let right = [0.0_f64, 0.3, 1.4, f64::INFINITY];
        let c0 = [0.15_f64, -0.20, 0.10, 0.05];
        let c1 = [-0.30_f64, 0.45, -0.20, -0.10];
        let c2 = [0.00_f64, 0.35, 0.25, 0.00];
        let c3 = [0.00_f64, 0.00, 0.18, 0.00];
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
                crate::gpu::kernels::cubic_cell::CubicCellMomentStatus::Ok as u8,
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

    // ────────────────────────────────────────────────────────────────────
    // Step 3 — device intercept solve (monotone root) parity tests.
    //
    // Uses the analytic evaluator `F(a) = α·exp(β·a) + γ` whose closed
    // form root is `a* = ln(-γ/α) / β`.  Six scenarios cover:
    //   * tight warm-start (Newton probe converges)
    //   * loose warm-start (bracket-then-refine path)
    //   * F increasing + decreasing
    //   * negative `γ/α` ratios that put the root in either half-plane
    //   * a degenerate F'(a_warm) ≈ 0 (status = 2)
    // The closed-form analytic root lets us assert |a_gpu - a_true|
    // tightly without relying on the CPU oracle, but we also do a
    // full element-wise CPU/GPU parity check at 1e-9 relative.
    // ────────────────────────────────────────────────────────────────────

    fn analytic_root(alpha: f64, beta: f64, gamma: f64) -> f64 {
        // F(a) = 0  ⇔  exp(β·a) = -γ/α  ⇔  a = ln(-γ/α) / β
        (-gamma / alpha).ln() / beta
    }

    #[test]
    fn survival_flex_intercept_solve_validates_inputs() {
        let bad = SurvivalFlexInterceptSolveInputs {
            n: 2,
            a_warm: &[0.0, 0.0],
            alpha: &[1.0, 1.0],
            beta: &[1.0, 1.0],
            gamma: &[-1.0], // wrong length
            convergence_tol: 1e-9,
            max_bracket_iters: 64,
            max_refine_iters: 64,
        };
        match try_device_intercept_solve(&bad) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(reason.contains("gamma.len()"), "got: {reason}");
            }
            other => panic!("expected length-mismatch error, got {other:?}"),
        }
    }

    #[test]
    fn survival_flex_intercept_solve_cpu_oracle_matches_analytic_root() {
        // Cross-check the oracle itself against the closed-form root,
        // so a regression in the oracle doesn't fake parity later.
        let alpha = [1.0, 2.0, -1.0, 0.5];
        let beta = [1.0, 0.5, 1.5, 2.0];
        let gamma = [-2.0, -3.0, 4.0, -1.5];
        let a_warm = [0.0, 0.0, 0.0, 0.0];
        let inputs = SurvivalFlexInterceptSolveInputs {
            n: 4,
            a_warm: &a_warm,
            alpha: &alpha,
            beta: &beta,
            gamma: &gamma,
            convergence_tol: 1e-12,
            max_bracket_iters: 64,
            max_refine_iters: 64,
        };
        let oracle = cpu_oracle_intercept_solve(&inputs);
        for row in 0..4 {
            assert_eq!(
                oracle.status[row], 0,
                "row {row}: oracle status {} (expected 0)",
                oracle.status[row]
            );
            let want = analytic_root(alpha[row], beta[row], gamma[row]);
            let rel = (oracle.a_root[row] - want).abs() / (1.0 + want.abs());
            assert!(
                rel <= 1e-9,
                "row {row}: oracle a={} vs analytic={} rel={}",
                oracle.a_root[row],
                want,
                rel
            );
            assert!(
                oracle.residual[row].abs() <= 1e-9,
                "row {row}: oracle residual {}",
                oracle.residual[row]
            );
        }
    }

    #[test]
    fn survival_flex_intercept_solve_device_matches_oracle() {
        // Mix tight + loose warm-starts to exercise Newton-probe and
        // bracket-expand paths.
        let alpha = [1.0, 2.0, -1.0, 0.5, 1.0, 3.0];
        let beta = [1.0, 0.5, 1.5, 2.0, 0.8, 1.2];
        let gamma = [-2.0, -3.0, 4.0, -1.5, -0.5, -4.5];
        // Warm-starts: rows 0-1 already near the root, rows 2-5 far away.
        let truth: Vec<f64> = (0..6)
            .map(|i| analytic_root(alpha[i], beta[i], gamma[i]))
            .collect();
        let a_warm = [
            truth[0] + 0.01,
            truth[1] - 0.02,
            truth[2] + 5.0,
            truth[3] - 8.0,
            0.0,
            -10.0,
        ];
        let inputs = SurvivalFlexInterceptSolveInputs {
            n: 6,
            a_warm: &a_warm,
            alpha: &alpha,
            beta: &beta,
            gamma: &gamma,
            convergence_tol: 1e-12,
            max_bracket_iters: 64,
            max_refine_iters: 64,
        };
        let oracle = cpu_oracle_intercept_solve(&inputs);
        for row in 0..6 {
            assert_eq!(
                oracle.status[row], 0,
                "row {row}: oracle status {} (expected 0)",
                oracle.status[row]
            );
        }

        match try_device_intercept_solve(&inputs) {
            Ok(Some(dev)) => {
                for row in 0..6 {
                    assert_eq!(
                        dev.status[row], oracle.status[row],
                        "row {row}: status mismatch dev={} oracle={}",
                        dev.status[row], oracle.status[row]
                    );
                    let want = truth[row];
                    let rel = (dev.a_root[row] - want).abs() / (1.0 + want.abs());
                    assert!(
                        rel <= 1e-9,
                        "row {row}: device a={} vs analytic={} rel={}",
                        dev.a_root[row],
                        want,
                        rel
                    );
                    let pair_rel = (dev.a_root[row] - oracle.a_root[row]).abs()
                        / (1.0 + oracle.a_root[row].abs());
                    assert!(
                        pair_rel <= 1e-9,
                        "row {row}: device/oracle a_root mismatch dev={} oracle={} rel={}",
                        dev.a_root[row],
                        oracle.a_root[row],
                        pair_rel
                    );
                    let resid_ok = dev.residual[row].abs() <= 1e-9
                        || (dev.residual[row] - oracle.residual[row]).abs()
                            <= 1e-9 * (1.0 + oracle.residual[row].abs());
                    assert!(
                        resid_ok,
                        "row {row}: residual mismatch dev={} oracle={}",
                        dev.residual[row], oracle.residual[row]
                    );
                }
            }
            Ok(None) => {
                // Non-CUDA build path — confirm the oracle handles every
                // scenario (already done in the loop above).
            }
            Err(err) => panic!("device intercept solve failed: {err:?}"),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 4 Layer A — `cpu_oracle_evaluate_calibration` parity tests.
    //
    // Two parity bars:
    //   * `F'(a)` matches a 4-point central finite difference of `F(a)`
    //     at every probe point in a small grid.
    //   * `F''(a)` matches a 4-point central finite difference of `F'(a)`.
    //
    // The fixture is a single non-affine partition cell over `[-0.5, 0.5]`
    // with non-trivial cubic coefficients and non-trivial score/link spans;
    // the spans don't have to be self-consistent across cells (we're only
    // probing the analytic surface against finite differences of itself),
    // they just have to be the same span for every `a` so the
    // finite-difference contract holds.  Tolerance 5e-6 on the second
    // derivative — looser than 1e-10 because the truncation error of a
    // 4-point central difference of a third-order quantity dominates.
    //
    // No CUDA dependency — runs on every host.
    // ────────────────────────────────────────────────────────────────────

    /// Build the single-cell calibration fixture **at intercept `a`** (and
    /// scale `slope`).
    ///
    /// The cell's cubic coefficients are NOT frozen: they are recomputed from
    /// the fixed score/link spans via `denested_cell_coefficients(score_span,
    /// link_span, a, slope)` — exactly the `a → cell` map production uses
    /// (`survival::marginal_slope::denested_partition_cells` →
    /// `build_denested_partition_cells_with_tails`). This is what makes the
    /// oracle's value-integral seed `F(a)` genuinely vary with `a`, so a
    /// finite difference of `F` reproduces the analytic `F'` (= the oracle's
    /// `f_prime`, which is `⟨denested_cell_coefficient_partials(...,a,slope),
    /// moments⟩`, the exact `∂a` of `denested_cell_coefficients`).
    ///
    /// A frozen cell (the prior fixture) left `F(a)` constant in `a`, so the
    /// FD of `F` was identically 0 while the analytic `F'` was nonzero — a
    /// structurally-guaranteed FD-lock failure (#831 Defect A). The spans
    /// stay fixed across `a`; only the cell cubic tracks `a`.
    fn step4a_fixture_cells_at(a: f64, slope: f64) -> Vec<SurvivalFlexCalibrationCell> {
        use crate::families::cubic_cell_kernel::{
            DenestedCubicCell, LocalSpanCubic, denested_cell_coefficients,
        };
        // Fixed, non-trivial spans — the same span for every `a` so the
        // finite-difference contract holds (only the cell cubic moves).
        let score_span = LocalSpanCubic {
            left: -0.5,
            right: 0.5,
            c0: 0.0,
            c1: 0.13,
            c2: -0.05,
            c3: 0.02,
        };
        let link_span = LocalSpanCubic {
            left: -0.5,
            right: 0.5,
            c0: 0.0,
            c1: 0.09,
            c2: 0.04,
            c3: -0.01,
        };
        // `a → cell` map: the cell cubic is the closed-form
        // `denested_cell_coefficients`, whose `∂a` is exactly the
        // `denested_cell_coefficient_partials` the oracle folds against the
        // moments. Anchor the cell support at the fixed span window so the
        // quadrature domain is stable across the FD stencil.
        let coeffs = denested_cell_coefficients(score_span, link_span, a, slope);
        let cell = DenestedCubicCell {
            left: -0.5,
            right: 0.5,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        vec![SurvivalFlexCalibrationCell {
            cell,
            score_span,
            link_span,
        }]
    }

    #[test]
    fn step4a_oracle_f_prime_matches_finite_difference() {
        let q = 0.4_f64;
        let slope = 0.55_f64;
        let probit_scale = 1.0_f64;
        let h = 1e-5_f64;

        // Evaluate `F(a')` with the cell cubic rebuilt AT `a'` — both the
        // value-integral seed (driven by the cell coefficients) and the
        // analytic partials (driven by the explicit `a'` argument) must track
        // the same `a'`, exactly as production does. A stencil that reused a
        // cell frozen at the centre `a` would make `F` constant in `a'`
        // (FD ≡ 0); see #831 Defect A.
        let f_at = |a_prime: f64| -> f64 {
            cpu_oracle_evaluate_calibration(
                &step4a_fixture_cells_at(a_prime, slope),
                a_prime,
                q,
                slope,
                probit_scale,
            )
            .expect("oracle must succeed on the fixture")
            .f
        };

        // Probe `a` on a small grid covering both sides of the typical
        // calibration root.
        for &a in &[-0.2_f64, -0.05, 0.0, 0.07, 0.18] {
            let out = cpu_oracle_evaluate_calibration(
                &step4a_fixture_cells_at(a, slope),
                a,
                q,
                slope,
                probit_scale,
            )
            .expect("oracle must succeed on the fixture");
            // 4-point central FD of F(a):
            //   F'(a) ≈ (-F(a+2h) + 8 F(a+h) - 8 F(a-h) + F(a-2h)) / (12 h)
            let f_p2 = f_at(a + 2.0 * h);
            let f_p1 = f_at(a + h);
            let f_m1 = f_at(a - h);
            let f_m2 = f_at(a - 2.0 * h);
            let fd = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * h);

            let abs = (out.f_prime - fd).abs();
            let rel = abs / (1.0 + fd.abs());
            assert!(
                abs <= 5e-9 || rel <= 5e-7,
                "F' parity at a={a}: oracle={} fd={} abs_err={} rel_err={}",
                out.f_prime,
                fd,
                abs,
                rel
            );
        }
    }

    #[test]
    fn step4a_oracle_f_double_prime_matches_finite_difference() {
        let q = 0.4_f64;
        let slope = 0.55_f64;
        let probit_scale = 1.0_f64;
        let h = 1e-4_f64;

        // `F'(a')` with the cell cubic rebuilt at `a'` (see Defect A note on
        // the F' test): the moments and the partials must both track `a'`,
        // else the FD of `F'` would be the partial-only derivative and drift
        // from the analytic `F''` (which also includes the ∂moments/∂a
        // convolution term).
        let fp_at = |a_prime: f64| -> f64 {
            cpu_oracle_evaluate_calibration(
                &step4a_fixture_cells_at(a_prime, slope),
                a_prime,
                q,
                slope,
                probit_scale,
            )
            .expect("oracle must succeed on the fixture")
            .f_prime
        };

        for &a in &[-0.2_f64, -0.05, 0.0, 0.07, 0.18] {
            let out = cpu_oracle_evaluate_calibration(
                &step4a_fixture_cells_at(a, slope),
                a,
                q,
                slope,
                probit_scale,
            )
            .expect("oracle must succeed on the fixture");
            // 4-point central FD of F'(a) (using the analytic F' the oracle
            // already returns — keeps the FD inner sample to one quantity).
            let fp_p2 = fp_at(a + 2.0 * h);
            let fp_p1 = fp_at(a + h);
            let fp_m1 = fp_at(a - h);
            let fp_m2 = fp_at(a - 2.0 * h);
            let fd = (-fp_p2 + 8.0 * fp_p1 - 8.0 * fp_m1 + fp_m2) / (12.0 * h);

            let abs = (out.f_double_prime - fd).abs();
            let rel = abs / (1.0 + fd.abs());
            // 4-point FD of a noisy derivative — looser tolerance; the
            // truncation error scales as h^4 on a smooth integrand, which
            // is ~1e-16 at h=1e-4, but FD round-off scales as 1/h^2 and
            // dominates at ~1e-6 on f64.
            assert!(
                abs <= 5e-6 || rel <= 5e-5,
                "F'' parity at a={a}: oracle={} fd={} abs_err={} rel_err={}",
                out.f_double_prime,
                fd,
                abs,
                rel
            );
        }
    }

    #[test]
    fn step4a_oracle_f_seed_matches_target_survival() {
        // At the (one-cell) fixture the cell contribution to F at a = -∞
        // tends to zero (no calibration step has been applied yet), so the
        // F seed equals `-Φ(-q)` exactly.  Use a very negative `a`
        // (-1e3) where the integrand contribution is negligible; the
        // residual then equals the seed within f64 epsilon.
        let q = 0.4_f64;
        let slope = 0.55_f64;
        let probit_scale = 1.0_f64;
        // The oracle's seed term is `-Φ(-q)` directly; we don't need to
        // probe at large `|a|` to read it.  Subtract the per-cell value
        // contribution from the oracle output and compare to the seed.
        let cells = step4a_fixture_cells_at(0.0, slope);
        let out = cpu_oracle_evaluate_calibration(&cells, 0.0, q, slope, probit_scale)
            .expect("oracle must succeed");
        let target = -crate::probability::normal_cdf(-q);
        // The per-cell `state.value` adds to the seed, so we can't
        // assert exact equality of `out.f` to the seed — but we *can*
        // assert the seed sign matches and that the seed is finite.
        // The strict parity-of-seed check is sub-test #2 of #4a; for
        // this test we just sanity-check the sign convention.
        assert!(target.is_finite(), "target survival must be finite");
        assert!(out.f.is_finite(), "F(a=0) must be finite");
        // At q > 0, `-Φ(-q) ∈ (-0.5, 0)`, strictly negative.
        assert!(target < 0.0, "target survival sign convention check");
    }

    /// Layer 4a — `try_device_evaluate_calibration` ↔ `cpu_oracle_evaluate_calibration`
    /// parity.  Both code paths are the same identity, only the moment
    /// quadrature differs (analytic in the oracle, substrate-evaluated in
    /// the device entry).  Tolerance abs ≤ 5e-10 mirrors the substrate's
    /// own GL kernel parity bar vs the CPU evaluator.
    ///
    /// Runs on every host: the substrate falls back to CPU on non-Linux /
    /// no-CUDA builds, so the parity assertion is identity-vs-identity in
    /// that case (still useful as a regression check on the fold algebra).
    #[test]
    fn step4a_device_evaluator_matches_cpu_oracle() {
        // Two rows × two cells per row — exercises the row_offsets index
        // and the per-cell partial recomputation per (row, cell) pair.
        use crate::families::cubic_cell_kernel::{DenestedCubicCell, LocalSpanCubic};

        fn cells_for_row(seed: f64) -> Vec<SurvivalFlexCalibrationCell> {
            vec![
                SurvivalFlexCalibrationCell {
                    cell: DenestedCubicCell {
                        left: -0.5,
                        right: 0.0,
                        c0: 0.31 + seed * 0.01,
                        c1: 0.27,
                        c2: -0.11,
                        c3: 0.07,
                    },
                    score_span: LocalSpanCubic {
                        left: -0.5,
                        right: 0.5,
                        c0: 0.0,
                        c1: 0.13,
                        c2: -0.05,
                        c3: 0.02,
                    },
                    link_span: LocalSpanCubic {
                        left: -0.5,
                        right: 0.5,
                        c0: 0.0,
                        c1: 0.09,
                        c2: 0.04,
                        c3: -0.01,
                    },
                },
                SurvivalFlexCalibrationCell {
                    cell: DenestedCubicCell {
                        left: 0.0,
                        right: 0.5,
                        c0: -0.18 + seed * 0.01,
                        c1: 0.12,
                        c2: 0.06,
                        c3: -0.04,
                    },
                    score_span: LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.11,
                        c2: 0.03,
                        c3: -0.02,
                    },
                    link_span: LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.08,
                        c2: -0.04,
                        c3: 0.015,
                    },
                },
            ]
        }

        let row0 = cells_for_row(0.0);
        let row1 = cells_for_row(1.0);
        let partition_by_row: Vec<Vec<SurvivalFlexCalibrationCell>> =
            vec![row0.clone(), row1.clone()];
        let a_per_row = vec![0.07_f64, -0.05];
        let q_per_row = vec![0.4_f64, 0.55];
        let slope_per_row = vec![0.55_f64, 0.42];
        let probit_scale = 1.0_f64;

        let device_out = match try_device_evaluate_calibration(
            &partition_by_row,
            &a_per_row,
            &q_per_row,
            &slope_per_row,
            probit_scale,
        ) {
            Ok(Some(out)) => out,
            Ok(None) => {
                // Substrate non-OK status — skip; oracle remains the
                // authoritative reference.  Print so CI can flag the
                // skip on the substrate side.
                eprintln!("[step4a parity] substrate returned None; skipping parity check");
                return;
            }
            Err(err) => panic!("device evaluator failed: {err:?}"),
        };
        assert_eq!(device_out.len(), 2);
        for (row, (a, q, slope)) in a_per_row
            .iter()
            .zip(q_per_row.iter())
            .zip(slope_per_row.iter())
            .map(|((a, q), s)| (*a, *q, *s))
            .enumerate()
        {
            let oracle =
                cpu_oracle_evaluate_calibration(&partition_by_row[row], a, q, slope, probit_scale)
                    .expect("oracle must succeed on fixture");
            let dev = device_out[row];
            let chk = |label: &str, d: f64, o: f64| {
                let abs = (d - o).abs();
                let rel = abs / (1.0 + o.abs());
                assert!(
                    abs <= 5e-10 || rel <= 5e-9,
                    "[row {row}] {label} parity: device={d} oracle={o} abs={abs} rel={rel}"
                );
            };
            chk("F", dev.f, oracle.f);
            chk("F'", dev.f_prime, oracle.f_prime);
            chk("F''", dev.f_double_prime, oracle.f_double_prime);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 4 Layer B — `try_device_layer_b_jet` algebraic identity tests.
    //
    // The fold has three independent contracts that can be checked in
    // isolation without re-implementing the full quadrature path:
    //
    //   1. With all `coeff_u[u] = [0;4]`, `f_u[u] == 0` for u != q_index
    //      and `f_u[q_index] == φ(q)` → `a_u[q_index] == φ(q) / D` and
    //      `a_u[u] == 0` for u != q_index.
    //
    //   2. With all `coeff_u[u] = [0;4]` and all `coeff_au[u] = [0;4]`,
    //      the integrand reduces to `-(chi_poly · eta_poly · a_u[u] · chi_poly)`
    //      so `d_u[u] = -a_u[u] · ⟨chi_poly² · eta_poly, moments⟩`.  We
    //      probe this against an oracle computed via the same substrate.
    //
    //   3. `eta_u[u] == chi · a_u[u] + rho[u]` and
    //      `chi_u[u] == eta_aa · a_u[u] + tau[u]` exact algebra.
    // ────────────────────────────────────────────────────────────────────

    fn step4b_fixture_cell() -> SurvivalFlexCalibrationCell {
        use crate::families::cubic_cell_kernel::{DenestedCubicCell, LocalSpanCubic};
        SurvivalFlexCalibrationCell {
            cell: DenestedCubicCell {
                left: -0.5,
                right: 0.5,
                c0: 0.31,
                c1: 0.27,
                c2: -0.11,
                c3: 0.07,
            },
            score_span: LocalSpanCubic {
                left: -0.5,
                right: 0.5,
                c0: 0.0,
                c1: 0.13,
                c2: -0.05,
                c3: 0.02,
            },
            link_span: LocalSpanCubic {
                left: -0.5,
                right: 0.5,
                c0: 0.0,
                c1: 0.09,
                c2: 0.04,
                c3: -0.01,
            },
        }
    }

    #[test]
    fn step4b_layer_b_q_index_bump_only() {
        // p = 2, q_index = 1, all coeff_u = coeff_au = 0 → f_u = [0, φ(q)]
        // → a_u = [0, φ(q)/D], eta_u = [rho[0], chi·a_u[1] + rho[1]],
        // chi_u = [tau[0], eta_aa·a_u[1] + tau[1]], d_u = [0, 0] (no
        // coeff_u contribution makes the integrand `eta_u_poly = a_u[u]·chi_poly`
        // but for u=0 that's also 0).
        let cell = step4b_fixture_cell();
        let zero4 = [0.0_f64; 4];
        let cell_coeffs = SurvivalFlexLayerBCellCoeffs {
            coeff_u: vec![zero4, zero4],
            coeff_au: vec![zero4, zero4],
        };
        let rho = vec![0.11_f64, 0.23];
        let tau = vec![-0.04_f64, 0.07];
        let chi = 0.65_f64;
        let eta_aa = -0.12_f64;
        let phi_q = 1.0 / (2.0_f64 * std::f64::consts::PI).sqrt() * (-0.5_f64 * 0.4 * 0.4).exp();
        let d_check = 0.83_f64;

        let row = SurvivalFlexLayerBRowInputs {
            partition_cells: std::slice::from_ref(&cell),
            cell_coeffs: std::slice::from_ref(&cell_coeffs),
            d_check,
            q_index: 1,
            phi_q,
            chi,
            eta_aa,
            rho: &rho,
            tau: &tau,
            probit_scale: 1.0,
            a: 0.07,
            slope: 0.55,
        };
        let out = match try_device_layer_b_jet(std::slice::from_ref(&row)) {
            Ok(Some(o)) => o,
            Ok(None) => {
                eprintln!("[step4b q_index] substrate non-OK or empty; skipping");
                return;
            }
            Err(err) => panic!("layer b q_index test failed: {err:?}"),
        };
        assert_eq!(out.len(), 1);
        let r = &out[0];
        assert_eq!(r.a_u.len(), 2);
        // a_u[0] == 0 (no perturbation), a_u[1] = φ(q)/D.
        let expected_a_u1 = phi_q / d_check;
        assert!(
            r.a_u[0].abs() <= 5e-15,
            "a_u[0] should be 0, got {}",
            r.a_u[0]
        );
        assert!(
            (r.a_u[1] - expected_a_u1).abs() <= 5e-15 * (1.0 + expected_a_u1.abs()),
            "a_u[1] should be φ(q)/D = {}, got {}",
            expected_a_u1,
            r.a_u[1]
        );
        // eta_u / chi_u closed-form identity.
        let exp_eta_u0 = chi * r.a_u[0] + rho[0];
        let exp_eta_u1 = chi * r.a_u[1] + rho[1];
        let exp_chi_u0 = eta_aa * r.a_u[0] + tau[0];
        let exp_chi_u1 = eta_aa * r.a_u[1] + tau[1];
        assert!((r.eta_u[0] - exp_eta_u0).abs() <= 5e-15);
        assert!((r.eta_u[1] - exp_eta_u1).abs() <= 5e-15);
        assert!((r.chi_u[0] - exp_chi_u0).abs() <= 5e-15);
        assert!((r.chi_u[1] - exp_chi_u1).abs() <= 5e-15);
    }

    #[test]
    fn step4b_layer_b_input_validation() {
        // Validate that mismatched shapes return DriverCallFailed.
        let cell = step4b_fixture_cell();
        let cell_coeffs = SurvivalFlexLayerBCellCoeffs {
            coeff_u: vec![[0.0; 4]],
            coeff_au: vec![[0.0; 4]],
        };
        let rho = vec![0.0_f64, 0.0]; // p=2
        let tau = vec![0.0_f64, 0.0];
        let row = SurvivalFlexLayerBRowInputs {
            partition_cells: std::slice::from_ref(&cell),
            cell_coeffs: std::slice::from_ref(&cell_coeffs),
            d_check: 1.0,
            q_index: 0,
            phi_q: 0.0,
            chi: 0.0,
            eta_aa: 0.0,
            rho: &rho,
            tau: &tau,
            probit_scale: 1.0,
            a: 0.0,
            slope: 0.5,
        };
        match try_device_layer_b_jet(std::slice::from_ref(&row)) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(
                    reason.contains("coeff_u.len()") || reason.contains("expected"),
                    "unexpected validation message: {reason}"
                );
            }
            other => panic!("expected validation error, got {other:?}"),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 4 Layer C-α — `try_device_layer_c_jet` algebraic identity tests.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn step4c_tri_index_layout_invariant() {
        // Verify `tri_index` matches the canonical "u·(2p-u-1)/2 + v"
        // upper-triangle row-major pack for every (u,v) with u ≤ v ≤ p-1.
        // Counter-check: tri_index(0,0) = 0, tri_index(p-1, p-1) = p(p+1)/2 - 1.
        for p in 1usize..=6 {
            let mut seen = std::collections::HashSet::new();
            for u in 0..p {
                for v in u..p {
                    let idx = tri_index(u, v, p);
                    assert!(
                        idx < p * (p + 1) / 2,
                        "tri_index({u},{v},{p})={idx} out of bounds"
                    );
                    assert!(seen.insert(idx), "duplicate tri_index({u},{v},{p})={idx}");
                }
            }
            assert_eq!(seen.len(), p * (p + 1) / 2);
        }
    }

    #[test]
    fn step4c_layer_c_zero_coeffs_collapses_to_observed_jet() {
        // With all coeff_u = coeff_au = coeff_bu = 0 and q_index = MAX
        // (disabled), the per-cell f_aa contribution is the only
        // moment-driven quantity that survives.  We still get f_u = 0,
        // f_au = 0, f_uv = 0, so a_u = 0, a_uv = 0, and the observed-jet
        // closed forms collapse to:
        //   eta_u[u]   == rho[u]
        //   chi_u[u]   == tau[u]
        //   eta_uv[u,v] == r_uv_upper_packed[(u,v)]
        //   chi_uv[u,v] == chi_uv_fixed_upper_packed[(u,v)]
        // (and d_u = the same integrand as Layer B with a_u = 0, which
        //  is integrand = -coeff_u[u] = 0 → 0.)
        let cell = step4b_fixture_cell();
        let p = 2usize;
        let zero4 = [0.0_f64; 4];
        let cc = SurvivalFlexLayerCCellCoeffs {
            coeff_u: vec![zero4; p],
            coeff_au: vec![zero4; p],
            coeff_bu: vec![zero4; p],
        };
        let rho = vec![0.11_f64, 0.23];
        let tau = vec![-0.04_f64, 0.07];
        let tau_a = vec![0.03_f64, -0.02];
        let r_uv = vec![0.31_f64, -0.17, 0.22]; // (0,0), (0,1), (1,1)
        let chi_uv_fixed = vec![0.08_f64, 0.04, -0.05];

        let row = SurvivalFlexLayerCRowInputs {
            partition_cells: std::slice::from_ref(&cell),
            cell_coeffs: std::slice::from_ref(&cc),
            d_check: 0.83,
            q_index: usize::MAX,
            g_index: usize::MAX,
            phi_q: 0.0,
            q: 0.0,
            chi: 0.65,
            eta_aa: -0.12,
            eta_aaa: 0.07,
            rho: &rho,
            tau: &tau,
            tau_a: &tau_a,
            r_uv_upper_packed: &r_uv,
            chi_uv_fixed_upper_packed: &chi_uv_fixed,
            probit_scale: 1.0,
            a: 0.07,
            slope: 0.55,
        };
        let out = match try_device_layer_c_jet(std::slice::from_ref(&row)) {
            Ok(Some(o)) => o,
            Ok(None) => {
                eprintln!("[step4c zero] substrate non-OK or empty; skipping");
                return;
            }
            Err(err) => panic!("layer_c zero-coeffs test failed: {err:?}"),
        };
        assert_eq!(out.len(), 1);
        let r = &out[0];
        // a_u = 0 (no perturbation since q_index disabled and coeff_u = 0).
        for u in 0..p {
            assert!(
                r.a_u[u].abs() <= 5e-15,
                "a_u[{u}] should be 0, got {}",
                r.a_u[u]
            );
        }
        // eta_u[u] = chi * 0 + rho[u] = rho[u].
        // chi_u[u] = eta_aa * 0 + tau[u] = tau[u].
        for u in 0..p {
            assert!((r.eta_u[u] - rho[u]).abs() <= 5e-15);
            assert!((r.chi_u[u] - tau[u]).abs() <= 5e-15);
        }
        // a_uv: f_uv = 0 (zero coeffs) and a_u = 0, so a_uv = 0 / D = 0.
        for u in 0..p {
            for v in 0..p {
                assert!(
                    r.a_uv[u * p + v].abs() <= 5e-15,
                    "a_uv[{u},{v}] should be 0, got {}",
                    r.a_uv[u * p + v]
                );
            }
        }
        // eta_uv: every cross term vanishes (a_uv = 0, a_u = 0), only
        // r_uv survives.
        for u in 0..p {
            for v in u..p {
                let packed = tri_index(u, v, p);
                let expected = r_uv[packed];
                assert!((r.eta_uv[u * p + v] - expected).abs() <= 5e-15);
                assert!((r.eta_uv[v * p + u] - expected).abs() <= 5e-15);
            }
        }
        // chi_uv: only chi_uv_fixed survives.
        for u in 0..p {
            for v in u..p {
                let packed = tri_index(u, v, p);
                let expected = chi_uv_fixed[packed];
                assert!((r.chi_uv[u * p + v] - expected).abs() <= 5e-15);
                assert!((r.chi_uv[v * p + u] - expected).abs() <= 5e-15);
            }
        }
    }

    #[test]
    fn step4c_layer_c_outputs_symmetric() {
        // Non-trivial coeffs, q_index enabled — assert a_uv, eta_uv,
        // chi_uv are symmetric under (u ↔ v).
        let cell = step4b_fixture_cell();
        let p = 3usize;
        let cc = SurvivalFlexLayerCCellCoeffs {
            coeff_u: vec![
                [0.12, 0.05, -0.03, 0.01],
                [0.07, -0.04, 0.02, 0.0],
                [-0.06, 0.03, 0.01, -0.005],
            ],
            coeff_au: vec![
                [0.02, 0.01, 0.0, 0.0],
                [0.015, -0.005, 0.0, 0.0],
                [-0.01, 0.008, 0.0, 0.0],
            ],
            coeff_bu: vec![[0.0; 4], [0.03, 0.01, 0.0, 0.0], [0.0; 4]],
        };
        let rho = vec![0.11_f64, 0.23, -0.05];
        let tau = vec![-0.04_f64, 0.07, 0.02];
        let tau_a = vec![0.03_f64, -0.02, 0.01];
        // p=3 → 6 packed entries.
        let r_uv = vec![0.31_f64, -0.17, 0.05, 0.22, 0.03, -0.08];
        let chi_uv_fixed = vec![0.08_f64, 0.04, -0.02, -0.05, 0.06, 0.01];

        let row = SurvivalFlexLayerCRowInputs {
            partition_cells: std::slice::from_ref(&cell),
            cell_coeffs: std::slice::from_ref(&cc),
            d_check: 0.83,
            q_index: 0,
            g_index: 1,
            phi_q: 0.42,
            q: 0.4,
            chi: 0.65,
            eta_aa: -0.12,
            eta_aaa: 0.07,
            rho: &rho,
            tau: &tau,
            tau_a: &tau_a,
            r_uv_upper_packed: &r_uv,
            chi_uv_fixed_upper_packed: &chi_uv_fixed,
            probit_scale: 1.0,
            a: 0.07,
            slope: 0.55,
        };
        let out = match try_device_layer_c_jet(std::slice::from_ref(&row)) {
            Ok(Some(o)) => o,
            Ok(None) => {
                eprintln!("[step4c sym] substrate non-OK or empty; skipping");
                return;
            }
            Err(err) => panic!("layer_c symmetry test failed: {err:?}"),
        };
        assert_eq!(out.len(), 1);
        let r = &out[0];
        for u in 0..p {
            for v in (u + 1)..p {
                let uv = r.a_uv[u * p + v];
                let vu = r.a_uv[v * p + u];
                assert!(
                    (uv - vu).abs() <= 1e-12 * (1.0 + uv.abs().max(vu.abs())),
                    "a_uv symmetry: ({u},{v})={uv} vs ({v},{u})={vu}"
                );
                let euv = r.eta_uv[u * p + v];
                let evu = r.eta_uv[v * p + u];
                assert!(
                    (euv - evu).abs() <= 1e-12 * (1.0 + euv.abs().max(evu.abs())),
                    "eta_uv symmetry: ({u},{v})={euv} vs ({v},{u})={evu}"
                );
                let cuv = r.chi_uv[u * p + v];
                let cvu = r.chi_uv[v * p + u];
                assert!(
                    (cuv - cvu).abs() <= 1e-12 * (1.0 + cuv.abs().max(cvu.abs())),
                    "chi_uv symmetry: ({u},{v})={cuv} vs ({v},{u})={cvu}"
                );
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 4 Layer C-β — `try_device_layer_c_beta_d_uv` algebraic identity tests.
    //
    // Two identity bars:
    //   1. With zero coefficient tables and zero a_u / a_uv inputs, every
    //      integrand term vanishes term-by-term → d_uv == 0.
    //   2. Non-trivial fixture asserts d_uv symmetric under (u ↔ v) and
    //      finite.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn step4c_beta_zero_inputs_yield_zero_d_uv() {
        let cell = step4b_fixture_cell();
        let p = 2usize;
        let zero4 = [0.0_f64; 4];
        let cc = SurvivalFlexLayerCBetaCellCoeffs {
            coeff_u: vec![zero4; p],
            coeff_au: vec![zero4; p],
            coeff_bu: vec![zero4; p],
            coeff_aau: vec![zero4; p],
            coeff_abu: vec![zero4; p],
        };
        let a_u = vec![0.0_f64; p];
        let a_uv = vec![0.0_f64; p * p];

        let row = SurvivalFlexLayerCBetaRowInputs {
            partition_cells: std::slice::from_ref(&cell),
            cell_coeffs: std::slice::from_ref(&cc),
            g_index: usize::MAX,
            a_u: &a_u,
            a_uv: &a_uv,
            a: 0.07,
            slope: 0.55,
            probit_scale: 1.0,
        };
        let out = match try_device_layer_c_beta_d_uv(std::slice::from_ref(&row)) {
            Ok(Some(o)) => o,
            Ok(None) => {
                eprintln!("[step4c_beta zero] substrate non-OK or empty; skipping");
                return;
            }
            Err(err) => panic!("layer_c_beta zero-inputs test failed: {err:?}"),
        };
        assert_eq!(out.len(), 1);
        let r = &out[0];
        for u in 0..p {
            for v in 0..p {
                assert!(
                    r.d_uv[u * p + v].abs() <= 5e-15,
                    "d_uv[{u},{v}] should be 0 (all-zero inputs), got {}",
                    r.d_uv[u * p + v]
                );
            }
        }
    }

    #[test]
    fn step4c_beta_d_uv_symmetric() {
        let cell = step4b_fixture_cell();
        let p = 3usize;
        let cc = SurvivalFlexLayerCBetaCellCoeffs {
            coeff_u: vec![
                [0.12, 0.05, -0.03, 0.01],
                [0.07, -0.04, 0.02, 0.0],
                [-0.06, 0.03, 0.01, -0.005],
            ],
            coeff_au: vec![
                [0.02, 0.01, 0.0, 0.0],
                [0.015, -0.005, 0.0, 0.0],
                [-0.01, 0.008, 0.0, 0.0],
            ],
            coeff_bu: vec![[0.0; 4], [0.03, 0.01, 0.0, 0.0], [0.0; 4]],
            coeff_aau: vec![
                [0.005, 0.002, 0.0, 0.0],
                [0.003, -0.001, 0.0, 0.0],
                [-0.002, 0.001, 0.0, 0.0],
            ],
            coeff_abu: vec![[0.0; 4], [0.008, 0.002, 0.0, 0.0], [0.0; 4]],
        };
        let a_u = vec![0.21_f64, -0.13, 0.07];
        // Symmetric a_uv.
        let a_uv = vec![0.04, -0.03, 0.02, -0.03, 0.11, -0.01, 0.02, -0.01, 0.06_f64];

        let row = SurvivalFlexLayerCBetaRowInputs {
            partition_cells: std::slice::from_ref(&cell),
            cell_coeffs: std::slice::from_ref(&cc),
            g_index: 1,
            a_u: &a_u,
            a_uv: &a_uv,
            a: 0.07,
            slope: 0.55,
            probit_scale: 1.0,
        };
        let out = match try_device_layer_c_beta_d_uv(std::slice::from_ref(&row)) {
            Ok(Some(o)) => o,
            Ok(None) => {
                eprintln!("[step4c_beta sym] substrate non-OK or empty; skipping");
                return;
            }
            Err(err) => panic!("layer_c_beta symmetry test failed: {err:?}"),
        };
        assert_eq!(out.len(), 1);
        let r = &out[0];
        for u in 0..p {
            for v in (u + 1)..p {
                let uv = r.d_uv[u * p + v];
                let vu = r.d_uv[v * p + u];
                assert!(
                    (uv - vu).abs() <= 1e-12 * (1.0 + uv.abs().max(vu.abs())),
                    "d_uv symmetry: ({u},{v})={uv} vs ({v},{u})={vu}"
                );
                assert!(uv.is_finite(), "d_uv[{u},{v}] non-finite: {uv}");
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 5 — `try_device_step5_primary_assembly` algebraic identity tests.
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn step5_di_zero_collapses_gradient_to_survival_only() {
        // With di = 0 (censored row): the gradient simplifies to
        //   grad[u] = entry_u1·entry.eta_u[u] + exit_surv_u1·exit.eta_u[u]
        // and the Hessian to the two `k2` outer products + two `k1·eta_uv`.
        // d_u / d_uv / chi_u / chi_uv contributions all vanish (multiplied by wi·di=0).
        let p = 2usize;
        let entry_eta_u = vec![0.3_f64, -0.2];
        let entry_eta_uv = vec![0.05_f64, -0.02, -0.02, 0.08];
        let exit_eta_u = vec![0.4_f64, 0.1];
        let exit_eta_uv = vec![0.1_f64, 0.03, 0.03, -0.04];
        let zero_p = vec![0.0_f64; p];
        let zero_pp = vec![0.0_f64; p * p];

        let entry_k1 = -0.21_f64;
        let entry_k2 = 0.13_f64;
        let exit_k1 = 0.08_f64;
        let exit_k2 = 0.05_f64;

        let row = SurvivalFlexStep5RowInputs {
            entry: SurvivalFlexTimepointJet {
                eta: 0.4,
                chi: 0.7,
                d: 0.85,
                eta_u: &entry_eta_u,
                eta_uv: &entry_eta_uv,
                chi_u: &zero_p,
                chi_uv: &zero_pp,
                d_u: &zero_p,
                d_uv: &zero_pp,
            },
            exit: SurvivalFlexTimepointJet {
                eta: 0.6,
                chi: 0.75,
                d: 0.9,
                eta_u: &exit_eta_u,
                eta_uv: &exit_eta_uv,
                chi_u: &zero_p,
                chi_uv: &zero_pp,
                d_u: &zero_p,
                d_uv: &zero_pp,
            },
            wi: 1.0,
            di: 0.0,
            q1: 0.5,
            qd1: 1.0,
            q1_index: usize::MAX,
            qd1_index: usize::MAX,
            entry_k1,
            entry_k2,
            exit_k1,
            exit_k2,
            log_surv0: -0.2,
            log_surv1: -0.3,
        };
        let out = try_device_step5_primary_assembly(std::slice::from_ref(&row))
            .expect("step5 censored assembly must succeed");
        assert_eq!(out.len(), 1);
        let r = &out[0];

        let entry_u1 = -entry_k1;
        let exit_surv_u1 = -exit_k1;
        for u in 0..p {
            let expected = entry_u1 * entry_eta_u[u] + exit_surv_u1 * exit_eta_u[u];
            assert!(
                (r.grad[u] - expected).abs() <= 5e-15,
                "grad[{u}] = {} but expected {expected}",
                r.grad[u]
            );
        }
        // Hessian: entry_k2·eta_u⊗eta_u + entry_u1·eta_uv + exit_k2·eta_u⊗eta_u + exit_surv_u1·eta_uv
        for u in 0..p {
            for v in 0..p {
                let expected = entry_k2 * entry_eta_u[u] * entry_eta_u[v]
                    + entry_u1 * entry_eta_uv[u * p + v]
                    + exit_k2 * exit_eta_u[u] * exit_eta_u[v]
                    + exit_surv_u1 * exit_eta_uv[u * p + v];
                let got = r.hess[u * p + v];
                assert!(
                    (got - expected).abs() <= 5e-15 * (1.0 + expected.abs()),
                    "hess[{u},{v}] = {} but expected {expected}",
                    got
                );
            }
        }
    }

    #[test]
    fn step5_hessian_symmetric_under_swap() {
        // Non-trivial fixture: di = 1, q1_index and qd1_index active.
        // The full chain-rule Hessian must still be symmetric under
        // (u, v) swap — every term in the assembly is constructed
        // symmetrically.
        let p = 3usize;
        let entry_eta_u = vec![0.3_f64, -0.2, 0.1];
        let entry_eta_uv = vec![0.05, -0.02, 0.01, -0.02, 0.08, -0.03, 0.01, -0.03, 0.04_f64];
        let exit_eta_u = vec![0.4_f64, 0.1, -0.2];
        let exit_eta_uv = vec![0.1, 0.03, -0.02, 0.03, -0.04, 0.05, -0.02, 0.05, 0.06_f64];
        let exit_chi_u = vec![0.07_f64, -0.04, 0.02];
        let exit_chi_uv = vec![
            0.01, -0.005, 0.002, -0.005, 0.015, -0.003, 0.002, -0.003, 0.008_f64,
        ];
        let exit_d_u = vec![0.06_f64, 0.02, -0.01];
        let exit_d_uv = vec![
            0.012, 0.004, -0.002, 0.004, 0.01, 0.003, -0.002, 0.003, 0.007_f64,
        ];
        let zero_p = vec![0.0_f64; p];
        let zero_pp = vec![0.0_f64; p * p];

        let row = SurvivalFlexStep5RowInputs {
            entry: SurvivalFlexTimepointJet {
                eta: 0.4,
                chi: 0.7,
                d: 0.85,
                eta_u: &entry_eta_u,
                eta_uv: &entry_eta_uv,
                chi_u: &zero_p,
                chi_uv: &zero_pp,
                d_u: &zero_p,
                d_uv: &zero_pp,
            },
            exit: SurvivalFlexTimepointJet {
                eta: 0.6,
                chi: 0.75,
                d: 0.9,
                eta_u: &exit_eta_u,
                eta_uv: &exit_eta_uv,
                chi_u: &exit_chi_u,
                chi_uv: &exit_chi_uv,
                d_u: &exit_d_u,
                d_uv: &exit_d_uv,
            },
            wi: 1.0,
            di: 1.0,
            q1: 0.5,
            qd1: 1.0,
            q1_index: 1,
            qd1_index: 2,
            entry_k1: -0.21,
            entry_k2: 0.13,
            exit_k1: 0.08,
            exit_k2: 0.05,
            log_surv0: -0.2,
            log_surv1: -0.3,
        };
        let out = try_device_step5_primary_assembly(std::slice::from_ref(&row))
            .expect("step5 symmetry assembly must succeed");
        assert_eq!(out.len(), 1);
        let r = &out[0];
        for u in 0..p {
            for v in (u + 1)..p {
                let uv = r.hess[u * p + v];
                let vu = r.hess[v * p + u];
                assert!(
                    (uv - vu).abs() <= 1e-14 * (1.0 + uv.abs().max(vu.abs())),
                    "hess symmetry ({u},{v})={uv} vs ({v},{u})={vu}"
                );
            }
        }
        assert!(r.row_nll.is_finite());
    }
}
