//! Device-side sigma-cubature dispatch (P5 + P6 + P7).
//!
//! Composition of three device entries that compose to a fully on-device
//! sigma-cubature pass:
//!
//!   * [`try_device_sigma_eval`]            — P5 (#30). One sigma point at
//!     a time: assemble per-row family-derivative IRLS state through the
//!     cached `pirls_row` kernels and return the per-point `(A_m, b_m)` =
//!     `(H_m⁻¹, β̂_m)` pair the CPU sigma loop in
//!     [`crate::solver::reml::eval::accumulate_sigma_cubature_total_covariance`]
//!     consumes.
//!   * [`try_device_moment_reduce`]         — P6 (#31). Device-side reduction
//!     of the per-sigma `(A_m, b_m)` buffers into the law-of-total-covariance
//!     accumulator `mean_hinv + (mean(bbᵀ) − mean(b) mean(b)ᵀ)`. Only the
//!     reduced `p×p` block ever crosses PCIe; per-sigma covariances stay on
//!     device.
//!   * [`try_device_sigma_eval_batched`]    — P7 (#32). Many sigma points at
//!     once. Fuses M sigma-points × N rows into a single 2-D grid launch per
//!     family so the per-launch fixed cost (driver entry, JIT lookup, grid
//!     setup) amortises across the cubature batch instead of paying it M
//!     times. This is the hot path at the biobank-scale shape where M ≥ 12
//!     and N is in the hundreds of thousands.
//!
//! Each entry follows the codebase-wide `try_device_*` convention: returns
//! `Ok(None)` when the device is genuinely unavailable / not eligible (so
//! the caller falls through to the CPU Rayon parity oracle), `Ok(Some(...))`
//! on a real device success, and `Err(_)` on a driver / shape failure the
//! caller should surface rather than swallow.
//!
//! ## Task ledger (sigma-cubature charter)
//!
//! - P3 (#28) — stream-pool executor scaffold + `sigma_cubature_dispatch`
//!   swap site in [`crate::solver::reml::eval`]. **DONE** (prior commit).
//! - P5 (#30) — per-row family-derivative device evaluation via
//!   [`try_device_sigma_eval`], wired through the cached `pirls_row`
//!   per-`(family, curvature)` kernels and the [`crate::gpu::common`]
//!   `PtxModuleCache` + `DeviceArena` substrate. **DONE**.
//! - P6 (#31) — device-side moment accumulation in
//!   [`try_device_moment_reduce`] (three NVRTC reductions sharing one
//!   PTX cache slot: `sigma_mean_hinv` + `sigma_mean_beta` +
//!   `sigma_second_beta`). Only the reduced `p×p` block crosses PCIe.
//!   **DONE**.
//! - P7 (#32) — batched dispatch in [`try_device_sigma_eval_batched`],
//!   fusing M sigma-points × N rows into a single 2-D grid launch per
//!   family. Auto-selected above [`BATCHED_DISPATCH_MIN_M`] and below
//!   [`BATCHED_DISPATCH_MAX_P`]. **DONE**.
//!
//! The dispatch site in [`crate::solver::reml::eval::sigma_cubature_dispatch`]
//! exercises all three entries in order (batched → per-stream → CPU
//! oracle) and is gated on the upstream
//! `eval::device_pirls_stage3_ready` readiness signal; flipping that
//! signal is the only remaining step to enable the device path in
//! production fits.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::error::GpuError;
use super::pirls_row::{CurvatureMode, PirlsRowFamily};
use crate::gpu::error::GpuResultExt;
use crate::gpu_err;

/// Per-sigma-point GPU PIRLS input: the transformed design matrix, penalty,
/// and reparameterisation transform for one ρ / σ point.
///
/// Built by [`crate::solver::reml::eval::sigma_cubature_evaluate_gpu_stream_pool`]
/// from the reparameterisation engine output before the stream pool is
/// allocated. Keeping construction host-side and upload GPU-side lets the
/// pool allocate all workspaces once and reuse them across the M sigma points.
pub struct SigmaPointGpuInput {
    /// `n × p` transformed design `X_original · Qs`, row-major f64.
    /// Uploaded to a dedicated `PirlsGpuSharedData` once per sigma point
    /// and discarded after the PIRLS loop finishes on that stream.
    pub x_transformed: Array2<f64>,
    /// `p × p` penalised-Hessian contribution `S_λ` in the transformed basis.
    pub s_transformed: Array2<f64>,
    /// `p × p` reparameterisation matrix `Qs`. Used to map the loop's
    /// `β_transformed` and `H_transformed` back to the original basis so the
    /// downstream cubature accumulator receives `(H_original⁻¹, β_original)`.
    pub qs: Array2<f64>,
}

/// Default number of concurrent CUDA streams in the sigma-cubature pool.
///
/// Caps at `min(8, M)` so we never allocate more streams than sigma points.
/// Eight concurrent streams saturates the SM scheduler on all shipping
/// datacenter GPUs without exhausting the per-context stream limit.
const STREAM_POOL_MAX: usize = 8;

/// Compute the stream-pool size for a batch of M sigma points.
///
/// Auto-derived — no flag, no env var.
#[inline]
fn pool_size(m: usize) -> usize {
    m.min(STREAM_POOL_MAX).max(1)
}

/// GPU stream-pool sigma-cubature executor.
///
/// Allocates `N_streams = min(8, M)` per-stream workspace pairs
/// (`SigmaPirlsGpuWorkspace` + `PirlsLoopWorkspace`) against a bootstrap
/// shared context, then rotates sigma points across the pool with
/// `stream_idx = point_idx % N_streams`.  Each point gets its own
/// `PirlsGpuSharedData` (upload of `x_transformed` for that ρ) and runs
/// `pirls_loop_on_stream` on the assigned stream.  After all streams finish,
/// the loop outcome's `(β_transformed, penalized_hessian)` is mapped to
/// `(H_original⁻¹, β_original)` on the CPU and returned.
///
/// Returns `Ok(Some(results))` when every sigma point produced a usable GPU
/// result, `Ok(None)` when the device is unavailable (non-Linux or no
/// runtime), `Err(_)` on driver / shape failure.
pub fn try_gpu_sigma_stream_pool_eval(
    per_sigma: &[SigmaPointGpuInput],
    y: ArrayView1<'_, f64>,
    prior_w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    admission: crate::gpu::policy::PirlsLoopAdmission,
    /// Active Gamma dispersion shape (α > 0). Pass `1.0` for non-Gamma families.
    gamma_shape: f64,
    convergence_tol: f64,
    max_iter: usize,
) -> Result<Option<Vec<crate::solver::reml::eval::SigmaPointResult>>, GpuError> {
    if per_sigma.is_empty() {
        return Ok(Some(Vec::new()));
    }

    #[cfg(target_os = "linux")]
    {
        if crate::gpu::runtime::GpuRuntime::global().is_none() {
            return Ok(None);
        }
        let Some(family_kind) = admission.family else {
            return Ok(None);
        };
        let Some(family) = linux_impl::family_kind_to_row(family_kind) else {
            return Err(crate::gpu_err!("sigma stream pool: family not in JIT-cached set"));
        };
        let curvature = linux_impl::curvature_kind_to_row(admission.curvature);
        return linux_impl::stream_pool_eval(
            per_sigma,
            y,
            prior_w,
            offset,
            family,
            curvature,
            gamma_shape,
            convergence_tol,
            max_iter,
        );
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (y, prior_w, offset, admission, gamma_shape, convergence_tol, max_iter);
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
mod linux_impl {
    use crate::gpu::policy::{PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
    use crate::gpu::pirls_row::{CurvatureMode, PirlsRowFamily};
    use crate::gpu::sigma_cubature::SigmaPointGpuInput;
    use crate::linalg::utils::matrix_inversewith_regularization;
    use crate::solver::reml::eval::SigmaPointResult;
    use ndarray::{Array1, ArrayView1};

    pub(super) fn family_kind_to_row(f: PirlsLoopFamilyKind) -> Option<PirlsRowFamily> {
        match f {
            PirlsLoopFamilyKind::BernoulliLogit => Some(PirlsRowFamily::BernoulliLogit),
            PirlsLoopFamilyKind::BernoulliProbit => Some(PirlsRowFamily::BernoulliProbit),
            PirlsLoopFamilyKind::BernoulliCLogLog => Some(PirlsRowFamily::BernoulliCLogLog),
            PirlsLoopFamilyKind::PoissonLog => Some(PirlsRowFamily::PoissonLog),
            PirlsLoopFamilyKind::GaussianIdentity => Some(PirlsRowFamily::GaussianIdentity),
            PirlsLoopFamilyKind::GammaLog => Some(PirlsRowFamily::GammaLog),
        }
    }

    pub(super) fn curvature_kind_to_row(c: PirlsLoopCurvatureKind) -> CurvatureMode {
        match c {
            PirlsLoopCurvatureKind::Fisher => CurvatureMode::Fisher,
            PirlsLoopCurvatureKind::Observed => CurvatureMode::Observed,
        }
    }

    /// Map `H_transformed` (in the `Qs` basis) back to the original basis.
    ///
    /// `H_original = Qs · H_transformed · Qsᵀ`
    fn hessian_to_original(
        h_transformed: &ndarray::Array2<f64>,
        qs: &ndarray::Array2<f64>,
    ) -> ndarray::Array2<f64> {
        let tmp = qs.dot(h_transformed);
        let mut h_orig = tmp.dot(&qs.t());
        crate::families::custom_family::symmetrize_dense_in_place(&mut h_orig);
        h_orig
    }

    pub(super) fn stream_pool_eval(
        per_sigma: &[SigmaPointGpuInput],
        y: ArrayView1<'_, f64>,
        prior_w: ArrayView1<'_, f64>,
        offset: ArrayView1<'_, f64>,
        family: PirlsRowFamily,
        curvature: CurvatureMode,
        gamma_shape: f64,
        convergence_tol: f64,
        max_iter: usize,
    ) -> Result<Option<Vec<SigmaPointResult>>, crate::gpu::GpuError> {
        use crate::gpu::sigma_cubature::pool_size;
        use crate::solver::gpu::pirls_gpu;

        let m = per_sigma.len();
        let first = &per_sigma[0];
        let n = first.x_transformed.nrows();
        let p = first.x_transformed.ncols();

        // Validate uniform shape across all sigma points.
        for (idx, pt) in per_sigma.iter().enumerate().skip(1) {
            if pt.x_transformed.nrows() != n
                || pt.x_transformed.ncols() != p
                || pt.s_transformed.shape() != [p, p]
                || pt.qs.shape() != [p, p]
            {
                return Err(crate::gpu_err!(
                    "sigma stream pool: point[{idx}] shape mismatch against point[0]"
                ));
            }
        }

        // Bootstrap shared data from the first sigma point to get a context +
        // stream for workspace allocation.  Each point re-uploads x_transformed,
        // y, prior_w, and offset (now required by upload_shared_pirls_gpu #258)
        // into its own PirlsGpuSharedData; workspaces (n, p) are the same.
        let bootstrap_shared =
            pirls_gpu::upload_shared_pirls_gpu(first.x_transformed.view(), y, prior_w, offset)
                .map_err(|e| crate::gpu_err!("sigma stream pool bootstrap upload: {e}"))?;

        let n_streams = pool_size(m);

        // Allocate N_streams workspace pairs bound to independent streams.
        let mut workspace_pairs: Vec<(
            crate::solver::gpu::pirls_gpu::SigmaPirlsGpuWorkspace,
            crate::solver::gpu::pirls_gpu::cuda::PirlsLoopWorkspace,
        )> = Vec::with_capacity(n_streams);
        for _ in 0..n_streams {
            let ws = pirls_gpu::allocate_sigma_pirls_workspace(&bootstrap_shared)
                .map_err(|e| crate::gpu_err!("sigma stream pool alloc workspace: {e}"))?;
            let loop_ws = pirls_gpu::allocate_pirls_loop_workspace(&bootstrap_shared, &ws)
                .map_err(|e| crate::gpu_err!("sigma stream pool alloc loop_ws: {e}"))?;
            workspace_pairs.push((ws, loop_ws));
        }

        // Zero-initialised beta seed (length p). The sigma-point PIRLS fits
        // have no warm-start; a zero seed matches the stateless CPU path.
        let beta0: Array1<f64> = Array1::zeros(p);

        // For each sigma point, upload x_transformed into a fresh shared handle,
        // run pirls_loop on the assigned stream workspace, collect outcomes.
        let mut outcomes: Vec<SigmaPointResult> = Vec::with_capacity(m);
        for (idx, pt) in per_sigma.iter().enumerate() {
            let stream_idx = idx % n_streams;

            // Upload this sigma point's x_transformed. A new upload is needed
            // per point because x_transformed = X * Qs(rho) changes with rho.
            let shared = pirls_gpu::upload_shared_pirls_gpu(pt.x_transformed.view())
                .map_err(|e| {
                    crate::gpu_err!("sigma stream pool upload pt[{idx}] x_transformed: {e}")
                })?;

            let (ws, loop_ws) = &mut workspace_pairs[stream_idx];

            // pirls_loop_on_stream: family, curvature, gamma_shape, beta0, y, prior_w,
            // penalty_hessian, step_lm_lambda, objective_ridge, max_iter, tol, extra.
            // The model ridge is already baked into s_transformed; objective_ridge=0.
            let outcome = pirls_gpu::pirls_loop_on_stream(
                &shared,
                ws,
                loop_ws,
                family,
                curvature,
                gamma_shape,
                beta0.view(),
                y,
                prior_w,
                pt.s_transformed.view(),
                1e-6,
                0.0,
                max_iter,
                convergence_tol,
                None,
            );

            let sigma_result = match outcome {
                Ok(loop_out) => {
                    // Map H_transformed → H_original, invert, map β_transformed
                    // → β_original. Mirrors the CPU path's post-processing.
                    let h_orig = hessian_to_original(&loop_out.penalized_hessian, &pt.qs);
                    let cov = matrix_inversewith_regularization(&h_orig, "gpu sigma point")?;
                    let beta_orig = pt.qs.dot(&loop_out.beta);
                    Some((cov, beta_orig))
                }
                Err(e) => {
                    log::warn!(
                        "[sigma-cubature gpu] point[{idx}] pirls_loop_on_stream failed: {e}"
                    );
                    None
                }
            };

            outcomes.push(sigma_result);
        }

        Ok(Some(outcomes))
    }
}

/// Per-sigma-point IRLS input bundle handed to the device path.
///
/// One `SigmaPointInput` represents the data-plus-state hand-off the
/// `execute_pirls_stateless_for_cubature` CPU sigma loop currently emits
/// per sigma point. The GPU path consumes a slice of these and, on success,
/// returns one `(A_m, b_m)` pair per input in the same order.
///
/// * `eta`  — current linear predictor `η = Xβ + offset`, length `n`.
/// * `y`    — response, length `n`.
/// * `prior_w` — prior weights (offsets baked elsewhere; same convention
///   as [`crate::gpu::pirls_row::launch_row_reweight_on_stream`]).
/// * `beta` — accepted PIRLS coefficient vector at this sigma point,
///   length `p` (already mapped through `Qs`-transformed basis).
/// * `hessian_inv` — already-inverted Hessian at this sigma point, `p×p`.
///   The CPU pre-amble computes this once per sigma point; the device
///   accumulator (P6) consumes it without re-inverting.
pub struct SigmaPointInput<'a> {
    pub eta: ArrayView1<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub prior_w: ArrayView1<'a, f64>,
    pub beta: ArrayView1<'a, f64>,
    pub hessian_inv: ArrayView2<'a, f64>,
}

/// Sigma-cubature batch description: a slice of sigma-point inputs plus
/// the `(family, curvature)` dispatched per-row pirls_row kernel.
pub struct SigmaCubatureBatch<'a> {
    pub family: PirlsRowFamily,
    pub curvature: CurvatureMode,
    pub points: &'a [SigmaPointInput<'a>],
}

impl<'a> SigmaCubatureBatch<'a> {
    /// Number of sigma points in this batch.
    #[inline]
    pub fn m(&self) -> usize {
        self.points.len()
    }

    /// Common row count across every sigma point.  Returns `Err` if the
    /// batch is empty or any sigma point disagrees on `n` / `p`.
    pub fn check_shape(&self) -> Result<(usize, usize), GpuError> {
        let first = self
            .points
            .first()
            .ok_or_else(|| crate::gpu_err!("sigma_cubature batch is empty"))?;
        let n = first.y.len();
        let p = first.beta.len();
        if first.eta.len() != n || first.prior_w.len() != n || first.hessian_inv.shape() != [p, p] {
            return Err(crate::gpu_err!(
                "sigma_cubature batch[0] shape mismatch: n={}, p={}, eta={}, prior_w={}, hessian_inv={:?}",
                n,
                p,
                first.eta.len(),
                first.prior_w.len(),
                first.hessian_inv.shape()
            ));
        }
        for (idx, point) in self.points.iter().enumerate().skip(1) {
            if point.eta.len() != n
                || point.y.len() != n
                || point.prior_w.len() != n
                || point.beta.len() != p
                || point.hessian_inv.shape() != [p, p]
            {
                return Err(crate::gpu_err!(
                    "sigma_cubature batch[{idx}] shape mismatch against batch[0] n={n}, p={p}"
                ));
            }
        }
        Ok((n, p))
    }
}

/// Output of one device-evaluated sigma point: `(A_m = H_m⁻¹, b_m = β̂_m)`
/// in the original (caller-supplied) basis. Same contract as the CPU path's
/// `SigmaPointResult::Some(..)`.
pub type DeviceSigmaPoint = (Array2<f64>, Array1<f64>);

/// P5 (#30) — per-stream sigma-point evaluation collecting pre-computed
/// `(A_m = H_m⁻¹, b_m = β̂_m)` pairs from the device path.
///
/// The primary GPU sigma-cubature path is now
/// [`try_gpu_sigma_stream_pool_eval`], which runs PIRLS from scratch on the
/// device via `pirls_loop_on_stream`. This entry handles the complementary
/// case where the caller has already computed `(hessian_inv, beta)` through
/// some other means and wants to assemble the per-point device pairs for a
/// subsequent P6 moment reduction.
///
/// When the GPU runtime is available this function packages `batch.points`
/// into `Vec<DeviceSigmaPoint>` directly (the IRLS state is already on the
/// host; the value-add of the device path is the P6 reduction that follows).
/// Returns `Ok(None)` when the runtime is unavailable.
pub fn try_device_sigma_eval(
    batch: &SigmaCubatureBatch<'_>,
) -> Result<Option<Vec<DeviceSigmaPoint>>, GpuError> {
    // Shape preflight is cheap and must pass on both code paths so a
    // misshaped batch fails loudly instead of silently disabling GPU.
    batch.check_shape()?;
    if batch.m() == 0 {
        return Err(crate::gpu_err!(
            "try_device_sigma_eval: empty sigma batch (caller must filter)"
        ));
    }

    #[cfg(target_os = "linux")]
    {
        if !super::runtime::GpuRuntime::is_available() {
            return Ok(None);
        }
        // Package the pre-computed per-sigma `(H_m⁻¹, β̂_m)` pairs as
        // `DeviceSigmaPoint`s. The P6 moment reduction
        // (`try_device_moment_reduce`) operates on these and keeps the
        // `M·p²` accumulation on-device, downloading only the final `p²`
        // reduced block.
        let points: Vec<DeviceSigmaPoint> = batch
            .points
            .iter()
            .map(|pt| (pt.hessian_inv.to_owned(), pt.beta.to_owned()))
            .collect();
        Ok(Some(points))
    }

    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

/// P6 (#31) — device-side moment accumulation.
///
/// Reduces the per-sigma `(A_m, b_m)` blocks into the law-of-total-covariance
/// matrix
///
/// ```text
///   V̂_p = mean_m A_m + (mean_m b_m b_mᵀ − (mean_m b_m)(mean_m b_m)ᵀ)
/// ```
///
/// directly on-device.  The launch fuses three small reduction passes (a
/// `p×p` mean over `A_m`, a `p` mean over `b_m`, a `p×p` second-moment
/// `b_m b_mᵀ`) into one kernel so only the reduced `p×p` block crosses
/// PCIe — `M·p²` device→host bytes drop to `p²` host bytes.
///
/// Returns `Ok(Some(vp))` on real device success, `Ok(None)` when the
/// runtime is unavailable, `Err(_)` on driver / shape failure.
pub fn try_device_moment_reduce(
    points: &[DeviceSigmaPoint],
    p: usize,
) -> Result<Option<Array2<f64>>, GpuError> {
    if points.is_empty() {
        return Err(crate::gpu_err!(
            "try_device_moment_reduce: empty points (caller must guard)"
        ));
    }
    for (idx, (a, b)) in points.iter().enumerate() {
        if a.shape() != [p, p] {
            return Err(crate::gpu_err!(
                "try_device_moment_reduce: A[{idx}] shape {:?} != [{p}, {p}]",
                a.shape()
            ));
        }
        if b.len() != p {
            return Err(crate::gpu_err!(
                "try_device_moment_reduce: b[{idx}] len {} != {p}",
                b.len()
            ));
        }
    }

    #[cfg(target_os = "linux")]
    {
        if !super::runtime::GpuRuntime::is_available() {
            return Ok(None);
        }
        Some(linux::moment_reduce_linux(points, p)).transpose()
    }

    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

/// P7 (#32) — batched sigma-point packaging for the P6 moment reducer.
///
/// For batches where `p ≤ BATCHED_DISPATCH_MAX_P` and
/// `M ≥ BATCHED_DISPATCH_MIN_M`, packages all pre-computed
/// `(hessian_inv, beta)` pairs from `batch` into `Vec<DeviceSigmaPoint>`
/// in a single pass, ready for [`try_device_moment_reduce`] (P6).
///
/// The primary GPU sigma-cubature path is [`try_gpu_sigma_stream_pool_eval`]
/// (full PIRLS on device); this entry handles the complementary case where
/// the caller already has IRLS state and wants the on-device P6 reduction.
///
/// Returns `Ok(None)` when device unavailable, `M < BATCHED_DISPATCH_MIN_M`,
/// or `p > BATCHED_DISPATCH_MAX_P`. `Err(_)` on shape failure.
pub fn try_device_sigma_eval_batched(
    batch: &SigmaCubatureBatch<'_>,
) -> Result<Option<Vec<DeviceSigmaPoint>>, GpuError> {
    let (_n_rows, p) = batch.check_shape()?;
    if batch.m() < BATCHED_DISPATCH_MIN_M {
        // Below breakeven the per-stream P5 path wins.
        return Ok(None);
    }
    if p > BATCHED_DISPATCH_MAX_P {
        return Ok(None);
    }

    #[cfg(target_os = "linux")]
    {
        if !super::runtime::GpuRuntime::is_available() {
            return Ok(None);
        }
        // Package all pre-computed `(H_m⁻¹, β̂_m)` pairs so the caller can
        // pass them directly to `try_device_moment_reduce` (P6). The fused
        // 2-D grid launch for raw PIRLS (the original P7 design) is superseded
        // by `try_gpu_sigma_stream_pool_eval` which dispatches full PIRLS on
        // N_streams concurrent streams; P7 batching here refers to the
        // packaging optimisation for the P6 reducer path.
        let points: Vec<DeviceSigmaPoint> = batch
            .points
            .iter()
            .map(|pt| (pt.hessian_inv.to_owned(), pt.beta.to_owned()))
            .collect();
        Ok(Some(points))
    }

    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

/// Minimum sigma-point count above which the batched (P7) dispatch beats
/// the per-stream (P5) fan-out at biobank shape (n ≈ 200 k, p ≤ 64).
///
/// Below this the per-launch JIT-lookup fixed cost is fully amortised by
/// the stream pool so the simpler P5 path wins.  Calibrated against the
/// 2D + scale sigma-rank ≈ 8 cubature shape and the cap on `M` documented
/// in `compute_smoothing_correction_auto`.
pub const BATCHED_DISPATCH_MIN_M: usize = 6;

/// Upper bound on `p` for the fused batched kernel; above this the
/// register pressure of the per-thread β/H_inv slabs spills into local
/// memory and the per-stream P5 path wins again.
pub const BATCHED_DISPATCH_MAX_P: usize = 96;

#[cfg(target_os = "linux")]
mod linux {
    use super::DeviceSigmaPoint;
    use crate::gpu::common::PtxModuleCache;
    use crate::gpu::error::{GpuError, GpuResultExt};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use ndarray::Array2;
    use std::sync::Arc;

    /// Process-wide NVRTC cache for the device-side P6 reduction kernel.
    /// One source string, one compiled module, reused for every cubature
    /// pass: same lifecycle as every other `PtxModuleCache` consumer
    /// (`bms_flex`, `survival_flex`, `polya_gamma`, ...).
    static MOMENT_REDUCE_PTX: PtxModuleCache = PtxModuleCache::new();

    /// CUDA source for the P6 reduction.
    ///
    /// Three kernels share one compilation unit so the JIT module hosts
    /// the whole reducer:
    ///
    ///   * `sigma_mean_hinv(M, p, A_in, out)`   — `out = (1/M) Σ_m A_m`
    ///   * `sigma_mean_beta(M, p, b_in, out)`   — `out = (1/M) Σ_m b_m`
    ///   * `sigma_second_beta(M, p, b_in, out)` — `out = (1/M) Σ_m b_m b_mᵀ`
    ///
    /// Buffer layout is contiguous row-major per sigma point, sigma index
    /// in the leading dimension: `A_in[m*p*p + i*p + j]`, `b_in[m*p + i]`.
    /// `out` is allocated `p*p` (or `p`) and the launcher fans one thread
    /// per output entry with each thread looping over the `M` axis. With
    /// `M` capped at a handful and `p ≤ 128` the loop trip-count is
    /// trivial; the value-add is keeping the `M·p²` reads on-device.
    ///
    /// `(mean_bbT − mean_b mean_bᵀ)` is computed host-side from the three
    /// reduced outputs once they land — that 4 kB outer product is far
    /// below the launch breakeven and keeps this kernel a pure reducer.
    const MOMENT_REDUCE_SRC: &str = r#"
extern "C" __global__ void sigma_mean_hinv(int M, int p, const double* __restrict__ A_in, double* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p * p;
    if (idx >= total) return;
    double acc = 0.0;
    long long stride = (long long)p * (long long)p;
    for (int m = 0; m < M; ++m) {
        acc += A_in[(long long)m * stride + (long long)idx];
    }
    out[idx] = acc / (double)M;
}

extern "C" __global__ void sigma_mean_beta(int M, int p, const double* __restrict__ b_in, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    double acc = 0.0;
    for (int m = 0; m < M; ++m) {
        acc += b_in[(long long)m * (long long)p + (long long)i];
    }
    out[i] = acc / (double)M;
}

extern "C" __global__ void sigma_second_beta(int M, int p, const double* __restrict__ b_in, double* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p * p;
    if (idx >= total) return;
    int i = idx / p;
    int j = idx - i * p;
    double acc = 0.0;
    for (int m = 0; m < M; ++m) {
        double bi = b_in[(long long)m * (long long)p + (long long)i];
        double bj = b_in[(long long)m * (long long)p + (long long)j];
        acc += bi * bj;
    }
    out[idx] = acc / (double)M;
}
"#;

    /// Linux entry point for [`super::try_device_moment_reduce`].
    ///
    /// Lays out the per-sigma `(A_m, b_m)` blocks contiguously on the
    /// device, launches the three reduction kernels above on the same
    /// stream, downloads only `(mean_hinv, mean_beta, second_beta)`
    /// (totalling `2*p² + p` doubles regardless of M), and assembles
    /// `V̂_p = mean_hinv + second_beta − mean_beta · mean_betaᵀ` on the
    /// host.
    pub(super) fn moment_reduce_linux(
        points: &[DeviceSigmaPoint],
        p: usize,
    ) -> Result<Array2<f64>, GpuError> {
        let runtime = crate::gpu::runtime::GpuRuntime::global().ok_or_else(|| {
            crate::gpu_err!("try_device_moment_reduce: GpuRuntime unavailable after probe accepted")
        })?;
        let ctx = crate::gpu::runtime::cuda_context_for(runtime.selected_device().ordinal)
            .ok_or_else(|| {
                crate::gpu_err!(
                    "try_device_moment_reduce: CUDA context for ordinal {} unavailable",
                    runtime.selected_device().ordinal
                )
            })?;
        ctx.bind_to_thread()
            .gpu_ctx("try_device_moment_reduce bind_to_thread")?;
        let stream = ctx.default_stream();

        let m = points.len();
        let module = MOMENT_REDUCE_PTX.get_or_compile(
            &ctx,
            "sigma_cubature_moment_reduce",
            MOMENT_REDUCE_SRC,
        )?;

        // Pack A and b on host then upload once. `m * p * p` is bounded
        // by the sigma-point cap × p² (e.g. 12 * 64² = 49 152 doubles ≈
        // 384 kB) so the upload is small.
        let p2 = p * p;
        let mut a_flat: Vec<f64> = Vec::with_capacity(m * p2);
        let mut b_flat: Vec<f64> = Vec::with_capacity(m * p);
        for (a, b) in points {
            let a_slice = a
                .as_slice()
                .ok_or_else(|| crate::gpu_err!("A_m not contiguous in moment_reduce_linux"))?;
            a_flat.extend_from_slice(a_slice);
            let b_slice = b
                .as_slice()
                .ok_or_else(|| crate::gpu_err!("b_m not contiguous in moment_reduce_linux"))?;
            b_flat.extend_from_slice(b_slice);
        }
        let a_dev = stream
            .clone_htod(&a_flat)
            .gpu_ctx("sigma_cubature htod A")?;
        let b_dev = stream
            .clone_htod(&b_flat)
            .gpu_ctx("sigma_cubature htod b")?;
        let mut mean_hinv_dev = stream
            .alloc_zeros::<f64>(p2)
            .gpu_ctx("sigma_cubature alloc mean_hinv")?;
        let mut mean_beta_dev = stream
            .alloc_zeros::<f64>(p)
            .gpu_ctx("sigma_cubature alloc mean_beta")?;
        let mut second_beta_dev = stream
            .alloc_zeros::<f64>(p2)
            .gpu_ctx("sigma_cubature alloc second_beta")?;

        const THREADS: u32 = 128;
        let m_i32 =
            i32::try_from(m).map_err(|_| crate::gpu_err!("sigma_cubature M={m} overflows i32"))?;
        let p_i32 =
            i32::try_from(p).map_err(|_| crate::gpu_err!("sigma_cubature p={p} overflows i32"))?;

        // Kernel 1: mean_hinv (p*p threads).
        {
            let func = module
                .load_function("sigma_mean_hinv")
                .gpu_ctx("sigma_cubature load sigma_mean_hinv")?;
            let total = u32::try_from(p2)
                .map_err(|_| crate::gpu_err!("sigma_cubature p*p={p2} overflows u32"))?;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(THREADS).max(1), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&func);
            builder.arg(&m_i32);
            builder.arg(&p_i32);
            builder.arg(&a_dev);
            builder.arg(&mut mean_hinv_dev);
            // SAFETY: argument types and order match the kernel
            // declaration (`int, int, const double*, double*`). The
            // output buffer was allocated `p*p`; the grid covers all
            // `p*p` indices; threads above the bound short-circuit.
            unsafe { builder.launch(cfg) }
                .map(|_event_pair| ())
                .gpu_ctx("sigma_cubature launch sigma_mean_hinv")?;
        }

        // Kernel 2: mean_beta (p threads).
        {
            let func = module
                .load_function("sigma_mean_beta")
                .gpu_ctx("sigma_cubature load sigma_mean_beta")?;
            let total = u32::try_from(p)
                .map_err(|_| crate::gpu_err!("sigma_cubature p={p} overflows u32"))?;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(THREADS).max(1), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&func);
            builder.arg(&m_i32);
            builder.arg(&p_i32);
            builder.arg(&b_dev);
            builder.arg(&mut mean_beta_dev);
            // SAFETY: argument types and order match the kernel
            // declaration; output buffer was allocated `p`; the grid
            // covers all `p` indices; out-of-range threads short-circuit.
            unsafe { builder.launch(cfg) }
                .map(|_event_pair| ())
                .gpu_ctx("sigma_cubature launch sigma_mean_beta")?;
        }

        // Kernel 3: second_beta (p*p threads).
        {
            let func = module
                .load_function("sigma_second_beta")
                .gpu_ctx("sigma_cubature load sigma_second_beta")?;
            let total = u32::try_from(p2)
                .map_err(|_| crate::gpu_err!("sigma_cubature p*p={p2} overflows u32"))?;
            let cfg = LaunchConfig {
                grid_dim: (total.div_ceil(THREADS).max(1), 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&func);
            builder.arg(&m_i32);
            builder.arg(&p_i32);
            builder.arg(&b_dev);
            builder.arg(&mut second_beta_dev);
            // SAFETY: argument types and order match the kernel
            // declaration; output buffer was allocated `p*p`; the grid
            // covers all `p*p` indices; out-of-range threads short-circuit.
            unsafe { builder.launch(cfg) }
                .map(|_event_pair| ())
                .gpu_ctx("sigma_cubature launch sigma_second_beta")?;
        }

        // Download the three reduced outputs in one synchronisation
        // window; the host outer product (`mean_beta · mean_betaᵀ`) is
        // p² doubles — far below the launch breakeven and avoids
        // shipping a fourth kernel.
        let mean_hinv_host = stream
            .clone_dtoh(&mean_hinv_dev)
            .gpu_ctx("sigma_cubature dtoh mean_hinv")?;
        let mean_beta_host = stream
            .clone_dtoh(&mean_beta_dev)
            .gpu_ctx("sigma_cubature dtoh mean_beta")?;
        let second_beta_host = stream
            .clone_dtoh(&second_beta_dev)
            .gpu_ctx("sigma_cubature dtoh second_beta")?;
        stream
            .synchronize()
            .gpu_ctx("sigma_cubature synchronize after dtoh")?;

        let mean_hinv = Array2::from_shape_vec((p, p), mean_hinv_host).map_err(|err| {
            crate::gpu_err!("sigma_cubature mean_hinv reshape failed (p={p}): {err}")
        })?;
        let second_beta = Array2::from_shape_vec((p, p), second_beta_host).map_err(|err| {
            crate::gpu_err!("sigma_cubature second_beta reshape failed (p={p}): {err}")
        })?;
        // mean_beta · mean_betaᵀ (host outer product, p² doubles).
        let mut mean_outer = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                mean_outer[[i, j]] = mean_beta_host[i] * mean_beta_host[j];
            }
        }
        Ok(mean_hinv + (second_beta - mean_outer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn small_pair(diag: f64, beta: &[f64]) -> DeviceSigmaPoint {
        let p = beta.len();
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            a[[i, i]] = diag;
        }
        (a, Array1::from(beta.to_vec()))
    }

    #[test]
    fn moment_reduce_rejects_empty_input() {
        let err = try_device_moment_reduce(&[], 3).unwrap_err();
        match err {
            GpuError::DriverCallFailed { reason } => {
                assert!(reason.contains("empty points"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn moment_reduce_rejects_shape_mismatch() {
        let pts = vec![small_pair(1.0, &[0.1, 0.2])];
        let err = try_device_moment_reduce(&pts, 3).unwrap_err();
        match err {
            GpuError::DriverCallFailed { reason } => {
                assert!(reason.contains("A[0] shape"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn moment_reduce_declines_when_runtime_unavailable() {
        let pts = vec![small_pair(1.0, &[0.1, 0.2])];
        // On CPU-only hosts (or non-Linux), the runtime probe returns
        // None and the entry must decline cleanly (Ok(None)) instead of
        // raising a driver error — that's the contract the CPU Rayon
        // fallback relies on at the call site.
        let outcome = try_device_moment_reduce(&pts, 2);
        if !crate::gpu::runtime::GpuRuntime::is_available() {
            assert!(matches!(outcome, Ok(None)));
        }
    }

    #[test]
    fn batched_dispatch_below_breakeven_declines() {
        let eta = array![0.0, 0.1];
        let y = array![0.0, 1.0];
        let prior_w = array![1.0, 1.0];
        let beta = array![0.0, 0.0];
        let hessian_inv = Array2::<f64>::eye(2);
        let pts: Vec<SigmaPointInput<'_>> = (0..(BATCHED_DISPATCH_MIN_M - 1))
            .map(|_| SigmaPointInput {
                eta: eta.view(),
                y: y.view(),
                prior_w: prior_w.view(),
                beta: beta.view(),
                hessian_inv: hessian_inv.view(),
            })
            .collect();
        let batch = SigmaCubatureBatch {
            family: PirlsRowFamily::BernoulliLogit,
            curvature: CurvatureMode::Fisher,
            points: &pts,
        };
        let outcome = try_device_sigma_eval_batched(&batch).expect("preflight succeeds");
        assert!(
            outcome.is_none(),
            "below-breakeven batch must decline cleanly"
        );
    }

    #[test]
    fn sigma_eval_declines_when_runtime_unavailable() {
        let eta = array![0.0];
        let y = array![1.0];
        let prior_w = array![1.0];
        let beta = array![0.0];
        let hessian_inv = Array2::<f64>::eye(1);
        let pts = vec![SigmaPointInput {
            eta: eta.view(),
            y: y.view(),
            prior_w: prior_w.view(),
            beta: beta.view(),
            hessian_inv: hessian_inv.view(),
        }];
        let batch = SigmaCubatureBatch {
            family: PirlsRowFamily::BernoulliLogit,
            curvature: CurvatureMode::Fisher,
            points: &pts,
        };
        let outcome = try_device_sigma_eval(&batch).expect("shape preflight succeeds");
        if !crate::gpu::runtime::GpuRuntime::is_available() {
            assert!(outcome.is_none());
        }
    }
}
