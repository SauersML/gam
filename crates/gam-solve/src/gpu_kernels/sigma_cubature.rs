//! Device-side sigma-cubature stream-pool dispatch.
//!
//! The live GPU entry is [`try_gpu_sigma_stream_pool_eval`]. It runs each
//! sigma point through the unified PIRLS stream-pool executor, returns the
//! per-point `(H_original^-1, beta_original)` pairs, and hands the shared
//! covariance accumulation back to
//! [`crate::estimate::reml::eval::accumulate_sigma_cubature_total_covariance`].

use ndarray::{Array1, Array2, ArrayView1};

use gam_gpu::gpu_error::GpuError;

#[derive(Debug)]
pub enum SigmaCubatureGpuError {
    Geometry(gam_problem::EstimationError),
    Runtime(GpuError),
}

impl std::fmt::Display for SigmaCubatureGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry(error) => write!(f, "{error}"),
            Self::Runtime(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SigmaCubatureGpuError {}

impl From<GpuError> for SigmaCubatureGpuError {
    fn from(error: GpuError) -> Self {
        Self::Runtime(error)
    }
}

/// Per-sigma-point GPU PIRLS input: penalty, reparameterisation transform,
/// and prior-mean shifts for one ρ / σ point.
///
/// Built by [`crate::estimate::reml::eval::sigma_cubature_evaluate_gpu_stream_pool`]
/// from the reparameterisation engine output before the stream pool is allocated.
/// The shared model data (`X_original`, `y`, `prior_w`, `offset`) is uploaded
/// ONCE into [`crate::gpu::pirls_gpu::PirlsGpuSharedData`]; only the
/// small per-point algebra (p×p Qs, p×p S, length-p shift, scalar) needs
/// uploading per sigma point.
pub struct SigmaPointGpuInput {
    /// `p × p` penalised-Hessian contribution `S_λ` in the transformed basis.
    pub s_transformed: Array2<f64>,
    /// `p × p` reparameterisation matrix `Qs`. Uploaded via
    /// `pirls_gpu::upload_qs_pirls` once per sigma point; also used on the
    /// CPU to map the loop's `β_transformed` and `H_transformed` back to the
    /// original basis so the downstream cubature accumulator receives
    /// `(H_original⁻¹, β_original)`.
    pub qs: Array2<f64>,
    /// Length-p linear shift `b` for the shifted-quadratic penalty
    /// `βᵀSβ − 2βᵀb + c`. All-zero for the default sigma-cubature path.
    pub linear_shift: Array1<f64>,
    /// Scalar constant shift `c`. Zero for the default sigma-cubature path.
    pub constant_shift: f64,
}

/// Default number of concurrent CUDA streams in the sigma-cubature pool.
///
/// Caps at `min(8, M)` so we never allocate more streams than sigma points.
/// Eight concurrent streams saturates the SM scheduler on all shipping
/// datacenter GPUs without exhausting the per-context stream limit.
#[cfg(target_os = "linux")]
const STREAM_POOL_MAX: usize = 8;

/// Initial Levenberg-Marquardt damping for each sigma-point PIRLS fit.
///
/// The sigma-point fits are cold-started from a zero β seed, so a small but
/// non-zero seed damping keeps the first Gauss-Newton step well-conditioned
/// when the design's `XᵀWX` is near-singular at β=0; the inner loop's own
/// trust-region logic then grows or shrinks it. `1e-6` is the same seed the
/// stateless CPU sigma-cubature path uses, so the two solvers take an
/// identical first step.
#[cfg(target_os = "linux")]
const SIGMA_PIRLS_INITIAL_LM_LAMBDA: f64 = 1e-6;

/// Compute the stream-pool size for a batch of M sigma points.
///
/// Auto-derived — no flag, no env var.
#[cfg(target_os = "linux")]
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
/// `x_original`: Original (pre-reparameterization) dense design matrix X_original, shape n × p.
/// Uploaded to device once and reused across all sigma points.
/// `gamma_shape`: Active Gamma dispersion shape (α > 0). Pass `1.0` for non-Gamma families.
pub fn try_gpu_sigma_stream_pool_eval(
    x_original: ndarray::ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    prior_w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    per_sigma: &[SigmaPointGpuInput],
    admission: gam_gpu::policy::PirlsLoopAdmission,
    gamma_shape: f64,
    convergence_tol: f64,
    max_iter: usize,
) -> Result<Option<Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>>, SigmaCubatureGpuError> {
    if per_sigma.is_empty() {
        return Ok(Some(Vec::new()));
    }
    validate_sigma_point_inputs(x_original.ncols(), per_sigma)?;

    #[cfg(target_os = "linux")]
    {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            return Ok(None);
        }
        let Some(family_kind) = admission.family else {
            return Ok(None);
        };
        let Some(family) = linux_impl::family_kind_to_row(family_kind) else {
            return Err(
                gam_gpu::gpu_err!("sigma stream pool: family not in JIT-cached set").into(),
            );
        };
        let curvature = linux_impl::curvature_kind_to_row(admission.curvature);
        return linux_impl::stream_pool_eval(
            x_original,
            y,
            prior_w,
            offset,
            per_sigma,
            family,
            curvature,
            gamma_shape,
            convergence_tol,
            max_iter,
        );
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Non-Linux: no CUDA runtime. Consume every parameter in a single
        // trace log so each binding is read once, satisfying -D warnings
        // without an #[allow(unused_variables)] suppression.
        log::trace!(
            "[sigma stream pool] non-Linux target: skipping dispatch \
             (x_original={}x{}, y_len={}, prior_w_len={}, offset_len={}, \
              n_sigma={}, family={:?}, curvature={:?}, gamma_shape={}, \
              tol={}, max_iter={})",
            x_original.nrows(),
            x_original.ncols(),
            y.len(),
            prior_w.len(),
            offset.len(),
            per_sigma.len(),
            admission.family,
            admission.curvature,
            gamma_shape,
            convergence_tol,
            max_iter,
        );
        Ok(None)
    }
}

fn validate_sigma_point_inputs(p: usize, per_sigma: &[SigmaPointGpuInput]) -> Result<(), GpuError> {
    for (idx, pt) in per_sigma.iter().enumerate() {
        if pt.s_transformed.shape() != [p, p] {
            return Err(gam_gpu::gpu_err!(
                "sigma stream pool: point[{idx}] S shape {:?} != [{p}, {p}]",
                pt.s_transformed.shape()
            ));
        }
        if pt.qs.shape() != [p, p] {
            return Err(gam_gpu::gpu_err!(
                "sigma stream pool: point[{idx}] Qs shape {:?} != [{p}, {p}]",
                pt.qs.shape()
            ));
        }
        if pt.linear_shift.len() != p {
            return Err(gam_gpu::gpu_err!(
                "sigma stream pool: point[{idx}] linear shift len {} != {p}",
                pt.linear_shift.len()
            ));
        }
        if !pt.constant_shift.is_finite() {
            return Err(gam_gpu::gpu_err!(
                "sigma stream pool: point[{idx}] non-finite constant shift {}",
                pt.constant_shift
            ));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod linux_impl {
    use crate::gpu_kernels::pirls_row::{CurvatureMode, PirlsRowFamily};
    use crate::gpu_kernels::sigma_cubature::{SigmaCubatureGpuError, SigmaPointGpuInput};
    use gam_gpu::gpu_error::GpuError;
    use gam_gpu::policy::{PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
    use gam_linalg::utils::matrix_inversewith_regularization;
    use ndarray::{Array1, Array2, ArrayView1};
    type SigmaPointResult = (Array2<f64>, Array1<f64>);

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
        gam_linalg::matrix::symmetrize_in_place(&mut h_orig);
        h_orig
    }

    pub(super) fn stream_pool_eval(
        x_original: ndarray::ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        prior_w: ArrayView1<'_, f64>,
        offset: ArrayView1<'_, f64>,
        per_sigma: &[SigmaPointGpuInput],
        family: PirlsRowFamily,
        curvature: CurvatureMode,
        gamma_shape: f64,
        convergence_tol: f64,
        max_iter: usize,
    ) -> Result<Option<Vec<SigmaPointResult>>, SigmaCubatureGpuError> {
        use crate::gpu::pirls_gpu;
        use crate::gpu_kernels::sigma_cubature::pool_size;

        let m = per_sigma.len();
        let p = x_original.ncols();

        // Validate uniform shape across all sigma points.
        for (idx, pt) in per_sigma.iter().enumerate() {
            if pt.s_transformed.shape() != [p, p] || pt.qs.shape() != [p, p] {
                return Err(gam_gpu::gpu_err!(
                    "sigma stream pool: point[{idx}] shape mismatch against point[0]"
                )
                .into());
            }
        }

        // Gaussian-identity exact PLS bypass (#272).
        //
        // For Gaussian-identity the working weight is prior_w (constant across
        // all sigma points). XᵀWX and XᵀW(y−offset) are therefore the same for
        // every point. Compute them once, then for each sigma point call the
        // exact Gaussian PLS solver with the per-point (Qs, S_transformed,
        // linear_shift) — no row-kernel PIRLS loop, no iterative solver.
        if family == PirlsRowFamily::GaussianIdentity {
            return gaussian_sigma_pool_eval(x_original, y, prior_w, offset, per_sigma, p)
                .map_err(SigmaCubatureGpuError::Runtime);
        }

        // Upload X_original, y, prior_w, offset once — shared across all sigma points.
        // Per sigma point, only Qs changes; it gets uploaded via upload_qs_pirls.
        let bootstrap_shared =
            pirls_gpu::upload_shared_pirls_gpu(x_original, y, prior_w, offset)
                .map_err(|e| gam_gpu::gpu_err!("sigma stream pool bootstrap upload: {e}"))?;

        let n_streams = pool_size(m);

        // Allocate N_streams workspace pairs bound to independent streams.
        let mut workspace_pairs: Vec<(
            crate::gpu::pirls_gpu::SigmaPirlsGpuWorkspace,
            crate::gpu::pirls_gpu::cuda::PirlsLoopWorkspace,
        )> = Vec::with_capacity(n_streams);
        for _ in 0..n_streams {
            let ws = pirls_gpu::allocate_sigma_pirls_workspace(&bootstrap_shared)
                .map_err(|e| gam_gpu::gpu_err!("sigma stream pool alloc workspace: {e}"))?;
            let loop_ws = pirls_gpu::allocate_pirls_loop_workspace(&bootstrap_shared, &ws)
                .map_err(|e| gam_gpu::gpu_err!("sigma stream pool alloc loop_ws: {e}"))?;
            workspace_pairs.push((ws, loop_ws));
        }

        // Zero-initialised beta seed (length p). The sigma-point PIRLS fits
        // have no warm-start; a zero seed matches the stateless CPU path.
        let beta0: Array1<f64> = Array1::zeros(p);

        // For each sigma point, upload Qs (small p×p) then run pirls_loop.
        // X_original, y, prior_w, offset stay in bootstrap_shared throughout.
        let mut outcomes: Vec<SigmaPointResult> = Vec::with_capacity(m);
        for (idx, pt) in per_sigma.iter().enumerate() {
            let stream_idx = idx % n_streams;

            let (ws, loop_ws) = &mut workspace_pairs[stream_idx];
            // Upload this sigma point's Qs matrix to the workspace.
            pirls_gpu::upload_qs_pirls(ws, pt.qs.view())
                .map_err(|e| gam_gpu::gpu_err!("sigma stream pool upload Qs pt[{idx}]: {e}"))?;
            let shared = &bootstrap_shared;

            // Use per-point linear_shift and constant_shift from SigmaPointGpuInput (#260).
            let outcome = pirls_gpu::pirls_loop_on_stream(
                shared,
                ws,
                loop_ws,
                family,
                curvature,
                gamma_shape,
                beta0.view(),
                pt.s_transformed.view(),
                pt.linear_shift.view(),
                pt.constant_shift,
                super::SIGMA_PIRLS_INITIAL_LM_LAMBDA,
                0.0,
                max_iter,
                convergence_tol,
                None,
            );

            let loop_out = match outcome {
                Ok(loop_out) => loop_out,
                Err(pirls_gpu::cuda::PirlsGpuLoopError::Geometry(error)) => {
                    return Err(SigmaCubatureGpuError::Geometry(error));
                }
                Err(pirls_gpu::cuda::PirlsGpuLoopError::Runtime(message)) => {
                    return Err(SigmaCubatureGpuError::Runtime(gam_gpu::gpu_err!(
                        "sigma point[{idx}] GPU PIRLS runtime failure: {message}"
                    )));
                }
            };

            // Map H_transformed → H_original, invert, map β_transformed
            // → β_original. Mirrors the CPU path's post-processing.
            let h_orig = hessian_to_original(&loop_out.penalized_hessian, &pt.qs);
            let cov =
                matrix_inversewith_regularization(&h_orig, "gpu sigma point").ok_or_else(|| {
                    gam_gpu::gpu_err!("gpu sigma point: penalised Hessian inverse not well-defined")
                })?;
            let beta_orig = pt.qs.dot(&loop_out.beta);
            let sigma_result = (cov, beta_orig);

            outcomes.push(sigma_result);
        }

        Ok(Some(outcomes))
    }

    /// Gaussian-identity sigma-cubature bypass (#272).
    ///
    /// For Gaussian-identity the working weight equals the prior weight, which
    /// is the same for every sigma point. XᵀWX and XᵀW(y−offset) are computed
    /// once from `x_original`, `prior_w`, `y`, `offset`, and then the exact GPU
    /// PLS solver is called per sigma point with the per-point `(Qs, S, shift)`.
    /// This eliminates the row-kernel PIRLS loop entirely for Gaussian fits,
    /// matching the single-fit `try_gpu_gaussian_pls_dispatch` bypass.
    fn gaussian_sigma_pool_eval(
        x_original: ndarray::ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        prior_w: ArrayView1<'_, f64>,
        offset: ArrayView1<'_, f64>,
        per_sigma: &[SigmaPointGpuInput],
        p: usize,
    ) -> Result<Option<Vec<SigmaPointResult>>, GpuError> {
        use ndarray::Array1;
        // XᵀWX = Xᵀ·diag(prior_w)·X (constant across all sigma points).
        // Computed on the GPU via weighted_crossprod_gpu.
        let xtwx = crate::gpu::pirls_gpu::weighted_crossprod_gpu(x_original, prior_w)
            .map_err(|e| gam_gpu::gpu_err!("gaussian sigma: XᵀWX gpu failed: {e}"))?;

        // XᵀW(y − offset) = Xᵀ·diag(prior_w)·(y − offset).
        // Compute on the host (n-vector; the GPU would need a separate kernel).
        let mut yw = y.to_owned();
        yw -= &offset;
        yw *= &prior_w;
        // Xᵀ·(prior_w · (y − offset)).
        let xtwy: Array1<f64> = x_original.t().dot(&yw);

        let prior_mean_zero: Array1<f64> = Array1::zeros(p);

        let mut outcomes: Vec<SigmaPointResult> = Vec::with_capacity(per_sigma.len());
        for (idx, pt) in per_sigma.iter().enumerate() {
            let pls = crate::gpu::pirls_gpu::solve_gaussian_pls_gpu(
                xtwx.view(),
                xtwy.view(),
                pt.s_transformed.view(),
                pt.linear_shift.view(),
                prior_mean_zero.view(),
                0.0,
                Some(pt.qs.view()),
            )
            .map_err(|e| gam_gpu::gpu_err!("gaussian sigma pool: point[{idx}] pls failed: {e}"))?;

            let h_orig = hessian_to_original(&pls.penalized_hessian, &pt.qs);
            let cov = matrix_inversewith_regularization(&h_orig, "gaussian sigma point")
                .ok_or_else(|| {
                    gam_gpu::gpu_err!(
                        "gaussian sigma point: penalised Hessian inverse not well-defined"
                    )
                })?;
            let beta_orig = pt.qs.dot(&pls.beta);
            outcomes.push((cov, beta_orig));
        }

        Ok(Some(outcomes))
    }
}
