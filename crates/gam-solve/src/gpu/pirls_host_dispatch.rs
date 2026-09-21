//! Stage 3.3 GPU dispatch glue — PIRLS-side host code.
//!
//! Owns the two Linux-gated dispatch helper functions that decide whether
//! to route a fixed-ρ fit through the CUDA-resident path:
//!
//! - `try_gaussian_pls_gpu` — Gaussian-Identity exact POTRF/POTRS dispatch
//!   (wraps `crate::gpu::pirls_dispatch_wire::try_gpu_gaussian_pls_dispatch`).
//! - `try_pirls_loop_gpu` — general PIRLS-loop device dispatch
//!   (wraps `crate::gpu::pirls_dispatch_wire::try_gpu_pirls_loop_dispatch`).
//!
//! Both functions return `Option<Result<..>>`: `Some(Ok(pair))` on a successful
//! admitted device solve, `Some(Err(..))` on a typed admitted-device failure,
//! and `None` only when the dispatch criteria are not met.
//!
//! The GPU **kernel** bodies live in `crate::gpu::pirls_gpu` and
//! `crate::gpu::pirls_dispatch_wire`; this file only owns the
//! host-side admission logic and struct assembly.

use crate::estimate::EstimationError;
#[cfg(target_os = "linux")]
use crate::pirls::loop_driver::make_reparam_operator;
use crate::pirls::{
    GaussianFixedCache, GaussianFrozenRows, LinearInequalityConstraints, PirlsConfig,
    PirlsCoordinateFrame, PirlsPenalty, PirlsResult, WorkingModelPirlsResult,
};
use gam_linalg::matrix::DesignMatrix;
use gam_problem::LinkFunction;
use gam_terms::construction::ReparamResult;
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

/// Stage 3.3-GI: Try to route a Gaussian-Identity fit through the CUDA
/// POTRF/POTRS path.
///
/// Returns `None` when dispatch criteria are not met (non-Linux, missing
/// runtime, non-Gaussian family, Firth active, bounds/constraints present,
/// no cache, or non-Dense penalty). Returns `Some(Ok(pair))` on success and
/// `Some(Err(..))` when the device solve errored so the caller can fall
/// through to the CPU identity path.
///
/// `materialize_reparam` is called lazily — only when every gating condition
/// is satisfied — to produce the `ReparamResult` the GPU input needs.
pub(crate) fn try_gaussian_pls_gpu<F>(
    link_function: LinkFunction,
    config: &PirlsConfig,
    penalty_coefficient_lower_bounds: Option<&Array1<f64>>,
    penalty_linear_constraints_original: Option<&LinearInequalityConstraints>,
    gaussian_fixed_cache: Option<&GaussianFixedCache>,
    penalty_active: &PirlsPenalty,
    qs_arc: &Option<Arc<Array2<f64>>>,
    x_original: &DesignMatrix,
    use_sparse_native: bool,
    penalty_p: usize,
    materialize_reparam: F,
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    coordinate_frame: PirlsCoordinateFrame,
    linear_constraints: &Option<LinearInequalityConstraints>,
    // gh#2544: the trial-invariant frozen-row bundle, when the caller is a
    // value-only rho probe. Threaded through so this dispatch can take the
    // same k-space synthesis #1868 installed on the branch it pre-empts.
    cost_only_gaussian_rows: Option<&Arc<GaussianFrozenRows>>,
) -> Option<Result<(PirlsResult, WorkingModelPirlsResult), EstimationError>>
where
    F: FnOnce() -> Result<ReparamResult, EstimationError>,
{
    #[cfg(not(target_os = "linux"))]
    {
        let callback_size = std::mem::size_of_val(&materialize_reparam);
        log::trace!(
            "[PIRLS GPU Gaussian PLS] declined on non-linux \
             (link_bytes={}, max_iter={}, lower_bounds={}, original_constraints={}, \
             fixed_cache={}, penalty_ptr={:p}, qs={}, x_ptr={:p}, sparse_native={}, p={}, \
             callback_size={}, y_len={}, prior_len={}, offset_len={}, frame_bytes={}, constraints={}, \
             frozen_rows={})",
            std::mem::size_of_val(&link_function),
            config.max_iterations,
            penalty_coefficient_lower_bounds.is_some(),
            penalty_linear_constraints_original.is_some(),
            gaussian_fixed_cache.is_some(),
            penalty_active,
            qs_arc.is_some(),
            x_original,
            use_sparse_native,
            penalty_p,
            callback_size,
            y.len(),
            priorweights.len(),
            offset.len(),
            std::mem::size_of_val(&coordinate_frame),
            linear_constraints.is_some(),
            cost_only_gaussian_rows.is_some(),
        );
    }
    #[cfg(target_os = "linux")]
    if matches!(link_function, LinkFunction::Identity)
        && config.likelihood.spec.is_gaussian_identity()
        && !config.firth_bias_reduction
        && penalty_coefficient_lower_bounds.is_none()
        && penalty_linear_constraints_original.is_none()
    {
        use crate::gpu::pirls_dispatch_wire::{
            GpuGaussianPlsInput, try_gpu_gaussian_pls_admit, try_gpu_gaussian_pls_dispatch,
        };
        // Admission runs before the input is assembled, as in
        // `try_pirls_loop_gpu`: the transformed-design operator and the owned
        // `ReparamResult` copy are only built for a fit the device will take.
        match try_gpu_gaussian_pls_admit(&config.likelihood) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(error) => {
                return Some(Err(EstimationError::RemlOptimizationFailed(format!(
                    "GPU Gaussian PLS runtime: Gaussian PLS admission runtime resolution \
                     failed: {error}"
                ))));
            }
        }
        if let Some(cache) = gaussian_fixed_cache {
            let PirlsPenalty::Dense { s_transformed, .. } = penalty_active;
            let qs_view = qs_arc.as_ref().map(|qs| qs.view());
            let qs_arc_for_design = qs_arc
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Arc::new(Array2::<f64>::eye(penalty_p)));
            let x_transformed_design =
                make_reparam_operator(x_original, &qs_arc_for_design, use_sparse_native);
            let reparam_for_gpu = match materialize_reparam() {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            let gpu_input = GpuGaussianPlsInput {
                xtwx_orig: cache.xtwx_orig.view(),
                xtwy_orig: cache.xtwy_orig.view(),
                s_transformed: s_transformed.view(),
                qs: qs_view,
                likelihood: &config.likelihood,
                inverse_link: &config.link_kind,
                x_original,
                y,
                priorweights,
                offset,
                reparam_result: reparam_for_gpu,
                x_transformed_design,
                coordinate_frame,
                linear_constraints: linear_constraints.clone(),
                centered_weighted_y_sq: cache.centered_weighted_y_sq,
                frozen_rows: cost_only_gaussian_rows.map(Arc::clone),
            };
            if let Some(result) = try_gpu_gaussian_pls_dispatch(gpu_input) {
                return Some(result.map_err(|message| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "GPU Gaussian PLS runtime: {message}"
                    ))
                }));
            }
        }
    }
    None
}

/// Stage 3.3: Try to route a dense-design PIRLS fit through the CUDA
/// device-resident loop.
///
/// Returns `None` when admission is denied (non-Linux, missing runtime, sparse
/// design, Firth active, constraints present, or shape/family
/// outside the dispatch policy). Returns `Some(Ok(pair))` on success and
/// `Some(Err(..))` on an admitted-device error; the typed error is propagated
/// without retrying a different numerical implementation.
///
/// `materialize_reparam` is called lazily — only when the admission shim
/// confirms the fit is eligible.
pub(crate) fn try_pirls_loop_gpu<F>(
    config: &PirlsConfig,
    penalty_active: &PirlsPenalty,
    use_sparse_native: bool,
    linear_constraints: &Option<LinearInequalityConstraints>,
    x_original: &DesignMatrix,
    qs_arc: &Option<Arc<Array2<f64>>>,
    penalty_p: usize,
    x_original_for_result: &DesignMatrix,
    materialize_reparam: F,
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    initial_beta: &Array1<f64>,
    link_function: LinkFunction,
    coordinate_frame: PirlsCoordinateFrame,
) -> Option<Result<(PirlsResult, WorkingModelPirlsResult), EstimationError>>
where
    F: FnOnce() -> Result<ReparamResult, EstimationError>,
{
    #[cfg(not(target_os = "linux"))]
    {
        let callback_size = std::mem::size_of_val(&materialize_reparam);
        log::trace!(
            "[PIRLS GPU dispatch] declined on non-linux \
             (max_iter={}, penalty_ptr={:p}, sparse_native={}, constraints={}, \
             x_ptr={:p}, qs={}, p={}, result_x_ptr={:p}, callback_size={}, y_len={}, prior_len={}, \
             offset_len={}, beta_len={}, link_bytes={}, frame_bytes={})",
            config.max_iterations,
            penalty_active,
            use_sparse_native,
            linear_constraints.is_some(),
            x_original,
            qs_arc.is_some(),
            penalty_p,
            x_original_for_result,
            callback_size,
            y.len(),
            priorweights.len(),
            offset.len(),
            initial_beta.len(),
            std::mem::size_of_val(&link_function),
            std::mem::size_of_val(&coordinate_frame),
        );
    }
    #[cfg(target_os = "linux")]
    use crate::pirls::HessianCurvatureKind;
    #[cfg(target_os = "linux")]
    {
        use crate::gpu::pirls_dispatch_wire::{
            GpuPirlsDispatchInput, try_gpu_pirls_loop_admit, try_gpu_pirls_loop_dispatch,
        };
        let dense_x = x_original.as_dense().map(|d| d.view());
        let no_sparse_native = !use_sparse_native;
        let no_firth = !config.firth_bias_reduction;
        let no_constraints = linear_constraints.is_none();
        if let (true, true, true, Some(x_dense)) = (
            no_sparse_native,
            no_firth,
            no_constraints,
            dense_x,
            ) {
            let n_admit = x_dense.nrows();
            let p_admit = x_dense.ncols();
            let gpu_admitted = match try_gpu_pirls_loop_admit(
                &config.likelihood,
                n_admit,
                p_admit,
            ) {
                Ok(admitted) => admitted,
                Err(error) => {
                    return Some(Err(EstimationError::InvalidInput(format!(
                        "PIRLS GPU admission failed: {error}"
                    ))));
                }
            };
            if gpu_admitted {
                let qs_view = qs_arc.as_ref().map(|qs| qs.view());
                let PirlsPenalty::Dense { s_transformed, .. } = penalty_active;
                let s_transformed_view = s_transformed.view();
                let qs_arc_for_design = qs_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| Arc::new(Array2::<f64>::eye(penalty_p)));
                let x_transformed_design = make_reparam_operator(
                    x_original_for_result,
                    &qs_arc_for_design,
                    use_sparse_native,
                );
                let reparam_for_dispatch = match materialize_reparam() {
                    Ok(r) => r,
                    Err(e) => return Some(Err(e)),
                };
                let initial_beta_owned = initial_beta.clone();
                let exported_curvature_kind = match link_function {
                    LinkFunction::Probit | LinkFunction::CLogLog => HessianCurvatureKind::Observed,
                    _ => HessianCurvatureKind::Fisher,
                };
                // Firth is already gated out upstream (no_firth check).
                let max_iterations = config.max_iterations;
                let dispatch = GpuPirlsDispatchInput {
                    likelihood: &config.likelihood,
                    inverse_link: &config.link_kind,
                    x_original: x_dense,
                    s_transformed: s_transformed_view,
                    y,
                    priorweights,
                    offset,
                    initial_beta: initial_beta_owned.view(),
                    initial_lm_lambda: config.initial_lm_lambda,
                    max_iterations,
                    convergence_tolerance: config.convergence_tolerance,
                    qs: qs_view,
                    reparam_result: reparam_for_dispatch,
                    x_transformed_design,
                    coordinate_frame,
                    exported_curvature: exported_curvature_kind,
                };
                if let Some(result) = try_gpu_pirls_loop_dispatch(dispatch) {
                    // Admission is a numerical execution decision, not a
                    // speculative fallback. Preserve exact typed row refusals
                    // and runtime failures instead of silently rerunning a
                    // different implementation.
                    return Some(result);
                }
            }
        }
    }
    None
}
