//! Stage 3.3 GPU dispatch glue — PIRLS-side host code.
//!
//! Owns the two Linux-gated dispatch helper functions that decide whether
//! to route a fixed-ρ fit through the CUDA-resident path:
//!
//! - `try_gaussian_pls_gpu` — Gaussian-Identity exact POTRF/POTRS dispatch
//!   (wraps `crate::solver::gpu::pirls_dispatch_wire::try_gpu_gaussian_pls_dispatch`).
//! - `try_pirls_loop_gpu` — general PIRLS-loop device dispatch
//!   (wraps `crate::solver::gpu::pirls_dispatch_wire::try_gpu_pirls_loop_dispatch`).
//!
//! Both functions return `Option<Result<..>>`: `Some(Ok(pair))` on a successful
//! device solve, `Some(Err(..))` on a device error (caller should fall through to
//! CPU), and `None` when the dispatch criteria are not met.
//!
//! The GPU **kernel** bodies live in `crate::solver::gpu::pirls_gpu` and
//! `crate::solver::gpu::pirls_dispatch_wire`; this file only owns the
//! host-side admission logic and struct assembly.

#[cfg(target_os = "linux")]
use super::FIXED_STABILIZATION_RIDGE;
#[cfg(target_os = "linux")]
use super::loop_driver::make_reparam_operator;
use super::{
    GaussianFixedCache, LinearInequalityConstraints, PirlsConfig, PirlsCoordinateFrame,
    PirlsPenalty, PirlsResult, WorkingModelPirlsResult,
};
use crate::construction::ReparamResult;
use crate::estimate::EstimationError;
use crate::matrix::DesignMatrix;
use crate::types::LinkFunction;
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
#[cfg_attr(not(target_os = "linux"), allow(unused_variables))]
pub(super) fn try_gaussian_pls_gpu<F>(
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
) -> Option<Result<(PirlsResult, WorkingModelPirlsResult), EstimationError>>
where
    F: FnOnce() -> Result<ReparamResult, EstimationError>,
{
    #[cfg(target_os = "linux")]
    if matches!(link_function, LinkFunction::Identity)
        && config.likelihood.spec.is_gaussian_identity()
        && !config.firth_bias_reduction
        && penalty_coefficient_lower_bounds.is_none()
        && penalty_linear_constraints_original.is_none()
    {
        use crate::solver::gpu::pirls_dispatch_wire::{
            GpuGaussianPlsInput, try_gpu_gaussian_pls_dispatch,
        };
        if let Some(cache) = gaussian_fixed_cache {
            if let PirlsPenalty::Dense {
                s_transformed,
                linear_shift,
                constant_shift,
                prior_mean_target,
                ..
            } = penalty_active
            {
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
                    linear_shift: linear_shift.view(),
                    prior_mean_target: prior_mean_target.view(),
                    constant_shift: *constant_shift,
                    qs: qs_view,
                    ridge: FIXED_STABILIZATION_RIDGE,
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
                };
                if let Some(result) = try_gpu_gaussian_pls_dispatch(gpu_input) {
                    match result {
                        Ok(pair) => return Some(Ok(pair)),
                        Err(err) => {
                            log::warn!(
                                "[PIRLS GPU Gaussian PLS] device solve error, falling back to CPU: {err}"
                            );
                            // Error logged; fall through to CPU path.
                        }
                    }
                }
            }
        }
    }
    None
}

/// Stage 3.3: Try to route a dense-design PIRLS fit through the CUDA
/// device-resident loop.
///
/// Returns `None` when admission is denied (non-Linux, missing runtime, sparse
/// or Kronecker design, Firth active, constraints present, or shape/family
/// outside the dispatch policy). Returns `Some(Ok(pair))` on success and
/// `Some(Err(..))` on a device error so the caller can fall through to the
/// CPU LM loop.
///
/// `materialize_reparam` is called lazily — only when the admission shim
/// confirms the fit is eligible.
#[cfg_attr(not(target_os = "linux"), allow(unused_variables))]
pub(super) fn try_pirls_loop_gpu<F>(
    config: &PirlsConfig,
    penalty_active: &PirlsPenalty,
    kronecker_runtime_is_none: bool,
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
    #[cfg(target_os = "linux")]
    use super::HessianCurvatureKind;
    #[cfg(target_os = "linux")]
    {
        use crate::solver::gpu::pirls_dispatch_wire::{
            GpuPirlsDispatchInput, try_gpu_pirls_loop_admit, try_gpu_pirls_loop_dispatch,
        };
        let dense_x = x_original.as_dense().map(|d| d.view());
        let dense_penalty = matches!(penalty_active, PirlsPenalty::Dense { .. });
        let no_kronecker = kronecker_runtime_is_none;
        let no_sparse_native = !use_sparse_native;
        let no_firth = !config.firth_bias_reduction;
        let no_constraints = linear_constraints.is_none();
        if let (true, true, true, true, true, Some(x_dense)) = (
            dense_penalty,
            no_kronecker,
            no_sparse_native,
            no_firth,
            no_constraints,
            dense_x,
        ) {
            let n_admit = x_dense.nrows();
            let p_admit = x_dense.ncols();
            if try_gpu_pirls_loop_admit(&config.likelihood, n_admit, p_admit) {
                let qs_view = qs_arc.as_ref().map(|qs| qs.view());
                let (s_transformed_view, linear_shift_view, constant_shift_val) =
                    match penalty_active {
                        PirlsPenalty::Dense {
                            s_transformed,
                            linear_shift,
                            constant_shift,
                            ..
                        } => (s_transformed.view(), linear_shift.view(), *constant_shift),
                        PirlsPenalty::Diagonal { .. } => {
                            // SAFETY: the enclosing dense_penalty admission
                            // ABOVE (matches!(penalty_active, PirlsPenalty::Dense{..}))
                            // restricts execution to the Dense variant; reaching
                            // here means a mid-function mutation has changed
                            // `penalty_active` out from under us, which is a
                            // programming error in this single-threaded function
                            // body. Falling back to None would silently mask the bug.
                            panic!("GPU PIRLS dispatch gated on PirlsPenalty::Dense above")
                        }
                    };
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
                    linear_shift: linear_shift_view,
                    constant_shift: constant_shift_val,
                    y,
                    priorweights,
                    offset,
                    initial_beta: initial_beta_owned.view(),
                    initial_lm_lambda: config.initial_lm_lambda,
                    max_iterations,
                    convergence_tolerance: config.convergence_tolerance,
                    linear_constraints: None,
                    qs: qs_view,
                    reparam_result: reparam_for_dispatch,
                    x_transformed_design,
                    coordinate_frame,
                    edf: None,
                    exported_curvature: exported_curvature_kind,
                };
                if let Some(result) = try_gpu_pirls_loop_dispatch(dispatch) {
                    match result {
                        Ok(pair) => return Some(Ok(pair)),
                        Err(err) => {
                            log::warn!(
                                "[PIRLS GPU dispatch] device loop returned error, falling back to CPU: {err}"
                            );
                            // Error logged; fall through to CPU LM loop.
                        }
                    }
                }
            }
        }
    }
    None
}
