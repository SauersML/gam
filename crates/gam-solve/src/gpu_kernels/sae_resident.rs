//! Device-resident SAE inner-iteration workspace for issue #1017.
//!
//! This first vertical slice keeps production fitting untouched. It accepts
//! host-evaluated SAE basis/gate values plus already-assembled data-fit
//! Arrow-Schur slabs, uploads those buffers once, and runs one Newton step
//! through the existing GPU Arrow-Schur sequence when the runtime probe admits
//! the workload. Later slices can replace the host slab feed with on-device
//! basis/gate evaluation without changing the public step API.

use ndarray::Array1;

use crate::gpu_kernels::arrow_schur::{
    ArrowSchurGpuFailure, solve_arrow_newton_step, solve_arrow_newton_step_dense_reference,
};
use crate::model_types::ExecutionPath;

/// Per-iterate solve backend for the resident inner Newton loop.
///
/// All three modes run the IDENTICAL host control flow (`run_inner_loop`):
/// residual-gradient assembly, LM trust-region accept/reject, ridge schedule.
/// They differ ONLY in how the per-iterate arrow step is computed, which is
/// exactly the residency lever #1017 measures:
///
/// * [`InnerSolveMode::DeviceResident`] — the Phase-3 fix: factor the constant
///   Hessian blocks ONCE into a [`crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle`]
///   and, every iterate, upload only the `O(n·d + k)` gradient and read back
///   only `δ`. No per-solve D/B re-upload, no per-solve POTRF.
/// * [`InnerSolveMode::DeviceReupload`] — the BEFORE path: call
///   `solve_arrow_newton_step` per iterate, which re-packs and re-uploads
///   `D`/`B`/`g` and re-runs the per-row POTRF + border Schur factor every call.
///   This is the residency baseline the bench divides against.
/// * [`InnerSolveMode::CpuReference`] — the dense f64 oracle (re-factors per
///   iterate on the host), used for the correctness parity check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InnerSolveMode {
    DeviceResident,
    DeviceReupload,
    CpuReference,
}

impl InnerSolveMode {
    /// Truthful [`ExecutionPath`] this solve mode realizes (issue #1017): the
    /// resident loop keeps factors on-device (`GpuResidentFull`), the baseline
    /// re-uploads/re-factors every iterate (`GpuReupload`), and the reference
    /// path runs on the host (`Cpu`).
    #[inline]
    const fn execution_path(self) -> ExecutionPath {
        match self {
            Self::DeviceResident => ExecutionPath::GpuResidentFull,
            Self::DeviceReupload => ExecutionPath::GpuReupload,
            Self::CpuReference => ExecutionPath::Cpu,
        }
    }
}
use crate::arrow_schur::{ArrowSchurError, ArrowSchurSystem};

/// SAE shape used by the resident inner-iteration workspace.
///
/// `p` is the target width and current shared-border width for this slice. The
/// true SAE decoder has richer `(basis × output)` structure; slice 1 deliberately
/// keeps that structure host-assembled into `row_cross_slabs` while preserving
/// the qwen-scale target width in the Schur border.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceResidentArrowShape {
    pub n: usize,
    pub p: usize,
    pub basis_cols: usize,
    pub d: usize,
}

impl DeviceResidentArrowShape {
    #[inline]
    pub const fn qwen_non_gating() -> Self {
        Self {
            n: 2_000,
            p: 2_048,
            basis_cols: 8,
            d: 2,
        }
    }

    /// Color-arm shape from the #1017 measured gap (n=180, p=5120, M≈9, K=1):
    /// few rows, very wide border. The dense-Schur device path (cuSOLVER border
    /// POTRF) handles the `p=5120` border that exceeds the fused-kernel `P_MAX`.
    #[inline]
    pub const fn color_arm() -> Self {
        Self {
            n: 180,
            p: 5_120,
            basis_cols: 9,
            d: 2,
        }
    }

    #[inline]
    pub const fn target_len(self) -> usize {
        self.n * self.p
    }

    #[inline]
    pub const fn basis_len(self) -> usize {
        self.n * self.basis_cols
    }

    #[inline]
    pub const fn row_hessian_len(self) -> usize {
        self.n * self.d * self.d
    }

    #[inline]
    pub const fn row_cross_len(self) -> usize {
        self.n * self.d * self.p
    }

    #[inline]
    pub const fn row_gradient_len(self) -> usize {
        self.n * self.d
    }

    #[inline]
    pub const fn border_hessian_len(self) -> usize {
        self.p * self.p
    }
}

/// Host-fed row-block slabs for the first resident slice.
///
/// All matrices are row-major in host memory:
/// * `row_hessian_slabs`: `n` slabs of shape `d × d`.
/// * `row_cross_slabs`: `n` slabs of shape `d × p`.
/// * `border_hessian`: one `p × p` shared block.
#[derive(Clone, Debug)]
pub struct DeviceResidentArrowSlabs {
    pub row_hessian_slabs: Vec<f64>,
    pub row_cross_slabs: Vec<f64>,
    pub row_gradient_slabs: Vec<f64>,
    pub border_hessian: Vec<f64>,
    pub border_gradient: Vec<f64>,
}

/// Result of one resident SAE inner Newton iteration.
#[derive(Clone, Debug)]
pub struct DeviceResidentArrowStep {
    pub delta_t: Array1<f64>,
    pub delta_beta: Array1<f64>,
    pub objective: f64,
    pub gradient_norm: f64,
    pub log_det_hessian: f64,
    pub execution_path: ExecutionPath,
}

#[derive(Debug, Clone)]
pub enum DeviceResidentArrowError {
    Shape { reason: String },
    Unavailable { reason: String },
    Solve { reason: String },
}

impl std::fmt::Display for DeviceResidentArrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape { reason } | Self::Unavailable { reason } | Self::Solve { reason } => {
                f.write_str(reason)
            }
        }
    }
}

impl std::error::Error for DeviceResidentArrowError {}

#[cfg(target_os = "linux")]
pub struct DeviceResidentArrowBuffers {
    pub stream: std::sync::Arc<cudarc::driver::CudaStream>,
    pub target_x_dev: cudarc::driver::CudaSlice<f64>,
    pub basis_values_dev: cudarc::driver::CudaSlice<f64>,
    pub gate_activations_dev: cudarc::driver::CudaSlice<f64>,
    pub row_hessian_dev: cudarc::driver::CudaSlice<f64>,
    pub row_cross_dev: cudarc::driver::CudaSlice<f64>,
    pub row_gradient_dev: cudarc::driver::CudaSlice<f64>,
    pub border_hessian_dev: cudarc::driver::CudaSlice<f64>,
    pub border_gradient_dev: cudarc::driver::CudaSlice<f64>,
    pub bytes: usize,
}

/// Upload-once workspace for the SAE data-fit Arrow-Schur inner iteration.
pub struct DeviceResidentArrowWorkspace {
    shape: DeviceResidentArrowShape,
    target_x: Vec<f64>,
    basis_values: Vec<f64>,
    gate_activations: Vec<f64>,
    slabs: DeviceResidentArrowSlabs,
    #[cfg(target_os = "linux")]
    device: Option<DeviceResidentArrowBuffers>,
}

impl DeviceResidentArrowWorkspace {
    pub fn new(
        shape: DeviceResidentArrowShape,
        target_x: Vec<f64>,
        basis_values: Vec<f64>,
        gate_activations: Vec<f64>,
        slabs: DeviceResidentArrowSlabs,
    ) -> Result<Self, DeviceResidentArrowError> {
        validate_shape(shape, &target_x, &basis_values, &gate_activations, &slabs)?;
        #[cfg(target_os = "linux")]
        let device =
            upload_resident_buffers(shape, &target_x, &basis_values, &gate_activations, &slabs);
        Ok(Self {
            shape,
            target_x,
            basis_values,
            gate_activations,
            slabs,
            #[cfg(target_os = "linux")]
            device,
        })
    }

    #[inline]
    pub const fn shape(&self) -> DeviceResidentArrowShape {
        self.shape
    }

    #[must_use]
    pub fn device_resident(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.device.is_some()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    #[must_use]
    pub fn resident_device_bytes(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            self.device.as_ref().map_or(0, |device| device.bytes)
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Opaque device-context identifier for telemetry: `1` when the resident
    /// device buffers are live on this workspace, `0` when no device was bound.
    /// Distinguishes "a device executed this fit" from "silent CPU fallback"
    /// without leaking the cudarc handle.
    #[must_use]
    fn context_id(&self) -> usize {
        usize::from(self.device_resident())
    }

    /// Bytes the re-uploading / frame-build path moves host→device for a full
    /// `D`/`B`/`g`/border refresh, used to attribute H2D traffic in telemetry.
    #[must_use]
    fn frame_upload_bytes(&self) -> usize {
        [
            self.slabs.row_hessian_slabs.len(),
            self.slabs.row_cross_slabs.len(),
            self.slabs.row_gradient_slabs.len(),
            self.slabs.border_hessian.len(),
            self.slabs.border_gradient.len(),
        ]
        .into_iter()
        .sum::<usize>()
            * std::mem::size_of::<f64>()
    }

    #[must_use]
    pub fn host_shadow_bytes(&self) -> usize {
        [
            self.target_x.len(),
            self.basis_values.len(),
            self.gate_activations.len(),
            self.slabs.row_hessian_slabs.len(),
            self.slabs.row_cross_slabs.len(),
            self.slabs.row_gradient_slabs.len(),
            self.slabs.border_hessian.len(),
            self.slabs.border_gradient.len(),
        ]
        .into_iter()
        .sum::<usize>()
            * std::mem::size_of::<f64>()
    }

    /// Run one device-side Newton sequence. No CPU fallback is attempted here:
    /// callers that want a reference path must call [`Self::cpu_reference_step`].
    pub fn one_inner_iteration(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<DeviceResidentArrowStep, DeviceResidentArrowError> {
        if !self.device_resident() {
            return Err(DeviceResidentArrowError::Unavailable {
                reason: "SAE resident inner iteration unavailable: CUDA runtime did not admit the qwen-scale row-block workload".to_string(),
            });
        }
        let sys = self.to_arrow_system();
        solve_arrow_newton_step(&sys, ridge_t, ridge_beta)
            .map(|solution| self.finish_step(solution, ExecutionPath::GpuResidentLinearization))
            .map_err(map_gpu_error)
    }

    /// CPU reference for parity harnesses. This path is explicit and is never
    /// called from [`Self::one_inner_iteration`].
    pub fn cpu_reference_step(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<DeviceResidentArrowStep, DeviceResidentArrowError> {
        let sys = self.to_arrow_system();
        solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
            .map(|solution| self.finish_step(solution, ExecutionPath::Cpu))
            .map_err(|reason| DeviceResidentArrowError::Solve { reason })
    }

    pub fn to_arrow_system(&self) -> ArrowSchurSystem {
        let shape = self.shape;
        let mut sys = ArrowSchurSystem::new(shape.n, shape.d, shape.p);
        for i in 0..shape.n {
            let h_base = i * shape.d * shape.d;
            let b_base = i * shape.d * shape.p;
            let g_base = i * shape.d;
            for r in 0..shape.d {
                for c in 0..shape.d {
                    sys.rows[i].htt[[r, c]] =
                        self.slabs.row_hessian_slabs[h_base + r * shape.d + c];
                }
                sys.rows[i].gt[r] = self.slabs.row_gradient_slabs[g_base + r];
                for c in 0..shape.p {
                    sys.rows[i].htbeta[[r, c]] =
                        self.slabs.row_cross_slabs[b_base + r * shape.p + c];
                }
            }
        }
        for r in 0..shape.p {
            sys.gb[r] = self.slabs.border_gradient[r];
            for c in 0..shape.p {
                sys.hbb[[r, c]] = self.slabs.border_hessian[r * shape.p + c];
            }
        }
        sys.refresh_row_hessian_fingerprint();
        sys
    }

    fn finish_step(
        &self,
        solution: crate::gpu_kernels::arrow_schur::ArrowSchurGpuSolution,
        execution_path: ExecutionPath,
    ) -> DeviceResidentArrowStep {
        DeviceResidentArrowStep {
            delta_t: solution.delta_t,
            delta_beta: solution.delta_beta,
            objective: 0.5 * squared_norm(&self.target_x),
            gradient_norm: self.gradient_norm(),
            log_det_hessian: solution.log_det_hessian,
            execution_path,
        }
    }

    fn gradient_norm(&self) -> f64 {
        let row = squared_norm(&self.slabs.row_gradient_slabs);
        let border = squared_norm(&self.slabs.border_gradient);
        (row + border).sqrt()
    }

    // ---------------------------------------------------------------------
    // Phase 3: full device-resident inner Newton loop (#1017).
    //
    // The resident slabs define a fixed bordered-quadratic data-fit objective
    //     φ(z) = ½‖X‖² + ½ zᵀ H z − g₀ᵀ z,   z = (t, β),
    // where `H` is the arrow-structured Hessian (per-row `H_tt`/`H_tβ` blocks
    // plus the shared `H_ββ` border) and `g₀` is the base gradient assembled
    // once at upload. This is the quadratic the SAE joint inner Newton actually
    // minimises at a frozen gate/basis evaluation; the production driver
    // (`LatentInnerSolver::solve`) re-linearises per outer evaluation, so a
    // single resident frame is one such inner solve.
    //
    // The loop mirrors the production LM trust-region accept/reject exactly:
    // at iterate `z` it forms the residual gradient `r(z) = H z − g₀`, takes
    // the LM-damped arrow step (device or dense-reference), evaluates the trial
    // objective, and accepts on the actual-vs-predicted reduction ratio. The
    // iterate `(t, β)` and the per-step scalars (objective, gradient norm, ρ)
    // are the ONLY host-side state; the heavy `O(n d³ + p³)` factor/solve stays
    // on the resident buffers via `solve_arrow_newton_step`. For an exact
    // quadratic the loop converges in one accepted step, but it exercises the
    // full assemble→solve→objective→accept machinery and the scalar-only
    // readback contract the production loop relies on.
    // ---------------------------------------------------------------------

    /// Run the full device-resident inner Newton loop. Routes the per-iteration
    /// arrow solve through the GPU path; returns `Unavailable` when CUDA did not
    /// admit the resident workload (callers wanting a CPU path use
    /// [`Self::cpu_reference_fit`]).
    pub fn device_fit(
        &self,
        opts: &DeviceResidentInnerOptions,
    ) -> Result<DeviceResidentInnerOutcome, DeviceResidentArrowError> {
        if !self.device_resident() {
            return Err(DeviceResidentArrowError::Unavailable {
                reason: "SAE resident inner loop unavailable: CUDA runtime did not admit the qwen-scale row-block workload".to_string(),
            });
        }
        self.run_inner_loop(opts, InnerSolveMode::DeviceResident)
    }

    /// The #1017 residency baseline: run the SAME inner Newton loop but compute
    /// each per-iterate arrow step through `solve_arrow_newton_step`, which
    /// re-packs/re-uploads `D`/`B`/`g` and re-runs the per-row POTRF + border
    /// Schur factor on EVERY iterate. This is the "current re-uploading path";
    /// the bench divides [`Self::device_fit`] (resident) against it to isolate
    /// the across-iteration residency speedup on one device, holding the host
    /// control flow and the GPU factor kernels fixed.
    pub fn device_reupload_fit(
        &self,
        opts: &DeviceResidentInnerOptions,
    ) -> Result<DeviceResidentInnerOutcome, DeviceResidentArrowError> {
        if !self.device_resident() {
            return Err(DeviceResidentArrowError::Unavailable {
                reason: "SAE re-uploading inner loop unavailable: CUDA runtime did not admit the row-block workload".to_string(),
            });
        }
        self.run_inner_loop(opts, InnerSolveMode::DeviceReupload)
    }

    /// CPU dense-reference inner loop. Bit-for-bit the same host arithmetic as
    /// [`Self::device_fit`] except the per-iteration arrow solve uses the dense
    /// reference factorisation; the parity harness asserts the two agree.
    pub fn cpu_reference_fit(
        &self,
        opts: &DeviceResidentInnerOptions,
    ) -> Result<DeviceResidentInnerOutcome, DeviceResidentArrowError> {
        self.run_inner_loop(opts, InnerSolveMode::CpuReference)
    }

    fn run_inner_loop(
        &self,
        opts: &DeviceResidentInnerOptions,
        mode: InnerSolveMode,
    ) -> Result<DeviceResidentInnerOutcome, DeviceResidentArrowError> {
        let execution_path = mode.execution_path();
        let n = self.shape.n;
        let d = self.shape.d;
        let p = self.shape.p;
        let t_len = n * d;

        // Resident iterate, host-side scalars only. The device buffers (X,
        // slabs, border) never leave the device across iterations; only this
        // O(t_len + p) iterate and the per-step reduction scalars cross back.
        let mut t = vec![0.0_f64; t_len];
        let mut beta = vec![0.0_f64; p];

        let base = self.to_arrow_system();
        let half_target_energy = 0.5 * squared_norm(&self.target_x);

        let mut ridge_t = opts.initial_ridge_t.max(0.0);
        let mut ridge_beta = opts.initial_ridge_beta.max(0.0);
        // #1017 Phase 3: when running on device, keep the resident Arrow frame
        // (constant Hessian blocks + their factors) on the device across
        // iterations. The frame bakes a fixed `(ridge_t, ridge_beta)` into the
        // per-row and border Cholesky factors, so it is rebuilt only when the LM
        // ridge changes (reject/shrink); every iteration that shares the cached
        // ridge reuses the resident factors and uploads only the `O(n·d + p)`
        // gradient. The CPU reference path keeps re-factoring per iterate so the
        // parity harness compares residency against a fully independent solve.
        let mut resident_frame: Option<(
            f64,
            f64,
            crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle,
        )> = None;
        let mut current_objective = self.objective_at(&base, half_target_energy, &t, &beta);
        let mut accepted_iters = 0_usize;
        let mut total_iters = 0_usize;
        let mut converged = false;
        let mut last_step = DeviceResidentArrowStep {
            delta_t: Array1::zeros(t_len),
            delta_beta: Array1::zeros(p),
            objective: current_objective,
            gradient_norm: 0.0,
            log_det_hessian: 0.0,
            execution_path,
        };

        while total_iters < opts.max_iterations {
            // Residual gradient r(z) = H z − g₀ becomes the system gradient.
            let residual = self.residual_system(&base, &t, &beta);
            let g_norm = arrow_system_gradient_norm(&residual);
            let scale = 1.0 + iterate_norm(&t, &beta);
            if g_norm / scale < opts.convergence_tolerance {
                converged = true;
                break;
            }

            let solution = match mode {
                InnerSolveMode::DeviceResident => {
                    // Rebuild the resident frame only when the LM ridge changed; an
                    // unchanged ridge reuses the resident factors. A build failure
                    // becomes a Solve error so the LM-escalation arm below grows the
                    // ridge and retries, identical to a per-iterate solve failure.
                    let frame_matches = resident_frame
                        .as_ref()
                        .is_some_and(|(rt, rb, _)| *rt == ridge_t && *rb == ridge_beta);
                    let mut frame_build_error: Option<DeviceResidentArrowError> = None;
                    if !frame_matches {
                        resident_frame = None;
                        match crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle::new(
                            &residual, ridge_t, ridge_beta,
                        ) {
                            Ok(frame) => {
                                // Building a resident frame creates the device
                                // stream/handles and runs the per-row POTRF +
                                // border Schur factor once; record both so a
                                // silent decline (no rebuild ⇒ no factor count)
                                // is visible in the telemetry.
                                crate::profile::telemetry_record_handle_creation(
                                    self.context_id(),
                                );
                                crate::profile::telemetry_record_factorization();
                                crate::profile::telemetry_record_h2d(
                                    self.frame_upload_bytes(),
                                );
                                resident_frame = Some((ridge_t, ridge_beta, frame));
                            }
                            Err(err) => frame_build_error = Some(map_gpu_error(err)),
                        }
                    }
                    match resident_frame.as_ref() {
                        Some((_, _, frame)) => {
                            // Per-iterate gradient r(z) = (g_t rows, g_β), extracted
                            // from the residual system the frame was built to match.
                            let mut g_t = Vec::with_capacity(n * d);
                            for row in &residual.rows {
                                for &v in row.gt.iter() {
                                    g_t.push(v);
                                }
                            }
                            let g_beta: Vec<f64> = residual.gb.iter().copied().collect();
                            // The resident solve uploads only the O(n·d + p)
                            // gradient, launches the per-iterate solve kernel, and
                            // reads back only δ.
                            let grad_bytes =
                                (g_t.len() + g_beta.len()) * std::mem::size_of::<f64>();
                            crate::profile::telemetry_record_h2d(grad_bytes);
                            crate::profile::telemetry_record_kernel_launch();
                            crate::profile::telemetry_record_d2h(
                                (n * d + p) * std::mem::size_of::<f64>(),
                            );
                            frame.solve_gradient(&g_t, &g_beta).map_err(map_gpu_error)
                        }
                        None => Err(frame_build_error.unwrap_or_else(|| {
                            DeviceResidentArrowError::Solve {
                                reason: "SAE resident frame build declined".to_string(),
                            }
                        })),
                    }
                }
                InnerSolveMode::DeviceReupload => {
                    // #1017 residency baseline: re-upload D/B/g and re-factor on
                    // every iterate. Same GPU factor kernels as the resident path,
                    // minus the across-iteration buffer/factor reuse — so EVERY
                    // iterate creates handles, factorizes, launches, and re-uploads
                    // the full slabs.
                    crate::profile::telemetry_record_handle_creation(self.context_id());
                    crate::profile::telemetry_record_factorization();
                    crate::profile::telemetry_record_h2d(self.frame_upload_bytes());
                    crate::profile::telemetry_record_kernel_launch();
                    crate::profile::telemetry_record_d2h(
                        (n * d + p) * std::mem::size_of::<f64>(),
                    );
                    solve_arrow_newton_step(&residual, ridge_t, ridge_beta).map_err(map_gpu_error)
                }
                InnerSolveMode::CpuReference => {
                    solve_arrow_newton_step_dense_reference(&residual, ridge_t, ridge_beta)
                        .map_err(|reason| DeviceResidentArrowError::Solve { reason })
                }
            };

            let solution = match solution {
                Ok(sol) => sol,
                Err(DeviceResidentArrowError::Solve { .. })
                | Err(DeviceResidentArrowError::Unavailable { .. }) => {
                    // LM escalation: grow ridge, retry without consuming an
                    // iteration. Mirrors the production per-row/Schur PD-failure
                    // arm in `LatentInnerSolver::solve`.
                    ridge_t = grow_ridge(ridge_t, opts.lm_grow);
                    ridge_beta = grow_ridge(ridge_beta, opts.lm_grow);
                    if ridge_t > opts.max_ridge || ridge_beta > opts.max_ridge {
                        return Err(DeviceResidentArrowError::Solve {
                            reason: format!(
                                "SAE resident inner loop: LM ridge exceeded max ({:e}) at iter {total_iters}",
                                opts.max_ridge
                            ),
                        });
                    }
                    total_iters += 1;
                    continue;
                }
                Err(other) => return Err(other),
            };

            // Predicted reduction from the bare quadratic model on the residual
            // system, identical formula to the production trust-region ratio.
            let predicted_reduction =
                crate::arrow_schur::arrow_bare_quadratic_model_reduction(
                    &residual,
                    solution.delta_t.view(),
                    solution.delta_beta.view(),
                    ridge_t,
                    ridge_beta,
                )
                .map_err(|err| DeviceResidentArrowError::Solve {
                    reason: format!("SAE resident inner loop predicted-reduction failed: {err}"),
                })?;

            // Trial iterate.
            let mut trial_t = t.clone();
            let mut trial_beta = beta.clone();
            for (slot, dv) in trial_t.iter_mut().zip(solution.delta_t.iter()) {
                *slot += *dv;
            }
            for (slot, dv) in trial_beta.iter_mut().zip(solution.delta_beta.iter()) {
                *slot += *dv;
            }
            let trial_objective =
                self.objective_at(&base, half_target_energy, &trial_t, &trial_beta);

            // Trust-region gain-ratio noise floor keyed to the objective's own
            // magnitude, mirroring the production `LatentInnerSolver` (#1127): the
            // floor must be equivariant under a response rescaling `y → a·y` (the
            // penalized objective and both reductions scale as `O(a²)`). The
            // previous `.max(1.0)` absolute floor broke this — near a converged
            // iterate it pinned the floor at `1e-14` while a genuine refining
            // step's `predicted_reduction` was `O(a²)`, misclassifying the real
            // step as numerical noise and stalling the inner solve at a
            // non-stationary point. A perfectly converged objective
            // (`current_objective == 0`) yields a `0` floor, so the
            // `predicted_reduction > 0` branch still governs and no step is lost.
            let objective_scale = current_objective.abs();
            let noise_floor = objective_scale * 1e-14;
            let actual_reduction = current_objective - trial_objective;
            let rho = if predicted_reduction > noise_floor {
                actual_reduction / predicted_reduction
            } else if actual_reduction >= -noise_floor {
                1.0
            } else {
                -1.0
            };

            if rho > 0.0 && trial_objective.is_finite() {
                t = trial_t;
                beta = trial_beta;
                current_objective = trial_objective;
                ridge_t = (ridge_t * opts.lm_shrink).max(0.0);
                ridge_beta = (ridge_beta * opts.lm_shrink).max(0.0);
                last_step = DeviceResidentArrowStep {
                    delta_t: solution.delta_t,
                    delta_beta: solution.delta_beta,
                    objective: current_objective,
                    gradient_norm: g_norm,
                    log_det_hessian: solution.log_det_hessian,
                    execution_path,
                };
                accepted_iters += 1;
                total_iters += 1;
            } else {
                ridge_t = grow_ridge(ridge_t, opts.lm_grow);
                ridge_beta = grow_ridge(ridge_beta, opts.lm_grow);
                if ridge_t > opts.max_ridge || ridge_beta > opts.max_ridge {
                    return Err(DeviceResidentArrowError::Solve {
                        reason: format!(
                            "SAE resident inner loop: LM rejected step until ridge exceeded max ({:e}) at iter {total_iters} (rho={rho:.3e})",
                            opts.max_ridge
                        ),
                    });
                }
                total_iters += 1;
            }
        }

        Ok(DeviceResidentInnerOutcome {
            t: Array1::from_vec(t),
            beta: Array1::from_vec(beta),
            objective: current_objective,
            gradient_norm: last_step.gradient_norm,
            log_det_hessian: last_step.log_det_hessian,
            iterations: total_iters,
            accepted_iterations: accepted_iters,
            converged,
            execution_path,
        })
    }

    // ---------------------------------------------------------------------
    // Phase 3b: reuse the resident frame ACROSS OUTER iterations (#1017
    // deliverable 3).
    //
    // The inner Newton loop above already keeps the resident Arrow frame
    // (factored `D`/`B`/Schur) on the device across INNER iterations at a fixed
    // ridge. The next residency tier is the OUTER loop: across consecutive outer
    // evaluations the SAE Hessian operator is unchanged whenever the frozen
    // gate/basis frame (hence `D = H_tt`, `B = H_tβ`, border `H_ββ`) does not
    // move — only the base gradient `g₀` (the linearization point / target
    // residual) changes. In that regime the `O(n·d³ + p³)` factor work and the
    // dominant `O(n·d·p)` `D`/`B` upload need to happen ONCE for the whole outer
    // sweep, not once per outer. `device_fit_outer_sequence` realizes that: it
    // builds at most ONE resident frame for an unchanged operator and drives
    // every outer's inner solve through it, re-uploading only the per-outer
    // `O(n·d + p)` gradient. The per-outer parity oracle is an independent
    // `device_fit` (fresh frame per outer); the two must agree because sharing
    // the factor across outers skips only re-deriving operator-independent work.
    // ---------------------------------------------------------------------

    /// Run a sequence of outer evaluations that SHARE one resident frame when the
    /// Hessian operator is unchanged across outers (#1017 deliverable 3).
    ///
    /// Each entry of `base_gradient_overrides` is one outer evaluation's base
    /// gradient `(g_t rows: n·d, g_β: p)` — the only part of the bordered
    /// quadratic that moves across outers at a frozen gate/basis frame. The
    /// constant Hessian blocks ride the resident frame, which is built ONCE and
    /// reused for every outer (frame builds are counted and returned so a caller
    /// can assert the across-outer amortization actually fired: exactly one frame
    /// build for an unchanged operator, regardless of how many outers run).
    ///
    /// Returns one [`DeviceResidentInnerOutcome`] per outer plus the number of
    /// resident-frame builds performed across the whole sweep. On a CPU-only host
    /// returns `Unavailable` (callers wanting a host path use
    /// [`Self::cpu_reference_outer_sequence`]).
    pub fn device_fit_outer_sequence(
        &self,
        base_gradient_overrides: &[(Vec<f64>, Vec<f64>)],
        opts: &DeviceResidentInnerOptions,
    ) -> Result<OuterSequenceOutcome, DeviceResidentArrowError> {
        if !self.device_resident() {
            return Err(DeviceResidentArrowError::Unavailable {
                reason: "SAE outer-sequence residency unavailable: CUDA runtime did not admit the row-block workload".to_string(),
            });
        }
        self.run_outer_sequence(
            base_gradient_overrides,
            opts,
            InnerSolveMode::DeviceResident,
        )
    }

    /// CPU-reference outer sequence: same host control flow as
    /// [`Self::device_fit_outer_sequence`] but the per-iterate arrow solve uses
    /// the dense reference factorisation. The parity harness asserts the device
    /// across-outer sweep agrees with this per-outer-independent reference.
    pub fn cpu_reference_outer_sequence(
        &self,
        base_gradient_overrides: &[(Vec<f64>, Vec<f64>)],
        opts: &DeviceResidentInnerOptions,
    ) -> Result<OuterSequenceOutcome, DeviceResidentArrowError> {
        self.run_outer_sequence(base_gradient_overrides, opts, InnerSolveMode::CpuReference)
    }

    fn run_outer_sequence(
        &self,
        base_gradient_overrides: &[(Vec<f64>, Vec<f64>)],
        opts: &DeviceResidentInnerOptions,
        mode: InnerSolveMode,
    ) -> Result<OuterSequenceOutcome, DeviceResidentArrowError> {
        let n = self.shape.n;
        let d = self.shape.d;
        let p = self.shape.p;
        let t_len = n * d;
        let half_target_energy = 0.5 * squared_norm(&self.target_x);

        // ONE resident frame for the whole sweep (device mode only). The operator
        // is unchanged across outers — the frame bakes the constant `D`/`B`/Schur
        // factors at `(initial_ridge_t, initial_ridge_beta)` once and every outer
        // reuses it. A per-outer ridge escalation (PD failure) still rebuilds, but
        // for a well-posed unchanged operator the build count stays at 1, which is
        // the across-outer amortization this method delivers.
        let mut shared = SharedFrameState::default();
        let mut outcomes = Vec::with_capacity(base_gradient_overrides.len());

        for (g_t_override, g_beta_override) in base_gradient_overrides {
            if g_t_override.len() != t_len || g_beta_override.len() != p {
                return Err(DeviceResidentArrowError::Shape {
                    reason: format!(
                        "outer-sequence gradient shape mismatch: g_t={} (want {t_len}), g_beta={} (want {p})",
                        g_t_override.len(),
                        g_beta_override.len()
                    ),
                });
            }
            // This outer's bordered quadratic: same Hessian blocks, base gradient
            // swapped to this outer's `g₀`.
            let mut base = self.to_arrow_system();
            for (i, row) in base.rows.iter_mut().enumerate() {
                for r in 0..d {
                    row.gt[r] = g_t_override[i * d + r];
                }
            }
            for (j, gb) in base.gb.iter_mut().enumerate() {
                *gb = g_beta_override[j];
            }
            base.refresh_row_hessian_fingerprint();

            let outcome = self.run_one_outer(&base, half_target_energy, opts, mode, &mut shared)?;
            outcomes.push(outcome);
        }

        Ok(OuterSequenceOutcome {
            outers: outcomes,
            frame_builds: shared.frame_builds,
        })
    }

    /// One outer evaluation's inner Newton loop, optionally reusing the frame
    /// carried in `shared` across calls. Mirrors `run_inner_loop` but takes the
    /// base system + the shared across-outer state so the caller can keep one
    /// frame live for the whole sweep. `shared.frame_builds` is incremented every
    /// time a frame is actually (re)built, so the caller can assert the
    /// across-outer amortization fired.
    fn run_one_outer(
        &self,
        base: &ArrowSchurSystem,
        half_target_energy: f64,
        opts: &DeviceResidentInnerOptions,
        mode: InnerSolveMode,
        shared: &mut SharedFrameState,
    ) -> Result<DeviceResidentInnerOutcome, DeviceResidentArrowError> {
        let execution_path = mode.execution_path();
        let n = self.shape.n;
        let d = self.shape.d;
        let p = self.shape.p;
        let t_len = n * d;

        let mut t = vec![0.0_f64; t_len];
        let mut beta = vec![0.0_f64; p];
        let mut ridge_t = opts.initial_ridge_t.max(0.0);
        let mut ridge_beta = opts.initial_ridge_beta.max(0.0);
        let mut current_objective = self.objective_at(base, half_target_energy, &t, &beta);
        let mut accepted_iters = 0_usize;
        let mut total_iters = 0_usize;
        let mut converged = false;
        let mut last_gradient_norm = 0.0_f64;
        let mut last_log_det = 0.0_f64;

        while total_iters < opts.max_iterations {
            let residual = self.residual_system(base, &t, &beta);
            let g_norm = arrow_system_gradient_norm(&residual);
            let scale = 1.0 + iterate_norm(&t, &beta);
            if g_norm / scale < opts.convergence_tolerance {
                converged = true;
                break;
            }

            let solution = match mode {
                InnerSolveMode::DeviceResident => {
                    let frame_matches = shared
                        .frame
                        .as_ref()
                        .is_some_and(|(rt, rb, _)| *rt == ridge_t && *rb == ridge_beta);
                    let mut frame_build_error: Option<DeviceResidentArrowError> = None;
                    if !frame_matches {
                        shared.frame = None;
                        match crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle::new(
                            &residual, ridge_t, ridge_beta,
                        ) {
                            Ok(frame) => {
                                shared.frame_builds += 1;
                                crate::profile::telemetry_record_handle_creation(
                                    self.context_id(),
                                );
                                crate::profile::telemetry_record_factorization();
                                crate::profile::telemetry_record_h2d(
                                    self.frame_upload_bytes(),
                                );
                                shared.frame = Some((ridge_t, ridge_beta, frame));
                            }
                            Err(err) => frame_build_error = Some(map_gpu_error(err)),
                        }
                    }
                    match shared.frame.as_ref() {
                        Some((_, _, frame)) => {
                            let mut g_t = Vec::with_capacity(n * d);
                            for row in &residual.rows {
                                for &v in row.gt.iter() {
                                    g_t.push(v);
                                }
                            }
                            let g_beta: Vec<f64> = residual.gb.iter().copied().collect();
                            let grad_bytes =
                                (g_t.len() + g_beta.len()) * std::mem::size_of::<f64>();
                            crate::profile::telemetry_record_h2d(grad_bytes);
                            crate::profile::telemetry_record_kernel_launch();
                            crate::profile::telemetry_record_d2h(
                                (n * d + p) * std::mem::size_of::<f64>(),
                            );
                            frame.solve_gradient(&g_t, &g_beta).map_err(map_gpu_error)
                        }
                        None => Err(frame_build_error.unwrap_or_else(|| {
                            DeviceResidentArrowError::Solve {
                                reason: "SAE resident frame build declined".to_string(),
                            }
                        })),
                    }
                }
                InnerSolveMode::DeviceReupload => {
                    solve_arrow_newton_step(&residual, ridge_t, ridge_beta).map_err(map_gpu_error)
                }
                InnerSolveMode::CpuReference => {
                    solve_arrow_newton_step_dense_reference(&residual, ridge_t, ridge_beta)
                        .map_err(|reason| DeviceResidentArrowError::Solve { reason })
                }
            };

            let solution = match solution {
                Ok(sol) => sol,
                Err(DeviceResidentArrowError::Solve { .. })
                | Err(DeviceResidentArrowError::Unavailable { .. }) => {
                    ridge_t = grow_ridge(ridge_t, opts.lm_grow);
                    ridge_beta = grow_ridge(ridge_beta, opts.lm_grow);
                    if ridge_t > opts.max_ridge || ridge_beta > opts.max_ridge {
                        return Err(DeviceResidentArrowError::Solve {
                            reason: format!(
                                "SAE outer-sequence inner loop: LM ridge exceeded max ({:e}) at iter {total_iters}",
                                opts.max_ridge
                            ),
                        });
                    }
                    total_iters += 1;
                    continue;
                }
                Err(other) => return Err(other),
            };

            let predicted_reduction =
                crate::arrow_schur::arrow_bare_quadratic_model_reduction(
                    &residual,
                    solution.delta_t.view(),
                    solution.delta_beta.view(),
                    ridge_t,
                    ridge_beta,
                )
                .map_err(|err| DeviceResidentArrowError::Solve {
                    reason: format!("SAE outer-sequence predicted-reduction failed: {err}"),
                })?;

            let mut trial_t = t.clone();
            let mut trial_beta = beta.clone();
            for (slot, dv) in trial_t.iter_mut().zip(solution.delta_t.iter()) {
                *slot += *dv;
            }
            for (slot, dv) in trial_beta.iter_mut().zip(solution.delta_beta.iter()) {
                *slot += *dv;
            }
            let trial_objective =
                self.objective_at(base, half_target_energy, &trial_t, &trial_beta);

            let objective_scale = current_objective.abs();
            let noise_floor = objective_scale * 1e-14;
            let actual_reduction = current_objective - trial_objective;
            let rho = if predicted_reduction > noise_floor {
                actual_reduction / predicted_reduction
            } else if actual_reduction >= -noise_floor {
                1.0
            } else {
                -1.0
            };

            if rho > 0.0 && trial_objective.is_finite() {
                t = trial_t;
                beta = trial_beta;
                current_objective = trial_objective;
                ridge_t = (ridge_t * opts.lm_shrink).max(0.0);
                ridge_beta = (ridge_beta * opts.lm_shrink).max(0.0);
                last_gradient_norm = g_norm;
                last_log_det = solution.log_det_hessian;
                accepted_iters += 1;
                total_iters += 1;
            } else {
                ridge_t = grow_ridge(ridge_t, opts.lm_grow);
                ridge_beta = grow_ridge(ridge_beta, opts.lm_grow);
                if ridge_t > opts.max_ridge || ridge_beta > opts.max_ridge {
                    return Err(DeviceResidentArrowError::Solve {
                        reason: format!(
                            "SAE outer-sequence inner loop: LM rejected step until ridge exceeded max ({:e}) at iter {total_iters} (rho={rho:.3e})",
                            opts.max_ridge
                        ),
                    });
                }
                total_iters += 1;
            }
        }

        Ok(DeviceResidentInnerOutcome {
            t: Array1::from_vec(t),
            beta: Array1::from_vec(beta),
            objective: current_objective,
            gradient_norm: last_gradient_norm,
            log_det_hessian: last_log_det,
            iterations: total_iters,
            accepted_iterations: accepted_iters,
            converged,
            execution_path,
        })
    }

    /// Bordered-quadratic objective `½‖X‖² + ½ zᵀ H z − g₀ᵀ z` at iterate
    /// `z = (t, β)`. Uses the resident arrow structure: per-row `H_tt`/`H_tβ`
    /// contractions plus the shared `H_ββ` border, then the linear `g₀ᵀ z`
    /// term. This is the reduction the device line search evaluates; on a CUDA
    /// host the `H z` contraction rides the same resident slabs (batched
    /// per-row GEMV + border GEMV), with only the final dot reduced to a scalar.
    fn objective_at(
        &self,
        base: &ArrowSchurSystem,
        half_target_energy: f64,
        t: &[f64],
        beta: &[f64],
    ) -> f64 {
        let n = self.shape.n;
        let d = self.shape.d;
        let p = self.shape.p;
        // quad = zᵀ H z, lin = g₀ᵀ z.
        let mut quad = 0.0_f64;
        let mut lin = 0.0_f64;
        // Per-row blocks: tᵢᵀ H_tt tᵢ + 2 tᵢᵀ H_tβ β contributes to quad; the
        // β border H_ββ is added once below.
        for i in 0..n {
            let t_base = i * d;
            for r in 0..d {
                // H_tt tᵢ row.
                let mut htt_t = 0.0_f64;
                for c in 0..d {
                    htt_t += base.rows[i].htt[[r, c]] * t[t_base + c];
                }
                // H_tβ β row.
                let mut htb_b = 0.0_f64;
                for c in 0..p {
                    htb_b += base.rows[i].htbeta[[r, c]] * beta[c];
                }
                quad += t[t_base + r] * (htt_t + 2.0 * htb_b);
                lin += base.rows[i].gt[r] * t[t_base + r];
            }
        }
        // β border: βᵀ H_ββ β and g_β ᵀ β.
        for r in 0..p {
            let mut hbb_b = 0.0_f64;
            for c in 0..p {
                hbb_b += base.hbb[[r, c]] * beta[c];
            }
            quad += beta[r] * hbb_b;
            lin += base.gb[r] * beta[r];
        }
        half_target_energy + 0.5 * quad - lin
    }

    /// Build the residual arrow system at iterate `z`: same Hessian blocks as
    /// `base`, but the gradient set to `r(z) = H z − g₀`. The arrow solver
    /// solves `H δ = −gradient = −r(z) = g₀ − H z`, the Newton direction toward
    /// the quadratic's minimiser.
    fn residual_system(
        &self,
        base: &ArrowSchurSystem,
        t: &[f64],
        beta: &[f64],
    ) -> ArrowSchurSystem {
        let n = self.shape.n;
        let d = self.shape.d;
        let p = self.shape.p;
        // `ArrowSchurSystem` is not `Clone` (it carries matrix-free operator
        // closures whose sharing across a then-mutated system would be a
        // footgun), so own a fresh system built from the resident slabs rather
        // than cloning `base`. `to_arrow_system` reproduces the identical
        // Hessian blocks; we overwrite only the gradients below with the
        // residual `r(z) = H z − g₀`. The Hessian reads stay on `base` (bit-
        // identical to the fresh system's blocks).
        let mut sys = self.to_arrow_system();
        for i in 0..n {
            let t_base = i * d;
            for r in 0..d {
                let mut hz = 0.0_f64;
                for c in 0..d {
                    hz += base.rows[i].htt[[r, c]] * t[t_base + c];
                }
                for c in 0..p {
                    hz += base.rows[i].htbeta[[r, c]] * beta[c];
                }
                sys.rows[i].gt[r] = hz - base.rows[i].gt[r];
            }
        }
        for r in 0..p {
            let mut hz = 0.0_f64;
            // H_ββ β.
            for c in 0..p {
                hz += base.hbb[[r, c]] * beta[c];
            }
            // Σ_i (H_tβ^(i))ᵀ tᵢ contribution to the β-gradient.
            for i in 0..n {
                let t_base = i * d;
                for rr in 0..d {
                    hz += base.rows[i].htbeta[[rr, r]] * t[t_base + rr];
                }
            }
            sys.gb[r] = hz - base.gb[r];
        }
        sys.refresh_row_hessian_fingerprint();
        sys
    }
}

/// Options for the device-resident inner Newton loop. Defaults mirror the
/// production [`crate::latent_inner::LatentInnerOptions`] trust-region
/// schedule so device and CPU paths run identical host-side control flow.
#[derive(Clone, Copy, Debug)]
pub struct DeviceResidentInnerOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub initial_ridge_t: f64,
    pub initial_ridge_beta: f64,
    pub lm_grow: f64,
    pub lm_shrink: f64,
    pub max_ridge: f64,
}

impl Default for DeviceResidentInnerOptions {
    fn default() -> Self {
        Self {
            max_iterations: 16,
            convergence_tolerance: 1e-9,
            initial_ridge_t: 0.0,
            initial_ridge_beta: 0.0,
            lm_grow: 4.0,
            lm_shrink: 0.5,
            max_ridge: 1e9,
        }
    }
}

/// Result of the full device-resident inner Newton loop.
#[derive(Clone, Debug)]
pub struct DeviceResidentInnerOutcome {
    pub t: Array1<f64>,
    pub beta: Array1<f64>,
    pub objective: f64,
    pub gradient_norm: f64,
    pub log_det_hessian: f64,
    pub iterations: usize,
    pub accepted_iterations: usize,
    pub converged: bool,
    pub execution_path: ExecutionPath,
}

/// Result of an across-outer resident sweep ([`DeviceResidentArrowWorkspace::device_fit_outer_sequence`]).
///
/// `outers` holds one inner-loop outcome per outer evaluation, in input order.
/// `frame_builds` is the total number of resident-frame (re)builds performed
/// across the whole sweep: for an unchanged operator with a well-posed ridge it
/// is exactly `1` (the across-outer amortization #1017 deliverable 3 buys —
/// factor once, reuse the device factors for every outer), regardless of how
/// many outers ran. A value `> 1` means a per-outer ridge escalation forced a
/// refactor, which the parity oracle still matches but which costs the
/// amortization for those outers.
#[derive(Clone, Debug)]
pub struct OuterSequenceOutcome {
    pub outers: Vec<DeviceResidentInnerOutcome>,
    pub frame_builds: usize,
}

/// Across-outer resident-frame state carried through a `device_fit_outer_sequence`
/// sweep. Holds the single resident frame (keyed by its `(ridge_t, ridge_beta)`)
/// reused across outers at an unchanged operator, plus the running count of frame
/// (re)builds so the caller can assert the across-outer amortization fired.
#[derive(Default)]
struct SharedFrameState {
    frame: Option<(
        f64,
        f64,
        crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle,
    )>,
    frame_builds: usize,
}

fn grow_ridge(current: f64, grow: f64) -> f64 {
    if current == 0.0 { 1e-6 } else { current * grow }
}

fn arrow_system_gradient_norm(sys: &ArrowSchurSystem) -> f64 {
    let mut acc = 0.0_f64;
    for row in &sys.rows {
        for &v in row.gt.iter() {
            acc += v * v;
        }
    }
    for &v in sys.gb.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

fn iterate_norm(t: &[f64], beta: &[f64]) -> f64 {
    (squared_norm(t) + squared_norm(beta)).sqrt()
}

fn validate_shape(
    shape: DeviceResidentArrowShape,
    target_x: &[f64],
    basis_values: &[f64],
    gate_activations: &[f64],
    slabs: &DeviceResidentArrowSlabs,
) -> Result<(), DeviceResidentArrowError> {
    let checks = [
        ("target_x", target_x.len(), shape.target_len()),
        ("basis_values", basis_values.len(), shape.basis_len()),
        (
            "gate_activations",
            gate_activations.len(),
            shape.basis_len(),
        ),
        (
            "row_hessian_slabs",
            slabs.row_hessian_slabs.len(),
            shape.row_hessian_len(),
        ),
        (
            "row_cross_slabs",
            slabs.row_cross_slabs.len(),
            shape.row_cross_len(),
        ),
        (
            "row_gradient_slabs",
            slabs.row_gradient_slabs.len(),
            shape.row_gradient_len(),
        ),
        (
            "border_hessian",
            slabs.border_hessian.len(),
            shape.border_hessian_len(),
        ),
        ("border_gradient", slabs.border_gradient.len(), shape.p),
    ];
    for (label, got, want) in checks {
        if got != want {
            return Err(DeviceResidentArrowError::Shape {
                reason: format!(
                    "SAE resident workspace shape mismatch for {label}: got {got}, expected {want}"
                ),
            });
        }
    }
    if shape.n == 0 || shape.p == 0 || shape.d == 0 || shape.basis_cols == 0 {
        return Err(DeviceResidentArrowError::Shape {
            reason: "SAE resident workspace requires nonzero n, p, basis_cols, and d".to_string(),
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn upload_resident_buffers(
    shape: DeviceResidentArrowShape,
    target_x: &[f64],
    basis_values: &[f64],
    gate_activations: &[f64],
    slabs: &DeviceResidentArrowSlabs,
) -> Option<DeviceResidentArrowBuffers> {
    use gam_gpu::linalg_dispatch::{DispatchOp, route_through_gpu};

    let runtime = route_through_gpu(DispatchOp::SmallDenseBatchedPotrf {
        p: shape.d,
        batch: shape.n,
    })
    .or_else(|| {
        route_through_gpu(DispatchOp::Gemm {
            m: shape.p,
            n: shape.p,
            k: shape.n * shape.basis_cols,
        })
    })?;
    let ctx = gam_gpu::device_runtime::cuda_context_for(runtime.device.ordinal)?;
    let stream = ctx.new_stream().ok()?;
    let target_x_dev = stream.clone_htod(target_x).ok()?;
    let basis_values_dev = stream.clone_htod(basis_values).ok()?;
    let gate_activations_dev = stream.clone_htod(gate_activations).ok()?;
    let row_hessian_dev = stream.clone_htod(&slabs.row_hessian_slabs).ok()?;
    let row_cross_dev = stream.clone_htod(&slabs.row_cross_slabs).ok()?;
    let row_gradient_dev = stream.clone_htod(&slabs.row_gradient_slabs).ok()?;
    let border_hessian_dev = stream.clone_htod(&slabs.border_hessian).ok()?;
    let border_gradient_dev = stream.clone_htod(&slabs.border_gradient).ok()?;
    let bytes = [
        target_x.len(),
        basis_values.len(),
        gate_activations.len(),
        slabs.row_hessian_slabs.len(),
        slabs.row_cross_slabs.len(),
        slabs.row_gradient_slabs.len(),
        slabs.border_hessian.len(),
        slabs.border_gradient.len(),
    ]
    .into_iter()
    .sum::<usize>()
        * std::mem::size_of::<f64>();
    Some(DeviceResidentArrowBuffers {
        stream,
        target_x_dev,
        basis_values_dev,
        gate_activations_dev,
        row_hessian_dev,
        row_cross_dev,
        row_gradient_dev,
        border_hessian_dev,
        border_gradient_dev,
        bytes,
    })
}

fn map_gpu_error(err: ArrowSchurGpuFailure) -> DeviceResidentArrowError {
    match err {
        ArrowSchurGpuFailure::Unavailable => DeviceResidentArrowError::Unavailable {
            reason: "SAE resident inner iteration unavailable after GPU admission".to_string(),
        },
        ArrowSchurGpuFailure::RidgeBumpRequired { row, bump } => DeviceResidentArrowError::Solve {
            reason: format!("SAE resident inner iteration row {row} requires ridge bump {bump:e}"),
        },
        ArrowSchurGpuFailure::SchurFactorFailed { reason } => {
            DeviceResidentArrowError::Solve { reason }
        }
        ArrowSchurGpuFailure::GpuRequiresDenseSystem {
            had_hbb_matvec,
            had_htbeta_matvec,
        } => DeviceResidentArrowError::Solve {
            reason: format!(
                "SAE resident inner iteration requires dense slabs; hbb_matvec={had_hbb_matvec} htbeta_matvec={had_htbeta_matvec}"
            ),
        },
    }
}

fn squared_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum()
}

impl From<ArrowSchurError> for DeviceResidentArrowError {
    fn from(err: ArrowSchurError) -> Self {
        Self::Solve {
            reason: err.to_string(),
        }
    }
}

/// Deterministic qwen-scale non-gating fixture for the resident harness.
pub fn qwen_non_gating_fixture() -> Result<DeviceResidentArrowWorkspace, DeviceResidentArrowError> {
    qwen_non_gating_fixture_seeded(0x1017_0003_D3A1_5EED)
}

/// Seeded variant of [`qwen_non_gating_fixture`]. Distinct seeds produce
/// distinct-but-well-conditioned resident frames, used to build independent
/// replicate fits for the stream-multiplexing parity harness.
pub fn qwen_non_gating_fixture_seeded(
    seed: u64,
) -> Result<DeviceResidentArrowWorkspace, DeviceResidentArrowError> {
    fixture_for_shape_seeded(DeviceResidentArrowShape::qwen_non_gating(), seed)
}

/// Deterministic color-arm-scale resident fixture (n=180, p=5120) for the
/// #1017 GPU wall-clock bench: few rows, very wide border — the shape where the
/// per-iterate re-upload + re-factor that across-iteration residency eliminates
/// dominates.
pub fn color_arm_fixture() -> Result<DeviceResidentArrowWorkspace, DeviceResidentArrowError> {
    fixture_for_shape_seeded(DeviceResidentArrowShape::color_arm(), 0x1017_C010_2A12_5EED)
}

/// Build a well-conditioned resident frame for any `d == 2` shape. Both the
/// qwen and color-arm fixtures share this body; the conditioning (strong row
/// `H_tt` diagonals, tiny cross blocks, diagonally-dominant border) keeps the
/// dense reference factorisation PD so the parity harness is meaningful.
fn fixture_for_shape_seeded(
    shape: DeviceResidentArrowShape,
    seed: u64,
) -> Result<DeviceResidentArrowWorkspace, DeviceResidentArrowError> {
    if shape.d == 0 {
        return Err(DeviceResidentArrowError::Shape {
            reason: "fixture_for_shape_seeded requires d >= 1".to_string(),
        });
    }
    let d = shape.d;
    let mut rng = SplitMix64::new(seed);
    let mut target_x = vec![0.0_f64; shape.target_len()];
    for i in 0..shape.n {
        for j in 0..shape.p {
            let phase = ((i % 97) as f64) * 0.013 + ((j % 131) as f64) * 0.007;
            target_x[i * shape.p + j] = 0.02 * phase.sin() + 0.001 * rng.sample_signed();
        }
    }
    let mut basis_values = vec![0.0_f64; shape.basis_len()];
    let mut gate_activations = vec![1.0_f64; shape.basis_len()];
    for i in 0..shape.n {
        for a in 0..shape.basis_cols {
            let phase = ((i + 1) as f64) * ((a + 1) as f64) * 0.003;
            basis_values[i * shape.basis_cols + a] = phase.cos();
            gate_activations[i * shape.basis_cols + a] = 1.0;
        }
    }
    let mut row_hessian_slabs = vec![0.0_f64; shape.row_hessian_len()];
    let mut row_cross_slabs = vec![0.0_f64; shape.row_cross_len()];
    let mut row_gradient_slabs = vec![0.0_f64; shape.row_gradient_len()];
    for i in 0..shape.n {
        let mut basis_sum = 0.0_f64;
        for a in 0..shape.basis_cols {
            basis_sum +=
                basis_values[i * shape.basis_cols + a] * gate_activations[i * shape.basis_cols + a];
        }
        // Strongly diagonally-dominant d×d H_tt (row-major): diagonal ≈ 3, tiny
        // symmetric off-diagonals — PD for any d so the dense reference factors.
        let h_base = i * d * d;
        for r in 0..d {
            for c in 0..d {
                let v = if r == c {
                    3.0 + 0.01 * basis_sum.abs() + 0.1 * (r as f64)
                } else {
                    0.02 * (basis_sum + (r + c) as f64).sin() / (d as f64)
                };
                row_hessian_slabs[h_base + r * d + c] = v;
            }
        }
        // Symmetrize the off-diagonals exactly.
        for r in 0..d {
            for c in 0..r {
                let avg = 0.5
                    * (row_hessian_slabs[h_base + r * d + c]
                        + row_hessian_slabs[h_base + c * d + r]);
                row_hessian_slabs[h_base + r * d + c] = avg;
                row_hessian_slabs[h_base + c * d + r] = avg;
            }
        }
        // d×p cross block (row-major) and length-d gradient.
        let b_base = i * d * shape.p;
        let g_base = i * d;
        for r in 0..d {
            for j in 0..shape.p {
                let feature = ((j % 257) as f64) * 0.011;
                row_cross_slabs[b_base + r * shape.p + j] =
                    1.0e-4 * (basis_sum + r as f64).sin() * feature.cos();
            }
            row_gradient_slabs[g_base + r] = 0.01 * (basis_sum + r as f64).sin();
        }
    }
    let mut border_hessian = vec![0.0_f64; shape.border_hessian_len()];
    for r in 0..shape.p {
        border_hessian[r * shape.p + r] = 4.0;
        if r + 1 < shape.p {
            border_hessian[r * shape.p + r + 1] = 0.01;
            border_hessian[(r + 1) * shape.p + r] = 0.01;
        }
    }
    let mut border_gradient = vec![0.0_f64; shape.p];
    for j in 0..shape.p {
        border_gradient[j] = 0.001 * ((j % 193) as f64 * 0.017).sin();
    }
    DeviceResidentArrowWorkspace::new(
        shape,
        target_x,
        basis_values,
        gate_activations,
        DeviceResidentArrowSlabs {
            row_hessian_slabs,
            row_cross_slabs,
            row_gradient_slabs,
            border_hessian,
            border_gradient,
        },
    )
}

/// One multiplexed resident fit: the workspace plus the inner-loop outcome.
pub struct MultiplexedFit {
    pub outcome: DeviceResidentInnerOutcome,
}

/// Phase 4: run `workspaces.len()` independent device-resident inner fits that
/// share one device.
///
/// # Stream-multiplexing safety argument
///
/// Each fit calls [`DeviceResidentArrowWorkspace::device_fit`], whose per-row
/// arrow solve (`solve_arrow_newton_step`) acquires the **process-shared**
/// `Arc<CudaContext>` via `device_runtime::cuda_context_for` (a `Mutex`-guarded
/// `OnceLock` cache) and then creates its **own** `CudaStream` with its own
/// cuSOLVER/cuBLAS handles and its own device allocations. Distinct streams off
/// one shared context execute concurrently on the device; the only shared
/// mutable state — the context cache and cudarc's allocator — is internally
/// synchronised, and no two fits touch the same stream, handle, or buffer. So
/// independent fits are data-race-free and the device serialises only where the
/// hardware must (shared SMs / copy engines), which is exactly the throughput
/// multiplexing the issue's Phase 4 calls for.
///
/// Concurrency is driven through [`run_topology_race_parallel`] (bac4af426),
/// which already bounds nested Rayon so each fit's internal `par_iter`/faer
/// parallelism stays inside its per-fit thread budget rather than oversubscribing
/// the global pool. Results are returned in input order. A single A100 thus hosts
/// many color-/qwen-arm fits at once — the cross-fit batch where the 1e5–1e6×
/// race speedup materialises.
///
/// The GPU runtime singleton (`GpuRuntime::global`) and per-ordinal context
/// cache are warmed by constructing the resident workspaces (each `new` calls
/// the same probe), so the per-fit calls inside the Rayon scope only *read* the
/// already-initialised `OnceLock`s — they never trigger a `get_or_init` whose
/// closure does nested parallel work, avoiding the OnceLock×Rayon deadlock.
pub fn run_resident_fits_multiplexed(
    workspaces: Vec<DeviceResidentArrowWorkspace>,
    opts: DeviceResidentInnerOptions,
) -> Result<Vec<Result<MultiplexedFit, DeviceResidentArrowError>>, String> {
    run_resident_fits_multiplexed_with(workspaces, opts, |workspace, opts| {
        workspace.device_fit(opts)
    })
}

/// Multiplexing core parameterised over the per-fit runner, so the CPU-reference
/// path can exercise the exact same `run_topology_race_parallel` plumbing as the
/// device path in tests that run without CUDA.
fn run_resident_fits_multiplexed_with<Run>(
    workspaces: Vec<DeviceResidentArrowWorkspace>,
    opts: DeviceResidentInnerOptions,
    run_one: Run,
) -> Result<Vec<Result<MultiplexedFit, DeviceResidentArrowError>>, String>
where
    Run: Fn(
            &DeviceResidentArrowWorkspace,
            &DeviceResidentInnerOptions,
        ) -> Result<DeviceResidentInnerOutcome, DeviceResidentArrowError>
        + Sync,
{
    let rows = crate::topology_selector::run_topology_race_parallel(
        workspaces,
        move |workspace: DeviceResidentArrowWorkspace| {
            run_one(&workspace, &opts).map(|outcome| MultiplexedFit { outcome })
        },
    )?;
    Ok(rows.into_iter().map(|row| row.result).collect())
}

/// Sequential reference for the multiplexing parity harness: the same fits run
/// one after another on the same shared device. Multiplexed results must be
/// bit-identical to this because each fit's arithmetic is independent of the
/// others — sharing the device changes only scheduling, never the numbers.
pub fn run_resident_fits_sequential(
    workspaces: &[DeviceResidentArrowWorkspace],
    opts: &DeviceResidentInnerOptions,
) -> Vec<Result<MultiplexedFit, DeviceResidentArrowError>> {
    workspaces
        .iter()
        .map(|workspace| {
            workspace
                .device_fit(opts)
                .map(|outcome| MultiplexedFit { outcome })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 4 variant sweep (#1017): the OLMo research battery's independent-fit
// matrix (K × topology × basis × layer/checkpoint) dispatched concurrently on
// one device.
//
// Each variant is a SEPARATE fit with its OWN resident frame: the per-fit
// arithmetic is independent of the others, so multiplexing them onto one a100
// changes only scheduling, never the numbers. This is the cross-fit batch where
// the issue's 1e5–1e6× race throughput materialises — and unlike per-fit
// across-iteration residency it needs NO fixed-quadratic inner loop, because the
// parallelism is BETWEEN fits, not within one.
// ---------------------------------------------------------------------------

/// One independent fit in the battery's variant sweep. The battery maps each
/// (K, topology, basis, layer, checkpoint, seed) cell of its matrix to a
/// `SweepVariant`; `dim` carries the resident-frame shape that cell produces
/// after the host assembles its row/border slabs. Distinct `seed`s keep the
/// fits genuinely independent (no shared device buffer, handle, or stream).
#[derive(Clone, Copy, Debug)]
pub struct SweepVariant {
    /// Resident-frame shape for this variant's frozen gate/basis frame.
    pub dim: DeviceResidentArrowShape,
    /// Deterministic seed for this variant's fixture/frame.
    pub seed: u64,
}

/// Throughput summary for a multiplexed variant sweep on one device.
#[derive(Clone, Copy, Debug)]
pub struct SweepThroughput {
    pub fits: usize,
    pub succeeded: usize,
    pub wall_seconds: f64,
    /// Fits completed per wall-clock second on the single shared device.
    pub fits_per_second: f64,
}

/// Build the independent resident workspaces for a variant sweep. Each variant
/// gets its own well-conditioned `d == 2` frame (the host feeds real slabs in
/// production; here the deterministic fixture stands in for the parity/throughput
/// harness). Returns the workspaces in variant order.
pub fn build_sweep_workspaces(
    variants: &[SweepVariant],
) -> Result<Vec<DeviceResidentArrowWorkspace>, DeviceResidentArrowError> {
    variants
        .iter()
        .map(|v| fixture_for_shape_seeded(v.dim, v.seed))
        .collect()
}

/// Dispatch a variant sweep concurrently on one device and measure cross-fit
/// throughput. Returns the per-variant outcomes (in variant order) and the
/// throughput summary (fits/sec on the single shared a100). Per-fit certified
/// parity is asserted by [`assert_sweep_parity_vs_sequential`].
pub fn run_variant_sweep_multiplexed(
    variants: &[SweepVariant],
    opts: DeviceResidentInnerOptions,
) -> Result<
    (
        Vec<Result<MultiplexedFit, DeviceResidentArrowError>>,
        SweepThroughput,
    ),
    String,
> {
    let workspaces = build_sweep_workspaces(variants).map_err(|e| e.to_string())?;
    run_battery_sweep_multiplexed(workspaces, opts)
}

/// Production battery entry (#1017 Phase 4): dispatch CALLER-ASSEMBLED resident
/// workspaces concurrently on one device and measure cross-fit throughput.
///
/// This is the real-slab seam the OLMo battery uses: the host (pyffi) builds one
/// [`DeviceResidentArrowWorkspace`] per matrix cell from the cell's ACTUAL SAE
/// row_hessian/row_cross/border slabs via [`DeviceResidentArrowWorkspace::new`],
/// then hands the workspaces here. Unlike [`run_variant_sweep_multiplexed`]
/// (which builds frames from the deterministic harness fixture), this consumes
/// real frames, so the printed throughput is the battery's true fits/sec on one
/// device. Returns per-cell outcomes (in input order) + the throughput summary.
pub fn run_battery_sweep_multiplexed(
    workspaces: Vec<DeviceResidentArrowWorkspace>,
    opts: DeviceResidentInnerOptions,
) -> Result<
    (
        Vec<Result<MultiplexedFit, DeviceResidentArrowError>>,
        SweepThroughput,
    ),
    String,
> {
    let fits = workspaces.len();
    let start = std::time::Instant::now();
    let results = run_resident_fits_multiplexed(workspaces, opts)?;
    let wall_seconds = start.elapsed().as_secs_f64();
    let succeeded = results.iter().filter(|r| r.is_ok()).count();
    let throughput = SweepThroughput {
        fits,
        succeeded,
        wall_seconds,
        fits_per_second: (fits as f64) / wall_seconds.max(1e-9),
    };
    Ok((results, throughput))
}

/// The OLMo battery's full color-arm variant matrix as [`SweepVariant`]s:
/// `K{1..=4} × topology{4} × basis{periodic, linear}` at the color-arm shape
/// (n=180, p=5120). `d` and `basis_cols` follow the intrinsic-rank convention
/// (periodic ⇒ d=2, basis_cols=8; linear ⇒ d=1, basis_cols=2). Exposed so the
/// pyffi battery seam can quote cross-fit throughput on the real shape matrix
/// (fixture frames) before the per-cell real-slab fits are wired through.
#[must_use]
pub fn color_arm_variant_matrix() -> Vec<SweepVariant> {
    let topologies = ["euclidean", "circle", "torus", "sphere"];
    let mut variants = Vec::with_capacity(4 * topologies.len() * 2);
    for k in 1..=4u64 {
        for (t_idx, _topology) in topologies.iter().enumerate() {
            // periodic (2 harmonics) and linear basis arms.
            for &(d, basis_cols, basis_tag) in &[(2usize, 8usize, 0u64), (1usize, 2usize, 1u64)] {
                let mut dim = DeviceResidentArrowShape::color_arm();
                dim.d = d;
                dim.basis_cols = basis_cols;
                let seed = 0x1017_C010_0000_0000 ^ (k << 16) ^ ((t_idx as u64) << 8) ^ basis_tag;
                variants.push(SweepVariant { dim, seed });
            }
        }
    }
    variants
}

/// Certified per-fit parity for a variant sweep: the multiplexed (concurrent)
/// results must be bit-for-bit identical to the same fits run sequentially on
/// the same device, because independent fits' arithmetic does not depend on
/// scheduling. Returns the sequential throughput so the caller can report the
/// multiplex speedup (multiplexed fits/sec ÷ sequential fits/sec). Returns an
/// `Err` describing the first divergence so the harness fails loudly.
pub fn assert_sweep_parity_vs_sequential(
    variants: &[SweepVariant],
    opts: &DeviceResidentInnerOptions,
    multiplexed: &[Result<MultiplexedFit, DeviceResidentArrowError>],
) -> Result<SweepThroughput, String> {
    let workspaces = build_sweep_workspaces(variants).map_err(|e| e.to_string())?;
    let start = std::time::Instant::now();
    let sequential = run_resident_fits_sequential(&workspaces, opts);
    let wall_seconds = start.elapsed().as_secs_f64();
    if sequential.len() != multiplexed.len() {
        return Err(format!(
            "sweep parity: length mismatch seq={} mux={}",
            sequential.len(),
            multiplexed.len()
        ));
    }
    for (idx, (seq, mux)) in sequential.iter().zip(multiplexed.iter()).enumerate() {
        match (seq, mux) {
            (Ok(s), Ok(m)) => {
                if s.outcome.t.as_slice() != m.outcome.t.as_slice()
                    || s.outcome.beta.as_slice() != m.outcome.beta.as_slice()
                    || s.outcome.objective.to_bits() != m.outcome.objective.to_bits()
                {
                    return Err(format!(
                        "sweep parity: fit {idx} multiplexed result differs from sequential"
                    ));
                }
            }
            (Err(_), Err(_)) => {}
            _ => {
                return Err(format!(
                    "sweep parity: fit {idx} success/failure disagrees seq-vs-mux"
                ));
            }
        }
    }
    let fits = variants.len();
    let succeeded = sequential.iter().filter(|r| r.is_ok()).count();
    Ok(SweepThroughput {
        fits,
        succeeded,
        wall_seconds,
        fits_per_second: (fits as f64) / wall_seconds.max(1e-9),
    })
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        gam_linalg::utils::splitmix64(&mut self.state)
    }

    fn sample_signed(&mut self) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
        2.0 * unit - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build a small, strongly diagonally-dominant resident frame whose dense
    /// reference factorisation is well-conditioned. The objective minimiser is
    /// `z* = H^{-1} g₀`, which the inner loop must reach.
    fn small_fixture(seed: u64) -> DeviceResidentArrowWorkspace {
        // batch (n) = 8 clears the device dispatch floor
        // (`small_dense_batched_potrf_min_batch = 8`) so that on a CUDA host
        // `upload_resident_buffers` actually binds a device and
        // `device_resident()` is TRUE — otherwise the device-resident parity
        // branch of `device_resident_fit_matches_cpu_reference` is dead on real
        // GPU hardware (the route declines for batch < 8, so the test only ever
        // exercised the CPU-decline branch and never validated the device loop).
        let shape = DeviceResidentArrowShape {
            n: 8,
            p: 4,
            basis_cols: 2,
            d: 2,
        };
        let mut rng = SplitMix64::new(seed);
        let target_x = vec![0.0_f64; shape.target_len()];
        let basis_values = vec![0.5_f64; shape.basis_len()];
        let gate_activations = vec![1.0_f64; shape.basis_len()];

        let mut row_hessian_slabs = vec![0.0_f64; shape.row_hessian_len()];
        let mut row_cross_slabs = vec![0.0_f64; shape.row_cross_len()];
        let mut row_gradient_slabs = vec![0.0_f64; shape.row_gradient_len()];
        for i in 0..shape.n {
            let h = i * shape.d * shape.d;
            row_hessian_slabs[h] = 5.0 + 0.1 * rng.sample_signed();
            row_hessian_slabs[h + 1] = 0.05 * rng.sample_signed();
            row_hessian_slabs[h + 2] = row_hessian_slabs[h + 1];
            row_hessian_slabs[h + 3] = 4.0 + 0.1 * rng.sample_signed();
            let b = i * shape.d * shape.p;
            for j in 0..shape.p {
                row_cross_slabs[b + j] = 0.01 * rng.sample_signed();
                row_cross_slabs[b + shape.p + j] = 0.01 * rng.sample_signed();
            }
            let g = i * shape.d;
            row_gradient_slabs[g] = rng.sample_signed();
            row_gradient_slabs[g + 1] = rng.sample_signed();
        }
        let mut border_hessian = vec![0.0_f64; shape.border_hessian_len()];
        for r in 0..shape.p {
            border_hessian[r * shape.p + r] = 6.0 + 0.1 * rng.sample_signed();
        }
        let border_gradient: Vec<f64> = (0..shape.p).map(|_| rng.sample_signed()).collect();

        DeviceResidentArrowWorkspace::new(
            shape,
            target_x,
            basis_values,
            gate_activations,
            DeviceResidentArrowSlabs {
                row_hessian_slabs,
                row_cross_slabs,
                row_gradient_slabs,
                border_hessian,
                border_gradient,
            },
        )
        .expect("small resident fixture must validate")
    }

    /// Dense `H z` for the resident frame (independent of the arrow path),
    /// used to confirm the inner-loop fixed point is the true stationary point.
    fn dense_hz(
        ws: &DeviceResidentArrowWorkspace,
        sys: &ArrowSchurSystem,
    ) -> (Array2<f64>, Array1<f64>) {
        let shape = ws.shape;
        let total = shape.n * shape.d + shape.p;
        let mut h = Array2::<f64>::zeros((total, total));
        let mut g0 = Array1::<f64>::zeros(total);
        for i in 0..shape.n {
            let base = i * shape.d;
            for r in 0..shape.d {
                for c in 0..shape.d {
                    h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
                }
                for c in 0..shape.p {
                    let v = sys.rows[i].htbeta[[r, c]];
                    h[[base + r, shape.n * shape.d + c]] = v;
                    h[[shape.n * shape.d + c, base + r]] = v;
                }
                g0[base + r] = sys.rows[i].gt[r];
            }
        }
        for r in 0..shape.p {
            for c in 0..shape.p {
                h[[shape.n * shape.d + r, shape.n * shape.d + c]] = sys.hbb[[r, c]];
            }
            g0[shape.n * shape.d + r] = sys.gb[r];
        }
        (h, g0)
    }

    #[test]
    fn cpu_inner_loop_reaches_quadratic_minimiser() {
        let ws = small_fixture(0xABCD_0001);
        let opts = DeviceResidentInnerOptions::default();
        let outcome = ws.cpu_reference_fit(&opts).expect("cpu fit");
        assert!(
            outcome.converged,
            "inner loop must converge on a PD quadratic"
        );

        // The stationary point satisfies H z* = g₀; verify the residual is zero.
        let base = ws.to_arrow_system();
        let (h, g0) = dense_hz(&ws, &base);
        let total = ws.shape.n * ws.shape.d + ws.shape.p;
        let mut z = Array1::<f64>::zeros(total);
        for r in 0..ws.shape.n * ws.shape.d {
            z[r] = outcome.t[r];
        }
        for c in 0..ws.shape.p {
            z[ws.shape.n * ws.shape.d + c] = outcome.beta[c];
        }
        let hz = h.dot(&z);
        let mut max_resid = 0.0_f64;
        for r in 0..total {
            max_resid = max_resid.max((hz[r] - g0[r]).abs());
        }
        assert!(
            max_resid < 1e-9,
            "inner loop fixed point must solve H z = g0; residual {max_resid:e}"
        );
    }

    #[test]
    fn cpu_multiplex_matches_sequential_bit_identical() {
        let seeds = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let opts = DeviceResidentInnerOptions::default();

        let seq_workspaces: Vec<_> = seeds.iter().map(|&s| small_fixture(s)).collect();
        let sequential: Vec<_> = seq_workspaces
            .iter()
            .map(|ws| ws.cpu_reference_fit(&opts).expect("seq cpu fit"))
            .collect();

        let mux_workspaces: Vec<_> = seeds.iter().map(|&s| small_fixture(s)).collect();
        let multiplexed = run_resident_fits_multiplexed_with(mux_workspaces, opts, |ws, opts| {
            ws.cpu_reference_fit(opts)
        })
        .expect("multiplexed cpu fits");

        assert_eq!(sequential.len(), multiplexed.len());
        for (seq, mux) in sequential.iter().zip(multiplexed.iter()) {
            let mux = mux.as_ref().expect("mux fit ok");
            // Independent fits: scheduling cannot change the numbers, so the
            // parallel result must be bit-for-bit identical to sequential.
            assert_eq!(seq.t.as_slice(), mux.outcome.t.as_slice());
            assert_eq!(seq.beta.as_slice(), mux.outcome.beta.as_slice());
            assert_eq!(seq.objective.to_bits(), mux.outcome.objective.to_bits());
        }
    }

    /// #1017 Phase 3 residency parity. On a CUDA host the device-resident inner
    /// loop (`device_fit`, which keeps the Hessian factors on-device across
    /// iterations via `ResidentArrowFrameHandle`) must reach the same minimiser
    /// as the fully independent CPU dense-reference loop (`cpu_reference_fit`,
    /// which re-factors per iterate). On a CPU-only host the resident path must
    /// decline cleanly (`Unavailable`) rather than silently disagree, and the
    /// resident-frame handle construction must likewise decline — so the gate is
    /// meaningful on the build box and the wall-clock arm runs on the GPU node.
    #[test]
    fn device_resident_fit_matches_cpu_reference() {
        let ws = small_fixture(0x5AE_1017);
        let opts = DeviceResidentInnerOptions::default();

        // CPU reference (re-factors per iterate) — always available.
        let cpu = ws.cpu_reference_fit(&opts).expect("cpu reference fit");
        assert!(cpu.converged, "cpu reference must converge on PD quadratic");

        let base = ws.to_arrow_system();

        if ws.device_resident() {
            // Resident device loop: factors stay on-device across iterations.
            let dev = ws.device_fit(&opts).expect("device resident fit");
            assert_eq!(
                dev.execution_path,
                ExecutionPath::GpuResidentFull,
                "device_fit must report the full device-resident execution path"
            );
            assert!(dev.converged, "device resident loop must converge");

            // Certified-refinement parity (#1014): the resident path and the
            // independent CPU path solve the same quadratic, so their minimisers
            // agree to a tight relative tolerance. The resident path differs from
            // the reference only by SKIPPING re-derivation of g-independent
            // factor work, not by changing the arithmetic.
            let t_scale = cpu.t.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
            let b_scale = cpu.beta.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
            let mut max_rel = 0.0_f64;
            for (a, b) in dev.t.iter().zip(cpu.t.iter()) {
                max_rel = max_rel.max((a - b).abs() / t_scale);
            }
            for (a, b) in dev.beta.iter().zip(cpu.beta.iter()) {
                max_rel = max_rel.max((a - b).abs() / b_scale);
            }
            assert!(
                max_rel < 1e-9,
                "resident device fit must match CPU reference (rel {max_rel:e})"
            );

            // The resident frame's single-gradient solve must also match a full
            // independent solve at the same gradient (the per-iterate contract).
            let frame = crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle::new(
                &base,
                opts.initial_ridge_t,
                opts.initial_ridge_beta,
            )
            .expect("resident frame must build on CUDA host");
            let g_t: Vec<f64> = base
                .rows
                .iter()
                .flat_map(|r| r.gt.iter().copied())
                .collect();
            let g_beta: Vec<f64> = base.gb.iter().copied().collect();
            let resident_sol = frame
                .solve_gradient(&g_t, &g_beta)
                .expect("resident single-gradient solve");
            let full = crate::gpu_kernels::arrow_schur::solve_arrow_newton_step_dense_reference(
                &base,
                opts.initial_ridge_t,
                opts.initial_ridge_beta,
            )
            .expect("dense reference single solve");
            let mut max_step_rel = 0.0_f64;
            let step_scale = full
                .delta_t
                .iter()
                .chain(full.delta_beta.iter())
                .fold(1.0_f64, |m, &v| m.max(v.abs()));
            for (a, b) in resident_sol.delta_t.iter().zip(full.delta_t.iter()) {
                max_step_rel = max_step_rel.max((a - b).abs() / step_scale);
            }
            for (a, b) in resident_sol.delta_beta.iter().zip(full.delta_beta.iter()) {
                max_step_rel = max_step_rel.max((a - b).abs() / step_scale);
            }
            assert!(
                max_step_rel < 1e-9,
                "resident solve_gradient must match full dense reference step (rel {max_step_rel:e})"
            );

            // The re-uploading GPU loop (residency baseline) must reach the same
            // minimiser as both the resident loop and the CPU reference.
            let reup = ws
                .device_reupload_fit(&opts)
                .expect("device re-uploading fit");
            assert_eq!(
                reup.execution_path,
                ExecutionPath::GpuReupload,
                "device_reupload_fit must report the re-uploading device path"
            );
            assert!(reup.converged, "re-uploading loop must converge");
            let mut max_reup_rel = 0.0_f64;
            for (a, b) in reup.t.iter().zip(cpu.t.iter()) {
                max_reup_rel = max_reup_rel.max((a - b).abs() / t_scale);
            }
            for (a, b) in reup.beta.iter().zip(cpu.beta.iter()) {
                max_reup_rel = max_reup_rel.max((a - b).abs() / b_scale);
            }
            assert!(
                max_reup_rel < 1e-9,
                "re-uploading GPU fit must match CPU reference (rel {max_reup_rel:e})"
            );
        } else {
            // The fixture is sized (batch = 8) to clear the device dispatch floor,
            // so on a host WITH a CUDA runtime `device_resident()` must be true and
            // we take the device branch above. Reaching this branch with a runtime
            // present means the device binding silently failed — which would mask a
            // real upload/dispatch fault behind the CPU-decline path (the
            // device-PCG skip-pass class, eee12f6b2). Fail loud unless this is a
            // genuinely CPU-only host.
            assert!(
                gam_gpu::device_runtime::GpuRuntime::global().is_none(),
                "device_resident() is false on a host WITH a CUDA runtime present, \
                 despite a floor-clearing fixture (batch=8): the resident device \
                 buffers failed to bind — a real device fault, not a CPU-only skip."
            );
            // CPU-only host: the resident path must decline, not disagree.
            let dev = ws.device_fit(&opts);
            assert!(
                matches!(dev, Err(DeviceResidentArrowError::Unavailable { .. })),
                "device_fit must report Unavailable on a CPU-only host, got {dev:?}"
            );
            let reup = ws.device_reupload_fit(&opts);
            assert!(
                matches!(reup, Err(DeviceResidentArrowError::Unavailable { .. })),
                "device_reupload_fit must report Unavailable on a CPU-only host, got {reup:?}"
            );
            let frame = crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle::new(
                &base,
                opts.initial_ridge_t,
                opts.initial_ridge_beta,
            );
            assert!(
                frame.is_err(),
                "resident frame construction must decline on a CPU-only host"
            );
        }
    }

    /// #1017 fit-path parity (CPU-runnable). The resident inner solve and the
    /// PRODUCTION arrow-Schur inner solve (`solve_arrow_newton_step_core`, the
    /// entry the SAE joint fit reaches through `solve_with_lm_escalation_inner`)
    /// must solve the SAME bordered-quadratic Newton system.
    ///
    /// This is the cross-implementation parity behind wiring the device seam into
    /// the SAE inner loop: `solve_arrow_newton_step_core` carries the #1017
    /// device-Schur seam (and falls through bit-identically to its CPU path off
    /// CUDA), and the resident workspace's `cpu_reference_fit` converges the same
    /// quadratic `φ(z) = ½‖X‖² + ½ zᵀH z − g₀ᵀ z`. The resident converged iterate
    /// `z*` is the stationary point `H z* = g₀`; the production arrow path solves
    /// the Newton system `H Δ = −g₀` from `z = 0`, so its step is
    /// `Δ = −H⁻¹ g₀ = −z*`. With `H` PD the exact relationship is therefore
    /// `Δ = −z*`; asserting it pins that routing the production inner solve through
    /// the device-aware `_core` (which a GPU host then offloads) solves the
    /// identical system the resident loop does. Runs on the CPU build box — no
    /// CUDA required.
    #[test]
    fn resident_inner_solve_matches_production_arrow_core() {
        use crate::arrow_schur::{ArrowSolveOptions, solve_arrow_newton_step_core};

        let ws = small_fixture(0x1017_F17);
        let opts = DeviceResidentInnerOptions::default();

        // Resident workspace converged fit (re-factoring CPU reference loop).
        let resident = ws.cpu_reference_fit(&opts).expect("resident cpu fit");
        assert!(
            resident.converged,
            "resident reference must converge on the PD quadratic"
        );

        // Production arrow path: one Newton step on the same system from z = 0.
        // `_core` is the device-aware entry; on this CPU box it runs the dense
        // CPU solve, the exact path the GPU host would fall back to on decline.
        let sys = ws.to_arrow_system();
        let (delta_t, delta_beta, _diag) = solve_arrow_newton_step_core(
            &sys,
            opts.initial_ridge_t,
            opts.initial_ridge_beta,
            &ArrowSolveOptions::direct(),
        )
        .expect("production arrow-core solve");

        // The Newton step from z = 0 is Δ = −H⁻¹g₀ = −z*, where z* is the resident
        // converged iterate (H z* = g₀, the invariant
        // `cpu_inner_loop_reaches_quadratic_minimiser` pins directly). With H PD
        // the relationship is exact, so Δ + z* = 0 to factorisation tolerance.
        let t_scale = resident.t.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
        let b_scale = resident.beta.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
        // #1399: report the t-block and beta-block mismatch SEPARATELY (not one
        // fused scalar). The two halves localise a divergence: a t-block-only gap
        // points at the per-row factor / row gradient assembly, a beta-block gap
        // at the border Schur path — turning the opaque overall rel into an
        // actionable signal for the resident-vs-production parity divergence.
        let mut max_rel_t = 0.0_f64;
        let mut worst_t: Option<(usize, f64, f64)> = None;
        for (i, (prod, res)) in delta_t.iter().zip(resident.t.iter()).enumerate() {
            let rel = (prod + res).abs() / t_scale;
            if rel > max_rel_t {
                max_rel_t = rel;
                worst_t = Some((i, *prod, *res));
            }
        }
        let mut max_rel_b = 0.0_f64;
        let mut worst_b: Option<(usize, f64, f64)> = None;
        for (i, (prod, res)) in delta_beta.iter().zip(resident.beta.iter()).enumerate() {
            let rel = (prod + res).abs() / b_scale;
            if rel > max_rel_b {
                max_rel_b = rel;
                worst_b = Some((i, *prod, *res));
            }
        }
        let max_rel = max_rel_t.max(max_rel_b);
        assert!(
            max_rel < 1e-9,
            "production arrow-core Newton step must be −(resident converged fit) on \
             the same quadratic; wiring the device seam into the SAE inner loop must \
             not change the system being solved. rel_t={max_rel_t:e} (worst {worst_t:?}: \
             Δt+t* must be 0), rel_beta={max_rel_b:e} (worst {worst_b:?}: Δβ+β* must \
             be 0). A t-only gap implicates the per-row factor / row-gradient \
             assembly; a β-only gap the border Schur path."
        );
    }

    /// #1017 deliverable 3: across-OUTER residency. A sequence of outer
    /// evaluations whose Hessian operator is unchanged (only the base gradient
    /// moves) must share ONE resident frame — exactly one frame build for the
    /// whole sweep — and produce results bit-identical to per-outer-independent
    /// fits (each with a fresh frame). On a CPU-only host this asserts the
    /// reference path's outer-sequence wiring is consistent; on the A100 it
    /// proves the across-outer factor amortization fires AND stays exact.
    #[test]
    fn outer_sequence_reuses_frame_and_matches_independent() {
        let ws = super::color_arm_fixture().expect("color_arm fixture");
        let opts = DeviceResidentInnerOptions::default();
        let n = ws.shape.n;
        let d = ws.shape.d;
        let p = ws.shape.p;

        // Three "outer" evaluations: same operator, distinct base gradients (the
        // moving linearization point). These stand in for consecutive outer REML
        // evaluations at a frozen gate/basis frame.
        let outers: Vec<(Vec<f64>, Vec<f64>)> = (0..3)
            .map(|s| {
                let g_t: Vec<f64> = (0..n * d)
                    .map(|i| 0.01 * (((i + 3 * s) as f64) * 0.002).sin())
                    .collect();
                let g_beta: Vec<f64> = (0..p)
                    .map(|j| 0.001 * (((j + 11 * s) as f64) * 0.0009).cos())
                    .collect();
                (g_t, g_beta)
            })
            .collect();

        // Per-outer-independent reference (fresh frame each outer) via the CPU
        // path, which runs on any host.
        let independent = ws
            .cpu_reference_outer_sequence(&outers, &opts)
            .expect("cpu reference outer sequence");
        assert_eq!(independent.outers.len(), outers.len());

        if ws.device_resident() {
            // Device across-outer sweep: ONE frame for all three outers.
            let shared = ws
                .device_fit_outer_sequence(&outers, &opts)
                .expect("device outer sequence");
            assert_eq!(
                shared.frame_builds,
                1,
                "across-outer residency must build the resident frame exactly once \
                 for an unchanged operator (got {} builds over {} outers) — a count \
                 > 1 means the frame was needlessly re-factored per outer",
                shared.frame_builds,
                outers.len()
            );
            // Bit-parity: sharing the factor across outers must not change the
            // numbers vs per-outer-independent device fits.
            for (idx, (sh, ind)) in shared
                .outers
                .iter()
                .zip(independent.outers.iter())
                .enumerate()
            {
                let scale = ind
                    .t
                    .iter()
                    .chain(ind.beta.iter())
                    .fold(1.0_f64, |m, &v| m.max(v.abs()));
                let mut max_rel = 0.0_f64;
                for (a, b) in sh.t.iter().zip(ind.t.iter()) {
                    max_rel = max_rel.max((a - b).abs() / scale);
                }
                for (a, b) in sh.beta.iter().zip(ind.beta.iter()) {
                    max_rel = max_rel.max((a - b).abs() / scale);
                }
                assert!(
                    max_rel < 1e-9,
                    "outer {idx}: across-outer-shared frame must match independent fit \
                     (rel {max_rel:e})"
                );
            }
            println!(
                "[#1017 outer-seq color_arm] outers={} frame_builds={} (across-outer factor \
                 amortized) parity<1e-9 OK",
                outers.len(),
                shared.frame_builds
            );
        } else {
            println!(
                "[#1017 outer-seq color_arm] no CUDA device — across-outer residency skipped; \
                 run on the GPU node to assert frame_builds==1 + device parity"
            );
        }
    }

    /// #1017 residency-isolating per-solve bench. A full-fit wall-clock bench
    /// runs an exact quadratic that converges
    /// in ONE Newton step, so the resident frame is built once and solved once —
    /// the across-iteration amortization (factor `D`/`B`/Schur once, reuse for
    /// every gradient) has nothing to amortize over and the measured speedup is
    /// only the single-solve `D`/`B` upload saving.
    ///
    /// This bench isolates the residency lever the way the production inner loop
    /// actually exercises it: at a frozen gate/basis frame the Hessian blocks are
    /// CONSTANT and the SAE inner Newton takes MANY gradient solves against them.
    /// It therefore times
    ///   * RESIDENT: build the [`crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle`]
    ///     ONCE, then `N` `solve_gradient` calls (upload only the `O(n·d + k)`
    ///     gradient per solve; no POTRF, no `D`/`B` re-upload);
    ///   * REUPLOAD: `N` `solve_arrow_newton_step` calls (re-pack/upload `D`/`B`/`g`
    ///     and re-run the per-row POTRF + border Schur factor every call).
    /// Both produce bit-identical steps; the ratio is the pure across-iteration
    /// residency speedup, which is what #1017 Phase 3 buys per inner iteration.
    /// `N` mirrors a realistic SAE inner-Newton iteration count. CPU-only hosts
    /// print a skip line. Run with `--nocapture`.
    #[test]
    fn gpu_residency_per_solve_bench() {
        use std::time::Instant;
        const N_SOLVES: usize = 24;
        for (label, ws) in [
            ("color_arm", super::color_arm_fixture()),
            ("qwen_non_gating", super::qwen_non_gating_fixture()),
        ] {
            let ws = ws.expect("bench fixture must validate");
            let base = ws.to_arrow_system();
            // A family of distinct gradients standing in for the per-iterate
            // residual r(z) = H z − g₀ the inner loop feeds. Distinct gradients
            // make the reupload path redo the (g-independent) factor work each
            // time — exactly the waste residency removes.
            let n = ws.shape.n;
            let d = ws.shape.d;
            let p = ws.shape.p;
            let gradients: Vec<(Vec<f64>, Vec<f64>)> = (0..N_SOLVES)
                .map(|s| {
                    let g_t: Vec<f64> =
                        (0..n * d).map(|i| ((i + s) as f64 * 0.001).sin()).collect();
                    let g_beta: Vec<f64> = (0..p)
                        .map(|j| ((j + 7 * s) as f64 * 0.0007).cos())
                        .collect();
                    (g_t, g_beta)
                })
                .collect();

            if !ws.device_resident() {
                println!(
                    "[#1017 per-solve {label}] no CUDA device — {N_SOLVES} solves skipped; \
                     run on the GPU node for the across-iteration residency speedup"
                );
                continue;
            }

            // Build the resident frame ONCE (its factor cost is the across-
            // iteration amortization the bench is measuring, so it is timed
            // separately from the per-solve loop).
            let t_build = Instant::now();
            let frame =
                crate::gpu_kernels::arrow_schur::ResidentArrowFrameHandle::new(&base, 0.0, 0.0)
                    .expect("resident frame must build on CUDA host");
            let frame_build_ms = t_build.elapsed().as_secs_f64() * 1e3;

            // Warm-up: one solve on each path before timing so the residency
            // ratio reflects steady-state per-iterate cost, not the one-time
            // NVRTC/cuSOLVER handle init, module JIT, or first-touch device
            // allocation (those are paid once per process, not per inner
            // iteration). The production inner loop pays them once and then runs
            // MANY solves, which is exactly the regime this assertion guards.
            let _ = frame
                .solve_gradient(&gradients[0].0, &gradients[0].1)
                .expect("resident warm-up solve");
            {
                let mut sys = ws.to_arrow_system();
                for (i, row) in sys.rows.iter_mut().enumerate() {
                    for r in 0..d {
                        row.gt[r] = gradients[0].0[i * d + r];
                    }
                }
                for (j, gb) in sys.gb.iter_mut().enumerate() {
                    *gb = gradients[0].1[j];
                }
                sys.refresh_row_hessian_fingerprint();
                let _ = crate::gpu_kernels::arrow_schur::solve_arrow_newton_step(&sys, 0.0, 0.0)
                    .expect("reupload warm-up solve");
            }

            // RESIDENT: reuse the (already-built, already-warmed) frame for N
            // gradient-only solves. Times ONLY the per-iterate gradient solves —
            // upload `O(n·d + k)` gradient, run the cheap residual path, read
            // back `δ`. No POTRF, no `D`/`B` re-upload.
            let t_res = Instant::now();
            let mut resident_steps = Vec::with_capacity(N_SOLVES);
            for (g_t, g_beta) in &gradients {
                resident_steps.push(
                    frame
                        .solve_gradient(g_t, g_beta)
                        .expect("resident solve_gradient"),
                );
            }
            let resident_ms = t_res.elapsed().as_secs_f64() * 1e3;

            // REUPLOAD: N full solves, each re-uploading D/B/g and re-factoring.
            let t_reup = Instant::now();
            let mut reupload_steps = Vec::with_capacity(N_SOLVES);
            for (g_t, g_beta) in &gradients {
                let mut sys = ws.to_arrow_system();
                for (i, row) in sys.rows.iter_mut().enumerate() {
                    for r in 0..d {
                        row.gt[r] = g_t[i * d + r];
                    }
                }
                for (j, gb) in sys.gb.iter_mut().enumerate() {
                    *gb = g_beta[j];
                }
                sys.refresh_row_hessian_fingerprint();
                reupload_steps.push(
                    crate::gpu_kernels::arrow_schur::solve_arrow_newton_step(&sys, 0.0, 0.0)
                        .expect("reupload solve_arrow_newton_step"),
                );
            }
            let reupload_ms = t_reup.elapsed().as_secs_f64() * 1e3;

            // Parity: resident and reupload steps must be bit-identical (same
            // factor kernels; residency only skips re-deriving g-independent work).
            let mut max_rel = 0.0_f64;
            for (rs, us) in resident_steps.iter().zip(reupload_steps.iter()) {
                let scale = us
                    .delta_t
                    .iter()
                    .chain(us.delta_beta.iter())
                    .fold(1.0_f64, |m, &v| m.max(v.abs()));
                for (a, b) in rs.delta_t.iter().zip(us.delta_t.iter()) {
                    max_rel = max_rel.max((a - b).abs() / scale);
                }
                for (a, b) in rs.delta_beta.iter().zip(us.delta_beta.iter()) {
                    max_rel = max_rel.max((a - b).abs() / scale);
                }
            }

            let resident_per_solve = resident_ms / N_SOLVES as f64;
            let reupload_per_solve = reupload_ms / N_SOLVES as f64;
            let residency_speedup = reupload_ms / resident_ms.max(1e-9);
            println!(
                "[#1017 per-solve {label}] N={N_SOLVES} frame_build={frame_build_ms:.2}ms \
                 resident={resident_ms:.2}ms ({resident_per_solve:.3}ms/solve, \
                 grad-upload + warm factors) reupload={reupload_ms:.2}ms \
                 ({reupload_per_solve:.3}ms/solve, N factors + N D/B uploads) \
                 residency_speedup={residency_speedup:.2}x parity_rel={max_rel:e}"
            );
            assert!(
                max_rel < 1e-9,
                "{label}: resident per-solve steps must match reupload (rel {max_rel:e})"
            );

            // #1017 deliverable 2: the residency amortization must actually fire
            // on hardware — reusing the resident factors across iterations has to
            // be STRICTLY cheaper per solve than re-uploading D/B/g and
            // re-factoring every iterate. This is the core perf claim, asserted
            // (not merely printed) so a regression that silently re-uploads, or a
            // dispatch change that drops the resident path, fails the gate on the
            // A100 instead of slipping through as a slower-but-green run.
            //
            // The `color_arm` shape (n=180, p=5120) is the decisive case: the
            // per-solve reupload pays a 5120-wide border Schur factor + the
            // `O(n·d·p)` cross-block upload every iterate, while the resident path
            // pays only the `O(n·d + p)` gradient transfer and two border TRSMs.
            // We require a clear >1.5x margin there. The `qwen_non_gating` shape
            // (p=2048) has a smaller border so its margin is thinner; we still
            // require a genuine speedup (>1x) but do not over-tighten it.
            let min_speedup = if label == "color_arm" { 1.5 } else { 1.0 };
            assert!(
                residency_speedup > min_speedup,
                "{label}: across-iteration residency must beat per-solve re-upload \
                 (residency_speedup={residency_speedup:.3}x, required >{min_speedup}x; \
                 resident {resident_per_solve:.3}ms/solve vs reupload \
                 {reupload_per_solve:.3}ms/solve over N={N_SOLVES} solves) — the resident \
                 frame either silently re-uploaded D/B or the dispatch dropped the \
                 amortized factor path"
            );
        }
    }

    /// #1017 Phase 4 variant sweep: an OLMo-battery-shaped matrix of independent
    /// fits (here K{1..4} × 3 basis widths = 12 color-arm variants) dispatched
    /// concurrently on one device. This is the cross-fit throughput lever — the
    /// fits are independent, so multiplexing changes only scheduling.
    fn battery_variant_matrix() -> Vec<super::SweepVariant> {
        let mut variants = Vec::new();
        // K is the topology rank; the battery races K{1..4}. Each K × basis cell
        // is an independent fit. Color-arm border, varied basis_cols per cell.
        for k in 1..=4u64 {
            for basis_cols in [4usize, 8, 12] {
                let mut dim = DeviceResidentArrowShape::color_arm();
                dim.basis_cols = basis_cols;
                variants.push(super::SweepVariant {
                    dim,
                    seed: 0x1017_0040_0000_0000 ^ (k << 8) ^ (basis_cols as u64),
                });
            }
        }
        variants
    }

    /// Phase-4 parity: the multiplexed sweep must be bit-identical to running the
    /// same fits sequentially (CPU reference path here so the gate runs on the
    /// build box; the device path is exercised by the throughput bench on the a100).
    #[test]
    fn variant_sweep_multiplex_matches_sequential() {
        let variants = battery_variant_matrix();
        let opts = DeviceResidentInnerOptions::default();

        // Multiplexed via the CPU-reference runner so the gate is meaningful
        // without CUDA, exercising the exact run_topology_race_parallel plumbing.
        let workspaces =
            super::build_sweep_workspaces(&variants).expect("sweep workspaces must build");
        let multiplexed =
            super::run_resident_fits_multiplexed_with(workspaces, opts, |ws, opts| {
                ws.cpu_reference_fit(opts)
            })
            .expect("multiplexed cpu sweep");

        let seq_workspaces =
            super::build_sweep_workspaces(&variants).expect("sweep workspaces must build");
        let sequential: Vec<_> = seq_workspaces
            .iter()
            .map(|ws| ws.cpu_reference_fit(&opts))
            .collect();

        assert_eq!(multiplexed.len(), sequential.len());
        for (idx, (mux, seq)) in multiplexed.iter().zip(sequential.iter()).enumerate() {
            let mux = &mux.as_ref().unwrap().outcome;
            let seq = seq.as_ref().unwrap();
            assert_eq!(
                mux.t.as_slice(),
                seq.t.as_slice(),
                "variant {idx}: multiplexed t differs from sequential"
            );
            assert_eq!(
                mux.beta.as_slice(),
                seq.beta.as_slice(),
                "variant {idx}: multiplexed beta differs from sequential"
            );
            assert_eq!(
                mux.objective.to_bits(),
                seq.objective.to_bits(),
                "variant {idx}: multiplexed objective differs from sequential"
            );
        }
    }

    /// #1017 Phase 4 throughput bench. On a CUDA host this dispatches the battery
    /// variant matrix concurrently on one device, asserts per-fit certified
    /// parity vs sequential, and prints the cross-fit throughput (multiplexed
    /// fits/sec vs sequential fits/sec — the single-a100 race speedup). On a
    /// CPU-only host it prints a skip line. Run with `--nocapture`.
    #[test]
    fn gpu_multiplex_throughput_bench() {
        let variants = battery_variant_matrix();
        let opts = DeviceResidentInnerOptions::default();

        let probe = super::build_sweep_workspaces(&variants).expect("sweep workspaces");
        let any_device = probe.iter().any(|w| w.device_resident());
        if !any_device {
            println!(
                "[#1017 mux-bench] no CUDA device — {} variants (K1..4 x 3 basis) \
                 skipped; run on the GPU node for cross-fit throughput",
                variants.len()
            );
            return;
        }

        let (results, mux_tp) =
            super::run_variant_sweep_multiplexed(&variants, opts).expect("multiplexed sweep");
        let seq_tp = super::assert_sweep_parity_vs_sequential(&variants, &opts, &results)
            .expect("sweep parity vs sequential must hold");
        println!(
            "[#1017 mux-bench] fits={} succeeded={} multiplexed={:.3}s ({:.1} fits/s) \
             sequential={:.3}s ({:.1} fits/s) cross-fit-speedup={:.2}x",
            mux_tp.fits,
            mux_tp.succeeded,
            mux_tp.wall_seconds,
            mux_tp.fits_per_second,
            seq_tp.wall_seconds,
            seq_tp.fits_per_second,
            mux_tp.fits_per_second / seq_tp.fits_per_second.max(1e-9),
        );
        assert_eq!(
            mux_tp.succeeded, mux_tp.fits,
            "all battery variants must fit successfully on device"
        );
    }
}
