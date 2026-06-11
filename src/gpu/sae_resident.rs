//! Device-resident SAE inner-iteration workspace for issue #1017.
//!
//! This first vertical slice keeps production fitting untouched. It accepts
//! host-evaluated SAE basis/gate values plus already-assembled data-fit
//! Arrow-Schur slabs, uploads those buffers once, and runs one Newton step
//! through the existing GPU Arrow-Schur sequence when the runtime probe admits
//! the workload. Later slices can replace the host slab feed with on-device
//! basis/gate evaluation without changing the public step API.

use ndarray::Array1;

use crate::gpu::arrow_schur::{
    ArrowSchurGpuFailure, solve_arrow_newton_step, solve_arrow_newton_step_dense_reference,
};
use crate::solver::arrow_schur::{ArrowSchurError, ArrowSchurSystem};

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
    pub used_device: bool,
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
            .map(|solution| self.finish_step(solution, true))
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
            .map(|solution| self.finish_step(solution, false))
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
        solution: crate::gpu::arrow_schur::ArrowSchurGpuSolution,
        used_device: bool,
    ) -> DeviceResidentArrowStep {
        DeviceResidentArrowStep {
            delta_t: solution.delta_t,
            delta_beta: solution.delta_beta,
            objective: 0.5 * squared_norm(&self.target_x),
            gradient_norm: self.gradient_norm(),
            log_det_hessian: solution.log_det_hessian,
            used_device,
        }
    }

    fn gradient_norm(&self) -> f64 {
        let row = squared_norm(&self.slabs.row_gradient_slabs);
        let border = squared_norm(&self.slabs.border_gradient);
        (row + border).sqrt()
    }
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
    use crate::gpu::linalg::{DispatchOp, route_through_gpu};

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
    let ctx = crate::gpu::runtime::cuda_context_for(runtime.device.ordinal)?;
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
    let shape = DeviceResidentArrowShape::qwen_non_gating();
    let mut rng = SplitMix64::new(0x1017_0003_D3A1_5EED);
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
        let h_base = i * shape.d * shape.d;
        row_hessian_slabs[h_base] = 3.0 + 0.01 * basis_sum.abs();
        row_hessian_slabs[h_base + 1] = 0.02 * basis_sum.sin();
        row_hessian_slabs[h_base + 2] = row_hessian_slabs[h_base + 1];
        row_hessian_slabs[h_base + 3] = 2.5 + 0.01 * basis_sum.abs();
        let b_base = i * shape.d * shape.p;
        for j in 0..shape.p {
            let feature = ((j % 257) as f64) * 0.011;
            row_cross_slabs[b_base + j] = 1.0e-4 * basis_sum.sin() * feature.cos();
            row_cross_slabs[b_base + shape.p + j] = 1.0e-4 * basis_sum.cos() * feature.sin();
        }
        let g_base = i * shape.d;
        row_gradient_slabs[g_base] = 0.01 * basis_sum.sin();
        row_gradient_slabs[g_base + 1] = 0.01 * basis_sum.cos();
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

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn sample_signed(&mut self) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
        2.0 * unit - 1.0
    }
}
