//! Survival marginal-slope rigid per-row V/G/H jet on the GPU.
//!
//! The production cache builder requests exactly the order-2 channels
//! `(value, gradient[4], Hessian[4][4])`. Large admitted batches execute the
//! order-2 CUDA lowering of the canonical five-feature row program followed by
//! its mechanical four-primary pullback; smaller or
//! unavailable-device batches use the ordinary per-row cache path. Contracted
//! third/fourth derivatives have separate live CPU consumers whose directions
//! vary by row and are intentionally not part of this batch API.
//!
//! The CUDA leaf uses native full-precision `erfc`, while NVRTC compilation
//! disables FMA contraction for close agreement with separately rounded host
//! arithmetic. Direct device tests cover both ordinary and probability-tail
//! rows against the CPU row program.

#[cfg(target_os = "linux")]
use crate::survival::marginal_slope::RIGID_FEATURE_PROGRAM_CUDA_VGH;
#[cfg(target_os = "linux")]
use cudarc::nvrtc::Ptx;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuError;

/// Flattened row-major value, gradient, and Hessian channels for `K = 4`.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SurvivalRowVghChannels {
    pub(crate) value: Vec<f64>,
    pub(crate) grad: Vec<f64>,
    pub(crate) hess: Vec<f64>,
}

/// Scalar-independent inputs for one rigid survival row.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone)]
pub(crate) struct SurvivalRowInputs {
    pub(crate) primaries: [f64; 4],
    pub(crate) wi: f64,
    pub(crate) di: f64,
    pub(crate) z_sum: f64,
    pub(crate) cov_ones: f64,
}

/// Minimum row count that amortises probe, transfer, and launch costs.
const DEVICE_ROW_THRESHOLD: usize = 100_000;

/// Whether this batch is admitted to the production CUDA V/G/H path.
///
/// Admission is a capability decision made before execution, not an
/// operating-system guess. A large CPU-only Linux fit therefore stays on the
/// ordinary row-kernel schedule; once a real device is admitted, subsequent
/// compile/launch failures remain errors and are never hidden by a retry.
#[inline]
pub(crate) fn survival_rigid_row_vgh_device_selected(n_rows: usize) -> Result<bool, String> {
    if n_rows < DEVICE_ROW_THRESHOLD {
        return Ok(false);
    }
    gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())
        .map(|runtime| runtime.is_some())
        .map_err(String::from)
}

/// Execute an already-admitted production V/G/H batch on CUDA.
#[cfg(target_os = "linux")]
#[must_use]
pub(crate) fn survival_rigid_row_vgh(
    rows: &[SurvivalRowInputs],
    probit_scale: f64,
) -> Result<SurvivalRowVghChannels, String> {
    gam_gpu::device_runtime::GpuRuntime::require()
        .map_err(|error| format!("survival VGH CUDA execution requires a device: {error}"))?;
    device::survival_rigid_row_vgh_device(rows, probit_scale)
        .map_err(|error| format!("survival VGH device execution failed: {error}"))
}

/// CUDA substrate for the four rigid survival primaries. The stable primitive
/// leaves and launch plumbing live here; the algebraic row schedule and its
/// nonzero value/gradient/packed-Hessian expressions are generated from the
/// canonical Rust declaration.
#[cfg(target_os = "linux")]
const SURVIVAL_ROWJET_TEMPLATE: &str = include_str!("survival_rowjet_kernel.cu");

#[cfg(target_os = "linux")]
const ROW_PROGRAM_MARKER: &str = "// __GAM_ROW_PROGRAM_CUDA_VGH__";

/// Mechanical `(q0,q1,qd1,g) -> (q0,q1,qd1,L,V)` order-two pullback for the
/// generated CUDA feature evaluator. This contains only the feature map and
/// chain rule; the likelihood expression exists solely in the row program.
#[cfg(target_os = "linux")]
const RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA: &str = r#"
__device__ __forceinline__ void rigid_feature_program_pullback4(
        double q0,
        double q1,
        double qd1,
        double g,
        const RowIn& in,
        double* row_value,
        double* row_gradient,
        double* row_hessian) {
    const double observed_g = in.probit_scale * g;
    const double linear = observed_g * in.z_sum;
    const double variance = (g * g) * in.covariance_ones;
    double feature_gradient[9];
    double feature_hessian[81];
    // The static-slope feature frame: a time-constant slope reaches the entry
    // and exit location channels through the SAME functional of `g`, likewise
    // the two variance channels, and the three follow-up-rate channels are
    // identically zero. Mirror of `static_slope_feature_frame` on the host.
    rigid_feature_program(
        q0,
        q1,
        qd1,
        linear,
        linear,
        0.0,
        variance,
        variance,
        0.0,
        in,
        row_value,
        feature_gradient,
        feature_hessian);

    const double d_linear = in.probit_scale * in.z_sum;
    const double d_variance = 2.0 * g * in.covariance_ones;
    // Slot order and summation order match `STATIC_SLOPE_ACTIVE_FEATURES` and
    // `order2_feature_pullback_into` on the host exactly, so the device and CPU
    // lowerings of the same declaration agree bit for bit.
    const int active[4] = {3, 4, 6, 7};
    const double active_jacobian[4] = {d_linear, d_linear, d_variance, d_variance};

    row_gradient[0] = feature_gradient[0];
    row_gradient[1] = feature_gradient[1];
    row_gradient[2] = feature_gradient[2];
    double slope_gradient = 0.0;
    for (int slot = 0; slot < 4; ++slot) {
        slope_gradient += feature_gradient[active[slot]] * active_jacobian[slot];
    }
    row_gradient[3] = slope_gradient;

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            row_hessian[a * 4 + b] = feature_hessian[a * 9 + b];
        }
        double channel = 0.0;
        for (int slot = 0; slot < 4; ++slot) {
            channel += feature_hessian[a * 9 + active[slot]] * active_jacobian[slot];
        }
        row_hessian[a * 4 + 3] = channel;
        row_hessian[3 * 4 + a] = channel;
    }

    double slope_curvature = 0.0;
    for (int left = 0; left < 4; ++left) {
        const double left_jacobian = active_jacobian[left];
        for (int right = 0; right < 4; ++right) {
            slope_curvature += left_jacobian
                * feature_hessian[active[left] * 9 + active[right]]
                * active_jacobian[right];
        }
    }
    // The feature map's own curvature: `d2 V_k / d g2 = 2 * covariance_ones` on
    // both variance channels; the location channels are linear in `g`.
    slope_curvature += (feature_gradient[6] + feature_gradient[7]) * 2.0 * in.covariance_ones;
    row_hessian[15] = slope_curvature;
}
"#;

#[cfg(target_os = "linux")]
fn survival_rowjet_source() -> &'static str {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE.get_or_init(|| {
        let (preamble, kernel) = SURVIVAL_ROWJET_TEMPLATE
            .split_once(ROW_PROGRAM_MARKER)
            .expect("survival rowjet CUDA template must contain the row-program marker");
        assert!(
            !kernel.contains(ROW_PROGRAM_MARKER),
            "survival rowjet CUDA template must contain exactly one row-program marker",
        );
        let mut source = String::with_capacity(
            preamble.len()
                + RIGID_FEATURE_PROGRAM_CUDA_VGH.len()
                + RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA.len()
                + kernel.len(),
        );
        source.push_str(preamble);
        source.push_str(RIGID_FEATURE_PROGRAM_CUDA_VGH);
        source.push_str(RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA);
        source.push_str(kernel);
        source
    })
}

/// Compile the exact CUDA source used by the production survival V/G/H module.
#[cfg(target_os = "linux")]
pub fn compile_survival_rowjet_ptx() -> Result<Ptx, GpuError> {
    gam_gpu::device_cache::compile_ptx_arch(survival_rowjet_source())
}

#[cfg(target_os = "linux")]
mod device {
    use super::{SurvivalRowInputs, SurvivalRowVghChannels, compile_survival_rowjet_ptx};
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

    struct Backend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        module: Mutex<Option<Arc<CudaModule>>>,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("survival_rowjet")?;
                Ok(Backend {
                    ctx: parts.ctx,
                    stream: parts.stream,
                    module: Mutex::new(None),
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    fn module(backend: &Backend) -> Result<Arc<CudaModule>, GpuError> {
        if let Ok(guard) = backend.module.lock() {
            if let Some(module) = guard.as_ref() {
                return Ok(module.clone());
            }
        }
        // The shared compiler pins the real device architecture and disables
        // FMA contraction for close parity with separately rounded host ops.
        let ptx = compile_survival_rowjet_ptx()
            .gpu_ctx_with(|error| format!("survival_rowjet NVRTC compile: {error}"))?;
        let module = backend
            .ctx
            .load_module(ptx)
            .gpu_ctx("survival_rowjet module load")?;
        if let Ok(mut guard) = backend.module.lock() {
            guard.get_or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    type FlatInputs = (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    );

    fn flatten_inputs(rows: &[SurvivalRowInputs]) -> FlatInputs {
        let n = rows.len();
        let mut q0 = Vec::with_capacity(n);
        let mut q1 = Vec::with_capacity(n);
        let mut qd1 = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut wi = Vec::with_capacity(n);
        let mut di = Vec::with_capacity(n);
        let mut z_sum = Vec::with_capacity(n);
        let mut cov_ones = Vec::with_capacity(n);
        for row in rows {
            q0.push(row.primaries[0]);
            q1.push(row.primaries[1]);
            qd1.push(row.primaries[2]);
            g.push(row.primaries[3]);
            wi.push(row.wi);
            di.push(row.di);
            z_sum.push(row.z_sum);
            cov_ones.push(row.cov_ones);
        }
        (q0, q1, qd1, g, wi, di, z_sum, cov_ones)
    }

    pub(super) fn survival_rigid_row_vgh_device(
        rows: &[SurvivalRowInputs],
        probit_scale: f64,
    ) -> Result<SurvivalRowVghChannels, GpuError> {
        let n = rows.len();
        if n == 0 {
            return Ok(SurvivalRowVghChannels {
                value: Vec::new(),
                grad: Vec::new(),
                hess: Vec::new(),
            });
        }
        let backend = backend()?;
        let module = module(backend)?;
        let function = module
            .load_function("survival_rowjet_vgh")
            .gpu_ctx("survival_rowjet_vgh load_function")?;
        let stream = backend.stream.clone();
        let (q0, q1, qd1, g, wi, di, z_sum, cov_ones) = flatten_inputs(rows);
        let q0_device = stream.clone_htod(&q0).gpu_ctx("vgh htod q0")?;
        let q1_device = stream.clone_htod(&q1).gpu_ctx("vgh htod q1")?;
        let qd1_device = stream.clone_htod(&qd1).gpu_ctx("vgh htod qd1")?;
        let g_device = stream.clone_htod(&g).gpu_ctx("vgh htod g")?;
        let wi_device = stream.clone_htod(&wi).gpu_ctx("vgh htod wi")?;
        let di_device = stream.clone_htod(&di).gpu_ctx("vgh htod di")?;
        let z_sum_device = stream.clone_htod(&z_sum).gpu_ctx("vgh htod z_sum")?;
        let cov_ones_device = stream.clone_htod(&cov_ones).gpu_ctx("vgh htod cov_ones")?;
        let mut value_device = stream.alloc_zeros::<f64>(n).gpu_ctx("vgh alloc value")?;
        let mut grad_device = stream.alloc_zeros::<f64>(n * 4).gpu_ctx("vgh alloc grad")?;
        let mut hess_device = stream
            .alloc_zeros::<f64>(n * 16)
            .gpu_ctx("vgh alloc hess")?;

        let n_i32 = i32::try_from(n)
            .map_err(|_| gam_gpu::gpu_err!("survival_rowjet_vgh n={n} overflows i32"))?;
        const THREADS_PER_BLOCK: u32 = 128;
        let config = LaunchConfig {
            grid_dim: (((n as u32).div_ceil(THREADS_PER_BLOCK)).max(1), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&n_i32)
            .arg(&q0_device)
            .arg(&q1_device)
            .arg(&qd1_device)
            .arg(&g_device)
            .arg(&wi_device)
            .arg(&di_device)
            .arg(&z_sum_device)
            .arg(&cov_ones_device)
            .arg(&probit_scale)
            .arg(&mut value_device)
            .arg(&mut grad_device)
            .arg(&mut hess_device);
        // SAFETY: all device slices match the kernel signature and lengths; the
        // kernel bounds-checks the final partial block.
        unsafe { builder.launch(config) }.gpu_ctx("survival_rowjet_vgh kernel launch")?;

        let mut value = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n * 4];
        let mut hess = vec![0.0_f64; n * 16];
        stream
            .memcpy_dtoh(&value_device, &mut value)
            .gpu_ctx("vgh dtoh value")?;
        stream
            .memcpy_dtoh(&grad_device, &mut grad)
            .gpu_ctx("vgh dtoh grad")?;
        stream
            .memcpy_dtoh(&hess_device, &mut hess)
            .gpu_ctx("vgh dtoh hess")?;
        stream
            .synchronize()
            .gpu_ctx("survival_rowjet_vgh synchronize")?;
        Ok(SurvivalRowVghChannels { value, grad, hess })
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_source_exports_only_the_production_vgh_kernel() {
        let source = survival_rowjet_source();
        assert_eq!(
            SURVIVAL_ROWJET_TEMPLATE.matches(ROW_PROGRAM_MARKER).count(),
            1
        );
        assert!(!SURVIVAL_ROWJET_TEMPLATE.contains("struct J2"));
        assert!(RIGID_FEATURE_PROGRAM_CUDA_VGH.contains("void rigid_feature_program"));
        assert!(
            RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA.contains("void rigid_feature_program_pullback4")
        );
        assert!(RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA.contains("rigid_feature_program("));
        assert!(!RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA.contains("neglog_phi"));
        assert!(!RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA.contains("log_normal_pdf"));
        assert!(!RIGID_FEATURE_PROGRAM_PULLBACK4_CUDA.contains("d_sqrt"));
        assert!(!RIGID_FEATURE_PROGRAM_CUDA_VGH.contains("j2_"));
        assert!(!RIGID_FEATURE_PROGRAM_CUDA_VGH.contains("* 0.0"));
        assert!(!RIGID_FEATURE_PROGRAM_CUDA_VGH.contains("0.0 *"));
        assert!(source.contains("survival_rowjet_vgh"));
        assert_eq!(source.matches("void rigid_feature_program(").count(), 1);
        assert!(!source.contains(concat!("rigid_row_", "program")));
        assert_eq!(source.matches("extern \"C\" __global__").count(), 1,);
        for removed in [
            "survival_rowjet_no_t4",
            "struct JS1",
            "struct JS2",
            "struct J2",
            "j2_",
            "nll_j2",
            "nll_js1",
            "nll_js2",
        ] {
            assert!(
                !source.contains(removed),
                "dead CUDA surface reintroduced: {removed}",
            );
        }
    }
}
