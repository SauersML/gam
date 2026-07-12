//! Survival marginal-slope rigid per-row V/G/H jet on the GPU.
//!
//! The production cache builder requests exactly the order-2 channels
//! `(value, gradient[4], Hessian[4][4])`. Large admitted batches execute the
//! order-2 CUDA lowering of
//! [`crate::survival::marginal_slope::row_kernel::rigid_row_nll`]; smaller or
//! unavailable-device batches use the ordinary per-row cache path. Contracted
//! third/fourth derivatives have separate live CPU consumers whose directions
//! vary by row and are intentionally not part of this batch API.
//!
//! The CUDA leaf uses native full-precision `erfc`, while NVRTC compilation
//! disables FMA contraction for close agreement with separately rounded host
//! arithmetic. Direct device tests cover both ordinary and probability-tail
//! rows against the CPU row program.

#[cfg(target_os = "linux")]
use crate::survival::marginal_slope::row_kernel::RIGID_ROW_PROGRAM_CUDA_VGH;

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
#[inline]
#[must_use]
pub(crate) const fn survival_rigid_row_vgh_device_selected(n_rows: usize) -> bool {
    cfg!(target_os = "linux") && n_rows >= DEVICE_ROW_THRESHOLD
}

/// Execute an already-admitted production V/G/H batch on CUDA.
#[cfg(target_os = "linux")]
#[must_use]
pub(crate) fn survival_rigid_row_vgh(
    rows: &[SurvivalRowInputs],
    probit_scale: f64,
) -> Result<SurvivalRowVghChannels, String> {
    assert!(
        survival_rigid_row_vgh_device_selected(rows.len()),
        "survival VGH CUDA execution requires an admitted batch",
    );
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
        let mut source =
            String::with_capacity(preamble.len() + RIGID_ROW_PROGRAM_CUDA_VGH.len() + kernel.len());
        source.push_str(preamble);
        source.push_str(RIGID_ROW_PROGRAM_CUDA_VGH);
        source.push_str(kernel);
        source
    })
}

#[cfg(target_os = "linux")]
mod device {
    use super::{SurvivalRowInputs, SurvivalRowVghChannels, survival_rowjet_source};
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
        let ptx = gam_gpu::device_cache::compile_ptx_arch(survival_rowjet_source())
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
    use crate::survival::marginal_slope::row_kernel::RigidRowInputs;
    use gam_math::nested_dual::JetField;

    #[inline]
    fn rigid_cpu_row_inputs(
        row: usize,
        input: &SurvivalRowInputs,
        probit_scale: f64,
    ) -> RigidRowInputs {
        RigidRowInputs {
            row,
            wi: input.wi,
            di: input.di,
            z_sum: input.z_sum,
            covariance_ones: input.cov_ones,
            probit_scale,
            // The batch caller validates the monotonicity guard before dispatch.
            qd1_lower: f64::NEG_INFINITY,
        }
    }

    /// CPU execution of the canonical row program at its order-2 scalar.
    #[must_use]
    fn survival_rigid_row_vgh_cpu(
        rows: &[SurvivalRowInputs],
        probit_scale: f64,
    ) -> SurvivalRowVghChannels {
        use crate::survival::marginal_slope::row_kernel::{
            RIGID_LINEAR_MASK, SparseOrder2, rigid_row_nll,
        };
        use gam_math::jet_scalar::JetScalar;

        let n = rows.len();
        let mut value = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n * 4];
        let mut hess = vec![0.0_f64; n * 16];
        for (row, input) in rows.iter().enumerate() {
            let in_row = rigid_cpu_row_inputs(row, input, probit_scale);
            let p = input.primaries;
            let vars: [SparseOrder2<RIGID_LINEAR_MASK>; 4] =
                std::array::from_fn(|axis| SparseOrder2::variable(p[axis], axis));
            if let Ok(out) = rigid_row_nll(&vars, &in_row) {
                value[row] = out.value();
                grad[row * 4..row * 4 + 4].copy_from_slice(&out.g());
                let row_hessian = out.h();
                for a in 0..4 {
                    hess[row * 16 + a * 4..row * 16 + a * 4 + 4].copy_from_slice(&row_hessian[a]);
                }
            }
        }
        SurvivalRowVghChannels { value, grad, hess }
    }

    #[cfg(target_os = "linux")]
    fn survival_rigid_row_vgh_device_only(
        rows: &[SurvivalRowInputs],
        probit_scale: f64,
    ) -> Result<SurvivalRowVghChannels, String> {
        device::survival_rigid_row_vgh_device(rows, probit_scale).map_err(|error| error.to_string())
    }

    fn fixture(n: usize) -> Vec<SurvivalRowInputs> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                SurvivalRowInputs {
                    primaries: [
                        -2.5 + 5.0 * (12.0 * t).sin(),
                        -1.5 + 4.0 * (9.0 * t + 0.3).cos(),
                        0.2 + 1.8 * (0.5 + 0.5 * (7.0 * t).sin()),
                        -1.0 + 2.0 * (5.0 * t + 1.1).sin(),
                    ],
                    wi: 1.0,
                    di: if i % 3 == 0 { 1.0 } else { 0.0 },
                    z_sum: 0.5 * (3.0 * t).cos(),
                    cov_ones: 0.4 + 0.3 * (0.5 + 0.5 * (2.0 * t).sin()),
                }
            })
            .collect()
    }

    fn edge_fixture() -> Vec<SurvivalRowInputs> {
        let row = |primaries, wi, di, z_sum, cov_ones| SurvivalRowInputs {
            primaries,
            wi,
            di,
            z_sum,
            cov_ones,
        };
        vec![
            row([-0.4, 0.6, 0.9, 0.3], 1.0, 1.0, 0.2, 0.5),
            row([-0.4, 0.6, 0.9, 0.3], 1.0, 0.0, 0.2, 0.5),
            row([8.0, 9.0, 1.2, 2.5], 1.0, 0.0, -3.0, 1.0),
            row([-8.0, -9.0, 1.2, -2.5], 1.0, 1.0, 3.0, 1.0),
            row([40.0, 41.0, 0.7, 3.0], 1.0, 0.0, 0.0, 2.0),
            row([-0.3, 0.5, 0.8, 1.5], 1.0, 1.0, 0.4, 1e-10),
            row([-0.2, 0.4, 1.1, 4.0], 1.0, 1.0, 0.1, 50.0),
            row([-0.5, 0.3, 0.6, 1e-9], 1.0, 0.0, 0.7, 0.9),
            row([-0.5, 0.3, 0.6, 0.4], 0.0, 1.0, 0.7, 0.9),
            row([-0.5, 0.3, 1e-3, 0.4], 1.0, 1.0, 0.2, 0.6),
        ]
    }

    #[test]
    fn cpu_vgh_matches_canonical_dense_order2() {
        use crate::survival::marginal_slope::row_kernel::rigid_row_nll;
        use gam_math::jet_scalar::{JetScalar, Order2};

        let rows = fixture(64);
        let out = survival_rigid_row_vgh_cpu(&rows, 0.7);
        for (row, input) in rows.iter().enumerate() {
            let row_inputs = rigid_cpu_row_inputs(row, input, 0.7);
            let variables: [Order2<4>; 4] =
                std::array::from_fn(|axis| Order2::variable(input.primaries[axis], axis));
            let expected = rigid_row_nll(&variables, &row_inputs).expect("dense order-2 row");
            assert!((expected.value() - out.value[row]).abs() <= 1e-12);
            for a in 0..4 {
                assert!((expected.g()[a] - out.grad[row * 4 + a]).abs() <= 1e-12);
                for b in 0..4 {
                    assert!(
                        (expected.h()[a][b] - out.hess[row * 16 + a * 4 + b]).abs() <= 1e-12,
                        "Hessian mismatch at row {row}, ({a}, {b})",
                    );
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    const PARITY_ABS_TOLERANCE: f64 = 1e-9;
    #[cfg(target_os = "linux")]
    const PARITY_REL_TOLERANCE: f64 = 1e-7;

    #[cfg(target_os = "linux")]
    fn assert_channel_parity(name: &str, cpu: &[f64], device: &[f64]) {
        assert_eq!(cpu.len(), device.len(), "{name} channel length");
        let scale = cpu
            .iter()
            .fold(0.0_f64, |current, value| current.max(value.abs()));
        let tolerance = PARITY_ABS_TOLERANCE + PARITY_REL_TOLERANCE * scale;
        let (worst_index, worst) = cpu
            .iter()
            .zip(device)
            .enumerate()
            .map(|(index, (left, right))| (index, (left - right).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((0, 0.0));
        assert!(
            worst <= tolerance,
            "survival VGH {name} device drift {worst:.3e} at {worst_index} exceeds \
             {tolerance:.3e} (scale {scale:.3e})",
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn admitted_dispatch_and_device_path_match_cpu_vgh() {
        let rows = fixture(DEVICE_ROW_THRESHOLD + 1024);
        let cpu = survival_rigid_row_vgh_cpu(&rows, 0.7);
        let dispatched = survival_rigid_row_vgh(&rows, 0.7).expect("admitted CUDA VGH batch");
        assert_channel_parity("dispatched value", &cpu.value, &dispatched.value);
        assert_channel_parity("dispatched gradient", &cpu.grad, &dispatched.grad);
        assert_channel_parity("dispatched Hessian", &cpu.hess, &dispatched.hess);

        if gam_gpu::device_runtime::GpuRuntime::global().is_some() {
            let device = survival_rigid_row_vgh_device_only(&rows, 0.7)
                .expect("CUDA runtime present but survival VGH device path failed");
            assert_channel_parity("device value", &cpu.value, &device.value);
            assert_channel_parity("device gradient", &cpu.grad, &device.grad);
            assert_channel_parity("device Hessian", &cpu.hess, &device.hess);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_only_vgh_matches_cpu_in_edge_regimes() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("CUDA runtime unavailable; skipping direct survival VGH device parity");
            return;
        }
        let rows = edge_fixture();
        let cpu = survival_rigid_row_vgh_cpu(&rows, 0.7);
        let device = survival_rigid_row_vgh_device_only(&rows, 0.7)
            .expect("CUDA runtime present but survival VGH edge sweep failed");
        assert_channel_parity("edge value", &cpu.value, &device.value);
        assert_channel_parity("edge gradient", &cpu.grad, &device.grad);
        assert_channel_parity("edge Hessian", &cpu.hess, &device.hess);
    }

    /// #932 production-boundary throughput measurement. The timed device call
    /// includes allocation, all host/device transfers, launch, and synchronize;
    /// module compilation is warmed outside the timing. This is intentionally
    /// stricter than a kernel-only number and reports whether CUDA wins at the
    /// exact API the row-kernel cache consumes.
    #[cfg(target_os = "linux")]
    #[test]
    fn measure_device_vgh_end_to_end_932() {
        use std::time::{Duration, Instant};

        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("CUDA runtime unavailable; skipping survival VGH throughput measurement");
            return;
        }
        const ROWS: usize = 1_000_000;
        let rows = fixture(ROWS);
        let warm =
            survival_rigid_row_vgh_device_only(&rows, 0.7).expect("warm survival VGH device call");

        let cpu_start = Instant::now();
        let cpu = survival_rigid_row_vgh_cpu(&rows, 0.7);
        let cpu_elapsed = cpu_start.elapsed();

        let mut best_elapsed = Duration::MAX;
        let mut best_device = warm;
        for round in 0..3 {
            std::hint::black_box(round);
            let device_start = Instant::now();
            let candidate = survival_rigid_row_vgh_device_only(&rows, 0.7)
                .expect("timed survival VGH device call");
            let elapsed = device_start.elapsed();
            if elapsed < best_elapsed {
                best_elapsed = elapsed;
                best_device = candidate;
            }
        }

        assert_channel_parity("measured value", &cpu.value, &best_device.value);
        assert_channel_parity("measured gradient", &cpu.grad, &best_device.grad);
        assert_channel_parity("measured Hessian", &cpu.hess, &best_device.hess);
        let cpu_ns = cpu_elapsed.as_secs_f64() * 1e9 / ROWS as f64;
        let device_ns = best_elapsed.as_secs_f64() * 1e9 / ROWS as f64;
        eprintln!(
            "SURVIVAL-VGH-CUDA-932 rows={ROWS} cpu={cpu_ns:.2} ns/row device-e2e={device_ns:.2} ns/row speedup={:.2}x",
            cpu_ns / device_ns,
        );
        assert!(cpu_ns.is_finite() && device_ns.is_finite() && device_ns > 0.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cuda_source_exports_only_the_production_vgh_kernel() {
        let source = survival_rowjet_source();
        assert_eq!(
            SURVIVAL_ROWJET_TEMPLATE.matches(ROW_PROGRAM_MARKER).count(),
            1
        );
        assert!(!SURVIVAL_ROWJET_TEMPLATE.contains("struct J2"));
        assert!(RIGID_ROW_PROGRAM_CUDA_VGH.contains("void rigid_row_program"));
        assert!(!RIGID_ROW_PROGRAM_CUDA_VGH.contains("j2_"));
        assert!(!RIGID_ROW_PROGRAM_CUDA_VGH.contains("* 0.0"));
        assert!(!RIGID_ROW_PROGRAM_CUDA_VGH.contains("0.0 *"));
        assert!(source.contains("survival_rowjet_vgh"));
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
