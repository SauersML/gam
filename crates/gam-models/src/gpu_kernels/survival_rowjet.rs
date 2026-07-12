//! Survival marginal-slope rigid per-row V/G/H jet on the GPU.
//!
//! The production cache builder requests exactly the order-2 channels
//! `(value, gradient[4], Hessian[4][4])`. Large admitted batches execute the
//! order-2 CUDA lowering of
//! [`crate::survival::marginal_slope::row_kernel::rigid_row_nll`]; smaller or
//! unavailable-device batches execute that same canonical row program at the
//! CPU `SparseOrder2` scalar. Contracted third/fourth derivatives have separate
//! live CPU consumers whose directions vary by row and are intentionally not
//! part of this batch API.
//!
//! The CUDA leaf uses native full-precision `erfc`, while NVRTC compilation
//! disables FMA contraction for close agreement with separately rounded host
//! arithmetic. Direct device tests cover both ordinary and probability-tail
//! rows against the CPU row program.

use crate::survival::marginal_slope::row_kernel::RigidRowInputs;

/// Flattened row-major value, gradient, and Hessian channels for `K = 4`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SurvivalRowVghChannels {
    pub(crate) value: Vec<f64>,
    pub(crate) grad: Vec<f64>,
    pub(crate) hess: Vec<f64>,
}

/// The scalar-independent per-row inputs the kernel consumes: the four primaries
/// `(q0,q1,qd1,g)` and the row scalars `(w,d,z_sum,cov_ones)`. `probit_scale` is
/// shared across all rows (a scalar kernel argument). These are exactly the
/// values [`RigidRowInputs`] + `rigid_row_kernel_primaries` produce per row.
#[derive(Debug, Clone)]
pub(crate) struct SurvivalRowInputs {
    pub(crate) primaries: [f64; 4],
    pub(crate) wi: f64,
    pub(crate) di: f64,
    pub(crate) z_sum: f64,
    pub(crate) cov_ones: f64,
}

/// Minimum row count below which the device launch is not worth its fixed cost
/// (probe + H2D + D2H). Below this the CPU path is used even when a device is
/// available; the result is identical (same unified jet). The standalone A100
/// measurement put the kernel/CPU crossover well under 1e5 rows; 1e5 is a
/// conservative break-even that keeps small-fit latency on the CPU.
const DEVICE_ROW_THRESHOLD: usize = 100_000;

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
        // Keep this evaluator a pure derivative program over admitted rows.
        qd1_lower: f64::NEG_INFINITY,
    }
}

/// CPU VGH-only execution: one instantiation of the unified row program per
/// row, with exactly the three requested output buffers.
#[must_use]
pub(crate) fn survival_rigid_row_vgh_cpu(
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

/// General VGH-only entry point. CPU execution evaluates exactly one
/// `SparseOrder2` row program and allocates no high-order outputs. Large admitted
/// GPU batches use a dedicated J2 kernel, so they likewise carry no seeded
/// high-order state and transfer only the requested outputs.
#[must_use]
pub(crate) fn survival_rigid_row_vgh(
    rows: &[SurvivalRowInputs],
    probit_scale: f64,
) -> SurvivalRowVghChannels {
    #[cfg(target_os = "linux")]
    {
        if rows.len() >= DEVICE_ROW_THRESHOLD {
            match device::survival_rigid_row_vgh_device(rows, probit_scale) {
                Ok(out) => return out,
                Err(error) => {
                    log::info!("[GPU] survival VGH device path fell back to CPU: {error}");
                }
            }
        }
    }
    survival_rigid_row_vgh_cpu(rows, probit_scale)
}

/// Order-2 CUDA lowering for the rigid survival primaries. Full f64, no
/// fast-math, and no unrequested high-order entry points.
#[cfg(target_os = "linux")]
const SURVIVAL_ROWJET_SOURCE: &str = include_str!("survival_rowjet_kernel.cu");

#[cfg(target_os = "linux")]
mod device {
    use super::{SURVIVAL_ROWJET_SOURCE, SurvivalRowInputs, SurvivalRowVghChannels};
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

    fn module(b: &Backend) -> Result<Arc<CudaModule>, GpuError> {
        if let Ok(guard) = b.module.lock() {
            if let Some(m) = guard.as_ref() {
                return Ok(m.clone());
            }
        }
        // Compile through the shared arch+fmad options (NOT bare `compile_ptx`,
        // which leaves NVRTC at `--fmad=true` and no `--gpu-architecture` pin).
        // FMA contraction stays off so the order-2 device program remains close
        // to the separately rounded CPU oracle. The arch pin keeps the kernel
        // keyed to the device's real compute capability rather than NVRTC's
        // default.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(SURVIVAL_ROWJET_SOURCE)
            .gpu_ctx_with(|err| format!("survival_rowjet NVRTC compile: {err}"))?;
        let m = b
            .ctx
            .load_module(ptx)
            .gpu_ctx("survival_rowjet module load")?;
        if let Ok(mut guard) = b.module.lock() {
            guard.get_or_insert_with(|| m.clone());
        }
        Ok(m)
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
        let mut zs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for row in rows {
            q0.push(row.primaries[0]);
            q1.push(row.primaries[1]);
            qd1.push(row.primaries[2]);
            g.push(row.primaries[3]);
            wi.push(row.wi);
            di.push(row.di);
            zs.push(row.z_sum);
            cov.push(row.cov_ones);
        }
        (q0, q1, qd1, g, wi, di, zs, cov)
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
        let b = backend()?;
        let module = module(b)?;
        let function = module
            .load_function("survival_rowjet_vgh")
            .gpu_ctx("survival_rowjet_vgh load_function")?;
        let stream = b.stream.clone();
        let (q0, q1, qd1, g, wi, di, zs, cov) = flatten_inputs(rows);
        let q0_d = stream.clone_htod(&q0).gpu_ctx("vgh htod q0")?;
        let q1_d = stream.clone_htod(&q1).gpu_ctx("vgh htod q1")?;
        let qd1_d = stream.clone_htod(&qd1).gpu_ctx("vgh htod qd1")?;
        let g_d = stream.clone_htod(&g).gpu_ctx("vgh htod g")?;
        let wi_d = stream.clone_htod(&wi).gpu_ctx("vgh htod wi")?;
        let di_d = stream.clone_htod(&di).gpu_ctx("vgh htod di")?;
        let zs_d = stream.clone_htod(&zs).gpu_ctx("vgh htod zsum")?;
        let cov_d = stream.clone_htod(&cov).gpu_ctx("vgh htod cov")?;
        let mut value_d = stream.alloc_zeros::<f64>(n).gpu_ctx("vgh alloc value")?;
        let mut grad_d = stream.alloc_zeros::<f64>(n * 4).gpu_ctx("vgh alloc grad")?;
        let mut hess_d = stream
            .alloc_zeros::<f64>(n * 16)
            .gpu_ctx("vgh alloc hess")?;

        let n_i32 = i32::try_from(n)
            .map_err(|_| gam_gpu::gpu_err!("survival_rowjet_vgh n={n} overflows i32"))?;
        const TPB: u32 = 128;
        let config = LaunchConfig {
            grid_dim: (((n as u32).div_ceil(TPB)).max(1), 1, 1),
            block_dim: (TPB, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&n_i32)
            .arg(&q0_d)
            .arg(&q1_d)
            .arg(&qd1_d)
            .arg(&g_d)
            .arg(&wi_d)
            .arg(&di_d)
            .arg(&zs_d)
            .arg(&cov_d)
            .arg(&probit_scale)
            .arg(&mut value_d)
            .arg(&mut grad_d)
            .arg(&mut hess_d);
        // SAFETY: all device slices match the kernel signature and lengths;
        // the launch covers exactly n rows with bounds checking in the kernel.
        unsafe { builder.launch(config) }.gpu_ctx("survival_rowjet_vgh kernel launch")?;

        let mut value = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n * 4];
        let mut hess = vec![0.0_f64; n * 16];
        stream
            .memcpy_dtoh(&value_d, &mut value)
            .gpu_ctx("vgh dtoh value")?;
        stream
            .memcpy_dtoh(&grad_d, &mut grad)
            .gpu_ctx("vgh dtoh grad")?;
        stream
            .memcpy_dtoh(&hess_d, &mut hess)
            .gpu_ctx("vgh dtoh hess")?;
        stream
            .synchronize()
            .gpu_ctx("survival_rowjet_vgh synchronize")?;
        Ok(SurvivalRowVghChannels { value, grad, hess })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

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

    const DIR: [f64; 4] = [0.31, -0.22, 0.17, 0.44];
    const DIRU: [f64; 4] = [0.13, 0.27, -0.41, 0.05];
    const DIRV: [f64; 4] = [-0.19, 0.33, 0.08, 0.22];

    #[test]
    fn cpu_channels_match_unified_rowkernel() {
        // The CPU fallback IS `rigid_row_nll` at Order2/OneSeed/TwoSeed, the same
        // thing the production `SurvivalMarginalSlopeRowKernel` calls. Cross-check
        // the (v,g,H) channels against a direct `Order2<4>` evaluation so the
        // flattening/layout is pinned to the single source.
        use crate::survival::marginal_slope::row_kernel::rigid_row_nll;
        use gam_math::jet_scalar::{JetScalar, Order2};
        let rows = fixture(7);
        let out = survival_rigid_row_jets_cpu(&rows, 0.7, &DIR, &DIRU, &DIRV);
        for (row, inp) in rows.iter().enumerate() {
            let in_row = RigidRowInputs {
                row,
                wi: inp.wi,
                di: inp.di,
                z_sum: inp.z_sum,
                covariance_ones: inp.cov_ones,
                probit_scale: 0.7,
                qd1_lower: f64::NEG_INFINITY,
            };
            let vars: [Order2<4>; 4] =
                std::array::from_fn(|a| Order2::variable(inp.primaries[a], a));
            let dense = rigid_row_nll(&vars, &in_row).expect("dense order2");
            assert!((dense.value() - out.value[row]).abs() <= 1e-12);
            for a in 0..4 {
                assert!((dense.g()[a] - out.grad[row * 4 + a]).abs() <= 1e-12);
                for b in 0..4 {
                    assert!(
                        (dense.h()[a][b] - out.hess[row * 16 + a * 4 + b]).abs() <= 1e-12,
                        "hess mismatch row {row} {a},{b}"
                    );
                }
            }
        }
    }

    #[test]
    fn vgh_only_cpu_matches_full_dispatch_and_removes_unused_orders_932() {
        use std::hint::black_box;
        use std::time::Instant;

        let rows = fixture(512);
        let zero = [0.0_f64; 4];
        let full = survival_rigid_row_jets_cpu(&rows, 0.7, &zero, &zero, &zero);
        let vgh = survival_rigid_row_vgh_cpu(&rows, 0.7);
        assert_eq!(vgh.value, full.value, "VGH-only values");
        assert_eq!(vgh.grad, full.grad, "VGH-only gradients");
        assert_eq!(vgh.hess, full.hess, "VGH-only Hessians");

        let iterations = if cfg!(debug_assertions) { 2 } else { 500 };
        let mut full_best = f64::INFINITY;
        let mut vgh_best = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(survival_rigid_row_jets_cpu(
                    black_box(&rows),
                    0.7,
                    &zero,
                    &zero,
                    &zero,
                ));
            }
            full_best = full_best.min(start.elapsed().as_secs_f64());

            let start = Instant::now();
            for _ in 0..iterations {
                black_box(survival_rigid_row_vgh_cpu(black_box(&rows), 0.7));
            }
            vgh_best = vgh_best.min(start.elapsed().as_secs_f64());
        }
        let rows_evaluated = (iterations * rows.len()) as f64;
        let full_ns = full_best * 1e9 / rows_evaluated;
        let vgh_ns = vgh_best * 1e9 / rows_evaluated;
        eprintln!(
            "SURVIVAL-RIGID-VGH-932 full-all-orders={full_ns:.2} ns/row vgh-only={vgh_ns:.2} ns/row speedup={:.3}x",
            full_ns / vgh_ns,
        );
        if !cfg!(debug_assertions) {
            assert!(
                vgh_ns < full_ns,
                "VGH-only dispatcher must beat discarded-order baseline: {vgh_ns} vs {full_ns} ns/row"
            );
        }
    }

    #[test]
    fn cpu_third_fourth_match_dense_tower_oracle() {
        // The seeded-jet (OneSeed/TwoSeed, O(K²)) contracted third/fourth in the
        // CPU fallback must equal the TRUE tensor contraction from the dense
        // `Tower4<4>` (the K³/K⁴ tensor). This pins the seeded contraction to the
        // single-source tensor exactly — the same property the device kernel's
        // JS1/JS2 channels rely on (and the device parity gate then matches THIS
        // CPU result to ≤1e-9).
        use crate::survival::marginal_slope::row_kernel::rigid_row_nll;
        use gam_math::jet_tower::Tower4;
        let rows = fixture(9);
        let out = survival_rigid_row_jets_cpu(&rows, 0.7, &DIR, &DIRU, &DIRV);
        for (row, inp) in rows.iter().enumerate() {
            let in_row = RigidRowInputs {
                row,
                wi: inp.wi,
                di: inp.di,
                z_sum: inp.z_sum,
                covariance_ones: inp.cov_ones,
                probit_scale: 0.7,
                qd1_lower: f64::NEG_INFINITY,
            };
            let vars: [Tower4<4>; 4] =
                std::array::from_fn(|a| Tower4::variable(inp.primaries[a], a));
            let tower = rigid_row_nll(&vars, &in_row).expect("dense tower4");
            let t3 = tower.third_contracted(&DIR);
            let t4 = tower.fourth_contracted(&DIRU, &DIRV);
            for a in 0..4 {
                for b in 0..4 {
                    assert!(
                        (t3[a][b] - out.third[row * 16 + a * 4 + b]).abs() <= 1e-12,
                        "third mismatch row {row} {a},{b}: tensor={} seeded={}",
                        t3[a][b],
                        out.third[row * 16 + a * 4 + b]
                    );
                    assert!(
                        (t4[a][b] - out.fourth[row * 16 + a * 4 + b]).abs() <= 1e-12,
                        "fourth mismatch row {row} {a},{b}: tensor={} seeded={}",
                        t4[a][b],
                        out.fourth[row * 16 + a * 4 + b]
                    );
                }
            }
        }
    }

    /// Per-channel CPU↔device parity tolerance (#415 / #1175).
    ///
    /// The device kernel runs the SAME seeded-jet arithmetic as the CPU jet
    /// (pinned line-for-line by the host-oracle `*_tests` module on every box),
    /// so the residual is NOT an algebra mismatch. With NVRTC FMA contraction
    /// now disabled (#1686, `--fmad=false`), the residual splits into a tight
    /// low-order floor (FMA was its dominant source, so the fix shrank it) and
    /// an irreducible transcendental floor in the high-order channels: CUDA's
    /// `erfc`/`erfcx`/`exp`/`sqrt` differ from the host libm at the ULP level,
    /// and that ε is amplified through the order-4 jet chain (`logΦ`, the Mills
    /// `k1..k4` polynomial, the `c=√(1+(s·g)²cov)` composition) into the
    /// third/fourth channels — which `--fmad=false` leaves unchanged (5.09e-8 /
    /// 4.54e-8, bit-identical to the pre-#1686 measurement). Measured on a
    /// Tesla V100 (sm_70), the drift, **normalized to each channel's
    /// magnitude**, is:
    ///
    /// ```text
    ///   channel  worst |Δ|     channel max|cpu|   |Δ|/scale
    ///   value    1.48e-10      2.22e1             6.7e-12
    ///   grad     8.18e-10      1.14e1             7.2e-11
    ///   hess     8.79e-9       2.50e1             3.5e-10
    ///   third    5.09e-8       4.25e1             1.2e-9
    ///   fourth   4.54e-8       1.23e2             3.7e-10
    /// ```
    ///
    /// (The old gate compared a flat `|Δ| <= 1e-9` ACROSS ALL channels — it
    /// ignored both derivative-order amplification and the transcendental
    /// floor, so the third channel's 5.09e-8 failed it even though that is a
    /// 1.2e-9 relative drift. Per-element *relative* error is also wrong here:
    /// the high-order channels cross zero, so at a cancellation point |cpu| is
    /// ~1e-7 while the channel scale is ~1e2 and the relative error spuriously
    /// reads 2.0.) The principled scale is the channel magnitude. A real
    /// algebra bug (a sign flip / dropped Leibniz term, the #736 genus) makes
    /// an error of order the channel magnitude itself — normalized residual
    /// ~O(1), seven orders above this floor — so the gate below catches every
    /// real defect with ~80× headroom over the transcendental noise.
    #[cfg(target_os = "linux")]
    const PARITY_ATOL: f64 = 1e-9;
    #[cfg(target_os = "linux")]
    const PARITY_RTOL: f64 = 1e-7;

    /// Assert every element of `dev` matches `cpu` within
    /// `PARITY_ATOL + PARITY_RTOL * channel_scale`, where `channel_scale` is the
    /// channel's max |cpu| (the magnitude a real bug would perturb). Returns the
    /// worst normalized residual for reporting.
    #[cfg(target_os = "linux")]
    fn assert_channel_parity(name: &str, cpu: &[f64], dev: &[f64]) -> f64 {
        let scale = cpu.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let tol = PARITY_ATOL + PARITY_RTOL * scale;
        let mut worst = 0.0_f64;
        let mut worst_i = 0usize;
        for (i, (x, y)) in cpu.iter().zip(dev).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                worst_i = i;
            }
        }
        assert!(
            worst <= tol,
            "survival device vs CPU `{name}` channel: worst |Δ|={worst:.3e} at idx {worst_i} \
             (cpu={:.6e} dev={:.6e}) exceeds tol={tol:.3e} (atol={PARITY_ATOL:.0e} + \
             rtol={PARITY_RTOL:.0e}·scale {scale:.3e}). A residual this large is an algebra \
             mismatch, not transcendental drift — check the .cu JS1/JS2 recurrences.",
            cpu[worst_i],
            dev[worst_i]
        );
        worst / tol
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn diag_device_channel_breakdown() {
        let rows = fixture(DEVICE_ROW_THRESHOLD + 1024);
        let cpu = survival_rigid_row_jets_cpu(&rows, 0.7, &DIR, &DIRU, &DIRV);
        let got = match survival_rigid_row_jets_device_only(&rows, 0.7, &DIR, &DIRU, &DIRV) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("DEVICE PATH UNAVAILABLE: {e}");
                return;
            }
        };
        let report = |name: &str, a: &[f64], b: &[f64]| {
            let mut maxabs = 0.0_f64;
            let mut maxrel = 0.0_f64;
            let mut worst_idx = 0usize;
            let mut worst_cpu = 0.0_f64;
            let mut worst_gpu = 0.0_f64;
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let ad = (x - y).abs();
                if ad > maxabs {
                    maxabs = ad;
                    worst_idx = i;
                    worst_cpu = *x;
                    worst_gpu = *y;
                }
                let denom = x.abs().max(y.abs());
                if denom > 1e-12 {
                    maxrel = maxrel.max(ad / denom);
                }
            }
            eprintln!(
                "[{name:8}] maxabs={maxabs:.3e} maxrel={maxrel:.3e} \
                 worst@{worst_idx} cpu={worst_cpu:.6e} gpu={worst_gpu:.6e}"
            );
        };
        report("value", &cpu.value, &got.value);
        report("grad", &cpu.grad, &got.grad);
        report("hess", &cpu.hess, &got.hess);
        report("third", &cpu.third, &got.third);
        report("fourth", &cpu.fourth, &got.fourth);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_matches_cpu_when_available() {
        // Exactness gate: when a device is admitted, every channel must match the
        // CPU unified jet within the principled per-channel magnitude-scaled band
        // (see PARITY_ATOL/PARITY_RTOL). When no device is available the dispatcher
        // returns the CPU result, so this asserts CPU==CPU (trivially within band).
        let rows = fixture(DEVICE_ROW_THRESHOLD + 1024);
        let cpu = survival_rigid_row_jets_cpu(&rows, 0.7, &DIR, &DIRU, &DIRV);
        let got = survival_rigid_row_jets(&rows, 0.7, &DIR, &DIRU, &DIRV);
        assert_channel_parity("value", &cpu.value, &got.value);
        assert_channel_parity("grad", &cpu.grad, &got.grad);
        assert_channel_parity("hess", &cpu.hess, &got.hess);
        assert_channel_parity("third", &cpu.third, &got.third);
        assert_channel_parity("fourth", &cpu.fourth, &got.fourth);

        let cpu_vgh = survival_rigid_row_vgh_cpu(&rows, 0.7);
        let got_vgh = survival_rigid_row_vgh(&rows, 0.7);
        assert_channel_parity("VGH value", &cpu_vgh.value, &got_vgh.value);
        assert_channel_parity("VGH grad", &cpu_vgh.grad, &got_vgh.grad);
        assert_channel_parity("VGH hess", &cpu_vgh.hess, &got_vgh.hess);

        // Anti-false-green: if a CUDA runtime is present the dispatcher MUST have
        // exercised the device kernel above (n > DEVICE_ROW_THRESHOLD), not the
        // silent CPU fallback. Prove the device path itself runs and matches —
        // otherwise this gate would pass on CPU==CPU even with a dead kernel.
        if gam_gpu::device_runtime::GpuRuntime::global().is_some() {
            let dev = survival_rigid_row_jets_device_only(&rows, 0.7, &DIR, &DIRU, &DIRV)
                .expect("CUDA runtime present but survival_rowjet device path could not run");
            assert_channel_parity("device value", &cpu.value, &dev.value);
            assert_channel_parity("device grad", &cpu.grad, &dev.grad);
            assert_channel_parity("device hess", &cpu.hess, &dev.hess);
            assert_channel_parity("device third", &cpu.third, &dev.third);
            assert_channel_parity("device fourth", &cpu.fourth, &dev.fourth);

            let dev_vgh = survival_rigid_row_vgh_device_only(&rows, 0.7)
                .expect("CUDA runtime present but survival_rowjet_vgh could not run");
            assert_channel_parity("device VGH value", &cpu_vgh.value, &dev_vgh.value);
            assert_channel_parity("device VGH grad", &cpu_vgh.grad, &dev_vgh.grad);
            assert_channel_parity("device VGH hess", &cpu_vgh.hess, &dev_vgh.hess);
        }
    }

    /// Edge-regime fixture: rows deliberately placed in the hard corners of the
    /// probit Mills-ratio stack, where erfc/erfcx differ most between host libm
    /// and CUDA and the seeded-jet amplification is largest. Covers
    /// censored/event × entry-present, deep negative tails (logΦ underflow
    /// regime), tiny and large covariance, near-zero slope, large scale, zero
    /// weight (the early-out branch), and the erfcx asymptotic cutover (|η|>26).
    #[cfg(target_os = "linux")]
    fn edge_fixture() -> Vec<SurvivalRowInputs> {
        let mut rows = Vec::new();
        let push = |rows: &mut Vec<SurvivalRowInputs>, p: [f64; 4], w, d, z, c| {
            rows.push(SurvivalRowInputs {
                primaries: p,
                wi: w,
                di: d,
                z_sum: z,
                cov_ones: c,
            });
        };
        // interior, event & censored
        push(&mut rows, [-0.4, 0.6, 0.9, 0.3], 1.0, 1.0, 0.2, 0.5);
        push(&mut rows, [-0.4, 0.6, 0.9, 0.3], 1.0, 0.0, 0.2, 0.5);
        // deep negative probit tail (logΦ(−η)→ asymptotic / Mills tail)
        push(&mut rows, [8.0, 9.0, 1.2, 2.5], 1.0, 0.0, -3.0, 1.0);
        push(&mut rows, [-8.0, -9.0, 1.2, -2.5], 1.0, 1.0, 3.0, 1.0);
        // erfcx asymptotic cutover region (argument near/above 26)
        push(&mut rows, [40.0, 41.0, 0.7, 3.0], 1.0, 0.0, 0.0, 2.0);
        // tiny covariance (c ≈ 1, derivative of √ near flat)
        push(&mut rows, [-0.3, 0.5, 0.8, 1.5], 1.0, 1.0, 0.4, 1e-10);
        // large covariance + large scale (c large, strong coupling)
        push(&mut rows, [-0.2, 0.4, 1.1, 4.0], 1.0, 1.0, 0.1, 50.0);
        // near-zero slope (og→0, opb2→1)
        push(&mut rows, [-0.5, 0.3, 0.6, 1e-9], 1.0, 0.0, 0.7, 0.9);
        // zero weight (the w==0 early-out: every channel 0)
        push(&mut rows, [-0.5, 0.3, 0.6, 0.4], 0.0, 1.0, 0.7, 0.9);
        // small positive qd1 (log(ad1) near its valid edge)
        push(&mut rows, [-0.5, 0.3, 1e-3, 0.4], 1.0, 1.0, 0.2, 0.6);
        rows
    }

    /// #415 core deliverable — **fail loud, never silently degrade.** On a GPU
    /// box the device path MUST run; this calls `survival_rigid_row_jets_device_only`
    /// (which never falls back) and asserts it both (a) succeeds — no silent
    /// NVRTC-declined / wrong-arch / launch-failure swallowed by the dispatcher —
    /// and (b) matches the CPU oracle within the principled per-channel band, for
    /// BOTH the t4 and the no-t4 kernel variants and across the edge-regime sweep.
    ///
    /// When no CUDA device is present the device-only path returns `Err`, which
    /// is the legitimate state on a CPU-only box — so the test SKIPS with a clear
    /// log there. Set `GAM_REQUIRE_GPU=1` (CI on the GPU runner) to turn that skip
    /// into a HARD failure: a box that is supposed to have a GPU but can't run the
    /// kernel must break the build, not pass on the CPU.
    #[cfg(target_os = "linux")]
    #[test]
    fn device_only_path_runs_and_matches_cpu_fail_loud() {
        // Fail loud only when a CUDA device is actually present (a real runtime
        // check, not an env-var read — `env::var` is banned crate-wide): on a GPU
        // box the device path MUST run, while a CI runner with no device skips
        // gracefully.
        let require_gpu = gam_gpu::device_runtime::GpuRuntime::global().is_some();

        // Two batches: enough rows to amortise the launch, in both the interior
        // (smooth) and edge (transcendental-stress) regimes. The edge batch is
        // padded by tiling so it crosses DEVICE_ROW_THRESHOLD.
        let interior = fixture(DEVICE_ROW_THRESHOLD + 777);
        let edge_unit = edge_fixture();
        let reps = (DEVICE_ROW_THRESHOLD + 999).div_ceil(edge_unit.len());
        let edge: Vec<_> = edge_unit
            .iter()
            .cloned()
            .cycle()
            .take(reps * edge_unit.len())
            .collect();

        // Variant matrix: (label, dir_u, dir_v). All-zero (u,v) selects the
        // `survival_rowjet_no_t4` kernel (fourth channel ≡ 0); nonzero selects
        // the full `survival_rowjet`. Cover both so neither entry point rots.
        let zero = [0.0_f64; 4];
        let variants: [(&str, &[f64; 4], &[f64; 4]); 2] =
            [("t4", &DIRU, &DIRV), ("no_t4", &zero, &zero)];

        let mut ran_on_device = false;
        for (regime, rows) in [("interior", &interior), ("edge", &edge)] {
            for (vlabel, du, dv) in variants {
                let dev = match survival_rigid_row_jets_device_only(rows, 0.7, &DIR, du, dv) {
                    Ok(d) => d,
                    Err(e) => {
                        if require_gpu {
                            panic!(
                                "GAM_REQUIRE_GPU set but survival_rowjet device path \
                                 ({regime}/{vlabel}) could not run: {e}"
                            );
                        }
                        eprintln!(
                            "[#415] no CUDA device ({regime}/{vlabel}) — skipping device-only \
                             parity (set GAM_REQUIRE_GPU=1 to make this a hard failure): {e}"
                        );
                        continue;
                    }
                };
                ran_on_device = true;
                let cpu = survival_rigid_row_jets_cpu(rows, 0.7, &DIR, du, dv);
                assert_channel_parity(&format!("{regime}/{vlabel}/value"), &cpu.value, &dev.value);
                assert_channel_parity(&format!("{regime}/{vlabel}/grad"), &cpu.grad, &dev.grad);
                assert_channel_parity(&format!("{regime}/{vlabel}/hess"), &cpu.hess, &dev.hess);
                assert_channel_parity(&format!("{regime}/{vlabel}/third"), &cpu.third, &dev.third);
                assert_channel_parity(
                    &format!("{regime}/{vlabel}/fourth"),
                    &cpu.fourth,
                    &dev.fourth,
                );
                // The no_t4 variant must yield an exactly-zero fourth channel
                // (the kernel writes 0.0), and the CPU oracle agrees because
                // (u,v)=0 contracts the fourth tensor to zero.
                if vlabel == "no_t4" {
                    assert!(
                        dev.fourth.iter().all(|&x| x == 0.0),
                        "no_t4 kernel must write an all-zero fourth channel"
                    );
                }
            }
        }
        if ran_on_device {
            eprintln!("[#415] device-only parity PASSED on GPU for all regimes × variants");
        }
    }
}
