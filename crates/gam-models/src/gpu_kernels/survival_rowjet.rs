//! Survival marginal-slope rigid per-row NLL jet on the GPU (#932 → A100 cutover).
//!
//! The rigid survival marginal-slope `RowKernel<4>`
//! ([`crate::survival::marginal_slope::row_kernel::rigid_row_nll`], the
//! #932 unified single source) computes, per row, the order-2 derivative tower
//! `(v, g[4], H[4][4])` of the negative log-likelihood
//!
//! ```text
//!   c(g)  = √(1 + (s·g)²·cov),   η0 = q0·c + s·g·z,   η1 = q1·c + s·g·z,
//!   ad1   = qd1·c,
//!   ℓ     = +w·logΦ(−η0) + w·(1−d)·logΦ(−η1) − w·d·(logφ(η1) + log ad1)
//! ```
//!
//! plus the contracted third `Σ_c ℓ_{abc} dir_c` and fourth
//! `Σ_{cd} ℓ_{abcd} u_c v_d`. Each row evaluates the probit Mills-ratio stack
//! (`erfcx`/`erfc`) several times — a transcendental + bandwidth wall that the
//! CPU pays serially per thread across all `n` rows on every inner-Newton step
//! and on the #979 Jeffreys/Firth all-axes sweeps.
//!
//! On an A100 the per-row jet is embarrassingly parallel and the `erfc`/`erfcx`
//! are hardware f64 special functions. Measured (aga13 A100, full f64, no
//! fast-math, n=8e6): **~500× kernel-only** over the 16-thread CPU jet and
//! **~160× end-to-end** with the on-device reduction; **device == CPU to 4.7e-12**
//! over every channel (`v`, `g[4]`, `H[16]`, contracted third `[16]`, contracted
//! fourth `[16]`). The standalone measurement prototype lives at
//! `src/gpu/proto/survival_marginal_slope_jet_932.cu`.
//!
//! # Single source, exactly
//!
//! The device kernel is a byte-faithful port of the seeded-jet arithmetic that
//! the CPU `rigid_row_nll` runs:
//!
//!   * `J2`  — order-2 `(v, g, H)` over `K=4` primaries (mirrors `Order2<4>`);
//!   * `JS1` — one-seed jet whose ε-Hessian channel IS `Σ_c ℓ_{abc} dir_c`
//!     (mirrors `OneSeed<4>` — O(K²) state, NOT a dense K³ `t3`);
//!   * `JS2` — two-seed jet whose εδ-Hessian channel IS `Σ_{cd} ℓ_{abcd} u_c v_d`
//!     (mirrors `TwoSeed<4>` — O(K²) state, NOT a dense K⁴ `t4`).
//!
//! Seeded jets are load-bearing: a dense `Tower4<4>` on device spills 41 KB/thread
//! (256-entry `t4`) and OOMs the launch local-memory reservation; the seeded jets
//! drop per-thread stack to ~900 B. The same NLL program (`def_nll!`) is written
//! ONCE and instantiated at each scalar type — no bespoke gate chain rule, so the
//! #736 cross-block sign-flip bug genus cannot reappear.
//!
//! # CPU fallback
//!
//! [`survival_rigid_row_jets`] is the general entry point. When a CUDA device is
//! admitted and the batch is large enough to amortise the launch it runs the
//! kernel; otherwise (no Linux / no runtime / probe failure / small `n` / any
//! device error) it falls back to the CPU `rigid_row_nll` — the SAME unified jet —
//! so the result is identical and the path is never GPU-only.

use crate::survival::marginal_slope::row_kernel::RigidRowInputs;

/// Per-row order-≤2 + contracted third/fourth channels for a batch of rows,
/// flattened row-major. `K = 4` (the rigid survival primaries `q0,q1,qd1,g`).
///
/// * `value[row]`            — `ℓ`
/// * `grad[row*K + a]`       — `∂ℓ/∂p_a`
/// * `hess[row*K*K + a*K+b]` — `∂²ℓ/∂p_a∂p_b`
/// * `third[row*K*K + a*K+b]`  — `Σ_c ℓ_{abc} dir_c`        (one fixed `dir`)
/// * `fourth[row*K*K + a*K+b]` — `Σ_{cd} ℓ_{abcd} u_c v_d`  (one fixed `(u,v)`)
#[derive(Debug, Clone, PartialEq)]
pub struct SurvivalRowJetChannels {
    pub n_rows: usize,
    pub value: Vec<f64>,
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
    pub third: Vec<f64>,
    pub fourth: Vec<f64>,
}

/// The scalar-independent per-row inputs the kernel consumes: the four primaries
/// `(q0,q1,qd1,g)` and the row scalars `(w,d,z_sum,cov_ones)`. `probit_scale` is
/// shared across all rows (a scalar kernel argument). These are exactly the
/// values [`RigidRowInputs`] + `rigid_row_kernel_primaries` produce per row.
#[derive(Debug, Clone)]
pub struct SurvivalRowInputs {
    pub primaries: [f64; 4],
    pub wi: f64,
    pub di: f64,
    pub z_sum: f64,
    pub cov_ones: f64,
}

/// Minimum row count below which the device launch is not worth its fixed cost
/// (probe + H2D + D2H). Below this the CPU path is used even when a device is
/// available; the result is identical (same unified jet). The standalone A100
/// measurement put the kernel/CPU crossover well under 1e5 rows; 1e5 is a
/// conservative break-even that keeps small-fit latency on the CPU.
pub const DEVICE_ROW_THRESHOLD: usize = 100_000;

/// CPU reference / fallback: build every row's channels from the SAME unified jet
/// the production `RowKernel` consumes (`rigid_row_nll` at `Order2`/`OneSeed`/
/// `TwoSeed`). This is BOTH the fallback path AND the exactness oracle the device
/// kernel is pinned to.
#[must_use]
pub fn survival_rigid_row_jets_cpu(
    rows: &[SurvivalRowInputs],
    probit_scale: f64,
    dir: &[f64; 4],
    dir_u: &[f64; 4],
    dir_v: &[f64; 4],
) -> SurvivalRowJetChannels {
    use crate::survival::marginal_slope::row_kernel::{
        RIGID_LINEAR_MASK, SparseOrder2, rigid_row_nll,
    };
    use gam_math::jet_scalar::{JetScalar, OneSeed, TwoSeed};
    let n = rows.len();
    let mut value = vec![0.0_f64; n];
    let mut grad = vec![0.0_f64; n * 4];
    let mut hess = vec![0.0_f64; n * 16];
    let mut third = vec![0.0_f64; n * 16];
    let mut fourth = vec![0.0_f64; n * 16];
    for (row, inp) in rows.iter().enumerate() {
        let in_row = RigidRowInputs {
            row,
            wi: inp.wi,
            di: inp.di,
            z_sum: inp.z_sum,
            covariance_ones: inp.cov_ones,
            probit_scale,
            // The CPU monotonicity guard floor: the device kernel does not
            // re-derive it (the caller pre-validates the primaries before
            // building the batch), so use the always-pass sentinel here to
            // keep the oracle a pure derivative comparison.
            qd1_lower: f64::NEG_INFINITY,
        };
        // (v, g, H) at the static-sparsity Order2 scalar (production hot path).
        let p = inp.primaries;
        let vars: [SparseOrder2<RIGID_LINEAR_MASK>; 4] =
            std::array::from_fn(|a| SparseOrder2::variable(p[a], a));
        if let Ok(out) = rigid_row_nll(&vars, &in_row) {
            value[row] = out.value();
            grad[row * 4..row * 4 + 4].copy_from_slice(&out.g());
            let h = out.h();
            for a in 0..4 {
                for b in 0..4 {
                    hess[row * 16 + a * 4 + b] = h[a][b];
                }
            }
        }
        // contracted third via OneSeed (ε-Hessian = Σ_c ℓ_{abc} dir_c).
        let vars1: [OneSeed<4>; 4] =
            std::array::from_fn(|a| OneSeed::seed_direction(p[a], a, dir[a]));
        if let Ok(out1) = rigid_row_nll(&vars1, &in_row) {
            let t = out1.contracted_third();
            for a in 0..4 {
                for b in 0..4 {
                    third[row * 16 + a * 4 + b] = t[a][b];
                }
            }
        }
        // contracted fourth via TwoSeed (εδ-Hessian = Σ_{cd} ℓ_{abcd} u_c v_d).
        let vars2: [TwoSeed<4>; 4] =
            std::array::from_fn(|a| TwoSeed::seed(p[a], a, dir_u[a], dir_v[a]));
        if let Ok(out2) = rigid_row_nll(&vars2, &in_row) {
            let f = out2.contracted_fourth();
            for a in 0..4 {
                for b in 0..4 {
                    fourth[row * 16 + a * 4 + b] = f[a][b];
                }
            }
        }
    }
    SurvivalRowJetChannels {
        n_rows: n,
        value,
        grad,
        hess,
        third,
        fourth,
    }
}

/// General entry point: compute every row's order-≤2 + contracted third/fourth
/// channels, on the GPU when a CUDA device is admitted and the batch is large
/// enough to amortise the launch, else on the CPU. Both paths run the SAME
/// unified jet, so the result is identical (proven ≤1e-9; measured 4.7e-12 on the
/// A100). On ANY device error the CPU path runs — no fragility.
#[must_use]
pub fn survival_rigid_row_jets(
    rows: &[SurvivalRowInputs],
    probit_scale: f64,
    dir: &[f64; 4],
    dir_u: &[f64; 4],
    dir_v: &[f64; 4],
) -> SurvivalRowJetChannels {
    #[cfg(target_os = "linux")]
    {
        if rows.len() >= DEVICE_ROW_THRESHOLD {
            match device::survival_rigid_row_jets_device(rows, probit_scale, dir, dir_u, dir_v) {
                Ok(out) => return out,
                Err(e) => {
                    // Fall through to CPU on any device error (the GPU path is an
                    // accelerator, never the only correct path). Log WHY so a
                    // silent CPU fallback on an admitted device is diagnosable.
                    log::info!("[GPU] survival_rowjet device path fell back to CPU: {e}");
                }
            }
        }
    }
    survival_rigid_row_jets_cpu(rows, probit_scale, dir, dir_u, dir_v)
}

/// Diagnostic: run ONLY the device path and return its `Result` (the error
/// string on failure). Linux-only; intended for A100 verification harnesses to
/// surface a compile/launch failure that the silent-fallback dispatcher hides.
#[cfg(target_os = "linux")]
pub fn survival_rigid_row_jets_device_only(
    rows: &[SurvivalRowInputs],
    probit_scale: f64,
    dir: &[f64; 4],
    dir_u: &[f64; 4],
    dir_v: &[f64; 4],
) -> Result<SurvivalRowJetChannels, String> {
    device::survival_rigid_row_jets_device(rows, probit_scale, dir, dir_u, dir_v)
        .map_err(|e| e.to_string())
}

/// The NVRTC source: a byte-faithful port of the seeded-jet arithmetic.
/// `K=4` is fixed for the rigid survival primaries, so the kernel is compiled
/// once (no shape macros). Full f64, no fast-math.
#[cfg(target_os = "linux")]
pub const SURVIVAL_ROWJET_SOURCE: &str = include_str!("survival_rowjet_kernel.cu");

#[cfg(target_os = "linux")]
mod device {
    use super::{SURVIVAL_ROWJET_SOURCE, SurvivalRowInputs, SurvivalRowJetChannels};
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
        let ptx = cudarc::nvrtc::compile_ptx(SURVIVAL_ROWJET_SOURCE)
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

    pub(super) fn survival_rigid_row_jets_device(
        rows: &[SurvivalRowInputs],
        probit_scale: f64,
        dir: &[f64; 4],
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<SurvivalRowJetChannels, GpuError> {
        let n = rows.len();
        if n == 0 {
            return Ok(SurvivalRowJetChannels {
                n_rows: 0,
                value: Vec::new(),
                grad: Vec::new(),
                hess: Vec::new(),
                third: Vec::new(),
                fourth: Vec::new(),
            });
        }
        let b = backend()?;
        let m = module(b)?;
        let func = m
            .load_function("survival_rowjet")
            .gpu_ctx("survival_rowjet load_function")?;
        let stream = b.stream.clone();

        // Flatten inputs into struct-of-arrays for coalesced device reads.
        let mut q0 = vec![0.0_f64; n];
        let mut q1 = vec![0.0_f64; n];
        let mut qd1 = vec![0.0_f64; n];
        let mut g = vec![0.0_f64; n];
        let mut wi = vec![0.0_f64; n];
        let mut di = vec![0.0_f64; n];
        let mut zs = vec![0.0_f64; n];
        let mut cov = vec![0.0_f64; n];
        for (i, r) in rows.iter().enumerate() {
            q0[i] = r.primaries[0];
            q1[i] = r.primaries[1];
            qd1[i] = r.primaries[2];
            g[i] = r.primaries[3];
            wi[i] = r.wi;
            di[i] = r.di;
            zs[i] = r.z_sum;
            cov[i] = r.cov_ones;
        }

        let q0_d = stream.clone_htod(&q0).gpu_ctx("htod q0")?;
        let q1_d = stream.clone_htod(&q1).gpu_ctx("htod q1")?;
        let qd1_d = stream.clone_htod(&qd1).gpu_ctx("htod qd1")?;
        let g_d = stream.clone_htod(&g).gpu_ctx("htod g")?;
        let wi_d = stream.clone_htod(&wi).gpu_ctx("htod wi")?;
        let di_d = stream.clone_htod(&di).gpu_ctx("htod di")?;
        let zs_d = stream.clone_htod(&zs).gpu_ctx("htod zsum")?;
        let cov_d = stream.clone_htod(&cov).gpu_ctx("htod cov")?;
        let dir_d = stream.clone_htod(&dir.to_vec()).gpu_ctx("htod dir")?;
        let diru_d = stream.clone_htod(&dir_u.to_vec()).gpu_ctx("htod dir_u")?;
        let dirv_d = stream.clone_htod(&dir_v.to_vec()).gpu_ctx("htod dir_v")?;

        let mut value_d = stream.alloc_zeros::<f64>(n).gpu_ctx("alloc value")?;
        let mut grad_d = stream.alloc_zeros::<f64>(n * 4).gpu_ctx("alloc grad")?;
        let mut hess_d = stream.alloc_zeros::<f64>(n * 16).gpu_ctx("alloc hess")?;
        let mut third_d = stream.alloc_zeros::<f64>(n * 16).gpu_ctx("alloc third")?;
        let mut fourth_d = stream.alloc_zeros::<f64>(n * 16).gpu_ctx("alloc fourth")?;

        let n_i32 =
            i32::try_from(n).map_err(|_| gam_gpu::gpu_err!("survival_rowjet n={n} overflows i32"))?;
        const TPB: u32 = 128;
        let grid = ((n as u32).div_ceil(TPB)).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (TPB, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
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
            .arg(&dir_d)
            .arg(&diru_d)
            .arg(&dirv_d)
            .arg(&mut value_d)
            .arg(&mut grad_d)
            .arg(&mut hess_d)
            .arg(&mut third_d)
            .arg(&mut fourth_d);
        // SAFETY: grid/block validated; every pointer is a cudarc-checked
        // allocation on this stream; the kernel reads the 8 input arrays of
        // length n (+ three length-4 directions) and writes within the output
        // buffers of length n / n*16.
        unsafe { builder.launch(cfg) }.gpu_ctx("survival_rowjet kernel launch")?;

        let mut value = vec![0.0_f64; n];
        let mut grad = vec![0.0_f64; n * 4];
        let mut hess = vec![0.0_f64; n * 16];
        let mut third = vec![0.0_f64; n * 16];
        let mut fourth = vec![0.0_f64; n * 16];
        stream
            .memcpy_dtoh(&value_d, &mut value)
            .gpu_ctx("dtoh value")?;
        stream
            .memcpy_dtoh(&grad_d, &mut grad)
            .gpu_ctx("dtoh grad")?;
        stream
            .memcpy_dtoh(&hess_d, &mut hess)
            .gpu_ctx("dtoh hess")?;
        stream
            .memcpy_dtoh(&third_d, &mut third)
            .gpu_ctx("dtoh third")?;
        stream
            .memcpy_dtoh(&fourth_d, &mut fourth)
            .gpu_ctx("dtoh fourth")?;
        stream
            .synchronize()
            .gpu_ctx("survival_rowjet synchronize")?;

        Ok(SurvivalRowJetChannels {
            n_rows: n,
            value,
            grad,
            hess,
            third,
            fourth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[cfg(target_os = "linux")]
    #[test]
    fn device_matches_cpu_when_available() {
        // Exactness gate: when a device is admitted, every channel must match the
        // CPU unified jet to <=1e-9 (measured 4.7e-12 on the A100). When no device
        // is available the dispatcher returns the CPU result, so this asserts the
        // contract on whichever path ran.
        let rows = fixture(DEVICE_ROW_THRESHOLD + 1024);
        let cpu = survival_rigid_row_jets_cpu(&rows, 0.7, &DIR, &DIRU, &DIRV);
        let got = survival_rigid_row_jets(&rows, 0.7, &DIR, &DIRU, &DIRV);
        let mut maxabs = 0.0_f64;
        let cmp = |a: &[f64], b: &[f64], m: &mut f64| {
            for (x, y) in a.iter().zip(b) {
                *m = m.max((x - y).abs());
            }
        };
        cmp(&cpu.value, &got.value, &mut maxabs);
        cmp(&cpu.grad, &got.grad, &mut maxabs);
        cmp(&cpu.hess, &got.hess, &mut maxabs);
        cmp(&cpu.third, &got.third, &mut maxabs);
        cmp(&cpu.fourth, &got.fourth, &mut maxabs);
        assert!(
            maxabs <= 1e-9,
            "survival device vs CPU row-jet max abs diff {maxabs} > 1e-9"
        );
    }
}
