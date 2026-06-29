//! DISPATCH SEAM SKETCH (NOT yet wired into `mod.rs`) for the SAE fused
//! reconstruction + on-device `sae_dot` contraction kernel
//! (`sae_fused_contraction_kernel.cu`).
//!
//! This mirrors two existing precedents:
//!   * `bms/gpu/flex.rs::row_primary_hessian_decision` — the `decide()` +
//!     `row_kernel_min_n` break-even policy gate.
//!   * `gpu_kernels/sae_rowjet.rs::device` — the NVRTC module-cache + resident
//!     upload + one-block-per-row launch.
//!
//! # Why a NEW kernel (vs the existing `sae_rowjet.rs`)
//!
//! `sae_rowjet.rs` returns the FULL `first[n*K*p]` / `second[n*K*K*p]` p-wide
//! tensors and contracts them on the CPU (`gauss_newton_row_hessian_slabs`).
//! That is the proto's documented transfer-bound trap (~17×, 424 B/row DtoH).
//! This kernel performs the `sae_dot` contraction ON-DEVICE (the p-axis is
//! reduced before the q² blow-up) and returns only the tiny `q×q` slabs —
//! `n*Q*Q` doubles instead of `n*Q*Q*p`. With p∈[16,64] that is a 16–64×
//! smaller DtoH, recovering the kernel-bound speedup (sibling survival proto:
//! 504× kernel-only / 162× end-to-end with on-device reduction, 4.7e-12 error).
//!
//! # Where it hooks
//!
//! `construction.rs`'s streaming arrow-logdet loop currently, per row, calls
//! `SaeManifoldTerm::row_jets_for_logdet` (building the full p-wide `SaeRowJets`)
//! and then contracts via `sae_dot` at construction.rs:8210 (Gram /
//! `H_ab = ⟨first_a, first_b⟩`) and :9239 (residual /
//! `r_ab = ⟨error_metric, second_ab⟩`). The seam below replaces that per-row
//! build+contract, for a SPAN of softmax rows above break-even, with one batched
//! launch that returns the contracted `q×q` Gram and residual slabs directly.
//!
//! Only the `AssignmentMode::Softmax` path is offloaded (the closed-form hand
//! path that `row_jets_for_logdet` already special-cases). `IBPMap` / `JumpReLU`
//! (per-atom-logistic) keep the CPU jet path. Below break-even, or on any device
//! error, the CPU **fused-SIMD batch4** path is used (the same arithmetic the
//! kernel ports — `fused_resid4` / `fused_gram4` in `scratchpad/sae_fused_bench.rs`,
//! productionised as `row_jets_for_logdet_batch4` + a CPU contraction), so the
//! result is identical and the path is never GPU-only.

#![allow(dead_code)] // seam sketch; not yet referenced.

use gam_gpu::{GpuDecision, GpuEligibility, GpuKernel, decide};

/// Per-row break-even below which the fused launch is not worth its fixed cost
/// (resident HtoD amortised across REML/Newton iterations; only `error_metric`
/// refreshes per inner step). Tuned to the same genus as
/// `sae_rowjet::DEVICE_ROW_THRESHOLD` (4_096) and the survival proto's measured
/// ~1e5 break-even; the runtime policy `row_kernel_min_n` (default 50_000) is
/// the authoritative gate, this is the floor.
pub const FUSED_CONTRACTION_MIN_ROWS: usize = 4_096;

/// Policy gate, mirroring `flex.rs::row_primary_hessian_decision`. The kernel is
/// eligible only when the SAE GPU backend is compiled in (Linux + cudarc),
/// `n_rows` clears the runtime `row_kernel_min_n` break-even, and there is at
/// least one primary (`q > 0`).
///
/// NOTE: reuses `GpuKernel::MarginalSlopeRows` as the closest existing kernel
/// tag (per-row order-2 jet through the gate). A dedicated `SaeFusedContraction`
/// variant in `gam_gpu::GpuKernel` would be cleaner for calibration/logging.
#[must_use]
pub fn fused_contraction_decision(n_rows: usize, q: usize) -> GpuDecision {
    let large_enough = gam_gpu::device_runtime::GpuRuntime::global()
        .map(|rt| {
            n_rows >= rt.policy().row_kernel_min_n.max(FUSED_CONTRACTION_MIN_ROWS) && q > 0
        })
        .unwrap_or(false);
    decide(
        GpuKernel::MarginalSlopeRows,
        GpuEligibility::from_flags(cfg!(target_os = "linux"), large_enough),
    )
}

/// The contracted per-row slabs the on-device reduction returns: the SAME
/// `row_hessian_slabs` layout (`[n_rows * q * q]`, row-major `q×q` per row) the
/// consumers at construction.rs:8210 / :9239 build via `sae_dot`.
#[derive(Debug, Clone)]
pub struct FusedContractionSlabs {
    pub n_rows: usize,
    pub q: usize,
    /// Gram `H_ab = Σ_c first_a[c]·first_b[c]`           (consumer 8210).
    pub gram: Vec<f64>,
    /// Residual `R_ab = Σ_c error_metric[c]·second_ab[c]` (consumer 9239).
    pub resid: Vec<f64>,
}

/// Shape + resident-input description for a span of softmax rows. `phi` /
/// `dphi` / `d2phi` / `decoder` stay resident across inner iterations; only
/// `error` (and `logits`, when the gate moves) is refreshed.
#[derive(Debug, Clone)]
pub struct FusedRowSpan {
    pub a: usize,   // atoms (K)
    pub l: usize,   // latent dim per atom
    pub nb: usize,  // basis fns per atom
    pub p: usize,   // out_dim
    pub inv_tau: f64,
    pub logits: Vec<f64>,   // [n*a]
    pub phi: Vec<f64>,      // [n*a*nb]
    pub dphi: Vec<f64>,     // [n*a*nb*l]
    pub d2phi: Vec<f64>,    // [n*a*nb*l*l]
    pub decoder: Vec<f64>,  // [n*a*nb*p]
    pub error: Vec<f64>,    // [n*p]
    pub n_rows: usize,
}

/// General entry: contract a span of softmax rows on the GPU when admitted, else
/// on the CPU fused-SIMD batch4 path. (CPU body delegates to the productionised
/// `row_jets_for_logdet_batch4` + the pure-reduction contraction; sketch only.)
#[must_use]
pub fn fused_contraction(span: &FusedRowSpan) -> FusedContractionSlabs {
    let q = span.a * (1 + span.l);
    #[cfg(target_os = "linux")]
    {
        let decision = fused_contraction_decision(span.n_rows, q);
        if decision.use_gpu {
            if let Ok(out) = device::fused_contraction_device(span) {
                return out;
            }
            // Fall through to CPU on any device error — never GPU-only.
        }
    }
    cpu::fused_contraction_cpu_batch4(span, q)
}

mod cpu {
    use super::{FusedContractionSlabs, FusedRowSpan};
    /// CPU fallback: the SAME fused arithmetic the kernel ports, in 4-row NEON
    /// batches. In production this delegates to the existing
    /// `SaeManifoldTerm::row_jets_for_logdet_batch4` (which already runs
    /// `reconstruction_all_columns_batch4`) followed by the `sae_dot`
    /// contraction; here it is a stub to keep the seam self-contained.
    pub(super) fn fused_contraction_cpu_batch4(
        span: &FusedRowSpan,
        q: usize,
    ) -> FusedContractionSlabs {
        // unimplemented in the sketch; wire to row_jets_for_logdet_batch4.
        FusedContractionSlabs {
            n_rows: span.n_rows,
            q,
            gram: vec![0.0; span.n_rows * q * q],
            resid: vec![0.0; span.n_rows * q * q],
        }
    }
}

#[cfg(target_os = "linux")]
mod device {
    use super::{FusedContractionSlabs, FusedRowSpan};
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

    /// The single source of truth: the `.cu` next to this module, with the four
    /// shape macros prepended (mirrors `sae_rowjet::softmax_kernel_source`).
    fn kernel_source(a: usize, l: usize, nb: usize, p: usize) -> String {
        const CU: &str = include_str!("sae_fused_contraction_kernel.cu");
        format!("#define AA {a}\n#define LL {l}\n#define NB_ {nb}\n#define PP {p}\n{CU}")
    }

    struct Backend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        modules: Mutex<HashMap<(usize, usize, usize, usize), Arc<CudaModule>>>,
    }
    fn backend() -> Result<&'static Backend, GpuError> {
        static B: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        B.get_or_init(|| {
            let parts = gam_gpu::backend_probe::probe_cuda_backend("sae_fused_contraction")?;
            Ok(Backend {
                ctx: parts.ctx,
                stream: parts.stream,
                modules: Mutex::new(HashMap::new()),
            })
        })
        .as_ref()
        .map_err(GpuError::clone)
    }
    fn module_for(
        b: &Backend,
        a: usize,
        l: usize,
        nb: usize,
        p: usize,
    ) -> Result<Arc<CudaModule>, GpuError> {
        let key = (a, l, nb, p);
        if let Ok(g) = b.modules.lock() {
            if let Some(m) = g.get(&key) {
                return Ok(m.clone());
            }
        }
        let ptx = cudarc::nvrtc::compile_ptx(kernel_source(a, l, nb, p))
            .gpu_ctx_with(|e| format!("sae_fused_contraction NVRTC (A={a},L={l},NB={nb},P={p}): {e}"))?;
        let m = b.ctx.load_module(ptx).gpu_ctx("sae_fused_contraction module load")?;
        if let Ok(mut g) = b.modules.lock() {
            g.entry(key).or_insert_with(|| m.clone());
        }
        Ok(m)
    }

    /// Upload resident inputs (phi/dphi/d2phi/decoder/logits — held across
    /// REML/Newton iterations) + the per-step `error`, launch one block per row
    /// (128 threads), download ONLY the `n*q*q` Gram + residual slabs.
    pub(super) fn fused_contraction_device(
        span: &FusedRowSpan,
    ) -> Result<FusedContractionSlabs, GpuError> {
        let (a, l, nb, p, n) = (span.a, span.l, span.nb, span.p, span.n_rows);
        let q = a * (1 + l);
        let b = backend()?;
        let module = module_for(b, a, l, nb, p)?;
        let s = b.stream.clone();

        // Resident inputs (in production these persist in the arena across the
        // REML loop; here uploaded each call for clarity).
        let logits = s.clone_htod(&span.logits).gpu_ctx("htod logits")?;
        let phi = s.clone_htod(&span.phi).gpu_ctx("htod phi")?;
        let dphi = s.clone_htod(&span.dphi).gpu_ctx("htod dphi")?;
        let d2phi = s.clone_htod(&span.d2phi).gpu_ctx("htod d2phi")?;
        let decoder = s.clone_htod(&span.decoder).gpu_ctx("htod decoder")?;
        let error = s.clone_htod(&span.error).gpu_ctx("htod error")?;
        let mut gram_dev = s.alloc_zeros::<f64>(n * q * q).gpu_ctx("alloc gram")?;
        let mut resid_dev = s.alloc_zeros::<f64>(n * q * q).gpu_ctx("alloc resid")?;
        // Gram needs p-wide per-block scratch (see kernel SHARED-MEMORY NOTE).
        let mut decoded_scr = s.alloc_zeros::<f64>(n * a * p).gpu_ctx("alloc decoded")?;
        let mut d1_scr = s.alloc_zeros::<f64>(n * a * l.max(1) * p).gpu_ctx("alloc d1")?;

        let n_i32 = i32::try_from(n).map_err(|_| gam_gpu::gpu_err!("n overflow"))?;
        let cfg = LaunchConfig {
            grid_dim: (n_i32 as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        // residual launch
        let resf = module.load_function("sae_fused_residual").gpu_ctx("load residual")?;
        let mut rb = s.launch_builder(&resf);
        rb.arg(&n_i32).arg(&span.inv_tau).arg(&logits).arg(&phi).arg(&dphi)
            .arg(&d2phi).arg(&decoder).arg(&error).arg(&mut resid_dev);
        // SAFETY: grid/block validated; pointers are cudarc allocations on this
        // stream; kernel writes within resid[0..n*q*q].
        unsafe { rb.launch(cfg) }.gpu_ctx("residual launch")?;

        // gram launch
        let gf = module.load_function("sae_fused_gram").gpu_ctx("load gram")?;
        let mut gb = s.launch_builder(&gf);
        gb.arg(&n_i32).arg(&span.inv_tau).arg(&logits).arg(&phi).arg(&dphi)
            .arg(&decoder).arg(&mut decoded_scr).arg(&mut d1_scr).arg(&mut gram_dev);
        // SAFETY: as above; kernel writes within gram[0..n*q*q] + the scratch slabs.
        unsafe { gb.launch(cfg) }.gpu_ctx("gram launch")?;

        let mut gram = vec![0.0; n * q * q];
        let mut resid = vec![0.0; n * q * q];
        s.memcpy_dtoh(&gram_dev, &mut gram).gpu_ctx("dtoh gram")?;
        s.memcpy_dtoh(&resid_dev, &mut resid).gpu_ctx("dtoh resid")?;
        s.synchronize().gpu_ctx("sync")?;
        Ok(FusedContractionSlabs { n_rows: n, q, gram, resid })
    }
}
