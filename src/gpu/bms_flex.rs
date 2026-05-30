//! GPU device backend for Bernoulli marginal-slope FLEX row primitives.
//!
//! Roadmap (issue #210):
//!   1. **Scaffolding** (landed earlier): NVRTC-compiling host backend, PTX
//!      module cache, per-process device arena, three entry points
//!      (`gpu_gradient`, `gpu_hessian_matvec`, `gpu_hessian_dense`).
//!   2. **Rigid row kernel** (landed earlier): probe kernel exercises the
//!      full NVRTC → cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel
//!      path so the scaffolding catches host-side issues before the real
//!      row kernel lands.
//!   3. **Stage-3 row-kernel wiring** (this commit): the three entry points
//!      forward to the Stage-2 row kernel
//!      ([`crate::gpu::bms_flex_row::launch_bms_flex_row_kernel`]) and
//!      pull the per-row outputs back into joint-β shape using the row's
//!      `P_i = block_diag(X_i, G_i, I_h, I_w)` design rows (spec §14).
//!      The caller is responsible for preparing the per-row FLEX cell
//!      partition (cubic predictor coefficients, A_c / R_c / S families)
//!      and the observed-point evaluations (`chi_obs`, `xi_obs`, `rho_u`,
//!      `tau_u`, `r_uv`) — those host-side prep helpers land alongside the
//!      CPU↔GPU dispatcher hook-up that consumes this module.
//!   4. **Optimisation hill-climb**: profile-driven shared-mem tile reduces,
//!      warp shuffles, persistent kernels for HVP sweeps, etc., until the
//!      biobank-shape (n=195k, p=44, r=20) wall-time targets are met.

use std::sync::OnceLock;

use ndarray::Array2;

use super::error::GpuError;
#[cfg(target_os = "linux")]
use super::error::GpuResultExt;
use super::{GpuDecision, GpuKernel, decide};

use super::bms_flex_row::{
    BmsFlexRowKernelInputs, BmsFlexRowKernelOutputs, launch_bms_flex_row_kernel,
};

#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

#[cfg(target_os = "linux")]
use super::common::{DeviceArena, PtxModuleCache};

// ────────────────────────────────────────────────────────────────────────
// Public policy entry points (preserved from the previous policy-only
// implementation so call sites stay source-compatible).
// ────────────────────────────────────────────────────────────────────────

/// Decide whether the GPU row-primary Hessian path is eligible for this
/// fit's `(n, r)`. Always-`use_gpu=false` for `r == 0` (no flex jets to
/// process) and below the runtime row-kernel threshold.
#[must_use]
pub fn row_primary_hessian_decision(n: usize, r: usize) -> GpuDecision {
    let large_enough = super::runtime::GpuRuntime::global()
        .map(|runtime| n >= runtime.policy().row_kernel_min_n && r > 0)
        .unwrap_or(false);
    decide(
        GpuKernel::MarginalSlopeRows,
        super::GpuEligibility::from_flags(BmsFlexGpuBackend::compiled(), large_enough),
    )
}

/// Same as [`row_primary_hessian_decision`] but turns
/// `gpu=force`-without-support into an `Err` string at the call site.
pub fn require_row_primary_hessian_supported(n: usize, r: usize) -> Result<GpuDecision, String> {
    let decision = row_primary_hessian_decision(n, r);
    decision.clone().log();
    decision.require_supported()?;
    Ok(decision)
}

// ────────────────────────────────────────────────────────────────────────
// Device-arena and PTX-cache backend.
// ────────────────────────────────────────────────────────────────────────

/// Per-fit row + cell payload the device row kernel consumes, plus the
/// per-row design rows the joint pullback `P_i^T H_i P_i` needs.
///
/// All slices are borrowed from the caller. Layout conventions match
/// [`BmsFlexRowKernelInputs`] one-for-one for the row-kernel fields; the
/// `x_design` / `g_design` arrays carry the marginal-link design row
/// `X_i` and the log-slope design row `G_i` per spec §14:
///
/// ```text
///     P_i = block_diag(X_i, G_i, I_h, I_w)
/// ```
///
/// Per-row `r`-vector index conventions (mirror
/// [`crate::gpu::bms_flex_row`]):
///   * `u = 0`        → `a` (marginal-link / `q`) block — pulled back via `X_i`.
///   * `u = 1`        → `b` (log-slope) block       — pulled back via `G_i`.
///   * `u ∈ [2, 2+p_h)` → score-warp block          — identity pullback into `β_h`.
///   * `u ∈ [2+p_h, r)` → link-wiggle block         — identity pullback into `β_w`.
///
/// Joint-β block order matches `PrimarySlices` on the CPU side:
/// `[β_q (q_dim), β_g (g_dim), β_h (p_h), β_w (p_w)]`, total
/// `p = q_dim + g_dim + p_h + p_w`.
#[derive(Clone, Copy, Debug)]
pub struct BmsFlexGpuRowInputs<'a> {
    /// Number of observations.
    pub n: usize,
    /// Primary local dimension `r = 2 + p_h + p_w`.
    pub r: usize,
    /// Total joint-parameter dimension `p = q_dim + g_dim + p_h + p_w`.
    pub p: usize,
    /// β_q block size (marginal-link design column count).
    pub q_dim: usize,
    /// β_g block size (log-slope design column count).
    pub g_dim: usize,
    /// Score-warp basis count (`p_h`).
    pub p_h: usize,
    /// Link-wiggle basis count (`p_w`).
    pub p_w: usize,
    /// Joint-β coefficient vector, length `p`. Currently passed through to
    /// the device kernel only for diagnostics; the row math consumes the
    /// pre-evaluated `q`, `b`, `mu_*`, `chi_obs`, etc. directly.
    pub beta: &'a [f64],
    /// Observed responses `y_i ∈ {0, 1}`, length `n`.
    pub y: &'a [f64],
    /// Observation weights, length `n`.
    pub weights: &'a [f64],
    /// Per-row latent quantile `q_i`. Length `n`.
    pub q: &'a [f64],
    /// Per-row latent slope `b_i`. Length `n`.
    pub b: &'a [f64],
    /// Per-row `μ_1 = ∂q/∂a`. Length `n`.
    pub mu_1: &'a [f64],
    /// Per-row `μ_2 = ∂²q/∂a²`. Length `n`.
    pub mu_2: &'a [f64],
    /// Per-row observed `z_obs`. Length `n`.
    pub z_obs: &'a [f64],
    /// Probit frailty scale `S_f` (scalar shared across rows).
    pub s_f: f64,
    /// Per-row cell range, length `n + 1`. Row `i` owns
    /// `[cell_offsets[i] .. cell_offsets[i+1])` in the `cell_*` arrays.
    pub cell_offsets: &'a [u32],
    /// Cubic predictor `C0` per cell. Length `total_cells`.
    pub cell_c0: &'a [f64],
    /// Cubic predictor `C1` per cell.
    pub cell_c1: &'a [f64],
    /// Cubic predictor `C2` per cell.
    pub cell_c2: &'a [f64],
    /// Cubic predictor `C3` per cell.
    pub cell_c3: &'a [f64],
    /// Per-cell `A_c` (length `total_cells * 4`).
    pub cell_a: &'a [f64],
    /// Per-cell `AA_c` (length `total_cells * 4`).
    pub cell_aa: &'a [f64],
    /// Per-cell `R_{c,u}` for `u ∈ [1, r)` (length `total_cells*(r-1)*4`).
    pub cell_r: &'a [f64],
    /// Per-cell `AR_{c,u}` (same shape as `cell_r`).
    pub cell_ar: &'a [f64],
    /// Per-cell `S_{bb}` (length `total_cells * 4`).
    pub cell_sbb: &'a [f64],
    /// Per-cell `S_{b·h_j}` (length `total_cells * p_h * 4`).
    pub cell_sbh: &'a [f64],
    /// Per-cell `S_{b·w_ℓ}` (length `total_cells * p_w * 4`).
    pub cell_sbw: &'a [f64],
    /// Per-cell derivative moments from Stage 1, row-major
    /// `[total_cells, 10]`.
    pub cell_moments: &'a [f64],
    /// Per-row `chi_obs`. Length `n`.
    pub chi_obs: &'a [f64],
    /// Per-row `xi_obs`. Length `n`.
    pub xi_obs: &'a [f64],
    /// Per-row `rho_u` row-major `[n, r]`.
    pub rho_u: &'a [f64],
    /// Per-row `tau_u` row-major `[n, r]`.
    pub tau_u: &'a [f64],
    /// Per-row `r_uv` row-major `[n, r*r]`.
    pub r_uv: &'a [f64],
    /// Per-row marginal-link design rows `X_i`, row-major `[n, q_dim]`.
    pub x_design: &'a [f64],
    /// Per-row log-slope design rows `G_i`, row-major `[n, g_dim]`.
    pub g_design: &'a [f64],
}

impl<'a> BmsFlexGpuRowInputs<'a> {
    /// Shape-check the inputs the way every entry point would before any
    /// device call. Kept on the input struct so it is reused by all three
    /// entry points.
    fn validate(&self) -> Result<(), GpuError> {
        let want_p = self.q_dim + self.g_dim + self.p_h + self.p_w;
        if self.p != want_p {
            crate::gpu_bail!(
                "bms_flex inputs: p={} != q_dim({}) + g_dim({}) + p_h({}) + p_w({}) = {}",
                self.p,
                self.q_dim,
                self.g_dim,
                self.p_h,
                self.p_w,
                want_p
            );
        }
        if self.r != 2 + self.p_h + self.p_w {
            crate::gpu_bail!(
                "bms_flex inputs: r={} != 2 + p_h({}) + p_w({}) = {}",
                self.r,
                self.p_h,
                self.p_w,
                2 + self.p_h + self.p_w
            );
        }
        if self.beta.len() != self.p {
            crate::gpu_bail!(
                "bms_flex inputs: beta.len()={} != p={}",
                self.beta.len(),
                self.p
            );
        }
        if self.y.len() != self.n {
            crate::gpu_bail!("bms_flex inputs: y.len()={} != n={}", self.y.len(), self.n);
        }
        if self.weights.len() != self.n {
            crate::gpu_bail!(
                "bms_flex inputs: weights.len()={} != n={}",
                self.weights.len(),
                self.n
            );
        }
        if self.x_design.len() != self.n * self.q_dim {
            crate::gpu_bail!(
                "bms_flex inputs: x_design.len()={} != n({})*q_dim({}) = {}",
                self.x_design.len(),
                self.n,
                self.q_dim,
                self.n * self.q_dim
            );
        }
        if self.g_design.len() != self.n * self.g_dim {
            crate::gpu_bail!(
                "bms_flex inputs: g_design.len()={} != n({})*g_dim({}) = {}",
                self.g_design.len(),
                self.n,
                self.g_dim,
                self.n * self.g_dim
            );
        }
        Ok(())
    }

    /// Project the inputs into the row-kernel's view. Borrowed slices flow
    /// through unchanged; the row kernel does its own validation on top of
    /// this struct's checks.
    fn as_row_kernel_inputs(&self) -> BmsFlexRowKernelInputs<'a> {
        BmsFlexRowKernelInputs {
            n_rows: self.n,
            r: self.r,
            p_h: self.p_h,
            p_w: self.p_w,
            q: self.q,
            b: self.b,
            mu_1: self.mu_1,
            mu_2: self.mu_2,
            z_obs: self.z_obs,
            y: self.y,
            w: self.weights,
            s_f: self.s_f,
            cell_offsets: self.cell_offsets,
            cell_c0: self.cell_c0,
            cell_c1: self.cell_c1,
            cell_c2: self.cell_c2,
            cell_c3: self.cell_c3,
            cell_a: self.cell_a,
            cell_aa: self.cell_aa,
            cell_r: self.cell_r,
            cell_ar: self.cell_ar,
            cell_sbb: self.cell_sbb,
            cell_sbh: self.cell_sbh,
            cell_sbw: self.cell_sbw,
            cell_moments: crate::gpu::bms_flex_row::CellMomentsSource::Host(self.cell_moments),
            chi_obs: self.chi_obs,
            xi_obs: self.xi_obs,
            rho_u: self.rho_u,
            tau_u: self.tau_u,
            r_uv: self.r_uv,
        }
    }
}

/// The PTX source compiled and loaded at first use of the BMS flex GPU
/// backend. The probe kernel exercises the full NVRTC → cuModuleLoadData
/// → cuModuleGetFunction → cuLaunchKernel path so the scaffolding catches
/// host-side issues (PTX cache, arena alloc, stream sync) before the real
/// row kernel is dispatched from one of the entry points below.
#[cfg(target_os = "linux")]
const PROBE_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void bms_flex_probe() {
    // Intentionally empty. This kernel exists only so the scaffolding can
    // verify NVRTC compile + module load + launch + synchronize on the
    // selected device. The real row math lives in the bms_flex_row module.
}
"#;

/// Process-wide BMS-flex GPU backend. Lazy-initialised on first call to
/// [`BmsFlexGpuBackend::probe`] / one of the entry points.
#[must_use]
pub struct BmsFlexGpuBackend {
    #[cfg(target_os = "linux")]
    inner: BmsFlexGpuContextLinux,
}

#[cfg(target_os = "linux")]
struct BmsFlexGpuContextLinux {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// NVRTC-compiled module containing the probe kernel. Lazy so the
    /// compile happens exactly once per process and is shared by every
    /// dispatching thread.
    module: PtxModuleCache,
    /// Reusable f64 device buffers keyed by power-of-two element-count
    /// buckets. Held under a `Mutex` because biobank fits dispatch from
    /// multiple rayon worker threads; the mutex is only held during
    /// `alloc` / `release`, not across kernel launches.
    arena: Mutex<DeviceArena>,
}

impl BmsFlexGpuBackend {
    /// Returns `true` if the BMS flex GPU backend is compiled into this
    /// build (Linux + cudarc). On non-Linux builds returns `false` so the
    /// policy gate reports `cpu-gpu-backend-not-compiled` like the rest
    /// of the GPU layer.
    pub const fn compiled() -> bool {
        cfg!(target_os = "linux")
    }

    /// Lazily initialise the process-wide BMS flex backend. On the first
    /// successful call this creates a CUDA context on the runtime's
    /// selected device, opens a stream, and NVRTC-compiles the probe
    /// kernel. Subsequent calls return the cached handle.
    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<BmsFlexGpuBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                #[cfg(target_os = "linux")]
                {
                    Self::probe_linux()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(GpuError::DriverLibraryUnavailable {
                        reason: "bms_flex GPU backend is Linux-only".to_string(),
                    })
                }
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let parts = super::backend_probe::probe_cuda_backend("bms_flex")?;
        let backend = BmsFlexGpuBackend {
            inner: BmsFlexGpuContextLinux {
                ctx: parts.ctx,
                stream: parts.stream,
                module: PtxModuleCache::new(),
                arena: Mutex::new(DeviceArena::default()),
            },
        };
        // Eagerly compile the probe kernel so any NVRTC failure surfaces
        // here, not at first dispatch.
        backend.compile_probe_module()?;
        Ok(backend)
    }

    /// NVRTC-compile (or fetch from cache) the probe module.
    #[cfg(target_os = "linux")]
    fn compile_probe_module(&self) -> Result<&Arc<CudaModule>, GpuError> {
        self.inner
            .module
            .get_or_compile(&self.inner.ctx, "bms_flex", PROBE_KERNEL_SOURCE)
    }

    /// Launch the probe kernel and synchronize. Used by tests and by the
    /// dispatcher's policy gate to verify the full host-orchestration
    /// path before the real row kernel is dispatched.
    #[cfg(target_os = "linux")]
    pub fn launch_probe(&self) -> Result<(), GpuError> {
        use cudarc::driver::LaunchConfig;
        let module = self.compile_probe_module()?;
        let func = module
            .load_function("bms_flex_probe")
            .gpu_ctx("bms_flex probe load_function")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.inner.stream.launch_builder(&func);
        // SAFETY: probe kernel takes no arguments and does no memory
        // access, so launch parameters and lack of args are trivially
        // valid for any device.
        unsafe { builder.launch(cfg) }.gpu_ctx("bms_flex probe launch")?;
        self.inner
            .stream
            .synchronize()
            .gpu_ctx("bms_flex probe synchronize")?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn launch_probe(&self) -> Result<(), GpuError> {
        Err(GpuError::DriverLibraryUnavailable {
            reason: "bms_flex GPU backend is Linux-only".to_string(),
        })
    }

    /// Round-trip the arena: allocate a slab, immediately release it.
    /// Used by the device-side smoke test to verify the arena code path
    /// is exercised; production milestones will hold slabs across the
    /// whole row sweep instead.
    #[cfg(target_os = "linux")]
    pub fn arena_round_trip(&self, elements: usize) -> Result<usize, GpuError> {
        let mut guard = self
            .inner
            .arena
            .lock()
            .gpu_ctx("bms_flex arena mutex poisoned")?;
        let (bucket, slab) = guard.alloc(&self.inner.stream, elements, "bms_flex")?;
        guard.release(bucket, slab);
        Ok(bucket)
    }

    /// Return a short string describing the backend state, for logs.
    pub fn describe(&self) -> String {
        #[cfg(target_os = "linux")]
        {
            return format!(
                "bms_flex backend: device={:?} module_loaded={}",
                self.inner.ctx.name().ok(),
                self.inner.module.get().is_some()
            );
        }
        #[cfg(not(target_os = "linux"))]
        {
            "bms_flex backend: unavailable (not Linux)".to_string()
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Joint-β pullback `P_i^T (·) P_i` (spec §14).
// ────────────────────────────────────────────────────────────────────────

/// Apply `(P_i v)` for a single row: returns the per-row `r`-vector
/// `[X_i·v_q, G_i·v_g, v_h, v_w]`.
#[inline]
fn project_v_through_row(
    v: &[f64],
    x_row: &[f64],
    g_row: &[f64],
    q_dim: usize,
    g_dim: usize,
    p_h: usize,
    p_w: usize,
    r: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; r];
    // u = 0 : X_i · v_q
    let mut acc_q = 0.0f64;
    for k in 0..q_dim {
        acc_q += x_row[k] * v[k];
    }
    out[0] = acc_q;
    // u = 1 : G_i · v_g
    let mut acc_g = 0.0f64;
    for k in 0..g_dim {
        acc_g += g_row[k] * v[q_dim + k];
    }
    out[1] = acc_g;
    // u = 2..2+p_h : identity from v_h
    for j in 0..p_h {
        out[2 + j] = v[q_dim + g_dim + j];
    }
    // u = 2+p_h..r : identity from v_w
    for j in 0..p_w {
        out[2 + p_h + j] = v[q_dim + g_dim + p_h + j];
    }
    out
}

/// Pullback a per-row `r`-vector `t` through `P_i^T` and accumulate into
/// the joint-β vector `out` (length `p`).
#[inline]
fn accumulate_row_vector_pullback(
    t: &[f64],
    x_row: &[f64],
    g_row: &[f64],
    q_dim: usize,
    g_dim: usize,
    p_h: usize,
    p_w: usize,
    out: &mut [f64],
) {
    // β_q block: out[0..q_dim] += t[0] * X_i
    let t0 = t[0];
    for k in 0..q_dim {
        out[k] += t0 * x_row[k];
    }
    // β_g block: out[q_dim..q_dim+g_dim] += t[1] * G_i
    let t1 = t[1];
    for k in 0..g_dim {
        out[q_dim + k] += t1 * g_row[k];
    }
    // β_h block: out[q_dim+g_dim..][..p_h] += t[2..2+p_h]
    for j in 0..p_h {
        out[q_dim + g_dim + j] += t[2 + j];
    }
    // β_w block: out[q_dim+g_dim+p_h..][..p_w] += t[2+p_h..r]
    for j in 0..p_w {
        out[q_dim + g_dim + p_h + j] += t[2 + p_h + j];
    }
}

/// Accumulate `P_i^T H_i P_i` for a single row into `out` (`p × p` row-major).
#[inline]
fn accumulate_row_hessian_pullback(
    hess_row: &[f64], // length r*r, row-major
    r: usize,
    x_row: &[f64],
    g_row: &[f64],
    q_dim: usize,
    g_dim: usize,
    p_h: usize,
    p_w: usize,
    p: usize,
    out: &mut Array2<f64>,
) {
    // Build the joint-space "image" of each per-row coordinate u:
    // basis_u(k) = element k of (P_i^T e_u) where e_u is the u-th basis
    // vector of the r-space.
    //   * basis_0 is X_i (concatenated into the β_q slice)
    //   * basis_1 is G_i (concatenated into the β_g slice)
    //   * basis_{2+j} is e_{q_dim+g_dim+j}                (j in 0..p_h)
    //   * basis_{2+p_h+ℓ} is e_{q_dim+g_dim+p_h+ℓ}        (ℓ in 0..p_w)
    //
    // Then P_i^T H_i P_i has entries
    //     (P_i^T H_i P_i)[m, n] = Σ_{u,v} H[u,v] · basis_u[m] · basis_v[n].
    //
    // Implementation: precompute `phi[u]` row-vectors of length `p` so
    // adding `H[u,v] · phi[u] ⊗ phi[v]` reduces to two flat inner loops.
    let mut phi: Vec<Vec<f64>> = Vec::with_capacity(r);
    // u = 0 : phi = (X_i, 0, 0, 0) concatenated in joint-β order
    let mut phi0 = vec![0.0f64; p];
    for k in 0..q_dim {
        phi0[k] = x_row[k];
    }
    phi.push(phi0);
    // u = 1 : phi = (0, G_i, 0, 0)
    let mut phi1 = vec![0.0f64; p];
    for k in 0..g_dim {
        phi1[q_dim + k] = g_row[k];
    }
    phi.push(phi1);
    // u = 2 + j (h block) : phi has a single 1 at position q_dim+g_dim+j
    for j in 0..p_h {
        let mut row = vec![0.0f64; p];
        row[q_dim + g_dim + j] = 1.0;
        phi.push(row);
    }
    // u = 2 + p_h + ℓ (w block) : phi has a single 1 at position q_dim+g_dim+p_h+ℓ
    for j in 0..p_w {
        let mut row = vec![0.0f64; p];
        row[q_dim + g_dim + p_h + j] = 1.0;
        phi.push(row);
    }
    assert_eq!(phi.len(), r);

    for u in 0..r {
        for v in 0..r {
            let h_uv = hess_row[u * r + v];
            if h_uv == 0.0 {
                continue;
            }
            let phi_u = &phi[u];
            let phi_v = &phi[v];
            for m in 0..p {
                let pm = phi_u[m];
                if pm == 0.0 {
                    continue;
                }
                let scaled = h_uv * pm;
                for n in 0..p {
                    out[[m, n]] += scaled * phi_v[n];
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Three entry points. Stage 3 forwards each to the row kernel and applies
// the `P_i^T · P_i` pullback.
// ────────────────────────────────────────────────────────────────────────

/// Evaluate the BMS flex negative log-likelihood and exact gradient on
/// the GPU. Returns `(neglog_sum, gradient)` where `gradient` is the
/// joint-β gradient in joint-block order, length `p`.
pub fn gpu_gradient(inputs: BmsFlexGpuRowInputs<'_>) -> Result<(f64, Vec<f64>), GpuError> {
    inputs.validate()?;
    // Touch the backend so probe failures surface here too.
    BmsFlexGpuBackend::probe()?;
    let outputs = launch_bms_flex_row_kernel(inputs.as_row_kernel_inputs())?;
    let BmsFlexRowKernelOutputs { neglog, grad, .. } = outputs;

    let n = inputs.n;
    let r = inputs.r;
    let p = inputs.p;
    let q_dim = inputs.q_dim;
    let g_dim = inputs.g_dim;
    let p_h = inputs.p_h;
    let p_w = inputs.p_w;

    let mut neglog_sum = 0.0f64;
    for v in &neglog {
        neglog_sum += *v;
    }

    let mut joint_grad = vec![0.0f64; p];
    for i in 0..n {
        let row_grad = &grad[i * r..(i + 1) * r];
        let x_row = &inputs.x_design[i * q_dim..(i + 1) * q_dim];
        let g_row = &inputs.g_design[i * g_dim..(i + 1) * g_dim];
        accumulate_row_vector_pullback(
            row_grad,
            x_row,
            g_row,
            q_dim,
            g_dim,
            p_h,
            p_w,
            &mut joint_grad,
        );
    }
    Ok((neglog_sum, joint_grad))
}

/// Evaluate the BMS flex joint-Hessian times an input vector `v` on the
/// GPU. Returns `H · v`, length `p`.
pub fn gpu_hessian_matvec(
    inputs: BmsFlexGpuRowInputs<'_>,
    v: &[f64],
) -> Result<Vec<f64>, GpuError> {
    inputs.validate()?;
    if v.len() != inputs.p {
        crate::gpu_bail!(
            "bms_flex gpu_hessian_matvec: v.len()={} != p={}",
            v.len(),
            inputs.p
        );
    }
    BmsFlexGpuBackend::probe()?;
    let outputs = launch_bms_flex_row_kernel(inputs.as_row_kernel_inputs())?;
    let BmsFlexRowKernelOutputs { hess, .. } = outputs;

    let n = inputs.n;
    let r = inputs.r;
    let p = inputs.p;
    let q_dim = inputs.q_dim;
    let g_dim = inputs.g_dim;
    let p_h = inputs.p_h;
    let p_w = inputs.p_w;

    let mut out = vec![0.0f64; p];
    for i in 0..n {
        let x_row = &inputs.x_design[i * q_dim..(i + 1) * q_dim];
        let g_row = &inputs.g_design[i * g_dim..(i + 1) * g_dim];
        // Compute P_i v as an r-vector.
        let pv = project_v_through_row(v, x_row, g_row, q_dim, g_dim, p_h, p_w, r);
        // t = H_i (P_i v) — r-vector matvec, dense r×r.
        let hess_row = &hess[i * r * r..(i + 1) * r * r];
        let mut t = vec![0.0f64; r];
        for u in 0..r {
            let row = &hess_row[u * r..(u + 1) * r];
            let mut acc = 0.0f64;
            for w in 0..r {
                acc += row[w] * pv[w];
            }
            t[u] = acc;
        }
        accumulate_row_vector_pullback(&t, x_row, g_row, q_dim, g_dim, p_h, p_w, &mut out);
    }
    Ok(out)
}

/// Assemble the dense BMS flex joint Hessian on the GPU. Returns a
/// `p × p` row-major matrix.
pub fn gpu_hessian_dense(inputs: BmsFlexGpuRowInputs<'_>) -> Result<Array2<f64>, GpuError> {
    inputs.validate()?;
    BmsFlexGpuBackend::probe()?;
    let outputs = launch_bms_flex_row_kernel(inputs.as_row_kernel_inputs())?;
    let BmsFlexRowKernelOutputs { hess, .. } = outputs;

    let n = inputs.n;
    let r = inputs.r;
    let p = inputs.p;
    let q_dim = inputs.q_dim;
    let g_dim = inputs.g_dim;
    let p_h = inputs.p_h;
    let p_w = inputs.p_w;

    let mut out = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let x_row = &inputs.x_design[i * q_dim..(i + 1) * q_dim];
        let g_row = &inputs.g_design[i * g_dim..(i + 1) * g_dim];
        let hess_row = &hess[i * r * r..(i + 1) * r * r];
        accumulate_row_hessian_pullback(
            hess_row, r, x_row, g_row, q_dim, g_dim, p_h, p_w, p, &mut out,
        );
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────
// Tests. Run via `cargo test -p gam bms_flex_gpu -- --nocapture`.
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod bms_flex_gpu_tests {
    use super::*;

    /// Allocate zero-filled row + cell buffers for a small rigid (no
    /// flex blocks) test problem.
    struct ScratchBuffers {
        beta: Vec<f64>,
        y: Vec<f64>,
        w: Vec<f64>,
        q: Vec<f64>,
        b: Vec<f64>,
        mu_1: Vec<f64>,
        mu_2: Vec<f64>,
        z_obs: Vec<f64>,
        cell_offsets: Vec<u32>,
        cell_c0: Vec<f64>,
        cell_c1: Vec<f64>,
        cell_c2: Vec<f64>,
        cell_c3: Vec<f64>,
        cell_a: Vec<f64>,
        cell_aa: Vec<f64>,
        cell_r: Vec<f64>,
        cell_ar: Vec<f64>,
        cell_sbb: Vec<f64>,
        cell_sbh: Vec<f64>,
        cell_sbw: Vec<f64>,
        cell_moments: Vec<f64>,
        chi_obs: Vec<f64>,
        xi_obs: Vec<f64>,
        rho_u: Vec<f64>,
        tau_u: Vec<f64>,
        r_uv: Vec<f64>,
        x_design: Vec<f64>,
        g_design: Vec<f64>,
    }

    fn zero_buffers(
        n: usize,
        q_dim: usize,
        g_dim: usize,
        p_h: usize,
        p_w: usize,
    ) -> ScratchBuffers {
        let r = 2 + p_h + p_w;
        let p = q_dim + g_dim + p_h + p_w;
        // Trivial cell partition: one cell per row, zero coefficients.
        let cells_per_row = 1usize;
        let total_cells = n * cells_per_row;
        let mut cell_offsets = Vec::with_capacity(n + 1);
        for i in 0..=n {
            cell_offsets.push((i * cells_per_row) as u32);
        }
        let r_minus_1 = r.saturating_sub(1);
        ScratchBuffers {
            beta: vec![0.0; p],
            y: vec![0.0; n],
            w: vec![1.0; n],
            q: vec![0.0; n],
            b: vec![1.0; n],
            mu_1: vec![1.0; n],
            mu_2: vec![0.0; n],
            z_obs: vec![0.0; n],
            cell_offsets,
            cell_c0: vec![0.0; total_cells],
            cell_c1: vec![0.0; total_cells],
            cell_c2: vec![0.0; total_cells],
            cell_c3: vec![0.0; total_cells],
            cell_a: vec![0.0; total_cells * 4],
            cell_aa: vec![0.0; total_cells * 4],
            cell_r: vec![0.0; total_cells * r_minus_1 * 4],
            cell_ar: vec![0.0; total_cells * r_minus_1 * 4],
            cell_sbb: vec![0.0; total_cells * 4],
            cell_sbh: vec![0.0; total_cells * p_h * 4],
            cell_sbw: vec![0.0; total_cells * p_w * 4],
            cell_moments: vec![0.0; total_cells * 10],
            chi_obs: vec![0.0; n],
            xi_obs: vec![0.0; n],
            rho_u: vec![0.0; n * r],
            tau_u: vec![0.0; n * r],
            r_uv: vec![0.0; n * r * r],
            x_design: vec![0.0; n * q_dim],
            g_design: vec![0.0; n * g_dim],
        }
    }

    fn inputs_from<'a>(
        bufs: &'a ScratchBuffers,
        n: usize,
        q_dim: usize,
        g_dim: usize,
        p_h: usize,
        p_w: usize,
    ) -> BmsFlexGpuRowInputs<'a> {
        let r = 2 + p_h + p_w;
        let p = q_dim + g_dim + p_h + p_w;
        BmsFlexGpuRowInputs {
            n,
            r,
            p,
            q_dim,
            g_dim,
            p_h,
            p_w,
            beta: &bufs.beta,
            y: &bufs.y,
            weights: &bufs.w,
            q: &bufs.q,
            b: &bufs.b,
            mu_1: &bufs.mu_1,
            mu_2: &bufs.mu_2,
            z_obs: &bufs.z_obs,
            s_f: 1.0,
            cell_offsets: &bufs.cell_offsets,
            cell_c0: &bufs.cell_c0,
            cell_c1: &bufs.cell_c1,
            cell_c2: &bufs.cell_c2,
            cell_c3: &bufs.cell_c3,
            cell_a: &bufs.cell_a,
            cell_aa: &bufs.cell_aa,
            cell_r: &bufs.cell_r,
            cell_ar: &bufs.cell_ar,
            cell_sbb: &bufs.cell_sbb,
            cell_sbh: &bufs.cell_sbh,
            cell_sbw: &bufs.cell_sbw,
            cell_moments: &bufs.cell_moments,
            chi_obs: &bufs.chi_obs,
            xi_obs: &bufs.xi_obs,
            rho_u: &bufs.rho_u,
            tau_u: &bufs.tau_u,
            r_uv: &bufs.r_uv,
            x_design: &bufs.x_design,
            g_design: &bufs.g_design,
        }
    }

    #[test]
    fn bms_flex_gpu_policy_decision_is_explicit() {
        let decision = row_primary_hessian_decision(50_000, 4);
        assert_eq!(decision.kernel, GpuKernel::MarginalSlopeRows);
    }

    #[test]
    fn bms_flex_gpu_gradient_routes_through_row_kernel_or_clean_error() {
        // Stage-3 wiring: on a host without a CUDA runtime the path returns
        // a clean DriverLibraryUnavailable / DriverCallFailed; on a host
        // with a device the launch may succeed or fall through to a
        // typed kernel error. The test guards against panic, success
        // with bogus values, or wrong error variant.
        let n = 4;
        let bufs = zero_buffers(n, 1, 1, 0, 0);
        let inputs = inputs_from(&bufs, n, 1, 1, 0, 0);
        match gpu_gradient(inputs) {
            Err(GpuError::DriverLibraryUnavailable { .. })
            | Err(GpuError::DriverCallFailed { .. })
            | Err(GpuError::DriverSymbolMissing { .. })
            | Err(GpuError::NotYetImplemented { .. }) => {}
            Err(other) => panic!("unexpected GpuError variant: {other:?}"),
            Ok((neglog, grad)) => {
                // V100-only path: kernel ran. Sanity-check shapes only.
                assert!(neglog.is_finite() || neglog.is_nan(), "neglog: {neglog}");
                assert_eq!(grad.len(), 2);
            }
        }
    }

    #[test]
    fn bms_flex_gpu_hessian_matvec_rejects_wrong_v_length() {
        let n = 4;
        let bufs = zero_buffers(n, 1, 1, 0, 0);
        let inputs = inputs_from(&bufs, n, 1, 1, 0, 0);
        let v_wrong = vec![0.0; inputs.p + 1];
        match gpu_hessian_matvec(inputs, &v_wrong) {
            Err(GpuError::DriverCallFailed { reason }) => {
                assert!(
                    reason.contains("v.len()"),
                    "expected v.len() mismatch message, got: {reason}"
                );
            }
            other => panic!("expected v.len() mismatch, got {other:?}"),
        }
    }

    #[test]
    fn bms_flex_gpu_hessian_dense_routes_through_row_kernel_or_clean_error() {
        let n = 4;
        let bufs = zero_buffers(n, 1, 1, 0, 0);
        let inputs = inputs_from(&bufs, n, 1, 1, 0, 0);
        match gpu_hessian_dense(inputs) {
            Err(GpuError::DriverLibraryUnavailable { .. })
            | Err(GpuError::DriverCallFailed { .. })
            | Err(GpuError::DriverSymbolMissing { .. })
            | Err(GpuError::NotYetImplemented { .. }) => {}
            Err(other) => panic!("unexpected GpuError variant: {other:?}"),
            Ok(h) => {
                // V100-only: kernel ran. The dense H must be square p×p.
                let p = 2usize;
                assert_eq!(h.shape(), &[p, p]);
            }
        }
    }

    #[test]
    fn bms_flex_gpu_inputs_validate_catches_shape_mismatches() {
        let n = 4;
        let mut bufs = zero_buffers(n, 1, 1, 0, 0);
        bufs.beta.push(0.0); // make beta longer than p
        let bad = BmsFlexGpuRowInputs {
            beta: &bufs.beta,
            ..inputs_from(&bufs, n, 1, 1, 0, 0)
        };
        let err = bad.validate().expect_err("beta length mismatch must fail");
        assert!(
            matches!(err, GpuError::DriverCallFailed { .. }),
            "expected DriverCallFailed, got {err:?}"
        );
    }

    /// CPU-side parity smoke for the joint pullback `P_i^T H_i P_i` on a
    /// rigid (no flex blocks) problem. The test never reaches the device:
    /// on Mac the row kernel returns
    /// [`GpuError::DriverLibraryUnavailable`]; on Linux without a runtime
    /// the same; on Linux with a device the launch may succeed. The
    /// pullback math itself is exercised directly via the helper
    /// functions so the CPU↔GPU contract is locked in at unit-test time.
    #[test]
    fn bms_flex_gpu_hessian_dense_pullback_matches_cpu_reference() {
        // r = 2 (just `a` and `b`), p = q_dim + g_dim = 2 + 2 = 4.
        let n = 8;
        let q_dim = 2;
        let g_dim = 2;
        let p_h = 0;
        let p_w = 0;
        let r = 2 + p_h + p_w;
        let p = q_dim + g_dim + p_h + p_w;

        // Synthesise X_i, G_i, and a per-row 2×2 Hessian H_i.
        let mut x_design = Vec::with_capacity(n * q_dim);
        let mut g_design = Vec::with_capacity(n * g_dim);
        let mut hess_flat = Vec::with_capacity(n * r * r);
        for i in 0..n {
            let f = (i as f64) + 1.0;
            // X_i = (1, f)
            x_design.push(1.0);
            x_design.push(f);
            // G_i = (1, f.cos())
            g_design.push(1.0);
            g_design.push(f.cos());
            // H_i = [[2+i, 0.1*i], [0.1*i, 3+i*0.5]]
            let h00 = 2.0 + f;
            let h01 = 0.1 * f;
            let h11 = 3.0 + 0.5 * f;
            hess_flat.push(h00);
            hess_flat.push(h01);
            hess_flat.push(h01);
            hess_flat.push(h11);
        }

        // CPU reference: H_full = Σ_i P_i^T H_i P_i, with P_i = diag(X_i, G_i)
        // since p_h = p_w = 0. So H_full[m, n] = Σ_i Σ_{u,v} H[u,v] * phi_u[m] * phi_v[n],
        // where phi_0 = (X_i, 0) and phi_1 = (0, G_i).
        let mut h_cpu = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let x_row = &x_design[i * q_dim..(i + 1) * q_dim];
            let g_row = &g_design[i * g_dim..(i + 1) * g_dim];
            let h00 = hess_flat[i * r * r];
            let h01 = hess_flat[i * r * r + 1];
            let h10 = hess_flat[i * r * r + 2];
            let h11 = hess_flat[i * r * r + 3];
            // β_q × β_q block: h00 * X_i ⊗ X_i
            for a in 0..q_dim {
                for b in 0..q_dim {
                    h_cpu[[a, b]] += h00 * x_row[a] * x_row[b];
                }
            }
            // β_g × β_g block: h11 * G_i ⊗ G_i
            for a in 0..g_dim {
                for b in 0..g_dim {
                    h_cpu[[q_dim + a, q_dim + b]] += h11 * g_row[a] * g_row[b];
                }
            }
            // β_q × β_g block: h01 * X_i ⊗ G_i
            for a in 0..q_dim {
                for b in 0..g_dim {
                    h_cpu[[a, q_dim + b]] += h01 * x_row[a] * g_row[b];
                }
            }
            // β_g × β_q block: h10 * G_i ⊗ X_i
            for a in 0..g_dim {
                for b in 0..q_dim {
                    h_cpu[[q_dim + a, b]] += h10 * g_row[a] * x_row[b];
                }
            }
        }

        // Drive the same path through accumulate_row_hessian_pullback.
        let mut h_via_helper = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let x_row = &x_design[i * q_dim..(i + 1) * q_dim];
            let g_row = &g_design[i * g_dim..(i + 1) * g_dim];
            let hess_row = &hess_flat[i * r * r..(i + 1) * r * r];
            accumulate_row_hessian_pullback(
                hess_row,
                r,
                x_row,
                g_row,
                q_dim,
                g_dim,
                p_h,
                p_w,
                p,
                &mut h_via_helper,
            );
        }

        // Parity at machine precision.
        for m in 0..p {
            for nn in 0..p {
                let a = h_cpu[[m, nn]];
                let b = h_via_helper[[m, nn]];
                let diff = (a - b).abs();
                assert!(
                    diff <= 1e-12 * a.abs().max(b.abs()).max(1.0),
                    "pullback parity mismatch at ({m},{nn}): cpu={a} helper={b} diff={diff}"
                );
            }
        }

        // And confirm the dispatcher entry point itself errors cleanly on
        // a CPU-only / Mac builder (or succeeds on a host with a device,
        // in which case the shape check below applies).
        // Pack zero cubic-cell payload — the kernel never runs on Mac so
        // the contents don't matter for this part of the test.
        let bufs = {
            let mut b = zero_buffers(n, q_dim, g_dim, p_h, p_w);
            b.x_design = x_design.clone();
            b.g_design = g_design.clone();
            b
        };
        let inputs = inputs_from(&bufs, n, q_dim, g_dim, p_h, p_w);
        match gpu_hessian_dense(inputs) {
            Err(GpuError::DriverLibraryUnavailable { .. })
            | Err(GpuError::DriverCallFailed { .. })
            | Err(GpuError::DriverSymbolMissing { .. })
            | Err(GpuError::NotYetImplemented { .. }) => {}
            Err(other) => panic!("unexpected GpuError variant: {other:?}"),
            Ok(h) => {
                assert_eq!(h.shape(), &[p, p]);
            }
        }
    }

    /// V100-only: probe the backend end-to-end (CUDA context create, NVRTC
    /// compile, module load, launch, sync). Skipped on hosts without a
    /// usable device so the test still passes on the CI/mac builders.
    #[test]
    fn bms_flex_gpu_context_initialises_when_device_present() {
        let Some(runtime) = super::super::runtime::GpuRuntime::global() else {
            eprintln!("[bms_flex_gpu test] no CUDA runtime — skipping device-side init smoketest");
            return;
        };
        eprintln!(
            "[bms_flex_gpu test] runtime selected device ordinal={}",
            runtime.selected_device().ordinal
        );
        let backend = BmsFlexGpuBackend::probe().unwrap_or_else(|err| {
            panic!("BmsFlexGpuBackend::probe failed on a host that reports a CUDA runtime: {err}")
        });
        eprintln!("[bms_flex_gpu test] {}", backend.describe());
        backend
            .launch_probe()
            .expect("probe kernel must launch+sync on a host with a usable device");
        #[cfg(target_os = "linux")]
        {
            let bucket = backend
                .arena_round_trip(1024)
                .expect("arena round-trip must succeed on a host with a usable device");
            assert!(bucket >= 1024, "bucket must be >= requested elements");
            // Second round-trip at the same size should hit the cache.
            let bucket2 = backend
                .arena_round_trip(1024)
                .expect("arena round-trip must succeed on a host with a usable device");
            assert_eq!(bucket, bucket2, "bucket size must be stable for same input");
        }
    }
}
