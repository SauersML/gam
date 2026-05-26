//! GPU device backend for Bernoulli marginal-slope FLEX row primitives.
//!
//! Roadmap (issue #210):
//!   1. **Scaffolding** (this module's current state): NVRTC-compiling host
//!      backend, PTX module cache, per-process device arena, three entry
//!      points (`gpu_gradient`, `gpu_hessian_matvec`, `gpu_hessian_dense`)
//!      whose bodies return [`GpuError::NotYetImplemented`]. The dispatcher
//!      in `bernoulli_marginal_slope.rs` routes through these first and falls
//!      back to the CPU path on that sentinel — so the host orchestration is
//!      under test (probe + context init + arena alloc + module load of a
//!      placeholder kernel) before any row-level math lands on device.
//!   2. **Rigid row kernel**: NVRTC source covering the flex=false subset
//!      (probit + Mills ratio + design-row contribution). Replaces the
//!      sentinel for the rigid branch. Parity-within-1e-8 against
//!      `rigid_row_kernel_eval`.
//!   3. **Flex row kernel**: full denested-cell + score-warp + link-wiggle
//!      jet calculus on device, mirroring
//!      `compute_row_analytic_flex_into_with_moments`. Largest milestone;
//!      uses the math team's per-row state simplifications.
//!   4. **Optimisation hill-climb**: profile-driven shared-mem tile reduces,
//!      warp shuffles, persistent kernels for HVP sweeps, etc., until the
//!      biobank-shape (n=195k, p=44, r=20) wall-time targets are met.
//!
//! Until the row math lands the call sites stay on CPU; the scaffolding is
//! the foundation that lets each subsequent milestone touch *only* the
//! device-side body of the relevant entry point without redesigning the
//! host glue.

use std::sync::OnceLock;

use ndarray::Array2;

use super::error::GpuError;
use super::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

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
        BmsFlexGpuBackend::compiled(),
        large_enough,
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

/// Per-fit minimal inputs the device row primitive will consume.
///
/// The struct is deliberately *additive*: as subsequent milestones expand
/// what the device kernel needs (cached cell moments, score-warp basis
/// tables, etc.) new optional fields are appended here without breaking
/// existing call sites. Milestone-2 entry points only inspect `n`, `r`,
/// `p` and use the other fields for the early-return shape checks that
/// the future kernels will need anyway.
#[derive(Clone, Copy, Debug)]
pub struct BmsFlexGpuRowInputs<'a> {
    /// Number of observations.
    pub n: usize,
    /// Primary local dimension (q + log-slope + score-warp + link-wiggle).
    /// Issue #210 pins `r = 20` for biobank shape; the kernel will be
    /// generic over `r` once flex math is on device.
    pub r: usize,
    /// Total joint-parameter dimension `p` (sum of all block sizes).
    pub p: usize,
    /// Current β coefficient vector, length `p`, in joint-block order.
    pub beta: &'a [f64],
    /// Observed responses `y_i ∈ {0, 1}`, length `n`.
    pub y: &'a [f64],
    /// Observation weights, length `n`.
    pub weights: &'a [f64],
}

impl<'a> BmsFlexGpuRowInputs<'a> {
    /// Shape-check the inputs the way every entry point would before any
    /// device call. Kept on the input struct so it is reused by all three
    /// entry points and the rigid-kernel sibling once it lands.
    fn validate(&self) -> Result<(), GpuError> {
        if self.beta.len() != self.p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex inputs: beta.len()={} != p={}",
                    self.beta.len(),
                    self.p
                ),
            });
        }
        if self.y.len() != self.n {
            return Err(GpuError::DriverCallFailed {
                reason: format!("bms_flex inputs: y.len()={} != n={}", self.y.len(), self.n),
            });
        }
        if self.weights.len() != self.n {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex inputs: weights.len()={} != n={}",
                    self.weights.len(),
                    self.n
                ),
            });
        }
        Ok(())
    }
}

/// The PTX source compiled and loaded at first use of the BMS flex GPU
/// backend. Kept intentionally trivial for milestone 2: a no-op probe
/// kernel that takes no arguments and immediately returns. Exercises the
/// full NVRTC → cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel
/// path so the scaffolding catches host-side issues (PTX cache, arena
/// alloc, stream sync) long before the real row kernel lands.
#[cfg(target_os = "linux")]
const PROBE_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void bms_flex_probe() {
    // Intentionally empty. This kernel exists only so the scaffolding can
    // verify NVRTC compile + module load + launch + synchronize on the
    // selected device. Real kernels land in milestone 3 onwards.
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
    /// NVRTC-compiled module containing the probe kernel (and, in later
    /// milestones, the real BMS flex row kernels). `OnceLock` so the
    /// compile happens exactly once per process and is shared by every
    /// dispatching thread.
    module: OnceLock<Arc<CudaModule>>,
    /// Reusable f64 device buffers keyed by power-of-two element-count
    /// buckets. Held under a `Mutex` because biobank fits dispatch from
    /// multiple rayon worker threads; the mutex is only held during
    /// `alloc` / `release`, not across kernel launches.
    arena: Mutex<DeviceArena>,
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct DeviceArena {
    free: std::collections::HashMap<usize, Vec<cudarc::driver::CudaSlice<f64>>>,
}

#[cfg(target_os = "linux")]
impl DeviceArena {
    fn bucket_of(elements: usize) -> usize {
        elements.max(1).next_power_of_two()
    }

    /// Allocate a device slice of at least `elements` f64s. Returns the
    /// bucket size actually allocated so the caller can release into the
    /// same bucket on drop.
    fn alloc(
        &mut self,
        stream: &Arc<CudaStream>,
        elements: usize,
    ) -> Result<(usize, cudarc::driver::CudaSlice<f64>), GpuError> {
        let bucket = Self::bucket_of(elements);
        if let Some(bucket_vec) = self.free.get_mut(&bucket)
            && let Some(slot) = bucket_vec.pop()
        {
            return Ok((bucket, slot));
        }
        let fresh = stream
            .alloc_zeros::<f64>(bucket)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex arena alloc_zeros<{bucket}>: {err}"),
            })?;
        Ok((bucket, fresh))
    }

    fn release(&mut self, bucket: usize, slab: cudarc::driver::CudaSlice<f64>) {
        self.free.entry(bucket).or_default().push(slab);
    }
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
        let runtime = super::runtime::GpuRuntime::global().ok_or_else(|| {
            GpuError::DriverLibraryUnavailable {
                reason: "bms_flex backend: no CUDA runtime available".to_string(),
            }
        })?;
        let ctx = super::runtime::cuda_context_for(runtime.selected_device().ordinal).ok_or_else(
            || GpuError::DriverCallFailed {
                reason: format!(
                    "bms_flex backend: failed to create CUDA context for device {}",
                    runtime.selected_device().ordinal
                ),
            },
        )?;
        let stream = ctx.default_stream();
        let backend = BmsFlexGpuBackend {
            inner: BmsFlexGpuContextLinux {
                ctx,
                stream,
                module: OnceLock::new(),
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
        if let Some(existing) = self.inner.module.get() {
            return Ok(existing);
        }
        let ptx = cudarc::nvrtc::compile_ptx(PROBE_KERNEL_SOURCE).map_err(|err| {
            GpuError::DriverCallFailed {
                reason: format!("bms_flex NVRTC compile failed: {err}"),
            }
        })?;
        let module =
            self.inner
                .ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("bms_flex module load failed: {err}"),
                })?;
        self.inner.module.set(module).ok();
        Ok(self
            .inner
            .module
            .get()
            .expect("module slot is populated after set"))
    }

    /// Launch the probe kernel and synchronize. Used by tests and by the
    /// dispatcher's policy gate to verify the full host-orchestration
    /// path before milestone 3 lands real math.
    #[cfg(target_os = "linux")]
    pub fn launch_probe(&self) -> Result<(), GpuError> {
        use cudarc::driver::LaunchConfig;
        let module = self.compile_probe_module()?;
        let func =
            module
                .load_function("bms_flex_probe")
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("bms_flex probe load_function: {err}"),
                })?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let builder = self.inner.stream.launch_builder(&func);
        // SAFETY: probe kernel takes no arguments and does no memory
        // access, so launch parameters and lack of args are trivially
        // valid for any device.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("bms_flex probe launch: {err}"),
        })?;
        self.inner
            .stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex probe synchronize: {err}"),
            })?;
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
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("bms_flex arena mutex poisoned: {err}"),
            })?;
        let (bucket, slab) = guard.alloc(&self.inner.stream, elements)?;
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
// Three entry points. Each currently returns `NotYetImplemented`; the
// dispatcher in `bernoulli_marginal_slope.rs` matches on that variant to
// fall through to the existing CPU path. The signatures are stable: the
// row math (milestone 3) plugs in by replacing only the body below the
// `validate()` call, not by changing the inputs or return types.
// ────────────────────────────────────────────────────────────────────────

/// Evaluate the BMS flex negative log-likelihood and exact gradient on
/// the GPU. Returns `(loglik, gradient)` where `gradient` is the joint-β
/// gradient in joint-block order, length `p`.
pub fn gpu_gradient(inputs: BmsFlexGpuRowInputs<'_>) -> Result<(f64, Vec<f64>), GpuError> {
    inputs.validate()?;
    // Touch the backend so probe failures surface at this entry point
    // rather than only at the dense-H / HVP entry points.
    BmsFlexGpuBackend::probe()?;
    Err(GpuError::NotYetImplemented {
        reason: "bms_flex gpu_gradient: row math not landed yet (issue #210 milestone 3)"
            .to_string(),
    })
}

/// Evaluate the BMS flex joint-Hessian times an input vector `v` on the
/// GPU. Returns `H · v`, length `p`.
pub fn gpu_hessian_matvec(
    inputs: BmsFlexGpuRowInputs<'_>,
    v: &[f64],
) -> Result<Vec<f64>, GpuError> {
    inputs.validate()?;
    if v.len() != inputs.p {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "bms_flex gpu_hessian_matvec: v.len()={} != p={}",
                v.len(),
                inputs.p
            ),
        });
    }
    BmsFlexGpuBackend::probe()?;
    Err(GpuError::NotYetImplemented {
        reason: "bms_flex gpu_hessian_matvec: row math not landed yet (issue #210 milestone 3)"
            .to_string(),
    })
}

/// Assemble the dense BMS flex joint Hessian on the GPU. Returns a
/// `p × p` row-major matrix.
pub fn gpu_hessian_dense(inputs: BmsFlexGpuRowInputs<'_>) -> Result<Array2<f64>, GpuError> {
    inputs.validate()?;
    BmsFlexGpuBackend::probe()?;
    Err(GpuError::NotYetImplemented {
        reason: "bms_flex gpu_hessian_dense: row math not landed yet (issue #210 milestone 3)"
            .to_string(),
    })
}

// ────────────────────────────────────────────────────────────────────────
// Tests. Run via `cargo test -p gam bms_flex_gpu -- --nocapture`.
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod bms_flex_gpu_tests {
    use super::*;

    fn make_inputs<'a>(
        n: usize,
        p: usize,
        beta: &'a [f64],
        y: &'a [f64],
        w: &'a [f64],
    ) -> BmsFlexGpuRowInputs<'a> {
        BmsFlexGpuRowInputs {
            n,
            r: 20,
            p,
            beta,
            y,
            weights: w,
        }
    }

    #[test]
    fn bms_flex_gpu_policy_decision_is_explicit() {
        let decision = row_primary_hessian_decision(50_000, 4);
        assert_eq!(decision.kernel, GpuKernel::MarginalSlopeRows);
    }

    #[test]
    fn bms_flex_gpu_gradient_returns_not_yet_implemented_until_kernel_lands() {
        let p = 4;
        let n = 8;
        let beta = vec![0.1, -0.2, 0.05, 0.0];
        let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = vec![1.0; n];
        let inputs = make_inputs(n, p, &beta, &y, &w);
        match gpu_gradient(inputs) {
            // Either we hit the sentinel (kernel not landed) or we hit
            // a clean driver failure (CPU-only host with no CUDA). Both
            // are valid milestone-2 outcomes — the test guards against
            // any *other* failure mode (panic, success with bogus
            // values, wrong error variant).
            Err(GpuError::NotYetImplemented { .. })
            | Err(GpuError::DriverLibraryUnavailable { .. })
            | Err(GpuError::DriverCallFailed { .. }) => {}
            Err(other) => panic!("unexpected GpuError variant: {other:?}"),
            Ok((loglik, grad)) => panic!(
                "gpu_gradient unexpectedly returned Ok(loglik={loglik}, grad.len()={})",
                grad.len()
            ),
        }
    }

    #[test]
    fn bms_flex_gpu_hessian_matvec_rejects_wrong_v_length() {
        let p = 4;
        let n = 4;
        let beta = vec![0.0; p];
        let y = vec![0.0; n];
        let w = vec![1.0; n];
        let inputs = make_inputs(n, p, &beta, &y, &w);
        let v_wrong = vec![0.0; p + 1];
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
    fn bms_flex_gpu_hessian_dense_returns_not_yet_implemented_or_clean_error() {
        let p = 3;
        let n = 4;
        let beta = vec![0.1, 0.2, -0.3];
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let w = vec![1.0; n];
        let inputs = make_inputs(n, p, &beta, &y, &w);
        match gpu_hessian_dense(inputs) {
            Err(GpuError::NotYetImplemented { .. })
            | Err(GpuError::DriverLibraryUnavailable { .. })
            | Err(GpuError::DriverCallFailed { .. }) => {}
            Err(other) => panic!("unexpected GpuError variant: {other:?}"),
            Ok(h) => panic!(
                "gpu_hessian_dense unexpectedly returned Ok(p={}, shape={:?})",
                p,
                h.shape()
            ),
        }
    }

    #[test]
    fn bms_flex_gpu_inputs_validate_catches_shape_mismatches() {
        let p = 3;
        let n = 4;
        let beta = vec![0.0; p + 1];
        let y = vec![0.0; n];
        let w = vec![1.0; n];
        let bad = BmsFlexGpuRowInputs {
            n,
            r: 20,
            p,
            beta: &beta,
            y: &y,
            weights: &w,
        };
        let err = bad.validate().expect_err("beta length mismatch must fail");
        assert!(
            matches!(err, GpuError::DriverCallFailed { .. }),
            "expected DriverCallFailed, got {err:?}"
        );
    }

    /// V100-only: probe the backend end-to-end (CUDA context create, NVRTC
    /// compile, module load, launch, sync). Skipped on hosts without a
    /// usable device so the test still passes on the CI/mac builders.
    #[test]
    fn bms_flex_gpu_context_initialises_when_device_present() {
        let Some(_runtime) = super::super::runtime::GpuRuntime::global() else {
            eprintln!(
                "[bms_flex_gpu test] no CUDA runtime — skipping device-side init smoketest"
            );
            return;
        };
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
