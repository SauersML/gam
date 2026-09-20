//! Bernoulli marginal-slope FLEX GPU policy and backend probe.

use gam_gpu::gpu_error::GpuError;
use gam_gpu::{
    GpuDecision, RowKernelAdmission, RowKernelShape, RuntimeDeviceProbe, decide_row_kernel,
};

use crate::bms::LatentIntegral;

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaModule;

/// The model a flexible BMS fit asks its row kernel to evaluate: how its rows
/// integrate over the latent law, and which blocks carry row primaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BmsFlexRowModel {
    pub(crate) latent_integral: LatentIntegral,
    pub(crate) score_warp: bool,
    pub(crate) link_deviation: bool,
    pub(crate) residual_repair: bool,
}

/// What a BMS FLEX row kernel computes exactly. Selection reads only this
/// declaration: a kernel is a candidate for a model iff it declares every
/// capability the model needs (gam#3000).
pub(crate) struct BmsFlexRowKernelCapability {
    pub(crate) latent_integrals: &'static [LatentIntegral],
    pub(crate) score_warp: bool,
    pub(crate) link_deviation: bool,
    pub(crate) residual_repair: bool,
}

/// The device row kernel (`crate::bms::gpu::row`) transcribes the Gaussian
/// cell-moment branch of `lower_bms_flex_row_order2_from_parts` field for
/// field, with score-warp and link-deviation blocks of any width. It has no
/// discrete-grid lowering and no residual repair rows.
pub(crate) const BMS_FLEX_ROW_KERNEL_CAPABILITY: BmsFlexRowKernelCapability =
    BmsFlexRowKernelCapability {
        latent_integrals: &[LatentIntegral::GaussianCellMoments],
        score_warp: true,
        link_deviation: true,
        residual_repair: false,
    };

impl BmsFlexRowKernelCapability {
    /// The first capability `model` needs that this kernel does not declare,
    /// or `None` when the kernel computes `model`'s row quantity.
    pub(crate) fn missing_for(&self, model: &BmsFlexRowModel) -> Option<&'static str> {
        if !self.latent_integrals.contains(&model.latent_integral) {
            return Some(model.latent_integral.capability());
        }
        if model.score_warp && !self.score_warp {
            return Some("the score-warp flex block");
        }
        if model.link_deviation && !self.link_deviation {
            return Some("the link-deviation flex block");
        }
        if model.residual_repair && !self.residual_repair {
            return Some("the residual repair block");
        }
        None
    }
}

/// Decide which kernel builds the row-primary Hessian for `model` on `shape`,
/// through the one row-kernel decision: the device kernel is eligible only for
/// a model it declares, then only when compiled in, and under `auto` its own
/// timings of `shape` pick the executor, or `shape` races (gam#3024).
/// `shape.kernel` names the device executor the designs admit: resident with
/// dense designs, or a host pin.
pub(crate) fn row_primary_hessian_decision(
    model: &BmsFlexRowModel,
    shape: RowKernelShape,
) -> Result<GpuDecision, GpuError> {
    decide_row_kernel(
        gam_gpu::global_policy(),
        RowKernelAdmission {
            missing_capability: BMS_FLEX_ROW_KERNEL_CAPABILITY.missing_for(model),
            compiled: BmsFlexGpuBackend::compiled(),
            shape,
        },
        &mut RuntimeDeviceProbe,
    )
}

/// Same as [`row_primary_hessian_decision`] but turns `gpu=required` for a
/// kernel that cannot run `model` into an `Err` string at the call site.
pub(crate) fn require_row_primary_hessian_supported(
    model: &BmsFlexRowModel,
    shape: RowKernelShape,
) -> Result<GpuDecision, String> {
    let decision = row_primary_hessian_decision(model, shape).map_err(String::from)?;
    decision.clone().log();
    decision.require_supported()?;
    Ok(decision)
}

/// Preserve the selected-GPU execution contract for every downstream
/// consumer. Once policy has produced device-resident BMS FLEX state, a CUDA
/// failure is an execution error; callers must not reinterpret it as permission
/// to run a different CPU algorithm.
// Its production callers compile under `cfg(target_os = "linux")` (the CUDA
// path); off-Linux the lib target has no caller and `-D dead-code` rejects it,
// the break that has been failing the macOS and Windows wheel jobs. Gate to the
// platforms that own the callers rather than suppressing the lint; the fixtures
// that exercise it are gated to Linux alongside it.
#[cfg(target_os = "linux")]
pub(crate) fn require_selected_gpu_result<T>(
    operation: &str,
    result: Result<T, GpuError>,
) -> Result<T, String> {
    result.map_err(|error| format!("BMS FLEX selected GPU {operation} failed: {error}"))
}

/// The PTX source compiled and loaded at first use of the BMS flex GPU
/// backend. The probe kernel exercises the full NVRTC → cuModuleLoadData
/// → cuModuleGetFunction → cuLaunchKernel path so the scaffolding catches
/// host-side issues (PTX cache, arena alloc, stream sync) before the real
/// row kernel is dispatched by the row-primary cache builder.
#[cfg(target_os = "linux")]
pub(crate) const PROBE_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void bms_flex_probe() {
    // Intentionally empty. This kernel exists only so the scaffolding can
    // verify NVRTC compile + module load + launch + synchronize on the
    // selected device. The real row math lives in the bms_flex_row module.
}
"#;

/// Process-wide BMS-flex GPU backend. Lazy-initialised on first call to
/// [`BmsFlexGpuBackend::probe`].
#[must_use]
pub(crate) struct BmsFlexGpuBackend {
    #[cfg(target_os = "linux")]
    pub(crate) inner: gam_gpu::backend_probe::CudaBackendContext,
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
        #[cfg(target_os = "linux")]
        {
            static BACKEND: gam_gpu::backend_probe::CachedBackend<BmsFlexGpuBackend> =
                gam_gpu::backend_probe::CachedBackend::new();
            BACKEND.get_or_probe("bms_flex", |parts| {
                let backend = BmsFlexGpuBackend {
                    inner: gam_gpu::backend_probe::CudaBackendContext::from_parts(parts),
                };
                // Eagerly compile the probe kernel so any NVRTC failure surfaces
                // here, not at first dispatch.
                backend.compile_probe_module()?;
                Ok(backend)
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(GpuError::DriverLibraryUnavailable {
                reason: "bms_flex GPU backend is Linux-only".to_string(),
            })
        }
    }

    /// NVRTC-compile (or fetch from cache) the probe module.
    #[cfg(target_os = "linux")]
    pub(crate) fn compile_probe_module(&self) -> Result<&Arc<CudaModule>, GpuError> {
        self.inner
            .module
            .get_or_compile(&self.inner.ctx, "bms_flex", PROBE_KERNEL_SOURCE)
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
// Tests. Run via `cargo test -p gam bms_flex_gpu -- --nocapture`.
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod bms_flex_gpu_tests {
    use super::*;
    use gam_gpu::GpuKernel;

    fn full_flex_model(latent_integral: LatentIntegral) -> BmsFlexRowModel {
        BmsFlexRowModel {
            latent_integral,
            score_warp: true,
            link_deviation: true,
            residual_repair: false,
        }
    }

    fn resident_shape(rows: usize) -> RowKernelShape {
        RowKernelShape {
            kernel: GpuKernel::MarginalSlopeRows,
            rows,
            widths: [20, 12, 8, 24],
            threads: 1,
        }
    }

    #[test]
    pub(crate) fn bms_flex_gpu_policy_decision_is_explicit() {
        let decision = row_primary_hessian_decision(
            &full_flex_model(LatentIntegral::GaussianCellMoments),
            resident_shape(50_000),
        )
        .expect("GPU policy resolution must be lossless");
        assert_eq!(decision.kernel, GpuKernel::MarginalSlopeRows);
        assert_eq!(decision.missing_capability, None);
    }

    /// gam#3000: the device row kernel declares the Gaussian cell-moment
    /// integral with both flex blocks, and nothing else.
    #[test]
    fn row_kernel_capability_covers_exactly_the_gaussian_cell_integral_3000() {
        let capability = &BMS_FLEX_ROW_KERNEL_CAPABILITY;
        assert_eq!(
            capability.missing_for(&full_flex_model(LatentIntegral::GaussianCellMoments)),
            None
        );
        for (score_warp, link_deviation) in [(true, false), (false, true)] {
            let model = BmsFlexRowModel {
                score_warp,
                link_deviation,
                ..full_flex_model(LatentIntegral::GaussianCellMoments)
            };
            assert_eq!(capability.missing_for(&model), None);
        }
        assert_eq!(
            capability.missing_for(&full_flex_model(LatentIntegral::DiscreteGrid)),
            Some(LatentIntegral::DiscreteGrid.capability())
        );
        let residual = BmsFlexRowModel {
            residual_repair: true,
            ..full_flex_model(LatentIntegral::GaussianCellMoments)
        };
        assert_eq!(
            capability.missing_for(&residual),
            Some("the residual repair block")
        );
    }

    /// gam#3000: an empirical-law FLEX model is never handed to the device row
    /// kernel, at any row count and under whichever policy this process runs,
    /// and the decision names the capability it lacks. On a CUDA host above the
    /// old borrowed row threshold this is the decision `auto` used to get wrong.
    #[test]
    fn empirical_law_flex_model_selects_the_cpu_row_kernel_3000() {
        for n in [0, 50_000, 10_000_000] {
            let decision = row_primary_hessian_decision(
                &full_flex_model(LatentIntegral::DiscreteGrid),
                resident_shape(n),
            )
            .expect("a capability refusal probes no device");
            assert!(!decision.use_gpu, "n={n}: {decision:?}");
            assert_eq!(
                decision.missing_capability,
                Some(LatentIntegral::DiscreteGrid.capability()),
                "n={n}: {decision:?}"
            );
        }
    }

    // Exercises the Linux-only selected-GPU contract helper, so it is gated with
    // it; stacked attributes read as AND.
    #[cfg(target_os = "linux")]
    #[test]
    pub(crate) fn selected_gpu_errors_propagate_without_algorithm_substitution_932() {
        let error = require_selected_gpu_result::<()>(
            "sentinel operation",
            Err(GpuError::DriverCallFailed {
                reason: "sentinel device fault".to_string(),
            }),
        )
        .expect_err("a selected CUDA failure must propagate");
        assert!(error.contains("selected GPU sentinel operation failed"));
        assert!(error.contains("sentinel device fault"));
    }

}
