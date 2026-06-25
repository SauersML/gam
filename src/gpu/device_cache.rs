//! Shared host-side scaffolding for every cudarc-backed module under
//! `src/gpu/*` and `src/solver/gpu/*`.
//!
//! Before this module existed, each device backend (`bms_flex`,
//! `survival_flex`, `polya_gamma`, `reml_trace`, ...) carried its own
//! near-identical copy of two patterns:
//!
//!   1. A power-of-two bucketed free list of reusable f64 device slices
//!      (the per-backend `DeviceArena`).
//!   2. A `OnceLock<Result<{module: Arc<CudaModule>}, GpuError>>` that
//!      NVRTC-compiled one source string the first time the backend
//!      dispatched and cached the resulting module for the process lifetime.
//!
//! Both are now provided here so every cudarc backend points at the same
//! implementation. The migration is atomic: no per-backend `DeviceArena`
//! type, no per-backend ad-hoc OnceLock, no transitional shim.

#[cfg(target_os = "linux")]
pub use linux::{DeviceArena, PtxModuleCache, compile_ptx_arch};

#[cfg(target_os = "linux")]
mod linux {
    use super::super::gpu_error::GpuError;
    use crate::gpu::gpu_error::GpuResultExt;
    use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream};
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    /// Power-of-two bucketed free list of f64 device slices.
    ///
    /// Allocations round the requested element count up to the next
    /// `usize::next_power_of_two`. On drop the slab is handed back to the
    /// arena under the same bucket via [`DeviceArena::release`]. Held under
    /// a `Mutex` by every backend that uses it because large-scale fits
    /// dispatch from multiple rayon workers; the mutex is only held during
    /// `alloc` / `release`, never across kernel launches.
    #[derive(Default)]
    pub struct DeviceArena {
        free: HashMap<usize, Vec<CudaSlice<f64>>>,
    }

    impl DeviceArena {
        #[inline]
        pub fn bucket_of(elements: usize) -> usize {
            elements.max(1).next_power_of_two()
        }

        /// Allocate a device slice of at least `elements` f64s. Returns the
        /// bucket size actually allocated so the caller can release into the
        /// same bucket on drop. `label` is woven into the error message if
        /// the underlying `alloc_zeros` fails so failures stay attributable
        /// to the originating backend (matching the pre-extraction wording).
        pub fn alloc(
            &mut self,
            stream: &Arc<CudaStream>,
            elements: usize,
            label: &'static str,
        ) -> Result<(usize, CudaSlice<f64>), GpuError> {
            let bucket = Self::bucket_of(elements);
            if let Some(bucket_vec) = self.free.get_mut(&bucket)
                && let Some(slot) = bucket_vec.pop()
            {
                return Ok((bucket, slot));
            }
            let fresh = stream
                .alloc_zeros::<f64>(bucket)
                .gpu_ctx_with(|err| format!("{label} arena alloc_zeros<{bucket}>: {err}"))?;
            Ok((bucket, fresh))
        }

        pub fn release(&mut self, bucket: usize, slab: CudaSlice<f64>) {
            self.free.entry(bucket).or_default().push(slab);
        }
    }

    /// Process-wide NVRTC module cache for a single PTX source string.
    ///
    /// The first call to [`PtxModuleCache::get_or_compile`] compiles the
    /// source via `cudarc::nvrtc::compile_ptx`, loads the module on the
    /// supplied context, and stores the resulting `Arc<CudaModule>`.
    /// Subsequent calls return the cached module without recompiling.
    ///
    /// The `label` is woven into the error message so the originating
    /// backend stays identifiable in logs; the wording matches each
    /// caller's previous bespoke `format!` so existing log assertions
    /// continue to hold.
    #[derive(Default)]
    pub struct PtxModuleCache {
        module: std::sync::OnceLock<Arc<CudaModule>>,
    }

    impl PtxModuleCache {
        pub const fn new() -> Self {
            Self {
                module: std::sync::OnceLock::new(),
            }
        }

        pub fn get(&self) -> Option<&Arc<CudaModule>> {
            self.module.get()
        }

        /// Compile `source` and load it on `ctx` the first time; return
        /// the cached `Arc<CudaModule>` on every subsequent call.
        pub fn get_or_compile(
            &self,
            ctx: &Arc<CudaContext>,
            label: &'static str,
            source: &str,
        ) -> Result<&Arc<CudaModule>, GpuError> {
            if let Some(existing) = self.module.get() {
                return Ok(existing);
            }
            let ptx = compile_ptx_with_opts(source, nvrtc_compile_options())
                .gpu_ctx_with(|err| format!("{label} NVRTC compile failed: {err}"))?;
            let module = ctx
                .load_module(ptx)
                .gpu_ctx_with(|err| format!("{label} module load failed: {err}"))?;
            self.module.set(module).ok();
            Ok(self
                .module
                .get()
                .expect("module slot populated immediately after set"))
        }
    }

    /// Compile a kernel source string to PTX with the SAME device-keyed NVRTC
    /// options [`PtxModuleCache::get_or_compile`] uses — crucially the
    /// `--gpu-architecture` pin (#1551), without which NVRTC defaults below
    /// `sm_60` and rejects `atomicAdd(double*, double)`. Call sites that compile
    /// via the bare `cudarc::nvrtc::compile_ptx` (no options) MUST route through
    /// this instead when their kernel uses double atomics, or the device path
    /// silently falls back to the CPU.
    pub fn compile_ptx_arch<S: AsRef<str>>(source: S) -> Result<cudarc::nvrtc::Ptx, GpuError> {
        compile_ptx_with_opts(source.as_ref(), nvrtc_compile_options())
            .gpu_ctx_with(|err| std::format!("NVRTC compile failed: {err}"))
    }

    fn nvrtc_compile_options() -> CompileOptions {
        let mut opts = CompileOptions::default();
        opts.include_paths = nvrtc_include_paths();
        // #1551: pin the NVRTC virtual arch to the selected device's compute
        // capability. Without it NVRTC defaults below sm_60, where the
        // `atomicAdd(double*, double)` overload is absent — so kernels using
        // double atomics (the SAE arrow/Schur PCG kernels) fail to compile and
        // the device path silently falls back to the CPU (SAE ran at 0% GPU).
        // `arch` is `Option<&'static str>`; `nvrtc_arch()` returns a static
        // `compute_NN` for the device's real capability.
        if let Some(runtime) = crate::gpu::device_runtime::GpuRuntime::global() {
            opts.arch = Some(runtime.selected_device().capability.nvrtc_arch());
        }
        opts
    }

    fn nvrtc_include_paths() -> Vec<String> {
        let mut paths = Vec::new();
        push_existing_include_path(&mut paths, Path::new("/usr/local/cuda/include"));
        push_existing_include_path(&mut paths, Path::new("/usr/include"));
        push_existing_include_path(&mut paths, Path::new("/usr/include/x86_64-linux-gnu"));
        push_gcc_include_paths(&mut paths, Path::new("/usr/lib/gcc/x86_64-linux-gnu"));
        paths
    }

    fn push_gcc_include_paths(paths: &mut Vec<String>, root: &Path) {
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            push_existing_include_path(paths, &entry.path().join("include"));
        }
    }

    fn push_existing_include_path(paths: &mut Vec<String>, path: &Path) {
        if !path.is_dir() {
            return;
        }
        let display = path.to_string_lossy().into_owned();
        if !paths.iter().any(|existing| existing == &display) {
            paths.push(display);
        }
    }
}
