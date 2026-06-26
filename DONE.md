gam-linalg extraction complete.

- Moved `src/linalg/**` into new workspace crate `crates/gam-linalg`.
- Inverted GPU dispatch with `gam_linalg::gpu_hook::GpuGemmDispatch`; `gam` registers `CudaGemmDispatch` from `init_parallelism()`.
- Moved `row_reduction_chunk_rows`/`row_reduction_chunk_count` into `gam_linalg::parallel`; `gam::solver::parallel_strategy` re-exports them.
- Relocated `ChunkedKernelDesignOperator` and `SpatialKernelEvaluator` up to `src/terms/chunked_kernel_design.rs` because they depend on `Gauge`.
- Moved linalg-owned `RidgePolicy`/`RidgeDeterminantMode` into `gam-linalg` and re-exported them from `gam::types`.
- Moved the nested-parallel faer guard into `gam_linalg::faer_ndarray`; `gam-problem` now re-exports that shared guard.
- `gam-linalg` dependencies: `dyn-stack`, `faer`, `gam-runtime`, `log`, `ndarray`, `num-traits`, `rayon`, `serde`, `thiserror`, `wide`.

Verification:

`export PATH=/Users/user/.cargo/bin:$PATH; export SCCACHE_DIR=/Users/user/.sccache-shared RUSTC_WRAPPER=/opt/homebrew/bin/sccache RUSTFLAGS="-A warnings"; cargo check --workspace`

Result: `Finished dev profile [unoptimized + debuginfo] target(s) in 2m 17s`.
