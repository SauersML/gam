//! Regression test for https://github.com/SauersML/gam/issues/171.
//!
//! On a CPU-only host (no `libcuda.{dylib,so,dll}` reachable via the platform
//! loader), a saved-and-then-loaded model must be predictable without
//! triggering `cudarc::panic_no_lib_found`. The original report showed
//! `Model.predict(df)` panicking inside the Rust
//! boundary on macOS because the dispatch decision reached into cudarc
//! (via `fallback-dynamic-loading`) without first checking for the CUDA
//! driver library outside cudarc.
//!
//! The fix landed in `GpuRuntime::probe`: every cudarc driver entry point
//! is now gated on gam's own `libloading` driver probe returning `true`, and
//! the typed runtime cache preserves an `Absent` outcome so subsequent
//! predict-path dispatch calls take the CPU fast path. This test pins the
//! contract by:
//!   1. exercising typed Auto resolution and the public dispatch
//!      `decide()` decision from the predict-relevant kernels (DenseMatvec,
//!      DenseMatMul, RowReduction);
//!   2. fitting a tiny GAM and running the predict-path design rebuild
//!      + matvec — the same arithmetic `StandardPredictor::predict_plugin_response`
//!      performs — and asserting it completes without panic.
//!
//! On a host with a working CUDA driver the test still passes; the
//! assertions only require that calls do not panic and that the GPU
//! dispatch decision is consistent with typed runtime availability.

