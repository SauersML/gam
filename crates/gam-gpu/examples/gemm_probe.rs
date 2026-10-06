//! The f32 products of a vpd4l library step (`Device::gemm` on CUDA: cuBLAS's f32 product with its
//! default algorithm) against every algorithm cuBLASLt's heuristic offers for the same product, on
//! the first CUDA device: whether choosing the product's algorithm per shape would beat the
//! default. One row per shape: the default's time and rate, the fastest cuBLASLt algorithm's, and
//! the heuristic's first choice's. CUDA only.
//!
//! Usage: gemm_probe [REPEATS]

#[cfg(target_os = "linux")]
mod probe {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
    use cudarc::cublaslt::{result as lt, sys as ltsys};
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use std::sync::Arc;
    use std::time::Instant;

    /// One product `C (m × n) = op(A) op(B)`, column-major as cuBLAS reads it, `k` the inner
    /// dimension, and what it is in the step.
    struct Shape {
        name: &'static str,
        ta: bool,
        tb: bool,
        m: usize,
        n: usize,
        k: usize,
    }

    const SHAPES: &[Shape] = &[
        Shape { name: "head logits chunk (count x rows, k = width)", ta: true, tb: false, m: 2688, n: 3072, k: 768 },
        Shape { name: "head logits chunk, more rows", ta: true, tb: false, m: 1280, n: 6144, k: 768 },
        Shape { name: "head expected rows (width x rows, k = count)", ta: false, tb: false, m: 768, n: 3072, k: 2688 },
        Shape { name: "attention projection (width x rows)", ta: true, tb: false, m: 768, n: 8192, k: 768 },
        Shape { name: "stacked q, k, v projections", ta: true, tb: false, m: 2304, n: 8192, k: 768 },
        Shape { name: "MLP up (hidden x rows)", ta: true, tb: false, m: 3072, n: 8192, k: 768 },
        Shape { name: "MLP down (width x rows, k = hidden)", ta: true, tb: false, m: 768, n: 8192, k: 3072 },
        Shape { name: "weight gradient (out x in, k = rows)", ta: false, tb: true, m: 768, n: 768, k: 8192 },
        Shape { name: "MLP weight gradient", ta: false, tb: true, m: 3072, n: 768, k: 8192 },
    ];

    fn filled(stream: &Arc<CudaStream>, n: usize, seed: u64) -> Result<CudaSlice<f32>, String> {
        let mut state = seed | 1;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        stream.clone_htod(&values).map_err(|e| format!("{e:?}"))
    }

    /// Seconds per call of `run`, over `repeats` calls after two warm ones.
    fn timed(stream: &Arc<CudaStream>, repeats: usize, mut run: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
        for _ in 0..2 {
            run()?;
        }
        stream.synchronize().map_err(|e| format!("{e:?}"))?;
        let start = Instant::now();
        for _ in 0..repeats {
            run()?;
        }
        stream.synchronize().map_err(|e| format!("{e:?}"))?;
        Ok(start.elapsed().as_secs_f64() / repeats as f64)
    }

    pub fn main(repeats: usize) -> Result<(), String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("{e:?}"))?;
        let handle = lt::create_handle().map_err(|e| format!("{e:?}"))?;
        let workspace_bytes: usize = 32 << 20;
        let mut workspace = stream.alloc_zeros::<u8>(workspace_bytes).map_err(|e| format!("{e:?}"))?;
        println!("shape\tm\tn\tk\tdefault_us\tdefault_tflops\tbest_lt_us\tbest_lt_tflops\tfirst_lt_us\talgorithms");
        for shape in SHAPES {
            let (rows_a, cols_a) = if shape.ta { (shape.k, shape.m) } else { (shape.m, shape.k) };
            let (rows_b, cols_b) = if shape.tb { (shape.n, shape.k) } else { (shape.k, shape.n) };
            let a = filled(&stream, rows_a * cols_a, 3)?;
            let b = filled(&stream, rows_b * cols_b, 5)?;
            let mut c = stream.alloc_zeros::<f32>(shape.m * shape.n).map_err(|e| format!("{e:?}"))?;
            let op = |t: bool| if t { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
            let config = GemmConfig {
                transa: op(shape.ta),
                transb: op(shape.tb),
                m: shape.m as i32,
                n: shape.n as i32,
                k: shape.k as i32,
                alpha: 1.0f32,
                lda: rows_a as i32,
                ldb: rows_b as i32,
                beta: 0.0f32,
                ldc: shape.m as i32,
            };
            // SAFETY: the buffers hold op(A) m × k, op(B) k × n and C m × n with these leading
            // dimensions.
            let default = timed(&stream, repeats, || unsafe { blas.gemm(config, &a, &b, &mut c) }.map_err(|e| format!("{e:?}")))?;
            // The same product through cuBLASLt, every algorithm its heuristic returns.
            let desc = lt::create_matmul_desc(ltsys::cublasComputeType_t::CUBLAS_COMPUTE_32F, ltsys::cudaDataType::CUDA_R_32F).map_err(|e| format!("{e:?}"))?;
            for (attribute, t) in [(ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, shape.ta), (ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, shape.tb)] {
                let value = op(t);
                // SAFETY: the attribute is a cublasOperation_t read from a live value of that size.
                unsafe { lt::set_matmul_desc_attribute(desc, attribute, (&value as *const cublasOperation_t).cast(), std::mem::size_of_val(&value)) }.map_err(|e| format!("{e:?}"))?;
            }
            let layout = |rows: usize, cols: usize| lt::create_matrix_layout(ltsys::cudaDataType::CUDA_R_32F, rows as u64, cols as u64, rows as i64).map_err(|e| format!("{e:?}"));
            let (la, lb, lc) = (layout(rows_a, cols_a)?, layout(rows_b, cols_b)?, layout(shape.m, shape.n)?);
            let preference = lt::create_matmul_pref().map_err(|e| format!("{e:?}"))?;
            let bytes = workspace_bytes as u64;
            // SAFETY: the attribute is a u64 read from a live value.
            unsafe { lt::set_matmul_pref_attribute(preference, ltsys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, (&bytes as *const u64).cast(), 8) }.map_err(|e| format!("{e:?}"))?;
            let mut results: Vec<ltsys::cublasLtMatmulHeuristicResult_t> = Vec::with_capacity(32);
            let mut count = 0i32;
            // SAFETY: room for 32 results; `count` is how many the call wrote.
            unsafe {
                ltsys::cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, preference, 32, results.as_mut_ptr(), &mut count).result().map_err(|e| format!("{e:?}"))?;
                results.set_len(count.max(0) as usize);
            }
            let (alpha, beta) = (1.0f32, 0.0f32);
            let mut times = Vec::new();
            for result in &results {
                if result.state != ltsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    continue;
                }
                let algo = result.algo;
                let seconds = timed(&stream, repeats, || {
                    let (pa, _ra) = a.device_ptr(&stream);
                    let (pb, _rb) = b.device_ptr(&stream);
                    let (pc, _rc) = c.device_ptr_mut(&stream);
                    let (pw, _rw) = workspace.device_ptr_mut(&stream);
                    // SAFETY: the layouts describe the live buffers; the algorithm came from the
                    // heuristic for these descriptors; the workspace holds `workspace_bytes`.
                    unsafe {
                        lt::matmul(
                            handle, desc, (&alpha as *const f32).cast(), (&beta as *const f32).cast(),
                            pa as *const _, la, pb as *const _, lb, pc as *const _, lc, pc as *mut _, lc,
                            &algo, pw as *mut _, workspace_bytes, stream.cu_stream() as ltsys::cudaStream_t,
                        )
                    }
                    .map_err(|e| format!("{e:?}"))
                });
                if let Ok(seconds) = seconds {
                    times.push(seconds);
                }
            }
            let flops = 2.0 * (shape.m * shape.n * shape.k) as f64;
            let best = times.iter().copied().fold(f64::INFINITY, f64::min);
            let first = times.first().copied().unwrap_or(f64::NAN);
            println!(
                "{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{}",
                shape.name, shape.m, shape.n, shape.k,
                default * 1e6, flops / default / 1e12, best * 1e6, flops / best / 1e12, first * 1e6, times.len()
            );
            // SAFETY: each descriptor is destroyed once, after its last use.
            unsafe {
                lt::destroy_matmul_pref(preference).map_err(|e| format!("{e:?}"))?;
                for l in [la, lb, lc] {
                    lt::destroy_matrix_layout(l).map_err(|e| format!("{e:?}"))?;
                }
                lt::destroy_matmul_desc(desc).map_err(|e| format!("{e:?}"))?;
            }
        }
        // SAFETY: the handle is destroyed once, after its last use.
        unsafe { lt::destroy_handle(handle) }.map_err(|e| format!("{e:?}"))?;
        Ok(())
    }
}

fn main() -> Result<(), String> {
    #[cfg(target_os = "linux")]
    return probe::main(std::env::args().nth(1).and_then(|r| r.parse().ok()).unwrap_or(20));
    #[cfg(not(target_os = "linux"))]
    Err("gemm_probe: CUDA only".to_string())
}
