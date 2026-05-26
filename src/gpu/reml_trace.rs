//! GPU Hutchinson stochastic trace estimator for the REML/LAML logdet
//! gradient, per math team block 2 (sections 12–18 of the V100 design).
//!
//! Public entry point: [`evidence_derivatives_hutchinson_gpu`]. For each
//! derivative Hessian `H_j` (`j = 1..D`) and a single penalized Hessian `H`
//! held resident on device, returns the unbiased Hutchinson estimate of
//!
//! ```text
//! t_j = tr(H^{-1} H_j)
//! ```
//!
//! plus the sample standard error of each estimate, computed from `K`
//! Rademacher probe vectors `z_k ∈ {±1}^p` whose entries are drawn from a
//! **stateless SplitMix64 counter hash** (no cuRAND state). The math
//! identity used on device is
//!
//! ```text
//! z^T H^{-1} H_j z  =  z^T H_j w   where   H w = z
//! ```
//!
//! so we factor `H` **once** with `cusolverDnDpotrf`, batch-solve `H W = Z`
//! with **one** `cusolverDnDpotrs` of `nrhs = K`, and then evaluate the
//! quadratic forms with a custom NVRTC reduction kernel. The REML logdet
//! gradient is `g_j = (1/2) · mean_k(q_{j,k})`.
//!
//! Two assembly variants for `H_j` are supported:
//!
//! * **Dense** — caller passes `H_j` as a `p × p` device or host matrix.
//!   GEMM forms `Y_j = H_j W`, then a custom reduction sums
//!   `z_k^T y_{j,k}` per (j, k). Cost: `D` GEMMs of size `p × p × K`.
//! * **Weighted-Gram structural** — caller provides the design `X`
//!   (`n × p`), weight vectors `A_j` (`n`, one per derivative — the
//!   diagonal of the design's row weights that `H_j` adds), and the
//!   per-derivative penalty contribution `Q_pen[j,k]` if any. The kernel
//!   forms `R_Z = X Z` and `R_W = X W` **once** via GEMM and then sums
//!   `sum_i a_j[i] · R_Z[i,k] · R_W[i,k]` per (j, k) without ever
//!   materialising the `p × p` `H_j` matrix. Cost: 2 GEMMs of size
//!   `n × p × K` shared across all `D` derivatives.
//!
//! The structural path is the high-value route for biobank-scale models
//! where `p` is hundreds and there are many derivatives.
//!
//! # Stateless probe RNG
//!
//! The probe entries are produced on device by a SplitMix64 finalizer over
//! `(seed, probe_index k, coordinate i)`. This has three consequences:
//!
//! 1. No cuRAND state — the kernel is fully stateless, threads write into
//!    `Z[i + k·p]` independently.
//! 2. **Common random numbers**: the first `K1` probes of a run with
//!    `K2 > K1` are bit-identical to a `K = K1` run with the same seed.
//!    This is the property that lets the adaptive `K` schedule build on
//!    earlier probes without re-running them, and lets CPU and GPU
//!    implementations of Hutchinson compare estimator-by-estimator (the
//!    same probes produce the same `q_{j,k}` to round-off).
//! 3. Reproducibility — a probe at `(seed, k, i)` is the same call after
//!    call regardless of how the grid was scheduled.
//!
//! # Gating
//!
//! The companion helper [`should_use_gpu_hutchinson`] mirrors the CPU
//! gate (`prefers_stochastic_trace_estimation` + matching kernel +
//! plain-SPD logdet path) and adds the GPU-specific minima from the math
//! team's section 18:
//!
//! * `p ≥ 512`
//! * `K ∈ [8, 128]`
//! * Hessian and design held resident or about to be uploaded
//! * The projected penalty-subspace trace is **inactive** (otherwise the
//!   CPU path projects through the IFT kernel — that route is required
//!   for marginal-slope ρ-saturated rows)

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::error::GpuError;

// ────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────

/// Stateless seed for the SplitMix64 Rademacher probe RNG.
#[derive(Clone, Copy, Debug)]
pub struct ProbeSeed(pub u64);

impl Default for ProbeSeed {
    fn default() -> Self {
        // Matches the CPU default seed (`StochasticTraceConfig::default()`)
        // so cross-implementation parity tests can use a shared constant.
        Self(0xCAFE_BABE)
    }
}

/// Description of one derivative-Hessian contribution `H_j`.
///
/// The estimator needs `H_j` only via the quadratic form `z^T H_j w`, so we
/// describe `H_j` *structurally* rather than as a dense matrix. The dense
/// case is recovered by the [`DerivativeHessian::Dense`] variant.
#[derive(Clone, Debug)]
pub enum DerivativeHessian<'a> {
    /// `H_j` is a `p × p` symmetric matrix. The reducer forms `Y = H_j W`
    /// via GEMM and then sums `z_k^T y_k`.
    Dense(ArrayView2<'a, f64>),
    /// `H_j = X^T diag(a_j) X + P_j`, where `a_j` is an `n`-vector of row
    /// weights and `P_j` is an optional `p × p` direct penalty contribution
    /// that is *added* to the structural part. The reducer evaluates
    /// `z^T H_j w  =  sum_i a_j[i] · (X z)[i] · (X w)[i]  +  z^T P_j w`
    /// without materialising the `p × p` `H_j`.
    WeightedGram {
        row_weights: ArrayView1<'a, f64>,
        penalty_extra: Option<ArrayView2<'a, f64>>,
    },
}

impl DerivativeHessian<'_> {
    fn dim_p(&self, expected_p: usize, expected_n: usize) -> Result<(), GpuError> {
        match self {
            DerivativeHessian::Dense(matrix) => {
                if matrix.nrows() != expected_p || matrix.ncols() != expected_p {
                    return Err(GpuError::DriverCallFailed {
                        reason: format!(
                            "reml_trace dense H_j: shape {:?} != ({expected_p}, {expected_p})",
                            matrix.dim()
                        ),
                    });
                }
            }
            DerivativeHessian::WeightedGram {
                row_weights,
                penalty_extra,
            } => {
                if row_weights.len() != expected_n {
                    return Err(GpuError::DriverCallFailed {
                        reason: format!(
                            "reml_trace structural H_j: row_weights.len()={} != n={expected_n}",
                            row_weights.len()
                        ),
                    });
                }
                if let Some(p_extra) = penalty_extra
                    && (p_extra.nrows() != expected_p || p_extra.ncols() != expected_p)
                {
                    return Err(GpuError::DriverCallFailed {
                        reason: format!(
                            "reml_trace structural H_j penalty_extra: shape {:?} != ({expected_p}, {expected_p})",
                            p_extra.dim()
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Inputs to [`evidence_derivatives_hutchinson_gpu`].
#[derive(Clone, Debug)]
pub struct RemlTraceHutchinsonInput<'a> {
    /// Penalized Hessian `H` (`p × p`, SPD).
    pub penalized_hessian: ArrayView2<'a, f64>,
    /// Per-derivative descriptors `H_j`. `D = derivatives.len()`.
    pub derivatives: Vec<DerivativeHessian<'a>>,
    /// Design matrix `X` (`n × p`). Required iff any `H_j` is structural;
    /// `None` is acceptable when **all** derivatives are dense.
    pub design: Option<ArrayView2<'a, f64>>,
    /// Number of probe vectors. Must be ≥ 2 (so a sample SE is defined).
    pub probe_count: usize,
    /// Stateless RNG seed.
    pub seed: ProbeSeed,
}

/// Output of [`evidence_derivatives_hutchinson_gpu`].
#[derive(Clone, Debug)]
pub struct RemlTraceHutchinsonEvidence {
    /// `log |H|` from the cached Cholesky factor (same value the exact GPU
    /// path returns; reusing the factor amortises this).
    pub logdet_hessian: f64,
    /// REML logdet gradient `g_j = (1/2) · mean_k(q_{j,k})`, length `D`.
    pub gradient_rho_logdet: Array1<f64>,
    /// Sample standard error of `g_j` across the `K` probes (already
    /// includes the leading `1/2`), length `D`.
    pub gradient_rho_stderr: Array1<f64>,
    /// `K` probes actually used (matches `input.probe_count`).
    pub probe_count: usize,
}

// ────────────────────────────────────────────────────────────────────────
// Gating
// ────────────────────────────────────────────────────────────────────────

/// Minimum joint-dimension at which the GPU Hutchinson path is enabled.
pub const HUTCHINSON_GPU_MIN_P: usize = 512;
/// Minimum and maximum probe counts the GPU path accepts (math section 18).
pub const HUTCHINSON_GPU_MIN_K: usize = 8;
pub const HUTCHINSON_GPU_MAX_K: usize = 128;
/// Adaptive schedule: initial probe budget.
pub const HUTCHINSON_GPU_K_INITIAL: usize = 16;
/// Adaptive schedule: probe-count step between accuracy checks.
pub const HUTCHINSON_GPU_K_STEP: usize = 8;

/// True when the GPU Hutchinson path is eligible at the current shape and
/// configuration. Caller still has to satisfy the CPU-side gate
/// (`prefers_stochastic_trace_estimation`, matching kernel, plain-SPD
/// logdet, projected penalty subspace **inactive**) — the parameters
/// `prefers_stochastic`, `kernel_matches_hinv`, `plain_spd_logdet`, and
/// `projected_penalty_subspace_active` carry those CPU-side gate booleans
/// into the dispatch decision.
#[must_use]
pub fn should_use_gpu_hutchinson(
    p: usize,
    probe_count: usize,
    prefers_stochastic: bool,
    kernel_matches_hinv: bool,
    plain_spd_logdet: bool,
    projected_penalty_subspace_active: bool,
) -> bool {
    p >= HUTCHINSON_GPU_MIN_P
        && (HUTCHINSON_GPU_MIN_K..=HUTCHINSON_GPU_MAX_K).contains(&probe_count)
        && prefers_stochastic
        && kernel_matches_hinv
        && plain_spd_logdet
        && !projected_penalty_subspace_active
}

// ────────────────────────────────────────────────────────────────────────
// Stateless SplitMix64 Rademacher RNG (host reference; mirrors the NVRTC
// kernel byte-for-byte so CPU and GPU produce identical probes for the
// same `(seed, k, i)`).
// ────────────────────────────────────────────────────────────────────────

/// SplitMix64 finalizer (Sebastiano Vigna, 2015).
#[inline]
pub fn splitmix64_mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Stateless Rademacher entry at probe index `k` (0-based), coordinate
/// `i` (0-based), seed `s`. Returns `+1.0` or `-1.0`.
///
/// The mix is `splitmix64(s ⊕ k·ζ ⊕ i·γ)` for two large odd constants
/// `ζ`, `γ`; the sign bit (bit 63 of the hash) selects the sign. The two
/// constants are *different* from the SplitMix increment so the row and
/// column hashes don't collide on small `(k, i)`.
#[inline]
pub fn rademacher_entry(seed: u64, k: u64, i: u64) -> f64 {
    const ZETA: u64 = 0xD1B5_4A32_D192_ED03;
    const GAMMA: u64 = 0x8CB9_2BA7_2F9D_E81F;
    let composite = seed ^ k.wrapping_mul(ZETA) ^ i.wrapping_mul(GAMMA);
    let h = splitmix64_mix(composite);
    if (h >> 63) == 0 { 1.0 } else { -1.0 }
}

/// Host-side reference: fill a column-major `(p, K)` Rademacher matrix.
/// Used by tests to verify the GPU kernel produces the same bits.
pub fn fill_rademacher_host(seed: ProbeSeed, p: usize, k: usize, out: &mut [f64]) {
    assert_eq!(
        out.len(),
        p * k,
        "fill_rademacher_host: out buffer length {} != p*K = {}*{}",
        out.len(),
        p,
        k
    );
    for col in 0..k {
        for row in 0..p {
            out[col * p + row] = rademacher_entry(seed.0, col as u64, row as u64);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// CPU reference implementation of the Hutchinson estimator
// ────────────────────────────────────────────────────────────────────────
//
// This path is what runs in CPU-only builds and is also what the V100
// parity tests check the device implementation against. It uses the same
// stateless SplitMix probes as the kernel.

/// Run the Hutchinson estimator on CPU using the exact same probe bits
/// the device kernel uses. Returns the same evidence struct.
pub fn evidence_derivatives_hutchinson_cpu(
    input: &RemlTraceHutchinsonInput<'_>,
) -> Result<RemlTraceHutchinsonEvidence, String> {
    validate_inputs(input)?;
    let p = input.penalized_hessian.nrows();
    let d = input.derivatives.len();
    let k = input.probe_count;

    // Cholesky factor of H (lower).
    let h = input.penalized_hessian.to_owned();
    let factor = cholesky_lower(&h)?;
    let logdet_hessian = 2.0 * (0..p).map(|i| factor[[i, i]].ln()).sum::<f64>();

    // Build Z (p, k) column-major in a flat vector.
    let mut z = vec![0.0_f64; p * k];
    fill_rademacher_host(input.seed, p, k, &mut z);

    // Solve H W = Z column by column on CPU (matches what the device
    // does in one batched potrs call).
    let mut w = vec![0.0_f64; p * k];
    for col in 0..k {
        let mut rhs = vec![0.0_f64; p];
        rhs.copy_from_slice(&z[col * p..(col + 1) * p]);
        let solved = solve_cholesky(&factor, &rhs);
        w[col * p..(col + 1) * p].copy_from_slice(&solved);
    }

    // Per-derivative quadratic forms.
    let mut q = vec![0.0_f64; d * k]; // row-major (d, k): q[j*k + m]
    for (j, derivative) in input.derivatives.iter().enumerate() {
        match derivative {
            DerivativeHessian::Dense(matrix) => {
                for col in 0..k {
                    let z_col = &z[col * p..(col + 1) * p];
                    let w_col = &w[col * p..(col + 1) * p];
                    // y = H_j w
                    let mut y = vec![0.0_f64; p];
                    for r in 0..p {
                        let mut acc = 0.0_f64;
                        for c in 0..p {
                            acc += matrix[[r, c]] * w_col[c];
                        }
                        y[r] = acc;
                    }
                    let mut zy = 0.0_f64;
                    for i in 0..p {
                        zy += z_col[i] * y[i];
                    }
                    q[j * k + col] = zy;
                }
            }
            DerivativeHessian::WeightedGram {
                row_weights,
                penalty_extra,
            } => {
                let design = input.design.as_ref().expect("design validated");
                let n = design.nrows();
                for col in 0..k {
                    let z_col = &z[col * p..(col + 1) * p];
                    let w_col = &w[col * p..(col + 1) * p];
                    // r_z = X z (length n), r_w = X w (length n)
                    let mut acc = 0.0_f64;
                    for row in 0..n {
                        let mut rz = 0.0_f64;
                        let mut rw = 0.0_f64;
                        for col_idx in 0..p {
                            rz += design[[row, col_idx]] * z_col[col_idx];
                            rw += design[[row, col_idx]] * w_col[col_idx];
                        }
                        acc += row_weights[row] * rz * rw;
                    }
                    if let Some(pen) = penalty_extra {
                        for r in 0..p {
                            let mut row_acc = 0.0_f64;
                            for c in 0..p {
                                row_acc += pen[[r, c]] * w_col[c];
                            }
                            acc += z_col[r] * row_acc;
                        }
                    }
                    q[j * k + col] = acc;
                }
            }
        }
    }

    let (means, stderrs) = reduce_mean_stderr(&q, d, k);
    let mut gradient_rho_logdet = Array1::<f64>::zeros(d);
    let mut gradient_rho_stderr = Array1::<f64>::zeros(d);
    for j in 0..d {
        gradient_rho_logdet[j] = 0.5 * means[j];
        gradient_rho_stderr[j] = 0.5 * stderrs[j];
    }

    Ok(RemlTraceHutchinsonEvidence {
        logdet_hessian,
        gradient_rho_logdet,
        gradient_rho_stderr,
        probe_count: k,
    })
}

// ────────────────────────────────────────────────────────────────────────
// Public dispatch entry point
// ────────────────────────────────────────────────────────────────────────

/// Compute `log |H|` and the Hutchinson estimate of `(1/2) tr(H^{-1} H_j)`
/// for every derivative. Dispatches to the device-resident path when the
/// CUDA runtime is up and probes the GPU successfully; otherwise runs the
/// CPU reference. Either way the probe bits are identical (stateless
/// SplitMix), so callers see the same estimator value to round-off.
pub fn evidence_derivatives_hutchinson_gpu(
    input: RemlTraceHutchinsonInput<'_>,
) -> Result<RemlTraceHutchinsonEvidence, String> {
    validate_inputs(&input)?;

    #[cfg(target_os = "linux")]
    {
        if super::runtime::GpuRuntime::global().is_some() {
            match linux_cuda::evidence_derivatives(&input) {
                Ok(evidence) => return Ok(evidence),
                Err(GpuError::NotYetImplemented { .. }) => {
                    // Fall through to CPU reference until the device path
                    // is fully landed by milestone 3.
                }
                Err(other) => return Err(String::from(other)),
            }
        }
    }

    evidence_derivatives_hutchinson_cpu(&input)
}

// ────────────────────────────────────────────────────────────────────────
// Linux/CUDA implementation
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux_cuda {
    use super::{
        DerivativeHessian, ProbeSeed, RemlTraceHutchinsonEvidence, RemlTraceHutchinsonInput,
        reduce_mean_stderr,
    };
    use crate::gpu::driver::to_col_major;
    use crate::gpu::error::GpuError;
    use crate::gpu::solver::{
        cholesky_logdet_from_col_major, context_and_stream, pinned_htod, potrf_in_place,
        potrs_in_place,
    };
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cusolver::DnHandle;
    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::{Arc, OnceLock};

    /// NVRTC source for the three custom kernels used by this path. All
    /// arithmetic is in `double` and the layouts are column-major to match
    /// cuBLAS/cuSOLVER conventions.
    ///
    /// * `fill_rademacher_splitmix(seed, p, K, Z)` — stateless ±1 fill.
    /// * `reduce_q_dense(p, K, D, Z, Y_stack, Q)` — `Q[j,k] = z_k^T Y_j[:,k]`
    ///   with `Y_j[:,k] = (H_j W)[:,k]`. `Y_stack` is column-major shape
    ///   `(p, K·D)` with derivative `j` occupying columns `[j·K, (j+1)·K)`.
    /// * `reduce_q_weighted_gram(n, K, D, RZ_stride, RZ, RW, A_stack, Q)`
    ///   — `Q[j,k] = sum_i A[i,j] · RZ[i,k] · RW[i,k]`. Used by the
    ///   structural path. `A_stack` is column-major `(n, D)`.
    ///
    /// The reductions use a per-block warp-shuffle pattern with one block
    /// per `(j, k)` output cell and `THREADS_PER_BLOCK` threads per block.
    pub(super) const PTX_SOURCE: &str = r#"
extern "C" __device__ unsigned long long splitmix64_mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    unsigned long long x = z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void fill_rademacher_splitmix(
    unsigned long long seed,
    unsigned int p,
    unsigned int K,
    double* __restrict__ Z)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = blockIdx.y;
    if (i >= p || k >= K) return;
    const unsigned long long ZETA  = 0xD1B54A32D192ED03ULL;
    const unsigned long long GAMMA = 0x8CB92BA72F9DE81FULL;
    unsigned long long composite =
        seed
        ^ (((unsigned long long)k) * ZETA)
        ^ (((unsigned long long)i) * GAMMA);
    unsigned long long h = splitmix64_mix(composite);
    double v = (h >> 63) == 0 ? 1.0 : -1.0;
    Z[(size_t)k * (size_t)p + (size_t)i] = v;
}

extern "C" __device__ double block_reduce_sum(double v) {
    __shared__ double smem[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    double total = 0.0;
    int n_warps = (blockDim.x + 31) >> 5;
    if (threadIdx.x < (unsigned)n_warps) total = smem[threadIdx.x];
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, off);
        }
    }
    return total;
}

extern "C" __global__ void reduce_q_dense(
    unsigned int p,
    unsigned int K,
    unsigned int D,
    const double* __restrict__ Z,
    const double* __restrict__ Y_stack,
    double* __restrict__ Q)
{
    unsigned int k = blockIdx.x;
    unsigned int j = blockIdx.y;
    if (k >= K || j >= D) return;
    const double* z_col = Z + (size_t)k * (size_t)p;
    const double* y_col = Y_stack + ((size_t)j * (size_t)K + (size_t)k) * (size_t)p;
    double partial = 0.0;
    for (unsigned int i = threadIdx.x; i < p; i += blockDim.x) {
        partial += z_col[i] * y_col[i];
    }
    double total = block_reduce_sum(partial);
    if (threadIdx.x == 0) {
        Q[(size_t)j * (size_t)K + (size_t)k] = total;
    }
}

extern "C" __global__ void reduce_q_weighted_gram(
    unsigned int n,
    unsigned int K,
    unsigned int D,
    const double* __restrict__ RZ,
    const double* __restrict__ RW,
    const double* __restrict__ A_stack,
    double* __restrict__ Q)
{
    unsigned int k = blockIdx.x;
    unsigned int j = blockIdx.y;
    if (k >= K || j >= D) return;
    const double* rz_col = RZ + (size_t)k * (size_t)n;
    const double* rw_col = RW + (size_t)k * (size_t)n;
    const double* a_col  = A_stack + (size_t)j * (size_t)n;
    double partial = 0.0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        partial += a_col[i] * rz_col[i] * rw_col[i];
    }
    double total = block_reduce_sum(partial);
    if (threadIdx.x == 0) {
        Q[(size_t)j * (size_t)K + (size_t)k] = total;
    }
}
"#;

    const THREADS_PER_BLOCK: u32 = 256;

    struct CompiledModule {
        module: Arc<CudaModule>,
    }

    fn module(ctx: &Arc<CudaContext>) -> Result<&'static CompiledModule, GpuError> {
        static MODULE: OnceLock<Result<CompiledModule, GpuError>> = OnceLock::new();
        let result = MODULE.get_or_init(|| {
            let ptx = cudarc::nvrtc::compile_ptx(PTX_SOURCE).map_err(|err| {
                GpuError::DriverCallFailed {
                    reason: format!("reml_trace NVRTC compile failed: {err}"),
                }
            })?;
            let m = ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("reml_trace module load failed: {err}"),
                })?;
            Ok(CompiledModule { module: m })
        });
        result.as_ref().map_err(GpuError::clone)
    }

    pub(super) fn evidence_derivatives(
        input: &RemlTraceHutchinsonInput<'_>,
    ) -> Result<RemlTraceHutchinsonEvidence, GpuError> {
        let p = input.penalized_hessian.nrows();
        let d = input.derivatives.len();
        let k = input.probe_count;
        let (ctx, stream) =
            context_and_stream().map_err(|reason| GpuError::DriverCallFailed { reason })?;
        let solver = DnHandle::new(stream.clone()).map_err(|err| GpuError::DriverCallFailed {
            reason: format!("reml_trace cusolver init: {err}"),
        })?;
        let blas = CudaBlas::new(stream.clone()).map_err(|err| GpuError::DriverCallFailed {
            reason: format!("reml_trace cublas init: {err}"),
        })?;
        let compiled = module(&ctx)?;

        // ── 1. Upload H, factor once.
        let h_col = to_col_major(&input.penalized_hessian);
        let mut h_dev = pinned_htod(&ctx, &stream, &h_col)
            .map_err(|reason| GpuError::DriverCallFailed { reason })?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)
            .map_err(|reason| GpuError::DriverCallFailed { reason })?;
        let factor_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("reml_trace download factor: {err}"),
            })?;
        let logdet_hessian = cholesky_logdet_from_col_major(&factor_col, p);

        // ── 2. Allocate Z (p, K) and fill with Rademacher entries on device.
        let total_z = p.checked_mul(k).ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!("reml_trace Z size overflow: p={p}, K={k}"),
        })?;
        let mut z_dev =
            stream
                .alloc_zeros::<f64>(total_z)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("reml_trace alloc Z: {err}"),
                })?;
        launch_fill_rademacher(&stream, &compiled.module, input.seed, p, k, &mut z_dev)?;

        // ── 3. Solve H W = Z in a single batched potrs call (nrhs = K).
        //     Copy Z into a fresh buffer first; potrs is in-place.
        let mut w_dev =
            stream
                .alloc_zeros::<f64>(total_z)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("reml_trace alloc W: {err}"),
                })?;
        copy_device_slice(&stream, &z_dev, &mut w_dev)?;
        potrs_in_place(&solver, &stream, p, k, &h_dev, &mut w_dev)
            .map_err(|reason| GpuError::DriverCallFailed { reason })?;

        // ── 4. Partition derivatives by kind.
        let mut dense_indices: Vec<usize> = Vec::new();
        let mut gram_indices: Vec<usize> = Vec::new();
        for (j, deriv) in input.derivatives.iter().enumerate() {
            match deriv {
                DerivativeHessian::Dense(_) => dense_indices.push(j),
                DerivativeHessian::WeightedGram { .. } => gram_indices.push(j),
            }
        }

        let mut q_host = vec![0.0_f64; d * k];

        // ── 5a. Dense path: for each dense H_j run a p×p × p×K GEMM and
        //       reduce. We loop over j rather than stacking the H_j's
        //       (would explode memory at biobank-p), but the GEMMs share
        //       the resident W buffer.
        if !dense_indices.is_empty() {
            for &j in &dense_indices {
                let DerivativeHessian::Dense(matrix) = &input.derivatives[j] else {
                    // SAFETY: dense_indices was populated in the partition loop above
                    // with exactly the indices whose variant is DerivativeHessian::Dense.
                    // input.derivatives is immutably borrowed for the whole function so
                    // the slot at index j cannot have been rewritten between partition and
                    // this read; reaching this branch can only mean a future refactor split
                    // the partition from its consumer. The panic names the offending index.
                    panic!(
                        "reml_trace dense path: derivative index {j} is in dense_indices but \
                         input.derivatives[{j}] is not DerivativeHessian::Dense — \
                         dense_indices partition invariant violated"
                    );
                };
                let hj_col = to_col_major(matrix);
                let hj_dev = pinned_htod(&ctx, &stream, &hj_col)
                    .map_err(|reason| GpuError::DriverCallFailed { reason })?;
                let mut y_dev = stream.alloc_zeros::<f64>(total_z).map_err(|err| {
                    GpuError::DriverCallFailed {
                        reason: format!("reml_trace alloc Y_j (j={j}): {err}"),
                    }
                })?;
                gemm_nn(
                    &blas,
                    GemmShape {
                        m: p,
                        n: k,
                        k_inner: p,
                        lda: p,
                        ldb: p,
                        ldc: p,
                    },
                    &hj_dev,
                    &w_dev,
                    &mut y_dev,
                )?;
                let mut q_j_dev =
                    stream
                        .alloc_zeros::<f64>(k)
                        .map_err(|err| GpuError::DriverCallFailed {
                            reason: format!("reml_trace alloc Q_j (j={j}): {err}"),
                        })?;
                launch_reduce_q_dense(
                    &stream,
                    &compiled.module,
                    p,
                    k,
                    1,
                    &z_dev,
                    &y_dev,
                    &mut q_j_dev,
                )?;
                let q_host_j =
                    stream
                        .clone_dtoh(&q_j_dev)
                        .map_err(|err| GpuError::DriverCallFailed {
                            reason: format!("reml_trace download Q_j (j={j}): {err}"),
                        })?;
                q_host[j * k..(j + 1) * k].copy_from_slice(&q_host_j);
            }
        }

        // ── 5b. Structural path: form R_Z = X Z and R_W = X W **once**,
        //       then run reduce_q_weighted_gram for each derivative.
        if !gram_indices.is_empty() {
            let design = input
                .design
                .as_ref()
                .ok_or_else(|| GpuError::DriverCallFailed {
                    reason: "reml_trace: structural derivative present but design=None".to_string(),
                })?;
            let n = design.nrows();
            let design_col = to_col_major(design);
            let x_dev = pinned_htod(&ctx, &stream, &design_col)
                .map_err(|reason| GpuError::DriverCallFailed { reason })?;
            let mut rz_dev = stream
                .alloc_zeros::<f64>(n.checked_mul(k).ok_or_else(|| GpuError::DriverCallFailed {
                    reason: format!("reml_trace RZ overflow: n={n}, K={k}"),
                })?)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("reml_trace alloc RZ: {err}"),
                })?;
            let mut rw_dev =
                stream
                    .alloc_zeros::<f64>(n * k)
                    .map_err(|err| GpuError::DriverCallFailed {
                        reason: format!("reml_trace alloc RW: {err}"),
                    })?;
            // R_Z = X Z   (n × p) · (p × K) -> (n × K)
            gemm_nn(
                &blas,
                GemmShape {
                    m: n,
                    n: k,
                    k_inner: p,
                    lda: n,
                    ldb: p,
                    ldc: n,
                },
                &x_dev,
                &z_dev,
                &mut rz_dev,
            )?;
            // R_W = X W
            gemm_nn(
                &blas,
                GemmShape {
                    m: n,
                    n: k,
                    k_inner: p,
                    lda: n,
                    ldb: p,
                    ldc: n,
                },
                &x_dev,
                &w_dev,
                &mut rw_dev,
            )?;

            // Stack the row-weight vectors into A_stack column-major (n × D_gram).
            let d_gram = gram_indices.len();
            let mut a_stack = Vec::<f64>::with_capacity(n * d_gram);
            for &j in &gram_indices {
                let DerivativeHessian::WeightedGram { row_weights, .. } = &input.derivatives[j]
                else {
                    // SAFETY: gram_indices was populated in the partition loop above with
                    // exactly the indices whose variant is DerivativeHessian::WeightedGram.
                    // input.derivatives is immutably borrowed for the whole function so the
                    // slot at j cannot have been rewritten between partition and read; a
                    // failure here is a future-refactor bug, not a runtime input issue.
                    panic!(
                        "reml_trace structural path: derivative index {j} is in gram_indices \
                         but input.derivatives[{j}] is not DerivativeHessian::WeightedGram — \
                         gram_indices partition invariant violated"
                    );
                };
                let slice = row_weights
                    .as_slice()
                    .ok_or_else(|| GpuError::DriverCallFailed {
                        reason: format!("reml_trace structural H_j={j} row_weights not contiguous"),
                    })?;
                a_stack.extend_from_slice(slice);
            }
            let a_dev = pinned_htod(&ctx, &stream, &a_stack)
                .map_err(|reason| GpuError::DriverCallFailed { reason })?;
            let mut q_dev = stream.alloc_zeros::<f64>(d_gram * k).map_err(|err| {
                GpuError::DriverCallFailed {
                    reason: format!("reml_trace alloc Q_gram: {err}"),
                }
            })?;
            launch_reduce_q_weighted_gram(
                &stream,
                &compiled.module,
                n,
                k,
                d_gram,
                &rz_dev,
                &rw_dev,
                &a_dev,
                &mut q_dev,
            )?;
            let q_host_gram =
                stream
                    .clone_dtoh(&q_dev)
                    .map_err(|err| GpuError::DriverCallFailed {
                        reason: format!("reml_trace download Q_gram: {err}"),
                    })?;
            for (slot, &j) in gram_indices.iter().enumerate() {
                q_host[j * k..(j + 1) * k].copy_from_slice(&q_host_gram[slot * k..(slot + 1) * k]);
            }
            // penalty_extra contributions (uncommon, dense p×p) — handled on
            // host to keep the kernel surface small; total cost p² · K per
            // derivative that has one.
            for &j in &gram_indices {
                let DerivativeHessian::WeightedGram { penalty_extra, .. } = &input.derivatives[j]
                else {
                    // SAFETY: gram_indices was populated by the partition loop above with
                    // exactly the WeightedGram-variant indices; the same indices are
                    // re-walked here to pick up the optional penalty_extra field.
                    // input.derivatives has been immutably borrowed since partitioning, so
                    // the variant at index j cannot have changed. A let-else failure here
                    // would mean a future refactor split partition from consumer loops.
                    panic!(
                        "reml_trace structural penalty_extra: derivative index {j} is in \
                         gram_indices but input.derivatives[{j}] is not \
                         DerivativeHessian::WeightedGram — gram_indices partition invariant \
                         violated"
                    );
                };
                if let Some(pen) = penalty_extra {
                    let z_host =
                        stream
                            .clone_dtoh(&z_dev)
                            .map_err(|err| GpuError::DriverCallFailed {
                                reason: format!("reml_trace download Z for penalty_extra: {err}"),
                            })?;
                    let w_host =
                        stream
                            .clone_dtoh(&w_dev)
                            .map_err(|err| GpuError::DriverCallFailed {
                                reason: format!("reml_trace download W for penalty_extra: {err}"),
                            })?;
                    for col in 0..k {
                        let z_col = &z_host[col * p..(col + 1) * p];
                        let w_col = &w_host[col * p..(col + 1) * p];
                        let mut acc = 0.0_f64;
                        for r in 0..p {
                            let mut row_acc = 0.0_f64;
                            for c in 0..p {
                                row_acc += pen[[r, c]] * w_col[c];
                            }
                            acc += z_col[r] * row_acc;
                        }
                        q_host[j * k + col] += acc;
                    }
                }
            }
        }

        let (means, stderrs) = reduce_mean_stderr(&q_host, d, k);
        let mut gradient_rho_logdet = ndarray::Array1::<f64>::zeros(d);
        let mut gradient_rho_stderr = ndarray::Array1::<f64>::zeros(d);
        for j in 0..d {
            gradient_rho_logdet[j] = 0.5 * means[j];
            gradient_rho_stderr[j] = 0.5 * stderrs[j];
        }

        Ok(RemlTraceHutchinsonEvidence {
            logdet_hessian,
            gradient_rho_logdet,
            gradient_rho_stderr,
            probe_count: k,
        })
    }

    // ───── kernel launch wrappers ────────────────────────────────────────

    fn launch_fill_rademacher(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        seed: ProbeSeed,
        p: usize,
        k: usize,
        z: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let func = module
            .load_function("fill_rademacher_splitmix")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("reml_trace load fill_rademacher: {err}"),
            })?;
        let grid_x = ((p as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, k as u32, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let seed_arg: u64 = seed.0;
        let p_arg: u32 = p as u32;
        let k_arg: u32 = k as u32;
        // SAFETY: kernel signature matches arg types; Z is a live device
        // buffer sized p*k.
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&seed_arg)
                .arg(&p_arg)
                .arg(&k_arg)
                .arg(z)
                .launch(cfg)
        }
        .map(|_| ())
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("reml_trace launch fill_rademacher: {err}"),
        })
    }

    fn launch_reduce_q_dense(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        p: usize,
        k: usize,
        d: usize,
        z: &cudarc::driver::CudaSlice<f64>,
        y_stack: &cudarc::driver::CudaSlice<f64>,
        q: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let func =
            module
                .load_function("reduce_q_dense")
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("reml_trace load reduce_q_dense: {err}"),
                })?;
        let cfg = LaunchConfig {
            grid_dim: (k as u32, d as u32, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let p_arg: u32 = p as u32;
        let k_arg: u32 = k as u32;
        let d_arg: u32 = d as u32;
        // SAFETY: kernel signature matches; Z is (p,K), Y_stack is (p,K*D),
        // Q is (D,K) row-major as documented.
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&p_arg)
                .arg(&k_arg)
                .arg(&d_arg)
                .arg(z)
                .arg(y_stack)
                .arg(q)
                .launch(cfg)
        }
        .map(|_| ())
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("reml_trace launch reduce_q_dense: {err}"),
        })
    }

    fn launch_reduce_q_weighted_gram(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        n: usize,
        k: usize,
        d: usize,
        rz: &cudarc::driver::CudaSlice<f64>,
        rw: &cudarc::driver::CudaSlice<f64>,
        a_stack: &cudarc::driver::CudaSlice<f64>,
        q: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let func = module
            .load_function("reduce_q_weighted_gram")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("reml_trace load reduce_q_weighted_gram: {err}"),
            })?;
        let cfg = LaunchConfig {
            grid_dim: (k as u32, d as u32, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg: u32 = n as u32;
        let k_arg: u32 = k as u32;
        let d_arg: u32 = d as u32;
        // SAFETY: kernel signature matches; RZ, RW are (n,K), A_stack is (n,D).
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_arg)
                .arg(&k_arg)
                .arg(&d_arg)
                .arg(rz)
                .arg(rw)
                .arg(a_stack)
                .arg(q)
                .launch(cfg)
        }
        .map(|_| ())
        .map_err(|err| GpuError::DriverCallFailed {
            reason: format!("reml_trace launch reduce_q_weighted_gram: {err}"),
        })
    }

    fn copy_device_slice(
        stream: &Arc<CudaStream>,
        src: &cudarc::driver::CudaSlice<f64>,
        dst: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        stream
            .memcpy_dtod(src, dst)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("reml_trace dtod copy: {err}"),
            })
    }

    struct GemmShape {
        m: usize,
        n: usize,
        k_inner: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    }

    fn gemm_nn(
        blas: &CudaBlas,
        shape: GemmShape,
        a: &cudarc::driver::CudaSlice<f64>,
        b: &cudarc::driver::CudaSlice<f64>,
        c: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let GemmShape {
            m,
            n,
            k_inner,
            lda,
            ldb,
            ldc,
        } = shape;
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: m as i32,
            n: n as i32,
            k: k_inner as i32,
            alpha: 1.0,
            lda: lda as i32,
            ldb: ldb as i32,
            beta: 0.0,
            ldc: ldc as i32,
        };
        // SAFETY: dgemm with column-major leading dims documented above;
        // buffers a, b, c sized lda*k_inner, ldb*n, ldc*n.
        unsafe { blas.gemm(cfg, a, b, c) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("reml_trace cublas dgemm: {err}"),
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// Shared validation + linear algebra helpers
// ────────────────────────────────────────────────────────────────────────

fn validate_inputs(input: &RemlTraceHutchinsonInput<'_>) -> Result<(), String> {
    let (p, p2) = input.penalized_hessian.dim();
    if p == 0 || p != p2 {
        return Err(format!("reml_trace input H must be square, got {p}x{p2}"));
    }
    if input.probe_count < 2 {
        return Err(format!(
            "reml_trace requires probe_count >= 2 for a sample SE, got {}",
            input.probe_count
        ));
    }
    let needs_design = input
        .derivatives
        .iter()
        .any(|d| matches!(d, DerivativeHessian::WeightedGram { .. }));
    if needs_design && input.design.is_none() {
        return Err("reml_trace: structural derivative present but design=None".to_string());
    }
    let n = input.design.as_ref().map(|x| x.nrows()).unwrap_or(0);
    if let Some(x) = input.design.as_ref()
        && x.ncols() != p
    {
        return Err(format!(
            "reml_trace design has {} columns, expected p={p}",
            x.ncols()
        ));
    }
    for (j, derivative) in input.derivatives.iter().enumerate() {
        derivative
            .dim_p(p, n)
            .map_err(String::from)
            .map_err(|e| format!("reml_trace derivative {j}: {e}"))?;
    }
    Ok(())
}

/// Compute the per-derivative sample mean and sample-mean SE from the
/// flat row-major (D, K) Q matrix. SE uses Bessel's correction (K-1) for
/// the variance and divides by `sqrt(K)` so the returned value is the
/// standard error of the mean.
fn reduce_mean_stderr(q: &[f64], d: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        q.len(),
        d * k,
        "reduce_mean_stderr: q buffer length {} != D*K = {}*{}",
        q.len(),
        d,
        k
    );
    let mut means = vec![0.0_f64; d];
    let mut stderrs = vec![0.0_f64; d];
    let inv_k = 1.0 / (k as f64);
    for j in 0..d {
        let row = &q[j * k..(j + 1) * k];
        let mean = row.iter().copied().sum::<f64>() * inv_k;
        means[j] = mean;
        if k >= 2 {
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ((k - 1) as f64);
            stderrs[j] = (var / (k as f64)).sqrt();
        }
    }
    (means, stderrs)
}

// ── Cholesky helpers (CPU reference only) ──────────────────────────────

fn cholesky_lower(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "reml_trace CPU Cholesky: non-SPD diagonal {sum} at row {i}"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

fn solve_cholesky(l: &Array2<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = l.nrows();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ArrayView2};

    fn make_spd(p: usize, jitter: f64) -> Array2<f64> {
        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                h[[i, j]] = if i == j {
                    p as f64 + jitter
                } else {
                    1.0 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        h
    }

    fn random_dense_sym(p: usize, seed: u64) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((p, p));
        let mut s = seed;
        for i in 0..p {
            for j in i..p {
                s = splitmix64_mix(s.wrapping_add(1));
                let v = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                a[[i, j]] = v;
                a[[j, i]] = v;
            }
        }
        a
    }

    fn exact_trace_hinv_a(h: ArrayView2<f64>, a: ArrayView2<f64>) -> f64 {
        let p = h.nrows();
        let factor = cholesky_lower(&h.to_owned()).expect("SPD");
        let mut trace = 0.0;
        for col in 0..p {
            let mut e = vec![0.0_f64; p];
            e[col] = 1.0;
            let w = solve_cholesky(&factor, &e);
            // (H^{-1} A) diag entry [col, col] = sum_i A[col, i] * w[i]
            let mut diag = 0.0;
            for i in 0..p {
                diag += a[[col, i]] * w[i];
            }
            trace += diag;
        }
        trace
    }

    #[test]
    fn splitmix_is_deterministic_and_disperses() {
        // Self-consistency: same input → same output, and a few near-by
        // inputs land in distinct buckets (no trivial collisions).
        assert_eq!(splitmix64_mix(42), splitmix64_mix(42));
        let mut bits_seen = 0u64;
        for x in 0u64..64 {
            bits_seen |= splitmix64_mix(x);
        }
        assert_eq!(
            bits_seen,
            u64::MAX,
            "splitmix should cover every bit position across 64 inputs"
        );
    }

    #[test]
    fn rademacher_entries_are_pm_one_and_stateless() {
        let seed = ProbeSeed(0xCAFE_BABE);
        for k in 0..16u64 {
            for i in 0..32u64 {
                let v = rademacher_entry(seed.0, k, i);
                assert!(
                    v == 1.0 || v == -1.0,
                    "non-pm1 entry at (k={k}, i={i}): {v}"
                );
                let v2 = rademacher_entry(seed.0, k, i);
                assert_eq!(v, v2, "same (k,i) must hash to same value");
            }
        }
    }

    #[test]
    fn rademacher_common_random_numbers_match_for_prefix() {
        // First 16 probes of a K=16 run must equal first 16 probes of K=32.
        let p = 50;
        let mut z16 = vec![0.0_f64; p * 16];
        let mut z32 = vec![0.0_f64; p * 32];
        fill_rademacher_host(ProbeSeed(7), p, 16, &mut z16);
        fill_rademacher_host(ProbeSeed(7), p, 32, &mut z32);
        for col in 0..16 {
            for row in 0..p {
                assert_eq!(
                    z16[col * p + row],
                    z32[col * p + row],
                    "CRN broken at (col={col}, row={row})"
                );
            }
        }
    }

    #[test]
    fn cpu_hutchinson_unbiased_against_exact_small_spd() {
        let p = 16;
        let h = make_spd(p, 0.5);
        let a1 = random_dense_sym(p, 0x1234);
        let a2 = random_dense_sym(p, 0x5678);
        let exact1 = exact_trace_hinv_a(h.view(), a1.view());
        let exact2 = exact_trace_hinv_a(h.view(), a2.view());
        let input = RemlTraceHutchinsonInput {
            penalized_hessian: h.view(),
            derivatives: vec![
                DerivativeHessian::Dense(a1.view()),
                DerivativeHessian::Dense(a2.view()),
            ],
            design: None,
            probe_count: 4096,
            seed: ProbeSeed(0xCAFE_BABE),
        };
        let evidence = evidence_derivatives_hutchinson_cpu(&input).expect("ok");
        // gradient = 0.5 * trace, so multiply estimate by 2 for the trace.
        let est1 = 2.0 * evidence.gradient_rho_logdet[0];
        let est2 = 2.0 * evidence.gradient_rho_logdet[1];
        let tol1 = 6.0 * evidence.gradient_rho_stderr[0].max(1e-8) * 2.0;
        let tol2 = 6.0 * evidence.gradient_rho_stderr[1].max(1e-8) * 2.0;
        assert!(
            (est1 - exact1).abs() <= tol1,
            "Hutchinson est {est1} too far from exact {exact1} (tol={tol1}, se={})",
            evidence.gradient_rho_stderr[0]
        );
        assert!(
            (est2 - exact2).abs() <= tol2,
            "Hutchinson est {est2} too far from exact {exact2} (tol={tol2})"
        );
    }

    #[test]
    fn structural_path_matches_dense_for_xtwx() {
        // Build H_j = X^T diag(a) X exactly; both the dense and the
        // structural descriptor must produce the same q value per probe.
        let n = 40;
        let p = 8;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut s = 11u64;
        for r in 0..n {
            for c in 0..p {
                s = splitmix64_mix(s.wrapping_add(1));
                x[[r, c]] = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            }
        }
        let a: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * (i as f64)).collect();
        let a_arr = ndarray::Array1::from(a);
        // H_j dense
        let mut hj_dense = Array2::<f64>::zeros((p, p));
        for r in 0..p {
            for c in 0..p {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += x[[i, r]] * a_arr[i] * x[[i, c]];
                }
                hj_dense[[r, c]] = acc;
            }
        }
        // SPD H so the solve is well posed.
        let mut h = make_spd(p, 1.0);
        for i in 0..p {
            h[[i, i]] += 1.0;
        }
        let input_dense = RemlTraceHutchinsonInput {
            penalized_hessian: h.view(),
            derivatives: vec![DerivativeHessian::Dense(hj_dense.view())],
            design: None,
            probe_count: 32,
            seed: ProbeSeed(123),
        };
        let input_struct = RemlTraceHutchinsonInput {
            penalized_hessian: h.view(),
            derivatives: vec![DerivativeHessian::WeightedGram {
                row_weights: a_arr.view(),
                penalty_extra: None,
            }],
            design: Some(x.view()),
            probe_count: 32,
            seed: ProbeSeed(123),
        };
        let e_dense = evidence_derivatives_hutchinson_cpu(&input_dense).expect("ok");
        let e_struct = evidence_derivatives_hutchinson_cpu(&input_struct).expect("ok");
        // Same probes, same H_j ⇒ identical estimator (modulo round-off).
        assert!(
            (e_dense.gradient_rho_logdet[0] - e_struct.gradient_rho_logdet[0]).abs() < 1e-9,
            "dense vs structural mismatch: dense={}, struct={}",
            e_dense.gradient_rho_logdet[0],
            e_struct.gradient_rho_logdet[0]
        );
    }

    #[test]
    fn finite_difference_check_against_logdet() {
        // For H(rho) = H0 + rho * A, d/d(rho) log|H| = tr(H^{-1} A).
        let p = 10;
        let h0 = make_spd(p, 0.2);
        let a = random_dense_sym(p, 0xABCD);
        let eps = 1e-4;
        let mut hp = h0.clone();
        let mut hm = h0.clone();
        for i in 0..p {
            for j in 0..p {
                hp[[i, j]] += eps * a[[i, j]];
                hm[[i, j]] -= eps * a[[i, j]];
            }
        }
        let ld = |m: &Array2<f64>| -> f64 {
            let l = cholesky_lower(m).unwrap();
            2.0 * (0..p).map(|i| l[[i, i]].ln()).sum::<f64>()
        };
        let fd = (ld(&hp) - ld(&hm)) / (2.0 * eps);
        let exact = exact_trace_hinv_a(h0.view(), a.view());
        assert!(
            (fd - exact).abs() / exact.abs().max(1e-12) < 1e-6,
            "FD logdet derivative {fd} != exact trace {exact}"
        );
        // And Hutchinson should land near 0.5 * exact (the gradient form).
        let input = RemlTraceHutchinsonInput {
            penalized_hessian: h0.view(),
            derivatives: vec![DerivativeHessian::Dense(a.view())],
            design: None,
            probe_count: 4096,
            seed: ProbeSeed(0xAA55),
        };
        let evidence = evidence_derivatives_hutchinson_cpu(&input).expect("ok");
        let tol = 8.0 * evidence.gradient_rho_stderr[0].max(1e-8);
        assert!(
            (evidence.gradient_rho_logdet[0] - 0.5 * exact).abs() < tol,
            "Hutchinson gradient {} not within 8·SE of 0.5·exact={}",
            evidence.gradient_rho_logdet[0],
            0.5 * exact
        );
    }

    #[test]
    fn gate_rejects_below_min_p() {
        assert!(!should_use_gpu_hutchinson(64, 16, true, true, true, false));
    }

    #[test]
    fn gate_rejects_k_out_of_range() {
        assert!(!should_use_gpu_hutchinson(2000, 4, true, true, true, false));
        assert!(!should_use_gpu_hutchinson(
            2000, 200, true, true, true, false
        ));
    }

    #[test]
    fn gate_rejects_when_subspace_active() {
        assert!(!should_use_gpu_hutchinson(2000, 16, true, true, true, true));
    }

    #[test]
    fn gate_accepts_canonical_case() {
        assert!(should_use_gpu_hutchinson(2000, 16, true, true, true, false));
    }
}
