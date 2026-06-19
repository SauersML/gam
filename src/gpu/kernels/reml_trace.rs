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
//! The structural path is the high-value route for large-scale models
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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use crate::gpu::gpu_error::GpuError;
use crate::linalg::pcg::pcg_core;

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
                    crate::gpu_bail!(
                        "reml_trace dense H_j: shape {:?} != ({expected_p}, {expected_p})",
                        matrix.dim()
                    );
                }
            }
            DerivativeHessian::WeightedGram {
                row_weights,
                penalty_extra,
            } => {
                if row_weights.len() != expected_n {
                    crate::gpu_bail!(
                        "reml_trace structural H_j: row_weights.len()={} != n={expected_n}",
                        row_weights.len()
                    );
                }
                if let Some(p_extra) = penalty_extra
                    && (p_extra.nrows() != expected_p || p_extra.ncols() != expected_p)
                {
                    crate::gpu_bail!(
                        "reml_trace structural H_j penalty_extra: shape {:?} != ({expected_p}, {expected_p})",
                        p_extra.dim()
                    );
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
    /// Per-probe sample standard deviation of the half-scaled gradient term
    /// `(1/2)·q_{j,·}` across the `K` probes (i.e. `0.5·sd`, NOT divided by
    /// `sqrt(K)`), length `D`. To obtain the standard error of the running
    /// mean `g_j`, divide by `sqrt(K)` (= `sqrt(probe_count)`).
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

/// SplitMix64 finalizer (Sebastiano Vigna, 2015). Thin wrapper over the
/// canonical implementation in [`crate::linalg::utils::splitmix64_hash`].
#[inline]
pub fn splitmix64_mix(z: u64) -> u64 {
    crate::linalg::utils::splitmix64_hash(z)
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
        if crate::gpu::device_runtime::GpuRuntime::global().is_some() {
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
// Adaptive K (Block 2.5)
// ────────────────────────────────────────────────────────────────────────

/// Default relative-error target for the adaptive-K stopping rule.
/// Matches `StochasticTraceConfig::default().relative_tol`.
pub const HUTCHINSON_ADAPTIVE_REL_TOL: f64 = 0.01;
/// Default near-zero-trace protection floor. Matches
/// `StochasticTraceConfig::default().tau_rel`.
pub const HUTCHINSON_ADAPTIVE_TAU_REL: f64 = 1e-8;

/// Adaptive-K Hutchinson trace schedule with common random numbers (CRN).
///
/// Repeatedly invokes [`evidence_derivatives_hutchinson_gpu`] with probe
/// counts `K = 16, 32, 64, 128`, stopping at the first `K` that satisfies
/// the per-coordinate relative-SE criterion
///
/// ```text
/// max_j  s_j / (sqrt(K) · max(|t_j|, τ))  ≤  ε
/// ```
///
/// where `s_j` is the sample standard deviation across the `K` probes (the
/// raw quadratic-form sample, *without* the `(1/2)` REML logdet scaling)
/// and `t_j` is the running mean. Because the SplitMix probe RNG is
/// stateless (`(seed, k_index, i) → ±1`), the first `K_prev` probes of a
/// `K = 2·K_prev` re-run are bit-identical to the previous batch, so each
/// step extends the prior estimate rather than starting fresh in
/// expectation. The implementation re-runs from scratch at each `K` for
/// simplicity; CRN is preserved by the stateless RNG seed.
///
/// Returns the **raw traces** `t_j = tr(H⁻¹ H_j) = mean_k q_{j,k}`
/// (length `D`), the `log|H|` from the cached Cholesky, and the final
/// probe count `K` actually used. The raw traces (not the `(1/2)` REML
/// logdet gradient) are what the outer evaluator wants — it applies the
/// logdet-gradient half-factor itself.
pub struct AdaptiveTraceEvidence {
    pub logdet_hessian: f64,
    pub traces: Array1<f64>,
    /// Per-probe sample standard deviation of the raw trace estimator
    /// `q_{j,·}` (NOT divided by `sqrt(K)`); divide by `sqrt(probe_count)`
    /// to obtain the standard error of the running mean `traces[j]`.
    pub stderrs: Array1<f64>,
    pub probe_count: usize,
    pub converged: bool,
}

pub fn evidence_traces_adaptive<'a>(
    penalized_hessian: ArrayView2<'a, f64>,
    derivatives: Vec<DerivativeHessian<'a>>,
    design: Option<ArrayView2<'a, f64>>,
    seed: ProbeSeed,
    rel_tol: f64,
    tau_rel: f64,
) -> Result<AdaptiveTraceEvidence, String> {
    // Adaptive schedule per math team block 2 §16: K = 16, 32, 64, 128.
    const SCHEDULE: [usize; 4] = [16, 32, 64, 128];

    let d = derivatives.len();
    if d == 0 {
        return Err("evidence_traces_adaptive: derivatives is empty".to_string());
    }
    if !(rel_tol > 0.0) {
        return Err(format!(
            "evidence_traces_adaptive: rel_tol must be > 0 (got {rel_tol})"
        ));
    }
    if !(tau_rel > 0.0) {
        return Err(format!(
            "evidence_traces_adaptive: tau_rel must be > 0 (got {tau_rel})"
        ));
    }

    let mut last_logdet = 0.0_f64;
    let mut last_traces = Array1::<f64>::zeros(d);
    let mut last_stderrs = Array1::<f64>::zeros(d);
    let mut last_k = 0_usize;
    let mut converged = false;

    for &k in &SCHEDULE {
        let input = RemlTraceHutchinsonInput {
            penalized_hessian,
            derivatives: derivatives.clone(),
            design,
            probe_count: k,
            seed,
        };
        let evidence = evidence_derivatives_hutchinson_gpu(input)?;
        last_logdet = evidence.logdet_hessian;
        last_k = k;

        // The dispatch entry returns the **(1/2)·mean** REML logdet
        // gradient and **(1/2)·per-probe sample SD**. Undo the half to
        // recover the raw `t_j = mean_k q_{j,k}` and the per-probe sample
        // standard deviation `s_j` the stopping rule wants.
        for j in 0..d {
            last_traces[j] = 2.0 * evidence.gradient_rho_logdet[j];
            last_stderrs[j] = 2.0 * evidence.gradient_rho_stderr[j];
        }

        // Stopping rule (math block 2 §16):
        //   max_j  s_j / (sqrt(K) · max(|t_j|, τ))  ≤  ε
        // `s_j` here is the per-probe sample standard deviation across the K
        // probes (`reduce_mean_stderr` returns the raw SD, NOT the SE-of-mean);
        // dividing by sqrt(K) once converts it to the standard error of the
        // running mean. (The earlier double-sqrt(K) division — once here and
        // once already inside `reduce_mean_stderr` — made this test sqrt(K)×
        // too lax, stopping the schedule at too-small K; see #829.)
        let sqrt_k = (k as f64).sqrt();
        let mut worst = 0.0_f64;
        for j in 0..d {
            let denom = sqrt_k * last_traces[j].abs().max(tau_rel);
            let r = last_stderrs[j] / denom;
            if r > worst {
                worst = r;
            }
        }
        if worst <= rel_tol {
            converged = true;
            break;
        }
    }

    Ok(AdaptiveTraceEvidence {
        logdet_hessian: last_logdet,
        traces: last_traces,
        stderrs: last_stderrs,
        probe_count: last_k,
        converged,
    })
}

// ────────────────────────────────────────────────────────────────────────
// Block 2.7: batched-PCG HVP variant of adaptive Hutchinson
// ────────────────────────────────────────────────────────────────────────

/// CG convergence tolerance for the per-probe solve `H w = z`. The outer
/// adaptive-K loop already drives Hutchinson variance to ~1%; a per-probe
/// relative residual of 1e-6 keeps the CG round-off well below the
/// stochastic SE without paying for double-machine convergence.
pub const PCG_HVP_REL_TOL: f64 = 1e-6;

/// Maximum CG iterations per probe before we stop and accept the partial
/// solve. Capped so a poorly conditioned `H` cannot make a single REML
/// step pay unbounded time — the Hutchinson estimator is statistically
/// robust to a few stale `w_k` values (it inflates SE, which the adaptive
/// stopping rule then catches by extending the schedule).
pub const PCG_HVP_MAX_ITERS: usize = 200;

/// Adaptive Hutchinson variant that consumes `H` as a matrix-free HVP
/// closure rather than a dense `ArrayView2`. Used by call sites where the
/// penalized Hessian is implicit (operator-only) and forming it densely
/// would blow the memory budget — e.g. the device-resident PCG path in
/// `gpu/bms_flex_row.rs` or the large-scale BMS Schur operator.
///
/// `hvp` must compute `out ← H · v` for an SPD `H`. The closure is called
/// once per CG iteration per probe (so `K · iters_per_probe` times in
/// total for each schedule step). It is responsible for any necessary
/// pre-conditioning state, threading, or device residency — the routine
/// itself is pure CPU.
///
/// `derivatives` are still passed as dense or `WeightedGram`; the
/// adaptive trace `t_j = mean_k z_k^T H_j w_k` only needs `H_j` to be
/// available as a matvec, and the dense / weighted-Gram variants of
/// `DerivativeHessian::quadratic_form` already provide that.
///
/// CRN is preserved exactly as in [`evidence_traces_adaptive`]: the
/// SplitMix probe RNG is stateless in `(seed, k_index, i)`, so the
/// `K=16, 32, 64, 128` schedule extends the prior estimate rather than
/// restarting it. Each schedule step re-runs all `K` solves; the
/// implementation is intentionally simple, the asymptotic cost is
/// dominated by the largest `K`.
///
/// Returns the same [`AdaptiveTraceEvidence`] shape as the dense path,
/// with one exception: `logdet_hessian` is **NaN** because no Cholesky
/// is performed. Callers needing both `tr(H⁻¹ H_j)` and `log|H|` from
/// the matrix-free path should obtain `log|H|` separately (e.g. via
/// stochastic Lanczos or by routing through the dense path when `H`
/// fits in memory).
pub fn evidence_traces_adaptive_hvp<F>(
    p: usize,
    mut hvp: F,
    derivatives: Vec<DerivativeHessian<'_>>,
    design: Option<ArrayView2<'_, f64>>,
    seed: ProbeSeed,
    rel_tol: f64,
    tau_rel: f64,
) -> Result<AdaptiveTraceEvidence, String>
where
    F: FnMut(&[f64], &mut [f64]),
{
    const SCHEDULE: [usize; 4] = [16, 32, 64, 128];

    let d = derivatives.len();
    if d == 0 {
        return Err("evidence_traces_adaptive_hvp: derivatives is empty".to_string());
    }
    if p == 0 {
        return Err("evidence_traces_adaptive_hvp: p must be > 0".to_string());
    }
    if !(rel_tol > 0.0) {
        return Err(format!(
            "evidence_traces_adaptive_hvp: rel_tol must be > 0 (got {rel_tol})"
        ));
    }
    if !(tau_rel > 0.0) {
        return Err(format!(
            "evidence_traces_adaptive_hvp: tau_rel must be > 0 (got {tau_rel})"
        ));
    }

    let mut last_traces = Array1::<f64>::zeros(d);
    let mut last_stderrs = Array1::<f64>::zeros(d);
    let mut last_k = 0_usize;
    let mut converged = false;

    let mut z = vec![0.0_f64; p];
    let mut w = vec![0.0_f64; p];

    // Per-derivative Welford accumulators (running mean and sum-of-squared
    // deviations M2) for a numerically stable online mean / sample variance.
    // The naive one-pass form E[q²] − E[q]² catastrophically cancels when the
    // per-probe q cluster far from zero with small spread — exactly the
    // near-converged regime the stopping rule cares about — so we track M2
    // directly to match the two-pass `reduce_mean_stderr` without that loss.
    let mut q_means = vec![0.0_f64; d];
    let mut q_m2 = vec![0.0_f64; d];

    for &k_target in &SCHEDULE {
        // Re-run from scratch at each schedule step — CRN guarantees the
        // first min(K_prev, K_target) probes are bit-identical, so the
        // estimator is monotone in expectation across schedule extensions.
        for s in q_means.iter_mut() {
            *s = 0.0;
        }
        for s in q_m2.iter_mut() {
            *s = 0.0;
        }

        for k_idx in 0..k_target {
            // Fill z_k from the stateless SplitMix RNG.
            for i in 0..p {
                z[i] = rademacher_entry(seed.0, k_idx as u64, i as u64);
            }
            // Solve H w = z by unpreconditioned CG.
            cg_solve(&mut hvp, &z, &mut w, PCG_HVP_REL_TOL, PCG_HVP_MAX_ITERS);

            // Reduce q_{j,k} = z^T H_j w for each derivative. Mirrors the
            // dense reference in `evidence_derivatives_hutchinson_cpu`.
            for j in 0..d {
                let q = match &derivatives[j] {
                    DerivativeHessian::Dense(matrix) => {
                        let mut y = 0.0_f64;
                        for r in 0..p {
                            let mut hr_w = 0.0_f64;
                            for c in 0..p {
                                hr_w += matrix[[r, c]] * w[c];
                            }
                            y += z[r] * hr_w;
                        }
                        y
                    }
                    DerivativeHessian::WeightedGram {
                        row_weights,
                        penalty_extra,
                    } => {
                        let design_view = design.as_ref().ok_or_else(|| {
                            "evidence_traces_adaptive_hvp: WeightedGram derivative requires \
                             design matrix"
                                .to_string()
                        })?;
                        let n = design_view.nrows();
                        let mut acc = 0.0_f64;
                        for row in 0..n {
                            let mut rz = 0.0_f64;
                            let mut rw = 0.0_f64;
                            for ci in 0..p {
                                rz += design_view[[row, ci]] * z[ci];
                                rw += design_view[[row, ci]] * w[ci];
                            }
                            acc += row_weights[row] * rz * rw;
                        }
                        if let Some(pen) = penalty_extra {
                            for r in 0..p {
                                let mut row_acc = 0.0_f64;
                                for c in 0..p {
                                    row_acc += pen[[r, c]] * w[c];
                                }
                                acc += z[r] * row_acc;
                            }
                        }
                        acc
                    }
                };
                // Welford update with the 1-based probe count (k_idx + 1).
                let count = (k_idx + 1) as f64;
                let delta = q - q_means[j];
                q_means[j] += delta / count;
                let delta2 = q - q_means[j];
                q_m2[j] += delta * delta2;
            }
        }

        let n = k_target as f64;
        let mut worst_ratio = 0.0_f64;
        for j in 0..d {
            let mean = q_means[j];
            // Sample variance M2 / (K−1) — Bessel's correction, matching the
            // two-pass `reduce_mean_stderr` exactly (no one-pass cancellation).
            // For K = 1 there is no spread to estimate, so the variance is 0.
            let var = if n > 1.0 { q_m2[j] / (n - 1.0) } else { 0.0 };
            let s = var.sqrt();
            last_traces[j] = mean;
            last_stderrs[j] = s;
            let denom = n.sqrt() * mean.abs().max(tau_rel);
            let r = s / denom;
            if r > worst_ratio {
                worst_ratio = r;
            }
        }
        last_k = k_target;
        if worst_ratio <= rel_tol {
            converged = true;
            break;
        }
    }

    Ok(AdaptiveTraceEvidence {
        logdet_hessian: f64::NAN,
        traces: last_traces,
        stderrs: last_stderrs,
        probe_count: last_k,
        converged,
    })
}

/// Unpreconditioned conjugate gradients for `H w = b` with `H` accessed
/// only through `hvp(v, out) → out ← H v`. SPD `H` is required.
/// Initial guess is `w = 0`; stops when `‖r‖ ≤ rel_tol · ‖b‖` or after
/// `max_iters` iterations.
///
/// Thin wrapper over the shared [`pcg_core`] (`linalg::pcg`): unpreconditioned
/// (all-ones Jacobi diagonal), no residual refresh (`refresh_period = 0`), and
/// no diagnostics — exactly the serial recurrence this used to inline. On a
/// breakdown (lost SPD near convergence, non-finite scalar) the core stops and
/// leaves the last valid iterate in `w`, which is the historical "accept
/// current w" behavior. The serial inner products in `pcg_core` reproduce the
/// byte-for-byte iterates of the previous hand-rolled loop.
fn cg_solve<F>(hvp: &mut F, b: &[f64], w: &mut [f64], rel_tol: f64, max_iters: usize)
where
    F: FnMut(&[f64], &mut [f64]),
{
    let n = b.len();
    assert!(w.len() == n);

    let rhs = ArrayView1::from(b);
    let precond = Array1::<f64>::ones(n);
    let mut solution = ArrayViewMut1::from(w);

    pcg_core(
        |v: &Array1<f64>, out: &mut Array1<f64>| {
            // The core hands contiguous vectors; `hvp` speaks raw slices.
            let v_slice = v.as_slice().expect("contiguous CG direction view");
            let out_slice = out.as_slice_mut().expect("contiguous CG matvec view");
            hvp(v_slice, out_slice);
        },
        &rhs,
        &precond.view(),
        rel_tol,
        max_iters,
        0,
        false,
        &mut solution,
    );
}

// ────────────────────────────────────────────────────────────────────────
// Outer logdet-gradient dispatch gate (Block 2.5)
// ────────────────────────────────────────────────────────────────────────

/// Composite gate predicate for the outer REML logdet-gradient bypass:
/// when this returns `true`, the unified evaluator should replace its
/// CPU stochastic-trace call with [`evidence_traces_adaptive`].
///
/// All five conditions must hold simultaneously:
/// * `p ≥ 512` and `K_initial..=K_max` is `[16, 128]`
/// * `H` is resident as a dense SPD operator (caller passes
///   `dense_spd_h_resident = true` when `hop.as_exact_dense_spectral()`
///   is `Some` AND the Cholesky succeeds — the latter is checked
///   indirectly by `plain_spd_logdet`).
/// * `plain_spd_logdet`: the operator's logdet kernel is `H⁻¹` exactly
///   (i.e. `hop.logdet_traces_match_hinv_kernel() && hop.is_dense()`),
///   so smooth-spectral and SCOP-warped paths are excluded.
/// * `prefers_stochastic`: `hop.prefers_stochastic_trace_estimation()`.
/// * `!projected_penalty_subspace_active`: the rank-deficient LAML
///   projected kernel `U_S H_proj⁻¹ U_Sᵀ` is **not** installed.
#[must_use]
pub fn should_bypass_cpu_with_gpu_adaptive(
    p: usize,
    dense_spd_h_resident: bool,
    plain_spd_logdet: bool,
    prefers_stochastic: bool,
    projected_penalty_subspace_active: bool,
) -> bool {
    p >= HUTCHINSON_GPU_MIN_P
        && dense_spd_h_resident
        && plain_spd_logdet
        && prefers_stochastic
        && !projected_penalty_subspace_active
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
    use crate::gpu::gpu_error::{GpuError, GpuResultExt};
    use crate::gpu::solver::{
        cholesky_logdet_from_col_major, context_and_stream, pinned_htod, potrf_in_place,
        potrs_in_place,
    };
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::cusolver::DnHandle;
    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::Arc;

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

    fn module(ctx: &Arc<CudaContext>) -> Result<&'static Arc<CudaModule>, GpuError> {
        static CACHE: crate::gpu::device_cache::PtxModuleCache =
            crate::gpu::device_cache::PtxModuleCache::new();
        CACHE.get_or_compile(ctx, "reml_trace", PTX_SOURCE)
    }

    pub(super) fn evidence_derivatives(
        input: &RemlTraceHutchinsonInput<'_>,
    ) -> Result<RemlTraceHutchinsonEvidence, GpuError> {
        let p = input.penalized_hessian.nrows();
        let d = input.derivatives.len();
        let k = input.probe_count;
        let (ctx, stream) =
            context_and_stream().map_err(|reason| GpuError::DriverCallFailed { reason })?;
        let solver = DnHandle::new(stream.clone()).gpu_ctx("reml_trace cusolver init")?;
        let blas = CudaBlas::new(stream.clone()).gpu_ctx("reml_trace cublas init")?;
        let compiled = module(&ctx)?;
        let module_handle: &Arc<CudaModule> = compiled;

        // ── 1. Upload H, factor once.
        let h_col = to_col_major(&input.penalized_hessian);
        let mut h_dev =
            pinned_htod(&stream, &h_col).map_err(|reason| GpuError::DriverCallFailed { reason })?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)
            .map_err(|reason| GpuError::DriverCallFailed { reason })?;
        let factor_col = stream
            .clone_dtoh(&h_dev)
            .gpu_ctx("reml_trace download factor")?;
        let logdet_hessian = cholesky_logdet_from_col_major(&factor_col, p);

        // ── 2. Allocate Z (p, K) and fill with Rademacher entries on device.
        let total_z = p
            .checked_mul(k)
            .ok_or_else(|| gpu_err!("reml_trace Z size overflow: p={p}, K={k}"))?;
        let mut z_dev = stream
            .alloc_zeros::<f64>(total_z)
            .gpu_ctx("reml_trace alloc Z")?;
        launch_fill_rademacher(&stream, module_handle, input.seed, p, k, &mut z_dev)?;

        // ── 3. Solve H W = Z in a single batched potrs call (nrhs = K).
        //     Copy Z into a fresh buffer first; potrs is in-place.
        let mut w_dev = stream
            .alloc_zeros::<f64>(total_z)
            .gpu_ctx("reml_trace alloc W")?;
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
        //       (would explode memory at large-scale-p), but the GEMMs share
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
                let hj_dev = pinned_htod(&stream, &hj_col)
                    .map_err(|reason| GpuError::DriverCallFailed { reason })?;
                let mut y_dev = stream
                    .alloc_zeros::<f64>(total_z)
                    .map_err(|err| gpu_err!("reml_trace alloc Y_j (j={j}): {err}"))?;
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
                let mut q_j_dev = stream
                    .alloc_zeros::<f64>(k)
                    .gpu_ctx_with(|err| format!("reml_trace alloc Q_j (j={j}): {err}"))?;
                launch_reduce_q_dense(
                    &stream,
                    module_handle,
                    p,
                    k,
                    1,
                    &z_dev,
                    &y_dev,
                    &mut q_j_dev,
                )?;
                let q_host_j = stream
                    .clone_dtoh(&q_j_dev)
                    .gpu_ctx_with(|err| format!("reml_trace download Q_j (j={j}): {err}"))?;
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
            let x_dev = pinned_htod(&stream, &design_col)
                .map_err(|reason| GpuError::DriverCallFailed { reason })?;
            let mut rz_dev = stream
                .alloc_zeros::<f64>(
                    n.checked_mul(k)
                        .ok_or_else(|| gpu_err!("reml_trace RZ overflow: n={n}, K={k}"))?,
                )
                .gpu_ctx("reml_trace alloc RZ")?;
            let mut rw_dev = stream
                .alloc_zeros::<f64>(n * k)
                .gpu_ctx("reml_trace alloc RW")?;
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
                let slice = row_weights.as_slice().ok_or_else(|| {
                    gpu_err!("reml_trace structural H_j={j} row_weights not contiguous")
                })?;
                a_stack.extend_from_slice(slice);
            }
            let a_dev = pinned_htod(&stream, &a_stack)
                .map_err(|reason| GpuError::DriverCallFailed { reason })?;
            let mut q_dev = stream
                .alloc_zeros::<f64>(d_gram * k)
                .map_err(|err| gpu_err!("reml_trace alloc Q_gram: {err}"))?;
            launch_reduce_q_weighted_gram(
                &stream,
                module_handle,
                n,
                k,
                d_gram,
                &rz_dev,
                &rw_dev,
                &a_dev,
                &mut q_dev,
            )?;
            let q_host_gram = stream
                .clone_dtoh(&q_dev)
                .gpu_ctx("reml_trace download Q_gram")?;
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
                    let z_host = stream
                        .clone_dtoh(&z_dev)
                        .gpu_ctx("reml_trace download Z for penalty_extra")?;
                    let w_host = stream
                        .clone_dtoh(&w_dev)
                        .gpu_ctx("reml_trace download W for penalty_extra")?;
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
            .gpu_ctx("reml_trace load fill_rademacher")?;
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
        .gpu_ctx("reml_trace launch fill_rademacher")
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
        let func = module
            .load_function("reduce_q_dense")
            .gpu_ctx("reml_trace load reduce_q_dense")?;
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
        .gpu_ctx("reml_trace launch reduce_q_dense")
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
            .gpu_ctx("reml_trace load reduce_q_weighted_gram")?;
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
        .gpu_ctx("reml_trace launch reduce_q_weighted_gram")
    }

    fn copy_device_slice(
        stream: &Arc<CudaStream>,
        src: &cudarc::driver::CudaSlice<f64>,
        dst: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        stream.memcpy_dtod(src, dst).gpu_ctx("reml_trace dtod copy")
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
        unsafe { blas.gemm(cfg, a, b, c) }.gpu_ctx("reml_trace cublas dgemm")
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

/// Compute the per-derivative sample mean and **per-probe sample standard
/// deviation** from the flat row-major (D, K) Q matrix. The SD uses Bessel's
/// correction (K-1) for the variance; it is NOT divided by `sqrt(K)`. Callers
/// that want the standard error of the running mean must divide by `sqrt(K)`
/// themselves (the stopping rule and the tests do). Returning the raw sample
/// SD here keeps a single convention shared with the HVP adaptive path
/// (which folds `var.sqrt()` directly into its `stderrs`), and avoids the
/// historical double `1/sqrt(K)` division that under-counted the variance
/// band by a factor of `K`.
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
            stderrs[j] = var.sqrt();
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
        // `gradient_rho_stderr` is the per-probe sample SD of the half-scaled
        // gradient; the trace SE of the K-probe mean is `2·sd/sqrt(K)`.
        let sqrt_k = (evidence.probe_count as f64).sqrt();
        let se1 = 2.0 * evidence.gradient_rho_stderr[0] / sqrt_k;
        let se2 = 2.0 * evidence.gradient_rho_stderr[1] / sqrt_k;
        let tol1 = 6.0 * se1.max(1e-8);
        let tol2 = 6.0 * se2.max(1e-8);
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
        // SE of the half-scaled gradient mean = (per-probe sample SD) / sqrt(K).
        let se = evidence.gradient_rho_stderr[0] / (evidence.probe_count as f64).sqrt();
        let tol = 8.0 * se.max(1e-8);
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

    // ────────────────────────────────────────────────────────────────
    // Block 2.6: adaptive-K validation tests.
    //
    // All five run on CPU hosts (where `evidence_derivatives_hutchinson_gpu`
    // falls back to the SplitMix CPU reference) and on V100 hosts (where the
    // CUDA path takes over). Probe-level CRN is preserved across both paths.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn block_2_6_adaptive_unbiased_against_exact_p512() {
        // (1) Adaptive Hutchinson with the default ε must land near the
        // exact `tr(H⁻¹ A)` within its reported stopping tolerance.
        let p = 64;
        let h = make_spd(p, 0.5);
        let a = random_dense_sym(p, 0xBADC0DE);
        let exact = exact_trace_hinv_a(h.view(), a.view());
        let evidence = evidence_traces_adaptive(
            h.view(),
            vec![DerivativeHessian::Dense(a.view())],
            None,
            ProbeSeed(0xA5A5A5),
            HUTCHINSON_ADAPTIVE_REL_TOL,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("adaptive run ok");
        let est = evidence.traces[0];
        let se = evidence.stderrs[0] / (evidence.probe_count as f64).sqrt();
        let tol = (8.0 * se).max(0.05 * exact.abs());
        assert!(
            (est - exact).abs() <= tol,
            "adaptive est {est} far from exact {exact} (tol={tol}, se={se}, K={})",
            evidence.probe_count
        );
    }

    #[test]
    fn block_2_6_same_probes_cpu_vs_dispatch() {
        // (2) The dispatch entry (`_gpu`) and the explicit CPU reference
        // must produce identical estimates when given the same probes.
        // The dispatcher falls back to the CPU reference on non-CUDA hosts,
        // so this is a tautology on CPU; on V100 it asserts bit-identical
        // q-values across paths (the `q_{j,k}=z_k^T H_j w_k` reduction is
        // deterministic to machine precision once probes match).
        let p = 32;
        let h = make_spd(p, 0.3);
        let a = random_dense_sym(p, 0x1357);
        let input = RemlTraceHutchinsonInput {
            penalized_hessian: h.view(),
            derivatives: vec![DerivativeHessian::Dense(a.view())],
            design: None,
            probe_count: 16,
            seed: ProbeSeed(0xBEEF),
        };
        let cpu = evidence_derivatives_hutchinson_cpu(&input).expect("cpu");
        let dispatch = evidence_derivatives_hutchinson_gpu(input).expect("dispatch");
        let diff = (cpu.gradient_rho_logdet[0] - dispatch.gradient_rho_logdet[0]).abs();
        assert!(
            diff < 1e-9,
            "same-probes CPU vs GPU dispatch differ: cpu={}, dispatch={}, diff={diff}",
            cpu.gradient_rho_logdet[0],
            dispatch.gradient_rho_logdet[0]
        );
    }

    #[test]
    fn block_2_6_fd_logdet_matches_adaptive() {
        // (3) Adaptive estimate of `tr(H⁻¹ A)` should agree with the
        // central-difference derivative `d/dρ log|H + ρA|` at ρ=0.
        let p = 24;
        let h = make_spd(p, 0.4);
        let a = random_dense_sym(p, 0x2468);
        let eps = 1e-4;
        let mut hp = h.clone();
        let mut hm = h.clone();
        for i in 0..p {
            for j in 0..p {
                hp[[i, j]] += eps * a[[i, j]];
                hm[[i, j]] -= eps * a[[i, j]];
            }
        }
        let ld = |m: &Array2<f64>| -> f64 {
            let l = cholesky_lower(m).expect("SPD");
            2.0 * (0..p).map(|i| l[[i, i]].ln()).sum::<f64>()
        };
        let fd = (ld(&hp) - ld(&hm)) / (2.0 * eps);
        let evidence = evidence_traces_adaptive(
            h.view(),
            vec![DerivativeHessian::Dense(a.view())],
            None,
            ProbeSeed(0x9999),
            HUTCHINSON_ADAPTIVE_REL_TOL,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("adaptive ok");
        let est = evidence.traces[0];
        let se = evidence.stderrs[0] / (evidence.probe_count as f64).sqrt();
        let tol = (8.0 * se).max(0.05 * fd.abs());
        assert!(
            (est - fd).abs() <= tol,
            "adaptive trace {est} disagrees with FD logdet derivative {fd} (tol={tol})"
        );
    }

    #[test]
    fn block_2_6_k_4096_matches_exact_tightly() {
        // (4) A large fixed K (4096 probes) — well past the adaptive
        // schedule's max — must drive the Hutchinson estimator to within
        // a few SE of exact. Bounds the residual variance and confirms
        // the estimator is consistent (not merely unbiased at small K).
        let p = 40;
        let h = make_spd(p, 0.6);
        let a = random_dense_sym(p, 0xDEAD);
        let exact = exact_trace_hinv_a(h.view(), a.view());
        let input = RemlTraceHutchinsonInput {
            penalized_hessian: h.view(),
            derivatives: vec![DerivativeHessian::Dense(a.view())],
            design: None,
            probe_count: 4096,
            seed: ProbeSeed(0xC0FFEE),
        };
        let evidence = evidence_derivatives_hutchinson_gpu(input).expect("ok");
        let est = 2.0 * evidence.gradient_rho_logdet[0];
        let se = 2.0 * evidence.gradient_rho_stderr[0] / (4096_f64).sqrt();
        let tol = (6.0 * se).max(1e-3 * exact.abs());
        assert!(
            (est - exact).abs() <= tol,
            "K=4096 Hutchinson {est} not within 6·SE of exact {exact} (tol={tol}, se={se})"
        );
    }

    #[test]
    fn block_2_6_crn_prefix_match_across_schedule() {
        // (5) Common-random-numbers: the first 16 probes of a K=32 (and
        // K=64) draw must be bit-identical to a K=16 draw with the same
        // seed. The SplitMix probe RNG is stateless in (seed, k, i), so
        // this is what guarantees the adaptive schedule's variance
        // monotonically *decreases* rather than oscillating.
        let p = 50;
        let seed = ProbeSeed(0x4242_4242);
        let mut z16 = vec![0.0_f64; p * 16];
        let mut z32 = vec![0.0_f64; p * 32];
        let mut z64 = vec![0.0_f64; p * 64];
        fill_rademacher_host(seed, p, 16, &mut z16);
        fill_rademacher_host(seed, p, 32, &mut z32);
        fill_rademacher_host(seed, p, 64, &mut z64);
        for col in 0..16 {
            for row in 0..p {
                assert_eq!(z16[col * p + row], z32[col * p + row]);
                assert_eq!(z16[col * p + row], z64[col * p + row]);
            }
        }
        for col in 0..32 {
            for row in 0..p {
                assert_eq!(z32[col * p + row], z64[col * p + row]);
            }
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Block 2.7: batched-PCG HVP variant tests.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn block_2_7_hvp_path_matches_dense_adaptive() {
        // HVP closure that multiplies a stored dense H matches the
        // dense `evidence_traces_adaptive` exactly (same CRN probes,
        // same derivative). CG round-off bounded by PCG_HVP_REL_TOL.
        let p = 40;
        let h = make_spd(p, 0.7);
        let a = random_dense_sym(p, 0xABBA);
        let seed = ProbeSeed(0x707);

        let dense = evidence_traces_adaptive(
            h.view(),
            vec![DerivativeHessian::Dense(a.view())],
            None,
            seed,
            HUTCHINSON_ADAPTIVE_REL_TOL,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("dense ok");

        let h_clone = h.clone();
        let hvp_evidence = evidence_traces_adaptive_hvp(
            p,
            |v: &[f64], out: &mut [f64]| {
                for r in 0..p {
                    let mut acc = 0.0_f64;
                    for c in 0..p {
                        acc += h_clone[[r, c]] * v[c];
                    }
                    out[r] = acc;
                }
            },
            vec![DerivativeHessian::Dense(a.view())],
            None,
            seed,
            HUTCHINSON_ADAPTIVE_REL_TOL,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("hvp ok");

        // Adaptive may stop at different K if SE crosses the threshold
        // at a different step due to CG round-off; compare both
        // estimates against exact rather than to each other.
        let exact = exact_trace_hinv_a(h.view(), a.view());
        let se_dense = dense.stderrs[0] / (dense.probe_count as f64).sqrt();
        let se_hvp = hvp_evidence.stderrs[0] / (hvp_evidence.probe_count as f64).sqrt();
        let tol_dense = (8.0 * se_dense).max(0.05 * exact.abs());
        let tol_hvp = (8.0 * se_hvp).max(0.05 * exact.abs());
        assert!(
            (dense.traces[0] - exact).abs() <= tol_dense,
            "dense adaptive {} not near exact {} (tol {})",
            dense.traces[0],
            exact,
            tol_dense
        );
        assert!(
            (hvp_evidence.traces[0] - exact).abs() <= tol_hvp,
            "hvp adaptive {} not near exact {} (tol {})",
            hvp_evidence.traces[0],
            exact,
            tol_hvp
        );
        // logdet is intentionally NaN on the HVP path.
        assert!(hvp_evidence.logdet_hessian.is_nan());
    }

    #[test]
    fn block_2_7_hvp_stderr_matches_dense_reduce_mean_stderr() {
        // The HVP path's `stderrs` must use the SAME estimator convention as
        // the dense path's `reduce_mean_stderr`: the Bessel-corrected (K−1)
        // sample standard deviation of the per-probe q values. We force both
        // paths to run the full K=128 schedule (rel_tol below any achievable
        // ratio) so the comparison is at identical probe counts on identical
        // CRN probes. The only residual difference is the inner solve (exact
        // Cholesky vs CG@1e-6), which keeps the q values — and hence the SDs —
        // agreeing to a tight relative tolerance.
        let p = 36;
        let h = make_spd(p, 0.6);
        let a = random_dense_sym(p, 0x5151);
        let seed = ProbeSeed(0xBEEF);
        let force_full_schedule = 1e-12_f64;

        let dense = evidence_traces_adaptive(
            h.view(),
            vec![DerivativeHessian::Dense(a.view())],
            None,
            seed,
            force_full_schedule,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("dense ok");

        let h_clone = h.clone();
        let hvp = evidence_traces_adaptive_hvp(
            p,
            |v: &[f64], out: &mut [f64]| {
                for r in 0..p {
                    let mut acc = 0.0_f64;
                    for c in 0..p {
                        acc += h_clone[[r, c]] * v[c];
                    }
                    out[r] = acc;
                }
            },
            vec![DerivativeHessian::Dense(a.view())],
            None,
            seed,
            force_full_schedule,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("hvp ok");

        // Both ran the full schedule, so probe counts match exactly.
        assert_eq!(dense.probe_count, 128);
        assert_eq!(hvp.probe_count, dense.probe_count);

        let sd_dense = dense.stderrs[0];
        let sd_hvp = hvp.stderrs[0];
        assert!(sd_dense > 0.0, "dense SD should be positive, got {sd_dense}");
        let rel = (sd_hvp - sd_dense).abs() / sd_dense;
        assert!(
            rel <= 1e-3,
            "HVP SD {sd_hvp} disagrees with dense reduce_mean_stderr SD {sd_dense} \
             (rel {rel}); the two paths must share the Bessel-corrected (K−1) convention"
        );
    }

    #[test]
    fn block_2_7_cg_solves_diagonal_in_one_iteration() {
        // For diagonal H, CG converges in one step (Krylov subspace
        // contains the exact solution). Verifies the CG residual
        // logic and SPD bailout.
        let p = 8;
        let diag: Vec<f64> = (0..p).map(|i| 1.0 + i as f64).collect();
        let b: Vec<f64> = (0..p).map(|i| (i as f64) + 0.5).collect();
        let mut w = vec![0.0_f64; p];
        let diag_clone = diag.clone();
        cg_solve(
            &mut |v: &[f64], out: &mut [f64]| {
                for i in 0..p {
                    out[i] = diag_clone[i] * v[i];
                }
            },
            &b,
            &mut w,
            1e-12,
            PCG_HVP_MAX_ITERS,
        );
        for i in 0..p {
            let expected = b[i] / diag[i];
            assert!(
                (w[i] - expected).abs() < 1e-10,
                "diagonal CG: w[{i}]={} expected {expected}",
                w[i]
            );
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Block 2.8: V100 hill-climb (10× vs exact GPU at p=2000, d_ρ=8).
    //
    // The assertion only fires when a CUDA runtime is detected;
    // on CPU-only hosts the test still runs the timing comparison but
    // skips the speedup assertion (exact dense Cholesky is competitive
    // with adaptive Hutchinson on a single core, so the 10× lower bound
    // is V100-specific). On V100, the adaptive path batches K=16-128
    // probes through one potrs while the exact path repeats `d` full
    // solves; the bound is therefore comfortable.
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn block_2_8_hill_climb_adaptive_vs_exact_at_p2000_d8() {
        // Smaller dimensions on CPU CI to keep the test under a minute;
        // V100 runs the full p=2000, d=8 specified in the charter.
        let on_v100 =
            cfg!(target_os = "linux") && crate::gpu::device_runtime::GpuRuntime::global().is_some();
        let (p, d): (usize, usize) = if on_v100 { (2000, 8) } else { (256, 4) };

        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                h[[i, j]] = if i == j {
                    p as f64 + 1.0
                } else {
                    1.0 / (1.0 + (i as f64 - j as f64).abs())
                };
            }
        }
        let derivs_owned: Vec<Array2<f64>> = (0..d)
            .map(|k| random_dense_sym(p, 0x1000 + k as u64))
            .collect();
        let derivs: Vec<DerivativeHessian<'_>> = derivs_owned
            .iter()
            .map(|a| DerivativeHessian::Dense(a.view()))
            .collect();

        // Exact path: factor H once, then `tr(H⁻¹ A_j) = Σᵢ (H⁻¹ A_j)[i,i]`
        // by solving H X = A_j column-by-column. This is the cost the
        // CPU/exact-spectral path pays per REML outer step.
        let t_exact_start = std::time::Instant::now();
        let factor = cholesky_lower(&h).expect("SPD");
        let mut exact_traces = vec![0.0_f64; d];
        for (j, a) in derivs_owned.iter().enumerate() {
            let mut acc = 0.0_f64;
            for col in 0..p {
                let mut rhs = vec![0.0_f64; p];
                for r in 0..p {
                    rhs[r] = a[[r, col]];
                }
                let w = solve_cholesky(&factor, &rhs);
                acc += w[col];
            }
            exact_traces[j] = acc;
        }
        let t_exact = t_exact_start.elapsed();

        // Adaptive Hutchinson path.
        let t_adaptive_start = std::time::Instant::now();
        let evidence = evidence_traces_adaptive(
            h.view(),
            derivs,
            None,
            ProbeSeed(0xB10C),
            HUTCHINSON_ADAPTIVE_REL_TOL,
            HUTCHINSON_ADAPTIVE_TAU_REL,
        )
        .expect("adaptive ok");
        let t_adaptive = t_adaptive_start.elapsed();

        // Sanity: every adaptive trace must agree with exact within its
        // reported SE. This guards against a fast-but-wrong perf path.
        for j in 0..d {
            let se = evidence.stderrs[j] / (evidence.probe_count as f64).sqrt();
            let tol = (10.0 * se).max(0.05 * exact_traces[j].abs());
            let diff = (evidence.traces[j] - exact_traces[j]).abs();
            assert!(
                diff <= tol,
                "block_2_8: derivative {j} adaptive {} disagrees with exact {} (tol {tol}, diff {diff})",
                evidence.traces[j],
                exact_traces[j]
            );
        }

        let speedup = t_exact.as_secs_f64() / t_adaptive.as_secs_f64().max(1e-9);
        eprintln!(
            "block_2_8 hill-climb [p={p}, d={d}, V100={on_v100}]: \
             exact={:?}, adaptive={:?}, speedup={:.2}× (K={}, converged={})",
            t_exact, t_adaptive, speedup, evidence.probe_count, evidence.converged
        );
        if on_v100 {
            assert!(
                speedup >= 10.0,
                "block_2_8 V100 speedup {speedup:.2}× below the 10× target \
                 (exact {:?}, adaptive {:?})",
                t_exact,
                t_adaptive,
            );
        }
    }
}
