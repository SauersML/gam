//! Survival marginal-slope FLEX GPU row primitives.
//!
//! The kernels here are domain math for `SurvivalMarginalSlope`; CUDA runtime,
//! memory, and policy ownership stays in [`gam_gpu`].

use ndarray::{Array1, Array2};

use gam_gpu::gpu_error::GpuError;
use gam_gpu::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaModule;

/// Decide whether the survival-flex GPU row primary path is eligible for
/// this fit's `(n, r)`.  `r == 0` (no primary jets to process) and below
/// the runtime row-kernel threshold force CPU.
#[must_use]
pub fn row_primary_hessian_decision(n: usize, r: usize) -> GpuDecision {
    let large_enough = gam_gpu::device_runtime::GpuRuntime::global()
        .map(|runtime| n >= runtime.policy().row_kernel_min_n && r > 0)
        .unwrap_or(false);
    decide(
        GpuKernel::MarginalSlopeRows,
        gam_gpu::GpuEligibility::from_flags(SurvivalFlexGpuBackend::compiled(), large_enough),
    )
}

// ────────────────────────────────────────────────────────────────────────
// Per-fit input descriptor.
// ────────────────────────────────────────────────────────────────────────

/// Inputs threaded into the three survival-flex entry points.  The struct
/// is intentionally additive: later steps append optional fields (per-row
/// time-design pointers, score-warp basis, link-deviation basis, cell
/// breakpoint tables, warm-start intercept slabs, …) without breaking the
/// Step-1 callers that only inspect `n`, `r`, `p`, `score_dim`, the rigid
/// row scalars (`q_0`, `q_1`, `q̇_1`, `g`, `z`) and the event/weight columns.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalFlexGpuRowInputs<'a> {
    /// Number of observations.
    pub n: usize,
    /// Primary local dimension (q_0 + q_1 + q̇_1 + g + score-warp + link-dev).
    pub r: usize,
    /// Total joint-parameter dimension `p` (sum of all block sizes).
    pub p: usize,
    /// Latent-score dimension `K`.  Step 1 + 6 require `K == 1` (scalar
    /// score); `K > 1` is an unsupported shape and the entry points return
    /// `Ok(None)` for it.
    pub score_dim: usize,
    /// Current β coefficient vector, length `p`, in joint-block order.
    pub beta: &'a [f64],
    /// Per-row entry quantile `q_0`, length `n`.
    pub q0: &'a [f64],
    /// Per-row exit quantile `q_1`, length `n`.
    pub q1: &'a [f64],
    /// Per-row exit-rate jacobian `q̇_1`, length `n`.  Rows with
    /// `q̇_1 < derivative_guard` (or non-finite) are rejected by the row
    /// primitive in line with the CPU `survival_derivative_guard_violated`.
    pub qd1: &'a [f64],
    /// Per-row latent score `z`, length `n`.  Scalar (K = 1) only in
    /// Step 1; the vector path lands in Step 4.
    pub z: &'a [f64],
    /// Per-row raw log-slope `g`, length `n`.
    pub g: &'a [f64],
    /// Observation weights, length `n`.
    pub weights: &'a [f64],
    /// Event indicator `d ∈ {0,1}`, length `n`.
    pub event: &'a [f64],
    /// `derivative_guard` for the monotonicity reject (matches CPU).
    pub derivative_guard: f64,
    /// `probit_scale` ≡ probit frailty scale `s` (matches CPU constant).
    pub probit_scale: f64,
}

impl<'a> SurvivalFlexGpuRowInputs<'a> {
    /// Shape-check every input array up front.  Kept on the struct so all
    /// three entry points reuse the same validation surface.
    fn validate(&self) -> Result<(), GpuError> {
        let n = self.n;
        let len_check = |label: &str, len: usize| -> Result<(), GpuError> {
            if len != n {
                return Err(GpuError::DriverCallFailed {
                    reason: format!("survival_flex inputs: {label}.len()={len} != n={n}"),
                });
            }
            Ok(())
        };
        if self.beta.len() != self.p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex inputs: beta.len()={} != p={}",
                    self.beta.len(),
                    self.p
                ),
            });
        }
        if self.r > self.p {
            return Err(GpuError::DriverCallFailed {
                reason: format!("survival_flex inputs: r={} exceeds p={}", self.r, self.p),
            });
        }
        len_check("q0", self.q0.len())?;
        len_check("q1", self.q1.len())?;
        len_check("qd1", self.qd1.len())?;
        len_check("z", self.z.len())?;
        len_check("g", self.g.len())?;
        len_check("weights", self.weights.len())?;
        len_check("event", self.event.len())?;
        if !(self.derivative_guard.is_finite() && self.derivative_guard > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex inputs: derivative_guard must be positive and finite, got {}",
                    self.derivative_guard
                ),
            });
        }
        if !(self.probit_scale.is_finite() && self.probit_scale > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex inputs: probit_scale must be positive and finite, got {}",
                    self.probit_scale
                ),
            });
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────
// Process-wide backend (NVRTC compile, device arena, kernel launch).
// ────────────────────────────────────────────────────────────────────────

#[must_use]
pub struct SurvivalFlexGpuBackend {
    #[cfg(target_os = "linux")]
    inner: gam_gpu::backend_probe::CudaBackendContext,
}

impl SurvivalFlexGpuBackend {
    /// True when this build can host the survival-flex GPU backend.
    /// Compiled on Linux only (Block 8 host is V100 / Linux).
    pub const fn compiled() -> bool {
        cfg!(target_os = "linux")
    }

    /// Lazily initialise the process-wide CUDA backend.
    #[cfg(target_os = "linux")]
    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<SurvivalFlexGpuBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(Self::probe_linux)
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let parts = gam_gpu::backend_probe::probe_cuda_backend("survival_flex")?;
        Ok(SurvivalFlexGpuBackend {
            inner: gam_gpu::backend_probe::CudaBackendContext::from_parts(parts),
        })
    }

    /// Round-trip the arena.  Mirrors `bms_flex` so the V100 smoke test
    /// has the same surface across Block 8 / sibling backends.
    #[cfg(target_os = "linux")]
    pub fn arena_round_trip(&self, elements: usize) -> Result<usize, GpuError> {
        let mut guard = self
            .inner
            .arena
            .lock()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex arena mutex poisoned: {err}"),
            })?;
        let (bucket, slab) = guard.alloc(&self.inner.stream, elements, "survival_flex")?;
        guard.release(bucket, slab);
        Ok(bucket)
    }
}

// ────────────────────────────────────────────────────────────────────────
// Step 2 — survival-flex row-batched cubic-cell moment evaluator.
//
// Steps 3-6 need per-row derivative moments of the de-nested cubic
// correction `η(z) = c_0 + c_1·z + c_2·z² + c_3·z³` integrated against
// `exp(-q(z))` over each cell of the row's partition.  The CPU side
// builds these partitions via
// `survival::marginal_slope::denested_partition_cells` and then loops
// `evaluate_cell_moments` / `evaluate_cell_derivative_moments_uncached`
// per cell.
//
// This Step 2 wrapper takes a *flat* concatenation of per-row cells
// (with per-row start offsets), classifies them through the shared
// `cubic_cell::branch` classifier, and routes them in one shot through
// the existing GPU substrate (`cubic_cell::device::try_device_moments`):
//
//   * NonAffineFinite cells → 384-pt Gauss-Legendre warp-cooperative
//     kernel (the substrate's primary device path).
//   * Affine / AffineTail cells → CPU closed-form `T_n` recurrence
//     (substrate falls back per-cell, no warp divergence on GPU).
//
// Output is row-major `[total_cells, max_degree + 1]` moments plus a
// parallel status byte array and a `row_offsets` lookup so Step 3/4
// callers can index `row i → cells[row_offsets[i] .. row_offsets[i+1]]`.
//
// The wrapper does *not* duplicate any substrate logic: classification,
// device dispatch, status accounting all live in `cubic_cell`.  Its only
// jobs are (a) build the SoA cell list survival-flex needs and (b)
// expose a survival-shaped error surface.
// ────────────────────────────────────────────────────────────────────────

/// Per-row partition layout: a flat list of `(left, right, c0, c1, c2, c3)`
/// quadruples plus a `row_offsets` array of length `n + 1` so that
/// row `i`'s cells live at indices `row_offsets[i] .. row_offsets[i+1]`.
///
/// The survival-flex CPU path produces this layout naturally — see
/// `survival::marginal_slope::denested_partition_cells`.  Callers can
/// flatten the per-row `Vec<DenestedPartitionCell>` lists into this
/// shape with a single pass.
#[derive(Clone, Debug)]
pub struct SurvivalFlexRowCellsBatch<'a> {
    /// Total cell count = sum of per-row partition lengths.
    pub n_cells: usize,
    /// Number of rows (logical observations).
    pub n_rows: usize,
    /// Highest moment degree to evaluate, in `0..=24`. The full CPU
    /// bidirectional survival path now needs degree 27, so callers must not
    /// route that path through this degree-24 substrate until the kernel bound
    /// is raised. Degree 9 is sufficient for value-only evaluations.
    pub max_degree: usize,
    /// Flat SoA cell quadruples, length `n_cells` each.
    pub left: &'a [f64],
    pub right: &'a [f64],
    pub c0: &'a [f64],
    pub c1: &'a [f64],
    pub c2: &'a [f64],
    pub c3: &'a [f64],
    /// Length `n_rows + 1`; row `i` owns cells `row_offsets[i] .. row_offsets[i+1]`.
    /// `row_offsets[0] == 0`, `row_offsets[n_rows] == n_cells`.
    pub row_offsets: &'a [usize],
}

/// Row-batched moment evaluation status.
#[derive(Clone, Debug)]
pub struct SurvivalFlexRowMoments {
    /// One status byte per cell. Values match
    /// `cubic_cell::CubicCellMomentStatus` byte-for-byte.
    pub status: Vec<u8>,
}

impl<'a> SurvivalFlexRowCellsBatch<'a> {
    /// Shape-validate the batch.  Returns a `GpuError::DriverCallFailed`
    /// with a message naming the failing invariant so callers get a
    /// single error surface across the wrapper.
    fn validate(&self) -> Result<(), GpuError> {
        let nc = self.n_cells;
        let invariants: [(&str, usize); 6] = [
            ("left", self.left.len()),
            ("right", self.right.len()),
            ("c0", self.c0.len()),
            ("c1", self.c1.len()),
            ("c2", self.c2.len()),
            ("c3", self.c3.len()),
        ];
        for (label, len) in invariants {
            if len != nc {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex row-cells batch: {label}.len()={len} != n_cells={nc}"
                    ),
                });
            }
        }
        if self.row_offsets.len() != self.n_rows + 1 {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: row_offsets.len()={} != n_rows+1={}",
                    self.row_offsets.len(),
                    self.n_rows + 1
                ),
            });
        }
        if !self.row_offsets.is_empty()
            && (self.row_offsets[0] != 0 || self.row_offsets[self.n_rows] != nc)
        {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: row_offsets must start at 0 and end at \
                     n_cells={nc}, got [{}, …, {}]",
                    self.row_offsets[0], self.row_offsets[self.n_rows]
                ),
            });
        }
        for i in 0..self.n_rows {
            if self.row_offsets[i] > self.row_offsets[i + 1] {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex row-cells batch: row_offsets not monotone at i={i} \
                         ({} > {})",
                        self.row_offsets[i],
                        self.row_offsets[i + 1]
                    ),
                });
            }
        }
        if self.max_degree > crate::gpu_kernels::cubic_cell::MAX_SUPPORTED_DEGREE {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex row-cells batch: max_degree={} exceeds substrate \
                     MAX_SUPPORTED_DEGREE={}",
                    self.max_degree,
                    crate::gpu_kernels::cubic_cell::MAX_SUPPORTED_DEGREE
                ),
            });
        }
        Ok(())
    }
}

/// Evaluate the derivative moments for every cell in `batch`.
///
/// Routes through the shared `cubic_cell` substrate so the survival-flex
/// path inherits the substrate's pre-classifier + 384-pt GL warp kernel
/// without any survival-specific kernel code.  The substrate itself
/// returns `Ok(None)` only on empty input; we surface that as
/// `Ok(None)` too so the dispatcher can short-circuit downstream Step 3
/// solves on rows that have no cells.
///
/// The substrate's host residency means this function works on every
/// platform: on Linux+CUDA the NonAffineFinite bucket runs on the
/// device, on macOS / CPU-only Linux every cell falls through to the
/// CPU evaluator that is the parity reference for the device kernel.
pub fn try_row_batched_cell_moments(
    batch: SurvivalFlexRowCellsBatch<'_>,
) -> Result<Option<SurvivalFlexRowMoments>, GpuError> {
    batch.validate()?;
    if batch.n_cells == 0 {
        return Ok(None);
    }

    // Build the substrate view in one pass.  The classification (Affine /
    // NonAffineFinite / AffineTail) is shared host code so doing it once
    // here lines up with what the substrate would re-derive internally;
    // we lift it out only because the substrate's `HostView` insists on
    // a `branches: &[GpuCellBranchTag]` matching the cell list.
    let mut cells = Vec::with_capacity(batch.n_cells);
    let mut branches = Vec::with_capacity(batch.n_cells);
    let mut prelim_status = Vec::with_capacity(batch.n_cells);
    for k in 0..batch.n_cells {
        let cell = crate::gpu_kernels::cubic_cell::GpuDenestedCubicCell {
            left: batch.left[k],
            right: batch.right[k],
            c0: batch.c0[k],
            c1: batch.c1[k],
            c2: batch.c2[k],
            c3: batch.c3[k],
        };
        match crate::gpu_kernels::cubic_cell::branch::classify_cell_for_gpu(cell) {
            Ok(tag) => {
                cells.push(cell);
                branches.push(tag);
                prelim_status.push(crate::gpu_kernels::cubic_cell::CubicCellMomentStatus::Ok as u8);
            }
            Err(code) => {
                // Substrate would also reject this cell.  Keep a placeholder
                // in the input so per-cell indexing stays aligned; the
                // substrate will set the matching status code itself.
                cells.push(cell);
                // The substrate's classifier runs again and writes the
                // authoritative status; any tag here is fine because the
                // substrate's "host_tag != caller_tag" path also routes to
                // an error code, and the substrate's *own* classification
                // is the one that wins.  Use the cheapest stable tag.
                branches.push(crate::gpu_kernels::cubic_cell::GpuCellBranchTag::Affine);
                prelim_status.push(code as u8);
            }
        }
    }

    let view = crate::gpu_kernels::cubic_cell::CubicCellDerivativeMomentHostView {
        cells: &cells,
        branches: &branches,
        max_degree: batch.max_degree,
        residency: crate::gpu_kernels::cubic_cell::CubicCellMomentResidency::Host,
    };
    let out = crate::gpu_kernels::cubic_cell::try_build_cubic_cell_derivative_moments(view)?
        .ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex row-cells batch: substrate returned None for n_cells={} > 0 \
                 (unexpected)",
                batch.n_cells
            ),
        })?;

    let mut status = match out {
        crate::gpu_kernels::cubic_cell::CubicCellDerivativeMomentOutput::Host { status } => status,
        #[cfg(target_os = "linux")]
        crate::gpu_kernels::cubic_cell::CubicCellDerivativeMomentOutput::Device { .. } => {
            return Err(GpuError::DriverCallFailed {
                reason: "survival_flex row-cells batch: substrate returned device-resident output \
                         but the survival-flex host pipeline consumes Host residency only"
                    .to_string(),
            });
        }
    };

    // Cells we pre-rejected (`prelim_status != Ok`) get a status code
    // from us if the substrate left them as Ok (it won't, because it
    // re-runs the classifier — but keeping this explicit guards against
    // a future substrate that trusts caller tags).
    for k in 0..batch.n_cells {
        if prelim_status[k] != crate::gpu_kernels::cubic_cell::CubicCellMomentStatus::Ok as u8
            && status[k] == crate::gpu_kernels::cubic_cell::CubicCellMomentStatus::Ok as u8
        {
            status[k] = prelim_status[k];
        }
    }

    Ok(Some(SurvivalFlexRowMoments { status }))
}

// ────────────────────────────────────────────────────────────────────────
// Step 3 — device monotone-root intercept solve.
//
// The flex calibration step solves `F(a) = ⟨Φ(-η(z;a))⟩ - Φ(-q) = 0`
// once per row.  The CPU side runs `monotone_root::solve_monotone_root_detailed`
// (`families::survival::marginal_slope.rs:5363`).  Step 3 ports the
// control flow (Newton probe → bracket expansion → bisection +
// safeguarded Halley/Newton refinement) into an NVRTC kernel so every
// row solves in parallel.
//
// The control-flow kernel is parameterised over the F-evaluator: Step 4
// substitutes the real survival calibration (which needs the cell
// moments from Step 2) by adding the relevant evaluator branch to the
// NVRTC source.  Step 3 ships and tests against an analytic evaluator
//
//     F(a)   = alpha · exp(beta · a) + gamma
//     F'(a)  = alpha · beta · exp(beta · a)
//     F''(a) = alpha · beta² · exp(beta · a)
//
// whose closed-form root `a* = ln(-gamma/alpha) / beta` lets the parity
// test verify Newton probe + bracket expansion + Halley/Newton refine
// down to the CPU `solve_monotone_root_detailed` tolerance.
//
// Warm-start design: per-row arrays `a_entry[row]`, `a_exit[row]` carry
// the previous-iter intercept solution.  The kernel reads them, runs
// the solver, and writes back the converged root *plus* the abs-deriv
// and residual that downstream Step 4 IFT corrections need.
//
// Bracket safety: matches the CPU cap exactly —
// `step_cap = max(1e6, 1024·(1+|a_init|))` — and the same
// `step_sign = -sign(f·F')` rule.  Convergence on
// `|F| ≤ tol` *or* bracket width `≤ tol·(1+|lo|+|hi|)`, identical to
// the CPU loop.
// ────────────────────────────────────────────────────────────────────────

/// Per-row inputs for the Step 3 device intercept solve.  Borrows
/// host-side warm-start arrays + the per-row evaluator coefficients.
/// The Step-4 wiring replaces `(alpha, beta, gamma)` with the real
/// survival calibration evaluator; the Step-3 test path uses these
/// directly for closed-form parity against the CPU monotone-root
/// solver.
#[derive(Clone, Debug)]
pub struct SurvivalFlexInterceptSolveInputs<'a> {
    /// Number of rows.
    pub n: usize,
    /// Warm-start seed per row.  For the rigid (flex=false) fallback the
    /// CPU side uses `a_seed = q · √(1 + (s·g)²) / s` — the caller is
    /// expected to provide either that rigid seed *or* the previous-iter
    /// converged root (whichever is fresher).
    pub a_warm: &'a [f64],
    /// Analytic evaluator coefficients per row.  Step 4 swaps this out
    /// for the real survival calibration evaluator inputs.
    pub alpha: &'a [f64],
    pub beta: &'a [f64],
    pub gamma: &'a [f64],
    /// `|F| ≤ convergence_tol` and bracket width `≤ tol·(1+|lo|+|hi|)`
    /// both stop the loop.  Matches the CPU contract.
    pub convergence_tol: f64,
    /// Bracket-expansion iteration cap.  CPU side uses 64 for survival.
    pub max_bracket_iters: u32,
    /// Refinement iteration cap.  CPU side uses 64.
    pub max_refine_iters: u32,
}

/// Step 3 per-row output.
#[derive(Clone, Debug)]
pub struct SurvivalFlexInterceptSolveOutputs {
    /// Per-row exit status:
    ///   0 — converged to `|F| ≤ tol`
    ///   1 — exited on bracket-width contraction (acceptable; root within tol)
    ///   2 — Newton probe degenerate (F'(a_warm) zero / non-finite)
    ///   3 — bracket search exhausted (no sign change after `max_bracket_iters`)
    ///   4 — refine loop exhausted without bracket/residual convergence
    ///   5 — non-finite produced by the evaluator (e.g. overflow)
    pub status: Vec<u8>,
}

impl<'a> SurvivalFlexInterceptSolveInputs<'a> {
    fn validate(&self) -> Result<(), GpuError> {
        let n = self.n;
        let lens: [(&str, usize); 4] = [
            ("a_warm", self.a_warm.len()),
            ("alpha", self.alpha.len()),
            ("beta", self.beta.len()),
            ("gamma", self.gamma.len()),
        ];
        for (label, len) in lens {
            if len != n {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "survival_flex intercept-solve inputs: {label}.len()={len} != n={n}"
                    ),
                });
            }
        }
        if !(self.convergence_tol.is_finite() && self.convergence_tol > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex intercept-solve inputs: convergence_tol must be positive \
                     finite, got {}",
                    self.convergence_tol
                ),
            });
        }
        if self.max_bracket_iters == 0 || self.max_refine_iters == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "survival_flex intercept-solve inputs: iter caps must be positive, got \
                     bracket={} refine={}",
                    self.max_bracket_iters, self.max_refine_iters
                ),
            });
        }
        Ok(())
    }
}

/// CPU oracle for the Step 3 intercept solve.  Drives the existing
/// `families::monotone_root::solve_monotone_root_detailed` against the
/// analytic evaluator (`alpha · exp(beta · a) + gamma`) so the device
/// kernel can be checked element-wise.  Returns the same output layout
/// as the device kernel.
///
/// Status codes match the device kernel's enumeration (0 converged,
/// 2 degenerate-derivative, 3 bracket-exhausted, 5 non-finite).  The
/// CPU solver collapses "Halley-only convergence" and "bisection-only
/// convergence" into a single status `0`, so the device parity bar is
/// status-equal *plus* numerical equality of `a_root`, `abs_deriv`,
/// `residual` at the CPU tolerance.
pub fn cpu_oracle_intercept_solve(
    inputs: &SurvivalFlexInterceptSolveInputs<'_>,
) -> SurvivalFlexInterceptSolveOutputs {
    use crate::monotone_root::{MonotoneRootError, solve_monotone_root_detailed};
    let mut status = vec![0u8; inputs.n];
    for row in 0..inputs.n {
        let alpha = inputs.alpha[row];
        let beta = inputs.beta[row];
        let gamma = inputs.gamma[row];
        let a_warm = inputs.a_warm[row];
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            let e = (beta * a).exp();
            if !e.is_finite() {
                return Err(format!("overflow at a={a}"));
            }
            let f = alpha * e + gamma;
            let fp = alpha * beta * e;
            let fpp = alpha * beta * beta * e;
            Ok((f, fp, fpp))
        };
        match solve_monotone_root_detailed(
            eval,
            a_warm,
            "survival_flex_intercept_oracle",
            inputs.convergence_tol,
            inputs.max_bracket_iters as usize,
            inputs.max_refine_iters as usize,
        ) {
            Ok(sol) => {
                status[row] = if sol.residual.abs() <= inputs.convergence_tol {
                    0
                } else {
                    1
                };
            }
            Err(MonotoneRootError::DegenerateDerivative { .. }) => {
                status[row] = 2;
            }
            Err(MonotoneRootError::BracketingExhausted { .. }) => {
                status[row] = 3;
            }
            Err(MonotoneRootError::RefinementDidNotConverge { .. }) => {
                status[row] = 4;
            }
            Err(_) => {
                status[row] = 5;
            }
        }
    }
    SurvivalFlexInterceptSolveOutputs { status }
}

// ────────────────────────────────────────────────────────────────────────
// NVRTC source — Step 3 (parameterised monotone root, analytic evaluator).
//
// One thread per row.  Identical control flow to the CPU
// `solve_monotone_root_detailed`:
//   * Up to 2 Newton probes from `a_warm[row]`.
//   * If un-converged, geometric step doubling (bracket phase) using
//     `step_sign = -sign(f · F')`, step_mag start = max(1.0, 0.25·(1+|a|)),
//     cap = max(1e6, 1024·(1+|a_warm|)).
//   * Phase 2: hybrid bisection / safeguarded Halley + Newton inside
//     the bracket; convergence on residual or bracket width.
//   * Best-of accounting for the residual, matching the CPU loop.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
const SURVIVAL_FLEX_INTERCEPT_SOLVE_SOURCE: &str = r#"
extern "C" __device__ __forceinline__ void
eval_F_analytic(double a, double alpha, double beta, double gamma,
                double *f, double *fp, double *fpp, int *ok) {
    double e = exp(beta * a);
    if (!isfinite(e)) { *f = 0.0; *fp = 0.0; *fpp = 0.0; *ok = 0; return; }
    *f   = alpha * e + gamma;
    *fp  = alpha * beta * e;
    *fpp = alpha * beta * beta * e;
    *ok  = 1;
}

extern "C" __global__ void survival_flex_intercept_solve(
    const double * __restrict__ a_warm_arr,
    const double * __restrict__ alpha_arr,
    const double * __restrict__ beta_arr,
    const double * __restrict__ gamma_arr,
    double                       convergence_tol,
    unsigned int                 max_bracket_iters,
    unsigned int                 max_refine_iters,
    int                          n,
    double * __restrict__        out_a_root,
    double * __restrict__        out_abs_deriv,
    double * __restrict__        out_residual,
    unsigned char * __restrict__ out_status
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double alpha = alpha_arr[row];
    double beta  = beta_arr[row];
    double gamma = gamma_arr[row];
    double a_init = a_warm_arr[row];

    double f, fp, fpp;
    int ok;
    eval_F_analytic(a_init, alpha, beta, gamma, &f, &fp, &fpp, &ok);
    if (!ok) { out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=nan(""); out_status[row]=5; return; }

    // Exact-root shortcut.
    if (fabs(f) <= convergence_tol) {
        double abs_d = fabs(fp);
        if (!isfinite(abs_d) || abs_d == 0.0) {
            out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=f; out_status[row]=2;
        } else {
            out_a_root[row]=a_init; out_abs_deriv[row]=abs_d; out_residual[row]=f; out_status[row]=0;
        }
        return;
    }

    if (!isfinite(fp) || fp == 0.0) {
        out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=nan(""); out_status[row]=2;
        return;
    }

    // --- Newton probe (≤ 2) ---
    double a = a_init;
    double f_init = f;
    double fp_init = fp;
    for (int probe = 0; probe < 2; ++probe) {
        if (fabs(f) <= convergence_tol) {
            double abs_d = fabs(fp);
            if (isfinite(abs_d) && abs_d != 0.0) {
                out_a_root[row]=a; out_abs_deriv[row]=abs_d; out_residual[row]=f; out_status[row]=0;
                return;
            }
            break;
        }
        if (!isfinite(fp) || fabs(fp) <= 1e-30) break;
        double step = -f / fp;
        if (!isfinite(step) || fabs(step) > 8.0 * (1.0 + fabs(a))) break;
        double cand = a + step;
        double f_c, fp_c, fpp_c; int ok_c;
        eval_F_analytic(cand, alpha, beta, gamma, &f_c, &fp_c, &fpp_c, &ok_c);
        if (!ok_c) break;
        if (fabs(f_c) <= convergence_tol) {
            double abs_d = fabs(fp_c);
            if (isfinite(abs_d) && abs_d != 0.0) {
                out_a_root[row]=cand; out_abs_deriv[row]=abs_d; out_residual[row]=f_c; out_status[row]=0;
                return;
            }
            break;
        }
        a = cand; f = f_c; fp = fp_c; fpp = fpp_c;
    }

    // --- Phase 1: bracket ---
    double step_sign = (f_init * fp_init < 0.0) ? 1.0 : -1.0;
    int f_init_neg = (f_init < 0.0) ? 1 : 0;
    double same_side = a_init;
    double step_mag = fmax(0.25 * (1.0 + fabs(a_init)), 1.0);
    double step_cap = fmax(1e6, 1024.0 * (1.0 + fabs(a_init)));

    int found_other = 0;
    double other = 0.0;
    for (unsigned int it = 0; it < max_bracket_iters; ++it) {
        double probe_pt = same_side + step_mag * step_sign;
        double f_probe, fp_probe, fpp_probe; int ok_probe;
        eval_F_analytic(probe_pt, alpha, beta, gamma, &f_probe, &fp_probe, &fpp_probe, &ok_probe);
        if (!ok_probe) break;
        int crossed = f_init_neg ? (f_probe >= 0.0) : (f_probe <= 0.0);
        if (crossed) { other = probe_pt; found_other = 1; break; }
        same_side = probe_pt;
        step_mag *= 2.0;
        if (step_mag > step_cap) break;
    }
    if (!found_other) {
        out_a_root[row]=a_init; out_abs_deriv[row]=0.0; out_residual[row]=nan(""); out_status[row]=3;
        return;
    }

    double neg_pt, pos_pt;
    if (f_init_neg) { neg_pt = same_side; pos_pt = other; }
    else            { neg_pt = other;     pos_pt = same_side; }

    // --- Phase 2: hybrid refine ---
    double best_a = a_init, best_f = f_init, best_abs_d = fabs(fp_init);
    int    converged_residual = 0, converged_bracket = 0;

    for (unsigned int it = 0; it < max_refine_iters; ++it) {
        double lo = fmin(neg_pt, pos_pt);
        double hi = fmax(neg_pt, pos_pt);
        double mid = 0.5 * (lo + hi);

        double f_mid, fp_mid, fpp_mid; int ok_mid;
        eval_F_analytic(mid, alpha, beta, gamma, &f_mid, &fp_mid, &fpp_mid, &ok_mid);
        if (!ok_mid) { out_a_root[row]=best_a; out_abs_deriv[row]=best_abs_d; out_residual[row]=best_f; out_status[row]=5; return; }
        if (fabs(f_mid) < fabs(best_f)) { best_a = mid; best_f = f_mid; best_abs_d = fabs(fp_mid); }

        if (fabs(f_mid) <= convergence_tol) { converged_residual = 1; break; }

        // Safeguarded Halley probe inside (lo, hi); fall back to Newton, else midpoint.
        double probe_pt = mid;
        int halley_ok = 0;
        if (isfinite(fp_mid) && fabs(fp_mid) > 1e-30) {
            double denom = 2.0 * fp_mid * fp_mid - f_mid * fpp_mid;
            if (isfinite(denom) && fabs(denom) > 1e-30) {
                double cand = mid - (2.0 * f_mid * fp_mid) / denom;
                if (cand > lo && cand < hi) { probe_pt = cand; halley_ok = 1; }
            }
        }
        if (!halley_ok && isfinite(fp_mid) && fabs(fp_mid) > 1e-30) {
            double cand = mid - f_mid / fp_mid;
            if (cand > lo && cand < hi) probe_pt = cand;
        }

        double f_b = f_mid;
        if (probe_pt != mid) {
            double f_p, fp_p, fpp_p; int ok_p;
            eval_F_analytic(probe_pt, alpha, beta, gamma, &f_p, &fp_p, &fpp_p, &ok_p);
            if (!ok_p) { out_a_root[row]=best_a; out_abs_deriv[row]=best_abs_d; out_residual[row]=best_f; out_status[row]=5; return; }
            if (fabs(f_p) < fabs(best_f)) { best_a = probe_pt; best_f = f_p; best_abs_d = fabs(fp_p); }
            f_b = f_p;
        } else {
            probe_pt = mid;
        }

        if (f_b <= 0.0) neg_pt = probe_pt; else pos_pt = probe_pt;

        double next_lo = fmin(neg_pt, pos_pt);
        double next_hi = fmax(neg_pt, pos_pt);
        if (fabs(next_hi - next_lo) <= convergence_tol * (1.0 + fabs(next_hi) + fabs(next_lo))) {
            converged_bracket = 1; break;
        }
    }

    if (!isfinite(best_abs_d) || best_abs_d == 0.0) {
        double f_r, fp_r, fpp_r; int ok_r;
        eval_F_analytic(best_a, alpha, beta, gamma, &f_r, &fp_r, &fpp_r, &ok_r);
        if (ok_r) best_abs_d = fabs(fp_r);
    }

    out_a_root[row]    = best_a;
    out_abs_deriv[row] = best_abs_d;
    out_residual[row]  = best_f;
    if      (converged_residual)             out_status[row] = 0;
    else if (converged_bracket)              out_status[row] = 1;
    else                                     out_status[row] = 4;
}
"#;

/// Launch the Step 3 device intercept solve.  Returns `Ok(None)` on
/// non-Linux / no-CUDA builds so the dispatcher can fall back to the
/// CPU oracle; returns `Err` only on genuine driver / compile failures.
pub fn try_device_intercept_solve(
    inputs: &SurvivalFlexInterceptSolveInputs<'_>,
) -> Result<Option<SurvivalFlexInterceptSolveOutputs>, GpuError> {
    inputs.validate()?;
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    #[cfg(target_os = "linux")]
    {
        let backend = match SurvivalFlexGpuBackend::probe() {
            Ok(b) => b,
            Err(GpuError::DriverLibraryUnavailable { .. }) => return Ok(None),
            Err(other) => return Err(other),
        };
        Some(backend.launch_intercept_solve_linux(inputs)).transpose()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
impl SurvivalFlexGpuBackend {
    /// NVRTC-compile (lazily, shared with other survival_flex modules) the
    /// Step 3 module.  Held in a static `OnceLock` so the compile runs
    /// once per process.
    fn compile_intercept_solve_module(&self) -> Result<Arc<CudaModule>, GpuError> {
        static INTERCEPT_MODULE: OnceLock<
            std::sync::Mutex<Option<Result<Arc<CudaModule>, GpuError>>>,
        > = OnceLock::new();
        let cell = INTERCEPT_MODULE.get_or_init(|| std::sync::Mutex::new(None));
        let mut guard = cell.lock().map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex intercept-solve module mutex poisoned: {err}"),
        })?;
        if let Some(existing) = guard.as_ref() {
            return existing.clone();
        }
        let result = (|| {
            // Shared arch+fmad options (NOT bare `compile_ptx`): #1686's
            // `--fmad=false` keeps the Newton intercept solve bit-comparable to
            // the CPU path, and the #1551 arch pin keys the kernel to the real
            // device compute capability rather than NVRTC's pre-sm_60 default.
            let ptx = gam_gpu::device_cache::compile_ptx_arch(SURVIVAL_FLEX_INTERCEPT_SOLVE_SOURCE)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve NVRTC compile: {err}"),
                })?;
            self.inner
                .ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve module load: {err}"),
                })
        })();
        *guard = Some(result.clone());
        result
    }

    fn launch_intercept_solve_linux(
        &self,
        inputs: &SurvivalFlexInterceptSolveInputs<'_>,
    ) -> Result<SurvivalFlexInterceptSolveOutputs, GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let module = self.compile_intercept_solve_module()?;
        let func = module
            .load_function("survival_flex_intercept_solve")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve load_function: {err}"),
            })?;

        let n = inputs.n;
        let stream = &self.inner.stream;
        let mk_htod = |slice: &[f64], name: &str| -> Result<_, GpuError> {
            stream
                .clone_htod(slice)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve memcpy_stod {name}: {err}"),
                })
        };
        let d_a_warm = mk_htod(inputs.a_warm, "a_warm")?;
        let d_alpha = mk_htod(inputs.alpha, "alpha")?;
        let d_beta = mk_htod(inputs.beta, "beta")?;
        let d_gamma = mk_htod(inputs.gamma, "gamma")?;

        let mut d_a_root =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc a_root: {err}"),
                })?;
        let mut d_abs_deriv =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc abs_deriv: {err}"),
                })?;
        let mut d_residual =
            stream
                .alloc_zeros::<f64>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc residual: {err}"),
                })?;
        let mut d_status =
            stream
                .alloc_zeros::<u8>(n)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex intercept-solve alloc status: {err}"),
                })?;

        let convergence_tol = inputs.convergence_tol;
        let max_bracket_iters = inputs.max_bracket_iters;
        let max_refine_iters = inputs.max_refine_iters;
        let n_i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex intercept-solve n={n} overflows i32"),
        })?;

        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&d_a_warm)
            .arg(&d_alpha)
            .arg(&d_beta)
            .arg(&d_gamma)
            .arg(&convergence_tol)
            .arg(&max_bracket_iters)
            .arg(&max_refine_iters)
            .arg(&n_i32)
            .arg(&mut d_a_root)
            .arg(&mut d_abs_deriv)
            .arg(&mut d_residual)
            .arg(&mut d_status);
        // SAFETY: argument types match the kernel signature; grid covers n rows.
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex intercept-solve launch: {err}"),
        })?;

        let status = stream
            .clone_dtoh(&d_status)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve memcpy_dtoh status: {err}"),
            })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex intercept-solve synchronize: {err}"),
            })?;

        Ok(SurvivalFlexInterceptSolveOutputs { status })
    }
}

// ────────────────────────────────────────────────────────────────────────
// Step 5 — per-row primary gradient + Hessian assembly from the full jet.
//
// Given the entry- and exit-time jets `(eta, chi, d, eta_u, eta_uv,
// chi_u, chi_uv, d_u, d_uv)` (from Layer C-α + C-β) plus the
// `signed_probit_neglog` derivatives `(k1, k2)` at `-entry.eta` /
// `-exit.eta`, the per-row NLL and its primary gradient + Hessian are
// pure scalar / vector algebra (CPU
// `compute_row_flex_primary_gradient_hessian_from_parts`, lines
// 7263-7384 of survival_marginal_slope.rs).
//
// The joint-β `axpy_row_into` pullback into the dense coefficient
// gradient / Hessian is family-owned (per-block design rows in
// `marginal_design` / `logslope_design` / `score_warp` / `link_dev`
// runtimes); Step 6 wires it behind the three try_* entry points.
// ────────────────────────────────────────────────────────────────────────

/// Per-row time-point jet bundle for the Step-5 assembly.
#[derive(Clone, Debug)]
pub struct SurvivalFlexTimepointJet<'a> {
    pub eta: f64,
    pub chi: f64,
    pub d: f64,
    /// Length `p`.
    pub eta_u: &'a [f64],
    /// Row-major `p × p`, symmetric.
    pub eta_uv: &'a [f64],
    /// Length `p`.
    pub chi_u: &'a [f64],
    /// Row-major `p × p`, symmetric.
    pub chi_uv: &'a [f64],
    /// Length `p`.
    pub d_u: &'a [f64],
    /// Row-major `p × p`, symmetric.
    pub d_uv: &'a [f64],
}

/// Per-row inputs for the Step-5 primary gradient + Hessian assembly.
#[derive(Clone, Debug)]
pub struct SurvivalFlexStep5RowInputs<'a> {
    pub entry: SurvivalFlexTimepointJet<'a>,
    pub exit: SurvivalFlexTimepointJet<'a>,
    pub wi: f64,
    pub di: f64,
    pub q1: f64,
    pub qd1: f64,
    /// `q1_index`: primary index of the q1 perturbation, `usize::MAX`
    /// to disable the `+ wi·di·q1` / `+ wi·di` bumps.
    pub q1_index: usize,
    /// `qd1_index`: primary index of the qd1 perturbation, `usize::MAX`
    /// to disable the `-wi·di/qd1` / `+wi·di/qd1²` bumps.
    pub qd1_index: usize,
    pub entry_k1: f64,
    pub entry_k2: f64,
    pub exit_k1: f64,
    pub exit_k2: f64,
    pub log_surv0: f64,
    pub log_surv1: f64,
}

/// Per-row Step-5 outputs.
#[derive(Clone, Debug)]
pub struct SurvivalFlexStep5RowOutputs {
    pub row_nll: f64,
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
}

/// Step-5 per-row primary gradient + Hessian assembly.  Pure scalar /
/// vector algebra over the supplied jet bundles; no quadrature.
pub fn try_device_step5_primary_assembly(
    rows: &[SurvivalFlexStep5RowInputs<'_>],
) -> Result<Vec<SurvivalFlexStep5RowOutputs>, GpuError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let p = r.entry.eta_u.len();
        let check = |label: &str, len: usize, expected: usize| -> Result<(), GpuError> {
            if len != expected {
                return Err(GpuError::DriverCallFailed {
                    reason: format!("step5 row {i}: {label}.len()={len} expected {expected}"),
                });
            }
            Ok(())
        };
        check("entry.eta_uv", r.entry.eta_uv.len(), p * p)?;
        check("entry.chi_u", r.entry.chi_u.len(), p)?;
        check("entry.chi_uv", r.entry.chi_uv.len(), p * p)?;
        check("entry.d_u", r.entry.d_u.len(), p)?;
        check("entry.d_uv", r.entry.d_uv.len(), p * p)?;
        check("exit.eta_u", r.exit.eta_u.len(), p)?;
        check("exit.eta_uv", r.exit.eta_uv.len(), p * p)?;
        check("exit.chi_u", r.exit.chi_u.len(), p)?;
        check("exit.chi_uv", r.exit.chi_uv.len(), p * p)?;
        check("exit.d_u", r.exit.d_u.len(), p)?;
        check("exit.d_uv", r.exit.d_uv.len(), p * p)?;

        if !(r.exit.chi.is_finite() && r.exit.chi > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "step5 row {i}: exit.chi must be positive finite, got {}",
                    r.exit.chi
                ),
            });
        }
        if !(r.exit.d.is_finite() && r.exit.d > 0.0) {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "step5 row {i}: exit.d must be positive finite, got {}",
                    r.exit.d
                ),
            });
        }

        let log_phi_eta1 = -0.5 * (r.exit.eta * r.exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (r.q1 * r.q1 + std::f64::consts::TAU.ln());
        let row_nll = r.wi
            * (r.log_surv0
                - (1.0 - r.di) * r.log_surv1
                - r.di * log_phi_eta1
                - r.di * r.exit.chi.ln()
                - r.di * log_phi_q1
                + r.di * r.exit.d.ln()
                - r.di * r.qd1.ln());

        let entry_u1 = -r.entry_k1;
        let entry_u2 = r.entry_k2;
        let exit_surv_u1 = -r.exit_k1;
        let exit_surv_u2 = r.exit_k2;

        let mut grad = vec![0.0_f64; p];
        for u in 0..p {
            let mut val = 0.0;
            val += entry_u1 * r.entry.eta_u[u];
            val += exit_surv_u1 * r.exit.eta_u[u];
            val += r.wi * r.di * r.exit.eta * r.exit.eta_u[u];
            val -= r.wi * r.di * r.exit.chi_u[u] / r.exit.chi;
            if u == r.q1_index {
                val += r.wi * r.di * r.q1;
            }
            val += r.wi * r.di * r.exit.d_u[u] / r.exit.d;
            if u == r.qd1_index {
                val -= r.wi * r.di / r.qd1;
            }
            grad[u] = val;
        }

        let mut hess = vec![0.0_f64; p * p];
        let chi_sq = r.exit.chi * r.exit.chi;
        let d_sq = r.exit.d * r.exit.d;
        for u in 0..p {
            for v in u..p {
                let mut val = 0.0;
                val += entry_u2 * r.entry.eta_u[u] * r.entry.eta_u[v]
                    + entry_u1 * r.entry.eta_uv[u * p + v];
                val += exit_surv_u2 * r.exit.eta_u[u] * r.exit.eta_u[v]
                    + exit_surv_u1 * r.exit.eta_uv[u * p + v];
                val += r.wi
                    * r.di
                    * (r.exit.eta_u[u] * r.exit.eta_u[v] + r.exit.eta * r.exit.eta_uv[u * p + v]);
                val -= r.wi
                    * r.di
                    * (r.exit.chi_uv[u * p + v] / r.exit.chi
                        - (r.exit.chi_u[u] * r.exit.chi_u[v]) / chi_sq);
                if u == r.q1_index && v == r.q1_index {
                    val += r.wi * r.di;
                }
                val += r.wi
                    * r.di
                    * (r.exit.d_uv[u * p + v] / r.exit.d - (r.exit.d_u[u] * r.exit.d_u[v]) / d_sq);
                if u == r.qd1_index && v == r.qd1_index {
                    val += r.wi * r.di / (r.qd1 * r.qd1);
                }
                hess[u * p + v] = val;
                if v != u {
                    hess[v * p + u] = val;
                }
            }
        }

        out.push(SurvivalFlexStep5RowOutputs {
            row_nll,
            grad,
            hess,
        });
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────
// Step 6 — joint-β pullback of the Step-5 primary gradient / Hessian into
// the dense coefficient space.
//
// Step 5 produces, per row, the primary-space gradient `g_p ∈ ℝ^r` and
// Hessian `H_p ∈ ℝ^{r×r}` (local coordinates: q0, q1, q̇1, g, and any
// score-warp / link-dev primaries).  The joint-β pullback maps those into
// the dense coefficient space through the per-row primary→coefficient
// Jacobian `J ∈ ℝ^{r×p}` (`J[a,j] = ∂ primary_a / ∂ β_j`, assembled by the
// family from its per-block design rows):
//
//     nll      = Σ_rows nll_row
//     grad[j]  = Σ_rows Σ_a  J[a,j] · g_p[a]                 = Σ_rows Jᵀ g_p
//     H[j,k]   = Σ_rows Σ_{a,b} J[a,j] · H_p[a,b] · J[b,k]   = Σ_rows Jᵀ H_p J
//
// (The primary coordinates are affine in β within a fit iteration — the
// rigid/flex map composes the fixed design rows with the per-row scalar
// chain — so there is no `∂²primary/∂β²` term here; the second-order curvature
// already lives inside `H_p`.  This is the same contraction the family's
// sparse CPU path `add_pullback_primary_hessian` performs block-by-block.)
//
// This is the device-shaped Step-6 kernel written as pure host algebra so it
// is bit-exact CPU-verifiable now; the CUDA dispatch (Linux/V100) folds the
// same contraction on-device once the substrate jet assembly lands.
// ────────────────────────────────────────────────────────────────────────

/// Per-row inputs for the Step-6 joint-β pullback: the Step-5 primary
/// gradient / Hessian for the row plus the row's dense primary→coefficient
/// Jacobian `J` (`r × p`, row-major: `J[a*p + j] = ∂ primary_a / ∂ β_j`).
#[derive(Clone, Debug)]
pub struct SurvivalFlexStep6RowPullback<'a> {
    /// Primary-space row outputs from [`try_device_step5_primary_assembly`].
    pub primary: &'a SurvivalFlexStep5RowOutputs,
    /// Row-major `r × p` primary→coefficient Jacobian.
    pub jacobian: &'a [f64],
}

/// Step-6 joint-β pullback: fold per-row Step-5 primary gradient / Hessian
/// into the dense coefficient-space `(nll, grad ∈ ℝ^p, H ∈ ℝ^{p×p})`.
///
/// Pure host algebra (`Σ_rows Jᵀ g_p` and `Σ_rows Jᵀ H_p J`); CPU-verifiable
/// and the reference the on-device contraction is checked against.  `p` is the
/// joint coefficient dimension; every row's Jacobian must be `r × p` where
/// `r` is that row's primary dimension (`primary.grad.len()`).
pub fn pullback_step6_joint_beta(
    rows: &[SurvivalFlexStep6RowPullback<'_>],
    p: usize,
) -> Result<(f64, Array1<f64>, Array2<f64>), GpuError> {
    let mut nll = 0.0_f64;
    let mut grad = Array1::<f64>::zeros(p);
    let mut hess = Array2::<f64>::zeros((p, p));

    for (i, row) in rows.iter().enumerate() {
        let r = row.primary.grad.len();
        if row.primary.hess.len() != r * r {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "step6 row {i}: primary.hess.len()={} expected r*r={}",
                    row.primary.hess.len(),
                    r * r
                ),
            });
        }
        if row.jacobian.len() != r * p {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "step6 row {i}: jacobian.len()={} expected r*p={}",
                    row.jacobian.len(),
                    r * p
                ),
            });
        }

        nll += row.primary.row_nll;

        // grad += Jᵀ g_p.
        let g_p = &row.primary.grad;
        let j = row.jacobian;
        for a in 0..r {
            let ga = g_p[a];
            if ga == 0.0 {
                continue;
            }
            let j_row = &j[a * p..a * p + p];
            for k in 0..p {
                grad[k] += j_row[k] * ga;
            }
        }

        // H += Jᵀ H_p J, computed as Σ_a Σ_b H_p[a,b] · (J_a ⊗ J_b).
        // Form the intermediate M = H_p · J (r × p), then H += Jᵀ M, so the
        // cost is O(r²·p + r·p²) instead of O(r²·p²).
        let h_p = &row.primary.hess;
        let mut m = vec![0.0_f64; r * p];
        for a in 0..r {
            for b in 0..r {
                let hab = h_p[a * r + b];
                if hab == 0.0 {
                    continue;
                }
                let j_b = &j[b * p..b * p + p];
                let m_a = &mut m[a * p..a * p + p];
                for k in 0..p {
                    m_a[k] += hab * j_b[k];
                }
            }
        }
        for a in 0..r {
            let j_a = &j[a * p..a * p + p];
            let m_a = &m[a * p..a * p + p];
            for col in 0..p {
                let jac = j_a[col];
                if jac == 0.0 {
                    continue;
                }
                for k in 0..p {
                    hess[[col, k]] += jac * m_a[k];
                }
            }
        }
    }

    // Symmetrize defensively: Jᵀ H_p J is symmetric when H_p is, but rounding
    // in the two accumulation orders can leave a sub-ULP asymmetry; average the
    // off-diagonal so downstream Cholesky / eigen paths see an exactly symmetric
    // matrix (mirrors the CPU assembler's symmetric write-back).
    for col in 0..p {
        for k in (col + 1)..p {
            let avg = 0.5 * (hess[[col, k]] + hess[[k, col]]);
            hess[[col, k]] = avg;
            hess[[k, col]] = avg;
        }
    }

    Ok((nll, grad, hess))
}

// ────────────────────────────────────────────────────────────────────────
// Step 6 (device) — on-CUDA per-row joint-β contraction.
//
// The host `pullback_step6_joint_beta` above is the bit-exact reference.  The
// device path below folds the *same* per-row contraction on the GPU: one CUDA
// block per row computes that row's dense contribution
//
//     grad_row[j]   = Σ_a J[a,j] · g_p[a]                       (∈ ℝ^p)
//     hess_row[j,k] = Σ_a Σ_b J[a,j] · H_p[a,b] · J[b,k]        (∈ ℝ^{p×p})
//
// using the same blocked `M = H_p · J` intermediate the host uses.  The per-row
// partials are copied back and summed on the host **in row order** — matching the
// host reference's sequential row accumulation — so the cross-row reduction order
// is identical.  The device total therefore agrees with `pullback_step6_joint_beta`
// to a small multiple of ULP, NOT bit-for-bit: the per-row `M = H_p·J` inner
// products reassociate (the device sums each contraction in a different order
// than the host's separate-rounding scalar loop), so the last bits differ
// (measured worst abs ~2.8e-14 on a Tesla V100). This is the residual AFTER
// #1686 disabled NVRTC FMA contraction (this module now compiles through
// `device_cache::compile_ptx_arch`, so `--fmad=false`): the drift is therefore
// pure reassociation, NOT FMA — fmad=false did not remove it. A #1175
// floating-point-order difference, not an algebra mismatch; parity tests
// therefore use a relative band, not `assert_eq!`.  The expensive
// O(r²p + r·p²) per-row contraction runs on the
// device; the cheap O(n·p²) row reduction stays on the host in CPU order.
//
// Concatenated SoA layout (host builds once, uploads once):
//   * `g_p_flat`  — Σ_rows r_row  primary-gradient scalars, row-major.
//   * `h_p_flat`  — Σ_rows r_row² primary-Hessian scalars, row-major.
//   * `jac_flat`  — Σ_rows r_row·p Jacobian scalars, row-major (J[a*p+j]).
//   * `r_arr`     — per-row primary dim r_row.
//   * `g_off/h_off/j_off` — per-row start offsets into the three flats.
// Output: `grad_rows` (n·p) and `hess_rows` (n·p·p), one dense block per row.
// ────────────────────────────────────────────────────────────────────────

/// Flattened SoA view of the Step-6 rows, plus the per-row dims/offsets the
/// device kernel indexes.  Built once from `&[SurvivalFlexStep6RowPullback]`.
#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct Step6DeviceBatch {
    n_rows: usize,
    p: usize,
    nll: f64,
    g_p_flat: Vec<f64>,
    h_p_flat: Vec<f64>,
    jac_flat: Vec<f64>,
    r_arr: Vec<u32>,
    g_off: Vec<u32>,
    h_off: Vec<u32>,
    j_off: Vec<u32>,
}

#[cfg(target_os = "linux")]
impl Step6DeviceBatch {
    fn build(rows: &[SurvivalFlexStep6RowPullback<'_>], p: usize) -> Result<Self, GpuError> {
        let n_rows = rows.len();
        let mut nll = 0.0_f64;
        let mut g_p_flat = Vec::new();
        let mut h_p_flat = Vec::new();
        let mut jac_flat = Vec::new();
        let mut r_arr = Vec::with_capacity(n_rows);
        let mut g_off = Vec::with_capacity(n_rows);
        let mut h_off = Vec::with_capacity(n_rows);
        let mut j_off = Vec::with_capacity(n_rows);
        for (i, row) in rows.iter().enumerate() {
            let r = row.primary.grad.len();
            if row.primary.hess.len() != r * r {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "step6 device row {i}: primary.hess.len()={} expected r*r={}",
                        row.primary.hess.len(),
                        r * r
                    ),
                });
            }
            if row.jacobian.len() != r * p {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "step6 device row {i}: jacobian.len()={} expected r*p={}",
                        row.jacobian.len(),
                        r * p
                    ),
                });
            }
            nll += row.primary.row_nll;
            g_off.push(
                u32::try_from(g_p_flat.len()).map_err(|_| GpuError::DriverCallFailed {
                    reason: "step6 device: g_p offset overflows u32".to_string(),
                })?,
            );
            h_off.push(
                u32::try_from(h_p_flat.len()).map_err(|_| GpuError::DriverCallFailed {
                    reason: "step6 device: h_p offset overflows u32".to_string(),
                })?,
            );
            j_off.push(
                u32::try_from(jac_flat.len()).map_err(|_| GpuError::DriverCallFailed {
                    reason: "step6 device: jac offset overflows u32".to_string(),
                })?,
            );
            r_arr.push(u32::try_from(r).map_err(|_| GpuError::DriverCallFailed {
                reason: "step6 device: r overflows u32".to_string(),
            })?);
            g_p_flat.extend_from_slice(&row.primary.grad);
            h_p_flat.extend_from_slice(&row.primary.hess);
            jac_flat.extend_from_slice(row.jacobian);
        }
        Ok(Self {
            n_rows,
            p,
            nll,
            g_p_flat,
            h_p_flat,
            jac_flat,
            r_arr,
            g_off,
            h_off,
            j_off,
        })
    }
}

/// NVRTC source for the per-row Step-6 contraction.  One block per row; threads
/// in the block cooperatively fill the row's `grad_row[p]` and `hess_row[p*p]`.
/// The intermediate `M = H_p · J` (r × p) is computed in registers/global per
/// thread-tile exactly as the host does, so each row's output is bit-identical
/// to the host per-row partial.
///
/// #415 parity-lock: this source is available on EVERY target (not just Linux)
/// so the CPU-side device-arithmetic emulator's structural lockstep guard
/// (`step6_tests::step6_device_emulator_source_lockstep_fingerprint_415`) can
/// assert the `.cu` still spells the arithmetic the CPU emulator mirrors — on
/// CPU CI, no CUDA required. Only the NVRTC compile/launch consumers below stay
/// Linux-gated. `allow(dead_code)` because on a non-Linux, non-test lib build
/// nothing references the string.
#[cfg_attr(all(not(target_os = "linux"), not(test)), allow(dead_code))]
const SURVIVAL_FLEX_STEP6_SOURCE: &str = r#"
extern "C" __global__ void survival_flex_step6_rows(
    const double * __restrict__ g_p_flat,
    const double * __restrict__ h_p_flat,
    const double * __restrict__ jac_flat,
    const unsigned int * __restrict__ r_arr,
    const unsigned int * __restrict__ g_off,
    const unsigned int * __restrict__ h_off,
    const unsigned int * __restrict__ j_off,
    int                                p,
    int                                n_rows,
    double * __restrict__              grad_rows,   // n_rows * p
    double * __restrict__              hess_rows,   // n_rows * p * p
    double * __restrict__              m_scratch    // n_rows * rmax * p (row-major per row, r*p used)
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int r   = (int) r_arr[row];
    int goff = (int) g_off[row];
    int hoff = (int) h_off[row];
    int joff = (int) j_off[row];

    const double * g_p = g_p_flat + goff;     // length r
    const double * h_p = h_p_flat + hoff;     // r*r row-major
    const double * j   = jac_flat + joff;     // r*p row-major (J[a*p+j])

    double * grad_row = grad_rows + (size_t) row * (size_t) p;     // length p
    double * hess_row = hess_rows + (size_t) row * (size_t) p * (size_t) p; // p*p
    double * m_row    = m_scratch + (size_t) row * (size_t) r * (size_t) p; // r*p (M = H_p J)

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    // 1) grad_row[k] = Σ_a J[a,k] · g_p[a].  Match the host accumulation order:
    //    outer over a, inner over k.  Per-output (k) accumulation is order-stable.
    for (int k = tid; k < p; k += stride) {
        double acc = 0.0;
        for (int a = 0; a < r; ++a) {
            double ga = g_p[a];
            if (ga != 0.0) acc += j[a * p + k] * ga;
        }
        grad_row[k] = acc;
    }

    // 2) M = H_p · J  (r × p):  M[a,k] = Σ_b H_p[a,b] · J[b,k].  Host order:
    //    outer a, inner b (skip zero hab), inner k.  We parallelise over the
    //    (a,k) output grid; each output sums over b in the same b-order.
    for (int idx = tid; idx < r * p; idx += stride) {
        int a = idx / p;
        int k = idx - a * p;
        double acc = 0.0;
        for (int b = 0; b < r; ++b) {
            double hab = h_p[a * r + b];
            if (hab != 0.0) acc += hab * j[b * p + k];
        }
        m_row[a * p + k] = acc;
    }
    __syncthreads();

    // 3) hess_row[col,k] = Σ_a J[a,col] · M[a,k].  Host order: outer a, skip
    //    zero jac, inner k.  Parallelise over (col,k); sum over a in a-order.
    for (int idx = tid; idx < p * p; idx += stride) {
        int col = idx / p;
        int k   = idx - col * p;
        double acc = 0.0;
        for (int a = 0; a < r; ++a) {
            double jac = j[a * p + col];
            if (jac != 0.0) acc += jac * m_row[a * p + k];
        }
        hess_row[col * p + k] = acc;
    }
}
"#;

/// Device Step-6 joint-β contraction.  Returns `Ok(None)` on non-Linux / no-CUDA
/// builds (caller folds on the host via [`pullback_step6_joint_beta`]); returns
/// `(nll, grad, hess)` agreeing with the host reference to a small ULP multiple
/// on a healthy CUDA device (NOT bit-exact — the per-row contraction reassociates;
/// FMA is off under #1686's --fmad=false, so the residual is pure reassociation,
/// see the Step-6 device section comment and #1175).
///
/// The per-row dense contraction (`Jᵀ g_p`, `Jᵀ H_p J`) runs on the GPU; the
/// per-row partials are summed on the host in row order so the cross-row reduction
/// order matches the host reference.  The dense Hessian is symmetrized with the
/// same averaging pass the host uses.
pub fn try_device_step6_joint_beta(
    rows: &[SurvivalFlexStep6RowPullback<'_>],
    p: usize,
) -> Result<Option<(f64, Array1<f64>, Array2<f64>)>, GpuError> {
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    if rows.is_empty() {
        // Empty fold is the host's zero answer; no device round-trip needed.
        return Ok(Some((0.0, Array1::zeros(p), Array2::zeros((p, p)))));
    }
    #[cfg(target_os = "linux")]
    {
        let batch = Step6DeviceBatch::build(rows, p)?;
        let backend = match SurvivalFlexGpuBackend::probe() {
            Ok(b) => b,
            Err(GpuError::DriverLibraryUnavailable { .. }) => return Ok(None),
            Err(other) => return Err(other),
        };
        Some(backend.launch_step6_joint_beta_linux(&batch)).transpose()
    }
    #[cfg(not(target_os = "linux"))]
    {
        // No CUDA toolchain off Linux; `rows` was already consumed by the
        // emptiness check above, and the host caller folds the contraction.
        Ok(None)
    }
}

#[cfg(target_os = "linux")]
impl SurvivalFlexGpuBackend {
    fn compile_step6_module(&self) -> Result<Arc<CudaModule>, GpuError> {
        static STEP6_MODULE: OnceLock<std::sync::Mutex<Option<Result<Arc<CudaModule>, GpuError>>>> =
            OnceLock::new();
        let cell = STEP6_MODULE.get_or_init(|| std::sync::Mutex::new(None));
        let mut guard = cell.lock().map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex step6 module mutex poisoned: {err}"),
        })?;
        if let Some(existing) = guard.as_ref() {
            return existing.clone();
        }
        let result = (|| {
            // Compile through the shared arch+fmad options (NOT bare
            // `compile_ptx`). #1686 set `--fmad=false` in those options so the
            // per-row `M = H_p·J` contraction is FMA-free and bit-comparable to
            // the separately-rounded CPU pullback `pullback_step6_joint_beta`;
            // bare `compile_ptx` leaves NVRTC at `--fmad=true`, fusing `a*b+c`
            // into one rounding and drifting ~2e-16 from the CPU oracle. The
            // arch pin (#1551) keys the kernel to the device's real compute
            // capability instead of NVRTC's pre-sm_60 default.
            let ptx = gam_gpu::device_cache::compile_ptx_arch(SURVIVAL_FLEX_STEP6_SOURCE).map_err(
                |err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 NVRTC compile: {err}"),
                },
            )?;
            self.inner
                .ctx
                .load_module(ptx)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 module load: {err}"),
                })
        })();
        *guard = Some(result.clone());
        result
    }

    fn launch_step6_joint_beta_linux(
        &self,
        batch: &Step6DeviceBatch,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), GpuError> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let module = self.compile_step6_module()?;
        let func = module
            .load_function("survival_flex_step6_rows")
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex step6 load_function: {err}"),
            })?;

        let n_rows = batch.n_rows;
        let p = batch.p;
        let rmax = batch.r_arr.iter().copied().max().unwrap_or(0) as usize;
        let stream = &self.inner.stream;

        let mk_htod_f64 = |slice: &[f64], name: &str| -> Result<_, GpuError> {
            stream
                .clone_htod(slice)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 htod {name}: {err}"),
                })
        };
        let mk_htod_u32 = |slice: &[u32], name: &str| -> Result<_, GpuError> {
            stream
                .clone_htod(slice)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 htod {name}: {err}"),
                })
        };

        let d_g_p = mk_htod_f64(&batch.g_p_flat, "g_p_flat")?;
        let d_h_p = mk_htod_f64(&batch.h_p_flat, "h_p_flat")?;
        let d_jac = mk_htod_f64(&batch.jac_flat, "jac_flat")?;
        let d_r = mk_htod_u32(&batch.r_arr, "r_arr")?;
        let d_goff = mk_htod_u32(&batch.g_off, "g_off")?;
        let d_hoff = mk_htod_u32(&batch.h_off, "h_off")?;
        let d_joff = mk_htod_u32(&batch.j_off, "j_off")?;

        let mut d_grad_rows =
            stream
                .alloc_zeros::<f64>(n_rows * p)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 alloc grad_rows: {err}"),
                })?;
        let mut d_hess_rows = stream.alloc_zeros::<f64>(n_rows * p * p).map_err(|err| {
            GpuError::DriverCallFailed {
                reason: format!("survival_flex step6 alloc hess_rows: {err}"),
            }
        })?;
        // M scratch sized to the worst-case r per row so every block has room.
        let mut d_m_scratch = stream
            .alloc_zeros::<f64>(n_rows * rmax.max(1) * p)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex step6 alloc m_scratch: {err}"),
            })?;

        let p_i32 = i32::try_from(p).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex step6 p={p} overflows i32"),
        })?;
        let n_i32 = i32::try_from(n_rows).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex step6 n_rows={n_rows} overflows i32"),
        })?;

        // One block per row; 256 threads cooperatively fill the row's p / p² grid.
        let block: u32 = 256;
        let grid: u32 = u32::try_from(n_rows).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("survival_flex step6 n_rows={n_rows} overflows grid u32"),
        })?;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&d_g_p)
            .arg(&d_h_p)
            .arg(&d_jac)
            .arg(&d_r)
            .arg(&d_goff)
            .arg(&d_hoff)
            .arg(&d_joff)
            .arg(&p_i32)
            .arg(&n_i32)
            .arg(&mut d_grad_rows)
            .arg(&mut d_hess_rows)
            .arg(&mut d_m_scratch);
        // SAFETY: argument types/order match the kernel signature; grid covers
        // every row; per-block output buffers are sized n_rows·p and n_rows·p².
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("survival_flex step6 launch: {err}"),
        })?;

        let grad_rows =
            stream
                .clone_dtoh(&d_grad_rows)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 dtoh grad_rows: {err}"),
                })?;
        let hess_rows =
            stream
                .clone_dtoh(&d_hess_rows)
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: format!("survival_flex step6 dtoh hess_rows: {err}"),
                })?;
        stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("survival_flex step6 synchronize: {err}"),
            })?;

        // Deterministic host reduction over rows (row-order) to match the host
        // reference's sequential accumulation exactly.
        let mut grad = Array1::<f64>::zeros(p);
        let mut hess = Array2::<f64>::zeros((p, p));
        for row in 0..n_rows {
            let gbase = row * p;
            for k in 0..p {
                grad[k] += grad_rows[gbase + k];
            }
            let hbase = row * p * p;
            for col in 0..p {
                for k in 0..p {
                    hess[[col, k]] += hess_rows[hbase + col * p + k];
                }
            }
        }
        // Same symmetrization pass as the host reference.
        for col in 0..p {
            for k in (col + 1)..p {
                let avg = 0.5 * (hess[[col, k]] + hess[[k, col]]);
                hess[[col, k]] = avg;
                hess[[k, col]] = avg;
            }
        }

        Ok((batch.nll, grad, hess))
    }
}

// ────────────────────────────────────────────────────────────────────────
// Three pullback entry points.  The device-side flex jet assembly (Steps 2–5:
// cubic-cell moments → intercept solve → η/χ/d jets → primary
// gradient/Hessian) is still gated by the CUDA backend, but once the host has
// supplied Step-6 rows these entry points perform the coefficient-space
// joint-β fold unconditionally after shape validation.  That keeps the
// `SurvivalFlexGpuRowInputs` surface honest: assembled row-primary derivatives
// are folded through the family-provided primary→coefficient Jacobian instead
// of dying behind the backend probe.
// ────────────────────────────────────────────────────────────────────────

/// Evaluate the survival-flex negative log-likelihood and joint-β
/// gradient on the GPU.  Returns `Ok(None)` if the GPU path is
/// unsupported for this shape (caller falls back to CPU); returns
/// `Err` only when the request *is* supported but the driver failed.
pub fn try_survival_flex_gradient(
    inputs: SurvivalFlexGpuRowInputs<'_>,
    intercept_solve: Option<&SurvivalFlexInterceptSolveInputs<'_>>,
    step6: Option<&[SurvivalFlexStep6RowPullback<'_>]>,
) -> Result<Option<(f64, Array1<f64>)>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    // Step 3 hookup: when an intercept-solve descriptor is provided,
    // run the device monotone-root kernel as the precheck stage so the
    // Step-3 path has a real production consumer before Step 4/5/6
    // joint-β assembly lands.  Step 4 will replace the analytic
    // evaluator on the device side with the real survival F(a)
    // calibration evaluator; the host-side hookup shape stays the
    // same.  On any non-OK device row we fall back to CPU; on every
    // OK row we accept the warm-started root and the dispatcher
    // continues to the (not-yet-landed) joint-β assembly — which for
    // Step 3 is the `Ok(None)` sentinel.
    if let Some(ints) = intercept_solve {
        // Prefer the device kernel; fall back to the CPU oracle on
        // non-CUDA builds.  The oracle is the same code path the
        // device kernel will be parity-tested against, so dispatcher
        // behaviour stays identical regardless of where the solve
        // ran.
        let out = match try_device_intercept_solve(ints)? {
            Some(out) => out,
            None => cpu_oracle_intercept_solve(ints),
        };
        if out.status.iter().any(|&s| s > 1) {
            return Ok(None);
        }
    }
    // Step 6: when the host has assembled the per-row primary jets (Step 5)
    // and the family supplied the primary→coefficient Jacobian rows, fold them
    // into the coefficient-space `(nll, grad)`.  This is the joint-β assembly
    // that was previously the `Ok(None)` sentinel; it is pure host algebra and
    // CPU-verifiable, and becomes the device contraction once the substrate
    // jet assembly lands.
    if let Some(rows) = step6 {
        // Prefer the on-device contraction; fall back to the host fold on
        // non-CUDA builds.  Both are bit-exact against each other.
        let (nll, grad, _hess) = match try_device_step6_joint_beta(rows, inputs.p)? {
            Some(triple) => triple,
            None => pullback_step6_joint_beta(rows, inputs.p)?,
        };
        return Ok(Some((nll, grad)));
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    Ok(None)
}

/// Evaluate the survival-flex joint-Hessian times a vector `v` on the
/// GPU.  Returns `H·v` ∈ ℝ^p, or `Ok(None)` for unsupported shapes.
///
/// When `step6` carries the assembled per-row primary jets + Jacobians the
/// product is the exact `H·v` from the Step-6 joint-β pullback; otherwise the
/// entry point returns `Ok(None)` so the caller falls back to CPU.
pub fn try_survival_flex_hvp(
    inputs: SurvivalFlexGpuRowInputs<'_>,
    v: &[f64],
    step6: Option<&[SurvivalFlexStep6RowPullback<'_>]>,
) -> Result<Option<Array1<f64>>, GpuError> {
    inputs.validate()?;
    if v.len() != inputs.p {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "survival_flex try_hvp: v.len()={} != p={}",
                v.len(),
                inputs.p
            ),
        });
    }
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if let Some(rows) = step6 {
        let (_nll, _grad, hess) = match try_device_step6_joint_beta(rows, inputs.p)? {
            Some(triple) => triple,
            None => pullback_step6_joint_beta(rows, inputs.p)?,
        };
        return Ok(Some(hess.dot(&Array1::from(v.to_vec()))));
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    Ok(None)
}

/// Assemble the dense survival-flex joint Hessian on the GPU.  Returns
/// a `p × p` row-major matrix, or `Ok(None)` for unsupported shapes.
///
/// When `cells` is `Some(_)` (Step 2 hookup) the entry point evaluates
/// the per-cell derivative moments via [`try_row_batched_cell_moments`]
/// first — this validates the moment-building stage end-to-end on the
/// device runtime.  When `step6` rows are supplied the entry point folds
/// them through the Step-6 joint-β pullback ([`pullback_step6_joint_beta`])
/// into the dense coefficient Hessian — the joint-β assembly is wired and
/// CPU-verified (see `tests::step6_joint_beta_pullback_matches_cpu_dense_assembly_flex_no_wiggle`).
/// Without `step6` rows the cell-moment precheck runs and the entry point
/// returns `Ok(None)` so the caller falls back to CPU; on any non-OK
/// substrate status the caller likewise falls back to CPU.
pub fn try_survival_flex_dense_hessian(
    inputs: SurvivalFlexGpuRowInputs<'_>,
    cells: Option<SurvivalFlexRowCellsBatch<'_>>,
    step6: Option<&[SurvivalFlexStep6RowPullback<'_>]>,
) -> Result<Option<Array2<f64>>, GpuError> {
    inputs.validate()?;
    if inputs.score_dim != 1 {
        return Ok(None);
    }
    if let Some(batch) = cells {
        // Validate the moment-building stage on the substrate runtime.
        // Step 4/5/6 will plug these moments into the joint-β
        // gradient/Hessian; here we only confirm the moments are
        // evaluatable so the dispatcher does not silently fall through
        // to CPU when the GPU substrate is healthy.
        let out = match try_row_batched_cell_moments(batch)? {
            Some(out) => out,
            None => return Ok(None),
        };
        let ok_byte = crate::gpu_kernels::cubic_cell::CubicCellMomentStatus::Ok as u8;
        if out.status.iter().any(|&b| b != ok_byte) {
            // Any cell that failed the substrate classifier or kernel is a CPU
            // fallback for this fit — we cannot stitch a partial answer from a
            // moment stage that did not fully evaluate.
            return Ok(None);
        }
    }
    // Step 6: fold the assembled per-row primary Hessians into the dense
    // coefficient-space joint Hessian via the joint-β pullback (on-device when a
    // CUDA backend is live, host fold otherwise — both bit-exact).
    if let Some(rows) = step6 {
        let (_nll, _grad, hess) = match try_device_step6_joint_beta(rows, inputs.p)? {
            Some(triple) => triple,
            None => pullback_step6_joint_beta(rows, inputs.p)?,
        };
        return Ok(Some(hess));
    }
    if !SurvivalFlexGpuBackend::compiled() {
        return Ok(None);
    }
    Ok(None)
}

// ────────────────────────────────────────────────────────────────────────
// Block 10 — third T_uv[r] and fourth Q_uv[r,s] directional contractions.
//
// Math reference: math block 10 §3.5 (third) and §3.6 (fourth).  The
// per-row CPU implementations are
// `SurvivalMarginalSlopeFamily::row_flex_primary_third_contracted_exact`
// and `_fourth_contracted_exact` in `src/families/survival_marginal_slope.rs`.
//
// These oracles are pure assemblers over the timepoint-jet substrate
// outputs (entry/exit base + per-direction extension + bidirectional).
// They mirror the CPU assembly term-for-term.
// ────────────────────────────────────────────────────────────────────────

/// Per-timepoint exact-jet substrate input.  Mirrors the crate-private
/// `SurvivalFlexTimepointExact` in `survival_marginal_slope.rs`.
/// All arrays are row-major dense over the primary dimension `p`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10TimepointBase {
    pub eta: f64,
    pub chi: f64,
    pub d: f64,
    pub eta_u: Vec<f64>,
    pub eta_uv: Vec<f64>,
    pub chi_u: Vec<f64>,
    pub chi_uv: Vec<f64>,
    pub d_u: Vec<f64>,
    pub d_uv: Vec<f64>,
}

/// Directional extension of a timepoint jet contracted with a single
/// direction `d ∈ ℝᵖ`.  Mirrors `SurvivalFlexTimepointDirectionalExact`.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10TimepointDirectional {
    pub eta_uv_dir: Vec<f64>,
    pub eta_u_dir: Vec<f64>,
    pub chi_u_dir: Vec<f64>,
    pub chi_uv_dir: Vec<f64>,
    pub d_u_dir: Vec<f64>,
    pub d_uv_dir: Vec<f64>,
}

/// Mixed second-directional extension `D_{d1} D_{d2}` of a timepoint jet.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10TimepointBiDirectional {
    pub eta_uv_uv: Vec<f64>,
    pub chi_uv_uv: Vec<f64>,
    pub d_uv_uv: Vec<f64>,
}

/// Inputs to the Block 10 third-contraction CPU oracle.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10ThirdInputs<'a> {
    pub p: usize,
    /// Index of the `qd1` primary coordinate; `usize::MAX` to disable.
    pub qd1_index: usize,
    pub qd1: f64,
    pub w: f64,
    pub d: f64,
    pub dir: &'a [f64],
    pub entry_base: &'a SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a SurvivalFlexBlock10TimepointBase,
    pub entry_ext: &'a SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext: &'a SurvivalFlexBlock10TimepointDirectional,
}

/// Inputs to the Block 10 fourth-contraction CPU oracle.
#[derive(Clone, Debug)]
pub struct SurvivalFlexBlock10FourthInputs<'a> {
    pub p: usize,
    pub qd1_index: usize,
    pub qd1: f64,
    pub w: f64,
    pub d: f64,
    pub dir_u: &'a [f64],
    pub dir_v: &'a [f64],
    pub entry_base: &'a SurvivalFlexBlock10TimepointBase,
    pub exit_base: &'a SurvivalFlexBlock10TimepointBase,
    pub entry_ext_u: &'a SurvivalFlexBlock10TimepointDirectional,
    pub entry_ext_v: &'a SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_u: &'a SurvivalFlexBlock10TimepointDirectional,
    pub exit_ext_v: &'a SurvivalFlexBlock10TimepointDirectional,
    pub entry_bi: &'a SurvivalFlexBlock10TimepointBiDirectional,
    pub exit_bi: &'a SurvivalFlexBlock10TimepointBiDirectional,
}

#[inline]
fn b10_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "b10_dot: length mismatch");
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[inline]
fn b10_mat_dot(m: &[f64], v: &[f64], p: usize) -> Vec<f64> {
    assert_eq!(m.len(), p * p, "b10_mat_dot: matrix shape mismatch");
    assert_eq!(v.len(), p, "b10_mat_dot: vector length mismatch");
    let mut out = vec![0.0_f64; p];
    for u in 0..p {
        let mut acc = 0.0_f64;
        let row = &m[u * p..(u + 1) * p];
        for k in 0..p {
            acc += row[k] * v[k];
        }
        out[u] = acc;
    }
    out
}

#[inline]
fn b10_at(m: &[f64], u: usize, v: usize, p: usize) -> f64 {
    m[u * p + v]
}

/// CPU oracle for the third directional contraction `T_uv[r] :=
/// (D_{dir} H)[u, v]` of the flexible survival path.  Pure assembler;
/// mirrors `row_flex_primary_third_contracted_exact` term-for-term.
pub fn cpu_oracle_third_contraction(
    inputs: &SurvivalFlexBlock10ThirdInputs<'_>,
) -> Result<Vec<f64>, String> {
    let p = inputs.p;
    if inputs.dir.len() != p {
        return Err(format!(
            "cpu_oracle_third_contraction: dir length {} != p {}",
            inputs.dir.len(),
            p
        ));
    }
    if inputs.dir.iter().all(|v| v.abs() == 0.0) {
        return Ok(vec![0.0_f64; p * p]);
    }
    let entry = inputs.entry_base;
    let exit = inputs.exit_base;
    let entry_ext = inputs.entry_ext;
    let exit_ext = inputs.exit_ext;
    let chi = exit.chi;
    if !chi.is_finite() || chi <= 0.0 {
        return Err(format!(
            "cpu_oracle_third_contraction: non-positive chi={chi:.3e}"
        ));
    }
    let d_val = exit.d;
    if !d_val.is_finite() || d_val == 0.0 {
        return Err(format!(
            "cpu_oracle_third_contraction: non-finite/zero D={d_val:.3e}"
        ));
    }

    let wi = inputs.w;
    let di = inputs.d;

    use crate::bms::signed_probit_neglog_derivatives_up_to_fourth;
    let (entry_k1, entry_k2, entry_k3, _) =
        signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
    let (exit_k1, exit_k2, exit_k3, _) =
        signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;

    let entry_u1 = -entry_k1;
    let entry_u2 = entry_k2;
    let entry_u3 = -entry_k3;
    let exit_u1 = -exit_k1;
    let exit_u2 = exit_k2;
    let exit_u3 = -exit_k3;

    let entry_eta_dir = b10_dot(&entry.eta_u, inputs.dir);
    let exit_eta_dir = b10_dot(&exit.eta_u, inputs.dir);
    let exit_chi_dir = b10_dot(&exit.chi_u, inputs.dir);
    let exit_d_dir = b10_dot(&exit.d_u, inputs.dir);
    let qd1_dir = if inputs.qd1_index < p {
        inputs.dir[inputs.qd1_index]
    } else {
        0.0
    };

    let entry_eta_u_dir = &entry_ext.eta_u_dir;
    let exit_eta_u_dir = &exit_ext.eta_u_dir;
    let exit_chi_u_dir = &exit_ext.chi_u_dir;
    let exit_d_u_dir = &exit_ext.d_u_dir;

    let chi_inv = 1.0 / chi;
    let chi_inv2 = chi_inv * chi_inv;
    let chi_inv3 = chi_inv2 * chi_inv;
    let d_inv = 1.0 / d_val;
    let d_inv2 = d_inv * d_inv;
    let d_inv3 = d_inv2 * d_inv;

    let mut out = vec![0.0_f64; p * p];
    for u in 0..p {
        for v in u..p {
            let mut val = 0.0_f64;

            // Entry probit
            val += entry_u3 * entry_eta_dir * entry.eta_u[u] * entry.eta_u[v];
            val += entry_u2
                * (entry_eta_u_dir[u] * entry.eta_u[v] + entry.eta_u[u] * entry_eta_u_dir[v]);
            val += entry_u2 * entry_eta_dir * b10_at(&entry.eta_uv, u, v, p);
            val += entry_u1 * b10_at(&entry_ext.eta_uv_dir, u, v, p);

            // Exit probit survival
            val += exit_u3 * exit_eta_dir * exit.eta_u[u] * exit.eta_u[v];
            val +=
                exit_u2 * (exit_eta_u_dir[u] * exit.eta_u[v] + exit.eta_u[u] * exit_eta_u_dir[v]);
            val += exit_u2 * exit_eta_dir * b10_at(&exit.eta_uv, u, v, p);
            val += exit_u1 * b10_at(&exit_ext.eta_uv_dir, u, v, p);

            // Event density
            val += wi
                * di
                * (exit_eta_u_dir[u] * exit.eta_u[v]
                    + exit.eta_u[u] * exit_eta_u_dir[v]
                    + exit_eta_dir * b10_at(&exit.eta_uv, u, v, p)
                    + exit.eta * b10_at(&exit_ext.eta_uv_dir, u, v, p));

            // Event chi
            let chi_uv_over_chi_dir = (b10_at(&exit_ext.chi_uv_dir, u, v, p) * chi
                - b10_at(&exit.chi_uv, u, v, p) * exit_chi_dir)
                * chi_inv2;
            let chi_u_chi_v_over_chi2_dir =
                (exit_chi_u_dir[u] * exit.chi_u[v] + exit.chi_u[u] * exit_chi_u_dir[v]) * chi_inv2
                    - 2.0 * exit.chi_u[u] * exit.chi_u[v] * exit_chi_dir * chi_inv3;
            val -= wi * di * (chi_uv_over_chi_dir - chi_u_chi_v_over_chi2_dir);

            // Event D
            let d_uv_over_d_dir = (b10_at(&exit_ext.d_uv_dir, u, v, p) * d_val
                - b10_at(&exit.d_uv, u, v, p) * exit_d_dir)
                * d_inv2;
            let d_u_d_v_over_d2_dir =
                (exit_d_u_dir[u] * exit.d_u[v] + exit.d_u[u] * exit_d_u_dir[v]) * d_inv2
                    - 2.0 * exit.d_u[u] * exit.d_u[v] * exit_d_dir * d_inv3;
            val += wi * di * (d_uv_over_d_dir - d_u_d_v_over_d2_dir);

            // qd1 term
            if inputs.qd1_index < p && u == inputs.qd1_index && v == inputs.qd1_index {
                val += wi * di * (-2.0 / (inputs.qd1 * inputs.qd1 * inputs.qd1)) * qd1_dir;
            }

            out[u * p + v] = val;
            out[v * p + u] = val;
        }
    }
    Ok(out)
}

/// One ordered fourth contracted matrix `D_{dir2}(D_{dir1} H)`; mirrors
/// `compute_survival_fourth_contracted_ordered`.
fn b10_fourth_ordered(
    p: usize,
    qd1_index: usize,
    qd1: f64,
    wi: f64,
    di: f64,
    dir1: &[f64],
    dir2: &[f64],
    entry_base: &SurvivalFlexBlock10TimepointBase,
    exit_base: &SurvivalFlexBlock10TimepointBase,
    entry_ext1: &SurvivalFlexBlock10TimepointDirectional,
    entry_ext2: &SurvivalFlexBlock10TimepointDirectional,
    exit_ext1: &SurvivalFlexBlock10TimepointDirectional,
    exit_ext2: &SurvivalFlexBlock10TimepointDirectional,
    entry_bi: &SurvivalFlexBlock10TimepointBiDirectional,
    exit_bi: &SurvivalFlexBlock10TimepointBiDirectional,
) -> Result<Vec<f64>, String> {
    use crate::bms::signed_probit_neglog_derivatives_up_to_fourth;

    let (entry_k1, entry_k2, entry_k3, entry_k4) =
        signed_probit_neglog_derivatives_up_to_fourth(-entry_base.eta, -wi)?;
    let (exit_k1, exit_k2, exit_k3, exit_k4) =
        signed_probit_neglog_derivatives_up_to_fourth(-exit_base.eta, wi * (1.0 - di))?;

    let entry_u1 = -entry_k1;
    let entry_u2 = entry_k2;
    let entry_u3 = -entry_k3;
    let exit_u1 = -exit_k1;
    let exit_u2 = exit_k2;
    let exit_u3 = -exit_k3;

    let entry_eta_d1 = b10_dot(&entry_base.eta_u, dir1);
    let entry_eta_d2 = b10_dot(&entry_base.eta_u, dir2);
    let exit_eta_d1 = b10_dot(&exit_base.eta_u, dir1);
    let exit_eta_d2 = b10_dot(&exit_base.eta_u, dir2);
    let exit_chi_d1 = b10_dot(&exit_base.chi_u, dir1);
    let exit_chi_d2 = b10_dot(&exit_base.chi_u, dir2);
    let exit_d_d1 = b10_dot(&exit_base.d_u, dir1);
    let exit_d_d2 = b10_dot(&exit_base.d_u, dir2);
    let qd1_d1 = if qd1_index < p { dir1[qd1_index] } else { 0.0 };
    let qd1_d2 = if qd1_index < p { dir2[qd1_index] } else { 0.0 };

    let entry_eta_u_d1 = entry_ext1.eta_u_dir.clone();
    let entry_eta_u_d2 = entry_ext2.eta_u_dir.clone();
    let exit_eta_u_d1 = exit_ext1.eta_u_dir.clone();
    let exit_eta_u_d2 = exit_ext2.eta_u_dir.clone();
    let exit_chi_u_d1 = b10_mat_dot(&exit_base.chi_uv, dir1, p);
    let exit_chi_u_d2 = b10_mat_dot(&exit_base.chi_uv, dir2, p);
    let exit_d_u_d2 = b10_mat_dot(&exit_base.d_uv, dir2, p);

    let entry_eta_d12 = b10_dot(&entry_eta_u_d2, dir1);
    let exit_eta_d12 = b10_dot(&exit_eta_u_d2, dir1);
    let exit_chi_d12 = b10_dot(&exit_chi_u_d2, dir1);
    let exit_d_d12 = b10_dot(&exit_d_u_d2, dir1);

    let entry_eta_u_d12 = b10_mat_dot(&entry_ext2.eta_uv_dir, dir1, p);
    let exit_eta_u_d12 = b10_mat_dot(&exit_ext2.eta_uv_dir, dir1, p);
    let exit_chi_u_d12 = b10_mat_dot(&exit_ext2.chi_uv_dir, dir1, p);
    let exit_d_u_d12 = b10_mat_dot(&exit_ext2.d_uv_dir, dir1, p);

    let chi = exit_base.chi;
    let chi_inv = 1.0 / chi;
    let chi_inv2 = chi_inv * chi_inv;
    let chi_inv3 = chi_inv2 * chi_inv;
    let chi_inv4 = chi_inv3 * chi_inv;
    let d_val = exit_base.d;
    let d_inv = 1.0 / d_val;
    let d_inv2 = d_inv * d_inv;
    let d_inv3 = d_inv2 * d_inv;
    let d_inv4 = d_inv3 * d_inv;

    let mut out = vec![0.0_f64; p * p];
    for u in 0..p {
        for v in u..p {
            let mut val = 0.0_f64;

            // Entry probit
            let eu = &entry_base.eta_u;
            let euv_uv = b10_at(&entry_base.eta_uv, u, v, p);

            let a_term = eu[u] * eu[v] * entry_eta_d1;
            let a_term_d2 = entry_eta_u_d2[u] * eu[v] * entry_eta_d1
                + eu[u] * entry_eta_u_d2[v] * entry_eta_d1
                + eu[u] * eu[v] * entry_eta_d12;
            let b_term = b10_at(&entry_ext1.eta_uv_dir, u, v, p);
            let b_term_d2 = b10_at(&entry_bi.eta_uv_uv, u, v, p);
            let c_term =
                entry_eta_u_d1[u] * eu[v] + eu[u] * entry_eta_u_d1[v] + entry_eta_d1 * euv_uv;
            let c_term_d2 = entry_eta_u_d12[u] * eu[v]
                + entry_eta_u_d1[u] * entry_eta_u_d2[v]
                + entry_eta_u_d2[u] * entry_eta_u_d1[v]
                + eu[u] * entry_eta_u_d12[v]
                + entry_eta_d12 * euv_uv
                + entry_eta_d1 * b10_at(&entry_ext2.eta_uv_dir, u, v, p);

            val += entry_k4 * entry_eta_d2 * a_term
                + entry_u3 * a_term_d2
                + entry_u3 * entry_eta_d2 * c_term
                + entry_u2 * c_term_d2
                + entry_u2 * entry_eta_d2 * b_term
                + entry_u1 * b_term_d2;

            // Exit probit
            let xu = &exit_base.eta_u;
            let xuv_uv = b10_at(&exit_base.eta_uv, u, v, p);

            let xa = xu[u] * xu[v] * exit_eta_d1;
            let xa_d2 = exit_eta_u_d2[u] * xu[v] * exit_eta_d1
                + xu[u] * exit_eta_u_d2[v] * exit_eta_d1
                + xu[u] * xu[v] * exit_eta_d12;
            let xb = b10_at(&exit_ext1.eta_uv_dir, u, v, p);
            let xb_d2 = b10_at(&exit_bi.eta_uv_uv, u, v, p);
            let xc = exit_eta_u_d1[u] * xu[v] + xu[u] * exit_eta_u_d1[v] + exit_eta_d1 * xuv_uv;
            let xc_d2 = exit_eta_u_d12[u] * xu[v]
                + exit_eta_u_d1[u] * exit_eta_u_d2[v]
                + exit_eta_u_d2[u] * exit_eta_u_d1[v]
                + xu[u] * exit_eta_u_d12[v]
                + exit_eta_d12 * xuv_uv
                + exit_eta_d1 * b10_at(&exit_ext2.eta_uv_dir, u, v, p);

            val += exit_k4 * exit_eta_d2 * xa
                + exit_u3 * xa_d2
                + exit_u3 * exit_eta_d2 * xc
                + exit_u2 * xc_d2
                + exit_u2 * exit_eta_d2 * xb
                + exit_u1 * xb_d2;

            // Event density
            val += wi
                * di
                * (exit_eta_u_d12[u] * xu[v]
                    + exit_eta_u_d1[u] * exit_eta_u_d2[v]
                    + exit_eta_u_d2[u] * exit_eta_u_d1[v]
                    + xu[u] * exit_eta_u_d12[v]
                    + exit_eta_d12 * xuv_uv
                    + exit_eta_d1 * b10_at(&exit_ext2.eta_uv_dir, u, v, p)
                    + exit_eta_d2 * b10_at(&exit_ext1.eta_uv_dir, u, v, p)
                    + exit_base.eta * b10_at(&exit_bi.eta_uv_uv, u, v, p));

            // Event chi
            let chi_uv_val = b10_at(&exit_base.chi_uv, u, v, p);
            let chi_u_val = exit_base.chi_u[u];
            let chi_v_val = exit_base.chi_u[v];
            let chi_uv_d1 = b10_at(&exit_ext1.chi_uv_dir, u, v, p);
            let chi_uv_d2 = b10_at(&exit_ext2.chi_uv_dir, u, v, p);
            let chi_u_d1 = exit_chi_u_d1[u];
            let chi_v_d1 = exit_chi_u_d1[v];
            let chi_u_d2 = exit_chi_u_d2[u];
            let chi_v_d2 = exit_chi_u_d2[v];
            let chi_u_d12v = exit_chi_u_d12[u];
            let chi_v_d12v = exit_chi_u_d12[v];

            let chi_uv_d12_val = b10_at(&exit_bi.chi_uv_uv, u, v, p);
            let d2_r_chi = chi_uv_d12_val * chi_inv
                - chi_uv_d1 * exit_chi_d2 * chi_inv2
                - chi_uv_d2 * exit_chi_d1 * chi_inv2
                - chi_uv_val * exit_chi_d12 * chi_inv2
                + 2.0 * chi_uv_val * exit_chi_d1 * exit_chi_d2 * chi_inv3;

            let d2_s_chi = (chi_u_d12v * chi_v_val
                + chi_u_d1 * chi_v_d2
                + chi_u_d2 * chi_v_d1
                + chi_u_val * chi_v_d12v)
                * chi_inv2
                - 2.0 * (chi_u_d1 * chi_v_val + chi_u_val * chi_v_d1) * exit_chi_d2 * chi_inv3
                - 2.0 * (chi_u_d2 * chi_v_val + chi_u_val * chi_v_d2) * exit_chi_d1 * chi_inv3
                - 2.0 * chi_u_val * chi_v_val * exit_chi_d12 * chi_inv3
                + 6.0 * chi_u_val * chi_v_val * exit_chi_d1 * exit_chi_d2 * chi_inv4;
            val -= wi * di * (d2_r_chi - d2_s_chi);

            // Event D
            let d_uv_val = b10_at(&exit_base.d_uv, u, v, p);
            let d_u_val = exit_base.d_u[u];
            let d_v_val = exit_base.d_u[v];
            let d_uv_d1 = b10_at(&exit_ext1.d_uv_dir, u, v, p);
            let d_uv_d2 = b10_at(&exit_ext2.d_uv_dir, u, v, p);
            let d_u_d1 = exit_ext1.d_u_dir[u];
            let d_v_d1 = exit_ext1.d_u_dir[v];
            let d_u_d2 = exit_ext2.d_u_dir[u];
            let d_v_d2 = exit_ext2.d_u_dir[v];
            let d_u_d12v = exit_d_u_d12[u];
            let d_v_d12v = exit_d_u_d12[v];

            let d_uv_d12_val = b10_at(&exit_bi.d_uv_uv, u, v, p);
            let d2_r_d = d_uv_d12_val * d_inv
                - d_uv_d1 * exit_d_d2 * d_inv2
                - d_uv_d2 * exit_d_d1 * d_inv2
                - d_uv_val * exit_d_d12 * d_inv2
                + 2.0 * d_uv_val * exit_d_d1 * exit_d_d2 * d_inv3;

            let d2_s_d =
                (d_u_d12v * d_v_val + d_u_d1 * d_v_d2 + d_u_d2 * d_v_d1 + d_u_val * d_v_d12v)
                    * d_inv2
                    - 2.0 * (d_u_d1 * d_v_val + d_u_val * d_v_d1) * exit_d_d2 * d_inv3
                    - 2.0 * (d_u_d2 * d_v_val + d_u_val * d_v_d2) * exit_d_d1 * d_inv3
                    - 2.0 * d_u_val * d_v_val * exit_d_d12 * d_inv3
                    + 6.0 * d_u_val * d_v_val * exit_d_d1 * exit_d_d2 * d_inv4;
            val += wi * di * (d2_r_d - d2_s_d);

            // qd1 term
            if qd1_index < p && u == qd1_index && v == qd1_index {
                val += wi * di * (6.0 / (qd1 * qd1 * qd1 * qd1)) * qd1_d1 * qd1_d2;
            }

            out[u * p + v] = val;
            out[v * p + u] = val;
        }
    }
    Ok(out)
}

/// CPU oracle for the fourth directional contraction with averaged
/// symmetrization `Q_sym = ½(Q_ordered[u, v] + Q_ordered[v, u])`.
/// Mirrors `row_flex_primary_fourth_contracted_exact`.
pub fn cpu_oracle_fourth_contraction(
    inputs: &SurvivalFlexBlock10FourthInputs<'_>,
) -> Result<Vec<f64>, String> {
    let p = inputs.p;
    if inputs.dir_u.len() != p || inputs.dir_v.len() != p {
        return Err(format!(
            "cpu_oracle_fourth_contraction: dir lengths ({},{}) != p {}",
            inputs.dir_u.len(),
            inputs.dir_v.len(),
            p
        ));
    }
    if inputs.dir_u.iter().all(|v| v.abs() == 0.0) || inputs.dir_v.iter().all(|v| v.abs() == 0.0) {
        return Ok(vec![0.0_f64; p * p]);
    }
    let chi = inputs.exit_base.chi;
    if !chi.is_finite() || chi <= 0.0 {
        return Err(format!(
            "cpu_oracle_fourth_contraction: non-positive chi={chi:.3e}"
        ));
    }
    let d_val = inputs.exit_base.d;
    if !d_val.is_finite() || d_val == 0.0 {
        return Err(format!(
            "cpu_oracle_fourth_contraction: non-finite/zero D={d_val:.3e}"
        ));
    }

    let ordered_uv = b10_fourth_ordered(
        p,
        inputs.qd1_index,
        inputs.qd1,
        inputs.w,
        inputs.d,
        inputs.dir_u,
        inputs.dir_v,
        inputs.entry_base,
        inputs.exit_base,
        inputs.entry_ext_u,
        inputs.entry_ext_v,
        inputs.exit_ext_u,
        inputs.exit_ext_v,
        inputs.entry_bi,
        inputs.exit_bi,
    )?;
    let ordered_vu = b10_fourth_ordered(
        p,
        inputs.qd1_index,
        inputs.qd1,
        inputs.w,
        inputs.d,
        inputs.dir_v,
        inputs.dir_u,
        inputs.entry_base,
        inputs.exit_base,
        inputs.entry_ext_v,
        inputs.entry_ext_u,
        inputs.exit_ext_v,
        inputs.exit_ext_u,
        inputs.entry_bi,
        inputs.exit_bi,
    )?;

    let mut out = vec![0.0_f64; p * p];
    for i in 0..(p * p) {
        out[i] = 0.5 * (ordered_uv[i] + ordered_vu[i]);
    }
    Ok(out)
}

#[cfg(test)]
mod step6_tests {
    use super::*;

    /// Build a reference coefficient-space pullback by the textbook
    /// quadruple-/triple-loop contraction (`Σ_rows Jᵀ g_p`, `Σ_rows Jᵀ H_p J`)
    /// so the production `pullback_step6_joint_beta` (which uses the blocked
    /// `M = H_p J` intermediate) is checked against an independent assembly.
    fn reference_pullback(
        rows: &[(f64, Vec<f64>, Vec<f64>, Vec<f64>)],
        r: usize,
        p: usize,
    ) -> (f64, Vec<f64>, Vec<f64>) {
        let mut nll = 0.0;
        let mut grad = vec![0.0_f64; p];
        let mut hess = vec![0.0_f64; p * p];
        for (row_nll, g_p, h_p, jac) in rows {
            nll += row_nll;
            for a in 0..r {
                for j in 0..p {
                    grad[j] += jac[a * p + j] * g_p[a];
                }
            }
            for a in 0..r {
                for b in 0..r {
                    let hab = h_p[a * r + b];
                    for j in 0..p {
                        for k in 0..p {
                            hess[j * p + k] += jac[a * p + j] * hab * jac[b * p + k];
                        }
                    }
                }
            }
        }
        (nll, grad, hess)
    }

    #[test]
    fn step6_pullback_matches_reference_contraction() {
        // r=3 primaries, p=4 coefficients, 2 rows. Hand-built jets + Jacobians.
        let r = 3usize;
        let p = 4usize;
        let row_specs: Vec<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> = vec![
            (
                1.5,
                vec![0.3, -0.7, 1.1],
                // symmetric 3x3
                vec![
                    2.0, -0.5, 0.4, //
                    -0.5, 1.3, 0.2, //
                    0.4, 0.2, 0.9, //
                ],
                // 3x4 Jacobian
                vec![
                    1.0, 0.0, 0.5, -0.2, //
                    0.0, 1.0, 0.0, 0.3, //
                    0.7, -0.1, 1.0, 0.0, //
                ],
            ),
            (
                -0.25,
                vec![-1.2, 0.4, 0.6],
                vec![
                    1.1, 0.3, -0.2, //
                    0.3, 0.8, 0.5, //
                    -0.2, 0.5, 1.4, //
                ],
                vec![
                    0.2, 1.0, 0.0, 0.0, //
                    -0.4, 0.0, 1.0, 0.6, //
                    0.0, 0.3, 0.0, 1.0, //
                ],
            ),
        ];

        let primary_outputs: Vec<SurvivalFlexStep5RowOutputs> = row_specs
            .iter()
            .map(|(nll, g, h, _)| SurvivalFlexStep5RowOutputs {
                row_nll: *nll,
                grad: g.clone(),
                hess: h.clone(),
            })
            .collect();
        let pullbacks: Vec<SurvivalFlexStep6RowPullback<'_>> = primary_outputs
            .iter()
            .zip(row_specs.iter())
            .map(|(po, (_, _, _, jac))| SurvivalFlexStep6RowPullback {
                primary: po,
                jacobian: jac,
            })
            .collect();

        let (nll, grad, hess) = pullback_step6_joint_beta(&pullbacks, p).expect("step6 pullback");
        let (ref_nll, ref_grad, ref_hess) = reference_pullback(&row_specs, r, p);

        assert!((nll - ref_nll).abs() < 1e-12, "nll mismatch");
        for j in 0..p {
            assert!(
                (grad[j] - ref_grad[j]).abs() < 1e-12,
                "grad[{j}] {} vs {}",
                grad[j],
                ref_grad[j]
            );
        }
        for j in 0..p {
            for k in 0..p {
                assert!(
                    (hess[[j, k]] - ref_hess[j * p + k]).abs() < 1e-12,
                    "hess[{j},{k}] {} vs {}",
                    hess[[j, k]],
                    ref_hess[j * p + k]
                );
                // Exactly symmetric after the symmetrization pass.
                assert_eq!(hess[[j, k]], hess[[k, j]], "H not symmetric at ({j},{k})");
            }
        }
    }

    #[test]
    fn step6_identity_jacobian_is_block_sum_of_primaries() {
        // With r == p and J = I per row, the pullback is just the row-sum of
        // the primary gradients / Hessians — a sanity anchor on the contraction.
        let p = 3usize;
        let g0 = vec![1.0, -2.0, 0.5];
        let h0 = vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0, //
        ];
        let g1 = vec![0.25, 0.25, -1.0];
        let h1 = vec![
            0.5, 0.1, 0.0, //
            0.1, 0.5, 0.0, //
            0.0, 0.0, 0.5, //
        ];
        let eye = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
        ];
        let outs = [
            SurvivalFlexStep5RowOutputs {
                row_nll: 2.0,
                grad: g0.clone(),
                hess: h0.clone(),
            },
            SurvivalFlexStep5RowOutputs {
                row_nll: 3.0,
                grad: g1.clone(),
                hess: h1.clone(),
            },
        ];
        let pb = [
            SurvivalFlexStep6RowPullback {
                primary: &outs[0],
                jacobian: &eye,
            },
            SurvivalFlexStep6RowPullback {
                primary: &outs[1],
                jacobian: &eye,
            },
        ];
        let (nll, grad, hess) = pullback_step6_joint_beta(&pb, p).expect("identity pullback");
        assert_eq!(nll, 5.0);
        for j in 0..p {
            assert!((grad[j] - (g0[j] + g1[j])).abs() < 1e-14);
            for k in 0..p {
                assert!((hess[[j, k]] - (h0[j * p + k] + h1[j * p + k])).abs() < 1e-14);
            }
        }
    }

    fn minimal_gpu_row_inputs<'a>(
        n: usize,
        p: usize,
        beta: &'a [f64],
        q0: &'a [f64],
        q1: &'a [f64],
        qd1: &'a [f64],
        z: &'a [f64],
        g: &'a [f64],
        weights: &'a [f64],
        event: &'a [f64],
    ) -> SurvivalFlexGpuRowInputs<'a> {
        SurvivalFlexGpuRowInputs {
            n,
            r: 3,
            p,
            score_dim: 1,
            beta,
            q0,
            q1,
            qd1,
            z,
            g,
            weights,
            event,
            derivative_guard: 1.0e-8,
            probit_scale: 1.0,
        }
    }

    #[test]
    fn flex_entrypoints_fold_supplied_step6_rows_before_backend_gate() {
        let n = 2usize;
        let p = 4usize;
        let beta = vec![0.0; p];
        let q0 = vec![0.1, 0.2];
        let q1 = vec![0.3, 0.4];
        let qd1 = vec![1.1, 1.2];
        let z = vec![-0.2, 0.6];
        let g = vec![0.05, -0.03];
        let weights = vec![1.0, 0.7];
        let event = vec![1.0, 0.0];

        let primary_rows = [
            SurvivalFlexStep5RowOutputs {
                row_nll: 1.5,
                grad: vec![0.3, -0.7, 1.1],
                hess: vec![
                    2.0, -0.5, 0.4, //
                    -0.5, 1.3, 0.2, //
                    0.4, 0.2, 0.9, //
                ],
            },
            SurvivalFlexStep5RowOutputs {
                row_nll: -0.25,
                grad: vec![-1.2, 0.4, 0.6],
                hess: vec![
                    1.1, 0.3, -0.2, //
                    0.3, 0.8, 0.5, //
                    -0.2, 0.5, 1.4, //
                ],
            },
        ];
        let jacobians = [
            vec![
                1.0, 0.0, 0.5, -0.2, //
                0.0, 1.0, 0.0, 0.3, //
                0.7, -0.1, 1.0, 0.0, //
            ],
            vec![
                0.2, 1.0, 0.0, 0.0, //
                -0.4, 0.0, 1.0, 0.6, //
                0.0, 0.3, 0.0, 1.0, //
            ],
        ];
        let step6_rows = [
            SurvivalFlexStep6RowPullback {
                primary: &primary_rows[0],
                jacobian: &jacobians[0],
            },
            SurvivalFlexStep6RowPullback {
                primary: &primary_rows[1],
                jacobian: &jacobians[1],
            },
        ];

        let (expected_nll, expected_grad, expected_hess) =
            pullback_step6_joint_beta(&step6_rows, p).expect("reference step6");

        // Parity band (#415 / #1175): the host `pullback_step6_joint_beta`
        // reference sums each row's contraction in a fixed scalar order; on a
        // CUDA host the entry points route the SAME contraction through the
        // device kernel `survival_flex_step6_rows`, whose per-row `M = H_p·J`
        // reduction reassociates (sums in a different order). FMA is off under
        // #1686's --fmad=false, so that is an irreducible ~ULP reassociation
        // difference, NOT an algebra mismatch, so a bit-exact `assert_eq!` is
        // wrong on a GPU box (it fails by ~2.8e-14 measured on a V100).
        // Use a relative band `atol + rtol·(1+|expected|)` — the SAME principle
        // as the sibling `step6_device_contraction_matches_cpu_reference` — which
        // a real contraction bug (O(value)) would still blow through.
        const ATOL: f64 = 1e-12;
        const RTOL: f64 = 1e-12;
        let close = |got: f64, want: f64, what: &str| {
            let tol = ATOL + RTOL * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= tol,
                "{what}: got {got} vs want {want} (|Δ|={:.3e} > tol {tol:.3e})",
                (got - want).abs()
            );
        };

        let inputs = minimal_gpu_row_inputs(n, p, &beta, &q0, &q1, &qd1, &z, &g, &weights, &event);
        let (nll, grad) = try_survival_flex_gradient(inputs, None, Some(&step6_rows))
            .expect("gradient entrypoint")
            .expect("step6 gradient should be assembled before backend gate");
        close(nll, expected_nll, "nll");
        for k in 0..p {
            close(grad[k], expected_grad[k], &format!("grad[{k}]"));
        }

        let v = vec![0.25, -0.5, 0.75, -1.0];
        let inputs = minimal_gpu_row_inputs(n, p, &beta, &q0, &q1, &qd1, &z, &g, &weights, &event);
        let hv = try_survival_flex_hvp(inputs, &v, Some(&step6_rows))
            .expect("hvp entrypoint")
            .expect("step6 hvp should be assembled before backend gate");
        let expected_hv = expected_hess.dot(&Array1::from(v));
        for k in 0..p {
            close(hv[k], expected_hv[k], &format!("hv[{k}]"));
        }

        let inputs = minimal_gpu_row_inputs(n, p, &beta, &q0, &q1, &qd1, &z, &g, &weights, &event);
        let hess = try_survival_flex_dense_hessian(inputs, None, Some(&step6_rows))
            .expect("dense hessian entrypoint")
            .expect("step6 dense Hessian should be assembled before backend gate");
        for a in 0..p {
            for b in 0..p {
                close(
                    hess[[a, b]],
                    expected_hess[[a, b]],
                    &format!("hess[{a},{b}]"),
                );
            }
        }
    }

    /// Build a deterministic, varied Step-6 batch (mixed r per row, sparse
    /// zeros in g_p / H_p / J so the kernel's zero-skip branches are exercised)
    /// for the GPU-vs-CPU parity test.
    fn varied_step6_rows(
        n_rows: usize,
        p: usize,
    ) -> (Vec<SurvivalFlexStep5RowOutputs>, Vec<Vec<f64>>) {
        let mut outs = Vec::with_capacity(n_rows);
        let mut jacs = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            // r alternates in {2,3,4} but never exceeds p.
            let r = (2 + (row % 3)).min(p);
            let mut g = vec![0.0_f64; r];
            let mut h = vec![0.0_f64; r * r];
            let mut jac = vec![0.0_f64; r * p];
            for a in 0..r {
                // Sparse-ish primary gradient.
                g[a] = if (a + row) % 4 == 0 {
                    0.0
                } else {
                    0.3 * (a as f64 + 1.0) - 0.17 * (row as f64) + 0.05 * (a * row) as f64
                };
                for b in a..r {
                    let v = if (a + b + row) % 5 == 0 {
                        0.0
                    } else {
                        0.11 * ((a + 1) * (b + 1)) as f64 - 0.07 * (row as f64)
                            + 0.9 * (a == b) as i32 as f64
                    };
                    h[a * r + b] = v;
                    h[b * r + a] = v;
                }
                for j in 0..p {
                    jac[a * p + j] = if (a + j + 2 * row) % 3 == 0 {
                        0.0
                    } else {
                        0.5 - 0.13 * (j as f64) + 0.21 * (a as f64) - 0.04 * (row as f64)
                    };
                }
            }
            outs.push(SurvivalFlexStep5RowOutputs {
                row_nll: 0.37 * (row as f64) - 1.1,
                grad: g,
                hess: h,
            });
            jacs.push(jac);
        }
        (outs, jacs)
    }

    // ────────────────────────────────────────────────────────────────────
    // #415 CPU-verifiable device-arithmetic emulator for the Step-6 row kernel.
    //
    // `emulate_step6_row_pullback_device` is a device-free CPU transcription of
    // the per-row body of `SURVIVAL_FLEX_STEP6_SOURCE`
    // (`survival_flex_step6_rows`). It writes this row's dense `grad_row[p]` and
    // `hess_row[p*p]` using the SAME per-output accumulation order and zero-skip
    // guards the `.cu` uses:
    //
    //   grad_row[k]   = Σ_a J[a,k]·g_p[a]     (inner a; skip g_p[a]==0)
    //   M[a,k]        = Σ_b H_p[a,b]·J[b,k]    (inner b; skip H_p[a,b]==0)
    //   hess_row[c,k] = Σ_a J[a,c]·M[a,k]      (inner a; skip J[a,c]==0)
    //
    // `emulate_step6_joint_beta_device` wraps it with the SAME shape validation
    // (`Step6DeviceBatch::build`), row-order NLL/partials reduction, and
    // off-diagonal symmetrization the device launch path
    // (`launch_step6_joint_beta_linux`) performs on the host. Because both
    // functions mirror the device program op-for-op, a `.cu` edit that changes
    // the arithmetic without a lockstep edit here makes
    // `step6_device_emulator_matches_host_pullback_415` fail on EVERY box — no
    // CUDA required — so device-algebra drift is caught in CPU CI, long before
    // it reaches the GPU runner (where `step6_device_contraction_matches_cpu_reference`
    // pins the compiled kernel and the `.cu`↔emulator fingerprint below reminds).
    //
    // Honest reach (identical to the survival_rowjet host-oracle lock): this
    // emulator is a hand transcription, so an in-place `.cu` arithmetic edit is
    // caught on CPU CI only when the emulator is updated in lockstep. The
    // fingerprint guard makes a STRUCTURAL `.cu` change (a dropped skip-zero
    // branch, a renamed/re-indexed contraction) fail the build directly; the
    // un-mirrored `.cu` bytes themselves are pinned numerically on a GPU box by
    // `step6_device_contraction_matches_cpu_reference`.
    // ────────────────────────────────────────────────────────────────────

    /// Per-row transcription of `survival_flex_step6_rows`. `m_scratch` (len
    /// `r*p`), `grad_row` (len `p`) and `hess_row` (len `p*p`) are written in
    /// place. Loop nesting / skip-zero guards mirror the `.cu` exactly.
    fn emulate_step6_row_pullback_device(
        g_p: &[f64],
        h_p: &[f64],
        jac: &[f64],
        r: usize,
        p: usize,
        m_scratch: &mut [f64],
        grad_row: &mut [f64],
        hess_row: &mut [f64],
    ) {
        // 1) grad_row[k] = Σ_a J[a*p+k]·g_p[a]  (per-output k, inner a; skip 0).
        for k in 0..p {
            let mut acc = 0.0_f64;
            for a in 0..r {
                let ga = g_p[a];
                if ga != 0.0 {
                    acc += jac[a * p + k] * ga;
                }
            }
            grad_row[k] = acc;
        }
        // 2) M[a*p+k] = Σ_b H_p[a*r+b]·J[b*p+k]  (per (a,k), inner b; skip 0).
        for a in 0..r {
            for k in 0..p {
                let mut acc = 0.0_f64;
                for b in 0..r {
                    let hab = h_p[a * r + b];
                    if hab != 0.0 {
                        acc += hab * jac[b * p + k];
                    }
                }
                m_scratch[a * p + k] = acc;
            }
        }
        // 3) hess_row[col*p+k] = Σ_a J[a*p+col]·M[a*p+k]  (per (col,k), inner a; skip 0).
        for col in 0..p {
            for k in 0..p {
                let mut acc = 0.0_f64;
                for a in 0..r {
                    let jac_ac = jac[a * p + col];
                    if jac_ac != 0.0 {
                        acc += jac_ac * m_scratch[a * p + k];
                    }
                }
                hess_row[col * p + k] = acc;
            }
        }
    }

    /// Batch driver mirroring `try_device_step6_joint_beta`'s device path:
    /// validate shapes as `Step6DeviceBatch::build` does, emulate each row's
    /// dense partial, reduce over rows in row order, then symmetrize the
    /// off-diagonal exactly as `launch_step6_joint_beta_linux` does.
    fn emulate_step6_joint_beta_device(
        rows: &[SurvivalFlexStep6RowPullback<'_>],
        p: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), GpuError> {
        let mut nll = 0.0_f64;
        for (i, row) in rows.iter().enumerate() {
            let r = row.primary.grad.len();
            if row.primary.hess.len() != r * r {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "step6 emulator row {i}: primary.hess.len()={} expected r*r={}",
                        row.primary.hess.len(),
                        r * r
                    ),
                });
            }
            if row.jacobian.len() != r * p {
                return Err(GpuError::DriverCallFailed {
                    reason: format!(
                        "step6 emulator row {i}: jacobian.len()={} expected r*p={}",
                        row.jacobian.len(),
                        r * p
                    ),
                });
            }
            nll += row.primary.row_nll;
        }

        let mut grad = Array1::<f64>::zeros(p);
        let mut hess = Array2::<f64>::zeros((p, p));
        let mut grad_row = vec![0.0_f64; p];
        let mut hess_row = vec![0.0_f64; p * p];
        for row in rows {
            let r = row.primary.grad.len();
            let mut m = vec![0.0_f64; r * p];
            emulate_step6_row_pullback_device(
                &row.primary.grad,
                &row.primary.hess,
                row.jacobian,
                r,
                p,
                &mut m,
                &mut grad_row,
                &mut hess_row,
            );
            for k in 0..p {
                grad[k] += grad_row[k];
            }
            for col in 0..p {
                for k in 0..p {
                    hess[[col, k]] += hess_row[col * p + k];
                }
            }
        }
        // Same off-diagonal symmetrization pass the device host-reduction uses.
        for col in 0..p {
            for k in (col + 1)..p {
                let avg = 0.5 * (hess[[col, k]] + hess[[k, col]]);
                hess[[col, k]] = avg;
                hess[[k, col]] = avg;
            }
        }
        Ok((nll, grad, hess))
    }

    /// #415 CPU-verifiable parity lock: the device-arithmetic emulator
    /// reproduces the production host contraction `pullback_step6_joint_beta`
    /// (and, on the fixed-`r` fixture, the fully independent textbook
    /// `reference_pullback`) on every box. The two paths differ only by the
    /// floating-point ASSOCIATION order of each output's inner reduction (the
    /// emulator sums grad per-output-k / inner-a like the `.cu`; the host sums
    /// per-a / inner-k), so a mixed abs+rel band absorbs that reassociation
    /// while any real algebra drift (dropped term, sign flip, index swap) moves
    /// an output by O(value), not O(ε). The worst normalized drift is asserted
    /// to keep ~100× headroom so the band pins the transcription, not slack.
    #[test]
    fn step6_device_emulator_matches_host_pullback_415() {
        const ABS_TOL: f64 = 1e-12;
        const REL_TOL: f64 = 1e-11;
        let mut worst_ratio = 0.0_f64;
        let mut close = |a: f64, b: f64| -> bool {
            if a == b {
                return true;
            }
            let diff = (a - b).abs();
            let band = ABS_TOL + REL_TOL * a.abs().max(b.abs());
            worst_ratio = worst_ratio.max(diff / band);
            diff <= band
        };

        let mut checked = 0usize;

        // Arm A — varied mixed-r batches (exercise every zero-skip branch).
        for &(n_rows, p) in &[(1usize, 3usize), (5, 4), (37, 6), (64, 8)] {
            let (outs, jacs) = varied_step6_rows(n_rows, p);
            let rows: Vec<SurvivalFlexStep6RowPullback<'_>> = outs
                .iter()
                .zip(jacs.iter())
                .map(|(o, j)| SurvivalFlexStep6RowPullback {
                    primary: o,
                    jacobian: j.as_slice(),
                })
                .collect();

            let (host_nll, host_grad, host_hess) =
                pullback_step6_joint_beta(&rows, p).expect("host reference");
            let (emu_nll, emu_grad, emu_hess) =
                emulate_step6_joint_beta_device(&rows, p).expect("device emulator");

            assert!(
                close(emu_nll, host_nll),
                "nll drift (n={n_rows}, p={p}): emu {emu_nll} vs host {host_nll}"
            );
            for j in 0..p {
                assert!(
                    close(emu_grad[j], host_grad[j]),
                    "grad[{j}] drift (n={n_rows}, p={p}): emu {} vs host {}",
                    emu_grad[j],
                    host_grad[j]
                );
            }
            for a in 0..p {
                for b in 0..p {
                    assert!(
                        close(emu_hess[[a, b]], host_hess[[a, b]]),
                        "hess[{a},{b}] drift (n={n_rows}, p={p}): emu {} vs host {}",
                        emu_hess[[a, b]],
                        host_hess[[a, b]]
                    );
                    // Emulator must produce an exactly symmetric H (same
                    // symmetrization pass as the device host-reduction).
                    assert_eq!(
                        emu_hess[[a, b]],
                        emu_hess[[b, a]],
                        "emulator H not symmetric at ({a},{b}) (n={n_rows}, p={p})"
                    );
                    checked += 1;
                }
            }
        }

        // Arm B — fixed-r hand fixture cross-checked against the INDEPENDENT
        // textbook `reference_pullback` (Σ_rows Jᵀ H_p J), not just the
        // production host path, so the emulator is anchored to a second
        // implementation of the contraction.
        let r = 3usize;
        let p = 4usize;
        let row_specs: Vec<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> = vec![
            (
                1.5,
                vec![0.3, -0.7, 1.1],
                vec![
                    2.0, -0.5, 0.4, //
                    -0.5, 1.3, 0.2, //
                    0.4, 0.2, 0.9, //
                ],
                vec![
                    1.0, 0.0, 0.5, -0.2, //
                    0.0, 1.0, 0.0, 0.3, //
                    0.7, -0.1, 1.0, 0.0, //
                ],
            ),
            (
                -0.25,
                vec![-1.2, 0.4, 0.6],
                vec![
                    1.1, 0.3, -0.2, //
                    0.3, 0.8, 0.5, //
                    -0.2, 0.5, 1.4, //
                ],
                vec![
                    0.2, 1.0, 0.0, 0.0, //
                    -0.4, 0.0, 1.0, 0.6, //
                    0.0, 0.3, 0.0, 1.0, //
                ],
            ),
        ];
        let primary_outputs: Vec<SurvivalFlexStep5RowOutputs> = row_specs
            .iter()
            .map(|(nll, g, h, _)| SurvivalFlexStep5RowOutputs {
                row_nll: *nll,
                grad: g.clone(),
                hess: h.clone(),
            })
            .collect();
        let pullbacks: Vec<SurvivalFlexStep6RowPullback<'_>> = primary_outputs
            .iter()
            .zip(row_specs.iter())
            .map(|(po, (_, _, _, jac))| SurvivalFlexStep6RowPullback {
                primary: po,
                jacobian: jac,
            })
            .collect();
        let (emu_nll, emu_grad, emu_hess) =
            emulate_step6_joint_beta_device(&pullbacks, p).expect("device emulator (hand fixture)");
        let (ref_nll, ref_grad, ref_hess) = reference_pullback(&row_specs, r, p);
        assert!(
            close(emu_nll, ref_nll),
            "hand fixture nll: emu {emu_nll} vs textbook {ref_nll}"
        );
        for j in 0..p {
            assert!(
                close(emu_grad[j], ref_grad[j]),
                "hand fixture grad[{j}]: emu {} vs textbook {}",
                emu_grad[j],
                ref_grad[j]
            );
            for k in 0..p {
                assert!(
                    close(emu_hess[[j, k]], ref_hess[j * p + k]),
                    "hand fixture hess[{j},{k}]: emu {} vs textbook {}",
                    emu_hess[[j, k]],
                    ref_hess[j * p + k]
                );
                checked += 1;
            }
        }

        assert!(
            checked > 0,
            "step6 emulator parity-lock swept zero elements — coverage regressed"
        );
        assert!(
            worst_ratio <= 1e-2,
            "step6 parity-lock headroom collapsed: worst drift/tolerance = {worst_ratio:.3e} \
             (want ≤ 1e-2); the band is absorbing conditioning noise rather than pinning the \
             device transcription"
        );
    }

    /// #415 structural lockstep guard: the `.cu` must still spell the exact
    /// arithmetic `emulate_step6_row_pullback_device` mirrors. A dropped
    /// skip-zero branch, a renamed output, or a re-indexed contraction removes
    /// one of these load-bearing substrings and fails the build ON EVERY BOX,
    /// flagging that the CPU emulator is now stale relative to the device
    /// program. (The source const is available on all targets specifically so
    /// this runs in CPU CI, not only on the GPU runner.)
    #[test]
    fn step6_device_emulator_source_lockstep_fingerprint_415() {
        let src = SURVIVAL_FLEX_STEP6_SOURCE;
        let required = [
            "survival_flex_step6_rows",
            // grad_row[k] = Σ_a J[a,k]·g_p[a], skip zero g.
            "if (ga != 0.0) acc += j[a * p + k] * ga;",
            "grad_row[k] = acc;",
            // M[a,k] = Σ_b H_p[a,b]·J[b,k], skip zero H.
            "double hab = h_p[a * r + b];",
            "if (hab != 0.0) acc += hab * j[b * p + k];",
            "m_row[a * p + k] = acc;",
            // hess_row[col,k] = Σ_a J[a,col]·M[a,k], skip zero J.
            "double jac = j[a * p + col];",
            "if (jac != 0.0) acc += jac * m_row[a * p + k];",
            "hess_row[col * p + k] = acc;",
        ];
        for tok in required {
            assert!(
                src.contains(tok),
                "SURVIVAL_FLEX_STEP6_SOURCE no longer contains `{tok}` — the CPU \
                 device-arithmetic emulator `emulate_step6_row_pullback_device` is now stale; \
                 re-derive it in lockstep with the .cu edit so the #415 CPU↔device parity-lock \
                 keeps testing the LIVE device program."
            );
        }
    }

    /// GPU-vs-CPU parity for the on-device Step-6 joint-β contraction.
    ///
    /// On a CUDA host this launches `survival_flex_step6_rows` and asserts the
    /// device `(nll, grad, H)` matches the host reference
    /// `pullback_step6_joint_beta` bit-tight.  Off CUDA the device entry returns
    /// `Ok(None)` and we assert the host fallback is finite (no false green on
    /// CPU — we never pretend a device ran).
    #[test]
    fn step6_device_contraction_matches_cpu_reference() {
        let n_rows = 37usize;
        let p = 6usize;
        let (outs, jacs) = varied_step6_rows(n_rows, p);
        let rows: Vec<SurvivalFlexStep6RowPullback<'_>> = outs
            .iter()
            .zip(jacs.iter())
            .map(|(o, j)| SurvivalFlexStep6RowPullback {
                primary: o,
                jacobian: j.as_slice(),
            })
            .collect();

        let (cpu_nll, cpu_grad, cpu_hess) =
            pullback_step6_joint_beta(&rows, p).expect("cpu reference");

        match try_device_step6_joint_beta(&rows, p).expect("device step6") {
            Some((gpu_nll, gpu_grad, gpu_hess)) => {
                // Near-tight: the per-row contraction uses the same blocked
                // M=H_pJ order and rows are summed in order, so the only slack is
                // FP non-associativity inside each output's per-element reduction.
                // With #1686's --fmad=false active the FMA component is gone;
                // the residual is pure reassociation (measured worst abs 2.8e-14
                // on a V100), well inside the band below.
                let tol = 1e-12;
                assert!(
                    (gpu_nll - cpu_nll).abs() <= tol * (1.0 + cpu_nll.abs()),
                    "nll: gpu {gpu_nll} vs cpu {cpu_nll}"
                );
                for j in 0..p {
                    assert!(
                        (gpu_grad[j] - cpu_grad[j]).abs() <= tol * (1.0 + cpu_grad[j].abs()),
                        "grad[{j}]: gpu {} vs cpu {}",
                        gpu_grad[j],
                        cpu_grad[j]
                    );
                }
                for a in 0..p {
                    for b in 0..p {
                        assert!(
                            (gpu_hess[[a, b]] - cpu_hess[[a, b]]).abs()
                                <= tol * (1.0 + cpu_hess[[a, b]].abs()),
                            "hess[{a},{b}]: gpu {} vs cpu {}",
                            gpu_hess[[a, b]],
                            cpu_hess[[a, b]]
                        );
                        assert_eq!(
                            gpu_hess[[a, b]],
                            gpu_hess[[b, a]],
                            "device H not symmetric at ({a},{b})"
                        );
                    }
                }
            }
            None => {
                assert!(cpu_nll.is_finite());
                assert!(cpu_grad.iter().all(|v| v.is_finite()));
                assert!(cpu_hess.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn step6_rejects_jacobian_shape_mismatch() {
        let out = SurvivalFlexStep5RowOutputs {
            row_nll: 0.0,
            grad: vec![1.0, 2.0],
            hess: vec![1.0, 0.0, 0.0, 1.0],
        };
        // r = 2, p = 3 expects jacobian.len() == 6; supply 5.
        let bad_jac = vec![0.0; 5];
        let pb = [SurvivalFlexStep6RowPullback {
            primary: &out,
            jacobian: &bad_jac,
        }];
        let err = pullback_step6_joint_beta(&pb, 3).expect_err("shape mismatch must error");
        match err {
            GpuError::DriverCallFailed { reason } => {
                assert!(reason.contains("jacobian.len()"), "got {reason}");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
