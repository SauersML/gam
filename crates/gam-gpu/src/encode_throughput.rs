//! Measured device-resident encode throughput for the SAE/LLM batched-solve
//! shape (#1412, #988, #1017 Phase-3).
//!
//! ## Why this module exists
//!
//! The historical throughput "decision gate" (#1412) asserted a `100_000`
//! rows/sec/GPU deployment target **without ever measuring a device**. Its
//! successor still keyed the deployment decision on a *CPU* measurement scaled
//! by a hardcoded `CPU_TO_GPU_SCALING = 100.0` fudge factor — so passing the
//! gate established nothing about real GPU throughput. #988 closed
//! `COMPLETED` while the maintainer's own follow-up confirmed the GPU
//! steady-state encode rate had never been measured.
//!
//! This module makes the measurement real and *testable as a library function*
//! (the prior real benchmark lived only in `examples/throughput_1412.rs`, which
//! nothing in CI ran or asserted). [`measure_resident_solve_throughput`] runs
//! the production IRLS inner step — upload `X` once, then repeatedly solve the
//! penalized normal equations `(XᵀWX + ridge·I)β = rhs` with the `p×p` Gram and
//! its Cholesky factor kept DEVICE-RESIDENT, downloading only the `p`-vector
//! `β` — on the real device, and reports the measured design-rows/sec.
//!
//! ## Fail-loud, never false-route
//!
//! The single recurring failure mode this guards against is *false GPU
//! routing*: claiming a device measurement while the work silently ran on the
//! CPU. [`ResidentSolveThroughput::engaged`] is `true` only when
//! [`ResidentDesignGram::try_new`] actually staged `X` on the device AND every
//! timed solve returned a device result. If the device path declines or fails
//! mid-measurement, `engaged` is `false` and `measured_rows_per_sec` is left at
//! `0.0` — a non-measurement that [`GpuThroughputVerdict`] can never report as
//! meeting the target. There is no CPU fallback inside the measurement: a
//! caller that wants the CPU oracle runs it separately for parity.

use std::hint::black_box;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::linalg_dispatch::ResidentDesignGram;
use super::policy::{GpuThroughputVerdict, GPU_THROUGHPUT_TARGET_ROWS_PER_SEC};

/// A representative LLM/SAE batched-solve work cell: `n` design rows, `p` wide
/// decoder border. (`d`, the per-atom reduced-Schur block size, is fixed by the
/// term and does not enter the resident-solve throughput.)
#[derive(Clone, Copy, Debug)]
pub struct EncodeShape {
    /// Human-readable label for reporting.
    pub label: &'static str,
    /// Design rows pushed through the device per fit.
    pub n: usize,
    /// Decoder-border width (the resident Gram is `p×p`).
    pub p: usize,
}

/// The canonical qwen/olmo-scale SAE residual-block shapes (matches the
/// `examples/throughput_1412.rs` workload so the library measurement and the
/// example agree).
pub const CANONICAL_ENCODE_SHAPES: &[EncodeShape] = &[
    EncodeShape {
        label: "sae-2k-2048",
        n: 2_000,
        p: 2_048,
    },
    EncodeShape {
        label: "sae-4k-4096",
        n: 4_000,
        p: 4_096,
    },
    EncodeShape {
        label: "sae-8k-1024",
        n: 8_000,
        p: 1_024,
    },
];

/// Outcome of measuring the device-resident penalized-solve throughput for one
/// [`EncodeShape`].
#[derive(Clone, Copy, Debug)]
pub struct ResidentSolveThroughput {
    /// The shape that was measured.
    pub shape: EncodeShape,
    /// `true` iff `X` was staged on the device AND every timed solve returned a
    /// device result. `false` means the device path declined or failed — the
    /// number below is **not** a device measurement.
    pub engaged: bool,
    /// Measured design-rows/sec for the resident solve, or `0.0` when the
    /// device path did not engage (a non-measurement).
    pub measured_rows_per_sec: f64,
    /// The verdict comparing `measured_rows_per_sec` against
    /// [`GPU_THROUGHPUT_TARGET_ROWS_PER_SEC`].
    pub verdict: GpuThroughputVerdict,
}

/// Deterministic LCG in `[-1, 1)` — no `rand` dependency, fully reproducible
/// across runs so the measured fixture is stable.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Build a deterministic `n×p` design fixture for the throughput measurement.
fn planted_design(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((n, p), |_| lcg(&mut s) * 0.05)
}

/// Measure the device-resident penalized-normal-equations solve throughput for
/// one shape: upload `X` once, then time `reps` solves that cross only `w`
/// (H2D), `rhs` (H2D, fixed), and `β` (D2H) — the production IRLS inner step.
///
/// `reps` is the number of timed solves; `w` is perturbed per rep so each solve
/// is genuine work, mirroring an IRLS weight update. Returns a
/// [`ResidentSolveThroughput`] whose `engaged` flag is the false-routing guard:
/// on a CPU-only host (or if the device declines) it is `false` and the rate is
/// `0.0`.
#[must_use]
pub fn measure_resident_solve_throughput(shape: EncodeShape, reps: usize) -> ResidentSolveThroughput {
    let EncodeShape { n, p, .. } = shape;
    let not_engaged = |shape| ResidentSolveThroughput {
        shape,
        engaged: false,
        measured_rows_per_sec: 0.0,
        verdict: GpuThroughputVerdict::from_measurement(0.0),
    };
    if n == 0 || p == 0 || reps == 0 {
        return not_engaged(shape);
    }

    let x = planted_design(n, p, 0x1412_a100_dead_beef);
    let w = {
        let mut s = 0x988_5ae_e0c0_de01u64;
        Array1::from_shape_fn(n, |_| lcg(&mut s).abs() + 1e-3)
    };
    let rhs = Array1::from_shape_fn(p, |j| ((j as f64 + 1.0) * 0.03).cos());
    let ridge = 1e-3_f64;

    // Stage X once. `None` => no device / shape below the Gram threshold => not
    // a device measurement.
    let handle = match ResidentDesignGram::try_new(x.view()) {
        Some(h) => h,
        None => return not_engaged(shape),
    };

    // Warm the resident solve (allocations, kernel handles) outside the timer;
    // if even the warm solve declines, the device path is not usable here.
    if handle.solve_normal_equations(w.view(), rhs.view(), ridge).is_none() {
        return not_engaged(shape);
    }

    let mut total = Duration::ZERO;
    for r in 0..reps {
        let wr = Array1::from_shape_fn(n, |i| (w[i] + 1e-3 * (r as f64)).abs());
        let start = Instant::now();
        match handle.solve_normal_equations(wr.view(), rhs.view(), ridge) {
            Some(beta) => {
                black_box(beta);
            }
            // A mid-measurement decline means the timed region is no longer a
            // pure device measurement — refuse to report it as one.
            None => return not_engaged(shape),
        }
        total += start.elapsed();
    }

    let secs = total.as_secs_f64() / reps as f64;
    let measured_rows_per_sec = if secs > 0.0 { n as f64 / secs } else { 0.0 };
    ResidentSolveThroughput {
        shape,
        engaged: measured_rows_per_sec > 0.0,
        measured_rows_per_sec,
        verdict: GpuThroughputVerdict::from_measurement(measured_rows_per_sec),
    }
}

/// CPU oracle for the same penalized normal-equations solve, used for parity:
/// `(XᵀWX + ridge·I)β = rhs` solved by a host Cholesky. This is the definition
/// of truth the device solve must match (up to IEEE-754 reduction order).
#[must_use]
pub fn cpu_oracle_normal_equations_solve(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
    rhs: ArrayView1<'_, f64>,
    ridge: f64,
) -> Array1<f64> {
    let (n, p) = x.dim();
    assert_eq!(w.len(), n, "w must have one entry per design row");
    assert_eq!(rhs.len(), p, "rhs must have one entry per border column");

    // Gram = Xᵀ diag(w) X + ridge·I, formed in f64.
    let mut gram = Array2::<f64>::zeros((p, p));
    for a in 0..p {
        for b in a..p {
            let mut acc = 0.0_f64;
            for i in 0..n {
                acc += x[[i, a]] * w[i] * x[[i, b]];
            }
            gram[[a, b]] = acc;
            gram[[b, a]] = acc;
        }
    }
    for j in 0..p {
        gram[[j, j]] += ridge;
    }

    // Cholesky: gram = L Lᵀ (lower), then solve L y = rhs, Lᵀ β = y.
    let mut l = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        let mut diag = gram[[j, j]];
        for s in 0..j {
            diag -= l[[j, s]] * l[[j, s]];
        }
        let ljj = diag.max(0.0).sqrt();
        l[[j, j]] = ljj;
        for i in (j + 1)..p {
            let mut off = gram[[i, j]];
            for s in 0..j {
                off -= l[[i, s]] * l[[j, s]];
            }
            l[[i, j]] = off / ljj;
        }
    }
    let mut y = rhs.to_owned();
    for i in 0..p {
        let mut acc = y[i];
        for s in 0..i {
            acc -= l[[i, s]] * y[s];
        }
        y[i] = acc / l[[i, i]];
    }
    let mut beta = y;
    for i in (0..p).rev() {
        let mut acc = beta[i];
        for s in (i + 1)..p {
            acc -= l[[s, i]] * beta[s];
        }
        beta[i] = acc / l[[i, i]];
    }
    beta
}

/// The deployment target, re-exported so callers measuring throughput do not
/// have to import the policy module directly.
pub const DEPLOYMENT_TARGET_ROWS_PER_SEC: f64 = GPU_THROUGHPUT_TARGET_ROWS_PER_SEC;
