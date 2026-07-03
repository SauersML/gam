//! Measured device-resident throughput of the SAE/LLM batched-solve COMPONENT —
//! the resident penalized normal-equations inner solve, NOT the full exact SAE
//! encode (see the SCOPE section below) (#1412, #988, #1017 Phase-3).
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
//! ## SCOPE — this is a COMPONENT benchmark, not the full exact SAE encode
//!
//! What is timed here is the resident penalized normal-equations *inner solve*
//! `(XᵀWX + ridge·I)β = rhs` ONLY. That is one component of the SAE encode, NOT
//! the full exact per-row SAE encode, and the measured rate is therefore NOT
//! evidence for a "batched exact per-row GPU encode" title claim. The full exact
//! encode would additionally require, per row: active-set routing (which atoms
//! are live), the per-row latent-coordinate Newton refinement on the manifold,
//! the assignment/gate (softmax/IBP) solve, and the certificate/fallback +
//! reconstruction-validation path. None of those are exercised or timed by this
//! function. Establishing the end-to-end encode-throughput claim requires a
//! separate benchmark that times the *production encode path itself* (routing +
//! latent-coordinate Newton + assignment/gate solve + fallback/certificate), not
//! this inner-solve cell. Treat the number below strictly as the resident
//! normal-equations inner-solve throughput.
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
use super::policy::{GPU_THROUGHPUT_TARGET_ROWS_PER_SEC, GpuThroughputVerdict};

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
pub fn measure_resident_solve_throughput(
    shape: EncodeShape,
    reps: usize,
) -> ResidentSolveThroughput {
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
    if handle
        .solve_normal_equations(w.view(), rhs.view(), ridge)
        .is_none()
    {
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

    // Gram = Xᵀ diag(w) X + ridge·I, formed in f64 as (√w⊙X)ᵀ(√w⊙X) via the
    // BLAS-backed `dot` (the scalar triple loop is O(n·p²) and dominates the
    // oracle at p in the thousands). Folding √w into both factors keeps the
    // weighting exact: row i contributes wᵢ·xᵢₐ·xᵢᵦ as (√wᵢxᵢₐ)(√wᵢxᵢᵦ).
    let mut xw = x.to_owned();
    for i in 0..n {
        let sw = w[i].sqrt();
        for a in 0..p {
            xw[[i, a]] *= sw;
        }
    }
    let mut gram = xw.t().dot(&xw);
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
        // The oracle exists to be the truth the device is checked against, so a
        // non-PD pivot must fail loudly here rather than clamp to 0 and launder
        // a divide-by-zero into a silent NaN in the back-substitution. For the
        // ridge·I + XᵀWX systems this is called on (ridge > 0, w > 0) the pivot
        // is always strictly positive; a non-positive pivot means the caller
        // passed a degenerate system and parity would be meaningless.
        assert!(
            diag > 0.0,
            "cpu_oracle: non-positive Cholesky pivot {diag:.3e} at index {j} — \
             the Gram is not positive-definite (need ridge>0 and w>0)"
        );
        let ljj = diag.sqrt();
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

// ===========================================================================
// FULL exact per-row encode throughput + correctness (#1412 follow-up).
//
// The component benchmark above times ONLY the resident normal-equations inner
// solve `(XᵀWX+ridge·I)β=rhs` and is explicit (see the SCOPE section) that this
// is NOT the full exact per-row SAE encode. The pieces below are the reusable,
// gam-sae-free instrument for benchmarking the *full* production encode path
// end-to-end — active-set/chart routing + per-row latent-coordinate Newton +
// gate/assignment (amplitude) + Kantorovich certificate/fallback +
// reconstruction. They live here (CPU-linkable, no `gam-sae` dependency: this
// crate is *below* `gam-sae`) so the timing harness and the correctness gate
// are shared, while the driver that actually calls the production
// `EncodeAtlas::certified_encode_batch` lives in
// `crates/gam-gpu/tests/encode_full_path_throughput.rs` (a dev-dependency cycle
// onto `gam-sae`, allowed by cargo for test-only edges).
//
// HONEST DEVICE STATUS. This helper is still backend-agnostic instrumentation:
// callers must set `device_encode_engaged` to `true` only when their encode was
// produced by a real device-resident exact-encode kernel. The current SAE device
// driver that can make that assertion lives in
// `gam_sae::gpu_kernels::sae_encode_resident::measure_device_encode_throughput`;
// older host-only full-path harnesses pass `false`. This benchmark therefore
// never fabricates a device "batched exact per-row GPU encode" number from a
// host encode — it reports the full-path timing and a correctness contract
// (support agreement, coordinate error, reconstruction explained-variance, and
// fallback rate), while the caller-owned engagement flag decides whether the
// #988 deployment/surrogate gate may consume the rate as a device measurement.
// ===========================================================================

/// End-to-end throughput of the FULL exact per-row encode for one batch.
///
/// Distinct from [`ResidentSolveThroughput`] (which times only the inner solve):
/// `rows_per_sec` here is `n_rows / encode_secs` for the *entire* production
/// `certified_encode_batch` — routing, per-row Newton, certificate, fallback,
/// and the per-row reconstruction selection included.
#[derive(Clone, Copy, Debug)]
pub struct FullEncodeThroughput {
    /// Rows encoded in the timed batch.
    pub n_rows: usize,
    /// Wall-clock seconds for the full encode of the batch.
    pub encode_secs: f64,
    /// `n_rows / encode_secs` (`0.0` for a degenerate / non-positive time).
    pub rows_per_sec: f64,
    /// `true` ONLY if a device-resident exact-encode kernel actually ran the
    /// encode. No such kernel exists yet, so this is `false` even on a GPU host
    /// — the flag is the false-routing guard that keeps the CPU encode rate from
    /// ever being reported as a device measurement.
    pub device_encode_engaged: bool,
}

impl FullEncodeThroughput {
    /// Build a throughput record from a measured elapsed time. `engaged` is the
    /// caller's honest assertion that a device-resident encode kernel produced
    /// the result; pass `false` for the host encode path.
    #[must_use]
    pub fn from_elapsed(n_rows: usize, elapsed: Duration, device_encode_engaged: bool) -> Self {
        let encode_secs = elapsed.as_secs_f64();
        let rows_per_sec = if n_rows > 0 && encode_secs > 0.0 {
            n_rows as f64 / encode_secs
        } else {
            0.0
        };
        Self {
            n_rows,
            encode_secs,
            rows_per_sec,
            device_encode_engaged,
        }
    }
}

/// Correctness of an encode result, measured against the production CPU encode
/// (a per-row reference) and the reconstruction it implies.
///
/// Every field is a quantity a "batched exact per-row encode" claim has to
/// stand on: it must AGREE with the production per-row encode (support +
/// coordinates), it must RECONSTRUCT the targets (explained variance), and it
/// must be honest about how many rows it could not certify (fallback rate).
#[derive(Clone, Copy, Debug)]
pub struct EncodeQualityMetrics {
    /// Rows compared.
    pub n_rows: usize,
    /// Rows the encode-under-test certified (`h ≤ ½`, exact-into-the-ball).
    pub certified_rows: usize,
    /// Fraction of rows the encode-under-test could NOT certify and flagged for
    /// the multi-start fallback (`1 - certified_rows/n_rows`). This is the
    /// "fallback rate".
    pub fallback_rate: f64,
    /// Fraction of rows whose certificate flag AGREES with the per-row reference
    /// encode. For a correct batched encode this is `1.0` (the batch is just the
    /// per-row encode fanned out).
    pub support_agreement: f64,
    /// Largest absolute latent-coordinate difference between the encode-under-test
    /// and the per-row reference encode, over all rows and coordinate dims. A
    /// correct batched encode matches the per-row encode to round-off (≈ `0`).
    pub max_coord_abs_err: f64,
    /// Largest absolute element-wise reconstruction residual `|x̂ − x|` over the
    /// whole batch (the "amplitude"/reconstruction error in raw output units).
    pub max_reconstruction_abs_err: f64,
    /// Reconstruction explained variance `1 − ‖X − X̂‖²_F / ‖X − X̄‖²_F`, with each
    /// output column centered by its own mean `X̄`. `1.0` is a perfect on-manifold
    /// reconstruction; `0.0` is no better than the per-column mean.
    pub reconstruction_ev: f64,
}

/// Compute [`EncodeQualityMetrics`] for an encode result.
///
/// * `coords` / `certified` — the encode UNDER TEST (`n×d` coords, `n` flags).
/// * `coords_ref` / `certified_ref` — the production per-row reference encode
///   (the definition of truth the batched/accelerated encode must match).
/// * `reconstruction` — the decoded reconstruction `x̂` implied by `coords`
///   (`n×p`, i.e. `amplitudeᵢ · Φ(coordsᵢ) · B`).
/// * `targets` — the encode inputs `x` (`n×p`).
///
/// Panics on a shape mismatch: this is a benchmark/correctness helper and a
/// mismatched comparison would silently launder a wrong number.
#[must_use]
pub fn encode_quality_metrics(
    coords: ArrayView2<'_, f64>,
    certified: &[bool],
    coords_ref: ArrayView2<'_, f64>,
    certified_ref: &[bool],
    reconstruction: ArrayView2<'_, f64>,
    targets: ArrayView2<'_, f64>,
) -> EncodeQualityMetrics {
    let (n, d) = coords.dim();
    assert_eq!(
        coords_ref.dim(),
        (n, d),
        "encode_quality_metrics: reference coords shape {:?} != under-test {:?}",
        coords_ref.dim(),
        (n, d)
    );
    assert_eq!(
        certified.len(),
        n,
        "certified flags must have one entry per row"
    );
    assert_eq!(
        certified_ref.len(),
        n,
        "reference certified flags must have one entry per row"
    );
    let (nt, p) = targets.dim();
    assert_eq!(nt, n, "targets must have one row per encoded row");
    assert_eq!(
        reconstruction.dim(),
        (n, p),
        "reconstruction shape {:?} != targets {:?}",
        reconstruction.dim(),
        (n, p)
    );

    let certified_rows = certified.iter().filter(|c| **c).count();
    let fallback_rate = if n > 0 {
        1.0 - certified_rows as f64 / n as f64
    } else {
        0.0
    };

    let agree = certified
        .iter()
        .zip(certified_ref.iter())
        .filter(|(a, b)| a == b)
        .count();
    let support_agreement = if n > 0 { agree as f64 / n as f64 } else { 1.0 };

    let mut max_coord_abs_err = 0.0_f64;
    for i in 0..n {
        for j in 0..d {
            max_coord_abs_err = max_coord_abs_err.max((coords[[i, j]] - coords_ref[[i, j]]).abs());
        }
    }

    // Reconstruction error + explained variance (per-column centering).
    let mut max_reconstruction_abs_err = 0.0_f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for c in 0..p {
        let mut mean = 0.0_f64;
        for i in 0..n {
            mean += targets[[i, c]];
        }
        if n > 0 {
            mean /= n as f64;
        }
        for i in 0..n {
            let resid = reconstruction[[i, c]] - targets[[i, c]];
            max_reconstruction_abs_err = max_reconstruction_abs_err.max(resid.abs());
            ss_res += resid * resid;
            let centered = targets[[i, c]] - mean;
            ss_tot += centered * centered;
        }
    }
    let reconstruction_ev = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        // Degenerate (all targets equal their column mean): a perfect
        // reconstruction is EV 1, anything else is 0 rather than a NaN.
        if ss_res == 0.0 { 1.0 } else { 0.0 }
    };

    EncodeQualityMetrics {
        n_rows: n,
        certified_rows,
        fallback_rate,
        support_agreement,
        max_coord_abs_err,
        max_reconstruction_abs_err,
        reconstruction_ev,
    }
}

#[cfg(test)]
mod full_encode_metric_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn throughput_is_rows_over_seconds_and_guards_degenerate_time() {
        let t = FullEncodeThroughput::from_elapsed(8_000, Duration::from_millis(100), false);
        assert_eq!(t.n_rows, 8_000);
        assert!(!t.device_encode_engaged);
        // 8000 rows / 0.1 s = 80_000 rows/sec.
        assert!(
            (t.rows_per_sec - 80_000.0).abs() < 1.0,
            "got {}",
            t.rows_per_sec
        );
        // Zero elapsed is a non-measurement, not an infinite rate.
        let z = FullEncodeThroughput::from_elapsed(8_000, Duration::ZERO, false);
        assert_eq!(z.rows_per_sec, 0.0);
    }

    #[test]
    fn perfect_match_scores_full_agreement_and_unit_ev() {
        // Two rows, 1 latent dim, 2 output dims. Reconstruction == targets.
        let coords = array![[0.10], [0.40]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        let m = encode_quality_metrics(
            coords.view(),
            &[true, true],
            coords.view(),
            &[true, true],
            targets.view(),
            targets.view(),
        );
        assert_eq!(m.n_rows, 2);
        assert_eq!(m.certified_rows, 2);
        assert_eq!(m.fallback_rate, 0.0);
        assert_eq!(m.support_agreement, 1.0);
        assert_eq!(m.max_coord_abs_err, 0.0);
        assert_eq!(m.max_reconstruction_abs_err, 0.0);
        assert!((m.reconstruction_ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn divergence_is_surfaced_in_every_axis() {
        let coords = array![[0.10], [0.40]];
        let coords_ref = array![[0.10], [0.50]]; // row 1 differs by 0.10
        let targets = array![[1.0, 0.0], [0.0, 1.0]];
        // Reconstruction misses target by 0.25 on one element.
        let recon = array![[1.0, 0.0], [0.0, 0.75]];
        let m = encode_quality_metrics(
            coords.view(),
            &[true, false], // row 1 uncertified under test
            coords_ref.view(),
            &[true, true], // reference certified both
            recon.view(),
            targets.view(),
        );
        assert_eq!(m.certified_rows, 1);
        assert!((m.fallback_rate - 0.5).abs() < 1e-12);
        assert!((m.support_agreement - 0.5).abs() < 1e-12); // row 1 flags disagree
        assert!((m.max_coord_abs_err - 0.10).abs() < 1e-12);
        assert!((m.max_reconstruction_abs_err - 0.25).abs() < 1e-12);
        assert!(m.reconstruction_ev < 1.0);
    }
}
