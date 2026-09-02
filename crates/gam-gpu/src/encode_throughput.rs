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

use super::policy::GpuThroughputVerdict;

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
    /// [`super::policy::GPU_THROUGHPUT_TARGET_ROWS_PER_SEC`].
    pub verdict: GpuThroughputVerdict,
}

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
