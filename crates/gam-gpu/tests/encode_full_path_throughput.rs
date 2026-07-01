//! FULL exact per-row SAE encode — end-to-end throughput AND correctness.
//!
//! ## Why this test exists
//!
//! The component benchmark in `gam_gpu::encode_throughput` (and its root test
//! `tests/gpu_encode_throughput_measured_1412.rs`) times ONLY the resident
//! penalized normal-equations inner solve `(XᵀWX+ridge·I)β=rhs` and is explicit
//! that this is NOT the full exact per-row SAE encode. Passing it therefore says
//! nothing about a "batched exact per-row encode" claim, because none of the
//! encode's real semantics are exercised: chart/active-set routing, the per-row
//! latent-coordinate Newton refinement, the gate/assignment (amplitude), the
//! Kantorovich certificate + fallback, and the per-row reconstruction selection.
//!
//! This test drives the ACTUAL production encode — `EncodeAtlas::certified_*`
//! (`crates/gam-sae/src/encode.rs`), which is exactly that pipeline — end to end
//! over a batch, and:
//!
//!   1. TIMES the full `certified_encode_batch` → rows/sec
//!      ([`FullEncodeThroughput`]);
//!   2. CHECKS correctness against the production per-row encode and the planted
//!      manifold via [`encode_quality_metrics`]: support agreement, latent
//!      coordinate error, reconstruction explained-variance, and fallback rate.
//!
//! ## Reuse, not reimplementation
//!
//! The encode math is NOT reimplemented here: the test calls the production
//! `EncodeAtlas::certified_encode_batch` / `certified_encode_row`. `gam-sae`
//! normally depends on `gam-gpu`; the dev-only back-edge in this crate's
//! `Cargo.toml` (allowed by cargo because the cycle has a dev edge) is what lets
//! this `gam-gpu` integration test call into `gam-sae`.
//!
//! ## Device status (honest)
//!
//! There is currently NO device-resident exact-encode kernel — the production
//! `certified_encode_*` path is host ndarray work (the only SAE GPU kernel,
//! `gam_sae::gpu_kernels::sae_rowjet`, accelerates the *fitting* jet tower, not
//! the encode). So `device_encode_engaged` is reported `false` and the measured
//! rate is the CPU encode throughput. This test does NOT fabricate a device
//! number; it establishes the end-to-end CPU baseline + the correctness contract
//! a future device encode must match, and it exercises `gam-gpu`'s runtime probe
//! + fail-closed (`GpuMode::Required`) guard so the GPU plumbing stays wired.

use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};

use gam_gpu::device_runtime::GpuRuntime;
use gam_gpu::encode_throughput::{encode_quality_metrics, FullEncodeThroughput};
use gam_gpu::{GpuError, GpuMode};

use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::encode::{AtlasConfig, EncodeAtlas};
use gam_sae::manifold::{SaeAtomBasisKind, SaeManifoldAtom};

/// Two orthonormal output vectors in `R^p`, deterministic, so the planted circle
/// `t ↦ z·(cos2πt·u + sin2πt·v)` is an INJECTIVE, well-conditioned immersion into
/// a wide (`p`-dim) decoder output — a realistic SAE residual block, not a toy
/// 2-D circle. Orthonormality makes `‖x(t)‖ = z` exactly.
fn orthonormal_pair(p: usize) -> (Array1<f64>, Array1<f64>) {
    let mut u = Array1::from_shape_fn(p, |j| (0.3 * j as f64 + 0.1).cos());
    let mut v = Array1::from_shape_fn(p, |j| (0.2 * j as f64 + 0.7).sin());
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    // Gram–Schmidt: remove the u-component from v, then normalize.
    let proj = v.dot(&u);
    v = &v - &(&u * proj);
    let vn = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / vn);
    (u, v)
}

/// Build the production fixture: one periodic-harmonic atom whose decoder embeds
/// a unit circle into `R^p`, plus `n` on-manifold target rows at known latent
/// coordinates `t_i` and per-row amplitudes `z_i`, and the certified
/// [`EncodeAtlas`] over the frozen dictionary.
///
/// Returns `(atom, atlas, targets, amplitudes, planted_t)`.
fn build_fixture(
    n: usize,
    p: usize,
) -> (
    SaeManifoldAtom,
    EncodeAtlas,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());

    // Seed the atom's stored basis on a small grid (the encode re-evaluates via
    // the installed evaluator at chart centers; this is just the atom's initial
    // dictionary state).
    let n_seed = 64usize;
    let seed: Array2<f64> =
        Array2::from_shape_fn((n_seed, 1), |(i, _)| i as f64 / n_seed as f64);
    let (seed_phi, seed_jet) = evaluator.evaluate(seed.view()).unwrap();
    let m = seed_phi.ncols(); // 5: [1, sin2πt, cos2πt, sin4πt, cos4πt]

    // Decoder (m × p): cos2πt (row 2) → u, sin2πt (row 1) → v.
    let (u, v) = orthonormal_pair(p);
    let mut decoder = Array2::<f64>::zeros((m, p));
    for c in 0..p {
        decoder[[2, c]] = u[c];
        decoder[[1, c]] = v[c];
    }

    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        seed_phi,
        seed_jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone());

    // Planted on-manifold data: t_i interior, amplitudes in [0.8, 1.2].
    let planted_t: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64 + 0.5) / n as f64);
    let amplitudes: Array1<f64> =
        Array1::from_shape_fn(n, |i| 0.8 + 0.4 * ((i as f64 * 0.123).sin() * 0.5 + 0.5));
    let coords_truth = Array2::from_shape_fn((n, 1), |(i, _)| planted_t[i]);
    let (phi_truth, _) = evaluator.evaluate(coords_truth.view()).unwrap();
    let decoded = phi_truth.dot(&decoder); // (n × p), amplitude-1
    let mut targets = decoded;
    for i in 0..n {
        let z = amplitudes[i];
        for c in 0..p {
            targets[[i, c]] *= z;
        }
    }

    let amplitude_bound = amplitudes.iter().cloned().fold(0.0_f64, f64::max);
    let mut target_norm_bound = 0.0_f64;
    for i in 0..n {
        target_norm_bound = target_norm_bound.max(targets.row(i).dot(&targets.row(i)).sqrt());
    }

    let atlas = EncodeAtlas::build(
        std::slice::from_ref(&atom),
        &[amplitude_bound],
        target_norm_bound,
        AtlasConfig {
            grid_resolution: 64,
            ridge: 1e-10,
            newton_steps: 8,
        },
    )
    .expect("encode atlas builds over the frozen dictionary");

    (atom, atlas, targets, amplitudes, planted_t)
}

/// Decode a batch of recovered latent coordinates back through the SAME basis +
/// decoder, scaled per row by the encode amplitude: `x̂ᵢ = zᵢ · Φ(tᵢ) · B`.
fn reconstruct(
    atom: &SaeManifoldAtom,
    coords: &Array2<f64>,
    amplitudes: &Array1<f64>,
) -> Array2<f64> {
    let evaluator = atom.basis_evaluator.as_ref().expect("atom has evaluator");
    let (phi, _) = evaluator.evaluate(coords.view()).unwrap();
    let mut recon = phi.dot(&atom.decoder_coefficients); // (n × p)
    for i in 0..coords.nrows() {
        let z = amplitudes[i];
        for c in 0..recon.ncols() {
            recon[[i, c]] *= z;
        }
    }
    recon
}

#[test]
fn full_exact_encode_throughput_and_correctness() {
    let n = 4_096usize; // > ENCODE_BATCH_PARALLEL_ROW_MIN: exercises the real batch fan-out.
    let p = 64usize;
    let (atom, atlas, targets, amplitudes, _planted_t) = build_fixture(n, p);

    // --- Production reference: per-row exact encode (the definition of truth a
    // batched/accelerated encode must reproduce). This IS the production CPU
    // encode (`certified_encode_row`), looped in row order. ---
    let mut coords_ref = Array2::<f64>::zeros((n, atom.latent_dim));
    let mut certified_ref = vec![false; n];
    for i in 0..n {
        let (t, cert) = atlas
            .certified_encode_row(&atom, 0, targets.row(i), amplitudes[i])
            .expect("per-row reference encode runs");
        coords_ref.row_mut(i).assign(&t);
        certified_ref[i] = cert.certified();
    }

    // --- The FULL exact per-row encode under test, timed end-to-end. One warm
    // run (allocations / first-touch) then one timed run. ---
    atlas
        .certified_encode_batch(&atom, 0, targets.view(), amplitudes.view())
        .expect("warm batch encode runs");
    let start = Instant::now();
    let result = atlas
        .certified_encode_batch(&atom, 0, targets.view(), amplitudes.view())
        .expect("timed batch encode runs");
    let elapsed = start.elapsed();

    // No device-resident exact-encode kernel exists yet, so this is honestly a
    // CPU measurement (engaged = false). It must still be a real measurement.
    let throughput = FullEncodeThroughput::from_elapsed(n, elapsed, false);
    assert!(!throughput.device_encode_engaged);
    assert!(
        throughput.rows_per_sec > 0.0,
        "the full encode must produce a positive rows/sec, got {}",
        throughput.rows_per_sec
    );

    // --- Correctness: reconstruct, then score against the per-row reference. ---
    let reconstruction = reconstruct(&atom, &result.coords, &amplitudes);
    let metrics = encode_quality_metrics(
        result.coords.view(),
        &result.certified,
        coords_ref.view(),
        &certified_ref,
        reconstruction.view(),
        targets.view(),
    );

    eprintln!(
        "[full-encode] n={n} p={p} rows/sec={:.1} (device_engaged={}) | \
         certified={}/{} fallback_rate={:.3} support_agreement={:.6} \
         max_coord_err={:.3e} reconstruction_ev={:.6} max_recon_err={:.3e}",
        throughput.rows_per_sec,
        throughput.device_encode_engaged,
        metrics.certified_rows,
        n,
        metrics.fallback_rate,
        metrics.support_agreement,
        metrics.max_coord_abs_err,
        metrics.reconstruction_ev,
        metrics.max_reconstruction_abs_err,
    );

    // (1) The batched encode must REPRODUCE the production per-row encode exactly
    // — same certificate decisions and the same latent coordinates (the batch is
    // the per-row encode fanned out; the production code documents bit-identity).
    assert_eq!(
        metrics.support_agreement, 1.0,
        "batched encode certificate flags must match the per-row reference on every row"
    );
    assert!(
        metrics.max_coord_abs_err < 1e-12,
        "batched encode coordinates must match the per-row reference to round-off; \
         max |Δcoord| = {:.3e}",
        metrics.max_coord_abs_err
    );

    // (2) The encode must RECOVER the planted manifold: on-manifold targets must
    // reconstruct to near-perfect explained variance.
    assert!(
        metrics.reconstruction_ev > 0.99,
        "exact encode must reconstruct on-manifold targets (EV > 0.99); got {:.6}",
        metrics.reconstruction_ev
    );
    assert!(
        metrics.max_reconstruction_abs_err < 1e-2,
        "worst per-element reconstruction residual too large: {:.3e}",
        metrics.max_reconstruction_abs_err
    );

    // (3) The certificate must cover real coverage of a well-conditioned circle
    // dictionary — not a vacuous all-uncertified (all-fallback) result.
    assert!(
        metrics.fallback_rate < 0.5,
        "the certified encode must certify a majority of a well-conditioned circle \
         dictionary; fallback_rate = {:.3}",
        metrics.fallback_rate
    );

    // (4) GPU plumbing stays wired and HONEST. The production encode does not
    // route to a device kernel, so on any host today `device_encode_engaged`
    // is false; separately, the fail-closed contract must hold — on a CPU-only
    // host `GpuMode::Required` surfaces a structured error (never a silent CPU
    // pass dressed as a device run), and on a real device it succeeds.
    let required = GpuRuntime::global_or_fail(GpuMode::Required);
    if GpuRuntime::is_available() {
        assert!(
            required.is_ok(),
            "GpuMode::Required must succeed when a device is present"
        );
    } else {
        assert!(
            matches!(required, Err(GpuError::DriverLibraryUnavailable { .. })),
            "GpuMode::Required must fail closed when the device is absent, got {required:?}"
        );
    }
    // GpuMode::Off always refuses, regardless of hardware.
    assert!(GpuRuntime::global_or_fail(GpuMode::Off).is_err());
}
