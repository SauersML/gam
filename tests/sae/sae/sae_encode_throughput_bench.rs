//! Issue #988 — Stage-3 encode throughput benchmark.
//!
//! DECISION GATE. The Stage-3 SAE pipeline encodes a row `z ∈ ℝ^p` into the
//! per-atom (gate, latent-coordinate) code by running the EXACT production
//! per-row Newton solve against a *frozen* decoder dictionary. That inner solve
//! is [`SaeManifoldTerm::run_fixed_decoder_arrow_schur`] — the same arrow-Schur
//! Gauss/Newton path the joint fit uses, but with the decoder held fixed so the
//! only unknowns are the per-row active set + latent coordinates. Batched over a
//! large block of rows, this is precisely the "encode throughput" we must size.
//!
//! The open design question #988 asks: is a *certified amortized surrogate*
//! (an amortized encoder that approximates the exact per-row solve and must then
//! be certified against it) EVER worth building? The answer is purely empirical:
//!   * The deployment gate is **10^5 rows/sec/GPU** for the exact batched encode.
//!   * If the exact solve already clears that gate, the amortized surrogate is
//!     **NEVER built** — there is no throughput headroom to win, and a surrogate
//!     would only add a certification liability.
//!   * If it does NOT clear the gate, the surrogate becomes justified.
//!
//! This test runs on CPU with NO GPU, so it CANNOT measure the 10^5 rows/sec/GPU
//! figure and CANNOT make the surrogate-vs-no-surrogate deployment decision. It
//! measures the exact CPU batched-encode throughput (as a correctness + perf
//! regression sentinel only — see `CPU_ENCODE_REGRESSION_FLOOR_ROWS_PER_SEC`) and
//! records the deployment decision as an honest tri-state
//! [`gam::gpu::policy::EncodeDeploymentDecision`]: on a CPU-only host it is
//! `Undetermined` (BLOCKED on hardware), NEVER "surrogate unneeded". Only a real
//! device measurement can move it to `Met`/`Unmet`.
//!
//! HISTORY (#1412, reopened twice): earlier versions projected the CPU rate
//! through an assumed `CPU_TO_GPU_SCALING = 100.0` and asserted the *projection*
//! cleared the gate — a CPU number dressed up as a GPU deployment certification.
//! That fudge is gone: there is no CPU→GPU factor and no CPU-derived surrogate
//! decision anywhere below. A CPU rate, however fast, cannot make this gate claim
//! the GPU target is met.
//!
//! The benchmark PRINTS the measured rows/sec per K verbatim; the assertion is
//! the gate, the print is the datum. When CUDA is available the gate also runs
//! the production device-resident certified encode and compares THAT measured
//! device rows/sec against the 100k rows/sec/GPU target; a CPU rate can only be
//! a regression sentinel and can never certify the deployment target.

use gam::terms::latent::LatentManifold;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator,
    sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

// ---- production inner-solve defaults (mirror `sae_manifold_fit` encode) ------
/// Ambient response dimension `p`.
const P: usize = 32;
/// Circle basis size per atom: const + 1 harmonic (sin, cos).
const M: usize = 3;
/// ordered independent Beta--Bernoulli relaxation temperature (production `TAU`).
const TAU: f64 = 0.5;
/// ordered independent Beta--Bernoulli concentration (production `ALPHA`).
const ALPHA: f64 = 1.0;
/// Sparsity / smoothness hyperparameters held frozen during the encode.
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
/// Exact per-row Newton iteration budget for the encode. This is the
/// production fixed-decoder inner-loop budget: enough to converge the active
/// set + latent coordinate of a single row to the arrow-Schur stationary point.
const ENCODE_NEWTON_ITERS: usize = 8;
/// Newton damping / line-search base step (production `LEARNING_RATE`).
const STEP_SIZE: f64 = 1.0;
/// External-coordinate ridge for the per-row block (production `RIDGE_EXT_COORD`).
const RIDGE_EXT_COORD: f64 = 1.0e-6;

/// Batch size encoded per K. Large enough that fixed assembly / setup overhead
/// is amortized and the measured figure reflects steady-state per-row cost.
const ENCODE_BATCH_ROWS: usize = 4096;

/// The two dictionary sizes the gate is evaluated at.
const K_SMALL: usize = 64;
const K_LARGE: usize = 1024;

/// The GPU deployment target, documented here purely as the number the DEVICE
/// measurement is compared against (see the real-device block at the end of the
/// gate). It is NOT projected from a CPU rate: #1412 was reopened twice precisely
/// because the deployment decision rested on a CPU measurement scaled by an
/// assumed CPU→GPU factor. There is no such factor here anymore — the deployment
/// decision ([`gam::gpu::policy::EncodeDeploymentDecision`]) can only be made from
/// a real device measurement, and on a CPU-only host it is honestly
/// `Undetermined` (BLOCKED on hardware), never green.
const GPU_DEPLOYMENT_TARGET_ROWS_PER_SEC_PER_GPU: f64 = 1.0e5;

/// **CPU regression sentinel — NOT a GPU projection.** This is a deliberately
/// conservative lower bound on the CPU exact-encode rate whose ONLY job is to
/// catch a catastrophic perf regression or a stalled/dead encode on the CPU
/// benchmark path. It is explicitly NOT multiplied by any CPU→GPU factor and it
/// makes NO claim about GPU throughput or the deployment target — that decision
/// is made solely by the device measurement below. (The removed
/// `CPU_TO_GPU_SCALING = 100.0` / `gate / scaling` floor was the #1412 defect: it
/// dressed a CPU number up as a GPU deployment certification.)
const CPU_ENCODE_REGRESSION_FLOOR_ROWS_PER_SEC: f64 = 100.0;

/// Deterministic Lehmer-style uniform in [0,1) keyed purely by index (no clock).
fn idx_uniform(seed: u64) -> f64 {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000)
}

/// Deterministic standard normal via Box-Muller on two index-seeded uniforms.
fn idx_normal(seed: u64) -> f64 {
    let u = idx_uniform(seed * 2 + 1).max(1.0e-12);
    let v = idx_uniform(seed * 2 + 2);
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// One frozen circle atom's planted ambient plane: a `p×2` orthonormal frame
/// drawn deterministically from index-seeded Gaussians (mutually "orthogonal-ish"
/// across atoms — for K planes in `p` dimensions with `K > p/2` exact mutual
/// orthogonality is impossible, so we plant independent random frames whose
/// expected cross-Gram is small). The two columns are orthonormalized *within*
/// the atom so each atom spans a genuine 2-plane (a circle).
fn planted_atom_frame(atom: usize) -> Array2<f64> {
    let mut plane = Array2::<f64>::zeros((P, 2));
    for col in 0..2 {
        for i in 0..P {
            plane[[i, col]] =
                idx_normal((atom as u64) * 1_000_003 + (col as u64) * 9_973 + i as u64);
        }
    }
    // Gram-Schmidt within the atom.
    let n0 = plane.column(0).iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..P {
        plane[[i, 0]] /= n0.max(1.0e-300);
    }
    let dot: f64 = (0..P).map(|i| plane[[i, 0]] * plane[[i, 1]]).sum();
    for i in 0..P {
        plane[[i, 1]] -= dot * plane[[i, 0]];
    }
    let n1 = plane.column(1).iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..P {
        plane[[i, 1]] /= n1.max(1.0e-300);
    }
    plane
}

/// Build the FROZEN planted dictionary's `(M × P)` decoder for atom `k` from its
/// planted plane. The periodic basis layout is row0=const, row1=sin, row2=cos,
/// so a unit-radius circle atom decodes to `g(θ) = cosθ·u_k1 + sinθ·u_k2` when
/// the cos row carries `u_k1` and the sin row carries `u_k2`; the const row is
/// zero (the planted atoms are centred).
fn planted_decoder(plane: &Array2<f64>) -> Array2<f64> {
    let mut decoder = Array2::<f64>::zeros((M, P));
    for j in 0..P {
        // row0 = const = 0, row1 = sin coeff = u_k2, row2 = cos coeff = u_k1.
        decoder[[1, j]] = plane[[j, 1]];
        decoder[[2, j]] = plane[[j, 0]];
    }
    decoder
}

/// Per-row planted truth: each row activates a small deterministic subset of the
/// K atoms (sparse codes — the regime the per-row active-set Newton is built
/// for), with a planted angle and amplitude per active atom.
struct Batch {
    z: Array2<f64>,
    theta: Vec<Vec<f64>>,   // [k][row] planted angle in [0,1)
    active: Vec<Vec<bool>>, // [k][row]
}

/// Each row activates `ACTIVE_PER_ROW` atoms chosen by an index-seeded stride,
/// so the active support is sparse and deterministic regardless of K.
const ACTIVE_PER_ROW: usize = 3;

fn planted_batch(k_atoms: usize, frames: &[Array2<f64>]) -> Batch {
    let n = ENCODE_BATCH_ROWS;
    let mut z = Array2::<f64>::zeros((n, P));
    let mut theta = vec![vec![0.0_f64; n]; k_atoms];
    let mut active = vec![vec![false; n]; k_atoms];
    for row in 0..n {
        // Choose ACTIVE_PER_ROW distinct atoms via an index-seeded offset+stride.
        let base = (idx_uniform((row as u64) * 31 + 7) * k_atoms as f64) as usize % k_atoms;
        let stride = 1 + ((idx_uniform((row as u64) * 53 + 11) * (k_atoms as f64 - 1.0)) as usize);
        for a in 0..ACTIVE_PER_ROW {
            let k = (base + a * stride) % k_atoms;
            if active[k][row] {
                continue;
            }
            active[k][row] = true;
            let ang = idx_uniform((row as u64) * 97 + (k as u64) * 13 + 3);
            theta[k][row] = ang;
            let amp = 0.85 + 0.30 * idx_uniform((row as u64) * 197 + (k as u64) * 17 + 5);
            let phase = std::f64::consts::TAU * ang;
            let c = phase.cos();
            let s = phase.sin();
            for j in 0..P {
                z[[row, j]] += amp * (c * frames[k][[j, 0]] + s * frames[k][[j, 1]]);
            }
        }
    }
    // Mild deterministic ambient noise (~3% of unit signal scale).
    for row in 0..n {
        for j in 0..P {
            z[[row, j]] += 0.03 * idx_normal(((row * P + j) as u64) * 3 + 1);
        }
    }
    Batch { z, theta, active }
}

/// Assemble the frozen-dictionary [`SaeManifoldTerm`] for the encode batch.
///
/// The decoders are the PLANTED frames (frozen — never updated during the
/// encode). Latent coordinates are seeded from the planted angles offset by a
/// small constant so the encode's per-row Newton must actually move the
/// coordinate (it is not handed the exact stationary point), and the ordered independent Beta--Bernoulli
/// routing logits are seeded from a per-row residual-energy heuristic so the
/// active-set Newton starts from a realistic warm state rather than the truth.
fn build_frozen_encode_term(batch: &Batch, frames: &[Array2<f64>]) -> SaeManifoldTerm {
    let k_atoms = frames.len();
    let n = batch.z.nrows();
    let evaluator = PeriodicHarmonicEvaluator::new(M).expect("periodic evaluator");

    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coords_k: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    let mut logits = Array2::<f64>::zeros((n, k_atoms));

    for k in 0..k_atoms {
        // Seed coords off the planted angle (small offset => Newton must work).
        let coords = Array2::from_shape_fn((n, 1), |(row, _)| {
            (batch.theta[k][row] + 0.05).rem_euclid(1.0)
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("evaluate basis");
        let decoder = planted_decoder(&frames[k]);
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("circle_{k}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(M),
        )
        .expect("frozen atom build")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(M).expect("periodic evaluator clone"),
        ));
        atoms.push(atom);

        // Residual-energy routing seed: a row whose response aligns with this
        // atom's plane gets a positive logit. Cheap projection energy heuristic
        // (the production cold seed is richer; here it only needs to place a
        // realistic warm active set for the per-row Newton to refine).
        for row in 0..n {
            // Projection energy onto the atom's planted plane: rows whose
            // response aligns with this atom get a positive routing logit.
            let mut c0 = 0.0_f64;
            let mut c1 = 0.0_f64;
            for j in 0..P {
                c0 += batch.z[[row, j]] * frames[k][[j, 0]];
                c1 += batch.z[[row, j]] * frames[k][[j, 1]];
            }
            let energy = (c0 * c0 + c1 * c1).sqrt();
            logits[[row, k]] = TAU.recip() * (energy - 0.5);
        }
        coords_k.push(coords);
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::ordered_beta_bernoulli(TAU, ALPHA, false),
    )
    .expect("assignment build");
    SaeManifoldTerm::new(atoms, assignment).expect("frozen encode term")
}

/// Run the exact batched per-row encode for a frozen K-atom dictionary and
/// return measured rows/sec.
fn measure_encode_rows_per_sec(k_atoms: usize) -> f64 {
    let frames: Vec<Array2<f64>> = (0..k_atoms).map(planted_atom_frame).collect();
    let batch = planted_batch(k_atoms, &frames);
    let mut term = build_frozen_encode_term(&batch, &frames);
    let mut rho = SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
        vec![Array1::<f64>::zeros(0); k_atoms],
    );

    // ── EXACT per-row Newton encode (frozen decoder) ───────────────────────
    //
    // This is the production inner solve: `run_fixed_decoder_arrow_schur`
    // assembles the arrow-Schur Gauss/Newton system over the WHOLE batch each
    // iteration (per-row blocks on the arrow legs, shared decoder on the spine,
    // but the decoder is FROZEN here so only the per-row active set + latent
    // coordinate move), then takes a damped/line-searched Newton step. Batched
    // == every row of the block is encoded together, exactly as a deployed
    // encode pass would process an incoming minibatch.
    //
    // GPU SEAM: a future GPU batched-encode path drops in *here* — the same
    // arrow-Schur assembly + per-row-block Cholesky factor/solve fans across
    // devices (row-block granularity, mirroring `solver::arrow_schur`'s
    // multi-GPU split), replacing this single call. The throughput contract
    // (rows/sec) and the frozen-decoder semantics are identical; only the
    // factor/solve backend changes. Nothing else in this benchmark moves.
    let start = Instant::now();
    let loss = term
        .run_fixed_decoder_arrow_schur(
            batch.z.view(),
            &mut rho,
            None,
            ENCODE_NEWTON_ITERS,
            STEP_SIZE,
            RIDGE_EXT_COORD,
        )
        .expect("exact fixed-decoder encode must complete");
    let elapsed = start.elapsed();

    let n = batch.z.nrows();
    let secs = elapsed.as_secs_f64().max(1.0e-12);
    let rows_per_sec = n as f64 / secs;

    // #1412: correctness must require actual SUPPORT RECOVERY, not merely finite
    // values — a near-zero or uniform encode produces finite assignments but
    // recovers nothing. Compare the mean assignment mass on PLANTED-ACTIVE atoms
    // against the mean on inactive atoms: a correct encode concentrates mass on
    // the planted support, so active mean must clear a positive floor AND
    // dominate the inactive mean by a clear margin.
    let assignments = term.assignment.assignments();
    let mut active_rows = 0usize;
    let mut inactive_count = 0usize;
    let mut recovered_mass = 0.0_f64;
    let mut inactive_mass = 0.0_f64;
    for row in 0..n {
        for k in 0..k_atoms {
            let a = assignments[[row, k]];
            assert!(
                a.is_finite(),
                "encode produced non-finite assignment at (row={row}, k={k})"
            );
            if batch.active[k][row] {
                active_rows += 1;
                recovered_mass += a;
            } else {
                inactive_count += 1;
                inactive_mass += a.abs();
            }
        }
    }
    let mean_active_mass = recovered_mass / active_rows.max(1) as f64;
    let mean_inactive_mass = inactive_mass / inactive_count.max(1) as f64;
    // Liveness floor: a dead/zero encode (all assignments ~0) must fail. This is
    // a conservative positive floor — NOT a calibrated reconstruction-quality
    // bound — so it cannot false-fail a correct-but-normalized encode; the
    // scale-invariant support-recovery ratio below is the real correctness gate.
    assert!(
        mean_active_mass > 1.0e-3,
        "K={k_atoms}: encode recovered negligible mass on the planted support          (mean active mass {mean_active_mass:.4e} <= 1e-3) — finite-but-empty encode"
    );
    // Support recovery (scale-invariant): planted-active atoms must carry clearly
    // more mass than inactive ones. A uniform encode has active ~ inactive
    // (ratio ~1) and a dead encode has both ~0 (ratio not > 3), so this rejects
    // exactly the "fast but wrong" passes the previous finite-only check allowed.
    assert!(
        mean_active_mass > 3.0 * mean_inactive_mass,
        "K={k_atoms}: encode did not recover the planted support          (mean active mass {mean_active_mass:.4} not > 3x mean inactive mass          {mean_inactive_mass:.4})"
    );

    println!(
        "K={k_atoms}: encoded {n} rows in {:.4}s => {rows_per_sec:.1} rows/sec  \
         (newton_iters={ENCODE_NEWTON_ITERS}, p={P}, M={M}, active/row={ACTIVE_PER_ROW}, \
         final_loss={:.6}, mean_active_mass={mean_active_mass:.4})",
        secs,
        loss.total(),
    );

    rows_per_sec
}

/// Build the smallest production certified-encode fixture that can honestly run
/// on the CUDA SAE encode kernel: a EuclideanPatch atom, its certified atlas,
/// and a batch of planted on-manifold rows large enough to clear the kernel's
/// launch threshold. This is deliberately separate from the CPU circle
/// dictionary above because the resident device encode kernel consumes the
/// Euclidean monomial basis/exponent table used by the production certified
/// atlas.
fn build_device_encode_fixture() -> (
    gam_sae::manifold::SaeManifoldAtom,
    gam_sae::encode::EncodeAtlas,
    Vec<Vec<f64>>,
    Vec<f64>,
) {
    use gam_sae::basis::{EuclideanPatchEvaluator, SaeBasisEvaluator};
    use gam_sae::encode::AtlasConfig;
    use gam_sae::gpu_kernels::sae_encode_resident::DEVICE_ROW_THRESHOLD;
    use gam_sae::manifold::{SaeAtomBasisKind, SaeManifoldAtom};

    let (d, degree, p) = (1usize, 2usize, 4usize);
    let config = AtlasConfig::default();
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(d, degree).expect("euclidean evaluator"));
    let seed = Array2::from_shape_fn((12, d), |(r, c)| {
        0.15 * ((r as f64 + 1.0) * (c as f64 + 2.0) * 0.37).sin()
    });
    let (phi, jet) = evaluator.evaluate(seed.view()).expect("seed basis eval");
    let m = phi.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        (1.0 / (1.0 + b as f64)) * (((b as f64 + 1.0) * (c as f64 + 1.0)) * 0.3).cos()
    });
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "euclid_device_encode_gate",
        SaeAtomBasisKind::EuclideanPatch,
        d,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("device encode atom")
    .with_basis_second_jet(evaluator.clone());
    let atlas = gam_sae::encode::EncodeAtlas::build(&[atom.clone()], &[2.0], 8.0, config)
        .expect("device encode atlas");

    let n = DEVICE_ROW_THRESHOLD + 64;
    let mut rows = Vec::with_capacity(n);
    let mut amplitudes = Vec::with_capacity(n);
    for row in 0..n {
        let tc = -0.4 + 0.8 * ((row % 97) as f64) / 96.0;
        let coords = Array2::from_shape_fn((1, d), |_| tc);
        let (phi_row, _) = evaluator.evaluate(coords.view()).expect("row basis eval");
        let amp = 0.9 + 0.2 * (((row as f64) * 0.013).sin() * 0.5 + 0.5);
        let mut x = vec![0.0; p];
        for c in 0..p {
            x[c] = amp
                * (0..m)
                    .map(|b| phi_row[[0, b]] * atom.decoder_coefficients[[b, c]])
                    .sum::<f64>();
        }
        rows.push(x);
        amplitudes.push(amp);
    }

    (atom, atlas, rows, amplitudes)
}

#[test]
fn sae_encode_throughput_decision_gate() {
    use gam::gpu::device_runtime::GpuRuntime;
    use gam::gpu::policy::{EncodeDecisionBlocked, EncodeDeploymentDecision};
    use gam_sae::encode::AtlasConfig;
    use gam_sae::gpu_kernels::sae_encode_resident::{
        EncodeAtomDevice, EncodePath, measure_device_encode_throughput, sae_certified_encode_batch,
    };

    println!("=== Stage-3 SAE encode throughput benchmark (#988 / #1412) ===");
    // #1412 HONESTY (post-reopen fix): this CPU benchmark does NOT measure the
    // 100k rows/sec/GPU deployment target and makes NO surrogate decision from a
    // CPU number. It asserts only what CPU can honestly prove — the exact encode
    // completes, recovers the planted support, and does not catastrophically
    // regress (a conservative CPU regression sentinel). The deployment /
    // surrogate DECISION is a tri-state `EncodeDeploymentDecision` that can reach
    // `Met`/`Unmet` ONLY from a real device measurement; on this host it is
    // `Undetermined` (BLOCKED on hardware) and the gate asserts it never
    // green-washes to "surrogate unneeded".
    //
    // The CPU rate is OPTIMISTIC by construction (multi-core Rayon assembly,
    // warm-started coordinates, setup outside the timer) — one more reason it is
    // unfit as a GPU projection and is used only as a regression sentinel.
    println!(
        "GPU deployment target = {GPU_DEPLOYMENT_TARGET_ROWS_PER_SEC_PER_GPU:.0} rows/sec/GPU \
         (NOT measured on CPU; NO CPU→GPU projection; the deployment decision is a device-only \
         tri-state — Undetermined here)."
    );

    let rps_small = measure_encode_rows_per_sec(K_SMALL);
    let rps_large = measure_encode_rows_per_sec(K_LARGE);

    // CPU regression sentinel ONLY — explicitly not a GPU projection and not a
    // deployment claim. A dead/stalled encode (near-zero rows/sec) fails here;
    // a healthy encode passes without asserting anything about the GPU target.
    let floor = CPU_ENCODE_REGRESSION_FLOOR_ROWS_PER_SEC;
    println!(
        "CPU-REGRESSION-SENTINEL: K={K_SMALL} {rps_small:.1} rows/sec, K={K_LARGE} {rps_large:.1} \
         rows/sec (sentinel {floor:.1} rows/sec — regression guard, NOT a GPU projection)"
    );
    assert!(
        rps_small >= floor,
        "K={K_SMALL} exact CPU encode {rps_small:.1} rows/sec fell below the regression sentinel \
         {floor:.1} rows/sec — the encode stalled or catastrophically regressed (this is a CPU \
         perf/liveness guard, NOT a GPU deployment claim)"
    );
    assert!(
        rps_large >= floor,
        "K={K_LARGE} exact CPU encode {rps_large:.1} rows/sec fell below the regression sentinel \
         {floor:.1} rows/sec — the encode stalled or catastrophically regressed (this is a CPU \
         perf/liveness guard, NOT a GPU deployment claim)"
    );

    // ── DEPLOYMENT / SURROGATE DECISION (#988, #1412) ──────────────────────
    //
    // The decision is empirical and DEVICE-ONLY. On a CPU-only host there is no
    // device measurement, so the result is `Undetermined`. On a CUDA host the
    // production device-resident certified encode is measured end-to-end below
    // and the decision MUST be the exact comparison between that measured device
    // rate and the 100k rows/sec/GPU target — no resident-solve component proxy
    // and no CPU×constant projection.
    let device_present = GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in encode benchmark: {error}"))
        .is_some_and(|runtime| runtime.device_count() > 0);
    let mut decision = if device_present {
        EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::DeviceNotEngaged)
    } else {
        EncodeDeploymentDecision::blocked(EncodeDecisionBlocked::NoDevice)
    };
    if device_present {
        let (atom, atlas, rows, amplitudes) = build_device_encode_fixture();
        let device_config = AtlasConfig::default();
        let dev = EncodeAtomDevice::from_atom_atlas(&atom, &atlas.atoms[0], &device_config)
            .expect("device encode fixture must lower to device form");
        let tput = measure_device_encode_throughput(&dev, &rows, &amplitudes);
        decision = tput.decision;
        println!(
            "DEVICE-FULL-ENCODE n={} path={:?} measured_rows_per_sec={:.0} \
             target={GPU_DEPLOYMENT_TARGET_ROWS_PER_SEC_PER_GPU:.0} decision={decision:?}",
            tput.n_rows, tput.path, tput.rows_per_sec
        );

        let (batch, path) = sae_certified_encode_batch(&dev, &rows, &amplitudes);
        assert_eq!(
            path, tput.path,
            "timed encode path and validation encode path must agree; otherwise the device \
             engagement flag is not stable enough to certify a deployment decision"
        );
        let certified = batch.iter().filter(|row| row.cert.certified()).count();
        assert!(
            certified > rows.len() / 2,
            "device encode fixture must be non-vacuous: certified {certified}/{} rows",
            rows.len()
        );

        assert!(
            matches!(tput.path, EncodePath::Device),
            "#1412: CUDA was present but the exact encode did not dispatch to the device; 0% GPU \
             utilization is a hard failure, not a CPU pass (path={:?}, decision={decision:?})",
            tput.path
        );
        assert!(
            decision.surrogate_unneeded(),
            "#1412/#988: the deployment gate must be established by the measured FULL exact-encode \
             device throughput. Measured {:.1} rows/sec/GPU did not clear the {:.1} target \
             (decision={decision:?})",
            tput.rows_per_sec,
            GPU_DEPLOYMENT_TARGET_ROWS_PER_SEC_PER_GPU
        );
    } else {
        println!(
            "DEVICE-FULL-ENCODE: no CUDA device — no device measurement; deployment decision \
             remains Undetermined (BLOCKED), never a CPU projection"
        );
    }

    println!("DEPLOYMENT DECISION (full exact encode): {decision:?}");
    if !device_present {
        assert!(
            decision.is_undetermined(),
            "#1412/#988: with no device exact-encode measurement the deployment decision must be \
             Undetermined (BLOCKED on hardware), got {decision:?}"
        );
        assert!(
            !decision.surrogate_unneeded(),
            "#1412: a CPU-only run must NOT claim the amortized surrogate is unneeded"
        );
    }
}
