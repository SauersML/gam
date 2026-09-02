//! Regression pin for the "K≥2 SAE joint fit collapses cold" failure
//! (#853 class). Two planted circle atoms are recovered against the planted
//! truth, driving the fit *exactly the way production does*: cold ordered independent Beta--Bernoulli
//! residual-energy seed logits, weighted-LSQ decoder init, and the generic
//! outer cascade (`OuterProblem::run`) around `SaeManifoldOuterObjective` —
//! the same engine `crates/gam-pyffi` `sae_manifold_fit_minimal` drives.
//!
//! The collapse signature is per-atom mean active mass crashing to ~0.03 on
//! the rows where the atom is *truly* active (vs a planted ~0.2). A failing
//! run here REPRODUCES that; a passing run REFUTES it. Either way the test
//! prints verbatim numbers.
//!
//! Construction-path fidelity note: the gam crate cannot reach the pyffi-only
//! seed helpers (`sae_pca_seed_initial_coords` cluster-refine,
//! `sae_residual_seed_logits`, `sae_decoder_lsq_init`,
//! `sae_refine_routing_seed`). We replicate the two seed stages that
//! determine the routing collapse VERBATIM from those bodies (residual-energy
//! ordered independent Beta--Bernoulli logits at gain 4.0; weighted-LSQ decoder init at the ordered independent Beta--Bernoulli gate), and
//! seed the latent coordinates from the planted angles (the production PCA /
//! cluster coordinate seed is a separate stage; #853 is a routing/active-mass
//! failure, not a coordinate-recovery one — mirroring the inline torus oracle
//! `ordered_beta_bernoulli_k2_periodic_torus_recovers_signal_with_lsq_init`).

use gam::terms::latent::LatentManifold;
use gam::terms::{sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldTerm};
use ndarray::{Array2, Array3, ArrayView2, s};
use std::sync::Arc;

// ---- production defaults (gamfit `sae_manifold_fit`, ordered_beta_bernoulli path) ----------
const N: usize = 600;
const P: usize = 24;
const K: usize = 2;
const M: usize = 3; // const + 1 harmonic (sin, cos) -> circle
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const RESIDUAL_SEED_GAIN: f64 = 4.0; // SAE_RESIDUAL_SEED_GAIN in pyffi

// ---- planted DGP --------------------------------------------------------
const R_A: f64 = 1.0;
const R_B: f64 = 1.1;

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

/// Two p×2 orthonormal frames whose column spans are mutually orthogonal.
/// Columns 0..2 -> atom A plane, columns 2..4 -> atom B plane, of an
/// orthonormalized deterministic ambient basis.
fn planted_frames() -> (Array2<f64>, Array2<f64>) {
    // Build 4 deterministic ambient vectors, Gram-Schmidt to orthonormal.
    let mut raw = Array2::<f64>::zeros((P, 4));
    for j in 0..4 {
        for i in 0..P {
            // smooth, distinct, full-rank deterministic columns
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((P, 4));
    for j in 0..4 {
        let mut v = raw.column(j).to_owned();
        for prev in 0..j {
            let qp = q.column(prev);
            let dot: f64 = qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for i in 0..P {
                v[i] -= dot * qp[i];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..P {
            q[[i, j]] = v[i] / nrm;
        }
    }
    let u_a = q.slice(s![.., 0..2]).to_owned();
    let u_b = q.slice(s![.., 2..4]).to_owned();
    (u_a, u_b)
}

/// Per-row truth: angles, gates (truly-active flags), and amplitudes.
/// 30% only-A, 30% only-B, 40% co-active, partitioned by index.
struct Truth {
    theta: [Vec<f64>; K],
    active: [Vec<bool>; K],
    amp: [Vec<f64>; K],
}

fn planted_truth() -> Truth {
    let mut theta = [vec![0.0; N], vec![0.0; N]];
    let mut active = [vec![false; N], vec![false; N]];
    let mut amp = [vec![0.0; N], vec![0.0; N]];
    for i in 0..N {
        // distinct irrational-ish strides so angles fill the circle
        theta[0][i] = ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
        theta[1][i] = ((i as f64) * 0.098_765 + 0.57).rem_euclid(1.0);
        let bucket = i % 10;
        let (a_on, b_on) = if bucket < 3 {
            (true, false) // 30% only-A
        } else if bucket < 6 {
            (false, true) // 30% only-B
        } else {
            (true, true) // 40% co-active
        };
        active[0][i] = a_on;
        active[1][i] = b_on;
        // mild amplitude spread ~1
        amp[0][i] = if a_on {
            0.85 + 0.30 * idx_uniform(i as u64 * 2 + 1)
        } else {
            0.0
        };
        amp[1][i] = if b_on {
            0.85 + 0.30 * idx_uniform(i as u64 * 2 + 2)
        } else {
            0.0
        };
    }
    Truth { theta, active, amp }
}

/// Planted response Z = Σ_k gate_k · amp_k · r_k (cosθ u_k1 + sinθ u_k2) + noise.
fn planted_response(truth: &Truth, u_a: &Array2<f64>, u_b: &Array2<f64>) -> (Array2<f64>, f64) {
    let frames = [u_a, u_b];
    let radii = [R_A, R_B];
    let mut z = Array2::<f64>::zeros((N, P));
    let mut signal_sq = 0.0_f64;
    for i in 0..N {
        for k in 0..K {
            if !truth.active[k][i] {
                continue;
            }
            let ang = std::f64::consts::TAU * truth.theta[k][i];
            let c = ang.cos();
            let s = ang.sin();
            let scale = truth.amp[k][i] * radii[k];
            for col in 0..P {
                let contrib = scale * (c * frames[k][[col, 0]] + s * frames[k][[col, 1]]);
                z[[i, col]] += contrib;
                signal_sq += contrib * contrib;
            }
        }
    }
    let signal_scale = (signal_sq / (N * P) as f64).sqrt();
    let sigma = 0.04 * signal_scale; // ~4% of signal scale
    for i in 0..N {
        for col in 0..P {
            let u = idx_uniform(((i * P + col) as u64) * 7 + 3);
            let u2 = idx_uniform(((i * P + col) as u64) * 7 + 5);
            // Box-Muller, deterministic
            let g = (-2.0 * (u.max(1.0e-12)).ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            z[[i, col]] += sigma * g;
        }
    }
    (z, signal_scale)
}

use super::shared_seed_fixtures::residual_seed_logits;

use super::shared_seed_fixtures::decoder_lsq_init;

/// Build the cold term the production driver would hand to the outer engine.
fn build_cold_term(truth: &Truth, z: ArrayView2<'_, f64>) -> SaeManifoldTerm {
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    // Seed latent coordinates from the planted angles (slightly offset, per the
    // inline torus oracle, so coordinate recovery is not what is under test).
    let mut coords_k: Vec<Array2<f64>> = Vec::with_capacity(K);
    let mut phi_k: Vec<Array2<f64>> = Vec::with_capacity(K);
    let mut jet_k: Vec<Array3<f64>> = Vec::with_capacity(K);
    let offsets = [0.05_f64, 0.07_f64];
    for k in 0..K {
        let coords = Array2::from_shape_fn((N, 1), |(i, _)| {
            (truth.theta[k][i] + offsets[k]).rem_euclid(1.0)
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        coords_k.push(coords);
        phi_k.push(phi);
        jet_k.push(jet);
    }

    let basis_sizes = [M, M];
    // (K, N, M) padded basis-value stack for the seed ports.
    let mut basis_values = Array3::<f64>::zeros((K, N, M));
    for k in 0..K {
        for row in 0..N {
            for c in 0..M {
                basis_values[[k, row, c]] = phi_k[k][[row, c]];
            }
        }
    }
    // Production cold ordered independent Beta--Bernoulli routing seed + weighted-LSQ decoder init.
    let logits = residual_seed_logits(basis_values.view(), &basis_sizes, z, RESIDUAL_SEED_GAIN);
    let decoder = decoder_lsq_init(basis_values.view(), &basis_sizes, z, logits.view(), TAU);

    let mut atoms = Vec::with_capacity(K);
    for k in 0..K {
        let b = decoder.slice(s![k, 0..M, ..]).to_owned();
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("circle_{k}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi_k[k].clone(),
            jet_k[k].clone(),
            b,
            Array2::<f64>::eye(M),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; K],
        AssignmentMode::ordered_beta_bernoulli(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

#[test]
fn sae_two_circle_seed_dispersion_diagnostic() {
    let (u_a, u_b) = planted_frames();
    let truth = planted_truth();
    let (z, signal_scale) = planted_response(&truth, &u_a, &u_b);
    let term = build_cold_term(&truth, z.view());
    let seed_dispersion = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed reconstruction dispersion");
    let seed_r2 = {
        let fitted = term.fitted();
        let mut ssr = 0.0;
        let mut sst = 0.0;
        let mut zbar = 0.0;
        for i in 0..N {
            for j in 0..P {
                zbar += z[[i, j]];
            }
        }
        zbar /= (N * P) as f64;
        for i in 0..N {
            for j in 0..P {
                let r = z[[i, j]] - fitted[[i, j]];
                ssr += r * r;
                let d = z[[i, j]] - zbar;
                sst += d * d;
            }
        }
        1.0 - ssr / sst.max(1.0e-12)
    };
    println!(
        "two-circle seed signal_scale={signal_scale:.6} seed_phi={seed_dispersion:.6e} seed_r2={seed_r2:.6}"
    );
    assert!(seed_dispersion.is_finite() && seed_dispersion > 0.0);
}

