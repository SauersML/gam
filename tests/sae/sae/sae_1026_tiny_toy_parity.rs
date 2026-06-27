//! #1026 — the *tiny toy* reconstruction-parity case, with KNOWN ground truth.
//!
//! The headline #1026 claim is "a curved/hybrid dictionary match-or-beats a
//! pure-linear SAE at matched dictionary size K." The existing in-tree pin
//! (`sae_manifold_reconstruction_parity.rs`) demonstrates it at scale — K=8
//! circles, p=48 ambient, N=2400, three fit arms (one fitting 80 atoms) — which
//! is expensive (per #1026, a single K=2 fit on real data already takes minutes,
//! and per-fit cost is super-linear in K). That scale is the right tool for the
//! large-K parity *measurement*, but it is heavy and the data is a random
//! orthonormal-frame embedding you have to trust rather than read.
//!
//! This test is the opposite: the SMALLEST possible case where every number is
//! understood by inspection, so the *mechanism* (not the magnitude) is pinned
//! cheaply on every CI run.
//!
//! ## The ground truth — literally the unit circle in the plane
//!
//!   z_i = (cos θ_i, sin θ_i) + tiny noise,   θ_i = 2π·i/n,   i = 0..n.
//!
//! That is it. n points marching once around the unit circle in p = 2 ambient
//! dims. Mean ≈ 0, every point has radius ≈ 1, total centered variance
//! SST ≈ Σ_i (cos²θ_i + sin²θ_i) = n. Nothing hidden.
//!
//! ## The matched-K comparison (K = 1, one atom each)
//!
//! * CURVED atom — a periodic harmonic decoder γ(t) = b₀ + b₁cos(2πt) + b₂sin(2πt).
//!   With the latent coordinate t = θ/2π it can trace the *entire* circle with a
//!   single atom, so it reconstructs z almost exactly → EV → 1.
//!
//! * LINEAR atom — the principled linear-SAE baseline: an affine decoder
//!   γ(t) = b₀ + b₁·t, the degree-1 monomial patch {1, t}. As t runs 0 → 1 the
//!   output traces a straight *segment* in ambient space. A straight segment
//!   cannot follow a closed curved loop, so one linear atom can only fit a
//!   secant/diameter of the circle and is starved → EV stays near 0.
//!
//! This is the #1026 "shatter" argument in its irreducible form: a Θ = 2π curved
//! feature needs N(ε) ≈ Θ/(2√(2ε)) linear secants to approximate; at K = 1 the
//! linear dictionary has exactly one secant and loses by a wide, measured margin.
//! Both arms are fit identically through the SAME production outer engine
//! (`SaeManifoldOuterObjective` + `OuterProblem::run`) the recovery pins use; the
//! only difference is the atom basis (periodic M = 3 vs linear M = 2).
//!
//! Cost: two K = 1 fits at n = 96, p = 2, M ≤ 3 — milliseconds-to-seconds, vs the
//! K = 8 / N = 2400 / 80-atom heavy pin. Same mechanism, ~negligible CPU.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use faer::Side as FaerSide;

const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
/// Fixed high gate seed (gate = σ(6) ≈ 1): all rows active, so this is a pure
/// reconstruction comparison with no gating subtlety — exactly the seed the K=1
/// radial-bias pin uses.
const SEED_LOGIT: f64 = 6.0 * TAU;

/// Deterministic Lehmer-style ~N(0,1) noise keyed purely by index (no RNG dep).
fn idx_noise(seed: u64) -> f64 {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((s >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    let mut s2 = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    s2 = s2
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u2 = ((s2 >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
    (u + u2 - 1.0) * std::f64::consts::SQRT_2
}

/// The known truth: n points once around the unit circle in p = 2 dims, plus
/// tiny per-coordinate noise of std `sigma`. Returns (z, θ-fraction in [0,1)).
fn planted_unit_circle(n: usize, sigma: f64) -> (Array2<f64>, Vec<f64>) {
    let mut z = Array2::<f64>::zeros((n, 2));
    let mut frac = Vec::with_capacity(n);
    for row in 0..n {
        let t = row as f64 / n as f64; // fraction of one period in [0,1)
        frac.push(t);
        let theta = std::f64::consts::TAU * t;
        z[[row, 0]] = theta.cos() + sigma * idx_noise(row as u64 * 2);
        z[[row, 1]] = theta.sin() + sigma * idx_noise(row as u64 * 2 + 1);
    }
    (z, frac)
}

/// Which atom basis a fit arm carries.
#[derive(Clone, Copy)]
enum Arm {
    /// Periodic harmonic decoder (cos/sin) on a Circle manifold — can trace the
    /// whole loop with one atom.
    Curved,
    /// Affine degree-1 monomial decoder {1, t} on a Euclidean manifold — the
    /// linear-SAE baseline; one atom is one secant.
    Linear,
}

impl Arm {
    fn basis_size(self) -> usize {
        match self {
            // const + (cos, sin) of the first harmonic.
            Arm::Curved => 3,
            // {1, t}.
            Arm::Linear => 2,
        }
    }
    fn basis_kind(self) -> SaeAtomBasisKind {
        match self {
            Arm::Curved => SaeAtomBasisKind::Periodic,
            // The validated #1026 reconstruction-parity pin uses the degree-1
            // `EuclideanPatch` patch ({1, t}) as its linear-SAE baseline; we
            // mirror that exact combination here.
            Arm::Linear => SaeAtomBasisKind::EuclideanPatch,
        }
    }
    fn evaluator(self) -> Arc<dyn SaeBasisEvaluator> {
        match self {
            Arm::Curved => Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap()),
            // degree-1 euclidean patch over a 1-D latent ⇒ basis {1, t}.
            Arm::Linear => Arc::new(EuclideanPatchEvaluator::new(1, 1).unwrap()),
        }
    }
    fn manifold(self) -> LatentManifold {
        match self {
            Arm::Curved => LatentManifold::Circle { period: 1.0 },
            Arm::Linear => LatentManifold::Euclidean,
        }
    }
}

/// Build the cold K = 1 term for `arm`, seeding the latent coordinate from the
/// KNOWN circle fraction (offset so coordinate recovery is not what is under
/// test — only the decoder's reach is) and the decoder from a closed-form LSQ
/// solve at the cold gate. Identical recipe to the K=1 radial-bias pin, only
/// generalized over the atom basis.
fn build_cold_term(arm: Arm, z: &Array2<f64>, frac: &[f64]) -> SaeManifoldTerm {
    let n = z.nrows();
    let m = arm.basis_size();
    let evaluator = arm.evaluator();
    // Same coordinate seed for BOTH arms (the true circle fraction, lightly
    // offset). The arms differ ONLY in what their decoder can do with it.
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| (frac[i] + 0.03).rem_euclid(1.0));
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    let logits = Array2::<f64>::from_elem((n, 1), SEED_LOGIT);
    let gate0 = 1.0 / (1.0 + (-SEED_LOGIT / TAU).exp()); // σ(seed/τ) ≈ 1

    // Closed-form decoder LSQ init at the cold gate: B = (XᵀX + εI)⁻¹ Xᵀ Z,
    // X = gate·Φ.
    let mut xw = Array2::<f64>::zeros((n, m));
    for row in 0..n {
        for c in 0..m {
            xw[[row, c]] = gate0 * phi[[row, c]];
        }
    }
    let mut xtx = fast_ata(&xw);
    let mut trace = 0.0_f64;
    for i in 0..m {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / m as f64).max(1.0) * 1.0e-8;
    for i in 0..m {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&xw, z);
    let decoder = xtx
        .cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&xtz);

    let atom = SaeManifoldAtom::new(
        "atom_0",
        arm.basis_kind(),
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(arm.evaluator());

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![arm.manifold()],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

/// Fit `arm` through the production outer engine and return the fitted term.
fn run_production_fit(arm: Arm, z: &Array2<f64>, frac: &[f64], label: &str) -> SaeManifoldTerm {
    let term = build_cold_term(arm, z, frac);
    let init_rho = SaeManifoldRho::new(SPARSITY.ln(), SMOOTHNESS.ln(), vec![Array1::<f64>::zeros(0)]);
    let init_flat = init_rho.to_flat();
    let n_params = init_flat.len();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );
    OuterProblem::new(n_params)
        .with_initial_rho(init_flat)
        .run(&mut objective, label)
        .expect("production outer fit must complete");
    objective.into_fitted().term
}

/// Reconstruction explained variance `1 − SSR/SST` (per-column centered),
/// measured identically for both arms so the comparison is unbiased.
fn reconstruction_ev(z: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let (n, p) = z.dim();
    let mut means = vec![0.0_f64; p];
    for j in 0..p {
        let mut acc = 0.0;
        for i in 0..n {
            acc += z[[i, j]];
        }
        means[j] = acc / n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - means[j];
            sst += d * d;
        }
    }
    1.0 - ssr / sst.max(1.0e-12)
}

/// The tiny toy parity pin: on a literal unit circle, one curved atom
/// reconstructs it; one linear atom (the matched-K linear-SAE baseline) can only
/// fit a secant and is starved. Curved match-or-beats linear at K = 1.
#[test]
fn sae_1026_tiny_toy_curved_beats_linear_at_k1_on_unit_circle() {
    let n = 96usize;
    let sigma = 0.02_f64; // ~2% noise — caps EV near 1 for the curved arm
    let (z, frac) = planted_unit_circle(n, sigma);

    let curved = run_production_fit(Arm::Curved, &z, &frac, "tiny-toy-curved-K1");
    let linear = run_production_fit(Arm::Linear, &z, &frac, "tiny-toy-linear-K1");

    let ev_curved = reconstruction_ev(&z, &curved.fitted());
    let ev_linear = reconstruction_ev(&z, &linear.fitted());

    println!("=== #1026 tiny toy reconstruction parity (K=1, unit circle, p=2, n={n}) ===");
    println!("known truth: z_i = (cos 2πt_i, sin 2πt_i) + {sigma:.0e}·noise, t_i = i/n");
    println!("reconstruction EV (1 − SSR/SST):");
    println!("  CURVED (1 periodic atom, M=3)   EV = {ev_curved:.6}");
    println!("  LINEAR (1 linear  atom, M=2)    EV = {ev_linear:.6}");
    println!("  curved − linear margin          = {:.6}", ev_curved - ev_linear);

    // (1) The curved atom reconstructs the known circle to high absolute EV —
    //     the same 0.85 bar the recovery pins hold (2% noise caps EV near 1; one
    //     periodic atom is the sufficient parameterization of one circle).
    assert!(
        ev_curved >= 0.85,
        "TINY-TOY (1) FAIL: curved EV {ev_curved:.6} below the 0.85 bar — one periodic atom \
         must reconstruct the planted unit circle (known truth z=(cosθ,sinθ))"
    );

    // (2) Curved match-or-beats matched-K linear by a wide margin. One linear
    //     atom can only fit a secant of a closed loop, so it is starved. The bar
    //     is a loose 0.10 EV; the measured gap is far larger (a straight segment
    //     cannot follow a Θ=2π circle).
    assert!(
        ev_curved >= ev_linear + 0.10,
        "TINY-TOY (2) FAIL: curved EV {ev_curved:.6} did not beat matched-K linear EV \
         {ev_linear:.6} by the 0.10 shatter margin — at K=1 a single linear secant cannot \
         approximate a full Θ=2π circle (this is the #1026 shatter penalty in miniature)"
    );
}
