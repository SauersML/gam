//! #1026 complexity-sweep PROBE (exploration harness, not a permanent pin).
//!
//! Drives the real production engine (`SaeManifoldOuterObjective` + the generic
//! `OuterProblem::run` cascade) on small, fully-understood planted manifolds and
//! prints reconstruction EV + wall time per config, to find where matched-K
//! curved-vs-linear dominance breaks (accuracy) or the fit gets slow (perf).
//!
//! Kept lean: n<=160, K<=2, M<=3 — seconds per fit. Run with:
//!   cargo test -p gam-sae --test probe_1026 -- --nocapture --test-threads=1

use std::sync::Arc;
use std::time::Instant;

use faer::Side as FaerSide;
use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{EuclideanPatchEvaluator, PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{
    SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use gam_solve::rho_optimizer::OuterProblem;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};

const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LR: f64 = 1.0;
const RIDGE_EXT: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
const SEED_LOGIT: f64 = 6.0 * TAU;

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

#[derive(Clone, Copy)]
enum Arm {
    Curved, // periodic M=3
    Linear, // euclidean degree-1, M=2
}
impl Arm {
    fn m(self) -> usize {
        match self {
            Arm::Curved => 3,
            Arm::Linear => 2,
        }
    }
    fn kind(self) -> SaeAtomBasisKind {
        match self {
            Arm::Curved => SaeAtomBasisKind::Periodic,
            Arm::Linear => SaeAtomBasisKind::EuclideanPatch,
        }
    }
    fn eval(self) -> Arc<dyn SaeBasisEvaluator> {
        match self {
            Arm::Curved => Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap()),
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

/// Closed-form per-atom LSQ decoder at gate≈1 from a (K,N,m_max) basis stack.
fn lsq_decoder(phis: &[Array2<f64>], z: &Array2<f64>, gate: f64) -> Vec<Array2<f64>> {
    phis.iter()
        .map(|phi| {
            let m = phi.ncols();
            let mut xw = phi.clone();
            xw.mapv_inplace(|v| v * gate);
            let mut xtx = fast_ata(&xw);
            let mut tr = 0.0;
            for i in 0..m {
                tr += xtx[[i, i]];
            }
            let jit = (tr / m as f64).max(1.0) * 1.0e-8;
            for i in 0..m {
                xtx[[i, i]] += jit;
            }
            let xtz = fast_atb(&xw, z);
            xtx.cholesky(FaerSide::Lower)
                .expect("decoder LSQ")
                .solve_mat(&xtz)
        })
        .collect()
}

/// Build + fit a K-atom term through the production engine; return (fitted EV, seconds).
fn fit_ev(arms: &[Arm], coords_k: &[Array2<f64>], z: &Array2<f64>, label: &str) -> (f64, f64) {
    let k = arms.len();
    let n = z.nrows();
    let gate = 1.0 / (1.0 + (-SEED_LOGIT / TAU).exp());
    let mut phis = Vec::with_capacity(k);
    let mut jets = Vec::with_capacity(k);
    for (a, arm) in arms.iter().enumerate() {
        let (phi, jet) = arm.eval().evaluate(coords_k[a].view()).unwrap();
        phis.push(phi);
        jets.push(jet);
    }
    let decoders = lsq_decoder(&phis, z, gate);
    let mut atoms = Vec::with_capacity(k);
    for (a, arm) in arms.iter().enumerate() {
        let m = arm.m();
        let atom = SaeManifoldAtom::new(
            format!("a{a}"),
            arm.kind(),
            1,
            phis[a].clone(),
            jets[a].clone(),
            decoders[a].clone(),
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(arm.eval());
        atoms.push(atom);
    }
    let logits = Array2::<f64>::from_elem((n, k), SEED_LOGIT);
    let manifolds: Vec<LatentManifold> = arms.iter().map(|a| a.manifold()).collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_k.to_vec(),
        manifolds,
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); k]);
    let init_flat = init_rho.to_flat();
    let np = init_flat.len();
    let mut obj = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        INNER_MAX_ITER,
        LR,
        RIDGE_EXT,
        RIDGE_BETA,
    );
    let t0 = Instant::now();
    OuterProblem::new(np)
        .with_initial_rho(init_flat)
        .run(&mut obj, label)
        .expect("outer fit");
    let secs = t0.elapsed().as_secs_f64();
    let fitted = obj.into_fitted().term.fitted();
    (ev(z, &fitted), secs)
}

fn ev(z: &Array2<f64>, fit: &Array2<f64>) -> f64 {
    let (n, p) = z.dim();
    let mut mean = vec![0.0; p];
    for j in 0..p {
        for i in 0..n {
            mean[j] += z[[i, j]];
        }
        mean[j] /= n as f64;
    }
    let mut ssr = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        for j in 0..p {
            let r = z[[i, j]] - fit[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - mean[j];
            sst += d * d;
        }
    }
    1.0 - ssr / sst.max(1.0e-12)
}

/// Gram–Schmidt of deterministic ambient vectors -> orthonormal frame columns.
fn ortho_frames(k: usize, p: usize) -> Array2<f64> {
    let cols = 2 * k;
    let mut raw = Array2::<f64>::zeros((p, cols));
    for j in 0..cols {
        for i in 0..p {
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((p, cols));
    for j in 0..cols {
        let mut v = raw.column(j).to_owned();
        for prev in 0..j {
            let qp = q.column(prev);
            let dot: f64 = qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for i in 0..p {
                v[i] -= dot * qp[i];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..p {
            q[[i, j]] = v[i] / nrm;
        }
    }
    q
}

#[test]
#[ignore = "exploration probe; run explicitly with --ignored --nocapture"]
fn probe_1026_complexity_sweep() {
    // ---- AXIS A: arc turning Θ (K=1, p=2) — confirm the engine reproduces the
    //      closed-form dominance boundary (margin dies ~Θ=2.4 rad). ----
    println!("\n== AXIS A: arc Θ, K=1, p=2, n=120, σ=0.02 (engine) ==");
    let n = 120usize;
    let sig = 0.02;
    for &turn in &[1.0_f64, 0.5, 0.25] {
        let theta = std::f64::consts::TAU * turn;
        let mut z = Array2::<f64>::zeros((n, 2));
        let mut frac = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / n as f64;
            frac[i] = t;
            let ang = theta * t;
            z[[i, 0]] = ang.cos() + sig * idx_noise(i as u64 * 2);
            z[[i, 1]] = ang.sin() + sig * idx_noise(i as u64 * 2 + 1);
        }
        let coord = Array2::from_shape_fn((n, 1), |(i, _)| (frac[i] + 0.03).rem_euclid(1.0));
        let (ec, tc) = fit_ev(&[Arm::Curved], std::slice::from_ref(&coord), &z, "A-curved");
        let (el, tl) = fit_ev(&[Arm::Linear], std::slice::from_ref(&coord), &z, "A-linear");
        println!(
            "  Θ={:.2} ({:.2} turn): curved={:.4} ({:.1}s)  linear={:.4} ({:.1}s)  margin={:+.4}",
            theta, turn, ec, tc, el, tl, ec - el
        );
    }

    // ---- AXIS D: K=2 superposition, orthogonal vs shared-subspace frames ----
    //      Two full circles, all-active (superposed). Curved K=2 should recover
    //      both; we read EV + time vs orthogonality.
    for &(p, tag) in &[(4usize, "orthogonal (p=4, 2K=4)"), (3usize, "shared-subspace (p=3<2K=4)")] {
        println!("\n== AXIS D: K=2 two superposed circles, {tag}, n=160, σ=0.02 (engine) ==");
        let n = 160usize;
        let q = ortho_frames(2, p.max(4)); // build in >=4-dim then project to p
        let mut z = Array2::<f64>::zeros((n, p));
        let mut coords = Vec::new();
        for a in 0..2 {
            let stride = 0.043 + 0.005 * a as f64;
            let phase = 0.11 * a as f64;
            let mut c = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                let th = ((i as f64) * stride + phase).rem_euclid(1.0);
                c[[i, 0]] = (th + 0.03).rem_euclid(1.0);
                let ang = std::f64::consts::TAU * th;
                for col in 0..p {
                    let u0 = q[[col % q.nrows(), 2 * a]];
                    let u1 = q[[col % q.nrows(), 2 * a + 1]];
                    z[[i, col]] += ang.cos() * u0 + ang.sin() * u1;
                }
            }
            coords.push(c);
        }
        for i in 0..n {
            for col in 0..p {
                z[[i, col]] += sig * idx_noise((i * p + col) as u64 * 7 + 1);
            }
        }
        let (ec, tc) = fit_ev(&[Arm::Curved, Arm::Curved], &coords, &z, "D-curved-K2");
        let (el, tl) = fit_ev(&[Arm::Linear, Arm::Linear], &coords, &z, "D-linear-K2");
        println!(
            "  curved K2 = {:.4} ({:.1}s)   linear K2 = {:.4} ({:.1}s)   margin = {:+.4}",
            ec, tc, el, tl, ec - el
        );
    }
}
