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
#[allow(dead_code)] // Linear arm retained for arc/superposition axes; rareness uses Curved only
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

/// Build + fit a K-atom term through the production engine; return (fitted Z, seconds).
fn fit_ev(arms: &[Arm], coords_k: &[Array2<f64>], z: &Array2<f64>, label: &str) -> (Array2<f64>, f64) {
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
    // Dispersion-scaled seed (as the production K=1 pins do): well-conditioned
    // starting rho so the outer cascade converges fast instead of wandering.
    let disp = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed dispersion");
    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); k])
        .seed_scaled_by_dispersion(disp)
        .expect("dispersion seed scaling");
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
    (fitted, secs)
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

/// EV measured over a row subset (the active rows of a rare feature).
fn ev_subset(z: &Array2<f64>, fit: &Array2<f64>, active: &[bool]) -> f64 {
    let p = z.ncols();
    let rows: Vec<usize> = (0..z.nrows()).filter(|&i| active[i]).collect();
    if rows.is_empty() {
        return f64::NAN;
    }
    let mut mean = vec![0.0; p];
    for &i in &rows {
        for j in 0..p {
            mean[j] += z[[i, j]];
        }
    }
    for m in &mut mean {
        *m /= rows.len() as f64;
    }
    let (mut ssr, mut sst) = (0.0, 0.0);
    for &i in &rows {
        for j in 0..p {
            let r = z[[i, j]] - fit[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - mean[j];
            sst += d * d;
        }
    }
    1.0 - ssr / sst.max(1.0e-12)
}

#[test]
#[ignore = "exploration probe; run explicitly with --ignored --nocapture"]
fn probe_1026_complexity_sweep() {
    // ---- SANITY: one dispersion-seeded K=1 fit should now converge FAST (the
    //      earlier un-seeded probe wandered for ~1 min/fit). ----
    {
        let n = 120usize;
        let sig = 0.02;
        let mut z = Array2::<f64>::zeros((n, 2));
        let mut frac = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / n as f64;
            frac[i] = t;
            let ang = std::f64::consts::TAU * t;
            z[[i, 0]] = ang.cos() + sig * idx_noise(i as u64 * 2);
            z[[i, 1]] = ang.sin() + sig * idx_noise(i as u64 * 2 + 1);
        }
        let coord = Array2::from_shape_fn((n, 1), |(i, _)| (frac[i] + 0.03).rem_euclid(1.0));
        let (fitted, secs) = fit_ev(&[Arm::Curved], std::slice::from_ref(&coord), &z, "sanity");
        println!(
            "== SANITY K=1 full circle (dispersion-seeded): EV={:.4} in {secs:.1}s ==",
            ev(&z, &fitted)
        );
    }

    // ---- RARENESS axis: a single curved feature active on only a FRACTION of
    //      rows (rest = pure noise), all gates seeded high. Does the K=1 IBP gate
    //      RECOVER the rare feature (drive inactive gates down, reconstruct the
    //      active rows), or COLLAPSE (prune it, EV_active -> 0)? EV is measured on
    //      the ACTIVE rows — the recoverable signal. ----
    println!("\n== RARENESS: K=1 circle active on a fraction of rows, p=4, n=400, σ=0.02 ==");
    let n = 400usize;
    let p = 4usize;
    let sig = 0.02;
    let q = ortho_frames(1, p);
    for &frac_active in &[1.0_f64, 0.25, 0.10, 0.04] {
        let nact = ((frac_active * n as f64).round() as usize).max(4);
        let mut active = vec![false; n];
        // Spread the active rows deterministically across [0,n).
        for t in 0..nact {
            active[(t * n) / nact] = true;
        }
        let mut z = Array2::<f64>::zeros((n, p));
        let mut coord = Array2::<f64>::zeros((n, 1));
        let mut a_idx = 0usize;
        for i in 0..n {
            if active[i] {
                let th = ((a_idx as f64) / nact as f64).rem_euclid(1.0);
                a_idx += 1;
                coord[[i, 0]] = (th + 0.03).rem_euclid(1.0);
                let ang = std::f64::consts::TAU * th;
                for col in 0..p {
                    z[[i, col]] = ang.cos() * q[[col, 0]] + ang.sin() * q[[col, 1]];
                }
            } else {
                // Inactive row: arbitrary coord, pure noise (no circle signal).
                coord[[i, 0]] = ((i as f64) * 0.013).rem_euclid(1.0);
            }
            for col in 0..p {
                z[[i, col]] += sig * idx_noise((i * p + col) as u64 * 7 + 1);
            }
        }
        let (fitted, secs) = fit_ev(&[Arm::Curved], std::slice::from_ref(&coord), &z, "rare");
        let ev_active = ev_subset(&z, &fitted, &active);
        let ev_all = ev(&z, &fitted);
        // Collapse signature: EV_active -> 0 means the IBP gate pruned the rare
        // feature (or failed to gate noise); EV_active staying high means recovery.
        let verdict = if ev_active < 0.5 { "  <-- RECOVERY COLLAPSE" } else { "" };
        println!(
            "  active={:.2} ({:>3} rows): EV_active={:.4}  EV_all={:.4}  ({:.1}s){}",
            frac_active, nact, ev_active, ev_all, secs, verdict
        );
    }
}
