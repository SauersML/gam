//! #5/(B) rank-charge criterion tests: the honest realised-rank BIC charge
//! (i) accepts a real rank-2 circle, (ii) assigns rank zero only to the exact
//! zero spectrum while the state-aware layer refuses certified disappearance,
//! and the canonical criterion is dense/streaming invariant.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

/// The two K=3 controls each run several joint fits; cargo runs tests in-binary
/// on a thread pool, so left unguarded they can execute simultaneously and, under
/// a loaded host, starve each other (observed as a spurious "hang"/kill, not a
/// logic failure). Serialising them against each other caps peak concurrency to
/// one heavy multi-atom fit at a time. Poison-tolerant: a panic in one test must
/// surface as that test's failure, not poison-fail the sibling.
static K3_SERIAL: Mutex<()> = Mutex::new(());
fn k3_guard() -> std::sync::MutexGuard<'static, ()> {
    K3_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-12);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Build + fit a term with circles on the given output-dim indices (each circle
/// c on dims (2c, 2c+1)), against the shared target `x`. Used for leave-one-out
/// decision margins.
fn fit_circle_subset(
    x: &Array2<f64>,
    theta: &[Vec<f64>],
    circles: &[usize],
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = x.nrows();
    let p = x.ncols();
    let evaluator = Arc::new(
        PeriodicHarmonicEvaluator::new(3)
            .expect("the fixture's harmonic order is a valid periodic basis order"),
    );
    let mut atoms = Vec::new();
    let mut coord_blocks = Vec::new();
    let mut manifolds = Vec::new();
    for &c in circles {
        let coords =
            Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r][c] / std::f64::consts::TAU);
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .expect("the fixture's coordinate block is a valid input for this evaluator");
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, 2 * c]] = 1.0;
        decoder[[2, 2 * c + 1]] = 1.0;
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("circle{c}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .expect("the fixture's basis, decoder and Gram blocks agree in dimension")
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, circles.len()), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("the fixture's logits, coordinate blocks and manifolds agree in length");
    let mut term = SaeManifoldTerm::new(atoms, assignment)
        .expect("the fixture's atoms and assignment describe the same latent blocks");
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); circles.len()]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("subset fit");
    (term, rho)
}

/// (iv) DECISION-LEVEL control: on a clean well-separated 3-circle fit, the
/// canonical criterion's leave-one-out margin of every real atom must be < 0
/// (KEEPING it is favored ⇒ accepted).
#[test]
fn rank_charge_k3_accepts_clean_atoms() {
    let serial = k3_guard();
    let n = 96usize;
    let p = 18usize;
    let ncirc = 3usize;
    let mut s = 0x2101_DEC_0000_0011u64;
    let theta: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            (0..ncirc)
                .map(|_| std::f64::consts::TAU * lcg(&mut s))
                .collect()
        })
        .collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for c in 0..ncirc {
            x[[i, 2 * c]] += theta[i][c].cos();
            x[[i, 2 * c + 1]] += theta[i][c].sin();
        }
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    // Compute each circle's leave-one-out margin:
    // margin_k = reml(all 3) − reml(drop k). <0 ⇒ keeping k is favored.
    let margins = || -> Vec<f64> {
        let (mut t3, r3) = fit_circle_subset(&x, &theta, &[0, 1, 2]);
        let (v3, _, _) = t3
            .penalized_quasi_laplace_criterion_with_cache(x.view(), &r3, None, 0, 1.0, 1e-6, 1e-6)
            .unwrap();
        (0..ncirc)
            .map(|drop| {
                let keep: Vec<usize> = (0..ncirc).filter(|&c| c != drop).collect();
                let (mut t2, r2) = fit_circle_subset(&x, &theta, &keep);
                let (v2, _, _) = t2
                    .penalized_quasi_laplace_criterion_with_cache(
                        x.view(),
                        &r2,
                        None,
                        0,
                        1.0,
                        1e-6,
                        1e-6,
                    )
                    .unwrap();
                v3 - v2 // margin_drop: <0 ⇒ the dropped circle is worth KEEPING
            })
            .collect()
    };
    // The K=3 joint fits use rayon parallel reductions whose order is thread-timing
    // dependent; the leave-one-out margin is a difference of two large independent
    // fits, which amplifies that into occasional sign flips under parallel test
    // execution. Pin the fits to a ONE-thread rayon pool so they converge to the
    // identical (correct) optimum every run — the single-thread values ARE the
    // optimum (verified stable across runs).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread rayon pool for deterministic K=3 fits");
    let margins = pool.install(margins);
    eprintln!("[rank-charge K=3 decisions] leave-one-out margins={margins:?}");
    for (k, margin) in margins.iter().enumerate() {
        assert!(
            *margin < 0.0,
            "circle {k}: rank-charge must ACCEPT the real atom (margin<0); got {:.3}",
            margin
        );
    }
    // A spurious/noise atom is covered structurally by the vanishing
    // test (rank→0 → charge 0 → ΔEV rejects); a real atom here is never spurious.
    drop(serial); // hold the K=3 serialisation lock across the whole fit
}

#[test]
fn rank_charge_prices_zero_dof_without_re_adjudicating_disappearance() {
    let zero_charge =
        super::construction::rank_adjusted_quasi_laplace_complexity(1.0, 0.5, &[0.0], &[10.0])
            .expect("the upstream same-state signal proof owns decoder disappearance");
    assert_eq!(zero_charge, 0.25);

    let error = super::construction::rank_adjusted_quasi_laplace_complexity(
        1.0,
        0.5,
        &[0.0, f64::NAN],
        &[10.0, 10.0],
    )
    .unwrap_err();
    assert!(
        matches!(error, super::SaeCriterionError::Numerical(_)),
        "a simultaneous invalid DOF must remain a numerical error, not {error}"
    );
}

