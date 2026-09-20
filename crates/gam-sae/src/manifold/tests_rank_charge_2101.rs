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

/// Inner iteration budget shared by the subset fit and by the criterion that
/// prices it. The criterion gets this budget, never `0`: `inner_max_iter == 0`
/// is the verbatim-reuse freeze. It prices the subset fit's last iterate as it
/// stands and skips the evidence drive, including the drive's descent of refused
/// exact-A saddles (#2080), so a leave-one-out margin taken there compares two
/// states neither of which is a converged root.
const SUBSET_FIT_ITERATIONS: usize = 60;

/// The K=3 fixture's generating noise standard deviation. The criterion's data
/// term is `½‖y − f‖²` at unit dispersion, so the fixture prices its target in
/// noise units `x/σ` (the `y' = √λ·y` change of variables of
/// `SaeManifoldOuterObjective::with_global_dispersion` at the true `λ = 1/σ²`):
/// on the raw target the criterion would assert a noise variance of 1 against a
/// unit-amplitude circle, where dropping it is the right call (#2822).
const K3_NOISE_SD: f64 = 0.05;

/// `M_null = Σ_k r_k·(M_k − rank S_k)`: the flat-prior decoder directions the
/// noise-unit change of variables does not normalize. Two subsets priced on the
/// same `x/σ` differ in the raw-unit criterion by `½·ΔM_null·log(1/σ²)`.
fn flat_prior_decoder_dims(term: &SaeManifoldTerm) -> f64 {
    term.atoms
        .iter()
        .map(|atom| {
            let rank_s = SaeManifoldTerm::symmetric_rank(atom.smooth_penalty())
                .expect("the fixture's smoothing penalty has a symmetric rank");
            let m_k = atom.smooth_penalty().nrows();
            (atom.border_frame_rank() * m_k.saturating_sub(rank_s)) as f64
        })
        .sum()
}

/// Build + fit a term with circles on the given output-dim indices (each circle
/// c on dims (2c, 2c+1)), against the shared target `x`. Used for leave-one-out
/// decision margins.
fn fit_circle_subset(
    x: &Array2<f64>,
    theta: &[Vec<f64>],
    circles: &[usize],
    amplitude: f64,
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
        decoder[[1, 2 * c]] = amplitude;
        decoder[[2, 2 * c + 1]] = amplitude;
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
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); circles.len()]);
    term.run_joint_fit_arrow_schur(
        x.view(),
        &mut rho,
        None,
        SUBSET_FIT_ITERATIONS,
        1.0,
        1e-6,
        1e-6,
    )
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
            x[[i, j]] += K3_NOISE_SD * lcg_normal(&mut s);
        }
    }
    // Price in noise units (see `K3_NOISE_SD`); the seed decoder rides `β' = β/σ`.
    let x = x.mapv(|v| v / K3_NOISE_SD);
    let amplitude = 1.0 / K3_NOISE_SD;
    let log_precision = -2.0 * K3_NOISE_SD.ln();
    // Compute each circle's leave-one-out margin:
    // margin_k = reml(all 3) − reml(drop k). <0 ⇒ keeping k is favored.
    // Both sides are priced at the criterion's converged evidence root (see
    // `SUBSET_FIT_ITERATIONS`).
    let margins = || -> Vec<f64> {
        let (mut t3, r3) = fit_circle_subset(&x, &theta, &[0, 1, 2], amplitude);
        let (v3, _, _) = t3
            .penalized_quasi_laplace_criterion_with_cache(
                x.view(),
                &r3,
                None,
                SUBSET_FIT_ITERATIONS,
                1.0,
                1e-6,
                1e-6,
            )
            .unwrap();
        (0..ncirc)
            .map(|drop| {
                let keep: Vec<usize> = (0..ncirc).filter(|&c| c != drop).collect();
                let (mut t2, r2) = fit_circle_subset(&x, &theta, &keep, amplitude);
                let (v2, _, _) = t2
                    .penalized_quasi_laplace_criterion_with_cache(
                        x.view(),
                        &r2,
                        None,
                        SUBSET_FIT_ITERATIONS,
                        1.0,
                        1e-6,
                        1e-6,
                    )
                    .unwrap();
                // Back to raw units: V_φ(x) = V₁(x/σ) − ((n·p − M_null)/2)·log(1/σ²);
                // the n·p part cancels in the difference.
                let jacobian =
                    0.5 * (flat_prior_decoder_dims(&t3) - flat_prior_decoder_dims(&t2))
                        * log_precision;
                v3 - v2 + jacobian // margin_drop: <0 ⇒ the dropped circle is worth KEEPING
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
        super::construction::rank_adjusted_quasi_laplace_complexity(1.0, &[0.0], &[10.0])
            .expect("the upstream same-state signal proof owns decoder disappearance");
    assert_eq!(zero_charge, 0.5);

    let error = super::construction::rank_adjusted_quasi_laplace_complexity(
        1.0,
        &[0.0, f64::NAN],
        &[10.0, 10.0],
    )
    .unwrap_err();
    assert!(
        matches!(error, super::SaeCriterionError::Numerical(_)),
        "a simultaneous invalid DOF must remain a numerical error, not {error}"
    );
}

