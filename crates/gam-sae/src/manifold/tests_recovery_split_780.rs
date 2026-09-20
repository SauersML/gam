#![cfg(test)]
// Split from tests.rs under the #780 oversized-file gate: recovery-suite +
// registry/assignment tests from line ~6560 onward. Shared fixtures come via
// the parent-module glob below.
//
// `manifold/mod.rs` declares this module only as `#[cfg(test)] mod
// tests_recovery_split_780;`, and this attribute states that test-only scope in
// the file itself — a claim the compiler enforces rather than one carried by
// the filename.
#![cfg(test)]

use super::*;
use gam_solve::arrow_schur::{
    ArrowFactorSlab, ArrowHtbetaCache, ArrowPcgDiagnostics, ArrowSolverMode, ArrowUndampedFactors,
    BetaSchurSpectralConditioning,
};
use ndarray::array;

/// Torus T^2 fit on synthetic data with a known two-frequency signal.
/// Drives a single torus atom through the [`SaeManifoldTerm`] Newton loop
/// and checks that the in-sample reconstruction R² clears 0.5.
#[test]
pub(crate) fn sae_torus_atom_recovers_two_frequency_synthetic() {
    let n = 96usize;
    let p = 4usize;
    let h = 3usize;
    let d = 2usize;
    let evaluator = TorusHarmonicEvaluator::new(d, h).unwrap();
    let m = evaluator.basis_size();
    // True coords on T^2 (phase in [0, 1)).
    let mut true_coords = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        true_coords[[i, 0]] = ((i as f64) * 0.137).rem_euclid(1.0);
        true_coords[[i, 1]] = ((i as f64) * 0.241 + 0.13).rem_euclid(1.0);
    }
    // Synthetic target: a low-frequency periodic signal on T^2 mixed
    // linearly into a p-dim ambient.
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t1 = 2.0 * std::f64::consts::PI * true_coords[[i, 0]];
        let t2 = 2.0 * std::f64::consts::PI * true_coords[[i, 1]];
        z[[i, 0]] = t1.sin() + 0.3 * t2.cos();
        z[[i, 1]] = t1.cos() + 0.2 * (t1 + t2).sin();
        z[[i, 2]] = t2.sin();
        z[[i, 3]] = 0.5 * (t1 - t2).cos();
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
    // Initialise from the true coords (this test exercises basis correctness
    // and decoder fit, not coordinate identification on T^2).
    let (phi0, jet0) = evaluator.evaluate(true_coords.view()).unwrap();
    // Penalty: identity-on-non-constant + tiny floor on constant.
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "torus_atom",
        SaeAtomBasisKind::Torus,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TorusHarmonicEvaluator::new(d, h).unwrap()));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_coords],
        vec![LatentManifold::Product(vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ])],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // ARD log-precision is per-axis (length == atom latent dim), not a
    // single scalar — see `SaeManifoldRho::to_flat` / `from_flat` and
    // the validation in `negative_log_ard_prior` (`ARD rho atom k has
    // len ... but atom dim is d`).
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(d)]);
    // #2822 — the data least-squares decoder at the planted chart; an entry refuses a zero decoder.
    term.refit_decoder_least_squares_at_current_state(z.view(), Some(&rho))
        .unwrap();
    let ridge = 1.0e-6;
    for _ in 0..10 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .unwrap();
        if !loss.total().is_finite() {
            break;
        }
    }
    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    let mut sse = 0.0_f64;
    for ((row, col), v) in fitted.indexed_iter() {
        let r = v - z[[row, col]];
        sse += r * r;
    }
    let r2 = 1.0 - sse / sst.max(1.0e-12);
    assert!(
        r2 >= 0.5,
        "torus atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
    );
}

/// Sphere S² fit on a synthetic spherical signal. Drives a single sphere
/// atom through the [`SaeManifoldTerm`] Newton loop and checks in-sample
/// R² ≥ 0.5.
#[test]
pub(crate) fn sae_sphere_atom_recovers_synthetic_signal() {
    let n = 96usize;
    let p = 3usize;
    // Ambient coordinate width: `S²` is intrinsically 2-D but carried as a unit
    // 3-vector.
    let d = 3usize;
    // A spiral of true directions. The generating (lat, lon) is a convenient way
    // to trace one, but the COORDINATE is the direction it names.
    let mut true_coords = Array2::<f64>::zeros((n, d));
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64) / (n as f64);
        let lat = -0.5 + 1.0 * t;
        let lon = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * t;
        let (x, y, zc) = (lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
        true_coords[[i, 0]] = x;
        true_coords[[i, 1]] = y;
        true_coords[[i, 2]] = zc;
        z[[i, 0]] = x;
        z[[i, 1]] = y;
        z[[i, 2]] = zc;
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
    let (phi0, jet0) = AmbientSphereHarmonicEvaluator::new(2)
        .unwrap()
        .evaluate(true_coords.view()).unwrap();
    let m = phi0.ncols();
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere_atom",
        SaeAtomBasisKind::Sphere,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_coords],
        vec![LatentManifold::Sphere { dim: 3 }],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // The sphere atom's coordinate is a 3-wide ambient vector, so per-axis ARD
    // carries one log-precision per AMBIENT axis. A shorter block would be
    // indexed out of bounds in the per-axis assembly loop and is rejected by
    // the per-axis ARD contract.
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(3)]);
    // #2822 — the data least-squares decoder at the planted chart; an entry refuses a zero decoder.
    term.refit_decoder_least_squares_at_current_state(z.view(), Some(&rho))
        .unwrap();
    let ridge = 1.0e-6;
    for _ in 0..10 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .unwrap();
        if !loss.total().is_finite() {
            break;
        }
    }
    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    let mut sse = 0.0_f64;
    for ((row, col), v) in fitted.indexed_iter() {
        let r = v - z[[row, col]];
        sse += r * r;
    }
    let r2 = 1.0 - sse / sst.max(1.0e-12);
    assert!(
        r2 >= 0.5,
        "sphere atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
    );
}

/// Mirror of the Python `test_sae_manifold_softmax_dispatch` shape: drive a
/// single periodic atom on a 1-harmonic synthetic target with 10 Newton
/// steps end-to-end in Rust and check that the multi-step loop achieves
/// in-sample R² ≥ 0.95.
#[test]
pub(crate) fn sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2() {
    let n = 64usize;
    let m = 3usize;
    let p = 1usize;

    let true_t: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * true_t[i];
        z[[i, 0]] = 0.7 * angle.sin() + 0.3 * angle.cos();
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();

    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut coords0_data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        // Phase-shifted initialization so the optimizer must do real work.
        coords0_data[[i, 0]] = (true_t[i] + 0.25).rem_euclid(1.0);
    }
    let (phi0, jet0) = evaluator.evaluate(coords0_data.view()).unwrap();

    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic_atom",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords0_data],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
    // #2822 — the data least-squares decoder at the seeded chart; an entry refuses a zero decoder.
    term.refit_decoder_least_squares_at_current_state(z.view(), Some(&rho))
        .unwrap();

    let max_iter = 10usize;
    let learning_rate = 1.0;
    let ridge = 1.0e-6;
    let mut prev_total = f64::INFINITY;
    for _ in 0..max_iter {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, learning_rate, ridge, ridge)
            .unwrap();
        let total = loss.total();
        if !total.is_finite() {
            break;
        }
        let denom = prev_total.abs().max(1.0e-12);
        let rel = (prev_total - total).abs() / denom;
        prev_total = total;
        if rel < 1.0e-6 {
            break;
        }
    }

    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    let mut ssr = 0.0;
    for i in 0..n {
        let r = z[[i, 0]] - fitted[[i, 0]];
        ssr += r * r;
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);
    assert!(
        r2 >= 0.95,
        "10-step in-sample R² = {r2:.4} (ssr={ssr:.6}, sst={sst:.6}) should be >= 0.95"
    );
}

#[test]
pub(crate) fn sae_registry_refuses_assignment_sparsity_penalties() {
    let n = 3usize;
    let k = 2usize;
    let logits = Array2::<f64>::zeros((n, k));
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::softmax(0.7),
    )
    .expect("valid assignment");
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|atom_idx| {
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("periodic_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                Array2::<f64>::ones((n, 1)),
                Array3::<f64>::zeros((n, 1, 1)),
                Array2::<f64>::zeros((1, 1)),
                Array2::<f64>::eye(1),
            )
            .expect("valid atom")
        })
        .collect();
    let term = SaeManifoldTerm::new(atoms, assignment).expect("valid SAE term");

    let mut softmax_registry = AnalyticPenaltyRegistry::new();
    softmax_registry.push(AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(
        gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(k, 0.7),
    )));
    let softmax_err = term
        .validate_analytic_penalty_registry(&softmax_registry)
        .expect_err("SAE registry must reject softmax assignment sparsity");
    assert!(softmax_err.contains("assignment sparsity"));

    let mut ordered_beta_bernoulli_registry = AnalyticPenaltyRegistry::new();
    ordered_beta_bernoulli_registry.push(AnalyticPenaltyKind::OrderedBetaBernoulli(Arc::new(
        gam_terms::analytic_penalties::OrderedBetaBernoulliPenalty::new(k, 1.2, 0.7, false),
    )));
    let ordered_beta_bernoulli_err = term
        .validate_analytic_penalty_registry(&ordered_beta_bernoulli_registry)
        .expect_err("SAE registry must reject ordered Beta--Bernoulli assignment sparsity");
    assert!(ordered_beta_bernoulli_err.contains("assignment sparsity"));
}

/// Regression test for issue #174: K>=2 periodic atoms with zero-init
/// decoder used to collapse to A≈0 because the assignment prior was the
/// only term with non-zero gradient at iter 0. The pyffi entry point now
/// seeds decoder coefficients via a joint LSQ projection of Z onto
/// [a_init · Phi_k]. This test exercises that exact seeding strategy
/// in pure Rust and verifies the joint Newton fit reaches positive R²
/// on a clean K=2 periodic torus signal, mirroring the failing
/// reproducer in #174.
#[test]
pub(crate) fn ordered_beta_bernoulli_k2_periodic_torus_recovers_signal_with_lsq_init() {
    use faer::Side as FaerSide;
    use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};

    let n = 200usize;
    let p = 8usize;
    let k = 2usize;
    let m = 5usize; // 1 (constant) + 2 harmonics * 2 (sin/cos) = 5

    // Build a synthetic K=2 torus signal Z = [cos th1, sin th1, cos th2, sin th2] @ mix
    // with two latent angles. Deterministic seed via index arithmetic.
    let mut theta = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        theta[[i, 0]] = ((i as f64) * 0.07) % 1.0;
        theta[[i, 1]] = ((i as f64) * 0.13 + 0.31) % 1.0;
    }
    let mut raw = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let a1 = 2.0 * std::f64::consts::PI * theta[[i, 0]];
        let a2 = 2.0 * std::f64::consts::PI * theta[[i, 1]];
        raw[[i, 0]] = a1.cos();
        raw[[i, 1]] = a1.sin();
        raw[[i, 2]] = a2.cos();
        raw[[i, 3]] = a2.sin();
    }
    // Deterministic 4x8 mixing matrix.
    let mix = Array2::<f64>::from_shape_fn((4, p), |(i, j)| {
        ((i as f64 + 1.0) * 0.37 + (j as f64) * 0.21).sin()
    });
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut acc = 0.0;
            for r in 0..4 {
                acc += raw[[i, r]] * mix[[r, j]];
            }
            z[[i, j]] = acc;
        }
    }
    // Centre Z so R² is well-defined relative to mean.
    let mut col_mean = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut acc = 0.0;
        for i in 0..n {
            acc += z[[i, j]];
        }
        col_mean[j] = acc / n as f64;
    }
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] -= col_mean[j];
        }
    }

    // Atom coordinates: use the (shifted) true angles so the periodic
    // basis aligns with the signal — the test isolates the decoder-init
    // collapse, not coordinate recovery.
    let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
    for i in 0..n {
        coords_k[0][[i, 0]] = (theta[[i, 0]] + 0.05).rem_euclid(1.0);
        coords_k[1][[i, 0]] = (theta[[i, 1]] + 0.07).rem_euclid(1.0);
    }
    // Periodic basis (constant + 2 harmonics → M=5) for each atom.
    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut phi_k = Vec::with_capacity(k);
    let mut jet_k = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
        phi_k.push(phi);
        jet_k.push(jet);
    }

    // LSQ seed: joint design X = [0.5 * Phi_1 | 0.5 * Phi_2] (ordered Beta--Bernoulli
    // logit 0 gives sigmoid(0/tau) = 0.5 for both atoms), solve normal
    // equations with a small ridge.
    let m_total = k * m;
    let mut x = Array2::<f64>::zeros((n, m_total));
    for atom_idx in 0..k {
        for i in 0..n {
            for col in 0..m {
                x[[i, atom_idx * m + col]] = 0.5 * phi_k[atom_idx][[i, col]];
            }
        }
    }
    let mut xtx = fast_ata(&x);
    let mut trace = 0.0_f64;
    for i in 0..m_total {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / m_total as f64).max(1.0) * 1.0e-8;
    for i in 0..m_total {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, &z);
    let b_joint = xtx
        .cholesky(FaerSide::Lower)
        .expect("LSQ Cholesky")
        .solve_mat(&xtz);

    let mut atoms = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let mut b = Array2::<f64>::zeros((m, p));
        for col in 0..m {
            for j in 0..p {
                b[[col, j]] = b_joint[[atom_idx * m + col, j]];
            }
        }
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("torus_atom_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi_k[atom_idx].clone(),
            jet_k[atom_idx].clone(),
            b,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    // `lambda_sparse` is the ordered Beta--Bernoulli assignment-sparsity prior weight (now wired
    // through `assignment_prior_grad_hdiag_weighted`'s ordered Beta--Bernoulli branch, #853). The
    // Beta-Bernoulli BCE energy toward the self-referential empirical active
    // fraction has its global minimum at the all-off gate, so at the old
    // full weight (`log_lambda_sparse = 0 → λ = 1`) it overwhelmed the
    // truth-seeded data fit and collapsed the assignment off both atoms. A
    // moderate prior weight keeps the sparsity pressure honest while letting
    // the LSQ-seeded reconstruction hold both real atoms active — the
    // realistic operating point this recovery test pins.
    let mut rho = SaeManifoldRho::new((0.02_f64).ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);

    let mut prev_total = f64::INFINITY;
    for _ in 0..30 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .unwrap();
        let total = loss.total();
        if !total.is_finite() {
            break;
        }
        let denom = prev_total.abs().max(1.0e-12);
        let rel = (prev_total - total).abs() / denom;
        prev_total = total;
        if rel < 1.0e-6 {
            break;
        }
    }

    let fitted = term.fitted();
    let mut ssr = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        for j in 0..p {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            sst += z[[i, j]] * z[[i, j]];
        }
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);
    assert!(
        r2 > 0.5,
        "K=2 periodic torus ordered Beta--Bernoulli R² = {r2:.4} (ssr={ssr:.4}, sst={sst:.4}) should be > 0.5 with LSQ-seeded decoder"
    );
    // Also confirm at least one atom remains active (assignment did not
    // collapse to ~0) — the active mass averaged over rows must exceed
    // a non-trivial threshold.
    let assignments = term.assignment.assignments();
    let mean_active: f64 = assignments.iter().copied().sum::<f64>() / (n as f64);
    assert!(
        mean_active > 0.2,
        "mean active mass across rows = {mean_active:.4} should exceed 0.2; assignment did not collapse"
    );
}

/// Regression test for issue #174 + #177 combined: softmax assignment
/// with K=2 periodic atoms should not crash and should reduce loss.
#[test]
pub(crate) fn softmax_k2_periodic_completes_joint_fit_step() {
    let n = 64usize;
    let p = 4usize;
    let k = 2usize;
    let m = 3usize;

    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        z[[i, 0]] = a.sin();
        z[[i, 1]] = a.cos();
        z[[i, 2]] = (2.0 * a).sin();
        z[[i, 3]] = (2.0 * a).cos();
    }

    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
    for i in 0..n {
        coords_k[0][[i, 0]] = (i as f64) / (n as f64);
        coords_k[1][[i, 0]] = ((i as f64) * 2.0 / (n as f64)).rem_euclid(1.0);
    }
    let mut atoms = Vec::new();
    for atom_idx in 0..k {
        let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
        // Non-trivial decoder init (simulate LSQ seeding) so the data-fit
        // signal is non-zero at iter 0.
        let b = Array2::<f64>::from_shape_fn((m, p), |(i, j)| {
            0.1 * ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
        });
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            format!("a_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            b,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);

    // First step must succeed (previously bailed with hessian-diag error).
    let loss0 = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
        .expect("softmax K=2 must complete first joint-fit step");
    assert!(loss0.total().is_finite());
    let loss1 = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
        .expect("softmax K=2 must complete second joint-fit step");
    assert!(loss1.total().is_finite());
}

/// End-to-end Isometry wiring oracle.
///
/// Build a SAE atom around an evaluator whose `second_jet` is now
/// implemented (periodic / sphere / torus), construct an
/// [`IsometryPenalty`] with matching `latent_dim` and `p_out`, refresh
/// the caches via [`refresh_isometry_caches_from_atom`], and check that
///
///   * `IsometryPenalty.value(target, rho)` is strictly positive (the
///     decoder we feed in is not orthonormal so the pullback metric is
///     not the identity, and the Euclidean reference picks up the gap).
///   * `IsometryPenalty.grad_target(target, rho)` is non-zero on at
///     least one latent-coordinate component.
///   * The analytic gradient matches a finite-difference oracle of
///     `value()` w.r.t. `target` (a single coord), where each FD probe
///     drives a fresh cache refresh — this is exactly the chain of
///     calls the SAE outer loop will make.
///
/// The FD oracle re-uses the existing [`refresh_isometry_caches_from_atom`]
/// helper for both the analytic side and the FD side, so any layout
/// mismatch between `J`/`H` would show up as a tolerance failure rather
/// than a silently zero gradient.
pub(crate) fn assert_isometry_wiring_matches_fd(
    evaluator: Arc<dyn SaeBasisSecondJet>,
    coords: Array2<f64>,
) {
    let n_obs = coords.nrows();
    let latent_dim = coords.ncols();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let p: usize = 3;
    // A deterministic non-orthonormal decoder: deterministic LCG-ish
    // floats keep the test reproducible without needing rand.
    let mut decoder = Array2::<f64>::zeros((m, p));
    for i in 0..m {
        for j in 0..p {
            let x = (i as f64) * 0.371 + (j as f64) * 0.193 + 0.5;
            decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
        }
    }
    let smooth = Array2::<f64>::eye(m);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "iso_wire_test",
        SaeAtomBasisKind::Periodic,
        latent_dim,
        phi.clone(),
        jet.clone(),
        decoder.clone(),
        smooth,
    )
    .unwrap()
    .with_basis_second_jet(evaluator);

    let target_slice = PsiSlice::full(n_obs * latent_dim, Some(latent_dim));
    let penalty = IsometryPenalty::new_euclidean(target_slice, p);
    let rho = Array1::<f64>::zeros(1);

    // Without a refresh the penalty holds no decoder jets, so its evaluation
    // precondition refuses by name. The post-refresh evaluation is the wiring
    // under test.
    let target_flat: Array1<f64> = coords.iter().copied().collect();
    let unrefreshed = penalty
        .evaluation_state_precondition(
            gam_terms::analytic_penalties::IsometryEvaluationOrder::Gradient,
            target_flat.len(),
        )
        .expect_err("an unrefreshed isometry penalty has no decoder Jacobian");
    assert!(
        unrefreshed.contains("decoder Jacobian J"),
        "the unrefreshed refusal must name the decoder Jacobian, got: {unrefreshed}"
    );

    // Refresh and re-evaluate.
    let installed_second =
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
    assert!(
        installed_second,
        "evaluator must implement second_jet for this oracle to run"
    );

    let value = penalty.value(target_flat.view(), rho.view());
    assert!(
        value > 1.0e-6,
        "expected non-trivial isometry loss after cache refresh, got {value}"
    );
    let grad = penalty.grad_target(target_flat.view(), rho.view());
    assert_eq!(grad.len(), target_flat.len());
    let max_abs = grad.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    assert!(
        max_abs > 1.0e-6,
        "expected non-zero isometry gradient on at least one component, max |grad|={max_abs}"
    );

    // FD check: bump one coord, refresh, compare value(t±h e_j) against
    // analytic grad[j]. Pick coord (row 0, axis 0).
    let h_fd = 1.0e-5;
    let probe_idx = 0usize; // (row=0, axis=0) flattens to 0.
    let mut coords_plus = coords.clone();
    coords_plus[[0, 0]] += h_fd;
    let mut coords_minus = coords.clone();
    coords_minus[[0, 0]] -= h_fd;

    refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view()).unwrap();
    let target_plus: Array1<f64> = coords_plus.iter().copied().collect();
    let v_plus = penalty.value(target_plus.view(), rho.view());

    refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view()).unwrap();
    let target_minus: Array1<f64> = coords_minus.iter().copied().collect();
    let v_minus = penalty.value(target_minus.view(), rho.view());

    // Reinstall the base caches before reading grad at the base point.
    refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
    let grad_base = penalty.grad_target(target_flat.view(), rho.view());

    let fd = (v_plus - v_minus) / (2.0 * h_fd);
    let analytic = grad_base[probe_idx];
    // Both `value` and `grad_target` use the cached `J` (and `grad_target`
    // also the cached `H`). With finite differencing the cache itself,
    // the analytic-vs-FD agreement bounds the entire pipeline (J build,
    // H build, accessor read, pullback metric, gradient assembly) to
    // O(h²) error. Tolerance 1e-3 leaves headroom for the per-evaluator
    // characteristic magnitude.
    assert!(
        (analytic - fd).abs() <= 1.0e-3 + 1.0e-4 * analytic.abs().max(fd.abs()),
        "isometry grad/FD mismatch at coord 0: analytic={analytic:.6e}, fd={fd:.6e}"
    );
}

#[test]
pub(crate) fn isometry_wiring_periodic_matches_fd() {
    assert_isometry_wiring_matches_fd(
        Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
        array![[0.12], [0.37], [0.58], [0.81]],
    );
}

#[test]
pub(crate) fn isometry_wiring_sphere_matches_fd() {
    assert_isometry_wiring_matches_fd(
        Arc::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()),
        array![
            [0.0, 0.0, 1.0],
            [0.6, -0.8, 0.0],
            [0.36, 0.48, 0.8]
        ],
    );
}

#[test]
pub(crate) fn isometry_wiring_torus_matches_fd() {
    assert_isometry_wiring_matches_fd(
        Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
        array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
    );
}

// [#780 line-count gate] The exact isometry-penalty HVP / PSD-majorizer
// cluster (`deterministic_decoder`, `build_isometry_atom_for_evaluator`,
// `assert_exact_isometry_hvp_*`, `assert_isometry_psd_majorizer_live_*`, the
// `isometry_exact_hvp_*` / `isometry_psd_majorizer_*` tests, and the
// `refresh_isometry_caches_pairs_each_penalty_to_its_own_atom` regression) was
// split into the sibling `tests_isometry_exact_hvp_majorizer_457.rs` module
// (declared in `mod.rs`) to keep this tracked file under the 10k limit. The
// cluster is self-contained: its helpers are referenced only within it.

/// Build a minimal single-atom periodic SAE outer objective for the
/// warm-start contract tests (gam#577 / gam#579).
pub(crate) fn warmstart_test_objective() -> SaeManifoldOuterObjective {
    // `PeriodicHarmonicEvaluator::new(3)` produces the SAME 3-column Fourier
    // basis `[1, sin(2πt), cos(2πt)]` and first jet as `periodic_basis`, plus
    // the analytic second jet that `logdet_theta_adjoint` (the softmax
    // assignment adjoint) needs. Installing it lets the full `eval` gradient
    // lane run instead of erroring on a missing second-jet evaluator.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = array![[0.10], [0.35], [0.62], [0.88]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        // Decoder mapping the 3 basis fns to a single output channel.
        array![[0.30], [-0.20], [0.15]],
        // Mild ridge-like smoothness penalty so the inner solve is PD.
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone())
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        // Nonzero assignment mass so H_tt carries genuine data curvature.
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

/// As [`warmstart_test_objective`], but the atom carries a full basis evaluator
/// AND second-jet evaluator (`PeriodicHarmonicEvaluator`), so the analytic outer
/// ρ-gradient lane (`eval` → `logdet_theta_adjoint`, which needs second jets for
/// the softmax assignment adjoint) can run. Required by the #1206 gradient-lane
/// contract test, which exercises the full `(cost, ∇f)` path.
pub(crate) fn warmstart_test_objective_with_evaluator() -> SaeManifoldOuterObjective {
    // `PeriodicHarmonicEvaluator::new(3)` produces the SAME 3-column Fourier
    // basis `[1, sin(2πt), cos(2πt)]` (1 harmonic) and matching first jet that
    // `periodic_basis` builds, so phi/jet are consistent with the decoder dims.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = array![[0.10_f64], [0.35], [0.62], [0.88]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.30_f64], [-0.20], [0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone())
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

pub(crate) fn diagonal_latent_cache(diagonal: &[f64]) -> ArrowFactorCache {
    let dim = diagonal.len();
    let mut factor = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        factor[[i, i]] = diagonal[i].sqrt();
    }
    ArrowFactorCache {
        htt_factors: ArrowFactorSlab::from_blocks(vec![factor]),
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: None,
        schur_factor_is_undamped: true,
        beta_schur_conditioning: None,
        joint_hessian_log_det: None,
        solver_mode: ArrowSolverMode::Direct,
        ridge_t: 0.0,
        ridge_beta: 0.0,
        htbeta: ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
        d: dim,
        row_dims: Arc::from(vec![dim].into_boxed_slice()),
        row_offsets: Arc::from(vec![0usize, dim].into_boxed_slice()),
        k: 0,
        manifold_mode_fingerprint: 0,
        row_hessian_fingerprint: 0,
        pcg_diagnostics: ArrowPcgDiagnostics::default(),
        gauge_deflated_directions: 0,
        deflated_row_directions: std::sync::Arc::from(Vec::new()),
        deflation_row_spectra: std::sync::Arc::from(Vec::new()),
        beta_gauge_quotient: None,
    }
}




/// #1436 — the analytic derivative error taxonomy must keep internal-invariant
/// failures distinct from genuine conditioning/non-identifiability. Every class
/// propagates if the projected solve cannot produce a reliable derivative, but
/// the diagnostic must remain machine-distinguishable.
#[test]
pub(crate) fn outer_gradient_internal_invariant_is_typed_1436() {
    let non_identifiable = OuterGradientError::NonIdentifiable {
        reason: "gauge-degenerate direction".to_string(),
    };
    let internal = OuterGradientError::InternalInvariant {
        reason: "shape mismatch".to_string(),
    };
    assert!(non_identifiable.to_string().contains("non-identifiable"));
    assert!(
        internal.to_string().contains("internal invariant"),
        "InternalInvariant Display must name the class; got: {}",
        internal
    );
}

/// #2653 — an exact implicit-gradient solve can be undefined at one otherwise
/// finite line-search rho while remaining well-defined at the incumbent. Keep
/// conditioning/non-identifiability rho-local so BFGS contracts the trial;
/// invariant violations still invalidate the whole optimization.
#[test]
pub(crate) fn outer_gradient_failure_preserves_rho_locality_2653() {
    let error = EstimationError::from(OuterGradientError::NonIdentifiable {
        reason: "exact stationarity operator remains singular".to_string(),
    });
    assert!(
        matches!(error, EstimationError::TrialPointRefused { .. })
            && error.is_trial_point_infeasible(),
        "conditioning at one rho must reject only that trial: {error}"
    );

    let invariant = EstimationError::from(OuterGradientError::InternalInvariant {
        reason: "gradient length differs from rho layout".to_string(),
    });
    assert!(
        matches!(invariant, EstimationError::RemlOptimizationFailed(_))
            && !invariant.is_trial_point_infeasible(),
        "an invariant violation must remain fatal: {invariant}"
    );
}

/// An empty β means that cache or typed reactive entry has no coefficient
/// state to install. The hook must preserve the objective-owned state and
/// report `NoSlot`, rather than interpreting zero length as a decoder mismatch.
#[test]
pub(crate) fn seed_inner_state_accepts_empty_beta_as_noslot() {
    let mut obj = warmstart_test_objective();
    let empty: Array1<f64> = Array1::zeros(0);
    let outcome = obj
        .seed_inner_state(&empty)
        .expect("empty-β seed must be accepted as a no-op, not rejected (gam#577/#579)");
    assert!(
        matches!(outcome, SeedOutcome::NoSlot),
        "empty-β seed must report NoSlot (proceed cold); got {outcome:?}"
    );
}

/// A populated β whose length matches the decoder dimension must be
/// INSTALLED and then GENUINELY REUSED by the next inner solve — this is
/// the exact-seed cache and typed reactive waypoint contract rely on. We verify
/// reuse behaviorally with a β produced by a converged evidence solve, then run
/// one eval with zero inner Newton iterations and confirm the published
/// `inner_beta_hint` is exactly that seed. An arbitrary off-optimum β can have no
/// defined frozen quasi-Laplace score and is not a valid ownership witness.
#[test]
pub(crate) fn seed_inner_state_installs_and_reuses_matching_beta() {
    let mut source = warmstart_test_objective();
    let source_rho = source.baseline_rho.clone();
    source
        .term
        .penalized_quasi_laplace_criterion_with_cache(
            source.target.view(),
            &source_rho,
            source.registry.as_ref(),
            source.inner_max_iter,
            source.learning_rate,
            source.ridge_ext_coord,
            source.ridge_beta,
        )
        .expect("source continuation state must have finite converged evidence");
    let seed = source.term.flatten_beta();

    let mut obj = warmstart_test_objective();
    let dim = obj.term.beta_dim();
    let pristine = obj.term.flatten_beta();
    assert_eq!(seed.len(), dim, "source β must match the target layout");
    assert!(
        (&seed - &pristine).iter().any(|d| d.abs() > 1e-6),
        "converged continuation β must differ from the pristine target β for the reuse check"
    );

    let outcome = obj
        .seed_inner_state(&seed)
        .expect("a length-matching β must install");
    assert!(
        matches!(outcome, SeedOutcome::Installed),
        "matching β must report Installed; got {outcome:?}"
    );

    // Freeze the inner solve at zero Newton iterations: β cannot move off
    // the warm-start, so the published hint must equal the seed exactly.
    obj.inner_max_iter = 0;
    let rho_flat = obj.baseline_rho.flat_coordinates();
    let eval =
        OuterObjective::eval(&mut obj, &rho_flat).expect("eval at the warm-started β must succeed");
    let hint = eval
        .inner_beta_hint
        .expect("the SAE objective must publish inner_beta_hint for continuation reuse");
    assert_eq!(
        hint.len(),
        dim,
        "published hint must have decoder dimension"
    );
    for (i, (&h, &s)) in hint.iter().zip(seed.iter()).enumerate() {
        assert!(
            (h - s).abs() < 1e-12,
            "warm-started β must be reused verbatim by the inner solve at coord {i}: \
                 hint {h} != seed {s} (gam#577/#579)"
        );
    }
}

/// A populated β whose length disagrees with the decoder dimension is the
/// trait's typed `SeedOutcome::Incompatible` case, not a fatal error. Cache
/// replay is the one production caller, and `reduce_basis_to_subspace` (#1117)
/// changes an atom's basis width in place, so a banked β can legitimately be
/// laid out for another width. The β must be declined without installing
/// anything, so the fit resumes ρ-only instead of failing (#2234: "β length 4
/// != decoder dim 6").
#[test]
pub(crate) fn seed_inner_state_rejects_wrong_length_populated_beta() {
    let mut obj = warmstart_test_objective();
    let dim = obj.term.beta_dim();
    let before = obj.term.flatten_beta();
    let wrong: Array1<f64> = Array1::zeros(dim + 1);
    let outcome = obj
        .seed_inner_state(&wrong)
        .expect("a populated β of the wrong length is declined, not a fatal error");
    assert!(
        matches!(outcome, SeedOutcome::Incompatible),
        "a wrong-length β must report Incompatible (ρ-only resume); got {outcome:?}"
    );
    assert!(
        obj.seeded_beta.is_none(),
        "a declined β must not become the pending seed"
    );
    let after = obj.term.flatten_beta();
    assert!(
        after.len() == before.len()
            && after
                .iter()
                .zip(before.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a declined β must leave the objective's coefficients bit-identical"
    );
}

pub(crate) fn gamma_fd_tiny_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let k_atoms = 2usize;
    let m = 3usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let weights = [
        [
            [0.10, -0.05, 0.03],
            [0.35, -0.20, 0.12],
            [-0.16, 0.18, 0.08],
        ],
        [
            [-0.08, 0.04, 0.06],
            [0.22, 0.10, -0.18],
            [0.11, -0.24, 0.15],
        ],
    ];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = (row as f64 + 0.35) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.21).fract();
        logits[[row, 0]] = if row % 2 == 0 { 0.8 } else { -0.6 };
        let assignments = softmax_row(logits.row(row), 0.9);
        for atom in 0..k_atoms {
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        assignments[atom] * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator.evaluate(coords[atom].view()).unwrap();
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("gamma_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::softmax(0.9),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(
        -6.0,
        -6.0,
        vec![Array1::from_vec(vec![-6.0]), Array1::from_vec(vec![-6.0])],
    );
    (term, target, rho)
}

// [#780 line-count gate] The #1557 arrow-Schur parallelism-invariance
// regression test (`arrow_schur_assembly_is_faer_parallelism_invariant_1557`)
// was split into the sibling `tests_parallelism_invariance_1557.rs` module
// (declared in `mod.rs`) to keep this tracked file under the 10k limit.
//
// The stationary-cache `∂log|H|/∂θ` adjoint and assignment-prior trace
// regressions were likewise split into the sibling
// `tests_logdet_adjoint_780.rs` module for the same gate; they still source the
// shared `gamma_fd_tiny_fixture` / `fixed_state_logdet_sample` helpers, which remain
// defined here.

/// Discrete branch identity for a finite-difference endpoint.
///
/// A finite-difference endpoint is a different cache by definition: its numeric
/// Hessian and eigenvalues should change. The quotient is a valid derivative
/// oracle only when the discrete classifier decisions stay fixed.
///
/// Every field below is authoritative classifier output or structural layout,
/// except `unresolved_eigengap`: whether a recorded row spectrum has two
/// eigenvalues closer than eigensolver round-off. Derivatives through individual
/// eigenpairs are undefined at such a degeneracy, so neither the center nor an
/// endpoint may carry one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FiniteDifferenceStratumCertificate {
    row_dims: Vec<usize>,
    row_offsets: Vec<usize>,
    beta_dim: usize,
    manifold_mode_fingerprint: u64,
    solver_mode: ArrowSolverMode,
    gauge_deflated_directions: usize,
    deflated_per_row: Vec<usize>,
    row_spectral_conditioning: Vec<Option<Vec<RowSpectralConditioning>>>,
    beta_schur_deflated: Option<Vec<bool>>,
    beta_gauge_rank: usize,
    unresolved_eigengap: bool,
    min_row_pivot_branch: PivotBranch,
    min_schur_pivot_branch: PivotBranch,
    min_pivot_branch: PivotBranch,
    max_pivot_branch: PivotBranch,
}

impl FiniteDifferenceStratumCertificate {
    pub(crate) fn from_arrow_cache(cache: &ArrowFactorCache) -> Self {
        let min_pivot = arrow_factor_min_pivot(cache);
        Self {
            row_dims: cache.row_dims.to_vec(),
            row_offsets: cache.row_offsets.to_vec(),
            beta_dim: cache.k,
            manifold_mode_fingerprint: cache.manifold_mode_fingerprint,
            solver_mode: cache.solver_mode,
            gauge_deflated_directions: cache.gauge_deflated_directions,
            deflated_per_row: cache.deflated_row_directions.iter().map(Vec::len).collect(),
            row_spectral_conditioning: cache
                .deflation_row_spectra
                .iter()
                .map(|spectrum| {
                    spectrum
                        .as_ref()
                        .map(|spectrum| spectrum.conditioning.to_vec())
                })
                .collect(),
            beta_schur_deflated: cache
                .beta_schur_conditioning
                .as_ref()
                .map(|spectrum| {
                    spectrum
                        .conditioning
                        .iter()
                        .map(|state| *state == BetaSchurSpectralConditioning::UnitDeflated)
                        .collect()
                }),
            beta_gauge_rank: cache
                .beta_gauge_quotient
                .as_ref()
                .map_or(0, |quotient| quotient.directions.len()),
            unresolved_eigengap: row_spectra_have_unresolved_eigengap(cache),
            min_row_pivot_branch: classify_fd_pivot(min_pivot.min_row_pivot),
            min_schur_pivot_branch: classify_fd_pivot(min_pivot.min_schur_pivot),
            min_pivot_branch: classify_fd_pivot(min_pivot.min_pivot),
            max_pivot_branch: classify_fd_pivot(arrow_factor_max_pivot(cache)),
        }
    }

    pub(crate) fn changed_fields(&self, endpoint: &Self) -> Vec<&'static str> {
        let mut changed = Vec::new();
        if self.row_dims != endpoint.row_dims {
            changed.push("row_dims");
        }
        if self.row_offsets != endpoint.row_offsets {
            changed.push("row_offsets");
        }
        if self.beta_dim != endpoint.beta_dim {
            changed.push("beta_dim");
        }
        if self.manifold_mode_fingerprint != endpoint.manifold_mode_fingerprint {
            changed.push("manifold_mode");
        }
        if self.solver_mode != endpoint.solver_mode {
            changed.push("solver_mode");
        }
        if self.gauge_deflated_directions != endpoint.gauge_deflated_directions {
            changed.push("gauge_deflated_directions");
        }
        if self.deflated_per_row != endpoint.deflated_per_row {
            changed.push("deflated_per_row");
        }
        if self.row_spectral_conditioning != endpoint.row_spectral_conditioning {
            changed.push("row_spectral_conditioning");
        }
        if self.beta_schur_deflated != endpoint.beta_schur_deflated {
            changed.push("beta_schur_deflated");
        }
        if self.beta_gauge_rank != endpoint.beta_gauge_rank {
            changed.push("beta_gauge_rank");
        }
        if self.unresolved_eigengap != endpoint.unresolved_eigengap {
            changed.push("unresolved_eigengap");
        }
        if self.min_row_pivot_branch != endpoint.min_row_pivot_branch {
            changed.push("min_row_pivot_branch");
        }
        if self.min_schur_pivot_branch != endpoint.min_schur_pivot_branch {
            changed.push("min_schur_pivot_branch");
        }
        if self.min_pivot_branch != endpoint.min_pivot_branch {
            changed.push("min_pivot_branch");
        }
        if self.max_pivot_branch != endpoint.max_pivot_branch {
            changed.push("max_pivot_branch");
        }
        changed
    }

    pub(crate) fn assert_same_stratum(&self, label: &str, endpoint: &Self) {
        assert!(
            !self.unresolved_eigengap,
            "{label}: finite-difference center has an unresolved spectral invariant-subspace block"
        );
        assert!(
            !endpoint.unresolved_eigengap,
            "{label}: finite-difference endpoint has an unresolved spectral invariant-subspace block"
        );
        let changed = self.changed_fields(endpoint);
        assert!(
            changed.is_empty(),
            "{label}: finite-difference endpoint crossed a nondifferentiable structural stratum; \
             changed_fields={changed:?}\ncenter={self:#?}\nendpoint={endpoint:#?}"
        );
    }
}

/// Whether any recorded row deflation spectrum has two finite eigenvalues closer
/// than eigensolver round-off, `eigen_gap_threshold(max |λ|, n)` over the widest
/// row and the largest finite magnitude.
fn row_spectra_have_unresolved_eigengap(cache: &ArrowFactorCache) -> bool {
    let mut min_gap = f64::INFINITY;
    let mut max_scale = 0.0_f64;
    let mut max_count = 0_usize;
    for spectrum in cache.deflation_row_spectra.iter().flatten() {
        let mut finite: Vec<f64> = spectrum
            .raw_evals
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        finite.sort_by(|left, right| left.total_cmp(right));
        for pair in finite.windows(2) {
            min_gap = min_gap.min((pair[1] - pair[0]).abs());
        }
        max_scale = finite.iter().fold(max_scale, |acc, value| acc.max(value.abs()));
        max_count = max_count.max(spectrum.raw_evals.len());
    }
    min_gap.is_finite() && min_gap < eigen_gap_threshold(max_scale, max_count)
}

/// Sign branch of one factor pivot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PivotBranch {
    Missing,
    Positive,
    NonPositive,
    NonFinite,
}

fn classify_fd_pivot(pivot: Option<f64>) -> PivotBranch {
    match pivot {
        None => PivotBranch::Missing,
        Some(value) if !value.is_finite() => PivotBranch::NonFinite,
        Some(value) if value > 0.0 => PivotBranch::Positive,
        Some(_) => PivotBranch::NonPositive,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FixedStateLogdetSample {
    pub(crate) value: f64,
    pub(crate) stratum: FiniteDifferenceStratumCertificate,
}

pub(crate) fn fixed_state_logdet_sample(
    mut term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> FixedStateLogdetSample {
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("fixed-state cache");
    let value = cache
        .arrow_log_det()
        .expect("fixed-state authoritative joint logdet");
    let stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    FixedStateLogdetSample { value, stratum }
}

pub(crate) fn certified_central_logdet_difference(
    label: &str,
    center: &FiniteDifferenceStratumCertificate,
    plus: FixedStateLogdetSample,
    minus: FixedStateLogdetSample,
    step: f64,
) -> f64 {
    assert!(
        step.is_finite() && step > 0.0,
        "{label}: central-difference step must be finite and positive, got {step}"
    );
    center.assert_same_stratum(&format!("{label} (+h)"), &plus.stratum);
    center.assert_same_stratum(&format!("{label} (-h)"), &minus.stratum);
    (plus.value - minus.value) / (2.0 * step)
}

/// What the value-free branch guard could establish about a stencil.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FdBranchRegime {
    /// Consecutive Richardson gaps fall like `h²`, so the SAMPLED function is
    /// smooth across the stencil. Carries the measured gap ratio.
    Smooth { ratio: f64 },
    /// The gaps have fallen to the roundoff floor, where their ratio measures
    /// rounding rather than the branch. The guard is INAPPLICABLE at this step
    /// and reports instead of certifying.
    RoundoffDominated { coarse_gap: f64, floor: f64 },
}

/// Value-free branch guard for a central difference (#2366).
///
/// # What it adds over the structural stratum certificate
///
/// [`FiniteDifferenceStratumCertificate`] pins every discrete decision the
/// cache CLASSIFIER makes. That is the right guard for classifier-visible
/// discreteness, and it is all these gates need, because they freeze `θ̂`
/// (`inner_max_iter = 0`) at every stencil point and so cannot land on a
/// different inner mode at `±h`. It is blind, however, to any nonsmoothness the
/// classifier does not label — including a branch that leaves and returns
/// between sample points.
///
/// This guard is blind to nothing, because it reads only the samples. For a
/// `C⁴` target, `CD(h) = f′ + f‴h²/6 + O(h⁴)`, so
///
/// ```text
///   CD(h)   − CD(h/2) = (3/8)·f‴h²  + O(h⁴)
///   CD(h/2) − CD(h/4) = (3/32)·f‴h² + O(h⁴)
/// ```
///
/// and the ratio of consecutive gaps is `1/4` with `f′` cancelling out — so
/// this tests the sampled function, never the claim under test. A jump inside
/// the stencil leaves the gaps `O(1)` (ratio ≈ 1); a kink leaves them `O(h)`
/// (ratio ≈ 1/2). The `0.35` bound matches the sibling predicate landed for
/// `#2354/#2366`, so the fleet asserts one constant.
///
/// # Applicability
///
/// The identity above holds while TRUNCATION dominates. Once the gaps reach the
/// roundoff floor `≈ ε·|f|/h_fine` the ratio is noise, and asserting it would
/// manufacture failures out of rounding. The guard therefore refuses to
/// conclude below that floor and says so, rather than certifying a stencil it
/// cannot see.
pub(crate) fn certified_branch_stable_central_difference(
    label: &str,
    center: &FiniteDifferenceStratumCertificate,
    step: f64,
    sample: impl Fn(f64) -> FixedStateLogdetSample,
) -> (f64, FdBranchRegime) {
    assert!(
        step.is_finite() && step > 0.0,
        "{label}: central-difference step must be finite and positive, got {step}"
    );
    let mut magnitude = 0.0_f64;
    let mut quotients = [0.0_f64; 3];
    for (index, scale) in [1.0_f64, 0.5, 0.25].into_iter().enumerate() {
        let h = step * scale;
        let plus = sample(h);
        let minus = sample(-h);
        center.assert_same_stratum(&format!("{label} (+{scale}h)"), &plus.stratum);
        center.assert_same_stratum(&format!("{label} (-{scale}h)"), &minus.stratum);
        magnitude = magnitude.max(plus.value.abs()).max(minus.value.abs());
        quotients[index] = (plus.value - minus.value) / (2.0 * h);
    }
    let coarse_gap = (quotients[0] - quotients[1]).abs();
    let fine_gap = (quotients[1] - quotients[2]).abs();
    // Roundoff floor of the FINEST quotient, which bounds both gaps from below.
    let floor = f64::EPSILON * magnitude / (0.25 * step);
    // A hundredfold margin puts the coarse gap unambiguously in the truncation
    // regime before its ratio is read as evidence about the branch.
    if coarse_gap <= 100.0 * floor {
        return (
            quotients[0],
            FdBranchRegime::RoundoffDominated { coarse_gap, floor },
        );
    }
    let ratio = fine_gap / coarse_gap;
    assert!(
        ratio <= 0.35,
        "{label}: central-difference gaps must fall as h² across a branch-stable \
         stencil; coarse(h={step:.3e}) {coarse_gap:.6e}, fine(h/2) {fine_gap:.6e}, \
         ratio {ratio:.4} (predicted 0.25; O(h) kink gives 0.5, O(1) jump gives 1). \
         The stratum certificate passed, so this is nonsmoothness the classifier \
         does not label."
    );
    (quotients[0], FdBranchRegime::Smooth { ratio })
}

/// Row-deflation regime a finite-difference anchor must PROVE.
///
/// The deflation state is not a detail of the fixture: it decides which
/// analytic object even exists at the point. A test whose whole subject is the
/// deflated Daleckii–Krein correction is vacuous on an undeflated cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FdAnchorDeflation {
    /// At least one row must deflate.
    SomeRowDeflates,
    /// Deflation does not change the object under test.
    Unconstrained,
}

/// The regime a finite-difference evaluation point must certify before it may
/// be used as a derivative-verification anchor.
///
/// Every field is a property of the STATE, never of the comparison the test
/// goes on to make. A search that accepted candidates by how well the analytic
/// value matched its finite difference would be fitting the oracle to the
/// answer; a search that accepts by these predicates only is choosing a point
/// at which the asserted derivative is defined.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FdAnchorRegime {
    pub(crate) deflation: FdAnchorDeflation,
}

impl FdAnchorRegime {
    /// The weakest regime: the frozen state must merely BE a maximum the
    /// criterion will price. Everything the gate then differentiates is defined
    /// there; nothing further is asserted about the state.
    pub(crate) fn any_maximum() -> Self {
        Self {
            deflation: FdAnchorDeflation::Unconstrained,
        }
    }

    /// The regime a deflated-correction test needs: the Daleckii–Krein path
    /// must actually fire, or the gate is vacuous.
    pub(crate) fn deflated() -> Self {
        Self {
            deflation: FdAnchorDeflation::SomeRowDeflates,
        }
    }
}

/// A frozen-θ̂ evaluation state whose regime has been PROVED rather than hoped
/// for, together with the structural stratum its finite differences must stay
/// inside.
pub(crate) struct CertifiedFdAnchor {
    pub(crate) term: SaeManifoldTerm,
    pub(crate) rho: SaeManifoldRho,
    pub(crate) cache: ArrowFactorCache,
    pub(crate) stratum: FiniteDifferenceStratumCertificate,
}

/// One declared member of an anchor family.
pub(crate) struct FdAnchorCandidate {
    /// What this member is, in the family's own terms. Reported on acceptance
    /// and on rejection.
    pub(crate) description: String,
    pub(crate) rho: SaeManifoldRho,
    pub(crate) term: SaeManifoldTerm,
    /// Inner-solve budget spent BEFORE the state is frozen. Zero anchors the
    /// candidate exactly where the caller put it; a positive budget declares
    /// that the candidate is "wherever the inner solve converges from here",
    /// and a solve that refuses is a rejection of this member, not a panic.
    pub(crate) converge_iters: usize,
    /// Inner-solve gradient/objective tolerance used by that budget. The
    /// frozen build below never solves, so this governs only how tightly the
    /// candidate's own mode is reached.
    pub(crate) converge_tolerance: f64,
}

/// Reject-or-accept one candidate state against a regime.
fn classify_fd_anchor_candidate(
    target: &Array2<f64>,
    regime: FdAnchorRegime,
    candidate: FdAnchorCandidate,
) -> Result<CertifiedFdAnchor, String> {
    let FdAnchorCandidate {
        rho,
        mut term,
        converge_iters,
        converge_tolerance,
        ..
    } = candidate;
    if converge_iters > 0 {
        term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            converge_iters,
            0.4,
            converge_tolerance,
            converge_tolerance,
        )
        .map_err(|error| format!("inner solve refused to converge: {error}"))?;
    }
    // `inner_max_iter = 0` freezes θ̂ at the candidate state: the anchor is the
    // point the test declares, not whatever the inner solve would wander to.
    let (value, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .map_err(|error| format!("criterion refused the frozen state: {error}"))?;
    if !(value.is_finite() && loss.total().is_finite()) {
        return Err(format!(
            "frozen state priced non-finitely (value={value}, loss={})",
            loss.total()
        ));
    }
    let deflated_rows = cache
        .deflated_row_directions
        .iter()
        .filter(|directions| !directions.is_empty())
        .count();
    if regime.deflation == FdAnchorDeflation::SomeRowDeflates && deflated_rows == 0 {
        return Err("no row deflates; regime requires at least one".to_string());
    }
    let stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    // Every stencil through the anchor asserts its center carries no unresolved row
    // invariant-subspace block (`assert_same_stratum`), so a center that carries one
    // is not a point where the asserted derivative is defined. That is a property of
    // the state, so it belongs in the certificate, not in the first sample.
    if stratum.unresolved_eigengap {
        return Err(
            "a row deflation spectrum has an unresolved invariant-subspace block, where \
             the frozen log-determinant's derivative is not defined"
                .to_string(),
        );
    }
    Ok(CertifiedFdAnchor {
        term,
        rho,
        cache,
        stratum,
    })
}

/// Accept the FIRST member of a declared, ordered, finite candidate family
/// whose frozen state certifies `regime`.
///
/// # Why a family and not a constant
///
/// The states that exercise the interesting θ-adjoint paths sit next to the
/// boundaries that make those paths interesting: the row-deflation floor and
/// the exact observed information's positive-definite face. A hand-written
/// constant that lands between them is correct only for the production code it
/// was measured against; the same constant is a refusal — not a weaker test, an
/// ABSENT one — as soon as the boundaries move.
///
/// A declared family plus a state predicate is the stable form of the same
/// intent. The family is ordered by how strongly it expresses the test's
/// purpose (most decisive first), the predicate is the regime the asserted
/// derivative needs to exist, and the accepted member is reported. Nothing in
/// the predicate can see the finite difference or the analytic value, so this
/// cannot converge on "whatever agrees".
pub(crate) fn certified_fd_anchor(
    label: &str,
    target: &Array2<f64>,
    regime: FdAnchorRegime,
    candidates: Vec<FdAnchorCandidate>,
) -> CertifiedFdAnchor {
    assert!(
        !candidates.is_empty(),
        "{label}: an anchor family must declare at least one candidate"
    );
    let mut rejections = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let description = candidate.description.clone();
        match classify_fd_anchor_candidate(target, regime, candidate) {
            Ok(anchor) => {
                eprintln!("{label}: anchor certified at {description} ({regime:?})");
                return anchor;
            }
            Err(reason) => rejections.push(format!("  {description}: {reason}")),
        }
    }
    panic!(
        "{label}: no member of the declared anchor family certifies {regime:?}. \
         The regime the asserted derivative needs does not exist anywhere on this \
         family, so widening the family is a fixture decision and weakening the \
         regime would change what is proved. Rejections:\n{}",
        rejections.join("\n")
    );
}

/// Freeze one already-converged state across a declared ladder of evaluation
/// `ρ`, ordered by how strongly each member expresses the gate's intent.
///
/// An evaluation `ρ` is the other hand-written constant these gates carry: a
/// lift chosen to put the deflated legs above finite-difference noise while
/// keeping the frozen state a maximum. Both halves of that requirement are
/// properties of production, so the ladder is declared and the accepted member
/// certified.
pub(crate) fn rho_ladder_family(
    term: &SaeManifoldTerm,
    rhos: Vec<(String, SaeManifoldRho)>,
    converge_iters: usize,
) -> Vec<FdAnchorCandidate> {
    rho_ladder_family_with_tolerance(
        term,
        rhos,
        converge_iters,
        DEFAULT_ANCHOR_CONVERGE_TOLERANCE,
    )
}

/// [`rho_ladder_family`] for a gate whose declared mode is reached at a
/// tolerance other than the shared default.
pub(crate) fn rho_ladder_family_with_tolerance(
    term: &SaeManifoldTerm,
    rhos: Vec<(String, SaeManifoldRho)>,
    converge_iters: usize,
    converge_tolerance: f64,
) -> Vec<FdAnchorCandidate> {
    rhos.into_iter()
        .map(|(description, rho)| FdAnchorCandidate {
            description,
            rho,
            term: term.clone(),
            converge_iters,
            converge_tolerance,
        })
        .collect()
}

/// The inner-solve tolerance the θ-adjoint gates converge their declared mode
/// at. These gates differentiate a FROZEN state, so the mode only has to exist;
/// the shared value keeps every anchor family reaching one under the same
/// contract.
const DEFAULT_ANCHOR_CONVERGE_TOLERANCE: f64 = 1.0e-6;

/// A declared `log λ_sparse` lift ladder over one base `ρ`, ordered by lift.
///
/// The assignment-strength penalty is the dial these gates use to move a state
/// between the deflating and non-deflating regimes, so it is the natural
/// declared axis for a regime the gate needs but cannot control directly.
pub(crate) fn sparse_lift_ladder(
    base: &SaeManifoldRho,
    lifts: &[f64],
) -> Vec<(String, SaeManifoldRho)> {
    lifts
        .iter()
        .map(|&lift| {
            let mut rho = base.clone();
            rho.log_lambda_sparse = lift;
            (format!("log_lambda_sparse={lift:.2}"), rho)
        })
        .collect()
}
