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

use super::derivative_oracle::{
    BranchCertificate, EigenDerivativeRoute, MajorizerAnchorMode, PivotBranch,
};
use super::dual::DualKinkBranch;
use super::*;
use approx::assert_abs_diff_eq;
use gam_solve::arrow_schur::{
    ArrowFactorSlab, ArrowHtbetaCache, ArrowPcgDiagnostics, ArrowSolverMode, ArrowUndampedFactors,
};
use gam_solve::evidence::arrow_log_det_from_cache;
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
    // through `assignment_prior_grad_hdiag`'s ordered Beta--Bernoulli branch, #853). The
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

    // Without a refresh, the safe default is zero and the gradient is
    // all zeros. Confirm the precondition so the post-refresh contrast
    // is meaningful.
    let target_flat: Array1<f64> = coords.iter().copied().collect();
    let v0 = penalty.value(target_flat.view(), rho.view());
    assert_eq!(v0, IsometryPenalty::DEFAULT_VALUE_ON_MISSING_CACHE);
    let g0 = penalty.grad_target(target_flat.view(), rho.view());
    assert!(
        g0.iter().all(|x| *x == 0.0),
        "grad_target without cache must be all zeros, got {g0:?}"
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
    let assignment = SaeAssignment::from_blocks_with_mode(
        // Nonzero assignment mass so H_tt carries genuine data curvature.
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
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
    let assignment = SaeAssignment::from_blocks_with_mode(
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

pub(crate) fn near_singular_outer_gradient_cache() -> ArrowFactorCache {
    ArrowFactorCache {
        htt_factors: ArrowFactorSlab::from_blocks(vec![array![[1.0_f64, 0.0], [0.0, 1.0e-7]]]),
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: Some(array![[1.0_f64]]),
        schur_factor_is_undamped: true,
        beta_schur_deflation: None,
        joint_hessian_log_det: None,
        solver_mode: ArrowSolverMode::Direct,
        ridge_t: 0.0,
        ridge_beta: 0.0,
        htbeta: ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
        d: 2,
        row_dims: Arc::from(vec![2usize].into_boxed_slice()),
        row_offsets: Arc::from(vec![0usize, 2usize].into_boxed_slice()),
        k: 1,
        manifold_mode_fingerprint: 0,
        row_hessian_fingerprint: 0,
        pcg_diagnostics: ArrowPcgDiagnostics::default(),
        gauge_deflated_directions: 0,
        deflated_row_directions: std::sync::Arc::from(Vec::new()),
        deflation_row_spectra: std::sync::Arc::from(Vec::new()),
        beta_gauge_quotient: None,
    }
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
        beta_schur_deflation: None,
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

#[test]
pub(crate) fn outer_gradient_solver_rejects_near_singular_cache_without_matching_gauge() {
    let cache = near_singular_outer_gradient_cache();
    let obj = warmstart_test_objective();

    // The raw conditioning gate is what names the ill-conditioned joint Hessian
    // and reports the pivot ratio + floor. Pin that message HERE, at its source
    // (`outer_gradient_conditioning_error`), so the diagnostic stays covered even
    // though the solver below now re-classifies the gauge-degenerate case.
    let conditioning_err = match SaeManifoldTerm::outer_gradient_conditioning_error(&cache) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("near-singular cache must trip the pivot-ratio conditioning gate"),
    };
    assert!(
        conditioning_err.contains("joint Hessian numerically singular"),
        "conditioning gate must name the ill-conditioned joint Hessian; got: {conditioning_err}"
    );
    assert!(
        conditioning_err.contains("min/max pivot ratio") && conditioning_err.contains("floor"),
        "conditioning gate must report the pivot ratio and floor; got: {conditioning_err}"
    );

    // #1436 (commit 21c49d14b): when the conditioning gate fires but NO chart
    // gauge / decoder-β-null / decoder-channel-null candidate can be recovered to
    // deflate the flat subspace, the flatness is genuinely OUTSIDE the gauge orbit
    // — a distinct, more specific diagnosis the solver surfaces as
    // `OuterGradientError::NonIdentifiable` (rather than echoing the raw
    // pivot-ratio `IllConditioned` trip). Both classes are FD-eligible, so the
    // recovery behaviour is unchanged; only the diagnostic is sharper. This is the
    // exact "without a matching gauge" path the test name describes.
    let err = match obj
        .term
        .outer_gradient_arrow_solver(&cache, &obj.current_rho.lambda_smooth_vec().unwrap())
    {
        Err(err) => err,
        Ok(..) => panic!("near-singular criterion factor without a matching gauge must reject"),
    };
    assert!(
        matches!(err, OuterGradientError::NonIdentifiable { .. }),
        "no-deflatable-direction rejection must be the NonIdentifiable diagnosis; got: {err}"
    );
    let err = err.to_string();
    assert!(
        err.contains("no deflatable gauge/decoder-null direction"),
        "guard error must name the absent deflation candidate; got: {err}"
    );
}

/// #1051: a euclidean-patch atom whose decoder design is RANK-DEFICIENT
/// (a straight line in a `p = 2` ambient: the decoder column space is rank
/// 1, so one output-channel direction is unidentified by the data) leaves a
/// genuine near-null direction of the joint Hessian in the β (decoder)
/// block. That direction is OUTSIDE the closed-form chart gauge orbit
/// (`dense_step_gauge_vectors` only spans per-latent-axis reparametrisation,
/// never per-output-channel decoder freedom), so before the fix
/// `outer_gradient_arrow_solver` could not deflate it and rejected the
/// trial ρ with "analytic outer gradient undefined" — the singular-pivot
/// continuation stall that made every euclidean/multi-atom atlas tile
/// TIMEOUT. With the β-basis admitted as a deflation candidate the flat
/// direction is Faddeev-Popov-deflated and the solve succeeds, regularising
/// the near-null β response to the Hessian scale (bounded, not 1e13).
pub(crate) fn rank_deficient_euclidean_outer_gradient_objective() -> SaeManifoldOuterObjective {
    // Linear euclidean basis Φ(t) = [1, t] (m = 2) over a 1-D latent.
    let coords = array![[-0.7_f64], [-0.2], [0.3], [0.8]];
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 2));
    let mut jet = Array3::<f64>::zeros((n, 2, 1));
    for row in 0..n {
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = coords[[row, 0]];
        jet[[row, 1, 0]] = 1.0; // d/dt of the linear column.
    }
    // p = 2 ambient, but the decoder maps only into output channel 0 (its
    // second column is identically zero), so the reconstruction `Φ·B` lives on
    // the 1-D subspace `{x : x₁ = 0}` of R² and output channel 1 is genuinely
    // unidentified. The decoder's right-singular null vector is then exactly the
    // channel-1 axis `(0, 1)`, matching the near-null direction the joint-Hessian
    // cache below places on that axis (β indices 1 and 3). This is the rank-1
    // decoder column-span deficiency `decoder_channel_null_directions` must
    // recover (#1051/#1273).
    let decoder = array![[1.0_f64, 0.0], [0.5, 0.0]];
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "euclidean_line",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(2),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode(
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[-1.0_f64, -2.0], [-0.3, -0.6], [0.4, 0.8], [1.1, 2.2]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

/// A joint Hessian cache whose β block carries one genuine near-null
/// direction along the SECOND output channel (`out_col = 1`) — the
/// rank-deficient decoder's unidentified direction — with the latent block
/// well-conditioned and `H_tβ = 0` so the singularity is purely in β. The
/// chart gauge orbit cannot reach this direction (#1051).
pub(crate) fn rank_deficient_beta_outer_gradient_cache() -> ArrowFactorCache {
    // The latent block must be dimensionally consistent with the paired
    // objective `rank_deficient_euclidean_outer_gradient_objective` so the
    // channel-null candidates (whose full length is the objective's
    // `n·q + β_dim`) survive the `dir.len() == full_len` guard in
    // `outer_gradient_arrow_solver`. That objective has n = 4 data rows and
    // `row_block_dim q = 1` (one latent axis, K = 1 softmax ⇒ no assignment
    // coord), so `delta_t_len` must be `n·q = 4`. A mismatched single-row cache
    // makes `full_len = 5` while the candidates have length 8, silently
    // dropping every channel-null direction and re-introducing the bug.
    let htt = ArrowFactorSlab::from_blocks(vec![
        array![[1.0_f64]],
        array![[1.0_f64]],
        array![[1.0_f64]],
        array![[1.0_f64]],
    ]);
    // β dim = m · p = 2 · 2 = 4, laid out (col, out_col) row-major like
    // `dense_step_gauge_vector_from_field`. Make output channel 1 (indices
    // 1 and 3) near-null: its lower-Cholesky pivot is 1e-7, so the
    // min/max pivot ratio falls below the 1e-12 floor and the conditioning
    // path engages. H_tβ = 0 (zero Dense blocks) decouples β from latent.
    let schur = array![
        [1.0_f64, 0.0, 0.0, 0.0],
        [0.0, 1.0e-7, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0e-7],
    ];
    ArrowFactorCache {
        htt_factors: htt,
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: Some(schur),
        schur_factor_is_undamped: true,
        beta_schur_deflation: None,
        joint_hessian_log_det: None,
        solver_mode: ArrowSolverMode::Direct,
        ridge_t: 0.0,
        ridge_beta: 0.0,
        htbeta: ArrowHtbetaCache::Dense {
            blocks: Arc::from(
                vec![
                    Array2::<f64>::zeros((1, 4)),
                    Array2::<f64>::zeros((1, 4)),
                    Array2::<f64>::zeros((1, 4)),
                    Array2::<f64>::zeros((1, 4)),
                ]
                .into_boxed_slice(),
            ),
            estimated_bytes: 0,
        },
        d: 4,
        row_dims: Arc::from(vec![1usize, 1usize, 1usize, 1usize].into_boxed_slice()),
        row_offsets: Arc::from(vec![0usize, 1usize, 2usize, 3usize, 4usize].into_boxed_slice()),
        k: 4,
        manifold_mode_fingerprint: 0,
        row_hessian_fingerprint: 0,
        pcg_diagnostics: ArrowPcgDiagnostics::default(),
        gauge_deflated_directions: 0,
        deflated_row_directions: std::sync::Arc::from(Vec::new()),
        deflation_row_spectra: std::sync::Arc::from(Vec::new()),
        beta_gauge_quotient: None,
    }
}

#[test]
pub(crate) fn outer_gradient_solver_deflates_rank_deficient_decoder_beta_null() {
    let obj = rank_deficient_euclidean_outer_gradient_objective();
    let cache = rank_deficient_beta_outer_gradient_cache();
    // Sanity: the cache genuinely trips the conditioning floor (the bug's
    // precondition) — without it this test would not exercise the fix.
    assert!(
        SaeManifoldTerm::outer_gradient_conditioning_error(&cache).is_err(),
        "fixture must be sub-floor singular so the conditioning path engages"
    );
    // The fix: the β-block near-null direction is admitted as a deflation
    // candidate and Faddeev-Popov-deflated, so the solver SUCCEEDS instead
    // of rejecting with "analytic outer gradient undefined".
    let solver = obj
        .term
        .outer_gradient_arrow_solver(&cache, &obj.current_rho.lambda_smooth_vec().unwrap())
        .expect("rank-deficient decoder β-null must be deflated, not rejected (#1051/#1273)");
    // The deflated solve must REGULARISE the near-null β response: a plain
    // inverse divides by the 1e-7 pivot and explodes; the deflated solve is
    // bounded at the Hessian scale.
    let beta_null_rhs = array![0.0_f64, 0.0, 0.0, 1.0]; // output channel 1, col 1.
    let rhs_t = Array1::<f64>::zeros(cache.delta_t_len());
    let plain = cache
        .full_inverse_apply(rhs_t.view(), beta_null_rhs.view())
        .expect("plain solve")
        .1;
    let deflated = solver
        .solve(rhs_t.view(), beta_null_rhs.view())
        .expect("deflated solve")
        .beta;
    assert!(
        plain[3].abs() > 1.0e13,
        "plain near-null β solve must explode; got {}",
        plain[3]
    );
    assert!(
        deflated.iter().all(|v| v.is_finite()) && deflated[3].abs() < 10.0,
        "deflated near-null β solve must be bounded at the Hessian scale; got {deflated:?}"
    );
}

/// #1436 — the analytic derivative error taxonomy must keep internal-invariant
/// failures distinct from genuine conditioning/non-identifiability. Every class
/// propagates if the projected solve cannot produce a reliable derivative, but
/// the diagnostic must remain machine-distinguishable.
#[test]
pub(crate) fn outer_gradient_internal_invariant_is_typed_1436() {
    let ill_conditioned = OuterGradientError::IllConditioned {
        reason: "near-singular joint Hessian".to_string(),
    };
    let non_identifiable = OuterGradientError::NonIdentifiable {
        reason: "gauge-degenerate direction".to_string(),
    };
    let internal = OuterGradientError::InternalInvariant {
        reason: "shape mismatch".to_string(),
    };
    assert!(ill_conditioned.to_string().contains("ill-conditioned"));
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
    for error in [
        OuterGradientError::IllConditioned {
            reason: "finite projected solve lost residual reduction".to_string(),
        },
        OuterGradientError::NonIdentifiable {
            reason: "gauge-deflated operator remains singular".to_string(),
        },
    ] {
        let error = EstimationError::from(error);
        assert!(
            matches!(error, EstimationError::TrialPointRefused { .. })
                && error.is_trial_point_infeasible(),
            "conditioning at one rho must reject only that trial: {error}"
        );
    }

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
    let rho_flat = obj.baseline_rho.to_flat();
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

/// The seed contract is only relaxed for the EMPTY sentinel. A populated
/// β whose length disagrees with the decoder dimension is a genuine
/// layout bug and must still surface a typed error rather than being
/// silently dropped.
#[test]
pub(crate) fn seed_inner_state_rejects_wrong_length_populated_beta() {
    let mut obj = warmstart_test_objective();
    let dim = obj.term.beta_dim();
    let wrong: Array1<f64> = Array1::zeros(dim + 1);
    let err = obj
        .seed_inner_state(&wrong)
        .expect_err("a populated β of the wrong length must be rejected");
    match err {
        EstimationError::RemlOptimizationFailed(msg) => {
            assert!(
                msg.contains("decoder dim"),
                "error must name the decoder-dim mismatch; got: {msg}"
            );
        }
        other => panic!("expected RemlOptimizationFailed, got {other:?}"),
    }
}

/// A supplied function-space Gram defines a fixed objective: changing the
/// decoder or current chart Jacobian cannot mutate it. The quadratic trace is
/// exactly the sum of the declared scalar-function seminorms over outputs.
#[test]
pub(crate) fn reference_function_gram_is_fixed_and_has_exact_trace_form() {
    let n = 4usize;
    let m = 3usize;
    let p = 2usize;
    let phi = Array2::<f64>::zeros((n, m));
    let jet = Array3::<f64>::zeros((n, m, 1));
    let decoder = Array2::from_shape_vec((m, p), vec![0.3, -0.2, 1.1, 0.4, -0.7, 0.9]).unwrap();
    let gram = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
        "fixed-reference",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder.clone(),
        gram.clone(),
    )
    .unwrap();
    let frozen = atom.smooth_penalty().clone();
    atom.decoder_coefficients_mut().mapv_inplace(|value| value * 7.0);
    atom.basis_jacobian.fill(13.0);
    assert_eq!(atom.smooth_penalty(), &frozen);
    assert_eq!(
        atom.reference_roughness_kind(),
        SaeReferenceRoughnessKind::ProvidedFunctionGram
    );

    let trace_form: f64 = (0..p)
        .map(|output| {
            let column = decoder.column(output);
            column.dot(&gram.dot(&column))
        })
        .sum();
    let bt_s_b = decoder.t().dot(&gram).dot(&decoder);
    let matrix_form = bt_s_b.diag().sum();
    assert_abs_diff_eq!(trace_form, matrix_form, epsilon = 1.0e-12);
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

/// Discrete branch identity for a finite-difference endpoint.
///
/// This is intentionally distinct from
/// [`super::derivative_oracle::BranchCertificate`]. That certificate proves
/// exact same-cache identity before analytic trace channels are combined, so
/// it correctly includes numeric Hessian fingerprints and exact eigengap
/// evidence. A finite-difference endpoint is a different cache by definition:
/// its numeric Hessian and eigenvalues should change. The quotient is a valid
/// derivative oracle only when the discrete classifier decisions stay fixed.
///
/// Every field below is authoritative classifier output or structural layout.
/// No threshold is re-derived from rounded factors and no continuous value is
/// mistaken for branch identity.
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
    eigen_derivative_route: EigenDerivativeRoute,
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
                .beta_schur_deflation
                .as_ref()
                .map(|spectrum| spectrum.deflated.to_vec()),
            beta_gauge_rank: cache
                .beta_gauge_quotient
                .as_ref()
                .map_or(0, |quotient| quotient.directions.len()),
            eigen_derivative_route: BranchCertificate::from_arrow_cache(
                cache,
                MajorizerAnchorMode::FrozenAnchor,
            )
            .eigen_derivative_route(),
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
        if self.eigen_derivative_route != endpoint.eigen_derivative_route {
            changed.push("eigen_derivative_route");
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
        assert_eq!(
            self.eigen_derivative_route,
            EigenDerivativeRoute::IndividualEigenpairs,
            "{label}: finite-difference center has an unresolved spectral invariant-subspace block"
        );
        assert_eq!(
            endpoint.eigen_derivative_route,
            EigenDerivativeRoute::IndividualEigenpairs,
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
    let value = arrow_log_det_from_cache(&cache).expect("fixed-state authoritative joint logdet");
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

fn floor_crossing_logdet_sample_2398(ratio: f64) -> FixedStateLogdetSample {
    assert!(ratio.is_finite() && ratio > 0.0);
    let relative_floor = gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR;
    let spectral_floor = relative_floor * 4.0;
    let mut system = ArrowSchurSystem::new(1, 3, 0);
    system.rows[0].htt = array![
        [4.0_f64, 0.0, 0.0],
        [0.0, ratio * spectral_floor, 0.0],
        [0.0, 0.0, -1.0],
    ];
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
    let options = ArrowSolveOptions::direct()
        .with_gpu_policy(gam_gpu::GpuPolicy::Off)
        .with_evidence_unit_deflation(relative_floor);
    // The ridge belongs only to the Newton factor, keeping the deliberately
    // indefinite fixture solvable. The independently assembled undamped
    // evidence factor still classifies the original spectrum above.
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 2.0, 0.0, &options)
        .expect("floor-crossing fixture must produce an evidence cache");
    let value =
        arrow_log_det_from_cache(&cache).expect("floor-crossing fixture must have a logdet");
    let stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
    FixedStateLogdetSample { value, stratum }
}

#[test]
fn certified_central_logdet_difference_refuses_floor_clamp_crossing_2398() {
    // These are the two sides of the production classifier's banded cutoff:
    // 0.995 is retained through the floor-clamped branch, while 1.05 is raw.
    // Their midpoint is also raw, so only the -h endpoint changes stratum.
    let minus_ratio = 0.995_f64;
    let plus_ratio = 1.05_f64;
    let center_ratio = (minus_ratio + plus_ratio) / 2.0;
    let step = (plus_ratio - minus_ratio) / 2.0;
    let center = floor_crossing_logdet_sample_2398(center_ratio);
    let plus = floor_crossing_logdet_sample_2398(plus_ratio);
    let minus = floor_crossing_logdet_sample_2398(minus_ratio);

    let expected_raw = vec![
        RowSpectralConditioning::UnitDeflated,
        RowSpectralConditioning::Raw,
        RowSpectralConditioning::Raw,
    ];
    let expected_clamped = vec![
        RowSpectralConditioning::UnitDeflated,
        RowSpectralConditioning::FloorClamped,
        RowSpectralConditioning::Raw,
    ];
    assert_eq!(
        center.stratum.row_spectral_conditioning,
        vec![Some(expected_raw.clone())],
    );
    assert_eq!(
        plus.stratum.row_spectral_conditioning,
        vec![Some(expected_raw)],
    );
    assert_eq!(
        minus.stratum.row_spectral_conditioning,
        vec![Some(expected_clamped)],
    );
    assert!(center.stratum.changed_fields(&plus.stratum).is_empty());
    assert_eq!(
        center.stratum.changed_fields(&minus.stratum),
        vec!["row_spectral_conditioning"],
    );

    let panic = std::panic::catch_unwind(move || {
        certified_central_logdet_difference(
            "spectral-floor stratum #2398",
            &center.stratum,
            plus,
            minus,
            step,
        )
    })
    .expect_err("a floor-clamp crossing must refuse before producing a quotient");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("refusal panic must carry a string message");
    assert!(
        message
            .contains("finite-difference endpoint crossed a nondifferentiable structural stratum")
    );
    assert!(message.contains("changed_fields=[\"row_spectral_conditioning\"]"));
    assert!(message.contains("spectral-floor stratum #2398 (-h)"));
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
/// cannot see. That distinction is the point: an inapplicable guard reported as
/// inapplicable is honest; an inapplicable guard asserted anyway is a gate that
/// fails for reasons unrelated to its subject.
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
/// deflated Daleckii–Krein correction is vacuous on an undeflated cache, and a
/// gate meant to isolate the PD regime is not testing that regime if a row
/// deflates under it. (#2712: the probe bundle DOES reconstruct the deflated
/// block — `undamped_factor` is the conditioned row Cholesky — so
/// `NoRowDeflates` is no longer a precondition of the from-probes routes, only
/// a statement about which regime a given gate means to pin.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FdAnchorDeflation {
    /// No row may deflate.
    NoRowDeflates,
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
    /// Require the production branch classifier to report one resolved smooth
    /// branch: no tied majorizer kink, no unresolved invariant-subspace block,
    /// and strictly positive Cholesky pivots. This is the classifier's OWN
    /// decision, not a numeric margin re-derived by the test.
    pub(crate) reportable_derivative_branch: bool,
}

impl FdAnchorRegime {
    /// The regime a `∂log|H|/∂θ` majorizer-channel finite difference needs: one
    /// resolved smooth branch, deflation irrelevant.
    pub(crate) fn smooth_majorizer_branch() -> Self {
        Self {
            deflation: FdAnchorDeflation::Unconstrained,
            reportable_derivative_branch: true,
        }
    }

    /// The PD regime: no row may deflate, so what the gate pins is the
    /// undeflated branch on its own. The deflated branch has its own gates
    /// (#2712) rather than being excluded here for want of a correction.
    pub(crate) fn undeflated() -> Self {
        Self {
            deflation: FdAnchorDeflation::NoRowDeflates,
            reportable_derivative_branch: false,
        }
    }

    /// The weakest regime: the frozen state must merely BE a maximum the
    /// criterion will price. Everything the gate then differentiates is defined
    /// there; nothing further is asserted about the state.
    pub(crate) fn any_maximum() -> Self {
        Self {
            deflation: FdAnchorDeflation::Unconstrained,
            reportable_derivative_branch: false,
        }
    }

    /// The regime a deflated-correction test needs: the Daleckii–Krein path
    /// must actually fire, or the gate is vacuous.
    pub(crate) fn deflated() -> Self {
        Self {
            deflation: FdAnchorDeflation::SomeRowDeflates,
            reportable_derivative_branch: false,
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
    /// Assignment-logit pattern this member homotopes toward AFTER its
    /// convergence budget, with its homotopy weight. `None` freezes the
    /// converged logits unchanged.
    pub(crate) decisive_mix: Option<(Array2<f64>, f64)>,
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
        decisive_mix,
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
    if let Some((decisive, weight)) = decisive_mix {
        if decisive.dim() != term.assignment.logits.dim() {
            return Err(format!(
                "decisive logit pattern {:?} does not match the assignment layout {:?}",
                decisive.dim(),
                term.assignment.logits.dim()
            ));
        }
        term.assignment.logits = &term.assignment.logits * (1.0 - weight) + &decisive * weight;
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
    match regime.deflation {
        FdAnchorDeflation::NoRowDeflates if deflated_rows > 0 => {
            return Err(format!(
                "{deflated_rows} row(s) deflate; regime requires none"
            ));
        }
        FdAnchorDeflation::SomeRowDeflates if deflated_rows == 0 => {
            return Err("no row deflates; regime requires at least one".to_string());
        }
        FdAnchorDeflation::NoRowDeflates
        | FdAnchorDeflation::SomeRowDeflates
        | FdAnchorDeflation::Unconstrained => {}
    }
    if regime.reportable_derivative_branch {
        let certificate =
            BranchCertificate::from_arrow_cache(&cache, MajorizerAnchorMode::FrozenAnchor);
        certificate
            .assert_derivative_reportable()
            .map_err(|error| format!("derivative branch is not reportable: {error}"))?;
        if let Some(record) = certificate
            .kink_branches
            .iter()
            .find(|record| record.branch == DualKinkBranch::Tie)
        {
            return Err(format!("majorizer kink branch is tied: {record:?}"));
        }
        for (name, branch) in [
            ("min row pivot", certificate.min_row_pivot_branch),
            ("min pivot", certificate.min_pivot_branch),
            ("max pivot", certificate.max_pivot_branch),
        ] {
            if branch != PivotBranch::Positive {
                return Err(format!("{name} branch is {branch:?}, not Positive"));
            }
        }
        if certificate.beta_dim > 0 && certificate.min_schur_pivot_branch != PivotBranch::Positive {
            return Err(format!(
                "min Schur pivot branch is {:?}, not Positive",
                certificate.min_schur_pivot_branch
            ));
        }
    }
    let stratum = FiniteDifferenceStratumCertificate::from_arrow_cache(&cache);
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
/// boundaries that make those paths interesting: the majorizer's `sign(H_kj)`
/// kink, the row-deflation floor, and the exact observed information's
/// positive-definite face. A hand-written constant that lands between them is
/// correct only for the production code it was measured against; the same
/// constant is a refusal — not a weaker test, an ABSENT one — as soon as the
/// boundaries move. Every such constant in this module had in fact become a
/// refusal.
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

/// The declared homotopy between a state that is positive definite BY
/// CONSTRUCTION and a state that is off the majorizer's sign kink BY
/// CONSTRUCTION.
///
/// `converged` is the inner mode the criterion just reached, so its exact
/// observed information is positive definite; but at a fitted optimum an
/// entropy-Hessian off-diagonal can sit exactly on the Gershgorin `|H_kj|`
/// sign flip, where no finite-difference stencil validates a subgradient.
/// `decisive` is a deterministic assignment pattern with margins far from that
/// flip, but far enough from the mode that the exact observed information can
/// leave its positive-definite face.
///
/// Neither endpoint is usable alone; the segment between them is where both
/// hold. Walking it from the decisive end takes the strongest off-kink state
/// that is still a maximum, which is exactly the anchor these gates always
/// meant to name.
pub(crate) fn decisive_logit_homotopy(
    converged: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    decisive: &Array2<f64>,
) -> Vec<FdAnchorCandidate> {
    let base = converged.assignment.logits.clone();
    assert_eq!(
        base.dim(),
        decisive.dim(),
        "decisive logit pattern must match the fixture's assignment layout"
    );
    DECISIVE_HOMOTOPY_WEIGHTS
        .into_iter()
        .map(|s| {
            let mut term = converged.clone();
            term.assignment.logits = &base * (1.0 - s) + decisive * s;
            FdAnchorCandidate {
                description: format!("decisive-logit homotopy s={s:.2}"),
                rho: rho.clone(),
                term,
                converge_iters: 0,
                converge_tolerance: DEFAULT_ANCHOR_CONVERGE_TOLERANCE,
                decisive_mix: None,
            }
        })
        .collect()
}

/// The declared homotopy ladder, walked from the fully decisive end toward the
/// inner mode. `s = 1` is the state these gates used to hard-code, so a run in
/// which the first member certifies reproduces the historical anchor exactly;
/// every later member is a strictly weaker departure from the mode. The ladder
/// stops at `s = 0.1` rather than `0`: at the mode itself the majorizer kink
/// the departure exists to avoid is back, so an anchor there would certify a
/// regime and still be the wrong point.
const DECISIVE_HOMOTOPY_WEIGHTS: [f64; 7] = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1];

/// Freeze one already-converged state across a declared ladder of evaluation
/// `ρ`, ordered by how strongly each member expresses the gate's intent.
///
/// An evaluation `ρ` is the other hand-written constant these gates carry: a
/// lift chosen to put the deflated legs above finite-difference noise while
/// keeping the frozen state a maximum. Both halves of that requirement are
/// properties of production, and the second half is exactly what fails when a
/// gate's `ρ` drifts onto an exact-`A` saddle. Declaring the ladder and
/// certifying the accepted member states the intent without pinning the
/// answer.
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
            decisive_mix: None,
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

/// The two-dimensional declared family for gates whose anchor must ALSO name a
/// smoothing level: the cross of a declared `log λ_smooth` ladder with the
/// decisive-logit homotopy, each member converging its own inner mode first.
///
/// A gate that hard-codes one `log λ_smooth` pair is asserting that a specific
/// smoothing level admits a converged mode with a resolved derivative branch.
/// That is a property of production, not of the gate, so it belongs in the
/// search space rather than in a constant.
pub(crate) fn smoothing_and_decisive_family(
    fixture: impl Fn() -> (SaeManifoldTerm, SaeManifoldRho),
    smooth_levels: &[(f64, f64)],
    converge_iters: usize,
) -> Vec<FdAnchorCandidate> {
    let mut candidates = Vec::with_capacity(smooth_levels.len() * DECISIVE_HOMOTOPY_WEIGHTS.len());
    for &(first, second) in smooth_levels {
        for s in DECISIVE_HOMOTOPY_WEIGHTS {
            let (term, mut rho) = fixture();
            assert_eq!(
                rho.log_lambda_smooth.len(),
                2,
                "the declared smoothing ladder is written for two smooth blocks"
            );
            rho.log_lambda_smooth = vec![first, second];
            let decisive = decisive_logit_pattern(&term);
            candidates.push(FdAnchorCandidate {
                description: format!(
                    "log_lambda_smooth=[{first:.2}, {second:.2}], decisive-logit homotopy s={s:.2}"
                ),
                rho,
                term,
                converge_iters,
                converge_tolerance: DEFAULT_ANCHOR_CONVERGE_TOLERANCE,
                decisive_mix: Some((decisive, s)),
            });
        }
    }
    candidates
}

/// The deterministic decisive-assignment pattern the θ-adjoint gates drive
/// toward: alternating row-varying margins around a slow drift, so no row is
/// symmetric and no two rows share a margin.
pub(crate) fn decisive_logit_pattern(term: &SaeManifoldTerm) -> Array2<f64> {
    let mut logits = term.assignment.logits.clone();
    for row in 0..term.n_obs() {
        let center = 0.05 * (row as f64);
        let margin = 1.55 + 0.04 * (row as f64);
        let (first, second) = if row % 2 == 0 {
            (center + margin, center - margin)
        } else {
            (center - 0.85 * margin, center + 0.85 * margin)
        };
        logits[[row, 0]] = first;
        if logits.ncols() > 1 {
            logits[[row, 1]] = second;
        }
    }
    logits
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
