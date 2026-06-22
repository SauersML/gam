use std::sync::Arc;

use ndarray::{Array1, Array2};

use gam::solver::arrow_schur::ArrowFactorCache;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm,
};

struct Fixture {
    term: SaeManifoldTerm,
    target: Array2<f64>,
    rho: SaeManifoldRho,
}

/// Build the K=2 periodic-harmonic SAE fixture under a given assignment `mode`.
///
/// `log_lambda_sparse` is exposed so the IBP-MAP arm can run its empirical-π
/// prior at a genuinely active weight (the fixed-`alpha` IBP penalty reads
/// `lambda_sparse` as its weight lever), which is what exercises the #1006
/// empirical-`M_k` third channel through the outer-ρ gradient.
fn fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> Fixture {
    let n = 80usize;
    let p = 6usize;
    let k_atoms = 2usize;
    let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");

    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    let weights0 = [
        [0.20, -0.10, 0.06, 0.03, -0.04, 0.08],
        [0.70, -0.25, 0.40, 0.12, -0.35, 0.18],
        [0.15, 0.55, -0.25, 0.28, 0.08, -0.22],
        [0.08, -0.04, 0.03, -0.02, 0.01, 0.06],
        [-0.06, 0.02, 0.05, 0.04, -0.03, 0.01],
    ];
    let weights1 = [
        [-0.10, 0.05, 0.08, -0.02, 0.05, -0.03],
        [-0.30, 0.42, 0.12, -0.20, 0.16, 0.30],
        [0.48, 0.10, -0.32, 0.18, 0.26, -0.14],
        [0.04, 0.07, -0.02, 0.03, -0.05, 0.02],
        [0.03, -0.05, 0.04, 0.01, 0.02, -0.04],
    ];

    for row in 0..n {
        let phase = (row as f64 + 0.25) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.18).fract();
        let route = if row < n / 2 { 1.7 } else { -1.7 };
        logits[[row, 0]] = route;
        // A genuine second active gate so the IBP / JumpReLU per-atom logits all
        // sit inside their optimization bands (softmax ignores the absolute
        // level, gate modes do not).
        logits[[row, 1]] = if row % 3 == 0 { 0.9 } else { 0.3 };
        let theta0 = std::f64::consts::TAU * coords[0][[row, 0]];
        let theta1 = std::f64::consts::TAU * coords[1][[row, 0]];
        let basis0 = [
            1.0,
            theta0.sin(),
            theta0.cos(),
            (2.0 * theta0).sin(),
            (2.0 * theta0).cos(),
        ];
        let basis1 = [
            1.0,
            theta1.sin(),
            theta1.cos(),
            (2.0 * theta1).sin(),
            (2.0 * theta1).cos(),
        ];
        let mix0 = 1.0 / (1.0 + (-route / 0.7).exp());
        let mix1 = 1.0 - mix0;
        for col in 0..p {
            let mut v0 = 0.0;
            let mut v1 = 0.0;
            for b in 0..m {
                v0 += basis0[b] * weights0[b][col];
                v1 += basis1[b] * weights1[b][col];
            }
            target[[row, col]] = mix0 * v0 + mix1 * v1;
        }
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis evaluation");
        let decoder = if atom_idx == 0 {
            Array2::from_shape_fn((m, p), |(row, col)| weights0[row][col])
        } else {
            Array2::from_shape_fn((m, p), |(row, col)| weights1[row][col])
        };
        let mut smooth = Array2::<f64>::eye(m);
        smooth[[0, 0]] = 0.0;
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("circle atom")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator clone"),
        ) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -8.0,
        vec![Array1::from_vec(vec![-8.0]), Array1::from_vec(vec![-8.0])],
    );
    Fixture { term, target, rho }
}

fn evaluate(
    start: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    inner_max_iter: usize,
) -> (SaeManifoldTerm, f64, SaeManifoldLoss, ArrowFactorCache) {
    let mut term = start.clone();
    let (value, loss, cache) = term
        .reml_criterion_with_cache(
            target.view(),
            rho,
            None,
            inner_max_iter,
            0.45,
            1.0e-6,
            1.0e-6,
        )
        .unwrap_or_else(|err| panic!("REML criterion failed: {err}"));
    (term, value, loss, cache)
}

fn centered_fd(
    start: &SaeManifoldTerm,
    target: &Array2<f64>,
    template: &SaeManifoldRho,
    coord: usize,
    inner_max_iter: usize,
) -> f64 {
    let h = 2.0e-4;
    let mut plus = template.to_flat();
    let mut minus = template.to_flat();
    plus[coord] += h;
    minus[coord] -= h;
    let rho_plus = template.from_flat(plus.view());
    let rho_minus = template.from_flat(minus.view());
    let (_, vp, _, _) = evaluate(start, target, &rho_plus, inner_max_iter);
    let (_, vm, _, _) = evaluate(start, target, &rho_minus, inner_max_iter);
    (vp - vm) / (2.0 * h)
}

/// The full analytic outer-ρ gradient — explicit + direct log-det traces +
/// Occam + the #1006 third-order implicit-state correction — must match a
/// centered finite difference of the actual REML criterion (inner problem
/// re-solved at each ρ, so the FD carries the envelope/IFT terms).
fn assert_full_gradient_matches_fd(label: &str, f: &Fixture) {
    let (converged, _value, loss, cache) = evaluate(&f.term, &f.target, &f.rho, 8);
    let components = converged
        .analytic_outer_rho_gradient_at_converged(f.target.view(), &f.rho, &loss, &cache)
        .expect("analytic components");
    let analytic = components.gradient();
    let n_params = f.rho.to_flat().len();

    for coord in 0..n_params {
        let fd = centered_fd(&converged, &f.target, &f.rho, coord, 8);
        let diff = (fd - analytic[coord]).abs();
        let tol = 2.5e-3 * (1.0 + fd.abs().max(analytic[coord].abs()));
        assert!(
            diff <= tol,
            "[{label}] full rho gradient coord {coord}: fd={fd:.8e}, analytic={:.8e}, diff={diff:.3e}, tol={tol:.3e}",
            analytic[coord]
        );
    }
}

/// K=2 fixture whose per-atom decoder design is RANK-DEFICIENT in the data —
/// the #1117 OLMo-circle degeneracy. The latent coordinate is squeezed into a
/// narrow phase band so the 2nd-harmonic basis pair `[sin 2θ, cos 2θ]` is
/// barely excited and the bare data Gram `G_k = D_kᵀ D_k` drops rank. Under K=2
/// the shared-row logit×coordinate Gauss-Newton cross term then drives a per-row
/// `H_tt` block genuinely indefinite at/near the stationary point, so the
/// undamped evidence factor must condition it by unit-stiffness SPECTRAL
/// deflation (eigenvalue → +1, ρ-independent log 1 = 0). This is precisely the
/// branch whose former ridge-damped fallback injected a ρ-dependent evidence
/// bias and desynced the outer value from the analytic ρ-gradient (#1117). The
/// certificate this test asserts — analytic ∂V/∂ρ ≈ centered FD of the actual
/// re-solved criterion — is exactly `grad·v ≈ fd·v`: it holds iff the value and
/// gradient ride the SAME deflated factorization.
fn rank_deficient_fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> Fixture {
    let n = 80usize;
    let p = 6usize;
    let k_atoms = 2usize;
    let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator");

    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    // Squeeze both atoms' latent coordinate into a ±0.5%-wide band around 0.5:
    // the periodic 2nd-harmonic columns `[sin 2θ, cos 2θ]` are then nearly
    // unexcited, so the bare data Gram `G_k = D_kᵀ D_k` drops rank (the #1117
    // OLMo-circle degeneracy). Under K=2 the shared-row logit×coordinate
    // Gauss-Newton cross term drives a per-row `H_tt` block indefinite at/near
    // the optimum, forcing the undamped evidence factor down the spectral
    // unit-stiffness deflation path this fix installs.
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = 0.5 + 0.005 * ((row as f64 / n as f64) - 0.5);
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = phase;
        let route = if row < n / 2 { 1.4 } else { -1.4 };
        logits[[row, 0]] = route;
        logits[[row, 1]] = if row % 3 == 0 { 0.9 } else { 0.3 };
        let theta = std::f64::consts::TAU * phase;
        let basis = [
            1.0,
            theta.sin(),
            theta.cos(),
            (2.0 * theta).sin(),
            (2.0 * theta).cos(),
        ];
        for col in 0..p {
            // Deterministic, finite target so the inner solve converges; the
            // exact values do not matter for the FD certificate.
            let mut v = 0.0;
            for (b, &bv) in basis.iter().enumerate() {
                v += bv * (0.1 + 0.03 * (b as f64) - 0.01 * (col as f64));
            }
            target[[row, col]] = v;
        }
    }

    let mut atoms = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis evaluation");
        let decoder = Array2::from_shape_fn((m, p), |(r, c)| {
            0.1 + 0.05 * (r as f64) - 0.02 * (c as f64) + 0.01 * (atom_idx as f64)
        });
        let mut smooth = Array2::<f64>::eye(m);
        smooth[[0, 0]] = 0.0;
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .expect("circle atom")
        .with_basis_evaluator(Arc::new(
            PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator clone"),
        ) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    let rho = SaeManifoldRho::new(
        log_lambda_sparse,
        -8.0,
        vec![Array1::from_vec(vec![-8.0]), Array1::from_vec(vec![-8.0])],
    );
    Fixture { term, target, rho }
}

#[test]
fn sae_outer_rho_gradient_components_match_centered_fd_softmax() {
    let f = fixture(AssignmentMode::softmax(0.7), -8.0);
    assert_full_gradient_matches_fd("softmax", &f);
}

#[test]
fn sae_outer_rho_gradient_certificate_consistent_under_rank_deficient_k2() {
    // K=2 rank-deficient circle: the indefinite per-row H_tt must be spectral-
    // deflated at unit stiffness, NOT ridge-damped, so the outer REML value and
    // its analytic ρ-gradient stay consistent (grad·v ≈ fd·v). A ρ-dependent
    // ridge bias would break this certificate and is what stalled the outer BFGS
    // line-search for multi-atom fits (#1117).
    let f = rank_deficient_fixture(AssignmentMode::softmax(0.7), -8.0);
    assert_full_gradient_matches_fd("rank_deficient_k2_softmax", &f);
}

#[test]
fn sae_outer_rho_gradient_components_match_centered_fd_jumprelu() {
    // JumpReLU with the threshold below the active logits so both gates sit in
    // the optimization band and the sigmoid-sparsity third channel is live.
    let f = fixture(AssignmentMode::jumprelu(0.7, 0.0), -1.5);
    assert_full_gradient_matches_fd("jumprelu", &f);
}

#[test]
fn sae_outer_rho_gradient_components_match_centered_fd_ibp_map() {
    // IBP-MAP exercises the #1006 empirical-π third channel: pi_k(M_k) couples
    // every row in a column, so the outer-ρ gradient through log|H| depends on
    // the cross-row M_k channel of `logdet_theta_adjoint`. lambda_sparse is the
    // active prior weight here, so coord 0's FD directly stresses it.
    let f = fixture(AssignmentMode::ibp_map(0.7, 0.9, false), -1.5);
    assert_full_gradient_matches_fd("ibp_map", &f);
}
