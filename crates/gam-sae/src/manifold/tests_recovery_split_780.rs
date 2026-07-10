// Split from tests.rs under the #780 oversized-file gate: recovery-suite +
// registry/assignment tests from line ~6560 onward. Same module scope via
// `use super::tests::*` so shared fixtures keep working.
#![allow(unused_imports)]
use super::tests::*;
use super::*;
use approx::assert_abs_diff_eq;
use gam_linalg::faer_ndarray::fast_ata;
use gam_solve::arrow_schur::{
    ArrowFactorSlab, ArrowHtbetaCache, ArrowPcgDiagnostics, ArrowSolverMode, ArrowUndampedFactors,
};
use gam_solve::evidence::arrow_log_det_from_cache;
use gam_terms::analytic_penalties::ARDPenalty;
use ndarray::{Array5, array};

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
    let atom = SaeManifoldAtom::new(
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
    let d = 2usize;
    // True (lat, lon) coords.
    let mut true_coords = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let t = (i as f64) / (n as f64);
        true_coords[[i, 0]] = -0.5 + 1.0 * t; // lat in [-0.5, 0.5]
        true_coords[[i, 1]] = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * t;
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let lat = true_coords[[i, 0]];
        let lon = true_coords[[i, 1]];
        let x = lat.cos() * lon.cos();
        let y = lat.cos() * lon.sin();
        let zc = lat.sin();
        z[[i, 0]] = x;
        z[[i, 1]] = y;
        z[[i, 2]] = zc;
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
    let (phi0, jet0) = SphereChartEvaluator.evaluate(true_coords.view()).unwrap();
    let m = phi0.ncols();
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new(
        "sphere_atom",
        SaeAtomBasisKind::Sphere,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(SphereChartEvaluator));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_coords],
        vec![LatentManifold::Product(vec![
            LatentManifold::Interval {
                lo: -std::f64::consts::FRAC_PI_2,
                hi: std::f64::consts::FRAC_PI_2,
            },
            LatentManifold::Circle {
                period: std::f64::consts::TAU,
            },
        ])],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // The sphere atom's coordinate is a dim-2 product manifold (lat × lon),
    // so per-axis ARD must carry one log-precision per axis (`atom dim = 2`).
    // A length-1 block would be indexed out of bounds at `axis == 1` in the
    // per-axis assembly loop and is rejected by the per-axis ARD contract.
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(2)]);
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

    let atom = SaeManifoldAtom::new(
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

/// Regression test for issue #177: softmax assignment used to bail out of
/// the row-block Hessian assembly with "softmax assignment hessian diag
/// unavailable". The penalty now exposes the analytic diagonal extracted
/// from its row-dense HVP, so the joint-fit driver completes one step.
#[test]
pub(crate) fn softmax_assignment_hessian_diag_is_available_for_k2() {
    let n = 4usize;
    let k = 2usize;
    let logits = Array2::<f64>::from_shape_fn((n, k), |(i, j)| 0.1 * (i as f64) - 0.2 * (j as f64));
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
        .expect("softmax assignment Hessian diagonal must be available");
    assert_eq!(grad.len(), n * k);
    assert_eq!(diag.len(), n * k);
    assert!(grad.iter().all(|v| v.is_finite()));
    assert!(diag.iter().all(|v| v.is_finite()));
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
            SaeManifoldAtom::new(
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

    let mut ibp_registry = AnalyticPenaltyRegistry::new();
    ibp_registry.push(AnalyticPenaltyKind::IBPAssignment(Arc::new(
        gam_terms::analytic_penalties::IBPAssignmentPenalty::new(k, 1.2, 0.7, false),
    )));
    let ibp_err = term
        .validate_analytic_penalty_registry(&ibp_registry)
        .expect_err("SAE registry must reject IBP assignment sparsity");
    assert!(ibp_err.contains("assignment sparsity"));
}

#[test]
pub(crate) fn ibp_fixed_alpha_assignment_value_matches_logit_gradient_fd() {
    let n = 4usize;
    let k = 3usize;
    let logits = Array2::<f64>::from_shape_vec(
        (n, k),
        vec![
            -0.4, 0.2, 0.7, 0.1, -0.3, 0.5, 0.8, -0.1, -0.6, 0.3, 0.6, -0.2,
        ],
    )
    .expect("valid IBP logit grid");
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::ibp_map(0.9, 1.4, false),
    )
    .expect("valid IBP assignment");
    let rho = SaeManifoldRho::new(0.23_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, _) =
        assignment_prior_grad_hdiag(&assignment, &rho).expect("IBP assignment gradient");
    let idx = 5usize;
    let step = 1.0e-6_f64;
    let mut plus = assignment.clone();
    plus.logits[[idx / k, idx % k]] += step;
    let mut minus = assignment.clone();
    minus.logits[[idx / k, idx % k]] -= step;
    let fd =
        (assignment_prior_value(&plus, &rho) - assignment_prior_value(&minus, &rho)) / (2.0 * step);

    assert_abs_diff_eq!(grad[idx], fd, epsilon = 2.0e-7);
}

/// #1038 assembly-site wiring: a live IBP-active multi-atom assembly must
/// emit the exact cross-row Woodbury source `IbpCrossRowSource` whose
/// entries reproduce the NUMERICAL off-diagonal (`i≠j`) logit Hessian of the
/// SAE objective end-to-end. The ONLY source of cross-row `i≠j` logit
/// coupling is the IBP empirical-mass prior `M_k = Σ_i z_ik` (the data-fit
/// reconstruction of each row depends only on that row's own logits), so
/// `∂²(assignment_prior_value)/∂ℓ_ik∂ℓ_jk = d_k·z'_ik·z'_jk` for `i≠j`, with
/// `d_k = cross_row_d[k]` and `z'_ik = z_jac[i·K+k]` — exactly the rank-one
/// `U D Uᵀ` the assembled `sys.ibp_cross_row` encodes and the arrow-Schur
/// consumer rides as the exact Woodbury (value + logdet + θ/ρ-adjoint).
///
/// This certifies the assembly-site source matches the consumer's `U`/index
/// convention bit-for-bit: each entry's `global_t_index` is the row's logit
/// slot in the latent block (`row_offsets[i] + k` for the dense IBP layout),
/// and the rank-one product against the central-difference Hessian closes.
#[test]
pub(crate) fn ibp_assembly_emits_cross_row_woodbury_source_matching_fd_hessian() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let atom0 = SaeManifoldAtom::new(
        "periodic0",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.25], [-0.35], [0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let atom1 = SaeManifoldAtom::new(
        "periodic1",
        SaeAtomBasisKind::Periodic,
        1,
        phi1,
        jet1,
        array![[-0.10], [0.20], [0.30]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    // IBP-active logits (positive ⇒ near-on gate, interior π so the
    // empirical-mass channel is live — `pi_jac ≠ 0`).
    let logits = array![[1.2, 0.4], [0.6, 1.0], [0.9, 0.3], [1.4, 0.7]];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.8, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let target = array![[0.12], [-0.03], [0.08], [0.20]];
    let rho = SaeManifoldRho::new(
        0.3_f64.ln(),
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
    );

    let n = term.assignment.n_obs();
    let k = term.assignment.k_atoms();

    // Assemble the live arrow system; it must now carry the IBP source.
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("IBP arrow assembly");
    let source = sys
        .ibp_cross_row
        .as_ref()
        .expect("an IBP-active assembly must emit the cross-row Woodbury source");
    assert_eq!(source.r, k, "the rank must be the atom count K");

    // Rebuild the dense `U` and `d` the consumer sees from the sparse entries,
    // and check the global-index convention is the dense IBP layout
    // (`row_offsets[i] + k`), i.e. atom `k`'s logit slot of row `i`.
    let total_t = sys.row_offsets[n];
    let mut u = Array2::<f64>::zeros((total_t, k));
    for &(g, atom_k, z_prime) in &source.entries {
        u[[g, atom_k]] += z_prime;
    }
    for i in 0..n {
        for atom_k in 0..k {
            let g = sys.row_offsets[i] + atom_k;
            // The entry for (row i, atom k) must sit at the row's logit slot.
            assert!(
                u[[g, atom_k]].abs() > 0.0 || term.assignment.logits[[i, atom_k]].abs() > 1.0e3,
                "row {i} atom {atom_k} logit slot must carry a z' entry"
            );
        }
    }

    // Central-difference the assignment-prior value cross-row (i≠j) Hessian
    // and assert it equals the rank-one `d_k·z'_ik·z'_jk` the source encodes.
    let d = source.d.clone();
    let step = 1.0e-5_f64;
    let fd_cross = |i: usize, j: usize, atom_k: usize| -> f64 {
        let bump = |si: f64, sj: f64| -> f64 {
            let mut a = term.assignment.clone();
            a.logits[[i, atom_k]] += si * step;
            a.logits[[j, atom_k]] += sj * step;
            assignment_prior_value(&a, &rho)
        };
        // mixed second difference ∂²V/∂ℓ_ik∂ℓ_jk
        (bump(1.0, 1.0) - bump(1.0, -1.0) - bump(-1.0, 1.0) + bump(-1.0, -1.0))
            / (4.0 * step * step)
    };

    for atom_k in 0..k {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let gi = sys.row_offsets[i] + atom_k;
                let gj = sys.row_offsets[j] + atom_k;
                let analytic = d[atom_k] * u[[gi, atom_k]] * u[[gj, atom_k]];
                let fd = fd_cross(i, j, atom_k);
                assert_abs_diff_eq!(analytic, fd, epsilon = 5.0e-6);
            }
        }
    }

    // Distinct atom columns do NOT couple cross-row (independent
    // stick-breaking masses): the off-diagonal in a DIFFERENT column is zero.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut a = term.assignment.clone();
            let cross = {
                let s = 1.0e-5_f64;
                let mut bump = |si: f64, sj: f64| -> f64 {
                    a.logits[[i, 0]] = term.assignment.logits[[i, 0]] + si * s;
                    a.logits[[j, 1]] = term.assignment.logits[[j, 1]] + sj * s;
                    assignment_prior_value(&a, &rho)
                };
                (bump(1.0, 1.0) - bump(1.0, -1.0) - bump(-1.0, 1.0) + bump(-1.0, -1.0))
                    / (4.0 * s * s)
            };
            assert_abs_diff_eq!(cross, 0.0, epsilon = 5.0e-6);
        }
    }
}

#[test]
pub(crate) fn jumprelu_assignment_value_matches_logit_gradient_fd() {
    let n = 4usize;
    let k = 2usize;
    let temperature = 0.35_f64;
    let threshold = 0.1_f64;
    let logits =
        Array2::<f64>::from_shape_vec((n, k), vec![-13.0, -0.2, 0.0, 0.05, 0.15, 0.4, 0.9, 1.5])
            .expect("valid JumpReLU logit grid");
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::threshold_gate(temperature, threshold),
    )
    .expect("valid JumpReLU assignment");
    let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, _) =
        assignment_prior_grad_hdiag(&assignment, &rho).expect("JumpReLU assignment gradient");
    let idx = 4usize;
    let step = 1.0e-6_f64;
    let mut plus = assignment.clone();
    plus.logits[[idx / k, idx % k]] += step;
    let mut minus = assignment.clone();
    minus.logits[[idx / k, idx % k]] -= step;
    let fd =
        (assignment_prior_value(&plus, &rho) - assignment_prior_value(&minus, &rho)) / (2.0 * step);

    assert_abs_diff_eq!(grad[idx], fd, epsilon = 2.0e-8);
}

#[test]
pub(crate) fn jumprelu_assignment_prior_hessian_diag_is_exact_over_logit_sweep() {
    let n = 6usize;
    let k = 2usize;
    let temperature = 0.35_f64;
    let threshold = 0.1_f64;
    let logits = Array2::<f64>::from_shape_vec(
        (n, k),
        vec![
            -2.0, -0.2, 0.0, 0.05, 0.1, 0.15, 0.4, 0.9, 1.5, 2.5, 4.0, 6.0,
        ],
    )
    .expect("valid logit grid");
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.clone(),
        coords,
        manifolds,
        AssignmentMode::threshold_gate(temperature, threshold),
    )
    .expect("valid JumpReLU assignment");
    let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
        .expect("JumpReLU assignment prior hessian diag");
    let inv_tau = 1.0 / temperature;
    let inv_tau2 = inv_tau * inv_tau;
    let sparsity_strength = rho.log_lambda_sparse.exp();

    assert_eq!(grad.len(), n * k);
    assert_eq!(diag.len(), n * k);
    let mut saw_negative = false;
    for (idx, &entry) in diag.iter().enumerate() {
        let logit = logits[[idx / k, idx % k]];
        // Expected = exact second derivative of the threshold-centered
        // surrogate σ((l−θ)/τ), using the same machine-precision support as
        // the value and gradient paths.
        let expected = if jumprelu_in_optimization_band(logit, threshold, temperature) {
            let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
            let slope = activation * (1.0 - activation);
            sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2
        } else {
            0.0
        };
        assert!(
            entry.is_finite(),
            "JumpReLU hessian_diag must be finite at index {idx}"
        );
        saw_negative |= entry < 0.0;
        assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
    }
    assert!(
        saw_negative,
        "exact JumpReLU hessian_diag must go negative above the threshold"
    );
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
pub(crate) fn ibp_map_k2_periodic_torus_recovers_signal_with_lsq_init() {
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

    // LSQ seed: joint design X = [0.5 * Phi_1 | 0.5 * Phi_2] (IBP-MAP
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
        let atom = SaeManifoldAtom::new(
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
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    // `lambda_sparse` is the IBP assignment-sparsity prior weight (now wired
    // through `assignment_prior_grad_hdiag`'s IBP branch, #853). The
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
        "K=2 periodic torus IBP-MAP R² = {r2:.4} (ssr={ssr:.4}, sst={sst:.4}) should be > 0.5 with LSQ-seeded decoder"
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
        let atom = SaeManifoldAtom::new(
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
    let atom = SaeManifoldAtom::new(
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
        Arc::new(SphereChartEvaluator),
        array![[-0.5, 0.3], [0.2, -1.1], [0.7, 0.9]],
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
    let atom = SaeManifoldAtom::new(
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
    let atom = SaeManifoldAtom::new(
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
        cross_row_woodbury: None,
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
        cross_row_woodbury: None,
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
        .outer_gradient_arrow_solver(&cache, &obj.current_rho.lambda_smooth_vec())
    {
        Err(err) => err,
        Ok(..) => panic!("near-singular evidence factor without a matching gauge must reject"),
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
    let atom = SaeManifoldAtom::new(
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
        cross_row_woodbury: None,
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
        .outer_gradient_arrow_solver(&cache, &obj.current_rho.lambda_smooth_vec())
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

/// gam#577 / gam#579 root cause: the continuation pre-warm forwards an
/// EMPTY β before the first accepted eval (`state.last_beta` starts
/// empty). The seed hook must treat that as the documented "no warm-start
/// available, proceed cold" no-op (`SeedOutcome::NoSlot`) rather than
/// erroring on `β length 0 != decoder dim` — the error dropped EVERY
/// continuation seed and forced a full cold solve on every outer seed.
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
/// the warm-start the continuation walk relies on for the big speedup
/// (gam#577 / gam#579). We verify reuse behaviorally: seed a known β, run
/// one eval with zero inner Newton iterations (so the solve cannot move
/// β off the seed), and confirm the published `inner_beta_hint` is exactly
/// the seeded β. A cold start would have published the term's pristine β
/// instead.
#[test]
pub(crate) fn seed_inner_state_installs_and_reuses_matching_beta() {
    let mut obj = warmstart_test_objective();
    let dim = obj.term.beta_dim();
    // A distinctive seed that differs from the term's pristine decoder.
    let pristine = obj.term.flatten_beta();
    let seed: Array1<f64> = Array1::from_shape_fn(dim, |i| pristine[i] + 0.5 + 0.01 * (i as f64));
    assert!(
        (&seed - &pristine).iter().any(|d| d.abs() > 1e-6),
        "seed must differ from the pristine β for the reuse check to be meaningful"
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

/// Build a non-periodic 1-D atom with a genuine order-2 finite-difference
/// roughness Gram, a non-constant-speed decoder, and explicit
/// `(basis_values, basis_jacobian)` so the intrinsic reweighting in
/// [`SaeManifoldAtom::refresh_intrinsic_smooth_penalty`] is exercised
/// directly. A localized (near-diagonal) basis makes each coefficient's
/// representative speed the speed at its own sample.
pub(crate) fn intrinsic_test_atom(jacobian_scale: f64) -> SaeManifoldAtom {
    let m = 5usize;
    let n = m;
    let p = 1usize;
    let mut phi = Array2::<f64>::zeros((n, m));
    let mut jet = Array3::<f64>::zeros((n, m, 1));
    let mut decoder = Array2::<f64>::zeros((m, p));
    for mu in 0..m {
        // Localized basis: Φ_μ(t_n) ≈ δ_{nμ}.
        phi[[mu, mu]] = 1.0;
        // Per-sample basis derivative (axis 0) grows with μ — a
        // non-constant-speed curve — scaled by `jacobian_scale` to emulate
        // a global linear reparameterization t -> t / jacobian_scale.
        jet[[mu, mu, 0]] = jacobian_scale * (1.0 + mu as f64);
        decoder[[mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    SaeManifoldAtom::new(
        "intrinsic-1d",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap()
}

/// The roughness operator order is recovered from the raw Gram's null
/// space: an order-2 difference penalty annihilates the affine functions,
/// so `nullity = 2` and the arc-length exponent is `β = ½ − 2 = −3/2`.
#[test]
pub(crate) fn intrinsic_penalty_recovers_order_two_from_nullity() {
    let atom = intrinsic_test_atom(1.0);
    assert_eq!(atom.smooth_penalty_order, 2);
}

#[test]
pub(crate) fn line_search_snapshot_restores_intrinsic_smooth_penalty() {
    let atom = intrinsic_test_atom(1.0);
    let n = atom.n_obs();
    let logits = Array2::<f64>::zeros((n, 1));
    let coords = vec![Array2::<f64>::zeros((n, 1))];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let original = term.atoms[0].smooth_penalty.clone();
    let snapshot = term.snapshot_mutable_state();

    term.atoms[0].decoder_coefficients[[0, 0]] *= 3.0;
    term.atoms[0].refresh_intrinsic_smooth_penalty();
    let changed = (&term.atoms[0].smooth_penalty - &original)
        .mapv(f64::abs)
        .sum();
    assert!(
        changed > 1e-6,
        "test setup must perturb the live intrinsic smoothness Gram"
    );

    term.restore_mutable_state(&snapshot)
        .expect("differential restore rebuilds the basis");
    let restored = (&term.atoms[0].smooth_penalty - &original)
        .mapv(f64::abs)
        .sum();
    assert!(
        restored < 1e-12,
        "line-search restore left a stale intrinsic smoothness Gram: {restored}"
    );
}

/// Gauge invariance (issue #673): a global reparameterization of the latent
/// coordinate scales every per-sample speed by a common factor, which
/// cancels in the centered reweighting — so the intrinsic Gram `S̃` (and
/// hence the topology evidence `tr(BᵀS̃B)`) is identical across the two
/// reparameterizations, even though the basis Jacobian (the metric) differs.
#[test]
pub(crate) fn intrinsic_penalty_is_invariant_to_speed_rescaling() {
    let a1 = intrinsic_test_atom(1.0);
    let a2 = intrinsic_test_atom(7.5);
    // Same raw Gram and decoder; only the basis Jacobian (speed) differs.
    assert_abs_diff_eq!(
        (&a1.smooth_penalty_raw - &a2.smooth_penalty_raw)
            .mapv(f64::abs)
            .sum(),
        0.0,
        epsilon = 1e-12
    );
    // The intrinsic (reweighted) Gram is identical despite the 7.5x speed
    // rescale: the centered ratios are invariant to a global speed factor.
    let diff = (&a1.smooth_penalty - &a2.smooth_penalty)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff < 1e-9,
        "intrinsic Gram changed under a global speed rescale (gauge leak): {diff}"
    );
}

pub(crate) fn affine_canonicalization_test_term() -> SaeManifoldTerm {
    let n = 80usize;
    let p = 2usize;
    let evaluator = EuclideanPatchEvaluator::new(1, 2).unwrap();
    let mut coords = Array2::<f64>::zeros((n, 1));
    for row in 0..n {
        coords[[row, 0]] = -4.0 + 12.0 * row as f64 / (n as f64 - 1.0);
    }
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[0, 0]] = 0.8;
    decoder[[1, 0]] = -0.4;
    decoder[[2, 0]] = 0.15;
    decoder[[0, 1]] = -0.2;
    decoder[[1, 1]] = 0.9;
    decoder[[2, 1]] = -0.08;
    let smooth_penalty = gam_terms::basis::create_difference_penalty_matrix(3, 2, None).unwrap();
    let atom = SaeManifoldAtom::new(
        "affine-canonicalization",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        smooth_penalty,
    )
    .unwrap()
    .with_basis_second_jet(Arc::new(evaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

#[test]
pub(crate) fn affine_canonicalization_transports_live_penalty_instead_of_recomputing() {
    let mut term = affine_canonicalization_test_term();
    let before: f64 = term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let old_smooth_penalty = term.atoms[0].smooth_penalty.clone();
    let old_decoder = term.atoms[0].decoder_coefficients.clone();

    term.canonicalize_atom_affine_gauge(0, None).unwrap();
    let after: f64 = term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let invariant_gap = (after - before).abs() / before.abs().max(1.0);
    assert!(
        invariant_gap < 1.0e-9,
        "canonicalization changed fixed-rho smoothness energy: before={before:.12e}, after={after:.12e}"
    );

    let mut recomputed_atom = term.atoms[0].clone();
    recomputed_atom.refresh_intrinsic_smooth_penalty();
    let recomputed_term = SaeManifoldTerm::new(
        vec![recomputed_atom],
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((term.n_obs(), 1)),
            vec![term.assignment.coords[0].as_matrix()],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap(),
    )
    .unwrap();
    let recomputed: f64 = recomputed_term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let recompute_jump = (recomputed - before).abs() / before.abs().max(1.0);
    assert!(
        recompute_jump > 1.0e-2,
        "test fixture failed to expose the intrinsic recompute energy jump: before={before:.12e}, recomputed={recomputed:.12e}"
    );

    let transport =
        solve_basis_transport(term.atoms[0].basis_values.view(), old_smooth_penalty.view())
            .expect_err("shape mismatch must reject invalid transport solve");
    assert!(
        transport.contains("row mismatch") || transport.contains("SVD failed"),
        "unexpected transport-shape diagnostic: {transport}"
    );
    let roundtrip = transport_smooth_penalty_for_decoder(
        solve_design_least_squares(
            term.atoms[0].decoder_coefficients.view(),
            old_decoder.view(),
        )
        .unwrap_or_else(|err| panic!("decoder transport fixture became singular: {err}"))
        .view(),
        old_smooth_penalty.view(),
    );
    assert!(
        roundtrip.is_err(),
        "non-square decoder transport must not be accepted as a penalty congruence"
    );
}

/// Non-constant speed genuinely reshapes the penalty: the intrinsic Gram
/// must differ from the raw Gram when the decoder curve is not
/// constant-speed, otherwise the reweighting is a no-op and the gauge fix
/// would be vacuous. The congruence preserves symmetry.
#[test]
pub(crate) fn intrinsic_penalty_differs_from_raw_under_varying_speed() {
    let atom = intrinsic_test_atom(1.0);
    let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff > 1e-6,
        "intrinsic reweighting was a no-op on a non-constant-speed curve: {diff}"
    );
    for i in 0..atom.basis_size() {
        for j in 0..atom.basis_size() {
            assert_abs_diff_eq!(
                atom.smooth_penalty[[i, j]],
                atom.smooth_penalty[[j, i]],
                epsilon = 1e-12
            );
        }
    }
}

/// Constant-speed atoms are untouched: when every sample shares one speed
/// (the periodic sin/cos limit), the centered weights are all `1`, so
/// `S̃ = S_raw` exactly and the topology comparison among constant-speed
/// atoms is unaffected.
#[test]
pub(crate) fn intrinsic_penalty_leaves_constant_speed_atom_unchanged() {
    let m = 6usize;
    let n = m;
    let mut phi = Array2::<f64>::zeros((n, m));
    let mut jet = Array3::<f64>::zeros((n, m, 1));
    let mut decoder = Array2::<f64>::zeros((m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        // Identical derivative magnitude at every sample => constant speed.
        jet[[mu, mu, 0]] = 2.0;
        decoder[[mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let atom = SaeManifoldAtom::new(
        "constant-speed",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff < 1e-9,
        "constant-speed atom's penalty was reweighted (should be identity): {diff}"
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
            SaeManifoldAtom::new(
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

pub(crate) fn fixed_state_logdet(
    mut term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> f64 {
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
        .expect("fixed-state cache");
    arrow_log_det_from_cache(&cache).expect("fixed-state authoritative joint logdet")
}

// [#780 line-count gate] The #1557 arrow-Schur parallelism-invariance
// regression test (`arrow_schur_assembly_is_faer_parallelism_invariant_1557`)
// was split into the sibling `tests_parallelism_invariance_1557.rs` module
// (declared in `mod.rs`) to keep this tracked file under the 10k limit.
//
// The four stationary-cache `∂log|H|/∂θ` adjoint regression tests
// (`sae_logdet_theta_adjoint_matches_dense_fd_*`,
// `ibp_rho_sparse_logdet_trace_matches_dense_fd_1416`,
// `learnable_ibp_alpha_logdet_trace_matches_dense_fd_1417`) were likewise split
// into the sibling `tests_logdet_adjoint_780.rs` module for the same gate; they
// still source the shared `gamma_fd_tiny_fixture` / `fixed_state_logdet`
// helpers, which remain defined here.
