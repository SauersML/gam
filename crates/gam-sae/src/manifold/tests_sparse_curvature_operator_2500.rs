// `manifold/mod.rs` declares this module only as `#[cfg(test)] mod tests_sparse_curvature_operator_2500;`,
// so every item here is test-only. Stating that scope in the file makes it a
// claim the compiler enforces rather than one carried by the filename.
#![cfg(test)]
//! #2500 — the assignment prior's sparse log-strength curvature operator
//! `∂H_tt/∂ρ_sparse` must be MODELLED for every family that mints a sparse outer
//! coordinate, not only for softmax — and it must be the derivative of the
//! operator the solver actually installs.
//!
//! `AssignmentStrengthLayout` mints a sparse outer coordinate for exactly three
//! families: `Softmax` (`SoftmaxEntropy`) and `OrderedBetaBernoulli` /
//! `ThresholdGate` (`PenaltyWeight`); `TopK` is `FixedSupport` and carries none.
//! The ordered-Beta–Bernoulli operator is genuinely cross-row and is owned by
//! `dense_exact_a_ordered_bb_sparse_trace`, so the ONE family whose diagonal
//! operator was simply absent is `ThresholdGate` — even though
//! `assignment_prior_log_strength_hdiag_weighted` has always computed it exactly
//! and the arrow assembly writes precisely that diagonal into `block.htt`.
//!
//! Installing it is necessary but NOT sufficient. The threshold gate is the only
//! family whose installed curvature `w·λ·s·(1−2a)/τ²` is SIGNED (every sibling is
//! PSD-majorized: `λS⊗I`, `α·softplus(cos κt)`, the softmax Gershgorin radius),
//! so wherever the gate sits above its threshold the per-row `H_tt` block goes
//! indefinite and the factorization spectrally deflates that direction to
//! ρ-independent unit stiffness. The raw operator then over-claims curvature on
//! exactly those directions. These gates pin both halves.

use super::*;
use crate::manifold::construction::ThetaAdjointDhChannel;
use ndarray::{Array1, Array2};

/// A `ThresholdGate` twin of `gamma_fd_tiny_fixture`: two periodic atoms on one
/// shared circle, K=2 free logits per row, and a target generated through the
/// SAME gate the fit uses (`a_k = σ((ℓ_k − θ)/τ)`), so the inner state is a real
/// fit rather than a mismatched-forward artefact.
///
/// `straddle` selects the two strata this defect lives on:
///
/// * `true` — logits straddle the threshold, so half the free logits carry
///   NEGATIVE prior curvature (`1−2a < 0`). Measured: all ten rows spectrally
///   deflate exactly one direction each. This is the stratum the raw operator
///   gets wrong.
/// * `false` — every logit sits below the threshold, so all prior curvature is
///   positive and NO row deflates. This is the stratum where the raw operator is
///   already exact, and it isolates the two mechanisms from each other.
pub(crate) fn threshold_gate_tiny_fixture(
    straddle: bool,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let k_atoms = 2usize;
    let m = 3usize;
    let tau = 1.0_f64;
    let threshold = 0.0_f64;
    let evaluator = Arc::new(
        PeriodicHarmonicEvaluator::new(m)
            .expect("m=3 is odd and positive, the basis width this evaluator requires"),
    );
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
        if straddle {
            logits[[row, 0]] = if row % 2 == 0 { -0.8 } else { 0.9 };
            logits[[row, 1]] = if row % 2 == 0 { 0.7 } else { -0.5 };
        } else {
            logits[[row, 0]] = -0.8;
            logits[[row, 1]] = -0.5;
        }
        for atom in 0..k_atoms {
            let gate = 1.0 / (1.0 + (-(logits[[row, atom]] - threshold) / tau).exp());
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        gate * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator
            .evaluate(coords[atom].view())
            .expect("fixture coords are finite and inside the unit-period circle chart");
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("tgate_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .expect("decoder is (m, p) and phi/jet come from the same evaluator")
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let mode = AssignmentMode::threshold_gate(tau, threshold);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        mode,
    )
    .expect("k_atoms coord blocks and k_atoms manifolds match the logits' k_atoms columns");
    let term = SaeManifoldTerm::new(atoms, assignment)
        .expect("the assignment was built with exactly one block per atom");
    // Moderate-penalty basin, mirroring `converged_state_with_residual`: every
    // channel stays live and the ±h FD probes stay factorizable.
    let rho = SaeManifoldRho::new(
        -1.0,
        -1.0,
        vec![Array1::from_vec(vec![-1.0]), Array1::from_vec(vec![-1.0])],
    )
    .for_assignment(mode);
    (term, target, rho)
}

/// Build the frozen-θ̂ cache the operator map is evaluated against. A ZERO inner
/// budget assembles `H(ρ) = H_data(θ̂) + penalty(ρ)` without re-running the fit,
/// which is exactly the fixed-stratum object `∂H/∂ρ` differentiates.
fn frozen_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> (SaeManifoldLoss, ArrowFactorCache) {
    let mut t = term.clone();
    let (_value, loss, cache) = t
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("threshold-gate fixed-theta cache");
    (loss, cache)
}

/// Total number of spectrally/gauge deflated per-row directions in a cache. The
/// FD gates below are only valid on a fixed DISCRETE stratum, so this doubles as
/// the branch-identity check across the ±h endpoints.
fn deflated_direction_count(term: &SaeManifoldTerm, cache: &ArrowFactorCache) -> usize {
    (0..term.n_obs())
        .map(|row| cache.deflated_row_directions[row].len())
        .sum()
}

/// The flat logit slot `(row, atom)`'s global index in the cache's t-layout.
/// ThresholdGate always uses the dense (full-support) layout, where the row's
/// free logits occupy the first `assignment_coord_dim()` positions of the block.
fn logit_slots(term: &SaeManifoldTerm, cache: &ArrowFactorCache) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let dim = term.assignment.assignment_coord_dim();
    for row in 0..term.n_obs() {
        let base = cache.row_offsets[row];
        for atom in 0..dim.min(cache.row_dims[row]) {
            out.push((row, atom, base + atom));
        }
    }
    out
}

/// #2500 GATE 2 — the modelled operator must BE `∂A/∂ρ_sparse` of the EXACT
/// stationarity Hessian the dense route materializes and ranks, entry by entry,
/// on BOTH strata: the deflation-free one and the one where every row deflates.
///
/// This is the property the refusal was protecting. An operator that assembles
/// but does not differentiate the installed `A` is worse than a refusal, because
/// it silently desyncs the ρ-gradient from the criterion — and that is exactly
/// what the RAW prior diagonal does here: `apply_cached_arrow_hessian` reads the
/// cache's CONDITIONED row factors, so on a deflated direction the installed
/// curvature is ρ-independent unit stiffness while the raw diagonal claims
/// `w·λ·s·(1−2a)/τ²`.
#[test]
fn threshold_gate_sparse_operator_is_the_installed_exact_a_derivative_2500() {
    for straddle in [false, true] {
        let (term, target, rho) = threshold_gate_tiny_fixture(straddle);
        let (_loss, cache) = frozen_cache(&term, &target, &rho);
        let sparse = rho
            .sparse_flat_index()
            .expect("a ThresholdGate rho must carry a sparse log-strength coordinate");
        let operators = term
            .penalty_curvature_operators_by_flat(&rho, &cache)
            .expect("#2500: the ThresholdGate sparse curvature operator must be modelled");
        let block = operators
            .get(&sparse)
            .expect("#2500: the sparse coordinate must own a curvature operator");
        // The exact-minus-majorizer delta has no threshold-gate part (the assembly
        // installs the exact signed prior curvature, unmajorized), so
        // `∂A/∂ρ_sparse` is the operator alone. Pin that rather than assume it.
        let deltas = term
            .exact_stationarity_penalty_derivative_delta_by_flat(&rho, &cache)
            .expect("exact-minus-majorizer delta map");
        assert!(
            !deltas.contains_key(&sparse),
            "#2500: ΔC carries no threshold-gate sparse part, so ∂A/∂ρ_sparse is the \
             operator alone; a delta appearing here means this gate is comparing the \
             wrong object"
        );

        let deflated = deflated_direction_count(&term, &cache);
        if straddle {
            assert!(
                deflated > 0,
                "#2500: the straddling arm must actually exercise the spectral-deflation \
                 stratum (the raw operator is exact without it, so a zero here would make \
                 this arm a duplicate of the other)"
            );
        } else {
            assert_eq!(
                deflated, 0,
                "#2500: the below-threshold arm must be deflation-free, so it isolates the \
                 operator from the deflation map"
            );
        }

        let dense_a = |r: &SaeManifoldRho| -> (Array2<f64>, usize) {
            let (_l, c) = frozen_cache(&term, &target, r);
            let a = term
                .materialize_exact_hessian_dense(r, target.view(), &c)
                .expect("dense exact A");
            let count = deflated_direction_count(&term, &c);
            (a, count)
        };
        let base = rho.to_flat();
        let h = 1.0e-5;
        let mut plus_flat = base.clone();
        plus_flat[sparse] += h;
        let mut minus_flat = base.clone();
        minus_flat[sparse] -= h;
        let (a_plus, deflated_plus) = dense_a(&rho.from_flat(plus_flat.view()).unwrap());
        let (a_minus, deflated_minus) = dense_a(&rho.from_flat(minus_flat.view()).unwrap());
        // Branch identity: a central difference across a change in the deflated
        // dimension is a difference of two different operators, not a derivative.
        assert!(
            deflated_plus == deflated && deflated_minus == deflated,
            "#2500: the ±h endpoints must sit on the SAME discrete deflation stratum \
             (base={deflated}, +h={deflated_plus}, -h={deflated_minus})"
        );

        let dim = a_plus.nrows();
        let mut worst = 0.0_f64;
        let mut worst_label = String::new();
        for i in 0..dim {
            for j in 0..dim {
                let fd = (a_plus[[i, j]] - a_minus[[i, j]]) / (2.0 * h);
                let analytic = block[[i, j]];
                let err = (fd - analytic).abs();
                let tol = 1.0e-7 + 1.0e-5 * analytic.abs();
                if err / tol > worst {
                    worst = err / tol;
                    worst_label =
                        format!("[{i},{j}] analytic={analytic:.12e} fd={fd:.12e} tol={tol:.3e}");
                }
            }
        }
        assert!(
            worst <= 1.0,
            "#2500 (straddle={straddle}, {deflated} deflated directions): the sparse \
             curvature operator must equal dA/drho_sparse of the INSTALLED exact Hessian; \
             worst normalized error {worst:.3} at {worst_label}"
        );
    }
}

/// #2500 GATE 4 — the end-to-end contract, and the one that catches the 13%
/// error the raw operator produced: the dense exact-A sparse log-determinant
/// trace `½[tr(A⁺ ∂A/∂ρ) − tr(A_tt⁺ ∂A/∂ρ)]` must be the central finite
/// difference of the PRODUCTION value it differentiates,
/// `½(log|A| − log|A_tt|)` from `exact_observed_information_log_dets`.
///
/// Measured before the deflation map: 4.40108e-1 analytic against 5.08109e-1 FD
/// on the straddling arm (stable across four decades of `h`, so not FD noise),
/// and exact on the deflation-free arm.
#[test]
fn threshold_gate_dense_exact_a_sparse_logdet_trace_matches_finite_difference_2500() {
    for straddle in [false, true] {
        let (term, target, rho) = threshold_gate_tiny_fixture(straddle);
        let (loss, cache) = frozen_cache(&term, &target, &rho);
        let sparse = rho.sparse_flat_index().expect("sparse coordinate");
        let trace = term
            .dense_exact_a_logdet_channels(target.view(), &rho, &loss, &cache)
            .expect("#2500: the dense exact-A logdet channels must assemble")
            .logdet_trace;
        assert!(
            trace[sparse].abs() > 1.0e-3,
            "#2500: the sparse logdet trace must be materially live on this fixture, else \
             the FD comparison is a zero-vs-zero gate: {}",
            trace[sparse]
        );

        let log_dets = |r: &SaeManifoldRho| -> (f64, f64) {
            let (_l, c) = frozen_cache(&term, &target, r);
            term.exact_observed_information_log_dets(r, target.view(), &c)
                .expect("production exact-A log dets")
        };
        let base = rho.to_flat();
        let h = 1.0e-5;
        let mut plus_flat = base.clone();
        plus_flat[sparse] += h;
        let mut minus_flat = base.clone();
        minus_flat[sparse] -= h;
        let (joint_plus, tt_plus) = log_dets(&rho.from_flat(plus_flat.view()).unwrap());
        let (joint_minus, tt_minus) = log_dets(&rho.from_flat(minus_flat.view()).unwrap());
        let fd = 0.5 * ((joint_plus - joint_minus) - (tt_plus - tt_minus)) / (2.0 * h);
        let err = (trace[sparse] - fd).abs();
        let tol = 1.0e-6 + 1.0e-5 * fd.abs();
        assert!(
            err <= tol,
            "#2500 (straddle={straddle}): the sparse logdet trace must differentiate the \
             ranked value; analytic={:.9e} fd={fd:.9e} err={err:.3e} tol={tol:.3e}",
            trace[sparse]
        );
    }
}

/// #2500 GATE 6 — the deflation map is coordinate-agnostic, so the ARD operator
/// must pass through it too. A ρ_ard perturbation on the straddling (deflating)
/// fixture is the same test as GATE 2 for a coordinate that has nothing to do
/// with the assignment prior; before the map it was wrong there for the same
/// reason, silently, and only the threshold gate made the stratum reachable.
#[test]
fn deflation_map_applies_to_every_row_local_curvature_coordinate_2500() {
    let (term, target, rho) = threshold_gate_tiny_fixture(true);
    let (_loss, cache) = frozen_cache(&term, &target, &rho);
    assert!(
        deflated_direction_count(&term, &cache) > 0,
        "#2500: this gate needs the deflating stratum"
    );
    let operators = term
        .penalty_curvature_operators_by_flat(&rho, &cache)
        .expect("operator map");
    let deltas = term
        .exact_stationarity_penalty_derivative_delta_by_flat(&rho, &cache)
        .expect("delta map");

    let base = rho.to_flat();
    let h = 1.0e-5;
    let total_t = cache.delta_t_len();
    let coord = rho.ard_flat_index(0, 0);
    let block = operators
        .get(&coord)
        .expect("#2500: the ARD coordinate must own a curvature operator");
    let expected = match deltas.get(&coord) {
        Some(delta) => block + delta,
        None => block.clone(),
    };
    let dense_a = |flat: &Array1<f64>| -> Array2<f64> {
        let r = rho.from_flat(flat.view()).unwrap();
        let (_l, c) = frozen_cache(&term, &target, &r);
        term.materialize_exact_hessian_dense(&r, target.view(), &c)
            .expect("dense exact A")
    };
    let mut plus_flat = base.clone();
    plus_flat[coord] += h;
    let mut minus_flat = base.clone();
    minus_flat[coord] -= h;
    let a_plus = dense_a(&plus_flat);
    let a_minus = dense_a(&minus_flat);
    // Restrict to the t-block: the deflation map acts on the row-local latent
    // slots, which is where the ARD operator lives.
    let mut worst = 0.0_f64;
    let mut label = String::new();
    for i in 0..total_t {
        for j in 0..total_t {
            let fd = (a_plus[[i, j]] - a_minus[[i, j]]) / (2.0 * h);
            let analytic = expected[[i, j]];
            let tol = 1.0e-6 + 1.0e-4 * analytic.abs();
            let ratio = (fd - analytic).abs() / tol;
            if ratio > worst {
                worst = ratio;
                label = format!("[{i},{j}] analytic={analytic:.9e} fd={fd:.9e}");
            }
        }
    }
    assert!(
        worst <= 1.0,
        "#2500: the ARD curvature operator must equal dA/drho on the t-block of a \
         deflating fixture; worst normalized error {worst:.3} at {label}"
    );
}

/// #2500 GATE 7 — the issue's actual ask, end to end: a ThresholdGate fit whose
/// ρ carries a sparse log-strength coordinate must be EVALUABLE by the outer
/// solver, not aborted at the outer-BFGS seed evaluation by
/// "penalty_curvature_operators_by_flat: rho carries a sparse log-strength
/// coordinate under an assignment prior whose ∂H/∂ρ_sparse operator this map does
/// not model".
///
/// The gate is deliberately about REACHABILITY, not fit quality: it asserts the
/// cascade never reports that refusal (nor its ch4/ch1 siblings), whatever else
/// the fit decides. A fixture-specific numerical outcome would make this test a
/// hostage to the seeding cascade; the refusal class is what #2500 is about.
#[test]
fn threshold_gate_outer_solve_is_not_aborted_by_an_unmodelled_sparse_operator_2500() {
    use gam_solve::rho_optimizer::OuterProblem;
    use gam_solve::seeding::SeedConfig;

    let (term, target, rho) = threshold_gate_tiny_fixture(true);
    let init_flat = rho.to_flat();
    let n_params = init_flat.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 8, 0.04, 1.0e-6, 1.0e-6);
    let result = OuterProblem::new(n_params)
        .with_initial_rho(init_flat)
        .with_seed_config(SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold");
    if let Err(err) = &result {
        let msg = err.to_string();
        for marker in [
            "operator this map does not model",
            "majorizer operator this channel does not yet model",
            "explicit second derivative this channel does not yet model",
        ] {
            assert!(
                !msg.contains(marker),
                "#2500: a ThresholdGate outer solve must not abort on an unmodelled sparse \
                 log-strength operator ({marker}); got: {msg}"
            );
        }
    }
}

/// #2500 GATE 8 — the measurement that justifies refusing the exact-A override
/// for a ThresholdGate fit: on the SAME inverse, the dense θ-adjoint
/// reconstruction and the production one disagree by MORE than the entry's own
/// magnitude. They are not interchangeable for this family, so a fit of it must
/// not have its logdet channels overwritten with the exact-A pair (GATE 9).
///
/// `logdet_theta_adjoint_dense` carries the softmax entropy Gershgorin majorizer,
/// the ordered-Beta–Bernoulli Patch-D cross-row adjoint and the periodic-ARD
/// majorizer diagonal, and is documented as self-checked against the production
/// `logdet_theta_adjoint` for softmax. It carries no per-atom-logistic GATE leg,
/// which is what a threshold-gate row needs — the same limitation
/// `third_order_forward_sensitivity_hessian` already refuses on. Against a
/// central finite difference of `½(log|A| − log|A_tt|)` in a logit the dense Γ
/// read `1.373e-1` where the FD read `3.521e-1`, with a sign flip on the next
/// logit, so this is not a tolerance question.
#[test]
fn dense_theta_adjoint_is_not_interchangeable_for_a_threshold_gate_2500() {
    for straddle in [false, true] {
        let (term, target, rho) = threshold_gate_tiny_fixture(straddle);
        let (_loss, cache) = frozen_cache(&term, &target, &rho);
        let solver = crate::manifold::arrow_solver::DeflatedArrowSolver::plain(&cache);
        let production = term
            .logdet_theta_adjoint(&rho, &cache, &solver)
            .expect("production theta adjoint");
        let g = term
            .materialize_joint_inverse(&cache, &solver)
            .expect("joint inverse");
        let dense = term
            .logdet_theta_adjoint_dense(
                &rho,
                &cache,
                &g,
                ThetaAdjointDhChannel::All,
                false,
                false,
                None,
            )
            .expect("dense theta adjoint");
        // Per ENTRY: an error exceeding that entry's own magnitude means the dense
        // value is not even the right scale there, let alone a usable substitute.
        let worst = production
            .t
            .iter()
            .zip(dense.t.iter())
            .filter(|(p, _)| p.abs() > 1.0e-3)
            .map(|(p, d)| (p - d).abs() / p.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst > 1.0,
            "#2500 (straddle={straddle}): refusing the exact-A override for this family is \
             only justified if the two θ-adjoints genuinely disagree; worst per-entry \
             relative gap = {worst:.3}"
        );
    }
}

/// #2500 GATE 8b — the leg of the production θ-adjoint that the ThresholdGate
/// gradient falls back to and that IS exact: `coordinate_block_logdet_theta_adjoint`
/// reproduces a central finite difference of the per-row undamped factors' own
/// log-determinant on every free logit.
///
/// Convention, measured rather than read: the returned vector is
/// `∂(Σ_i log|H_tt^(i)|)/∂θ` with NO leading ½, even though the function's own doc
/// describes it as the derivative of `½ Σ_i log|H_tt^(i)|`. Halving the reference
/// puts every entry off by exactly a factor of two.
///
/// Measured on this fixture the analytic/FD ratio is `1.0000` on every entry
/// checked, so this is a tight gate rather than a loose one, and it pins the half
/// of the fallback that is load-bearing for the row-local prior curvature this
/// issue added.
#[test]
fn threshold_gate_coordinate_block_theta_adjoint_matches_finite_difference_2500() {
    for straddle in [false, true] {
        let (term, target, rho) = threshold_gate_tiny_fixture(straddle);
        let (_loss, cache) = frozen_cache(&term, &target, &rho);
        let coord = term
            .coordinate_block_logdet_theta_adjoint(
                &rho,
                &cache,
                crate::manifold::EvidenceOperator::Majorizer,
                None,
            )
            .expect("coordinate-block theta adjoint");
        let base_deflated = deflated_direction_count(&term, &cache);
        let block_logdet = |t: &SaeManifoldTerm| -> (f64, usize) {
            let (_l, c) = frozen_cache(t, &target, &rho);
            let mut acc = 0.0_f64;
            for row in 0..t.n_obs() {
                let factor = c.undamped_factor(row);
                for d in 0..c.row_dims[row] {
                    acc += 2.0 * factor[[d, d]].ln();
                }
            }
            (acc, deflated_direction_count(t, &c))
        };
        let h = 1.0e-6;
        let mut worst = 0.0_f64;
        let mut label = String::new();
        let mut checked = 0usize;
        for (row, atom, slot) in logit_slots(&term, &cache) {
            let mut plus = term.clone();
            plus.assignment.logits[[row, atom]] += h;
            let mut minus = term.clone();
            minus.assignment.logits[[row, atom]] -= h;
            let (lp, dp) = block_logdet(&plus);
            let (lm, dm) = block_logdet(&minus);
            // A central difference across a change in the deflated dimension
            // differences two different operators, not a derivative.
            if dp != base_deflated || dm != base_deflated {
                continue;
            }
            let fd = (lp - lm) / (2.0 * h);
            let analytic = coord.t[slot];
            let tol = 1.0e-6 + 1.0e-4 * analytic.abs().max(fd.abs());
            let ratio = (analytic - fd).abs() / tol;
            if ratio > worst {
                worst = ratio;
                label = format!(
                    "row {row} atom {atom}: analytic={analytic:.9e} fd={fd:.9e} tol={tol:.3e}"
                );
            }
            checked += 1;
        }
        assert!(
            checked >= 4,
            "#2500 (straddle={straddle}): the FD gate must reach at least four logits on a \
             fixed deflation stratum; checked={checked}"
        );
        assert!(
            worst <= 1.0,
            "#2500 (straddle={straddle}): the coordinate-block theta-adjoint must \
             differentiate the per-row log|H_tt| it is defined as; worst normalized error \
             {worst:.3} at {label}"
        );
    }
}

/// #2330/#2336 FALSIFICATION GATE - is `A` really `d2L/dtheta2`?
///
/// The `IndefiniteObservedInformation` refusal
/// (`SaeManifoldTerm::exact_observed_information_log_dets`) reads the spectrum of
/// the dense `A = B + dC` that `materialize_exact_hessian_dense` builds column by
/// column from the production `apply_exact_hessian`. Every consumer of that
/// refusal - the `+inf` probe pricing in `outer_objective`, the criterion value
/// path, and the IFT adjoint's `A^-1` - inherits the claim that `A` IS the second
/// derivative of the penalized objective. Nothing gated that claim DIRECTLY: the
/// #2500 gates above finite-difference the log-det TRACE and the theta-adjoint in
/// `rho`, i.e. they check `dA/drho` and contractions of `A^-1`, never `A` itself
/// against `dg/dtheta`.
///
/// This gate closes that, and it is deliberately aimed at its own author's
/// conclusion. "The mode converged but `A` is indefinite" admits exactly two
/// readings, with OPPOSITE fixes:
///
/// * the inner solve really does stop at a saddle of `L` - a defect UPSTREAM, in
///   a majorized solver whose PD model `B` structurally cannot see a negative
///   direction, curable only by a negative-curvature escape;
/// * or `A` is not the curvature of the thing whose gradient the solve drove to
///   zero - a defect AT the refusal site, curable there and only there.
///
/// `g = (gt, gb)` read off `assemble_arrow_schur` is the EXACT KKT gradient: the
/// same vector `terminal_exact_newton_polish` negates to form its Newton
/// right-hand side, with no majorization anywhere in it. So a central difference
/// of `g` along `v` is `dg/dtheta . v` on the nose. If it disagrees with `A.v`,
/// the second reading is the true one and the converged-but-indefinite verdict is
/// an artefact of the refusal site.
///
/// The BELOW-THRESHOLD arm is used on purpose: it is the deflation-free stratum
/// (asserted, not assumed), so no per-row direction changes discrete class
/// between the `+/-h` endpoints and the central difference stays a difference of
/// ONE smooth branch - the same branch-identity discipline the `rho`-FD gates in
/// this file already apply. `h = 1e-5` is the step those gates use.
#[test]
fn dense_exact_a_matches_finite_difference_of_the_kkt_gradient_2330() {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let (_loss, cache) = frozen_cache(&term, &target, &rho);
    assert_eq!(
        deflated_direction_count(&term, &cache),
        0,
        "#2330 FD gate: the below-threshold arm must be deflation-free, or the +/-h \
         endpoints can straddle a discrete deflation change and the central \
         difference stops being a difference of one smooth branch"
    );

    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("dense exact A at the frozen fixture mode");
    let total_t = cache.delta_t_len();
    let k = cache.k;
    let dim = total_t + k;
    assert_eq!(
        a.nrows(),
        dim,
        "#2330 FD gate: dense A must be (total_t + k) square before it can be compared \
         against the (gt, gb) gradient layout"
    );

    // The exact KKT gradient at whatever state `t` is in, concatenated in the
    // SAME (t, beta) order the Newton right-hand side uses in
    // `terminal_exact_newton_polish`. `&mut` because `assemble_arrow_schur`
    // takes `&mut self` (it refreshes row-layout state); this is only ever
    // called on the locally-owned `moved` clone below, and the analytic side is
    // already an owned `Array2`, so no borrow of `term` is live across it.
    let gradient = |t: &mut SaeManifoldTerm| -> Array1<f64> {
        let sys = t
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly at the finite-difference endpoint");
        let mut g = Array1::<f64>::zeros(dim);
        let mut offset = 0usize;
        for row in &sys.rows {
            for (axis, &value) in row.gt.iter().enumerate() {
                g[offset + axis] = value;
            }
            offset += row.gt.len();
        }
        assert_eq!(
            offset, total_t,
            "#2330 FD gate: concatenated per-row gt width must equal cache.delta_t_len(), \
             or the gradient and A are not in the same coordinates"
        );
        assert_eq!(
            sys.gb.len(),
            k,
            "#2330 FD gate: border gradient width must equal cache.k"
        );
        for (axis, &value) in sys.gb.iter().enumerate() {
            g[total_t + axis] = value;
        }
        g
    };

    let h = 1.0e-5_f64;
    // TWO directions, so the gate cannot pass by being blind in one block. A
    // coordinate-axis probe would test a single column of `A` and could miss an
    // entire mis-assembled sub-block, so both probes are dense and deterministic
    // and each leans on a different block.
    for (label, weight_t, weight_beta) in [
        ("coordinate-weighted", 1.0_f64, 0.25_f64),
        ("border-weighted", 0.25_f64, 1.0_f64),
    ] {
        let mut v_t = Array1::<f64>::zeros(total_t);
        let mut v_beta = Array1::<f64>::zeros(k);
        let mut v = Array1::<f64>::zeros(dim);
        for idx in 0..dim {
            let phase = 0.7 + 0.31 * (idx as f64);
            let raw = phase.sin() + 0.5 * (2.0 * phase).cos();
            let value = raw * if idx < total_t { weight_t } else { weight_beta };
            v[idx] = value;
        }
        let norm = v.dot(&v).sqrt();
        assert!(
            norm.is_finite() && norm > 0.0,
            "#2330 FD gate ({label}): probe direction must be finite and nonzero"
        );
        v.mapv_inplace(|value| value / norm);
        for idx in 0..total_t {
            v_t[idx] = v[idx];
        }
        for idx in 0..k {
            v_beta[idx] = v[total_t + idx];
        }

        let analytic = a.dot(&v);

        // `apply_newton_step` REFUSES a non-positive `step_size`
        // (`apply_newton_step_impl_with_parallelism`: "step_size must be finite
        // and positive"). It applies `step_size · delta`, so the `-h` endpoint
        // has to negate the DIRECTION and keep the step positive. Passing
        // `sign * h` made the minus endpoint an `Err` that the `expect` turned
        // into a panic BEFORE any finite difference was formed — this gate has
        // never reached its own comparison. Measured on `bc5d6bdde`: it dies at
        // this line with `got -0.00001`.
        let endpoint = |sign: f64| -> Array1<f64> {
            let mut moved = term.clone();
            let signed_t = v_t.mapv(|value| sign * value);
            let signed_beta = v_beta.mapv(|value| sign * value);
            moved
                .apply_newton_step(signed_t.view(), signed_beta.view(), h)
                .expect("finite-difference endpoint step");
            gradient(&mut moved)
        };
        let g_plus = endpoint(1.0);
        let g_minus = endpoint(-1.0);
        let fd = (&g_plus - &g_minus).mapv(|delta| delta / (2.0 * h));

        // A zero-vs-zero comparison passes any tolerance. The analytic side must
        // carry real curvature along this direction before the assertion below
        // means anything at all.
        let analytic_norm = analytic.dot(&analytic).sqrt();
        let fd_norm = fd.dot(&fd).sqrt();
        assert!(
            analytic_norm > 1.0e-6,
            "#2330 FD gate ({label}): |A.v|={analytic_norm:.6e} is too small for this \
             comparison to be a gate rather than a zero-vs-zero tautology"
        );

        let mut worst = 0.0_f64;
        let mut worst_idx = 0usize;
        for idx in 0..dim {
            let scale = analytic[idx].abs().max(fd[idx].abs()).max(1.0e-8);
            let relative = (analytic[idx] - fd[idx]).abs() / scale;
            if relative > worst {
                worst = relative;
                worst_idx = idx;
            }
        }
        // Report the directional pair as well as the worst component: a ratio of
        // -1 is a SIGN convention and a ratio of 2 is a majorization leak, and
        // neither is distinguishable from noise if only a magnitude is printed.
        let directional = analytic.dot(&v);
        let directional_fd = fd.dot(&v);
        assert!(
            worst <= 1.0e-4,
            "#2330: the dense exact A is NOT the second derivative of the penalized \
             objective whose gradient the inner solve drives to zero. Worst component \
             relative error {worst:.6e} at index {worst_idx} (A.v={:.9e}, FD={:.9e}); \
             |A.v|={analytic_norm:.6e} |FD|={fd_norm:.6e}; v'Av={directional:.9e} vs \
             v'(dg/dtheta)v={directional_fd:.9e} (ratio {:.6e}). If this fires, the \
             IndefiniteObservedInformation refusal is measuring the wrong operator, the \
             converged-but-indefinite verdict is an artefact of the refusal site rather \
             than a saddle upstream, and the negative-curvature escape is the WRONG fix. \
             Probe: {label}, h={h:.3e}",
            analytic[worst_idx],
            fd[worst_idx],
            directional_fd / directional,
        );
    }
}
