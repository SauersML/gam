#![cfg(test)]
//! #2933 F36 — the joint fitted-response divergence `tr(∂f̂/∂y)` of the SAE
//! reconstruction against direct re-solves of the perturbed inner problem.
//!
//! The divergence of a penalized fit is the trace of its response Jacobian
//! `R = ∂f̂/∂y`, and the oracle here measures that Jacobian directly. It perturbs
//! one target entry by `±h`, drives the penalized stationarity
//! `∇L(θ; y ± h·e) = 0` from the fitted state to its roundoff plateau with exact-A
//! Newton steps, and differences every fitted value, which gives one column of
//! `R`. The root of that loop is fixed by the assembled gradient alone; the exact
//! stationarity solve only chooses the path to it, so the oracle does not score
//! the operator against itself. The production inner driver stops inside a
//! `1e-5` relative KKT band, which is coarser than the response being
//! differenced, so the re-solve runs to roundoff instead. Collapse-prevention
//! gates stay frozen at the fitted state: the same basin and the same objective
//! the divergence describes.
//!
//! The same Jacobian prices the dispersion's residual degrees of freedom
//! `‖I − R‖²_F` (#2933 F40), so `tr R` and `‖I − R‖²_F` are both checked against
//! one measured response rather than against each other.
//!
//! The per-axis completion this replaced summed scalar fractions
//! `htt/(htt+c+V'') − htt/(htt+V'')` from diagonal curvatures, so an off-diagonal
//! residual curvature `A = [[1, c], [c, 1]]` returned zero where the trace
//! correction is `2/(1−c²) − 2`. The fixtures carry what that routine could not
//! see: a two-dimensional torus chart (mixed coordinate curvature), two atoms on
//! the same rows (cross-atom coupling), free gate logits, an ARD block on every
//! atom, and ARD rows on the concave side of a periodic axis.

use super::tests_fitted_response_frames_2933::rademacher_quadratic_form_variance;
use super::*;
use crate::basis::{AmbientSphereHarmonicEvaluator, TorusHarmonicEvaluator};
use crate::manifold::arrow_solver::{SaeArrowVector, SaeLocalRowVar};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2, array};
use std::sync::Arc;

/// Largest number of exact-A Newton steps one re-solve may take. From a state
/// one `h` away from a root, quadratic convergence reaches roundoff in a few.
const ROOT_POLISH_STEPS: usize = 40;

/// Central-difference step on a target entry. The targets are of order one, so
/// the truncation error `h²·f'''/6` sits far below the tolerance below.
const FD_STEP: f64 = 1.0e-4;

/// Relative agreement required between a priced quantity and its re-solved
/// counterpart. The truncation error at `FD_STEP` and the roundoff of a root
/// driven to its plateau are both orders of magnitude smaller.
const RELATIVE_TOLERANCE: f64 = 1.0e-4;

/// A root is admitted only once its gradient has reached the arithmetic floor
/// of an order-one objective on a few dozen scalar observations.
const ROOT_GRADIENT_CEILING: f64 = 1.0e-10;

fn gradient(system: &ArrowSchurSystem) -> SaeArrowVector {
    SaeArrowVector {
        t: system
            .rows
            .iter()
            .flat_map(|row| row.gt.iter().copied())
            .collect(),
        beta: system.gb.clone(),
    }
}

/// The assembled penalized gradient `∇L(θ; target)` and its norm at the term's
/// current state.
fn assembled_gradient(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> (ArrowSchurSystem, SaeArrowVector, f64) {
    let system = term
        .assemble_arrow_schur(target, rho, None)
        .expect("the arrow system assembles at every re-solve iterate");
    let g = gradient(&system);
    let norm = (g.t.dot(&g.t) + g.beta.dot(&g.beta)).sqrt();
    (system, g, norm)
}

/// Drive `∇L(θ; target) = 0` from the term's current state by exact-A Newton
/// steps, and return the undamped evidence factorization at the root with the
/// root's gradient norm.
///
/// The step `Δ = A⁺g` is a descent direction for `½‖g‖²`: its slope along `−Δ` is
/// `−gᵀA·A⁺g = −‖P_range(A) g‖²`. So halving the step until `‖g‖` falls always
/// terminates away from an in-band gradient. A state outside Newton's quadratic
/// basin needs that. The torus fixture's first undamped step raised `‖g‖` from
/// `2.9e-3` to `2.3e-2` (job 1117483), and at log α = −1 the undamped iteration
/// never reached the ceiling in 40 steps (job 1143891). The root does not depend
/// on the path, since it is fixed by the assembled gradient alone. The plateau test
/// engages only under the root ceiling: there the iteration stops once the norm no
/// longer halves.
fn polish_to_root(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> (ArrowFactorCache, f64) {
    let options = term.evidence_factor_options();
    let gates = term.collapse_prevention_gates();
    let mut previous = f64::INFINITY;
    let mut history: Vec<f64> = Vec::with_capacity(ROOT_POLISH_STEPS);
    for _ in 0..ROOT_POLISH_STEPS {
        let (system, g, norm) = assembled_gradient(term, target, rho);
        history.push(norm);
        let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
            .expect("the majorizer factors undamped at every re-solve iterate");
        if norm <= ROOT_GRADIENT_CEILING && !(norm < 0.5 * previous) {
            return (cache, norm);
        }
        let step = term
            .solve_exact_stationarity(rho, target, &cache, &g)
            .expect("the exact stationarity pseudoinverse solves at every re-solve iterate");
        let mut fraction = 1.0_f64;
        let mut accepted = None;
        // A fraction below 2^-53 changes no digit of an order-one coordinate.
        for _ in 0..f64::MANTISSA_DIGITS {
            let mut trial = term.clone();
            trial.declare_collapse_prevention_gates(&gates);
            trial
                .apply_newton_step((-&step.t).view(), (-&step.beta).view(), fraction)
                .expect("the exact Newton step applies to the term state");
            let (_, _, trial_norm) = assembled_gradient(&mut trial, target, rho);
            if trial_norm < norm {
                accepted = Some(trial);
                break;
            }
            fraction *= 0.5;
        }
        match accepted {
            Some(trial) => *term = trial,
            // Under the root ceiling no fraction of the step reduces `‖g‖`: that is
            // the roundoff plateau itself (job 1145056 read `2.8e-16` there).
            None if norm <= ROOT_GRADIENT_CEILING => return (cache, norm),
            None => panic!(
                "no fraction of the exact-A Newton step reduces ‖g‖={norm:.3e}; ‖g‖ history {}",
                scientific(&history)
            ),
        }
        previous = norm;
    }
    panic!(
        "the exact-A Newton re-solve did not reach its roundoff plateau in {ROOT_POLISH_STEPS} steps; \
         ‖g‖ history {}",
        scientific(&history)
    );
}

fn scientific(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.3e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The re-solved response Jacobian and the two functionals of it the fit prices.
struct ResolvedResponse {
    /// `R` over the raw output scalars.
    jacobian: Array2<f64>,
    /// `tr R`.
    trace: f64,
    /// `‖I − R‖²_F` over the raw output scalars.
    residual_dof: f64,
}

/// `R = ∂f̂/∂y` by central differences of fits re-solved to their roots, over the
/// `n·p` target scalars in row-major order: column `j` is the response of every
/// fitted scalar to target scalar `j`.
fn resolved_response(
    base: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ResolvedResponse {
    let (n, p) = target.dim();
    let scalars = n * p;
    let mut jacobian = Array2::<f64>::zeros((scalars, scalars));
    for row in 0..n {
        for col in 0..p {
            let mut fitted: Vec<Array2<f64>> = Vec::with_capacity(2);
            for sign in [1.0_f64, -1.0] {
                // A clone re-derives the collapse-prevention gates; the response is taken
                // under the gates the fitted state declared.
                let mut term = base.clone();
                term.declare_collapse_prevention_gates(&base.collapse_prevention_gates());
                let mut perturbed = target.clone();
                perturbed[[row, col]] += sign * FD_STEP;
                let (_, norm) = polish_to_root(&mut term, perturbed.view(), rho);
                assert!(
                    norm <= ROOT_GRADIENT_CEILING,
                    "the re-solve at entry ({row}, {col}), side {sign} stalled at ‖g‖={norm:.3e}"
                );
                let residual = term
                    .reconstruction_residual(perturbed.view(), rho)
                    .expect("the re-solved fit has a residual");
                fitted.push(&residual + &perturbed);
            }
            let column = (&fitted[0] - &fitted[1]) / (2.0 * FD_STEP);
            for (index, &value) in column.iter().enumerate() {
                jacobian[[index, row * p + col]] = value;
            }
        }
    }
    let trace = (0..scalars).map(|index| jacobian[[index, index]]).sum();
    let residual_dof = jacobian
        .indexed_iter()
        .map(|((i, j), &value)| {
            let identity = if i == j { 1.0 } else { 0.0 };
            (identity - value) * (identity - value)
        })
        .sum();
    ResolvedResponse {
        jacobian,
        trace,
        residual_dof,
    }
}

fn assert_agrees(label: &str, value: f64, resolved: f64) {
    let gap = (value - resolved).abs();
    assert!(
        gap <= RELATIVE_TOLERANCE * resolved.abs().max(1.0),
        "{label}: {value:.9e} against the re-solved response {resolved:.9e} (gap {gap:.3e})"
    );
}

/// The divergence, its raw residual dof, and the scale `reconstruction_dispersion`
/// divides the residual energy by, each against the re-solved response.
fn assert_prices_the_resolved_response(
    label: &str,
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    resolved: &ResolvedResponse,
) {
    assert!(
        !term.frames_active(),
        "{label}: the fixture profiles no decoder frame, so no frame dimension enters the residual dof"
    );
    assert!(
        term.row_loss_weights
            .as_deref()
            .is_none_or(|weights| weights.iter().all(|&weight| weight > 0.0)),
        "{label}: every row carries positive weight, so every raw scalar counts"
    );
    let response = term
        .fitted_response_divergence(target.view(), rho, cache)
        .expect("the fixture admits the exact spectral divergence");
    assert!(
        matches!(response.estimator, FittedResponseDivergenceEstimator::ExactSpectral),
        "{label}: a fixture this small is on the exact spectral route, got {:?}",
        response.estimator
    );
    let loss = term
        .loss(target.view(), rho)
        .expect("the fitted state has a loss");
    let residual = term
        .reconstruction_residual(target.view(), rho)
        .expect("the fitted state has a residual");
    let dispersion = term
        .reconstruction_dispersion(&loss, cache, rho, residual.view())
        .expect("the fitted state prices a dispersion");
    // With no frame and every row weighted, the raw output noise variance is
    // `RSS/‖I − R‖²_F` over the raw scalars (#2933 F40), and the dispersion adds
    // no selection degrees of freedom (#2933 F37).
    let rss: f64 = residual.iter().map(|value| value * value).sum();
    let priced_residual_dof = rss / dispersion.raw_output_noise_variance;
    eprintln!(
        "[#2933 F36 {label}] divergence={:.9e} tr R={:.9e}; raw residual dof={:.9e} priced \
         {priced_residual_dof:.9e} ‖I − R‖²_F={:.9e}",
        response.divergence, resolved.trace, response.raw_residual_dof, resolved.residual_dof
    );
    let scalars = target.len() as f64;
    assert!(
        resolved.trace > 1.0 && resolved.residual_dof > 1.0 && resolved.trace < scalars - 1.0,
        "{label}: the re-solved response (tr R={}, ‖I − R‖²_F={}) must be material and leave \
         residual dof for the comparison to mean anything",
        resolved.trace,
        resolved.residual_dof
    );
    assert_agrees(
        &format!("{label} exact spectral divergence"),
        response.divergence,
        resolved.trace,
    );
    assert_agrees(
        &format!("{label} raw residual dof"),
        response.raw_residual_dof,
        resolved.residual_dof,
    );
    assert_agrees(
        &format!("{label} residual dof reconstruction_dispersion prices"),
        priced_residual_dof,
        resolved.residual_dof,
    );
}

/// The softmax witness at its converged inner state, before the root polish:
/// two circle atoms on every row, gated by one softmax, so the logits are free
/// and couple the atoms through the simplex.
fn softmax_two_circle_fitted() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (mut term, target, mut rho) = crate::manifold::tests::gamma_fd_tiny_fixture();
    // The moderate-penalty basin where this fixture's exact observed information
    // is positive definite and its residual curvature is live.
    rho.log_lambda_sparse = 0.0;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -1.0;
        }
    }
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    )
    .expect("the softmax fixture converges in its moderate-penalty basin");
    let gates = term.collapse_prevention_gates();
    term.declare_collapse_prevention_gates(&gates);
    (term, target, rho)
}

fn softmax_two_circle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, rho) = softmax_two_circle_fitted();
    let (cache, norm) = polish_to_root(&mut term, target.view(), &rho);
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "the softmax fixture's root stalled at ‖g‖={norm:.3e}"
    );
    (term, target, rho, cache)
}

/// The ordered Beta--Bernoulli witness at its fitted state, before the root
/// polish: a torus atom (two coupled chart axes, ARD on both, rows on both sides
/// of each axis) and a circle atom with ARD disabled, on the same rows with
/// independent free logits.
fn obb_torus_and_circle_fitted() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 12usize;
    let p = 4usize;
    let torus = Arc::new(
        TorusHarmonicEvaluator::new(2, 1).expect("one harmonic per axis is a valid torus basis"),
    );
    let circle = Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("an odd harmonic count is a valid periodic basis"),
    );
    let torus_coords = Array2::<f64>::from_shape_fn((n, 2), |(row, axis)| {
        let (rate, offset) = [(0.137, 0.05), (0.241, 0.43)][axis];
        (row as f64 * rate + offset).rem_euclid(1.0)
    });
    let circle_coords =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (0.11 + row as f64 / n as f64).rem_euclid(1.0));
    let (torus_phi, torus_jet) = torus
        .evaluate(torus_coords.view())
        .expect("the torus coordinates are wrapped into the unit period");
    let (circle_phi, circle_jet) = circle
        .evaluate(circle_coords.view())
        .expect("the circle coordinates are wrapped into the unit period");
    let torus_width = torus_phi.ncols();
    let circle_width = circle_phi.ncols();
    let torus_decoder = Array2::<f64>::from_shape_fn((torus_width, p), |(basis, out)| {
        0.35 * ((3 * basis + out) as f64 * 0.7 + 0.2).sin()
    });
    let circle_decoder = Array2::<f64>::from_shape_fn((circle_width, p), |(basis, out)| {
        0.3 * ((2 * basis + 3 * out) as f64 * 0.9 + 0.5).cos()
    });
    // The signal is several times the off-model term, so the data pin every
    // coordinate near its generating value and the rows stay spread around both
    // circles, where a comparable ARD precision leaves live negative curvature on
    // each axis's concave half. A weak signal let the fit gather every row on the
    // convex half (job 1129926).
    let mut target = torus_phi.dot(&torus_decoder) * 3.0 + circle_phi.dot(&circle_decoder) * 4.0;
    for row in 0..n {
        for out in 0..p {
            target[[row, out]] += 0.05 * (1.7 * row as f64 + 2.3 * out as f64).sin();
        }
    }
    let torus_atom = SaeManifoldAtom::new_with_provided_function_gram(
        "torus".to_string(),
        SaeAtomBasisKind::Torus,
        2,
        torus_phi,
        torus_jet,
        torus_decoder,
        Array2::<f64>::eye(torus_width),
    )
    .expect("torus atom: basis width, latent dimension and decoder shape agree")
    .with_basis_second_jet(torus);
    let circle_atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        circle_phi,
        circle_jet,
        circle_decoder,
        Array2::<f64>::eye(circle_width),
    )
    .expect("circle atom: basis width, latent dimension and decoder shape agree")
    .with_basis_second_jet(circle);
    let logits = Array2::<f64>::from_shape_fn((n, 2), |(row, atom)| {
        if atom == 0 { 1.2 } else { 0.4 + 0.05 * row as f64 }
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![torus_coords, circle_coords],
        vec![
            LatentManifold::Product(vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ]),
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("assignment: one logit column and one coordinate block per atom");
    let mut term = SaeManifoldTerm::new(vec![torus_atom, circle_atom], assignment)
        .expect("term: every atom's basis width matches its assignment block");
    // ARD at one precision on every periodic axis, the torus's two and the circle's, so the
    // rows on the concave side of each axis carry a live negative prior curvature. #2822 —
    // the circle used to carry no ARD block. With the block it carries the torus's log α = 1,
    // and two changes keep the fixture's two premises, measured on a grid (jobs 1255625,
    // 1258829, 1262175): the root the exact-A re-solve reaches, and a response material
    // enough to leave residual dof.
    // - The circle's signal is four times its decoder, the same order as the torus's three.
    //   At twice its decoder the circle's concave-side curvature dominated its rows, and the
    //   re-solve stalled at ‖g‖ ≈ 0.19.
    // - There are four outputs, not three. At three, ‖I − R‖²_F was 1.18 on main and 0.90 to
    //   0.95 at circle log α −4, 0 and 0.5, under the premise's 1. At four it is 1.86.
    // - The premises are asserted where they are used: the re-solve's root ceiling, and the
    //   material-response check.
    let mut rho = SaeManifoldRho::new(0.0, -1.0, vec![array![1.0, 1.0], array![1.0]]);
    term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 80, 1.0, 1.0e-7, 1.0e-7)
        .expect("the torus and circle fixture fits");
    let gates = term.collapse_prevention_gates();
    term.declare_collapse_prevention_gates(&gates);
    (term, target, rho)
}

fn obb_torus_and_circle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, rho) = obb_torus_and_circle_fitted();
    let n = target.nrows();
    let (cache, norm) = polish_to_root(&mut term, target.view(), &rho);
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "the torus and circle fixture's root stalled at ‖g‖={norm:.3e}"
    );
    let concave_rows = (0..n)
        .filter(|&row| {
            (0..2).any(|axis| (std::f64::consts::TAU * term.assignment.coords[0].row(row)[axis]).cos() < 0.0)
        })
        .count();
    eprintln!(
        "[#2933 F36 torus] root ‖g‖={norm:.3e} log α={:?} torus coordinates {:?}, {concave_rows} of {n} \
         rows on a concave half",
        rho.log_ard[0],
        term.assignment.coords[0].as_matrix()
    );
    assert!(
        concave_rows > 0,
        "the fixture must place ARD rows on the concave side of a periodic axis"
    );
    (term, target, rho, cache)
}

#[test]
fn softmax_divergence_matches_the_resolved_response_2933() {
    let (term, target, rho, cache) = softmax_two_circle_state();
    let resolved = resolved_response(&term, &target, &rho);
    assert_prices_the_resolved_response("softmax", &term, &target, &rho, &cache, &resolved);
}

#[test]
fn torus_and_circle_divergence_prices_the_dispersion_2933() {
    let (term, target, rho, cache) = obb_torus_and_circle_state();
    let resolved = resolved_response(&term, &target, &rho);
    assert_prices_the_resolved_response("torus", &term, &target, &rho, &cache, &resolved);
}

#[test]
fn hutchinson_divergence_brackets_the_resolved_response_2933() {
    let (mut term, target, rho, cache) = obb_torus_and_circle_state();
    let resolved = resolved_response(&term, &target, &rho);
    // No host memory admits the dense eigensystem, so the matrix-free estimator
    // must carry the divergence.
    term.host_available_bytes = 0;
    let response = term
        .fitted_response_divergence(target.view(), &rho, &cache)
        .expect("the matrix-free estimator solves where the dense route is refused");
    let FittedResponseDivergenceEstimator::Hutchinson { likelihood, .. } = response.estimator
    else {
        panic!(
            "without an admitted eigensystem the divergence must be the Hutchinson estimate, \
             got {:?}",
            response.estimator
        );
    };
    // The standard errors of the estimator at the probe count it chose, from the
    // re-solved response itself, so the bracket is not read off the estimate's own
    // sample spread.
    let probes = likelihood.probes;
    let complement = Array2::<f64>::eye(resolved.jacobian.nrows()) - &resolved.jacobian;
    let gram = complement.t().dot(&complement);
    let residual_dof_standard_error =
        (rademacher_quadratic_form_variance(&gram) / probes as f64).sqrt();
    let divergence_standard_error =
        (rademacher_quadratic_form_variance(&resolved.jacobian) / probes as f64).sqrt();
    eprintln!(
        "[#2933 F36 Hutchinson] probes={probes} divergence={:.9e} (sample se {:.3e}, resolved \
         se {divergence_standard_error:.3e}) resolved tr R={:.9e}; residual dof {:.9e} (sample \
         se {:.3e}, resolved se {residual_dof_standard_error:.3e}) resolved ‖I−R‖²={:.9e}",
        response.divergence,
        likelihood.divergence_standard_error,
        resolved.trace,
        response.likelihood_residual_dof,
        likelihood.residual_dof_standard_error,
        resolved.residual_dof
    );
    assert!(
        probes >= 2
            && likelihood.residual_dof_standard_error.powi(2) <= 2.0 * likelihood.residual_dof,
        "the probes stopped at {probes} with a residual-dof Monte Carlo variance {:.3e} above \
         the 2ν̂ = {:.3e} the dispersion carries",
        likelihood.residual_dof_standard_error.powi(2),
        2.0 * likelihood.residual_dof
    );
    // Four standard errors of a fixed-seed estimate: a deterministic bracket whose
    // width is the estimator's own sampling error, not a tuned band.
    assert!(
        (response.likelihood_residual_dof - resolved.residual_dof).abs()
            <= 4.0 * residual_dof_standard_error,
        "the Hutchinson residual dof {} is more than four standard errors \
         ({residual_dof_standard_error:.3e}) from the re-solved ‖I − R‖²_F {}",
        response.likelihood_residual_dof,
        resolved.residual_dof
    );
    assert!(
        (response.divergence - resolved.trace).abs() <= 4.0 * divergence_standard_error,
        "the Hutchinson divergence {} is more than four standard errors \
         ({divergence_standard_error:.3e}) from the re-solved response trace {}",
        response.divergence,
        resolved.trace
    );
}

/// The assembled gradient in the joint `(t, β)` layout after the term's own
/// Newton-step apply (the retraction on constrained coordinates) along
/// `h·direction`.
fn retracted_gradient(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    direction: &Array1<f64>,
    total_t: usize,
    h: f64,
) -> Array1<f64> {
    let mut moved = term.clone();
    moved.declare_collapse_prevention_gates(&term.collapse_prevention_gates());
    let delta = direction * h;
    moved
        .apply_newton_step(
            delta.slice(ndarray::s![..total_t]),
            delta.slice(ndarray::s![total_t..]),
            1.0,
        )
        .expect("the retracted step applies");
    let system = moved
        .assemble_arrow_schur(target, rho, None)
        .expect("the arrow system assembles at the retracted state");
    let g = gradient(&system);
    Array1::from_iter(g.t.iter().chain(g.beta.iter()).copied())
}

/// The sphere witness at its converged inner state: a degree-one ambient
/// harmonic atom on `S²`, stored as unit 3-vectors, and a circle atom, both on
/// every row with free softmax logits. The target is off the model, so the
/// residual curvature along each sphere normal is live.
fn softmax_sphere_and_circle_fitted() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let sphere = Arc::new(
        AmbientSphereHarmonicEvaluator::new(1).expect("degree one is a valid ambient sphere basis"),
    );
    let circle = Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("an odd harmonic count is a valid periodic basis"),
    );
    let sphere_coords = Array2::<f64>::from_shape_fn((n, 3), |(row, axis)| {
        let fraction = row as f64 / n as f64;
        let latitude = -0.6 + 1.2 * fraction;
        let longitude = 0.3 + 0.9 * std::f64::consts::TAU * fraction;
        [
            latitude.cos() * longitude.cos(),
            latitude.cos() * longitude.sin(),
            latitude.sin(),
        ][axis]
    });
    let circle_coords =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (0.17 + row as f64 / n as f64).rem_euclid(1.0));
    let (sphere_phi, sphere_jet) = sphere
        .evaluate(sphere_coords.view())
        .expect("the sphere coordinates are unit vectors");
    let (circle_phi, circle_jet) = circle
        .evaluate(circle_coords.view())
        .expect("the circle coordinates are wrapped into the unit period");
    let sphere_width = sphere_phi.ncols();
    let circle_width = circle_phi.ncols();
    let sphere_decoder = Array2::<f64>::from_shape_fn((sphere_width, p), |(basis, out)| {
        0.4 * ((2 * basis + out) as f64 * 0.8 + 0.3).sin()
    });
    let circle_decoder = Array2::<f64>::from_shape_fn((circle_width, p), |(basis, out)| {
        0.3 * ((basis + 2 * out) as f64 * 1.1 + 0.4).cos()
    });
    let mut target = sphere_phi.dot(&sphere_decoder) * 0.6 + circle_phi.dot(&circle_decoder) * 0.4;
    for row in 0..n {
        for out in 0..p {
            target[[row, out]] += 0.25 * (1.3 * row as f64 + 1.9 * out as f64).cos();
        }
    }
    let sphere_atom = SaeManifoldAtom::new_with_provided_function_gram(
        "sphere".to_string(),
        SaeAtomBasisKind::Sphere,
        3,
        sphere_phi,
        sphere_jet,
        sphere_decoder,
        Array2::<f64>::eye(sphere_width),
    )
    .expect("sphere atom: basis width, latent dimension and decoder shape agree")
    .with_basis_second_jet(sphere);
    let circle_atom = SaeManifoldAtom::new_with_provided_function_gram(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        circle_phi,
        circle_jet,
        circle_decoder,
        Array2::<f64>::eye(circle_width),
    )
    .expect("circle atom: basis width, latent dimension and decoder shape agree")
    .with_basis_second_jet(circle);
    let logits = Array2::<f64>::from_shape_fn((n, 2), |(row, atom)| {
        if atom == 0 { 0.4 + 0.1 * (row % 3) as f64 } else { -0.2 }
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![sphere_coords, circle_coords],
        vec![
            LatentManifold::Sphere { dim: 3 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.9),
    )
    .expect("assignment: one logit column and one coordinate block per atom");
    let mut term = SaeManifoldTerm::new(vec![sphere_atom, circle_atom], assignment)
        .expect("term: every atom's basis width matches its assignment block");
    // ARD on the sphere's ambient axes, so its constrained-support prior enters, and on
    // the circle axis.
    let rho = SaeManifoldRho::new(0.0, -1.0, vec![array![-1.0, -1.0, -1.0], array![-1.0]]);
    let (criterion, _, _) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("the sphere fixture converges");
    eprintln!("[#2933 F36 sphere] criterion V={criterion:.12e}");
    let gates = term.collapse_prevention_gates();
    term.declare_collapse_prevention_gates(&gates);
    (term, target, rho)
}

fn softmax_sphere_and_circle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, ArrowFactorCache) {
    let (mut term, target, rho) = softmax_sphere_and_circle_fitted();
    let (cache, norm) = polish_to_root(&mut term, target.view(), &rho);
    assert!(
        norm <= ROOT_GRADIENT_CEILING,
        "the sphere fixture's root stalled at ‖g‖={norm:.3e}"
    );
    (term, target, rho, cache)
}

/// The joint slot of the sphere atom's first ambient axis in `row`, read off the
/// row layout independently of the production projector.
fn sphere_block_start(term: &SaeManifoldTerm, cache: &ArrowFactorCache, row: usize) -> usize {
    let vars = term
        .row_vars_for_row_dim(row, cache.row_dims[row])
        .expect("the row layout names its variables");
    let local = vars
        .iter()
        .position(|var| matches!(var, SaeLocalRowVar::Coord { atom: 0, axis: 0 }))
        .expect("every row holds the sphere atom's coordinates");
    for axis in 1..3 {
        assert!(
            matches!(vars[local + axis], SaeLocalRowVar::Coord { atom: 0, axis: other } if other == axis),
            "row {row} holds the sphere axes in contiguous slots"
        );
    }
    cache.row_offsets[row] + local
}

/// Largest entry of `A·t − t` over every sphere normal `t` a direction of `A`
/// may couple, against `‖A‖max`. The operator is assembled from applies, so its
/// arithmetic floor is of order `dim·ε·‖A‖`; `1e-10·‖A‖max` sits orders above it
/// and orders below any live residual curvature.
const NORMAL_PIN_RELATIVE_FLOOR: f64 = 1.0e-10;

#[test]
fn sphere_normal_keeps_only_its_pin_in_the_observed_information_2933() {
    let (term, target, rho, cache) = softmax_sphere_and_circle_state();
    let (a, _) = term
        .materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache)
        .expect("the sphere fixture materializes A");
    let dim = a.nrows();
    let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for row in 0..target.nrows() {
        let start = sphere_block_start(&term, &cache, row);
        let point = term.assignment.coords[0].row(row);
        let mut normal = Array1::<f64>::zeros(dim);
        for axis in 0..3 {
            normal[start + axis] = point[axis];
        }
        let leak = a.dot(&normal) - &normal;
        worst = leak.iter().fold(worst, |m, v| m.max(v.abs()));
    }
    let log_det = term
        .exact_observed_information_log_dets(&rho, target.view(), &cache)
        .expect("the sphere fixture prices log|A|");
    eprintln!(
        "[#2933 F36 sphere] max‖A·t − t‖∞={worst:.3e} ‖A‖max={scale:.3e} ½log|A|={:.12e}",
        0.5 * log_det
    );
    assert!(
        worst <= NORMAL_PIN_RELATIVE_FLOOR * scale,
        "a sphere normal must be an exact unit eigenvector of A with no coupling: \
         max‖A·t − t‖∞={worst:.3e} against ‖A‖max={scale:.3e}"
    );
}

#[test]
fn sphere_tangent_information_matches_the_retracted_gradient_2933() {
    let (term, target, rho, cache) = softmax_sphere_and_circle_state();
    let (a, _) = term
        .materialize_exact_hessian_dense_with_gap_border(&rho, target.view(), &cache)
        .expect("the sphere fixture materializes A");
    let total_t = cache.delta_t_len();
    let dim = a.nrows();
    for row in [0usize, target.nrows() / 2] {
        let start = sphere_block_start(&term, &cache, row);
        let point: Vec<f64> = term.assignment.coords[0].row(row).to_vec();
        let least_aligned = (0..3)
            .min_by(|&i, &j| point[i].abs().total_cmp(&point[j].abs()))
            .expect("three ambient axes");
        let mut first = [0.0_f64; 3];
        first[least_aligned] = 1.0;
        let along: f64 = (0..3).map(|axis| first[axis] * point[axis]).sum();
        for axis in 0..3 {
            first[axis] -= along * point[axis];
        }
        let length = first.iter().map(|value| value * value).sum::<f64>().sqrt();
        for value in first.iter_mut() {
            *value /= length;
        }
        let second = [
            point[1] * first[2] - point[2] * first[1],
            point[2] * first[0] - point[0] * first[2],
            point[0] * first[1] - point[1] * first[0],
        ];
        for tangent in [first, second] {
            let mut direction = Array1::<f64>::zeros(dim);
            for axis in 0..3 {
                direction[start + axis] = tangent[axis];
            }
            let predicted = a.dot(&direction);
            let plus = retracted_gradient(&term, target.view(), &rho, &direction, total_t, FD_STEP);
            let minus = retracted_gradient(&term, target.view(), &rho, &direction, total_t, -FD_STEP);
            let mut error = (&plus - &minus) / (2.0 * FD_STEP) - &predicted;
            // The assembled sphere gradient is tangent-projected at the moving point,
            // so its derivative carries a normal part; the Riemannian Hessian is the
            // tangent part at this row (Absil--Mahony--Sepulchre).
            let normal_part: f64 = (0..3).map(|axis| point[axis] * error[start + axis]).sum();
            for axis in 0..3 {
                error[start + axis] -= normal_part * point[axis];
            }
            let gap = error.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let size = predicted.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            eprintln!(
                "[#2933 F36 sphere FD] row {row} tangent {tangent:?}: ‖A·ξ‖max={size:.6e} \
                 tangent gap={gap:.3e}"
            );
            assert!(
                size > 1.0e-3,
                "row {row}: the tangent curvature {size:.3e} must be material for the comparison"
            );
            assert!(
                gap <= RELATIVE_TOLERANCE * size.max(1.0),
                "row {row}: A·ξ {size:.6e} disagrees with the retracted gradient's tangent \
                 derivative by {gap:.3e}"
            );
        }
    }
}

#[test]
fn sphere_divergence_matches_the_resolved_response_2933() {
    let (term, target, rho, cache) = softmax_sphere_and_circle_state();
    let resolved = resolved_response(&term, &target, &rho);
    assert_prices_the_resolved_response("sphere", &term, &target, &rho, &cache, &resolved);
}
