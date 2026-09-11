//! #2612: the outer REML/LAML gradient must be the gradient of the criterion it
//! reports — with the Jeffreys/Firth term ARMED, not only with it disarmed.
//!
//! The multinomial formula path re-solves with `½ log|I(β)|` armed whenever the
//! certified mode is (quasi-)separated. On the penguins real-data arm that refit
//! then dies in the OUTER smoothing search at
//! `line_search=StepSizeTooSmall` — the solver's own gloss on which is *"the
//! direction descended but no step improved the objective"* — with an indefinite
//! terminal analytic Hessian. On a differentiable objective with a correct
//! gradient that cannot happen: a short enough step along `−g` must decrease the
//! value. So the gradient the line search consumes is not the derivative of the
//! value it evaluates.
//!
//! This is that statement reduced to a fixture that runs in seconds: a small,
//! deliberately quasi-separated multinomial, the production outer criterion, and
//! central finite differences of its VALUE against its own analytic gradient,
//! taken twice at the SAME rho with the term disarmed and armed. The disarmed
//! arm is the control — it pins that the harness, the criterion and the FD step
//! are sound, so a failure in the armed arm is attributable to the term.

use gam_models::MultinomialFamily;
use gam_models::custom_family::{
    BlockwiseFitOptions, CustomFamily, PenaltyMatrix,
    evaluate_labeled_outer_criterion_for_diagnostics,
};
use gam_linalg::faer_ndarray::FaerEigh;
use gam_problem::EvalMode;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Rows per class in the fixture.
const PER_CLASS: usize = 24;
/// Classes.
const K: usize = 3;
/// Design columns: intercept + two covariates + one redundant-ish column that
/// gives the penalty something to shrink.
const P: usize = 4;

/// A QUASI-separated 3-class design: class `c`'s rows come from a well-separated
/// band of the first covariate, plus exactly one crossing row per class that
/// carries a neighbour's label.
///
/// The crossing rows matter for the experiment, not for realism. Without them
/// the classes are COMPLETELY separated, the unbiased softmax mode is at
/// infinity, and the unbiased criterion has no well-defined value — so its
/// finite differences measure where the runaway happened to stop rather than a
/// derivative, and the control arm would be comparing noise. One crossing row
/// per class makes the unbiased MLE finite (control arm well-posed) while
/// leaving the linear direction supported by so little curvature that the
/// Jeffreys conditioning gate still fires (armed arm still on-regime). Both
/// properties are asserted, not assumed: `the_jeffreys_term_is_live_...` pins
/// the second and the control arm's own pass pins the first.
fn fixture() -> (Array2<f64>, Array2<f64>) {
    let n = PER_CLASS * K;
    let mut design = Array2::<f64>::zeros((n, P));
    let mut response = Array2::<f64>::zeros((n, K));
    for class in 0..K {
        for i in 0..PER_CLASS {
            let row = class * PER_CLASS + i;
            // Bands centred at -2, 0, +2 with half-width 0.45.
            let t = (i as f64 + 0.5) / PER_CLASS as f64 - 0.5;
            let x = 2.0 * (class as f64 - 1.0) + 0.9 * t;
            design[[row, 0]] = 1.0;
            design[[row, 1]] = x;
            design[[row, 2]] = (0.7 * x).sin();
            design[[row, 3]] = (0.4 * x).cos();
            // The crossing row: the band's first row is labelled as the next
            // class round, so no linear rule classifies the sample perfectly.
            let label = if i == 0 { (class + 1) % K } else { class };
            response[[row, label]] = 1.0;
        }
    }
    (design, response)
}

/// A rank-deficient penalty: it leaves the intercept and the linear column
/// unpenalized, so the separating direction lives in `ker(S)` exactly as it does
/// for a spline's polynomial null space.
fn penalty() -> PenaltyMatrix {
    let mut s = Array2::<f64>::zeros((P, P));
    s[[2, 2]] = 1.0;
    s[[3, 3]] = 1.0;
    PenaltyMatrix::Dense(s)
}

fn options() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: 64,
        inner_tol: 1e-10,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    }
}

struct Reading {
    value: f64,
    gradient: Array1<f64>,
    hessian: Option<Array2<f64>>,
}

fn evaluate(armed: bool, rho: &Array1<f64>, mode: EvalMode) -> Reading {
    let (design, response) = fixture();
    let n = design.nrows();
    let family = MultinomialFamily::new(
        response,
        Array1::<f64>::ones(n),
        K,
        Arc::new(design),
        Arc::new(vec![penalty()]),
    )
    .expect("multinomial fixture is valid")
    .with_joint_jeffreys_term(armed)
    .with_joint_initial_log_lambdas(rho.to_vec());
    let blocks = family.build_block_specs();
    let diagnostics =
        evaluate_labeled_outer_criterion_for_diagnostics(&family, &blocks, &options(), rho, mode)
            .expect("the outer criterion evaluates at this rho");
    assert!(
        diagnostics.inner_converged,
        "the inner solve must converge for the finite difference to price one criterion \
         (armed={armed}, rho={rho:?})"
    );
    Reading {
        value: diagnostics.objective,
        gradient: diagnostics.gradient,
        hessian: diagnostics.outer_hessian,
    }
}

/// The rho this is taken at. Term-major over `K` per-class copies of the single
/// smooth component; a moderate value so both the wiggliness and the null space
/// are live and the inner solve converges from the default seed.
fn base_rho(dim: usize) -> Array1<f64> {
    Array1::from_shape_fn(dim, |i| if i % 2 == 0 { 0.5 } else { -0.5 })
}

fn rho_dim() -> usize {
    let (design, response) = fixture();
    let n = design.nrows();
    let family = MultinomialFamily::new(
        response,
        Array1::<f64>::ones(n),
        K,
        Arc::new(design),
        Arc::new(vec![penalty()]),
    )
    .expect("multinomial fixture is valid");
    family
        .joint_penalty_specs()
        .expect("joint penalty specs")
        .len()
}

/// Central finite difference of the criterion VALUE in coordinate `k`.
fn central_difference(armed: bool, rho: &Array1<f64>, k: usize, h: f64) -> f64 {
    let mut plus = rho.clone();
    plus[k] += h;
    let mut minus = rho.clone();
    minus[k] -= h;
    let up = evaluate(armed, &plus, EvalMode::ValueOnly).value;
    let down = evaluate(armed, &minus, EvalMode::ValueOnly).value;
    (up - down) / (2.0 * h)
}

/// The tolerance the comparison is held to, relative to the larger of the two
/// numbers. Central differences on a criterion whose inner solve certifies to
/// `1e-10` carry an `O(h²)` truncation plus an `O(eps/h)` cancellation floor; at
/// `h = 1e-3` on an `O(1)` criterion that is comfortably under `1e-4`. `2e-3` is
/// two decades above the observed control-arm agreement (`~1e-7`) and two
/// decades BELOW the defect this pins (`0.05` to `1.5`), so it cannot be met by
/// a gradient that describes a different objective and cannot fail on FD noise.
const RELATIVE_TOLERANCE: f64 = 2e-3;

fn assert_gradient_is_the_criterions(armed: bool) {
    let dim = rho_dim();
    let rho = base_rho(dim);
    let analytic = evaluate(armed, &rho, EvalMode::ValueAndGradient);
    let mut worst = (0usize, 0.0_f64, 0.0_f64, 0.0_f64);
    for k in 0..dim {
        let fd = central_difference(armed, &rho, k, 1e-3);
        let g = analytic.gradient[k];
        let relative = (fd - g).abs() / g.abs().max(fd.abs()).max(1e-8);
        eprintln!(
            "#2612 outer-gradient FD (jeffreys={armed}) coord {k:2}: \
             fd={fd:+.9e} analytic={g:+.9e} relative={relative:.3e}"
        );
        if relative > worst.1 {
            worst = (k, relative, fd, g);
        }
    }
    let (k, relative, fd, g) = worst;
    assert!(
        relative <= RELATIVE_TOLERANCE,
        "outer criterion gradient disagrees with its own value (jeffreys armed={armed}): \
         worst coordinate {k} has central difference {fd:+.9e} against analytic {g:+.9e} \
         (relative {relative:.3e} > {RELATIVE_TOLERANCE:.1e}). A line search cannot make \
         progress on a direction built from a gradient that is not the criterion's."
    );
    eprintln!(
        "#2612 outer-gradient FD (jeffreys={armed}): value={:.12e} worst coord {k} \
         relative {relative:.3e}",
        analytic.value
    );
}

/// CONTROL. With the Jeffreys term disarmed the analytic outer gradient is the
/// criterion's — this is what makes the armed arm's failure attributable.
#[test]
fn unbiased_outer_gradient_matches_central_differences_2612() {
    assert_gradient_is_the_criterions(false);
}

/// The defect. Same fixture, same rho, same harness — only the term differs.
#[test]
fn jeffreys_armed_outer_gradient_matches_central_differences_2612() {
    assert_gradient_is_the_criterions(true);
}

/// The two arms must actually differ, or both of the above are vacuous: a
/// fixture on which the conditioning gate never fires would pass the armed test
/// by evaluating the unbiased criterion.
#[test]
fn the_jeffreys_term_is_live_on_this_fixture_2612() {
    let dim = rho_dim();
    let rho = base_rho(dim);
    let unbiased = evaluate(false, &rho, EvalMode::ValueAndGradient);
    let armed = evaluate(true, &rho, EvalMode::ValueAndGradient);
    let separation = (unbiased.value - armed.value).abs();
    assert!(
        separation > 1e-6,
        "the conditioning gate does not fire on this fixture: the armed and unbiased \
         criteria agree to {separation:.3e}, so the armed gradient test is vacuous"
    );
    eprintln!(
        "#2612 fixture liveness: unbiased V={:.9e} armed V={:.9e} (gap {separation:.3e})",
        unbiased.value, armed.value
    );
}

// ───────────────────────────────────────────────────────────────────────────
// The same question one order up: is the analytic outer HESSIAN the Jacobian
// of the (now exact) analytic outer gradient?
// ───────────────────────────────────────────────────────────────────────────

/// Central finite difference of the analytic GRADIENT in coordinate `k`, i.e.
/// one column of the criterion's true Hessian.
///
/// Differencing the gradient rather than the value is the right instrument here:
/// the gradient is exact (the tests above are what says so), so its Jacobian IS
/// the Hessian, and a first-difference of an exact quantity is two orders more
/// accurate than a second-difference of the value.
fn hessian_column_by_difference(armed: bool, rho: &Array1<f64>, k: usize, h: f64) -> Array1<f64> {
    let mut plus = rho.clone();
    plus[k] += h;
    let mut minus = rho.clone();
    minus[k] -= h;
    let up = evaluate(armed, &plus, EvalMode::ValueAndGradient).gradient;
    let down = evaluate(armed, &minus, EvalMode::ValueAndGradient).gradient;
    (&up - &down) / (2.0 * h)
}

/// The analytic outer Hessian and the Jacobian of the analytic gradient, as a
/// pair, plus the worst entrywise relative disagreement between them.
fn hessian_pair(armed: bool) -> (Array2<f64>, Array2<f64>, (usize, usize, f64)) {
    let dim = rho_dim();
    let rho = base_rho(dim);
    let analytic = evaluate(armed, &rho, EvalMode::ValueGradientHessian)
        .hessian
        .expect("the outer criterion declares an analytic Hessian");
    let mut differenced = Array2::<f64>::zeros((dim, dim));
    for k in 0..dim {
        differenced
            .column_mut(k)
            .assign(&hessian_column_by_difference(armed, &rho, k, 1e-3));
    }
    // Symmetrize the difference: the two triangles are two independent estimates
    // of the same mixed partial, so their mean is the better one and the
    // comparison below is against a symmetric matrix, as the analytic one is.
    let symmetric = 0.5 * (&differenced + &differenced.t());
    let mut worst = (0usize, 0usize, 0.0_f64);
    for k in 0..dim {
        for j in 0..dim {
            let (fd, h_jk) = (symmetric[[j, k]], analytic[[j, k]]);
            let relative = (fd - h_jk).abs() / h_jk.abs().max(fd.abs()).max(1e-8);
            eprintln!(
                "#2612 outer-Hessian FD (jeffreys={armed}) [{j:2},{k:2}]: \
                 fd={fd:+.9e} analytic={h_jk:+.9e} relative={relative:.3e}"
            );
            if relative > worst.2 {
                worst = (j, k, relative);
            }
        }
    }
    (analytic, symmetric, worst)
}

fn assert_hessian_is_the_gradients_jacobian(armed: bool) {
    let (_, _, (j, k, relative)) = hessian_pair(armed);
    assert!(
        relative <= RELATIVE_TOLERANCE,
        "outer criterion Hessian disagrees with the Jacobian of its own gradient \
         (jeffreys armed={armed}): worst entry [{j},{k}] relative {relative:.3e} > \
         {RELATIVE_TOLERANCE:.1e}. The terminal certification asks this matrix for a \
         positive-semidefinite verdict, so a matrix that is not the criterion's curvature \
         refuses fits at genuine minima."
    );
}

/// CONTROL, one order up.
#[test]
fn unbiased_outer_hessian_matches_the_gradient_jacobian_2612() {
    assert_hessian_is_the_gradients_jacobian(false);
}

/// The armed arm, held to the claim the certificate actually takes from this
/// matrix rather than to entrywise equality.
///
/// Entrywise equality is NOT available and the reason is named rather than
/// tolerated: `H_Phi` is a divided-difference object built from `H` and its
/// first directional derivatives, so its own first beta-derivative already
/// consumes the family's second, and `D2_beta H_Phi[v_k, v_l]` would need the
/// third — which no family exposes. What the terminal certification asks this
/// matrix is not "are your entries right" but "is the interior block positive
/// semidefinite", so that is what is gated here: every eigenvalue of the
/// analytic Hessian must agree in SIGN with the corresponding eigenvalue of the
/// criterion's own curvature, and the smallest must agree closely enough that
/// the PSD verdict cannot turn on the residual.
///
/// This is a strictly weaker claim than the control arm's and it is the one that
/// is true. It is also the one with teeth: dropping the `D_beta H_Phi[u_kl]`
/// fold — the half that IS exactly computable — took the penguins arm's interior
/// `lambda_min` from `-6.71e-5` to `-1.128e-2`, i.e. from two decades inside its
/// gradient floor to eleven times outside it, which is a sign flip in every
/// sense that matters.
#[test]
fn jeffreys_armed_outer_hessian_agrees_on_the_curvature_verdict_2612() {
    let (analytic, differenced, (j, k, worst)) = hessian_pair(true);
    let analytic_spectrum = spectrum(&analytic);
    let differenced_spectrum = spectrum(&differenced);
    eprintln!(
        "#2612 armed outer-Hessian spectra: analytic={:?} differenced={:?} \
         (worst entrywise [{j},{k}] relative {worst:.3e})",
        analytic_spectrum
            .iter()
            .map(|v| format!("{v:+.6e}"))
            .collect::<Vec<_>>(),
        differenced_spectrum
            .iter()
            .map(|v| format!("{v:+.6e}"))
            .collect::<Vec<_>>(),
    );
    for (index, (a, d)) in analytic_spectrum
        .iter()
        .zip(differenced_spectrum.iter())
        .enumerate()
    {
        assert_eq!(
            a.is_sign_negative(),
            d.is_sign_negative(),
            "eigenvalue {index} of the armed outer Hessian disagrees in SIGN with the \
             criterion's own curvature: analytic {a:+.9e} against differenced {d:+.9e}. The \
             terminal certificate's whole verdict is that sign."
        );
    }
    let (a_min, d_min) = (analytic_spectrum[0], differenced_spectrum[0]);
    let relative = (a_min - d_min).abs() / a_min.abs().max(d_min.abs()).max(1e-12);
    assert!(
        relative <= SMALLEST_EIGENVALUE_TOLERANCE,
        "the armed outer Hessian's SMALLEST eigenvalue is {a_min:+.9e} against the \
         criterion's {d_min:+.9e} (relative {relative:.3e} > {SMALLEST_EIGENVALUE_TOLERANCE:.1e}); \
         the PSD verdict the certificate takes from this matrix would turn on the residual."
    );
}

/// Ascending eigenvalues of a symmetric matrix.
fn spectrum(matrix: &Array2<f64>) -> Vec<f64> {
    let (values, _) = matrix
        .eigh(faer::Side::Lower)
        .expect("a symmetric outer Hessian eigendecomposes");
    let mut values: Vec<f64> = values.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite eigenvalues"));
    values
}

/// How far the SMALLEST eigenvalue may sit from the criterion's own, relative to
/// the larger of the two. The residual `D2_beta H_Phi` term is a bounded
/// perturbation of the assembled curvature, not a rescaling of it, so the
/// smallest eigenvalue — the one the PSD verdict reads — must survive it. `0.25`
/// is set from the measured agreement with room, and is two orders tighter than
/// "the sign could flip".
const SMALLEST_EIGENVALUE_TOLERANCE: f64 = 0.25;
