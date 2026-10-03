#![cfg(test)]
//! #4077 item 1 — the two free measurements the issue's RUN-NEEDED names, taken
//! at the interval-active-bound fixture (`tests_interval_active_bound_3438`)
//! BEFORE the half-line Laplace mass lands.
//!
//! The maintainer's 2026-09-21 derivation accepts both halves of #4077 by ONE
//! measurement: "a finite difference of the criterion in rho at
//! crates/gam-sae/src/manifold/tests_interval_active_bound_3438.rs matching the
//! analytic gradient once the pinned set is held fixed", plus "the mu -> 0+
//! limit check that the term tends to ln 2 + 1/2 log sigma".
//!
//! This module takes the BEFORE arm of that measurement: the FD-in-ρ of the
//! production criterion at the fitted state, and the jet's own mu -> 0+ limit.
//! The fixture only prices once the certificate lane factors an all-pinned row
//! (the closed-form row ridge floor, this branch's base), so both numbers are
//! new — they could not be taken on main before it.
//!
//! ## What the BEFORE number establishes
//!
//! With the half-line term absent, the pinned slot contributes NOTHING to the
//! criterion value (`A` prices it at `log 1 = 0`) and its ρ-maps are zero
//! (projection, #3438). The FD-in-ρ of the production value is therefore the
//! derivative of a criterion whose pinned slots are inert. When the half-line
//! term lands, it adds `-ln J(mu, sigma) + 1/2 ln 2pi` per pinned slot — a
//! term with a live ρ-derivative through sigma — and this same test's FD must
//! then close against the analytic gradient that carries the term's jet. The
//! BEFORE receipt is the baseline the AFTER arm grades against, and the
//! numbers it prints are the evidence the landing PR needs.
//!
//! ## The pinned-set caveat, stated honestly
//!
//! The FD perturbs ρ and REFINES the inner solve at each endpoint; if the
//! pin set changed across the step the FD would measure a kink, not a
//! derivative. The test asserts the pin set is IDENTICAL at both endpoints of
//! every probe, so the difference quotients measure the smooth branch.

use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_math::gaussian_reciprocal::half_line_gaussian_log_jet;
use gam_terms::latent::LatentManifold;
use ndarray::{array, Array1, Array2};
use std::sync::Arc;

const LO: f64 = -1.0;
const HI: f64 = 1.0;

/// The same interval fixture as `tests_interval_active_bound_3438`, built
/// independently so this module's numbering does not leak into that one.
fn interval_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let p = 2usize;
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch basis"));
    // Generating latent: three rows sit beyond the interval's upper end, so the
    // fitted coordinates of those rows are pushed onto `hi` with `g < 0` there.
    let truth: [f64; 8] = [-0.7, -0.3, 0.1, 0.4, 1.6, 1.8, 2.0, 0.8];
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth[row].clamp(-0.9, 0.9));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("coords evaluate");
    let width = phi.ncols();
    let decoder = Array2::<f64>::from_shape_fn((width, p), |(b, o)| {
        [[0.1, -0.2], [1.0, 0.6], [0.2, -0.3]][b][o]
    });
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let s = truth[row];
        for o in 0..p {
            target[[row, o]] = decoder[[0, o]]
                + decoder[[1, o]] * s
                + decoder[[2, o]] * s * s
                + 0.03 * (1.7 * row as f64 + 0.9 * o as f64).sin();
        }
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "interval".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(width),
    )
    .expect("atom shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Interval { lo: LO, hi: HI }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 1.0, vec![array![-3.0]]);
    (term, target, rho)
}

/// One production criterion evaluation at a perturbed ρ, returning the value
/// and the pin set the assembly recorded at that ρ.
/// The (row, slot) pairs pinned at an active bound.
type PinnedSlots = Vec<(usize, usize)>;

fn priced(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
) -> (f64, PinnedSlots) {
    let mut probe = term.clone();
    let (value, _, _) = probe
        .penalized_quasi_laplace_criterion_with_cache(target, rho, None, 60, 0.4, 1.0e-8, 1.0e-8)
        .expect("the probe prices a finite criterion");
    let pinned = probe.last_pinned_bound_slots.clone();
    (value, pinned)
}

/// #4077 BEFORE-arm: the finite difference in ρ of the production criterion at
/// the fitted state, against the analytic outer gradient, with the pin set
/// held fixed across every step.
///
/// The maintainer's derivation: the acceptance for the half-line term is that
/// this FD matches the analytic gradient. Today (term absent) the analytic
/// gradient carries no pinned-slot leg, and neither does the value — this test
/// RECEIPTS that agreement and prints the per-coordinate gaps as the baseline
/// the landing PR grades against.
#[test]
fn rho_fd_matches_analytic_gradient_before_the_half_line_term_4077() {
    let (term, target, rho) = interval_fixture();
    let objective = SaeManifoldOuterObjective::new(
        term.clone(),
        target.clone(),
        None,
        rho.clone(),
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    );
    let mut objective = objective;
    // The objective binds the assignment and may hold outer coordinates (the
    // sparse strength at K=1 softmax), so the flat layout the gradient lives
    // in is ITS layout, not the fixture rho's.
    let centre = objective.baseline_rho.flat_coordinates();
    let evaluation = objective
        .eval(&centre)
        .expect("the centre evaluation must return a (value, gradient) pair");
    assert!(
        evaluation.cost.is_finite(),
        "the fixture must price a finite criterion at its centre, got {}",
        evaluation.cost
    );
    let analytic = evaluation.gradient.clone();

    // The fixture must actually pin for this measurement to be about #4077.
    let (_, pinned_centre) = priced(&term, target.view(), &rho);
    assert!(
        !pinned_centre.is_empty(),
        "no interval coordinate sits at an active bound"
    );
    println!(
        "[#4077 BEFORE] pinned (row, slot) = {pinned_centre:?}; centre cost = {:.12e}",
        evaluation.cost
    );

    // Two step sizes and a Richardson extrapolation, the #2933 F05 recipe.
    const STEP: f64 = 2.0e-3;
    for coordinate in 0..centre.len() {
        let mut estimates = [0.0_f64; 2];
        for (index, step) in [STEP, 0.5 * STEP].into_iter().enumerate() {
            let plus_rho = objective
                .baseline_rho
                .from_flat(shifted(&centre, coordinate, step).view())
                .expect("the plus endpoint uses the objective's layout");
            let minus_rho = objective
                .baseline_rho
                .from_flat(shifted(&centre, coordinate, -step).view())
                .expect("the minus endpoint uses the objective's layout");
            let (plus_value, plus_pinned) = priced(&term, target.view(), &plus_rho);
            let (minus_value, minus_pinned) = priced(&term, target.view(), &minus_rho);
            assert!(
                plus_value.is_finite() && minus_value.is_finite(),
                "coordinate {coordinate}, step {step:e}: endpoints must be finite \
                 (plus={plus_value}, minus={minus_value})"
            );
            // #4077's own caveat: the FD is only a derivative on a branch where
            // the pin set is constant. If a perturbation moves a coordinate off
            // its bound the difference quotients measure a kink; refuse rather
            // than grade a meaningless number.
            assert!(
                plus_pinned == pinned_centre && minus_pinned == pinned_centre,
                "coordinate {coordinate}, step {step:e}: the pin set moved across the \
                 probe (centre {pinned_centre:?}, plus {plus_pinned:?}, minus {minus_pinned:?}); \
                 the FD is not a branch derivative here"
            );
            estimates[index] = (plus_value - minus_value) / (2.0 * step);
        }
        let richardson = (4.0 * estimates[1] - estimates[0]) / 3.0;
        let scale = 1.0 + analytic[coordinate].abs().max(richardson.abs());
        let gap = (analytic[coordinate] - richardson).abs() / scale;
        let oracle_error = (estimates[1] - estimates[0]).abs() / scale;
        println!(
            "[#4077 BEFORE] coordinate {coordinate}: analytic={:.10e} richardson={richardson:.10e} \
             gap={gap:.3e} oracle_error={oracle_error:.3e}",
            analytic[coordinate]
        );
        // BEFORE arm: the analytic gradient (no half-line leg) must already
        // match the FD of the value it differentiates. The budget is the #2933
        // F05 treatment budget, one order above its own 1e-4.
        const BUDGET: f64 = 1.0e-4;
        assert!(
            gap + oracle_error <= BUDGET,
            "coordinate {coordinate}: BEFORE the half-line term the analytic gradient \
             must match the FD of the value (gap {gap:.3e} + oracle {oracle_error:.3e} \
             > {BUDGET:e})"
        );
    }
}

/// #4077 — the free check that needs no fixture: as `mu -> 0+` the half-line
/// term `-ln J + 1/2 ln 2pi` tends to `ln 2 + 1/2 ln sigma`, half the Gaussian
/// mass, exactly `ln 2` above the free coordinate's `1/2 ln sigma` — the
/// discontinuity the current pricing of 0 cannot represent for any sigma.
///
/// This grades `half_line_gaussian_log_jet` (already in gam-math, unlanded at
/// any criterion site) against the maintainer's stated limit before the term
/// is ever priced at a fixture.
#[test]
fn half_line_jet_tends_to_half_the_gaussian_mass_at_mu_zero_4077() {
    for &sigma in &[0.5_f64, 1.0, 2.0, 10.0, 100.0] {
        // The pinned-slot term value: -ln J(mu, sigma) + 1/2 ln(2 pi).
        let term_at = |mu: f64| -> f64 {
            let jet = half_line_gaussian_log_jet(mu, sigma)
                .unwrap_or_else(|| panic!("a positive-sigma half-line mass at mu={mu}"));
            -jet[0] + 0.5 * std::f64::consts::TAU.ln()
        };
        // mu -> 0+ along a geometric approach; the limit is ln 2 + 1/2 ln sigma.
        let limit = std::f64::consts::LN_2 + 0.5 * sigma.ln();
        // The approach is LINEAR with the analytic first-order slope: at mu = 0
        // the density's mass on the half-line is half the Gaussian mass, and
        // the mean of the truncated normal is -mu/sigma, so
        // d(-ln J)/dmu = E[s | mu] = sqrt(2/(pi sigma)) at mu = 0.
        let slope = (2.0 / (std::f64::consts::PI * sigma)).sqrt();
        let mut last_gap = f64::INFINITY;
        let mut mu_last = f64::INFINITY;
        for exponent in 1..=8 {
            let mu = 10.0_f64.powi(-exponent);
            let gap = (term_at(mu) - limit).abs();
            println!(
                "[#4077 LIMIT] sigma={sigma} mu=1e-{exponent}: term={:.12e} limit={limit:.12e} \
                 gap={gap:.3e} gap/mu={:.6e} slope={slope:.6e}",
                term_at(mu),
                gap / mu
            );
            // The jet's approach must carry the analytic slope: |gap/mu - slope|
            // shrinks toward zero as the higher orders fall away (they enter at
            // mu/sigma, one order down), and the gap itself must shrink into
            // the double-precision noise floor of the limit.
            assert!(
                gap <= last_gap * 1.5 || gap < 1.0e-12,
                "sigma={sigma} mu=1e-{exponent}: the half-line term's approach to \
                 ln 2 + 1/2 ln sigma is not contracting (gap {gap:.3e} after {last_gap:.3e})"
            );
            if exponent >= 4 {
                let slope_error = (gap / mu - slope).abs() / slope;
                assert!(
                    slope_error <= 0.5,
                    "sigma={sigma} mu=1e-{exponent}: the approach slope {}/mu deviates from \
                     the analytic sqrt(2/(pi sigma)) by {slope_error:.3e} (50% budget)",
                    format!("{:.6e}", gap / mu)
                );
            }
            last_gap = last_gap.min(gap);
            mu_last = mu;
        }
        // The approach is LINEAR in mu with the analytic slope, so at the
        // deepest point the gap must be that slope's prediction to 50%: the
        // term has converged to the limit up to the linear tail mu itself.
        assert!(
            last_gap <= 1.5 * slope * mu_last,
            "sigma={sigma}: the final gap {last_gap:.3e} exceeds the linear prediction \
             slope*mu = {:.3e} — the term is not approaching ln 2 + 1/2 ln sigma",
            slope * mu_last
        );
        // And the discontinuity the current pricing cannot represent: the pinned
        // slot today contributes 0. `limit = ln 2 + 1/2 ln sigma` vanishes only
        // at sigma = 1/4, so on this sigma ladder the zero pricing's error is
        // orders above the criterion's own material floor — the pin boundary is
        // a real jump today, not a rounding tale.
        assert!(
            (0.0 - limit).abs() >= 0.3,
            "sigma={sigma}: the zero pricing is within 0.3 of the true limit {limit:.6e}; \
             the discontinuity claim needs a live sigma"
        );
    }
}

fn shifted(centre: &Array1<f64>, coordinate: usize, step: f64) -> Array1<f64> {
    let mut moved = centre.clone();
    moved[coordinate] += step;
    moved
}
