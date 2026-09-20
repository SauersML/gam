//! #2080 — the gate prior's logit Jacobian `−ln[z(1 − z)/τ]`.
//!
//! The ordered Beta--Bernoulli and ThresholdGate priors are densities on the gate
//! probability, while the inner solve and the Laplace evidence integrate over its logit, so
//! the penalized objective carries the change of variables. These pins check every
//! derivative order against a central difference of the order below it, the loss value
//! against the gradient and curvature the assembly installs, and the finite mode the term
//! gives a gate whose data want it saturated.

use super::tests_outer_quasi_laplace_probe_budget_2080::{
    one_circle_wide_target, two_circle_periodic_term,
};
use super::*;
use crate::assignment::{
    GateLogitJacobian, assignment_prior_value_weighted, gate_logit_jacobian_grad_hdiag_weighted,
    gate_logit_jacobian_third_weighted, gate_logit_jacobian_value_weighted, sigmoid_gate_frame,
};
use ndarray::{Array2, array};

/// `(f(x + h) − f(x − h))/(2h)` and the bar it must meet: twice the mean-value truncation
/// `h²/6·sup_third`, where `sup_third` bounds `|f‴|` on the stencil (the slack keeps a
/// difference taken at `|f‴|`'s own extremum clear of rounding in the stencil), plus
/// round-off from the two values and from rounding the argument `x` (and the gate threshold
/// inside `f`), whose effect on `f` `sup_slope` bounds.
fn central_difference(
    f: impl Fn(f64) -> f64,
    x: f64,
    h: f64,
    argument_scale: f64,
    sup_slope: f64,
    sup_third: f64,
) -> (f64, f64) {
    let up = f(x + h);
    let down = f(x - h);
    let truncation = 2.0 * h * h / 6.0 * sup_third;
    let round_off =
        8.0 * f64::EPSILON * (up.abs() + down.abs() + sup_slope * (argument_scale + h)) / h;
    ((up - down) / (2.0 * h), truncation + round_off)
}

/// Suprema over every logit of `|J′|`, `|J″|`, `|J‴|`, `|J⁗|` and `|J⁽⁵⁾|` for a gate of
/// weight `w` and temperature `τ`: `w/τ`, `w/(2τ²)`, `w/(3√3·τ³)`, `w/(4τ⁴)` and at most `w/τ⁵`
/// (from `|2z − 1| ≤ 1`, `z(1 − z) ≤ ¼` and their products).
fn jacobian_derivative_suprema(weight: f64, temperature: f64) -> [f64; 5] {
    let inv_tau = temperature.recip();
    [
        weight * inv_tau,
        weight * inv_tau.powi(2) / 2.0,
        weight * inv_tau.powi(3) / (3.0 * 3.0_f64.sqrt()),
        weight * inv_tau.powi(4) / 4.0,
        weight * inv_tau.powi(5),
    ]
}

/// Each of `J′`, `J″`, `J‴` is the central difference of the order below it, from deep
/// saturation on either side through the centre, in the ordered Beta--Bernoulli frame and a
/// design-weighted ThresholdGate frame.
#[test]
fn gate_logit_jacobian_orders_match_central_differences_2080() {
    for (weight, threshold, temperature) in [(1.0_f64, 0.0_f64, 1.0_f64), (2.5, 0.7, 0.35)] {
        let h = temperature * f64::EPSILON.cbrt();
        let sup = jacobian_derivative_suprema(weight, temperature);
        let at = |logit: f64| GateLogitJacobian::eval(weight, logit, threshold, temperature);
        for offset in [-40.0_f64, -18.0, -6.0, -1.3, 0.0, 0.4, 2.2, 9.0, 25.0] {
            let logit = threshold + temperature * offset;
            let scale = logit.abs() + threshold.abs();
            let here = at(logit);
            let orders = [
                (
                    "gradient",
                    central_difference(|l| at(l).value(), logit, h, scale, sup[0], sup[2]),
                    here.gradient(),
                ),
                (
                    "curvature",
                    central_difference(|l| at(l).gradient(), logit, h, scale, sup[1], sup[3]),
                    here.curvature(),
                ),
                (
                    "third",
                    central_difference(|l| at(l).curvature(), logit, h, scale, sup[2], sup[4]),
                    here.third(),
                ),
            ];
            for (order, (difference, bar), analytic) in orders {
                assert!(
                    (difference - analytic).abs() <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                    "gate logit Jacobian {order} at w={weight}, θ={threshold}, τ={temperature}, \
                     x={offset}: analytic {analytic:.12e}, central difference {difference:.12e}, \
                     bar {bar:.3e}"
                );
            }
        }
    }
}

/// The aggregate producers the loss, the assembly and the θ-adjoints read are one derivation:
/// under row weights, in both sigmoid modes, each order is the central difference of the
/// order below it along every logit.
#[test]
fn gate_logit_jacobian_producers_are_one_derivation_2080() {
    let logits = array![[6.0, -2.5], [-18.0, 0.3], [1.1, 25.0], [-0.4, -7.0]];
    let (n, k) = logits.dim();
    let row_weights = [0.5, 1.0, 2.0, 1.25];
    for mode in [
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        AssignmentMode::threshold_gate(0.35, 0.7),
    ] {
        let mut coords = Vec::with_capacity(k);
        let mut manifolds = Vec::with_capacity(k);
        for atom in 0..k {
            coords.push(Array2::from_shape_fn((n, 1), |(row, axis)| {
                0.05 * (row + axis + atom) as f64
            }));
            manifolds.push(LatentManifold::Circle { period: 1.0 });
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords,
            manifolds,
            mode,
        )
        .expect("one logit column, coordinate block and manifold per atom");
        let (threshold, temperature) = sigmoid_gate_frame(&assignment.mode)
            .expect("ordered Beta--Bernoulli and ThresholdGate gates are per-logit sigmoids");
        let h = temperature * f64::EPSILON.cbrt();
        let (gradient, curvature) =
            gate_logit_jacobian_grad_hdiag_weighted(&assignment, Some(&row_weights));
        for row in 0..n {
            let sup = jacobian_derivative_suprema(row_weights[row], temperature);
            for atom in 0..k {
                let index = row * k + atom;
                let logit = assignment.logits[[row, atom]];
                let scale = logit.abs() + threshold.abs();
                let moved = |value: f64| {
                    let mut shifted = assignment.clone();
                    shifted.logits[[row, atom]] = value;
                    shifted
                };
                let orders = [
                    (
                        "gradient",
                        central_difference(
                            |l| gate_logit_jacobian_value_weighted(&moved(l), Some(&row_weights)),
                            logit,
                            h,
                            scale,
                            sup[0],
                            sup[2],
                        ),
                        gradient[index],
                    ),
                    (
                        "curvature",
                        central_difference(
                            |l| {
                                gate_logit_jacobian_grad_hdiag_weighted(&moved(l), Some(&row_weights))
                                    .0[index]
                            },
                            logit,
                            h,
                            scale,
                            sup[1],
                            sup[3],
                        ),
                        curvature[index],
                    ),
                    (
                        "third",
                        central_difference(
                            |l| {
                                gate_logit_jacobian_grad_hdiag_weighted(&moved(l), Some(&row_weights))
                                    .1[index]
                            },
                            logit,
                            h,
                            scale,
                            sup[2],
                            sup[4],
                        ),
                        gate_logit_jacobian_third_weighted(&assignment, Some(&row_weights), row, atom),
                    ),
                ];
                for (order, (difference, bar), analytic) in orders {
                    assert!(
                        (difference - analytic).abs() <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                        "{} gate Jacobian {order} at row {row}, atom {atom}, logit {logit}: \
                         analytic {analytic:.12e}, central difference {difference:.12e}, \
                         bar {bar:.3e}",
                        assignment.mode.family_label()
                    );
                }
            }
        }
    }
}

/// On a real term whose gates are seeded saturated, the loss field carries the Jacobian and
/// nothing else beyond the sparsity prior, its logit gradient is the producer's, and the
/// assembled `H_tt` logit diagonal holds at least the Jacobian's curvature (the data
/// Gauss--Newton term and the majorized prior term it sits beside are non-negative).
#[test]
fn loss_carries_the_jacobian_the_assembly_installs_2080() {
    let z = one_circle_wide_target(24, 8, 0.05);
    let term = two_circle_periodic_term(z.view(), 2, 1).0;
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 2])
        .for_assignment(&term.assignment);
    let (threshold, temperature) = sigmoid_gate_frame(&term.assignment.mode)
        .expect("the fixture's ordered Beta--Bernoulli gates are per-logit sigmoids");
    let h = temperature * f64::EPSILON.cbrt();
    let (n, k) = term.assignment.logits.dim();
    let weights = term.row_loss_weights.clone();
    let (gradient, curvature) =
        gate_logit_jacobian_grad_hdiag_weighted(&term.assignment, weights.as_deref());
    let jacobian_in_loss = |shifted: &SaeManifoldTerm| -> f64 {
        let loss = shifted.loss(z.view(), &rho).expect("loss at the shifted state");
        let prior =
            assignment_prior_value_weighted(&shifted.assignment, &rho, weights.as_deref())
                .expect("assignment prior value at the shifted state");
        loss.assignment_sparsity - prior
    };
    assert!(
        (jacobian_in_loss(&term)
            - gate_logit_jacobian_value_weighted(&term.assignment, weights.as_deref()))
        .abs()
            <= 64.0 * f64::EPSILON * (1.0 + jacobian_in_loss(&term).abs()),
        "the loss field must carry exactly the gate Jacobian beyond the sparsity prior"
    );
    for row in 0..n {
        let weight = weights.as_ref().map_or(1.0, |w| w[row]);
        let sup = jacobian_derivative_suprema(weight, temperature);
        for atom in 0..k {
            let index = row * k + atom;
            let logit = term.assignment.logits[[row, atom]];
            let moved = |value: f64| {
                let mut shifted = term.clone();
                shifted.assignment.logits[[row, atom]] = value;
                jacobian_in_loss(&shifted)
            };
            let (difference, bar) =
                central_difference(moved, logit, h, logit.abs() + threshold.abs(), sup[0], sup[2]);
            assert!(
                (difference - gradient[index]).abs() <= bar + 4.0 * f64::EPSILON * gradient[index].abs(),
                "loss-field Jacobian gradient at row {row}, atom {atom}, logit {logit}: producer \
                 {:.12e}, central difference {difference:.12e}, bar {bar:.3e}",
                gradient[index]
            );
        }
    }
    let mut assembled = term.clone();
    let system = assembled
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("assemble the fixture's arrow system");
    for (row, block) in system.rows.iter().enumerate() {
        let vars = assembled
            .row_vars_for_row_dim(row, block.htt.nrows())
            .expect("row layout of the assembled system");
        for (slot, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                let installed = block.htt[[slot, slot]];
                let jacobian = curvature[row * k + atom];
                assert!(
                    installed >= jacobian,
                    "row {row}, atom {atom}: installed logit curvature {installed:.12e} is below \
                     the gate Jacobian's own {jacobian:.12e}"
                );
            }
        }
    }
}

/// A gate whose remaining objective gains `μ` per unit of gate probability wants the gate
/// fully on: `g(z) = −μ·z`, so along its logit `f(ℓ) = −μ·z + J(ℓ)` with `z = σ(ℓ/τ)`.
///
/// Without `J`, `f′ = −μ·z(1 − z)/τ < 0` at every finite logit, so there is no mode. With it
/// `f′ = 0` at `μ·z(1 − z) = w·(2z − 1)`, i.e. `z* = [(μ − 2w) + √(μ² + 4w²)]/(2μ)`, and there
/// `f″ = w·(2z*² − 2z* + 1)/τ²`. On `z ≥ ½`, `f″ = z(1 − z)·(2w − μ·(1 − 2z))/τ² > 0`, so `f′`
/// increases from `−μ/(4τ)` at `ℓ = 0` to `w/τ` in the limit and the mode is its one root there.
#[test]
fn a_saturating_gate_has_a_finite_mode_at_the_derived_curvature_2080() {
    for (mu, weight, temperature) in [(100.0_f64, 1.0_f64, 1.0_f64), (4.0e3, 0.5, 0.25), (12.0, 3.0, 2.0)]
    {
        let inv_tau = temperature.recip();
        let gate = |logit: f64| gam_math::special::logistic(logit * inv_tau);
        let slope = |logit: f64| {
            let z = gate(logit);
            -mu * z * (1.0 - z) * inv_tau
                + GateLogitJacobian::eval(weight, logit, 0.0, temperature).gradient()
        };
        let curvature = |logit: f64| {
            let z = gate(logit);
            -mu * z * (1.0 - z) * (1.0 - 2.0 * z) * inv_tau * inv_tau
                + GateLogitJacobian::eval(weight, logit, 0.0, temperature).curvature()
        };
        // Bracket the root of the increasing `f′` on `ℓ ≥ 0`, then bisect to float resolution.
        let mut low = 0.0_f64;
        let mut high = temperature;
        while slope(high) <= 0.0 {
            low = high;
            high *= 2.0;
            assert!(high.is_finite(), "the gate slope never turned positive");
        }
        loop {
            let middle = 0.5 * (low + high);
            if middle <= low || middle >= high {
                break;
            }
            if slope(middle) <= 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
        let mode = 0.5 * (low + high);
        let z_mode = gate(mode);
        let z_derived = ((mu - 2.0 * weight) + (mu * mu + 4.0 * weight * weight).sqrt()) / (2.0 * mu);
        let x_mode = mode * inv_tau;
        assert!(
            mode.is_finite() && (z_mode - z_derived).abs() <= 16.0 * f64::EPSILON * (1.0 + x_mode.abs()),
            "μ={mu}, w={weight}, τ={temperature}: the mode's gate {z_mode:.15e} is not the derived \
             {z_derived:.15e}"
        );
        let derived_curvature =
            weight * (2.0 * z_derived * z_derived - 2.0 * z_derived + 1.0) * inv_tau * inv_tau;
        assert!(
            (curvature(mode) - derived_curvature).abs()
                <= 64.0 * f64::EPSILON * (mu + 4.0 * weight) * inv_tau * inv_tau * (1.0 + x_mode.abs()),
            "μ={mu}, w={weight}, τ={temperature}: curvature at the mode {:.15e} is not the derived \
             {derived_curvature:.15e}",
            curvature(mode)
        );
        // Negative control: without the Jacobian the same gate has no stationary logit past the
        // mode, where its slope stays strictly negative. The slope is taken as
        // `e^{−|x|}/(1 + e^{−|x|})²` rather than `z(1 − z)`, whose `1 − z` rounds to zero past
        // `x ≈ 37`.
        for step in 0..=6 {
            let logit = mode + 10.0 * step as f64 * temperature;
            let tail = (-(logit * inv_tau).abs()).exp();
            let bare_slope = -mu * tail / ((1.0 + tail) * (1.0 + tail)) * inv_tau;
            assert!(
                bare_slope < 0.0,
                "μ={mu}, τ={temperature}: without the Jacobian the slope at logit {logit} must be \
                 negative, got {bare_slope:.3e}"
            );
        }
    }
}

/// A softmax assignment over `logits` at `temperature`, with one circle coordinate per atom.
fn softmax_assignment(logits: Array2<f64>, temperature: f64) -> SaeAssignment {
    let (n, k) = logits.dim();
    let mut coords = Vec::with_capacity(k);
    let mut manifolds = Vec::with_capacity(k);
    for atom in 0..k {
        coords.push(Array2::from_shape_fn((n, 1), |(row, axis)| {
            0.05 * (row + axis + atom) as f64
        }));
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::softmax(temperature),
    )
    .expect("one logit column, coordinate block and manifold per atom")
}

/// A softmax row's Jacobian orders against central differences, along every free logit of two
/// K=4 rows with saturated minority atoms, under row weights. The value producer's slope must be
/// the gradient producer, the gradient's slope the row block, and the block's slope
/// `simplex_gate_logit_jacobian_third`. The bars use `|∂ⁿJ| ≤ m_n·c/τⁿ` with
/// `m = 1, 1, 3, 12, 60`, since each z-factor's logit derivative is bounded by the factor.
#[test]
fn simplex_gate_logit_jacobian_orders_match_central_differences_2080() {
    let temperature = 0.6_f64;
    let inv_tau = temperature.recip();
    let assignment = softmax_assignment(
        array![[6.0, -8.0, 0.5, 0.0], [-1.2, 2.4, -30.0, 0.0]],
        temperature,
    );
    let (n, k) = assignment.logits.dim();
    let row_weights = [1.5, 0.75];
    let count = crate::assignment::simplex_gate_free_count(&assignment)
        .expect("a K=4 softmax row has three free logits");
    let h = temperature * f64::EPSILON.cbrt();
    let gradient = gate_logit_jacobian_grad_hdiag_weighted(&assignment, Some(&row_weights)).0;
    for row in 0..n {
        let weight = row_weights[row];
        let sup = |order: i32, multiple: f64| weight * count * multiple * inv_tau.powi(order);
        let block = crate::assignment::simplex_gate_logit_jacobian_row_block(
            &assignment,
            Some(&row_weights),
            row,
        )
        .expect("a free softmax row");
        let z = crate::assignment::softmax_row(assignment.logits.row(row), temperature);
        let z_slice = z.as_slice().expect("contiguous softmax row");
        for i in 0..k - 1 {
            let logit = assignment.logits[[row, i]];
            let scale = logit.abs();
            let moved = |value: f64| {
                let mut shifted = assignment.clone();
                shifted.logits[[row, i]] = value;
                shifted
            };
            let (difference, bar) = central_difference(
                |l| gate_logit_jacobian_value_weighted(&moved(l), Some(&row_weights)),
                logit,
                h,
                scale,
                sup(1, 1.0),
                sup(3, 3.0),
            );
            let analytic = gradient[row * k + i];
            assert!(
                (difference - analytic).abs() <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                "softmax Jacobian gradient at row {row}, slot {i}: analytic {analytic:.12e}, \
                 central difference {difference:.12e}, bar {bar:.3e}"
            );
            for j in 0..k - 1 {
                let (difference, bar) = central_difference(
                    |l| {
                        gate_logit_jacobian_grad_hdiag_weighted(&moved(l), Some(&row_weights)).0
                            [row * k + j]
                    },
                    logit,
                    h,
                    scale,
                    sup(2, 1.0),
                    sup(4, 12.0),
                );
                let analytic = block[[j, i]];
                assert!(
                    (difference - analytic).abs() <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                    "softmax Jacobian curvature [{j}, {i}] at row {row}: analytic \
                     {analytic:.12e}, central difference {difference:.12e}, bar {bar:.3e}"
                );
                for w in 0..k - 1 {
                    let (difference, bar) = central_difference(
                        |l| {
                            crate::assignment::simplex_gate_logit_jacobian_row_block(
                                &moved(l),
                                Some(&row_weights),
                                row,
                            )
                            .expect("a free softmax row")[[j, w]]
                        },
                        logit,
                        h,
                        scale,
                        sup(3, 3.0),
                        sup(5, 60.0),
                    );
                    let analytic = weight
                        * crate::assignment::simplex_gate_logit_jacobian_third(
                            z_slice, j, w, i, count, inv_tau,
                        );
                    assert!(
                        (difference - analytic).abs()
                            <= bar + 4.0 * f64::EPSILON * analytic.abs(),
                        "softmax Jacobian third [{j}, {w}, {i}] at row {row}: analytic \
                         {analytic:.12e}, central difference {difference:.12e}, bar {bar:.3e}"
                    );
                }
            }
        }
    }
}

/// At K=2 the softmax chart's one free logit is the sigmoid gate `z₀ = σ(ℓ₀/τ)` against the
/// reference at zero, so the simplex Jacobian must equal [`GateLogitJacobian`] exactly: the
/// value, the gradient `(2z − 1)/τ` and the curvature `2z(1 − z)/τ²`.
#[test]
fn two_atom_softmax_jacobian_is_the_sigmoid_gate_jacobian_2080() {
    let temperature = 0.8_f64;
    let assignment = softmax_assignment(array![[3.5, 0.0], [-12.0, 0.0], [0.25, 0.0]], temperature);
    let (n, k) = assignment.logits.dim();
    let sigmoid = |row: usize| {
        GateLogitJacobian::eval(1.0, assignment.logits[[row, 0]], 0.0, temperature)
    };
    let softmax_value = gate_logit_jacobian_value_weighted(&assignment, None);
    let sigmoid_value: f64 = (0..n).map(|row| sigmoid(row).value()).sum();
    assert!(
        (softmax_value - sigmoid_value).abs() <= 64.0 * f64::EPSILON * (1.0 + sigmoid_value.abs()),
        "K=2 softmax Jacobian value {softmax_value:.15e} is not the sigmoid gate's \
         {sigmoid_value:.15e}"
    );
    let (gradient, curvature) = gate_logit_jacobian_grad_hdiag_weighted(&assignment, None);
    for row in 0..n {
        let expected = sigmoid(row);
        let pairs = [
            ("gradient", gradient[row * k], expected.gradient()),
            ("curvature", curvature[row * k], expected.curvature()),
        ];
        for (order, softmax, gate) in pairs {
            assert!(
                (softmax - gate).abs() <= 64.0 * f64::EPSILON * (1.0 + gate.abs()),
                "K=2 softmax Jacobian {order} at row {row}: {softmax:.15e} against the sigmoid \
                 gate's {gate:.15e}"
            );
        }
    }
}

/// A softmax row whose remaining objective charges `μ` per unit of a minority atom's
/// probability wants that atom's logit at `−∞`: along `ℓ_m`, with the other logits held,
/// `f(ℓ_m) = μ·z_m + J(ℓ)`.
///
/// Without the Jacobian, `f′ = μ·z_m(1 − z_m)/τ > 0` at every finite logit, so there is no mode.
/// With it `μ·z(1 − z) + c·z − 1 = 0`, i.e. `z* = 2/[(μ + c) + √((μ + c)² − 4μ)]`, where
/// `f″ = z*(1 − z*)·(μ(1 − 2z*) + c)/τ²`. On `z < ½`, `f″ > 0`, so `f′` increases from `−1/τ`
/// toward positive values and the mode is its one root there.
#[test]
fn a_minority_softmax_atom_has_a_finite_mode_at_the_derived_curvature_2080() {
    for (mu, temperature) in [(50.0_f64, 1.0_f64), (2.0e3, 0.3), (8.0, 1.7)] {
        let inv_tau = temperature.recip();
        // Minority atom 0 and dominant atom 1 are free; atom 2 is the reference at zero.
        let assignment = softmax_assignment(array![[0.0, 3.0, 0.0]], temperature);
        let count = crate::assignment::simplex_gate_free_count(&assignment)
            .expect("a K=3 softmax row has two free logits");
        let along = |logit: f64| {
            let mut shifted = assignment.clone();
            shifted.logits[[0, 0]] = logit;
            shifted
        };
        let minority = |logit: f64| {
            crate::assignment::softmax_row(along(logit).logits.row(0), temperature)[0]
        };
        let slope = |logit: f64| {
            let z = minority(logit);
            mu * z * (1.0 - z) * inv_tau
                + gate_logit_jacobian_grad_hdiag_weighted(&along(logit), None).0[0]
        };
        // `z₀ = ½` at `ℓ₀ = τ·ln(e^{3/τ} + 1)`, where the slope is positive; walk down until it
        // is negative, then bisect to float resolution.
        let mut high = temperature * ((3.0 * inv_tau).exp() + 1.0).ln();
        assert!(slope(high) > 0.0, "the slope at z₀ = ½ must be positive");
        let mut step = temperature;
        let mut low = high - step;
        while slope(low) >= 0.0 {
            high = low;
            step *= 2.0;
            low -= step;
            assert!(low.is_finite(), "the minority slope never turned negative");
        }
        loop {
            let middle = 0.5 * (low + high);
            if middle <= low || middle >= high {
                break;
            }
            if slope(middle) < 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
        let mode = 0.5 * (low + high);
        let z_mode = minority(mode);
        let z_derived = 2.0 / ((mu + count) + ((mu + count) * (mu + count) - 4.0 * mu).sqrt());
        let x_mode = mode * inv_tau;
        assert!(
            mode.is_finite()
                && (z_mode - z_derived).abs() <= 64.0 * f64::EPSILON * (1.0 + x_mode.abs()),
            "μ={mu}, τ={temperature}: the minority gate at the mode {z_mode:.15e} is not the \
             derived {z_derived:.15e}"
        );
        let block = crate::assignment::simplex_gate_logit_jacobian_row_block(&along(mode), None, 0)
            .expect("a free softmax row");
        let curvature =
            mu * z_mode * (1.0 - z_mode) * (1.0 - 2.0 * z_mode) * inv_tau * inv_tau + block[[0, 0]];
        let derived_curvature =
            z_derived * (1.0 - z_derived) * (mu * (1.0 - 2.0 * z_derived) + count) * inv_tau * inv_tau;
        assert!(
            (curvature - derived_curvature).abs()
                <= 64.0 * f64::EPSILON * (mu + count) * inv_tau * inv_tau * (1.0 + x_mode.abs()),
            "μ={mu}, τ={temperature}: curvature at the mode {curvature:.15e} is not the derived \
             {derived_curvature:.15e}"
        );
        // Negative control: without the Jacobian the slope below the mode stays strictly
        // positive, so the minority logit has nowhere finite to stop.
        for walk in 1..=6 {
            let logit = mode - 10.0 * walk as f64 * temperature;
            let z = minority(logit);
            let bare_slope = mu * z * (1.0 - z) * inv_tau;
            assert!(
                bare_slope > 0.0,
                "μ={mu}, τ={temperature}: without the Jacobian the slope at logit {logit} must be \
                 positive, got {bare_slope:.3e}"
            );
        }
    }
}

/// A three-atom softmax term whose atoms are one decoder on one chart, so the reconstruction's
/// logit derivative `Σ_k (∂z_k/∂ℓ_j)·f_k = f·Σ_k ∂z_k/∂ℓ_j` is zero to rounding. The data
/// Gauss--Newton block on the logits vanishes, and the installed `H_tt` logit block is the
/// entropy prior's diagonal Gershgorin majorizer plus the Jacobian's own block. Off the
/// diagonal the installed entry must be the Jacobian's `−w·c·z_i z_j/τ²`, which the fixture
/// holds away from zero; on the diagonal the remainder is non-negative. The loss field carries
/// the Jacobian value beyond the entropy prior. By AM–GM on the `c` masses that value is at
/// least `c·ln c + |F|·ln τ > 0` per row, so neither lane can pass by omitting the term.
#[test]
fn installed_softmax_logit_block_holds_the_jacobian_block_2080() {
    let temperature = 0.8_f64;
    let inv_tau = temperature.recip();
    let z = one_circle_wide_target(24, 8, 0.05);
    let mut term = two_circle_periodic_term(z.view(), 3, 1).0;
    term.assignment.mode = AssignmentMode::softmax(temperature);
    for atom in 1..3 {
        term.atoms[atom] = term.atoms[0].clone();
        term.assignment.coords[atom] = term.assignment.coords[0].clone();
    }
    for row in 0..term.assignment.logits.nrows() {
        term.assignment.logits[[row, 2]] = 0.0;
    }
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]; 3])
        .for_assignment(&term.assignment);
    let weights = term.row_loss_weights.clone();
    let n = term.assignment.logits.nrows();
    let count = crate::assignment::simplex_gate_free_count(&term.assignment)
        .expect("a K=3 softmax row has two free logits");
    let jacobian_value = gate_logit_jacobian_value_weighted(&term.assignment, weights.as_deref());
    let total_weight = weights.as_ref().map_or(n as f64, |w| w.iter().sum());
    let row_floor = count * count.ln() + (count - 1.0) * temperature.ln();
    assert!(
        row_floor > 0.0
            && jacobian_value >= total_weight * row_floor - 64.0 * f64::EPSILON * jacobian_value,
        "the softmax Jacobian value {jacobian_value:.12e} is below its AM–GM floor \
         {:.12e}",
        total_weight * row_floor
    );
    let loss = term.loss(z.view(), &rho).expect("loss at the fixture state");
    let prior = assignment_prior_value_weighted(&term.assignment, &rho, weights.as_deref())
        .expect("entropy prior value at the fixture state");
    assert!(
        (loss.assignment_sparsity - prior - jacobian_value).abs()
            <= 64.0 * f64::EPSILON * (1.0 + prior.abs() + jacobian_value.abs()),
        "the loss field must carry exactly the softmax Jacobian {jacobian_value:.12e} beyond the \
         entropy prior {prior:.12e}; it carries {:.12e}",
        loss.assignment_sparsity - prior
    );
    let mut assembled = term.clone();
    let system = assembled
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("assemble the softmax fixture's arrow system");
    for (row, block) in system.rows.iter().enumerate() {
        let weight = weights.as_ref().map_or(1.0, |w| w[row]);
        let jacobian = crate::assignment::simplex_gate_logit_jacobian_row_block(
            &term.assignment,
            weights.as_deref(),
            row,
        )
        .expect("a free softmax row");
        let slots = jacobian.nrows();
        let vars = assembled
            .row_vars_for_row_dim(row, block.htt.nrows())
            .expect("row layout of the assembled system");
        for (slot, var) in vars.iter().take(slots).enumerate() {
            assert!(
                matches!(*var, SaeLocalRowVar::Logit { atom } if atom == slot),
                "row {row}: slot {slot} must be the free logit of atom {slot}"
            );
        }
        // `z₀ = z₁ = 1/(2 + e^{−6/τ})`, so `z₀z₁ > 0.2`.
        assert!(
            -jacobian[[0, 1]] >= 0.2 * weight * count * inv_tau * inv_tau,
            "row {row}: the two dominant atoms must couple through the Jacobian, got \
             {:.12e}",
            jacobian[[0, 1]]
        );
        let scale = (0..slots)
            .map(|i| block.htt[[i, i]].abs().max(jacobian[[i, i]].abs()))
            .fold(0.0_f64, f64::max);
        let tolerance = 64.0 * f64::EPSILON * (1.0 + scale);
        for i in 0..slots {
            assert!(
                block.htt[[i, i]] - jacobian[[i, i]] >= -tolerance,
                "row {row}, slot {i}: installed logit curvature {:.12e} is below the Jacobian's \
                 {:.12e}",
                block.htt[[i, i]],
                jacobian[[i, i]]
            );
            for j in 0..slots {
                if j != i {
                    assert!(
                        (block.htt[[i, j]] - jacobian[[i, j]]).abs() <= tolerance,
                        "row {row}, slots ({i}, {j}): installed logit coupling {:.12e} is not \
                         the Jacobian's {:.12e}",
                        block.htt[[i, j]],
                        jacobian[[i, j]]
                    );
                }
            }
        }
    }
}
