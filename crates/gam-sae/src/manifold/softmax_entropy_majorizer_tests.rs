//! #2818 recovery of the derivative contracts whose public convenience
//! wrappers were deleted. The subjects here are the live assembly leafs.

use super::*;
use gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty;

fn probabilities(logits: &[f64], inv_tau: f64) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut values: Vec<f64> = logits
        .iter()
        .map(|value| ((value - maximum) * inv_tau).exp())
        .collect();
    let total: f64 = values.iter().sum();
    for value in &mut values {
        *value /= total;
    }
    values
}

fn isolated_crossing() -> ([f64; 3], f64, f64) {
    let entry = |z0: f64| {
        let a = probabilities(&[z0, 0.0, -0.7], 1.0);
        softmax_dense_entropy_hessian_entry(&a, 0, 1, softmax_majorizer_log_mean(&a), 1.3)
    };
    // Three atoms keep the rest of this Hessian row nonzero at H_01=0.
    // With two atoms gauge invariance forces the entire row to vanish there,
    // removing the very smoothing seam this fixture must exercise.
    let (mut lo, mut hi) = (-8.0_f64, 0.0_f64);
    assert!(entry(lo) < 0.0 && entry(hi) > 0.0);
    loop {
        let mid = lo + 0.5 * (hi - lo);
        if mid <= lo || mid >= hi {
            break;
        }
        let value = entry(mid);
        if value == 0.0 {
            lo = mid;
            hi = mid;
            break;
        } else if value < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let row = [lo + 0.5 * (hi - lo), 0.0, -0.7];
    let a = probabilities(&row, 1.0);
    let mean = softmax_majorizer_log_mean(&a);
    let norm = (0..3)
        .map(|j| softmax_dense_entropy_hessian_entry(&a, 0, j, mean, 1.3).powi(2))
        .sum::<f64>()
        .sqrt();
    // H_01 itself is analytic at the seam. Its independent finite difference
    // calibrates the width; no removed Hessian-derivative helper is required.
    let step = f64::EPSILON.cbrt() * (1.0 + row[0].abs());
    let slope = ((entry(row[0] + step) - entry(row[0] - step)) / (2.0 * step)).abs();
    let epsilon = SoftmaxAssignmentSparsityPenalty::soft_abs_temperature(3) * norm;
    let band = epsilon / slope;
    assert!(norm > 1e-4 && slope > 1e-4 && band > 1e-13 && band.is_finite());
    (row, band, slope)
}

#[test]
fn gershgorin_majorizer_logit_derivative_matches_fd_1419() {
    let logits = [0.3_f64, -0.6, 0.9, 0.2];
    let inv_tau = 1.0 / 0.8;
    let scale = 1.1 * inv_tau * inv_tau;
    let a = probabilities(&logits, inv_tau);
    let mean = softmax_majorizer_log_mean(&a);
    let step = 1e-6;
    let mut largest_reference = 0.0_f64;
    let mut largest_error = 0.0_f64;
    for w in 0..4 {
        let mut plus = logits;
        let mut minus = logits;
        plus[w] += step;
        minus[w] -= step;
        let ap = probabilities(&plus, inv_tau);
        let am = probabilities(&minus, inv_tau);
        let mp = softmax_majorizer_log_mean(&ap);
        let mm = softmax_majorizer_log_mean(&am);
        for kk in 0..4 {
            let actual =
                active_softmax_majorizer_logit_derivative_entry(&a, kk, w, mean, scale, inv_tau);
            let reference = (active_softmax_gershgorin_majorizer_entry(&ap, kk, mp, scale)
                - active_softmax_gershgorin_majorizer_entry(&am, kk, mm, scale))
                / (plus[w] - minus[w]);
            largest_reference = largest_reference.max(reference.abs());
            largest_error = largest_error.max((actual - reference).abs());
            assert!(
                (actual - reference).abs() < 1e-6,
                "atom={kk} logit={w}: analytic={actual} FD={reference}"
            );
        }
    }
    assert!(
        largest_reference > 1e-3,
        "a zero derivative must fail this oracle"
    );
    eprintln!(
        "#1419 adjoint FD max_error={largest_error:.6e} max_reference={largest_reference:.6e}"
    );
}

#[test]
fn smooth_gershgorin_adjoint_is_continuous_across_a_zero_crossing_2339() {
    let (row, band, slope) = isolated_crossing();
    let adjoint = |logits: &[f64; 3]| {
        let a = probabilities(logits, 1.0);
        active_softmax_majorizer_logit_derivative_entry(
            &a,
            0,
            0,
            softmax_majorizer_log_mean(&a),
            1.3,
            1.0,
        )
    };
    // The removed hard-radius derivative is an independent counterfactual.
    // Its Hessian-entry slopes are differenced through the smooth production
    // Hessian, so no old test-support derivative routine is reintroduced.
    let hard_adjoint = |logits: &[f64; 3]| {
        let a = probabilities(logits, 1.0);
        let mean = softmax_majorizer_log_mean(&a);
        let step = f64::EPSILON.cbrt() * (1.0 + logits[0].abs());
        let mut plus = *logits;
        let mut minus = *logits;
        plus[0] += step;
        minus[0] -= step;
        let ap = probabilities(&plus, 1.0);
        let am = probabilities(&minus, 1.0);
        let mp = softmax_majorizer_log_mean(&ap);
        let mm = softmax_majorizer_log_mean(&am);
        (0..3)
            .map(|j| {
                let value = softmax_dense_entropy_hessian_entry(&a, 0, j, mean, 1.3);
                let derivative = (softmax_dense_entropy_hessian_entry(&ap, 0, j, mp, 1.3)
                    - softmax_dense_entropy_hessian_entry(&am, 0, j, mm, 1.3))
                    / (plus[0] - minus[0]);
                if value == 0.0 {
                    0.0
                } else {
                    value.signum() * derivative
                }
            })
            .sum::<f64>()
    };
    let mut smooth_jumps = Vec::new();
    for divisor in [100.0, 1000.0] {
        let mut plus = row;
        let mut minus = row;
        plus[0] += band / divisor;
        minus[0] -= band / divisor;
        assert!(plus[0] > minus[0], "the seam probe must survive rounding");
        let smooth = (adjoint(&plus) - adjoint(&minus)).abs();
        let hard = (hard_adjoint(&plus) - hard_adjoint(&minus)).abs();
        assert!(
            hard >= slope,
            "counterfactual hard radius must retain its jump"
        );
        assert!(smooth <= 0.05 * hard, "smooth={smooth} hard={hard}");
        smooth_jumps.push(smooth);
        eprintln!(
            "#2339 crossing divisor={divisor} smooth_jump={smooth:.6e} hard_jump={hard:.6e} band={band:.6e}"
        );
    }
    assert!(smooth_jumps[1] <= 0.25 * smooth_jumps[0]);
}

#[test]
fn smooth_gershgorin_adjoint_matches_fd_inside_the_smoothing_band_2339() {
    let (row, band, slope) = isolated_crossing();
    let a = probabilities(&row, 1.0);
    let mean = softmax_majorizer_log_mean(&a);
    let step = band / 100.0;
    let mut largest_error = 0.0_f64;
    assert!(step > 1e-15 && slope > 1e-4);
    for w in 0..3 {
        let mut plus = row;
        let mut minus = row;
        plus[w] += step;
        minus[w] -= step;
        assert!(plus[w] > minus[w]);
        let ap = probabilities(&plus, 1.0);
        let am = probabilities(&minus, 1.0);
        let mp = softmax_majorizer_log_mean(&ap);
        let mm = softmax_majorizer_log_mean(&am);
        for kk in 0..3 {
            let analytic =
                active_softmax_majorizer_logit_derivative_entry(&a, kk, w, mean, 1.3, 1.0);
            let reference = (active_softmax_gershgorin_majorizer_entry(&ap, kk, mp, 1.3)
                - active_softmax_gershgorin_majorizer_entry(&am, kk, mm, 1.3))
                / (plus[w] - minus[w]);
            largest_error = largest_error.max((analytic - reference).abs());
            assert!(
                (analytic - reference).abs() < 1e-3,
                "atom={kk} logit={w}: analytic={analytic} FD={reference}"
            );
        }
    }
    eprintln!(
        "#2339 in-band FD max_error={largest_error:.6e} step={step:.6e} crossing_slope={slope:.6e}"
    );
}

#[test]
fn smooth_gershgorin_adjoint_is_degree_one_homogeneous_in_scale_2339() {
    let a = probabilities(&[0.3, -0.6, 0.9, 0.2, -1.4], 1.0 / 0.9);
    let mean = softmax_majorizer_log_mean(&a);
    let mut largest = 0.0_f64;
    for kk in 0..5 {
        for w in 0..5 {
            let base =
                active_softmax_majorizer_logit_derivative_entry(&a, kk, w, mean, 0.625, 1.0 / 0.9);
            largest = largest.max(base.abs());
            for factor in [0.5, 2.0, 4.0] {
                let scaled = active_softmax_majorizer_logit_derivative_entry(
                    &a,
                    kk,
                    w,
                    mean,
                    factor * 0.625,
                    1.0 / 0.9,
                );
                assert_eq!(scaled, factor * base, "atom={kk} logit={w} factor={factor}");
            }
        }
    }
    assert!(largest > 1e-3);
}

#[test]
fn smooth_gershgorin_adjoint_is_exactly_zero_on_an_underflowed_atom_2339() {
    let a = probabilities(&[0.0, -800.0, -1.0], 1.0);
    let mean = softmax_majorizer_log_mean(&a);
    assert_eq!(a[1], 0.0);
    assert!(active_softmax_gershgorin_majorizer_entry(&a, 0, mean, 2.0) > 0.01);
    let mut largest_live = 0.0_f64;
    for w in 0..3 {
        assert_eq!(
            active_softmax_majorizer_logit_derivative_entry(&a, 1, w, mean, 2.0, 1.0),
            0.0
        );
        for kk in [0, 2] {
            let value = active_softmax_majorizer_logit_derivative_entry(&a, kk, w, mean, 2.0, 1.0);
            assert!(value.is_finite());
            largest_live = largest_live.max(value.abs());
        }
    }
    assert!(
        largest_live > 1e-3,
        "live atoms must exercise a nonzero derivative"
    );
}
