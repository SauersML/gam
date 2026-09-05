//! Recovered #2818 contracts for the live entropy-curvature operator.
//!
//! The sweep removed the dense-Hessian and majorizer-adjoint convenience
//! methods. These tests use the production HVP and value interfaces directly;
//! no deleted API is reintroduced just to make an old fixture compile.

use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::utils::{SPECTRAL_DEFLATION_REL_FLOOR, splitmix64};
use gam_terms::analytic_penalties::{AnalyticPenalty, SoftmaxAssignmentSparsityPenalty};
use ndarray::{Array1, Array2, array};

#[test]
fn gershgorin_majorizes_entropy_where_fisher_does_not_1419() {
    let penalty = SoftmaxAssignmentSparsityPenalty::new(2, 1.0);
    let row = array![19.0_f64.ln(), 0.0];
    let rho = array![0.0];
    let direction = array![1.0, -1.0];
    let hv = penalty.hvp(row.view(), rho.view(), direction.view());
    // Independent scalar entropy differentiation at p=0.95. This is also
    // the issue's counterexample: Fisher curvature 0.0475 is too small.
    let p = 0.95_f64;
    let fisher = p * (1.0 - p);
    let entropy = fisher * ((2.0 * p - 1.0) * (p / (1.0 - p)).ln() - 1.0);
    assert!((entropy - 0.0783747664).abs() < 1e-9);
    assert!((hv[0] - 2.0 * entropy).abs() < 1e-14);
    assert!((hv[1] + 2.0 * entropy).abs() < 1e-14);
    assert!(4.0 * (fisher - entropy) < -0.1);
    let majorizer = penalty.row_psd_majorizer(row.as_slice().unwrap(), 1.0);
    let gap = direction.dot(&majorizer.dot(&direction)) - direction.dot(&hv);
    assert!(gap >= -1e-14, "Gershgorin must majorize entropy: gap={gap}");
    assert_eq!(majorizer[[0, 1]], 0.0);
    assert_eq!(majorizer[[1, 0]], 0.0);
    assert!(majorizer[[0, 0]] > entropy && majorizer[[1, 1]] > entropy);
}

#[test]
fn smooth_gershgorin_majorizes_entropy_within_the_derived_budget_2339() {
    let mut checked_rows = 0;
    let mut max_roundoff_budget_fraction = 0.0_f64;
    let mut max_majorization_deficit_in_roundoff = 0.0_f64;
    for (k, temperature, scale) in [(2, 1.0, 1.0), (3, 0.75, 2.5), (5, 1.4, 0.3), (8, 0.6, 1.7)] {
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        let rho = array![(scale * temperature * temperature).ln()];
        let mut seed = 0x2339_0000 + k as u64;
        for fixture in 0..8 {
            let row = Array1::from_shape_fn(k, |axis| match fixture {
                0 => 0.02,
                1 => {
                    if axis == 0 {
                        3.0
                    } else if axis == k / 2 {
                        2.25
                    } else {
                        -4.5
                    }
                }
                _ => ((splitmix64(&mut seed) >> 11) as f64) / ((1_u64 << 53) as f64) * 8.0 - 4.0,
            });
            // Recover the exact Hessian through independently accumulated HVP
            // columns, not the deleted helper sharing the radius's expression.
            let mut hessian = Array2::<f64>::zeros((k, k));
            for axis in 0..k {
                let mut direction = Array1::<f64>::zeros(k);
                direction[axis] = 1.0;
                hessian.column_mut(axis).assign(&penalty.hvp(
                    row.view(),
                    rho.view(),
                    direction.view(),
                ));
            }
            let diagonal = penalty.psd_majorizer_abs_row_sums(row.as_slice().unwrap(), scale);
            let magnitude = hessian
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                magnitude > 1e-6,
                "a zero Hessian cannot exercise majorization"
            );
            // Entropy Hessian entries can be small because large terms cancel.
            // Bound arithmetic error from the ABSOLUTE, uncontracted terms,
            // not from the final curvature. For H_ij the HVP expands as
            // scale*a_i*((delta_ij-a_j)*(m-L_i-1)
            //            + sum_l a_l*(delta_lj-a_j)*L_l).
            // Replace every summand by its magnitude before any cancellation.
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut probabilities = row.mapv(|value| ((value - maximum) / temperature).exp());
            probabilities /= probabilities.sum();
            let logs = probabilities.mapv(|value| value.ln() + 1.0);
            let absolute_mean = probabilities
                .iter()
                .zip(logs.iter())
                .map(|(&probability, &log)| probability * log.abs())
                .sum::<f64>();
            let mut work_scales = vec![0.0_f64; k];
            for i in 0..k {
                let mut row_work = 0.0;
                for j in 0..k {
                    let centered = ((if i == j { 1.0 } else { 0.0 }) - probabilities[j]).abs();
                    let centered_mean = (0..k)
                        .map(|l| {
                            probabilities[l]
                                * ((if l == j { 1.0 } else { 0.0 }) - probabilities[j]).abs()
                                * logs[l].abs()
                        })
                        .sum::<f64>();
                    row_work += scale
                        * probabilities[i]
                        * (centered * (absolute_mean + logs[i].abs() + 1.0) + centered_mean);
                }
                work_scales[i] = row_work;
            }
            // At most 32K+32 operations on a dependency path per algebraic
            // form: softmax/normalization, entropy means, curvature, row sum.
            // Both independently rounded forms contribute to the comparison.
            let operations = 2.0 * (32 * k + 32) as f64;
            let accumulated_roundoff = operations * (f64::EPSILON / 2.0);
            let arithmetic_factor = accumulated_roundoff / (1.0 - accumulated_roundoff);
            let mut spectral_rounding = 0.0_f64;
            for axis in 0..k {
                // A nearly inactive atom's row has its own small arithmetic
                // scale. Charging it another atom's cancellation budget would
                // make the smoothing comparison unresolved unnecessarily.
                let rounding = arithmetic_factor * work_scales[axis];
                spectral_rounding = spectral_rounding.max(rounding);
                let hard = hessian
                    .row(axis)
                    .iter()
                    .map(|value| value.abs())
                    .sum::<f64>();
                assert!(
                    rounding < 0.01 * SPECTRAL_DEFLATION_REL_FLOOR * hard,
                    "the arithmetic oracle must resolve the smoothing budget"
                );
                max_roundoff_budget_fraction = max_roundoff_budget_fraction
                    .max(rounding / (SPECTRAL_DEFLATION_REL_FLOOR * hard));
                max_majorization_deficit_in_roundoff = max_majorization_deficit_in_roundoff
                    .max((hard - diagonal[axis]).max(0.0) / rounding);
                assert!(diagonal[axis] >= 0.0);
                assert!(
                    diagonal[axis] + rounding >= hard,
                    "k={k} fixture={fixture} axis={axis}: radius={} < hard={hard}",
                    diagonal[axis]
                );
                assert!(
                    diagonal[axis] - hard <= SPECTRAL_DEFLATION_REL_FLOOR * hard + rounding,
                    "smoothing exceeded its curvature-resolution budget"
                );
            }
            let mut gap = -hessian;
            for axis in 0..k {
                gap[[axis, axis]] += diagonal[axis];
            }
            let (spectrum, _) = gap
                .eigh(faer::Side::Lower)
                .expect("majorizer gap eigensolve");
            assert!(
                spectrum
                    .iter()
                    .all(|&eigenvalue| eigenvalue >= -spectral_rounding)
            );
            checked_rows += 1;
        }
    }
    assert_eq!(checked_rows, 32);
    eprintln!(
        "#2339 rows={checked_rows} max_roundoff_budget_fraction={max_roundoff_budget_fraction:.6e} max_majorization_deficit_in_roundoff={max_majorization_deficit_in_roundoff:.6e}"
    );
}

#[test]
fn smooth_gershgorin_is_degree_one_homogeneous_in_scale_2339() {
    let penalty = SoftmaxAssignmentSparsityPenalty::new(5, 0.9);
    let mut seed = 0x2339_0100;
    for _ in 0..8 {
        let row: Vec<f64> = (0..5)
            .map(|_| ((splitmix64(&mut seed) >> 11) as f64) / ((1_u64 << 53) as f64) * 8.0 - 4.0)
            .collect();
        let base = penalty.psd_majorizer_abs_row_sums(&row, 0.625);
        assert!(base.iter().any(|&value| value > 1e-4));
        for factor in [0.5, 2.0, 4.0] {
            let scaled = penalty.psd_majorizer_abs_row_sums(&row, factor * 0.625);
            for axis in 0..5 {
                assert_eq!(scaled[axis], factor * base[axis]);
            }
        }
        let scaled = penalty.psd_majorizer_abs_row_sums(&row, 3.5 * 0.625);
        for axis in 0..5 {
            assert!((scaled[axis] - 3.5 * base[axis]).abs() <= 32.0 * f64::EPSILON * scaled[axis]);
        }
    }
}

#[test]
fn smooth_gershgorin_weighting_routes_agree_by_scale_equivariance_2339() {
    let temperature = 1.1_f64;
    let row = array![0.3, -0.6, 0.9, 0.2, -1.4, 1.7];
    let rho = array![(0.75 * temperature * temperature).ln()];
    let scale = rho[0].exp() * (1.0 / temperature) * (1.0 / temperature);
    for weight in [0.5, 2.0, 4.0, 0.37, 1.9, 6.25] {
        let penalty =
            SoftmaxAssignmentSparsityPenalty::new(6, temperature).with_row_weights(Some(&[weight]));
        // Compare the actual trait channel to the arrow assembly's folded
        // strength. This exercises the two consumers, not one helper twice.
        let trait_diagonal = penalty.psd_majorizer_diag(row.view(), rho.view()).unwrap();
        let assembly = penalty.row_psd_majorizer(row.as_slice().unwrap(), scale * weight);
        assert!(trait_diagonal.iter().all(|&value| value > 0.0));
        for axis in 0..6 {
            let actual = assembly[[axis, axis]];
            let expected = trait_diagonal[axis];
            assert!((actual - expected).abs() <= 32.0 * f64::EPSILON * expected);
        }
    }
}

#[test]
fn smooth_gershgorin_is_exactly_zero_on_an_underflowed_atom_2339() {
    let penalty = SoftmaxAssignmentSparsityPenalty::new(3, 1.0);
    let row = array![0.0, -800.0, -1.0];
    let rho = array![2.0_f64.ln()];
    let diagonal = penalty.psd_majorizer_abs_row_sums(row.as_slice().unwrap(), 2.0);
    assert_eq!(diagonal[1], 0.0);
    assert!(diagonal[0] > 0.01 && diagonal[2] > 0.01);
    let majorizer = penalty.row_psd_majorizer(row.as_slice().unwrap(), 2.0);
    let gradient = penalty.grad_target(row.view(), rho.view());
    assert_eq!(gradient[1], 0.0);
    for axis in 0..3 {
        assert_eq!(majorizer[[1, axis]], 0.0);
        assert_eq!(majorizer[[axis, 1]], 0.0);
        let mut direction = Array1::<f64>::zeros(3);
        direction[axis] = 1.0;
        let hv = penalty.hvp(row.view(), rho.view(), direction.view());
        assert_eq!(hv[1], 0.0);
        assert!(hv.iter().all(|value| value.is_finite()));
    }
}
