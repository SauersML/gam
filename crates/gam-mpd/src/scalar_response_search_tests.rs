use super::*;
fn budget() -> Budget {
    Budget {
        max_cells: 4096,
        max_evaluations: 4096,
    }
}
#[test]
fn proportional_and_negative_scales_enclose_zero_optimum() {
    for a in [3.0, -2.0] {
        let r = search(&[vec![1.0, 2.0]], &[vec![a, 2.0 * a]], budget())
            .expect("valid proportional problem");
        assert_eq!(r.best_amplitude_bits, Some((a as f32).to_bits()));
        assert!(r.upper_bound.expect("upper") >= 0.0);
        assert_eq!(r.lower_bound, 0.0);
        assert!(r.complete_finite_inventory);
        assert_eq!(
            r.total_finite_keys,
            r.lower_bound_excluded_keys
                + r.constant_response_equivalent_keys
                + r.tested.len() as u64
        );
    }
}
#[test]
fn conflicting_rows_choose_minimax_not_least_squares() {
    let r = search(&[vec![1.0], vec![2.0]], &[vec![0.0], vec![4.0]], budget())
        .expect("valid conflicting rows");
    let a = f32::from_bits(r.best_amplitude_bits.expect("tested upper exists"));
    assert_eq!(a, (4.0_f64 / 3.0) as f32);
    assert_ne!(a, 1.6_f32);
    let exact_finite_score = f64::from(a).powi(2).max((2.0 * f64::from(a) - 4.0).powi(2));
    assert!(r.lower_bound <= exact_finite_score);
    assert!(r.upper_bound.expect("upper") >= exact_finite_score);
    assert!(r.upper_bound.expect("upper") >= 16.0 / 9.0);
    assert!(
        r.complete_finite_inventory,
        "{} remaining cells",
        r.unexplored.len()
    );
    // Finite-inventory closure need not make the outward arithmetic gap zero.
    assert!(r.gap_upper_bound.expect("gap") >= 0.0);
}
#[test]
fn zero_responses_constant_objective_and_signed_zero_inventory() {
    let r = search(&[vec![0.0, 0.0]], &[vec![3.0, 4.0]], budget()).expect("constant objective");
    assert!(r.upper_bound.expect("upper") >= 25.0);
    assert!(r.lower_bound <= 25.0);
    assert!(r.complete_finite_inventory);
    assert_eq!(
        r.constant_response_equivalent_keys + r.tested.len() as u64,
        r.total_finite_keys
    );
    assert_eq!(amplitude(FIRST), -f32::MAX);
    assert_eq!(amplitude(LAST), f32::MAX);
    assert_eq!(amplitude(key(-0.0)).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(key(0.0), key(-0.0) + 1);
}
#[test]
fn exhausted_budget_keeps_every_unexplored_key_and_positive_gap() {
    let b = Budget {
        max_cells: 1,
        max_evaluations: 1,
    };
    let r = search(&[vec![1.0], vec![2.0]], &[vec![0.0], vec![4.0]], b).expect("bounded search");
    assert!(!r.complete_finite_inventory);
    assert!(r.gap_upper_bound.expect("tested upper") > 0.0);
    let retained: u64 = r
        .unexplored
        .iter()
        .map(|c| u64::from(c.last_key) - u64::from(c.first_key) + 1)
        .sum();
    assert_eq!(
        retained + r.tested.len() as u64 + r.lower_bound_excluded_keys,
        u64::from(LAST) - u64::from(FIRST) + 1
    );
    assert!(
        r.unexplored
            .iter()
            .all(|c| c.lower_bound <= r.upper_bound.expect("upper"))
    );
    let zero = search(
        &[vec![1.0]],
        &[vec![2.0]],
        Budget {
            max_cells: 0,
            max_evaluations: 0,
        },
    )
    .expect("zero budget retained");
    assert_eq!(zero.unexplored.len(), 1);
    assert!(zero.upper_bound.is_none());
    assert_eq!(zero.lower_bound, 0.0);
}
#[test]
fn extreme_and_nonfinite_inputs_remain_unresolved() {
    for x in [1e200, 1e-200, 1e-155, f64::INFINITY, f64::NAN] {
        let r = search(&[vec![x]], &[vec![1.0]], budget())
            .expect("numeric failures are evidence states");
        assert!(!r.complete_finite_inventory);
        assert_eq!(r.unexplored.len(), 1);
        assert!(r.upper_bound.is_none());
        assert_eq!(r.lower_bound, 0.0);
    }
}
#[test]
fn cell_bounds_enclose_independently_evaluated_finite_points() {
    let v = [vec![1.0, -2.0], vec![-3.0, 0.5]];
    let y = [vec![2.0, 1.0], vec![0.0, -4.0]];
    let q = quadratics(&v, &y).expect("finite coefficients");
    for (lo, hi) in [(-4.0, -1.0), (-1.0, 1.0), (1.0, 4.0)] {
        let bound = cell_bound(&v, &y, &q, key(lo), key(hi));
        for a in [lo, lo * 0.5 + hi * 0.5, hi] {
            assert!(bound <= point(&v, &y, a).expect("test point").hi);
        }
    }
}
