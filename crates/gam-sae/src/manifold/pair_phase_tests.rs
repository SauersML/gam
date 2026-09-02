// Tests for the F4 phase-coupling screen + App D phase circuits. Included from
// `pair_phase.rs` via `include!` so the helpers below share its private items.

use super::*;

/// e-BH unit check: with a single huge e-value in a family of nulls (e≈1), only the
/// large one is rejected at α=0.05.
#[test]
fn ebh_rejects_dominant_e_value() {
    let mut es = vec![1.0_f64; 20];
    es[7] = 500.0;
    let rej = ebh_reject(&es, 0.05);
    assert_eq!(rej, vec![7], "only the dominant e-value clears m/(αk)");
    // No discoveries when nothing dominates.
    let flat = vec![1.0_f64; 20];
    assert!(ebh_reject(&flat, 0.05).is_empty());
}

