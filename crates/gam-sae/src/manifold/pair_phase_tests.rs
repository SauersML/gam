// Tests for the F4 phase-coupling screen + App D phase circuits. Included from
// `pair_phase.rs` via `include!` so the helpers below share its private items.

use super::*;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-12);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

const NOISE: f64 = 0.02;

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

