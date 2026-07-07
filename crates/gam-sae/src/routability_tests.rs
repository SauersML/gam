//! Tests for the routability / interference-floor diagnostic.
//!
//! The load-bearing checks are the Monte-Carlo agreement between the closed-form
//! floor and the empirical max cross-gate of a random dictionary
//! ([`monte_carlo_floor_bounds_and_tightness`]) and the plant-and-route phase
//! transition around the floor ([`phase_transition_at_floor`]). Both use a seeded
//! LCG + Box–Muller Gaussian generator (no `rand` dependency → bit-reproducible),
//! matching the house convention in the block lane's tests.

use super::*;
use ndarray::{Array2, ArrayView1};

/// Deterministic LCG uniform in `(0, 1)` (same multiplier/increment as the block
/// lane's test LCG, remapped to the open unit interval for Box–Muller).
fn lcg_u01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Top 53 bits → a double in (0, 1); +1 keeps it strictly positive for the log.
    (((*state >> 11) as f64) + 1.0) / ((1u64 << 53) as f64 + 1.0)
}

/// One standard-normal sample via Box–Muller from the LCG.
fn gaussian(state: &mut u64) -> f64 {
    let u1 = lcg_u01(state);
    let u2 = lcg_u01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A `rows × p` matrix of unit-norm rows (isotropic Gaussian directions).
fn unit_rows(rows: usize, p: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    let mut m = Array2::<f32>::zeros((rows, p));
    for i in 0..rows {
        let mut norm2 = 0.0f64;
        for c in 0..p {
            let v = gaussian(&mut s);
            m[[i, c]] = v as f32;
            norm2 += v * v;
        }
        let inv = 1.0 / norm2.sqrt().max(1.0e-30);
        for c in 0..p {
            m[[i, c]] = (m[[i, c]] as f64 * inv) as f32;
        }
    }
    m
}

/// Max cross-gate `max_g ‖r D_gᵀ‖₂ / ‖r‖₂` of one row against a `block_size`-lane
/// dictionary — the same quantity `routability_audit` accumulates, recomputed
/// standalone so the Monte-Carlo test does not depend on the audit under test.
fn max_cross_gate(row: ArrayView1<'_, f32>, decoder: &Array2<f32>, block_size: usize) -> f64 {
    let n_blocks = decoder.nrows() / block_size;
    let mut norm2 = 0.0f64;
    for &v in row.iter() {
        norm2 += v as f64 * v as f64;
    }
    let norm = norm2.sqrt();
    let mut best = 0.0f64;
    for g in 0..n_blocks {
        let mut energy = 0.0f64;
        for r in 0..block_size {
            let atom = decoder.row(g * block_size + r);
            let mut dot = 0.0f64;
            for (rv, av) in row.iter().zip(atom.iter()) {
                dot += *rv as f64 * *av as f64;
            }
            energy += dot * dot;
        }
        best = best.max(energy.sqrt());
    }
    best / norm.max(1.0e-30)
}

#[test]
fn closed_form_matches_derivation() {
    // floor = √(b_max/p) + √(2·ln(K/δ)/p), δ = 1 recovers √(b/p)+√(2 ln K/p).
    let p = 256usize;
    let k = 2000usize;
    let f = routability_floor(p, k, 1, 1.0);
    let expect = (1.0f64 / p as f64).sqrt() + (2.0 * (k as f64).ln() / p as f64).sqrt();
    assert!(
        (f.floor - expect).abs() < 1.0e-12,
        "δ=1 floor {} != derivation {expect}",
        f.floor
    );
    // Smaller δ widens the floor by exactly √(2·ln(1/δ)/p).
    let f01 = routability_floor(p, k, 1, 0.01);
    let widen = (2.0 * (1.0f64 / 0.01).ln() / p as f64).sqrt();
    // √(a) + extra vs √(a + more): compare the union terms directly.
    let union_1 = (2.0 * (k as f64).ln() / p as f64).sqrt();
    let union_01 = (2.0 * ((k as f64) / 0.01).ln() / p as f64).sqrt();
    assert!(f01.floor > f.floor, "smaller δ must widen the floor");
    assert!(
        ((union_01 * union_01) - (union_1 * union_1) - (widen * widen)).abs() < 1.0e-12,
        "union term must grow by ln(1/δ) in the squared scale"
    );
    // Minimum routable energy is a monotone image of the floor in (0,1).
    let e = minimum_routable_energy(&f);
    assert!((0.0..1.0).contains(&e), "energy fraction out of (0,1): {e}");
    assert!(
        (e - f.floor * f.floor / (1.0 + f.floor * f.floor)).abs() < 1.0e-12,
        "minimum_routable_energy must equal floor²/(1+floor²)"
    );
}

#[test]
fn monte_carlo_floor_bounds_and_tightness() {
    // K=2000 unit atoms in p=256, random unit residuals. The closed-form floor at
    // δ=0.01 must upper-bound the empirical max cross-gate for ≥99% of rows, and
    // the δ=0.5 floor must be TIGHT: within a factor of 2 of the empirical 99th
    // percentile (a loose bound would make the diagnostic useless).
    let (p, k, n) = (256usize, 2000usize, 4000usize);
    let decoder = unit_rows(k, p, 0xF100_D001);
    let residuals = unit_rows(n, p, 0x0FF5_E7ED);

    let mut gates: Vec<f64> = residuals
        .outer_iter()
        .map(|r| max_cross_gate(r, &decoder, 1))
        .collect();
    gates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let f01 = routability_floor(p, k, 1, 0.01).floor;
    let f5 = routability_floor(p, k, 1, 0.5).floor;

    let frac_below = gates.iter().filter(|&&g| g <= f01).count() as f64 / n as f64;
    assert!(
        frac_below >= 0.99,
        "δ=0.01 floor {f01:.4} must bound ≥99% of rows; got {frac_below:.4}"
    );

    let q99 = gates[((0.99 * (n - 1) as f64).round() as usize).min(n - 1)];
    let ratio = q99 / f5;
    assert!(
        (0.5..2.0).contains(&ratio),
        "δ=0.5 floor {f5:.4} must be within a factor of 2 of the empirical q99 {q99:.4} \
         (ratio {ratio:.3})"
    );
    // Sanity: the audit reproduces the same picture. Its coherence_excess is the
    // empirical (1−δ)-quantile over the closed-form floor; at δ=0.5 that is the
    // MEDIAN max cross-gate over the δ=0.5 floor, which must match our own gates.
    let audit = routability_audit(decoder.view(), residuals.view(), 1, 0.5, &[0.5, 0.99]).unwrap();
    assert_eq!(audit.n_rows, n);
    let median = gates[((0.5 * (n - 1) as f64).round() as usize).min(n - 1)];
    assert!(
        (audit.coherence_excess - median / f5).abs() < 1.0e-9,
        "audit coherence_excess {} must equal (median max-cross-gate)/floor {}",
        audit.coherence_excess,
        median / f5
    );
    assert!(
        (0.5..2.0).contains(&audit.coherence_excess),
        "random dictionary must sit near the generic-position floor (excess {:.3})",
        audit.coherence_excess
    );
}

#[test]
fn phase_transition_at_floor() {
    // Plant a target atom firing at amplitude a·floor·ν (ν = 1) against a random
    // dictionary + unit interference, and route by argmax gate. At 0.5× the floor
    // routing fails mostly; at 2× it succeeds essentially always.
    let (p, k) = (256usize, 2000usize);
    let f = routability_floor(p, k, 1, 0.5).floor;
    let trials = 300usize;

    let run = |a: f64, seed0: u64| -> f64 {
        let mut successes = 0usize;
        for t in 0..trials {
            let seed = seed0.wrapping_add((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let decoder = unit_rows(k, p, seed);
            let target = unit_rows(1, p, seed ^ 0xA5A5_A5A5);
            let interference = unit_rows(1, p, seed ^ 0x5A5A_5A5A);
            let amp = a * f; // amplitude a·floor, interference norm ν = 1
            let mut x = Array2::<f32>::zeros((1, p));
            for c in 0..p {
                x[[0, c]] = amp as f32 * target[[0, c]] + interference[[0, c]];
            }
            let g_target = {
                let mut d = 0.0f64;
                for c in 0..p {
                    d += x[[0, c]] as f64 * target[[0, c]] as f64;
                }
                d.abs()
            };
            let g_off = max_cross_gate(x.row(0), &decoder, 1)
                * {
                    // max_cross_gate divides by ‖x‖; undo it to compare raw gates.
                    let mut n2 = 0.0f64;
                    for c in 0..p {
                        n2 += x[[0, c]] as f64 * x[[0, c]] as f64;
                    }
                    n2.sqrt()
                };
            if g_target > g_off {
                successes += 1;
            }
        }
        successes as f64 / trials as f64
    };

    let below = run(0.5, 0x1234_5678);
    let above = run(2.0, 0x8765_4321);
    assert!(
        below < 0.5,
        "at 0.5× the floor routing should fail mostly; success frac {below:.3}"
    );
    assert!(
        above > 0.9,
        "at 2× the floor routing should succeed mostly; success frac {above:.3}"
    );
}

#[test]
fn perfect_reconstruction_zero_residual_is_a_defined_audit_not_an_error() {
    // A dictionary that reconstructs the data exactly leaves an all-zero residual:
    // there is no residual mass left to (mis)route. That is the ideal — fully
    // routable — case and must return a DEFINED audit, not an error. (Regression:
    // the empty-`per_row` branch used to `Err("... no residual rows with positive
    // norm")`, which made a perfect SAE reconstruction unauditable.)
    let (p, k, n) = (4usize, 6usize, 10usize);
    let decoder = unit_rows(k, p, 0xDEAD_BEEF);
    // Residuals identically zero (float 0.0 → norm 0 → every row skipped).
    let residuals = Array2::<f32>::zeros((n, p));

    let audit = routability_audit(decoder.view(), residuals.view(), 1, 0.05, &[0.5, 0.9])
        .expect("all-zero residual is a valid perfect-reconstruction audit, not an error");

    // No residual rows were audited, but the floor (geometry only) is still real.
    assert_eq!(audit.n_rows, 0);
    let expected_floor = routability_floor(p, k, 1, 0.05);
    assert!((audit.floor.floor - expected_floor.floor).abs() < 1.0e-12);
    // Zero unroutable mass: every empirical cross-gate is 0, and the full (zero)
    // residual mass sits trivially at/below the floor.
    assert_eq!(audit.empirical_mean, 0.0);
    assert_eq!(audit.empirical_max, 0.0);
    assert_eq!(audit.confidence_quantile, 0.0);
    assert_eq!(audit.coherence_excess, 0.0);
    assert_eq!(audit.fraction_below_floor, 1.0);
    for &(level, value) in &audit.quantiles {
        assert_eq!(value, 0.0, "quantile at level {level} must be 0 with no residual");
    }
    assert_eq!(audit.quantiles.len(), 2);

    // Near-zero (below the 1e-12 skip threshold) residuals behave identically.
    // Per-element 1e-13 over p=4 columns → row norm 2e-13 < 1e-12, so every row skips.
    let tiny = Array2::<f32>::from_elem((n, p), 1.0e-13_f32);
    let tiny_audit = routability_audit(decoder.view(), tiny.view(), 1, 0.05, &[0.5, 0.9])
        .expect("sub-threshold residual is still a defined audit");
    assert_eq!(tiny_audit.n_rows, 0);
    assert_eq!(tiny_audit.fraction_below_floor, 1.0);
}

#[test]
fn block_variant_floor_and_audit() {
    // Block lane, b=2 orthonormal frames. The b_max=2 floor at δ=0.01 must bound
    // the empirical max block cross-gate for ≥99% of rows, and the audit must
    // report the block floor with the b_max term included.
    let (p, g, b, n) = (256usize, 1000usize, 2usize, 3000usize);
    let decoder = block_frames(g, b, p, 0xB10C_C0DE);
    let residuals = unit_rows(n, p, 0x0FF5_B10C);

    let mut gates: Vec<f64> = residuals
        .outer_iter()
        .map(|r| max_cross_gate(r, &decoder, b))
        .collect();
    gates.sort_by(|a, x| a.partial_cmp(x).unwrap());

    let f01 = routability_floor(p, g, b, 0.01);
    // The b=2 floor must exceed the b=1 floor at the same K (larger subspace term).
    let f01_b1 = routability_floor(p, g, 1, 0.01);
    assert!(
        f01.floor > f01_b1.floor,
        "b=2 floor {} must exceed b=1 floor {}",
        f01.floor,
        f01_b1.floor
    );
    let frac_below = gates.iter().filter(|&&x| x <= f01.floor).count() as f64 / n as f64;
    assert!(
        frac_below >= 0.99,
        "b=2 δ=0.01 floor {:.4} must bound ≥99% of rows; got {frac_below:.4}",
        f01.floor
    );

    let audit = routability_audit(decoder.view(), residuals.view(), b, 0.5, &[0.5, 0.99]).unwrap();
    assert_eq!(audit.floor.b_max, b);
    assert_eq!(audit.floor.n_blocks, g);
    assert!(
        (0.5..2.0).contains(&audit.coherence_excess),
        "random block frames must sit near the generic-position floor (excess {:.3})",
        audit.coherence_excess
    );
    // Frames really are orthonormal within each block (D_g D_gᵀ = I_b).
    for gg in 0..3 {
        for r1 in 0..b {
            for r2 in 0..b {
                let mut dot = 0.0f64;
                for c in 0..p {
                    dot += decoder[[gg * b + r1, c]] as f64 * decoder[[gg * b + r2, c]] as f64;
                }
                let want = if r1 == r2 { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1.0e-4,
                    "block {gg} frame not orthonormal: <{r1},{r2}> = {dot}"
                );
            }
        }
    }
}

/// A `G·b × P` decoder of random orthonormal `b`-frames (each block's `b` rows
/// Gram–Schmidt orthonormalised from independent Gaussians), block `g` in rows
/// `[g·b, g·b+b)`.
fn block_frames(g: usize, b: usize, p: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    let mut d = Array2::<f32>::zeros((g * b, p));
    for block in 0..g {
        // Gram–Schmidt b Gaussian rows in f64.
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(b);
        for _ in 0..b {
            let mut v: Vec<f64> = (0..p).map(|_| gaussian(&mut s)).collect();
            for u in basis.iter() {
                let dot: f64 = v.iter().zip(u).map(|(a, x)| a * x).sum();
                for (vc, uc) in v.iter_mut().zip(u) {
                    *vc -= dot * uc;
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0e-30);
            for vc in v.iter_mut() {
                *vc /= norm;
            }
            basis.push(v);
        }
        for (r, v) in basis.iter().enumerate() {
            for c in 0..p {
                d[[block * b + r, c]] = v[c] as f32;
            }
        }
    }
    d
}
