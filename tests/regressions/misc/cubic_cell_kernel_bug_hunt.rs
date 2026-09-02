use gam::families::cubic_cell_kernel::{DenestedCubicCell, LocalSpanCubic, affine_anchor_moment_vector, evaluate_cell_moments, global_cubic_from_local, reset_tail_cell_moment_cache, set_tail_cell_moment_cache_enabled, tail_cell_moment_cache_stats};
use std::sync::{Arc, Barrier};

#[test]
fn bug_tail_cell_cache_second_thread_waits_for_first_computation() {
    set_tail_cell_moment_cache_enabled(true);
    reset_tail_cell_moment_cache();
    let cell = DenestedCubicCell {
        left: f64::NEG_INFINITY,
        right: -2.0,
        c0: 0.4,
        c1: 0.8,
        c2: 0.0,
        c3: 0.0,
    };
    let barrier = Arc::new(Barrier::new(3));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let b = barrier.clone();
        handles.push(std::thread::spawn(move || {
            b.wait();
            evaluate_cell_moments(cell, 48).expect("tail moments should evaluate")
        }));
    }
    barrier.wait();
    let a = handles.remove(0).join().expect("thread 1 joins");
    let b = handles.remove(0).join().expect("thread 2 joins");
    assert_eq!(
        a.value.to_bits(),
        b.value.to_bits(),
        "Expected concurrent evaluations for the same tail-cell key to return bit-identical value results"
    );
    assert_eq!(
        a.moments.len(),
        b.moments.len(),
        "Expected concurrent evaluations for the same tail-cell key to return identical moment lengths"
    );
    for i in 0..a.moments.len() {
        assert_eq!(
            a.moments[i].to_bits(),
            b.moments[i].to_bits(),
            "Expected concurrent evaluations for the same tail-cell key to return bit-identical moments at index {i}"
        );
    }
    let stats = tail_cell_moment_cache_stats();
    assert!(
        stats.hits >= 2,
        "Expected the second concurrent caller to reuse an already-computed tail-cell cache entry without serialization stalls"
    );
}

#[test]
fn bug_affine_anchor_identity_moments_are_not_preserved() {
    // `affine_anchor_moment_vector` returns the RAW substrate moments
    // `T_n = ∫ z^n exp(-½z²) dz`, NOT a normalized density: this is the
    // `∫ z^n exp(-q) dz` convention the cubic-cell substrate, every production
    // consumer (`evaluate_affine_cell_state` / `_derivative_state`,
    // transformation-normal, BMS), and the CPU/GPU parity reference all share
    // (the `1/√(2π)` is folded in downstream via `INV_TWO_PI`). The identity
    // invariant this guards is that at `alpha=beta=0` over the whole line the
    // anchor reduces to the *standard normal* — whose raw moments are the
    // normalized `{1, 0, 1, 0}` scaled by the whole-line mass `√(2π)`, i.e.
    // M0 = M2 = √(2π) and M1 = M3 = 0. Asserting a normalized `M0 = 1` here is
    // mis-specified: it contradicts the whole-line mass `∫ exp(-½z²) dz = √(2π)`
    // that the #352 both-tails / deep-tail precision guards pin, and no
    // consumer wants the wrapper normalized.
    let alpha = 0.0;
    let beta = 0.0;
    let out = affine_anchor_moment_vector(alpha, beta, f64::NEG_INFINITY, f64::INFINITY, 6);
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    assert!(
        (out[0] - sqrt_2pi).abs() < 1e-13,
        "Expected raw anchor moment M0 = √(2π) ≈ {sqrt_2pi:.6} for alpha=0 beta=0 over (-inf,inf); got {}",
        out[0]
    );
    assert!(
        out[1].abs() < 1e-13,
        "Expected anchor moment M1 to be 0.0 for alpha=0 beta=0 over (-inf,inf); got {}",
        out[1]
    );
    assert!(
        (out[2] - sqrt_2pi).abs() < 1e-13,
        "Expected raw anchor moment M2 = √(2π) ≈ {sqrt_2pi:.6} for alpha=0 beta=0 over (-inf,inf); got {}",
        out[2]
    );
    assert!(
        out[3].abs() < 1e-13,
        "Expected anchor moment M3 to be 0.0 for alpha=0 beta=0 over (-inf,inf); got {}",
        out[3]
    );
}

#[test]
fn bug_cubic_cell_boundary_value_is_discontinuous_between_neighbors() {
    let left = DenestedCubicCell {
        left: -1.0,
        right: 0.5,
        c0: -0.2,
        c1: 0.9,
        c2: -0.1,
        c3: 0.03,
    };
    let boundary = left.right;
    let eta_boundary = left.eta(boundary);
    let slope_boundary = left.c1 + 2.0 * left.c2 * boundary + 3.0 * left.c3 * boundary * boundary;
    // The right cell is specified with a cell-LOCAL Taylor parameterization
    // anchored at the shared boundary: its local `c0`/`c1` are the left cell's
    // value and slope there, plus a chosen local curvature. But
    // `DenestedCubicCell::eta` evaluates its coefficients as a polynomial in
    // GLOBAL `z` — exactly the convention the production kernel uses when it
    // builds a cell's coefficients via `global_cubic_from_local` /
    // `denested_cell_coefficients`. So the local coefficients must be converted
    // into the global cubic basis before they are stored in the cell; assigning
    // the local Taylor coefficients directly (the original mis-specification)
    // would make the cell represent a DIFFERENT polynomial than the intended
    // local expansion, manufacturing a spurious boundary discontinuity that the
    // production kernel — which shares one global η(z) across neighbors — never
    // exhibits. Converting through the kernel's own path makes this a genuine
    // C0-continuity check.
    let right_local = LocalSpanCubic {
        left: boundary,
        right: 1.4,
        c0: eta_boundary,
        c1: slope_boundary,
        c2: 0.2,
        c3: -0.05,
    };
    let (rc0, rc1, rc2, rc3) = global_cubic_from_local(right_local);
    let right = DenestedCubicCell {
        left: boundary,
        right: 1.4,
        c0: rc0,
        c1: rc1,
        c2: rc2,
        c3: rc3,
    };
    // C0 continuity is a statement about the shared boundary point itself:
    // both neighboring cells must agree in value there. Evaluate each cell AT
    // the boundary (not at boundary ± eps, which would inject a spurious
    // O(eps · slope) gap unrelated to continuity and defeat any tight bound).
    //
    // #2073: the tolerance was previously relaxed to 1e-12 on a vague appeal to
    // the local→global round-trip. But this hand-made pair is well conditioned
    // (boundary = 0.5, O(1) coefficients, only +/-/* — no transcendentals and no
    // FMA contraction under default Rust codegen), so `global_cubic_from_local`
    // reconstructs the boundary value to a *deterministic* 2.78e-17 gap (~0.5 ULP
    // of ~0.23). 1e-14 keeps ~300x margin over that measured worst case while
    // being 2 orders tighter — and any real mis-composed global cubic would open
    // an O(0.1) gap, so this stays trivially discriminating.
    let l = left.eta(boundary);
    let r = right.eta(boundary);
    assert!(
        (l - r).abs() < 1e-14,
        "Expected cubic-cell evaluation to be continuous at shared boundary from both neighboring cells: left={l}, right={r}, gap={}",
        (l - r).abs()
    );
}

