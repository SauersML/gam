use gam::families::cubic_cell_kernel::{
    CellMomentScratch, DenestedCubicCell, LocalSpanCubic, affine_anchor_moment_vector,
    build_denested_partition_cells, evaluate_cell_moments, evaluate_cell_moments_with_scratch,
    global_cubic_from_local, reduce_quartic_moments, reduce_sextic_moments,
    reset_tail_cell_moment_cache, set_tail_cell_moment_cache_enabled, tail_cell_moment_cache_stats,
};
use std::sync::{Arc, Barrier};

fn gauss_legendre_integral(cell: DenestedCubicCell, degree: usize) -> f64 {
    let n = 12000usize;
    let left = cell.left;
    let right = cell.right;
    let width = right - left;
    let h = width / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let z = left + i as f64 * h;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += w * z.powi(degree as i32) * (-cell.q(z)).exp();
    }
    sum * h / 3.0
}

#[test]
fn bug_cell_moment_recurrence_quartic_disagrees_with_direct_quadrature() {
    let cell = DenestedCubicCell {
        left: -0.9,
        right: 1.1,
        c0: 0.2,
        c1: 1.3,
        c2: -0.35,
        c3: 0.0,
    };
    let base = [0usize, 1, 2].map(|k| gauss_legendre_integral(cell, k));
    let reduced = reduce_quartic_moments(cell, base, 9).expect("quartic reduction should succeed");
    for k in 0..=9 {
        let direct = gauss_legendre_integral(cell, k);
        let err = (reduced[k] - direct).abs();
        assert!(
            err < 1e-12,
            "Expected quartic recurrence to match direct quadrature to 1e-12 for moment k={k}, but absolute error was {err:e}"
        );
    }
}

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
    let l = left.eta(boundary);
    let r = right.eta(boundary);
    assert!(
        (l - r).abs() < 1e-12,
        "Expected cubic-cell evaluation to be continuous at shared boundary from both neighboring cells: left={l}, right={r}, gap={}",
        (l - r).abs()
    );
}

#[test]
fn bug_cubic_cell_partition_is_c0_and_c1_continuous_across_every_boundary() {
    // Production-path continuity (#1837, second angle). The synthetic
    // `bug_cubic_cell_boundary_value_is_discontinuous_between_neighbors` builds a
    // single hand-made cell pair; this test instead drives the REAL partition
    // builder `build_denested_partition_cells` and checks continuity across
    // every interior boundary it emits — including the two qualitatively
    // distinct boundary kinds: a fixed score break and a link-knot crossing
    // where the active span flips from affine (tail) to genuinely cubic.
    //
    // The denested function is `η(z) = a + b·z + b·S(z) + L(a + b·z)`. With
    // `a = 0, b = 1` this is `η(z) = z + S(z) + L(z)`. We make `S ≡ 0` and put
    // all curvature in `L`, a C¹ piecewise cubic:
    //   L(u) = -1 + 3u             for u < 0        (affine),
    //   L(u) = (u - 1)³            for 0 ≤ u ≤ 2    (cubic),
    //   L(u) =  1 + 3(u - 2)       for u > 2        (affine).
    // L is C¹ at u = 0 and u = 2 (value and slope agree: L(0)=-1, L'(0)=3;
    // L(2)=1, L'(2)=3), and affine in both tails so the outer cells satisfy the
    // builder's affine-tail contract. Because every span closure returns the
    // EXACT Taylor expansion of the active branch about the query point,
    // `global_cubic_from_local` reconstructs the exact global cubic per cell, so
    // adjacent cells that share a C¹ boundary of `L` must agree in both value
    // (C0) and slope (C1) there. A kernel that mis-composed the global cubic
    // (the discontinuity this issue feared) would break these agreements.
    let a = 0.0_f64;
    let b = 1.0_f64;
    let score_breaks = [1.0_f64];
    let link_breaks = [0.0_f64, 2.0_f64];

    // S ≡ 0: a flat score span whose global cubic is identically zero.
    let score_span_at = |x: f64| {
        Ok(LocalSpanCubic {
            left: x,
            right: x + 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        })
    };
    // L as above, returned as its exact Taylor expansion about the query point
    // `x` (so `global_cubic_from_local` recovers the branch's global cubic).
    let link_span_at = |x: f64| {
        let span = if x < 0.0 {
            LocalSpanCubic {
                left: x,
                right: x + 1.0,
                c0: -1.0 + 3.0 * x,
                c1: 3.0,
                c2: 0.0,
                c3: 0.0,
            }
        } else if x <= 2.0 {
            // L(u) = (u-1)^3  ->  Taylor about x:
            //   L(x) = (x-1)^3, L'(x) = 3(x-1)^2, L''/2 = 3(x-1), L'''/6 = 1.
            LocalSpanCubic {
                left: x,
                right: x + 1.0,
                c0: (x - 1.0).powi(3),
                c1: 3.0 * (x - 1.0).powi(2),
                c2: 3.0 * (x - 1.0),
                c3: 1.0,
            }
        } else {
            LocalSpanCubic {
                left: x,
                right: x + 1.0,
                c0: 1.0 + 3.0 * (x - 2.0),
                c1: 3.0,
                c2: 0.0,
                c3: 0.0,
            }
        };
        Ok(span)
    };

    let cells = build_denested_partition_cells(
        a,
        b,
        &score_breaks,
        &link_breaks,
        score_span_at,
        link_span_at,
    )
    .expect("partition builder should succeed for a C1 piecewise-cubic link");

    // Expect the four cells (-inf,0], [0,1], [1,2], [2,+inf) — a genuinely cubic
    // interior flanked by affine tails, with three shared boundaries.
    assert_eq!(
        cells.len(),
        4,
        "expected 4 partition cells (two tails + two interior); got {}",
        cells.len()
    );
    // At least one interior cell must be truly cubic, else the test would not
    // exercise the non-affine boundary composition it is meant to guard.
    assert!(
        cells
            .iter()
            .any(|pc| pc.cell.c2.abs() > 1e-6 || pc.cell.c3.abs() > 1e-6),
        "no cubic interior cell was produced; the continuity check would be vacuous"
    );

    let cell_slope = |c: &DenestedCubicCell, z: f64| c.c1 + 2.0 * c.c2 * z + 3.0 * c.c3 * z * z;

    for window in cells.windows(2) {
        let left = &window[0].cell;
        let right = &window[1].cell;
        let boundary = left.right;
        // Sanity: the partition is a contiguous cover — the right cell starts
        // exactly where the left one ends.
        assert_eq!(
            boundary, right.left,
            "partition is not contiguous: left.right={} != right.left={}",
            boundary, right.left
        );
        if !boundary.is_finite() {
            continue;
        }
        // C0: shared boundary value agrees from both neighbors.
        let lv = left.eta(boundary);
        let rv = right.eta(boundary);
        assert!(
            (lv - rv).abs() < 1e-10,
            "C0 discontinuity at z={boundary}: left.eta={lv}, right.eta={rv}, gap={}",
            (lv - rv).abs()
        );
        // C1: shared boundary slope agrees too (the underlying L is C1 there).
        let ls = cell_slope(left, boundary);
        let rs = cell_slope(right, boundary);
        assert!(
            (ls - rs).abs() < 1e-9,
            "C1 discontinuity at z={boundary}: left slope={ls}, right slope={rs}, gap={}",
            (ls - rs).abs()
        );
    }
}

#[test]
fn bug_sextic_moment_reduction_disagrees_with_direct_quadrature() {
    let cell = DenestedCubicCell {
        left: -0.7,
        right: 1.0,
        c0: 0.5,
        c1: 1.1,
        c2: -0.3,
        c3: 0.08,
    };
    let base = [0usize, 1, 2, 3, 4].map(|k| gauss_legendre_integral(cell, k));
    let reduced = reduce_sextic_moments(cell, base, 11).expect("sextic reduction should succeed");
    for k in 0..=11 {
        let direct = gauss_legendre_integral(cell, k);
        let err = (reduced[k] - direct).abs();
        assert!(
            err < 1e-12,
            "Expected sextic recurrence to match direct quadrature to 1e-12 for moment k={k}, but absolute error was {err:e}"
        );
    }
}

#[test]
fn bug_moment_boundary_term_sign_flips_in_quartic_recurrence() {
    let cell = DenestedCubicCell {
        left: -0.8,
        right: 0.9,
        c0: -0.1,
        c1: 1.2,
        c2: 0.25,
        c3: 0.0,
    };
    let base = [0usize, 1, 2].map(|k| gauss_legendre_integral(cell, k));
    let reduced = reduce_quartic_moments(cell, base, 7).expect("quartic reduction should succeed");
    let direct_m7 = gauss_legendre_integral(cell, 7);
    assert!(
        (reduced[7] - direct_m7).abs() < 1e-13,
        "Expected integration-by-parts boundary term sign convention to agree with direct quadrature for quartic M7"
    );
}

#[test]
fn bug_cell_moment_scratch_resize_corrupts_existing_entries() {
    let cell_a = DenestedCubicCell {
        left: -0.6,
        right: 0.7,
        c0: 0.1,
        c1: 1.0,
        c2: 0.2,
        c3: -0.04,
    };
    let cell_b = DenestedCubicCell {
        left: -1.1,
        right: -0.2,
        c0: -0.3,
        c1: 0.7,
        c2: 0.0,
        c3: 0.0,
    };
    let mut scratch = CellMomentScratch::with_capacity(2);
    let snap = {
        let first = evaluate_cell_moments_with_scratch(cell_a, 14, &mut scratch)
            .expect("first scratch evaluation should succeed");
        first.moments.to_vec()
    };
    evaluate_cell_moments_with_scratch(cell_b, 3, &mut scratch)
        .expect("second scratch evaluation should succeed");
    let fresh = evaluate_cell_moments(cell_a, 14).expect("fresh moment evaluation should succeed");
    for i in 0..snap.len() {
        assert!(
            (snap[i] - fresh.moments[i]).abs() < 1e-18,
            "Expected scratch resize reuse to preserve earlier written values exactly after changing output length; mismatch at index {i}"
        );
    }
    assert_eq!(
        snap[0].to_bits(),
        fresh.moments[0].to_bits(),
        "Expected scratch-backed moment storage to remain untouched after a second call with a different requested length"
    );
}
