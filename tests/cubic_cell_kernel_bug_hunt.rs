use gam::families::cubic_cell_kernel::{
    CellMomentScratch, DenestedCubicCell, affine_anchor_moment_vector, evaluate_cell_moments,
    evaluate_cell_moments_with_scratch, reduce_quartic_moments, reduce_sextic_moments,
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
    let alpha = 0.0;
    let beta = 0.0;
    let out = affine_anchor_moment_vector(alpha, beta, f64::NEG_INFINITY, f64::INFINITY, 6);
    assert!(
        (out[0] - 1.0).abs() < 1e-15,
        "Expected anchor moment M0 to be exactly 1.0 for alpha=0 beta=0 over (-inf,inf)"
    );
    assert!(
        out[1].abs() < 1e-15,
        "Expected anchor moment M1 to be exactly 0.0 for alpha=0 beta=0 over (-inf,inf)"
    );
    assert!(
        (out[2] - 1.0).abs() < 1e-15,
        "Expected anchor moment M2 to be exactly 1.0 for alpha=0 beta=0 over (-inf,inf)"
    );
    assert!(
        out[3].abs() < 1e-15,
        "Expected anchor moment M3 to be exactly 0.0 for alpha=0 beta=0 over (-inf,inf)"
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
    let right = DenestedCubicCell {
        left: boundary,
        right: 1.4,
        c0: eta_boundary,
        c1: slope_boundary,
        c2: 0.2,
        c3: -0.05,
    };
    let eps = 1e-12;
    let l = left.eta(boundary - eps);
    let r = right.eta(boundary + eps);
    assert!(
        (l - r).abs() < 1e-15,
        "Expected cubic-cell evaluation to be continuous at shared boundary from both neighboring cells"
    );
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
    let _second = evaluate_cell_moments_with_scratch(cell_b, 3, &mut scratch)
        .expect("second scratch evaluation should succeed");
    let fresh = evaluate_cell_moments(cell_a, 14).expect("fresh moment evaluation should succeed");
    for i in 0..snap.len() {
        assert!(
            (snap[i] - fresh.moments[i]).abs() < 1e-18,
            "Expected scratch resize reuse to preserve earlier written values exactly after changing output length; mismatch at index {i}"
        );
    }
    assert!(
        (snap[0] - fresh.moments[0]).abs() < 0.0,
        "Expected scratch-backed moment storage to remain untouched after a second call with a different requested length"
    );
}
