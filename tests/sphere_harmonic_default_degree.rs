//! Regression guard for `default_spherical_harmonic_degree`.
//!
//! Before fix: the function clamped `target = sqrt(n).clamp(3,12)` and then
//! looked for the smallest L with `L(L+2) >= target`. Since target was always
//! ≤ 12, the loop always returned L=3 (15 cols) regardless of n. The doc
//! claimed "Targets ~k = 50 columns" but the math never reached more than
//! ~15.
//!
//! After fix: target is a column count, scaling as min(n/4, 50) with floor 3.
//! For modest-to-large n we hit L=6 or 7 (~48–63 cols), matching mgcv's
//! `bs="sos"` default of k=50.

use gam::basis::default_spherical_harmonic_degree;

#[test]
fn default_degree_small_n_does_not_overfit() {
    // n=8: target ≈ 2 → floored at 3 → L=1 with 1*3=3>=3 → returns max(1,2)=2.
    let l = default_spherical_harmonic_degree(8);
    assert!(l >= 2 && l <= 3, "n=8 expected L in {{2,3}}, got {l}");
    let cols = l * (l + 2);
    assert!(
        cols <= 16,
        "n=8 default L={l} ({cols} cols) is too generous for the data",
    );
}

#[test]
fn default_degree_medium_n_targets_around_50_cols() {
    // n=200..1000: target should grow to 50 cols → L=6 (48 cols) or L=7 (63).
    for n in [200usize, 288, 500, 1000] {
        let l = default_spherical_harmonic_degree(n);
        let cols = l * (l + 2);
        assert!(
            cols >= 24 && cols <= 80,
            "n={n} expected ~24–80 default cols (mgcv ~50), got L={l} → {cols} cols",
        );
    }
}

#[test]
fn default_degree_large_n_caps_at_l12() {
    // n=100k+: still capped at L=12 → 168 cols, never exceeded.
    for n in [10_000usize, 100_000, 1_000_000] {
        let l = default_spherical_harmonic_degree(n);
        assert!(l <= 12, "n={n} default L={l} exceeded L=12 cap");
        assert!(l >= 6, "n={n} default L={l} too small for biobank-scale");
    }
}

#[test]
fn default_degree_monotonic_in_n() {
    let mut prev_l = 0usize;
    for &n in &[10usize, 50, 100, 500, 2_000, 50_000, 1_000_000] {
        let l = default_spherical_harmonic_degree(n);
        assert!(
            l >= prev_l,
            "default L should be non-decreasing in n; n={n} gave L={l} but prior L={prev_l}",
        );
        prev_l = l;
    }
}
