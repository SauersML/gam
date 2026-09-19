//! gam#2926: the standard-normal adequacy screen's bounds are the null quantiles
//! of its own statistics at the sample's effective size, so an exact N(0, 1)
//! score fails it at no more than the design rate at every `n`, and a planted
//! departure is caught once `n` resolves it.

use super::{
    AUTO_Z_NORMAL_SCREEN_ALPHA, LatentNormalAdequacy, LatentZPolicy, latent_z_normal_adequacy,
};
use crate::probability::normal_two_sided_probability;
use ndarray::Array1;

fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// `n` exact standard-normal draws (Box-Muller), mapped through `plant`.
fn scores(n: usize, seed: u64, plant: impl Fn(f64) -> f64) -> Array1<f64> {
    let mut state = seed;
    Array1::from_iter((0..n).map(|_| {
        let radius = (-2.0 * next_unit(&mut state).ln()).sqrt();
        let angle = std::f64::consts::TAU * next_unit(&mut state);
        plant(radius * angle.cos())
    }))
}

fn screen(z: &Array1<f64>) -> LatentNormalAdequacy {
    latent_z_normal_adequacy(z, &Array1::ones(z.len()), &LatentZPolicy::default())
        .expect("the screen measures a finite sample")
}

/// The number of failures out of `trials` that an exact Gaussian reaches with
/// probability below 1e-3 when each trial fails at the design rate: the pin's
/// band, stated from the rate rather than from what the draws gave.
fn failure_band(trials: u32) -> u32 {
    let alpha = AUTO_Z_NORMAL_SCREEN_ALPHA;
    let mut pmf = (1.0 - alpha).powi(trials as i32);
    let mut cdf = pmf;
    let mut k = 0;
    while 1.0 - cdf > 1e-3 {
        pmf *= (f64::from(trials - k) / f64::from(k + 1)) * (alpha / (1.0 - alpha));
        cdf += pmf;
        k += 1;
    }
    k
}

/// An exact N(0, 1) score fails the screen at no more than the design rate,
/// at n = 2000 and at n = 1e5 (40 seeds each; the measured rate is printed).
#[test]
fn an_exact_gaussian_score_fails_the_screen_at_most_at_its_design_rate_2926() {
    let trials = 40u32;
    for n in [2_000usize, 100_000] {
        let failures = (0..trials)
            .filter(|&seed| !screen(&scores(n, 0x2926_5C4E_0000 + u64::from(seed), |x| x)).passes())
            .count() as u32;
        eprintln!(
            "[2926 screen] exact N(0,1), n={n}: {failures}/{trials} fail (rate {:.3}; design {})",
            f64::from(failures) / f64::from(trials),
            AUTO_Z_NORMAL_SCREEN_ALPHA
        );
        assert!(
            failures <= failure_band(trials),
            "an exact Gaussian failed {failures}/{trials} at n={n}, above the design rate's \
             1e-3 band {}",
            failure_band(trials)
        );
    }
}

/// Every bound scales with the effective size: the moment and KS bounds shrink
/// as 1/√n and the largest-|z| bound grows, so no bound is a constant.
#[test]
fn the_screen_bounds_scale_with_the_effective_size_2926() {
    let small = screen(&scores(2_000, 0x2926_5C4E_1000, |x| x));
    let large = screen(&scores(100_000, 0x2926_5C4E_1001, |x| x));
    let root_ratio = (100_000.0_f64 / 2_000.0).sqrt();
    for (name, at_small, at_large) in [
        ("mean", small.mean_tol, large.mean_tol),
        ("skew", small.skew_tol, large.skew_tol),
        ("ks", small.ks_tol, large.ks_tol),
    ] {
        let ratio = at_small / at_large;
        assert!(
            (ratio / root_ratio - 1.0).abs() < 0.02,
            "the {name} bound must shrink as 1/sqrt(n): {at_small:e} at 2000, {at_large:e} at 1e5"
        );
    }
    assert!(large.max_abs_tol > small.max_abs_tol);
    // Kolmogorov's survival 2 Σ (−1)^{j−1} e^{−2 j² λ²} is its first term to
    // below 1e-20 here, so its level-α/8 value is sqrt(ln(16/α)/2) = 1.698,
    // over Stephens' sqrt(n) + 0.12 + 0.11/sqrt(n): 0.0379 at n = 2000 and
    // 0.0054 at n = 1e5, on either side of the former constant 0.025.
    let lambda = ((16.0 / AUTO_Z_NORMAL_SCREEN_ALPHA).ln() / 2.0).sqrt();
    for (n, adequacy) in [(2_000.0_f64, &small), (100_000.0, &large)] {
        let expected = lambda / (n.sqrt() + 0.12 + 0.11 / n.sqrt());
        assert!(
            (adequacy.ks_tol / expected - 1.0).abs() < 1e-6,
            "ks bound at n={n}: {} against {expected}",
            adequacy.ks_tol
        );
    }
    assert!(small.ks_tol > 0.025 && large.ks_tol < 0.025);
}

/// A defect planted in the tails, lighter tails beyond 1.8 with the bulk
/// untouched (KS about 0.01), fails the screen at n = 1e5, so the fit takes
/// the estimated law.
#[test]
fn a_planted_tail_defect_fails_the_screen_at_large_n_2926() {
    let compress = |x: f64| {
        if x.abs() > 1.8 {
            x.signum() * (1.8 + 0.5 * (x.abs() - 1.8))
        } else {
            x
        }
    };
    let adequacy = screen(&scores(100_000, 0x2926_5C4E_2000, compress));
    eprintln!("[2926 screen] planted tail defect, n=1e5: {}", adequacy.ledger());
    assert!(
        adequacy.ks > 0.005 && adequacy.ks < 0.02,
        "the planted defect is a KS-about-0.01 departure: {}",
        adequacy.ks
    );
    assert!(!adequacy.passes(), "{}", adequacy.ledger());
}

/// A departure every fixed bound the screen had let through at n = 1e5 (KS
/// 0.01 < 0.025, |skew| < 0.10, |excess kurtosis| < 0.25, the 4σ tail mass under
/// twice the Gaussian tail plus 1e-5, |z| < 8) fails
/// the derived screen there, on its KS distance. At n = 2000 the same departure
/// is below what the sample resolves, and its power there is printed.
#[test]
fn a_departure_the_fixed_bounds_passed_fails_the_derived_screen_2926() {
    let wiggle = |x: f64| x + 0.03 * (2.0 * x).sin();
    let adequacy = screen(&scores(100_000, 0x2926_5C4E_3000, wiggle));
    eprintln!("[2926 screen] bulk wiggle, n=1e5: {}", adequacy.ledger());
    assert!(
        adequacy.ks < 0.025
            && adequacy.skew.abs() < 0.10
            && adequacy.excess_kurtosis.abs() < 0.25
            && adequacy.tail_mass_inner <= 2.0 * normal_two_sided_probability(4.0) + 1e-5
            && adequacy.max_abs < 8.0,
        "the departure must pass every former fixed bound: {}",
        adequacy.ledger()
    );
    assert!(adequacy.ks > adequacy.ks_tol, "{}", adequacy.ledger());
    assert!(!adequacy.passes(), "{}", adequacy.ledger());

    let trials = 40u32;
    let caught = (0..trials)
        .filter(|&seed| !screen(&scores(2_000, 0x2926_5C4E_4000 + u64::from(seed), wiggle)).passes())
        .count();
    eprintln!("[2926 screen] bulk wiggle, n=2000: power {caught}/{trials}");
}
