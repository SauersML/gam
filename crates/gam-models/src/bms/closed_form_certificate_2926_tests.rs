//! gam#2926: the closed-form certificate keeps the closed form unless the rows'
//! residual energy under the estimated law exceeds its exact null law's upper
//! `CLOSED_FORM_CERTIFICATE_ALPHA` quantile, the law it has when the score is
//! exactly Gaussian and the residuals are the estimated law's sampling error.

use super::estimated_latent_law::build_empirical_law_on_own_axis;
use super::{
    AnchorNoiseGram, CLOSED_FORM_CERTIFICATE_ALPHA, ClosedFormAnchorResidual,
    DEFAULT_EMPIRICAL_LATENT_GRID_SIZE, EmpiricalZGrid, closed_form_kept_by_null_tail,
};
use crate::probability::{
    TailProbability, WeightedChiSquareTerm, normal_cdf, signed_weighted_chi_square_sf,
};
use ndarray::Array1;

fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn gaussian(state: &mut u64) -> f64 {
    let radius = (-2.0 * next_unit(state).ln()).sqrt();
    radius * (std::f64::consts::TAU * next_unit(state)).cos()
}

/// The Bernoulli closed form's certificate at `n` rows whose scores are drawn
/// through `plant`, with marginal index `q = 0.6·x` over an independent covariate
/// `x` and slope `b`: each row's residual `Σ_m w_m Φ(q√(1+b²) + b u_m) − Φ(q)` under
/// the law estimated from the scores, as the fit's certificate pass reads it.
fn certificate(
    n: usize,
    seed: u64,
    slope: f64,
    plant: impl Fn(f64) -> f64,
) -> ClosedFormAnchorResidual {
    let mut state = seed;
    let z = Array1::from_iter((0..n).map(|_| plant(gaussian(&mut state))));
    let weights = Array1::ones(n);
    let law = build_empirical_law_on_own_axis(
        z.view(),
        weights.view(),
        DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
        "closed-form certificate pin",
    )
    .expect("an estimated law");
    certificate_on(&law, n, seed ^ 0xC0FF_EE00, slope)
}

fn certificate_on(
    law: &EmpiricalZGrid,
    n: usize,
    seed: u64,
    slope: f64,
) -> ClosedFormAnchorResidual {
    let mut state = seed;
    let mut noise = AnchorNoiseGram::new(law.nodes.len());
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let q = 0.6 * gaussian(&mut state);
        let pi = normal_cdf(q);
        let alpha = q * (1.0 + slope * slope).sqrt();
        let probabilities: Vec<f64> = law
            .nodes
            .iter()
            .map(|&u| normal_cdf(alpha + slope * u))
            .collect();
        let mean: f64 = law.weights.iter().zip(&probabilities).map(|(w, p)| w * p).sum();
        let variance: f64 = law
            .weights
            .iter()
            .zip(&probabilities)
            .map(|(w, p)| w * (p - mean) * (p - mean))
            .sum();
        let scale = pi * (1.0 - pi);
        noise
            .add_anchor(1.0 / scale, &law.weights, &probabilities, n as f64)
            .expect("an anchor on the law's atoms");
        rows.push((mean - pi, variance.sqrt() / (n as f64).sqrt(), scale, 1.0));
    }
    ClosedFormAnchorResidual::from_rows(&rows, &noise, law.nodes.len(), n as f64)
        .expect("a measurable certificate")
}

/// The failures out of `trials` that an exact Gaussian reaches with probability
/// below 1e-3 when each trial fires at the design rate.
fn fire_band(trials: u32) -> u32 {
    let alpha = CLOSED_FORM_CERTIFICATE_ALPHA;
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

/// On an exactly Gaussian score the certificate prefers the estimated law at no
/// more than its design rate, at n = 2000 and n = 1e5 (40 seeds each; the measured
/// rate and the former sign rule's rate on the same draws are printed).
#[test]
fn an_exact_gaussian_score_fires_the_certificate_at_its_design_rate_2926() {
    let trials = 40u32;
    for n in [2_000usize, 100_000] {
        let (mut fires, mut sign_fires) = (0u32, 0u32);
        for seed in 0..trials {
            let certificate = certificate(n, 0x2926_CE57_0000 + u64::from(seed), 1.0, |x| x);
            fires += u32::from(!certificate.closed_form_chosen);
            sign_fires += u32::from(certificate.excess_kl > 0.0);
        }
        eprintln!(
            "[2926 certificate] exact N(0,1), n={n}: fires {fires}/{trials} (design {}); the sign \
             of D̂ fired {sign_fires}/{trials}",
            CLOSED_FORM_CERTIFICATE_ALPHA
        );
        assert!(
            fires <= fire_band(trials),
            "the certificate fired {fires}/{trials} on exact Gaussian scores at n={n}, above the \
             design rate's 1e-3 band {}",
            fire_band(trials)
        );
    }
}

/// The certificate's power on two planted departures, derived before it is pinned.
///
/// Each anchor's bias is linear in the law: `β_i = ∫ f_i d(G − N)` with
/// `f_i(u) = Φ(α_i + b·u)`, so any departure enters it at first order. Split `f_i`
/// into its even and odd parts. A symmetric departure integrates only the even part,
/// `½[Φ(α + bu) + Φ(α − bu)] − Φ(α) = −½ b²u² α φ(α) + O(u⁴)`, so its bias is
/// `≈ −½ b² α_i φ(α_i) (Var_G − 1)` plus a fourth-cumulant term: zero at `α_i = 0`
/// and of opposite sign on the two sides of it. The law is kept on the score's own
/// axis, so the lighter tails' variance change (0.93) is not standardized out. An
/// asymmetric departure also integrates the odd part, at first order `b φ(α_i) E_G[u]`,
/// one sign for every anchor.
/// The noise `ε = (E_Ĝ − E_G)[f]` is dominated by the location mode (about 92% of
/// `Σλ`), and the power follows from the noncentral weighted chi-square
/// `T = Σ_k λ_k (Z_k + ν_k)² + ‖C^{1/2}β_⊥‖²` over the noise Gram's eigenbasis
/// (the population model of this pin, evaluated in `power2926.py` and
/// `power2926b.py` on MSI, gam#2926).
/// It reproduces the pin's null exactly: fire rate 0.050 at both sizes, and noise
/// energy 0.3076 against the recorded 0.3081. Its predictions:
/// - the upper tail stretched above +1σ (mean +0.042, variance 1.176, skewness
///   +0.374, excess kurtosis +0.653) enters mostly through its variance, the even
///   part (the bias's correlation with `q` is −0.94). Its bias energy
///   `Σ c β² ≈ 20` is against a null quantile of 1.21 at n = 1e5, so power 1 with a
///   miss probability of at most 2.5e-94 (Chernoff); power 0.167 at n = 2000;
/// - lighter tails beyond 1.8σ (symmetric; variance 0.933, excess kurtosis −0.454)
///   enter through the even part alone (correlation +0.996). Bias energy ≈ 0.98
///   against 1.12 at n = 1e5 gives power 0.571, and 0.053 at n = 2000. The
///   recorded run's `T = 0.819` sits at the 10.8th percentile of that alternative,
///   so its miss (null tail 0.095) is the predicted outcome, not a defect.
///
/// The moments are Gauss-Hermite integrals of each map under N(0, 1) (160 nodes);
/// the energies, quantiles and correlations come from two draws of the anchors
/// (20.5 and 20.2, 0.977 and 0.995), which is the spread the ≈ carries.
///
/// So only the stretch at n = 1e5 is asserted. The two powers at n = 2000 and the
/// lighter tails' tail at n = 1e5 are printed.
#[test]
fn a_planted_first_order_defect_fires_the_certificate_2926() {
    let upper_stretch = |x: f64| if x > 1.0 { 1.0 + 1.5 * (x - 1.0) } else { x };
    let large = certificate(100_000, 0x2926_CE57_1000, 1.0, upper_stretch);
    eprintln!("[2926 certificate] upper-tail stretch, n=1e5: {}", large.summary());
    assert!(
        !large.closed_form_chosen
            && large
                .null_p_value
                .is_some_and(|p| p < CLOSED_FORM_CERTIFICATE_ALPHA),
        "{}",
        large.summary()
    );

    let trials = 40u32;
    let fires = (0..trials)
        .filter(|&seed| {
            !certificate(2_000, 0x2926_CE57_2000 + u64::from(seed), 1.0, upper_stretch)
                .closed_form_chosen
        })
        .count();
    eprintln!("[2926 certificate] upper-tail stretch, n=2000: power {fires}/{trials}");

    let lighter_tails = |x: f64| {
        if x.abs() > 1.8 {
            x.signum() * (1.8 + 0.5 * (x.abs() - 1.8))
        } else {
            x
        }
    };
    let second_order = certificate(100_000, 0x2926_CE57_1000, 1.0, lighter_tails);
    eprintln!(
        "[2926 certificate] lighter tails (second order), n=1e5: {}",
        second_order.summary()
    );
}

/// `(noise energy, null standard deviation √(2 Σλ²))` of `n` anchors whose
/// values at the law's atoms are `values(anchor, atom)`, read on `law`.
fn null_law_on(law: &EmpiricalZGrid, n: usize, values: impl Fn(usize, f64) -> f64) -> (f64, f64) {
    let mut noise = AnchorNoiseGram::new(law.nodes.len());
    let mut rows = Vec::with_capacity(n);
    for anchor in 0..n {
        let probabilities: Vec<f64> = law.nodes.iter().map(|&u| values(anchor, u)).collect();
        let mean: f64 = law.weights.iter().zip(&probabilities).map(|(w, p)| w * p).sum();
        let variance: f64 = law
            .weights
            .iter()
            .zip(&probabilities)
            .map(|(w, p)| w * (p - mean) * (p - mean))
            .sum();
        noise
            .add_anchor(1.0, &law.weights, &probabilities, n as f64)
            .expect("an anchor on the law's atoms");
        rows.push((0.0, variance.sqrt() / (n as f64).sqrt(), 1.0, 1.0));
    }
    let certificate = ClosedFormAnchorResidual::from_rows(&rows, &noise, law.nodes.len(), n as f64)
        .expect("a measurable certificate");
    let modes = certificate.null_modes.expect("a recorded null law");
    (
        certificate.noise_energy,
        (2.0 * certificate.noise_energy * certificate.noise_energy / modes).sqrt(),
    )
}

/// The certificate's null law is read on the 65-node compression the estimated law
/// is, not on the raw sample. The compressor keeps the sample's mean and variance
/// exactly (it rescales its bin means to them), so for anchors linear in the score
/// the null law on the compression is the raw sample's to rounding. For the probit
/// anchors the certificate reads it states no bound, so their compressed-to-raw
/// ratios at n = 500 are printed rather than asserted against a chosen tolerance
/// (gam#2926: recorded as a finding).
#[test]
fn the_compressed_law_keeps_the_null_law_of_a_linear_anchor_2926() {
    let n = 500;
    let mut state = 0x2926_CE57_3000_u64;
    let z = Array1::from_iter((0..n).map(|_| gaussian(&mut state)));
    let weights = Array1::ones(n);
    let compressed = build_empirical_law_on_own_axis(
        z.view(),
        weights.view(),
        DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
        "closed-form certificate pin",
    )
    .expect("an estimated law");
    let mut raw_nodes = z.to_vec();
    raw_nodes.sort_by(f64::total_cmp);
    let raw = EmpiricalZGrid::new(raw_nodes, vec![1.0 / n as f64; n], "raw sample")
        .expect("the raw sample as a law");
    // Linear anchors, `c_i·u` with distinct scales: one shared mode.
    let linear = |anchor: usize, u: f64| (1.0 + 0.01 * anchor as f64) * u;
    let (compressed_mean, compressed_sd) = null_law_on(&compressed, n, linear);
    let (raw_mean, raw_sd) = null_law_on(&raw, n, linear);
    let rounding = 16.0 * n as f64 * f64::EPSILON;
    assert!(
        (compressed_mean / raw_mean - 1.0).abs() <= rounding
            && (compressed_sd / raw_sd - 1.0).abs() <= rounding,
        "a linear anchor's null law must be the raw sample's on the compression: mean {} against \
         {}, sd {} against {}",
        compressed_mean,
        raw_mean,
        compressed_sd,
        raw_sd
    );
    // The probit anchors the certificate reads, at the pins' q = 0.6·x and b = 1.
    let mut anchor_state = 0x2926_CE57_3001_u64;
    let q: Vec<f64> = (0..n).map(|_| 0.6 * gaussian(&mut anchor_state)).collect();
    let probit = |anchor: usize, u: f64| normal_cdf(q[anchor] * 2.0_f64.sqrt() + u);
    let (compressed_mean, compressed_sd) = null_law_on(&compressed, n, probit);
    let (raw_mean, raw_sd) = null_law_on(&raw, n, probit);
    eprintln!(
        "[2926 certificate] probit anchors at n={n}: compressed/raw null mean {:.4}, null sd {:.4}",
        compressed_mean / raw_mean,
        compressed_sd / raw_sd
    );
}

/// A certificate saved before the null law was recorded loads with none of its
/// fields and its decision as made, and a current one round-trips.
#[test]
fn a_certificate_saved_before_the_null_law_loads_without_it_2926() {
    let current = certificate(2_000, 0x2926_CE57_4000, 1.0, |x| x);
    let text = serde_json::to_string(&current).expect("serialize");
    let reloaded: ClosedFormAnchorResidual = serde_json::from_str(&text).expect("round-trip");
    assert_eq!(reloaded, current);
    let mut older: serde_json::Value = serde_json::to_value(&current).expect("serialize");
    let object = older.as_object_mut().expect("a certificate is an object");
    object.remove("null_p_value").expect("the tail is written");
    object
        .remove("null_p_value_relative_error")
        .expect("the tail's bound is written");
    object.remove("null_modes").expect("the modes are written");
    let loaded: ClosedFormAnchorResidual = serde_json::from_value(older).expect("an older record");
    assert!(
        loaded.null_p_value.is_none()
            && loaded.null_p_value_relative_error.is_none()
            && loaded.null_modes.is_none()
    );
    assert_eq!(loaded.closed_form_chosen, current.closed_form_chosen);
}

/// The certificate decides only where the null tail's bound does. On `Q = χ²_1`
/// (mean 1, variance 2) a tail of 0.157 at `t = 2` keeps the closed form and one of
/// 0.0143 at `t = 6` fires. At `t = 2000` the tail is below the subnormal range, so
/// its own bound resolves nothing, and Cantelli's `2/(2 + 1999²)` fires it. A bound
/// that straddles the rate, an unresolved tail that Cantelli cannot place below it,
/// and a tail that is not a number decide nothing.
#[test]
fn the_null_tail_decides_the_certificate_only_where_its_bound_does_2926() {
    let alpha = CLOSED_FORM_CERTIFICATE_ALPHA;
    let one_mode = [WeightedChiSquareTerm {
        weight: 1.0,
        degrees_of_freedom: 1.0,
    }];
    let decide = |t: f64| {
        closed_form_kept_by_null_tail(signed_weighted_chi_square_sf(&one_mode, t), t, 1.0, 2.0)
    };
    assert_eq!(decide(2.0), Some(true), "P(χ²_1 > 2) = 0.157 is above the rate");
    assert_eq!(decide(6.0), Some(false), "P(χ²_1 > 6) = 0.0143 is below the rate");
    let deep = signed_weighted_chi_square_sf(&one_mode, 2000.0);
    assert!(
        deep.relative_error >= 1.0,
        "the premise: P(χ²_1 > 2000) is below the subnormal range and unresolved, got {deep:?}"
    );
    assert_eq!(
        decide(2000.0),
        Some(false),
        "Cantelli places the unresolved deep tail below the rate"
    );
    let straddle = TailProbability {
        probability: alpha,
        relative_error: 0.1,
    };
    assert_eq!(closed_form_kept_by_null_tail(straddle, 3.8, 1.0, 2.0), None);
    let unresolved = TailProbability {
        probability: 0.0,
        relative_error: 1.0,
    };
    assert_eq!(
        closed_form_kept_by_null_tail(unresolved, 1.0 + 2.0_f64.sqrt(), 1.0, 2.0),
        None,
        "Cantelli's bound one sd above the mean is 1/2, which places nothing"
    );
    let invalid = TailProbability {
        probability: f64::NAN,
        relative_error: f64::NAN,
    };
    assert_eq!(closed_form_kept_by_null_tail(invalid, 2000.0, 1.0, 2.0), None);
}
