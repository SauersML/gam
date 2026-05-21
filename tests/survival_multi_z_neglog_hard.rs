//! Heavy correctness battery for
//! `gam::survival_marginal_slope::survival_marginal_slope_vector_neglog`.
//!
//! All expected values are recomputed from primitive building blocks
//! (`survival_marginal_slope_vector_scale`, `quadratic_form`,
//! `normal_cdf`) so the test file is self-contained.

use gam::bernoulli_marginal_slope::MarginalSlopeCovariance;
use gam::probability::normal_cdf;
use gam::survival_marginal_slope::{
    survival_marginal_slope_vector_neglog, survival_marginal_slope_vector_scale,
};
use ndarray::{Array1, Array2};

// -------- inline splitmix64 PRNG ----------------------------------------

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1), never 0 or 1.
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits
        (bits as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_unit()
    }
}

// -------- closed-form neglog from primitives ----------------------------

/// Reference computation mirroring the algebra documented inside
/// `survival_marginal_slope_vector_neglog` (see
/// tests/survival_multi_z_marginal_slope.rs:66-106 for the pattern).
fn neglog_reference(
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    covariance: &MarginalSlopeCovariance,
    weight: f64,
    event: f64,
    probit_scale: f64,
) -> f64 {
    let observed: Vec<f64> = slopes.iter().map(|&g| probit_scale * g).collect();
    let var = covariance.quadratic_form(&observed).expect("quadratic form");
    let c = (1.0 + var).sqrt();

    // Cross-check vs. library scale.
    let c_lib = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale)
        .expect("library scale");
    assert!(
        (c - c_lib).abs() <= 1e-14 * (1.0 + c.abs()),
        "scale mismatch: ref={c:.17e} lib={c_lib:.17e}"
    );

    let linear: f64 = observed
        .iter()
        .zip(z.iter())
        .map(|(&o, &zi)| o * zi)
        .sum();
    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let ad1 = qd1 * c;
    weight
        * ((1.0 - event) * (-(normal_cdf(-eta1)).ln())
            + normal_cdf(-eta0).ln()
            - event * log_phi_eta1
            - event * ad1.ln())
}

// -------- fixture generators --------------------------------------------

fn make_full_psd(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    // Sigma = L L^T + 0.1 I, symmetric PSD.
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            l[[i, j]] = rng.uniform(-0.9, 0.9);
        }
    }
    let mut sigma = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for t in 0..k {
                s += l[[i, t]] * l[[j, t]];
            }
            sigma[[i, j]] = s;
        }
        sigma[[i, i]] += 0.1;
    }
    for i in 0..k {
        for j in (i + 1)..k {
            let avg = 0.5 * (sigma[[i, j]] + sigma[[j, i]]);
            sigma[[i, j]] = avg;
            sigma[[j, i]] = avg;
        }
    }
    MarginalSlopeCovariance::Full(sigma)
}

fn make_diagonal(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    let mut diag = Array1::<f64>::zeros(k);
    for i in 0..k {
        diag[i] = rng.uniform(0.05, 2.0);
    }
    MarginalSlopeCovariance::Diagonal(diag)
}

fn make_low_rank(rng: &mut SplitMix64, k: usize, rank: usize) -> MarginalSlopeCovariance {
    let r = rank.max(1).min(k + 1);
    let mut factor = Array2::<f64>::zeros((k, r));
    for i in 0..k {
        for j in 0..r {
            factor[[i, j]] = rng.uniform(-0.7, 0.7);
        }
    }
    MarginalSlopeCovariance::LowRank(factor)
}

fn random_slopes(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    (0..k).map(|_| rng.uniform(-0.6, 0.6)).collect()
}

fn random_z(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    (0..k).map(|_| rng.uniform(-1.5, 1.5)).collect()
}

// -------- 1. Closed-form match across dims/shapes/events ----------------

#[test]
fn neglog_matches_closed_form_across_dims_shapes_events() {
    let ks = [1usize, 2, 3, 5];
    let events = [0.0_f64, 1.0];
    let mut seed = 0xC0FFEE_BAD_F00D_u64;

    for &k in &ks {
        for &event in &events {
            for shape_id in 0..3u32 {
                for fixture in 0..30u64 {
                    seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut rng =
                        SplitMix64::new(seed ^ ((k as u64) << 32) ^ fixture ^ ((shape_id as u64) << 16));

                    let covariance = match shape_id {
                        0 => make_diagonal(&mut rng, k),
                        1 => make_full_psd(&mut rng, k),
                        _ => make_low_rank(&mut rng, k, ((fixture as usize % k.max(1)) + 1).max(1)),
                    };
                    let slopes = random_slopes(&mut rng, k);
                    let z = random_z(&mut rng, k);
                    let probit_scale = rng.uniform(0.4, 1.6);
                    let weight = rng.uniform(0.2, 2.5);
                    let q0 = rng.uniform(-1.5, 1.5);
                    let q1 = rng.uniform(-1.5, 1.5);
                    let qd1 = rng.uniform(0.1, 3.0);

                    let actual = survival_marginal_slope_vector_neglog(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &covariance,
                        weight,
                        event,
                        1e-9,
                        probit_scale,
                    )
                    .expect("neglog");
                    let expected = neglog_reference(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &covariance,
                        weight,
                        event,
                        probit_scale,
                    );
                    let scale = 1.0 + expected.abs();
                    assert!(
                        (actual - expected).abs() <= 1e-13 * scale,
                        "k={k} shape={shape_id} event={event} fixture={fixture}: \
                         actual={actual:.17e} expected={expected:.17e}"
                    );
                }
            }
        }
    }
}

// -------- 2. event=0 -> qd1 term vanishes (bitwise) ---------------------

#[test]
fn neglog_event_zero_is_bitwise_invariant_under_qd1() {
    let slopes = [0.31, -0.18, 0.05];
    let z = [0.7, -1.2, 0.4];
    let covariance = MarginalSlopeCovariance::Full(ndarray::array![
        [1.2, 0.3, -0.1],
        [0.3, 0.9, 0.2],
        [-0.1, 0.2, 1.1],
    ]);
    let probit_scale = 0.85;
    let weight = 1.4;
    let q0 = 0.21;
    let q1 = 0.83;

    let v_small = survival_marginal_slope_vector_neglog(
        q0,
        q1,
        1e-3,
        &slopes,
        &z,
        &covariance,
        weight,
        0.0,
        1e-9,
        probit_scale,
    )
    .expect("neglog small qd1");
    let v_large = survival_marginal_slope_vector_neglog(
        q0,
        q1,
        1e6,
        &slopes,
        &z,
        &covariance,
        weight,
        0.0,
        1e-9,
        probit_scale,
    )
    .expect("neglog large qd1");
    assert_eq!(
        v_small.to_bits(),
        v_large.to_bits(),
        "event=0 must make qd1 irrelevant: small={v_small:.17e} large={v_large:.17e}"
    );
}

// -------- 3. Derivative guard violation ---------------------------------

#[test]
fn neglog_errors_when_derivative_guard_is_violated() {
    for k in [1usize, 2, 3] {
        let mut rng = SplitMix64::new(0xDEAD_BEEF_0000 ^ (k as u64));
        let covariance = make_full_psd(&mut rng, k);
        let slopes = random_slopes(&mut rng, k);
        let z = random_z(&mut rng, k);
        let err = survival_marginal_slope_vector_neglog(
            0.1,
            0.2,
            1e-7,
            &slopes,
            &z,
            &covariance,
            1.0,
            1.0,
            1e-6,
            1.0,
        )
        .expect_err("derivative guard should error");
        assert!(
            err.contains("monotonicity violated"),
            "k={k}: error message missing 'monotonicity violated': {err}"
        );
    }
}

// -------- 4. Continuity above the guard ---------------------------------

#[test]
fn neglog_finite_and_continuous_just_above_guard() {
    let covariance =
        MarginalSlopeCovariance::Full(ndarray::array![[1.0, 0.3], [0.3, 0.8]]);
    let slopes = [0.21, -0.14];
    let z = [0.5, -0.7];
    let probit_scale = 0.9;
    let weight = 1.0;
    let event = 1.0;
    let q0 = 0.1;
    let q1 = 0.4;

    let lo: f64 = 1.01e-6;
    let hi: f64 = 2e-6;
    let n: usize = 50;
    let log_lo = lo.ln();
    let log_hi = hi.ln();

    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        // Sweep log-linearly from hi (2e-6) down to lo (1.01e-6); all > guard 1e-6.
        let qd1 = (log_hi + (log_lo - log_hi) * t).exp();
        assert!(qd1 > 1e-6, "qd1={qd1:.3e} must clear guard");
        let v = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slopes,
            &z,
            &covariance,
            weight,
            event,
            1e-6,
            probit_scale,
        )
        .expect("neglog above guard");
        assert!(v.is_finite(), "v not finite at i={i} qd1={qd1:.3e}: {v:.17e}");
        values.push(v);
    }
    for (i, w) in values.windows(2).enumerate() {
        let diff = (w[1] - w[0]).abs();
        assert!(diff < 1e6, "i={i}: consecutive diff too large: {diff:.3e}");
    }
}

// -------- 5. Weight linearity -------------------------------------------

#[test]
fn neglog_is_linear_in_weight_and_zero_for_zero_weight() {
    let covariance =
        MarginalSlopeCovariance::Diagonal(ndarray::array![1.2, 0.7, 0.5]);
    let slopes = [0.18, -0.22, 0.1];
    let z = [0.6, -0.4, 0.9];
    let probit_scale = 0.8;
    let q0 = 0.15;
    let q1 = 0.55;
    let qd1 = 0.9;
    let event = 1.0;

    let base = survival_marginal_slope_vector_neglog(
        q0, q1, qd1, &slopes, &z, &covariance, 1.0, event, 1e-9, probit_scale,
    )
    .expect("neglog unit weight");

    for &w in &[0.25_f64, 0.5, 2.0, 3.7] {
        let v = survival_marginal_slope_vector_neglog(
            q0, q1, qd1, &slopes, &z, &covariance, w, event, 1e-9, probit_scale,
        )
        .expect("neglog weighted");
        let expected = w * base;
        let scale = 1.0 + expected.abs();
        assert!(
            (v - expected).abs() <= 1e-13 * scale,
            "w={w}: v={v:.17e} expected={expected:.17e}"
        );
    }

    let zero = survival_marginal_slope_vector_neglog(
        q0, q1, qd1, &slopes, &z, &covariance, 0.0, event, 1e-9, probit_scale,
    )
    .expect("neglog zero weight");
    assert_eq!(zero, 0.0, "weight=0 should yield exactly 0.0 (got {zero:.17e})");
    assert_eq!(zero.to_bits(), 0.0_f64.to_bits(), "weight=0 should be +0.0");
}

// -------- 6. Joint (z, slopes) negation invariance ----------------------

#[test]
fn neglog_invariant_under_joint_negation_of_z_and_slopes() {
    let ks = [2usize, 3, 4];
    let mut seed = 0xABCDEF_1234_u64;

    for &k in &ks {
        for fixture in 0..50u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut rng = SplitMix64::new(seed ^ (k as u64) ^ fixture);
            let shape_id = fixture % 3;
            let covariance = match shape_id {
                0 => make_diagonal(&mut rng, k),
                1 => make_full_psd(&mut rng, k),
                _ => make_low_rank(&mut rng, k, ((fixture as usize % k.max(1)) + 1).max(1)),
            };
            let slopes = random_slopes(&mut rng, k);
            let z = random_z(&mut rng, k);
            let neg_slopes: Vec<f64> = slopes.iter().map(|&v| -v).collect();
            let neg_z: Vec<f64> = z.iter().map(|&v| -v).collect();
            let probit_scale = rng.uniform(0.5, 1.4);
            let weight = rng.uniform(0.3, 2.0);
            let event = if fixture % 2 == 0 { 1.0 } else { 0.0 };
            let q0 = rng.uniform(-1.2, 1.2);
            let q1 = rng.uniform(-1.2, 1.2);
            let qd1 = rng.uniform(0.2, 2.5);

            let a = survival_marginal_slope_vector_neglog(
                q0,
                q1,
                qd1,
                &slopes,
                &z,
                &covariance,
                weight,
                event,
                1e-9,
                probit_scale,
            )
            .expect("a");
            let b = survival_marginal_slope_vector_neglog(
                q0,
                q1,
                qd1,
                &neg_slopes,
                &neg_z,
                &covariance,
                weight,
                event,
                1e-9,
                probit_scale,
            )
            .expect("b");
            let scale = 1.0 + a.abs();
            assert!(
                (a - b).abs() <= 1e-14 * scale,
                "k={k} fixture={fixture}: a={a:.17e} b={b:.17e}"
            );
        }
    }
}

// -------- 7. Lone-sign flip changes value (sanity) ----------------------

#[test]
fn neglog_lone_sign_flip_changes_value() {
    let covariance = MarginalSlopeCovariance::Full(ndarray::array![
        [1.0, 0.25],
        [0.25, 0.7],
    ]);
    let slopes = [0.31, -0.22];
    let z = [0.8, -0.5];
    let probit_scale = 0.95;
    let weight = 1.2;
    let event = 1.0;
    let q0 = 0.2;
    let q1 = 0.55;
    let qd1 = 0.9;

    let base = survival_marginal_slope_vector_neglog(
        q0, q1, qd1, &slopes, &z, &covariance, weight, event, 1e-9, probit_scale,
    )
    .expect("base");
    let neg_z: Vec<f64> = z.iter().map(|&v| -v).collect();
    let neg_slopes: Vec<f64> = slopes.iter().map(|&v| -v).collect();

    let flip_z = survival_marginal_slope_vector_neglog(
        q0, q1, qd1, &slopes, &neg_z, &covariance, weight, event, 1e-9, probit_scale,
    )
    .expect("flip_z");
    let flip_slopes = survival_marginal_slope_vector_neglog(
        q0,
        q1,
        qd1,
        &neg_slopes,
        &z,
        &covariance,
        weight,
        event,
        1e-9,
        probit_scale,
    )
    .expect("flip_slopes");
    assert!(
        (base - flip_z).abs() > 1e-6,
        "flipping z alone should change neglog: base={base:.17e} flip_z={flip_z:.17e}"
    );
    assert!(
        (base - flip_slopes).abs() > 1e-6,
        "flipping slopes alone should change neglog: base={base:.17e} flip_slopes={flip_slopes:.17e}"
    );
}

// -------- 8. Probit-scale / slope homogeneity ---------------------------

#[test]
fn neglog_invariant_under_probit_scale_slope_rescale() {
    let covariance = MarginalSlopeCovariance::Full(ndarray::array![
        [1.1, 0.2, -0.05],
        [0.2, 0.9, 0.15],
        [-0.05, 0.15, 1.3],
    ]);
    let slopes = [0.22, -0.31, 0.17];
    let z = [0.7, -0.4, 1.1];
    let weight = 1.1;
    let event = 1.0;
    let q0 = 0.18;
    let q1 = 0.62;
    let qd1 = 0.85;
    let s_base = 0.95_f64;

    let base = survival_marginal_slope_vector_neglog(
        q0, q1, qd1, &slopes, &z, &covariance, weight, event, 1e-9, s_base,
    )
    .expect("base");

    for &alpha in &[0.5_f64, 1.0, 2.0, 10.0] {
        let s_scaled = alpha * s_base;
        let slopes_scaled: Vec<f64> = slopes.iter().map(|&g| g / alpha).collect();
        let v = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slopes_scaled,
            &z,
            &covariance,
            weight,
            event,
            1e-9,
            s_scaled,
        )
        .expect("scaled");
        let scale = 1.0 + base.abs();
        assert!(
            (v - base).abs() <= 1e-13 * scale,
            "alpha={alpha}: v={v:.17e} base={base:.17e}"
        );
    }
}

// -------- 9. Dimension mismatch -----------------------------------------

#[test]
fn neglog_errors_on_dimension_mismatch() {
    let covariance =
        MarginalSlopeCovariance::Full(ndarray::array![[1.0, 0.2], [0.2, 0.8]]);
    let slopes_ok = [0.3, -0.2];
    let z_ok = [0.5, -0.4];

    // z too short.
    let err = survival_marginal_slope_vector_neglog(
        0.1, 0.4, 0.9, &slopes_ok, &[0.5], &covariance, 1.0, 1.0, 1e-9, 1.0,
    )
    .expect_err("z dim mismatch must error");
    assert!(!err.is_empty());

    // slopes too long.
    let err = survival_marginal_slope_vector_neglog(
        0.1,
        0.4,
        0.9,
        &[0.3, -0.2, 0.1],
        &z_ok,
        &covariance,
        1.0,
        1.0,
        1e-9,
        1.0,
    )
    .expect_err("slopes dim mismatch must error");
    assert!(!err.is_empty());

    // Both wrong vs. covariance.
    let err = survival_marginal_slope_vector_neglog(
        0.1,
        0.4,
        0.9,
        &[0.3, -0.2, 0.1],
        &[0.5, -0.4, 0.2],
        &covariance,
        1.0,
        1.0,
        1e-9,
        1.0,
    )
    .expect_err("both dim mismatch must error");
    assert!(!err.is_empty());
}

// -------- 10. Non-finite inputs ----------------------------------------

#[test]
fn neglog_nonfinite_inputs_yield_error_or_nonfinite() {
    let covariance =
        MarginalSlopeCovariance::Diagonal(ndarray::array![1.0, 0.5]);
    let good_slopes = [0.21, -0.13];
    let good_z = [0.5, -0.3];
    let probit_scale = 0.9;
    let weight = 1.0;
    let event = 1.0;

    let check = |label: &str, result: Result<f64, String>| match result {
        Err(_) => {}
        Ok(v) => assert!(
            !v.is_finite(),
            "{label}: expected Err or non-finite output, got finite {v:.17e}"
        ),
    };

    let nan = f64::NAN;
    let inf = f64::INFINITY;

    check(
        "q0=NaN",
        survival_marginal_slope_vector_neglog(
            nan,
            0.4,
            0.9,
            &good_slopes,
            &good_z,
            &covariance,
            weight,
            event,
            1e-9,
            probit_scale,
        ),
    );
    check(
        "q1=Inf",
        survival_marginal_slope_vector_neglog(
            0.1,
            inf,
            0.9,
            &good_slopes,
            &good_z,
            &covariance,
            weight,
            event,
            1e-9,
            probit_scale,
        ),
    );
    check(
        "qd1=NaN",
        survival_marginal_slope_vector_neglog(
            0.1,
            0.4,
            nan,
            &good_slopes,
            &good_z,
            &covariance,
            weight,
            event,
            1e-9,
            probit_scale,
        ),
    );
    check(
        "slopes has NaN",
        survival_marginal_slope_vector_neglog(
            0.1,
            0.4,
            0.9,
            &[nan, -0.13],
            &good_z,
            &covariance,
            weight,
            event,
            1e-9,
            probit_scale,
        ),
    );
    check(
        "z has Inf",
        survival_marginal_slope_vector_neglog(
            0.1,
            0.4,
            0.9,
            &good_slopes,
            &[inf, -0.3],
            &covariance,
            weight,
            event,
            1e-9,
            probit_scale,
        ),
    );
    check(
        "weight=NaN",
        survival_marginal_slope_vector_neglog(
            0.1,
            0.4,
            0.9,
            &good_slopes,
            &good_z,
            &covariance,
            nan,
            event,
            1e-9,
            probit_scale,
        ),
    );
    check(
        "probit_scale=Inf",
        survival_marginal_slope_vector_neglog(
            0.1,
            0.4,
            0.9,
            &good_slopes,
            &good_z,
            &covariance,
            weight,
            event,
            1e-9,
            inf,
        ),
    );
}
