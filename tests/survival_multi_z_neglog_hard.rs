//! Hard correctness, monotonicity, and edge-case tests for
//! `survival_marginal_slope_vector_neglog`.
//!
//! See module header comment in this file for the design of each test.

use gam::bernoulli_marginal_slope::MarginalSlopeCovariance;
use gam::probability::normal_cdf;
use gam::survival_marginal_slope::survival_marginal_slope_vector_neglog;
use ndarray::{Array1, Array2};

// ------------------- splitmix64 PRNG -------------------

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
    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    /// Uniform in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

// ------------------- helpers -------------------

fn random_slopes(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    (0..k).map(|_| rng.uniform(-0.6, 0.6)).collect()
}

fn random_z(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    (0..k).map(|_| rng.uniform(-1.5, 1.5)).collect()
}

/// Build a random covariance of each kind given seed.
fn random_diagonal(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    let arr: Vec<f64> = (0..k).map(|_| rng.uniform(0.3, 2.0)).collect();
    MarginalSlopeCovariance::Diagonal(Array1::from(arr))
}

fn random_full(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    // Build A then A A^T + small jitter for SPD
    let mut a = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            a[[i, j]] = rng.uniform(-0.7, 0.7);
        }
    }
    let mut cov = a.dot(&a.t());
    for i in 0..k {
        cov[[i, i]] += 0.4;
    }
    MarginalSlopeCovariance::Full(cov)
}

fn random_low_rank(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    let r = std::cmp::max(1, k.saturating_sub(1)).min(k);
    let mut factor = Array2::<f64>::zeros((k, r));
    for i in 0..k {
        for j in 0..r {
            factor[[i, j]] = rng.uniform(-0.8, 0.8);
        }
    }
    MarginalSlopeCovariance::LowRank(factor)
}

fn covariance_quadratic_form(cov: &MarginalSlopeCovariance, v: &[f64]) -> f64 {
    cov.quadratic_form(v).expect("quadratic form")
}

/// Closed-form computation following tests/survival_multi_z_marginal_slope.rs:66-106
/// pattern, generalized to all covariance shapes.
fn closed_form_neglog(
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
    let c = (1.0 + covariance_quadratic_form(covariance, &observed)).sqrt();
    let linear: f64 = observed.iter().zip(z.iter()).map(|(&o, &zi)| o * zi).sum();
    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    weight
        * ((1.0 - event) * (-normal_cdf(-eta1).ln()) + normal_cdf(-eta0).ln()
            - event * log_phi_eta1
            - event * (qd1 * c).ln())
}

fn make_covariance(rng: &mut SplitMix64, shape: usize, k: usize) -> MarginalSlopeCovariance {
    match shape {
        0 => random_diagonal(rng, k),
        1 => random_full(rng, k),
        _ => random_low_rank(rng, k),
    }
}

// ------------------- Test 1: closed-form match -------------------

#[test]
fn closed_form_match_across_k_and_cov_shapes() {
    let mut rng = SplitMix64::new(0xC0FFEE_u64);
    let ks = [1usize, 2, 3, 5];
    let mut total = 0usize;
    let mut worst: f64 = 0.0;
    for &k in &ks {
        for shape in 0..3 {
            for event_i in 0..2 {
                let event = event_i as f64;
                for fixture in 0..30 {
                    let _ = fixture;
                    let slopes = random_slopes(&mut rng, k);
                    let z = random_z(&mut rng, k);
                    let cov = make_covariance(&mut rng, shape, k);
                    let probit_scale = rng.uniform(0.4, 1.6);
                    let weight = rng.uniform(0.2, 3.0);
                    let q0 = rng.uniform(-1.0, 1.0);
                    let q1 = rng.uniform(-1.0, 1.0);
                    // Keep qd1 well above derivative guard.
                    let qd1 = rng.uniform(0.05, 2.0);

                    let actual = survival_marginal_slope_vector_neglog(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &cov,
                        weight,
                        event,
                        1e-6,
                        probit_scale,
                    )
                    .expect("vector neglog");
                    let expected = closed_form_neglog(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &cov,
                        weight,
                        event,
                        probit_scale,
                    );
                    let diff = (actual - expected).abs();
                    if diff > worst {
                        worst = diff;
                    }
                    assert!(
                        diff <= 1e-13,
                        "k={k} shape={shape} event={event} diff={diff:.3e} actual={actual:.17e} expected={expected:.17e}"
                    );
                    total += 1;
                }
            }
        }
    }
    assert!(total == 4 * 3 * 2 * 30);
    eprintln!("closed_form_match: {total} fixtures, worst |diff| = {worst:.3e}");
}

// ------------------- Test 2: event=0 collapses qd1 term -------------------

#[test]
fn event_zero_makes_qd1_irrelevant_bitwise() {
    let cov = MarginalSlopeCovariance::Full(
        Array2::from_shape_vec((2, 2), vec![1.4, 0.3, 0.3, 0.9]).unwrap(),
    );
    let slopes = [0.31, -0.2];
    let z = [0.5, -0.4];
    let common = |qd1: f64| -> f64 {
        survival_marginal_slope_vector_neglog(
            -0.1, 0.4, qd1, &slopes, &z, &cov, 1.25, 0.0, 1e-6, 0.85,
        )
        .expect("neglog")
    };
    let small = common(1e-3);
    let large = common(1e6);
    assert_eq!(
        small.to_bits(),
        large.to_bits(),
        "event=0 must not depend on qd1; got small={small:.17e} large={large:.17e}"
    );
}

// ------------------- Test 3: derivative guard violation -------------------

#[test]
fn derivative_guard_violation_event_one_errors() {
    for k in [1usize, 2, 3] {
        let cov = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0; k]));
        let slopes = vec![0.2; k];
        let z = vec![0.4; k];
        let err = survival_marginal_slope_vector_neglog(
            0.0, 0.2, 1e-7, &slopes, &z, &cov, 1.0, 1.0, 1e-6, 1.0,
        )
        .expect_err("derivative guard violation must Err for k={k}");
        assert!(
            err.contains("monotonicity violated"),
            "k={k}: expected monotonicity-violated error, got: {err}"
        );
    }
}

// ------------------- Test 4: derivative-guard boundary continuity -------------------

#[test]
fn derivative_guard_boundary_continuity_k2_full() {
    let cov = MarginalSlopeCovariance::Full(
        Array2::from_shape_vec((2, 2), vec![1.3, 0.4, 0.4, 0.7]).unwrap(),
    );
    let slopes = [0.27, -0.15];
    let z = [0.6, -0.4];
    let guard = 1e-6;
    let lo = 1.01e-6;
    let hi = 2e-6;
    let n = 50usize;
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        // Sweep from hi down to lo log-linearly.
        let t = i as f64 / (n - 1) as f64;
        let qd1 = (log_hi + (log_lo - log_hi) * t).exp();
        assert!(qd1 > guard, "test bug: qd1 must remain above guard");
        let val = survival_marginal_slope_vector_neglog(
            -0.1, 0.3, qd1, &slopes, &z, &cov, 1.2, 1.0, guard, 0.9,
        )
        .expect("neglog should not error above guard");
        assert!(val.is_finite(), "non-finite at qd1={qd1:.3e}: {val}");
        values.push((qd1, val));
    }
    for w in values.windows(2) {
        let step = (w[1].1 - w[0].1).abs();
        assert!(
            step < 1e6,
            "huge jump between qd1={:.3e} ({:.6e}) and qd1={:.3e} ({:.6e}); step={:.3e}",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1,
            step
        );
    }
}

// ------------------- Test 5: weight scaling linearity -------------------

#[test]
fn weight_scaling_is_linear() {
    let mut rng = SplitMix64::new(0xBADC0DE_u64);
    for _ in 0..40 {
        let k = 2 + (rng.next_u64() % 3) as usize; // 2..=4
        let shape = (rng.next_u64() % 3) as usize;
        let cov = make_covariance(&mut rng, shape, k);
        let slopes = random_slopes(&mut rng, k);
        let z = random_z(&mut rng, k);
        let probit_scale = rng.uniform(0.5, 1.4);
        let q0 = rng.uniform(-0.9, 0.9);
        let q1 = rng.uniform(-0.9, 0.9);
        let qd1 = rng.uniform(0.05, 1.5);
        let event = if rng.next_u64() & 1 == 0 { 0.0 } else { 1.0 };

        let base = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slopes,
            &z,
            &cov,
            1.0,
            event,
            1e-6,
            probit_scale,
        )
        .expect("base neglog");

        // The exact zero case.
        let zeroed = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slopes,
            &z,
            &cov,
            0.0,
            event,
            1e-6,
            probit_scale,
        )
        .expect("w=0 neglog");
        assert_eq!(
            zeroed.to_bits(),
            0.0_f64.to_bits(),
            "w=0 must return exactly 0.0, got {zeroed:.17e}"
        );

        for &w in &[0.25, 0.7, 1.5, 3.3, 4.9] {
            let actual = survival_marginal_slope_vector_neglog(
                q0,
                q1,
                qd1,
                &slopes,
                &z,
                &cov,
                w,
                event,
                1e-6,
                probit_scale,
            )
            .expect("scaled neglog");
            let expected = w * base;
            let tol = 1e-13_f64.max(1e-13 * expected.abs());
            assert!(
                (actual - expected).abs() <= tol,
                "w={w} actual={actual:.17e} expected={expected:.17e}"
            );
        }
    }
}

// ------------------- Test 6: joint z and slope negation invariance -------------------

#[test]
fn joint_z_slope_negation_leaves_neglog_unchanged() {
    let mut rng = SplitMix64::new(0xDEC0DE5_u64);
    for k in [2usize, 3, 4] {
        for _ in 0..50 {
            let shape = (rng.next_u64() % 3) as usize;
            let cov = make_covariance(&mut rng, shape, k);
            let slopes = random_slopes(&mut rng, k);
            let z = random_z(&mut rng, k);
            let probit_scale = rng.uniform(0.5, 1.3);
            let q0 = rng.uniform(-0.8, 0.8);
            let q1 = rng.uniform(-0.8, 0.8);
            let qd1 = rng.uniform(0.05, 1.5);
            let weight = rng.uniform(0.3, 2.5);
            let event = if rng.next_u64() & 1 == 0 { 0.0 } else { 1.0 };

            let neg_slopes: Vec<f64> = slopes.iter().map(|&g| -g).collect();
            let neg_z: Vec<f64> = z.iter().map(|&zi| -zi).collect();

            let a = survival_marginal_slope_vector_neglog(
                q0,
                q1,
                qd1,
                &slopes,
                &z,
                &cov,
                weight,
                event,
                1e-6,
                probit_scale,
            )
            .expect("a");
            let b = survival_marginal_slope_vector_neglog(
                q0,
                q1,
                qd1,
                &neg_slopes,
                &neg_z,
                &cov,
                weight,
                event,
                1e-6,
                probit_scale,
            )
            .expect("b");
            let diff = (a - b).abs();
            let tol = 1e-14_f64.max(1e-14 * a.abs().max(b.abs()));
            assert!(
                diff <= tol,
                "k={k}: joint negation changed neglog: a={a:.17e} b={b:.17e} diff={diff:.3e}"
            );
        }
    }
}

// ------------------- Test 7: single negation usually changes the value -------------------

#[test]
fn single_negation_changes_neglog() {
    // Use a fixture with definitely nonzero z and slopes and linear coupling.
    let cov = MarginalSlopeCovariance::Full(
        Array2::from_shape_vec((3, 3), vec![1.2, 0.3, -0.1, 0.3, 0.9, 0.2, -0.1, 0.2, 1.1])
            .unwrap(),
    );
    let slopes = [0.4, -0.3, 0.25];
    let z = [0.8, -0.7, 1.1];
    let probit_scale = 0.95;
    let q0 = 0.2;
    let q1 = 0.55;
    let qd1 = 0.7;
    let weight = 1.3;
    for &event in &[0.0_f64, 1.0_f64] {
        let base = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slopes,
            &z,
            &cov,
            weight,
            event,
            1e-6,
            probit_scale,
        )
        .expect("base");

        let neg_z: Vec<f64> = z.iter().map(|&zi| -zi).collect();
        let z_flipped = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slopes,
            &neg_z,
            &cov,
            weight,
            event,
            1e-6,
            probit_scale,
        )
        .expect("z flipped");
        assert!(
            (base - z_flipped).abs() > 1e-6,
            "event={event}: flipping only z must change neglog (base={base:.6e} z_flipped={z_flipped:.6e})"
        );

        let neg_slopes: Vec<f64> = slopes.iter().map(|&g| -g).collect();
        let slope_flipped = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &neg_slopes,
            &z,
            &cov,
            weight,
            event,
            1e-6,
            probit_scale,
        )
        .expect("slope flipped");
        assert!(
            (base - slope_flipped).abs() > 1e-6,
            "event={event}: flipping only slopes must change neglog (base={base:.6e} slope_flipped={slope_flipped:.6e})"
        );
    }
}

// ------------------- Test 8: probit-scale / slope homogeneity -------------------

#[test]
fn probit_scale_slope_homogeneity() {
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0_u64);
    for k in [1usize, 2, 3] {
        for _ in 0..30 {
            let shape = (rng.next_u64() % 3) as usize;
            let cov = make_covariance(&mut rng, shape, k);
            let slopes = random_slopes(&mut rng, k);
            let z = random_z(&mut rng, k);
            let probit_scale = rng.uniform(0.5, 1.3);
            let q0 = rng.uniform(-0.8, 0.8);
            let q1 = rng.uniform(-0.8, 0.8);
            let qd1 = rng.uniform(0.05, 1.5);
            let weight = rng.uniform(0.3, 2.5);
            let event = if rng.next_u64() & 1 == 0 { 0.0 } else { 1.0 };

            let base = survival_marginal_slope_vector_neglog(
                q0,
                q1,
                qd1,
                &slopes,
                &z,
                &cov,
                weight,
                event,
                1e-6,
                probit_scale,
            )
            .expect("base");

            for &alpha in &[0.5_f64, 1.0, 2.0, 10.0] {
                let scaled_slopes: Vec<f64> = slopes.iter().map(|&g| g / alpha).collect();
                let scaled_probit = alpha * probit_scale;
                let v = survival_marginal_slope_vector_neglog(
                    q0,
                    q1,
                    qd1,
                    &scaled_slopes,
                    &z,
                    &cov,
                    weight,
                    event,
                    1e-6,
                    scaled_probit,
                )
                .expect("scaled");
                let diff = (v - base).abs();
                let tol = 1e-13_f64.max(1e-13 * base.abs().max(v.abs()));
                assert!(
                    diff <= tol,
                    "k={k} alpha={alpha}: homogeneity broken, base={base:.17e} v={v:.17e} diff={diff:.3e}"
                );
            }
        }
    }
}

// ------------------- Test 9: dimension mismatch rejection -------------------

#[test]
fn dimension_mismatch_rejected() {
    // slope length != z length
    let cov = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0, 1.0]));
    let err = survival_marginal_slope_vector_neglog(
        0.0,
        0.3,
        0.5,
        &[0.2, -0.1],
        &[0.4],
        &cov,
        1.0,
        1.0,
        1e-6,
        1.0,
    )
    .expect_err("slope/z mismatch must Err");
    assert!(
        err.to_lowercase().contains("dimension mismatch")
            || err.to_lowercase().contains("mismatch"),
        "expected dimension mismatch error, got: {err}"
    );

    // z length != covariance dim
    let cov2 = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0, 1.0, 1.0]));
    let err2 = survival_marginal_slope_vector_neglog(
        0.0,
        0.3,
        0.5,
        &[0.2, -0.1],
        &[0.4, 0.5],
        &cov2,
        1.0,
        1.0,
        1e-6,
        1.0,
    )
    .expect_err("z/cov mismatch must Err");
    assert!(
        err2.to_lowercase().contains("mismatch"),
        "expected mismatch error, got: {err2}"
    );
}

// ------------------- Test 10: NaN/Inf propagation -------------------

#[test]
fn nan_inf_inputs_do_not_silently_return_finite() {
    let cov = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0, 1.0]));
    let slopes = [0.2, -0.1];
    let z = [0.4, 0.5];

    // q0 = NaN: result must be NaN or an Err.
    let r1 = survival_marginal_slope_vector_neglog(
        f64::NAN,
        0.3,
        0.5,
        &slopes,
        &z,
        &cov,
        1.0,
        1.0,
        1e-6,
        1.0,
    );
    match r1 {
        Err(_) => {}
        Ok(v) => assert!(
            v.is_nan(),
            "q0=NaN must yield NaN or Err, got finite {v:.17e}"
        ),
    }

    // slopes containing Inf
    let inf_slopes = [f64::INFINITY, -0.1];
    let r2 = survival_marginal_slope_vector_neglog(
        0.0,
        0.3,
        0.5,
        &inf_slopes,
        &z,
        &cov,
        1.0,
        1.0,
        1e-6,
        1.0,
    );
    match r2 {
        Err(_) => {}
        Ok(v) => assert!(
            !v.is_finite(),
            "slopes=Inf must yield non-finite or Err, got {v:.17e}"
        ),
    }

    // weight = NaN
    let r3 = survival_marginal_slope_vector_neglog(
        0.0,
        0.3,
        0.5,
        &slopes,
        &z,
        &cov,
        f64::NAN,
        1.0,
        1e-6,
        1.0,
    );
    match r3 {
        Err(_) => {}
        Ok(v) => assert!(
            v.is_nan(),
            "weight=NaN must yield NaN or Err, got finite {v:.17e}"
        ),
    }
}
