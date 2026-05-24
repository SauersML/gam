//! Hard tests for `marginal_slope_covariance_from_scores`: the shape
//! classifier that picks between Diagonal / Full / LowRank from a sample
//! score matrix. Used by the multi-z survival fit pipeline
//! (src/families/survival_marginal_slope.rs:17169).
//!
//! Each test below is intentionally adversarial — if the classifier picks
//! the wrong shape, the test is supposed to fail. Do not weaken thresholds
//! to make these pass.

use gam::bernoulli_marginal_slope::{
    MarginalSlopeCovariance, MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores,
};
use ndarray::{Array1, Array2, ArrayView2};

// ---------- inline RNG: splitmix64 + Box-Muller -------------------------

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn next_unit_f64(state: &mut u64) -> f64 {
    // 53-bit mantissa in [0,1)
    let bits = splitmix64(state) >> 11;
    (bits as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[inline]
fn next_open_unit(state: &mut u64) -> f64 {
    // (0,1) — avoid log(0) in Box-Muller
    loop {
        let u = next_unit_f64(state);
        if u > 0.0 {
            return u;
        }
    }
}

fn standard_normal_pair(state: &mut u64) -> (f64, f64) {
    let u1 = next_open_unit(state);
    let u2 = next_unit_f64(state);
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn fill_standard_normal(out: &mut [f64], state: &mut u64) {
    let mut i = 0;
    while i + 1 < out.len() {
        let (a, b) = standard_normal_pair(state);
        out[i] = a;
        out[i + 1] = b;
        i += 2;
    }
    if i < out.len() {
        let (a, _) = standard_normal_pair(state);
        out[i] = a;
    }
}

fn make_iid_normal_scores(n: usize, k: usize, state: &mut u64) -> Array2<f64> {
    let mut data = vec![0.0; n * k];
    fill_standard_normal(&mut data, state);
    Array2::from_shape_vec((n, k), data).expect("shape")
}

// ---------- helpers -----------------------------------------------------

fn ones_weights(n: usize) -> Array1<f64> {
    Array1::from_elem(n, 1.0)
}

fn cov_quadratic_full(sigma: &Array2<f64>, v: &[f64]) -> f64 {
    let k = sigma.nrows();
    let mut total = 0.0;
    for i in 0..k {
        let mut row = 0.0;
        for j in 0..k {
            row += sigma[[i, j]] * v[j];
        }
        total += v[i] * row;
    }
    total
}

fn classify(scores: ArrayView2<'_, f64>, w: &Array1<f64>) -> MarginalSlopeCovarianceShape {
    marginal_slope_covariance_from_scores(scores, w)
        .expect("classifier should not error")
        .shape()
}

// ====================================================================
// Test 1: pure diagonal recovery
// ====================================================================
#[test]
fn t01_pure_diagonal_iid_normals_classifies_as_diagonal() {
    let n = 10_000;
    let k = 4;
    let seeds = 50;
    let weights = ones_weights(n);

    let mut diagonal_hits = 0;
    let mut full_hits = 0;
    let mut low_rank_hits = 0;
    let mut failures: Vec<String> = Vec::new();
    for seed in 0..seeds {
        let mut state = 0xC0FF_EE00u64.wrapping_add(seed as u64 * 0x9E37_79B9);
        let scores = make_iid_normal_scores(n, k, &mut state);
        match classify(scores.view(), &weights) {
            MarginalSlopeCovarianceShape::Diagonal => diagonal_hits += 1,
            MarginalSlopeCovarianceShape::Full => {
                full_hits += 1;
                if failures.len() < 4 {
                    failures.push(format!("seed {seed}: got Full"));
                }
            }
            MarginalSlopeCovarianceShape::LowRank => {
                low_rank_hits += 1;
                if failures.len() < 4 {
                    failures.push(format!("seed {seed}: got LowRank"));
                }
            }
        }
    }
    println!(
        "t01 summary: diagonal={diagonal_hits}/{seeds}, full={full_hits}, low_rank={low_rank_hits}"
    );
    assert!(
        diagonal_hits >= 48,
        "IID-normal scores should be classified Diagonal in ≥48/50 seeds; got {diagonal_hits}/{seeds} (full={full_hits}, low_rank={low_rank_hits}); first failures: {failures:?}"
    );
}

// ====================================================================
// Test 2: rank deficiency detection
// ====================================================================
#[test]
fn t02a_perfect_col2_eq_1_5_col1_is_low_rank() {
    let n = 10_000;
    let k = 4;
    let seeds = 50;
    let weights = ones_weights(n);
    for seed in 0..seeds {
        let mut state = 0xDEAD_BEEFu64.wrapping_add(seed as u64 * 0x9E37_79B9);
        let mut scores = make_iid_normal_scores(n, k, &mut state);
        // Force col2 (index 1) = 1.5 * col1 (index 0)
        for i in 0..n {
            scores[[i, 1]] = 1.5 * scores[[i, 0]];
        }
        let shape = classify(scores.view(), &weights);
        assert_eq!(
            shape,
            MarginalSlopeCovarianceShape::LowRank,
            "seed {seed}: expected LowRank for col2=1.5·col1, got {shape:?}"
        );
    }
}

#[test]
fn t02b_perfect_col3_eq_half_col1_plus_half_col2_is_low_rank() {
    let n = 10_000;
    let k = 4;
    let seeds = 50;
    let weights = ones_weights(n);
    for seed in 0..seeds {
        let mut state = 0xFEED_FACEu64.wrapping_add(seed as u64 * 0x9E37_79B9);
        let mut scores = make_iid_normal_scores(n, k, &mut state);
        for i in 0..n {
            scores[[i, 2]] = 0.5 * scores[[i, 0]] + 0.5 * scores[[i, 1]];
        }
        let shape = classify(scores.view(), &weights);
        assert_eq!(
            shape,
            MarginalSlopeCovarianceShape::LowRank,
            "seed {seed}: expected LowRank for col3=½·col1+½·col2, got {shape:?}"
        );
    }
}

#[test]
fn t02c_perfect_col5_eq_col1_duplicate_is_low_rank() {
    let n = 10_000;
    let k = 5;
    let seeds = 50;
    let weights = ones_weights(n);
    for seed in 0..seeds {
        let mut state = 0xBAD0_F00Du64.wrapping_add(seed as u64 * 0x9E37_79B9);
        let mut scores = make_iid_normal_scores(n, k, &mut state);
        for i in 0..n {
            scores[[i, 4]] = scores[[i, 0]];
        }
        let shape = classify(scores.view(), &weights);
        assert_eq!(
            shape,
            MarginalSlopeCovarianceShape::LowRank,
            "seed {seed}: expected LowRank for col5=col1, got {shape:?}"
        );
    }
}

// ====================================================================
// Test 3: full covariance recovery (correlated SPD)
// ====================================================================
#[test]
fn t03_correlated_spd_sigma_classifies_full_and_quadratic_form_matches() {
    // Pick a fixed SPD Sigma (3x3). L is lower-triangular Cholesky;
    // Sigma = L L'.
    let l = [[1.20_f64, 0.0, 0.0], [0.45, 0.95, 0.0], [-0.30, 0.20, 0.80]];
    let mut sigma = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for kidx in 0..3 {
                s += l[i][kidx] * l[j][kidx];
            }
            sigma[[i, j]] = s;
        }
    }

    let n = 10_000;
    let seeds = 50;
    let weights = ones_weights(n);

    let mut max_rel_err: f64 = 0.0;
    let mut full_hits = 0;
    for seed in 0..seeds {
        let mut state = 0xA11C_E000u64.wrapping_add(seed as u64 * 0x9E37_79B9);
        let z = make_iid_normal_scores(n, 3, &mut state); // IID N(0,1)
        // x_i = L z_i so cov(x) ≈ Sigma
        let mut scores = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            for a in 0..3 {
                let mut s = 0.0;
                for b in 0..=a {
                    s += l[a][b] * z[[i, b]];
                }
                scores[[i, a]] = s;
            }
        }
        let cov =
            marginal_slope_covariance_from_scores(scores.view(), &weights).expect("classifier ok");
        assert_eq!(
            cov.shape(),
            MarginalSlopeCovarianceShape::Full,
            "seed {seed}: SPD-correlated scores should classify Full, got {:?}",
            cov.shape()
        );
        full_hits += 1;
        // Test quadratic form against population Sigma on a few random v.
        let mut vstate = 0xBEEF_0000u64.wrapping_add(seed as u64);
        for _ in 0..6 {
            let (v0, v1) = standard_normal_pair(&mut vstate);
            let (v2, _) = standard_normal_pair(&mut vstate);
            let v = [v0, v1, v2];
            let got = cov.quadratic_form(&v).expect("qf");
            let want = cov_quadratic_full(&sigma, &v);
            let rel = (got - want).abs() / want.abs().max(1e-12);
            max_rel_err = max_rel_err.max(rel);
            assert!(
                rel <= 0.05,
                "seed {seed}: quadratic form rel err {rel:.3e} (got {got:.6}, want {want:.6}) on v={v:?}"
            );
        }
    }
    println!("t03 summary: full={full_hits}/{seeds}, max_quadratic_form_rel_err={max_rel_err:.3e}");
}

// ====================================================================
// Test 4: threshold/boundary stress on near-collinearity
// ====================================================================
#[test]
fn t04_epsilon_boundary_lowrank_to_full_transition() {
    let n = 10_000;
    let weights = ones_weights(n);
    let epsilons = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0_f64];
    let mut shapes_per_eps: Vec<(f64, MarginalSlopeCovarianceShape)> = Vec::new();
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let base = make_iid_normal_scores(n, 3, &mut state);
    let noise = make_iid_normal_scores(n, 1, &mut state); // unit-variance noise

    for &eps in &epsilons {
        let mut scores = base.clone();
        // col3 (idx 2) = col1 (idx 0) + eps * noise
        for i in 0..n {
            scores[[i, 2]] = scores[[i, 0]] + eps * noise[[i, 0]];
        }
        let shape = classify(scores.view(), &weights);
        eprintln!("eps={eps:e} shape={shape:?}");
        shapes_per_eps.push((eps, shape));
    }
    println!("t04 threshold sweep (col3 = col1 + eps*noise):");
    for (eps, shape) in &shapes_per_eps {
        println!("    eps={eps:>10.0e} -> {shape:?}");
    }

    let first = shapes_per_eps.first().unwrap();
    let last = shapes_per_eps.last().unwrap();
    assert_eq!(
        first.1,
        MarginalSlopeCovarianceShape::LowRank,
        "eps={} (1e-12) should be LowRank, got {:?}",
        first.0,
        first.1
    );
    assert_eq!(
        last.1,
        MarginalSlopeCovarianceShape::Full,
        "eps={} (1.0) should be Full, got {:?}",
        last.0,
        last.1
    );
}

// ====================================================================
// Test 5: weighted sensitivity — first half correlated, second half independent
// ====================================================================
#[test]
fn t05_weights_select_correlated_half() {
    let n = 200;
    let k = 2;
    let half = n / 2;
    let mut state = 0xCAFE_BABEu64;
    // First half: strongly correlated (col2 = col1 + tiny noise -> nearly collinear)
    // Second half: independent IID
    let raw = make_iid_normal_scores(n, k + 1, &mut state); // 3 cols: u, v, noise
    let mut scores = Array2::<f64>::zeros((n, k));
    for i in 0..half {
        let u = raw[[i, 0]];
        let noise = raw[[i, 2]];
        scores[[i, 0]] = u;
        scores[[i, 1]] = u + 1e-2 * noise; // highly but not perfectly collinear
    }
    for i in half..n {
        scores[[i, 0]] = raw[[i, 0]];
        scores[[i, 1]] = raw[[i, 1]]; // independent
    }

    // Heavy weight on correlated half
    let mut w_corr = Array1::<f64>::zeros(n);
    for i in 0..half {
        w_corr[i] = 0.9 / half as f64;
    }
    for i in half..n {
        w_corr[i] = 0.1 / (n - half) as f64;
    }

    // Heavy weight on independent half
    let mut w_indep = Array1::<f64>::zeros(n);
    for i in 0..half {
        w_indep[i] = 0.1 / half as f64;
    }
    for i in half..n {
        w_indep[i] = 0.9 / (n - half) as f64;
    }

    // Uniform
    let w_uniform = Array1::<f64>::from_elem(n, 1.0);

    let cov_corr = marginal_slope_covariance_from_scores(scores.view(), &w_corr).expect("cov corr");
    let cov_indep =
        marginal_slope_covariance_from_scores(scores.view(), &w_indep).expect("cov indep");
    let cov_uniform =
        marginal_slope_covariance_from_scores(scores.view(), &w_uniform).expect("cov uniform");

    println!(
        "t05 shapes: w_corr={:?}, w_indep={:?}, w_uniform={:?}",
        cov_corr.shape(),
        cov_indep.shape(),
        cov_uniform.shape()
    );

    // Compute the weighted off-diagonal / diag-max ratio explicitly to
    // verify "follows weighted covariance".
    // Use the quadratic form on v=[1,-1]/sqrt(2): this is small when col1
    // and col2 are highly correlated and ~variance otherwise.
    let v = [
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ];
    let q_corr = cov_corr.quadratic_form(&v).expect("qf corr");
    let q_indep = cov_indep.quadratic_form(&v).expect("qf indep");
    println!("t05 quadratic form on (1,-1)/√2: corr={q_corr:.3e}, indep={q_indep:.3e}");
    assert!(
        q_corr < q_indep,
        "weighting toward correlated rows should shrink Var(col1-col2): corr={q_corr:.3e}, indep={q_indep:.3e}"
    );
}

// ====================================================================
// Test 6: N < K and N == K degenerate
// ====================================================================
#[test]
fn t06a_n_less_than_k_must_err_or_lowrank() {
    let n = 3;
    let k = 4;
    let mut state = 0x1357_9BDFu64;
    let scores = make_iid_normal_scores(n, k, &mut state);
    let w = ones_weights(n);
    let result = marginal_slope_covariance_from_scores(scores.view(), &w);
    match result {
        Err(_) => {} // OK
        Ok(cov) => match cov {
            MarginalSlopeCovariance::Diagonal(_) => {
                // Diagonal is technically rank ≤ K, but with N=3 < K=4 and
                // generic scores it's vanishingly unlikely off-diagonals are
                // ≤ 1e-10. Allow but warn.
                println!("t06a: N=3 K=4 returned Diagonal (acceptable but surprising)");
            }
            MarginalSlopeCovariance::Full(_) => {
                panic!("N=3 < K=4 must NOT return Full (would be rank-deficient)");
            }
            MarginalSlopeCovariance::LowRank(factor) => {
                assert!(
                    factor.ncols() <= n,
                    "LowRank factor with N=3 must have rank ≤ 3, got {}",
                    factor.ncols()
                );
            }
        },
    }
}

#[test]
fn t06b_n_equal_k_must_err_or_lowrank() {
    let n = 4;
    let k = 4;
    let mut state = 0x2468_ACE0u64;
    let scores = make_iid_normal_scores(n, k, &mut state);
    let w = ones_weights(n);
    let result = marginal_slope_covariance_from_scores(scores.view(), &w);
    match result {
        Err(_) => {} // OK
        Ok(cov) => match cov {
            MarginalSlopeCovariance::Diagonal(_) => {
                println!("t06b: N=K=4 returned Diagonal (acceptable but surprising)");
            }
            MarginalSlopeCovariance::Full(_) => {
                panic!(
                    "N=K=4 must NOT return full-rank Full: sample covariance has rank ≤ N-1 = 3"
                );
            }
            MarginalSlopeCovariance::LowRank(factor) => {
                assert!(
                    factor.ncols() < k,
                    "LowRank factor with N=K=4 must have rank < 4, got {}",
                    factor.ncols()
                );
            }
        },
    }
}

// ====================================================================
// Test 7: zero-weight rows are ignored bit-equally
// ====================================================================
#[test]
fn t07_zero_weight_rows_ignored() {
    let n_real = 200;
    let n_garbage = 1000;
    let k = 3;
    let seeds = 20;
    for seed in 0..seeds {
        let mut state = 0x7777_AAAAu64.wrapping_add(seed as u64 * 0x9E37_79B9);
        let real = make_iid_normal_scores(n_real, k, &mut state);
        let mut garbage = make_iid_normal_scores(n_garbage, k, &mut state);
        // Make the garbage genuinely toxic: enormous values, NaN-avoiding.
        for i in 0..n_garbage {
            for j in 0..k {
                garbage[[i, j]] = 1e9 * garbage[[i, j]].signum() * (garbage[[i, j]].abs() + 1.0);
            }
        }
        let mut combined = Array2::<f64>::zeros((n_real + n_garbage, k));
        for i in 0..n_garbage {
            for j in 0..k {
                combined[[i, j]] = garbage[[i, j]];
            }
        }
        for i in 0..n_real {
            for j in 0..k {
                combined[[n_garbage + i, j]] = real[[i, j]];
            }
        }
        let mut w_combined = Array1::<f64>::zeros(n_real + n_garbage);
        for i in 0..n_real {
            w_combined[n_garbage + i] = 1.0;
        }
        let w_real = Array1::<f64>::from_elem(n_real, 1.0);

        let cov_combined = marginal_slope_covariance_from_scores(combined.view(), &w_combined)
            .expect("cov combined");
        let cov_real =
            marginal_slope_covariance_from_scores(real.view(), &w_real).expect("cov real");

        assert_eq!(
            cov_combined.shape(),
            cov_real.shape(),
            "seed {seed}: shape mismatch with zero-weighted garbage"
        );
        // Check quadratic form bit-tolerance on several v.
        let mut vstate = 0x9999_1111u64.wrapping_add(seed as u64);
        for _ in 0..6 {
            let (a, b) = standard_normal_pair(&mut vstate);
            let (c, _) = standard_normal_pair(&mut vstate);
            let v = [a, b, c];
            let q_combined = cov_combined.quadratic_form(&v).expect("qf combined");
            let q_real = cov_real.quadratic_form(&v).expect("qf real");
            let diff = (q_combined - q_real).abs();
            assert!(
                diff <= 1e-12,
                "seed {seed}: quadratic form differs by {diff:.3e} (combined={q_combined}, real={q_real}) on v={v:?}"
            );
        }
    }
}

// ====================================================================
// Test 8: all-zero weights must Err
// ====================================================================
#[test]
fn t08_all_zero_weights_must_err() {
    let n = 100;
    let k = 3;
    let mut state = 0x3333_5555u64;
    let scores = make_iid_normal_scores(n, k, &mut state);
    let w = Array1::<f64>::zeros(n);
    let result = marginal_slope_covariance_from_scores(scores.view(), &w);
    assert!(
        result.is_err(),
        "all-zero weights must Err, got Ok({:?})",
        result.as_ref().map(|c| c.shape())
    );
}

// ====================================================================
// Test 9: non-finite scores must Err
// ====================================================================
#[test]
fn t09a_nan_score_must_err() {
    let n = 100;
    let k = 3;
    let mut state = 0x8888_AAAAu64;
    let mut scores = make_iid_normal_scores(n, k, &mut state);
    scores[[37, 1]] = f64::NAN;
    let w = ones_weights(n);
    let result = marginal_slope_covariance_from_scores(scores.view(), &w);
    assert!(result.is_err(), "NaN score must Err");
}

#[test]
fn t09b_inf_score_must_err() {
    let n = 100;
    let k = 3;
    let mut state = 0xCCCC_EEEEu64;
    let mut scores = make_iid_normal_scores(n, k, &mut state);
    scores[[12, 2]] = f64::INFINITY;
    let w = ones_weights(n);
    let result = marginal_slope_covariance_from_scores(scores.view(), &w);
    assert!(result.is_err(), "+Inf score must Err");

    let mut scores2 = make_iid_normal_scores(n, k, &mut state);
    scores2[[55, 0]] = f64::NEG_INFINITY;
    let result2 = marginal_slope_covariance_from_scores(scores2.view(), &w);
    assert!(result2.is_err(), "-Inf score must Err");
}

// ====================================================================
// Test 10: K=1 always Diagonal (or Err on zero-variance)
// ====================================================================
#[test]
fn t10_k_eq_1_always_diagonal_or_err() {
    let n = 100;
    let cases = 20;
    let mut state = 0xFACE_F00Du64;
    for case in 0..cases {
        // Vary "shape" of the single column
        let mut col = Array2::<f64>::zeros((n, 1));
        match case % 5 {
            0 => {
                // IID positive variance
                for i in 0..n {
                    col[[i, 0]] = standard_normal_pair(&mut state).0;
                }
            }
            1 => {
                // All zeros — zero variance
                // (leave as zeros)
            }
            2 => {
                // Single non-zero row
                col[[case % n, 0]] = 1.0;
            }
            3 => {
                // Constant non-zero
                let c = 3.0 + (case as f64) * 0.1;
                for i in 0..n {
                    col[[i, 0]] = c;
                }
            }
            _ => {
                // Larger variance
                for i in 0..n {
                    col[[i, 0]] = 5.0 * standard_normal_pair(&mut state).0;
                }
            }
        }
        let w = ones_weights(n);
        let result = marginal_slope_covariance_from_scores(col.view(), &w);
        match result {
            Ok(cov) => match cov {
                MarginalSlopeCovariance::Diagonal(diag) => {
                    assert_eq!(
                        diag.len(),
                        1,
                        "case {case}: Diagonal must have length 1, got {}",
                        diag.len()
                    );
                }
                other => panic!(
                    "case {case}: K=1 must return Diagonal, got {:?}",
                    other.shape()
                ),
            },
            Err(_) => {
                // Err is also acceptable (e.g. degenerate zero variance).
            }
        }
    }
}
