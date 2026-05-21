// Hard reductions: multi-z marginal-slope code paths must collapse exactly to the
// scalar / lower-dimensional / structurally-equivalent cases. These tests are
// designed to surface drift between specialised K=1 / Diagonal / LowRank routes
// and the generic Full path. If any assertion fails, that is a real bug.

use gam::bernoulli_marginal_slope::{
    marginal_slope_covariance_from_scores, marginal_slope_preserving_scale,
    marginal_slope_probit_eta, MarginalSlopeCovariance, MarginalSlopeCovarianceShape,
};
use gam::probability::normal_cdf;
use gam::survival_marginal_slope::{
    survival_marginal_slope_vector_eta, survival_marginal_slope_vector_neglog,
    survival_marginal_slope_vector_scale,
};
use ndarray::{Array1, Array2};

// ------------------------------------------------------------------
// Inline deterministic PRNG: splitmix64-style 64-bit state mixer.
// We expose a small typed wrapper so every test gets reproducible draws.
// ------------------------------------------------------------------

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Standard splitmix64 finalizer.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        // 53 bits of randomness into a double.
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
    }

    /// Uniform in (lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Approximate standard normal via Box-Muller (deterministic from this PRNG).
    fn normal(&mut self) -> f64 {
        // Avoid log(0).
        let mut u1 = self.uniform();
        if u1 < 1e-300 {
            u1 = 1e-300;
        }
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        r * theta.cos()
    }

    fn usize_in(&mut self, lo: usize, hi_inclusive: usize) -> usize {
        let span = (hi_inclusive - lo + 1) as u64;
        lo + (self.next_u64() % span) as usize
    }
}

fn make_full_cov(rng: &mut SplitMix64, k: usize) -> Array2<f64> {
    // Sigma = A A^T + d I, A is k x k Gaussian, d > 0 ensures PSD-with-margin.
    let mut a = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            a[[i, j]] = rng.normal();
        }
    }
    let mut sigma = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for r in 0..k {
                s += a[[i, r]] * a[[j, r]];
            }
            sigma[[i, j]] = s;
        }
    }
    let jitter = 0.05 + 0.1 * rng.uniform();
    for i in 0..k {
        sigma[[i, i]] += jitter;
    }
    // Symmetrise exactly to suppress any FP asymmetry.
    for i in 0..k {
        for j in 0..i {
            let avg = 0.5 * (sigma[[i, j]] + sigma[[j, i]]);
            sigma[[i, j]] = avg;
            sigma[[j, i]] = avg;
        }
    }
    sigma
}

fn factor_to_full(factor: &Array2<f64>) -> Array2<f64> {
    let k = factor.nrows();
    let r = factor.ncols();
    let mut sigma = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for rr in 0..r {
                s += factor[[i, rr]] * factor[[j, rr]];
            }
            sigma[[i, j]] = s;
        }
    }
    // Force exact symmetry.
    for i in 0..k {
        for j in 0..i {
            let avg = 0.5 * (sigma[[i, j]] + sigma[[j, i]]);
            sigma[[i, j]] = avg;
            sigma[[j, i]] = avg;
        }
    }
    sigma
}

// ------------------------------------------------------------------
// Test 1: survival K=1 bitwise == scalar identity, 200 fixtures.
// scalar identity: eta = q * sqrt(1 + r^2) + r * z, with r = probit_scale*slope.
// We use Diagonal([1.0]) so Sigma_11 = 1.
// ------------------------------------------------------------------
#[test]
fn survival_k1_eta_bitwise_scalar_identity_200_fixtures() {
    let mut rng = SplitMix64::new(0xA11C_E5_5EED_u64 ^ 0x01);
    for trial in 0..200 {
        let q = rng.range(-3.0, 3.0);
        let z = [rng.normal()];
        let slope = [rng.range(-1.5, 1.5)];
        // Keep probit_scale strictly positive but otherwise free.
        let probit_scale = rng.range(0.05, 2.5);
        let covariance = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0]));
        let r = probit_scale * slope[0];
        let scalar = q * (1.0 + r * r).sqrt() + r * z[0];
        let got =
            survival_marginal_slope_vector_eta(q, &z, &slope, &covariance, probit_scale).expect(
                "survival K=1 eta",
            );
        assert_eq!(
            got.to_bits(),
            scalar.to_bits(),
            "trial {trial}: survival K=1 eta drifted from scalar identity (got={got:.17e}, scalar={scalar:.17e})"
        );
    }
}

// ------------------------------------------------------------------
// Test 2: bernoulli K=1 bitwise == scalar identity, 200 fixtures.
// ------------------------------------------------------------------
#[test]
fn bernoulli_k1_eta_bitwise_scalar_identity_200_fixtures() {
    let mut rng = SplitMix64::new(0xBEEF_D00D_CAFE_u64);
    for trial in 0..200 {
        let q = rng.range(-3.0, 3.0);
        let z = [rng.normal()];
        let slope = [rng.range(-1.5, 1.5)];
        let probit_scale = rng.range(0.05, 2.5);
        let covariance = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0]));
        let r = probit_scale * slope[0];
        let scalar = q * (1.0 + r * r).sqrt() + r * z[0];
        let got = marginal_slope_probit_eta(q, &z, &slope, &covariance, probit_scale)
            .expect("bernoulli K=1 eta");
        assert_eq!(
            got.to_bits(),
            scalar.to_bits(),
            "trial {trial}: bernoulli K=1 eta drifted from scalar identity (got={got:.17e}, scalar={scalar:.17e})"
        );
    }
}

// ------------------------------------------------------------------
// Test 3: bernoulli and survival eta agreement at K in 1..=6, full Sigma.
// Same q, z, slopes, covariance, probit_scale -> identical eta to <= 1e-15.
// We test all three covariance shapes.
// ------------------------------------------------------------------
#[test]
fn bernoulli_survival_eta_agreement_k_up_to_6_all_shapes() {
    let mut rng = SplitMix64::new(0xC0FE_C0DE_FACE_u64);
    let tol = 1e-15;
    for k in 1..=6 {
        for trial in 0..40 {
            let q = rng.range(-2.5, 2.5);
            let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
            let slopes: Vec<f64> = (0..k).map(|_| rng.range(-1.2, 1.2)).collect();
            let probit_scale = rng.range(0.1, 2.0);

            // Shape A: Full
            let sigma_full = make_full_cov(&mut rng, k);
            let cov_full = MarginalSlopeCovariance::Full(sigma_full.clone());

            // Shape B: Diagonal (just the diagonal of sigma_full)
            let cov_diag = MarginalSlopeCovariance::Diagonal(sigma_full.diag().to_owned());

            // Shape C: LowRank: random factor of width = min(k, rng-pick)
            let rank = rng.usize_in(1, k);
            let mut factor = Array2::<f64>::zeros((k, rank));
            for i in 0..k {
                for j in 0..rank {
                    factor[[i, j]] = rng.normal();
                }
            }
            let cov_low = MarginalSlopeCovariance::LowRank(factor);

            for cov in [&cov_full, &cov_diag, &cov_low] {
                let eb = marginal_slope_probit_eta(q, &z, &slopes, cov, probit_scale)
                    .expect("bernoulli eta");
                let es = survival_marginal_slope_vector_eta(q, &z, &slopes, cov, probit_scale)
                    .expect("survival eta");
                let diff = (eb - es).abs();
                assert!(
                    diff <= tol,
                    "k={k} trial={trial} shape={:?}: bernoulli vs survival eta disagree (bernoulli={eb:.17e}, survival={es:.17e}, diff={diff:.3e})",
                    cov.shape()
                );
            }
        }
    }
}

// ------------------------------------------------------------------
// Test 4: block-diagonal independence. Sigma = blockdiag(Sigma_A, Sigma_B),
// K_A=K_B=3, slopes_B = 0. Then:
//   scale(z, [slopes_A; 0], Sigma) == scale(z_A, slopes_A, Sigma_A).
//   eta(...) decomposition holds.
// 50 seeds.
// ------------------------------------------------------------------
#[test]
fn block_diagonal_independence_with_zero_slopes_in_block_b() {
    let mut rng = SplitMix64::new(0xB10C_D146_0007_u64);
    let tol_scale = 1e-14;
    let tol_eta = 1e-14;
    let ka = 3usize;
    let kb = 3usize;
    let k = ka + kb;
    for trial in 0..50 {
        let sigma_a = make_full_cov(&mut rng, ka);
        let sigma_b = make_full_cov(&mut rng, kb);
        let mut sigma = Array2::<f64>::zeros((k, k));
        for i in 0..ka {
            for j in 0..ka {
                sigma[[i, j]] = sigma_a[[i, j]];
            }
        }
        for i in 0..kb {
            for j in 0..kb {
                sigma[[ka + i, ka + j]] = sigma_b[[i, j]];
            }
        }
        let slopes_a: Vec<f64> = (0..ka).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut slopes = slopes_a.clone();
        slopes.extend(std::iter::repeat(0.0).take(kb));
        let z_a: Vec<f64> = (0..ka).map(|_| rng.normal()).collect();
        let z_b: Vec<f64> = (0..kb).map(|_| rng.normal()).collect();
        let mut z = z_a.clone();
        z.extend(z_b.iter().copied());
        let probit_scale = rng.range(0.2, 1.7);
        let q = rng.range(-2.0, 2.0);

        let cov_full = MarginalSlopeCovariance::Full(sigma);
        let cov_a = MarginalSlopeCovariance::Full(sigma_a);

        let scale_full =
            marginal_slope_preserving_scale(&slopes, &cov_full, probit_scale).expect("scale full");
        let scale_a =
            marginal_slope_preserving_scale(&slopes_a, &cov_a, probit_scale).expect("scale A");
        assert!(
            (scale_full - scale_a).abs() <= tol_scale,
            "trial {trial}: scale block-diag != scale_A (full={scale_full:.17e}, A={scale_a:.17e})"
        );

        let eta_full = marginal_slope_probit_eta(q, &z, &slopes, &cov_full, probit_scale)
            .expect("eta full");
        let eta_a =
            marginal_slope_probit_eta(q, &z_a, &slopes_a, &cov_a, probit_scale).expect("eta A");
        // Slopes for block B are zero, so the linear contribution from z_B is zero.
        // Thus eta_full should equal eta_A exactly (to within FP rounding).
        assert!(
            (eta_full - eta_a).abs() <= tol_eta,
            "trial {trial}: eta block-diag decomposition broken (full={eta_full:.17e}, A={eta_a:.17e})"
        );
    }
}

// ------------------------------------------------------------------
// Test 5: appending a zero-slope column must leave eta/scale invariant.
// 50 random seeds, K in {1..=5} -> K+1.
// ------------------------------------------------------------------
#[test]
fn zero_slope_extra_column_does_not_change_eta_or_scale() {
    let mut rng = SplitMix64::new(0x0EE5_7E57_u64);
    let tol = 1e-15;
    for trial in 0..50 {
        let k = rng.usize_in(1, 5);
        let sigma = make_full_cov(&mut rng, k);
        let slopes: Vec<f64> = (0..k).map(|_| rng.range(-1.5, 1.5)).collect();
        let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
        let probit_scale = rng.range(0.1, 2.0);
        let q = rng.range(-2.0, 2.0);

        // Extend by one column with slope_new=0; cross-covariances zero, diag > 0.
        let mut sigma_ext = Array2::<f64>::zeros((k + 1, k + 1));
        for i in 0..k {
            for j in 0..k {
                sigma_ext[[i, j]] = sigma[[i, j]];
            }
        }
        let diag_new = rng.range(0.1, 3.0);
        sigma_ext[[k, k]] = diag_new;
        // Cross-row/col are already zero from initial zeros().

        let mut slopes_ext = slopes.clone();
        slopes_ext.push(0.0);
        let mut z_ext = z.clone();
        z_ext.push(rng.normal()); // arbitrary; coefficient is zero so must not affect output

        let cov = MarginalSlopeCovariance::Full(sigma);
        let cov_ext = MarginalSlopeCovariance::Full(sigma_ext);

        let s_base = marginal_slope_preserving_scale(&slopes, &cov, probit_scale).expect("s base");
        let s_ext = marginal_slope_preserving_scale(&slopes_ext, &cov_ext, probit_scale)
            .expect("s ext");
        assert!(
            (s_base - s_ext).abs() <= tol,
            "trial {trial} k={k}: scale changed after appending zero-slope column (base={s_base:.17e}, ext={s_ext:.17e})"
        );

        let eta_base = marginal_slope_probit_eta(q, &z, &slopes, &cov, probit_scale)
            .expect("eta base");
        let eta_ext = marginal_slope_probit_eta(q, &z_ext, &slopes_ext, &cov_ext, probit_scale)
            .expect("eta ext");
        assert!(
            (eta_base - eta_ext).abs() <= tol,
            "trial {trial} k={k}: eta changed after appending zero-slope column (base={eta_base:.17e}, ext={eta_ext:.17e})"
        );
    }
}

// ------------------------------------------------------------------
// Test 6: LowRank(F) == Full(F F^T) for both scale and eta.
// K in 2..=6, rank in 1..=K, 100 seeds. Tol 1e-13.
// ------------------------------------------------------------------
#[test]
fn lowrank_matches_full_of_outer_product() {
    let mut rng = SplitMix64::new(0xF11A_B0B0_BABE_u64);
    let tol = 1e-13;
    for trial in 0..100 {
        let k = rng.usize_in(2, 6);
        let rank = rng.usize_in(1, k);
        let mut factor = Array2::<f64>::zeros((k, rank));
        for i in 0..k {
            for j in 0..rank {
                factor[[i, j]] = rng.normal();
            }
        }
        let sigma_full = factor_to_full(&factor);
        let cov_low = MarginalSlopeCovariance::LowRank(factor);
        let cov_full = MarginalSlopeCovariance::Full(sigma_full);

        let slopes: Vec<f64> = (0..k).map(|_| rng.range(-1.2, 1.2)).collect();
        let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
        let probit_scale = rng.range(0.2, 1.8);
        let q = rng.range(-2.0, 2.0);

        let s_low = survival_marginal_slope_vector_scale(&slopes, &cov_low, probit_scale)
            .expect("scale low");
        let s_full = survival_marginal_slope_vector_scale(&slopes, &cov_full, probit_scale)
            .expect("scale full");
        assert!(
            (s_low - s_full).abs() <= tol,
            "trial {trial} k={k} rank={rank}: LowRank vs Full scale mismatch (low={s_low:.17e}, full={s_full:.17e})"
        );

        let e_low = survival_marginal_slope_vector_eta(q, &z, &slopes, &cov_low, probit_scale)
            .expect("eta low");
        let e_full = survival_marginal_slope_vector_eta(q, &z, &slopes, &cov_full, probit_scale)
            .expect("eta full");
        assert!(
            (e_low - e_full).abs() <= tol,
            "trial {trial} k={k} rank={rank}: LowRank vs Full eta mismatch (low={e_low:.17e}, full={e_full:.17e})"
        );
    }
}

// ------------------------------------------------------------------
// Test 7: Diagonal(d) == Full(diag(d)) for scale and eta. Tol 1e-15.
// ------------------------------------------------------------------
#[test]
fn diagonal_matches_full_of_diag() {
    let mut rng = SplitMix64::new(0xD1A6_F1A6_u64);
    let tol = 1e-15;
    for trial in 0..50 {
        let k = rng.usize_in(1, 5);
        let diag: Vec<f64> = (0..k).map(|_| rng.range(0.05, 3.0)).collect();
        let mut sigma_full = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            sigma_full[[i, i]] = diag[i];
        }
        let cov_diag = MarginalSlopeCovariance::Diagonal(Array1::from(diag.clone()));
        let cov_full = MarginalSlopeCovariance::Full(sigma_full);

        let slopes: Vec<f64> = (0..k).map(|_| rng.range(-1.3, 1.3)).collect();
        let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
        let probit_scale = rng.range(0.1, 2.0);
        let q = rng.range(-2.0, 2.0);

        let sd = survival_marginal_slope_vector_scale(&slopes, &cov_diag, probit_scale)
            .expect("scale diag");
        let sf = survival_marginal_slope_vector_scale(&slopes, &cov_full, probit_scale)
            .expect("scale full");
        assert!(
            (sd - sf).abs() <= tol,
            "trial {trial} k={k}: Diagonal vs Full scale mismatch (diag={sd:.17e}, full={sf:.17e})"
        );

        let ed = survival_marginal_slope_vector_eta(q, &z, &slopes, &cov_diag, probit_scale)
            .expect("eta diag");
        let ef = survival_marginal_slope_vector_eta(q, &z, &slopes, &cov_full, probit_scale)
            .expect("eta full");
        assert!(
            (ed - ef).abs() <= tol,
            "trial {trial} k={k}: Diagonal vs Full eta mismatch (diag={ed:.17e}, full={ef:.17e})"
        );
    }
}

// ------------------------------------------------------------------
// Test 8: marginal_slope_covariance_from_scores reductions:
//   (a) two columns with col2 = alpha * col1 exactly -> LowRank (rank 1)
//   (b) three exactly-orthogonal scaled columns      -> Diagonal
// We check `.shape()` only (numerical content tested by other tests).
// ------------------------------------------------------------------
#[test]
fn auto_derivation_shape_reductions() {
    // (a) Perfect collinearity -> LowRank
    let n = 64;
    let mut col1 = Array1::<f64>::zeros(n);
    let mut rng = SplitMix64::new(0x5C0F_E5_u64);
    for i in 0..n {
        col1[i] = rng.normal();
    }
    let alpha = 1.7_f64;
    let mut scores2 = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        scores2[[i, 0]] = col1[i];
        scores2[[i, 1]] = alpha * col1[i];
    }
    let weights = Array1::<f64>::from(vec![1.0; n]);
    let cov_collinear = marginal_slope_covariance_from_scores(scores2.view(), &weights)
        .expect("from_scores collinear");
    assert_eq!(
        cov_collinear.shape(),
        MarginalSlopeCovarianceShape::LowRank,
        "collinear K=2 case should auto-detect LowRank (rank-1), got {:?}",
        cov_collinear.shape()
    );
    if let MarginalSlopeCovariance::LowRank(factor) = &cov_collinear {
        assert_eq!(
            factor.ncols(),
            1,
            "rank-1 collinear case should produce a single-column factor, got {} cols",
            factor.ncols()
        );
    } else {
        panic!("unreachable: shape check passed but enum variant did not");
    }

    // (b) Orthogonal scaled indicator columns -> Diagonal.
    // Take three "scaled selector" columns whose row supports do not overlap;
    // their sample cross-products are exactly zero, so off-diagonal entries
    // vanish and the auto-derivation must collapse to Diagonal.
    let n = 9;
    let mut scores3 = Array2::<f64>::zeros((n, 3));
    // First three rows feed column 0 only; next three rows feed column 1; etc.
    for i in 0..3 {
        scores3[[i, 0]] = 1.0 + (i as f64) * 0.1;
    }
    for i in 0..3 {
        scores3[[3 + i, 1]] = 2.0 - (i as f64) * 0.2;
    }
    for i in 0..3 {
        scores3[[6 + i, 2]] = -0.7 + (i as f64) * 0.05;
    }
    let weights3 = Array1::<f64>::from(vec![1.0; n]);
    let cov_orth = marginal_slope_covariance_from_scores(scores3.view(), &weights3)
        .expect("from_scores orthogonal");
    assert_eq!(
        cov_orth.shape(),
        MarginalSlopeCovarianceShape::Diagonal,
        "K=3 orthogonal scaled columns should auto-detect Diagonal, got {:?}",
        cov_orth.shape()
    );
}

// ------------------------------------------------------------------
// Test 9: survival_marginal_slope_vector_neglog reduction to closed-form at K=1.
// Closed form (matches the comment in src):
//   c   = sqrt(1 + r^2 * sigma) where r = probit_scale*slope; here sigma=1.
//   eta0 = q0 * c + r * z;  eta1 = q1 * c + r * z.
//   ad1  = qd1 * c
//   ell  = w * [ (1 - d) * (-log Phi(-eta1)) + log Phi(-eta0)
//                - d * log phi(eta1) - d * log(ad1) ]
// 20 random fixtures, tol 1e-14.
// ------------------------------------------------------------------
#[test]
fn survival_neglog_k1_matches_closed_form_20_fixtures() {
    let mut rng = SplitMix64::new(0x5117_E1E1_u64);
    let tol = 1e-14;
    for trial in 0..20 {
        let q0 = rng.range(-2.0, 2.0);
        let q1 = q0 + rng.range(0.05, 1.5); // q1 > q0 ensured (matches monotone time)
        let qd1 = rng.range(0.1, 2.5); // > 0
        let z = [rng.normal()];
        let slope = [rng.range(-1.2, 1.2)];
        let weight = rng.range(0.3, 2.0);
        // Event indicator: alternate between 0 and 1 deterministically.
        let event = if trial % 2 == 0 { 0.0 } else { 1.0 };
        let probit_scale = rng.range(0.1, 2.0);
        let cov = MarginalSlopeCovariance::Diagonal(Array1::from(vec![1.0]));
        let derivative_guard = 1e-12;

        let got = survival_marginal_slope_vector_neglog(
            q0,
            q1,
            qd1,
            &slope,
            &z,
            &cov,
            weight,
            event,
            derivative_guard,
            probit_scale,
        )
        .expect("survival vector neglog K=1");

        // Hand-computed closed form.
        let r = probit_scale * slope[0];
        let c = (1.0 + r * r).sqrt();
        let eta0 = q0 * c + r * z[0];
        let eta1 = q1 * c + r * z[0];
        let log_cdf_neg_eta0 = normal_cdf(-eta0).ln();
        let log_cdf_neg_eta1 = normal_cdf(-eta1).ln();
        let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
        let ad1 = qd1 * c;
        let expected = weight
            * ((1.0 - event) * (-log_cdf_neg_eta1) + log_cdf_neg_eta0
                - event * log_phi_eta1
                - event * ad1.ln());
        let diff = (got - expected).abs();
        let scale = 1.0 + got.abs().max(expected.abs());
        assert!(
            diff <= tol * scale,
            "trial {trial}: neglog K=1 diverged from closed form (got={got:.17e}, expected={expected:.17e}, diff={diff:.3e})"
        );
    }
}
