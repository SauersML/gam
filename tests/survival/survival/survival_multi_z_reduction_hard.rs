// Hard reductions: multi-z marginal-slope code paths must collapse exactly to the
// scalar / lower-dimensional / structurally-equivalent cases. These tests are
// designed to surface drift between specialised K=1 / Diagonal / LowRank routes
// and the generic Full path. If any assertion fails, that is a real bug.

use gam::families::bms::{MarginalSlopeCovariance, MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores};
use gam::families::survival::marginal_slope::{RigidVectorValueWorkspace, survival_marginal_slope_vector_neglog};
use gam::probability::normal_cdf;
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

}

// ------------------------------------------------------------------
// Scalar-reduction comparison.
//
// The K=1 reduction is an identity over the reals, not over IEEE-754.
// `marginal_slope_preserving_scale` squares the probit scale once and applies
// it to the quadratic form of the *raw* slopes, and the diagonal quadratic
// form accumulates `coefficient * slope * slope`, so production evaluates
// `fl(fl(p*p) * fl(s*s))`. The scalar identity below folds the scale into the
// slope first and evaluates `fl(fl(p*s) * fl(p*s))`. Those are the same real
// number and differ by up to an ulp, so a `to_bits()` assertion pins the
// association order rather than the reduction. The production representation
// policy is deliberate, so the reduction is pinned to a few-ulp bound.
//
// `magnitude` is the size of the largest intermediates the reference sums, so
// a cancelling total is still held to the accuracy its inputs allow rather
// than to an unachievable relative bound on a near-zero result.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 1: survival K=1 == scalar identity to a few ulps, 200 fixtures.
// scalar identity: eta = q * sqrt(1 + r^2) + r * z, with r = probit_scale*slope.
// We use Diagonal([1.0]) so Sigma_11 = 1.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 2: bernoulli K=1 == scalar identity to a few ulps, 200 fixtures.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 3: bernoulli and survival eta agreement at K in 1..=6, full Sigma.
// Same q, z, slopes, covariance, probit_scale -> identical eta to <= 1e-15.
// We test all three covariance shapes.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 4: block-diagonal independence. Sigma = blockdiag(Sigma_A, Sigma_B),
// K_A=K_B=3, slopes_B = 0. Then:
//   scale(z, [slopes_A; 0], Sigma) == scale(z_A, slopes_A, Sigma_A).
//   eta(...) decomposition holds.
// 50 seeds.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 5: appending a zero-slope column must leave eta/scale invariant.
// 50 random seeds, K in {1..=5} -> K+1.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 6: LowRank(F) == Full(F F^T) for both scale and eta.
// K in 2..=6, rank in 1..=K, 100 seeds. Tol 1e-13.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 7: Diagonal(d) == Full(diag(d)) for scale and eta. Tol 1e-15.
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Test 8: marginal_slope_covariance_from_scores reductions:
//   (a) two columns with col2 = alpha * col1 exactly -> Full, retaining coupling
//   (b) three exactly-orthogonal scaled columns      -> Diagonal
// We check `.shape()` only (numerical content tested by other tests).
// ------------------------------------------------------------------
#[test]
fn auto_derivation_shape_reductions() {
    // (a) Perfect collinearity has a nonzero off-diagonal, so it remains Full.
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
        MarginalSlopeCovarianceShape::Full,
        "collinear K=2 case must retain exact coupling as Full, got {:?}",
        cov_collinear.shape()
    );
    assert_ne!(cov_collinear.to_dense()[[0, 1]], 0.0);

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
        let cov = MarginalSlopeCovariance::diagonal(Array1::from(vec![1.0])).unwrap();
        let derivative_guard = 1e-12;

        let got = survival_marginal_slope_vector_neglog(
            0,
            q0,
            q1,
            qd1,
            &slope,
            &z,
            &RigidVectorValueWorkspace::new(&cov.clone().into()),
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
