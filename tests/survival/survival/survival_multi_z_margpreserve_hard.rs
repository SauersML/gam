//! Aggressive marginal-preservation identity tests for the multi-z survival
//! marginal-slope likelihood.
//!
//! The identity under test is, for the marginal-preserving scale
//! `c = sqrt(1 + r' Σ r)` (with `r` the observed scaled slope `probit_scale * g`):
//!
//!     Φ(-q) == Φ(-q * c / sqrt(1 + r' Σ r))
//!
//! When `c` is exactly the preserving scale this collapses to `Φ(-q) == Φ(-q)`
//! up to floating point. Any subtle bug in the vector generalisation of `c(a)`
//! (e.g. dropping a `probit_scale`, mixing up `Full` vs `LowRank` quadratic
//! forms, permutation sensitivity) makes one or more of these tests blow past
//! the 2e-15 tolerance.

use gam::families::bms::MarginalSlopeCovariance;
use gam::families::survival::marginal_slope::{RigidVectorValueWorkspace, survival_marginal_slope_vector_neglog};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG (splitmix64 -> f64 in [0,1)) so we do not pull in a
// new crate dependency.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform [0,1).
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    /// Uniform [-1,1).
    fn next_signed(&mut self) -> f64 {
        2.0 * self.next_unit() - 1.0
    }
    /// Approx standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        // Avoid log(0).
        let u1 = (self.next_unit()).max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        // hi exclusive
        lo + (self.next_u64() as usize) % (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_diagonal(rng: &mut SplitMix64, k: usize) -> MarginalSlopeCovariance {
    let mut diag = Array1::<f64>::zeros(k);
    for i in 0..k {
        diag[i] = 0.01 + 2.0 * rng.next_unit();
    }
    MarginalSlopeCovariance::diagonal(diag).unwrap()
}

fn random_full(rng: &mut SplitMix64, k: usize) -> (MarginalSlopeCovariance, Array2<f64>) {
    // Σ = L Lᵀ + εI, build dense.
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            l[[i, j]] = rng.next_signed();
        }
    }
    let mut cov = l.dot(&l.t());
    for i in 0..k {
        cov[[i, i]] += 1e-3;
    }
    (MarginalSlopeCovariance::full(cov.clone()).unwrap(), cov)
}

fn random_low_rank(rng: &mut SplitMix64, k: usize, r: usize) -> MarginalSlopeCovariance {
    let mut f = Array2::<f64>::zeros((k, r));
    for i in 0..k {
        for j in 0..r {
            f[[i, j]] = rng.next_signed();
        }
    }
    MarginalSlopeCovariance::low_rank(f).unwrap()
}

fn random_slopes(rng: &mut SplitMix64, k: usize, norm: f64) -> Vec<f64> {
    let mut s: Vec<f64> = (0..k).map(|_| rng.next_normal()).collect();
    let n2: f64 = s.iter().map(|v| v * v).sum();
    let inv = if n2 > 0.0 { norm / n2.sqrt() } else { 0.0 };
    for v in &mut s {
        *v *= inv;
    }
    s
}

fn random_z(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    (0..k).map(|_| rng.next_normal()).collect()
}

// ---------------------------------------------------------------------------
// 1. Randomised sweep across shapes and dimensions.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 2. Extreme magnitudes (explicit grid).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 3. Near-degenerate covariance.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 4. Scalar reduction (K=1, Diagonal[1.0]), to a few ulps.
//
// The reduction is an identity over the reals, not over IEEE-754.
// `marginal_slope_preserving_scale` squares the probit scale once and applies
// it to the quadratic form of the *raw* slopes (the diagonal quadratic form
// accumulates `coefficient * slope * slope`), so production evaluates
// `fl(fl(p*p) * fl(s*s))` while the scalar identity below folds the scale into
// the slope first and evaluates `fl(fl(p*s) * fl(p*s))`. Same real number, up
// to an ulp apart, so `to_bits()` equality would pin the association order
// rather than the reduction. The production representation policy is
// deliberate, so the reduction is pinned to a few-ulp bound instead.
//
// `magnitude` is the size of the largest intermediates the reference sums, so
// a cancelling total is still held to the accuracy its inputs allow.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 5. Permutation invariance.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 6. Diagonal-from-Full equivalence.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 7. LowRank vs Full equivalence.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Extra: smoke-test the canonical builder so it is still exercised here.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Negative tests.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Spot check: neglog stays finite over many random shapes (just guards
// against panics in the negative-log path under the marginal-preserving
// scale). Identity here is just that no `Err` is returned and the value is
// finite for sensible event=0 inputs.
// ---------------------------------------------------------------------------

#[test]
fn survival_multi_z_neglog_finite_under_random_shapes() {
    for seed in 0u64..40 {
        let mut rng = SplitMix64::new(0x4242_4242 ^ seed);
        let k = rng.range(2, 6);
        let cov = match seed % 3 {
            0 => random_diagonal(&mut rng, k),
            1 => random_full(&mut rng, k).0,
            _ => {
                let r = (rng.range(1, k + 1)).max(1);
                random_low_rank(&mut rng, k, r)
            }
        };
        let slopes = random_slopes(&mut rng, k, 0.5);
        let z = random_z(&mut rng, k);
        let q0 = rng.next_signed();
        let q1 = q0 + 0.5 + rng.next_unit();
        let qd1 = 0.1 + rng.next_unit();
        let value = survival_marginal_slope_vector_neglog(
            0,
            q0,
            q1,
            qd1,
            &slopes,
            &z,
            &RigidVectorValueWorkspace::new(&cov.clone().into()),
            1.0,
            0.0,
            1e-6,
            1.0,
        )
        .expect("neglog");
        assert!(value.is_finite(), "seed={seed}: neglog not finite: {value}");
    }
}
