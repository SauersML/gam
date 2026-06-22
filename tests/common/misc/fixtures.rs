//! Common inline PRNG (`Splitmix64`) shared by survival-marginal-slope
//! tests.  Each test that needs the RNG includes this file directly via a
//! `#[path = "common/fixtures.rs"]` attribute; per-test helpers that only a
//! single binary uses (e.g. central-difference Jacobian, paired Box-Muller)
//! live inside that binary rather than here, so each integration test
//! compiles without per-binary `dead_code` noise under the workspace's
//! `warnings = "deny"` lint policy.

/// Deterministic, no-allocation PRNG backed by splitmix64.
///
/// Construct with a seed and call `next_unit` / `next_gauss` to draw samples.
pub struct Splitmix64 {
    state: u64,
}

impl Splitmix64 {
    /// Create a new RNG with the given 64-bit seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return the raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)` (53-bit mantissa).
    pub fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0_f64 / ((1u64 << 53) as f64))
    }

    /// Standard normal sample via Box-Muller.
    pub fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_unit().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        r * theta.cos()
    }
}
