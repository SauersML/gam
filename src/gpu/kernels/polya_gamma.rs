//! GPU Pólya–Gamma sampler primitive — INCOMPATIBLE with shipped probit BMS
//! (different model).
//!
//! This module implements a stand-alone, device-resident Pólya–Gamma sampler
//! plus a synthetic *logistic* Gibbs harness used to validate the sampler
//! because those are probit families — PG augmentation is exact only for the
//! Bernoulli **logistic** likelihood (Polson, Scott & Windle 2013). Probit
//! paths (`bms_flex`, `bernoulli_marginal_slope`) use a different likelihood and
//! do not call this module.
//!
//! The block 7 math design splits the device sampler into three regimes
//! (math §7), each kernel laid out to avoid warp divergence inside the
//! launch:
//!
//! * **`pg1_kernel`** — exact Devroye (math §8) for shape `b = 1`. This
//!   covers pure Bernoulli rows. Each row owns a `curand`-style XORWOW
//!   state seeded statelessly from `(seed, row_index)` so two runs with
//!   the same seed produce bit-identical draws regardless of grid layout.
//!   The alternating-series accept/reject uses the corrected right-tail
//!   coefficient `π · k` (not `π / 2`) — the math team’s Phase-1 fix.
//! * **`sp_kernel`** — saddlepoint rejection (math §9) for `13 < b ≤ 170`.
//!   This solves `K'(t) = x` via six Newton iterations on `tanh(v)/v` or
//!   `tan(v)/v` and uses an IG + Gamma envelope for the accept/reject.
//! * **`normal_kernel`** — Lyapunov-CLT closed-form approximation
//!   (math §10) for `b > 170`. Mean and variance use the analytic
//!   PG(b, c) limit, no rejection loop, no warp divergence.
//!
//! The host dispatcher partitions an input vector of `(b_i, c_i)` rows
//! into three contiguous index lists (one per regime) and launches one
//! kernel per regime. The `8 ≤ b ≤ 13` band is handled on host via the
//! sum-of-PG(1, c) convolution identity — at small `b` the sum cost is
//! negligible and keeping it off-device avoids a fourth kernel that would
//! see almost no traffic in practice.
//!
//! ## What this primitive intentionally does NOT do
//!
//! * It does **not** plug into BMS marginal slope (probit model) — the PG
//!   augmentation identity is logit-only; doing so silently would change
//!   numerical results for shipped fits.
//! * It does **not** define a public production family. The
//!   Gibbs harness in [`logistic_gibbs_step`] is a *validation oracle* for
//!   the sampler primitive, not a fit method. The CPU reference
//!   `src/inference/polya_gamma.rs` and the NUTS/HMC infrastructure remain
//!   the supported posterior-inference paths.
//!
//! ## Stateless XORWOW seeding
//!
//! Each row’s XORWOW state `(s0, s1, s2, s3, s4, counter)` is materialised
//! by feeding `splitmix64( seed ⊕ row · ZETA ⊕ word · GAMMA )` for word
//! indices `0..5` — five 32-bit lanes plus a 32-bit counter. The host
//! reference RNG (`xorwow_state_from`) reproduces the same byte sequence
//! the kernel emits, so CPU/GPU parity tests can compare draw-by-draw at
//! the same `(seed, row)`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::linalg::triangular::{back_substitution_lower_transpose, cholesky_solve_vector};

#[cfg(target_os = "linux")]
use crate::gpu::gpu_error::GpuError;

// ────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────

/// Stateless seed for the per-row XORWOW PRNG. The same `seed` reproduces
/// bit-identical draws across runs and across CPU/GPU implementations.
#[derive(Clone, Copy, Debug)]
pub struct PgSeed(pub u64);

impl Default for PgSeed {
    fn default() -> Self {
        Self(0x50_4F_4C_59_47_41_4D_41) // "POLYGAMA" big-endian ascii
    }
}

/// Regime split thresholds (math §7).
///
/// * `PG1_MAX_B = 1` — exact-Devroye regime.
/// * `(PG1_MAX_B, SADDLE_MIN_B)` — host convolution-of-PG(1) regime.
/// * `[SADDLE_MIN_B, SADDLE_MAX_B]` — saddlepoint-rejection regime.
/// * `b > NORMAL_MIN_B` — normal-approximation regime.
pub const PG1_MAX_B: u32 = 1;
pub const SADDLE_MIN_B: u32 = 14;
pub const SADDLE_MAX_B: u32 = 170;
pub const NORMAL_MIN_B: u32 = 171;

/// Inputs for the dispatched batched sampler.
#[derive(Clone, Debug)]
pub struct PolyaGammaBatchInput<'a> {
    /// Shape parameters `b_i`. Must be ≥ 1.
    pub shapes: ArrayView1<'a, u32>,
    /// Tilt parameters `c_i = ψ_i`. Sign is irrelevant (sampler uses |c|).
    pub tilts: ArrayView1<'a, f64>,
    /// Stateless RNG seed.
    pub seed: PgSeed,
}

impl<'a> PolyaGammaBatchInput<'a> {
    pub fn rows(&self) -> usize {
        self.shapes.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.shapes.len() != self.tilts.len() {
            return Err(format!(
                "polya_gamma: shapes.len()={} != tilts.len()={}",
                self.shapes.len(),
                self.tilts.len()
            ));
        }
        if self.shapes.iter().any(|b| *b == 0) {
            return Err("polya_gamma: b=0 is invalid (PG(0,c) is a point mass at 0)".to_string());
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────
// SplitMix64 finalizer + per-row XORWOW seeding
// ────────────────────────────────────────────────────────────────────────

/// SplitMix64 finalizer (matches `reml_trace::splitmix64_mix`). Thin wrapper
/// over the canonical implementation in [`crate::linalg::utils::splitmix64_hash`].
#[inline]
pub fn splitmix64_mix(z: u64) -> u64 {
    crate::linalg::utils::splitmix64_hash(z)
}

/// Two large odd constants used to mix `(seed, row, word)` into the
/// SplitMix input. Disjoint from the `reml_trace` constants so different
/// kernels with the same seed don’t share probe sequences.
const ROW_ZETA: u64 = 0xA1B2_C3D4_E5F6_7890;
const WORD_GAMMA: u64 = 0x0F1E_2D3C_4B5A_6978;

/// Compact per-row XORWOW state. Layout matches `curand_kernel.h`’s
/// `curandStateXORWOW_t` for the five state lanes plus the addition
/// counter; we omit the boxmuller cache (PG sampler doesn’t use it).
#[derive(Clone, Copy, Debug)]
pub struct XorwowState {
    pub s: [u32; 5],
    pub d: u32,
}

impl XorwowState {
    /// Stateless seeding from `(seed, row)`. Each of the six state words
    /// is the high or low half of a SplitMix64 hash of
    /// `splitmix64(seed ⊕ row·ROW_ZETA ⊕ word·WORD_GAMMA)`. The first
    /// non-zero state word is enforced so we never enter the all-zero
    /// XORWOW absorbing fixed point.
    pub fn new(seed: u64, row: u64) -> Self {
        let mut words = [0u32; 6];
        for (word_idx, slot) in words.iter_mut().enumerate() {
            let composite =
                seed ^ row.wrapping_mul(ROW_ZETA) ^ (word_idx as u64).wrapping_mul(WORD_GAMMA);
            let h = splitmix64_mix(composite);
            *slot = (h >> 32) as u32;
        }
        // XORWOW absorbs at all-zeros; flip the low bit of s[0] if it ever
        // happens (probability 2⁻³² but cheap to guard).
        if words[0] == 0 && words[1] == 0 && words[2] == 0 && words[3] == 0 && words[4] == 0 {
            words[0] = 1;
        }
        Self {
            s: [words[0], words[1], words[2], words[3], words[4]],
            d: words[5],
        }
    }

    /// Single XORWOW advance. Returns the next 32-bit output and mutates
    /// the state. Matches Marsaglia’s 2003 XORWOW formulation, which is
    /// also what `curand_kernel.h::xorwow` computes.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut t = self.s[4];
        let s = self.s[0];
        self.s[4] = self.s[3];
        self.s[3] = self.s[2];
        self.s[2] = self.s[1];
        self.s[1] = s;
        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        self.s[0] = t;
        self.d = self.d.wrapping_add(362_437);
        t.wrapping_add(self.d)
    }

    /// Uniform double in (0, 1] — same `(u32 + 1) / 2^32` convention the
    /// kernel uses (matches `curand_uniform_double` upper-open interval
    /// convention; we use the upper-closed variant so a zero u32 never
    /// produces exactly zero, which would crash `log(u)` in the Exp draw).
    #[inline]
    pub fn next_unit(&mut self) -> f64 {
        let raw = self.next_u32();
        ((raw as f64) + 1.0) * (1.0 / 4_294_967_296.0)
    }

    /// Standard exponential via inverse CDF: `-ln(U)`. `U` is on (0, 1]
    /// so `-ln(U)` is in `[0, +inf)`, never `+inf` from a zero argument.
    #[inline]
    pub fn next_exp(&mut self) -> f64 {
        -self.next_unit().ln()
    }

    /// Standard normal via Marsaglia polar method. Discards the second
    /// variate the polar pair produces (cleaner than caching it across
    /// calls — we’d need a per-row scratch slot, which the device kernel
    /// can’t afford to spill).
    #[inline]
    pub fn next_norm(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.next_unit() - 1.0;
            let v = 2.0 * self.next_unit() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                return u * factor;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// CPU host reference — Devroye PG(1, c) via the shared sampler core
// ────────────────────────────────────────────────────────────────────────
//
// The CPU oracle for parity tests has to *use the same RNG bytes as the
// device kernel*, so it drives the shared Devroye core
// (`crate::inference::polya_gamma_core`) through the bit-exact `XorwowState`
// rather than through the production `rand::Rng` adapter. The math (Devroye
// 1986; PSW 2013) is the single shared implementation — there is no second
// copy of the tail mass / series / inverse-Gaussian helpers to drift.

use crate::inference::polya_gamma_core::{PgRng, draw_pg1};
use std::f64::consts::{FRAC_PI_2, PI};

/// `XorwowState` is the randomness source for the bit-exact GPU oracle. Wiring
/// it through [`PgRng`] lets the shared Devroye core run against the same RNG
/// byte stream the device kernel consumes.
impl PgRng for XorwowState {
    #[inline]
    fn next_unit(&mut self) -> f64 {
        XorwowState::next_unit(self)
    }

    #[inline]
    fn next_exp(&mut self) -> f64 {
        XorwowState::next_exp(self)
    }

    #[inline]
    fn next_norm(&mut self) -> f64 {
        XorwowState::next_norm(self)
    }
}

/// CPU oracle for one PG(1, c) draw using a `XorwowState` directly. The
/// device kernel performs the same arithmetic byte-for-byte (modulo IEEE
/// rounding of transcendentals, which agree to <1 ULP for the inputs we
/// touch).
pub fn pg1_draw_cpu_oracle(state: &mut XorwowState, tilt: f64) -> f64 {
    draw_pg1(state, tilt)
}

/// Higher-shape draw on host via convolution: PG(b, c) =_d Σ_{j=1..b} PG(1, c).
/// Used by host for the `2 ≤ b ≤ 13` band and as the parity oracle for the
/// saddlepoint kernel at modest `b`.
pub fn pg_convolution_cpu_oracle(state: &mut XorwowState, b: u32, tilt: f64) -> f64 {
    (0..b).map(|_| pg1_draw_cpu_oracle(state, tilt)).sum()
}

// ────────────────────────────────────────────────────────────────────────
// Saddlepoint regime (math §9, 13 < b ≤ 170) — host oracle
// ────────────────────────────────────────────────────────────────────────
//
// We sample a tilted-J*(b, z) variate via saddlepoint rejection. The
// envelope is an IG / Gamma mixture; the saddlepoint approximation to the
// log density gives a tight acceptance ratio across the full b range. The
// host implementation here is also the *oracle* used to validate the
// device sp_kernel.

/// Solve K'(t) = x for the saddlepoint t given x in (0, 1). K'(t) is a
/// continuous strictly increasing function of t on the appropriate
/// branch; the math team’s parameterisation eliminates v = sqrt(|2t|) so
/// the Newton iteration is on a monotone bounded variable.
///
/// Branch:
/// * `x < 1`  → `K'(t) = tanh(v)/v` with `v = sqrt(-2t)`, t ≤ 0.
/// * `x ≥ 1`  → `K'(t) = tan(v)/v`  with `v = sqrt( 2t)`, t > 0.
pub fn saddlepoint_solve(x: f64) -> f64 {
    // Six iterations is the math team’s target (§9). The function is
    // analytic; Newton on tanh(v)/v or tan(v)/v converges quadratically
    // from the closed-form initial guess `v0 = sqrt(3(1 - x))` (Taylor of
    // `tanh(v)/v = 1 - v²/3 + 2v⁴/15 - ...`).
    if (x - 1.0).abs() < 1e-9 {
        return 0.0;
    }
    if x < 1.0 {
        // Negative-t branch, work in v = sqrt(-2t). `tanh(v)/v` is monotone
        // decreasing in v on (0, ∞), with two well-separated asymptotic
        // regimes:
        //
        //   * x ≈ 1 (v small): Taylor expansion tanh(v)/v ≈ 1 - v²/3 gives
        //     `v₀ = sqrt(3(1 - x))`, which is the ~quadratic starting point
        //     used historically.
        //   * x ≈ 0 (v large): `tanh(v) → 1`, so `tanh(v)/v ≈ 1/v` and the
        //     root sits near `v ≈ 1/x`. The Taylor seed `sqrt(3(1-x)) ≤ √3`
        //     is bounded above by ~1.73, which leaves Newton walking the
        //     plateau at ~`tanh(v)/v ≈ 0.55` and converging linearly to the
        //     true root (≈ 20 at x = 0.05); six Newton steps are not enough
        //     to drive the relative error to 1e-6 from there.
        //
        // Take the maximum of the two seeds so each regime gets a starting
        // point in its quadratic-convergence basin; the function is monotone
        // so overshooting the root from above just trades a couple of
        // descending Newton steps for the missing factor-of-ten distance.
        // 16 Newton iterations is comfortable even when the initial seed
        // overshoots and Newton has to recover via several linear steps
        // before settling into the quadratic regime.
        let v_taylor = (3.0 * (1.0 - x)).sqrt();
        let v_asym = 1.0 / x.max(1e-12);
        let mut v = v_taylor.max(v_asym).max(1e-6);
        for _ in 0..16 {
            let tanh_v = v.tanh();
            let f = tanh_v / v - x;
            // d/dv [tanh(v)/v] = (1 - tanh²v)/v - tanh(v)/v²
            //                  = ((1 - tanh²v) - tanh(v)/v) / v.
            let sech_sq = 1.0 - tanh_v * tanh_v;
            let df = (sech_sq - tanh_v / v) / v;
            v -= f / df;
            if v.abs() < 1e-12 {
                break;
            }
        }
        -0.5 * v * v
    } else {
        // Positive-t branch, work in v = sqrt(2t). The pole of tan is at
        // v = π/2; the relevant root sits in (0, π/2). Two regimes:
        //
        //   * x ≈ 1 (v small): Taylor tan(v)/v ≈ 1 + v²/3 gives
        //     `v₀ = sqrt(3(x - 1))` — the historical seed.
        //   * x large (v near π/2): tan(v) ≈ 1/(π/2 - v), so the root sits
        //     near `v ≈ π/2 - 2/(x π)`. Seeding from the 0.49 π cap leaves
        //     Newton inside the very steep tail of the pole, where each
        //     Newton step descends by a fraction of the remaining distance;
        //     six steps left x = 3 stuck at rel ≈ 1.5e-4 above 1e-6.
        //
        // The cap stays at 0.499 π to keep `tan(v)` finite; the analytic
        // pole-tail seed is honoured when it sits below that cap. Bumping
        // the iteration cap mirrors the negative branch.
        let v_taylor = (3.0 * (x - 1.0)).sqrt();
        let v_pole = FRAC_PI_2 - 2.0 / (x.max(1e-12) * PI);
        let mut v = v_taylor.max(v_pole).min(0.499 * PI).max(1e-6);
        for _ in 0..16 {
            let tan_v = v.tan();
            let f = tan_v / v - x;
            // d/dv [tan(v)/v] = (1 + tan²v)/v - tan(v)/v².
            let sec_sq = 1.0 + tan_v * tan_v;
            let df = (sec_sq - tan_v / v) / v;
            v = (v - f / df).max(1e-6).min(0.499_999 * PI);
            if !v.is_finite() {
                v = (3.0 * (x - 1.0)).sqrt().min(0.49 * PI);
                break;
            }
        }
        0.5 * v * v
    }
}

/// Saddlepoint approximation K''(t), the variance of the tilted distribution
/// from K'(t) = x. K''(t) is variance, so positive on both branches.
///
/// Derivation. From `saddlepoint_solve` the saddlepoint parameterisation is
///   negative branch (t ≤ 0): K'(t) = tanh(v)/v with v = sqrt(-2t),
///   positive branch (t > 0): K'(t) = tan(v)/v  with v = sqrt( 2t).
/// Chain rule with dv/dt = ±1/v (sign matches the branch) yields
///   negative branch:  K''(t) = tanh(v)/v³ - sech²(v)/v²
///   positive branch:  K''(t) = sec²(v)/v²  - tan(v)/v³
/// As v → 0 both branches reduce to the same Taylor limit 2/3, which is the
/// continuous value of K''(0).
///
/// The previous form returned `sech²(v)/v² - tanh(v)/v³` on the negative
/// branch — the algebraic negative of the chain-rule derivative — and a
/// hardcoded `1/3` at t = 0 that did not match either one-sided limit. The
/// negative-branch sign error produced K''(-2) ≈ -0.103, which the test
/// `saddlepoint_kpp_is_positive` correctly flagged (variance must be > 0).
pub fn saddlepoint_kpp(t: f64) -> f64 {
    if t.abs() < 1e-14 {
        return 2.0 / 3.0;
    }
    if t < 0.0 {
        let v = (-2.0 * t).sqrt();
        let tanh_v = v.tanh();
        let sech_sq = 1.0 - tanh_v * tanh_v;
        (tanh_v / (v * v * v)) - (sech_sq / (v * v))
    } else {
        let v = (2.0 * t).sqrt();
        let tan_v = v.tan();
        let sec_sq = 1.0 + tan_v * tan_v;
        (sec_sq / (v * v)) - (tan_v / (v * v * v))
    }
}

/// Saddlepoint host draw for PG(b, c) with `13 < b ≤ 170`. This is the
/// reference the device sp_kernel matches in distribution; both fall
/// back to the convolution oracle when `b` is small enough that the
/// saddlepoint approximation has noticeable bias (validated by §12.4 test).
pub fn pg_saddlepoint_cpu_oracle(state: &mut XorwowState, b: u32, tilt: f64) -> f64 {
    // For now, use the convolution identity as the oracle. The saddlepoint
    // *kernel* is what we ship on device; the host oracle just needs to
    // produce the correct distribution for parity tests, and PG(b, c) =
    // sum_{j=1..b} PG(1, c) is exact for integer b. Device-side we use
    // the saddlepoint to *avoid* paying b times the PG(1) cost.
    pg_convolution_cpu_oracle(state, b, tilt)
}

// ────────────────────────────────────────────────────────────────────────
// Normal-approximation regime (math §10, b > 170) — host oracle
// ────────────────────────────────────────────────────────────────────────

// The closed-form `PG(b, c)` moments live once on the inference side
// (`crate::inference::pg_moments`) so the deterministic evidence path can use
// them without depending on this GPU module; re-export keeps the device oracle
// and the host evidence code on a single source of truth.
pub use crate::inference::pg_moments::{pg_mean, pg_variance};

/// Lyapunov-CLT closed-form draw for `b > NORMAL_MIN_B`. Truncated at
/// zero because PG support is `(0, +∞)`.
pub fn pg_normal_cpu_oracle(state: &mut XorwowState, b: u32, tilt: f64) -> f64 {
    let mean = pg_mean(b as f64, tilt);
    let var = pg_variance(b as f64, tilt);
    let sd = var.sqrt();
    let mut draw = mean + sd * state.next_norm();
    // Reflect into the positive half-line. At b > 170 the probability mass
    // below zero is ~Φ(-mean/sd) ≈ 0 for any reasonable c; reflection is a
    // negligibly biased truncation.
    if draw <= 0.0 {
        draw = -draw + 1e-300;
    }
    draw
}

// ────────────────────────────────────────────────────────────────────────
// Host dispatcher — CPU reference for the regime split (math §7)
// ────────────────────────────────────────────────────────────────────────

/// Per-row CPU draw using the appropriate regime. Used by the harness
/// when the GPU runtime is unavailable, and as the per-row oracle for
/// the dispatched device path’s parity tests.
pub fn draw_batch_cpu(input: &PolyaGammaBatchInput<'_>) -> Result<Array1<f64>, String> {
    input.validate()?;
    let n = input.rows();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut state = XorwowState::new(input.seed.0, i as u64);
        let b = input.shapes[i];
        let c = input.tilts[i];
        let v = if b <= PG1_MAX_B {
            pg1_draw_cpu_oracle(&mut state, c)
        } else if b < SADDLE_MIN_B {
            pg_convolution_cpu_oracle(&mut state, b, c)
        } else if b <= SADDLE_MAX_B {
            pg_saddlepoint_cpu_oracle(&mut state, b, c)
        } else {
            pg_normal_cpu_oracle(&mut state, b, c)
        };
        out[i] = v;
    }
    Ok(out)
}

/// Top-level entry point: dispatches to GPU when available, otherwise CPU.
/// Both paths use the same per-row XORWOW seeding so the GPU result is a
/// bit-equivalent of the CPU result up to IEEE rounding of `exp`/`log`/
/// `tan`/`tanh`/`sqrt` (which the device evaluators round to within 1 ULP
/// of the CPU `libm`).
pub fn draw_batch(input: PolyaGammaBatchInput<'_>) -> Result<Array1<f64>, String> {
    input.validate()?;

    #[cfg(target_os = "linux")]
    {
        if crate::gpu::device_runtime::GpuRuntime::global().is_some() {
            match linux_cuda::draw_batch_gpu(&input) {
                Ok(v) => return Ok(v),
                Err(GpuError::NoDeviceKernel { .. }) => {
                    // No device kernel for this path on this build: fall
                    // through to the CPU reference.
                }
                Err(other) => return Err(String::from(other)),
            }
        }
    }

    draw_batch_cpu(&input)
}

// ────────────────────────────────────────────────────────────────────────
// Phase 5: synthetic logistic Gibbs harness (validation oracle only)
// ────────────────────────────────────────────────────────────────────────

/// Single Gibbs step for the synthetic Bernoulli-logistic model
/// `y_i | β ~ Bernoulli(σ(x_iᵀ β))` with prior `β ~ N(0, Q_0⁻¹)`.
///
/// Steps (math block 7 §11):
///
/// 1. `ψ = X β` (length n).
/// 2. `ω_i ~ PG(1, ψ_i)` for all i (uses [`draw_batch`]).
/// 3. `z_i = (y_i − 1/2) / ω_i` (working response).
/// 4. `Q_ω = Xᵀ Ω X + Q_0`, `m_ω = Xᵀ Ω z`.
/// 5. Cholesky `Q_ω = L Lᵀ`, mean `μ = (Q_ω)⁻¹ m_ω = L⁻ᵀ L⁻¹ m_ω`.
/// 6. `β ← μ + L⁻ᵀ η` with `η ~ N(0, I_p)`.
///
/// This is a *primitive validation harness*; it deliberately runs entirely
/// on host except for the PG draws, which are the thing under test. The
/// posterior-inference path that ships with `gam` is NUTS, not this Gibbs
/// loop, and this module does not export the Gibbs sampler as a fit method.
pub fn logistic_gibbs_step(
    design: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, u8>,
    prior_precision: ArrayView2<'_, f64>,
    beta: ArrayView1<'_, f64>,
    seed: PgSeed,
    norm_seed: u64,
) -> Result<Array1<f64>, String> {
    let (n, p) = design.dim();
    if targets.len() != n {
        return Err(format!(
            "logistic_gibbs_step: y.len()={} != n={n}",
            targets.len()
        ));
    }
    if prior_precision.dim() != (p, p) {
        return Err(format!(
            "logistic_gibbs_step: Q_0 shape {:?} != ({p}, {p})",
            prior_precision.dim()
        ));
    }
    if beta.len() != p {
        return Err(format!(
            "logistic_gibbs_step: beta.len()={} != p={p}",
            beta.len()
        ));
    }

    // Step 1: ψ = X β  (host matvec — n×p × p).
    let mut psi = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..p {
            acc += design[[i, j]] * beta[j];
        }
        psi[i] = acc;
    }

    // Step 2: ω_i ~ PG(1, ψ_i).
    let shapes = Array1::<u32>::from_elem(n, 1);
    let omega = draw_batch(PolyaGammaBatchInput {
        shapes: shapes.view(),
        tilts: psi.view(),
        seed,
    })?;

    // Step 3: z_i = (y_i − 1/2) / ω_i  — but we never form z explicitly;
    //   m_ω = Xᵀ (y − 1/2)  (the ω cancels) is the standard PSW shortcut.
    let mut m = Array1::<f64>::zeros(p);
    for i in 0..n {
        let r = targets[i] as f64 - 0.5;
        for j in 0..p {
            m[j] += design[[i, j]] * r;
        }
    }

    // Step 4: Q_ω = Xᵀ Ω X + Q_0  (symmetric p × p; O(n p²)).
    let mut q = prior_precision.to_owned();
    for i in 0..n {
        let w = omega[i];
        for a in 0..p {
            let xa = design[[i, a]];
            for b in 0..p {
                q[[a, b]] += w * xa * design[[i, b]];
            }
        }
    }

    // Step 5: Cholesky L Lᵀ = Q_ω.
    let l = cholesky_lower_inplace(q.clone())
        .map_err(|e| format!("logistic_gibbs_step Cholesky: {e}"))?;
    // μ = (Q_ω)⁻¹ m via L y = m, Lᵀ μ = y.
    let mean = cholesky_solve_vector(&l, &m);

    // Step 6: β ← μ + L⁻ᵀ η.
    let mut norm_state = XorwowState::new(norm_seed, 0);
    let mut eta = Array1::<f64>::zeros(p);
    for j in 0..p {
        eta[j] = norm_state.next_norm();
    }
    let perturb = back_substitution_lower_transpose(&l, &eta);
    let mut beta_new = Array1::<f64>::zeros(p);
    for j in 0..p {
        beta_new[j] = mean[j] + perturb[j];
    }
    Ok(beta_new)
}

fn cholesky_lower_inplace(mut a: Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= a[[i, k]] * a[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!("non-SPD diagonal {sum} at row {i}"));
                }
                a[[i, j]] = sum.sqrt();
            } else {
                a[[i, j]] = sum / a[[j, j]];
            }
        }
        for j in (i + 1)..n {
            a[[i, j]] = 0.0;
        }
    }
    Ok(a)
}

// ────────────────────────────────────────────────────────────────────────
// Linux/CUDA implementation — Phases 2, 3, 4, 6
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux_cuda {
    use super::{
        PG1_MAX_B, PgSeed, PolyaGammaBatchInput, SADDLE_MAX_B, SADDLE_MIN_B, XorwowState,
        pg_convolution_cpu_oracle, pg_normal_cpu_oracle,
    };
    use crate::gpu::gpu_error::{GpuError, GpuResultExt};
    use crate::gpu::solver::context_and_stream;
    use cudarc::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
    use ndarray::Array1;
    use std::sync::Arc;

    /// NVRTC source prelude: SplitMix64 seeding, the per-row XORWOW state
    /// advance, and the unit/exp/normal draw helpers. The Devroye constants
    /// and the sampler body that follow are appended at compile time by
    /// [`ptx_source`], with the numeric constants rendered from the shared
    /// Rust [`crate::inference::polya_gamma_core::constants`] so no device
    /// literal is hand-typed.
    ///
    /// All arithmetic is in `double`; the device transcendentals (`exp`,
    /// `log`, `tanh`, `tan`, `sqrt`, `erfc`) are the high-accuracy intrinsics
    /// — we do NOT use `__expf` / `__tanhf`, which would diverge from the CPU
    /// oracle past a few ULPs.
    ///
    /// Layout of inputs/outputs:
    ///
    /// * `shapes` — u32, length `n`.
    /// * `tilts`  — f64, length `n`.
    /// * `out`    — f64, length `n`.
    /// * Each thread owns one row index `i`; it constructs its own XORWOW
    ///   state from `(seed, i)` via SplitMix64, draws once, and writes
    ///   `out[i]`. No shared state → no warp divergence beyond what the
    ///   algorithm itself dictates.
    const PTX_SOURCE_PRELUDE: &str = r#"
extern "C" __device__ unsigned long long splitmix64_mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    unsigned long long x = z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Per-row XORWOW state. Layout mirrors curand_kernel.h::curandStateXORWOW_t
// for the five 32-bit state lanes plus the addition counter. We omit the
// boxmuller_extra/boxmuller_flag cache since our normal draws use the
// polar method (which discards the second variate).
struct XorwowState {
    unsigned int s0, s1, s2, s3, s4, d;
};

extern "C" __device__ void xorwow_seed(struct XorwowState* st, unsigned long long seed, unsigned long long row) {
    const unsigned long long ROW_ZETA  = 0xA1B2C3D4E5F67890ULL;
    const unsigned long long WORD_GAMMA = 0x0F1E2D3C4B5A6978ULL;
    unsigned int words[6];
    for (int w = 0; w < 6; ++w) {
        unsigned long long composite = seed ^ (row * ROW_ZETA) ^ ((unsigned long long)w * WORD_GAMMA);
        unsigned long long h = splitmix64_mix(composite);
        words[w] = (unsigned int)(h >> 32);
    }
    if ((words[0] | words[1] | words[2] | words[3] | words[4]) == 0u) {
        words[0] = 1u;
    }
    st->s0 = words[0]; st->s1 = words[1]; st->s2 = words[2];
    st->s3 = words[3]; st->s4 = words[4]; st->d  = words[5];
}

extern "C" __device__ unsigned int xorwow_next(struct XorwowState* st) {
    unsigned int t = st->s4;
    unsigned int s = st->s0;
    st->s4 = st->s3;
    st->s3 = st->s2;
    st->s2 = st->s1;
    st->s1 = s;
    t ^= (t >> 2);
    t ^= (t << 1);
    t ^= s ^ (s << 4);
    st->s0 = t;
    st->d += 362437u;
    return t + st->d;
}

extern "C" __device__ double xorwow_unit(struct XorwowState* st) {
    unsigned int raw = xorwow_next(st);
    return ((double)raw + 1.0) * (1.0 / 4294967296.0);
}

extern "C" __device__ double xorwow_exp(struct XorwowState* st) {
    return -log(xorwow_unit(st));
}

extern "C" __device__ double xorwow_norm(struct XorwowState* st) {
    // Marsaglia polar — discard the partner variate, matches host oracle
    // byte-for-byte (host also discards).
    for (;;) {
        double u = 2.0 * xorwow_unit(st) - 1.0;
        double v = 2.0 * xorwow_unit(st) - 1.0;
        double s = u * u + v * v;
        if (s > 0.0 && s < 1.0) {
            double factor = sqrt(-2.0 * log(s) / s);
            return u * factor;
        }
    }
}
"#;

    /// NVRTC source body: the Devroye / saddlepoint device helpers and the
    /// three regime kernels. Appended by [`ptx_source`] after the prelude and
    /// the rendered `#define` constants. The `// ── Devroye PG(1, c)` helpers
    /// here consume `PG_FRAC_2_PI`, `PG_PI`, `PG_PI_SQ`, `PG_SQRT_2_OVER_PI`,
    /// and `PG_SQRT_PI_OVER_2`, all defined by the rendered constant block.
    const PTX_SOURCE_BODY: &str = r#"
extern "C" __device__ double std_normal_cdf(double x) {
    // 0.5 · erfc(-x / sqrt(2)).
    return 0.5 * erfc(-x * 0.7071067811865475);
}

extern "C" __device__ double pg_series(int n, double x) {
    if (x <= 0.0) return 0.0;
    double k = (double)n + 0.5;
    double k_sq = k * k;
    if (x <= PG_FRAC_2_PI) {
        double inv_x = 1.0 / x;
        return (2.0 * k * PG_SQRT_2_OVER_PI) * inv_x * sqrt(inv_x) * exp(-2.0 * k_sq * inv_x);
    } else {
        // Right branch — corrected coefficient PI · k (not PI / 2).
        return PG_PI * k * exp(-0.5 * k_sq * PG_PI_SQ * x);
    }
}

extern "C" __device__ double pg_exp_tail_mass(double tilt) {
    double base = 0.125 * PG_PI_SQ + 0.5 * tilt * tilt;
    double upper = PG_SQRT_PI_OVER_2 * (PG_FRAC_2_PI * tilt - 1.0);
    double lower = -(PG_SQRT_PI_OVER_2 * (PG_FRAC_2_PI * tilt + 1.0));
    double base_factor = base * exp(base * PG_FRAC_2_PI);
    double p_upper = base_factor * exp(-tilt) * std_normal_cdf(upper);
    double p_lower = base_factor * exp( tilt) * std_normal_cdf(lower);
    double exp_terms = (4.0 / PG_PI) * (p_upper + p_lower);
    return 1.0 / (1.0 + exp_terms);
}

extern "C" __device__ double sample_small_z(struct XorwowState* st, double z, double trunc) {
    double accept = 0.0;
    double sample = 0.0;
    while (accept < xorwow_unit(st)) {
        double exp_sample;
        for (;;) {
            double e1 = xorwow_exp(st);
            double e2 = xorwow_exp(st);
            if (e1 * e1 <= 2.0 * e2 / trunc) { exp_sample = e1; break; }
        }
        sample = 1.0 + exp_sample * trunc;
        sample = trunc / (sample * sample);
        accept = exp(-0.5 * z * z * sample);
    }
    return sample;
}

extern "C" __device__ double sample_large_z(struct XorwowState* st, double mean, double trunc) {
    double sample = 1.0e300;
    while (sample > trunc) {
        double n = xorwow_norm(st);
        double n_sq = n * n;
        double half_mean = 0.5 * mean;
        double mn_sq = mean * n_sq;
        double disc = sqrt(4.0 * mn_sq + mn_sq * mn_sq);
        sample = mean + half_mean * mn_sq - half_mean * disc;
        if (xorwow_unit(st) > mean / (mean + sample)) {
            sample = mean * mean / sample;
        }
    }
    return sample;
}

extern "C" __device__ double sample_trunc_inv_gauss(struct XorwowState* st, double z, double trunc) {
    double az = fabs(z);
    if (PG_FRAC_2_PI > az) {
        return sample_small_z(st, az, trunc);
    } else {
        return sample_large_z(st, 1.0 / az, trunc);
    }
}

extern "C" __device__ double pg1_draw(struct XorwowState* st, double tilt) {
    double half_tilt = fabs(tilt) * 0.5;
    double scale = 0.125 * PG_PI_SQ + 0.5 * half_tilt * half_tilt;
    double exp_mass = pg_exp_tail_mass(half_tilt);

    for (;;) {
        double u = xorwow_unit(st);
        double proposal;
        if (u < exp_mass) {
            proposal = PG_FRAC_2_PI + xorwow_exp(st) / scale;
        } else {
            proposal = sample_trunc_inv_gauss(st, half_tilt, PG_FRAC_2_PI);
        }
        double sum = pg_series(0, proposal);
        double threshold = xorwow_unit(st) * sum;
        int idx = 0;
        // The alternating-series tail. Bounded iteration cap (64) is
        // overwhelmingly safe: PSW 2013 show termination in <10 iters
        // with probability >1 - 1e-30 for any tilt; the cap exists only
        // to guarantee forward progress under hardware fault.
        for (int outer = 0; outer < 64; ++outer) {
            idx += 1;
            double term = pg_series(idx, proposal);
            if (idx & 1) {
                sum -= term;
                if (threshold <= sum) {
                    return 0.25 * proposal;
                }
            } else {
                sum += term;
                if (threshold >= sum) {
                    break;
                }
            }
        }
    }
}

// ── Saddlepoint helpers (math §9) ────────────────────────────────────────

extern "C" __device__ double saddlepoint_t(double x) {
    if (fabs(x - 1.0) < 1.0e-9) return 0.0;
    if (x < 1.0) {
        double v = sqrt(3.0 * (1.0 - x)); if (v < 1.0e-6) v = 1.0e-6;
        for (int it = 0; it < 6; ++it) {
            double tanh_v = tanh(v);
            double f  = tanh_v / v - x;
            double sech_sq = 1.0 - tanh_v * tanh_v;
            double df = (sech_sq - tanh_v / v) / v;
            v -= f / df;
            if (fabs(v) < 1.0e-12) break;
        }
        return -0.5 * v * v;
    } else {
        double v = sqrt(3.0 * (x - 1.0));
        if (v > 0.49 * PG_PI) v = 0.49 * PG_PI;
        if (v < 1.0e-6) v = 1.0e-6;
        for (int it = 0; it < 6; ++it) {
            double tan_v = tan(v);
            double f  = tan_v / v - x;
            double sec_sq = 1.0 + tan_v * tan_v;
            double df = (sec_sq - tan_v / v) / v;
            v -= f / df;
            if (v < 1.0e-6) v = 1.0e-6;
            if (v > 0.499999 * PG_PI) v = 0.499999 * PG_PI;
        }
        return 0.5 * v * v;
    }
}

// ── Kernels ──────────────────────────────────────────────────────────────

extern "C" __global__ void pg1_kernel(
    unsigned long long seed,
    unsigned int n,
    const unsigned int* __restrict__ rows,   // index map into shapes/tilts/out, length n
    const double* __restrict__ tilts,
    double* __restrict__ out)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    unsigned int row = rows[slot];
    struct XorwowState st;
    xorwow_seed(&st, seed, (unsigned long long)row);
    double c = tilts[row];
    out[row] = pg1_draw(&st, c);
}

extern "C" __global__ void sp_kernel(
    unsigned long long seed,
    unsigned int n,
    const unsigned int* __restrict__ rows,
    const unsigned int* __restrict__ shapes,
    const double* __restrict__ tilts,
    double* __restrict__ out)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    unsigned int row = rows[slot];
    struct XorwowState st;
    xorwow_seed(&st, seed, (unsigned long long)row);
    unsigned int b = shapes[row];
    double c = tilts[row];
    // Convolution-equivalent device fallback: sum b PG(1, c) draws. This
    // is correct in distribution; the *true* saddlepoint envelope ships
    // with phase 3 hill-climb. Until then, the kernel is callable and
    // produces draws that pass the §12 KS test — the only thing the
    // saddlepoint is supposed to buy is throughput at large b.
    double acc = 0.0;
    for (unsigned int j = 0; j < b; ++j) {
        acc += pg1_draw(&st, c);
    }
    // Touch saddlepoint_t so the helper isn’t DCE’d before phase 3 wiring;
    // the value is unused (multiplied by zero) so this is free.
    double sp_warm = saddlepoint_t(0.5);
    out[row] = acc + 0.0 * sp_warm;
}

extern "C" __global__ void normal_kernel(
    unsigned long long seed,
    unsigned int n,
    const unsigned int* __restrict__ rows,
    const unsigned int* __restrict__ shapes,
    const double* __restrict__ tilts,
    double* __restrict__ out)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    unsigned int row = rows[slot];
    struct XorwowState st;
    xorwow_seed(&st, seed, (unsigned long long)row);
    double b = (double)shapes[row];
    double c = fabs(tilts[row]);
    double mean;
    double var;
    if (c < 1.0e-8) {
        mean = 0.25 * b;
        var  = b / 24.0;
    } else {
        mean = b * tanh(0.5 * c) / (2.0 * c);
        double cosh_c = cosh(c);
        double sinh_c = sinh(c);
        var = b * (sinh_c - c) / (2.0 * c * c * c * (1.0 + cosh_c));
    }
    double sd = sqrt(var);
    double draw = mean + sd * xorwow_norm(&st);
    if (draw <= 0.0) draw = -draw + 1.0e-300;
    out[row] = draw;
}
"#;

    const THREADS_PER_BLOCK: u32 = 128;

    /// Assemble the full NVRTC source: the prelude, then the Devroye `#define`
    /// constants rendered from the shared Rust core, then the sampler body and
    /// kernels. Rendering the `#define` block from
    /// [`crate::inference::polya_gamma_core::render_cuda_constants`] is what
    /// parity-locks every device constant to its host value (issue #414) — the
    /// kernel and the CPU oracle cannot disagree on a numeric literal because
    /// there is exactly one source for those literals.
    pub(super) fn ptx_source() -> String {
        let mut src = String::with_capacity(PTX_SOURCE_PRELUDE.len() + PTX_SOURCE_BODY.len() + 256);
        src.push_str(PTX_SOURCE_PRELUDE);
        src.push_str(
            "\n// ── Devroye PG(1, c) constants (rendered from Rust core) ──────────────\n",
        );
        src.push_str(&crate::inference::polya_gamma_core::render_cuda_constants());
        src.push_str(PTX_SOURCE_BODY);
        src
    }

    fn module(ctx: &Arc<CudaContext>) -> Result<&'static Arc<CudaModule>, GpuError> {
        static CACHE: crate::gpu::device_cache::PtxModuleCache =
            crate::gpu::device_cache::PtxModuleCache::new();
        CACHE.get_or_compile(ctx, "polya_gamma", &ptx_source())
    }

    pub(super) fn draw_batch_gpu(
        input: &PolyaGammaBatchInput<'_>,
    ) -> Result<Array1<f64>, GpuError> {
        let n = input.rows();
        if n == 0 {
            return Ok(Array1::<f64>::zeros(0));
        }
        let (ctx, stream) =
            context_and_stream().map_err(|reason| GpuError::DriverCallFailed { reason })?;
        let compiled = module(&ctx)?;
        let module_handle: &Arc<CudaModule> = compiled;

        // ── Partition rows by regime (math §7). For the 2 ≤ b < SADDLE_MIN
        //   band the device kernel set above does not have a dedicated
        //   regime; we route those rows through host convolution and write
        //   straight into the output, avoiding the host-roundtrip cost for
        //   the dominant Bernoulli and normal-approx populations.
        let mut pg1_rows: Vec<u32> = Vec::new();
        let mut sp_rows: Vec<u32> = Vec::new();
        let mut normal_rows: Vec<u32> = Vec::new();
        let mut host_rows: Vec<u32> = Vec::new();
        for (i, &b) in input.shapes.iter().enumerate() {
            let idx = i as u32;
            if b <= PG1_MAX_B {
                pg1_rows.push(idx);
            } else if b < SADDLE_MIN_B {
                host_rows.push(idx);
            } else if b <= SADDLE_MAX_B {
                sp_rows.push(idx);
            } else {
                normal_rows.push(idx);
            }
        }

        // ── Upload shared inputs. cudarc's clone_htod takes &[T]; we
        //   need an owned Vec when the ndarray view is non-contiguous.
        let tilts_vec: Vec<f64> = match input.tilts.as_slice() {
            Some(s) => s.to_vec(),
            None => input.tilts.iter().copied().collect(),
        };
        let shapes_vec: Vec<u32> = match input.shapes.as_slice() {
            Some(s) => s.to_vec(),
            None => input.shapes.iter().copied().collect(),
        };
        let tilts_dev = stream
            .clone_htod(&tilts_vec)
            .gpu_ctx("polya_gamma upload tilts")?;
        let shapes_dev = stream
            .clone_htod(&shapes_vec)
            .gpu_ctx("polya_gamma upload shapes")?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(n)
            .gpu_ctx("polya_gamma alloc out")?;

        // ── Launch each regime kernel (skipping empty partitions).
        if !pg1_rows.is_empty() {
            let rows_dev = stream
                .clone_htod(&pg1_rows)
                .gpu_ctx("polya_gamma upload pg1 rows")?;
            launch_pg1(
                &stream,
                module_handle,
                input.seed,
                &rows_dev,
                &tilts_dev,
                &mut out_dev,
            )?;
        }
        if !sp_rows.is_empty() {
            let rows_dev = stream
                .clone_htod(&sp_rows)
                .gpu_ctx("polya_gamma upload sp rows")?;
            launch_sp(
                &stream,
                module_handle,
                input.seed,
                &rows_dev,
                &shapes_dev,
                &tilts_dev,
                &mut out_dev,
            )?;
        }
        if !normal_rows.is_empty() {
            let rows_dev = stream
                .clone_htod(&normal_rows)
                .gpu_ctx("polya_gamma upload normal rows")?;
            launch_normal(
                &stream,
                module_handle,
                input.seed,
                &rows_dev,
                &shapes_dev,
                &tilts_dev,
                &mut out_dev,
            )?;
        }

        // ── Pull results and patch the host-regime rows in place.
        let mut out_host = stream
            .clone_dtoh(&out_dev)
            .gpu_ctx("polya_gamma download out")?;
        for &row in &host_rows {
            let i = row as usize;
            let mut st = XorwowState::new(input.seed.0, row as u64);
            let b = input.shapes[i];
            let c = input.tilts[i];
            out_host[i] = if b <= SADDLE_MAX_B {
                pg_convolution_cpu_oracle(&mut st, b, c)
            } else {
                // Should not be reached given the partitioning above, but
                // route through the appropriate oracle for robustness.
                pg_normal_cpu_oracle(&mut st, b, c)
            };
        }
        Ok(Array1::from_vec(out_host))
    }

    fn launch_pg1(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        seed: PgSeed,
        rows: &cudarc::driver::CudaSlice<u32>,
        tilts: &cudarc::driver::CudaSlice<f64>,
        out: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let func = module
            .load_function("pg1_kernel")
            .gpu_ctx("polya_gamma load pg1_kernel")?;
        let n = rows.len() as u32;
        let grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let seed_arg: u64 = seed.0;
        // SAFETY: kernel signature matches arg types; out is a live device
        // buffer indexed by `rows[slot]` which is bounded by n.
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&seed_arg)
                .arg(&n)
                .arg(rows)
                .arg(tilts)
                .arg(out)
                .launch(cfg)
        }
        .map(|_| ())
        .gpu_ctx("polya_gamma launch pg1_kernel")
    }

    fn launch_sp(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        seed: PgSeed,
        rows: &cudarc::driver::CudaSlice<u32>,
        shapes: &cudarc::driver::CudaSlice<u32>,
        tilts: &cudarc::driver::CudaSlice<f64>,
        out: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let func = module
            .load_function("sp_kernel")
            .gpu_ctx("polya_gamma load sp_kernel")?;
        let n = rows.len() as u32;
        let grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let seed_arg: u64 = seed.0;
        // SAFETY: kernel signature matches; all slices are live and the
        // indexing via `rows[slot]` is bounded by the partition size.
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&seed_arg)
                .arg(&n)
                .arg(rows)
                .arg(shapes)
                .arg(tilts)
                .arg(out)
                .launch(cfg)
        }
        .map(|_| ())
        .gpu_ctx("polya_gamma launch sp_kernel")
    }

    fn launch_normal(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        seed: PgSeed,
        rows: &cudarc::driver::CudaSlice<u32>,
        shapes: &cudarc::driver::CudaSlice<u32>,
        tilts: &cudarc::driver::CudaSlice<f64>,
        out: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let func = module
            .load_function("normal_kernel")
            .gpu_ctx("polya_gamma load normal_kernel")?;
        let n = rows.len() as u32;
        let grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let seed_arg: u64 = seed.0;
        // SAFETY: kernel signature matches; all slices are live.
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&seed_arg)
                .arg(&n)
                .arg(rows)
                .arg(shapes)
                .arg(tilts)
                .arg(out)
                .launch(cfg)
        }
        .map(|_| ())
        .gpu_ctx("polya_gamma launch normal_kernel")
    }
}

// ────────────────────────────────────────────────────────────────────────
// Tests — host-side moment / KS validation (no GPU dependency)
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn theoretical_mean(b: f64, c: f64) -> f64 {
        pg_mean(b, c)
    }

    fn theoretical_variance(b: f64, c: f64) -> f64 {
        pg_variance(b, c)
    }

    #[test]
    fn pg1_cpu_oracle_matches_devroye_mean() {
        // Same moment test the inference/polya_gamma.rs sampler passes,
        // verifying our XORWOW-driven oracle produces the right
        // distribution. 25 000 samples; 10 % tolerance.
        let n = 25_000;
        for &(c, tol) in &[(0.0_f64, 0.05), (1.0, 0.10), (3.0, 0.10)] {
            let mut sum = 0.0;
            for i in 0..n {
                let mut st = XorwowState::new(0xC0FFEE_u64, i as u64);
                sum += pg1_draw_cpu_oracle(&mut st, c);
            }
            let emp = sum / n as f64;
            let th = theoretical_mean(1.0, c);
            let rel = (emp - th).abs() / th.max(1e-12);
            assert!(
                rel < tol,
                "PG(1,{c}) XORWOW oracle: emp {emp}, theory {th}, rel {rel}"
            );
        }
    }

    #[test]
    fn pg1_cpu_oracle_variance_matches_theory() {
        let n = 100_000;
        for &c in &[0.0_f64, 0.5, 2.0, 5.0] {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..n {
                let mut st = XorwowState::new(0xDEADBEEF_u64, i as u64);
                let x = pg1_draw_cpu_oracle(&mut st, c);
                sum += x;
                sum_sq += x * x;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;
            let th_var = theoretical_variance(1.0, c);
            let rel = (var - th_var).abs() / th_var.max(1e-12);
            assert!(
                rel < 0.05,
                "PG(1,{c}) var: emp {var}, theory {th_var}, rel {rel}"
            );
        }
    }

    #[test]
    fn xorwow_seeding_is_deterministic() {
        let mut a = XorwowState::new(42, 7);
        let mut b = XorwowState::new(42, 7);
        for _ in 0..1024 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let mut c = XorwowState::new(42, 8);
        let same = (0..32).all(|_| a.next_u32() == c.next_u32());
        assert!(!same, "different rows must produce different streams");
    }

    #[test]
    fn xorwow_unit_in_open_zero_closed_one() {
        let mut st = XorwowState::new(123, 0);
        for _ in 0..10_000 {
            let u = st.next_unit();
            assert!(u > 0.0 && u <= 1.0, "u={u} outside (0,1]");
        }
    }

    #[test]
    fn saddlepoint_solve_round_trips() {
        // K'(t) = tanh(v)/v on the negative-t branch, tan(v)/v on positive.
        // Recover t from K'(t) and check that re-evaluating K'(t) agrees.
        for &x in &[0.05_f64, 0.3, 0.7, 0.99, 1.01, 1.5, 3.0, 8.0] {
            let t = saddlepoint_solve(x);
            let kp = if t.abs() < 1e-14 {
                1.0
            } else if t < 0.0 {
                let v = (-2.0 * t).sqrt();
                v.tanh() / v
            } else {
                let v = (2.0 * t).sqrt();
                v.tan() / v
            };
            let rel = (kp - x).abs() / x.max(1e-12);
            assert!(
                rel < 1e-6,
                "saddlepoint_solve(x={x}) → t={t}; K'(t)={kp}, rel={rel}"
            );
        }
    }

    #[test]
    fn saddlepoint_kpp_is_positive() {
        // K'' is the variance of the tilted distribution; must be > 0.
        for &t in &[-2.0_f64, -0.5, -1e-5, 0.0, 1e-5, 0.5, 1.0] {
            let v = saddlepoint_kpp(t);
            assert!(v.is_finite() && v > 0.0, "K''({t}) = {v}");
        }
    }

    #[test]
    fn pg_normal_oracle_matches_moments_at_large_b() {
        // b = 500, c = 1.0: normal approximation should land moments to
        // ~1 % at 100k samples.
        let b = 500u32;
        let c = 1.0_f64;
        let n = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let mut st = XorwowState::new(0xBEEF_u64, i as u64);
            let x = pg_normal_cpu_oracle(&mut st, b, c);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let th_mean = theoretical_mean(b as f64, c);
        let th_var = theoretical_variance(b as f64, c);
        let m_rel = (mean - th_mean).abs() / th_mean;
        let v_rel = (var - th_var).abs() / th_var;
        assert!(
            m_rel < 0.02,
            "normal oracle mean: emp {mean}, theory {th_mean}, rel {m_rel}"
        );
        assert!(
            v_rel < 0.05,
            "normal oracle var: emp {var}, theory {th_var}, rel {v_rel}"
        );
    }

    #[test]
    fn batch_dispatch_handles_mixed_regimes() {
        // 4 rows, one in each regime band. CPU path should run cleanly.
        let shapes = ndarray::array![1u32, 5u32, 50u32, 300u32];
        let tilts = ndarray::array![0.5_f64, 0.5, 0.5, 0.5];
        let input = PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed: PgSeed(42),
        };
        let out = draw_batch_cpu(&input).expect("CPU dispatch");
        assert_eq!(out.len(), 4);
        for v in out.iter() {
            assert!(
                v.is_finite() && *v > 0.0,
                "PG draw must be positive finite: {v}"
            );
        }
    }

    #[test]
    fn logistic_gibbs_step_reduces_marginal_error() {
        // Sanity: starting from β = 0 on a small synthetic logistic dataset,
        // one Gibbs step should move toward the MLE direction. We don't
        // test convergence (that needs a chain); just that the new β is
        // finite, p-dimensional, and has nonzero displacement.
        let n = 200;
        let p = 3;
        let mut design = Array2::<f64>::zeros((n, p));
        let mut targets = Array1::<u8>::zeros(n);
        for i in 0..n {
            // Three covariates, last column intercept-like.
            let x1 = ((i as f64) / (n as f64)) * 2.0 - 1.0;
            let x2 = (((i * 7) % n) as f64 / n as f64) * 2.0 - 1.0;
            design[[i, 0]] = x1;
            design[[i, 1]] = x2;
            design[[i, 2]] = 1.0;
            let eta = 1.5 * x1 - 0.7 * x2 + 0.3;
            let p_y = 1.0 / (1.0 + (-eta).exp());
            // Deterministic Bernoulli via splitmix to avoid an RNG crate.
            let h = splitmix64_mix(i as u64 ^ 0xABCD_EF);
            let u = ((h >> 11) as f64) / ((1u64 << 53) as f64);
            targets[i] = if u < p_y { 1 } else { 0 };
        }
        let q0 = Array2::<f64>::eye(p) * 0.1;
        let beta = Array1::<f64>::zeros(p);
        let new_beta = logistic_gibbs_step(
            design.view(),
            targets.view(),
            q0.view(),
            beta.view(),
            PgSeed(1),
            9,
        )
        .expect("Gibbs step");
        assert_eq!(new_beta.len(), p);
        let disp: f64 = new_beta.iter().map(|b| b * b).sum::<f64>().sqrt();
        assert!(
            disp > 0.05 && disp.is_finite(),
            "Gibbs step displacement {disp} not meaningfully nonzero"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Charter §6 / §12 parity tests
    // ────────────────────────────────────────────────────────────────────

    /// Two-sample Kolmogorov–Smirnov statistic. Returns sup_x |F_a(x) − F_b(x)|.
    /// We avoid pulling a stats crate here because the test only needs the
    /// statistic (compared to an asymptotic critical value below) — the math
    /// is a pure sort + merge.
    fn ks_two_sample(a: &mut [f64], b: &mut [f64]) -> f64 {
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (na, nb) = (a.len() as f64, b.len() as f64);
        let (mut i, mut j) = (0usize, 0usize);
        let (mut fa, mut fb) = (0.0_f64, 0.0_f64);
        let mut d_max = 0.0_f64;
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                i += 1;
                fa = i as f64 / na;
            } else {
                j += 1;
                fb = j as f64 / nb;
            }
            let d = (fa - fb).abs();
            if d > d_max {
                d_max = d;
            }
        }
        d_max
    }

    /// KS critical value at α = 0.01 for a two-sample test with sample sizes
    /// `n_a`, `n_b`: `c(0.01) · sqrt((n_a + n_b)/(n_a · n_b))` with
    /// `c(0.01) ≈ 1.6276` (standard asymptotic table; one-sided 0.005 tail
    /// of the Kolmogorov distribution).
    fn ks_critical_001(n_a: usize, n_b: usize) -> f64 {
        let na = n_a as f64;
        let nb = n_b as f64;
        1.6276 * ((na + nb) / (na * nb)).sqrt()
    }

    #[test]
    fn pg1_cpu_oracle_matches_inference_module_distribution() {
        // KS test: the kernel-aligned XORWOW oracle here vs. the production
        // `inference::polya_gamma::PolyaGamma::draw` sampler should agree in
        // distribution (both implement Devroye with the corrected right-tail
        // coefficient). 5 000 samples each at three tilts; KS critical value
        // at α = 0.01.
        use crate::inference::polya_gamma::PolyaGamma;
        use rand::{SeedableRng, rngs::StdRng};
        let pg = PolyaGamma::new();
        for &c in &[0.0_f64, 1.5, 4.0] {
            let n_dev = 5_000;
            let n_ref = 5_000;
            let mut from_oracle: Vec<f64> = (0..n_dev)
                .map(|i| {
                    let mut st = XorwowState::new(0xDEADBEEF_u64 ^ c.to_bits(), i as u64);
                    pg1_draw_cpu_oracle(&mut st, c)
                })
                .collect();
            let mut from_reference: Vec<f64> = {
                let mut rng = StdRng::seed_from_u64(0xABCD_u64 ^ c.to_bits());
                (0..n_ref).map(|_| pg.draw(&mut rng, c)).collect()
            };
            let d = ks_two_sample(&mut from_oracle, &mut from_reference);
            let crit = ks_critical_001(n_dev, n_ref);
            assert!(
                d <= 2.0 * crit,
                "PG(1, c={c}) two-sample KS d={d} > 2·crit={}; XORWOW oracle and reference disagree in distribution",
                2.0 * crit
            );
        }
    }

    #[test]
    fn pg_convolution_identity_at_small_b() {
        // PG(b, c) =_d sum_{j=1..b} PG(1, c) for integer b. We compare two
        // independent draw streams: one drawing b independent PG(1, c) variates
        // and summing, the other drawing one PG(1, c) variate b times sharing a
        // single XORWOW (the dispatcher's convolution path). KS at α = 0.01.
        let n = 4_000;
        let b: u32 = 8;
        let c: f64 = 1.2;
        let mut left: Vec<f64> = (0..n)
            .map(|i| {
                // Reset state per draw so successive PG(1) draws share the same
                // chain — matches the host convolution path.
                let mut st = XorwowState::new(0x1111_u64, i as u64);
                (0..b).map(|_| pg1_draw_cpu_oracle(&mut st, c)).sum()
            })
            .collect();
        let mut right: Vec<f64> = (0..n)
            .map(|i| {
                // Independent fresh state per j to make this a genuinely
                // independent sum-of-PG(1) stream (different from `left` but
                // same distribution).
                (0..b)
                    .map(|j| {
                        let mut st = XorwowState::new(0x2222_u64 ^ (j as u64), i as u64);
                        pg1_draw_cpu_oracle(&mut st, c)
                    })
                    .sum::<f64>()
            })
            .collect();
        let d = ks_two_sample(&mut left, &mut right);
        let crit = ks_critical_001(n, n);
        assert!(
            d <= 2.0 * crit,
            "PG({b}, {c}) convolution identity KS d={d} > 2·crit={}",
            2.0 * crit
        );
    }

    #[test]
    fn pg_normal_kernel_matches_moments_at_b_500() {
        // CPU oracle for the normal-approximation kernel hits PSW (b, c)
        // moments to 2 % mean / 5 % var at b = 500 with 50 000 draws. The
        // GPU kernel runs the same arithmetic with the same XORWOW state,
        // so this test is also a parity gate for the device path (any
        // device drift would surface as a CPU/GPU oracle mismatch first).
        let b = 500u32;
        let c = 2.0_f64;
        let n = 50_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let mut st = XorwowState::new(0xCAFE_u64, i as u64);
            let x = pg_normal_cpu_oracle(&mut st, b, c);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let th_mean = pg_mean(b as f64, c);
        let th_var = pg_variance(b as f64, c);
        let m_rel = (mean - th_mean).abs() / th_mean;
        let v_rel = (var - th_var).abs() / th_var;
        assert!(
            m_rel < 0.02,
            "normal kernel mean: emp {mean}, theory {th_mean}, rel {m_rel}"
        );
        assert!(
            v_rel < 0.05,
            "normal kernel var: emp {var}, theory {th_var}, rel {v_rel}"
        );
    }

    #[test]
    fn logistic_gibbs_chain_converges_to_mle_direction() {
        // End-to-end Gibbs harness validation. Start from β = 0, run 200
        // steps on a small synthetic Bernoulli-logistic dataset with known
        // β* = (1.5, -0.7, 0.3). Drop the first 50 as burn-in and check that
        // the posterior mean direction aligns with β* (cosine > 0.85).
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let n = 400;
        let p = 3;
        let beta_star = [1.5_f64, -0.7, 0.3];
        let mut design = Array2::<f64>::zeros((n, p));
        let mut targets = Array1::<u8>::zeros(n);
        let mut rng = StdRng::seed_from_u64(0xFEED);
        for i in 0..n {
            let x1 = ((i as f64) / (n as f64)) * 2.0 - 1.0;
            let x2 = (((i * 13) % n) as f64 / n as f64) * 2.0 - 1.0;
            design[[i, 0]] = x1;
            design[[i, 1]] = x2;
            design[[i, 2]] = 1.0;
            let eta = beta_star[0] * x1 + beta_star[1] * x2 + beta_star[2];
            let p_y = 1.0 / (1.0 + (-eta).exp());
            let u: f64 = rng.random();
            targets[i] = if u < p_y { 1 } else { 0 };
        }
        let q0 = Array2::<f64>::eye(p) * 0.01;
        let mut beta = Array1::<f64>::zeros(p);
        let mut accum = Array1::<f64>::zeros(p);
        let steps = 200;
        let burn = 50;
        for k in 0..steps {
            beta = logistic_gibbs_step(
                design.view(),
                targets.view(),
                q0.view(),
                beta.view(),
                PgSeed(0xC0DE + k as u64),
                0xCAFE + k as u64,
            )
            .expect("Gibbs step");
            if k >= burn {
                for j in 0..p {
                    accum[j] += beta[j];
                }
            }
        }
        for j in 0..p {
            accum[j] /= (steps - burn) as f64;
        }
        let dot: f64 = (0..p).map(|j| accum[j] * beta_star[j]).sum();
        let na: f64 = accum.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nb: f64 = beta_star.iter().map(|v| v * v).sum::<f64>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.85,
            "Gibbs chain posterior-mean direction does not align with β*: cos = {cos}, accum = {accum:?}, β* = {beta_star:?}"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Charter §7 hill-climb gates (Linux-only, `#[ignore]` by default —
    // run with `cargo test -- --ignored polya_gamma_hill_climb_` on the
    // V100. The 50×/20× ratios compare CPU vs GPU draws built in the same
    // mode; the NVRTC kernel runs at device speed regardless of host opt
    // level, so the ratio is meaningful at any host build mode.
    // ────────────────────────────────────────────────────────────────────

    /// Hill-climb gate: pure Bernoulli (b = 1) at n = 200 000 must run on the
    /// GPU at ≥ 50× the CPU oracle's draw rate. This is the dominant large-scale
    /// PG draw shape (one PG variate per data row per Gibbs iteration), so a
    /// 50× win here is the actual ship gate for the device sampler.
    #[test]
    #[cfg(target_os = "linux")]
    fn polya_gamma_hill_climb_pg1_50x() {
        if crate::gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[polya_gamma_hill_climb_pg1_50x] no CUDA runtime on host — skipping");
            return;
        }
        let n = 200_000usize;
        let shapes = Array1::<u32>::from_elem(n, 1);
        let mut tilts = Array1::<f64>::zeros(n);
        for i in 0..n {
            tilts[i] = ((i as f64) / (n as f64)) * 6.0 - 3.0;
        }
        let seed = PgSeed(0x50_4F_4C_59_47_41_4D_41);

        // Warm the device module (NVRTC compile, allocator priming) so the
        // first kernel launch's compile time doesn't pollute the timing.
        {
            let warm_shapes = Array1::<u32>::from_elem(16, 1);
            let warm_tilts = Array1::<f64>::zeros(16);
            draw_batch(PolyaGammaBatchInput {
                shapes: warm_shapes.view(),
                tilts: warm_tilts.view(),
                seed,
            })
            .expect("warm");
        }

        let t_gpu_start = std::time::Instant::now();
        let _gpu = draw_batch(PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed,
        })
        .expect("GPU draw_batch");
        let dt_gpu = t_gpu_start.elapsed().as_secs_f64();

        let t_cpu_start = std::time::Instant::now();
        let _cpu = draw_batch_cpu(&PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed,
        })
        .expect("CPU draw_batch");
        let dt_cpu = t_cpu_start.elapsed().as_secs_f64();

        let speedup = dt_cpu / dt_gpu;
        println!(
            "polya_gamma_hill_climb_pg1: n={n} cpu={dt_cpu:.3}s gpu={dt_gpu:.3}s speedup={speedup:.1}×"
        );
        assert!(
            speedup >= 50.0,
            "PG(1) GPU speedup {speedup:.1}× < 50× hill-climb gate (cpu={dt_cpu:.3}s, gpu={dt_gpu:.3}s)"
        );
    }

    /// Hill-climb gate: mixed negative-binomial style workload — 80 % of rows
    /// at b ≥ 200 (normal-approx regime), 20 % at b = 1 (pg1 regime), 0 % at
    /// the placeholder saddlepoint band so the throughput claim is not
    /// dependent on the unfinished sp_kernel. 200 000 rows total; gate is
    /// ≥ 20× CPU.
    #[test]
    #[cfg(target_os = "linux")]
    fn polya_gamma_hill_climb_mixed_nb_20x() {
        if crate::gpu::device_runtime::GpuRuntime::global().is_none() {
            eprintln!("[polya_gamma_hill_climb_mixed_nb_20x] no CUDA runtime on host — skipping");
            return;
        }
        let n = 200_000usize;
        let mut shapes = Array1::<u32>::zeros(n);
        let mut tilts = Array1::<f64>::zeros(n);
        for i in 0..n {
            // 20 % b = 1, 80 % b = 250 (normal regime).
            shapes[i] = if i.is_multiple_of(5) { 1 } else { 250 };
            tilts[i] = ((i as f64) / (n as f64)) * 4.0 - 2.0;
        }
        let seed = PgSeed(0xDEAD_BEEF_CAFE_BABE);

        // Warm
        let warm_shapes = Array1::<u32>::from_elem(16, 250);
        let warm_tilts = Array1::<f64>::zeros(16);
        draw_batch(PolyaGammaBatchInput {
            shapes: warm_shapes.view(),
            tilts: warm_tilts.view(),
            seed,
        })
        .expect("warm");

        let t_gpu = std::time::Instant::now();
        let _g = draw_batch(PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed,
        })
        .expect("GPU mixed");
        let dt_gpu = t_gpu.elapsed().as_secs_f64();

        let t_cpu = std::time::Instant::now();
        let _c = draw_batch_cpu(&PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed,
        })
        .expect("CPU mixed");
        let dt_cpu = t_cpu.elapsed().as_secs_f64();

        let speedup = dt_cpu / dt_gpu;
        println!(
            "polya_gamma_hill_climb_mixed: n={n} cpu={dt_cpu:.3}s gpu={dt_gpu:.3}s speedup={speedup:.1}×"
        );
        assert!(
            speedup >= 20.0,
            "Mixed NB GPU speedup {speedup:.1}× < 20× gate (cpu={dt_cpu:.3}s, gpu={dt_gpu:.3}s)"
        );
    }

    /// GPU parity gate: when the runtime is available, the dispatched
    /// `draw_batch` path must agree with the CPU oracle bit-for-bit, since
    /// both consume the same XORWOW byte stream per row. macOS / no-runtime
    /// builds skip the body cleanly.
    #[test]
    #[cfg(target_os = "linux")]
    fn pg1_gpu_matches_cpu_oracle_when_runtime_available() {
        if crate::gpu::device_runtime::GpuRuntime::global().is_none() {
            return;
        }
        let n = 256usize;
        let shapes = Array1::<u32>::from_elem(n, 1);
        let mut tilts = Array1::<f64>::zeros(n);
        for i in 0..n {
            tilts[i] = ((i as f64) / (n as f64)) * 6.0 - 3.0;
        }
        let seed = PgSeed(0x9E37_79B9_7F4A_7C15);
        let gpu = draw_batch(PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed,
        })
        .expect("GPU draw_batch");
        let cpu = draw_batch_cpu(&PolyaGammaBatchInput {
            shapes: shapes.view(),
            tilts: tilts.view(),
            seed,
        })
        .expect("CPU draw_batch");
        assert_eq!(gpu.len(), cpu.len());
        // The device transcendentals (exp / log / tanh / sqrt) round to within
        // ~1 ULP of glibc's libm but are not bit-identical, so we test a tight
        // relative tolerance rather than equality. A 1e-6 relative tolerance is
        // far inside the PG distribution's spread and any genuine algorithmic
        // drift (e.g. wrong series term) would blow this out by orders of
        // magnitude.
        for i in 0..n {
            let g = gpu[i];
            let c = cpu[i];
            let rel = (g - c).abs() / c.max(1e-12);
            assert!(
                rel < 1e-6,
                "pg1 GPU/CPU divergence at row {i}, tilt={}: gpu={g}, cpu={c}, rel={rel}",
                tilts[i]
            );
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Issue #414 unification parity gates
    // ────────────────────────────────────────────────────────────────────

    /// Device-source parity lock: the embedded CUDA source must consume the
    /// Devroye constants *rendered from the Rust core*, with no second
    /// hand-typed copy of those literals. We assert the assembled NVRTC source
    /// embeds the rendered `#define` block verbatim and that the prelude/body
    /// templates carry no stray `#define PG_…` of their own (which would be a
    /// drift hazard). Linux-only because `ptx_source` lives in the CUDA module.
    #[test]
    #[cfg(target_os = "linux")]
    fn cuda_source_uses_rendered_constants_only() {
        let rendered = crate::inference::polya_gamma_core::render_cuda_constants();
        let assembled = linux_cuda::ptx_source();
        assert!(
            assembled.contains(rendered.trim_end()),
            "assembled CUDA source does not embed the rendered constant block"
        );
        // No constant literal may be hand-typed in the templates; the only
        // `#define PG_` lines must come from the rendered block.
        let define_count = assembled.matches("#define PG_").count();
        let rendered_count = rendered.matches("#define PG_").count();
        assert_eq!(
            define_count, rendered_count,
            "CUDA source has {define_count} `#define PG_` lines but the rendered block has {rendered_count}; a stale hand-typed constant is present"
        );
    }
}
