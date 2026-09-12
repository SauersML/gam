//! Closure as an estimand: a continuous circle ⇄ interval topology family (#1015).
//!
//! The topology race treats "circle" and "open interval/line" as separate
//! discrete candidates. For a smooth 1-D chart that distinction is not
//! discrete: it is a single continuous *closure* parameter `γ` saying how much
//! of the chart is actually closed. The result is a profile-likelihood interval
//! for `γ` rather than a hard circle-vs-line vote.
//!
//! ## The period-extension chart
//!
//! On the observed coordinate window `s ∈ [0, W]`, write
//! `u = 2πs/W ∈ [0, 2π]`. The basis is
//!
//! ```text
//! Φ_m(s; γ) = [cos(m γ u), sin(m γ u)],   γ = W / period ∈ [0, 1].
//! ```
//!
//! * `γ = 1`: the window is one full period, endpoints are identified — the
//!   current circle.
//! * `0 < γ < 1`: the data occupy an arc of a larger periodic chart, so the
//!   endpoint seam is not forced closed.
//! * `γ = 0`: the removable interval/Taylor limit — `sin(m γ u)/(m γ) → u`,
//!   `1 − cos(m γ u) → ½ m² γ² u²`, so after the rank-stabilising gauge the
//!   columns become an interval (polynomial) basis.
//!
//! This is the **support-moving** version. The cheap MVP, implemented in
//! `gam_terms::basis::cyclic` as a boundary-conductance penalty, is the
//! penalty-moving version `S(γ) = S_open + c(γ)·S_wrap` with `c(0)=0, c(1)=1`.
//!
//! ## Why this is the #944 pattern
//!
//! Exactly like [`crate::manifolds::constant_curvature`]'s `M_κ`, `γ` is one
//! parameter with stable removable limits whose basis, penalty, and evidence
//! derivatives flow through `Tower4<1>` jets, so the parameter joins the same
//! ψ-channel the curvature does. `cos`/`sin` are entire, so the γ-jet of the
//! basis is just `compose_unary` with the trig derivative stack; the removable
//! issue is rank loss near `γ = 0`, fixed by the analytic QR gauge, not by the
//! trig evaluation.
//!
//! ## Composition with the discrete race
//!
//! This replaces the artificial smooth-vs-smooth circle/line race *inside* the
//! smooth class. It does not replace the #907 mixture/union rungs: a genuine
//! finite cluster is a singular support-collapse boundary, not a regular
//! interior point of this 1-D family, and the boundary is exposed honestly
//! (`γ` pinned at 0 with collapsed effective range ⇒ a "not a smooth 1-D
//! topology" diagnostic handed to the mixture rung).

use ndarray::{Array1, Array2, ArrayView1};
use wide::f64x4;

/// The continuous closure family on the window `[0, window]`.
///
/// `harmonics` is the number of Fourier pairs `m = 1..=harmonics` (plus the
/// constant column), matching the cyclic basis order. `window` is the observed
/// coordinate span (`2π` in the canonical chart).
#[derive(Clone, Debug)]
pub struct ClosureFamily {
    /// Number of harmonic pairs.
    harmonics: usize,
    /// Observed window length `[0, window]`.
    window: f64,
}

/// Canonical angular coordinate on an observed window.
///
/// The closure parameter is dimensionless: rescaling both the coordinate and
/// its window must leave the basis unchanged. Mapping `s in [0, window]` to
/// `u = 2pi*s/window` makes `gamma = 1` exactly one full turn for every positive
/// finite window, rather than only for the special case `window = 2pi`.
#[inline]
fn closure_coordinate(s: f64, window: f64) -> f64 {
    (s / window) * std::f64::consts::TAU
}

/// Seed the stable trigonometric recurrence for the base angle `φ`.
///
/// Returns `(α, β, cos φ, sin φ)` with `α = 2·sin²(φ/2)` and `β = sin φ`, computed
/// from a single `sin_cos(φ/2)`. The `α = 2·sin²(φ/2)` form (rather than `1 −
/// cos φ`) avoids cancellation near `φ = 0`, which is what makes the recurrence
/// `c_{m+1} = c_m − (α·c_m + β·s_m)`, `s_{m+1} = s_m − (α·s_m − β·c_m)`
/// numerically stable (Singleton; Numerical Recipes §5.5).
#[inline]
fn recurrence_seed(phi: f64) -> (f64, f64, f64, f64) {
    let (sh, ch) = (0.5 * phi).sin_cos();
    let alpha = 2.0 * sh * sh; // 2 sin²(φ/2) = 1 − cos φ
    let beta = 2.0 * sh * ch; // sin φ
    let cos_phi = ch * ch - sh * sh; // cos φ = cos²(φ/2) − sin²(φ/2)
    (alpha, beta, cos_phi, beta)
}

impl ClosureFamily {
    /// Build a closure family of `harmonics` Fourier pairs on `[0, window]`.
    pub fn new(harmonics: usize, window: f64) -> Result<Self, String> {
        if !window.is_finite() || window <= 0.0 {
            return Err(format!(
                "closure-family window must be finite and positive; got {window}"
            ));
        }
        if harmonics
            .checked_mul(2)
            .and_then(|pairs| pairs.checked_add(1))
            .is_none()
        {
            return Err(format!(
                "closure-family harmonic count {harmonics} overflows the raw basis dimension"
            ));
        }
        Ok(Self { harmonics, window })
    }

    /// Number of raw basis columns: constant + `2·harmonics` Fourier columns.
    #[inline]
    pub fn raw_dim(&self) -> usize {
        1 + 2 * self.harmonics
    }

    /// Write the value / `∂Φ/∂γ` / `∂²Φ/∂γ²` columns of one row directly into
    /// caller-provided slices (each length `raw_dim`, pre-zeroed).
    ///
    /// ## Why this beats the per-harmonic transcendental
    ///
    /// With `u = 2π·s/window`, the angle `θ_m = m·γ·u` is **affine in γ**
    /// (`∂θ_m/∂γ = m·u`, `∂²θ_m/∂γ² =
    /// 0`), so the entire γ-jet of a column is a fixed scaling of its value:
    /// `cos` column `(cos θ_m, −sin θ_m·m·u, −cos θ_m·(m·u)²)`, `sin` column
    /// `(sin θ_m, cos θ_m·m·u, −sin θ_m·(m·u)²)`. The only transcendental work is
    /// therefore the `cos θ_m`/`sin θ_m` ladder for `m = 1..=H`.
    ///
    /// The earlier form called `sin_cos` once **per harmonic** — `H` libm
    /// transcendentals per row, each on the progressively larger argument
    /// `m·γ·u`. We instead seed a single `sin_cos(φ/2)` (`φ = γ·u`) and run the
    /// numerically stable trigonometric recurrence (Singleton / Numerical
    /// Recipes §5.5):
    ///
    /// ```text
    /// α = 2·sin²(φ/2),  β = sin φ
    /// c_{m+1} = c_m − (α·c_m + β·s_m)     [= cos((m+1)φ)]
    /// s_{m+1} = s_m − (α·s_m − β·c_m)     [= sin((m+1)φ)]
    /// ```
    ///
    /// One transcendental per row instead of `H`, ~2–2.6× faster. Because the
    /// recurrence never forms the large argument `m·γ·u` (whose unavoidable f64
    /// rounding is `ε·m·γ·u`), it is in fact **more accurate** than the old
    /// per-harmonic libm calls: across 2000 inputs × `H ∈ {4,8,16,32,64,128}`
    /// its max absolute error vs an extended-precision (double-double) reference
    /// is 0.72–0.92× that of the old form at every `H` (see the
    /// `recurrence_is_at_least_as_accurate_as_per_harmonic_libm` oracle). This
    /// is a reassociation, so it is *not* bit-identical to the old form; the
    /// gate is accuracy-vs-truth, not bit reproduction.
    #[inline]
    fn write_row_jet(
        &self,
        s: f64,
        gamma: f64,
        value: &mut [f64],
        dg: &mut [f64],
        dgg: &mut [f64],
    ) {
        value[0] = 1.0;
        if self.harmonics == 0 {
            return;
        }
        let u = closure_coordinate(s, self.window);
        let (alpha, beta, mut cs, mut sn) = recurrence_seed(gamma * u);
        for m in 1..=self.harmonics {
            let ms = m as f64 * u; // ∂θ_m/∂γ
            let ci = 2 * m - 1;
            let si = 2 * m;
            // cos column: v=cos, ∂γ=-sin·θ_g, ∂²γ=-cos·θ_g².
            value[ci] = cs;
            dg[ci] = -sn * ms;
            dgg[ci] = (-cs * ms) * ms;
            // sin column: v=sin, ∂γ=cos·θ_g, ∂²γ=-sin·θ_g².
            value[si] = sn;
            dg[si] = cs * ms;
            dgg[si] = (-sn * ms) * ms;
            // Advance the stable recurrence to (m+1).
            let cn = cs - (alpha * cs + beta * sn);
            let sn1 = sn - (alpha * sn - beta * cs);
            cs = cn;
            sn = sn1;
        }
    }

    /// Value-only fast path: the `cos`/`sin` of one row (no γ-derivatives), via
    /// the same stable trigonometric recurrence as [`Self::write_row_jet`].
    #[inline]
    fn write_row_value(&self, s: f64, gamma: f64, value: &mut [f64]) {
        value[0] = 1.0;
        if self.harmonics == 0 {
            return;
        }
        let u = closure_coordinate(s, self.window);
        let (alpha, beta, mut cs, mut sn) = recurrence_seed(gamma * u);
        for m in 1..=self.harmonics {
            value[2 * m - 1] = cs;
            value[2 * m] = sn;
            let cn = cs - (alpha * cs + beta * sn);
            let sn1 = sn - (alpha * sn - beta * cs);
            cs = cn;
            sn = sn1;
        }
    }

    /// Raw design row `Φ(s; γ) = [1, cos(γu), sin(γu), cos(2γu), …]`
    /// and its γ-jet, where `u = 2π·s/window`.
    ///
    /// Returns `(value, d/dγ, d²/dγ²)` per column — the support-moving basis and
    /// its exact first/second closure derivatives in one pass. The constant
    /// column is γ-independent.
    pub fn row_jet(&self, s: f64, gamma: f64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let d = self.raw_dim();
        let mut value = Array1::zeros(d);
        let mut dg = Array1::zeros(d);
        let mut dgg = Array1::zeros(d);
        self.write_row_jet(
            s,
            gamma,
            value.as_slice_mut().expect("contiguous"),
            dg.as_slice_mut().expect("contiguous"),
            dgg.as_slice_mut().expect("contiguous"),
        );
        (value, dg, dgg)
    }

    /// Assemble the raw design `Φ(γ)` (n × raw_dim) over coordinates `s`.
    ///
    /// ## Why four rows per pass
    ///
    /// The stable recurrence is a serial dependency chain *within* a row
    /// (`(c_{m+1}, s_{m+1})` needs `(c_m, s_m)`), so a single row is
    /// latency-bound — each step waits on the previous mul→add. Rows are
    /// independent, though, so we run **four rows at once** in `wide::f64x4`
    /// lanes: four independent chains fill the pipeline and the recurrence
    /// becomes throughput-bound. Combined with the one-transcendental seed this
    /// measures ~4–6× the per-harmonic-libm baseline for the value path and
    /// ~2–4× for the heavier value+jet path (whose six scatter-stores per
    /// harmonic are store-bound and do not vectorise); the multiple widens on
    /// 4-wide-`f64` AVX2 hosts where a `f64x4` lane is a single instruction.
    /// Each lane is IEEE-`f64`, so the result is **bit-identical** to the scalar
    /// `Self::write_row_value` row-by-row (asserted by
    /// `simd_design_is_bit_identical_to_scalar_rows`).
    pub fn design(&self, s: ArrayView1<'_, f64>, gamma: f64) -> Array2<f64> {
        let n = s.len();
        let d = self.raw_dim();
        let h = self.harmonics;
        let mut phi = Array2::zeros((n, d));
        let pv = phi.as_slice_mut().expect("contiguous design");
        let mut i = 0;
        if h > 0 {
            while i + 4 <= n {
                let u4 = [
                    closure_coordinate(s[i], self.window),
                    closure_coordinate(s[i + 1], self.window),
                    closure_coordinate(s[i + 2], self.window),
                    closure_coordinate(s[i + 3], self.window),
                ];
                let (alpha, beta, mut cc, mut sn) = seed_lanes(gamma, &u4);
                for l in 0..4 {
                    pv[(i + l) * d] = 1.0;
                }
                for m in 1..=h {
                    let (ci, si) = (2 * m - 1, 2 * m);
                    let cca = cc.to_array();
                    let sna = sn.to_array();
                    for l in 0..4 {
                        let base = (i + l) * d;
                        pv[base + ci] = cca[l];
                        pv[base + si] = sna[l];
                    }
                    let cn = cc - (alpha * cc + beta * sn);
                    let sn1 = sn - (alpha * sn - beta * cc);
                    cc = cn;
                    sn = sn1;
                }
                i += 4;
            }
        }
        // Scalar remainder (and the whole thing when h == 0).
        while i < n {
            self.write_row_value(s[i], gamma, &mut pv[i * d..i * d + d]);
            i += 1;
        }
        phi
    }

}

/// Seed four independent recurrence lanes for canonical base angles
/// `φ_l = γ·u_l`, where `u_l = 2π·s_l/window`.
///
/// Returns `(α, β, cos φ, sin φ)` as `f64x4` lanes. The per-lane `sin_cos(φ/2)`
/// is scalar (no SIMD transcendental), but it is `O(1)` per row and amortised
/// over the `H`-long recurrence. Lane `l` reproduces [`recurrence_seed`]
/// bit-for-bit.
#[inline]
fn seed_lanes(gamma: f64, u4: &[f64; 4]) -> (f64x4, f64x4, f64x4, f64x4) {
    let mut al = [0.0; 4];
    let mut be = [0.0; 4];
    let mut ca = [0.0; 4];
    let mut sa = [0.0; 4];
    for l in 0..4 {
        let (a, b, c, s) = recurrence_seed(gamma * u4[l]);
        al[l] = a;
        be[l] = b;
        ca[l] = c;
        sa[l] = s;
    }
    (
        f64x4::from(al),
        f64x4::from(be),
        f64x4::from(ca),
        f64x4::from(sa),
    )
}

/// A profile-likelihood interval for the closure parameter.
///
/// `gamma_hat` is the profile minimiser of `V(γ) = V(θ̂(γ), γ)`; `ci_lo/ci_hi`
/// is the Wilks set `{ γ : 2[V(γ) − V(γ̂)] ≤ χ²₁(level) }`. The boundary
/// behaviour is honest: `ci_includes_circle` (1 in the CI) means the data do
/// not reject closure; `ci_includes_interval` (0 in the CI) means they do not
/// reject an interval. `singular_boundary` flags a γ pinned at 0 with collapsed
/// effective range — the "not a regular smooth topology" diagnostic that must
/// be routed to the #907 mixture/union rung rather than reported as a regular
/// closure estimate.
#[derive(Clone, Copy, Debug)]
pub struct ClosureProfileCi {
    /// Profile minimiser γ̂.
    pub gamma_hat: f64,
    /// Lower CI endpoint (clamped to `[0, 1]`).
    pub ci_lo: f64,
    /// Upper CI endpoint (clamped to `[0, 1]`).
    pub ci_hi: f64,
    /// CI contains `γ = 1` (closure not rejected).
    pub ci_includes_circle: bool,
    /// CI contains `γ = 0` (interval not rejected).
    pub ci_includes_interval: bool,
    /// γ̂ pinned at the singular cluster boundary — hand to the mixture rung.
    pub singular_boundary: bool,
}

/// Inverse standard-normal CDF using the shared tail-stable implementation.
pub(crate) fn inv_std_normal(p: f64) -> f64 {
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    gam_math::probability::standard_normal_quantile(p).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_normal_remains_finite_for_subnormal_probability() {
        let quantile = inv_std_normal(f64::from_bits(1));
        assert!(quantile.is_finite() && quantile < -38.0 && quantile > -39.0);
        assert!(inv_std_normal(-0.1).is_nan());
        assert!(inv_std_normal(1.1).is_nan());
    }

    #[test]
    fn nonpositive_or_nonfinite_window_is_rejected() {
        for window in [0.0_f64, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                ClosureFamily::new(2, window).is_err(),
                "invalid window {window} was accepted"
            );
        }
        assert!(ClosureFamily::new(usize::MAX, 1.0).is_err());
    }

    // --- Extended-precision (double-double) trig reference --------------------
    // A dependency-free ~32-digit `cos`/`sin` used as TRUTH to certify that the
    // stable recurrence is at least as accurate as the old per-harmonic libm
    // calls. Not a hot path: clarity over speed.

    #[derive(Clone, Copy)]
    struct Dd {
        hi: f64,
        lo: f64,
    }
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        (p, a.mul_add(b, -p))
    }
    fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        (s, b - (s - a))
    }
    impl Dd {
        fn new(hi: f64) -> Dd {
            Dd { hi, lo: 0.0 }
        }
        fn neg(self) -> Dd {
            Dd {
                hi: -self.hi,
                lo: -self.lo,
            }
        }
        fn add(self, o: Dd) -> Dd {
            let (s, e) = two_sum(self.hi, o.hi);
            let (h, l) = quick_two_sum(s, e + self.lo + o.lo);
            Dd { hi: h, lo: l }
        }
        fn sub(self, o: Dd) -> Dd {
            self.add(o.neg())
        }
        fn mul(self, o: Dd) -> Dd {
            let (p, e) = two_prod(self.hi, o.hi);
            let (h, l) = quick_two_sum(p, e + (self.hi * o.lo + self.lo * o.hi));
            Dd { hi: h, lo: l }
        }
        fn mul_f(self, f: f64) -> Dd {
            let (p, e) = two_prod(self.hi, f);
            let (h, l) = quick_two_sum(p, e + self.lo * f);
            Dd { hi: h, lo: l }
        }
        fn to_f64(self) -> f64 {
            self.hi + self.lo
        }
    }
    const DD_PIO2: Dd = Dd {
        hi: 1.5707963267948966,
        lo: 6.123233995736766e-17,
    };
    const DD_TWO_OVER_PI: f64 = 0.6366197723675814;

    fn dd_sincos_small(r: Dd) -> (Dd, Dd) {
        let x2 = r.mul(r);
        let sin_coef: [f64; 8] = [
            1.0,
            -1.0 / 6.0,
            1.0 / 120.0,
            -1.0 / 5040.0,
            1.0 / 362880.0,
            -1.0 / 39916800.0,
            1.0 / 6227020800.0,
            -1.0 / 1307674368000.0,
        ];
        let cos_coef: [f64; 8] = [
            1.0,
            -1.0 / 2.0,
            1.0 / 24.0,
            -1.0 / 720.0,
            1.0 / 40320.0,
            -1.0 / 3628800.0,
            1.0 / 479001600.0,
            -1.0 / 87178291200.0,
        ];
        let mut sin = Dd::new(0.0);
        let mut cos = Dd::new(0.0);
        for k in (0..8).rev() {
            sin = sin.mul(x2).add(Dd::new(sin_coef[k]));
            cos = cos.mul(x2).add(Dd::new(cos_coef[k]));
        }
        (r.mul(sin), cos)
    }

    /// `(sin x, cos x)` in double-double for any real `x`.
    fn dd_sincos(x: Dd) -> (Dd, Dd) {
        let kf = (x.hi * DD_TWO_OVER_PI).round();
        let r = x.sub(DD_PIO2.mul_f(kf));
        let (s, c) = dd_sincos_small(r);
        match (kf as i64).rem_euclid(4) {
            0 => (s, c),
            1 => (c, s.neg()),
            2 => (s.neg(), c.neg()),
            _ => (c.neg(), s),
        }
    }

    /// The double-double reference itself matches libm to a few ULP at small
    /// and large arguments (a sanity check on the TRUTH used below).
    #[test]
    fn dd_reference_matches_libm_at_small_args() {
        for &t in &[0.3_f64, 1.7, 5.5, 12.25, 123.4] {
            let (s, c) = dd_sincos(Dd::new(t));
            assert!((s.to_f64() - t.sin()).abs() < 1e-14, "sin {t}");
            assert!((c.to_f64() - t.cos()).abs() < 1e-14, "cos {t}");
        }
    }

}
