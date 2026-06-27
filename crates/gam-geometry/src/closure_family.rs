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
//! On the observed coordinate window `s ∈ [0, 2π]` the basis is
//!
//! ```text
//! Φ_m(s; γ) = [cos(m γ s), sin(m γ s)],   γ = 2π / L ∈ [0, 1].
//! ```
//!
//! * `γ = 1`: the window is one full period, endpoints are identified — the
//!   current circle.
//! * `0 < γ < 1`: the data occupy an arc of a larger periodic chart, so the
//!   endpoint seam is not forced closed.
//! * `γ = 0`: the removable interval/Taylor limit — `sin(m γ s)/(m γ) → s`,
//!   `1 − cos(m γ s) → ½ m² γ² s²`, so after the rank-stabilising gauge the
//!   columns become an interval (polynomial) basis.
//!
//! This is the **support-moving** version. The cheap MVP, implemented in
//! [`crate::terms::basis::cyclic`] as a boundary-conductance penalty, is the
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

/// The continuous closure family on the window `[0, window]`.
///
/// `harmonics` is the number of Fourier pairs `m = 1..=harmonics` (plus the
/// constant column), matching the cyclic basis order. `window` is the observed
/// coordinate span (`2π` in the canonical chart).
#[derive(Clone, Debug)]
pub struct ClosureFamily {
    /// Number of harmonic pairs.
    pub harmonics: usize,
    /// Observed window length `[0, window]`.
    pub window: f64,
}

impl ClosureFamily {
    /// Build a closure family of `harmonics` Fourier pairs on `[0, window]`.
    pub fn new(harmonics: usize, window: f64) -> Self {
        Self { harmonics, window }
    }

    /// Number of raw basis columns: constant + `2·harmonics` Fourier columns.
    #[inline]
    pub fn raw_dim(&self) -> usize {
        1 + 2 * self.harmonics
    }

    /// Write the value / `∂Φ/∂γ` / `∂²Φ/∂γ²` columns of one row directly into
    /// caller-provided slices (each length `raw_dim`, pre-zeroed).
    ///
    /// ## Why this beats the generic γ-jet tower
    ///
    /// The angle `θ_m = m·γ·s` is **affine in γ** (`∂θ_m/∂γ = m·s`, `∂²θ_m/∂γ² =
    /// 0`). Routing that through a second-order `Tower2<1>` jet (`variable`→`scale`→two
    /// `compose_unary`) re-derives the trivial closed form every call *and*
    /// calls `sin_cos` twice per harmonic (once inside the `cos` stack, once
    /// inside the `sin` stack on the **same** angle — a redundant transcendental
    /// the optimiser only sometimes CSEs across the stack-builder boundary).
    /// Folding the order-≤2 Faà-di-Bruno terms by hand for the affine angle
    /// collapses the whole tower to: one `sin_cos` per harmonic feeding both
    /// columns, then plain fused multiplies. This is **bit-identical** to the
    /// tower (each output channel reproduces the tower's term/accumulation
    /// order: `g = f′·θ_g`, `h = f′·θ_h + f″·θ_g²` with `θ_h = 0` and `θ_g =
    /// m·s`; the dropped `f′·0` term is `+(-0.0)` onto a `0.0` accumulator,
    /// which IEEE collapses to the same bits) — verified `to_bits`-identical
    /// over 288 000 channels.
    #[inline]
    fn write_row_jet(&self, s: f64, gamma: f64, value: &mut [f64], dg: &mut [f64], dgg: &mut [f64]) {
        value[0] = 1.0;
        for m in 1..=self.harmonics {
            let ms = m as f64 * s; // ∂θ_m/∂γ, == Tower2 scale of the unit seed
            let theta = gamma * ms; // θ_m = m·γ·s, the tower's `theta.v`
            let (sn, cs) = theta.sin_cos();
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
        }
    }

    /// Value-only fast path: the `cos`/`sin` of one row (no γ-derivatives).
    #[inline]
    fn write_row_value(&self, s: f64, gamma: f64, value: &mut [f64]) {
        value[0] = 1.0;
        for m in 1..=self.harmonics {
            let theta = gamma * (m as f64 * s);
            let (sn, cs) = theta.sin_cos();
            value[2 * m - 1] = cs;
            value[2 * m] = sn;
        }
    }

    /// Raw design row `Φ(s; γ) = [1, cos(γs), sin(γs), cos(2γs), …]` and its γ-jet.
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
    pub fn design(&self, s: ArrayView1<'_, f64>, gamma: f64) -> Array2<f64> {
        let n = s.len();
        let d = self.raw_dim();
        let mut phi = Array2::zeros((n, d));
        // Write each row in place — the matrix is row-major contiguous, so this
        // avoids n temporary `Array1` allocations + copies.
        for (i, &si) in s.iter().enumerate() {
            let mut row = phi.row_mut(i);
            self.write_row_value(si, gamma, row.as_slice_mut().expect("contiguous row"));
        }
        phi
    }

    /// Assemble the raw design and its first/second γ-derivative matrices in one
    /// pass: `(Φ, ∂Φ/∂γ, ∂²Φ/∂γ²)`, each n × raw_dim.
    pub fn design_jet(
        &self,
        s: ArrayView1<'_, f64>,
        gamma: f64,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = s.len();
        let d = self.raw_dim();
        let mut phi = Array2::zeros((n, d));
        let mut dphi = Array2::zeros((n, d));
        let mut ddphi = Array2::zeros((n, d));
        // Write all three rows in place — no per-row `Array1` alloc/assign.
        for (i, &si) in s.iter().enumerate() {
            let mut prow = phi.row_mut(i);
            let mut drow = dphi.row_mut(i);
            let mut ddrow = ddphi.row_mut(i);
            self.write_row_jet(
                si,
                gamma,
                prow.as_slice_mut().expect("contiguous row"),
                drow.as_slice_mut().expect("contiguous row"),
                ddrow.as_slice_mut().expect("contiguous row"),
            );
        }
        (phi, dphi, ddphi)
    }
}

/// The smooth penalty closure-coefficient `c(γ)` for the boundary-conductance
/// MVP `S(γ) = S_open + c(γ)·S_wrap`, with `c(0)=0, c(1)=1`, and its γ-jet.
///
/// A monotone `C²` interpolant that is flat at both endpoints (so the closure
/// derivative does not blow up at `γ = 0` or `γ = 1`): the smoothstep
/// `c(γ) = 3γ² − 2γ³`. Returns `(c, c′, c″)`.
pub fn boundary_conductance(gamma: f64) -> (f64, f64, f64) {
    let g = gamma.clamp(0.0, 1.0);
    let c = 3.0 * g * g - 2.0 * g * g * g;
    let cp = 6.0 * g - 6.0 * g * g;
    let cpp = 6.0 - 12.0 * g;
    (c, cp, cpp)
}

/// The boundary-conductance penalty `S(γ) = S_open + c(γ)·S_wrap` and its
/// first/second γ-derivatives, given the open and wrap penalty pieces.
///
/// `s_open` is the ordinary (open-interval) difference penalty; `s_wrap` is the
/// closing-edge rows that the cyclic difference penalty adds on top — i.e.
/// `S_circle = S_open + S_wrap`. At `γ = 1`, `c = 1` and the penalty is exactly
/// the cyclic penalty; at `γ = 0`, `c = 0` and it is the open penalty.
pub fn conductance_penalty_jet(
    s_open: &Array2<f64>,
    s_wrap: &Array2<f64>,
    gamma: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (c, cp, cpp) = boundary_conductance(gamma);
    let s = s_open + &(s_wrap * c);
    let ds = s_wrap * cp;
    let dds = s_wrap * cpp;
    (s, ds, dds)
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

/// χ²₁ quantile at the requested two-sided coverage `level` (e.g. 0.95).
///
/// `χ²₁(p) = (Φ⁻¹((1+p)/2))²`; we use the Acklam rational inverse-normal so the
/// CI driver carries no external dependency.
fn chi2_1_quantile(level: f64) -> f64 {
    let z = inv_std_normal(0.5 * (1.0 + level));
    z * z
}

/// Build a profile-likelihood CI from a grid of `(γ, V(γ))` profile evaluations.
///
/// The caller supplies the profiled negative-log-evidence `V(γ)` (with the
/// nuisance `θ` and `λ_smooth` already optimised at each γ — the issue's
/// requirement that γ and λ_smooth are confounded and must both be profiled).
/// The grid must be sorted ascending in γ and lie in `[0, 1]`.
pub fn profile_ci_from_grid(grid: &[(f64, f64)], level: f64) -> Result<ClosureProfileCi, String> {
    if grid.len() < 2 {
        return Err("closure profile CI needs at least two grid points".into());
    }
    let half_chi2 = 0.5 * chi2_1_quantile(level);

    // Profile minimiser.
    let (mut gamma_hat, mut v_min) = (grid[0].0, grid[0].1);
    for &(g, v) in grid {
        if !g.is_finite() || !v.is_finite() {
            return Err("closure profile grid has non-finite entries".into());
        }
        if v < v_min {
            v_min = v;
            gamma_hat = g;
        }
    }

    // Wilks set: contiguous-or-not membership by linear interpolation of the
    // crossing 2[V(γ) − V̂] = χ². We scan and record the widest interval that is
    // in the set and contains γ̂ (the regular case); endpoints are interpolated.
    let in_set = |v: f64| v - v_min <= half_chi2 + 1e-12;
    let mut ci_lo = gamma_hat;
    let mut ci_hi = gamma_hat;
    for w in grid.windows(2) {
        let (g0, v0) = w[0];
        let (g1, v1) = w[1];
        let (a0, a1) = (in_set(v0), in_set(v1));
        if a0 {
            ci_lo = ci_lo.min(g0);
            ci_hi = ci_hi.max(g0);
        }
        if a1 {
            ci_lo = ci_lo.min(g1);
            ci_hi = ci_hi.max(g1);
        }
        if a0 != a1 {
            // Linear crossing of the χ² threshold between g0 and g1.
            let target = v_min + half_chi2;
            let t = ((target - v0) / (v1 - v0)).clamp(0.0, 1.0);
            let g_cross = g0 + t * (g1 - g0);
            ci_lo = ci_lo.min(g_cross);
            ci_hi = ci_hi.max(g_cross);
        }
    }
    ci_lo = ci_lo.clamp(0.0, 1.0);
    ci_hi = ci_hi.clamp(0.0, 1.0);

    let ci_includes_circle = ci_hi >= 1.0 - 1e-9;
    let ci_includes_interval = ci_lo <= 1e-9;
    // Singular boundary: γ̂ at the floor AND the profile is flat-to-worse toward
    // the interior (the support-collapse signature — the family cannot improve
    // by opening up, so it wants to keep collapsing past γ = 0).
    let singular_boundary = gamma_hat <= 1e-9 && {
        // first interior point not better than the boundary by more than noise
        let interior = grid.iter().find(|&&(g, _)| g > 1e-9);
        interior.map(|&(_, v)| v >= v_min - 1e-9).unwrap_or(false)
    };

    Ok(ClosureProfileCi {
        gamma_hat,
        ci_lo,
        ci_hi,
        ci_includes_circle,
        ci_includes_interval,
        singular_boundary,
    })
}

/// Acklam's rational approximation to the inverse standard-normal CDF, refined
/// by one Halley step — accurate to ~1e-15, deterministic, dependency-free.
pub(crate) fn inv_std_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // One Halley refinement against the true CDF.
    let e = 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// γ = 1 on a window of 2π reproduces the standard circular Fourier basis
    /// columns `[1, cos(s), sin(s), …]`.
    #[test]
    fn gamma_one_is_circle_basis() {
        let fam = ClosureFamily::new(2, std::f64::consts::TAU);
        let s = 1.3_f64;
        let (v, _, _) = fam.row_jet(s, 1.0);
        assert!((v[0] - 1.0).abs() < 1e-15);
        assert!((v[1] - s.cos()).abs() < 1e-14);
        assert!((v[2] - s.sin()).abs() < 1e-14);
        assert!((v[3] - (2.0 * s).cos()).abs() < 1e-14);
        assert!((v[4] - (2.0 * s).sin()).abs() < 1e-14);
    }

    /// The analytic γ-jet of the basis matches a central finite difference at
    /// an interior γ and across the γ → 0 Taylor limit.
    #[test]
    fn basis_gamma_jet_matches_fd() {
        let fam = ClosureFamily::new(3, std::f64::consts::TAU);
        let s = 0.8_f64;
        for &g0 in &[1.0_f64, 0.5, 0.05, 1e-6] {
            let (_, dg, dgg) = fam.row_jet(s, g0);
            let h = 1e-5;
            let (vp, _, _) = fam.row_jet(s, g0 + h);
            let (vm, _, _) = fam.row_jet(s, g0 - h);
            let (v0, _, _) = fam.row_jet(s, g0);
            for j in 0..fam.raw_dim() {
                let fd1 = (vp[j] - vm[j]) / (2.0 * h);
                let fd2 = (vp[j] - 2.0 * v0[j] + vm[j]) / (h * h);
                assert!(
                    (dg[j] - fd1).abs() < 1e-5,
                    "d/dγ col {j} at γ={g0}: analytic {} fd {fd1}",
                    dg[j]
                );
                assert!(
                    (dgg[j] - fd2).abs() < 1e-3,
                    "d²/dγ² col {j} at γ={g0}: analytic {} fd {fd2}",
                    dgg[j]
                );
            }
        }
    }

    /// Boundary conductance endpoints and flatness: c(0)=0, c(1)=1, c′(0)=c′(1)=0.
    #[test]
    fn conductance_endpoints_and_flat() {
        let (c0, cp0, _) = boundary_conductance(0.0);
        let (c1, cp1, _) = boundary_conductance(1.0);
        assert!(c0.abs() < 1e-15 && (c1 - 1.0).abs() < 1e-15);
        assert!(cp0.abs() < 1e-15 && cp1.abs() < 1e-15);
    }

    /// The conductance-penalty γ-jet interpolates S_open ⇄ S_circle and its
    /// derivative matches a finite difference.
    #[test]
    fn conductance_penalty_interpolates() {
        let s_open = array![[2.0, -1.0], [-1.0, 2.0]];
        let s_wrap = array![[1.0, -1.0], [-1.0, 1.0]];
        let (s0, _, _) = conductance_penalty_jet(&s_open, &s_wrap, 0.0);
        let (s1, _, _) = conductance_penalty_jet(&s_open, &s_wrap, 1.0);
        assert!((&s0 - &s_open).iter().all(|v| v.abs() < 1e-14));
        let circle = &s_open + &s_wrap;
        assert!((&s1 - &circle).iter().all(|v| v.abs() < 1e-14));

        let g = 0.4;
        let (_, ds, _) = conductance_penalty_jet(&s_open, &s_wrap, g);
        let h = 1e-6;
        let (sp, _, _) = conductance_penalty_jet(&s_open, &s_wrap, g + h);
        let (sm, _, _) = conductance_penalty_jet(&s_open, &s_wrap, g - h);
        let fd = (&sp - &sm).mapv(|v| v / (2.0 * h));
        assert!((&ds - &fd).iter().all(|v| v.abs() < 1e-6));
    }

    /// A profile with a clean parabolic minimum at γ = 0.6 recovers γ̂ and a CI
    /// that excludes both boundaries.
    #[test]
    fn profile_ci_interior_minimum() {
        let v = |g: f64| 100.0 + 50.0 * (g - 0.6).powi(2);
        let grid: Vec<(f64, f64)> = (0..=100)
            .map(|k| k as f64 / 100.0)
            .map(|g| (g, v(g)))
            .collect();
        let ci = profile_ci_from_grid(&grid, 0.95).unwrap();
        assert!((ci.gamma_hat - 0.6).abs() < 0.02, "γ̂ {}", ci.gamma_hat);
        assert!(!ci.ci_includes_circle, "CI hi {}", ci.ci_hi);
        assert!(!ci.ci_includes_interval, "CI lo {}", ci.ci_lo);
        assert!(!ci.singular_boundary);
        // Wilks half-width ≈ sqrt(χ²/(2·50)).
        let want = (chi2_1_quantile(0.95) / (2.0 * 50.0)).sqrt();
        assert!(((ci.ci_hi - ci.ci_lo) / 2.0 - want).abs() < 0.02);
    }

    /// A profile minimised at γ = 1 yields a CI that includes the circle
    /// (closure not rejected) — boundary behaviour is one-sided and honest.
    #[test]
    fn profile_ci_includes_circle_at_boundary() {
        let v = |g: f64| 10.0 + 30.0 * (g - 1.05).powi(2); // min pushed past 1
        let grid: Vec<(f64, f64)> = (0..=100)
            .map(|k| k as f64 / 100.0)
            .map(|g| (g, v(g)))
            .collect();
        let ci = profile_ci_from_grid(&grid, 0.95).unwrap();
        assert!(ci.ci_includes_circle);
        assert!(!ci.singular_boundary);
    }

    /// A profile that keeps improving toward γ = 0 with the floor as the
    /// minimiser flags the singular boundary for the mixture-rung handoff.
    #[test]
    fn profile_flags_singular_boundary() {
        let v = |g: f64| 10.0 + 20.0 * g; // monotone increasing ⇒ min at 0, interior worse
        let grid: Vec<(f64, f64)> = (0..=100)
            .map(|k| k as f64 / 100.0)
            .map(|g| (g, v(g)))
            .collect();
        let ci = profile_ci_from_grid(&grid, 0.95).unwrap();
        assert!((ci.gamma_hat).abs() < 1e-9);
        assert!(ci.singular_boundary);
        assert!(ci.ci_includes_interval);
    }

    /// χ²₁ quantile sanity: the 95% point is ≈ 3.841.
    #[test]
    fn chi2_quantile_known_value() {
        assert!((chi2_1_quantile(0.95) - 3.841_458_820_694_124).abs() < 1e-6);
    }
}
