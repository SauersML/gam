//! Certified 1-D Chebyshev profile of the radial design scalars `(φ, q, t)`.
//!
//! ## Why
//!
//! Within one spatial-hyperparameter trial (fixed `κ` / partial-fraction
//! coefficients), the radial scalars consumed by every `(row, center)` pair
//! of an anisotropic radial design sweep are a single smooth function of one
//! variable: `J(r) = (φ(r), q(r), t(r))`, a finite sum of `r^{2m−d}` power
//! blocks and `r^ν K_ν(κr)` Matérn blocks — analytic on `(0, ∞)`. The sweep
//! evaluates `J` at `n × k` radii (≈ 480k at the large-scale conditional-PGS
//! shape), and each exact evaluation costs tens of microseconds across the
//! partial-fraction blocks. That product was measured (#979 stack profile)
//! as the dominant cost of every Duchon κ-trial: ~15–20 s per outer
//! evaluation at the 20k-row CTN stage.
//!
//! In log-coordinates `u = ln r` the observed radius range is compact and
//! `J(e^u)` is analytic there, so Chebyshev interpolation converges
//! geometrically: a once-per-trial build from a few hundred exact jet
//! evaluations replaces per-point transcendental work with a short Clenshaw
//! contraction.
//!
//! ## Certification, not approximation-by-fiat
//!
//! [`RadialProfile::build`] returns `None` (callers fall back to exact
//! per-point evaluation) unless BOTH
//! 1. the Chebyshev coefficient tail decays below [`PROFILE_CERT_RTOL`] of
//!    each channel's scale — the geometric-decay certificate for analytic
//!    interpolands, with node-count escalation; and
//! 2. deterministic off-grid spot checks against the exact evaluator agree
//!    to [`PROFILE_SPOT_RTOL`].
//!
//! Radii outside the built range (or any non-finite evaluation) are answered
//! by the exact evaluator via [`RadialProfile::eval_or_exact`] — the same
//! certified-or-fallback discipline as the non-affine quadrature ladder and
//! the cell-moment families.

use super::{BasisError, RadialScalarKind};

/// Relative ceiling on the Chebyshev coefficient tail for certification.
pub const PROFILE_CERT_RTOL: f64 = 1.0e-13;

/// Relative agreement required at the off-grid spot checks.
pub const PROFILE_SPOT_RTOL: f64 = 1.0e-12;

/// Node-count escalation ladder for the profile build.
pub const PROFILE_NODE_LADDER: [usize; 3] = [64, 128, 256];

/// Number of deterministic off-grid spot-check points.
pub const PROFILE_SPOT_CHECK_POINTS: usize = 5;

/// Certified Chebyshev interpolant of `(φ, q, t)` over `u = ln r ∈
/// [u_lo, u_hi]` for one frozen [`RadialScalarKind`].
pub struct RadialProfile {
    pub(crate) u_lo: f64,
    pub(crate) u_hi: f64,
    pub(crate) m: usize,
    /// `coeff[c][p]`: Chebyshev coefficient `p` of channel `c ∈ {φ, q, t}`.
    pub(crate) coeff: [Vec<f64>; 3],
}

impl RadialProfile {
    /// Build and certify a profile for `kind` covering `[r_min, r_max]`.
    ///
    /// `None` when the radius range is degenerate/non-positive, any exact
    /// evaluation fails or is non-finite (e.g. kernels that are degenerate
    /// at collision inside the range), or no ladder rung certifies.
    pub fn build(kind: &RadialScalarKind, r_min: f64, r_max: f64) -> Option<Self> {
        if !(r_min.is_finite() && r_max.is_finite()) || r_min <= 0.0 || r_max <= r_min {
            return None;
        }
        let u_lo = r_min.ln();
        let u_hi = r_max.ln();
        for &m in PROFILE_NODE_LADDER.iter() {
            let Some(profile) = Self::build_at(kind, u_lo, u_hi, m) else {
                // An exact evaluation failed or was non-finite somewhere in
                // the range — no larger rung can fix that.
                return None;
            };
            if profile.certify(kind) {
                return Some(profile);
            }
        }
        None
    }

    pub(crate) fn build_at(
        kind: &RadialScalarKind,
        u_lo: f64,
        u_hi: f64,
        m: usize,
    ) -> Option<Self> {
        // Chebyshev nodes of the first kind (no endpoints).
        let mut values: [Vec<f64>; 3] = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
        let mut nodes_x = vec![0.0_f64; m];
        for (i, x_slot) in nodes_x.iter_mut().enumerate() {
            let x = (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * m) as f64).cos();
            *x_slot = x;
            let u = 0.5 * (u_lo + u_hi) + 0.5 * (u_hi - u_lo) * x;
            let r = u.exp();
            let (phi, q, t) = kind.eval_design_triplet(r).ok()?;
            if !(phi.is_finite() && q.is_finite() && t.is_finite()) {
                return None;
            }
            values[0][i] = phi;
            values[1][i] = q;
            values[2][i] = t;
        }
        // First-kind discrete orthogonality:
        //   c_p = (γ_p / m) Σ_i f(x_i) T_p(x_i),  γ_0 = 1, γ_p = 2.
        let mut basis = vec![0.0_f64; m * m];
        for (i, &x) in nodes_x.iter().enumerate() {
            basis[i * m] = 1.0;
            if m > 1 {
                basis[i * m + 1] = x;
            }
            for p in 2..m {
                basis[i * m + p] = 2.0 * x * basis[i * m + p - 1] - basis[i * m + p - 2];
            }
        }
        let coeff = values.map(|vals| {
            let mut c = vec![0.0_f64; m];
            for (p, c_slot) in c.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (i, &v) in vals.iter().enumerate() {
                    acc += v * basis[i * m + p];
                }
                let gamma = if p == 0 { 1.0 } else { 2.0 };
                *c_slot = gamma * acc / m as f64;
            }
            c
        });
        Some(Self {
            u_lo,
            u_hi,
            m,
            coeff,
        })
    }

    pub(crate) fn certify(&self, kind: &RadialScalarKind) -> bool {
        // 1. Tail decay per channel, relative to that channel's own scale.
        let tail_band = (self.m / 16).max(2);
        for c in &self.coeff {
            let scale = c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            if scale == 0.0 {
                continue;
            }
            let tail = c[self.m - tail_band..]
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            if tail > PROFILE_CERT_RTOL * scale {
                return false;
            }
        }
        // 2. Deterministic off-grid spot checks (golden-ratio interior
        //    points — reproducible, no RNG).
        let phi_ratio = 0.618_033_988_749_894_9_f64;
        for s in 1..=PROFILE_SPOT_CHECK_POINTS {
            let f = (0.37 + s as f64 * phi_ratio).fract();
            let u = self.u_lo + f * (self.u_hi - self.u_lo);
            let r = u.exp();
            let Ok((phi_e, q_e, t_e)) = kind.eval_design_triplet(r) else {
                return false;
            };
            let (phi_i, q_i, t_i) = self.eval_inside(r);
            for (interp, exact) in [(phi_i, phi_e), (q_i, q_e), (t_i, t_e)] {
                let scale = exact.abs().max(interp.abs()).max(f64::MIN_POSITIVE);
                if (interp - exact).abs() > PROFILE_SPOT_RTOL * scale {
                    return false;
                }
            }
        }
        true
    }

    /// `true` when `r` lies inside the certified interpolation range.
    #[inline]
    pub fn covers(&self, r: f64) -> bool {
        if !(r > 0.0) {
            return false;
        }
        let u = r.ln();
        u >= self.u_lo && u <= self.u_hi
    }

    /// Interpolated `(φ, q, t)` for an in-range radius (caller must have
    /// checked [`Self::covers`]). Clenshaw over the three channels sharing
    /// one basis recurrence.
    #[inline]
    pub(crate) fn eval_inside(&self, r: f64) -> (f64, f64, f64) {
        let u = r.ln();
        let x = (2.0 * u - (self.u_lo + self.u_hi)) / (self.u_hi - self.u_lo);
        let two_x = 2.0 * x;
        let mut out = [0.0_f64; 3];
        for (c, slot) in self.coeff.iter().zip(out.iter_mut()) {
            // Clenshaw recurrence.
            let mut b1 = 0.0_f64;
            let mut b2 = 0.0_f64;
            for &a in c.iter().skip(1).rev() {
                let b0 = a + two_x * b1 - b2;
                b2 = b1;
                b1 = b0;
            }
            *slot = c[0] + x * b1 - b2;
        }
        (out[0], out[1], out[2])
    }

    /// `(φ, q, t)` at `r`: interpolated when in range, exact otherwise.
    #[inline]
    pub fn eval_or_exact(
        &self,
        kind: &RadialScalarKind,
        r: f64,
    ) -> Result<(f64, f64, f64), BasisError> {
        if self.covers(r) {
            Ok(self.eval_inside(r))
        } else {
            kind.eval_design_triplet(r)
        }
    }

    /// Exact Chebyshev spectral differentiation in the `x ∈ [−1, 1]`
    /// variable: given the coefficients `a_d` of `J(x) = Σ a_d T_d(x)`,
    /// return the coefficients `b_d` of `J′(x) = Σ b_d T_d(x)` via the
    /// backward recurrence
    ///
    /// ```text
    /// b_D = 0;  b_{D-1} = 2·D·a_D;  b_{d-1} = b_{d+1} + 2·d·a_d  (d = D−1 … 1);
    /// then b_0 ← b_0 / 2.
    /// ```
    ///
    /// No new function evaluations — `J′` shares the value channel's
    /// representation, so value and derivative are immune to the
    /// objective↔gradient desync class.
    pub(crate) fn differentiate_coeffs(a: &[f64]) -> Vec<f64> {
        let len = a.len();
        let mut b = vec![0.0_f64; len];
        if len <= 1 {
            return b;
        }
        let big_d = len - 1;
        // b_{D-1} = 2·D·a_D (with b_D = 0 already).
        b[big_d - 1] = 2.0 * big_d as f64 * a[big_d];
        // b_{d-1} = b_{d+1} + 2·d·a_d for d = D−1 … 1.
        for d in (1..big_d).rev() {
            b[d - 1] = b[d + 1] + 2.0 * d as f64 * a[d];
        }
        // Halve the constant term per the first-kind convention.
        b[0] *= 0.5;
        b
    }

    /// Clenshaw contraction of one coefficient stack at the mapped point `x`.
    #[inline]
    pub(crate) fn clenshaw(c: &[f64], x: f64, two_x: f64) -> f64 {
        let mut b1 = 0.0_f64;
        let mut b2 = 0.0_f64;
        for &a in c.iter().skip(1).rev() {
            let b0 = a + two_x * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        c.first().copied().unwrap_or(0.0) + x * b1 - b2
    }

    /// Per-channel jet tower at an in-range radius: for each channel `c ∈
    /// {φ, q, t}` returns `[c(r), dc/dr, d²c/dr², d³c/dr³]`.
    ///
    /// The derivative stacks come from iterating [`Self::differentiate_coeffs`]
    /// in the Chebyshev `x` variable; the affine map `x(u)` contributes a
    /// constant factor `dx/du = 2/(u_hi − u_lo)` per `u`-derivative, and the
    /// log map `u = ln r` contributes the `du/dr = 1/r` chain. Concretely, with
    /// `s = dx/du` and `g(u) = J(x(u))`:
    ///
    /// ```text
    /// J_u   = s · J_x
    /// J_uu  = s² · J_xx
    /// J_uuu = s³ · J_xxx
    /// dc/dr      = J_u / r
    /// d²c/dr²    = (J_uu − J_u) / r²
    /// d³c/dr³    = (J_uuu − 3·J_uu + 2·J_u) / r³
    /// ```
    ///
    /// (the `r`-power chain factors follow from `d/dr = (1/r) d/du`).
    #[inline]
    pub fn eval_jets_inside(&self, r: f64) -> [[f64; 4]; 3] {
        let u = r.ln();
        let x = (2.0 * u - (self.u_lo + self.u_hi)) / (self.u_hi - self.u_lo);
        let two_x = 2.0 * x;
        let s = 2.0 / (self.u_hi - self.u_lo); // dx/du
        let inv_r = 1.0 / r;
        let mut out = [[0.0_f64; 4]; 3];
        for (ci, c0) in self.coeff.iter().enumerate() {
            let c1 = Self::differentiate_coeffs(c0); // coeffs of J_x
            let c2 = Self::differentiate_coeffs(&c1); // coeffs of J_xx
            let c3 = Self::differentiate_coeffs(&c2); // coeffs of J_xxx
            let j_val = Self::clenshaw(c0, x, two_x);
            let j_x = Self::clenshaw(&c1, x, two_x);
            let j_xx = Self::clenshaw(&c2, x, two_x);
            let j_xxx = Self::clenshaw(&c3, x, two_x);
            // x → u → r chain.
            let j_u = s * j_x;
            let j_uu = s * s * j_xx;
            let j_uuu = s * s * s * j_xxx;
            out[ci][0] = j_val; // c(r)
            out[ci][1] = j_u * inv_r; // dc/dr
            out[ci][2] = (j_uu - j_u) * inv_r * inv_r; // d²c/dr²
            out[ci][3] = (j_uuu - 3.0 * j_uu + 2.0 * j_u) * inv_r * inv_r * inv_r; // d³c/dr³
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::duchon_partial_fraction_coeffs;
    use super::*;

    pub(crate) fn production_duchon_kind() -> RadialScalarKind {
        // The large-scale conditional-PGS configuration:
        // duchon(16 PCs, order=0 → p=1, power=9 → s=9, length_scale=1).
        let (p_order, s_order, dim, length_scale) = (1usize, 9usize, 16usize, 1.0_f64);
        RadialScalarKind::Duchon {
            length_scale,
            p_order,
            s_order,
            dim,
            coeffs: duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale),
        }
    }

    #[test]
    pub(crate) fn duchon_profile_certifies_and_matches_exact_on_dense_grid() {
        let kind = production_duchon_kind();
        let (r_min, r_max) = (0.05_f64, 30.0_f64);
        let profile =
            RadialProfile::build(&kind, r_min, r_max).expect("production Duchon profile certifies");
        let n = 2_000usize;
        for i in 0..n {
            let r = r_min * (r_max / r_min).powf((i as f64 + 0.5) / n as f64);
            let (phi_e, q_e, t_e) = kind.eval_design_triplet(r).expect("exact triplet");
            let (phi_i, q_i, t_i) = profile.eval_or_exact(&kind, r).expect("profile eval");
            for (interp, exact) in [(phi_i, phi_e), (q_i, q_e), (t_i, t_e)] {
                let scale = exact.abs().max(interp.abs()).max(f64::MIN_POSITIVE);
                assert!(
                    (interp - exact).abs() <= 1.0e-11 * scale,
                    "profile vs exact at r={r}: {interp:e} vs {exact:e}"
                );
            }
        }
    }

    #[test]
    pub(crate) fn out_of_range_radii_fall_back_to_exact() {
        let kind = production_duchon_kind();
        let profile = RadialProfile::build(&kind, 0.1, 10.0).expect("profile certifies");
        for &r in &[0.01_f64, 50.0] {
            assert!(!profile.covers(r));
            let exact = kind.eval_design_triplet(r).expect("exact");
            let via = profile.eval_or_exact(&kind, r).expect("fallback");
            assert_eq!(exact, via, "fallback must be the exact evaluator verbatim");
        }
    }

    #[test]
    pub(crate) fn jet_tower_matches_central_differences_of_value_clenshaw() {
        // The differentiated-coefficient jet must match central differences
        // of the value channel's own Clenshaw across the radius range — i.e.
        // value and derivative are ONE source of truth (no transcendental
        // re-evaluation, immune to the desync class).
        let kind = production_duchon_kind();
        let (r_min, r_max) = (0.05_f64, 30.0_f64);
        let profile =
            RadialProfile::build(&kind, r_min, r_max).expect("production Duchon profile certifies");
        let n = 200usize;
        // Stay strictly interior so central-difference stencils remain in range.
        let (lo, hi) = (r_min * 1.2, r_max / 1.2);
        for i in 0..n {
            let r = lo * (hi / lo).powf((i as f64 + 0.5) / n as f64);
            // Relative step in r; profile is smooth in u = ln r.
            let h = r * 1.0e-4;
            let jets = profile.eval_jets_inside(r);
            for (ci, jet) in jets.iter().enumerate() {
                let v = |rr: f64| profile.eval_inside(rr);
                let val = |rr: f64| {
                    let (p, q, t) = v(rr);
                    [p, q, t][ci]
                };
                let f_m2 = val(r - 2.0 * h);
                let f_m1 = val(r - h);
                let f_p1 = val(r + h);
                let f_p2 = val(r + 2.0 * h);
                let f_0 = val(r);
                // 4th-order central first/second/third derivatives.
                let d1 = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * h);
                let d2 = (-f_p2 + 16.0 * f_p1 - 30.0 * f_0 + 16.0 * f_m1 - f_m2) / (12.0 * h * h);
                let d3 = (f_p2 - 2.0 * f_p1 + 2.0 * f_m1 - f_m2) / (2.0 * h * h * h);
                let checks = [(jet[0], f_0), (jet[1], d1), (jet[2], d2), (jet[3], d3)];
                // Looser per-order tolerance: higher FD orders amplify the
                // truncation/rounding floor of the finite step.
                let rtols = [1.0e-10_f64, 1.0e-5, 1.0e-3, 5.0e-2];
                for (k, (jet_v, fd_v)) in checks.iter().enumerate() {
                    let scale = jet_v.abs().max(fd_v.abs()).max(1.0e-8);
                    assert!(
                        (jet_v - fd_v).abs() <= rtols[k] * scale,
                        "channel {ci} order {k} at r={r}: jet={jet_v:e} fd={fd_v:e}"
                    );
                }
            }
        }
    }

    #[test]
    pub(crate) fn degenerate_range_refuses() {
        let kind = production_duchon_kind();
        assert!(RadialProfile::build(&kind, 1.0, 1.0).is_none());
        assert!(RadialProfile::build(&kind, -1.0, 2.0).is_none());
    }
}
